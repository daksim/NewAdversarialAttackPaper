# Latest Adversarial Attack Papers
**update at 2023-05-23 19:29:05**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. And/or trade-off in artificial neurons: impact on adversarial robustness**

和/或人工神经元的权衡：对抗性稳健性的影响 cs.LG

**SubmitDate**: 2023-05-22    [abs](http://arxiv.org/abs/2102.07389v3) [paper-pdf](http://arxiv.org/pdf/2102.07389v3)

**Authors**: Alessandro Fontana

**Abstract**: Despite the success of neural networks, the issue of classification robustness remains, particularly highlighted by adversarial examples. In this paper, we address this challenge by focusing on the continuum of functions implemented in artificial neurons, ranging from pure AND gates to pure OR gates. Our hypothesis is that the presence of a sufficient number of OR-like neurons in a network can lead to classification brittleness and increased vulnerability to adversarial attacks. We define AND-like neurons and propose measures to increase their proportion in the network. These measures involve rescaling inputs to the [-1,1] interval and reducing the number of points in the steepest section of the sigmoidal activation function. A crucial component of our method is the comparison between a neuron's output distribution when fed with the actual dataset and a randomised version called the "scrambled dataset." Experimental results on the MNIST dataset suggest that our approach holds promise as a direction for further exploration.

摘要: 尽管神经网络取得了成功，但分类稳健性的问题仍然存在，尤其是对抗性的例子。在本文中，我们通过关注在人工神经元中实现的功能的连续体来解决这一挑战，从纯AND门到纯OR门。我们的假设是，网络中存在足够数量的类OR神经元会导致分类脆性，并增加对对手攻击的脆弱性。我们定义了AND-Like神经元，并提出了提高其在网络中所占比例的措施。这些措施包括将输入重新调整到[-1，1]区间，并减少S型激活函数最陡峭部分中的点数。我们方法的一个关键组成部分是比较神经元在输入实际数据集时的输出分布与被称为“扰乱数据集”的随机版本之间的比较。在MNIST数据集上的实验结果表明，我们的方法有望成为进一步探索的方向。



## **2. Analyzing the Shuffle Model through the Lens of Quantitative Information Flow**

从定量信息流的视角分析洗牌模型 cs.CR

**SubmitDate**: 2023-05-22    [abs](http://arxiv.org/abs/2305.13075v1) [paper-pdf](http://arxiv.org/pdf/2305.13075v1)

**Authors**: Mireya Jurado, Ramon G. Gonze, Mário S. Alvim, Catuscia Palamidessi

**Abstract**: Local differential privacy (LDP) is a variant of differential privacy (DP) that avoids the need for a trusted central curator, at the cost of a worse trade-off between privacy and utility. The shuffle model is a way to provide greater anonymity to users by randomly permuting their messages, so that the link between users and their reported values is lost to the data collector. By combining an LDP mechanism with a shuffler, privacy can be improved at no cost for the accuracy of operations insensitive to permutations, thereby improving utility in many tasks. However, the privacy implications of shuffling are not always immediately evident, and derivations of privacy bounds are made on a case-by-case basis.   In this paper, we analyze the combination of LDP with shuffling in the rigorous framework of quantitative information flow (QIF), and reason about the resulting resilience to inference attacks. QIF naturally captures randomization mechanisms as information-theoretic channels, thus allowing for precise modeling of a variety of inference attacks in a natural way and for measuring the leakage of private information under these attacks. We exploit symmetries of the particular combination of k-RR mechanisms with the shuffle model to achieve closed formulas that express leakage exactly. In particular, we provide formulas that show how shuffling improves protection against leaks in the local model, and study how leakage behaves for various values of the privacy parameter of the LDP mechanism.   In contrast to the strong adversary from differential privacy, we focus on an uninformed adversary, who does not know the value of any individual in the dataset. This adversary is often more realistic as a consumer of statistical datasets, and we show that in some situations mechanisms that are equivalent w.r.t. the strong adversary can provide different privacy guarantees under the uninformed one.

摘要: 本地差异隐私(LDP)是差异隐私(DP)的变体，它避免了对受信任的中央管理员的需要，但代价是在隐私和效用之间进行了更糟糕的权衡。混洗模型是一种通过随机排列用户的消息为用户提供更大匿名性的方法，从而使数据收集器失去用户与其报告的值之间的联系。通过将LDP机制与洗牌器相结合，可以在不花费任何代价的情况下提高私密性，因为操作对排列不敏感，从而提高了在许多任务中的实用性。然而，洗牌对隐私的影响并不总是立竿见影的，隐私界限的推导是基于个案的。在严格的量化信息流(QIF)框架下，我们分析了LDP和置乱的结合，并对其对推理攻击的适应性进行了分析。QIF自然将随机化机制捕获为信息论通道，从而允许以自然的方式对各种推理攻击进行精确建模，并测量这些攻击下私人信息的泄漏。我们利用k-RR机制与Shuffle模型的特殊组合的对称性来获得准确表达泄漏的封闭公式。特别地，我们提供了公式，展示了置乱如何改善对本地模型中的泄漏的保护，并研究了LDP机制的隐私参数的不同值下的泄漏行为。与来自差异隐私的强大对手不同，我们关注的是不知情的对手，他不知道数据集中任何个人的价值。作为统计数据集的消费者，这个对手通常更现实，我们证明了在某些情况下，机制是等价的w.r.t.强大的对手可以在不知情的情况下提供不同的隐私保障。



## **3. Latent Magic: An Investigation into Adversarial Examples Crafted in the Semantic Latent Space**

潜魔力：语义潜在空间中的对抗性例证研究 cs.LG

**SubmitDate**: 2023-05-22    [abs](http://arxiv.org/abs/2305.12906v1) [paper-pdf](http://arxiv.org/pdf/2305.12906v1)

**Authors**: BoYang Zheng

**Abstract**: Adversarial attacks against Deep Neural Networks(DNN) have been a crutial topic ever since \cite{goodfellow} purposed the vulnerability of DNNs. However, most prior works craft adversarial examples in the pixel space, following the $l_p$ norm constraint. In this paper, we give intuitional explain about why crafting adversarial examples in the latent space is equally efficient and important. We purpose a framework for crafting adversarial examples in semantic latent space based on an pre-trained Variational Auto Encoder from state-of-art Stable Diffusion Model\cite{SDM}. We also show that adversarial examples crafted in the latent space can also achieve a high level of fool rate. However, examples crafted from latent space are often hard to evaluated, as they doesn't follow a certain $l_p$ norm constraint, which is a big challenge for existing researches. To efficiently and accurately evaluate the adversarial examples crafted in the latent space, we purpose \textbf{a novel evaluation matric} based on SSIM\cite{SSIM} loss and fool rate.Additionally, we explain why FID\cite{FID} is not suitable for measuring such adversarial examples. To the best of our knowledge, it's the first evaluation metrics that is specifically designed to evaluate the quality of a adversarial attack. We also investigate the transferability of adversarial examples crafted in the latent space and show that they have superiority over adversarial examples crafted in the pixel space.

摘要: 对深度神经网络(DNN)的敌意攻击，自从古德费罗针对DNN的脆弱性而提出以来，一直是一个重要的话题。然而，大多数以前的工作都是在像素空间中制作对抗性例子，遵循$L_p$范数约束。在这篇文章中，我们直观地解释了为什么在潜在空间中构造对抗性例子是同样有效和重要的。我们的目的是在语义潜在空间中构建一个框架，该框架基于最新的稳定扩散模型{SDM}的预先训练的变分自动编码器。我们还表明，在潜在空间中精心制作的对抗性例子也可以达到高水平的傻瓜率。然而，从潜在空间构造的例子往往很难评估，因为它们不遵循某个$L_p$范数约束，这对现有的研究是一个很大的挑战。为了高效、准确地评价潜在空间中产生的敌意实例，提出了一种新的基于ssim{ssim}损失率和傻瓜率的评价矩阵，并解释了为什么FID\cite{FID}不适合度量此类敌意实例.据我们所知，这是第一个专门为评估敌方攻击质量而设计的评估指标。我们还考察了在潜在空间中生成的对抗性实例的可转移性，并证明了它们比在像素空间中生成的对抗性实例具有更好的可转移性。



## **4. Byzantine Robust Cooperative Multi-Agent Reinforcement Learning as a Bayesian Game**

拜占庭稳健合作多智能体强化学习的贝叶斯博弈 cs.GT

**SubmitDate**: 2023-05-22    [abs](http://arxiv.org/abs/2305.12872v1) [paper-pdf](http://arxiv.org/pdf/2305.12872v1)

**Authors**: Simin Li, Jun Guo, Jingqiao Xiu, Xini Yu, Jiakai Wang, Aishan Liu, Yaodong Yang, Xianglong Liu

**Abstract**: In this study, we explore the robustness of cooperative multi-agent reinforcement learning (c-MARL) against Byzantine failures, where any agent can enact arbitrary, worst-case actions due to malfunction or adversarial attack. To address the uncertainty that any agent can be adversarial, we propose a Bayesian Adversarial Robust Dec-POMDP (BARDec-POMDP) framework, which views Byzantine adversaries as nature-dictated types, represented by a separate transition. This allows agents to learn policies grounded on their posterior beliefs about the type of other agents, fostering collaboration with identified allies and minimizing vulnerability to adversarial manipulation. We define the optimal solution to the BARDec-POMDP as an ex post robust Bayesian Markov perfect equilibrium, which we proof to exist and weakly dominates the equilibrium of previous robust MARL approaches. To realize this equilibrium, we put forward a two-timescale actor-critic algorithm with almost sure convergence under specific conditions. Experimentation on matrix games, level-based foraging and StarCraft II indicate that, even under worst-case perturbations, our method successfully acquires intricate micromanagement skills and adaptively aligns with allies, demonstrating resilience against non-oblivious adversaries, random allies, observation-based attacks, and transfer-based attacks.

摘要: 在这项研究中，我们探讨了协作多智能体强化学习(c-Marl)对拜占庭故障的稳健性，在拜占庭故障中，任何智能体都可以由于故障或对手攻击而执行任意的、最坏的操作。为了解决任何智能体都可能是对抗性的不确定性，我们提出了一种贝叶斯对抗性鲁棒DEC-POMDP(BARDEC-POMDP)框架，该框架将拜占庭对手视为自然决定的类型，由单独的转换表示。这使代理能够基于他们对其他代理类型的后验信念来学习策略，促进与确定的盟友的合作，并将受到对手操纵的脆弱性降至最低。我们将BARDEC-POMDP的最优解定义为一个事后稳健的贝叶斯马尔可夫完全均衡，并证明了它的存在，并且弱控制了以前的稳健Marl方法的均衡。为了实现这一均衡，我们提出了一个在特定条件下几乎必然收敛的双时间尺度的行动者-批评者算法。在矩阵游戏、基于关卡的觅食和星际争霸II上的实验表明，即使在最坏的情况下，我们的方法也成功地获得了复杂的微观管理技能，并自适应地与盟友结盟，展示了对非遗忘对手、随机盟友、基于观察的攻击和基于转移的攻击的弹性。



## **5. Towards Benchmarking and Assessing Visual Naturalness of Physical World Adversarial Attacks**

对物理世界对抗性攻击的视觉自然性进行基准和评估 cs.CV

**SubmitDate**: 2023-05-22    [abs](http://arxiv.org/abs/2305.12863v1) [paper-pdf](http://arxiv.org/pdf/2305.12863v1)

**Authors**: Simin Li, Shuing Zhang, Gujun Chen, Dong Wang, Pu Feng, Jiakai Wang, Aishan Liu, Xin Yi, Xianglong Liu

**Abstract**: Physical world adversarial attack is a highly practical and threatening attack, which fools real world deep learning systems by generating conspicuous and maliciously crafted real world artifacts. In physical world attacks, evaluating naturalness is highly emphasized since human can easily detect and remove unnatural attacks. However, current studies evaluate naturalness in a case-by-case fashion, which suffers from errors, bias and inconsistencies. In this paper, we take the first step to benchmark and assess visual naturalness of physical world attacks, taking autonomous driving scenario as the first attempt. First, to benchmark attack naturalness, we contribute the first Physical Attack Naturalness (PAN) dataset with human rating and gaze. PAN verifies several insights for the first time: naturalness is (disparately) affected by contextual features (i.e., environmental and semantic variations) and correlates with behavioral feature (i.e., gaze signal). Second, to automatically assess attack naturalness that aligns with human ratings, we further introduce Dual Prior Alignment (DPA) network, which aims to embed human knowledge into model reasoning process. Specifically, DPA imitates human reasoning in naturalness assessment by rating prior alignment and mimics human gaze behavior by attentive prior alignment. We hope our work fosters researches to improve and automatically assess naturalness of physical world attacks. Our code and dataset can be found at https://github.com/zhangsn-19/PAN.

摘要: 物理世界敌意攻击是一种具有高度实用性和威胁性的攻击，它通过生成明显的、恶意制作的真实世界文物来愚弄现实世界的深度学习系统。在物理世界的攻击中，由于人类可以很容易地检测和删除非自然攻击，因此非常重视对自然度的评估。然而，目前的研究是以个案的方式来评估自然度，这存在错误、偏见和不一致之处。在本文中，我们以自动驾驶场景为第一次尝试，对物理世界攻击的视觉自然度进行了基准测试和评估。首先，为了基准攻击自然度，我们贡献了第一个带有人类评级和凝视的物理攻击自然度(PAN)数据集。潘首次验证了几个观点：自然性受到语境特征(即环境和语义变化)的(不同)影响，并与行为特征(即凝视信号)相关。其次，为了自动评估与人类评分一致的攻击自然度，我们进一步引入了双重先验对齐(DPA)网络，旨在将人类知识嵌入到模型推理过程中。具体地说，DPA通过对先验排列进行评级来模仿人类在自然度评估中的推理，并通过注意的先验排列来模仿人类的凝视行为。我们希望我们的工作能促进改进和自动评估物理世界攻击的自然性的研究。我们的代码和数据集可以在https://github.com/zhangsn-19/PAN.上找到



## **6. Flying Adversarial Patches: Manipulating the Behavior of Deep Learning-based Autonomous Multirotors**

飞行对抗性补丁：操纵基于深度学习的自主多旋翼的行为 cs.RO

6 pages, 5 figures, Workshop on Multi-Robot Learning, International  Conference on Robotics and Automation (ICRA)

**SubmitDate**: 2023-05-22    [abs](http://arxiv.org/abs/2305.12859v1) [paper-pdf](http://arxiv.org/pdf/2305.12859v1)

**Authors**: Pia Hanfeld, Marina M. -C. Höhne, Michael Bussmann, Wolfgang Hönig

**Abstract**: Autonomous flying robots, e.g. multirotors, often rely on a neural network that makes predictions based on a camera image. These deep learning (DL) models can compute surprising results if applied to input images outside the training domain. Adversarial attacks exploit this fault, for example, by computing small images, so-called adversarial patches, that can be placed in the environment to manipulate the neural network's prediction. We introduce flying adversarial patches, where an image is mounted on another flying robot and therefore can be placed anywhere in the field of view of a victim multirotor. For an effective attack, we compare three methods that simultaneously optimize the adversarial patch and its position in the input image. We perform an empirical validation on a publicly available DL model and dataset for autonomous multirotors. Ultimately, our attacking multirotor would be able to gain full control over the motions of the victim multirotor.

摘要: 自主飞行机器人，如多旋翼，通常依赖于神经网络，根据相机图像进行预测。如果将这些深度学习(DL)模型应用于训练域之外的输入图像，则可以计算出令人惊讶的结果。对抗性攻击利用了这一缺陷，例如，通过计算小图像，即所谓的对抗性补丁，可以放置在环境中来操纵神经网络的预测。我们引入了飞行对抗性补丁，将图像安装在另一个飞行机器人上，因此可以放置在受害者多旋翼的视野中的任何地方。对于有效的攻击，我们比较了三种同时优化对抗性补丁及其在输入图像中的位置的方法。我们在一个公开可用的自主多旋翼的DL模型和数据集上进行了经验验证。最终，我们的攻击型多旋翼将能够完全控制受害者多旋翼的运动。



## **7. Uncertainty-based Detection of Adversarial Attacks in Semantic Segmentation**

语义分割中基于不确定性的对抗性攻击检测 cs.CV

**SubmitDate**: 2023-05-22    [abs](http://arxiv.org/abs/2305.12825v1) [paper-pdf](http://arxiv.org/pdf/2305.12825v1)

**Authors**: Kira Maag, Asja Fischer

**Abstract**: State-of-the-art deep neural networks have proven to be highly powerful in a broad range of tasks, including semantic image segmentation. However, these networks are vulnerable against adversarial attacks, i.e., non-perceptible perturbations added to the input image causing incorrect predictions, which is hazardous in safety-critical applications like automated driving. Adversarial examples and defense strategies are well studied for the image classification task, while there has been limited research in the context of semantic segmentation. First works however show that the segmentation outcome can be severely distorted by adversarial attacks. In this work, we introduce an uncertainty-based method for the detection of adversarial attacks in semantic segmentation. We observe that uncertainty as for example captured by the entropy of the output distribution behaves differently on clean and perturbed images using this property to distinguish between the two cases. Our method works in a light-weight and post-processing manner, i.e., we do not modify the model or need knowledge of the process used for generating adversarial examples. In a thorough empirical analysis, we demonstrate the ability of our approach to detect perturbed images across multiple types of adversarial attacks.

摘要: 最先进的深度神经网络已被证明在包括语义图像分割在内的广泛任务中具有非常强大的功能。然而，这些网络容易受到对抗性攻击，即在输入图像中添加不可感知的扰动导致错误预测，这在自动驾驶等安全关键型应用中是危险的。对抗性例子和防御策略在图像分类任务中得到了很好的研究，而在语义分割方面的研究却很有限。然而，最初的工作表明，分割结果可能会被对抗性攻击严重扭曲。在这项工作中，我们介绍了一种基于不确定性的语义分割中的对抗性攻击检测方法。我们观察到，例如，由输出分布的熵捕获的不确定性在干净和扰动的图像上表现不同，使用该属性来区分这两种情况。我们的方法是以轻量级和后处理的方式工作的，也就是说，我们不修改模型，也不需要知道用于生成对抗性例子的过程。在深入的经验分析中，我们展示了我们的方法在多种类型的对抗性攻击中检测扰动图像的能力。



## **8. The defender's perspective on automatic speaker verification: An overview**

辩护人对自动说话人确认的观点：综述 cs.SD

Submitted to IJCAI 2023 Workshop

**SubmitDate**: 2023-05-22    [abs](http://arxiv.org/abs/2305.12804v1) [paper-pdf](http://arxiv.org/pdf/2305.12804v1)

**Authors**: Haibin Wu, Jiawen Kang, Lingwei Meng, Helen Meng, Hung-yi Lee

**Abstract**: Automatic speaker verification (ASV) plays a critical role in security-sensitive environments. Regrettably, the reliability of ASV has been undermined by the emergence of spoofing attacks, such as replay and synthetic speech, as well as adversarial attacks and the relatively new partially fake speech. While there are several review papers that cover replay and synthetic speech, and adversarial attacks, there is a notable gap in a comprehensive review that addresses defense against adversarial attacks and the recently emerged partially fake speech. Thus, the aim of this paper is to provide a thorough and systematic overview of the defense methods used against these types of attacks.

摘要: 自动说话人验证(ASV)在安全敏感环境中起着至关重要的作用。令人遗憾的是，诸如重放和合成语音等欺骗性攻击的出现，以及对抗性攻击和相对较新的部分虚假语音的出现，破坏了ASV的可靠性。虽然有几篇评论论文涵盖了重播和合成演讲，以及对抗性攻击，但在针对对抗性攻击和最近出现的部分虚假演讲的防御的全面评论中，有一个明显的空白。因此，本文的目的是对针对这些类型的攻击所使用的防御方法进行全面和系统的概述。



## **9. FGAM:Fast Adversarial Malware Generation Method Based on Gradient Sign**

FGAM：基于梯度符号的恶意软件快速生成方法 cs.CR

**SubmitDate**: 2023-05-22    [abs](http://arxiv.org/abs/2305.12770v1) [paper-pdf](http://arxiv.org/pdf/2305.12770v1)

**Authors**: Kun Li, Fan Zhang, Wei Guo

**Abstract**: Malware detection models based on deep learning have been widely used, but recent research shows that deep learning models are vulnerable to adversarial attacks. Adversarial attacks are to deceive the deep learning model by generating adversarial samples. When adversarial attacks are performed on the malware detection model, the attacker will generate adversarial malware with the same malicious functions as the malware, and make the detection model classify it as benign software. Studying adversarial malware generation can help model designers improve the robustness of malware detection models. At present, in the work on adversarial malware generation for byte-to-image malware detection models, there are mainly problems such as large amount of injection perturbation and low generation efficiency. Therefore, this paper proposes FGAM (Fast Generate Adversarial Malware), a method for fast generating adversarial malware, which iterates perturbed bytes according to the gradient sign to enhance adversarial capability of the perturbed bytes until the adversarial malware is successfully generated. It is experimentally verified that the success rate of the adversarial malware deception model generated by FGAM is increased by about 84\% compared with existing methods.

摘要: 基于深度学习的恶意软件检测模型已经得到了广泛的应用，但最近的研究表明，深度学习模型容易受到对手的攻击。对抗性攻击是通过生成对抗性样本来欺骗深度学习模型。当对恶意软件检测模型进行对抗性攻击时，攻击者会生成与恶意软件具有相同恶意功能的对抗性恶意软件，并使检测模型将其归类为良性软件。研究恶意软件的生成可以帮助模型设计者提高恶意软件检测模型的健壮性。目前，在字节到图像恶意软件检测模型的恶意软件生成工作中，主要存在注入扰动量大、生成效率低等问题。为此，本文提出了一种快速生成恶意代码的方法--快速生成恶意代码FGAM，该方法根据梯度符号迭代扰动字节，以增强扰动字节的对抗能力，直到成功生成恶意代码。实验证明，FGAM生成的敌意恶意软件欺骗模型的成功率比现有方法提高了约84%。



## **10. On The Empirical Effectiveness of Unrealistic Adversarial Hardening Against Realistic Adversarial Attacks**

论非现实对抗强化对抗现实对抗攻击的经验有效性 cs.LG

S&P 2023

**SubmitDate**: 2023-05-22    [abs](http://arxiv.org/abs/2202.03277v2) [paper-pdf](http://arxiv.org/pdf/2202.03277v2)

**Authors**: Salijona Dyrmishi, Salah Ghamizi, Thibault Simonetto, Yves Le Traon, Maxime Cordy

**Abstract**: While the literature on security attacks and defense of Machine Learning (ML) systems mostly focuses on unrealistic adversarial examples, recent research has raised concern about the under-explored field of realistic adversarial attacks and their implications on the robustness of real-world systems. Our paper paves the way for a better understanding of adversarial robustness against realistic attacks and makes two major contributions. First, we conduct a study on three real-world use cases (text classification, botnet detection, malware detection)) and five datasets in order to evaluate whether unrealistic adversarial examples can be used to protect models against realistic examples. Our results reveal discrepancies across the use cases, where unrealistic examples can either be as effective as the realistic ones or may offer only limited improvement. Second, to explain these results, we analyze the latent representation of the adversarial examples generated with realistic and unrealistic attacks. We shed light on the patterns that discriminate which unrealistic examples can be used for effective hardening. We release our code, datasets and models to support future research in exploring how to reduce the gap between unrealistic and realistic adversarial attacks.

摘要: 虽然关于机器学习系统的安全攻击和防御的文献大多集中在不现实的对抗性例子上，但最近的研究已经引起了对现实对抗性攻击及其对真实世界系统的健壮性的影响的关注。我们的论文为更好地理解对手对现实攻击的稳健性铺平了道路，并做出了两个主要贡献。首先，我们在三个真实世界的用例(文本分类、僵尸网络检测、恶意软件检测)和五个数据集上进行了研究，以评估不现实的对抗性例子是否可以用来保护模型免受现实例子的影响。我们的结果揭示了用例之间的差异，在这些用例中，不切实际的例子可能与现实的例子一样有效，或者可能只提供有限的改进。其次，为了解释这些结果，我们分析了现实攻击和非现实攻击产生的对抗性例子的潜在表征。我们阐明了区分哪些不切实际的例子可以用于有效强化的模式。我们发布了我们的代码、数据集和模型，以支持未来的研究，探索如何缩小不现实和现实的对抗性攻击之间的差距。



## **11. RAIN: RegulArization on Input and Network for Black-Box Domain Adaptation**

RAIN：黑箱域自适应的输入和网络规则化 cs.CV

Accepted by IJCAI 2023

**SubmitDate**: 2023-05-21    [abs](http://arxiv.org/abs/2208.10531v3) [paper-pdf](http://arxiv.org/pdf/2208.10531v3)

**Authors**: Qucheng Peng, Zhengming Ding, Lingjuan Lyu, Lichao Sun, Chen Chen

**Abstract**: Source-Free domain adaptation transits the source-trained model towards target domain without exposing the source data, trying to dispel these concerns about data privacy and security. However, this paradigm is still at risk of data leakage due to adversarial attacks on the source model. Hence, the Black-Box setting only allows to use the outputs of source model, but still suffers from overfitting on the source domain more severely due to source model's unseen weights. In this paper, we propose a novel approach named RAIN (RegulArization on Input and Network) for Black-Box domain adaptation from both input-level and network-level regularization. For the input-level, we design a new data augmentation technique as Phase MixUp, which highlights task-relevant objects in the interpolations, thus enhancing input-level regularization and class consistency for target models. For network-level, we develop a Subnetwork Distillation mechanism to transfer knowledge from the target subnetwork to the full target network via knowledge distillation, which thus alleviates overfitting on the source domain by learning diverse target representations. Extensive experiments show that our method achieves state-of-the-art performance on several cross-domain benchmarks under both single- and multi-source black-box domain adaptation.

摘要: 无源域自适应在不暴露源数据的情况下将源训练模型过渡到目标域，试图消除这些对数据隐私和安全的担忧。然而，由于对源模型的对抗性攻击，该范例仍然面临数据泄露的风险。因此，黑盒设置只允许使用源模型的输出，但由于源模型的不可见权重，仍然受到源域上的过度拟合的更严重影响。本文从输入级正则化和网络级正则化两个方面提出了一种新的黑箱域自适应方法RAIN。对于输入层，我们设计了一种新的数据增强技术--阶段混合，它在内插中突出与任务相关的对象，从而增强了输入层的正则性和目标模型的类一致性。对于网络级，我们提出了一种子网络精馏机制，通过知识精馏将知识从目标子网络传递到整个目标网络，从而通过学习不同的目标表示来缓解源域的过度匹配。大量的实验表明，在单源和多源黑盒领域自适应的情况下，我们的方法在多个跨域基准测试上都达到了最好的性能。



## **12. Dynamic Transformers Provide a False Sense of Efficiency**

动态变形金刚带来了一种错误的能效感 cs.CL

Accepted by ACL2023

**SubmitDate**: 2023-05-20    [abs](http://arxiv.org/abs/2305.12228v1) [paper-pdf](http://arxiv.org/pdf/2305.12228v1)

**Authors**: Yiming Chen, Simin Chen, Zexin Li, Wei Yang, Cong Liu, Robby T. Tan, Haizhou Li

**Abstract**: Despite much success in natural language processing (NLP), pre-trained language models typically lead to a high computational cost during inference. Multi-exit is a mainstream approach to address this issue by making a trade-off between efficiency and accuracy, where the saving of computation comes from an early exit. However, whether such saving from early-exiting is robust remains unknown. Motivated by this, we first show that directly adapting existing adversarial attack approaches targeting model accuracy cannot significantly reduce inference efficiency. To this end, we propose a simple yet effective attacking framework, SAME, a novel slowdown attack framework on multi-exit models, which is specially tailored to reduce the efficiency of the multi-exit models. By leveraging the multi-exit models' design characteristics, we utilize all internal predictions to guide the adversarial sample generation instead of merely considering the final prediction. Experiments on the GLUE benchmark show that SAME can effectively diminish the efficiency gain of various multi-exit models by 80% on average, convincingly validating its effectiveness and generalization ability.

摘要: 尽管在自然语言处理(NLP)方面取得了很大的成功，但预训练的语言模型通常会导致推理过程中的计算成本很高。多重退出是解决这一问题的主流方法，它在效率和精度之间进行了权衡，其中计算的节省来自于提前退出。然而，这种从提前退出中节省下来的做法是否稳健仍是个未知数。基于此，我们首先证明了直接采用现有的针对模型准确性的对抗性攻击方法并不会显著降低推理效率。为此，我们提出了一个简单而有效的攻击框架--SAME，一种新的针对多出口模型的减速攻击框架，该框架专门为降低多出口模型的效率而量身定做。通过利用多退出模型的设计特点，我们利用所有内部预测来指导对抗性样本的生成，而不是仅仅考虑最终预测。在GLUE基准上的实验表明，SAME可以有效地降低各种多出口模型的效率增益，平均降低80%，令人信服地验证了其有效性和泛化能力。



## **13. RNNS: Representation Nearest Neighbor Search Black-Box Attack on Code Models**

RNNS：代码模型上的表示最近邻搜索黑盒攻击 cs.CR

**SubmitDate**: 2023-05-20    [abs](http://arxiv.org/abs/2305.05896v2) [paper-pdf](http://arxiv.org/pdf/2305.05896v2)

**Authors**: Jie Zhang, Wei Ma, Qiang Hu, Xiaofei Xie, Yves Le Traon, Yang Liu

**Abstract**: Pre-trained code models are mainly evaluated using the in-distribution test data. The robustness of models, i.e., the ability to handle hard unseen data, still lacks evaluation. In this paper, we propose a novel search-based black-box adversarial attack guided by model behaviours for pre-trained programming language models, named Representation Nearest Neighbor Search(RNNS), to evaluate the robustness of Pre-trained PL models. Unlike other black-box adversarial attacks, RNNS uses the model-change signal to guide the search in the space of the variable names collected from real-world projects. Specifically, RNNS contains two main steps, 1) indicate which variable (attack position location) we should attack based on model uncertainty, and 2) search which adversarial tokens we should use for variable renaming according to the model behaviour observations. We evaluate RNNS on 6 code tasks (e.g., clone detection), 3 programming languages (Java, Python, and C), and 3 pre-trained code models: CodeBERT, GraphCodeBERT, and CodeT5. The results demonstrate that RNNS outperforms the state-of-the-art black-box attacking methods (MHM and ALERT) in terms of attack success rate (ASR) and query times (QT). The perturbation of generated adversarial examples from RNNS is smaller than the baselines with respect to the number of replaced variables and the variable length change. Our experiments also show that RNNS is efficient in attacking the defended models and is useful for adversarial training.

摘要: 预先训练的代码模型主要使用分发内测试数据进行评估。模型的稳健性，即处理硬的看不见的数据的能力，仍然缺乏评估。针对预先训练的程序设计语言模型，提出了一种以模型行为为导向的基于搜索的黑盒对抗攻击方法--表示最近邻搜索算法(RNNS)，以评估预先训练的程序设计语言模型的健壮性。与其他黑盒对抗性攻击不同，RNNS使用模型更改信号来指导在从现实世界项目中收集的变量名称空间中的搜索。具体地说，RNNS包含两个主要步骤，1)根据模型的不确定性指示我们应该攻击哪个变量(攻击位置)，2)根据模型行为观察寻找应该使用哪些敌意标记进行变量重命名。我们在6个代码任务(例如克隆检测)、3种编程语言(Java、Python和C)以及3种预先训练的代码模型上对RNNS进行了评估：CodeBERT、GraphCodeBERT和CodeT5。结果表明，RNNS在攻击成功率(ASR)和查询次数(Qt)方面均优于目前最先进的黑盒攻击方法(MHM和ALERT)。从RNNS生成的对抗性样本在替换变量的数量和可变长度变化方面的扰动小于基线。我们的实验还表明，RNNS在攻击防御模型方面是有效的，并且对于对抗性训练是有用的。



## **14. Towards Adversarially Robust Recommendation from Adaptive Fraudster Detection**

基于自适应诈骗检测的逆稳性推荐 cs.IR

**SubmitDate**: 2023-05-20    [abs](http://arxiv.org/abs/2211.11534v3) [paper-pdf](http://arxiv.org/pdf/2211.11534v3)

**Authors**: Yuni Lai, Yulin Zhu, Wenqi Fan, Xiaoge Zhang, Kai Zhou

**Abstract**: The robustness of recommender systems under node injection attacks has garnered significant attention. Recently, GraphRfi, a GNN-based recommender system, was proposed and shown to effectively mitigate the impact of injected fake users. However, we demonstrate that GraphRfi remains vulnerable to attacks due to the supervised nature of its fraudster detection component, where obtaining clean labels is challenging in practice. In particular, we propose a powerful poisoning attack, MetaC, against both GNN-based and MF-based recommender systems. Furthermore, we analyze why GraphRfi fails under such an attack. Then, based on our insights obtained from vulnerability analysis, we design an adaptive fraudster detection module that explicitly considers label uncertainty. This module can serve as a plug-in for different recommender systems, resulting in a robust framework named PDR. Comprehensive experiments show that our defense approach outperforms other benchmark methods under attacks. Overall, our research presents an effective framework for integrating fraudster detection into recommendation systems to achieve adversarial robustness.

摘要: 推荐系统在节点注入攻击下的健壮性已经引起了广泛的关注。最近，GraphRfi，一个基于GNN的推荐系统，被提出并被证明有效地缓解了注入虚假用户的影响。然而，我们证明GraphRfi仍然容易受到攻击，因为其欺诈者检测组件的监督性质，在实践中获得干净的标签是具有挑战性的。特别是，我们提出了一种针对基于GNN和基于MF的推荐系统的强大的毒化攻击Metac。此外，我们还分析了GraphRfi在这样的攻击下失败的原因。然后，基于漏洞分析所获得的见解，我们设计了一个显式考虑标签不确定性的自适应诈骗检测模块。该模块可以作为不同推荐系统的插件，从而产生一个名为PDR的健壮框架。综合实验表明，我们的防御方法在攻击下的性能优于其他基准方法。总体而言，我们的研究提出了一个有效的框架，将欺诈者检测集成到推荐系统中，以实现对手健壮性。



## **15. Annealing Self-Distillation Rectification Improves Adversarial Training**

退火自蒸馏精馏改善对手训练 cs.LG

10 pages + Appendix

**SubmitDate**: 2023-05-20    [abs](http://arxiv.org/abs/2305.12118v1) [paper-pdf](http://arxiv.org/pdf/2305.12118v1)

**Authors**: Yu-Yu Wu, Hung-Jui Wang, Shang-Tse Chen

**Abstract**: In standard adversarial training, models are optimized to fit one-hot labels within allowable adversarial perturbation budgets. However, the ignorance of underlying distribution shifts brought by perturbations causes the problem of robust overfitting. To address this issue and enhance adversarial robustness, we analyze the characteristics of robust models and identify that robust models tend to produce smoother and well-calibrated outputs. Based on the observation, we propose a simple yet effective method, Annealing Self-Distillation Rectification (ADR), which generates soft labels as a better guidance mechanism that accurately reflects the distribution shift under attack during adversarial training. By utilizing ADR, we can obtain rectified distributions that significantly improve model robustness without the need for pre-trained models or extensive extra computation. Moreover, our method facilitates seamless plug-and-play integration with other adversarial training techniques by replacing the hard labels in their objectives. We demonstrate the efficacy of ADR through extensive experiments and strong performances across datasets.

摘要: 在标准的对抗性训练中，模型经过优化，以适应允许的对抗性扰动预算内的单一热门标签。然而，对扰动带来的潜在分布漂移的忽视导致了稳健过拟合的问题。为了解决这一问题并增强对手的稳健性，我们分析了稳健模型的特征，并发现稳健模型往往会产生更平滑和校准良好的输出。在此基础上，我们提出了一种简单而有效的方法--退火法自蒸馏纠错(ADR)，该方法生成软标签作为一种更好的指导机制，准确地反映了对抗训练中攻击下的分布变化。通过利用ADR，我们可以获得显著提高模型稳健性的校正分布，而不需要预先训练的模型或大量的额外计算。此外，我们的方法通过替换目标中的硬标签，促进了与其他对抗性训练技术的无缝即插即用集成。我们通过广泛的实验和在数据集上的强劲表现证明了ADR的有效性。



## **16. SneakyPrompt: Evaluating Robustness of Text-to-image Generative Models' Safety Filters**

SneakyPrompt：评估文本到图像生成模型的安全过滤器的健壮性 cs.LG

**SubmitDate**: 2023-05-20    [abs](http://arxiv.org/abs/2305.12082v1) [paper-pdf](http://arxiv.org/pdf/2305.12082v1)

**Authors**: Yuchen Yang, Bo Hui, Haolin Yuan, Neil Gong, Yinzhi Cao

**Abstract**: Text-to-image generative models such as Stable Diffusion and DALL$\cdot$E 2 have attracted much attention since their publication due to their wide application in the real world. One challenging problem of text-to-image generative models is the generation of Not-Safe-for-Work (NSFW) content, e.g., those related to violence and adult. Therefore, a common practice is to deploy a so-called safety filter, which blocks NSFW content based on either text or image features. Prior works have studied the possible bypass of such safety filters. However, existing works are largely manual and specific to Stable Diffusion's official safety filter. Moreover, the bypass ratio of Stable Diffusion's safety filter is as low as 23.51% based on our evaluation.   In this paper, we propose the first automated attack framework, called SneakyPrompt, to evaluate the robustness of real-world safety filters in state-of-the-art text-to-image generative models. Our key insight is to search for alternative tokens in a prompt that generates NSFW images so that the generated prompt (called an adversarial prompt) bypasses existing safety filters. Specifically, SneakyPrompt utilizes reinforcement learning (RL) to guide an agent with positive rewards on semantic similarity and bypass success.   Our evaluation shows that SneakyPrompt successfully generated NSFW content using an online model DALL$\cdot$E 2 with its default, closed-box safety filter enabled. At the same time, we also deploy several open-source state-of-the-art safety filters on a Stable Diffusion model and show that SneakyPrompt not only successfully generates NSFW content, but also outperforms existing adversarial attacks in terms of the number of queries and image qualities.

摘要: 从文本到图像的生成模型，如稳定扩散模型和Dall$\CDOT$E2模型自问世以来，由于其在现实世界中的广泛应用而引起了人们的广泛关注。文本到图像生成模型的一个具有挑战性的问题是生成非安全工作(NSFW)内容，例如与暴力和成人有关的内容。因此，一种常见的做法是部署所谓的安全过滤器，即根据文本或图像特征阻止NSFW内容。以前的工作已经研究了这种安全过滤器的可能旁路。然而，现有的工作主要是手动的，专门针对稳定扩散的官方安全过滤器。此外，根据我们的评估，稳定扩散安全过滤器的旁路比低至23.51%。在本文中，我们提出了第一个自动攻击框架，称为SneakyPrompt，用于评估最新的文本到图像生成模型中现实世界安全过滤器的稳健性。我们的主要见解是在生成NSFW图像的提示中搜索替代令牌，以便生成的提示(称为对抗性提示)绕过现有的安全过滤器。具体地说，SneakyPrompt利用强化学习(RL)来指导代理在语义相似性方面获得积极回报，并绕过成功。我们的评估表明，SneakyPrompt成功地使用在线模型DALL$\CDOT$E 2生成了NSFW内容，并启用了默认的闭箱安全过滤器。同时，我们还在一个稳定的扩散模型上部署了几个开源的最先进的安全过滤器，并表明SneakyPrompt不仅成功地生成了NSFW内容，而且在查询数量和图像质量方面都优于现有的对抗性攻击。



## **17. STDLens: Model Hijacking-Resilient Federated Learning for Object Detection**

STDLens：用于目标检测的模型劫持-弹性联合学习 cs.CR

CVPR 2023. Source Code: https://github.com/git-disl/STDLens

**SubmitDate**: 2023-05-20    [abs](http://arxiv.org/abs/2303.11511v3) [paper-pdf](http://arxiv.org/pdf/2303.11511v3)

**Authors**: Ka-Ho Chow, Ling Liu, Wenqi Wei, Fatih Ilhan, Yanzhao Wu

**Abstract**: Federated Learning (FL) has been gaining popularity as a collaborative learning framework to train deep learning-based object detection models over a distributed population of clients. Despite its advantages, FL is vulnerable to model hijacking. The attacker can control how the object detection system should misbehave by implanting Trojaned gradients using only a small number of compromised clients in the collaborative learning process. This paper introduces STDLens, a principled approach to safeguarding FL against such attacks. We first investigate existing mitigation mechanisms and analyze their failures caused by the inherent errors in spatial clustering analysis on gradients. Based on the insights, we introduce a three-tier forensic framework to identify and expel Trojaned gradients and reclaim the performance over the course of FL. We consider three types of adaptive attacks and demonstrate the robustness of STDLens against advanced adversaries. Extensive experiments show that STDLens can protect FL against different model hijacking attacks and outperform existing methods in identifying and removing Trojaned gradients with significantly higher precision and much lower false-positive rates.

摘要: 联邦学习(FL)作为一种协作学习框架，在分布的客户群上训练基于深度学习的目标检测模型，已经越来越受欢迎。尽管有优势，但FL很容易受到模特劫持的攻击。攻击者可以通过在协作学习过程中仅使用少量受攻击的客户端植入特洛伊木马梯度来控制对象检测系统的不当行为。本文介绍了STDLens，一种保护FL免受此类攻击的原则性方法。我们首先调查了现有的缓解机制，并分析了它们由于梯度空间聚类分析的固有错误而导致的失败。基于这些见解，我们引入了一个三层取证框架来识别和排除特洛伊木马的梯度，并在FL过程中恢复性能。我们考虑了三种类型的自适应攻击，并证明了STDLens对高级攻击者的健壮性。大量的实验表明，STDLens能够保护FL免受不同模型的劫持攻击，并且在识别和去除特洛伊木马梯度方面优于现有的方法，具有明显更高的精度和更低的误检率。



## **18. Dynamic Gradient Balancing for Enhanced Adversarial Attacks on Multi-Task Models**

多任务模型增强对抗性攻击的动态梯度平衡算法 cs.LG

19 pages, 5 figures

**SubmitDate**: 2023-05-20    [abs](http://arxiv.org/abs/2305.12066v1) [paper-pdf](http://arxiv.org/pdf/2305.12066v1)

**Authors**: Lijun Zhang, Xiao Liu, Kaleel Mahmood, Caiwen Ding, Hui Guan

**Abstract**: Multi-task learning (MTL) creates a single machine learning model called multi-task model to simultaneously perform multiple tasks. Although the security of single task classifiers has been extensively studied, there are several critical security research questions for multi-task models including 1) How secure are multi-task models to single task adversarial machine learning attacks, 2) Can adversarial attacks be designed to attack multiple tasks simultaneously, and 3) Does task sharing and adversarial training increase multi-task model robustness to adversarial attacks? In this paper, we answer these questions through careful analysis and rigorous experimentation. First, we develop na\"ive adaptation of single-task white-box attacks and analyze their inherent drawbacks. We then propose a novel attack framework, Dynamic Gradient Balancing Attack (DGBA). Our framework poses the problem of attacking a multi-task model as an optimization problem based on averaged relative loss change, which can be solved by approximating the problem as an integer linear programming problem. Extensive evaluation on two popular MTL benchmarks, NYUv2 and Tiny-Taxonomy, demonstrates the effectiveness of DGBA compared to na\"ive multi-task attack baselines on both clean and adversarially trained multi-task models. The results also reveal a fundamental trade-off between improving task accuracy by sharing parameters across tasks and undermining model robustness due to increased attack transferability from parameter sharing.

摘要: 多任务学习(MTL)创建了一种称为多任务模型的单机学习模型，用于同时执行多个任务。尽管单任务分类器的安全性已经得到了广泛的研究，但多任务模型仍然存在几个关键的安全研究问题，包括：1)多任务模型对抗性机器学习攻击的安全性如何；2)对抗性攻击能否被设计为同时攻击多个任务；3)任务共享和对抗性训练是否提高了多任务模型对对抗性攻击的健壮性？在本文中，我们通过仔细的分析和严谨的实验回答了这些问题。首先，我们提出了单任务白盒攻击的自适应方法，并分析了它们的固有缺陷。然后，我们提出了一种新的攻击框架--动态梯度平衡攻击(DGBA)。我们的框架将攻击多任务模型的问题归结为一个基于平均相对损失变化的优化问题，该问题可以通过将问题近似为一个整数线性规划问题来解决。对两个流行的MTL基准NYUv2和Tiny-Taxonomy进行了广泛的评估，证明了DGBA相对于NAIVE多任务攻击基线在干净和恶意训练的多任务模型上的有效性。结果还揭示了通过在任务间共享参数来提高任务精度和由于参数共享增加攻击可传递性而削弱模型稳健性之间的基本权衡。



## **19. DAP: A Dynamic Adversarial Patch for Evading Person Detectors**

DAP：一种用于躲避个人探测器的动态对抗性补丁 cs.CR

**SubmitDate**: 2023-05-19    [abs](http://arxiv.org/abs/2305.11618v1) [paper-pdf](http://arxiv.org/pdf/2305.11618v1)

**Authors**: Amira Guesmi, Ruitian Ding, Muhammad Abdullah Hanif, Ihsen Alouani, Muhammad Shafique

**Abstract**: In this paper, we present a novel approach for generating naturalistic adversarial patches without using GANs. Our proposed approach generates a Dynamic Adversarial Patch (DAP) that looks naturalistic while maintaining high attack efficiency and robustness in real-world scenarios. To achieve this, we redefine the optimization problem by introducing a new objective function, where a similarity metric is used to construct a similarity loss. This guides the patch to follow predefined patterns while maximizing the victim model's loss function. Our technique is based on directly modifying the pixel values in the patch which gives higher flexibility and larger space to incorporate multiple transformations compared to the GAN-based techniques. Furthermore, most clothing-based physical attacks assume static objects and ignore the possible transformations caused by non-rigid deformation due to changes in a person's pose. To address this limitation, we incorporate a ``Creases Transformation'' (CT) block, i.e., a preprocessing block following an Expectation Over Transformation (EOT) block used to generate a large variation of transformed patches incorporated in the training process to increase its robustness to different possible real-world distortions (e.g., creases in the clothing, rotation, re-scaling, random noise, brightness and contrast variations, etc.). We demonstrate that the presence of different real-world variations in clothing and object poses (i.e., above-mentioned distortions) lead to a drop in the performance of state-of-the-art attacks. For instance, these techniques can merely achieve 20\% in the physical world and 30.8\% in the digital world while our attack provides superior success rate of up to 65\% and 84.56\%, respectively when attacking the YOLOv3tiny detector deployed in smart cameras at the edge.

摘要: 在这篇文章中，我们提出了一种新的方法来生成自然的对抗性补丁，而不使用遗传算法。我们提出的方法生成了一个动态的对抗性补丁(DAP)，它看起来很自然，同时在真实场景中保持了高攻击效率和健壮性。为了实现这一点，我们通过引入一个新的目标函数来重新定义优化问题，其中使用一个相似性度量来构造相似性损失。这将引导补丁遵循预定义的模式，同时最大化受害者模型的损失函数。我们的技术基于直接修改贴片中的像素值，与基于GaN的技术相比，这提供了更高的灵活性和更大的空间来合并多个变换。此外，大多数基于服装的物理攻击假设静态对象，而忽略了由于人的姿势变化而导致的非刚性变形可能导致的变形。为了解决这一限制，我们结合了“折痕变换”(CT)块，即，在用于生成包含在训练过程中的变换后的面片的大变化的期望过度变换(EOT)块之后的预处理块，以增加其对不同可能的真实世界失真(例如，衣服中的折痕、旋转、重新缩放、随机噪声、亮度和对比度变化等)的稳健性。我们证明，服装和物体姿势的不同真实世界变化(即上述失真)的存在会导致最先进攻击的性能下降。例如，这些技术在现实世界中只能达到20%的成功率，在数字世界中只能达到30.8%，而我们的攻击在攻击部署在边缘的智能摄像机中的YOLOv3微小探测器时，分别提供了高达65%和84.56%的优越成功率。



## **20. Mitigating Backdoor Poisoning Attacks through the Lens of Spurious Correlation**

通过伪关联镜头缓解后门中毒攻击 cs.CL

14 pages, 4 figures

**SubmitDate**: 2023-05-19    [abs](http://arxiv.org/abs/2305.11596v1) [paper-pdf](http://arxiv.org/pdf/2305.11596v1)

**Authors**: Xuanli He, Qiongkai Xu, Jun Wang, Benjamin Rubinstein, Trevor Cohn

**Abstract**: Modern NLP models are often trained over large untrusted datasets, raising the potential for a malicious adversary to compromise model behaviour. For instance, backdoors can be implanted through crafting training instances with a specific textual trigger and a target label. This paper posits that backdoor poisoning attacks exhibit spurious correlation between simple text features and classification labels, and accordingly, proposes methods for mitigating spurious correlation as means of defence. Our empirical study reveals that the malicious triggers are highly correlated to their target labels; therefore such correlations are extremely distinguishable compared to those scores of benign features, and can be used to filter out potentially problematic instances. Compared with several existing defences, our defence method significantly reduces attack success rates across backdoor attacks, and in the case of insertion based attacks, our method provides a near-perfect defence.

摘要: 现代NLP模型通常是在不可信的大型数据集上进行训练的，这增加了恶意对手危害模型行为的可能性。例如，可以通过制作带有特定文本触发器和目标标签的训练实例来植入后门。文章认为，后门中毒攻击在简单文本特征和分类标签之间表现出伪相关性，并相应地提出了减轻伪相关性的方法作为防御手段。我们的经验研究表明，恶意触发器与其目标标签高度相关；因此，与那些良性特征相比，这种相关性非常容易区分，可以用来过滤潜在的问题实例。与现有的几种防御方法相比，我们的防御方法显著降低了后门攻击的攻击成功率，并且在基于插入的攻击中，我们的方法提供了近乎完美的防御。



## **21. Denial-of-Service or Fine-Grained Control: Towards Flexible Model Poisoning Attacks on Federated Learning**

拒绝服务或细粒度控制：对联邦学习的灵活模型中毒攻击 cs.LG

**SubmitDate**: 2023-05-19    [abs](http://arxiv.org/abs/2304.10783v2) [paper-pdf](http://arxiv.org/pdf/2304.10783v2)

**Authors**: Hangtao Zhang, Zeming Yao, Leo Yu Zhang, Shengshan Hu, Chao Chen, Alan Liew, Zhetao Li

**Abstract**: Federated learning (FL) is vulnerable to poisoning attacks, where adversaries corrupt the global aggregation results and cause denial-of-service (DoS). Unlike recent model poisoning attacks that optimize the amplitude of malicious perturbations along certain prescribed directions to cause DoS, we propose a Flexible Model Poisoning Attack (FMPA) that can achieve versatile attack goals. We consider a practical threat scenario where no extra knowledge about the FL system (e.g., aggregation rules or updates on benign devices) is available to adversaries. FMPA exploits the global historical information to construct an estimator that predicts the next round of the global model as a benign reference. It then fine-tunes the reference model to obtain the desired poisoned model with low accuracy and small perturbations. Besides the goal of causing DoS, FMPA can be naturally extended to launch a fine-grained controllable attack, making it possible to precisely reduce the global accuracy. Armed with precise control, malicious FL service providers can gain advantages over their competitors without getting noticed, hence opening a new attack surface in FL other than DoS. Even for the purpose of DoS, experiments show that FMPA significantly decreases the global accuracy, outperforming six state-of-the-art attacks.The code can be found at https://github.com/ZhangHangTao/Poisoning-Attack-on-FL.

摘要: 联合学习(FL)容易受到中毒攻击，攻击者破坏全局聚合结果并导致拒绝服务(DoS)。与目前的模型中毒攻击不同的是，我们提出了一种灵活的模型中毒攻击(FMPA)，它可以实现多种攻击目标。我们考虑了一个实际的威胁场景，其中没有关于FL系统的额外知识(例如，聚合规则或良性设备上的更新)可供攻击者使用。FMPA利用全球历史信息来构建一个估计器，该估计器预测下一轮全球模型作为良性参考。然后，它微调参考模型，以获得所需的低精度和小扰动的中毒模型。除了造成DoS的目标外，FMPA还可以自然地扩展到发起细粒度的可控攻击，从而有可能精准地降低全局精度。拥有精确控制的恶意FL服务提供商可以在不被注意的情况下获得相对于竞争对手的优势，从而打开了FL除DoS之外的新攻击面。即使是在拒绝服务攻击的情况下，实验也表明，FMPA显著降低了全局精度，性能超过了六种最先进的攻击。代码可以在https://github.com/ZhangHangTao/Poisoning-Attack-on-FL.上找到



## **22. Free Lunch for Privacy Preserving Distributed Graph Learning**

保护隐私的分布式图学习的免费午餐 cs.LG

**SubmitDate**: 2023-05-19    [abs](http://arxiv.org/abs/2305.10869v2) [paper-pdf](http://arxiv.org/pdf/2305.10869v2)

**Authors**: Nimesh Agrawal, Nikita Malik, Sandeep Kumar

**Abstract**: Learning on graphs is becoming prevalent in a wide range of applications including social networks, robotics, communication, medicine, etc. These datasets belonging to entities often contain critical private information. The utilization of data for graph learning applications is hampered by the growing privacy concerns from users on data sharing. Existing privacy-preserving methods pre-process the data to extract user-side features, and only these features are used for subsequent learning. Unfortunately, these methods are vulnerable to adversarial attacks to infer private attributes. We present a novel privacy-respecting framework for distributed graph learning and graph-based machine learning. In order to perform graph learning and other downstream tasks on the server side, this framework aims to learn features as well as distances without requiring actual features while preserving the original structural properties of the raw data. The proposed framework is quite generic and highly adaptable. We demonstrate the utility of the Euclidean space, but it can be applied with any existing method of distance approximation and graph learning for the relevant spaces. Through extensive experimentation on both synthetic and real datasets, we demonstrate the efficacy of the framework in terms of comparing the results obtained without data sharing to those obtained with data sharing as a benchmark. This is, to our knowledge, the first privacy-preserving distributed graph learning framework.

摘要: 基于图的学习在社交网络、机器人、通信、医学等广泛的应用中变得普遍。这些属于实体的数据集通常包含关键的私人信息。由于用户对数据共享的隐私性日益关注，阻碍了用于图形学习应用的数据的利用。现有的隐私保护方法对数据进行预处理，提取用户侧特征，只有这些特征才能用于后续学习。不幸的是，这些方法容易受到敌意攻击来推断私有属性。提出了一种新的隐私保护框架，用于分布式图学习和基于图的机器学习。为了在服务器端执行图学习和其他下游任务，该框架旨在学习特征和距离，而不需要实际特征，同时保持原始数据的原始结构属性。该框架具有较强的通用性和较强的适应性。我们演示了欧几里得空间的效用，但它可以应用于相关空间的任何现有的距离逼近和图学习方法。通过在合成数据集和真实数据集上的广泛实验，我们证明了该框架在比较没有数据共享的结果和以数据共享为基准的结果方面的有效性。据我们所知，这是第一个隐私保护的分布式图学习框架。



## **23. Security of Nakamoto Consensus under Congestion**

拥堵条件下中本共识的安全性 cs.CR

**SubmitDate**: 2023-05-19    [abs](http://arxiv.org/abs/2303.09113v2) [paper-pdf](http://arxiv.org/pdf/2303.09113v2)

**Authors**: Lucianna Kiffer, Joachim Neu, Srivatsan Sridhar, Aviv Zohar, David Tse

**Abstract**: Nakamoto consensus (NC) powers major proof-of-work (PoW) and proof-of-stake (PoS) blockchains such as Bitcoin or Cardano. Given a network of nodes with certain communication and computation capacities, against what fraction of adversarial power (the resilience) is Nakamoto consensus secure for a given block production rate? Prior security analyses of NC used a bounded delay model which does not capture network congestion resulting from high block production rates, bursty release of adversarial blocks, and in PoS, spamming due to equivocations. For PoW, we find a new attack, called teasing attack, that exploits congestion to increase the time taken to download and verify blocks, thereby succeeding at lower adversarial power than the private attack which was deemed to be the worst-case attack in prior analysis. By adopting a bounded bandwidth model to capture congestion, and through an improved analysis method, we identify the resilience of PoW NC for a given block production rate. In PoS, we augment our attack with equivocations to further increase congestion, making the vanilla PoS NC protocol insecure against any adversarial power except at very low block production rates. To counter equivocation spamming in PoS, we present a new NC-style protocol Sanitizing PoS (SaPoS) which achieves the same resilience as PoW NC.

摘要: Nakamoto Consensus(NC)为比特币或Cardano等主要工作证明(PoW)和风险证明(POS)区块链提供支持。给定一个具有一定通信和计算能力的节点网络，对于给定的块生产率，相对于多小部分的对抗能力(弹性)，Nakamoto共识是安全的？以前对NC的安全分析使用的是有界延迟模型，该模型没有捕获由于高块产生率、敌意块的突发释放以及PoS中由于模棱两可而产生的垃圾邮件而导致的网络拥塞。对于POW，我们发现了一种新的攻击，称为逗弄攻击，它利用拥塞来增加下载和验证块所需的时间，从而在较低的对抗能力下成功，而私人攻击在先前的分析中被认为是最糟糕的攻击。通过采用有限带宽模型捕获拥塞，并通过一种改进的分析方法，我们识别了PoW NC在给定的块生产率下的弹性。在PoS中，我们使用模棱两可的方式来增强我们的攻击，以进一步增加拥塞，使得Vanilla PoS NC协议不能抵抗任何敌意力量，除非在非常低的块生产率下。为了对抗PoS中的模棱两可的垃圾邮件，我们提出了一种新的NC风格的POS协议Sapos，该协议具有与PoW NC相同的弹性。



## **24. Quantifying the robustness of deep multispectral segmentation models against natural perturbations and data poisoning**

量化深层多光谱分割模型对自然扰动和数据中毒的稳健性 cs.CV

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.11347v1) [paper-pdf](http://arxiv.org/pdf/2305.11347v1)

**Authors**: Elise Bishoff, Charles Godfrey, Myles McKay, Eleanor Byler

**Abstract**: In overhead image segmentation tasks, including additional spectral bands beyond the traditional RGB channels can improve model performance. However, it is still unclear how incorporating this additional data impacts model robustness to adversarial attacks and natural perturbations. For adversarial robustness, the additional information could improve the model's ability to distinguish malicious inputs, or simply provide new attack avenues and vulnerabilities. For natural perturbations, the additional information could better inform model decisions and weaken perturbation effects or have no significant influence at all. In this work, we seek to characterize the performance and robustness of a multispectral (RGB and near infrared) image segmentation model subjected to adversarial attacks and natural perturbations. While existing adversarial and natural robustness research has focused primarily on digital perturbations, we prioritize on creating realistic perturbations designed with physical world conditions in mind. For adversarial robustness, we focus on data poisoning attacks whereas for natural robustness, we focus on extending ImageNet-C common corruptions for fog and snow that coherently and self-consistently perturbs the input data. Overall, we find both RGB and multispectral models are vulnerable to data poisoning attacks regardless of input or fusion architectures and that while physically realizable natural perturbations still degrade model performance, the impact differs based on fusion architecture and input data.

摘要: 在开销图像分割任务中，在传统的RGB通道之外加入额外的光谱段可以提高模型的性能。然而，纳入这些额外的数据如何影响模型对对抗性攻击和自然扰动的稳健性仍不清楚。对于对手的稳健性，额外的信息可以提高模型区分恶意输入的能力，或者只是提供新的攻击途径和漏洞。对于自然扰动，附加信息可以更好地为模型决策提供信息，减弱扰动效应或根本没有显著影响。在这项工作中，我们试图表征一个多光谱(RGB和近红外)图像分割模型在敌意攻击和自然扰动下的性能和稳健性。虽然现有的对抗性和自然健壮性研究主要集中在数字扰动上，但我们优先考虑创建考虑到物理世界条件的现实扰动。对于敌意的健壮性，我们关注的是数据中毒攻击，而对于自然健壮性，我们关注的是扩展ImageNet-C常见的雾和雪的损坏，它们一致地和自我一致地扰乱输入数据。总体而言，我们发现RGB和多光谱模型都容易受到数据中毒攻击，无论输入或融合架构如何，尽管物理上可实现的自然扰动仍会降低模型的性能，但影响因融合架构和输入数据而异。



## **25. On the Noise Stability and Robustness of Adversarially Trained Networks on NVM Crossbars**

NVM Crosbar上对抗性训练网络的噪声稳定性和稳健性 cs.LG

13 pages, 14 figures

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2109.09060v2) [paper-pdf](http://arxiv.org/pdf/2109.09060v2)

**Authors**: Chun Tao, Deboleena Roy, Indranil Chakraborty, Kaushik Roy

**Abstract**: Applications based on Deep Neural Networks (DNNs) have grown exponentially in the past decade. To match their increasing computational needs, several Non-Volatile Memory (NVM) crossbar based accelerators have been proposed. Recently, researchers have shown that apart from improved energy efficiency and performance, such approximate hardware also possess intrinsic robustness for defense against adversarial attacks. Prior works quantified this intrinsic robustness for vanilla DNNs trained on unperturbed inputs. However, adversarial training of DNNs is the benchmark technique for robustness, and sole reliance on intrinsic robustness of the hardware may not be sufficient. In this work, we explore the design of robust DNNs through the amalgamation of adversarial training and intrinsic robustness of NVM crossbar-based analog hardware. First, we study the noise stability of such networks on unperturbed inputs and observe that internal activations of adversarially trained networks have lower Signal-to-Noise Ratio (SNR), and are sensitive to noise compared to vanilla networks. As a result, they suffer on average 2x performance degradation due to the approximate computations on analog hardware. Noise stability analyses show the instability of adversarially trained DNNs. On the other hand, for adversarial images generated using Square Black Box attacks, ResNet-10/20 adversarially trained on CIFAR-10/100 display a robustness gain of 20-30%. For adversarial images generated using Projected-Gradient-Descent (PGD) White-Box attacks, adversarially trained DNNs present a 5-10% gain in robust accuracy due to underlying NVM crossbar when $\epsilon_{attack}$ is greater than $\epsilon_{train}$. Our results indicate that implementing adversarially trained networks on analog hardware requires careful calibration between hardware non-idealities and $\epsilon_{train}$ for optimum robustness and performance.

摘要: 基于深度神经网络(DNN)的应用在过去十年中呈指数级增长。为了满足他们日益增长的计算需求，已经提出了几种基于非易失性存储器(NVM)交叉开关的加速器。最近，研究人员表明，除了提高能量效率和性能外，这种近似硬件还具有内在的健壮性，用于防御对手攻击。以前的工作量化了在不受干扰的输入上训练的普通DNN的这种内在稳健性。然而，DNN的对抗性训练是健壮性的基准技术，仅依赖硬件的内在健壮性可能是不够的。在这项工作中，我们通过融合对抗性训练和基于NVM Crosbar的模拟硬件的内在健壮性来探索健壮DNN的设计。首先，我们研究了这类网络在无扰动输入下的噪声稳定性，观察到对抗性训练网络的内部激活具有较低的信噪比，并且与香草网络相比对噪声敏感。因此，由于在模拟硬件上进行近似计算，它们的性能平均下降2倍。噪声稳定性分析表明，反向训练的DNN是不稳定的。另一方面，对于使用Square黑盒攻击生成的对抗性图像，在CIFAR-10/100上进行对抗性训练的ResNet-10/20显示出20%-30%的稳健性收益。对于使用投影梯度下降(PGD)白盒攻击生成的敌意图像，当$\epsilon_{Attack}$大于$\epsilon_{Train}$时，由于潜在的NVM纵横杆，经过对抗性训练的DNN的稳健准确率提高了5%-10%。我们的结果表明，要在模拟硬件上实现对抗性训练的网络，需要在硬件非理想性和训练性能之间进行仔细的校准，以获得最佳的健壮性和性能。



## **26. TrustSER: On the Trustworthiness of Fine-tuning Pre-trained Speech Embeddings For Speech Emotion Recognition**

TrustSER：用于语音情感识别的精调预训练语音嵌入的可信性 cs.SD

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.11229v1) [paper-pdf](http://arxiv.org/pdf/2305.11229v1)

**Authors**: Tiantian Feng, Rajat Hebbar, Shrikanth Narayanan

**Abstract**: Recent studies have explored the use of pre-trained embeddings for speech emotion recognition (SER), achieving comparable performance to conventional methods that rely on low-level knowledge-inspired acoustic features. These embeddings are often generated from models trained on large-scale speech datasets using self-supervised or weakly-supervised learning objectives. Despite the significant advancements made in SER through the use of pre-trained embeddings, there is a limited understanding of the trustworthiness of these methods, including privacy breaches, unfair performance, vulnerability to adversarial attacks, and computational cost, all of which may hinder the real-world deployment of these systems. In response, we introduce TrustSER, a general framework designed to evaluate the trustworthiness of SER systems using deep learning methods, with a focus on privacy, safety, fairness, and sustainability, offering unique insights into future research in the field of SER. Our code is publicly available under: https://github.com/usc-sail/trust-ser.

摘要: 最近的研究探索了将预先训练的嵌入用于语音情感识别(SER)，获得了与依赖于低水平知识启发的声学特征的传统方法相当的性能。这些嵌入通常是从使用自监督或弱监督学习目标在大规模语音数据集上训练的模型生成的。尽管通过使用预先训练的嵌入在SER中取得了重大进展，但人们对这些方法的可信性的了解有限，包括隐私泄露、不公平的性能、对对手攻击的脆弱性和计算成本，所有这些都可能阻碍这些系统的真实世界的部署。作为回应，我们引入了TrustSER，这是一个通用框架，旨在使用深度学习方法评估SER系统的可信性，重点关注隐私、安全、公平和可持续性，为未来SER领域的研究提供独特的见解。我们的代码在以下位置公开提供：https://github.com/usc-sail/trust-ser.



## **27. Attacks on Online Learners: a Teacher-Student Analysis**

对网络学习者的攻击：一种师生分析 stat.ML

15 pages, 6 figures

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.11132v1) [paper-pdf](http://arxiv.org/pdf/2305.11132v1)

**Authors**: Riccardo Giuseppe Margiotta, Sebastian Goldt, Guido Sanguinetti

**Abstract**: Machine learning models are famously vulnerable to adversarial attacks: small ad-hoc perturbations of the data that can catastrophically alter the model predictions. While a large literature has studied the case of test-time attacks on pre-trained models, the important case of attacks in an online learning setting has received little attention so far. In this work, we use a control-theoretical perspective to study the scenario where an attacker may perturb data labels to manipulate the learning dynamics of an online learner. We perform a theoretical analysis of the problem in a teacher-student setup, considering different attack strategies, and obtaining analytical results for the steady state of simple linear learners. These results enable us to prove that a discontinuous transition in the learner's accuracy occurs when the attack strength exceeds a critical threshold. We then study empirically attacks on learners with complex architectures using real data, confirming the insights of our theoretical analysis. Our findings show that greedy attacks can be extremely efficient, especially when data stream in small batches.

摘要: 众所周知，机器学习模型容易受到对抗性攻击：对数据的微小特别扰动可能会灾难性地改变模型预测。虽然有大量文献研究了测试时间攻击预先训练的模型的案例，但到目前为止，在线学习环境中的重要攻击案例很少受到关注。在这项工作中，我们使用控制理论的观点来研究攻击者可能扰乱数据标签以操纵在线学习者的学习动态的场景。我们在教师-学生系统中对该问题进行了理论分析，考虑了不同的攻击策略，得到了简单线性学习者稳态的解析结果。这些结果使我们能够证明，当攻击强度超过临界阈值时，学习者的准确率会发生不连续的转变。然后，我们使用真实数据对具有复杂架构的学习者进行了实证研究，证实了我们的理论分析的真知灼见。我们的发现表明，贪婪攻击可以非常有效，特别是当数据流以小批量传输时。



## **28. Deep PackGen: A Deep Reinforcement Learning Framework for Adversarial Network Packet Generation**

Deep PackGen：一种用于对抗性网络数据包生成的深度强化学习框架 cs.CR

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.11039v1) [paper-pdf](http://arxiv.org/pdf/2305.11039v1)

**Authors**: Soumyadeep Hore, Jalal Ghadermazi, Diwas Paudel, Ankit Shah, Tapas K. Das, Nathaniel D. Bastian

**Abstract**: Recent advancements in artificial intelligence (AI) and machine learning (ML) algorithms, coupled with the availability of faster computing infrastructure, have enhanced the security posture of cybersecurity operations centers (defenders) through the development of ML-aided network intrusion detection systems (NIDS). Concurrently, the abilities of adversaries to evade security have also increased with the support of AI/ML models. Therefore, defenders need to proactively prepare for evasion attacks that exploit the detection mechanisms of NIDS. Recent studies have found that the perturbation of flow-based and packet-based features can deceive ML models, but these approaches have limitations. Perturbations made to the flow-based features are difficult to reverse-engineer, while samples generated with perturbations to the packet-based features are not playable.   Our methodological framework, Deep PackGen, employs deep reinforcement learning to generate adversarial packets and aims to overcome the limitations of approaches in the literature. By taking raw malicious network packets as inputs and systematically making perturbations on them, Deep PackGen camouflages them as benign packets while still maintaining their functionality. In our experiments, using publicly available data, Deep PackGen achieved an average adversarial success rate of 66.4\% against various ML models and across different attack types. Our investigation also revealed that more than 45\% of the successful adversarial samples were out-of-distribution packets that evaded the decision boundaries of the classifiers. The knowledge gained from our study on the adversary's ability to make specific evasive perturbations to different types of malicious packets can help defenders enhance the robustness of their NIDS against evolving adversarial attacks.

摘要: 人工智能(AI)和机器学习(ML)算法的最新进展，加上更快的计算基础设施的可用性，通过开发ML辅助的网络入侵检测系统(NID)，增强了网络安全运营中心(防御者)的安全态势。同时，在AI/ML模型的支持下，攻击者逃避安全的能力也有所增强。因此，防御者需要主动准备利用网络入侵检测系统的检测机制进行规避攻击。最近的研究发现，基于流和基于分组的特征的扰动可以欺骗ML模型，但这些方法都有局限性。对基于流的特征进行的扰动很难进行反向工程，而利用对基于分组的特征的扰动生成的样本是不可播放的。我们的方法框架，Deep PackGen，使用深度强化学习来生成对抗性分组，旨在克服文献中方法的局限性。通过将原始恶意网络数据包作为输入并系统地对其进行干扰，Deep PackGen将其伪装成良性数据包，同时仍保持其功能。在我们的实验中，使用公开的数据，Deep PackGen在不同的ML模型和不同的攻击类型上获得了66.4%的平均攻击成功率。我们的调查还发现，超过45%的成功对抗样本是绕过分类器决策边界的非分发分组。我们从研究对手对不同类型的恶意数据包进行特定规避扰动的能力中获得的知识可以帮助防御者增强其网络入侵检测系统对不断演变的敌意攻击的健壮性。



## **29. SoK: Data Privacy in Virtual Reality**

SOK：虚拟现实中的数据隐私 cs.HC

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2301.05940v2) [paper-pdf](http://arxiv.org/pdf/2301.05940v2)

**Authors**: Gonzalo Munilla Garrido, Vivek Nair, Dawn Song

**Abstract**: The adoption of virtual reality (VR) technologies has rapidly gained momentum in recent years as companies around the world begin to position the so-called "metaverse" as the next major medium for accessing and interacting with the internet. While consumers have become accustomed to a degree of data harvesting on the web, the real-time nature of data sharing in the metaverse indicates that privacy concerns are likely to be even more prevalent in the new "Web 3.0." Research into VR privacy has demonstrated that a plethora of sensitive personal information is observable by various would-be adversaries from just a few minutes of telemetry data. On the other hand, we have yet to see VR parallels for many privacy-preserving tools aimed at mitigating threats on conventional platforms. This paper aims to systematize knowledge on the landscape of VR privacy threats and countermeasures by proposing a comprehensive taxonomy of data attributes, protections, and adversaries based on the study of 68 collected publications. We complement our qualitative discussion with a statistical analysis of the risk associated with various data sources inherent to VR in consideration of the known attacks and defenses. By focusing on highlighting the clear outstanding opportunities, we hope to motivate and guide further research into this increasingly important field.

摘要: 近年来，虚拟现实(VR)技术的采用势头迅速增强，世界各地的公司开始将所谓的“虚拟现实”定位为访问互联网和与互联网互动的下一个主要媒介。虽然消费者已经习惯了在一定程度上从网络上获取数据，但虚拟世界中数据共享的实时性质表明，对隐私的担忧可能会在新的“Web 3.0”中更加普遍。对VR隐私的研究表明，各种潜在的对手只需几分钟的遥测数据就可以观察到过多的敏感个人信息。另一方面，我们还没有看到许多旨在缓解传统平台上威胁的隐私保护工具的VR相似之处。本文旨在通过对收集到的68种出版物的研究，提出数据属性、保护和对手的全面分类，以系统化关于虚拟现实隐私威胁和对策的知识。考虑到已知的攻击和防御，我们用与VR固有的各种数据源相关的风险的统计分析来补充我们的定性讨论。通过重点突出明确的突出机遇，我们希望激励和指导对这一日益重要的领域的进一步研究。



## **30. Certified Robust Neural Networks: Generalization and Corruption Resistance**

认证的稳健神经网络：泛化和抗腐蚀性 stat.ML

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2303.02251v2) [paper-pdf](http://arxiv.org/pdf/2303.02251v2)

**Authors**: Amine Bennouna, Ryan Lucas, Bart Van Parys

**Abstract**: Recent work have demonstrated that robustness (to "corruption") can be at odds with generalization. Adversarial training, for instance, aims to reduce the problematic susceptibility of modern neural networks to small data perturbations. Surprisingly, overfitting is a major concern in adversarial training despite being mostly absent in standard training. We provide here theoretical evidence for this peculiar "robust overfitting" phenomenon. Subsequently, we advance a novel distributionally robust loss function bridging robustness and generalization. We demonstrate both theoretically as well as empirically the loss to enjoy a certified level of robustness against two common types of corruption--data evasion and poisoning attacks--while ensuring guaranteed generalization. We show through careful numerical experiments that our resulting holistic robust (HR) training procedure yields SOTA performance. Finally, we indicate that HR training can be interpreted as a direct extension of adversarial training and comes with a negligible additional computational burden. A ready-to-use python library implementing our algorithm is available at https://github.com/RyanLucas3/HR_Neural_Networks.

摘要: 最近的研究表明，健壮性(对“腐败”)可能与泛化不一致。例如，对抗性训练旨在降低现代神经网络对小数据扰动的问题敏感度。令人惊讶的是，尽管在标准训练中大多缺席，但过度适应是对抗性训练中的一个主要问题。我们在这里为这一特殊的“稳健过拟合”现象提供了理论证据。随后，我们提出了一种新的分布稳健损失函数，它在稳健性和泛化之间架起了桥梁。我们在理论上和经验上都证明了在确保泛化的同时，对两种常见的腐败类型--数据逃避和中毒攻击--具有经过认证的健壮性水平。我们通过仔细的数值实验表明，我们由此产生的整体稳健(HR)训练过程产生了SOTA性能。最后，我们指出，HR训练可以被解释为对抗性训练的直接扩展，并且伴随着可以忽略的额外计算负担。在https://github.com/RyanLucas3/HR_Neural_Networks.上有一个实现我们的算法的现成的Python库



## **31. Architecture-agnostic Iterative Black-box Certified Defense against Adversarial Patches**

架构不可知的迭代黑盒认证防御对手补丁 cs.CV

9 pages

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.10929v1) [paper-pdf](http://arxiv.org/pdf/2305.10929v1)

**Authors**: Di Yang, Yihao Huang, Qing Guo, Felix Juefei-Xu, Ming Hu, Yang Liu, Geguang Pu

**Abstract**: The adversarial patch attack aims to fool image classifiers within a bounded, contiguous region of arbitrary changes, posing a real threat to computer vision systems (e.g., autonomous driving, content moderation, biometric authentication, medical imaging) in the physical world. To address this problem in a trustworthy way, proposals have been made for certified patch defenses that ensure the robustness of classification models and prevent future patch attacks from breaching the defense. State-of-the-art certified defenses can be compatible with any model architecture, as well as achieve high clean and certified accuracy. Although the methods are adaptive to arbitrary patch positions, they inevitably need to access the size of the adversarial patch, which is unreasonable and impractical in real-world attack scenarios. To improve the feasibility of the architecture-agnostic certified defense in a black-box setting (i.e., position and size of the patch are both unknown), we propose a novel two-stage Iterative Black-box Certified Defense method, termed IBCD.In the first stage, it estimates the patch size in a search-based manner by evaluating the size relationship between the patch and mask with pixel masking. In the second stage, the accuracy results are calculated by the existing white-box certified defense methods with the estimated patch size. The experiments conducted on two popular model architectures and two datasets verify the effectiveness and efficiency of IBCD.

摘要: 敌意补丁攻击旨在愚弄任意变化的有界连续区域内的图像分类器，对物理世界中的计算机视觉系统(例如，自动驾驶、内容审核、生物特征验证、医学成像)构成真正的威胁。为了以可信的方式解决这个问题，已经提出了认证补丁防御的建议，以确保分类模型的健壮性，并防止未来的补丁攻击破坏防御。最先进的认证防御可以兼容任何型号的架构，以及实现高清洁和认证的准确性。虽然这些方法能够适应任意的补丁位置，但不可避免地需要访问敌方补丁的大小，这在现实世界的攻击场景中是不合理和不切实际的。为了提高在黑盒环境下(即补丁的位置和大小都未知)下基于体系结构的认证防御的可行性，提出了一种新的两阶段迭代黑盒认证防御方法IBCD。在第二阶段，利用现有的白盒认证防御方法和估计的补丁大小来计算精度结果。在两个流行的模型架构和两个数据集上进行的实验验证了IBCD的有效性和高效性。



## **32. How Deep Learning Sees the World: A Survey on Adversarial Attacks & Defenses**

深度学习如何看待世界：对抗性攻防研究综述 cs.CV

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.10862v1) [paper-pdf](http://arxiv.org/pdf/2305.10862v1)

**Authors**: Joana C. Costa, Tiago Roxo, Hugo Proença, Pedro R. M. Inácio

**Abstract**: Deep Learning is currently used to perform multiple tasks, such as object recognition, face recognition, and natural language processing. However, Deep Neural Networks (DNNs) are vulnerable to perturbations that alter the network prediction (adversarial examples), raising concerns regarding its usage in critical areas, such as self-driving vehicles, malware detection, and healthcare. This paper compiles the most recent adversarial attacks, grouped by the attacker capacity, and modern defenses clustered by protection strategies. We also present the new advances regarding Vision Transformers, summarize the datasets and metrics used in the context of adversarial settings, and compare the state-of-the-art results under different attacks, finishing with the identification of open issues.

摘要: 深度学习目前用于执行多个任务，如对象识别、人脸识别和自然语言处理。然而，深度神经网络(DNN)很容易受到改变网络预测的扰动(对手的例子)，这引发了人们对其在关键领域的使用的担忧，如自动驾驶车辆、恶意软件检测和医疗保健。这篇论文汇编了最新的对抗性攻击，按攻击者的能力分组，以及按保护策略分组的现代防御。我们还介绍了关于Vision Transformers的新进展，总结了在对抗性环境下使用的数据集和度量，并比较了不同攻击下的最新结果，最后确定了有待解决的问题。



## **33. Towards an Accurate and Secure Detector against Adversarial Perturbations**

走向准确和安全的检测器以抵御对手的扰动 cs.CV

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.10856v1) [paper-pdf](http://arxiv.org/pdf/2305.10856v1)

**Authors**: Chao Wang, Shuren Qi, Zhiqiu Huang, Yushu Zhang, Xiaochun Cao

**Abstract**: The vulnerability of deep neural networks to adversarial perturbations has been widely perceived in the computer vision community. From a security perspective, it poses a critical risk for modern vision systems, e.g., the popular Deep Learning as a Service (DLaaS) frameworks. For protecting off-the-shelf deep models while not modifying them, current algorithms typically detect adversarial patterns through discriminative decomposition of natural-artificial data. However, these decompositions are biased towards frequency or spatial discriminability, thus failing to capture subtle adversarial patterns comprehensively. More seriously, they are typically invertible, meaning successful defense-aware (secondary) adversarial attack (i.e., evading the detector as well as fooling the model) is practical under the assumption that the adversary is fully aware of the detector (i.e., the Kerckhoffs's principle). Motivated by such facts, we propose an accurate and secure adversarial example detector, relying on a spatial-frequency discriminative decomposition with secret keys. It expands the above works on two aspects: 1) the introduced Krawtchouk basis provides better spatial-frequency discriminability and thereby is more suitable for capturing adversarial patterns than the common trigonometric or wavelet basis; 2) the extensive parameters for decomposition are generated by a pseudo-random function with secret keys, hence blocking the defense-aware adversarial attack. Theoretical and numerical analysis demonstrates the increased accuracy and security of our detector w.r.t. a number of state-of-the-art algorithms.

摘要: 深度神经网络对对抗性扰动的脆弱性已经在计算机视觉领域得到了广泛的认识。从安全的角度来看，它对现代视觉系统构成了严重的风险，例如流行的深度学习即服务(DLaaS)框架。为了在不修改现有深度模型的同时保护它们，目前的算法通常通过对自然-人工数据的区别性分解来检测对抗性模式。然而，这些分解偏向于频率或空间可区分性，因此无法全面地捕捉到微妙的对抗性模式。更严重的是，它们通常是可逆的，这意味着在假设对手完全知道检测器(即Kerckhoff原理)的情况下，成功的防御感知(次要)对手攻击(即，躲避检测器以及愚弄模型)是实用的。在此基础上，提出了一种基于密钥的空频判别分解的准确、安全的对抗性样本检测器。它从两个方面对上述工作进行了扩展：1)引入的Krawtchouk基提供了更好的空频分辨能力，因此比普通的三角或小波基更适合于捕获敌意模式；2)分解的广泛参数是由带有密钥的伪随机函数产生的，从而阻止了具有防御意识的敌意攻击。理论和数值分析表明，我们的探测器的准确度和安全性都有所提高。一些最先进的算法。



## **34. Adversarial Scratches: Deployable Attacks to CNN Classifiers**

对抗性抓痕：对CNN分类器的可部署攻击 cs.LG

This work is published at Pattern Recognition (Elsevier). This paper  stems from 'Scratch that! An Evolution-based Adversarial Attack against  Neural Networks' for which an arXiv preprint is available at  arXiv:1912.02316. Further studies led to a complete overhaul of the work,  resulting in this paper

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2204.09397v3) [paper-pdf](http://arxiv.org/pdf/2204.09397v3)

**Authors**: Loris Giulivi, Malhar Jere, Loris Rossi, Farinaz Koushanfar, Gabriela Ciocarlie, Briland Hitaj, Giacomo Boracchi

**Abstract**: A growing body of work has shown that deep neural networks are susceptible to adversarial examples. These take the form of small perturbations applied to the model's input which lead to incorrect predictions. Unfortunately, most literature focuses on visually imperceivable perturbations to be applied to digital images that often are, by design, impossible to be deployed to physical targets. We present Adversarial Scratches: a novel L0 black-box attack, which takes the form of scratches in images, and which possesses much greater deployability than other state-of-the-art attacks. Adversarial Scratches leverage B\'ezier Curves to reduce the dimension of the search space and possibly constrain the attack to a specific location. We test Adversarial Scratches in several scenarios, including a publicly available API and images of traffic signs. Results show that, often, our attack achieves higher fooling rate than other deployable state-of-the-art methods, while requiring significantly fewer queries and modifying very few pixels.

摘要: 越来越多的研究表明，深度神经网络很容易受到敌意例子的影响。这些采用的形式是应用于模型输入的小扰动，从而导致不正确的预测。不幸的是，大多数文献关注的是应用于数字图像的视觉上不可察觉的扰动，而根据设计，数字图像通常不可能被部署到物理目标上。我们提出了对抗性划痕：一种新颖的L0黑盒攻击，它采用图像划痕的形式，并且比其他最先进的攻击具有更大的可部署性。对抗性划痕利用B‘ezier曲线来减少搜索空间的维度，并可能将攻击限制在特定位置。我们在几个场景中测试了对抗性划痕，包括公开可用的API和交通标志图像。结果表明，我们的攻击通常比其他可部署的最先进方法获得更高的愚骗率，同时需要的查询和修改的像素也非常少。



## **35. Adversarial Amendment is the Only Force Capable of Transforming an Enemy into a Friend**

对抗性修正案是唯一能化敌为友的力量 cs.AI

Accepted to IJCAI 2023, 10 pages, 5 figures

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.10766v1) [paper-pdf](http://arxiv.org/pdf/2305.10766v1)

**Authors**: Chong Yu, Tao Chen, Zhongxue Gan

**Abstract**: Adversarial attack is commonly regarded as a huge threat to neural networks because of misleading behavior. This paper presents an opposite perspective: adversarial attacks can be harnessed to improve neural models if amended correctly. Unlike traditional adversarial defense or adversarial training schemes that aim to improve the adversarial robustness, the proposed adversarial amendment (AdvAmd) method aims to improve the original accuracy level of neural models on benign samples. We thoroughly analyze the distribution mismatch between the benign and adversarial samples. This distribution mismatch and the mutual learning mechanism with the same learning ratio applied in prior art defense strategies is the main cause leading the accuracy degradation for benign samples. The proposed AdvAmd is demonstrated to steadily heal the accuracy degradation and even leads to a certain accuracy boost of common neural models on benign classification, object detection, and segmentation tasks. The efficacy of the AdvAmd is contributed by three key components: mediate samples (to reduce the influence of distribution mismatch with a fine-grained amendment), auxiliary batch norm (to solve the mutual learning mechanism and the smoother judgment surface), and AdvAmd loss (to adjust the learning ratios according to different attack vulnerabilities) through quantitative and ablation experiments.

摘要: 对抗性攻击通常被认为是对神经网络的巨大威胁，因为它具有误导性。本文提出了一种相反的观点：如果修正正确，可以利用对抗性攻击来改进神经模型。与传统的对抗性防御或对抗性训练方案不同，提出的对抗性修正(AdvAmd)方法旨在提高神经模型对良性样本的原始精度水平。我们深入分析了良性样本和恶意样本之间的分布不匹配。这种分布失配和现有技术防御策略中采用的具有相同学习比率的相互学习机制是导致良性样本精度下降的主要原因。实验结果表明，该算法能够稳定地修复神经网络模型在良性分类、目标检测和分割等任务中的精度下降，甚至可以在一定程度上提高神经模型的精度。通过定量和烧蚀实验，AdvAmd的有效性由三个关键成分贡献：中间样本(通过细粒度修正减少分布失配的影响)、辅助批次范数(解决相互学习机制和更光滑的判断曲面)和AdvAmd损失(根据不同的攻击漏洞调整学习比率)。



## **36. Re-thinking Data Availablity Attacks Against Deep Neural Networks**

对深度神经网络数据可用性攻击的再思考 cs.CR

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.10691v1) [paper-pdf](http://arxiv.org/pdf/2305.10691v1)

**Authors**: Bin Fang, Bo Li, Shuang Wu, Ran Yi, Shouhong Ding, Lizhuang Ma

**Abstract**: The unauthorized use of personal data for commercial purposes and the clandestine acquisition of private data for training machine learning models continue to raise concerns. In response to these issues, researchers have proposed availability attacks that aim to render data unexploitable. However, many current attack methods are rendered ineffective by adversarial training. In this paper, we re-examine the concept of unlearnable examples and discern that the existing robust error-minimizing noise presents an inaccurate optimization objective. Building on these observations, we introduce a novel optimization paradigm that yields improved protection results with reduced computational time requirements. We have conducted extensive experiments to substantiate the soundness of our approach. Moreover, our method establishes a robust foundation for future research in this area.

摘要: 未经授权将个人数据用于商业目的以及秘密获取用于训练机器学习模型的私人数据继续引起关注。针对这些问题，研究人员提出了旨在使数据无法利用的可用性攻击。然而，目前的许多进攻方法由于对抗性训练而变得无效。在本文中，我们重新检查了不可学习示例的概念，并发现现有的鲁棒误差最小化噪声呈现了一个不准确的优化目标。在这些观察的基础上，我们引入了一种新的优化范例，该范例可以在减少计算时间的情况下产生更好的保护结果。我们已经进行了广泛的实验，以证实我们的方法的合理性。此外，我们的方法为这一领域的未来研究奠定了坚实的基础。



## **37. Content-based Unrestricted Adversarial Attack**

基于内容的无限制对抗性攻击 cs.CV

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.10665v1) [paper-pdf](http://arxiv.org/pdf/2305.10665v1)

**Authors**: Zhaoyu Chen, Bo Li, Shuang Wu, Kaixun Jiang, Shouhong Ding, Wenqiang Zhang

**Abstract**: Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic, demonstrating their ability to deceive human perception and deep neural networks with stealth and success. However, current works usually sacrifice unrestricted degrees and subjectively select some image content to guarantee the photorealism of unrestricted adversarial examples, which limits its attack performance. To ensure the photorealism of adversarial examples and boost attack performance, we propose a novel unrestricted attack framework called Content-based Unrestricted Adversarial Attack. By leveraging a low-dimensional manifold that represents natural images, we map the images onto the manifold and optimize them along its adversarial direction. Therefore, within this framework, we implement Adversarial Content Attack based on Stable Diffusion and can generate high transferable unrestricted adversarial examples with various adversarial contents. Extensive experimentation and visualization demonstrate the efficacy of ACA, particularly in surpassing state-of-the-art attacks by an average of 13.3-50.4% and 16.8-48.0% in normally trained models and defense methods, respectively.

摘要: 不受限制的对抗性攻击通常会操纵图像的语义内容(例如，颜色或纹理)，以创建既有效又逼真的对抗性示例，展示它们以隐蔽和成功的方式欺骗人类感知和深层神经网络的能力。然而，目前的作品往往牺牲不受限制的程度，主观地选择一些图像内容来保证不受限制的对抗性例子的照片真实感，这限制了其攻击性能。为了保证对抗性实例的真实感，提高攻击性能，我们提出了一种新的无限制攻击框架，称为基于内容的无限对抗性攻击。通过利用表示自然图像的低维流形，我们将图像映射到流形上，并沿着其相反的方向进行优化。因此，在该框架下，我们实现了基于稳定扩散的对抗性内容攻击，并且可以生成具有多种对抗性内容的高可转移性的无限制对抗性实例。广泛的实验和可视化证明了蚁群算法的有效性，特别是在正常训练的模型和防御方法上，平均分别超过最先进的攻击13.3%-50.4%和16.8%-48.0%。



## **38. Exact Recovery for System Identification with More Corrupt Data than Clean Data**

使用比干净数据更多的损坏数据准确恢复系统标识 cs.LG

24 pages, 2 figures

**SubmitDate**: 2023-05-17    [abs](http://arxiv.org/abs/2305.10506v1) [paper-pdf](http://arxiv.org/pdf/2305.10506v1)

**Authors**: Baturalp Yalcin, Javad Lavaei, Murat Arcak

**Abstract**: In this paper, we study the system identification problem for linear discrete-time systems under adversaries and analyze two lasso-type estimators. We study both asymptotic and non-asymptotic properties of these estimators in two separate scenarios, corresponding to deterministic and stochastic models for the attack times. Since the samples collected from the system are correlated, the existing results on lasso are not applicable. We show that when the system is stable and the attacks are injected periodically, the sample complexity for the exact recovery of the system dynamics is O(n), where n is the dimension of the states. When the adversarial attacks occur at each time instance with probability p, the required sample complexity for the exact recovery scales as O(\log(n)p/(1-p)^2). This result implies the almost sure convergence to the true system dynamics under the asymptotic regime. As a by-product, even when more than half of the data is compromised, our estimators still learn the system correctly. This paper provides the first mathematical guarantee in the literature on learning from correlated data for dynamical systems in the case when there is less clean data than corrupt data.

摘要: 本文研究了对手作用下线性离散时间系统的系统辨识问题，分析了两种套索型估值器。在两种不同的情形下，我们研究了这些估计量的渐近和非渐近性质，分别对应于攻击时间的确定性和随机性模型。由于从系统采集的样本是相关的，现有的套索结果不适用。我们证明了当系统稳定且攻击被周期性注入时，精确恢复系统动力学的样本复杂度为O(N)，其中n是状态的维度。当对抗性攻击发生在概率为p的每个时刻时，精确恢复所需的样本复杂度为O(\log(N)p/(1-p)^2)。这一结果意味着在渐近状态下几乎必然收敛于真实的系统动力学。作为副产品，即使超过一半的数据被泄露，我们的估计者仍然正确地学习系统。本文首次为动态系统在干净数据少于损坏数据的情况下从相关数据中学习提供了数学保证。



## **39. Raising the Bar for Certified Adversarial Robustness with Diffusion Models**

利用扩散模型提高认证对抗性稳健性的标准 cs.LG

**SubmitDate**: 2023-05-17    [abs](http://arxiv.org/abs/2305.10388v1) [paper-pdf](http://arxiv.org/pdf/2305.10388v1)

**Authors**: Thomas Altstidl, David Dobre, Björn Eskofier, Gauthier Gidel, Leo Schwinn

**Abstract**: Certified defenses against adversarial attacks offer formal guarantees on the robustness of a model, making them more reliable than empirical methods such as adversarial training, whose effectiveness is often later reduced by unseen attacks. Still, the limited certified robustness that is currently achievable has been a bottleneck for their practical adoption. Gowal et al. and Wang et al. have shown that generating additional training data using state-of-the-art diffusion models can considerably improve the robustness of adversarial training. In this work, we demonstrate that a similar approach can substantially improve deterministic certified defenses. In addition, we provide a list of recommendations to scale the robustness of certified training approaches. One of our main insights is that the generalization gap, i.e., the difference between the training and test accuracy of the original model, is a good predictor of the magnitude of the robustness improvement when using additional generated data. Our approach achieves state-of-the-art deterministic robustness certificates on CIFAR-10 for the $\ell_2$ ($\epsilon = 36/255$) and $\ell_\infty$ ($\epsilon = 8/255$) threat models, outperforming the previous best results by $+3.95\%$ and $+1.39\%$, respectively. Furthermore, we report similar improvements for CIFAR-100.

摘要: 针对对抗性攻击的认证防御为模型的健壮性提供了正式保证，使它们比对抗性训练等经验方法更可靠，后者的有效性后来往往因看不见的攻击而降低。尽管如此，目前可以实现的有限的认证健壮性一直是它们实际采用的瓶颈。Gowal等人。和Wang等人。已经表明，使用最先进的扩散模型生成额外的训练数据可以显著提高对抗性训练的稳健性。在这项工作中，我们证明了类似的方法可以实质性地改进确定性认证防御。此外，我们还提供了一系列建议，以衡量认证培训方法的健壮性。我们的主要见解之一是，泛化差距，即原始模型的训练精度和测试精度之间的差异，是使用额外生成的数据时稳健性改善幅度的一个很好的预测指标。我们的方法在CIFAR-10上为$\ell_2$($\epsilon=36/255$)和$\ell_inty$($\epsilon=8/255$)威胁模型获得了最先进的确定性稳健性证书，分别比之前的最好结果高出$+3.95\$和$+1.39\$。此外，我们还报告了CIFAR-100的类似改进。



## **40. Certified Invertibility in Neural Networks via Mixed-Integer Programming**

基于混合整数规划的神经网络可逆性证明 cs.LG

22 pages, 7 figures

**SubmitDate**: 2023-05-17    [abs](http://arxiv.org/abs/2301.11783v2) [paper-pdf](http://arxiv.org/pdf/2301.11783v2)

**Authors**: Tianqi Cui, Thomas Bertalan, George J. Pappas, Manfred Morari, Ioannis G. Kevrekidis, Mahyar Fazlyab

**Abstract**: Neural networks are known to be vulnerable to adversarial attacks, which are small, imperceptible perturbations that can significantly alter the network's output. Conversely, there may exist large, meaningful perturbations that do not affect the network's decision (excessive invariance). In our research, we investigate this latter phenomenon in two contexts: (a) discrete-time dynamical system identification, and (b) the calibration of a neural network's output to that of another network. We examine noninvertibility through the lens of mathematical optimization, where the global solution measures the ``safety" of the network predictions by their distance from the non-invertibility boundary. We formulate mixed-integer programs (MIPs) for ReLU networks and $L_p$ norms ($p=1,2,\infty$) that apply to neural network approximators of dynamical systems. We also discuss how our findings can be useful for invertibility certification in transformations between neural networks, e.g. between different levels of network pruning.

摘要: 众所周知，神经网络容易受到对抗性攻击，这种攻击是微小的、不可察觉的扰动，可以显著改变网络的输出。相反，可能存在不影响网络决策的大的、有意义的扰动(过度的不变性)。在我们的研究中，我们在两个背景下研究后一种现象：(A)离散时间动态系统辨识，和(B)神经网络输出到另一网络的校准。我们通过数学最优化的视角研究不可逆性，其中全局解通过网络预测到不可逆边界的距离来衡量网络预测的“安全性”。我们为RELU网络和$L_p$范数($p=1，2，\inty$)建立了混合整数规划(MIP)。我们还讨论了我们的结果如何用于神经网络之间的变换，例如在不同级别的网络剪枝之间的可逆性证明。



## **41. Manipulating Visually-aware Federated Recommender Systems and Its Countermeasures**

操纵视觉感知的联邦推荐系统及其对策 cs.IR

**SubmitDate**: 2023-05-16    [abs](http://arxiv.org/abs/2305.08183v2) [paper-pdf](http://arxiv.org/pdf/2305.08183v2)

**Authors**: Wei Yuan, Shilong Yuan, Chaoqun Yang, Quoc Viet Hung Nguyen, Hongzhi Yin

**Abstract**: Federated recommender systems (FedRecs) have been widely explored recently due to their ability to protect user data privacy. In FedRecs, a central server collaboratively learns recommendation models by sharing model public parameters with clients, thereby offering a privacy-preserving solution. Unfortunately, the exposure of model parameters leaves a backdoor for adversaries to manipulate FedRecs. Existing works about FedRec security already reveal that items can easily be promoted by malicious users via model poisoning attacks, but all of them mainly focus on FedRecs with only collaborative information (i.e., user-item interactions). We argue that these attacks are effective because of the data sparsity of collaborative signals. In practice, auxiliary information, such as products' visual descriptions, is used to alleviate collaborative filtering data's sparsity. Therefore, when incorporating visual information in FedRecs, all existing model poisoning attacks' effectiveness becomes questionable. In this paper, we conduct extensive experiments to verify that incorporating visual information can beat existing state-of-the-art attacks in reasonable settings. However, since visual information is usually provided by external sources, simply including it will create new security problems. Specifically, we propose a new kind of poisoning attack for visually-aware FedRecs, namely image poisoning attacks, where adversaries can gradually modify the uploaded image to manipulate item ranks during FedRecs' training process. Furthermore, we reveal that the potential collaboration between image poisoning attacks and model poisoning attacks will make visually-aware FedRecs more vulnerable to being manipulated. To safely use visual information, we employ a diffusion model in visually-aware FedRecs to purify each uploaded image and detect the adversarial images.

摘要: 联邦推荐系统(FedRecs)由于具有保护用户数据隐私的能力，近年来得到了广泛的研究。在FedRecs中，中央服务器通过与客户共享模型公共参数来协作学习推荐模型，从而提供隐私保护解决方案。不幸的是，模型参数的曝光为对手操纵FedRecs留下了后门。已有的关于FedRec安全的研究已经表明，恶意用户很容易通过模型中毒攻击来推销物品，但这些研究主要集中在只有协作信息的FedRecs上(即用户与物品的交互)。我们认为，由于协同信号的数据稀疏性，这些攻击是有效的。在实际应用中，产品的视觉描述等辅助信息被用来缓解协同过滤数据的稀疏性。因此，当在FedRecs中加入视觉信息时，所有现有的模型中毒攻击的有效性都会受到质疑。在本文中，我们进行了大量的实验，以验证在合理的设置下，结合视觉信息可以抵抗现有的最先进的攻击。然而，由于可视信息通常由外部来源提供，简单地将其包括在内将会产生新的安全问题。具体地说，我们提出了一种新的针对视觉感知FedRecs的中毒攻击，即图像中毒攻击，在FedRecs的训练过程中，攻击者可以逐渐修改上传的图像来操纵物品等级。此外，我们揭示了图像中毒攻击和模型中毒攻击之间的潜在合作将使视觉感知的FedRecs更容易被操纵。为了安全地使用视觉信息，我们在视觉感知的FedRecs中使用了扩散模型来净化每一张上传的图像并检测出恶意图像。



## **42. A theoretical basis for Blockchain Extractable Value**

区块链可提取价值的理论基础 cs.CR

**SubmitDate**: 2023-05-16    [abs](http://arxiv.org/abs/2302.02154v3) [paper-pdf](http://arxiv.org/pdf/2302.02154v3)

**Authors**: Massimo Bartoletti, Roberto Zunino

**Abstract**: Extractable Value refers to a wide class of economic attacks to public blockchains, where adversaries with the power to reorder, drop or insert transactions in a block can "extract" value from smart contracts. Empirical research has shown that mainstream protocols, like e.g. decentralized exchanges, are massively targeted by these attacks, with detrimental effects on their users and on the blockchain network. Despite the growing impact of these attacks in the real world, theoretical foundations are still missing. We propose a formal theory of Extractable Value, based on a general, abstract model of blockchains and smart contracts. Our theory is the basis for proofs of security against Extractable Value attacks.

摘要: 可提取价值指的是对公共区块链的一大类经济攻击，在这些攻击中，有能力在区块中重新排序、丢弃或插入交易的对手可以从智能合约中“提取”价值。经验研究表明，主流协议，如分散交换，是这些攻击的大规模目标，对其用户和区块链网络造成有害影响。尽管这些袭击在现实世界中的影响越来越大，但理论基础仍然缺乏。基于区块链和智能合约的一般抽象模型，我们提出了可提取价值的形式理论。我们的理论是针对可提取值攻击的安全性证明的基础。



## **43. Exploring the Connection between Robust and Generative Models**

探索健壮性模型和生成性模型之间的联系 cs.LG

technical report, 6 pages, 6 figures

**SubmitDate**: 2023-05-16    [abs](http://arxiv.org/abs/2304.04033v3) [paper-pdf](http://arxiv.org/pdf/2304.04033v3)

**Authors**: Senad Beadini, Iacopo Masi

**Abstract**: We offer a study that connects robust discriminative classifiers trained with adversarial training (AT) with generative modeling in the form of Energy-based Models (EBM). We do so by decomposing the loss of a discriminative classifier and showing that the discriminative model is also aware of the input data density. Though a common assumption is that adversarial points leave the manifold of the input data, our study finds out that, surprisingly, untargeted adversarial points in the input space are very likely under the generative model hidden inside the discriminative classifier -- have low energy in the EBM. We present two evidence: untargeted attacks are even more likely than the natural data and their likelihood increases as the attack strength increases. This allows us to easily detect them and craft a novel attack called High-Energy PGD that fools the classifier yet has energy similar to the data set.

摘要: 我们提供了一项研究，将经过对抗性训练(AT)训练的稳健区分分类器与基于能量的模型(EBM)形式的生成性建模相结合。我们通过分解判别分类器的损失来做到这一点，并表明判别模型也知道输入数据的密度。虽然一个普遍的假设是敌对点离开了输入数据的流形，但我们的研究发现，令人惊讶的是，在隐藏在判别分类器中的生成模型下，输入空间中的非目标对抗性点很可能在EBM中具有低能量。我们提出了两个证据：非目标攻击的可能性甚至比自然数据更高，并且随着攻击强度的增加，它们的可能性也会增加。这使我们能够轻松地检测到它们，并创建一种名为高能PGD的新型攻击，它愚弄了分类器，但具有与数据集相似的能量。



## **44. Unlearnable Examples Give a False Sense of Security: Piercing through Unexploitable Data with Learnable Examples**

无法学习的例子给人一种错误的安全感：用可学习的例子穿透不可利用的数据 cs.LG

**SubmitDate**: 2023-05-16    [abs](http://arxiv.org/abs/2305.09241v1) [paper-pdf](http://arxiv.org/pdf/2305.09241v1)

**Authors**: Wan Jiang, Yunfeng Diao, He Wang, Jianxin Sun, Meng Wang, Richang Hong

**Abstract**: Safeguarding data from unauthorized exploitation is vital for privacy and security, especially in recent rampant research in security breach such as adversarial/membership attacks. To this end, \textit{unlearnable examples} (UEs) have been recently proposed as a compelling protection, by adding imperceptible perturbation to data so that models trained on them cannot classify them accurately on original clean distribution. Unfortunately, we find UEs provide a false sense of security, because they cannot stop unauthorized users from utilizing other unprotected data to remove the protection, by turning unlearnable data into learnable again. Motivated by this observation, we formally define a new threat by introducing \textit{learnable unauthorized examples} (LEs) which are UEs with their protection removed. The core of this approach is a novel purification process that projects UEs onto the manifold of LEs. This is realized by a new joint-conditional diffusion model which denoises UEs conditioned on the pixel and perceptual similarity between UEs and LEs. Extensive experiments demonstrate that LE delivers state-of-the-art countering performance against both supervised UEs and unsupervised UEs in various scenarios, which is the first generalizable countermeasure to UEs across supervised learning and unsupervised learning.

摘要: 保护数据不被未经授权的利用对隐私和安全至关重要，特别是在最近对安全漏洞的猖獗研究中，例如对抗性/成员攻击。为此，最近提出了不可学习的例子(UE)作为一种强制保护，通过向数据添加不可察觉的扰动，使得训练在这些数据上的模型不能根据原始的干净分布对它们进行准确的分类。不幸的是，我们发现UE提供了一种错误的安全感，因为它们无法阻止未经授权的用户利用其他不受保护的数据来取消保护，方法是将无法学习的数据再次变为可学习的数据。受此观察的启发，我们正式定义了一种新的威胁，引入了去除了保护的可学习未经授权示例(LES)。这种方法的核心是一种新颖的净化过程，将UE投射到LES的流形上。这是通过一种新的联合条件扩散模型来实现的，该模型根据UE和LES之间的像素和感知相似性来对UE进行去噪。大量的实验表明，在不同的场景下，LE对监督UE和非监督UE都提供了最先进的对抗性能，这是针对监督学习和非监督学习的UE的第一个可推广的对策。



## **45. Iterative Adversarial Attack on Image-guided Story Ending Generation**

图像导引故事结尾生成的迭代对抗性攻击 cs.CV

**SubmitDate**: 2023-05-16    [abs](http://arxiv.org/abs/2305.13208v1) [paper-pdf](http://arxiv.org/pdf/2305.13208v1)

**Authors**: Youze Wang, Wenbo Hu, Richang Hong

**Abstract**: Multimodal learning involves developing models that can integrate information from various sources like images and texts. In this field, multimodal text generation is a crucial aspect that involves processing data from multiple modalities and outputting text. The image-guided story ending generation (IgSEG) is a particularly significant task, targeting on an understanding of complex relationships between text and image data with a complete story text ending. Unfortunately, deep neural networks, which are the backbone of recent IgSEG models, are vulnerable to adversarial samples. Current adversarial attack methods mainly focus on single-modality data and do not analyze adversarial attacks for multimodal text generation tasks that use cross-modal information. To this end, we propose an iterative adversarial attack method (Iterative-attack) that fuses image and text modality attacks, allowing for an attack search for adversarial text and image in an more effective iterative way. Experimental results demonstrate that the proposed method outperforms existing single-modal and non-iterative multimodal attack methods, indicating the potential for improving the adversarial robustness of multimodal text generation models, such as multimodal machine translation, multimodal question answering, etc.

摘要: 多模式学习涉及开发能够整合来自图像和文本等各种来源的信息的模型。在该领域中，多模式文本生成是一个关键的方面，它涉及处理来自多个模式的数据和输出文本。图像引导的故事结尾生成(IGSEG)是一项特别重要的任务，其目标是理解文本和图像数据之间的复杂关系，并最终生成完整的故事文本。不幸的是，深度神经网络是最近的IGSEG模型的支柱，很容易受到敌意样本的影响。目前的对抗性攻击方法主要集中在单通道数据上，没有对使用跨通道信息的多通道文本生成任务进行对抗性攻击分析。为此，我们提出了一种融合图像和文本通道攻击的迭代对抗性攻击方法(Iterative-Attack)，允许以更有效的迭代方式对对抗性文本和图像进行攻击搜索。实验结果表明，该方法优于已有的单模式和非迭代多模式攻击方法，具有提高多模式文本生成模型对抗性的潜力，如多模式机器翻译、多模式问答等。



## **46. Ortho-ODE: Enhancing Robustness and of Neural ODEs against Adversarial Attacks**

正交法：增强神经网络对敌方攻击的稳健性和稳健性 cs.LG

Final project paper

**SubmitDate**: 2023-05-16    [abs](http://arxiv.org/abs/2305.09179v1) [paper-pdf](http://arxiv.org/pdf/2305.09179v1)

**Authors**: Vishal Purohit

**Abstract**: Neural Ordinary Differential Equations (NODEs) probed the usage of numerical solvers to solve the differential equation characterized by a Neural Network (NN), therefore initiating a new paradigm of deep learning models with infinite depth. NODEs were designed to tackle the irregular time series problem. However, NODEs have demonstrated robustness against various noises and adversarial attacks. This paper is about the natural robustness of NODEs and examines the cause behind such surprising behaviour. We show that by controlling the Lipschitz constant of the ODE dynamics the robustness can be significantly improved. We derive our approach from Grownwall's inequality. Further, we draw parallels between contractivity theory and Grownwall's inequality. Experimentally we corroborate the enhanced robustness on numerous datasets - MNIST, CIFAR-10, and CIFAR 100. We also present the impact of adaptive and non-adaptive solvers on the robustness of NODEs.

摘要: 神经常微分方程组(节点)探索了用数值求解器来求解以神经网络(NN)为特征的微分方程，从而开创了一种无限深度深度学习模型的新范式。节点的设计是为了处理不规则的时间序列问题。然而，节点对各种噪声和敌意攻击表现出了健壮性。这篇论文是关于节点的自然健壮性的，并研究了这种令人惊讶的行为背后的原因。结果表明，通过控制常微分方程组的Lipschitz常数，可以显着提高系统的鲁棒性。我们的方法源于Grownwall的不等式性。此外，我们还比较了可逆性理论和Grownwall不等式之间的异同。在实验上，我们在许多数据集--MNIST、CIFAR-10和CIFAR 100上证实了增强的稳健性。我们还给出了自适应和非自适应求解器对节点健壮性的影响。



## **47. Run-Off Election: Improved Provable Defense against Data Poisoning Attacks**

决选：改进了针对数据中毒攻击的可证明防御 cs.LG

Accepted to ICML 2023

**SubmitDate**: 2023-05-16    [abs](http://arxiv.org/abs/2302.02300v3) [paper-pdf](http://arxiv.org/pdf/2302.02300v3)

**Authors**: Keivan Rezaei, Kiarash Banihashem, Atoosa Chegini, Soheil Feizi

**Abstract**: In data poisoning attacks, an adversary tries to change a model's prediction by adding, modifying, or removing samples in the training data. Recently, ensemble-based approaches for obtaining provable defenses against data poisoning have been proposed where predictions are done by taking a majority vote across multiple base models. In this work, we show that merely considering the majority vote in ensemble defenses is wasteful as it does not effectively utilize available information in the logits layers of the base models. Instead, we propose Run-Off Election (ROE), a novel aggregation method based on a two-round election across the base models: In the first round, models vote for their preferred class and then a second, Run-Off election is held between the top two classes in the first round. Based on this approach, we propose DPA+ROE and FA+ROE defense methods based on Deep Partition Aggregation (DPA) and Finite Aggregation (FA) approaches from prior work. We evaluate our methods on MNIST, CIFAR-10, and GTSRB and obtain improvements in certified accuracy by up to 3%-4%. Also, by applying ROE on a boosted version of DPA, we gain improvements around 12%-27% comparing to the current state-of-the-art, establishing a new state-of-the-art in (pointwise) certified robustness against data poisoning. In many cases, our approach outperforms the state-of-the-art, even when using 32 times less computational power.

摘要: 在数据中毒攻击中，对手试图通过添加、修改或删除训练数据中的样本来更改模型的预测。最近，已经提出了基于集成的方法来获得针对数据中毒的可证明防御，其中预测是通过在多个基础模型上获得多数票来完成的。在这项工作中，我们表明，仅仅在集成防御中考虑多数投票是浪费的，因为它没有有效地利用基本模型的Logits层中的可用信息。相反，我们提出了决选选举(ROE)，这是一种基于基础模型之间的两轮选举的新型聚合方法：在第一轮中，模型投票选择他们喜欢的类，然后在第一轮中前两个类之间举行第二次决选。在此基础上，提出了基于深度划分聚集(DPA)和有限聚集(FA)的DPA+ROE和FA+ROE防御方法。我们在MNIST、CIFAR-10和GTSRB上对我们的方法进行了评估，并在认证的准确性方面获得了高达3%-4%的改进。此外，通过在增强版本的DPA上应用ROE，与当前最先进的版本相比，我们获得了约12%-27%的改进，从而建立了针对数据中毒的(按点)经认证的新的最先进的健壮性。在许多情况下，我们的方法优于最先进的方法，即使在使用32倍的计算能力时也是如此。



## **48. Training Neural Networks without Backpropagation: A Deeper Dive into the Likelihood Ratio Method**

无反向传播训练神经网络：对似然比方法的深入探讨 cs.LG

**SubmitDate**: 2023-05-15    [abs](http://arxiv.org/abs/2305.08960v1) [paper-pdf](http://arxiv.org/pdf/2305.08960v1)

**Authors**: Jinyang Jiang, Zeliang Zhang, Chenliang Xu, Zhaofei Yu, Yijie Peng

**Abstract**: Backpropagation (BP) is the most important gradient estimation method for training neural networks in deep learning. However, the literature shows that neural networks trained by BP are vulnerable to adversarial attacks. We develop the likelihood ratio (LR) method, a new gradient estimation method, for training a broad range of neural network architectures, including convolutional neural networks, recurrent neural networks, graph neural networks, and spiking neural networks, without recursive gradient computation. We propose three methods to efficiently reduce the variance of the gradient estimation in the neural network training process. Our experiments yield numerical results for training different neural networks on several datasets. All results demonstrate that the LR method is effective for training various neural networks and significantly improves the robustness of the neural networks under adversarial attacks relative to the BP method.

摘要: 反向传播(BP)是深度学习中训练神经网络最重要的梯度估计方法。然而，文献表明，BP训练的神经网络很容易受到对手的攻击。我们发展了一种新的梯度估计方法-似然比方法，用于训练包括卷积神经网络、递归神经网络、图神经网络和尖峰神经网络在内的广泛的神经网络结构，而不需要递归梯度计算。在神经网络训练过程中，我们提出了三种有效降低梯度估计方差的方法。我们的实验给出了在几个数据集上训练不同神经网络的数值结果。结果表明，与BP方法相比，LR方法对训练各种神经网络是有效的，并且显著提高了神经网络在对抗攻击下的鲁棒性。



## **49. Attacking Perceptual Similarity Metrics**

攻击感知相似性度量 cs.CV

TMLR 2023 (Featured Certification). Code is available at  https://tinyurl.com/attackingpsm

**SubmitDate**: 2023-05-15    [abs](http://arxiv.org/abs/2305.08840v1) [paper-pdf](http://arxiv.org/pdf/2305.08840v1)

**Authors**: Abhijay Ghildyal, Feng Liu

**Abstract**: Perceptual similarity metrics have progressively become more correlated with human judgments on perceptual similarity; however, despite recent advances, the addition of an imperceptible distortion can still compromise these metrics. In our study, we systematically examine the robustness of these metrics to imperceptible adversarial perturbations. Following the two-alternative forced-choice experimental design with two distorted images and one reference image, we perturb the distorted image closer to the reference via an adversarial attack until the metric flips its judgment. We first show that all metrics in our study are susceptible to perturbations generated via common adversarial attacks such as FGSM, PGD, and the One-pixel attack. Next, we attack the widely adopted LPIPS metric using spatial-transformation-based adversarial perturbations (stAdv) in a white-box setting to craft adversarial examples that can effectively transfer to other similarity metrics in a black-box setting. We also combine the spatial attack stAdv with PGD ($\ell_\infty$-bounded) attack to increase transferability and use these adversarial examples to benchmark the robustness of both traditional and recently developed metrics. Our benchmark provides a good starting point for discussion and further research on the robustness of metrics to imperceptible adversarial perturbations.

摘要: 知觉相似性指标已逐渐与人类对知觉相似性的判断更加相关；然而，尽管最近取得了进展，添加了不可察觉的失真仍然可能损害这些指标。在我们的研究中，我们系统地检查了这些度量对不可察觉的对抗性扰动的稳健性。在两个失真图像和一个参考图像的两种选择强迫选择实验设计之后，我们通过对抗性攻击使失真图像更接近参考图像，直到度量颠倒其判断。我们首先表明，我们研究中的所有指标都容易受到常见的对抗性攻击(如FGSM、PGD和单像素攻击)产生的扰动的影响。接下来，我们在白盒环境中使用基于空间变换的对抗性扰动(StAdv)来攻击广泛采用的LPIPS度量，以创建可以有效地转换到黑盒环境中的其他相似性度量的对抗性示例。我们还将空间攻击stAdv与pgd($\ell_\inty$-bound)攻击相结合，以增加可转移性，并使用这些对抗性示例来对传统度量和最近开发的度量的健壮性进行基准测试。我们的基准为讨论和进一步研究度量对不可察觉的对抗性扰动的健壮性提供了一个很好的起点。



## **50. Defending Against Misinformation Attacks in Open-Domain Question Answering**

开放领域答疑中防误报攻击的研究 cs.CL

**SubmitDate**: 2023-05-15    [abs](http://arxiv.org/abs/2212.10002v2) [paper-pdf](http://arxiv.org/pdf/2212.10002v2)

**Authors**: Orion Weller, Aleem Khan, Nathaniel Weir, Dawn Lawrie, Benjamin Van Durme

**Abstract**: Recent work in open-domain question answering (ODQA) has shown that adversarial poisoning of the search collection can cause large drops in accuracy for production systems. However, little to no work has proposed methods to defend against these attacks. To do so, we rely on the intuition that redundant information often exists in large corpora. To find it, we introduce a method that uses query augmentation to search for a diverse set of passages that could answer the original question but are less likely to have been poisoned. We integrate these new passages into the model through the design of a novel confidence method, comparing the predicted answer to its appearance in the retrieved contexts (what we call \textit{Confidence from Answer Redundancy}, i.e. CAR). Together these methods allow for a simple but effective way to defend against poisoning attacks that provides gains of nearly 20\% exact match across varying levels of data poisoning/knowledge conflicts.

摘要: 最近在开放领域问答(ODQA)方面的研究表明，搜索集合的敌意中毒会导致产生式系统的准确率大幅下降。然而，几乎没有工作提出了防御这些攻击的方法。要做到这一点，我们依赖于这样一种直觉，即大型语料库中往往存在冗余信息。为了找到它，我们引入了一种方法，使用查询增强来搜索一组不同的段落，这些段落可以回答原始问题，但不太可能被毒化。我们通过设计一种新的置信度方法将这些新的段落集成到模型中，将预测的答案与其在检索到的上下文中的表现进行比较(我们称其为来自答案冗余的置信度)，即CAR。这些方法结合在一起，提供了一种简单但有效的方法来防御中毒攻击，在不同级别的数据中毒/知识冲突中提供了近20%的精确匹配。



