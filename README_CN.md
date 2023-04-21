# Latest Adversarial Attack Papers
**update at 2023-04-21 11:46:34**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. More is Better (Mostly): On the Backdoor Attacks in Federated Graph Neural Networks**

越多越好(多数)：联邦图神经网络中的后门攻击 cs.CR

15 pages, 13 figures

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2202.03195v5) [paper-pdf](http://arxiv.org/pdf/2202.03195v5)

**Authors**: Jing Xu, Rui Wang, Stefanos Koffas, Kaitai Liang, Stjepan Picek

**Abstract**: Graph Neural Networks (GNNs) are a class of deep learning-based methods for processing graph domain information. GNNs have recently become a widely used graph analysis method due to their superior ability to learn representations for complex graph data. However, due to privacy concerns and regulation restrictions, centralized GNNs can be difficult to apply to data-sensitive scenarios. Federated learning (FL) is an emerging technology developed for privacy-preserving settings when several parties need to train a shared global model collaboratively. Although several research works have applied FL to train GNNs (Federated GNNs), there is no research on their robustness to backdoor attacks.   This paper bridges this gap by conducting two types of backdoor attacks in Federated GNNs: centralized backdoor attacks (CBA) and distributed backdoor attacks (DBA). Our experiments show that the DBA attack success rate is higher than CBA in almost all evaluated cases. For CBA, the attack success rate of all local triggers is similar to the global trigger even if the training set of the adversarial party is embedded with the global trigger. To further explore the properties of two backdoor attacks in Federated GNNs, we evaluate the attack performance for a different number of clients, trigger sizes, poisoning intensities, and trigger densities. Moreover, we explore the robustness of DBA and CBA against one defense. We find that both attacks are robust against the investigated defense, necessitating the need to consider backdoor attacks in Federated GNNs as a novel threat that requires custom defenses.

摘要: 图神经网络是一类基于深度学习的图域信息处理方法。由于其优越的学习复杂图形数据表示的能力，GNN最近已成为一种广泛使用的图形分析方法。然而，由于隐私问题和监管限制，集中式GNN可能很难适用于数据敏感的情况。联合学习(FL)是一种新兴的技术，是为保护隐私而开发的，当多个参与方需要协作训练共享的全球模型时。虽然一些研究工作已经将FL用于训练GNN(Federated GNN)，但还没有关于其对后门攻击的健壮性的研究。本文通过在联邦GNN中实施两种类型的后门攻击来弥合这一差距：集中式后门攻击(CBA)和分布式后门攻击(DBA)。我们的实验表明，DBA攻击的成功率几乎在所有评估案例中都高于CBA。对于CBA，即使敌方的训练集嵌入了全局触发器，所有局部触发器的攻击成功率也与全局触发器相似。为了进一步探索联邦GNN中两种后门攻击的特性，我们评估了不同客户端数量、触发器大小、中毒强度和触发器密度下的攻击性能。此外，我们还探讨了DBA和CBA对一种防御的健壮性。我们发现，这两种攻击对所调查的防御都是健壮的，因此有必要将联邦GNN中的后门攻击视为一种需要自定义防御的新威胁。



## **2. Certified Adversarial Robustness Within Multiple Perturbation Bounds**

多扰动界下的认证对抗稳健性 cs.LG

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2304.10446v1) [paper-pdf](http://arxiv.org/pdf/2304.10446v1)

**Authors**: Soumalya Nandi, Sravanti Addepalli, Harsh Rangwani, R. Venkatesh Babu

**Abstract**: Randomized smoothing (RS) is a well known certified defense against adversarial attacks, which creates a smoothed classifier by predicting the most likely class under random noise perturbations of inputs during inference. While initial work focused on robustness to $\ell_2$ norm perturbations using noise sampled from a Gaussian distribution, subsequent works have shown that different noise distributions can result in robustness to other $\ell_p$ norm bounds as well. In general, a specific noise distribution is optimal for defending against a given $\ell_p$ norm based attack. In this work, we aim to improve the certified adversarial robustness against multiple perturbation bounds simultaneously. Towards this, we firstly present a novel \textit{certification scheme}, that effectively combines the certificates obtained using different noise distributions to obtain optimal results against multiple perturbation bounds. We further propose a novel \textit{training noise distribution} along with a \textit{regularized training scheme} to improve the certification within both $\ell_1$ and $\ell_2$ perturbation norms simultaneously. Contrary to prior works, we compare the certified robustness of different training algorithms across the same natural (clean) accuracy, rather than across fixed noise levels used for training and certification. We also empirically invalidate the argument that training and certifying the classifier with the same amount of noise gives the best results. The proposed approach achieves improvements on the ACR (Average Certified Radius) metric across both $\ell_1$ and $\ell_2$ perturbation bounds.

摘要: 随机平滑(RS)是一种著名的对抗攻击的认证防御方法，它通过在推理过程中输入的随机噪声扰动下预测最可能的类别来创建平滑的分类器。虽然最初的工作集中于使用从高斯分布采样的噪声对$\ell_2$范数扰动的鲁棒性，但随后的工作表明，不同的噪声分布也可以导致对其他$\ell_p$范数界的鲁棒性。一般来说，特定的噪声分布对于防御给定的基于$\ell_p$范数的攻击是最优的。在这项工作中，我们的目标是同时提高认证对手对多个扰动界的稳健性。为此，我们首先提出了一种新的认证方案，该方案有效地结合了使用不同噪声分布获得的证书，从而在多个扰动界下获得最优结果。我们进一步提出了一种新的训练噪声分布和一种正则化训练方案，以同时改进在$1和$2扰动范数下的证明。与以前的工作相反，我们在相同的自然(干净)精度范围内比较不同训练算法的认证稳健性，而不是在用于训练和认证的固定噪声水平上进行比较。我们还从经验上证明了训练和认证具有相同噪声量的分类器可以得到最好的结果的论点。所提出的方法在$\ell_1$和$\ell_2$摄动界上实现了对ACR(平均认证半径)度量的改进。



## **3. An Analysis of the Completion Time of the BB84 Protocol**

对BB84议定书完成时间的分析 cs.PF

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2304.10218v1) [paper-pdf](http://arxiv.org/pdf/2304.10218v1)

**Authors**: Sounak Kar, Jean-Yves Le Boudec

**Abstract**: The BB84 QKD protocol is based on the idea that the sender and the receiver can reconcile a certain fraction of the teleported qubits to detect eavesdropping or noise and decode the rest to use as a private key. Under the present hardware infrastructure, decoherence of quantum states poses a significant challenge to performing perfect or efficient teleportation, meaning that a teleportation-based protocol must be run multiple times to observe success. Thus, performance analyses of such protocols usually consider the completion time, i.e., the time until success, rather than the duration of a single attempt. Moreover, due to decoherence, the success of an attempt is in general dependent on the duration of individual phases of that attempt, as quantum states must wait in memory while the success or failure of a generation phase is communicated to the relevant parties. In this work, we do a performance analysis of the completion time of the BB84 protocol in a setting where the sender and the receiver are connected via a single quantum repeater and the only quantum channel between them does not see any adversarial attack. Assuming certain distributional forms for the generation and communication phases of teleportation, we provide a method to compute the MGF of the completion time and subsequently derive an estimate of the CDF and a bound on the tail probability. This result helps us gauge the (tail) behaviour of the completion time in terms of the parameters characterising the elementary phases of teleportation, without having to run the protocol multiple times. We also provide an efficient simulation scheme to generate the completion time, which relies on expressing the completion time in terms of aggregated teleportation times. We numerically compare our approach with a full-scale simulation and observe good agreement between them.

摘要: BB84量子密钥分发协议基于这样的思想，即发送者和接收者可以协调传送的量子比特的特定部分，以检测窃听或噪声，并解码其余的量子比特作为私钥。在目前的硬件基础设施下，量子态的退相干对执行完美或有效的隐形传态构成了巨大的挑战，这意味着基于隐形传态的协议必须多次运行才能观察到成功。因此，此类协议的性能分析通常考虑完成时间，即成功之前的时间，而不是单次尝试的持续时间。此外，由于退相干，尝试的成功通常取决于尝试的各个阶段的持续时间，因为当生成阶段的成功或失败被传递给相关方时，量子态必须在存储器中等待。在这项工作中，我们对BB84协议的完成时间进行了性能分析，在发送方和接收方通过单个量子中继器连接并且它们之间唯一的量子信道没有任何敌意攻击的情况下。假设隐形传态的产生和通信阶段有一定的分布形式，我们提供了一种计算完成时间的MGF的方法，并由此得到了CDF的估计和尾概率的界。这一结果帮助我们根据表征隐形传态的基本阶段的参数来衡量完成时间的(尾部)行为，而不必多次运行协议。我们还提供了一种高效的仿真方案来生成完成时间，该方案依赖于将完成时间表示为聚合隐形传态时间。我们将我们的方法与全尺寸模拟进行了数值比较，并观察到它们之间有很好的一致性。



## **4. Quantum-secure message authentication via blind-unforgeability**

基于盲不可伪造性的量子安全消息认证 quant-ph

37 pages, v4: Erratum added. We removed a result that had an error in  its proof

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/1803.03761v4) [paper-pdf](http://arxiv.org/pdf/1803.03761v4)

**Authors**: Gorjan Alagic, Christian Majenz, Alexander Russell, Fang Song

**Abstract**: Formulating and designing authentication of classical messages in the presence of adversaries with quantum query access has been a longstanding challenge, as the familiar classical notions of unforgeability do not directly translate into meaningful notions in the quantum setting. A particular difficulty is how to fairly capture the notion of "predicting an unqueried value" when the adversary can query in quantum superposition.   We propose a natural definition of unforgeability against quantum adversaries called blind unforgeability. This notion defines a function to be predictable if there exists an adversary who can use "partially blinded" oracle access to predict values in the blinded region. We support the proposal with a number of technical results. We begin by establishing that the notion coincides with EUF-CMA in the classical setting and go on to demonstrate that the notion is satisfied by a number of simple guiding examples, such as random functions and quantum-query-secure pseudorandom functions. We then show the suitability of blind unforgeability for supporting canonical constructions and reductions. We prove that the "hash-and-MAC" paradigm and the Lamport one-time digital signature scheme are indeed unforgeable according to the definition. To support our analysis, we additionally define and study a new variety of quantum-secure hash functions called Bernoulli-preserving.   Finally, we demonstrate that blind unforgeability is stronger than a previous definition of Boneh and Zhandry [EUROCRYPT '13, CRYPTO '13] in the sense that we can construct an explicit function family which is forgeable by an attack that is recognized by blind-unforgeability, yet satisfies the definition by Boneh and Zhandry.

摘要: 在具有量子查询访问的攻击者在场的情况下，制定和设计经典消息的认证一直是一个长期存在的挑战，因为熟悉的不可伪造性的经典概念不能直接转化为在量子环境中有意义的概念。一个特别的困难是，当对手可以在量子叠加中进行查询时，如何公平地捕捉到“预测一个未被质疑的值”的概念。针对量子对手，我们提出了不可伪造性的自然定义，称为盲不可伪造性。这个概念定义了一个函数是可预测的，如果存在一个对手，该对手可以使用“部分盲目的”先知访问来预测盲区中的值。我们用一系列技术成果支持这项提议。我们首先建立了这个概念在经典设置下与EUF-CMA重合，然后通过一些简单的指导性例子，例如随机函数和量子查询安全伪随机函数，证明了这个概念是满足的。然后，我们证明了盲不可伪造性对于支持规范构造和约简的适用性。根据定义，我们证明了Hash-and-MAC范例和Lamport一次性数字签名方案确实是不可伪造的。为了支持我们的分析，我们还定义和研究了一种新的量子安全散列函数，称为伯努利保持。最后，我们证明了盲不可伪造性比Boneh和Zhandry[Eurocrypt‘13，Crypto’13]的定义更强，因为我们可以构造一个显式函数族，该函数族可以被盲不可伪造性识别的攻击伪造，但满足Boneh和Zhandry的定义。



## **5. Diversifying the High-level Features for better Adversarial Transferability**

使高级功能多样化，以实现更好的对手可转换性 cs.CV

15 pages

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2304.10136v1) [paper-pdf](http://arxiv.org/pdf/2304.10136v1)

**Authors**: Zhiyuan Wang, Zeliang Zhang, Siyuan Liang, Xiaosen Wang

**Abstract**: Given the great threat of adversarial attacks against Deep Neural Networks (DNNs), numerous works have been proposed to boost transferability to attack real-world applications. However, existing attacks often utilize advanced gradient calculation or input transformation but ignore the white-box model. Inspired by the fact that DNNs are over-parameterized for superior performance, we propose diversifying the high-level features (DHF) for more transferable adversarial examples. In particular, DHF perturbs the high-level features by randomly transforming the high-level features and mixing them with the feature of benign samples when calculating the gradient at each iteration. Due to the redundancy of parameters, such transformation does not affect the classification performance but helps identify the invariant features across different models, leading to much better transferability. Empirical evaluations on ImageNet dataset show that DHF could effectively improve the transferability of existing momentum-based attacks. Incorporated into the input transformation-based attacks, DHF generates more transferable adversarial examples and outperforms the baselines with a clear margin when attacking several defense models, showing its generalization to various attacks and high effectiveness for boosting transferability.

摘要: 鉴于针对深度神经网络的敌意攻击的巨大威胁，人们已经提出了许多工作来提高可转移性以攻击真实世界的应用。然而，现有的攻击往往利用先进的梯度计算或输入变换，而忽略了白盒模型。受DNN过度参数化以获得卓越性能这一事实的启发，我们建议将高级特征(DHF)多样化，以获得更多可转移的对抗性示例。特别是，DHF在每次迭代计算梯度时，通过随机变换高层特征并将其与良性样本的特征混合来扰动高层特征。由于参数的冗余性，这种变换不会影响分类性能，但有助于识别不同模型之间的不变特征，从而产生更好的可移植性。在ImageNet数据集上的实验评估表明，DHF能够有效地提高现有动量攻击的可转移性。DHF结合到基于输入变换的攻击中，生成了更多可转移的对抗性实例，在攻击多种防御模型时以明显的优势超过基线，显示了其对各种攻击的通用性和提高可转移性的高效性。



## **6. Towards the Universal Defense for Query-Based Audio Adversarial Attacks**

面向基于查询的音频攻击的通用防御 eess.AS

Submitted to Cybersecurity journal

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2304.10088v1) [paper-pdf](http://arxiv.org/pdf/2304.10088v1)

**Authors**: Feng Guo, Zheng Sun, Yuxuan Chen, Lei Ju

**Abstract**: Recently, studies show that deep learning-based automatic speech recognition (ASR) systems are vulnerable to adversarial examples (AEs), which add a small amount of noise to the original audio examples. These AE attacks pose new challenges to deep learning security and have raised significant concerns about deploying ASR systems and devices. The existing defense methods are either limited in application or only defend on results, but not on process. In this work, we propose a novel method to infer the adversary intent and discover audio adversarial examples based on the AEs generation process. The insight of this method is based on the observation: many existing audio AE attacks utilize query-based methods, which means the adversary must send continuous and similar queries to target ASR models during the audio AE generation process. Inspired by this observation, We propose a memory mechanism by adopting audio fingerprint technology to analyze the similarity of the current query with a certain length of memory query. Thus, we can identify when a sequence of queries appears to be suspectable to generate audio AEs. Through extensive evaluation on four state-of-the-art audio AE attacks, we demonstrate that on average our defense identify the adversary intent with over 90% accuracy. With careful regard for robustness evaluations, we also analyze our proposed defense and its strength to withstand two adaptive attacks. Finally, our scheme is available out-of-the-box and directly compatible with any ensemble of ASR defense models to uncover audio AE attacks effectively without model retraining.

摘要: 最近的研究表明，基于深度学习的自动语音识别(ASR)系统容易受到对抗性样本(AES)的攻击，这些样本会在原始音频样本中添加少量噪声。这些AE攻击对深度学习安全提出了新的挑战，并引发了对部署ASR系统和设备的重大担忧。现有的防御方法要么局限于应用，要么只针对结果进行防御，而不是针对过程进行防御。在这项工作中，我们提出了一种新的方法来推断对手的意图，并发现音频对抗性实例的基础上，AES的生成过程。这种方法的洞察力是基于观察到的：现有的许多音频AE攻击都使用基于查询的方法，这意味着在音频AE生成过程中，攻击者必须向目标ASR模型发送连续的相似查询。受此启发，我们提出了一种采用音频指纹技术的记忆机制来分析当前查询与一定长度的记忆查询的相似度。因此，我们可以识别查询序列何时看起来可疑以生成音频AE。通过对四种最先进的音频AE攻击的广泛评估，我们证明了我们的防御平均识别对手意图的准确率超过90%。在仔细考虑健壮性评估的同时，我们还分析了我们提出的防御方案及其抵抗两种自适应攻击的能力。最后，我们的方案是开箱即用的，并且直接与任何ASR防御模型集成兼容，以有效地发现音频AE攻击，而不需要重新训练模型。



## **7. A Search-Based Testing Approach for Deep Reinforcement Learning Agents**

一种基于搜索的深度强化学习代理测试方法 cs.SE

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2206.07813v3) [paper-pdf](http://arxiv.org/pdf/2206.07813v3)

**Authors**: Amirhossein Zolfagharian, Manel Abdellatif, Lionel Briand, Mojtaba Bagherzadeh, Ramesh S

**Abstract**: Deep Reinforcement Learning (DRL) algorithms have been increasingly employed during the last decade to solve various decision-making problems such as autonomous driving and robotics. However, these algorithms have faced great challenges when deployed in safety-critical environments since they often exhibit erroneous behaviors that can lead to potentially critical errors. One way to assess the safety of DRL agents is to test them to detect possible faults leading to critical failures during their execution. This raises the question of how we can efficiently test DRL policies to ensure their correctness and adherence to safety requirements. Most existing works on testing DRL agents use adversarial attacks that perturb states or actions of the agent. However, such attacks often lead to unrealistic states of the environment. Their main goal is to test the robustness of DRL agents rather than testing the compliance of agents' policies with respect to requirements. Due to the huge state space of DRL environments, the high cost of test execution, and the black-box nature of DRL algorithms, the exhaustive testing of DRL agents is impossible. In this paper, we propose a Search-based Testing Approach of Reinforcement Learning Agents (STARLA) to test the policy of a DRL agent by effectively searching for failing executions of the agent within a limited testing budget. We use machine learning models and a dedicated genetic algorithm to narrow the search towards faulty episodes. We apply STARLA on Deep-Q-Learning agents which are widely used as benchmarks and show that it significantly outperforms Random Testing by detecting more faults related to the agent's policy. We also investigate how to extract rules that characterize faulty episodes of the DRL agent using our search results. Such rules can be used to understand the conditions under which the agent fails and thus assess its deployment risks.

摘要: 在过去的十年中，深度强化学习(DRL)算法被越来越多地用于解决各种决策问题，如自动驾驶和机器人技术。然而，当这些算法部署在安全关键环境中时，它们面临着巨大的挑战，因为它们经常表现出可能导致潜在关键错误的错误行为。评估DRL代理安全性的一种方法是对其进行测试，以检测在其执行期间可能导致严重故障的故障。这提出了一个问题，即我们如何有效地测试DRL政策，以确保它们的正确性和对安全要求的遵守。大多数现有的测试DRL代理的工作使用对抗性攻击，扰乱代理的状态或动作。然而，这样的攻击往往会导致不切实际的环境状况。他们的主要目标是测试DRL代理的健壮性，而不是测试代理策略与需求的符合性。由于DRL环境的状态空间巨大、测试执行成本高以及DRL算法的黑箱性质，对DRL代理进行穷举测试是不可能的。在本文中，我们提出了一种基于搜索的强化学习代理测试方法(STARLA)，通过在有限的测试预算内有效地搜索代理的失败执行来测试DRL代理的策略。我们使用机器学习模型和专门的遗传算法将搜索范围缩小到故障剧集。我们将Starla应用于被广泛用作基准测试的Deep-Q-Learning代理上，结果表明它在检测到更多与代理策略相关的错误方面明显优于随机测试。我们还研究了如何使用搜索结果提取表征DRL代理故障情节的规则。此类规则可用于了解代理发生故障的条件，从而评估其部署风险。



## **8. Quantifying the Preferential Direction of the Model Gradient in Adversarial Training With Projected Gradient Descent**

用投影梯度下降法量化对抗性训练中模型梯度的优先方向 stat.ML

This paper was published in Pattern Recognition

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2009.04709v5) [paper-pdf](http://arxiv.org/pdf/2009.04709v5)

**Authors**: Ricardo Bigolin Lanfredi, Joyce D. Schroeder, Tolga Tasdizen

**Abstract**: Adversarial training, especially projected gradient descent (PGD), has proven to be a successful approach for improving robustness against adversarial attacks. After adversarial training, gradients of models with respect to their inputs have a preferential direction. However, the direction of alignment is not mathematically well established, making it difficult to evaluate quantitatively. We propose a novel definition of this direction as the direction of the vector pointing toward the closest point of the support of the closest inaccurate class in decision space. To evaluate the alignment with this direction after adversarial training, we apply a metric that uses generative adversarial networks to produce the smallest residual needed to change the class present in the image. We show that PGD-trained models have a higher alignment than the baseline according to our definition, that our metric presents higher alignment values than a competing metric formulation, and that enforcing this alignment increases the robustness of models.

摘要: 对抗性训练，特别是投影梯度下降(PGD)，已被证明是提高对抗攻击的稳健性的一种成功方法。经过对抗性训练后，模型相对于其输入的梯度具有优先的方向。然而，对齐的方向在数学上没有很好的确定，因此很难进行定量评估。我们提出了一种新的方向定义，即向量指向决策空间中最接近的不准确类的支持度的最近点的方向。为了在对抗性训练后评估与这一方向的一致性，我们应用了一种度量，该度量使用生成性对抗性网络来产生改变图像中存在的类别所需的最小残差。我们表明，根据我们的定义，PGD训练的模型具有比基线更高的比对，我们的指标比竞争指标公式提供了更高的比对值，并且强制执行这种比对提高了模型的稳健性。



## **9. Jedi: Entropy-based Localization and Removal of Adversarial Patches**

绝地：基于熵的敌方补丁定位与移除 cs.CR

9 pages, 11 figures. To appear in CVPR 2023

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2304.10029v1) [paper-pdf](http://arxiv.org/pdf/2304.10029v1)

**Authors**: Bilel Tarchoun, Anouar Ben Khalifa, Mohamed Ali Mahjoub, Nael Abu-Ghazaleh, Ihsen Alouani

**Abstract**: Real-world adversarial physical patches were shown to be successful in compromising state-of-the-art models in a variety of computer vision applications. Existing defenses that are based on either input gradient or features analysis have been compromised by recent GAN-based attacks that generate naturalistic patches. In this paper, we propose Jedi, a new defense against adversarial patches that is resilient to realistic patch attacks. Jedi tackles the patch localization problem from an information theory perspective; leverages two new ideas: (1) it improves the identification of potential patch regions using entropy analysis: we show that the entropy of adversarial patches is high, even in naturalistic patches; and (2) it improves the localization of adversarial patches, using an autoencoder that is able to complete patch regions from high entropy kernels. Jedi achieves high-precision adversarial patch localization, which we show is critical to successfully repair the images. Since Jedi relies on an input entropy analysis, it is model-agnostic, and can be applied on pre-trained off-the-shelf models without changes to the training or inference of the protected models. Jedi detects on average 90% of adversarial patches across different benchmarks and recovers up to 94% of successful patch attacks (Compared to 75% and 65% for LGS and Jujutsu, respectively).

摘要: 现实世界中的对抗性物理补丁被证明在各种计算机视觉应用中成功地折衷了最先进的模型。现有的基于输入梯度或特征分析的防御已经被最近基于GaN的攻击所破坏，这些攻击产生了自然主义的补丁。在这篇文章中，我们提出了一种新的防御对手补丁的绝地，它对现实的补丁攻击具有弹性。绝地从信息论的角度解决了补丁定位问题；利用了两个新的想法：(1)它利用熵分析改进了潜在补丁区域的识别：我们证明了对抗性补丁的熵很高，即使在自然补丁中也是如此；(2)它改进了对抗性补丁的定位，使用了能够从高熵内核完成补丁区域的自动编码器。绝地实现了高精度的对抗性补丁定位，我们证明这是成功修复图像的关键。由于绝地依赖于输入熵分析，它与模型无关，可以应用于预先训练的现成模型，而不需要改变受保护模型的训练或推断。绝地平均可以通过不同的基准检测90%的敌方补丁，并恢复高达94%的成功补丁攻击(相比之下，LGS和Jujutsu的这一比例分别为75%和65%)。



## **10. GREAT Score: Global Robustness Evaluation of Adversarial Perturbation using Generative Models**

高分：使用生成模型对对抗性扰动进行全局稳健性评估 cs.LG

**SubmitDate**: 2023-04-19    [abs](http://arxiv.org/abs/2304.09875v1) [paper-pdf](http://arxiv.org/pdf/2304.09875v1)

**Authors**: Li Zaitang, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Current studies on adversarial robustness mainly focus on aggregating local robustness results from a set of data samples to evaluate and rank different models. However, the local statistics may not well represent the true global robustness of the underlying unknown data distribution. To address this challenge, this paper makes the first attempt to present a new framework, called GREAT Score , for global robustness evaluation of adversarial perturbation using generative models. Formally, GREAT Score carries the physical meaning of a global statistic capturing a mean certified attack-proof perturbation level over all samples drawn from a generative model. For finite-sample evaluation, we also derive a probabilistic guarantee on the sample complexity and the difference between the sample mean and the true mean. GREAT Score has several advantages: (1) Robustness evaluations using GREAT Score are efficient and scalable to large models, by sparing the need of running adversarial attacks. In particular, we show high correlation and significantly reduced computation cost of GREAT Score when compared to the attack-based model ranking on RobustBench (Croce,et. al. 2021). (2) The use of generative models facilitates the approximation of the unknown data distribution. In our ablation study with different generative adversarial networks (GANs), we observe consistency between global robustness evaluation and the quality of GANs. (3) GREAT Score can be used for remote auditing of privacy-sensitive black-box models, as demonstrated by our robustness evaluation on several online facial recognition services.

摘要: 目前关于对抗稳健性的研究主要集中在从一组数据样本中聚集局部稳健性结果来评估和排序不同的模型。然而，局部统计可能不能很好地代表潜在未知数据分布的真实全局稳健性。为了应对这一挑战，本文首次尝试提出了一种新的框架，称为Great Score，用于利用产生式模型评估对抗扰动的全局稳健性。在形式上，高分具有全球统计的物理意义，该统计捕获来自生成模型的所有样本的平均经认证的防攻击扰动水平。对于有限样本评价，我们还得到了样本复杂度和样本均值与真均值之差的概率保证。Great Score有几个优点：(1)使用Great Score进行健壮性评估是高效的，并且可以扩展到大型模型，因为它避免了运行对抗性攻击的需要。特别是，与基于攻击的模型排名相比，我们表现出了高度的相关性和显著的降低了计算开销。艾尔2021年)。(2)生成模型的使用有利于未知数据分布的近似。在我们对不同生成对抗网络(GANS)的消融研究中，我们观察到全局健壮性评估与GANS质量之间的一致性。(3)Great Score可以用于隐私敏感的黑盒模型的远程审计，我们在几种在线人脸识别服务上的健壮性评估证明了这一点。



## **11. Experimental Certification of Quantum Transmission via Bell's Theorem**

用贝尔定理进行量子传输的实验验证 quant-ph

34 pages, 14 figures

**SubmitDate**: 2023-04-19    [abs](http://arxiv.org/abs/2304.09605v1) [paper-pdf](http://arxiv.org/pdf/2304.09605v1)

**Authors**: Simon Neves, Laura dos Santos Martins, Verena Yacoub, Pascal Lefebvre, Ivan Supic, Damian Markham, Eleni Diamanti

**Abstract**: Quantum transmission links are central elements in essentially all implementations of quantum information protocols. Emerging progress in quantum technologies involving such links needs to be accompanied by appropriate certification tools. In adversarial scenarios, a certification method can be vulnerable to attacks if too much trust is placed on the underlying system. Here, we propose a protocol in a device independent framework, which allows for the certification of practical quantum transmission links in scenarios where minimal assumptions are made about the functioning of the certification setup. In particular, we take unavoidable transmission losses into account by modeling the link as a completely-positive trace-decreasing map. We also, crucially, remove the assumption of independent and identically distributed samples, which is known to be incompatible with adversarial settings. Finally, in view of the use of the certified transmitted states for follow-up applications, our protocol moves beyond certification of the channel to allow us to estimate the quality of the transmitted state itself. To illustrate the practical relevance and the feasibility of our protocol with currently available technology we provide an experimental implementation based on a state-of-the-art polarization entangled photon pair source in a Sagnac configuration and analyze its robustness for realistic losses and errors.

摘要: 量子传输链路是几乎所有量子信息协议实现的核心要素。涉及这种联系的量子技术的新进展需要伴随着适当的认证工具。在对抗性场景中，如果对底层系统的信任过高，则认证方法可能容易受到攻击。在这里，我们提出了一个独立于设备的框架中的协议，它允许在对认证设置的功能做出最小假设的情况下对实际量子传输链路进行认证。特别地，我们通过将链路建模为完全正迹递减映射来考虑不可避免的传输损耗。至关重要的是，我们还取消了独立和同分布样本的假设，这是已知与对抗性设置不兼容的。最后，考虑到后续应用对认证的传输状态的使用，我们的协议超越了对信道的认证，允许我们估计传输状态本身的质量。为了说明我们的协议与现有技术的实际相关性和可行性，我们提供了一个基于Sagnac结构中最先进的偏振纠缠光子对源的实验实现，并分析了其对现实损失和错误的稳健性。



## **12. Masked Language Model Based Textual Adversarial Example Detection**

基于掩蔽语言模型的文本对抗性实例检测 cs.CR

13 pages,3 figures

**SubmitDate**: 2023-04-19    [abs](http://arxiv.org/abs/2304.08767v2) [paper-pdf](http://arxiv.org/pdf/2304.08767v2)

**Authors**: Xiaomei Zhang, Zhaoxi Zhang, Qi Zhong, Xufei Zheng, Yanjun Zhang, Shengshan Hu, Leo Yu Zhang

**Abstract**: Adversarial attacks are a serious threat to the reliable deployment of machine learning models in safety-critical applications. They can misguide current models to predict incorrectly by slightly modifying the inputs. Recently, substantial work has shown that adversarial examples tend to deviate from the underlying data manifold of normal examples, whereas pre-trained masked language models can fit the manifold of normal NLP data. To explore how to use the masked language model in adversarial detection, we propose a novel textual adversarial example detection method, namely Masked Language Model-based Detection (MLMD), which can produce clearly distinguishable signals between normal examples and adversarial examples by exploring the changes in manifolds induced by the masked language model. MLMD features a plug and play usage (i.e., no need to retrain the victim model) for adversarial defense and it is agnostic to classification tasks, victim model's architectures, and to-be-defended attack methods. We evaluate MLMD on various benchmark textual datasets, widely studied machine learning models, and state-of-the-art (SOTA) adversarial attacks (in total $3*4*4 = 48$ settings). Experimental results show that MLMD can achieve strong performance, with detection accuracy up to 0.984, 0.967, and 0.901 on AG-NEWS, IMDB, and SST-2 datasets, respectively. Additionally, MLMD is superior, or at least comparable to, the SOTA detection defenses in detection accuracy and F1 score. Among many defenses based on the off-manifold assumption of adversarial examples, this work offers a new angle for capturing the manifold change. The code for this work is openly accessible at \url{https://github.com/mlmddetection/MLMDdetection}.

摘要: 对抗性攻击对机器学习模型在安全关键型应用中的可靠部署构成严重威胁。他们可以通过稍微修改输入来误导当前的模型进行错误的预测。最近的大量工作表明，对抗性例子往往偏离正常例子的底层数据流形，而预先训练的掩蔽语言模型可以适应正常NLP数据的流形。为了探索掩蔽语言模型在对抗性检测中的应用，我们提出了一种新的文本对抗性实例检测方法，即基于掩蔽语言模型的检测方法(MLMD)，该方法通过研究掩蔽语言模型引起的流形变化来产生能够清晰区分正常例子和对抗性例子的信号。MLMD具有即插即用的特点(即，不需要重新训练受害者模型)用于对抗防御，并且它与分类任务、受害者模型的体系结构和要防御的攻击方法无关。我们在各种基准文本数据集、广泛研究的机器学习模型和最先进的(SOTA)对手攻击(总计$3*4*4=48$设置)上评估MLMD。实验结果表明，该算法在AG-NEWS、IMDB和SST-2数据集上的检测准确率分别达到0.984、0.967和0.901。此外，MLMD在检测精度和F1得分方面优于SOTA检测防御，或至少与SOTA检测防御相当。在许多基于对抗性例子的非流形假设的防御中，这项工作为捕捉流形变化提供了一个新的角度。这项工作的代码可以在\url{https://github.com/mlmddetection/MLMDdetection}.上公开访问



## **13. Understanding Overfitting in Adversarial Training via Kernel Regression**

用核回归方法理解对抗性训练中的过度适应 stat.ML

**SubmitDate**: 2023-04-19    [abs](http://arxiv.org/abs/2304.06326v2) [paper-pdf](http://arxiv.org/pdf/2304.06326v2)

**Authors**: Teng Zhang, Kang Li

**Abstract**: Adversarial training and data augmentation with noise are widely adopted techniques to enhance the performance of neural networks. This paper investigates adversarial training and data augmentation with noise in the context of regularized regression in a reproducing kernel Hilbert space (RKHS). We establish the limiting formula for these techniques as the attack and noise size, as well as the regularization parameter, tend to zero. Based on this limiting formula, we analyze specific scenarios and demonstrate that, without appropriate regularization, these two methods may have larger generalization error and Lipschitz constant than standard kernel regression. However, by selecting the appropriate regularization parameter, these two methods can outperform standard kernel regression and achieve smaller generalization error and Lipschitz constant. These findings support the empirical observations that adversarial training can lead to overfitting, and appropriate regularization methods, such as early stopping, can alleviate this issue.

摘要: 对抗性训练和带噪声的数据增强是提高神经网络性能的广泛采用的技术。在再生核Hilbert空间(RKHS)的正则化回归背景下，研究了带噪声的对抗性训练和数据增强问题。当攻击和噪声大小以及正则化参数趋于零时，我们建立了这些技术的极限公式。基于这一极限公式，我们分析了具体的情形，并证明了在没有适当的正则化的情况下，这两种方法可能具有比标准核回归更大的泛化误差和Lipschitz常数。然而，通过选择合适的正则化参数，这两种方法都可以获得比标准核回归更好的性能，并获得更小的泛化误差和Lipschitz常数。这些发现支持了经验观察，即对抗性训练会导致过度适应，而适当的正规化方法，如提前停止，可以缓解这一问题。



## **14. Secure Split Learning against Property Inference, Data Reconstruction, and Feature Space Hijacking Attacks**

抗属性推理、数据重构和特征空间劫持攻击的安全分裂学习 cs.LG

23 pages

**SubmitDate**: 2023-04-19    [abs](http://arxiv.org/abs/2304.09515v1) [paper-pdf](http://arxiv.org/pdf/2304.09515v1)

**Authors**: Yunlong Mao, Zexi Xin, Zhenyu Li, Jue Hong, Qingyou Yang, Sheng Zhong

**Abstract**: Split learning of deep neural networks (SplitNN) has provided a promising solution to learning jointly for the mutual interest of a guest and a host, which may come from different backgrounds, holding features partitioned vertically. However, SplitNN creates a new attack surface for the adversarial participant, holding back its practical use in the real world. By investigating the adversarial effects of highly threatening attacks, including property inference, data reconstruction, and feature hijacking attacks, we identify the underlying vulnerability of SplitNN and propose a countermeasure. To prevent potential threats and ensure the learning guarantees of SplitNN, we design a privacy-preserving tunnel for information exchange between the guest and the host. The intuition is to perturb the propagation of knowledge in each direction with a controllable unified solution. To this end, we propose a new activation function named R3eLU, transferring private smashed data and partial loss into randomized responses in forward and backward propagations, respectively. We give the first attempt to secure split learning against three threatening attacks and present a fine-grained privacy budget allocation scheme. The analysis proves that our privacy-preserving SplitNN solution provides a tight privacy budget, while the experimental results show that our solution performs better than existing solutions in most cases and achieves a good tradeoff between defense and model usability.

摘要: 深度神经网络的分裂学习(SplitNN)为来自不同背景、拥有垂直分割特征的宾主共同兴趣的联合学习提供了一种很有前途的解决方案。然而，SplitNN为对手参与者创造了一个新的攻击面，阻碍了其在现实世界中的实际应用。通过研究高威胁性攻击的对抗性，包括属性推断、数据重构和特征劫持攻击，我们识别了SplitNN的潜在脆弱性，并提出了对策。为了防止潜在的威胁，并保证SplitNN的学习保证，我们设计了一条隐私保护隧道，用于客户和主机之间的信息交换。直觉是用一种可控的统一解决方案扰乱知识在各个方向的传播。为此，我们提出了一种新的激活函数R3eLU，在前向传播和后向传播中分别将私有粉碎数据和部分丢失转移到随机响应中。我们首次尝试保护分裂学习不受三种威胁攻击，并提出了一种细粒度的隐私预算分配方案。分析证明，我们的隐私保护SplitNN方案提供了较少的隐私预算，而实验结果表明，我们的方案在大多数情况下都比现有的方案性能更好，并在防御性和模型可用性之间取得了良好的折衷。



## **15. Maybenot: A Framework for Traffic Analysis Defenses**

Maybenot：一种流量分析防御框架 cs.CR

**SubmitDate**: 2023-04-19    [abs](http://arxiv.org/abs/2304.09510v1) [paper-pdf](http://arxiv.org/pdf/2304.09510v1)

**Authors**: Tobias Pulls

**Abstract**: End-to-end encryption is a powerful tool for protecting the privacy of Internet users. Together with the increasing use of technologies such as Tor, VPNs, and encrypted messaging, it is becoming increasingly difficult for network adversaries to monitor and censor Internet traffic. One remaining avenue for adversaries is traffic analysis: the analysis of patterns in encrypted traffic to infer information about the users and their activities. Recent improvements using deep learning have made traffic analysis attacks more effective than ever before.   We present Maybenot, a framework for traffic analysis defenses. Maybenot is designed to be easy to use and integrate into existing end-to-end encrypted protocols. It is implemented in the Rust programming language as a crate (library), together with a simulator to further the development of defenses. Defenses in Maybenot are expressed as probabilistic state machines that schedule actions to inject padding or block outgoing traffic. Maybenot is an evolution from the Tor Circuit Padding Framework by Perry and Kadianakis, designed to support a wide range of protocols and use cases.

摘要: 端到端加密是保护互联网用户隐私的有力工具。随着ToR、VPN和加密消息等技术的使用越来越多，网络对手监控和审查互联网流量变得越来越困难。攻击者剩下的另一个途径是流量分析：分析加密流量中的模式，以推断有关用户及其活动的信息。最近使用深度学习的改进使流量分析攻击比以往任何时候都更有效。我们提出了一个用于流量分析防御的框架--Maybenot。Maybenot被设计为易于使用并集成到现有的端到端加密协议中。它在Rust编程语言中被实现为一个板条箱(库)，以及一个模拟器来进一步开发防御。Maybenot中的防御被表示为概率状态机，该状态机调度注入填充或阻止传出流量的动作。Maybenot是由Perry和Kadianakis提出的Tor电路填充框架的演变，旨在支持广泛的协议和用例。



## **16. Wavelets Beat Monkeys at Adversarial Robustness**

小波在对抗健壮性上击败猴子 cs.LG

Machine Learning and the Physical Sciences Workshop, NeurIPS 2022

**SubmitDate**: 2023-04-19    [abs](http://arxiv.org/abs/2304.09403v1) [paper-pdf](http://arxiv.org/pdf/2304.09403v1)

**Authors**: Jingtong Su, Julia Kempe

**Abstract**: Research on improving the robustness of neural networks to adversarial noise - imperceptible malicious perturbations of the data - has received significant attention. The currently uncontested state-of-the-art defense to obtain robust deep neural networks is Adversarial Training (AT), but it consumes significantly more resources compared to standard training and trades off accuracy for robustness. An inspiring recent work [Dapello et al.] aims to bring neurobiological tools to the question: How can we develop Neural Nets that robustly generalize like human vision? [Dapello et al.] design a network structure with a neural hidden first layer that mimics the primate primary visual cortex (V1), followed by a back-end structure adapted from current CNN vision models. It seems to achieve non-trivial adversarial robustness on standard vision benchmarks when tested on small perturbations. Here we revisit this biologically inspired work, and ask whether a principled parameter-free representation with inspiration from physics is able to achieve the same goal. We discover that the wavelet scattering transform can replace the complex V1-cortex and simple uniform Gaussian noise can take the role of neural stochasticity, to achieve adversarial robustness. In extensive experiments on the CIFAR-10 benchmark with adaptive adversarial attacks we show that: 1) Robustness of VOneBlock architectures is relatively weak (though non-zero) when the strength of the adversarial attack radius is set to commonly used benchmarks. 2) Replacing the front-end VOneBlock by an off-the-shelf parameter-free Scatternet followed by simple uniform Gaussian noise can achieve much more substantial adversarial robustness without adversarial training. Our work shows how physically inspired structures yield new insights into robustness that were previously only thought possible by meticulously mimicking the human cortex.

摘要: 提高神经网络对敌意噪声--数据的不可察觉的恶意扰动--的稳健性的研究受到了极大的关注。目前无可争议的获得稳健深度神经网络的最先进的防御是对抗性训练(AT)，但与标准训练相比，它消耗的资源明显更多，并以精度换取健壮性。最近一部鼓舞人心的作品[Dapello等人]旨在将神经生物学工具引入这个问题：我们如何才能开发出像人类视觉一样具有强大泛化能力的神经网络？[Dapello等人]设计一个网络结构，该网络结构具有一个模仿灵长类初级视觉皮质(V1)的神经隐藏第一层，然后是一个改编自当前CNN视觉模型的后端结构。当在小扰动上测试时，它似乎在标准视觉基准上实现了非平凡的对抗性健壮性。在这里，我们重新审视这项受生物启发的工作，并询问受物理学启发的原则性无参数表示法是否能够实现同样的目标。我们发现，小波散射变换可以代替复杂的V1皮层，而简单的均匀高斯噪声可以起到神经随机性的作用，达到对抗的稳健性。在CIFAR-10基准上进行的大量自适应攻击实验表明：1)当敌方攻击半径设置为常用基准时，VOneBlock架构的健壮性相对较弱(尽管非零值)。2)用无参数散射网和简单的均匀高斯噪声代替前端的VOneBlock，无需对抗性训练即可获得更强的对抗性健壮性。我们的工作展示了受物理启发的结构如何产生对健壮性的新见解，而这在以前只被认为是通过精心模仿人类皮质才可能实现的。



## **17. CodeAttack: Code-Based Adversarial Attacks for Pre-trained Programming Language Models**

CodeAttack：针对预先训练的编程语言模型的基于代码的对抗性攻击 cs.CL

AAAI Conference on Artificial Intelligence (AAAI) 2023

**SubmitDate**: 2023-04-18    [abs](http://arxiv.org/abs/2206.00052v3) [paper-pdf](http://arxiv.org/pdf/2206.00052v3)

**Authors**: Akshita Jha, Chandan K. Reddy

**Abstract**: Pre-trained programming language (PL) models (such as CodeT5, CodeBERT, GraphCodeBERT, etc.,) have the potential to automate software engineering tasks involving code understanding and code generation. However, these models operate in the natural channel of code, i.e., they are primarily concerned with the human understanding of the code. They are not robust to changes in the input and thus, are potentially susceptible to adversarial attacks in the natural channel. We propose, CodeAttack, a simple yet effective black-box attack model that uses code structure to generate effective, efficient, and imperceptible adversarial code samples and demonstrates the vulnerabilities of the state-of-the-art PL models to code-specific adversarial attacks. We evaluate the transferability of CodeAttack on several code-code (translation and repair) and code-NL (summarization) tasks across different programming languages. CodeAttack outperforms state-of-the-art adversarial NLP attack models to achieve the best overall drop in performance while being more efficient, imperceptible, consistent, and fluent. The code can be found at https://github.com/reddy-lab-code-research/CodeAttack.

摘要: 预先训练的编程语言(PL)模型(如CodeT5、CodeBERT、GraphCodeBERT等)有可能自动化涉及代码理解和代码生成的软件工程任务。然而，这些模型在代码的自然通道中运行，即它们主要关注人类对代码的理解。它们对输入的变化不是很健壮，因此，在自然通道中可能容易受到对抗性攻击。我们提出了一个简单而有效的黑盒攻击模型CodeAttack，它使用代码结构来生成有效、高效和不可察觉的对抗性代码样本，并展示了最新的PL模型对代码特定的对抗性攻击的脆弱性。我们评估了CodeAttack在几个代码-代码(翻译和修复)和代码-NL(摘要)任务上跨不同编程语言的可移植性。CodeAttack超越了最先进的对抗性NLP攻击模型，在更高效、更隐蔽、更一致和更流畅的同时，实现了最佳的整体性能下降。代码可在https://github.com/reddy-lab-code-research/CodeAttack.上找到



## **18. Analyzing Activity and Suspension Patterns of Twitter Bots Attacking Turkish Twitter Trends by a Longitudinal Dataset**

利用纵向数据集分析Twitter机器人攻击土耳其Twitter趋势的活跃度和暂停模式 cs.SI

Accepted to Cyber Social Threats (CySoc) 2023 colocated with  WebConf23

**SubmitDate**: 2023-04-18    [abs](http://arxiv.org/abs/2304.07907v2) [paper-pdf](http://arxiv.org/pdf/2304.07907v2)

**Authors**: Tuğrulcan Elmas

**Abstract**: Twitter bots amplify target content in a coordinated manner to make them appear popular, which is an astroturfing attack. Such attacks promote certain keywords to push them to Twitter trends to make them visible to a broader audience. Past work on such fake trends revealed a new astroturfing attack named ephemeral astroturfing that employs a very unique bot behavior in which bots post and delete generated tweets in a coordinated manner. As such, it is easy to mass-annotate such bots reliably, making them a convenient source of ground truth for bot research. In this paper, we detect and disclose over 212,000 such bots targeting Turkish trends, which we name astrobots. We also analyze their activity and suspension patterns. We found that Twitter purged those bots en-masse 6 times since June 2018. However, the adversaries reacted quickly and deployed new bots that were created years ago. We also found that many such bots do not post tweets apart from promoting fake trends, which makes it challenging for bot detection methods to detect them. Our work provides insights into platforms' content moderation practices and bot detection research. The dataset is publicly available at https://github.com/tugrulz/EphemeralAstroturfing.

摘要: Twitter机器人以一种协调的方式放大目标内容，使它们看起来很受欢迎，这是一种占星术的攻击。这类攻击推广某些关键字，将它们推向Twitter趋势，让更多的受众看到它们。过去对这种虚假趋势的研究揭示了一种名为短暂占星术的新的占星术攻击，它采用了一种非常独特的机器人行为，机器人以协调的方式发布和删除生成的推文。因此，很容易对这类机器人进行可靠的大规模注释，使它们成为机器人研究的一个方便的基本事实来源。在本文中，我们检测并披露了超过21.2万个针对土耳其趋势的此类机器人，我们将其命名为机器人。我们还分析了它们的活动和悬浮模式。我们发现，自2018年6月以来，Twitter对这些机器人进行了6次集体清除。然而，对手们反应迅速，部署了几年前创造的新机器人。我们还发现，许多这类机器人除了宣传虚假趋势外，不会发布推文，这使得机器人检测方法对检测它们具有挑战性。我们的工作为平台的内容审核实践和机器人检测研究提供了见解。该数据集在https://github.com/tugrulz/EphemeralAstroturfing.上公开提供



## **19. An Analysis of Robustness of Non-Lipschitz Networks**

非Lipschitz网络的稳健性分析 cs.LG

To appear in Journal of Machine Learning Research (JMLR)

**SubmitDate**: 2023-04-18    [abs](http://arxiv.org/abs/2010.06154v4) [paper-pdf](http://arxiv.org/pdf/2010.06154v4)

**Authors**: Maria-Florina Balcan, Avrim Blum, Dravyansh Sharma, Hongyang Zhang

**Abstract**: Despite significant advances, deep networks remain highly susceptible to adversarial attack. One fundamental challenge is that small input perturbations can often produce large movements in the network's final-layer feature space. In this paper, we define an attack model that abstracts this challenge, to help understand its intrinsic properties. In our model, the adversary may move data an arbitrary distance in feature space but only in random low-dimensional subspaces. We prove such adversaries can be quite powerful: defeating any algorithm that must classify any input it is given. However, by allowing the algorithm to abstain on unusual inputs, we show such adversaries can be overcome when classes are reasonably well-separated in feature space. We further provide strong theoretical guarantees for setting algorithm parameters to optimize over accuracy-abstention trade-offs using data-driven methods. Our results provide new robustness guarantees for nearest-neighbor style algorithms, and also have application to contrastive learning, where we empirically demonstrate the ability of such algorithms to obtain high robust accuracy with low abstention rates. Our model is also motivated by strategic classification, where entities being classified aim to manipulate their observable features to produce a preferred classification, and we provide new insights into that area as well.

摘要: 尽管取得了重大进展，深度网络仍然极易受到对手的攻击。一个基本的挑战是，小的输入扰动通常会在网络的最后一层特征空间中产生大的运动。在本文中，我们定义了一个抽象这一挑战的攻击模型，以帮助理解其内在属性。在我们的模型中，敌手可以将数据在特征空间中移动任意距离，但只能在随机低维子空间中移动。我们证明了这样的对手可以是相当强大的：击败任何必须对给定的任何输入进行分类的算法。然而，通过允许算法在不寻常的输入上弃权，我们证明了当类在特征空间中合理地分离时，这样的对手可以被克服。我们进一步为使用数据驱动方法设置算法参数以优化过度精确度权衡提供了有力的理论保证。我们的结果为最近邻式算法提供了新的稳健性保证，并在对比学习中也得到了应用，我们的经验证明了这种算法能够在较低的弃权率下获得较高的鲁棒性精度。我们的模型也受到战略分类的推动，在战略分类中，被分类的实体旨在操纵它们的可观察特征来产生首选的分类，我们也提供了对该领域的新见解。



## **20. BadVFL: Backdoor Attacks in Vertical Federated Learning**

BadVFL：垂直联合学习中的后门攻击 cs.LG

**SubmitDate**: 2023-04-18    [abs](http://arxiv.org/abs/2304.08847v1) [paper-pdf](http://arxiv.org/pdf/2304.08847v1)

**Authors**: Mohammad Naseri, Yufei Han, Emiliano De Cristofaro

**Abstract**: Federated learning (FL) enables multiple parties to collaboratively train a machine learning model without sharing their data; rather, they train their own model locally and send updates to a central server for aggregation. Depending on how the data is distributed among the participants, FL can be classified into Horizontal (HFL) and Vertical (VFL). In VFL, the participants share the same set of training instances but only host a different and non-overlapping subset of the whole feature space. Whereas in HFL, each participant shares the same set of features while the training set is split into locally owned training data subsets.   VFL is increasingly used in applications like financial fraud detection; nonetheless, very little work has analyzed its security. In this paper, we focus on robustness in VFL, in particular, on backdoor attacks, whereby an adversary attempts to manipulate the aggregate model during the training process to trigger misclassifications. Performing backdoor attacks in VFL is more challenging than in HFL, as the adversary i) does not have access to the labels during training and ii) cannot change the labels as she only has access to the feature embeddings. We present a first-of-its-kind clean-label backdoor attack in VFL, which consists of two phases: a label inference and a backdoor phase. We demonstrate the effectiveness of the attack on three different datasets, investigate the factors involved in its success, and discuss countermeasures to mitigate its impact.

摘要: 联合学习(FL)使多方能够协作地训练机器学习模型，而不共享他们的数据；相反，他们在本地训练他们自己的模型，并将更新发送到中央服务器以进行聚合。根据数据在参与者之间的分布情况，外语可分为水平(HFL)和垂直(VFL)。在VFL中，参与者共享相同的训练实例集，但仅托管整个特征空间的不同且不重叠的子集。而在HFL中，每个参与者共享相同的特征集，而训练集被分成本地拥有的训练数据子集。VFL越来越多地被用于金融欺诈检测等应用中；然而，很少有工作分析它的安全性。在本文中，我们关注VFL的稳健性，特别是后门攻击，即对手试图在训练过程中操纵聚合模型以触发错误分类。在VFL中执行后门攻击比在HFL中更具挑战性，因为对手i)在训练期间无法访问标签，ii)无法更改标签，因为她只能访问特征嵌入。在VFL中，我们提出了一种首次的干净标签后门攻击，它包括两个阶段：标签推理和后门阶段。我们在三个不同的数据集上演示了攻击的有效性，调查了其成功的相关因素，并讨论了减轻其影响的对策。



## **21. Towards the Transferable Audio Adversarial Attack via Ensemble Methods**

基于集成方法的可转移音频对抗攻击 cs.CR

Submitted to Cybersecurity journal 2023

**SubmitDate**: 2023-04-18    [abs](http://arxiv.org/abs/2304.08811v1) [paper-pdf](http://arxiv.org/pdf/2304.08811v1)

**Authors**: Feng Guo, Zheng Sun, Yuxuan Chen, Lei Ju

**Abstract**: In recent years, deep learning (DL) models have achieved significant progress in many domains, such as autonomous driving, facial recognition, and speech recognition. However, the vulnerability of deep learning models to adversarial attacks has raised serious concerns in the community because of their insufficient robustness and generalization. Also, transferable attacks have become a prominent method for black-box attacks. In this work, we explore the potential factors that impact adversarial examples (AEs) transferability in DL-based speech recognition. We also discuss the vulnerability of different DL systems and the irregular nature of decision boundaries. Our results show a remarkable difference in the transferability of AEs between speech and images, with the data relevance being low in images but opposite in speech recognition. Motivated by dropout-based ensemble approaches, we propose random gradient ensembles and dynamic gradient-weighted ensembles, and we evaluate the impact of ensembles on the transferability of AEs. The results show that the AEs created by both approaches are valid for transfer to the black box API.

摘要: 近年来，深度学习模型在自动驾驶、人脸识别、语音识别等领域取得了长足的进步。然而，深度学习模型对敌意攻击的脆弱性已经引起了社会各界的严重关注，因为它们的健壮性和泛化程度不够。此外，可转移攻击已经成为黑盒攻击的一种重要方法。在这项工作中，我们探讨了在基于数字图书馆的语音识别中影响对抗性例子(AES)可转移性的潜在因素。我们还讨论了不同DL系统的脆弱性和决策边界的不规则性。我们的结果表明，语音和图像之间的声学效应的可转移性有显著差异，图像中的数据相关性较低，而语音识别中的数据相关性则相反。在基于辍学的集成方法的启发下，我们提出了随机梯度集成和动态梯度加权集成，并评估了集成对高级进化的可转移性的影响。结果表明，这两种方法生成的动态平衡函数都可以有效地转移到黑盒API中。



## **22. Order-Disorder: Imitation Adversarial Attacks for Black-box Neural Ranking Models**

有序-无序：对黑盒神经网络排序模型的模仿敌意攻击 cs.IR

15 pages, 4 figures, accepted by ACM CCS 2022, Best Paper Nomination

**SubmitDate**: 2023-04-18    [abs](http://arxiv.org/abs/2209.06506v2) [paper-pdf](http://arxiv.org/pdf/2209.06506v2)

**Authors**: Jiawei Liu, Yangyang Kang, Di Tang, Kaisong Song, Changlong Sun, Xiaofeng Wang, Wei Lu, Xiaozhong Liu

**Abstract**: Neural text ranking models have witnessed significant advancement and are increasingly being deployed in practice. Unfortunately, they also inherit adversarial vulnerabilities of general neural models, which have been detected but remain underexplored by prior studies. Moreover, the inherit adversarial vulnerabilities might be leveraged by blackhat SEO to defeat better-protected search engines. In this study, we propose an imitation adversarial attack on black-box neural passage ranking models. We first show that the target passage ranking model can be transparentized and imitated by enumerating critical queries/candidates and then train a ranking imitation model. Leveraging the ranking imitation model, we can elaborately manipulate the ranking results and transfer the manipulation attack to the target ranking model. For this purpose, we propose an innovative gradient-based attack method, empowered by the pairwise objective function, to generate adversarial triggers, which causes premeditated disorderliness with very few tokens. To equip the trigger camouflages, we add the next sentence prediction loss and the language model fluency constraint to the objective function. Experimental results on passage ranking demonstrate the effectiveness of the ranking imitation attack model and adversarial triggers against various SOTA neural ranking models. Furthermore, various mitigation analyses and human evaluation show the effectiveness of camouflages when facing potential mitigation approaches. To motivate other scholars to further investigate this novel and important problem, we make the experiment data and code publicly available.

摘要: 神经文本排序模型已经取得了显著的进步，并越来越多地应用于实践中。不幸的是，它们也继承了一般神经模型的对抗性漏洞，这些漏洞已经被检测到，但仍未被先前的研究充分探索。此外，BlackHat SEO可能会利用继承的敌意漏洞来击败保护更好的搜索引擎。在这项研究中，我们提出了一种对黑盒神经通路排序模型的模仿对抗性攻击。我们首先证明了目标文章排序模型可以通过列举关键查询/候选来透明化和模仿，然后训练一个排序模仿模型。利用排序模拟模型，可以对排序结果进行精细的操作，并将操纵攻击转移到目标排序模型。为此，我们提出了一种创新的基于梯度的攻击方法，该方法在两两目标函数的授权下，生成对抗性触发器，以极少的令牌造成有预谋的无序。为了装备触发伪装，我们在目标函数中加入了下一句预测损失和语言模型流畅度约束。文章排序的实验结果证明了该排序模仿攻击模型和敌意触发器对各种SOTA神经排序模型的有效性。此外，各种缓解分析和人类评估表明，当面临潜在的缓解方法时，伪装是有效的。为了激励其他学者进一步研究这一新颖而重要的问题，我们公开了实验数据和代码。



## **23. A Survey of Adversarial Defences and Robustness in NLP**

自然语言处理中的对抗性防御和健壮性研究综述 cs.CL

Accepted for publication at ACM Computing Surveys

**SubmitDate**: 2023-04-18    [abs](http://arxiv.org/abs/2203.06414v4) [paper-pdf](http://arxiv.org/pdf/2203.06414v4)

**Authors**: Shreya Goyal, Sumanth Doddapaneni, Mitesh M. Khapra, Balaraman Ravindran

**Abstract**: In the past few years, it has become increasingly evident that deep neural networks are not resilient enough to withstand adversarial perturbations in input data, leaving them vulnerable to attack. Various authors have proposed strong adversarial attacks for computer vision and Natural Language Processing (NLP) tasks. As a response, many defense mechanisms have also been proposed to prevent these networks from failing. The significance of defending neural networks against adversarial attacks lies in ensuring that the model's predictions remain unchanged even if the input data is perturbed. Several methods for adversarial defense in NLP have been proposed, catering to different NLP tasks such as text classification, named entity recognition, and natural language inference. Some of these methods not only defend neural networks against adversarial attacks but also act as a regularization mechanism during training, saving the model from overfitting. This survey aims to review the various methods proposed for adversarial defenses in NLP over the past few years by introducing a novel taxonomy. The survey also highlights the fragility of advanced deep neural networks in NLP and the challenges involved in defending them.

摘要: 在过去的几年里，越来越明显的是，深度神经网络没有足够的弹性来抵御输入数据中的对抗性扰动，从而使它们容易受到攻击。许多作者提出了针对计算机视觉和自然语言处理(NLP)任务的强对抗性攻击。作为回应，许多防御机制也被提出，以防止这些网络崩溃。保护神经网络免受敌意攻击的意义在于确保即使输入数据受到扰动，模型的预测也保持不变。针对自然语言处理中文本分类、命名实体识别、自然语言推理等不同的自然语言处理任务，提出了几种自然语言处理对抗性防御方法。这些方法中的一些不仅保护神经网络免受对手攻击，而且在训练过程中充当正则化机制，避免了模型的过拟合。本综述旨在通过引入一种新的分类方法来回顾过去几年中提出的在NLP中进行对抗防御的各种方法。这项调查还突显了NLP中先进的深度神经网络的脆弱性以及捍卫它们所涉及的挑战。



## **24. Binarized ResNet: Enabling Robust Automatic Modulation Classification at the resource-constrained Edge**

二值化ResNet：在资源受限的边缘实现稳健的自动调制分类 cs.IT

This version has a total of 8 figures and 3 tables. It has extra  content on the adversarial robustness of the proposed method that was not  present in the previous submission. Also one more ensemble method called  RBLResNet-MCK is proposed to improve the performance further

**SubmitDate**: 2023-04-18    [abs](http://arxiv.org/abs/2110.14357v2) [paper-pdf](http://arxiv.org/pdf/2110.14357v2)

**Authors**: Deepsayan Sadhukhan, Nitin Priyadarshini Shankar, Nancy Nayak, Thulasi Tholeti, Sheetal Kalyani

**Abstract**: Recently, deep neural networks (DNNs) have been used extensively for automatic modulation classification (AMC), and the results have been quite promising. However, DNNs have high memory and computation requirements making them impractical for edge networks where the devices are resource-constrained. They are also vulnerable to adversarial attacks, which is a significant security concern. This work proposes a rotated binary large ResNet (RBLResNet) for AMC that can be deployed at the edge network because of low memory and computational complexity. The performance gap between the RBLResNet and existing architectures with floating-point weights and activations can be closed by two proposed ensemble methods: (i) multilevel classification (MC), and (ii) bagging multiple RBLResNets while retaining low memory and computational power. The MC method achieves an accuracy of $93.39\%$ at $10$dB over all the $24$ modulation classes of the Deepsig dataset. This performance is comparable to state-of-the-art (SOTA) performances, with $4.75$ times lower memory and $1214$ times lower computation. Furthermore, RBLResNet also has high adversarial robustness compared to existing DNN models. The proposed MC method with RBLResNets has an adversarial accuracy of $87.25\%$ over a wide range of SNRs, surpassing the robustness of all existing SOTA methods to the best of our knowledge. Properties such as low memory, low computation, and the highest adversarial robustness make it a better choice for robust AMC in low-power edge devices.

摘要: 近年来，深度神经网络(DNN)已被广泛应用于自动调制分类(AMC)，并取得了良好的效果。然而，DNN对内存和计算的要求很高，这使得它们不适用于设备资源有限的边缘网络。它们还容易受到敌意攻击，这是一个重大的安全问题。本文提出了一种适用于AMC的轮转二进制大型ResNet(RBLResNet)，该网络具有较低的存储容量和计算复杂度，可以部署在边缘网络中。RBLResNet与现有的具有浮点权重和激活的体系结构之间的性能差距可以通过两种提出的集成方法来缩小：(I)多级分类(MC)和(Ii)在保持低存储和计算能力的情况下打包多个RBLResNet。MC方法在Deepsig数据集的所有$24$调制类别上，在$10$分贝下实现了$93.39\$的精度。这一性能与最先进的(SOTA)性能相当，内存减少了4.75美元，计算减少了1214美元。此外，与现有的DNN模型相比，RBLResNet还具有较高的对抗健壮性。提出的基于RBLResNets的MC方法在较宽的信噪比范围内具有87.25美元的对抗精度，超过了现有SOTA方法的稳健性。低内存、低计算量和最高的对抗健壮性等特性使其成为低功率边缘设备中健壮AMC的更好选择。



## **25. Employing Deep Ensemble Learning for Improving the Security of Computer Networks against Adversarial Attacks**

利用深度集成学习提高计算机网络抗敌意攻击的安全性 cs.CR

**SubmitDate**: 2023-04-17    [abs](http://arxiv.org/abs/2209.12195v2) [paper-pdf](http://arxiv.org/pdf/2209.12195v2)

**Authors**: Ehsan Nowroozi, Mohammadreza Mohammadi, Erkay Savas, Mauro Conti, Yassine Mekdad

**Abstract**: In the past few years, Convolutional Neural Networks (CNN) have demonstrated promising performance in various real-world cybersecurity applications, such as network and multimedia security. However, the underlying fragility of CNN structures poses major security problems, making them inappropriate for use in security-oriented applications including such computer networks. Protecting these architectures from adversarial attacks necessitates using security-wise architectures that are challenging to attack.   In this study, we present a novel architecture based on an ensemble classifier that combines the enhanced security of 1-Class classification (known as 1C) with the high performance of conventional 2-Class classification (known as 2C) in the absence of attacks.Our architecture is referred to as the 1.5-Class (SPRITZ-1.5C) classifier and constructed using a final dense classifier, one 2C classifier (i.e., CNNs), and two parallel 1C classifiers (i.e., auto-encoders). In our experiments, we evaluated the robustness of our proposed architecture by considering eight possible adversarial attacks in various scenarios. We performed these attacks on the 2C and SPRITZ-1.5C architectures separately. The experimental results of our study showed that the Attack Success Rate (ASR) of the I-FGSM attack against a 2C classifier trained with the N-BaIoT dataset is 0.9900. In contrast, the ASR is 0.0000 for the SPRITZ-1.5C classifier.

摘要: 在过去的几年里，卷积神经网络(CNN)在各种现实世界的网络安全应用中表现出了良好的性能，如网络和多媒体安全。然而，CNN结构的潜在脆弱性造成了重大的安全问题，使其不适合用于包括此类计算机网络在内的面向安全的应用程序。要保护这些架构免受敌意攻击，就必须使用具有安全性的架构，这些架构很难受到攻击。在这项研究中，我们提出了一种新的基于集成分类器的体系结构，它结合了1类分类(称为1C)的增强安全性和传统2类分类(称为2C)的高性能，在没有攻击的情况下被称为1.5类(SPRITZ-1.5C)分类器，并使用最终的密集分类器、一个2C分类器(即CNN)和两个并行的1C分类器(即自动编码器)来构建。在我们的实验中，我们通过考虑不同场景中八种可能的对抗性攻击来评估我们所提出的体系结构的健壮性。我们分别在2C和SPRITZ-1.5C架构上执行了这些攻击。实验结果表明，利用N-BaIoT数据集训练的2C分类器对I-FGSM的攻击成功率为0.9900。相比之下，SPRITZ-1.5C分级机的ASR为0.0000。



## **26. RNN-Guard: Certified Robustness Against Multi-frame Attacks for Recurrent Neural Networks**

RNN-Guard：递归神经网络对多帧攻击的验证鲁棒性 cs.LG

13 pages, 7 figures, 6 tables

**SubmitDate**: 2023-04-17    [abs](http://arxiv.org/abs/2304.07980v1) [paper-pdf](http://arxiv.org/pdf/2304.07980v1)

**Authors**: Yunruo Zhang, Tianyu Du, Shouling Ji, Peng Tang, Shanqing Guo

**Abstract**: It is well-known that recurrent neural networks (RNNs), although widely used, are vulnerable to adversarial attacks including one-frame attacks and multi-frame attacks. Though a few certified defenses exist to provide guaranteed robustness against one-frame attacks, we prove that defending against multi-frame attacks remains a challenging problem due to their enormous perturbation space. In this paper, we propose the first certified defense against multi-frame attacks for RNNs called RNN-Guard. To address the above challenge, we adopt the perturb-all-frame strategy to construct perturbation spaces consistent with those in multi-frame attacks. However, the perturb-all-frame strategy causes a precision issue in linear relaxations. To address this issue, we introduce a novel abstract domain called InterZono and design tighter relaxations. We prove that InterZono is more precise than Zonotope yet carries the same time complexity. Experimental evaluations across various datasets and model structures show that the certified robust accuracy calculated by RNN-Guard with InterZono is up to 2.18 times higher than that with Zonotope. In addition, we extend RNN-Guard as the first certified training method against multi-frame attacks to directly enhance RNNs' robustness. The results show that the certified robust accuracy of models trained with RNN-Guard against multi-frame attacks is 15.47 to 67.65 percentage points higher than those with other training methods.

摘要: 众所周知，递归神经网络(RNN)虽然被广泛使用，但很容易受到包括一帧攻击和多帧攻击在内的对抗性攻击。尽管存在一些经过认证的防御措施可以提供对单帧攻击的保证健壮性，但我们证明，由于多帧攻击的巨大扰动空间，防御多帧攻击仍然是一个具有挑战性的问题。在本文中，我们提出了第一个经过认证的针对RNN的多帧攻击防御机制RNN-Guard。为了应对上述挑战，我们采用扰动全帧策略来构造与多帧攻击中一致的扰动空间。然而，扰动全框架策略会导致线性松弛中的精度问题。为了解决这个问题，我们引入了一个新的抽象域InterZono，并设计了更紧密的松弛。我们证明了InterZono比Zonotope更精确，但具有相同的时间复杂度。在不同的数据集和模型结构上的实验评估表明，RNN-Guard使用InterZono计算的健壮性精度比使用Zonotope计算的健壮性高2.18倍。此外，我们扩展了RNN-Guard作为第一种经过认证的针对多帧攻击的训练方法，以直接增强RNN的健壮性。结果表明，用RNN-Guard训练的模型对多帧攻击的鲁棒性比其他训练方法高15.47~67.65个百分点。



## **27. A Review of Speech-centric Trustworthy Machine Learning: Privacy, Safety, and Fairness**

以语音为中心的可信机器学习：隐私、安全和公平 cs.SD

**SubmitDate**: 2023-04-16    [abs](http://arxiv.org/abs/2212.09006v2) [paper-pdf](http://arxiv.org/pdf/2212.09006v2)

**Authors**: Tiantian Feng, Rajat Hebbar, Nicholas Mehlman, Xuan Shi, Aditya Kommineni, and Shrikanth Narayanan

**Abstract**: Speech-centric machine learning systems have revolutionized many leading domains ranging from transportation and healthcare to education and defense, profoundly changing how people live, work, and interact with each other. However, recent studies have demonstrated that many speech-centric ML systems may need to be considered more trustworthy for broader deployment. Specifically, concerns over privacy breaches, discriminating performance, and vulnerability to adversarial attacks have all been discovered in ML research fields. In order to address the above challenges and risks, a significant number of efforts have been made to ensure these ML systems are trustworthy, especially private, safe, and fair. In this paper, we conduct the first comprehensive survey on speech-centric trustworthy ML topics related to privacy, safety, and fairness. In addition to serving as a summary report for the research community, we point out several promising future research directions to inspire the researchers who wish to explore further in this area.

摘要: 以语音为中心的机器学习系统已经彻底改变了从交通和医疗到教育和国防等许多领先领域，深刻地改变了人们的生活、工作和相互互动的方式。然而，最近的研究表明，许多以语音为中心的ML系统可能需要被认为更值得信赖，以便更广泛地部署。具体地说，在ML研究领域中发现了对隐私泄露、区分性能和对对手攻击的脆弱性的担忧。为了应对上述挑战和风险，已经做出了大量努力，以确保这些ML系统是值得信任的，特别是私密、安全和公平。在本文中，我们首次对以语音为中心的可信ML话题进行了全面的调查，这些话题涉及隐私、安全和公平。除了作为研究界的总结报告外，我们还指出了几个有前途的未来研究方向，以激励希望在这一领域进一步探索的研究人员。



## **28. Visual Prompting for Adversarial Robustness**

对抗健壮性的视觉提示 cs.CV

ICASSP 2023

**SubmitDate**: 2023-04-15    [abs](http://arxiv.org/abs/2210.06284v3) [paper-pdf](http://arxiv.org/pdf/2210.06284v3)

**Authors**: Aochuan Chen, Peter Lorenz, Yuguang Yao, Pin-Yu Chen, Sijia Liu

**Abstract**: In this work, we leverage visual prompting (VP) to improve adversarial robustness of a fixed, pre-trained model at testing time. Compared to conventional adversarial defenses, VP allows us to design universal (i.e., data-agnostic) input prompting templates, which have plug-and-play capabilities at testing time to achieve desired model performance without introducing much computation overhead. Although VP has been successfully applied to improving model generalization, it remains elusive whether and how it can be used to defend against adversarial attacks. We investigate this problem and show that the vanilla VP approach is not effective in adversarial defense since a universal input prompt lacks the capacity for robust learning against sample-specific adversarial perturbations. To circumvent it, we propose a new VP method, termed Class-wise Adversarial Visual Prompting (C-AVP), to generate class-wise visual prompts so as to not only leverage the strengths of ensemble prompts but also optimize their interrelations to improve model robustness. Our experiments show that C-AVP outperforms the conventional VP method, with 2.1X standard accuracy gain and 2X robust accuracy gain. Compared to classical test-time defenses, C-AVP also yields a 42X inference time speedup.

摘要: 在这项工作中，我们利用视觉提示(VP)来提高测试时固定的、预先训练的模型的对抗健壮性。与传统的对抗性防御相比，VP允许我们设计通用的(即数据不可知的)输入提示模板，该模板在测试时具有即插即用的能力，在不引入太多计算开销的情况下达到期望的模型性能。虽然VP已经被成功地应用于改进模型泛化，但它是否以及如何被用来防御对手攻击仍然是一个难以捉摸的问题。我们研究了这个问题，并证明了普通VP方法在对抗防御中不是有效的，因为通用的输入提示缺乏针对特定样本的对抗扰动的稳健学习能力。针对这一问题，我们提出了一种新的分类对抗性视觉提示生成方法--分类对抗性视觉提示(C-AVP)，该方法不仅可以利用集成提示的优点，而且可以优化它们之间的相互关系，从而提高模型的稳健性。实验表明，C-AVP比传统的VP方法有2.1倍的标准精度增益和2倍的稳健精度增益。与经典的测试时间防御相比，C-AVP的推理时间加速比也提高了42倍。



## **29. XploreNAS: Explore Adversarially Robust & Hardware-efficient Neural Architectures for Non-ideal Xbars**

XploreNAS：探索针对非理想Xbar的强大且硬件高效的神经体系结构 cs.LG

Accepted to ACM Transactions on Embedded Computing Systems in April  2023

**SubmitDate**: 2023-04-15    [abs](http://arxiv.org/abs/2302.07769v2) [paper-pdf](http://arxiv.org/pdf/2302.07769v2)

**Authors**: Abhiroop Bhattacharjee, Abhishek Moitra, Priyadarshini Panda

**Abstract**: Compute In-Memory platforms such as memristive crossbars are gaining focus as they facilitate acceleration of Deep Neural Networks (DNNs) with high area and compute-efficiencies. However, the intrinsic non-idealities associated with the analog nature of computing in crossbars limits the performance of the deployed DNNs. Furthermore, DNNs are shown to be vulnerable to adversarial attacks leading to severe security threats in their large-scale deployment. Thus, finding adversarially robust DNN architectures for non-ideal crossbars is critical to the safe and secure deployment of DNNs on the edge. This work proposes a two-phase algorithm-hardware co-optimization approach called XploreNAS that searches for hardware-efficient & adversarially robust neural architectures for non-ideal crossbar platforms. We use the one-shot Neural Architecture Search (NAS) approach to train a large Supernet with crossbar-awareness and sample adversarially robust Subnets therefrom, maintaining competitive hardware-efficiency. Our experiments on crossbars with benchmark datasets (SVHN, CIFAR10 & CIFAR100) show upto ~8-16% improvement in the adversarial robustness of the searched Subnets against a baseline ResNet-18 model subjected to crossbar-aware adversarial training. We benchmark our robust Subnets for Energy-Delay-Area-Products (EDAPs) using the Neurosim tool and find that with additional hardware-efficiency driven optimizations, the Subnets attain ~1.5-1.6x lower EDAPs than ResNet-18 baseline.

摘要: 内存交叉开关等计算内存平台因其高面积和高计算效率促进深度神经网络(DNN)的加速而备受关注。然而，与交叉开关中计算的模拟性质相关联的固有非理想性限制了所部署的DNN的性能。此外，DNN在大规模部署时容易受到对抗性攻击，从而造成严重的安全威胁。因此，为非理想的交叉开关找到相对健壮的DNN架构对于在边缘安全地部署DNN至关重要。该工作提出了一种称为XploreNAS的两阶段算法-硬件联合优化方法，该方法为非理想的纵横制平台寻找硬件高效且相对健壮的神经结构。我们使用一次神经体系结构搜索(NAS)方法来训练一个具有交叉开关感知的大型超网，并从中采样具有敌意健壮性的子网，从而保持具有竞争力的硬件效率。我们使用基准数据集(SVHN、CIFAR10和CIFAR100)在Crosbar上的实验表明，相对于接受Crosbar感知对抗性训练的基线ResNet-18模型，搜索到的子网的对抗健壮性提高了约8-16%。我们使用Neurosim工具对我们的健壮的能量延迟面积产品(EDAP)子网进行了基准测试，发现在额外的硬件效率驱动的优化下，这些子网获得的EDAP比ResNet-18基准低约1.5-1.6倍。



## **30. Visually Adversarial Attacks and Defenses in the Physical World: A Survey**

物理世界中的视觉对抗性攻击和防御：综述 cs.CV

**SubmitDate**: 2023-04-15    [abs](http://arxiv.org/abs/2211.01671v4) [paper-pdf](http://arxiv.org/pdf/2211.01671v4)

**Authors**: Xingxing Wei, Bangzheng Pu, Jiefan Lu, Baoyuan Wu

**Abstract**: Although Deep Neural Networks (DNNs) have been widely applied in various real-world scenarios, they are vulnerable to adversarial examples. The current adversarial attacks in computer vision can be divided into digital attacks and physical attacks according to their different attack forms. Compared with digital attacks, which generate perturbations in the digital pixels, physical attacks are more practical in the real world. Owing to the serious security problem caused by physically adversarial examples, many works have been proposed to evaluate the physically adversarial robustness of DNNs in the past years. In this paper, we summarize a survey versus the current physically adversarial attacks and physically adversarial defenses in computer vision. To establish a taxonomy, we organize the current physical attacks from attack tasks, attack forms, and attack methods, respectively. Thus, readers can have a systematic knowledge of this topic from different aspects. For the physical defenses, we establish the taxonomy from pre-processing, in-processing, and post-processing for the DNN models to achieve full coverage of the adversarial defenses. Based on the above survey, we finally discuss the challenges of this research field and further outlook on the future direction.

摘要: 尽管深度神经网络(DNN)已被广泛应用于各种现实场景中，但它们很容易受到对手例子的影响。根据攻击形式的不同，目前计算机视觉中的对抗性攻击可分为数字攻击和物理攻击。与在数字像素中产生扰动的数字攻击相比，物理攻击在现实世界中更实用。由于物理对抗实例带来了严重的安全问题，在过去的几年里，人们已经提出了许多工作来评估DNN的物理对抗健壮性。本文对当前计算机视觉中的身体对抗攻击和身体对抗防御进行了综述。为了建立分类，我们分别从攻击任务、攻击形式和攻击方法三个方面对当前的物理攻击进行了组织。因此，读者可以从不同的方面对这一主题有一个系统的了解。对于物理防御，我们从DNN模型的前处理、内处理和后处理三个方面建立了分类，以实现对抗性防御的全覆盖。在上述调查的基础上，我们最后讨论了该研究领域面临的挑战和对未来方向的进一步展望。



## **31. Combining Generators of Adversarial Malware Examples to Increase Evasion Rate**

组合敌意恶意软件示例的生成器以提高逃避率 cs.CR

9 pages, 5 figures, 2 tables. Under review

**SubmitDate**: 2023-04-14    [abs](http://arxiv.org/abs/2304.07360v1) [paper-pdf](http://arxiv.org/pdf/2304.07360v1)

**Authors**: Matouš Kozák, Martin Jureček

**Abstract**: Antivirus developers are increasingly embracing machine learning as a key component of malware defense. While machine learning achieves cutting-edge outcomes in many fields, it also has weaknesses that are exploited by several adversarial attack techniques. Many authors have presented both white-box and black-box generators of adversarial malware examples capable of bypassing malware detectors with varying success. We propose to combine contemporary generators in order to increase their potential. Combining different generators can create more sophisticated adversarial examples that are more likely to evade anti-malware tools. We demonstrated this technique on five well-known generators and recorded promising results. The best-performing combination of AMG-random and MAB-Malware generators achieved an average evasion rate of 15.9% against top-tier antivirus products. This represents an average improvement of more than 36% and 627% over using only the AMG-random and MAB-Malware generators, respectively. The generator that benefited the most from having another generator follow its procedure was the FGSM injection attack, which improved the evasion rate on average between 91.97% and 1,304.73%, depending on the second generator used. These results demonstrate that combining different generators can significantly improve their effectiveness against leading antivirus programs.

摘要: 反病毒开发人员越来越多地将机器学习作为恶意软件防御的关键组件。虽然机器学习在许多领域取得了尖端成果，但它也有被几种对抗性攻击技术利用的弱点。许多作者都提出了敌意恶意软件的白盒和黑盒生成器的例子，这些例子能够绕过恶意软件检测器，取得了不同的成功。我们建议将当代发电机结合起来，以增加它们的潜力。组合不同的生成器可以创建更复杂的敌意示例，这些示例更有可能逃避反恶意软件工具。我们在五个著名的发电机上演示了这项技术，并记录了有希望的结果。AMG-RANDOM和MAB-恶意软件生成器的最佳组合对顶级反病毒产品的平均逃避率为15.9%。这意味着分别比只使用AMG-RANDOM和MAB-恶意软件生成器的平均性能提高了36%和627%以上。从让另一个生成器遵循其过程中受益最大的生成器是FGSM注入攻击，它根据使用的第二个生成器的不同，将逃避率平均提高了91.97%到1304.73%。这些结果表明，组合不同的生成器可以显著提高它们对抗领先反病毒程序的有效性。



## **32. Pool Inference Attacks on Local Differential Privacy: Quantifying the Privacy Guarantees of Apple's Count Mean Sketch in Practice**

局部差分隐私的Pool推理攻击：Apple计数均值素描的隐私保证量化实践 cs.CR

Published at USENIX Security 2022. This is the full version, please  cite the USENIX version (see journal reference field)

**SubmitDate**: 2023-04-14    [abs](http://arxiv.org/abs/2304.07134v1) [paper-pdf](http://arxiv.org/pdf/2304.07134v1)

**Authors**: Andrea Gadotti, Florimond Houssiau, Meenatchi Sundaram Muthu Selva Annamalai, Yves-Alexandre de Montjoye

**Abstract**: Behavioral data generated by users' devices, ranging from emoji use to pages visited, are collected at scale to improve apps and services. These data, however, contain fine-grained records and can reveal sensitive information about individual users. Local differential privacy has been used by companies as a solution to collect data from users while preserving privacy. We here first introduce pool inference attacks, where an adversary has access to a user's obfuscated data, defines pools of objects, and exploits the user's polarized behavior in multiple data collections to infer the user's preferred pool. Second, we instantiate this attack against Count Mean Sketch, a local differential privacy mechanism proposed by Apple and deployed in iOS and Mac OS devices, using a Bayesian model. Using Apple's parameters for the privacy loss $\varepsilon$, we then consider two specific attacks: one in the emojis setting -- where an adversary aims at inferring a user's preferred skin tone for emojis -- and one against visited websites -- where an adversary wants to learn the political orientation of a user from the news websites they visit. In both cases, we show the attack to be much more effective than a random guess when the adversary collects enough data. We find that users with high polarization and relevant interest are significantly more vulnerable, and we show that our attack is well-calibrated, allowing the adversary to target such vulnerable users. We finally validate our results for the emojis setting using user data from Twitter. Taken together, our results show that pool inference attacks are a concern for data protected by local differential privacy mechanisms with a large $\varepsilon$, emphasizing the need for additional technical safeguards and the need for more research on how to apply local differential privacy for multiple collections.

摘要: 用户设备产生的行为数据，从表情符号的使用到访问的页面，都会大规模收集，以改进应用程序和服务。然而，这些数据包含细粒度的记录，可能会泄露有关个人用户的敏感信息。本地差异隐私已被公司用作在保护隐私的同时收集用户数据的解决方案。这里我们首先引入池推理攻击，在这种攻击中，敌手可以访问用户的混淆数据，定义对象池，并利用用户在多个数据集合中的两极分化行为来推断用户的首选池。其次，我们使用贝叶斯模型实例化了针对Count Mean Sketch的攻击，Count Mean Sketch是Apple提出的一种本地差异隐私机制，部署在iOS和Mac OS设备上。使用苹果的隐私损失参数，然后我们考虑了两种具体的攻击：一种是在表情符号设置中--对手的目标是推断用户喜欢的表情符号的肤色--另一种是针对访问的网站--对手想要从用户访问的新闻网站了解用户的政治倾向。在这两种情况下，我们都表明，当对手收集到足够的数据时，攻击比随机猜测要有效得多。我们发现，具有高极化和相关兴趣的用户明显更容易受到攻击，并且我们表明我们的攻击经过了很好的校准，允许对手针对这些易受攻击的用户。最后，我们使用Twitter上的用户数据验证了表情符号设置的结果。总而言之，我们的结果表明，池推理攻击是对受本地差异隐私机制保护的数据的关注，需要花费很大的$\varepsilon$，这强调了需要额外的技术保障，以及需要更多的研究如何将本地差异隐私应用于多个集合。



## **33. Interpretability is a Kind of Safety: An Interpreter-based Ensemble for Adversary Defense**

可解释性是一种安全：一种基于口译员的对手防御合奏 cs.LG

10 pages, accepted to KDD'20

**SubmitDate**: 2023-04-14    [abs](http://arxiv.org/abs/2304.06919v1) [paper-pdf](http://arxiv.org/pdf/2304.06919v1)

**Authors**: Jingyuan Wang, Yufan Wu, Mingxuan Li, Xin Lin, Junjie Wu, Chao Li

**Abstract**: While having achieved great success in rich real-life applications, deep neural network (DNN) models have long been criticized for their vulnerability to adversarial attacks. Tremendous research efforts have been dedicated to mitigating the threats of adversarial attacks, but the essential trait of adversarial examples is not yet clear, and most existing methods are yet vulnerable to hybrid attacks and suffer from counterattacks. In light of this, in this paper, we first reveal a gradient-based correlation between sensitivity analysis-based DNN interpreters and the generation process of adversarial examples, which indicates the Achilles's heel of adversarial attacks and sheds light on linking together the two long-standing challenges of DNN: fragility and unexplainability. We then propose an interpreter-based ensemble framework called X-Ensemble for robust adversary defense. X-Ensemble adopts a novel detection-rectification process and features in building multiple sub-detectors and a rectifier upon various types of interpretation information toward target classifiers. Moreover, X-Ensemble employs the Random Forests (RF) model to combine sub-detectors into an ensemble detector for adversarial hybrid attacks defense. The non-differentiable property of RF further makes it a precious choice against the counterattack of adversaries. Extensive experiments under various types of state-of-the-art attacks and diverse attack scenarios demonstrate the advantages of X-Ensemble to competitive baseline methods.

摘要: 尽管深度神经网络(DNN)模型在丰富的现实应用中取得了巨大的成功，但长期以来一直因其易受对手攻击而受到批评。为了缓解对抗性攻击的威胁，人们进行了大量的研究工作，但对抗性例子的本质特征尚不清楚，而且现有的大多数方法仍然容易受到混合攻击和反击。有鉴于此，本文首先揭示了基于敏感度分析的DNN解释器与敌意实例生成过程之间的梯度关系，这表明了敌意攻击的致命弱点，并有助于将DNN的两个长期挑战：脆弱性和不可解释性联系在一起。然后，我们提出了一个基于解释器的集成框架，称为X-集成，用于强健的对手防御。X-Ensymble采用了一种新颖的检测-纠正过程，其特点是根据目标分类器的各种解释信息构建多个子检测器和一个校正器。此外，X-EnSemble采用随机森林(RF)模型将子检测器组合为集成检测器，用于对抗性混合攻击防御。射频的不可微性进一步使其成为对抗对手反击的宝贵选择。在各种最先进的攻击和不同的攻击场景下进行的广泛实验证明了X-Ensymble相对于竞争基线方法的优势。



## **34. Robust Multivariate Time-Series Forecasting: Adversarial Attacks and Defense Mechanisms**

稳健的多变量时间序列预测：对抗性攻击与防御机制 cs.LG

**SubmitDate**: 2023-04-14    [abs](http://arxiv.org/abs/2207.09572v3) [paper-pdf](http://arxiv.org/pdf/2207.09572v3)

**Authors**: Linbo Liu, Youngsuk Park, Trong Nghia Hoang, Hilaf Hasson, Jun Huan

**Abstract**: This work studies the threats of adversarial attack on multivariate probabilistic forecasting models and viable defense mechanisms. Our studies discover a new attack pattern that negatively impact the forecasting of a target time series via making strategic, sparse (imperceptible) modifications to the past observations of a small number of other time series. To mitigate the impact of such attack, we have developed two defense strategies. First, we extend a previously developed randomized smoothing technique in classification to multivariate forecasting scenarios. Second, we develop an adversarial training algorithm that learns to create adversarial examples and at the same time optimizes the forecasting model to improve its robustness against such adversarial simulation. Extensive experiments on real-world datasets confirm that our attack schemes are powerful and our defense algorithms are more effective compared with baseline defense mechanisms.

摘要: 本文研究了对抗性攻击对多变量概率预测模型的威胁和可行的防御机制。我们的研究发现了一种新的攻击模式，它通过对过去对少量其他时间序列的观察进行战略性的稀疏(不可察觉的)修改，对目标时间序列的预测产生负面影响。为了减轻此类攻击的影响，我们制定了两种防御策略。首先，我们将以前开发的随机平滑分类技术扩展到多变量预测场景。其次，我们开发了一种对抗性训练算法，该算法学习创建对抗性示例，同时优化预测模型，以提高其对此类对抗性模拟的健壮性。在真实数据集上的广泛实验证实了我们的攻击方案是强大的，我们的防御算法比基线防御机制更有效。



## **35. Generating Adversarial Examples with Better Transferability via Masking Unimportant Parameters of Surrogate Model**

通过屏蔽代理模型的不重要参数生成可移植性更好的对抗性实例 cs.LG

Accepted at 2023 International Joint Conference on Neural Networks  (IJCNN)

**SubmitDate**: 2023-04-14    [abs](http://arxiv.org/abs/2304.06908v1) [paper-pdf](http://arxiv.org/pdf/2304.06908v1)

**Authors**: Dingcheng Yang, Wenjian Yu, Zihao Xiao, Jiaqi Luo

**Abstract**: Deep neural networks (DNNs) have been shown to be vulnerable to adversarial examples. Moreover, the transferability of the adversarial examples has received broad attention in recent years, which means that adversarial examples crafted by a surrogate model can also attack unknown models. This phenomenon gave birth to the transfer-based adversarial attacks, which aim to improve the transferability of the generated adversarial examples. In this paper, we propose to improve the transferability of adversarial examples in the transfer-based attack via masking unimportant parameters (MUP). The key idea in MUP is to refine the pretrained surrogate models to boost the transfer-based attack. Based on this idea, a Taylor expansion-based metric is used to evaluate the parameter importance score and the unimportant parameters are masked during the generation of adversarial examples. This process is simple, yet can be naturally combined with various existing gradient-based optimizers for generating adversarial examples, thus further improving the transferability of the generated adversarial examples. Extensive experiments are conducted to validate the effectiveness of the proposed MUP-based methods.

摘要: 深度神经网络(DNN)已被证明容易受到敌意例子的攻击。此外，对抗性实例的可转移性近年来受到了广泛的关注，这意味着由代理模型构造的对抗性实例也可以攻击未知模型。这一现象催生了基于迁移的对抗性攻击，其目的是提高生成的对抗性实例的可转移性。在本文中，我们提出了通过掩蔽不重要参数(MUP)来提高基于传输的攻击中敌意实例的可转移性。MUP的关键思想是改进预先训练的代理模型，以增强基于传输的攻击。基于这一思想，使用基于泰勒展开的度量来评估参数重要性得分，并在对抗性实例的生成过程中屏蔽不重要的参数。这个过程简单，但可以自然地与现有的各种基于梯度的优化器相结合来生成对抗性实例，从而进一步提高生成的对抗性实例的可转移性。大量的实验验证了所提出的基于MUP的方法的有效性。



## **36. Don't Knock! Rowhammer at the Backdoor of DNN Models**

别敲门！DNN模型的后门Rowhammer cs.LG

2023 53rd Annual IEEE/IFIP International Conference on Dependable  Systems and Networks (DSN)

**SubmitDate**: 2023-04-13    [abs](http://arxiv.org/abs/2110.07683v3) [paper-pdf](http://arxiv.org/pdf/2110.07683v3)

**Authors**: M. Caner Tol, Saad Islam, Andrew J. Adiletta, Berk Sunar, Ziming Zhang

**Abstract**: State-of-the-art deep neural networks (DNNs) have been proven to be vulnerable to adversarial manipulation and backdoor attacks. Backdoored models deviate from expected behavior on inputs with predefined triggers while retaining performance on clean data. Recent works focus on software simulation of backdoor injection during the inference phase by modifying network weights, which we find often unrealistic in practice due to restrictions in hardware.   In contrast, in this work for the first time, we present an end-to-end backdoor injection attack realized on actual hardware on a classifier model using Rowhammer as the fault injection method. To this end, we first investigate the viability of backdoor injection attacks in real-life deployments of DNNs on hardware and address such practical issues in hardware implementation from a novel optimization perspective. We are motivated by the fact that vulnerable memory locations are very rare, device-specific, and sparsely distributed. Consequently, we propose a novel network training algorithm based on constrained optimization to achieve a realistic backdoor injection attack in hardware. By modifying parameters uniformly across the convolutional and fully-connected layers as well as optimizing the trigger pattern together, we achieve state-of-the-art attack performance with fewer bit flips. For instance, our method on a hardware-deployed ResNet-20 model trained on CIFAR-10 achieves over 89% test accuracy and 92% attack success rate by flipping only 10 out of 2.2 million bits.

摘要: 最先进的深度神经网络(DNN)已被证明容易受到对手操纵和后门攻击。回溯模型偏离了具有预定义触发器的输入的预期行为，同时保持了对干净数据的性能。最近的工作集中在通过修改网络权重来模拟推理阶段的后门注入，但由于硬件的限制，这在实践中往往是不现实的。相反，在这项工作中，我们首次提出了一种端到端的后门注入攻击，在以Rowhammer作为故障注入方法的分类器模型上在实际硬件上实现。为此，我们首先研究了DNN在硬件上的实际部署中后门注入攻击的可行性，并从一个新的优化角度解决了这些硬件实现中的实际问题。我们的动机是，易受攻击的内存位置非常罕见，特定于设备，并且分布稀疏。因此，我们提出了一种新的基于约束优化的网络训练算法，在硬件上实现了逼真的后门注入攻击。通过统一修改卷积层和全连通层的参数，以及一起优化触发模式，我们以更少的比特翻转实现了最先进的攻击性能。例如，我们的方法在硬件部署的ResNet-20模型上训练CIFAR-10，通过在220万比特中仅翻转10比特，获得了超过89%的测试准确率和92%的攻击成功率。



## **37. False Claims against Model Ownership Resolution**

针对所有权解决方案范本的虚假索赔 cs.CR

13pages,3 figures

**SubmitDate**: 2023-04-13    [abs](http://arxiv.org/abs/2304.06607v1) [paper-pdf](http://arxiv.org/pdf/2304.06607v1)

**Authors**: Jian Liu, Rui Zhang, Sebastian Szyller, Kui Ren, N. Asokan

**Abstract**: Deep neural network (DNN) models are valuable intellectual property of model owners, constituting a competitive advantage. Therefore, it is crucial to develop techniques to protect against model theft. Model ownership resolution (MOR) is a class of techniques that can deter model theft. A MOR scheme enables an accuser to assert an ownership claim for a suspect model by presenting evidence, such as a watermark or fingerprint, to show that the suspect model was stolen or derived from a source model owned by the accuser. Most of the existing MOR schemes prioritize robustness against malicious suspects, ensuring that the accuser will win if the suspect model is indeed a stolen model.   In this paper, we show that common MOR schemes in the literature are vulnerable to a different, equally important but insufficiently explored, robustness concern: a malicious accuser. We show how malicious accusers can successfully make false claims against independent suspect models that were not stolen. Our core idea is that a malicious accuser can deviate (without detection) from the specified MOR process by finding (transferable) adversarial examples that successfully serve as evidence against independent suspect models. To this end, we first generalize the procedures of common MOR schemes and show that, under this generalization, defending against false claims is as challenging as preventing (transferable) adversarial examples. Via systematic empirical evaluation we demonstrate that our false claim attacks always succeed in all prominent MOR schemes with realistic configurations, including against a real-world model: Amazon's Rekognition API.

摘要: 深度神经网络(DNN)模型是模型所有者宝贵的知识产权，构成了竞争优势。因此，开发防止模型盗窃的技术至关重要。模型所有权解析(MOR)是一类能够阻止模型被盗的技术。MOR方案使原告能够通过出示证据(例如水印或指纹)来断言可疑模型的所有权主张，以显示可疑模型是被盗的或从原告拥有的源模型派生的。现有的大多数MOR方案都优先考虑针对恶意嫌疑人的健壮性，确保在可疑模型确实是被盗模型的情况下原告获胜。在这篇文章中，我们证明了文献中常见的MOR方案容易受到另一个同样重要但未被充分研究的健壮性问题的影响：恶意指控者。我们展示了恶意原告如何成功地对未被窃取的独立可疑模型做出虚假声明。我们的核心思想是，恶意指控者可以通过找到(可转移的)对抗性例子来偏离指定的MOR过程(而不被检测到)，这些例子成功地充当了针对独立嫌疑人模型的证据。为此，我们首先推广了常见MOR方案的步骤，并证明在这种推广下，对虚假声明的防御与防止(可转移)对抗性例子一样具有挑战性。通过系统的经验评估，我们证明了我们的虚假声明攻击在所有具有现实配置的著名MOR方案中总是成功的，包括针对真实世界的模型：亚马逊的Rekognition API。



## **38. EGC: Image Generation and Classification via a Diffusion Energy-Based Model**

EGC：基于扩散能量模型的图像生成和分类 cs.CV

**SubmitDate**: 2023-04-13    [abs](http://arxiv.org/abs/2304.02012v3) [paper-pdf](http://arxiv.org/pdf/2304.02012v3)

**Authors**: Qiushan Guo, Chuofan Ma, Yi Jiang, Zehuan Yuan, Yizhou Yu, Ping Luo

**Abstract**: Learning image classification and image generation using the same set of network parameters is a challenging problem. Recent advanced approaches perform well in one task often exhibit poor performance in the other. This work introduces an energy-based classifier and generator, namely EGC, which can achieve superior performance in both tasks using a single neural network. Unlike a conventional classifier that outputs a label given an image (i.e., a conditional distribution $p(y|\mathbf{x})$), the forward pass in EGC is a classifier that outputs a joint distribution $p(\mathbf{x},y)$, enabling an image generator in its backward pass by marginalizing out the label $y$. This is done by estimating the energy and classification probability given a noisy image in the forward pass, while denoising it using the score function estimated in the backward pass. EGC achieves competitive generation results compared with state-of-the-art approaches on ImageNet-1k, CelebA-HQ and LSUN Church, while achieving superior classification accuracy and robustness against adversarial attacks on CIFAR-10. This work represents the first successful attempt to simultaneously excel in both tasks using a single set of network parameters. We believe that EGC bridges the gap between discriminative and generative learning.

摘要: 使用相同的网络参数学习图像分类和图像生成是一个具有挑战性的问题。最近的高级方法在一项任务中表现良好，但在另一项任务中往往表现不佳。这项工作介绍了一种基于能量的分类器和生成器，即EGC，它可以使用单个神经网络在这两个任务中获得优越的性能。与输出给定图像的标签(即，条件分布$p(y|\mathbf{x})$)的传统分类器不同，EGC中的前向通道是输出联合分布$p(\mathbf{x}，y)$的分类器，从而使图像生成器在其后向通道中通过边缘化标签$y$来实现。这是通过在前传中给出噪声图像的情况下估计能量和分类概率来完成的，同时使用在后传中估计的得分函数来对其进行去噪。EGC在ImageNet-1k、CelebA-HQ和LSUN Church上实现了与最先进的方法相比具有竞争力的生成结果，同时在CIFAR-10上获得了卓越的分类准确性和对对手攻击的稳健性。这项工作代表了首次成功尝试使用一组网络参数同时在两项任务中脱颖而出。我们认为，EGC弥合了歧视性学习和生成性学习之间的差距。



## **39. Certified Zeroth-order Black-Box Defense with Robust UNet Denoiser**

经过认证的零阶黑匣子防御，具有坚固的UNT消噪器 cs.CV

**SubmitDate**: 2023-04-13    [abs](http://arxiv.org/abs/2304.06430v1) [paper-pdf](http://arxiv.org/pdf/2304.06430v1)

**Authors**: Astha Verma, Siddhesh Bangar, A V Subramanyam, Naman Lal, Rajiv Ratn Shah, Shin'ichi Satoh

**Abstract**: Certified defense methods against adversarial perturbations have been recently investigated in the black-box setting with a zeroth-order (ZO) perspective. However, these methods suffer from high model variance with low performance on high-dimensional datasets due to the ineffective design of the denoiser and are limited in their utilization of ZO techniques. To this end, we propose a certified ZO preprocessing technique for removing adversarial perturbations from the attacked image in the black-box setting using only model queries. We propose a robust UNet denoiser (RDUNet) that ensures the robustness of black-box models trained on high-dimensional datasets. We propose a novel black-box denoised smoothing (DS) defense mechanism, ZO-RUDS, by prepending our RDUNet to the black-box model, ensuring black-box defense. We further propose ZO-AE-RUDS in which RDUNet followed by autoencoder (AE) is prepended to the black-box model. We perform extensive experiments on four classification datasets, CIFAR-10, CIFAR-10, Tiny Imagenet, STL-10, and the MNIST dataset for image reconstruction tasks. Our proposed defense methods ZO-RUDS and ZO-AE-RUDS beat SOTA with a huge margin of $35\%$ and $9\%$, for low dimensional (CIFAR-10) and with a margin of $20.61\%$ and $23.51\%$ for high-dimensional (STL-10) datasets, respectively.

摘要: 针对对抗性扰动的认证防御方法最近在零阶(ZO)视角的黑盒环境中被研究。然而，由于去噪器的设计不合理，这些方法在高维数据集上存在模型方差大、性能低的问题，限制了ZO技术的应用。为此，我们提出了一种经过验证的ZO预处理技术，用于在仅使用模型查询的情况下从黑盒环境中去除攻击图像中的对抗性扰动。我们提出了一种稳健的UNET去噪器(RDUNet)，以确保在高维数据集上训练的黑盒模型的稳健性。提出了一种新的黑盒去噪平滑防御机制ZO-RUDS，将RDUNet加入到黑盒模型中，保证了黑盒防御。我们进一步提出了ZO-AE-RUDS，其中RDUNet后跟自动编码器(AE)优先于黑盒模型。我们在CIFAR-10、CIFAR-10、Tiny Imagenet、STL-10和MNIST四个分类数据集上进行了大量的实验，用于图像重建任务。我们提出的防御方法ZO-RUDS和ZO-AE-RUDS对于低维数据集(CIFAR-10)分别以35美元和9美元的巨大优势击败了SOTA，而对于高维数据集(STL-10)分别以20.61美元和23.51美元的优势击败了SOTA。



## **40. How to Sign Quantum Messages**

如何对量子消息进行签名 quant-ph

22 pages

**SubmitDate**: 2023-04-13    [abs](http://arxiv.org/abs/2304.06325v1) [paper-pdf](http://arxiv.org/pdf/2304.06325v1)

**Authors**: Mohammed Barhoush, Louis Salvail

**Abstract**: Signing quantum messages has been shown to be impossible even under computational assumptions. We show that this result can be circumvented by relying on verification keys that change with time or that are large quantum states. Correspondingly, we give two new approaches to sign quantum information. The first approach assumes quantum-secure one-way functions (QOWF) to obtain a time-dependent signature scheme where the algorithms take into account time. The keys are classical but the verification key needs to be continually updated. The second construction uses fixed quantum verification keys and achieves information-theoretic secure signatures against adversaries with bounded quantum memory i.e. in the bounded quantum storage model. Furthermore, we apply our time-dependent signatures to authenticate keys in quantum public key encryption schemes and achieve indistinguishability under chosen quantum key and ciphertext attack (qCKCA).

摘要: 已经证明，即使在计算假设下，签署量子消息也是不可能的。我们表明，这一结果可以通过依赖随时间变化的验证密钥或大量子状态来规避。相应地，我们给出了两种新的量子信息签名方法。第一种方法采用量子安全的单向函数(QOWF)来获得依赖于时间的签名方案，其中算法考虑了时间。密钥是经典的，但验证密钥需要不断更新。第二种构造使用固定的量子验证密钥，针对具有有限量子存储的攻击者实现信息论安全签名，即在有限量子存储模型下。此外，我们将我们的依赖时间的签名应用于量子公钥加密方案中的密钥认证，并且在选择量子密钥和密文攻击(QCKCA)下实现了不可区分。



## **41. Multi-Glimpse Network: A Robust and Efficient Classification Architecture based on Recurrent Downsampled Attention**

多瞥网络：一种基于循环下采样注意的稳健高效分类体系结构 cs.CV

Accepted at BMVC 2021

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2111.02018v2) [paper-pdf](http://arxiv.org/pdf/2111.02018v2)

**Authors**: Sia Huat Tan, Runpei Dong, Kaisheng Ma

**Abstract**: Most feedforward convolutional neural networks spend roughly the same efforts for each pixel. Yet human visual recognition is an interaction between eye movements and spatial attention, which we will have several glimpses of an object in different regions. Inspired by this observation, we propose an end-to-end trainable Multi-Glimpse Network (MGNet) which aims to tackle the challenges of high computation and the lack of robustness based on recurrent downsampled attention mechanism. Specifically, MGNet sequentially selects task-relevant regions of an image to focus on and then adaptively combines all collected information for the final prediction. MGNet expresses strong resistance against adversarial attacks and common corruptions with less computation. Also, MGNet is inherently more interpretable as it explicitly informs us where it focuses during each iteration. Our experiments on ImageNet100 demonstrate the potential of recurrent downsampled attention mechanisms to improve a single feedforward manner. For example, MGNet improves 4.76% accuracy on average in common corruptions with only 36.9% computational cost. Moreover, while the baseline incurs an accuracy drop to 7.6%, MGNet manages to maintain 44.2% accuracy in the same PGD attack strength with ResNet-50 backbone. Our code is available at https://github.com/siahuat0727/MGNet.

摘要: 大多数前馈卷积神经网络对于每个像素花费大致相同的努力。然而，人类的视觉识别是眼球运动和空间注意力之间的相互作用，我们会对不同地区的一个物体有几次瞥见。受此启发，我们提出了一种端到端可训练的多掠影网络(MGNet)，旨在解决基于循环下采样注意机制的高计算量和健壮性不足的挑战。具体地说，MGNet按顺序选择图像中与任务相关的区域进行聚焦，然后自适应地组合所有收集的信息以进行最终预测。MGNet以较少的运算量对敌意攻击和常见的腐败表现出很强的抵抗力。此外，MGNet天生更易于解释，因为它在每次迭代中明确地通知我们它关注的地方。我们在ImageNet100上的实验证明了循环下采样注意机制改善单一前馈方式的潜力。例如，MGNet在常见的腐败问题上平均提高了4.76%的准确率，而计算代价仅为36.9%。此外，当基线导致准确率下降到7.6%时，MGNet在与ResNet-50主干相同的PGD攻击强度下设法保持44.2%的准确率。我们的代码可以在https://github.com/siahuat0727/MGNet.上找到



## **42. Identification of Systematic Errors of Image Classifiers on Rare Subgroups**

稀有子群上图像分类器系统误差的辨识 cs.CV

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2303.05072v2) [paper-pdf](http://arxiv.org/pdf/2303.05072v2)

**Authors**: Jan Hendrik Metzen, Robin Hutmacher, N. Grace Hua, Valentyn Boreiko, Dan Zhang

**Abstract**: Despite excellent average-case performance of many image classifiers, their performance can substantially deteriorate on semantically coherent subgroups of the data that were under-represented in the training data. These systematic errors can impact both fairness for demographic minority groups as well as robustness and safety under domain shift. A major challenge is to identify such subgroups with subpar performance when the subgroups are not annotated and their occurrence is very rare. We leverage recent advances in text-to-image models and search in the space of textual descriptions of subgroups ("prompts") for subgroups where the target model has low performance on the prompt-conditioned synthesized data. To tackle the exponentially growing number of subgroups, we employ combinatorial testing. We denote this procedure as PromptAttack as it can be interpreted as an adversarial attack in a prompt space. We study subgroup coverage and identifiability with PromptAttack in a controlled setting and find that it identifies systematic errors with high accuracy. Thereupon, we apply PromptAttack to ImageNet classifiers and identify novel systematic errors on rare subgroups.

摘要: 尽管许多图像分类器的平均情况性能很好，但在训练数据中表示不足的数据的语义连贯子组上，它们的性能可能会显著恶化。这些系统性错误既会影响人口少数群体的公平性，也会影响领域转移下的稳健性和安全性。一个主要的挑战是在子组没有被注释并且它们的出现非常罕见的情况下，识别这样的子组具有低于标准的性能。我们利用文本到图像模型中的最新进展，并在目标模型对提示条件合成数据的性能较低的子组的子组(提示)的文本描述空间中进行搜索。为了解决指数级增长的子组数量，我们使用了组合测试。我们将这个过程表示为PromptAttack，因为它可以解释为提示空间中的对抗性攻击。我们在受控环境下研究了PromptAttack的子组复盖率和可识别性，发现它识别系统错误的准确率很高。于是，我们将PromptAttack应用于ImageNet分类器，并在稀有子群上识别出新的系统误差。



## **43. Optimal Detector Placement in Networked Control Systems under Cyber-attacks with Applications to Power Networks**

网络攻击下网络控制系统检测器的最优配置及其在电网中的应用 eess.SY

7 pages, 4 figures, accepted to IFAC 2023

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2304.05710v1) [paper-pdf](http://arxiv.org/pdf/2304.05710v1)

**Authors**: Anh Tung Nguyen, Sribalaji C. Anand, André M. H. Teixeira, Alexander Medvedev

**Abstract**: This paper proposes a game-theoretic method to address the problem of optimal detector placement in a networked control system under cyber-attacks. The networked control system is composed of interconnected agents where each agent is regulated by its local controller over unprotected communication, which leaves the system vulnerable to malicious cyber-attacks. To guarantee a given local performance, the defender optimally selects a single agent on which to place a detector at its local controller with the purpose of detecting cyber-attacks. On the other hand, an adversary optimally chooses a single agent on which to conduct a cyber-attack on its input with the aim of maximally worsening the local performance while remaining stealthy to the defender. First, we present a necessary and sufficient condition to ensure that the maximal attack impact on the local performance is bounded, which restricts the possible actions of the defender to a subset of available agents. Then, by considering the maximal attack impact on the local performance as a game payoff, we cast the problem of finding optimal actions of the defender and the adversary as a zero-sum game. Finally, with the possible action sets of the defender and the adversary, an algorithm is devoted to determining the Nash equilibria of the zero-sum game that yield the optimal detector placement. The proposed method is illustrated on an IEEE benchmark for power systems.

摘要: 针对网络控制系统中检测器的最优配置问题，提出了一种基于博弈论的方法。网络控制系统由相互连接的代理组成，其中每个代理由其本地控制器通过不受保护的通信进行管理，这使得系统容易受到恶意网络攻击。为了保证给定的本地性能，防御者最优化地选择单个代理，在其本地控制器上放置检测器，以检测网络攻击。另一方面，对手最优地选择一个代理对其输入进行网络攻击，目的是最大限度地恶化本地性能，同时保持对防御者的隐身。首先，我们给出了确保攻击对局部性能的最大影响是有界的充要条件，该条件将防御者可能的行为限制在可用代理的子集上。然后，通过考虑攻击对局部性能的最大影响作为博弈收益，将寻找防御者和对手的最优行动的问题转化为零和博弈。最后，利用防御者和对手可能的行动集，给出了一个算法来确定零和博弈中产生最优检测器配置的纳什均衡。以IEEE电力系统基准为例，说明了该方法的有效性。



## **44. SoK: Certified Robustness for Deep Neural Networks**

SOK：深度神经网络认证的健壮性 cs.LG

To appear at 2023 IEEE Symposium on Security and Privacy (SP)  (Version 8); include recent progress till Apr 2023 in Version 9; 14 pages for  the main text; benchmark & tool website:  http://sokcertifiedrobustness.github.io/

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2009.04131v9) [paper-pdf](http://arxiv.org/pdf/2009.04131v9)

**Authors**: Linyi Li, Tao Xie, Bo Li

**Abstract**: Great advances in deep neural networks (DNNs) have led to state-of-the-art performance on a wide range of tasks. However, recent studies have shown that DNNs are vulnerable to adversarial attacks, which have brought great concerns when deploying these models to safety-critical applications such as autonomous driving. Different defense approaches have been proposed against adversarial attacks, including: a) empirical defenses, which can usually be adaptively attacked again without providing robustness certification; and b) certifiably robust approaches, which consist of robustness verification providing the lower bound of robust accuracy against any attacks under certain conditions and corresponding robust training approaches. In this paper, we systematize certifiably robust approaches and related practical and theoretical implications and findings. We also provide the first comprehensive benchmark on existing robustness verification and training approaches on different datasets. In particular, we 1) provide a taxonomy for the robustness verification and training approaches, as well as summarize the methodologies for representative algorithms, 2) reveal the characteristics, strengths, limitations, and fundamental connections among these approaches, 3) discuss current research progresses, theoretical barriers, main challenges, and future directions for certifiably robust approaches for DNNs, and 4) provide an open-sourced unified platform to evaluate 20+ representative certifiably robust approaches.

摘要: 深度神经网络(DNN)的巨大进步导致了在各种任务中最先进的性能。然而，最近的研究表明，DNN很容易受到对手攻击，这在将这些模型部署到自动驾驶等安全关键型应用时带来了极大的担忧。针对敌意攻击已经提出了不同的防御方法，包括：a)经验防御，通常可以在不提供健壮性证明的情况下自适应地再次攻击；b)可证明的健壮性方法，包括在一定条件下提供对任何攻击的健壮性精度下界的健壮性验证和相应的健壮训练方法。在这篇文章中，我们系统化了可证明的稳健方法以及相关的实践和理论意义和发现。我们还提供了关于不同数据集上现有稳健性验证和训练方法的第一个全面基准。具体地说，我们1)提供了健壮性验证和训练方法的分类，并总结了典型算法的方法论；2)揭示了这些方法的特点、优势、局限性和基本联系；3)讨论了当前的研究进展、理论障碍、主要挑战和未来的发展方向；4)提供了一个开源的统一平台来评估20多个具有代表性的可证健壮方法。



## **45. Generative Adversarial Networks-Driven Cyber Threat Intelligence Detection Framework for Securing Internet of Things**

产生式对抗网络驱动的物联网安全网络威胁情报检测框架 cs.CR

The paper is accepted and will be published in the IEEE DCOSS-IoT  2023 Conference Proceedings

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2304.05644v1) [paper-pdf](http://arxiv.org/pdf/2304.05644v1)

**Authors**: Mohamed Amine Ferrag, Djallel Hamouda, Merouane Debbah, Leandros Maglaras, Abderrahmane Lakas

**Abstract**: While the benefits of 6G-enabled Internet of Things (IoT) are numerous, providing high-speed, low-latency communication that brings new opportunities for innovation and forms the foundation for continued growth in the IoT industry, it is also important to consider the security challenges and risks associated with the technology. In this paper, we propose a two-stage intrusion detection framework for securing IoTs, which is based on two detectors. In the first stage, we propose an adversarial training approach using generative adversarial networks (GAN) to help the first detector train on robust features by supplying it with adversarial examples as validation sets. Consequently, the classifier would perform very well against adversarial attacks. Then, we propose a deep learning (DL) model for the second detector to identify intrusions. We evaluated the proposed approach's efficiency in terms of detection accuracy and robustness against adversarial attacks. Experiment results with a new cyber security dataset demonstrate the effectiveness of the proposed methodology in detecting both intrusions and persistent adversarial examples with a weighted avg of 96%, 95%, 95%, and 95% for precision, recall, f1-score, and accuracy, respectively.

摘要: 虽然支持6G的物联网(IoT)带来了许多好处，提供了高速、低延迟的通信，为创新带来了新的机遇，并为物联网行业的持续增长奠定了基础，但考虑与该技术相关的安全挑战和风险也很重要。本文提出了一种基于两个检测器的两阶段物联网安全入侵检测框架。在第一阶段，我们提出了一种使用生成性对抗性网络(GAN)的对抗性训练方法，通过向第一个检测器提供对抗性实例作为验证集来帮助它训练健壮特征。因此，分类器将在对抗对手攻击时表现得非常好。然后，我们提出了一种用于第二个检测器的深度学习模型来识别入侵。我们从检测准确率和对抗攻击的健壮性两个方面对该方法的效率进行了评估。在一个新的网络安全数据集上的实验结果表明，该方法在检测入侵和持续敌意示例方面的有效性，其准确率、召回率、F1得分和准确率的加权平均值分别为96%、95%、95%和95%。



## **46. Overload: Latency Attacks on Object Detection for Edge Devices**

过载：边缘设备对象检测的延迟攻击 cs.CV

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2304.05370v2) [paper-pdf](http://arxiv.org/pdf/2304.05370v2)

**Authors**: Erh-Chung Chen, Pin-Yu Chen, I-Hsin Chung, Che-rung Lee

**Abstract**: Nowadays, the deployment of deep learning based applications on edge devices is an essential task owing to the increasing demands on intelligent services. However, the limited computing resources on edge nodes make the models vulnerable to attacks, such that the predictions made by models are unreliable. In this paper, we investigate latency attacks on deep learning applications. Unlike common adversarial attacks for misclassification, the goal of latency attacks is to increase the inference time, which may stop applications from responding to the requests within a reasonable time. This kind of attack is ubiquitous for various applications, and we use object detection to demonstrate how such kind of attacks work. We also design a framework named Overload to generate latency attacks at scale. Our method is based on a newly formulated optimization problem and a novel technique, called spatial attention, to increase the inference time of object detection. We have conducted experiments using YOLOv5 models on Nvidia NX. The experimental results show that with latency attacks, the inference time of a single image can be increased ten times longer in reference to the normal setting. Moreover, comparing to existing methods, our attacking method is simpler and more effective.

摘要: 随着人们对智能服务需求的不断增长，在边缘设备上部署基于深度学习的应用是一项必不可少的任务。然而，边缘节点有限的计算资源使得模型容易受到攻击，使得模型做出的预测是不可靠的。本文研究了深度学习应用程序中的延迟攻击。与常见的误分类对抗性攻击不同，延迟攻击的目标是增加推理时间，这可能会使应用程序在合理的时间内停止对请求的响应。这种攻击在各种应用中普遍存在，我们使用对象检测来演示这种攻击是如何工作的。我们还设计了一个名为OverLoad的框架来生成大规模的延迟攻击。我们的方法基于一个新的优化问题和一种新的技术，称为空间注意，以增加目标检测的推理时间。我们已经在NVIDIA NX上使用YOLOv5模型进行了实验。实验结果表明，在延迟攻击的情况下，单幅图像的推理时间可以比正常设置增加十倍。此外，与现有的攻击方法相比，我们的攻击方法更简单有效。



## **47. Enhancing the Self-Universality for Transferable Targeted Attacks**

增强可转移定向攻击的自我普适性 cs.CV

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2209.03716v3) [paper-pdf](http://arxiv.org/pdf/2209.03716v3)

**Authors**: Zhipeng Wei, Jingjing Chen, Zuxuan Wu, Yu-Gang Jiang

**Abstract**: In this paper, we propose a novel transfer-based targeted attack method that optimizes the adversarial perturbations without any extra training efforts for auxiliary networks on training data. Our new attack method is proposed based on the observation that highly universal adversarial perturbations tend to be more transferable for targeted attacks. Therefore, we propose to make the perturbation to be agnostic to different local regions within one image, which we called as self-universality. Instead of optimizing the perturbations on different images, optimizing on different regions to achieve self-universality can get rid of using extra data. Specifically, we introduce a feature similarity loss that encourages the learned perturbations to be universal by maximizing the feature similarity between adversarial perturbed global images and randomly cropped local regions. With the feature similarity loss, our method makes the features from adversarial perturbations to be more dominant than that of benign images, hence improving targeted transferability. We name the proposed attack method as Self-Universality (SU) attack. Extensive experiments demonstrate that SU can achieve high success rates for transfer-based targeted attacks. On ImageNet-compatible dataset, SU yields an improvement of 12\% compared with existing state-of-the-art methods. Code is available at https://github.com/zhipeng-wei/Self-Universality.

摘要: 在本文中，我们提出了一种新的基于转移的定向攻击方法，该方法在不需要对训练数据进行任何额外训练的情况下优化了对抗性扰动。我们提出的新攻击方法是基于这样的观察，即高度普遍的对抗性扰动倾向于更可转移到定向攻击。因此，我们提出使微扰对同一图像内的不同局部区域是不可知的，我们称之为自普适性。与对不同图像上的扰动进行优化不同，通过对不同区域进行优化来实现自普适性，可以避免使用额外的数据。具体地说，我们引入了特征相似度损失，通过最大化对抗性扰动的全局图像和随机裁剪的局部区域之间的特征相似度，鼓励学习的扰动具有普遍性。在特征相似度损失的情况下，我们的方法使得来自对抗性扰动的特征比来自良性图像的特征更具优势，从而提高了目标可转移性。我们将所提出的攻击方法命名为自普适性攻击。广泛的实验证明，宿灿对基于转会的靶向攻击取得了很高的成功率。在与ImageNet兼容的数据集上，与现有的最先进方法相比，SU方法的性能提高了12%。代码可在https://github.com/zhipeng-wei/Self-Universality.上找到



## **48. On the Adversarial Inversion of Deep Biometric Representations**

关于深层生物特征表示的对抗性反转 cs.CV

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2304.05561v1) [paper-pdf](http://arxiv.org/pdf/2304.05561v1)

**Authors**: Gioacchino Tangari, Shreesh Keskar, Hassan Jameel Asghar, Dali Kaafar

**Abstract**: Biometric authentication service providers often claim that it is not possible to reverse-engineer a user's raw biometric sample, such as a fingerprint or a face image, from its mathematical (feature-space) representation. In this paper, we investigate this claim on the specific example of deep neural network (DNN) embeddings. Inversion of DNN embeddings has been investigated for explaining deep image representations or synthesizing normalized images. Existing studies leverage full access to all layers of the original model, as well as all possible information on the original dataset. For the biometric authentication use case, we need to investigate this under adversarial settings where an attacker has access to a feature-space representation but no direct access to the exact original dataset nor the original learned model. Instead, we assume varying degree of attacker's background knowledge about the distribution of the dataset as well as the original learned model (architecture and training process). In these cases, we show that the attacker can exploit off-the-shelf DNN models and public datasets, to mimic the behaviour of the original learned model to varying degrees of success, based only on the obtained representation and attacker's prior knowledge. We propose a two-pronged attack that first infers the original DNN by exploiting the model footprint on the embedding, and then reconstructs the raw data by using the inferred model. We show the practicality of the attack on popular DNNs trained for two prominent biometric modalities, face and fingerprint recognition. The attack can effectively infer the original recognition model (mean accuracy 83\% for faces, 86\% for fingerprints), and can craft effective biometric reconstructions that are successfully authenticated with 1-vs-1 authentication accuracy of up to 92\% for some models.

摘要: 生物特征认证服务提供商经常声称，不可能从其数学(特征空间)表示中逆向工程用户的原始生物特征样本，例如指纹或面部图像。在本文中，我们以深度神经网络(DNN)嵌入的具体例子来研究这一论断。DNN嵌入的逆已被用于解释深层图像表示或合成归一化图像。现有研究充分利用了对原始模型的所有层的访问权限，以及原始数据集的所有可能信息。对于生物认证用例，我们需要在敌意设置下调查这一点，在这种情况下，攻击者可以访问特征空间表示，但不能直接访问确切的原始数据集或原始学习模型。相反，我们假设攻击者对数据集的分布以及原始学习模型(体系结构和训练过程)有不同程度的背景知识。在这些情况下，我们表明攻击者可以利用现成的DNN模型和公共数据集，仅基于获得的表示和攻击者的先验知识来模仿原始学习模型的行为，以获得不同程度的成功。我们提出了一种双管齐下的攻击方法，首先利用嵌入的模型足迹推断出原始的DNN，然后利用推断出的模型重构原始数据。我们展示了该攻击对两种主要的生物识别模式--人脸和指纹识别--训练的流行DNN的实用性。该攻击可以有效地推断出原始的识别模型(人脸的平均准确率为83%，指纹的平均准确率为86%)，并且可以成功地构造出有效的生物特征重建，对于某些模型，1-vs-1的认证准确率高达92%。



## **49. Unfooling Perturbation-Based Post Hoc Explainers**

基于非愚弄扰动的帖子随机解说器 cs.AI

Accepted to AAAI-23. See the companion blog post at  https://medium.com/@craymichael/noncompliance-in-algorithmic-audits-and-defending-auditors-5b9fbdab2615.  9 pages (not including references and supplemental)

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2205.14772v3) [paper-pdf](http://arxiv.org/pdf/2205.14772v3)

**Authors**: Zachariah Carmichael, Walter J Scheirer

**Abstract**: Monumental advancements in artificial intelligence (AI) have lured the interest of doctors, lenders, judges, and other professionals. While these high-stakes decision-makers are optimistic about the technology, those familiar with AI systems are wary about the lack of transparency of its decision-making processes. Perturbation-based post hoc explainers offer a model agnostic means of interpreting these systems while only requiring query-level access. However, recent work demonstrates that these explainers can be fooled adversarially. This discovery has adverse implications for auditors, regulators, and other sentinels. With this in mind, several natural questions arise - how can we audit these black box systems? And how can we ascertain that the auditee is complying with the audit in good faith? In this work, we rigorously formalize this problem and devise a defense against adversarial attacks on perturbation-based explainers. We propose algorithms for the detection (CAD-Detect) and defense (CAD-Defend) of these attacks, which are aided by our novel conditional anomaly detection approach, KNN-CAD. We demonstrate that our approach successfully detects whether a black box system adversarially conceals its decision-making process and mitigates the adversarial attack on real-world data for the prevalent explainers, LIME and SHAP.

摘要: 人工智能(AI)的巨大进步吸引了医生、贷款人、法官和其他专业人士的兴趣。尽管这些事关重大的决策者对这项技术持乐观态度，但那些熟悉人工智能系统的人对其决策过程缺乏透明度持谨慎态度。基于扰动的后自组织解释器提供了一种模型不可知的方法来解释这些系统，而只需要查询级别的访问。然而，最近的研究表明，这些解释程序可能会被相反的人愚弄。这一发现对审计师、监管者和其他哨兵产生了不利影响。考虑到这一点，几个自然的问题就产生了--我们如何审计这些黑匣子系统？我们如何确定被审计人是真诚地遵守审计的？在这项工作中，我们严格地形式化了这个问题，并设计了一个防御对基于扰动的解释器的敌意攻击。在新的条件异常检测方法KNN-CAD的辅助下，我们提出了针对这些攻击的检测(CAD-检测)和防御(CAD-防御)算法。我们证明，我们的方法成功地检测到黑盒系统是否恶意地隐藏了其决策过程，并缓解了流行的解释程序LIME和Shap对真实数据的恶意攻击。



## **50. ADI: Adversarial Dominating Inputs in Vertical Federated Learning Systems**

ADI：垂直联合学习系统中的对抗性主导输入 cs.CR

**SubmitDate**: 2023-04-11    [abs](http://arxiv.org/abs/2201.02775v3) [paper-pdf](http://arxiv.org/pdf/2201.02775v3)

**Authors**: Qi Pang, Yuanyuan Yuan, Shuai Wang, Wenting Zheng

**Abstract**: Vertical federated learning (VFL) system has recently become prominent as a concept to process data distributed across many individual sources without the need to centralize it. Multiple participants collaboratively train models based on their local data in a privacy-aware manner. To date, VFL has become a de facto solution to securely learn a model among organizations, allowing knowledge to be shared without compromising privacy of any individuals. Despite the prosperous development of VFL systems, we find that certain inputs of a participant, named adversarial dominating inputs (ADIs), can dominate the joint inference towards the direction of the adversary's will and force other (victim) participants to make negligible contributions, losing rewards that are usually offered regarding the importance of their contributions in federated learning scenarios. We conduct a systematic study on ADIs by first proving their existence in typical VFL systems. We then propose gradient-based methods to synthesize ADIs of various formats and exploit common VFL systems. We further launch greybox fuzz testing, guided by the saliency score of ``victim'' participants, to perturb adversary-controlled inputs and systematically explore the VFL attack surface in a privacy-preserving manner. We conduct an in-depth study on the influence of critical parameters and settings in synthesizing ADIs. Our study reveals new VFL attack opportunities, promoting the identification of unknown threats before breaches and building more secure VFL systems.

摘要: 垂直联合学习(VFL)系统最近作为一个概念变得突出起来，它可以处理分布在多个独立来源的数据，而不需要集中这些数据。多个参与者以隐私感知的方式基于他们的本地数据协作训练模型。到目前为止，VFL已经成为在组织之间安全地学习模式的事实上的解决方案，允许在不损害任何个人隐私的情况下共享知识。尽管VFL系统的蓬勃发展，我们发现参与者的某些输入，称为对抗性主导输入(ADI)，可以主导朝着对手意愿方向的联合推理，并迫使其他(受害者)参与者做出可以忽略不计的贡献，失去通常提供的关于他们在联合学习场景中贡献的重要性的奖励。我们首先通过证明ADI在典型的VFL系统中的存在来对ADI进行系统的研究。然后，我们提出了基于梯度的方法来合成各种格式的ADI，并开发了常见的VFL系统。我们进一步推出灰盒模糊测试，以“受害者”参与者的显著分数为指导，扰乱对手控制的输入，并以保护隐私的方式系统地探索VFL攻击面。我们深入研究了关键参数和设置对ADI合成的影响。我们的研究揭示了新的VFL攻击机会，促进了在入侵之前识别未知威胁，并建立了更安全的VFL系统。



