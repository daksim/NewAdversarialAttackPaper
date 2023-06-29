# Latest Adversarial Attack Papers
**update at 2023-06-29 15:30:22**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Mitigating the Accuracy-Robustness Trade-off via Multi-Teacher Adversarial Distillation**

通过多教师对抗性蒸馏缓解精度与稳健性的权衡 cs.LG

**SubmitDate**: 2023-06-28    [abs](http://arxiv.org/abs/2306.16170v1) [paper-pdf](http://arxiv.org/pdf/2306.16170v1)

**Authors**: Shiji Zhao, Xizhe Wang, Xingxing Wei

**Abstract**: Adversarial training is a practical approach for improving the robustness of deep neural networks against adversarial attacks. Although bringing reliable robustness, the performance toward clean examples is negatively affected after adversarial training, which means a trade-off exists between accuracy and robustness. Recently, some studies have tried to use knowledge distillation methods in adversarial training, achieving competitive performance in improving the robustness but the accuracy for clean samples is still limited. In this paper, to mitigate the accuracy-robustness trade-off, we introduce the Multi-Teacher Adversarial Robustness Distillation (MTARD) to guide the model's adversarial training process by applying a strong clean teacher and a strong robust teacher to handle the clean examples and adversarial examples, respectively. During the optimization process, to ensure that different teachers show similar knowledge scales, we design the Entropy-Based Balance algorithm to adjust the teacher's temperature and keep the teachers' information entropy consistent. Besides, to ensure that the student has a relatively consistent learning speed from multiple teachers, we propose the Normalization Loss Balance algorithm to adjust the learning weights of different types of knowledge. A series of experiments conducted on public datasets demonstrate that MTARD outperforms the state-of-the-art adversarial training and distillation methods against various adversarial attacks.

摘要: 对抗性训练是提高深层神经网络抗敌意攻击能力的一种实用方法。虽然带来了可靠的稳健性，但经过对抗性训练后，对干净样本的性能会受到负面影响，这意味着在准确性和稳健性之间存在权衡。近年来，一些研究尝试将知识提取方法应用于对抗性训练，在提高鲁棒性方面取得了较好的性能，但对清洁样本的准确率仍然有限。为了缓解准确性和稳健性之间的权衡，我们引入了多教师对抗稳健性蒸馏(MTARD)来指导模型的对抗训练过程，分别采用强清洁教师和强稳健教师来处理干净实例和对抗性实例。在优化过程中，为了保证不同教师表现出相似的知识尺度，设计了基于熵的均衡算法来调整教师的温度，保持教师信息熵的一致性。此外，为了确保学生从多个老师那里获得相对一致的学习速度，我们提出了归一化损失平衡算法来调整不同类型知识的学习权重。在公开数据集上进行的一系列实验表明，MTARD在对抗各种对抗性攻击方面优于最先进的对抗性训练和蒸馏方法。



## **2. Distributional Modeling for Location-Aware Adversarial Patches**

位置感知敌方补丁的分布式建模 cs.CV

**SubmitDate**: 2023-06-28    [abs](http://arxiv.org/abs/2306.16131v1) [paper-pdf](http://arxiv.org/pdf/2306.16131v1)

**Authors**: Xingxing Wei, Shouwei Ruan, Yinpeng Dong, Hang Su

**Abstract**: Adversarial patch is one of the important forms of performing adversarial attacks in the physical world. To improve the naturalness and aggressiveness of existing adversarial patches, location-aware patches are proposed, where the patch's location on the target object is integrated into the optimization process to perform attacks. Although it is effective, efficiently finding the optimal location for placing the patches is challenging, especially under the black-box attack settings. In this paper, we propose the Distribution-Optimized Adversarial Patch (DOPatch), a novel method that optimizes a multimodal distribution of adversarial locations instead of individual ones. DOPatch has several benefits: Firstly, we find that the locations' distributions across different models are pretty similar, and thus we can achieve efficient query-based attacks to unseen models using a distributional prior optimized on a surrogate model. Secondly, DOPatch can generate diverse adversarial samples by characterizing the distribution of adversarial locations. Thus we can improve the model's robustness to location-aware patches via carefully designed Distributional-Modeling Adversarial Training (DOP-DMAT). We evaluate DOPatch on various face recognition and image recognition tasks and demonstrate its superiority and efficiency over existing methods. We also conduct extensive ablation studies and analyses to validate the effectiveness of our method and provide insights into the distribution of adversarial locations.

摘要: 对抗性补丁是物理世界中实施对抗性攻击的重要形式之一。为了提高现有敌方补丁的自然度和攻击性，提出了位置感知补丁，将补丁在目标对象上的位置整合到优化过程中进行攻击。虽然它是有效的，但高效地找到放置补丁的最佳位置是具有挑战性的，特别是在黑盒攻击环境下。在本文中，我们提出了分布优化的敌方补丁(DOPatch)，这是一种优化敌方位置的多模式分布而不是单个位置的新方法。DOPatch有几个优点：首先，我们发现位置在不同模型上的分布非常相似，因此我们可以利用在代理模型上优化的分布先验来实现对不可见模型的高效基于查询的攻击。其次，DOPatch可以通过刻画敌方位置的分布来生成不同的敌方样本。因此，我们可以通过精心设计的分布式建模对抗训练(DOP-DMAT)来提高模型对位置感知补丁的健壮性。我们对DOPatch在各种人脸识别和图像识别任务上的性能进行了评估，并证明了它比现有方法的优越性和有效性。我们还进行了广泛的消融研究和分析，以验证我们方法的有效性，并为敌方位置的分布提供见解。



## **3. Evaluating Similitude and Robustness of Deep Image Denoising Models via Adversarial Attack**

对抗性攻击下深度图像去噪模型的相似性和稳健性评价 cs.CV

12 pages, 15 figures

**SubmitDate**: 2023-06-28    [abs](http://arxiv.org/abs/2306.16050v1) [paper-pdf](http://arxiv.org/pdf/2306.16050v1)

**Authors**: Jie Ning, Yao Li, Zhichang Guo

**Abstract**: Deep neural networks (DNNs) have a wide range of applications in the field of image denoising, and they are superior to traditional image denoising. However, DNNs inevitably show vulnerability, which is the weak robustness in the face of adversarial attacks. In this paper, we find some similitudes between existing deep image denoising methods, as they are consistently fooled by adversarial attacks. First, denoising-PGD is proposed which is a denoising model full adversarial method. The current mainstream non-blind denoising models (DnCNN, FFDNet, ECNDNet, BRDNet), blind denoising models (DnCNN-B, Noise2Noise, RDDCNN-B, FAN), and plug-and-play (DPIR, CurvPnP) and unfolding denoising models (DeamNet) applied to grayscale and color images can be attacked by the same set of methods. Second, since the transferability of denoising-PGD is prominent in the image denoising task, we design experiments to explore the characteristic of the latent under the transferability. We correlate transferability with similitude and conclude that the deep image denoising models have high similitude. Third, we investigate the characteristic of the adversarial space and use adversarial training to complement the vulnerability of deep image denoising to adversarial attacks on image denoising. Finally, we constrain this adversarial attack method and propose the L2-denoising-PGD image denoising adversarial attack method that maintains the Gaussian distribution. Moreover, the model-driven image denoising BM3D shows some resistance in the face of adversarial attacks.

摘要: 深度神经网络(DNN)在图像去噪领域有着广泛的应用，并且优于传统的图像去噪。然而，DNN不可避免地表现出脆弱性，这就是面对对手攻击时的弱健壮性。在本文中，我们发现了现有的深度图像去噪方法之间的一些相似之处，因为它们总是被对手攻击所愚弄。首先，提出了一种完全对抗去噪模型的去噪方法--PGD。目前主流的非盲去噪模型(DnCNN、FFDNet、ECNDNet、BRDNet)、盲去噪模型(DnCNN-B、Noise2Noise、RDDCNN-B、FAN)、即插即用模型(DPIR、CurvPnP)和展开去噪模型(DeamNet)适用于灰度和彩色图像，可以用同一套方法攻击。其次，针对去噪-PGD在图像去噪任务中的可转移性突出的特点，设计了实验来探索可转移性下的潜伏期特征。将可转移性与相似性联系起来，得出深层图像去噪模型具有较高的相似性的结论。第三，研究了对抗性空间的特点，并利用对抗性训练来弥补深层图像去噪对图像去噪的对抗性攻击的脆弱性。最后，对这种对抗攻击方法进行了约束，提出了保持高斯分布的L2-去噪-PGD图像去噪对抗攻击方法。此外，模型驱动的图像去噪算法BM3D在面对敌方攻击时表现出一定的抵抗力。



## **4. Enrollment-stage Backdoor Attacks on Speaker Recognition Systems via Adversarial Ultrasound**

利用对抗性超声对说话人识别系统进行注册阶段的后门攻击 cs.SD

**SubmitDate**: 2023-06-28    [abs](http://arxiv.org/abs/2306.16022v1) [paper-pdf](http://arxiv.org/pdf/2306.16022v1)

**Authors**: Xinfeng Li, Junning Ze, Chen Yan, Yushi Cheng, Xiaoyu Ji, Wenyuan Xu

**Abstract**: Automatic Speaker Recognition Systems (SRSs) have been widely used in voice applications for personal identification and access control. A typical SRS consists of three stages, i.e., training, enrollment, and recognition. Previous work has revealed that SRSs can be bypassed by backdoor attacks at the training stage or by adversarial example attacks at the recognition stage. In this paper, we propose TUNER, a new type of backdoor attack against the enrollment stage of SRS via adversarial ultrasound modulation, which is inaudible, synchronization-free, content-independent, and black-box. Our key idea is to first inject the backdoor into the SRS with modulated ultrasound when a legitimate user initiates the enrollment, and afterward, the polluted SRS will grant access to both the legitimate user and the adversary with high confidence. Our attack faces a major challenge of unpredictable user articulation at the enrollment stage. To overcome this challenge, we generate the ultrasonic backdoor by augmenting the optimization process with random speech content, vocalizing time, and volume of the user. Furthermore, to achieve real-world robustness, we improve the ultrasonic signal over traditional methods using sparse frequency points, pre-compensation, and single-sideband (SSB) modulation. We extensively evaluate TUNER on two common datasets and seven representative SRS models. Results show that our attack can successfully bypass speaker recognition systems while remaining robust to various speakers, speech content, et

摘要: 自动说话人识别系统(SRSS)已被广泛应用于个人身份识别和访问控制的语音应用中。一个典型的SRS包括三个阶段，即培训、注册和认可。先前的工作表明，在训练阶段通过后门攻击或在识别阶段通过对抗性示例攻击可以绕过SRSS。本文提出了一种基于对抗性超声调制的针对SRS注册阶段的新型后门攻击--Tuner，它具有不可听、无同步、内容无关、黑盒等特点。我们的核心思想是，当合法用户发起注册时，首先使用调制超声波将后门注入SRS，然后，受污染的SRS将以高置信度向合法用户和对手授予访问权限。我们的攻击在注册阶段面临着不可预测的用户表达的重大挑战。为了克服这一挑战，我们通过增加随机语音内容、发声时间和用户音量的优化过程来生成超声波后门。此外，为了实现真实世界的稳健性，我们使用稀疏频点、预补偿和单边带(SSB)调制对超声信号进行了改进。我们在两个常见的数据集和七个有代表性的SRS模型上对Tuner进行了广泛的评估。结果表明，我们的攻击可以成功地绕过说话人识别系统，同时对不同的说话人、语音内容等保持鲁棒性



## **5. What is the Solution for State-Adversarial Multi-Agent Reinforcement Learning?**

状态对抗性多智能体强化学习的解决方案是什么？ cs.AI

Workshop on New Frontiers in Learning, Control, and Dynamical Systems  at the International Conference on Machine Learning (ICML), Honolulu, Hawaii,  USA, 2023

**SubmitDate**: 2023-06-28    [abs](http://arxiv.org/abs/2212.02705v4) [paper-pdf](http://arxiv.org/pdf/2212.02705v4)

**Authors**: Songyang Han, Sanbao Su, Sihong He, Shuo Han, Haizhao Yang, Fei Miao

**Abstract**: Various methods for Multi-Agent Reinforcement Learning (MARL) have been developed with the assumption that agents' policies are based on accurate state information. However, policies learned through Deep Reinforcement Learning (DRL) are susceptible to adversarial state perturbation attacks. In this work, we propose a State-Adversarial Markov Game (SAMG) and make the first attempt to investigate the fundamental properties of MARL under state uncertainties. Our analysis shows that the commonly used solution concepts of optimal agent policy and robust Nash equilibrium do not always exist in SAMGs. To circumvent this difficulty, we consider a new solution concept called robust agent policy, where agents aim to maximize the worst-case expected state value. We prove the existence of robust agent policy for finite state and finite action SAMGs. Additionally, we propose a Robust Multi-Agent Adversarial Actor-Critic (RMA3C) algorithm to learn robust policies for MARL agents under state uncertainties. Our experiments demonstrate that our algorithm outperforms existing methods when faced with state perturbations and greatly improves the robustness of MARL policies. Our code is public on https://songyanghan.github.io/what_is_solution/.

摘要: 多智能体强化学习(MAIL)的各种方法都是在假设智能体的策略基于准确的状态信息的基础上提出的。然而，通过深度强化学习(DRL)学习的策略容易受到对抗性状态扰动攻击。在这项工作中，我们提出了一种状态-对手马尔可夫博弈(SAMG)，并首次尝试研究了状态不确定条件下Marl的基本性质。我们的分析表明，最优代理策略和稳健纳什均衡等解的概念在SAMG中并不总是存在的。为了规避这一困难，我们考虑了一个新的解决方案概念，称为稳健代理策略，其中代理的目标是最大化最坏情况下的预期状态值。我们证明了有限状态和有限动作SAMG的鲁棒代理策略的存在性。此外，我们还提出了一种健壮的多智能体对抗行为者-批评者(RMA3C)算法来学习状态不确定条件下MAIL智能体的健壮策略。实验表明，该算法在面对状态扰动时的性能优于已有方法，并大大提高了MAIL策略的稳健性。我们的代码在https://songyanghan.github.io/what_is_solution/.上是公开的



## **6. Boosting Adversarial Transferability with Learnable Patch-wise Masks**

用可学习的补丁口罩提高对手的可转移性 cs.CV

**SubmitDate**: 2023-06-28    [abs](http://arxiv.org/abs/2306.15931v1) [paper-pdf](http://arxiv.org/pdf/2306.15931v1)

**Authors**: Xingxing Wei, Shiji Zhao

**Abstract**: Adversarial examples have raised widespread attention in security-critical applications because of their transferability across different models. Although many methods have been proposed to boost adversarial transferability, a gap still exists in the practical demand. In this paper, we argue that the model-specific discriminative regions are a key factor to cause the over-fitting to the source model, and thus reduce the transferability to the target model. For that, a patch-wise mask is utilized to prune the model-specific regions when calculating adversarial perturbations. To accurately localize these regions, we present a learnable approach to optimize the mask automatically. Specifically, we simulate the target models in our framework, and adjust the patch-wise mask according to the feedback of simulated models. To improve the efficiency, Differential Evolutionary (DE) algorithm is utilized to search for patch-wise masks for a specific image. During iterative attacks, the learned masks are applied to the image to drop out the patches related to model-specific regions, thus making the gradients more generic and improving the adversarial transferability. The proposed approach is a pre-processing method and can be integrated with existing gradient-based methods to further boost the transfer attack success rate. Extensive experiments on the ImageNet dataset demonstrate the effectiveness of our method. We incorporate the proposed approach with existing methods in the ensemble attacks and achieve an average success rate of 93.01% against seven advanced defense methods, which can effectively enhance the state-of-the-art transfer-based attack performance.

摘要: 对抗性的例子在安全关键应用中引起了广泛的关注，因为它们可以在不同的模型之间转移。虽然已经提出了许多方法来提高对抗性可转移性，但在实际需求中仍存在差距。在本文中，我们认为，特定于模型的区分区域是导致对源模型过度拟合从而降低到目标模型的可转换性的关键因素。为此，在计算对抗性扰动时，使用补丁掩码来修剪特定于模型的区域。为了准确地定位这些区域，我们提出了一种可学习的方法来自动优化掩码。具体地说，我们在我们的框架中模拟目标模型，并根据模拟模型的反馈调整面片掩码。为了提高算法的效率，采用了差分进化算法来搜索特定图像的面片模板。在迭代攻击过程中，将学习到的模板应用于图像，去除与模型特定区域相关的补丁，从而使梯度更具通用性，提高了对抗可转移性。该方法是一种预处理方法，可以与现有的基于梯度的方法相结合，进一步提高传输攻击的成功率。在ImageNet数据集上的大量实验证明了该方法的有效性。将提出的方法与现有的集成攻击方法相结合，对7种先进的防御方法取得了93.01%的平均成功率，有效地提高了基于传输的攻击性能。



## **7. A Diamond Model Analysis on Twitter's Biggest Hack**

Twitter最大黑客攻击的钻石模型分析 cs.CR

8 pages, 3 figures, 2 tables

**SubmitDate**: 2023-06-28    [abs](http://arxiv.org/abs/2306.15878v1) [paper-pdf](http://arxiv.org/pdf/2306.15878v1)

**Authors**: Chaitanya Rahalkar

**Abstract**: Cyberattacks have prominently increased over the past few years now, and have targeted actors from a wide variety of domains. Understanding the motivation, infrastructure, attack vectors, etc. behind such attacks is vital to proactively work against preventing such attacks in the future and also to analyze the economic and social impact of such attacks. In this paper, we leverage the diamond model to perform an intrusion analysis case study of the 2020 Twitter account hijacking Cyberattack. We follow this standardized incident response model to map the adversary, capability, infrastructure, and victim and perform a comprehensive analysis of the attack, and the impact posed by the attack from a Cybersecurity policy standpoint.

摘要: 网络攻击在过去几年里显著增加，目标是来自不同领域的参与者。了解此类攻击背后的动机、基础设施、攻击媒介等对于主动预防未来此类攻击以及分析此类攻击的经济和社会影响至关重要。在本文中，我们利用钻石模型对2020年Twitter账户劫持网络攻击进行了入侵分析案例研究。我们遵循这个标准化的事件响应模型来映射对手、能力、基础设施和受害者，并从网络安全策略的角度对攻击和攻击造成的影响进行全面分析。



## **8. Condorcet Attack Against Fair Transaction Ordering**

针对公平交易排序的Condorcet攻击 cs.CR

**SubmitDate**: 2023-06-27    [abs](http://arxiv.org/abs/2306.15743v1) [paper-pdf](http://arxiv.org/pdf/2306.15743v1)

**Authors**: Mohammad Amin Vafadar, Majid Khabbazian

**Abstract**: We introduce the Condorcet attack, a new threat to fair transaction ordering. Specifically, the attack undermines batch-order-fairness, the strongest notion of transaction fair ordering proposed to date. The batch-order-fairness guarantees that a transaction tx is ordered before tx' if a majority of nodes in the system receive tx before tx'; the only exception (due to an impossibility result) is when tx and tx' fall into a so-called "Condorcet cycle". When this happens, tx and tx' along with other transactions within the cycle are placed in a batch, and any unfairness inside a batch is ignored. In the Condorcet attack, an adversary attempts to undermine the system's fairness by imposing Condorcet cycles to the system. In this work, we show that the adversary can indeed impose a Condorcet cycle by submitting as few as two otherwise legitimate transactions to the system. Remarkably, the adversary (e.g., a malicious client) can achieve this even when all the nodes in the system behave honestly. A notable feature of the attack is that it is capable of "trapping" transactions that do not naturally fall inside a cycle, i.e. those that are transmitted at significantly different times (with respect to the network latency). To mitigate the attack, we propose three methods based on three different complementary approaches. We show the effectiveness of the proposed mitigation methods through simulations, and explain their limitations.

摘要: 我们引入了Condorcet攻击，这是对公平交易秩序的一种新威胁。具体地说，这次攻击破坏了批次排序公平，这是迄今为止提出的最强的交易公平排序概念。如果系统中的大多数节点在Tx‘之前接收到Tx，则批排序公平性保证了事务Tx在Tx’之前被排序；唯一的例外(由于不可能的结果)是当Tx和Tx‘落入所谓的“Condorcet循环”时。当这种情况发生时，Tx和Tx‘连同周期内的其他交易被放置在批中，并且批内的任何不公平都被忽略。在Condorcet攻击中，攻击者试图通过将Condorcet循环强加给系统来破坏系统的公平性。在这项工作中，我们证明了对手确实可以通过向系统提交少至两个原本合法的交易来施加Condorcet循环。值得注意的是，即使当系统中的所有节点都诚实地运行时，敌手(例如，恶意客户端)也可以实现这一点。该攻击的一个显著特征是，它能够“捕获”并非自然落入周期内的事务，即在显著不同的时间传输的事务(就网络延迟而言)。为了缓解攻击，我们基于三种不同的互补方法提出了三种方法。我们通过仿真验证了所提出的抑制方法的有效性，并解释了它们的局限性。



## **9. Cooperation or Competition: Avoiding Player Domination for Multi-Target Robustness via Adaptive Budgets**

合作还是竞争：通过自适应预算避免玩家主导多目标稳健性 cs.AI

**SubmitDate**: 2023-06-27    [abs](http://arxiv.org/abs/2306.15482v1) [paper-pdf](http://arxiv.org/pdf/2306.15482v1)

**Authors**: Yimu Wang, Dinghuai Zhang, Yihan Wu, Heng Huang, Hongyang Zhang

**Abstract**: Despite incredible advances, deep learning has been shown to be susceptible to adversarial attacks. Numerous approaches have been proposed to train robust networks both empirically and certifiably. However, most of them defend against only a single type of attack, while recent work takes steps forward in defending against multiple attacks. In this paper, to understand multi-target robustness, we view this problem as a bargaining game in which different players (adversaries) negotiate to reach an agreement on a joint direction of parameter updating. We identify a phenomenon named player domination in the bargaining game, namely that the existing max-based approaches, such as MAX and MSD, do not converge. Based on our theoretical analysis, we design a novel framework that adjusts the budgets of different adversaries to avoid any player dominance. Experiments on standard benchmarks show that employing the proposed framework to the existing approaches significantly advances multi-target robustness.

摘要: 尽管取得了令人难以置信的进步，但深度学习已被证明容易受到对手的攻击。已经提出了许多方法来训练稳健的网络，既有经验的，也有可证明的。然而，它们中的大多数只防御一种类型的攻击，而最近的工作在防御多种攻击方面取得了进展。在本文中，为了理解多目标的稳健性，我们将这一问题视为一个讨价还价博弈，不同的参与者(对手)通过协商就参数更新的共同方向达成协议。我们在讨价还价博弈中发现了一种称为玩家支配的现象，即现有的基于最大值的方法，如MAX和MSD，不收敛。在理论分析的基础上，我们设计了一个新的框架来调整不同对手的预算，以避免任何玩家的支配地位。在标准基准上的实验表明，在现有方法的基础上使用该框架可以显著提高多目标的稳健性。



## **10. Robust Proxy: Improving Adversarial Robustness by Robust Proxy Learning**

稳健代理：通过稳健代理学习提高对手的稳健性 cs.CV

Accepted at IEEE Transactions on Information Forensics and Security  (TIFS)

**SubmitDate**: 2023-06-27    [abs](http://arxiv.org/abs/2306.15457v1) [paper-pdf](http://arxiv.org/pdf/2306.15457v1)

**Authors**: Hong Joo Lee, Yong Man Ro

**Abstract**: Recently, it has been widely known that deep neural networks are highly vulnerable and easily broken by adversarial attacks. To mitigate the adversarial vulnerability, many defense algorithms have been proposed. Recently, to improve adversarial robustness, many works try to enhance feature representation by imposing more direct supervision on the discriminative feature. However, existing approaches lack an understanding of learning adversarially robust feature representation. In this paper, we propose a novel training framework called Robust Proxy Learning. In the proposed method, the model explicitly learns robust feature representations with robust proxies. To this end, firstly, we demonstrate that we can generate class-representative robust features by adding class-wise robust perturbations. Then, we use the class representative features as robust proxies. With the class-wise robust features, the model explicitly learns adversarially robust features through the proposed robust proxy learning framework. Through extensive experiments, we verify that we can manually generate robust features, and our proposed learning framework could increase the robustness of the DNNs.

摘要: 近年来，众所周知，深度神经网络具有很高的脆弱性，很容易被敌方攻击攻破。为了缓解恶意攻击的脆弱性，人们提出了许多防御算法。近年来，为了提高对抗的稳健性，许多工作试图通过对区分特征施加更直接的监督来增强特征表示。然而，现有的方法缺乏对学习对抗性稳健特征表示的理解。在本文中，我们提出了一种新的训练框架，称为稳健代理学习。在所提出的方法中，模型通过稳健的代理显式地学习稳健的特征表示。为此，首先，我们证明了我们可以通过添加类稳健扰动来生成具有类代表性的鲁棒特征。然后，我们使用类代表特征作为健壮性代理。该模型利用分类稳健特征，通过所提出的稳健代理学习框架显式地学习对抗性稳健特征。通过大量的实验，我们验证了我们可以手动生成健壮的特征，并且我们提出的学习框架可以提高DNN的健壮性。



## **11. Advancing Adversarial Training by Injecting Booster Signal**

注入助推器信号推进对抗性训练 cs.CV

Accepted at IEEE Transactions on Neural Networks and Learning Systems

**SubmitDate**: 2023-06-27    [abs](http://arxiv.org/abs/2306.15451v1) [paper-pdf](http://arxiv.org/pdf/2306.15451v1)

**Authors**: Hong Joo Lee, Youngjoon Yu, Yong Man Ro

**Abstract**: Recent works have demonstrated that deep neural networks (DNNs) are highly vulnerable to adversarial attacks. To defend against adversarial attacks, many defense strategies have been proposed, among which adversarial training has been demonstrated to be the most effective strategy. However, it has been known that adversarial training sometimes hurts natural accuracy. Then, many works focus on optimizing model parameters to handle the problem. Different from the previous approaches, in this paper, we propose a new approach to improve the adversarial robustness by using an external signal rather than model parameters. In the proposed method, a well-optimized universal external signal called a booster signal is injected into the outside of the image which does not overlap with the original content. Then, it boosts both adversarial robustness and natural accuracy. The booster signal is optimized in parallel to model parameters step by step collaboratively. Experimental results show that the booster signal can improve both the natural and robust accuracies over the recent state-of-the-art adversarial training methods. Also, optimizing the booster signal is general and flexible enough to be adopted on any existing adversarial training methods.

摘要: 最近的研究表明，深度神经网络(DNN)非常容易受到敌意攻击。为了防御对抗性攻击，人们提出了许多防御策略，其中对抗性训练被证明是最有效的策略。然而，众所周知，对抗性训练有时会损害自然的准确性。于是，很多工作都集中在优化模型参数来处理这一问题上。与以往的方法不同，本文提出了一种利用外部信号而不是模型参数来提高对抗稳健性的新方法。在该方法中，一种经过优化的通用外部信号被注入到图像的外部，该信号与原始内容不重叠。然后，它既增强了对手的健壮性，又提高了自然的准确性。协同地对升压信号进行并行优化以逐步建立模型参数。实验结果表明，与目前最先进的对抗性训练方法相比，增强信号可以提高自然准确率和稳健准确率。此外，优化助推信号是通用的和灵活的，可以在任何现有的对抗性训练方法中采用。



## **12. Adversarial Training for Graph Neural Networks**

图神经网络的对抗性训练 cs.LG

**SubmitDate**: 2023-06-27    [abs](http://arxiv.org/abs/2306.15427v1) [paper-pdf](http://arxiv.org/pdf/2306.15427v1)

**Authors**: Lukas Gosch, Simon Geisler, Daniel Sturm, Bertrand Charpentier, Daniel Zügner, Stephan Günnemann

**Abstract**: Despite its success in the image domain, adversarial training does not (yet) stand out as an effective defense for Graph Neural Networks (GNNs) against graph structure perturbations. In the pursuit of fixing adversarial training (1) we show and overcome fundamental theoretical as well as practical limitations of the adopted graph learning setting in prior work; (2) we reveal that more flexible GNNs based on learnable graph diffusion are able to adjust to adversarial perturbations, while the learned message passing scheme is naturally interpretable; (3) we introduce the first attack for structure perturbations that, while targeting multiple nodes at once, is capable of handling global (graph-level) as well as local (node-level) constraints. Including these contributions, we demonstrate that adversarial training is a state-of-the-art defense against adversarial structure perturbations.

摘要: 尽管它在图像领域取得了成功，但对抗性训练并不是图神经网络(GNN)对抗图结构扰动的有效防御。在固定对抗性训练的过程中，(1)我们展示并克服了以前工作中采用的图学习设置的基本理论和实践限制；(2)我们揭示了基于可学习图扩散的更灵活的GNN能够适应对抗性扰动，而学习的消息传递方案自然是可解释的；(3)我们引入了针对结构扰动的第一次攻击，虽然一次针对多个节点，但能够处理全局(图级)和局部(节点级)约束。包括这些贡献，我们证明了对抗性训练是对对抗性结构扰动的一种最先进的防御。



## **13. Your Attack Is Too DUMB: Formalizing Attacker Scenarios for Adversarial Transferability**

你的攻击太愚蠢了：将攻击者的场景形式化，以实现对抗性转移 cs.CR

Accepted at RAID 2023

**SubmitDate**: 2023-06-27    [abs](http://arxiv.org/abs/2306.15363v1) [paper-pdf](http://arxiv.org/pdf/2306.15363v1)

**Authors**: Marco Alecci, Mauro Conti, Francesco Marchiori, Luca Martinelli, Luca Pajola

**Abstract**: Evasion attacks are a threat to machine learning models, where adversaries attempt to affect classifiers by injecting malicious samples. An alarming side-effect of evasion attacks is their ability to transfer among different models: this property is called transferability. Therefore, an attacker can produce adversarial samples on a custom model (surrogate) to conduct the attack on a victim's organization later. Although literature widely discusses how adversaries can transfer their attacks, their experimental settings are limited and far from reality. For instance, many experiments consider both attacker and defender sharing the same dataset, balance level (i.e., how the ground truth is distributed), and model architecture.   In this work, we propose the DUMB attacker model. This framework allows analyzing if evasion attacks fail to transfer when the training conditions of surrogate and victim models differ. DUMB considers the following conditions: Dataset soUrces, Model architecture, and the Balance of the ground truth. We then propose a novel testbed to evaluate many state-of-the-art evasion attacks with DUMB; the testbed consists of three computer vision tasks with two distinct datasets each, four types of balance levels, and three model architectures. Our analysis, which generated 13K tests over 14 distinct attacks, led to numerous novel findings in the scope of transferable attacks with surrogate models. In particular, mismatches between attackers and victims in terms of dataset source, balance levels, and model architecture lead to non-negligible loss of attack performance.

摘要: 逃避攻击是对机器学习模型的威胁，在机器学习模型中，攻击者试图通过注入恶意样本来影响分类器。规避攻击的一个令人担忧的副作用是它们在不同模型之间传输的能力：这种特性被称为可转移性。因此，攻击者可以在自定义模型(代理)上生成对抗性样本，以便稍后对受害者的组织进行攻击。虽然文献中广泛讨论了对手如何转移他们的攻击，但他们的实验设置有限，与现实相去甚远。例如，许多实验认为攻击者和防御者共享相同的数据集、平衡级别(即地面真相是如何分布的)和模型体系结构。在这项工作中，我们提出了哑巴攻击者模型。该框架允许分析当代理模型和受害者模型的训练条件不同时，逃避攻击是否无法转移。哑巴考虑以下条件：数据集源、模型体系结构和基本事实的平衡。然后，我们提出了一种新的测试平台来评估许多最新的哑巴规避攻击；该测试平台由三个计算机视觉任务组成，每个任务有两个不同的数据集，四种类型的平衡级别和三个模型体系结构。我们的分析对14个不同的攻击进行了13K测试，在使用代理模型的可转移攻击的范围内得出了许多新的发现。特别是，攻击者和受害者在数据集源、平衡级别和模型体系结构方面的不匹配会导致不可忽视的攻击性能损失。



## **14. GPS-Spoofing Attack Detection Mechanism for UAV Swarms**

无人机群的GPS欺骗攻击检测机制 cs.CR

8 pages, 3 figures

**SubmitDate**: 2023-06-27    [abs](http://arxiv.org/abs/2301.12766v2) [paper-pdf](http://arxiv.org/pdf/2301.12766v2)

**Authors**: Pavlo Mykytyn, Marcin Brzozowski, Zoya Dyka, Peter Langendoerfer

**Abstract**: Recently autonomous and semi-autonomous Unmanned Aerial Vehicle (UAV) swarms started to receive a lot of research interest and demand from various civil application fields. However, for successful mission execution, UAV swarms require Global navigation satellite system signals and in particular, Global Positioning System (GPS) signals for navigation. Unfortunately, civil GPS signals are unencrypted and unauthenticated, which facilitates the execution of GPS spoofing attacks. During these attacks, adversaries mimic the authentic GPS signal and broadcast it to the targeted UAV in order to change its course, and force it to land or crash. In this study, we propose a GPS spoofing detection mechanism capable of detecting single-transmitter and multi-transmitter GPS spoofing attacks to prevent the outcomes mentioned above. Our detection mechanism is based on comparing the distance between each two swarm members calculated from their GPS coordinates to the distance acquired from Impulse Radio Ultra-Wideband ranging between the same swarm members. If the difference in distances is larger than a chosen threshold the GPS spoofing attack is declared detected.

摘要: 近年来，自主和半自主无人机群体开始受到各个民用应用领域的研究兴趣和需求。然而，为了成功执行任务，无人机群需要全球导航卫星系统信号，特别是导航全球定位系统(GPS)信号。不幸的是，民用GPS信号是未加密和未认证的，这为GPS欺骗攻击的执行提供了便利。在这些攻击中，对手模仿真实的GPS信号，将其广播到目标无人机，以改变其航线，并迫使其着陆或坠毁。在这项研究中，我们提出了一种GPS欺骗检测机制，能够检测到单发送器和多发送器的GPS欺骗攻击，以防止上述结果。我们的检测机制是基于比较两个蜂群成员之间的距离，该距离是根据它们的GPS坐标计算的，与通过ImPulse Radio超宽带获得的相同蜂群成员之间的距离进行比较。如果距离之差大于选定的阈值，则宣布检测到GPS欺骗攻击。



## **15. Feature Adversarial Distillation for Point Cloud Classification**

点云分类中的特征对抗性提取 cs.CV

Accepted to ICIP2023

**SubmitDate**: 2023-06-27    [abs](http://arxiv.org/abs/2306.14221v2) [paper-pdf](http://arxiv.org/pdf/2306.14221v2)

**Authors**: YuXing Lee, Wei Wu

**Abstract**: Due to the point cloud's irregular and unordered geometry structure, conventional knowledge distillation technology lost a lot of information when directly used on point cloud tasks. In this paper, we propose Feature Adversarial Distillation (FAD) method, a generic adversarial loss function in point cloud distillation, to reduce loss during knowledge transfer. In the feature extraction stage, the features extracted by the teacher are used as the discriminator, and the students continuously generate new features in the training stage. The feature of the student is obtained by attacking the feedback from the teacher and getting a score to judge whether the student has learned the knowledge well or not. In experiments on standard point cloud classification on ModelNet40 and ScanObjectNN datasets, our method reduced the information loss of knowledge transfer in distillation in 40x model compression while maintaining competitive performance.

摘要: 由于点云的不规则和无序的几何结构，传统的知识提取技术在直接用于点云任务时会丢失大量的信息。为了减少知识转移过程中的损失，本文提出了特征对抗蒸馏(FAD)方法，它是点云蒸馏中一种通用的对抗性损失函数。在特征提取阶段，将教师提取的特征作为判别器，在训练阶段，学生不断产生新的特征。学生的特征是通过攻击教师的反馈并获得分数来判断学生是否学好了知识。在ModelNet40和ScanObjectNN数据集上的标准点云分类实验中，我们的方法在保持竞争性能的同时，减少了40倍模型压缩下蒸馏过程中知识传递的信息损失。



## **16. A Highly Accurate Query-Recovery Attack against Searchable Encryption using Non-Indexed Documents**

一种针对使用非索引文档的可搜索加密的高精度查询恢复攻击 cs.CR

Published in USENIX 2021. Full version with extended appendices and  removed some typos

**SubmitDate**: 2023-06-27    [abs](http://arxiv.org/abs/2306.15302v1) [paper-pdf](http://arxiv.org/pdf/2306.15302v1)

**Authors**: Marc Damie, Florian Hahn, Andreas Peter

**Abstract**: Cloud data storage solutions offer customers cost-effective and reduced data management. While attractive, data security issues remain to be a core concern. Traditional encryption protects stored documents, but hinders simple functionalities such as keyword search. Therefore, searchable encryption schemes have been proposed to allow for the search on encrypted data. Efficient schemes leak at least the access pattern (the accessed documents per keyword search), which is known to be exploitable in query recovery attacks assuming the attacker has a significant amount of background knowledge on the stored documents. Existing attacks can only achieve decent results with strong adversary models (e.g. at least 20% of previously known documents or require additional knowledge such as on query frequencies) and they give no metric to evaluate the certainty of recovered queries. This hampers their practical utility and questions their relevance in the real-world.   We propose a refined score attack which achieves query recovery rates of around 85% without requiring exact background knowledge on stored documents; a distributionally similar, but otherwise different (i.e., non-indexed), dataset suffices. The attack starts with very few known queries (around 10 known queries in our experiments over different datasets of varying size) and then iteratively recovers further queries with confidence scores by adding previously recovered queries that had high confidence scores to the set of known queries. Additional to high recovery rates, our approach yields interpretable results in terms of confidence scores.

摘要: 云数据存储解决方案为客户提供经济高效且减少的数据管理。虽然很有吸引力，但数据安全问题仍然是一个核心问题。传统加密保护存储的文档，但会阻碍关键字搜索等简单功能。因此，已经提出了可搜索的加密方案以允许对加密数据进行搜索。高效的方案至少泄漏访问模式(每个关键字搜索访问的文档)，假设攻击者对存储的文档有大量的背景知识，则已知该访问模式在查询恢复攻击中可被利用。现有的攻击只能通过强对手模型(例如，至少20%的已知文档或需要额外的知识，如查询频率)才能达到令人满意的结果，并且它们没有给出评估恢复的查询的确定性的度量。这妨碍了它们的实用性，并质疑它们在现实世界中的相关性。我们提出了一种改进的Score攻击，在不需要存储文档的确切背景知识的情况下，查询恢复率达到85%左右；分布相似但不同(即，非索引)的数据集就足够了。该攻击从极少的已知查询开始(在我们的实验中，在不同大小的不同数据集上约有10个已知查询)，然后通过将先前恢复的具有高置信度分数的查询添加到已知查询集来迭代地恢复具有置信度分数的进一步查询。除了高恢复率，我们的方法还产生了可以解释的信心分数结果。



## **17. On the Universal Adversarial Perturbations for Efficient Data-free Adversarial Detection**

有效无数据敌意检测的泛化敌意扰动研究 cs.CL

Accepted by ACL2023 (Short Paper)

**SubmitDate**: 2023-06-27    [abs](http://arxiv.org/abs/2306.15705v1) [paper-pdf](http://arxiv.org/pdf/2306.15705v1)

**Authors**: Songyang Gao, Shihan Dou, Qi Zhang, Xuanjing Huang, Jin Ma, Ying Shan

**Abstract**: Detecting adversarial samples that are carefully crafted to fool the model is a critical step to socially-secure applications. However, existing adversarial detection methods require access to sufficient training data, which brings noteworthy concerns regarding privacy leakage and generalizability. In this work, we validate that the adversarial sample generated by attack algorithms is strongly related to a specific vector in the high-dimensional inputs. Such vectors, namely UAPs (Universal Adversarial Perturbations), can be calculated without original training data. Based on this discovery, we propose a data-agnostic adversarial detection framework, which induces different responses between normal and adversarial samples to UAPs. Experimental results show that our method achieves competitive detection performance on various text classification tasks, and maintains an equivalent time consumption to normal inference.

摘要: 检测精心设计来愚弄模型的敌意样本是实现社交安全应用程序的关键一步。然而，现有的敌意检测方法需要访问足够的训练数据，这带来了对隐私泄露和泛化的关注。在这项工作中，我们验证了攻击算法生成的对抗性样本与高维输入中的特定向量具有很强的相关性。这样的矢量，即UAP(通用对抗性扰动)，可以在没有原始训练数据的情况下计算。基于这一发现，我们提出了一个数据不可知的敌意检测框架，该框架在正常样本和敌意样本之间对UAP产生不同的响应。实验结果表明，我们的方法在不同的文本分类任务上达到了竞争性的检测性能，并且保持了与正常推理相当的时间消耗。



## **18. DSRM: Boost Textual Adversarial Training with Distribution Shift Risk Minimization**

DSRM：用分布转移风险最小化促进文本对抗性训练 cs.CL

Accepted by ACL2023

**SubmitDate**: 2023-06-27    [abs](http://arxiv.org/abs/2306.15164v1) [paper-pdf](http://arxiv.org/pdf/2306.15164v1)

**Authors**: Songyang Gao, Shihan Dou, Yan Liu, Xiao Wang, Qi Zhang, Zhongyu Wei, Jin Ma, Ying Shan

**Abstract**: Adversarial training is one of the best-performing methods in improving the robustness of deep language models. However, robust models come at the cost of high time consumption, as they require multi-step gradient ascents or word substitutions to obtain adversarial samples. In addition, these generated samples are deficient in grammatical quality and semantic consistency, which impairs the effectiveness of adversarial training. To address these problems, we introduce a novel, effective procedure for instead adversarial training with only clean data. Our procedure, distribution shift risk minimization (DSRM), estimates the adversarial loss by perturbing the input data's probability distribution rather than their embeddings. This formulation results in a robust model that minimizes the expected global loss under adversarial attacks. Our approach requires zero adversarial samples for training and reduces time consumption by up to 70\% compared to current best-performing adversarial training methods. Experiments demonstrate that DSRM considerably improves BERT's resistance to textual adversarial attacks and achieves state-of-the-art robust accuracy on various benchmarks.

摘要: 对抗性训练是提高深层语言模型稳健性的最佳方法之一。然而，健壮的模型是以高时间消耗为代价的，因为它们需要多步梯度上升或单词替换来获得对抗性样本。此外，这些生成的样本在语法质量和语义一致性方面存在不足，这损害了对抗性训练的有效性。为了解决这些问题，我们引入了一种新颖、有效的程序，用于仅使用干净数据进行对抗性训练。我们的过程，分布转移风险最小化(DSRM)，通过扰动输入数据的概率分布而不是它们的嵌入来估计对手损失。这一公式导致了一个健壮的模型，该模型将在对抗性攻击下的预期全球损失降至最低。与目前性能最好的对抗性训练方法相比，我们的方法需要零个对抗性样本进行训练，并且减少了高达70%的时间消耗。实验表明，DSRM在很大程度上提高了BERT对文本对手攻击的抵抗力，并在各种基准上获得了最先进的鲁棒准确率。



## **19. Towards Sybil Resilience in Decentralized Learning**

分散学习中的Sybil弹性研究 cs.DC

**SubmitDate**: 2023-06-26    [abs](http://arxiv.org/abs/2306.15044v1) [paper-pdf](http://arxiv.org/pdf/2306.15044v1)

**Authors**: Thomas Werthenbach, Johan Pouwelse

**Abstract**: Federated learning is a privacy-enforcing machine learning technology but suffers from limited scalability. This limitation mostly originates from the internet connection and memory capacity of the central parameter server, and the complexity of the model aggregation function. Decentralized learning has recently been emerging as a promising alternative to federated learning. This novel technology eliminates the need for a central parameter server by decentralizing the model aggregation across all participating nodes. Numerous studies have been conducted on improving the resilience of federated learning against poisoning and Sybil attacks, whereas the resilience of decentralized learning remains largely unstudied. This research gap serves as the main motivator for this study, in which our objective is to improve the Sybil poisoning resilience of decentralized learning.   We present SybilWall, an innovative algorithm focused on increasing the resilience of decentralized learning against targeted Sybil poisoning attacks. By combining a Sybil-resistant aggregation function based on similarity between Sybils with a novel probabilistic gossiping mechanism, we establish a new benchmark for scalable, Sybil-resilient decentralized learning.   A comprehensive empirical evaluation demonstrated that SybilWall outperforms existing state-of-the-art solutions designed for federated learning scenarios and is the only algorithm to obtain consistent accuracy over a range of adversarial attack scenarios. We also found SybilWall to diminish the utility of creating many Sybils, as our evaluations demonstrate a higher success rate among adversaries employing fewer Sybils. Finally, we suggest a number of possible improvements to SybilWall and highlight promising future research directions.

摘要: 联合学习是一种隐私保护的机器学习技术，但可扩展性有限。这一限制主要源于中央参数服务器的互联网连接和存储容量，以及模型聚合功能的复杂性。去中心化学习最近正在成为联合学习的一种有前途的替代方案。这项新技术通过将模型聚合分散到所有参与节点，从而消除了对中央参数服务器的需求。关于提高联合学习对中毒和Sybil攻击的弹性已经进行了大量研究，而分散学习的弹性在很大程度上还没有研究。这一研究差距是本研究的主要动机，我们的目标是提高分散学习的Sybil中毒韧性。我们提出了SybilWall，这是一种创新算法，专注于提高分散学习对有针对性的Sybil中毒攻击的弹性。通过将基于Sybils之间相似性的抗Sybil聚集函数与一种新的概率八卦机制相结合，我们为可扩展的Sybil弹性分散学习建立了一个新的基准。一项全面的经验评估表明，SybilWall的性能优于为联合学习场景设计的现有最先进的解决方案，并且是在一系列对抗性攻击场景中获得一致准确性的唯一算法。我们还发现SybilWall降低了创建多个Sybils的效用，因为我们的评估显示，在使用较少Sybils的对手中，成功率更高。最后，我们对SybilWall提出了一些可能的改进建议，并强调了未来的研究方向。



## **20. Are aligned neural networks adversarially aligned?**

对齐的神经网络是相反对齐的吗？ cs.CL

**SubmitDate**: 2023-06-26    [abs](http://arxiv.org/abs/2306.15447v1) [paper-pdf](http://arxiv.org/pdf/2306.15447v1)

**Authors**: Nicholas Carlini, Milad Nasr, Christopher A. Choquette-Choo, Matthew Jagielski, Irena Gao, Anas Awadalla, Pang Wei Koh, Daphne Ippolito, Katherine Lee, Florian Tramer, Ludwig Schmidt

**Abstract**: Large language models are now tuned to align with the goals of their creators, namely to be "helpful and harmless." These models should respond helpfully to user questions, but refuse to answer requests that could cause harm. However, adversarial users can construct inputs which circumvent attempts at alignment. In this work, we study to what extent these models remain aligned, even when interacting with an adversarial user who constructs worst-case inputs (adversarial examples). These inputs are designed to cause the model to emit harmful content that would otherwise be prohibited. We show that existing NLP-based optimization attacks are insufficiently powerful to reliably attack aligned text models: even when current NLP-based attacks fail, we can find adversarial inputs with brute force. As a result, the failure of current attacks should not be seen as proof that aligned text models remain aligned under adversarial inputs.   However the recent trend in large-scale ML models is multimodal models that allow users to provide images that influence the text that is generated. We show these models can be easily attacked, i.e., induced to perform arbitrary un-aligned behavior through adversarial perturbation of the input image. We conjecture that improved NLP attacks may demonstrate this same level of adversarial control over text-only models.

摘要: 大型语言模型现在被调整为与它们的创建者的目标保持一致，即“有益和无害”。这些模型应该对用户的问题做出有益的回应，但拒绝回答可能造成伤害的请求。然而，敌意用户可以构建绕过对齐尝试的输入。在这项工作中，我们研究这些模型在多大程度上保持一致，即使当与构建最坏情况输入的敌意用户交互时(对抗性例子)。这些输入旨在导致模型排放本来被禁止的有害内容。我们证明了现有的基于NLP的优化攻击不足以可靠地攻击对齐的文本模型：即使当前基于NLP的攻击失败，我们也可以发现具有暴力的敌意输入。因此，当前攻击的失败不应被视为对齐的文本模型在敌意输入下保持对齐的证据。然而，大规模ML模型的最新趋势是允许用户提供影响所生成文本的图像的多模式模型。我们证明了这些模型可以很容易地被攻击，即通过对输入图像的对抗性扰动来诱导执行任意的非对齐行为。我们推测，改进的NLP攻击可能会展示出对纯文本模型的同样水平的敌意控制。



## **21. On the Resilience of Machine Learning-Based IDS for Automotive Networks**

基于机器学习的车载网络入侵检测系统的弹性研究 cs.CR

**SubmitDate**: 2023-06-26    [abs](http://arxiv.org/abs/2306.14782v1) [paper-pdf](http://arxiv.org/pdf/2306.14782v1)

**Authors**: Ivo Zenden, Han Wang, Alfonso Iacovazzi, Arash Vahidi, Rolf Blom, Shahid Raza

**Abstract**: Modern automotive functions are controlled by a large number of small computers called electronic control units (ECUs). These functions span from safety-critical autonomous driving to comfort and infotainment. ECUs communicate with one another over multiple internal networks using different technologies. Some, such as Controller Area Network (CAN), are very simple and provide minimal or no security services. Machine learning techniques can be used to detect anomalous activities in such networks. However, it is necessary that these machine learning techniques are not prone to adversarial attacks. In this paper, we investigate adversarial sample vulnerabilities in four different machine learning-based intrusion detection systems for automotive networks. We show that adversarial samples negatively impact three of the four studied solutions. Furthermore, we analyze transferability of adversarial samples between different systems. We also investigate detection performance and the attack success rate after using adversarial samples in the training. After analyzing these results, we discuss whether current solutions are mature enough for a use in modern vehicles.

摘要: 现代汽车的功能是由大量称为电子控制单元(ECU)的小型计算机控制的。这些功能的范围从对安全至关重要的自动驾驶到舒适和信息娱乐。ECU使用不同的技术通过多个内部网络相互通信。有些网络非常简单，例如控制器区域网络(CAN)，只提供最低限度的安全服务，或者根本不提供安全服务。机器学习技术可用于检测此类网络中的异常活动。然而，有必要的是，这些机器学习技术不容易受到对抗性攻击。在本文中，我们研究了四个不同的基于机器学习的汽车网络入侵检测系统的对手样本漏洞。我们发现，对抗性样本对所研究的四个解决方案中的三个产生了负面影响。此外，我们还分析了对抗性样本在不同系统之间的可转移性。我们还考察了在训练中使用对抗性样本后的检测性能和攻击成功率。在分析了这些结果之后，我们讨论了当前的解决方案是否足够成熟，可以在现代车辆中使用。



## **22. No Need to Know Physics: Resilience of Process-based Model-free Anomaly Detection for Industrial Control Systems**

无需了解物理：基于过程的工业控制系统无模型异常检测的弹性 cs.CR

An updated version of the paper has been published at ACSAC'2022:  Assessing Model-free Anomaly Detection in Industrial Control Systems Against  Generic Concealment Attacks https://dl.acm.org/doi/10.1145/3564625.3564633

**SubmitDate**: 2023-06-26    [abs](http://arxiv.org/abs/2012.03586v2) [paper-pdf](http://arxiv.org/pdf/2012.03586v2)

**Authors**: Alessandro Erba, Nils Ole Tippenhauer

**Abstract**: In recent years, a number of process-based anomaly detection schemes for Industrial Control Systems were proposed. In this work, we provide the first systematic analysis of such schemes, and introduce a taxonomy of properties that are verified by those detection systems. We then present a novel general framework to generate adversarial spoofing signals that violate physical properties of the system, and use the framework to analyze four anomaly detectors published at top security conferences. We find that three of those detectors are susceptible to a number of adversarial manipulations (e.g., spoofing with precomputed patterns), which we call Synthetic Sensor Spoofing and one is resilient against our attacks. We investigate the root of its resilience and demonstrate that it comes from the properties that we introduced. Our attacks reduce the Recall (True Positive Rate) of the attacked schemes making them not able to correctly detect anomalies. Thus, the vulnerabilities we discovered in the anomaly detectors show that (despite an original good detection performance), those detectors are not able to reliably learn physical properties of the system. Even attacks that prior work was expected to be resilient against (based on verified properties) were found to be successful. We argue that our findings demonstrate the need for both more complete attacks in datasets, and more critical analysis of process-based anomaly detectors. We plan to release our implementation as open-source, together with an extension of two public datasets with a set of Synthetic Sensor Spoofing attacks as generated by our framework.

摘要: 近年来，针对工业控制系统提出了许多基于过程的异常检测方案。在这项工作中，我们首次对这些方案进行了系统的分析，并介绍了由这些检测系统验证的属性的分类。然后，我们提出了一个新的通用框架来生成违反系统物理特性的敌意欺骗信号，并使用该框架分析了在顶级安全会议上发布的四个异常检测器。我们发现，其中三个检测器容易受到许多敌意操纵(例如，利用预计算模式进行欺骗)，我们称之为合成传感器欺骗，其中一个对我们的攻击具有弹性。我们研究了它的弹性的根源，并证明了它来自于我们引入的性质。我们的攻击降低了被攻击方案的召回率(True Positive Rate)，使得它们无法正确检测异常。因此，我们在异常检测器中发现的漏洞表明(尽管最初具有良好的检测性能)，这些检测器无法可靠地学习系统的物理属性。即使先前的工作被期望对(基于已验证的属性)具有弹性的攻击也被发现是成功的。我们认为，我们的发现表明，既需要在数据集中进行更完整的攻击，也需要对基于进程的异常检测器进行更关键的分析。我们计划将我们的实现作为开源发布，以及两个公共数据集的扩展，以及由我们的框架生成的一组合成传感器欺骗攻击。



## **23. PWSHAP: A Path-Wise Explanation Model for Targeted Variables**

PWSHAP：目标变量的路径智能解释模型 stat.ML

**SubmitDate**: 2023-06-26    [abs](http://arxiv.org/abs/2306.14672v1) [paper-pdf](http://arxiv.org/pdf/2306.14672v1)

**Authors**: Lucile Ter-Minassian, Oscar Clivio, Karla Diaz-Ordaz, Robin J. Evans, Chris Holmes

**Abstract**: Predictive black-box models can exhibit high accuracy but their opaque nature hinders their uptake in safety-critical deployment environments. Explanation methods (XAI) can provide confidence for decision-making through increased transparency. However, existing XAI methods are not tailored towards models in sensitive domains where one predictor is of special interest, such as a treatment effect in a clinical model, or ethnicity in policy models. We introduce Path-Wise Shapley effects (PWSHAP), a framework for assessing the targeted effect of a binary (e.g.~treatment) variable from a complex outcome model. Our approach augments the predictive model with a user-defined directed acyclic graph (DAG). The method then uses the graph alongside on-manifold Shapley values to identify effects along causal pathways whilst maintaining robustness to adversarial attacks. We establish error bounds for the identified path-wise Shapley effects and for Shapley values. We show PWSHAP can perform local bias and mediation analyses with faithfulness to the model. Further, if the targeted variable is randomised we can quantify local effect modification. We demonstrate the resolution, interpretability, and true locality of our approach on examples and a real-world experiment.

摘要: 预测黑盒模型可以显示出高精度，但其不透明的性质阻碍了它们在安全关键型部署环境中的应用。解释方法(XAI)可以通过增加透明度为决策提供信心。然而，现有的XAI方法不是针对敏感领域中的模型量身定做的，在这些领域中，一个预测因子是特别重要的，例如临床模型中的治疗效果，或政策模型中的种族。我们介绍了Path-Wise Shapley Effects(PWSHAP)，这是一个用于评估复杂结果模型中的二元变量(例如，~治疗)的靶向效应的框架。我们的方法使用用户定义的有向无环图(DAG)来扩充预测模型。然后，该方法使用该图和流形上的Shapley值来识别沿因果路径的影响，同时保持对对手攻击的健壮性。我们建立了已识别的路径Shapley效应和Shapley值的误差界。结果表明，PWSHAP能够在忠实于模型的前提下进行局部偏差分析和中介分析。此外，如果目标变量是随机的，我们可以量化局部效应修正。我们通过示例和真实世界的实验证明了我们方法的分辨率、可解释性和真正的局部性。



## **24. A Threat-Intelligence Driven Methodology to Incorporate Uncertainty in Cyber Risk Analysis and Enhance Decision Making**

一种威胁情报驱动的方法，将不确定性纳入网络风险分析并提高决策能力 cs.CR

**SubmitDate**: 2023-06-26    [abs](http://arxiv.org/abs/2302.13082v2) [paper-pdf](http://arxiv.org/pdf/2302.13082v2)

**Authors**: Martijn Dekker, Lampis Alevizos

**Abstract**: The challenge of decision-making under uncertainty in information security has become increasingly important, given the unpredictable probabilities and effects of events in the ever-changing cyber threat landscape. Cyber threat intelligence provides decision-makers with the necessary information and context to understand and anticipate potential threats, reducing uncertainty and improving the accuracy of risk analysis. The latter is a principal element of evidence-based decision-making, and it is essential to recognize that addressing uncertainty requires a new, threat-intelligence driven methodology and risk analysis approach. We propose a solution to this challenge by introducing a threat-intelligence based security assessment methodology and a decision-making strategy that considers both known unknowns and unknown unknowns. The proposed methodology aims to enhance the quality of decision-making by utilizing causal graphs, which offer an alternative to conventional methodologies that rely on attack trees, resulting in a reduction of uncertainty. Furthermore, we consider tactics, techniques, and procedures that are possible, probable, and plausible, improving the predictability of adversary behavior. Our proposed solution provides practical guidance for information security leaders to make informed decisions in uncertain situations. This paper offers a new perspective on addressing the challenge of decision-making under uncertainty in information security by introducing a methodology that can help decision-makers navigate the intricacies of the dynamic and continuously evolving landscape of cyber threats.

摘要: 鉴于瞬息万变的网络威胁格局中事件的不可预测的可能性和影响，信息安全中的不确定性决策的挑战变得越来越重要。网络威胁情报为决策者提供必要的信息和背景，以了解和预测潜在威胁，减少不确定性并提高风险分析的准确性。后者是基于证据的决策的一个主要因素，必须认识到，解决不确定性问题需要一种新的、以威胁情报为导向的方法和风险分析方法。我们提出了通过引入基于威胁情报的安全评估方法和既考虑已知未知因素又考虑未知未知因素的决策策略来解决这一挑战。该方法旨在通过使用因果图来提高决策的质量，它提供了一种替代依赖于攻击树的传统方法，从而减少了不确定性。此外，我们认为战术、技术和程序是可能的、可能的和看似合理的，可以改善对手行为的可预测性。我们提出的解决方案为信息安全领导者在不确定的情况下做出明智的决策提供了实用的指导。本文介绍了一种能够帮助决策者驾驭复杂的动态和不断变化的网络威胁格局的方法论，从而为解决信息安全中不确定因素下的决策挑战提供了一个新的视角。



## **25. 3D-Aware Adversarial Makeup Generation for Facial Privacy Protection**

面向人脸隐私保护的3D感知敌意化妆生成 cs.CV

Accepted by TPAMI 2023

**SubmitDate**: 2023-06-26    [abs](http://arxiv.org/abs/2306.14640v1) [paper-pdf](http://arxiv.org/pdf/2306.14640v1)

**Authors**: Yueming Lyu, Yue Jiang, Ziwen He, Bo Peng, Yunfan Liu, Jing Dong

**Abstract**: The privacy and security of face data on social media are facing unprecedented challenges as it is vulnerable to unauthorized access and identification. A common practice for solving this problem is to modify the original data so that it could be protected from being recognized by malicious face recognition (FR) systems. However, such ``adversarial examples'' obtained by existing methods usually suffer from low transferability and poor image quality, which severely limits the application of these methods in real-world scenarios. In this paper, we propose a 3D-Aware Adversarial Makeup Generation GAN (3DAM-GAN). which aims to improve the quality and transferability of synthetic makeup for identity information concealing. Specifically, a UV-based generator consisting of a novel Makeup Adjustment Module (MAM) and Makeup Transfer Module (MTM) is designed to render realistic and robust makeup with the aid of symmetric characteristics of human faces. Moreover, a makeup attack mechanism with an ensemble training strategy is proposed to boost the transferability of black-box models. Extensive experiment results on several benchmark datasets demonstrate that 3DAM-GAN could effectively protect faces against various FR models, including both publicly available state-of-the-art models and commercial face verification APIs, such as Face++, Baidu and Aliyun.

摘要: 社交媒体上人脸数据的隐私和安全面临着前所未有的挑战，因为它容易受到未经授权的访问和识别。解决这一问题的一种常见做法是修改原始数据，以防止其被恶意人脸识别(FR)系统识别。然而，现有方法得到的“对抗性例子”往往存在可转移性低、图像质量差等问题，这严重限制了这些方法在实际场景中的应用。在本文中，我们提出了一种3D感知的对抗性化妆生成GAN(3DAM-GAN)。旨在提高身份信息隐藏合成化妆的质量和可转移性。具体地说，设计了一种基于UV的生成器，该生成器由一个新颖的化妆调整模块(MAM)和化妆传输模块(MTM)组成，可以利用人脸的对称特征来渲染逼真和健壮的化妆。此外，为了提高黑盒模型的可转移性，提出了一种带有集成训练策略的补充攻击机制。在几个基准数据集上的广泛实验结果表明，3DAM-GAN可以有效地保护人脸免受各种FR模型的攻击，包括公开提供的最先进的模型和商业人脸验证API，如Face++、百度和阿里云。



## **26. The race to robustness: exploiting fragile models for urban camouflage and the imperative for machine learning security**

健壮性竞赛：利用脆弱的城市伪装模型和机器学习安全势在必行 cs.LG

Accepted to IEEE TENSYMP 2023

**SubmitDate**: 2023-06-26    [abs](http://arxiv.org/abs/2306.14609v1) [paper-pdf](http://arxiv.org/pdf/2306.14609v1)

**Authors**: Harriet Farlow, Matthew Garratt, Gavin Mount, Tim Lynar

**Abstract**: Adversarial Machine Learning (AML) represents the ability to disrupt Machine Learning (ML) algorithms through a range of methods that broadly exploit the architecture of deep learning optimisation. This paper presents Distributed Adversarial Regions (DAR), a novel method that implements distributed instantiations of computer vision-based AML attack methods that may be used to disguise objects from image recognition in both white and black box settings. We consider the context of object detection models used in urban environments, and benchmark the MobileNetV2, NasNetMobile and DenseNet169 models against a subset of relevant images from the ImageNet dataset. We evaluate optimal parameters (size, number and perturbation method), and compare to state-of-the-art AML techniques that perturb the entire image. We find that DARs can cause a reduction in confidence of 40.4% on average, but with the benefit of not requiring the entire image, or the focal object, to be perturbed. The DAR method is a deliberately simple approach where the intention is to highlight how an adversary with very little skill could attack models that may already be productionised, and to emphasise the fragility of foundational object detection models. We present this as a contribution to the field of ML security as well as AML. This paper contributes a novel adversarial method, an original comparison between DARs and other AML methods, and frames it in a new context - that of urban camouflage and the necessity for ML security and model robustness.

摘要: 对抗性机器学习(AML)代表了通过广泛利用深度学习优化体系结构的一系列方法来扰乱机器学习(ML)算法的能力。提出了一种新的分布式对抗区域(DAR)方法，它实现了基于计算机视觉的AML攻击方法的分布式实例化，该方法可用于在白盒和黑盒环境下伪装图像识别中的对象。我们考虑了城市环境中使用的目标检测模型的背景，并将MobileNetV2、NasNetMobile和DenseNet169模型与ImageNet数据集中相关图像的子集进行了基准测试。我们评估最佳参数(大小、数量和扰动方法)，并与扰乱整个图像的最先进的AML技术进行比较。我们发现，DARS平均可以导致40.4%的置信度下降，但好处是不需要扰动整个图像或焦点对象。DAR方法是一种刻意简单的方法，其目的是强调技能很低的对手如何攻击可能已被生产的模型，并强调基础目标检测模型的脆弱性。我们认为这是对ML安全和AML领域的贡献。本文提出了一种新的对抗方法，对DARS方法和其他AML方法进行了初步的比较，并将其置于一个新的背景下--城市伪装以及ML安全性和模型稳健性的必要性。



## **27. Towards Out-of-Distribution Adversarial Robustness**

向分布外对手稳健性迈进 cs.LG

Version of NeurIPS 2023 submission

**SubmitDate**: 2023-06-26    [abs](http://arxiv.org/abs/2210.03150v4) [paper-pdf](http://arxiv.org/pdf/2210.03150v4)

**Authors**: Adam Ibrahim, Charles Guille-Escuret, Ioannis Mitliagkas, Irina Rish, David Krueger, Pouya Bashivan

**Abstract**: Adversarial robustness continues to be a major challenge for deep learning. A core issue is that robustness to one type of attack often fails to transfer to other attacks. While prior work establishes a theoretical trade-off in robustness against different $L_p$ norms, we show that there is potential for improvement against many commonly used attacks by adopting a domain generalisation approach. Concretely, we treat each type of attack as a domain, and apply the Risk Extrapolation method (REx), which promotes similar levels of robustness against all training attacks. Compared to existing methods, we obtain similar or superior worst-case adversarial robustness on attacks seen during training. Moreover, we achieve superior performance on families or tunings of attacks only encountered at test time. On ensembles of attacks, our approach improves the accuracy from 3.4% with the best existing baseline to 25.9% on MNIST, and from 16.9% to 23.5% on CIFAR10.

摘要: 对抗的稳健性仍然是深度学习的主要挑战。一个核心问题是，对一种攻击的稳健性往往无法转移到其他攻击上。虽然以前的工作建立了对不同$L_p$范数的稳健性的理论权衡，但我们表明，通过采用域泛化方法，对许多常用的攻击有改进的潜力。具体地说，我们将每种类型的攻击视为一个域，并应用风险外推方法(REX)，该方法提高了对所有训练攻击的类似健壮性。与现有方法相比，对于训练过程中看到的攻击，我们获得了类似或更好的最坏情况下的对抗鲁棒性。此外，我们在仅在测试时遇到的攻击的家庭或调谐上实现了卓越的性能。在攻击集合上，我们的方法在MNIST上将准确率从3.4%提高到25.9%，在CIFAR10上从16.9%提高到23.5%。



## **28. Computational Asymmetries in Robust Classification**

稳健分类中的计算不对称性 cs.LG

**SubmitDate**: 2023-06-25    [abs](http://arxiv.org/abs/2306.14326v1) [paper-pdf](http://arxiv.org/pdf/2306.14326v1)

**Authors**: Samuele Marro, Michele Lombardi

**Abstract**: In the context of adversarial robustness, we make three strongly related contributions. First, we prove that while attacking ReLU classifiers is $\mathit{NP}$-hard, ensuring their robustness at training time is $\Sigma^2_P$-hard (even on a single example). This asymmetry provides a rationale for the fact that robust classifications approaches are frequently fooled in the literature. Second, we show that inference-time robustness certificates are not affected by this asymmetry, by introducing a proof-of-concept approach named Counter-Attack (CA). Indeed, CA displays a reversed asymmetry: running the defense is $\mathit{NP}$-hard, while attacking it is $\Sigma_2^P$-hard. Finally, motivated by our previous result, we argue that adversarial attacks can be used in the context of robustness certification, and provide an empirical evaluation of their effectiveness. As a byproduct of this process, we also release UG100, a benchmark dataset for adversarial attacks.

摘要: 在对抗健壮性的背景下，我们做出了三个强相关的贡献。首先，我们证明了当攻击RELU分类器是$\mathit{NP}$-困难时，确保它们在训练时的健壮性是$\Sigma^2_P$-困难(即使在单个例子上)。这种不对称性为健壮的分类方法在文献中经常被愚弄这一事实提供了理由。其次，通过引入一种名为反攻击(CA)的概念验证方法，我们证明了推理时健壮性证书不受这种不对称性的影响。事实上，CA表现出相反的不对称性：运行防御是$\mathit{NP}$-Hard，而攻击它是$\Sigma_2^P$-Hard。最后，基于我们之前的结果，我们论证了对抗性攻击可以在健壮性认证的背景下使用，并提供了对其有效性的经验评估。作为这一过程的副产品，我们还发布了UG100，这是一个用于对抗性攻击的基准数据集。



## **29. Enhancing Adversarial Training via Reweighting Optimization Trajectory**

通过重新加权优化轨迹加强对抗性训练 cs.LG

Accepted by ECML 2023

**SubmitDate**: 2023-06-25    [abs](http://arxiv.org/abs/2306.14275v1) [paper-pdf](http://arxiv.org/pdf/2306.14275v1)

**Authors**: Tianjin Huang, Shiwei Liu, Tianlong Chen, Meng Fang, Li Shen, Vlaod Menkovski, Lu Yin, Yulong Pei, Mykola Pechenizkiy

**Abstract**: Despite the fact that adversarial training has become the de facto method for improving the robustness of deep neural networks, it is well-known that vanilla adversarial training suffers from daunting robust overfitting, resulting in unsatisfactory robust generalization. A number of approaches have been proposed to address these drawbacks such as extra regularization, adversarial weights perturbation, and training with more data over the last few years. However, the robust generalization improvement is yet far from satisfactory. In this paper, we approach this challenge with a brand new perspective -- refining historical optimization trajectories. We propose a new method named \textbf{Weighted Optimization Trajectories (WOT)} that leverages the optimization trajectories of adversarial training in time. We have conducted extensive experiments to demonstrate the effectiveness of WOT under various state-of-the-art adversarial attacks. Our results show that WOT integrates seamlessly with the existing adversarial training methods and consistently overcomes the robust overfitting issue, resulting in better adversarial robustness. For example, WOT boosts the robust accuracy of AT-PGD under AA-$L_{\infty}$ attack by 1.53\% $\sim$ 6.11\% and meanwhile increases the clean accuracy by 0.55\%$\sim$5.47\% across SVHN, CIFAR-10, CIFAR-100, and Tiny-ImageNet datasets.

摘要: 尽管对抗性训练已经成为提高深度神经网络鲁棒性的事实上的方法，但众所周知，对抗性训练存在令人望而生畏的健壮性过拟合问题，导致不能令人满意的健壮泛化。在过去的几年里，已经提出了一些方法来解决这些缺点，例如额外的正则化、对抗性权重扰动和使用更多数据进行训练。然而，健壮的泛化改进还远远不能令人满意。在本文中，我们以一种全新的视角来应对这一挑战--提炼历史优化轨迹。我们提出了一种新的方法我们已经进行了大量的实验，以证明WOT在各种最先进的对抗性攻击下的有效性。实验结果表明，WOT算法与现有的对抗性训练方法无缝结合，始终克服了健壮性超调的问题，具有更好的对抗性。例如，WOT将AT-PGD在AA-L攻击下的稳健准确率提高了1.53$\sim$6.11\%，同时在SVHN、CIFAR-10、CIFAR-100和微型ImageNet数据集中将CLEAN准确率提高了0.55$\sim$5.47\%。



## **30. A Spectral Perspective towards Understanding and Improving Adversarial Robustness**

理解和提高对手稳健性的光谱视角 cs.CV

**SubmitDate**: 2023-06-25    [abs](http://arxiv.org/abs/2306.14262v1) [paper-pdf](http://arxiv.org/pdf/2306.14262v1)

**Authors**: Binxiao Huang, Rui Lin, Chaofan Tao, Ngai Wong

**Abstract**: Deep neural networks (DNNs) are incredibly vulnerable to crafted, imperceptible adversarial perturbations. While adversarial training (AT) has proven to be an effective defense approach, the AT mechanism for robustness improvement is not fully understood. This work investigates AT from a spectral perspective, adding new insights to the design of effective defenses. In particular, we show that AT induces the deep model to focus more on the low-frequency region, which retains the shape-biased representations, to gain robustness. Further, we find that the spectrum of a white-box attack is primarily distributed in regions the model focuses on, and the perturbation attacks the spectral bands where the model is vulnerable. Based on this observation, to train a model tolerant to frequency-varying perturbation, we propose a spectral alignment regularization (SAR) such that the spectral output inferred by an attacked adversarial input stays as close as possible to its natural input counterpart. Experiments demonstrate that SAR and its weight averaging (WA) extension could significantly improve the robust accuracy by 1.14% ~ 3.87% relative to the standard AT, across multiple datasets (CIFAR-10, CIFAR-100 and Tiny ImageNet), and various attacks (PGD, C&W and Autoattack), without any extra data.

摘要: 深度神经网络(DNN)非常容易受到精心设计的、不可察觉的对抗性扰动。虽然对抗训练(AT)已被证明是一种有效的防御方法，但AT提高健壮性的机制尚未完全被理解。这项工作从光谱的角度研究AT，为有效防御的设计增加了新的见解。特别是，我们证明了AT诱导深层模型更多地关注保留形状偏向表示的低频区域，以获得稳健性。此外，我们发现白盒攻击的频谱主要分布在模型关注的区域，而扰动则攻击模型易受攻击的频段。基于这一观察结果，为了训练一个容忍频率变化扰动的模型，我们提出了一种谱对齐正则化(SAR)，使得由攻击的对抗性输入推断出的谱输出尽可能地接近其自然输入。实验表明，在不增加任何额外数据的情况下，SAR及其加权平均(WA)扩展可以在多个数据集(CIFAR-10、CIFAR-100和Tiny ImageNet)以及各种攻击(PGD、C&W和AutoAttack)的情况下，比标准AT显著提高1.14%~3.87%的鲁棒准确率。



## **31. Backdoor Attacks in Peer-to-Peer Federated Learning**

对等联合学习中的后门攻击 cs.LG

**SubmitDate**: 2023-06-25    [abs](http://arxiv.org/abs/2301.09732v3) [paper-pdf](http://arxiv.org/pdf/2301.09732v3)

**Authors**: Gokberk Yar, Simona Boboila, Cristina Nita-Rotaru, Alina Oprea

**Abstract**: Most machine learning applications rely on centralized learning processes, opening up the risk of exposure of their training datasets. While federated learning (FL) mitigates to some extent these privacy risks, it relies on a trusted aggregation server for training a shared global model. Recently, new distributed learning architectures based on Peer-to-Peer Federated Learning (P2PFL) offer advantages in terms of both privacy and reliability. Still, their resilience to poisoning attacks during training has not been investigated. In this paper, we propose new backdoor attacks for P2PFL that leverage structural graph properties to select the malicious nodes, and achieve high attack success, while remaining stealthy. We evaluate our attacks under various realistic conditions, including multiple graph topologies, limited adversarial visibility of the network, and clients with non-IID data. Finally, we show the limitations of existing defenses adapted from FL and design a new defense that successfully mitigates the backdoor attacks, without an impact on model accuracy.

摘要: 大多数机器学习应用程序依赖于集中的学习过程，从而打开了暴露其训练数据集的风险。虽然联合学习(FL)在一定程度上缓解了这些隐私风险，但它依赖于可信的聚合服务器来训练共享的全局模型。近年来，基于对等联合学习(P2P-to-Peer Federated Learning，简称P2PFL)的新型分布式学习体系结构在保密性和可靠性方面都具有优势。尽管如此，他们在训练期间对中毒攻击的抵抗力还没有得到调查。本文提出了一种新的针对P2P PFL的后门攻击，利用结构图的性质来选择恶意节点，在保持隐蔽性的同时获得较高的攻击成功率。我们在各种现实条件下评估我们的攻击，包括多个图拓扑、有限的网络敌意可见性以及具有非IID数据的客户端。最后，我们指出了现有防御方案的局限性，并设计了一种新的防御方案，在不影响模型精度的情况下，成功地缓解了后门攻击。



## **32. On Evaluating the Adversarial Robustness of Semantic Segmentation Models**

语义分割模型的对抗健壮性评价 cs.CV

**SubmitDate**: 2023-06-25    [abs](http://arxiv.org/abs/2306.14217v1) [paper-pdf](http://arxiv.org/pdf/2306.14217v1)

**Authors**: Levente Halmosi, Mark Jelasity

**Abstract**: Achieving robustness against adversarial input perturbation is an important and intriguing problem in machine learning. In the area of semantic image segmentation, a number of adversarial training approaches have been proposed as a defense against adversarial perturbation, but the methodology of evaluating the robustness of the models is still lacking, compared to image classification. Here, we demonstrate that, just like in image classification, it is important to evaluate the models over several different and hard attacks. We propose a set of gradient based iterative attacks and show that it is essential to perform a large number of iterations. We include attacks against the internal representations of the models as well. We apply two types of attacks: maximizing the error with a bounded perturbation, and minimizing the perturbation for a given level of error. Using this set of attacks, we show for the first time that a number of models in previous work that are claimed to be robust are in fact not robust at all. We then evaluate simple adversarial training algorithms that produce reasonably robust models even under our set of strong attacks. Our results indicate that a key design decision to achieve any robustness is to use only adversarial examples during training. However, this introduces a trade-off between robustness and accuracy.

摘要: 实现对敌意输入扰动的鲁棒性是机器学习中一个重要而有趣的问题。在语义图像分割领域，已经提出了一些对抗性训练方法来对抗对抗性扰动，但与图像分类相比，目前还缺乏评估模型稳健性的方法。在这里，我们演示了，就像在图像分类中一样，评估模型在几种不同的硬攻击中是重要的。我们提出了一组基于梯度的迭代攻击，并证明了执行大量迭代是必要的。我们还包括对模型的内部表示的攻击。我们应用了两种类型的攻击：在有界扰动下最大化误差，以及在给定的误差水平下最小化扰动。使用这组攻击，我们第一次表明，在以前的工作中，许多声称健壮的模型实际上根本不健壮。然后，我们评估简单的对抗性训练算法，这些算法即使在我们的一组强大攻击下也能产生相当健壮的模型。我们的结果表明，实现任何稳健性的关键设计决策是在训练过程中仅使用对抗性示例。然而，这引入了稳健性和准确性之间的权衡。



## **33. The defender's perspective on automatic speaker verification: An overview**

辩护人对自动说话人确认的观点：综述 cs.SD

Accepted to IJCAI 2023 Workshop

**SubmitDate**: 2023-06-25    [abs](http://arxiv.org/abs/2305.12804v2) [paper-pdf](http://arxiv.org/pdf/2305.12804v2)

**Authors**: Haibin Wu, Jiawen Kang, Lingwei Meng, Helen Meng, Hung-yi Lee

**Abstract**: Automatic speaker verification (ASV) plays a critical role in security-sensitive environments. Regrettably, the reliability of ASV has been undermined by the emergence of spoofing attacks, such as replay and synthetic speech, as well as adversarial attacks and the relatively new partially fake speech. While there are several review papers that cover replay and synthetic speech, and adversarial attacks, there is a notable gap in a comprehensive review that addresses defense against adversarial attacks and the recently emerged partially fake speech. Thus, the aim of this paper is to provide a thorough and systematic overview of the defense methods used against these types of attacks.

摘要: 自动说话人验证(ASV)在安全敏感环境中起着至关重要的作用。令人遗憾的是，诸如重放和合成语音等欺骗性攻击的出现，以及对抗性攻击和相对较新的部分虚假语音的出现，破坏了ASV的可靠性。虽然有几篇评论论文涵盖了重播和合成演讲，以及对抗性攻击，但在针对对抗性攻击和最近出现的部分虚假演讲的防御的全面评论中，有一个明显的空白。因此，本文的目的是对针对这些类型的攻击所使用的防御方法进行全面和系统的概述。



## **34. Robust Spatiotemporal Traffic Forecasting with Reinforced Dynamic Adversarial Training**

增强动态对抗性训练的稳健时空交通预测 cs.LG

Accepted by KDD 2023

**SubmitDate**: 2023-06-25    [abs](http://arxiv.org/abs/2306.14126v1) [paper-pdf](http://arxiv.org/pdf/2306.14126v1)

**Authors**: Fan Liu, Weijia Zhang, Hao Liu

**Abstract**: Machine learning-based forecasting models are commonly used in Intelligent Transportation Systems (ITS) to predict traffic patterns and provide city-wide services. However, most of the existing models are susceptible to adversarial attacks, which can lead to inaccurate predictions and negative consequences such as congestion and delays. Therefore, improving the adversarial robustness of these models is crucial for ITS. In this paper, we propose a novel framework for incorporating adversarial training into spatiotemporal traffic forecasting tasks. We demonstrate that traditional adversarial training methods designated for static domains cannot be directly applied to traffic forecasting tasks, as they fail to effectively defend against dynamic adversarial attacks. Then, we propose a reinforcement learning-based method to learn the optimal node selection strategy for adversarial examples, which simultaneously strengthens the dynamic attack defense capability and reduces the model overfitting. Additionally, we introduce a self-knowledge distillation regularization module to overcome the "forgetting issue" caused by continuously changing adversarial nodes during training. We evaluate our approach on two real-world traffic datasets and demonstrate its superiority over other baselines. Our method effectively enhances the adversarial robustness of spatiotemporal traffic forecasting models. The source code for our framework is available at https://github.com/usail-hkust/RDAT.

摘要: 基于机器学习的预测模型是智能交通系统(ITS)中常用的预测模型，用于预测交通模式和提供全市范围的服务。然而，现有的大多数模型都容易受到对抗性攻击，这可能会导致不准确的预测和拥塞、延误等负面后果。因此，提高这些模型的对抗健壮性对ITS至关重要。在本文中，我们提出了一个新的框架，将对抗性训练融入到时空交通预测任务中。我们证明了传统的针对静态领域的对抗性训练方法不能直接应用于流量预测任务，因为它们不能有效地防御动态对抗性攻击。在此基础上，提出了一种基于强化学习的对抗性实例最优节点选择策略学习方法，在增强动态攻击防御能力的同时减少了模型的过度拟合。此外，我们还引入了自知识提炼正则化模块来克服训练过程中不断变化的敌方节点所带来的“遗忘问题”。我们在两个真实的流量数据集上对我们的方法进行了评估，并证明了它比其他基线方法的优越性。该方法有效地增强了时空交通预测模型的对抗性。我们框架的源代码可以在https://github.com/usail-hkust/RDAT.上找到



## **35. Identifying Adversarially Attackable and Robust Samples**

识别恶意攻击和健壮样本 cs.LG

**SubmitDate**: 2023-06-25    [abs](http://arxiv.org/abs/2301.12896v3) [paper-pdf](http://arxiv.org/pdf/2301.12896v3)

**Authors**: Vyas Raina, Mark Gales

**Abstract**: Adversarial attacks insert small, imperceptible perturbations to input samples that cause large, undesired changes to the output of deep learning models. Despite extensive research on generating adversarial attacks and building defense systems, there has been limited research on understanding adversarial attacks from an input-data perspective. This work introduces the notion of sample attackability, where we aim to identify samples that are most susceptible to adversarial attacks (attackable samples) and conversely also identify the least susceptible samples (robust samples). We propose a deep-learning-based detector to identify the adversarially attackable and robust samples in an unseen dataset for an unseen target model. Experiments on standard image classification datasets enables us to assess the portability of the deep attackability detector across a range of architectures. We find that the deep attackability detector performs better than simple model uncertainty-based measures for identifying the attackable/robust samples. This suggests that uncertainty is an inadequate proxy for measuring sample distance to a decision boundary. In addition to better understanding adversarial attack theory, it is found that the ability to identify the adversarially attackable and robust samples has implications for improving the efficiency of sample-selection tasks.

摘要: 对抗性攻击在输入样本中插入微小的、不可察觉的扰动，从而导致深度学习模型的输出发生巨大的、不希望看到的变化。尽管对生成对抗性攻击和建立防御系统进行了广泛的研究，但从输入数据的角度理解对抗性攻击的研究有限。这项工作引入了样本可攻击性的概念，其中我们的目标是识别最容易受到对手攻击的样本(可攻击样本)，反过来也识别最不敏感的样本(稳健样本)。我们提出了一种基于深度学习的检测器来识别不可见目标模型中不可见数据集中的可攻击样本和稳健样本。在标准图像分类数据集上的实验使我们能够评估深度可攻击性检测器在一系列体系结构中的可移植性。我们发现，深度可攻击性检测器在识别可攻击/稳健样本方面比基于简单模型不确定性的度量方法表现得更好。这表明，不确定性不足以衡量样本到决策边界的距离。除了更好地理解敌意攻击理论外，研究发现，识别敌意可攻击和稳健样本的能力对于提高样本选择任务的效率具有重要意义。



## **36. Sentiment Perception Adversarial Attacks on Neural Machine Translation Systems**

神经机器翻译系统中情感感知的敌意攻击 cs.CL

**SubmitDate**: 2023-06-25    [abs](http://arxiv.org/abs/2305.01437v2) [paper-pdf](http://arxiv.org/pdf/2305.01437v2)

**Authors**: Vyas Raina, Mark Gales

**Abstract**: With the advent of deep learning methods, Neural Machine Translation (NMT) systems have become increasingly powerful. However, deep learning based systems are susceptible to adversarial attacks, where imperceptible changes to the input can cause undesirable changes at the output of the system. To date there has been little work investigating adversarial attacks on sequence-to-sequence systems, such as NMT models. Previous work in NMT has examined attacks with the aim of introducing target phrases in the output sequence. In this work, adversarial attacks for NMT systems are explored from an output perception perspective. Thus the aim of an attack is to change the perception of the output sequence, without altering the perception of the input sequence. For example, an adversary may distort the sentiment of translated reviews to have an exaggerated positive sentiment. In practice it is challenging to run extensive human perception experiments, so a proxy deep-learning classifier applied to the NMT output is used to measure perception changes. Experiments demonstrate that the sentiment perception of NMT systems' output sequences can be changed significantly with small imperceptible changes to input sequences.

摘要: 随着深度学习方法的出现，神经机器翻译(NMT)系统变得越来越强大。然而，基于深度学习的系统很容易受到对抗性攻击，在这种攻击中，输入的不可察觉的变化可能会导致系统输出的不希望看到的变化。到目前为止，很少有人研究针对序列到序列系统的对抗性攻击，例如NMT模型。NMT之前的工作已经检查了攻击，目的是在输出序列中引入目标短语。本文从输出感知的角度研究了NMT系统的敌意攻击问题。因此，攻击的目的是改变对输出序列的感知，而不改变对输入序列的感知。例如，对手可能会扭曲翻译后的评论的情绪，使其具有夸大的积极情绪。在实践中，进行广泛的人类感知实验是具有挑战性的，因此将代理深度学习分类器应用于NMT输出来测量感知变化。实验表明，NMT系统输出序列的情感感知可以在输入序列微小的不可察觉变化的情况下发生显着变化。



## **37. Machine Learning needs its own Randomness Standard: Randomised Smoothing and PRNG-based attacks**

机器学习需要自己的随机性标准：随机平滑和基于PRNG的攻击 cs.LG

**SubmitDate**: 2023-06-24    [abs](http://arxiv.org/abs/2306.14043v1) [paper-pdf](http://arxiv.org/pdf/2306.14043v1)

**Authors**: Pranav Dahiya, Ilia Shumailov, Ross Anderson

**Abstract**: Randomness supports many critical functions in the field of machine learning (ML) including optimisation, data selection, privacy, and security. ML systems outsource the task of generating or harvesting randomness to the compiler, the cloud service provider or elsewhere in the toolchain. Yet there is a long history of attackers exploiting poor randomness, or even creating it -- as when the NSA put backdoors in random number generators to break cryptography. In this paper we consider whether attackers can compromise an ML system using only the randomness on which they commonly rely. We focus our effort on Randomised Smoothing, a popular approach to train certifiably robust models, and to certify specific input datapoints of an arbitrary model. We choose Randomised Smoothing since it is used for both security and safety -- to counteract adversarial examples and quantify uncertainty respectively. Under the hood, it relies on sampling Gaussian noise to explore the volume around a data point to certify that a model is not vulnerable to adversarial examples. We demonstrate an entirely novel attack against it, where an attacker backdoors the supplied randomness to falsely certify either an overestimate or an underestimate of robustness. We demonstrate that such attacks are possible, that they require very small changes to randomness to succeed, and that they can be hard to detect. As an example, we hide an attack in the random number generator and show that the randomness tests suggested by NIST fail to detect it. We advocate updating the NIST guidelines on random number testing to make them more appropriate for safety-critical and security-critical machine-learning applications.

摘要: 随机性支持机器学习(ML)领域的许多关键功能，包括优化、数据选择、隐私和安全。ML系统将生成或获取随机性的任务外包给编译器、云服务提供商或工具链中的其他地方。然而，攻击者利用糟糕的随机性甚至创造随机性的历史由来已久--比如美国国家安全局在随机数生成器中放置后门来破解密码。在本文中，我们考虑攻击者是否可以仅使用他们通常依赖的随机性来危害ML系统。我们把精力集中在随机平滑上，这是一种流行的方法，用于训练可证明稳健的模型，并验证任意模型的特定输入数据点。我们选择随机平滑，因为它既是为了安全，也是为了安全--分别用来抵消敌意例子和量化不确定性。在幕后，它依靠采样高斯噪声来探索数据点周围的体积，以证明模型不容易受到对手例子的影响。我们演示了一种针对它的全新攻击，其中攻击者后门提供的随机性来错误地证明健壮性的高估或低估。我们证明了这样的攻击是可能的，它们需要对随机性进行非常小的改变才能成功，而且它们可能很难被检测到。作为一个例子，我们在随机数生成器中隐藏了一个攻击，并表明NIST建议的随机性测试无法检测到它。我们主张更新NIST关于随机数测试的指南，使其更适合于安全关键和安全关键的机器学习应用程序。



## **38. Boosting Model Inversion Attacks with Adversarial Examples**

用对抗性例子增强模型反转攻击 cs.CR

18 pages, 13 figures

**SubmitDate**: 2023-06-24    [abs](http://arxiv.org/abs/2306.13965v1) [paper-pdf](http://arxiv.org/pdf/2306.13965v1)

**Authors**: Shuai Zhou, Tianqing Zhu, Dayong Ye, Xin Yu, Wanlei Zhou

**Abstract**: Model inversion attacks involve reconstructing the training data of a target model, which raises serious privacy concerns for machine learning models. However, these attacks, especially learning-based methods, are likely to suffer from low attack accuracy, i.e., low classification accuracy of these reconstructed data by machine learning classifiers. Recent studies showed an alternative strategy of model inversion attacks, GAN-based optimization, can improve the attack accuracy effectively. However, these series of GAN-based attacks reconstruct only class-representative training data for a class, whereas learning-based attacks can reconstruct diverse data for different training data in each class. Hence, in this paper, we propose a new training paradigm for a learning-based model inversion attack that can achieve higher attack accuracy in a black-box setting. First, we regularize the training process of the attack model with an added semantic loss function and, second, we inject adversarial examples into the training data to increase the diversity of the class-related parts (i.e., the essential features for classification tasks) in training data. This scheme guides the attack model to pay more attention to the class-related parts of the original data during the data reconstruction process. The experimental results show that our method greatly boosts the performance of existing learning-based model inversion attacks. Even when no extra queries to the target model are allowed, the approach can still improve the attack accuracy of reconstructed data. This new attack shows that the severity of the threat from learning-based model inversion adversaries is underestimated and more robust defenses are required.

摘要: 模型反转攻击涉及重建目标模型的训练数据，这给机器学习模型带来了严重的隐私问题。然而，这些攻击，特别是基于学习的方法，可能会受到攻击准确率的影响，即机器学习分类器对这些重建数据的分类精度较低。最近的研究表明，模型反转攻击的一种替代策略--基于遗传算法的优化，可以有效地提高攻击的准确性。然而，这一系列基于GAN的攻击只为一个类重建具有类代表性的训练数据，而基于学习的攻击可以为每个类中的不同训练数据重建不同的数据。因此，在本文中，我们提出了一种新的训练范式，用于基于学习的模型反转攻击，在黑盒环境下可以达到更高的攻击精度。首先，我们通过增加语义损失函数来规范攻击模型的训练过程；其次，我们在训练数据中注入对抗性例子，以增加训练数据中与类相关的部分(即分类任务的本质特征)的多样性。该方案引导攻击模型在数据重构过程中更加关注原始数据中与类相关的部分。实验结果表明，该方法大大提高了已有的基于学习的模型反转攻击的性能。即使在不允许对目标模型进行额外查询的情况下，该方法仍然可以提高重建数据的攻击准确率。这一新的攻击表明，来自基于学习的模型反转对手的威胁的严重性被低估了，需要更强大的防御。



## **39. Similarity Preserving Adversarial Graph Contrastive Learning**

保持相似性的对抗性图对比学习 cs.LG

9 pages; KDD'23

**SubmitDate**: 2023-06-24    [abs](http://arxiv.org/abs/2306.13854v1) [paper-pdf](http://arxiv.org/pdf/2306.13854v1)

**Authors**: Yeonjun In, Kanghoon Yoon, Chanyoung Park

**Abstract**: Recent works demonstrate that GNN models are vulnerable to adversarial attacks, which refer to imperceptible perturbation on the graph structure and node features. Among various GNN models, graph contrastive learning (GCL) based methods specifically suffer from adversarial attacks due to their inherent design that highly depends on the self-supervision signals derived from the original graph, which however already contains noise when the graph is attacked. To achieve adversarial robustness against such attacks, existing methods adopt adversarial training (AT) to the GCL framework, which considers the attacked graph as an augmentation under the GCL framework. However, we find that existing adversarially trained GCL methods achieve robustness at the expense of not being able to preserve the node feature similarity. In this paper, we propose a similarity-preserving adversarial graph contrastive learning (SP-AGCL) framework that contrasts the clean graph with two auxiliary views of different properties (i.e., the node similarity-preserving view and the adversarial view). Extensive experiments demonstrate that SP-AGCL achieves a competitive performance on several downstream tasks, and shows its effectiveness in various scenarios, e.g., a network with adversarial attacks, noisy labels, and heterophilous neighbors. Our code is available at https://github.com/yeonjun-in/torch-SP-AGCL.

摘要: 最近的工作表明，GNN模型容易受到敌意攻击，即对图结构和节点特征的不可察觉的扰动。在各种GNN模型中，基于图对比学习(GCL)的方法由于其固有的设计高度依赖于来自原始图的自我监督信号而特别容易受到对抗性攻击，然而当图被攻击时，该信号已经包含噪声。为了达到对抗这类攻击的健壮性，现有的方法对GCL框架采用对抗性训练(AT)，将被攻击图视为GCL框架下的增强。然而，我们发现，现有的对抗性训练的GCL方法以不能保持节点特征相似性为代价来实现健壮性。本文提出了一种保持相似性的对抗性图对比学习(SP-AGCL)框架，该框架将干净的图与两个不同性质的辅助视图(即保持节点相似性的视图和对抗性视图)进行对比。大量的实验表明，SP-AGCL在多个下行任务上取得了与之相当的性能，并在各种场景中显示了其有效性，例如，在具有对抗性攻击、噪声标签和异嗜性邻居的网络中。我们的代码可以在https://github.com/yeonjun-in/torch-SP-AGCL.上找到



## **40. A First Order Meta Stackelberg Method for Robust Federated Learning**

一种用于鲁棒联邦学习的一阶Meta Stackelberg方法 cs.LG

Accepted to ICML 2023 Workshop on The 2nd New Frontiers In  Adversarial Machine Learning. arXiv admin note: substantial text overlap with  arXiv:2306.13273

**SubmitDate**: 2023-06-23    [abs](http://arxiv.org/abs/2306.13800v1) [paper-pdf](http://arxiv.org/pdf/2306.13800v1)

**Authors**: Yunian Pan, Tao Li, Henger Li, Tianyi Xu, Zizhan Zheng, Quanyan Zhu

**Abstract**: Previous research has shown that federated learning (FL) systems are exposed to an array of security risks. Despite the proposal of several defensive strategies, they tend to be non-adaptive and specific to certain types of attacks, rendering them ineffective against unpredictable or adaptive threats. This work models adversarial federated learning as a Bayesian Stackelberg Markov game (BSMG) to capture the defender's incomplete information of various attack types. We propose meta-Stackelberg learning (meta-SL), a provably efficient meta-learning algorithm, to solve the equilibrium strategy in BSMG, leading to an adaptable FL defense. We demonstrate that meta-SL converges to the first-order $\varepsilon$-equilibrium point in $O(\varepsilon^{-2})$ gradient iterations, with $O(\varepsilon^{-4})$ samples needed per iteration, matching the state of the art. Empirical evidence indicates that our meta-Stackelberg framework performs exceptionally well against potent model poisoning and backdoor attacks of an uncertain nature.

摘要: 先前的研究表明，联合学习(FL)系统面临一系列安全风险。尽管提出了几种防御战略，但它们往往是非适应性的，并且特定于某些类型的攻击，使得它们对不可预测或适应性威胁无效。该工作将对抗性联邦学习建模为贝叶斯Stackelberg马尔可夫博弈(BSMG)，以捕获防御者各种攻击类型的不完全信息。我们提出了元Stackelberg学习算法(META-SL)来解决BSMG中的均衡策略，从而得到一种自适应的FL防御。我们证明了META-SL在$O(varepsilon^{-2})$梯度迭代中收敛到一阶$varepsilon$-均衡点，每次迭代需要$O(varepsilon^{-4})$样本，与现有技术相匹配。经验证据表明，我们的Meta-Stackelberg框架在对抗强大的模型中毒和不确定性质的后门攻击时表现得非常好。



## **41. Creating Valid Adversarial Examples of Malware**

创建有效的恶意软件对抗性示例 cs.CR

19 pages, 4 figures

**SubmitDate**: 2023-06-23    [abs](http://arxiv.org/abs/2306.13587v1) [paper-pdf](http://arxiv.org/pdf/2306.13587v1)

**Authors**: Matouš Kozák, Martin Jureček, Mark Stamp, Fabio Di Troia

**Abstract**: Machine learning is becoming increasingly popular as a go-to approach for many tasks due to its world-class results. As a result, antivirus developers are incorporating machine learning models into their products. While these models improve malware detection capabilities, they also carry the disadvantage of being susceptible to adversarial attacks. Although this vulnerability has been demonstrated for many models in white-box settings, a black-box attack is more applicable in practice for the domain of malware detection. We present a generator of adversarial malware examples using reinforcement learning algorithms. The reinforcement learning agents utilize a set of functionality-preserving modifications, thus creating valid adversarial examples. Using the proximal policy optimization (PPO) algorithm, we achieved an evasion rate of 53.84% against the gradient-boosted decision tree (GBDT) model. The PPO agent previously trained against the GBDT classifier scored an evasion rate of 11.41% against the neural network-based classifier MalConv and an average evasion rate of 2.31% against top antivirus programs. Furthermore, we discovered that random application of our functionality-preserving portable executable modifications successfully evades leading antivirus engines, with an average evasion rate of 11.65%. These findings indicate that machine learning-based models used in malware detection systems are vulnerable to adversarial attacks and that better safeguards need to be taken to protect these systems.

摘要: 由于其世界级的结果，机器学习作为许多任务的首选方法正变得越来越受欢迎。因此，反病毒开发人员正在将机器学习模型整合到他们的产品中。虽然这些模型提高了恶意软件检测能力，但它们也存在易受对手攻击的缺点。虽然该漏洞已经在白盒设置的许多型号中得到了演示，但黑盒攻击在实践中更适用于恶意软件检测领域。我们提出了一个使用强化学习算法的恶意软件实例生成器。强化学习代理利用一组保留功能的修改，从而创建有效的对抗性示例。使用最近策略优化(PPO)算法，相对于梯度增强决策树(GBDT)模型，我们获得了53.84%的逃避率。之前针对GBDT分类器训练的PPO代理对基于神经网络的分类器MalConv的逃避率为11.41%，对顶级防病毒程序的平均逃避率为2.31%。此外，我们发现，随机应用我们的功能保留可移植的可执行修改成功地躲避了领先的反病毒引擎，平均逃避率为11.65%。这些发现表明，恶意软件检测系统中使用的基于机器学习的模型容易受到对手攻击，需要采取更好的保护措施来保护这些系统。



## **42. A First Order Meta Stackelberg Method for Robust Federated Learning (Technical Report)**

一种稳健联邦学习的一阶Meta Stackelberg方法(技术报告) cs.CR

**SubmitDate**: 2023-06-23    [abs](http://arxiv.org/abs/2306.13273v1) [paper-pdf](http://arxiv.org/pdf/2306.13273v1)

**Authors**: Henger Li, Tianyi Xu, Tao Li, Yunian Pan, Quanyan Zhu, Zizhan Zheng

**Abstract**: Recent research efforts indicate that federated learning (FL) systems are vulnerable to a variety of security breaches. While numerous defense strategies have been suggested, they are mainly designed to counter specific attack patterns and lack adaptability, rendering them less effective when facing uncertain or adaptive threats. This work models adversarial FL as a Bayesian Stackelberg Markov game (BSMG) between the defender and the attacker to address the lack of adaptability to uncertain adaptive attacks. We further devise an effective meta-learning technique to solve for the Stackelberg equilibrium, leading to a resilient and adaptable defense. The experiment results suggest that our meta-Stackelberg learning approach excels in combating intense model poisoning and backdoor attacks of indeterminate types.

摘要: 最近的研究表明，联邦学习(FL)系统容易受到各种安全漏洞的攻击。虽然已经提出了许多防御策略，但它们主要是针对特定的攻击模式而设计的，缺乏适应性，使得它们在面临不确定或适应性威胁时效率较低。该工作将对抗性FL建模为防御者和攻击者之间的贝叶斯Stackelberg马尔可夫博弈(BSMG)，以解决对不确定自适应攻击缺乏适应性的问题。我们进一步设计了一种有效的元学习技术来求解Stackelberg均衡，从而导致具有弹性和适应性的防御。实验结果表明，我们的Meta-Stackelberg学习方法在对抗强烈的模型中毒和不确定类型的后门攻击方面表现出色。



## **43. Document Image Cleaning using Budget-Aware Black-Box Approximation**

基于预算感知黑盒近似的文档图像清洗 cs.CV

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2306.13236v1) [paper-pdf](http://arxiv.org/pdf/2306.13236v1)

**Authors**: Ganesh Tata, Katyani Singh, Eric Van Oeveren, Nilanjan Ray

**Abstract**: Recent work has shown that by approximating the behaviour of a non-differentiable black-box function using a neural network, the black-box can be integrated into a differentiable training pipeline for end-to-end training. This methodology is termed "differentiable bypass,'' and a successful application of this method involves training a document preprocessor to improve the performance of a black-box OCR engine. However, a good approximation of an OCR engine requires querying it for all samples throughout the training process, which can be computationally and financially expensive. Several zeroth-order optimization (ZO) algorithms have been proposed in black-box attack literature to find adversarial examples for a black-box model by computing its gradient in a query-efficient manner. However, the query complexity and convergence rate of such algorithms makes them infeasible for our problem. In this work, we propose two sample selection algorithms to train an OCR preprocessor with less than 10% of the original system's OCR engine queries, resulting in more than 60% reduction of the total training time without significant loss of accuracy. We also show an improvement of 4% in the word-level accuracy of a commercial OCR engine with only 2.5% of the total queries and a 32x reduction in monetary cost. Further, we propose a simple ranking technique to prune 30% of the document images from the training dataset without affecting the system's performance.

摘要: 最近的工作表明，通过使用神经网络近似不可微黑盒函数的行为，黑盒可以被集成到端到端训练的可微训练流水线中。这种方法被称为“可区分旁路”，这种方法的成功应用包括训练文档预处理器以提高黑盒OCR引擎的性能。然而，OCR引擎的良好近似性需要在整个训练过程中查询它的所有样本，这可能在计算和经济上都很昂贵。在黑盒攻击文献中已经提出了几种零阶优化(ZO)算法，通过以查询高效的方式计算黑盒模型的梯度来寻找黑盒模型的对抗性例子。然而，这些算法的查询复杂性和收敛速度使得它们不适用于我们的问题。在这项工作中，我们提出了两种样本选择算法来训练一个OCR预处理器，而原始系统的OCR引擎查询数不到10%，从而在不显著损失精度的情况下使总的训练时间减少了60%以上。我们还表明，商业OCR引擎的词级准确率提高了4%，而查询总数仅为2.5%，货币成本降低了32倍。此外，我们提出了一种简单的排序技术，在不影响系统性能的情况下，从训练数据集中剪除30%的文档图像。



## **44. Visual Adversarial Examples Jailbreak Large Language Models**

视觉对抗性示例越狱大型语言模型 cs.CR

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2306.13213v1) [paper-pdf](http://arxiv.org/pdf/2306.13213v1)

**Authors**: Xiangyu Qi, Kaixuan Huang, Ashwinee Panda, Mengdi Wang, Prateek Mittal

**Abstract**: Recently, there has been a surge of interest in introducing vision into Large Language Models (LLMs). The proliferation of large Visual Language Models (VLMs), such as Flamingo, BLIP-2, and GPT-4, signifies an exciting convergence of advancements in both visual and language foundation models. Yet, the risks associated with this integrative approach are largely unexamined. In this paper, we shed light on the security and safety implications of this trend. First, we underscore that the continuous and high-dimensional nature of the additional visual input space intrinsically makes it a fertile ground for adversarial attacks. This unavoidably expands the attack surfaces of LLMs. Second, we highlight that the broad functionality of LLMs also presents visual attackers with a wider array of achievable adversarial objectives, extending the implications of security failures beyond mere misclassification. To elucidate these risks, we study adversarial examples in the visual input space of a VLM. Specifically, against MiniGPT-4, which incorporates safety mechanisms that can refuse harmful instructions, we present visual adversarial examples that can circumvent the safety mechanisms and provoke harmful behaviors of the model. Remarkably, we discover that adversarial examples, even if optimized on a narrow, manually curated derogatory corpus against specific social groups, can universally jailbreak the model's safety mechanisms. A single such adversarial example can generally undermine MiniGPT-4's safety, enabling it to heed a wide range of harmful instructions and produce harmful content far beyond simply imitating the derogatory corpus used in optimization. Unveiling these risks, we accentuate the urgent need for comprehensive risk assessments, robust defense strategies, and the implementation of responsible practices for the secure and safe utilization of VLMs.

摘要: 最近，将VISION引入大型语言模型(LLM)的兴趣激增。诸如Flamingo、BLIP-2和GPT-4等大型可视化语言模型(VLM)的激增，标志着可视化和语言基础模型方面的进步令人兴奋地汇聚在一起。然而，与这种综合方法相关的风险在很大程度上没有得到审查。在这篇文章中，我们阐明了这一趋势的安全和安全影响。首先，我们强调，额外的视觉输入空间的连续性和高维性本质上使其成为敌方攻击的沃土。这不可避免地扩大了LLMS的攻击面。其次，我们强调，LLMS的广泛功能还为视觉攻击者提供了更广泛的可实现的对抗性目标，将安全故障的影响扩展到仅仅是错误分类。为了阐明这些风险，我们研究了VLM视觉输入空间中的对抗性例子。具体地说，针对MiniGPT-4，它包含了可以拒绝有害指令的安全机制，我们给出了可以绕过安全机制并引发模型有害行为的可视对抗性示例。值得注意的是，我们发现，对抗性的例子，即使在针对特定社会群体的狭窄的、手动管理的贬损语料库上进行了优化，也可以普遍地破解该模型的安全机制。一个这样的恶意例子通常会破坏小GPT-4的S安全，使其能够听取广泛的有害指令并产生有害内容，而不仅仅是简单地模仿优化中使用的贬损语料库。在揭示这些风险的同时，我们强调迫切需要全面的风险评估、强有力的防御战略和实施负责任的做法，以安全和安全地使用VLM。



## **45. Evading Forensic Classifiers with Attribute-Conditioned Adversarial Faces**

基于属性条件的对抗性面孔规避法医分类器 cs.CV

Accepted in CVPR 2023. Project page:  https://koushiksrivats.github.io/face_attribute_attack/

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2306.13091v1) [paper-pdf](http://arxiv.org/pdf/2306.13091v1)

**Authors**: Fahad Shamshad, Koushik Srivatsan, Karthik Nandakumar

**Abstract**: The ability of generative models to produce highly realistic synthetic face images has raised security and ethical concerns. As a first line of defense against such fake faces, deep learning based forensic classifiers have been developed. While these forensic models can detect whether a face image is synthetic or real with high accuracy, they are also vulnerable to adversarial attacks. Although such attacks can be highly successful in evading detection by forensic classifiers, they introduce visible noise patterns that are detectable through careful human scrutiny. Additionally, these attacks assume access to the target model(s) which may not always be true. Attempts have been made to directly perturb the latent space of GANs to produce adversarial fake faces that can circumvent forensic classifiers. In this work, we go one step further and show that it is possible to successfully generate adversarial fake faces with a specified set of attributes (e.g., hair color, eye size, race, gender, etc.). To achieve this goal, we leverage the state-of-the-art generative model StyleGAN with disentangled representations, which enables a range of modifications without leaving the manifold of natural images. We propose a framework to search for adversarial latent codes within the feature space of StyleGAN, where the search can be guided either by a text prompt or a reference image. We also propose a meta-learning based optimization strategy to achieve transferable performance on unknown target models. Extensive experiments demonstrate that the proposed approach can produce semantically manipulated adversarial fake faces, which are true to the specified attribute set and can successfully fool forensic face classifiers, while remaining undetectable by humans. Code: https://github.com/koushiksrivats/face_attribute_attack.

摘要: 生成模型生成高度逼真的合成人脸图像的能力引发了安全和伦理方面的担忧。作为对抗这种假脸的第一道防线，基于深度学习的法医分类器已经被开发出来。虽然这些取证模型可以高精度地检测人脸图像是合成的还是真实的，但它们也容易受到对手的攻击。虽然这类攻击可以非常成功地躲避法医分类器的检测，但它们引入了可通过仔细的人类检查检测到的可见噪声模式。此外，这些攻击假定可以访问目标模型(S)，但这可能并不总是正确的。有人试图直接扰乱Gans的潜在空间，以产生可以绕过法医分类器的敌对假面。在这项工作中，我们更进一步，证明了可以成功地生成具有特定属性集(例如，头发颜色、眼睛大小、种族、性别等)的敌意伪脸。为了实现这一目标，我们利用最先进的生成模型StyleGAN和分离的表示法，可以在不离开各种自然图像的情况下进行一系列修改。提出了一种在StyleGAN的特征空间内搜索敌意潜在码的框架，该框架可以通过文本提示或参考图像来指导搜索。我们还提出了一种基于元学习的优化策略，以实现在未知目标模型上的可转移性能。大量的实验表明，该方法可以生成符合指定属性集的语义操纵的敌意伪脸，并且能够成功地欺骗取证人脸分类器，而又不被人类发现。代码：https://github.com/koushiksrivats/face_attribute_attack.



## **46. Impacts and Risk of Generative AI Technology on Cyber Defense**

产生式人工智能技术对网络防御的影响和风险 cs.CR

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2306.13033v1) [paper-pdf](http://arxiv.org/pdf/2306.13033v1)

**Authors**: Subash Neupane, Ivan A. Fernandez, Sudip Mittal, Shahram Rahimi

**Abstract**: Generative Artificial Intelligence (GenAI) has emerged as a powerful technology capable of autonomously producing highly realistic content in various domains, such as text, images, audio, and videos. With its potential for positive applications in creative arts, content generation, virtual assistants, and data synthesis, GenAI has garnered significant attention and adoption. However, the increasing adoption of GenAI raises concerns about its potential misuse for crafting convincing phishing emails, generating disinformation through deepfake videos, and spreading misinformation via authentic-looking social media posts, posing a new set of challenges and risks in the realm of cybersecurity. To combat the threats posed by GenAI, we propose leveraging the Cyber Kill Chain (CKC) to understand the lifecycle of cyberattacks, as a foundational model for cyber defense. This paper aims to provide a comprehensive analysis of the risk areas introduced by the offensive use of GenAI techniques in each phase of the CKC framework. We also analyze the strategies employed by threat actors and examine their utilization throughout different phases of the CKC, highlighting the implications for cyber defense. Additionally, we propose GenAI-enabled defense strategies that are both attack-aware and adaptive. These strategies encompass various techniques such as detection, deception, and adversarial training, among others, aiming to effectively mitigate the risks posed by GenAI-induced cyber threats.

摘要: 生成性人工智能(GenAI)已经成为一项强大的技术，能够在文本、图像、音频和视频等各个领域自主产生高度逼真的内容。凭借其在创意艺术、内容生成、虚拟助手和数据合成方面的积极应用潜力，GenAI获得了极大的关注和采用。然而，越来越多的人使用GenAI引发了人们的担忧，即它可能被滥用来制作令人信服的钓鱼电子邮件，通过深度虚假视频产生虚假信息，以及通过看起来真实的社交媒体帖子传播错误信息，给网络安全领域带来了一系列新的挑战和风险。为了应对GenAI构成的威胁，我们建议利用网络杀伤链(CKC)来了解网络攻击的生命周期，作为网络防御的基础模型。本文旨在全面分析在CKC框架的每个阶段中攻击性使用GenAI技术所带来的风险领域。我们还分析了威胁参与者使用的策略，并检查了它们在CKC不同阶段的使用情况，强调了对网络防御的影响。此外，我们还提出了支持GenAI的攻击感知和自适应防御策略。这些战略包括各种技术，如检测、欺骗和对抗性训练等，旨在有效地缓解GenAI引发的网络威胁带来的风险。



## **47. AI Security for Geoscience and Remote Sensing: Challenges and Future Trends**

地球科学和遥感的人工智能安全：挑战和未来趋势 cs.CV

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2212.09360v2) [paper-pdf](http://arxiv.org/pdf/2212.09360v2)

**Authors**: Yonghao Xu, Tao Bai, Weikang Yu, Shizhen Chang, Peter M. Atkinson, Pedram Ghamisi

**Abstract**: Recent advances in artificial intelligence (AI) have significantly intensified research in the geoscience and remote sensing (RS) field. AI algorithms, especially deep learning-based ones, have been developed and applied widely to RS data analysis. The successful application of AI covers almost all aspects of Earth observation (EO) missions, from low-level vision tasks like super-resolution, denoising and inpainting, to high-level vision tasks like scene classification, object detection and semantic segmentation. While AI techniques enable researchers to observe and understand the Earth more accurately, the vulnerability and uncertainty of AI models deserve further attention, considering that many geoscience and RS tasks are highly safety-critical. This paper reviews the current development of AI security in the geoscience and RS field, covering the following five important aspects: adversarial attack, backdoor attack, federated learning, uncertainty and explainability. Moreover, the potential opportunities and trends are discussed to provide insights for future research. To the best of the authors' knowledge, this paper is the first attempt to provide a systematic review of AI security-related research in the geoscience and RS community. Available code and datasets are also listed in the paper to move this vibrant field of research forward.

摘要: 人工智能(AI)的最新进展极大地加强了地学和遥感(RS)领域的研究。人工智能算法，特别是基于深度学习的人工智能算法在遥感数据分析中得到了广泛的应用。人工智能的成功应用几乎涵盖了地球观测(EO)任务的方方面面，从超分辨率、去噪和修复等低层视觉任务，到场景分类、目标检测和语义分割等高级视觉任务。虽然人工智能技术使研究人员能够更准确地观察和了解地球，但考虑到许多地学和遥感任务是高度安全关键的，人工智能模型的脆弱性和不确定性值得进一步关注。本文回顾了人工智能安全在地学和遥感领域的发展现状，包括对抗性攻击、后门攻击、联邦学习、不确定性和可解释性五个重要方面。此外，还讨论了潜在的机会和趋势，为未来的研究提供了见解。据作者所知，本文是第一次对地学和遥感社区中与人工智能安全相关的研究进行系统回顾。文中还列出了可用的代码和数据集，以推动这一充满活力的研究领域向前发展。



## **48. Robust Semantic Segmentation: Strong Adversarial Attacks and Fast Training of Robust Models**

稳健语义分割：强对抗性攻击和稳健模型的快速训练 cs.CV

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2306.12941v1) [paper-pdf](http://arxiv.org/pdf/2306.12941v1)

**Authors**: Francesco Croce, Naman D Singh, Matthias Hein

**Abstract**: While a large amount of work has focused on designing adversarial attacks against image classifiers, only a few methods exist to attack semantic segmentation models. We show that attacking segmentation models presents task-specific challenges, for which we propose novel solutions. Our final evaluation protocol outperforms existing methods, and shows that those can overestimate the robustness of the models. Additionally, so far adversarial training, the most successful way for obtaining robust image classifiers, could not be successfully applied to semantic segmentation. We argue that this is because the task to be learned is more challenging, and requires significantly higher computational effort than for image classification. As a remedy, we show that by taking advantage of recent advances in robust ImageNet classifiers, one can train adversarially robust segmentation models at limited computational cost by fine-tuning robust backbones.

摘要: 虽然大量的工作集中在设计针对图像分类器的对抗性攻击，但只有少数方法存在攻击语义分割模型。我们发现攻击分段模型带来了特定于任务的挑战，为此我们提出了新的解决方案。我们的最终评估协议的性能优于现有的方法，并表明这些方法可能高估了模型的稳健性。此外，对抗性训练是获得稳健的图像分类器的最成功的方法，到目前为止还不能成功地应用于语义分割。我们认为这是因为要学习的任务更具挑战性，并且需要比图像分类更高的计算工作量。作为补救措施，我们表明，通过利用健壮ImageNet分类器的最新进展，可以通过微调健壮的主干来以有限的计算成本训练相反的健壮分割模型。



## **49. Cross-lingual Cross-temporal Summarization: Dataset, Models, Evaluation**

跨语言跨期摘要：数据集、模型、评价 cs.CL

Work in progress

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2306.12916v1) [paper-pdf](http://arxiv.org/pdf/2306.12916v1)

**Authors**: Ran Zhang, Jihed Ouni, Steffen Eger

**Abstract**: While summarization has been extensively researched in natural language processing (NLP), cross-lingual cross-temporal summarization (CLCTS) is a largely unexplored area that has the potential to improve cross-cultural accessibility, information sharing, and understanding. This paper comprehensively addresses the CLCTS task, including dataset creation, modeling, and evaluation. We build the first CLCTS corpus, leveraging historical fictive texts and Wikipedia summaries in English and German, and examine the effectiveness of popular transformer end-to-end models with different intermediate task finetuning tasks. Additionally, we explore the potential of ChatGPT for CLCTS as a summarizer and an evaluator. Overall, we report evaluations from humans, ChatGPT, and several recent automatic evaluation metrics where we find our intermediate task finetuned end-to-end models generate bad to moderate quality summaries; ChatGPT as a summarizer (without any finetuning) provides moderate to good quality outputs and as an evaluator correlates moderately with human evaluations though it is prone to giving lower scores. ChatGPT also seems to be very adept at normalizing historical text. We finally test ChatGPT in a scenario with adversarially attacked and unseen source documents and find that ChatGPT is better at omission and entity swap than negating against its prior knowledge.

摘要: 摘要在自然语言处理(NLP)领域得到了广泛的研究，而跨语言的跨时序摘要(CLCTS)在很大程度上是一个未被开发的领域，它有可能改善跨文化的可获得性、信息共享和理解。本文全面介绍了CLCTS的任务，包括数据集的创建、建模和评估。我们构建了第一个CLCTS语料库，利用英语和德语的历史虚构文本和维基百科摘要，并检查了具有不同中间任务精调任务的流行变压器端到端模型的有效性。此外，我们还探讨了ChatGPT作为CLCTS的摘要和评价器的潜力。总体而言，我们报告了来自人工、ChatGPT和最近几个自动评估指标的评估，在这些评估指标中，我们发现我们的中间任务微调的端到端模型生成了较差到中等质量的摘要；ChatGPT作为汇总器(没有任何微调)提供了中等到良好的质量输出，并且作为评估者与人工评估适度相关，尽管它倾向于给出较低的分数。ChatGPT似乎也非常擅长将历史文本正常化。最后，我们在恶意攻击和不可见源文档的情况下对ChatGPT进行了测试，发现ChatGPT在遗漏和实体交换方面比否认其先验知识更好。



## **50. On the explainable properties of 1-Lipschitz Neural Networks: An Optimal Transport Perspective**

1-Lipschitz神经网络的可解释性：最优传输观点 cs.AI

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2206.06854v2) [paper-pdf](http://arxiv.org/pdf/2206.06854v2)

**Authors**: Mathieu Serrurier, Franck Mamalet, Thomas Fel, Louis Béthune, Thibaut Boissin

**Abstract**: Input gradients have a pivotal role in a variety of applications, including adversarial attack algorithms for evaluating model robustness, explainable AI techniques for generating Saliency Maps, and counterfactual explanations. However, Saliency Maps generated by traditional neural networks are often noisy and provide limited insights. In this paper, we demonstrate that, on the contrary, the Saliency Maps of 1-Lipschitz neural networks, learnt with the dual loss of an optimal transportation problem, exhibit desirable XAI properties: They are highly concentrated on the essential parts of the image with low noise, significantly outperforming state-of-the-art explanation approaches across various models and metrics. We also prove that these maps align unprecedentedly well with human explanations on ImageNet. To explain the particularly beneficial properties of the Saliency Map for such models, we prove this gradient encodes both the direction of the transportation plan and the direction towards the nearest adversarial attack. Following the gradient down to the decision boundary is no longer considered an adversarial attack, but rather a counterfactual explanation that explicitly transports the input from one class to another. Thus, Learning with such a loss jointly optimizes the classification objective and the alignment of the gradient , i.e. the Saliency Map, to the transportation plan direction. These networks were previously known to be certifiably robust by design, and we demonstrate that they scale well for large problems and models, and are tailored for explainability using a fast and straightforward method.

摘要: 输入梯度在各种应用中具有举足轻重的作用，包括用于评估模型稳健性的对抗性攻击算法、用于生成显著图的可解释人工智能技术以及反事实解释。然而，传统神经网络生成的显著图往往噪声较大，提供的洞察力有限。相反，我们证明了在最优传输问题的对偶损失下学习的1-Lipschitz神经网络的显著图表现出理想的XAI性质：它们高度集中在低噪声的图像的基本部分，在各种模型和度量上显著优于最新的解释方法。我们还证明，这些地图与人们在ImageNet上的解释前所未有地吻合。为了解释显著图对这类模型特别有益的特性，我们证明了这种梯度既编码了运输计划的方向，也编码了指向最近的敌方攻击的方向。沿着梯度向下到决策边界不再被认为是对抗性攻击，而是一种反事实的解释，明确地将输入从一个类别传输到另一个类别。因此，具有这样的损失的学习联合优化了分类目标和梯度(即显著图)与运输计划方向的对准。众所周知，这些网络在设计上是可靠的，我们展示了它们对于大型问题和模型的良好伸缩性，并使用快速而直接的方法针对可解释性进行了定制。



