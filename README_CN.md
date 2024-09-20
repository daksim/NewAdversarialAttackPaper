# Latest Adversarial Attack Papers
**update at 2024-09-20 16:14:57**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Defending against Reverse Preference Attacks is Difficult**

防御反向偏好攻击很困难 cs.LG

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.12914v1) [paper-pdf](http://arxiv.org/pdf/2409.12914v1)

**Authors**: Domenic Rosati, Giles Edkins, Harsh Raj, David Atanasov, Subhabrata Majumdar, Janarthanan Rajendran, Frank Rudzicz, Hassan Sajjad

**Abstract**: While there has been progress towards aligning Large Language Models (LLMs) with human values and ensuring safe behaviour at inference time, safety-aligned LLMs are known to be vulnerable to training-time attacks such as supervised fine-tuning (SFT) on harmful datasets. In this paper, we ask if LLMs are vulnerable to adversarial reinforcement learning. Motivated by this goal, we propose Reverse Preference Attacks (RPA), a class of attacks to make LLMs learn harmful behavior using adversarial reward during reinforcement learning from human feedback (RLHF). RPAs expose a critical safety gap of safety-aligned LLMs in RL settings: they easily explore the harmful text generation policies to optimize adversarial reward. To protect against RPAs, we explore a host of mitigation strategies. Leveraging Constrained Markov-Decision Processes, we adapt a number of mechanisms to defend against harmful fine-tuning attacks into the RL setting. Our experiments show that ``online" defenses that are based on the idea of minimizing the negative log likelihood of refusals -- with the defender having control of the loss function -- can effectively protect LLMs against RPAs. However, trying to defend model weights using ``offline" defenses that operate under the assumption that the defender has no control over the loss function are less effective in the face of RPAs. These findings show that attacks done using RL can be used to successfully undo safety alignment in open-weight LLMs and use them for malicious purposes.

摘要: 虽然在使大型语言模型(LLM)与人类价值观保持一致并确保推理时的安全行为方面取得了进展，但众所周知，与安全保持一致的LLM容易受到训练时的攻击，例如对有害数据集的监督微调(SFT)。在本文中，我们询问LLMS是否容易受到对抗性强化学习的影响。基于这一目标，我们提出了反向偏好攻击(RPA)，这是一类在人类反馈强化学习(RLHF)过程中利用对抗性奖励使LLM学习有害行为的攻击。RPA暴露了RL环境中安全对齐的LLM的一个关键安全漏洞：它们很容易探索有害文本生成策略，以优化对抗性奖励。为了防范RPA，我们探索了一系列缓解策略。利用受限的马尔可夫决策过程，我们采用了一些机制来防御RL设置中的有害微调攻击。我们的实验表明，基于最小化拒绝的负对数可能性的思想的“在线”防御--防御者控制着损失函数--可以有效地保护LLM免受RPA的攻击。然而，试图使用在防御者无法控制损失函数的假设下运行的“离线”防御来捍卫模型权重，在面对RPA时效果较差。这些发现表明，使用RL进行的攻击可以成功地取消开放重量LLM中的安全对齐，并将其用于恶意目的。



## **2. Boosting Certified Robustness for Time Series Classification with Efficient Self-Ensemble**

通过高效的自我整合增强时间序列分类的认证鲁棒性 cs.LG

6 figures, 4 tables, 10 pages

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.02802v3) [paper-pdf](http://arxiv.org/pdf/2409.02802v3)

**Authors**: Chang Dong, Zhengyang Li, Liangwei Zheng, Weitong Chen, Wei Emma Zhang

**Abstract**: Recently, the issue of adversarial robustness in the time series domain has garnered significant attention. However, the available defense mechanisms remain limited, with adversarial training being the predominant approach, though it does not provide theoretical guarantees. Randomized Smoothing has emerged as a standout method due to its ability to certify a provable lower bound on robustness radius under $\ell_p$-ball attacks. Recognizing its success, research in the time series domain has started focusing on these aspects. However, existing research predominantly focuses on time series forecasting, or under the non-$\ell_p$ robustness in statistic feature augmentation for time series classification~(TSC). Our review found that Randomized Smoothing performs modestly in TSC, struggling to provide effective assurances on datasets with poor robustness. Therefore, we propose a self-ensemble method to enhance the lower bound of the probability confidence of predicted labels by reducing the variance of classification margins, thereby certifying a larger radius. This approach also addresses the computational overhead issue of Deep Ensemble~(DE) while remaining competitive and, in some cases, outperforming it in terms of robustness. Both theoretical analysis and experimental results validate the effectiveness of our method, demonstrating superior performance in robustness testing compared to baseline approaches.

摘要: 最近，时间序列域中的对抗性稳健性问题引起了人们的广泛关注。然而，现有的防御机制仍然有限，对抗性训练是主要的方法，尽管它不提供理论上的保证。由于随机化平滑方法能够证明在$ell_p$-ball攻击下的健壮性半径的一个可证明的下界，所以它已经成为一种优秀的方法。认识到它的成功，时间序列领域的研究已经开始集中在这些方面。然而，现有的研究主要集中在时间序列预测，或在统计特征增强对时间序列分类具有非埃尔p稳健性的情况下。我们的综述发现，随机平滑在TSC中表现平平，难以对稳健性较差的数据集提供有效的保证。因此，我们提出了一种自集成方法，通过减小分类裕度的方差来提高预测标签的概率置信度下界，从而证明更大的半径。这种方法还解决了深层集成~(DE)的计算开销问题，同时保持了竞争力，在某些情况下，在健壮性方面优于它。理论分析和实验结果都验证了该方法的有效性，在稳健性测试中表现出了优于基线方法的性能。



## **3. Deep generative models as an adversarial attack strategy for tabular machine learning**

深度生成模型作为表格机器学习的对抗攻击策略 cs.LG

Accepted at ICMLC 2024 (International Conference on Machine Learning  and Cybernetics)

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.12642v1) [paper-pdf](http://arxiv.org/pdf/2409.12642v1)

**Authors**: Salijona Dyrmishi, Mihaela Cătălina Stoian, Eleonora Giunchiglia, Maxime Cordy

**Abstract**: Deep Generative Models (DGMs) have found application in computer vision for generating adversarial examples to test the robustness of machine learning (ML) systems. Extending these adversarial techniques to tabular ML presents unique challenges due to the distinct nature of tabular data and the necessity to preserve domain constraints in adversarial examples. In this paper, we adapt four popular tabular DGMs into adversarial DGMs (AdvDGMs) and evaluate their effectiveness in generating realistic adversarial examples that conform to domain constraints.

摘要: 深度生成模型（DGM）已在计算机视觉中应用，用于生成对抗性示例来测试机器学习（ML）系统的稳健性。由于表格数据的独特性质以及在对抗性示例中保留域约束的必要性，将这些对抗性技术扩展到表格ML带来了独特的挑战。在本文中，我们将四种流行的表格式DGM调整为对抗性DGM（AdvDGM），并评估它们在生成符合领域约束的现实对抗性示例方面的有效性。



## **4. Exploring Privacy and Fairness Risks in Sharing Diffusion Models: An Adversarial Perspective**

探索共享扩散模型中的隐私和公平风险：对抗的视角 cs.LG

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2402.18607v3) [paper-pdf](http://arxiv.org/pdf/2402.18607v3)

**Authors**: Xinjian Luo, Yangfan Jiang, Fei Wei, Yuncheng Wu, Xiaokui Xiao, Beng Chin Ooi

**Abstract**: Diffusion models have recently gained significant attention in both academia and industry due to their impressive generative performance in terms of both sampling quality and distribution coverage. Accordingly, proposals are made for sharing pre-trained diffusion models across different organizations, as a way of improving data utilization while enhancing privacy protection by avoiding sharing private data directly. However, the potential risks associated with such an approach have not been comprehensively examined.   In this paper, we take an adversarial perspective to investigate the potential privacy and fairness risks associated with the sharing of diffusion models. Specifically, we investigate the circumstances in which one party (the sharer) trains a diffusion model using private data and provides another party (the receiver) black-box access to the pre-trained model for downstream tasks. We demonstrate that the sharer can execute fairness poisoning attacks to undermine the receiver's downstream models by manipulating the training data distribution of the diffusion model. Meanwhile, the receiver can perform property inference attacks to reveal the distribution of sensitive features in the sharer's dataset. Our experiments conducted on real-world datasets demonstrate remarkable attack performance on different types of diffusion models, which highlights the critical importance of robust data auditing and privacy protection protocols in pertinent applications.

摘要: 扩散模型由于其在抽样质量和分布复盖率方面令人印象深刻的生成性能，最近在学术界和工业界都得到了极大的关注。因此，提出了在不同组织之间共享预先培训的传播模式的建议，以此作为提高数据利用率的一种方式，同时通过避免直接共享私人数据来加强隐私保护。然而，与这种方法相关的潜在风险尚未得到全面审查。在这篇文章中，我们采取对抗性的视角来调查与共享扩散模型相关的潜在的隐私和公平风险。具体地说，我们调查了一方(共享者)使用私有数据训练扩散模型，并为另一方(接收者)提供对下游任务的预训练模型的黑箱访问的情况。我们证明了共享者可以通过操纵扩散模型的训练数据分布来执行公平毒化攻击来破坏接收者的下游模型。同时，接收者可以执行属性推断攻击，以揭示共享者数据集中敏感特征的分布。我们在真实数据集上进行的实验表明，在不同类型的扩散模型上具有显著的攻击性能，这突显了健壮的数据审计和隐私保护协议在相关应用中的关键重要性。



## **5. Adversarial Attack for Explanation Robustness of Rationalization Models**

对合理化模型解释稳健性的对抗攻击 cs.CL

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2408.10795v3) [paper-pdf](http://arxiv.org/pdf/2408.10795v3)

**Authors**: Yuankai Zhang, Lingxiao Kong, Haozhao Wang, Ruixuan Li, Jun Wang, Yuhua Li, Wei Liu

**Abstract**: Rationalization models, which select a subset of input text as rationale-crucial for humans to understand and trust predictions-have recently emerged as a prominent research area in eXplainable Artificial Intelligence. However, most of previous studies mainly focus on improving the quality of the rationale, ignoring its robustness to malicious attack. Specifically, whether the rationalization models can still generate high-quality rationale under the adversarial attack remains unknown. To explore this, this paper proposes UAT2E, which aims to undermine the explainability of rationalization models without altering their predictions, thereby eliciting distrust in these models from human users. UAT2E employs the gradient-based search on triggers and then inserts them into the original input to conduct both the non-target and target attack. Experimental results on five datasets reveal the vulnerability of rationalization models in terms of explanation, where they tend to select more meaningless tokens under attacks. Based on this, we make a series of recommendations for improving rationalization models in terms of explanation.

摘要: 合理化模型选择输入文本的一个子集作为理论基础--这对人类理解和信任预测至关重要--最近已成为可解释人工智能的一个重要研究领域。然而，以往的研究大多侧重于提高理论基础的质量，而忽略了其对恶意攻击的健壮性。具体地说，在对抗性攻击下，合理化模型是否仍能产生高质量的推理仍是未知的。为了探索这一点，本文提出了UAT2E，其目的是在不改变其预测的情况下削弱合理化模型的可解释性，从而引起人类用户对这些模型的不信任。UAT2E在触发器上采用基于梯度的搜索，然后将它们插入到原始输入中，以进行非目标攻击和目标攻击。在五个数据集上的实验结果揭示了合理化模型在解释方面的脆弱性，在攻击下，它们倾向于选择更多无意义的标记。在此基础上，本文从解释的角度提出了一系列改进合理化模型的建议。



## **6. Designing an attack-defense game: how to increase robustness of financial transaction models via a competition**

设计攻守游戏：如何通过竞争提高金融交易模型的稳健性 cs.LG

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2308.11406v3) [paper-pdf](http://arxiv.org/pdf/2308.11406v3)

**Authors**: Alexey Zaytsev, Maria Kovaleva, Alex Natekin, Evgeni Vorsin, Valerii Smirnov, Georgii Smirnov, Oleg Sidorshin, Alexander Senin, Alexander Dudin, Dmitry Berestnev

**Abstract**: Banks routinely use neural networks to make decisions. While these models offer higher accuracy, they are susceptible to adversarial attacks, a risk often overlooked in the context of event sequences, particularly sequences of financial transactions, as most works consider computer vision and NLP modalities.   We propose a thorough approach to studying these risks: a novel type of competition that allows a realistic and detailed investigation of problems in financial transaction data. The participants directly oppose each other, proposing attacks and defenses -- so they are examined in close-to-real-life conditions.   The paper outlines our unique competition structure with direct opposition of participants, presents results for several different top submissions, and analyzes the competition results. We also introduce a new open dataset featuring financial transactions with credit default labels, enhancing the scope for practical research and development.

摘要: 银行经常使用神经网络来做出决策。虽然这些模型提供了更高的准确性，但它们很容易受到对抗攻击，这一风险在事件序列（尤其是金融交易序列）的背景下经常被忽视，因为大多数作品都考虑计算机视觉和NLP模式。   我们提出了一种彻底的方法来研究这些风险：一种新型竞争，可以对金融交易数据中的问题进行现实而详细的调查。参与者直接相互反对，提出攻击和防御--因此他们在接近现实生活的条件下受到审查。   该论文概述了我们独特的竞争结构，参与者直接反对，列出了几种不同的顶级提交的结果，并分析了竞争结果。我们还引入了一个新的开放数据集，以带有信用违约标签的金融交易为特色，扩大了实践研究和开发的范围。



## **7. Magmaw: Modality-Agnostic Adversarial Attacks on Machine Learning-Based Wireless Communication Systems**

Magmaw：对基于机器学习的无线通信系统的模式不可知的对抗攻击 cs.CR

Accepted at NDSS 2025

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2311.00207v2) [paper-pdf](http://arxiv.org/pdf/2311.00207v2)

**Authors**: Jung-Woo Chang, Ke Sun, Nasimeh Heydaribeni, Seira Hidano, Xinyu Zhang, Farinaz Koushanfar

**Abstract**: Machine Learning (ML) has been instrumental in enabling joint transceiver optimization by merging all physical layer blocks of the end-to-end wireless communication systems. Although there have been a number of adversarial attacks on ML-based wireless systems, the existing methods do not provide a comprehensive view including multi-modality of the source data, common physical layer protocols, and wireless domain constraints. This paper proposes Magmaw, a novel wireless attack methodology capable of generating universal adversarial perturbations for any multimodal signal transmitted over a wireless channel. We further introduce new objectives for adversarial attacks on downstream applications. We adopt the widely-used defenses to verify the resilience of Magmaw. For proof-of-concept evaluation, we build a real-time wireless attack platform using a software-defined radio system. Experimental results demonstrate that Magmaw causes significant performance degradation even in the presence of strong defense mechanisms. Furthermore, we validate the performance of Magmaw in two case studies: encrypted communication channel and channel modality-based ML model.

摘要: 机器学习(ML)通过合并端到端无线通信系统的所有物理层块，在实现联合收发器优化方面发挥了重要作用。尽管已经有一些针对基于ML的无线系统的对抗性攻击，但现有的方法不能提供包括源数据的多模态、公共物理层协议和无线域限制在内的全面视角。本文提出了一种新的无线攻击方法Magmaw，它能够对无线信道上传输的任何多模信号产生通用的对抗性扰动。我们进一步引入了针对下游应用程序的对抗性攻击的新目标。我们采用了广泛使用的防御措施来验证Magmaw的弹性。对于概念验证评估，我们使用软件定义的无线电系统构建了一个实时无线攻击平台。实验结果表明，即使在强防御机制存在的情况下，Magmaw也会导致性能显著下降。此外，我们在加密通信通道和基于通道通道的ML模型两个案例中验证了MAGMAW的性能。



## **8. TEAM: Temporal Adversarial Examples Attack Model against Network Intrusion Detection System Applied to RNN**

TEAM：应用于RNN的针对网络入侵检测系统的时间对抗示例攻击模型 cs.CR

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.12472v1) [paper-pdf](http://arxiv.org/pdf/2409.12472v1)

**Authors**: Ziyi Liu, Dengpan Ye, Long Tang, Yunming Zhang, Jiacheng Deng

**Abstract**: With the development of artificial intelligence, neural networks play a key role in network intrusion detection systems (NIDS). Despite the tremendous advantages, neural networks are susceptible to adversarial attacks. To improve the reliability of NIDS, many research has been conducted and plenty of solutions have been proposed. However, the existing solutions rarely consider the adversarial attacks against recurrent neural networks (RNN) with time steps, which would greatly affect the application of NIDS in real world. Therefore, we first propose a novel RNN adversarial attack model based on feature reconstruction called \textbf{T}emporal adversarial \textbf{E}xamples \textbf{A}ttack \textbf{M}odel \textbf{(TEAM)}, which applied to time series data and reveals the potential connection between adversarial and time steps in RNN. That is, the past adversarial examples within the same time steps can trigger further attacks on current or future original examples. Moreover, TEAM leverages Time Dilation (TD) to effectively mitigates the effect of temporal among adversarial examples within the same time steps. Experimental results show that in most attack categories, TEAM improves the misjudgment rate of NIDS on both black and white boxes, making the misjudgment rate reach more than 96.68%. Meanwhile, the maximum increase in the misjudgment rate of the NIDS for subsequent original samples exceeds 95.57%.

摘要: 随着人工智能的发展，神经网络在网络入侵检测系统中发挥着关键作用。尽管神经网络具有巨大的优势，但它很容易受到对手的攻击。为了提高网络入侵检测系统的可靠性，人们进行了大量的研究，并提出了大量的解决方案。然而，现有的解决方案很少考虑对带时间步长的递归神经网络的对抗性攻击，这将极大地影响网络入侵检测系统在现实世界中的应用。为此，我们首先提出了一种新的基于特征重构的RNN对抗性攻击模型也就是说，在相同的时间步骤内的过去的对抗性例子可以触发对当前或未来的原始例子的进一步攻击。此外，团队利用时间膨胀(TD)来有效地缓解相同时间步长内的对抗性例子之间的时间效应。实验结果表明，在大多数攻击类别中，Team都提高了黑盒和白盒的误判率，使误判率达到96.68%以上。同时，网络入侵检测系统对后续原始样本的误判率最大增幅超过95.57%。



## **9. Object-fabrication Targeted Attack for Object Detection**

物体制造用于物体检测的定向攻击 cs.CV

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2212.06431v3) [paper-pdf](http://arxiv.org/pdf/2212.06431v3)

**Authors**: Xuchong Zhang, Changfeng Sun, Haoliang Han, Hongbin Sun

**Abstract**: Recent studies have demonstrated that object detection networks are usually vulnerable to adversarial examples. Generally, adversarial attacks for object detection can be categorized into targeted and untargeted attacks. Compared with untargeted attacks, targeted attacks present greater challenges and all existing targeted attack methods launch the attack by misleading detectors to mislabel the detected object as a specific wrong label. However, since these methods must depend on the presence of the detected objects within the victim image, they suffer from limitations in attack scenarios and attack success rates. In this paper, we propose a targeted feature space attack method that can mislead detectors to `fabricate' extra designated objects regardless of whether the victim image contains objects or not. Specifically, we introduce a guided image to extract coarse-grained features of the target objects and design an innovative dual attention mechanism to filter out the critical features of the target objects efficiently. The attack performance of the proposed method is evaluated on MS COCO and BDD100K datasets with FasterRCNN and YOLOv5. Evaluation results indicate that the proposed targeted feature space attack method shows significant improvements in terms of image-specific, universality, and generalization attack performance, compared with the previous targeted attack for object detection.

摘要: 最近的研究表明，目标检测网络通常容易受到敌意例子的攻击。通常，用于目标检测的对抗性攻击可以分为目标攻击和非目标攻击。与非定向攻击相比，定向攻击提出了更大的挑战，现有的所有定向攻击方法都是通过误导检测器将检测到的对象错误地标记为特定的错误标签来发起攻击。然而，由于这些方法必须依赖于受害者图像中检测到的对象的存在，它们在攻击场景和攻击成功率方面受到限制。在本文中，我们提出了一种目标特征空间攻击方法，该方法可以误导检测器‘捏造’额外的指定对象，而不管受害者图像中是否包含对象。具体地说，我们引入引导图像来提取目标对象的粗粒度特征，并设计了一种创新的双重注意机制来高效地过滤出目标对象的关键特征。使用FasterRCNN和YOLOv5对该方法在MS COCO和BDD100K数据集上的攻击性能进行了评估。评估结果表明，与已有的目标检测的目标攻击方法相比，本文提出的目标特征空间攻击方法在图像专用性、通用性和泛化性能方面都有明显的提高。



## **10. Towards Physically-Realizable Adversarial Attacks in Embodied Vision Navigation**

视觉导航中实现物理可实现的对抗攻击 cs.CV

8 pages, 6 figures, submitted to the 2025 IEEE International  Conference on Robotics & Automation (ICRA)

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.10071v2) [paper-pdf](http://arxiv.org/pdf/2409.10071v2)

**Authors**: Meng Chen, Jiawei Tu, Chao Qi, Yonghao Dang, Feng Zhou, Wei Wei, Jianqin Yin

**Abstract**: The deployment of embodied navigation agents in safety-critical environments raises concerns about their vulnerability to adversarial attacks on deep neural networks. However, current attack methods often lack practicality due to challenges in transitioning from the digital to the physical world, while existing physical attacks for object detection fail to achieve both multi-view effectiveness and naturalness. To address this, we propose a practical attack method for embodied navigation by attaching adversarial patches with learnable textures and opacity to objects. Specifically, to ensure effectiveness across varying viewpoints, we employ a multi-view optimization strategy based on object-aware sampling, which uses feedback from the navigation model to optimize the patch's texture. To make the patch inconspicuous to human observers, we introduce a two-stage opacity optimization mechanism, where opacity is refined after texture optimization. Experimental results show our adversarial patches reduce navigation success rates by about 40%, outperforming previous methods in practicality, effectiveness, and naturalness. Code is available at: [https://github.com/chen37058/Physical-Attacks-in-Embodied-Navigation].

摘要: 在安全关键环境中部署具体化导航代理引起了人们对它们在深层神经网络上易受敌意攻击的担忧。然而，由于从数字世界向物理世界过渡的挑战，现有的攻击方法往往缺乏实用性，而现有的针对目标检测的物理攻击无法达到多视角的有效性和自然性。为了解决这一问题，我们提出了一种实用的具身导航攻击方法，通过将具有可学习纹理和不透明度的敌意补丁附加到对象上。具体地说，为了确保不同视点的有效性，我们采用了一种基于对象感知采样的多视点优化策略，该策略利用导航模型的反馈来优化面片的纹理。为了使面片不易被人察觉，我们引入了一种两阶段不透明度优化机制，在纹理优化后对不透明度进行细化。实验结果表明，我们的对抗性补丁使导航成功率降低了约40%，在实用性、有效性和自然性方面都优于以往的方法。代码可从以下网址获得：[https://github.com/chen37058/Physical-Attacks-in-Embodied-Navigation].



## **11. Typography Leads Semantic Diversifying: Amplifying Adversarial Transferability across Multimodal Large Language Models**

字体设计引领语义多元化：增强多模式大型语言模型之间的对抗性可移植性 cs.CV

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2405.20090v2) [paper-pdf](http://arxiv.org/pdf/2405.20090v2)

**Authors**: Hao Cheng, Erjia Xiao, Jiayan Yang, Jiahang Cao, Le Yang, Jize Zhang, Kaidi Xu, Jindong Gu, Renjing Xu

**Abstract**: Following the advent of the Artificial Intelligence (AI) era of large models, Multimodal Large Language Models (MLLMs) with the ability to understand cross-modal interactions between vision and text have attracted wide attention. Adversarial examples with human-imperceptible perturbation are shown to possess a characteristic known as transferability, which means that a perturbation generated by one model could also mislead another different model. Augmenting the diversity in input data is one of the most significant methods for enhancing adversarial transferability. This method has been certified as a way to significantly enlarge the threat impact under black-box conditions. Research works also demonstrate that MLLMs can be exploited to generate adversarial examples in the white-box scenario. However, the adversarial transferability of such perturbations is quite limited, failing to achieve effective black-box attacks across different models. In this paper, we propose the Typographic-based Semantic Transfer Attack (TSTA), which is inspired by: (1) MLLMs tend to process semantic-level information; (2) Typographic Attack could effectively distract the visual information captured by MLLMs. In the scenarios of Harmful Word Insertion and Important Information Protection, our TSTA demonstrates superior performance.

摘要: 随着大模型人工智能时代的到来，能够理解视觉和文本之间跨通道交互的多通道大语言模型引起了人们的广泛关注。具有人类不可察觉的扰动的对抗性例子具有被称为可转移性的特征，这意味着一个模型产生的扰动也可能误导另一个不同的模型。增加输入数据的多样性是增强对抗性转移的最重要的方法之一。这种方法已被证明是一种在黑箱条件下显著扩大威胁影响的方法。研究工作还表明，在白盒情况下，MLLMS可以被用来生成对抗性示例。然而，此类扰动的对抗性可转移性相当有限，无法实现跨不同模型的有效黑盒攻击。本文提出了基于排版的语义传输攻击(TSTA)，其灵感来自：(1)MLLMS倾向于处理语义级的信息；(2)排版攻击可以有效地分散MLLMS捕获的视觉信息。在有害词语插入和重要信息保护的场景中，我们的TSTA表现出了卓越的性能。



## **12. ITPatch: An Invisible and Triggered Physical Adversarial Patch against Traffic Sign Recognition**

ITpatch：针对交通标志识别的隐形且触发的物理对抗补丁 cs.CV

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.12394v1) [paper-pdf](http://arxiv.org/pdf/2409.12394v1)

**Authors**: Shuai Yuan, Hongwei Li, Xingshuo Han, Guowen Xu, Wenbo Jiang, Tao Ni, Qingchuan Zhao, Yuguang Fang

**Abstract**: Physical adversarial patches have emerged as a key adversarial attack to cause misclassification of traffic sign recognition (TSR) systems in the real world. However, existing adversarial patches have poor stealthiness and attack all vehicles indiscriminately once deployed. In this paper, we introduce an invisible and triggered physical adversarial patch (ITPatch) with a novel attack vector, i.e., fluorescent ink, to advance the state-of-the-art. It applies carefully designed fluorescent perturbations to a target sign, an attacker can later trigger a fluorescent effect using invisible ultraviolet light, causing the TSR system to misclassify the sign and potentially resulting in traffic accidents. We conducted a comprehensive evaluation to investigate the effectiveness of ITPatch, which shows a success rate of 98.31% in low-light conditions. Furthermore, our attack successfully bypasses five popular defenses and achieves a success rate of 96.72%.

摘要: 物理对抗补丁已成为导致现实世界中交通标志识别（TSB）系统错误分类的关键对抗攻击。然而，现有的对抗补丁的隐蔽性较差，一旦部署，就会不加区别地攻击所有车辆。在本文中，我们引入了一种具有新型攻击载体的隐形触发物理对抗补丁（ITpatch），即，荧光墨水，推进最新技术水平。它将精心设计的荧光扰动应用于目标标志，攻击者随后可以使用不可见的紫外光触发荧光效应，导致TSB系统错误分类标志，并可能导致交通事故。我们进行了全面的评估来调查ITpatch的有效性，结果显示在弱光条件下的成功率为98.31%。此外，我们的攻击成功绕过了五种流行防御，成功率达到96.72%。



## **13. Enhancing 3D Robotic Vision Robustness by Minimizing Adversarial Mutual Information through a Curriculum Training Approach**

通过课程培训方法最大限度地减少对抗互信息，增强3D机器人视觉的稳健性 cs.CV

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.12379v1) [paper-pdf](http://arxiv.org/pdf/2409.12379v1)

**Authors**: Nastaran Darabi, Dinithi Jayasuriya, Devashri Naik, Theja Tulabandhula, Amit Ranjan Trivedi

**Abstract**: Adversarial attacks exploit vulnerabilities in a model's decision boundaries through small, carefully crafted perturbations that lead to significant mispredictions. In 3D vision, the high dimensionality and sparsity of data greatly expand the attack surface, making 3D vision particularly vulnerable for safety-critical robotics. To enhance 3D vision's adversarial robustness, we propose a training objective that simultaneously minimizes prediction loss and mutual information (MI) under adversarial perturbations to contain the upper bound of misprediction errors. This approach simplifies handling adversarial examples compared to conventional methods, which require explicit searching and training on adversarial samples. However, minimizing prediction loss conflicts with minimizing MI, leading to reduced robustness and catastrophic forgetting. To address this, we integrate curriculum advisors in the training setup that gradually introduce adversarial objectives to balance training and prevent models from being overwhelmed by difficult cases early in the process. The advisors also enhance robustness by encouraging training on diverse MI examples through entropy regularizers. We evaluated our method on ModelNet40 and KITTI using PointNet, DGCNN, SECOND, and PointTransformers, achieving 2-5% accuracy gains on ModelNet40 and a 5-10% mAP improvement in object detection. Our code is publicly available at https://github.com/nstrndrbi/Mine-N-Learn.

摘要: 对抗性攻击利用模型决策边界中的漏洞，通过精心设计的小扰动导致严重的错误预测。在3D视觉中，数据的高维性和稀疏性极大地扩大了攻击面，使得3D视觉特别容易受到安全关键型机器人的攻击。为了增强3D视觉的对抗鲁棒性，我们提出了一种训练目标，该目标同时最小化对抗扰动下的预测损失和互信息(MI)，以遏制误预测误差的上界。与传统方法相比，这种方法简化了对对抗性样本的处理，因为传统方法需要对对抗性样本进行明确的搜索和训练。然而，最小化预测损失与最小化MI相冲突，导致健壮性降低和灾难性遗忘。为了解决这个问题，我们在培训设置中整合了课程顾问，逐步引入对抗性目标，以平衡培训，并防止模型在过程早期被困难的案例淹没。顾问还通过鼓励通过熵正则化对不同的MI示例进行培训来增强稳健性。我们在ModelNet40和Kitti上使用PointNet、DGCNN、Second和PointTransformers对我们的方法进行了评估，在ModelNet40上获得了2%-5%的准确率改进，在目标检测方面获得了5%-10%的MAP改进。我们的代码在https://github.com/nstrndrbi/Mine-N-Learn.上公开提供



## **14. AirGapAgent: Protecting Privacy-Conscious Conversational Agents**

AirGapAgent：保护有隐私意识的对话代理人 cs.CR

at CCS'24

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2405.05175v2) [paper-pdf](http://arxiv.org/pdf/2405.05175v2)

**Authors**: Eugene Bagdasarian, Ren Yi, Sahra Ghalebikesabi, Peter Kairouz, Marco Gruteser, Sewoong Oh, Borja Balle, Daniel Ramage

**Abstract**: The growing use of large language model (LLM)-based conversational agents to manage sensitive user data raises significant privacy concerns. While these agents excel at understanding and acting on context, this capability can be exploited by malicious actors. We introduce a novel threat model where adversarial third-party apps manipulate the context of interaction to trick LLM-based agents into revealing private information not relevant to the task at hand.   Grounded in the framework of contextual integrity, we introduce AirGapAgent, a privacy-conscious agent designed to prevent unintended data leakage by restricting the agent's access to only the data necessary for a specific task. Extensive experiments using Gemini, GPT, and Mistral models as agents validate our approach's effectiveness in mitigating this form of context hijacking while maintaining core agent functionality. For example, we show that a single-query context hijacking attack on a Gemini Ultra agent reduces its ability to protect user data from 94% to 45%, while an AirGapAgent achieves 97% protection, rendering the same attack ineffective.

摘要: 越来越多地使用基于大型语言模型(LLM)的会话代理来管理敏感用户数据，这引发了严重的隐私问题。虽然这些代理擅长理解上下文并根据上下文执行操作，但这种能力可能会被恶意行为者利用。我们引入了一种新的威胁模型，在该模型中，敌意的第三方应用程序操纵交互的上下文，以欺骗基于LLM的代理泄露与手头任务无关的私人信息。基于上下文完整性的框架，我们引入了AirGapAgent，这是一个具有隐私意识的代理，旨在通过限制代理仅访问特定任务所需的数据来防止意外的数据泄露。使用Gemini、GPT和Mistral模型作为代理的大量实验验证了我们的方法在保持核心代理功能的同时缓解这种形式的上下文劫持的有效性。例如，我们表明，对Gemini Ultra代理的单查询上下文劫持攻击将其保护用户数据的能力从94%降低到45%，而AirGapAgent实现了97%的保护，使得相同的攻击无效。



## **15. LookAhead: Preventing DeFi Attacks via Unveiling Adversarial Contracts**

展望未来：通过揭露对抗性合同来防止DeFi攻击 cs.CR

21 pages, 7 figures

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2401.07261v4) [paper-pdf](http://arxiv.org/pdf/2401.07261v4)

**Authors**: Shoupeng Ren, Lipeng He, Tianyu Tu, Di Wu, Jian Liu, Kui Ren, Chun Chen

**Abstract**: Decentralized Finance (DeFi) incidents stemming from the exploitation of smart contract vulnerabilities have culminated in financial damages exceeding 3 billion US dollars. Existing defense mechanisms typically focus on detecting and reacting to malicious transactions executed by attackers that target victim contracts. However, with the emergence of private transaction pools where transactions are sent directly to miners without first appearing in public mempools, current detection tools face significant challenges in identifying attack activities effectively.   Based on the fact that most attack logic rely on deploying one or more intermediate smart contracts as supporting components to the exploitation of victim contracts, in this paper, we propose a new direction for detecting DeFi attacks that focuses on identifying adversarial contracts instead of adversarial transactions. Our approach allows us to leverage common attack patterns, code semantics and intrinsic characteristics found in malicious smart contracts to build the LookAhead system based on Machine Learning (ML) classifiers and a transformer model that is able to effectively distinguish adversarial contracts from benign ones, and make just-in-time predictions of potential zero-day attacks. Our contributions are three-fold: First, we construct a comprehensive dataset consisting of features extracted and constructed from recent contracts deployed on the Ethereum and BSC blockchains. Secondly, we design a condensed representation of smart contract programs called Pruned Semantic-Control Flow Tokenization (PSCFT) and use it to train a combination of ML models that understand the behaviour of malicious codes based on function calls, control flows and other pattern-conforming features. Lastly, we provide the complete implementation of LookAhead and the evaluation of its performance metrics for detecting adversarial contracts.

摘要: 因利用智能合同漏洞而引发的去中心化金融(Defi)事件已造成超过30亿美元的经济损失。现有的防御机制通常专注于检测和响应攻击者执行的针对受害者合同的恶意交易。然而，随着私人交易池的出现，交易直接发送给矿工，而不是首先出现在公共记忆池中，当前的检测工具在有效识别攻击活动方面面临重大挑战。基于大多数攻击逻辑依赖于部署一个或多个中间智能合约作为攻击受害者合约的支持组件的事实，本文提出了一种新的检测Defi攻击的方向，该方向侧重于识别对手合约而不是对手交易。我们的方法允许我们利用恶意智能合同中发现的常见攻击模式、代码语义和内在特征来构建基于机器学习(ML)分类器和转换器模型的前瞻性系统，该系统能够有效区分敌意合同和良性合同，并及时预测潜在的零日攻击。我们的贡献有三个方面：首先，我们构建了一个全面的数据集，其中包含从Etherum和BSC区块链上部署的最近合同中提取和构建的特征。其次，我们设计了智能合同程序的精简表示，称为剪枝语义控制流令牌化(PSCFT)，并使用它来训练ML模型的组合，这些模型基于函数调用、控制流和其他符合模式的特征来理解恶意代码的行为。最后，我们给出了LookHead的完整实现，并对其用于检测敌对合同的性能度量进行了评估。



## **16. Bi-objective trail-planning for a robot team orienteering in a hazardous environment**

危险环境中机器人队定向运动的双目标路径规划 cs.RO

v0.0

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2409.12114v1) [paper-pdf](http://arxiv.org/pdf/2409.12114v1)

**Authors**: Cory M. Simon, Jeffrey Richley, Lucas Overbey, Darleen Perez-Lavin

**Abstract**: Teams of mobile [aerial, ground, or aquatic] robots have applications in resource delivery, patrolling, information-gathering, agriculture, forest fire fighting, chemical plume source localization and mapping, and search-and-rescue. Robot teams traversing hazardous environments -- with e.g. rough terrain or seas, strong winds, or adversaries capable of attacking or capturing robots -- should plan and coordinate their trails in consideration of risks of disablement, destruction, or capture. Specifically, the robots should take the safest trails, coordinate their trails to cooperatively achieve the team-level objective with robustness to robot failures, and balance the reward from visiting locations against risks of robot losses. Herein, we consider bi-objective trail-planning for a mobile team of robots orienteering in a hazardous environment. The hazardous environment is abstracted as a directed graph whose arcs, when traversed by a robot, present known probabilities of survival. Each node of the graph offers a reward to the team if visited by a robot (which e.g. delivers a good to or images the node). We wish to search for the Pareto-optimal robot-team trail plans that maximize two [conflicting] team objectives: the expected (i) team reward and (ii) number of robots that survive the mission. A human decision-maker can then select trail plans that balance, according to their values, reward and robot survival. We implement ant colony optimization, guided by heuristics, to search for the Pareto-optimal set of robot team trail plans. As a case study, we illustrate with an information-gathering mission in an art museum.

摘要: 移动[空中、地面或水上]机器人团队在资源输送、巡逻、信息收集、农业、森林灭火、化学烟雾源定位和测绘以及搜救中有应用。机器人团队穿越危险环境--例如崎岖的地形或海洋、强风或有能力攻击或捕获机器人的敌人--应考虑到致残、破坏或捕获的风险来规划和协调他们的路径。具体地说，机器人应该选择最安全的路径，协调它们的路径，以协作地实现团队级目标，并对机器人故障具有鲁棒性，并在访问地点的回报和机器人损失的风险之间进行平衡。在这里，我们考虑在危险环境中定向的移动机器人团队的双目标路径规划。危险环境被抽象为一个有向图，当机器人穿过该图的弧线时，表示已知的生存概率。如果被机器人访问，图中的每个节点向团队提供奖励(例如，机器人向节点递送商品或为节点成像)。我们希望寻找帕累托最优的机器人团队路径计划，以最大化两个[相互冲突的]团队目标：预期的(I)团队奖励和(Ii)在任务中幸存下来的机器人数量。然后，人类决策者可以根据他们的价值观、奖励和机器人生存来选择平衡的试验计划。在启发式算法的指导下，采用蚁群算法来搜索机器人团队路径规划的帕累托最优集合。作为一个案例研究，我们以一家美术馆的信息收集任务为例进行说明。



## **17. From ML to LLM: Evaluating the Robustness of Phishing Webpage Detection Models against Adversarial Attacks**

从ML到LLM：评估网络钓鱼网页检测模型对抗对抗攻击的稳健性 cs.CR

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2407.20361v2) [paper-pdf](http://arxiv.org/pdf/2407.20361v2)

**Authors**: Aditya Kulkarni, Vivek Balachandran, Dinil Mon Divakaran, Tamal Das

**Abstract**: Phishing attacks attempt to deceive users into stealing sensitive information, posing a significant cybersecurity threat. Advances in machine learning (ML) and deep learning (DL) have led to the development of numerous phishing webpage detection solutions, but these models remain vulnerable to adversarial attacks. Evaluating their robustness against adversarial phishing webpages is essential. Existing tools contain datasets of pre-designed phishing webpages for a limited number of brands, and lack diversity in phishing features.   To address these challenges, we develop PhishOracle, a tool that generates adversarial phishing webpages by embedding diverse phishing features into legitimate webpages. We evaluate the robustness of two existing models, Stack model and Phishpedia, in classifying PhishOracle-generated adversarial phishing webpages. Additionally, we study a commercial large language model, Gemini Pro Vision, in the context of adversarial attacks. We conduct a user study to determine whether PhishOracle-generated adversarial phishing webpages deceive users. Our findings reveal that many PhishOracle-generated phishing webpages evade current phishing webpage detection models and deceive users, but Gemini Pro Vision is robust to the attack. We also develop the PhishOracle web app, allowing users to input a legitimate URL, select relevant phishing features and generate a corresponding phishing webpage. All resources are publicly available on GitHub.

摘要: 网络钓鱼攻击试图欺骗用户窃取敏感信息，构成重大的网络安全威胁。机器学习(ML)和深度学习(DL)的进步导致了许多钓鱼网页检测解决方案的发展，但这些模型仍然容易受到对手攻击。评估它们对敌意网络钓鱼网页的健壮性是至关重要的。现有工具包含为有限数量的品牌预先设计的钓鱼网页的数据集，并且在钓鱼功能方面缺乏多样性。为了应对这些挑战，我们开发了PhishOracle，这是一个通过在合法网页中嵌入不同的钓鱼功能来生成敌意钓鱼网页的工具。我们评估了现有的两种模型Stack模型和Phishpedia模型对PhishOracle生成的敌意钓鱼网页进行分类的稳健性。此外，我们研究了一个商业大型语言模型，Gemini Pro Vision，在对抗性攻击的背景下。我们进行了一项用户研究，以确定PhishOracle生成的敌意钓鱼网页是否欺骗了用户。我们的研究结果显示，许多PhishOracle生成的钓鱼网页逃避了当前的钓鱼网页检测模型并欺骗用户，但Gemini Pro Vision对攻击具有健壮性。我们还开发了PhishOracle Web应用程序，允许用户输入合法的URL，选择相关的网络钓鱼功能并生成相应的网络钓鱼网页。所有资源都在GitHub上公开提供。



## **18. Adversarial attacks on neural networks through canonical Riemannian foliations**

通过典型的Riemann叶联对神经网络的对抗攻击 stat.ML

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2203.00922v3) [paper-pdf](http://arxiv.org/pdf/2203.00922v3)

**Authors**: Eliot Tron, Nicolas Couellan, Stéphane Puechmorel

**Abstract**: Deep learning models are known to be vulnerable to adversarial attacks. Adversarial learning is therefore becoming a crucial task. We propose a new vision on neural network robustness using Riemannian geometry and foliation theory. The idea is illustrated by creating a new adversarial attack that takes into account the curvature of the data space. This new adversarial attack, called the two-step spectral attack is a piece-wise linear approximation of a geodesic in the data space. The data space is treated as a (degenerate) Riemannian manifold equipped with the pullback of the Fisher Information Metric (FIM) of the neural network. In most cases, this metric is only semi-definite and its kernel becomes a central object to study. A canonical foliation is derived from this kernel. The curvature of transverse leaves gives the appropriate correction to get a two-step approximation of the geodesic and hence a new efficient adversarial attack. The method is first illustrated on a 2D toy example in order to visualize the neural network foliation and the corresponding attacks. Next, we report numerical results on the MNIST and CIFAR10 datasets with the proposed technique and state of the art attacks presented in Zhao et al. (2019) (OSSA) and Croce et al. (2020) (AutoAttack). The result show that the proposed attack is more efficient at all levels of available budget for the attack (norm of the attack), confirming that the curvature of the transverse neural network FIM foliation plays an important role in the robustness of neural networks. The main objective and interest of this study is to provide a mathematical understanding of the geometrical issues at play in the data space when constructing efficient attacks on neural networks.

摘要: 众所周知，深度学习模型容易受到对手的攻击。因此，对抗性学习正成为一项至关重要的任务。利用黎曼几何和分层理论，我们对神经网络的稳健性提出了新的看法。通过创建一种新的考虑数据空间曲率的对抗性攻击来说明这一想法。这种新的敌意攻击被称为两步谱攻击，它是数据空间中测地线的分段线性近似。数据空间被视为一个(退化的)黎曼流形，带有神经网络的Fisher信息度量(FIM)的回撤。在大多数情况下，这个度量只是半定的，它的核心成为研究的中心对象。一个典型的叶理是从这个核派生出来的。横叶的曲率给出了适当的修正，得到了测地线的两步近似，从而得到了一种新的有效的对抗性攻击。为了可视化神经网络的分层和相应的攻击，首先以2D玩具为例说明了该方法。接下来，我们报告了在MNIST和CIFAR10数据集上的数值结果，以及赵等人提出的技术和最新攻击。(2019)(Ossa)和Croce等人。(2020)(AutoAttack)。结果表明，该攻击在攻击的所有可用预算级别(攻击范数)上都是有效的，证实了横向神经网络FIM分层的曲率对神经网络的健壮性起着重要作用。这项研究的主要目的和兴趣是在构造对神经网络的有效攻击时，提供对数据空间中所起作用的几何问题的数学理解。



## **19. Secure Control Systems for Autonomous Quadrotors against Cyber-Attacks**

针对网络攻击的自主四螺旋桨安全控制系统 cs.RO

The paper is based on an undergraduate thesis and is not intended for  publication in a journal

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2409.11897v1) [paper-pdf](http://arxiv.org/pdf/2409.11897v1)

**Authors**: Samuel Belkadi

**Abstract**: The problem of safety for robotic systems has been extensively studied. However, little attention has been given to security issues for three-dimensional systems, such as quadrotors. Malicious adversaries can compromise robot sensors and communication networks, causing incidents, achieving illegal objectives, or even injuring people. This study first designs an intelligent control system for autonomous quadrotors. Then, it investigates the problems of optimal false data injection attack scheduling and countermeasure design for unmanned aerial vehicles. Using a state-of-the-art deep learning-based approach, an optimal false data injection attack scheme is proposed to deteriorate a quadrotor's tracking performance with limited attack energy. Subsequently, an optimal tracking control strategy is learned to mitigate attacks and recover the quadrotor's tracking performance. We base our work on Agilicious, a state-of-the-art quadrotor recently deployed for autonomous settings. This paper is the first in the United Kingdom to deploy this quadrotor and implement reinforcement learning on its platform. Therefore, to promote easy reproducibility with minimal engineering overhead, we further provide (1) a comprehensive breakdown of this quadrotor, including software stacks and hardware alternatives; (2) a detailed reinforcement-learning framework to train autonomous controllers on Agilicious agents; and (3) a new open-source environment that builds upon PyFlyt for future reinforcement learning research on Agilicious platforms. Both simulated and real-world experiments are conducted to show the effectiveness of the proposed frameworks in section 5.2.

摘要: 机器人系统的安全问题已经得到了广泛的研究。然而，三维系统的安全问题却很少受到关注，比如四旋翼飞行器。恶意攻击者可以破坏机器人传感器和通信网络，引发事件，实现非法目标，甚至伤害人。本研究首先设计了自主四旋翼飞行器的智能控制系统。然后，研究了无人机最优虚假数据注入攻击调度和对抗设计问题。利用最新的基于深度学习的方法，提出了一种在攻击能量有限的情况下恶化四旋翼跟踪性能的最优虚假数据注入攻击方案。随后，学习了一种最优跟踪控制策略，以减轻攻击并恢复四旋翼的跟踪性能。我们的工作基于Agilous，这是一款最先进的四旋翼飞机，最近部署在自主环境中。本文在英国首次部署了这种四旋翼，并在其平台上实现了强化学习。因此，为了以最小的工程开销促进易重复性，我们进一步提供(1)这个四旋翼的全面细分，包括软件堆栈和硬件替代；(2)详细的强化学习框架，以培训Agilous代理上的自主控制器；以及(3)一个新的开源环境，该环境构建在PyFlyt基础上，用于未来在Agilous平台上的强化学习研究。仿真和真实世界的实验都表明了5.2节中提出的框架的有效性。



## **20. NPAT Null-Space Projected Adversarial Training Towards Zero Deterioration**

NMat零空间投影对抗训练实现零恶化 cs.LG

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2409.11754v1) [paper-pdf](http://arxiv.org/pdf/2409.11754v1)

**Authors**: Hanyi Hu, Qiao Han, Kui Chen, Yao Yang

**Abstract**: To mitigate the susceptibility of neural networks to adversarial attacks, adversarial training has emerged as a prevalent and effective defense strategy. Intrinsically, this countermeasure incurs a trade-off, as it sacrifices the model's accuracy in processing normal samples. To reconcile the trade-off, we pioneer the incorporation of null-space projection into adversarial training and propose two innovative Null-space Projection based Adversarial Training(NPAT) algorithms tackling sample generation and gradient optimization, named Null-space Projected Data Augmentation (NPDA) and Null-space Projected Gradient Descent (NPGD), to search for an overarching optimal solutions, which enhance robustness with almost zero deterioration in generalization performance. Adversarial samples and perturbations are constrained within the null-space of the decision boundary utilizing a closed-form null-space projector, effectively mitigating threat of attack stemming from unreliable features. Subsequently, we conducted experiments on the CIFAR10 and SVHN datasets and reveal that our methodology can seamlessly combine with adversarial training methods and obtain comparable robustness while keeping generalization close to a high-accuracy model.

摘要: 为了减轻神经网络对对抗性攻击的敏感性，对抗性训练已经成为一种普遍而有效的防御策略。本质上，这种对策需要权衡取舍，因为它牺牲了模型在处理正常样本时的准确性。为了协调两者之间的权衡，我们将零空间投影引入对抗性训练中，提出了两种新的基于零空间投影的对抗性训练(NPAT)算法，即零空间投影数据增强(NPDA)和零空间投影梯度下降(NPGD)算法，以寻求在泛化性能几乎为零恶化的情况下增强鲁棒性的零空间投影梯度下降(NPGD)算法。利用封闭的零空间投影仪将对抗性样本和扰动限制在决策边界的零空间内，有效地减轻了来自不可靠特征的攻击威胁。随后，我们在CIFAR10和SVHN数据集上进行了实验，结果表明，我们的方法可以与对抗性训练方法无缝结合，在保持泛化接近高精度模型的情况下获得相当的鲁棒性。



## **21. Hard-Label Cryptanalytic Extraction of Neural Network Models**

神经网络模型的硬标签密码分析提取 cs.CR

Accepted by Asiacrypt 2024

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2409.11646v1) [paper-pdf](http://arxiv.org/pdf/2409.11646v1)

**Authors**: Yi Chen, Xiaoyang Dong, Jian Guo, Yantian Shen, Anyu Wang, Xiaoyun Wang

**Abstract**: The machine learning problem of extracting neural network parameters has been proposed for nearly three decades. Functionally equivalent extraction is a crucial goal for research on this problem. When the adversary has access to the raw output of neural networks, various attacks, including those presented at CRYPTO 2020 and EUROCRYPT 2024, have successfully achieved this goal. However, this goal is not achieved when neural networks operate under a hard-label setting where the raw output is inaccessible.   In this paper, we propose the first attack that theoretically achieves functionally equivalent extraction under the hard-label setting, which applies to ReLU neural networks. The effectiveness of our attack is validated through practical experiments on a wide range of ReLU neural networks, including neural networks trained on two real benchmarking datasets (MNIST, CIFAR10) widely used in computer vision. For a neural network consisting of $10^5$ parameters, our attack only requires several hours on a single core.

摘要: 提取神经网络参数的机器学习问题已经提出了近三十年。功能等价抽取是这一问题研究的一个重要目标。当对手可以访问神经网络的原始输出时，各种攻击，包括在加密2020和欧洲加密2024上提出的攻击，都成功地实现了这一目标。然而，当神经网络在原始输出不可访问的硬标签设置下运行时，这一目标无法实现。在本文中，我们提出了第一个攻击，该攻击在理论上实现了硬标签设置下的函数等价提取，适用于RELU神经网络。通过在广泛的RELU神经网络上的实际实验，包括在两个在计算机视觉中广泛使用的真实基准数据集(MNIST，CIFAR10)上训练的神经网络，验证了该攻击的有效性。对于一个由$10^5$参数组成的神经网络，我们的攻击只需要在一个核上几个小时。



## **22. Image Hijacks: Adversarial Images can Control Generative Models at Runtime**

图像劫持：对抗图像可以随时控制生成模型 cs.LG

Project page at https://image-hijacks.github.io

**SubmitDate**: 2024-09-17    [abs](http://arxiv.org/abs/2309.00236v4) [paper-pdf](http://arxiv.org/pdf/2309.00236v4)

**Authors**: Luke Bailey, Euan Ong, Stuart Russell, Scott Emmons

**Abstract**: Are foundation models secure against malicious actors? In this work, we focus on the image input to a vision-language model (VLM). We discover image hijacks, adversarial images that control the behaviour of VLMs at inference time, and introduce the general Behaviour Matching algorithm for training image hijacks. From this, we derive the Prompt Matching method, allowing us to train hijacks matching the behaviour of an arbitrary user-defined text prompt (e.g. 'the Eiffel Tower is now located in Rome') using a generic, off-the-shelf dataset unrelated to our choice of prompt. We use Behaviour Matching to craft hijacks for four types of attack, forcing VLMs to generate outputs of the adversary's choice, leak information from their context window, override their safety training, and believe false statements. We study these attacks against LLaVA, a state-of-the-art VLM based on CLIP and LLaMA-2, and find that all attack types achieve a success rate of over 80%. Moreover, our attacks are automated and require only small image perturbations.

摘要: 基础模型针对恶意行为者是否安全？在这项工作中，我们关注的是图像输入到视觉语言模型(VLM)。我们发现了图像劫持，即在推理时控制VLM行为的敌意图像，并介绍了用于训练图像劫持的通用行为匹配算法。从这里，我们得到了提示匹配方法，允许我们使用与我们选择的提示无关的通用现成数据集来训练与任意用户定义的文本提示(例如‘埃菲尔铁塔现在位于罗马’)的行为匹配的劫持者。我们使用行为匹配来为四种类型的攻击制作劫持，迫使VLM生成对手选择的输出，从他们的上下文窗口泄露信息，覆盖他们的安全培训，并相信虚假陈述。我们对基于CLIP和LLAMA-2的最新VLM LLaVA进行了研究，发现所有类型的攻击都达到了80%以上的成功率。此外，我们的攻击是自动化的，只需要很小的图像扰动。



## **23. Golden Ratio Search: A Low-Power Adversarial Attack for Deep Learning based Modulation Classification**

黄金比率搜索：针对基于深度学习的调制分类的低功耗对抗攻击 cs.CR

5 pages, 1 figure, 3 tables

**SubmitDate**: 2024-09-17    [abs](http://arxiv.org/abs/2409.11454v1) [paper-pdf](http://arxiv.org/pdf/2409.11454v1)

**Authors**: Deepsayan Sadhukhan, Nitin Priyadarshini Shankar, Sheetal Kalyani

**Abstract**: We propose a minimal power white box adversarial attack for Deep Learning based Automatic Modulation Classification (AMC). The proposed attack uses the Golden Ratio Search (GRS) method to find powerful attacks with minimal power. We evaluate the efficacy of the proposed method by comparing it with existing adversarial attack approaches. Additionally, we test the robustness of the proposed attack against various state-of-the-art architectures, including defense mechanisms such as adversarial training, binarization, and ensemble methods. Experimental results demonstrate that the proposed attack is powerful, requires minimal power, and can be generated in less time, significantly challenging the resilience of current AMC methods.

摘要: 我们针对基于深度学习的自动调制分类（AMC）提出了一种最小功率白盒对抗攻击。拟议的攻击使用黄金比例搜索（GRS）方法来以最小的功率找到强大的攻击。我们通过与现有的对抗攻击方法进行比较来评估所提出方法的有效性。此外，我们还测试了针对各种最先进架构的拟议攻击的稳健性，包括对抗性训练、二进制化和集成方法等防御机制。实验结果表明，提出的攻击强大，所需功率最小，并且可以在更短的时间内产生，极大地挑战了当前AMC方法的弹性。



## **24. EIA: Environmental Injection Attack on Generalist Web Agents for Privacy Leakage**

EIA：针对多面手网络代理隐私泄露的环境注入攻击 cs.CR

24 pages

**SubmitDate**: 2024-09-17    [abs](http://arxiv.org/abs/2409.11295v1) [paper-pdf](http://arxiv.org/pdf/2409.11295v1)

**Authors**: Zeyi Liao, Lingbo Mo, Chejian Xu, Mintong Kang, Jiawei Zhang, Chaowei Xiao, Yuan Tian, Bo Li, Huan Sun

**Abstract**: Generalist web agents have evolved rapidly and demonstrated remarkable potential. However, there are unprecedented safety risks associated with these them, which are nearly unexplored so far. In this work, we aim to narrow this gap by conducting the first study on the privacy risks of generalist web agents in adversarial environments. First, we present a threat model that discusses the adversarial targets, constraints, and attack scenarios. Particularly, we consider two types of adversarial targets: stealing users' specific personally identifiable information (PII) or stealing the entire user request. To achieve these objectives, we propose a novel attack method, termed Environmental Injection Attack (EIA). This attack injects malicious content designed to adapt well to different environments where the agents operate, causing them to perform unintended actions. This work instantiates EIA specifically for the privacy scenario. It inserts malicious web elements alongside persuasive instructions that mislead web agents into leaking private information, and can further leverage CSS and JavaScript features to remain stealthy. We collect 177 actions steps that involve diverse PII categories on realistic websites from the Mind2Web dataset, and conduct extensive experiments using one of the most capable generalist web agent frameworks to date, SeeAct. The results demonstrate that EIA achieves up to 70% ASR in stealing users' specific PII. Stealing full user requests is more challenging, but a relaxed version of EIA can still achieve 16% ASR. Despite these concerning results, it is important to note that the attack can still be detectable through careful human inspection, highlighting a trade-off between high autonomy and security. This leads to our detailed discussion on the efficacy of EIA under different levels of human supervision as well as implications on defenses for generalist web agents.

摘要: 多面手网络代理发展迅速，并显示出非凡的潜力。然而，它们存在着前所未有的安全风险，到目前为止几乎没有人探索过。在这项工作中，我们旨在通过对对抗环境中通才网络代理的隐私风险进行第一次研究来缩小这一差距。首先，我们提出了一个威胁模型，该模型讨论了对抗性目标、约束和攻击场景。具体地说，我们考虑了两种类型的对抗目标：窃取用户特定的个人身份信息(PII)或窃取整个用户请求。为了实现这些目标，我们提出了一种新的攻击方法，称为环境注入攻击(EIA)。此攻击注入恶意内容，旨在很好地适应代理程序运行的不同环境，导致它们执行意外操作。这项工作专门为隐私场景实例化了EIA。它将恶意的网络元素与具有说服力的指令一起插入，误导网络代理泄露私人信息，并可以进一步利用CSS和JavaScript功能来保持隐蔽性。我们从Mind2Web数据集中收集了177个动作步骤，涉及现实网站上的不同PII类别，并使用迄今最有能力的通用Web代理框架之一SeeAct进行了广泛的实验。结果表明，在窃取用户特定PII时，EIA的ASR高达70%。窃取完整的用户请求更具挑战性，但宽松版本的EIA仍可实现16%的ASR。尽管有这些令人担忧的结果，但必须指出的是，通过仔细的人工检查仍然可以检测到攻击，这突显了高度自治和安全之间的权衡。这导致了我们详细讨论了在不同级别的人类监督下的EIA的有效性，以及对多面手网络代理的防御的影响。



## **25. Backdoor Attacks in Peer-to-Peer Federated Learning**

点对点联邦学习中的后门攻击 cs.LG

**SubmitDate**: 2024-09-17    [abs](http://arxiv.org/abs/2301.09732v4) [paper-pdf](http://arxiv.org/pdf/2301.09732v4)

**Authors**: Georgios Syros, Gokberk Yar, Simona Boboila, Cristina Nita-Rotaru, Alina Oprea

**Abstract**: Most machine learning applications rely on centralized learning processes, opening up the risk of exposure of their training datasets. While federated learning (FL) mitigates to some extent these privacy risks, it relies on a trusted aggregation server for training a shared global model. Recently, new distributed learning architectures based on Peer-to-Peer Federated Learning (P2PFL) offer advantages in terms of both privacy and reliability. Still, their resilience to poisoning attacks during training has not been investigated. In this paper, we propose new backdoor attacks for P2PFL that leverage structural graph properties to select the malicious nodes, and achieve high attack success, while remaining stealthy. We evaluate our attacks under various realistic conditions, including multiple graph topologies, limited adversarial visibility of the network, and clients with non-IID data. Finally, we show the limitations of existing defenses adapted from FL and design a new defense that successfully mitigates the backdoor attacks, without an impact on model accuracy.

摘要: 大多数机器学习应用程序依赖于集中的学习过程，从而打开了暴露其训练数据集的风险。虽然联合学习(FL)在一定程度上缓解了这些隐私风险，但它依赖于可信的聚合服务器来训练共享的全局模型。近年来，基于对等联合学习(P2P-to-Peer Federated Learning，简称P2PFL)的新型分布式学习体系结构在保密性和可靠性方面都具有优势。尽管如此，他们在训练期间对中毒攻击的抵抗力还没有得到调查。本文提出了一种新的针对P2P PFL的后门攻击，利用结构图的性质来选择恶意节点，在保持隐蔽性的同时获得较高的攻击成功率。我们在各种现实条件下评估我们的攻击，包括多个图拓扑、有限的网络敌意可见性以及具有非IID数据的客户端。最后，我们指出了现有防御方案的局限性，并设计了一种新的防御方案，在不影响模型精度的情况下，成功地缓解了后门攻击。



## **26. A Survey of Machine Unlearning**

机器学习研究 cs.LG

extend the survey with more recent published work and add more  discussions

**SubmitDate**: 2024-09-17    [abs](http://arxiv.org/abs/2209.02299v6) [paper-pdf](http://arxiv.org/pdf/2209.02299v6)

**Authors**: Thanh Tam Nguyen, Thanh Trung Huynh, Zhao Ren, Phi Le Nguyen, Alan Wee-Chung Liew, Hongzhi Yin, Quoc Viet Hung Nguyen

**Abstract**: Today, computer systems hold large amounts of personal data. Yet while such an abundance of data allows breakthroughs in artificial intelligence, and especially machine learning (ML), its existence can be a threat to user privacy, and it can weaken the bonds of trust between humans and AI. Recent regulations now require that, on request, private information about a user must be removed from both computer systems and from ML models, i.e. ``the right to be forgotten''). While removing data from back-end databases should be straightforward, it is not sufficient in the AI context as ML models often `remember' the old data. Contemporary adversarial attacks on trained models have proven that we can learn whether an instance or an attribute belonged to the training data. This phenomenon calls for a new paradigm, namely machine unlearning, to make ML models forget about particular data. It turns out that recent works on machine unlearning have not been able to completely solve the problem due to the lack of common frameworks and resources. Therefore, this paper aspires to present a comprehensive examination of machine unlearning's concepts, scenarios, methods, and applications. Specifically, as a category collection of cutting-edge studies, the intention behind this article is to serve as a comprehensive resource for researchers and practitioners seeking an introduction to machine unlearning and its formulations, design criteria, removal requests, algorithms, and applications. In addition, we aim to highlight the key findings, current trends, and new research areas that have not yet featured the use of machine unlearning but could benefit greatly from it. We hope this survey serves as a valuable resource for ML researchers and those seeking to innovate privacy technologies. Our resources are publicly available at https://github.com/tamlhp/awesome-machine-unlearning.

摘要: 今天，计算机系统保存着大量的个人数据。然而，尽管如此丰富的数据使人工智能，特别是机器学习(ML)取得了突破，但它的存在可能会对用户隐私构成威胁，并可能削弱人类与人工智能之间的信任纽带。最近的法规现在要求，根据请求，必须从计算机系统和ML模型中删除关于用户的私人信息，即“被遗忘权”)。虽然从后端数据库中删除数据应该是直接的，但在人工智能上下文中这是不够的，因为ML模型经常‘记住’旧数据。当代针对训练模型的对抗性攻击已经证明，我们可以学习到一个实例或一个属性是否属于训练数据。这种现象呼唤一种新的范式，即机器遗忘，以使ML模型忘记特定的数据。事实证明，由于缺乏通用的框架和资源，最近关于机器遗忘的研究并不能完全解决这个问题。因此，本文致力于对机器遗忘的概念、场景、方法和应用进行全面的考察。具体地说，作为尖端研究的类别集合，本文背后的目的是为寻求介绍机器遗忘及其公式、设计标准、移除请求、算法和应用的研究人员和从业者提供全面的资源。此外，我们的目标是强调关键的发现、当前的趋势和新的研究领域，这些领域还没有使用机器遗忘，但可以从中受益匪浅。我们希望这项调查对ML研究人员和那些寻求创新隐私技术的人来说是一个有价值的资源。我们的资源可在https://github.com/tamlhp/awesome-machine-unlearning.上公开获取



## **27. Remote Keylogging Attacks in Multi-user VR Applications**

多用户VR应用程序中的远程键盘记录攻击 cs.CR

Accepted for Usenix 2024

**SubmitDate**: 2024-09-17    [abs](http://arxiv.org/abs/2405.14036v2) [paper-pdf](http://arxiv.org/pdf/2405.14036v2)

**Authors**: Zihao Su, Kunlin Cai, Reuben Beeler, Lukas Dresel, Allan Garcia, Ilya Grishchenko, Yuan Tian, Christopher Kruegel, Giovanni Vigna

**Abstract**: As Virtual Reality (VR) applications grow in popularity, they have bridged distances and brought users closer together. However, with this growth, there have been increasing concerns about security and privacy, especially related to the motion data used to create immersive experiences. In this study, we highlight a significant security threat in multi-user VR applications, which are applications that allow multiple users to interact with each other in the same virtual space. Specifically, we propose a remote attack that utilizes the avatar rendering information collected from an adversary's game clients to extract user-typed secrets like credit card information, passwords, or private conversations. We do this by (1) extracting motion data from network packets, and (2) mapping motion data to keystroke entries. We conducted a user study to verify the attack's effectiveness, in which our attack successfully inferred 97.62% of the keystrokes. Besides, we performed an additional experiment to underline that our attack is practical, confirming its effectiveness even when (1) there are multiple users in a room, and (2) the attacker cannot see the victims. Moreover, we replicated our proposed attack on four applications to demonstrate the generalizability of the attack. Lastly, we proposed a defense against the attack, which has been implemented by major players in the VR industry. These results underscore the severity of the vulnerability and its potential impact on millions of VR social platform users.

摘要: 随着虚拟现实(VR)应用越来越受欢迎，它们弥合了距离，拉近了用户之间的距离。然而，随着这种增长，人们对安全和隐私的担忧也越来越多，特别是与用于创建身临其境体验的运动数据有关。在这项研究中，我们强调了多用户VR应用中的一个重大安全威胁，即允许多个用户在同一虚拟空间中相互交互的应用。具体地说，我们提出了一种远程攻击，它利用从对手的游戏客户端收集的化身渲染信息来提取用户键入的秘密，如信用卡信息、密码或私人对话。我们通过(1)从网络分组中提取运动数据，以及(2)将运动数据映射到击键条目来实现这一点。我们进行了用户研究来验证攻击的有效性，其中我们的攻击成功推断了97.62%的击键。此外，我们还执行了一个额外的实验，以强调我们的攻击是实用的，即使在(1)一个房间有多个用户，以及(2)攻击者看不到受害者的情况下，也证实了它的有效性。此外，我们在四个应用程序上复制了我们提出的攻击，以证明该攻击的泛化能力。最后，我们提出了针对攻击的防御方案，并已被VR行业的主要参与者实施。这些结果突显了该漏洞的严重性及其对数百万VR社交平台用户的潜在影响。



## **28. An Anti-disguise Authentication System Using the First Impression of Avatar in Metaverse**

利用虚拟宇宙阿凡达第一印象的反伪装认证系统 cs.CR

19 pages, 16 figures

**SubmitDate**: 2024-09-17    [abs](http://arxiv.org/abs/2409.10850v1) [paper-pdf](http://arxiv.org/pdf/2409.10850v1)

**Authors**: Zhenyong Zhang, Kedi Yang, Youliang Tian, Jianfeng Ma

**Abstract**: Metaverse is a vast virtual world parallel to the physical world, where the user acts as an avatar to enjoy various services that break through the temporal and spatial limitations of the physical world. Metaverse allows users to create arbitrary digital appearances as their own avatars by which an adversary may disguise his/her avatar to fraud others. In this paper, we propose an anti-disguise authentication method that draws on the idea of the first impression from the physical world to recognize an old friend. Specifically, the first meeting scenario in the metaverse is stored and recalled to help the authentication between avatars. To prevent the adversary from replacing and forging the first impression, we construct a chameleon-based signcryption mechanism and design a ciphertext authentication protocol to ensure the public verifiability of encrypted identities. The security analysis shows that the proposed signcryption mechanism meets not only the security requirement but also the public verifiability. Besides, the ciphertext authentication protocol has the capability of defending against the replacing and forging attacks on the first impression. Extensive experiments show that the proposed avatar authentication system is able to achieve anti-disguise authentication at a low storage consumption on the blockchain.

摘要: Metverse是一个与物理世界平行的广阔虚拟世界，用户在其中扮演化身，享受各种突破物理世界时空限制的服务。Metverse允许用户创建任意的数字外观作为他们自己的化身，对手可以利用这些化身来伪装他/她的化身以欺骗他人。在本文中，我们提出了一种反伪装认证方法，该方法借鉴了物理世界第一印象的思想来识别老朋友。具体地说，存储和调用虚拟世界中的第一个会议场景，以帮助在化身之间进行身份验证。为了防止攻击者替换和伪造第一印象，我们构造了一个基于变色龙的签密机制，并设计了一个密文认证协议来确保加密身份的公开可验证性。安全性分析表明，该签密机制不仅满足安全性要求，而且具有公开可验证性。此外，密文认证协议还具有抵抗替换攻击和伪造第一印象攻击的能力。大量实验表明，所提出的头像认证系统能够在区块链上以较低的存储消耗实现反伪装认证。



## **29. Weak Superimposed Codes of Improved Asymptotic Rate and Their Randomized Construction**

改进渐进率的弱叠加码及其随机构造 cs.IT

6 pages, accepted for presentation at the 2022 IEEE International  Symposium on Information Theory (ISIT)

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2409.10511v1) [paper-pdf](http://arxiv.org/pdf/2409.10511v1)

**Authors**: Yu Tsunoda, Yuichiro Fujiwara

**Abstract**: Weak superimposed codes are combinatorial structures related closely to generalized cover-free families, superimposed codes, and disjunct matrices in that they are only required to satisfy similar but less stringent conditions. This class of codes may also be seen as a stricter variant of what are known as locally thin families in combinatorics. Originally, weak superimposed codes were introduced in the context of multimedia content protection against illegal distribution of copies under the assumption that a coalition of malicious users may employ the averaging attack with adversarial noise. As in many other kinds of codes in information theory, it is of interest and importance in the study of weak superimposed codes to find the highest achievable rate in the asymptotic regime and give an efficient construction that produces an infinite sequence of codes that achieve it. Here, we prove a tighter lower bound than the sharpest known one on the rate of optimal weak superimposed codes and give a polynomial-time randomized construction algorithm for codes that asymptotically attain our improved bound with high probability. Our probabilistic approach is versatile and applicable to many other related codes and arrays.

摘要: 弱叠加码是一种与广义无覆盖族、叠加码和析取矩阵密切相关的组合结构，它们只需要满足相似但不那么严格的条件。这类代码也可以被视为组合数学中所知的局部瘦族的更严格的变体。最初，在防止非法分发副本的多媒体内容保护的环境中引入了弱叠加代码，假设恶意用户的联盟可能采用带有对抗性噪声的平均攻击。与信息论中的许多其他类型的码一样，在弱重叠码的研究中，寻找渐近状态下的最高可达速率，并给出一个有效的构造，从而产生实现它的无限序列，是一个有趣而重要的问题。在这里，我们证明了最优弱叠加码码率的一个比已知的最强下界更紧的下界，并给出了一个多项式时间的随机构造算法，它以很高的概率渐近地达到我们的改进界。我们的概率方法是通用的，并适用于许多其他相关的代码和数组。



## **30. Assessing biomedical knowledge robustness in large language models by query-efficient sampling attacks**

通过查询高效抽样攻击评估大型语言模型中生物医学知识的稳健性 cs.CL

28 pages incl. appendix, updated version

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2402.10527v2) [paper-pdf](http://arxiv.org/pdf/2402.10527v2)

**Authors**: R. Patrick Xian, Alex J. Lee, Satvik Lolla, Vincent Wang, Qiming Cui, Russell Ro, Reza Abbasi-Asl

**Abstract**: The increasing depth of parametric domain knowledge in large language models (LLMs) is fueling their rapid deployment in real-world applications. Understanding model vulnerabilities in high-stakes and knowledge-intensive tasks is essential for quantifying the trustworthiness of model predictions and regulating their use. The recent discovery of named entities as adversarial examples (i.e. adversarial entities) in natural language processing tasks raises questions about their potential impact on the knowledge robustness of pre-trained and finetuned LLMs in high-stakes and specialized domains. We examined the use of type-consistent entity substitution as a template for collecting adversarial entities for billion-parameter LLMs with biomedical knowledge. To this end, we developed an embedding-space attack based on powerscaled distance-weighted sampling to assess the robustness of their biomedical knowledge with a low query budget and controllable coverage. Our method has favorable query efficiency and scaling over alternative approaches based on random sampling and blackbox gradient-guided search, which we demonstrated for adversarial distractor generation in biomedical question answering. Subsequent failure mode analysis uncovered two regimes of adversarial entities on the attack surface with distinct characteristics and we showed that entity substitution attacks can manipulate token-wise Shapley value explanations, which become deceptive in this setting. Our approach complements standard evaluations for high-capacity models and the results highlight the brittleness of domain knowledge in LLMs.

摘要: 大型语言模型(LLM)中参数领域知识的不断深入推动了它们在现实世界应用程序中的快速部署。了解高风险和知识密集型任务中的模型脆弱性对于量化模型预测的可信度和规范其使用至关重要。最近在自然语言处理任务中发现了命名实体作为对抗性实例(即对抗性实体)，这引发了人们对高风险和专门领域中预先训练和精细调整的LLM知识稳健性的潜在影响的问题。我们研究了使用类型一致的实体替换作为收集具有生物医学知识的10亿参数LLM的对抗性实体的模板。为此，我们提出了一种基于加权距离加权抽样的嵌入空间攻击方法，以较低的查询预算和可控的覆盖率来评估他们的生物医学知识的稳健性。与基于随机抽样和黑盒梯度引导搜索的方法相比，我们的方法具有良好的查询效率和伸缩性，并在生物医学问答中的对抗性干扰项生成中得到了验证。随后的失效模式分析揭示了攻击面上具有不同特征的两种对抗实体的机制，我们表明实体替换攻击可以操纵令人信服的Shapley值解释，在这种情况下，这种解释变得具有欺骗性。我们的方法补充了对大容量模型的标准评估，结果突出了领域知识在LLMS中的脆性。



## **31. Deep Reinforcement Learning for Autonomous Cyber Operations: A Survey**

自主网络运营的深度强化学习：调查 cs.LG

89 pages, 14 figures, 4 tables

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2310.07745v2) [paper-pdf](http://arxiv.org/pdf/2310.07745v2)

**Authors**: Gregory Palmer, Chris Parry, Daniel J. B. Harrold, Chris Willis

**Abstract**: The rapid increase in the number of cyber-attacks in recent years raises the need for principled methods for defending networks against malicious actors. Deep reinforcement learning (DRL) has emerged as a promising approach for mitigating these attacks. However, while DRL has shown much potential for cyber defence, numerous challenges must be overcome before DRL can be applied to autonomous cyber operations (ACO) at scale. Principled methods are required for environments that confront learners with very high-dimensional state spaces, large multi-discrete action spaces, and adversarial learning. Recent works have reported success in solving these problems individually. There have also been impressive engineering efforts towards solving all three for real-time strategy games. However, applying DRL to the full ACO problem remains an open challenge. Here, we survey the relevant DRL literature and conceptualize an idealised ACO-DRL agent. We provide: i.) A summary of the domain properties that define the ACO problem; ii.) A comprehensive comparison of current ACO environments used for benchmarking DRL approaches; iii.) An overview of state-of-the-art approaches for scaling DRL to domains that confront learners with the curse of dimensionality, and; iv.) A survey and critique of current methods for limiting the exploitability of agents within adversarial settings from the perspective of ACO. We conclude with open research questions that we hope will motivate future directions for researchers and practitioners working on ACO.

摘要: 近年来，网络攻击数量的迅速增加增加了对保护网络免受恶意行为侵害的原则性方法的需求。深度强化学习(DRL)已成为缓解这些攻击的一种很有前途的方法。然而，尽管DRL在网络防御方面显示出了很大的潜力，但在DRL能够大规模应用于自主网络作战(ACO)之前，必须克服许多挑战。对于学习者面对高维状态空间、大的多离散动作空间和对抗性学习的环境，需要有原则性的方法。最近的研究报告成功地单独解决了这些问题。也有令人印象深刻的工程努力，以解决所有这三个实时战略游戏。然而，将DRL应用于整个蚁群优化问题仍然是一个开放的挑战。在这里，我们回顾了相关的DRL文献，并概念化了一个理想的ACO-DRL试剂。我们提供：i.)定义ACO问题的域属性摘要；ii.)对当前用于基准DRL方法的ACO环境进行了全面比较；三.)概述将DRL扩展到学习者面临维度诅咒的领域的最新方法，以及；i.)从蚁群算法的角度对当前在对抗性环境中限制代理的可利用性的方法进行了调查和评论。我们以开放的研究问题结束，我们希望这些问题将激励从事ACO工作的研究人员和从业者未来的方向。



## **32. Machine Against the RAG: Jamming Retrieval-Augmented Generation with Blocker Documents**

机器对抗RAG：用阻止器文档干扰检索增强生成 cs.CR

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2406.05870v2) [paper-pdf](http://arxiv.org/pdf/2406.05870v2)

**Authors**: Avital Shafran, Roei Schuster, Vitaly Shmatikov

**Abstract**: Retrieval-augmented generation (RAG) systems respond to queries by retrieving relevant documents from a knowledge database, then generating an answer by applying an LLM to the retrieved documents. We demonstrate that RAG systems that operate on databases with untrusted content are vulnerable to a new class of denial-of-service attacks we call jamming. An adversary can add a single ``blocker'' document to the database that will be retrieved in response to a specific query and result in the RAG system not answering this query - ostensibly because it lacks the information or because the answer is unsafe.   We describe and measure the efficacy of several methods for generating blocker documents, including a new method based on black-box optimization. This method (1) does not rely on instruction injection, (2) does not require the adversary to know the embedding or LLM used by the target RAG system, and (3) does not use an auxiliary LLM to generate blocker documents.   We evaluate jamming attacks on several LLMs and embeddings and demonstrate that the existing safety metrics for LLMs do not capture their vulnerability to jamming. We then discuss defenses against blocker documents.

摘要: 检索-增强生成(RAG)系统通过从知识数据库中检索相关文档，然后通过将LLM应用于所检索的文档来生成答案来响应查询。我们证明，在含有不可信内容的数据库上运行的RAG系统容易受到一种新的拒绝服务攻击，我们称之为干扰。敌手可以向数据库添加一个“拦截器”文档，该文档将响应于特定查询而被检索，并导致RAG系统不回答该查询--表面上是因为它缺乏信息或因为答案不安全。我们描述并测试了几种生成拦截器文档的方法的有效性，其中包括一种基于黑盒优化的新方法。该方法(1)不依赖于指令注入，(2)不要求对手知道目标RAG系统使用的嵌入或LLM，以及(3)不使用辅助LLM来生成拦截器文档。我们评估了几个LLM和嵌入上的干扰攻击，并证明了现有的LLM安全度量没有捕捉到它们对干扰的脆弱性。然后我们讨论针对拦截器文档的防御。



## **33. Towards Evaluating the Robustness of Visual State Space Models**

评估视觉状态空间模型的稳健性 cs.CV

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2406.09407v2) [paper-pdf](http://arxiv.org/pdf/2406.09407v2)

**Authors**: Hashmat Shadab Malik, Fahad Shamshad, Muzammal Naseer, Karthik Nandakumar, Fahad Shahbaz Khan, Salman Khan

**Abstract**: Vision State Space Models (VSSMs), a novel architecture that combines the strengths of recurrent neural networks and latent variable models, have demonstrated remarkable performance in visual perception tasks by efficiently capturing long-range dependencies and modeling complex visual dynamics. However, their robustness under natural and adversarial perturbations remains a critical concern. In this work, we present a comprehensive evaluation of VSSMs' robustness under various perturbation scenarios, including occlusions, image structure, common corruptions, and adversarial attacks, and compare their performance to well-established architectures such as transformers and Convolutional Neural Networks. Furthermore, we investigate the resilience of VSSMs to object-background compositional changes on sophisticated benchmarks designed to test model performance in complex visual scenes. We also assess their robustness on object detection and segmentation tasks using corrupted datasets that mimic real-world scenarios. To gain a deeper understanding of VSSMs' adversarial robustness, we conduct a frequency-based analysis of adversarial attacks, evaluating their performance against low-frequency and high-frequency perturbations. Our findings highlight the strengths and limitations of VSSMs in handling complex visual corruptions, offering valuable insights for future research. Our code and models will be available at https://github.com/HashmatShadab/MambaRobustness.

摘要: 视觉状态空间模型(VSSMS)是一种结合了递归神经网络和潜变量模型优点的新型结构，通过有效地捕捉长距离依赖关系和建模复杂的视觉动力学，在视觉感知任务中表现出了显著的性能。然而，它们在自然和对抗性扰动下的稳健性仍然是一个严重的问题。在这项工作中，我们对VSSM在各种扰动场景下的健壮性进行了全面的评估，包括遮挡、图像结构、常见的腐败和敌对攻击，并将它们的性能与成熟的架构，如变压器和卷积神经网络进行了比较。此外，我们在复杂的基准测试中考察了VSSM对对象-背景成分变化的弹性，该基准旨在测试复杂视觉场景中的模型性能。我们还使用模拟真实世界场景的损坏数据集评估了它们在对象检测和分割任务中的稳健性。为了更深入地了解VSSM的对抗稳健性，我们对对抗攻击进行了基于频率的分析，评估了它们对低频和高频扰动的性能。我们的发现突出了VSSM在处理复杂视觉腐败方面的优势和局限性，为未来的研究提供了有价值的见解。我们的代码和模型将在https://github.com/HashmatShadab/MambaRobustness.上提供



## **34. Multi-agent Attacks for Black-box Social Recommendations**

针对黑匣子社交推荐的多代理攻击 cs.SI

Accepted by ACM TOIS

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2311.07127v4) [paper-pdf](http://arxiv.org/pdf/2311.07127v4)

**Authors**: Shijie Wang, Wenqi Fan, Xiao-yong Wei, Xiaowei Mei, Shanru Lin, Qing Li

**Abstract**: The rise of online social networks has facilitated the evolution of social recommender systems, which incorporate social relations to enhance users' decision-making process. With the great success of Graph Neural Networks (GNNs) in learning node representations, GNN-based social recommendations have been widely studied to model user-item interactions and user-user social relations simultaneously. Despite their great successes, recent studies have shown that these advanced recommender systems are highly vulnerable to adversarial attacks, in which attackers can inject well-designed fake user profiles to disrupt recommendation performances. While most existing studies mainly focus on argeted attacks to promote target items on vanilla recommender systems, untargeted attacks to degrade the overall prediction performance are less explored on social recommendations under a black-box scenario. To perform untargeted attacks on social recommender systems, attackers can construct malicious social relationships for fake users to enhance the attack performance. However, the coordination of social relations and item profiles is challenging for attacking black-box social recommendations. To address this limitation, we first conduct several preliminary studies to demonstrate the effectiveness of cross-community connections and cold-start items in degrading recommendations performance. Specifically, we propose a novel framework MultiAttack based on multi-agent reinforcement learning to coordinate the generation of cold-start item profiles and cross-community social relations for conducting untargeted attacks on black-box social recommendations. Comprehensive experiments on various real-world datasets demonstrate the effectiveness of our proposed attacking framework under the black-box setting.

摘要: 在线社交网络的兴起促进了社交推荐系统的发展，社交推荐系统整合了社会关系，以增强用户的决策过程。随着图神经网络(GNN)在学习节点表示方面的巨大成功，基于GNN的社交推荐被广泛研究以同时建模用户-项目交互和用户-用户社会关系。尽管它们取得了巨大的成功，但最近的研究表明，这些先进的推荐系统非常容易受到对手攻击，攻击者可以注入精心设计的虚假用户配置文件来破坏推荐性能。虽然现有的研究主要集中在香草推荐系统上为推广目标项而进行的有针对性的攻击，但在黑盒场景下的社交推荐中，降低整体预测性能的非目标攻击的研究较少。为了对社交推荐系统进行无针对性的攻击，攻击者可以为虚假用户构建恶意的社交关系，以提高攻击性能。然而，社交关系和项目简介的协调对于攻击黑箱社交推荐是具有挑战性的。为了解决这一局限性，我们首先进行了几项初步研究，以证明跨社区联系和冷启动项目在降低推荐性能方面的有效性。具体地说，我们提出了一种基于多智能体强化学习的新型框架MultiAttack，用于协调冷启动项目配置文件的生成和跨社区社会关系的生成，以对黑盒社交推荐进行无针对性的攻击。在各种真实数据集上的综合实验证明了我们提出的攻击框架在黑盒环境下的有效性。



## **35. Towards Adversarial Robustness And Backdoor Mitigation in SSL**

SSL中的对抗稳健性和后门缓解 cs.CV

8 pages, 2 figures

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2403.15918v3) [paper-pdf](http://arxiv.org/pdf/2403.15918v3)

**Authors**: Aryan Satpathy, Nilaksh Singh, Dhruva Rajwade, Somesh Kumar

**Abstract**: Self-Supervised Learning (SSL) has shown great promise in learning representations from unlabeled data. The power of learning representations without the need for human annotations has made SSL a widely used technique in real-world problems. However, SSL methods have recently been shown to be vulnerable to backdoor attacks, where the learned model can be exploited by adversaries to manipulate the learned representations, either through tampering the training data distribution, or via modifying the model itself. This work aims to address defending against backdoor attacks in SSL, where the adversary has access to a realistic fraction of the SSL training data, and no access to the model. We use novel methods that are computationally efficient as well as generalizable across different problem settings. We also investigate the adversarial robustness of SSL models when trained with our method, and show insights into increased robustness in SSL via frequency domain augmentations. We demonstrate the effectiveness of our method on a variety of SSL benchmarks, and show that our method is able to mitigate backdoor attacks while maintaining high performance on downstream tasks. Code for our work is available at github.com/Aryan-Satpathy/Backdoor

摘要: 自监督学习(SSL)在从未标记数据中学习表示方面显示出巨大的前景。无需人工注释即可学习表示的能力使SSL成为实际问题中广泛使用的技术。然而，最近已证明SSL方法容易受到后门攻击，攻击者可以通过篡改训练数据分发或通过修改模型本身来利用学习的模型来操纵学习的表示。这项工作旨在解决针对SSL中的后门攻击的防御，在这种攻击中，攻击者可以访问真实的一小部分SSL训练数据，而不能访问模型。我们使用新的方法，这些方法在计算上是有效的，并且可以在不同的问题设置中推广。我们还研究了使用我们的方法训练的SSL模型的对抗稳健性，并展示了通过频域增强来增强SSL的稳健性的见解。我们在不同的SSL基准测试上证明了我们的方法的有效性，并表明我们的方法能够在保持下游任务的高性能的同时减少后门攻击。我们工作的代码可在githorb.com/aryan-Satthy/Backdoor上找到



## **36. Trading Devil: Robust backdoor attack via Stochastic investment models and Bayesian approach**

交易魔鬼：通过随机投资模型和Bayesian方法进行强有力的后门攻击 cs.CR

(Last update!, a constructive comment from arxiv led to this latest  update ) Stochastic investment models and a Bayesian approach to better  modeling of uncertainty : adversarial machine learning or Stochastic market.  arXiv admin note: substantial text overlap with arXiv:2402.05967 (see this  link to the paper by : Orson Mengara)

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2406.10719v4) [paper-pdf](http://arxiv.org/pdf/2406.10719v4)

**Authors**: Orson Mengara

**Abstract**: With the growing use of voice-activated systems and speech recognition technologies, the danger of backdoor attacks on audio data has grown significantly. This research looks at a specific type of attack, known as a Stochastic investment-based backdoor attack (MarketBack), in which adversaries strategically manipulate the stylistic properties of audio to fool speech recognition systems. The security and integrity of machine learning models are seriously threatened by backdoor attacks, in order to maintain the reliability of audio applications and systems, the identification of such attacks becomes crucial in the context of audio data. Experimental results demonstrated that MarketBack is feasible to achieve an average attack success rate close to 100% in seven victim models when poisoning less than 1% of the training data.

摘要: 随着语音激活系统和语音识别技术的日益广泛使用，对音频数据进行后门攻击的危险显着增加。这项研究着眼于一种特定类型的攻击，称为基于随机投资的后门攻击（MarketBack），其中对手战略性地操纵音频的风格属性来愚弄语音识别系统。机器学习模型的安全性和完整性受到后门攻击的严重威胁，为了维护音频应用和系统的可靠性，识别此类攻击在音频数据环境中变得至关重要。实验结果表明，当毒害少于1%的训练数据时，MarketBack可以在7个受害者模型中实现接近100%的平均攻击成功率。



## **37. Exact Recovery Guarantees for Parameterized Non-linear System Identification Problem under Adversarial Attacks**

对抗攻击下参数化非线性系统识别问题的精确恢复保证 math.OC

33 pages

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2409.00276v2) [paper-pdf](http://arxiv.org/pdf/2409.00276v2)

**Authors**: Haixiang Zhang, Baturalp Yalcin, Javad Lavaei, Eduardo D. Sontag

**Abstract**: In this work, we study the system identification problem for parameterized non-linear systems using basis functions under adversarial attacks. Motivated by the LASSO-type estimators, we analyze the exact recovery property of a non-smooth estimator, which is generated by solving an embedded $\ell_1$-loss minimization problem. First, we derive necessary and sufficient conditions for the well-specifiedness of the estimator and the uniqueness of global solutions to the underlying optimization problem. Next, we provide exact recovery guarantees for the estimator under two different scenarios of boundedness and Lipschitz continuity of the basis functions. The non-asymptotic exact recovery is guaranteed with high probability, even when there are more severely corrupted data than clean data. Finally, we numerically illustrate the validity of our theory. This is the first study on the sample complexity analysis of a non-smooth estimator for the non-linear system identification problem.

摘要: 在这项工作中，我们研究了对抗攻击下使用基函数的参数化非线性系统的系统识别问题。受LANSO型估计器的激励，我们分析了非光滑估计器的精确恢复性质，该估计器是通过解决嵌入的$\ell_1 $-损失最小化问题而生成的。首先，我们推导出估计量的良好指定性和基本优化问题的全局解的唯一性的充要条件。接下来，我们在基函数的有界性和Lipschitz连续性两种不同场景下为估计器提供精确的恢复保证。即使存在比干净数据更严重的损坏数据，也能以高概率保证非渐进精确恢复。最后，我们用数字说明了我们理论的有效性。这是首次对非线性系统识别问题的非光滑估计器的样本复杂性分析进行研究。



## **38. LLM Whisperer: An Inconspicuous Attack to Bias LLM Responses**

LLM Whisperer：对LLM偏见回应的不起眼攻击 cs.CR

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2406.04755v2) [paper-pdf](http://arxiv.org/pdf/2406.04755v2)

**Authors**: Weiran Lin, Anna Gerchanovsky, Omer Akgul, Lujo Bauer, Matt Fredrikson, Zifan Wang

**Abstract**: Writing effective prompts for large language models (LLM) can be unintuitive and burdensome. In response, services that optimize or suggest prompts have emerged. While such services can reduce user effort, they also introduce a risk: the prompt provider can subtly manipulate prompts to produce heavily biased LLM responses. In this work, we show that subtle synonym replacements in prompts can increase the likelihood (by a difference up to 78%) that LLMs mention a target concept (e.g., a brand, political party, nation). We substantiate our observations through a user study, showing our adversarially perturbed prompts 1) are indistinguishable from unaltered prompts by humans, 2) push LLMs to recommend target concepts more often, and 3) make users more likely to notice target concepts, all without arousing suspicion. The practicality of this attack has the potential to undermine user autonomy. Among other measures, we recommend implementing warnings against using prompts from untrusted parties.

摘要: 为大型语言模型(LLM)编写有效的提示可能是不直观和繁琐的。作为回应，优化或建议提示的服务应运而生。虽然这类服务可以减少用户的工作，但它们也带来了风险：提示提供商可能会巧妙地操纵提示，以产生严重偏见的LLM响应。在这项工作中，我们表明，提示中微妙的同义词替换可以增加LLMS提到目标概念(例如，品牌、政党、国家)的可能性(差异高达78%)。我们通过一项用户研究证实了我们的观察结果，表明我们受到敌意干扰的提示1)与人类未更改的提示无法区分，2)推动LLMS更频繁地推荐目标概念，3)使用户更有可能注意到目标概念，所有这些都不会引起怀疑。这种攻击的实用性有可能破坏用户的自主性。在其他措施中，我们建议实施警告，以防止使用来自不受信任方的提示。



## **39. Revisiting Physical-World Adversarial Attack on Traffic Sign Recognition: A Commercial Systems Perspective**

重新审视对交通标志识别的物理世界对抗攻击：商业系统的角度 cs.CR

Accepted by NDSS 2025

**SubmitDate**: 2024-09-15    [abs](http://arxiv.org/abs/2409.09860v1) [paper-pdf](http://arxiv.org/pdf/2409.09860v1)

**Authors**: Ningfei Wang, Shaoyuan Xie, Takami Sato, Yunpeng Luo, Kaidi Xu, Qi Alfred Chen

**Abstract**: Traffic Sign Recognition (TSR) is crucial for safe and correct driving automation. Recent works revealed a general vulnerability of TSR models to physical-world adversarial attacks, which can be low-cost, highly deployable, and capable of causing severe attack effects such as hiding a critical traffic sign or spoofing a fake one. However, so far existing works generally only considered evaluating the attack effects on academic TSR models, leaving the impacts of such attacks on real-world commercial TSR systems largely unclear. In this paper, we conduct the first large-scale measurement of physical-world adversarial attacks against commercial TSR systems. Our testing results reveal that it is possible for existing attack works from academia to have highly reliable (100\%) attack success against certain commercial TSR system functionality, but such attack capabilities are not generalizable, leading to much lower-than-expected attack success rates overall. We find that one potential major factor is a spatial memorization design that commonly exists in today's commercial TSR systems. We design new attack success metrics that can mathematically model the impacts of such design on the TSR system-level attack success, and use them to revisit existing attacks. Through these efforts, we uncover 7 novel observations, some of which directly challenge the observations or claims in prior works due to the introduction of the new metrics.

摘要: 交通标志识别(TSR)对于安全、正确的驾驶自动化至关重要。最近的工作揭示了TSR模型对物理世界对抗性攻击的普遍脆弱性，这些攻击可以是低成本的，高度可部署的，并且能够造成严重的攻击效果，例如隐藏关键交通标志或欺骗假交通标志。然而，到目前为止，现有的工作一般只考虑评估攻击对学术TSR模型的影响，而对现实世界商业TSR系统的影响很大程度上是未知的。在本文中，我们首次进行了针对商业TSR系统的物理世界对抗性攻击的大规模测量。我们的测试结果表明，学术界现有的攻击工作有可能对某些商用TSR系统功能具有高可靠性(100%)的攻击成功，但这种攻击能力不是通用的，导致总体攻击成功率远低于预期。我们发现一个潜在的主要因素是空间记忆设计，这种设计普遍存在于今天的商业TSR系统中。我们设计了新的攻击成功度量，可以对这种设计对TSR系统级攻击成功的影响进行数学建模，并使用它们来重新审视现有的攻击。通过这些努力，我们发现了7个新颖的观察结果，其中一些由于新度量的引入直接挑战了先前工作中的观察或主张。



## **40. Federated Learning in Adversarial Environments: Testbed Design and Poisoning Resilience in Cybersecurity**

对抗环境中的联邦学习：网络安全中的测试床设计和毒害韧性 cs.CR

7 pages, 4 figures

**SubmitDate**: 2024-09-15    [abs](http://arxiv.org/abs/2409.09794v1) [paper-pdf](http://arxiv.org/pdf/2409.09794v1)

**Authors**: Hao Jian Huang, Bekzod Iskandarov, Mizanur Rahman, Hakan T. Otal, M. Abdullah Canbaz

**Abstract**: This paper presents the design and implementation of a Federated Learning (FL) testbed, focusing on its application in cybersecurity and evaluating its resilience against poisoning attacks. Federated Learning allows multiple clients to collaboratively train a global model while keeping their data decentralized, addressing critical needs for data privacy and security, particularly in sensitive fields like cybersecurity. Our testbed, built using the Flower framework, facilitates experimentation with various FL frameworks, assessing their performance, scalability, and ease of integration. Through a case study on federated intrusion detection systems, we demonstrate the testbed's capabilities in detecting anomalies and securing critical infrastructure without exposing sensitive network data. Comprehensive poisoning tests, targeting both model and data integrity, evaluate the system's robustness under adversarial conditions. Our results show that while federated learning enhances data privacy and distributed learning, it remains vulnerable to poisoning attacks, which must be mitigated to ensure its reliability in real-world applications.

摘要: 本文介绍了联邦学习(FL)测试平台的设计与实现，重点研究了其在网络安全中的应用，并对其抗中毒攻击的能力进行了评估。联合学习允许多个客户协作培训全球模型，同时保持他们的数据分散，满足数据隐私和安全的关键需求，特别是在网络安全等敏感领域。我们的试验台使用Flower框架构建，促进了各种FL框架的实验，评估了它们的性能、可扩展性和集成简易性。通过一个联合入侵检测系统的案例研究，我们展示了测试床在检测异常和保护关键基础设施方面的能力，而不会暴露敏感的网络数据。针对模型和数据完整性的全面中毒测试，评估系统在对抗条件下的健壮性。我们的结果表明，尽管联合学习增强了数据隐私和分布式学习，但它仍然容易受到中毒攻击，必须加以缓解，以确保其在现实世界应用中的可靠性。



## **41. Real-world Adversarial Defense against Patch Attacks based on Diffusion Model**

基于扩散模型的现实世界补丁攻击对抗防御 cs.CV

**SubmitDate**: 2024-09-14    [abs](http://arxiv.org/abs/2409.09406v1) [paper-pdf](http://arxiv.org/pdf/2409.09406v1)

**Authors**: Xingxing Wei, Caixin Kang, Yinpeng Dong, Zhengyi Wang, Shouwei Ruan, Yubo Chen, Hang Su

**Abstract**: Adversarial patches present significant challenges to the robustness of deep learning models, making the development of effective defenses become critical for real-world applications. This paper introduces DIFFender, a novel DIFfusion-based DeFender framework that leverages the power of a text-guided diffusion model to counter adversarial patch attacks. At the core of our approach is the discovery of the Adversarial Anomaly Perception (AAP) phenomenon, which enables the diffusion model to accurately detect and locate adversarial patches by analyzing distributional anomalies. DIFFender seamlessly integrates the tasks of patch localization and restoration within a unified diffusion model framework, enhancing defense efficacy through their close interaction. Additionally, DIFFender employs an efficient few-shot prompt-tuning algorithm, facilitating the adaptation of the pre-trained diffusion model to defense tasks without the need for extensive retraining. Our comprehensive evaluation, covering image classification and face recognition tasks, as well as real-world scenarios, demonstrates DIFFender's robust performance against adversarial attacks. The framework's versatility and generalizability across various settings, classifiers, and attack methodologies mark a significant advancement in adversarial patch defense strategies. Except for the popular visible domain, we have identified another advantage of DIFFender: its capability to easily expand into the infrared domain. Consequently, we demonstrate the good flexibility of DIFFender, which can defend against both infrared and visible adversarial patch attacks alternatively using a universal defense framework.

摘要: 对抗性补丁对深度学习模型的健壮性提出了重大挑战，使得有效防御的发展成为现实世界应用的关键。本文介绍了DIFFender，一个新的基于扩散的防御框架，它利用文本引导的扩散模型的能力来对抗敌意补丁攻击。该方法的核心是发现对抗性异常感知(AAP)现象，使扩散模型能够通过分析分布异常来准确地检测和定位对抗性补丁。DIFFender在一个统一的扩散模型框架内无缝集成了补丁定位和恢复任务，通过它们的密切交互提高了防御效率。此外，DIFFender采用了一种高效的少镜头即时调整算法，便于将预先训练的扩散模型适应于防御任务，而不需要进行广泛的再训练。我们的综合评估涵盖了图像分类和人脸识别任务，以及真实世界的场景，证明了DIFFender在对抗对手攻击方面的强大性能。该框架在各种设置、分类器和攻击方法上的多功能性和通用性标志着对抗性补丁防御策略的重大进步。除了流行的可见光领域外，我们还发现了DIFFender的另一个优势：它可以很容易地扩展到红外线领域。因此，我们展示了DIFFender良好的灵活性，它可以使用通用的防御框架交替防御红外和可见光对手补丁攻击。



## **42. Regret-Optimal Defense Against Stealthy Adversaries: A System Level Approach**

针对潜行对手的遗憾最佳防御：系统级方法 eess.SY

Accepted, IEEE Conference on Decision and Control (CDC), 2024

**SubmitDate**: 2024-09-14    [abs](http://arxiv.org/abs/2407.18448v2) [paper-pdf](http://arxiv.org/pdf/2407.18448v2)

**Authors**: Hiroyasu Tsukamoto, Joudi Hajar, Soon-Jo Chung, Fred Y. Hadaegh

**Abstract**: Modern control designs in robotics, aerospace, and cyber-physical systems rely heavily on real-world data obtained through system outputs. However, these outputs can be compromised by system faults and malicious attacks, distorting critical system information needed for secure and reliable operation. In this paper, we introduce a novel regret-optimal control framework for designing controllers that make a linear system robust against stealthy attacks, including both sensor and actuator attacks. Specifically, we present (a) a convex optimization-based system metric to quantify the regret under the worst-case stealthy attack (the difference between actual performance and optimal performance with hindsight of the attack), which adapts and improves upon the $\mathcal{H}_2$ and $\mathcal{H}_{\infty}$ norms in the presence of stealthy adversaries, (b) an optimization problem for minimizing the regret of (a) in system-level parameterization, enabling localized and distributed implementation in large-scale systems, and (c) a rank-constrained optimization problem equivalent to the optimization of (b), which can be solved using convex rank minimization methods. We also present numerical simulations that demonstrate the effectiveness of our proposed framework.

摘要: 机器人、航空航天和计算机物理系统的现代控制设计在很大程度上依赖于通过系统输出获得的真实数据。但是，这些输出可能会受到系统故障和恶意攻击的影响，从而扭曲安全可靠运行所需的关键系统信息。在本文中，我们介绍了一种新的后悔最优控制框架，用于设计控制器，使线性系统对隐身攻击具有鲁棒性，包括传感器和执行器攻击。具体地说，我们提出了(A)基于凸优化的系统度量来量化在最坏情况下的隐蔽攻击(实际性能和最优性能之间的差值)下的遗憾，该度量适应并改进了在存在隐形对手的情况下的数学{H}_2和$\数学{H}_1，(B)用于最小化(A)在系统级参数化中的遗憾的优化问题，使得在大规模系统中实现局部和分布式实现，以及(C)等同于(B)的优化的秩约束优化问题，该问题可以使用凸秩化方法来求解。我们还给出了数值模拟，证明了我们所提出的框架的有效性。



## **43. Towards Resilient and Efficient LLMs: A Comparative Study of Efficiency, Performance, and Adversarial Robustness**

迈向弹性和高效的法学硕士：效率、绩效和对抗稳健性的比较研究 cs.CL

**SubmitDate**: 2024-09-14    [abs](http://arxiv.org/abs/2408.04585v3) [paper-pdf](http://arxiv.org/pdf/2408.04585v3)

**Authors**: Xiaojing Fan, Chunliang Tao

**Abstract**: With the increasing demand for practical applications of Large Language Models (LLMs), many attention-efficient models have been developed to balance performance and computational cost. However, the adversarial robustness of these models remains under-explored. In this work, we design a framework to investigate the trade-off between efficiency, performance, and adversarial robustness of LLMs and conduct extensive experiments on three prominent models with varying levels of complexity and efficiency -- Transformer++, Gated Linear Attention (GLA) Transformer, and MatMul-Free LM -- utilizing the GLUE and AdvGLUE datasets. The AdvGLUE dataset extends the GLUE dataset with adversarial samples designed to challenge model robustness. Our results show that while the GLA Transformer and MatMul-Free LM achieve slightly lower accuracy on GLUE tasks, they demonstrate higher efficiency and either superior or comparative robustness on AdvGLUE tasks compared to Transformer++ across different attack levels. These findings highlight the potential of simplified architectures to achieve a compelling balance between efficiency, performance, and adversarial robustness, offering valuable insights for applications where resource constraints and resilience to adversarial attacks are critical.

摘要: 随着大型语言模型的实际应用需求的增加，人们已经开发了许多注意力高效的模型来平衡性能和计算成本。然而，这些模型的对抗性稳健性仍然没有得到充分的研究。在这项工作中，我们设计了一个框架来研究LLMS的效率、性能和对抗健壮性之间的权衡，并利用GLUE和AdvGLUE数据集在三个不同复杂度和效率的重要模型上进行了广泛的实验--Transformer++、门控线性注意(GLA)Transformer和MatMul-Free LM。AdvGLUE数据集使用旨在挑战模型稳健性的对抗性样本扩展了GLUE数据集。我们的结果表明，虽然GLA Transformer和MatMul-Free LM在粘合任务上的准确率略低，但在不同攻击级别上，它们在AdvGLUE任务上表现出比Transformer++更高的效率和更好的健壮性或相对较高的稳健性。这些发现突出了简化体系结构在效率、性能和对手攻击健壮性之间实现引人注目的平衡的潜力，为资源约束和对抗攻击的弹性至关重要的应用程序提供了宝贵的见解。



## **44. Tamper-Resistant Safeguards for Open-Weight LLMs**

开放重量LLM的防篡改保障措施 cs.LG

Website: https://www.tamper-resistant-safeguards.com

**SubmitDate**: 2024-09-14    [abs](http://arxiv.org/abs/2408.00761v3) [paper-pdf](http://arxiv.org/pdf/2408.00761v3)

**Authors**: Rishub Tamirisa, Bhrugu Bharathi, Long Phan, Andy Zhou, Alice Gatti, Tarun Suresh, Maxwell Lin, Justin Wang, Rowan Wang, Ron Arel, Andy Zou, Dawn Song, Bo Li, Dan Hendrycks, Mantas Mazeika

**Abstract**: Rapid advances in the capabilities of large language models (LLMs) have raised widespread concerns regarding their potential for malicious use. Open-weight LLMs present unique challenges, as existing safeguards lack robustness to tampering attacks that modify model weights. For example, recent works have demonstrated that refusal and unlearning safeguards can be trivially removed with a few steps of fine-tuning. These vulnerabilities necessitate new approaches for enabling the safe release of open-weight LLMs. We develop a method, called TAR, for building tamper-resistant safeguards into open-weight LLMs such that adversaries cannot remove the safeguards even after thousands of steps of fine-tuning. In extensive evaluations and red teaming analyses, we find that our method greatly improves tamper-resistance while preserving benign capabilities. Our results demonstrate that tamper-resistance is a tractable problem, opening up a promising new avenue to improve the safety and security of open-weight LLMs.

摘要: 大型语言模型(LLM)功能的快速发展引起了人们对其潜在恶意使用的广泛关注。开放重量LLM提出了独特的挑战，因为现有的保障措施缺乏对篡改模型权重的篡改攻击的稳健性。例如，最近的研究表明，通过几个步骤的微调，就可以很容易地消除拒绝和遗忘的保障措施。这些漏洞需要新的方法来实现安全释放未加重量的低密度脂蛋白。我们开发了一种名为TAR的方法，用于在开放重量的LLM中构建防篡改保护措施，以便对手即使在数千个步骤的微调之后也无法移除这些保护措施。在广泛的评估和红团队分析中，我们发现我们的方法在保持良性性能的同时大大提高了防篡改能力。我们的结果表明，防篡改是一个容易解决的问题，为提高开重LLMS的安全性开辟了一条很有前途的新途径。



## **45. Eliminating Catastrophic Overfitting Via Abnormal Adversarial Examples Regularization**

通过异常对抗示例规范化消除灾难性过度匹配 cs.LG

Accepted by NeurIPS 2023

**SubmitDate**: 2024-09-14    [abs](http://arxiv.org/abs/2404.08154v2) [paper-pdf](http://arxiv.org/pdf/2404.08154v2)

**Authors**: Runqi Lin, Chaojian Yu, Tongliang Liu

**Abstract**: Single-step adversarial training (SSAT) has demonstrated the potential to achieve both efficiency and robustness. However, SSAT suffers from catastrophic overfitting (CO), a phenomenon that leads to a severely distorted classifier, making it vulnerable to multi-step adversarial attacks. In this work, we observe that some adversarial examples generated on the SSAT-trained network exhibit anomalous behaviour, that is, although these training samples are generated by the inner maximization process, their associated loss decreases instead, which we named abnormal adversarial examples (AAEs). Upon further analysis, we discover a close relationship between AAEs and classifier distortion, as both the number and outputs of AAEs undergo a significant variation with the onset of CO. Given this observation, we re-examine the SSAT process and uncover that before the occurrence of CO, the classifier already displayed a slight distortion, indicated by the presence of few AAEs. Furthermore, the classifier directly optimizing these AAEs will accelerate its distortion, and correspondingly, the variation of AAEs will sharply increase as a result. In such a vicious circle, the classifier rapidly becomes highly distorted and manifests as CO within a few iterations. These observations motivate us to eliminate CO by hindering the generation of AAEs. Specifically, we design a novel method, termed Abnormal Adversarial Examples Regularization (AAER), which explicitly regularizes the variation of AAEs to hinder the classifier from becoming distorted. Extensive experiments demonstrate that our method can effectively eliminate CO and further boost adversarial robustness with negligible additional computational overhead.

摘要: 单步对抗训练(SSAT)已经证明了实现效率和稳健性的潜力。然而，SSAT存在灾难性过匹配(CO)，这一现象导致分类器严重失真，使其容易受到多步骤对抗性攻击。在这项工作中，我们观察到在SSAT训练的网络上产生的一些对抗性样本表现出异常行为，即这些训练样本虽然是由内部最大化过程产生的，但其关联损失反而减少，我们称之为异常对抗性样本(AAES)。通过进一步的分析，我们发现AAEs与分类器失真之间有密切的关系，因为AAEs的数量和输出都随着CO的开始而发生显著的变化。鉴于这一观察，我们重新检查SSAT过程并发现，在CO发生之前，分类器已经显示出轻微的失真，这表明存在很少的AAE。而且，直接对这些AAEs进行优化的分类器会加速AAEs的失真，相应地，AAEs的变化量也会急剧增加。在这样的恶性循环中，分类器迅速变得高度失真，并在几次迭代内表现为CO。这些观察结果促使我们通过阻碍AAEs的产生来消除CO。具体地说，我们设计了一种新的方法，称为异常对抗实例正则化(AAER)，它显式地规则化AAE的变化，以防止分类器变得失真。大量的实验表明，该方法可以有效地消除CO，并在几乎不增加计算开销的情况下进一步提高对手攻击的健壮性。



## **46. Layer-Aware Analysis of Catastrophic Overfitting: Revealing the Pseudo-Robust Shortcut Dependency**

灾难性过度匹配的分层感知分析：揭示伪稳健的预设依赖 cs.LG

Accepted by ICML 2024

**SubmitDate**: 2024-09-14    [abs](http://arxiv.org/abs/2405.16262v2) [paper-pdf](http://arxiv.org/pdf/2405.16262v2)

**Authors**: Runqi Lin, Chaojian Yu, Bo Han, Hang Su, Tongliang Liu

**Abstract**: Catastrophic overfitting (CO) presents a significant challenge in single-step adversarial training (AT), manifesting as highly distorted deep neural networks (DNNs) that are vulnerable to multi-step adversarial attacks. However, the underlying factors that lead to the distortion of decision boundaries remain unclear. In this work, we delve into the specific changes within different DNN layers and discover that during CO, the former layers are more susceptible, experiencing earlier and greater distortion, while the latter layers show relative insensitivity. Our analysis further reveals that this increased sensitivity in former layers stems from the formation of pseudo-robust shortcuts, which alone can impeccably defend against single-step adversarial attacks but bypass genuine-robust learning, resulting in distorted decision boundaries. Eliminating these shortcuts can partially restore robustness in DNNs from the CO state, thereby verifying that dependence on them triggers the occurrence of CO. This understanding motivates us to implement adaptive weight perturbations across different layers to hinder the generation of pseudo-robust shortcuts, consequently mitigating CO. Extensive experiments demonstrate that our proposed method, Layer-Aware Adversarial Weight Perturbation (LAP), can effectively prevent CO and further enhance robustness.

摘要: 灾难性过拟合(CO)是单步对抗训练(AT)中的一个重大挑战，表现为高度扭曲的深度神经网络(DNN)，容易受到多步对抗攻击。然而，导致决策边界扭曲的潜在因素仍然不清楚。在这项工作中，我们深入研究了不同DNN层内的具体变化，发现在CO过程中，前一层更容易受到影响，经历更早和更大的失真，而后一层表现出相对不敏感。我们的分析进一步表明，前几层敏感度的增加源于伪稳健捷径的形成，这些捷径可以无懈可击地防御单步对手攻击，但绕过了真正的稳健学习，导致决策边界扭曲。消除这些捷径可以从CO状态部分恢复DNN的稳健性，从而验证对它们的依赖是否触发了CO的发生。这种理解促使我们在不同的层上实现自适应的权重扰动，以阻止伪稳健捷径的生成，从而减少CO。大量实验表明，本文提出的层感知对抗性权重扰动(LAP)方法能够有效地防止CO，并进一步增强了鲁棒性。



## **47. Cybersecurity Software Tool Evaluation Using a 'Perfect' Network Model**

使用“完美”网络模型的网络安全软件工具评估 cs.CR

The U.S. federal sponsor has requested that we not include funding  acknowledgement for this publication

**SubmitDate**: 2024-09-13    [abs](http://arxiv.org/abs/2409.09175v1) [paper-pdf](http://arxiv.org/pdf/2409.09175v1)

**Authors**: Jeremy Straub

**Abstract**: Cybersecurity software tool evaluation is difficult due to the inherently adversarial nature of the field. A penetration testing (or offensive) tool must be tested against a viable defensive adversary and a defensive tool must, similarly, be tested against a viable offensive adversary. Characterizing the tool's performance inherently depends on the quality of the adversary, which can vary from test to test. This paper proposes the use of a 'perfect' network, representing computing systems, a network and the attack pathways through it as a methodology to use for testing cybersecurity decision-making tools. This facilitates testing by providing a known and consistent standard for comparison. It also allows testing to include researcher-selected levels of error, noise and uncertainty to evaluate cybersecurity tools under these experimental conditions.

摘要: 由于该领域固有的对抗性，网络安全软件工具评估很困难。渗透测试（或进攻性）工具必须针对可行的防御对手进行测试，同样，防御工具也必须针对可行的进攻性对手进行测试。描述工具的性能本质上取决于对手的质量，而对手的质量可能因测试而异。本文建议使用“完美”网络，代表计算系统、网络和通过它的攻击路径，作为用于测试网络安全决策工具的方法论。这通过提供已知且一致的比较标准来促进测试。它还允许测试包括研究人员选择的错误、噪音和不确定性水平，以在这些实验条件下评估网络安全工具。



## **48. Clean Label Attacks against SLU Systems**

针对SL U系统的干净标签攻击 cs.CR

Accepted at IEEE SLT 2024

**SubmitDate**: 2024-09-13    [abs](http://arxiv.org/abs/2409.08985v1) [paper-pdf](http://arxiv.org/pdf/2409.08985v1)

**Authors**: Henry Li Xinyuan, Sonal Joshi, Thomas Thebaud, Jesus Villalba, Najim Dehak, Sanjeev Khudanpur

**Abstract**: Poisoning backdoor attacks involve an adversary manipulating the training data to induce certain behaviors in the victim model by inserting a trigger in the signal at inference time. We adapted clean label backdoor (CLBD)-data poisoning attacks, which do not modify the training labels, on state-of-the-art speech recognition models that support/perform a Spoken Language Understanding task, achieving 99.8% attack success rate by poisoning 10% of the training data. We analyzed how varying the signal-strength of the poison, percent of samples poisoned, and choice of trigger impact the attack. We also found that CLBD attacks are most successful when applied to training samples that are inherently hard for a proxy model. Using this strategy, we achieved an attack success rate of 99.3% by poisoning a meager 1.5% of the training data. Finally, we applied two previously developed defenses against gradient-based attacks, and found that they attain mixed success against poisoning.

摘要: 中毒后门攻击涉及对手操纵训练数据，通过在推理时在信号中插入触发器来诱导受害者模型中的某些行为。我们在支持/执行口语理解任务的最先进语音识别模型上采用了干净标签后门（CLBD）-数据中毒攻击，其不会修改训练标签，通过毒害10%的训练数据，实现了99.8%的攻击成功率。我们分析了毒物的信号强度、中毒样本的百分比以及触发器的选择如何影响攻击。我们还发现，CLBD攻击在应用于对于代理模型来说本质上很难的训练样本时最为成功。使用该策略，我们通过毒害可怜的1.5%的训练数据，实现了99.3%的攻击成功率。最后，我们应用了两种之前开发的针对基于梯度的攻击的防御措施，并发现它们在对抗中毒方面取得了好坏参半的成功。



## **49. XSub: Explanation-Driven Adversarial Attack against Blackbox Classifiers via Feature Substitution**

XSub：通过特征替代对黑匣子分类器的描述驱动的对抗攻击 cs.LG

**SubmitDate**: 2024-09-13    [abs](http://arxiv.org/abs/2409.08919v1) [paper-pdf](http://arxiv.org/pdf/2409.08919v1)

**Authors**: Kiana Vu, Phung Lai, Truc Nguyen

**Abstract**: Despite its significant benefits in enhancing the transparency and trustworthiness of artificial intelligence (AI) systems, explainable AI (XAI) has yet to reach its full potential in real-world applications. One key challenge is that XAI can unintentionally provide adversaries with insights into black-box models, inevitably increasing their vulnerability to various attacks. In this paper, we develop a novel explanation-driven adversarial attack against black-box classifiers based on feature substitution, called XSub. The key idea of XSub is to strategically replace important features (identified via XAI) in the original sample with corresponding important features from a "golden sample" of a different label, thereby increasing the likelihood of the model misclassifying the perturbed sample. The degree of feature substitution is adjustable, allowing us to control how much of the original samples information is replaced. This flexibility effectively balances a trade-off between the attacks effectiveness and its stealthiness. XSub is also highly cost-effective in that the number of required queries to the prediction model and the explanation model in conducting the attack is in O(1). In addition, XSub can be easily extended to launch backdoor attacks in case the attacker has access to the models training data. Our evaluation demonstrates that XSub is not only effective and stealthy but also cost-effective, enabling its application across a wide range of AI models.

摘要: 尽管可解释人工智能(XAI)在提高人工智能(AI)系统的透明度和可信性方面具有显著优势，但它在现实世界的应用中尚未充分发挥其潜力。一个关键的挑战是，XAI可能会无意中向对手提供对黑盒模型的洞察，从而不可避免地增加他们对各种攻击的脆弱性。本文提出了一种新的基于特征替换的解释驱动的对抗性黑盒分类器攻击方法XSub。XSub的关键思想是策略性地将原始样本中的重要特征(通过XAI识别)替换为来自不同标签的“黄金样本”的相应重要特征，从而增加模型错误分类扰动样本的可能性。特征替换的程度是可调的，允许我们控制原始样本信息的替换量。这种灵活性有效地平衡了攻击的有效性和隐蔽性之间的权衡。XSub还具有很高的性价比，因为在进行攻击时，对预测模型和解释模型所需的查询数量为O(1)。此外，XSub可以很容易地扩展为在攻击者有权访问模型训练数据的情况下发动后门攻击。我们的评估表明，XSub不仅有效和隐身，而且性价比高，使其能够在广泛的人工智能模型中应用。



## **50. Are Existing Road Design Guidelines Suitable for Autonomous Vehicles?**

现有的道路设计指南适合自动驾驶车辆吗？ cs.CV

Currently under review by IEEE Transactions on Software Engineering  (TSE)

**SubmitDate**: 2024-09-13    [abs](http://arxiv.org/abs/2409.10562v1) [paper-pdf](http://arxiv.org/pdf/2409.10562v1)

**Authors**: Yang Sun, Christopher M. Poskitt, Jun Sun

**Abstract**: The emergence of Autonomous Vehicles (AVs) has spurred research into testing the resilience of their perception systems, i.e. to ensure they are not susceptible to making critical misjudgements. It is important that they are tested not only with respect to other vehicles on the road, but also those objects placed on the roadside. Trash bins, billboards, and greenery are all examples of such objects, typically placed according to guidelines that were developed for the human visual system, and which may not align perfectly with the needs of AVs. Existing tests, however, usually focus on adversarial objects with conspicuous shapes/patches, that are ultimately unrealistic given their unnatural appearances and the need for white box knowledge. In this work, we introduce a black box attack on the perception systems of AVs, in which the objective is to create realistic adversarial scenarios (i.e. satisfying road design guidelines) by manipulating the positions of common roadside objects, and without resorting to `unnatural' adversarial patches. In particular, we propose TrashFuzz , a fuzzing algorithm to find scenarios in which the placement of these objects leads to substantial misperceptions by the AV -- such as mistaking a traffic light's colour -- with overall the goal of causing it to violate traffic laws. To ensure the realism of these scenarios, they must satisfy several rules encoding regulatory guidelines about the placement of objects on public streets. We implemented and evaluated these attacks for the Apollo, finding that TrashFuzz induced it into violating 15 out of 24 different traffic laws.

摘要: 自动驾驶汽车(AVs)的出现促使人们研究测试其感知系统的弹性，即确保它们不容易做出关键的误判。重要的是，不仅要对道路上的其他车辆进行测试，还要对放置在路边的那些物体进行测试。垃圾桶、广告牌和绿色植物都是这种物体的例子，通常是根据为人类视觉系统开发的指导方针放置的，可能不能完全符合AVs的需求。然而，现有的测试通常集中在具有明显形状/补丁的对抗性对象上，考虑到它们不自然的外观和对白盒知识的需要，这些最终是不现实的。在这项工作中，我们引入了一种针对自动驾驶系统感知系统的黑盒攻击，其目的是通过操纵常见路旁对象的位置来创建现实的对抗性场景(即满足道路设计准则)，而不求助于不自然的对抗性补丁。特别是，我们提出了TrashFuzz，这是一种模糊算法，用于查找这些对象的放置导致AV产生重大误解的场景--例如错误地识别红绿灯的颜色--总体目标是导致它违反交通法规。为了确保这些场景的真实性，它们必须满足几项规则，这些规则编码了关于在公共街道上放置物体的监管指南。我们为阿波罗实施并评估了这些攻击，发现TrashFuzz导致它违反了24项不同交通法规中的15项。



