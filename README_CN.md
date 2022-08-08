# Latest Adversarial Attack Papers
**update at 2022-08-09 06:31:23**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Attacking Adversarial Defences by Smoothing the Loss Landscape**

通过平滑损失图景来攻击对抗性防御 cs.LG

**SubmitDate**: 2022-08-05    [paper-pdf](http://arxiv.org/pdf/2208.00862v2)

**Authors**: Panagiotis Eustratiadis, Henry Gouk, Da Li, Timothy Hospedales

**Abstracts**: This paper investigates a family of methods for defending against adversarial attacks that owe part of their success to creating a noisy, discontinuous, or otherwise rugged loss landscape that adversaries find difficult to navigate. A common, but not universal, way to achieve this effect is via the use of stochastic neural networks. We show that this is a form of gradient obfuscation, and propose a general extension to gradient-based adversaries based on the Weierstrass transform, which smooths the surface of the loss function and provides more reliable gradient estimates. We further show that the same principle can strengthen gradient-free adversaries. We demonstrate the efficacy of our loss-smoothing method against both stochastic and non-stochastic adversarial defences that exhibit robustness due to this type of obfuscation. Furthermore, we provide analysis of how it interacts with Expectation over Transformation; a popular gradient-sampling method currently used to attack stochastic defences.

摘要: 本文研究了一系列防御对手攻击的方法，这些攻击的成功部分归因于创建了一个嘈杂的、不连续的或以其他方式崎岖的损失场景，对手发现很难导航。实现这一效果的一种常见但并不普遍的方法是通过使用随机神经网络。我们证明了这是一种梯度混淆的形式，并提出了一种基于魏尔斯特拉斯变换的对基于梯度的攻击的一般扩展，它平滑了损失函数的表面，并提供了更可靠的梯度估计。我们进一步证明，同样的原理可以加强无梯度的对手。我们证明了我们的损失平滑方法对随机和非随机对抗防御的有效性，这些防御由于这种类型的混淆而表现出稳健性。此外，我们还分析了它如何与变换上的期望相互作用，变换上的期望是目前用于攻击随机防御的一种流行的梯度抽样方法。



## **2. Adversarial Robustness of MR Image Reconstruction under Realistic Perturbations**

现实摄动下MR图像重建的对抗稳健性 eess.IV

Accepted at the MICCAI-2022 workshop: Machine Learning for Medical  Image Reconstruction

**SubmitDate**: 2022-08-05    [paper-pdf](http://arxiv.org/pdf/2208.03161v1)

**Authors**: Jan Nikolas Morshuis, Sergios Gatidis, Matthias Hein, Christian F. Baumgartner

**Abstracts**: Deep Learning (DL) methods have shown promising results for solving ill-posed inverse problems such as MR image reconstruction from undersampled $k$-space data. However, these approaches currently have no guarantees for reconstruction quality and the reliability of such algorithms is only poorly understood. Adversarial attacks offer a valuable tool to understand possible failure modes and worst case performance of DL-based reconstruction algorithms. In this paper we describe adversarial attacks on multi-coil $k$-space measurements and evaluate them on the recently proposed E2E-VarNet and a simpler UNet-based model. In contrast to prior work, the attacks are targeted to specifically alter diagnostically relevant regions. Using two realistic attack models (adversarial $k$-space noise and adversarial rotations) we are able to show that current state-of-the-art DL-based reconstruction algorithms are indeed sensitive to such perturbations to a degree where relevant diagnostic information may be lost. Surprisingly, in our experiments the UNet and the more sophisticated E2E-VarNet were similarly sensitive to such attacks. Our findings add further to the evidence that caution must be exercised as DL-based methods move closer to clinical practice.

摘要: 深度学习方法在解决不适定反问题，如从欠采样的$k$空间数据重建MR图像方面显示出良好的结果。然而，这些方法目前并不能保证重建质量，而且人们对这些算法的可靠性知之甚少。对抗性攻击为理解基于DL的重构算法可能的失效模式和最坏情况下的性能提供了有价值的工具。在这篇文章中，我们描述了对多线圈$k$空间测量的对抗性攻击，并在最近提出的E2E-Varnet和一个更简单的基于UNT的模型上对它们进行了评估。与以前的工作不同，这些攻击的目标是专门改变诊断相关的区域。使用两个真实的攻击模型(对抗性空间噪声和对抗性旋转)，我们能够证明当前最先进的基于DL的重建算法确实对此类扰动敏感到可能丢失相关诊断信息的程度。令人惊讶的是，在我们的实验中，UNT和更复杂的E2E-Varnet对此类攻击同样敏感。我们的发现进一步证明，随着基于DL的方法越来越接近临床实践，必须谨慎行事。



## **3. A Systematic Survey of Attack Detection and Prevention in Connected and Autonomous Vehicles**

互联无人驾驶汽车攻击检测与防御的系统研究 cs.CR

This article is published in the Vehicular Communications journal

**SubmitDate**: 2022-08-05    [paper-pdf](http://arxiv.org/pdf/2203.14965v2)

**Authors**: Trupil Limbasiya, Ko Zheng Teng, Sudipta Chattopadhyay, Jianying Zhou

**Abstracts**: The number of Connected and Autonomous Vehicles (CAVs) is increasing rapidly in various smart transportation services and applications, considering many benefits to society, people, and the environment. Several research surveys for CAVs were conducted by primarily focusing on various security threats and vulnerabilities in the domain of CAVs to classify different types of attacks, impacts of attacks, attack features, cyber-risk, defense methodologies against attacks, and safety standards. However, the importance of attack detection and prevention approaches for CAVs has not been discussed extensively in the state-of-the-art surveys, and there is a clear gap in the existing literature on such methodologies to detect new and conventional threats and protect the CAV systems from unexpected hazards on the road. Some surveys have a limited discussion on Attacks Detection and Prevention Systems (ADPS), but such surveys provide only partial coverage of different types of ADPS for CAVs. Furthermore, there is a scope for discussing security, privacy, and efficiency challenges in ADPS that can give an overview of important security and performance attributes.   This survey paper, therefore, presents the significance of CAVs in the market, potential challenges in CAVs, key requirements of essential security and privacy properties, various capabilities of adversaries, possible attacks in CAVs, and performance evaluation parameters for ADPS. An extensive analysis is discussed of different ADPS categories for CAVs and state-of-the-art research works based on each ADPS category that gives the latest findings in this research domain. This survey also discusses crucial and open security research problems that are required to be focused on the secure deployment of CAVs in the market.

摘要: 考虑到对社会、人民和环境的诸多好处，在各种智能交通服务和应用中，互联和自动驾驶车辆(CAV)的数量正在迅速增加。通过主要关注CAV领域中的各种安全威胁和漏洞，对CAV进行了几项研究调查，以对不同类型的攻击、攻击的影响、攻击特征、网络风险、攻击防御方法和安全标准进行分类。然而，在最新的调查中，攻击检测和预防方法对CAV的重要性没有得到广泛的讨论，并且在现有文献中，对于检测新的和常规的威胁并保护CAV系统免受道路上的意外危险的方法，存在着明显的空白。一些调查对攻击检测和预防系统(ADPS)进行了有限的讨论，但这些调查只提供了针对骑士队不同类型的ADP的部分覆盖。此外，ADPS中还有讨论安全、隐私和效率挑战的范围，可以概述重要的安全和性能属性。因此，这份调查报告介绍了CAV在市场上的重要性、CAV中的潜在挑战、基本安全和隐私属性的关键要求、对手的各种能力、CAV中可能的攻击以及ADPS的性能评估参数。对CAV的不同ADPS类别进行了广泛的分析，并根据每个ADPS类别进行了最新的研究工作，给出了该研究领域的最新发现。本次调查还讨论了需要重点关注Cavs在市场上的安全部署的关键和开放的安全研究问题。



## **4. Differentially Private Counterfactuals via Functional Mechanism**

从作用机制看区分私法反事实 cs.LG

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02878v1)

**Authors**: Fan Yang, Qizhang Feng, Kaixiong Zhou, Jiahao Chen, Xia Hu

**Abstracts**: Counterfactual, serving as one emerging type of model explanation, has attracted tons of attentions recently from both industry and academia. Different from the conventional feature-based explanations (e.g., attributions), counterfactuals are a series of hypothetical samples which can flip model decisions with minimal perturbations on queries. Given valid counterfactuals, humans are capable of reasoning under ``what-if'' circumstances, so as to better understand the model decision boundaries. However, releasing counterfactuals could be detrimental, since it may unintentionally leak sensitive information to adversaries, which brings about higher risks on both model security and data privacy. To bridge the gap, in this paper, we propose a novel framework to generate differentially private counterfactual (DPC) without touching the deployed model or explanation set, where noises are injected for protection while maintaining the explanation roles of counterfactual. In particular, we train an autoencoder with the functional mechanism to construct noisy class prototypes, and then derive the DPC from the latent prototypes based on the post-processing immunity of differential privacy. Further evaluations demonstrate the effectiveness of the proposed framework, showing that DPC can successfully relieve the risks on both extraction and inference attacks.

摘要: 反事实作为一种新兴的模型解释，近年来引起了产业界和学术界的广泛关注。与传统的基于特征的解释(例如，属性)不同，反事实是一系列假设样本，可以在对查询的扰动最小的情况下反转模型决策。在有了有效的反事实的情况下，人类能够在“假设”的情况下进行推理，从而更好地理解模型决策的边界。然而，发布反事实可能是有害的，因为它可能会无意中将敏感信息泄露给对手，这会给模型安全和数据隐私带来更高的风险。为了弥补这一差距，在本文中，我们提出了一种新的框架，在不接触部署的模型或解释集的情况下，生成差异私有反事实(DPC)，其中注入噪声以保护，同时保持反事实的解释角色。特别是，我们训练了一个具有构造噪声类原型的功能机制的自动编码器，然后基于差分隐私的后处理免疫力从潜在原型中推导出DPC。进一步的测试证明了该框架的有效性，表明DPC能够成功地缓解抽取攻击和推理攻击的风险。



## **5. Self-Ensembling Vision Transformer (SEViT) for Robust Medical Image Classification**

自集成视觉转换器(SEViT)用于稳健的医学图像分类 cs.CV

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02851v1)

**Authors**: Faris Almalik, Mohammad Yaqub, Karthik Nandakumar

**Abstracts**: Vision Transformers (ViT) are competing to replace Convolutional Neural Networks (CNN) for various computer vision tasks in medical imaging such as classification and segmentation. While the vulnerability of CNNs to adversarial attacks is a well-known problem, recent works have shown that ViTs are also susceptible to such attacks and suffer significant performance degradation under attack. The vulnerability of ViTs to carefully engineered adversarial samples raises serious concerns about their safety in clinical settings. In this paper, we propose a novel self-ensembling method to enhance the robustness of ViT in the presence of adversarial attacks. The proposed Self-Ensembling Vision Transformer (SEViT) leverages the fact that feature representations learned by initial blocks of a ViT are relatively unaffected by adversarial perturbations. Learning multiple classifiers based on these intermediate feature representations and combining these predictions with that of the final ViT classifier can provide robustness against adversarial attacks. Measuring the consistency between the various predictions can also help detect adversarial samples. Experiments on two modalities (chest X-ray and fundoscopy) demonstrate the efficacy of SEViT architecture to defend against various adversarial attacks in the gray-box (attacker has full knowledge of the target model, but not the defense mechanism) setting. Code: https://github.com/faresmalik/SEViT

摘要: 视觉转换器(VIT)正竞相取代卷积神经网络(CNN)用于医学成像中的各种计算机视觉任务，如分类和分割。虽然CNN对敌意攻击的脆弱性是一个众所周知的问题，但最近的研究表明，VITS也容易受到此类攻击，并且在攻击下性能显著下降。VITS对精心设计的对抗性样本的脆弱性引起了人们对其临床安全性的严重担忧。在本文中，我们提出了一种新的自集成方法来增强VIT在存在对手攻击时的健壮性。提出的自集成视觉转换器(SEViT)利用了这样一个事实，即VIT的初始块学习的特征表示相对不受对抗性扰动的影响。基于这些中间特征表示学习多个分类器，并将这些预测与最终的VIT分类器的预测相结合，可以提供对对手攻击的稳健性。衡量各种预测之间的一致性也有助于检测敌意样本。在两种模式(胸部X光和眼底镜)上的实验证明了SEViT架构在灰盒(攻击者完全知道目标模型，但不知道防御机制)环境下防御各种对抗性攻击的有效性。代码：https://github.com/faresmalik/SEViT



## **6. Mass Exit Attacks on the Lightning Network**

闪电网络上的大规模出口攻击 cs.CR

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.01908v2)

**Authors**: Cosimo Sguanci, Anastasios Sidiropoulos

**Abstracts**: The Lightning Network (LN) has enjoyed rapid growth over recent years, and has become the most popular scaling solution for the Bitcoin blockchain. The security of the LN hinges on the ability of the nodes to close a channel by settling their balances, which requires confirming a transaction on the Bitcoin blockchain within a pre-agreed time period. This inherent timing restriction that the LN must satisfy, make it susceptible to attacks that seek to increase the congestion on the Bitcoin blockchain, thus preventing correct protocol execution. We study the susceptibility of the LN to \emph{mass exit} attacks, in the presence of a small coalition of adversarial nodes. This is a scenario where an adversary forces a large set of honest protocol participants to interact with the blockchain. We focus on two types of attacks: (i) The first is a \emph{zombie} attack, where a set of $k$ nodes become unresponsive with the goal to lock the funds of many channels for a period of time longer than what the LN protocol dictates. (ii) The second is a \emph{mass double-spend} attack, where a set of $k$ nodes attempt to steal funds by submitting many closing transactions that settle channels using expired protocol states; this causes many honest nodes to have to quickly respond by submitting invalidating transactions. We show via simulations that, under historically-plausible congestion conditions, with mild statistical assumptions on channel balances, both of the attacks can be performed by a very small coalition. To perform our simulations, we formulate the problem of finding a worst-case coalition of $k$ adversarial nodes as a graph cut problem. Our experimental findings are supported by a theoretical justification based on the scale-free topology of the LN.

摘要: 闪电网络(Lightning Network，LN)近年来增长迅速，已成为比特币区块链最受欢迎的扩展解决方案。LN的安全性取决于节点通过结算余额关闭通道的能力，这需要在预先商定的时间段内确认比特币区块链上的交易。LN必须满足的这一固有时间限制使其容易受到攻击，这些攻击试图增加比特币区块链上的拥塞，从而阻止正确的协议执行。我们研究了在存在一个小的敌方节点联盟的情况下，LN对EMPH{MASS EXIT}攻击的敏感性。这是一种对手迫使大量诚实的协议参与者与区块链交互的场景。我们主要关注两种类型的攻击：(I)第一种是僵尸攻击，其中一组$k$节点变得无响应，目标是锁定多个频道的资金长于LN协议规定的时间段。(Ii)第二种攻击是大规模双重花费攻击，其中一组$k$节点试图通过提交许多关闭的事务来窃取资金，这些事务使用过期的协议状态来结算通道；这导致许多诚实的节点不得不通过提交无效事务来快速响应。我们通过模拟表明，在历史上看似合理的拥塞条件下，在对信道平衡的温和统计假设下，这两种攻击都可以由非常小的联盟来执行。为了执行我们的模拟，我们将寻找$k$个敌对节点的最坏情况联盟的问题描述为一个图割问题。我们的实验结果得到了基于LN的无标度拓扑的理论证明。



## **7. Design Considerations and Architecture for a Resilient Risk based Adaptive Authentication and Authorization (RAD-AA) Framework**

基于风险的弹性自适应身份验证和授权(RAD-AA)框架的设计注意事项和体系结构 cs.CR

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02592v1)

**Authors**: Jaimandeep Singh, Chintan Patel, Naveen Kumar Chaudhary

**Abstracts**: A strong cyber attack is capable of degrading the performance of any Information Technology (IT) or Operational Technology (OT) system. In recent cyber attacks, credential theft emerged as one of the primary vectors of gaining entry into the system. Once, an attacker has a foothold in the system, they use token manipulation techniques to elevate the privileges and access protected resources. This makes authentication and authorization a critical component for a secure and resilient cyber system. In this paper we consider the design considerations for such a secure and resilient authentication and authorization framework capable of self-adapting based on the risk scores and trust profiles. We compare this design with the existing standards such as OAuth 2.0, OpenID Connect and SAML 2.0. We then study popular threat models such as STRIDE and PASTA and summarize the resilience of the proposed architecture against common and relevant threat vectors. We call this framework Resilient Risk-based Adaptive Authentication and Authorization (RAD-AA). The proposed framework excessively increases the cost for an adversary to launch any cyber attack and provides much-needed strength to critical infrastructure.

摘要: 强大的网络攻击能够降低任何信息技术(IT)或操作技术(OT)系统的性能。在最近的网络攻击中，凭据盗窃成为进入系统的主要载体之一。一旦攻击者在系统中站稳脚跟，他们就会使用令牌操作技术来提升权限并访问受保护的资源。这使得身份验证和授权成为安全和有弹性的网络系统的关键组件。在本文中，我们考虑了这样一个安全的、具有弹性的认证和授权框架的设计考虑因素，该框架能够基于风险分数和信任配置文件自适应。我们将该设计与OAuth 2.0、OpenID Connect和SAML 2.0等现有标准进行了比较。然后，我们研究了流行的威胁模型，如STRIDE和PASA，并总结了所提出的体系结构对常见和相关威胁向量的恢复能力。我们将此框架称为弹性基于风险的自适应身份验证和授权(RAD-AA)。拟议的框架过度增加了对手发动任何网络攻击的成本，并为关键基础设施提供了亟需的力量。



## **8. Prompt Tuning for Generative Multimodal Pretrained Models**

产生式多模式预训练模型的快速调整 cs.CL

Work in progress

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02532v1)

**Authors**: Hao Yang, Junyang Lin, An Yang, Peng Wang, Chang Zhou, Hongxia Yang

**Abstracts**: Prompt tuning has become a new paradigm for model tuning and it has demonstrated success in natural language pretraining and even vision pretraining. In this work, we explore the transfer of prompt tuning to multimodal pretraining, with a focus on generative multimodal pretrained models, instead of contrastive ones. Specifically, we implement prompt tuning on the unified sequence-to-sequence pretrained model adaptive to both understanding and generation tasks. Experimental results demonstrate that the light-weight prompt tuning can achieve comparable performance with finetuning and surpass other light-weight tuning methods. Besides, in comparison with finetuned models, the prompt-tuned models demonstrate improved robustness against adversarial attacks. We further figure out that experimental factors, including the prompt length, prompt depth, and reparameteratization, have great impacts on the model performance, and thus we empirically provide a recommendation for the setups of prompt tuning. Despite the observed advantages, we still find some limitations in prompt tuning, and we correspondingly point out the directions for future studies. Codes are available at \url{https://github.com/OFA-Sys/OFA}

摘要: 即时调优已经成为一种新的模型调优范式，在自然语言预训练甚至视觉预训练中都取得了成功。在这项工作中，我们探索了从即时调整到多模式预训练的转换，重点是生成性的多模式预训练模型，而不是对比模型。具体地说，我们实现了对统一的序列到序列的预训练模型的即时调整，该模型同时适用于理解和生成任务。实验结果表明，轻量级快速调谐可以达到与精调相当的性能，并超过其他轻量级调谐方法。此外，与精调模型相比，快速调谐模型具有更好的抗敌意攻击能力。我们进一步发现，实验因素，包括提示长度、提示深度和再参数化，对模型的性能有很大的影响，因此，我们实证地为提示调整的设置提供了建议。尽管观察到了这些优点，但我们仍然发现了快速调谐的一些局限性，并相应地为未来的研究指明了方向。代码可在\url{https://github.com/OFA-Sys/OFA}



## **9. NoiLIn: Improving Adversarial Training and Correcting Stereotype of Noisy Labels**

NoiLin：改进对抗性训练，纠正对嘈杂标签的刻板印象 cs.LG

Accepted at Transactions on Machine Learning Research (TMLR) at June  2022

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2105.14676v2)

**Authors**: Jingfeng Zhang, Xilie Xu, Bo Han, Tongliang Liu, Gang Niu, Lizhen Cui, Masashi Sugiyama

**Abstracts**: Adversarial training (AT) formulated as the minimax optimization problem can effectively enhance the model's robustness against adversarial attacks. The existing AT methods mainly focused on manipulating the inner maximization for generating quality adversarial variants or manipulating the outer minimization for designing effective learning objectives. However, empirical results of AT always exhibit the robustness at odds with accuracy and the existence of the cross-over mixture problem, which motivates us to study some label randomness for benefiting the AT. First, we thoroughly investigate noisy labels (NLs) injection into AT's inner maximization and outer minimization, respectively and obtain the observations on when NL injection benefits AT. Second, based on the observations, we propose a simple but effective method -- NoiLIn that randomly injects NLs into training data at each training epoch and dynamically increases the NL injection rate once robust overfitting occurs. Empirically, NoiLIn can significantly mitigate the AT's undesirable issue of robust overfitting and even further improve the generalization of the state-of-the-art AT methods. Philosophically, NoiLIn sheds light on a new perspective of learning with NLs: NLs should not always be deemed detrimental, and even in the absence of NLs in the training set, we may consider injecting them deliberately. Codes are available in https://github.com/zjfheart/NoiLIn.

摘要: 对抗性训练(AT)被描述为极小极大优化问题，可以有效地增强模型对对手攻击的鲁棒性。现有的AT方法主要集中在操纵内部最大化来生成高质量的对抗性变体，或者操纵外部最小化来设计有效的学习目标。然而，AT的实验结果总是表现出与准确性不一致的稳健性，以及交叉混合问题的存在，这促使我们研究一些标签随机性，以利于AT。首先，我们深入研究了噪声标签(NLS)注入到AT的内极大化和外极小化中，并得到了NL注入何时有利于AT的观察。其次，在此基础上，提出了一种简单而有效的方法--NoiLIn方法，该方法在每个训练时段将NLS随机注入到训练数据中，并在出现稳健过拟合时动态增加NL的注入率。经验证明，NoiLIn可以显著缓解AT的健壮性过拟合的不良问题，甚至进一步改进最先进的AT方法的普适性。从哲学上讲，NoiLin揭示了使用NLS学习的新视角：NLS不应该总是被认为是有害的，即使训练集中没有NLS，我们也可以考虑故意注入它们。代码在https://github.com/zjfheart/NoiLIn.中可用



## **10. A Robust graph attention network with dynamic adjusted Graph**

一种具有动态调整图的健壮图注意网络 cs.LG

21 pages,13 figures

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2009.13038v3)

**Authors**: Xianchen Zhou, Yaoyun Zeng, Hongxia Wang

**Abstracts**: Graph Attention Networks(GATs) are useful deep learning models to deal with the graph data. However, recent works show that the classical GAT is vulnerable to adversarial attacks. It degrades dramatically with slight perturbations. Therefore, how to enhance the robustness of GAT is a critical problem. Robust GAT(RoGAT) is proposed in this paper to improve the robustness of GAT based on the revision of the attention mechanism. Different from the original GAT, which uses the attention mechanism for different edges but is still sensitive to the perturbation, RoGAT adds an extra dynamic attention score progressively and improves the robustness. Firstly, RoGAT revises the edges weight based on the smoothness assumption which is quite common for ordinary graphs. Secondly, RoGAT further revises the features to suppress features' noise. Then, an extra attention score is generated by the dynamic edge's weight and can be used to reduce the impact of adversarial attacks. Different experiments against targeted and untargeted attacks on citation data on citation data demonstrate that RoGAT outperforms most of the recent defensive methods.

摘要: 图注意网络是处理图数据的一种有用的深度学习模型。然而，最近的研究表明，经典的GAT很容易受到对抗性攻击。在轻微的扰动下，它会急剧退化。因此，如何增强GAT的健壮性是一个关键问题。在对注意机制进行修改的基础上，提出了健壮性GAT(ROGAT)，以提高GAT的健壮性。与原有的GAT算法对不同的边缘使用注意机制但对扰动仍然敏感不同，该算法渐进地增加了额外的动态注意分数，提高了算法的鲁棒性。首先，基于光滑性假设对边的权值进行修正，这在普通图中是很常见的。其次，ROAT进一步修正特征以抑制特征的噪声。然后，通过动态边缘的权重产生额外的注意力分数，并可用于减少对抗性攻击的影响。针对引文数据上的定向攻击和非定向攻击的不同实验表明，蟑螂的表现优于大多数最近的防御方法。



## **11. Privacy Safe Representation Learning via Frequency Filtering Encoder**

基于频率滤波编码器的隐私安全表征学习 cs.CV

The IJCAI-ECAI-22 Workshop on Artificial Intelligence Safety  (AISafety 2022)

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02482v1)

**Authors**: Jonghu Jeong, Minyong Cho, Philipp Benz, Jinwoo Hwang, Jeewook Kim, Seungkwan Lee, Tae-hoon Kim

**Abstracts**: Deep learning models are increasingly deployed in real-world applications. These models are often deployed on the server-side and receive user data in an information-rich representation to solve a specific task, such as image classification. Since images can contain sensitive information, which users might not be willing to share, privacy protection becomes increasingly important. Adversarial Representation Learning (ARL) is a common approach to train an encoder that runs on the client-side and obfuscates an image. It is assumed, that the obfuscated image can safely be transmitted and used for the task on the server without privacy concerns. However, in this work, we find that training a reconstruction attacker can successfully recover the original image of existing ARL methods. To this end, we introduce a novel ARL method enhanced through low-pass filtering, limiting the available information amount to be encoded in the frequency domain. Our experimental results reveal that our approach withstands reconstruction attacks while outperforming previous state-of-the-art methods regarding the privacy-utility trade-off. We further conduct a user study to qualitatively assess our defense of the reconstruction attack.

摘要: 深度学习模型越来越多地部署在现实世界的应用中。这些模型通常部署在服务器端，并以信息丰富的表示形式接收用户数据，以解决特定任务，如图像分类。由于图像可能包含用户可能不愿意分享的敏感信息，因此隐私保护变得越来越重要。对抗表示学习(ARL)是一种常见的方法，用于训练运行在客户端并对图像进行混淆的编码器。假设模糊后的图像可以安全地传输并用于服务器上的任务，而不会引起隐私问题。然而，在这项工作中，我们发现训练一个重建攻击者可以成功地恢复现有ARL方法的原始图像。为此，我们引入了一种新的ARL方法，通过低通滤波来增强，限制了可在频域中编码的信息量。我们的实验结果表明，我们的方法经受住了重建攻击，同时优于之前关于隐私效用权衡的最新方法。我们进一步进行了一项用户研究，以定性评估我们对重建攻击的防御。



## **12. Node Copying: A Random Graph Model for Effective Graph Sampling**

节点复制：一种有效图采样的随机图模型 stat.ML

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02435v1)

**Authors**: Florence Regol, Soumyasundar Pal, Jianing Sun, Yingxue Zhang, Yanhui Geng, Mark Coates

**Abstracts**: There has been an increased interest in applying machine learning techniques on relational structured-data based on an observed graph. Often, this graph is not fully representative of the true relationship amongst nodes. In these settings, building a generative model conditioned on the observed graph allows to take the graph uncertainty into account. Various existing techniques either rely on restrictive assumptions, fail to preserve topological properties within the samples or are prohibitively expensive for larger graphs. In this work, we introduce the node copying model for constructing a distribution over graphs. Sampling of a random graph is carried out by replacing each node's neighbors by those of a randomly sampled similar node. The sampled graphs preserve key characteristics of the graph structure without explicitly targeting them. Additionally, sampling from this model is extremely simple and scales linearly with the nodes. We show the usefulness of the copying model in three tasks. First, in node classification, a Bayesian formulation based on node copying achieves higher accuracy in sparse data settings. Second, we employ our proposed model to mitigate the effect of adversarial attacks on the graph topology. Last, incorporation of the model in a recommendation system setting improves recall over state-of-the-art methods.

摘要: 在基于观察到的图的关系结构数据上应用机器学习技术已经引起了越来越多的兴趣。通常，此图不能完全代表节点之间的真实关系。在这些设置中，建立以观察到的图为条件的生成模型允许将图的不确定性考虑在内。现有的各种技术要么依赖于限制性假设，要么不能保持样本中的拓扑属性，要么对于更大的图来说昂贵得令人望而却步。在这项工作中，我们引入了节点复制模型来构造图上的分布。随机图的采样是通过用随机采样的相似节点的邻居替换每个节点的邻居来执行的。采样的图形保留了图形结构的关键特征，而没有明确地以它们为目标。此外，该模型的采样非常简单，并且随着节点的增加而线性扩展。我们在三个任务中展示了复制模型的有效性。首先，在节点分类中，基于节点复制的贝叶斯公式在稀疏数据环境下实现了更高的精度。其次，我们使用我们提出的模型来缓解对抗性攻击对图拓扑的影响。最后，将该模型结合到推荐系统设置中，提高了对最先进方法的召回率。



## **13. Is current research on adversarial robustness addressing the right problem?**

目前关于对手稳健性的研究解决了正确的问题吗？ cs.CV

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.00539v2)

**Authors**: Ali Borji

**Abstracts**: Short answer: Yes, Long answer: No! Indeed, research on adversarial robustness has led to invaluable insights helping us understand and explore different aspects of the problem. Many attacks and defenses have been proposed over the last couple of years. The problem, however, remains largely unsolved and poorly understood. Here, I argue that the current formulation of the problem serves short term goals, and needs to be revised for us to achieve bigger gains. Specifically, the bound on perturbation has created a somewhat contrived setting and needs to be relaxed. This has misled us to focus on model classes that are not expressive enough to begin with. Instead, inspired by human vision and the fact that we rely more on robust features such as shape, vertices, and foreground objects than non-robust features such as texture, efforts should be steered towards looking for significantly different classes of models. Maybe instead of narrowing down on imperceptible adversarial perturbations, we should attack a more general problem which is finding architectures that are simultaneously robust to perceptible perturbations, geometric transformations (e.g. rotation, scaling), image distortions (lighting, blur), and more (e.g. occlusion, shadow). Only then we may be able to solve the problem of adversarial vulnerability.

摘要: 简短的答案是：是的，长期的答案是：不！事实上，对对手健壮性的研究已经带来了宝贵的见解，帮助我们理解和探索问题的不同方面。在过去的几年里，已经提出了许多攻击和防御措施。然而，这个问题在很大程度上仍然没有得到解决，人们对此知之甚少。在这里，我认为，目前对问题的表述是为短期目标服务的，需要进行修改，以便我们实现更大的收益。具体地说，微扰的界限创造了一种有点做作的设置，需要放松。这误导了我们将注意力集中在一开始就不够有表现力的模型类上。取而代之的是，受人类视觉的启发，以及我们更依赖于形状、顶点和前景对象等稳健特征而不是纹理等非稳健特征的事实，应该努力寻找显著不同类别的模型。也许我们不应该缩小到不可感知的对抗性扰动，而应该解决一个更一般的问题，即寻找同时对可感知扰动、几何变换(例如旋转、缩放)、图像失真(照明、模糊)以及更多(例如遮挡、阴影)具有健壮性的体系结构。只有到那时，我们才可能解决对手脆弱性的问题。



## **14. A New Kind of Adversarial Example**

一种新的对抗性例证 cs.CV

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02430v1)

**Authors**: Ali Borji

**Abstracts**: Almost all adversarial attacks are formulated to add an imperceptible perturbation to an image in order to fool a model. Here, we consider the opposite which is adversarial examples that can fool a human but not a model. A large enough and perceptible perturbation is added to an image such that a model maintains its original decision, whereas a human will most likely make a mistake if forced to decide (or opt not to decide at all). Existing targeted attacks can be reformulated to synthesize such adversarial examples. Our proposed attack, dubbed NKE, is similar in essence to the fooling images, but is more efficient since it uses gradient descent instead of evolutionary algorithms. It also offers a new and unified perspective into the problem of adversarial vulnerability. Experimental results over MNIST and CIFAR-10 datasets show that our attack is quite efficient in fooling deep neural networks. Code is available at https://github.com/aliborji/NKE.

摘要: 几乎所有的对抗性攻击都是为了给图像添加一个难以察觉的扰动，以愚弄模型。在这里，我们考虑的是相反的情况，即可以愚弄人类但不能愚弄模型的对抗性例子。一个足够大和可感知的扰动被添加到图像中，使得模型保持其原始决定，而如果被迫做出决定(或者选择根本不决定)，人类很可能会犯错误。现有的有针对性的攻击可以重新制定，以合成这种对抗性的例子。我们提出的名为NKE的攻击在本质上类似于愚弄图像，但由于它使用了梯度下降而不是进化算法，因此效率更高。它还为敌方脆弱性问题提供了一个新的统一视角。在MNIST和CIFAR-10数据集上的实验结果表明，我们的攻击在欺骗深度神经网络方面是相当有效的。代码可在https://github.com/aliborji/NKE.上找到



## **15. MOVE: Effective and Harmless Ownership Verification via Embedded External Features**

Move：通过嵌入式外部功能进行有效、无害的所有权验证 cs.CR

15 pages. The journal extension of our conference paper in AAAI 2022  (https://ojs.aaai.org/index.php/AAAI/article/view/20036). arXiv admin note:  substantial text overlap with arXiv:2112.03476

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02820v1)

**Authors**: Yiming Li, Linghui Zhu, Xiaojun Jia, Yang Bai, Yong Jiang, Shu-Tao Xia, Xiaochun Cao

**Abstracts**: Currently, deep neural networks (DNNs) are widely adopted in different applications. Despite its commercial values, training a well-performed DNN is resource-consuming. Accordingly, the well-trained model is valuable intellectual property for its owner. However, recent studies revealed the threats of model stealing, where the adversaries can obtain a function-similar copy of the victim model, even when they can only query the model. In this paper, we propose an effective and harmless model ownership verification (MOVE) to defend against different types of model stealing simultaneously, without introducing new security risks. In general, we conduct the ownership verification by verifying whether a suspicious model contains the knowledge of defender-specified external features. Specifically, we embed the external features by tempering a few training samples with style transfer. We then train a meta-classifier to determine whether a model is stolen from the victim. This approach is inspired by the understanding that the stolen models should contain the knowledge of features learned by the victim model. In particular, we develop our MOVE method under both white-box and black-box settings to provide comprehensive model protection. Extensive experiments on benchmark datasets verify the effectiveness of our method and its resistance to potential adaptive attacks. The codes for reproducing the main experiments of our method are available at \url{https://github.com/THUYimingLi/MOVE}.

摘要: 目前，深度神经网络(DNN)被广泛应用于不同的领域。尽管它具有商业价值，但培训一名表现良好的DNN是耗费资源的。因此，训练有素的车型对其所有者来说是宝贵的知识产权。然而，最近的研究揭示了模型窃取的威胁，其中攻击者可以获得与受害者模型功能相似的副本，即使他们只能查询模型。在本文中，我们提出了一种有效且无害的模型所有权验证(MOVE)来同时防御不同类型的模型窃取，而不会引入新的安全风险。一般来说，我们通过验证可疑模型是否包含防御者指定的外部特征的知识来进行所有权验证。具体地说，我们通过使用风格转移来回火一些训练样本来嵌入外部特征。然后，我们训练元分类器来确定模型是否从受害者那里被盗。这种方法的灵感来自于这样一种理解，即被盗模型应该包含受害者模型学习的特征知识。特别是，我们在白盒和黑盒设置下开发了移动方法，以提供全面的模型保护。在基准数据集上的大量实验验证了该方法的有效性和对潜在自适应攻击的抵抗力。复制我们方法的主要实验的代码可在URL{https://github.com/THUYimingLi/MOVE}.



## **16. Deep VULMAN: A Deep Reinforcement Learning-Enabled Cyber Vulnerability Management Framework**

Deep VULMAN：一种深度强化学习的网络漏洞管理框架 cs.AI

12 pages, 3 figures

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.02369v1)

**Authors**: Soumyadeep Hore, Ankit Shah, Nathaniel D. Bastian

**Abstracts**: Cyber vulnerability management is a critical function of a cybersecurity operations center (CSOC) that helps protect organizations against cyber-attacks on their computer and network systems. Adversaries hold an asymmetric advantage over the CSOC, as the number of deficiencies in these systems is increasing at a significantly higher rate compared to the expansion rate of the security teams to mitigate them in a resource-constrained environment. The current approaches are deterministic and one-time decision-making methods, which do not consider future uncertainties when prioritizing and selecting vulnerabilities for mitigation. These approaches are also constrained by the sub-optimal distribution of resources, providing no flexibility to adjust their response to fluctuations in vulnerability arrivals. We propose a novel framework, Deep VULMAN, consisting of a deep reinforcement learning agent and an integer programming method to fill this gap in the cyber vulnerability management process. Our sequential decision-making framework, first, determines the near-optimal amount of resources to be allocated for mitigation under uncertainty for a given system state and then determines the optimal set of prioritized vulnerability instances for mitigation. Our proposed framework outperforms the current methods in prioritizing the selection of important organization-specific vulnerabilities, on both simulated and real-world vulnerability data, observed over a one-year period.

摘要: 网络漏洞管理是网络安全运营中心(CSOC)的一项重要职能，有助于保护组织免受对其计算机和网络系统的网络攻击。与CSOC相比，对手拥有不对称的优势，因为与安全团队的扩张率相比，这些系统中的缺陷数量正在以显著更高的速度增加，以在资源受限的环境中缓解这些缺陷。目前的方法是确定性和一次性决策方法，在确定和选择要缓解的脆弱性时，不考虑未来的不确定性。这些办法还受到资源分配次优的限制，无法灵活地调整其对脆弱抵达人数波动的反应。我们提出了一种新的框架--Deep VULMAN，它由深度强化学习代理和整数规划方法组成，以填补网络漏洞管理过程中的这一空白。我们的顺序决策框架首先确定在给定系统状态下的不确定性情况下为缓解而分配的接近最优的资源量，然后确定用于缓解的最优优先级漏洞实例集。我们提出的框架在优先选择重要的特定于组织的漏洞方面优于目前的方法，该方法基于模拟和真实世界的漏洞数据，在一年的时间内观察到。



## **17. Membership Inference Attacks and Defenses in Neural Network Pruning**

神经网络修剪中的隶属度推理攻击与防御 cs.CR

This paper has been accepted to USENIX Security Symposium 2022. This  is an extended version with more experimental results

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2202.03335v2)

**Authors**: Xiaoyong Yuan, Lan Zhang

**Abstracts**: Neural network pruning has been an essential technique to reduce the computation and memory requirements for using deep neural networks for resource-constrained devices. Most existing research focuses primarily on balancing the sparsity and accuracy of a pruned neural network by strategically removing insignificant parameters and retraining the pruned model. Such efforts on reusing training samples pose serious privacy risks due to increased memorization, which, however, has not been investigated yet.   In this paper, we conduct the first analysis of privacy risks in neural network pruning. Specifically, we investigate the impacts of neural network pruning on training data privacy, i.e., membership inference attacks. We first explore the impact of neural network pruning on prediction divergence, where the pruning process disproportionately affects the pruned model's behavior for members and non-members. Meanwhile, the influence of divergence even varies among different classes in a fine-grained manner. Enlighten by such divergence, we proposed a self-attention membership inference attack against the pruned neural networks. Extensive experiments are conducted to rigorously evaluate the privacy impacts of different pruning approaches, sparsity levels, and adversary knowledge. The proposed attack shows the higher attack performance on the pruned models when compared with eight existing membership inference attacks. In addition, we propose a new defense mechanism to protect the pruning process by mitigating the prediction divergence based on KL-divergence distance, whose effectiveness has been experimentally demonstrated to effectively mitigate the privacy risks while maintaining the sparsity and accuracy of the pruned models.

摘要: 对于资源受限的设备，为了减少对深层神经网络的计算和存储需求，神经网络修剪已经成为一项基本技术。现有的大多数研究主要集中在通过有策略地去除无关紧要的参数和重新训练修剪的模型来平衡修剪神经网络的稀疏性和准确性。这种重复使用训练样本的努力由于增加了记忆而带来了严重的隐私风险，然而，这一点尚未得到调查。本文首先对神经网络修剪中的隐私风险进行了分析。具体地说，我们研究了神经网络剪枝对训练数据隐私的影响，即成员推理攻击。我们首先探讨了神经网络修剪对预测发散的影响，其中修剪过程不成比例地影响修剪后的模型对成员和非成员的行为。同时，分歧的影响甚至在不同的阶层之间以一种细粒度的方式存在差异。受这种分歧的启发，我们提出了一种针对修剪后的神经网络的自注意成员推理攻击。进行了大量的实验，以严格评估不同的剪枝方法、稀疏程度和敌意知识对隐私的影响。与现有的8种成员关系推理攻击相比，该攻击在剪枝模型上表现出更高的攻击性能。此外，我们提出了一种新的防御机制来保护剪枝过程，通过减少基于KL-发散距离的预测发散来保护剪枝过程，实验证明该机制在保持剪枝模型的稀疏性和准确性的同时有效地缓解了隐私风险。



## **18. Design of secure and robust cognitive system for malware detection**

一种安全健壮的恶意软件检测认知系统设计 cs.CR

arXiv admin note: substantial text overlap with arXiv:2104.06652

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.02310v1)

**Authors**: Sanket Shukla

**Abstracts**: Machine learning based malware detection techniques rely on grayscale images of malware and tends to classify malware based on the distribution of textures in graycale images. Albeit the advancement and promising results shown by machine learning techniques, attackers can exploit the vulnerabilities by generating adversarial samples. Adversarial samples are generated by intelligently crafting and adding perturbations to the input samples. There exists majority of the software based adversarial attacks and defenses. To defend against the adversaries, the existing malware detection based on machine learning and grayscale images needs a preprocessing for the adversarial data. This can cause an additional overhead and can prolong the real-time malware detection. So, as an alternative to this, we explore RRAM (Resistive Random Access Memory) based defense against adversaries. Therefore, the aim of this thesis is to address the above mentioned critical system security issues. The above mentioned challenges are addressed by demonstrating proposed techniques to design a secure and robust cognitive system. First, a novel technique to detect stealthy malware is proposed. The technique uses malware binary images and then extract different features from the same and then employ different ML-classifiers on the dataset thus obtained. Results demonstrate that this technique is successful in differentiating classes of malware based on the features extracted. Secondly, I demonstrate the effects of adversarial attacks on a reconfigurable RRAM-neuromorphic architecture with different learning algorithms and device characteristics. I also propose an integrated solution for mitigating the effects of the adversarial attack using the reconfigurable RRAM architecture.

摘要: 基于机器学习的恶意软件检测技术依赖于恶意软件的灰度图像，并倾向于根据灰度图像中纹理的分布对恶意软件进行分类。尽管机器学习技术具有先进性和可喜的结果，但攻击者可以通过生成敌意样本来利用这些漏洞。敌意样本是通过智能地制作并向输入样本添加扰动来生成的。存在大多数基于软件的对抗性攻击和防御。为了防御恶意软件攻击，现有的基于机器学习和灰度图像的恶意软件检测方法需要对恶意数据进行预处理。这可能会导致额外的开销，并会延长实时恶意软件检测的时间。因此，作为一种替代方案，我们探索了基于RRAM(电阻随机存取存储器)的攻击防御。因此，本文的研究目的就是解决上述关键的系统安全问题。上述挑战通过演示设计安全和健壮的认知系统的拟议技术来解决。首先，提出了一种检测隐形恶意软件的新技术。该技术使用恶意软件二值图像，然后从相同的二值图像中提取不同的特征，然后对得到的数据集使用不同的ML分类器。实验结果表明，该方法能够很好地根据提取的特征区分恶意软件的类别。其次，论证了对抗性攻击对具有不同学习算法和设备特征的可重构RRAM-神经形态结构的影响。我还提出了一种使用可重构RRAM体系结构来缓解敌意攻击影响的集成解决方案。



## **19. Generating Image Adversarial Examples by Embedding Digital Watermarks**

嵌入数字水印生成图像对抗性实例 cs.CV

10 pages, 4 figures

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2009.05107v2)

**Authors**: Yuexin Xiang, Tiantian Li, Wei Ren, Tianqing Zhu, Kim-Kwang Raymond Choo

**Abstracts**: With the increasing attention to deep neural network (DNN) models, attacks are also upcoming for such models. For example, an attacker may carefully construct images in specific ways (also referred to as adversarial examples) aiming to mislead the DNN models to output incorrect classification results. Similarly, many efforts are proposed to detect and mitigate adversarial examples, usually for certain dedicated attacks. In this paper, we propose a novel digital watermark-based method to generate image adversarial examples to fool DNN models. Specifically, partial main features of the watermark image are embedded into the host image almost invisibly, aiming to tamper with and damage the recognition capabilities of the DNN models. We devise an efficient mechanism to select host images and watermark images and utilize the improved discrete wavelet transform (DWT) based Patchwork watermarking algorithm with a set of valid hyperparameters to embed digital watermarks from the watermark image dataset into original images for generating image adversarial examples. The experimental results illustrate that the attack success rate on common DNN models can reach an average of 95.47% on the CIFAR-10 dataset and the highest at 98.71%. Besides, our scheme is able to generate a large number of adversarial examples efficiently, concretely, an average of 1.17 seconds for completing the attacks on each image on the CIFAR-10 dataset. In addition, we design a baseline experiment using the watermark images generated by Gaussian noise as the watermark image dataset that also displays the effectiveness of our scheme. Similarly, we also propose the modified discrete cosine transform (DCT) based Patchwork watermarking algorithm. To ensure repeatability and reproducibility, the source code is available on GitHub.

摘要: 随着深度神经网络(DNN)模型受到越来越多的关注，针对这类模型的攻击也随之而来。例如，攻击者可能会以特定的方式仔细构建图像(也称为对抗性示例)，目的是误导DNN模型输出错误的分类结果。同样，提出了许多努力来检测和减轻敌意示例，通常是针对某些特定的专用攻击。本文提出了一种新的基于数字水印的生成图像对抗性实例的方法来欺骗DNN模型。具体地说，水印图像的部分主要特征被嵌入到宿主图像中，几乎是不可见的，目的是篡改和破坏DNN模型的识别能力。我们设计了一种有效的选择宿主图像和水印图像的机制，并利用基于改进的离散小波变换(DWT)的拼接水印算法和一组有效的超参数来将水印图像数据集中的数字水印嵌入到原始图像中，以生成图像对抗性示例。实验结果表明，在CIFAR-10数据集上，常用DNN模型的攻击成功率平均可达95.47%，最高可达98.71%。此外，我们的方案能够高效地生成大量的对抗性实例，具体而言，完成对CIFAR-10数据集上的每幅图像的攻击平均需要1.17秒。此外，利用高斯噪声产生的水印图像作为水印图像数据集，设计了一个基线实验，验证了该算法的有效性。同样，我们还提出了基于修正离散余弦变换(DCT)的补丁水印算法。为了确保可重复性和再现性，GitHub上提供了源代码。



## **20. Abusing Commodity DRAMs in IoT Devices to Remotely Spy on Temperature**

在物联网设备中滥用商品DRAM远程监视温度 cs.CR

Submitted to IEEE TIFS and currently under review

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.02125v1)

**Authors**: Florian Frank, Wenjie Xiong, Nikolaos Athanasios Anagnostopoulos, André Schaller, Tolga Arul, Farinaz Koushanfar, Stefan Katzenbeisser, Ulrich Ruhrmair, Jakub Szefer

**Abstracts**: The ubiquity and pervasiveness of modern Internet of Things (IoT) devices opens up vast possibilities for novel applications, but simultaneously also allows spying on, and collecting data from, unsuspecting users to a previously unseen extent. This paper details a new attack form in this vein, in which the decay properties of widespread, off-the-shelf DRAM modules are exploited to accurately sense the temperature in the vicinity of the DRAM-carrying device. Among others, this enables adversaries to remotely and purely digitally spy on personal behavior in users' private homes, or to collect security-critical data in server farms, cloud storage centers, or commercial production lines. We demonstrate that our attack can be performed by merely compromising the software of an IoT device and does not require hardware modifications or physical access at attack time. It can achieve temperature resolutions of up to 0.5{\deg}C over a range of 0{\deg}C to 70{\deg}C in practice. Perhaps most interestingly, it even works in devices that do not have a dedicated temperature sensor on board. To complete our work, we discuss practical attack scenarios as well as possible countermeasures against our temperature espionage attacks.

摘要: 现代物联网(IoT)设备的无处不在和无处不在，为新的应用打开了巨大的可能性，但同时也允许对毫无戒心的用户进行间谍活动，并从他们那里收集数据，达到前所未有的程度。本文详细介绍了一种新的攻击形式，利用广泛存在的现成DRAM模块的衰减特性来准确检测DRAM携带设备附近的温度。其中，这使攻击者能够远程、纯数字地监视用户私人住宅中的个人行为，或者收集服务器群、云存储中心或商业生产线中的安全关键数据。我们证明，我们的攻击可以仅通过危害物联网设备的软件来执行，并且在攻击时不需要修改硬件或进行物理访问。实际应用表明，在0~70℃的温度范围内，温度分辨率最高可达0.5℃。也许最有趣的是，它甚至可以在没有专用温度传感器的设备上工作。为了完成我们的工作，我们讨论了实际的攻击方案以及针对我们的温度间谍攻击的可能对策。



## **21. Local Differential Privacy for Federated Learning**

联合学习中的局部差分隐私 cs.CR

17 pages

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2202.06053v2)

**Authors**: M. A. P. Chamikara, Dongxi Liu, Seyit Camtepe, Surya Nepal, Marthie Grobler, Peter Bertok, Ibrahim Khalil

**Abstracts**: Advanced adversarial attacks such as membership inference and model memorization can make federated learning (FL) vulnerable and potentially leak sensitive private data. Local differentially private (LDP) approaches are gaining more popularity due to stronger privacy notions and native support for data distribution compared to other differentially private (DP) solutions. However, DP approaches assume that the FL server (that aggregates the models) is honest (run the FL protocol honestly) or semi-honest (run the FL protocol honestly while also trying to learn as much information as possible). These assumptions make such approaches unrealistic and unreliable for real-world settings. Besides, in real-world industrial environments (e.g., healthcare), the distributed entities (e.g., hospitals) are already composed of locally running machine learning models (this setting is also referred to as the cross-silo setting). Existing approaches do not provide a scalable mechanism for privacy-preserving FL to be utilized under such settings, potentially with untrusted parties. This paper proposes a new local differentially private FL (named LDPFL) protocol for industrial settings. LDPFL can run in industrial settings with untrusted entities while enforcing stronger privacy guarantees than existing approaches. LDPFL shows high FL model performance (up to 98%) under small privacy budgets (e.g., epsilon = 0.5) in comparison to existing methods.

摘要: 高级对抗性攻击，如成员推理和模型记忆，会使联邦学习(FL)容易受到攻击，并可能泄露敏感的私人数据。与其他差异私有(DP)解决方案相比，本地差异私有(LDP)方法由于更强的隐私概念和对数据分发的本地支持而越来越受欢迎。然而，DP方法假设FL服务器(聚集模型)是诚实的(诚实地运行FL协议)或半诚实的(诚实地运行FL协议，同时还试图了解尽可能多的信息)。这些假设使得这种方法对于现实世界的设置来说是不现实和不可靠的。此外，在真实世界的工业环境(例如，医疗保健)中，分布式实体(例如，医院)已经由本地运行的机器学习模型组成(该设置也被称为跨竖井设置)。现有方法没有提供用于保护隐私的FL的可扩展机制以在这样的设置下使用，可能与不可信方一起使用。提出了一种适用于工业环境的局部差分私有FL协议(简称LDPFL)。LDPFL可以在具有不可信实体的工业环境中运行，同时执行比现有方法更强大的隐私保障。与现有方法相比，LDPFL在较小的隐私预算(例如，epsilon=0.5)下表现出高的FL模型性能(高达98%)。



## **22. SAC-AP: Soft Actor Critic based Deep Reinforcement Learning for Alert Prioritization**

SAC-AP：基于软参与者批评者的深度强化学习告警优先级 cs.CR

8 pages, 8 figures, IEEE WORLD CONGRESS ON COMPUTATIONAL INTELLIGENCE  2022

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2207.13666v3)

**Authors**: Lalitha Chavali, Tanay Gupta, Paresh Saxena

**Abstracts**: Intrusion detection systems (IDS) generate a large number of false alerts which makes it difficult to inspect true positives. Hence, alert prioritization plays a crucial role in deciding which alerts to investigate from an enormous number of alerts that are generated by IDS. Recently, deep reinforcement learning (DRL) based deep deterministic policy gradient (DDPG) off-policy method has shown to achieve better results for alert prioritization as compared to other state-of-the-art methods. However, DDPG is prone to the problem of overfitting. Additionally, it also has a poor exploration capability and hence it is not suitable for problems with a stochastic environment. To address these limitations, we present a soft actor-critic based DRL algorithm for alert prioritization (SAC-AP), an off-policy method, based on the maximum entropy reinforcement learning framework that aims to maximize the expected reward while also maximizing the entropy. Further, the interaction between an adversary and a defender is modeled as a zero-sum game and a double oracle framework is utilized to obtain the approximate mixed strategy Nash equilibrium (MSNE). SAC-AP finds robust alert investigation policies and computes pure strategy best response against opponent's mixed strategy. We present the overall design of SAC-AP and evaluate its performance as compared to other state-of-the art alert prioritization methods. We consider defender's loss, i.e., the defender's inability to investigate the alerts that are triggered due to attacks, as the performance metric. Our results show that SAC-AP achieves up to 30% decrease in defender's loss as compared to the DDPG based alert prioritization method and hence provides better protection against intrusions. Moreover, the benefits are even higher when SAC-AP is compared to other traditional alert prioritization methods including Uniform, GAIN, RIO and Suricata.

摘要: 入侵检测系统(入侵检测系统)产生大量的错误警报，使得对真实阳性的检测变得困难。因此，警报优先级在决定从由入侵检测系统生成的大量警报中调查哪些警报时起着至关重要的作用。近年来，与其他方法相比，基于深度强化学习(DRL)的深度确定性策略梯度(DDPG)非策略方法能够获得更好的告警优先级排序结果。然而，DDPG容易出现过度匹配的问题。此外，它的探测能力也很差，因此不适合于具有随机环境的问题。针对这些局限性，我们提出了一种基于软参与者-批评者的DRL警报优先排序算法(SAC-AP)，这是一种基于最大熵强化学习框架的非策略方法，旨在最大化期望回报的同时最大化熵。在此基础上，将对手和防御者之间的相互作用建模为零和博弈，并利用双预言框架得到近似的混合策略纳什均衡。SAC-AP发现稳健的警戒调查策略，并针对对手的混合策略计算纯策略的最佳响应。我们介绍了SAC-AP的总体设计，并与其他最先进的警报优先排序方法进行了比较，评估了其性能。我们将防御者的损失，即防御者无法调查由于攻击而触发的警报作为性能指标。结果表明，与基于DDPG的告警优先级排序方法相比，SAC-AP可以减少高达30%的防御者损失，从而提供更好的防御入侵保护。此外，当SAC-AP与其他传统的警报优先排序方法(包括Uniform、Gain、Rio和Suricata)相比时，好处甚至更高。



## **23. Spectrum Focused Frequency Adversarial Attacks for Automatic Modulation Classification**

用于自动调制分类的频谱聚焦频率对抗攻击 cs.CR

6 pages, 9 figures

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.01919v1)

**Authors**: Sicheng Zhang, Jiarun Yu, Zhida Bao, Shiwen Mao, Yun Lin

**Abstracts**: Artificial intelligence (AI) technology has provided a potential solution for automatic modulation recognition (AMC). Unfortunately, AI-based AMC models are vulnerable to adversarial examples, which seriously threatens the efficient, secure and trusted application of AI in AMC. This issue has attracted the attention of researchers. Various studies on adversarial attacks and defenses evolve in a spiral. However, the existing adversarial attack methods are all designed in the time domain. They introduce more high-frequency components in the frequency domain, due to abrupt updates in the time domain. For this issue, from the perspective of frequency domain, we propose a spectrum focused frequency adversarial attacks (SFFAA) for AMC model, and further draw on the idea of meta-learning, propose a Meta-SFFAA algorithm to improve the transferability in the black-box attacks. Extensive experiments, qualitative and quantitative metrics demonstrate that the proposed algorithm can concentrate the adversarial energy on the spectrum where the signal is located, significantly improve the adversarial attack performance while maintaining the concealment in the frequency domain.

摘要: 人工智能(AI)技术为自动调制识别(AMC)提供了一种潜在的解决方案。不幸的是，基于人工智能的AMC模型容易受到敌意例子的攻击，这严重威胁了人工智能在AMC中的高效、安全和可信的应用。这个问题已经引起了研究人员的关注。关于对抗性攻击和防御的各种研究呈螺旋式发展。然而，现有的对抗性攻击方法都是在时间域设计的。由于时间域中的突然更新，它们在频域中引入了更多的高频分量。针对这一问题，从频域的角度出发，提出了一种针对AMC模型的频谱聚焦频率对抗攻击算法(SFFAA)，并进一步借鉴元学习的思想，提出了一种Meta-SFFAA算法来提高黑盒攻击的可转移性。大量的实验、定性和定量指标表明，该算法可以将对抗能量集中在信号所在的频谱上，在保持频域隐蔽性的同时，显著提高了对抗攻击的性能。



## **24. On the Evaluation of User Privacy in Deep Neural Networks using Timing Side Channel**

基于时序侧通道的深度神经网络用户隐私评估研究 cs.CR

15 pages, 20 figures

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.01113v2)

**Authors**: Shubhi Shukla, Manaar Alam, Sarani Bhattacharya, Debdeep Mukhopadhyay, Pabitra Mitra

**Abstracts**: Recent Deep Learning (DL) advancements in solving complex real-world tasks have led to its widespread adoption in practical applications. However, this opportunity comes with significant underlying risks, as many of these models rely on privacy-sensitive data for training in a variety of applications, making them an overly-exposed threat surface for privacy violations. Furthermore, the widespread use of cloud-based Machine-Learning-as-a-Service (MLaaS) for its robust infrastructure support has broadened the threat surface to include a variety of remote side-channel attacks. In this paper, we first identify and report a novel data-dependent timing side-channel leakage (termed Class Leakage) in DL implementations originating from non-constant time branching operation in a widely used DL framework PyTorch. We further demonstrate a practical inference-time attack where an adversary with user privilege and hard-label black-box access to an MLaaS can exploit Class Leakage to compromise the privacy of MLaaS users. DL models are vulnerable to Membership Inference Attack (MIA), where an adversary's objective is to deduce whether any particular data has been used while training the model. In this paper, as a separate case study, we demonstrate that a DL model secured with differential privacy (a popular countermeasure against MIA) is still vulnerable to MIA against an adversary exploiting Class Leakage. We develop an easy-to-implement countermeasure by making a constant-time branching operation that alleviates the Class Leakage and also aids in mitigating MIA. We have chosen two standard benchmarking image classification datasets, CIFAR-10 and CIFAR-100 to train five state-of-the-art pre-trained DL models, over two different computing environments having Intel Xeon and Intel i7 processors to validate our approach.

摘要: 最近深度学习(DL)在解决复杂现实世界任务方面的进步导致了它在实际应用中的广泛采用。然而，这种机会伴随着巨大的潜在风险，因为这些模型中的许多依赖于隐私敏感数据来进行各种应用程序的培训，使它们成为侵犯隐私的过度暴露的威胁表面。此外，基于云的机器学习即服务(MLaaS)因其强大的基础设施支持而广泛使用，扩大了威胁面，包括各种远程侧通道攻击。在这篇文章中，我们首先识别和报告了一种新的数据相关的定时侧通道泄漏(称为类泄漏)，该泄漏是由广泛使用的动态链接库框架中的非常数时间分支操作引起的。我们进一步展示了一个实用的推理时间攻击，其中具有用户权限和硬标签黑盒访问MLaaS的攻击者可以利用类泄漏来危害MLaaS用户的隐私。DL模型容易受到成员推理攻击(MIA)，对手的目标是推断在训练模型时是否使用了特定的数据。在本文中，作为一个单独的案例研究，我们证明了在差异隐私保护下的DL模型(一种流行的针对MIA的对策)仍然容易受到MIA对利用类泄漏的攻击者的攻击。我们开发了一种易于实现的对策，通过进行恒定时间分支操作来缓解类泄漏，并帮助缓解MIA。我们选择了两个标准的基准图像分类数据集，CIFAR-10和CIFAR-100来训练五个最先进的预训练的DL模型，在两种不同的计算环境中使用Intel Xeon和Intel i7处理器来验证我们的方法。



## **25. Adversarial Attacks on ASR Systems: An Overview**

对ASR系统的敌意攻击：综述 cs.SD

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.02250v1)

**Authors**: Xiao Zhang, Hao Tan, Xuan Huang, Denghui Zhang, Keke Tang, Zhaoquan Gu

**Abstracts**: With the development of hardware and algorithms, ASR(Automatic Speech Recognition) systems evolve a lot. As The models get simpler, the difficulty of development and deployment become easier, ASR systems are getting closer to our life. On the one hand, we often use APPs or APIs of ASR to generate subtitles and record meetings. On the other hand, smart speaker and self-driving car rely on ASR systems to control AIoT devices. In past few years, there are a lot of works on adversarial examples attacks against ASR systems. By adding a small perturbation to the waveforms, the recognition results make a big difference. In this paper, we describe the development of ASR system, different assumptions of attacks, and how to evaluate these attacks. Next, we introduce the current works on adversarial examples attacks from two attack assumptions: white-box attack and black-box attack. Different from other surveys, we pay more attention to which layer they perturb waveforms in ASR system, the relationship between these attacks, and their implementation methods. We focus on the effect of their works.

摘要: 随着硬件和算法的发展，ASR(Automatic Speech Recognition，自动语音识别)系统也在不断发展。随着模型变得更简单，开发和部署的难度变得更容易，ASR系统越来越接近我们的生活。一方面，我们经常使用ASR的APP或API来生成字幕和录制会议。另一方面，智能音箱和自动驾驶汽车依靠ASR系统来控制AIoT设备。在过去的几年里，已经有很多关于针对ASR系统的对抗性例子攻击的工作。通过对波形添加小的扰动，识别结果有很大的不同。在本文中，我们描述了ASR系统的发展，不同的攻击假设，以及如何评估这些攻击。接下来，我们从白盒攻击和黑盒攻击两个攻击假设出发，介绍了目前对抗性例子攻击的研究成果。与其他研究不同的是，我们更关注它们对ASR系统中的哪一层波形的扰动，这些攻击之间的关系，以及它们的实现方法。我们关注的是他们作品的效果。



## **26. Robust Graph Neural Networks using Weighted Graph Laplacian**

基于加权图拉普拉斯的稳健图神经网络 cs.LG

Accepted at IEEE International Conference on Signal Processing and  Communications (SPCOM), 2022

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.01853v1)

**Authors**: Bharat Runwal, Vivek, Sandeep Kumar

**Abstracts**: Graph neural network (GNN) is achieving remarkable performances in a variety of application domains. However, GNN is vulnerable to noise and adversarial attacks in input data. Making GNN robust against noises and adversarial attacks is an important problem. The existing defense methods for GNNs are computationally demanding and are not scalable. In this paper, we propose a generic framework for robustifying GNN known as Weighted Laplacian GNN (RWL-GNN). The method combines Weighted Graph Laplacian learning with the GNN implementation. The proposed method benefits from the positive semi-definiteness property of Laplacian matrix, feature smoothness, and latent features via formulating a unified optimization framework, which ensures the adversarial/noisy edges are discarded and connections in the graph are appropriately weighted. For demonstration, the experiments are conducted with Graph convolutional neural network(GCNN) architecture, however, the proposed framework is easily amenable to any existing GNN architecture. The simulation results with benchmark dataset establish the efficacy of the proposed method, both in accuracy and computational efficiency. Code can be accessed at https://github.com/Bharat-Runwal/RWL-GNN.

摘要: 图形神经网络(GNN)在各种应用领域都取得了令人瞩目的成绩。然而，GNN很容易受到输入数据中的噪声和对抗性攻击。如何使GNN对噪声和敌意攻击具有健壮性是一个重要的问题。现有的GNN防御方法计算量大且不可扩展。在本文中，我们提出了一种称为加权拉普拉斯GNN(RWL-GNN)的通用GNN框架。该方法将加权图拉普拉斯学习与GNN实现相结合。该方法充分利用了拉普拉斯矩阵的正半定性、特征的光滑性和潜在特征，建立了统一的优化框架，确保了对敌边/噪声边的丢弃和图中连接的适当加权。为了进行演示，实验使用了图卷积神经网络(GCNN)结构，然而，所提出的框架可以很容易地服从于任何现有的GNN结构。利用基准数据集的仿真结果验证了该方法在精度和计算效率上的有效性。代码可在https://github.com/Bharat-Runwal/RWL-GNN.上访问



## **27. Multiclass ASMA vs Targeted PGD Attack in Image Segmentation**

图像分割中多类ASMA与靶向PGD攻击的比较 cs.CV

10 pages, 6 figures

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.01844v1)

**Authors**: Johnson Vo, Jiabao Xie, Sahil Patel

**Abstracts**: Deep learning networks have demonstrated high performance in a large variety of applications, such as image classification, speech recognition, and natural language processing. However, there exists a major vulnerability exploited by the use of adversarial attacks. An adversarial attack imputes images by altering the input image very slightly, making it nearly undetectable to the naked eye, but results in a very different classification by the network. This paper explores the projected gradient descent (PGD) attack and the Adaptive Mask Segmentation Attack (ASMA) on the image segmentation DeepLabV3 model using two types of architectures: MobileNetV3 and ResNet50, It was found that PGD was very consistent in changing the segmentation to be its target while the generalization of ASMA to a multiclass target was not as effective. The existence of such attack however puts all of image classification deep learning networks in danger of exploitation.

摘要: 深度学习网络在图像分类、语音识别、自然语言处理等多种应用中表现出了很高的性能。然而，存在一个通过使用对抗性攻击来利用的重大漏洞。敌意攻击通过非常轻微地更改输入图像来计算图像，使其几乎无法被肉眼检测到，但会导致网络进行非常不同的分类。利用两种结构：MobileNetV3和ResNet50，对DeepLabV3图像分割模型进行了投影梯度下降(PGD)攻击和自适应掩码分割攻击(ASMA)的研究，发现PGD在改变分割为其目标方面具有很好的一致性，而ASMA对多类目标的泛化效果不佳。然而，这种攻击的存在使所有的图像分类深度学习网络都处于被利用的危险之中。



## **28. Adversarial Camouflage for Node Injection Attack on Graphs**

图上节点注入攻击的对抗性伪装 cs.LG

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.01819v1)

**Authors**: Shuchang Tao, Qi Cao, Huawei Shen, Yunfan Wu, Liang Hou, Xueqi Cheng

**Abstracts**: Node injection attacks against Graph Neural Networks (GNNs) have received emerging attention as a practical attack scenario, where the attacker injects malicious nodes instead of modifying node features or edges to degrade the performance of GNNs. Despite the initial success of node injection attacks, we find that the injected nodes by existing methods are easy to be distinguished from the original normal nodes by defense methods and limiting their attack performance in practice. To solve the above issues, we devote to camouflage node injection attack, i.e., camouflaging injected malicious nodes (structure/attributes) as the normal ones that appear legitimate/imperceptible to defense methods. The non-Euclidean nature of graph data and the lack of human prior brings great challenges to the formalization, implementation, and evaluation of camouflage on graphs. In this paper, we first propose and formulate the camouflage of injected nodes from both the fidelity and diversity of the ego networks centered around injected nodes. Then, we design an adversarial CAmouflage framework for Node injection Attack, namely CANA, to improve the camouflage while ensuring the attack performance. Several novel indicators for graph camouflage are further designed for a comprehensive evaluation. Experimental results demonstrate that when equipping existing node injection attack methods with our proposed CANA framework, the attack performance against defense methods as well as node camouflage is significantly improved.

摘要: 针对图神经网络的节点注入攻击作为一种实用的攻击场景受到了越来越多的关注，即攻击者注入恶意节点而不是修改节点特征或边来降低图神经网络的性能。尽管节点注入攻击取得了初步的成功，但我们发现，现有方法注入的节点很容易通过防御方法与原来的正常节点区分开来，限制了它们在实践中的攻击性能。为了解决上述问题，我们致力于伪装节点注入攻击，即伪装注入的恶意节点(结构/属性)作为正常的合法/不可察觉的防御方法。图数据的非欧几里得性质和人类先验知识的缺乏给图上伪装的形式化、实现和评估带来了巨大的挑战。本文首先从以注入节点为中心的EGO网络的保真度和多样性两个方面提出并构造了注入节点的伪装。然后，设计了一种节点注入攻击的对抗性伪装框架CANA，在保证攻击性能的同时提高伪装性能。进一步设计了几种新的图形伪装指标，进行了综合评价。实验结果表明，在现有的节点注入攻击方法中加入CANA框架后，对防御方法的攻击性能以及对节点伪装的攻击性能都得到了显著提高。



## **29. Success of Uncertainty-Aware Deep Models Depends on Data Manifold Geometry**

不确定性感知深度模型的成功依赖于数据流形几何 cs.LG

9 pages

**SubmitDate**: 2022-08-02    [paper-pdf](http://arxiv.org/pdf/2208.01705v1)

**Authors**: Mark Penrod, Harrison Termotto, Varshini Reddy, Jiayu Yao, Finale Doshi-Velez, Weiwei Pan

**Abstracts**: For responsible decision making in safety-critical settings, machine learning models must effectively detect and process edge-case data. Although existing works show that predictive uncertainty is useful for these tasks, it is not evident from literature which uncertainty-aware models are best suited for a given dataset. Thus, we compare six uncertainty-aware deep learning models on a set of edge-case tasks: robustness to adversarial attacks as well as out-of-distribution and adversarial detection. We find that the geometry of the data sub-manifold is an important factor in determining the success of various models. Our finding suggests an interesting direction in the study of uncertainty-aware deep learning models.

摘要: 为了在安全关键环境中做出负责任的决策，机器学习模型必须有效地检测和处理边缘案例数据。虽然现有的工作表明，预测不确定性对这些任务是有用的，但从文献中并不明显地看到，哪些不确定性感知模型最适合给定的数据集。因此，我们在一组边缘情况任务上比较了六种不确定性感知的深度学习模型：对对手攻击的健壮性以及分布外和对抗性检测。我们发现，数据子流形的几何形状是决定各种模型成功与否的重要因素。我们的发现为不确定性感知深度学习模型的研究提供了一个有趣的方向。



## **30. CAPD: A Context-Aware, Policy-Driven Framework for Secure and Resilient IoBT Operations**

CAPD：环境感知、策略驱动的IoBT安全弹性运营框架 cs.CR

**SubmitDate**: 2022-08-02    [paper-pdf](http://arxiv.org/pdf/2208.01703v1)

**Authors**: Sai Sree Laya Chukkapalli, Anupam Joshi, Tim Finin, Robert F. Erbacher

**Abstracts**: The Internet of Battlefield Things (IoBT) will advance the operational effectiveness of infantry units. However, this requires autonomous assets such as sensors, drones, combat equipment, and uncrewed vehicles to collaborate, securely share information, and be resilient to adversary attacks in contested multi-domain operations. CAPD addresses this problem by providing a context-aware, policy-driven framework supporting data and knowledge exchange among autonomous entities in a battlespace. We propose an IoBT ontology that facilitates controlled information sharing to enable semantic interoperability between systems. Its key contributions include providing a knowledge graph with a shared semantic schema, integration with background knowledge, efficient mechanisms for enforcing data consistency and drawing inferences, and supporting attribute-based access control. The sensors in the IoBT provide data that create populated knowledge graphs based on the ontology. This paper describes using CAPD to detect and mitigate adversary actions. CAPD enables situational awareness using reasoning over the sensed data and SPARQL queries. For example, adversaries can cause sensor failure or hijacking and disrupt the tactical networks to degrade video surveillance. In such instances, CAPD uses an ontology-based reasoner to see how alternative approaches can still support the mission. Depending on bandwidth availability, the reasoner initiates the creation of a reduced frame rate grayscale video by active transcoding or transmits only still images. This ability to reason over the mission sensed environment and attack context permits the autonomous IoBT system to exhibit resilience in contested conditions.

摘要: 战场物联网(IoBT)将提高步兵部队的作战效能。然而，这需要传感器、无人机、作战设备和无人驾驶车辆等自主资产进行协作，安全地共享信息，并在有争议的多领域行动中对对手攻击具有弹性。CAPD通过提供支持战场空间中自治实体之间的数据和知识交换的上下文感知、策略驱动的框架来解决这一问题。我们提出了一种IoBT本体，它促进了受控信息共享，从而实现了系统之间的语义互操作。它的主要贡献包括提供具有共享语义模式的知识图，与背景知识的集成，执行数据一致性和推理的有效机制，以及支持基于属性的访问控制。IoBT中的传感器提供基于本体创建填充的知识图的数据。本文描述了使用CAPD来检测和缓解恶意行为。CAPD通过对感测数据和SPARQL查询进行推理来实现态势感知。例如，敌手可能会导致传感器故障或劫持，并扰乱战术网络以降低视频监控。在这种情况下，CAPD使用基于本体的推理机来查看替代方法如何仍然能够支持任务。根据带宽可用性，推理器通过主动代码转换来启动降低帧速率的灰度视频的创建，或者仅传输静止图像。这种对任务感知环境和攻击上下文进行推理的能力使自主IoBT系统在竞争条件下表现出弹性。



## **31. Adversarial Detection Avoidance Attacks: Evaluating the robustness of perceptual hashing-based client-side scanning**

对抗性检测避免攻击：评估基于感知散列的客户端扫描的健壮性 cs.CR

This is a revised version of the paper published at USENIX Security  2022. We now use a semi-automated procedure to remove duplicates from the  ImageNet dataset

**SubmitDate**: 2022-08-02    [paper-pdf](http://arxiv.org/pdf/2106.09820v3)

**Authors**: Shubham Jain, Ana-Maria Cretu, Yves-Alexandre de Montjoye

**Abstracts**: End-to-end encryption (E2EE) by messaging platforms enable people to securely and privately communicate with one another. Its widespread adoption however raised concerns that illegal content might now be shared undetected. Following the global pushback against key escrow systems, client-side scanning based on perceptual hashing has been recently proposed by tech companies, governments and researchers to detect illegal content in E2EE communications. We here propose the first framework to evaluate the robustness of perceptual hashing-based client-side scanning to detection avoidance attacks and show current systems to not be robust. More specifically, we propose three adversarial attacks--a general black-box attack and two white-box attacks for discrete cosine transform-based algorithms--against perceptual hashing algorithms. In a large-scale evaluation, we show perceptual hashing-based client-side scanning mechanisms to be highly vulnerable to detection avoidance attacks in a black-box setting, with more than 99.9% of images successfully attacked while preserving the content of the image. We furthermore show our attack to generate diverse perturbations, strongly suggesting that straightforward mitigation strategies would be ineffective. Finally, we show that the larger thresholds necessary to make the attack harder would probably require more than one billion images to be flagged and decrypted daily, raising strong privacy concerns. Taken together, our results shed serious doubts on the robustness of perceptual hashing-based client-side scanning mechanisms currently proposed by governments, organizations, and researchers around the world.

摘要: 消息传递平台的端到端加密(E2EE)使人们能够安全且私密地相互通信。然而，它的广泛采用引发了人们的担忧，即非法内容现在可能被分享而不被发现。继全球对密钥托管系统的抵制之后，科技公司、政府和研究人员最近提出了基于感知散列的客户端扫描，以检测E2EE通信中的非法内容。我们在这里提出了第一个框架来评估基于感知散列的客户端扫描对检测规避攻击的健壮性，并表明当前的系统是不健壮的。更具体地说，我们针对感知散列算法提出了三种对抗性攻击--一种通用的黑盒攻击和两种基于离散余弦变换的算法的白盒攻击。在大规模的评估中，我们发现基于感知散列的客户端扫描机制在黑盒环境下非常容易受到检测回避攻击，99.9%以上的图像在保护图像内容的同时被攻击成功。此外，我们还展示了我们的攻击会产生不同的扰动，强烈表明直接的缓解策略将是无效的。最后，我们指出，提高攻击难度所需的更大门槛可能需要每天标记和解密超过10亿张图像，这引发了强烈的隐私问题。综上所述，我们的结果对目前世界各地的政府、组织和研究人员提出的基于感知散列的客户端扫描机制的健壮性提出了严重的质疑。



## **32. Quantum Lock: A Provable Quantum Communication Advantage**

量子锁：一种可证明的量子通信优势 quant-ph

Replacement of paper "Hybrid PUF: A Novel Way to Enhance the Security  of Classical PUFs" (arXiv:2110.09469)

**SubmitDate**: 2022-08-02    [paper-pdf](http://arxiv.org/pdf/2110.09469v3)

**Authors**: Kaushik Chakraborty, Mina Doosti, Yao Ma, Chirag Wadhwa, Myrto Arapinis, Elham Kashefi

**Abstracts**: Physical unclonable functions(PUFs) provide a unique fingerprint to a physical entity by exploiting the inherent physical randomness. Gao et al. discussed the vulnerability of most current-day PUFs to sophisticated machine learning-based attacks. We address this problem by integrating classical PUFs and existing quantum communication technology. Specifically, this paper proposes a generic design of provably secure PUFs, called hybrid locked PUFs(HLPUFs), providing a practical solution for securing classical PUFs. An HLPUF uses a classical PUF(CPUF), and encodes the output into non-orthogonal quantum states to hide the outcomes of the underlying CPUF from any adversary. Here we introduce a quantum lock to protect the HLPUFs from any general adversaries. The indistinguishability property of the non-orthogonal quantum states, together with the quantum lockdown technique prevents the adversary from accessing the outcome of the CPUFs. Moreover, we show that by exploiting non-classical properties of quantum states, the HLPUF allows the server to reuse the challenge-response pairs for further client authentication. This result provides an efficient solution for running PUF-based client authentication for an extended period while maintaining a small-sized challenge-response pairs database on the server side. Later, we support our theoretical contributions by instantiating the HLPUFs design using accessible real-world CPUFs. We use the optimal classical machine-learning attacks to forge both the CPUFs and HLPUFs, and we certify the security gap in our numerical simulation for construction which is ready for implementation.

摘要: 物理不可克隆函数(PUF)通过利用固有的物理随机性为物理实体提供唯一指纹。高等人。讨论了当前大多数PUF对复杂的基于机器学习的攻击的脆弱性。我们通过将经典的PUF和现有的量子通信技术相结合来解决这个问题。具体地说，本文提出了一种可证明安全的PUF的通用设计，称为混合锁定PUF(HLPUF)，为保护经典PUF提供了一种实用的解决方案。HLPUF使用经典的PUF(CPUF)，并将输出编码为非正交的量子态，以向任何对手隐藏底层CPUF的结果。在这里，我们引入量子锁来保护HLPUF免受任何一般对手的攻击。非正交量子态的不可分辨特性，加上量子锁定技术，阻止了攻击者访问CPUF的结果。此外，我们证明了通过利用量子态的非经典属性，HLPUF允许服务器重用挑战-响应对来进行进一步的客户端认证。这一结果为长期运行基于PUF的客户端身份验证提供了一个有效的解决方案，同时在服务器端维护一个小型的挑战-响应对数据库。随后，我们通过使用可访问的真实CPUF来实例化HLPUF设计来支持我们的理论贡献。我们使用最优经典机器学习攻击来伪造CPUF和HLPUF，并证明了我们的构造数值模拟中的安全漏洞。



## **33. SCFI: State Machine Control-Flow Hardening Against Fault Attacks**

SCFI：针对故障攻击的状态机控制流强化 cs.CR

**SubmitDate**: 2022-08-02    [paper-pdf](http://arxiv.org/pdf/2208.01356v1)

**Authors**: Pascal Nasahl, Martin Unterguggenberger, Rishub Nagpal, Robert Schilling, David Schrammel, Stefan Mangard

**Abstracts**: Fault injection (FI) is a powerful attack methodology allowing an adversary to entirely break the security of a target device. As finite-state machines (FSMs) are fundamental hardware building blocks responsible for controlling systems, inducing faults into these controllers enables an adversary to hijack the execution of the integrated circuit. A common defense strategy mitigating these attacks is to manually instantiate FSMs multiple times and detect faults using a majority voting logic. However, as each additional FSM instance only provides security against one additional induced fault, this approach scales poorly in a multi-fault attack scenario.   In this paper, we present SCFI: a strong, probabilistic FSM protection mechanism ensuring that control-flow deviations from the intended control-flow are detected even in the presence of multiple faults. At its core, SCFI consists of a hardened next-state function absorbing the execution history as well as the FSM's control signals to derive the next state. When either the absorbed inputs, the state registers, or the function itself are affected by faults, SCFI triggers an error with no detection latency. We integrate SCFI into a synthesis tool capable of automatically hardening arbitrary unprotected FSMs without user interaction and open-source the tool. Our evaluation shows that SCFI provides strong protection guarantees with a better area-time product than FSMs protected using classical redundancy-based approaches. Finally, we formally verify the resilience of the protected state machines using a pre-silicon fault analysis tool.

摘要: 故障注入(FI)是一种强大的攻击方法，允许对手完全破坏目标设备的安全。由于有限状态机(FSM)是负责控制系统的基本硬件构建块，因此在这些控制器中引入故障使对手能够劫持集成电路的执行。缓解这些攻击的常见防御策略是多次手动实例化FSM并使用多数投票逻辑检测故障。然而，由于每个额外的FSM实例仅针对一个额外的诱发故障提供安全性，因此该方法在多故障攻击场景中伸缩性不佳。在本文中，我们提出了SCFI：一种强大的、概率的有限状态机保护机制，确保即使在存在多个故障的情况下也能检测到控制流与预期控制流的偏差。在其核心，SCFI由一个强化的下一状态函数组成，该函数吸收执行历史以及FSM的控制信号以得出下一状态。当吸收的输入、状态寄存器或功能本身受到故障影响时，SCFI会在没有检测延迟的情况下触发错误。我们将SCFI集成到一个合成工具中，该工具能够自动硬化任意不受保护的FSM，而无需用户交互并开放该工具的源代码。我们的评估表明，SCFI提供了强大的保护保证，比使用经典的基于冗余的方法保护的FSM具有更好的区域-时间乘积。最后，我们使用预硅故障分析工具正式验证了受保护状态机的弹性。



## **34. Understanding Adversarial Robustness of Vision Transformers via Cauchy Problem**

通过柯西问题理解视觉变形器的对抗稳健性 cs.CV

Accepted by ECML-PKDD 2022

**SubmitDate**: 2022-08-01    [paper-pdf](http://arxiv.org/pdf/2208.00906v1)

**Authors**: Zheng Wang, Wenjie Ruan

**Abstracts**: Recent research on the robustness of deep learning has shown that Vision Transformers (ViTs) surpass the Convolutional Neural Networks (CNNs) under some perturbations, e.g., natural corruption, adversarial attacks, etc. Some papers argue that the superior robustness of ViT comes from the segmentation of its input images; others say that the Multi-head Self-Attention (MSA) is the key to preserving the robustness. In this paper, we aim to introduce a principled and unified theoretical framework to investigate such an argument on ViT's robustness. We first theoretically prove that, unlike Transformers in Natural Language Processing, ViTs are Lipschitz continuous. Then we theoretically analyze the adversarial robustness of ViTs from the perspective of the Cauchy Problem, via which we can quantify how the robustness propagates through layers. We demonstrate that the first and last layers are the critical factors to affect the robustness of ViTs. Furthermore, based on our theory, we empirically show that unlike the claims from existing research, MSA only contributes to the adversarial robustness of ViTs under weak adversarial attacks, e.g., FGSM, and surprisingly, MSA actually comprises the model's adversarial robustness under stronger attacks, e.g., PGD attacks.

摘要: 最近关于深度学习稳健性的研究表明，视觉转换器(VITS)在某些扰动下优于卷积神经网络(CNNS)，如自然腐败、敌意攻击等。一些文献认为VIT优越的稳健性来自于其输入图像的分割；另一些文献则认为多头自我注意(MSA)是保持稳健性的关键。在本文中，我们旨在引入一个原则性和统一的理论框架来研究这种关于VIT稳健性的争论。我们首先从理论上证明，与自然语言处理中的变形金刚不同，VITS是Lipschitz连续的。然后，我们从柯西问题的角度对VITS的对抗健壮性进行了理论分析，通过它我们可以量化健壮性是如何通过层传播的。我们证明了第一层和最后一层是影响VITS稳健性的关键因素。此外，基于我们的理论，我们的经验表明，与现有研究的结论不同，MSA仅有助于VITS在弱对抗攻击(如FGSM)下的对抗健壮性，并且令人惊讶的是，MSA实际上包含了该模型在更强攻击(如PGD攻击)下的对抗健壮性。



## **35. On the Detection of Adaptive Adversarial Attacks in Speaker Verification Systems**

说话人确认系统中自适应敌意攻击的检测 cs.CR

**SubmitDate**: 2022-08-01    [paper-pdf](http://arxiv.org/pdf/2202.05725v2)

**Authors**: Zesheng Chen

**Abstracts**: Speaker verification systems have been widely used in smart phones and Internet of things devices to identify legitimate users. In recent work, it has been shown that adversarial attacks, such as FAKEBOB, can work effectively against speaker verification systems. The goal of this paper is to design a detector that can distinguish an original audio from an audio contaminated by adversarial attacks. Specifically, our designed detector, called MEH-FEST, calculates the minimum energy in high frequencies from the short-time Fourier transform of an audio and uses it as a detection metric. Through both analysis and experiments, we show that our proposed detector is easy to implement, fast to process an input audio, and effective in determining whether an audio is corrupted by FAKEBOB attacks. The experimental results indicate that the detector is extremely effective: with near zero false positive and false negative rates for detecting FAKEBOB attacks in Gaussian mixture model (GMM) and i-vector speaker verification systems. Moreover, adaptive adversarial attacks against our proposed detector and their countermeasures are discussed and studied, showing the game between attackers and defenders.

摘要: 说话人验证系统已被广泛应用于智能手机和物联网设备中，以识别合法用户。最近的工作表明，FAKEBOB等对抗性攻击可以有效地对抗说话人验证系统。本文的目标是设计一种能够区分原始音频和被敌意攻击污染的音频的检测器。具体地说，我们设计的检测器MEH-FEST从音频的短时傅里叶变换计算高频最小能量，并将其用作检测度量。通过分析和实验表明，我们提出的检测器实现简单，处理输入音频的速度快，并能有效地判断音频是否被FAKEBOB攻击破坏。实验结果表明，该检测器对混合高斯模型(GMM)和I-向量说话人确认系统中的FAKEBOB攻击具有极高的检测效率：几乎为零的误检率和漏检率。此外，还讨论和研究了针对我们提出的检测器的自适应对抗性攻击及其对策，展示了攻击者和防御者之间的博弈。



## **36. The Geometry of Adversarial Training in Binary Classification**

二元分类中对抗性训练的几何问题 cs.LG

**SubmitDate**: 2022-08-01    [paper-pdf](http://arxiv.org/pdf/2111.13613v2)

**Authors**: Leon Bungert, Nicolás García Trillos, Ryan Murray

**Abstracts**: We establish an equivalence between a family of adversarial training problems for non-parametric binary classification and a family of regularized risk minimization problems where the regularizer is a nonlocal perimeter functional. The resulting regularized risk minimization problems admit exact convex relaxations of the type $L^1+$ (nonlocal) $\operatorname{TV}$, a form frequently studied in image analysis and graph-based learning. A rich geometric structure is revealed by this reformulation which in turn allows us to establish a series of properties of optimal solutions of the original problem, including the existence of minimal and maximal solutions (interpreted in a suitable sense), and the existence of regular solutions (also interpreted in a suitable sense). In addition, we highlight how the connection between adversarial training and perimeter minimization problems provides a novel, directly interpretable, statistical motivation for a family of regularized risk minimization problems involving perimeter/total variation. The majority of our theoretical results are independent of the distance used to define adversarial attacks.

摘要: 我们建立了一类非参数二分类的对抗性训练问题和一类正则化风险最小化问题之间的等价关系，其中正则化子是非局部周长泛函。由此产生的正则化风险最小化问题允许类型为$L^1+$(非局部)$操作符{TV}$的精确凸松弛，这是图像分析和基于图的学习中经常研究的一种形式。它揭示了原问题最优解的一系列性质，包括最小解和最大解的存在性(在适当的意义上解释)和正则解的存在(在适当的意义上解释)。此外，我们强调了对抗性训练和周长最小化问题之间的联系如何为一类涉及周长/总变异的正则化风险最小化问题提供了一种新颖的、直接可解释的统计动机。我们的大多数理论结果与用于定义对抗性攻击的距离无关。



## **37. DNNShield: Dynamic Randomized Model Sparsification, A Defense Against Adversarial Machine Learning**

DNNShield：动态随机化模型稀疏化，对抗对抗性机器学习 cs.CR

**SubmitDate**: 2022-07-31    [paper-pdf](http://arxiv.org/pdf/2208.00498v1)

**Authors**: Mohammad Hossein Samavatian, Saikat Majumdar, Kristin Barber, Radu Teodorescu

**Abstracts**: DNNs are known to be vulnerable to so-called adversarial attacks that manipulate inputs to cause incorrect results that can be beneficial to an attacker or damaging to the victim. Recent works have proposed approximate computation as a defense mechanism against machine learning attacks. We show that these approaches, while successful for a range of inputs, are insufficient to address stronger, high-confidence adversarial attacks. To address this, we propose DNNSHIELD, a hardware-accelerated defense that adapts the strength of the response to the confidence of the adversarial input. Our approach relies on dynamic and random sparsification of the DNN model to achieve inference approximation efficiently and with fine-grain control over the approximation error. DNNSHIELD uses the output distribution characteristics of sparsified inference compared to a dense reference to detect adversarial inputs. We show an adversarial detection rate of 86% when applied to VGG16 and 88% when applied to ResNet50, which exceeds the detection rate of the state of the art approaches, with a much lower overhead. We demonstrate a software/hardware-accelerated FPGA prototype, which reduces the performance impact of DNNSHIELD relative to software-only CPU and GPU implementations.

摘要: 众所周知，DNN容易受到所谓的对抗性攻击，即操纵输入导致不正确的结果，从而对攻击者有利或对受害者造成损害。最近的工作提出了近似计算作为一种防御机器学习攻击的机制。我们表明，这些方法虽然对一系列投入是成功的，但不足以应对更强大、高信心的对抗性攻击。为了解决这个问题，我们提出了DNNSHIELD，这是一种硬件加速防御，它根据对手输入的置信度来调整响应的强度。我们的方法依赖于DNN模型的动态和随机稀疏化，以实现高效的推理逼近和对逼近误差的细粒度控制。DNNSHIELD利用稀疏推理相对于密集引用的输出分布特性来检测敌意输入。当应用于VGG16时，敌意检测率为86%，当应用于ResNet50时，敌意检测率为88%，这超过了现有方法的检测率，并且开销要低得多。我们展示了一个软件/硬件加速的FPGA原型，它相对于纯软件的CPU和GPU实现降低了DNNSHIELD的性能影响。



## **38. Adversarial Robustness Verification and Attack Synthesis in Stochastic Systems**

随机系统中的对抗健壮性验证与攻击综合 cs.CR

To Appear, 35th IEEE Computer Security Foundations Symposium (2022)

**SubmitDate**: 2022-07-31    [paper-pdf](http://arxiv.org/pdf/2110.02125v2)

**Authors**: Lisa Oakley, Alina Oprea, Stavros Tripakis

**Abstracts**: Probabilistic model checking is a useful technique for specifying and verifying properties of stochastic systems including randomized protocols and reinforcement learning models. Existing methods rely on the assumed structure and probabilities of certain system transitions. These assumptions may be incorrect, and may even be violated by an adversary who gains control of system components.   In this paper, we develop a formal framework for adversarial robustness in systems modeled as discrete time Markov chains (DTMCs). We base our framework on existing methods for verifying probabilistic temporal logic properties and extend it to include deterministic, memoryless policies acting in Markov decision processes (MDPs). Our framework includes a flexible approach for specifying structure-preserving and non structure-preserving adversarial models. We outline a class of threat models under which adversaries can perturb system transitions, constrained by an $\varepsilon$ ball around the original transition probabilities.   We define three main DTMC adversarial robustness problems: adversarial robustness verification, maximal $\delta$ synthesis, and worst case attack synthesis. We present two optimization-based solutions to these three problems, leveraging traditional and parametric probabilistic model checking techniques. We then evaluate our solutions on two stochastic protocols and a collection of Grid World case studies, which model an agent acting in an environment described as an MDP. We find that the parametric solution results in fast computation for small parameter spaces. In the case of less restrictive (stronger) adversaries, the number of parameters increases, and directly computing property satisfaction probabilities is more scalable. We demonstrate the usefulness of our definitions and solutions by comparing system outcomes over various properties, threat models, and case studies.

摘要: 概率模型检验是描述和验证随机系统性质的一种有用技术，包括随机化协议和强化学习模型。现有的方法依赖于某些系统转变的假设结构和概率。这些假设可能是不正确的，甚至可能被控制系统组件的对手违反。在这篇文章中，我们发展了一个形式化的框架，在离散时间马尔可夫链(DTMC)建模的系统中的对手稳健性。我们的框架基于现有的概率时态逻辑属性验证方法，并将其扩展到马尔可夫决策过程(MDP)中的确定性、无记忆策略。我们的框架包括一种灵活的方法来指定结构保持和非结构保持的对抗性模型。我们概述了一类威胁模型，在该模型下，攻击者可以干扰系统的转移，并受围绕原始转移概率的$\varepsilon$球的约束。我们定义了三个主要的DTMC攻击健壮性问题：攻击健壮性验证、最大$\Delta$合成和最坏情况攻击合成。对于这三个问题，我们提出了两种基于优化的解决方案，利用传统的和参数概率模型检测技术。然后，我们在两个随机协议和一系列网格世界案例研究上评估我们的解决方案，这些案例研究对在被描述为MDP的环境中行为的代理进行建模。我们发现，对于小的参数空间，参数解导致了快速的计算。在限制较少(较强)的对手的情况下，参数的数量增加，并且直接计算属性满意概率更具可扩展性。我们通过比较各种属性、威胁模型和案例研究的系统结果来证明我们的定义和解决方案的有效性。



## **39. Robust Real-World Image Super-Resolution against Adversarial Attacks**

抵抗敌意攻击的稳健的真实世界图像超分辨率 cs.CV

ACM-MM 2021, Code:  https://github.com/lhaof/Robust-SR-against-Adversarial-Attacks

**SubmitDate**: 2022-07-31    [paper-pdf](http://arxiv.org/pdf/2208.00428v1)

**Authors**: Jiutao Yue, Haofeng Li, Pengxu Wei, Guanbin Li, Liang Lin

**Abstracts**: Recently deep neural networks (DNNs) have achieved significant success in real-world image super-resolution (SR). However, adversarial image samples with quasi-imperceptible noises could threaten deep learning SR models. In this paper, we propose a robust deep learning framework for real-world SR that randomly erases potential adversarial noises in the frequency domain of input images or features. The rationale is that on the SR task clean images or features have a different pattern from the attacked ones in the frequency domain. Observing that existing adversarial attacks usually add high-frequency noises to input images, we introduce a novel random frequency mask module that blocks out high-frequency components possibly containing the harmful perturbations in a stochastic manner. Since the frequency masking may not only destroys the adversarial perturbations but also affects the sharp details in a clean image, we further develop an adversarial sample classifier based on the frequency domain of images to determine if applying the proposed mask module. Based on the above ideas, we devise a novel real-world image SR framework that combines the proposed frequency mask modules and the proposed adversarial classifier with an existing super-resolution backbone network. Experiments show that our proposed method is more insensitive to adversarial attacks and presents more stable SR results than existing models and defenses.

摘要: 最近，深度神经网络(DNN)在真实世界图像超分辨率(SR)方面取得了显著的成功。然而，含有准不可感知噪声的对抗性图像样本可能会威胁到深度学习随机共振模型。在本文中，我们提出了一种稳健的深度学习框架，该框架可以在输入图像或特征的频域中随机消除潜在的对抗性噪声。其基本原理是，在SR任务中，干净的图像或特征在频域具有与受攻击的图像或特征不同的模式。针对现有的敌意攻击通常会在输入图像中加入高频噪声的问题，提出了一种新的随机频率掩码模块，以随机的方式屏蔽掉可能包含有害扰动的高频分量。由于频率掩蔽不仅会破坏图像中的对抗性扰动，而且会影响清晰图像中的清晰细节，因此我们进一步提出了一种基于图像频域的对抗性样本分类器，以确定是否应用所提出的掩码模块。基于上述思想，我们设计了一种新的真实图像SR框架，将所提出的频率掩码模块和所提出的对抗性分类器与现有的超分辨率骨干网络相结合。实验表明，与已有的模型和防御方法相比，我们提出的方法对敌意攻击更不敏感，并且提供了更稳定的SR结果。



## **40. Electromagnetic Signal Injection Attacks on Differential Signaling**

对差分信令的电磁信号注入攻击 cs.CR

14 pages, 15 figures

**SubmitDate**: 2022-07-31    [paper-pdf](http://arxiv.org/pdf/2208.00343v1)

**Authors**: Youqian Zhang, Kasper Rasmussen

**Abstracts**: Differential signaling is a method of data transmission that uses two complementary electrical signals to encode information. This allows a receiver to reject any noise by looking at the difference between the two signals, assuming the noise affects both signals in the same way. Many protocols such as USB, Ethernet, and HDMI use differential signaling to achieve a robust communication channel in a noisy environment. This generally works well and has led many to believe that it is infeasible to remotely inject attacking signals into such a differential pair. In this paper we challenge this assumption and show that an adversary can in fact inject malicious signals from a distance, purely using common-mode injection, i.e., injecting into both wires at the same time. We show how this allows an attacker to inject bits or even arbitrary messages into a communication line. Such an attack is a significant threat to many applications, from home security and privacy to automotive systems, critical infrastructure, or implantable medical devices; in which incorrect data or unauthorized control could cause significant damage, or even fatal accidents.   We show in detail the principles of how an electromagnetic signal can bypass the noise rejection of differential signaling, and eventually result in incorrect bits in the receiver. We show how an attacker can exploit this to achieve a successful injection of an arbitrary bit, and we analyze the success rate of injecting longer arbitrary messages. We demonstrate the attack on a real system and show that the success rate can reach as high as $90\%$. Finally, we present a case study where we wirelessly inject a message into a Controller Area Network (CAN) bus, which is a differential signaling bus protocol used in many critical applications, including the automotive and aviation sector.

摘要: 差分信令是一种数据传输方法，它使用两个互补的电信号来编码信息。这允许接收器通过查看两个信号之间的差异来抑制任何噪声，假设噪声以相同的方式影响两个信号。USB、以太网和HDMI等许多协议使用差分信令在噪声环境中实现可靠的通信通道。这通常效果良好，并导致许多人认为，将攻击信号远程注入到这样的差分对中是不可行的。在这篇文章中，我们挑战了这一假设，并证明了敌手实际上可以从远处注入恶意信号，纯粹使用共模注入，即同时注入两条线路。我们展示了这如何允许攻击者向通信线路注入比特甚至任意消息。这样的攻击对许多应用程序都是一个重大威胁，从家庭安全和隐私到汽车系统、关键基础设施或植入式医疗设备；在这些应用程序中，错误的数据或未经授权的控制可能会导致重大破坏，甚至致命的事故。我们详细展示了电磁信号如何绕过差分信号的噪声抑制，并最终导致接收器中的错误比特的原理。我们展示了攻击者如何利用这一点来成功注入任意位，并分析了注入更长的任意消息的成功率。我们在一个真实的系统上演示了攻击，并表明攻击成功率可以高达90美元。最后，我们给出了一个案例研究，在该案例中，我们将消息无线注入控制器区域网络(CAN)总线，这是一种在许多关键应用中使用的差分信号总线协议，包括汽车和航空领域。



## **41. Backdoor Attack is a Devil in Federated GAN-based Medical Image Synthesis**

后门攻击是联邦GAN医学图像合成中的一大难题 cs.CV

13 pages, 4 figures, Accepted by MICCAI 2022 SASHIMI Workshop

**SubmitDate**: 2022-07-30    [paper-pdf](http://arxiv.org/pdf/2207.00762v2)

**Authors**: Ruinan Jin, Xiaoxiao Li

**Abstracts**: Deep Learning-based image synthesis techniques have been applied in healthcare research for generating medical images to support open research. Training generative adversarial neural networks (GAN) usually requires large amounts of training data. Federated learning (FL) provides a way of training a central model using distributed data from different medical institutions while keeping raw data locally. However, FL is vulnerable to backdoor attack, an adversarial by poisoning training data, given the central server cannot access the original data directly. Most backdoor attack strategies focus on classification models and centralized domains. In this study, we propose a way of attacking federated GAN (FedGAN) by treating the discriminator with a commonly used data poisoning strategy in backdoor attack classification models. We demonstrate that adding a small trigger with size less than 0.5 percent of the original image size can corrupt the FL-GAN model. Based on the proposed attack, we provide two effective defense strategies: global malicious detection and local training regularization. We show that combining the two defense strategies yields a robust medical image generation.

摘要: 基于深度学习的图像合成技术已被应用于医疗保健研究，以生成支持开放研究的医学图像。生成对抗神经网络(GAN)的训练通常需要大量的训练数据。联合学习(FL)提供了一种使用来自不同医疗机构的分布式数据训练中央模型的方法，同时保持本地的原始数据。然而，由于中央服务器不能直接访问原始数据，FL很容易受到后门攻击，这是通过毒化训练数据而产生的敌意。大多数后门攻击策略侧重于分类模型和集中域。在这项研究中，我们提出了一种利用后门攻击分类模型中常用的数据中毒策略来处理鉴别器来攻击联邦GAN(FedGAN)的方法。我们证明，添加一个尺寸小于原始图像尺寸0.5%的小触发器可以破坏FL-GaN模型。基于提出的攻击，我们提出了两种有效的防御策略：全局恶意检测和局部训练正则化。我们表明，结合这两种防御策略可以产生稳健的医学图像生成。



## **42. Towards Privacy-Preserving, Real-Time and Lossless Feature Matching**

走向隐私保护、实时无损的特征匹配 cs.CV

**SubmitDate**: 2022-07-30    [paper-pdf](http://arxiv.org/pdf/2208.00214v1)

**Authors**: Qiang Meng, Feng Zhou

**Abstracts**: Most visual retrieval applications store feature vectors for downstream matching tasks. These vectors, from where user information can be spied out, will cause privacy leakage if not carefully protected. To mitigate privacy risks, current works primarily utilize non-invertible transformations or fully cryptographic algorithms. However, transformation-based methods usually fail to achieve satisfying matching performances while cryptosystems suffer from heavy computational overheads. In addition, secure levels of current methods should be improved to confront potential adversary attacks. To address these issues, this paper proposes a plug-in module called SecureVector that protects features by random permutations, 4L-DEC converting and existing homomorphic encryption techniques. For the first time, SecureVector achieves real-time and lossless feature matching among sanitized features, along with much higher security levels than current state-of-the-arts. Extensive experiments on face recognition, person re-identification, image retrieval, and privacy analyses demonstrate the effectiveness of our method. Given limited public projects in this field, codes of our method and implemented baselines are made open-source in https://github.com/IrvingMeng/SecureVector.

摘要: 大多数视觉检索应用存储用于下游匹配任务的特征向量。这些媒介可以窥探用户信息，如果不小心保护，将导致隐私泄露。为了减轻隐私风险，目前的工作主要使用不可逆变换或完全加密算法。然而，基于变换的方法往往不能达到令人满意的匹配性能，而密码系统的计算开销很大。此外，应改进现有方法的安全级别，以应对潜在的对手攻击。为了解决这些问题，本文提出了一种称为安全向量的插件模块，它通过随机排列、4L-DEC转换和现有的同态加密技术来保护特征。SecureVector首次实现了经过消毒的功能之间的实时和无损功能匹配，以及比当前最先进的安全级别高得多的安全级别。在人脸识别、身份识别、图像检索和隐私分析等方面的大量实验证明了该方法的有效性。考虑到这一领域的公共项目有限，我们的方法代码和实现的基线在https://github.com/IrvingMeng/SecureVector.中是开源的



## **43. Towards Bridging the gap between Empirical and Certified Robustness against Adversarial Examples**

弥合经验性和认证的对抗实例稳健性之间的差距 cs.LG

An abridged version of this work has been presented at ICLR 2021  Workshop on Security and Safety in Machine Learning Systems:  https://aisecure-workshop.github.io/aml-iclr2021/papers/2.pdf

**SubmitDate**: 2022-07-30    [paper-pdf](http://arxiv.org/pdf/2102.05096v3)

**Authors**: Jay Nandy, Sudipan Saha, Wynne Hsu, Mong Li Lee, Xiao Xiang Zhu

**Abstracts**: The current state-of-the-art defense methods against adversarial examples typically focus on improving either empirical or certified robustness. Among them, adversarially trained (AT) models produce empirical state-of-the-art defense against adversarial examples without providing any robustness guarantees for large classifiers or higher-dimensional inputs. In contrast, existing randomized smoothing based models achieve state-of-the-art certified robustness while significantly degrading the empirical robustness against adversarial examples. In this paper, we propose a novel method, called \emph{Certification through Adaptation}, that transforms an AT model into a randomized smoothing classifier during inference to provide certified robustness for $\ell_2$ norm without affecting their empirical robustness against adversarial attacks. We also propose \emph{Auto-Noise} technique that efficiently approximates the appropriate noise levels to flexibly certify the test examples using randomized smoothing technique. Our proposed \emph{Certification through Adaptation} with \emph{Auto-Noise} technique achieves an \textit{average certified radius (ACR) scores} up to $1.102$ and $1.148$ respectively for CIFAR-10 and ImageNet datasets using AT models without affecting their empirical robustness or benign accuracy. Therefore, our paper is a step towards bridging the gap between the empirical and certified robustness against adversarial examples by achieving both using the same classifier.

摘要: 当前针对敌意例子的最先进的防御方法通常侧重于提高经验性或经验证的稳健性。其中，对抗性训练(AT)模型针对对抗性实例提供了经验最新的防御，而没有为大型分类器或高维输入提供任何健壮性保证。相比之下，现有的基于随机平滑的模型实现了最先进的经过验证的稳健性，同时显著降低了对对抗性例子的经验稳健性。在本文中，我们提出了一种新的方法，称为自适应认证，该方法在推理过程中将AT模型转换为随机平滑分类器，从而在不影响其对对手攻击的经验健壮性的情况下，提供对$EELL_2$范数的认证稳健性。我们还提出了有效逼近适当噪声水平的自动噪声技术，以灵活地使用随机化平滑技术来证明测试用例。在不影响其经验稳健性和良好精度的情况下，我们提出的采用自适应技术的自适应认证技术在使用AT模型的CIFAR-10和ImageNet数据集上分别获得了高达1.102美元和1.148美元的平均认证半径分数。因此，我们的论文通过使用相同的分类器实现了经验和验证的对抗实例之间的鲁棒性之间的差距，从而迈出了一步。



## **44. Robust Trajectory Prediction against Adversarial Attacks**

对抗敌方攻击的稳健弹道预测 cs.LG

**SubmitDate**: 2022-07-29    [paper-pdf](http://arxiv.org/pdf/2208.00094v1)

**Authors**: Yulong Cao, Danfei Xu, Xinshuo Weng, Zhuoqing Mao, Anima Anandkumar, Chaowei Xiao, Marco Pavone

**Abstracts**: Trajectory prediction using deep neural networks (DNNs) is an essential component of autonomous driving (AD) systems. However, these methods are vulnerable to adversarial attacks, leading to serious consequences such as collisions. In this work, we identify two key ingredients to defend trajectory prediction models against adversarial attacks including (1) designing effective adversarial training methods and (2) adding domain-specific data augmentation to mitigate the performance degradation on clean data. We demonstrate that our method is able to improve the performance by 46% on adversarial data and at the cost of only 3% performance degradation on clean data, compared to the model trained with clean data. Additionally, compared to existing robust methods, our method can improve performance by 21% on adversarial examples and 9% on clean data. Our robust model is evaluated with a planner to study its downstream impacts. We demonstrate that our model can significantly reduce the severe accident rates (e.g., collisions and off-road driving).

摘要: 基于深度神经网络的轨迹预测是自动驾驶系统的重要组成部分。然而，这些方法容易受到对抗性攻击，导致碰撞等严重后果。在这项工作中，我们确定了两个关键因素来防御弹道预测模型的对抗性攻击，包括(1)设计有效的对抗性训练方法和(2)添加特定领域的数据增强来缓解在干净数据上的性能下降。我们证明，与使用干净数据训练的模型相比，我们的方法能够在对抗性数据上提高46%的性能，而在干净数据上的代价只有3%的性能下降。此外，与现有的稳健方法相比，我们的方法在对抗性实例上的性能提高了21%，在干净数据上的性能提高了9%。我们的稳健模型由规划者评估，以研究其下游影响。我们证明，我们的模型可以显著降低严重事故率(例如，碰撞和越野驾驶)。



## **45. Sampling Attacks on Meta Reinforcement Learning: A Minimax Formulation and Complexity Analysis**

元强化学习中的抽样攻击：一种极小极大公式及其复杂性分析 cs.LG

**SubmitDate**: 2022-07-29    [paper-pdf](http://arxiv.org/pdf/2208.00081v1)

**Authors**: Tao Li, Haozhe Lei, Quanyan Zhu

**Abstracts**: Meta reinforcement learning (meta RL), as a combination of meta-learning ideas and reinforcement learning (RL), enables the agent to adapt to different tasks using a few samples. However, this sampling-based adaptation also makes meta RL vulnerable to adversarial attacks. By manipulating the reward feedback from sampling processes in meta RL, an attacker can mislead the agent into building wrong knowledge from training experience, which deteriorates the agent's performance when dealing with different tasks after adaptation. This paper provides a game-theoretical underpinning for understanding this type of security risk. In particular, we formally define the sampling attack model as a Stackelberg game between the attacker and the agent, which yields a minimax formulation. It leads to two online attack schemes: Intermittent Attack and Persistent Attack, which enable the attacker to learn an optimal sampling attack, defined by an $\epsilon$-first-order stationary point, within $\mathcal{O}(\epsilon^{-2})$ iterations. These attack schemes freeride the learning progress concurrently without extra interactions with the environment. By corroborating the convergence results with numerical experiments, we observe that a minor effort of the attacker can significantly deteriorate the learning performance, and the minimax approach can also help robustify the meta RL algorithms.

摘要: 元强化学习作为元学习思想和强化学习的结合，使智能体能够利用少量的样本来适应不同的任务。然而，这种基于采样的自适应也使得Meta RL容易受到对手攻击。通过操纵Meta RL中采样过程的奖励反馈，攻击者可以误导代理从训练经验中建立错误的知识，从而降低代理在适应后处理不同任务的性能。本文为理解这种类型的安全风险提供了博弈论基础。特别地，我们将抽样攻击模型正式定义为攻击者和代理之间的Stackelberg博弈，从而产生极小极大公式。它导致了两种在线攻击方案：间歇攻击和持续攻击，使攻击者能够在$\mathcal{O}(\epsilon^{-2})$迭代内学习由$\epsilon$-一阶固定点定义的最优抽样攻击。这些攻击方案同时加快了学习过程，而无需与环境进行额外的交互。通过数值实验证实了收敛结果，我们观察到攻击者的微小努力会显著降低学习性能，并且极小极大方法也有助于增强Meta RL算法的健壮性。



## **46. Can We Mitigate Backdoor Attack Using Adversarial Detection Methods?**

我们可以使用对抗性检测方法来减少后门攻击吗？ cs.LG

Accepted by IEEE TDSC

**SubmitDate**: 2022-07-28    [paper-pdf](http://arxiv.org/pdf/2006.14871v2)

**Authors**: Kaidi Jin, Tianwei Zhang, Chao Shen, Yufei Chen, Ming Fan, Chenhao Lin, Ting Liu

**Abstracts**: Deep Neural Networks are well known to be vulnerable to adversarial attacks and backdoor attacks, where minor modifications on the input are able to mislead the models to give wrong results. Although defenses against adversarial attacks have been widely studied, investigation on mitigating backdoor attacks is still at an early stage. It is unknown whether there are any connections and common characteristics between the defenses against these two attacks. We conduct comprehensive studies on the connections between adversarial examples and backdoor examples of Deep Neural Networks to seek to answer the question: can we detect backdoor using adversarial detection methods. Our insights are based on the observation that both adversarial examples and backdoor examples have anomalies during the inference process, highly distinguishable from benign samples. As a result, we revise four existing adversarial defense methods for detecting backdoor examples. Extensive evaluations indicate that these approaches provide reliable protection against backdoor attacks, with a higher accuracy than detecting adversarial examples. These solutions also reveal the relations of adversarial examples, backdoor examples and normal samples in model sensitivity, activation space and feature space. This is able to enhance our understanding about the inherent features of these two attacks and the defense opportunities.

摘要: 众所周知，深度神经网络容易受到对抗性攻击和后门攻击，在这些攻击中，对输入的微小修改能够误导模型给出错误的结果。尽管针对敌意攻击的防御已经被广泛研究，但关于减轻后门攻击的调查仍处于早期阶段。目前尚不清楚针对这两种攻击的防御之间是否有任何联系和共同特征。我们对深度神经网络的对抗性实例和后门实例之间的联系进行了全面的研究，试图回答这样一个问题：我们是否可以使用对抗性检测方法来检测后门。我们的洞察是基于这样的观察，即对抗性例子和后门例子在推理过程中都有异常，与良性样本具有高度的区分性。因此，我们对现有的四种检测后门实例的对抗性防御方法进行了修改。广泛的评估表明，这些方法提供了可靠的后门攻击保护，比检测敌意示例具有更高的准确性。这些解还揭示了对抗性样本、后门样本和正常样本在模型敏感度、激活空间和特征空间中的关系。这能够增进我们对这两起袭击的内在特征和防御机会的了解。



## **47. Look Closer to Your Enemy: Learning to Attack via Teacher-student Mimicking**

走近你的敌人：通过师生模仿学习攻击 cs.CV

13 pages, 8 figures, NDSS

**SubmitDate**: 2022-07-28    [paper-pdf](http://arxiv.org/pdf/2207.13381v2)

**Authors**: Mingejie Wang, Zhiqing Tang, Sirui Li, Dingwen Xiao

**Abstracts**: This paper aims to generate realistic attack samples of person re-identification, ReID, by reading the enemy's mind (VM). In this paper, we propose a novel inconspicuous and controllable ReID attack baseline, LCYE, to generate adversarial query images. Concretely, LCYE first distills VM's knowledge via teacher-student memory mimicking in the proxy task. Then this knowledge prior acts as an explicit cipher conveying what is essential and realistic, believed by VM, for accurate adversarial misleading. Besides, benefiting from the multiple opposing task framework of LCYE, we further investigate the interpretability and generalization of ReID models from the view of the adversarial attack, including cross-domain adaption, cross-model consensus, and online learning process. Extensive experiments on four ReID benchmarks show that our method outperforms other state-of-the-art attackers with a large margin in white-box, black-box, and target attacks. Our code is now available at https://gitfront.io/r/user-3704489/mKXusqDT4ffr/LCYE/.

摘要: 本文旨在通过读取敌人的心理(Vm)来生成真实的人重新识别的攻击样本，Reid。本文提出了一种新的隐蔽可控的Reid攻击基线--LCYE，用于生成敌意查询图像。具体来说，LCYE首先通过模仿代理任务中的师生记忆来提取VM的知识。然后，这种先验知识就像一个明确的密码，传达了被VM认为是必要和现实的东西，以实现准确的对抗性误导。此外，得益于LCYE的多重对立任务框架，我们从对抗性攻击的角度进一步考察了Reid模型的可解释性和泛化，包括跨域适应、跨模型共识和在线学习过程。在四个Reid基准测试上的大量实验表明，我们的方法在白盒、黑盒和目标攻击中的性能远远优于其他最先进的攻击者。我们的代码现已在https://gitfront.io/r/user-3704489/mKXusqDT4ffr/LCYE/.上提供



## **48. Privacy-Preserving Federated Recurrent Neural Networks**

隐私保护的联邦递归神经网络 cs.CR

**SubmitDate**: 2022-07-28    [paper-pdf](http://arxiv.org/pdf/2207.13947v1)

**Authors**: Sinem Sav, Abdulrahman Diaa, Apostolos Pyrgelis, Jean-Philippe Bossuat, Jean-Pierre Hubaux

**Abstracts**: We present RHODE, a novel system that enables privacy-preserving training of and prediction on Recurrent Neural Networks (RNNs) in a federated learning setting by relying on multiparty homomorphic encryption (MHE). RHODE preserves the confidentiality of the training data, the model, and the prediction data; and it mitigates the federated learning attacks that target the gradients under a passive-adversary threat model. We propose a novel packing scheme, multi-dimensional packing, for a better utilization of Single Instruction, Multiple Data (SIMD) operations under encryption. With multi-dimensional packing, RHODE enables the efficient processing, in parallel, of a batch of samples. To avoid the exploding gradients problem, we also provide several clip-by-value approximations for enabling gradient clipping under encryption. We experimentally show that the model performance with RHODE remains similar to non-secure solutions both for homogeneous and heterogeneous data distribution among the data holders. Our experimental evaluation shows that RHODE scales linearly with the number of data holders and the number of timesteps, sub-linearly and sub-quadratically with the number of features and the number of hidden units of RNNs, respectively. To the best of our knowledge, RHODE is the first system that provides the building blocks for the training of RNNs and its variants, under encryption in a federated learning setting.

摘要: 我们提出了一种新的系统Rhode，它依靠多方同态加密(MHE)在联邦学习环境中实现对递归神经网络(RNN)的隐私保护训练和预测。Rhode保留了训练数据、模型和预测数据的机密性；它缓解了被动对手威胁模型下针对梯度的联合学习攻击。为了更好地利用加密环境下的单指令、多数据(SIMD)运算，提出了一种新的打包方案--多维打包。通过多维包装，Rhode能够并行高效地处理一批样品。为了避免爆炸的梯度问题，我们还提供了几种逐值近似的方法来实现加密下的梯度裁剪。我们的实验表明，对于数据持有者之间的同质和异质数据分布，Rhode模型的性能与非安全解决方案相似。我们的实验评估表明，Rhode与数据持有者数量和时间步数成线性关系，分别与RNN的特征数和隐含单元数成亚线性和次二次关系。据我们所知，Rhode是第一个在联合学习环境中加密的、为RNN及其变体的训练提供构建块的系统。



## **49. Label-Only Membership Inference Attack against Node-Level Graph Neural Networks**

针对节点级图神经网络的仅标签隶属度推理攻击 cs.CR

**SubmitDate**: 2022-07-27    [paper-pdf](http://arxiv.org/pdf/2207.13766v1)

**Authors**: Mauro Conti, Jiaxin Li, Stjepan Picek, Jing Xu

**Abstracts**: Graph Neural Networks (GNNs), inspired by Convolutional Neural Networks (CNNs), aggregate the message of nodes' neighbors and structure information to acquire expressive representations of nodes for node classification, graph classification, and link prediction. Previous studies have indicated that GNNs are vulnerable to Membership Inference Attacks (MIAs), which infer whether a node is in the training data of GNNs and leak the node's private information, like the patient's disease history. The implementation of previous MIAs takes advantage of the models' probability output, which is infeasible if GNNs only provide the prediction label (label-only) for the input.   In this paper, we propose a label-only MIA against GNNs for node classification with the help of GNNs' flexible prediction mechanism, e.g., obtaining the prediction label of one node even when neighbors' information is unavailable. Our attacking method achieves around 60\% accuracy, precision, and Area Under the Curve (AUC) for most datasets and GNN models, some of which are competitive or even better than state-of-the-art probability-based MIAs implemented under our environment and settings. Additionally, we analyze the influence of the sampling method, model selection approach, and overfitting level on the attack performance of our label-only MIA. Both of those factors have an impact on the attack performance. Then, we consider scenarios where assumptions about the adversary's additional dataset (shadow dataset) and extra information about the target model are relaxed. Even in those scenarios, our label-only MIA achieves a better attack performance in most cases. Finally, we explore the effectiveness of possible defenses, including Dropout, Regularization, Normalization, and Jumping knowledge. None of those four defenses prevent our attack completely.

摘要: 图神经网络(GNN)受卷积神经网络(CNN)的启发，将节点的邻居信息和结构信息聚合在一起，得到节点的表达形式，用于节点分类、图分类和链接预测。以往的研究表明，GNN容易受到成员关系推断攻击(MIA)，MIA可以推断节点是否在GNN的训练数据中，并泄露节点的私有信息，如患者的病史。以前的MIA的实现利用了模型的概率输出，如果GNN只为输入提供预测标签(仅标签)，这是不可行的。本文利用GNN灵活的预测机制，提出了一种针对GNN的只有标签的MIA用于节点分类，例如，即使在邻居信息不可用的情况下也能获得一个节点的预测标签。对于大多数数据集和GNN模型，我们的攻击方法达到了大约60%的准确率、精确度和曲线下面积(AUC)，其中一些可以与在我们的环境和设置下实现的最先进的基于概率的MIA相媲美，甚至更好。此外，我们还分析了采样方法、模型选择方法和过拟合程度对仅标签MIA攻击性能的影响。这两个因素都会对攻击性能产生影响。然后，我们考虑放松对对手的额外数据集(阴影数据集)和关于目标模型的额外信息的假设。即使在这些情况下，我们的仅标签MIA在大多数情况下也可以实现更好的攻击性能。最后，我们探讨了可能的防御措施的有效性，包括丢弃、正则化、正规化和跳跃知识。这四种防御手段都不能完全阻止我们的进攻。



## **50. Membership Inference Attacks via Adversarial Examples**

基于对抗性例子的成员关系推理攻击 cs.LG

**SubmitDate**: 2022-07-27    [paper-pdf](http://arxiv.org/pdf/2207.13572v1)

**Authors**: Hamid Jalalzai, Elie Kadoche, Rémi Leluc, Vincent Plassier

**Abstracts**: The raise of machine learning and deep learning led to significant improvement in several domains. This change is supported by both the dramatic rise in computation power and the collection of large datasets. Such massive datasets often include personal data which can represent a threat to privacy. Membership inference attacks are a novel direction of research which aims at recovering training data used by a learning algorithm. In this paper, we develop a mean to measure the leakage of training data leveraging a quantity appearing as a proxy of the total variation of a trained model near its training samples. We extend our work by providing a novel defense mechanism. Our contributions are supported by empirical evidence through convincing numerical experiments.

摘要: 机器学习和深度学习的兴起导致了几个领域的显著改善。计算能力的戏剧性增长和大型数据集的收集都支持这种变化。如此庞大的数据集通常包括可能对隐私构成威胁的个人数据。隶属度推理攻击是一个新的研究方向，其目的是恢复学习算法所使用的训练数据。在本文中，我们开发了一种方法来衡量训练数据的泄漏，该方法利用一个量来衡量训练模型在其训练样本附近的总变异。我们通过提供一种新颖的防御机制来扩展我们的工作。通过令人信服的数值实验，我们的贡献得到了经验证据的支持。



