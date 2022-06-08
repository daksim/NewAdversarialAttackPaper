# Latest Adversarial Attack Papers
**update at 2022-06-09 06:31:31**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Optimal Clock Synchronization with Signatures**

利用签名实现最优时钟同步 cs.DC

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2203.02553v2)

**Authors**: Christoph Lenzen, Julian Loss

**Abstracts**: Cryptographic signatures can be used to increase the resilience of distributed systems against adversarial attacks, by increasing the number of faulty parties that can be tolerated. While this is well-studied for consensus, it has been underexplored in the context of fault-tolerant clock synchronization, even in fully connected systems. Here, the honest parties of an $n$-node system are required to compute output clocks of small skew (i.e., maximum phase offset) despite local clock rates varying between $1$ and $\vartheta>1$, end-to-end communication delays varying between $d-u$ and $d$, and the interference from malicious parties. So far, it is only known that clock pulses of skew $d$ can be generated with (trivially optimal) resilience of $\lceil n/2\rceil-1$ (PODC `19), improving over the tight bound of $\lceil n/3\rceil-1$ holding without signatures for \emph{any} skew bound (STOC `84, PODC `85). Since typically $d\gg u$ and $\vartheta-1\ll 1$, this is far from the lower bound of $u+(\vartheta-1)d$ that applies even in the fault-free case (IPL `01).   We prove matching upper and lower bounds of $\Theta(u+(\vartheta-1)d)$ on the skew for the resilience range from $\lceil n/3\rceil$ to $\lceil n/2\rceil-1$. The algorithm showing the upper bound is, under the assumption that the adversary cannot forge signatures, deterministic. The lower bound holds even if clocks are initially perfectly synchronized, message delays between honest nodes are known, $\vartheta$ is arbitrarily close to one, and the synchronization algorithm is randomized. This has crucial implications for network designers that seek to leverage signatures for providing more robust time. In contrast to the setting without signatures, they must ensure that an attacker cannot easily bypass the lower bound on the delay on links with a faulty endpoint.

摘要: 通过增加可容忍的错误方的数量，可以使用加密签名来提高分布式系统对对手攻击的恢复能力。虽然这一点已经得到了广泛的研究，但在容错时钟同步的背景下，甚至在完全连接的系统中，这一点也没有得到充分的研究。这里，尽管本地时钟速率在$1$和$\vartheta>1$之间变化，端到端通信延迟在$d-u$和$d$之间变化，以及来自恶意方的干扰，但$n$节点系统的诚实方被要求计算小偏差(即最大相位偏移)的输出时钟。到目前为止，只有已知的歪斜$d$时钟脉冲能够以$\lceil n/2\rceil$(PODC`19)的(最优的)弹性产生，改进了没有签名的$\lceil n/3\rceil$保持的紧凑界限(STEC`84，PODC`85)。由于通常是$d\gg u$和$\vartheta-1\ll 1$，这远远不是即使在无故障的情况下也适用的$u+(\vartheta-1)d$的下限(IPL‘01)。我们证明了$theta(u+(vartheta-1)d)$在从$lceil n/3\rceil$到$lceil n/2\rceil-1$的斜斜度上的上下界是匹配的。在假设对手不能伪造签名的情况下，给出上界的算法是确定性的。即使时钟最初是完全同步的，诚实节点之间的消息延迟是已知的，$\vartheta$任意接近于1，并且同步算法是随机的，这个下界仍然成立。这对寻求利用签名来提供更可靠时间的网络设计人员具有至关重要的影响。与没有签名的设置相比，它们必须确保攻击者不能轻松绕过具有故障端点的链路上的延迟下限。



## **2. Plug & Play Attacks: Towards Robust and Flexible Model Inversion Attacks**

即插即用攻击：朝向健壮灵活的模型反转攻击 cs.LG

Accepted by ICML 2022 as Oral

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2201.12179v3)

**Authors**: Lukas Struppek, Dominik Hintersdorf, Antonio De Almeida Correia, Antonia Adler, Kristian Kersting

**Abstracts**: Model inversion attacks (MIAs) aim to create synthetic images that reflect the class-wise characteristics from a target classifier's private training data by exploiting the model's learned knowledge. Previous research has developed generative MIAs that use generative adversarial networks (GANs) as image priors tailored to a specific target model. This makes the attacks time- and resource-consuming, inflexible, and susceptible to distributional shifts between datasets. To overcome these drawbacks, we present Plug & Play Attacks, which relax the dependency between the target model and image prior, and enable the use of a single GAN to attack a wide range of targets, requiring only minor adjustments to the attack. Moreover, we show that powerful MIAs are possible even with publicly available pre-trained GANs and under strong distributional shifts, for which previous approaches fail to produce meaningful results. Our extensive evaluation confirms the improved robustness and flexibility of Plug & Play Attacks and their ability to create high-quality images revealing sensitive class characteristics.

摘要: 模型反转攻击(MIA)的目的是利用目标分类器的学习知识，从目标分类器的私有训练数据中创建反映类别特征的合成图像。以前的研究已经开发出生成性MIA，它使用生成性对抗网络(GANS)作为针对特定目标模型量身定做的图像先验。这使得攻击耗费时间和资源，不灵活，并且容易受到数据集之间的分布变化的影响。为了克服这些缺点，我们提出了即插即用攻击，它放松了目标模型和图像先验之间的依赖，使单个GAN能够攻击范围广泛的目标，只需要对攻击进行微小的调整。此外，我们表明，即使在公开可用的预先训练的GAN和强烈的分布变化下，强大的MIA也是可能的，以前的方法无法产生有意义的结果。我们广泛的评估证实了即插即用攻击的健壮性和灵活性的提高，以及它们创建揭示敏感类别特征的高质量图像的能力。



## **3. Towards Understanding and Mitigating Audio Adversarial Examples for Speaker Recognition**

说话人识别中音频对抗性实例的理解与缓解 cs.SD

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2206.03393v1)

**Authors**: Guangke Chen, Zhe Zhao, Fu Song, Sen Chen, Lingling Fan, Feng Wang, Jiashui Wang

**Abstracts**: Speaker recognition systems (SRSs) have recently been shown to be vulnerable to adversarial attacks, raising significant security concerns. In this work, we systematically investigate transformation and adversarial training based defenses for securing SRSs. According to the characteristic of SRSs, we present 22 diverse transformations and thoroughly evaluate them using 7 recent promising adversarial attacks (4 white-box and 3 black-box) on speaker recognition. With careful regard for best practices in defense evaluations, we analyze the strength of transformations to withstand adaptive attacks. We also evaluate and understand their effectiveness against adaptive attacks when combined with adversarial training. Our study provides lots of useful insights and findings, many of them are new or inconsistent with the conclusions in the image and speech recognition domains, e.g., variable and constant bit rate speech compressions have different performance, and some non-differentiable transformations remain effective against current promising evasion techniques which often work well in the image domain. We demonstrate that the proposed novel feature-level transformation combined with adversarial training is rather effective compared to the sole adversarial training in a complete white-box setting, e.g., increasing the accuracy by 13.62% and attack cost by two orders of magnitude, while other transformations do not necessarily improve the overall defense capability. This work sheds further light on the research directions in this field. We also release our evaluation platform SPEAKERGUARD to foster further research.

摘要: 说话人识别系统(SRSS)最近被证明容易受到敌意攻击，这引发了严重的安全问题。在这项工作中，我们系统地研究了基于变换和对抗性训练的安全SRSS防御。根据SRSS的特点，我们提出了22种不同的变换，并用最近在说话人识别方面有希望的7种对抗性攻击(4个白盒和3个黑盒)对它们进行了全面的评估。在仔细考虑防御评估中的最佳实践的情况下，我们分析了转换抵御自适应攻击的强度。我们还评估和理解了它们与对抗性训练相结合时对抗适应性攻击的有效性。我们的研究提供了许多有价值的见解和发现，其中许多是新的或与图像和语音识别领域的结论不一致的，例如，可变比特率和恒定比特率语音压缩具有不同的性能，一些不可微变换仍然有效地对抗当前有希望的规避技术，这些技术在图像领域通常效果很好。与完全白盒环境下的单一对抗性训练相比，本文提出的新的特征级变换结合对抗性训练是相当有效的，例如提高了13.62%的准确率和两个数量级的攻击代价，而其他变换并不一定提高整体防御能力。这项工作进一步揭示了这一领域的研究方向。我们还发布了我们的评估平台SPEAKERGUARD，以促进进一步的研究。



## **4. Building Robust Ensembles via Margin Boosting**

通过提高利润率来构建稳健的整体 cs.LG

Accepted by ICML 2022

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2206.03362v1)

**Authors**: Dinghuai Zhang, Hongyang Zhang, Aaron Courville, Yoshua Bengio, Pradeep Ravikumar, Arun Sai Suggala

**Abstracts**: In the context of adversarial robustness, a single model does not usually have enough power to defend against all possible adversarial attacks, and as a result, has sub-optimal robustness. Consequently, an emerging line of work has focused on learning an ensemble of neural networks to defend against adversarial attacks. In this work, we take a principled approach towards building robust ensembles. We view this problem from the perspective of margin-boosting and develop an algorithm for learning an ensemble with maximum margin. Through extensive empirical evaluation on benchmark datasets, we show that our algorithm not only outperforms existing ensembling techniques, but also large models trained in an end-to-end fashion. An important byproduct of our work is a margin-maximizing cross-entropy (MCE) loss, which is a better alternative to the standard cross-entropy (CE) loss. Empirically, we show that replacing the CE loss in state-of-the-art adversarial training techniques with our MCE loss leads to significant performance improvement.

摘要: 在对抗性稳健性的背景下，单个模型通常不具有足够的能力来防御所有可能的对抗性攻击，因此具有次优的稳健性。因此，一项新兴的工作重点是学习一组神经网络，以抵御对手的攻击。在这项工作中，我们采取了一种原则性的方法来建立稳健的合奏。我们从边际提升的角度来考虑这一问题，并提出了一个学习具有最大边际的集成的算法。通过在基准数据集上的广泛实验评估，我们的算法不仅优于现有的集成技术，而且优于以端到端方式训练的大型模型。我们工作的一个重要副产品是边际最大化交叉熵(MCE)损失，这是标准交叉熵(CE)损失的更好替代。经验表明，用我们的MCE损失取代最先进的对抗性训练技术中的CE损失会导致显著的性能改进。



## **5. Adaptive Regularization for Adversarial Training**

自适应正则化在对抗性训练中的应用 stat.ML

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2206.03353v1)

**Authors**: Dongyoon Yang, Insung Kong, Yongdai Kim

**Abstracts**: Adversarial training, which is to enhance robustness against adversarial attacks, has received much attention because it is easy to generate human-imperceptible perturbations of data to deceive a given deep neural network. In this paper, we propose a new adversarial training algorithm that is theoretically well motivated and empirically superior to other existing algorithms. A novel feature of the proposed algorithm is to use a data-adaptive regularization for robustifying a prediction model. We apply more regularization to data which are more vulnerable to adversarial attacks and vice versa. Even though the idea of data-adaptive regularization is not new, our data-adaptive regularization has a firm theoretical base of reducing an upper bound of the robust risk. Numerical experiments illustrate that our proposed algorithm improves the generalization (accuracy on clean samples) and robustness (accuracy on adversarial attacks) simultaneously to achieve the state-of-the-art performance.

摘要: 对抗性训练是为了提高对抗攻击的稳健性，因为它很容易产生人类无法察觉的数据扰动来欺骗给定的深度神经网络。在本文中，我们提出了一种新的对抗性训练算法，该算法在理论上动机良好，在经验上优于其他现有的算法。该算法的一个新特点是使用数据自适应正则化来增强预测模型的健壮性。我们对更容易受到对手攻击的数据应用更多的正则化，反之亦然。尽管数据自适应正则化的思想并不新鲜，但我们的数据自适应正则化在降低稳健风险上界方面有着坚实的理论基础。数值实验表明，我们提出的算法同时提高了泛化(对干净样本的准确率)和稳健性(对敌意攻击的准确率)，达到了最好的性能。



## **6. AS2T: Arbitrary Source-To-Target Adversarial Attack on Speaker Recognition Systems**

AS2T：说话人识别系统的任意源-目标对抗攻击 cs.SD

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2206.03351v1)

**Authors**: Guangke Chen, Zhe Zhao, Fu Song, Sen Chen, Lingling Fan, Yang Liu

**Abstracts**: Recent work has illuminated the vulnerability of speaker recognition systems (SRSs) against adversarial attacks, raising significant security concerns in deploying SRSs. However, they considered only a few settings (e.g., some combinations of source and target speakers), leaving many interesting and important settings in real-world attack scenarios alone. In this work, we present AS2T, the first attack in this domain which covers all the settings, thus allows the adversary to craft adversarial voices using arbitrary source and target speakers for any of three main recognition tasks. Since none of the existing loss functions can be applied to all the settings, we explore many candidate loss functions for each setting including the existing and newly designed ones. We thoroughly evaluate their efficacy and find that some existing loss functions are suboptimal. Then, to improve the robustness of AS2T towards practical over-the-air attack, we study the possible distortions occurred in over-the-air transmission, utilize different transformation functions with different parameters to model those distortions, and incorporate them into the generation of adversarial voices. Our simulated over-the-air evaluation validates the effectiveness of our solution in producing robust adversarial voices which remain effective under various hardware devices and various acoustic environments with different reverberation, ambient noises, and noise levels. Finally, we leverage AS2T to perform thus far the largest-scale evaluation to understand transferability among 14 diverse SRSs. The transferability analysis provides many interesting and useful insights which challenge several findings and conclusion drawn in previous works in the image domain. Our study also sheds light on future directions of adversarial attacks in the speaker recognition domain.

摘要: 最近的工作揭示了说话人识别系统(SRSS)对对手攻击的脆弱性，这引发了人们在部署SRSS时的重大安全担忧。然而，他们只考虑了几个设置(例如，源说话人和目标说话人的一些组合)，将许多有趣和重要的设置留在了现实世界的攻击场景中。在这项工作中，我们提出了AS2T，这是该领域的第一次攻击，覆盖了所有设置，从而允许攻击者使用任意来源和目标说话人来创建敌意语音，用于三个主要识别任务中的任何一个。由于现有的损失函数都不能适用于所有的设置，因此我们探索了每个设置的许多候选损失函数，包括现有的和新设计的损失函数。我们对它们的有效性进行了深入的评估，发现现有的一些损失函数是次优的。然后，为了提高AS2T对实际空中攻击的稳健性，我们研究了空中传输中可能出现的失真，利用不同参数的不同变换函数对这些失真进行建模，并将其融入到对抗声音的生成中。我们的模拟空中评估验证了我们的解决方案在产生健壮的对抗性声音方面的有效性，这些声音在各种硬件设备和具有不同混响、环境噪声和噪声水平的各种声学环境中仍然有效。最后，我们利用AS2T执行到目前为止最大规模的评估，以了解14个不同SRS之间的可转移性。可转移性分析提供了许多有趣和有用的见解，挑战了图像领域以前工作中得出的一些发现和结论。我们的研究也为说话人识别领域未来的对抗性攻击提供了方向。



## **7. Subject Membership Inference Attacks in Federated Learning**

联合学习中的主体成员推理攻击 cs.LG

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2206.03317v1)

**Authors**: Anshuman Suri, Pallika Kanani, Virendra J. Marathe, Daniel W. Peterson

**Abstracts**: Privacy in Federated Learning (FL) is studied at two different granularities: item-level, which protects individual data points, and user-level, which protects each user (participant) in the federation. Nearly all of the private FL literature is dedicated to studying privacy attacks and defenses at these two granularities. Recently, subject-level privacy has emerged as an alternative privacy granularity to protect the privacy of individuals (data subjects) whose data is spread across multiple (organizational) users in cross-silo FL settings. An adversary might be interested in recovering private information about these individuals (a.k.a. \emph{data subjects}) by attacking the trained model. A systematic study of these patterns requires complete control over the federation, which is impossible with real-world datasets. We design a simulator for generating various synthetic federation configurations, enabling us to study how properties of the data, model design and training, and the federation itself impact subject privacy risk. We propose three attacks for \emph{subject membership inference} and examine the interplay between all factors within a federation that affect the attacks' efficacy. We also investigate the effectiveness of Differential Privacy in mitigating this threat. Our takeaways generalize to real-world datasets like FEMNIST, giving credence to our findings.

摘要: 联合学习(FL)中的隐私在两个不同的粒度上进行研究：项级和用户级，前者保护单个数据点，后者保护联合中的每个用户(参与者)。几乎所有的私人FL文献都致力于在这两个粒度上研究隐私攻击和防御。最近，主题级别隐私已经作为一种替代隐私粒度出现，以保护其数据在跨竖井FL设置中跨多个(组织)用户分布的个人(数据主体)的隐私。对手可能对恢复这些个人的私人信息感兴趣(也称为。\emph{数据主题})攻击训练的模型。对这些模式的系统研究需要完全控制联邦，这在现实世界的数据集中是不可能的。我们设计了一个模拟器来生成各种合成联邦配置，使我们能够研究数据的属性、模型设计和训练以及联邦本身如何影响主体隐私风险。我们提出了三种针对主体成员关系推理的攻击，并考察了影响攻击效果的联邦内所有因素之间的相互作用。我们还研究了差异隐私在缓解这一威胁方面的有效性。我们的结论是推广到像FEMNIST这样的真实世界数据集，这让我们的发现更可信。



## **8. Quickest Change Detection in the Presence of Transient Adversarial Attacks**

存在瞬时敌意攻击时的最快变化检测 eess.SP

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2206.03245v1)

**Authors**: Thirupathaiah Vasantam, Don Towsley, Venugopal V. Veeravalli

**Abstracts**: We study a monitoring system in which the distributions of sensors' observations change from a nominal distribution to an abnormal distribution in response to an adversary's presence. The system uses the quickest change detection procedure, the Shewhart rule, to detect the adversary that uses its resources to affect the abnormal distribution, so as to hide its presence. The metric of interest is the probability of missed detection within a predefined number of time-slots after the changepoint. Assuming that the adversary's resource constraints are known to the detector, we find the number of required sensors to make the worst-case probability of missed detection less than an acceptable level. The distributions of observations are assumed to be Gaussian, and the presence of the adversary affects their mean. We also provide simulation results to support our analysis.

摘要: 我们研究了一个监测系统，其中传感器的观测值的分布随着对手的出现而从名义分布变为非正常分布。该系统使用最快的变化检测过程--休哈特规则来检测利用其资源影响异常分布的对手，从而隐藏其存在。感兴趣的度量是在变化点之后的预定数量的时隙内遗漏检测的概率。假设检测器知道对手的资源限制，我们找到使最坏情况下的漏检概率小于可接受水平所需的传感器数量。观测值的分布被假定为高斯分布，而对手的存在会影响其平均值。我们还提供了仿真结果来支持我们的分析。



## **9. Robust Adversarial Attacks Detection based on Explainable Deep Reinforcement Learning For UAV Guidance and Planning**

基于可解释深度强化学习的无人机制导规划鲁棒对抗攻击检测 cs.LG

13 pages, 20 figures

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2206.02670v2)

**Authors**: Thomas Hickling, Nabil Aouf, Phillippa Spencer

**Abstracts**: The danger of adversarial attacks to unprotected Uncrewed Aerial Vehicle (UAV) agents operating in public is growing. Adopting AI-based techniques and more specifically Deep Learning (DL) approaches to control and guide these UAVs can be beneficial in terms of performance but add more concerns regarding the safety of those techniques and their vulnerability against adversarial attacks causing the chances of collisions going up as the agent becomes confused. This paper proposes an innovative approach based on the explainability of DL methods to build an efficient detector that will protect these DL schemes and thus the UAVs adopting them from potential attacks. The agent is adopting a Deep Reinforcement Learning (DRL) scheme for guidance and planning. It is formed and trained with a Deep Deterministic Policy Gradient (DDPG) with Prioritised Experience Replay (PER) DRL scheme that utilises Artificial Potential Field (APF) to improve training times and obstacle avoidance performance. The adversarial attacks are generated by Fast Gradient Sign Method (FGSM) and Basic Iterative Method (BIM) algorithms and reduced obstacle course completion rates from 80\% to 35\%. A Realistic Synthetic environment for UAV explainable DRL based planning and guidance including obstacles and adversarial attacks is built. Two adversarial attack detectors are proposed. The first one adopts a Convolutional Neural Network (CNN) architecture and achieves an accuracy in detection of 80\%. The second detector is developed based on a Long Short Term Memory (LSTM) network and achieves an accuracy of 91\% with much faster computing times when compared to the CNN based detector.

摘要: 对在公共场合工作的无保护无人驾驶飞行器(UAV)特工进行敌意攻击的危险正在增加。采用基于人工智能的技术，更具体地说，深度学习(DL)方法来控制和引导这些无人机，在性能方面可能是有益的，但也增加了人们对这些技术的安全性及其对抗对手攻击的脆弱性的更多担忧，随着代理变得困惑，碰撞的可能性会增加。本文提出了一种基于DL方法的可解释性的创新方法，以构建一个有效的检测器来保护这些DL方案，从而保护采用这些方案的无人机免受潜在的攻击。该代理正在采用深度强化学习(DRL)方案来指导和规划。它是利用深度确定性策略梯度(DDPG)和优先经验重播(PER)DRL方案形成和训练的，该方案利用人工势场(APF)来改进训练时间和避障性能。采用快速梯度符号法(FGSM)和基本迭代法(BIM)算法生成对抗性攻击，将障碍路径完成率从80%降低到35%。建立了包括障碍物和对抗性攻击在内的基于DRL的无人机可解释规划和制导的现实综合环境。提出了两种对抗性攻击检测器。第一种方法采用卷积神经网络(CNN)结构，检测精度达到80%。第二种检测器是基于长短期记忆(LSTM)网络开发的，与基于CNN的检测器相比，具有91%的精度和更快的计算时间。



## **10. VLC Physical Layer Security through RIS-aided Jamming Receiver for 6G Wireless Networks**

基于RIS辅助干扰接收机的6G无线网络VLC物理层安全 cs.CR

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2205.09026v2)

**Authors**: Simone Soderi, Alessandro Brighente, Federico Turrin, Mauro Conti

**Abstracts**: Visible Light Communication (VLC) is one the most promising enabling technology for future 6G networks to overcome Radio-Frequency (RF)-based communication limitations thanks to a broader bandwidth, higher data rate, and greater efficiency. However, from the security perspective, VLCs suffer from all known wireless communication security threats (e.g., eavesdropping and integrity attacks). For this reason, security researchers are proposing innovative Physical Layer Security (PLS) solutions to protect such communication. Among the different solutions, the novel Reflective Intelligent Surface (RIS) technology coupled with VLCs has been successfully demonstrated in recent work to improve the VLC communication capacity. However, to date, the literature still lacks analysis and solutions to show the PLS capability of RIS-based VLC communication. In this paper, we combine watermarking and jamming primitives through the Watermark Blind Physical Layer Security (WBPLSec) algorithm to secure VLC communication at the physical layer. Our solution leverages RIS technology to improve the security properties of the communication. By using an optimization framework, we can calculate RIS phases to maximize the WBPLSec jamming interference schema over a predefined area in the room. In particular, compared to a scenario without RIS, our solution improves the performance in terms of secrecy capacity without any assumption about the adversary's location. We validate through numerical evaluations the positive impact of RIS-aided solution to increase the secrecy capacity of the legitimate jamming receiver in a VLC indoor scenario. Our results show that the introduction of RIS technology extends the area where secure communication occurs and that by increasing the number of RIS elements the outage probability decreases.

摘要: 可见光通信(VLC)是未来6G网络最有前途的使能技术之一，可以克服基于射频(RF)的通信限制，因为它具有更宽的带宽、更高的数据速率和更高的效率。然而，从安全的角度来看，VLC受到所有已知的无线通信安全威胁(例如，窃听和完整性攻击)。为此，安全研究人员提出了创新的物理层安全(PLS)解决方案来保护此类通信。在不同的解决方案中，新型的反射智能表面(RIS)技术与VLC相结合已经在最近的工作中被成功地展示出来，以提高VLC的通信容量。然而，到目前为止，文献仍然缺乏分析和解决方案来展示基于RIS的VLC通信的偏最小二乘能力。在本文中，我们通过水印盲物理层安全(WBPLSec)算法将水印和干扰基元相结合来保护物理层的VLC通信。我们的解决方案利用RIS技术来提高通信的安全属性。通过使用优化框架，我们可以计算RIS相位，以最大化房间中预定义区域内的WBPLSec干扰方案。特别是，与没有RIS的场景相比，我们的方案在保密能力方面提高了性能，而不需要假设对手的位置。我们通过数值评估验证了RIS辅助解决方案对提高VLC室内场景中合法干扰接收机的保密容量的积极影响。我们的结果表明，RIS技术的引入扩展了安全通信发生的区域，并且随着RIS单元数量的增加，中断概率降低。



## **11. Sampling without Replacement Leads to Faster Rates in Finite-Sum Minimax Optimization**

有限和极小极大优化中无替换抽样的快速算法 math.OC

48 pages, 3 figures

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2206.02953v1)

**Authors**: Aniket Das, Bernhard Schölkopf, Michael Muehlebach

**Abstracts**: We analyze the convergence rates of stochastic gradient algorithms for smooth finite-sum minimax optimization and show that, for many such algorithms, sampling the data points without replacement leads to faster convergence compared to sampling with replacement. For the smooth and strongly convex-strongly concave setting, we consider gradient descent ascent and the proximal point method, and present a unified analysis of two popular without-replacement sampling strategies, namely Random Reshuffling (RR), which shuffles the data every epoch, and Single Shuffling or Shuffle Once (SO), which shuffles only at the beginning. We obtain tight convergence rates for RR and SO and demonstrate that these strategies lead to faster convergence than uniform sampling. Moving beyond convexity, we obtain similar results for smooth nonconvex-nonconcave objectives satisfying a two-sided Polyak-{\L}ojasiewicz inequality. Finally, we demonstrate that our techniques are general enough to analyze the effect of data-ordering attacks, where an adversary manipulates the order in which data points are supplied to the optimizer. Our analysis also recovers tight rates for the incremental gradient method, where the data points are not shuffled at all.

摘要: 我们分析了光滑有限和极大极小优化问题的随机梯度算法的收敛速度，并证明了对于许多这类算法，对数据点进行不替换采样比用替换采样可以更快地收敛。对于光滑和强凸-强凹的情况，我们考虑了梯度下降上升和近似点方法，并对两种流行的无替换抽样策略进行了统一的分析，即随机重洗(RR)和单次洗牌(SO)。随机重洗(RR)是每一个时期都要洗牌的抽样策略，而单次洗牌(SO)是只在开始洗牌的抽样策略。我们得到了RR和SO的紧收敛速度，并证明了这两种策略比均匀抽样的收敛速度更快。超越凸性，我们得到了满足双边Polyak-L ojasiewicz不等式的光滑非凸-非凹目标的类似结果。最后，我们演示了我们的技术足够通用，可以分析数据排序攻击的影响，在这种攻击中，对手操纵向优化器提供数据点的顺序。我们的分析还恢复了增量梯度法的紧缩率，在这种方法中，数据点根本没有被洗牌。



## **12. A Robust Deep Learning Enabled Semantic Communication System for Text**

一种支持深度学习的健壮文本语义交流系统 eess.SP

6 pages

**SubmitDate**: 2022-06-06    [paper-pdf](http://arxiv.org/pdf/2206.02596v1)

**Authors**: Xiang Peng, Zhijin Qin, Danlan Huang, Xiaoming Tao, Jianhua Lu, Guangyi Liu, Chengkang Pan

**Abstracts**: With the advent of the 6G era, the concept of semantic communication has attracted increasing attention. Compared with conventional communication systems, semantic communication systems are not only affected by physical noise existing in the wireless communication environment, e.g., additional white Gaussian noise, but also by semantic noise due to the source and the nature of deep learning-based systems. In this paper, we elaborate on the mechanism of semantic noise. In particular, we categorize semantic noise into two categories: literal semantic noise and adversarial semantic noise. The former is caused by written errors or expression ambiguity, while the latter is caused by perturbations or attacks added to the embedding layer via the semantic channel. To prevent semantic noise from influencing semantic communication systems, we present a robust deep learning enabled semantic communication system (R-DeepSC) that leverages a calibrated self-attention mechanism and adversarial training to tackle semantic noise. Compared with baseline models that only consider physical noise for text transmission, the proposed R-DeepSC achieves remarkable performance in dealing with semantic noise under different signal-to-noise ratios.

摘要: 随着6G时代的到来，语义沟通的概念越来越受到关注。与传统的通信系统相比，语义通信系统不仅受到无线通信环境中存在的物理噪声(例如附加的高斯白噪声)的影响，而且由于基于深度学习的系统的来源和性质而受到语义噪声的影响。本文对语义噪声的产生机制进行了详细的阐述。特别地，我们将语义噪声分为两类：字面语义噪声和对抗性语义噪声。前者是由书面错误或表达歧义引起的，后者是通过语义通道对嵌入层进行扰动或攻击造成的。为了防止语义噪声对语义通信系统的影响，我们提出了一种健壮的深度学习语义通信系统(R-DeepSC)，该系统利用校准的自我注意机制和对抗性训练来应对语义噪声。与仅考虑物理噪声的文本传输基线模型相比，R-DeepSC在处理不同信噪比下的语义噪声方面取得了显著的性能。



## **13. Certified Robustness in Federated Learning**

联合学习中的认证稳健性 cs.LG

17 pages, 10 figures. Code available at  https://github.com/MotasemAlfarra/federated-learning-with-pytorch

**SubmitDate**: 2022-06-06    [paper-pdf](http://arxiv.org/pdf/2206.02535v1)

**Authors**: Motasem Alfarra, Juan C. Pérez, Egor Shulgin, Peter Richtárik, Bernard Ghanem

**Abstracts**: Federated learning has recently gained significant attention and popularity due to its effectiveness in training machine learning models on distributed data privately. However, as in the single-node supervised learning setup, models trained in federated learning suffer from vulnerability to imperceptible input transformations known as adversarial attacks, questioning their deployment in security-related applications. In this work, we study the interplay between federated training, personalization, and certified robustness. In particular, we deploy randomized smoothing, a widely-used and scalable certification method, to certify deep networks trained on a federated setup against input perturbations and transformations. We find that the simple federated averaging technique is effective in building not only more accurate, but also more certifiably-robust models, compared to training solely on local data. We further analyze personalization, a popular technique in federated training that increases the model's bias towards local data, on robustness. We show several advantages of personalization over both~(that is, only training on local data and federated training) in building more robust models with faster training. Finally, we explore the robustness of mixtures of global and local~(\ie personalized) models, and find that the robustness of local models degrades as they diverge from the global model

摘要: 由于联邦学习在训练分布式数据上的机器学习模型方面的有效性，它最近获得了极大的关注和普及。然而，与单节点监督学习设置中一样，在联合学习中训练的模型容易受到称为对抗性攻击的不可察觉的输入转换的影响，从而质疑其在安全相关应用中的部署。在这项工作中，我们研究了联合训练、个性化和经过认证的健壮性之间的相互作用。特别是，我们采用了随机化平滑，这是一种广泛使用和可扩展的认证方法，用于认证在联合设置上训练的深层网络不受输入扰动和转换的影响。我们发现，与仅基于本地数据进行训练相比，简单的联合平均技术不仅在建立更准确的模型方面是有效的，而且在可证明的健壮性方面也更有效。我们进一步分析了个性化，这是联合训练中的一种流行技术，它增加了模型对本地数据的偏差，并对稳健性进行了分析。我们展示了个性化比这两者(即只在本地数据上训练和联合训练)在建立更健壮的模型和更快的训练方面的几个优势。最后，我们研究了全局模型和局部模型的混合模型的稳健性，发现局部模型的稳健性随着偏离全局模型而降低



## **14. Fast Adversarial Training with Adaptive Step Size**

步长自适应的快速对抗性训练 cs.LG

**SubmitDate**: 2022-06-06    [paper-pdf](http://arxiv.org/pdf/2206.02417v1)

**Authors**: Zhichao Huang, Yanbo Fan, Chen Liu, Weizhong Zhang, Yong Zhang, Mathieu Salzmann, Sabine Süsstrunk, Jue Wang

**Abstracts**: While adversarial training and its variants have shown to be the most effective algorithms to defend against adversarial attacks, their extremely slow training process makes it hard to scale to large datasets like ImageNet. The key idea of recent works to accelerate adversarial training is to substitute multi-step attacks (e.g., PGD) with single-step attacks (e.g., FGSM). However, these single-step methods suffer from catastrophic overfitting, where the accuracy against PGD attack suddenly drops to nearly 0% during training, destroying the robustness of the networks. In this work, we study the phenomenon from the perspective of training instances. We show that catastrophic overfitting is instance-dependent and fitting instances with larger gradient norm is more likely to cause catastrophic overfitting. Based on our findings, we propose a simple but effective method, Adversarial Training with Adaptive Step size (ATAS). ATAS learns an instancewise adaptive step size that is inversely proportional to its gradient norm. The theoretical analysis shows that ATAS converges faster than the commonly adopted non-adaptive counterparts. Empirically, ATAS consistently mitigates catastrophic overfitting and achieves higher robust accuracy on CIFAR10, CIFAR100 and ImageNet when evaluated on various adversarial budgets.

摘要: 尽管对抗性训练及其变体已被证明是防御对抗性攻击的最有效算法，但它们的训练过程极其缓慢，很难扩展到像ImageNet这样的大型数据集。最近加速对抗性训练的工作的关键思想是用单步攻击(例如FGSM)代替多步攻击(例如PGD)。然而，这些单步方法存在灾难性的过拟合问题，在训练过程中对PGD攻击的准确率突然下降到近0%，破坏了网络的健壮性。在这项工作中，我们从训练实例的角度来研究这一现象。我们发现，灾难性过拟合是依赖于实例的，并且具有较大梯度范数的拟合实例更有可能导致灾难性过拟合。基于我们的研究结果，我们提出了一种简单而有效的方法--自适应步长对抗性训练(ATAS)。ATAS学习一种与其梯度范数成反比的实例化自适应步长。理论分析表明，与常用的非自适应算法相比，ATAS的收敛速度更快。从经验上看，ATAS一致地缓解了灾难性的过拟合，并在CIFAR10、CIFAR100和ImageNet上实现了更高的稳健精度，当在各种对抗性预算上进行评估时。



## **15. The art of defense: letting networks fool the attacker**

防御艺术：让网络愚弄攻击者 cs.CV

**SubmitDate**: 2022-06-06    [paper-pdf](http://arxiv.org/pdf/2104.02963v3)

**Authors**: Jinlai Zhang, Yinpeng Dong, Binbin Liu, Bo Ouyang, Jihong Zhu, Minchi Kuang, Houqing Wang, Yanmei Meng

**Abstracts**: Robust environment perception is critical for autonomous cars, and adversarial defenses are the most effective and widely studied ways to improve the robustness of environment perception. However, all of previous defense methods decrease the natural accuracy, and the nature of the DNNs itself has been overlooked. To this end, in this paper, we propose a novel adversarial defense for 3D point cloud classifier that makes full use of the nature of the DNNs. Due to the disorder of point cloud, all point cloud classifiers have the property of permutation invariant to the input point cloud. Based on this nature, we design invariant transformations defense (IT-Defense). We show that, even after accounting for obfuscated gradients, our IT-Defense is a resilient defense against state-of-the-art (SOTA) 3D attacks. Moreover, IT-Defense do not hurt clean accuracy compared to previous SOTA 3D defenses. Our code is available at: {\footnotesize{\url{https://github.com/cuge1995/IT-Defense}}}.

摘要: 稳健的环境感知是自动驾驶汽车的关键，而对抗防御是提高环境感知健壮性的最有效和被广泛研究的方法。然而，以往的防御方法都降低了DNN的自然准确率，并且忽略了DNN本身的性质。为此，在本文中，我们提出了一种新的针对三维点云分类器的对抗性防御方案，该方案充分利用了DNN的性质。由于点云的无序性，所有的点云分类器都具有对输入点云的置换不变性。基于这一性质，我们设计了不变变换防御(IT-Defense)。我们表明，即使在考虑模糊梯度之后，我们的IT防御也是针对最先进的(SOTA)3D攻击的弹性防御。此外，与以前的Sota 3D防御相比，IT防御不会损害干净的准确性。我们的代码请访问：{\footnotesize{\url{https://github.com/cuge1995/IT-Defense}}}.



## **16. Quantized and Distributed Subgradient Optimization Method with Malicious Attack**

具有恶意攻击的量化分布式次梯度优化方法 math.OC

**SubmitDate**: 2022-06-05    [paper-pdf](http://arxiv.org/pdf/2206.02272v1)

**Authors**: Iyanuoluwa Emiola, Chinwendu Enyioha

**Abstracts**: This paper considers a distributed optimization problem in a multi-agent system where a fraction of the agents act in an adversarial manner. Specifically, the malicious agents steer the network of agents away from the optimal solution by sending false information to their neighbors and consume significant bandwidth in the communication process. We propose a distributed gradient-based optimization algorithm in which the non-malicious agents exchange quantized information with one another. We prove convergence of the solution to a neighborhood of the optimal solution, and characterize the solutions obtained under the communication-constrained environment and presence of malicious agents. Numerical simulations to illustrate the results are also presented.

摘要: 本文考虑多智能体系统中的分布式优化问题，其中部分智能体以对抗性的方式行动。具体地说，恶意代理通过向其邻居发送虚假信息来引导代理网络远离最佳解决方案，并在通信过程中消耗大量带宽。我们提出了一种基于梯度的分布式优化算法，在该算法中非恶意代理之间交换量化信息。我们证明了解收敛到最优解的一个邻域，并刻画了在通信受限环境和恶意代理存在的情况下所得到的解。文中还给出了数值模拟结果。



## **17. Vanilla Feature Distillation for Improving the Accuracy-Robustness Trade-Off in Adversarial Training**

在对抗性训练中提高精确度和稳健性权衡的普通特征提取 cs.CV

12 pages

**SubmitDate**: 2022-06-05    [paper-pdf](http://arxiv.org/pdf/2206.02158v1)

**Authors**: Guodong Cao, Zhibo Wang, Xiaowei Dong, Zhifei Zhang, Hengchang Guo, Zhan Qin, Kui Ren

**Abstracts**: Adversarial training has been widely explored for mitigating attacks against deep models. However, most existing works are still trapped in the dilemma between higher accuracy and stronger robustness since they tend to fit a model towards robust features (not easily tampered with by adversaries) while ignoring those non-robust but highly predictive features. To achieve a better robustness-accuracy trade-off, we propose the Vanilla Feature Distillation Adversarial Training (VFD-Adv), which conducts knowledge distillation from a pre-trained model (optimized towards high accuracy) to guide adversarial training towards higher accuracy, i.e., preserving those non-robust but predictive features. More specifically, both adversarial examples and their clean counterparts are forced to be aligned in the feature space by distilling predictive representations from the pre-trained/clean model, while previous works barely utilize predictive features from clean models. Therefore, the adversarial training model is updated towards maximally preserving the accuracy as gaining robustness. A key advantage of our method is that it can be universally adapted to and boost existing works. Exhaustive experiments on various datasets, classification models, and adversarial training algorithms demonstrate the effectiveness of our proposed method.

摘要: 对抗性训练已被广泛探索用于减轻对深度模型的攻击。然而，大多数现有的工作仍然陷于更高的准确率和更强的稳健性之间的两难境地，因为它们倾向于向健壮的特征(不易被对手篡改)拟合模型，而忽略了那些非健壮但高度预测的特征。为了达到更好的稳健性和精确度之间的权衡，我们提出了Vanilla特征提取对抗训练(VFD-ADV)，它从预先训练的模型中进行知识提取(向高准确度优化)，以引导对抗训练朝着更高的准确率方向发展，即保留那些非稳健但可预测的特征。更具体地说，通过从预先训练/干净的模型中提取预测表示，迫使对抗性例子和它们的干净例子在特征空间中对齐，而以前的工作几乎不利用来自干净模型的预测特征。因此，对抗性训练模型在获得稳健性的同时，朝着最大限度地保持准确性的方向更新。我们方法的一个关键优势是它可以普遍适用于并促进现有的工作。在各种数据集、分类模型和对抗性训练算法上的详尽实验证明了该方法的有效性。



## **18. Federated Adversarial Training with Transformers**

与变形金刚进行联合对抗性训练 cs.LG

**SubmitDate**: 2022-06-05    [paper-pdf](http://arxiv.org/pdf/2206.02131v1)

**Authors**: Ahmed Aldahdooh, Wassim Hamidouche, Olivier Déforges

**Abstracts**: Federated learning (FL) has emerged to enable global model training over distributed clients' data while preserving its privacy. However, the global trained model is vulnerable to the evasion attacks especially, the adversarial examples (AEs), carefully crafted samples to yield false classification. Adversarial training (AT) is found to be the most promising approach against evasion attacks and it is widely studied for convolutional neural network (CNN). Recently, vision transformers have been found to be effective in many computer vision tasks. To the best of the authors' knowledge, there is no work that studied the feasibility of AT in a FL process for vision transformers. This paper investigates such feasibility with different federated model aggregation methods and different vision transformer models with different tokenization and classification head techniques. In order to improve the robust accuracy of the models with the not independent and identically distributed (Non-IID), we propose an extension to FedAvg aggregation method, called FedWAvg. By measuring the similarities between the last layer of the global model and the last layer of the client updates, FedWAvg calculates the weights to aggregate the local models updates. The experiments show that FedWAvg improves the robust accuracy when compared with other state-of-the-art aggregation methods.

摘要: 联合学习(FL)已经出现，以实现对分布式客户数据的全局模型训练，同时保护其隐私。然而，全局训练的模型很容易受到逃避攻击，尤其是对抗性例子(AEs)，精心制作的样本会产生错误的分类。对抗训练(AT)被认为是对抗逃避攻击的最有前途的方法，卷积神经网络(CNN)对其进行了广泛的研究。最近，视觉转换器被发现在许多计算机视觉任务中是有效的。就作者所知，还没有研究在视觉转换器的FL过程中AT的可行性的工作。本文采用不同的联邦模型聚合方法和不同标记化和分类头技术的视觉转换器模型，研究了这种方法的可行性。为了提高非独立同分布(Non-IID)模型的稳健精度，提出了一种扩展的FedAvg集结方法，称为FedWAvg。通过测量全局模型的最后一层和客户端更新的最后一层之间的相似性，FedWAvg计算权重以聚合本地模型更新。实验表明，FedWAvg与其他最先进的聚合方法相比，提高了健壮性。



## **19. Data-Efficient Backdoor Attacks**

数据高效的后门攻击 cs.CV

Accepted to IJCAI 2022 Long Oral

**SubmitDate**: 2022-06-05    [paper-pdf](http://arxiv.org/pdf/2204.12281v2)

**Authors**: Pengfei Xia, Ziqiang Li, Wei Zhang, Bin Li

**Abstracts**: Recent studies have proven that deep neural networks are vulnerable to backdoor attacks. Specifically, by mixing a small number of poisoned samples into the training set, the behavior of the trained model can be maliciously controlled. Existing attack methods construct such adversaries by randomly selecting some clean data from the benign set and then embedding a trigger into them. However, this selection strategy ignores the fact that each poisoned sample contributes inequally to the backdoor injection, which reduces the efficiency of poisoning. In this paper, we formulate improving the poisoned data efficiency by the selection as an optimization problem and propose a Filtering-and-Updating Strategy (FUS) to solve it. The experimental results on CIFAR-10 and ImageNet-10 indicate that the proposed method is effective: the same attack success rate can be achieved with only 47% to 75% of the poisoned sample volume compared to the random selection strategy. More importantly, the adversaries selected according to one setting can generalize well to other settings, exhibiting strong transferability. The prototype code of our method is now available at https://github.com/xpf/Data-Efficient-Backdoor-Attacks.

摘要: 最近的研究证明，深度神经网络很容易受到后门攻击。具体地说，通过将少量有毒样本混合到训练集中，可以恶意控制训练模型的行为。现有的攻击方法通过从良性集合中随机选择一些干净的数据，然后在其中嵌入触发器来构建这样的攻击者。然而，这种选择策略忽略了这样一个事实，即每个有毒样本对后门注入的贡献是不相等的，这降低了中毒的效率。在本文中，我们将通过选择来提高有毒数据效率的问题描述为一个优化问题，并提出了一种过滤和更新策略(FUS)来解决该问题。在CIFAR-10和ImageNet-10上的实验结果表明，该方法是有效的：与随机选择策略相比，只需47%~75%的中毒样本量即可获得相同的攻击成功率。更重要的是，根据一种设置选择的对手可以很好地推广到其他设置，表现出很强的可转移性。我们方法的原型代码现已在https://github.com/xpf/Data-Efficient-Backdoor-Attacks.上提供



## **20. Connecting adversarial attacks and optimal transport for domain adaptation**

连接对抗性攻击和最优传输以实现域自适应 cs.LG

**SubmitDate**: 2022-06-04    [paper-pdf](http://arxiv.org/pdf/2205.15424v2)

**Authors**: Arip Asadulaev, Vitaly Shutov, Alexander Korotin, Alexander Panfilov, Andrey Filchenkov

**Abstracts**: We present a novel algorithm for domain adaptation using optimal transport. In domain adaptation, the goal is to adapt a classifier trained on the source domain samples to the target domain. In our method, we use optimal transport to map target samples to the domain named source fiction. This domain differs from the source but is accurately classified by the source domain classifier. Our main idea is to generate a source fiction by c-cyclically monotone transformation over the target domain. If samples with the same labels in two domains are c-cyclically monotone, the optimal transport map between these domains preserves the class-wise structure, which is the main goal of domain adaptation. To generate a source fiction domain, we propose an algorithm that is based on our finding that adversarial attacks are a c-cyclically monotone transformation of the dataset. We conduct experiments on Digits and Modern Office-31 datasets and achieve improvement in performance for simple discrete optimal transport solvers for all adaptation tasks.

摘要: 我们提出了一种新的基于最优传输的域自适应算法。在领域自适应中，目标是使在源域样本上训练的分类器适应于目标域。在我们的方法中，我们使用最优传输将目标样本映射到名为源虚构的域。此域与源不同，但源域分类器会对其进行准确分类。我们的主要思想是通过目标域上的c-循环单调变换来生成源小说。如果两个结构域中具有相同标记的样本是c-循环单调的，那么这些结构域之间的最优传输映射保持了类结构，这是结构域适应的主要目标。为了生成源虚构领域，我们提出了一种算法，该算法基于我们的发现，即对抗性攻击是数据集的c循环单调变换。我们在Digits和现代Office-31数据集上进行了实验，并在所有适应任务的简单离散最优传输求解器的性能上取得了改进。



## **21. A General Framework for Evaluating Robustness of Combinatorial Optimization Solvers on Graphs**

评价图上组合优化求解器稳健性的通用框架 math.OC

**SubmitDate**: 2022-06-04    [paper-pdf](http://arxiv.org/pdf/2201.00402v2)

**Authors**: Han Lu, Zenan Li, Runzhong Wang, Qibing Ren, Junchi Yan, Xiaokang Yang

**Abstracts**: Solving combinatorial optimization (CO) on graphs is among the fundamental tasks for upper-stream applications in data mining, machine learning and operations research. Despite the inherent NP-hard challenge for CO, heuristics, branch-and-bound, learning-based solvers are developed to tackle CO problems as accurately as possible given limited time budgets. However, a practical metric for the sensitivity of CO solvers remains largely unexplored. Existing theoretical metrics require the optimal solution which is infeasible, and the gradient-based adversarial attack metric from deep learning is not compatible with non-learning solvers that are usually non-differentiable. In this paper, we develop the first practically feasible robustness metric for general combinatorial optimization solvers. We develop a no worse optimal cost guarantee thus do not require optimal solutions, and we tackle the non-differentiable challenge by resorting to black-box adversarial attack methods. Extensive experiments are conducted on 14 unique combinations of solvers and CO problems, and we demonstrate that the performance of state-of-the-art solvers like Gurobi can degenerate by over 20% under the given time limit bound on the hard instances discovered by our robustness metric, raising concerns about the robustness of combinatorial optimization solvers.

摘要: 求解图上的组合优化问题是数据挖掘、机器学习和运筹学中上游应用的基本任务之一。尽管CO存在固有的NP-Hard挑战，但启发式、分支定界、基于学习的求解器被开发出来，以在有限的时间预算内尽可能准确地处理CO问题。然而，CO解算器灵敏度的实用指标在很大程度上仍未被探索。现有的理论度量要求最优解是不可行的，基于深度学习的基于梯度的敌意攻击度量与通常不可微的非学习求解器不兼容。在这篇文章中，我们为一般的组合优化求解器发展了第一个实用可行的稳健性度量。我们开发了一个不会更差的最优成本保证，因此不需要最优解决方案，我们通过求助于黑箱对抗性攻击方法来应对不可区分的挑战。在14个独特的求解器和CO问题组合上进行了广泛的实验，我们证明了最先进的求解器，如Gurobi，在我们的健壮性度量发现的困难实例上，在给定的时间限制下，性能可以退化超过20%，这引起了人们对组合优化求解器的健壮性的担忧。



## **22. Guided Diffusion Model for Adversarial Purification**

对抗性净化中的引导扩散模型 cs.CV

**SubmitDate**: 2022-06-04    [paper-pdf](http://arxiv.org/pdf/2205.14969v2)

**Authors**: Jinyi Wang, Zhaoyang Lyu, Dahua Lin, Bo Dai, Hongfei Fu

**Abstracts**: With wider application of deep neural networks (DNNs) in various algorithms and frameworks, security threats have become one of the concerns. Adversarial attacks disturb DNN-based image classifiers, in which attackers can intentionally add imperceptible adversarial perturbations on input images to fool the classifiers. In this paper, we propose a novel purification approach, referred to as guided diffusion model for purification (GDMP), to help protect classifiers from adversarial attacks. The core of our approach is to embed purification into the diffusion denoising process of a Denoised Diffusion Probabilistic Model (DDPM), so that its diffusion process could submerge the adversarial perturbations with gradually added Gaussian noises, and both of these noises can be simultaneously removed following a guided denoising process. On our comprehensive experiments across various datasets, the proposed GDMP is shown to reduce the perturbations raised by adversarial attacks to a shallow range, thereby significantly improving the correctness of classification. GDMP improves the robust accuracy by 5%, obtaining 90.1% under PGD attack on the CIFAR10 dataset. Moreover, GDMP achieves 70.94% robustness on the challenging ImageNet dataset.

摘要: 随着深度神经网络(DNN)在各种算法和框架中的广泛应用，安全威胁已成为人们关注的问题之一。对抗性攻击干扰了基于DNN的图像分类器，攻击者可以故意在输入图像上添加不可察觉的对抗性扰动来愚弄分类器。在本文中，我们提出了一种新的净化方法，称为引导扩散净化模型(GDMP)，以帮助保护分类器免受对手攻击。该方法的核心是将净化嵌入到去噪扩散概率模型(DDPM)的扩散去噪过程中，使其扩散过程能够淹没带有逐渐增加的高斯噪声的对抗性扰动，并在引导去噪过程后同时去除这两种噪声。在不同数据集上的综合实验表明，所提出的GDMP将对抗性攻击引起的扰动减少到较小的范围，从而显著提高了分类的正确性。GDMP在CIFAR10数据集上的稳健准确率提高了5%，在PGD攻击下达到了90.1%。此外，GDMP在具有挑战性的ImageNet数据集上获得了70.94%的健壮性。



## **23. Soft Adversarial Training Can Retain Natural Accuracy**

软对抗训练可以保持自然的准确性 cs.LG

7 pages, 6 figures

**SubmitDate**: 2022-06-04    [paper-pdf](http://arxiv.org/pdf/2206.01904v1)

**Authors**: Abhijith Sharma, Apurva Narayan

**Abstracts**: Adversarial training for neural networks has been in the limelight in recent years. The advancement in neural network architectures over the last decade has led to significant improvement in their performance. It sparked an interest in their deployment for real-time applications. This process initiated the need to understand the vulnerability of these models to adversarial attacks. It is instrumental in designing models that are robust against adversaries. Recent works have proposed novel techniques to counter the adversaries, most often sacrificing natural accuracy. Most suggest training with an adversarial version of the inputs, constantly moving away from the original distribution. The focus of our work is to use abstract certification to extract a subset of inputs for (hence we call it 'soft') adversarial training. We propose a training framework that can retain natural accuracy without sacrificing robustness in a constrained setting. Our framework specifically targets moderately critical applications which require a reasonable balance between robustness and accuracy. The results testify to the idea of soft adversarial training for the defense against adversarial attacks. At last, we propose the scope of future work for further improvement of this framework.

摘要: 近年来，神经网络的对抗性训练一直是人们关注的焦点。在过去的十年中，神经网络结构的进步导致了它们的性能的显著提高。这引发了人们对它们在实时应用程序中的部署的兴趣。这一过程引发了了解这些模型在对抗攻击中的脆弱性的需要。它在设计对对手具有健壮性的模型方面很有帮助。最近的作品提出了新的技术来对抗对手，最常见的是牺牲了自然的准确性。大多数人建议使用对抗性版本的投入进行培训，不断远离原始分布。我们的工作重点是使用抽象认证来提取对抗性训练的输入子集(因此我们称之为“软”)。我们提出了一种训练框架，它可以在约束环境下保持自然的准确性，而不会牺牲鲁棒性。我们的框架专门针对需要在健壮性和准确性之间取得合理平衡的中等关键应用程序。结果证明了软对抗性训练对对抗攻击的防御思想。最后，对该框架的进一步完善提出了下一步的工作范围。



## **24. Saliency Attack: Towards Imperceptible Black-box Adversarial Attack**

突显攻击：向潜伏的黑盒对抗性攻击 cs.LG

**SubmitDate**: 2022-06-04    [paper-pdf](http://arxiv.org/pdf/2206.01898v1)

**Authors**: Zeyu Dai, Shengcai Liu, Ke Tang, Qing Li

**Abstracts**: Deep neural networks are vulnerable to adversarial examples, even in the black-box setting where the attacker is only accessible to the model output. Recent studies have devised effective black-box attacks with high query efficiency. However, such performance is often accompanied by compromises in attack imperceptibility, hindering the practical use of these approaches. In this paper, we propose to restrict the perturbations to a small salient region to generate adversarial examples that can hardly be perceived. This approach is readily compatible with many existing black-box attacks and can significantly improve their imperceptibility with little degradation in attack success rate. Further, we propose the Saliency Attack, a new black-box attack aiming to refine the perturbations in the salient region to achieve even better imperceptibility. Extensive experiments show that compared to the state-of-the-art black-box attacks, our approach achieves much better imperceptibility scores, including most apparent distortion (MAD), $L_0$ and $L_2$ distances, and also obtains significantly higher success rates judged by a human-like threshold on MAD. Importantly, the perturbations generated by our approach are interpretable to some extent. Finally, it is also demonstrated to be robust to different detection-based defenses.

摘要: 深度神经网络很容易受到敌意例子的攻击，即使在攻击者只能通过模型输出访问的黑盒环境中也是如此。最近的研究已经设计出有效的黑盒攻击，具有很高的查询效率。然而，这样的表现往往伴随着攻击隐蔽性的妥协，阻碍了这些方法的实际使用。在本文中，我们建议将扰动限制在一个很小的显著区域内，以产生难以察觉的对抗性例子。这种方法很容易与许多现有的黑盒攻击兼容，并且可以在几乎不降低攻击成功率的情况下显著提高它们的隐蔽性。此外，我们提出了显著攻击，这是一种新的黑盒攻击，旨在细化显著区域的扰动，以获得更好的不可见性。大量的实验表明，与最新的黑盒攻击相比，我们的方法获得了更好的不可见性分数，包括最明显失真(MAD)、$L0$和$L2$距离，并且以MAD上类似人类的阈值来判断成功率。重要的是，我们的方法产生的扰动在某种程度上是可以解释的。最后，还证明了该算法对不同的基于检测的防御具有较强的鲁棒性。



## **25. Reward Poisoning Attacks on Offline Multi-Agent Reinforcement Learning**

基于离线多智能体强化学习的奖励毒化攻击 cs.LG

**SubmitDate**: 2022-06-04    [paper-pdf](http://arxiv.org/pdf/2206.01888v1)

**Authors**: Young Wu, Jermey McMahan, Xiaojin Zhu, Qiaomin Xie

**Abstracts**: We expose the danger of reward poisoning in offline multi-agent reinforcement learning (MARL), whereby an attacker can modify the reward vectors to different learners in an offline data set while incurring a poisoning cost. Based on the poisoned data set, all rational learners using some confidence-bound-based MARL algorithm will infer that a target policy - chosen by the attacker and not necessarily a solution concept originally - is the Markov perfect dominant strategy equilibrium for the underlying Markov Game, hence they will adopt this potentially damaging target policy in the future. We characterize the exact conditions under which the attacker can install a target policy. We further show how the attacker can formulate a linear program to minimize its poisoning cost. Our work shows the need for robust MARL against adversarial attacks.

摘要: 我们揭示了离线多智能体强化学习(MAIL)中奖励中毒的危险，即攻击者可以在离线数据集中修改不同学习者的奖励向量，同时招致中毒成本。基于中毒数据集，所有使用基于置信度的Marl算法的理性学习者都会推断，由攻击者选择的目标策略-最初不一定是解的概念-是潜在马尔可夫博弈的马尔可夫完美支配策略均衡，因此他们将在未来采用这种潜在的破坏性目标策略。我们描述了攻击者可以安装目标策略的确切条件。我们进一步展示了攻击者如何制定一个线性规划来最小化其中毒成本。我们的工作表明，需要健壮的Marl来抵御对手攻击。



## **26. Kallima: A Clean-label Framework for Textual Backdoor Attacks**

Kallima：一种针对文本后门攻击的干净标签框架 cs.CR

**SubmitDate**: 2022-06-03    [paper-pdf](http://arxiv.org/pdf/2206.01832v1)

**Authors**: Xiaoyi Chen, Yinpeng Dong, Zeyu Sun, Shengfang Zhai, Qingni Shen, Zhonghai Wu

**Abstracts**: Although Deep Neural Network (DNN) has led to unprecedented progress in various natural language processing (NLP) tasks, research shows that deep models are extremely vulnerable to backdoor attacks. The existing backdoor attacks mainly inject a small number of poisoned samples into the training dataset with the labels changed to the target one. Such mislabeled samples would raise suspicion upon human inspection, potentially revealing the attack. To improve the stealthiness of textual backdoor attacks, we propose the first clean-label framework Kallima for synthesizing mimesis-style backdoor samples to develop insidious textual backdoor attacks. We modify inputs belonging to the target class with adversarial perturbations, making the model rely more on the backdoor trigger. Our framework is compatible with most existing backdoor triggers. The experimental results on three benchmark datasets demonstrate the effectiveness of the proposed method.

摘要: 尽管深度神经网络(DNN)在各种自然语言处理(NLP)任务中取得了前所未有的进步，但研究表明，深度模型极易受到后门攻击。现有的后门攻击主要是将少量有毒样本注入训练数据集，并将标签更改为目标样本。这种贴错标签的样本在人工检查时会引起怀疑，可能会揭示攻击。为了提高文本后门攻击的隐蔽性，我们提出了第一个干净标签框架Kallima，用于合成模仿风格的后门样本来开发隐蔽的文本后门攻击。我们使用对抗性扰动来修改属于目标类的输入，使模型更依赖于后门触发器。我们的框架与大多数现有的后门触发器兼容。在三个基准数据集上的实验结果证明了该方法的有效性。



## **27. Almost Tight L0-norm Certified Robustness of Top-k Predictions against Adversarial Perturbations**

Top-k预测对敌方扰动的几乎紧L0范数认证稳健性 cs.CR

Published as a conference paper at ICLR 2022

**SubmitDate**: 2022-06-03    [paper-pdf](http://arxiv.org/pdf/2011.07633v2)

**Authors**: Jinyuan Jia, Binghui Wang, Xiaoyu Cao, Hongbin Liu, Neil Zhenqiang Gong

**Abstracts**: Top-k predictions are used in many real-world applications such as machine learning as a service, recommender systems, and web searches. $\ell_0$-norm adversarial perturbation characterizes an attack that arbitrarily modifies some features of an input such that a classifier makes an incorrect prediction for the perturbed input. $\ell_0$-norm adversarial perturbation is easy to interpret and can be implemented in the physical world. Therefore, certifying robustness of top-$k$ predictions against $\ell_0$-norm adversarial perturbation is important. However, existing studies either focused on certifying $\ell_0$-norm robustness of top-$1$ predictions or $\ell_2$-norm robustness of top-$k$ predictions. In this work, we aim to bridge the gap. Our approach is based on randomized smoothing, which builds a provably robust classifier from an arbitrary classifier via randomizing an input. Our major theoretical contribution is an almost tight $\ell_0$-norm certified robustness guarantee for top-$k$ predictions. We empirically evaluate our method on CIFAR10 and ImageNet. For instance, our method can build a classifier that achieves a certified top-3 accuracy of 69.2\% on ImageNet when an attacker can arbitrarily perturb 5 pixels of a testing image.

摘要: Top-k预测被用于许多真实世界的应用中，例如机器学习即服务、推荐系统和网络搜索。$\ell_0$-范数对抗性扰动刻画了这样一种攻击：任意修改输入的某些特征，使得分类器对扰动输入做出错误的预测。$\ell_0$-范数对抗性摄动很容易解释，并且可以在物理世界中实现。因此，证明top-$k$预测对$\ell_0$-范数对抗扰动的稳健性是很重要的。然而，现有的研究要么集中于证明TOP-$1$预测的$\ELL_0$-范数稳健性，要么集中于证明TOP-$K$预测的$\ELL_2$-范数稳健性。在这项工作中，我们的目标是弥合这一差距。我们的方法基于随机化平滑，通过随机化输入，从任意分类器构建可证明稳健的分类器。我们的主要理论贡献是对top-$k$预测提供了几乎紧的$\ell_0$-范数证明的稳健性保证。我们在CIFAR10和ImageNet上对我们的方法进行了实证评估。例如，我们的方法可以构建一个分类器，当攻击者可以任意扰乱测试图像的5个像素时，该分类器在ImageNet上的认证准确率为69.2\%。



## **28. Gradient Obfuscation Checklist Test Gives a False Sense of Security**

梯度混淆核对表测试给人一种错误的安全感 cs.CV

**SubmitDate**: 2022-06-03    [paper-pdf](http://arxiv.org/pdf/2206.01705v1)

**Authors**: Nikola Popovic, Danda Pani Paudel, Thomas Probst, Luc Van Gool

**Abstracts**: One popular group of defense techniques against adversarial attacks is based on injecting stochastic noise into the network. The main source of robustness of such stochastic defenses however is often due to the obfuscation of the gradients, offering a false sense of security. Since most of the popular adversarial attacks are optimization-based, obfuscated gradients reduce their attacking ability, while the model is still susceptible to stronger or specifically tailored adversarial attacks. Recently, five characteristics have been identified, which are commonly observed when the improvement in robustness is mainly caused by gradient obfuscation. It has since become a trend to use these five characteristics as a sufficient test, to determine whether or not gradient obfuscation is the main source of robustness. However, these characteristics do not perfectly characterize all existing cases of gradient obfuscation, and therefore can not serve as a basis for a conclusive test. In this work, we present a counterexample, showing this test is not sufficient for concluding that gradient obfuscation is not the main cause of improvements in robustness.

摘要: 针对敌意攻击的一组流行的防御技术是基于向网络中注入随机噪声。然而，这种随机防御的主要健壮性来源往往是由于对梯度的混淆，提供了一种错误的安全感。由于大多数流行的对抗性攻击都是基于优化的，模糊梯度降低了它们的攻击能力，而该模型仍然容易受到更强或特定定制的对抗性攻击。最近，已经确定了五个特征，当稳健性的改善主要由梯度混淆引起时，通常观察到这些特征。自那以后，使用这五个特征作为充分的测试来确定梯度混淆是否是健壮性的主要来源已经成为一种趋势。然而，这些特征并不能完美地描述所有现有的梯度模糊情况，因此不能作为决定性测试的基础。在这项工作中，我们提供了一个反例，表明这个测试不足以得出梯度混淆不是健壮性提高的主要原因的结论。



## **29. Evaluating Transfer-based Targeted Adversarial Perturbations against Real-World Computer Vision Systems based on Human Judgments**

基于人的判断评估基于迁移的目标对抗性扰动对真实世界计算机视觉系统的影响 cs.CV

technical report

**SubmitDate**: 2022-06-03    [paper-pdf](http://arxiv.org/pdf/2206.01467v1)

**Authors**: Zhengyu Zhao, Nga Dang, Martha Larson

**Abstracts**: Computer vision systems are remarkably vulnerable to adversarial perturbations. Transfer-based adversarial images are generated on one (source) system and used to attack another (target) system. In this paper, we take the first step to investigate transfer-based targeted adversarial images in a realistic scenario where the target system is trained on some private data with its inventory of semantic labels not publicly available. Our main contributions include an extensive human-judgment-based evaluation of attack success on the Google Cloud Vision API and additional analysis of the different behaviors of Google Cloud Vision in face of original images vs. adversarial images. Resources are publicly available at \url{https://github.com/ZhengyuZhao/Targeted-Tansfer/blob/main/google_results.zip}.

摘要: 计算机视觉系统非常容易受到对抗性干扰的影响。基于传输的敌意图像在一个(源)系统上生成，并用于攻击另一个(目标)系统。在本文中，我们第一步研究了基于转移的目标敌意图像，在现实场景中，目标系统是在一些私有数据上进行训练的，其语义标签库不是公开的。我们的主要贡献包括对Google Cloud Vision API的攻击成功进行了广泛的基于人的判断的评估，以及对Google Cloud Vision在面对原始图像和对手图像时的不同行为进行了额外的分析。资源可在\url{https://github.com/ZhengyuZhao/Targeted-Tansfer/blob/main/google_results.zip}.上公开获得



## **30. Adversarial Attacks on Human Vision**

对人类视觉的对抗性攻击 cs.CV

21 pages, 8 figures, 1 table

**SubmitDate**: 2022-06-03    [paper-pdf](http://arxiv.org/pdf/2206.01365v1)

**Authors**: Victor A. Mateescu, Ivan V. Bajić

**Abstracts**: This article presents an introduction to visual attention retargeting, its connection to visual saliency, the challenges associated with it, and ideas for how it can be approached. The difficulty of attention retargeting as a saliency inversion problem lies in the lack of one-to-one mapping between saliency and the image domain, in addition to the possible negative impact of saliency alterations on image aesthetics. A few approaches from recent literature to solve this challenging problem are reviewed, and several suggestions for future development are presented.

摘要: 这篇文章介绍了视觉注意重定目标，它与视觉显著的联系，与之相关的挑战，以及如何处理它的想法。注意力重定向作为显著反转问题的困难在于缺乏显著与图像域之间的一对一映射，以及显著变化可能对图像美学造成的负面影响。从最近的文献中回顾了一些解决这一挑战性问题的方法，并对未来的发展提出了几点建议。



## **31. On the Privacy Properties of GAN-generated Samples**

GaN样品的保密特性研究 cs.LG

AISTATS 2021

**SubmitDate**: 2022-06-03    [paper-pdf](http://arxiv.org/pdf/2206.01349v1)

**Authors**: Zinan Lin, Vyas Sekar, Giulia Fanti

**Abstracts**: The privacy implications of generative adversarial networks (GANs) are a topic of great interest, leading to several recent algorithms for training GANs with privacy guarantees. By drawing connections to the generalization properties of GANs, we prove that under some assumptions, GAN-generated samples inherently satisfy some (weak) privacy guarantees. First, we show that if a GAN is trained on m samples and used to generate n samples, the generated samples are (epsilon, delta)-differentially-private for (epsilon, delta) pairs where delta scales as O(n/m). We show that under some special conditions, this upper bound is tight. Next, we study the robustness of GAN-generated samples to membership inference attacks. We model membership inference as a hypothesis test in which the adversary must determine whether a given sample was drawn from the training dataset or from the underlying data distribution. We show that this adversary can achieve an area under the ROC curve that scales no better than O(m^{-1/4}).

摘要: 生成性对抗网络(GAN)的隐私影响是一个非常感兴趣的话题，导致了最近几个用于训练具有隐私保证的GAN的算法。通过与GANS的泛化性质的联系，我们证明了在某些假设下，GAN生成的样本内在地满足某些(弱)隐私保证。首先，我们证明了如果一个GaN被训练在m个样本上并用来产生n个样本，所产生的样本对于(epsilon，Delta)对是(epsilon，Delta)-差分-私有的，其中Delta尺度为O(n/m)。我们证明了在某些特殊条件下，这个上界是紧的。接下来，我们研究了GAN生成的样本对成员推理攻击的稳健性。我们将成员推理建模为假设检验，其中对手必须确定给定的样本是从训练数据集还是从底层数据分布中提取的。我们证明了这个对手可以在ROC曲线下获得一个尺度不超过O(m^{-1/4})的区域。



## **32. Adaptive Adversarial Training to Improve Adversarial Robustness of DNNs for Medical Image Segmentation and Detection**

自适应对抗训练提高DNN在医学图像分割和检测中的对抗鲁棒性 eess.IV

8 pages

**SubmitDate**: 2022-06-02    [paper-pdf](http://arxiv.org/pdf/2206.01736v1)

**Authors**: Linhai Ma, Liang Liang

**Abstracts**: Recent methods based on Deep Neural Networks (DNNs) have reached high accuracy for medical image analysis, including the three basic tasks: segmentation, landmark detection, and object detection. It is known that DNNs are vulnerable to adversarial attacks, and the adversarial robustness of DNNs could be improved by adding adversarial noises to training data (i.e., adversarial training). In this study, we show that the standard adversarial training (SAT) method has a severe issue that limits its practical use: it generates a fixed level of noise for DNN training, and it is difficult for the user to choose an appropriate noise level, because a high noise level may lead to a large reduction in model performance, and a low noise level may have little effect. To resolve this issue, we have designed a novel adaptive-margin adversarial training (AMAT) method that generates adaptive adversarial noises for DNN training, which are dynamically tailored for each individual training sample. We have applied our AMAT method to state-of-the-art DNNs for the three basic tasks, using five publicly available datasets. The experimental results demonstrate that our AMAT method outperforms the SAT method in adversarial robustness on noisy data and prediction accuracy on clean data. Please contact the author for the source code.

摘要: 近年来，基于深度神经网络(DNNS)的医学图像分析方法已经达到了很高的精度，包括分割、地标检测和目标检测三个基本任务。众所周知，DNN容易受到对抗性攻击，通过在训练数据中添加对抗性噪声(即对抗性训练)可以提高DNN的对抗性健壮性。在这项研究中，我们发现标准的对抗训练(SAT)方法有一个严重的问题限制了它的实际应用：它为DNN训练产生固定的噪声水平，用户很难选择合适的噪声水平，因为高水平的噪声可能会导致模型性能的大幅下降，而低水平的噪声可能影响很小。为了解决这一问题，我们设计了一种新的自适应差值对抗训练方法(AMAT)，该方法为DNN训练生成自适应对抗噪声，这些噪声是为每个训练样本动态定制的。我们已经将我们的AMAT方法应用于三个基本任务的最先进的DNN，使用了五个公开可用的数据集。实验结果表明，我们的AMAT方法在对噪声数据的对抗稳健性和对干净数据的预测精度方面优于SAT方法。请联系作者以获取源代码。



## **33. A Barrier Certificate-based Simplex Architecture with Application to Microgrids**

一种基于屏障证书的单纯形体系结构及其在微网格中的应用 eess.SY

**SubmitDate**: 2022-06-02    [paper-pdf](http://arxiv.org/pdf/2202.09710v2)

**Authors**: Amol Damare, Shouvik Roy, Scott A. Smolka, Scott D. Stoller

**Abstracts**: We present Barrier Certificate-based Simplex (BC-Simplex), a new, provably correct design for runtime assurance of continuous dynamical systems. BC-Simplex is centered around the Simplex Control Architecture, which consists of a high-performance advanced controller which is not guaranteed to maintain safety of the plant, a verified-safe baseline controller, and a decision module that switches control of the plant between the two controllers to ensure safety without sacrificing performance. In BC-Simplex, Barrier certificates are used to prove that the baseline controller ensures safety. Furthermore, BC-Simplex features a new automated method for deriving, from the barrier certificate, the conditions for switching between the controllers. Our method is based on the Taylor expansion of the barrier certificate and yields computationally inexpensive switching conditions. We consider a significant application of BC-Simplex to a microgrid featuring an advanced controller in the form of a neural network trained using reinforcement learning. The microgrid is modeled in RTDS, an industry-standard high-fidelity, real-time power systems simulator. Our results demonstrate that BC-Simplex can automatically derive switching conditions for complex systems, the switching conditions are not overly conservative, and BC-Simplex ensures safety even in the presence of adversarial attacks on the neural controller.

摘要: 提出了基于屏障证书的单纯形(BC-Simplex)，这是一种新的、可证明是正确的连续动态系统运行时保证设计。BC-Simplex围绕Simplex控制架构展开，该架构由不能保证维护工厂安全的高性能高级控制器、经过验证的安全基准控制器以及在两个控制器之间切换工厂控制以确保安全而不牺牲性能的决策模块组成。在BC-Simplex中，屏障证书被用来证明基线控制器确保了安全性。此外，BC-Simplex具有一种新的自动方法，用于从屏障证书推导控制器之间切换的条件。我们的方法是基于障碍证书的泰勒展开式，并且产生了计算上不昂贵的切换条件。我们考虑了BC-单纯形在微电网中的一个重要应用，该微网具有一个采用强化学习训练的神经网络形式的高级控制器。微电网是在RTDS中建模的，RTDS是一种行业标准的高保真、实时电力系统仿真器。结果表明，BC-单纯形能够自动推导出复杂系统的切换条件，切换条件不会过于保守，即使在神经控制器受到敌意攻击的情况下，BC-单纯形也能保证安全性。



## **34. Adversarial Laser Spot: Robust and Covert Physical Adversarial Attack to DNNs**

敌意激光斑点：对DNN的稳健和隐蔽的物理攻击 cs.CV

**SubmitDate**: 2022-06-02    [paper-pdf](http://arxiv.org/pdf/2206.01034v1)

**Authors**: Chengyin Hu

**Abstracts**: Most existing deep neural networks (DNNs) are easily disturbed by slight noise. As far as we know, there are few researches on physical adversarial attack technology by deploying lighting equipment. The light-based physical adversarial attack technology has excellent covertness, which brings great security risks to many applications based on deep neural networks (such as automatic driving technology). Therefore, we propose a robust physical adversarial attack technology with excellent covertness, called adversarial laser point (AdvLS), which optimizes the physical parameters of laser point through genetic algorithm to perform physical adversarial attack. It realizes robust and covert physical adversarial attack by using low-cost laser equipment. As far as we know, AdvLS is the first light-based adversarial attack technology that can perform physical adversarial attacks in the daytime. A large number of experiments in the digital and physical environments show that AdvLS has excellent robustness and concealment. In addition, through in-depth analysis of the experimental data, we find that the adversarial perturbations generated by AdvLS have superior adversarial attack migration. The experimental results show that AdvLS impose serious interference to the advanced deep neural networks, we call for the attention of the proposed physical adversarial attack technology.

摘要: 现有的大部分深度神经网络(DNN)都容易受到微弱噪声的干扰。据我们所知，通过部署照明设备进行物理对抗攻击技术的研究还很少。基于光的物理对抗攻击技术具有良好的隐蔽性，这给许多基于深度神经网络的应用(如自动驾驶技术)带来了极大的安全隐患。因此，我们提出了一种具有良好隐蔽性的健壮物理对抗攻击技术，称为对抗激光点(AdvLS)，它通过遗传算法优化激光点的物理参数来执行物理对抗攻击。它利用低成本的激光设备实现了健壮隐蔽的物理对抗攻击。据我们所知，AdvLS是第一个可以在白天进行物理对抗攻击的基于光的对抗攻击技术。在数字和物理环境中的大量实验表明，AdvLS具有良好的稳健性和隐蔽性。此外，通过对实验数据的深入分析，我们发现AdvLS产生的对抗性扰动具有更好的对抗性攻击迁移能力。实验结果表明，AdvLS对高级深度神经网络造成了严重的干扰，呼吁对提出的物理对抗攻击技术给予重视。



## **35. FACM: Correct the Output of Deep Neural Network with Middle Layers Features against Adversarial Samples**

FACM：针对敌方样本修正具有中间层特征的深度神经网络输出 cs.CV

**SubmitDate**: 2022-06-02    [paper-pdf](http://arxiv.org/pdf/2206.00924v1)

**Authors**: Xiangyuan Yang, Jie Lin, Hanlin Zhang, Xinyu Yang, Peng Zhao

**Abstracts**: In the strong adversarial attacks against deep neural network (DNN), the output of DNN will be misclassified if and only if the last feature layer of the DNN is completely destroyed by adversarial samples, while our studies found that the middle feature layers of the DNN can still extract the effective features of the original normal category in these adversarial attacks. To this end, in this paper, a middle $\bold{F}$eature layer $\bold{A}$nalysis and $\bold{C}$onditional $\bold{M}$atching prediction distribution (FACM) model is proposed to increase the robustness of the DNN against adversarial samples through correcting the output of DNN with the features extracted by the middle layers of DNN. In particular, the middle $\bold{F}$eature layer $\bold{A}$nalysis (FA) module, the conditional matching prediction distribution (CMPD) module and the output decision module are included in our FACM model to collaboratively correct the classification of adversarial samples. The experiments results show that, our FACM model can significantly improve the robustness of the naturally trained model against various attacks, and our FA model can significantly improve the robustness of the adversarially trained model against white-box attacks with weak transferability and black box attacks where FA model includes the FA module and the output decision module, not the CMPD module.

摘要: 在针对深度神经网络(DNN)的强对抗性攻击中，当且仅当DNN的最后一个特征层被对抗性样本完全破坏时，DNN的输出才会被误分类，而我们的研究发现，在这些对抗性攻击中，DNN的中间特征层仍然可以提取原始正常类别的有效特征。为此，本文提出了一种中间特征层分析和条件匹配预测分布(FACM)模型，通过利用DNN中间层提取的特征对DNN的输出进行修正，提高了DNN对敌意样本的鲁棒性。特别是，在FACM模型中加入了中间特征层分析(FA)模块、条件匹配预测分布(CMPD)模块和输出决策模块，以协同纠正敌方样本的分类。实验结果表明，我们的FACM模型可以显著提高自然训练模型对各种攻击的稳健性，而我们的FA模型可以显著提高对抗性训练模型对具有弱可传递性的白盒攻击和包括FA模块和输出判决模块而不是CMPD模块的黑盒攻击的鲁棒性。



## **36. Mask-Guided Divergence Loss Improves the Generalization and Robustness of Deep Neural Network**

掩模引导的发散损失提高了深度神经网络的泛化能力和鲁棒性 cs.LG

**SubmitDate**: 2022-06-02    [paper-pdf](http://arxiv.org/pdf/2206.00913v1)

**Authors**: Xiangyuan Yang, Jie Lin, Hanlin Zhang, Xinyu Yang, Peng Zhao

**Abstracts**: Deep neural network (DNN) with dropout can be regarded as an ensemble model consisting of lots of sub-DNNs (i.e., an ensemble sub-DNN where the sub-DNN is the remaining part of the DNN after dropout), and through increasing the diversity of the ensemble sub-DNN, the generalization and robustness of the DNN can be effectively improved. In this paper, a mask-guided divergence loss function (MDL), which consists of a cross-entropy loss term and an orthogonal term, is proposed to increase the diversity of the ensemble sub-DNN by the added orthogonal term. Particularly, the mask technique is introduced to assist in generating the orthogonal term for avoiding overfitting of the diversity learning. The theoretical analysis and extensive experiments on 4 datasets (i.e., MNIST, FashionMNIST, CIFAR10, and CIFAR100) manifest that MDL can improve the generalization and robustness of standard training and adversarial training. For CIFAR10 and CIFAR100, in standard training, the maximum improvement of accuracy is $1.38\%$ on natural data, $30.97\%$ on FGSM (i.e., Fast Gradient Sign Method) attack, $38.18\%$ on PGD (i.e., Projected Gradient Descent) attack. While in adversarial training, the maximum improvement is $1.68\%$ on natural data, $4.03\%$ on FGSM attack and $2.65\%$ on PGD attack.

摘要: 具有丢包的深度神经网络(DNN)可以看作是由多个子DNN(即一个集成的子DNN，其中该子DNN是丢弃后DNN的剩余部分)组成的集成模型，通过增加集成子DNN的多样性，可以有效地提高DNN的泛化能力和鲁棒性。本文提出了一种掩模引导的发散损失函数(MDL)，该函数由交叉熵损失项和正交项组成，通过增加正交项来增加集成次DNN的分集。特别是，引入掩码技术来帮助生成正交项，以避免多样性学习的过拟合。在MNIST、FashionMNIST、CIFAR10和CIFAR100四个数据集上的理论分析和大量实验表明，MDL能够提高标准训练和对抗性训练的泛化能力和健壮性。对于CIFAR10和CIFAR100，在标准训练中，对自然数据的准确率最大提高为1.38美元，对FGSM(即快速梯度符号)攻击的准确率最大提高为30.97美元，对PGD(即投影梯度下降)攻击的准确率最大提高为38.18美元。而在对抗性训练中，对自然数据的最大改进为1.68美元，对FGSM攻击的最大改进为4.03美元，对PGD攻击的最大改进为2.65美元。



## **37. Adversarial RAW: Image-Scaling Attack Against Imaging Pipeline**

对抗性RAW：针对成像管道的图像缩放攻击 cs.CV

**SubmitDate**: 2022-06-02    [paper-pdf](http://arxiv.org/pdf/2206.01733v1)

**Authors**: Junjian Li, Honglong Chen

**Abstracts**: Deep learning technologies have become the backbone for the development of computer vision. With further explorations, deep neural networks have been found vulnerable to well-designed adversarial attacks. Most of the vision devices are equipped with image signal processing (ISP) pipeline to implement RAW-to-RGB transformations and embedded into data preprocessing module for efficient image processing. Actually, ISP pipeline can introduce adversarial behaviors to post-capture images while data preprocessing may destroy attack patterns. However, none of the existing adversarial attacks takes into account the impacts of both ISP pipeline and data preprocessing. In this paper, we develop an image-scaling attack targeting on ISP pipeline, where the crafted adversarial RAW can be transformed into attack image that presents entirely different appearance once being scaled to a specific-size image. We first consider the gradient-available ISP pipeline, i.e., the gradient information can be directly used in the generation process of adversarial RAW to launch the attack. To make the adversarial attack more applicable, we further consider the gradient-unavailable ISP pipeline, in which a proxy model that well learns the RAW-to-RGB transformations is proposed as the gradient oracles. Extensive experiments show that the proposed adversarial attacks can craft adversarial RAW data against the target ISP pipelines with high attack rates.

摘要: 深度学习技术已经成为计算机视觉发展的支柱。随着进一步的探索，深度神经网络被发现容易受到精心设计的对手攻击。大多数视觉设备都配备了图像信号处理(ISP)流水线来实现RAW到RGB的转换，并嵌入到数据预处理模块中，以实现高效的图像处理。实际上，网络服务提供商流水线可以在捕获后的图像中引入敌意行为，而数据预处理可能会破坏攻击模式。然而，现有的对抗性攻击都没有考虑到互联网服务提供商流水线和数据预处理的影响。本文提出了一种针对网络服务提供商流水线的图像缩放攻击方法，将精心制作的敌意原始图像变换成攻击图像，当图像缩放到特定大小的图像时，呈现出完全不同的外观。我们首先考虑了梯度可用的ISP管道，即在恶意RAW的生成过程中可以直接利用梯度信息来发起攻击。为了使对抗性攻击更加适用，我们进一步考虑了梯度不可用的ISP流水线，其中提出了一个能够很好地学习RAW到RGB转换的代理模型作为梯度预言。大量实验表明，所提出的对抗性攻击能够以较高的攻击率伪造针对目标网络服务提供商管道的对抗性原始数据。



## **38. Robust Feature-Level Adversaries are Interpretability Tools**

强大的功能级对手是可解释的工具 cs.LG

Code available at  https://github.com/thestephencasper/feature_level_adv

**SubmitDate**: 2022-06-02    [paper-pdf](http://arxiv.org/pdf/2110.03605v4)

**Authors**: Stephen Casper, Max Nadeau, Dylan Hadfield-Menell, Gabriel Kreiman

**Abstracts**: The literature on adversarial attacks in computer vision typically focuses on pixel-level perturbations. These tend to be very difficult to interpret. Recent work that manipulates the latent representations of image generators to create "feature-level" adversarial perturbations gives us an opportunity to explore interpretable adversarial attacks. We make three contributions. First, we observe that feature-level attacks provide useful classes of inputs for studying the representations in models. Second, we show that these adversaries are versatile and highly robust. We demonstrate that they can be used to produce targeted, universal, disguised, physically-realizable, and black-box attacks at the ImageNet scale. Third, we show how these adversarial images can be used as a practical interpretability tool for identifying bugs in networks. We use these adversaries to make predictions about spurious associations between features and classes which we then test by designing "copy/paste" attacks in which one natural image is pasted into another to cause a targeted misclassification. Our results indicate that feature-level attacks are a promising approach for rigorous interpretability research. They support the design of tools to better understand what a model has learned and diagnose brittle feature associations.

摘要: 关于计算机视觉中的对抗性攻击的文献通常集中在像素级的扰动上。这些往往很难解释。最近的工作是利用图像生成器的潜在表示来创建“特征级别”的对抗性扰动，这给了我们一个探索可解释的对抗性攻击的机会。我们有三点贡献。首先，我们观察到特征级别的攻击为研究模型中的表示提供了有用的输入类。其次，我们证明了这些对手是多才多艺的，并且非常健壮。我们证明了它们可以用于在ImageNet规模上产生有针对性的、普遍的、伪装的、物理上可实现的和黑匣子攻击。第三，我们展示了如何将这些对抗性图像用作识别网络漏洞的实用可解释性工具。我们利用这些对手来预测特征和类别之间的虚假关联，然后通过设计“复制/粘贴”攻击来测试这些关联，在这种攻击中，一幅自然图像被粘贴到另一幅图像中，从而导致有针对性的误分类。我们的结果表明，特征级攻击对于严格的可解释性研究是一种很有前途的方法。它们支持工具的设计，以更好地理解模型学习到的内容并诊断脆弱的特征关联。



## **39. On the reversibility of adversarial attacks**

论对抗性攻击的可逆性 cs.LG

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2206.00772v1)

**Authors**: Chau Yi Li, Ricardo Sánchez-Matilla, Ali Shahin Shamsabadi, Riccardo Mazzon, Andrea Cavallaro

**Abstracts**: Adversarial attacks modify images with perturbations that change the prediction of classifiers. These modified images, known as adversarial examples, expose the vulnerabilities of deep neural network classifiers. In this paper, we investigate the predictability of the mapping between the classes predicted for original images and for their corresponding adversarial examples. This predictability relates to the possibility of retrieving the original predictions and hence reversing the induced misclassification. We refer to this property as the reversibility of an adversarial attack, and quantify reversibility as the accuracy in retrieving the original class or the true class of an adversarial example. We present an approach that reverses the effect of an adversarial attack on a classifier using a prior set of classification results. We analyse the reversibility of state-of-the-art adversarial attacks on benchmark classifiers and discuss the factors that affect the reversibility.

摘要: 对抗性攻击通过改变分类器预测的扰动来修改图像。这些修改后的图像被称为对抗性例子，暴露了深度神经网络分类器的漏洞。在这篇文章中，我们研究了对原始图像和其对应的对抗性例子所预测的类别之间的映射的可预测性。这种可预测性与检索原始预测的可能性有关，从而逆转了诱导的错误分类。我们将这一性质称为对抗性攻击的可逆性，并将可逆性量化为检索对抗性示例的原始类或真实类的准确性。我们提出了一种方法，该方法使用先前的分类结果集来逆转对抗性攻击对分类器的影响。我们分析了针对基准分类器的最新对手攻击的可逆性，并讨论了影响可逆性的因素。



## **40. Training privacy-preserving video analytics pipelines by suppressing features that reveal information about private attributes**

通过抑制泄露有关私有属性信息的特征来训练保护隐私的视频分析管道 cs.CV

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2203.02635v2)

**Authors**: Chau Yi Li, Andrea Cavallaro

**Abstracts**: Deep neural networks are increasingly deployed for scene analytics, including to evaluate the attention and reaction of people exposed to out-of-home advertisements. However, the features extracted by a deep neural network that was trained to predict a specific, consensual attribute (e.g. emotion) may also encode and thus reveal information about private, protected attributes (e.g. age or gender). In this work, we focus on such leakage of private information at inference time. We consider an adversary with access to the features extracted by the layers of a deployed neural network and use these features to predict private attributes. To prevent the success of such an attack, we modify the training of the network using a confusion loss that encourages the extraction of features that make it difficult for the adversary to accurately predict private attributes. We validate this training approach on image-based tasks using a publicly available dataset. Results show that, compared to the original network, the proposed PrivateNet can reduce the leakage of private information of a state-of-the-art emotion recognition classifier by 2.88% for gender and by 13.06% for age group, with a minimal effect on task accuracy.

摘要: 深度神经网络越来越多地被用于场景分析，包括评估接触户外广告的人的注意力和反应。然而，被训练成预测特定的一致属性(例如，情绪)的深度神经网络所提取的特征也可以编码并且因此揭示关于私人的、受保护的属性(例如，年龄或性别)的信息。在这项工作中，我们关注的是推理时隐私信息的泄露。我们考虑一个对手可以访问由部署的神经网络的各层提取的特征，并使用这些特征来预测私有属性。为了防止此类攻击的成功，我们使用混淆损失来修改网络的训练，该混淆损失鼓励提取使对手难以准确预测私有属性的特征。我们使用公开可用的数据集在基于图像的任务上验证了这种训练方法。实验结果表明，与原网络相比，提出的PrivateNet能够将最先进的情感识别分类器的隐私信息泄漏减少2.88%(性别)和13.06%(年龄组)，并且对任务准确率的影响最小。



## **41. Adversarial Attacks on Gaussian Process Bandits**

对高斯过程环的对抗性攻击 stat.ML

Accepted to ICML 2022

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2110.08449v2)

**Authors**: Eric Han, Jonathan Scarlett

**Abstracts**: Gaussian processes (GP) are a widely-adopted tool used to sequentially optimize black-box functions, where evaluations are costly and potentially noisy. Recent works on GP bandits have proposed to move beyond random noise and devise algorithms robust to adversarial attacks. This paper studies this problem from the attacker's perspective, proposing various adversarial attack methods with differing assumptions on the attacker's strength and prior information. Our goal is to understand adversarial attacks on GP bandits from theoretical and practical perspectives. We focus primarily on targeted attacks on the popular GP-UCB algorithm and a related elimination-based algorithm, based on adversarially perturbing the function $f$ to produce another function $\tilde{f}$ whose optima are in some target region $\mathcal{R}_{\rm target}$. Based on our theoretical analysis, we devise both white-box attacks (known $f$) and black-box attacks (unknown $f$), with the former including a Subtraction attack and Clipping attack, and the latter including an Aggressive subtraction attack. We demonstrate that adversarial attacks on GP bandits can succeed in forcing the algorithm towards $\mathcal{R}_{\rm target}$ even with a low attack budget, and we test our attacks' effectiveness on a diverse range of objective functions.

摘要: 高斯过程(GP)是一种广泛采用的工具，用于顺序优化黑盒函数，其中评估成本较高，并且可能存在噪声。最近关于GP盗贼的研究已经提出超越随机噪声，设计出对对手攻击强大的算法。本文从攻击者的角度研究这一问题，提出了各种对抗性攻击方法，并对攻击者的强度和先验信息进行了不同的假设。我们的目标是从理论和实践的角度来理解对GP土匪的敌意攻击。我们主要关注对流行的GP-UCB算法和相关的基于消元的算法的定向攻击，该算法基于对函数$f$的恶意扰动来产生另一个函数$\tide{f}$，其最优值位于某个目标区域$\数学{R}_{\rm目标}$。在理论分析的基础上，我们设计了白盒攻击(已知$f$)和黑盒攻击(未知$f$)，前者包括减法攻击和剪裁攻击，后者包括侵略性减法攻击。我们证明了对GP盗贼的敌意攻击即使在较低的攻击预算下也能成功地迫使算法向数学上的{R}_{\Rm目标}$逼近，并在不同的目标函数上测试了我们的攻击的有效性。



## **42. The robust way to stack and bag: the local Lipschitz way**

稳健的堆叠和打包方式：当地的利普希茨方式 cs.LG

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2206.00513v1)

**Authors**: Thulasi Tholeti, Sheetal Kalyani

**Abstracts**: Recent research has established that the local Lipschitz constant of a neural network directly influences its adversarial robustness. We exploit this relationship to construct an ensemble of neural networks which not only improves the accuracy, but also provides increased adversarial robustness. The local Lipschitz constants for two different ensemble methods - bagging and stacking - are derived and the architectures best suited for ensuring adversarial robustness are deduced. The proposed ensemble architectures are tested on MNIST and CIFAR-10 datasets in the presence of white-box attacks, FGSM and PGD. The proposed architecture is found to be more robust than a) a single network and b) traditional ensemble methods.

摘要: 最近的研究表明，神经网络的局部Lipschitz常数直接影响其对抗鲁棒性。我们利用这种关系来构造神经网络集成，这不仅提高了精度，而且增加了对手的稳健性。推导了两种不同的集成方法--袋装和堆叠--的局部Lipschitz常数，并推导出最适合于确保对抗性稳健性的结构。在MNIST和CIFAR-10数据集上，在白盒攻击、FGSM和PGD的情况下对所提出的集成架构进行了测试。研究发现，该体系结构比a)单一网络和b)传统集成方法更健壮。



## **43. Attack-Agnostic Adversarial Detection**

攻击不可知的敌意检测 cs.CV

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2206.00489v1)

**Authors**: Jiaxin Cheng, Mohamed Hussein, Jay Billa, Wael AbdAlmageed

**Abstracts**: The growing number of adversarial attacks in recent years gives attackers an advantage over defenders, as defenders must train detectors after knowing the types of attacks, and many models need to be maintained to ensure good performance in detecting any upcoming attacks. We propose a way to end the tug-of-war between attackers and defenders by treating adversarial attack detection as an anomaly detection problem so that the detector is agnostic to the attack. We quantify the statistical deviation caused by adversarial perturbations in two aspects. The Least Significant Component Feature (LSCF) quantifies the deviation of adversarial examples from the statistics of benign samples and Hessian Feature (HF) reflects how adversarial examples distort the landscape of the model's optima by measuring the local loss curvature. Empirical results show that our method can achieve an overall ROC AUC of 94.9%, 89.7%, and 94.6% on CIFAR10, CIFAR100, and SVHN, respectively, and has comparable performance to adversarial detectors trained with adversarial examples on most of the attacks.

摘要: 近年来，越来越多的对抗性攻击使攻击者相对于防御者具有优势，因为防御者必须在知道攻击类型后培训检测器，并且需要维护许多模型，以确保在检测任何即将到来的攻击时具有良好的性能。我们提出了一种结束攻击者和防御者之间的拉锯战的方法，将对抗性攻击检测视为一个异常检测问题，使得检测器对攻击是不可知的。我们从两个方面对对抗性扰动造成的统计偏差进行了量化。最低有效成分特征(LSCF)量化了对抗性样本与良性样本统计的偏差，而海森特征(HF)则通过测量局部损失曲率来反映对抗性样本如何扭曲模型的最优解。实验结果表明，我们的方法在CIFAR10、CIFAR100和SVHN上的总体ROC AUC分别达到94.9%、89.7%和94.6%，并且在大多数攻击上具有与使用对抗性实例训练的对抗性检测器相当的性能。



## **44. Generating End-to-End Adversarial Examples for Malware Classifiers Using Explainability**

使用可解释性为恶意软件分类器生成端到端对抗性示例 cs.CR

Accepted as a conference paper at IJCNN 2020

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2009.13243v2)

**Authors**: Ishai Rosenberg, Shai Meir, Jonathan Berrebi, Ilay Gordon, Guillaume Sicard, Eli David

**Abstracts**: In recent years, the topic of explainable machine learning (ML) has been extensively researched. Up until now, this research focused on regular ML users use-cases such as debugging a ML model. This paper takes a different posture and show that adversaries can leverage explainable ML to bypass multi-feature types malware classifiers. Previous adversarial attacks against such classifiers only add new features and not modify existing ones to avoid harming the modified malware executable's functionality. Current attacks use a single algorithm that both selects which features to modify and modifies them blindly, treating all features the same. In this paper, we present a different approach. We split the adversarial example generation task into two parts: First we find the importance of all features for a specific sample using explainability algorithms, and then we conduct a feature-specific modification, feature-by-feature. In order to apply our attack in black-box scenarios, we introduce the concept of transferability of explainability, that is, applying explainability algorithms to different classifiers using different features subsets and trained on different datasets still result in a similar subset of important features. We conclude that explainability algorithms can be leveraged by adversaries and thus the advocates of training more interpretable classifiers should consider the trade-off of higher vulnerability of those classifiers to adversarial attacks.

摘要: 近年来，可解释机器学习得到了广泛的研究。到目前为止，这项研究主要针对常规的ML用户用例，比如调试一个ML模型。本文采取了一种不同的姿态，并展示了攻击者可以利用可解释的ML绕过多特征类型的恶意软件分类器。以前针对此类分类器的敌意攻击只添加新功能，而不修改现有功能，以避免损害修改后的恶意软件可执行文件的功能。当前的攻击使用单一的算法，既选择要修改的特征，又盲目地修改它们，对所有特征一视同仁。在本文中，我们提出了一种不同的方法。我们将对抗性示例生成任务分为两部分：首先使用可解释性算法找出特定样本中所有特征的重要性，然后逐个特征地进行特定特征的修改。为了将我们的攻击应用到黑盒场景中，我们引入了可解释性的概念，即使用不同的特征子集对不同的分类器应用可解释性算法，并在不同的数据集上进行训练，仍然会产生相似的重要特征子集。我们的结论是，可解释性算法可以被对手利用，因此训练更多可解释分类器的倡导者应该考虑这些分类器对对手攻击的更高脆弱性的权衡。



## **45. Anti-Forgery: Towards a Stealthy and Robust DeepFake Disruption Attack via Adversarial Perceptual-aware Perturbations**

防伪：通过对抗性感知扰动实现隐形且强大的DeepFake中断攻击 cs.CR

Accepted by IJCAI 2022

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2206.00477v1)

**Authors**: Run Wang, Ziheng Huang, Zhikai Chen, Li Liu, Jing Chen, Lina Wang

**Abstracts**: DeepFake is becoming a real risk to society and brings potential threats to both individual privacy and political security due to the DeepFaked multimedia are realistic and convincing. However, the popular DeepFake passive detection is an ex-post forensics countermeasure and failed in blocking the disinformation spreading in advance. To address this limitation, researchers study the proactive defense techniques by adding adversarial noises into the source data to disrupt the DeepFake manipulation. However, the existing studies on proactive DeepFake defense via injecting adversarial noises are not robust, which could be easily bypassed by employing simple image reconstruction revealed in a recent study MagDR.   In this paper, we investigate the vulnerability of the existing forgery techniques and propose a novel \emph{anti-forgery} technique that helps users protect the shared facial images from attackers who are capable of applying the popular forgery techniques. Our proposed method generates perceptual-aware perturbations in an incessant manner which is vastly different from the prior studies by adding adversarial noises that is sparse. Experimental results reveal that our perceptual-aware perturbations are robust to diverse image transformations, especially the competitive evasion technique, MagDR via image reconstruction. Our findings potentially open up a new research direction towards thorough understanding and investigation of perceptual-aware adversarial attack for protecting facial images against DeepFakes in a proactive and robust manner. We open-source our tool to foster future research. Code is available at https://github.com/AbstractTeen/AntiForgery/.

摘要: DeepFake正在成为一个真正的社会风险，并给个人隐私和政治安全带来潜在的威胁，因为DeepFak的多媒体是真实和令人信服的。然而，流行的DeepFake被动探测是一种事后取证对策，未能提前阻止虚假信息的传播。为了解决这一局限性，研究人员研究了主动防御技术，通过在源数据中添加对抗性噪声来破坏DeepFake的操纵。然而，现有的通过注入对抗性噪声的主动DeepFake防御的研究并不稳健，这可以通过最近的一项研究MagDR揭示的简单图像重建来容易地绕过。本文研究了现有伪造技术的脆弱性，并提出了一种新的防伪技术，帮助用户保护共享的人脸图像免受攻击者的攻击，攻击者能够应用流行的伪造技术。我们提出的方法以一种不间断的方式产生感知扰动，这与以往的研究通过添加稀疏的对抗性噪声而有很大不同。实验结果表明，我们的感知扰动对不同的图像变换，特别是竞争规避技术，即通过图像重建的MagDR具有很强的鲁棒性。我们的发现可能为深入理解和研究感知感知的敌意攻击以主动和稳健的方式保护面部图像免受DeepFake攻击开辟了新的研究方向。我们将我们的工具开源，以促进未来的研究。代码可在https://github.com/AbstractTeen/AntiForgery/.上找到



## **46. PerDoor: Persistent Non-Uniform Backdoors in Federated Learning using Adversarial Perturbations**

PerDoor：使用对抗性扰动的联合学习中持久的非一致后门 cs.CR

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2205.13523v2)

**Authors**: Manaar Alam, Esha Sarkar, Michail Maniatakos

**Abstracts**: Federated Learning (FL) enables numerous participants to train deep learning models collaboratively without exposing their personal, potentially sensitive data, making it a promising solution for data privacy in collaborative training. The distributed nature of FL and unvetted data, however, makes it inherently vulnerable to backdoor attacks: In this scenario, an adversary injects backdoor functionality into the centralized model during training, which can be triggered to cause the desired misclassification for a specific adversary-chosen input. A range of prior work establishes successful backdoor injection in an FL system; however, these backdoors are not demonstrated to be long-lasting. The backdoor functionality does not remain in the system if the adversary is removed from the training process since the centralized model parameters continuously mutate during successive FL training rounds. Therefore, in this work, we propose PerDoor, a persistent-by-construction backdoor injection technique for FL, driven by adversarial perturbation and targeting parameters of the centralized model that deviate less in successive FL rounds and contribute the least to the main task accuracy. An exhaustive evaluation considering an image classification scenario portrays on average $10.5\times$ persistence over multiple FL rounds compared to traditional backdoor attacks. Through experiments, we further exhibit the potency of PerDoor in the presence of state-of-the-art backdoor prevention techniques in an FL system. Additionally, the operation of adversarial perturbation also assists PerDoor in developing non-uniform trigger patterns for backdoor inputs compared to uniform triggers (with fixed patterns and locations) of existing backdoor techniques, which are prone to be easily mitigated.

摘要: 联合学习(FL)使众多参与者能够协作地训练深度学习模型，而不会暴露他们的个人、潜在敏感数据，使其成为协作培训中数据隐私的一种有前途的解决方案。然而，FL和未经审查的数据的分布式性质使其天生就容易受到后门攻击：在这种情况下，对手在训练期间向集中式模型注入后门功能，这可能会被触发，导致对特定对手选择的输入造成所需的错误分类。先前的一系列工作在FL系统中建立了成功的后门注入；然而，这些后门并没有被证明是持久的。如果将对手从训练过程中移除，则后门功能不会保留在系统中，因为集中式模型参数在连续的FL训练轮期间不断变化。因此，在这项工作中，我们提出了PerDoor，这是一种持久的构造后门注入技术，受对手扰动和集中式模型的目标参数的驱动，这些参数在连续的FL轮中偏离较小，对主任务精度的贡献最小。与传统的后门攻击相比，考虑图像分类场景的详尽评估描绘了在多个FL轮上平均花费10.5\x$持久性。通过实验，我们进一步展示了PerDoor在FL系统中存在最先进的后门预防技术时的有效性。此外，对抗性扰动的操作还有助于PerDoor为后门输入开发非统一的触发模式，而不是现有后门技术的统一触发(具有固定的模式和位置)，后者容易被缓解。



## **47. NeuroUnlock: Unlocking the Architecture of Obfuscated Deep Neural Networks**

NeuroUnlock：解锁模糊深度神经网络的体系结构 cs.CR

The definitive Version of Record will be Published in the 2022  International Joint Conference on Neural Networks (IJCNN)

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2206.00402v1)

**Authors**: Mahya Morid Ahmadi, Lilas Alrahis, Alessio Colucci, Ozgur Sinanoglu, Muhammad Shafique

**Abstracts**: The advancements of deep neural networks (DNNs) have led to their deployment in diverse settings, including safety and security-critical applications. As a result, the characteristics of these models have become sensitive intellectual properties that require protection from malicious users. Extracting the architecture of a DNN through leaky side-channels (e.g., memory access) allows adversaries to (i) clone the model, and (ii) craft adversarial attacks. DNN obfuscation thwarts side-channel-based architecture stealing (SCAS) attacks by altering the run-time traces of a given DNN while preserving its functionality. In this work, we expose the vulnerability of state-of-the-art DNN obfuscation methods to these attacks. We present NeuroUnlock, a novel SCAS attack against obfuscated DNNs. Our NeuroUnlock employs a sequence-to-sequence model that learns the obfuscation procedure and automatically reverts it, thereby recovering the original DNN architecture. We demonstrate the effectiveness of NeuroUnlock by recovering the architecture of 200 randomly generated and obfuscated DNNs running on the Nvidia RTX 2080 TI graphics processing unit (GPU). Moreover, NeuroUnlock recovers the architecture of various other obfuscated DNNs, such as the VGG-11, VGG-13, ResNet-20, and ResNet-32 networks. After recovering the architecture, NeuroUnlock automatically builds a near-equivalent DNN with only a 1.4% drop in the testing accuracy. We further show that launching a subsequent adversarial attack on the recovered DNNs boosts the success rate of the adversarial attack by 51.7% in average compared to launching it on the obfuscated versions. Additionally, we propose a novel methodology for DNN obfuscation, ReDLock, which eradicates the deterministic nature of the obfuscation and achieves 2.16X more resilience to the NeuroUnlock attack. We release the NeuroUnlock and the ReDLock as open-source frameworks.

摘要: 深度神经网络(DNN)的进步导致它们在不同的环境中部署，包括安全和安全关键应用。因此，这些模型的特征已成为敏感的知识产权，需要保护其免受恶意用户的攻击。通过泄漏的旁路(例如，存储器访问)提取DNN的体系结构允许攻击者(I)克隆模型和(Ii)精心设计敌意攻击。DNN混淆通过改变给定DNN的运行时踪迹，同时保持其功能，从而阻止基于侧通道的体系结构窃取(SCAS)攻击。在这项工作中，我们暴露了最先进的DNN混淆方法对这些攻击的脆弱性。提出了一种新的针对混淆DNN的SCAS攻击--NeuroUnlock。我们的NeuroUnlock采用了序列到序列的模型，该模型学习混淆过程并自动恢复它，从而恢复原始的DNN架构。我们通过恢复在NVIDIA RTX 2080 TI图形处理单元(GPU)上运行的200个随机生成和混淆的DNN的架构来演示NeuroUnlock的有效性。此外，NeuroUnlock恢复了各种其他模糊DNN的架构，如VGG-11、VGG-13、ResNet-20和ResNet-32网络。在恢复架构后，NeuroUnlock会自动构建一个近乎相同的DNN，而测试精度只会下降1.4%。我们进一步表明，与在混淆版本上发起攻击相比，对恢复的DNN发起后续敌意攻击的成功率平均提高了51.7%。此外，我们还提出了一种新的DNN混淆方法ReDLock，它消除了混淆的确定性，并获得了2.16倍的抗NeuroUnlock攻击的能力。我们将NeuroUnlock和ReDLock作为开源框架发布。



## **48. Support Vector Machines under Adversarial Label Contamination**

对抗性标签污染下的支持向量机 cs.LG

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2206.00352v1)

**Authors**: Huang Xiao, Battista Biggio, Blaine Nelson, Han Xiao, Claudia Eckert, Fabio Roli

**Abstracts**: Machine learning algorithms are increasingly being applied in security-related tasks such as spam and malware detection, although their security properties against deliberate attacks have not yet been widely understood. Intelligent and adaptive attackers may indeed exploit specific vulnerabilities exposed by machine learning techniques to violate system security. Being robust to adversarial data manipulation is thus an important, additional requirement for machine learning algorithms to successfully operate in adversarial settings. In this work, we evaluate the security of Support Vector Machines (SVMs) to well-crafted, adversarial label noise attacks. In particular, we consider an attacker that aims to maximize the SVM's classification error by flipping a number of labels in the training data. We formalize a corresponding optimal attack strategy, and solve it by means of heuristic approaches to keep the computational complexity tractable. We report an extensive experimental analysis on the effectiveness of the considered attacks against linear and non-linear SVMs, both on synthetic and real-world datasets. We finally argue that our approach can also provide useful insights for developing more secure SVM learning algorithms, and also novel techniques in a number of related research areas, such as semi-supervised and active learning.

摘要: 机器学习算法正越来越多地应用于垃圾邮件和恶意软件检测等与安全相关的任务中，尽管它们针对故意攻击的安全特性尚未得到广泛了解。智能和适应性攻击者确实可能利用机器学习技术暴露的特定漏洞来破坏系统安全。因此，对对抗性数据操纵具有健壮性是机器学习算法在对抗性环境中成功运行的一个重要的额外要求。在这项工作中，我们评估了支持向量机(SVMs)对精心设计的对抗性标签噪声攻击的安全性。特别是，我们考虑了一个攻击者，他的目标是通过翻转训练数据中的多个标签来最大化支持向量机的分类错误。我们形式化了相应的最优攻击策略，并利用启发式方法进行求解，以保持计算的复杂性。我们对所考虑的针对线性和非线性支持向量机的攻击的有效性进行了广泛的实验分析，包括在合成数据集和真实数据集上的攻击。最后，我们认为，我们的方法还可以为开发更安全的支持向量机学习算法提供有用的见解，并为半监督和主动学习等相关研究领域提供新的技术。



## **49. A Simple Structure For Building A Robust Model**

一种用于建立稳健模型的简单结构 cs.CV

Accepted by Fifth International Conference on Intelligence Science  (ICIS2022); 10 pages, 3 figures, 4 tables

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2204.11596v2)

**Authors**: Xiao Tan, Jingbo Gao, Ruolin Li

**Abstracts**: As deep learning applications, especially programs of computer vision, are increasingly deployed in our lives, we have to think more urgently about the security of these applications.One effective way to improve the security of deep learning models is to perform adversarial training, which allows the model to be compatible with samples that are deliberately created for use in attacking the model.Based on this, we propose a simple architecture to build a model with a certain degree of robustness, which improves the robustness of the trained network by adding an adversarial sample detection network for cooperative training. At the same time, we design a new data sampling strategy that incorporates multiple existing attacks, allowing the model to adapt to many different adversarial attacks with a single training.We conducted some experiments to test the effectiveness of this design based on Cifar10 dataset, and the results indicate that it has some degree of positive effect on the robustness of the model.Our code could be found at https://github.com/dowdyboy/simple_structure_for_robust_model .

摘要: 随着深度学习应用，特别是计算机视觉应用的日益广泛，我们不得不更加迫切地考虑这些应用的安全性。对抗性训练是提高深度学习模型安全性的有效方法之一，它可以使模型与特意用于攻击模型的样本相兼容。在此基础上，我们提出了一种简单的架构来构建具有一定鲁棒性的模型，通过增加对抗性样本检测网络来进行协作训练，从而提高了训练网络的健壮性。同时，我们设计了一种新的数据采样策略，融合了多种已有的攻击，使得该模型能够通过一次训练来适应多种不同的对手攻击，并基于Cifar10数据集进行了一些实验，结果表明该设计对模型的健壮性有一定的积极作用。我们的代码可以在https://github.com/dowdyboy/simple_structure_for_robust_model上找到。



## **50. Bounding Membership Inference**

边界隶属度推理 cs.LG

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2202.12232v2)

**Authors**: Anvith Thudi, Ilia Shumailov, Franziska Boenisch, Nicolas Papernot

**Abstracts**: Differential Privacy (DP) is the de facto standard for reasoning about the privacy guarantees of a training algorithm. Despite the empirical observation that DP reduces the vulnerability of models to existing membership inference (MI) attacks, a theoretical underpinning as to why this is the case is largely missing in the literature. In practice, this means that models need to be trained with DP guarantees that greatly decrease their accuracy. In this paper, we provide a tighter bound on the positive accuracy (i.e., attack precision) of any MI adversary when a training algorithm provides $\epsilon$-DP or $(\epsilon, \delta)$-DP. Our bound informs the design of a novel privacy amplification scheme, where an effective training set is sub-sampled from a larger set prior to the beginning of training, to greatly reduce the bound on MI accuracy. As a result, our scheme enables DP users to employ looser DP guarantees when training their model to limit the success of any MI adversary; this ensures that the model's accuracy is less impacted by the privacy guarantee. Finally, we discuss implications of our MI bound on the field of machine unlearning.

摘要: 差分隐私(DP)是对训练算法的隐私保证进行推理的事实标准。尽管经验观察表明DP降低了模型对现有成员推理(MI)攻击的脆弱性，但文献中很大程度上缺乏关于为什么会这样的理论基础。在实践中，这意味着需要用DP保证来训练模型，这会大大降低它们的准确性。在本文中，我们给出了当训练算法提供$\epsilon$-DP或$(\epsilon，\Delta)$-DP时，MI对手的正确率(即攻击精度)的一个更严格的界。我们的界提供了一种新的隐私放大方案的设计，其中有效的训练集在训练开始之前从较大的集合中被亚采样，以极大地降低对MI准确率的界。因此，我们的方案允许DP用户在训练他们的模型时采用更宽松的DP保证来限制任何MI对手的成功；这确保了模型的准确性较少地受到隐私保证的影响。最后，我们讨论了我们的MI界在机器遗忘领域的意义。



