# Latest Adversarial Attack Papers
**update at 2025-02-12 10:58:18**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Approximate Energetic Resilience of Nonlinear Systems under Partial Loss of Control Authority**

部分失去控制权下非线性系统的能量弹性 math.OC

20 pages, 1 figure

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07603v1) [paper-pdf](http://arxiv.org/pdf/2502.07603v1)

**Authors**: Ram Padmanabhan, Melkior Ornik

**Abstract**: In this paper, we quantify the resilience of nonlinear dynamical systems by studying the increased energy used by all inputs of a system that suffers a partial loss of control authority, either through actuator malfunctions or through adversarial attacks. To quantify the maximal increase in energy, we introduce the notion of an energetic resilience metric. Prior work in this particular setting considers only simple linear models and not general nonlinear dynamical systems. We first characterize the mean value of the control signal in both the nominal and malfunctioning systems, which allows us to approximate the energy in the control. We then obtain a worst-case approximation of this energy for the malfunctioning system, over all malfunctioning inputs. Based on this approximation, we derive bounds on the energetic resilience metric when control authority is lost over one actuator. A simulation example on an academic nonlinear system demonstrates that the metric is useful in quantifying the resilience of the system without significant conservatism, despite the approximations used in obtaining control energies.

摘要: 在本文中，我们通过研究由于执行器故障或通过对手攻击而遭受部分控制权威丧失的系统的所有输入所使用的增加的能量来量化非线性动力系统的弹性。为了量化能量的最大增长，我们引入了能量弹性度量的概念。在这种特殊情况下，以前的工作只考虑简单的线性模型，而不考虑一般的非线性动力系统。我们首先刻画了标称系统和故障系统中控制信号的平均值，这允许我们近似控制中的能量。然后，我们得到所有故障输入的故障系统的能量的最坏情况的近似值。基于这一近似，我们推导出了当一个执行器失去控制权时能量弹性度量的界。对一个学术非线性系统的仿真实例表明，尽管在获得控制能量时使用了近似，但该度量在量化系统的弹性方面是有用的，而不具有显著的保守性。



## **2. Efficient Image-to-Image Diffusion Classifier for Adversarial Robustness**

具有对抗鲁棒性的高效图像到图像扩散分类器 cs.CV

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2408.08502v2) [paper-pdf](http://arxiv.org/pdf/2408.08502v2)

**Authors**: Hefei Mei, Minjing Dong, Chang Xu

**Abstract**: Diffusion models (DMs) have demonstrated great potential in the field of adversarial robustness, where DM-based defense methods can achieve superior defense capability without adversarial training. However, they all require huge computational costs due to the usage of large-scale pre-trained DMs, making it difficult to conduct full evaluation under strong attacks and compare with traditional CNN-based methods. Simply reducing the network size and timesteps in DMs could significantly harm the image generation quality, which invalidates previous frameworks. To alleviate this issue, we redesign the diffusion framework from generating high-quality images to predicting distinguishable image labels. Specifically, we employ an image translation framework to learn many-to-one mapping from input samples to designed orthogonal image labels. Based on this framework, we introduce an efficient Image-to-Image diffusion classifier with a pruned U-Net structure and reduced diffusion timesteps. Besides the framework, we redesign the optimization objective of DMs to fit the target of image classification, where a new classification loss is incorporated in the DM-based image translation framework to distinguish the generated label from those of other classes. We conduct sufficient evaluations of the proposed classifier under various attacks on popular benchmarks. Extensive experiments show that our method achieves better adversarial robustness with fewer computational costs than DM-based and CNN-based methods. The code is available at https://github.com/hfmei/IDC

摘要: 扩散模型在对抗鲁棒性领域显示出了巨大的潜力，基于扩散模型的防御方法可以在不需要对手训练的情况下获得优越的防御能力。然而，由于它们都需要使用大规模的预先训练的DM，因此它们都需要巨大的计算代价，这使得在强攻击下进行充分评估并与传统的基于CNN的方法进行比较是困难的。简单地减少DM中的网络大小和时间步长可能会严重损害映像生成质量，从而使之前的框架失效。为了缓解这个问题，我们重新设计了扩散框架，从生成高质量的图像到预测可区分的图像标签。具体地说，我们使用一个图像转换框架来学习从输入样本到设计的正交图像标签的多对一映射。基于该框架，我们提出了一种高效的图像到图像扩散分类器，该分类器具有修剪的U网结构和减少的扩散时间。除了该框架外，我们还重新设计了DMS的优化目标，以适应图像分类的目标，其中在基于DM的图像翻译框架中引入了新的分类损失，以区分生成的标签和其他类别的标签。在各种针对流行基准的攻击下，我们对提出的分类器进行了充分的评估。大量实验表明，与基于DM的方法和基于CNN的方法相比，我们的方法具有更好的对抗健壮性和更少的计算代价。代码可在https://github.com/hfmei/IDC上获得



## **3. RoMA: Robust Malware Attribution via Byte-level Adversarial Training with Global Perturbations and Adversarial Consistency Regularization**

RoMA：通过具有全局扰动和对抗一致性正规化的字节级对抗训练来实现稳健的恶意软件归因 cs.CR

13 pages, 4 figures

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07492v1) [paper-pdf](http://arxiv.org/pdf/2502.07492v1)

**Authors**: Yuxia Sun, Huihong Chen, Jingcai Guo, Aoxiang Sun, Zhetao Li, Haolin Liu

**Abstract**: Attributing APT (Advanced Persistent Threat) malware to their respective groups is crucial for threat intelligence and cybersecurity. However, APT adversaries often conceal their identities, rendering attribution inherently adversarial. Existing machine learning-based attribution models, while effective, remain highly vulnerable to adversarial attacks. For example, the state-of-the-art byte-level model MalConv sees its accuracy drop from over 90% to below 2% under PGD (projected gradient descent) attacks. Existing gradient-based adversarial training techniques for malware detection or image processing were applied to malware attribution in this study, revealing that both robustness and training efficiency require significant improvement. To address this, we propose RoMA, a novel single-step adversarial training approach that integrates global perturbations to generate enhanced adversarial samples and employs adversarial consistency regularization to improve representation quality and resilience. A novel APT malware dataset named AMG18, with diverse samples and realistic class imbalances, is introduced for evaluation. Extensive experiments show that RoMA significantly outperforms seven competing methods in both adversarial robustness (e.g., achieving over 80% robust accuracy-more than twice that of the next-best method under PGD attacks) and training efficiency (e.g., more than twice as fast as the second-best method in terms of accuracy), while maintaining superior standard accuracy in non-adversarial scenarios.

摘要: 将APT(高级持续威胁)恶意软件归因于各自的组织对于威胁情报和网络安全至关重要。然而，聪明的对手往往隐藏自己的身份，使归因具有内在的对抗性。现有的基于机器学习的归因模型虽然有效，但仍然非常容易受到对手的攻击。例如，最先进的字节级模型MalConv在PGD(投影梯度下降)攻击下的准确率从90%以上下降到2%以下。将已有的基于梯度的恶意软件检测或图像处理的对抗性训练技术应用到恶意软件属性识别中，发现无论是稳健性还是训练效率都需要显著提高。为了解决这一问题，我们提出了一种新颖的单步对抗性训练方法，该方法结合全局扰动来生成增强的对抗性样本，并使用对抗性一致性正则化来提高表示质量和韧性。引入了一个新的APT恶意软件数据集AMG18，该数据集具有多样化的样本和真实的类别不平衡。大量实验表明，在对抗性稳健性(例如，在PGD攻击下达到80%以上的健壮性--是次佳方法的两倍多)和训练效率(例如，在准确率方面是次佳方法的两倍以上)方面，ROMA显著优于七种竞争方法，同时在非对抗性场景中保持了卓越的标准准确率。



## **4. Mining Power Destruction Attacks in the Presence of Petty-Compliant Mining Pools**

存在小兼容性采矿设备的情况下的采矿电力破坏攻击 cs.CR

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07410v1) [paper-pdf](http://arxiv.org/pdf/2502.07410v1)

**Authors**: Roozbeh Sarenche, Svetla Nikova, Bart Preneel

**Abstract**: Bitcoin's security relies on its Proof-of-Work consensus, where miners solve puzzles to propose blocks. The puzzle's difficulty is set by the difficulty adjustment mechanism (DAM), based on the network's available mining power. Attacks that destroy some portion of mining power can exploit the DAM to lower difficulty, making such attacks profitable. In this paper, we analyze three types of mining power destruction attacks in the presence of petty-compliant mining pools: selfish mining, bribery, and mining power distraction attacks. We analyze selfish mining while accounting for the distribution of mining power among pools, a factor often overlooked in the literature. Our findings indicate that selfish mining can be more destructive when the non-adversarial mining share is well distributed among pools. We also introduce a novel bribery attack, where the adversarial pool bribes petty-compliant pools to orphan others' blocks. For small pools, we demonstrate that the bribery attack can dominate strategies like selfish mining or undercutting. Lastly, we present the mining distraction attack, where the adversarial pool incentivizes petty-compliant pools to abandon Bitcoin's puzzle and mine for a simpler puzzle, thus wasting some part of their mining power. Similar to the previous attacks, this attack can lower the mining difficulty, but with the difference that it does not generate any evidence of mining power destruction, such as orphan blocks.

摘要: 比特币的安全性依赖于其工作证明共识，即矿工通过解决谜题来提出块。谜题的难度是由难度调整机制(大坝)根据网络的可用采矿力设定的。摧毁部分采矿力量的攻击可以利用大坝来降低难度，使此类攻击有利可图。在本文中，我们分析了三种类型的矿权破坏攻击：自私开采、贿赂和矿权分散攻击。我们分析了自私的采矿，同时考虑了矿权在不同矿池之间的分配，这是文献中经常被忽视的一个因素。我们的发现表明，当非对抗性的采矿份额在池中均匀分布时，自私开采可能会更具破坏性。我们还引入了一种新的贿赂攻击，在这种攻击中，对手池贿赂符合小额规则的池以孤立他人的块。对于较小的资金池，我们证明了贿赂攻击可以主导自私开采或偷工减料等策略。最后，我们提出了挖掘分心攻击，在这种攻击中，敌意的池激励遵守规则的池放弃比特币的谜题，转而使用更简单的谜题，从而浪费了他们的部分挖掘力。与之前的攻击类似，这次攻击可以降低采矿难度，但不同的是，它不会产生任何矿权破坏的证据，如孤儿区块。



## **5. Enhancing Security and Privacy in Federated Learning using Low-Dimensional Update Representation and Proximity-Based Defense**

使用低维更新表示和基于邻近度的防御增强联邦学习中的安全性和隐私 cs.CR

14 pages

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2405.18802v2) [paper-pdf](http://arxiv.org/pdf/2405.18802v2)

**Authors**: Wenjie Li, Kai Fan, Jingyuan Zhang, Hui Li, Wei Yang Bryan Lim, Qiang Yang

**Abstract**: Federated Learning (FL) is a promising privacy-preserving machine learning paradigm that allows data owners to collaboratively train models while keeping their data localized. Despite its potential, FL faces challenges related to the trustworthiness of both clients and servers, particularly against curious or malicious adversaries. In this paper, we introduce a novel framework named \underline{F}ederated \underline{L}earning with Low-Dimensional \underline{U}pdate \underline{R}epresentation and \underline{P}roximity-Based defense (FLURP), designed to address privacy preservation and resistance to Byzantine attacks in distributed learning environments. FLURP employs $\mathsf{LinfSample}$ method, enabling clients to compute the $l_{\infty}$ norm across sliding windows of updates, resulting in a Low-Dimensional Update Representation (LUR). Calculating the shared distance matrix among LURs, rather than updates, significantly reduces the overhead of Secure Multi-Party Computation (SMPC) by three orders of magnitude while effectively distinguishing between benign and poisoned updates. Additionally, FLURP integrates a privacy-preserving proximity-based defense mechanism utilizing optimized SMPC protocols to minimize communication rounds. Our experiments demonstrate FLURP's effectiveness in countering Byzantine adversaries with low communication and runtime overhead. FLURP offers a scalable framework for secure and reliable FL in distributed environments, facilitating its application in scenarios requiring robust data management and security.

摘要: 联合学习(FL)是一种很有前途的隐私保护机器学习范例，允许数据所有者在保持数据本地化的同时协作训练模型。尽管有潜力，FL仍面临着与客户端和服务器的可信性相关的挑战，特别是在面对好奇或恶意的对手时。为了解决分布式学习环境中隐私保护和抵抗拜占庭攻击的问题，本文提出了一种新的框架-FURPFLURP使用$\mathsf{LinfSample}$方法，使客户端能够跨更新的滑动窗口计算$L_{\inty}$范数，从而产生低维更新表示(LUR)。计算LURs之间的共享距离矩阵，而不是更新，显著减少了安全多方计算(SMPC)的开销三个数量级，同时有效地区分了良性和有毒更新。此外，FLURP利用优化的SMPC协议集成了基于隐私保护的邻近防御机制，以最大限度地减少通信轮次。我们的实验证明了FLURP以较低的通信和运行时间开销对抗拜占庭攻击的有效性。FLURP为分布式环境中安全可靠的FL提供了一个可扩展的框架，促进了其在需要强大的数据管理和安全的场景中的应用。



## **6. CAT: Contrastive Adversarial Training for Evaluating the Robustness of Protective Perturbations in Latent Diffusion Models**

CAT：用于评估潜在扩散模型中保护性扰动稳健性的对比对抗训练 cs.CV

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07225v1) [paper-pdf](http://arxiv.org/pdf/2502.07225v1)

**Authors**: Sen Peng, Mingyue Wang, Jianfei He, Jijia Yang, Xiaohua Jia

**Abstract**: Latent diffusion models have recently demonstrated superior capabilities in many downstream image synthesis tasks. However, customization of latent diffusion models using unauthorized data can severely compromise the privacy and intellectual property rights of data owners. Adversarial examples as protective perturbations have been developed to defend against unauthorized data usage by introducing imperceptible noise to customization samples, preventing diffusion models from effectively learning them. In this paper, we first reveal that the primary reason adversarial examples are effective as protective perturbations in latent diffusion models is the distortion of their latent representations, as demonstrated through qualitative and quantitative experiments. We then propose the Contrastive Adversarial Training (CAT) utilizing adapters as an adaptive attack against these protection methods, highlighting their lack of robustness. Extensive experiments demonstrate that our CAT method significantly reduces the effectiveness of protective perturbations in customization configurations, urging the community to reconsider and enhance the robustness of existing protective perturbation methods. Code is available at \hyperlink{here}{https://github.com/senp98/CAT}.

摘要: 最近，潜扩散模型在许多下游图像合成任务中表现出了优越的性能。然而，使用未经授权的数据定制潜在扩散模型可能会严重损害数据所有者的隐私和知识产权。作为保护性扰动的对抗性例子已经被开发出来，通过向定制样本引入不可察觉的噪声来防御未经授权的数据使用，从而防止扩散模型有效地学习它们。在这篇文章中，我们首先揭示了对抗性例子在潜在扩散模型中作为保护性扰动有效的主要原因是其潜在表示的扭曲，通过定性和定量实验证明了这一点。然后，我们提出了利用适配器作为对这些保护方法的自适应攻击的对比性对抗训练(CAT)，突出了它们的健壮性不足。大量的实验表明，我们的CAT方法显著降低了定制配置中保护扰动的有效性，促使社区重新考虑并增强现有保护扰动方法的健壮性。代码可在\hyperlink{here}{https://github.com/senp98/CAT}.上找到



## **7. LUNAR: LLM Unlearning via Neural Activation Redirection**

LUNAR：LLM通过神经激活重定向消除学习 cs.LG

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07218v1) [paper-pdf](http://arxiv.org/pdf/2502.07218v1)

**Authors**: William F. Shen, Xinchi Qiu, Meghdad Kurmanji, Alex Iacob, Lorenzo Sani, Yihong Chen, Nicola Cancedda, Nicholas D. Lane

**Abstract**: Large Language Models (LLMs) benefit from training on ever larger amounts of textual data, but as a result, they increasingly incur the risk of leaking private information. The ability to selectively remove knowledge from LLMs is, therefore, a highly desirable capability. In this paper, we propose LUNAR, a novel unlearning methodology grounded in the Linear Representation Hypothesis. LUNAR operates by redirecting the representations of unlearned data to regions that trigger the model's inherent ability to express its inability to answer. LUNAR achieves state-of-the-art unlearning performance while significantly enhancing the controllability of the unlearned model during inference. Specifically, LUNAR achieves between 2.9x to 11.7x improvements on combined "unlearning efficacy" and "model utility" score ("Deviation Score") on the PISTOL dataset across various base models. We also demonstrate, through quantitative analysis and qualitative examples, LUNAR's superior controllability in generating coherent and contextually aware responses, mitigating undesired side effects of existing methods. Moreover, we demonstrate that LUNAR is robust against white-box adversarial attacks and versatile in handling real-world scenarios, such as processing sequential unlearning requests.

摘要: 大型语言模型(LLM)受益于对越来越多的文本数据进行培训，但结果是，它们越来越多地招致泄露私人信息的风险。因此，有选择地从LLM中移除知识的能力是一种非常理想的能力。在本文中，我们提出了一种基于线性表征假设的去学习方法--LUNAR。LUNAR通过将未学习数据的表示重定向到触发模型表达其无法回答问题的固有能力的区域来运行。LUNAR实现了最先进的遗忘性能，同时显著增强了推理过程中未学习模型的可控性。具体地说，在各种基本型号的手枪数据集上，LUNAR在组合的“遗忘效能”和“模型效用”分数(“偏差分数”)上取得了2.9倍到11.7倍的改进。我们还通过定量分析和定性例子证明，月球在产生连贯的和上下文感知的响应方面具有优越的可控性，减轻了现有方法的不良副作用。此外，我们还证明了LUNAR对白盒攻击具有很强的健壮性，并且在处理真实场景(如处理顺序遗忘请求)方面具有很强的通用性。



## **8. SMAB: MAB based word Sensitivity Estimation Framework and its Applications in Adversarial Text Generation**

SMAB：基于MAB的词敏感度估计框架及其在对抗性文本生成中的应用 cs.CL

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.07101v1) [paper-pdf](http://arxiv.org/pdf/2502.07101v1)

**Authors**: Saurabh Kumar Pandey, Sachin Vashistha, Debrup Das, Somak Aditya, Monojit Choudhury

**Abstract**: To understand the complexity of sequence classification tasks, Hahn et al. (2021) proposed sensitivity as the number of disjoint subsets of the input sequence that can each be individually changed to change the output. Though effective, calculating sensitivity at scale using this framework is costly because of exponential time complexity. Therefore, we introduce a Sensitivity-based Multi-Armed Bandit framework (SMAB), which provides a scalable approach for calculating word-level local (sentence-level) and global (aggregated) sensitivities concerning an underlying text classifier for any dataset. We establish the effectiveness of our approach through various applications. We perform a case study on CHECKLIST generated sentiment analysis dataset where we show that our algorithm indeed captures intuitively high and low-sensitive words. Through experiments on multiple tasks and languages, we show that sensitivity can serve as a proxy for accuracy in the absence of gold data. Lastly, we show that guiding perturbation prompts using sensitivity values in adversarial example generation improves attack success rate by 15.58%, whereas using sensitivity as an additional reward in adversarial paraphrase generation gives a 12.00% improvement over SOTA approaches. Warning: Contains potentially offensive content.

摘要: 为了理解序列分类任务的复杂性，Hahn等人。(2021)提出的敏感度是输入序列的不相交子集的数目，每个子集都可以单独改变以改变输出。虽然有效，但由于指数时间复杂性，使用该框架在规模上计算敏感度代价高昂。因此，我们引入了一种基于敏感度的多臂Bandit框架(SMAB)，它提供了一种可扩展的方法来计算关于任何数据集的底层文本分类器的词级局部(句子级)和全局(聚合)敏感度。我们通过各种应用证明了我们方法的有效性。我们在核对表生成的情感分析数据集上进行了实例研究，结果表明，我们的算法确实能够直观地捕获高敏感度和低敏感度的词。通过对多个任务和语言的实验，我们发现，在没有GOLD数据的情况下，敏感度可以作为准确度的代理。最后，我们展示了在对抗性范例生成中使用敏感度来引导扰动提示将攻击成功率提高了15.58%，而在对抗性释义生成中使用敏感度作为额外奖励则比SOTA方法提高了12.00%。警告：包含潜在的攻击性内容。



## **9. DROP: Poison Dilution via Knowledge Distillation for Federated Learning**

Drop：通过联邦学习的知识蒸馏进行毒药稀释 cs.LG

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.07011v1) [paper-pdf](http://arxiv.org/pdf/2502.07011v1)

**Authors**: Georgios Syros, Anshuman Suri, Farinaz Koushanfar, Cristina Nita-Rotaru, Alina Oprea

**Abstract**: Federated Learning is vulnerable to adversarial manipulation, where malicious clients can inject poisoned updates to influence the global model's behavior. While existing defense mechanisms have made notable progress, they fail to protect against adversaries that aim to induce targeted backdoors under different learning and attack configurations. To address this limitation, we introduce DROP (Distillation-based Reduction Of Poisoning), a novel defense mechanism that combines clustering and activity-tracking techniques with extraction of benign behavior from clients via knowledge distillation to tackle stealthy adversaries that manipulate low data poisoning rates and diverse malicious client ratios within the federation. Through extensive experimentation, our approach demonstrates superior robustness compared to existing defenses across a wide range of learning configurations. Finally, we evaluate existing defenses and our method under the challenging setting of non-IID client data distribution and highlight the challenges of designing a resilient FL defense in this setting.

摘要: 联合学习很容易受到敌意操纵，恶意客户端可以注入有毒更新来影响全局模型的行为。虽然现有的防御机制已经取得了显著的进展，但它们无法防御旨在根据不同的学习和攻击配置诱导有针对性的后门的对手。为了解决这一局限性，我们引入了Drop(基于蒸馏的毒化减少)，这是一种新型的防御机制，它将集群和活动跟踪技术与通过知识蒸馏从客户中提取良性行为相结合，以应对在联邦内操纵低数据毒化率和不同恶意客户比率的隐蔽对手。通过广泛的实验，与现有的防御相比，我们的方法在广泛的学习配置上表现出了卓越的稳健性。最后，在非IID客户端数据分发的挑战环境下，我们评估了现有的防御措施和我们的方法，并强调了在这种情况下设计弹性FL防御措施所面临的挑战。



## **10. Breaking Quantum Key Distributions under Quantum Switch-Based Attack**

基于量子交换机的攻击下破坏量子密钥分布 quant-ph

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.06780v1) [paper-pdf](http://arxiv.org/pdf/2502.06780v1)

**Authors**: Sumit Nandi, Biswaranjan Panda, Pankaj Agrawal, Arun K Pati

**Abstract**: Quantum key distribution (QKD) enables secure key sharing between distant parties, with several protocols proven resilient against conventional eavesdropping strategies. Here, we introduce a new attack scenario where an eavesdropper, Eve, exploits a quantum switch using the indefinite causal order to intercept and manipulate quantum communication channel. Using multiple metrics such as the information gain, mutual information, and Bell violation, we demonstrate that the presence of a quantum switch significantly compromises QKD security. Our results highlight a previously overlooked vulnerability, emphasizing the need for countermeasures against quantum-controlled adversarial strategies.

摘要: 量子密钥分发（QKD）实现了远程方之间的安全密钥共享，几种协议被证明具有抵御传统窃听策略的能力。在这里，我们引入了一种新的攻击场景，其中窃听者Eve利用量子开关，使用不确定因果顺序来拦截和操纵量子通信通道。使用信息收益、互信息和Bell破坏等多种指标，我们证明量子交换机的存在会显着损害QKD安全性。我们的结果强调了以前被忽视的漏洞，强调了针对量子控制对抗策略的对策的必要性。



## **11. When Witnesses Defend: A Witness Graph Topological Layer for Adversarial Graph Learning**

当证人辩护时：对抗图学习的证人图布局层 cs.LG

Accepted at AAAI 2025

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2409.14161v3) [paper-pdf](http://arxiv.org/pdf/2409.14161v3)

**Authors**: Naheed Anjum Arafat, Debabrota Basu, Yulia Gel, Yuzhou Chen

**Abstract**: Capitalizing on the intuitive premise that shape characteristics are more robust to perturbations, we bridge adversarial graph learning with the emerging tools from computational topology, namely, persistent homology representations of graphs. We introduce the concept of witness complex to adversarial analysis on graphs, which allows us to focus only on the salient shape characteristics of graphs, yielded by the subset of the most essential nodes (i.e., landmarks), with minimal loss of topological information on the whole graph. The remaining nodes are then used as witnesses, governing which higher-order graph substructures are incorporated into the learning process. Armed with the witness mechanism, we design Witness Graph Topological Layer (WGTL), which systematically integrates both local and global topological graph feature representations, the impact of which is, in turn, automatically controlled by the robust regularized topological loss. Given the attacker's budget, we derive the important stability guarantees of both local and global topology encodings and the associated robust topological loss. We illustrate the versatility and efficiency of WGTL by its integration with five GNNs and three existing non-topological defense mechanisms. Our extensive experiments across six datasets demonstrate that WGTL boosts the robustness of GNNs across a range of perturbations and against a range of adversarial attacks. Our datasets and source codes are available at https://github.com/toggled/WGTL.

摘要: 基于形状特征对扰动的鲁棒性更强这一直观前提，我们利用计算拓扑学中的新兴工具，即图的持久同调表示，在对抗性图学习之间架起桥梁。我们将证人复合体的概念引入到图的对抗分析中，使得我们只关注图的显著形状特征，这些特征是由最重要的结点(即地标)的子集产生的，而整个图的拓扑信息损失最小。然后，剩余的节点被用作见证，控制哪些更高阶图的子结构被合并到学习过程中。结合见证机制，我们设计了见证图拓扑层(Witness Graph Topology Layer，WGTL)，该拓扑层系统地集成了局部拓扑图和全局拓扑图的特征表示，其影响由稳健的正则化拓扑损失自动控制。在给定攻击者预算的情况下，我们推导出了局部和全局拓扑编码的重要稳定性保证以及相关的稳健拓扑损失。我们通过与五个GNN和三个现有的非拓扑防御机制的集成来说明WGTL的通用性和有效性。我们在六个数据集上的广泛实验表明，WGTL增强了GNN在一系列扰动和一系列对手攻击下的健壮性。我们的数据集和源代码可在https://github.com/toggled/WGTL.上获得



## **12. Tamper-Resistant Safeguards for Open-Weight LLMs**

开放重量LLM的防篡改保障措施 cs.LG

Website: https://www.tamper-resistant-safeguards.com

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2408.00761v4) [paper-pdf](http://arxiv.org/pdf/2408.00761v4)

**Authors**: Rishub Tamirisa, Bhrugu Bharathi, Long Phan, Andy Zhou, Alice Gatti, Tarun Suresh, Maxwell Lin, Justin Wang, Rowan Wang, Ron Arel, Andy Zou, Dawn Song, Bo Li, Dan Hendrycks, Mantas Mazeika

**Abstract**: Rapid advances in the capabilities of large language models (LLMs) have raised widespread concerns regarding their potential for malicious use. Open-weight LLMs present unique challenges, as existing safeguards lack robustness to tampering attacks that modify model weights. For example, recent works have demonstrated that refusal and unlearning safeguards can be trivially removed with a few steps of fine-tuning. These vulnerabilities necessitate new approaches for enabling the safe release of open-weight LLMs. We develop a method, called TAR, for building tamper-resistant safeguards into open-weight LLMs such that adversaries cannot remove the safeguards even after hundreds of steps of fine-tuning. In extensive evaluations and red teaming analyses, we find that our method greatly improves tamper-resistance while preserving benign capabilities. Our results demonstrate that progress on tamper-resistance is possible, opening up a promising new avenue to improve the safety and security of open-weight LLMs.

摘要: 大型语言模型(LLM)功能的快速发展引起了人们对其潜在恶意使用的广泛关注。开放重量LLM提出了独特的挑战，因为现有的保障措施缺乏对篡改模型权重的篡改攻击的稳健性。例如，最近的研究表明，通过几个步骤的微调，就可以很容易地消除拒绝和遗忘的保障措施。这些漏洞需要新的方法来实现安全释放未加重量的低密度脂蛋白。我们开发了一种名为TAR的方法，用于在开放重量的LLM中构建防篡改保护措施，以便即使在数百个步骤的微调之后，对手也无法移除这些保护措施。在广泛的评估和红团队分析中，我们发现我们的方法在保持良性性能的同时大大提高了防篡改能力。我们的结果表明，在防篡改方面取得进展是可能的，为提高开放重量LLMS的安全性开辟了一条有希望的新途径。



## **13. Exploring Audio Editing Features as User-Centric Privacy Defenses Against Large Language Model(LLM) Based Emotion Inference Attacks**

探索音频编辑功能，以用户为中心的隐私防御基于大型语言模型（LLM）的情感推理攻击 cs.CR

Accepted for presentation(Poster) at PPAI-25: The 6th AAAI Workshop  on Privacy-Preserving Artificial Intelligence

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2501.18727v2) [paper-pdf](http://arxiv.org/pdf/2501.18727v2)

**Authors**: Mohd. Farhan Israk Soumik, W. K. M. Mithsara, Abdur R. Shahid, Ahmed Imteaj

**Abstract**: The rapid proliferation of speech-enabled technologies, including virtual assistants, video conferencing platforms, and wearable devices, has raised significant privacy concerns, particularly regarding the inference of sensitive emotional information from audio data. Existing privacy-preserving methods often compromise usability and security, limiting their adoption in practical scenarios. This paper introduces a novel, user-centric approach that leverages familiar audio editing techniques, specifically pitch and tempo manipulation, to protect emotional privacy without sacrificing usability. By analyzing popular audio editing applications on Android and iOS platforms, we identified these features as both widely available and usable. We rigorously evaluated their effectiveness against a threat model, considering adversarial attacks from diverse sources, including Deep Neural Networks (DNNs), Large Language Models (LLMs), and and reversibility testing. Our experiments, conducted on three distinct datasets, demonstrate that pitch and tempo manipulation effectively obfuscates emotional data. Additionally, we explore the design principles for lightweight, on-device implementation to ensure broad applicability across various devices and platforms.

摘要: 包括虚拟助理、视频会议平台和可穿戴设备在内的语音支持技术的迅速普及引发了人们对隐私的严重担忧，特别是关于从音频数据推断敏感情感信息的问题。现有的隐私保护方法往往会损害可用性和安全性，限制了它们在实际场景中的采用。本文介绍了一种新颖的、以用户为中心的方法，该方法利用熟悉的音频编辑技术，特别是音调和节奏操作，在不牺牲可用性的情况下保护情感隐私。通过分析Android和iOS平台上流行的音频编辑应用程序，我们发现这些功能广泛使用和使用。我们严格评估了它们对威胁模型的有效性，考虑了来自不同来源的对抗性攻击，包括深度神经网络(DNN)、大型语言模型(LLMS)和可逆性测试。我们在三个不同的数据集上进行的实验表明，音调和节奏操作有效地混淆了情感数据。此外，我们还探讨了轻量级设备上实施的设计原则，以确保跨各种设备和平台的广泛适用性。



## **14. LIAR: Leveraging Inference Time Alignment (Best-of-N) to Jailbreak LLMs in Seconds**

LIAR：利用推理时间对齐（N中最佳）以秒为单位越狱LLM cs.CL

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2412.05232v2) [paper-pdf](http://arxiv.org/pdf/2412.05232v2)

**Authors**: James Beetham, Souradip Chakraborty, Mengdi Wang, Furong Huang, Amrit Singh Bedi, Mubarak Shah

**Abstract**: Traditional jailbreaks have successfully exposed vulnerabilities in LLMs, primarily relying on discrete combinatorial optimization, while more recent methods focus on training LLMs to generate adversarial prompts. However, both approaches are computationally expensive and slow, often requiring significant resources to generate a single successful attack. We hypothesize that the inefficiency of these methods arises from an inadequate characterization of the jailbreak problem itself. To address this gap, we approach the jailbreak problem as an alignment problem, leading us to propose LIAR (Leveraging Inference time Alignment to jailbReak), a fast and efficient best-of-N approach tailored for jailbreak attacks. LIAR offers several key advantages: it eliminates the need for additional training, operates in a fully black-box setting, significantly reduces computational overhead, and produces more human-readable adversarial prompts while maintaining competitive attack success rates. Our results demonstrate that a best-of-N approach is a simple yet highly effective strategy for evaluating the robustness of aligned LLMs, achieving attack success rates (ASR) comparable to state-of-the-art methods while offering a 10x improvement in perplexity and a significant speedup in Time-to-Attack, reducing execution time from tens of hours to seconds. Additionally, We also provide sub-optimality guarantees for the proposed LIAR. Our work highlights the potential of efficient, alignment-based jailbreak strategies for assessing and stress-testing AI safety measures.

摘要: 传统的越狱已经成功地暴露了LLMS中的漏洞，主要依赖于离散组合优化，而最近的方法则专注于训练LLMS生成对抗性提示。然而，这两种方法在计算上都很昂贵且速度很慢，通常需要大量资源才能生成一次成功的攻击。我们假设，这些方法的低效源于对越狱问题本身的不充分描述。为了解决这一差距，我们将越狱问题视为一个对齐问题，导致我们提出了LIAR(利用推理时间对齐越狱)，这是一种为越狱攻击量身定做的快速高效的N中之最方法。Liar提供了几个关键优势：它消除了对额外培训的需要，在完全黑箱设置下运行，显著减少了计算开销，并在保持有竞争力的攻击成功率的同时生成更易读的对抗性提示。我们的结果表明，N中最佳方法是一种简单但高效的策略来评估对齐的LLM的健壮性，实现了与最先进方法相当的攻击成功率(ASR)，同时将困惑程度提高了10倍，攻击时间显著加快，执行时间从数十小时减少到数秒。此外，我们还为被提议的说谎者提供次最优保证。我们的工作突出了高效、基于路线的越狱战略在评估和压力测试人工智能安全措施方面的潜力。



## **15. Automatic ISA analysis for Secure Context Switching**

用于安全上下文切换的自动ISA分析 cs.OS

15 pages, 6 figures, 2 tables, 4 listings

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.06609v1) [paper-pdf](http://arxiv.org/pdf/2502.06609v1)

**Authors**: Neelu S. Kalani, Thomas Bourgeat, Guerney D. H. Hunt, Wojciech Ozga

**Abstract**: Instruction set architectures are complex, with hundreds of registers and instructions that can modify dozens of them during execution, variably on each instance. Prose-style ISA specifications struggle to capture these intricacies of the ISAs, where often the important details about a single register are spread out across hundreds of pages of documentation. Ensuring that all ISA-state is swapped in context switch implementations of privileged software requires meticulous examination of these pages. This manual process is tedious and error-prone.   We propose a tool called Sailor that leverages machine-readable ISA specifications written in Sail to automate this task. Sailor determines the ISA-state necessary to swap during the context switch using the data collected from Sail and a novel algorithm to classify ISA-state as security-sensitive. Using Sailor's output, we identify three different classes of mishandled ISA-state across four open-source confidential computing systems. We further reveal five distinct security vulnerabilities that can be exploited using the mishandled ISA-state. This research exposes an often overlooked attack surface that stems from mishandled ISA-state, enabling unprivileged adversaries to exploit system vulnerabilities.

摘要: 指令集体系结构很复杂，有数百个寄存器和指令，这些寄存器和指令可以在执行期间修改数十个寄存器和指令，这些寄存器和指令在每个实例上都是可变的。散文式的ISA规范很难捕捉到ISA的这些错综复杂之处，其中关于单个寄存器的重要细节通常分布在数百页的文档中。要确保在特权软件的上下文切换实现中交换所有ISA状态，需要仔细检查这些页面。此手动过程繁琐且容易出错。我们提出了一个名为Sailor的工具，它利用用Sail编写的机器可读的ISA规范来自动执行这项任务。Silor使用从SAIL收集的数据和一种新的算法将ISA状态分类为安全敏感的，来确定在上下文切换期间交换所需的ISA状态。使用Sailor的输出，我们确定了四个开源机密计算系统中三种不同类别的处理不当的ISA-STATE。我们进一步揭示了五个不同的安全漏洞，可以使用处理不当的ISA状态来利用这些漏洞。这项研究揭露了一个经常被忽视的攻击面，这个攻击面源于处理不当的ISA状态，使没有特权的对手能够利用系统漏洞。



## **16. Krum Federated Chain (KFC): Using blockchain to defend against adversarial attacks in Federated Learning**

Krum Federated Chain（KFC）：使用区块链防御联邦学习中的对抗性攻击 cs.LG

Submitted to Neural Networks

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.06917v1) [paper-pdf](http://arxiv.org/pdf/2502.06917v1)

**Authors**: Mario García-Márquez, Nuria Rodríguez-Barroso, M. Victoria Luzón, Francisco Herrera

**Abstract**: Federated Learning presents a nascent approach to machine learning, enabling collaborative model training across decentralized devices while safeguarding data privacy. However, its distributed nature renders it susceptible to adversarial attacks. Integrating blockchain technology with Federated Learning offers a promising avenue to enhance security and integrity. In this paper, we tackle the potential of blockchain in defending Federated Learning against adversarial attacks. First, we test Proof of Federated Learning, a well known consensus mechanism designed ad-hoc to federated contexts, as a defense mechanism demonstrating its efficacy against Byzantine and backdoor attacks when at least one miner remains uncompromised. Second, we propose Krum Federated Chain, a novel defense strategy combining Krum and Proof of Federated Learning, valid to defend against any configuration of Byzantine or backdoor attacks, even when all miners are compromised. Our experiments conducted on image classification datasets validate the effectiveness of our proposed approaches.

摘要: 联合学习提供了一种新的机器学习方法，在保护数据隐私的同时，实现了跨分散设备的协作模型培训。然而，它的分布式特性使其容易受到对手的攻击。将区块链技术与联合学习相结合为增强安全性和完整性提供了一条很有前途的途径。在本文中，我们讨论了区块链在保护联邦学习免受对手攻击方面的潜力。首先，我们测试了联合学习的证据，这是一种著名的共识机制，专为联合环境而设计，作为一种防御机制，在至少一个矿工保持不受危害的情况下，展示了其对抗拜占庭和后门攻击的有效性。其次，我们提出了Krum联邦链，这是一种结合了Krum和联合学习证明的新型防御策略，即使在所有矿工都被攻破的情况下，也能有效地防御任何配置的拜占庭攻击或后门攻击。我们在图像分类数据集上进行的实验验证了所提方法的有效性。



## **17. Robust Watermarks Leak: Channel-Aware Feature Extraction Enables Adversarial Watermark Manipulation**

稳健的水印泄露：队列感知特征提取实现对抗性水印操纵 cs.CV

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.06418v1) [paper-pdf](http://arxiv.org/pdf/2502.06418v1)

**Authors**: Zhongjie Ba, Yitao Zhang, Peng Cheng, Bin Gong, Xinyu Zhang, Qinglong Wang, Kui Ren

**Abstract**: Watermarking plays a key role in the provenance and detection of AI-generated content. While existing methods prioritize robustness against real-world distortions (e.g., JPEG compression and noise addition), we reveal a fundamental tradeoff: such robust watermarks inherently improve the redundancy of detectable patterns encoded into images, creating exploitable information leakage. To leverage this, we propose an attack framework that extracts leakage of watermark patterns through multi-channel feature learning using a pre-trained vision model. Unlike prior works requiring massive data or detector access, our method achieves both forgery and detection evasion with a single watermarked image. Extensive experiments demonstrate that our method achieves a 60\% success rate gain in detection evasion and 51\% improvement in forgery accuracy compared to state-of-the-art methods while maintaining visual fidelity. Our work exposes the robustness-stealthiness paradox: current "robust" watermarks sacrifice security for distortion resistance, providing insights for future watermark design.

摘要: 水印在人工智能生成的内容的来源和检测中起着关键作用。虽然现有的方法优先考虑对真实世界的扭曲(例如，JPEG压缩和噪声添加)的稳健性，但我们揭示了一个基本的权衡：这种健壮的水印内在地提高了编码到图像中的可检测模式的冗余性，造成了可利用的信息泄漏。为了利用这一点，我们提出了一种攻击框架，该框架使用预先训练的视觉模型通过多通道特征学习来提取水印模式的泄漏。与以往需要访问大量数据或检测器的工作不同，我们的方法实现了对单个水印图像的伪造和检测规避。大量实验表明，该方法在保持视觉保真度的前提下，与现有方法相比，检测规避成功率提高了60%，伪造准确率提高了51%。我们的工作揭示了健壮性-隐蔽性悖论：当前的“健壮性”水印以牺牲安全性为代价来抵抗失真，为未来的水印设计提供了见解。



## **18. Hyperparameters in Score-Based Membership Inference Attacks**

基于分数的成员推断攻击中的超参数 cs.LG

This work has been accepted for publication in the 3rd IEEE  Conference on Secure and Trustworthy Machine Learning (SaTML'25). The final  version will be available on IEEE Xplore

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.06374v1) [paper-pdf](http://arxiv.org/pdf/2502.06374v1)

**Authors**: Gauri Pradhan, Joonas Jälkö, Marlon Tobaben, Antti Honkela

**Abstract**: Membership Inference Attacks (MIAs) have emerged as a valuable framework for evaluating privacy leakage by machine learning models. Score-based MIAs are distinguished, in particular, by their ability to exploit the confidence scores that the model generates for particular inputs. Existing score-based MIAs implicitly assume that the adversary has access to the target model's hyperparameters, which can be used to train the shadow models for the attack. In this work, we demonstrate that the knowledge of target hyperparameters is not a prerequisite for MIA in the transfer learning setting. Based on this, we propose a novel approach to select the hyperparameters for training the shadow models for MIA when the attacker has no prior knowledge about them by matching the output distributions of target and shadow models. We demonstrate that using the new approach yields hyperparameters that lead to an attack near indistinguishable in performance from an attack that uses target hyperparameters to train the shadow models. Furthermore, we study the empirical privacy risk of unaccounted use of training data for hyperparameter optimization (HPO) in differentially private (DP) transfer learning. We find no statistically significant evidence that performing HPO using training data would increase vulnerability to MIA.

摘要: 成员关系推理攻击(MIA)已经成为机器学习模型评估隐私泄露的一个有价值的框架。基于分数的MIA的特别之处在于，它们能够利用模型为特定输入生成的置信度分数。现有的基于分数的MIA隐含地假设对手可以访问目标模型的超参数，这些超参数可以用于训练攻击的影子模型。在这项工作中，我们证明了在迁移学习环境下，目标超参数的知识不是MIA的先决条件。基于此，我们提出了一种新的方法，当攻击者对阴影模型一无所知时，通过匹配目标和阴影模型的输出分布来选择用于训练MIA阴影模型的超参数。我们证明，使用新方法产生的超参数导致的攻击在性能上与使用目标超参数来训练阴影模型的攻击几乎没有区别。此外，我们还研究了在差异私有(DP)迁移学习中使用训练数据进行超参数优化(HPO)时的经验隐私风险。我们没有发现有统计学意义的证据表明，使用训练数据执行HPO会增加MIA的易感性。



## **19. TASAR: Transfer-based Attack on Skeletal Action Recognition**

TASSAR：基于传输的对Skelty动作识别的攻击 cs.CV

arXiv admin note: text overlap with arXiv:2407.08572

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2409.02483v4) [paper-pdf](http://arxiv.org/pdf/2409.02483v4)

**Authors**: Yunfeng Diao, Baiqi Wu, Ruixuan Zhang, Ajian Liu, Xiaoshuai Hao, Xingxing Wei, Meng Wang, He Wang

**Abstract**: Skeletal sequences, as well-structured representations of human behaviors, play a vital role in Human Activity Recognition (HAR). The transferability of adversarial skeletal sequences enables attacks in real-world HAR scenarios, such as autonomous driving, intelligent surveillance, and human-computer interactions. However, most existing skeleton-based HAR (S-HAR) attacks are primarily designed for white-box scenarios and exhibit weak adversarial transferability. Therefore, they cannot be considered true transfer-based S-HAR attacks. More importantly, the reason for this failure remains unclear. In this paper, we study this phenomenon through the lens of loss surface, and find that its sharpness contributes to the weak transferability in S-HAR. Inspired by this observation, we assume and empirically validate that smoothening the rugged loss landscape could potentially improve adversarial transferability in S-HAR. To this end, we propose the first \textbf{T}ransfer-based \textbf{A}ttack on \textbf{S}keletal \textbf{A}ction \textbf{R}ecognition, TASAR. TASAR explores the smoothed model posterior without requiring surrogate re-training, which is achieved by a new post-train Dual Bayesian optimization strategy. Furthermore, unlike previous transfer-based attacks that treat each frame independently and overlook temporal coherence within sequences, TASAR incorporates motion dynamics into the Bayesian attack gradient, effectively disrupting the spatial-temporal coherence of S-HARs. To exhaustively evaluate the effectiveness of existing methods and our method, we build the first large-scale robust S-HAR benchmark, comprising 7 S-HAR models, 10 attack methods, 3 S-HAR datasets and 2 defense methods. Extensive results demonstrate the superiority of TASAR. Our benchmark enables easy comparisons for future studies, with the code available in the supplementary material.

摘要: 骨架序列作为人类行为的良好结构表征，在人类活动识别(HAR)中起着至关重要的作用。对抗性骨架序列的可转移性使攻击能够在真实世界的HAR场景中进行，例如自动驾驶、智能监控和人机交互。然而，现有的大多数基于骨架的HAR(S-HAR)攻击主要是针对白盒场景而设计的，表现出较弱的对抗可转移性。因此，它们不能被认为是真正的基于转移的S-哈尔袭击。更重要的是，这一失败的原因尚不清楚。本文从损失面的角度对这一现象进行了研究，发现损失面的锐性是S-哈尔转移性较弱的原因之一。受到这一观察的启发，我们假设并经验验证，平滑崎岖的损失图景可能会提高S-哈尔的对抗性转移能力。为此，我们提出了第一个基于Textbf{T}转移的Tasar骨骼/Textbf{A}骨骼/Textbf{R}生态识别方法。Tasar探索平滑的后验模型，不需要代理重新训练，这是通过一种新的训练后双贝叶斯优化策略实现的。此外，与以往基于传输的攻击独立对待每一帧并忽略序列内部的时间一致性不同，Tasar将运动动力学融入到贝叶斯攻击梯度中，有效地破坏了S-HARs的时空一致性。为了全面评估已有方法和本文方法的有效性，我们构建了第一个大规模稳健的S-HAR基准，包括7个S-HAR模型、10种攻击方法、3个S-HAR数据集和2种防御方法。广泛的结果证明了Tasar的优越性。我们的基准可以很容易地与补充材料中提供的代码进行比较，以便将来进行研究。



## **20. POEX: Understanding and Mitigating Policy Executable Jailbreak Attacks against Embodied AI**

POEX：了解和缓解针对被授权人工智能的政策可执行越狱攻击 cs.RO

Homepage: https://poex-eai-jailbreak.github.io/

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2412.16633v2) [paper-pdf](http://arxiv.org/pdf/2412.16633v2)

**Authors**: Xuancun Lu, Zhengxian Huang, Xinfeng Li, Xiaoyu ji, Wenyuan Xu

**Abstract**: Embodied AI systems are rapidly evolving due to the integration of LLMs as planning modules, which transform complex instructions into executable policies. However, LLMs are vulnerable to jailbreak attacks, which can generate malicious content. This paper investigates the feasibility and rationale behind applying traditional LLM jailbreak attacks to EAI systems. We aim to answer three questions: (1) Do traditional LLM jailbreak attacks apply to EAI systems? (2) What challenges arise if they do not? and (3) How can we defend against EAI jailbreak attacks? To this end, we first measure existing LLM-based EAI systems using a newly constructed dataset, i.e., the Harmful-RLbench. Our study confirms that traditional LLM jailbreak attacks are not directly applicable to EAI systems and identifies two unique challenges. First, the harmful text does not necessarily constitute harmful policies. Second, even if harmful policies can be generated, they are not necessarily executable by the EAI systems, which limits the potential risk. To facilitate a more comprehensive security analysis, we refine and introduce POEX, a novel red teaming framework that optimizes adversarial suffixes to induce harmful yet executable policies against EAI systems. The design of POEX employs adversarial constraints, policy evaluators, and suffix optimization to ensure successful policy execution while evading safety detection inside an EAI system. Experiments on the real-world robotic arm and simulator using Harmful-RLbench demonstrate the efficacy, highlighting severe safety vulnerabilities and high transferability across models. Finally, we propose prompt-based and model-based defenses, achieving an 85% success rate in mitigating attacks and enhancing safety awareness in EAI systems. Our findings underscore the urgent need for robust security measures to ensure the safe deployment of EAI in critical applications.

摘要: 由于将LLM作为规划模块进行集成，将复杂的指令转换为可执行的策略，因此具体化人工智能系统正在快速发展。然而，LLMS容易受到越狱攻击，这可能会生成恶意内容。本文研究了将传统的LLM越狱攻击应用于EAI系统的可行性和基本原理。我们的目标是回答三个问题：(1)传统的LLM越狱攻击适用于EAI系统吗？(2)如果不适用，会带来什么挑战？以及(3)我们如何防御EAI越狱攻击？为此，我们首先使用一个新构建的数据集--有害RLbench来度量现有的基于LLM的EAI系统。我们的研究证实了传统的LLM越狱攻击不直接适用于EAI系统，并确定了两个独特的挑战。首先，有害的文本不一定构成有害的政策。其次，即使可以产生有害的政策，它们也不一定可以由EAI系统执行，这限制了潜在的风险。为了便于更全面的安全分析，我们改进并引入了POEX，这是一个新的红色团队框架，它优化了敌意后缀，以诱导针对EAI系统的有害但可执行的策略。POEX的设计采用对抗性约束、策略评估器和后缀优化，以确保成功执行策略，同时避免EAI系统内的安全检测。在真实世界的机械臂和模拟器上的实验证明了该方法的有效性，突出了严重的安全漏洞和高度的跨模型可移植性。最后，我们提出了基于提示和基于模型的防御，在缓解攻击和增强EAI系统的安全意识方面取得了85%的成功率。我们的发现强调了迫切需要强有力的安全措施，以确保在关键应用中安全地部署EAI。



## **21. Detecting Backdoor Samples in Contrastive Language Image Pretraining**

对比语言图像预训练中后门样本检测 cs.LG

ICLR2025

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.01385v2) [paper-pdf](http://arxiv.org/pdf/2502.01385v2)

**Authors**: Hanxun Huang, Sarah Erfani, Yige Li, Xingjun Ma, James Bailey

**Abstract**: Contrastive language-image pretraining (CLIP) has been found to be vulnerable to poisoning backdoor attacks where the adversary can achieve an almost perfect attack success rate on CLIP models by poisoning only 0.01\% of the training dataset. This raises security concerns on the current practice of pretraining large-scale models on unscrutinized web data using CLIP. In this work, we analyze the representations of backdoor-poisoned samples learned by CLIP models and find that they exhibit unique characteristics in their local subspace, i.e., their local neighborhoods are far more sparse than that of clean samples. Based on this finding, we conduct a systematic study on detecting CLIP backdoor attacks and show that these attacks can be easily and efficiently detected by traditional density ratio-based local outlier detectors, whereas existing backdoor sample detection methods fail. Our experiments also reveal that an unintentional backdoor already exists in the original CC3M dataset and has been trained into a popular open-source model released by OpenCLIP. Based on our detector, one can clean up a million-scale web dataset (e.g., CC3M) efficiently within 15 minutes using 4 Nvidia A100 GPUs. The code is publicly available in our \href{https://github.com/HanxunH/Detect-CLIP-Backdoor-Samples}{GitHub repository}.

摘要: 对比语言图像预训练(CLIP)被发现容易受到中毒后门攻击，对手只需中毒0.01%的训练数据集就可以在CLIP模型上获得近乎完美的攻击成功率。这引发了人们对当前使用CLIP对大规模模型进行未经审查的网络数据的预培训的安全担忧。在这项工作中，我们分析了通过CLIP模型学习的后门中毒样本的表示，发现它们在局部子空间中表现出独特的特征，即它们的局部邻域比干净样本的局部邻域稀疏得多。基于这一发现，我们对CLIP后门攻击的检测进行了系统的研究，结果表明，传统的基于密度比的局部离群点检测方法可以轻松有效地检测到这些攻击，而现有的后门样本检测方法却无法检测到这些攻击。我们的实验还表明，在原始的CC3M数据集中已经存在一个无意的后门，并且已经被训练成OpenCLIP发布的流行的开源模型。基于我们的检测器，您可以使用4个NVIDIA A100图形处理器在15分钟内高效清理百万级Web数据集(例如CC3M)。代码在我们的\href{https://github.com/HanxunH/Detect-CLIP-Backdoor-Samples}{GitHub存储库中公开提供。



## **22. Confidence Elicitation: A New Attack Vector for Large Language Models**

信心激发：大型语言模型的新攻击载体 cs.LG

Published in ICLR 2025. The code is publicly available at  https://github.com/Aniloid2/Confidence_Elicitation_Attacks

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.04643v2) [paper-pdf](http://arxiv.org/pdf/2502.04643v2)

**Authors**: Brian Formento, Chuan Sheng Foo, See-Kiong Ng

**Abstract**: A fundamental issue in deep learning has been adversarial robustness. As these systems have scaled, such issues have persisted. Currently, large language models (LLMs) with billions of parameters suffer from adversarial attacks just like their earlier, smaller counterparts. However, the threat models have changed. Previously, having gray-box access, where input embeddings or output logits/probabilities were visible to the user, might have been reasonable. However, with the introduction of closed-source models, no information about the model is available apart from the generated output. This means that current black-box attacks can only utilize the final prediction to detect if an attack is successful. In this work, we investigate and demonstrate the potential of attack guidance, akin to using output probabilities, while having only black-box access in a classification setting. This is achieved through the ability to elicit confidence from the model. We empirically show that the elicited confidence is calibrated and not hallucinated for current LLMs. By minimizing the elicited confidence, we can therefore increase the likelihood of misclassification. Our new proposed paradigm demonstrates promising state-of-the-art results on three datasets across two models (LLaMA-3-8B-Instruct and Mistral-7B-Instruct-V0.3) when comparing our technique to existing hard-label black-box attack methods that introduce word-level substitutions.

摘要: 深度学习的一个基本问题是对手的稳健性。随着这些系统的规模扩大，这样的问题一直存在。目前，具有数十亿个参数的大型语言模型(LLM)就像它们早期的较小对应模型一样，受到对手攻击。然而，威胁模型已经发生了变化。以前，拥有灰箱访问，其中输入嵌入或输出日志/概率对用户可见，可能是合理的。然而，随着闭源模型的引入，除了生成的输出之外，没有关于模型的信息可用。这意味着当前的黑盒攻击只能利用最终预测来检测攻击是否成功。在这项工作中，我们调查和演示了攻击指导的潜力，类似于使用输出概率，而在分类设置中只有黑盒访问。这是通过从模型中获得信心的能力来实现的。我们的经验表明，对于当前的LLM来说，引发的信心是经过校准的，而不是幻觉的。因此，通过将引起的置信度降至最低，我们可以增加错误分类的可能性。我们提出的新范式在两个模型(骆驼-3-8B-指令和Mistral-7B-指令-V0.3)的三个数据集上展示了有希望的最先进结果，当将我们的技术与现有的引入词级替换的硬标签黑盒攻击方法进行比较时。



## **23. ETA: Evaluating Then Aligning Safety of Vision Language Models at Inference Time**

埃塔：在推理时间评估然后调整视觉语言模型的安全性 cs.CV

29pages

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2410.06625v2) [paper-pdf](http://arxiv.org/pdf/2410.06625v2)

**Authors**: Yi Ding, Bolian Li, Ruqi Zhang

**Abstract**: Vision Language Models (VLMs) have become essential backbones for multimodal intelligence, yet significant safety challenges limit their real-world application. While textual inputs are often effectively safeguarded, adversarial visual inputs can easily bypass VLM defense mechanisms. Existing defense methods are either resource-intensive, requiring substantial data and compute, or fail to simultaneously ensure safety and usefulness in responses. To address these limitations, we propose a novel two-phase inference-time alignment framework, Evaluating Then Aligning (ETA): 1) Evaluating input visual contents and output responses to establish a robust safety awareness in multimodal settings, and 2) Aligning unsafe behaviors at both shallow and deep levels by conditioning the VLMs' generative distribution with an interference prefix and performing sentence-level best-of-N to search the most harmless and helpful generation paths. Extensive experiments show that ETA outperforms baseline methods in terms of harmlessness, helpfulness, and efficiency, reducing the unsafe rate by 87.5% in cross-modality attacks and achieving 96.6% win-ties in GPT-4 helpfulness evaluation. The code is publicly available at https://github.com/DripNowhy/ETA.

摘要: 视觉语言模型已经成为多模式智能的重要支柱，但巨大的安全挑战限制了它们在现实世界中的应用。虽然文本输入通常受到有效保护，但对抗性视觉输入可以很容易地绕过VLM防御机制。现有的防御方法要么是资源密集型的，需要大量的数据和计算，要么无法同时确保响应的安全性和实用性。为了解决这些局限性，我们提出了一种新的两阶段推理-时间对齐框架，评估然后对齐(ETA)：1)评估输入视觉内容和输出响应以在多模式环境中建立稳健的安全意识；2)通过用干扰前缀限制VLM的生成分布并执行句子级的Best-of-N来搜索最无害和最有帮助的生成路径，在浅层和深层对不安全行为进行对齐。大量实验表明，ETA在无害性、有助性和有效性方面都优于基线方法，在跨通道攻击中降低了87.5%的不安全率，在GPT-4有助性评估中获得了96.6%的优胜率。该代码可在https://github.com/DripNowhy/ETA.上公开获得



## **24. A Conditional Tabular GAN-Enhanced Intrusion Detection System for Rare Attacks in IoT Networks**

针对物联网网络中罕见攻击的条件表格GAN增强型入侵检测系统 cs.CR

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2502.06031v1) [paper-pdf](http://arxiv.org/pdf/2502.06031v1)

**Authors**: Safaa Menssouri, El Mehdi Amhoud

**Abstract**: Internet of things (IoT) networks, boosted by 6G technology, are transforming various industries. However, their widespread adoption introduces significant security risks, particularly in detecting rare but potentially damaging cyber-attacks. This makes the development of robust IDS crucial for monitoring network traffic and ensuring their safety. Traditional IDS often struggle with detecting rare attacks due to severe class imbalances in IoT data. In this paper, we propose a novel two-stage system called conditional tabular generative synthetic minority data generation with deep neural network (CTGSM-DNN). In the first stage, a conditional tabular generative adversarial network (CTGAN) is employed to generate synthetic data for rare attack classes. In the second stage, the SMOTEENN method is applied to improve dataset quality. The full study was conducted using the CSE-CIC-IDS2018 dataset, and we assessed the performance of the proposed IDS using different evaluation metrics. The experimental results demonstrated the effectiveness of the proposed multiclass classifier, achieving an overall accuracy of 99.90% and 80% accuracy in detecting rare attacks.

摘要: 在6G技术的推动下，物联网(IoT)网络正在改变各个行业。然而，它们的广泛采用带来了重大的安全风险，特别是在检测罕见但具有潜在破坏性的网络攻击方面。这使得开发健壮的入侵检测系统对于监控网络流量和确保其安全至关重要。由于物联网数据中的严重类别不平衡，传统的入侵检测系统经常难以检测到罕见的攻击。本文提出了一种新的基于深度神经网络的条件表格生成式少数群体数据生成系统(CTGSM-DNN)。在第一阶段，使用条件表格生成对抗网络(CTGAN)来生成稀有攻击类别的合成数据。在第二阶段，应用SMOTEENN方法来提高数据集质量。整个研究是使用CSE-CIC-IDS2018数据集进行的，我们使用不同的评估指标评估了建议的入侵检测系统的性能。实验结果证明了该多类分类器的有效性，在检测罕见攻击时，总体正确率为99.90%，正确率为80%。



## **25. Detection of Physiological Data Tampering Attacks with Quantum Machine Learning**

利用量子机器学习检测生理数据篡改攻击 quant-ph

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2502.05966v1) [paper-pdf](http://arxiv.org/pdf/2502.05966v1)

**Authors**: Md. Saif Hassan Onim, Himanshu Thapliyal

**Abstract**: The widespread use of cloud-based medical devices and wearable sensors has made physiological data susceptible to tampering. These attacks can compromise the reliability of healthcare systems which can be critical and life-threatening. Detection of such data tampering is of immediate need. Machine learning has been used to detect anomalies in datasets but the performance of Quantum Machine Learning (QML) is still yet to be evaluated for physiological sensor data. Thus, our study compares the effectiveness of QML for detecting physiological data tampering, focusing on two types of white-box attacks: data poisoning and adversarial perturbation. The results show that QML models are better at identifying label-flipping attacks, achieving accuracy rates of 75%-95% depending on the data and attack severity. This superior performance is due to the ability of quantum algorithms to handle complex and high-dimensional data. However, both QML and classical models struggle to detect more sophisticated adversarial perturbation attacks, which subtly alter data without changing its statistical properties. Although QML performed poorly against this attack with around 45%-65% accuracy, it still outperformed classical algorithms in some cases.

摘要: 基于云的医疗设备和可穿戴传感器的广泛使用使得生理数据很容易被篡改。这些攻击可能会危及医疗系统的可靠性，这可能是至关重要的，并可能危及生命。检测这种数据篡改是迫在眉睫的。机器学习已被用于检测数据集中的异常，但量子机器学习(QML)的性能仍有待于生理传感器数据的评估。因此，我们的研究比较了QML在检测生理数据篡改方面的有效性，重点关注两种类型的白盒攻击：数据中毒和对抗性扰动。结果表明，QML模型能更好地识别翻转标签攻击，根据数据和攻击严重程度的不同，准确率在75%-95%之间。这种优越的性能归功于量子算法处理复杂和高维数据的能力。然而，QML和经典模型都很难检测到更复杂的对抗性扰动攻击，这些攻击在不改变数据统计特性的情况下微妙地改变数据。尽管QML在抵抗这种攻击时表现不佳，准确率约为45%-65%，但在某些情况下，它的表现仍然优于经典算法。



## **26. A Practical Examination of AI-Generated Text Detectors for Large Language Models**

大型语言模型的人工智能生成文本检测器的实践检验 cs.CL

9 pages

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2412.05139v4) [paper-pdf](http://arxiv.org/pdf/2412.05139v4)

**Authors**: Brian Tufts, Xuandong Zhao, Lei Li

**Abstract**: The proliferation of large language models has raised growing concerns about their misuse, particularly in cases where AI-generated text is falsely attributed to human authors. Machine-generated content detectors claim to effectively identify such text under various conditions and from any language model. This paper critically evaluates these claims by assessing several popular detectors (RADAR, Wild, T5Sentinel, Fast-DetectGPT, PHD, LogRank, Binoculars) on a range of domains, datasets, and models that these detectors have not previously encountered. We employ various prompting strategies to simulate practical adversarial attacks, demonstrating that even moderate efforts can significantly evade detection. We emphasize the importance of the true positive rate at a specific false positive rate (TPR@FPR) metric and demonstrate that these detectors perform poorly in certain settings, with TPR@.01 as low as 0%. Our findings suggest that both trained and zero-shot detectors struggle to maintain high sensitivity while achieving a reasonable true positive rate.

摘要: 大型语言模型的激增引发了人们对它们滥用的日益担忧，特别是在人工智能生成的文本被错误地归因于人类作者的情况下。机器生成的内容检测器声称可以在各种条件下从任何语言模型有效地识别此类文本。本文通过评估几种流行的探测器(雷达、Wild、T5Sentinel、Fast-DetectGPT、PHD、logrank、双筒望远镜)，对这些声称进行了批判性的评估，这些探测器以前从未遇到过。我们使用各种提示策略来模拟实际的对抗性攻击，表明即使是适度的攻击也可以显著地躲避检测。我们强调了在特定的假阳性率(TPR@fPR)度量下的真阳性率的重要性，并证明了这些检测器在某些设置下表现很差，TPR@.01低至0%。我们的发现表明，训练有素的探测器和零射探测器都很难在保持高灵敏度的同时获得合理的真阳性率。



## **27. Optimization under Attack: Resilience, Vulnerability, and the Path to Collapse**

攻击下的优化：韧性、脆弱性和崩溃之路 cs.MA

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2502.05954v1) [paper-pdf](http://arxiv.org/pdf/2502.05954v1)

**Authors**: Amal Aldawsari, Evangelos Pournaras

**Abstract**: Optimization is instrumental for improving operations of large-scale socio-technical infrastructures of Smart Cities, for instance, energy and traffic systems. In particular, understanding the performance of multi-agent discrete-choice combinatorial optimization under distributed adversary attacks is a compelling and underexplored problem, since multi-agent systems exhibit a large number of remote control variables that can influence in an unprecedented way the cost-effectiveness of distributed optimization heuristics. This paper unravels for the first time the trajectories of distributed optimization from resilience to vulnerability, and finally to collapse under varying adversary influence. Using real-world data to emulate over 28 billion multi-agent optimization scenarios, we exhaustively assess how the number of agents with different adversarial severity and network positioning influences optimization performance, including the influence on Pareto optimal points. With this novel large-scale dataset, made openly available as a benchmark, we disentangle how optimization remains resilient to adversaries and which adversary conditions are required to make optimization vulnerable or collapsed. These new findings can provide new insights for designing self-healing strategies for fault-tolerance and fault-correction in adversarial distributed optimization that have been missing so far.

摘要: 优化有助于改善智能城市的大规模社会技术基础设施的运行，例如能源和交通系统。特别是，了解分布式攻击下多智能体离散选择组合优化的性能是一个引人注目而又未被探索的问题，因为多智能体系统表现出大量的远程控制变量，这些变量可能以前所未有的方式影响分布式优化启发式算法的成本效益。本文首次揭示了分布式优化从弹性到脆弱性，再到在不同对手影响下崩溃的轨迹。使用真实世界的数据模拟了超过280亿个多智能体优化场景，我们详尽地评估了具有不同对手严重性和网络位置的智能体数量对优化性能的影响，包括对帕累托最优点的影响。通过公开提供的这个新的大规模数据集作为基准，我们理清了优化如何保持对对手的弹性，以及哪些对手条件需要使优化变得脆弱或崩溃。这些新的发现可以为设计对抗性分布式优化中的容错和纠错的自愈策略提供新的见解，这些策略到目前为止还没有找到。



## **28. Protecting Intellectual Property of EEG-based Neural Networks with Watermarking**

利用水印保护基于脑电的神经网络的知识产权 cs.LG

21 pages, 13 figures, and 6 tables

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2502.05931v1) [paper-pdf](http://arxiv.org/pdf/2502.05931v1)

**Authors**: Ahmed Abdelaziz, Ahmed Fathi, Ahmed Fares

**Abstract**: EEG-based neural networks, pivotal in medical diagnosis and brain-computer interfaces, face significant intellectual property (IP) risks due to their reliance on sensitive neurophysiological data and resource-intensive development. Current watermarking methods, particularly those using abstract trigger sets, lack robust authentication and fail to address the unique challenges of EEG models. This paper introduces a cryptographic wonder filter-based watermarking framework tailored for EEG-based neural networks. Leveraging collision-resistant hashing and public-key encryption, the wonder filter embeds the watermark during training, ensuring minimal distortion ($\leq 5\%$ drop in EEG task accuracy) and high reliability (100\% watermark detection). The framework is rigorously evaluated against adversarial attacks, including fine-tuning, transfer learning, and neuron pruning. Results demonstrate persistent watermark retention, with classification accuracy for watermarked states remaining above 90\% even after aggressive pruning, while primary task performance degrades faster, deterring removal attempts. Piracy resistance is validated by the inability to embed secondary watermarks without severe accuracy loss ( $>10\%$ in EEGNet and CCNN models). Cryptographic hashing ensures authentication, reducing brute-force attack success probabilities. Evaluated on the DEAP dataset across models (CCNN, EEGNet, TSception), the method achieves $>99.4\%$ null-embedding accuracy, effectively eliminating false positives. By integrating wonder filters with EEG-specific adaptations, this work bridges a critical gap in IP protection for neurophysiological models, offering a secure, tamper-proof solution for healthcare and biometric applications. The framework's robustness against adversarial modifications underscores its potential to safeguard sensitive EEG models while maintaining diagnostic utility.

摘要: 基于脑电的神经网络在医学诊断和脑-机接口中起着关键作用，但由于其依赖于敏感的神经生理数据和资源密集型开发，因此面临着巨大的知识产权风险。当前的水印方法，特别是那些使用抽象触发集的方法，缺乏健壮的认证，并且不能解决EEG模型的独特挑战。介绍了一种适用于脑电神经网络的基于密码奇异值滤波的数字水印框架。WONDER过滤器利用防碰撞散列和公钥加密，在训练过程中嵌入水印，确保最小的失真(EEG任务精度下降5美元)和高可靠性(100水印检测)。该框架针对对手攻击进行了严格的评估，包括微调、迁移学习和神经元修剪。结果表明，水印具有持久的保留性，即使在主动剪枝后，水印状态的分类准确率仍保持在90%以上，而主任务性能下降得更快，阻止了移除尝试。在EEGNet和CCNN模型中，不能在不损失严重精度的情况下嵌入二次水印($>10\cn)，从而验证了抗盗版能力。加密散列可确保身份验证，从而降低暴力攻击成功的概率。在DEAP数据集(CCNN，EEGNet，TScept)上的评估表明，该方法达到了$>99.4$空嵌入的精度，有效地消除了误报。通过将Wonder过滤器与特定于EEG的自适应相结合，这项工作弥合了神经生理模型的IP保护方面的关键差距，为医疗保健和生物识别应用提供了安全、防篡改的解决方案。该框架对敌意修改的健壮性突显了它在保持诊断实用的同时保护敏感脑电模型的潜力。



## **29. ADBM: Adversarial diffusion bridge model for reliable adversarial purification**

ADBM：用于可靠对抗净化的对抗扩散桥模型 cs.LG

ICLR 2025

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2408.00315v2) [paper-pdf](http://arxiv.org/pdf/2408.00315v2)

**Authors**: Xiao Li, Wenxuan Sun, Huanran Chen, Qiongxiu Li, Yining Liu, Yingzhe He, Jie Shi, Xiaolin Hu

**Abstract**: Recently Diffusion-based Purification (DiffPure) has been recognized as an effective defense method against adversarial examples. However, we find DiffPure which directly employs the original pre-trained diffusion models for adversarial purification, to be suboptimal. This is due to an inherent trade-off between noise purification performance and data recovery quality. Additionally, the reliability of existing evaluations for DiffPure is questionable, as they rely on weak adaptive attacks. In this work, we propose a novel Adversarial Diffusion Bridge Model, termed ADBM. ADBM directly constructs a reverse bridge from the diffused adversarial data back to its original clean examples, enhancing the purification capabilities of the original diffusion models. Through theoretical analysis and experimental validation across various scenarios, ADBM has proven to be a superior and robust defense mechanism, offering significant promise for practical applications.

摘要: 最近，基于扩散的纯化（DiffPure）被认为是针对对抗性例子的有效防御方法。然而，我们发现直接使用原始预训练的扩散模型进行对抗性纯化的迪夫Pure是次优的。这是由于噪音净化性能和数据恢复质量之间固有的权衡。此外，现有的DistPure评估的可靠性值得怀疑，因为它们依赖于弱适应性攻击。在这项工作中，我们提出了一种新型的对抗扩散桥模型，称为ADBM。ADBM直接构建了从扩散的对抗数据到其原始干净示例的反向桥梁，增强了原始扩散模型的净化能力。通过各种场景的理论分析和实验验证，ADBM已被证明是一种卓越且强大的防御机制，为实际应用提供了巨大的前景。



## **30. Effective Black-Box Multi-Faceted Attacks Breach Vision Large Language Model Guardrails**

有效的黑匣子多面攻击破坏视觉大型语言模型护栏 cs.CV

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2502.05772v1) [paper-pdf](http://arxiv.org/pdf/2502.05772v1)

**Authors**: Yijun Yang, Lichao Wang, Xiao Yang, Lanqing Hong, Jun Zhu

**Abstract**: Vision Large Language Models (VLLMs) integrate visual data processing, expanding their real-world applications, but also increasing the risk of generating unsafe responses. In response, leading companies have implemented Multi-Layered safety defenses, including alignment training, safety system prompts, and content moderation. However, their effectiveness against sophisticated adversarial attacks remains largely unexplored. In this paper, we propose MultiFaceted Attack, a novel attack framework designed to systematically bypass Multi-Layered Defenses in VLLMs. It comprises three complementary attack facets: Visual Attack that exploits the multimodal nature of VLLMs to inject toxic system prompts through images; Alignment Breaking Attack that manipulates the model's alignment mechanism to prioritize the generation of contrasting responses; and Adversarial Signature that deceives content moderators by strategically placing misleading information at the end of the response. Extensive evaluations on eight commercial VLLMs in a black-box setting demonstrate that MultiFaceted Attack achieves a 61.56% attack success rate, surpassing state-of-the-art methods by at least 42.18%.

摘要: 视觉大语言模型(VLLM)集成了可视化数据处理，扩展了它们在现实世界中的应用，但也增加了生成不安全响应的风险。作为回应，领先的公司实施了多层次的安全防御措施，包括对齐培训、安全系统提示和内容审核。然而，它们对抗复杂的对抗性攻击的有效性在很大程度上仍未得到探索。在本文中，我们提出了一种新的攻击框架--多方面攻击，旨在系统地绕过VLLMS中的多层防御。它包括三个互补的攻击方面：利用VLLM的多模式特性通过图像注入有毒系统提示的视觉攻击；操纵模型的对齐机制以优先生成对比响应的对齐破坏攻击；以及通过在响应的末尾战略性地放置误导性信息来欺骗内容审核者的对抗性签名。在黑匣子环境下对8个商用VLLM进行了广泛的评估，结果表明，多面攻击的攻击成功率达到了61.56%，至少比最先进的攻击方法高出42.18%。



## **31. Filter, Obstruct and Dilute: Defending Against Backdoor Attacks on Semi-Supervised Learning**

过滤、阻碍和稀释：抵御对半监督学习的后门攻击 cs.LG

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2502.05755v1) [paper-pdf](http://arxiv.org/pdf/2502.05755v1)

**Authors**: Xinrui Wang, Chuanxing Geng, Wenhai Wan, Shao-yuan Li, Songcan Chen

**Abstract**: Recent studies have verified that semi-supervised learning (SSL) is vulnerable to data poisoning backdoor attacks. Even a tiny fraction of contaminated training data is sufficient for adversaries to manipulate up to 90\% of the test outputs in existing SSL methods. Given the emerging threat of backdoor attacks designed for SSL, this work aims to protect SSL against such risks, marking it as one of the few known efforts in this area. Specifically, we begin by identifying that the spurious correlations between the backdoor triggers and the target class implanted by adversaries are the primary cause of manipulated model predictions during the test phase. To disrupt these correlations, we utilize three key techniques: Gaussian Filter, complementary learning and trigger mix-up, which collectively filter, obstruct and dilute the influence of backdoor attacks in both data pre-processing and feature learning. Experimental results demonstrate that our proposed method, Backdoor Invalidator (BI), significantly reduces the average attack success rate from 84.7\% to 1.8\% across different state-of-the-art backdoor attacks. It is also worth mentioning that BI does not sacrifice accuracy on clean data and is supported by a theoretical guarantee of its generalization capability.

摘要: 最近的研究证明，半监督学习(SSL)容易受到数据中毒后门攻击。即使是极小一部分受污染的训练数据，也足以让攻击者在现有的SSL方法中操纵高达90%的测试输出。鉴于针对SSL设计的后门攻击的新威胁，这项工作旨在保护SSL免受此类风险，这是该领域为数不多的已知努力之一。具体地说，我们首先确定后门触发器和对手植入的目标类之间的虚假关联是测试阶段操纵模型预测的主要原因。为了打破这种相关性，我们使用了三种关键技术：高斯滤波、互补学习和触发混合，在数据预处理和特征学习中共同过滤、阻挡和稀释后门攻击的影响。实验结果表明，我们提出的后门验证器(BI)方法可以显著降低不同类型后门攻击的平均攻击成功率，从84.7%降低到1.8%。还值得一提的是，BI不牺牲干净数据的准确性，并得到其泛化能力的理论保证的支持。



## **32. The Evolution of Dataset Distillation: Toward Scalable and Generalizable Solutions**

数据集蒸馏的演变：迈向可扩展和可推广的解决方案 cs.CV

**SubmitDate**: 2025-02-08    [abs](http://arxiv.org/abs/2502.05673v1) [paper-pdf](http://arxiv.org/pdf/2502.05673v1)

**Authors**: Ping Liu, Jiawei Du

**Abstract**: Dataset distillation, which condenses large-scale datasets into compact synthetic representations, has emerged as a critical solution for training modern deep learning models efficiently. While prior surveys focus on developments before 2023, this work comprehensively reviews recent advances, emphasizing scalability to large-scale datasets such as ImageNet-1K and ImageNet-21K. We categorize progress into a few key methodologies: trajectory matching, gradient matching, distribution matching, scalable generative approaches, and decoupling optimization mechanisms. As a comprehensive examination of recent dataset distillation advances, this survey highlights breakthrough innovations: the SRe2L framework for efficient and effective condensation, soft label strategies that significantly enhance model accuracy, and lossless distillation techniques that maximize compression while maintaining performance. Beyond these methodological advancements, we address critical challenges, including robustness against adversarial and backdoor attacks, effective handling of non-IID data distributions. Additionally, we explore emerging applications in video and audio processing, multi-modal learning, medical imaging, and scientific computing, highlighting its domain versatility. By offering extensive performance comparisons and actionable research directions, this survey equips researchers and practitioners with practical insights to advance efficient and generalizable dataset distillation, paving the way for future innovations.

摘要: 数据集蒸馏将大规模数据集浓缩为紧凑的综合表示，已成为有效训练现代深度学习模型的关键解决方案。虽然以前的调查侧重于2023年之前的发展，但这项工作全面回顾了最近的进展，强调了对大规模数据集的可扩展性，如ImageNet-1K和ImageNet-21K。我们将进展归类为几种关键方法：轨迹匹配、梯度匹配、分布匹配、可伸缩的产生式方法和解耦优化机制。随着对最新数据集蒸馏进展的全面研究，本调查重点介绍了突破性创新：用于高效和有效缩合的SRe2L框架、显著提高模型精度的软标签策略，以及在保持性能的同时最大化压缩的无损蒸馏技术。除了这些方法上的进步，我们还解决了关键挑战，包括对对手和后门攻击的健壮性，对非IID数据分发的有效处理。此外，我们还探讨了视频和音频处理、多模式学习、医学成像和科学计算中的新兴应用，突出了其领域的多功能性。通过提供广泛的性能比较和可操作的研究方向，此调查为研究人员和从业者提供了实用的见解，以推进高效和可推广的数据集蒸馏，为未来的创新铺平道路。



## **33. Rigid Body Adversarial Attacks**

刚性身体对抗攻击 cs.CV

17 pages, 14 figures, 3DV 2025

**SubmitDate**: 2025-02-08    [abs](http://arxiv.org/abs/2502.05669v1) [paper-pdf](http://arxiv.org/pdf/2502.05669v1)

**Authors**: Aravind Ramakrishnan, David I. W. Levin, Alec Jacobson

**Abstract**: Due to their performance and simplicity, rigid body simulators are often used in applications where the objects of interest can considered very stiff. However, no material has infinite stiffness, which means there are potentially cases where the non-zero compliance of the seemingly rigid object can cause a significant difference between its trajectories when simulated in a rigid body or deformable simulator.   Similarly to how adversarial attacks are developed against image classifiers, we propose an adversarial attack against rigid body simulators. In this adversarial attack, we solve an optimization problem to construct perceptually rigid adversarial objects that have the same collision geometry and moments of mass to a reference object, so that they behave identically in rigid body simulations but maximally different in more accurate deformable simulations. We demonstrate the validity of our method by comparing simulations of several examples in commercially available simulators.

摘要: 由于其性能和简单性，刚性体模拟器经常用于感兴趣物体可能被认为非常僵硬的应用中。然而，没有任何材料具有无限的硬度，这意味着在某些情况下，看似刚性的物体的非零柔度可能会导致其轨迹之间的显着差异，在刚性体或可变形模拟器中进行模拟。   与针对图像分类器的对抗攻击的发展方式类似，我们提出了针对刚性身体模拟器的对抗攻击。在这种对抗攻击中，我们解决了一个优化问题，以构建感知上刚性的对抗对象，这些对象具有与参考对象相同的碰撞几何形状和质量矩，以便它们在刚性物体模拟中表现相同，但在更准确的可变形模拟中表现最大不同。我们通过比较市售模拟器中几个例子的模拟来证明我们方法的有效性。



## **34. Adversarial Machine Learning: Attacks, Defenses, and Open Challenges**

对抗性机器学习：攻击、防御和开放挑战 cs.CR

**SubmitDate**: 2025-02-08    [abs](http://arxiv.org/abs/2502.05637v1) [paper-pdf](http://arxiv.org/pdf/2502.05637v1)

**Authors**: Pranav K Jha

**Abstract**: Adversarial Machine Learning (AML) addresses vulnerabilities in AI systems where adversaries manipulate inputs or training data to degrade performance. This article provides a comprehensive analysis of evasion and poisoning attacks, formalizes defense mechanisms with mathematical rigor, and discusses the challenges of implementing robust solutions in adaptive threat models. Additionally, it highlights open challenges in certified robustness, scalability, and real-world deployment.

摘要: 对抗性机器学习（ML）解决了人工智能系统中的漏洞，其中对手操纵输入或训练数据以降低性能。本文对规避和中毒攻击进行了全面分析，以数学严谨性形式化了防御机制，并讨论了在自适应威胁模型中实施稳健解决方案的挑战。此外，它还强调了认证稳健性、可扩展性和现实部署方面的公开挑战。



## **35. Democratic Training Against Universal Adversarial Perturbations**

反对普遍对抗性干扰的民主训练 cs.LG

**SubmitDate**: 2025-02-08    [abs](http://arxiv.org/abs/2502.05542v1) [paper-pdf](http://arxiv.org/pdf/2502.05542v1)

**Authors**: Bing Sun, Jun Sun, Wei Zhao

**Abstract**: Despite their advances and success, real-world deep neural networks are known to be vulnerable to adversarial attacks. Universal adversarial perturbation, an input-agnostic attack, poses a serious threat for them to be deployed in security-sensitive systems. In this case, a single universal adversarial perturbation deceives the model on a range of clean inputs without requiring input-specific optimization, which makes it particularly threatening. In this work, we observe that universal adversarial perturbations usually lead to abnormal entropy spectrum in hidden layers, which suggests that the prediction is dominated by a small number of ``feature'' in such cases (rather than democratically by many features). Inspired by this, we propose an efficient yet effective defense method for mitigating UAPs called \emph{Democratic Training} by performing entropy-based model enhancement to suppress the effect of the universal adversarial perturbations in a given model. \emph{Democratic Training} is evaluated with 7 neural networks trained on 5 benchmark datasets and 5 types of state-of-the-art universal adversarial attack methods. The results show that it effectively reduces the attack success rate, improves model robustness and preserves the model accuracy on clean samples.

摘要: 尽管它们取得了进步和成功，但现实世界中的深度神经网络仍然容易受到对手的攻击。通用对抗性扰动是一种与输入无关的攻击，对它们在安全敏感系统中的部署构成了严重威胁。在这种情况下，单一的通用对抗性扰动在一系列干净的输入上欺骗了模型，而不需要特定于输入的优化，这使得它特别具有威胁性。在这项工作中，我们观察到普遍的对抗性扰动通常会导致隐含层中的异常熵谱，这表明在这种情况下，预测是由少数“特征”主导的(而不是民主地由许多特征主导)。受此启发，我们提出了一种有效的防御方法--基于熵的模型增强，以抑制给定模型中普遍的对抗性扰动的影响，从而减轻UAP的影响。使用7个神经网络对5个基准数据集和5种最先进的通用对抗性攻击方法进行了评估。实验结果表明，该方法有效地降低了攻击成功率，提高了模型的稳健性，并保持了干净样本的模型精度。



## **36. Do Spikes Protect Privacy? Investigating Black-Box Model Inversion Attacks in Spiking Neural Networks**

Spikes保护隐私吗？研究尖峰神经网络中的黑匣子模型倒置攻击 cs.LG

7 pages, 4 figures

**SubmitDate**: 2025-02-08    [abs](http://arxiv.org/abs/2502.05509v1) [paper-pdf](http://arxiv.org/pdf/2502.05509v1)

**Authors**: Hamed Poursiami, Ayana Moshruba, Maryam Parsa

**Abstract**: As machine learning models become integral to security-sensitive applications, concerns over data leakage from adversarial attacks continue to rise. Model Inversion (MI) attacks pose a significant privacy threat by enabling adversaries to reconstruct training data from model outputs. While MI attacks on Artificial Neural Networks (ANNs) have been widely studied, Spiking Neural Networks (SNNs) remain largely unexplored in this context. Due to their event-driven and discrete computations, SNNs introduce fundamental differences in information processing that may offer inherent resistance to such attacks. A critical yet underexplored aspect of this threat lies in black-box settings, where attackers operate through queries without direct access to model parameters or gradients-representing a more realistic adversarial scenario in deployed systems. This work presents the first study of black-box MI attacks on SNNs. We adapt a generative adversarial MI framework to the spiking domain by incorporating rate-based encoding for input transformation and decoding mechanisms for output interpretation. Our results show that SNNs exhibit significantly greater resistance to MI attacks than ANNs, as demonstrated by degraded reconstructions, increased instability in attack convergence, and overall reduced attack effectiveness across multiple evaluation metrics. Further analysis suggests that the discrete and temporally distributed nature of SNN decision boundaries disrupts surrogate modeling, limiting the attacker's ability to approximate the target model.

摘要: 随着机器学习模型成为安全敏感应用程序不可或缺的一部分，人们对敌意攻击造成的数据泄露的担忧继续上升。模型反转(MI)攻击使攻击者能够根据模型输出重建训练数据，从而对隐私构成重大威胁。虽然人工神经网络(ANN)上的MI攻击已经得到了广泛的研究，但尖峰神经网络(SNN)在这方面的研究仍然很少。由于它们的事件驱动和离散计算，SNN在信息处理方面引入了根本的差异，这可能提供对此类攻击的内在抵抗。这一威胁的一个关键但未被开发的方面存在于黑盒设置中，在黑盒设置中，攻击者通过查询操作，而不直接访问模型参数或梯度-在已部署的系统中代表更现实的对抗性场景。本文首次研究了黑盒MI攻击对SNN的攻击。通过将基于速率的编码用于输入转换和解码机制用于输出解释，我们将生成性对抗性MI框架适应于尖峰领域。我们的结果表明，SNN比ANN表现出更强的抵抗MI攻击的能力，这表现在重建降级，攻击收敛的不稳定性增加，以及多个评估指标的攻击有效性总体降低。进一步的分析表明，SNN决策边界的离散和时间分布的性质破坏了代理建模，限制了攻击者近似目标模型的能力。



## **37. Towards Trustworthy Retrieval Augmented Generation for Large Language Models: A Survey**

面向大型语言模型的可信检索增强生成：一项调查 cs.CL

**SubmitDate**: 2025-02-08    [abs](http://arxiv.org/abs/2502.06872v1) [paper-pdf](http://arxiv.org/pdf/2502.06872v1)

**Authors**: Bo Ni, Zheyuan Liu, Leyao Wang, Yongjia Lei, Yuying Zhao, Xueqi Cheng, Qingkai Zeng, Luna Dong, Yinglong Xia, Krishnaram Kenthapadi, Ryan Rossi, Franck Dernoncourt, Md Mehrab Tanjim, Nesreen Ahmed, Xiaorui Liu, Wenqi Fan, Erik Blasch, Yu Wang, Meng Jiang, Tyler Derr

**Abstract**: Retrieval-Augmented Generation (RAG) is an advanced technique designed to address the challenges of Artificial Intelligence-Generated Content (AIGC). By integrating context retrieval into content generation, RAG provides reliable and up-to-date external knowledge, reduces hallucinations, and ensures relevant context across a wide range of tasks. However, despite RAG's success and potential, recent studies have shown that the RAG paradigm also introduces new risks, including robustness issues, privacy concerns, adversarial attacks, and accountability issues. Addressing these risks is critical for future applications of RAG systems, as they directly impact their trustworthiness. Although various methods have been developed to improve the trustworthiness of RAG methods, there is a lack of a unified perspective and framework for research in this topic. Thus, in this paper, we aim to address this gap by providing a comprehensive roadmap for developing trustworthy RAG systems. We place our discussion around five key perspectives: reliability, privacy, safety, fairness, explainability, and accountability. For each perspective, we present a general framework and taxonomy, offering a structured approach to understanding the current challenges, evaluating existing solutions, and identifying promising future research directions. To encourage broader adoption and innovation, we also highlight the downstream applications where trustworthy RAG systems have a significant impact.

摘要: 检索-增强生成(RAG)是一种高级技术，旨在应对人工智能生成内容(AIGC)的挑战。通过将上下文检索集成到内容生成中，RAG提供了可靠和最新的外部知识，减少了幻觉，并确保了广泛任务中的相关上下文。然而，尽管RAG取得了成功和潜力，但最近的研究表明，RAG范例也带来了新的风险，包括健壮性问题、隐私问题、敌意攻击和责任问题。解决这些风险对RAG系统的未来应用至关重要，因为它们直接影响到它们的可信度。虽然已经开发了各种方法来提高RAG方法的可信度，但对于这一课题的研究缺乏统一的视角和框架。因此，在本文中，我们的目标是通过为开发可信赖的RAG系统提供全面的路线图来解决这一差距。我们围绕五个关键角度展开讨论：可靠性、隐私、安全性、公平性、可解释性和问责制。对于每个角度，我们都提供了一个一般的框架和分类，提供了一种结构化的方法来了解当前的挑战，评估现有的解决方案，并确定有前途的未来研究方向。为了鼓励更广泛的采用和创新，我们还重点介绍了值得信赖的RAG系统具有重大影响的下游应用程序。



## **38. SMaCk: Efficient Instruction Cache Attacks via Self-Modifying Code Conflicts**

SMaCk：通过自修改代码冲突进行高效指令缓存攻击 cs.CR

Proceedings of the 30th ACM International Conference on Architectural  Support for Programming Languages and Operating Systems (ASPLOS) accepted

**SubmitDate**: 2025-02-08    [abs](http://arxiv.org/abs/2502.05429v1) [paper-pdf](http://arxiv.org/pdf/2502.05429v1)

**Authors**: Seonghun Son, Daniel Moghimi, Berk Gulmezoglu

**Abstract**: Self-modifying code (SMC) allows programs to alter their own instructions, optimizing performance and functionality on x86 processors. Despite its benefits, SMC introduces unique microarchitectural behaviors that can be exploited for malicious purposes. In this paper, we explore the security implications of SMC by examining how specific x86 instructions affecting instruction cache lines lead to measurable timing discrepancies between cache hits and misses. These discrepancies facilitate refined cache attacks, making them less noisy and more effective. We introduce novel attack techniques that leverage these timing variations to enhance existing methods such as Prime+Probe and Flush+Reload. Our advanced techniques allow adversaries to more precisely attack cryptographic keys and create covert channels akin to Spectre across various x86 platforms. Finally, we propose a dynamic detection methodology utilizing hardware performance counters to mitigate these enhanced threats.

摘要: 自修改代码（SMC）允许程序更改自己的指令，优化x86处理器上的性能和功能。尽管SMC有优势，但它引入了独特的微体系结构行为，这些行为可能会被用于恶意目的。在本文中，我们通过研究影响指令缓存行的特定x86指令如何导致缓存命中和未命中之间可测量的时间差异来探讨SMC的安全影响。这些差异促进了精确的缓存攻击，使其噪音更小且更有效。我们引入了新颖的攻击技术，利用这些时间变化来增强现有方法，例如Prime+Probe和Flush+ Deliverad。我们的先进技术使对手能够更精确地攻击加密密钥，并在各种x86平台上创建类似于Spectre的秘密通道。最后，我们提出了一种利用硬件性能计数器的动态检测方法来缓解这些增强的威胁。



## **39. Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond**

迈向LLM摆脱学习对重新学习攻击的弹性：敏锐意识的最小化视角及超越 cs.LG

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.05374v1) [paper-pdf](http://arxiv.org/pdf/2502.05374v1)

**Authors**: Chongyu Fan, Jinghan Jia, Yihua Zhang, Anil Ramakrishna, Mingyi Hong, Sijia Liu

**Abstract**: The LLM unlearning technique has recently been introduced to comply with data regulations and address the safety and ethical concerns of LLMs by removing the undesired data-model influence. However, state-of-the-art unlearning methods face a critical vulnerability: they are susceptible to ``relearning'' the removed information from a small number of forget data points, known as relearning attacks. In this paper, we systematically investigate how to make unlearned models robust against such attacks. For the first time, we establish a connection between robust unlearning and sharpness-aware minimization (SAM) through a unified robust optimization framework, in an analogy to adversarial training designed to defend against adversarial attacks. Our analysis for SAM reveals that smoothness optimization plays a pivotal role in mitigating relearning attacks. Thus, we further explore diverse smoothing strategies to enhance unlearning robustness. Extensive experiments on benchmark datasets, including WMDP and MUSE, demonstrate that SAM and other smoothness optimization approaches consistently improve the resistance of LLM unlearning to relearning attacks. Notably, smoothness-enhanced unlearning also helps defend against (input-level) jailbreaking attacks, broadening our proposal's impact in robustifying LLM unlearning. Codes are available at https://github.com/OPTML-Group/Unlearn-Smooth.

摘要: 最近引入了LLM解除学习技术，以遵守数据法规，并通过消除不希望看到的数据模型影响来解决LLM的安全和伦理问题。然而，最先进的遗忘方法面临着一个严重的漏洞：它们容易受到从少数忘记数据点移除的信息的“重新学习”，称为重新学习攻击。在本文中，我们系统地研究了如何使未学习模型对此类攻击具有健壮性。第一次，我们通过一个统一的稳健优化框架在稳健遗忘和敏锐度感知最小化(SAM)之间建立了联系，类似于旨在防御对手攻击的对抗性训练。我们对SAM的分析表明，平滑优化在减轻再学习攻击方面起着关键作用。因此，我们进一步探索不同的平滑策略来增强遗忘的稳健性。在WMDP和MUSE等基准数据集上的大量实验表明，SAM和其他平滑优化方法一致地提高了LLM遗忘对重新学习攻击的抵抗力。值得注意的是，流畅性增强的遗忘也有助于防御(输入级)越狱攻击，扩大了我们的提议在强化LLM遗忘方面的影响。有关代码，请访问https://github.com/OPTML-Group/Unlearn-Smooth.



## **40. Neural Encrypted State Transduction for Ransomware Classification: A Novel Approach Using Cryptographic Flow Residuals**

用于勒索软件分类的神经加密状态转换：一种使用密码流剩余量的新方法 cs.CR

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.05341v1) [paper-pdf](http://arxiv.org/pdf/2502.05341v1)

**Authors**: Barnaby Fortescue, Edmund Hawksmoor, Alistair Wetherington, Frederick Marlowe, Kevin Pekepok

**Abstract**: Encrypted behavioral patterns provide a unique avenue for classifying complex digital threats without reliance on explicit feature extraction, enabling detection frameworks to remain effective even when conventional static and behavioral methodologies fail. A novel approach based on Neural Encrypted State Transduction (NEST) is introduced to analyze cryptographic flow residuals and classify threats through their encrypted state transitions, mitigating evasion tactics employed through polymorphic and obfuscated attack strategies. The mathematical formulation of NEST leverages transduction principles to map state transitions dynamically, enabling high-confidence classification without requiring direct access to decrypted execution traces. Experimental evaluations demonstrate that the proposed framework achieves improved detection accuracy across multiple ransomware families while exhibiting resilience against adversarial perturbations and previously unseen attack variants. The model maintains competitive processing efficiency, offering a practical balance between classification performance and computational resource constraints, making it suitable for large-scale security deployments. Comparative assessments reveal that NEST consistently outperforms baseline classification models, particularly in detecting ransomware samples employing delayed encryption, entropy-based obfuscation, and memory-resident execution techniques. The capacity to generalize across diverse execution environments reinforces the applicability of encrypted transduction methodologies in adversarial classification tasks beyond conventional malware detection pipelines. The integration of residual learning mechanisms within the transduction layers further enhances classification robustness, minimizing both false positives and misclassification rates across varied operational contexts.

摘要: 加密的行为模式为分类复杂的数字威胁提供了一种独特的途径，而无需依赖明确的特征提取，从而使检测框架即使在传统的静态和行为方法失败时也能保持有效。提出了一种新的基于神经加密状态转换(Nest)的方法，通过加密状态转换来分析密码流的残差并对威胁进行分类，从而减少了通过多态和模糊攻击策略所采用的规避策略。Nest的数学公式利用转导原理动态映射状态转换，从而实现高置信度分类，而无需直接访问解密的执行轨迹。实验评估表明，该框架提高了对多个勒索软件家族的检测准确率，同时表现出对敌意扰动和以前未见的攻击变体的弹性。该模型保持了具有竞争力的处理效率，在分类性能和计算资源约束之间提供了实用的平衡，使其适用于大规模安全部署。比较评估表明，Nest的性能始终优于基准分类模型，特别是在检测采用延迟加密、基于熵的混淆和内存驻留执行技术的勒索软件样本方面。跨不同执行环境的通用性增强了加密转导方法在传统恶意软件检测管道之外的敌意分类任务中的适用性。转导层内残留学习机制的集成进一步增强了分类的稳健性，最大限度地减少了不同操作环境中的假阳性和错误分类率。



## **41. ADAPT to Robustify Prompt Tuning Vision Transformers**

ADAPT以Robustify提示调整视觉变形金刚 cs.LG

Published in Transactions on Machine Learning Research (2025)

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2403.13196v2) [paper-pdf](http://arxiv.org/pdf/2403.13196v2)

**Authors**: Masih Eskandar, Tooba Imtiaz, Zifeng Wang, Jennifer Dy

**Abstract**: The performance of deep models, including Vision Transformers, is known to be vulnerable to adversarial attacks. Many existing defenses against these attacks, such as adversarial training, rely on full-model fine-tuning to induce robustness in the models. These defenses require storing a copy of the entire model, that can have billions of parameters, for each task. At the same time, parameter-efficient prompt tuning is used to adapt large transformer-based models to downstream tasks without the need to save large copies. In this paper, we examine parameter-efficient prompt tuning of Vision Transformers for downstream tasks under the lens of robustness. We show that previous adversarial defense methods, when applied to the prompt tuning paradigm, suffer from gradient obfuscation and are vulnerable to adaptive attacks. We introduce ADAPT, a novel framework for performing adaptive adversarial training in the prompt tuning paradigm. Our method achieves competitive robust accuracy of ~40% w.r.t. SOTA robustness methods using full-model fine-tuning, by tuning only ~1% of the number of parameters.

摘要: 众所周知，包括Vision Transformers在内的深度模型的性能很容易受到对手的攻击。许多现有的针对这些攻击的防御，如对抗性训练，都依赖于全模型微调来诱导模型的健壮性。这些防御需要为每个任务存储整个模型的副本，该副本可以具有数十亿个参数。同时，使用参数高效的即时调整来使大型基于变压器的模型适应下游任务，而不需要保存大量副本。在这篇文章中，我们研究了稳健性镜头下的视觉变形器的参数高效的下游任务的快速调整。我们表明，以前的对抗性防御方法，当应用于即时调整范例时，遭受梯度混淆，并且容易受到适应性攻击。我们介绍了Adapt，这是一种在即时调整范式中执行自适应对抗性训练的新框架。我们的方法获得了~40%的W.r.t.SOTA稳健性方法采用全模型微调，只需调整~1%的参数即可。



## **42. Do Unlearning Methods Remove Information from Language Model Weights?**

取消学习方法会从语言模型权重中删除信息吗？ cs.LG

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2410.08827v3) [paper-pdf](http://arxiv.org/pdf/2410.08827v3)

**Authors**: Aghyad Deeb, Fabien Roger

**Abstract**: Large Language Models' knowledge of how to perform cyber-security attacks, create bioweapons, and manipulate humans poses risks of misuse. Previous work has proposed methods to unlearn this knowledge. Historically, it has been unclear whether unlearning techniques are removing information from the model weights or just making it harder to access. To disentangle these two objectives, we propose an adversarial evaluation method to test for the removal of information from model weights: we give an attacker access to some facts that were supposed to be removed, and using those, the attacker tries to recover other facts from the same distribution that cannot be guessed from the accessible facts. We show that using fine-tuning on the accessible facts can recover 88% of the pre-unlearning accuracy when applied to current unlearning methods for information learned during pretraining, revealing the limitations of these methods in removing information from the model weights. Our results also suggest that unlearning evaluations that measure unlearning robustness on information learned during an additional fine-tuning phase may overestimate robustness compared to evaluations that attempt to unlearn information learned during pretraining.

摘要: 大型语言模型在如何执行网络安全攻击、创建生物武器和操纵人类方面的知识构成了误用的风险。以前的工作已经提出了忘记这一知识的方法。从历史上看，人们一直不清楚遗忘技术是在移除模型重量中的信息，还是只是增加了获取信息的难度。为了分离这两个目标，我们提出了一种对抗性评估方法来测试从模型权重中移除信息的情况：我们允许攻击者访问一些应该被移除的事实，并且使用这些事实，攻击者试图从相同的分布中恢复无法从可访问的事实中猜测的其他事实。结果表明，对可访问的事实进行微调可以恢复88%的预忘学习准确率，当应用于现有的遗忘方法时，这些方法在去除模型权重中的信息方面存在局限性。我们的结果还表明，与试图忘却在预训练中学习的信息的评估相比，衡量在额外微调阶段学习到的信息的遗忘健壮性的遗忘评估可能高估了健壮性。



## **43. Federated Learning for Anomaly Detection in Energy Consumption Data: Assessing the Vulnerability to Adversarial Attacks**

用于能源消耗数据异常检测的联邦学习：评估对抗性攻击的脆弱性 cs.LG

12th IEEE Conference on Technologies for Sustainability

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.05041v1) [paper-pdf](http://arxiv.org/pdf/2502.05041v1)

**Authors**: Yohannis Kifle Telila, Damitha Senevirathne, Dumindu Tissera, Apurva Narayan, Miriam A. M. Capretz, Katarina Grolinger

**Abstract**: Anomaly detection is crucial in the energy sector to identify irregular patterns indicating equipment failures, energy theft, or other issues. Machine learning techniques for anomaly detection have achieved great success, but are typically centralized, involving sharing local data with a central server which raises privacy and security concerns. Federated Learning (FL) has been gaining popularity as it enables distributed learning without sharing local data. However, FL depends on neural networks, which are vulnerable to adversarial attacks that manipulate data, leading models to make erroneous predictions. While adversarial attacks have been explored in the image domain, they remain largely unexplored in time series problems, especially in the energy domain. Moreover, the effect of adversarial attacks in the FL setting is also mostly unknown. This paper assesses the vulnerability of FL-based anomaly detection in energy data to adversarial attacks. Specifically, two state-of-the-art models, Long Short Term Memory (LSTM) and Transformers, are used to detect anomalies in an FL setting, and two white-box attack methods, Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD), are employed to perturb the data. The results show that FL is more sensitive to PGD attacks than to FGSM attacks, attributed to PGD's iterative nature, resulting in an accuracy drop of over 10% even with naive, weaker attacks. Moreover, FL is more affected by these attacks than centralized learning, highlighting the need for defense mechanisms in FL.

摘要: 异常检测在能源部门至关重要，可以识别表明设备故障、能源盗窃或其他问题的不规则模式。用于异常检测的机器学习技术取得了巨大成功，但通常是集中式的，涉及与中央服务器共享本地数据，这会引起隐私和安全问题。联合学习(FL)由于能够在不共享本地数据的情况下实现分布式学习而越来越受欢迎。然而，FL依赖于神经网络，而神经网络容易受到操纵数据的敌意攻击，导致模型做出错误的预测。虽然对抗性攻击已经在图像领域得到了探索，但在时间序列问题上，特别是在能量领域，它们在很大程度上仍未被探索。此外，在外语环境下，对抗性攻击的效果也大多是未知的。本文评估了基于FL的能量数据异常检测在敌意攻击下的脆弱性。具体地说，两个最新的模型，长期短期记忆(LSTM)和变形金刚，被用来检测FL环境中的异常，两种白盒攻击方法，快速梯度符号方法(FGSM)和投影梯度下降方法(PGD)，被用来扰动数据。结果表明，FL对PGD攻击比FGSM攻击更敏感，这归因于PGD的迭代性质，导致即使是幼稚、较弱的攻击，准确率也会下降10%以上。此外，外语受这些攻击的影响比集中学习更大，这突显了外语学习中防御机制的必要性。



## **44. Robust Graph Learning Against Adversarial Evasion Attacks via Prior-Free Diffusion-Based Structure Purification**

通过无先验扩散结构纯化来对抗对抗规避攻击的鲁棒图学习 cs.LG

Accepted for poster at WWW 2025

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.05000v1) [paper-pdf](http://arxiv.org/pdf/2502.05000v1)

**Authors**: Jiayi Luo, Qingyun Sun, Haonan Yuan, Xingcheng Fu, Jianxin Li

**Abstract**: Adversarial evasion attacks pose significant threats to graph learning, with lines of studies that have improved the robustness of Graph Neural Networks (GNNs). However, existing works rely on priors about clean graphs or attacking strategies, which are often heuristic and inconsistent. To achieve robust graph learning over different types of evasion attacks and diverse datasets, we investigate this problem from a prior-free structure purification perspective. Specifically, we propose a novel Diffusion-based Structure Purification framework named DiffSP, which creatively incorporates the graph diffusion model to learn intrinsic distributions of clean graphs and purify the perturbed structures by removing adversaries under the direction of the captured predictive patterns without relying on priors. DiffSP is divided into the forward diffusion process and the reverse denoising process, during which structure purification is achieved. To avoid valuable information loss during the forward process, we propose an LID-driven nonisotropic diffusion mechanism to selectively inject noise anisotropically. To promote semantic alignment between the clean graph and the purified graph generated during the reverse process, we reduce the generation uncertainty by the proposed graph transfer entropy guided denoising mechanism. Extensive experiments demonstrate the superior robustness of DiffSP against evasion attacks.

摘要: 对抗性逃避攻击对图学习构成了重大威胁，已有一系列研究提高了图神经网络(GNN)的稳健性。然而，现有的工作依赖于关于干净图或攻击策略的先验知识，这些先验知识往往是启发式的和不一致的。为了在不同类型的逃避攻击和不同的数据集上实现稳健的图学习，我们从无先验结构净化的角度研究了这个问题。具体地说，我们提出了一种新的基于扩散的结构净化框架DiffSP，它创造性地结合了图扩散模型来学习干净图的内在分布，并通过在捕获的预测模式的指导下去除对手来净化扰动结构，而不依赖于先验。DiffSP分为正向扩散过程和反向去噪过程，在这两个过程中实现了结构净化。为了避免正演过程中有价值的信息丢失，我们提出了一种盖子驱动的非各向同性扩散机制来选择性地各向异性地注入噪声。为了促进反求过程中生成的清洁图和净化图之间的语义对齐，我们通过提出的图传递熵引导的去噪机制来降低生成的不确定性。大量实验证明了DiffSP对逃避攻击具有良好的健壮性。



## **45. Securing 5G Bootstrapping: A Two-Layer IBS Authentication Protocol**

确保5G引导：两层IBS认证协议 cs.CR

13 pages, 4 figures, 3 tables. This work has been submitted to the  IEEE for possible publication

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.04915v1) [paper-pdf](http://arxiv.org/pdf/2502.04915v1)

**Authors**: Yilu Dong, Rouzbeh Behnia, Attila A. Yavuz, Syed Rafiul Hussain

**Abstract**: The lack of authentication during the initial bootstrapping phase between cellular devices and base stations allows attackers to deploy fake base stations and send malicious messages to the devices. These attacks have been a long-existing problem in cellular networks, enabling adversaries to launch denial-of-service (DoS), information leakage, and location-tracking attacks. While some defense mechanisms are introduced in 5G, (e.g., encrypting user identifiers to mitigate IMSI catchers), the initial communication between devices and base stations remains unauthenticated, leaving a critical security gap. To address this, we propose E2IBS, a novel and efficient two-layer identity-based signature scheme designed for seamless integration with existing cellular protocols. We implement E2IBS on an open-source 5G stack and conduct a comprehensive performance evaluation against alternative solutions. Compared to the state-of-the-art Schnorr-HIBS, E2IBS reduces attack surfaces, enables fine-grained lawful interception, and achieves 2x speed in verification, making it a practical solution for securing 5G base station authentication.

摘要: 在蜂窝设备和基站之间的初始引导阶段缺乏身份验证，使得攻击者能够部署伪基站并向设备发送恶意消息。这些攻击是蜂窝网络中长期存在的问题，使攻击者能够发起拒绝服务(DoS)、信息泄漏和位置跟踪攻击。虽然在5G中引入了一些防御机制(例如，加密用户标识以减少IMSI捕获器)，但设备和基站之间的初始通信仍未经过身份验证，留下了一个严重的安全漏洞。为了解决这一问题，我们提出了一种新颖高效的两层基于身份的签名方案E2IBS，旨在与现有的蜂窝协议无缝集成。我们在开源5G堆栈上实施E2IBS，并针对替代解决方案进行全面的性能评估。与最先进的Schnorr-Hibs相比，E2IBS减少了攻击面，实现了细粒度的合法拦截，验证速度达到了2倍，是5G基站认证安全的实用解决方案。



## **46. From Allies to Adversaries: Manipulating LLM Tool-Calling through Adversarial Injection**

从盟友到对手：通过对抗注入操纵LLM工具调用 cs.CR

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2412.10198v2) [paper-pdf](http://arxiv.org/pdf/2412.10198v2)

**Authors**: Haowei Wang, Rupeng Zhang, Junjie Wang, Mingyang Li, Yuekai Huang, Dandan Wang, Qing Wang

**Abstract**: Tool-calling has changed Large Language Model (LLM) applications by integrating external tools, significantly enhancing their functionality across diverse tasks. However, this integration also introduces new security vulnerabilities, particularly in the tool scheduling mechanisms of LLM, which have not been extensively studied. To fill this gap, we present ToolCommander, a novel framework designed to exploit vulnerabilities in LLM tool-calling systems through adversarial tool injection. Our framework employs a well-designed two-stage attack strategy. Firstly, it injects malicious tools to collect user queries, then dynamically updates the injected tools based on the stolen information to enhance subsequent attacks. These stages enable ToolCommander to execute privacy theft, launch denial-of-service attacks, and even manipulate business competition by triggering unscheduled tool-calling. Notably, the ASR reaches 91.67% for privacy theft and hits 100% for denial-of-service and unscheduled tool calling in certain cases. Our work demonstrates that these vulnerabilities can lead to severe consequences beyond simple misuse of tool-calling systems, underscoring the urgent need for robust defensive strategies to secure LLM Tool-calling systems.

摘要: 工具调用通过集成外部工具改变了大型语言模型(LLM)应用程序，显著增强了它们在不同任务中的功能。然而，这种集成也引入了新的安全漏洞，特别是在LLM的工具调度机制中，这些漏洞还没有得到广泛的研究。为了填补这一空白，我们提出了一种新的框架，它旨在通过敌意工具注入来利用LLM工具调用系统中的漏洞。我们的框架采用了精心设计的两阶段攻击策略。它首先注入恶意工具来收集用户查询，然后根据窃取的信息动态更新注入的工具，以加强后续攻击。这些阶段使工具指挥官能够执行隐私窃取、发起拒绝服务攻击，甚至通过触发计划外的工具调用来操纵业务竞争。值得注意的是，在某些情况下，隐私窃取的ASR达到91.67%，拒绝服务和非计划工具调用的ASR达到100%。我们的工作表明，这些漏洞可能导致严重后果，而不仅仅是简单地滥用工具调用系统，这突显了迫切需要强大的防御战略来保护LLM工具调用系统。



## **47. DMPA: Model Poisoning Attacks on Decentralized Federated Learning for Model Differences**

DMPA：因模型差异而对去中心联邦学习的模型中毒攻击 cs.LG

8 pages, 3 figures

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.04771v1) [paper-pdf](http://arxiv.org/pdf/2502.04771v1)

**Authors**: Chao Feng, Yunlong Li, Yuanzhe Gao, Alberto Huertas Celdrán, Jan von der Assen, Gérôme Bovet, Burkhard Stiller

**Abstract**: Federated learning (FL) has garnered significant attention as a prominent privacy-preserving Machine Learning (ML) paradigm. Decentralized FL (DFL) eschews traditional FL's centralized server architecture, enhancing the system's robustness and scalability. However, these advantages of DFL also create new vulnerabilities for malicious participants to execute adversarial attacks, especially model poisoning attacks. In model poisoning attacks, malicious participants aim to diminish the performance of benign models by creating and disseminating the compromised model. Existing research on model poisoning attacks has predominantly concentrated on undermining global models within the Centralized FL (CFL) paradigm, while there needs to be more research in DFL. To fill the research gap, this paper proposes an innovative model poisoning attack called DMPA. This attack calculates the differential characteristics of multiple malicious client models and obtains the most effective poisoning strategy, thereby orchestrating a collusive attack by multiple participants. The effectiveness of this attack is validated across multiple datasets, with results indicating that the DMPA approach consistently surpasses existing state-of-the-art FL model poisoning attack strategies.

摘要: 联合学习(FL)作为一种重要的隐私保护机器学习(ML)范型已经引起了人们的广泛关注。分散式FL(DFL)避开了传统FL的集中式服务器架构，增强了系统的健壮性和可扩展性。然而，DFL的这些优势也为恶意参与者执行对抗性攻击，特别是模型中毒攻击创造了新的漏洞。在模型中毒攻击中，恶意参与者旨在通过创建和传播受危害的模型来降低良性模型的性能。现有的关于模型中毒攻击的研究主要集中在集中式FL(CFL)范式中破坏全局模型，而在DFL(DFL)中还需要更多的研究。为了填补这一研究空白，本文提出了一种新的中毒攻击模型DMPA。该攻击计算了多个恶意客户端模型的差异特征，获得了最有效的中毒策略，从而策划了多个参与者的合谋攻击。该攻击的有效性在多个数据集上得到了验证，结果表明，DMPA方法始终优于现有的最先进的FL模型中毒攻击策略。



## **48. Real-Time Privacy Risk Measurement with Privacy Tokens for Gradient Leakage**

使用隐私令牌进行实时隐私风险测量以应对梯度泄漏 cs.LG

There is something wrong with the order of Figures 8-11. And I need  to add an experiment with differential privacy quantization mutual  information value

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.02913v3) [paper-pdf](http://arxiv.org/pdf/2502.02913v3)

**Authors**: Jiayang Meng, Tao Huang, Hong Chen, Xin Shi, Qingyu Huang, Chen Hou

**Abstract**: The widespread deployment of deep learning models in privacy-sensitive domains has amplified concerns regarding privacy risks, particularly those stemming from gradient leakage during training. Current privacy assessments primarily rely on post-training attack simulations. However, these methods are inherently reactive, unable to encompass all potential attack scenarios, and often based on idealized adversarial assumptions. These limitations underscore the need for proactive approaches to privacy risk assessment during the training process. To address this gap, we propose the concept of privacy tokens, which are derived directly from private gradients during training. Privacy tokens encapsulate gradient features and, when combined with data features, offer valuable insights into the extent of private information leakage from training data, enabling real-time measurement of privacy risks without relying on adversarial attack simulations. Additionally, we employ Mutual Information (MI) as a robust metric to quantify the relationship between training data and gradients, providing precise and continuous assessments of privacy leakage throughout the training process. Extensive experiments validate our framework, demonstrating the effectiveness of privacy tokens and MI in identifying and quantifying privacy risks. This proactive approach marks a significant advancement in privacy monitoring, promoting the safer deployment of deep learning models in sensitive applications.

摘要: 深度学习模型在隐私敏感领域的广泛部署加剧了人们对隐私风险的担忧，特别是培训期间梯度泄漏造成的风险。目前的隐私评估主要依赖于训练后的攻击模拟。然而，这些方法本质上是被动的，无法涵盖所有潜在的攻击场景，并且通常基于理想化的对抗性假设。这些限制强调了在培训过程中对隐私风险评估采取积极主动的方法的必要性。为了弥补这一差距，我们提出了隐私令牌的概念，它直接从训练过程中的隐私梯度派生出来。隐私令牌封装了梯度特征，当与数据特征相结合时，可以提供对训练数据中私人信息泄漏程度的有价值的见解，从而能够实时测量隐私风险，而不需要依赖对抗性攻击模拟。此外，我们使用相互信息(MI)作为一个稳健的度量来量化训练数据和梯度之间的关系，在整个训练过程中提供对隐私泄露的准确和连续的评估。大量的实验验证了我们的框架，证明了隐私令牌和MI在识别和量化隐私风险方面的有效性。这种主动的方法标志着隐私监控方面的重大进步，促进了在敏感应用程序中更安全地部署深度学习模型。



## **49. Mechanistic Understandings of Representation Vulnerabilities and Engineering Robust Vision Transformers**

对表示漏洞的机械理解和工程鲁棒视觉转换器 cs.CV

10 pages, 5 figures

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.04679v1) [paper-pdf](http://arxiv.org/pdf/2502.04679v1)

**Authors**: Chashi Mahiul Islam, Samuel Jacob Chacko, Mao Nishino, Xiuwen Liu

**Abstract**: While transformer-based models dominate NLP and vision applications, their underlying mechanisms to map the input space to the label space semantically are not well understood. In this paper, we study the sources of known representation vulnerabilities of vision transformers (ViT), where perceptually identical images can have very different representations and semantically unrelated images can have the same representation. Our analysis indicates that imperceptible changes to the input can result in significant representation changes, particularly in later layers, suggesting potential instabilities in the performance of ViTs. Our comprehensive study reveals that adversarial effects, while subtle in early layers, propagate and amplify through the network, becoming most pronounced in middle to late layers. This insight motivates the development of NeuroShield-ViT, a novel defense mechanism that strategically neutralizes vulnerable neurons in earlier layers to prevent the cascade of adversarial effects. We demonstrate NeuroShield-ViT's effectiveness across various attacks, particularly excelling against strong iterative attacks, and showcase its remarkable zero-shot generalization capabilities. Without fine-tuning, our method achieves a competitive accuracy of 77.8% on adversarial examples, surpassing conventional robustness methods. Our results shed new light on how adversarial effects propagate through ViT layers, while providing a promising approach to enhance the robustness of vision transformers against adversarial attacks. Additionally, they provide a promising approach to enhance the robustness of vision transformers against adversarial attacks.

摘要: 虽然基于转换器的模型在NLP和VISION应用中占据主导地位，但它们将输入空间语义映射到标签空间的底层机制还没有得到很好的理解。在这篇文章中，我们研究了视觉转换器(VIT)的已知表征漏洞的来源，其中感知相同的图像可以具有非常不同的表征，而语义无关的图像可以具有相同的表征。我们的分析表明，对输入的不可察觉的变化会导致显著的表示变化，特别是在较晚的层中，这表明VITS的性能存在潜在的不稳定性。我们的综合研究表明，对抗性效应，虽然在早期层微妙，但通过网络传播和放大，在中后期变得最明显。这种洞察力推动了NeuroShield-Vit的发展，这是一种新的防御机制，战略性地中和早期层中的脆弱神经元，以防止级联的对抗性效应。我们展示了NeuroShield-VIT在各种攻击中的有效性，特别是在对抗强大的迭代攻击方面的出色表现，并展示了其非凡的零命中泛化能力。在没有微调的情况下，我们的方法在对抗性样本上的准确率达到了77.8%，超过了传统的稳健性方法。我们的结果揭示了对抗性效应是如何通过VIT层传播的，同时提供了一种有希望的方法来增强视觉转换器对抗对抗性攻击的健壮性。此外，它们还提供了一种很有前途的方法来增强视觉转换器对对手攻击的稳健性。



## **50. Regularized Robustly Reliable Learners and Instance Targeted Attacks**

正规的鲁棒可靠的学习者和实例有针对性的攻击 cs.LG

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2410.10572v2) [paper-pdf](http://arxiv.org/pdf/2410.10572v2)

**Authors**: Avrim Blum, Donya Saless

**Abstract**: Instance-targeted data poisoning attacks, where an adversary corrupts a training set to induce errors on specific test points, have raised significant concerns. Balcan et al (2022) proposed an approach to addressing this challenge by defining a notion of robustly-reliable learners that provide per-instance guarantees of correctness under well-defined assumptions, even in the presence of data poisoning attacks. They then give a generic optimal (but computationally inefficient) robustly reliable learner as well as a computationally efficient algorithm for the case of linear separators over log-concave distributions.   In this work, we address two challenges left open by Balcan et al (2022). The first is that the definition of robustly-reliable learners in Balcan et al (2022) becomes vacuous for highly-flexible hypothesis classes: if there are two classifiers h_0, h_1 \in H both with zero error on the training set such that h_0(x) \neq h_1(x), then a robustly-reliable learner must abstain on x. We address this problem by defining a modified notion of regularized robustly-reliable learners that allows for nontrivial statements in this case. The second is that the generic algorithm of Balcan et al (2022) requires re-running an ERM oracle (essentially, retraining the classifier) on each test point x, which is generally impractical even if ERM can be implemented efficiently. To tackle this problem, we show that at least in certain interesting cases we can design algorithms that can produce their outputs in time sublinear in training time, by using techniques from dynamic algorithm design.

摘要: 针对实例的数据中毒攻击，即对手破坏训练集以在特定测试点上引发错误，已经引起了严重的担忧。Balcan等人(2022)提出了一种应对这一挑战的方法，定义了稳健可靠的学习者的概念，即使在存在数据中毒攻击的情况下，也可以在定义明确的假设下提供逐个实例的正确性保证。然后，对于对数凹分布上的线性分隔符的情况，他们给出了一个通用的最优(但计算效率低)鲁棒可靠的学习器以及一个计算高效的算法。在这项工作中，我们解决了Balcan等人(2022)留下的两个挑战。首先，对于高度灵活的假设类，Balcan等人(2022)中的稳健可靠学习者的定义变得空洞：如果在训练集上存在两个都是零误差的分类器h_0，h_1\in H，使得h_0(X)\neq h_1(X)，那么稳健可靠的学习者必须在x上弃权。我们通过定义一个修正的正则化稳健可靠学习者的概念来解决这个问题，它允许在这种情况下非平凡的陈述。其次，Balcan等人(2022)的通用算法需要在每个测试点x上重新运行ERM预言(本质上是重新训练分类器)，即使ERM可以有效地实施，这通常也是不切实际的。为了解决这个问题，我们证明了，至少在某些有趣的情况下，我们可以设计算法，通过使用动态算法设计的技术，在训练时间内产生时间上的次线性输出。



