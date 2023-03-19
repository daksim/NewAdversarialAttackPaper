# Latest Adversarial Attack Papers
**update at 2023-03-19 16:27:03**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Among Us: Adversarially Robust Collaborative Perception by Consensus**

在我们中间：基于共识的相反的强健协作感知 cs.RO

**SubmitDate**: 2023-03-16    [abs](http://arxiv.org/abs/2303.09495v1) [paper-pdf](http://arxiv.org/pdf/2303.09495v1)

**Authors**: Yiming Li, Qi Fang, Jiamu Bai, Siheng Chen, Felix Juefei-Xu, Chen Feng

**Abstract**: Multiple robots could perceive a scene (e.g., detect objects) collaboratively better than individuals, although easily suffer from adversarial attacks when using deep learning. This could be addressed by the adversarial defense, but its training requires the often-unknown attacking mechanism. Differently, we propose ROBOSAC, a novel sampling-based defense strategy generalizable to unseen attackers. Our key idea is that collaborative perception should lead to consensus rather than dissensus in results compared to individual perception. This leads to our hypothesize-and-verify framework: perception results with and without collaboration from a random subset of teammates are compared until reaching a consensus. In such a framework, more teammates in the sampled subset often entail better perception performance but require longer sampling time to reject potential attackers. Thus, we derive how many sampling trials are needed to ensure the desired size of an attacker-free subset, or equivalently, the maximum size of such a subset that we can successfully sample within a given number of trials. We validate our method on the task of collaborative 3D object detection in autonomous driving scenarios.

摘要: 多个机器人可以比个人更好地协作感知场景(例如，检测对象)，尽管在使用深度学习时很容易受到对抗性攻击。这可以通过对抗性防守来解决，但它的训练需要往往未知的攻击机制。不同的是，我们提出了ROBOSAC，一种新的基于采样的防御策略，可以推广到看不见的攻击者。我们的关键思想是，与个人感知相比，合作感知应该在结果中导致共识，而不是分歧。这导致了我们的假设和验证框架：对随机的队友子集进行协作和不协作的感知结果进行比较，直到达成共识。在这样的框架中，采样子集中更多的队友通常会带来更好的感知性能，但需要更长的采样时间来拒绝潜在的攻击者。因此，我们推导出需要多少次抽样试验才能确保没有攻击者的子集的期望大小，或者等价地，在给定的试验次数内可以成功抽样的子集的最大大小。我们在自主驾驶场景下的协同3D目标检测任务中验证了我们的方法。



## **2. Rickrolling the Artist: Injecting Backdoors into Text Encoders for Text-to-Image Synthesis**

点击艺术家：将后门注入文本编码器以进行文本到图像的合成 cs.LG

30 pages, 20 figures, 5 tables

**SubmitDate**: 2023-03-16    [abs](http://arxiv.org/abs/2211.02408v2) [paper-pdf](http://arxiv.org/pdf/2211.02408v2)

**Authors**: Lukas Struppek, Dominik Hintersdorf, Kristian Kersting

**Abstract**: While text-to-image synthesis currently enjoys great popularity among researchers and the general public, the security of these models has been neglected so far. Many text-guided image generation models rely on pre-trained text encoders from external sources, and their users trust that the retrieved models will behave as promised. Unfortunately, this might not be the case. We introduce backdoor attacks against text-guided generative models and demonstrate that their text encoders pose a major tampering risk. Our attacks only slightly alter an encoder so that no suspicious model behavior is apparent for image generations with clean prompts. By then inserting a single character trigger into the prompt, e.g., a non-Latin character or emoji, the adversary can trigger the model to either generate images with pre-defined attributes or images following a hidden, potentially malicious description. We empirically demonstrate the high effectiveness of our attacks on Stable Diffusion and highlight that the injection process of a single backdoor takes less than two minutes. Besides phrasing our approach solely as an attack, it can also force an encoder to forget phrases related to certain concepts, such as nudity or violence, and help to make image generation safer.

摘要: 虽然文本到图像的合成目前在研究人员和普通大众中很受欢迎，但到目前为止，这些模型的安全性一直被忽视。许多文本制导的图像生成模型依赖于来自外部来源的预先训练的文本编码器，它们的用户相信检索到的模型将如承诺的那样运行。不幸的是，情况可能并非如此。我们引入了针对文本引导的生成模型的后门攻击，并证明了它们的文本编码器构成了主要的篡改风险。我们的攻击只略微改变了编码器，因此对于具有干净提示的图像生成来说，没有明显的可疑模型行为。然后，通过将单个字符触发器(例如，非拉丁字符或表情符号)插入到提示中，对手可以触发模型以生成具有预定义属性的图像或者在隐藏的潜在恶意描述之后生成图像。我们从经验上证明了我们对稳定扩散攻击的高效性，并强调了单个后门的注入过程只需不到两分钟。除了将我们的方法仅作为一种攻击来表述外，它还可以迫使编码者忘记与某些概念相关的短语，如裸体或暴力，并有助于使图像生成更安全。



## **3. Image Classifiers Leak Sensitive Attributes About Their Classes**

图像分类器泄漏有关其类的敏感属性 cs.LG

40 pages, 32 figures, 4 tables

**SubmitDate**: 2023-03-16    [abs](http://arxiv.org/abs/2303.09289v1) [paper-pdf](http://arxiv.org/pdf/2303.09289v1)

**Authors**: Lukas Struppek, Dominik Hintersdorf, Felix Friedrich, Manuel Brack, Patrick Schramowski, Kristian Kersting

**Abstract**: Neural network-based image classifiers are powerful tools for computer vision tasks, but they inadvertently reveal sensitive attribute information about their classes, raising concerns about their privacy. To investigate this privacy leakage, we introduce the first Class Attribute Inference Attack (Caia), which leverages recent advances in text-to-image synthesis to infer sensitive attributes of individual classes in a black-box setting, while remaining competitive with related white-box attacks. Our extensive experiments in the face recognition domain show that Caia can accurately infer undisclosed sensitive attributes, such as an individual's hair color, gender and racial appearance, which are not part of the training labels. Interestingly, we demonstrate that adversarial robust models are even more vulnerable to such privacy leakage than standard models, indicating that a trade-off between robustness and privacy exists.

摘要: 基于神经网络的图像分类器是计算机视觉任务的强大工具，但它们无意中泄露了有关其类别的敏感属性信息，引发了对其隐私的担忧。为了调查这种隐私泄露，我们引入了第一类属性推理攻击(CAIA)，它利用文本到图像合成的最新进展来推断黑盒环境中个别类的敏感属性，同时保持与相关白盒攻击的竞争力。我们在人脸识别领域的广泛实验表明，CAIA可以准确地推断出未披露的敏感属性，如个人的头发颜色、性别和种族外观，这些属性不属于训练标签的一部分。有趣的是，我们证明了对抗性稳健模型比标准模型更容易受到这种隐私泄露的影响，这表明存在稳健性和隐私之间的权衡。



## **4. QuickSync: A Quickly Synchronizing PoS-Based Blockchain Protocol**

QuickSync：一种基于PoS的区块链快速同步协议 cs.CR

**SubmitDate**: 2023-03-16    [abs](http://arxiv.org/abs/2005.03564v4) [paper-pdf](http://arxiv.org/pdf/2005.03564v4)

**Authors**: Shoeb Siddiqui, Varul Srivastava, Raj Maheshwari, Sujit Gujar

**Abstract**: To implement a blockchain, we need a blockchain protocol for all the nodes to follow. To design a blockchain protocol, we need a block publisher selection mechanism and a chain selection rule. In Proof-of-Stake (PoS) based blockchain protocols, block publisher selection mechanism selects the node to publish the next block based on the relative stake held by the node. However, PoS protocols, such as Ouroboros v1, may face vulnerability to fully adaptive corruptions.   In this paper, we propose a novel PoS-based blockchain protocol, QuickSync, to achieve security against fully adaptive corruptions while improving on performance. We propose a metric called block power, a value defined for each block, derived from the output of the verifiable random function based on the digital signature of the block publisher. With this metric, we compute chain power, the sum of block powers of all the blocks comprising the chain, for all the valid chains. These metrics are a function of the block publisher's stake to enable the PoS aspect of the protocol. The chain selection rule selects the chain with the highest chain power as the one to extend. This chain selection rule hence determines the selected block publisher of the previous block. When we use metrics to define the chain selection rule, it may lead to vulnerabilities against Sybil attacks. QuickSync uses a Sybil attack resistant function implemented using histogram matching. We prove that QuickSync satisfies common prefix, chain growth, and chain quality properties and hence it is secure. We also show that it is resilient to different types of adversarial attack strategies. Our analysis demonstrates that QuickSync performs better than Bitcoin by an order of magnitude on both transactions per second and time to finality, and better than Ouroboros v1 by a factor of three on time to finality.

摘要: 要实现区块链，我们需要一个区块链协议，让所有节点都能遵循。要设计区块链协议，我们需要一个区块发布者选择机制和一个链选择规则。在基于利害关系证明(POS)的区块链协议中，块发布者选择机制根据节点持有的相对利害关系来选择要发布下一个块的节点。然而，Ouroboros v1等POS协议可能面临完全自适应损坏的漏洞。在本文中，我们提出了一种新的基于PoS的区块链协议QuickSync，该协议在提高性能的同时，实现了对完全自适应攻击的安全性。我们提出了一种称为块功率的度量，这是一个为每个块定义的值，该值来自基于块发布者的数字签名的可验证随机函数的输出。使用此度量，我们计算所有有效链的链功率，即构成链的所有块的块功率之和。这些度量是块发布者的股份的函数，以启用协议的POS方面。链选择规则选择链功率最高的链作为要延伸的链。因此，该链选择规则确定前一个块的选定块发布者。当我们使用度量来定义链选择规则时，可能会导致对Sybil攻击的漏洞。QuickSync使用通过直方图匹配实现的Sybil攻击抵抗功能。我们证明了QuickSync满足公共前缀、链增长和链质量性质，因此是安全的。我们还证明了它对不同类型的对抗性攻击策略具有很强的弹性。我们的分析表明，QuickSync在每秒交易量和完成时间上都比比特币好一个数量级，在完成之前的时间上比Ouroboros v1高出三倍。



## **5. Security of Blockchains at Capacity**

区块链的最大容量安全 cs.CR

**SubmitDate**: 2023-03-16    [abs](http://arxiv.org/abs/2303.09113v1) [paper-pdf](http://arxiv.org/pdf/2303.09113v1)

**Authors**: Lucianna Kiffer, Joachim Neu, Srivatsan Sridhar, Aviv Zohar, David Tse

**Abstract**: Given a network of nodes with certain communication and computation capacities, what is the maximum rate at which a blockchain can run securely? We study this question for proof-of-work (PoW) and proof-of-stake (PoS) longest chain protocols under a 'bounded bandwidth' model which captures queuing and processing delays due to high block rate relative to capacity, bursty release of adversarial blocks, and in PoS, spamming due to equivocations.   We demonstrate that security of both PoW and PoS longest chain, when operating at capacity, requires carefully designed scheduling policies that correctly prioritize which blocks are processed first, as we show attack strategies tailored to such policies. In PoS, we show an attack exploiting equivocations, which highlights that the throughput of the PoS longest chain protocol with a broad class of scheduling policies must decrease as the desired security error probability decreases. At the same time, through an improved analysis method, our work is the first to identify block production rates under which PoW longest chain is secure in the bounded bandwidth setting. We also present the first PoS longest chain protocol, SaPoS, which is secure with a block production rate independent of the security error probability, by using an 'equivocation removal' policy to prevent equivocation spamming.

摘要: 给定一个具有一定通信和计算能力的节点网络，区块链可以安全运行的最大速率是多少？我们研究了工作证明(PoW)和利害关系证明(POS)最长链协议在有界带宽模型下的这个问题，该模型捕获了由于相对于容量的高阻塞率、敌意块的突发释放以及在POS中由于模棱两可而导致的垃圾邮件的排队和处理延迟。我们演示了PoW和PoS最长链在满负荷运行时的安全性，需要精心设计的调度策略正确地优先处理哪些块，正如我们展示的针对此类策略量身定做的攻击策略。在PoS中，我们展示了一个利用模棱两可的攻击，它强调了具有广泛类别的调度策略的PoS最长链协议的吞吐量必须随着期望的安全错误概率的降低而降低。同时，通过一种改进的分析方法，我们的工作首次确定了在有限带宽环境下PoW最长链安全的块生产速率。我们还提出了第一个PoS最长链协议SAOS，它是安全的，其块生成率与安全错误概率无关，通过使用一种模糊去除策略来防止模糊垃圾邮件。



## **6. Rethinking Model Ensemble in Transfer-based Adversarial Attacks**

基于迁移的对抗性攻击中模型集成的再思考 cs.CV

**SubmitDate**: 2023-03-16    [abs](http://arxiv.org/abs/2303.09105v1) [paper-pdf](http://arxiv.org/pdf/2303.09105v1)

**Authors**: Huanran Chen, Yichi Zhang, Yinpeng Dong, Jun Zhu

**Abstract**: Deep learning models are vulnerable to adversarial examples. Transfer-based adversarial attacks attract tremendous attention as they can identify the weaknesses of deep learning models in a black-box manner. An effective strategy to improve the transferability of adversarial examples is attacking an ensemble of models. However, previous works simply average the outputs of different models, lacking an in-depth analysis on how and why model ensemble can strongly improve the transferability. In this work, we rethink the ensemble in adversarial attacks and define the common weakness of model ensemble with the properties of the flatness of loss landscape and the closeness to the local optimum of each model. We empirically and theoretically show that these two properties are strongly correlated with the transferability and propose a Common Weakness Attack (CWA) to generate more transferable adversarial examples by promoting these two properties. Experimental results on both image classification and object detection tasks validate the effectiveness of our approach to improve the adversarial transferability, especially when attacking adversarially trained models.

摘要: 深度学习模型很容易受到对抗性例子的影响。基于迁移的对抗性攻击由于能够以黑箱方式识别深度学习模型的弱点而引起了极大的关注。提高对抗性例子可转移性的一个有效策略是攻击模型集合。然而，以往的工作只是简单地对不同模型的输出进行平均，而缺乏对模型集成如何以及为什么能够显著提高可转移性的深入分析。在这项工作中，我们对对抗性攻击中的集成进行了重新思考，并用损失图景的平坦性和每个模型接近局部最优的性质定义了模型集成的共同弱点。我们从经验和理论上证明了这两个性质与可转移性有很强的相关性，并提出了一种共同弱点攻击(CWA)，通过提升这两个性质来生成更多可转移的对抗性实例。在图像分类和目标检测任务上的实验结果验证了该方法的有效性，特别是在攻击对抗性训练模型时。



## **7. Well-classified Examples are Underestimated in Classification with Deep Neural Networks**

深度神经网络在分类中低估分类好的样本 cs.LG

Accepted by AAAI 2022; 18 pages, 11 figures, 13 tables

**SubmitDate**: 2023-03-16    [abs](http://arxiv.org/abs/2110.06537v6) [paper-pdf](http://arxiv.org/pdf/2110.06537v6)

**Authors**: Guangxiang Zhao, Wenkai Yang, Xuancheng Ren, Lei Li, Yunfang Wu, Xu Sun

**Abstract**: The conventional wisdom behind learning deep classification models is to focus on bad-classified examples and ignore well-classified examples that are far from the decision boundary. For instance, when training with cross-entropy loss, examples with higher likelihoods (i.e., well-classified examples) contribute smaller gradients in back-propagation. However, we theoretically show that this common practice hinders representation learning, energy optimization, and margin growth. To counteract this deficiency, we propose to reward well-classified examples with additive bonuses to revive their contribution to the learning process. This counterexample theoretically addresses these three issues. We empirically support this claim by directly verifying the theoretical results or significant performance improvement with our counterexample on diverse tasks, including image classification, graph classification, and machine translation. Furthermore, this paper shows that we can deal with complex scenarios, such as imbalanced classification, OOD detection, and applications under adversarial attacks because our idea can solve these three issues. Code is available at: https://github.com/lancopku/well-classified-examples-are-underestimated.

摘要: 学习深度分类模型背后的传统智慧是专注于分类不好的例子，而忽略远离决策边界的分类良好的例子。例如，当使用交叉熵损失进行训练时，具有较高似然的示例(即，分类良好的示例)在反向传播中贡献较小的梯度。然而，我们从理论上表明，这种常见的做法阻碍了表示学习、能量优化和利润率增长。为了弥补这一不足，我们建议用额外的奖金奖励分类良好的例子，以恢复他们对学习过程的贡献。这个反例从理论上解决了这三个问题。我们通过直接验证理论结果或通过我们的反例在包括图像分类、图形分类和机器翻译在内的不同任务上的显著性能改进来经验地支持这一论断。此外，本文还表明，我们的思想可以解决这三个问题，因此我们可以处理复杂的场景，如不平衡分类、面向对象的检测和对手攻击下的应用。代码可从以下网址获得：https://github.com/lancopku/well-classified-examples-are-underestimated.



## **8. Robust Evaluation of Diffusion-Based Adversarial Purification**

基于扩散的对抗净化算法的稳健性评价 cs.CV

**SubmitDate**: 2023-03-16    [abs](http://arxiv.org/abs/2303.09051v1) [paper-pdf](http://arxiv.org/pdf/2303.09051v1)

**Authors**: Minjong Lee, Dongwoo Kim

**Abstract**: We question the current evaluation practice on diffusion-based purification methods. Diffusion-based purification methods aim to remove adversarial effects from an input data point at test time. The approach gains increasing attention as an alternative to adversarial training due to the disentangling between training and testing. Well-known white-box attacks are often employed to measure the robustness of the purification. However, it is unknown whether these attacks are the most effective for the diffusion-based purification since the attacks are often tailored for adversarial training. We analyze the current practices and provide a new guideline for measuring the robustness of purification methods against adversarial attacks. Based on our analysis, we further propose a new purification strategy showing competitive results against the state-of-the-art adversarial training approaches.

摘要: 我们对目前基于扩散的纯化方法的评估实践提出了质疑。基于扩散的净化方法旨在消除测试时输入数据点的对抗性影响。由于训练和测试之间的分离，这种方法作为对抗性训练的替代方法受到越来越多的关注。通常使用众所周知的白盒攻击来衡量净化的健壮性。然而，目前尚不清楚这些攻击对于基于扩散的净化是否最有效，因为这些攻击通常是为对抗性训练量身定做的。我们分析了目前的实践，并为衡量净化方法对对手攻击的健壮性提供了新的指导方针。基于我们的分析，我们进一步提出了一种新的净化策略，与最先进的对抗性训练方法相比，显示了竞争的结果。



## **9. DeeBBAA: A benchmark Deep Black Box Adversarial Attack against Cyber-Physical Power Systems**

DeeBBAA：一种针对网络物理电力系统的基准深黑盒对抗性攻击 cs.CR

**SubmitDate**: 2023-03-16    [abs](http://arxiv.org/abs/2303.09024v1) [paper-pdf](http://arxiv.org/pdf/2303.09024v1)

**Authors**: Arnab Bhattacharjee, Tapan K. Saha, Ashu Verma, Sukumar Mishra

**Abstract**: An increased energy demand, and environmental pressure to accommodate higher levels of renewable energy and flexible loads like electric vehicles have led to numerous smart transformations in the modern power systems. These transformations make the cyber-physical power system highly susceptible to cyber-adversaries targeting its numerous operations. In this work, a novel black box adversarial attack strategy is proposed targeting the AC state estimation operation of an unknown power system using historical data. Specifically, false data is injected into the measurements obtained from a small subset of the power system components which leads to significant deviations in the state estimates. Experiments carried out on the IEEE 39 bus and 118 bus test systems make it evident that the proposed strategy, called DeeBBAA, can evade numerous conventional and state-of-the-art attack detection mechanisms with very high probability.

摘要: 能源需求的增加，以及适应更高水平的可再生能源和电动汽车等灵活负载的环境压力，导致了现代电力系统的许多智能转型。这些转变使网络物理电力系统非常容易受到针对其众多操作的网络对手的攻击。针对未知电力系统的交流状态估计操作，提出了一种利用历史数据进行黑箱对抗攻击的新策略。具体地说，虚假数据被注入到从电力系统部件的一小部分获得的测量值中，这导致状态估计的显著偏差。在IEEE 39节点和118节点测试系统上进行的实验表明，所提出的策略DeeBBAA能够以很高的概率避开众多传统和最新的攻击检测机制。



## **10. Identifying Threats, Cybercrime and Digital Forensic Opportunities in Smart City Infrastructure via Threat Modeling**

通过威胁建模识别智能城市基础设施中的威胁、网络犯罪和数字取证机会 cs.CR

Updated to include amendments from peer review process. Accepted in  Forensic Science International: Digital Investigation

**SubmitDate**: 2023-03-16    [abs](http://arxiv.org/abs/2210.14692v2) [paper-pdf](http://arxiv.org/pdf/2210.14692v2)

**Authors**: Yee Ching Tok, Sudipta Chattopadhyay

**Abstract**: Technological advances have enabled multiple countries to consider implementing Smart City Infrastructure to provide in-depth insights into different data points and enhance the lives of citizens. Unfortunately, these new technological implementations also entice adversaries and cybercriminals to execute cyber-attacks and commit criminal acts on these modern infrastructures. Given the borderless nature of cyber attacks, varying levels of understanding of smart city infrastructure and ongoing investigation workloads, law enforcement agencies and investigators would be hard-pressed to respond to these kinds of cybercrime. Without an investigative capability by investigators, these smart infrastructures could become new targets favored by cybercriminals.   To address the challenges faced by investigators, we propose a common definition of smart city infrastructure. Based on the definition, we utilize the STRIDE threat modeling methodology and the Microsoft Threat Modeling Tool to identify threats present in the infrastructure and create a threat model which can be further customized or extended by interested parties. Next, we map offences, possible evidence sources and types of threats identified to help investigators understand what crimes could have been committed and what evidence would be required in their investigation work. Finally, noting that Smart City Infrastructure investigations would be a global multi-faceted challenge, we discuss technical and legal opportunities in digital forensics on Smart City Infrastructure.

摘要: 技术进步使多个国家能够考虑实施智慧城市基础设施，以深入了解不同的数据点，并改善公民的生活。不幸的是，这些新的技术实施也引诱对手和网络罪犯对这些现代基础设施进行网络攻击和犯罪行为。鉴于网络攻击的无边界性质、对智能城市基础设施的不同程度的了解以及正在进行的调查工作量，执法机构和调查人员将很难对此类网络犯罪做出回应。如果调查人员没有调查能力，这些智能基础设施可能会成为网络犯罪分子青睐的新目标。为了应对调查人员面临的挑战，我们提出了智能城市基础设施的共同定义。在定义的基础上，我们利用STRIDE威胁建模方法和Microsoft威胁建模工具来识别基础设施中存在的威胁，并创建可由感兴趣的各方进一步定制或扩展的威胁模型。接下来，我们将绘制罪行、可能的证据来源和已确定的威胁类型的地图，以帮助调查人员了解哪些罪行可能发生，以及在调查工作中需要哪些证据。最后，注意到智能城市基础设施调查将是一项全球多方面的挑战，我们讨论了智能城市基础设施数字取证的技术和法律机会。



## **11. Certifiable (Multi)Robustness Against Patch Attacks Using ERM**

使用ERM实现对补丁攻击的可认证(多)稳健性 cs.LG

**SubmitDate**: 2023-03-15    [abs](http://arxiv.org/abs/2303.08944v1) [paper-pdf](http://arxiv.org/pdf/2303.08944v1)

**Authors**: Saba Ahmadi, Avrim Blum, Omar Montasser, Kevin Stangl

**Abstract**: Consider patch attacks, where at test-time an adversary manipulates a test image with a patch in order to induce a targeted misclassification. We consider a recent defense to patch attacks, Patch-Cleanser (Xiang et al. [2022]). The Patch-Cleanser algorithm requires a prediction model to have a ``two-mask correctness'' property, meaning that the prediction model should correctly classify any image when any two blank masks replace portions of the image. Xiang et al. learn a prediction model to be robust to two-mask operations by augmenting the training set with pairs of masks at random locations of training images and performing empirical risk minimization (ERM) on the augmented dataset.   However, in the non-realizable setting when no predictor is perfectly correct on all two-mask operations on all images, we exhibit an example where ERM fails. To overcome this challenge, we propose a different algorithm that provably learns a predictor robust to all two-mask operations using an ERM oracle, based on prior work by Feige et al. [2015]. We also extend this result to a multiple-group setting, where we can learn a predictor that achieves low robust loss on all groups simultaneously.

摘要: 考虑补丁攻击，在测试时，对手使用补丁操纵测试图像，以诱导有针对性的错误分类。我们考虑了最近对补丁攻击的防御，Patch-Cleanser(向等人)。[2022]))。Patch-Cleanser算法要求预测模型具有“双掩码正确性”属性，这意味着当任何两个空白掩码替换图像的部分时，预测模型应该正确地对任何图像进行分类。香等人。通过在训练图像的随机位置用多对掩码来扩充训练集，并对扩充后的数据集执行经验风险最小化(ERM)，学习预测模型以对双掩码操作具有健壮性。然而，在不可实现的情况下，当没有预测器对所有图像的所有双掩码操作都是完全正确的时，我们展示了ERM失败的例子。为了克服这一挑战，我们提出了一种不同的算法，该算法基于Feige等人先前的工作，使用ERM预言学习了对所有双掩码操作具有健壮性的预测器。[2015][2015]。我们还将这一结果推广到多组设置，在这种情况下，我们可以学习一个同时在所有组上实现低稳健损失的预测器。



## **12. Learning When to Use Adaptive Adversarial Image Perturbations against Autonomous Vehicles**

学习何时对自主车辆使用自适应对抗性图像扰动 cs.RO

**SubmitDate**: 2023-03-15    [abs](http://arxiv.org/abs/2212.13667v2) [paper-pdf](http://arxiv.org/pdf/2212.13667v2)

**Authors**: Hyung-Jin Yoon, Hamidreza Jafarnejadsani, Petros Voulgaris

**Abstract**: The deep neural network (DNN) models for object detection using camera images are widely adopted in autonomous vehicles. However, DNN models are shown to be susceptible to adversarial image perturbations. In the existing methods of generating the adversarial image perturbations, optimizations take each incoming image frame as the decision variable to generate an image perturbation. Therefore, given a new image, the typically computationally-expensive optimization needs to start over as there is no learning between the independent optimizations. Very few approaches have been developed for attacking online image streams while considering the underlying physical dynamics of autonomous vehicles, their mission, and the environment. We propose a multi-level stochastic optimization framework that monitors an attacker's capability of generating the adversarial perturbations. Based on this capability level, a binary decision attack/not attack is introduced to enhance the effectiveness of the attacker. We evaluate our proposed multi-level image attack framework using simulations for vision-guided autonomous vehicles and actual tests with a small indoor drone in an office environment. The results show our method's capability to generate the image attack in real-time while monitoring when the attacker is proficient given state estimates.

摘要: 利用摄像机图像进行目标检测的深度神经网络(DNN)模型在自动驾驶车辆中得到了广泛的应用。然而，DNN模型被证明容易受到对抗性图像扰动的影响。在现有的产生对抗性图像扰动的方法中，优化将每一进入的图像帧作为决策变量来产生图像扰动。因此，给定一个新的图像，通常计算代价高昂的优化需要重新开始，因为在独立的优化之间没有学习。在考虑自动驾驶车辆的基本物理动力学、它们的任务和环境的同时，很少有人开发出攻击在线图像流的方法。我们提出了一个多层次随机优化框架，用于监控攻击者产生敌意扰动的能力。基于这一能力水平，引入了二元判决攻击/非攻击，以增强攻击者的有效性。我们通过对视觉制导自动驾驶车辆的模拟和办公室环境中小型室内无人机的实际测试来评估我们提出的多级图像攻击框架。结果表明，我们的方法能够实时生成图像攻击，同时监控攻击者何时熟练掌握给定的状态估计。



## **13. Visual Prompting for Adversarial Robustness**

对抗健壮性的视觉提示 cs.CV

ICASSP 2023

**SubmitDate**: 2023-03-15    [abs](http://arxiv.org/abs/2210.06284v2) [paper-pdf](http://arxiv.org/pdf/2210.06284v2)

**Authors**: Aochuan Chen, Peter Lorenz, Yuguang Yao, Pin-Yu Chen, Sijia Liu

**Abstract**: In this work, we leverage visual prompting (VP) to improve adversarial robustness of a fixed, pre-trained model at testing time. Compared to conventional adversarial defenses, VP allows us to design universal (i.e., data-agnostic) input prompting templates, which have plug-and-play capabilities at testing time to achieve desired model performance without introducing much computation overhead. Although VP has been successfully applied to improving model generalization, it remains elusive whether and how it can be used to defend against adversarial attacks. We investigate this problem and show that the vanilla VP approach is not effective in adversarial defense since a universal input prompt lacks the capacity for robust learning against sample-specific adversarial perturbations. To circumvent it, we propose a new VP method, termed Class-wise Adversarial Visual Prompting (C-AVP), to generate class-wise visual prompts so as to not only leverage the strengths of ensemble prompts but also optimize their interrelations to improve model robustness. Our experiments show that C-AVP outperforms the conventional VP method, with 2.1X standard accuracy gain and 2X robust accuracy gain. Compared to classical test-time defenses, C-AVP also yields a 42X inference time speedup.

摘要: 在这项工作中，我们利用视觉提示(VP)来提高测试时固定的、预先训练的模型的对抗健壮性。与传统的对抗性防御相比，VP允许我们设计通用的(即数据不可知的)输入提示模板，该模板在测试时具有即插即用的能力，在不引入太多计算开销的情况下达到期望的模型性能。虽然VP已经被成功地应用于改进模型泛化，但它是否以及如何被用来防御对手攻击仍然是一个难以捉摸的问题。我们研究了这个问题，并证明了普通VP方法在对抗防御中不是有效的，因为通用的输入提示缺乏针对特定样本的对抗扰动的稳健学习能力。针对这一问题，我们提出了一种新的分类对抗性视觉提示生成方法--分类对抗性视觉提示(C-AVP)，该方法不仅可以利用集成提示的优点，而且可以优化它们之间的相互关系，从而提高模型的稳健性。实验表明，C-AVP比传统的VP方法有2.1倍的标准精度增益和2倍的稳健精度增益。与经典的测试时间防御相比，C-AVP的推理时间加速比也提高了42倍。



## **14. EvalAttAI: A Holistic Approach to Evaluating Attribution Maps in Robust and Non-Robust Models**

EvalAttAI：一种在稳健和非稳健模型中评估属性图的整体方法 cs.LG

**SubmitDate**: 2023-03-15    [abs](http://arxiv.org/abs/2303.08866v1) [paper-pdf](http://arxiv.org/pdf/2303.08866v1)

**Authors**: Ian E. Nielsen, Ravi P. Ramachandran, Nidhal Bouaynaya, Hassan M. Fathallah-Shaykh, Ghulam Rasool

**Abstract**: The expansion of explainable artificial intelligence as a field of research has generated numerous methods of visualizing and understanding the black box of a machine learning model. Attribution maps are generally used to highlight the parts of the input image that influence the model to make a specific decision. On the other hand, the robustness of machine learning models to natural noise and adversarial attacks is also being actively explored. This paper focuses on evaluating methods of attribution mapping to find whether robust neural networks are more explainable. We explore this problem within the application of classification for medical imaging. Explainability research is at an impasse. There are many methods of attribution mapping, but no current consensus on how to evaluate them and determine the ones that are the best. Our experiments on multiple datasets (natural and medical imaging) and various attribution methods reveal that two popular evaluation metrics, Deletion and Insertion, have inherent limitations and yield contradictory results. We propose a new explainability faithfulness metric (called EvalAttAI) that addresses the limitations of prior metrics. Using our novel evaluation, we found that Bayesian deep neural networks using the Variational Density Propagation technique were consistently more explainable when used with the best performing attribution method, the Vanilla Gradient. However, in general, various types of robust neural networks may not be more explainable, despite these models producing more visually plausible attribution maps.

摘要: 可解释人工智能作为一个研究领域的扩展，产生了许多可视化和理解机器学习模型的黑匣子的方法。属性图通常用于突出显示输入图像中影响模型做出特定决策的部分。另一方面，机器学习模型对自然噪声和对抗性攻击的稳健性也在积极探索。本文重点研究了属性映射的评价方法，以确定稳健神经网络是否具有更好的解释性。我们在医学影像分类的应用中探讨了这个问题。可解释性研究陷入僵局。归因映射的方法有很多，但目前还没有关于如何评估它们并确定哪些是最好的方法的共识。我们在多个数据集(自然图像和医学图像)和不同的属性方法上的实验表明，两种流行的评估指标--删除和插入--具有内在的局限性，并产生相互矛盾的结果。我们提出了一种新的可解释性忠实性度量(称为EvalAttAI)，解决了已有度量的局限性。使用我们的新评估，我们发现使用变密度传播技术的贝叶斯深度神经网络在与性能最好的属性方法Vanilla梯度一起使用时始终更具解释性。然而，总的来说，各种类型的健壮神经网络可能不会更好地解释，尽管这些模型产生了更多视觉上可信的属性图。



## **15. Enhancing Diffusion-Based Image Synthesis with Robust Classifier Guidance**

稳健分类器引导增强基于扩散的图像合成 cs.CV

Accepted to TMLR

**SubmitDate**: 2023-03-15    [abs](http://arxiv.org/abs/2208.08664v2) [paper-pdf](http://arxiv.org/pdf/2208.08664v2)

**Authors**: Bahjat Kawar, Roy Ganz, Michael Elad

**Abstract**: Denoising diffusion probabilistic models (DDPMs) are a recent family of generative models that achieve state-of-the-art results. In order to obtain class-conditional generation, it was suggested to guide the diffusion process by gradients from a time-dependent classifier. While the idea is theoretically sound, deep learning-based classifiers are infamously susceptible to gradient-based adversarial attacks. Therefore, while traditional classifiers may achieve good accuracy scores, their gradients are possibly unreliable and might hinder the improvement of the generation results. Recent work discovered that adversarially robust classifiers exhibit gradients that are aligned with human perception, and these could better guide a generative process towards semantically meaningful images. We utilize this observation by defining and training a time-dependent adversarially robust classifier and use it as guidance for a generative diffusion model. In experiments on the highly challenging and diverse ImageNet dataset, our scheme introduces significantly more intelligible intermediate gradients, better alignment with theoretical findings, as well as improved generation results under several evaluation metrics. Furthermore, we conduct an opinion survey whose findings indicate that human raters prefer our method's results.

摘要: 去噪扩散概率模型(DDPM)是最近出现的一类产生式模型，可以得到最先进的结果。为了获得类条件生成，建议使用依赖于时间的分类器的梯度来指导扩散过程。虽然这个想法在理论上是合理的，但基于深度学习的分类器很容易受到基于梯度的对抗性攻击。因此，虽然传统的分类器可以获得很好的精度分数，但它们的梯度可能是不可靠的，并可能阻碍生成结果的改进。最近的研究发现，逆序稳健的分类器表现出与人类感知一致的梯度，这些梯度可以更好地引导生成过程走向有语义意义的图像。我们通过定义和训练一个时间相关的对抗性稳健分类器来利用这一观察结果，并将其用作生成性扩散模型的指导。在高度挑战性和多样性的ImageNet数据集上的实验中，我们的方案引入了明显更易理解的中间梯度，更好地与理论结果保持一致，以及在几种评估指标下改进的生成结果。此外，我们还进行了一项民意调查，调查结果表明，人类评分者更喜欢我们的方法的结果。



## **16. Black-box Adversarial Example Attack towards FCG Based Android Malware Detection under Incomplete Feature Information**

特征信息不完全条件下基于FCG的Android恶意软件检测黑盒恶意示例攻击 cs.SE

**SubmitDate**: 2023-03-15    [abs](http://arxiv.org/abs/2303.08509v1) [paper-pdf](http://arxiv.org/pdf/2303.08509v1)

**Authors**: Heng Li, Zhang Cheng, Bang Wu, Liheng Yuan, Cuiying Gao, Wei Yuan, Xiapu Luo

**Abstract**: The function call graph (FCG) based Android malware detection methods have recently attracted increasing attention due to their promising performance. However, these methods are susceptible to adversarial examples (AEs). In this paper, we design a novel black-box AE attack towards the FCG based malware detection system, called BagAmmo. To mislead its target system, BagAmmo purposefully perturbs the FCG feature of malware through inserting "never-executed" function calls into malware code. The main challenges are two-fold. First, the malware functionality should not be changed by adversarial perturbation. Second, the information of the target system (e.g., the graph feature granularity and the output probabilities) is absent.   To preserve malware functionality, BagAmmo employs the try-catch trap to insert function calls to perturb the FCG of malware. Without the knowledge about feature granularity and output probabilities, BagAmmo adopts the architecture of generative adversarial network (GAN), and leverages a multi-population co-evolution algorithm (i.e., Apoem) to generate the desired perturbation. Every population in Apoem represents a possible feature granularity, and the real feature granularity can be achieved when Apoem converges.   Through extensive experiments on over 44k Android apps and 32 target models, we evaluate the effectiveness, efficiency and resilience of BagAmmo. BagAmmo achieves an average attack success rate of over 99.9% on MaMaDroid, APIGraph and GCN, and still performs well in the scenario of concept drift and data imbalance. Moreover, BagAmmo outperforms the state-of-the-art attack SRL in attack success rate.

摘要: 基于函数调用图(FCG)的Android恶意软件检测方法因其良好的性能而受到越来越多的关注。然而，这些方法容易受到对抗性例子(AE)的影响。本文针对基于FCG的恶意软件检测系统设计了一种新的黑盒AE攻击，称为BagAmmo。为了误导其目标系统，BagAmmo故意通过在恶意软件代码中插入“从未执行”的函数调用来扰乱恶意软件的FCG功能。主要的挑战是双重的。首先，恶意软件的功能不应因敌意干扰而改变。第二，缺少目标系统的信息(如图特征粒度和输出概率)。为了保留恶意软件的功能，BagAmmo使用Try-Catch陷阱插入函数调用来扰乱恶意软件的FCG。在不知道特征粒度和输出概率的情况下，BagAmmo采用了产生式对抗网络(GAN)的体系结构，并利用多种群协同进化算法(即A POEM)来产生期望的扰动。每个种群代表一个可能的特征粒度，而真正的特征粒度可以在Apome收敛时获得。通过在超过44K个Android应用程序和32个目标机型上的广泛实验，我们评估了BagAmmo的有效性、效率和弹性。BagAmmo在MaMaDroid、APIGgraph和GCN上的平均攻击成功率超过99.9%，在概念漂移和数据不平衡的场景下仍表现良好。此外，BagAmmo在攻击成功率方面优于最先进的攻击SRL。



## **17. The Devil's Advocate: Shattering the Illusion of Unexploitable Data using Diffusion Models**

魔鬼代言人：使用扩散模型粉碎不可利用数据的幻觉 cs.LG

**SubmitDate**: 2023-03-15    [abs](http://arxiv.org/abs/2303.08500v1) [paper-pdf](http://arxiv.org/pdf/2303.08500v1)

**Authors**: Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie

**Abstract**: Protecting personal data against the exploitation of machine learning models is of paramount importance. Recently, availability attacks have shown great promise to provide an extra layer of protection against the unauthorized use of data to train neural networks. These methods aim to add imperceptible noise to clean data so that the neural networks cannot extract meaningful patterns from the protected data, claiming that they can make personal data "unexploitable." In this paper, we provide a strong countermeasure against such approaches, showing that unexploitable data might only be an illusion. In particular, we leverage the power of diffusion models and show that a carefully designed denoising process can defuse the ramifications of the data-protecting perturbations. We rigorously analyze our algorithm, and theoretically prove that the amount of required denoising is directly related to the magnitude of the data-protecting perturbations. Our approach, called AVATAR, delivers state-of-the-art performance against a suite of recent availability attacks in various scenarios, outperforming adversarial training. Our findings call for more research into making personal data unexploitable, showing that this goal is far from over.

摘要: 保护个人数据免受机器学习模型的利用是至关重要的。最近，可用性攻击显示出巨大的希望，可以提供额外的一层保护，防止未经授权使用数据来训练神经网络。这些方法的目的是在干净的数据中添加难以察觉的噪声，以便神经网络无法从受保护的数据中提取有意义的模式，声称它们可以使个人数据“无法利用”。在这篇文章中，我们针对这种方法提供了一个强有力的对策，表明不可利用的数据可能只是一种错觉。特别是，我们利用扩散模型的力量，并表明精心设计的去噪过程可以化解数据保护扰动的影响。我们对算法进行了严格的分析，并从理论上证明了所需的去噪量与数据保护扰动的大小直接相关。我们的方法被称为阿凡达，在各种情况下提供最先进的性能来对抗最近的一系列可用性攻击，表现优于对手训练。我们的发现呼吁进行更多的研究，让个人数据无法被利用，这表明这一目标远未结束。



## **18. Similarity of Neural Architectures Based on Input Gradient Transferability**

基于输入梯度传递的神经结构相似性研究 cs.LG

21pages, 10 figures, 1.5MB

**SubmitDate**: 2023-03-15    [abs](http://arxiv.org/abs/2210.11407v2) [paper-pdf](http://arxiv.org/pdf/2210.11407v2)

**Authors**: Jaehui Hwang, Dongyoon Han, Byeongho Heo, Song Park, Sanghyuk Chun, Jong-Seok Lee

**Abstract**: In recent years, a huge amount of deep neural architectures have been developed for image classification. It remains curious whether these models are similar or different and what factors contribute to their similarities or differences. To address this question, we aim to design a quantitative and scalable similarity function between neural architectures. We utilize adversarial attack transferability, which has information related to input gradients and decision boundaries that are widely used to understand model behaviors. We conduct a large-scale analysis on 69 state-of-the-art ImageNet classifiers using our proposed similarity function to answer the question. Moreover, we observe neural architecture-related phenomena using model similarity that model diversity can lead to better performance on model ensembles and knowledge distillation under specific conditions. Our results provide insights into why the development of diverse neural architectures with distinct components is necessary.

摘要: 近年来，已有大量的深层神经结构被用于图像分类。这些模式是相似还是不同，以及是什么因素导致了它们的相似或不同，这一点仍然令人好奇。为了解决这个问题，我们的目标是设计一种量化的、可伸缩的神经结构之间的相似性函数。我们利用对抗性攻击的可移动性，它具有与输入梯度和决策边界相关的信息，这些信息被广泛用于理解模型行为。我们使用我们提出的相似度函数对69个最先进的ImageNet分类器进行了大规模的分析。此外，我们使用模型相似性来观察与神经结构相关的现象，即在特定条件下，模型多样性可以在模型集成和知识提取方面带来更好的性能。我们的结果为为什么开发具有不同组件的不同神经架构提供了洞察力。



## **19. Quantum adversarial metric learning model based on triplet loss function**

基于三重损失函数的量子对抗度量学习模型 quant-ph

arXiv admin note: substantial text overlap with arXiv:2303.07906

**SubmitDate**: 2023-03-15    [abs](http://arxiv.org/abs/2303.08293v1) [paper-pdf](http://arxiv.org/pdf/2303.08293v1)

**Authors**: Yan-Yan Hou, Jian Li, Xiu-Bo Chen, Chong-Qiang Ye

**Abstract**: Metric learning plays an essential role in image analysis and classification, and it has attracted more and more attention. In this paper, we propose a quantum adversarial metric learning (QAML) model based on the triplet loss function, where samples are embedded into the high-dimensional Hilbert space and the optimal metric is obtained by minimizing the triplet loss function. The QAML model employs entanglement and interference to build superposition states for triplet samples so that only one parameterized quantum circuit is needed to calculate sample distances, which reduces the demand for quantum resources. Considering the QAML model is fragile to adversarial attacks, an adversarial sample generation strategy is designed based on the quantum gradient ascent method, effectively improving the robustness against the functional adversarial attack. Simulation results show that the QAML model can effectively distinguish samples of MNIST and Iris datasets and has higher robustness accuracy over the general quantum metric learning. The QAML model is a fundamental research problem of machine learning. As a subroutine of classification and clustering tasks, the QAML model opens an avenue for exploring quantum advantages in machine learning.

摘要: 度量学习在图像分析和分类中起着至关重要的作用，受到越来越多的关注。提出了一种基于三重态损失函数的量子对抗度量学习(QAML)模型，该模型将样本嵌入高维Hilbert空间，通过最小化三重态损失函数得到最优度量。QAML模型利用纠缠和干涉来建立三重态样本的叠加态，因此只需要一个参数化的量子电路来计算样本距离，从而减少了对量子资源的需求。针对QAML模型对敌意攻击的脆弱性，设计了一种基于量子梯度上升方法的对抗性样本生成策略，有效地提高了对功能性对抗性攻击的鲁棒性。仿真结果表明，QAML模型能够有效区分MNIST和IRIS数据集的样本，并且比一般的量子度量学习具有更高的鲁棒性精度。QAML模型是机器学习的一个基础研究问题。作为分类和聚类任务的子例程，QAML模型为探索机器学习中的量子优势开辟了一条途径。



## **20. Can Adversarial Examples Be Parsed to Reveal Victim Model Information?**

对抗性例子可以被解析以揭示受害者模型信息吗？ cs.CV

**SubmitDate**: 2023-03-15    [abs](http://arxiv.org/abs/2303.07474v2) [paper-pdf](http://arxiv.org/pdf/2303.07474v2)

**Authors**: Yuguang Yao, Jiancheng Liu, Yifan Gong, Xiaoming Liu, Yanzhi Wang, Xue Lin, Sijia Liu

**Abstract**: Numerous adversarial attack methods have been developed to generate imperceptible image perturbations that can cause erroneous predictions of state-of-the-art machine learning (ML) models, in particular, deep neural networks (DNNs). Despite intense research on adversarial attacks, little effort was made to uncover 'arcana' carried in adversarial attacks. In this work, we ask whether it is possible to infer data-agnostic victim model (VM) information (i.e., characteristics of the ML model or DNN used to generate adversarial attacks) from data-specific adversarial instances. We call this 'model parsing of adversarial attacks' - a task to uncover 'arcana' in terms of the concealed VM information in attacks. We approach model parsing via supervised learning, which correctly assigns classes of VM's model attributes (in terms of architecture type, kernel size, activation function, and weight sparsity) to an attack instance generated from this VM. We collect a dataset of adversarial attacks across 7 attack types generated from 135 victim models (configured by 5 architecture types, 3 kernel size setups, 3 activation function types, and 3 weight sparsity ratios). We show that a simple, supervised model parsing network (MPN) is able to infer VM attributes from unseen adversarial attacks if their attack settings are consistent with the training setting (i.e., in-distribution generalization assessment). We also provide extensive experiments to justify the feasibility of VM parsing from adversarial attacks, and the influence of training and evaluation factors in the parsing performance (e.g., generalization challenge raised in out-of-distribution evaluation). We further demonstrate how the proposed MPN can be used to uncover the source VM attributes from transfer attacks, and shed light on a potential connection between model parsing and attack transferability.

摘要: 已经开发了许多对抗性攻击方法来产生不可察觉的图像扰动，这可能导致对最先进的机器学习(ML)模型的错误预测，特别是深度神经网络(DNN)。尽管对对抗性攻击进行了密集的研究，但几乎没有努力去发现对抗性攻击中携带的“奥秘”。在这项工作中，我们问是否有可能从特定于数据的对抗性实例中推断出与数据无关的受害者模型(VM)信息(即，用于生成对抗性攻击的ML模型或DNN的特征)。我们称之为“对抗性攻击的模型解析”--根据攻击中隐藏的VM信息来发现“奥秘”的任务。我们通过有监督的学习来实现模型解析，它正确地将VM的模型属性的类别(根据体系结构类型、核大小、激活函数和权重稀疏性)分配给从该VM生成的攻击实例。我们收集了从135个受害者模型(由5个体系结构类型、3个核大小设置、3个激活函数类型和3个权重稀疏率配置)生成的7种攻击类型的对抗性攻击的数据集。我们证明了一个简单的有监督的模型解析网络(MPN)能够从未知的敌意攻击中推断出VM属性，如果它们的攻击设置与训练设置一致(即分布内泛化评估)。我们还提供了大量的实验，以验证在敌意攻击下进行VM解析的可行性，以及训练和评估因素(例如，在分布外评估中提出的泛化挑战)对解析性能的影响。我们进一步演示了如何使用所提出的MPN来发现来自传输攻击的源VM属性，并阐明了模型解析和攻击可转移性之间的潜在联系。



## **21. Improving Adversarial Robustness with Hypersphere Embedding and Angular-based Regularizations**

利用超球面嵌入和基于角度的正则化提高对手稳健性 cs.LG

**SubmitDate**: 2023-03-15    [abs](http://arxiv.org/abs/2303.08289v1) [paper-pdf](http://arxiv.org/pdf/2303.08289v1)

**Authors**: Olukorede Fakorede, Ashutosh Nirala, Modeste Atsague, Jin Tian

**Abstract**: Adversarial training (AT) methods have been found to be effective against adversarial attacks on deep neural networks. Many variants of AT have been proposed to improve its performance. Pang et al. [1] have recently shown that incorporating hypersphere embedding (HE) into the existing AT procedures enhances robustness. We observe that the existing AT procedures are not designed for the HE framework, and thus fail to adequately learn the angular discriminative information available in the HE framework. In this paper, we propose integrating HE into AT with regularization terms that exploit the rich angular information available in the HE framework. Specifically, our method, termed angular-AT, adds regularization terms to AT that explicitly enforce weight-feature compactness and inter-class separation; all expressed in terms of angular features. Experimental results show that angular-AT further improves adversarial robustness.

摘要: 对抗性训练(AT)方法已被发现对深层神经网络上的对抗性攻击是有效的。已经提出了许多AT的变体来提高其性能。Pang et al.[1]最近发现将超球面嵌入(HE)引入到现有的AT过程中可以增强健壮性。我们观察到，现有的AT过程并不是为HE框架设计的，因此无法充分学习HE框架中可用的角度区分信息。在本文中，我们提出了利用HE框架中丰富的角度信息，利用正则化项将HE整合到AT中。具体地说，我们的方法被称为角度-AT，它向AT添加了正则化项，明确地强制执行权重-特征紧致性和类间分离；所有这些都以角度特征来表示。实验结果表明，ANGING-AT算法进一步提高了攻击的健壮性。



## **22. Resilient Dynamic Average Consensus based on Trusted agents**

基于可信代理的弹性动态平均一致性 eess.SY

Initial Draft

**SubmitDate**: 2023-03-14    [abs](http://arxiv.org/abs/2303.08171v1) [paper-pdf](http://arxiv.org/pdf/2303.08171v1)

**Authors**: Shamik Bhattacharyya, Rachel Kalpana Kalaimani

**Abstract**: In this paper, we address the discrete-time dynamic average consensus (DAC) of a multi-agent system in the presence of adversarial attacks. The adversarial attack is considered to be of Byzantine type, which compromises the computation capabilities of the agent and sends arbitrary false data to its neighbours. We assume a few of the agents cannot be compromised by adversaries, which we term trusted agents. We first formally define resilient DAC in the presence of Byzantine adversaries. Then we propose our novel Resilient Dynamic Average Consensus (ResDAC) algorithm that ensures the trusted and ordinary agents achieve resilient DAC in the presence of adversarial agents. The only requirements are that of the trusted agents forming a connected dominating set and the first-order differences of the reference signals being bounded. We do not impose any restriction on the tolerable number of adversarial agents that can be present in the network. We also do not restrict the reference signals to be bounded. Finally, we provide numerical simulations to illustrate the effectiveness of the proposed ResDAC algorithm.

摘要: 本文研究了存在对抗性攻击的多智能体系统的离散时间动态平均共识问题。敌意攻击被认为是拜占庭式的，它损害了代理的计算能力，并向其邻居发送任意虚假数据。我们假设少数代理不会被对手攻破，我们称之为可信代理。我们首先在拜占庭式的对手面前正式定义了有弹性的DAC。在此基础上，提出了一种新的弹性动态平均共识(ResDAC)算法，该算法保证了可信代理和普通代理在对抗代理存在的情况下实现弹性DAC。唯一的要求是形成连通支配集的可信代理并且参考信号的一阶差分是有界的。我们对网络中可以存在的对抗性代理的可容忍数量没有任何限制。我们也不限制参考信号是有界的。最后，通过数值仿真验证了所提出的ResDAC算法的有效性。



## **23. BODEGA: Benchmark for Adversarial Example Generation in Credibility Assessment**

BODEGA：可信度评估中的对抗性范例生成基准 cs.CL

**SubmitDate**: 2023-03-14    [abs](http://arxiv.org/abs/2303.08032v1) [paper-pdf](http://arxiv.org/pdf/2303.08032v1)

**Authors**: Piotr Przybyła, Alexander Shvets, Horacio Saggion

**Abstract**: Text classification methods have been widely investigated as a way to detect content of low credibility: fake news, social media bots, propaganda, etc. Quite accurate models (likely based on deep neural networks) help in moderating public electronic platforms and often cause content creators to face rejection of their submissions or removal of already published texts. Having the incentive to evade further detection, content creators try to come up with a slightly modified version of the text (known as an attack with an adversarial example) that exploit the weaknesses of classifiers and result in a different output. Here we introduce BODEGA: a benchmark for testing both victim models and attack methods on four misinformation detection tasks in an evaluation framework designed to simulate real use-cases of content moderation. We also systematically test the robustness of popular text classifiers against available attacking techniques and discover that, indeed, in some cases barely significant changes in input text can mislead the models. We openly share the BODEGA code and data in hope of enhancing the comparability and replicability of further research in this area.

摘要: 文本分类方法被广泛研究为检测可信度较低的内容的一种方式：假新闻、社交媒体机器人、宣传等。相当准确的模型(可能基于深度神经网络)有助于调节公共电子平台，并经常导致内容创建者面临提交的拒绝或已发布的文本的删除。出于逃避进一步检测的动机，内容创建者试图对文本进行稍微修改的版本(称为带有敌意的示例的攻击)，以利用分类器的弱点并产生不同的输出。在这里，我们介绍了Bodes a：一个基准，用于在四个错误信息检测任务中测试受害者模型和攻击方法，该评估框架旨在模拟真实的内容审核用例。我们还系统地测试了流行的文本分类器对现有攻击技术的健壮性，并发现在某些情况下，输入文本几乎没有显著变化会误导模型。我们公开分享这些代码和数据，希望增强这一领域进一步研究的可比性和可复制性。



## **24. Dynamic Efficient Adversarial Training Guided by Gradient Magnitude**

梯度幅度引导下的动态高效对抗训练 cs.LG

18 pages, 6 figures

**SubmitDate**: 2023-03-14    [abs](http://arxiv.org/abs/2103.03076v2) [paper-pdf](http://arxiv.org/pdf/2103.03076v2)

**Authors**: Fu Wang, Yanghao Zhang, Yanbin Zheng, Wenjie Ruan

**Abstract**: Adversarial training is an effective but time-consuming way to train robust deep neural networks that can withstand strong adversarial attacks. As a response to its inefficiency, we propose Dynamic Efficient Adversarial Training (DEAT), which gradually increases the adversarial iteration during training. We demonstrate that the gradient's magnitude correlates with the curvature of the trained model's loss landscape, allowing it to reflect the effect of adversarial training. Therefore, based on the magnitude of the gradient, we propose a general acceleration strategy, M+ acceleration, which enables an automatic and highly effective method of adjusting the training procedure. M+ acceleration is computationally efficient and easy to implement. It is suited for DEAT and compatible with the majority of existing adversarial training techniques. Extensive experiments have been done on CIFAR-10 and ImageNet datasets with various training environments. The results show that the proposed M+ acceleration significantly improves the training efficiency of existing adversarial training methods while achieving similar robustness performance. This demonstrates that the strategy is highly adaptive and offers a valuable solution for automatic adversarial training.

摘要: 对抗性训练是一种有效但耗时的方法，可以训练健壮的深层神经网络，使其能够抵抗强大的对抗性攻击。针对其效率低下的问题，我们提出了动态有效的对抗性训练(DEAT)，在训练过程中逐步增加对抗性迭代。我们证明了梯度的大小与训练模型的损失景观的曲率相关，从而使其能够反映对抗性训练的效果。因此，基于梯度的大小，我们提出了一种通用的加速策略M+Acceleration，它使得一种自动而高效的训练过程调整方法成为可能。M+加速在计算上是高效的，并且易于实现。它适用于DEAT，并与大多数现有的对抗性训练技术兼容。在CIFAR-10和ImageNet数据集上进行了广泛的实验，并在不同的训练环境下进行了实验。实验结果表明，提出的M+加速算法在保持相似健壮性的同时，显著提高了现有对抗性训练方法的训练效率。这表明该策略具有很强的适应性，为自动对抗性训练提供了一种有价值的解决方案。



## **25. Constrained Adversarial Learning and its applicability to Automated Software Testing: a systematic review**

受限对抗性学习及其在自动化软件测试中的适用性 cs.SE

32 pages, 5 tables, 2 figures, Information and Software Technology  journal

**SubmitDate**: 2023-03-14    [abs](http://arxiv.org/abs/2303.07546v1) [paper-pdf](http://arxiv.org/pdf/2303.07546v1)

**Authors**: João Vitorino, Tiago Dias, Tiago Fonseca, Eva Maia, Isabel Praça

**Abstract**: Every novel technology adds hidden vulnerabilities ready to be exploited by a growing number of cyber-attacks. Automated software testing can be a promising solution to quickly analyze thousands of lines of code by generating and slightly modifying function-specific testing data to encounter a multitude of vulnerabilities and attack vectors. This process draws similarities to the constrained adversarial examples generated by adversarial learning methods, so there could be significant benefits to the integration of these methods in automated testing tools. Therefore, this systematic review is focused on the current state-of-the-art of constrained data generation methods applied for adversarial learning and software testing, aiming to guide researchers and developers to enhance testing tools with adversarial learning methods and improve the resilience and robustness of their digital systems. The found constrained data generation applications for adversarial machine learning were systematized, and the advantages and limitations of approaches specific for software testing were thoroughly analyzed, identifying research gaps and opportunities to improve testing tools with adversarial attack methods.

摘要: 每一项新技术都会增加隐藏的漏洞，随时可能被越来越多的网络攻击所利用。自动化软件测试可能是一种很有前途的解决方案，可以通过生成并略微修改特定于函数的测试数据来快速分析数千行代码，以遇到大量漏洞和攻击矢量。这个过程与对抗性学习方法生成的受限对抗性示例有相似之处，因此将这些方法集成到自动化测试工具中可能会有显著的好处。因此，本文系统地综述了应用于对抗性学习和软件测试的受限数据生成方法的现状，旨在指导研究人员和开发人员使用对抗性学习方法来增强测试工具，提高他们的数字系统的弹性和健壮性。对发现的用于对抗性机器学习的受限数据生成应用进行了系统化，并深入分析了针对软件测试的方法的优势和局限性，找出了研究差距和机会，以改进对抗性攻击方法的测试工具。



## **26. Symmetry Defense Against CNN Adversarial Perturbation Attacks**

CNN对抗扰动攻击的对称性防御 cs.LG

13 pages

**SubmitDate**: 2023-03-13    [abs](http://arxiv.org/abs/2210.04087v2) [paper-pdf](http://arxiv.org/pdf/2210.04087v2)

**Authors**: Blerta Lindqvist

**Abstract**: Convolutional neural network classifiers (CNNs) are susceptible to adversarial attacks that perturb original samples to fool classifiers such as an autonomous vehicle's road sign image classifier. CNNs also lack invariance in the classification of symmetric samples because CNNs can classify symmetric samples differently. Considered together, the CNN lack of adversarial robustness and the CNN lack of invariance mean that the classification of symmetric adversarial samples can differ from their incorrect classification. Could symmetric adversarial samples revert to their correct classification? This paper answers this question by designing a symmetry defense that inverts or horizontally flips adversarial samples before classification against adversaries unaware of the defense. Against adversaries aware of the defense, the defense devises a Klein four symmetry subgroup that includes the horizontal flip and pixel inversion symmetries. The symmetry defense uses the subgroup symmetries in accuracy evaluation and the subgroup closure property to confine the transformations that an adaptive adversary can apply before or after generating the adversarial sample. Without changing the preprocessing, parameters, or model, the proposed symmetry defense counters the Projected Gradient Descent (PGD) and AutoAttack attacks with near-default accuracies for ImageNet. Without using attack knowledge or adversarial samples, the proposed defense exceeds the current best defense, which trains on adversarial samples. The defense maintains and even improves the classification accuracy of non-adversarial samples.

摘要: 卷积神经网络分类器(CNN)容易受到敌意攻击，这些攻击会将原始样本干扰到愚弄分类器，如自动驾驶汽车的道路标志图像分类器。CNN在对称样本的分类上也缺乏不变性，因为CNN可以对对称样本进行不同的分类。综合考虑，CNN缺乏对抗稳健性和CNN缺乏不变性，这意味着对称对抗样本的分类可能不同于它们的错误分类。对称的对抗性样本能恢复到正确的分类吗？本文通过设计一种对称防御来回答这个问题，该对称防御在分类之前颠倒或水平翻转对手样本，以对抗不知道该防御的对手。针对意识到防御的对手，防御方设计了一个克莱因四对称子群，其中包括水平翻转和像素反转对称。对称性防御利用子群对称性进行精度评估，并利用子群封闭性来限制自适应对手在生成对抗样本之前或之后可以应用的变换。在不更改预处理、参数或模型的情况下，所提出的对称防御以近乎默认的ImageNet精度对抗投影梯度下降(PGD)和AutoAttack攻击。在不使用攻击知识或对抗性样本的情况下，建议的防御超过了目前最好的防御，后者基于对抗性样本进行训练。答辩保持甚至提高了非对抗性样本的分类准确率。



## **27. Review on the Feasibility of Adversarial Evasion Attacks and Defenses for Network Intrusion Detection Systems**

网络入侵检测系统对抗性规避攻击与防御的可行性研究综述 cs.CR

Under review (Submitted to Computer Networks - Elsevier)

**SubmitDate**: 2023-03-13    [abs](http://arxiv.org/abs/2303.07003v1) [paper-pdf](http://arxiv.org/pdf/2303.07003v1)

**Authors**: Islam Debicha, Benjamin Cochez, Tayeb Kenaza, Thibault Debatty, Jean-Michel Dricot, Wim Mees

**Abstract**: Nowadays, numerous applications incorporate machine learning (ML) algorithms due to their prominent achievements. However, many studies in the field of computer vision have shown that ML can be fooled by intentionally crafted instances, called adversarial examples. These adversarial examples take advantage of the intrinsic vulnerability of ML models. Recent research raises many concerns in the cybersecurity field. An increasing number of researchers are studying the feasibility of such attacks on security systems based on ML algorithms, such as Intrusion Detection Systems (IDS). The feasibility of such adversarial attacks would be influenced by various domain-specific constraints. This can potentially increase the difficulty of crafting adversarial examples. Despite the considerable amount of research that has been done in this area, much of it focuses on showing that it is possible to fool a model using features extracted from the raw data but does not address the practical side, i.e., the reverse transformation from theory to practice. For this reason, we propose a review browsing through various important papers to provide a comprehensive analysis. Our analysis highlights some challenges that have not been addressed in the reviewed papers.

摘要: 如今，由于机器学习(ML)算法的显著成就，许多应用程序都融入了它们。然而，计算机视觉领域的许多研究表明，ML可以被故意制作的实例所愚弄，这些实例被称为对抗性实例。这些敌对的例子利用了ML模型的内在脆弱性。最近的研究在网络安全领域引发了许多担忧。越来越多的研究人员正在研究对基于ML算法的安全系统(如入侵检测系统)进行此类攻击的可行性。这种对抗性攻击的可行性将受到各种特定领域限制的影响。这可能会增加制作对抗性例子的难度。尽管在这一领域已经做了相当多的研究，但大部分研究的重点是表明，使用从原始数据提取的特征来愚弄模型是可能的，但没有解决实际方面，即从理论到实践的反向转换。为此，我们建议通过浏览各种重要论文进行综述，以提供全面的分析。我们的分析突出了审查文件中没有涉及的一些挑战。



## **28. Towards Making a Trojan-horse Attack on Text-to-Image Retrieval**

对文本到图像检索的木马攻击 cs.MM

Accepted by ICASSP 2023

**SubmitDate**: 2023-03-13    [abs](http://arxiv.org/abs/2202.03861v4) [paper-pdf](http://arxiv.org/pdf/2202.03861v4)

**Authors**: Fan Hu, Aozhu Chen, Xirong Li

**Abstract**: While deep learning based image retrieval is reported to be vulnerable to adversarial attacks, existing works are mainly on image-to-image retrieval with their attacks performed at the front end via query modification. By contrast, we present in this paper the first study about a threat that occurs at the back end of a text-to-image retrieval (T2IR) system. Our study is motivated by the fact that the image collection indexed by the system will be regularly updated due to the arrival of new images from various sources such as web crawlers and advertisers. With malicious images indexed, it is possible for an attacker to indirectly interfere with the retrieval process, letting users see certain images that are completely irrelevant w.r.t. their queries. We put this thought into practice by proposing a novel Trojan-horse attack (THA). In particular, we construct a set of Trojan-horse images by first embedding word-specific adversarial information into a QR code and then putting the code on benign advertising images. A proof-of-concept evaluation, conducted on two popular T2IR datasets (Flickr30k and MS-COCO), shows the effectiveness of the proposed THA in a white-box mode.

摘要: 虽然基于深度学习的图像检索被报道容易受到敌意攻击，但现有的工作主要是针对图像到图像的检索，他们的攻击是通过修改查询在前端进行的。相比之下，我们在本文中提出了第一个关于发生在文本到图像检索(T2IR)系统后端的威胁的研究。我们研究的动机是，由于来自网络爬虫和广告商等各种来源的新图像的到来，系统索引的图像集合将定期更新。对恶意图像进行索引后，攻击者可能会间接干扰检索过程，让用户看到完全不相关的某些图像。他们的疑问。我们将这一思想付诸实践，提出了一种新颖的特洛伊木马攻击(THA)。具体地说，我们首先在二维码中嵌入特定于单词的敌意信息，然后将该代码放在良性广告图像上，从而构建了一组特洛伊木马图像。在两个流行的T2IR数据集(Flickr30k和MS-COCO)上进行的概念验证评估表明，所提出的THA在白盒模式下是有效的。



## **29. Robust Contrastive Language-Image Pretraining against Adversarial Attacks**

抵抗对抗性攻击的健壮对比语言-图像预训练 cs.CV

**SubmitDate**: 2023-03-13    [abs](http://arxiv.org/abs/2303.06854v1) [paper-pdf](http://arxiv.org/pdf/2303.06854v1)

**Authors**: Wenhan Yang, Baharan Mirzasoleiman

**Abstract**: Contrastive vision-language representation learning has achieved state-of-the-art performance for zero-shot classification, by learning from millions of image-caption pairs crawled from the internet. However, the massive data that powers large multimodal models such as CLIP, makes them extremely vulnerable to various types of adversarial attacks, including targeted and backdoor data poisoning attacks. Despite this vulnerability, robust contrastive vision-language pretraining against adversarial attacks has remained unaddressed. In this work, we propose RoCLIP, the first effective method for robust pretraining {and fine-tuning} multimodal vision-language models. RoCLIP effectively breaks the association between poisoned image-caption pairs by considering a pool of random examples, and (1) matching every image with the text that is most similar to its caption in the pool, and (2) matching every caption with the image that is most similar to its image in the pool. Our extensive experiments show that our method renders state-of-the-art targeted data poisoning and backdoor attacks ineffective during pre-training or fine-tuning of CLIP. In particular, RoCLIP decreases the poison and backdoor attack success rates down to 0\% during pre-training and 1\%-4\% during fine-tuning, and effectively improves the model's performance.

摘要: 对比视觉-语言表征学习通过从互联网上爬行的数百万个图像-字幕对进行学习，实现了最先进的零镜头分类性能。然而，为CLIP等大型多模式模型提供动力的海量数据使它们极易受到各种类型的对抗性攻击，包括定向和后门数据中毒攻击。尽管存在这一弱点，但针对对抗性攻击的强有力的对比视觉语言预训仍然没有得到解决。在这项工作中，我们提出了RoCLIP，这是第一个稳健的预训练和微调多通道视觉语言模型的有效方法。RoCLIP通过考虑一组随机示例，以及(1)将每个图像与池中与其字幕最相似的文本匹配，以及(2)将每个标题与池中与其图像最相似的图像匹配，有效地打破了有毒图像-字幕对之间的关联。我们的大量实验表明，我们的方法使得最先进的有针对性的数据中毒和后门攻击在预训练或微调CLIP期间无效。特别是，RoCLIP将毒害和后门攻击的成功率在预训练时降至0，在微调时降至1 4，有效地提高了模型的性能。



## **30. Adversarial Attacks to Direct Data-driven Control for Destabilization**

针对不稳定的直接数据驱动控制的对抗性攻击 eess.SY

6 pages

**SubmitDate**: 2023-03-13    [abs](http://arxiv.org/abs/2303.06837v1) [paper-pdf](http://arxiv.org/pdf/2303.06837v1)

**Authors**: Hampei Sasahara

**Abstract**: This study investigates the vulnerability of direct data-driven control to adversarial attacks in the form of a small but sophisticated perturbation added to the original data. The directed gradient sign method (DGSM) is developed as a specific attack method, based on the fast gradient sign method (FGSM), which has originally been considered in image classification. DGSM uses the gradient of the eigenvalues of the resulting closed-loop system and crafts a perturbation in the direction where the system becomes less stable. It is demonstrated that the system can be destabilized by the attack, even if the original closed-loop system with the clean data has a large margin of stability. To increase the robustness against the attack, regularization methods that have been developed to deal with random disturbances are considered. Their effectiveness is evaluated by numerical experiments using an inverted pendulum model.

摘要: 这项研究调查了直接数据驱动控制在对抗性攻击中的脆弱性，这种攻击是以添加到原始数据上的微小但复杂的扰动的形式进行的。有向梯度符号方法(DGSM)是在最初被认为用于图像分类的快速梯度符号方法(FGSM)的基础上发展起来的一种特定攻击方法。DGSM使用最终闭环系统的特征值的梯度，并在系统变得不稳定的方向上制造扰动。结果表明，即使原始闭环系统具有较大的稳定裕度，攻击也会破坏系统的稳定性。为了增加对攻击的鲁棒性，考虑了已开发的处理随机干扰的正则化方法。利用倒立摆模型进行了数值实验，对其有效性进行了评价。



## **31. Protecting Quantum Procrastinators with Signature Lifting: A Case Study in Cryptocurrencies**

用签名提升保护量子拖延者：加密货币的案例研究 cs.CR

**SubmitDate**: 2023-03-12    [abs](http://arxiv.org/abs/2303.06754v1) [paper-pdf](http://arxiv.org/pdf/2303.06754v1)

**Authors**: Or Sattath, Shai Wyborski

**Abstract**: Current solutions to quantum vulnerabilities of widely used cryptographic schemes involve migrating users to post-quantum schemes before quantum attacks become feasible. This work deals with protecting quantum procrastinators: users that failed to migrate to post-quantum cryptography in time.   To address this problem in the context of digital signatures, we introduce a technique called signature lifting, that allows us to lift a deployed pre-quantum signature scheme satisfying a certain property to a post-quantum signature scheme that uses the same keys. Informally, the said property is that a post-quantum one-way function is used "somewhere along the way" to derive the public-key from the secret-key. Our constructions of signature lifting relies heavily on the post-quantum digital signature scheme Picnic (Chase et al., CCS'17).   Our main case-study is cryptocurrencies, where this property holds in two scenarios: when the public-key is generated via a key-derivation function or when the public-key hash is posted instead of the public-key itself. We propose a modification, based on signature lifting, that can be applied in many cryptocurrencies for securely spending pre-quantum coins in presence of quantum adversaries. Our construction improves upon existing constructions in two major ways: it is not limited to pre-quantum coins whose ECDSA public-key has been kept secret (and in particular, it handles all coins that are stored in addresses generated by HD wallets), and it does not require access to post-quantum coins or using side payments to pay for posting the transaction.

摘要: 目前针对广泛使用的密码方案的量子漏洞的解决方案包括在量子攻击变得可行之前将用户迁移到后量子方案。这项工作涉及保护量子拖延者：未能及时迁移到后量子密码学的用户。为了在数字签名的背景下解决这个问题，我们引入了一种称为签名提升的技术，该技术允许我们将满足一定性质的部署的前量子签名方案提升到使用相同密钥的后量子签名方案。非正式地，所述性质是使用后量子单向函数来从秘密密钥导出公钥。我们的签名提升的构造在很大程度上依赖于后量子数字签名方案Picnic(Chase等人，CCS‘17)。我们的主要案例研究是加密货币，其中该属性在两种情况下成立：当公钥是通过密钥派生函数生成时，或者当公钥散列被发布而不是公钥本身时。我们提出了一种基于签名提升的改进方案，该方案可以应用于多种加密货币，以便在存在量子对手的情况下安全地消费前量子币。我们的结构在两个主要方面对现有结构进行了改进：它不仅限于其ECDSA公钥被保密的前量子硬币(尤其是，它处理存储在HD钱包生成的地址中的所有硬币)，并且它不需要访问后量子硬币或使用附带支付来支付发布交易的费用。



## **32. DNN-Alias: Deep Neural Network Protection Against Side-Channel Attacks via Layer Balancing**

DNN-Alias：基于层平衡的深层神经网络抗旁路攻击 cs.CR

10 pages

**SubmitDate**: 2023-03-12    [abs](http://arxiv.org/abs/2303.06746v1) [paper-pdf](http://arxiv.org/pdf/2303.06746v1)

**Authors**: Mahya Morid Ahmadi, Lilas Alrahis, Ozgur Sinanoglu, Muhammad Shafique

**Abstract**: Extracting the architecture of layers of a given deep neural network (DNN) through hardware-based side channels allows adversaries to steal its intellectual property and even launch powerful adversarial attacks on the target system. In this work, we propose DNN-Alias, an obfuscation method for DNNs that forces all the layers in a given network to have similar execution traces, preventing attack models from differentiating between the layers. Towards this, DNN-Alias performs various layer-obfuscation operations, e.g., layer branching, layer deepening, etc, to alter the run-time traces while maintaining the functionality. DNN-Alias deploys an evolutionary algorithm to find the best combination of obfuscation operations in terms of maximizing the security level while maintaining a user-provided latency overhead budget. We demonstrate the effectiveness of our DNN-Alias technique by obfuscating the architecture of 700 randomly generated and obfuscated DNNs running on multiple Nvidia RTX 2080 TI GPU-based machines. Our experiments show that state-of-the-art side-channel architecture stealing attacks cannot extract the original DNN accurately. Moreover, we obfuscate the architecture of various DNNs, such as the VGG-11, VGG-13, ResNet-20, and ResNet-32 networks. Training the DNNs using the standard CIFAR10 dataset, we show that our DNN-Alias maintains the functionality of the original DNNs by preserving the original inference accuracy. Further, the experiments highlight that adversarial attack on obfuscated DNNs is unsuccessful.

摘要: 通过基于硬件的侧通道提取给定深度神经网络(DNN)的层次结构，使得攻击者能够窃取其知识产权，甚至对目标系统发起强大的对抗性攻击。在这项工作中，我们提出了DNN-Alias，这是一种DNN的混淆方法，它强制给定网络中的所有层具有相似的执行轨迹，防止攻击模型在层之间区分。为此，DNN-Alias执行各种层混淆操作，例如层分支、层加深等，以在保持功能的同时改变运行时跟踪。DNN-Alias部署了一种进化算法，以找到模糊操作的最佳组合，从而最大限度地提高安全级别，同时保持用户提供的延迟开销预算。我们通过对运行在多台基于NVIDIA RTX 2080 TI GPU的机器上的700个随机生成和模糊的DNN的架构进行模糊处理，展示了我们的DNN-Alias技术的有效性。实验表明，现有的旁路结构窃取攻击不能准确提取原始DNN。此外，我们还混淆了各种DNN的体系结构，如VGG-11、VGG-13、ResNet-20和ResNet-32网络。使用标准的CIFAR10数据集对DNN进行训练，我们的DNN-Alias通过保持原始DNN的推理精度来保持原始DNN的功能。此外，实验表明，对模糊DNN的对抗性攻击是不成功的。



## **33. Adv-Bot: Realistic Adversarial Botnet Attacks against Network Intrusion Detection Systems**

ADV-Bot：针对网络入侵检测系统的现实对抗性僵尸网络攻击 cs.CR

This work is published in Computers & Security (an Elsevier journal)  https://www.sciencedirect.com/science/article/pii/S016740482300086X

**SubmitDate**: 2023-03-12    [abs](http://arxiv.org/abs/2303.06664v1) [paper-pdf](http://arxiv.org/pdf/2303.06664v1)

**Authors**: Islam Debicha, Benjamin Cochez, Tayeb Kenaza, Thibault Debatty, Jean-Michel Dricot, Wim Mees

**Abstract**: Due to the numerous advantages of machine learning (ML) algorithms, many applications now incorporate them. However, many studies in the field of image classification have shown that MLs can be fooled by a variety of adversarial attacks. These attacks take advantage of ML algorithms' inherent vulnerability. This raises many questions in the cybersecurity field, where a growing number of researchers are recently investigating the feasibility of such attacks against machine learning-based security systems, such as intrusion detection systems. The majority of this research demonstrates that it is possible to fool a model using features extracted from a raw data source, but it does not take into account the real implementation of such attacks, i.e., the reverse transformation from theory to practice. The real implementation of these adversarial attacks would be influenced by various constraints that would make their execution more difficult. As a result, the purpose of this study was to investigate the actual feasibility of adversarial attacks, specifically evasion attacks, against network-based intrusion detection systems (NIDS), demonstrating that it is entirely possible to fool these ML-based IDSs using our proposed adversarial algorithm while assuming as many constraints as possible in a black-box setting. In addition, since it is critical to design defense mechanisms to protect ML-based IDSs against such attacks, a defensive scheme is presented. Realistic botnet traffic traces are used to assess this work. Our goal is to create adversarial botnet traffic that can avoid detection while still performing all of its intended malicious functionality.

摘要: 由于机器学习(ML)算法的众多优势，现在许多应用程序都将其纳入其中。然而，图像分类领域的许多研究表明，MLS可以被各种对抗性攻击所愚弄。这些攻击利用了ML算法固有的漏洞。这在网络安全领域引发了许多问题，最近越来越多的研究人员正在调查针对入侵检测系统等基于机器学习的安全系统进行此类攻击的可行性。大多数研究表明，使用从原始数据源提取的特征来愚弄模型是可能的，但它没有考虑到此类攻击的真正实现，即从理论到实践的反向转换。这些对抗性攻击的真正实施将受到各种限制的影响，这些限制将使它们的执行更加困难。因此，本研究的目的是调查针对基于网络的入侵检测系统(NID)的对抗性攻击，特别是逃避攻击的实际可行性，证明使用我们提出的对抗性算法来愚弄这些基于ML的入侵检测系统是完全可能的，同时假设在黑盒设置下尽可能多的约束。此外，由于设计防御机制以保护基于ML的入侵检测系统免受此类攻击是至关重要的，因此提出了一种防御方案。使用真实的僵尸网络流量跟踪来评估这项工作。我们的目标是创建敌意僵尸网络流量，以避免检测，同时仍可执行其所有预期的恶意功能。



## **34. Interpreting Hidden Semantics in the Intermediate Layers of 3D Point Cloud Classification Neural Network**

三维点云分类神经网络中间层隐含语义解释 cs.CV

**SubmitDate**: 2023-03-12    [abs](http://arxiv.org/abs/2303.06652v1) [paper-pdf](http://arxiv.org/pdf/2303.06652v1)

**Authors**: Weiquan Liu, Minghao Liu, Shijun Zheng, Cheng Wang

**Abstract**: Although 3D point cloud classification neural network models have been widely used, the in-depth interpretation of the activation of the neurons and layers is still a challenge. We propose a novel approach, named Relevance Flow, to interpret the hidden semantics of 3D point cloud classification neural networks. It delivers the class Relevance to the activated neurons in the intermediate layers in a back-propagation manner, and associates the activation of neurons with the input points to visualize the hidden semantics of each layer. Specially, we reveal that the 3D point cloud classification neural network has learned the plane-level and part-level hidden semantics in the intermediate layers, and utilize the normal and IoU to evaluate the consistency of both levels' hidden semantics. Besides, by using the hidden semantics, we generate the adversarial attack samples to attack 3D point cloud classifiers. Experiments show that our proposed method reveals the hidden semantics of the 3D point cloud classification neural network on ModelNet40 and ShapeNet, which can be used for the unsupervised point cloud part segmentation without labels and attacking the 3D point cloud classifiers.

摘要: 虽然三维点云分类神经网络模型已经得到了广泛的应用，但对神经元和层的激活过程的深入解释仍然是一个挑战。提出了一种新的解释三维点云分类神经网络隐含语义的方法--关联流。它以反向传播的方式将类相关性传递给中间层中被激活的神经元，并将神经元的激活与输入点相关联，以可视化每一层的隐藏语义。特别是，我们揭示了三维点云分类神经网络在中间层学习了平面级和零部件级的隐藏语义，并利用Normal和IOU来评估这两个级别的隐藏语义的一致性。此外，利用隐含语义生成了攻击三维点云分类器的对抗性攻击样本。实验表明，该方法在ModelNet40和ShapeNet上揭示了三维点云分类神经网络的隐含语义，可用于无标签的无监督点云部分分割和攻击三维点云分类器。



## **35. Query Attack by Multi-Identity Surrogates**

多身份代理的查询攻击 cs.LG

IEEE TRANSACTIONS ON ARTIFICIAL INTELLIGENCE

**SubmitDate**: 2023-03-12    [abs](http://arxiv.org/abs/2105.15010v5) [paper-pdf](http://arxiv.org/pdf/2105.15010v5)

**Authors**: Sizhe Chen, Zhehao Huang, Qinghua Tao, Xiaolin Huang

**Abstract**: Deep Neural Networks (DNNs) are acknowledged as vulnerable to adversarial attacks, while the existing black-box attacks require extensive queries on the victim DNN to achieve high success rates. For query-efficiency, surrogate models of the victim are used to generate transferable Adversarial Examples (AEs) because of their Gradient Similarity (GS), i.e., surrogates' attack gradients are similar to the victim's ones. However, it is generally neglected to exploit their similarity on outputs, namely the Prediction Similarity (PS), to filter out inefficient queries by surrogates without querying the victim. To jointly utilize and also optimize surrogates' GS and PS, we develop QueryNet, a unified attack framework that can significantly reduce queries. QueryNet creatively attacks by multi-identity surrogates, i.e., crafts several AEs for one sample by different surrogates, and also uses surrogates to decide on the most promising AE for the query. After that, the victim's query feedback is accumulated to optimize not only surrogates' parameters but also their architectures, enhancing both the GS and the PS. Although QueryNet has no access to pre-trained surrogates' prior, it reduces queries by averagely about an order of magnitude compared to alternatives within an acceptable time, according to our comprehensive experiments: 11 victims (including two commercial models) on MNIST/CIFAR10/ImageNet, allowing only 8-bit image queries, and no access to the victim's training data. The code is available at https://github.com/Sizhe-Chen/QueryNet.

摘要: 深度神经网络(DNN)被认为容易受到对抗性攻击，而现有的黑盒攻击需要对受害者DNN进行广泛的查询才能获得高的成功率。为了提高查询效率，受害者的代理模型被用来生成可转移的对抗实例，因为它们具有梯度相似性，即代理的攻击梯度与受害者的攻击梯度相似。然而，通常忽略了利用它们在输出上的相似性，即预测相似度(PS)来过滤代理在不查询受害者的情况下的低效查询。为了联合利用并优化代理的GS和PS，我们开发了QueryNet，这是一个可以显著减少查询的统一攻击框架。QueryNet创造性地利用多身份代理进行攻击，即通过不同的代理为一个样本构造多个代理实体，并使用代理为查询选择最有希望的代理实体。之后，受害者的查询反馈被累积，不仅优化了代理的参数，还优化了它们的体系结构，提高了GS和PS。虽然QueryNet无法访问预先训练的代理人的先前，但根据我们的综合实验：11名受害者(包括两个商业模型)在MNIST/CIFAR10/ImageNet上仅允许8位图像查询，并且无法访问受害者的训练数据，与替代方案相比，它在可接受的时间内平均减少了一个数量级的查询。代码可在https://github.com/Sizhe-Chen/QueryNet.上获得



## **36. Adaptive Local Adversarial Attacks on 3D Point Clouds for Augmented Reality**

增强现实中3D点云的自适应局部对抗攻击 cs.CV

**SubmitDate**: 2023-03-12    [abs](http://arxiv.org/abs/2303.06641v1) [paper-pdf](http://arxiv.org/pdf/2303.06641v1)

**Authors**: Weiquan Liu, Shijun Zheng, Cheng Wang

**Abstract**: As the key technology of augmented reality (AR), 3D recognition and tracking are always vulnerable to adversarial examples, which will cause serious security risks to AR systems. Adversarial examples are beneficial to improve the robustness of the 3D neural network model and enhance the stability of the AR system. At present, most 3D adversarial attack methods perturb the entire point cloud to generate adversarial examples, which results in high perturbation costs and difficulty in reconstructing the corresponding real objects in the physical world. In this paper, we propose an adaptive local adversarial attack method (AL-Adv) on 3D point clouds to generate adversarial point clouds. First, we analyze the vulnerability of the 3D network model and extract the salient regions of the input point cloud, namely the vulnerable regions. Second, we propose an adaptive gradient attack algorithm that targets vulnerable regions. The proposed attack algorithm adaptively assigns different disturbances in different directions of the three-dimensional coordinates of the point cloud. Experimental results show that our proposed method AL-Adv achieves a higher attack success rate than the global attack method. Specifically, the adversarial examples generated by the AL-Adv demonstrate good imperceptibility and small generation costs.

摘要: 作为增强现实(AR)的关键技术，3D识别与跟踪往往容易受到敌意攻击，这将给AR系统带来严重的安全隐患。对抗性例子有利于提高3D神经网络模型的鲁棒性，增强AR系统的稳定性。目前，大多数3D对抗性攻击方法都是对整个点云进行扰动来生成对抗性实例，这导致了较高的扰动代价和重建物理世界中对应的真实对象的困难。本文提出了一种基于三维点云的自适应局部对抗攻击方法(AL-ADV)来生成对抗点云。首先，分析三维网络模型的脆弱性，提取输入点云的显著区域，即脆弱区域。其次，提出了一种针对易受攻击区域的自适应梯度攻击算法。该攻击算法在点云三维坐标的不同方向上自适应地分配不同的干扰。实验结果表明，我们提出的方法AL-ADV比全局攻击方法具有更高的攻击成功率。具体地说，AL-ADV生成的对抗性示例具有良好的隐蔽性和较小的生成成本。



## **37. Multi-metrics adaptively identifies backdoors in Federated learning**

多指标自适应地识别联合学习中的后门 cs.CR

13 pages, 8 figures and 6 tables

**SubmitDate**: 2023-03-12    [abs](http://arxiv.org/abs/2303.06601v1) [paper-pdf](http://arxiv.org/pdf/2303.06601v1)

**Authors**: Siquan Huang, Yijiang Li, Chong Chen, Leyu Shi, Ying Gao

**Abstract**: The decentralized and privacy-preserving nature of federated learning (FL) makes it vulnerable to backdoor attacks aiming to manipulate the behavior of the resulting model on specific adversary-chosen inputs. However, most existing defenses based on statistical differences take effect only against specific attacks, especially when the malicious gradients are similar to benign ones or the data are highly non-independent and identically distributed (non-IID). In this paper, we revisit the distance-based defense methods and discover that i) Euclidean distance becomes meaningless in high dimensions and ii) malicious gradients with diverse characteristics cannot be identified by a single metric. To this end, we present a simple yet effective defense strategy with multi-metrics and dynamic weighting to identify backdoors adaptively. Furthermore, our novel defense has no reliance on predefined assumptions over attack settings or data distributions and little impact on benign performance. To evaluate the effectiveness of our approach, we conduct comprehensive experiments on different datasets under various attack settings, where our method achieves the best defensive performance. For instance, we achieve the lowest backdoor accuracy of 3.06% under the difficult Edge-case PGD, showing significant superiority over previous defenses. The results also demonstrate that our method can be well-adapted to a wide range of non-IID degrees without sacrificing the benign performance.

摘要: 联邦学习(FL)的去中心化和隐私保护特性使其容易受到后门攻击，目的是在特定对手选择的输入上操纵结果模型的行为。然而，现有的大多数基于统计差异的防御措施只对特定的攻击有效，特别是当恶意梯度类似于良性梯度或数据具有高度非独立和同分布(Non-IID)时。在本文中，我们回顾了基于距离的防御方法，发现i)欧氏距离在高维中变得没有意义，ii)具有不同特征的恶意梯度不能用单一的度量来识别。为此，我们提出了一种简单而有效的防御策略，采用多指标和动态加权来自适应地识别后门。此外，我们的新型防御不依赖于对攻击设置或数据分布的预定义假设，并且对良性性能几乎没有影响。为了评估该方法的有效性，我们在不同的攻击环境下对不同的数据集进行了全面的实验，其中我们的方法取得了最好的防御性能。例如，在困难的Edge-Case PGD下，我们实现了3.06%的最低后门精度，显示出明显优于以前的防御。结果还表明，我们的方法可以很好地适应广泛的非IID程度，而不牺牲良好的性能。



## **38. STPrivacy: Spatio-Temporal Privacy-Preserving Action Recognition**

STPrivacy：时空隐私保护动作识别 cs.CV

**SubmitDate**: 2023-03-12    [abs](http://arxiv.org/abs/2301.03046v2) [paper-pdf](http://arxiv.org/pdf/2301.03046v2)

**Authors**: Ming Li, Xiangyu Xu, Hehe Fan, Pan Zhou, Jun Liu, Jia-Wei Liu, Jiahe Li, Jussi Keppo, Mike Zheng Shou, Shuicheng Yan

**Abstract**: Existing methods of privacy-preserving action recognition (PPAR) mainly focus on frame-level (spatial) privacy removal through 2D CNNs. Unfortunately, they have two major drawbacks. First, they may compromise temporal dynamics in input videos, which are critical for accurate action recognition. Second, they are vulnerable to practical attacking scenarios where attackers probe for privacy from an entire video rather than individual frames. To address these issues, we propose a novel framework STPrivacy to perform video-level PPAR. For the first time, we introduce vision Transformers into PPAR by treating a video as a tubelet sequence, and accordingly design two complementary mechanisms, i.e., sparsification and anonymization, to remove privacy from a spatio-temporal perspective. In specific, our privacy sparsification mechanism applies adaptive token selection to abandon action-irrelevant tubelets. Then, our anonymization mechanism implicitly manipulates the remaining action-tubelets to erase privacy in the embedding space through adversarial learning. These mechanisms provide significant advantages in terms of privacy preservation for human eyes and action-privacy trade-off adjustment during deployment. We additionally contribute the first two large-scale PPAR benchmarks, VP-HMDB51 and VP-UCF101, to the community. Extensive evaluations on them, as well as two other tasks, validate the effectiveness and generalization capability of our framework.

摘要: 现有的隐私保护动作识别(PPAR)方法主要集中在通过2D CNN去除帧级(空间)隐私。不幸的是，它们有两个主要缺陷。首先，它们可能会影响输入视频中的时间动态，而时间动态对于准确的动作识别至关重要。其次，它们容易受到实际攻击场景的攻击，即攻击者从整个视频而不是单个帧来探测隐私。为了解决这些问题，我们提出了一种新的框架STPrivacy来执行视频级PPAR。首次将视觉变形器引入到PPAR中，将视频看作一个元组序列，并相应地设计了稀疏化和匿名化两种互补机制，从时空的角度去除隐私。具体地说，我们的隐私稀疏机制采用自适应令牌选择来丢弃与动作无关的tubelet。然后，我们的匿名化机制隐含地操纵剩余的动作元组，通过对抗性学习消除嵌入空间中的隐私。这些机制在人眼隐私保护和部署过程中的动作-隐私权衡调整方面具有显著优势。此外，我们还向社区贡献了头两个大型PPAR基准，VP-HMDB51和VP-UCF101。对它们的广泛评估以及另外两项任务，验证了我们框架的有效性和推广能力。



## **39. Disclosure Risk from Homogeneity Attack in Differentially Private Frequency Distribution**

差分私密频率分布中同质性攻击的泄漏风险 cs.CR

**SubmitDate**: 2023-03-11    [abs](http://arxiv.org/abs/2101.00311v5) [paper-pdf](http://arxiv.org/pdf/2101.00311v5)

**Authors**: Fang Liu, Xingyuan Zhao

**Abstract**: Differential privacy (DP) provides a robust model to achieve privacy guarantees for released information. We examine the protection potency of sanitized multi-dimensional frequency distributions via DP randomization mechanisms against homogeneity attack (HA). HA allows adversaries to obtain the exact values on sensitive attributes for their targets without having to identify them from the released data. We propose measures for disclosure risk from HA and derive closed-form relationships between the privacy loss parameters in DP and the disclosure risk from HA. The availability of the closed-form relationships assists understanding the abstract concepts of DP and privacy loss parameters by putting them in the context of a concrete privacy attack and offers a perspective for choosing privacy loss parameters when employing DP mechanisms in information sanitization and release in practice. We apply the closed-form mathematical relationships in real-life datasets to demonstrate the assessment of disclosure risk due to HA on differentially private sanitized frequency distributions at various privacy loss parameters.

摘要: 差异隐私(DP)提供了一种健壮的模型来实现对发布信息的隐私保障。我们通过DP随机化机制来检验经过消毒的多维频率分布对同质性攻击(HA)的保护效力。HA允许攻击者获得其目标的敏感属性的精确值，而不必从发布的数据中识别它们。提出了HA信息泄露风险的度量方法，并推导出DP中的隐私损失参数与HA信息泄露风险之间的闭合关系。封闭关系的可用性通过将DP和隐私丢失参数置于具体的隐私攻击的上下文中来帮助理解DP和隐私丢失参数的抽象概念，并为在实践中使用DP机制进行信息清理和发布时选择隐私丢失参数提供了一个视角。我们将封闭形式的数学关系应用于真实数据集中，以演示在不同隐私损失参数下，HA对不同隐私消毒频率分布的泄露风险的评估。



## **40. Anomaly Detection with Ensemble of Encoder and Decoder**

基于编解码器集成的异常检测 cs.LG

**SubmitDate**: 2023-03-11    [abs](http://arxiv.org/abs/2303.06431v1) [paper-pdf](http://arxiv.org/pdf/2303.06431v1)

**Authors**: Xijuan Sun, Di Wu, Arnaud Zinflou, Benoit Boulet

**Abstract**: Hacking and false data injection from adversaries can threaten power grids' everyday operations and cause significant economic loss. Anomaly detection in power grids aims to detect and discriminate anomalies caused by cyber attacks against the power system, which is essential for keeping power grids working correctly and efficiently. Different methods have been applied for anomaly detection, such as statistical methods and machine learning-based methods. Usually, machine learning-based methods need to model the normal data distribution. In this work, we propose a novel anomaly detection method by modeling the data distribution of normal samples via multiple encoders and decoders. Specifically, the proposed method maps input samples into a latent space and then reconstructs output samples from latent vectors. The extra encoder finally maps reconstructed samples to latent representations. During the training phase, we optimize parameters by minimizing the reconstruction loss and encoding loss. Training samples are re-weighted to focus more on missed correlations between features of normal data. Furthermore, we employ the long short-term memory model as encoders and decoders to test its effectiveness. We also investigate a meta-learning-based framework for hyper-parameter tuning of our approach. Experiment results on network intrusion and power system datasets demonstrate the effectiveness of our proposed method, where our models consistently outperform all baselines.

摘要: 来自对手的黑客攻击和虚假数据注入可能威胁电网的日常运行，并造成重大经济损失。电网异常检测的目的是检测和识别网络攻击对电力系统造成的异常，这是保证电网正常高效运行的关键。不同的方法被应用于异常检测，如统计方法和基于机器学习的方法。通常，基于机器学习的方法需要对正态数据分布进行建模。在这项工作中，我们提出了一种新的异常检测方法，通过多个编解码器对正常样本的数据分布进行建模。具体地说，该方法将输入样本映射到潜在空间，然后从潜在向量重构输出样本。额外的编码器最终将重构的样本映射到潜在表示。在训练阶段，我们通过最小化重建损失和编码损失来优化参数。训练样本被重新加权，以更多地关注正常数据的特征之间的遗漏相关性。此外，我们使用长短期记忆模型作为编解码器来测试其有效性。我们还研究了一个基于元学习的框架，用于对我们的方法进行超参数调整。在网络入侵和电力系统数据集上的实验结果表明，我们提出的方法是有效的，我们的模型一致地优于所有基线。



## **41. Improving the Robustness of Deep Convolutional Neural Networks Through Feature Learning**

利用特征学习提高深卷积神经网络的稳健性 cs.CV

8 pages, 12 figures, 6 tables. Work in process

**SubmitDate**: 2023-03-11    [abs](http://arxiv.org/abs/2303.06425v1) [paper-pdf](http://arxiv.org/pdf/2303.06425v1)

**Authors**: Jin Ding, Jie-Chao Zhao, Yong-Zhi Sun, Ping Tan, Ji-En Ma, You-Tong Fang

**Abstract**: Deep convolutional neural network (DCNN for short) models are vulnerable to examples with small perturbations. Adversarial training (AT for short) is a widely used approach to enhance the robustness of DCNN models by data augmentation. In AT, the DCNN models are trained with clean examples and adversarial examples (AE for short) which are generated using a specific attack method, aiming to gain ability to defend themselves when facing the unseen AEs. However, in practice, the trained DCNN models are often fooled by the AEs generated by the novel attack methods. This naturally raises a question: can a DCNN model learn certain features which are insensitive to small perturbations, and further defend itself no matter what attack methods are presented. To answer this question, this paper makes a beginning effort by proposing a shallow binary feature module (SBFM for short), which can be integrated into any popular backbone. The SBFM includes two types of layers, i.e., Sobel layer and threshold layer. In Sobel layer, there are four parallel feature maps which represent horizontal, vertical, and diagonal edge features, respectively. And in threshold layer, it turns the edge features learnt by Sobel layer to the binary features, which then are feeded into the fully connected layers for classification with the features learnt by the backbone. We integrate SBFM into VGG16 and ResNet34, respectively, and conduct experiments on multiple datasets. Experimental results demonstrate, under FGSM attack with $\epsilon=8/255$, the SBFM integrated models can achieve averagely 35\% higher accuracy than the original ones, and in CIFAR-10 and TinyImageNet datasets, the SBFM integrated models can achieve averagely 75\% classification accuracy. The work in this paper shows it is promising to enhance the robustness of DCNN models through feature learning.

摘要: 深层卷积神经网络(DCNN)模型容易受到小扰动样本的影响。对抗性训练是一种广泛使用的通过数据增强来增强DCNN模型稳健性的方法。在AT中，DCNN模型使用特定攻击方法生成的干净实例和敌意实例(简称AE)进行训练，目的是在面对看不见的AE时获得自卫能力。然而，在实际应用中，训练好的DCNN模型往往会被新的攻击方法产生的攻击事件所愚弄。这自然提出了一个问题：DCNN模型是否能够学习对小扰动不敏感的某些特征，并在任何攻击方法提出的情况下进一步自卫。为了回答这个问题，本文首先提出了一种可集成到任何主流主干中的浅二进制特征模块(SBFM)。SBFM包括两种类型的层，即Sobel层和阈值层。在Sobel层中，有四个平行的特征图，分别表示水平、垂直和对角的边缘特征。在阈值层中，将Sobel层学习到的边缘特征转化为二值特征，然后将二值特征送入全连通层，与主干学习的特征进行分类。我们将SBFM分别集成到VGG16和ResNet34中，并在多个数据集上进行了实验。实验结果表明，在$epsilon=8/255$的FGSM攻击下，SBFM集成模型的分类正确率比原始模型平均提高了35%，在CIFAR-10和TinyImageNet数据集中，SBFM集成模型的分类正确率平均达到75%。本文的工作表明，通过特征学习来增强DCNN模型的稳健性是很有前途的。



## **42. MorDIFF: Recognition Vulnerability and Attack Detectability of Face Morphing Attacks Created by Diffusion Autoencoders**

MorDIFF：扩散自动编码器造成的人脸变形攻击的识别漏洞和攻击可检测性 cs.CV

Accepted at the 11th International Workshop on Biometrics and  Forensics 2023 (IWBF 2023)

**SubmitDate**: 2023-03-11    [abs](http://arxiv.org/abs/2302.01843v2) [paper-pdf](http://arxiv.org/pdf/2302.01843v2)

**Authors**: Naser Damer, Meiling Fang, Patrick Siebke, Jan Niklas Kolf, Marco Huber, Fadi Boutros

**Abstract**: Investigating new methods of creating face morphing attacks is essential to foresee novel attacks and help mitigate them. Creating morphing attacks is commonly either performed on the image-level or on the representation-level. The representation-level morphing has been performed so far based on generative adversarial networks (GAN) where the encoded images are interpolated in the latent space to produce a morphed image based on the interpolated vector. Such a process was constrained by the limited reconstruction fidelity of GAN architectures. Recent advances in the diffusion autoencoder models have overcome the GAN limitations, leading to high reconstruction fidelity. This theoretically makes them a perfect candidate to perform representation-level face morphing. This work investigates using diffusion autoencoders to create face morphing attacks by comparing them to a wide range of image-level and representation-level morphs. Our vulnerability analyses on four state-of-the-art face recognition models have shown that such models are highly vulnerable to the created attacks, the MorDIFF, especially when compared to existing representation-level morphs. Detailed detectability analyses are also performed on the MorDIFF, showing that they are as challenging to detect as other morphing attacks created on the image- or representation-level. Data and morphing script are made public: https://github.com/naserdamer/MorDIFF.

摘要: 研究创建面部变形攻击的新方法对于预见新的攻击并帮助缓解它们是至关重要的。创建变形攻击通常是在图像级或表示级执行的。到目前为止，表示级变形是基于生成对抗网络(GAN)执行的，其中在潜在空间中对编码图像进行内插，以基于内插向量产生变形图像。这一过程受到GaN结构有限重建保真度的限制。扩散式自动编码器模型的最新进展克服了GaN的限制，导致了高重建保真度。从理论上讲，这使它们成为执行表示级面部变形的完美候选者。这项工作使用扩散自动编码器来创建人脸变形攻击，通过将它们与广泛的图像级和表示级变形进行比较。我们对四个最先进的人脸识别模型进行的漏洞分析表明，这些模型对创建的攻击MorDIFF非常脆弱，特别是与现有的表示级变形相比。还对MorDIFF进行了详细的可检测性分析，表明它们与在图像或表示层上创建的其他变形攻击一样具有挑战性。数据和变形脚本公开：https://github.com/naserdamer/MorDIFF.



## **43. Adversarial Attacks and Defenses in Machine Learning-Powered Networks: A Contemporary Survey**

机器学习网络中的对抗性攻击与防御：当代综述 cs.LG

46 pages, 21 figures

**SubmitDate**: 2023-03-11    [abs](http://arxiv.org/abs/2303.06302v1) [paper-pdf](http://arxiv.org/pdf/2303.06302v1)

**Authors**: Yulong Wang, Tong Sun, Shenghong Li, Xin Yuan, Wei Ni, Ekram Hossain, H. Vincent Poor

**Abstract**: Adversarial attacks and defenses in machine learning and deep neural network have been gaining significant attention due to the rapidly growing applications of deep learning in the Internet and relevant scenarios. This survey provides a comprehensive overview of the recent advancements in the field of adversarial attack and defense techniques, with a focus on deep neural network-based classification models. Specifically, we conduct a comprehensive classification of recent adversarial attack methods and state-of-the-art adversarial defense techniques based on attack principles, and present them in visually appealing tables and tree diagrams. This is based on a rigorous evaluation of the existing works, including an analysis of their strengths and limitations. We also categorize the methods into counter-attack detection and robustness enhancement, with a specific focus on regularization-based methods for enhancing robustness. New avenues of attack are also explored, including search-based, decision-based, drop-based, and physical-world attacks, and a hierarchical classification of the latest defense methods is provided, highlighting the challenges of balancing training costs with performance, maintaining clean accuracy, overcoming the effect of gradient masking, and ensuring method transferability. At last, the lessons learned and open challenges are summarized with future research opportunities recommended.

摘要: 由于深度学习在互联网和相关场景中的应用日益广泛，机器学习和深度神经网络中的对抗性攻击和防御已经得到了广泛的关注。这篇综述全面概述了对抗性攻击和防御技术领域的最新进展，重点介绍了基于深度神经网络的分类模型。具体地说，我们根据攻击原理对目前的对抗性攻击方法和最新的对抗性防御技术进行了全面的分类，并以视觉上吸引人的表格和树形图来呈现它们。这是基于对现有作品的严格评估，包括对它们的优点和局限性的分析。我们还将这些方法分为反攻击检测和稳健性增强两类，重点介绍了基于正则化的增强稳健性的方法。还探索了新的攻击途径，包括基于搜索、基于决策、基于Drop和物理世界的攻击，并提供了最新防御方法的分层分类，突出了在平衡训练成本和性能、保持干净准确性、克服梯度掩蔽的影响和确保方法可转移性方面的挑战。最后，总结了本研究的经验教训和面临的挑战，并对未来的研究方向进行了展望。



## **44. Investigating Stateful Defenses Against Black-Box Adversarial Examples**

黑箱对抗状态防御的研究实例 cs.CR

**SubmitDate**: 2023-03-11    [abs](http://arxiv.org/abs/2303.06280v1) [paper-pdf](http://arxiv.org/pdf/2303.06280v1)

**Authors**: Ryan Feng, Ashish Hooda, Neal Mangaokar, Kassem Fawaz, Somesh Jha, Atul Prakash

**Abstract**: Defending machine-learning (ML) models against white-box adversarial attacks has proven to be extremely difficult. Instead, recent work has proposed stateful defenses in an attempt to defend against a more restricted black-box attacker. These defenses operate by tracking a history of incoming model queries, and rejecting those that are suspiciously similar. The current state-of-the-art stateful defense Blacklight was proposed at USENIX Security '22 and claims to prevent nearly 100% of attacks on both the CIFAR10 and ImageNet datasets. In this paper, we observe that an attacker can significantly reduce the accuracy of a Blacklight-protected classifier (e.g., from 82.2% to 6.4% on CIFAR10) by simply adjusting the parameters of an existing black-box attack. Motivated by this surprising observation, since existing attacks were evaluated by the Blacklight authors, we provide a systematization of stateful defenses to understand why existing stateful defense models fail. Finally, we propose a stronger evaluation strategy for stateful defenses comprised of adaptive score and hard-label based black-box attacks. We use these attacks to successfully reduce even reconfigured versions of Blacklight to as low as 0% robust accuracy.

摘要: 保护机器学习(ML)模型免受白盒对手攻击已被证明是极其困难的。相反，最近的工作提出了状态防御，试图防御更受限制的黑匣子攻击者。这些防御通过跟踪传入模型查询的历史，并拒绝那些可疑的相似查询来运行。目前最先进的状态防御Blacklight是在USENIX Security‘22上提出的，声称可以防止对CIFAR10和ImageNet数据集的近100%攻击。在本文中，我们观察到攻击者可以通过简单地调整现有黑盒攻击的参数来显著降低受Blacklight保护的分类器的准确率(例如，在CIFAR10上从82.2%降低到6.4%)。出于这一令人惊讶的观察，由于现有攻击是由Blacklight作者评估的，我们提供了状态防御的系统化，以了解现有状态防御模型失败的原因。最后，提出了一种更强的状态防御评估策略，该策略由自适应评分和基于硬标签的黑盒攻击组成。我们使用这些攻击成功地将重新配置的Blacklight版本降低到低至0%的稳健准确率。



## **45. Do we need entire training data for adversarial training?**

对抗性训练需要完整的训练数据吗？ cs.CV

6 pages, 4 figures

**SubmitDate**: 2023-03-10    [abs](http://arxiv.org/abs/2303.06241v1) [paper-pdf](http://arxiv.org/pdf/2303.06241v1)

**Authors**: Vipul Gupta, Apurva Narayan

**Abstract**: Deep Neural Networks (DNNs) are being used to solve a wide range of problems in many domains including safety-critical domains like self-driving cars and medical imagery. DNNs suffer from vulnerability against adversarial attacks. In the past few years, numerous approaches have been proposed to tackle this problem by training networks using adversarial training. Almost all the approaches generate adversarial examples for the entire training dataset, thus increasing the training time drastically. We show that we can decrease the training time for any adversarial training algorithm by using only a subset of training data for adversarial training. To select the subset, we filter the adversarially-prone samples from the training data. We perform a simple adversarial attack on all training examples to filter this subset. In this attack, we add a small perturbation to each pixel and a few grid lines to the input image.   We perform adversarial training on the adversarially-prone subset and mix it with vanilla training performed on the entire dataset. Our results show that when our method-agnostic approach is plugged into FGSM, we achieve a speedup of 3.52x on MNIST and 1.98x on the CIFAR-10 dataset with comparable robust accuracy. We also test our approach on state-of-the-art Free adversarial training and achieve a speedup of 1.2x in training time with a marginal drop in robust accuracy on the ImageNet dataset.

摘要: 深度神经网络(DNN)正被用来解决许多领域的广泛问题，包括自动驾驶汽车和医学成像等安全关键领域。DNN容易受到敌意攻击。在过去的几年里，已经提出了许多办法来解决这一问题，方法是使用对抗性训练来训练网络。几乎所有的方法都为整个训练数据集生成对抗性的样本，从而大大增加了训练时间。我们证明，只要使用训练数据的一个子集进行对抗性训练，就可以减少任何对抗性训练算法的训练时间。为了选择子集，我们从训练数据中过滤出易受攻击的样本。我们对所有训练样本执行简单的对抗性攻击来过滤这个子集。在这种攻击中，我们为每个像素添加一个小扰动，并在输入图像中添加一些网格线。我们对易发生对抗性的子集进行对抗性训练，并将其与在整个数据集上执行的普通训练相混合。我们的结果表明，当我们的方法无关的方法被插入到FGSM中时，我们在MNIST上获得了3.52倍的加速比，在CIFAR-10数据集上获得了1.98倍的加速比，并且具有相当的鲁棒性。我们还在最先进的自由对手训练上测试了我们的方法，在ImageNet数据集上的稳健准确率略有下降的情况下，训练时间加速了1.2倍。



## **46. Turning Strengths into Weaknesses: A Certified Robustness Inspired Attack Framework against Graph Neural Networks**

变优势为劣势：一种经验证的图神经网络健壮性启发攻击框架 cs.CR

Accepted by CVPR 2023

**SubmitDate**: 2023-03-10    [abs](http://arxiv.org/abs/2303.06199v1) [paper-pdf](http://arxiv.org/pdf/2303.06199v1)

**Authors**: Binghui Wang, Meng Pang, Yun Dong

**Abstract**: Graph neural networks (GNNs) have achieved state-of-the-art performance in many graph learning tasks. However, recent studies show that GNNs are vulnerable to both test-time evasion and training-time poisoning attacks that perturb the graph structure. While existing attack methods have shown promising attack performance, we would like to design an attack framework to further enhance the performance. In particular, our attack framework is inspired by certified robustness, which was originally used by defenders to defend against adversarial attacks. We are the first, from the attacker perspective, to leverage its properties to better attack GNNs. Specifically, we first derive nodes' certified perturbation sizes against graph evasion and poisoning attacks based on randomized smoothing, respectively. A larger certified perturbation size of a node indicates this node is theoretically more robust to graph perturbations. Such a property motivates us to focus more on nodes with smaller certified perturbation sizes, as they are easier to be attacked after graph perturbations. Accordingly, we design a certified robustness inspired attack loss, when incorporated into (any) existing attacks, produces our certified robustness inspired attack counterpart. We apply our framework to the existing attacks and results show it can significantly enhance the existing base attacks' performance.

摘要: 图形神经网络(GNN)在许多图形学习任务中取得了最好的性能。然而，最近的研究表明，GNN容易受到测试时间逃避和训练时间中毒攻击，这些攻击扰乱了图的结构。虽然现有的攻击方法已经显示出良好的攻击性能，但我们希望设计一个攻击框架来进一步提高性能。特别是，我们的攻击框架的灵感来自认证的健壮性，这最初是防御者用来防御对手攻击的。从攻击者的角度来看，我们是第一个利用其特性更好地攻击GNN的人。具体地说，我们首先基于随机化平滑分别推导出针对图规避攻击和中毒攻击的节点认证扰动大小。节点的认证扰动大小越大，表明该节点在理论上对图扰动的鲁棒性更强。这样的性质促使我们更多地关注具有较小认证扰动大小的节点，因为它们在图扰动后更容易受到攻击。因此，我们设计了一个认证的健壮性启发攻击损失，当整合到(任何)现有攻击中时，产生我们认证的健壮性启发攻击对手。我们将该框架应用于现有的攻击中，结果表明，该框架可以显著提高现有的基本攻击的性能。



## **47. Harnessing the Speed and Accuracy of Machine Learning to Advance Cybersecurity**

利用机器学习的速度和准确性提高网络安全 cs.CR

**SubmitDate**: 2023-03-10    [abs](http://arxiv.org/abs/2302.12415v2) [paper-pdf](http://arxiv.org/pdf/2302.12415v2)

**Authors**: Khatoon Mohammed

**Abstract**: As cyber attacks continue to increase in frequency and sophistication, detecting malware has become a critical task for maintaining the security of computer systems. Traditional signature-based methods of malware detection have limitations in detecting complex and evolving threats. In recent years, machine learning (ML) has emerged as a promising solution to detect malware effectively. ML algorithms are capable of analyzing large datasets and identifying patterns that are difficult for humans to identify. This paper presents a comprehensive review of the state-of-the-art ML techniques used in malware detection, including supervised and unsupervised learning, deep learning, and reinforcement learning. We also examine the challenges and limitations of ML-based malware detection, such as the potential for adversarial attacks and the need for large amounts of labeled data. Furthermore, we discuss future directions in ML-based malware detection, including the integration of multiple ML algorithms and the use of explainable AI techniques to enhance the interpret ability of ML-based detection systems. Our research highlights the potential of ML-based techniques to improve the speed and accuracy of malware detection, and contribute to enhancing cybersecurity

摘要: 随着网络攻击的频率和复杂性不断增加，检测恶意软件已成为维护计算机系统安全的关键任务。传统的基于特征码的恶意软件检测方法在检测复杂和不断变化的威胁方面存在局限性。近年来，机器学习作为一种有效检测恶意软件的解决方案应运而生。ML算法能够分析大型数据集，并识别人类难以识别的模式。本文对恶意软件检测中使用的最大似然学习技术进行了全面的综述，包括监督学习和非监督学习、深度学习和强化学习。我们还研究了基于ML的恶意软件检测的挑战和局限性，例如潜在的对抗性攻击和对大量标记数据的需求。此外，我们还讨论了基于ML的恶意软件检测的未来发展方向，包括集成多种ML算法和使用可解释人工智能技术来增强基于ML的检测系统的解释能力。我们的研究突出了基于ML的技术在提高恶意软件检测的速度和准确性方面的潜力，并有助于增强网络安全



## **48. Learning the Legibility of Visual Text Perturbations**

学习视觉文本扰动的易读性 cs.CL

14 pages, 7 figures. Accepted at EACL 2023 (main, long)

**SubmitDate**: 2023-03-10    [abs](http://arxiv.org/abs/2303.05077v2) [paper-pdf](http://arxiv.org/pdf/2303.05077v2)

**Authors**: Dev Seth, Rickard Stureborg, Danish Pruthi, Bhuwan Dhingra

**Abstract**: Many adversarial attacks in NLP perturb inputs to produce visually similar strings ('ergo' $\rightarrow$ '$\epsilon$rgo') which are legible to humans but degrade model performance. Although preserving legibility is a necessary condition for text perturbation, little work has been done to systematically characterize it; instead, legibility is typically loosely enforced via intuitions around the nature and extent of perturbations. Particularly, it is unclear to what extent can inputs be perturbed while preserving legibility, or how to quantify the legibility of a perturbed string. In this work, we address this gap by learning models that predict the legibility of a perturbed string, and rank candidate perturbations based on their legibility. To do so, we collect and release LEGIT, a human-annotated dataset comprising the legibility of visually perturbed text. Using this dataset, we build both text- and vision-based models which achieve up to $0.91$ F1 score in predicting whether an input is legible, and an accuracy of $0.86$ in predicting which of two given perturbations is more legible. Additionally, we discover that legible perturbations from the LEGIT dataset are more effective at lowering the performance of NLP models than best-known attack strategies, suggesting that current models may be vulnerable to a broad range of perturbations beyond what is captured by existing visual attacks. Data, code, and models are available at https://github.com/dvsth/learning-legibility-2023.

摘要: NLP中的许多对抗性攻击会扰乱输入以产生视觉上相似的字符串(‘ergo’$\right tarrow$‘$\epsilon$rgo’)，这些字符串人类可读，但会降低模型性能。尽管保持易读性是文本扰动的必要条件，但几乎没有做过系统地描述它的工作；相反，易读性通常是通过围绕扰动的性质和程度的直觉来松散地强制执行的。特别是，目前还不清楚在保持易读性的同时，输入可以被干扰到什么程度，也不清楚如何量化被干扰的字符串的可读性。在这项工作中，我们通过学习预测扰动字符串的可读性的模型来解决这一差距，并根据它们的可读性对候选扰动进行排名。为了做到这一点，我们收集并发布Legit，这是一个人类注释的数据集，包括视觉上受到干扰的文本的可读性。使用这个数据集，我们建立了基于文本和基于视觉的模型，在预测输入是否清晰方面达到了高达0.91美元的F1分数，在预测两个给定扰动中的哪一个更易读方面达到了0.86美元的精度。此外，我们发现，与最著名的攻击策略相比，来自合法数据集的可识别的扰动在降低NLP模型的性能方面更有效，这表明当前的模型可能容易受到现有视觉攻击捕获的更大范围的扰动的影响。有关数据、代码和模型，请访问https://github.com/dvsth/learning-legibility-2023.



## **49. Exploring the Relationship between Architecture and Adversarially Robust Generalization**

探索体系结构和相反的健壮性泛化之间的关系 cs.LG

**SubmitDate**: 2023-03-10    [abs](http://arxiv.org/abs/2209.14105v2) [paper-pdf](http://arxiv.org/pdf/2209.14105v2)

**Authors**: Aishan Liu, Shiyu Tang, Siyuan Liang, Ruihao Gong, Boxi Wu, Xianglong Liu, Dacheng Tao

**Abstract**: Adversarial training has been demonstrated to be one of the most effective remedies for defending adversarial examples, yet it often suffers from the huge robustness generalization gap on unseen testing adversaries, deemed as the adversarially robust generalization problem. Despite the preliminary understandings devoted to adversarially robust generalization, little is known from the architectural perspective. To bridge the gap, this paper for the first time systematically investigated the relationship between adversarially robust generalization and architectural design. Inparticular, we comprehensively evaluated 20 most representative adversarially trained architectures on ImageNette and CIFAR-10 datasets towards multiple `p-norm adversarial attacks. Based on the extensive experiments, we found that, under aligned settings, Vision Transformers (e.g., PVT, CoAtNet) often yield better adversarially robust generalization while CNNs tend to overfit on specific attacks and fail to generalize on multiple adversaries. To better understand the nature behind it, we conduct theoretical analysis via the lens of Rademacher complexity. We revealed the fact that the higher weight sparsity contributes significantly towards the better adversarially robust generalization of Transformers, which can be often achieved by the specially-designed attention blocks. We hope our paper could help to better understand the mechanism for designing robust DNNs. Our model weights can be found at http://robust.art.

摘要: 对抗性训练已经被证明是防御对抗性例子的最有效的补救方法之一，然而它经常在看不见的测试对手上遭受巨大的健壮性泛化差距，被认为是对抗性健壮性泛化问题。尽管对相反的健壮性泛化有了初步的理解，但从体系结构的角度来看却知之甚少。为了弥补这一差距，本文首次系统地研究了逆稳性泛化与建筑设计之间的关系。特别是，我们在ImageNette和CIFAR-10数据集上全面评估了20种最具代表性的经过对手训练的体系结构，以应对多个p范数对手攻击。基于大量的实验，我们发现，在一致的设置下，Vision Transformers(如PVT，CoAtNet)往往能产生更好的对抗健壮性泛化，而CNN往往对特定攻击过于适应，无法对多个对手泛化。为了更好地理解其背后的本质，我们通过Rademacher复杂性的镜头进行了理论分析。我们揭示的事实是，较高的权重稀疏性显著有助于更好的逆境稳健的变形金刚泛化，这通常可以通过特殊设计的注意块来实现。我们希望我们的论文能够帮助我们更好地理解设计健壮DNN的机制。我们的模型重量可以在http://robust.art.上找到



## **50. Machine Learning Security in Industry: A Quantitative Survey**

机器学习在工业中的安全性：一项定量调查 cs.LG

Accepted at TIFS, version with more detailed appendix containing more  detailed statistical results. 17 pages, 6 tables and 4 figures

**SubmitDate**: 2023-03-10    [abs](http://arxiv.org/abs/2207.05164v2) [paper-pdf](http://arxiv.org/pdf/2207.05164v2)

**Authors**: Kathrin Grosse, Lukas Bieringer, Tarek Richard Besold, Battista Biggio, Katharina Krombholz

**Abstract**: Despite the large body of academic work on machine learning security, little is known about the occurrence of attacks on machine learning systems in the wild. In this paper, we report on a quantitative study with 139 industrial practitioners. We analyze attack occurrence and concern and evaluate statistical hypotheses on factors influencing threat perception and exposure. Our results shed light on real-world attacks on deployed machine learning. On the organizational level, while we find no predictors for threat exposure in our sample, the amount of implement defenses depends on exposure to threats or expected likelihood to become a target. We also provide a detailed analysis of practitioners' replies on the relevance of individual machine learning attacks, unveiling complex concerns like unreliable decision making, business information leakage, and bias introduction into models. Finally, we find that on the individual level, prior knowledge about machine learning security influences threat perception. Our work paves the way for more research about adversarial machine learning in practice, but yields also insights for regulation and auditing.

摘要: 尽管有大量关于机器学习安全的学术工作，但人们对野外发生的针对机器学习系统的攻击知之甚少。在本文中，我们报告了一项对139名工业从业者的定量研究。我们分析了攻击的发生和关注，并对影响威胁感知和暴露的因素进行了统计假设评估。我们的结果揭示了对部署的机器学习的真实世界攻击。在组织层面上，虽然我们在样本中没有发现威胁暴露的预测因素，但实施防御的数量取决于威胁暴露或成为目标的预期可能性。我们还提供了对从业者对单个机器学习攻击相关性的回复的详细分析，揭示了不可靠的决策、商业信息泄露和模型中的偏见引入等复杂问题。最后，我们发现在个体层面上，关于机器学习安全的先验知识会影响威胁感知。我们的工作为在实践中对对抗性机器学习进行更多的研究铺平了道路，但也为监管和审计提供了见解。



