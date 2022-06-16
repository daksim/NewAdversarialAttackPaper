# Latest Adversarial Attack Papers
**update at 2022-06-17 06:31:33**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Learn to Adapt: Robust Drift Detection in Security Domain**

学会适应：安全域中的稳健漂移检测 cs.CR

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2206.07581v1)

**Authors**: Aditya Kuppa, Nhien-An Le-Khac

**Abstracts**: Deploying robust machine learning models has to account for concept drifts arising due to the dynamically changing and non-stationary nature of data. Addressing drifts is particularly imperative in the security domain due to the ever-evolving threat landscape and lack of sufficiently labeled training data at the deployment time leading to performance degradation. Recently proposed concept drift detection methods in literature tackle this problem by identifying the changes in feature/data distributions and periodically retraining the models to learn new concepts. While these types of strategies should absolutely be conducted when possible, they are not robust towards attacker-induced drifts and suffer from a delay in detecting new attacks. We aim to address these shortcomings in this work. we propose a robust drift detector that not only identifies drifted samples but also discovers new classes as they arrive in an on-line fashion. We evaluate the proposed method with two security-relevant data sets -- network intrusion data set released in 2018 and APT Command and Control dataset combined with web categorization data. Our evaluation shows that our drifting detection method is not only highly accurate but also robust towards adversarial drifts and discovers new classes from drifted samples.

摘要: 部署健壮的机器学习模型必须考虑到由于数据的动态变化和非静态性质而产生的概念漂移。在安全领域，解决漂移问题尤为迫切，因为威胁形势不断演变，而且在部署时缺乏标记充分的训练数据，从而导致性能下降。最近文献中提出的概念漂移检测方法通过识别特征/数据分布的变化并周期性地重新训练模型来学习新概念来解决这一问题。虽然这些类型的策略绝对应该在可能的情况下进行，但它们对攻击者引发的漂移并不健壮，并且在检测新攻击方面存在延迟。我们的目标是解决这项工作中的这些缺点。我们提出了一种稳健的漂移检测器，它不仅识别漂移的样本，而且当它们以在线方式到达时发现新的类别。我们用2018年发布的网络入侵数据集和APT指挥控制数据集结合网络分类数据对该方法进行了评估。我们的评估表明，我们的漂移检测方法不仅具有很高的准确率，而且对敌意漂移具有很强的鲁棒性，并从漂移样本中发现新的类别。



## **2. Creating a Secure Underlay for the Internet**

创建Internet的安全参考底图 cs.NI

Usenix Security 2022

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2206.06879v2)

**Authors**: Henry Birge-Lee, Joel Wanner, Grace Cimaszewski, Jonghoon Kwon, Liang Wang, Francois Wirz, Prateek Mittal, Adrian Perrig, Yixin Sun

**Abstracts**: Adversaries can exploit inter-domain routing vulnerabilities to intercept communication and compromise the security of critical Internet applications. Meanwhile the deployment of secure routing solutions such as Border Gateway Protocol Security (BGPsec) and Scalability, Control and Isolation On Next-generation networks (SCION) are still limited. How can we leverage emerging secure routing backbones and extend their security properties to the broader Internet?   We design and deploy an architecture to bootstrap secure routing. Our key insight is to abstract the secure routing backbone as a virtual Autonomous System (AS), called Secure Backbone AS (SBAS). While SBAS appears as one AS to the Internet, it is a federated network where routes are exchanged between participants using a secure backbone. SBAS makes BGP announcements for its customers' IP prefixes at multiple locations (referred to as Points of Presence or PoPs) allowing traffic from non-participating hosts to be routed to a nearby SBAS PoP (where it is then routed over the secure backbone to the true prefix owner). In this manner, we are the first to integrate a federated secure non-BGP routing backbone with the BGP-speaking Internet.   We present a real-world deployment of our architecture that uses SCIONLab to emulate the secure backbone and the PEERING framework to make BGP announcements to the Internet. A combination of real-world attacks and Internet-scale simulations shows that SBAS substantially reduces the threat of routing attacks. Finally, we survey network operators to better understand optimal governance and incentive models.

摘要: 攻击者可以利用域间路由漏洞来拦截通信并危及关键Internet应用程序的安全。同时，边界网关协议安全(BGPSEC)和下一代网络的可扩展性、控制和隔离(SCION)等安全路由解决方案的部署仍然有限。我们如何利用新兴的安全路由主干并将其安全属性扩展到更广泛的互联网？我们设计并部署了一个用于引导安全路由的体系结构。我们的主要见解是将安全路由主干抽象为一个虚拟自治系统(AS)，称为安全主干AS(SBAS)。虽然SBAS在Internet上看起来像一个整体，但它是一个联合网络，参与者之间使用安全的主干交换路由。SBAS在多个位置(称为入网点或POP)为其客户的IP前缀发布BGP公告，允许将来自非参与主机的流量路由到附近的SBAS POP(然后通过安全主干将其路由到真正的前缀所有者)。通过这种方式，我们是第一个将联合安全的非BGP路由骨干网与BGP语音互联网集成在一起的公司。我们提供了我们的架构的实际部署，该架构使用SCIONLab模拟安全主干，并使用对等框架向互联网发布BGP公告。现实世界的攻击和互联网规模的模拟相结合表明，SBAS大大降低了路由攻击的威胁。最后，我们对网络运营商进行了调查，以更好地了解最优治理和激励模式。



## **3. TPC: Transformation-Specific Smoothing for Point Cloud Models**

TPC：点云模型的变换特定平滑 cs.CV

Accepted as a conference paper at ICML 2022

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2201.12733v3)

**Authors**: Wenda Chu, Linyi Li, Bo Li

**Abstracts**: Point cloud models with neural network architectures have achieved great success and have been widely used in safety-critical applications, such as Lidar-based recognition systems in autonomous vehicles. However, such models are shown vulnerable to adversarial attacks which aim to apply stealthy semantic transformations such as rotation and tapering to mislead model predictions. In this paper, we propose a transformation-specific smoothing framework TPC, which provides tight and scalable robustness guarantees for point cloud models against semantic transformation attacks. We first categorize common 3D transformations into three categories: additive (e.g., shearing), composable (e.g., rotation), and indirectly composable (e.g., tapering), and we present generic robustness certification strategies for all categories respectively. We then specify unique certification protocols for a range of specific semantic transformations and their compositions. Extensive experiments on several common 3D transformations show that TPC significantly outperforms the state of the art. For example, our framework boosts the certified accuracy against twisting transformation along z-axis (within 20$^\circ$) from 20.3$\%$ to 83.8$\%$. Codes and models are available at https://github.com/Qianhewu/Point-Cloud-Smoothing.

摘要: 神经网络结构的点云模型已经取得了巨大的成功，并在安全关键应用中得到了广泛的应用，例如自动车辆中基于激光雷达的识别系统。然而，这类模型被证明容易受到敌意攻击，这些攻击旨在应用诸如旋转和缩减等隐蔽的语义转换来误导模型预测。在本文中，我们提出了一个特定于变换的平滑框架TPC，它为点云模型抵抗语义变换攻击提供了紧凑和可伸缩的健壮性保证。我们首先将常见的3D变换分为三类：加性变换(如剪切)、可合成变换(如旋转变换)和间接合成变换(如锥化变换)，并分别针对所有类别提出了通用的健壮性认证策略。然后，我们为一系列特定的语义转换及其组合指定唯一的认证协议。对几种常见的3D变换进行的大量实验表明，TPC的性能明显优于最先进的技术。例如，我们的框架将z轴(在20$^\cic$内)抗扭曲变换的认证精度从20.3$\$提高到83.8$\$。有关代码和型号，请访问https://github.com/Qianhewu/Point-Cloud-Smoothing.



## **4. Towards Understanding Adversarial Robustness of Optical Flow Networks**

理解光流网络的对抗健壮性 cs.CV

CVPR 2022

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2103.16255v3)

**Authors**: Simon Schrodi, Tonmoy Saikia, Thomas Brox

**Abstracts**: Recent work demonstrated the lack of robustness of optical flow networks to physical patch-based adversarial attacks. The possibility to physically attack a basic component of automotive systems is a reason for serious concerns. In this paper, we analyze the cause of the problem and show that the lack of robustness is rooted in the classical aperture problem of optical flow estimation in combination with bad choices in the details of the network architecture. We show how these mistakes can be rectified in order to make optical flow networks robust to physical patch-based attacks. Additionally, we take a look at global white-box attacks in the scope of optical flow. We find that targeted white-box attacks can be crafted to bias flow estimation models towards any desired output, but this requires access to the input images and model weights. However, in the case of universal attacks, we find that optical flow networks are robust. Code is available at https://github.com/lmb-freiburg/understanding_flow_robustness.

摘要: 最近的研究表明，光流网络对基于物理补丁的敌意攻击缺乏健壮性。对汽车系统的基本部件进行物理攻击的可能性是一个令人严重担忧的原因。在本文中，我们分析了问题的原因，并指出健壮性的缺乏根源于经典的光流估计孔径问题以及在网络体系结构细节上的糟糕选择。我们展示了如何纠正这些错误，以使光流网络对基于物理补丁的攻击具有健壮性。此外，我们还在光流的范围内研究了全局白盒攻击。我们发现，可以精心设计有针对性的白盒攻击，以使流量估计模型偏向任何期望的输出，但这需要访问输入图像和模型权重。然而，在普遍攻击的情况下，我们发现光流网络是健壮的。代码可在https://github.com/lmb-freiburg/understanding_flow_robustness.上找到



## **5. Hardening DNNs against Transfer Attacks during Network Compression using Greedy Adversarial Pruning**

基于贪婪对抗性剪枝的DNN在网络压缩过程中抗传输攻击的强化 cs.LG

4 pages, 2 figures

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2206.07406v1)

**Authors**: Jonah O'Brien Weiss, Tiago Alves, Sandip Kundu

**Abstracts**: The prevalence and success of Deep Neural Network (DNN) applications in recent years have motivated research on DNN compression, such as pruning and quantization. These techniques accelerate model inference, reduce power consumption, and reduce the size and complexity of the hardware necessary to run DNNs, all with little to no loss in accuracy. However, since DNNs are vulnerable to adversarial inputs, it is important to consider the relationship between compression and adversarial robustness. In this work, we investigate the adversarial robustness of models produced by several irregular pruning schemes and by 8-bit quantization. Additionally, while conventional pruning removes the least important parameters in a DNN, we investigate the effect of an unconventional pruning method: removing the most important model parameters based on the gradient on adversarial inputs. We call this method Greedy Adversarial Pruning (GAP) and we find that this pruning method results in models that are resistant to transfer attacks from their uncompressed counterparts.

摘要: 近年来，深度神经网络(DNN)的广泛应用和成功应用促进了对DNN压缩的研究，如剪枝和量化。这些技术加快了模型推断，降低了功耗，并降低了运行DNN所需的硬件的大小和复杂性，所有这些都几乎不会损失精度。然而，由于DNN容易受到敌意输入的影响，因此重要的是考虑压缩和敌意稳健性之间的关系。在这项工作中，我们研究了几种非规则剪枝方案和8位量化产生的模型的对抗稳健性。此外，虽然传统的剪枝方法去除了DNN中最不重要的参数，但我们研究了一种非传统剪枝方法的效果：基于对抗性输入的梯度来去除最重要的模型参数。我们将这种方法称为贪婪对抗性剪枝(GAP)，我们发现这种剪枝方法产生的模型能够抵抗来自未压缩对手的攻击。



## **6. Morphence-2.0: Evasion-Resilient Moving Target Defense Powered by Out-of-Distribution Detection**

Morphence-2.0：基于分布外检测的抗躲避移动目标防御 cs.CR

13 pages, 6 figures, 2 tables. arXiv admin note: substantial text  overlap with arXiv:2108.13952

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2206.07321v1)

**Authors**: Abderrahmen Amich, Ata Kaboudi, Birhanu Eshete

**Abstracts**: Evasion attacks against machine learning models often succeed via iterative probing of a fixed target model, whereby an attack that succeeds once will succeed repeatedly. One promising approach to counter this threat is making a model a moving target against adversarial inputs. To this end, we introduce Morphence-2.0, a scalable moving target defense (MTD) powered by out-of-distribution (OOD) detection to defend against adversarial examples. By regularly moving the decision function of a model, Morphence-2.0 makes it significantly challenging for repeated or correlated attacks to succeed. Morphence-2.0 deploys a pool of models generated from a base model in a manner that introduces sufficient randomness when it responds to prediction queries. Via OOD detection, Morphence-2.0 is equipped with a scheduling approach that assigns adversarial examples to robust decision functions and benign samples to an undefended accurate models. To ensure repeated or correlated attacks fail, the deployed pool of models automatically expires after a query budget is reached and the model pool is seamlessly replaced by a new model pool generated in advance. We evaluate Morphence-2.0 on two benchmark image classification datasets (MNIST and CIFAR10) against 4 reference attacks (3 white-box and 1 black-box). Morphence-2.0 consistently outperforms prior defenses while preserving accuracy on clean data and reducing attack transferability. We also show that, when powered by OOD detection, Morphence-2.0 is able to precisely make an input-based movement of the model's decision function that leads to higher prediction accuracy on both adversarial and benign queries.

摘要: 针对机器学习模型的逃避攻击通常通过迭代探测固定目标模型而成功，从而一次成功的攻击将重复成功。对抗这一威胁的一个有希望的方法是使模型成为对抗对手输入的移动目标。为此，我们引入了Morphence-2.0，这是一种基于分布外(OOD)检测的可扩展移动目标防御(MTD)，用于防御恶意示例。通过定期移动模型的决策功能，Morphence-2.0使重复或相关攻击的成功具有极大的挑战性。Morphence-2.0部署了从基本模型生成的模型池，其方式在响应预测查询时引入了足够的随机性。通过OOD检测，Morphence-2.0配备了一种调度方法，将对抗性样本分配给稳健的决策函数，将良性样本分配给无防御的准确模型。为了确保重复或相关攻击失败，部署的模型池在达到查询预算后自动到期，并由预先生成的新模型池无缝替换。我们在两个基准图像分类数据集(MNIST和CIFAR10)上测试了Morphence-2.0在4种参考攻击(3个白盒和1个黑盒)下的性能。Morphence-2.0始终优于以前的防御系统，同时保持干净数据的准确性，并降低攻击的可转移性。我们还表明，在OOD检测的支持下，Morphence-2.0能够精确地对模型的决策函数进行基于输入的移动，从而对敌意查询和良性查询都能产生更高的预测精度。



## **7. Autoregressive Perturbations for Data Poisoning**

数据中毒的自回归摄动 cs.LG

22 pages, 13 figures. Code available at  https://github.com/psandovalsegura/autoregressive-poisoning

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2206.03693v2)

**Authors**: Pedro Sandoval-Segura, Vasu Singla, Jonas Geiping, Micah Goldblum, Tom Goldstein, David W. Jacobs

**Abstracts**: The prevalence of data scraping from social media as a means to obtain datasets has led to growing concerns regarding unauthorized use of data. Data poisoning attacks have been proposed as a bulwark against scraping, as they make data "unlearnable" by adding small, imperceptible perturbations. Unfortunately, existing methods require knowledge of both the target architecture and the complete dataset so that a surrogate network can be trained, the parameters of which are used to generate the attack. In this work, we introduce autoregressive (AR) poisoning, a method that can generate poisoned data without access to the broader dataset. The proposed AR perturbations are generic, can be applied across different datasets, and can poison different architectures. Compared to existing unlearnable methods, our AR poisons are more resistant against common defenses such as adversarial training and strong data augmentations. Our analysis further provides insight into what makes an effective data poison.

摘要: 从社交媒体上窃取数据作为获取数据集的一种手段的盛行，导致对未经授权使用数据的担忧日益加剧。数据中毒攻击被认为是防止抓取的堡垒，因为它们通过添加微小的、不可察觉的干扰而使数据“无法学习”。不幸的是，现有方法需要目标体系结构和完整数据集的知识，以便可以训练代理网络，其参数用于生成攻击。在这项工作中，我们引入了自回归(AR)中毒，这是一种在不访问更广泛的数据集的情况下生成有毒数据的方法。所提出的AR扰动是通用的，可以应用于不同的数据集，并且可能毒害不同的体系结构。与现有的无法学习的方法相比，我们的AR毒药对常见的防御措施更具抵抗力，例如对抗性训练和强大的数据增强。我们的分析进一步提供了对有效数据毒害的原因的洞察。



## **8. Fast and Reliable Evaluation of Adversarial Robustness with Minimum-Margin Attack**

利用最小差额攻击快速可靠地评估对手的健壮性 cs.LG

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2206.07314v1)

**Authors**: Ruize Gao, Jiongxiao Wang, Kaiwen Zhou, Feng Liu, Binghui Xie, Gang Niu, Bo Han, James Cheng

**Abstracts**: The AutoAttack (AA) has been the most reliable method to evaluate adversarial robustness when considerable computational resources are available. However, the high computational cost (e.g., 100 times more than that of the project gradient descent attack) makes AA infeasible for practitioners with limited computational resources, and also hinders applications of AA in the adversarial training (AT). In this paper, we propose a novel method, minimum-margin (MM) attack, to fast and reliably evaluate adversarial robustness. Compared with AA, our method achieves comparable performance but only costs 3% of the computational time in extensive experiments. The reliability of our method lies in that we evaluate the quality of adversarial examples using the margin between two targets that can precisely identify the most adversarial example. The computational efficiency of our method lies in an effective Sequential TArget Ranking Selection (STARS) method, ensuring that the cost of the MM attack is independent of the number of classes. The MM attack opens a new way for evaluating adversarial robustness and provides a feasible and reliable way to generate high-quality adversarial examples in AT.

摘要: 当有大量的计算资源可用时，AutoAttack(AA)一直是评估对手健壮性的最可靠的方法。然而，较高的计算代价(例如，是项目梯度下降攻击的100倍)使得AA对于计算资源有限的实践者来说是不可行的，也阻碍了AA在对抗性训练(AT)中的应用。本文提出了一种新的快速可靠地评估对手健壮性的方法--最小差值攻击。在大量的实验中，我们的方法取得了与AA相当的性能，但只需要3%的计算时间。我们方法的可靠性在于，我们使用两个目标之间的差值来评估对抗性实例的质量，从而可以准确地识别最具对抗性的实例。该方法的计算效率在于采用了一种有效的顺序目标排序选择(STAR)方法，保证了MM攻击的代价与类别数无关。MM攻击为评估对手的健壮性开辟了一条新的途径，并为自动测试中生成高质量的对手实例提供了一种可行和可靠的方法。



## **9. Human Eyes Inspired Recurrent Neural Networks are More Robust Against Adversarial Noises**

人眼启发的递归神经网络对对抗性噪声的更强鲁棒性 cs.CV

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2206.07282v1)

**Authors**: Minkyu Choi, Yizhen Zhang, Kuan Han, Xiaokai Wang, Zhongming Liu

**Abstracts**: Compared to human vision, computer vision based on convolutional neural networks (CNN) are more vulnerable to adversarial noises. This difference is likely attributable to how the eyes sample visual input and how the brain processes retinal samples through its dorsal and ventral visual pathways, which are under-explored for computer vision. Inspired by the brain, we design recurrent neural networks, including an input sampler that mimics the human retina, a dorsal network that guides where to look next, and a ventral network that represents the retinal samples. Taking these modules together, the models learn to take multiple glances at an image, attend to a salient part at each glance, and accumulate the representation over time to recognize the image. We test such models for their robustness against a varying level of adversarial noises with a special focus on the effect of different input sampling strategies. Our findings suggest that retinal foveation and sampling renders a model more robust against adversarial noises, and the model may correct itself from an attack when it is given a longer time to take more glances at an image. In conclusion, robust visual recognition can benefit from the combined use of three brain-inspired mechanisms: retinal transformation, attention guided eye movement, and recurrent processing, as opposed to feedforward-only CNNs.

摘要: 与人类视觉相比，基于卷积神经网络的计算机视觉更容易受到对抗性噪声的影响。这种差异可能归因于眼睛如何对视觉输入进行采样，以及大脑如何通过其背侧和腹侧视觉路径处理视网膜样本，而计算机视觉对这些路径的探索不足。受大脑的启发，我们设计了递归神经网络，包括模拟人类视网膜的输入采样器，指导下一步往哪里看的背部网络，以及代表视网膜样本的腹面网络。把这些模块放在一起，这些模型学会了对一张图像进行多次扫视，在每一次扫视中注意到显著的部分，并随着时间的推移积累表示法来识别图像。我们测试了这些模型对不同水平的对抗性噪声的稳健性，并特别关注不同输入采样策略的效果。我们的发现表明，视网膜凹陷和采样使模型对敌对噪声更加稳健，并且当给予更长的时间来更多地瞥一眼图像时，该模型可能会在攻击中自我纠正。总而言之，强健的视觉识别可以受益于三种大脑启发机制的组合使用：视网膜转换、注意力引导的眼球运动和递归处理，而不是仅限前馈的CNN。



## **10. Can Linear Programs Have Adversarial Examples? A Causal Perspective**

线性规划能有对抗性的例子吗？因果关系视角 cs.LG

Main paper: 9 pages, References: 2 page, Supplement: 2 pages. Main  paper: 2 figures, 3 tables, Supplement: 1 figure, 1 table

**SubmitDate**: 2022-06-14    [paper-pdf](http://arxiv.org/pdf/2105.12697v5)

**Authors**: Matej Zečević, Devendra Singh Dhami, Kristian Kersting

**Abstracts**: The recent years have been marked by extended research on adversarial attacks, especially on deep neural networks. With this work we intend on posing and investigating the question of whether the phenomenon might be more general in nature, that is, adversarial-style attacks outside classification. Specifically, we investigate optimization problems starting with Linear Programs (LPs). We start off by demonstrating the shortcoming of a naive mapping between the formalism of adversarial examples and LPs, to then reveal how we can provide the missing piece -- intriguingly, through the Pearlian notion of Causality. Characteristically, we show the direct influence of the Structural Causal Model (SCM) onto the subsequent LP optimization, which ultimately exposes a notion of confounding in LPs (inherited by said SCM) that allows for adversarial-style attacks. We provide both the general proof formally alongside existential proofs of such intriguing LP-parameterizations based on SCM for three combinatorial problems, namely Linear Assignment, Shortest Path and a real world problem of energy systems.

摘要: 近年来，对抗性攻击，特别是深度神经网络的研究不断深入。通过这项工作，我们打算提出和调查这一现象是否更具一般性的问题，即分类外的对抗性攻击。具体地说，我们从线性规划(LP)开始研究优化问题。我们从展示对抗性例子的形式主义和有限合伙人之间的天真映射的缺陷开始，然后揭示我们如何能够提供缺失的部分--有趣的是，通过菲莱尔的因果关系概念。在特征上，我们展示了结构因果模型(SCM)对随后的LP优化的直接影响，这最终暴露了LP中的混淆概念(由所述SCM继承)，允许对抗性风格的攻击。对于三个组合问题，即线性分配问题、最短路问题和一个真实世界的能量系统问题，我们给出了基于SCM的这类有趣的LP-参数化的形式证明和存在性证明。



## **11. Defending Observation Attacks in Deep Reinforcement Learning via Detection and Denoising**

基于检测和去噪的深度强化学习防御观测攻击 cs.LG

**SubmitDate**: 2022-06-14    [paper-pdf](http://arxiv.org/pdf/2206.07188v1)

**Authors**: Zikang Xiong, Joe Eappen, He Zhu, Suresh Jagannathan

**Abstracts**: Neural network policies trained using Deep Reinforcement Learning (DRL) are well-known to be susceptible to adversarial attacks. In this paper, we consider attacks manifesting as perturbations in the observation space managed by the external environment. These attacks have been shown to downgrade policy performance significantly. We focus our attention on well-trained deterministic and stochastic neural network policies in the context of continuous control benchmarks subject to four well-studied observation space adversarial attacks. To defend against these attacks, we propose a novel defense strategy using a detect-and-denoise schema. Unlike previous adversarial training approaches that sample data in adversarial scenarios, our solution does not require sampling data in an environment under attack, thereby greatly reducing risk during training. Detailed experimental results show that our technique is comparable with state-of-the-art adversarial training approaches.

摘要: 众所周知，使用深度强化学习(DRL)训练的神经网络策略容易受到对手攻击。在本文中，我们考虑的攻击表现为外部环境管理的观测空间中的扰动。事实证明，这些攻击会显著降低策略性能。在连续控制基准的背景下，我们将注意力集中在训练有素的确定性和随机神经网络策略上，这些基准受到四个经过充分研究的观测空间对抗性攻击。为了防御这些攻击，我们提出了一种新的防御策略，使用检测和去噪方案。与以往在对抗性场景中对数据进行采样的对抗性训练方法不同，我们的解决方案不需要在受到攻击的环境中对数据进行采样，从而大大降低了训练期间的风险。详细的实验结果表明，我们的技术可以与目前最先进的对抗性训练方法相媲美。



## **12. Proximal Splitting Adversarial Attacks for Semantic Segmentation**

基于近邻分裂的对抗性语义分割攻击 cs.LG

Code available at:  https://github.com/jeromerony/alma_prox_segmentation

**SubmitDate**: 2022-06-14    [paper-pdf](http://arxiv.org/pdf/2206.07179v1)

**Authors**: Jérôme Rony, Jean-Christophe Pesquet, Ismail Ben Ayed

**Abstracts**: Classification has been the focal point of research on adversarial attacks, but only a few works investigate methods suited to denser prediction tasks, such as semantic segmentation. The methods proposed in these works do not accurately solve the adversarial segmentation problem and, therefore, are overoptimistic in terms of size of the perturbations required to fool models. Here, we propose a white-box attack for these models based on a proximal splitting to produce adversarial perturbations with much smaller $\ell_1$, $\ell_2$, or $\ell_\infty$ norms. Our attack can handle large numbers of constraints within a nonconvex minimization framework via an Augmented Lagrangian approach, coupled with adaptive constraint scaling and masking strategies. We demonstrate that our attack significantly outperforms previously proposed ones, as well as classification attacks that we adapted for segmentation, providing a first comprehensive benchmark for this dense task. Our results push current limits concerning robustness evaluations in segmentation tasks.

摘要: 分类一直是对抗性攻击研究的重点，但很少有人研究适合于密集预测任务的方法，如语义分割。这些工作中提出的方法不能准确地解决对抗性分割问题，因此在愚弄模型所需的扰动大小方面过于乐观。在这里，我们对这些模型提出了一种基于近邻分裂的白盒攻击，以产生具有更小的$\ell_1$、$\ell_2$或$\ell_\inty$范数的对抗性扰动。我们的攻击可以通过增广拉格朗日方法，结合自适应约束缩放和掩蔽策略，在非凸极小化框架内处理大量约束。我们证明，我们的攻击显著优于之前提出的攻击，以及我们为分割而采用的分类攻击，为这一密集任务提供了第一个全面的基准。我们的结果突破了目前关于分割任务中稳健性评估的限制。



## **13. FENCE: Feasible Evasion Attacks on Neural Networks in Constrained Environments**

Figure：受限环境下神经网络的可行逃避攻击 cs.CR

35 pages

**SubmitDate**: 2022-06-14    [paper-pdf](http://arxiv.org/pdf/1909.10480v4)

**Authors**: Alesia Chernikova, Alina Oprea

**Abstracts**: As advances in Deep Neural Networks (DNNs) demonstrate unprecedented levels of performance in many critical applications, their vulnerability to attacks is still an open question. We consider evasion attacks at testing time against Deep Learning in constrained environments, in which dependencies between features need to be satisfied. These situations may arise naturally in tabular data or may be the result of feature engineering in specific application domains, such as threat detection in cyber security. We propose a general iterative gradient-based framework called FENCE for crafting evasion attacks that take into consideration the specifics of constrained domains and application requirements. We apply it against Feed-Forward Neural Networks trained for two cyber security applications: network traffic botnet classification and malicious domain classification, to generate feasible adversarial examples. We extensively evaluate the success rate and performance of our attacks, compare their improvement over several baselines, and analyze factors that impact the attack success rate, including the optimization objective and the data imbalance. We show that with minimal effort (e.g., generating 12 additional network connections), an attacker can change the model's prediction from the Malicious class to Benign and evade the classifier. We show that models trained on datasets with higher imbalance are more vulnerable to our FENCE attacks. Finally, we demonstrate the potential of performing adversarial training in constrained domains to increase the model resilience against these evasion attacks.

摘要: 随着深度神经网络(DNN)在许多关键应用中表现出前所未有的性能水平，它们对攻击的脆弱性仍然是一个悬而未决的问题。我们考虑了受限环境中针对深度学习的测试时的逃避攻击，在这种环境中，需要满足特征之间的依赖关系。这些情况可能自然地出现在表格数据中，或者可能是特定应用领域的功能工程的结果，例如网络安全中的威胁检测。我们提出了一种通用的基于迭代梯度的框架，称为FEARK，用于定制规避攻击，该框架考虑了受约束的域和应用程序需求的特殊性。我们将其应用于针对两种网络安全应用：网络流量僵尸网络分类和恶意域分类而训练的前馈神经网络，以生成可行的对抗性实例。我们广泛评估我们的攻击成功率和性能，比较它们在几个基线上的改善，并分析影响攻击成功率的因素，包括优化目标和数据不平衡。我们表明，攻击者只需很少的努力(例如，生成12个额外的网络连接)，就可以将模型的预测从恶意类更改为良性类，并逃避分类器。我们表明，在失衡程度较高的数据集上训练的模型更容易受到我们的围栏攻击。最后，我们展示了在受限领域中执行对抗性训练的潜力，以提高模型对这些逃避攻击的弹性。



## **14. A Layered Reference Model for Penetration Testing with Reinforcement Learning and Attack Graphs**

基于强化学习和攻击图的层次化渗透测试参考模型 cs.CR

**SubmitDate**: 2022-06-14    [paper-pdf](http://arxiv.org/pdf/2206.06934v1)

**Authors**: Tyler Cody

**Abstracts**: This paper considers key challenges to using reinforcement learning (RL) with attack graphs to automate penetration testing in real-world applications from a systems perspective. RL approaches to automated penetration testing are actively being developed, but there is no consensus view on the representation of computer networks with which RL should be interacting. Moreover, there are significant open challenges to how those representations can be grounded to the real networks where RL solution methods are applied. This paper elaborates on representation and grounding using topic challenges of interacting with real networks in real-time, emulating realistic adversary behavior, and handling unstable, evolving networks. These challenges are both practical and mathematical, and they directly concern the reliability and dependability of penetration testing systems. This paper proposes a layered reference model to help organize related research and engineering efforts. The presented layered reference model contrasts traditional models of attack graph workflows because it is not scoped to a sequential, feed-forward generation and analysis process, but to broader aspects of lifecycle and continuous deployment. Researchers and practitioners can use the presented layered reference model as a first-principles outline to help orient the systems engineering of their penetration testing systems.

摘要: 本文从系统的角度考虑了在实际应用中使用强化学习(RL)和攻击图来自动化渗透测试的关键挑战。自动渗透测试的RL方法正在积极开发中，但对于RL应该与之交互的计算机网络的表示方式，还没有一致的看法。此外，如何将这些表示与应用RL解决方案方法的实际网络相结合，还存在着重大的开放挑战。本文详细阐述了如何使用与真实网络实时交互、模拟现实对手行为以及处理不稳定、不断变化的网络的主题挑战来表示和扎根。这些挑战既是实际的，也是数学的，它们直接关系到渗透测试系统的可靠性和可靠性。本文提出了一个分层参考模型，以帮助组织相关的研究和工程工作。提出的分层参考模型与传统的攻击图工作流模型进行了对比，因为它的范围不限于顺序的前馈生成和分析过程，而是生命周期和持续部署的更广泛方面。研究人员和实践者可以使用提出的分层参考模型作为第一原则大纲，以帮助定位他们的渗透测试系统的系统工程。



## **15. When adversarial attacks become interpretable counterfactual explanations**

当对抗性攻击成为可解释的反事实解释时 cs.AI

**SubmitDate**: 2022-06-14    [paper-pdf](http://arxiv.org/pdf/2206.06854v1)

**Authors**: Mathieu Serrurier, Franck Mamalet, Thomas Fel, Louis Béthune, Thibaut Boissin

**Abstracts**: We argue that, when learning a 1-Lipschitz neural network with the dual loss of an optimal transportation problem, the gradient of the model is both the direction of the transportation plan and the direction to the closest adversarial attack. Traveling along the gradient to the decision boundary is no more an adversarial attack but becomes a counterfactual explanation, explicitly transporting from one class to the other. Through extensive experiments on XAI metrics, we find that the simple saliency map method, applied on such networks, becomes a reliable explanation, and outperforms the state-of-the-art explanation approaches on unconstrained models. The proposed networks were already known to be certifiably robust, and we prove that they are also explainable with a fast and simple method.

摘要: 我们认为，当学习具有最优运输问题的对偶损失的1-Lipschitz神经网络时，模型的梯度既是运输计划的方向，也是最接近对手攻击的方向。沿着梯度行进到决策边界不再是一种对抗性攻击，而是一种反事实的解释，明确地从一个类别传递到另一个类别。通过对XAI度量的大量实验，我们发现应用于此类网络的简单显著图方法成为一种可靠的解释方法，并且在无约束模型上的性能优于最新的解释方法。所提出的网络已经被认为是可证明的健壮性，并且我们证明了它们也是可以用快速和简单的方法解释的。



## **16. Exploring Adversarial Attacks and Defenses in Vision Transformers trained with DINO**

探索与恐龙一起训练的视觉变形金刚的对抗性攻击和防御 cs.CV

6 pages workshop paper accepted at AdvML Frontiers (ICML 2022)

**SubmitDate**: 2022-06-14    [paper-pdf](http://arxiv.org/pdf/2206.06761v1)

**Authors**: Javier Rando, Nasib Naimi, Thomas Baumann, Max Mathys

**Abstracts**: This work conducts the first analysis on the robustness against adversarial attacks on self-supervised Vision Transformers trained using DINO. First, we evaluate whether features learned through self-supervision are more robust to adversarial attacks than those emerging from supervised learning. Then, we present properties arising for attacks in the latent space. Finally, we evaluate whether three well-known defense strategies can increase adversarial robustness in downstream tasks by only fine-tuning the classification head to provide robustness even in view of limited compute resources. These defense strategies are: Adversarial Training, Ensemble Adversarial Training and Ensemble of Specialized Networks.

摘要: 本文首次对使用Dino训练的自监督视觉转换器的抗敌意攻击能力进行了分析。首先，我们评估通过自我监督学习的特征是否比通过监督学习获得的特征对对手攻击更健壮。然后，我们给出了潜在空间中攻击产生的性质。最后，我们评估了三种著名的防御策略是否能够在下游任务中通过微调分类头来提高对手的健壮性，即使在计算资源有限的情况下也是如此。这些防御策略是：对抗性训练、系列性对抗性训练和专业网络系列化。



## **17. Adversarial Vulnerability of Randomized Ensembles**

随机化系综的对抗性脆弱性 cs.LG

Published as a conference paper in ICML 2022

**SubmitDate**: 2022-06-14    [paper-pdf](http://arxiv.org/pdf/2206.06737v1)

**Authors**: Hassan Dbouk, Naresh R. Shanbhag

**Abstracts**: Despite the tremendous success of deep neural networks across various tasks, their vulnerability to imperceptible adversarial perturbations has hindered their deployment in the real world. Recently, works on randomized ensembles have empirically demonstrated significant improvements in adversarial robustness over standard adversarially trained (AT) models with minimal computational overhead, making them a promising solution for safety-critical resource-constrained applications. However, this impressive performance raises the question: Are these robustness gains provided by randomized ensembles real? In this work we address this question both theoretically and empirically. We first establish theoretically that commonly employed robustness evaluation methods such as adaptive PGD provide a false sense of security in this setting. Subsequently, we propose a theoretically-sound and efficient adversarial attack algorithm (ARC) capable of compromising random ensembles even in cases where adaptive PGD fails to do so. We conduct comprehensive experiments across a variety of network architectures, training schemes, datasets, and norms to support our claims, and empirically establish that randomized ensembles are in fact more vulnerable to $\ell_p$-bounded adversarial perturbations than even standard AT models. Our code can be found at https://github.com/hsndbk4/ARC.

摘要: 尽管深度神经网络在各种任务中取得了巨大的成功，但它们在难以察觉的对抗性扰动下的脆弱性阻碍了它们在现实世界中的部署。最近，对随机集成的研究已经经验证明，与标准的对抗性训练(AT)模型相比，在计算开销最小的情况下，它们在对抗性稳健性方面有显著的改进，使得它们成为安全关键的资源受限应用的一种有前途的解决方案。然而，这种令人印象深刻的表现提出了一个问题：这些由随机组合提供的稳健性收益是真的吗？在这项工作中，我们从理论和经验两个方面解决了这个问题。我们首先从理论上证明，通常使用的健壮性评估方法，如自适应PGD，在这种情况下提供了一种错误的安全感。随后，我们提出了一种理论上合理和高效的对抗攻击算法(ARC)，即使在自适应PGD不能做到这一点的情况下，该算法也能够危害随机集成。我们在各种网络架构、训练方案、数据集和规范上进行了全面的实验，以支持我们的主张，并经验证明，随机集成实际上比标准AT模型更容易受到$\ell_p$-有界的对手扰动的影响。我们的代码可以在https://github.com/hsndbk4/ARC.上找到



## **18. Downlink Power Allocation in Massive MIMO via Deep Learning: Adversarial Attacks and Training**

基于深度学习的大规模MIMO下行链路功率分配：对抗性攻击和训练 cs.LG

13 pages, 14 figures, published in IEEE Transactions on Cognitive  Communications and Networking

**SubmitDate**: 2022-06-14    [paper-pdf](http://arxiv.org/pdf/2206.06592v1)

**Authors**: B. R. Manoj, Meysam Sadeghi, Erik G. Larsson

**Abstracts**: The successful emergence of deep learning (DL) in wireless system applications has raised concerns about new security-related challenges. One such security challenge is adversarial attacks. Although there has been much work demonstrating the susceptibility of DL-based classification tasks to adversarial attacks, regression-based problems in the context of a wireless system have not been studied so far from an attack perspective. The aim of this paper is twofold: (i) we consider a regression problem in a wireless setting and show that adversarial attacks can break the DL-based approach and (ii) we analyze the effectiveness of adversarial training as a defensive technique in adversarial settings and show that the robustness of DL-based wireless system against attacks improves significantly. Specifically, the wireless application considered in this paper is the DL-based power allocation in the downlink of a multicell massive multi-input-multi-output system, where the goal of the attack is to yield an infeasible solution by the DL model. We extend the gradient-based adversarial attacks: fast gradient sign method (FGSM), momentum iterative FGSM, and projected gradient descent method to analyze the susceptibility of the considered wireless application with and without adversarial training. We analyze the deep neural network (DNN) models performance against these attacks, where the adversarial perturbations are crafted using both the white-box and black-box attacks.

摘要: 深度学习在无线系统应用中的成功出现引发了人们对新的安全相关挑战的担忧。其中一个安全挑战是对抗性攻击。虽然已经有很多工作证明了基于DL的分类任务对敌意攻击的敏感性，但到目前为止，还没有人从攻击的角度来研究无线系统中基于回归的问题。本文的目的有两个：(I)我们考虑了无线环境下的回归问题，证明了敌方攻击可以打破基于DL的方法；(Ii)我们分析了对抗性训练作为一种防御技术在对抗性环境中的有效性，表明基于DL的无线系统对抗攻击的健壮性显著提高。具体地说，本文所考虑的无线应用是多小区大规模多输入多输出系统下行链路中基于下行链路的功率分配问题，攻击的目标是通过下行链路模型得到一个不可行解。我们扩展了基于梯度的对抗攻击方法：快速梯度符号方法(FGSM)、动量迭代FGSM方法和投影梯度下降方法，以分析考虑的无线应用在有和没有对抗性训练的情况下的敏感度。我们分析了深度神经网络(DNN)模型对这些攻击的性能，其中敌意扰动是使用白盒攻击和黑盒攻击来构造的。



## **19. Person Re-identification Method Based on Color Attack and Joint Defence**

基于颜色攻击和联合防御的身份识别方法 cs.CV

Accepted by CVPR2022 Workshops  (https://openaccess.thecvf.com/content/CVPR2022W/HCIS/html/Gong_Person_Re-Identification_Method_Based_on_Color_Attack_and_Joint_Defence_CVPRW_2022_paper.html)

**SubmitDate**: 2022-06-14    [paper-pdf](http://arxiv.org/pdf/2111.09571v4)

**Authors**: Yunpeng Gong, Liqing Huang, Lifei Chen

**Abstracts**: The main challenges of ReID is the intra-class variations caused by color deviation under different camera conditions. Simultaneously, we find that most of the existing adversarial metric attacks are realized by interfering with the color characteristics of the sample. Based on this observation, we first propose a local transformation attack (LTA) based on color variation. It uses more obvious color variation to randomly disturb the color of the retrieved image, rather than adding random noise. Experiments show that the performance of the proposed LTA method is better than the advanced attack methods. Furthermore, considering that the contour feature is the main factor of the robustness of adversarial training, and the color feature will directly affect the success rate of attack. Therefore, we further propose joint adversarial defense (JAD) method, which include proactive defense and passive defense. Proactive defense fuse multi-modality images to enhance the contour feature and color feature, and considers local homomorphic transformation to solve the over-fitting problem. Passive defense exploits the invariance of contour feature during image scaling to mitigate the adversarial disturbance on contour feature. Finally, a series of experimental results show that the proposed joint adversarial defense method is more competitive than a state-of-the-art methods.

摘要: REID的主要挑战是在不同的摄像头条件下，颜色偏差导致的类内差异。同时，我们发现现有的对抗性度量攻击大多是通过干扰样本的颜色特征来实现的。基于此，我们首先提出了一种基于颜色变化的局部变换攻击(LTA)。它使用更明显的颜色变化来随机干扰检索到的图像的颜色，而不是添加随机噪声。实验表明，提出的LTA方法的性能优于先进的攻击方法。此外，考虑到轮廓特征是影响对抗训练稳健性的主要因素，而颜色特征将直接影响攻击的成功率。因此，我们进一步提出了联合对抗防御(JAD)方法，包括主动防御和被动防御。主动防御融合多模式图像以增强轮廓特征和颜色特征，并考虑局部同态变换来解决过拟合问题。被动防御利用图像缩放过程中轮廓特征的不变性来缓解轮廓特征受到的对抗性干扰。最后，一系列的实验结果表明，本文提出的联合对抗防御方法比现有的方法更具竞争力。



## **20. Towards Alternative Techniques for Improving Adversarial Robustness: Analysis of Adversarial Training at a Spectrum of Perturbations**

提高对抗性稳健性的替代技术：对抗性训练在扰动频谱中的分析 cs.LG

**SubmitDate**: 2022-06-13    [paper-pdf](http://arxiv.org/pdf/2206.06496v1)

**Authors**: Kaustubh Sridhar, Souradeep Dutta, Ramneet Kaur, James Weimer, Oleg Sokolsky, Insup Lee

**Abstracts**: Adversarial training (AT) and its variants have spearheaded progress in improving neural network robustness to adversarial perturbations and common corruptions in the last few years. Algorithm design of AT and its variants are focused on training models at a specified perturbation strength $\epsilon$ and only using the feedback from the performance of that $\epsilon$-robust model to improve the algorithm. In this work, we focus on models, trained on a spectrum of $\epsilon$ values. We analyze three perspectives: model performance, intermediate feature precision and convolution filter sensitivity. In each, we identify alternative improvements to AT that otherwise wouldn't have been apparent at a single $\epsilon$. Specifically, we find that for a PGD attack at some strength $\delta$, there is an AT model at some slightly larger strength $\epsilon$, but no greater, that generalizes best to it. Hence, we propose overdesigning for robustness where we suggest training models at an $\epsilon$ just above $\delta$. Second, we observe (across various $\epsilon$ values) that robustness is highly sensitive to the precision of intermediate features and particularly those after the first and second layer. Thus, we propose adding a simple quantization to defenses that improves accuracy on seen and unseen adaptive attacks. Third, we analyze convolution filters of each layer of models at increasing $\epsilon$ and notice that those of the first and second layer may be solely responsible for amplifying input perturbations. We present our findings and demonstrate our techniques through experiments with ResNet and WideResNet models on the CIFAR-10 and CIFAR-10-C datasets.

摘要: 在过去的几年里，对抗性训练(AT)及其变体在提高神经网络对对抗性扰动和常见腐败的稳健性方面取得了进展。AT及其变种的算法设计集中于在给定的扰动强度下训练模型，并且只利用该稳健模型的性能反馈来改进算法。在这项工作中，我们将重点放在模型上，并针对$\epsilon$值的谱进行训练。我们从模型性能、中间特征精度和卷积滤波灵敏度三个方面进行了分析。在每一项中，我们都确定了AT的其他改进，否则这些改进在单个$\epsilon$中不会很明显。具体地说，我们发现对于某个强度的PGD攻击，存在一个强度略大但不大于的AT模型，这是对它的最好概括。因此，我们建议过度设计健壮性，建议训练模型的$\epsilon$略高于$\Delta$。其次，我们观察到(在不同的$\epsilon$值上)稳健性对中间特征的精度高度敏感，特别是第一层和第二层之后的那些特征。因此，我们建议在防御中增加一个简单的量化，以提高对可见和不可见自适应攻击的准确性。第三，我们分析了每一层模型的卷积滤波，注意到第一层和第二层的卷积滤波可能是唯一负责放大输入扰动的。我们介绍了我们的发现，并通过在CIFAR-10和CIFAR-10-C数据集上使用ResNet和WideResNet模型的实验来演示我们的技术。



## **21. Distributed Adversarial Training to Robustify Deep Neural Networks at Scale**

深度神经网络规模化的分布式对抗性训练 cs.LG

**SubmitDate**: 2022-06-13    [paper-pdf](http://arxiv.org/pdf/2206.06257v1)

**Authors**: Gaoyuan Zhang, Songtao Lu, Yihua Zhang, Xiangyi Chen, Pin-Yu Chen, Quanfu Fan, Lee Martie, Lior Horesh, Mingyi Hong, Sijia Liu

**Abstracts**: Current deep neural networks (DNNs) are vulnerable to adversarial attacks, where adversarial perturbations to the inputs can change or manipulate classification. To defend against such attacks, an effective and popular approach, known as adversarial training (AT), has been shown to mitigate the negative impact of adversarial attacks by virtue of a min-max robust training method. While effective, it remains unclear whether it can successfully be adapted to the distributed learning context. The power of distributed optimization over multiple machines enables us to scale up robust training over large models and datasets. Spurred by that, we propose distributed adversarial training (DAT), a large-batch adversarial training framework implemented over multiple machines. We show that DAT is general, which supports training over labeled and unlabeled data, multiple types of attack generation methods, and gradient compression operations favored for distributed optimization. Theoretically, we provide, under standard conditions in the optimization theory, the convergence rate of DAT to the first-order stationary points in general non-convex settings. Empirically, we demonstrate that DAT either matches or outperforms state-of-the-art robust accuracies and achieves a graceful training speedup (e.g., on ResNet-50 under ImageNet). Codes are available at https://github.com/dat-2022/dat.

摘要: 当前的深度神经网络(DNN)很容易受到敌意攻击，对输入的敌意扰动可以改变或操纵分类。为了防御这种攻击，一种被称为对抗性训练(AT)的有效和流行的方法已经被证明通过最小-最大稳健训练方法来减轻对抗性攻击的负面影响。虽然有效，但它是否能成功地适应分布式学习环境仍不清楚。在多台机器上进行分布式优化的能力使我们能够在大型模型和数据集上扩大健壮的训练。受此启发，我们提出了分布式对抗训练(DAT)，这是一种在多台机器上实现的大批量对抗训练框架。我们证明DAT是通用的，它支持对有标签和无标签数据的训练，支持多种类型的攻击生成方法，以及有利于分布式优化的梯度压缩操作。理论上，在最优化理论的标准条件下，我们给出了一般非凸集上DAT收敛到一阶驻点的收敛速度。在实验上，我们证明了DAT匹配或超过了最先进的稳健精度，并实现了优雅的训练加速比(例如，在ImageNet下的ResNet-50上)。有关代码，请访问https://github.com/dat-2022/dat.



## **22. Adversarial Models Towards Data Availability and Integrity of Distributed State Estimation for Industrial IoT-Based Smart Grid**

基于工业物联网的智能电网分布式状态估计的数据可用性和完整性对抗模型 cs.CR

11 pages (DC), Journal manuscript

**SubmitDate**: 2022-06-13    [paper-pdf](http://arxiv.org/pdf/2206.06027v1)

**Authors**: Haftu Tasew Reda, Abdun Mahmood, Adnan Anwar, Naveen Chilamkurti

**Abstracts**: Security issue of distributed state estimation (DSE) is an important prospect for the rapidly growing smart grid ecosystem. Any coordinated cyberattack targeting the distributed system of state estimators can cause unrestrained estimation errors and can lead to a myriad of security risks, including failure of power system operation. This article explores the security threats of a smart grid arising from the exploitation of DSE vulnerabilities. To this aim, novel adversarial strategies based on two-stage data availability and integrity attacks are proposed towards a distributed industrial Internet of Things-based smart grid. The former's attack goal is to prevent boundary data exchange among distributed control centers, while the latter's attack goal is to inject a falsified data to cause local and global system unobservability. The proposed framework is evaluated on IEEE standard 14-bus system and benchmarked against the state-of-the-art research. Experimental results show that the proposed two-stage cyberattack results in an estimated error of approximately 34.74% compared to an error of the order of 10^-3 under normal operating conditions.

摘要: 分布式状态估计(DSE)的安全问题是快速发展的智能电网生态系统的一个重要前景。任何针对分布式状态估计器系统的协同网络攻击都可能导致不受限制的估计误差，并可能导致无数安全风险，包括电力系统运行故障。本文探讨了利用DSE漏洞对智能电网造成的安全威胁。为此，针对基于分布式工业物联网的智能电网，提出了基于数据可用性和完整性两阶段攻击的新型对抗策略。前者的攻击目标是防止分布式控制中心之间的边界数据交换，而后者的攻击目标是注入伪造的数据，导致局部和全局系统不可观测。提出的框架在IEEE标准14节点系统上进行了评估，并以最新研究成果为基准进行了基准测试。实验结果表明，与正常运行条件下10^-3量级的估计误差相比，提出的两阶段网络攻击的估计误差约为34.74%。



## **23. Universal, transferable and targeted adversarial attacks**

普遍的、可转移的和有针对性的对抗性攻击 cs.LG

**SubmitDate**: 2022-06-13    [paper-pdf](http://arxiv.org/pdf/1908.11332v4)

**Authors**: Junde Wu, Rao Fu

**Abstracts**: Deep Neural Networks have been found vulnerable re-cently. A kind of well-designed inputs, which called adver-sarial examples, can lead the networks to make incorrectpredictions. Depending on the different scenarios, goalsand capabilities, the difficulties of the attacks are different.For example, a targeted attack is more difficult than a non-targeted attack, a universal attack is more difficult than anon-universal attack, a transferable attack is more difficultthan a nontransferable one. The question is: Is there existan attack that can meet all these requirements? In this pa-per, we answer this question by producing a kind of attacksunder these conditions. We learn a universal mapping tomap the sources to the adversarial examples. These exam-ples can fool classification networks to classify all of theminto one targeted class, and also have strong transferability.Our code is released at: xxxxx.

摘要: 深度神经网络最近被发现是脆弱的。一种设计良好的输入，也就是所谓的反常例子，会导致网络做出错误的预测。根据场景、目标和能力的不同，攻击的难度也不同，例如，定向攻击比非定向攻击更难，通用攻击比非通用攻击更难，可转移攻击比不可转移攻击更难。问题是：是否存在能够满足所有这些要求的攻击？在本文中，我们通过在这些条件下产生一种攻击来回答这个问题。我们学习了一种普遍的映射，将来源映射到对抗性的例子。这些例子可以骗过分类网络将它们归类到一个目标类中，并且具有很强的可移植性。我们的代码发布在：xxxxx。



## **24. Revisiting and Advancing Fast Adversarial Training Through The Lens of Bi-Level Optimization**

用双层优化镜头重温和推进快速对抗性训练 cs.LG

**SubmitDate**: 2022-06-13    [paper-pdf](http://arxiv.org/pdf/2112.12376v5)

**Authors**: Yihua Zhang, Guanhua Zhang, Prashant Khanduri, Mingyi Hong, Shiyu Chang, Sijia Liu

**Abstracts**: Adversarial training (AT) is a widely recognized defense mechanism to gain the robustness of deep neural networks against adversarial attacks. It is built on min-max optimization (MMO), where the minimizer (i.e., defender) seeks a robust model to minimize the worst-case training loss in the presence of adversarial examples crafted by the maximizer (i.e., attacker). However, the conventional MMO method makes AT hard to scale. Thus, Fast-AT (Wong et al., 2020) and other recent algorithms attempt to simplify MMO by replacing its maximization step with the single gradient sign-based attack generation step. Although easy to implement, Fast-AT lacks theoretical guarantees, and its empirical performance is unsatisfactory due to the issue of robust catastrophic overfitting when training with strong adversaries. In this paper, we advance Fast-AT from the fresh perspective of bi-level optimization (BLO). We first show that the commonly-used Fast-AT is equivalent to using a stochastic gradient algorithm to solve a linearized BLO problem involving a sign operation. However, the discrete nature of the sign operation makes it difficult to understand the algorithm performance. Inspired by BLO, we design and analyze a new set of robust training algorithms termed Fast Bi-level AT (Fast-BAT), which effectively defends sign-based projected gradient descent (PGD) attacks without using any gradient sign method or explicit robust regularization. In practice, we show our method yields substantial robustness improvements over baselines across multiple models and datasets. Codes are available at https://github.com/OPTML-Group/Fast-BAT.

摘要: 对抗训练(AT)是一种公认的防御机制，用来增强深度神经网络对对手攻击的健壮性。它建立在最小-最大优化(MMO)的基础上，其中最小化器(即防御者)寻求一个健壮的模型来最小化最坏情况下的训练损失，其中存在由最大化者(即攻击者)制作的对抗性例子。然而，传统的MMO方法使AT难以规模化。因此，Fast-AT(Wong等人，2020)和其他最近的算法试图通过将其最大化步骤替换为基于单一梯度符号的攻击生成步骤来简化MMO。尽管FAST-AT易于实现，但它缺乏理论保证，而且在与强大对手进行训练时，由于存在稳健的灾难性过拟合问题，其经验性能也不尽如人意。本文从双层优化(BLO)的新视角提出了FAST-AT算法。我们首先证明了常用的Fast-AT算法等价于使用随机梯度算法来求解包含符号运算的线性化BLO问题。然而，符号运算的离散性使得很难理解算法的性能。受BLO的启发，我们设计并分析了一组新的稳健训练算法--快速双水平AT(Fast-BAT)，该算法无需使用任何梯度符号方法或显式稳健正则化，即可有效地防御基于符号的投影梯度下降(PGD)攻击。在实践中，我们证明了我们的方法在多个模型和数据集的基线上产生了实质性的稳健性改进。有关代码，请访问https://github.com/OPTML-Group/Fast-BAT.



## **25. Deploying Convolutional Networks on Untrusted Platforms Using 2D Holographic Reduced Representations**

使用2D全息简化表示在不可信平台上部署卷积网络 cs.LG

To appear in the Proceedings of the 39 th International Conference on  Machine Learning, Baltimore, Maryland, USA, PMLR 162, 2022

**SubmitDate**: 2022-06-13    [paper-pdf](http://arxiv.org/pdf/2206.05893v1)

**Authors**: Mohammad Mahmudul Alam, Edward Raff, Tim Oates, James Holt

**Abstracts**: Due to the computational cost of running inference for a neural network, the need to deploy the inferential steps on a third party's compute environment or hardware is common. If the third party is not fully trusted, it is desirable to obfuscate the nature of the inputs and outputs, so that the third party can not easily determine what specific task is being performed. Provably secure protocols for leveraging an untrusted party exist but are too computational demanding to run in practice. We instead explore a different strategy of fast, heuristic security that we call Connectionist Symbolic Pseudo Secrets. By leveraging Holographic Reduced Representations (HRR), we create a neural network with a pseudo-encryption style defense that empirically shows robustness to attack, even under threat models that unrealistically favor the adversary.

摘要: 由于运行神经网络推理的计算成本，通常需要在第三方的计算环境或硬件上部署推理步骤。如果第三方不是完全可信的，则需要混淆输入和输出的性质，以便第三方不能容易地确定正在执行什么特定任务。存在用于利用不可信方的可证明安全的协议，但其计算要求太高而不能在实践中运行。相反，我们探索了一种不同的快速启发式安全策略，我们称之为连接主义符号伪秘密。通过利用全息简化表示(HRR)，我们创建了一个具有伪加密样式防御的神经网络，该防御经验地显示出对攻击的鲁棒性，即使在不切实际地有利于对手的威胁模型下也是如此。



## **26. InBiaseD: Inductive Bias Distillation to Improve Generalization and Robustness through Shape-awareness**

InBiaseD：通过形状感知提高泛化和健壮性的归纳偏差蒸馏 cs.CV

Accepted at 1st Conference on Lifelong Learning Agents (CoLLAs 2022)

**SubmitDate**: 2022-06-12    [paper-pdf](http://arxiv.org/pdf/2206.05846v1)

**Authors**: Shruthi Gowda, Bahram Zonooz, Elahe Arani

**Abstracts**: Humans rely less on spurious correlations and trivial cues, such as texture, compared to deep neural networks which lead to better generalization and robustness. It can be attributed to the prior knowledge or the high-level cognitive inductive bias present in the brain. Therefore, introducing meaningful inductive bias to neural networks can help learn more generic and high-level representations and alleviate some of the shortcomings. We propose InBiaseD to distill inductive bias and bring shape-awareness to the neural networks. Our method includes a bias alignment objective that enforces the networks to learn more generic representations that are less vulnerable to unintended cues in the data which results in improved generalization performance. InBiaseD is less susceptible to shortcut learning and also exhibits lower texture bias. The better representations also aid in improving robustness to adversarial attacks and we hence plugin InBiaseD seamlessly into the existing adversarial training schemes to show a better trade-off between generalization and robustness.

摘要: 与深度神经网络相比，人类对虚假相关性和琐碎线索(如纹理)的依赖较少，深层神经网络导致更好的泛化和健壮性。这可以归因于大脑中存在的先验知识或高级认知归纳偏见。因此，将有意义的归纳偏差引入神经网络可以帮助学习更通用和更高级别的表示，并缓解一些缺点。我们提出了InBiaseD来提取归纳偏差，并将形状感知引入神经网络。我们的方法包括一个偏差对齐目标，该目标强制网络学习更一般的表示，这些表示不太容易受到数据中意外提示的影响，从而提高泛化性能。InBiaseD不太容易受到捷径学习的影响，也表现出较低的纹理偏向。更好的表示也有助于提高对对抗性攻击的健壮性，因此我们将InBiaseD无缝地插入到现有的对抗性训练方案中，以在泛化和健壮性之间进行更好的权衡。



## **27. Consistent Attack: Universal Adversarial Perturbation on Embodied Vision Navigation**

一致攻击：具身视觉导航的普遍对抗性扰动 cs.LG

**SubmitDate**: 2022-06-12    [paper-pdf](http://arxiv.org/pdf/2206.05751v1)

**Authors**: You Qiaoben, Chengyang Ying, Xinning Zhou, Hang Su, Jun Zhu, Bo Zhang

**Abstracts**: Embodied agents in vision navigation coupled with deep neural networks have attracted increasing attention. However, deep neural networks are vulnerable to malicious adversarial noises, which may potentially cause catastrophic failures in Embodied Vision Navigation. Among these adversarial noises, universal adversarial perturbations (UAP), i.e., the image-agnostic perturbation applied on each frame received by the agent, are more critical for Embodied Vision Navigation since they are computation-efficient and application-practical during the attack. However, existing UAP methods do not consider the system dynamics of Embodied Vision Navigation. For extending UAP in the sequential decision setting, we formulate the disturbed environment under the universal noise $\delta$, as a $\delta$-disturbed Markov Decision Process ($\delta$-MDP). Based on the formulation, we analyze the properties of $\delta$-MDP and propose two novel Consistent Attack methods for attacking Embodied agents, which first consider the dynamic of the MDP by estimating the disturbed Q function and the disturbed distribution. In spite of victim models, our Consistent Attack can cause a significant drop in the performance for the Goalpoint task in habitat. Extensive experimental results indicate that there exist potential risks for applying Embodied Vision Navigation methods to the real world.

摘要: 视觉导航中的具身智能体与深度神经网络相结合，越来越受到人们的关注。然而，深度神经网络很容易受到恶意的对抗性噪声的影响，这可能会导致体现视觉导航中的灾难性故障。在这些对抗性噪声中，通用对抗性扰动(UAP)，即应用于代理接收到的每一帧上的与图像无关的扰动，对于嵌入视觉导航来说更为关键，因为它们在攻击过程中具有计算效率和应用实用性。然而，现有的UAP方法没有考虑体现视觉导航的系统动力学。为了在序贯决策环境中扩展UAP，我们将普遍噪声$\Delta$下的扰动环境描述为$\Delta$-扰动马尔可夫决策过程($\Delta$-MDP)。在此基础上，分析了$Delta$-MDP的性质，提出了两种新的一致性攻击方法，首先通过估计扰动Q函数和扰动分布来考虑MDP的动态性。尽管有受害者模型，但我们的持续攻击会导致栖息地Goalpoint任务的性能显著下降。大量的实验结果表明，将具身视觉导航方法应用于现实世界存在潜在的风险。



## **28. Darknet Traffic Classification and Adversarial Attacks**

暗网流量分类与对抗性攻击 cs.LG

**SubmitDate**: 2022-06-12    [paper-pdf](http://arxiv.org/pdf/2206.06371v1)

**Authors**: Nhien Rust-Nguyen, Mark Stamp

**Abstracts**: The anonymous nature of darknets is commonly exploited for illegal activities. Previous research has employed machine learning and deep learning techniques to automate the detection of darknet traffic in an attempt to block these criminal activities. This research aims to improve darknet traffic detection by assessing Support Vector Machines (SVM), Random Forest (RF), Convolutional Neural Networks (CNN), and Auxiliary-Classifier Generative Adversarial Networks (AC-GAN) for classification of such traffic and the underlying application types. We find that our RF model outperforms the state-of-the-art machine learning techniques used in prior work with the CIC-Darknet2020 dataset. To evaluate the robustness of our RF classifier, we obfuscate select application type classes to simulate realistic adversarial attack scenarios. We demonstrate that our best-performing classifier can be defeated by such attacks, and we consider ways to deal with such adversarial attacks.

摘要: 飞镖的匿名性通常被用于非法活动。之前的研究已经使用机器学习和深度学习技术来自动检测暗网流量，试图阻止这些犯罪活动。这项研究旨在通过评估支持向量机(SVM)、随机森林(RF)、卷积神经网络(CNN)和辅助分类器生成对抗网络(AC-GAN)来改进暗网络流量检测，以分类此类流量和潜在的应用类型。我们发现，我们的RF模型的性能优于之前使用CIC-Darknet2020数据集进行的最先进的机器学习技术。为了评估我们的RF分类器的健壮性，我们混淆了选择的应用类型类来模拟真实的对抗性攻击场景。我们证明了我们性能最好的分类器可以被这样的攻击击败，并且我们考虑了处理这种对抗性攻击的方法。



## **29. Security of Machine Learning-Based Anomaly Detection in Cyber Physical Systems**

基于机器学习的网络物理系统异常检测的安全性 cs.DC

**SubmitDate**: 2022-06-12    [paper-pdf](http://arxiv.org/pdf/2206.05678v1)

**Authors**: Zahra Jadidi, Shantanu Pal, Nithesh Nayak K, Arawinkumaar Selvakkumar, Chih-Chia Chang, Maedeh Beheshti, Alireza Jolfaei

**Abstracts**: In this study, we focus on the impact of adversarial attacks on deep learning-based anomaly detection in CPS networks and implement a mitigation approach against the attack by retraining models using adversarial samples. We use the Bot-IoT and Modbus IoT datasets to represent the two CPS networks. We train deep learning models and generate adversarial samples using these datasets. These datasets are captured from IoT and Industrial IoT (IIoT) networks. They both provide samples of normal and attack activities. The deep learning model trained with these datasets showed high accuracy in detecting attacks. An Artificial Neural Network (ANN) is adopted with one input layer, four intermediate layers, and one output layer. The output layer has two nodes representing the binary classification results. To generate adversarial samples for the experiment, we used a function called the `fast_gradient_method' from the Cleverhans library. The experimental result demonstrates the influence of FGSM adversarial samples on the accuracy of the predictions and proves the effectiveness of using the retrained model to defend against adversarial attacks.

摘要: 在本研究中，我们重点研究了对抗性攻击对CPS网络中基于深度学习的异常检测的影响，并通过使用对抗性样本重新训练模型来实现针对攻击的缓解方法。我们使用Bot-IoT和MODBUS IoT数据集来表示两个CPS网络。我们训练深度学习模型，并使用这些数据集生成对抗性样本。这些数据集是从物联网和工业物联网(IIoT)网络捕获的。它们都提供了正常和攻击活动的样本。利用这些数据集训练的深度学习模型在检测攻击方面表现出了较高的准确率。采用一个输入层、四个中间层和一个输出层的人工神经网络。输出层具有表示二进制分类结果的两个节点。为了为实验生成对抗性样本，我们使用了Cleverhans库中的一个名为`Fast_Gendent_Method‘的函数。实验结果证明了FGSM敌方样本对预测精度的影响，证明了使用重新训练的模型来防御敌方攻击的有效性。



## **30. An Efficient Method for Sample Adversarial Perturbations against Nonlinear Support Vector Machines**

一种针对非线性支持向量机的样本对抗扰动的有效方法 cs.LG

**SubmitDate**: 2022-06-12    [paper-pdf](http://arxiv.org/pdf/2206.05664v1)

**Authors**: Wen Su, Qingna Li

**Abstracts**: Adversarial perturbations have drawn great attentions in various machine learning models. In this paper, we investigate the sample adversarial perturbations for nonlinear support vector machines (SVMs). Due to the implicit form of the nonlinear functions mapping data to the feature space, it is difficult to obtain the explicit form of the adversarial perturbations. By exploring the special property of nonlinear SVMs, we transform the optimization problem of attacking nonlinear SVMs into a nonlinear KKT system. Such a system can be solved by various numerical methods. Numerical results show that our method is efficient in computing adversarial perturbations.

摘要: 对抗性扰动在各种机器学习模型中引起了极大的关注。本文研究了非线性支持向量机的样本对抗扰动。由于将数据映射到特征空间的非线性函数的隐式形式，很难得到对抗性扰动的显式形式。通过探索非线性支持向量机的特殊性质，我们将攻击非线性支持向量机的优化问题转化为非线性KKT系统。这样的系统可以用各种数值方法来求解。数值结果表明，该方法在计算对抗性扰动时是有效的。



## **31. Reward Poisoning Attacks on Offline Multi-Agent Reinforcement Learning**

基于离线多智能体强化学习的奖励毒化攻击 cs.LG

**SubmitDate**: 2022-06-11    [paper-pdf](http://arxiv.org/pdf/2206.01888v2)

**Authors**: Young Wu, Jeremey McMahan, Xiaojin Zhu, Qiaomin Xie

**Abstracts**: We expose the danger of reward poisoning in offline multi-agent reinforcement learning (MARL), whereby an attacker can modify the reward vectors to different learners in an offline data set while incurring a poisoning cost. Based on the poisoned data set, all rational learners using some confidence-bound-based MARL algorithm will infer that a target policy - chosen by the attacker and not necessarily a solution concept originally - is the Markov perfect dominant strategy equilibrium for the underlying Markov Game, hence they will adopt this potentially damaging target policy in the future. We characterize the exact conditions under which the attacker can install a target policy. We further show how the attacker can formulate a linear program to minimize its poisoning cost. Our work shows the need for robust MARL against adversarial attacks.

摘要: 我们揭示了离线多智能体强化学习(MAIL)中奖励中毒的危险，即攻击者可以在离线数据集中修改不同学习者的奖励向量，同时招致中毒成本。基于中毒数据集，所有使用基于置信度的Marl算法的理性学习者都会推断，由攻击者选择的目标策略-最初不一定是解的概念-是潜在马尔可夫博弈的马尔可夫完美支配策略均衡，因此他们将在未来采用这种潜在的破坏性目标策略。我们描述了攻击者可以安装目标策略的确切条件。我们进一步展示了攻击者如何制定一个线性规划来最小化其中毒成本。我们的工作表明，需要健壮的Marl来抵御对手攻击。



## **32. SemAttack: Natural Textual Attacks via Different Semantic Spaces**

SemAttack：基于不同语义空间的自然文本攻击 cs.CL

Published at Findings of NAACL 2022

**SubmitDate**: 2022-06-11    [paper-pdf](http://arxiv.org/pdf/2205.01287v3)

**Authors**: Boxin Wang, Chejian Xu, Xiangyu Liu, Yu Cheng, Bo Li

**Abstracts**: Recent studies show that pre-trained language models (LMs) are vulnerable to textual adversarial attacks. However, existing attack methods either suffer from low attack success rates or fail to search efficiently in the exponentially large perturbation space. We propose an efficient and effective framework SemAttack to generate natural adversarial text by constructing different semantic perturbation functions. In particular, SemAttack optimizes the generated perturbations constrained on generic semantic spaces, including typo space, knowledge space (e.g., WordNet), contextualized semantic space (e.g., the embedding space of BERT clusterings), or the combination of these spaces. Thus, the generated adversarial texts are more semantically close to the original inputs. Extensive experiments reveal that state-of-the-art (SOTA) large-scale LMs (e.g., DeBERTa-v2) and defense strategies (e.g., FreeLB) are still vulnerable to SemAttack. We further demonstrate that SemAttack is general and able to generate natural adversarial texts for different languages (e.g., English and Chinese) with high attack success rates. Human evaluations also confirm that our generated adversarial texts are natural and barely affect human performance. Our code is publicly available at https://github.com/AI-secure/SemAttack.

摘要: 最近的研究表明，预先训练的语言模型(LMS)容易受到文本攻击。然而，现有的攻击方法要么攻击成功率低，要么不能在指数级的大扰动空间中进行有效的搜索。通过构造不同的语义扰动函数，提出了一种高效的自然对抗性文本生成框架SemAttack。具体地，SemAttack优化约束在通用语义空间上的所生成的扰动，所述通用语义空间包括打字错误空间、知识空间(例如，WordNet)、上下文化的语义空间(例如，BERT聚类的嵌入空间)或这些空间的组合。因此，生成的对抗性文本在语义上更接近原始输入。大量实验表明，最先进的大规模LMS(如DeBERTa-v2)和防御策略(如FreeLB)仍然容易受到SemAttack的攻击。我们进一步证明了SemAttack是通用的，能够生成不同语言(如英语和汉语)的自然对抗性文本，具有很高的攻击成功率。人类评估还证实，我们生成的对抗性文本是自然的，几乎不会影响人类的表现。我们的代码在https://github.com/AI-secure/SemAttack.上公开提供



## **33. How does Heterophily Impact Robustness of Graph Neural Networks? Theoretical Connections and Practical Implications**

异构性如何影响图神经网络的健壮性？理论联系和实践意义 cs.LG

Accepted to KDD 2022; complete version with full appendix; 21 pages,  2 figures

**SubmitDate**: 2022-06-11    [paper-pdf](http://arxiv.org/pdf/2106.07767v3)

**Authors**: Jiong Zhu, Junchen Jin, Donald Loveland, Michael T. Schaub, Danai Koutra

**Abstracts**: We bridge two research directions on graph neural networks (GNNs), by formalizing the relation between heterophily of node labels (i.e., connected nodes tend to have dissimilar labels) and the robustness of GNNs to adversarial attacks. Our theoretical and empirical analyses show that for homophilous graph data, impactful structural attacks always lead to reduced homophily, while for heterophilous graph data the change in the homophily level depends on the node degrees. These insights have practical implications for defending against attacks on real-world graphs: we deduce that separate aggregators for ego- and neighbor-embeddings, a design principle which has been identified to significantly improve prediction for heterophilous graph data, can also offer increased robustness to GNNs. Our comprehensive experiments show that GNNs merely adopting this design achieve improved empirical and certifiable robustness compared to the best-performing unvaccinated model. Additionally, combining this design with explicit defense mechanisms against adversarial attacks leads to an improved robustness with up to 18.33% performance increase under attacks compared to the best-performing vaccinated model.

摘要: 通过形式化节点标签的异质性(即连接的节点往往具有不同的标签)与图神经网络对对手攻击的健壮性之间的关系，我们在图神经网络(GNN)的两个研究方向之间架起了桥梁。我们的理论和实证分析表明，对于同嗜性的图数据，有效的结构攻击总是导致同质性的降低，而对于异嗜性的图数据，同质性水平的变化取决于节点度。这些见解对防御真实世界图上的攻击具有实际意义：我们推断，针对自我和邻居嵌入的单独聚集器，这一设计原则已被确定为显著改善对异嗜图数据的预测，也可以提高GNN的健壮性。我们的综合实验表明，与性能最好的未接种疫苗模型相比，仅采用这种设计的GNN获得了更好的经验和可证明的稳健性。此外，将此设计与针对对手攻击的显式防御机制相结合，可以提高健壮性，与性能最好的疫苗模型相比，在攻击下的性能最高可提高18.33%。



## **34. Game-Theoretic Neyman-Pearson Detection to Combat Strategic Evasion**

对抗战略规避的博弈论Neyman-Pearson检测 cs.CR

**SubmitDate**: 2022-06-10    [paper-pdf](http://arxiv.org/pdf/2206.05276v1)

**Authors**: Yinan Hu, Quanyan Zhu

**Abstracts**: The security in networked systems depends greatly on recognizing and identifying adversarial behaviors. Traditional detection methods focus on specific categories of attacks and have become inadequate for increasingly stealthy and deceptive attacks that are designed to bypass detection strategically. This work aims to develop a holistic theory to countermeasure such evasive attacks. We focus on extending a fundamental class of statistical-based detection methods based on Neyman-Pearson's (NP) hypothesis testing formulation. We propose game-theoretic frameworks to capture the conflicting relationship between a strategic evasive attacker and an evasion-aware NP detector. By analyzing both the equilibrium behaviors of the attacker and the NP detector, we characterize their performance using Equilibrium Receiver-Operational-Characteristic (EROC) curves. We show that the evasion-aware NP detectors outperform the passive ones in the way that the former can act strategically against the attacker's behavior and adaptively modify their decision rules based on the received messages. In addition, we extend our framework to a sequential setting where the user sends out identically distributed messages. We corroborate the analytical results with a case study of anomaly detection.

摘要: 网络系统的安全性在很大程度上取决于对敌方行为的识别和识别。传统的检测方法侧重于特定类别的攻击，已不适用于日益隐蔽和欺骗性的攻击，这些攻击旨在从战略上绕过检测。这项工作旨在开发一种整体理论来对抗这种规避攻击。基于Neyman-Pearson(NP)假设检验公式，我们重点扩展了一类基本的基于统计的检测方法。我们提出了博弈论框架来捕捉战略规避攻击者和规避感知NP检测器之间的冲突关系。通过分析攻击者和NP检测器的均衡行为，我们用均衡接收-操作-特征(EROC)曲线来表征它们的性能。我们证明了逃避感知NP检测器的性能优于被动NP检测器，前者可以针对攻击者的行为采取策略性行动，并根据收到的消息自适应地修改其决策规则。此外，我们将我们的框架扩展到顺序设置，在该设置中，用户发送相同分布的消息。我们通过一个异常检测的案例验证了分析结果。



## **35. Blades: A Simulator for Attacks and Defenses in Federated Learning**

Blade：联邦学习中的攻防模拟器 cs.CR

**SubmitDate**: 2022-06-10    [paper-pdf](http://arxiv.org/pdf/2206.05359v1)

**Authors**: Shenghui Li, Li Ju, Tianru Zhang, Edith Ngai, Thiemo Voigt

**Abstracts**: Federated learning enables distributed training across a set of clients, without requiring any of the participants to reveal their private training data to a centralized entity or each other. Due to the nature of decentralized execution, federated learning is vulnerable to attacks from adversarial (Byzantine) clients by modifying the local updates to their desires. Therefore, it is important to develop robust federated learning algorithms that can defend Byzantine clients without losing model convergence and performance. In the study of robustness problems, a simulator can simplify and accelerate the implementation and evaluation of attack and defense strategies. However, there is a lack of open-source simulators to meet such needs. Herein, we present Blades, a scalable, extensible, and easily configurable simulator to assist researchers and developers in efficiently implementing and validating novel strategies against baseline algorithms in robust federated learning. Blades is built upon a versatile distributed framework Ray, making it effortless to parallelize single machine code from a single CPU to multi-core, multi-GPU, or multi-node with minimal configurations. Blades contains built-in implementations of representative attack and defense strategies and provides user-friendly interfaces to easily incorporate new ideas. We maintain the source code and documents at https://github.com/bladesteam/blades.

摘要: 联合学习支持跨一组客户端的分布式培训，而不需要任何参与者向中央实体或彼此透露他们的私人培训数据。由于分散执行的性质，联邦学习通过修改本地更新来满足对手(拜占庭)客户端的需求，很容易受到攻击。因此，开发稳健的联邦学习算法非常重要，它可以在不损失模型收敛和性能的情况下保护拜占庭客户端。在健壮性问题的研究中，模拟器可以简化和加速攻防策略的实施和评估。然而，目前还缺乏满足这种需求的开源模拟器。在这里，我们提出了Blade，一个可伸缩、可扩展和易于配置的模拟器，以帮助研究人员和开发人员有效地实施和验证针对稳健联邦学习中的基线算法的新策略。Blade构建在一个通用的分布式框架Ray上，使得以最低配置将单个机器代码从单CPU并行到多核、多GPU或多节点变得轻而易举。Blade包含典型攻击和防御策略的内置实现，并提供用户友好的界面，以轻松融入新想法。我们在https://github.com/bladesteam/blades.维护源代码和文档



## **36. Hierarchical Federated Learning with Privacy**

带隐私的分层联邦学习 cs.LG

**SubmitDate**: 2022-06-10    [paper-pdf](http://arxiv.org/pdf/2206.05209v1)

**Authors**: Varun Chandrasekaran, Suman Banerjee, Diego Perino, Nicolas Kourtellis

**Abstracts**: Federated learning (FL), where data remains at the federated clients, and where only gradient updates are shared with a central aggregator, was assumed to be private. Recent work demonstrates that adversaries with gradient-level access can mount successful inference and reconstruction attacks. In such settings, differentially private (DP) learning is known to provide resilience. However, approaches used in the status quo (\ie central and local DP) introduce disparate utility vs. privacy trade-offs. In this work, we take the first step towards mitigating such trade-offs through {\em hierarchical FL (HFL)}. We demonstrate that by the introduction of a new intermediary level where calibrated DP noise can be added, better privacy vs. utility trade-offs can be obtained; we term this {\em hierarchical DP (HDP)}. Our experiments with 3 different datasets (commonly used as benchmarks for FL) suggest that HDP produces models as accurate as those obtained using central DP, where noise is added at a central aggregator. Such an approach also provides comparable benefit against inference adversaries as in the local DP case, where noise is added at the federated clients.

摘要: 联合学习(FL)被认为是私有的，其中数据保留在联合客户端，并且仅与中央聚合器共享梯度更新。最近的工作表明，具有梯度访问权限的攻击者可以发起成功的推理和重构攻击。在这种情况下，差分私有(DP)学习已知能够提供弹性。然而，现状中使用的方法(即中央和地方DP)引入了不同的效用与隐私之间的权衡。在这项工作中，我们迈出了第一步，通过{\em分层FL(HFL)}来缓解这种权衡。我们证明，通过引入一个可以添加校准DP噪声的新的中间层，可以获得更好的私密性与公用事业的权衡；我们称之为{\em分层DP(HDP)}。我们用3个不同的数据集(通常用作FL的基准)的实验表明，HDP产生的模型与使用中央DP获得的模型一样准确，其中噪声是在中央聚集器添加的。这种方法还提供了与在本地DP情况下类似的针对推理对手的好处，在本地DP情况下，在联合客户端添加了噪声。



## **37. Localized adversarial artifacts for compressed sensing MRI**

用于压缩传感磁共振成像的局部化对抗性伪影 eess.IV

14 pages, 7 figures

**SubmitDate**: 2022-06-10    [paper-pdf](http://arxiv.org/pdf/2206.05289v1)

**Authors**: Rima Alaifari, Giovanni S. Alberti, Tandri Gauksson

**Abstracts**: As interest in deep neural networks (DNNs) for image reconstruction tasks grows, their reliability has been called into question (Antun et al., 2020; Gottschling et al., 2020). However, recent work has shown that compared to total variation (TV) minimization, they show similar robustness to adversarial noise in terms of $\ell^2$-reconstruction error (Genzel et al., 2022). We consider a different notion of robustness, using the $\ell^\infty$-norm, and argue that localized reconstruction artifacts are a more relevant defect than the $\ell^2$-error. We create adversarial perturbations to undersampled MRI measurements which induce severe localized artifacts in the TV-regularized reconstruction. The same attack method is not as effective against DNN based reconstruction. Finally, we show that this phenomenon is inherent to reconstruction methods for which exact recovery can be guaranteed, as with compressed sensing reconstructions with $\ell^1$- or TV-minimization.

摘要: 随着人们对用于图像重建任务的深度神经网络(DNN)的兴趣与日俱增，其可靠性受到质疑(Antun等人，2020；Gottschling等人，2020)。然而，最近的工作表明，与总变分(TV)最小化相比，它们在重构误差方面表现出类似的对抗性噪声的稳健性(Genzel等人，2022)。我们考虑一种不同的健壮性概念，使用$\ell^\ininty$-范数，并认为局部重建构件是比$\ell^2$-错误更相关的缺陷。我们对欠采样的MRI测量进行对抗性扰动，在TV正则化重建中导致严重的局部化伪影。同样的攻击方法对基于DNN的重构并不有效。最后，我们证明了这种现象是可以保证精确恢复的重建方法所固有的，就像压缩感知重建中的最小化或TV最小化一样。



## **38. SERVFAIL: The Unintended Consequences of Algorithm Agility in DNSSEC**

SERVFAIL：DNSSEC中算法敏捷性的意外后果 cs.CR

Withdrawn on request of one of the persons listed as authors

**SubmitDate**: 2022-06-10    [paper-pdf](http://arxiv.org/pdf/2205.10608v2)

**Authors**: Elias Heftrig, Jean-Pierre Seifert, Haya Shulman, Peter Thomassen, Michael Waidner, Nils Wisiol

**Abstracts**: Cryptographic algorithm agility is an important property for DNSSEC: it allows easy deployment of new algorithms if the existing ones are no longer secure. Significant operational and research efforts are dedicated to pushing the deployment of new algorithms in DNSSEC forward. Recent research shows that DNSSEC is gradually achieving algorithm agility: most DNSSEC supporting resolvers can validate a number of different algorithms and domains are increasingly signed with cryptographically strong ciphers.   In this work we show for the first time that the cryptographic agility in DNSSEC, although critical for making DNS secure with strong cryptography, also introduces a severe vulnerability. We find that under certain conditions, when new algorithms are listed in signed DNS responses, the resolvers do not validate DNSSEC. As a result, domains that deploy new ciphers, risk exposing the validating resolvers to cache poisoning attacks.   We use this to develop DNSSEC-downgrade attacks and show that in some situations these attacks can be launched even by off-path adversaries. We experimentally and ethically evaluate our attacks against popular DNS resolver implementations, public DNS providers, and DNS services used by web clients worldwide. We validate the success of DNSSEC-downgrade attacks by poisoning the resolvers: we inject fake records, in signed domains, into the caches of validating resolvers. We find that major DNS providers, such as Google Public DNS and Cloudflare, as well as 70% of DNS resolvers used by web clients are vulnerable to our attacks.   We trace the factors that led to this situation and provide recommendations.

摘要: 密码算法灵活性是DNSSEC的一个重要属性：如果现有算法不再安全，它允许轻松部署新算法。大量的操作和研究工作致力于推动在DNSSEC中部署新算法。最近的研究表明，DNSSEC正在逐步实现算法灵活性：大多数支持DNSSEC的解析器可以验证一些不同的算法，并且越来越多的域使用密码强密码签名。在这项工作中，我们第一次展示了DNSSEC的密码敏捷性，尽管对于使用强大的密码学来确保DNS安全至关重要，但也引入了一个严重的漏洞。我们发现，在某些条件下，当新算法在签名的DNS响应中列出时，解析器不会验证DNSSEC。因此，部署新密码的域有可能使验证解析器面临缓存中毒攻击。我们利用这一点来开发DNSSEC降级攻击，并表明在某些情况下，这些攻击甚至可以由偏离路径的对手发起。我们从实验和伦理上评估我们对全球Web客户端使用的流行的DNS解析器实现、公共DNS提供商和DNS服务的攻击。我们通过毒化解析器来验证DNSSEC降级攻击的成功：我们在有符号的域中向验证解析器的缓存中注入虚假记录。我们发现，主要的域名服务提供商，如Google Public DNS和Cloudflare，以及网络客户端使用的70%的域名解析程序，都容易受到我们的攻击。我们追踪了导致这种情况的因素并提出了建议。



## **39. Enhancing Clean Label Backdoor Attack with Two-phase Specific Triggers**

使用两阶段特定触发器增强干净标签后门攻击 cs.CR

**SubmitDate**: 2022-06-10    [paper-pdf](http://arxiv.org/pdf/2206.04881v1)

**Authors**: Nan Luo, Yuanzhang Li, Yajie Wang, Shangbo Wu, Yu-an Tan, Quanxin Zhang

**Abstracts**: Backdoor attacks threaten Deep Neural Networks (DNNs). Towards stealthiness, researchers propose clean-label backdoor attacks, which require the adversaries not to alter the labels of the poisoned training datasets. Clean-label settings make the attack more stealthy due to the correct image-label pairs, but some problems still exist: first, traditional methods for poisoning training data are ineffective; second, traditional triggers are not stealthy which are still perceptible. To solve these problems, we propose a two-phase and image-specific triggers generation method to enhance clean-label backdoor attacks. Our methods are (1) powerful: our triggers can both promote the two phases (i.e., the backdoor implantation and activation phase) in backdoor attacks simultaneously; (2) stealthy: our triggers are generated from each image. They are image-specific instead of fixed triggers. Extensive experiments demonstrate that our approach can achieve a fantastic attack success rate~(98.98%) with low poisoning rate~(5%), high stealthiness under many evaluation metrics and is resistant to backdoor defense methods.

摘要: 后门攻击威胁深度神经网络(DNN)。对于隐蔽性，研究人员提出了干净标签的后门攻击，要求对手不更改有毒训练数据集的标签。由于图像-标签对的正确设置，干净标签设置使攻击更加隐蔽，但仍然存在一些问题：第一，传统的毒化训练数据的方法效果不佳；第二，传统的触发器不隐蔽，仍然可以感知。为了解决这些问题，我们提出了一种针对图像的两阶段触发器生成方法来增强干净标签后门攻击。我们的方法是(1)强大：我们的触发器都可以同时推动后门攻击的两个阶段(即后门植入和激活阶段)；(2)隐蔽性：我们的触发器是从每个镜像生成的。它们是特定于图像的INSTEAD而不是固定触发器。大量实验表明，该方法具有攻击成功率98.98%、中毒率低5%、在多种评价指标下隐蔽性高、抵抗后门防御等优点。



## **40. ReFace: Real-time Adversarial Attacks on Face Recognition Systems**

REFACE：对人脸识别系统的实时敌意攻击 cs.CV

**SubmitDate**: 2022-06-09    [paper-pdf](http://arxiv.org/pdf/2206.04783v1)

**Authors**: Shehzeen Hussain, Todd Huster, Chris Mesterharm, Paarth Neekhara, Kevin An, Malhar Jere, Harshvardhan Sikka, Farinaz Koushanfar

**Abstracts**: Deep neural network based face recognition models have been shown to be vulnerable to adversarial examples. However, many of the past attacks require the adversary to solve an input-dependent optimization problem using gradient descent which makes the attack impractical in real-time. These adversarial examples are also tightly coupled to the attacked model and are not as successful in transferring to different models. In this work, we propose ReFace, a real-time, highly-transferable attack on face recognition models based on Adversarial Transformation Networks (ATNs). ATNs model adversarial example generation as a feed-forward neural network. We find that the white-box attack success rate of a pure U-Net ATN falls substantially short of gradient-based attacks like PGD on large face recognition datasets. We therefore propose a new architecture for ATNs that closes this gap while maintaining a 10000x speedup over PGD. Furthermore, we find that at a given perturbation magnitude, our ATN adversarial perturbations are more effective in transferring to new face recognition models than PGD. ReFace attacks can successfully deceive commercial face recognition services in a transfer attack setting and reduce face identification accuracy from 82% to 16.4% for AWS SearchFaces API and Azure face verification accuracy from 91% to 50.1%.

摘要: 基于深度神经网络的人脸识别模型已被证明容易受到敌意例子的影响。然而，过去的许多攻击都需要对手使用梯度下降来解决依赖于输入的优化问题，这使得攻击不能实时进行。这些对抗性的例子也与被攻击的模型紧密耦合，在转移到不同的模型上不是那么成功。在这项工作中，我们提出了REFACE，一种基于对抗性变换网络(ATNS)的人脸识别模型的实时、高可转移性攻击。ATNS模型将对抗性实例生成作为前馈神经网络。我们发现，在大规模人脸识别数据集上，纯U-Net ATN的白盒攻击成功率明显低于PGD等基于梯度的攻击。因此，我们提出了一种新的ATNS体系结构，它可以缩小这一差距，同时保持比PGD高10000倍的加速比。此外，我们发现在给定的扰动幅度下，我们的ATN对抗性扰动比PGD更有效地转移到新的人脸识别模型。ReFace攻击可以在Transfer攻击设置下成功欺骗商业人脸识别服务，并将AWS SearchFaces API和Azure人脸验证准确率从91%降低到50.1%。



## **41. Network insensitivity to parameter noise via adversarial regularization**

基于对抗性正则化的网络对参数噪声不敏感性 cs.LG

**SubmitDate**: 2022-06-09    [paper-pdf](http://arxiv.org/pdf/2106.05009v3)

**Authors**: Julian Büchel, Fynn Faber, Dylan R. Muir

**Abstracts**: Neuromorphic neural network processors, in the form of compute-in-memory crossbar arrays of memristors, or in the form of subthreshold analog and mixed-signal ASICs, promise enormous advantages in compute density and energy efficiency for NN-based ML tasks. However, these technologies are prone to computational non-idealities, due to process variation and intrinsic device physics. This degrades the task performance of networks deployed to the processor, by introducing parameter noise into the deployed model. While it is possible to calibrate each device, or train networks individually for each processor, these approaches are expensive and impractical for commercial deployment. Alternative methods are therefore needed to train networks that are inherently robust against parameter variation, as a consequence of network architecture and parameters. We present a new adversarial network optimisation algorithm that attacks network parameters during training, and promotes robust performance during inference in the face of parameter variation. Our approach introduces a regularization term penalising the susceptibility of a network to weight perturbation. We compare against previous approaches for producing parameter insensitivity such as dropout, weight smoothing and introducing parameter noise during training. We show that our approach produces models that are more robust to targeted parameter variation, and equally robust to random parameter variation. Our approach finds minima in flatter locations in the weight-loss landscape compared with other approaches, highlighting that the networks found by our technique are less sensitive to parameter perturbation. Our work provides an approach to deploy neural network architectures to inference devices that suffer from computational non-idealities, with minimal loss of performance. ...

摘要: 神经形态神经网络处理器，以记忆电阻的计算-内存交叉开关阵列的形式，或以亚阈值模拟和混合信号ASIC的形式，有望在基于神经网络的ML任务的计算密度和能量效率方面具有巨大的优势。然而，由于工艺变化和本征器件物理，这些技术容易出现计算上的非理想化。这会在部署的模型中引入参数噪声，从而降低部署到处理器的网络的任务性能。虽然有可能校准每个设备，或者为每个处理器单独训练网络，但这些方法成本高昂，对于商业部署来说不切实际。因此，需要替代方法来训练作为网络体系结构和参数的结果而对参数变化具有内在健壮性的网络。我们提出了一种新的对抗性网络优化算法，该算法在训练过程中攻击网络参数，在面对参数变化的情况下提高推理过程中的鲁棒性。我们的方法引入了一个正则化项，惩罚了网络对权重扰动的敏感性。我们比较了以往产生参数不敏感度的方法，如丢弃、权重平滑和在训练过程中引入参数噪声。我们表明，我们的方法产生的模型对目标参数变化更稳健，对随机参数变化同样稳健。与其他方法相比，我们的方法在减肥场景中更平坦的位置找到了最小值，突出表明我们的技术找到的网络对参数扰动不那么敏感。我们的工作提供了一种方法，以最小的性能损失将神经网络结构部署到遭受计算非理想影响的推理设备。..。



## **42. Unlearning Protected User Attributes in Recommendations with Adversarial Training**

在带有对抗性训练的推荐中忘记受保护的用户属性 cs.IR

Accepted at SIGIR 2022

**SubmitDate**: 2022-06-09    [paper-pdf](http://arxiv.org/pdf/2206.04500v1)

**Authors**: Christian Ganhör, David Penz, Navid Rekabsaz, Oleg Lesota, Markus Schedl

**Abstracts**: Collaborative filtering algorithms capture underlying consumption patterns, including the ones specific to particular demographics or protected information of users, e.g. gender, race, and location. These encoded biases can influence the decision of a recommendation system (RS) towards further separation of the contents provided to various demographic subgroups, and raise privacy concerns regarding the disclosure of users' protected attributes. In this work, we investigate the possibility and challenges of removing specific protected information of users from the learned interaction representations of a RS algorithm, while maintaining its effectiveness. Specifically, we incorporate adversarial training into the state-of-the-art MultVAE architecture, resulting in a novel model, Adversarial Variational Auto-Encoder with Multinomial Likelihood (Adv-MultVAE), which aims at removing the implicit information of protected attributes while preserving recommendation performance. We conduct experiments on the MovieLens-1M and LFM-2b-DemoBias datasets, and evaluate the effectiveness of the bias mitigation method based on the inability of external attackers in revealing the users' gender information from the model. Comparing with baseline MultVAE, the results show that Adv-MultVAE, with marginal deterioration in performance (w.r.t. NDCG and recall), largely mitigates inherent biases in the model on both datasets.

摘要: 协作过滤算法捕获潜在的消费模式，包括特定的人口统计数据或受保护的用户信息，例如性别、种族和位置。这些编码的偏见会影响推荐系统(RS)对提供给各种人口统计子组的内容的进一步分离的决策，并引起关于披露用户的受保护属性的隐私问题。在这项工作中，我们研究了从RS算法的学习交互表示中移除用户特定的受保护信息的可能性和挑战，同时保持其有效性。具体地说，我们将对抗性训练融入到最新的MultVAE体系结构中，形成了一种新的模型--对抗性多项似然变分自动编码器(ADV-MultVAE)，其目的是在保持推荐性能的同时消除受保护属性的隐含信息。我们在MovieLens-1M和LFM-2b-DemoBias数据集上进行了实验，并评估了基于外部攻击者无法从模型中透露用户性别信息的偏差缓解方法的有效性。与基线的MultVAE相比，结果显示ADV-MultVAE，性能略有下降(W.r.t.NDCG和Recall)，在很大程度上缓解了模型在这两个数据集上的固有偏差。



## **43. Subfield Algorithms for Ideal- and Module-SVP Based on the Decomposition Group**

基于分解群的理想和模SVP的子场算法 cs.CR

29 pages plus appendix, to appear in Banach Center Publications

**SubmitDate**: 2022-06-09    [paper-pdf](http://arxiv.org/pdf/2105.03219v3)

**Authors**: Christian Porter, Andrew Mendelsohn, Cong Ling

**Abstracts**: Whilst lattice-based cryptosystems are believed to be resistant to quantum attack, they are often forced to pay for that security with inefficiencies in implementation. This problem is overcome by ring- and module-based schemes such as Ring-LWE or Module-LWE, whose keysize can be reduced by exploiting its algebraic structure, allowing for faster computations. Many rings may be chosen to define such cryptoschemes, but cyclotomic rings, due to their cyclic nature allowing for easy multiplication, are the community standard. However, there is still much uncertainty as to whether this structure may be exploited to an adversary's benefit. In this paper, we show that the decomposition group of a cyclotomic ring of arbitrary conductor can be utilised to significantly decrease the dimension of the ideal (or module) lattice required to solve a given instance of SVP. Moreover, we show that there exist a large number of rational primes for which, if the prime ideal factors of an ideal lie over primes of this form, give rise to an "easy" instance of SVP. It is important to note that the work on ideal SVP does not break Ring-LWE, since its security reduction is from worst case ideal SVP to average case Ring-LWE, and is one way.

摘要: 虽然基于格子的密码系统被认为能够抵抗量子攻击，但它们经常被迫为这种安全性买单，因为实现效率低下。这个问题可以通过基于环和模块的方案来解决，例如环-LWE或模块-LWE，其密钥大小可以通过利用其代数结构来减小，从而允许更快的计算。可以选择许多环来定义这样的密码方案，但割圆环由于其循环性质允许容易相乘，是社区标准。然而，对于这种结构是否会被利用来为对手谋取利益，仍然存在很大的不确定性。在这篇文章中，我们证明了任意导体的分圆环的分解群可以用来显著降低求解给定SVP实例所需的理想(或模)格的维度。此外，我们还证明了存在大量的有理素数，对于这些有理素数，如果理想的素数理想因子位于这种形式的素数之上，则会产生SVP的“简单”实例。值得注意的是，关于理想SVP的工作不会破坏Ring-LWE，因为它的安全性降低是从最坏情况的理想SVP到平均情况的Ring-LWE，并且是单向的。



## **44. CARLA-GeAR: a Dataset Generator for a Systematic Evaluation of Adversarial Robustness of Vision Models**

Carla-Gear：用于系统评估视觉模型对抗稳健性的数据集生成器 cs.CV

**SubmitDate**: 2022-06-09    [paper-pdf](http://arxiv.org/pdf/2206.04365v1)

**Authors**: Federico Nesti, Giulio Rossolini, Gianluca D'Amico, Alessandro Biondi, Giorgio Buttazzo

**Abstracts**: Adversarial examples represent a serious threat for deep neural networks in several application domains and a huge amount of work has been produced to investigate them and mitigate their effects. Nevertheless, no much work has been devoted to the generation of datasets specifically designed to evaluate the adversarial robustness of neural models. This paper presents CARLA-GeAR, a tool for the automatic generation of photo-realistic synthetic datasets that can be used for a systematic evaluation of the adversarial robustness of neural models against physical adversarial patches, as well as for comparing the performance of different adversarial defense/detection methods. The tool is built on the CARLA simulator, using its Python API, and allows the generation of datasets for several vision tasks in the context of autonomous driving. The adversarial patches included in the generated datasets are attached to billboards or the back of a truck and are crafted by using state-of-the-art white-box attack strategies to maximize the prediction error of the model under test. Finally, the paper presents an experimental study to evaluate the performance of some defense methods against such attacks, showing how the datasets generated with CARLA-GeAR might be used in future work as a benchmark for adversarial defense in the real world. All the code and datasets used in this paper are available at http://carlagear.retis.santannapisa.it.

摘要: 对抗性的例子在几个应用领域对深度神经网络构成了严重的威胁，并且已经产生了大量的工作来调查它们并减轻它们的影响。然而，没有太多的工作致力于生成专门设计来评估神经模型的对抗性稳健性的数据集。本文介绍了一个自动生成照片真实感合成数据集的工具Carla-Gear，它可以用来系统地评估神经模型对物理对抗性补丁的对抗性健壮性，以及比较不同对抗性防御/检测方法的性能。该工具建立在Carla模拟器上，使用其PythonAPI，并允许在自动驾驶的背景下为几个视觉任务生成数据集。生成的数据集中包含的对抗性补丁被附加到广告牌或卡车后部，并通过使用最先进的白盒攻击策略来制作，以最大限度地提高测试模型的预测误差。最后，本文给出了一个实验研究，评估了一些防御方法对这类攻击的性能，展示了使用Carla-Gear生成的数据集如何在未来的工作中用作现实世界中对抗性防御的基准。本文中使用的所有代码和数据集都可以在http://carlagear.retis.santannapisa.it.上找到



## **45. Plug & Play Attacks: Towards Robust and Flexible Model Inversion Attacks**

即插即用攻击：朝向健壮灵活的模型反转攻击 cs.LG

Accepted by ICML 2022

**SubmitDate**: 2022-06-09    [paper-pdf](http://arxiv.org/pdf/2201.12179v4)

**Authors**: Lukas Struppek, Dominik Hintersdorf, Antonio De Almeida Correia, Antonia Adler, Kristian Kersting

**Abstracts**: Model inversion attacks (MIAs) aim to create synthetic images that reflect the class-wise characteristics from a target classifier's private training data by exploiting the model's learned knowledge. Previous research has developed generative MIAs that use generative adversarial networks (GANs) as image priors tailored to a specific target model. This makes the attacks time- and resource-consuming, inflexible, and susceptible to distributional shifts between datasets. To overcome these drawbacks, we present Plug & Play Attacks, which relax the dependency between the target model and image prior, and enable the use of a single GAN to attack a wide range of targets, requiring only minor adjustments to the attack. Moreover, we show that powerful MIAs are possible even with publicly available pre-trained GANs and under strong distributional shifts, for which previous approaches fail to produce meaningful results. Our extensive evaluation confirms the improved robustness and flexibility of Plug & Play Attacks and their ability to create high-quality images revealing sensitive class characteristics.

摘要: 模型反转攻击(MIA)的目的是利用目标分类器的学习知识，从目标分类器的私有训练数据中创建反映类别特征的合成图像。以前的研究已经开发出生成性MIA，它使用生成性对抗网络(GANS)作为针对特定目标模型量身定做的图像先验。这使得攻击耗费时间和资源，不灵活，并且容易受到数据集之间的分布变化的影响。为了克服这些缺点，我们提出了即插即用攻击，它放松了目标模型和图像先验之间的依赖，使单个GAN能够攻击范围广泛的目标，只需要对攻击进行微小的调整。此外，我们表明，即使在公开可用的预先训练的GAN和强烈的分布变化下，强大的MIA也是可能的，以前的方法无法产生有意义的结果。我们广泛的评估证实了即插即用攻击的健壮性和灵活性的提高，以及它们创建揭示敏感类别特征的高质量图像的能力。



## **46. Learning to Break Deep Perceptual Hashing: The Use Case NeuralHash**

学习打破深度感知散列：用例NeuralHash cs.LG

Accepted by ACM FAccT 2022 as Oral

**SubmitDate**: 2022-06-09    [paper-pdf](http://arxiv.org/pdf/2111.06628v4)

**Authors**: Lukas Struppek, Dominik Hintersdorf, Daniel Neider, Kristian Kersting

**Abstracts**: Apple recently revealed its deep perceptual hashing system NeuralHash to detect child sexual abuse material (CSAM) on user devices before files are uploaded to its iCloud service. Public criticism quickly arose regarding the protection of user privacy and the system's reliability. In this paper, we present the first comprehensive empirical analysis of deep perceptual hashing based on NeuralHash. Specifically, we show that current deep perceptual hashing may not be robust. An adversary can manipulate the hash values by applying slight changes in images, either induced by gradient-based approaches or simply by performing standard image transformations, forcing or preventing hash collisions. Such attacks permit malicious actors easily to exploit the detection system: from hiding abusive material to framing innocent users, everything is possible. Moreover, using the hash values, inferences can still be made about the data stored on user devices. In our view, based on our results, deep perceptual hashing in its current form is generally not ready for robust client-side scanning and should not be used from a privacy perspective.

摘要: 苹果最近公布了其深度感知哈希系统NeuralHash，用于在文件上传到其iCloud服务之前检测用户设备上的儿童性虐待材料(CSAM)。公众很快就对保护用户隐私和系统的可靠性提出了批评。本文首次提出了基于NeuralHash的深度感知哈希算法的综合实证分析。具体地说，我们证明了当前的深度感知散列可能并不健壮。攻击者可以通过在图像中应用微小的更改来操纵散列值，这可以是由基于梯度的方法引起的，也可以只是通过执行标准图像转换来强制或防止散列冲突。这种攻击让恶意行为者很容易利用检测系统：从隐藏滥用材料到陷害无辜用户，一切皆有可能。此外，使用散列值，仍然可以对存储在用户设备上的数据进行推断。在我们看来，根据我们的结果，当前形式的深度感知散列通常还不能用于健壮的客户端扫描，不应该从隐私的角度使用。



## **47. Bounding Training Data Reconstruction in Private (Deep) Learning**

私密(深度)学习中的边界训练数据重构 cs.LG

**SubmitDate**: 2022-06-09    [paper-pdf](http://arxiv.org/pdf/2201.12383v3)

**Authors**: Chuan Guo, Brian Karrer, Kamalika Chaudhuri, Laurens van der Maaten

**Abstracts**: Differential privacy is widely accepted as the de facto method for preventing data leakage in ML, and conventional wisdom suggests that it offers strong protection against privacy attacks. However, existing semantic guarantees for DP focus on membership inference, which may overestimate the adversary's capabilities and is not applicable when membership status itself is non-sensitive. In this paper, we derive the first semantic guarantees for DP mechanisms against training data reconstruction attacks under a formal threat model. We show that two distinct privacy accounting methods -- Renyi differential privacy and Fisher information leakage -- both offer strong semantic protection against data reconstruction attacks.

摘要: 在ML中，差异隐私被广泛接受为防止数据泄露的事实上的方法，传统观点认为，它提供了针对隐私攻击的强大保护。然而，现有的DP语义保证侧重于成员关系推理，这可能会高估对手的能力，并且不适用于成员身份本身不敏感的情况。本文首先在形式化威胁模型下给出了DP机制抵抗训练数据重构攻击的语义保证。我们发现，两种不同的隐私记账方法--Renyi Differential Privacy和Fisher信息泄漏--都提供了对数据重构攻击的强大语义保护。



## **48. Blacklight: Scalable Defense for Neural Networks against Query-Based Black-Box Attacks**

Blacklight：针对基于查询的黑盒攻击的神经网络可扩展防御 cs.CR

**SubmitDate**: 2022-06-09    [paper-pdf](http://arxiv.org/pdf/2006.14042v3)

**Authors**: Huiying Li, Shawn Shan, Emily Wenger, Jiayun Zhang, Haitao Zheng, Ben Y. Zhao

**Abstracts**: Deep learning systems are known to be vulnerable to adversarial examples. In particular, query-based black-box attacks do not require knowledge of the deep learning model, but can compute adversarial examples over the network by submitting queries and inspecting returns. Recent work largely improves the efficiency of those attacks, demonstrating their practicality on today's ML-as-a-service platforms.   We propose Blacklight, a new defense against query-based black-box adversarial attacks. The fundamental insight driving our design is that, to compute adversarial examples, these attacks perform iterative optimization over the network, producing image queries highly similar in the input space. Blacklight detects query-based black-box attacks by detecting highly similar queries, using an efficient similarity engine operating on probabilistic content fingerprints. We evaluate Blacklight against eight state-of-the-art attacks, across a variety of models and image classification tasks. Blacklight identifies them all, often after only a handful of queries. By rejecting all detected queries, Blacklight prevents any attack to complete, even when attackers persist to submit queries after account ban or query rejection. Blacklight is also robust against several powerful countermeasures, including an optimal black-box attack that approximates white-box attacks in efficiency. Finally, we illustrate how Blacklight generalizes to other domains like text classification.

摘要: 众所周知，深度学习系统很容易受到敌意例子的攻击。特别是，基于查询的黑盒攻击不需要深度学习模型的知识，但可以通过提交查询和检查返回来计算网络上的对抗性示例。最近的工作在很大程度上提高了这些攻击的效率，证明了它们在今天的ML即服务平台上的实用性。我们提出了Blacklight，一种新的针对基于查询的黑盒对抗攻击的防御方案。驱动我们设计的基本见解是，为了计算对抗性的例子，这些攻击在网络上执行迭代优化，产生在输入空间中高度相似的图像查询。Blacklight使用对概率内容指纹进行操作的高效相似性引擎，通过检测高度相似的查询来检测基于查询的黑盒攻击。我们针对各种型号和图像分类任务中的八种最先进的攻击对Blacklight进行评估。Blacklight通常只在几个问题之后就能识别出所有这些问题。通过拒绝所有检测到的查询，Blacklight可以阻止任何攻击完成，即使攻击者在帐户禁用或查询拒绝后仍坚持提交查询。Blacklight对几种强大的对策也很健壮，包括在效率上接近白盒攻击的最佳黑盒攻击。最后，我们说明了Blacklight如何推广到文本分类等其他领域。



## **49. Adversarial Text Normalization**

对抗性文本规范化 cs.CL

**SubmitDate**: 2022-06-08    [paper-pdf](http://arxiv.org/pdf/2206.04137v1)

**Authors**: Joanna Bitton, Maya Pavlova, Ivan Evtimov

**Abstracts**: Text-based adversarial attacks are becoming more commonplace and accessible to general internet users. As these attacks proliferate, the need to address the gap in model robustness becomes imminent. While retraining on adversarial data may increase performance, there remains an additional class of character-level attacks on which these models falter. Additionally, the process to retrain a model is time and resource intensive, creating a need for a lightweight, reusable defense. In this work, we propose the Adversarial Text Normalizer, a novel method that restores baseline performance on attacked content with low computational overhead. We evaluate the efficacy of the normalizer on two problem areas prone to adversarial attacks, i.e. Hate Speech and Natural Language Inference. We find that text normalization provides a task-agnostic defense against character-level attacks that can be implemented supplementary to adversarial retraining solutions, which are more suited for semantic alterations.

摘要: 基于文本的敌意攻击正变得越来越常见，普通互联网用户也可以访问。随着这些攻击的激增，解决模型健壮性差距的需求变得迫在眉睫。虽然对对抗性数据的再训练可能会提高性能，但仍然存在一种额外的字符级攻击，这些模型在这种攻击上步履蹒跚。此外，重新训练模型的过程是时间和资源密集型的，这就产生了对轻型、可重复使用的防御的需求。在这项工作中，我们提出了对抗性文本规格化器，这是一种新的方法，以较低的计算开销恢复受攻击内容的基线性能。我们评估了归一化在两个容易受到敌意攻击的问题领域的有效性，即仇恨言论和自然语言推理。我们发现，文本归一化提供了一种针对字符级攻击的与任务无关的防御，可以实现对对抗性再训练解决方案的补充，后者更适合于语义变化。



## **50. PrivHAR: Recognizing Human Actions From Privacy-preserving Lens**

PrivHAR：从隐私保护镜头识别人类行为 cs.CV

**SubmitDate**: 2022-06-08    [paper-pdf](http://arxiv.org/pdf/2206.03891v1)

**Authors**: Carlos Hinojosa, Miguel Marquez, Henry Arguello, Ehsan Adeli, Li Fei-Fei, Juan Carlos Niebles

**Abstracts**: The accelerated use of digital cameras prompts an increasing concern about privacy and security, particularly in applications such as action recognition. In this paper, we propose an optimizing framework to provide robust visual privacy protection along the human action recognition pipeline. Our framework parameterizes the camera lens to successfully degrade the quality of the videos to inhibit privacy attributes and protect against adversarial attacks while maintaining relevant features for activity recognition. We validate our approach with extensive simulations and hardware experiments.

摘要: 数码相机的加速使用促使人们越来越关注隐私和安全，特别是在动作识别等应用中。在这篇文章中，我们提出了一个优化的框架，以提供稳健的视觉隐私保护沿人类行为识别管道。我们的框架对摄像机镜头进行了参数化处理，成功地降低了视频的质量，从而抑制了隐私属性并防止了敌意攻击，同时保持了活动识别的相关特征。我们通过大量的仿真和硬件实验来验证我们的方法。



