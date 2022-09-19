# Latest Adversarial Attack Papers
**update at 2022-09-20 06:31:25**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Adversarial Driving: Attacking End-to-End Autonomous Driving**

对抗性驾驶：攻击型端到端自动驾驶 cs.CV

7 pages, 6 figures

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2103.09151v4)

**Authors**: Han Wu, Syed Yunas, Sareh Rowlands, Wenjie Ruan, Johan Wahlstrom

**Abstracts**: As research in deep neural networks has advanced, deep convolutional networks have become feasible for automated driving tasks. In particular, there is an emerging trend of employing end-to-end neural network models for the automation of driving tasks. However, previous research has shown that deep neural network classifiers are vulnerable to adversarial attacks. For regression tasks, however, the effect of adversarial attacks is not as well understood. In this paper, we devise two white-box targeted attacks against end-to-end autonomous driving systems. The driving systems use a regression model that takes an image as input and outputs a steering angle. Our attacks manipulate the behavior of the autonomous driving system by perturbing the input image. Both attacks can be initiated in real-time on CPUs without employing GPUs. The efficiency of the attacks is illustrated using experiments conducted in Udacity. Demo video: https://youtu.be/I0i8uN2oOP0.

摘要: 随着深度神经网络研究的深入，深度卷积网络对于自动驾驶任务已经变得可行。特别是，使用端到端神经网络模型来实现驾驶任务自动化是一种新兴的趋势。然而，以往的研究表明，深度神经网络分类器容易受到敌意攻击。然而，对于回归任务，对抗性攻击的影响并没有被很好地理解。在本文中，我们设计了两种针对端到端自动驾驶系统的白盒针对性攻击。驾驶系统使用一个回归模型，该模型将图像作为输入，并输出转向角度。我们的攻击通过干扰输入图像来操纵自动驾驶系统的行为。这两种攻击都可以在不使用GPU的情况下在CPU上实时发起。通过在Udacity上进行的实验，说明了攻击的有效性。演示视频：https://youtu.be/I0i8uN2oOP0.



## **2. A Systematic Evaluation of Node Embedding Robustness**

节点嵌入健壮性的系统评估 cs.LG

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2209.08064v1)

**Authors**: Alexandru Mara, Jefrey Lijffijt, Stephan Günnemann, Tijl De Bie

**Abstracts**: Node embedding methods map network nodes to low dimensional vectors that can be subsequently used in a variety of downstream prediction tasks. The popularity of these methods has significantly increased in recent years, yet, their robustness to perturbations of the input data is still poorly understood. In this paper, we assess the empirical robustness of node embedding models to random and adversarial poisoning attacks. Our systematic evaluation covers representative embedding methods based on Skip-Gram, matrix factorization, and deep neural networks. We compare edge addition, deletion and rewiring strategies computed using network properties as well as node labels. We also investigate the effect of label homophily and heterophily on robustness. We report qualitative results via embedding visualization and quantitative results in terms of downstream node classification and network reconstruction performances. We found that node classification suffers from higher performance degradation as opposed to network reconstruction, and that degree-based and label-based attacks are on average the most damaging.

摘要: 节点嵌入方法将网络节点映射到可随后用于各种下行预测任务的低维向量。近年来，这些方法的普及率显著提高，然而，人们对它们对输入数据扰动的稳健性仍然知之甚少。在本文中，我们评估了节点嵌入模型对随机和对抗性中毒攻击的经验稳健性。我们的系统评价涵盖了基于Skip-Gram的典型嵌入方法、矩阵分解和深度神经网络。我们比较了使用网络属性和节点标签计算的边添加、删除和重新布线策略。我们还研究了标签的同质性和异质性对稳健性的影响。我们通过嵌入可视化和定量结果来报告下游节点分类和网络重构性能方面的定性结果。我们发现，与网络重建相比，节点分类遭受了更高的性能降级，基于度和基于标签的攻击平均破坏性最大。



## **3. PA-Boot: A Formally Verified Authentication Protocol for Multiprocessor Secure Boot**

PA-Boot：一种形式化验证的多处理器安全引导认证协议 cs.CR

Manuscript submitted to IEEE Trans. Dependable Secure Comput

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2209.07936v1)

**Authors**: Zhuoruo Zhang, Chenyang Yu, He Huang, Rui Chang, Mingshuai Chen, Qinming Dai, Wenbo Shen, Yongwang Zhao, Kui Ren

**Abstracts**: Hardware supply-chain attacks are raising significant security threats to the boot process of multiprocessor systems. This paper identifies a new, prevalent hardware supply-chain attack surface that can bypass multiprocessor secure boot due to the absence of processor-authentication mechanisms. To defend against such attacks, we present PA-Boot, the first formally verified processor-authentication protocol for secure boot in multiprocessor systems. PA-Boot is proved functionally correct and is guaranteed to detect multiple adversarial behaviors, e.g., processor replacements, man-in-the-middle attacks, and tampering with certificates. The fine-grained formalization of PA-Boot and its fully mechanized security proofs are carried out in the Isabelle/HOL theorem prover with 306 lemmas/theorems and ~7,100 LoC. Experiments on a proof-of-concept implementation indicate that PA-Boot can effectively identify boot-process attacks with a considerably minor overhead and thereby improve the security of multiprocessor systems.

摘要: 硬件供应链攻击正在给多处理器系统的引导过程带来严重的安全威胁。本文提出了一种新的、流行的硬件供应链攻击面，由于缺乏处理器认证机制，该攻击面可以绕过多处理器安全引导。为了防御此类攻击，我们提出了PA-Boot，这是第一个经过正式验证的用于多处理器系统安全引导的处理器认证协议。PA-Boot被证明在功能上是正确的，并保证可以检测到多种敌对行为，例如处理器更换、中间人攻击和篡改证书。PA-Boot的细粒度形式化及其全机械化安全证明是在Isabelle/HOL定理证明器上进行的，具有306个引理/定理和~7100个LoC。在概念验证实现上的实验表明，PA-Boot能够以相当小的开销有效地识别引导过程攻击，从而提高多处理器系统的安全性。



## **4. SplitGuard: Detecting and Mitigating Training-Hijacking Attacks in Split Learning**

SplitGuard：检测和缓解分裂学习中的训练劫持攻击 cs.CR

Proceedings of the 21st Workshop on Privacy in the Electronic Society  (WPES '22), November 7, 2022, Los Angeles, CA, USA

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2108.09052v3)

**Authors**: Ege Erdogan, Alptekin Kupcu, A. Ercument Cicek

**Abstracts**: Distributed deep learning frameworks such as split learning provide great benefits with regards to the computational cost of training deep neural networks and the privacy-aware utilization of the collective data of a group of data-holders. Split learning, in particular, achieves this goal by dividing a neural network between a client and a server so that the client computes the initial set of layers, and the server computes the rest. However, this method introduces a unique attack vector for a malicious server attempting to steal the client's private data: the server can direct the client model towards learning any task of its choice, e.g. towards outputting easily invertible values. With a concrete example already proposed (Pasquini et al., CCS '21), such training-hijacking attacks present a significant risk for the data privacy of split learning clients.   In this paper, we propose SplitGuard, a method by which a split learning client can detect whether it is being targeted by a training-hijacking attack or not. We experimentally evaluate our method's effectiveness, compare it with potential alternatives, and discuss in detail various points related to its use. We conclude that SplitGuard can effectively detect training-hijacking attacks while minimizing the amount of information recovered by the adversaries.

摘要: 分布式深度学习框架，如分裂学习，在训练深度神经网络的计算成本和对一组数据持有者的集体数据的隐私意识利用方面提供了巨大的好处。特别是，分裂学习通过在客户端和服务器之间划分神经网络来实现这一目标，以便客户端计算初始层集，服务器计算其余层。然而，这种方法为试图窃取客户端私有数据的恶意服务器引入了唯一的攻击矢量：服务器可以引导客户端模型学习其选择的任何任务，例如输出容易逆转的值。结合已提出的一个具体实例(Pasquini等人，CCS‘21)，这种训练劫持攻击给分裂学习客户端的数据隐私带来了很大的风险。在本文中，我们提出了一种分裂学习客户端可以检测其是否成为训练劫持攻击目标的方法SplitGuard。我们通过实验评估了该方法的有效性，并将其与潜在的替代方案进行了比较，并详细讨论了与其使用相关的各个要点。我们得出结论：SplitGuard能够有效地检测训练劫持攻击，同时最小化对手恢复的信息量。



## **5. Privacy-Preserving Distributed Expectation Maximization for Gaussian Mixture Model using Subspace Perturbation**

基于子空间扰动的混合高斯模型隐私保护分布期望最大化 cs.LG

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2209.07833v1)

**Authors**: Qiongxiu Li, Jaron Skovsted Gundersen, Katrine Tjell, Rafal Wisniewski, Mads Græsbøll Christensen

**Abstracts**: Privacy has become a major concern in machine learning. In fact, the federated learning is motivated by the privacy concern as it does not allow to transmit the private data but only intermediate updates. However, federated learning does not always guarantee privacy-preservation as the intermediate updates may also reveal sensitive information. In this paper, we give an explicit information-theoretical analysis of a federated expectation maximization algorithm for Gaussian mixture model and prove that the intermediate updates can cause severe privacy leakage. To address the privacy issue, we propose a fully decentralized privacy-preserving solution, which is able to securely compute the updates in each maximization step. Additionally, we consider two different types of security attacks: the honest-but-curious and eavesdropping adversary models. Numerical validation shows that the proposed approach has superior performance compared to the existing approach in terms of both the accuracy and privacy level.

摘要: 隐私已经成为机器学习中的一个主要问题。事实上，联合学习是出于隐私考虑，因为它不允许传输私有数据，而只允许传输中间更新。然而，联合学习并不总是保证隐私保护，因为中间更新也可能泄露敏感信息。本文对一种联合期望最大化算法进行了详细的信息论分析，证明了中间更新会导致严重的隐私泄露。为了解决隐私问题，我们提出了一种完全去中心化的隐私保护解决方案，该方案能够安全地计算每个最大化步骤中的更新。此外，我们考虑了两种不同类型的安全攻击：诚实但好奇的和窃听对手模型。数值验证表明，与已有方法相比，该方法在准确率和保密性方面都具有更好的性能。



## **6. A Large-scale Multiple-objective Method for Black-box Attack against Object Detection**

针对目标检测的大规模多目标黑盒攻击方法 cs.CV

14 pages, 5 figures, ECCV2022

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2209.07790v1)

**Authors**: Siyuan Liang, Longkang Li, Yanbo Fan, Xiaojun Jia, Jingzhi Li, Baoyuan Wu, Xiaochun Cao

**Abstracts**: Recent studies have shown that detectors based on deep models are vulnerable to adversarial examples, even in the black-box scenario where the attacker cannot access the model information. Most existing attack methods aim to minimize the true positive rate, which often shows poor attack performance, as another sub-optimal bounding box may be detected around the attacked bounding box to be the new true positive one. To settle this challenge, we propose to minimize the true positive rate and maximize the false positive rate, which can encourage more false positive objects to block the generation of new true positive bounding boxes. It is modeled as a multi-objective optimization (MOP) problem, of which the generic algorithm can search the Pareto-optimal. However, our task has more than two million decision variables, leading to low searching efficiency. Thus, we extend the standard Genetic Algorithm with Random Subset selection and Divide-and-Conquer, called GARSDC, which significantly improves the efficiency. Moreover, to alleviate the sensitivity to population quality in generic algorithms, we generate a gradient-prior initial population, utilizing the transferability between different detectors with similar backbones. Compared with the state-of-art attack methods, GARSDC decreases by an average 12.0 in the mAP and queries by about 1000 times in extensive experiments. Our codes can be found at https://github.com/LiangSiyuan21/ GARSDC.

摘要: 最近的研究表明，基于深度模型的检测器容易受到敌意示例的攻击，即使在攻击者无法访问模型信息的黑盒场景中也是如此。现有的大多数攻击方法都以最小化真实正确率为目标，这往往表现出较差的攻击性能，因为可能会在被攻击的边界框周围检测到另一个次优边界框，即新的真正边界框。为了解决这一挑战，我们提出了最小化真阳性率和最大化假阳性率的方法，这可以鼓励更多的假阳性对象阻止新的真阳性边界框的生成。将其建模为多目标优化问题，利用遗传算法搜索Pareto最优解。然而，我们的任务有200多万个决策变量，导致搜索效率较低。因此，我们对标准遗传算法GARSDC进行了扩展，使其具有随机子集选择和分而治之的特点，大大提高了求解效率。此外，为了缓解遗传算法对种群质量的敏感性，我们利用具有相似骨架的不同检测器之间的可转移性，生成一个梯度先验的初始种群。在大量的实验中，与现有的攻击方法相比，GARSDC在地图和查询上平均减少了12.0倍左右。我们的代码可以在https://github.com/LiangSiyuan21/GARSDC找到。



## **7. PointCAT: Contrastive Adversarial Training for Robust Point Cloud Recognition**

PointCAT：用于稳健点云识别的对比性对抗性训练 cs.CV

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2209.07788v1)

**Authors**: Qidong Huang, Xiaoyi Dong, Dongdong Chen, Hang Zhou, Weiming Zhang, Kui Zhang, Gang Hua, Nenghai Yu

**Abstracts**: Notwithstanding the prominent performance achieved in various applications, point cloud recognition models have often suffered from natural corruptions and adversarial perturbations. In this paper, we delve into boosting the general robustness of point cloud recognition models and propose Point-Cloud Contrastive Adversarial Training (PointCAT). The main intuition of PointCAT is encouraging the target recognition model to narrow the decision gap between clean point clouds and corrupted point clouds. Specifically, we leverage a supervised contrastive loss to facilitate the alignment and uniformity of the hypersphere features extracted by the recognition model, and design a pair of centralizing losses with the dynamic prototype guidance to avoid these features deviating from their belonging category clusters. To provide the more challenging corrupted point clouds, we adversarially train a noise generator along with the recognition model from the scratch, instead of using gradient-based attack as the inner loop like previous adversarial training methods. Comprehensive experiments show that the proposed PointCAT outperforms the baseline methods and dramatically boosts the robustness of different point cloud recognition models, under a variety of corruptions including isotropic point noises, the LiDAR simulated noises, random point dropping and adversarial perturbations.

摘要: 尽管点云识别模型在各种应用中取得了显著的性能，但它经常受到自然的破坏和对抗性的扰动。本文对提高点云识别模型的整体稳健性进行了深入研究，提出了点云对抗性训练(PointCAT)。PointCAT的主要直觉是鼓励目标识别模型缩小干净的点云和损坏的点云之间的决策差距。具体地说，我们利用有监督的对比损失来促进识别模型提取的超球特征的对齐和一致性，并在动态原型引导下设计了一对集中损失来避免这些特征偏离其所属的类别簇。为了提供更具挑战性的被破坏的点云，我们从零开始对抗性地训练噪声产生器和识别模型，而不是像以前的对抗性训练方法那样使用基于梯度的攻击作为内环。综合实验表明，在各向同性点噪声、LiDAR模拟噪声、随机点丢弃和对抗性扰动等多种情况下，PointCAT的性能优于基线方法，并显著提高了不同点云识别模型的稳健性。



## **8. On the Robustness of Graph Neural Diffusion to Topology Perturbations**

关于图神经扩散对拓扑扰动的稳健性 cs.LG

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2209.07754v1)

**Authors**: Yang Song, Qiyu Kang, Sijie Wang, Zhao Kai, Wee Peng Tay

**Abstracts**: Neural diffusion on graphs is a novel class of graph neural networks that has attracted increasing attention recently. The capability of graph neural partial differential equations (PDEs) in addressing common hurdles of graph neural networks (GNNs), such as the problems of over-smoothing and bottlenecks, has been investigated but not their robustness to adversarial attacks. In this work, we explore the robustness properties of graph neural PDEs. We empirically demonstrate that graph neural PDEs are intrinsically more robust against topology perturbation as compared to other GNNs. We provide insights into this phenomenon by exploiting the stability of the heat semigroup under graph topology perturbations. We discuss various graph diffusion operators and relate them to existing graph neural PDEs. Furthermore, we propose a general graph neural PDE framework based on which a new class of robust GNNs can be defined. We verify that the new model achieves comparable state-of-the-art performance on several benchmark datasets.

摘要: 图上的神经扩散是一类新的图神经网络，近年来受到越来越多的关注。图神经偏微分方程组(PDE)在解决图神经网络(GNN)的常见障碍(如过光滑和瓶颈问题)方面的能力已被研究，但其对对手攻击的稳健性尚未得到研究。在这项工作中，我们研究了图神经偏微分方程的稳健性。我们的经验证明，与其他GNN相比，图神经PDE在本质上对拓扑扰动具有更强的鲁棒性。通过利用图的拓扑扰动下热半群的稳定性，我们提供了对这一现象的见解。我们讨论了各种图扩散算子，并将它们与现有的图神经偏微分方程联系起来。此外，我们还提出了一个通用的图神经偏微分方程框架，基于该框架可以定义一类新的健壮GNN。我们在几个基准数据集上验证了新模型取得了相当于最先进的性能。



## **9. IPvSeeYou: Exploiting Leaked Identifiers in IPv6 for Street-Level Geolocation**

IPv6 SeeYou：利用IPv6中泄漏的标识符进行街道级地理定位 cs.NI

Accepted to S&P '23

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2208.06767v2)

**Authors**: Erik Rye, Robert Beverly

**Abstracts**: We present IPvSeeYou, a privacy attack that permits a remote and unprivileged adversary to physically geolocate many residential IPv6 hosts and networks with street-level precision. The crux of our method involves: 1) remotely discovering wide area (WAN) hardware MAC addresses from home routers; 2) correlating these MAC addresses with their WiFi BSSID counterparts of known location; and 3) extending coverage by associating devices connected to a common penultimate provider router.   We first obtain a large corpus of MACs embedded in IPv6 addresses via high-speed network probing. These MAC addresses are effectively leaked up the protocol stack and largely represent WAN interfaces of residential routers, many of which are all-in-one devices that also provide WiFi. We develop a technique to statistically infer the mapping between a router's WAN and WiFi MAC addresses across manufacturers and devices, and mount a large-scale data fusion attack that correlates WAN MACs with WiFi BSSIDs available in wardriving (geolocation) databases. Using these correlations, we geolocate the IPv6 prefixes of $>$12M routers in the wild across 146 countries and territories. Selected validation confirms a median geolocation error of 39 meters. We then exploit technology and deployment constraints to extend the attack to a larger set of IPv6 residential routers by clustering and associating devices with a common penultimate provider router. While we responsibly disclosed our results to several manufacturers and providers, the ossified ecosystem of deployed residential cable and DSL routers suggests that our attack will remain a privacy threat into the foreseeable future.

摘要: 我们提出了IPv6 SeeYou，这是一种隐私攻击，允许远程和非特权对手以街道级别的精度物理定位许多住宅IPv6主机和网络。我们方法的关键涉及：1)从家庭路由器远程发现广域(WAN)硬件MAC地址；2)将这些MAC地址与已知位置的对应WiFi BSSID关联；以及3)通过关联连接到公共倒数第二个提供商路由器的设备来扩展覆盖范围。我们首先通过高速网络探测获得嵌入在IPv6地址中的大量MAC语料库。这些MAC地址有效地沿协议堆栈向上泄露，主要代表住宅路由器的广域网接口，其中许多是也提供WiFi的一体化设备。我们开发了一种技术来统计推断路由器的广域网和跨制造商和设备的WiFi MAC地址之间的映射，并发动大规模数据融合攻击，将广域网MAC与战争驾驶(地理定位)数据库中提供的WiFi BSSID相关联。利用这些相关性，我们在146个国家和地区对价值超过1200万美元的路由器的IPv6前缀进行了地理定位。选定的验证确认地理位置误差的中位数为39米。然后，我们利用技术和部署限制将攻击扩展到更大的一组IPv6住宅路由器，方法是将设备与常见的倒数第二个提供商路由器进行集群和关联。虽然我们负责任地向几家制造商和供应商披露了我们的结果，但已部署的住宅有线电视和DSL路由器的僵化生态系统表明，在可预见的未来，我们的攻击仍将对隐私构成威胁。



## **10. Adversarial Detection: Attacking Object Detection in Real Time**

对抗性检测：攻击目标的实时检测 cs.AI

7 pages, 10 figures

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2209.01962v2)

**Authors**: Han Wu, Syed Yunas, Sareh Rowlands, Wenjie Ruan, Johan Wahlstrom

**Abstracts**: Intelligent robots rely on object detection models to perceive the environment. Following advances in deep learning security it has been revealed that object detection models are vulnerable to adversarial attacks. However, prior research primarily focuses on attacking static images or offline videos. Therefore, it is still unclear if such attacks could jeopardize real-world robotic applications in dynamic environments. This paper bridges this gap by presenting the first real-time online attack against object detection models. We devise three attacks that fabricate bounding boxes for nonexistent objects at desired locations. The attacks achieve a success rate of about 90% within about 20 iterations. The demo video is available at: https://youtu.be/zJZ1aNlXsMU.

摘要: 智能机器人依靠物体检测模型来感知环境。随着深度学习安全性的进步，人们发现目标检测模型容易受到敌意攻击。然而，以往的研究主要集中在攻击静态图像或离线视频上。因此，目前尚不清楚此类攻击是否会危及动态环境中真实世界的机器人应用。本文通过提出第一个针对目标检测的实时在线攻击模型来弥补这一差距。我们设计了三种攻击，在所需位置为不存在的对象制造边界框。这些攻击在大约20次迭代内实现了约90%的成功率。演示视频可在以下网站上查看：https://youtu.be/zJZ1aNlXsMU.



## **11. A Man-in-the-Middle Attack against Object Detection Systems**

一种针对目标检测系统的中间人攻击 cs.RO

7 pages, 8 figures

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2208.07174v2)

**Authors**: Han Wu, Sareh Rowlands, Johan Wahlstrom

**Abstracts**: Thanks to the increasing power of CPUs and GPUs in embedded systems, deep-learning-enabled object detection systems have become pervasive in a multitude of robotic applications. While deep learning models are vulnerable to several well-known adversarial attacks, the applicability of these attacks is severely limited by strict assumptions on, for example, access to the detection system. Inspired by Man-in-the-Middle attacks in cryptography, we propose a novel hardware attack on object detection systems that overcomes these limitations. Experiments prove that it is possible to generate an efficient Universal Adversarial Perturbation (UAP) within one minute and then use the perturbation to attack a detection system via the Man-in-the-Middle attack. These findings raise serious concerns for applications of deep learning models in safety-critical systems, such as autonomous driving. Demo Video: https://youtu.be/OvIpe-R3ZS8.

摘要: 由于嵌入式系统中CPU和GPU的能力不断增强，支持深度学习的目标检测系统已经在许多机器人应用中变得普遍。虽然深度学习模型容易受到几种著名的对抗性攻击，但这些攻击的适用性受到对检测系统访问权限的严格假设的严重限制。受密码学中中间人攻击的启发，我们提出了一种新的针对目标检测系统的硬件攻击，克服了这些局限性。实验证明，可以在一分钟内产生一个有效的通用对抗扰动(UAP)，然后利用该扰动通过中间人攻击来攻击检测系统。这些发现引发了人们对深度学习模型在自动驾驶等安全关键系统中应用的严重担忧。演示视频：https://youtu.be/OvIpe-R3ZS8.



## **12. Adversarial Training for High-Stakes Reliability**

高风险可靠性的对抗性训练 cs.LG

31 pages, 6 figures, fixed incorrect citation

**SubmitDate**: 2022-09-15    [paper-pdf](http://arxiv.org/pdf/2205.01663v3)

**Authors**: Daniel M. Ziegler, Seraphina Nix, Lawrence Chan, Tim Bauman, Peter Schmidt-Nielsen, Tao Lin, Adam Scherlis, Noa Nabeshima, Ben Weinstein-Raun, Daniel de Haas, Buck Shlegeris, Nate Thomas

**Abstracts**: In the future, powerful AI systems may be deployed in high-stakes settings, where a single failure could be catastrophic. One technique for improving AI safety in high-stakes settings is adversarial training, which uses an adversary to generate examples to train on in order to achieve better worst-case performance.   In this work, we used a language generation task as a testbed for achieving high reliability through adversarial training. We created a series of adversarial training techniques -- including a tool that assists human adversaries -- to find and eliminate failures in a classifier that filters text completions suggested by a generator. In our simple "avoid injuries" task, we determined that we can set very conservative classifier thresholds without significantly impacting the quality of the filtered outputs. With our chosen thresholds, filtering with our baseline classifier decreases the rate of unsafe completions from about 2.4% to 0.003% on in-distribution data, which is near the limit of our ability to measure. We found that adversarial training significantly increased robustness to the adversarial attacks that we trained on, without affecting in-distribution performance. We hope to see further work in the high-stakes reliability setting, including more powerful tools for enhancing human adversaries and better ways to measure high levels of reliability, until we can confidently rule out the possibility of catastrophic deployment-time failures of powerful models.

摘要: 未来，强大的人工智能系统可能会部署在高风险的环境中，在那里，单一的故障可能是灾难性的。在高风险环境中提高人工智能安全性的一种技术是对抗性训练，它使用对手生成样本进行训练，以实现更好的最坏情况下的性能。在这项工作中，我们使用一个语言生成任务作为测试平台，通过对抗性训练来实现高可靠性。我们创建了一系列对抗性训练技术--包括一个帮助人类对手的工具--来发现并消除分类器中的故障，该分类器过滤生成器建议的文本完成。在我们简单的“避免受伤”任务中，我们确定可以设置非常保守的分类器阈值，而不会显著影响过滤输出的质量。在我们选择的阈值下，使用我们的基准分类器进行过滤可以将分发内数据的不安全完成率从大约2.4%降低到0.003%，这接近我们的测量能力极限。我们发现，对抗性训练显著提高了对我们训练的对抗性攻击的健壮性，而不影响分发内性能。我们希望在高风险的可靠性环境中看到进一步的工作，包括更强大的工具来增强人类对手，以及更好的方法来衡量高水平的可靠性，直到我们可以自信地排除强大模型部署时灾难性故障的可能性。



## **13. How to Attack and Defend NextG Radio Access Network Slicing with Reinforcement Learning**

基于强化学习的下一代无线接入网络分片攻防 cs.NI

**SubmitDate**: 2022-09-15    [paper-pdf](http://arxiv.org/pdf/2101.05768v2)

**Authors**: Yi Shi, Yalin E. Sagduyu, Tugba Erpek, M. Cenk Gursoy

**Abstracts**: In this paper, reinforcement learning (RL) for network slicing is considered in NextG radio access networks, where the base station (gNodeB) allocates resource blocks (RBs) to the requests of user equipments and aims to maximize the total reward of accepted requests over time. Based on adversarial machine learning, a novel over-the-air attack is introduced to manipulate the RL algorithm and disrupt NextG network slicing. The adversary observes the spectrum and builds its own RL based surrogate model that selects which RBs to jam subject to an energy budget with the objective of maximizing the number of failed requests due to jammed RBs. By jamming the RBs, the adversary reduces the RL algorithm's reward. As this reward is used as the input to update the RL algorithm, the performance does not recover even after the adversary stops jamming. This attack is evaluated in terms of both the recovery time and the (maximum and total) reward loss, and it is shown to be much more effective than benchmark (random and myopic) jamming attacks. Different reactive and proactive defense schemes (protecting the RL algorithm's updates or misleading the adversary's learning process) are introduced to show that it is viable to defend NextG network slicing against this attack.

摘要: 本文研究了下一代无线接入网络中网络切片的强化学习方法，其中基站(GNodeB)为用户设备的请求分配资源块(RB)，并以最大化接受请求的总回报为目标。在对抗性机器学习的基础上，引入了一种新的空中攻击来操纵RL算法并破坏NextG网络切片。敌手观察频谱并构建其自己的基于RL的代理模型，该代理模型选择在能量预算下阻塞哪个RB，目标是最大化由于阻塞的RB而失败的请求的数量。通过干扰RBS，对手降低了RL算法的奖励。由于该奖励被用作更新RL算法的输入，因此即使在对手停止干扰之后，性能也不会恢复。该攻击从恢复时间和(最大和总的)报酬损失两个方面进行评估，并且被证明比基准(随机和近视)干扰攻击要有效得多。不同的反应性和主动性防御方案(保护RL算法的更新或误导对手的学习过程)被引入，以表明防御NextG网络切片攻击是可行的。



## **14. A Light Recipe to Train Robust Vision Transformers**

培养健壮的视觉变形器的光明秘诀 cs.CV

Code available at https://github.com/dedeswim/vits-robustness-torch

**SubmitDate**: 2022-09-15    [paper-pdf](http://arxiv.org/pdf/2209.07399v1)

**Authors**: Edoardo Debenedetti, Vikash Sehwag, Prateek Mittal

**Abstracts**: In this paper, we ask whether Vision Transformers (ViTs) can serve as an underlying architecture for improving the adversarial robustness of machine learning models against evasion attacks. While earlier works have focused on improving Convolutional Neural Networks, we show that also ViTs are highly suitable for adversarial training to achieve competitive performance. We achieve this objective using a custom adversarial training recipe, discovered using rigorous ablation studies on a subset of the ImageNet dataset. The canonical training recipe for ViTs recommends strong data augmentation, in part to compensate for the lack of vision inductive bias of attention modules, when compared to convolutions. We show that this recipe achieves suboptimal performance when used for adversarial training. In contrast, we find that omitting all heavy data augmentation, and adding some additional bag-of-tricks ($\varepsilon$-warmup and larger weight decay), significantly boosts the performance of robust ViTs. We show that our recipe generalizes to different classes of ViT architectures and large-scale models on full ImageNet-1k. Additionally, investigating the reasons for the robustness of our models, we show that it is easier to generate strong attacks during training when using our recipe and that this leads to better robustness at test time. Finally, we further study one consequence of adversarial training by proposing a way to quantify the semantic nature of adversarial perturbations and highlight its correlation with the robustness of the model. Overall, we recommend that the community should avoid translating the canonical training recipes in ViTs to robust training and rethink common training choices in the context of adversarial training.

摘要: 在这篇文章中，我们问视觉转换器(VITS)是否可以作为一个底层架构来提高机器学习模型对逃避攻击的对抗性健壮性。虽然早期的工作集中在改进卷积神经网络上，但我们也表明VITS非常适合于对抗性训练，以获得竞争性的性能。我们通过对ImageNet数据集的子集进行严格的消融研究，发现了一种定制的对抗性训练配方，从而实现了这一目标。VITS的规范训练配方建议进行强大的数据增强，部分原因是为了弥补与卷曲相比，注意力模块缺乏视觉诱导偏差。我们表明，当用于对抗性训练时，这个配方达到了次优的性能。相反，我们发现，省略所有繁重的数据增强，并添加一些额外的技巧($varepsilon$-热身和更大的权重衰减)，显著提高了健壮VITS的性能。我们表明，我们的配方适用于不同类别的VIT体系结构和完整的ImageNet-1k上的大型模型。此外，研究了我们的模型健壮性的原因，我们表明，当使用我们的配方时，在训练期间更容易产生强攻击，这导致在测试时更好的健壮性。最后，我们进一步研究了对抗性训练的一个后果，提出了一种量化对抗性扰动的语义性质的方法，并强调了它与模型的稳健性的相关性。总体而言，我们建议社会应避免将VITS中的规范训练食谱转化为稳健的训练，并在对抗性训练的背景下重新考虑常见的训练选择。



## **15. Continuous Patrolling Games**

连续巡逻小游戏 cs.DM

**SubmitDate**: 2022-09-15    [paper-pdf](http://arxiv.org/pdf/2008.07369v2)

**Authors**: Steve Alpern, Thuy Bui, Thomas Lidbetter, Katerina Papadaki

**Abstracts**: We study a patrolling game played on a network $Q$, considered as a metric space. The Attacker chooses a point of $Q$ (not necessarily a node) to attack during a chosen time interval of fixed duration. The Patroller chooses a unit speed path on $Q$ and intercepts the attack (and wins) if she visits the attacked point during the attack time interval. This zero-sum game models the problem of protecting roads or pipelines from an adversarial attack. The payoff to the maximizing Patroller is the probability that the attack is intercepted. Our results include the following: (i) a solution to the game for any network $Q$, as long as the time required to carry out the attack is sufficiently short, (ii) a solution to the game for all tree networks that satisfy a certain condition on their extremities, and (iii) a solution to the game for any attack duration for stars with one long arc and the remaining arcs equal in length. We present a conjecture on the solution of the game for arbitrary trees and establish it in certain cases.

摘要: 我们研究了在被认为是度量空间的网络$Q$上进行的巡逻对策。攻击者在选定的固定持续时间间隔内选择一个$Q$点(不一定是节点)进行攻击。巡护员在$Q$上选择一条单位速度路径，如果她在攻击时间间隔内访问被攻击点，则拦截攻击(并获胜)。这个零和博弈模拟了保护道路或管道免受对手攻击的问题。最大化巡逻的回报是攻击被拦截的概率。我们的结果包括：(I)任意网络$q$的对策解，只要进行攻击所需的时间足够短；(Ii)对于所有在其末端满足一定条件的树网络的对策解；(Iii)对于具有一条长弧线且其余弧长相等的恒星的任意攻击持续时间的对策解。我们对任意树的对策的解提出了一个猜想，并在某些情况下建立了它。



## **16. Defending From Physically-Realizable Adversarial Attacks Through Internal Over-Activation Analysis**

通过内部过度激活分析防御物理上可实现的敌意攻击 cs.CV

**SubmitDate**: 2022-09-15    [paper-pdf](http://arxiv.org/pdf/2203.07341v2)

**Authors**: Giulio Rossolini, Federico Nesti, Fabio Brau, Alessandro Biondi, Giorgio Buttazzo

**Abstracts**: This work presents Z-Mask, a robust and effective strategy to improve the adversarial robustness of convolutional networks against physically-realizable adversarial attacks. The presented defense relies on specific Z-score analysis performed on the internal network features to detect and mask the pixels corresponding to adversarial objects in the input image. To this end, spatially contiguous activations are examined in shallow and deep layers to suggest potential adversarial regions. Such proposals are then aggregated through a multi-thresholding mechanism. The effectiveness of Z-Mask is evaluated with an extensive set of experiments carried out on models for both semantic segmentation and object detection. The evaluation is performed with both digital patches added to the input images and printed patches positioned in the real world. The obtained results confirm that Z-Mask outperforms the state-of-the-art methods in terms of both detection accuracy and overall performance of the networks under attack. Additional experiments showed that Z-Mask is also robust against possible defense-aware attacks.

摘要: 这项工作提出了一种健壮而有效的Z-MASK策略来提高卷积网络对物理可实现的敌意攻击的健壮性。所提出的防御依赖于对内部网络特征执行的特定Z-Score分析来检测和掩蔽输入图像中与敌对对象对应的像素。为此，在浅层和深层检查了空间上连续的激活，以暗示潜在的对抗性区域。然后，通过多门槛机制汇总这些建议。通过在语义分割和目标检测模型上进行的大量实验，对Z-MASK的有效性进行了评估。使用添加到输入图像的数字补丁和位于真实世界中的打印补丁来执行评估。实验结果表明，Z-MASK在检测准确率和网络整体性能方面均优于现有的方法。其他实验表明，Z-MASK对可能的防御感知攻击也具有很强的健壮性。



## **17. Improving Robust Fairness via Balance Adversarial Training**

通过平衡对抗训练提高稳健公平性 cs.LG

**SubmitDate**: 2022-09-15    [paper-pdf](http://arxiv.org/pdf/2209.07534v1)

**Authors**: Chunyu Sun, Chenye Xu, Chengyuan Yao, Siyuan Liang, Yichao Wu, Ding Liang, XiangLong Liu, Aishan Liu

**Abstracts**: Adversarial training (AT) methods are effective against adversarial attacks, yet they introduce severe disparity of accuracy and robustness between different classes, known as the robust fairness problem. Previously proposed Fair Robust Learning (FRL) adaptively reweights different classes to improve fairness. However, the performance of the better-performed classes decreases, leading to a strong performance drop. In this paper, we observed two unfair phenomena during adversarial training: different difficulties in generating adversarial examples from each class (source-class fairness) and disparate target class tendencies when generating adversarial examples (target-class fairness). From the observations, we propose Balance Adversarial Training (BAT) to address the robust fairness problem. Regarding source-class fairness, we adjust the attack strength and difficulties of each class to generate samples near the decision boundary for easier and fairer model learning; considering target-class fairness, by introducing a uniform distribution constraint, we encourage the adversarial example generation process for each class with a fair tendency. Extensive experiments conducted on multiple datasets (CIFAR-10, CIFAR-100, and ImageNette) demonstrate that our method can significantly outperform other baselines in mitigating the robust fairness problem (+5-10\% on the worst class accuracy)

摘要: 对抗训练(AT)方法对对抗攻击是有效的，但它们在不同类别之间引入了严重的准确性和稳健性差异，称为鲁棒公平性问题。以前提出的公平稳健学习(FRL)自适应地调整不同类别的权重以提高公平性。然而，表现较好的类的性能会下降，导致性能大幅下降。在本文中，我们观察到对抗性训练中的两种不公平现象：从每一类生成对抗性实例的难度不同(源类公平)和生成对抗性实例时不同的目标类倾向(目标类公平性)。通过观察，我们提出了平衡对抗训练(BAT)来解决稳健的公平性问题。在源类公平性方面，我们调整每个类的攻击强度和难度，在决策边界附近生成样本，使模型学习更容易、更公平；在目标类公平性方面，通过引入均匀分布约束，鼓励每个类具有公平倾向的对抗性样本生成过程。在多个数据集(CIFAR-10、CIFAR-100和ImageNette)上进行的大量实验表明，我们的方法在缓解稳健公平性问题方面可以显著优于其他基线(在最差分类准确率上+5-10%)



## **18. Decision-based Black-box Attack Against Vision Transformers via Patch-wise Adversarial Removal**

基于决策的基于补丁对抗性去除的视觉变形金刚黑盒攻击 cs.CV

**SubmitDate**: 2022-09-15    [paper-pdf](http://arxiv.org/pdf/2112.03492v2)

**Authors**: Yucheng Shi, Yahong Han, Yu-an Tan, Xiaohui Kuang

**Abstracts**: Vision transformers (ViTs) have demonstrated impressive performance and stronger adversarial robustness compared to Convolutional Neural Networks (CNNs). On the one hand, ViTs' focus on global interaction between individual patches reduces the local noise sensitivity of images. On the other hand, the neglect of noise sensitivity differences between image regions by existing decision-based attacks further compromises the efficiency of noise compression, especially for ViTs. Therefore, validating the black-box adversarial robustness of ViTs when the target model can only be queried still remains a challenging problem. In this paper, we theoretically analyze the limitations of existing decision-based attacks from the perspective of noise sensitivity difference between regions of the image, and propose a new decision-based black-box attack against ViTs, termed Patch-wise Adversarial Removal (PAR). PAR divides images into patches through a coarse-to-fine search process and compresses the noise on each patch separately. PAR records the noise magnitude and noise sensitivity of each patch and selects the patch with the highest query value for noise compression. In addition, PAR can be used as a noise initialization method for other decision-based attacks to improve the noise compression efficiency on both ViTs and CNNs without introducing additional calculations. Extensive experiments on three datasets demonstrate that PAR achieves a much lower noise magnitude with the same number of queries.

摘要: 与卷积神经网络(CNN)相比，视觉转换器(VITS)表现出了令人印象深刻的性能和更强的对抗鲁棒性。一方面，VITS对单个斑块之间全局交互的关注降低了图像的局部噪声敏感度。另一方面，现有的基于决策的攻击忽略了图像区域之间的噪声敏感度差异，进一步影响了噪声压缩的效率，特别是对VITS。因此，当目标模型只能被查询时，验证VITS的黑箱对抗健壮性仍然是一个具有挑战性的问题。本文从图像区域噪声敏感度差异的角度，从理论上分析了现有的基于决策的攻击方法的局限性，提出了一种新的基于决策的VITS黑盒攻击方法，称为Patch-Wise Aversarial Removal(PAR)。PAR通过从粗到精的搜索过程将图像分成多个块，并分别压缩每个块上的噪声。PAR记录每个面片的噪声大小和噪声敏感度，并选择查询值最高的面片进行噪声压缩。此外，PAR可以用作其他基于决策的攻击的噪声初始化方法，从而在不引入额外计算的情况下提高VITS和CNN的噪声压缩效率。在三个数据集上的大量实验表明，在相同的查询次数下，PAR获得了低得多的噪声幅度。



## **19. PointACL:Adversarial Contrastive Learning for Robust Point Clouds Representation under Adversarial Attack**

PointACL：对抗性攻击下鲁棒点云表示的对抗性对比学习 cs.CV

arXiv admin note: text overlap with arXiv:2109.00179 by other authors

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06971v1)

**Authors**: Junxuan Huang, Yatong An, Lu cheng, Bai Chen, Junsong Yuan, Chunming Qiao

**Abstracts**: Despite recent success of self-supervised based contrastive learning model for 3D point clouds representation, the adversarial robustness of such pre-trained models raised concerns. Adversarial contrastive learning (ACL) is considered an effective way to improve the robustness of pre-trained models. In contrastive learning, the projector is considered an effective component for removing unnecessary feature information during contrastive pretraining and most ACL works also use contrastive loss with projected feature representations to generate adversarial examples in pretraining, while "unprojected " feature representations are used in generating adversarial inputs during inference.Because of the distribution gap between projected and "unprojected" features, their models are constrained of obtaining robust feature representations for downstream tasks. We introduce a new method to generate high-quality 3D adversarial examples for adversarial training by utilizing virtual adversarial loss with "unprojected" feature representations in contrastive learning framework. We present our robust aware loss function to train self-supervised contrastive learning framework adversarially. Furthermore, we find selecting high difference points with the Difference of Normal (DoN) operator as additional input for adversarial self-supervised contrastive learning can significantly improve the adversarial robustness of the pre-trained model. We validate our method, PointACL on downstream tasks, including 3D classification and 3D segmentation with multiple datasets. It obtains comparable robust accuracy over state-of-the-art contrastive adversarial learning methods.

摘要: 尽管最近基于自监督的对比学习模型在三维点云表示中取得了成功，但这种预训练模型的对抗性健壮性引起了人们的关注。对抗性对比学习被认为是提高预训练模型稳健性的有效方法。在对比学习中，投影器被认为是对比预训练中去除不必要特征信息的有效部件，大多数ACL工作在预训练中也使用对比损失和投影特征表征来生成对抗性样本，而在推理过程中则使用“非投影”特征表征来生成对抗性输入，由于投影和非投影特征之间的分布差距，它们的模型在获得下游任务的稳健特征表征方面受到限制。我们介绍了一种新的方法，通过在对比学习框架中利用虚拟对抗性损失和非投影的特征表示来生成用于对抗性训练的高质量3D对抗性实例。我们提出了稳健的意识损失函数来对抗性地训练自监督对比学习框架。此外，我们发现选择高差点和正态(DON)算子的差作为对抗性自监督对比学习的附加输入可以显著提高预训练模型的对抗性健壮性。我们在下游任务上验证了我们的方法，PointACL，包括3D分类和多个数据集的3D分割。与最先进的对比对抗性学习方法相比，它获得了相当的鲁棒性精度。



## **20. Finetuning Pretrained Vision-Language Models with Correlation Information Bottleneck for Robust Visual Question Answering**

基于关联信息瓶颈的视觉问答精调算法 cs.CV

20 pages, 4 figures, 13 tables

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06954v1)

**Authors**: Jingjing Jiang, Ziyi Liu, Nanning Zheng

**Abstracts**: Benefiting from large-scale Pretrained Vision-Language Models (VL-PMs), the performance of Visual Question Answering (VQA) has started to approach human oracle performance. However, finetuning large-scale VL-PMs with limited data for VQA usually faces overfitting and poor generalization issues, leading to a lack of robustness. In this paper, we aim to improve the robustness of VQA systems (ie, the ability of the systems to defend against input variations and human-adversarial attacks) from the perspective of Information Bottleneck when finetuning VL-PMs for VQA. Generally, internal representations obtained by VL-PMs inevitably contain irrelevant and redundant information for the downstream VQA task, resulting in statistically spurious correlations and insensitivity to input variations. To encourage representations to converge to a minimal sufficient statistic in vision-language learning, we propose the Correlation Information Bottleneck (CIB) principle, which seeks a tradeoff between representation compression and redundancy by minimizing the mutual information (MI) between the inputs and internal representations while maximizing the MI between the outputs and the representations. Meanwhile, CIB measures the internal correlations among visual and linguistic inputs and representations by a symmetrized joint MI estimation. Extensive experiments on five VQA benchmarks of input robustness and two VQA benchmarks of human-adversarial robustness demonstrate the effectiveness and superiority of the proposed CIB in improving the robustness of VQA systems.

摘要: 得益于大规模预训练视觉语言模型(VL-PM)，视觉问答(VQA)的性能已经开始接近人类的预言性能。然而，在VQA数据有限的情况下，对大规模的VL-PM进行精调通常会面临过拟合和泛化能力差的问题，从而导致缺乏健壮性。本文从信息瓶颈的角度对VQA的VL-PM进行优化，旨在提高VQA系统的健壮性(即系统抵抗输入变化和人类攻击的能力)。通常，VL-PM得到的内部表示不可避免地包含与下游VQA任务无关和冗余的信息，导致统计上虚假的相关性和对输入变化的不敏感。为了在视觉语言学习中鼓励表征收敛到最小的充分统计量，我们提出了关联信息瓶颈(CIB)原则，该原则通过最小化输入和内部表征之间的互信息(MI)同时最大化输出和表征之间的MI来寻求表征压缩和冗余之间的折衷。同时，CIB通过对称化的联合MI估计来衡量视觉输入和语言输入与表征之间的内在关联。在5个输入健壮性VQA基准和2个人-对手健壮性VQA基准上的大量实验证明了所提出的CIB在提高VQA系统稳健性方面的有效性和优越性。



## **21. On the interplay of adversarial robustness and architecture components: patches, convolution and attention**

关于对抗性健壮性和体系结构组件的相互作用：补丁、卷积和注意力 cs.CV

Presented at the "New Frontiers in Adversarial Machine Learning"  Workshop at ICML 2022

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06953v1)

**Authors**: Francesco Croce, Matthias Hein

**Abstracts**: In recent years novel architecture components for image classification have been developed, starting with attention and patches used in transformers. While prior works have analyzed the influence of some aspects of architecture components on the robustness to adversarial attacks, in particular for vision transformers, the understanding of the main factors is still limited. We compare several (non)-robust classifiers with different architectures and study their properties, including the effect of adversarial training on the interpretability of the learnt features and robustness to unseen threat models. An ablation from ResNet to ConvNeXt reveals key architectural changes leading to almost $10\%$ higher $\ell_\infty$-robustness.

摘要: 近年来，从变压器中使用的注意和补丁开始，开发了用于图像分类的新的体系结构组件。虽然以前的工作已经分析了体系结构组件的某些方面对对抗攻击的健壮性的影响，特别是对于视觉转换器，但对主要因素的理解仍然有限。我们比较了几种不同结构的(非)稳健分类器，并研究了它们的性质，包括对抗性训练对学习特征的可解释性的影响以及对不可见威胁模型的稳健性。从ResNet到ConvNeXt的消融揭示了导致$10\$更高的$\ell_\inty$-健壮性的关键架构变化。



## **22. Robust Constrained Reinforcement Learning**

稳健的约束强化学习 cs.LG

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06866v1)

**Authors**: Yue Wang, Fei Miao, Shaofeng Zou

**Abstracts**: Constrained reinforcement learning is to maximize the expected reward subject to constraints on utilities/costs. However, the training environment may not be the same as the test one, due to, e.g., modeling error, adversarial attack, non-stationarity, resulting in severe performance degradation and more importantly constraint violation. We propose a framework of robust constrained reinforcement learning under model uncertainty, where the MDP is not fixed but lies in some uncertainty set, the goal is to guarantee that constraints on utilities/costs are satisfied for all MDPs in the uncertainty set, and to maximize the worst-case reward performance over the uncertainty set. We design a robust primal-dual approach, and further theoretically develop guarantee on its convergence, complexity and robust feasibility. We then investigate a concrete example of $\delta$-contamination uncertainty set, design an online and model-free algorithm and theoretically characterize its sample complexity.

摘要: 约束强化学习是在效用/成本约束下最大化期望收益的学习方法。然而，由于例如建模错误、对抗性攻击、非平稳性等原因，训练环境可能与测试环境不同，从而导致严重的性能降级，更重要的是违反约束。我们提出了一种模型不确定性下的鲁棒约束强化学习框架，其中MDP不是固定的，而是位于某个不确定集合中，目标是保证不确定集合中的所有MDP都满足效用/成本约束，并且最大化不确定集合上的最坏情况下的回报性能。我们设计了一种稳健的原始-对偶方法，并从理论上对其收敛、复杂性和稳健可行性进行了保证。然后，我们研究了一个具体的例子--污染不确定集，设计了一个在线的、无模型的算法，并从理论上刻画了其样本复杂性。



## **23. Certified Robustness to Word Substitution Ranking Attack for Neural Ranking Models**

神经排序模型对单词替换排序攻击的验证稳健性 cs.IR

Accepted by CIKM2022

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06691v1)

**Authors**: Chen Wu, Ruqing Zhang, Jiafeng Guo, Wei Chen, Yixing Fan, Maarten de Rijke, Xueqi Cheng

**Abstracts**: Neural ranking models (NRMs) have achieved promising results in information retrieval. NRMs have also been shown to be vulnerable to adversarial examples. A typical Word Substitution Ranking Attack (WSRA) against NRMs was proposed recently, in which an attacker promotes a target document in rankings by adding human-imperceptible perturbations to its text. This raises concerns when deploying NRMs in real-world applications. Therefore, it is important to develop techniques that defend against such attacks for NRMs. In empirical defenses adversarial examples are found during training and used to augment the training set. However, such methods offer no theoretical guarantee on the models' robustness and may eventually be broken by other sophisticated WSRAs. To escape this arms race, rigorous and provable certified defense methods for NRMs are needed.   To this end, we first define the \textit{Certified Top-$K$ Robustness} for ranking models since users mainly care about the top ranked results in real-world scenarios. A ranking model is said to be Certified Top-$K$ Robust on a ranked list when it is guaranteed to keep documents that are out of the top $K$ away from the top $K$ under any attack. Then, we introduce a Certified Defense method, named CertDR, to achieve certified top-$K$ robustness against WSRA, based on the idea of randomized smoothing. Specifically, we first construct a smoothed ranker by applying random word substitutions on the documents, and then leverage the ranking property jointly with the statistical property of the ensemble to provably certify top-$K$ robustness. Extensive experiments on two representative web search datasets demonstrate that CertDR can significantly outperform state-of-the-art empirical defense methods for ranking models.

摘要: 神经排序模型(NRM)在信息检索中取得了良好的效果。NRM也被证明容易受到敌意例子的影响。最近提出了一种针对NRMS的典型单词替换排名攻击(WSRA)，攻击者通过在文本中添加人类无法察觉的扰动来提升目标文档的排名。在实际应用程序中部署NRM时，这会引起人们的担忧。因此，开发针对NRM的防御此类攻击的技术非常重要。在经验防御中，对抗性的例子是在训练期间发现的，并被用来增加训练集。然而，这些方法对模型的稳健性没有提供理论上的保证，最终可能会被其他复杂的WSRA所打破。为了避免这场军备竞赛，需要针对NRM的严格和可证明的认证防御方法。为此，由于用户主要关心真实场景中排名靠前的结果，因此我们首先定义了模型排名的\textit{认证排名-$K$健壮性}。当保证在任何攻击下使排名前$K$之外的文档远离排名前$K$的文档时，排名模型在排名列表上被称为认证的Top-$K$健壮。然后，基于随机平滑的思想，提出了一种认证防御方法CertDR，以实现对WSRA的认证TOP-K$健壮性。具体地说，我们首先通过对文档进行随机单词替换来构造平滑的排序器，然后利用排序属性和集成的统计属性来证明top-$K$的稳健性。在两个具有代表性的网络搜索数据集上的大量实验表明，CertDR在排序模型方面的性能明显优于最先进的经验防御方法。



## **24. Order-Disorder: Imitation Adversarial Attacks for Black-box Neural Ranking Models**

有序-无序：对黑盒神经网络排序模型的模仿敌意攻击 cs.IR

15 pages, 4 figures, accepted by ACM CCS 2022

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06506v1)

**Authors**: Jiawei Liu, Yangyang Kang, Di Tang, Kaisong Song, Changlong Sun, Xiaofeng Wang, Wei Lu, Xiaozhong Liu

**Abstracts**: Neural text ranking models have witnessed significant advancement and are increasingly being deployed in practice. Unfortunately, they also inherit adversarial vulnerabilities of general neural models, which have been detected but remain underexplored by prior studies. Moreover, the inherit adversarial vulnerabilities might be leveraged by blackhat SEO to defeat better-protected search engines. In this study, we propose an imitation adversarial attack on black-box neural passage ranking models. We first show that the target passage ranking model can be transparentized and imitated by enumerating critical queries/candidates and then train a ranking imitation model. Leveraging the ranking imitation model, we can elaborately manipulate the ranking results and transfer the manipulation attack to the target ranking model. For this purpose, we propose an innovative gradient-based attack method, empowered by the pairwise objective function, to generate adversarial triggers, which causes premeditated disorderliness with very few tokens. To equip the trigger camouflages, we add the next sentence prediction loss and the language model fluency constraint to the objective function. Experimental results on passage ranking demonstrate the effectiveness of the ranking imitation attack model and adversarial triggers against various SOTA neural ranking models. Furthermore, various mitigation analyses and human evaluation show the effectiveness of camouflages when facing potential mitigation approaches. To motivate other scholars to further investigate this novel and important problem, we make the experiment data and code publicly available.

摘要: 神经文本排序模型已经取得了显著的进步，并越来越多地应用于实践中。不幸的是，它们也继承了一般神经模型的对抗性漏洞，这些漏洞已经被检测到，但仍未被先前的研究充分探索。此外，BlackHat SEO可能会利用继承的敌意漏洞来击败保护更好的搜索引擎。在这项研究中，我们提出了一种对黑盒神经通路排序模型的模仿对抗性攻击。我们首先证明了目标文章排序模型可以通过列举关键查询/候选来透明化和模仿，然后训练一个排序模仿模型。利用排序模拟模型，可以对排序结果进行精细的操作，并将操纵攻击转移到目标排序模型。为此，我们提出了一种创新的基于梯度的攻击方法，该方法在两两目标函数的授权下，生成对抗性触发器，以极少的令牌造成有预谋的无序。为了装备触发伪装，我们在目标函数中加入了下一句预测损失和语言模型流畅度约束。文章排序的实验结果证明了该排序模仿攻击模型和敌意触发器对各种SOTA神经排序模型的有效性。此外，各种缓解分析和人类评估表明，当面临潜在的缓解方法时，伪装是有效的。为了激励其他学者进一步研究这一新颖而重要的问题，我们公开了实验数据和代码。



## **25. Targeting interventions for displacement minimization in opinion dynamics**

意见动力学中位移最小化的定向干预 cs.SI

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06481v1)

**Authors**: Luca Damonte, Giacomo Como, Fabio Fagnani

**Abstracts**: Social influence is largely recognized as a key factor in opinion formation processes. Recently, the role of external forces in inducing opinion displacement and polarization in social networks has attracted significant attention. This is in particular motivated by the necessity to understand and possibly prevent interference phenomena during political campaigns and elections. In this paper, we formulate and solve a targeted intervention problem for opinion displacement minimization on a social network. Specifically, we consider a min-max problem whereby a social planner (the defender) aims at selecting the optimal network intervention within her given budget constraint in order to minimize the opinion displacement in the system that an adversary (the attacker) is instead trying to maximize. Our results show that the optimal intervention of the defender has two regimes. For large enough budget, the optimal intervention of the social planner acts on all nodes proportionally to a new notion of network centrality. For lower budget values, such optimal intervention has a more delicate structure and is rather concentrated on a few target individuals.

摘要: 社会影响力在很大程度上被认为是舆论形成过程中的一个关键因素。最近，外力在社交网络中引发意见转移和两极分化的作用引起了人们的极大关注。这尤其是因为有必要了解并可能防止政治竞选和选举期间的干扰现象。在这篇文章中，我们制定并解决了一个针对社会网络上的意见移位最小化的定向干预问题。具体地说，我们考虑了一个最小-最大问题，在这个问题中，社会规划者(防御者)的目标是在她给定的预算限制内选择最优的网络干预，以便最小化系统中对手(攻击者)试图最大化的意见位移。我们的结果表明，防御者的最优干预有两种机制。对于足够大的预算，社会规划者的最优干预按照网络中心性的新概念按比例作用于所有节点。对于较低的预算价值，这种最佳干预具有更微妙的结构，并且相当集中在少数目标个人身上。



## **26. Private Eye: On the Limits of Textual Screen Peeking via Eyeglass Reflections in Video Conferencing**

私人眼睛：视频会议中通过眼镜反射窥视文本屏幕的限度 cs.CR

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2205.03971v2)

**Authors**: Yan Long, Chen Yan, Shilin Xiao, Shivan Prasad, Wenyuan Xu, Kevin Fu

**Abstracts**: Using mathematical modeling and human subjects experiments, this research explores the extent to which emerging webcams might leak recognizable textual and graphical information gleaming from eyeglass reflections captured by webcams. The primary goal of our work is to measure, compute, and predict the factors, limits, and thresholds of recognizability as webcam technology evolves in the future. Our work explores and characterizes the viable threat models based on optical attacks using multi-frame super resolution techniques on sequences of video frames. Our models and experimental results in a controlled lab setting show it is possible to reconstruct and recognize with over 75% accuracy on-screen texts that have heights as small as 10 mm with a 720p webcam. We further apply this threat model to web textual contents with varying attacker capabilities to find thresholds at which text becomes recognizable. Our user study with 20 participants suggests present-day 720p webcams are sufficient for adversaries to reconstruct textual content on big-font websites. Our models further show that the evolution towards 4K cameras will tip the threshold of text leakage to reconstruction of most header texts on popular websites. Besides textual targets, a case study on recognizing a closed-world dataset of Alexa top 100 websites with 720p webcams shows a maximum recognition accuracy of 94% with 10 participants even without using machine-learning models. Our research proposes near-term mitigations including a software prototype that users can use to blur the eyeglass areas of their video streams. For possible long-term defenses, we advocate an individual reflection testing procedure to assess threats under various settings, and justify the importance of following the principle of least privilege for privacy-sensitive scenarios.

摘要: 通过数学建模和人体实验，这项研究探索了新兴的网络摄像头可能在多大程度上泄露从网络摄像头捕获的眼镜反射中闪烁的可识别的文本和图形信息。我们工作的主要目标是测量、计算和预测随着未来网络摄像头技术的发展而产生的可识别性的因素、限制和阈值。我们的工作利用视频帧序列上的多帧超分辨率技术，探索和表征了基于光学攻击的可行威胁模型。我们的模型和在受控实验室环境下的实验结果表明，使用720p网络摄像头可以重建和识别高度低至10 mm的屏幕文本，准确率超过75%。我们进一步将该威胁模型应用于具有不同攻击者能力的Web文本内容，以找出文本变得可识别的阈值。我们对20名参与者的用户研究表明，目前的720p网络摄像头足以让对手在大字体网站上重建文本内容。我们的模型进一步表明，向4K摄像头的演变将使文本泄漏的门槛倾斜到重建流行网站上的大多数标题文本。除了文本目标，在使用720P网络摄像头识别Alexa前100名网站的封闭世界数据集上的案例研究显示，即使不使用机器学习模型，在10个参与者的情况下，最高识别准确率也达到94%。我们的研究提出了近期的缓解措施，包括一个软件原型，用户可以使用它来模糊他们视频流的眼镜区域。对于可能的长期防御，我们主张使用个人反射测试程序来评估各种环境下的威胁，并证明在隐私敏感场景中遵循最小特权原则的重要性。



## **27. TSFool: Crafting High-quality Adversarial Time Series through Multi-objective Optimization to Fool Recurrent Neural Network Classifiers**

TSFool：基于多目标优化的递归神经网络分类器生成高质量的对抗性时间序列 cs.LG

9 pages, 5 figures

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06388v1)

**Authors**: Yanyun Wang, Dehui Du, Yuanhao Liu

**Abstracts**: Deep neural network (DNN) classifiers are vulnerable to adversarial attacks. Although the existing gradient-based attacks have achieved good performance in feed-forward model and image recognition tasks, the extension for time series classification in the recurrent neural network (RNN) remains a dilemma, because the cyclical structure of RNN prevents direct model differentiation and the visual sensitivity to perturbations of time series data challenges the traditional local optimization objective to minimize perturbation. In this paper, an efficient and widely applicable approach called TSFool for crafting high-quality adversarial time series for the RNN classifier is proposed. We propose a novel global optimization objective named Camouflage Coefficient to consider how well the adversarial samples hide in class clusters, and accordingly redefine the high-quality adversarial attack as a multi-objective optimization problem. We also propose a new idea to use intervalized weighted finite automata (IWFA) to capture deeply embedded vulnerable samples having otherness between features and latent manifold to guide the approximation to the optimization solution. Experiments on 22 UCR datasets are conducted to confirm that TSFool is a widely effective, efficient and high-quality approach with 93.22% less local perturbation, 32.33% better global camouflage, and 1.12 times speedup to existing methods.

摘要: 深度神经网络(DNN)分类器容易受到敌意攻击。尽管现有的基于梯度的攻击在前馈模型和图像识别任务中取得了良好的性能，但由于递归神经网络的周期性结构阻止了直接的模型区分，以及对时间序列数据扰动的视觉敏感性挑战了传统的局部优化目标以最小化扰动，因此递归神经网络对时间序列分类的扩展仍然是一个两难问题。本文提出了一种用于为RNN分类器生成高质量对抗性时间序列的方法TSFool，该方法具有较高的效率和广泛的适用性。我们提出了一种新的全局优化目标伪装系数来考虑敌方样本在类簇中的隐藏程度，从而将高质量的敌方攻击重新定义为一个多目标优化问题。我们还提出了一种新的思想，使用区间加权有限自动机(IWFA)来捕获特征和潜在流形之间存在差异的深度嵌入的易受攻击样本，以指导逼近到最优解。在22个UCR数据集上的实验表明，TSFool是一种广泛有效、高效和高质量的方法，局部扰动减少了93.22%，全局伪装效果提高了32.33%，是现有方法的1.12倍。



## **28. PINCH: An Adversarial Extraction Attack Framework for Deep Learning Models**

PINCH：一种面向深度学习模型的对抗性抽取攻击框架 cs.CR

15 pages, 11 figures, 2 tables

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.06300v1)

**Authors**: William Hackett, Stefan Trawicki, Zhengxin Yu, Neeraj Suri, Peter Garraghan

**Abstracts**: Deep Learning (DL) models increasingly power a diversity of applications. Unfortunately, this pervasiveness also makes them attractive targets for extraction attacks which can steal the architecture, parameters, and hyper-parameters of a targeted DL model. Existing extraction attack studies have observed varying levels of attack success for different DL models and datasets, yet the underlying cause(s) behind their susceptibility often remain unclear. Ascertaining such root-cause weaknesses would help facilitate secure DL systems, though this requires studying extraction attacks in a wide variety of scenarios to identify commonalities across attack success and DL characteristics. The overwhelmingly high technical effort and time required to understand, implement, and evaluate even a single attack makes it infeasible to explore the large number of unique extraction attack scenarios in existence, with current frameworks typically designed to only operate for specific attack types, datasets and hardware platforms. In this paper we present PINCH: an efficient and automated extraction attack framework capable of deploying and evaluating multiple DL models and attacks across heterogeneous hardware platforms. We demonstrate the effectiveness of PINCH by empirically evaluating a large number of previously unexplored extraction attack scenarios, as well as secondary attack staging. Our key findings show that 1) multiple characteristics affect extraction attack success spanning DL model architecture, dataset complexity, hardware, attack type, and 2) partially successful extraction attacks significantly enhance the success of further adversarial attack staging.

摘要: 深度学习(DL)模型越来越支持各种应用程序。不幸的是，这种普及性也使它们成为提取攻击的有吸引力的目标，这些攻击可以窃取目标DL模型的体系结构、参数和超参数。现有的提取攻击研究已经观察到不同的DL模型和数据集的攻击成功程度不同，但其易感性背后的潜在原因往往尚不清楚。查明此类根本原因弱点将有助于促进数字图书馆系统的安全，尽管这需要研究各种情况下的提取攻击，以确定攻击成功和数字图书馆特征之间的共性。理解、实施和评估单个攻击所需的极高的技术工作量和时间，使得探索现有的大量独特的提取攻击场景变得不可行，目前的框架通常设计为仅针对特定攻击类型、数据集和硬件平台运行。在本文中，我们提出了PINCH：一个高效的自动提取攻击框架，能够部署和评估跨不同硬件平台的多个DL模型和攻击。我们通过对大量以前未探索的提取攻击场景以及二次攻击阶段的经验评估，证明了PIPCH的有效性。我们的主要发现表明，1)多个特征影响跨DL模型架构、数据集复杂性、硬件、攻击类型的提取攻击成功；2)部分成功的提取攻击显著提高了进一步对抗性攻击的成功率。



## **29. Certified Defences Against Adversarial Patch Attacks on Semantic Segmentation**

基于语义分割的对抗性补丁攻击认证防御 cs.CV

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05980v1)

**Authors**: Maksym Yatsura, Kaspar Sakmann, N. Grace Hua, Matthias Hein, Jan Hendrik Metzen

**Abstracts**: Adversarial patch attacks are an emerging security threat for real world deep learning applications. We present Demasked Smoothing, the first approach (up to our knowledge) to certify the robustness of semantic segmentation models against this threat model. Previous work on certifiably defending against patch attacks has mostly focused on image classification task and often required changes in the model architecture and additional training which is undesirable and computationally expensive. In Demasked Smoothing, any segmentation model can be applied without particular training, fine-tuning, or restriction of the architecture. Using different masking strategies, Demasked Smoothing can be applied both for certified detection and certified recovery. In extensive experiments we show that Demasked Smoothing can on average certify 64% of the pixel predictions for a 1% patch in the detection task and 48% against a 0.5% patch for the recovery task on the ADE20K dataset.

摘要: 对抗性补丁攻击是现实世界深度学习应用面临的一种新的安全威胁。我们提出了去任务平滑，这是第一种(据我们所知)来证明语义分割模型对这种威胁模型的稳健性的方法。以前关于可证明防御补丁攻击的工作主要集中在图像分类任务上，并且经常需要改变模型体系结构和额外的训练，这是不受欢迎的，并且计算代价高昂。在去任务平滑中，任何分割模型都可以在没有特定训练、微调或体系结构限制的情况下应用。使用不同的掩码策略，去掩码平滑可以应用于认证检测和认证恢复。在ADE20K数据集上的大量实验中，对于检测任务中1%的块，去任务平滑平均可以保证64%的像素预测，对于恢复任务，对于0.5%的块，去任务平滑平均可以保证48%的像素预测。



## **30. Adversarial Inter-Group Link Injection Degrades the Fairness of Graph Neural Networks**

对抗性组间链路注入降低了图神经网络的公平性 cs.LG

A shorter version of this work has been accepted by IEEE ICDM 2022

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05957v1)

**Authors**: Hussain Hussain, Meng Cao, Sandipan Sikdar, Denis Helic, Elisabeth Lex, Markus Strohmaier, Roman Kern

**Abstracts**: We present evidence for the existence and effectiveness of adversarial attacks on graph neural networks (GNNs) that aim to degrade fairness. These attacks can disadvantage a particular subgroup of nodes in GNN-based node classification, where nodes of the underlying network have sensitive attributes, such as race or gender. We conduct qualitative and experimental analyses explaining how adversarial link injection impairs the fairness of GNN predictions. For example, an attacker can compromise the fairness of GNN-based node classification by injecting adversarial links between nodes belonging to opposite subgroups and opposite class labels. Our experiments on empirical datasets demonstrate that adversarial fairness attacks can significantly degrade the fairness of GNN predictions (attacks are effective) with a low perturbation rate (attacks are efficient) and without a significant drop in accuracy (attacks are deceptive). This work demonstrates the vulnerability of GNN models to adversarial fairness attacks. We hope our findings raise awareness about this issue in our community and lay a foundation for the future development of GNN models that are more robust to such attacks.

摘要: 我们提出了针对图神经网络(GNN)的对抗性攻击的存在和有效性的证据，这些攻击旨在降低公平性。这些攻击可能使基于GNN的节点分类中的特定节点子组处于不利地位，其中底层网络的节点具有敏感属性，如种族或性别。我们进行了定性和实验分析，解释了敌意链接注入如何损害GNN预测的公平性。例如，攻击者可以通过在属于相反子组和相反类标签的节点之间注入敌对链接来损害基于GNN的节点分类的公平性。我们在经验数据集上的实验表明，对抗性公平攻击能够以较低的扰动率(攻击是有效的)显著降低GNN预测的公平性(攻击是有效的)，并且不会显著降低准确率(攻击是欺骗性的)。这项工作证明了GNN模型对敌意公平攻击的脆弱性。我们希望我们的发现提高我们社区对这个问题的认识，并为未来开发更稳健地抵御此类攻击的GNN模型奠定基础。



## **31. An Evolutionary, Gradient-Free, Query-Efficient, Black-Box Algorithm for Generating Adversarial Instances in Deep Networks**

一种进化、无梯度、查询高效的深层网络对抗性实例生成黑盒算法 cs.CV

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2208.08297v2)

**Authors**: Raz Lapid, Zvika Haramaty, Moshe Sipper

**Abstracts**: Deep neural networks (DNNs) are sensitive to adversarial data in a variety of scenarios, including the black-box scenario, where the attacker is only allowed to query the trained model and receive an output. Existing black-box methods for creating adversarial instances are costly, often using gradient estimation or training a replacement network. This paper introduces \textbf{Qu}ery-Efficient \textbf{E}volutiona\textbf{ry} \textbf{Attack}, \textit{QuEry Attack}, an untargeted, score-based, black-box attack. QuEry Attack is based on a novel objective function that can be used in gradient-free optimization problems. The attack only requires access to the output logits of the classifier and is thus not affected by gradient masking. No additional information is needed, rendering our method more suitable to real-life situations. We test its performance with three different state-of-the-art models -- Inception-v3, ResNet-50, and VGG-16-BN -- against three benchmark datasets: MNIST, CIFAR10 and ImageNet. Furthermore, we evaluate QuEry Attack's performance on non-differential transformation defenses and state-of-the-art robust models. Our results demonstrate the superior performance of QuEry Attack, both in terms of accuracy score and query efficiency.

摘要: 深度神经网络(DNN)对各种场景中的敌意数据很敏感，包括黑盒场景，在这种场景中，攻击者只被允许查询训练的模型并接收输出。现有的创建对抗性实例的黑盒方法成本很高，通常使用梯度估计或训练替换网络。本文介绍了一种非常有效的无目标、基于分数的黑盒攻击查询攻击基于一种新的目标函数，可用于无梯度优化问题。攻击只需要访问分类器的输出逻辑，因此不受梯度掩蔽的影响。不需要额外的信息，使我们的方法更适合实际情况。我们使用三个不同的最先进的模型--先启-v3、ResNet-50和VGG-16-BN--针对三个基准数据集：MNIST、CIFAR10和ImageNet测试其性能。此外，我们还评估了查询攻击在非差分变换防御和现有健壮性模型上的性能。实验结果表明，查询攻击在准确率和查询效率方面都具有较好的性能。



## **32. Bayesian Pseudo Labels: Expectation Maximization for Robust and Efficient Semi-Supervised Segmentation**

贝叶斯伪标签：稳健有效的半监督分割的期望最大化 cs.CV

MICCAI 2022 (Early accept, Student Travel Award)

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2208.04435v3)

**Authors**: Mou-Cheng Xu, Yukun Zhou, Chen Jin, Marius de Groot, Daniel C. Alexander, Neil P. Oxtoby, Yipeng Hu, Joseph Jacob

**Abstracts**: This paper concerns pseudo labelling in segmentation. Our contribution is fourfold. Firstly, we present a new formulation of pseudo-labelling as an Expectation-Maximization (EM) algorithm for clear statistical interpretation. Secondly, we propose a semi-supervised medical image segmentation method purely based on the original pseudo labelling, namely SegPL. We demonstrate SegPL is a competitive approach against state-of-the-art consistency regularisation based methods on semi-supervised segmentation on a 2D multi-class MRI brain tumour segmentation task and a 3D binary CT lung vessel segmentation task. The simplicity of SegPL allows less computational cost comparing to prior methods. Thirdly, we demonstrate that the effectiveness of SegPL may originate from its robustness against out-of-distribution noises and adversarial attacks. Lastly, under the EM framework, we introduce a probabilistic generalisation of SegPL via variational inference, which learns a dynamic threshold for pseudo labelling during the training. We show that SegPL with variational inference can perform uncertainty estimation on par with the gold-standard method Deep Ensemble.

摘要: 本文研究的是分割中的伪标注问题。我们的贡献是四倍的。首先，我们提出了一种新的伪标记公式，作为一种用于清晰统计解释的期望最大化(EM)算法。其次，提出了一种完全基于原始伪标记的半监督医学图像分割方法--SegPL。在2D多类MRI脑肿瘤分割任务和3D二值CT肺血管分割任务中，我们证明了SegPL是一种与最先进的基于一致性正则化的半监督分割方法相竞争的方法。与以前的方法相比，SegPL的简单性允许更少的计算成本。第三，我们证明了SegPL的有效性可能源于它对分布外噪声和对手攻击的健壮性。最后，在EM框架下，我们通过变分推理对SegPL进行概率推广，在训练过程中学习伪标签的动态阈值。我们证明了带变分推理的SegPL方法可以与金标准方法深层集成一样进行不确定度估计。



## **33. Adversarial Coreset Selection for Efficient Robust Training**

用于高效稳健训练的对抗性同位重置选择 cs.LG

Extended version of the ECCV2022 paper: arXiv:2112.00378. arXiv admin  note: substantial text overlap with arXiv:2112.00378

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05785v1)

**Authors**: Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie

**Abstracts**: Neural networks are vulnerable to adversarial attacks: adding well-crafted, imperceptible perturbations to their input can modify their output. Adversarial training is one of the most effective approaches to training robust models against such attacks. Unfortunately, this method is much slower than vanilla training of neural networks since it needs to construct adversarial examples for the entire training data at every iteration. By leveraging the theory of coreset selection, we show how selecting a small subset of training data provides a principled approach to reducing the time complexity of robust training. To this end, we first provide convergence guarantees for adversarial coreset selection. In particular, we show that the convergence bound is directly related to how well our coresets can approximate the gradient computed over the entire training data. Motivated by our theoretical analysis, we propose using this gradient approximation error as our adversarial coreset selection objective to reduce the training set size effectively. Once built, we run adversarial training over this subset of the training data. Unlike existing methods, our approach can be adapted to a wide variety of training objectives, including TRADES, $\ell_p$-PGD, and Perceptual Adversarial Training. We conduct extensive experiments to demonstrate that our approach speeds up adversarial training by 2-3 times while experiencing a slight degradation in the clean and robust accuracy.

摘要: 神经网络很容易受到敌意攻击：在它们的输入中添加精心设计的、不可察觉的扰动可以修改它们的输出。对抗性训练是训练抵抗此类攻击的稳健模型的最有效方法之一。遗憾的是，这种方法比普通的神经网络训练要慢得多，因为它需要在每次迭代中为整个训练数据构造对抗性样本。通过利用核心选择理论，我们展示了如何选择一小部分训练数据提供了一种原则性的方法来降低稳健训练的时间复杂性。为此，我们首先为对抗性核心重置选择提供收敛保证。特别是，我们证明了收敛界与我们的核集在整个训练数据上计算的梯度的逼近程度直接相关。在理论分析的基础上，我们提出了利用这一梯度逼近误差作为对抗性核心集选择的目标，以有效地减少训练集的规模。一旦构建，我们就对训练数据的这个子集进行对抗性训练。与现有方法不同，我们的方法可以适应广泛的训练目标，包括交易、$\ell_p$-PGD和感知对手训练。我们进行了大量的实验，证明了我们的方法将对手训练速度提高了2-3倍，同时经历了干净和健壮的准确性的轻微下降。



## **34. Adaptive Perturbation Generation for Multiple Backdoors Detection**

多后门检测的自适应扰动生成 cs.CV

7 pages, 5 figures

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05244v2)

**Authors**: Yuhang Wang, Huafeng Shi, Rui Min, Ruijia Wu, Siyuan Liang, Yichao Wu, Ding Liang, Aishan Liu

**Abstracts**: Extensive evidence has demonstrated that deep neural networks (DNNs) are vulnerable to backdoor attacks, which motivates the development of backdoor detection methods. Existing backdoor detection methods are typically tailored for backdoor attacks with individual specific types (e.g., patch-based or perturbation-based). However, adversaries are likely to generate multiple types of backdoor attacks in practice, which challenges the current detection strategies. Based on the fact that adversarial perturbations are highly correlated with trigger patterns, this paper proposes the Adaptive Perturbation Generation (APG) framework to detect multiple types of backdoor attacks by adaptively injecting adversarial perturbations. Since different trigger patterns turn out to show highly diverse behaviors under the same adversarial perturbations, we first design the global-to-local strategy to fit the multiple types of backdoor triggers via adjusting the region and budget of attacks. To further increase the efficiency of perturbation injection, we introduce a gradient-guided mask generation strategy to search for the optimal regions for adversarial attacks. Extensive experiments conducted on multiple datasets (CIFAR-10, GTSRB, Tiny-ImageNet) demonstrate that our method outperforms state-of-the-art baselines by large margins(+12%).

摘要: 大量证据表明，深度神经网络(DNN)很容易受到后门攻击，这促使了后门检测方法的发展。现有的后门检测方法通常是为个别特定类型的后门攻击量身定做的(例如，基于补丁或基于扰动)。然而，攻击者在实践中可能会产生多种类型的后门攻击，这对现有的检测策略提出了挑战。基于敌意扰动与触发模式高度相关的事实，提出了自适应扰动生成(APG)框架，通过自适应注入敌意扰动来检测多种类型的后门攻击。由于不同的触发模式在相同的对抗性扰动下表现出高度不同的行为，我们首先设计了全局到局部的策略，通过调整攻击的区域和预算来适应多种类型的后门触发。为了进一步提高扰动注入的效率，我们引入了一种梯度引导的掩码生成策略来搜索对抗性攻击的最优区域。在多个数据集(CIFAR-10，GTSRB，Tiny-ImageNet)上进行的大量实验表明，我们的方法比最先进的基线方法有很大的优势(+12%)。



## **35. A Tale of HodgeRank and Spectral Method: Target Attack Against Rank Aggregation Is the Fixed Point of Adversarial Game**

HodgeRank和谱方法的故事：针对等级聚集的目标攻击是对抗性游戏的固定点 cs.LG

33 pages,  https://github.com/alphaprime/Target_Attack_Rank_Aggregation

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05742v1)

**Authors**: Ke Ma, Qianqian Xu, Jinshan Zeng, Guorong Li, Xiaochun Cao, Qingming Huang

**Abstracts**: Rank aggregation with pairwise comparisons has shown promising results in elections, sports competitions, recommendations, and information retrieval. However, little attention has been paid to the security issue of such algorithms, in contrast to numerous research work on the computational and statistical characteristics. Driven by huge profits, the potential adversary has strong motivation and incentives to manipulate the ranking list. Meanwhile, the intrinsic vulnerability of the rank aggregation methods is not well studied in the literature. To fully understand the possible risks, we focus on the purposeful adversary who desires to designate the aggregated results by modifying the pairwise data in this paper. From the perspective of the dynamical system, the attack behavior with a target ranking list is a fixed point belonging to the composition of the adversary and the victim. To perform the targeted attack, we formulate the interaction between the adversary and the victim as a game-theoretic framework consisting of two continuous operators while Nash equilibrium is established. Then two procedures against HodgeRank and RankCentrality are constructed to produce the modification of the original data. Furthermore, we prove that the victims will produce the target ranking list once the adversary masters the complete information. It is noteworthy that the proposed methods allow the adversary only to hold incomplete information or imperfect feedback and perform the purposeful attack. The effectiveness of the suggested target attack strategies is demonstrated by a series of toy simulations and several real-world data experiments. These experimental results show that the proposed methods could achieve the attacker's goal in the sense that the leading candidate of the perturbed ranking list is the designated one by the adversary.

摘要: 在选举、体育竞赛、推荐和信息检索等领域，采用配对比较的排名聚合方法已显示出良好的效果。然而，与大量关于算法的计算和统计特性的研究工作相比，对这类算法的安全问题关注较少。在巨额利润的驱动下，潜在对手操纵排行榜的动机和动机很强。同时，文献中对秩聚类方法的内在脆弱性的研究还不够深入。为了充分理解可能的风险，我们将重点放在有目的的对手身上，他们希望通过修改成对数据来指定聚合结果。从动力系统的角度来看，具有目标排行榜的攻击行为是属于对手和受害者组成的固定点。为了进行有针对性的攻击，我们将对手和受害者之间的相互作用描述为一个由两个连续算子组成的博弈论框架，同时建立了纳什均衡。然后构造了针对HodgeRank和RankCentrality的两个过程来产生对原始数据的修改。此外，我们证明了一旦对手掌握了完整的信息，受害者就会产生目标排名表。值得注意的是，所提出的方法只允许对手持有不完全信息或不完全反馈，并执行有目的的攻击。通过一系列玩具仿真和几个真实世界的数据实验，验证了所提出的目标攻击策略的有效性。实验结果表明，所提出的方法能够达到攻击者的目的，即扰动排序列表的领先候选者就是对手指定的候选者。



## **36. Sample Complexity of an Adversarial Attack on UCB-based Best-arm Identification Policy**

基于UCB的最佳武器识别策略下对抗性攻击的样本复杂性 cs.LG

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05692v1)

**Authors**: Varsha Pendyala

**Abstracts**: In this work I study the problem of adversarial perturbations to rewards, in a Multi-armed bandit (MAB) setting. Specifically, I focus on an adversarial attack to a UCB type best-arm identification policy applied to a stochastic MAB. The UCB attack presented in [1] results in pulling a target arm K very often. I used the attack model of [1] to derive the sample complexity required for selecting target arm K as the best arm. I have proved that the stopping condition of UCB based best-arm identification algorithm given in [2], can be achieved by the target arm K in T rounds, where T depends only on the total number of arms and $\sigma$ parameter of $\sigma^2-$ sub-Gaussian random rewards of the arms.

摘要: 在这项工作中，我研究了多臂强盗(MAB)环境下的对抗性报酬摄动问题。具体地说，我将重点放在对应用于随机MAB的UCB类型的最佳ARM识别策略的对抗性攻击上。文[1]中提出的UCB攻击导致经常拉动目标手臂K。我使用了[1]的攻击模型来推导出选择目标手臂K作为最佳手臂所需的样本复杂度。证明了文[2]中给出的基于UCB的最佳手臂识别算法的停止条件可以由目标手臂K在T轮中实现，其中T仅取决于手臂的总数和手臂的$\sigma^2-$亚高斯随机奖励的$\sigma参数。



## **37. Replay-based Recovery for Autonomous Robotic Vehicles from Sensor Deception Attacks**

基于重放的自主机器人对传感器欺骗攻击的恢复 cs.RO

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.04554v2)

**Authors**: Pritam Dash, Guanpeng Li, Mehdi Karimibiuki, Karthik Pattabiraman

**Abstracts**: Sensors are crucial for autonomous operation in robotic vehicles (RV). Physical attacks on sensors such as sensor tampering or spoofing can feed erroneous values to RVs through physical channels, which results in mission failures. In this paper, we present DeLorean, a comprehensive diagnosis and recovery framework for securing autonomous RVs from physical attacks. We consider a strong form of physical attack called sensor deception attacks (SDAs), in which the adversary targets multiple sensors of different types simultaneously (even including all sensors). Under SDAs, DeLorean inspects the attack induced errors, identifies the targeted sensors, and prevents the erroneous sensor inputs from being used in RV's feedback control loop. DeLorean replays historic state information in the feedback control loop and recovers the RV from attacks. Our evaluation on four real and two simulated RVs shows that DeLorean can recover RVs from different attacks, and ensure mission success in 94% of the cases (on average), without any crashes. DeLorean incurs low performance, memory and battery overheads.

摘要: 传感器对于机器人车辆(RV)的自主操作至关重要。对传感器的物理攻击，如传感器篡改或欺骗，可能会通过物理通道向房车提供错误的值，从而导致任务失败。在本文中，我们提出了DeLorean，一个全面的诊断和恢复框架，用于保护自主房车免受物理攻击。我们考虑了一种称为传感器欺骗攻击(SDA)的强物理攻击形式，在这种攻击中，对手同时针对不同类型的多个传感器(甚至包括所有传感器)。在SDAS下，DeLorean检查攻击导致的错误，识别目标传感器，并防止错误的传感器输入用于房车的反馈控制回路。DeLorean在反馈控制环路中重放历史状态信息，并恢复RV免受攻击。我们对四辆真实房车和两辆模拟房车的评估表明，DeLorean可以从不同的攻击中恢复房车，并确保94%的任务成功(平均而言)，而不会发生任何崩溃。DeLorean的性能、内存和电池开销都很低。



## **38. Boosting Robustness Verification of Semantic Feature Neighborhoods**

增强语义特征邻域的健壮性验证 cs.LG

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2209.05446v1)

**Authors**: Anan Kabaha, Dana Drachsler-Cohen

**Abstracts**: Deep neural networks have been shown to be vulnerable to adversarial attacks that perturb inputs based on semantic features. Existing robustness analyzers can reason about semantic feature neighborhoods to increase the networks' reliability. However, despite the significant progress in these techniques, they still struggle to scale to deep networks and large neighborhoods. In this work, we introduce VeeP, an active learning approach that splits the verification process into a series of smaller verification steps, each is submitted to an existing robustness analyzer. The key idea is to build on prior steps to predict the next optimal step. The optimal step is predicted by estimating the certification velocity and sensitivity via parametric regression. We evaluate VeeP on MNIST, Fashion-MNIST, CIFAR-10 and ImageNet and show that it can analyze neighborhoods of various features: brightness, contrast, hue, saturation, and lightness. We show that, on average, given a 90 minute timeout, VeeP verifies 96% of the maximally certifiable neighborhoods within 29 minutes, while existing splitting approaches verify, on average, 73% of the maximally certifiable neighborhoods within 58 minutes.

摘要: 深度神经网络已被证明容易受到基于语义特征的输入扰乱的对抗性攻击。现有的健壮性分析器可以对语义特征邻域进行推理，以增加网络的可靠性。然而，尽管这些技术取得了重大进展，它们仍难以扩展到深度网络和大型社区。在这项工作中，我们引入了VeEP，这是一种主动学习方法，它将验证过程划分为一系列较小的验证步骤，每个步骤都提交给现有的健壮性分析器。其关键思想是建立在先前步骤的基础上，以预测下一个最佳步骤。通过参数回归估计认证速度和灵敏度，预测最优步骤。我们在MNIST、Fashion-MNIST、CIFAR-10和ImageNet上对Veep进行了评估，结果表明，它可以分析各种特征的社区：亮度、对比度、色调、饱和度和亮度。我们发现，在平均90分钟的超时时间内，Veep在29分钟内验证了96%的最大可证明邻域，而现有的分割方法平均在58分钟内验证了73%的最大可证明邻域。



## **39. Bilevel Optimization with a Lower-level Contraction: Optimal Sample Complexity without Warm-Start**

低水平收缩的两层优化：无热启动的最优样本复杂性 stat.ML

35 pages, 2 figures. Code at  https://github.com/CSML-IIT-UCL/bioptexps

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2202.03397v2)

**Authors**: Riccardo Grazzi, Massimiliano Pontil, Saverio Salzo

**Abstracts**: We analyze a general class of bilevel problems, in which the upper-level problem consists in the minimization of a smooth objective function and the lower-level problem is to find the fixed point of a smooth contraction map. This type of problems include instances of meta-learning, equilibrium models, hyperparameter optimization and data poisoning adversarial attacks. Several recent works have proposed algorithms which warm-start the lower level problem, i.e. they use the previous lower-level approximate solution as a staring point for the lower-level solver. This warm-start procedure allows one to improve the sample complexity in both the stochastic and deterministic settings, achieving in some cases the order-wise optimal sample complexity. However, there are situations, e.g., meta learning and equilibrium models, in which the warm-start procedure is not well-suited or ineffective. In this work we show that without warm-start, it is still possible to achieve order-wise optimal or near-optimal sample complexity. In particular, we propose a simple method which uses stochastic fixed point iterations at the lower-level and projected inexact gradient descent at the upper-level, that reaches an $\epsilon$-stationary point using $O(\epsilon^{-2})$ and $\tilde{O}(\epsilon^{-1})$ samples for the stochastic and the deterministic setting, respectively. Finally, compared to methods using warm-start, our approach yields a simpler analysis that does not need to study the coupled interactions between the upper-level and lower-level iterates

摘要: 我们分析了一类一般的两层问题，其中上层问题在于光滑目标函数的最小化，下层问题是寻找光滑压缩映射的不动点。这类问题包括元学习、均衡模型、超参数优化和数据中毒攻击等实例。最近的一些工作提出了暖启动下层问题的算法，即使用先前的下层近似解作为下层求解器的起始点。这种热启动过程允许人们在随机和确定性设置下改善样本复杂性，在某些情况下实现顺序最优的样本复杂性。然而，在一些情况下，例如元学习和平衡模型，热启动程序不是很适合或无效的。在这项工作中，我们证明了在没有热启动的情况下，仍然有可能获得按顺序最优或接近最优的样本复杂度。特别地，我们提出了一种简单的方法，它在低层使用随机不动点迭代，在上层使用投影的不精确梯度下降，分别在随机和确定环境下使用$O(epsilon^{-2})$和$tide{O}(epsilon^{-1})$样本达到$epsilon$-驻点。最后，与使用热启动的方法相比，我们的方法产生了更简单的分析，不需要研究上层和下层迭代之间的耦合作用



## **40. Fiat-Shamir for Proofs Lacks a Proof Even in the Presence of Shared Entanglement**

菲亚特-沙米尔的证据缺乏证据，即使在存在共同纠缠的情况下 quant-ph

62 pages, 2 figures

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2204.02265v2)

**Authors**: Frédéric Dupuis, Philippe Lamontagne, Louis Salvail

**Abstracts**: We explore the cryptographic power of arbitrary shared physical resources. The most general such resource is access to a fresh entangled quantum state at the outset of each protocol execution. We call this the Common Reference Quantum State (CRQS) model, in analogy to the well-known Common Reference String (CRS). The CRQS model is a natural generalization of the CRS model but appears to be more powerful: in the two-party setting, a CRQS can sometimes exhibit properties associated with a Random Oracle queried once by measuring a maximally entangled state in one of many mutually unbiased bases. We formalize this notion as a Weak One-Time Random Oracle (WOTRO), where we only ask of the $m$--bit output to have some randomness when conditioned on the $n$--bit input.   We show that when $n-m\in\omega(\lg n)$, any protocol for WOTRO in the CRQS model can be attacked by an (inefficient) adversary. Moreover, our adversary is efficiently simulatable, which rules out the possibility of proving the computational security of a scheme by a black-box reduction to a cryptographic game assumption. On the other hand, we introduce a non-game quantum assumption for hash functions that implies WOTRO in the CRQ\$ model (where the CRQS consists only of EPR pairs). We first build a statistically secure WOTRO protocol where $m=n$, then hash the output.   The impossibility of WOTRO has the following consequences. First, we show the black-box impossibility of a quantum Fiat-Shamir transform, extending the impossibility result of Bitansky et al. (TCC '13) to the CRQS model. Second, we show a black-box impossibility result for a strenghtened version of quantum lightning (Zhandry, Eurocrypt '19) where quantum bolts have an additional parameter that cannot be changed without generating new bolts.

摘要: 我们探索任意共享物理资源的加密能力。最常见的这类资源是在每个协议执行开始时访问新的纠缠量子态。我们称之为公共参考量子态(CRQS)模型，类似于众所周知的公共参考弦(CRS)。CRQS模型是CRS模型的自然推广，但似乎更强大：在两方设置中，CRQS有时可以通过测量许多相互无偏的碱基之一中的最大纠缠态来展示与查询一次的随机Oracle相关联的属性。我们将这个概念形式化为弱一次性随机Oracle(WOTRO)，其中我们只要求$m$位的输出在以$n$位输入为条件时具有一定的随机性。我们证明了当$n-m\in\omega(\lg n)$时，CRQS模型中用于WOTRO的任何协议都可以被(低效的)攻击者攻击。此外，我们的对手是高效可模拟的，这排除了通过将黑盒简化为密码博弈假设来证明方案的计算安全性的可能性。另一方面，我们引入了散列函数的非博弈量子假设，在CRQ模型(其中CRQS只由EPR对组成)中隐含了WOTRO。我们首先构建一个统计安全的WOTRO协议，其中$m=n$，然后对输出进行散列。WOTRO的不可能性会产生以下后果。首先，我们证明了量子Fiat-Shamir变换的黑盒不可能性，推广了Bitansky等人的不可能结果。其次，我们给出了一个加强版量子闪电(Zhandry，Eurocrypt‘19)的黑箱不可能结果，其中量子闪电有一个额外的参数，如果不产生新的闪电，这个参数就不能改变。



## **41. A Survey of Machine Unlearning**

机器遗忘研究综述 cs.LG

fixed overlaps

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2209.02299v4)

**Authors**: Thanh Tam Nguyen, Thanh Trung Huynh, Phi Le Nguyen, Alan Wee-Chung Liew, Hongzhi Yin, Quoc Viet Hung Nguyen

**Abstracts**: Computer systems hold a large amount of personal data over decades. On the one hand, such data abundance allows breakthroughs in artificial intelligence (AI), especially machine learning (ML) models. On the other hand, it can threaten the privacy of users and weaken the trust between humans and AI. Recent regulations require that private information about a user can be removed from computer systems in general and from ML models in particular upon request (e.g. the "right to be forgotten"). While removing data from back-end databases should be straightforward, it is not sufficient in the AI context as ML models often "remember" the old data. Existing adversarial attacks proved that we can learn private membership or attributes of the training data from the trained models. This phenomenon calls for a new paradigm, namely machine unlearning, to make ML models forget about particular data. It turns out that recent works on machine unlearning have not been able to solve the problem completely due to the lack of common frameworks and resources. In this survey paper, we seek to provide a thorough investigation of machine unlearning in its definitions, scenarios, mechanisms, and applications. Specifically, as a categorical collection of state-of-the-art research, we hope to provide a broad reference for those seeking a primer on machine unlearning and its various formulations, design requirements, removal requests, algorithms, and uses in a variety of ML applications. Furthermore, we hope to outline key findings and trends in the paradigm as well as highlight new areas of research that have yet to see the application of machine unlearning, but could nonetheless benefit immensely. We hope this survey provides a valuable reference for ML researchers as well as those seeking to innovate privacy technologies. Our resources are at https://github.com/tamlhp/awesome-machine-unlearning.

摘要: 几十年来，计算机系统保存着大量的个人数据。一方面，这样的数据丰富使人工智能(AI)，特别是机器学习(ML)模型取得了突破。另一方面，它会威胁用户的隐私，削弱人类与AI之间的信任。最近的法规要求，一般情况下，可以从计算机系统中删除关于用户的私人信息，特别是在请求时可以从ML模型中删除用户的私人信息(例如，“被遗忘权”)。虽然从后端数据库中删除数据应该很简单，但在人工智能环境中这是不够的，因为ML模型经常“记住”旧数据。现有的对抗性攻击证明，我们可以从训练好的模型中学习训练数据的私人成员或属性。这种现象呼唤一种新的范式，即机器遗忘，以使ML模型忘记特定的数据。事实证明，由于缺乏通用的框架和资源，最近关于机器遗忘的研究并不能完全解决这个问题。在这篇调查论文中，我们试图对机器遗忘的定义、场景、机制和应用进行全面的调查。具体地说，作为最新研究的分类集合，我们希望为那些寻求机器遗忘及其各种公式、设计要求、移除请求、算法和在各种ML应用中使用的入门知识的人提供广泛的参考。此外，我们希望概述该范式中的主要发现和趋势，并强调尚未看到机器遗忘应用的新研究领域，但仍可能受益匪浅。我们希望这项调查为ML研究人员以及那些寻求创新隐私技术的人提供有价值的参考。我们的资源在https://github.com/tamlhp/awesome-machine-unlearning.



## **42. GRNN: Generative Regression Neural Network -- A Data Leakage Attack for Federated Learning**

GRNN：生成回归神经网络--一种面向联邦学习的数据泄漏攻击 cs.LG

The source code can be found at: https://github.com/Rand2AI/GRNN

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2105.00529v3)

**Authors**: Hanchi Ren, Jingjing Deng, Xianghua Xie

**Abstracts**: Data privacy has become an increasingly important issue in Machine Learning (ML), where many approaches have been developed to tackle this challenge, e.g. cryptography (Homomorphic Encryption (HE), Differential Privacy (DP), etc.) and collaborative training (Secure Multi-Party Computation (MPC), Distributed Learning and Federated Learning (FL)). These techniques have a particular focus on data encryption or secure local computation. They transfer the intermediate information to the third party to compute the final result. Gradient exchanging is commonly considered to be a secure way of training a robust model collaboratively in Deep Learning (DL). However, recent researches have demonstrated that sensitive information can be recovered from the shared gradient. Generative Adversarial Network (GAN), in particular, has shown to be effective in recovering such information. However, GAN based techniques require additional information, such as class labels which are generally unavailable for privacy-preserved learning. In this paper, we show that, in the FL system, image-based privacy data can be easily recovered in full from the shared gradient only via our proposed Generative Regression Neural Network (GRNN). We formulate the attack to be a regression problem and optimize two branches of the generative model by minimizing the distance between gradients. We evaluate our method on several image classification tasks. The results illustrate that our proposed GRNN outperforms state-of-the-art methods with better stability, stronger robustness, and higher accuracy. It also has no convergence requirement to the global FL model. Moreover, we demonstrate information leakage using face re-identification. Some defense strategies are also discussed in this work.

摘要: 数据隐私已经成为机器学习(ML)中一个日益重要的问题，人们已经开发了许多方法来应对这一挑战，例如密码学(同态加密(HE)、差分隐私(DP)等)。和协作培训(安全多方计算(MPC)、分布式学习和联合学习(FL))。这些技术特别关注数据加密或安全本地计算。他们将中间信息传递给第三方来计算最终结果。梯度交换通常被认为是深度学习中协作训练健壮模型的一种安全方式。然而，最近的研究表明，敏感信息可以从共享梯度中恢复出来。尤其是生成性对抗网络(GAN)在恢复这类信息方面是有效的。然而，基于GaN的技术需要额外的信息，例如类别标签，这些信息通常不能用于隐私保护学习。在本文中，我们证明，在FL系统中，仅通过我们提出的生成回归神经网络(GRNN)就可以很容易地从共享梯度中完全恢复基于图像的隐私数据。我们将攻击描述为一个回归问题，并通过最小化梯度之间的距离来优化生成模型的两个分支。我们在几个图像分类任务上对我们的方法进行了评估。实验结果表明，本文提出的GRNN方法在稳定性、鲁棒性和准确率等方面均优于现有的方法。它对全局FL模型也没有收敛要求。此外，我们还使用人脸重新识别来演示信息泄漏。文中还讨论了一些防御策略。



## **43. Semantic-Preserving Adversarial Code Comprehension**

保留语义的对抗性代码理解 cs.CL

Accepted by COLING 2022

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2209.05130v1)

**Authors**: Yiyang Li, Hongqiu Wu, Hai Zhao

**Abstracts**: Based on the tremendous success of pre-trained language models (PrLMs) for source code comprehension tasks, current literature studies either ways to further improve the performance (generalization) of PrLMs, or their robustness against adversarial attacks. However, they have to compromise on the trade-off between the two aspects and none of them consider improving both sides in an effective and practical way. To fill this gap, we propose Semantic-Preserving Adversarial Code Embeddings (SPACE) to find the worst-case semantic-preserving attacks while forcing the model to predict the correct labels under these worst cases. Experiments and analysis demonstrate that SPACE can stay robust against state-of-the-art attacks while boosting the performance of PrLMs for code.

摘要: 基于预先训练的语言模型在源代码理解任务中的巨大成功，目前的文献研究要么是进一步提高预先训练的语言模型的性能(泛化)，要么是研究它们对对手攻击的健壮性。然而，他们不得不在这两个方面的权衡上妥协，没有一个人考虑以有效和实际的方式改善双方。为了填补这一空白，我们提出了保持语义的对抗性代码嵌入(SPACE)，以发现最坏情况下保持语义的攻击，同时迫使模型在这些最坏情况下预测正确的标签。实验和分析表明，SPACE在提高PrLMS代码性能的同时，可以保持对最先进攻击的健壮性。



## **44. Passive Triangulation Attack on ORide**

ORIDE上的被动三角剖分攻击 cs.CR

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2208.12216v2)

**Authors**: Shyam Murthy, Srinivas Vivek

**Abstracts**: Privacy preservation in Ride Hailing Services is intended to protect privacy of drivers and riders. ORide is one of the early RHS proposals published at USENIX Security Symposium 2017. In the ORide protocol, riders and drivers, operating in a zone, encrypt their locations using a Somewhat Homomorphic Encryption scheme (SHE) and forward them to the Service Provider (SP). SP homomorphically computes the squared Euclidean distance between riders and available drivers. Rider receives the encrypted distances and selects the optimal rider after decryption. In order to prevent a triangulation attack, SP randomly permutes the distances before sending them to the rider. In this work, we use propose a passive attack that uses triangulation to determine coordinates of all participating drivers whose permuted distances are available from the points of view of multiple honest-but-curious adversary riders. An attack on ORide was published at SAC 2021. The same paper proposes a countermeasure using noisy Euclidean distances to thwart their attack. We extend our attack to determine locations of drivers when given their permuted and noisy Euclidean distances from multiple points of reference, where the noise perturbation comes from a uniform distribution. We conduct experiments with different number of drivers and for different perturbation values. Our experiments show that we can determine locations of all drivers participating in the ORide protocol. For the perturbed distance version of the ORide protocol, our algorithm reveals locations of about 25% to 50% of participating drivers. Our algorithm runs in time polynomial in number of drivers.

摘要: 网约车服务中的隐私保护旨在保护司机和乘客的隐私。ORIDE是USENIX安全研讨会2017上发布的早期RHS提案之一。在ORIDE协议中，在区域中操作的乘客和司机使用某种同态加密方案(SHE)加密他们的位置，并将其转发给服务提供商(SP)。SP同态计算乘客和可用司机之间的平方欧几里得距离。骑手收到加密的距离，解密后选择最优的骑手。为了防止三角测量攻击，SP在将距离发送给骑手之前随机排列距离。在这项工作中，我们使用了一种被动攻击，该攻击使用三角测量来确定所有参与的司机的坐标，这些司机的置换距离是从多个诚实但好奇的对手车手的角度出发的。对ORide的攻击在SAC 2021上发表。同时提出了一种利用噪声欧几里德距离来阻止他们攻击的对策。当给定司机与多个参考点的置换和噪声欧几里德距离时，我们将我们的攻击扩展到确定司机的位置，其中噪声扰动来自均匀分布。我们对不同数量的驱动器和不同的摄动值进行了实验。我们的实验表明，我们可以确定所有参与ORIDE协议的司机的位置。对于受干扰的距离版本的ORide协议，我们的算法显示了大约25%到50%的参与司机的位置。我们的算法以时间多项式的形式运行在驱动器的数量上。



## **45. CARE: Certifiably Robust Learning with Reasoning via Variational Inference**

注意：通过变分推理进行推理的可证明稳健学习 cs.LG

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2209.05055v1)

**Authors**: Jiawei Zhang, Linyi Li, Ce Zhang, Bo Li

**Abstracts**: Despite great recent advances achieved by deep neural networks (DNNs), they are often vulnerable to adversarial attacks. Intensive research efforts have been made to improve the robustness of DNNs; however, most empirical defenses can be adaptively attacked again, and the theoretically certified robustness is limited, especially on large-scale datasets. One potential root cause of such vulnerabilities for DNNs is that although they have demonstrated powerful expressiveness, they lack the reasoning ability to make robust and reliable predictions. In this paper, we aim to integrate domain knowledge to enable robust learning with the reasoning paradigm. In particular, we propose a certifiably robust learning with reasoning pipeline (CARE), which consists of a learning component and a reasoning component. Concretely, we use a set of standard DNNs to serve as the learning component to make semantic predictions, and we leverage the probabilistic graphical models, such as Markov logic networks (MLN), to serve as the reasoning component to enable knowledge/logic reasoning. However, it is known that the exact inference of MLN (reasoning) is #P-complete, which limits the scalability of the pipeline. To this end, we propose to approximate the MLN inference via variational inference based on an efficient expectation maximization algorithm. In particular, we leverage graph convolutional networks (GCNs) to encode the posterior distribution during variational inference and update the parameters of GCNs (E-step) and the weights of knowledge rules in MLN (M-step) iteratively. We conduct extensive experiments on different datasets and show that CARE achieves significantly higher certified robustness compared with the state-of-the-art baselines. We additionally conducted different ablation studies to demonstrate the empirical robustness of CARE and the effectiveness of different knowledge integration.

摘要: 尽管深度神经网络(DNN)最近取得了很大的进展，但它们往往容易受到对手的攻击。人们已经进行了大量的研究来提高DNN的稳健性，然而，大多数经验防御都可以再次自适应攻击，理论上证明的健壮性是有限的，特别是在大规模数据集上。DNN这种漏洞的一个潜在根本原因是，尽管它们表现出强大的表现力，但它们缺乏做出稳健和可靠预测的推理能力。在本文中，我们的目标是将领域知识集成到推理范式中，以实现稳健的学习。特别地，我们提出了一种带推理的可证明稳健学习流水线(CARE)，该流水线由学习组件和推理组件组成。具体地说，我们使用一组标准的DNN作为学习组件进行语义预测，并利用马尔可夫逻辑网络(MLN)等概率图形模型作为推理组件来实现知识/逻辑推理。然而，众所周知，MLN(推理)的精确推理是#P-完全的，这限制了流水线的可扩展性。为此，我们提出了基于一种有效的期望最大化算法的变分推理来逼近最大似然推理。特别是，我们利用图卷积网络(GCNS)对变分推理过程中的后验分布进行编码，并迭代地更新GCNS的参数(E步)和MLN中知识规则的权值(M步)。我们在不同的数据集上进行了广泛的实验，并表明与最先进的基线相比，CARE实现了显著更高的认证稳健性。此外，我们还进行了不同的消融研究，以证明CARE的经验稳健性和不同知识整合的有效性。



## **46. GFCL: A GRU-based Federated Continual Learning Framework against Data Poisoning Attacks in IoV**

GFCL：一种基于GRU的联合持续学习框架对抗IoV中的数据中毒攻击 cs.LG

11 pages, 12 figures, 3 tables; This work has been submitted to the  IEEE Transactions on Vehicular Technology for possible publication. Copyright  may be transferred without notice, after which this version may no longer be  accessible

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2204.11010v2)

**Authors**: Anum Talpur, Mohan Gurusamy

**Abstracts**: Integration of machine learning (ML) in 5G-based Internet of Vehicles (IoV) networks has enabled intelligent transportation and smart traffic management. Nonetheless, the security against adversarial poisoning attacks is also increasingly becoming a challenging task. Specifically, Deep Reinforcement Learning (DRL) is one of the widely used ML designs in IoV applications. The standard ML security techniques are not effective in DRL where the algorithm learns to solve sequential decision-making through continuous interaction with the environment, and the environment is time-varying, dynamic, and mobile. In this paper, we propose a Gated Recurrent Unit (GRU)-based federated continual learning (GFCL) anomaly detection framework against Sybil-based data poisoning attacks in IoV. The objective is to present a lightweight and scalable framework that learns and detects the illegitimate behavior without having a-priori training dataset consisting of attack samples. We use GRU to predict a future data sequence to analyze and detect illegitimate behavior from vehicles in a federated learning-based distributed manner. We investigate the performance of our framework using real-world vehicle mobility traces. The results demonstrate the effectiveness of our proposed solution in terms of different performance metrics.

摘要: 将机器学习(ML)集成到基于5G的车联网(IoV)网络中，实现了智能交通和智能交通管理。尽管如此，防御对抗性中毒攻击的安全也日益成为一项具有挑战性的任务。其中，深度强化学习(DRL)是IoV应用中广泛使用的ML设计之一。标准的ML安全技术在DRL中并不有效，在DRL中，算法通过与环境的持续交互来学习解决顺序决策，并且环境是时变的、动态的和移动的。针对物联网中基于Sybil的数据中毒攻击，提出了一种基于门控递归单元(GRU)的联合连续学习(GFCL)异常检测框架。其目的是提供一个轻量级和可扩展的框架，在没有包含攻击样本的先验训练数据集的情况下学习和检测非法行为。我们使用GRU来预测未来的数据序列，以基于联合学习的分布式方式来分析和检测车辆的非法行为。我们使用真实世界的车辆移动轨迹来研究我们的框架的性能。结果表明，在不同的性能指标下，我们提出的解决方案是有效的。



## **47. Generate novel and robust samples from data: accessible sharing without privacy concerns**

从数据中生成新颖且可靠的样本：无隐私问题的可访问共享 cs.LG

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2209.06113v1)

**Authors**: David Banh, Alan Huang

**Abstracts**: Generating new samples from data sets can mitigate extra expensive operations, increased invasive procedures, and mitigate privacy issues. These novel samples that are statistically robust can be used as a temporary and intermediate replacement when privacy is a concern. This method can enable better data sharing practices without problems relating to identification issues or biases that are flaws for an adversarial attack.

摘要: 从数据集生成新样本可以减少额外昂贵的操作、增加侵入性程序，并缓解隐私问题。当隐私受到关注时，这些在统计上稳健的新样本可以用作临时和中间替代。这种方法可以实现更好的数据共享实践，而不会出现与识别问题或作为对抗性攻击缺陷的偏见有关的问题。



## **48. Resisting Deep Learning Models Against Adversarial Attack Transferability via Feature Randomization**

基于特征随机化的抗敌意攻击传递的深度学习模型 cs.CR

**SubmitDate**: 2022-09-11    [paper-pdf](http://arxiv.org/pdf/2209.04930v1)

**Authors**: Ehsan Nowroozi, Mohammadreza Mohammadi, Pargol Golmohammadi, Yassine Mekdad, Mauro Conti, Selcuk Uluagac

**Abstracts**: In the past decades, the rise of artificial intelligence has given us the capabilities to solve the most challenging problems in our day-to-day lives, such as cancer prediction and autonomous navigation. However, these applications might not be reliable if not secured against adversarial attacks. In addition, recent works demonstrated that some adversarial examples are transferable across different models. Therefore, it is crucial to avoid such transferability via robust models that resist adversarial manipulations. In this paper, we propose a feature randomization-based approach that resists eight adversarial attacks targeting deep learning models in the testing phase. Our novel approach consists of changing the training strategy in the target network classifier and selecting random feature samples. We consider the attacker with a Limited-Knowledge and Semi-Knowledge conditions to undertake the most prevalent types of adversarial attacks. We evaluate the robustness of our approach using the well-known UNSW-NB15 datasets that include realistic and synthetic attacks. Afterward, we demonstrate that our strategy outperforms the existing state-of-the-art approach, such as the Most Powerful Attack, which consists of fine-tuning the network model against specific adversarial attacks. Finally, our experimental results show that our methodology can secure the target network and resists adversarial attack transferability by over 60%.

摘要: 在过去的几十年里，人工智能的崛起给了我们解决日常生活中最具挑战性的问题的能力，比如癌症预测和自主导航。但是，如果不确保这些应用程序不能抵御敌意攻击，则这些应用程序可能不可靠。此外，最近的工作表明，一些对抗性例子可以在不同的模型之间转移。因此，通过稳健的模型抵抗敌意操纵来避免这种可转移性是至关重要的。在本文中，我们提出了一种基于特征随机化的方法，该方法在测试阶段抵抗了八种针对深度学习模型的对抗性攻击。我们的新方法包括改变目标网络分类器中的训练策略和随机选择特征样本。我们认为攻击者在有限知识和半知识条件下可以进行最常见的对抗性攻击类型。我们使用著名的UNSW-NB15数据集评估了我们方法的健壮性，其中包括现实攻击和合成攻击。之后，我们证明了我们的策略优于现有的最先进的方法，例如最强大的攻击，它包括针对特定的对手攻击对网络模型进行微调。实验结果表明，该方法能够保证目标网络的安全，并能抵抗60%以上的敌意攻击可转移性。



## **49. Detecting Adversarial Perturbations in Multi-Task Perception**

多任务感知中的对抗性扰动检测 cs.CV

Accepted at IROS 2022

**SubmitDate**: 2022-09-11    [paper-pdf](http://arxiv.org/pdf/2203.01177v2)

**Authors**: Marvin Klingner, Varun Ravi Kumar, Senthil Yogamani, Andreas Bär, Tim Fingscheidt

**Abstracts**: While deep neural networks (DNNs) achieve impressive performance on environment perception tasks, their sensitivity to adversarial perturbations limits their use in practical applications. In this paper, we (i) propose a novel adversarial perturbation detection scheme based on multi-task perception of complex vision tasks (i.e., depth estimation and semantic segmentation). Specifically, adversarial perturbations are detected by inconsistencies between extracted edges of the input image, the depth output, and the segmentation output. To further improve this technique, we (ii) develop a novel edge consistency loss between all three modalities, thereby improving their initial consistency which in turn supports our detection scheme. We verify our detection scheme's effectiveness by employing various known attacks and image noises. In addition, we (iii) develop a multi-task adversarial attack, aiming at fooling both tasks as well as our detection scheme. Experimental evaluation on the Cityscapes and KITTI datasets shows that under an assumption of a 5% false positive rate up to 100% of images are correctly detected as adversarially perturbed, depending on the strength of the perturbation. Code is available at https://github.com/ifnspaml/AdvAttackDet. A short video at https://youtu.be/KKa6gOyWmH4 provides qualitative results.

摘要: 虽然深度神经网络(DNN)在环境感知任务中取得了令人印象深刻的性能，但它们对对抗性扰动的敏感性限制了它们在实际应用中的应用。本文(I)提出了一种基于复杂视觉任务多任务感知(深度估计和语义分割)的对抗性扰动检测方法。具体地，通过所提取的输入图像的边缘、深度输出和分割输出之间的不一致来检测对抗性扰动。为了进一步改进这一技术，我们(Ii)在所有三种模式之间开发了一种新的边缘一致性损失，从而改善了它们的初始一致性，这反过来又支持我们的检测方案。通过使用各种已知攻击和图像噪声来验证我们的检测方案的有效性。此外，我们(Iii)开发了一种多任务对抗性攻击，旨在愚弄两个任务和我们的检测方案。在CITYSCAPES和KITTI数据集上的实验评估表明，在假阳性率为5%的假设下，高达100%的图像被正确检测为恶意扰动，这取决于扰动的强度。代码可在https://github.com/ifnspaml/AdvAttackDet.上找到Https://youtu.be/KKa6gOyWmH4上的一段简短视频提供了定性的结果。



## **50. Robustness through Cognitive Dissociation Mitigation in Contrastive Adversarial Training**

对比性对抗训练中认知分离缓解的稳健性 cs.LG

Accepted to ICMLC 2022

**SubmitDate**: 2022-09-11    [paper-pdf](http://arxiv.org/pdf/2203.08959v3)

**Authors**: Adir Rahamim, Itay Naeh

**Abstracts**: In this paper, we introduce a novel neural network training framework that increases model's adversarial robustness to adversarial attacks while maintaining high clean accuracy by combining contrastive learning (CL) with adversarial training (AT). We propose to improve model robustness to adversarial attacks by learning feature representations that are consistent under both data augmentations and adversarial perturbations. We leverage contrastive learning to improve adversarial robustness by considering an adversarial example as another positive example, and aim to maximize the similarity between random augmentations of data samples and their adversarial example, while constantly updating the classification head in order to avoid a cognitive dissociation between the classification head and the embedding space. This dissociation is caused by the fact that CL updates the network up to the embedding space, while freezing the classification head which is used to generate new positive adversarial examples. We validate our method, Contrastive Learning with Adversarial Features(CLAF), on the CIFAR-10 dataset on which it outperforms both robust accuracy and clean accuracy over alternative supervised and self-supervised adversarial learning methods.

摘要: 本文提出了一种新的神经网络训练框架，通过将对比学习(CL)和对抗训练(AT)相结合，在保持较高精度的同时，提高了模型对对手攻击的鲁棒性。我们提出通过学习在数据扩充和对抗性扰动下都是一致的特征表示来提高模型对对抗性攻击的稳健性。我们利用对比学习来提高对抗性样本的稳健性，将一个对抗性样本作为另一个正例，目标是最大化随机增加的数据样本与其对抗性样本之间的相似度，同时不断更新分类头，以避免分类头与嵌入空间之间的认知分离。这种分离是由于CL将网络更新到嵌入空间，同时冻结用于生成新的正面对抗性实例的分类头。我们在CIFAR-10数据集上验证了我们的方法，即带有对抗性特征的对比学习(CLAF)，在CIFAR-10数据集上，它的性能优于其他监督和自我监督对抗性学习方法的稳健准确率和干净准确率。



