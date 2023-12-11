# Latest Adversarial Attack Papers
**update at 2023-12-11 16:13:30**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Topology-Based Reconstruction Prevention for Decentralised Learning**

基于拓扑结构的分布式学习重构预防 cs.CR

15 pages, 8 figures, submitted to IEEE S&P 2024, for associated  experiment source code see doi:10.4121/21572601

**SubmitDate**: 2023-12-08    [abs](http://arxiv.org/abs/2312.05248v1) [paper-pdf](http://arxiv.org/pdf/2312.05248v1)

**Authors**: Florine W. Dekker, Zekeriya Erkin, Mauro Conti

**Abstract**: Decentralised learning has recently gained traction as an alternative to federated learning in which both data and coordination are distributed over its users. To preserve the confidentiality of users' data, decentralised learning relies on differential privacy, multi-party computation, or a combination thereof. However, running multiple privacy-preserving summations in sequence may allow adversaries to perform reconstruction attacks. Unfortunately, current reconstruction countermeasures either cannot trivially be adapted to the distributed setting, or add excessive amounts of noise.   In this work, we first show that passive honest-but-curious adversaries can reconstruct other users' private data after several privacy-preserving summations. For example, in subgraphs with 18 users, we show that only three passive honest-but-curious adversaries succeed at reconstructing private data 11.0% of the time, requiring an average of 8.8 summations per adversary. The success rate is independent of the size of the full network. We consider weak adversaries, who do not control the graph topology and can exploit neither the workings of the summation protocol nor the specifics of users' data.   We develop a mathematical understanding of how reconstruction relates to topology and propose the first topology-based decentralised defence against reconstruction attacks. Specifically, we show that reconstruction requires a number of adversaries linear in the length of the network's shortest cycle. Consequently, reconstructing private data from privacy-preserving summations is impossible in acyclic networks.   Our work is a stepping stone for a formal theory of decentralised reconstruction defences based on topology. Such a theory would generalise our countermeasure beyond summation, define confidentiality in terms of entropy, and describe the effects of (topology-aware) differential privacy.

摘要: 分散式学习最近作为联合学习的替代方案获得了吸引力，在联合学习中，数据和协调都分布在用户身上。为了保护用户数据的机密性，分散学习依赖于差异隐私、多方计算或它们的组合。但是，按顺序运行多个隐私保护摘要可能会允许攻击者执行重建攻击。不幸的是，当前的重建对策要么不能简单地适应分布式设置，要么增加了过多的噪声。在这项工作中，我们首先证明了被动诚实但好奇的攻击者可以在多次隐私保护汇总后重建其他用户的私人数据。例如，在具有18个用户的子图中，我们表明只有三个被动的诚实但好奇的对手在11.0%的时间内成功重建私人数据，每个对手平均需要8.8次求和。成功率与整个网络的大小无关。我们考虑弱对手，他们不控制图形拓扑，既不能利用求和协议的工作原理，也不能利用用户数据的细节。我们发展了对重构如何与拓扑相关的数学理解，并提出了第一个基于拓扑的分布式防御重构攻击的方法。具体地说，我们证明了重建需要若干个与网络最短周期长度成线性关系的对手。因此，在非循环网络中，从保护隐私的求和中重建私有数据是不可能的。我们的工作是基于拓扑学的分散重建防御的形式理论的垫脚石。这样的理论将概括我们的对策，超越总和，用熵来定义机密性，并描述(拓扑感知的)差异隐私的影响。



## **2. On the Robustness of Large Multimodal Models Against Image Adversarial Attacks**

大型多模模型对图像攻击的稳健性研究 cs.CV

**SubmitDate**: 2023-12-08    [abs](http://arxiv.org/abs/2312.03777v2) [paper-pdf](http://arxiv.org/pdf/2312.03777v2)

**Authors**: Xuanming Cui, Alejandro Aparcedo, Young Kyun Jang, Ser-Nam Lim

**Abstract**: Recent advances in instruction tuning have led to the development of State-of-the-Art Large Multimodal Models (LMMs). Given the novelty of these models, the impact of visual adversarial attacks on LMMs has not been thoroughly examined. We conduct a comprehensive study of the robustness of various LMMs against different adversarial attacks, evaluated across tasks including image classification, image captioning, and Visual Question Answer (VQA). We find that in general LMMs are not robust to visual adversarial inputs. However, our findings suggest that context provided to the model via prompts, such as questions in a QA pair helps to mitigate the effects of visual adversarial inputs. Notably, the LMMs evaluated demonstrated remarkable resilience to such attacks on the ScienceQA task with only an 8.10% drop in performance compared to their visual counterparts which dropped 99.73%. We also propose a new approach to real-world image classification which we term query decomposition. By incorporating existence queries into our input prompt we observe diminished attack effectiveness and improvements in image classification accuracy. This research highlights a previously under-explored facet of LMM robustness and sets the stage for future work aimed at strengthening the resilience of multimodal systems in adversarial environments.

摘要: 最近在指令调优方面的进展导致了最先进的大型多通道模型(LMM)的发展。鉴于这些模型的新颖性，视觉对抗性攻击对LMM的影响尚未得到彻底研究。我们全面研究了不同LMM对不同对手攻击的稳健性，评估了包括图像分类、图像字幕和视觉问答(VQA)在内的任务。我们发现，在一般情况下，LMM对视觉对抗性输入并不健壮。然而，我们的发现表明，通过提示提供给模型的上下文，例如QA对中的问题，有助于减轻视觉对抗性输入的影响。值得注意的是，被评估的LMM对Science QA任务的此类攻击表现出了非凡的弹性，与视觉同行相比，性能仅下降了8.10%，下降了99.73%。我们还提出了一种新的图像分类方法，我们称之为查询分解。通过将存在查询合并到我们的输入提示中，我们观察到攻击有效性的降低和图像分类准确率的提高。这项研究突出了LMM稳健性以前未被充分探索的一个方面，并为未来旨在加强多通道系统在对抗性环境中的弹性的工作奠定了基础。



## **3. Partial-Information, Longitudinal Cyber Attacks on LiDAR in Autonomous Vehicles**

部分信息，纵向网络攻击对自动驾驶汽车中的LiDAR cs.CR

**SubmitDate**: 2023-12-08    [abs](http://arxiv.org/abs/2303.03470v3) [paper-pdf](http://arxiv.org/pdf/2303.03470v3)

**Authors**: R. Spencer Hallyburton, Qingzhao Zhang, Z. Morley Mao, Miroslav Pajic

**Abstract**: What happens to an autonomous vehicle (AV) if its data are adversarially compromised? Prior security studies have addressed this question through mostly unrealistic threat models, with limited practical relevance, such as white-box adversarial learning or nanometer-scale laser aiming and spoofing. With growing evidence that cyber threats pose real, imminent danger to AVs and cyber-physical systems (CPS) in general, we present and evaluate a novel AV threat model: a cyber-level attacker capable of disrupting sensor data but lacking any situational awareness. We demonstrate that even though the attacker has minimal knowledge and only access to raw data from a single sensor (i.e., LiDAR), she can design several attacks that critically compromise perception and tracking in multi-sensor AVs. To mitigate vulnerabilities and advance secure architectures in AVs, we introduce two improvements for security-aware fusion: a probabilistic data-asymmetry monitor and a scalable track-to-track fusion of 3D LiDAR and monocular detections (T2T-3DLM); we demonstrate that the approaches significantly reduce attack effectiveness. To support objective safety and security evaluations in AVs, we release our security evaluation platform, AVsec, which is built on security-relevant metrics to benchmark AVs on gold-standard longitudinal AV datasets and AV simulators.

摘要: 如果自动驾驶汽车（AV）的数据遭到恶意破坏，会发生什么？以前的安全研究通过大多数不切实际的威胁模型来解决这个问题，实际相关性有限，例如白盒对抗学习或纳米级激光瞄准和欺骗。随着越来越多的证据表明，网络威胁对AV和网络物理系统（CPS）构成了真正的，迫在眉睫的危险，我们提出并评估了一种新的AV威胁模型：网络级攻击者能够破坏传感器数据，但缺乏任何态势感知。我们证明，即使攻击者具有最少的知识，并且只能访问来自单个传感器的原始数据（即，LiDAR），她可以设计几种攻击，严重损害多传感器AV的感知和跟踪。为了减轻漏洞并推进AV中的安全架构，我们引入了两种安全感知融合的改进：概率数据不对称监视器和3D LiDAR和单目检测（T2 T-3DLM）的可扩展跟踪到跟踪融合;我们证明了这些方法显着降低了攻击效率。为了支持对AV进行客观的安全评估，我们发布了我们的安全评估平台AVsec，该平台基于安全相关指标，可在黄金标准纵向AV数据集和AV模拟器上对AV进行基准测试。



## **4. MIMIR: Masked Image Modeling for Mutual Information-based Adversarial Robustness**

MIMIR：基于交互信息的对抗性掩蔽图像建模 cs.CV

**SubmitDate**: 2023-12-08    [abs](http://arxiv.org/abs/2312.04960v1) [paper-pdf](http://arxiv.org/pdf/2312.04960v1)

**Authors**: Xiaoyun Xu, Shujian Yu, Jingzheng Wu, Stjepan Picek

**Abstract**: Vision Transformers (ViTs) achieve superior performance on various tasks compared to convolutional neural networks (CNNs), but ViTs are also vulnerable to adversarial attacks. Adversarial training is one of the most successful methods to build robust CNN models. Thus, recent works explored new methodologies for adversarial training of ViTs based on the differences between ViTs and CNNs, such as better training strategies, preventing attention from focusing on a single block, or discarding low-attention embeddings. However, these methods still follow the design of traditional supervised adversarial training, limiting the potential of adversarial training on ViTs. This paper proposes a novel defense method, MIMIR, which aims to build a different adversarial training methodology by utilizing Masked Image Modeling at pre-training. We create an autoencoder that accepts adversarial examples as input but takes the clean examples as the modeling target. Then, we create a mutual information (MI) penalty following the idea of the Information Bottleneck. Among the two information source inputs and corresponding adversarial perturbation, the perturbation information is eliminated due to the constraint of the modeling target. Next, we provide a theoretical analysis of MIMIR using the bounds of the MI penalty. We also design two adaptive attacks when the adversary is aware of the MIMIR defense and show that MIMIR still performs well. The experimental results show that MIMIR improves (natural and adversarial) accuracy on average by 4.19\% on CIFAR-10 and 5.52\% on ImageNet-1K, compared to baselines. On Tiny-ImageNet, we obtained improved natural accuracy of 2.99\% on average and comparable adversarial accuracy. Our code and trained models are publicly available\footnote{\url{https://anonymous.4open.science/r/MIMIR-5444/README.md}}.

摘要: 与卷积神经网络(CNN)相比，视觉转换器(VITS)在各种任务中取得了优越的性能，但VITS也容易受到对手的攻击。对抗性训练是建立稳健的CNN模型最成功的方法之一。因此，最近的工作探索了基于VITS和CNN之间的差异的VITS对抗性训练的新方法，例如更好的训练策略，防止注意力集中在单个区块上，或者放弃低注意嵌入。然而，这些方法仍然遵循传统的监督对抗性训练的设计，限制了对抗性训练在VITS上的潜力。本文提出了一种新的防御方法MIMIR，旨在通过在训练前利用蒙版图像建模来构建一种不同的对手训练方法。我们创建了一个自动编码器，它接受对抗性例子作为输入，但以干净的例子作为建模目标。然后，我们遵循信息瓶颈的思想创建了一个互信息(MI)惩罚。在两个信息源输入和对应的对抗性扰动中，由于建模目标的限制，扰动信息被消除。接下来，我们利用MI惩罚的界对MIMIR进行了理论分析。我们还设计了两个自适应攻击，当对手知道Mimir防御时，表明Mimir仍然执行得很好。实验结果表明，与基线相比，MIMIR在CIFAR-10和ImageNet-1K上的(自然和对抗)准确率分别平均提高了4.19和5.52。在Micro-ImageNet上，我们获得了平均2.99\%的改进的自然准确率和相当的对手准确率。我们的代码和经过训练的模型是公开的available\footnote{\url{https://anonymous.4open.science/r/MIMIR-5444/README.md}}.



## **5. AHSecAgg and TSKG: Lightweight Secure Aggregation for Federated Learning Without Compromise**

AHSecAgg和TSKG：面向联合学习的轻量级安全聚合 cs.CR

**SubmitDate**: 2023-12-08    [abs](http://arxiv.org/abs/2312.04937v1) [paper-pdf](http://arxiv.org/pdf/2312.04937v1)

**Authors**: Siqing Zhang, Yong Liao, Pengyuan Zhou

**Abstract**: Leveraging federated learning (FL) to enable cross-domain privacy-sensitive data mining represents a vital breakthrough to accomplish privacy-preserving learning. However, attackers can infer the original user data by analyzing the uploaded intermediate parameters during the aggregation process. Therefore, secure aggregation has become a critical issue in the field of FL. Many secure aggregation protocols face the problem of high computation costs, which severely limits their applicability. To this end, we propose AHSecAgg, a lightweight secure aggregation protocol using additive homomorphic masks. AHSecAgg significantly reduces computation overhead without compromising the dropout handling capability or model accuracy. We prove the security of AHSecAgg in semi-honest and active adversary settings. In addition, in cross-silo scenarios where the group of participants is relatively fixed during each round, we propose TSKG, a lightweight Threshold Signature based masking key generation method. TSKG can generate different temporary secrets and shares for different aggregation rounds using the initial key and thus effectively eliminates the cost of secret sharing and key agreement. We prove TSKG does not sacrifice security. Extensive experiments show that AHSecAgg significantly outperforms state-of-the-art mask-based secure aggregation protocols in terms of computational efficiency, and TSKG effectively reduces the computation and communication costs for existing secure aggregation protocols.

摘要: 利用联合学习(FL)实现跨域隐私敏感数据挖掘是实现隐私保护学习的重要突破。然而，攻击者可以通过分析聚合过程中上传的中间参数来推断原始用户数据。因此，安全聚合已经成为FL领域的一个关键问题。许多安全聚合协议都面临着计算代价高的问题，这严重限制了它们的适用性。为此，我们提出了一种基于加性同态掩码的轻量级安全聚合协议AHSecAgg。AHSecAgg在不影响丢弃处理能力或模型精度的情况下显著减少了计算开销。我们证明了AHSecAgg在半诚实和主动对手环境下的安全性。此外，在每轮参与者群相对固定的跨竖井场景中，我们提出了一种基于轻量级门限签名的掩码密钥生成方法TSKG。TSKG可以使用初始密钥为不同的聚合轮次生成不同的临时秘密和份额，从而有效地消除了秘密共享和密钥协商的代价。我们证明TSKG不会牺牲安全性。大量实验表明，AHSecAgg在计算效率上明显优于现有的基于掩码的安全聚合协议，TSKG有效地降低了现有安全聚合协议的计算和通信开销。



## **6. SA-Attack: Improving Adversarial Transferability of Vision-Language Pre-training Models via Self-Augmentation**

SA-Attack：通过自我增强提高视觉语言预训练模型的对抗性迁移 cs.CV

**SubmitDate**: 2023-12-08    [abs](http://arxiv.org/abs/2312.04913v1) [paper-pdf](http://arxiv.org/pdf/2312.04913v1)

**Authors**: Bangyan He, Xiaojun Jia, Siyuan Liang, Tianrui Lou, Yang Liu, Xiaochun Cao

**Abstract**: Current Visual-Language Pre-training (VLP) models are vulnerable to adversarial examples. These adversarial examples present substantial security risks to VLP models, as they can leverage inherent weaknesses in the models, resulting in incorrect predictions. In contrast to white-box adversarial attacks, transfer attacks (where the adversary crafts adversarial examples on a white-box model to fool another black-box model) are more reflective of real-world scenarios, thus making them more meaningful for research. By summarizing and analyzing existing research, we identified two factors that can influence the efficacy of transfer attacks on VLP models: inter-modal interaction and data diversity. Based on these insights, we propose a self-augment-based transfer attack method, termed SA-Attack. Specifically, during the generation of adversarial images and adversarial texts, we apply different data augmentation methods to the image modality and text modality, respectively, with the aim of improving the adversarial transferability of the generated adversarial images and texts. Experiments conducted on the FLickr30K and COCO datasets have validated the effectiveness of our method. Our code will be available after this paper is accepted.

摘要: 当前的视觉语言预训练(VLP)模型容易受到对抗性例子的影响。这些对抗性的例子给VLP模型带来了很大的安全风险，因为它们可以利用模型中的固有弱点，导致错误的预测。与白盒对抗性攻击相比，传输攻击(对手在白盒模型上伪造敌意例子以愚弄另一个黑盒模型)更能反映真实世界的场景，因此更有研究意义。通过对已有研究的总结和分析，我们确定了影响VLP模型传输攻击效果的两个因素：通道间交互和数据多样性。在此基础上，我们提出了一种基于自增强的传输攻击方法SA-Attack。具体地说，在对抗性图像和对抗性文本的生成过程中，我们分别对图像和文本通道应用了不同的数据增强方法，目的是提高生成的对抗性图像和文本的对抗性可转移性。在FLickr30K和COCO数据集上进行的实验验证了该方法的有效性。我们的代码将在本文被接受后可用。



## **7. HC-Ref: Hierarchical Constrained Refinement for Robust Adversarial Training of GNNs**

HC-Ref：用于GNN稳健对抗性训练的分层约束求精 cs.LG

**SubmitDate**: 2023-12-08    [abs](http://arxiv.org/abs/2312.04879v1) [paper-pdf](http://arxiv.org/pdf/2312.04879v1)

**Authors**: Xiaobing Pei, Haoran Yang, Gang Shen

**Abstract**: Recent studies have shown that attackers can catastrophically reduce the performance of GNNs by maliciously modifying the graph structure or node features on the graph. Adversarial training, which has been shown to be one of the most effective defense mechanisms against adversarial attacks in computer vision, holds great promise for enhancing the robustness of GNNs. There is limited research on defending against attacks by performing adversarial training on graphs, and it is crucial to delve deeper into this approach to optimize its effectiveness. Therefore, based on robust adversarial training on graphs, we propose a hierarchical constraint refinement framework (HC-Ref) that enhances the anti-perturbation capabilities of GNNs and downstream classifiers separately, ultimately leading to improved robustness. We propose corresponding adversarial regularization terms that are conducive to adaptively narrowing the domain gap between the normal part and the perturbation part according to the characteristics of different layers, promoting the smoothness of the predicted distribution of both parts. Moreover, existing research on graph robust adversarial training primarily concentrates on training from the standpoint of node feature perturbations and seldom takes into account alterations in the graph structure. This limitation makes it challenging to prevent attacks based on topological changes in the graph. This paper generates adversarial examples by utilizing graph structure perturbations, offering an effective approach to defend against attack methods that are based on topological changes. Extensive experiments on two real-world graph benchmarks show that HC-Ref successfully resists various attacks and has better node classification performance compared to several baseline methods.

摘要: 最近的研究表明，攻击者可以通过恶意修改图的结构或图上的节点特征来灾难性地降低GNN的性能。对抗性训练是计算机视觉中对抗对抗性攻击的最有效的防御机制之一，它为增强GNN的健壮性带来了巨大的希望。通过对图进行对抗性训练来防御攻击的研究有限，深入研究这种方法以优化其有效性是至关重要的。因此，基于图上的稳健对抗性训练，我们提出了一种层次化约束求精框架(HC-Ref)，该框架分别增强了GNN和下游分类器的抗扰动能力，最终提高了鲁棒性。根据不同层的特点，提出了相应的对抗性正则化条件，有利于自适应地缩小正常部分和扰动部分之间的域间距，提高了两部分预测分布的光滑性。此外，现有的图稳健对抗训练的研究主要集中在从节点特征扰动的角度进行训练，很少考虑图结构的变化。这种限制使得防止基于图中的拓扑变化的攻击变得具有挑战性。利用图的结构扰动生成对抗性实例，为防御基于拓扑变化的攻击方法提供了一种有效的途径。在两个真实图基准上的大量实验表明，HC-Ref能够成功地抵抗各种攻击，并且与几种基准方法相比具有更好的节点分类性能。



## **8. Data-Driven Identification of Attack-free Sensors in Networked Control Systems**

网络控制系统中无攻击传感器的数据驱动识别 eess.SY

Conference submission

**SubmitDate**: 2023-12-08    [abs](http://arxiv.org/abs/2312.04845v1) [paper-pdf](http://arxiv.org/pdf/2312.04845v1)

**Authors**: Sribalaji C. Anand, Michelle S. Chong, André M. H. Teixeira

**Abstract**: This paper proposes a data-driven framework to identify the attack-free sensors in a networked control system when some of the sensors are corrupted by an adversary. An operator with access to offline input-output attack-free trajectories of the plant is considered. Then, a data-driven algorithm is proposed to identify the attack-free sensors when the plant is controlled online. We also provide necessary conditions, based on the properties of the plant, under which the algorithm is feasible. An extension of the algorithm is presented to identify the sensors completely online against certain classes of attacks. The efficacy of our algorithm is depicted through numerical examples.

摘要: 提出了一种数据驱动的框架，用于在网络控制系统中部分传感器被对手破坏时识别无攻击的传感器。考虑一个操作员可以访问对象的离线输入-输出无攻击轨迹。然后，提出了一种数据驱动算法，用于在线控制对象时识别无攻击传感器。根据被控对象的性质，给出了算法可行的必要条件。提出了该算法的一种扩展，以针对某些类型的攻击完全在线地识别传感器。通过数值算例说明了该算法的有效性。



## **9. MimicDiffusion: Purifying Adversarial Perturbation via Mimicking Clean Diffusion Model**

仿真扩散：模仿清洁扩散模型净化对抗性扰动 cs.CV

**SubmitDate**: 2023-12-08    [abs](http://arxiv.org/abs/2312.04802v1) [paper-pdf](http://arxiv.org/pdf/2312.04802v1)

**Authors**: Kaiyu Song, Hanjiang Lai

**Abstract**: Deep neural networks (DNNs) are vulnerable to adversarial perturbation, where an imperceptible perturbation is added to the image that can fool the DNNs. Diffusion-based adversarial purification focuses on using the diffusion model to generate a clean image against such adversarial attacks. Unfortunately, the generative process of the diffusion model is also inevitably affected by adversarial perturbation since the diffusion model is also a deep network where its input has adversarial perturbation. In this work, we propose MimicDiffusion, a new diffusion-based adversarial purification technique, that directly approximates the generative process of the diffusion model with the clean image as input. Concretely, we analyze the differences between the guided terms using the clean image and the adversarial sample. After that, we first implement MimicDiffusion based on Manhattan distance. Then, we propose two guidance to purify the adversarial perturbation and approximate the clean diffusion model. Extensive experiments on three image datasets including CIFAR-10, CIFAR-100, and ImageNet with three classifier backbones including WideResNet-70-16, WideResNet-28-10, and ResNet50 demonstrate that MimicDiffusion significantly performs better than the state-of-the-art baselines. On CIFAR-10, CIFAR-100, and ImageNet, it achieves 92.67\%, 61.35\%, and 61.53\% average robust accuracy, which are 18.49\%, 13.23\%, and 17.64\% higher, respectively. The code is available in the supplementary material.

摘要: 深度神经网络(DNN)容易受到敌意扰动，在图像中添加一个不可察觉的扰动可以愚弄DNN。基于扩散的敌意净化侧重于使用扩散模型来生成对抗此类敌意攻击的干净形象。不幸的是，扩散模型的生成过程也不可避免地受到对抗性扰动的影响，因为扩散模型也是一个输入具有对抗性扰动的深层网络。在这项工作中，我们提出了一种新的基于扩散的对抗性净化技术MimicDiffulation，它以干净的图像作为输入，直接逼近扩散模型的生成过程。具体地说，我们用干净的图像和对抗性的样本分析了引导词的差异。之后，我们首先实现了基于曼哈顿距离的MimicDiffsion。然后，我们提出了两种方法来净化对抗性扰动和近似清洁扩散模型。在包括CIFAR-10、CIFAR-100和ImageNet在内的三个图像数据集上的广泛实验表明，MimicDiffsion的性能明显好于最先进的基线。在CIFAR-10、CIFAR-100和ImageNet上的平均稳健准确率分别为92.67、61.35和61.53，分别高出18.49、13.23和17.64。该代码可在补充材料中找到。



## **10. DeceptPrompt: Exploiting LLM-driven Code Generation via Adversarial Natural Language Instructions**

DeceptPrompt：通过对抗性自然语言指令利用LLM驱动的代码生成 cs.CR

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2312.04730v1) [paper-pdf](http://arxiv.org/pdf/2312.04730v1)

**Authors**: Fangzhou Wu, Xiaogeng Liu, Chaowei Xiao

**Abstract**: With the advancement of Large Language Models (LLMs), significant progress has been made in code generation, enabling LLMs to transform natural language into programming code. These Code LLMs have been widely accepted by massive users and organizations. However, a dangerous nature is hidden in the code, which is the existence of fatal vulnerabilities. While some LLM providers have attempted to address these issues by aligning with human guidance, these efforts fall short of making Code LLMs practical and robust. Without a deep understanding of the performance of the LLMs under the practical worst cases, it would be concerning to apply them to various real-world applications. In this paper, we answer the critical issue: Are existing Code LLMs immune to generating vulnerable code? If not, what is the possible maximum severity of this issue in practical deployment scenarios? In this paper, we introduce DeceptPrompt, a novel algorithm that can generate adversarial natural language instructions that drive the Code LLMs to generate functionality correct code with vulnerabilities. DeceptPrompt is achieved through a systematic evolution-based algorithm with a fine grain loss design. The unique advantage of DeceptPrompt enables us to find natural prefix/suffix with totally benign and non-directional semantic meaning, meanwhile, having great power in inducing the Code LLMs to generate vulnerable code. This feature can enable us to conduct the almost-worstcase red-teaming on these LLMs in a real scenario, where users are using natural language. Our extensive experiments and analyses on DeceptPrompt not only validate the effectiveness of our approach but also shed light on the huge weakness of LLMs in the code generation task. When applying the optimized prefix/suffix, the attack success rate (ASR) will improve by average 50% compared with no prefix/suffix applying.

摘要: 随着大型语言模型(LLMS)的发展，在代码生成方面取得了重大进展，使LLMS能够将自然语言转换为编程代码。这些CodeLLM已被广大用户和组织广泛接受。然而，代码中隐藏着一个危险的性质，那就是存在致命的漏洞。虽然一些LLM提供商试图通过与人类的指导保持一致来解决这些问题，但这些努力并不能使Code LLM实用和健壮。如果不深入了解LLMS在实际最坏情况下的性能，将它们应用于各种现实世界应用将是令人担忧的。在这篇文章中，我们回答了一个关键问题：现有的代码LLM是否不会生成易受攻击的代码？如果不是，此问题在实际部署方案中可能的最大严重程度是多少？在本文中，我们介绍了DeceptPrompt算法，它可以生成敌意的自然语言指令，这些指令驱动Code LLMS生成有漏洞的功能正确的代码。DeceptPrompt是通过基于系统进化的算法实现的，具有细粒度的损耗设计。DeceptPrompt的独特优势使我们能够找到具有完全良性和非方向性语义的自然前缀/后缀，同时对诱使Code LLMS生成易受攻击的代码具有强大的能力。这一功能使我们能够在用户使用自然语言的真实场景中对这些LLM进行几乎最糟糕的红色团队。我们在DeceptPrompt上的大量实验和分析不仅验证了我们方法的有效性，而且揭示了LLMS在代码生成任务中的巨大弱点。当应用优化的前缀/后缀时，与不应用前缀/后缀相比，攻击成功率(ASR)将平均提高50%。



## **11. gcDLSeg: Integrating Graph-cut into Deep Learning for Binary Semantic Segmentation**

gcDLSeg：将Graph-cut集成到深度学习中进行二进制语义分割 cs.CV

12 pages

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2312.04713v1) [paper-pdf](http://arxiv.org/pdf/2312.04713v1)

**Authors**: Hui Xie, Weiyu Xu, Ya Xing Wang, John Buatti, Xiaodong Wu

**Abstract**: Binary semantic segmentation in computer vision is a fundamental problem. As a model-based segmentation method, the graph-cut approach was one of the most successful binary segmentation methods thanks to its global optimality guarantee of the solutions and its practical polynomial-time complexity. Recently, many deep learning (DL) based methods have been developed for this task and yielded remarkable performance, resulting in a paradigm shift in this field. To combine the strengths of both approaches, we propose in this study to integrate the graph-cut approach into a deep learning network for end-to-end learning. Unfortunately, backward propagation through the graph-cut module in the DL network is challenging due to the combinatorial nature of the graph-cut algorithm. To tackle this challenge, we propose a novel residual graph-cut loss and a quasi-residual connection, enabling the backward propagation of the gradients of the residual graph-cut loss for effective feature learning guided by the graph-cut segmentation model. In the inference phase, globally optimal segmentation is achieved with respect to the graph-cut energy defined on the optimized image features learned from DL networks. Experiments on the public AZH chronic wound data set and the pancreas cancer data set from the medical segmentation decathlon (MSD) demonstrated promising segmentation accuracy, and improved robustness against adversarial attacks.

摘要: 计算机视觉中的二值语义分割是一个基本问题。图割法作为一种基于模型的分割方法，由于其解的全局最优性保证和实用的多项式时间复杂度，是最成功的二值分割方法之一。最近，许多基于深度学习的方法被开发出来，并取得了显著的性能，导致了该领域的范式转变。为了结合两种方法的优点，我们在本研究中建议将图割方法整合到深度学习网络中进行端到端学习。不幸的是，由于图割算法的组合性质，通过DL网络中的图割模块的反向传播是具有挑战性的。为了应对这一挑战，我们提出了一种新的残差图割损失和准残差连接，使得残差图割损失的梯度能够反向传播，从而在图割分割模型的指导下进行有效的特征学习。在推理阶段，对于从DL网络学习的优化图像特征定义的图割能量，实现全局最优分割。在公共AZH慢性创伤数据集和来自医学分段十项全能(MSD)的胰腺癌数据集上的实验表明，分割精度是有希望的，并提高了对对手攻击的鲁棒性。



## **12. Diffence: Fencing Membership Privacy With Diffusion Models**

Diffence：使用扩散模型来保护成员隐私 cs.CR

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2312.04692v1) [paper-pdf](http://arxiv.org/pdf/2312.04692v1)

**Authors**: Yuefeng Peng, Ali Naseh, Amir Houmansadr

**Abstract**: Deep learning models, while achieving remarkable performance across various tasks, are vulnerable to member inference attacks, wherein adversaries identify if a specific data point was part of a model's training set. This susceptibility raises substantial privacy concerns, especially when models are trained on sensitive datasets. Current defense methods often struggle to provide robust protection without hurting model utility, and they often require retraining the model or using extra data. In this work, we introduce a novel defense framework against membership attacks by leveraging generative models. The key intuition of our defense is to remove the differences between member and non-member inputs which can be used to perform membership attacks, by re-generating input samples before feeding them to the target model. Therefore, our defense works \emph{pre-inference}, which is unlike prior defenses that are either training-time (modify the model) or post-inference time (modify the model's output).   A unique feature of our defense is that it works on input samples only, without modifying the training or inference phase of the target model. Therefore, it can be cascaded with other defense mechanisms as we demonstrate through experiments. Through extensive experimentation, we show that our approach can serve as a robust plug-n-play defense mechanism, enhancing membership privacy without compromising model utility in both baseline and defended settings. For example, our method enhanced the effectiveness of recent state-of-the-art defenses, reducing attack accuracy by an average of 5.7\% to 12.4\% across three datasets, without any impact on the model's accuracy. By integrating our method with prior defenses, we achieve new state-of-the-art performance in the privacy-utility trade-off.

摘要: 深度学习模型虽然在各种任务中取得了出色的性能，但容易受到成员推断攻击，其中攻击者识别特定数据点是否是模型训练集的一部分。这种敏感性引起了大量的隐私问题，特别是当模型在敏感数据集上训练时。目前的防御方法通常难以在不损害模型效用的情况下提供强大的保护，并且它们通常需要重新训练模型或使用额外的数据。在这项工作中，我们引入了一个新的防御框架，利用生成模型对成员攻击。我们防御的关键直觉是通过在将输入样本馈送到目标模型之前重新生成输入样本来消除可用于执行成员攻击的成员和非成员输入之间的差异。因此，我们的防御可以进行预推理，这与之前的防御不同，之前的防御要么是训练时间（修改模型），要么是后推理时间（修改模型的输出）。   我们的防御的一个独特之处在于，它只对输入样本起作用，而不修改目标模型的训练或推理阶段。因此，正如我们通过实验所证明的那样，它可以与其他防御机制级联。通过广泛的实验，我们表明，我们的方法可以作为一个强大的即插即用的防御机制，提高会员隐私，而不损害模型效用在基线和防御设置。例如，我们的方法增强了最近最先进防御的有效性，在三个数据集上平均将攻击准确性降低了5.7%至12.4%，而对模型的准确性没有任何影响。通过将我们的方法与先前的防御相结合，我们在隐私-效用权衡中实现了新的最先进的性能。



## **13. Defending Against Alignment-Breaking Attacks via Robustly Aligned LLM**

通过强健对齐的LLM防御对齐破坏攻击 cs.CL

16 Pages, 5 Figures, 6 Tables

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2309.14348v2) [paper-pdf](http://arxiv.org/pdf/2309.14348v2)

**Authors**: Bochuan Cao, Yuanpu Cao, Lu Lin, Jinghui Chen

**Abstract**: Recently, Large Language Models (LLMs) have made significant advancements and are now widely used across various domains. Unfortunately, there has been a rising concern that LLMs can be misused to generate harmful or malicious content. Though a line of research has focused on aligning LLMs with human values and preventing them from producing inappropriate content, such alignments are usually vulnerable and can be bypassed by alignment-breaking attacks via adversarially optimized or handcrafted jailbreaking prompts. In this work, we introduce a Robustly Aligned LLM (RA-LLM) to defend against potential alignment-breaking attacks. RA-LLM can be directly constructed upon an existing aligned LLM with a robust alignment checking function, without requiring any expensive retraining or fine-tuning process of the original LLM. Furthermore, we also provide a theoretical analysis for RA-LLM to verify its effectiveness in defending against alignment-breaking attacks. Through real-world experiments on open-source large language models, we demonstrate that RA-LLM can successfully defend against both state-of-the-art adversarial prompts and popular handcrafted jailbreaking prompts by reducing their attack success rates from nearly 100% to around 10% or less.

摘要: 近年来，大型语言模型(LLM)取得了长足的进步，现已广泛应用于各个领域。不幸的是，人们越来越担心LLMS可能被滥用来生成有害或恶意的内容。尽管有一系列研究专注于将LLM与人类价值观保持一致，并防止它们产生不适当的内容，但这种调整通常是脆弱的，可以通过恶意优化或手工制作的越狱提示被破坏顺序的攻击绕过。在这项工作中，我们引入了一种鲁棒对齐LLM(RA-LLM)来防御潜在的对齐破坏攻击。RA-LLM可以直接构建在现有的对准LLM上，具有健壮的对准检查功能，而不需要对原始LLM进行任何昂贵的再培训或微调过程。此外，我们还对RA-LLM进行了理论分析，以验证其在抵抗对齐破坏攻击方面的有效性。通过在开源大型语言模型上的真实世界实验，我们证明了RA-LLM能够成功地防御最新的敌意提示和流行的手工越狱提示，将攻击成功率从近100%降低到10%左右或更低。



## **14. FreqFed: A Frequency Analysis-Based Approach for Mitigating Poisoning Attacks in Federated Learning**

FreqFed：一种基于频率分析的联合学习中毒攻击缓解方法 cs.CR

To appear in the Network and Distributed System Security (NDSS)  Symposium 2024. 16 pages, 8 figures, 12 tables, 1 algorithm, 3 equations

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2312.04432v1) [paper-pdf](http://arxiv.org/pdf/2312.04432v1)

**Authors**: Hossein Fereidooni, Alessandro Pegoraro, Phillip Rieger, Alexandra Dmitrienko, Ahmad-Reza Sadeghi

**Abstract**: Federated learning (FL) is a collaborative learning paradigm allowing multiple clients to jointly train a model without sharing their training data. However, FL is susceptible to poisoning attacks, in which the adversary injects manipulated model updates into the federated model aggregation process to corrupt or destroy predictions (untargeted poisoning) or implant hidden functionalities (targeted poisoning or backdoors). Existing defenses against poisoning attacks in FL have several limitations, such as relying on specific assumptions about attack types and strategies or data distributions or not sufficiently robust against advanced injection techniques and strategies and simultaneously maintaining the utility of the aggregated model. To address the deficiencies of existing defenses, we take a generic and completely different approach to detect poisoning (targeted and untargeted) attacks. We present FreqFed, a novel aggregation mechanism that transforms the model updates (i.e., weights) into the frequency domain, where we can identify the core frequency components that inherit sufficient information about weights. This allows us to effectively filter out malicious updates during local training on the clients, regardless of attack types, strategies, and clients' data distributions. We extensively evaluate the efficiency and effectiveness of FreqFed in different application domains, including image classification, word prediction, IoT intrusion detection, and speech recognition. We demonstrate that FreqFed can mitigate poisoning attacks effectively with a negligible impact on the utility of the aggregated model.

摘要: 联合学习(FL)是一种协作学习范式，允许多个客户在不共享训练数据的情况下联合训练一个模型。然而，FL很容易受到中毒攻击，在这种攻击中，对手将被操纵的模型更新注入到联合模型聚合过程中，以破坏或破坏预测(非定向中毒)或植入隐藏功能(定向中毒或后门)。现有的FL对中毒攻击的防御有几个局限性，例如依赖于对攻击类型和策略或数据分布的特定假设，或者对先进的注入技术和策略不够健壮，同时保持聚集模型的实用性。为了解决现有防御的不足，我们采取了一种通用的、完全不同的方法来检测中毒(目标攻击和非目标攻击)。我们提出了一种新的聚合机制FreqFed，它将模型更新(即权值)转换到频域，在频域中我们可以识别继承了足够的权值信息的核心频率分量。这样，无论攻击类型、策略和客户端的数据分布如何，我们都可以在客户端的本地培训期间有效地过滤掉恶意更新。我们广泛评估了FreqFed在不同应用领域的效率和效果，包括图像分类、单词预测、物联网入侵检测和语音识别。我们证明了FreqFed能够有效地减少中毒攻击，而对聚集模型的效用的影响可以忽略不计。



## **15. OT-Attack: Enhancing Adversarial Transferability of Vision-Language Models via Optimal Transport Optimization**

OT-Attack：通过最优传输优化增强视觉语言模型的对抗性转移 cs.CV

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2312.04403v1) [paper-pdf](http://arxiv.org/pdf/2312.04403v1)

**Authors**: Dongchen Han, Xiaojun Jia, Yang Bai, Jindong Gu, Yang Liu, Xiaochun Cao

**Abstract**: Vision-language pre-training (VLP) models demonstrate impressive abilities in processing both images and text. However, they are vulnerable to multi-modal adversarial examples (AEs). Investigating the generation of high-transferability adversarial examples is crucial for uncovering VLP models' vulnerabilities in practical scenarios. Recent works have indicated that leveraging data augmentation and image-text modal interactions can enhance the transferability of adversarial examples for VLP models significantly. However, they do not consider the optimal alignment problem between dataaugmented image-text pairs. This oversight leads to adversarial examples that are overly tailored to the source model, thus limiting improvements in transferability. In our research, we first explore the interplay between image sets produced through data augmentation and their corresponding text sets. We find that augmented image samples can align optimally with certain texts while exhibiting less relevance to others. Motivated by this, we propose an Optimal Transport-based Adversarial Attack, dubbed OT-Attack. The proposed method formulates the features of image and text sets as two distinct distributions and employs optimal transport theory to determine the most efficient mapping between them. This optimal mapping informs our generation of adversarial examples to effectively counteract the overfitting issues. Extensive experiments across various network architectures and datasets in image-text matching tasks reveal that our OT-Attack outperforms existing state-of-the-art methods in terms of adversarial transferability.

摘要: 视觉语言预训练(VLP)模型在处理图像和文本方面表现出令人印象深刻的能力。然而，它们很容易受到多模式对抗性例子(AE)的攻击。研究高可转移性对抗性实例的生成对于揭示VLP模型在实际场景中的漏洞至关重要。最近的工作表明，利用数据增强和图文模式交互可以显著提高VLP模型中对抗性例子的可转移性。然而，它们没有考虑数据增强的图文对之间的最优对齐问题。这种疏忽导致了过度针对源模型量身定做的对抗性示例，从而限制了可转移性的改进。在我们的研究中，我们首先探讨了通过数据增强产生的图像集与其对应的文本集之间的相互作用。我们发现，增强的图像样本可以最优地与某些文本对齐，而与其他文本相关性较小。受此启发，我们提出了一种基于最优传输的对抗性攻击，称为OT-攻击。该方法将图像和文本的特征描述为两个不同的分布，并利用最优传输理论来确定它们之间的最有效映射。这种最优映射通知我们生成的对抗性例子，以有效地抵消过度适应的问题。在图文匹配任务中，在不同的网络结构和数据集上进行的大量实验表明，我们的OT-攻击在对抗可转移性方面优于现有的最先进的方法。



## **16. Similarity of Neural Architectures using Adversarial Attack Transferability**

基于对抗性攻击传递性的神经结构相似性 cs.LG

20pages, 13 figures, 2.3MB

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2210.11407v3) [paper-pdf](http://arxiv.org/pdf/2210.11407v3)

**Authors**: Jaehui Hwang, Dongyoon Han, Byeongho Heo, Song Park, Sanghyuk Chun, Jong-Seok Lee

**Abstract**: In recent years, many deep neural architectures have been developed for image classification. Whether they are similar or dissimilar and what factors contribute to their (dis)similarities remains curious. To address this question, we aim to design a quantitative and scalable similarity measure between neural architectures. We propose Similarity by Attack Transferability (SAT) from the observation that adversarial attack transferability contains information related to input gradients and decision boundaries widely used to understand model behaviors. We conduct a large-scale analysis on 69 state-of-the-art ImageNet classifiers using our proposed similarity function to answer the question. Moreover, we observe neural architecture-related phenomena using model similarity that model diversity can lead to better performance on model ensembles and knowledge distillation under specific conditions. Our results provide insights into why developing diverse neural architectures with distinct components is necessary.

摘要: 近年来，已经发展了许多用于图像分类的深层神经结构。它们是相似的还是不相似的，以及是什么因素导致了它们(不同)的相似之处，仍然令人好奇。为了解决这个问题，我们的目标是设计一种量化的、可伸缩的神经结构之间的相似性度量。基于对抗性攻击的可移动性包含与输入梯度和决策边界有关的信息，被广泛用于理解模型行为，我们提出了攻击可转移性相似性(SAT)。我们使用我们提出的相似度函数对69个最先进的ImageNet分类器进行了大规模的分析。此外，我们使用模型相似性来观察与神经结构相关的现象，即在特定条件下，模型多样性可以在模型集成和知识提取方面带来更好的性能。我们的结果为为什么开发具有不同组件的不同神经架构提供了洞察力。



## **17. Temporal Shuffling for Defending Deep Action Recognition Models against Adversarial Attacks**

用于防御对抗性攻击的深度动作识别模型的时间洗牌 cs.CV

12 pages, accepted to Neural Networks

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2112.07921v2) [paper-pdf](http://arxiv.org/pdf/2112.07921v2)

**Authors**: Jaehui Hwang, Huan Zhang, Jun-Ho Choi, Cho-Jui Hsieh, Jong-Seok Lee

**Abstract**: Recently, video-based action recognition methods using convolutional neural networks (CNNs) achieve remarkable recognition performance. However, there is still lack of understanding about the generalization mechanism of action recognition models. In this paper, we suggest that action recognition models rely on the motion information less than expected, and thus they are robust to randomization of frame orders. Furthermore, we find that motion monotonicity remaining after randomization also contributes to such robustness. Based on this observation, we develop a novel defense method using temporal shuffling of input videos against adversarial attacks for action recognition models. Another observation enabling our defense method is that adversarial perturbations on videos are sensitive to temporal destruction. To the best of our knowledge, this is the first attempt to design a defense method without additional training for 3D CNN-based video action recognition models.

摘要: 最近，基于视频的动作识别方法使用卷积神经网络（CNN）实现了显着的识别性能。然而，人们对动作识别模型的泛化机制还缺乏了解。在本文中，我们建议，动作识别模型依赖于运动信息比预期的少，因此，他们是鲁棒的随机帧顺序。此外，我们发现，随机化后的运动单调性也有助于这种鲁棒性。基于这一观察，我们开发了一种新的防御方法，使用输入视频的时间重排来对抗动作识别模型的对抗性攻击。使我们的防御方法成为可能的另一个观察结果是，视频上的对抗性扰动对时间破坏很敏感。据我们所知，这是第一次尝试设计一种防御方法，而无需对基于3D CNN的视频动作识别模型进行额外的训练。



## **18. Experimentally Certified Transmission of a Quantum Message through an Untrusted and Lossy Quantum Channel via Bell's Theorem**

通过贝尔定理在不可信和有损耗的量子信道中实验验证量子消息的传输 quant-ph

35 pages, 14 figures

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2304.09605v2) [paper-pdf](http://arxiv.org/pdf/2304.09605v2)

**Authors**: Simon Neves, Laura dos Santos Martins, Verena Yacoub, Pascal Lefebvre, Ivan Supic, Damian Markham, Eleni Diamanti

**Abstract**: Quantum transmission links are central elements in essentially all protocols involving the exchange of quantum messages. Emerging progress in quantum technologies involving such links needs to be accompanied by appropriate certification tools. In adversarial scenarios, a certification method can be vulnerable to attacks if too much trust is placed on the underlying system. Here, we propose a protocol in a device independent framework, which allows for the certification of practical quantum transmission links in scenarios where minimal assumptions are made about the functioning of the certification setup. In particular, we take unavoidable transmission losses into account by modeling the link as a completely-positive trace-decreasing map. We also, crucially, remove the assumption of independent and identically distributed samples, which is known to be incompatible with adversarial settings. Finally, in view of the use of the certified transmitted states for follow-up applications, our protocol moves beyond certification of the channel to allow us to estimate the quality of the transmitted quantum message itself. To illustrate the practical relevance and the feasibility of our protocol with currently available technology we provide an experimental implementation based on a state-of-the-art polarization entangled photon pair source in a Sagnac configuration and analyze its robustness for realistic losses and errors.

摘要: 量子传输链路是基本上所有涉及量子消息交换的协议中的核心元素。涉及这种联系的量子技术的新进展需要有适当的认证工具。在对抗性场景中，如果对底层系统过于信任，则认证方法可能容易受到攻击。在这里，我们提出了一个协议，在一个设备独立的框架，它允许认证的实际量子传输链路的情况下，最小的假设是关于认证设置的功能。特别是，我们考虑到不可避免的传输损耗建模的链路作为一个完全积极的痕迹减少地图。至关重要的是，我们还删除了独立和同分布样本的假设，这与对抗性设置不兼容。最后，考虑到后续应用中使用的认证传输状态，我们的协议超越了通道的认证，使我们能够估计传输的量子消息本身的质量。为了说明我们的协议与现有技术的实际相关性和可行性，我们提供了一个实验实现的基础上，最先进的偏振纠缠光子对源在Sagnac配置和分析其鲁棒性的现实损失和错误。



## **19. Adv-4-Adv: Thwarting Changing Adversarial Perturbations via Adversarial Domain Adaptation**

ADV-4-ADV：通过对抗性领域适应挫败不断变化的对抗性扰动 cs.CV

22 pages

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2112.00428v3) [paper-pdf](http://arxiv.org/pdf/2112.00428v3)

**Authors**: Tianyue Zheng, Zhe Chen, Shuya Ding, Chao Cai, Jun Luo

**Abstract**: Whereas adversarial training can be useful against specific adversarial perturbations, they have also proven ineffective in generalizing towards attacks deviating from those used for training. However, we observe that this ineffectiveness is intrinsically connected to domain adaptability, another crucial issue in deep learning for which adversarial domain adaptation appears to be a promising solution. Consequently, we proposed Adv-4-Adv as a novel adversarial training method that aims to retain robustness against unseen adversarial perturbations. Essentially, Adv-4-Adv treats attacks incurring different perturbations as distinct domains, and by leveraging the power of adversarial domain adaptation, it aims to remove the domain/attack-specific features. This forces a trained model to learn a robust domain-invariant representation, which in turn enhances its generalization ability. Extensive evaluations on Fashion-MNIST, SVHN, CIFAR-10, and CIFAR-100 demonstrate that a model trained by Adv-4-Adv based on samples crafted by simple attacks (e.g., FGSM) can be generalized to more advanced attacks (e.g., PGD), and the performance exceeds state-of-the-art proposals on these datasets.

摘要: 虽然对抗性训练对对抗特定的对抗性干扰是有用的，但事实证明，它们也不能有效地概括出偏离用于训练的攻击。然而，我们观察到这种低效与领域适应性有内在联系，这是深度学习中的另一个关键问题，对抗性领域适应似乎是一个有希望的解决方案。因此，我们提出了ADV-4-ADV作为一种新的对抗性训练方法，旨在保持对不可见的对抗性扰动的鲁棒性。从本质上讲，ADV-4-ADV将引起不同扰动的攻击视为不同的域，并通过利用对抗性域自适应的能力，旨在去掉域/攻击特定的特征。这迫使训练的模型学习健壮的领域不变表示，从而增强其泛化能力。在Fashion-MNIST、SVHN、CIFAR-10和CIFAR-100上的广泛评估表明，ADV-4-ADV基于简单攻击(例如FGSM)生成的样本训练的模型可以推广到更高级的攻击(例如PGD)，并且性能超过了这些数据集的最新建议。



## **20. Point Cloud Attacks in Graph Spectral Domain: When 3D Geometry Meets Graph Signal Processing**

图谱域中的点云攻击：当3D几何遇到图信号处理时 cs.CV

Accepted to IEEE Transactions on Pattern Analysis and Machine  Intelligence (TPAMI). arXiv admin note: substantial text overlap with  arXiv:2202.07261

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2207.13326v2) [paper-pdf](http://arxiv.org/pdf/2207.13326v2)

**Authors**: Daizong Liu, Wei Hu, Xin Li

**Abstract**: With the increasing attention in various 3D safety-critical applications, point cloud learning models have been shown to be vulnerable to adversarial attacks. Although existing 3D attack methods achieve high success rates, they delve into the data space with point-wise perturbation, which may neglect the geometric characteristics. Instead, we propose point cloud attacks from a new perspective -- the graph spectral domain attack, aiming to perturb graph transform coefficients in the spectral domain that corresponds to varying certain geometric structure. Specifically, leveraging on graph signal processing, we first adaptively transform the coordinates of points onto the spectral domain via graph Fourier transform (GFT) for compact representation. Then, we analyze the influence of different spectral bands on the geometric structure, based on which we propose to perturb the GFT coefficients via a learnable graph spectral filter. Considering the low-frequency components mainly contribute to the rough shape of the 3D object, we further introduce a low-frequency constraint to limit perturbations within imperceptible high-frequency components. Finally, the adversarial point cloud is generated by transforming the perturbed spectral representation back to the data domain via the inverse GFT. Experimental results demonstrate the effectiveness of the proposed attack in terms of both the imperceptibility and attack success rates.

摘要: 随着各种3D安全关键应用的日益关注，点云学习模型已被证明容易受到对抗性攻击。虽然现有的3D攻击方法取得了很高的成功率，他们深入到数据空间与逐点扰动，这可能会忽略几何特征。相反，我们提出了点云攻击从一个新的角度-图谱域攻击，旨在扰动图变换系数在谱域中对应于改变一定的几何结构。具体地说，利用图形信号处理，我们首先自适应地通过图形傅立叶变换（GFT）将点的坐标变换到谱域上以进行紧凑表示。然后，我们分析了不同的光谱波段上的几何结构的影响，在此基础上，我们提出了扰动的GFT系数通过一个可学习的图谱滤波器。考虑到低频分量主要对3D对象的粗糙形状有贡献，我们进一步引入低频约束以限制不可感知的高频分量内的扰动。最后，通过逆GFT将扰动的谱表示变换回数据域来生成对抗点云。实验结果表明，所提出的攻击的不可见性和攻击成功率方面的有效性。



## **21. Defense against ML-based Power Side-channel Attacks on DNN Accelerators with Adversarial Attacks**

对抗攻击下DNN加速器的基于ML的功率侧信道攻击防御 cs.CR

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2312.04035v1) [paper-pdf](http://arxiv.org/pdf/2312.04035v1)

**Authors**: Xiaobei Yan, Chip Hong Chang, Tianwei Zhang

**Abstract**: Artificial Intelligence (AI) hardware accelerators have been widely adopted to enhance the efficiency of deep learning applications. However, they also raise security concerns regarding their vulnerability to power side-channel attacks (SCA). In these attacks, the adversary exploits unintended communication channels to infer sensitive information processed by the accelerator, posing significant privacy and copyright risks to the models. Advanced machine learning algorithms are further employed to facilitate the side-channel analysis and exacerbate the privacy issue of AI accelerators. Traditional defense strategies naively inject execution noise to the runtime of AI models, which inevitably introduce large overheads.   In this paper, we present AIAShield, a novel defense methodology to safeguard FPGA-based AI accelerators and mitigate model extraction threats via power-based SCAs. The key insight of AIAShield is to leverage the prominent adversarial attack technique from the machine learning community to craft delicate noise, which can significantly obfuscate the adversary's side-channel observation while incurring minimal overhead to the execution of the protected model. At the hardware level, we design a new module based on ring oscillators to achieve fine-grained noise generation. At the algorithm level, we repurpose Neural Architecture Search to worsen the adversary's extraction results. Extensive experiments on the Nvidia Deep Learning Accelerator (NVDLA) demonstrate that AIAShield outperforms existing solutions with excellent transferability.

摘要: 人工智能(AI)硬件加速器已被广泛采用，以提高深度学习应用的效率。然而，它们也提出了安全方面的担忧，即它们易受电源旁路攻击(SCA)的攻击。在这些攻击中，攻击者利用非故意的通信渠道来推断加速器处理的敏感信息，给模型带来了严重的隐私和版权风险。进一步使用了先进的机器学习算法来促进旁路分析，并加剧了AI加速器的隐私问题。传统的防御策略幼稚地向AI模型的运行时注入执行噪声，这不可避免地引入了较大的开销。在本文中，我们提出了一种新的防御方法AIAShield，它通过基于功率的SCA来保护基于FPGA的AI加速器并缓解模型提取威胁。AIAShield的关键洞察力是利用机器学习社区的突出对手攻击技术来制造微妙的噪声，这种噪声可以显著混淆对手的侧通道观察，同时对执行受保护的模型产生最小的开销。在硬件层面，设计了一种基于环形振荡器的新型噪声产生模块，实现了细粒度噪声产生。在算法层面，我们改变了神经结构搜索的用途，以恶化对手的提取结果。在NVIDIA深度学习加速器(NVDLA)上的广泛实验表明，AIAShield的性能优于现有解决方案，具有出色的可移植性。



## **22. Characterizing the Optimal 0-1 Loss for Multi-class Classification with a Test-time Attacker**

具有测试时间攻击的多类分类最优0-1损失的刻画 cs.LG

NeurIPS 2023 Spotlight

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2302.10722v2) [paper-pdf](http://arxiv.org/pdf/2302.10722v2)

**Authors**: Sihui Dai, Wenxin Ding, Arjun Nitin Bhagoji, Daniel Cullina, Ben Y. Zhao, Haitao Zheng, Prateek Mittal

**Abstract**: Finding classifiers robust to adversarial examples is critical for their safe deployment. Determining the robustness of the best possible classifier under a given threat model for a given data distribution and comparing it to that achieved by state-of-the-art training methods is thus an important diagnostic tool. In this paper, we find achievable information-theoretic lower bounds on loss in the presence of a test-time attacker for multi-class classifiers on any discrete dataset. We provide a general framework for finding the optimal 0-1 loss that revolves around the construction of a conflict hypergraph from the data and adversarial constraints. We further define other variants of the attacker-classifier game that determine the range of the optimal loss more efficiently than the full-fledged hypergraph construction. Our evaluation shows, for the first time, an analysis of the gap to optimal robustness for classifiers in the multi-class setting on benchmark datasets.

摘要: 寻找对敌意例子具有健壮性的分类器对于它们的安全部署至关重要。因此，确定在给定数据分布的给定威胁模型下的最佳可能分类器的稳健性，并将其与最先进的训练方法所实现的稳健性进行比较，是一种重要的诊断工具。在这篇文章中，我们找到了在任意离散数据集上的多类分类器在测试时间攻击者存在的情况下可获得的信息论损失下界。我们提供了一个寻找最优0-1损失的一般框架，该框架围绕着从数据和对抗性约束构造冲突超图。我们进一步定义了攻击者-分类器博弈的其他变体，它们比完整的超图构造更有效地确定最优损失的范围。我们的评估首次分析了在基准数据集上的多类设置下分类器的最优稳健性的差距。



## **23. Dr. Jekyll and Mr. Hyde: Two Faces of LLMs**

杰基尔博士和海德先生：LLMS的两张面孔 cs.CR

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2312.03853v1) [paper-pdf](http://arxiv.org/pdf/2312.03853v1)

**Authors**: Matteo Gioele Collu, Tom Janssen-Groesbeek, Stefanos Koffas, Mauro Conti, Stjepan Picek

**Abstract**: This year, we witnessed a rise in the use of Large Language Models, especially when combined with applications like chatbot assistants. Safety mechanisms and specialized training procedures are put in place to prevent improper responses from these assistants. In this work, we bypass these measures for ChatGPT and Bard (and, to some extent, Bing chat) by making them impersonate complex personas with opposite characteristics as those of the truthful assistants they are supposed to be. We start by creating elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversation followed a role-play style to get the response the assistant was not allowed to provide. By making use of personas, we show that the response that is prohibited is actually provided, making it possible to obtain unauthorized, illegal, or harmful information. This work shows that by using adversarial personas, one can overcome safety mechanisms set out by ChatGPT and Bard. It also introduces several ways of activating such adversarial personas, altogether showing that both chatbots are vulnerable to this kind of attack.

摘要: 今年，我们见证了大型语言模型的使用增加，特别是与聊天机器人助手等应用程序结合使用时。建立了安全机制和专门的培训程序，以防止这些助理做出不当反应。在这项工作中，我们绕过了ChatGPT和Bard(在某种程度上，还有Bing聊天)的这些措施，让他们模仿复杂的人物角色，具有与他们应该是的诚实助手相反的特征。我们首先为这些角色创建精致的传记，然后在与相同的聊天机器人的新会话中使用。我们的谈话遵循了角色扮演的风格，得到了助手不允许提供的回应。通过使用人物角色，我们表明实际上提供了被禁止的响应，使得获得未经授权的、非法的或有害的信息成为可能。这项工作表明，通过使用对抗性人物角色，一个人可以克服ChatGPT和Bard提出的安全机制。它还介绍了几种激活这种敌对角色的方法，共同表明这两个聊天机器人都容易受到这种攻击。



## **24. Memory Triggers: Unveiling Memorization in Text-To-Image Generative Models through Word-Level Duplication**

记忆触发器：通过词级复制揭示文本到图像生成模型中的记忆 cs.CR

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2312.03692v1) [paper-pdf](http://arxiv.org/pdf/2312.03692v1)

**Authors**: Ali Naseh, Jaechul Roh, Amir Houmansadr

**Abstract**: Diffusion-based models, such as the Stable Diffusion model, have revolutionized text-to-image synthesis with their ability to produce high-quality, high-resolution images. These advancements have prompted significant progress in image generation and editing tasks. However, these models also raise concerns due to their tendency to memorize and potentially replicate exact training samples, posing privacy risks and enabling adversarial attacks. Duplication in training datasets is recognized as a major factor contributing to memorization, and various forms of memorization have been studied so far. This paper focuses on two distinct and underexplored types of duplication that lead to replication during inference in diffusion-based models, particularly in the Stable Diffusion model. We delve into these lesser-studied duplication phenomena and their implications through two case studies, aiming to contribute to the safer and more responsible use of generative models in various applications.

摘要: 基于扩散的模型，如稳定扩散模型，以其生成高质量、高分辨率图像的能力，使文本到图像的合成发生了革命性的变化。这些进步推动了图像生成和编辑任务的重大进步。然而，这些模型也引起了人们的担忧，因为它们倾向于记忆并可能复制准确的训练样本，这会带来隐私风险，并使对抗性攻击成为可能。训练数据集的重复被认为是导致记忆的主要因素，到目前为止，人们已经研究了各种形式的记忆。在基于扩散的模型中，特别是在稳定扩散模型中，本文重点研究了两种不同的、未被充分研究的复制类型，它们在基于扩散的模型中的推理过程中导致复制。我们通过两个案例研究来深入研究这些较少研究的复制现象及其影响，旨在有助于在各种应用中更安全和更负责任地使用生成模型。



## **25. Temporal Robustness against Data Poisoning**

抗数据中毒的时间稳健性 cs.LG

37th Conference on Neural Information Processing Systems (NeurIPS  2023)

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2302.03684v3) [paper-pdf](http://arxiv.org/pdf/2302.03684v3)

**Authors**: Wenxiao Wang, Soheil Feizi

**Abstract**: Data poisoning considers cases when an adversary manipulates the behavior of machine learning algorithms through malicious training data. Existing threat models of data poisoning center around a single metric, the number of poisoned samples. In consequence, if attackers can poison more samples than expected with affordable overhead, as in many practical scenarios, they may be able to render existing defenses ineffective in a short time. To address this issue, we leverage timestamps denoting the birth dates of data, which are often available but neglected in the past. Benefiting from these timestamps, we propose a temporal threat model of data poisoning with two novel metrics, earliness and duration, which respectively measure how long an attack started in advance and how long an attack lasted. Using these metrics, we define the notions of temporal robustness against data poisoning, providing a meaningful sense of protection even with unbounded amounts of poisoned samples when the attacks are temporally bounded. We present a benchmark with an evaluation protocol simulating continuous data collection and periodic deployments of updated models, thus enabling empirical evaluation of temporal robustness. Lastly, we develop and also empirically verify a baseline defense, namely temporal aggregation, offering provable temporal robustness and highlighting the potential of our temporal threat model for data poisoning.

摘要: 数据中毒考虑对手通过恶意训练数据操纵机器学习算法的行为的情况。现有的数据中毒威胁模型主要以中毒样本数量这一单一指标为中心。因此，如果攻击者可以用负担得起的开销毒化比预期更多的样本，就像在许多实际场景中一样，他们可能能够在短时间内使现有防御系统失效。为了解决这个问题，我们利用表示数据出生日期的时间戳，这些数据通常是可用的，但在过去被忽视了。得益于这些时间戳，我们提出了一种数据中毒的时态威胁模型，该模型采用了两个新的度量标准：提前期和持续时间，分别衡量攻击提前开始的时间和攻击持续的时间。使用这些度量，我们定义了针对数据中毒的时间健壮性的概念，即使在攻击是时间有界的情况下，也提供了一种有意义的保护感，即使是在无限数量的有毒样本的情况下。我们提出了一个基准测试和评估协议，模拟了连续的数据收集和定期部署更新的模型，从而能够对时间稳健性进行经验评估。最后，我们开发了一个基线防御，也就是时间聚合，提供了可证明的时间稳健性，并突出了我们的时间威胁模型对数据中毒的潜力。



## **26. PyraTrans: Attention-Enriched Pyramid Transformer for Malicious URL Detection**

金字塔：用于恶意URL检测的高关注度金字塔转换器 cs.CR

12 pages, 7 figures

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2312.00508v2) [paper-pdf](http://arxiv.org/pdf/2312.00508v2)

**Authors**: Ruitong Liu, Yanbin Wang, Zhenhao Guo, Haitao Xu, Zhan Qin, Wenrui Ma, Fan Zhang

**Abstract**: Although advancements in machine learning have driven the development of malicious URL detection technology, current techniques still face significant challenges in their capacity to generalize and their resilience against evolving threats. In this paper, we propose PyraTrans, a novel method that integrates pretrained Transformers with pyramid feature learning to detect malicious URL. PyraTrans utilizes a pretrained CharBERT as its foundation and is augmented with three interconnected feature modules: 1) Encoder Feature Extraction, extracting multi-order feature matrices from each CharBERT encoder layer; 2) Multi-Scale Feature Learning, capturing local contextual insights at various scales and aggregating information across encoder layers; and 3) Spatial Pyramid Attention, focusing on regional-level attention to emphasize areas rich in expressive information. The proposed approach addresses the limitations of the Transformer in local feature learning and regional relational awareness, which are vital for capturing URL-specific word patterns, character combinations, or structural anomalies. In several challenging experimental scenarios, the proposed method has shown significant improvements in accuracy, generalization, and robustness in malicious URL detection. For instance, it achieved a peak F1-score improvement of 40% in class-imbalanced scenarios, and exceeded the best baseline result by 14.13% in accuracy in adversarial attack scenarios. Additionally, we conduct a case study where our method accurately identifies all 30 active malicious web pages, whereas two pior SOTA methods miss 4 and 7 malicious web pages respectively. Codes and data are available at:https://github.com/Alixyvtte/PyraTrans.

摘要: 尽管机器学习的进步推动了恶意URL检测技术的发展，但当前的技术在泛化能力和对不断变化的威胁的弹性方面仍然面临着巨大的挑战。在本文中，我们提出了一种将预先训练的变换和金字塔特征学习相结合的检测恶意URL的新方法--PyraTrans。金字塔利用预先训练的CharBERT作为其基础，并增加了三个相互关联的特征模块：1)编码器特征提取，从每个CharBERT编码层提取多阶特征矩阵；2)多尺度特征学习，捕获不同尺度上的局部上下文洞察力，并聚合编码层之间的信息；以及3)空间金字塔关注，专注于区域级别的关注，以强调具有丰富表现力信息的区域。提出的方法解决了Transformer在本地特征学习和区域关系感知方面的局限性，这对于捕获特定于URL的单词模式、字符组合或结构异常至关重要。在几个具有挑战性的实验场景中，所提出的方法在恶意URL检测的准确性、泛化和稳健性方面显示出显著的改进。例如，在班级不平衡场景下，它的F1得分峰值提高了40%，在对抗性攻击场景中，它的准确率超过了最佳基线结果14.13%。此外，我们还进行了一个案例研究，其中我们的方法准确地识别了所有30个活跃的恶意网页，而两种高级SOTA方法分别漏掉了4个和7个恶意网页。代码和数据可在以下网址获得：https://github.com/Alixyvtte/PyraTrans.



## **27. Defense Against Adversarial Attacks using Convolutional Auto-Encoders**

利用卷积自动编码器防御对抗性攻击 cs.CV

9 pages, 6 figures, 3 tables

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2312.03520v1) [paper-pdf](http://arxiv.org/pdf/2312.03520v1)

**Authors**: Shreyasi Mandal

**Abstract**: Deep learning models, while achieving state-of-the-art performance on many tasks, are susceptible to adversarial attacks that exploit inherent vulnerabilities in their architectures. Adversarial attacks manipulate the input data with imperceptible perturbations, causing the model to misclassify the data or produce erroneous outputs. This work is based on enhancing the robustness of targeted classifier models against adversarial attacks. To achieve this, an convolutional autoencoder-based approach is employed that effectively counters adversarial perturbations introduced to the input images. By generating images closely resembling the input images, the proposed methodology aims to restore the model's accuracy.

摘要: 深度学习模型虽然在许多任务上实现了最先进的性能，但很容易受到利用其体系结构中固有漏洞的对手攻击。对抗性攻击以不可察觉的扰动操纵输入数据，导致模型对数据进行错误分类或产生错误的输出。这项工作的基础是增强目标分类器模型对对手攻击的稳健性。为了实现这一点，采用了一种基于卷积自动编码器的方法，该方法有效地对抗了引入到输入图像的对抗性扰动。通过生成与输入图像非常相似的图像，所提出的方法旨在恢复模型的准确性。



## **28. Quantum-secured single-pixel imaging under general spoofing attack**

一般欺骗攻击下的量子安全单像素成像 quant-ph

9 pages, 6 figures

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2312.03465v1) [paper-pdf](http://arxiv.org/pdf/2312.03465v1)

**Authors**: Jaesung Heo, Taek Jeong, Nam Hun Park, Yonggi Jo

**Abstract**: In this paper, we introduce a quantum-secured single-pixel imaging (QS-SPI) technique designed to withstand spoofing attacks, wherein adversaries attempt to deceive imaging systems with fake signals. Unlike previous quantum-secured protocols that impose a threshold error rate limiting their operation, even with the existence of true signals, our approach not only identifies spoofing attacks but also facilitates the reconstruction of a true image. Our method involves the analysis of a specific mode correlation of a photon-pair, which is independent of the mode used for image construction, to check security. Through this analysis, we can identify both the targeted image region by the attack and the type of spoofing attack, enabling reconstruction of the true image. A proof-of-principle demonstration employing polarization-correlation of a photon-pair is provided, showcasing successful image reconstruction even under the condition of spoofing signals 2000 times stronger than the true signals. We expect our approach to be applied to quantum-secured signal processing such as quantum target detection or ranging.

摘要: 在这篇文章中，我们介绍了一种量子安全单像素成像(QS-SPI)技术，旨在抵抗欺骗攻击，其中攻击者试图用虚假信号欺骗成像系统。与以前的量子安全协议不同，即使在真实信号存在的情况下，我们的方法也会施加阈值错误率来限制它们的操作，我们的方法不仅可以识别欺骗攻击，还可以帮助重建真实图像。我们的方法包括分析与用于图像构建的模式无关的光子对的特定模式相关性，以检查安全性。通过这种分析，我们可以根据攻击和欺骗攻击的类型来识别目标图像区域，从而能够重建出真实的图像。提供了使用光子对的偏振相关的原理证明演示，展示了即使在欺骗信号比真实信号强2000倍的情况下也成功地重建图像。我们希望将我们的方法应用于量子安全信号处理，如量子目标检测或测距。



## **29. Synthesizing Physical Backdoor Datasets: An Automated Framework Leveraging Deep Generative Models**

综合物理后门数据集：利用深度生成模型的自动化框架 cs.CR

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2312.03419v1) [paper-pdf](http://arxiv.org/pdf/2312.03419v1)

**Authors**: Sze Jue Yang, Chinh D. La, Quang H. Nguyen, Eugene Bagdasaryan, Kok-Seng Wong, Anh Tuan Tran, Chee Seng Chan, Khoa D. Doan

**Abstract**: Backdoor attacks, representing an emerging threat to the integrity of deep neural networks, have garnered significant attention due to their ability to compromise deep learning systems clandestinely. While numerous backdoor attacks occur within the digital realm, their practical implementation in real-world prediction systems remains limited and vulnerable to disturbances in the physical world. Consequently, this limitation has given rise to the development of physical backdoor attacks, where trigger objects manifest as physical entities within the real world. However, creating the requisite dataset to train or evaluate a physical backdoor model is a daunting task, limiting the backdoor researchers and practitioners from studying such physical attack scenarios. This paper unleashes a recipe that empowers backdoor researchers to effortlessly create a malicious, physical backdoor dataset based on advances in generative modeling. Particularly, this recipe involves 3 automatic modules: suggesting the suitable physical triggers, generating the poisoned candidate samples (either by synthesizing new samples or editing existing clean samples), and finally refining for the most plausible ones. As such, it effectively mitigates the perceived complexity associated with creating a physical backdoor dataset, transforming it from a daunting task into an attainable objective. Extensive experiment results show that datasets created by our "recipe" enable adversaries to achieve an impressive attack success rate on real physical world data and exhibit similar properties compared to previous physical backdoor attack studies. This paper offers researchers a valuable toolkit for studies of physical backdoors, all within the confines of their laboratories.

摘要: 后门攻击对深度神经网络的完整性构成了新的威胁，由于它们能够秘密地危害深度学习系统，因此引起了极大的关注。虽然在数字领域内发生了许多后门攻击，但它们在现实世界预测系统中的实际实施仍然有限，容易受到物理世界的干扰。因此，这种限制导致了物理后门攻击的发展，其中触发器对象在真实世界中表现为物理实体。然而，创建必要的数据集来训练或评估物理后门模型是一项艰巨的任务，限制了后门研究人员和实践者研究此类物理攻击场景。这篇文章揭示了一个配方，它使后门研究人员能够基于生成性建模的进步，毫不费力地创建恶意的物理后门数据集。特别是，这个配方涉及三个自动模块：建议合适的物理触发器，生成中毒的候选样本(通过合成新样本或编辑现有的干净样本)，最后提炼出最可信的样本。因此，它有效地减轻了与创建物理后门数据集相关的感知复杂性，将其从令人望而生畏的任务转变为可实现的目标。大量的实验结果表明，由我们的“配方”创建的数据集使攻击者能够在真实的物理世界数据上获得令人印象深刻的攻击成功率，并显示出与之前的物理后门攻击研究类似的特性。这篇论文为研究人员提供了一个宝贵的工具包，用于研究物理后门，所有这些都在他们的实验室范围内。



## **30. SAIF: Sparse Adversarial and Imperceptible Attack Framework**

SAIF：稀疏对抗性和不可察觉攻击框架 cs.CV

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2212.07495v2) [paper-pdf](http://arxiv.org/pdf/2212.07495v2)

**Authors**: Tooba Imtiaz, Morgan Kohler, Jared Miller, Zifeng Wang, Mario Sznaier, Octavia Camps, Jennifer Dy

**Abstract**: Adversarial attacks hamper the decision-making ability of neural networks by perturbing the input signal. The addition of calculated small distortion to images, for instance, can deceive a well-trained image classification network. In this work, we propose a novel attack technique called Sparse Adversarial and Interpretable Attack Framework (SAIF). Specifically, we design imperceptible attacks that contain low-magnitude perturbations at a small number of pixels and leverage these sparse attacks to reveal the vulnerability of classifiers. We use the Frank-Wolfe (conditional gradient) algorithm to simultaneously optimize the attack perturbations for bounded magnitude and sparsity with $O(1/\sqrt{T})$ convergence. Empirical results show that SAIF computes highly imperceptible and interpretable adversarial examples, and outperforms state-of-the-art sparse attack methods on the ImageNet dataset.

摘要: 对抗性攻击通过干扰输入信号来阻碍神经网络的决策能力。例如，将计算的小失真添加到图像可以欺骗训练有素的图像分类网络。在这项工作中，我们提出了一种新的攻击技术，称为稀疏对抗性和可解释攻击框架(SAIF)。具体地说，我们设计了在少量像素处包含低幅度扰动的不可察觉攻击，并利用这些稀疏攻击来揭示分类器的脆弱性。我们使用Frank-Wolfe(条件梯度)算法来同时优化有界模和稀疏性的攻击扰动，并且具有$O(1/\Sqrt{T})$收敛。实验结果表明，该算法能够计算高度不可察觉和可解释的敌意实例，并且在ImageNet数据集上的性能优于最新的稀疏攻击方法。



## **31. Privacy-Preserving Task-Oriented Semantic Communications Against Model Inversion Attacks**

面向任务的抗模型反演攻击的隐私保护语义通信 cs.IT

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2312.03252v1) [paper-pdf](http://arxiv.org/pdf/2312.03252v1)

**Authors**: Yanhu Wang, Shuaishuai Guo, Yiqin Deng, Haixia Zhang, Yuguang Fang

**Abstract**: Semantic communication has been identified as a core technology for the sixth generation (6G) of wireless networks. Recently, task-oriented semantic communications have been proposed for low-latency inference with limited bandwidth. Although transmitting only task-related information does protect a certain level of user privacy, adversaries could apply model inversion techniques to reconstruct the raw data or extract useful information, thereby infringing on users' privacy. To mitigate privacy infringement, this paper proposes an information bottleneck and adversarial learning (IBAL) approach to protect users' privacy against model inversion attacks. Specifically, we extract task-relevant features from the input based on the information bottleneck (IB) theory. To overcome the difficulty in calculating the mutual information in high-dimensional space, we derive a variational upper bound to estimate the true mutual information. To prevent data reconstruction from task-related features by adversaries, we leverage adversarial learning to train encoder to fool adversaries by maximizing reconstruction distortion. Furthermore, considering the impact of channel variations on privacy-utility trade-off and the difficulty in manually tuning the weights of each loss, we propose an adaptive weight adjustment method. Numerical results demonstrate that the proposed approaches can effectively protect privacy without significantly affecting task performance and achieve better privacy-utility trade-offs than baseline methods.

摘要: 语义通信已被确定为第六代(6G)无线网络的核心技术。最近，面向任务的语义通信被提出用于在有限带宽下进行低延迟推理。虽然只传输与任务相关的信息确实保护了一定程度的用户隐私，但攻击者可以应用模型反转技术来重建原始数据或提取有用信息，从而侵犯用户隐私。为了减少隐私侵犯，提出了一种信息瓶颈和对抗学习(IBAL)方法来保护用户的隐私免受模型反转攻击。具体地，我们基于信息瓶颈(IB)理论从输入中提取与任务相关的特征。为了克服高维空间互信息计算的困难，我们给出了一个估计真互信息的变分上界。为了防止敌手从任务相关特征中重建数据，我们利用对抗性学习来训练编码者通过最大化重建失真来愚弄对手。此外，考虑到信道变化对隐私-效用权衡的影响，以及人工调整每次损失的权重的难度，我们提出了一种自适应的权重调整方法。数值结果表明，所提出的方法能够在不显著影响任务性能的情况下有效地保护隐私，并且取得了比基线方法更好的隐私效用折衷。



## **32. A Simple Framework to Enhance the Adversarial Robustness of Deep Learning-based Intrusion Detection System**

一种增强基于深度学习的入侵检测系统抗攻击能力的简单框架 cs.CR

Accepted by Computers & Security

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2312.03245v1) [paper-pdf](http://arxiv.org/pdf/2312.03245v1)

**Authors**: Xinwei Yuan, Shu Han, Wei Huang, Hongliang Ye, Xianglong Kong, Fan Zhang

**Abstract**: Deep learning based intrusion detection systems (DL-based IDS) have emerged as one of the best choices for providing security solutions against various network intrusion attacks. However, due to the emergence and development of adversarial deep learning technologies, it becomes challenging for the adoption of DL models into IDS. In this paper, we propose a novel IDS architecture that can enhance the robustness of IDS against adversarial attacks by combining conventional machine learning (ML) models and Deep Learning models. The proposed DLL-IDS consists of three components: DL-based IDS, adversarial example (AE) detector, and ML-based IDS. We first develop a novel AE detector based on the local intrinsic dimensionality (LID). Then, we exploit the low attack transferability between DL models and ML models to find a robust ML model that can assist us in determining the maliciousness of AEs. If the input traffic is detected as an AE, the ML-based IDS will predict the maliciousness of input traffic, otherwise the DL-based IDS will work for the prediction. The fusion mechanism can leverage the high prediction accuracy of DL models and low attack transferability between DL models and ML models to improve the robustness of the whole system. In our experiments, we observe a significant improvement in the prediction performance of the IDS when subjected to adversarial attack, achieving high accuracy with low resource consumption.

摘要: 基于深度学习的入侵检测系统已经成为针对各种网络入侵攻击提供安全解决方案的最佳选择之一。然而，随着对抗性深度学习技术的出现和发展，将动态链式学习模型应用到入侵检测系统中变得越来越困难。本文将传统的机器学习模型和深度学习模型相结合，提出了一种新的入侵检测体系结构，能够增强入侵检测系统对敌意攻击的健壮性。提出的动态链接库-入侵检测系统由三部分组成：基于动态链接库的入侵检测系统、对抗性实例(AE)检测器和基于ML的入侵检测系统。我们首先提出了一种基于局部内禀维度(LID)的新型声发射检测器。然后，我们利用DL模型和ML模型之间的低攻击传递性来找到一个健壮的ML模型，该模型可以帮助我们确定攻击实体的恶意程度。如果输入流量被检测为AE，则基于ML的入侵检测系统将预测输入流量的恶意程度，否则，基于DL的入侵检测系统将进行预测。该融合机制利用了DL模型的高预测精度和DL模型与ML模型之间较低的攻击传递性，提高了整个系统的鲁棒性。在我们的实验中，我们观察到入侵检测系统在遭受敌意攻击时预测性能有了显著的提高，在低资源消耗的情况下达到了高精度。



## **33. Model-tuning Via Prompts Makes NLP Models Adversarially Robust**

通过提示进行模型调整使NLP模型变得异常健壮 cs.CL

Accepted to the EMNLP 2023 Conference

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2303.07320v2) [paper-pdf](http://arxiv.org/pdf/2303.07320v2)

**Authors**: Mrigank Raman, Pratyush Maini, J. Zico Kolter, Zachary C. Lipton, Danish Pruthi

**Abstract**: In recent years, NLP practitioners have converged on the following practice: (i) import an off-the-shelf pretrained (masked) language model; (ii) append a multilayer perceptron atop the CLS token's hidden representation (with randomly initialized weights); and (iii) fine-tune the entire model on a downstream task (MLP-FT). This procedure has produced massive gains on standard NLP benchmarks, but these models remain brittle, even to mild adversarial perturbations. In this work, we demonstrate surprising gains in adversarial robustness enjoyed by Model-tuning Via Prompts (MVP), an alternative method of adapting to downstream tasks. Rather than appending an MLP head to make output prediction, MVP appends a prompt template to the input, and makes prediction via text infilling/completion. Across 5 NLP datasets, 4 adversarial attacks, and 3 different models, MVP improves performance against adversarial substitutions by an average of 8% over standard methods and even outperforms adversarial training-based state-of-art defenses by 3.5%. By combining MVP with adversarial training, we achieve further improvements in adversarial robustness while maintaining performance on unperturbed examples. Finally, we conduct ablations to investigate the mechanism underlying these gains. Notably, we find that the main causes of vulnerability of MLP-FT can be attributed to the misalignment between pre-training and fine-tuning tasks, and the randomly initialized MLP parameters.

摘要: 近年来，NLP实践者在以下实践上趋同：(I)引入现成的预训练(掩蔽)语言模型；(Ii)在CLS令牌的隐藏表示上附加多层感知器(具有随机初始化权重)；以及(Iii)在下游任务(MLP-FT)上微调整个模型。这一过程在标准NLP基准上产生了巨大的收益，但这些模型仍然脆弱，即使是轻微的对抗性扰动。在这项工作中，我们展示了通过提示调整模型(MVP)在对抗健壮性方面的惊人收益，这是一种适应下游任务的替代方法。MVP不是附加MLP头来进行输出预测，而是将提示模板附加到输入，并通过文本填充/完成进行预测。在5个NLP数据集、4个对抗性攻击和3个不同的模型中，MVP在对抗对抗性替换时的性能比标准方法平均提高了8%，甚至比基于对抗性训练的最新防御性能提高了3.5%。通过将MVP与对抗性训练相结合，我们在保持在未受干扰的示例上的性能的同时，进一步提高了对抗性健壮性。最后，我们进行消融来研究这些收益背后的机制。值得注意的是，我们发现MLP-FT脆弱性的主要原因可以归因于预先训练和微调任务之间的不匹配，以及随机初始化的MLP参数。



## **34. Effective Backdoor Mitigation Depends on the Pre-training Objective**

有效的后门缓解取决于培训前的目标 cs.LG

Accepted for oral presentation at BUGS workshop @ NeurIPS 2023  (https://neurips2023-bugs.github.io/)

**SubmitDate**: 2023-12-05    [abs](http://arxiv.org/abs/2311.14948v3) [paper-pdf](http://arxiv.org/pdf/2311.14948v3)

**Authors**: Sahil Verma, Gantavya Bhatt, Avi Schwarzschild, Soumye Singhal, Arnav Mohanty Das, Chirag Shah, John P Dickerson, Jeff Bilmes

**Abstract**: Despite the advanced capabilities of contemporary machine learning (ML) models, they remain vulnerable to adversarial and backdoor attacks. This vulnerability is particularly concerning in real-world deployments, where compromised models may exhibit unpredictable behavior in critical scenarios. Such risks are heightened by the prevalent practice of collecting massive, internet-sourced datasets for pre-training multimodal models, as these datasets may harbor backdoors. Various techniques have been proposed to mitigate the effects of backdooring in these models such as CleanCLIP which is the current state-of-the-art approach. In this work, we demonstrate that the efficacy of CleanCLIP in mitigating backdoors is highly dependent on the particular objective used during model pre-training. We observe that stronger pre-training objectives correlate with harder to remove backdoors behaviors. We show this by training multimodal models on two large datasets consisting of 3 million (CC3M) and 6 million (CC6M) datapoints, under various pre-training objectives, followed by poison removal using CleanCLIP. We find that CleanCLIP is ineffective when stronger pre-training objectives are used, even with extensive hyperparameter tuning. Our findings underscore critical considerations for ML practitioners who pre-train models using large-scale web-curated data and are concerned about potential backdoor threats. Notably, our results suggest that simpler pre-training objectives are more amenable to effective backdoor removal. This insight is pivotal for practitioners seeking to balance the trade-offs between using stronger pre-training objectives and security against backdoor attacks.

摘要: 尽管当代机器学习(ML)模型具有先进的能力，但它们仍然容易受到对手和后门攻击。此漏洞在实际部署中尤其令人担忧，在实际部署中，受危害的模型可能会在关键情况下表现出不可预测的行为。为训练前的多模式模型收集来自互联网的海量数据集的普遍做法加剧了这种风险，因为这些数据集可能有后门。已经提出了各种技术来减轻这些模型中回溯的影响，例如CleanCLIP，这是当前最先进的方法。在这项工作中，我们证明了CleanCLIP在缓解后门方面的有效性高度依赖于在模型预培训期间使用的特定目标。我们观察到，较强的培训前目标与较难消除后门行为相关。我们通过在两个由300万(CC3M)和600万(CC6M)数据点组成的大型数据集上训练多模模型，在不同的预训练目标下，然后使用CleanCLIP去除毒物来证明这一点。我们发现，当使用更强的预培训目标时，即使进行了广泛的超参数调整，CleanCLIP也是无效的。我们的发现强调了ML从业者的关键考虑，他们使用大规模的网络管理数据对模型进行预培训，并担心潜在的后门威胁。值得注意的是，我们的结果表明，简单的预培训目标更容易有效地移除后门。对于寻求在使用更强的预培训目标和针对后门攻击的安全性之间进行权衡的从业者来说，这一见解至关重要。



## **35. Beyond Detection: Unveiling Fairness Vulnerabilities in Abusive Language Models**

超越检测：揭开辱骂语言模型中的公平漏洞 cs.CL

Under review

**SubmitDate**: 2023-12-05    [abs](http://arxiv.org/abs/2311.09428v2) [paper-pdf](http://arxiv.org/pdf/2311.09428v2)

**Authors**: Yueqing Liang, Lu Cheng, Ali Payani, Kai Shu

**Abstract**: This work investigates the potential of undermining both fairness and detection performance in abusive language detection. In a dynamic and complex digital world, it is crucial to investigate the vulnerabilities of these detection models to adversarial fairness attacks to improve their fairness robustness. We propose a simple yet effective framework FABLE that leverages backdoor attacks as they allow targeted control over the fairness and detection performance. FABLE explores three types of trigger designs (i.e., rare, artificial, and natural triggers) and novel sampling strategies. Specifically, the adversary can inject triggers into samples in the minority group with the favored outcome (i.e., "non-abusive") and flip their labels to the unfavored outcome, i.e., "abusive". Experiments on benchmark datasets demonstrate the effectiveness of FABLE attacking fairness and utility in abusive language detection.

摘要: 这项工作探讨了潜在的破坏公平性和检测性能在滥用语言检测。在动态复杂的数字世界中，研究这些检测模型对对抗性公平攻击的脆弱性以提高其公平鲁棒性至关重要。我们提出了一个简单而有效的框架FABLE，利用后门攻击，因为它们允许有针对性地控制公平性和检测性能。FABLE探索了三种类型的触发器设计（即，罕见的、人工的和天然的触发物）和新颖的采样策略。具体地，对手可以将触发器注入到具有有利结果的少数群体中的样本中（即，“非虐待”），并将他们的标签翻转到不受欢迎的结果，即，“虐待”在基准数据集上的实验证明了FABLE攻击公平性和实用性在滥用语言检测中的有效性。



## **36. ScAR: Scaling Adversarial Robustness for LiDAR Object Detection**

SCAR：激光雷达目标检测的对抗性缩放算法 cs.CV

**SubmitDate**: 2023-12-05    [abs](http://arxiv.org/abs/2312.03085v1) [paper-pdf](http://arxiv.org/pdf/2312.03085v1)

**Authors**: Xiaohu Lu, Hayder Radha

**Abstract**: The adversarial robustness of a model is its ability to resist adversarial attacks in the form of small perturbations to input data. Universal adversarial attack methods such as Fast Sign Gradient Method (FSGM) and Projected Gradient Descend (PGD) are popular for LiDAR object detection, but they are often deficient compared to task-specific adversarial attacks. Additionally, these universal methods typically require unrestricted access to the model's information, which is difficult to obtain in real-world applications. To address these limitations, we present a black-box Scaling Adversarial Robustness (ScAR) method for LiDAR object detection. By analyzing the statistical characteristics of 3D object detection datasets such as KITTI, Waymo, and nuScenes, we have found that the model's prediction is sensitive to scaling of 3D instances. We propose three black-box scaling adversarial attack methods based on the available information: model-aware attack, distribution-aware attack, and blind attack. We also introduce a strategy for generating scaling adversarial examples to improve the model's robustness against these three scaling adversarial attacks. Comparison with other methods on public datasets under different 3D object detection architectures demonstrates the effectiveness of our proposed method.

摘要: 模型的对抗性健壮性在于其抵抗以输入数据的小扰动形式的对抗性攻击的能力。快速符号梯度法(FSGM)和投影梯度下降法(PGD)等通用对抗攻击方法是激光雷达目标检测的常用方法，但与特定任务的对抗攻击相比往往存在不足。此外，这些通用方法通常需要不受限制地访问模型的信息，这在现实世界的应用程序中是很难获得的。为了解决这些局限性，我们提出了一种用于激光雷达目标检测的黑盒尺度对抗稳健性(SCAR)方法。通过分析Kitti、Waymo和nuScenes等3D目标检测数据集的统计特性，我们发现模型的预测对3D实例的缩放很敏感。我们根据已有的信息提出了三种黑盒尺度对抗性攻击方法：模型感知攻击、分布感知攻击和盲目攻击。我们还介绍了一种生成伸缩敌意实例的策略，以提高模型对这三种伸缩对手攻击的稳健性。在不同3D目标检测体系结构下的公共数据集上与其他方法进行了比较，验证了该方法的有效性。



## **37. Realistic Scatterer Based Adversarial Attacks on SAR Image Classifiers**

基于真实感散射体的SAR图像分类器对抗性攻击 cs.CV

**SubmitDate**: 2023-12-05    [abs](http://arxiv.org/abs/2312.02912v1) [paper-pdf](http://arxiv.org/pdf/2312.02912v1)

**Authors**: Tian Ye, Rajgopal Kannan, Viktor Prasanna, Carl Busart, Lance Kaplan

**Abstract**: Adversarial attacks have highlighted the vulnerability of classifiers based on machine learning for Synthetic Aperture Radar (SAR) Automatic Target Recognition (ATR) tasks. An adversarial attack perturbs SAR images of on-ground targets such that the classifiers are misled into making incorrect predictions. However, many existing attacking techniques rely on arbitrary manipulation of SAR images while overlooking the feasibility of executing the attacks on real-world SAR imagery. Instead, adversarial attacks should be able to be implemented by physical actions, for example, placing additional false objects as scatterers around the on-ground target to perturb the SAR image and fool the SAR ATR.   In this paper, we propose the On-Target Scatterer Attack (OTSA), a scatterer-based physical adversarial attack. To ensure the feasibility of its physical execution, we enforce a constraint on the positioning of the scatterers. Specifically, we restrict the scatterers to be placed only on the target instead of in the shadow regions or the background. To achieve this, we introduce a positioning score based on Gaussian kernels and formulate an optimization problem for our OTSA attack. Using a gradient ascent method to solve the optimization problem, the OTSA can generate a vector of parameters describing the positions, shapes, sizes and amplitudes of the scatterers to guide the physical execution of the attack that will mislead SAR image classifiers. The experimental results show that our attack obtains significantly higher success rates under the positioning constraint compared with the existing method.

摘要: 对抗性攻击突出了基于机器学习的分类器在合成孔径雷达(SAR)自动目标识别(ATR)任务中的脆弱性。对抗性攻击会干扰地面目标的合成孔径雷达图像，从而误导分类器做出错误的预测。然而，现有的许多攻击技术依赖于对SAR图像的任意篡改，而忽略了对现实世界的SAR图像执行攻击的可行性。相反，对抗性攻击应该能够通过物理行动来实施，例如，在地面目标周围放置额外的虚假目标作为散射体，以扰乱SAR图像并愚弄SAR ATR。本文提出了一种基于散射体的物理对抗攻击--目标上散射体攻击(OTSA)。为了确保其物理执行的可行性，我们对散射体的位置施加了约束。具体地说，我们将散射体限制为只放置在目标上，而不是放置在阴影区域或背景中。为了实现这一点，我们引入了一个基于高斯核的定位分数，并为我们的OTSA攻击建立了一个优化问题。利用梯度上升法求解优化问题，OTSA可以生成描述散射体位置、形状、大小和幅度的参数向量，以指导攻击的物理执行，从而误导SAR图像分类器。实验结果表明，与现有的攻击方法相比，在位置约束下，我们的攻击获得了更高的成功率。



## **38. Scaling Laws for Adversarial Attacks on Language Model Activations**

针对语言模型激活的对抗性攻击的标度律 cs.LG

15 pages, 9 figures

**SubmitDate**: 2023-12-05    [abs](http://arxiv.org/abs/2312.02780v1) [paper-pdf](http://arxiv.org/pdf/2312.02780v1)

**Authors**: Stanislav Fort

**Abstract**: We explore a class of adversarial attacks targeting the activations of language models. By manipulating a relatively small subset of model activations, $a$, we demonstrate the ability to control the exact prediction of a significant number (in some cases up to 1000) of subsequent tokens $t$. We empirically verify a scaling law where the maximum number of target tokens $t_\mathrm{max}$ predicted depends linearly on the number of tokens $a$ whose activations the attacker controls as $t_\mathrm{max} = \kappa a$. We find that the number of bits of control in the input space needed to control a single bit in the output space (what we call attack resistance $\chi$) is remarkably constant between $\approx 16$ and $\approx 25$ over 2 orders of magnitude of model sizes for different language models. Compared to attacks on tokens, attacks on activations are predictably much stronger, however, we identify a surprising regularity where one bit of input steered either via activations or via tokens is able to exert control over a similar amount of output bits. This gives support for the hypothesis that adversarial attacks are a consequence of dimensionality mismatch between the input and output spaces. A practical implication of the ease of attacking language model activations instead of tokens is for multi-modal and selected retrieval models, where additional data sources are added as activations directly, sidestepping the tokenized input. This opens up a new, broad attack surface. By using language models as a controllable test-bed to study adversarial attacks, we were able to experiment with input-output dimensions that are inaccessible in computer vision, especially where the output dimension dominates.

摘要: 我们探索了一类针对语言模型激活的对抗性攻击。通过操作相对较小的模型激活子集$a$，我们展示了控制大量(在某些情况下高达1000)后续令牌$t$的准确预测的能力。我们经验地验证了一个标度律，其中预测的目标令牌的最大数量$t_\mathm{max}$与攻击者控制的令牌的数量$a$线性相关，即$t_\mathm{max}=\kappa$。我们发现，对于不同的语言模型，在两个数量级的模型大小上，输入空间中控制输出空间中的单个比特所需的控制比特的数量(我们称之为攻击抵抗力$\chi$)在$\约16$到$\\约25$之间显著恒定。与对令牌的攻击相比，对激活的攻击可以预测得更强，然而，我们发现了一种令人惊讶的规律性，即通过激活或通过令牌引导的一位输入能够对类似数量的输出位施加控制。这支持了这样的假设，即对抗性攻击是输入和输出空间之间维度不匹配的结果。容易攻击语言模型激活而不是令牌的一个实际含义是对于多模式和选定的检索模型，在这些模型中，额外的数据源作为激活被直接添加，绕过了标记化的输入。这打开了一个新的、广泛的攻击面。通过使用语言模型作为研究对抗性攻击的可控试验台，我们能够试验计算机视觉中无法访问的输入-输出维度，特别是在输出维度占主导地位的情况下。



## **39. Generating Visually Realistic Adversarial Patch**

生成视觉逼真的对抗性补丁 cs.CV

14 pages

**SubmitDate**: 2023-12-05    [abs](http://arxiv.org/abs/2312.03030v1) [paper-pdf](http://arxiv.org/pdf/2312.03030v1)

**Authors**: Xiaosen Wang, Kunyu Wang

**Abstract**: Deep neural networks (DNNs) are vulnerable to various types of adversarial examples, bringing huge threats to security-critical applications. Among these, adversarial patches have drawn increasing attention due to their good applicability to fool DNNs in the physical world. However, existing works often generate patches with meaningless noise or patterns, making it conspicuous to humans. To address this issue, we explore how to generate visually realistic adversarial patches to fool DNNs. Firstly, we analyze that a high-quality adversarial patch should be realistic, position irrelevant, and printable to be deployed in the physical world. Based on this analysis, we propose an effective attack called VRAP, to generate visually realistic adversarial patches. Specifically, VRAP constrains the patch in the neighborhood of a real image to ensure the visual reality, optimizes the patch at the poorest position for position irrelevance, and adopts Total Variance loss as well as gamma transformation to make the generated patch printable without losing information. Empirical evaluations on the ImageNet dataset demonstrate that the proposed VRAP exhibits outstanding attack performance in the digital world. Moreover, the generated adversarial patches can be disguised as the scrawl or logo in the physical world to fool the deep models without being detected, bringing significant threats to DNNs-enabled applications.

摘要: 深度神经网络（DNN）容易受到各种类型的对抗性示例的攻击，给安全关键型应用带来巨大威胁。其中，对抗补丁由于其在物理世界中欺骗DNN的良好适用性而引起了越来越多的关注。然而，现有的作品往往会产生无意义的噪音或图案的补丁，使其对人类来说很显眼。为了解决这个问题，我们探索如何生成视觉上逼真的对抗补丁来欺骗DNN。首先，我们分析了一个高质量的对抗补丁应该是真实的，位置无关的，可打印的，以部署在物理世界。基于这种分析，我们提出了一种有效的攻击称为VRAP，生成视觉上逼真的对抗补丁。具体地说，VRAP算法将图像块约束在真实图像的邻域内以保证图像的视觉真实性，并在最差位置优化图像块以保证图像块的位置无关性，同时采用总方差损失和伽玛变换使生成的图像块可打印而不丢失信息。ImageNet数据集上的实证评估表明，所提出的VRAP在数字世界中表现出出色的攻击性能。此外，生成的对抗性补丁可以伪装成物理世界中的涂鸦或徽标，以欺骗深度模型而不被检测到，从而给启用DNN的应用程序带来重大威胁。



## **40. Byzantine-Robust Distributed Online Learning: Taming Adversarial Participants in An Adversarial Environment**

拜占庭-稳健的分布式在线学习：在对抗性环境中驯服对抗性参与者 cs.LG

**SubmitDate**: 2023-12-05    [abs](http://arxiv.org/abs/2307.07980v3) [paper-pdf](http://arxiv.org/pdf/2307.07980v3)

**Authors**: Xingrong Dong, Zhaoxian Wu, Qing Ling, Zhi Tian

**Abstract**: This paper studies distributed online learning under Byzantine attacks. The performance of an online learning algorithm is often characterized by (adversarial) regret, which evaluates the quality of one-step-ahead decision-making when an environment provides adversarial losses, and a sublinear bound is preferred. But we prove that, even with a class of state-of-the-art robust aggregation rules, in an adversarial environment and in the presence of Byzantine participants, distributed online gradient descent can only achieve a linear adversarial regret bound, which is tight. This is the inevitable consequence of Byzantine attacks, even though we can control the constant of the linear adversarial regret to a reasonable level. Interestingly, when the environment is not fully adversarial so that the losses of the honest participants are i.i.d. (independent and identically distributed), we show that sublinear stochastic regret, in contrast to the aforementioned adversarial regret, is possible. We develop a Byzantine-robust distributed online momentum algorithm to attain such a sublinear stochastic regret bound. Extensive numerical experiments corroborate our theoretical analysis.

摘要: 本文研究了拜占庭攻击下的分布式在线学习。在线学习算法的性能通常以（对抗性）后悔为特征，当环境提供对抗性损失时，它评估一步决策的质量，并且次线性界限是首选。但是我们证明了，即使有一类最先进的鲁棒聚集规则，在对抗环境中，在拜占庭参与者的存在下，分布式在线梯度下降只能实现线性对抗遗憾界，这是紧的。这是拜占庭攻击的必然结果，即使我们可以控制线性对抗性后悔的常数到一个合理的水平。有趣的是，当环境不是完全对抗时，诚实参与者的损失是独立同分布的。（独立同分布），我们表明，次线性随机遗憾，与上述对抗性遗憾，是可能的。我们开发了一个拜占庭鲁棒分布式在线动量算法，以达到这样的次线性随机遗憾界。大量的数值实验证实了我们的理论分析。



## **41. FedBayes: A Zero-Trust Federated Learning Aggregation to Defend Against Adversarial Attacks**

FedBayes：一种防御敌意攻击的零信任联合学习聚合 cs.CR

Accepted to IEEE CCWC 2024

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.04587v1) [paper-pdf](http://arxiv.org/pdf/2312.04587v1)

**Authors**: Marc Vucovich, Devin Quinn, Kevin Choi, Christopher Redino, Abdul Rahman, Edward Bowen

**Abstract**: Federated learning has created a decentralized method to train a machine learning model without needing direct access to client data. The main goal of a federated learning architecture is to protect the privacy of each client while still contributing to the training of the global model. However, the main advantage of privacy in federated learning is also the easiest aspect to exploit. Without being able to see the clients' data, it is difficult to determine the quality of the data. By utilizing data poisoning methods, such as backdoor or label-flipping attacks, or by sending manipulated information about their data back to the server, malicious clients are able to corrupt the global model and degrade performance across all clients within a federation. Our novel aggregation method, FedBayes, mitigates the effect of a malicious client by calculating the probabilities of a client's model weights given to the prior model's weights using Bayesian statistics. Our results show that this approach negates the effects of malicious clients and protects the overall federation.

摘要: 联合学习创造了一种分散的方法来训练机器学习模型，而不需要直接访问客户数据。联合学习架构的主要目标是保护每个客户的隐私，同时仍然为全球模型的培训做出贡献。然而，联合学习中隐私的主要优势也是最容易利用的方面。如果不能看到客户的数据，就很难确定数据的质量。通过利用数据中毒方法，如后门或标签翻转攻击，或通过将有关其数据的被操纵的信息发送回服务器，恶意客户端能够破坏全局模型，并降低联盟内所有客户端的性能。我们的新聚合方法，FedBayes，通过使用贝叶斯统计计算客户端的模型权重赋予先前模型的权重的概率来减轻恶意客户端的影响。我们的结果表明，该方法消除了恶意客户端的影响，并保护了整个联盟。



## **42. InstructTA: Instruction-Tuned Targeted Attack for Large Vision-Language Models**

InstructTA：针对大型视觉语言模型的指令调整定向攻击 cs.CV

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.01886v1) [paper-pdf](http://arxiv.org/pdf/2312.01886v1)

**Authors**: Xunguang Wang, Zhenlan Ji, Pingchuan Ma, Zongjie Li, Shuai Wang

**Abstract**: Large vision-language models (LVLMs) have demonstrated their incredible capability in image understanding and response generation. However, this rich visual interaction also makes LVLMs vulnerable to adversarial examples. In this paper, we formulate a novel and practical gray-box attack scenario that the adversary can only access the visual encoder of the victim LVLM, without the knowledge of its prompts (which are often proprietary for service providers and not publicly available) and its underlying large language model (LLM). This practical setting poses challenges to the cross-prompt and cross-model transferability of targeted adversarial attack, which aims to confuse the LVLM to output a response that is semantically similar to the attacker's chosen target text. To this end, we propose an instruction-tuned targeted attack (dubbed InstructTA) to deliver the targeted adversarial attack on LVLMs with high transferability. Initially, we utilize a public text-to-image generative model to "reverse" the target response into a target image, and employ GPT-4 to infer a reasonable instruction $\boldsymbol{p}^\prime$ from the target response. We then form a local surrogate model (sharing the same visual encoder with the victim LVLM) to extract instruction-aware features of an adversarial image example and the target image, and minimize the distance between these two features to optimize the adversarial example. To further improve the transferability, we augment the instruction $\boldsymbol{p}^\prime$ with instructions paraphrased from an LLM. Extensive experiments demonstrate the superiority of our proposed method in targeted attack performance and transferability.

摘要: 大型视觉语言模型（LVLM）已经证明了它们在图像理解和响应生成方面令人难以置信的能力。然而，这种丰富的视觉交互也使LVLM容易受到对抗性示例的攻击。在本文中，我们制定了一个新的和实用的灰盒攻击的情况下，对手只能访问受害者LVLM的视觉编码器，而不知道其提示（这往往是专有的服务提供商和不公开）和其底层的大语言模型（LLM）。这种实际设置对针对性对抗攻击的跨提示和跨模型可转移性提出了挑战，其目的是混淆LVLM以输出与攻击者选择的目标文本在语义上相似的响应。为此，我们提出了一种防御调整的有针对性的攻击（称为指令TA），以提供具有高可转移性的LVLM上的有针对性的对抗攻击。首先，我们利用一个公共的文本到图像生成模型来“反转”目标响应到目标图像，并采用GPT-4从目标响应中推断出合理的指令$\boldsymbol{p}^\prime$。然后，我们形成一个本地代理模型（与受害者LVLM共享相同的视觉编码器）来提取对抗图像示例和目标图像的防御感知特征，并最小化这两个特征之间的距离以优化对抗示例。为了进一步提高可移植性，我们增加了指令$\boldsymbol{p}^\prime$与从LLM解释的指令。大量的实验表明，我们提出的方法在有针对性的攻击性能和可移植性的优越性。



## **43. Two-stage optimized unified adversarial patch for attacking visible-infrared cross-modal detectors in the physical world**

用于攻击物理世界中可见光-红外交叉模式探测器的两阶段优化统一对抗补丁 cs.CV

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.01789v1) [paper-pdf](http://arxiv.org/pdf/2312.01789v1)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstract**: Currently, many studies have addressed security concerns related to visible and infrared detectors independently. In practical scenarios, utilizing cross-modal detectors for tasks proves more reliable than relying on single-modal detectors. Despite this, there is a lack of comprehensive security evaluations for cross-modal detectors. While existing research has explored the feasibility of attacks against cross-modal detectors, the implementation of a robust attack remains unaddressed. This work introduces the Two-stage Optimized Unified Adversarial Patch (TOUAP) designed for performing attacks against visible-infrared cross-modal detectors in real-world, black-box settings. The TOUAP employs a two-stage optimization process: firstly, PSO optimizes an irregular polygonal infrared patch to attack the infrared detector; secondly, the color QR code is optimized, and the shape information of the infrared patch from the first stage is used as a mask. The resulting irregular polygon visible modal patch executes an attack on the visible detector. Through extensive experiments conducted in both digital and physical environments, we validate the effectiveness and robustness of the proposed method. As the TOUAP surpasses baseline performance, we advocate for its widespread attention.

摘要: 目前，许多研究已经独立地解决了与可见光和红外探测器相关的安全问题。在实际场景中，使用跨模式检测器执行任务被证明比依赖单模式检测器更可靠。尽管如此，目前还缺乏对跨模式探测器的全面安全评估。虽然现有的研究已经探索了针对跨模式检测器的攻击的可行性，但健壮攻击的实现仍然没有得到解决。这项工作介绍了两阶段优化的统一对抗补丁(TOUAP)，设计用于在现实世界的黑匣子环境中执行对可见光-红外交叉模式探测器的攻击。TOUAP算法采用两阶段优化过程：首先，粒子群算法对攻击红外探测器的不规则多边形红外贴片进行优化；其次，对颜色二维码进行优化，并将第一阶段得到的红外贴片的形状信息作为掩码。所得到的不规则多边形可见模式面片对可见检测器执行攻击。通过在数字和物理环境中进行的大量实验，验证了该方法的有效性和稳健性。由于TOUAP超过了基线表现，我们主张广泛关注它。



## **44. Singular Regularization with Information Bottleneck Improves Model's Adversarial Robustness**

带信息瓶颈的奇异正则化提高模型的对抗稳健性 cs.CV

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.02237v1) [paper-pdf](http://arxiv.org/pdf/2312.02237v1)

**Authors**: Guanlin Li, Naishan Zheng, Man Zhou, Jie Zhang, Tianwei Zhang

**Abstract**: Adversarial examples are one of the most severe threats to deep learning models. Numerous works have been proposed to study and defend adversarial examples. However, these works lack analysis of adversarial information or perturbation, which cannot reveal the mystery of adversarial examples and lose proper interpretation. In this paper, we aim to fill this gap by studying adversarial information as unstructured noise, which does not have a clear pattern. Specifically, we provide some empirical studies with singular value decomposition, by decomposing images into several matrices, to analyze adversarial information for different attacks. Based on the analysis, we propose a new module to regularize adversarial information and combine information bottleneck theory, which is proposed to theoretically restrict intermediate representations. Therefore, our method is interpretable. Moreover, the fashion of our design is a novel principle that is general and unified. Equipped with our new module, we evaluate two popular model structures on two mainstream datasets with various adversarial attacks. The results indicate that the improvement in robust accuracy is significant. On the other hand, we prove that our method is efficient with only a few additional parameters and able to be explained under regional faithfulness analysis.

摘要: 对抗性例子是深度学习模式面临的最严重威胁之一。已经提出了大量的工作来研究和辩护对抗性例子。然而，这些作品缺乏对对抗性信息或扰动的分析，不能揭示对抗性例子的奥秘，从而失去了适当的解释。在本文中，我们旨在通过研究非结构化噪声来填补这一空白，这种非结构化噪声没有明确的模式。具体地说，我们提供了一些奇异值分解的实证研究，通过将图像分解成几个矩阵，来分析不同攻击的对抗性信息。在此基础上，我们提出了一种新的对抗性信息正规化模型，并结合信息瓶颈理论，从理论上对中间表征进行了约束。因此，我们的方法是可解释的。此外，我们的设计时尚是一种新的原则，是通用的和统一的。使用我们的新模块，我们在两个主流数据集上对两个流行的模型结构进行了评估，并进行了各种对抗性攻击。结果表明，该方法在稳健性精度方面有显著的提高。另一方面，我们证明了我们的方法是有效的，只需要很少的附加参数，并且能够在区域忠诚度分析下得到解释。



## **45. Warfare:Breaking the Watermark Protection of AI-Generated Content**

战争：打破人工智能生成内容的水印保护 cs.CV

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2310.07726v2) [paper-pdf](http://arxiv.org/pdf/2310.07726v2)

**Authors**: Guanlin Li, Yifei Chen, Jie Zhang, Jiwei Li, Shangwei Guo, Tianwei Zhang

**Abstract**: AI-Generated Content (AIGC) is gaining great popularity, with many emerging commercial services and applications. These services leverage advanced generative models, such as latent diffusion models and large language models, to generate creative content (e.g., realistic images and fluent sentences) for users. The usage of such generated content needs to be highly regulated, as the service providers need to ensure the users do not violate the usage policies (e.g., abuse for commercialization, generating and distributing unsafe content). A promising solution to achieve this goal is watermarking, which adds unique and imperceptible watermarks on the content for service verification and attribution. Numerous watermarking approaches have been proposed recently. However, in this paper, we show that an adversary can easily break these watermarking mechanisms. Specifically, we consider two possible attacks. (1) Watermark removal: the adversary can easily erase the embedded watermark from the generated content and then use it freely bypassing the regulation of the service provider. (2) Watermark forging: the adversary can create illegal content with forged watermarks from another user, causing the service provider to make wrong attributions. We propose Warfare, a unified methodology to achieve both attacks in a holistic way. The key idea is to leverage a pre-trained diffusion model for content processing and a generative adversarial network for watermark removal or forging. We evaluate Warfare on different datasets and embedding setups. The results prove that it can achieve high success rates while maintaining the quality of the generated content. Compared to existing diffusion model-based attacks, Warfare is 5,050~11,000x faster.

摘要: 人工智能生成的内容（AIGC）越来越受欢迎，有许多新兴的商业服务和应用程序。这些服务利用诸如潜在扩散模型和大型语言模型之类的高级生成模型来生成创意内容（例如，逼真的图像和流畅的句子）。这种生成的内容的使用需要高度管制，因为服务提供商需要确保用户不违反使用策略（例如，滥用以商业化、生成和分发不安全内容）。实现这一目标的一个有前途的解决方案是水印，它在内容上添加唯一的和不可感知的水印，用于服务验证和属性。近来已经提出了许多水印方法。然而，在本文中，我们表明，对手可以很容易地打破这些水印机制。具体来说，我们考虑两种可能的攻击。(1)水印去除：对手可以容易地从所生成的内容中擦除嵌入的水印，然后绕过服务提供商的规定自由地使用它。(2)水印锻造：对手可以利用来自另一用户的伪造水印来创建非法内容，从而导致服务提供商做出错误的归属。我们提出了战争，一个统一的方法来实现这两种攻击的整体方式。其关键思想是利用预训练的扩散模型进行内容处理，并利用生成对抗网络进行水印删除或伪造。我们在不同的数据集和嵌入设置上评估战争。结果证明，它可以实现高成功率，同时保持生成内容的质量。与现有的基于扩散模型的攻击相比，Warfare的速度快了5,050~ 11,000倍。



## **46. Malicious Lateral Movement in 5G Core With Network Slicing And Its Detection**

基于网络分片的5G核心恶意侧移及其检测 cs.CR

Accepted for publication in the Proceedings of IEEE ITNAC-2023

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.01681v1) [paper-pdf](http://arxiv.org/pdf/2312.01681v1)

**Authors**: Ayush Kumar, Vrizlynn L. L. Thing

**Abstract**: 5G networks are susceptible to cyber attacks due to reasons such as implementation issues and vulnerabilities in 3GPP standard specifications. In this work, we propose lateral movement strategies in a 5G Core (5GC) with network slicing enabled, as part of a larger attack campaign by well-resourced adversaries such as APT groups. Further, we present 5GLatte, a system to detect such malicious lateral movement. 5GLatte operates on a host-container access graph built using host/NF container logs collected from the 5GC. Paths inferred from the access graph are scored based on selected filtering criteria and subsequently presented as input to a threshold-based anomaly detection algorithm to reveal malicious lateral movement paths. We evaluate 5GLatte on a dataset containing attack campaigns (based on MITRE ATT&CK and FiGHT frameworks) launched in a 5G test environment which shows that compared to other lateral movement detectors based on state-of-the-art, it can achieve higher true positive rates with similar false positive rates.

摘要: 由于3GPP标准规范中的实现问题和漏洞等原因，5G网络容易受到网络攻击。在这项工作中，我们提出了启用网络切片的5G核心(5GC)中的横向移动策略，作为资源丰富的对手(如APT组)更大攻击活动的一部分。此外，我们还提出了5G Latte，这是一个检测此类恶意横向移动的系统。5GLatte在使用从5GC收集的主机/NF容器日志构建的主机-容器访问图上运行。根据选择的过滤标准对从访问图推断的路径进行评分，并随后将其作为基于阈值的异常检测算法的输入，以揭示恶意的横向移动路径。我们在包含5G测试环境中发起的攻击活动(基于MITRE ATT&CK和Fight框架)的数据集上对5G Latte进行了评估，结果表明，与其他基于最新技术的侧向运动检测器相比，它可以在相似的误警率下获得更高的真阳性率。



## **47. Adversarial Medical Image with Hierarchical Feature Hiding**

基于分层特征隐藏的对抗性医学图像 eess.IV

Our code is available at  \url{https://github.com/qsyao/Hierarchical_Feature_Constraint}. arXiv admin  note: text overlap with arXiv:2012.09501

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.01679v1) [paper-pdf](http://arxiv.org/pdf/2312.01679v1)

**Authors**: Qingsong Yao, Zecheng He, Yuexiang Li, Yi Lin, Kai Ma, Yefeng Zheng, S. Kevin Zhou

**Abstract**: Deep learning based methods for medical images can be easily compromised by adversarial examples (AEs), posing a great security flaw in clinical decision-making. It has been discovered that conventional adversarial attacks like PGD which optimize the classification logits, are easy to distinguish in the feature space, resulting in accurate reactive defenses. To better understand this phenomenon and reassess the reliability of the reactive defenses for medical AEs, we thoroughly investigate the characteristic of conventional medical AEs. Specifically, we first theoretically prove that conventional adversarial attacks change the outputs by continuously optimizing vulnerable features in a fixed direction, thereby leading to outlier representations in the feature space. Then, a stress test is conducted to reveal the vulnerability of medical images, by comparing with natural images. Interestingly, this vulnerability is a double-edged sword, which can be exploited to hide AEs. We then propose a simple-yet-effective hierarchical feature constraint (HFC), a novel add-on to conventional white-box attacks, which assists to hide the adversarial feature in the target feature distribution. The proposed method is evaluated on three medical datasets, both 2D and 3D, with different modalities. The experimental results demonstrate the superiority of HFC, \emph{i.e.,} it bypasses an array of state-of-the-art adversarial medical AE detectors more efficiently than competing adaptive attacks, which reveals the deficiencies of medical reactive defense and allows to develop more robust defenses in future.

摘要: 基于深度学习的医学图像处理方法容易受到对抗性实例的攻击，在临床决策中存在很大的安全缺陷。已经发现，像PGD这样的传统对抗性攻击优化了分类逻辑，在特征空间中很容易区分，从而产生准确的反应性防御。为了更好地理解这一现象，并重新评估医用AEs反应性防御的可靠性，我们深入研究了传统医用AEs的特点。具体地说，我们首先从理论上证明了传统的对抗性攻击通过在固定方向上不断优化易受攻击的特征来改变输出，从而导致特征空间中的孤立点表示。然后，通过与自然图像的比较，进行了压力测试，揭示了医学图像的脆弱性。有趣的是，这个漏洞是一把双刃剑，可以被利用来隐藏AE。然后，我们提出了一种简单有效的层次特征约束(HFC)，这是对传统白盒攻击的一种新的补充，它帮助隐藏目标特征分布中的对抗性特征。该方法在三个医学数据集上进行了评估，包括2D和3D，使用不同的模式。实验结果证明了HFC的优越性，即它比竞争的自适应攻击更有效地绕过了一系列最先进的对抗性医学AE检测器，这揭示了医学反应性防御的不足，并为未来开发更健壮的防御奠定了基础。



## **48. The Queen's Guard: A Secure Enforcement of Fine-grained Access Control In Distributed Data Analytics Platforms**

女王卫队：分布式数据分析平台中细粒度访问控制的安全执行 cs.CR

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2106.13123v4) [paper-pdf](http://arxiv.org/pdf/2106.13123v4)

**Authors**: Fahad Shaon, Sazzadur Rahaman, Murat Kantarcioglu

**Abstract**: Distributed data analytics platforms (i.e., Apache Spark, Hadoop) provide high-level APIs to programmatically write analytics tasks that are run distributedly in multiple computing nodes. The design of these frameworks was primarily motivated by performance and usability. Thus, the security takes a back seat. Consequently, they do not inherently support fine-grained access control or offer any plugin mechanism to enable it, making them risky to be used in multi-tier organizational settings.   There have been attempts to build "add-on" solutions to enable fine-grained access control for distributed data analytics platforms. In this paper, first, we show that straightforward enforcement of ``add-on'' access control is insecure under adversarial code execution. Specifically, we show that an attacker can abuse platform-provided APIs to evade access controls without leaving any traces. Second, we designed a two-layered (i.e., proactive and reactive) defense system to protect against API abuses. On submission of a user code, our proactive security layer statically screens it to find potential attack signatures prior to its execution. The reactive security layer employs code instrumentation-based runtime checks and sandboxed execution to throttle any exploits at runtime. Next, we propose a new fine-grained access control framework with an enhanced policy language that supports map and filter primitives. Finally, we build a system named SecureDL with our new access control framework and defense system on top of Apache Spark, which ensures secure access control policy enforcement under adversaries capable of executing code.   To the best of our knowledge, this is the first fine-grained attribute-based access control framework for distributed data analytics platforms that is secure against platform API abuse attacks. Performance evaluation showed that the overhead due to added security is low.

摘要: 分布式数据分析平台(即，ApacheSpark、Hadoop)提供高级API，以编程方式编写在多个计算节点上分布式运行的分析任务。这些框架的设计主要是出于性能和可用性的考虑。因此，安全措施就退居次要地位了。因此，它们本身并不支持细粒度的访问控制，也不提供任何插件机制来启用它，这使得它们在多层组织设置中使用存在风险。已经有人尝试构建“附加”解决方案来实现分布式数据分析平台的细粒度访问控制。在这篇文章中，我们首先证明了直接实施“附加”访问控制在恶意代码执行下是不安全的。具体地说，我们展示了攻击者可以滥用平台提供的API来逃避访问控制，而不会留下任何痕迹。其次，我们设计了一个双层(即主动和被动)防御体系，以防止API滥用。在提交用户代码时，我们的主动安全层会在代码执行之前对其进行静态筛选，以发现潜在的攻击特征。反应式安全层采用基于代码检测的运行时检查和沙箱执行，以在运行时遏制任何利用漏洞。接下来，我们提出了一种新的细粒度访问控制框架，该框架具有增强的策略语言，支持映射和过滤原语。最后，我们使用新的访问控制框架和防御系统在ApacheSpark之上构建了一个名为SecureDL的系统，确保了在能够执行代码的攻击者的情况下安全地执行访问控制策略。据我们所知，这是第一个针对分布式数据分析平台的细粒度基于属性的访问控制框架，可以安全地抵御平台API滥用攻击。性能评估表明，由于增加安全性而产生的开销很低。



## **49. Robust Evaluation of Diffusion-Based Adversarial Purification**

基于扩散的对抗净化算法的稳健性评价 cs.CV

Accepted by ICCV 2023, oral presentation. Code is available at  https://github.com/ml-postech/robust-evaluation-of-diffusion-based-purification

**SubmitDate**: 2023-12-03    [abs](http://arxiv.org/abs/2303.09051v3) [paper-pdf](http://arxiv.org/pdf/2303.09051v3)

**Authors**: Minjong Lee, Dongwoo Kim

**Abstract**: We question the current evaluation practice on diffusion-based purification methods. Diffusion-based purification methods aim to remove adversarial effects from an input data point at test time. The approach gains increasing attention as an alternative to adversarial training due to the disentangling between training and testing. Well-known white-box attacks are often employed to measure the robustness of the purification. However, it is unknown whether these attacks are the most effective for the diffusion-based purification since the attacks are often tailored for adversarial training. We analyze the current practices and provide a new guideline for measuring the robustness of purification methods against adversarial attacks. Based on our analysis, we further propose a new purification strategy improving robustness compared to the current diffusion-based purification methods.

摘要: 我们对目前基于扩散的纯化方法的评估实践提出了质疑。基于扩散的净化方法旨在消除测试时输入数据点的对抗性影响。由于训练和测试之间的分离，这种方法作为对抗性训练的替代方法受到越来越多的关注。通常使用众所周知的白盒攻击来衡量净化的健壮性。然而，目前尚不清楚这些攻击对于基于扩散的净化是否最有效，因为这些攻击通常是为对抗性训练量身定做的。我们分析了目前的实践，并为衡量净化方法对对手攻击的健壮性提供了新的指导方针。在分析的基础上，进一步提出了一种新的纯化策略，与现有的基于扩散的纯化方法相比，提高了算法的稳健性。



## **50. QuantAttack: Exploiting Dynamic Quantization to Attack Vision Transformers**

QuantAttack：利用动态量化攻击视觉变形金刚 cs.CV

**SubmitDate**: 2023-12-03    [abs](http://arxiv.org/abs/2312.02220v1) [paper-pdf](http://arxiv.org/pdf/2312.02220v1)

**Authors**: Amit Baras, Alon Zolfi, Yuval Elovici, Asaf Shabtai

**Abstract**: In recent years, there has been a significant trend in deep neural networks (DNNs), particularly transformer-based models, of developing ever-larger and more capable models. While they demonstrate state-of-the-art performance, their growing scale requires increased computational resources (e.g., GPUs with greater memory capacity). To address this problem, quantization techniques (i.e., low-bit-precision representation and matrix multiplication) have been proposed. Most quantization techniques employ a static strategy in which the model parameters are quantized, either during training or inference, without considering the test-time sample. In contrast, dynamic quantization techniques, which have become increasingly popular, adapt during inference based on the input provided, while maintaining full-precision performance. However, their dynamic behavior and average-case performance assumption makes them vulnerable to a novel threat vector -- adversarial attacks that target the model's efficiency and availability. In this paper, we present QuantAttack, a novel attack that targets the availability of quantized models, slowing down the inference, and increasing memory usage and energy consumption. We show that carefully crafted adversarial examples, which are designed to exhaust the resources of the operating system, can trigger worst-case performance. In our experiments, we demonstrate the effectiveness of our attack on vision transformers on a wide range of tasks, both uni-modal and multi-modal. We also examine the effect of different attack variants (e.g., a universal perturbation) and the transferability between different models.

摘要: 近年来，深度神经网络(DNN)，特别是基于变压器的模型，有一个显著的趋势，即开发更大、更有能力的模型。虽然它们展示了一流的性能，但其不断增长的规模需要增加计算资源(例如，具有更大内存容量的GPU)。为了解决这个问题，人们提出了量化技术(即低位精度表示和矩阵乘法)。大多数量化技术采用静态策略，其中模型参数在训练或推理期间被量化，而不考虑测试时间样本。相比之下，已经变得越来越流行的动态量化技术在基于所提供的输入进行推理期间进行调整，同时保持全精度性能。然而，它们的动态行为和平均情况性能假设使它们容易受到一种新的威胁矢量--以模型的效率和可用性为目标的对抗性攻击。在本文中，我们提出了一种新的攻击QuantAttack，它的目标是量化模型的可用性，减慢推理速度，增加内存使用和能量消耗。我们展示了精心设计的敌意例子，这些例子旨在耗尽操作系统的资源，可以触发最坏的情况下的性能。在我们的实验中，我们展示了我们对视觉转换器的攻击在各种任务中的有效性，包括单模式和多模式。我们还考察了不同攻击变量(例如，通用扰动)的影响以及不同模型之间的可转移性。



