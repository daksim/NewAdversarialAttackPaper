# Latest Adversarial Attack Papers
**update at 2022-09-30 06:31:28**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. A Survey on Physical Adversarial Attack in Computer Vision**

计算机视觉中的身体对抗攻击研究综述 cs.CV

**SubmitDate**: 2022-09-28    [paper-pdf](http://arxiv.org/pdf/2209.14262v1)

**Authors**: Donghua Wang, Wen Yao, Tingsong Jiang, Guijiang Tang, Xiaoqian Chen

**Abstracts**: In the past decade, deep learning has dramatically changed the traditional hand-craft feature manner with strong feature learning capability, resulting in tremendous improvement of conventional tasks. However, deep neural networks have recently been demonstrated vulnerable to adversarial examples, a kind of malicious samples crafted by small elaborately designed noise, which mislead the DNNs to make the wrong decisions while remaining imperceptible to humans. Adversarial examples can be divided into digital adversarial attacks and physical adversarial attacks. The digital adversarial attack is mostly performed in lab environments, focusing on improving the performance of adversarial attack algorithms. In contrast, the physical adversarial attack focus on attacking the physical world deployed DNN systems, which is a more challenging task due to the complex physical environment (i.e., brightness, occlusion, and so on). Although the discrepancy between digital adversarial and physical adversarial examples is small, the physical adversarial examples have a specific design to overcome the effect of the complex physical environment. In this paper, we review the development of physical adversarial attacks in DNN-based computer vision tasks, including image recognition tasks, object detection tasks, and semantic segmentation. For the sake of completeness of the algorithm evolution, we will briefly introduce the works that do not involve the physical adversarial attack. We first present a categorization scheme to summarize the current physical adversarial attacks. Then discuss the advantages and disadvantages of the existing physical adversarial attacks and focus on the technique used to maintain the adversarial when applied into physical environment. Finally, we point out the issues of the current physical adversarial attacks to be solved and provide promising research directions.

摘要: 在过去的十年里，深度学习以其强大的特征学习能力，极大地改变了传统的手工特征学习方式，使常规任务得到了极大的改善。然而，深度神经网络最近被证明容易受到敌意例子的攻击，这是一种由精心设计的小噪声制作的恶意样本，它误导DNN做出错误的决定，同时保持对人类的不可察觉。对抗性攻击可分为数字对抗性攻击和物理对抗性攻击。数字对抗攻击大多在实验室环境中进行，致力于提高对抗攻击算法的性能。相比之下，物理对抗性攻击侧重于攻击物理世界中部署的DNN系统，由于物理环境复杂(即亮度、遮挡等)，这是一项更具挑战性的任务。虽然数字对抗例子和物理对抗例子之间的差异很小，但物理对抗例子有一个特定的设计来克服复杂物理环境的影响。本文回顾了基于DNN的计算机视觉任务中物理对抗攻击的发展，包括图像识别任务、目标检测任务和语义分割任务。为了算法演化的完备性，我们将简要介绍不涉及物理对抗攻击的工作。我们首先提出了一种分类方案来总结当前的物理对抗性攻击。然后讨论了现有物理对抗性攻击的优缺点，并重点介绍了应用于物理环境中维护对抗性的技术。最后，指出了当前物理对抗性攻击需要解决的问题，并提出了有前景的研究方向。



## **2. A Closer Look at Evaluating the Bit-Flip Attack Against Deep Neural Networks**

深入研究对深度神经网络的比特翻转攻击 cs.CR

Extended version from IEEE IOLTS'2022 short paper

**SubmitDate**: 2022-09-28    [paper-pdf](http://arxiv.org/pdf/2209.14243v1)

**Authors**: Kevin Hector, Mathieu Dumont, Pierre-Alain Moellic, Jean-Max Dutertre

**Abstracts**: Deep neural network models are massively deployed on a wide variety of hardware platforms. This results in the appearance of new attack vectors that significantly extend the standard attack surface, extensively studied by the adversarial machine learning community. One of the first attack that aims at drastically dropping the performance of a model, by targeting its parameters (weights) stored in memory, is the Bit-Flip Attack (BFA). In this work, we point out several evaluation challenges related to the BFA. First of all, the lack of an adversary's budget in the standard threat model is problematic, especially when dealing with physical attacks. Moreover, since the BFA presents critical variability, we discuss the influence of some training parameters and the importance of the model architecture. This work is the first to present the impact of the BFA against fully-connected architectures that present different behaviors compared to convolutional neural networks. These results highlight the importance of defining robust and sound evaluation methodologies to properly evaluate the dangers of parameter-based attacks as well as measure the real level of robustness offered by a defense.

摘要: 深度神经网络模型被大量部署在各种硬件平台上。这导致了新的攻击矢量的出现，这些攻击矢量显著扩展了标准攻击面，这是对抗性机器学习社区广泛研究的结果。最早的攻击之一是位翻转攻击(BFA)，该攻击旨在通过攻击存储在内存中的参数(权重)来大幅降低模型的性能。在这项工作中，我们指出了与博鳌亚洲论坛相关的几个评估挑战。首先，在标准威胁模型中缺乏对手的预算是有问题的，特别是在处理物理攻击时。此外，由于BFA呈现关键的可变性，我们讨论了一些训练参数的影响和模型体系结构的重要性。这项工作首次展示了BFA对全连接体系结构的影响，与卷积神经网络相比，全连接体系结构呈现出不同的行为。这些结果突显了定义稳健和合理的评估方法的重要性，以适当地评估基于参数的攻击的危险，以及衡量防御提供的真实健壮性水平。



## **3. Exploring the Relationship between Architecture and Adversarially Robust Generalization**

探索体系结构和相反的健壮性泛化之间的关系 cs.LG

**SubmitDate**: 2022-09-28    [paper-pdf](http://arxiv.org/pdf/2209.14105v1)

**Authors**: Shiyu Tang, Siyuan Liang, Ruihao Gong, Aishan Liu, Xianglong Liu, Dacheng Tao

**Abstracts**: Adversarial training has been demonstrated to be one of the most effective remedies for defending adversarial examples, yet it often suffers from the huge robustness generalization gap on unseen testing adversaries, deemed as the \emph{adversarially robust generalization problem}. Despite the preliminary understandings devoted on adversarially robust generalization, little is known from the architectural perspective. Thus, this paper tries to bridge the gap by systematically examining the most representative architectures (e.g., Vision Transformers and CNNs). In particular, we first comprehensively evaluated \emph{20} adversarially trained architectures on ImageNette and CIFAR-10 datasets towards several adversaries (multiple $\ell_p$-norm adversarial attacks), and found that Vision Transformers (e.g., PVT, CoAtNet) often yield better adversarially robust generalization. To further understand what architectural ingredients favor adversarially robust generalization, we delve into several key building blocks and revealed the fact via the lens of Rademacher complexity that the higher weight sparsity contributes significantly towards the better adversarially robust generalization of Vision Transformers, which can be often achieved by attention layers. Our extensive studies discovered the close relationship between architectural design and adversarially robust generalization, and instantiated several important insights. We hope our findings could help to better understand the mechanism towards designing robust deep learning architectures.

摘要: 对抗性训练已被证明是为对抗性例子辩护的最有效的补救方法之一，然而它经常在看不见的测试对手上遭受巨大的健壮性泛化差距，被认为是对抗性健壮性泛化问题。尽管对相反的健壮性泛化有了初步的理解，但从体系结构的角度来看却知之甚少。因此，本文试图通过系统地研究最具代表性的体系结构(例如，Vision Transformers和CNN)来弥合这一差距。特别是，我们首先在ImageNette和CIFAR-10数据集上综合评估了{20}个经过攻击训练的架构对几个对手(多次$\ell_p$-范数攻击)进行了评估，发现Vision Transformers(例如PVT、CoAtNet)通常能够产生更好的攻击健壮性泛化。为了进一步了解哪些架构成分有利于逆境健壮的概括，我们深入研究了几个关键的构建块，并通过Rademacher复杂性的镜头揭示了这样一个事实，即较高的权重稀疏性对更好的逆境健壮的视觉转换器概括有很大贡献，这通常可以通过关注层来实现。我们广泛的研究发现了建筑设计和相反的健壮概括之间的密切关系，并例证了几个重要的见解。我们希望我们的发现能够帮助我们更好地理解设计健壮的深度学习架构的机制。



## **4. Machine Beats Machine: Machine Learning Models to Defend Against Adversarial Attacks**

机器击败机器：防御对手攻击的机器学习模型 cs.LG

**SubmitDate**: 2022-09-28    [paper-pdf](http://arxiv.org/pdf/2209.13963v1)

**Authors**: Jože M. Rožanec, Dimitrios Papamartzivanos, Entso Veliou, Theodora Anastasiou, Jelle Keizer, Blaž Fortuna, Dunja Mladenić

**Abstracts**: We propose using a two-layered deployment of machine learning models to prevent adversarial attacks. The first layer determines whether the data was tampered, while the second layer solves a domain-specific problem. We explore three sets of features and three dataset variations to train machine learning models. Our results show clustering algorithms achieved promising results. In particular, we consider the best results were obtained by applying the DBSCAN algorithm to the structured structural similarity index measure computed between the images and a white reference image.

摘要: 我们建议使用机器学习模型的两层部署来防止对抗性攻击。第一层确定数据是否被篡改，而第二层解决特定于域的问题。我们探索了三组特征和三种数据集变体来训练机器学习模型。实验结果表明，聚类算法取得了良好的效果。特别地，我们认为将DBSCAN算法应用于计算图像与白色参考图像之间的结构化结构相似性指数获得了最好的结果。



## **5. Adaptive Image Transformations for Transfer-based Adversarial Attack**

基于传输的对抗性攻击中的自适应图像变换 cs.CV

34 pages, 7 figures, 11 tables. Accepted by ECCV2022

**SubmitDate**: 2022-09-28    [paper-pdf](http://arxiv.org/pdf/2111.13844v4)

**Authors**: Zheng Yuan, Jie Zhang, Shiguang Shan

**Abstracts**: Adversarial attacks provide a good way to study the robustness of deep learning models. One category of methods in transfer-based black-box attack utilizes several image transformation operations to improve the transferability of adversarial examples, which is effective, but fails to take the specific characteristic of the input image into consideration. In this work, we propose a novel architecture, called Adaptive Image Transformation Learner (AITL), which incorporates different image transformation operations into a unified framework to further improve the transferability of adversarial examples. Unlike the fixed combinational transformations used in existing works, our elaborately designed transformation learner adaptively selects the most effective combination of image transformations specific to the input image. Extensive experiments on ImageNet demonstrate that our method significantly improves the attack success rates on both normally trained models and defense models under various settings.

摘要: 对抗性攻击为研究深度学习模型的稳健性提供了一种很好的方法。一类基于转移的黑盒攻击方法利用多幅图像变换操作来提高对抗性样本的可转移性，这种方法是有效的，但没有考虑到输入图像的具体特征。在这项工作中，我们提出了一种新的体系结构，称为自适应图像变换学习器(AITL)，它将不同的图像变换操作整合到一个统一的框架中，以进一步提高对抗性例子的可转移性。与现有工作中使用的固定组合变换不同，我们精心设计的变换学习器自适应地选择特定于输入图像的最有效的图像变换组合。在ImageNet上的大量实验表明，该方法在正常训练模型和防御模型上的攻击成功率在各种设置下都有显著提高。



## **6. Understanding Real-world Threats to Deep Learning Models in Android Apps**

了解Android应用程序中深度学习模型面临的现实威胁 cs.CR

18 pages, 9 figures, accepted by CCS'22

**SubmitDate**: 2022-09-28    [paper-pdf](http://arxiv.org/pdf/2209.09577v2)

**Authors**: Zizhuang Deng, Kai Chen, Guozhu Meng, Xiaodong Zhang, Ke Xu, Yao Cheng

**Abstracts**: Famous for its superior performance, deep learning (DL) has been popularly used within many applications, which also at the same time attracts various threats to the models. One primary threat is from adversarial attacks. Researchers have intensively studied this threat for several years and proposed dozens of approaches to create adversarial examples (AEs). But most of the approaches are only evaluated on limited models and datasets (e.g., MNIST, CIFAR-10). Thus, the effectiveness of attacking real-world DL models is not quite clear. In this paper, we perform the first systematic study of adversarial attacks on real-world DNN models and provide a real-world model dataset named RWM. Particularly, we design a suite of approaches to adapt current AE generation algorithms to the diverse real-world DL models, including automatically extracting DL models from Android apps, capturing the inputs and outputs of the DL models in apps, generating AEs and validating them by observing the apps' execution. For black-box DL models, we design a semantic-based approach to build suitable datasets and use them for training substitute models when performing transfer-based attacks. After analyzing 245 DL models collected from 62,583 real-world apps, we have a unique opportunity to understand the gap between real-world DL models and contemporary AE generation algorithms. To our surprise, the current AE generation algorithms can only directly attack 6.53% of the models. Benefiting from our approach, the success rate upgrades to 47.35%.

摘要: 深度学习以其优越的性能而著称，在众多应用中得到了广泛的应用，但同时也给模型带来了各种威胁。其中一个主要威胁来自对抗性攻击。几年来，研究人员对这种威胁进行了深入的研究，并提出了数十种创建对抗性例子(AE)的方法。但大多数方法只在有限的模型和数据集(例如MNIST、CIFAR-10)上进行评估。因此，攻击真实世界的数字图书馆模型的有效性还不是很清楚。在本文中，我们首次对真实世界DNN模型的对抗性攻击进行了系统的研究，并提供了一个真实世界模型数据集RWM。特别是，我们设计了一套方法来使现有的AE生成算法适应不同的真实DL模型，包括自动从Android应用程序中提取DL模型，捕获应用程序中DL模型的输入和输出，生成AE并通过观察应用程序的执行来验证它们。对于黑盒DL模型，我们设计了一种基于语义的方法来构建合适的数据集，并在执行基于传输的攻击时使用它们来训练替代模型。在分析了从62,583个现实世界应用程序中收集的245个DL模型之后，我们有了一个独特的机会来了解现实世界DL模型和当代AE生成算法之间的差距。令我们惊讶的是，目前的AE生成算法只能直接攻击6.53%的模型。受益于我们的方法，成功率提升到47.35%。



## **7. On the Limitations of Stochastic Pre-processing Defenses**

论随机前处理防御的局限性 cs.LG

Accepted by Proceedings of the 36th Conference on Neural Information  Processing Systems

**SubmitDate**: 2022-09-28    [paper-pdf](http://arxiv.org/pdf/2206.09491v2)

**Authors**: Yue Gao, Ilia Shumailov, Kassem Fawaz, Nicolas Papernot

**Abstracts**: Defending against adversarial examples remains an open problem. A common belief is that randomness at inference increases the cost of finding adversarial inputs. An example of such a defense is to apply a random transformation to inputs prior to feeding them to the model. In this paper, we empirically and theoretically investigate such stochastic pre-processing defenses and demonstrate that they are flawed. First, we show that most stochastic defenses are weaker than previously thought; they lack sufficient randomness to withstand even standard attacks like projected gradient descent. This casts doubt on a long-held assumption that stochastic defenses invalidate attacks designed to evade deterministic defenses and force attackers to integrate the Expectation over Transformation (EOT) concept. Second, we show that stochastic defenses confront a trade-off between adversarial robustness and model invariance; they become less effective as the defended model acquires more invariance to their randomization. Future work will need to decouple these two effects. We also discuss implications and guidance for future research.

摘要: 抵御敌意的例子仍然是一个悬而未决的问题。一种普遍的看法是，推理的随机性增加了寻找敌对输入的成本。这种防御的一个例子是在将输入提供给模型之前对它们应用随机转换。在本文中，我们从经验和理论上研究了这种随机预处理防御机制，并证明了它们是有缺陷的。首先，我们证明了大多数随机防御比之前认为的要弱；它们缺乏足够的随机性，即使是像投影梯度下降这样的标准攻击也是如此。这让人对一个长期持有的假设产生了怀疑，即随机防御使旨在逃避确定性防御的攻击无效，并迫使攻击者整合期望过转换(EOT)概念。其次，我们证明了随机防御面临着对抗稳健性和模型不变性之间的权衡；随着被防御模型对其随机化获得更多的不变性，它们变得不那么有效。未来的工作将需要将这两种影响脱钩。我们还讨论了对未来研究的启示和指导。



## **8. Attacking Compressed Vision Transformers**

攻击压缩视觉变形金刚 cs.LG

**SubmitDate**: 2022-09-28    [paper-pdf](http://arxiv.org/pdf/2209.13785v1)

**Authors**: Swapnil Parekh, Devansh Shah, Pratyush Shukla

**Abstracts**: Vision Transformers are increasingly embedded in industrial systems due to their superior performance, but their memory and power requirements make deploying them to edge devices a challenging task. Hence, model compression techniques are now widely used to deploy models on edge devices as they decrease the resource requirements and make model inference very fast and efficient. But their reliability and robustness from a security perspective is another major issue in safety-critical applications. Adversarial attacks are like optical illusions for ML algorithms and they can severely impact the accuracy and reliability of models. In this work we investigate the transferability of adversarial samples across the SOTA Vision Transformer models across 3 SOTA compressed versions and infer the effects different compression techniques have on adversarial attacks.

摘要: 由于其卓越的性能，Vision Transformers越来越多地嵌入到工业系统中，但其内存和电源要求使其部署到边缘设备是一项具有挑战性的任务。因此，模型压缩技术现在被广泛用于在边缘设备上部署模型，因为它们减少了资源需求，使得模型推理非常快速和高效。但从安全角度来看，它们的可靠性和健壮性是安全关键型应用程序的另一个主要问题。对抗性攻击就像ML算法的视觉错觉，它们会严重影响模型的准确性和可靠性。在这项工作中，我们研究了敌意样本在3个SOTA压缩版本中跨Sota视觉变换模型的可转移性，并推断了不同的压缩技术对敌意攻击的影响。



## **9. Suppress with a Patch: Revisiting Universal Adversarial Patch Attacks against Object Detection**

用补丁压制：再论针对目标检测的通用对抗性补丁攻击 cs.CV

Accepted for publication at ICECCME 2022

**SubmitDate**: 2022-09-27    [paper-pdf](http://arxiv.org/pdf/2209.13353v1)

**Authors**: Svetlana Pavlitskaya, Jonas Hendl, Sebastian Kleim, Leopold Müller, Fabian Wylczoch, J. Marius Zöllner

**Abstracts**: Adversarial patch-based attacks aim to fool a neural network with an intentionally generated noise, which is concentrated in a particular region of an input image. In this work, we perform an in-depth analysis of different patch generation parameters, including initialization, patch size, and especially positioning a patch in an image during training. We focus on the object vanishing attack and run experiments with YOLOv3 as a model under attack in a white-box setting and use images from the COCO dataset. Our experiments have shown, that inserting a patch inside a window of increasing size during training leads to a significant increase in attack strength compared to a fixed position. The best results were obtained when a patch was positioned randomly during training, while patch position additionally varied within a batch.

摘要: 基于补丁的对抗性攻击旨在通过故意生成的噪声来愚弄神经网络，这些噪声集中在输入图像的特定区域。在这项工作中，我们对不同的块生成参数进行了深入的分析，包括初始化、块大小，特别是在训练过程中在图像中定位块。我们重点研究了目标消失攻击，并以YOLOv3为模型在白盒环境下进行了实验，并使用了COCO数据集中的图像。我们的实验表明，与固定位置相比，在训练期间在不断增大的窗口内插入补丁可以显著增加攻击强度。当补丁在训练过程中随机定位时，当补丁的位置在批次内另外变化时，效果最好。



## **10. Mitigating Attacks on Artificial Intelligence-based Spectrum Sensing for Cellular Network Signals**

减轻基于人工智能的蜂窝网络信号频谱感知攻击 cs.NI

IEEE GLOBECOM 2022 Publication

**SubmitDate**: 2022-09-27    [paper-pdf](http://arxiv.org/pdf/2209.13007v1)

**Authors**: Ferhat Ozgur Catak, Murat Kuzlu, Salih Sarp, Evren Catak, Umit Cali

**Abstracts**: Cellular networks (LTE, 5G, and beyond) are dramatically growing with high demand from consumers and more promising than the other wireless networks with advanced telecommunication technologies. The main goal of these networks is to connect billions of devices, systems, and users with high-speed data transmission, high cell capacity, and low latency, as well as to support a wide range of new applications, such as virtual reality, metaverse, telehealth, online education, autonomous and flying vehicles, advanced manufacturing, and many more. To achieve these goals, spectrum sensing has been paid more attention, along with new approaches using artificial intelligence (AI) methods for spectrum management in cellular networks. This paper provides a vulnerability analysis of spectrum sensing approaches using AI-based semantic segmentation models for identifying cellular network signals under adversarial attacks with and without defensive distillation methods. The results showed that mitigation methods can significantly reduce the vulnerabilities of AI-based spectrum sensing models against adversarial attacks.

摘要: 随着消费者的高需求，蜂窝网络(LTE、5G和更高)正在急剧增长，比其他采用先进电信技术的无线网络更具前景。这些网络的主要目标是通过高速数据传输、高信元容量和低延迟将数十亿设备、系统和用户连接起来，并支持广泛的新应用，如虚拟现实、虚拟世界、远程医疗、在线教育、自动驾驶和飞行车辆、先进制造等。为了实现这些目标，频谱感知以及在蜂窝网络中使用人工智能(AI)方法进行频谱管理的新方法受到了更多的关注。提出了一种基于人工智能语义分割模型的频谱感知方法的脆弱性分析方法，用于识别具有和不具有防御蒸馏方法的对抗性攻击下的蜂窝网络信号。结果表明，缓解方法可以显著降低基于人工智能的频谱感知模型抵抗敌意攻击的脆弱性。



## **11. Reliable Representations Make A Stronger Defender: Unsupervised Structure Refinement for Robust GNN**

可靠的表示使防御者更强大：健壮GNN的无监督结构求精 cs.LG

Accepted in KDD2022

**SubmitDate**: 2022-09-27    [paper-pdf](http://arxiv.org/pdf/2207.00012v3)

**Authors**: Kuan Li, Yang Liu, Xiang Ao, Jianfeng Chi, Jinghua Feng, Hao Yang, Qing He

**Abstracts**: Benefiting from the message passing mechanism, Graph Neural Networks (GNNs) have been successful on flourish tasks over graph data. However, recent studies have shown that attackers can catastrophically degrade the performance of GNNs by maliciously modifying the graph structure. A straightforward solution to remedy this issue is to model the edge weights by learning a metric function between pairwise representations of two end nodes, which attempts to assign low weights to adversarial edges. The existing methods use either raw features or representations learned by supervised GNNs to model the edge weights. However, both strategies are faced with some immediate problems: raw features cannot represent various properties of nodes (e.g., structure information), and representations learned by supervised GNN may suffer from the poor performance of the classifier on the poisoned graph. We need representations that carry both feature information and as mush correct structure information as possible and are insensitive to structural perturbations. To this end, we propose an unsupervised pipeline, named STABLE, to optimize the graph structure. Finally, we input the well-refined graph into a downstream classifier. For this part, we design an advanced GCN that significantly enhances the robustness of vanilla GCN without increasing the time complexity. Extensive experiments on four real-world graph benchmarks demonstrate that STABLE outperforms the state-of-the-art methods and successfully defends against various attacks.

摘要: 得益于消息传递机制，图神经网络(GNN)已经成功地处理了大量的图数据任务。然而，最近的研究表明，攻击者可以通过恶意修改图结构来灾难性地降低GNN的性能。解决这一问题的一个直接解决方案是通过学习两个末端节点的成对表示之间的度量函数来对边权重进行建模，该度量函数试图为对抗性边分配较低的权重。现有的方法要么使用原始特征，要么使用由监督GNN学习的表示来对边权重进行建模。然而，这两种策略都面临着一些迫在眉睫的问题：原始特征不能表示节点的各种属性(例如结构信息)，而有监督GNN学习的表示可能会受到有毒图上分类器性能较差的影响。我们需要既携带特征信息又尽可能正确的结构信息并对结构扰动不敏感的表示法。为此，我们提出了一种名为STRATE的无监督流水线来优化图的结构。最后，我们将精化后的图输入到下游分类器中。对于这一部分，我们设计了一种改进的GCN，它在不增加时间复杂度的情况下显著增强了普通GCN的健壮性。在四个真实图形基准上的大量实验表明，STRATE的性能优于最先进的方法，并成功地防御了各种攻击。



## **12. FG-UAP: Feature-Gathering Universal Adversarial Perturbation**

FG-UAP：特征收集通用对抗性扰动 cs.CV

27 pages, 4 figures

**SubmitDate**: 2022-09-27    [paper-pdf](http://arxiv.org/pdf/2209.13113v1)

**Authors**: Zhixing Ye, Xinwen Cheng, Xiaolin Huang

**Abstracts**: Deep Neural Networks (DNNs) are susceptible to elaborately designed perturbations, whether such perturbations are dependent or independent of images. The latter one, called Universal Adversarial Perturbation (UAP), is very attractive for model robustness analysis, since its independence of input reveals the intrinsic characteristics of the model. Relatively, another interesting observation is Neural Collapse (NC), which means the feature variability may collapse during the terminal phase of training. Motivated by this, we propose to generate UAP by attacking the layer where NC phenomenon happens. Because of NC, the proposed attack could gather all the natural images' features to its surrounding, which is hence called Feature-Gathering UAP (FG-UAP).   We evaluate the effectiveness our proposed algorithm on abundant experiments, including untargeted and targeted universal attacks, attacks under limited dataset, and transfer-based black-box attacks among different architectures including Vision Transformers, which are believed to be more robust. Furthermore, we investigate FG-UAP in the view of NC by analyzing the labels and extracted features of adversarial examples, finding that collapse phenomenon becomes stronger after the model is corrupted. The code will be released when the paper is accepted.

摘要: 深度神经网络(DNN)容易受到精心设计的扰动的影响，无论这种扰动是依赖于还是独立于图像。后者被称为通用对抗性摄动(UAP)，由于它独立于输入，揭示了模型的内在特征，因此对模型的稳健性分析非常有吸引力。相对而言，另一个有趣的观察是神经崩溃(NC)，这意味着特征变异性可能在训练的最后阶段崩溃。受此启发，我们提出通过攻击NC现象发生的层来生成UAP。由于NC的存在，该攻击可以将自然图像的所有特征聚集到其周围，因此被称为特征收集UAP(FG-UAP)。我们在大量的实验中对我们提出的算法的有效性进行了评估，包括非目标攻击和目标通用攻击，有限数据集下的攻击，以及不同架构之间的基于传输的黑盒攻击，包括Vision Transformers，它们被认为是更健壮的。此外，我们从NC的角度对FG-UAP进行了研究，通过分析对抗性实例的标签和提取的特征，发现模型被破坏后崩溃现象变得更加强烈。当论文被接受时，代码将被发布。



## **13. Cascading Failures in Power Grids**

电网中的连锁故障 eess.SY

**SubmitDate**: 2022-09-26    [paper-pdf](http://arxiv.org/pdf/2209.08116v2)

**Authors**: Rounak Meyur

**Abstracts**: This paper studies the consequences of a human-initiated targeted attack on the national electric power system. We consider two kinds of attacks: ($i$) an attack by an adversary that uses a tactical weapon and destroys a large part of the grid, by physically targeting a large geographic region; ($ii$) a targeted attack by an adversary that takes out a small number of critical components in the network simultaneously. Our analysis uses ($i$) a realistic representation of the underlying power grid, including the topology, the control and protection components, ($ii$) a realistic representation of the targeted attack scenario, and ($iii$) a dynamic stability analysis, that goes beyond traditional work comprising structural and linear flow analysis. Such realistic analysis is expensive, but critical since it can capture cascading failures that result from transient instabilities introduced due to the attack. Our model acknowledges the presence of hidden failures in the protection systems resulting in relay misoperations. We analyze the extent of cascading outages for different levels of hidden failures. Our results show that: ($i$) the power grid is vulnerable to both these attacks, ($ii$) the tactical attack has significant social, economic and health damage but need not result in a regional cascade; on the contrary the targeted attack can cause significant cascade and lead to power outage over a large region. Our work shows the necessity to harden the power grid not just to cyber-attacks but also to physical attacks. Furthermore, we show that realistic representations and analysis can lead to fundamentally new insights that simplified models are unlikely to capture. Finally, the methods and results help us identify critical elements in the grid; the system can then be hardened in a more precise manner to reduce the vulnerabilities.

摘要: 本文研究了人类发起的对国家电力系统的定向攻击的后果。我们考虑了两种类型的攻击：($I$)对手使用战术武器，通过物理上以大片地理区域为目标摧毁大部分电网的攻击；($II$)对手的有针对性的攻击，同时摧毁网络中的少量关键组件。我们的分析使用了($I$)基本电网的真实表示，包括拓扑、控制和保护组件，($II$)目标攻击场景的真实表示，以及($III$)动态稳定性分析，这超出了传统的包括结构和线性流分析的工作。这种现实的分析成本很高，但很关键，因为它可以捕获由攻击导致的瞬时不稳定导致的级联故障。我们的模型承认保护系统中存在导致继电保护误动作的隐藏故障。我们分析了不同级别的隐藏故障的连锁故障程度。我们的结果表明：($I$)电网容易受到这两种攻击，($II$)战术攻击具有重大的社会、经济和健康损害，但不一定会导致区域级联；相反，定向攻击可以造成重大的级联，并导致大范围的停电。我们的工作表明，加强电网不仅要抵御网络攻击，还要抵御物理攻击。此外，我们还表明，现实的表示和分析可以带来简化模型不太可能捕捉到的根本新的见解。最后，这些方法和结果帮助我们识别网格中的关键元素；然后可以以更精确的方式加强系统，以减少漏洞。



## **14. ASK: Adversarial Soft k-Nearest Neighbor Attack and Defense**

问：对抗性软k近邻攻击与防御 cs.LG

**SubmitDate**: 2022-09-26    [paper-pdf](http://arxiv.org/pdf/2106.14300v4)

**Authors**: Ren Wang, Tianqi Chen, Philip Yao, Sijia Liu, Indika Rajapakse, Alfred Hero

**Abstracts**: K-Nearest Neighbor (kNN)-based deep learning methods have been applied to many applications due to their simplicity and geometric interpretability. However, the robustness of kNN-based classification models has not been thoroughly explored and kNN attack strategies are underdeveloped. In this paper, we propose an Adversarial Soft kNN (ASK) loss to both design more effective kNN attack strategies and to develop better defenses against them. Our ASK loss approach has two advantages. First, ASK loss can better approximate the kNN's probability of classification error than objectives proposed in previous works. Second, the ASK loss is interpretable: it preserves the mutual information between the perturbed input and the in-class-reference data. We use the ASK loss to generate a novel attack method called the ASK-Attack (ASK-Atk), which shows superior attack efficiency and accuracy degradation relative to previous kNN attacks. Based on the ASK-Atk, we then derive an ASK-\underline{Def}ense (ASK-Def) method that optimizes the worst-case training loss induced by ASK-Atk. Experiments on CIFAR-10 (ImageNet) show that (i) ASK-Atk achieves $\geq 13\%$ ($\geq 13\%$) improvement in attack success rate over previous kNN attacks, and (ii) ASK-Def outperforms the conventional adversarial training method by $\geq 6.9\%$ ($\geq 3.5\%$) in terms of robustness improvement.

摘要: 基于K-近邻(KNN)的深度学习方法因其简单性和几何可解释性而被广泛应用。然而，基于KNN的分类模型的稳健性还没有得到深入的研究，KNN攻击策略也不够完善。在本文中，我们提出了一种对抗性软KNN(ASK)损失，以设计更有效的KNN攻击策略，并对它们进行更好的防御。我们的要价损失方法有两个优点。首先，与以往工作中提出的目标分类相比，ASK损失能够更好地逼近KNN的分类错误概率。其次，ASK损失是可解释的：它保留了扰动输入和类内参考数据之间的互信息。利用ASK损失生成了一种新的攻击方法ASK-ATK(ASK-ATK)，相对于以往的KNN攻击，该方法具有更高的攻击效率和更低的准确率。在ASK-ATK的基础上，我们推导出了一种ASK-下划线{Def}ense(ASK-Def)方法，该方法优化了ASK-ATK造成的最坏情况下的训练损失。在CIFAR-10(ImageNet)上的实验表明：(1)ASK-ATK的攻击成功率比以前的KNN攻击提高了1 3(1 3)；(2)在健壮性方面，ASK-Def比传统的对抗性训练方法提高了6.9(3.5)%。



## **15. Formally verified asymptotic consensus in robust networks**

形式验证鲁棒网络中的渐近一致性 cs.PL

**SubmitDate**: 2022-09-26    [paper-pdf](http://arxiv.org/pdf/2202.13833v2)

**Authors**: Mohit Tekriwal, Avi Tachna-Fram, Jean-Baptiste Jeannin, Manos Kapritsos, Dimitra Panagou

**Abstracts**: Distributed architectures are used to improve performance and reliability of various systems. An important capability of a distributed architecture is the ability to reach consensus among all its nodes. To achieve this, several consensus algorithms have been proposed for various scenarii, and many of these algorithms come with proofs of correctness that are not mechanically checked. Unfortunately, those proofs are known to be intricate and prone to errors.   In this paper, we formalize and mechanically check a consensus algorithm widely used in the distributed controls community: the Weighted-Mean Subsequence Reduced (W-MSR) algorithm proposed by Le Blanc et al. This algorithm provides a way to achieve asymptotic consensus in a distributed controls scenario in the presence of adversarial agents (attackers) that may not update their states based on the nominal consensus protocol, and may share inaccurate information with their neighbors. Using the Coq proof assistant, we formalize the necessary and sufficient conditions required to achieve resilient asymptotic consensus under the assumed attacker model. We leverage the existing Coq formalizations of graph theory, finite sets and sequences of the mathcomp library for our development. To our knowledge, this is the first mechanical proof of an asymptotic consensus algorithm. During the formalization, we clarify several imprecisions in the paper proof, including an imprecision on quantifiers in the main theorem.

摘要: 分布式体系结构被用来提高各种系统的性能和可靠性。分布式体系结构的一个重要能力是在其所有节点之间达成共识的能力。为了实现这一点，已经针对不同的场景提出了几种共识算法，其中许多算法都带有不经过机械检查的正确性证明。不幸的是，众所周知，这些证明是错综复杂的，容易出错。本文对一种广泛应用于分布式控制领域的一致性算法进行了形式化和机械检验：由Le Blanc等人提出的加权平均子序列简化(W-MSR)算法。该算法提供了一种在分布式控制场景中获得渐近共识的方法，在存在可能不基于名义共识协议更新其状态并且可能与其邻居共享不准确信息的对手代理(攻击者)存在的情况下。利用CoQ证明助手，我们形式化了在假设的攻击者模型下实现弹性渐近共识所需的充要条件。我们利用现有的图论、有限集和Mathcomp库序列的CoQ形式化来进行开发。据我们所知，这是渐近共识算法的第一个机械证明。在形式化过程中，我们澄清了论文证明中的几个不精确之处，包括主要定理中关于量词的不精确。



## **16. RORL: Robust Offline Reinforcement Learning via Conservative Smoothing**

RORL：基于保守平滑的稳健离线强化学习 cs.LG

Accepted by Advances in Neural Information Processing Systems  (NeurIPS) 2022

**SubmitDate**: 2022-09-26    [paper-pdf](http://arxiv.org/pdf/2206.02829v2)

**Authors**: Rui Yang, Chenjia Bai, Xiaoteng Ma, Zhaoran Wang, Chongjie Zhang, Lei Han

**Abstracts**: Offline reinforcement learning (RL) provides a promising direction to exploit the massive amount of offline data for complex decision-making tasks. Due to the distribution shift issue, current offline RL algorithms are generally designed to be conservative in value estimation and action selection. However, such conservatism can impair the robustness of learned policies when encountering observation deviation under realistic conditions, such as sensor errors and adversarial attacks. To trade off robustness and conservatism, we propose Robust Offline Reinforcement Learning (RORL) with a novel conservative smoothing technique. In RORL, we explicitly introduce regularization on the policy and the value function for states near the dataset, as well as additional conservative value estimation on these OOD states. Theoretically, we show RORL enjoys a tighter suboptimality bound than recent theoretical results in linear MDPs. We demonstrate that RORL can achieve state-of-the-art performance on the general offline RL benchmark and is considerably robust to adversarial observation perturbations.

摘要: 离线强化学习(RL)为利用海量的离线数据进行复杂的决策任务提供了一个很有前途的方向。由于分布平移问题，目前的离线RL算法在价值估计和动作选择上通常被设计为保守的。然而，当在实际条件下遇到观测偏差时，这种保守性会削弱学习策略的稳健性，例如传感器错误和对抗性攻击。为了权衡稳健性和保守性，我们提出了一种新的保守平滑技术的稳健离线强化学习(RORL)。在RORL中，我们明确地引入了策略的正则化和数据集附近状态的值函数，以及对这些OOD状态的附加保守值估计。理论上，我们证明了RORL在线性MDP中享有比最近的理论结果更紧的次优界。我们证明了RORL可以在一般的离线RL基准上获得最先进的性能，并且对对抗性观测扰动具有相当强的鲁棒性。



## **17. Black-Box Dissector: Towards Erasing-based Hard-Label Model Stealing Attack**

黑盒剖析器：面向擦除的硬标签模型窃取攻击 cs.CV

**SubmitDate**: 2022-09-26    [paper-pdf](http://arxiv.org/pdf/2105.00623v3)

**Authors**: Yixu Wang, Jie Li, Hong Liu, Yan Wang, Yongjian Wu, Feiyue Huang, Rongrong Ji

**Abstracts**: Previous studies have verified that the functionality of black-box models can be stolen with full probability outputs. However, under the more practical hard-label setting, we observe that existing methods suffer from catastrophic performance degradation. We argue this is due to the lack of rich information in the probability prediction and the overfitting caused by hard labels. To this end, we propose a novel hard-label model stealing method termed \emph{black-box dissector}, which consists of two erasing-based modules. One is a CAM-driven erasing strategy that is designed to increase the information capacity hidden in hard labels from the victim model. The other is a random-erasing-based self-knowledge distillation module that utilizes soft labels from the substitute model to mitigate overfitting. Extensive experiments on four widely-used datasets consistently demonstrate that our method outperforms state-of-the-art methods, with an improvement of at most $8.27\%$. We also validate the effectiveness and practical potential of our method on real-world APIs and defense methods. Furthermore, our method promotes other downstream tasks, \emph{i.e.}, transfer adversarial attacks.

摘要: 以前的研究已经证实，黑盒模型的功能可以通过全概率输出被窃取。然而，在更实际的硬标签设置下，我们观察到现有方法遭受灾难性的性能下降。我们认为，这是由于概率预测中缺乏丰富的信息，以及硬标签造成的过度拟合造成的。为此，我们提出了一种新的硬标签模型窃取方法--EMPH(黑盒解析器)，它由两个基于擦除的模块组成。一种是CAM驱动的擦除策略，旨在增加隐藏在硬标签中的信息容量，使其不受受害者模型的影响。另一种是基于随机擦除的自我知识提炼模块，它利用替换模型中的软标签来缓解过度拟合。在四个广泛使用的数据集上的大量实验一致表明，我们的方法比最先进的方法性能更好，最多只有8.27美元的改进。我们还在真实的API和防御方法上验证了我们的方法的有效性和实用潜力。此外，我们的方法还促进了其他下游任务，即转移敌意攻击。



## **18. Activation Learning by Local Competitions**

通过本地竞争实现激活学习 cs.NE

31pages

**SubmitDate**: 2022-09-26    [paper-pdf](http://arxiv.org/pdf/2209.13400v1)

**Authors**: Hongchao Zhou

**Abstracts**: The backpropagation that drives the success of deep learning is most likely different from the learning mechanism of the brain. In this paper, we develop a biology-inspired learning rule that discovers features by local competitions among neurons, following the idea of Hebb's famous proposal. It is demonstrated that the unsupervised features learned by this local learning rule can serve as a pre-training model to improve the performance of some supervised learning tasks. More importantly, this local learning rule enables us to build a new learning paradigm very different from the backpropagation, named activation learning, where the output activation of the neural network roughly measures how probable the input patterns are. The activation learning is capable of learning plentiful local features from few shots of input patterns, and demonstrates significantly better performances than the backpropagation algorithm when the number of training samples is relatively small. This learning paradigm unifies unsupervised learning, supervised learning and generative models, and is also more secure against adversarial attack, paving a road to some possibilities of creating general-task neural networks.

摘要: 驱动深度学习成功的反向传播很可能与大脑的学习机制不同。在本文中，我们根据Hebb著名的建议，开发了一种生物启发的学习规则，通过神经元之间的局部竞争来发现特征。实验结果表明，该局部学习规则所学习的非监督特征可以作为预训练模型，以提高某些监督学习任务的性能。更重要的是，这种局部学习规则使我们能够建立一种与反向传播非常不同的新学习范式，称为激活学习，在后向传播中，神经网络的输出激活大致衡量输入模式的概率。激活学习能够从较少的输入模式中学习丰富的局部特征，并且在训练样本数量相对较少的情况下表现出明显优于反向传播算法的性能。这种学习范式统一了无监督学习、监督学习和产生式模型，并且对对手攻击更加安全，为创建通用任务神经网络铺平了道路。



## **19. Exploiting Trust for Resilient Hypothesis Testing with Malicious Robots**

利用信任对恶意机器人进行弹性假设检验 cs.RO

12 pages, 4 figures, extended version of conference submission

**SubmitDate**: 2022-09-25    [paper-pdf](http://arxiv.org/pdf/2209.12285v1)

**Authors**: Matthew Cavorsi, Orhan Eren Akgün, Michal Yemini, Andrea Goldsmith, Stephanie Gil

**Abstracts**: We develop a resilient binary hypothesis testing framework for decision making in adversarial multi-robot crowdsensing tasks. This framework exploits stochastic trust observations between robots to arrive at tractable, resilient decision making at a centralized Fusion Center (FC) even when i) there exist malicious robots in the network and their number may be larger than the number of legitimate robots, and ii) the FC uses one-shot noisy measurements from all robots. We derive two algorithms to achieve this. The first is the Two Stage Approach (2SA) that estimates the legitimacy of robots based on received trust observations, and provably minimizes the probability of detection error in the worst-case malicious attack. Here, the proportion of malicious robots is known but arbitrary. For the case of an unknown proportion of malicious robots, we develop the Adversarial Generalized Likelihood Ratio Test (A-GLRT) that uses both the reported robot measurements and trust observations to estimate the trustworthiness of robots, their reporting strategy, and the correct hypothesis simultaneously. We exploit special problem structure to show that this approach remains computationally tractable despite several unknown problem parameters. We deploy both algorithms in a hardware experiment where a group of robots conducts crowdsensing of traffic conditions on a mock-up road network similar in spirit to Google Maps, subject to a Sybil attack. We extract the trust observations for each robot from actual communication signals which provide statistical information on the uniqueness of the sender. We show that even when the malicious robots are in the majority, the FC can reduce the probability of detection error to 30.5% and 29% for the 2SA and the A-GLRT respectively.

摘要: 提出了一种用于对抗性多机器人群体感知任务决策的弹性二元假设检验框架。该框架利用机器人之间的随机信任观察，即使在i)网络中存在恶意机器人并且它们的数量可能大于合法机器人的数量，以及ii)FC使用来自所有机器人的一次噪声测量的情况下，也可以在集中式融合中心(FC)获得易于处理的、有弹性的决策。我们推导了两个算法来实现这一点。第一种是两阶段方法(2SA)，它根据接收到的信任观察来估计机器人的合法性，并证明在最坏情况下恶意攻击的检测错误概率最小。在这里，恶意机器人的比例是已知的，但是随意的。对于恶意机器人比例未知的情况，我们提出了对抗性广义似然比检验(A-GLRT)，它同时使用报告的机器人测量值和信任观察来估计机器人的可信性、报告策略和正确的假设。我们利用特殊的问题结构表明，尽管有几个未知的问题参数，该方法在计算上仍然是容易处理的。我们在硬件实验中部署了这两种算法，在硬件实验中，一组机器人在一个类似于谷歌地图的模拟道路网络上对交通状况进行众感，受到Sybil攻击。我们从提供关于发送者唯一性的统计信息的实际通信信号中提取每个机器人的信任观察。实验结果表明，即使恶意机器人占多数，FC算法也能将2SA和A-GLRT的误检率分别降低到30.5%和29%。



## **20. Residue-Based Natural Language Adversarial Attack Detection**

基于残差的自然语言敌意攻击检测 cs.CL

**SubmitDate**: 2022-09-25    [paper-pdf](http://arxiv.org/pdf/2204.10192v2)

**Authors**: Vyas Raina, Mark Gales

**Abstracts**: Deep learning based systems are susceptible to adversarial attacks, where a small, imperceptible change at the input alters the model prediction. However, to date the majority of the approaches to detect these attacks have been designed for image processing systems. Many popular image adversarial detection approaches are able to identify adversarial examples from embedding feature spaces, whilst in the NLP domain existing state of the art detection approaches solely focus on input text features, without consideration of model embedding spaces. This work examines what differences result when porting these image designed strategies to Natural Language Processing (NLP) tasks - these detectors are found to not port over well. This is expected as NLP systems have a very different form of input: discrete and sequential in nature, rather than the continuous and fixed size inputs for images. As an equivalent model-focused NLP detection approach, this work proposes a simple sentence-embedding "residue" based detector to identify adversarial examples. On many tasks, it out-performs ported image domain detectors and recent state of the art NLP specific detectors.

摘要: 基于深度学习的系统很容易受到对抗性攻击，在这种攻击中，输入端微小的、不可察觉的变化就会改变模型预测。然而，到目前为止，大多数检测这些攻击的方法都是为图像处理系统设计的。许多流行的图像对抗性检测方法能够从嵌入的特征空间中识别对抗性样本，而在NLP领域，现有的检测方法只关注输入文本特征，而没有考虑模型嵌入空间。这项工作考察了将这些图像设计的策略移植到自然语言处理(NLP)任务中时会产生什么不同--这些检测器被发现移植得不好。这是意料之中的，因为NLP系统具有非常不同的输入形式：本质上是离散的和连续的，而不是图像的连续和固定大小的输入。作为一种等价的基于模型的NLP检测方法，本文提出了一种简单的基于句子嵌入“残差”的检测器来识别对抗性实例。在许多任务上，它的性能优于端口图像域检测器和最新的NLP特定检测器。



## **21. SPRITZ-1.5C: Employing Deep Ensemble Learning for Improving the Security of Computer Networks against Adversarial Attacks**

SPRITZ-1.5C：利用深度集成学习提高计算机网络抗恶意攻击的安全性 cs.CR

**SubmitDate**: 2022-09-25    [paper-pdf](http://arxiv.org/pdf/2209.12195v1)

**Authors**: Ehsan Nowroozi, Mohammadreza Mohammadi, Erkay Savas, Mauro Conti, Yassine Mekdad

**Abstracts**: In the past few years, Convolutional Neural Networks (CNN) have demonstrated promising performance in various real-world cybersecurity applications, such as network and multimedia security. However, the underlying fragility of CNN structures poses major security problems, making them inappropriate for use in security-oriented applications including such computer networks. Protecting these architectures from adversarial attacks necessitates using security-wise architectures that are challenging to attack.   In this study, we present a novel architecture based on an ensemble classifier that combines the enhanced security of 1-Class classification (known as 1C) with the high performance of conventional 2-Class classification (known as 2C) in the absence of attacks.Our architecture is referred to as the 1.5-Class (SPRITZ-1.5C) classifier and constructed using a final dense classifier, one 2C classifier (i.e., CNNs), and two parallel 1C classifiers (i.e., auto-encoders). In our experiments, we evaluated the robustness of our proposed architecture by considering eight possible adversarial attacks in various scenarios. We performed these attacks on the 2C and SPRITZ-1.5C architectures separately. The experimental results of our study showed that the Attack Success Rate (ASR) of the I-FGSM attack against a 2C classifier trained with the N-BaIoT dataset is 0.9900. In contrast, the ASR is 0.0000 for the SPRITZ-1.5C classifier.

摘要: 在过去的几年里，卷积神经网络(CNN)在各种现实世界的网络安全应用中表现出了良好的性能，如网络和多媒体安全。然而，CNN结构的潜在脆弱性造成了重大的安全问题，使其不适合用于包括此类计算机网络在内的面向安全的应用程序。要保护这些架构免受敌意攻击，就必须使用具有安全性的架构，这些架构很难受到攻击。在这项研究中，我们提出了一种新的基于集成分类器的体系结构，它结合了1类分类(称为1C)的增强安全性和传统2类分类(称为2C)的高性能，在没有攻击的情况下被称为1.5类(SPRITZ-1.5C)分类器，并使用最终的密集分类器、一个2C分类器(即CNN)和两个并行的1C分类器(即自动编码器)来构建。在我们的实验中，我们通过考虑不同场景中八种可能的对抗性攻击来评估我们所提出的体系结构的健壮性。我们分别在2C和SPRITZ-1.5C架构上执行了这些攻击。实验结果表明，利用N-BaIoT数据集训练的2C分类器对I-FGSM的攻击成功率为0.9900。相比之下，SPRITZ-1.5C分级机的ASR为0.0000。



## **22. Robust Reinforcement Learning as a Stackelberg Game via Adaptively-Regularized Adversarial Training**

基于自适应正则化对抗性训练的Stackelberg博弈的稳健强化学习 cs.LG

**SubmitDate**: 2022-09-24    [paper-pdf](http://arxiv.org/pdf/2202.09514v2)

**Authors**: Peide Huang, Mengdi Xu, Fei Fang, Ding Zhao

**Abstracts**: Robust Reinforcement Learning (RL) focuses on improving performances under model errors or adversarial attacks, which facilitates the real-life deployment of RL agents. Robust Adversarial Reinforcement Learning (RARL) is one of the most popular frameworks for robust RL. However, most of the existing literature models RARL as a zero-sum simultaneous game with Nash equilibrium as the solution concept, which could overlook the sequential nature of RL deployments, produce overly conservative agents, and induce training instability. In this paper, we introduce a novel hierarchical formulation of robust RL - a general-sum Stackelberg game model called RRL-Stack - to formalize the sequential nature and provide extra flexibility for robust training. We develop the Stackelberg Policy Gradient algorithm to solve RRL-Stack, leveraging the Stackelberg learning dynamics by considering the adversary's response. Our method generates challenging yet solvable adversarial environments which benefit RL agents' robust learning. Our algorithm demonstrates better training stability and robustness against different testing conditions in the single-agent robotics control and multi-agent highway merging tasks.

摘要: 稳健强化学习(RL)专注于提高在模型错误或敌意攻击下的性能，这有助于RL代理的实际部署。稳健对抗强化学习(RARL)是目前最流行的稳健对抗强化学习框架之一。然而，现有文献大多将RARL建模为以纳什均衡为解概念的零和同时博弈，这可能会忽略RL部署的序贯性质，产生过于保守的代理，并导致训练不稳定。在本文中，我们引入了一种新的健壮RL的分层表示-一个称为RRL-Stack的一般和Stackelberg博弈模型-以形式化顺序性质并为健壮训练提供额外的灵活性。我们开发了Stackelberg策略梯度算法来求解RRL-Stack，通过考虑对手的响应来利用Stackelberg学习动态。我们的方法产生具有挑战性但可解决的对抗环境，这有利于RL代理的稳健学习。在单智能体机器人控制和多智能体公路合并任务中，我们的算法对不同的测试条件表现出了更好的训练稳定性和鲁棒性。



## **23. RSD-GAN: Regularized Sobolev Defense GAN Against Speech-to-Text Adversarial Attacks**

RSD-GAN：正规化Sobolev防御GAN防止语音到文本的对抗性攻击 cs.SD

Paper ACCEPTED FOR PUBLICATION IEEE Signal Processing Letters Journal

**SubmitDate**: 2022-09-24    [paper-pdf](http://arxiv.org/pdf/2207.06858v2)

**Authors**: Mohammad Esmaeilpour, Nourhene Chaalia, Patrick Cardinal

**Abstracts**: This paper introduces a new synthesis-based defense algorithm for counteracting with a varieties of adversarial attacks developed for challenging the performance of the cutting-edge speech-to-text transcription systems. Our algorithm implements a Sobolev-based GAN and proposes a novel regularizer for effectively controlling over the functionality of the entire generative model, particularly the discriminator network during training. Our achieved results upon carrying out numerous experiments on the victim DeepSpeech, Kaldi, and Lingvo speech transcription systems corroborate the remarkable performance of our defense approach against a comprehensive range of targeted and non-targeted adversarial attacks.

摘要: 本文介绍了一种新的基于合成的防御算法，用于对抗各种针对尖端语音到文本转录系统性能的挑战而开发的对抗性攻击。我们的算法实现了一种基于Sobolev的GAN，并提出了一种新的正则化算法来有效地控制整个生成模型的功能，特别是在训练过程中的鉴别器网络。我们在受害者DeepSpeech、Kaldi和Lingvo语音转录系统上进行的大量实验所取得的结果证实了我们的防御方法在应对全面的定向和非定向对手攻击方面的卓越表现。



## **24. Approximate better, Attack stronger: Adversarial Example Generation via Asymptotically Gaussian Mixture Distribution**

近似更好，攻击更强：基于渐近高斯混合分布的对抗性实例生成 cs.LG

**SubmitDate**: 2022-09-24    [paper-pdf](http://arxiv.org/pdf/2209.11964v1)

**Authors**: Zhengwei Fang, Rui Wang, Tao Huang, Liping Jing

**Abstracts**: Strong adversarial examples are the keys to evaluating and enhancing the robustness of deep neural networks. The popular adversarial attack algorithms maximize the non-concave loss function using the gradient ascent. However, the performance of each attack is usually sensitive to, for instance, minor image transformations due to insufficient information (only one input example, few white-box source models and unknown defense strategies). Hence, the crafted adversarial examples are prone to overfit the source model, which limits their transferability to unidentified architectures. In this paper, we propose Multiple Asymptotically Normal Distribution Attacks (MultiANDA), a novel method that explicitly characterizes adversarial perturbations from a learned distribution. Specifically, we approximate the posterior distribution over the perturbations by taking advantage of the asymptotic normality property of stochastic gradient ascent (SGA), then apply the ensemble strategy on this procedure to estimate a Gaussian mixture model for a better exploration of the potential optimization space. Drawing perturbations from the learned distribution allow us to generate any number of adversarial examples for each input. The approximated posterior essentially describes the stationary distribution of SGA iterations, which captures the geometric information around the local optimum. Thus, the samples drawn from the distribution reliably maintain the transferability. Our proposed method outperforms nine state-of-the-art black-box attacks on deep learning models with or without defenses through extensive experiments on seven normally trained and seven defence models.

摘要: 强对抗性例子是评价和提高深度神经网络健壮性的关键。流行的对抗性攻击算法利用梯度上升最大化非凹损失函数。然而，由于信息不足(只有一个输入，很少的白盒源模型和未知的防御策略)，每个攻击的性能通常对例如较小的图像变换很敏感。因此，精心制作的敌意示例容易与源模型过度匹配，这限制了它们向未识别的体系结构的可转移性。在本文中，我们提出了多重渐近正态分布攻击(Multiple渐近正态分布攻击)，这是一种从学习分布中显式刻画敌意扰动的新方法。具体地说，我们利用随机梯度上升(SGA)的渐近正态性质来逼近扰动下的后验分布，然后将集成策略应用于该过程来估计高斯混合模型，以更好地探索潜在的优化空间。从学习的分布中提取扰动允许我们为每个输入生成任意数量的对抗性示例。近似后验概率本质上描述了SGA迭代的平稳分布，它捕捉了局部最优解附近的几何信息。因此，从分布中提取的样本可靠地保持了可转移性。通过在7个正常训练的模型和7个防御模型上的大量实验，我们提出的方法在有或没有防御的深度学习模型上的性能超过了9种最先进的黑盒攻击。



## **25. Faith: An Efficient Framework for Transformer Verification on GPUs**

FACES：一种高效的基于GPU的变压器验证框架 cs.LG

Published in ATC'22

**SubmitDate**: 2022-09-23    [paper-pdf](http://arxiv.org/pdf/2209.12708v1)

**Authors**: Boyuan Feng, Tianqi Tang, Yuke Wang, Zhaodong Chen, Zheng Wang, Shu Yang, Yuan Xie, Yufei Ding

**Abstracts**: Transformer verification draws increasing attention in machine learning research and industry. It formally verifies the robustness of transformers against adversarial attacks such as exchanging words in a sentence with synonyms. However, the performance of transformer verification is still not satisfactory due to bound-centric computation which is significantly different from standard neural networks. In this paper, we propose Faith, an efficient framework for transformer verification on GPUs. We first propose a semantic-aware computation graph transformation to identify semantic information such as bound computation in transformer verification. We exploit such semantic information to enable efficient kernel fusion at the computation graph level. Second, we propose a verification-specialized kernel crafter to efficiently map transformer verification to modern GPUs. This crafter exploits a set of GPU hardware supports to accelerate verification specialized operations which are usually memory-intensive. Third, we propose an expert-guided autotuning to incorporate expert knowledge on GPU backends to facilitate large search space exploration. Extensive evaluations show that Faith achieves $2.1\times$ to $3.4\times$ ($2.6\times$ on average) speedup over state-of-the-art frameworks.

摘要: 变压器验证在机器学习研究和工业领域受到越来越多的关注。它形式化地验证了转换器对对抗性攻击的健壮性，例如在句子中使用同义词交换单词。然而，由于以边界为中心的计算与标准神经网络有很大的不同，变压器验证的性能仍然不令人满意。本文提出了一种高效的基于GPU的变压器验证框架FAITH。我们首先提出了一种语义感知的计算图变换来识别变压器验证中的边界计算等语义信息。我们利用这些语义信息在计算图级实现有效的核融合。其次，我们提出了一种验证专用的内核工艺器来高效地将转换器验证映射到现代GPU上。这种技术利用了一组GPU硬件支持来加速通常是内存密集型的专用验证操作。第三，我们提出了一种专家引导的自动调优方法，融合了关于GPU后端的专家知识，以促进大搜索空间的探索。广泛的评估表明，Faith在最先进的框架上实现了2.1倍到3.4倍的加速(平均为2.6倍)。



## **26. Real-time Adversarial Perturbations against Deep Reinforcement Learning Policies: Attacks and Defenses**

深度强化学习策略的实时对抗性扰动：攻击与防御 cs.LG

Will appear in the proceedings of ESORICS 2022; 13 pages, 6 figures,  6 tables

**SubmitDate**: 2022-09-23    [paper-pdf](http://arxiv.org/pdf/2106.08746v4)

**Authors**: Buse G. A. Tekgul, Shelly Wang, Samuel Marchal, N. Asokan

**Abstracts**: Deep reinforcement learning (DRL) is vulnerable to adversarial perturbations. Adversaries can mislead the policies of DRL agents by perturbing the state of the environment observed by the agents. Existing attacks are feasible in principle, but face challenges in practice, either by being too slow to fool DRL policies in real time or by modifying past observations stored in the agent's memory. We show that Universal Adversarial Perturbations (UAP), independent of the individual inputs to which they are applied, can fool DRL policies effectively and in real time. We introduce three attack variants leveraging UAP. Via an extensive evaluation using three Atari 2600 games, we show that our attacks are effective, as they fully degrade the performance of three different DRL agents (up to 100%, even when the $l_\infty$ bound on the perturbation is as small as 0.01). It is faster than the frame rate (60 Hz) of image capture and considerably faster than prior attacks ($\approx 1.8$ms). Our attack technique is also efficient, incurring an online computational cost of $\approx 0.027$ms. Using two tasks involving robotic movement, we confirm that our results generalize to complex DRL tasks. Furthermore, we demonstrate that the effectiveness of known defenses diminishes against universal perturbations. We introduce an effective technique that detects all known adversarial perturbations against DRL policies, including all universal perturbations presented in this paper.

摘要: 深度强化学习(DRL)容易受到对抗性扰动的影响。攻击者可以通过扰乱代理观察到的环境状态来误导DRL代理的策略。现有的攻击在原则上是可行的，但在实践中面临挑战，要么太慢，无法实时愚弄DRL策略，要么修改存储在代理内存中的过去观察。我们证明了通用对抗摄动(UAP)，独立于应用它们的个体输入，可以有效和实时地愚弄DRL策略。我们介绍了三种利用UAP的攻击变体。通过使用三款Atari 2600游戏进行的广泛评估，我们表明我们的攻击是有效的，因为它们完全降低了三种不同DRL代理的性能(高达100%，即使在扰动上的$l_\infty$约束小到0.01)。它比图像捕获的帧速率(60赫兹)更快，也比以前的攻击($\约1.8$ms)快得多。我们的攻击技术也很有效，在线计算成本约为0.027美元毫秒。使用两个涉及机器人移动的任务，我们确认我们的结果推广到复杂的DRL任务。此外，我们还证明了已知防御措施对普遍扰动的有效性会降低。我们介绍了一种有效的技术，该技术可以检测所有已知的针对DRL策略的对抗性扰动，包括本文提出的所有通用扰动。



## **27. MixTailor: Mixed Gradient Aggregation for Robust Learning Against Tailored Attacks**

MixTailor：针对定制攻击的稳健学习的混合梯度聚合 cs.LG

To appear at the Transactions on Machine Learning Research (TMLR)

**SubmitDate**: 2022-09-23    [paper-pdf](http://arxiv.org/pdf/2207.07941v2)

**Authors**: Ali Ramezani-Kebrya, Iman Tabrizian, Fartash Faghri, Petar Popovski

**Abstracts**: Implementations of SGD on distributed systems create new vulnerabilities, which can be identified and misused by one or more adversarial agents. Recently, it has been shown that well-known Byzantine-resilient gradient aggregation schemes are indeed vulnerable to informed attackers that can tailor the attacks (Fang et al., 2020; Xie et al., 2020b). We introduce MixTailor, a scheme based on randomization of the aggregation strategies that makes it impossible for the attacker to be fully informed. Deterministic schemes can be integrated into MixTailor on the fly without introducing any additional hyperparameters. Randomization decreases the capability of a powerful adversary to tailor its attacks, while the resulting randomized aggregation scheme is still competitive in terms of performance. For both iid and non-iid settings, we establish almost sure convergence guarantees that are both stronger and more general than those available in the literature. Our empirical studies across various datasets, attacks, and settings, validate our hypothesis and show that MixTailor successfully defends when well-known Byzantine-tolerant schemes fail.

摘要: SGD在分布式系统上的实现会产生新的漏洞，这些漏洞可能会被一个或多个对抗性代理识别和滥用。最近，有研究表明，众所周知的拜占庭弹性梯度聚合方案确实容易受到可以定制攻击的知情攻击者的攻击(方等人，2020；谢等人，2020b)。我们引入了MixTailor，这是一种基于聚合策略随机化的方案，使得攻击者不可能被完全告知。确定性方案可以动态地集成到MixTailor中，而不需要引入任何额外的超参数。随机化降低了强大对手定制其攻击的能力，而由此产生的随机化聚集方案在性能方面仍然具有竞争力。对于iID和非iID设置，我们几乎肯定建立了比文献中提供的更强大和更一般的收敛保证。我们对各种数据集、攻击和环境的经验研究验证了我们的假设，并表明当众所周知的拜占庭容忍方案失败时，MixTailor成功地进行了辩护。



## **28. Reducing Exploitability with Population Based Training**

通过基于人口的培训减少可利用性 cs.LG

Presented at New Frontiers in Adversarial Machine Learning Workshop,  ICML 2022

**SubmitDate**: 2022-09-23    [paper-pdf](http://arxiv.org/pdf/2208.05083v2)

**Authors**: Pavel Czempin, Adam Gleave

**Abstracts**: Self-play reinforcement learning has achieved state-of-the-art, and often superhuman, performance in a variety of zero-sum games. Yet prior work has found that policies that are highly capable against regular opponents can fail catastrophically against adversarial policies: an opponent trained explicitly against the victim. Prior defenses using adversarial training were able to make the victim robust to a specific adversary, but the victim remained vulnerable to new ones. We conjecture this limitation was due to insufficient diversity of adversaries seen during training. We propose a defense using population based training to pit the victim against a diverse set of opponents. We evaluate this defense's robustness against new adversaries in two low-dimensional environments. Our defense increases robustness against adversaries, as measured by number of attacker training timesteps to exploit the victim. Furthermore, we show that robustness is correlated with the size of the opponent population.

摘要: 自我发挥强化学习在各种零和游戏中实现了最先进的，往往是超人的表现。然而，先前的工作已经发现，对常规对手具有高度能力的政策，可能会在对抗对手的政策上灾难性地失败：一个明确针对受害者的对手。使用对抗性训练的先前防御能够使受害者对特定的对手变得健壮，但受害者仍然容易受到新对手的攻击。我们推测，这一限制是由于训练过程中看到的对手多样性不足所致。我们建议使用基于人口的训练来防御，让受害者与不同的对手对抗。我们在两个低维环境中评估了该防御对新对手的健壮性。我们的防御提高了对抗对手的健壮性，这是通过攻击者训练时间步数来衡量的，以利用受害者。此外，我们还证明了健壮性与对手种群的大小相关。



## **29. Privacy Attacks Against Biometric Models with Fewer Samples: Incorporating the Output of Multiple Models**

针对样本较少的生物识别模型的隐私攻击：合并多个模型的输出 cs.CV

This is a major revision of a paper titled "Inverting Biometric  Models with Fewer Samples: Incorporating the Output of Multiple Models" by  the same authors that appears at IJCB 2022

**SubmitDate**: 2022-09-22    [paper-pdf](http://arxiv.org/pdf/2209.11020v1)

**Authors**: Sohaib Ahmad, Benjamin Fuller, Kaleel Mahmood

**Abstracts**: Authentication systems are vulnerable to model inversion attacks where an adversary is able to approximate the inverse of a target machine learning model. Biometric models are a prime candidate for this type of attack. This is because inverting a biometric model allows the attacker to produce a realistic biometric input to spoof biometric authentication systems.   One of the main constraints in conducting a successful model inversion attack is the amount of training data required. In this work, we focus on iris and facial biometric systems and propose a new technique that drastically reduces the amount of training data necessary. By leveraging the output of multiple models, we are able to conduct model inversion attacks with 1/10th the training set size of Ahmad and Fuller (IJCB 2020) for iris data and 1/1000th the training set size of Mai et al. (Pattern Analysis and Machine Intelligence 2019) for facial data. We denote our new attack technique as structured random with alignment loss. Our attacks are black-box, requiring no knowledge of the weights of the target neural network, only the dimension, and values of the output vector.   To show the versatility of the alignment loss, we apply our attack framework to the task of membership inference (Shokri et al., IEEE S&P 2017) on biometric data. For the iris, membership inference attack against classification networks improves from 52% to 62% accuracy.

摘要: 认证系统容易受到模型反转攻击，其中攻击者能够近似目标机器学习模型的反转。生物识别模型是此类攻击的主要候选对象。这是因为颠倒生物识别模型允许攻击者产生真实的生物识别输入来欺骗生物识别身份验证系统。进行成功的模型反转攻击的主要限制之一是所需的训练数据量。在这项工作中，我们专注于虹膜和面部生物识别系统，并提出了一种新的技术，大大减少了所需的训练数据量。通过利用多个模型的输出，我们能够以Ahmad和Fuller(IJCB 2020)对虹膜数据训练集大小的十分之一和Mai等人训练集大小的千分之一进行模型反转攻击。(模式分析和机器智能2019)，用于面部数据。我们将我们的新攻击技术表示为具有排列损失的结构化随机。我们的攻击是黑箱的，不需要知道目标神经网络的权重，只需要知道输出向量的维度和值。为了显示比对损失的多功能性，我们将我们的攻击框架应用于生物特征数据的成员关系推断任务(Shokri等人，IEEE S&P2017)。对于虹膜，针对分类网络的隶属度推理攻击将准确率从52%提高到62%。



## **30. In Differential Privacy, There is Truth: On Vote Leakage in Ensemble Private Learning**

差异隐私中有真理--论合奏私学中的选票泄露 cs.LG

To appear at NeurIPS 2022

**SubmitDate**: 2022-09-22    [paper-pdf](http://arxiv.org/pdf/2209.10732v1)

**Authors**: Jiaqi Wang, Roei Schuster, Ilia Shumailov, David Lie, Nicolas Papernot

**Abstracts**: When learning from sensitive data, care must be taken to ensure that training algorithms address privacy concerns. The canonical Private Aggregation of Teacher Ensembles, or PATE, computes output labels by aggregating the predictions of a (possibly distributed) collection of teacher models via a voting mechanism. The mechanism adds noise to attain a differential privacy guarantee with respect to the teachers' training data. In this work, we observe that this use of noise, which makes PATE predictions stochastic, enables new forms of leakage of sensitive information. For a given input, our adversary exploits this stochasticity to extract high-fidelity histograms of the votes submitted by the underlying teachers. From these histograms, the adversary can learn sensitive attributes of the input such as race, gender, or age. Although this attack does not directly violate the differential privacy guarantee, it clearly violates privacy norms and expectations, and would not be possible at all without the noise inserted to obtain differential privacy. In fact, counter-intuitively, the attack becomes easier as we add more noise to provide stronger differential privacy. We hope this encourages future work to consider privacy holistically rather than treat differential privacy as a panacea.

摘要: 在学习敏感数据时，必须注意确保训练算法解决隐私问题。教师集合的规范私有聚合，或Pate，通过投票机制聚合(可能是分布式的)教师模型集合的预测来计算输出标签。该机制增加噪声以获得关于教师训练数据的差异化隐私保证。在这项工作中，我们观察到这种使Pate预测随机化的噪声的使用，使得敏感信息的新形式泄漏成为可能。对于给定的输入，我们的对手利用这种随机性来提取潜在教师提交的选票的高保真直方图。从这些直方图中，对手可以了解输入的敏感属性，如种族、性别或年龄。虽然这一攻击并没有直接违反差异化隐私保障，但它显然违反了隐私规范和期望，如果没有插入噪声来获得差异化隐私，根本不可能。事实上，与直觉相反的是，当我们添加更多噪声以提供更强的差异隐私时，攻击变得更容易。我们希望这会鼓励未来的工作从整体上考虑隐私，而不是将差异隐私视为灵丹妙药。



## **31. Fair Robust Active Learning by Joint Inconsistency**

基于联合不一致性的公平鲁棒主动学习 cs.LG

11 pages, 3 figures

**SubmitDate**: 2022-09-22    [paper-pdf](http://arxiv.org/pdf/2209.10729v1)

**Authors**: Tsung-Han Wu, Shang-Tse Chen, Winston H. Hsu

**Abstracts**: Fair Active Learning (FAL) utilized active learning techniques to achieve high model performance with limited data and to reach fairness between sensitive groups (e.g., genders). However, the impact of the adversarial attack, which is vital for various safety-critical machine learning applications, is not yet addressed in FAL. Observing this, we introduce a novel task, Fair Robust Active Learning (FRAL), integrating conventional FAL and adversarial robustness. FRAL requires ML models to leverage active learning techniques to jointly achieve equalized performance on benign data and equalized robustness against adversarial attacks between groups. In this new task, previous FAL methods generally face the problem of unbearable computational burden and ineffectiveness. Therefore, we develop a simple yet effective FRAL strategy by Joint INconsistency (JIN). To efficiently find samples that can boost the performance and robustness of disadvantaged groups for labeling, our method exploits the prediction inconsistency between benign and adversarial samples as well as between standard and robust models. Extensive experiments under diverse datasets and sensitive groups demonstrate that our method not only achieves fairer performance on benign samples but also obtains fairer robustness under white-box PGD attacks compared with existing active learning and FAL baselines. We are optimistic that FRAL would pave a new path for developing safe and robust ML research and applications such as facial attribute recognition in biometrics systems.

摘要: 公平主动学习(FAL)利用主动学习技术在有限的数据下获得高的模型性能，并在敏感群体(例如，性别)之间达到公平。然而，FAL尚未解决对抗性攻击的影响，这对各种安全关键型机器学习应用程序至关重要。考虑到这一点，我们引入了一种新的任务，公平稳健主动学习(FRAL)，它综合了传统FAL和对手健壮性。FRAL要求ML模型利用主动学习技术，共同实现对良性数据的均衡性能和对组之间敌对攻击的均衡稳健性。在这一新的任务中，以往的FAL方法普遍面临着计算负担难以承受和效率低下的问题。因此，我们提出了一种简单而有效的联合不一致(JIN)策略。为了有效地找到能够提高弱势群体的标注性能和稳健性的样本，我们的方法利用了良性样本和敌意样本以及标准模型和稳健模型之间的预测不一致性。在不同的数据集和敏感组上的大量实验表明，与现有的主动学习和FAL基线相比，我们的方法不仅在良性样本上获得了更公平的性能，而且在白盒PGD攻击下获得了更公平的鲁棒性。我们乐观地认为，Fral将为开发安全和健壮的ML研究和应用程序铺平一条新的道路，例如生物识别系统中的面部属性识别。



## **32. Formulating Robustness Against Unforeseen Attacks**

针对不可预见的攻击形成健壮性 cs.LG

NeurIPS 2022

**SubmitDate**: 2022-09-21    [paper-pdf](http://arxiv.org/pdf/2204.13779v2)

**Authors**: Sihui Dai, Saeed Mahloujifar, Prateek Mittal

**Abstracts**: Existing defenses against adversarial examples such as adversarial training typically assume that the adversary will conform to a specific or known threat model, such as $\ell_p$ perturbations within a fixed budget. In this paper, we focus on the scenario where there is a mismatch in the threat model assumed by the defense during training, and the actual capabilities of the adversary at test time. We ask the question: if the learner trains against a specific "source" threat model, when can we expect robustness to generalize to a stronger unknown "target" threat model during test-time? Our key contribution is to formally define the problem of learning and generalization with an unforeseen adversary, which helps us reason about the increase in adversarial risk from the conventional perspective of a known adversary. Applying our framework, we derive a generalization bound which relates the generalization gap between source and target threat models to variation of the feature extractor, which measures the expected maximum difference between extracted features across a given threat model. Based on our generalization bound, we propose adversarial training with variation regularization (AT-VR) which reduces variation of the feature extractor across the source threat model during training. We empirically demonstrate that AT-VR can lead to improved generalization to unforeseen attacks during test-time compared to standard adversarial training. Additionally, we combine variation regularization with perceptual adversarial training [Laidlaw et al. 2021] to achieve state-of-the-art robustness on unforeseen attacks. Our code is publicly available at https://github.com/inspire-group/variation-regularization.

摘要: 现有的针对对抗性示例的防御，例如对抗性训练，通常假设对手将符合特定或已知的威胁模型，例如固定预算内的$\ell_p$扰动。在本文中，我们重点讨论在训练过程中防御方假设的威胁模型与测试时对手的实际能力存在不匹配的情况。我们问这样一个问题：如果学习者针对特定的“源”威胁模型进行训练，我们何时才能期望健壮性在测试期间推广到更强的未知“目标”威胁模型？我们的主要贡献是正式定义了与不可预见的对手的学习和泛化问题，这有助于我们从已知对手的传统角度来推理对手风险的增加。应用我们的框架，我们得到了一个泛化界限，它将源威胁模型和目标威胁模型之间的泛化差距与特征抽取器的变化联系起来，它度量了在给定威胁模型中提取的特征之间的期望最大差异。基于我们的泛化界，我们提出了带变异正则化的对抗性训练(AT-VR)，它减少了训练过程中特征提取子在源威胁模型上的变异。我们的经验表明，与标准的对抗性训练相比，AT-VR可以在测试时间内提高对意外攻击的泛化能力。此外，我们将变异正则化与知觉对抗训练相结合[Laidlaw等人。2021]在不可预见的攻击中实现最先进的健壮性。我们的代码在https://github.com/inspire-group/variation-regularization.上公开提供



## **33. Adversarial Formal Semantics of Attack Trees and Related Problems**

攻击树的对抗性形式语义及相关问题 cs.GT

In Proceedings GandALF 2022, arXiv:2209.09333

**SubmitDate**: 2022-09-21    [paper-pdf](http://arxiv.org/pdf/2209.10322v1)

**Authors**: Thomas Brihaye, Sophie Pinchinat, Alexandre Terefenko

**Abstracts**: Security is a subject of increasing attention in our actual society in order to protect critical resources from information disclosure, theft or damage. The informal model of attack trees introduced by Schneier, and widespread in the industry, is advocated in the 2008 NATO report to govern the evaluation of the threat in risk analysis. Attack-defense trees have since been the subject of many theoretical works addressing different formal approaches.   In 2017, M. Audinot et al. introduced a path semantics over a transition system for attack trees. Inspired by the later, we propose a two-player interpretation of the attack-tree formalism. To do so, we replace transition systems by concurrent game arenas and our associated semantics consist of strategies. We then show that the emptiness problem, known to be NP-complete for the path semantics, is now PSPACE-complete. Additionally, we show that the membership problem is coNP-complete for our two-player interpretation while it collapses to P in the path semantics.

摘要: 为了保护关键资源不受信息泄露、盗窃或损坏，安全在我们的现实社会中是一个越来越受关注的主题。由Schneier引入并在行业中广泛使用的非正式攻击树模型，在2008年北约报告中得到倡导，以管理风险分析中的威胁评估。从那时起，攻防树就成为了许多解决不同形式方法的理论著作的主题。2017年，M.Audinot等人提出。引入了攻击树转换系统上的路径语义。受后者的启发，我们提出了攻击树形式主义的两人解释。为了做到这一点，我们用并发游戏竞技场取代过渡系统，我们关联的语义由策略组成。然后，我们证明了空问题，已知的路径语义的NP完全问题，现在是PSPACE完全问题。此外，我们证明了对于我们的两人解释来说，成员资格问题是coNP-完全的，而它在路径语义中折叠为P。



## **34. Can You Still See Me?: Reconstructing Robot Operations Over End-to-End Encrypted Channels**

你还能看到我吗？：在端到端加密通道上重建机器人操作 cs.CR

13 pages, 7 figures, 9 tables, Poster presented at wisec'22

**SubmitDate**: 2022-09-21    [paper-pdf](http://arxiv.org/pdf/2205.08426v2)

**Authors**: Ryan Shah, Chuadhry Mujeeb Ahmed, Shishir Nagaraja

**Abstracts**: Connected robots play a key role in Industry 4.0, providing automation and higher efficiency for many industrial workflows. Unfortunately, these robots can leak sensitive information regarding these operational workflows to remote adversaries. While there exists mandates for the use of end-to-end encryption for data transmission in such settings, it is entirely possible for passive adversaries to fingerprint and reconstruct entire workflows being carried out -- establishing an understanding of how facilities operate. In this paper, we investigate whether a remote attacker can accurately fingerprint robot movements and ultimately reconstruct operational workflows. Using a neural network approach to traffic analysis, we find that one can predict TLS-encrypted movements with around ~60% accuracy, increasing to near-perfect accuracy under realistic network conditions. Further, we also find that attackers can reconstruct warehousing workflows with similar success. Ultimately, simply adopting best cybersecurity practices is clearly not enough to stop even weak (passive) adversaries.

摘要: 互联机器人在工业4.0中扮演着关键角色，为许多工业工作流程提供自动化和更高的效率。不幸的是，这些机器人可能会将有关这些操作工作流程的敏感信息泄露给远程对手。虽然在这种情况下有使用端到端加密进行数据传输的规定，但被动攻击者完全有可能对正在执行的整个工作流程进行指纹识别和重建--建立对设施如何运行的理解。在本文中，我们调查远程攻击者是否能够准确地识别机器人的运动并最终重建操作工作流。使用神经网络方法进行流量分析，我们发现可以预测TLS加密的移动，准确率约为60%，在现实网络条件下提高到接近完美的精度。此外，我们还发现攻击者可以成功地重构仓储工作流。归根结底，简单地采用最佳网络安全实践显然不足以阻止即使是弱小的(被动的)对手。



## **35. Fingerprinting Robot Movements via Acoustic Side Channel**

基于声学侧通道的指纹识别机器人运动 cs.CR

11 pages, 4 figures, 7 tables

**SubmitDate**: 2022-09-21    [paper-pdf](http://arxiv.org/pdf/2209.10240v1)

**Authors**: Ryan Shah, Mujeeb Ahmed, Shishir Nagaraja

**Abstracts**: In this paper, we present an acoustic side channel attack which makes use of smartphone microphones recording a robot in operation to exploit acoustic properties of the sound to fingerprint a robot's movements. In this work we consider the possibility of an insider adversary who is within physical proximity of a robotic system (such as a technician or robot operator), equipped with only their smartphone microphone. Through the acoustic side-channel, we demonstrate that it is indeed possible to fingerprint not only individual robot movements within 3D space, but also patterns of movements which could lead to inferring the purpose of the movements (i.e. surgical procedures which a surgical robot is undertaking) and hence, resulting in potential privacy violations. Upon evaluation, we find that individual robot movements can be fingerprinted with around 75% accuracy, decreasing slightly with more fine-grained movement meta-data such as distance and speed. Furthermore, workflows could be reconstructed with around 62% accuracy as a whole, with more complex movements such as pick-and-place or packing reconstructed with near perfect accuracy. As well as this, in some environments such as surgical settings, audio may be recorded and transmitted over VoIP, such as for education/teaching purposes or in remote telemedicine. The question here is, can the same attack be successful even when VoIP communication is employed, and how does packet loss impact the captured audio and the success of the attack? Using the same characteristics of acoustic sound for plain audio captured by the smartphone, the attack was 90% accurate in fingerprinting VoIP samples on average, 15% higher than the baseline without the VoIP codec employed. This opens up new research questions regarding anonymous communications to protect robotic systems from acoustic side channel attacks via VoIP communication networks.

摘要: 在本文中，我们提出了一种声学侧通道攻击，利用智能手机麦克风记录机器人运行时的声音，利用声音的声学特性来识别机器人的动作。在这项工作中，我们考虑了内部对手的可能性，他在物理上接近机器人系统(例如技术人员或机器人操作员)，只配备了他们的智能手机麦克风。通过声学侧通道，我们证明了不仅可以识别3D空间中的单个机器人运动，而且可以识别运动模式，从而推断运动的目的(即外科机器人正在进行的手术过程)，从而导致潜在的隐私侵犯。经过评估，我们发现可以对单个机器人的运动进行指纹识别，准确率约为75%，但随着距离和速度等更细粒度的运动元数据的增加，准确率略有下降。此外，可以以大约62%的整体准确率重建工作流程，以近乎完美的准确度重建更复杂的运动，如拾取和放置或打包。除此之外，在某些环境中，例如外科手术环境中，音频可以被记录并通过VoIP传输，例如用于教育/教学目的或远程远程医疗。这里的问题是，即使使用VoIP通信，同样的攻击也能成功吗？丢包对捕获的音频和攻击的成功有何影响？使用智能手机捕获的普通音频的相同声学特征，攻击平均对VoIP样本进行指纹识别的准确率为90%，比没有使用VoIP编解码器的基线高出15%。这开启了有关匿名通信的新的研究问题，以保护机器人系统免受通过VoIP通信网络的声学侧信道攻击。



## **36. Reconstructing Robot Operations via Radio-Frequency Side-Channel**

基于射频旁路的机器人作业重构 cs.CR

10 pages, 7 figures, 4 tables

**SubmitDate**: 2022-09-21    [paper-pdf](http://arxiv.org/pdf/2209.10179v1)

**Authors**: Ryan Shah, Mujeeb Ahmed, Shishir Nagaraja

**Abstracts**: Connected teleoperated robotic systems play a key role in ensuring operational workflows are carried out with high levels of accuracy and low margins of error. In recent years, a variety of attacks have been proposed that actively target the robot itself from the cyber domain. However, little attention has been paid to the capabilities of a passive attacker. In this work, we investigate whether an insider adversary can accurately fingerprint robot movements and operational warehousing workflows via the radio frequency side channel in a stealthy manner. Using an SVM for classification, we found that an adversary can fingerprint individual robot movements with at least 96% accuracy, increasing to near perfect accuracy when reconstructing entire warehousing workflows.

摘要: 联网的遥控机器人系统在确保业务工作流程以高精度和低误差水平执行方面发挥着关键作用。近年来，已经提出了各种从网络领域主动针对机器人本身的攻击。然而，被动攻击者的能力却鲜有人关注。在这项工作中，我们调查了内部攻击者是否能够通过射频侧通道以隐蔽的方式准确地识别机器人的移动和操作仓储工作流。使用支持向量机进行分类，我们发现对手可以识别单个机器人的运动，准确率至少为96%，在重建整个仓储工作流时，准确率提高到接近完美的水平。



## **37. Audit and Improve Robustness of Private Neural Networks on Encrypted Data**

私有神经网络对加密数据的审计及健壮性改进 cs.LG

10 pages, 10 figures

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2209.09996v1)

**Authors**: Jiaqi Xue, Lei Xu, Lin Chen, Weidong Shi, Kaidi Xu, Qian Lou

**Abstracts**: Performing neural network inference on encrypted data without decryption is one popular method to enable privacy-preserving neural networks (PNet) as a service. Compared with regular neural networks deployed for machine-learning-as-a-service, PNet requires additional encoding, e.g., quantized-precision numbers, and polynomial activation. Encrypted input also introduces novel challenges such as adversarial robustness and security. To the best of our knowledge, we are the first to study questions including (i) Whether PNet is more robust against adversarial inputs than regular neural networks? (ii) How to design a robust PNet given the encrypted input without decryption? We propose PNet-Attack to generate black-box adversarial examples that can successfully attack PNet in both target and untarget manners. The attack results show that PNet robustness against adversarial inputs needs to be improved. This is not a trivial task because the PNet model owner does not have access to the plaintext of the input values, which prevents the application of existing detection and defense methods such as input tuning, model normalization, and adversarial training. To tackle this challenge, we propose a new fast and accurate noise insertion method, called RPNet, to design Robust and Private Neural Networks. Our comprehensive experiments show that PNet-Attack reduces at least $2.5\times$ queries than prior works. We theoretically analyze our RPNet methods and demonstrate that RPNet can decrease $\sim 91.88\%$ attack success rate.

摘要: 在不解密的情况下对加密数据执行神经网络推理是实现隐私保护神经网络(PNET)作为服务的一种流行方法。与用于机器学习即服务的常规神经网络相比，PNET需要额外的编码，例如量化精度的数字和多项式激活。加密输入还带来了新的挑战，如对抗性和安全性。就我们所知，我们是第一个研究问题的人，包括(I)PNET是否比常规神经网络对对手输入更健壮？(Ii)如何在输入加密而不解密的情况下设计一个健壮的PNET？我们提出了PNET-Attack来生成黑盒对抗性实例，该实例可以在目标和非目标两种方式下成功攻击PNET。攻击结果表明，PNET对敌意输入的健壮性有待提高。这不是一项微不足道的任务，因为PNET模型所有者无法访问输入值的明文，这会阻止应用现有的检测和防御方法，如输入调整、模型标准化和对抗性训练。为了应对这一挑战，我们提出了一种新的快速准确的噪声插入方法，称为RPNet，用于设计健壮的私有神经网络。我们的综合实验表明，PNET-Attack比以前的工作减少了至少2.5倍的查询数。我们从理论上分析了我们的RPNet方法，并证明了RPNet方法可以降低攻击成功率。



## **38. SoK: Decentralized Finance (DeFi) Attacks**

SOK：去中心化金融(Defi)攻击 cs.CR

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2208.13035v2)

**Authors**: Liyi Zhou, Xihan Xiong, Jens Ernstberger, Stefanos Chaliasos, Zhipeng Wang, Ye Wang, Kaihua Qin, Roger Wattenhofer, Dawn Song, Arthur Gervais

**Abstracts**: Within just four years, the blockchain-based Decentralized Finance (DeFi) ecosystem has accumulated a peak total value locked (TVL) of more than 253 billion USD. This surge in DeFi's popularity has, unfortunately, been accompanied by many impactful incidents. According to our data, users, liquidity providers, speculators, and protocol operators suffered a total loss of at least 3.24 USD from Apr 30, 2018 to Apr 30, 2022. Given the blockchain's transparency and increasing incident frequency, two questions arise: How can we systematically measure, evaluate, and compare DeFi incidents? How can we learn from past attacks to strengthen DeFi security?   In this paper, we introduce a common reference frame to systematically evaluate and compare DeFi incidents, including both attacks and accidents. We investigate 77 academic papers, 30 audit reports, and 181 real-world incidents. Our open data reveals several gaps between academia and the practitioners' community. For example, few academic papers address "price oracle attacks" and "permissonless interactions", while our data suggests that they are the two most frequent incident types (15% and 10.5% correspondingly). We also investigate potential defenses, and find that: (i) 103 (56%) of the attacks are not executed atomically, granting a rescue time frame for defenders; (ii) SoTA bytecode similarity analysis can at least detect 31 vulnerable/23 adversarial contracts; and (iii) 33 (15.3%) of the adversaries leak potentially identifiable information by interacting with centralized exchanges.

摘要: 短短四年时间，基于区块链的去中心化金融(DEFI)生态系统已经积累了超过2530亿美元的峰值总价值锁定(TVL)。不幸的是，Defi人气的飙升伴随着许多有影响力的事件。根据我们的数据，从2018年4月30日到2022年4月30日，用户、流动性提供商、投机者和协议运营商总共遭受了至少3.24美元的损失。鉴于区块链的透明度和不断增加的事件频率，出现了两个问题：我们如何系统地衡量、评估和比较Defi事件？我们如何从过去的袭击中吸取教训，以加强Defi安全？在这篇文章中，我们引入了一个通用的参照系来系统地评估和比较DEFI事件，包括攻击和事故。我们调查了77篇学术论文，30份审计报告和181起真实世界的事件。我们的公开数据揭示了学术界和从业者社区之间的几个差距。举例来说，很少有学术论文涉及“价格先知攻击”和“不允许的相互作用”，而我们的数据显示，它们是最常见的两种事件类型(分别为15%和10.5%)。我们还调查了潜在的防御措施，发现：(I)103(56%)的攻击不是自动执行的，这为防御者提供了救援时间框架；(Ii)Sota字节码相似性分析至少可以检测到31个VULNERABLE/23个对手合同；以及(Iii)33个(15.3%)的对手通过与中央交易所的交互泄露了潜在的可识别信息。



## **39. Leveraging Local Patch Differences in Multi-Object Scenes for Generative Adversarial Attacks**

利用多目标场景中局部斑块差异进行生成性对抗性攻击 cs.CV

Accepted at WACV 2023 (Round 1)

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2209.09883v1)

**Authors**: Abhishek Aich, Shasha Li, Chengyu Song, M. Salman Asif, Srikanth V. Krishnamurthy, Amit K. Roy-Chowdhury

**Abstracts**: State-of-the-art generative model-based attacks against image classifiers overwhelmingly focus on single-object (i.e., single dominant object) images. Different from such settings, we tackle a more practical problem of generating adversarial perturbations using multi-object (i.e., multiple dominant objects) images as they are representative of most real-world scenes. Our goal is to design an attack strategy that can learn from such natural scenes by leveraging the local patch differences that occur inherently in such images (e.g. difference between the local patch on the object `person' and the object `bike' in a traffic scene). Our key idea is: to misclassify an adversarial multi-object image, each local patch in the image should confuse the victim classifier. Based on this, we propose a novel generative attack (called Local Patch Difference or LPD-Attack) where a novel contrastive loss function uses the aforesaid local differences in feature space of multi-object scenes to optimize the perturbation generator. Through various experiments across diverse victim convolutional neural networks, we show that our approach outperforms baseline generative attacks with highly transferable perturbations when evaluated under different white-box and black-box settings.

摘要: 最新的基于产生式模型的针对图像分类器的攻击绝大多数集中在单一对象(即单一优势对象)图像上。与这样的设置不同，我们解决了一个更实际的问题，即使用多对象(即，多个主导对象)图像来生成对抗性扰动，因为它们代表了大多数真实世界的场景。我们的目标是设计一种攻击策略，通过利用这类图像中固有的局部斑块差异(例如，交通场景中对象‘人’和对象‘自行车’上的局部斑块之间的差异)来学习此类自然场景。我们的核心思想是：为了对对抗性多目标图像进行错误分类，图像中的每个局部块都应该混淆受害者分类器。在此基础上，我们提出了一种新的生成性攻击(称为局部补丁差异或LPD-攻击)，其中一种新的对比损失函数利用多目标场景特征空间中的上述局部差异来优化扰动生成器。通过对不同受害者卷积神经网络的实验，我们表明，在不同的白盒和黑盒设置下，我们的方法优于具有高度可转移性扰动的基线生成性攻击。



## **40. Sparse Vicious Attacks on Graph Neural Networks**

图神经网络上的稀疏恶意攻击 cs.LG

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2209.09688v1)

**Authors**: Giovanni Trappolini, Valentino Maiorca, Silvio Severino, Emanuele Rodolà, Fabrizio Silvestri, Gabriele Tolomei

**Abstracts**: Graph Neural Networks (GNNs) have proven to be successful in several predictive modeling tasks for graph-structured data.   Amongst those tasks, link prediction is one of the fundamental problems for many real-world applications, such as recommender systems.   However, GNNs are not immune to adversarial attacks, i.e., carefully crafted malicious examples that are designed to fool the predictive model.   In this work, we focus on a specific, white-box attack to GNN-based link prediction models, where a malicious node aims to appear in the list of recommended nodes for a given target victim.   To achieve this goal, the attacker node may also count on the cooperation of other existing peers that it directly controls, namely on the ability to inject a number of ``vicious'' nodes in the network.   Specifically, all these malicious nodes can add new edges or remove existing ones, thereby perturbing the original graph.   Thus, we propose SAVAGE, a novel framework and a method to mount this type of link prediction attacks.   SAVAGE formulates the adversary's goal as an optimization task, striking the balance between the effectiveness of the attack and the sparsity of malicious resources required.   Extensive experiments conducted on real-world and synthetic datasets demonstrate that adversarial attacks implemented through SAVAGE indeed achieve high attack success rate yet using a small amount of vicious nodes.   Finally, despite those attacks require full knowledge of the target model, we show that they are successfully transferable to other black-box methods for link prediction.

摘要: 图神经网络(GNN)已被证明在几个针对图结构数据的预测建模任务中是成功的。在这些任务中，链接预测是许多实际应用的基本问题之一，例如推荐系统。然而，GNN也不能幸免于敌意攻击，即精心设计的恶意示例，旨在愚弄预测模型。在这项工作中，我们专注于对基于GNN的链接预测模型的特定白盒攻击，其中恶意节点的目标是出现在给定目标受害者的推荐节点列表中。为了实现这一目标，攻击者节点还可以依靠其直接控制的其他现有对等方的合作，即向网络中注入多个“恶意”节点的能力。具体地说，所有这些恶意节点都可以添加新的边或删除现有的边，从而扰乱原始图。因此，我们提出了SAWAGE、一个新的框架和一种方法来发动这种类型的链接预测攻击。Savage将对手的目标定义为优化任务，在攻击的有效性和所需恶意资源的稀疏性之间取得平衡。在真实数据集和人工数据集上进行的大量实验表明，通过Savage实现的对抗性攻击确实取得了很高的攻击成功率，但使用了少量的恶意节点。最后，尽管这些攻击需要完全了解目标模型，但我们证明了它们可以成功地转移到其他用于链接预测的黑盒方法。



## **41. I-GWAS: Privacy-Preserving Interdependent Genome-Wide Association Studies**

I-GWAS：隐私保护、相互依赖的全基因组关联研究 q-bio.GN

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2208.08361v2)

**Authors**: Túlio Pascoal, Jérémie Decouchant, Antoine Boutet, Marcus Völp

**Abstracts**: Genome-wide Association Studies (GWASes) identify genomic variations that are statistically associated with a trait, such as a disease, in a group of individuals. Unfortunately, careless sharing of GWAS statistics might give rise to privacy attacks. Several works attempted to reconcile secure processing with privacy-preserving releases of GWASes. However, we highlight that these approaches remain vulnerable if GWASes utilize overlapping sets of individuals and genomic variations. In such conditions, we show that even when relying on state-of-the-art techniques for protecting releases, an adversary could reconstruct the genomic variations of up to 28.6% of participants, and that the released statistics of up to 92.3% of the genomic variations would enable membership inference attacks. We introduce I-GWAS, a novel framework that securely computes and releases the results of multiple possibly interdependent GWASes. I-GWAS continuously releases privacy-preserving and noise-free GWAS results as new genomes become available.

摘要: 全基因组关联研究(GWASes)确定在一组个体中与某种特征(如疾病)在统计上相关的基因组变异。不幸的是，粗心大意地分享GWAS统计数据可能会导致隐私攻击。有几部作品试图调和GWAS的安全处理和隐私保护版本之间的关系。然而，我们强调，如果GWAS利用重叠的个体集合和基因组变异，这些方法仍然容易受到攻击。在这种情况下，我们表明，即使依靠最先进的技术来保护释放，对手也可以重建高达28.6%的参与者的基因组变异，并且公布的高达92.3%的基因组变异的统计数据将使成员关系推理攻击成为可能。我们介绍了I-GWAS，这是一个新的框架，可以安全地计算和发布多个可能相互依赖的GWASs的结果。随着新基因组的出现，I-GWAS不断发布隐私保护和无噪音的GWA结果。



## **42. FrequencyLowCut Pooling -- Plug & Play against Catastrophic Overfitting**

FredencyLowCut池--针对灾难性过拟合的即插即用 cs.CV

accepted at ECCV 2022

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2204.00491v2)

**Authors**: Julia Grabinski, Steffen Jung, Janis Keuper, Margret Keuper

**Abstracts**: Over the last years, Convolutional Neural Networks (CNNs) have been the dominating neural architecture in a wide range of computer vision tasks. From an image and signal processing point of view, this success might be a bit surprising as the inherent spatial pyramid design of most CNNs is apparently violating basic signal processing laws, i.e. Sampling Theorem in their down-sampling operations. However, since poor sampling appeared not to affect model accuracy, this issue has been broadly neglected until model robustness started to receive more attention. Recent work [17] in the context of adversarial attacks and distribution shifts, showed after all, that there is a strong correlation between the vulnerability of CNNs and aliasing artifacts induced by poor down-sampling operations. This paper builds on these findings and introduces an aliasing free down-sampling operation which can easily be plugged into any CNN architecture: FrequencyLowCut pooling. Our experiments show, that in combination with simple and fast FGSM adversarial training, our hyper-parameter free operator significantly improves model robustness and avoids catastrophic overfitting.

摘要: 在过去的几年里，卷积神经网络(CNN)已经成为在广泛的计算机视觉任务中占主导地位的神经结构。从图像和信号处理的角度来看，这一成功可能有点令人惊讶，因为大多数CNN固有的空间金字塔设计显然违反了基本的信号处理定律，即下采样操作中的采样定理。然而，由于较差的采样似乎不会影响模型的精度，所以这个问题一直被广泛忽视，直到模型的稳健性开始受到更多的关注。最近的工作[17]在对抗性攻击和分布转移的背景下，毕竟表明在CNN的脆弱性和糟糕的下采样操作引起的混叠伪像之间存在很强的相关性。本文以这些发现为基础，介绍了一种无混叠的下采样操作，该操作可以很容易地插入到任何CNN架构中：FrequencyLowCut池。我们的实验表明，结合简单快速的FGSM对抗性训练，我们的超参数自由算子显著地提高了模型的稳健性，并避免了灾难性的过拟合。



## **43. GAMA: Generative Adversarial Multi-Object Scene Attacks**

GAMA：生成性对抗性多目标场景攻击 cs.CV

Accepted at NeurIPS 2022; First two authors contributed equally;  Includes Supplementary Material

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2209.09502v1)

**Authors**: Abhishek Aich, Calvin Khang-Ta, Akash Gupta, Chengyu Song, Srikanth V. Krishnamurthy, M. Salman Asif, Amit K. Roy-Chowdhury

**Abstracts**: The majority of methods for crafting adversarial attacks have focused on scenes with a single dominant object (e.g., images from ImageNet). On the other hand, natural scenes include multiple dominant objects that are semantically related. Thus, it is crucial to explore designing attack strategies that look beyond learning on single-object scenes or attack single-object victim classifiers. Due to their inherent property of strong transferability of perturbations to unknown models, this paper presents the first approach of using generative models for adversarial attacks on multi-object scenes. In order to represent the relationships between different objects in the input scene, we leverage upon the open-sourced pre-trained vision-language model CLIP (Contrastive Language-Image Pre-training), with the motivation to exploit the encoded semantics in the language space along with the visual space. We call this attack approach Generative Adversarial Multi-object scene Attacks (GAMA). GAMA demonstrates the utility of the CLIP model as an attacker's tool to train formidable perturbation generators for multi-object scenes. Using the joint image-text features to train the generator, we show that GAMA can craft potent transferable perturbations in order to fool victim classifiers in various attack settings. For example, GAMA triggers ~16% more misclassification than state-of-the-art generative approaches in black-box settings where both the classifier architecture and data distribution of the attacker are different from the victim. Our code will be made publicly available soon.

摘要: 大多数制作敌意攻击的方法都集中在具有单一主导对象的场景(例如，来自ImageNet的图像)。另一方面，自然场景包括多个语义相关的主导对象。因此，探索设计超越学习单对象场景或攻击单对象受害者分类器的攻击策略是至关重要的。由于产生式模型对未知模型具有很强的可转移性，本文首次提出了利用产生式模型进行多目标场景对抗性攻击的方法。为了表示输入场景中不同对象之间的关系，我们利用开源的预先训练的视觉语言模型剪辑(Contrastive Language-Image Pre-Training)，目的是利用语言空间和视觉空间中的编码语义。我们称这种攻击方式为生成性对抗性多对象场景攻击(GAMA)。GAMA演示了剪辑模型作为攻击者的工具的效用，以训练用于多对象场景的强大的扰动生成器。使用联合图文特征训练生成器，我们证明了GAMA能够在不同的攻击环境下制造有效的可转移扰动来愚弄受害者分类器。例如，在攻击者的分类器体系结构和数据分布都与受害者不同的黑盒环境中，GAMA触发的错误分类方法比最先进的生成性方法高出约16%。我们的代码将很快公之于众。



## **44. Learn2Weight: Parameter Adaptation against Similar-domain Adversarial Attacks**

Learn2Weight：参数自适应抵御类似领域的敌意攻击 cs.LG

Accepted in COLING 2022

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2205.07315v2)

**Authors**: Siddhartha Datta

**Abstracts**: Recent work in black-box adversarial attacks for NLP systems has attracted much attention. Prior black-box attacks assume that attackers can observe output labels from target models based on selected inputs. In this work, inspired by adversarial transferability, we propose a new type of black-box NLP adversarial attack that an attacker can choose a similar domain and transfer the adversarial examples to the target domain and cause poor performance in target model. Based on domain adaptation theory, we then propose a defensive strategy, called Learn2Weight, which trains to predict the weight adjustments for a target model in order to defend against an attack of similar-domain adversarial examples. Using Amazon multi-domain sentiment classification datasets, we empirically show that Learn2Weight is effective against the attack compared to standard black-box defense methods such as adversarial training and defensive distillation. This work contributes to the growing literature on machine learning safety.

摘要: 最近针对NLP系统的黑盒对抗攻击的研究引起了人们的极大关注。以前的黑盒攻击假设攻击者可以根据选定的输入观察目标模型的输出标签。在这项工作中，受对抗性转移的启发，我们提出了一种新的黑盒NLP对抗性攻击，攻击者可以选择一个相似的域并将对抗性实例转移到目标域，从而导致目标模型的性能较差。基于领域自适应理论，我们提出了一种防御策略，称为Learn2Weight，该策略训练预测目标模型的权重调整，以防御类似领域对手示例的攻击。使用Amazon多领域情感分类数据集，与对抗性训练和防御蒸馏等标准黑盒防御方法相比，我们的经验表明Learn2Weight对攻击是有效的。这项工作有助于不断增长的关于机器学习安全的文献。



## **45. Security and Privacy of Wireless Beacon Systems**

无线信标系统的安全与隐私 cs.CR

13 pages, 3 figures

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2107.05868v2)

**Authors**: Aldar C-F. Chan, Raymond M. H. Chung

**Abstracts**: Bluetooth Low Energy (BLE) beacons have been increasingly used in smart city applications, such as location-based and proximity-based services, to enable Internet of Things to interact with people in vicinity or enhance context-awareness. Their widespread deployment in human-centric applications makes them an attractive target to adversaries for social or economic reasons. In fact, beacons are reportedly exposed to various security issues and privacy concerns. A characterization of attacks against beacon systems is given to help understand adversary motives, required adversarial capabilities, potential impact and possible defence mechanisms for different threats, with a view to facilitating security evaluation and protection formulation for beacon systems.

摘要: 蓝牙低能耗(BLE)信标已越来越多地应用于智能城市应用中，例如基于位置和基于邻近的服务，以使物联网能够与附近的人交互或增强上下文感知。它们在以人为中心的应用程序中的广泛部署使它们成为出于社会或经济原因而吸引对手的目标。事实上，据报道，信标面临着各种安全问题和隐私问题。给出了针对信标系统的攻击的特征，以帮助理解对手的动机、所需的对抗能力、潜在的影响以及针对不同威胁的可能的防御机制，以期促进信标系统的安全评估和保护的制定。



## **46. Parallel Proof-of-Work with Concrete Bounds**

具有具体界限的并行工作证明 cs.CR

**SubmitDate**: 2022-09-19    [paper-pdf](http://arxiv.org/pdf/2204.00034v2)

**Authors**: Patrik Keller, Rainer Böhme

**Abstracts**: Authorization is challenging in distributed systems that cannot rely on the identification of nodes. Proof-of-work offers an alternative gate-keeping mechanism, but its probabilistic nature is incompatible with conventional security definitions. Recent related work establishes concrete bounds for the failure probability of Bitcoin's sequential proof-of-work mechanism. We propose a family of state replication protocols using parallel proof-of-work. Our bottom-up design from an agreement sub-protocol allows us to give concrete bounds for the failure probability in adversarial synchronous networks. After the typical interval of 10 minutes, parallel proof-of-work offers two orders of magnitude more security than sequential proof-of-work. This means that state updates can be sufficiently secure to support commits after one block (i.e., after 10 minutes), removing the risk of double-spending in many applications. We offer guidance on the optimal choice of parameters for a wide range of network and attacker assumptions. Simulations show that the proposed construction is robust against violations of design assumptions.

摘要: 在不能依赖节点标识的分布式系统中，授权是具有挑战性的。工作证明提供了一种替代的把关机制，但其概率性质与传统的安全定义不兼容。最近的相关工作为比特币的序贯验证机制的失效概率建立了具体的界限。我们提出了一类使用并行工作证明的状态复制协议。我们从协议子协议开始的自下而上的设计允许我们给出对抗性同步网络中故障概率的具体界。在典型的10分钟间隔之后，并行工作证明提供的安全性比顺序工作证明高两个数量级。这意味着状态更新可以足够安全，以支持在一个数据块(即10分钟之后)后提交，从而消除了许多应用程序中重复支出的风险。我们为各种网络和攻击者假设提供参数最佳选择的指导。仿真结果表明，所提出的结构对违反设计假设具有较强的鲁棒性。



## **47. A Transferable and Automatic Tuning of Deep Reinforcement Learning for Cost Effective Phishing Detection**

一种可移植的自动调整深度强化学习的高效网络钓鱼检测方法 cs.CR

**SubmitDate**: 2022-09-19    [paper-pdf](http://arxiv.org/pdf/2209.09033v1)

**Authors**: Orel Lavie, Asaf Shabtai, Gilad Katz

**Abstracts**: Many challenging real-world problems require the deployment of ensembles multiple complementary learning models to reach acceptable performance levels. While effective, applying the entire ensemble to every sample is costly and often unnecessary. Deep Reinforcement Learning (DRL) offers a cost-effective alternative, where detectors are dynamically chosen based on the output of their predecessors, with their usefulness weighted against their computational cost. Despite their potential, DRL-based solutions are not widely used in this capacity, partly due to the difficulties in configuring the reward function for each new task, the unpredictable reactions of the DRL agent to changes in the data, and the inability to use common performance metrics (e.g., TPR/FPR) to guide the algorithm's performance. In this study we propose methods for fine-tuning and calibrating DRL-based policies so that they can meet multiple performance goals. Moreover, we present a method for transferring effective security policies from one dataset to another. Finally, we demonstrate that our approach is highly robust against adversarial attacks.

摘要: 许多具有挑战性的现实世界问题需要部署集成多个互补的学习模型以达到可接受的性能水平。虽然有效，但将整个整体应用于每个样本都是昂贵的，而且往往没有必要。深度强化学习(DRL)提供了一种经济有效的替代方法，其中检测器是根据其前辈的输出动态选择的，其有用性与其计算成本相权衡。尽管它们具有潜力，但基于DRL的解决方案在该能力中没有被广泛使用，部分原因是为每个新任务配置奖励函数的困难、DRL代理对数据变化的不可预测的反应、以及不能使用公共性能度量(例如，TPR/FPR)来指导算法的性能。在这项研究中，我们提出了微调和校准基于DRL的策略的方法，以使它们能够满足多个性能目标。此外，我们还提出了一种将有效的安全策略从一个数据集转移到另一个数据集的方法。最后，我们证明了我们的方法对敌意攻击具有很强的健壮性。



## **48. Encrypted Semantic Communication Using Adversarial Training for Privacy Preserving**

使用对抗性训练进行隐私保护的加密语义通信 cs.IT

submitted to IEEE Wireless Communications Letters

**SubmitDate**: 2022-09-19    [paper-pdf](http://arxiv.org/pdf/2209.09008v1)

**Authors**: Xinlai Luo, Zhiyong Chen, Meixia Tao, Feng Yang

**Abstracts**: Semantic communication is implemented based on shared background knowledge, but the sharing mechanism risks privacy leakage. In this letter, we propose an encrypted semantic communication system (ESCS) for privacy preserving, which combines universality and confidentiality. The universality is reflected in that all network modules of the proposed ESCS are trained based on a shared database, which is suitable for large-scale deployment in practical scenarios. Meanwhile, the confidentiality is achieved by symmetric encryption. Based on the adversarial training, we design an adversarial encryption training scheme to guarantee the accuracy of semantic communication in both encrypted and unencrypted modes. Experiment results show that the proposed ESCS with the adversarial encryption training scheme can perform well regardless of whether the semantic information is encrypted. It is difficult for the attacker to reconstruct the original semantic information from the eavesdropped message.

摘要: 语义交流是基于共享的背景知识实现的，但这种共享机制存在隐私泄露的风险。在这封信中，我们提出了一个用于隐私保护的加密语义通信系统(ESCS)，它结合了通用性和保密性。普适性体现在，拟议的ESCS的所有网络模块都是基于共享数据库进行训练的，适合在实际场景中大规模部署。同时，通过对称加密实现了保密性。在对抗性训练的基础上，设计了对抗性加密训练方案，保证了加密和非加密模式下语义通信的准确性。实验结果表明，无论语义信息是否加密，采用对抗性加密训练方案的ESCS都能获得较好的性能。攻击者很难从被窃听的消息中重构原始的语义信息。



## **49. Catoptric Light can be Dangerous: Effective Physical-World Attack by Natural Phenomenon**

反射光可能是危险的：自然现象对物理世界的有效攻击 cs.CV

arXiv admin note: substantial text overlap with arXiv:2209.09652,  arXiv:2209.02430

**SubmitDate**: 2022-09-19    [paper-pdf](http://arxiv.org/pdf/2209.11739v1)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstracts**: Deep neural networks (DNNs) have achieved great success in many tasks. Therefore, it is crucial to evaluate the robustness of advanced DNNs. The traditional methods use stickers as physical perturbations to fool the classifiers, which is difficult to achieve stealthiness and there exists printing loss. Some new types of physical attacks use light beam to perform attacks (e.g., laser, projector), whose optical patterns are artificial rather than natural. In this work, we study a new type of physical attack, called adversarial catoptric light (AdvCL), in which adversarial perturbations are generated by common natural phenomena, catoptric light, to achieve stealthy and naturalistic adversarial attacks against advanced DNNs in physical environments. Carefully designed experiments demonstrate the effectiveness of the proposed method in simulated and real-world environments. The attack success rate is 94.90% in a subset of ImageNet and 83.50% in the real-world environment. We also discuss some of AdvCL's transferability and defense strategy against this attack.

摘要: 深度神经网络(DNN)在许多领域都取得了巨大的成功。因此，对高级DNN的健壮性进行评估是至关重要的。传统的方法使用贴纸作为物理扰动来愚弄分类器，这不仅难以实现隐蔽性，而且存在印刷损失。一些新类型的物理攻击使用光束来执行攻击(例如，激光、投影仪)，其光学图案是人工的而不是自然的。在这项工作中，我们研究了一种新型的物理攻击，称为对抗反射光(AdvCL)，其中对抗扰动是由常见的自然现象反射光产生的，以实现对物理环境中高级DNN的隐身和自然主义的对抗攻击。精心设计的实验证明了该方法在模拟和真实环境中的有效性。在ImageNet子集上的攻击成功率为94.90%，在真实环境下的攻击成功率为83.50%。我们还讨论了AdvCL的一些可转移性和针对这种攻击的防御策略。



## **50. Adversarial Color Projection: A Projector-Based Physical Attack to DNNs**

对抗性颜色投影：一种基于投影器的对DNN的物理攻击 cs.CR

arXiv admin note: substantial text overlap with arXiv:2209.02430

**SubmitDate**: 2022-09-19    [paper-pdf](http://arxiv.org/pdf/2209.09652v1)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstracts**: Recent advances have shown that deep neural networks (DNNs) are susceptible to adversarial perturbations. Therefore, it is necessary to evaluate the robustness of advanced DNNs using adversarial attacks. However, traditional physical attacks that use stickers as perturbations are more vulnerable than recent light-based physical attacks. In this work, we propose a projector-based physical attack called adversarial color projection (AdvCP), which performs an adversarial attack by manipulating the physical parameters of the projected light. Experiments show the effectiveness of our method in both digital and physical environments. The experimental results demonstrate that the proposed method has excellent attack transferability, which endows AdvCP with effective blackbox attack. We prospect AdvCP threats to future vision-based systems and applications and propose some ideas for light-based physical attacks.

摘要: 最近的研究表明，深度神经网络(DNN)容易受到对抗性扰动的影响。因此，有必要对使用对抗性攻击的高级DNN的健壮性进行评估。然而，使用贴纸作为扰动的传统物理攻击比最近基于光线的物理攻击更容易受到攻击。在这项工作中，我们提出了一种基于投影仪的物理攻击，称为对抗性颜色投影(AdvCP)，它通过操纵投射光的物理参数来执行对抗性攻击。实验表明，我们的方法在数字和物理环境中都是有效的。实验结果表明，该方法具有良好的攻击可转移性，使AdvCP具有有效的黑盒攻击能力。我们展望了AdvCP对未来基于视觉的系统和应用的威胁，并提出了一些基于光的物理攻击的想法。



