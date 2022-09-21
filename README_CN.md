# Latest Adversarial Attack Papers
**update at 2022-09-22 06:31:33**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Leveraging Local Patch Differences in Multi-Object Scenes for Generative Adversarial Attacks**

利用多目标场景中局部斑块差异进行生成性对抗性攻击 cs.CV

Accepted at WACV 2023 (Round 1)

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2209.09883v1)

**Authors**: Abhishek Aich, Shasha Li, Chengyu Song, M. Salman Asif, Srikanth V. Krishnamurthy, Amit K. Roy-Chowdhury

**Abstracts**: State-of-the-art generative model-based attacks against image classifiers overwhelmingly focus on single-object (i.e., single dominant object) images. Different from such settings, we tackle a more practical problem of generating adversarial perturbations using multi-object (i.e., multiple dominant objects) images as they are representative of most real-world scenes. Our goal is to design an attack strategy that can learn from such natural scenes by leveraging the local patch differences that occur inherently in such images (e.g. difference between the local patch on the object `person' and the object `bike' in a traffic scene). Our key idea is: to misclassify an adversarial multi-object image, each local patch in the image should confuse the victim classifier. Based on this, we propose a novel generative attack (called Local Patch Difference or LPD-Attack) where a novel contrastive loss function uses the aforesaid local differences in feature space of multi-object scenes to optimize the perturbation generator. Through various experiments across diverse victim convolutional neural networks, we show that our approach outperforms baseline generative attacks with highly transferable perturbations when evaluated under different white-box and black-box settings.

摘要: 最新的基于产生式模型的针对图像分类器的攻击绝大多数集中在单一对象(即单一优势对象)图像上。与这样的设置不同，我们解决了一个更实际的问题，即使用多对象(即，多个主导对象)图像来生成对抗性扰动，因为它们代表了大多数真实世界的场景。我们的目标是设计一种攻击策略，通过利用这类图像中固有的局部斑块差异(例如，交通场景中对象‘人’和对象‘自行车’上的局部斑块之间的差异)来学习此类自然场景。我们的核心思想是：为了对对抗性多目标图像进行错误分类，图像中的每个局部块都应该混淆受害者分类器。在此基础上，我们提出了一种新的生成性攻击(称为局部补丁差异或LPD-攻击)，其中一种新的对比损失函数利用多目标场景特征空间中的上述局部差异来优化扰动生成器。通过对不同受害者卷积神经网络的实验，我们表明，在不同的白盒和黑盒设置下，我们的方法优于具有高度可转移性扰动的基线生成性攻击。



## **2. Sparse Vicious Attacks on Graph Neural Networks**

图神经网络上的稀疏恶意攻击 cs.LG

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2209.09688v1)

**Authors**: Giovanni Trappolini, Valentino Maiorca, Silvio Severino, Emanuele Rodolà, Fabrizio Silvestri, Gabriele Tolomei

**Abstracts**: Graph Neural Networks (GNNs) have proven to be successful in several predictive modeling tasks for graph-structured data.   Amongst those tasks, link prediction is one of the fundamental problems for many real-world applications, such as recommender systems.   However, GNNs are not immune to adversarial attacks, i.e., carefully crafted malicious examples that are designed to fool the predictive model.   In this work, we focus on a specific, white-box attack to GNN-based link prediction models, where a malicious node aims to appear in the list of recommended nodes for a given target victim.   To achieve this goal, the attacker node may also count on the cooperation of other existing peers that it directly controls, namely on the ability to inject a number of ``vicious'' nodes in the network.   Specifically, all these malicious nodes can add new edges or remove existing ones, thereby perturbing the original graph.   Thus, we propose SAVAGE, a novel framework and a method to mount this type of link prediction attacks.   SAVAGE formulates the adversary's goal as an optimization task, striking the balance between the effectiveness of the attack and the sparsity of malicious resources required.   Extensive experiments conducted on real-world and synthetic datasets demonstrate that adversarial attacks implemented through SAVAGE indeed achieve high attack success rate yet using a small amount of vicious nodes.   Finally, despite those attacks require full knowledge of the target model, we show that they are successfully transferable to other black-box methods for link prediction.

摘要: 图神经网络(GNN)已被证明在几个针对图结构数据的预测建模任务中是成功的。在这些任务中，链接预测是许多实际应用的基本问题之一，例如推荐系统。然而，GNN也不能幸免于敌意攻击，即精心设计的恶意示例，旨在愚弄预测模型。在这项工作中，我们专注于对基于GNN的链接预测模型的特定白盒攻击，其中恶意节点的目标是出现在给定目标受害者的推荐节点列表中。为了实现这一目标，攻击者节点还可以依靠其直接控制的其他现有对等方的合作，即向网络中注入多个“恶意”节点的能力。具体地说，所有这些恶意节点都可以添加新的边或删除现有的边，从而扰乱原始图。因此，我们提出了SAWAGE、一个新的框架和一种方法来发动这种类型的链接预测攻击。Savage将对手的目标定义为优化任务，在攻击的有效性和所需恶意资源的稀疏性之间取得平衡。在真实数据集和人工数据集上进行的大量实验表明，通过Savage实现的对抗性攻击确实取得了很高的攻击成功率，但使用了少量的恶意节点。最后，尽管这些攻击需要完全了解目标模型，但我们证明了它们可以成功地转移到其他用于链接预测的黑盒方法。



## **3. Understanding Real-world Threats to Deep Learning Models in Android Apps**

了解Android应用程序中深度学习模型面临的现实威胁 cs.CR

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2209.09577v1)

**Authors**: Zizhuang Deng, Kai Chen, Guozhu Meng, Xiaodong Zhang, Ke Xu, Yao Cheng

**Abstracts**: Famous for its superior performance, deep learning (DL) has been popularly used within many applications, which also at the same time attracts various threats to the models. One primary threat is from adversarial attacks. Researchers have intensively studied this threat for several years and proposed dozens of approaches to create adversarial examples (AEs). But most of the approaches are only evaluated on limited models and datasets (e.g., MNIST, CIFAR-10). Thus, the effectiveness of attacking real-world DL models is not quite clear. In this paper, we perform the first systematic study of adversarial attacks on real-world DNN models and provide a real-world model dataset named RWM. Particularly, we design a suite of approaches to adapt current AE generation algorithms to the diverse real-world DL models, including automatically extracting DL models from Android apps, capturing the inputs and outputs of the DL models in apps, generating AEs and validating them by observing the apps' execution. For black-box DL models, we design a semantic-based approach to build suitable datasets and use them for training substitute models when performing transfer-based attacks. After analyzing 245 DL models collected from 62,583 real-world apps, we have a unique opportunity to understand the gap between real-world DL models and contemporary AE generation algorithms. To our surprise, the current AE generation algorithms can only directly attack 6.53% of the models. Benefiting from our approach, the success rate upgrades to 47.35%.

摘要: 深度学习以其优越的性能而著称，在众多应用中得到了广泛的应用，但同时也给模型带来了各种威胁。其中一个主要威胁来自对抗性攻击。几年来，研究人员对这种威胁进行了深入的研究，并提出了数十种创建对抗性例子(AE)的方法。但大多数方法只在有限的模型和数据集(例如MNIST、CIFAR-10)上进行评估。因此，攻击真实世界的数字图书馆模型的有效性还不是很清楚。在本文中，我们首次对真实世界DNN模型的对抗性攻击进行了系统的研究，并提供了一个真实世界模型数据集RWM。特别是，我们设计了一套方法来使现有的AE生成算法适应不同的真实DL模型，包括自动从Android应用程序中提取DL模型，捕获应用程序中DL模型的输入和输出，生成AE并通过观察应用程序的执行来验证它们。对于黑盒DL模型，我们设计了一种基于语义的方法来构建合适的数据集，并在执行基于传输的攻击时使用它们来训练替代模型。在分析了从62,583个现实世界应用程序中收集的245个DL模型之后，我们有了一个独特的机会来了解现实世界DL模型和当代AE生成算法之间的差距。令我们惊讶的是，目前的AE生成算法只能直接攻击6.53%的模型。受益于我们的方法，成功率提升到47.35%。



## **4. I-GWAS: Privacy-Preserving Interdependent Genome-Wide Association Studies**

I-GWAS：隐私保护、相互依赖的全基因组关联研究 q-bio.GN

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2208.08361v2)

**Authors**: Túlio Pascoal, Jérémie Decouchant, Antoine Boutet, Marcus Völp

**Abstracts**: Genome-wide Association Studies (GWASes) identify genomic variations that are statistically associated with a trait, such as a disease, in a group of individuals. Unfortunately, careless sharing of GWAS statistics might give rise to privacy attacks. Several works attempted to reconcile secure processing with privacy-preserving releases of GWASes. However, we highlight that these approaches remain vulnerable if GWASes utilize overlapping sets of individuals and genomic variations. In such conditions, we show that even when relying on state-of-the-art techniques for protecting releases, an adversary could reconstruct the genomic variations of up to 28.6% of participants, and that the released statistics of up to 92.3% of the genomic variations would enable membership inference attacks. We introduce I-GWAS, a novel framework that securely computes and releases the results of multiple possibly interdependent GWASes. I-GWAS continuously releases privacy-preserving and noise-free GWAS results as new genomes become available.

摘要: 全基因组关联研究(GWASes)确定在一组个体中与某种特征(如疾病)在统计上相关的基因组变异。不幸的是，粗心大意地分享GWAS统计数据可能会导致隐私攻击。有几部作品试图调和GWAS的安全处理和隐私保护版本之间的关系。然而，我们强调，如果GWAS利用重叠的个体集合和基因组变异，这些方法仍然容易受到攻击。在这种情况下，我们表明，即使依靠最先进的技术来保护释放，对手也可以重建高达28.6%的参与者的基因组变异，并且公布的高达92.3%的基因组变异的统计数据将使成员关系推理攻击成为可能。我们介绍了I-GWAS，这是一个新的框架，可以安全地计算和发布多个可能相互依赖的GWASs的结果。随着新基因组的出现，I-GWAS不断发布隐私保护和无噪音的GWA结果。



## **5. FrequencyLowCut Pooling -- Plug & Play against Catastrophic Overfitting**

FredencyLowCut池--针对灾难性过拟合的即插即用 cs.CV

accepted at ECCV 2022

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2204.00491v2)

**Authors**: Julia Grabinski, Steffen Jung, Janis Keuper, Margret Keuper

**Abstracts**: Over the last years, Convolutional Neural Networks (CNNs) have been the dominating neural architecture in a wide range of computer vision tasks. From an image and signal processing point of view, this success might be a bit surprising as the inherent spatial pyramid design of most CNNs is apparently violating basic signal processing laws, i.e. Sampling Theorem in their down-sampling operations. However, since poor sampling appeared not to affect model accuracy, this issue has been broadly neglected until model robustness started to receive more attention. Recent work [17] in the context of adversarial attacks and distribution shifts, showed after all, that there is a strong correlation between the vulnerability of CNNs and aliasing artifacts induced by poor down-sampling operations. This paper builds on these findings and introduces an aliasing free down-sampling operation which can easily be plugged into any CNN architecture: FrequencyLowCut pooling. Our experiments show, that in combination with simple and fast FGSM adversarial training, our hyper-parameter free operator significantly improves model robustness and avoids catastrophic overfitting.

摘要: 在过去的几年里，卷积神经网络(CNN)已经成为在广泛的计算机视觉任务中占主导地位的神经结构。从图像和信号处理的角度来看，这一成功可能有点令人惊讶，因为大多数CNN固有的空间金字塔设计显然违反了基本的信号处理定律，即下采样操作中的采样定理。然而，由于较差的采样似乎不会影响模型的精度，所以这个问题一直被广泛忽视，直到模型的稳健性开始受到更多的关注。最近的工作[17]在对抗性攻击和分布转移的背景下，毕竟表明在CNN的脆弱性和糟糕的下采样操作引起的混叠伪像之间存在很强的相关性。本文以这些发现为基础，介绍了一种无混叠的下采样操作，该操作可以很容易地插入到任何CNN架构中：FrequencyLowCut池。我们的实验表明，结合简单快速的FGSM对抗性训练，我们的超参数自由算子显著地提高了模型的稳健性，并避免了灾难性的过拟合。



## **6. GAMA: Generative Adversarial Multi-Object Scene Attacks**

GAMA：生成性对抗性多目标场景攻击 cs.CV

Accepted at NeurIPS 2022; First two authors contributed equally;  Includes Supplementary Material

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2209.09502v1)

**Authors**: Abhishek Aich, Calvin Khang-Ta, Akash Gupta, Chengyu Song, Srikanth V. Krishnamurthy, M. Salman Asif, Amit K. Roy-Chowdhury

**Abstracts**: The majority of methods for crafting adversarial attacks have focused on scenes with a single dominant object (e.g., images from ImageNet). On the other hand, natural scenes include multiple dominant objects that are semantically related. Thus, it is crucial to explore designing attack strategies that look beyond learning on single-object scenes or attack single-object victim classifiers. Due to their inherent property of strong transferability of perturbations to unknown models, this paper presents the first approach of using generative models for adversarial attacks on multi-object scenes. In order to represent the relationships between different objects in the input scene, we leverage upon the open-sourced pre-trained vision-language model CLIP (Contrastive Language-Image Pre-training), with the motivation to exploit the encoded semantics in the language space along with the visual space. We call this attack approach Generative Adversarial Multi-object scene Attacks (GAMA). GAMA demonstrates the utility of the CLIP model as an attacker's tool to train formidable perturbation generators for multi-object scenes. Using the joint image-text features to train the generator, we show that GAMA can craft potent transferable perturbations in order to fool victim classifiers in various attack settings. For example, GAMA triggers ~16% more misclassification than state-of-the-art generative approaches in black-box settings where both the classifier architecture and data distribution of the attacker are different from the victim. Our code will be made publicly available soon.

摘要: 大多数制作敌意攻击的方法都集中在具有单一主导对象的场景(例如，来自ImageNet的图像)。另一方面，自然场景包括多个语义相关的主导对象。因此，探索设计超越学习单对象场景或攻击单对象受害者分类器的攻击策略是至关重要的。由于产生式模型对未知模型具有很强的可转移性，本文首次提出了利用产生式模型进行多目标场景对抗性攻击的方法。为了表示输入场景中不同对象之间的关系，我们利用开源的预先训练的视觉语言模型剪辑(Contrastive Language-Image Pre-Training)，目的是利用语言空间和视觉空间中的编码语义。我们称这种攻击方式为生成性对抗性多对象场景攻击(GAMA)。GAMA演示了剪辑模型作为攻击者的工具的效用，以训练用于多对象场景的强大的扰动生成器。使用联合图文特征训练生成器，我们证明了GAMA能够在不同的攻击环境下制造有效的可转移扰动来愚弄受害者分类器。例如，在攻击者的分类器体系结构和数据分布都与受害者不同的黑盒环境中，GAMA触发的错误分类方法比最先进的生成性方法高出约16%。我们的代码将很快公之于众。



## **7. Learn2Weight: Parameter Adaptation against Similar-domain Adversarial Attacks**

Learn2Weight：参数自适应抵御类似领域的敌意攻击 cs.LG

Accepted in COLING 2022

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2205.07315v2)

**Authors**: Siddhartha Datta

**Abstracts**: Recent work in black-box adversarial attacks for NLP systems has attracted much attention. Prior black-box attacks assume that attackers can observe output labels from target models based on selected inputs. In this work, inspired by adversarial transferability, we propose a new type of black-box NLP adversarial attack that an attacker can choose a similar domain and transfer the adversarial examples to the target domain and cause poor performance in target model. Based on domain adaptation theory, we then propose a defensive strategy, called Learn2Weight, which trains to predict the weight adjustments for a target model in order to defend against an attack of similar-domain adversarial examples. Using Amazon multi-domain sentiment classification datasets, we empirically show that Learn2Weight is effective against the attack compared to standard black-box defense methods such as adversarial training and defensive distillation. This work contributes to the growing literature on machine learning safety.

摘要: 最近针对NLP系统的黑盒对抗攻击的研究引起了人们的极大关注。以前的黑盒攻击假设攻击者可以根据选定的输入观察目标模型的输出标签。在这项工作中，受对抗性转移的启发，我们提出了一种新的黑盒NLP对抗性攻击，攻击者可以选择一个相似的域并将对抗性实例转移到目标域，从而导致目标模型的性能较差。基于领域自适应理论，我们提出了一种防御策略，称为Learn2Weight，该策略训练预测目标模型的权重调整，以防御类似领域对手示例的攻击。使用Amazon多领域情感分类数据集，与对抗性训练和防御蒸馏等标准黑盒防御方法相比，我们的经验表明Learn2Weight对攻击是有效的。这项工作有助于不断增长的关于机器学习安全的文献。



## **8. Security and Privacy of Wireless Beacon Systems**

无线信标系统的安全与隐私 cs.CR

13 pages, 3 figures

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2107.05868v2)

**Authors**: Aldar C-F. Chan, Raymond M. H. Chung

**Abstracts**: Bluetooth Low Energy (BLE) beacons have been increasingly used in smart city applications, such as location-based and proximity-based services, to enable Internet of Things to interact with people in vicinity or enhance context-awareness. Their widespread deployment in human-centric applications makes them an attractive target to adversaries for social or economic reasons. In fact, beacons are reportedly exposed to various security issues and privacy concerns. A characterization of attacks against beacon systems is given to help understand adversary motives, required adversarial capabilities, potential impact and possible defence mechanisms for different threats, with a view to facilitating security evaluation and protection formulation for beacon systems.

摘要: 蓝牙低能耗(BLE)信标已越来越多地应用于智能城市应用中，例如基于位置和基于邻近的服务，以使物联网能够与附近的人交互或增强上下文感知。它们在以人为中心的应用程序中的广泛部署使它们成为出于社会或经济原因而吸引对手的目标。事实上，据报道，信标面临着各种安全问题和隐私问题。给出了针对信标系统的攻击的特征，以帮助理解对手的动机、所需的对抗能力、潜在的影响以及针对不同威胁的可能的防御机制，以期促进信标系统的安全评估和保护的制定。



## **9. Parallel Proof-of-Work with Concrete Bounds**

具有具体界限的并行工作证明 cs.CR

**SubmitDate**: 2022-09-19    [paper-pdf](http://arxiv.org/pdf/2204.00034v2)

**Authors**: Patrik Keller, Rainer Böhme

**Abstracts**: Authorization is challenging in distributed systems that cannot rely on the identification of nodes. Proof-of-work offers an alternative gate-keeping mechanism, but its probabilistic nature is incompatible with conventional security definitions. Recent related work establishes concrete bounds for the failure probability of Bitcoin's sequential proof-of-work mechanism. We propose a family of state replication protocols using parallel proof-of-work. Our bottom-up design from an agreement sub-protocol allows us to give concrete bounds for the failure probability in adversarial synchronous networks. After the typical interval of 10 minutes, parallel proof-of-work offers two orders of magnitude more security than sequential proof-of-work. This means that state updates can be sufficiently secure to support commits after one block (i.e., after 10 minutes), removing the risk of double-spending in many applications. We offer guidance on the optimal choice of parameters for a wide range of network and attacker assumptions. Simulations show that the proposed construction is robust against violations of design assumptions.

摘要: 在不能依赖节点标识的分布式系统中，授权是具有挑战性的。工作证明提供了一种替代的把关机制，但其概率性质与传统的安全定义不兼容。最近的相关工作为比特币的序贯验证机制的失效概率建立了具体的界限。我们提出了一类使用并行工作证明的状态复制协议。我们从协议子协议开始的自下而上的设计允许我们给出对抗性同步网络中故障概率的具体界。在典型的10分钟间隔之后，并行工作证明提供的安全性比顺序工作证明高两个数量级。这意味着状态更新可以足够安全，以支持在一个数据块(即10分钟之后)后提交，从而消除了许多应用程序中重复支出的风险。我们为各种网络和攻击者假设提供参数最佳选择的指导。仿真结果表明，所提出的结构对违反设计假设具有较强的鲁棒性。



## **10. A Transferable and Automatic Tuning of Deep Reinforcement Learning for Cost Effective Phishing Detection**

一种可移植的自动调整深度强化学习的高效网络钓鱼检测方法 cs.CR

**SubmitDate**: 2022-09-19    [paper-pdf](http://arxiv.org/pdf/2209.09033v1)

**Authors**: Orel Lavie, Asaf Shabtai, Gilad Katz

**Abstracts**: Many challenging real-world problems require the deployment of ensembles multiple complementary learning models to reach acceptable performance levels. While effective, applying the entire ensemble to every sample is costly and often unnecessary. Deep Reinforcement Learning (DRL) offers a cost-effective alternative, where detectors are dynamically chosen based on the output of their predecessors, with their usefulness weighted against their computational cost. Despite their potential, DRL-based solutions are not widely used in this capacity, partly due to the difficulties in configuring the reward function for each new task, the unpredictable reactions of the DRL agent to changes in the data, and the inability to use common performance metrics (e.g., TPR/FPR) to guide the algorithm's performance. In this study we propose methods for fine-tuning and calibrating DRL-based policies so that they can meet multiple performance goals. Moreover, we present a method for transferring effective security policies from one dataset to another. Finally, we demonstrate that our approach is highly robust against adversarial attacks.

摘要: 许多具有挑战性的现实世界问题需要部署集成多个互补的学习模型以达到可接受的性能水平。虽然有效，但将整个整体应用于每个样本都是昂贵的，而且往往没有必要。深度强化学习(DRL)提供了一种经济有效的替代方法，其中检测器是根据其前辈的输出动态选择的，其有用性与其计算成本相权衡。尽管它们具有潜力，但基于DRL的解决方案在该能力中没有被广泛使用，部分原因是为每个新任务配置奖励函数的困难、DRL代理对数据变化的不可预测的反应、以及不能使用公共性能度量(例如，TPR/FPR)来指导算法的性能。在这项研究中，我们提出了微调和校准基于DRL的策略的方法，以使它们能够满足多个性能目标。此外，我们还提出了一种将有效的安全策略从一个数据集转移到另一个数据集的方法。最后，我们证明了我们的方法对敌意攻击具有很强的健壮性。



## **11. Encrypted Semantic Communication Using Adversarial Training for Privacy Preserving**

使用对抗性训练进行隐私保护的加密语义通信 cs.IT

submitted to IEEE Wireless Communications Letters

**SubmitDate**: 2022-09-19    [paper-pdf](http://arxiv.org/pdf/2209.09008v1)

**Authors**: Xinlai Luo, Zhiyong Chen, Meixia Tao, Feng Yang

**Abstracts**: Semantic communication is implemented based on shared background knowledge, but the sharing mechanism risks privacy leakage. In this letter, we propose an encrypted semantic communication system (ESCS) for privacy preserving, which combines universality and confidentiality. The universality is reflected in that all network modules of the proposed ESCS are trained based on a shared database, which is suitable for large-scale deployment in practical scenarios. Meanwhile, the confidentiality is achieved by symmetric encryption. Based on the adversarial training, we design an adversarial encryption training scheme to guarantee the accuracy of semantic communication in both encrypted and unencrypted modes. Experiment results show that the proposed ESCS with the adversarial encryption training scheme can perform well regardless of whether the semantic information is encrypted. It is difficult for the attacker to reconstruct the original semantic information from the eavesdropped message.

摘要: 语义交流是基于共享的背景知识实现的，但这种共享机制存在隐私泄露的风险。在这封信中，我们提出了一个用于隐私保护的加密语义通信系统(ESCS)，它结合了通用性和保密性。普适性体现在，拟议的ESCS的所有网络模块都是基于共享数据库进行训练的，适合在实际场景中大规模部署。同时，通过对称加密实现了保密性。在对抗性训练的基础上，设计了对抗性加密训练方案，保证了加密和非加密模式下语义通信的准确性。实验结果表明，无论语义信息是否加密，采用对抗性加密训练方案的ESCS都能获得较好的性能。攻击者很难从被窃听的消息中重构原始的语义信息。



## **12. Adversarial Color Projection: A Projector-Based Physical Attack to DNNs**

对抗性颜色投影：一种基于投影器的对DNN的物理攻击 cs.CR

arXiv admin note: substantial text overlap with arXiv:2209.02430

**SubmitDate**: 2022-09-19    [paper-pdf](http://arxiv.org/pdf/2209.09652v1)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstracts**: Recent advances have shown that deep neural networks (DNNs) are susceptible to adversarial perturbations. Therefore, it is necessary to evaluate the robustness of advanced DNNs using adversarial attacks. However, traditional physical attacks that use stickers as perturbations are more vulnerable than recent light-based physical attacks. In this work, we propose a projector-based physical attack called adversarial color projection (AdvCP), which performs an adversarial attack by manipulating the physical parameters of the projected light. Experiments show the effectiveness of our method in both digital and physical environments. The experimental results demonstrate that the proposed method has excellent attack transferability, which endows AdvCP with effective blackbox attack. We prospect AdvCP threats to future vision-based systems and applications and propose some ideas for light-based physical attacks.

摘要: 最近的研究表明，深度神经网络(DNN)容易受到对抗性扰动的影响。因此，有必要对使用对抗性攻击的高级DNN的健壮性进行评估。然而，使用贴纸作为扰动的传统物理攻击比最近基于光线的物理攻击更容易受到攻击。在这项工作中，我们提出了一种基于投影仪的物理攻击，称为对抗性颜色投影(AdvCP)，它通过操纵投射光的物理参数来执行对抗性攻击。实验表明，我们的方法在数字和物理环境中都是有效的。实验结果表明，该方法具有良好的攻击可转移性，使AdvCP具有有效的黑盒攻击能力。我们展望了AdvCP对未来基于视觉的系统和应用的威胁，并提出了一些基于光的物理攻击的想法。



## **13. A Systematic Evaluation of Node Embedding Robustness**

节点嵌入健壮性的系统评估 cs.LG

**SubmitDate**: 2022-09-19    [paper-pdf](http://arxiv.org/pdf/2209.08064v2)

**Authors**: Alexandru Mara, Jefrey Lijffijt, Stephan Günnemann, Tijl De Bie

**Abstracts**: Node embedding methods map network nodes to low dimensional vectors that can be subsequently used in a variety of downstream prediction tasks. The popularity of these methods has significantly increased in recent years, yet, their robustness to perturbations of the input data is still poorly understood. In this paper, we assess the empirical robustness of node embedding models to random and adversarial poisoning attacks. Our systematic evaluation covers representative embedding methods based on Skip-Gram, matrix factorization, and deep neural networks. We compare edge addition, deletion and rewiring strategies computed using network properties as well as node labels. We also investigate the effect of label homophily and heterophily on robustness. We report qualitative results via embedding visualization and quantitative results in terms of downstream node classification and network reconstruction performances. We found that node classification suffers from higher performance degradation as opposed to network reconstruction, and that degree-based and label-based attacks are on average the most damaging.

摘要: 节点嵌入方法将网络节点映射到可随后用于各种下行预测任务的低维向量。近年来，这些方法的普及率显著提高，然而，人们对它们对输入数据扰动的稳健性仍然知之甚少。在本文中，我们评估了节点嵌入模型对随机和对抗性中毒攻击的经验稳健性。我们的系统评价涵盖了基于Skip-Gram的典型嵌入方法、矩阵分解和深度神经网络。我们比较了使用网络属性和节点标签计算的边添加、删除和重新布线策略。我们还研究了标签的同质性和异质性对稳健性的影响。我们通过嵌入可视化和定量结果来报告下游节点分类和网络重构性能方面的定性结果。我们发现，与网络重建相比，节点分类遭受了更高的性能降级，基于度和基于标签的攻击平均破坏性最大。



## **14. Indicators of Attack Failure: Debugging and Improving Optimization of Adversarial Examples**

攻击失败的指标：对抗性实例的调试和改进优化 cs.LG

Accepted at NeurIPS 2022

**SubmitDate**: 2022-09-19    [paper-pdf](http://arxiv.org/pdf/2106.09947v2)

**Authors**: Maura Pintor, Luca Demetrio, Angelo Sotgiu, Ambra Demontis, Nicholas Carlini, Battista Biggio, Fabio Roli

**Abstracts**: Evaluating robustness of machine-learning models to adversarial examples is a challenging problem. Many defenses have been shown to provide a false sense of robustness by causing gradient-based attacks to fail, and they have been broken under more rigorous evaluations. Although guidelines and best practices have been suggested to improve current adversarial robustness evaluations, the lack of automatic testing and debugging tools makes it difficult to apply these recommendations in a systematic manner. In this work, we overcome these limitations by: (i) categorizing attack failures based on how they affect the optimization of gradient-based attacks, while also unveiling two novel failures affecting many popular attack implementations and past evaluations; (ii) proposing six novel indicators of failure, to automatically detect the presence of such failures in the attack optimization process; and (iii) suggesting a systematic protocol to apply the corresponding fixes. Our extensive experimental analysis, involving more than 15 models in 3 distinct application domains, shows that our indicators of failure can be used to debug and improve current adversarial robustness evaluations, thereby providing a first concrete step towards automatizing and systematizing them. Our open-source code is available at: https://github.com/pralab/IndicatorsOfAttackFailure.

摘要: 评估机器学习模型对对抗性样本的稳健性是一个具有挑战性的问题。事实证明，许多防御措施通过导致基于梯度的攻击失败来提供一种错误的健壮感，这些防御措施已经在更严格的评估下被打破。虽然有人建议采用准则和最佳做法来改进目前的对抗性评估，但由于缺乏自动测试和调试工具，很难系统地适用这些建议。在这项工作中，我们克服了这些局限性：(I)根据攻击失败如何影响基于梯度的攻击的优化进行分类，同时也揭示了影响许多流行攻击实现和过去评估的两个新失败；(Ii)提出了六个新的失败指示器，以自动检测攻击优化过程中此类失败的存在；以及(Iii)提出了一个系统的协议来应用相应的修复。我们广泛的实验分析，涉及3个不同应用领域的15个模型，表明我们的失败指示器可以用于调试和改进当前的对手健壮性评估，从而为实现自动化和系统化迈出了具体的第一步。我们的开源代码可以在https://github.com/pralab/IndicatorsOfAttackFailure.上找到



## **15. Evaluating Machine Unlearning via Epistemic Uncertainty**

基于认知不确定性的机器遗忘评估 cs.LG

Rejected at ECML 2021. Even though the paper was rejected, we want to  "publish" it on arxiv, since we believe that it is nevertheless interesting  to investigate the connections between unlearning and uncertainty v2: Added  acknowledgment and code repository

**SubmitDate**: 2022-09-19    [paper-pdf](http://arxiv.org/pdf/2208.10836v2)

**Authors**: Alexander Becker, Thomas Liebig

**Abstracts**: There has been a growing interest in Machine Unlearning recently, primarily due to legal requirements such as the General Data Protection Regulation (GDPR) and the California Consumer Privacy Act. Thus, multiple approaches were presented to remove the influence of specific target data points from a trained model. However, when evaluating the success of unlearning, current approaches either use adversarial attacks or compare their results to the optimal solution, which usually incorporates retraining from scratch. We argue that both ways are insufficient in practice. In this work, we present an evaluation metric for Machine Unlearning algorithms based on epistemic uncertainty. This is the first definition of a general evaluation metric for Machine Unlearning to our best knowledge.

摘要: 最近，人们对机器遗忘的兴趣与日俱增，主要是因为法律要求，如一般数据保护法规(GDPR)和加州消费者隐私法。因此，人们提出了多种方法来消除特定目标数据点对训练模型的影响。然而，在评估遗忘的成功时，目前的方法要么使用对抗性攻击，要么将结果与最优解决方案进行比较，后者通常包括从头开始的再培训。我们认为，这两种方式在实践中都是不够的。在这项工作中，我们提出了一种基于认知不确定性的机器遗忘算法评价指标。据我们所知，这是对机器遗忘的一般评估指标的第一次定义。



## **16. AdvDO: Realistic Adversarial Attacks for Trajectory Prediction**

AdvDO：弹道预测的现实对抗性攻击 cs.LG

To appear in ECCV 2022

**SubmitDate**: 2022-09-19    [paper-pdf](http://arxiv.org/pdf/2209.08744v1)

**Authors**: Yulong Cao, Chaowei Xiao, Anima Anandkumar, Danfei Xu, Marco Pavone

**Abstracts**: Trajectory prediction is essential for autonomous vehicles (AVs) to plan correct and safe driving behaviors. While many prior works aim to achieve higher prediction accuracy, few study the adversarial robustness of their methods. To bridge this gap, we propose to study the adversarial robustness of data-driven trajectory prediction systems. We devise an optimization-based adversarial attack framework that leverages a carefully-designed differentiable dynamic model to generate realistic adversarial trajectories. Empirically, we benchmark the adversarial robustness of state-of-the-art prediction models and show that our attack increases the prediction error for both general metrics and planning-aware metrics by more than 50% and 37%. We also show that our attack can lead an AV to drive off road or collide into other vehicles in simulation. Finally, we demonstrate how to mitigate the adversarial attacks using an adversarial training scheme.

摘要: 轨迹预测对于自动驾驶车辆规划正确、安全的驾驶行为至关重要。虽然许多前人的工作都是为了达到更高的预测精度，但很少有人研究他们方法的对抗性稳健性。为了弥补这一差距，我们建议研究数据驱动的弹道预测系统的对抗健壮性。我们设计了一个基于优化的对抗性攻击框架，该框架利用精心设计的可微动态模型来生成真实的对抗性轨迹。在实验上，我们对最先进的预测模型的对手健壮性进行了基准测试，结果表明，我们的攻击使一般指标和规划感知指标的预测误差分别增加了50%和37%以上。我们还在仿真中证明了我们的攻击可以导致无人机偏离道路或与其他车辆相撞。最后，我们演示了如何使用对抗性训练方案来缓解对抗性攻击。



## **17. On the Adversarial Transferability of ConvMixer Models**

关于ConvMixer模型的对抗性转移 cs.LG

5 pages, 5 figures, 5 tables. arXiv admin note: substantial text  overlap with arXiv:2209.02997

**SubmitDate**: 2022-09-19    [paper-pdf](http://arxiv.org/pdf/2209.08724v1)

**Authors**: Ryota Iijima, Miki Tanaka, Isao Echizen, Hitoshi Kiya

**Abstracts**: Deep neural networks (DNNs) are well known to be vulnerable to adversarial examples (AEs). In addition, AEs have adversarial transferability, which means AEs generated for a source model can fool another black-box model (target model) with a non-trivial probability. In this paper, we investigate the property of adversarial transferability between models including ConvMixer, which is an isotropic network, for the first time. To objectively verify the property of transferability, the robustness of models is evaluated by using a benchmark attack method called AutoAttack. In an image classification experiment, ConvMixer is confirmed to be weak to adversarial transferability.

摘要: 众所周知，深度神经网络(DNN)很容易受到敌意例子(AEs)的攻击。此外，AEs具有对抗性，这意味着为一个源模型生成的AEs可以以非平凡的概率愚弄另一个黑盒模型(目标模型)。本文首次研究了包含各向同性网络的ConvMixer模型之间的对抗性转移性质。为了客观地验证模型的可转移性，使用一种称为AutoAttack的基准攻击方法对模型的稳健性进行了评估。在图像分类实验中，证实了ConvMixer对攻击的可转移性较弱。



## **18. Reinforcement learning-based optimised control for tracking of nonlinear systems with adversarial attacks**

基于强化学习的对抗性非线性系统跟踪优化控制 eess.SY

Submitted for The 10th RSI International Conference on Robotics and  Mechatronics (ICRoM 2022)

**SubmitDate**: 2022-09-18    [paper-pdf](http://arxiv.org/pdf/2209.02165v2)

**Authors**: Farshad Rahimi, Sepideh Ziaei

**Abstracts**: This paper introduces a reinforcement learning-based tracking control approach for a class of nonlinear systems using neural networks. In this approach, adversarial attacks were considered both in the actuator and on the outputs. This approach incorporates a simultaneous tracking and optimization process. It is necessary to be able to solve the Hamilton-Jacobi-Bellman equation (HJB) in order to obtain optimal control input, but this is difficult due to the strong nonlinearity terms in the equation. In order to find the solution to the HJB equation, we used a reinforcement learning approach. In this online adaptive learning approach, three neural networks are simultaneously adapted: the critic neural network, the actor neural network, and the adversary neural network. Ultimately, simulation results are presented to demonstrate the effectiveness of the introduced method on a manipulator.

摘要: 针对一类神经网络非线性系统，提出了一种基于强化学习的跟踪控制方法。在这种方法中，在执行器和输出端都考虑了对抗性攻击。这种方法结合了同步跟踪和优化过程。为了获得最优控制输入，必须能解Hamilton-Jacobi-Bellman方程(HJB)，但由于方程中的强非线性项，这是很困难的。为了找到HJB方程的解，我们使用了强化学习方法。在这种在线自适应学习方法中，同时自适应了三个神经网络：批评者神经网络、行动者神经网络和对手神经网络。最后，以机械手为例，给出了仿真结果，验证了该方法的有效性。



## **19. Distribution inference risks: Identifying and mitigating sources of leakage**

分配推断风险：识别和减少泄漏的来源 cs.CR

14 pages, 8 figures

**SubmitDate**: 2022-09-18    [paper-pdf](http://arxiv.org/pdf/2209.08541v1)

**Authors**: Valentin Hartmann, Léo Meynent, Maxime Peyrard, Dimitrios Dimitriadis, Shruti Tople, Robert West

**Abstracts**: A large body of work shows that machine learning (ML) models can leak sensitive or confidential information about their training data. Recently, leakage due to distribution inference (or property inference) attacks is gaining attention. In this attack, the goal of an adversary is to infer distributional information about the training data. So far, research on distribution inference has focused on demonstrating successful attacks, with little attention given to identifying the potential causes of the leakage and to proposing mitigations. To bridge this gap, as our main contribution, we theoretically and empirically analyze the sources of information leakage that allows an adversary to perpetrate distribution inference attacks. We identify three sources of leakage: (1) memorizing specific information about the $\mathbb{E}[Y|X]$ (expected label given the feature values) of interest to the adversary, (2) wrong inductive bias of the model, and (3) finiteness of the training data. Next, based on our analysis, we propose principled mitigation techniques against distribution inference attacks. Specifically, we demonstrate that causal learning techniques are more resilient to a particular type of distribution inference risk termed distributional membership inference than associative learning methods. And lastly, we present a formalization of distribution inference that allows for reasoning about more general adversaries than was previously possible.

摘要: 大量工作表明，机器学习(ML)模型可能会泄露有关其训练数据的敏感或机密信息。近年来，分布推理(或属性推理)攻击引起的信息泄漏问题日益引起人们的关注。在这种攻击中，对手的目标是推断有关训练数据的分布信息。到目前为止，对分布推断的研究主要集中在展示成功的攻击上，很少关注识别泄漏的潜在原因和提出缓解措施。为了弥合这一差距，作为我们的主要贡献，我们从理论和经验上分析了允许对手实施分布式推理攻击的信息泄漏来源。我们发现泄漏的三个来源：(1)记忆关于对手感兴趣的$\mathbb{E}[Y|X]$(给定特征值的期望标签)的特定信息，(2)模型的错误归纳偏差，以及(3)训练数据的有限性。其次，在分析的基础上，提出了针对分布式推理攻击的原则性缓解技术。具体地说，我们证明了因果学习技术比联想学习方法对一种特定类型的分布推理风险更具弹性，这种分布推理被称为分布成员关系推理。最后，我们提出了一种分布推理的形式化，允许对比以前可能的更一般的对手进行推理。



## **20. pFedDef: Defending Grey-Box Attacks for Personalized Federated Learning**

PFedDef：个性化联合学习防御灰盒攻击 cs.LG

16 pages, 5 figures (11 images if counting sub-figures separately),  longer version of paper submitted to CrossFL 2022 poster workshop, code  available at (https://github.com/tj-kim/pFedDef_v1)

**SubmitDate**: 2022-09-17    [paper-pdf](http://arxiv.org/pdf/2209.08412v1)

**Authors**: Taejin Kim, Shubhranshu Singh, Nikhil Madaan, Carlee Joe-Wong

**Abstracts**: Personalized federated learning allows for clients in a distributed system to train a neural network tailored to their unique local data while leveraging information at other clients. However, clients' models are vulnerable to attacks during both the training and testing phases. In this paper we address the issue of adversarial clients crafting evasion attacks at test time to deceive other clients. For example, adversaries may aim to deceive spam filters and recommendation systems trained with personalized federated learning for monetary gain. The adversarial clients have varying degrees of personalization based on the method of distributed learning, leading to a "grey-box" situation. We are the first to characterize the transferability of such internal evasion attacks for different learning methods and analyze the trade-off between model accuracy and robustness depending on the degree of personalization and similarities in client data. We introduce a defense mechanism, pFedDef, that performs personalized federated adversarial training while respecting resource limitations at clients that inhibit adversarial training. Overall, pFedDef increases relative grey-box adversarial robustness by 62% compared to federated adversarial training and performs well even under limited system resources.

摘要: 个性化联合学习允许分布式系统中的客户端训练针对其独特的本地数据量身定做的神经网络，同时利用其他客户端的信息。然而，客户的模型在培训和测试阶段都容易受到攻击。在本文中，我们讨论了敌意客户在测试时精心设计逃避攻击以欺骗其他客户的问题。例如，对手的目标可能是欺骗经过个性化联合学习培训的垃圾邮件过滤器和推荐系统，以换取金钱利益。基于分布式学习的方法，敌方客户具有不同程度的个性化，导致了灰箱的情况。我们首次针对不同的学习方法刻画了这种内部规避攻击的可转移性，并根据客户数据的个性化程度和相似性分析了模型精度和稳健性之间的权衡。我们引入了一种防御机制pFedDef，它执行个性化的联合对抗训练，同时尊重客户端阻碍对抗训练的资源限制。总体而言，与联合对抗训练相比，pFedDef将相对灰箱对抗健壮性提高了62%，即使在有限的系统资源下也能很好地执行。



## **21. Decentralization Paradox: A Study of Hegemonic and Risky ERC-20 Tokens**

去中心化悖论：霸权性和风险性ERC-20代币研究 cs.CR

2022 Engineering Graduate Research Symposium (EGRS)

**SubmitDate**: 2022-09-17    [paper-pdf](http://arxiv.org/pdf/2209.08370v1)

**Authors**: Nikolay Ivanov, Qiben Yan

**Abstracts**: In this work, we explore the class of Ethereum smart contracts called the administrated ERC20 tokens. We demonstrate that these contracts are more owner-controlled and less safe than the services they try to disrupt, such as banks and centralized online payment systems. We develop a binary classifier for identification of administrated ERC20 tokens, and conduct extensive data analysis, which reveals that nearly 9 out of 10 ERC20 tokens on Ethereum are administrated, and thereby unsafe to engage with even under the assumption of trust towards their owners. We design and implement SafelyAdministrated - a Solidity abstract class that safeguards users of administrated ERC20 tokens from adversarial attacks or frivolous behavior of the tokens' owners.

摘要: 在这项工作中，我们探索了一类被称为管理ERC20令牌的以太智能合约。我们证明，与它们试图扰乱的服务(如银行和集中式在线支付系统)相比，这些合同更多的是由所有者控制，而不是那么安全。我们开发了一个二进制分类器来识别被管理的ERC20令牌，并进行了广泛的数据分析，结果表明，在Etherum上，近10个ERC20令牌中有9个是被管理的，因此即使在信任其所有者的假设下，参与也是不安全的。我们设计并实现了SafelyAdminated-一个可靠的抽象类，它保护受管理的ERC20令牌的用户免受令牌所有者的敌意攻击或轻率行为。



## **22. Robust Online and Distributed Mean Estimation Under Adversarial Data Corruption**

对抗数据腐败下的稳健在线和分布均值估计 cs.CR

8 pages, 5 figures, 61st IEEE Conference on Decision and Control  (CDC)

**SubmitDate**: 2022-09-17    [paper-pdf](http://arxiv.org/pdf/2209.09624v1)

**Authors**: Tong Yao, Shreyas Sundaram

**Abstracts**: We study robust mean estimation in an online and distributed scenario in the presence of adversarial data attacks. At each time step, each agent in a network receives a potentially corrupted data point, where the data points were originally independent and identically distributed samples of a random variable. We propose online and distributed algorithms for all agents to asymptotically estimate the mean. We provide the error-bound and the convergence properties of the estimates to the true mean under our algorithms. Based on the network topology, we further evaluate each agent's trade-off in convergence rate between incorporating data from neighbors and learning with only local observations.

摘要: 我们研究了在线和分布式场景中存在敌意数据攻击时的稳健均值估计。在每个时间步长，网络中的每个代理都会收到一个可能被破坏的数据点，其中这些数据点最初是随机变量的独立且相同分布的样本。我们提出了在线和分布式算法，使所有代理都能渐近估计平均值。在我们的算法下，我们给出了估计到真均值的误差界和收敛性质。基于网络拓扑结构，我们进一步评估了每个代理在合并邻居数据和仅使用本地观测进行学习的收敛速度方面的权衡。



## **23. Replay-based Recovery for Autonomous Robotic Vehicles from Sensor Deception Attacks**

基于重放的自主机器人对传感器欺骗攻击的恢复 cs.RO

**SubmitDate**: 2022-09-17    [paper-pdf](http://arxiv.org/pdf/2209.04554v3)

**Authors**: Pritam Dash, Guanpeng Li, Mehdi Karimibiuki, Karthik Pattabiraman

**Abstracts**: Sensors are crucial for autonomous operation in robotic vehicles (RV). Physical attacks on sensors such as sensor tampering or spoofing can feed erroneous values to RVs through physical channels, which results in mission failures. In this paper, we present DeLorean, a comprehensive diagnosis and recovery framework for securing autonomous RVs from physical attacks. We consider a strong form of physical attack called sensor deception attacks (SDAs), in which the adversary targets multiple sensors of different types simultaneously (even including all sensors). Under SDAs, DeLorean inspects the attack induced errors, identifies the targeted sensors, and prevents the erroneous sensor inputs from being used in RV's feedback control loop. DeLorean replays historic state information in the feedback control loop and recovers the RV from attacks. Our evaluation on four real and two simulated RVs shows that DeLorean can recover RVs from different attacks, and ensure mission success in 94% of the cases (on average), without any crashes. DeLorean incurs low performance, memory and battery overheads.

摘要: 传感器对于机器人车辆(RV)的自主操作至关重要。对传感器的物理攻击，如传感器篡改或欺骗，可能会通过物理通道向房车提供错误的值，从而导致任务失败。在本文中，我们提出了DeLorean，一个全面的诊断和恢复框架，用于保护自主房车免受物理攻击。我们考虑了一种称为传感器欺骗攻击(SDA)的强物理攻击形式，在这种攻击中，对手同时针对不同类型的多个传感器(甚至包括所有传感器)。在SDAS下，DeLorean检查攻击导致的错误，识别目标传感器，并防止错误的传感器输入用于房车的反馈控制回路。DeLorean在反馈控制环路中重放历史状态信息，并恢复RV免受攻击。我们对四辆真实房车和两辆模拟房车的评估表明，DeLorean可以从不同的攻击中恢复房车，并确保94%的任务成功(平均而言)，而不会发生任何崩溃。DeLorean的性能、内存和电池开销都很低。



## **24. Resilient Risk based Adaptive Authentication and Authorization (RAD-AA) Framework**

基于弹性风险的自适应身份验证和授权(RAD-AA)框架 cs.CR

**SubmitDate**: 2022-09-17    [paper-pdf](http://arxiv.org/pdf/2208.02592v2)

**Authors**: Jaimandeep Singh, Chintan Patel, Naveen Kumar Chaudhary

**Abstracts**: In recent cyber attacks, credential theft has emerged as one of the primary vectors of gaining entry into the system. Once attacker(s) have a foothold in the system, they use various techniques including token manipulation to elevate the privileges and access protected resources. This makes authentication and token based authorization a critical component for a secure and resilient cyber system. In this paper we discuss the design considerations for such a secure and resilient authentication and authorization framework capable of self-adapting based on the risk scores and trust profiles. We compare this design with the existing standards such as OAuth 2.0, OpenID Connect and SAML 2.0. We then study popular threat models such as STRIDE and PASTA and summarize the resilience of the proposed architecture against common and relevant threat vectors. We call this framework as Resilient Risk based Adaptive Authentication and Authorization (RAD-AA). The proposed framework excessively increases the cost for an adversary to launch and sustain any cyber attack and provides much-needed strength to critical infrastructure. We also discuss the machine learning (ML) approach for the adaptive engine to accurately classify transactions and arrive at risk scores.

摘要: 在最近的网络攻击中，凭据盗窃已成为进入系统的主要载体之一。一旦攻击者在系统中站稳脚跟，他们就会使用包括令牌操作在内的各种技术来提升权限并访问受保护的资源。这使得身份验证和基于令牌的授权成为安全和有弹性的网络系统的关键组件。在本文中，我们讨论了这样一个安全的、具有弹性的认证和授权框架的设计考虑，该框架能够基于风险分数和信任配置文件自适应。我们将该设计与OAuth 2.0、OpenID Connect和SAML 2.0等现有标准进行了比较。然后，我们研究了流行的威胁模型，如STRIDE和PASA，并总结了所提出的体系结构对常见和相关威胁向量的恢复能力。我们将此框架称为基于弹性风险的自适应身份验证和授权(RAD-AA)。拟议的框架过度增加了对手发动和维持任何网络攻击的成本，并为关键基础设施提供了亟需的力量。我们还讨论了机器学习(ML)方法，使自适应引擎能够准确地对交易进行分类，并得出风险分数。



## **25. Robust Prototypical Few-Shot Organ Segmentation with Regularized Neural-ODEs**

基于正则化神经节点的典型少发器官分割 cs.CV

**SubmitDate**: 2022-09-17    [paper-pdf](http://arxiv.org/pdf/2208.12428v2)

**Authors**: Prashant Pandey, Mustafa Chasmai, Tanuj Sur, Brejesh Lall

**Abstracts**: Despite the tremendous progress made by deep learning models in image semantic segmentation, they typically require large annotated examples, and increasing attention is being diverted to problem settings like Few-Shot Learning (FSL) where only a small amount of annotation is needed for generalisation to novel classes. This is especially seen in medical domains where dense pixel-level annotations are expensive to obtain. In this paper, we propose Regularized Prototypical Neural Ordinary Differential Equation (R-PNODE), a method that leverages intrinsic properties of Neural-ODEs, assisted and enhanced by additional cluster and consistency losses to perform Few-Shot Segmentation (FSS) of organs. R-PNODE constrains support and query features from the same classes to lie closer in the representation space thereby improving the performance over the existing Convolutional Neural Network (CNN) based FSS methods. We further demonstrate that while many existing Deep CNN based methods tend to be extremely vulnerable to adversarial attacks, R-PNODE exhibits increased adversarial robustness for a wide array of these attacks. We experiment with three publicly available multi-organ segmentation datasets in both in-domain and cross-domain FSS settings to demonstrate the efficacy of our method. In addition, we perform experiments with seven commonly used adversarial attacks in various settings to demonstrate R-PNODE's robustness. R-PNODE outperforms the baselines for FSS by significant margins and also shows superior performance for a wide array of attacks varying in intensity and design.

摘要: 尽管深度学习模型在图像语义分割方面取得了巨大的进步，但它们通常需要大量的注释示例，并且越来越多的注意力被转移到像少镜头学习(FSL)这样的问题环境中，其中只需要少量的注释就可以概括到新的类。这在医学领域中尤其常见，在医学领域中，密集像素级注释的获取成本很高。在本文中，我们提出了正则化的原型神经常微分方程(R-PNODE)，该方法利用神经节点的固有特性，通过额外的聚类和一致性损失来辅助和增强器官的少镜头分割(FSS)。R-PNODE约束支持和查询来自同一类的特征在表示空间中更接近，从而提高了现有基于卷积神经网络(CNN)的FSS方法的性能。我们进一步证明，虽然许多现有的基于Deep CNN的方法往往非常容易受到对抗性攻击，但R-PNODE对一系列此类攻击表现出更强的对抗性。我们用三个公开可用的多器官分割数据集在域内和跨域的FSS环境中进行了实验，以证明我们方法的有效性。此外，我们在不同的环境下对七种常用的对抗性攻击进行了实验，以验证R-PNODE的健壮性。R-PNODE的表现远远超过FSS的基线，并在各种强度和设计的攻击中显示出卓越的性能。



## **26. Secure and Efficient Query Processing in Outsourced Databases**

外包数据库中安全高效的查询处理 cs.CR

Ph.D. thesis

**SubmitDate**: 2022-09-17    [paper-pdf](http://arxiv.org/pdf/2206.10753v2)

**Authors**: Dmytro Bogatov

**Abstracts**: Various cryptographic techniques are used in outsourced database systems to ensure data privacy while allowing for efficient querying. This work proposes a definition and components of a new secure and efficient outsourced database system, which answers various types of queries, with different privacy guarantees in different security models. This work starts with the survey of five order-revealing encryption schemes that can be used directly in many database indices and five range query protocols with various security / efficiency tradeoffs. The survey systematizes the state-of-the-art range query solutions in a snapshot adversary setting and offers some non-obvious observations regarding the efficiency of the constructions. In $\mathcal{E}\text{psolute}$, a secure range query engine, security is achieved in a setting with a much stronger adversary where she can continuously observe everything on the server, and leaking even the result size can enable a reconstruction attack. $\mathcal{E}\text{psolute}$ proposes a definition, construction, analysis, and experimental evaluation of a system that provably hides both access pattern and communication volume while remaining efficient. The work concludes with $k\text{-a}n\text{o}n$ -- a secure similarity search engine in a snapshot adversary model. The work presents a construction in which the security of $k\text{NN}$ queries is achieved similarly to OPE / ORE solutions -- encrypting the input with an approximate Distance Comparison Preserving Encryption scheme so that the inputs, the points in a hyperspace, are perturbed, but the query algorithm still produces accurate results. We use TREC datasets and queries for the search, and track the rank quality metrics such as MRR and nDCG. For the attacks, we build an LSTM model that trains on the correlation between a sentence and its embedding and then predicts words from the embedding.

摘要: 在外包数据库系统中使用了各种加密技术，以确保数据隐私，同时允许高效查询。提出了一种新的安全高效的外包数据库系统的定义和组成，该系统可以回答不同类型的查询，在不同的安全模型下具有不同的隐私保障。这项工作首先调查了五种可以直接用于许多数据库索引的顺序揭示加密方案和五种具有各种安全/效率权衡的范围查询协议。该调查将快照对手环境中最先进的范围查询解决方案系统化，并提供了一些关于构造效率的不明显的观察。在安全范围查询引擎$\Mathcal{E}\Text{psolte}$中，安全是在一个更强大的对手的设置下实现的，在这种设置中，她可以连续观察服务器上的一切，即使泄漏结果大小也可能导致重建攻击。$\Mathcal{E}\Text{psolte}$提出了一种系统的定义、构造、分析和实验评估，该系统可以证明在保持效率的同时隐藏了访问模式和通信量。这项工作以$k\Text{-a}n\Text{o}n$结束--快照对手模型中的安全相似性搜索引擎。该工作提出了一种结构，其中$k\Text{NN}$查询的安全性类似于OPE/ORE解决方案--使用一种保持近似距离比较的加密方案对输入进行加密，使得输入，即超空间中的点，被扰动，但查询算法仍然产生准确的结果。我们使用TREC数据集和查询进行搜索，并跟踪排名质量度量，如MRR和nDCG。对于攻击，我们建立了一个LSTM模型，该模型根据句子与其嵌入之间的相关性进行训练，然后从嵌入中预测单词。



## **27. Robust Ensemble Morph Detection with Domain Generalization**

基于区域泛化的稳健集成形态检测 cs.CV

Accepted in IJCB 2022

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2209.08130v1)

**Authors**: Hossein Kashiani, Shoaib Meraj Sami, Sobhan Soleymani, Nasser M. Nasrabadi

**Abstracts**: Although a substantial amount of studies is dedicated to morph detection, most of them fail to generalize for morph faces outside of their training paradigm. Moreover, recent morph detection methods are highly vulnerable to adversarial attacks. In this paper, we intend to learn a morph detection model with high generalization to a wide range of morphing attacks and high robustness against different adversarial attacks. To this aim, we develop an ensemble of convolutional neural networks (CNNs) and Transformer models to benefit from their capabilities simultaneously. To improve the robust accuracy of the ensemble model, we employ multi-perturbation adversarial training and generate adversarial examples with high transferability for several single models. Our exhaustive evaluations demonstrate that the proposed robust ensemble model generalizes to several morphing attacks and face datasets. In addition, we validate that our robust ensemble model gain better robustness against several adversarial attacks while outperforming the state-of-the-art studies.

摘要: 虽然有大量的研究致力于变形检测，但大多数研究都没有对训练范例之外的变形人脸进行泛化。此外，最近的变形检测方法非常容易受到对手攻击。在本文中，我们打算学习一种对广泛的变形攻击具有高度泛化能力的变形检测模型，并对不同的敌意攻击具有较强的鲁棒性。为此，我们开发了卷积神经网络(CNN)和变压器模型的集成，以同时受益于它们的能力。为了提高集成模型的稳健精度，我们使用多扰动对抗性训练，并为多个单一模型生成具有高可转移性的对抗性实例。我们的详尽评估表明，所提出的稳健集成模型适用于多个变形攻击和人脸数据集。此外，我们还验证了我们的稳健集成模型在抵抗几个对手攻击时获得了更好的稳健性，同时性能优于最先进的研究。



## **28. Improving Robustness of Jet Tagging Algorithms with Adversarial Training**

通过对抗性训练提高Jet标签算法的稳健性 physics.data-an

17 pages, 16 figures, 2 tables. Replaced with the published version.  Added the journal reference and the DOI. Code accessible under  https://github.com/AnnikaStein/Adversarial-Training-for-Jet-Tagging

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2203.13890v2)

**Authors**: Annika Stein, Xavier Coubez, Spandan Mondal, Andrzej Novak, Alexander Schmidt

**Abstracts**: Deep learning is a standard tool in the field of high-energy physics, facilitating considerable sensitivity enhancements for numerous analysis strategies. In particular, in identification of physics objects, such as jet flavor tagging, complex neural network architectures play a major role. However, these methods are reliant on accurate simulations. Mismodeling can lead to non-negligible differences in performance in data that need to be measured and calibrated against. We investigate the classifier response to input data with injected mismodelings and probe the vulnerability of flavor tagging algorithms via application of adversarial attacks. Subsequently, we present an adversarial training strategy that mitigates the impact of such simulated attacks and improves the classifier robustness. We examine the relationship between performance and vulnerability and show that this method constitutes a promising approach to reduce the vulnerability to poor modeling.

摘要: 深度学习是高能物理领域的标准工具，可大大提高许多分析策略的灵敏度。特别是，在喷气香精等物理对象的识别中，复杂的神经网络结构扮演着重要的角色。然而，这些方法依赖于准确的模拟。错误的建模可能会导致需要测量和校准的数据的性能出现不可忽略的差异。我们研究了分类器对带有注入误建模的输入数据的响应，并通过应用对抗性攻击来探索味道标注算法的脆弱性。随后，我们提出了一种对抗性训练策略，减轻了这类模拟攻击的影响，提高了分类器的稳健性。我们研究了性能和脆弱性之间的关系，并表明该方法是一种很有前途的方法，可以降低因建模不当而造成的脆弱性。



## **29. Cascading Failures in Power Grids**

电网中的连锁故障 eess.SY

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2209.08116v1)

**Authors**: Rounak Meyur, Anamitra Pal, Mina Youssef, Christopher L. Barrett, Achla Marathe, Stephen Eubank, Anil Vullikanti, Virgilio Centeno, Simon Levin, H. Vincent Poor, Arun Phadke, Madhav Marathe

**Abstracts**: This paper studies the consequences of a human-initiated targeted attack on the national electric power system. We consider two kinds of attacks: ($i$) an attack by an adversary that uses a tactical weapon and destroys a large part of the grid, by physically targeting a large geographic region; ($ii$) a targeted attack by an adversary that takes out a small number of critical components in the network simultaneously. Our analysis uses ($i$) a realistic representation of the underlying power grid, including the topology, the control and protection components, ($ii$) a realistic representation of the targeted attack scenario, and ($iii$) a dynamic stability analysis, that goes beyond traditional work comprising structural and linear flow analysis. Such realistic analysis is expensive, but critical since it can capture cascading failures that result from transient instabilities introduced due to the attack. Our model acknowledges the presence of hidden failures in the protection systems resulting in relay misoperations. We analyze the extent of cascading outages for different levels of hidden failures. Our results show that: ($i$) the power grid is vulnerable to both these attacks, ($ii$) the tactical attack has significant social, economic and health damage but need not result in a regional cascade; on the contrary the targeted attack can cause significant cascade and lead to power outage over a large region. Our work shows the necessity to harden the power grid not just to cyber-attacks but also to physical attacks. Furthermore, we show that realistic representations and analysis can lead to fundamentally new insights that simplified models are unlikely to capture. Finally, the methods and results help us identify critical elements in the grid; the system can then be hardened in a more precise manner to reduce the vulnerabilities.

摘要: 本文研究了人类发起的对国家电力系统的定向攻击的后果。我们考虑了两种类型的攻击：($I$)对手使用战术武器，通过物理上以大片地理区域为目标摧毁大部分电网的攻击；($II$)对手的有针对性的攻击，同时摧毁网络中的少量关键组件。我们的分析使用了($I$)基本电网的真实表示，包括拓扑、控制和保护组件，($II$)目标攻击场景的真实表示，以及($III$)动态稳定性分析，这超出了传统的包括结构和线性流分析的工作。这种现实的分析成本很高，但很关键，因为它可以捕获由攻击导致的瞬时不稳定导致的级联故障。我们的模型承认保护系统中存在导致继电保护误动作的隐藏故障。我们分析了不同级别的隐藏故障的连锁故障程度。我们的结果表明：($I$)电网容易受到这两种攻击，($II$)战术攻击具有重大的社会、经济和健康损害，但不一定会导致区域级联；相反，定向攻击可以造成重大的级联，并导致大范围的停电。我们的工作表明，加强电网不仅要抵御网络攻击，还要抵御物理攻击。此外，我们还表明，现实的表示和分析可以带来简化模型不太可能捕捉到的根本新的见解。最后，这些方法和结果帮助我们识别网格中的关键元素；然后可以以更精确的方式加强系统，以减少漏洞。



## **30. Adversarial Driving: Attacking End-to-End Autonomous Driving**

对抗性驾驶：攻击型端到端自动驾驶 cs.CV

7 pages, 6 figures

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2103.09151v4)

**Authors**: Han Wu, Syed Yunas, Sareh Rowlands, Wenjie Ruan, Johan Wahlstrom

**Abstracts**: As research in deep neural networks has advanced, deep convolutional networks have become feasible for automated driving tasks. In particular, there is an emerging trend of employing end-to-end neural network models for the automation of driving tasks. However, previous research has shown that deep neural network classifiers are vulnerable to adversarial attacks. For regression tasks, however, the effect of adversarial attacks is not as well understood. In this paper, we devise two white-box targeted attacks against end-to-end autonomous driving systems. The driving systems use a regression model that takes an image as input and outputs a steering angle. Our attacks manipulate the behavior of the autonomous driving system by perturbing the input image. Both attacks can be initiated in real-time on CPUs without employing GPUs. The efficiency of the attacks is illustrated using experiments conducted in Udacity. Demo video: https://youtu.be/I0i8uN2oOP0.

摘要: 随着深度神经网络研究的深入，深度卷积网络对于自动驾驶任务已经变得可行。特别是，使用端到端神经网络模型来实现驾驶任务自动化是一种新兴的趋势。然而，以往的研究表明，深度神经网络分类器容易受到敌意攻击。然而，对于回归任务，对抗性攻击的影响并没有被很好地理解。在本文中，我们设计了两种针对端到端自动驾驶系统的白盒针对性攻击。驾驶系统使用一个回归模型，该模型将图像作为输入，并输出转向角度。我们的攻击通过干扰输入图像来操纵自动驾驶系统的行为。这两种攻击都可以在不使用GPU的情况下在CPU上实时发起。通过在Udacity上进行的实验，说明了攻击的有效性。演示视频：https://youtu.be/I0i8uN2oOP0.



## **31. PA-Boot: A Formally Verified Authentication Protocol for Multiprocessor Secure Boot**

PA-Boot：一种形式化验证的多处理器安全引导认证协议 cs.CR

Manuscript submitted to IEEE Trans. Dependable Secure Comput

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2209.07936v1)

**Authors**: Zhuoruo Zhang, Chenyang Yu, He Huang, Rui Chang, Mingshuai Chen, Qinming Dai, Wenbo Shen, Yongwang Zhao, Kui Ren

**Abstracts**: Hardware supply-chain attacks are raising significant security threats to the boot process of multiprocessor systems. This paper identifies a new, prevalent hardware supply-chain attack surface that can bypass multiprocessor secure boot due to the absence of processor-authentication mechanisms. To defend against such attacks, we present PA-Boot, the first formally verified processor-authentication protocol for secure boot in multiprocessor systems. PA-Boot is proved functionally correct and is guaranteed to detect multiple adversarial behaviors, e.g., processor replacements, man-in-the-middle attacks, and tampering with certificates. The fine-grained formalization of PA-Boot and its fully mechanized security proofs are carried out in the Isabelle/HOL theorem prover with 306 lemmas/theorems and ~7,100 LoC. Experiments on a proof-of-concept implementation indicate that PA-Boot can effectively identify boot-process attacks with a considerably minor overhead and thereby improve the security of multiprocessor systems.

摘要: 硬件供应链攻击正在给多处理器系统的引导过程带来严重的安全威胁。本文提出了一种新的、流行的硬件供应链攻击面，由于缺乏处理器认证机制，该攻击面可以绕过多处理器安全引导。为了防御此类攻击，我们提出了PA-Boot，这是第一个经过正式验证的用于多处理器系统安全引导的处理器认证协议。PA-Boot被证明在功能上是正确的，并保证可以检测到多种敌对行为，例如处理器更换、中间人攻击和篡改证书。PA-Boot的细粒度形式化及其全机械化安全证明是在Isabelle/HOL定理证明器上进行的，具有306个引理/定理和~7100个LoC。在概念验证实现上的实验表明，PA-Boot能够以相当小的开销有效地识别引导过程攻击，从而提高多处理器系统的安全性。



## **32. SplitGuard: Detecting and Mitigating Training-Hijacking Attacks in Split Learning**

SplitGuard：检测和缓解分裂学习中的训练劫持攻击 cs.CR

Proceedings of the 21st Workshop on Privacy in the Electronic Society  (WPES '22), November 7, 2022, Los Angeles, CA, USA

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2108.09052v3)

**Authors**: Ege Erdogan, Alptekin Kupcu, A. Ercument Cicek

**Abstracts**: Distributed deep learning frameworks such as split learning provide great benefits with regards to the computational cost of training deep neural networks and the privacy-aware utilization of the collective data of a group of data-holders. Split learning, in particular, achieves this goal by dividing a neural network between a client and a server so that the client computes the initial set of layers, and the server computes the rest. However, this method introduces a unique attack vector for a malicious server attempting to steal the client's private data: the server can direct the client model towards learning any task of its choice, e.g. towards outputting easily invertible values. With a concrete example already proposed (Pasquini et al., CCS '21), such training-hijacking attacks present a significant risk for the data privacy of split learning clients.   In this paper, we propose SplitGuard, a method by which a split learning client can detect whether it is being targeted by a training-hijacking attack or not. We experimentally evaluate our method's effectiveness, compare it with potential alternatives, and discuss in detail various points related to its use. We conclude that SplitGuard can effectively detect training-hijacking attacks while minimizing the amount of information recovered by the adversaries.

摘要: 分布式深度学习框架，如分裂学习，在训练深度神经网络的计算成本和对一组数据持有者的集体数据的隐私意识利用方面提供了巨大的好处。特别是，分裂学习通过在客户端和服务器之间划分神经网络来实现这一目标，以便客户端计算初始层集，服务器计算其余层。然而，这种方法为试图窃取客户端私有数据的恶意服务器引入了唯一的攻击矢量：服务器可以引导客户端模型学习其选择的任何任务，例如输出容易逆转的值。结合已提出的一个具体实例(Pasquini等人，CCS‘21)，这种训练劫持攻击给分裂学习客户端的数据隐私带来了很大的风险。在本文中，我们提出了一种分裂学习客户端可以检测其是否成为训练劫持攻击目标的方法SplitGuard。我们通过实验评估了该方法的有效性，并将其与潜在的替代方案进行了比较，并详细讨论了与其使用相关的各个要点。我们得出结论：SplitGuard能够有效地检测训练劫持攻击，同时最小化对手恢复的信息量。



## **33. Privacy-Preserving Distributed Expectation Maximization for Gaussian Mixture Model using Subspace Perturbation**

基于子空间扰动的混合高斯模型隐私保护分布期望最大化 cs.LG

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2209.07833v1)

**Authors**: Qiongxiu Li, Jaron Skovsted Gundersen, Katrine Tjell, Rafal Wisniewski, Mads Græsbøll Christensen

**Abstracts**: Privacy has become a major concern in machine learning. In fact, the federated learning is motivated by the privacy concern as it does not allow to transmit the private data but only intermediate updates. However, federated learning does not always guarantee privacy-preservation as the intermediate updates may also reveal sensitive information. In this paper, we give an explicit information-theoretical analysis of a federated expectation maximization algorithm for Gaussian mixture model and prove that the intermediate updates can cause severe privacy leakage. To address the privacy issue, we propose a fully decentralized privacy-preserving solution, which is able to securely compute the updates in each maximization step. Additionally, we consider two different types of security attacks: the honest-but-curious and eavesdropping adversary models. Numerical validation shows that the proposed approach has superior performance compared to the existing approach in terms of both the accuracy and privacy level.

摘要: 隐私已经成为机器学习中的一个主要问题。事实上，联合学习是出于隐私考虑，因为它不允许传输私有数据，而只允许传输中间更新。然而，联合学习并不总是保证隐私保护，因为中间更新也可能泄露敏感信息。本文对一种联合期望最大化算法进行了详细的信息论分析，证明了中间更新会导致严重的隐私泄露。为了解决隐私问题，我们提出了一种完全去中心化的隐私保护解决方案，该方案能够安全地计算每个最大化步骤中的更新。此外，我们考虑了两种不同类型的安全攻击：诚实但好奇的和窃听对手模型。数值验证表明，与已有方法相比，该方法在准确率和保密性方面都具有更好的性能。



## **34. A Large-scale Multiple-objective Method for Black-box Attack against Object Detection**

针对目标检测的大规模多目标黑盒攻击方法 cs.CV

14 pages, 5 figures, ECCV2022

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2209.07790v1)

**Authors**: Siyuan Liang, Longkang Li, Yanbo Fan, Xiaojun Jia, Jingzhi Li, Baoyuan Wu, Xiaochun Cao

**Abstracts**: Recent studies have shown that detectors based on deep models are vulnerable to adversarial examples, even in the black-box scenario where the attacker cannot access the model information. Most existing attack methods aim to minimize the true positive rate, which often shows poor attack performance, as another sub-optimal bounding box may be detected around the attacked bounding box to be the new true positive one. To settle this challenge, we propose to minimize the true positive rate and maximize the false positive rate, which can encourage more false positive objects to block the generation of new true positive bounding boxes. It is modeled as a multi-objective optimization (MOP) problem, of which the generic algorithm can search the Pareto-optimal. However, our task has more than two million decision variables, leading to low searching efficiency. Thus, we extend the standard Genetic Algorithm with Random Subset selection and Divide-and-Conquer, called GARSDC, which significantly improves the efficiency. Moreover, to alleviate the sensitivity to population quality in generic algorithms, we generate a gradient-prior initial population, utilizing the transferability between different detectors with similar backbones. Compared with the state-of-art attack methods, GARSDC decreases by an average 12.0 in the mAP and queries by about 1000 times in extensive experiments. Our codes can be found at https://github.com/LiangSiyuan21/ GARSDC.

摘要: 最近的研究表明，基于深度模型的检测器容易受到敌意示例的攻击，即使在攻击者无法访问模型信息的黑盒场景中也是如此。现有的大多数攻击方法都以最小化真实正确率为目标，这往往表现出较差的攻击性能，因为可能会在被攻击的边界框周围检测到另一个次优边界框，即新的真正边界框。为了解决这一挑战，我们提出了最小化真阳性率和最大化假阳性率的方法，这可以鼓励更多的假阳性对象阻止新的真阳性边界框的生成。将其建模为多目标优化问题，利用遗传算法搜索Pareto最优解。然而，我们的任务有200多万个决策变量，导致搜索效率较低。因此，我们对标准遗传算法GARSDC进行了扩展，使其具有随机子集选择和分而治之的特点，大大提高了求解效率。此外，为了缓解遗传算法对种群质量的敏感性，我们利用具有相似骨架的不同检测器之间的可转移性，生成一个梯度先验的初始种群。在大量的实验中，与现有的攻击方法相比，GARSDC在地图和查询上平均减少了12.0倍左右。我们的代码可以在https://github.com/LiangSiyuan21/GARSDC找到。



## **35. PointCAT: Contrastive Adversarial Training for Robust Point Cloud Recognition**

PointCAT：用于稳健点云识别的对比性对抗性训练 cs.CV

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2209.07788v1)

**Authors**: Qidong Huang, Xiaoyi Dong, Dongdong Chen, Hang Zhou, Weiming Zhang, Kui Zhang, Gang Hua, Nenghai Yu

**Abstracts**: Notwithstanding the prominent performance achieved in various applications, point cloud recognition models have often suffered from natural corruptions and adversarial perturbations. In this paper, we delve into boosting the general robustness of point cloud recognition models and propose Point-Cloud Contrastive Adversarial Training (PointCAT). The main intuition of PointCAT is encouraging the target recognition model to narrow the decision gap between clean point clouds and corrupted point clouds. Specifically, we leverage a supervised contrastive loss to facilitate the alignment and uniformity of the hypersphere features extracted by the recognition model, and design a pair of centralizing losses with the dynamic prototype guidance to avoid these features deviating from their belonging category clusters. To provide the more challenging corrupted point clouds, we adversarially train a noise generator along with the recognition model from the scratch, instead of using gradient-based attack as the inner loop like previous adversarial training methods. Comprehensive experiments show that the proposed PointCAT outperforms the baseline methods and dramatically boosts the robustness of different point cloud recognition models, under a variety of corruptions including isotropic point noises, the LiDAR simulated noises, random point dropping and adversarial perturbations.

摘要: 尽管点云识别模型在各种应用中取得了显著的性能，但它经常受到自然的破坏和对抗性的扰动。本文对提高点云识别模型的整体稳健性进行了深入研究，提出了点云对抗性训练(PointCAT)。PointCAT的主要直觉是鼓励目标识别模型缩小干净的点云和损坏的点云之间的决策差距。具体地说，我们利用有监督的对比损失来促进识别模型提取的超球特征的对齐和一致性，并在动态原型引导下设计了一对集中损失来避免这些特征偏离其所属的类别簇。为了提供更具挑战性的被破坏的点云，我们从零开始对抗性地训练噪声产生器和识别模型，而不是像以前的对抗性训练方法那样使用基于梯度的攻击作为内环。综合实验表明，在各向同性点噪声、LiDAR模拟噪声、随机点丢弃和对抗性扰动等多种情况下，PointCAT的性能优于基线方法，并显著提高了不同点云识别模型的稳健性。



## **36. On the Robustness of Graph Neural Diffusion to Topology Perturbations**

关于图神经扩散对拓扑扰动的稳健性 cs.LG

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2209.07754v1)

**Authors**: Yang Song, Qiyu Kang, Sijie Wang, Zhao Kai, Wee Peng Tay

**Abstracts**: Neural diffusion on graphs is a novel class of graph neural networks that has attracted increasing attention recently. The capability of graph neural partial differential equations (PDEs) in addressing common hurdles of graph neural networks (GNNs), such as the problems of over-smoothing and bottlenecks, has been investigated but not their robustness to adversarial attacks. In this work, we explore the robustness properties of graph neural PDEs. We empirically demonstrate that graph neural PDEs are intrinsically more robust against topology perturbation as compared to other GNNs. We provide insights into this phenomenon by exploiting the stability of the heat semigroup under graph topology perturbations. We discuss various graph diffusion operators and relate them to existing graph neural PDEs. Furthermore, we propose a general graph neural PDE framework based on which a new class of robust GNNs can be defined. We verify that the new model achieves comparable state-of-the-art performance on several benchmark datasets.

摘要: 图上的神经扩散是一类新的图神经网络，近年来受到越来越多的关注。图神经偏微分方程组(PDE)在解决图神经网络(GNN)的常见障碍(如过光滑和瓶颈问题)方面的能力已被研究，但其对对手攻击的稳健性尚未得到研究。在这项工作中，我们研究了图神经偏微分方程的稳健性。我们的经验证明，与其他GNN相比，图神经PDE在本质上对拓扑扰动具有更强的鲁棒性。通过利用图的拓扑扰动下热半群的稳定性，我们提供了对这一现象的见解。我们讨论了各种图扩散算子，并将它们与现有的图神经偏微分方程联系起来。此外，我们还提出了一个通用的图神经偏微分方程框架，基于该框架可以定义一类新的健壮GNN。我们在几个基准数据集上验证了新模型取得了相当于最先进的性能。



## **37. IPvSeeYou: Exploiting Leaked Identifiers in IPv6 for Street-Level Geolocation**

IPv6 SeeYou：利用IPv6中泄漏的标识符进行街道级地理定位 cs.NI

Accepted to S&P '23

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2208.06767v2)

**Authors**: Erik Rye, Robert Beverly

**Abstracts**: We present IPvSeeYou, a privacy attack that permits a remote and unprivileged adversary to physically geolocate many residential IPv6 hosts and networks with street-level precision. The crux of our method involves: 1) remotely discovering wide area (WAN) hardware MAC addresses from home routers; 2) correlating these MAC addresses with their WiFi BSSID counterparts of known location; and 3) extending coverage by associating devices connected to a common penultimate provider router.   We first obtain a large corpus of MACs embedded in IPv6 addresses via high-speed network probing. These MAC addresses are effectively leaked up the protocol stack and largely represent WAN interfaces of residential routers, many of which are all-in-one devices that also provide WiFi. We develop a technique to statistically infer the mapping between a router's WAN and WiFi MAC addresses across manufacturers and devices, and mount a large-scale data fusion attack that correlates WAN MACs with WiFi BSSIDs available in wardriving (geolocation) databases. Using these correlations, we geolocate the IPv6 prefixes of $>$12M routers in the wild across 146 countries and territories. Selected validation confirms a median geolocation error of 39 meters. We then exploit technology and deployment constraints to extend the attack to a larger set of IPv6 residential routers by clustering and associating devices with a common penultimate provider router. While we responsibly disclosed our results to several manufacturers and providers, the ossified ecosystem of deployed residential cable and DSL routers suggests that our attack will remain a privacy threat into the foreseeable future.

摘要: 我们提出了IPv6 SeeYou，这是一种隐私攻击，允许远程和非特权对手以街道级别的精度物理定位许多住宅IPv6主机和网络。我们方法的关键涉及：1)从家庭路由器远程发现广域(WAN)硬件MAC地址；2)将这些MAC地址与已知位置的对应WiFi BSSID关联；以及3)通过关联连接到公共倒数第二个提供商路由器的设备来扩展覆盖范围。我们首先通过高速网络探测获得嵌入在IPv6地址中的大量MAC语料库。这些MAC地址有效地沿协议堆栈向上泄露，主要代表住宅路由器的广域网接口，其中许多是也提供WiFi的一体化设备。我们开发了一种技术来统计推断路由器的广域网和跨制造商和设备的WiFi MAC地址之间的映射，并发动大规模数据融合攻击，将广域网MAC与战争驾驶(地理定位)数据库中提供的WiFi BSSID相关联。利用这些相关性，我们在146个国家和地区对价值超过1200万美元的路由器的IPv6前缀进行了地理定位。选定的验证确认地理位置误差的中位数为39米。然后，我们利用技术和部署限制将攻击扩展到更大的一组IPv6住宅路由器，方法是将设备与常见的倒数第二个提供商路由器进行集群和关联。虽然我们负责任地向几家制造商和供应商披露了我们的结果，但已部署的住宅有线电视和DSL路由器的僵化生态系统表明，在可预见的未来，我们的攻击仍将对隐私构成威胁。



## **38. Adversarial Detection: Attacking Object Detection in Real Time**

对抗性检测：攻击目标的实时检测 cs.AI

7 pages, 10 figures

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2209.01962v2)

**Authors**: Han Wu, Syed Yunas, Sareh Rowlands, Wenjie Ruan, Johan Wahlstrom

**Abstracts**: Intelligent robots rely on object detection models to perceive the environment. Following advances in deep learning security it has been revealed that object detection models are vulnerable to adversarial attacks. However, prior research primarily focuses on attacking static images or offline videos. Therefore, it is still unclear if such attacks could jeopardize real-world robotic applications in dynamic environments. This paper bridges this gap by presenting the first real-time online attack against object detection models. We devise three attacks that fabricate bounding boxes for nonexistent objects at desired locations. The attacks achieve a success rate of about 90% within about 20 iterations. The demo video is available at: https://youtu.be/zJZ1aNlXsMU.

摘要: 智能机器人依靠物体检测模型来感知环境。随着深度学习安全性的进步，人们发现目标检测模型容易受到敌意攻击。然而，以往的研究主要集中在攻击静态图像或离线视频上。因此，目前尚不清楚此类攻击是否会危及动态环境中真实世界的机器人应用。本文通过提出第一个针对目标检测的实时在线攻击模型来弥补这一差距。我们设计了三种攻击，在所需位置为不存在的对象制造边界框。这些攻击在大约20次迭代内实现了约90%的成功率。演示视频可在以下网站上查看：https://youtu.be/zJZ1aNlXsMU.



## **39. A Man-in-the-Middle Attack against Object Detection Systems**

一种针对目标检测系统的中间人攻击 cs.RO

7 pages, 8 figures

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2208.07174v2)

**Authors**: Han Wu, Sareh Rowlands, Johan Wahlstrom

**Abstracts**: Thanks to the increasing power of CPUs and GPUs in embedded systems, deep-learning-enabled object detection systems have become pervasive in a multitude of robotic applications. While deep learning models are vulnerable to several well-known adversarial attacks, the applicability of these attacks is severely limited by strict assumptions on, for example, access to the detection system. Inspired by Man-in-the-Middle attacks in cryptography, we propose a novel hardware attack on object detection systems that overcomes these limitations. Experiments prove that it is possible to generate an efficient Universal Adversarial Perturbation (UAP) within one minute and then use the perturbation to attack a detection system via the Man-in-the-Middle attack. These findings raise serious concerns for applications of deep learning models in safety-critical systems, such as autonomous driving. Demo Video: https://youtu.be/OvIpe-R3ZS8.

摘要: 由于嵌入式系统中CPU和GPU的能力不断增强，支持深度学习的目标检测系统已经在许多机器人应用中变得普遍。虽然深度学习模型容易受到几种著名的对抗性攻击，但这些攻击的适用性受到对检测系统访问权限的严格假设的严重限制。受密码学中中间人攻击的启发，我们提出了一种新的针对目标检测系统的硬件攻击，克服了这些局限性。实验证明，可以在一分钟内产生一个有效的通用对抗扰动(UAP)，然后利用该扰动通过中间人攻击来攻击检测系统。这些发现引发了人们对深度学习模型在自动驾驶等安全关键系统中应用的严重担忧。演示视频：https://youtu.be/OvIpe-R3ZS8.



## **40. Adversarial Training for High-Stakes Reliability**

高风险可靠性的对抗性训练 cs.LG

31 pages, 6 figures, fixed incorrect citation

**SubmitDate**: 2022-09-15    [paper-pdf](http://arxiv.org/pdf/2205.01663v3)

**Authors**: Daniel M. Ziegler, Seraphina Nix, Lawrence Chan, Tim Bauman, Peter Schmidt-Nielsen, Tao Lin, Adam Scherlis, Noa Nabeshima, Ben Weinstein-Raun, Daniel de Haas, Buck Shlegeris, Nate Thomas

**Abstracts**: In the future, powerful AI systems may be deployed in high-stakes settings, where a single failure could be catastrophic. One technique for improving AI safety in high-stakes settings is adversarial training, which uses an adversary to generate examples to train on in order to achieve better worst-case performance.   In this work, we used a language generation task as a testbed for achieving high reliability through adversarial training. We created a series of adversarial training techniques -- including a tool that assists human adversaries -- to find and eliminate failures in a classifier that filters text completions suggested by a generator. In our simple "avoid injuries" task, we determined that we can set very conservative classifier thresholds without significantly impacting the quality of the filtered outputs. With our chosen thresholds, filtering with our baseline classifier decreases the rate of unsafe completions from about 2.4% to 0.003% on in-distribution data, which is near the limit of our ability to measure. We found that adversarial training significantly increased robustness to the adversarial attacks that we trained on, without affecting in-distribution performance. We hope to see further work in the high-stakes reliability setting, including more powerful tools for enhancing human adversaries and better ways to measure high levels of reliability, until we can confidently rule out the possibility of catastrophic deployment-time failures of powerful models.

摘要: 未来，强大的人工智能系统可能会部署在高风险的环境中，在那里，单一的故障可能是灾难性的。在高风险环境中提高人工智能安全性的一种技术是对抗性训练，它使用对手生成样本进行训练，以实现更好的最坏情况下的性能。在这项工作中，我们使用一个语言生成任务作为测试平台，通过对抗性训练来实现高可靠性。我们创建了一系列对抗性训练技术--包括一个帮助人类对手的工具--来发现并消除分类器中的故障，该分类器过滤生成器建议的文本完成。在我们简单的“避免受伤”任务中，我们确定可以设置非常保守的分类器阈值，而不会显著影响过滤输出的质量。在我们选择的阈值下，使用我们的基准分类器进行过滤可以将分发内数据的不安全完成率从大约2.4%降低到0.003%，这接近我们的测量能力极限。我们发现，对抗性训练显著提高了对我们训练的对抗性攻击的健壮性，而不影响分发内性能。我们希望在高风险的可靠性环境中看到进一步的工作，包括更强大的工具来增强人类对手，以及更好的方法来衡量高水平的可靠性，直到我们可以自信地排除强大模型部署时灾难性故障的可能性。



## **41. How to Attack and Defend NextG Radio Access Network Slicing with Reinforcement Learning**

基于强化学习的下一代无线接入网络分片攻防 cs.NI

**SubmitDate**: 2022-09-15    [paper-pdf](http://arxiv.org/pdf/2101.05768v2)

**Authors**: Yi Shi, Yalin E. Sagduyu, Tugba Erpek, M. Cenk Gursoy

**Abstracts**: In this paper, reinforcement learning (RL) for network slicing is considered in NextG radio access networks, where the base station (gNodeB) allocates resource blocks (RBs) to the requests of user equipments and aims to maximize the total reward of accepted requests over time. Based on adversarial machine learning, a novel over-the-air attack is introduced to manipulate the RL algorithm and disrupt NextG network slicing. The adversary observes the spectrum and builds its own RL based surrogate model that selects which RBs to jam subject to an energy budget with the objective of maximizing the number of failed requests due to jammed RBs. By jamming the RBs, the adversary reduces the RL algorithm's reward. As this reward is used as the input to update the RL algorithm, the performance does not recover even after the adversary stops jamming. This attack is evaluated in terms of both the recovery time and the (maximum and total) reward loss, and it is shown to be much more effective than benchmark (random and myopic) jamming attacks. Different reactive and proactive defense schemes (protecting the RL algorithm's updates or misleading the adversary's learning process) are introduced to show that it is viable to defend NextG network slicing against this attack.

摘要: 本文研究了下一代无线接入网络中网络切片的强化学习方法，其中基站(GNodeB)为用户设备的请求分配资源块(RB)，并以最大化接受请求的总回报为目标。在对抗性机器学习的基础上，引入了一种新的空中攻击来操纵RL算法并破坏NextG网络切片。敌手观察频谱并构建其自己的基于RL的代理模型，该代理模型选择在能量预算下阻塞哪个RB，目标是最大化由于阻塞的RB而失败的请求的数量。通过干扰RBS，对手降低了RL算法的奖励。由于该奖励被用作更新RL算法的输入，因此即使在对手停止干扰之后，性能也不会恢复。该攻击从恢复时间和(最大和总的)报酬损失两个方面进行评估，并且被证明比基准(随机和近视)干扰攻击要有效得多。不同的反应性和主动性防御方案(保护RL算法的更新或误导对手的学习过程)被引入，以表明防御NextG网络切片攻击是可行的。



## **42. A Light Recipe to Train Robust Vision Transformers**

培养健壮的视觉变形器的光明秘诀 cs.CV

Code available at https://github.com/dedeswim/vits-robustness-torch

**SubmitDate**: 2022-09-15    [paper-pdf](http://arxiv.org/pdf/2209.07399v1)

**Authors**: Edoardo Debenedetti, Vikash Sehwag, Prateek Mittal

**Abstracts**: In this paper, we ask whether Vision Transformers (ViTs) can serve as an underlying architecture for improving the adversarial robustness of machine learning models against evasion attacks. While earlier works have focused on improving Convolutional Neural Networks, we show that also ViTs are highly suitable for adversarial training to achieve competitive performance. We achieve this objective using a custom adversarial training recipe, discovered using rigorous ablation studies on a subset of the ImageNet dataset. The canonical training recipe for ViTs recommends strong data augmentation, in part to compensate for the lack of vision inductive bias of attention modules, when compared to convolutions. We show that this recipe achieves suboptimal performance when used for adversarial training. In contrast, we find that omitting all heavy data augmentation, and adding some additional bag-of-tricks ($\varepsilon$-warmup and larger weight decay), significantly boosts the performance of robust ViTs. We show that our recipe generalizes to different classes of ViT architectures and large-scale models on full ImageNet-1k. Additionally, investigating the reasons for the robustness of our models, we show that it is easier to generate strong attacks during training when using our recipe and that this leads to better robustness at test time. Finally, we further study one consequence of adversarial training by proposing a way to quantify the semantic nature of adversarial perturbations and highlight its correlation with the robustness of the model. Overall, we recommend that the community should avoid translating the canonical training recipes in ViTs to robust training and rethink common training choices in the context of adversarial training.

摘要: 在这篇文章中，我们问视觉转换器(VITS)是否可以作为一个底层架构来提高机器学习模型对逃避攻击的对抗性健壮性。虽然早期的工作集中在改进卷积神经网络上，但我们也表明VITS非常适合于对抗性训练，以获得竞争性的性能。我们通过对ImageNet数据集的子集进行严格的消融研究，发现了一种定制的对抗性训练配方，从而实现了这一目标。VITS的规范训练配方建议进行强大的数据增强，部分原因是为了弥补与卷曲相比，注意力模块缺乏视觉诱导偏差。我们表明，当用于对抗性训练时，这个配方达到了次优的性能。相反，我们发现，省略所有繁重的数据增强，并添加一些额外的技巧($varepsilon$-热身和更大的权重衰减)，显著提高了健壮VITS的性能。我们表明，我们的配方适用于不同类别的VIT体系结构和完整的ImageNet-1k上的大型模型。此外，研究了我们的模型健壮性的原因，我们表明，当使用我们的配方时，在训练期间更容易产生强攻击，这导致在测试时更好的健壮性。最后，我们进一步研究了对抗性训练的一个后果，提出了一种量化对抗性扰动的语义性质的方法，并强调了它与模型的稳健性的相关性。总体而言，我们建议社会应避免将VITS中的规范训练食谱转化为稳健的训练，并在对抗性训练的背景下重新考虑常见的训练选择。



## **43. Continuous Patrolling Games**

连续巡逻小游戏 cs.DM

**SubmitDate**: 2022-09-15    [paper-pdf](http://arxiv.org/pdf/2008.07369v2)

**Authors**: Steve Alpern, Thuy Bui, Thomas Lidbetter, Katerina Papadaki

**Abstracts**: We study a patrolling game played on a network $Q$, considered as a metric space. The Attacker chooses a point of $Q$ (not necessarily a node) to attack during a chosen time interval of fixed duration. The Patroller chooses a unit speed path on $Q$ and intercepts the attack (and wins) if she visits the attacked point during the attack time interval. This zero-sum game models the problem of protecting roads or pipelines from an adversarial attack. The payoff to the maximizing Patroller is the probability that the attack is intercepted. Our results include the following: (i) a solution to the game for any network $Q$, as long as the time required to carry out the attack is sufficiently short, (ii) a solution to the game for all tree networks that satisfy a certain condition on their extremities, and (iii) a solution to the game for any attack duration for stars with one long arc and the remaining arcs equal in length. We present a conjecture on the solution of the game for arbitrary trees and establish it in certain cases.

摘要: 我们研究了在被认为是度量空间的网络$Q$上进行的巡逻对策。攻击者在选定的固定持续时间间隔内选择一个$Q$点(不一定是节点)进行攻击。巡护员在$Q$上选择一条单位速度路径，如果她在攻击时间间隔内访问被攻击点，则拦截攻击(并获胜)。这个零和博弈模拟了保护道路或管道免受对手攻击的问题。最大化巡逻的回报是攻击被拦截的概率。我们的结果包括：(I)任意网络$q$的对策解，只要进行攻击所需的时间足够短；(Ii)对于所有在其末端满足一定条件的树网络的对策解；(Iii)对于具有一条长弧线且其余弧长相等的恒星的任意攻击持续时间的对策解。我们对任意树的对策的解提出了一个猜想，并在某些情况下建立了它。



## **44. Defending From Physically-Realizable Adversarial Attacks Through Internal Over-Activation Analysis**

通过内部过度激活分析防御物理上可实现的敌意攻击 cs.CV

**SubmitDate**: 2022-09-15    [paper-pdf](http://arxiv.org/pdf/2203.07341v2)

**Authors**: Giulio Rossolini, Federico Nesti, Fabio Brau, Alessandro Biondi, Giorgio Buttazzo

**Abstracts**: This work presents Z-Mask, a robust and effective strategy to improve the adversarial robustness of convolutional networks against physically-realizable adversarial attacks. The presented defense relies on specific Z-score analysis performed on the internal network features to detect and mask the pixels corresponding to adversarial objects in the input image. To this end, spatially contiguous activations are examined in shallow and deep layers to suggest potential adversarial regions. Such proposals are then aggregated through a multi-thresholding mechanism. The effectiveness of Z-Mask is evaluated with an extensive set of experiments carried out on models for both semantic segmentation and object detection. The evaluation is performed with both digital patches added to the input images and printed patches positioned in the real world. The obtained results confirm that Z-Mask outperforms the state-of-the-art methods in terms of both detection accuracy and overall performance of the networks under attack. Additional experiments showed that Z-Mask is also robust against possible defense-aware attacks.

摘要: 这项工作提出了一种健壮而有效的Z-MASK策略来提高卷积网络对物理可实现的敌意攻击的健壮性。所提出的防御依赖于对内部网络特征执行的特定Z-Score分析来检测和掩蔽输入图像中与敌对对象对应的像素。为此，在浅层和深层检查了空间上连续的激活，以暗示潜在的对抗性区域。然后，通过多门槛机制汇总这些建议。通过在语义分割和目标检测模型上进行的大量实验，对Z-MASK的有效性进行了评估。使用添加到输入图像的数字补丁和位于真实世界中的打印补丁来执行评估。实验结果表明，Z-MASK在检测准确率和网络整体性能方面均优于现有的方法。其他实验表明，Z-MASK对可能的防御感知攻击也具有很强的健壮性。



## **45. Improving Robust Fairness via Balance Adversarial Training**

通过平衡对抗训练提高稳健公平性 cs.LG

**SubmitDate**: 2022-09-15    [paper-pdf](http://arxiv.org/pdf/2209.07534v1)

**Authors**: Chunyu Sun, Chenye Xu, Chengyuan Yao, Siyuan Liang, Yichao Wu, Ding Liang, XiangLong Liu, Aishan Liu

**Abstracts**: Adversarial training (AT) methods are effective against adversarial attacks, yet they introduce severe disparity of accuracy and robustness between different classes, known as the robust fairness problem. Previously proposed Fair Robust Learning (FRL) adaptively reweights different classes to improve fairness. However, the performance of the better-performed classes decreases, leading to a strong performance drop. In this paper, we observed two unfair phenomena during adversarial training: different difficulties in generating adversarial examples from each class (source-class fairness) and disparate target class tendencies when generating adversarial examples (target-class fairness). From the observations, we propose Balance Adversarial Training (BAT) to address the robust fairness problem. Regarding source-class fairness, we adjust the attack strength and difficulties of each class to generate samples near the decision boundary for easier and fairer model learning; considering target-class fairness, by introducing a uniform distribution constraint, we encourage the adversarial example generation process for each class with a fair tendency. Extensive experiments conducted on multiple datasets (CIFAR-10, CIFAR-100, and ImageNette) demonstrate that our method can significantly outperform other baselines in mitigating the robust fairness problem (+5-10\% on the worst class accuracy)

摘要: 对抗训练(AT)方法对对抗攻击是有效的，但它们在不同类别之间引入了严重的准确性和稳健性差异，称为鲁棒公平性问题。以前提出的公平稳健学习(FRL)自适应地调整不同类别的权重以提高公平性。然而，表现较好的类的性能会下降，导致性能大幅下降。在本文中，我们观察到对抗性训练中的两种不公平现象：从每一类生成对抗性实例的难度不同(源类公平)和生成对抗性实例时不同的目标类倾向(目标类公平性)。通过观察，我们提出了平衡对抗训练(BAT)来解决稳健的公平性问题。在源类公平性方面，我们调整每个类的攻击强度和难度，在决策边界附近生成样本，使模型学习更容易、更公平；在目标类公平性方面，通过引入均匀分布约束，鼓励每个类具有公平倾向的对抗性样本生成过程。在多个数据集(CIFAR-10、CIFAR-100和ImageNette)上进行的大量实验表明，我们的方法在缓解稳健公平性问题方面可以显著优于其他基线(在最差分类准确率上+5-10%)



## **46. Decision-based Black-box Attack Against Vision Transformers via Patch-wise Adversarial Removal**

基于决策的基于补丁对抗性去除的视觉变形金刚黑盒攻击 cs.CV

**SubmitDate**: 2022-09-15    [paper-pdf](http://arxiv.org/pdf/2112.03492v2)

**Authors**: Yucheng Shi, Yahong Han, Yu-an Tan, Xiaohui Kuang

**Abstracts**: Vision transformers (ViTs) have demonstrated impressive performance and stronger adversarial robustness compared to Convolutional Neural Networks (CNNs). On the one hand, ViTs' focus on global interaction between individual patches reduces the local noise sensitivity of images. On the other hand, the neglect of noise sensitivity differences between image regions by existing decision-based attacks further compromises the efficiency of noise compression, especially for ViTs. Therefore, validating the black-box adversarial robustness of ViTs when the target model can only be queried still remains a challenging problem. In this paper, we theoretically analyze the limitations of existing decision-based attacks from the perspective of noise sensitivity difference between regions of the image, and propose a new decision-based black-box attack against ViTs, termed Patch-wise Adversarial Removal (PAR). PAR divides images into patches through a coarse-to-fine search process and compresses the noise on each patch separately. PAR records the noise magnitude and noise sensitivity of each patch and selects the patch with the highest query value for noise compression. In addition, PAR can be used as a noise initialization method for other decision-based attacks to improve the noise compression efficiency on both ViTs and CNNs without introducing additional calculations. Extensive experiments on three datasets demonstrate that PAR achieves a much lower noise magnitude with the same number of queries.

摘要: 与卷积神经网络(CNN)相比，视觉转换器(VITS)表现出了令人印象深刻的性能和更强的对抗鲁棒性。一方面，VITS对单个斑块之间全局交互的关注降低了图像的局部噪声敏感度。另一方面，现有的基于决策的攻击忽略了图像区域之间的噪声敏感度差异，进一步影响了噪声压缩的效率，特别是对VITS。因此，当目标模型只能被查询时，验证VITS的黑箱对抗健壮性仍然是一个具有挑战性的问题。本文从图像区域噪声敏感度差异的角度，从理论上分析了现有的基于决策的攻击方法的局限性，提出了一种新的基于决策的VITS黑盒攻击方法，称为Patch-Wise Aversarial Removal(PAR)。PAR通过从粗到精的搜索过程将图像分成多个块，并分别压缩每个块上的噪声。PAR记录每个面片的噪声大小和噪声敏感度，并选择查询值最高的面片进行噪声压缩。此外，PAR可以用作其他基于决策的攻击的噪声初始化方法，从而在不引入额外计算的情况下提高VITS和CNN的噪声压缩效率。在三个数据集上的大量实验表明，在相同的查询次数下，PAR获得了低得多的噪声幅度。



## **47. PointACL:Adversarial Contrastive Learning for Robust Point Clouds Representation under Adversarial Attack**

PointACL：对抗性攻击下鲁棒点云表示的对抗性对比学习 cs.CV

arXiv admin note: text overlap with arXiv:2109.00179 by other authors

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06971v1)

**Authors**: Junxuan Huang, Yatong An, Lu cheng, Bai Chen, Junsong Yuan, Chunming Qiao

**Abstracts**: Despite recent success of self-supervised based contrastive learning model for 3D point clouds representation, the adversarial robustness of such pre-trained models raised concerns. Adversarial contrastive learning (ACL) is considered an effective way to improve the robustness of pre-trained models. In contrastive learning, the projector is considered an effective component for removing unnecessary feature information during contrastive pretraining and most ACL works also use contrastive loss with projected feature representations to generate adversarial examples in pretraining, while "unprojected " feature representations are used in generating adversarial inputs during inference.Because of the distribution gap between projected and "unprojected" features, their models are constrained of obtaining robust feature representations for downstream tasks. We introduce a new method to generate high-quality 3D adversarial examples for adversarial training by utilizing virtual adversarial loss with "unprojected" feature representations in contrastive learning framework. We present our robust aware loss function to train self-supervised contrastive learning framework adversarially. Furthermore, we find selecting high difference points with the Difference of Normal (DoN) operator as additional input for adversarial self-supervised contrastive learning can significantly improve the adversarial robustness of the pre-trained model. We validate our method, PointACL on downstream tasks, including 3D classification and 3D segmentation with multiple datasets. It obtains comparable robust accuracy over state-of-the-art contrastive adversarial learning methods.

摘要: 尽管最近基于自监督的对比学习模型在三维点云表示中取得了成功，但这种预训练模型的对抗性健壮性引起了人们的关注。对抗性对比学习被认为是提高预训练模型稳健性的有效方法。在对比学习中，投影器被认为是对比预训练中去除不必要特征信息的有效部件，大多数ACL工作在预训练中也使用对比损失和投影特征表征来生成对抗性样本，而在推理过程中则使用“非投影”特征表征来生成对抗性输入，由于投影和非投影特征之间的分布差距，它们的模型在获得下游任务的稳健特征表征方面受到限制。我们介绍了一种新的方法，通过在对比学习框架中利用虚拟对抗性损失和非投影的特征表示来生成用于对抗性训练的高质量3D对抗性实例。我们提出了稳健的意识损失函数来对抗性地训练自监督对比学习框架。此外，我们发现选择高差点和正态(DON)算子的差作为对抗性自监督对比学习的附加输入可以显著提高预训练模型的对抗性健壮性。我们在下游任务上验证了我们的方法，PointACL，包括3D分类和多个数据集的3D分割。与最先进的对比对抗性学习方法相比，它获得了相当的鲁棒性精度。



## **48. Finetuning Pretrained Vision-Language Models with Correlation Information Bottleneck for Robust Visual Question Answering**

基于关联信息瓶颈的视觉问答精调算法 cs.CV

20 pages, 4 figures, 13 tables

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06954v1)

**Authors**: Jingjing Jiang, Ziyi Liu, Nanning Zheng

**Abstracts**: Benefiting from large-scale Pretrained Vision-Language Models (VL-PMs), the performance of Visual Question Answering (VQA) has started to approach human oracle performance. However, finetuning large-scale VL-PMs with limited data for VQA usually faces overfitting and poor generalization issues, leading to a lack of robustness. In this paper, we aim to improve the robustness of VQA systems (ie, the ability of the systems to defend against input variations and human-adversarial attacks) from the perspective of Information Bottleneck when finetuning VL-PMs for VQA. Generally, internal representations obtained by VL-PMs inevitably contain irrelevant and redundant information for the downstream VQA task, resulting in statistically spurious correlations and insensitivity to input variations. To encourage representations to converge to a minimal sufficient statistic in vision-language learning, we propose the Correlation Information Bottleneck (CIB) principle, which seeks a tradeoff between representation compression and redundancy by minimizing the mutual information (MI) between the inputs and internal representations while maximizing the MI between the outputs and the representations. Meanwhile, CIB measures the internal correlations among visual and linguistic inputs and representations by a symmetrized joint MI estimation. Extensive experiments on five VQA benchmarks of input robustness and two VQA benchmarks of human-adversarial robustness demonstrate the effectiveness and superiority of the proposed CIB in improving the robustness of VQA systems.

摘要: 得益于大规模预训练视觉语言模型(VL-PM)，视觉问答(VQA)的性能已经开始接近人类的预言性能。然而，在VQA数据有限的情况下，对大规模的VL-PM进行精调通常会面临过拟合和泛化能力差的问题，从而导致缺乏健壮性。本文从信息瓶颈的角度对VQA的VL-PM进行优化，旨在提高VQA系统的健壮性(即系统抵抗输入变化和人类攻击的能力)。通常，VL-PM得到的内部表示不可避免地包含与下游VQA任务无关和冗余的信息，导致统计上虚假的相关性和对输入变化的不敏感。为了在视觉语言学习中鼓励表征收敛到最小的充分统计量，我们提出了关联信息瓶颈(CIB)原则，该原则通过最小化输入和内部表征之间的互信息(MI)同时最大化输出和表征之间的MI来寻求表征压缩和冗余之间的折衷。同时，CIB通过对称化的联合MI估计来衡量视觉输入和语言输入与表征之间的内在关联。在5个输入健壮性VQA基准和2个人-对手健壮性VQA基准上的大量实验证明了所提出的CIB在提高VQA系统稳健性方面的有效性和优越性。



## **49. On the interplay of adversarial robustness and architecture components: patches, convolution and attention**

关于对抗性健壮性和体系结构组件的相互作用：补丁、卷积和注意力 cs.CV

Presented at the "New Frontiers in Adversarial Machine Learning"  Workshop at ICML 2022

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06953v1)

**Authors**: Francesco Croce, Matthias Hein

**Abstracts**: In recent years novel architecture components for image classification have been developed, starting with attention and patches used in transformers. While prior works have analyzed the influence of some aspects of architecture components on the robustness to adversarial attacks, in particular for vision transformers, the understanding of the main factors is still limited. We compare several (non)-robust classifiers with different architectures and study their properties, including the effect of adversarial training on the interpretability of the learnt features and robustness to unseen threat models. An ablation from ResNet to ConvNeXt reveals key architectural changes leading to almost $10\%$ higher $\ell_\infty$-robustness.

摘要: 近年来，从变压器中使用的注意和补丁开始，开发了用于图像分类的新的体系结构组件。虽然以前的工作已经分析了体系结构组件的某些方面对对抗攻击的健壮性的影响，特别是对于视觉转换器，但对主要因素的理解仍然有限。我们比较了几种不同结构的(非)稳健分类器，并研究了它们的性质，包括对抗性训练对学习特征的可解释性的影响以及对不可见威胁模型的稳健性。从ResNet到ConvNeXt的消融揭示了导致$10\$更高的$\ell_\inty$-健壮性的关键架构变化。



## **50. Robust Constrained Reinforcement Learning**

稳健的约束强化学习 cs.LG

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06866v1)

**Authors**: Yue Wang, Fei Miao, Shaofeng Zou

**Abstracts**: Constrained reinforcement learning is to maximize the expected reward subject to constraints on utilities/costs. However, the training environment may not be the same as the test one, due to, e.g., modeling error, adversarial attack, non-stationarity, resulting in severe performance degradation and more importantly constraint violation. We propose a framework of robust constrained reinforcement learning under model uncertainty, where the MDP is not fixed but lies in some uncertainty set, the goal is to guarantee that constraints on utilities/costs are satisfied for all MDPs in the uncertainty set, and to maximize the worst-case reward performance over the uncertainty set. We design a robust primal-dual approach, and further theoretically develop guarantee on its convergence, complexity and robust feasibility. We then investigate a concrete example of $\delta$-contamination uncertainty set, design an online and model-free algorithm and theoretically characterize its sample complexity.

摘要: 约束强化学习是在效用/成本约束下最大化期望收益的学习方法。然而，由于例如建模错误、对抗性攻击、非平稳性等原因，训练环境可能与测试环境不同，从而导致严重的性能降级，更重要的是违反约束。我们提出了一种模型不确定性下的鲁棒约束强化学习框架，其中MDP不是固定的，而是位于某个不确定集合中，目标是保证不确定集合中的所有MDP都满足效用/成本约束，并且最大化不确定集合上的最坏情况下的回报性能。我们设计了一种稳健的原始-对偶方法，并从理论上对其收敛、复杂性和稳健可行性进行了保证。然后，我们研究了一个具体的例子--污染不确定集，设计了一个在线的、无模型的算法，并从理论上刻画了其样本复杂性。



