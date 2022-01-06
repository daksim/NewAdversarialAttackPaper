# Latest Adversarial Attack Papers
**update at 2022-01-06 11:37:06**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Aspis: A Robust Detection System for Distributed Learning**

ASPIS：一种健壮的分布式学习检测系统 cs.LG

17 pages, 23 figures

**SubmitDate**: 2022-01-05    [paper-pdf](http://arxiv.org/pdf/2108.02416v2)

**Authors**: Konstantinos Konstantinidis, Aditya Ramamoorthy

**Abstracts**: State-of-the-art machine learning models are routinely trained on large-scale distributed clusters. Crucially, such systems can be compromised when some of the computing devices exhibit abnormal (Byzantine) behavior and return arbitrary results to the parameter server (PS). This behavior may be attributed to a plethora of reasons, including system failures and orchestrated attacks. Existing work suggests robust aggregation and/or computational redundancy to alleviate the effect of distorted gradients. However, most of these schemes are ineffective when an adversary knows the task assignment and can choose the attacked workers judiciously to induce maximal damage. Our proposed method Aspis assigns gradient computations to worker nodes using a subset-based assignment which allows for multiple consistency checks on the behavior of a worker node. Examination of the calculated gradients and post-processing (clique-finding in an appropriately constructed graph) by the central node allows for efficient detection and subsequent exclusion of adversaries from the training process. We prove the Byzantine resilience and detection guarantees of Aspis under weak and strong attacks and extensively evaluate the system on various large-scale training scenarios. The principal metric for our experiments is the test accuracy, for which we demonstrate a significant improvement of about 30% compared to many state-of-the-art approaches on the CIFAR-10 dataset. The corresponding reduction of the fraction of corrupted gradients ranges from 16% to 99%.

摘要: 最先进的机器学习模型通常在大规模分布式集群上进行训练。重要的是，当一些计算设备表现出异常(拜占庭)行为并向参数服务器(PS)返回任意结果时，这样的系统可能受到危害。此行为可能归因于过多的原因，包括系统故障和精心策划的攻击。现有的工作表明，稳健的聚集和/或计算冗余可以减轻扭曲梯度的影响。然而，当敌手知道任务分配，并且能够明智地选择被攻击的工人以造成最大的损害时，这些方案大多是无效的。我们提出的方法ASPIS使用基于子集的分配将梯度计算分配给工作节点，这允许对工作节点的行为进行多个一致性检查。由中心节点检查计算出的梯度和后处理(在适当构造的图中发现集团)允许有效地检测并随后将对手排除在训练过程之外。我们证明了ASPIS在弱攻击和强攻击下的拜占庭抗攻击能力和检测保证，并在各种大规模训练场景下对该系统进行了广泛的评估。我们实验的主要指标是测试精度，在CIFAR-10数据集上，与许多最先进的方法相比，我们证明了这一点有大约30%的显著改进。受污染梯度比例的相应降低范围为16%至99%。



## **2. ROOM: Adversarial Machine Learning Attacks Under Real-Time Constraints**

房间：实时约束下的对抗性机器学习攻击 cs.CR

12 pages

**SubmitDate**: 2022-01-05    [paper-pdf](http://arxiv.org/pdf/2201.01621v1)

**Authors**: Amira Guesmi, Khaled N. Khasawneh, Nael Abu-Ghazaleh, Ihsen Alouani

**Abstracts**: Advances in deep learning have enabled a wide range of promising applications. However, these systems are vulnerable to Adversarial Machine Learning (AML) attacks; adversarially crafted perturbations to their inputs could cause them to misclassify. Several state-of-the-art adversarial attacks have demonstrated that they can reliably fool classifiers making these attacks a significant threat. Adversarial attack generation algorithms focus primarily on creating successful examples while controlling the noise magnitude and distribution to make detection more difficult. The underlying assumption of these attacks is that the adversarial noise is generated offline, making their execution time a secondary consideration. However, recently, just-in-time adversarial attacks where an attacker opportunistically generates adversarial examples on the fly have been shown to be possible. This paper introduces a new problem: how do we generate adversarial noise under real-time constraints to support such real-time adversarial attacks? Understanding this problem improves our understanding of the threat these attacks pose to real-time systems and provides security evaluation benchmarks for future defenses. Therefore, we first conduct a run-time analysis of adversarial generation algorithms. Universal attacks produce a general attack offline, with no online overhead, and can be applied to any input; however, their success rate is limited because of their generality. In contrast, online algorithms, which work on a specific input, are computationally expensive, making them inappropriate for operation under time constraints. Thus, we propose ROOM, a novel Real-time Online-Offline attack construction Model where an offline component serves to warm up the online algorithm, making it possible to generate highly successful attacks under time constraints.

摘要: 深度学习的进展使其具有广泛的应用前景。然而，这些系统很容易受到对抗性机器学习(AML)攻击；对它们的输入进行恶意设计的扰动可能会导致它们错误分类。几个最先进的对抗性攻击已经证明，它们可以可靠地欺骗分类器，使这些攻击成为一个重大威胁。对抗性攻击生成算法主要集中在创建成功的示例，同时控制噪声的大小和分布，使检测变得更加困难。这些攻击的基本假设是敌意噪音是离线生成的，这使得它们的执行时间成为次要考虑因素。然而，最近，攻击者在飞翔上机会主义地生成对抗性示例的即时对抗性攻击已被证明是可能的。本文提出了一个新的问题：如何在实时约束下产生对抗性噪声来支持这种实时对抗性攻击？了解这一问题有助于我们理解这些攻击对实时系统构成的威胁，并为未来的防御提供安全评估基准。因此，我们首先对对抗性生成算法进行运行时分析。通用攻击离线产生一般攻击，没有在线开销，可以应用于任何输入；但是，由于其通用性，其成功率是有限的。相比之下，处理特定输入的在线算法计算成本较高，因此不适合在时间限制下运行。因此，我们提出了一种新颖的实时在线-离线攻击构建模型--ROOM，其中离线组件用于在线算法的预热，使得在时间约束下生成高度成功的攻击成为可能。



## **3. A Survey on Adversarial Attacks for Malware Analysis**

面向恶意软件分析的敌意攻击研究综述 cs.CR

48 Pages, 31 Figures, 11 Tables

**SubmitDate**: 2022-01-05    [paper-pdf](http://arxiv.org/pdf/2111.08223v2)

**Authors**: Kshitiz Aryal, Maanak Gupta, Mahmoud Abdelsalam

**Abstracts**: Machine learning has witnessed tremendous growth in its adoption and advancement in the last decade. The evolution of machine learning from traditional algorithms to modern deep learning architectures has shaped the way today's technology functions. Its unprecedented ability to discover knowledge/patterns from unstructured data and automate the decision-making process led to its application in wide domains. High flying machine learning arena has been recently pegged back by the introduction of adversarial attacks. Adversaries are able to modify data, maximizing the classification error of the models. The discovery of blind spots in machine learning models has been exploited by adversarial attackers by generating subtle intentional perturbations in test samples. Increasing dependency on data has paved the blueprint for ever-high incentives to camouflage machine learning models. To cope with probable catastrophic consequences in the future, continuous research is required to find vulnerabilities in form of adversarial and design remedies in systems. This survey aims at providing the encyclopedic introduction to adversarial attacks that are carried out against malware detection systems. The paper will introduce various machine learning techniques used to generate adversarial and explain the structure of target files. The survey will also model the threat posed by the adversary and followed by brief descriptions of widely accepted adversarial algorithms. Work will provide a taxonomy of adversarial evasion attacks on the basis of attack domain and adversarial generation techniques. Adversarial evasion attacks carried out against malware detectors will be discussed briefly under each taxonomical headings and compared with concomitant researches. Analyzing the current research challenges in an adversarial generation, the survey will conclude by pinpointing the open future research directions.

摘要: 在过去的十年里，机器学习在其采用和发展方面取得了巨大的增长。机器学习从传统算法到现代深度学习体系结构的演变塑造了当今技术的运作方式。它前所未有的从非结构化数据中发现知识/模式并使决策过程自动化的能力使其在广泛的领域得到了应用。最近，由于对抗性攻击的引入，高飞行机器学习领域受到了阻碍。攻击者能够修改数据，最大化模型的分类错误。机器学习模型中盲点的发现已经被敌意攻击者通过在测试样本中产生微妙的故意扰动来利用。对数据的日益依赖已经为伪装机器学习模型的持续高额激励铺平了蓝图。为了应对未来可能出现的灾难性后果，需要不断进行研究，以发现系统中对抗性形式的漏洞和设计补救措施。本调查旨在提供针对恶意软件检测系统进行的敌意攻击的百科全书介绍。本文将介绍用于生成对抗性的各种机器学习技术，并解释目标文件的结构。调查还将模拟对手构成的威胁，随后简要描述被广泛接受的对抗性算法。工作将提供基于攻击域和敌意生成技术的对抗性逃避攻击的分类。针对恶意软件检测器进行的敌意规避攻击将在每个分类标题下简要讨论，并与相应的研究进行比较。通过分析当前对抗性世代的研究挑战，本调查将通过指出开放的未来研究方向来得出结论。



## **4. Fast Gradient Non-sign Methods**

快速梯度无符号方法 cs.CV

**SubmitDate**: 2022-01-05    [paper-pdf](http://arxiv.org/pdf/2110.12734v2)

**Authors**: Yaya Cheng, Xiaosu Zhu, Qilong Zhang, Lianli Gao, Jingkuan Song

**Abstracts**: Adversarial attacks make their success in "fooling" DNNs and among them, gradient-based algorithms become one of the mainstreams. Based on the linearity hypothesis [12], under $\ell_\infty$ constraint, $sign$ operation applied to the gradients is a good choice for generating perturbations. However, the side-effect from such operation exists since it leads to the bias of direction between the real gradients and the perturbations. In other words, current methods contain a gap between real gradients and actual noises, which leads to biased and inefficient attacks. Therefore in this paper, based on the Taylor expansion, the bias is analyzed theoretically and the correction of $\sign$, i.e., Fast Gradient Non-sign Method (FGNM), is further proposed. Notably, FGNM is a general routine, which can seamlessly replace the conventional $sign$ operation in gradient-based attacks with negligible extra computational cost. Extensive experiments demonstrate the effectiveness of our methods. Specifically, ours outperform them by \textbf{27.5\%} at most and \textbf{9.5\%} on average. Our anonymous code is publicly available: \url{https://git.io/mm-fgnm}.

摘要: 敌意攻击使它们成功地“欺骗”了DNN，其中基于梯度的算法成为主流之一。基于线性假设[12]，在$\ell_\infty$约束下，对梯度进行$sign$运算是产生扰动的一个很好的选择。然而，这种操作的副作用是存在的，因为它导致了真实梯度和扰动之间的方向偏差。换言之，目前的方法存在真实梯度和实际噪声之间的差距，这导致了有偏的、低效的攻击。因此，本文在泰勒展开的基础上，对偏差进行了理论分析，并进一步提出了符号修正方法，即快速梯度无符号方法(FGNM)。值得注意的是，FGNM是一个通用例程，在基于梯度的攻击中可以无缝地取代传统的$sign$操作，而额外的计算代价可以忽略不计。大量实验证明了该方法的有效性。具体地说，我们的性能最多比它们高\textbf{27.5\%}，平均\textbf{9.5\%}。我们的匿名代码是公开提供的：\url{https://git.io/mm-fgnm}.



## **5. Adversarial Feature Desensitization**

对抗性特征脱敏 cs.LG

Accepted at Neurips 2021

**SubmitDate**: 2022-01-04    [paper-pdf](http://arxiv.org/pdf/2006.04621v3)

**Authors**: Pouya Bashivan, Reza Bayat, Adam Ibrahim, Kartik Ahuja, Mojtaba Faramarzi, Touraj Laleh, Blake Aaron Richards, Irina Rish

**Abstracts**: Neural networks are known to be vulnerable to adversarial attacks -- slight but carefully constructed perturbations of the inputs which can drastically impair the network's performance. Many defense methods have been proposed for improving robustness of deep networks by training them on adversarially perturbed inputs. However, these models often remain vulnerable to new types of attacks not seen during training, and even to slightly stronger versions of previously seen attacks. In this work, we propose a novel approach to adversarial robustness, which builds upon the insights from the domain adaptation field. Our method, called Adversarial Feature Desensitization (AFD), aims at learning features that are invariant towards adversarial perturbations of the inputs. This is achieved through a game where we learn features that are both predictive and robust (insensitive to adversarial attacks), i.e. cannot be used to discriminate between natural and adversarial data. Empirical results on several benchmarks demonstrate the effectiveness of the proposed approach against a wide range of attack types and attack strengths. Our code is available at https://github.com/BashivanLab/afd.

摘要: 众所周知，神经网络容易受到对抗性攻击--轻微但精心构建的输入扰动会严重损害网络的性能。为了提高深层网络的鲁棒性，已经提出了许多防御方法，通过对深层网络进行恶意扰动输入的训练来提高它们的健壮性。然而，这些模型经常仍然容易受到训练期间未见的新类型攻击的攻击，甚至对以前见过的略强版本的攻击也是如此。在这项工作中，我们提出了一种新的方法来实现对手鲁棒性，该方法建立在领域自适应领域的洞察力的基础上。我们的方法被称为对抗性特征脱敏(AFD)，目的是学习对输入的对抗性扰动不变的特征。这是通过一个游戏来实现的，在这个游戏中，我们学习了既具有预测性又具有健壮性(对敌方攻击不敏感)的功能，即不能用于区分自然数据和对抗性数据。在几个基准测试上的实验结果表明，该方法对大范围的攻击类型和攻击强度都是有效的。我们的代码可在https://github.com/BashivanLab/afd.获得



## **6. On the Minimal Adversarial Perturbation for Deep Neural Networks with Provable Estimation Error**

具有可证明估计误差的深度神经网络的最小对抗性摄动 cs.LG

Under review on IEEE journal Transactions on Pattern Analysis and  Machine Intelligence

**SubmitDate**: 2022-01-04    [paper-pdf](http://arxiv.org/pdf/2201.01235v1)

**Authors**: Fabio Brau, Giulio Rossolini, Alessandro Biondi, Giorgio Buttazzo

**Abstracts**: Although Deep Neural Networks (DNNs) have shown incredible performance in perceptive and control tasks, several trustworthy issues are still open. One of the most discussed topics is the existence of adversarial perturbations, which has opened an interesting research line on provable techniques capable of quantifying the robustness of a given input. In this regard, the Euclidean distance of the input from the classification boundary denotes a well-proved robustness assessment as the minimal affordable adversarial perturbation. Unfortunately, computing such a distance is highly complex due the non-convex nature of NNs. Despite several methods have been proposed to address this issue, to the best of our knowledge, no provable results have been presented to estimate and bound the error committed. This paper addresses this issue by proposing two lightweight strategies to find the minimal adversarial perturbation. Differently from the state-of-the-art, the proposed approach allows formulating an error estimation theory of the approximate distance with respect to the theoretical one. Finally, a substantial set of experiments is reported to evaluate the performance of the algorithms and support the theoretical findings. The obtained results show that the proposed strategies approximate the theoretical distance for samples close to the classification boundary, leading to provable robustness guarantees against any adversarial attacks.

摘要: 尽管深度神经网络(DNNs)在感知和控制任务中表现出了令人难以置信的性能，但一些值得信赖的问题仍然悬而未决。讨论最多的话题之一是对抗性扰动的存在，这为能够量化给定输入的稳健性的可证明技术开辟了一条有趣的研究路线。在这一点上，输入到分类边界的欧几里德距离表示作为最小负担得起的对抗性扰动的健壮性评估。不幸的是，由于神经网络的非凸性，计算这样的距离非常复杂。尽管已经提出了几种方法来解决这个问题，但据我们所知，还没有提出可证明的结果来估计和限制所犯的错误。本文通过提出两种轻量级策略来解决这一问题，以求最小对抗性扰动。与最新的方法不同，该方法允许建立相对于理论距离的近似距离的误差估计理论。最后，通过大量的实验对算法的性能进行了评估，并支持了理论研究结果。结果表明，对于接近分类边界的样本，所提出的策略近似于理论距离，对任何敌意攻击都具有可证明的鲁棒性保证。



## **7. Adversarial Transformation of Spoofing Attacks for Voice Biometrics**

语音生物特征欺骗攻击的对抗性变换 eess.AS

**SubmitDate**: 2022-01-04    [paper-pdf](http://arxiv.org/pdf/2201.01226v1)

**Authors**: Alejandro Gomez-Alanis, Jose A. Gonzalez-Lopez, Antonio M. Peinado

**Abstracts**: Voice biometric systems based on automatic speaker verification (ASV) are exposed to \textit{spoofing} attacks which may compromise their security. To increase the robustness against such attacks, anti-spoofing or presentation attack detection (PAD) systems have been proposed for the detection of replay, synthesis and voice conversion based attacks. Recently, the scientific community has shown that PAD systems are also vulnerable to adversarial attacks. However, to the best of our knowledge, no previous work have studied the robustness of full voice biometrics systems (ASV + PAD) to these new types of adversarial \textit{spoofing} attacks. In this work, we develop a new adversarial biometrics transformation network (ABTN) which jointly processes the loss of the PAD and ASV systems in order to generate white-box and black-box adversarial \textit{spoofing} attacks. The core idea of this system is to generate adversarial \textit{spoofing} attacks which are able to fool the PAD system without being detected by the ASV system. The experiments were carried out on the ASVspoof 2019 corpus, including both logical access (LA) and physical access (PA) scenarios. The experimental results show that the proposed ABTN clearly outperforms some well-known adversarial techniques in both white-box and black-box attack scenarios.

摘要: 基于自动说话人验证(ASV)的语音生物识别系统容易受到威胁其安全性的攻击。为了提高对此类攻击的鲁棒性，已经提出了用于检测基于重放、合成和语音转换的攻击的反欺骗或呈现攻击检测(PAD)系统。最近，科学界表明，PAD系统也容易受到敌意攻击。然而，据我们所知，还没有研究全语音生物特征识别系统(ASV+PAD)对这些新型的敌意\文本{欺骗}攻击的鲁棒性。在这项工作中，我们开发了一个新的对抗性生物特征转换网络(ABTN)，它联合处理PAD和ASV系统的丢失，以产生白盒和黑盒对抗性文本{欺骗}攻击。该系统的核心思想是生成敌意\文本{欺骗}攻击，这些攻击能够欺骗PAD系统而不被ASV系统检测到。这些实验是在ASVspoof 2019年语料库上进行的，包括逻辑访问(LA)和物理访问(PA)场景。实验结果表明，提出的ABTN在白盒和黑盒攻击场景中的性能都明显优于一些著名的对抗性技术。



## **8. Associative Adversarial Learning Based on Selective Attack**

基于选择性攻击的联想对抗性学习 cs.CV

**SubmitDate**: 2022-01-04    [paper-pdf](http://arxiv.org/pdf/2112.13989v2)

**Authors**: Runqi Wang, Xiaoyue Duan, Baochang Zhang, Song Xue, Wentao Zhu, David Doermann, Guodong Guo

**Abstracts**: A human's attention can intuitively adapt to corrupted areas of an image by recalling a similar uncorrupted image they have previously seen. This observation motivates us to improve the attention of adversarial images by considering their clean counterparts. To accomplish this, we introduce Associative Adversarial Learning (AAL) into adversarial learning to guide a selective attack. We formulate the intrinsic relationship between attention and attack (perturbation) as a coupling optimization problem to improve their interaction. This leads to an attention backtracking algorithm that can effectively enhance the attention's adversarial robustness. Our method is generic and can be used to address a variety of tasks by simply choosing different kernels for the associative attention that select other regions for a specific attack. Experimental results show that the selective attack improves the model's performance. We show that our method improves the recognition accuracy of adversarial training on ImageNet by 8.32% compared with the baseline. It also increases object detection mAP on PascalVOC by 2.02% and recognition accuracy of few-shot learning on miniImageNet by 1.63%.

摘要: 人类的注意力可以通过回忆他们以前看到的类似的未被破坏的图像来直观地适应图像的受损区域。这一观察激励我们通过考虑干净的对应物来提高对抗性形象的注意力。为此，我们将联想对抗性学习(AAL)引入到对抗性学习中来指导选择性攻击。我们将注意和攻击(扰动)之间的内在关系描述为一个耦合优化问题，以提高它们之间的交互性。这就产生了一种注意力回溯算法，可以有效地增强注意力的对抗性鲁棒性。我们的方法是通用的，可以通过简单地为关联注意力选择不同的内核来处理各种任务，这些内核为特定的攻击选择其他区域。实验结果表明，选择性攻击提高了模型的性能。实验表明，与基线相比，我们的方法在ImageNet上的对抗性训练的识别准确率提高了8.32%。在PascalVOC上的目标检测地图提高了2.02%，在MiniImageNet上的少镜头学习的识别准确率提高了1.63%。



## **9. Towards Transferable Unrestricted Adversarial Examples with Minimum Changes**

朝向变化最小的可转移的不受限制的对抗性例子 cs.CV

First place in the CVPR'21 Security AI Challenger: Unrestricted  Adversarial Attacks on ImageNet

**SubmitDate**: 2022-01-04    [paper-pdf](http://arxiv.org/pdf/2201.01102v1)

**Authors**: Fangcheng Liu, Chao Zhang, Hongyang Zhang

**Abstracts**: Transfer-based adversarial example is one of the most important classes of black-box attacks. However, there is a trade-off between transferability and imperceptibility of the adversarial perturbation. Prior work in this direction often requires a fixed but large $\ell_p$-norm perturbation budget to reach a good transfer success rate, leading to perceptible adversarial perturbations. On the other hand, most of the current unrestricted adversarial attacks that aim to generate semantic-preserving perturbations suffer from weaker transferability to the target model. In this work, we propose a geometry-aware framework to generate transferable adversarial examples with minimum changes. Analogous to model selection in statistical machine learning, we leverage a validation model to select the optimal perturbation budget for each image under both the $\ell_{\infty}$-norm and unrestricted threat models. Extensive experiments verify the effectiveness of our framework on balancing imperceptibility and transferability of the crafted adversarial examples. The methodology is the foundation of our entry to the CVPR'21 Security AI Challenger: Unrestricted Adversarial Attacks on ImageNet, in which we ranked 1st place out of 1,559 teams and surpassed the runner-up submissions by 4.59% and 23.91% in terms of final score and average image quality level, respectively. Code is available at https://github.com/Equationliu/GA-Attack.

摘要: 基于转移的对抗性例子是黑盒攻击中最重要的一类。然而，在对抗性扰动的可转移性和不可感知性之间存在权衡。以前在这个方向上的工作通常需要固定但很大的$\ell_p$-范数扰动预算才能达到良好的转移成功率，从而导致可感知的对抗性扰动。另一方面，目前大多数旨在产生语义保持扰动的无限制对抗性攻击都存在对目标模型的可移植性较弱的问题。在这项工作中，我们提出了一个几何感知框架，以最小的变化生成可转移的对抗性示例。与统计机器学习中的模型选择类似，我们利用验证模型在$\\ell_{\infty}$-范数和无限制威胁模型下为每幅图像选择最优扰动预算。大量实验验证了该框架在平衡恶意例子的不可感知性和可转移性方面的有效性。这种方法是我们参加CVPR‘21 Security AI Challenger：对ImageNet的无限制对手攻击的基础，在1559个团队中，我们排名第一，在最终得分和平均图像质量水平方面分别以4.59%和23.91%的优势超过了亚军提交的项目。代码可在https://github.com/Equationliu/GA-Attack.上获得



## **10. Exploring Architectural Ingredients of Adversarially Robust Deep Neural Networks**

对抗性鲁棒深度神经网络的结构要素探索 cs.LG

NeurIPS 2021

**SubmitDate**: 2022-01-04    [paper-pdf](http://arxiv.org/pdf/2110.03825v4)

**Authors**: Hanxun Huang, Yisen Wang, Sarah Monazam Erfani, Quanquan Gu, James Bailey, Xingjun Ma

**Abstracts**: Deep neural networks (DNNs) are known to be vulnerable to adversarial attacks. A range of defense methods have been proposed to train adversarially robust DNNs, among which adversarial training has demonstrated promising results. However, despite preliminary understandings developed for adversarial training, it is still not clear, from the architectural perspective, what configurations can lead to more robust DNNs. In this paper, we address this gap via a comprehensive investigation on the impact of network width and depth on the robustness of adversarially trained DNNs. Specifically, we make the following key observations: 1) more parameters (higher model capacity) does not necessarily help adversarial robustness; 2) reducing capacity at the last stage (the last group of blocks) of the network can actually improve adversarial robustness; and 3) under the same parameter budget, there exists an optimal architectural configuration for adversarial robustness. We also provide a theoretical analysis explaning why such network configuration can help robustness. These architectural insights can help design adversarially robust DNNs. Code is available at \url{https://github.com/HanxunH/RobustWRN}.

摘要: 深度神经网络(DNNs)是公认的易受敌意攻击的网络。人们已经提出了一系列的防御方法来训练对抗健壮的DNN，其中对抗训练已经显示出良好的效果。然而，尽管发展了对抗性训练的初步理解，但从体系结构的角度来看，仍然不清楚什么配置可以导致更健壮的DNN。在本文中，我们通过全面调查网络宽度和深度对恶意训练的DNN健壮性的影响来解决这一差距。具体地说，我们做了以下关键的观察：1)更多的参数(更高的模型容量)并不一定有助于对抗鲁棒性；2)在网络的最后阶段(最后一组块)减少容量实际上可以提高对抗鲁棒性；3)在相同的参数预算下，存在对抗鲁棒性的最优结构配置。我们还提供了理论分析，解释了为什么这样的网络配置可以帮助增强健壮性。这些架构洞察力可以帮助设计异常健壮的DNN。代码位于\url{https://github.com/HanxunH/RobustWRN}.



## **11. Towards Understanding and Harnessing the Effect of Image Transformation in Adversarial Detection**

认识和利用图像变换在对抗性检测中的作用 cs.CV

**SubmitDate**: 2022-01-04    [paper-pdf](http://arxiv.org/pdf/2201.01080v1)

**Authors**: Hui Liu, Bo Zhao, Yuefeng Peng, Weidong Li, Peng Liu

**Abstracts**: Deep neural networks (DNNs) are under threat from adversarial examples. Adversarial detection is a fundamental work for robust DNNs-based service, which distinguishes adversarial images from benign images. Image transformation is one of the most effective approaches to detect adversarial examples. During the last few years, a variety of image transformations have been studied and discussed to design reliable adversarial detectors. In this paper, we systematically review the recent progress on adversarial detection via image transformations with a novel taxonomy. Then we conduct an extensive set of experiments to test the detection performance of image transformations towards the state-of-the-art adversarial attacks. Furthermore, we reveal that the single transformation is not capable of detecting robust adversarial examples, and propose an improved approach by combining multiple image transformations. The results show that the joint approach achieves significant improvement in detection accuracy and recall. We suggest that the joint detector is a more effective tool to detect adversarial examples.

摘要: 深度神经网络(DNNs)正受到敌意例子的威胁。敌意检测是健壮的基于DNNs的服务的基础工作，用于区分敌意图像和良性图像。图像变换是检测敌意例子最有效的方法之一。在过去的几年里，人们对各种图像变换进行了研究和讨论，以设计可靠的对抗性检测器。本文以一种新的分类方法，系统地综述了基于图像变换的敌意检测的最新进展。然后，我们进行了大量的实验来测试图像变换对最先进的对抗性攻击的检测性能。此外，我们还揭示了单一变换不能检测出健壮的对抗性示例，并提出了一种结合多个图像变换的改进方法。实验结果表明，该联合方法在检测准确率和召回率方面都有明显的提高。我们认为联合检测器是一种更有效的检测对抗性例子的工具。



## **12. A Survey On Universal Adversarial Attack**

通用对抗性攻击研究综述 cs.LG

Accepted by IJCAI 2021, survey track:  https://www.ijcai.org/proceedings/2021/635

**SubmitDate**: 2022-01-04    [paper-pdf](http://arxiv.org/pdf/2103.01498v2)

**Authors**: Chaoning Zhang, Philipp Benz, Chenguo Lin, Adil Karjauv, Jing Wu, In So Kweon

**Abstracts**: The intriguing phenomenon of adversarial examples has attracted significant attention in machine learning and what might be more surprising to the community is the existence of universal adversarial perturbations (UAPs), i.e. a single perturbation to fool the target DNN for most images. With the focus on UAP against deep classifiers, this survey summarizes the recent progress on universal adversarial attacks, discussing the challenges from both the attack and defense sides, as well as the reason for the existence of UAP. We aim to extend this work as a dynamic survey that will regularly update its content to follow new works regarding UAP or universal attack in a wide range of domains, such as image, audio, video, text, etc. Relevant updates will be discussed at: https://bit.ly/2SbQlLG. We welcome authors of future works in this field to contact us for including your new finding.

摘要: 对抗性例子的有趣现象在机器学习中引起了极大的关注，更令人惊讶的是普遍对抗性扰动(UAP)的存在，即对于大多数图像来说，单一的扰动就可以欺骗目标DNN。本文以UAP对抗深度分类器为重点，总结了通用对抗性攻击的最新进展，讨论了攻防双方面临的挑战，以及UAP存在的原因。我们的目标是将这项工作扩展为一项动态调查，将定期更新其内容，以跟踪有关UAP或广泛领域的通用攻击的新工作，如图像、音频、视频、文本等。相关更新将在以下网址进行讨论：https://bit.ly/2SbQlLG.我们欢迎这一领域未来作品的作者与我们联系，以便将您的新发现包括在内。



## **13. Accelerated Zeroth-Order and First-Order Momentum Methods from Mini to Minimax Optimization**

从Mini到Minimax优化的加速零阶和一阶动量方法 math.OC

Published in Journal of Machine Learning Research (JMLR). arXiv admin  note: text overlap with arXiv:2010.06097

**SubmitDate**: 2022-01-04    [paper-pdf](http://arxiv.org/pdf/2008.08170v5)

**Authors**: Feihu Huang, Shangqian Gao, Jian Pei, Heng Huang

**Abstracts**: In the paper, we propose a class of accelerated zeroth-order and first-order momentum methods for both nonconvex mini-optimization and minimax-optimization. Specifically, we propose a new accelerated zeroth-order momentum (Acc-ZOM) method for black-box mini-optimization. Moreover, we prove that our Acc-ZOM method achieves a lower query complexity of $\tilde{O}(d^{3/4}\epsilon^{-3})$ for finding an $\epsilon$-stationary point, which improves the best known result by a factor of $O(d^{1/4})$ where $d$ denotes the variable dimension. In particular, the Acc-ZOM does not require large batches required in the existing zeroth-order stochastic algorithms. Meanwhile, we propose an accelerated \textbf{zeroth-order} momentum descent ascent (Acc-ZOMDA) method for \textbf{black-box} minimax-optimization, which obtains a query complexity of $\tilde{O}((d_1+d_2)^{3/4}\kappa_y^{4.5}\epsilon^{-3})$ without large batches for finding an $\epsilon$-stationary point, where $d_1$ and $d_2$ denote variable dimensions and $\kappa_y$ is condition number. Moreover, we propose an accelerated \textbf{first-order} momentum descent ascent (Acc-MDA) method for \textbf{white-box} minimax optimization, which has a gradient complexity of $\tilde{O}(\kappa_y^{4.5}\epsilon^{-3})$ without large batches for finding an $\epsilon$-stationary point. In particular, our Acc-MDA can obtain a lower gradient complexity of $\tilde{O}(\kappa_y^{2.5}\epsilon^{-3})$ with a batch size $O(\kappa_y^4)$. Extensive experimental results on the black-box adversarial attack to deep neural networks (DNNs) and poisoning attack demonstrate efficiency of our algorithms.

摘要: 本文提出了一类求解非凸极小优化和极小极大优化的加速零阶和一阶动量方法。具体地说，我们提出了一种新的加速零阶动量(ACC-ZOM)方法来求解黑盒最小优化问题。此外，我们还证明了我们的ACC-zom方法在寻找$\epsilon$-驻点时的查询复杂度为$tilde{O}(d^{3/4}\epsilon^{-3})$，这将最著名的结果改进了$O(d^{1/4})$，其中$d$表示变量维数。特别地，ACC-ZOM不需要现有零阶随机算法所要求的大批量。同时，我们提出了一种求解Textbf{黑盒}极小极大优化问题的加速动量下降上升方法，该方法无需大批量即可获得$\tilde{O}((d_1+d_2)^{3/4}\kappa_y^{4.5}\epsilon^{-3})$的查询复杂度，其中$d1$和$d2$表示可变维数，$kappay$表示条件数。此外，我们还提出了一种加速的textbf{一阶}动量下降上升(ACC-MDA)方法来求解textbf{白盒}极小极大优化问题，该方法的梯度复杂度为$tide{O}(\kappa_y^{4.5}\epsilon^{-3})$，不需要大批量地寻找$\epsilon$-驻点。特别地，在批量为$O(\kappa_y^4)$的情况下，我们的ACC-MDA可以获得$\tide{O}(\kappa_y^{2.5}\epsilon^{-3})$的较低梯度复杂度。针对深度神经网络(DNNs)的黑盒攻击和中毒攻击的大量实验结果证明了该算法的有效性。



## **14. An Adversarial Benchmark for Fake News Detection Models**

一种针对假新闻检测模型的对抗性基准 cs.CL

6 pages, 2 figures, Presented at AAAI 2022, Workshop on Adversarial  Machine Learning and Beyond

**SubmitDate**: 2022-01-03    [paper-pdf](http://arxiv.org/pdf/2201.00912v1)

**Authors**: Lorenzo Jaime Yu Flores, Yiding Hao

**Abstracts**: With the proliferation of online misinformation, fake news detection has gained importance in the artificial intelligence community. In this paper, we propose an adversarial benchmark that tests the ability of fake news detectors to reason about real-world facts. We formulate adversarial attacks that target three aspects of "understanding": compositional semantics, lexical relations, and sensitivity to modifiers. We test our benchmark using BERT classifiers fine-tuned on the LIAR arXiv:arch-ive/1705648 and Kaggle Fake-News datasets, and show that both models fail to respond to changes in compositional and lexical meaning. Our results strengthen the need for such models to be used in conjunction with other fact checking methods.

摘要: 随着网络虚假信息的泛滥，假新闻检测在人工智能领域变得越来越重要。在这篇文章中，我们提出了一个对抗性的基准来测试假新闻检测器对真实世界事实的推理能力。我们制定了针对“理解”的三个方面的对抗性攻击：组成语义、词汇关系和对修饰语的敏感性。我们使用BERT分类器测试了我们的基准，该分类器在说谎者arxiv：ARCH-IVE/1705648和Kaggle假新闻数据集上进行了微调，结果表明这两个模型都不能响应成分意义和词汇意义的变化。我们的结果加强了这样的模型与其他事实核查方法结合使用的必要性。



## **15. Revisiting PGD Attacks for Stability Analysis of Large-Scale Nonlinear Systems and Perception-Based Control**

非线性大系统稳定性分析和基于感知控制的PGD攻击再探 math.OC

Submitted to L4DC 2022

**SubmitDate**: 2022-01-03    [paper-pdf](http://arxiv.org/pdf/2201.00801v1)

**Authors**: Aaron Havens, Darioush Keivan, Peter Seiler, Geir Dullerud, Bin Hu

**Abstracts**: Many existing region-of-attraction (ROA) analysis tools find difficulty in addressing feedback systems with large-scale neural network (NN) policies and/or high-dimensional sensing modalities such as cameras. In this paper, we tailor the projected gradient descent (PGD) attack method developed in the adversarial learning community as a general-purpose ROA analysis tool for large-scale nonlinear systems and end-to-end perception-based control. We show that the ROA analysis can be approximated as a constrained maximization problem whose goal is to find the worst-case initial condition which shifts the terminal state the most. Then we present two PGD-based iterative methods which can be used to solve the resultant constrained maximization problem. Our analysis is not based on Lyapunov theory, and hence requires minimum information of the problem structures. In the model-based setting, we show that the PGD updates can be efficiently performed using back-propagation. In the model-free setting (which is more relevant to ROA analysis of perception-based control), we propose a finite-difference PGD estimate which is general and only requires a black-box simulator for generating the trajectories of the closed-loop system given any initial state. We demonstrate the scalability and generality of our analysis tool on several numerical examples with large-scale NN policies and high-dimensional image observations. We believe that our proposed analysis serves as a meaningful initial step toward further understanding of closed-loop stability of large-scale nonlinear systems and perception-based control.

摘要: 许多现有的吸引区域(ROA)分析工具难以利用大规模神经网络(NN)策略和/或诸如相机的高维传感模态来处理反馈系统。本文针对大规模非线性系统和基于感知的端到端控制，定制了对抗性学习界发展起来的投影梯度下降(PGD)攻击方法，作为一种通用的ROA分析工具。我们证明了ROA分析可以近似为一个约束最大化问题，其目标是找到最坏情况下改变终端状态最多的初始条件。然后，我们提出了两种基于PGD的迭代方法，可以用来求解所得到的约束极大化问题。我们的分析不是基于Lyapunov理论，因此只需要最少的问题结构信息。在基于模型的设置中，我们展示了使用反向传播可以有效地执行PGD更新。在无模型设置下(与基于感知控制的ROA分析更相关)，我们提出了一种通用的有限差分PGD估计，它只需要一个黑盒仿真器就可以生成闭环系统在任何初始状态下的轨迹。我们使用大规模神经网络策略和高维图像观测在几个数值示例上展示了我们的分析工具的可扩展性和通用性。我们相信，我们提出的分析对于进一步理解大规模非线性系统的闭环稳定性和基于感知的控制是有意义的第一步。



## **16. Robust Natural Language Processing: Recent Advances, Challenges, and Future Directions**

鲁棒自然语言处理：最新进展、挑战和未来方向 cs.CL

Survey; 2 figures, 4 tables

**SubmitDate**: 2022-01-03    [paper-pdf](http://arxiv.org/pdf/2201.00768v1)

**Authors**: Marwan Omar, Soohyeon Choi, DaeHun Nyang, David Mohaisen

**Abstracts**: Recent natural language processing (NLP) techniques have accomplished high performance on benchmark datasets, primarily due to the significant improvement in the performance of deep learning. The advances in the research community have led to great enhancements in state-of-the-art production systems for NLP tasks, such as virtual assistants, speech recognition, and sentiment analysis. However, such NLP systems still often fail when tested with adversarial attacks. The initial lack of robustness exposed troubling gaps in current models' language understanding capabilities, creating problems when NLP systems are deployed in real life. In this paper, we present a structured overview of NLP robustness research by summarizing the literature in a systemic way across various dimensions. We then take a deep-dive into the various dimensions of robustness, across techniques, metrics, embeddings, and benchmarks. Finally, we argue that robustness should be multi-dimensional, provide insights into current research, identify gaps in the literature to suggest directions worth pursuing to address these gaps.

摘要: 最近的自然语言处理(NLP)技术已经在基准数据集上实现了高性能，这主要是由于深度学习性能的显著提高。研究界的进步极大地增强了用于NLP任务的最先进的生产系统，例如虚拟助理、语音识别和情感分析。然而，这样的NLP系统在进行对抗性攻击测试时仍然经常失败。最初缺乏健壮性暴露了当前模型的语言理解能力中令人担忧的差距，从而在实际生活中部署NLP系统时产生了问题。在本文中，我们通过系统地总结各个维度的文献，对NLP稳健性研究进行了结构化的综述。然后，我们深入探讨健壮性的各个维度，包括技术、度量、嵌入和基准测试。最后，我们认为稳健性应该是多维的，提供对当前研究的见解，找出文献中的差距，提出解决这些差距的值得追求的方向。



## **17. DeepSight: Mitigating Backdoor Attacks in Federated Learning Through Deep Model Inspection**

DeepSight：通过深度模型检测缓解联合学习中的后门攻击 cs.CR

18 pages, 8 figures; to appear in the Network and Distributed System  Security Symposium (NDSS)

**SubmitDate**: 2022-01-03    [paper-pdf](http://arxiv.org/pdf/2201.00763v1)

**Authors**: Phillip Rieger, Thien Duc Nguyen, Markus Miettinen, Ahmad-Reza Sadeghi

**Abstracts**: Federated Learning (FL) allows multiple clients to collaboratively train a Neural Network (NN) model on their private data without revealing the data. Recently, several targeted poisoning attacks against FL have been introduced. These attacks inject a backdoor into the resulting model that allows adversary-controlled inputs to be misclassified. Existing countermeasures against backdoor attacks are inefficient and often merely aim to exclude deviating models from the aggregation. However, this approach also removes benign models of clients with deviating data distributions, causing the aggregated model to perform poorly for such clients.   To address this problem, we propose DeepSight, a novel model filtering approach for mitigating backdoor attacks. It is based on three novel techniques that allow to characterize the distribution of data used to train model updates and seek to measure fine-grained differences in the internal structure and outputs of NNs. Using these techniques, DeepSight can identify suspicious model updates. We also develop a scheme that can accurately cluster model updates. Combining the results of both components, DeepSight is able to identify and eliminate model clusters containing poisoned models with high attack impact. We also show that the backdoor contributions of possibly undetected poisoned models can be effectively mitigated with existing weight clipping-based defenses. We evaluate the performance and effectiveness of DeepSight and show that it can mitigate state-of-the-art backdoor attacks with a negligible impact on the model's performance on benign data.

摘要: 联合学习(FL)允许多个客户端协作地训练神经网络(NN)模型，以处理他们的私有数据，而不会泄露数据。最近，针对FL的几次有针对性的中毒攻击被引入。这些攻击将后门注入到结果模型中，从而允许对手控制的输入被错误分类。针对后门攻击的现有对策效率低下，通常只是将偏离模型排除在聚合之外。但是，此方法还删除了具有偏差数据分布的客户端的良性模型，从而导致聚合模型对此类客户端的性能较差。针对这一问题，我们提出了一种新的模型过滤方法DeepSight来缓解后门攻击。它基于三种新技术，这三种技术允许表征用于训练模型更新的数据的分布，并寻求测量神经网络内部结构和输出的细粒度差异。使用这些技术，DeepSight可以识别可疑的模型更新。我们还开发了一种能够准确地对模型更新进行聚类的方案。结合这两个组件的结果，DeepSight能够识别和消除包含高攻击影响的有毒模型的模型群集。我们还表明，可能未检测到的中毒模型的后门贡献可以通过现有的基于权重剪切的防御有效地减轻。我们评估了DeepSight的性能和有效性，结果表明，它可以在对良性数据的模型性能影响微乎其微的情况下，缓解最先进的后门攻击。



## **18. Actor-Critic Network for Q&A in an Adversarial Environment**

对抗性环境下的演员-评论家问答网络 cs.CL

6 pages, 3 figures, 3 tables

**SubmitDate**: 2022-01-03    [paper-pdf](http://arxiv.org/pdf/2201.00455v1)

**Authors**: Bejan Sadeghian

**Abstracts**: Significant work has been placed in the Q&A NLP space to build models that are more robust to adversarial attacks. Two key areas of focus are in generating adversarial data for the purposes of training against these situations or modifying existing architectures to build robustness within. This paper introduces an approach that joins these two ideas together to train a critic model for use in an almost reinforcement learning framework. Using the Adversarial SQuAD "Add One Sent" dataset we show that there are some promising signs for this method in protecting against Adversarial attacks.

摘要: 在问答NLP领域已经进行了大量的工作，以构建对对手攻击更健壮的模型。重点关注的两个关键领域是为了针对这些情况进行训练而生成对抗性数据，或者修改现有架构以在其中构建健壮性。本文介绍了一种将这两种思想结合在一起的方法，以训练一个用于几乎强化学习框架的批评模型。使用对抗性小队“添加一个已发送”的数据集，我们显示了这种方法在防御对抗性攻击方面有一些很有前途的迹象。



## **19. Towards Transferable Adversarial Attacks on Vision Transformers**

对视觉变形金刚的可转移敌意攻击 cs.CV

**SubmitDate**: 2022-01-02    [paper-pdf](http://arxiv.org/pdf/2109.04176v3)

**Authors**: Zhipeng Wei, Jingjing Chen, Micah Goldblum, Zuxuan Wu, Tom Goldstein, Yu-Gang Jiang

**Abstracts**: Vision transformers (ViTs) have demonstrated impressive performance on a series of computer vision tasks, yet they still suffer from adversarial examples. % crafted in a similar fashion as CNNs. In this paper, we posit that adversarial attacks on transformers should be specially tailored for their architecture, jointly considering both patches and self-attention, in order to achieve high transferability. More specifically, we introduce a dual attack framework, which contains a Pay No Attention (PNA) attack and a PatchOut attack, to improve the transferability of adversarial samples across different ViTs. We show that skipping the gradients of attention during backpropagation can generate adversarial examples with high transferability. In addition, adversarial perturbations generated by optimizing randomly sampled subsets of patches at each iteration achieve higher attack success rates than attacks using all patches. We evaluate the transferability of attacks on state-of-the-art ViTs, CNNs and robustly trained CNNs. The results of these experiments demonstrate that the proposed dual attack can greatly boost transferability between ViTs and from ViTs to CNNs. In addition, the proposed method can easily be combined with existing transfer methods to boost performance. Code is available at https://github.com/zhipeng-wei/PNA-PatchOut.

摘要: 视觉转换器(VITS)在一系列计算机视觉任务中表现出了令人印象深刻的表现，但它们仍然受到对手例子的影响。%以与CNN类似的方式制作。在这篇文章中，我们假设对变压器的敌意攻击应该针对其体系结构进行专门的定制，同时考虑补丁和自我关注，以实现高可移植性。更具体地说，我们引入了双重攻击框架，其中包括一个PNA攻击和一个PatchOut攻击，以提高敌意样本在不同VITS之间的可传输性。我们表明，在反向传播过程中跳过注意力梯度可以生成具有很高可转移性的敌意示例。此外，在每次迭代中，通过优化随机采样的补丁子集生成的对抗性扰动比使用所有补丁的攻击获得更高的攻击成功率。我们评估了针对最先进的VITS、CNN和训练有素的CNN的攻击的可转移性。实验结果表明，所提出的双重攻击可以极大地提高VITS之间以及VITS到CNN之间的可转移性。此外，所提出的方法可以很容易地与现有的传输方法相结合来提高性能。代码可在https://github.com/zhipeng-wei/PNA-PatchOut.上获得



## **20. Rethinking Feature Uncertainty in Stochastic Neural Networks for Adversarial Robustness**

重新考虑随机神经网络中的特征不确定性以提高对抗鲁棒性 cs.LG

**SubmitDate**: 2022-01-01    [paper-pdf](http://arxiv.org/pdf/2201.00148v1)

**Authors**: Hao Yang, Min Wang, Zhengfei Yu, Yun Zhou

**Abstracts**: It is well-known that deep neural networks (DNNs) have shown remarkable success in many fields. However, when adding an imperceptible magnitude perturbation on the model input, the model performance might get rapid decrease. To address this issue, a randomness technique has been proposed recently, named Stochastic Neural Networks (SNNs). Specifically, SNNs inject randomness into the model to defend against unseen attacks and improve the adversarial robustness. However, existed studies on SNNs mainly focus on injecting fixed or learnable noises to model weights/activations. In this paper, we find that the existed SNNs performances are largely bottlenecked by the feature representation ability. Surprisingly, simply maximizing the variance per dimension of the feature distribution leads to a considerable boost beyond all previous methods, which we named maximize feature distribution variance stochastic neural network (MFDV-SNN). Extensive experiments on well-known white- and black-box attacks show that MFDV-SNN achieves a significant improvement over existing methods, which indicates that it is a simple but effective method to improve model robustness.

摘要: 众所周知，深度神经网络(DNNs)在许多领域都取得了令人瞩目的成功。然而，当在模型输入上加入不可察觉的幅度摄动时，模型的性能可能会迅速下降。为了解决这个问题，最近提出了一种随机性技术，称为随机神经网络(SNNs)。具体地说，SNN将随机性注入到模型中，以防御看不见的攻击，并提高了对手的鲁棒性。然而，已有的SNN研究主要集中在注入固定或可学习的噪声来建模权重/激活。在本文中，我们发现已有的SNN的性能很大程度上受到特征表示能力的制约。令人惊讶的是，简单地最大化特征分布的每维方差比以往的所有方法都有相当大的提高，我们称之为最大化特征分布方差随机神经网络(MFDV-SNN)。在著名的白盒和黑盒攻击上的大量实验表明，MFDV-SNN比现有的方法有明显的改进，这表明它是一种简单而有效的提高模型鲁棒性的方法。



## **21. Characterizing Speech Adversarial Examples Using Self-Attention U-Net Enhancement**

基于自我关注U网增强的言语对抗性实例刻画 eess.AS

The authors have revised some annotations in Table 4 to improve the  clarity. The authors thank reading feedbacks from Jonathan Le Roux. The first  draft was finished in August 2019. Accepted to IEEE ICASSP 2020

**SubmitDate**: 2022-01-01    [paper-pdf](http://arxiv.org/pdf/2003.13917v2)

**Authors**: Chao-Han Huck Yang, Jun Qi, Pin-Yu Chen, Xiaoli Ma, Chin-Hui Lee

**Abstracts**: Recent studies have highlighted adversarial examples as ubiquitous threats to the deep neural network (DNN) based speech recognition systems. In this work, we present a U-Net based attention model, U-Net$_{At}$, to enhance adversarial speech signals. Specifically, we evaluate the model performance by interpretable speech recognition metrics and discuss the model performance by the augmented adversarial training. Our experiments show that our proposed U-Net$_{At}$ improves the perceptual evaluation of speech quality (PESQ) from 1.13 to 2.78, speech transmission index (STI) from 0.65 to 0.75, short-term objective intelligibility (STOI) from 0.83 to 0.96 on the task of speech enhancement with adversarial speech examples. We conduct experiments on the automatic speech recognition (ASR) task with adversarial audio attacks. We find that (i) temporal features learned by the attention network are capable of enhancing the robustness of DNN based ASR models; (ii) the generalization power of DNN based ASR model could be enhanced by applying adversarial training with an additive adversarial data augmentation. The ASR metric on word-error-rates (WERs) shows that there is an absolute 2.22 $\%$ decrease under gradient-based perturbation, and an absolute 2.03 $\%$ decrease, under evolutionary-optimized perturbation, which suggests that our enhancement models with adversarial training can further secure a resilient ASR system.

摘要: 最近的研究强调了对基于深度神经网络(DNN)的语音识别系统的无处不在的威胁。在这项工作中，我们提出了一个基于U-网的注意力模型，U-Net${at}$，用于增强对抗性语音信号。具体地说，我们通过可解释的语音识别度量来评估模型的性能，并通过增强的对抗性训练来讨论模型的性能。实验结果表明，在对抗性语音增强任务中，我们提出的U-Net将语音质量感知评价(PESQ)从1.13提高到2.78，语音传输指数(STI)从0.65提高到0.75，短期客观清晰度(STOI)从0.83提高到0.96。我们在自动语音识别(ASR)任务上进行了对抗性音频攻击的实验。我们发现：(I)注意力网络学习的时间特征能够增强基于DNN的ASR模型的鲁棒性；(Ii)通过应用对抗性训练并增加对抗性数据，可以增强基于DNN的ASR模型的泛化能力。误字率(WERS)的ASR度量表明，在基于梯度的扰动下，ASR系统绝对减少了2.22美元，在进化优化扰动下，绝对减少了2.03美元，这表明我们的对抗性训练增强模型可以进一步确保具有弹性的ASR系统。



## **22. An Attention Score Based Attacker for Black-box NLP Classifier**

一种基于注意力得分的黑盒NLP分类器攻击者 cs.LG

**SubmitDate**: 2022-01-01    [paper-pdf](http://arxiv.org/pdf/2112.11660v2)

**Authors**: Yueyang Liu, Hunmin Lee, Zhipeng Cai

**Abstracts**: Deep neural networks have a wide range of applications in solving various real-world tasks and have achieved satisfactory results, in domains such as computer vision, image classification, and natural language processing. Meanwhile, the security and robustness of neural networks have become imperative, as diverse researches have shown the vulnerable aspects of neural networks. Case in point, in Natural language processing tasks, the neural network may be fooled by an attentively modified text, which has a high similarity to the original one. As per previous research, most of the studies are focused on the image domain; Different from image adversarial attacks, the text is represented in a discrete sequence, traditional image attack methods are not applicable in the NLP field. In this paper, we propose a word-level NLP sentiment classifier attack model, which includes a self-attention mechanism-based word selection method and a greedy search algorithm for word substitution. We experiment with our attack model by attacking GRU and 1D-CNN victim models on IMDB datasets. Experimental results demonstrate that our model achieves a higher attack success rate and more efficient than previous methods due to the efficient word selection algorithms are employed and minimized the word substitute number. Also, our model is transferable, which can be used in the image domain with several modifications.

摘要: 深度神经网络在计算机视觉、图像分类、自然语言处理等领域有着广泛的应用，并取得了令人满意的结果。同时，随着各种研究表明神经网络的脆弱性，神经网络的安全性和鲁棒性也变得势在必行。例如，在自然语言处理任务中，神经网络可能会被精心修改的文本所欺骗，因为它与原始文本具有很高的相似性。根据以往的研究，大多数研究都集中在图像领域；与图像对抗性攻击不同，文本是离散序列表示的，传统的图像攻击方法不适用于自然语言处理领域。本文提出了一种词级NLP情感分类器攻击模型，该模型包括一种基于自我注意机制的选词方法和一种贪婪的词替换搜索算法。我们在IMDB数据集上通过攻击GRU和1D-CNN受害者模型来测试我们的攻击模型。实验结果表明，由于采用了高效的选词算法并最小化了替身的词数，该模型取得了比以往方法更高的攻击成功率和更高的效率。此外，我们的模型是可移植的，可以在图像域中使用，只需做一些修改即可。



## **23. Adversarial Attack via Dual-Stage Network Erosion**

基于双阶段网络侵蚀的对抗性攻击 cs.CV

**SubmitDate**: 2022-01-01    [paper-pdf](http://arxiv.org/pdf/2201.00097v1)

**Authors**: Yexin Duan, Junhua Zou, Xingyu Zhou, Wu Zhang, Jin Zhang, Zhisong Pan

**Abstracts**: Deep neural networks are vulnerable to adversarial examples, which can fool deep models by adding subtle perturbations. Although existing attacks have achieved promising results, it still leaves a long way to go for generating transferable adversarial examples under the black-box setting. To this end, this paper proposes to improve the transferability of adversarial examples, and applies dual-stage feature-level perturbations to an existing model to implicitly create a set of diverse models. Then these models are fused by the longitudinal ensemble during the iterations. The proposed method is termed Dual-Stage Network Erosion (DSNE). We conduct comprehensive experiments both on non-residual and residual networks, and obtain more transferable adversarial examples with the computational cost similar to the state-of-the-art method. In particular, for the residual networks, the transferability of the adversarial examples can be significantly improved by biasing the residual block information to the skip connections. Our work provides new insights into the architectural vulnerability of neural networks and presents new challenges to the robustness of neural networks.

摘要: 深层神经网络容易受到敌意示例的攻击，这些示例可以通过添加微妙的扰动来愚弄深层模型。虽然现有的攻击已经取得了很好的效果，但要在黑箱环境下生成可转移的对抗性实例，还有很长的路要走。为此，本文提出提高对抗性实例的可移植性，并对已有模型进行双阶段特征级扰动，隐式创建一组不同的模型。然后在迭代过程中通过纵向集成对这些模型进行融合。该方法被称为双阶段网络侵蚀(DSNE)。我们在非残差网络和残差网络上进行了全面的实验，获得了更多可移植的对抗性实例，其计算代价与最新的方法相似。特别地，对于残差网络，通过将残馀挡路信息偏向跳过连接，可以显著提高敌意实例的可转移性。我们的工作提供了对神经网络体系结构脆弱性的新见解，并对神经网络的健壮性提出了新的挑战。



## **24. Privacy-Protecting COVID-19 Exposure Notification Based on Cluster Events**

基于集群事件的隐私保护冠状病毒暴露通知 cs.CR

11 pages. This paper was presented at the NIST Workshop on Challenges  for Digital Proximity Detection in Pandemics: Privacy, Accuracy, and Impact,  January 28 02021

**SubmitDate**: 2021-12-31    [paper-pdf](http://arxiv.org/pdf/2201.00031v1)

**Authors**: Paul Syverson

**Abstracts**: We provide a rough sketch of a simple system design for exposure notification of COVID-19 infections based on copresence at cluster events -- locations and times where a threshold number of tested-positive (TP) individuals were present. Unlike other designs, such as DP3T or the Apple-Google exposure-notification system, this design does not track or notify based on detecting direct proximity to TP individuals.   The design makes use of existing or in-development tests for COVID-19 that are relatively cheap and return results in less than an hour, and that have high specificity but may have lower sensitivity. It also uses readily available location tracking for mobile phones and similar devices. It reports events at which TP individuals were present but does not link events with individuals or with other events in an individual's history. Participating individuals are notified of detected cluster events. They can then compare these locally to their own location history. Detected cluster events can be publicized through public channels. Thus, individuals not participating in the reporting system can still be notified of exposure.   A proper security analysis is beyond the scope of this design sketch. We do, however, discuss resistance to various adversaries and attacks on privacy as well as false-reporting attacks.

摘要: 我们提供了一个简单的系统设计的粗略图，该系统用于基于群集事件(检测阳性(TP)个体出现的阈值数量的位置和时间)的共现性进行冠状病毒感染的暴露通知。与其他设计(如DP3T或Apple-Google暴露通知系统)不同，此设计不会基于检测到与TP个人的直接接近来跟踪或通知。该设计利用了现有的或正在开发的冠状病毒测试，这些测试相对便宜，在不到一小时内就能得出结果，而且具有高特异性，但敏感性可能较低。它还对移动电话和类似设备使用现成的位置跟踪。它报告TP个人在场的事件，但不将事件与个人或个人历史中的其他事件联系起来。向参与的个人通知检测到的群集事件。然后，他们可以在本地将这些信息与自己的位置历史记录进行比较。检测到的集群事件可以通过公共渠道公布。因此，没有参与报告系统的个人仍然可以收到关于接触的通知。正确的安全分析超出了本设计草图的范围。然而，我们确实讨论了对各种对手的抵抗、对隐私的攻击以及虚假报告攻击。



## **25. QueryNet: Attack by Multi-Identity Surrogates**

QueryNet：多身份代理的攻击 cs.LG

QueryNet reduces queries by about an order of magnitude against SOTA  black-box attacks

**SubmitDate**: 2021-12-31    [paper-pdf](http://arxiv.org/pdf/2105.15010v3)

**Authors**: Sizhe Chen, Zhehao Huang, Qinghua Tao, Xiaolin Huang

**Abstracts**: Deep Neural Networks (DNNs) are acknowledged as vulnerable to adversarial attacks, while the existing black-box attacks require extensive queries on the victim DNN to achieve high success rates. For query-efficiency, surrogate models of the victim are used to generate transferable Adversarial Examples (AEs) because of their Gradient Similarity (GS), i.e., surrogates' attack gradients are similar to the victim's ones. However, it is generally neglected to exploit their similarity on outputs, namely the Prediction Similarity (PS), to filter out inefficient queries by surrogates without querying the victim. To jointly utilize and also optimize surrogates' GS and PS, we develop QueryNet, a unified attack framework that can significantly reduce queries. QueryNet creatively attacks by multi-identity surrogates, i.e., crafts several AEs for one sample by different surrogates, and also uses surrogates to decide on the most promising AE for the query. After that, the victim's query feedback is accumulated to optimize not only surrogates' parameters but also their architectures, enhancing both the GS and the PS. Although QueryNet has no access to pre-trained surrogates' prior, it reduces queries by averagely about an order of magnitude compared to alternatives within an acceptable time, according to our comprehensive experiments: 11 victims (including two commercial models) on MNIST/CIFAR10/ImageNet, allowing only 8-bit image queries, and no access to the victim's training data. The code is available at https://github.com/AllenChen1998/QueryNet.

摘要: 深度神经网络(DNNs)是公认的易受敌意攻击的网络，而现有的黑盒攻击需要对受害者DNN进行广泛的查询才能获得较高的成功率。为提高查询效率，利用受害者代理模型的梯度相似性(GS)，即代理攻击梯度与受害者攻击梯度相似，生成可转移的对抗性实例(AES)。然而，通常忽略了利用它们在输出上的相似性，即预测相似度(PS)来进行过滤的低效查询，而不是在没有查询受害者的情况下通过代理来发出低效的查询。为了联合利用和优化代理的GS和PS，我们开发了QueryNet，这是一个可以显著减少查询的统一攻击框架。QueryNet创造性地利用多身份代理进行攻击，即通过不同的代理为一个样本制作多个AE，并使用代理来决定最有希望的查询AE。然后，累积受害者的查询反馈，不仅优化代理的参数，而且优化它们的体系结构，从而提高GS和PS。虽然QueryNet无法访问预先训练的代理人的先前图像，但根据我们的综合实验：11名受害者(包括两个商业模型)在MNIST/CIFAR10/ImageNet上仅允许8位图像查询，并且无法访问受害者的训练数据，与其他选择相比，它在可接受的时间内平均减少了关于一个数量级的查询。代码可在https://github.com/AllenChen1998/QueryNet.上获得



## **26. NCIS: Neural Contextual Iterative Smoothing for Purifying Adversarial Perturbations**

NCIS：净化对抗性扰动的神经上下文迭代平滑 cs.CV

Preprint version

**SubmitDate**: 2021-12-30    [paper-pdf](http://arxiv.org/pdf/2106.11644v2)

**Authors**: Sungmin Cha, Naeun Ko, Youngjoon Yoo, Taesup Moon

**Abstracts**: We propose a novel and effective purification based adversarial defense method against pre-processor blind white- and black-box attacks. Our method is computationally efficient and trained only with self-supervised learning on general images, without requiring any adversarial training or retraining of the classification model. We first show an empirical analysis on the adversarial noise, defined to be the residual between an original image and its adversarial example, has almost zero mean, symmetric distribution. Based on this observation, we propose a very simple iterative Gaussian Smoothing (GS) which can effectively smooth out adversarial noise and achieve substantially high robust accuracy. To further improve it, we propose Neural Contextual Iterative Smoothing (NCIS), which trains a blind-spot network (BSN) in a self-supervised manner to reconstruct the discriminative features of the original image that is also smoothed out by GS. From our extensive experiments on the large-scale ImageNet using four classification models, we show that our method achieves both competitive standard accuracy and state-of-the-art robust accuracy against most strong purifier-blind white- and black-box attacks. Also, we propose a new benchmark for evaluating a purification method based on commercial image classification APIs, such as AWS, Azure, Clarifai and Google. We generate adversarial examples by ensemble transfer-based black-box attack, which can induce complete misclassification of APIs, and demonstrate that our method can be used to increase adversarial robustness of APIs.

摘要: 针对预处理器盲白盒和黑盒攻击，提出了一种新颖有效的基于净化的对抗性防御方法。我们的方法在计算上是有效的，并且只需要对一般图像进行自监督学习就可以进行训练，而不需要对分类模型进行任何对抗性训练或重新训练。我们首先对对抗性噪声进行了实证分析，该噪声定义为原始图像与其对抗性样本之间的残差，具有几乎为零的平均对称分布。基于这一观察结果，我们提出了一种非常简单的迭代高斯平滑(GS)算法，它可以有效地平滑对抗性噪声，并获得相当高的鲁棒精度。为了进一步改善这一点，我们提出了神经上下文迭代平滑(NCIS)，它以自监督的方式训练盲点网络(BSN)来重建原始图像的鉴别特征，这些特征也被GS平滑了。从我们使用四种分类模型对大规模图像网进行的大量实验中，我们表明我们的方法在抵抗大多数强净化盲白盒和黑盒攻击时都达到了好胜标准的准确率和最新的鲁棒准确率。结果表明，对于大多数强净化盲白盒和黑盒攻击，我们的方法都达到了好胜标准的准确率和最新的鲁棒准确率。此外，我们还提出了一个新的基准来评估基于商业图像分类API的净化方法，如AWS、Azure、Clarifai和Google。我们通过基于集成传输的黑盒攻击生成敌意实例，这会导致API的完全错误分类，并证明了我们的方法可以用来提高API的敌意健壮性。



## **27. Efficient Robust Training via Backward Smoothing**

基于向后平滑的高效鲁棒训练 cs.LG

12 pages, 15 tables, 6 figures. In AAAI 2022

**SubmitDate**: 2021-12-30    [paper-pdf](http://arxiv.org/pdf/2010.01278v2)

**Authors**: Jinghui Chen, Yu Cheng, Zhe Gan, Quanquan Gu, Jingjing Liu

**Abstracts**: Adversarial training is so far the most effective strategy in defending against adversarial examples. However, it suffers from high computational costs due to the iterative adversarial attacks in each training step. Recent studies show that it is possible to achieve fast Adversarial Training by performing a single-step attack with random initialization. However, such an approach still lags behind state-of-the-art adversarial training algorithms on both stability and model robustness. In this work, we develop a new understanding towards Fast Adversarial Training, by viewing random initialization as performing randomized smoothing for better optimization of the inner maximization problem. Following this new perspective, we also propose a new initialization strategy, backward smoothing, to further improve the stability and model robustness over single-step robust training methods. Experiments on multiple benchmarks demonstrate that our method achieves similar model robustness as the original TRADES method while using much less training time ($\sim$3x improvement with the same training schedule).

摘要: 对抗性训练是迄今为止防御对抗性例子最有效的策略。然而，由于在每个训练步骤中都要进行迭代的对抗性攻击，该算法存在计算代价高的问题。最近的研究表明，通过随机初始化进行单步攻击可以实现快速的对抗性训练。然而，这种方法在稳定性和模型鲁棒性方面仍然落后于最先进的对抗性训练算法。在这项工作中，我们对快速对抗性训练有了新的理解，将随机初始化看作是为了更好地优化内极大化问题而进行的随机平滑。遵循这一新的观点，我们还提出了一种新的初始化策略-向后平滑，以进一步提高单步鲁棒训练方法的稳定性和模型鲁棒性。在多个基准上的实验表明，我们的方法获得了与原始TRADS方法相似的模型鲁棒性，而使用的训练时间要少得多(在相同的训练时间表下，改进了$\sim$3倍)。



## **28. Improved Gradient based Adversarial Attacks for Quantized Networks**

改进的基于梯度的量化网络敌意攻击 cs.CV

AAAI 2022

**SubmitDate**: 2021-12-29    [paper-pdf](http://arxiv.org/pdf/2003.13511v2)

**Authors**: Kartik Gupta, Thalaiyasingam Ajanthan

**Abstracts**: Neural network quantization has become increasingly popular due to efficient memory consumption and faster computation resulting from bitwise operations on the quantized networks. Even though they exhibit excellent generalization capabilities, their robustness properties are not well-understood. In this work, we systematically study the robustness of quantized networks against gradient based adversarial attacks and demonstrate that these quantized models suffer from gradient vanishing issues and show a fake sense of robustness. By attributing gradient vanishing to poor forward-backward signal propagation in the trained network, we introduce a simple temperature scaling approach to mitigate this issue while preserving the decision boundary. Despite being a simple modification to existing gradient based adversarial attacks, experiments on multiple image classification datasets with multiple network architectures demonstrate that our temperature scaled attacks obtain near-perfect success rate on quantized networks while outperforming original attacks on adversarially trained models as well as floating-point networks. Code is available at https://github.com/kartikgupta-at-anu/attack-bnn.

摘要: 神经网络量化由于对量化网络进行逐位运算所产生的高效的内存消耗和更快的计算速度而变得越来越流行。尽管它们表现出很好的泛化能力，但它们的健壮性并没有得到很好的理解。在这项工作中，我们系统地研究了量化网络对基于梯度的攻击的健壮性，并证明了这些量化模型存在梯度消失问题，并且表现出一种虚假的健壮性。通过将梯度消失归因于前向后向信号在训练网络中的不良传播，我们引入了一种简单的温度缩放方法来缓解这一问题，同时保持决策边界。尽管对已有的基于梯度的敌意攻击进行了简单的改进，但在具有多种网络结构的多图像分类数据集上的实验表明，我们的温度缩放攻击在量化网络上获得了近乎完美的成功率，而在对抗性训练的模型和浮点网络上的性能却优于原有的攻击。代码可在https://github.com/kartikgupta-at-anu/attack-bnn.上获得



## **29. Perfectly Secure Message Transmission against Rational Adversaries**

针对Rational对手的完全安全的消息传输 cs.CR

**SubmitDate**: 2021-12-29    [paper-pdf](http://arxiv.org/pdf/2009.07513v3)

**Authors**: Maiki Fujita, Takeshi Koshiba, Kenji Yasunaga

**Abstracts**: Secure Message Transmission (SMT) is a two-party cryptographic protocol by which the sender can securely and reliably transmit messages to the receiver using multiple channels. An adversary can corrupt a subset of the channels and commit eavesdropping and tampering attacks over the channels. In this work, we introduce a game-theoretic security model for SMT in which adversaries have some preferences for protocol execution. We define rational "timid" adversaries who prefer to violate security requirements but do not prefer the tampering to be detected.   First, we consider the basic setting where a single adversary attacks the protocol. We construct perfect SMT protocols against any rational adversary corrupting all but one of the channels. Since minority corruption is required in the traditional setting, our results demonstrate a way of circumventing the cryptographic impossibility results by a game-theoretic approach.   Next, we study the setting in which all the channels can be corrupted by multiple adversaries who do not cooperate. Since we cannot hope for any security if a single adversary corrupts all the channels or multiple adversaries cooperate maliciously, the scenario can arise from a game-theoretic model. We also study the scenario in which both malicious and rational adversaries exist.

摘要: 安全消息传输(SMT)是一种双方加密协议，通过该协议，发送方可以使用多个通道将消息安全可靠地传输到接收方。敌手可以破坏信道的子集，并在信道上进行窃听和篡改攻击。在这项工作中，我们介绍了一个博弈论的SMT安全模型，在该模型中，攻击者对协议执行有一定的偏好。我们定义理性的“胆小”的对手，他们喜欢违反安全需求，但不希望篡改被检测到。首先，我们考虑单个对手攻击协议的基本设置。我们构建了完美的SMT协议，防止任何理性的对手破坏除一个渠道之外的所有渠道。由于在传统设置中需要少数人腐败，我们的结果展示了一种通过博弈论方法规避密码不可能性结果的方法。接下来，我们研究所有通道都可能被多个不合作的对手破坏的设置。由于我们不能指望在单个对手破坏所有渠道或多个对手恶意合作的情况下有任何安全保障，因此这种情况可能来自博弈论模型。我们还研究了恶意和理性对手同时存在的情况。



## **30. Domain Knowledge Alleviates Adversarial Attacks in Multi-Label Classifiers**

领域知识减轻多标签分类器中的敌意攻击 cs.LG

Accepted for publications in IEEE TPAMI journal

**SubmitDate**: 2021-12-29    [paper-pdf](http://arxiv.org/pdf/2006.03833v4)

**Authors**: Stefano Melacci, Gabriele Ciravegna, Angelo Sotgiu, Ambra Demontis, Battista Biggio, Marco Gori, Fabio Roli

**Abstracts**: Adversarial attacks on machine learning-based classifiers, along with defense mechanisms, have been widely studied in the context of single-label classification problems. In this paper, we shift the attention to multi-label classification, where the availability of domain knowledge on the relationships among the considered classes may offer a natural way to spot incoherent predictions, i.e., predictions associated to adversarial examples lying outside of the training data distribution. We explore this intuition in a framework in which first-order logic knowledge is converted into constraints and injected into a semi-supervised learning problem. Within this setting, the constrained classifier learns to fulfill the domain knowledge over the marginal distribution, and can naturally reject samples with incoherent predictions. Even though our method does not exploit any knowledge of attacks during training, our experimental analysis surprisingly unveils that domain-knowledge constraints can help detect adversarial examples effectively, especially if such constraints are not known to the attacker.

摘要: 针对基于机器学习的分类器的对抗性攻击以及防御机制，已经在单标签分类问题的背景下得到了广泛的研究。在本文中，我们将注意力转移到多标签分类上，其中关于所考虑的类之间关系的领域知识的可用性可以提供一种自然的方法来发现不一致的预测，即与位于训练数据分布之外的对抗性示例相关联的预测。我们在一个框架中探索这种直觉，在这个框架中，一阶逻辑知识被转换为约束，并注入到半监督学习问题中。在这种情况下，受约束的分类器学习满足边缘分布上的领域知识，并且可以自然地拒绝具有不一致预测的样本。尽管我们的方法在训练期间没有利用任何攻击知识，但我们的实验分析令人惊讶地发现，领域知识约束可以帮助有效地检测敌意示例，特别是在攻击者不知道这样的约束的情况下。



## **31. Invertible Image Dataset Protection**

可逆图像数据集保护 cs.CV

Submitted to ICME 2022. Authors are from University of Science and  Technology of China, Fudan University, China. A potential extended version of  this work is under way

**SubmitDate**: 2021-12-29    [paper-pdf](http://arxiv.org/pdf/2112.14420v1)

**Authors**: Kejiang Chen, Xianhan Zeng, Qichao Ying, Sheng Li, Zhenxing Qian, Xinpeng Zhang

**Abstracts**: Deep learning has achieved enormous success in various industrial applications. Companies do not want their valuable data to be stolen by malicious employees to train pirated models. Nor do they wish the data analyzed by the competitors after using them online. We propose a novel solution for dataset protection in this scenario by robustly and reversibly transform the images into adversarial images. We develop a reversible adversarial example generator (RAEG) that introduces slight changes to the images to fool traditional classification models. Even though malicious attacks train pirated models based on the defensed versions of the protected images, RAEG can significantly weaken the functionality of these models. Meanwhile, the reversibility of RAEG ensures the performance of authorized models. Extensive experiments demonstrate that RAEG can better protect the data with slight distortion against adversarial defense than previous methods.

摘要: 深度学习在各种工业应用中取得了巨大的成功。公司不希望他们的宝贵数据被恶意员工窃取，以训练盗版模特。他们也不希望数据在网上使用后被竞争对手分析。在这种情况下，我们提出了一种新的数据集保护方案，将图像鲁棒且可逆地变换为对抗性图像。我们开发了一个可逆的对抗性示例生成器(RAEG)，它通过对图像进行细微的修改来愚弄传统的分类模型。即使恶意攻击基于受保护映像的防御版本训练盗版模型，RAEG也会显著削弱这些模型的功能。同时，RAEG的可逆性保证了授权模型的性能。大量实验表明，与以往的方法相比，RAEG能够更好地保护具有轻微失真的数据抵抗敌意防御。



## **32. Super-Efficient Super Resolution for Fast Adversarial Defense at the Edge**

用于边缘快速对抗防御的超高效超分辨率 eess.IV

This preprint is for personal use only. The official article will  appear in proceedings of Design, Automation & Test in Europe (DATE), 2022, as  part of the Special Initiative on Autonomous Systems Design (ASD)

**SubmitDate**: 2021-12-29    [paper-pdf](http://arxiv.org/pdf/2112.14340v1)

**Authors**: Kartikeya Bhardwaj, Dibakar Gope, James Ward, Paul Whatmough, Danny Loh

**Abstracts**: Autonomous systems are highly vulnerable to a variety of adversarial attacks on Deep Neural Networks (DNNs). Training-free model-agnostic defenses have recently gained popularity due to their speed, ease of deployment, and ability to work across many DNNs. To this end, a new technique has emerged for mitigating attacks on image classification DNNs, namely, preprocessing adversarial images using super resolution -- upscaling low-quality inputs into high-resolution images. This defense requires running both image classifiers and super resolution models on constrained autonomous systems. However, super resolution incurs a heavy computational cost. Therefore, in this paper, we investigate the following question: Does the robustness of image classifiers suffer if we use tiny super resolution models? To answer this, we first review a recent work called Super-Efficient Super Resolution (SESR) that achieves similar or better image quality than prior art while requiring 2x to 330x fewer Multiply-Accumulate (MAC) operations. We demonstrate that despite being orders of magnitude smaller than existing models, SESR achieves the same level of robustness as significantly larger networks. Finally, we estimate end-to-end performance of super resolution-based defenses on a commercial Arm Ethos-U55 micro-NPU. Our findings show that SESR achieves nearly 3x higher FPS than a baseline while achieving similar robustness.

摘要: 自治系统极易受到深度神经网络(DNNs)的各种敌意攻击。无需培训的模型不可知防御系统最近因其速度快、易于部署以及能够跨多个DNN工作而广受欢迎。为此，出现了一种缓解图像分类DNN攻击的新技术，即利用超分辨率对敌方图像进行预处理--将低质量输入提升为高分辨率图像。这种防御需要在受限的自治系统上同时运行图像分类器和超分辨率模型。然而，超分辨率带来了很大的计算量。因此，在本文中，我们研究了以下问题：如果我们使用微小的超分辨率模型，图像分类器的鲁棒性是否会受到影响？为了回答这个问题，我们首先回顾了最近的一项名为超高效超分辨率(SESR)的工作，该工作实现了与现有技术相似或更好的图像质量，而所需的乘法累加(MAC)运算减少了2倍到330倍。我们证明，尽管SESR比现有模型小几个数量级，但它实现了与大得多的网络相同的健壮性水平。最后，我们评估了基于超分辨率的防御在商用ARM ethos-U55微型NPU上的端到端性能。我们的研究结果表明，SESR在实现类似健壮性的同时，实现了比基准高近3倍的FPS。



## **33. DeepAdversaries: Examining the Robustness of Deep Learning Models for Galaxy Morphology Classification**

深度事件：检验银河系形态分类深度学习模型的稳健性 cs.LG

19 pages, 7 figures, 5 tables, submitted to Astronomy & Computing

**SubmitDate**: 2021-12-28    [paper-pdf](http://arxiv.org/pdf/2112.14299v1)

**Authors**: Aleksandra Ćiprijanović, Diana Kafkes, Gregory Snyder, F. Javier Sánchez, Gabriel Nathan Perdue, Kevin Pedro, Brian Nord, Sandeep Madireddy, Stefan M. Wild

**Abstracts**: Data processing and analysis pipelines in cosmological survey experiments introduce data perturbations that can significantly degrade the performance of deep learning-based models. Given the increased adoption of supervised deep learning methods for processing and analysis of cosmological survey data, the assessment of data perturbation effects and the development of methods that increase model robustness are increasingly important. In the context of morphological classification of galaxies, we study the effects of perturbations in imaging data. In particular, we examine the consequences of using neural networks when training on baseline data and testing on perturbed data. We consider perturbations associated with two primary sources: 1) increased observational noise as represented by higher levels of Poisson noise and 2) data processing noise incurred by steps such as image compression or telescope errors as represented by one-pixel adversarial attacks. We also test the efficacy of domain adaptation techniques in mitigating the perturbation-driven errors. We use classification accuracy, latent space visualizations, and latent space distance to assess model robustness. Without domain adaptation, we find that processing pixel-level errors easily flip the classification into an incorrect class and that higher observational noise makes the model trained on low-noise data unable to classify galaxy morphologies. On the other hand, we show that training with domain adaptation improves model robustness and mitigates the effects of these perturbations, improving the classification accuracy by 23% on data with higher observational noise. Domain adaptation also increases by a factor of ~2.3 the latent space distance between the baseline and the incorrectly classified one-pixel perturbed image, making the model more robust to inadvertent perturbations.

摘要: 宇宙测量实验中的数据处理和分析管道引入了数据扰动，这会显著降低基于深度学习的模型的性能。考虑到越来越多的人采用有监督的深度学习方法来处理和分析宇宙测量数据，数据扰动效应的评估和提高模型稳健性的方法的开发变得越来越重要。在星系形态分类的背景下，我们研究了摄动对成像数据的影响。特别地，当训练基线数据和测试扰动数据时，我们检查使用神经网络的后果。我们考虑与两个主要来源相关的扰动：1)以更高水平的泊松噪声为代表的增加的观测噪声；2)由诸如图像压缩或望远镜误差等步骤引起的数据处理噪声，以单像素对抗性攻击为代表。我们还测试了领域自适应技术在减轻扰动驱动的错误方面的有效性。我们使用分类精度、潜在空间可视化和潜在空间距离来评估模型的稳健性。在没有域自适应的情况下，我们发现处理像素级误差很容易将分类反转到错误的类别，并且较高的观测噪声使得基于低噪声数据训练的模型无法对星系形态进行分类。另一方面，我们表明，域自适应训练提高了模型的鲁棒性，减轻了这些扰动的影响，在观测噪声较高的数据上，分类精度提高了23%。域自适应还将基线和错误分类的单像素扰动图像之间的潜在空间距离增加了约2.3倍，使模型对无意扰动更具鲁棒性。



## **34. Constrained Gradient Descent: A Powerful and Principled Evasion Attack Against Neural Networks**

约束梯度下降：一种强大的、原则性的神经网络规避攻击 cs.LG

**SubmitDate**: 2021-12-28    [paper-pdf](http://arxiv.org/pdf/2112.14232v1)

**Authors**: Weiran Lin, Keane Lucas, Lujo Bauer, Michael K. Reiter, Mahmood Sharif

**Abstracts**: Minimal adversarial perturbations added to inputs have been shown to be effective at fooling deep neural networks. In this paper, we introduce several innovations that make white-box targeted attacks follow the intuition of the attacker's goal: to trick the model to assign a higher probability to the target class than to any other, while staying within a specified distance from the original input. First, we propose a new loss function that explicitly captures the goal of targeted attacks, in particular, by using the logits of all classes instead of just a subset, as is common. We show that Auto-PGD with this loss function finds more adversarial examples than it does with other commonly used loss functions. Second, we propose a new attack method that uses a further developed version of our loss function capturing both the misclassification objective and the $L_{\infty}$ distance limit $\epsilon$. This new attack method is relatively 1.5--4.2% more successful on the CIFAR10 dataset and relatively 8.2--14.9% more successful on the ImageNet dataset, than the next best state-of-the-art attack. We confirm using statistical tests that our attack outperforms state-of-the-art attacks on different datasets and values of $\epsilon$ and against different defenses.

摘要: 添加到输入中的最小对抗性扰动已被证明在愚弄深度神经网络方面是有效的。在本文中，我们介绍了几个创新，使白盒定向攻击遵循攻击者的目标的直觉：欺骗模型赋予目标类比任何其他类更高的概率，同时保持在与原始输入的指定距离内。首先，我们提出了一种新的损失函数，它明确地捕捉了目标攻击的目标，特别是通过使用所有类别的日志而不是通常使用的一个子集。我们表明，与其他常用的损失函数相比，使用该损失函数的Auto-PGD能够找到更多的对抗性例子。其次，我们提出了一种新的攻击方法，它使用我们的损失函数的进一步发展版本来捕捉误分类目标和距离限制$\epsilon$。这种新的攻击方法在CIFAR10数据集上的成功率相对较高1.5-4.2%，在ImageNet数据集上的成功率相对较高8.2-14.9%，比第二好的最先进的攻击方法要高出1.2%-14.9%。我们使用统计测试证实，我们的攻击在不同数据集和$\epsilon$的值以及不同防御系统上的性能优于最先进的攻击。



## **35. Understanding and Measuring Robustness of Multimodal Learning**

理解和测量多模态学习的稳健性 cs.LG

**SubmitDate**: 2021-12-28    [paper-pdf](http://arxiv.org/pdf/2112.12792v2)

**Authors**: Nishant Vishwamitra, Hongxin Hu, Ziming Zhao, Long Cheng, Feng Luo

**Abstracts**: The modern digital world is increasingly becoming multimodal. Although multimodal learning has recently revolutionized the state-of-the-art performance in multimodal tasks, relatively little is known about the robustness of multimodal learning in an adversarial setting. In this paper, we introduce a comprehensive measurement of the adversarial robustness of multimodal learning by focusing on the fusion of input modalities in multimodal models, via a framework called MUROAN (MUltimodal RObustness ANalyzer). We first present a unified view of multimodal models in MUROAN and identify the fusion mechanism of multimodal models as a key vulnerability. We then introduce a new type of multimodal adversarial attacks called decoupling attack in MUROAN that aims to compromise multimodal models by decoupling their fused modalities. We leverage the decoupling attack of MUROAN to measure several state-of-the-art multimodal models and find that the multimodal fusion mechanism in all these models is vulnerable to decoupling attacks. We especially demonstrate that, in the worst case, the decoupling attack of MUROAN achieves an attack success rate of 100% by decoupling just 1.16% of the input space. Finally, we show that traditional adversarial training is insufficient to improve the robustness of multimodal models with respect to decoupling attacks. We hope our findings encourage researchers to pursue improving the robustness of multimodal learning.

摘要: 现代数字世界正日益变得多式联运。虽然多模态学习最近彻底改变了多模态任务的最新表现，但人们对多模态学习在对抗性环境中的稳健性知之甚少。本文通过一个称为MUROAN(多模态鲁棒性分析器)的框架，以多模态模型中输入模态的融合为重点，介绍了一种多模态学习对抗鲁棒性的综合度量方法。我们首先给出了MUROAN中多模态模型的统一视图，并指出多模态模型的融合机制是一个关键漏洞。然后，我们在MUROAN中引入了一种新的多模态对抗性攻击，称为解耦攻击，其目的是通过解耦多模态模型来折衷多模态模型。我们利用MUROAN的解耦攻击对几种最新的多模态模型进行了测试，发现这些模型中的多模态融合机制都容易受到解耦攻击。我们特别证明了，在最坏的情况下，MUROAN的解耦攻击只需要解耦1.16%的输入空间就可以达到100%的攻击成功率。最后，我们证明了传统的对抗性训练不足以提高多模态模型对于解耦攻击的鲁棒性。我们希望我们的发现能鼓励研究人员提高多模态学习的稳健性。



## **36. Mind Your Solver! On Adversarial Attack and Defense for Combinatorial Optimization**

当心你的解算器！论组合优化中的对抗性攻防 math.OC

**SubmitDate**: 2021-12-28    [paper-pdf](http://arxiv.org/pdf/2201.00402v1)

**Authors**: Han Lu, Zenan Li, Runzhong Wang, Qibing Ren, Junchi Yan, Xiaokang Yang

**Abstracts**: Combinatorial optimization (CO) is a long-standing challenging task not only in its inherent complexity (e.g. NP-hard) but also the possible sensitivity to input conditions. In this paper, we take an initiative on developing the mechanisms for adversarial attack and defense towards combinatorial optimization solvers, whereby the solver is treated as a black-box function and the original problem's underlying graph structure (which is often available and associated with the problem instance, e.g. DAG, TSP) is attacked under a given budget. In particular, we present a simple yet effective defense strategy to modify the graph structure to increase the robustness of solvers, which shows its universal effectiveness across tasks and solvers.

摘要: 组合优化(CO)是一个长期具有挑战性的任务，不仅因为其固有的复杂性(例如NP-hard)，而且可能对输入条件敏感。在本文中，我们主动提出了针对组合优化求解器的对抗性攻击和防御机制，将求解器视为一个黑箱函数，并在给定的预算下攻击原始问题的底层图结构(通常与问题实例相关联，如DAG，TSP)。特别地，我们提出了一种简单而有效的防御策略来修改图结构以增加求解器的健壮性，这表明了其跨任务和求解器的普遍有效性。



## **37. Boosting the Transferability of Video Adversarial Examples via Temporal Translation**

通过时间翻译提高视频对抗性例句的可转移性 cs.CV

**SubmitDate**: 2021-12-28    [paper-pdf](http://arxiv.org/pdf/2110.09075v2)

**Authors**: Zhipeng Wei, Jingjing Chen, Zuxuan Wu, Yu-Gang Jiang

**Abstracts**: Although deep-learning based video recognition models have achieved remarkable success, they are vulnerable to adversarial examples that are generated by adding human-imperceptible perturbations on clean video samples. As indicated in recent studies, adversarial examples are transferable, which makes it feasible for black-box attacks in real-world applications. Nevertheless, most existing adversarial attack methods have poor transferability when attacking other video models and transfer-based attacks on video models are still unexplored. To this end, we propose to boost the transferability of video adversarial examples for black-box attacks on video recognition models. Through extensive analysis, we discover that different video recognition models rely on different discriminative temporal patterns, leading to the poor transferability of video adversarial examples. This motivates us to introduce a temporal translation attack method, which optimizes the adversarial perturbations over a set of temporal translated video clips. By generating adversarial examples over translated videos, the resulting adversarial examples are less sensitive to temporal patterns existed in the white-box model being attacked and thus can be better transferred. Extensive experiments on the Kinetics-400 dataset and the UCF-101 dataset demonstrate that our method can significantly boost the transferability of video adversarial examples. For transfer-based attack against video recognition models, it achieves a 61.56% average attack success rate on the Kinetics-400 and 48.60% on the UCF-101. Code is available at https://github.com/zhipeng-wei/TT.

摘要: 尽管基于深度学习的视频识别模型已经取得了显著的成功，但它们很容易受到通过在干净的视频样本上添加人类无法察觉的扰动而产生的敌意示例的影响。最近的研究表明，敌意例子是可移植的，这使得黑盒攻击在现实世界中的应用是可行的。然而，现有的大多数对抗性攻击方法在攻击其他视频模型时可移植性较差，基于传输的视频模型攻击仍未被探索。为此，我们建议提高针对视频识别模型的黑盒攻击的视频对抗性示例的可转移性。通过广泛的分析，我们发现不同的视频识别模型依赖于不同的区分时间模式，导致视频对抗性实例的可移植性较差。这促使我们引入了一种时间翻译攻击方法，该方法优化了一组时间翻译视频片段上的对抗性扰动。通过在翻译的视频上生成对抗性示例，生成的对抗性示例对被攻击的白盒模型中存在的时间模式不那么敏感，因此可以更好地传输。在Kinetics-400数据集和UCF-101数据集上的大量实验表明，我们的方法可以显著提高视频对抗性例子的可转移性。针对视频识别模型的基于传输的攻击，在Kinetics-400和UCF-101上的平均攻击成功率分别为61.56%和48.60%。代码可在https://github.com/zhipeng-wei/TT.上获得



## **38. On anti-stochastic properties of unlabeled graphs**

关于无标号图的反随机性 cs.DM

**SubmitDate**: 2021-12-28    [paper-pdf](http://arxiv.org/pdf/2112.04395v2)

**Authors**: Sergei Kiselev, Andrey Kupavskii, Oleg Verbitsky, Maksim Zhukovskii

**Abstracts**: We study vulnerability of a uniformly distributed random graph to an attack by an adversary who aims for a global change of the distribution while being able to make only a local change in the graph. We call a graph property $A$ anti-stochastic if the probability that a random graph $G$ satisfies $A$ is small but, with high probability, there is a small perturbation transforming $G$ into a graph satisfying $A$. While for labeled graphs such properties are easy to obtain from binary covering codes, the existence of anti-stochastic properties for unlabeled graphs is not so evident. If an admissible perturbation is either the addition or the deletion of one edge, we exhibit an anti-stochastic property that is satisfied by a random unlabeled graph of order $n$ with probability $(2+o(1))/n^2$, which is as small as possible. We also express another anti-stochastic property in terms of the degree sequence of a graph. This property has probability $(2+o(1))/(n\ln n)$, which is optimal up to factor of 2.

摘要: 我们研究了均匀分布随机图在遭受敌手攻击时的脆弱性，该敌手的目标是改变分布的全局，但只能对图进行局部改变。如果一个随机图$G$满足$A$的概率很小，但在很高的概率下，存在一个将$G$转换成满足$A$的图的小扰动，我们称图性质$A$是反随机的。而对于有标号的图，这样的性质很容易从二元覆盖码中获得，而无标号图的反随机性的存在就不那么明显了。如果一个允许的扰动是增加或删除一条边，我们表现出一个反随机性质，它由一个概率为$(2+o(1))/n^2$的n阶随机无标号图所满足，它是尽可能小的。我们还用图的度序列来表示另一个反随机性质。该属性的概率为$(2+o(1))/(n\ln)$，最优为因子2。



## **39. PORTFILER: Port-Level Network Profiling for Self-Propagating Malware Detection**

PORTFILER：用于自传播恶意软件检测的端口级网络分析 cs.CR

An earlier version is accepted to be published in IEEE Conference on  Communications and Network Security (CNS) 2021

**SubmitDate**: 2021-12-27    [paper-pdf](http://arxiv.org/pdf/2112.13798v1)

**Authors**: Talha Ongun, Oliver Spohngellert, Benjamin Miller, Simona Boboila, Alina Oprea, Tina Eliassi-Rad, Jason Hiser, Alastair Nottingham, Jack Davidson, Malathi Veeraraghavan

**Abstracts**: Recent self-propagating malware (SPM) campaigns compromised hundred of thousands of victim machines on the Internet. It is challenging to detect these attacks in their early stages, as adversaries utilize common network services, use novel techniques, and can evade existing detection mechanisms. We propose PORTFILER (PORT-Level Network Traffic ProFILER), a new machine learning system applied to network traffic for detecting SPM attacks. PORTFILER extracts port-level features from the Zeek connection logs collected at a border of a monitored network, applies anomaly detection techniques to identify suspicious events, and ranks the alerts across ports for investigation by the Security Operations Center (SOC). We propose a novel ensemble methodology for aggregating individual models in PORTFILER that increases resilience against several evasion strategies compared to standard ML baselines. We extensively evaluate PORTFILER on traffic collected from two university networks, and show that it can detect SPM attacks with different patterns, such as WannaCry and Mirai, and performs well under evasion. Ranking across ports achieves precision over 0.94 with low false positive rates in the top ranked alerts. When deployed on the university networks, PORTFILER detected anomalous SPM-like activity on one of the campus networks, confirmed by the university SOC as malicious. PORTFILER also detected a Mirai attack recreated on the two university networks with higher precision and recall than deep-learning-based autoencoder methods.

摘要: 最近的自我传播恶意软件(SPM)活动危害了互联网上数十万受攻击的计算机。在攻击的早期阶段检测这些攻击是具有挑战性的，因为攻击者利用普通的网络服务，使用新的技术，并且可以逃避现有的检测机制。本文提出了PORTFILER(Port-Level Network Traffic Profiler)，一种新的应用于网络流量检测SPM攻击的机器学习系统PORTFILER(Port-Level Network Traffic Profiler)。PORTFILER从在受监控网络边界收集的Zeek连接日志中提取端口级特征，应用异常检测技术来识别可疑事件，并跨端口对警报进行排序，以供安全运营中心(SOC)调查。我们提出了一种新的集成方法，用于在PORTFILER中聚合单个模型，与标准ML基线相比，该方法提高了对几种规避策略的弹性。我们对PORTFILER在两个大学网络上收集的流量进行了广泛的评估，结果表明它可以检测出不同模式的SPM攻击，如WannaCry和Mirai，并且在规避的情况下表现得很好。跨端口排名可实现0.94以上的精确度，且排名靠前的警报的误报率较低。当PORTFILER部署在大学网络上时，它在其中一个校园网上检测到异常的类似SPM的活动，并被大学SOC确认为恶意活动。PORTFILER还检测到在两个大学网络上重新创建的Mirai攻击，与基于深度学习的自动编码器方法相比，具有更高的精确度和召回率。



## **40. Adversarial Attack for Asynchronous Event-based Data**

异步事件数据的敌意攻击 cs.CV

8 pages, 6 figures, Thirty-Sixth AAAI Conference on Artificial  Intelligence (AAAI-22)

**SubmitDate**: 2021-12-27    [paper-pdf](http://arxiv.org/pdf/2112.13534v1)

**Authors**: Wooju Lee, Hyun Myung

**Abstracts**: Deep neural networks (DNNs) are vulnerable to adversarial examples that are carefully designed to cause the deep learning model to make mistakes. Adversarial examples of 2D images and 3D point clouds have been extensively studied, but studies on event-based data are limited. Event-based data can be an alternative to a 2D image under high-speed movements, such as autonomous driving. However, the given adversarial events make the current deep learning model vulnerable to safety issues. In this work, we generate adversarial examples and then train the robust models for event-based data, for the first time. Our algorithm shifts the time of the original events and generates additional adversarial events. Additional adversarial events are generated in two stages. First, null events are added to the event-based data to generate additional adversarial events. The perturbation size can be controlled with the number of null events. Second, the location and time of additional adversarial events are set to mislead DNNs in a gradient-based attack. Our algorithm achieves an attack success rate of 97.95\% on the N-Caltech101 dataset. Furthermore, the adversarial training model improves robustness on the adversarial event data compared to the original model.

摘要: 深度神经网络(DNNs)很容易受到精心设计的敌意示例的攻击，从而导致深度学习模型出错。二维图像和三维点云的对抗性例子已经得到了广泛的研究，但对基于事件的数据的研究却很有限。在高速运动(例如自动驾驶)下，基于事件的数据可以替代2D图像。然而，给定的对抗性事件使得当前的深度学习模型容易受到安全问题的影响。在这项工作中，我们首次生成对抗性示例，然后训练基于事件的数据的鲁棒模型。我们的算法移动了原始事件的时间，并生成了额外的对抗性事件。其他对抗性事件分两个阶段生成。首先，将空事件添加到基于事件的数据以生成附加的对抗性事件。可以通过空事件的数量来控制扰动大小。其次，设置附加对抗性事件的位置和时间以在基于梯度的攻击中误导DNN。该算法在N-Caltech101数据集上的攻击成功率为97.95。此外，与原始模型相比，该对抗性训练模型提高了对对抗性事件数据的鲁棒性。



## **41. Killing One Bird with Two Stones: Model Extraction and Attribute Inference Attacks against BERT-based APIs**

一举两得：对基于BERT的API的模型提取和属性推理攻击 cs.CR

**SubmitDate**: 2021-12-26    [paper-pdf](http://arxiv.org/pdf/2105.10909v2)

**Authors**: Chen Chen, Xuanli He, Lingjuan Lyu, Fangzhao Wu

**Abstracts**: The collection and availability of big data, combined with advances in pre-trained models (e.g., BERT, XLNET, etc), have revolutionized the predictive performance of modern natural language processing tasks, ranging from text classification to text generation. This allows corporations to provide machine learning as a service (MLaaS) by encapsulating fine-tuned BERT-based models as APIs. However, BERT-based APIs have exhibited a series of security and privacy vulnerabilities. For example, prior work has exploited the security issues of the BERT-based APIs through the adversarial examples crafted by the extracted model. However, the privacy leakage problems of the BERT-based APIs through the extracted model have not been well studied. On the other hand, due to the high capacity of BERT-based APIs, the fine-tuned model is easy to be overlearned, but what kind of information can be leaked from the extracted model remains unknown. In this work, we bridge this gap by first presenting an effective model extraction attack, where the adversary can practically steal a BERT-based API (the target/victim model) by only querying a limited number of queries. We further develop an effective attribute inference attack which can infer the sensitive attribute of the training data used by the BERT-based APIs. Our extensive experiments on benchmark datasets under various realistic settings validate the potential vulnerabilities of BERT-based APIs. Moreover, we demonstrate that two promising defense methods become ineffective against our attacks, which calls for more effective defense methods.

摘要: 大数据的收集和可用性，与预先训练的模型(例如，BERT、XLNET等)的进步相结合，彻底改变了从文本分类到文本生成的现代自然语言处理任务的预测性能。这允许公司通过将微调的基于ERT的模型封装为API来提供机器学习即服务(MLaaS)。然而，基于BERT的API出现了一系列安全和隐私漏洞。例如，以前的工作已经通过提取的模型制作的敌意示例来利用基于BERT的API的安全问题。然而，基于ERT的API通过提取模型的隐私泄露问题还没有得到很好的研究。另一方面，由于基于BERT的API容量大，微调后的模型容易过度学习，但从提取的模型中能泄露出什么样的信息还是个未知数。在这项工作中，我们通过首先提出一种有效的模型提取攻击来弥补这一差距，在这种攻击中，攻击者实际上可以通过查询有限数量的查询来窃取基于BERT的API(目标/受害者模型)。我们进一步开发了一种有效的属性推理攻击，可以推断基于ERT的API所使用的训练数据的敏感属性。我们在各种现实环境下的基准数据集上进行了大量的实验，验证了基于ERT的API的潜在漏洞。此外，我们还证明了两种很有前途的防御方法对我们的攻击无效，这就需要更有效的防御方法。



## **42. Task and Model Agnostic Adversarial Attack on Graph Neural Networks**

基于图神经网络的任务和模型不可知的敌意攻击 cs.LG

**SubmitDate**: 2021-12-25    [paper-pdf](http://arxiv.org/pdf/2112.13267v1)

**Authors**: Kartik Sharma, Samidha Verma, Sourav Medya, Sayan Ranu, Arnab Bhattacharya

**Abstracts**: Graph neural networks (GNNs) have witnessed significant adoption in the industry owing to impressive performance on various predictive tasks. Performance alone, however, is not enough. Any widely deployed machine learning algorithm must be robust to adversarial attacks. In this work, we investigate this aspect for GNNs, identify vulnerabilities, and link them to graph properties that may potentially lead to the development of more secure and robust GNNs. Specifically, we formulate the problem of task and model agnostic evasion attacks where adversaries modify the test graph to affect the performance of any unknown downstream task. The proposed algorithm, GRAND ($Gr$aph $A$ttack via $N$eighborhood $D$istortion) shows that distortion of node neighborhoods is effective in drastically compromising prediction performance. Although neighborhood distortion is an NP-hard problem, GRAND designs an effective heuristic through a novel combination of Graph Isomorphism Network with deep $Q$-learning. Extensive experiments on real datasets show that, on average, GRAND is up to $50\%$ more effective than state of the art techniques, while being more than $100$ times faster.

摘要: 图形神经网络(GNNs)由于在各种预测任务上的出色性能，在行业中得到了广泛的采用。然而，仅有业绩是不够的。任何广泛部署的机器学习算法都必须对敌意攻击具有健壮性。在这项工作中，我们研究GNN的这一方面，识别漏洞，并将它们链接到可能导致开发更安全和健壮的GNN的图属性。具体地说，我们描述了任务问题和模型不可知性逃避攻击，其中攻击者修改测试图以影响任何未知下游任务的性能。提出的GRAND($Gr$APH$A$ttack via$N$80 borhood$D$istortion)算法表明，节点邻域失真对预测性能的影响是有效的。虽然邻域失真是一个NP-hard问题，但Gland通过将图同构网络与深度$Q学习相结合，设计了一种有效的启发式算法。在真实数据集上的广泛实验表明，平均而言，GRAND比最先进的技术效率高达50美元，同时速度快100美元以上。



## **43. Denoised Internal Models: a Brain-Inspired Autoencoder against Adversarial Attacks**

去噪内部模型：一种抗敌意攻击的脑启发自动编码器 cs.CV

16 pages, 3 figures

**SubmitDate**: 2021-12-25    [paper-pdf](http://arxiv.org/pdf/2111.10844v3)

**Authors**: Kaiyuan Liu, Xingyu Li, Yurui Lai, Ge Zhang, Hang Su, Jiachen Wang, Chunxu Guo, Jisong Guan, Yi Zhou

**Abstracts**: Despite its great success, deep learning severely suffers from robustness; that is, deep neural networks are very vulnerable to adversarial attacks, even the simplest ones. Inspired by recent advances in brain science, we propose the Denoised Internal Models (DIM), a novel generative autoencoder-based model to tackle this challenge. Simulating the pipeline in the human brain for visual signal processing, DIM adopts a two-stage approach. In the first stage, DIM uses a denoiser to reduce the noise and the dimensions of inputs, reflecting the information pre-processing in the thalamus. Inspired from the sparse coding of memory-related traces in the primary visual cortex, the second stage produces a set of internal models, one for each category. We evaluate DIM over 42 adversarial attacks, showing that DIM effectively defenses against all the attacks and outperforms the SOTA on the overall robustness.

摘要: 尽管深度学习取得了巨大的成功，但它的健壮性严重不足；也就是说，深度神经网络非常容易受到对手的攻击，即使是最简单的攻击。受脑科学最新进展的启发，我们提出了去噪内部模型(DIM)，这是一种新颖的基于生成式自动编码器的模型，以应对撞击的这一挑战。模拟人脑中视觉信号处理的管道，DIM采用了两个阶段的方法。在第一阶段，DIM使用去噪器来降低输入的噪声和维数，反映了丘脑的信息预处理。第二阶段的灵感来自于初级视觉皮层中与记忆相关的痕迹的稀疏编码，第二阶段产生了一组内部模型，每个类别一个。我们对DIM42个对抗性攻击进行了评估，结果表明，DIM有效地防御了所有攻击，并且在整体鲁棒性上优于SOTA。



## **44. Stealthy Attack on Algorithmic-Protected DNNs via Smart Bit Flipping**

通过智能位翻转对受算法保护的DNN的隐蔽攻击 cs.CR

Accepted for the 23rd International Symposium on Quality Electronic  Design (ISQED'22)

**SubmitDate**: 2021-12-25    [paper-pdf](http://arxiv.org/pdf/2112.13162v1)

**Authors**: Behnam Ghavami, Seyd Movi, Zhenman Fang, Lesley Shannon

**Abstracts**: Recently, deep neural networks (DNNs) have been deployed in safety-critical systems such as autonomous vehicles and medical devices. Shortly after that, the vulnerability of DNNs were revealed by stealthy adversarial examples where crafted inputs -- by adding tiny perturbations to original inputs -- can lead a DNN to generate misclassification outputs. To improve the robustness of DNNs, some algorithmic-based countermeasures against adversarial examples have been introduced thereafter.   In this paper, we propose a new type of stealthy attack on protected DNNs to circumvent the algorithmic defenses: via smart bit flipping in DNN weights, we can reserve the classification accuracy for clean inputs but misclassify crafted inputs even with algorithmic countermeasures. To fool protected DNNs in a stealthy way, we introduce a novel method to efficiently find their most vulnerable weights and flip those bits in hardware. Experimental results show that we can successfully apply our stealthy attack against state-of-the-art algorithmic-protected DNNs.

摘要: 最近，深度神经网络(DNNs)已经被部署在自动驾驶汽车和医疗设备等安全关键系统中。在那之后不久，DNN的脆弱性通过隐蔽的敌意例子暴露出来，在这些例子中，精心编制的输入-通过向原始输入添加微小扰动-可以导致DNN生成错误分类输出。为了提高DNNs的鲁棒性，在此之后引入了一些基于算法的对抗敌意示例的对策。本文提出了一种新型的针对受保护DNN的隐蔽攻击，以规避算法防御：通过DNN权重中的智能位翻转，我们可以为干净的输入保留分类精度，但即使使用算法对策，也可以错误地对精心制作的输入进行分类。为了隐蔽地欺骗受保护的DNN，我们引入了一种新的方法来有效地找到它们最易受攻击的权重并在硬件中翻转这些位。实验结果表明，我们可以成功地对最先进的受算法保护的DNNs进行隐蔽攻击。



## **45. SoK: A Study of the Security on Voice Processing Systems**

SOK：语音处理系统的安全性研究 cs.CR

**SubmitDate**: 2021-12-24    [paper-pdf](http://arxiv.org/pdf/2112.13144v1)

**Authors**: Robert Chang, Logan Kuo, Arthur Liu, Nader Sehatbakhsh

**Abstracts**: As the use of Voice Processing Systems (VPS) continues to become more prevalent in our daily lives through the increased reliance on applications such as commercial voice recognition devices as well as major text-to-speech software, the attacks on these systems are increasingly complex, varied, and constantly evolving. With the use cases for VPS rapidly growing into new spaces and purposes, the potential consequences regarding privacy are increasingly more dangerous. In addition, the growing number and increased practicality of over-the-air attacks have made system failures much more probable. In this paper, we will identify and classify an arrangement of unique attacks on voice processing systems. Over the years research has been moving from specialized, untargeted attacks that result in the malfunction of systems and the denial of services to more general, targeted attacks that can force an outcome controlled by an adversary. The current and most frequently used machine learning systems and deep neural networks, which are at the core of modern voice processing systems, were built with a focus on performance and scalability rather than security. Therefore, it is critical for us to reassess the developing voice processing landscape and to identify the state of current attacks and defenses so that we may suggest future developments and theoretical improvements.

摘要: 随着对商用语音识别设备和主要的文本到语音转换软件等应用程序的日益依赖，语音处理系统(VPS)的使用在我们的日常生活中继续变得更加普遍，针对这些系统的攻击也变得越来越复杂、多样和不断发展。随着VPS的使用案例迅速发展到新的空间和目的，有关隐私的潜在后果也越来越危险。此外，越来越多的空中攻击和越来越多的实用性使得系统故障的可能性大大增加。在本文中，我们将对一系列针对语音处理系统的独特攻击进行识别和分类。多年来，研究已经从导致系统故障和拒绝服务的专门的、无针对性的攻击转移到更一般的、有针对性的攻击，这些攻击可以迫使对手控制结果。当前和最常用的机器学习系统和深度神经网络是现代语音处理系统的核心，其构建的重点是性能和可扩展性，而不是安全性。因此，对我们来说，重新评估正在发展的语音处理环境并识别当前攻击和防御的状态是至关重要的，这样我们就可以为未来的发展和理论改进提供建议。



## **46. CatchBackdoor: Backdoor Testing by Critical Trojan Neural Path Identification via Differential Fuzzing**

CatchBackDoor：基于差分模糊识别关键木马神经路径的后门测试 cs.CR

13 pages

**SubmitDate**: 2021-12-24    [paper-pdf](http://arxiv.org/pdf/2112.13064v1)

**Authors**: Haibo Jin, Ruoxi Chen, Jinyin Chen, Yao Cheng, Chong Fu, Ting Wang, Yue Yu, Zhaoyan Ming

**Abstracts**: The success of deep neural networks (DNNs) in real-world applications has benefited from abundant pre-trained models. However, the backdoored pre-trained models can pose a significant trojan threat to the deployment of downstream DNNs. Existing DNN testing methods are mainly designed to find incorrect corner case behaviors in adversarial settings but fail to discover the backdoors crafted by strong trojan attacks. Observing the trojan network behaviors shows that they are not just reflected by a single compromised neuron as proposed by previous work but attributed to the critical neural paths in the activation intensity and frequency of multiple neurons. This work formulates the DNN backdoor testing and proposes the CatchBackdoor framework. Via differential fuzzing of critical neurons from a small number of benign examples, we identify the trojan paths and particularly the critical ones, and generate backdoor testing examples by simulating the critical neurons in the identified paths. Extensive experiments demonstrate the superiority of CatchBackdoor, with higher detection performance than existing methods. CatchBackdoor works better on detecting backdoors by stealthy blending and adaptive attacks, which existing methods fail to detect. Moreover, our experiments show that CatchBackdoor may reveal the potential backdoors of models in Model Zoo.

摘要: 深度神经网络(DNNs)在实际应用中的成功得益于丰富的预训练模型。然而，后退的预先训练的模型可能会对下游DNN的部署构成重大的特洛伊木马威胁。现有的DNN测试方法主要用于发现敌方环境中的不正确角例行为，而无法发现强木马攻击所造成的后门。对特洛伊木马网络行为的观察表明，木马网络行为并不像前人所说的那样只反映在单个受损神经元身上，而是归因于多个神经元激活强度和频率的关键神经路径。本文对DNN后门测试进行了阐述，提出了CatchBackdoor框架。通过对少量良性样本中的关键神经元进行差分模糊化，识别木马路径，特别是关键路径，并通过模拟识别路径中的关键神经元生成后门测试用例。大量实验证明了CatchBackdoor的优越性，其检测性能优于现有的检测方法。CatchBackdoor通过隐形混合和自适应攻击来更好地检测后门，这是现有方法无法检测到的。此外，我们的实验表明，CatchBackdoor可能会揭示模型动物园中模型的潜在后门。



## **47. NIP: Neuron-level Inverse Perturbation Against Adversarial Attacks**

NIP：对抗对抗性攻击的神经元级逆摄动 cs.CV

14 pages

**SubmitDate**: 2021-12-24    [paper-pdf](http://arxiv.org/pdf/2112.13060v1)

**Authors**: Ruoxi Chen, Haibo Jin, Jinyin Chen, Haibin Zheng, Yue Yu, Shouling Ji

**Abstracts**: Although deep learning models have achieved unprecedented success, their vulnerabilities towards adversarial attacks have attracted increasing attention, especially when deployed in security-critical domains. To address the challenge, numerous defense strategies, including reactive and proactive ones, have been proposed for robustness improvement. From the perspective of image feature space, some of them cannot reach satisfying results due to the shift of features. Besides, features learned by models are not directly related to classification results. Different from them, We consider defense method essentially from model inside and investigated the neuron behaviors before and after attacks. We observed that attacks mislead the model by dramatically changing the neurons that contribute most and least to the correct label. Motivated by it, we introduce the concept of neuron influence and further divide neurons into front, middle and tail part. Based on it, we propose neuron-level inverse perturbation(NIP), the first neuron-level reactive defense method against adversarial attacks. By strengthening front neurons and weakening those in the tail part, NIP can eliminate nearly all adversarial perturbations while still maintaining high benign accuracy. Besides, it can cope with different sizes of perturbations via adaptivity, especially larger ones. Comprehensive experiments conducted on three datasets and six models show that NIP outperforms the state-of-the-art baselines against eleven adversarial attacks. We further provide interpretable proofs via neuron activation and visualization for better understanding.

摘要: 尽管深度学习模型取得了前所未有的成功，但它们对敌意攻击的脆弱性引起了越来越多的关注，特别是当它们部署在安全关键领域时。为了应对这一挑战，已经提出了许多防御策略，包括反应性和主动性策略，以提高健壮性。从图像特征空间的角度来看，有些算法由于特征的偏移而不能达到令人满意的结果。此外，模型学习的特征与分类结果没有直接关系。与它们不同的是，我们主要从模型内部考虑防御方法，并研究了攻击前后的神经元行为。我们观察到，攻击极大地改变了对正确标签贡献最大和最少的神经元，从而误导了模型。受此启发，我们引入了神经元影响的概念，并将神经元进一步划分为前、中、尾三个部分。在此基础上，我们提出了第一种针对敌意攻击的神经元级反应性防御方法--神经元级逆摄动(NIP)。通过加强前部神经元和弱化尾部神经元，NIP可以消除几乎所有的对抗性扰动，同时仍然保持较高的良性准确率。此外，它还可以通过自适应来应对不同大小的扰动，特别是较大的扰动。在三个数据集和六个模型上进行的综合实验表明，NIP在对抗11个对手攻击时的性能优于最新的基线。为了更好地理解，我们进一步通过神经元激活和可视化提供了可解释的证据。



## **48. One Bad Apple Spoils the Bunch: Transaction DoS in MimbleWimble Blockchains**

一个坏苹果抢走了一群人：MimbleWimble区块链中的交易DoS cs.CR

9 pages, 4 figures

**SubmitDate**: 2021-12-24    [paper-pdf](http://arxiv.org/pdf/2112.13009v1)

**Authors**: Seyed Ali Tabatabaee, Charlene Nicer, Ivan Beschastnikh, Chen Feng

**Abstracts**: As adoption of blockchain-based systems grows, more attention is being given to privacy of these systems. Early systems like BitCoin provided few privacy features. As a result, systems with strong privacy guarantees, including Monero, Zcash, and MimbleWimble have been developed. Compared to BitCoin, these cryptocurrencies are much less understood. In this paper, we focus on MimbleWimble, which uses the Dandelion++ protocol for private transaction relay and transaction aggregation to provide transaction content privacy. We find that in combination these two features make MimbleWimble susceptible to a new type of denial-of-service attacks. We design, prototype, and evaluate this attack on the Beam network using a private test network and a network simulator. We find that by controlling only 10% of the network nodes, the adversary can prevent over 45% of all transactions from ending up in the blockchain. We also discuss several potential approaches for mitigating this attack.

摘要: 随着基于区块链的系统的采用越来越多，这些系统的隐私问题受到了更多的关注。像比特币这样的早期系统几乎没有提供隐私功能。因此，已经开发出具有强大隐私保障的系统，包括Monero、Zash和MimbleWimble。与比特币相比，人们对这些加密货币的了解要少得多。本文重点研究了MimbleWimble，它使用蒲公英++协议进行私有事务中继和事务聚合，以提供事务内容的私密性。我们发现，结合这两个功能，MimbleWimble很容易受到一种新型的拒绝服务攻击。我们利用一个专用测试网络和一个网络模拟器设计、原型并评估了该攻击在BEAM网络上的性能。我们发现，通过仅控制10%的网络节点，对手可以阻止超过45%的交易最终进入区块链。我们还讨论了缓解这种攻击的几种可能的方法。



## **49. Parameter identifiability of a deep feedforward ReLU neural network**

一种深度前馈RELU神经网络的参数可辨识性 math.ST

**SubmitDate**: 2021-12-24    [paper-pdf](http://arxiv.org/pdf/2112.12982v1)

**Authors**: Joachim Bona-Pellissier, François Bachoc, François Malgouyres

**Abstracts**: The possibility for one to recover the parameters-weights and biases-of a neural network thanks to the knowledge of its function on a subset of the input space can be, depending on the situation, a curse or a blessing. On one hand, recovering the parameters allows for better adversarial attacks and could also disclose sensitive information from the dataset used to construct the network. On the other hand, if the parameters of a network can be recovered, it guarantees the user that the features in the latent spaces can be interpreted. It also provides foundations to obtain formal guarantees on the performances of the network. It is therefore important to characterize the networks whose parameters can be identified and those whose parameters cannot. In this article, we provide a set of conditions on a deep fully-connected feedforward ReLU neural network under which the parameters of the network are uniquely identified-modulo permutation and positive rescaling-from the function it implements on a subset of the input space.

摘要: 根据情况，由于知道神经网络在输入空间的子集上的功能，恢复神经网络的参数(权重和偏差)的可能性可能是诅咒，也可能是祝福。一方面，恢复参数可以进行更好的对抗性攻击，还可能泄露用于构建网络的数据集中的敏感信息。另一方面，如果能够恢复网络的参数，则可以保证用户能够解释潜在空间中的特征。它还为获得对网络性能的正式保证提供了基础。因此，重要的是要对其参数可以识别和参数不能识别的网络进行表征。本文给出了一个深度全连通的前馈RELU神经网络的一组条件，在该条件下，网络的参数可以从它在输入空间子集上实现的函数中唯一地识别出来-模置换和正重标度。



## **50. Revisiting and Advancing Fast Adversarial Training Through The Lens of Bi-Level Optimization**

用双层优化镜头重温和推进快速对抗性训练 cs.LG

**SubmitDate**: 2021-12-24    [paper-pdf](http://arxiv.org/pdf/2112.12376v2)

**Authors**: Yihua Zhang, Guanhua Zhang, Prashant Khanduri, Mingyi Hong, Shiyu Chang, Sijia Liu

**Abstracts**: Adversarial training (AT) has become a widely recognized defense mechanism to improve the robustness of deep neural networks against adversarial attacks. It solves a min-max optimization problem, where the minimizer (i.e., defender) seeks a robust model to minimize the worst-case training loss in the presence of adversarial examples crafted by the maximizer (i.e., attacker). However, the min-max nature makes AT computationally intensive and thus difficult to scale. Meanwhile, the FAST-AT algorithm, and in fact many recent algorithms that improve AT, simplify the min-max based AT by replacing its maximization step with the simple one-shot gradient sign based attack generation step. Although easy to implement, FAST-AT lacks theoretical guarantees, and its practical performance can be unsatisfactory, suffering from the robustness catastrophic overfitting when training with strong adversaries.   In this paper, we propose to design FAST-AT from the perspective of bi-level optimization (BLO). We first make the key observation that the most commonly-used algorithmic specification of FAST-AT is equivalent to using some gradient descent-type algorithm to solve a bi-level problem involving a sign operation. However, the discrete nature of the sign operation makes it difficult to understand the algorithm performance. Based on the above observation, we propose a new tractable bi-level optimization problem, design and analyze a new set of algorithms termed Fast Bi-level AT (FAST-BAT). FAST-BAT is capable of defending sign-based projected gradient descent (PGD) attacks without calling any gradient sign method and explicit robust regularization. Furthermore, we empirically show that our method outperforms state-of-the-art FAST-AT baselines, by achieving superior model robustness without inducing robustness catastrophic overfitting, or suffering from any loss of standard accuracy.

摘要: 对抗训练(AT)已成为一种被广泛认可的防御机制，以提高深层神经网络对抗对手攻击的鲁棒性。它解决了一个最小-最大优化问题，其中最小化器(即防御者)在存在最大化者(即攻击者)制作的对抗性示例的情况下寻求一个鲁棒模型来最小化最坏情况下的训练损失。然而，最小-最大特性使得AT计算密集，因此很难扩展。同时，FAST-AT算法，以及最近许多改进AT的算法，通过用简单的基于一次梯度符号的攻击生成步骤代替其最大化步骤，简化了基于最小-最大的AT。FAST-AT虽然易于实现，但缺乏理论保证，实际应用效果不理想，在与强对手进行训练时存在健壮性和灾难性过拟合问题。本文提出从双层优化(BLO)的角度设计FAST-AT。我们首先观察到FAST-AT最常用的算法规范等价于使用某种梯度下降型算法来解决涉及符号运算的双层问题。然而，符号运算的离散性使得很难理解算法的性能。基于上述观察，我们提出了一个新的易于处理的双层优化问题，设计并分析了一套新的算法，称为快速双层AT(FAST-BAT)。FAST-BAT能够抵抗基于符号的投影梯度下降(PGD)攻击，无需调用任何梯度符号方法和显式鲁棒正则化。此外，我们的经验表明，我们的方法优于最先进的FAST-AT基线，因为它实现了卓越的模型稳健性，而不会导致稳健性灾难性过拟合，也不会损失任何标准精度。



