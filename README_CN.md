# Latest Adversarial Attack Papers
**update at 2024-12-11 10:19:10**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Defending Against Neural Network Model Inversion Attacks via Data Poisoning**

通过数据中毒防御神经网络模型倒置攻击 cs.CR

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07575v1) [paper-pdf](http://arxiv.org/pdf/2412.07575v1)

**Authors**: Shuai Zhou, Dayong Ye, Tianqing Zhu, Wanlei Zhou

**Abstract**: Model inversion attacks pose a significant privacy threat to machine learning models by reconstructing sensitive data from their outputs. While various defenses have been proposed to counteract these attacks, they often come at the cost of the classifier's utility, thus creating a challenging trade-off between privacy protection and model utility. Moreover, most existing defenses require retraining the classifier for enhanced robustness, which is impractical for large-scale, well-established models. This paper introduces a novel defense mechanism to better balance privacy and utility, particularly against adversaries who employ a machine learning model (i.e., inversion model) to reconstruct private data. Drawing inspiration from data poisoning attacks, which can compromise the performance of machine learning models, we propose a strategy that leverages data poisoning to contaminate the training data of inversion models, thereby preventing model inversion attacks.   Two defense methods are presented. The first, termed label-preserving poisoning attacks for all output vectors (LPA), involves subtle perturbations to all output vectors while preserving their labels. Our findings demonstrate that these minor perturbations, introduced through a data poisoning approach, significantly increase the difficulty of data reconstruction without compromising the utility of the classifier. Subsequently, we introduce a second method, label-flipping poisoning for partial output vectors (LFP), which selectively perturbs a small subset of output vectors and alters their labels during the process. Empirical results indicate that LPA is notably effective, outperforming the current state-of-the-art defenses. Our data poisoning-based defense provides a new retraining-free defense paradigm that preserves the victim classifier's utility.

摘要: 模型反转攻击通过从输出中重建敏感数据，对机器学习模型构成了严重的隐私威胁。虽然已经提出了各种防御措施来对抗这些攻击，但它们往往是以分类器的效用为代价的，因此在隐私保护和模型效用之间创建了一个具有挑战性的权衡。此外，大多数现有的防御措施需要重新训练分类器以增强稳健性，这对于大规模、成熟的模型来说是不切实际的。本文介绍了一种新的防御机制，以更好地平衡隐私和效用，特别是针对使用机器学习模型(即倒置模型)重建私人数据的攻击者。从影响机器学习模型性能的数据中毒攻击中得到启发，提出了一种利用数据中毒来污染倒置模型训练数据的策略，从而防止模型倒置攻击。提出了两种防御方法。第一种称为对所有输出向量的标签保持毒化攻击(LPA)，它涉及对所有输出向量的微妙扰动，同时保持它们的标签。我们的发现表明，这些通过数据中毒方法引入的微小扰动显著增加了数据重建的难度，而不会影响分类器的实用性。随后，我们介绍了第二种方法，部分输出向量的标签翻转中毒(LFP)，它选择性地扰动一小部分输出向量，并在过程中改变它们的标签。经验结果表明，LPA非常有效，表现优于目前最先进的防御措施。我们基于数据中毒的防御提供了一种新的无需再培训的防御范例，保留了受害者分类器的实用性。



## **2. Adaptive Epsilon Adversarial Training for Robust Gravitational Wave Parameter Estimation Using Normalizing Flows**

使用正规化流进行鲁棒引力波参数估计的自适应Episodes对抗训练 cs.LG

7 pages, 9 figures

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07559v1) [paper-pdf](http://arxiv.org/pdf/2412.07559v1)

**Authors**: Yiqian Yang, Xihua Zhu, Fan Zhang

**Abstract**: Adversarial training with Normalizing Flow (NF) models is an emerging research area aimed at improving model robustness through adversarial samples. In this study, we focus on applying adversarial training to NF models for gravitational wave parameter estimation. We propose an adaptive epsilon method for Fast Gradient Sign Method (FGSM) adversarial training, which dynamically adjusts perturbation strengths based on gradient magnitudes using logarithmic scaling. Our hybrid architecture, combining ResNet and Inverse Autoregressive Flow, reduces the Negative Log Likelihood (NLL) loss by 47\% under FGSM attacks compared to the baseline model, while maintaining an NLL of 4.2 on clean data (only 5\% higher than the baseline). For perturbation strengths between 0.01 and 0.1, our model achieves an average NLL of 5.8, outperforming both fixed-epsilon (NLL: 6.7) and progressive-epsilon (NLL: 7.2) methods. Under stronger Projected Gradient Descent attacks with perturbation strength of 0.05, our model maintains an NLL of 6.4, demonstrating superior robustness while avoiding catastrophic overfitting.

摘要: 利用归一化流量模型进行对抗性训练是一个新兴的研究领域，其目的是通过对抗性样本提高模型的稳健性。在这项研究中，我们将对抗性训练应用到引力波参数估计的神经网络模型中。提出了一种用于快速梯度符号法(FGSM)对抗训练的自适应epsilon方法，该方法利用对数尺度根据梯度大小动态调整扰动强度。我们的混合架构结合了ResNet和反向自回归流，与基线模型相比，在FGSM攻击下，负对数似然(NLL)损失降低了47%，而对于干净的数据，NLL保持在4.2(仅比基线高5%)。当微扰强度在0.01到0.1之间时，我们的模型的平均NLL为5.8，优于固定epsilon(NLL：6.7)和渐进epsilon(NLL：7.2)方法。在扰动强度为0.05的较强投影梯度下降攻击下，我们的模型保持了6.4的NLL，在避免灾难性过拟合的同时表现出了优越的稳健性。



## **3. Quantifying the Prediction Uncertainty of Machine Learning Models for Individual Data**

量化个体数据机器学习模型的预测不确定性 cs.LG

PHD thesis

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07520v1) [paper-pdf](http://arxiv.org/pdf/2412.07520v1)

**Authors**: Koby Bibas

**Abstract**: Machine learning models have exhibited exceptional results in various domains. The most prevalent approach for learning is the empirical risk minimizer (ERM), which adapts the model's weights to reduce the loss on a training set and subsequently leverages these weights to predict the label for new test data. Nonetheless, ERM makes the assumption that the test distribution is similar to the training distribution, which may not always hold in real-world situations. In contrast, the predictive normalized maximum likelihood (pNML) was proposed as a min-max solution for the individual setting where no assumptions are made on the distribution of the tested input. This study investigates pNML's learnability for linear regression and neural networks, and demonstrates that pNML can improve the performance and robustness of these models on various tasks. Moreover, the pNML provides an accurate confidence measure for its output, showcasing state-of-the-art results for out-of-distribution detection, resistance to adversarial attacks, and active learning.

摘要: 机器学习模型在各个领域都显示出了出众的结果。最流行的学习方法是经验风险最小化(ERM)，它通过调整模型的权重来减少训练集上的损失，然后利用这些权重来预测新测试数据的标签。尽管如此，ERM假设测试分布类似于训练分布，这在现实世界中可能并不总是成立的。相反，预测归一化最大似然(PNML)被提出作为对测试输入的分布不作任何假设的个人设置的最小-最大解。本研究考察了PNML对线性回归和神经网络的学习能力，并证明了PNML可以提高这些模型在各种任务上的性能和稳健性。此外，PNML为其输出提供了准确的置信度度量，展示了在分布外检测、抵抗对手攻击和主动学习方面的最先进结果。



## **4. AHSG: Adversarial Attacks on High-level Semantics in Graph Neural Networks**

AHSG：对图神经网络中高级语义的对抗性攻击 cs.LG

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07468v1) [paper-pdf](http://arxiv.org/pdf/2412.07468v1)

**Authors**: Kai Yuan, Xiaobing Pei, Haoran Yang

**Abstract**: Graph Neural Networks (GNNs) have garnered significant interest among researchers due to their impressive performance in graph learning tasks. However, like other deep neural networks, GNNs are also vulnerable to adversarial attacks. In existing adversarial attack methods for GNNs, the metric between the attacked graph and the original graph is usually the attack budget or a measure of global graph properties. However, we have found that it is possible to generate attack graphs that disrupt the primary semantics even within these constraints. To address this problem, we propose a Adversarial Attacks on High-level Semantics in Graph Neural Networks (AHSG), which is a graph structure attack model that ensures the retention of primary semantics. The latent representations of each node can extract rich semantic information by applying convolutional operations on graph data. These representations contain both task-relevant primary semantic information and task-irrelevant secondary semantic information. The latent representations of same-class nodes with the same primary semantics can fulfill the objective of modifying secondary semantics while preserving the primary semantics. Finally, the latent representations with attack effects is mapped to an attack graph using Projected Gradient Descent (PGD) algorithm. By attacking graph deep learning models with some advanced defense strategies, we validate that AHSG has superior attack effectiveness compared to other attack methods. Additionally, we employ Contextual Stochastic Block Models (CSBMs) as a proxy for the primary semantics to detect the attacked graph, confirming that AHSG almost does not disrupt the original primary semantics of the graph.

摘要: 图形神经网络(GNN)因其在图形学习任务中的出色表现而引起了研究者的极大兴趣。然而，像其他深度神经网络一样，GNN也容易受到对手的攻击。在现有的GNN对抗攻击方法中，攻击图和原始图之间的度量通常是攻击预算或全局图性质的度量。然而，我们发现，即使在这些约束下，也可能生成破坏主要语义的攻击图。针对这一问题，我们提出了一种基于图神经网络高级语义的对抗性攻击(AHSG)，它是一种图结构攻击模型，保证了基本语义的保留。通过对图数据进行卷积运算，每个节点的潜在表示可以提取丰富的语义信息。这些表征既包含与任务相关的初级语义信息，也包含与任务无关的次要语义信息。具有相同初级语义的同类节点的潜在表示可以在保持初级语义的同时达到修改次级语义的目的。最后，使用投影梯度下降(PGD)算法将具有攻击效果的潜在表示映射到攻击图。通过攻击图的深度学习模型和一些先进的防御策略，验证了AHSG与其他攻击方法相比具有更好的攻击效果。此外，我们使用上下文随机块模型(CSBM)作为主要语义的代理来检测被攻击的图，证实了AHSG几乎没有破坏图的原始主要语义。



## **5. Addressing Key Challenges of Adversarial Attacks and Defenses in the Tabular Domain: A Methodological Framework for Coherence and Consistency**

应对表格领域对抗性攻击和防御的关键挑战：一致性和一致性的方法论框架 cs.LG

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07326v1) [paper-pdf](http://arxiv.org/pdf/2412.07326v1)

**Authors**: Yael Itzhakev, Amit Giloni, Yuval Elovici, Asaf Shabtai

**Abstract**: Machine learning models trained on tabular data are vulnerable to adversarial attacks, even in realistic scenarios where attackers have access only to the model's outputs. Researchers evaluate such attacks by considering metrics like success rate, perturbation magnitude, and query count. However, unlike other data domains, the tabular domain contains complex interdependencies among features, presenting a unique aspect that should be evaluated: the need for the attack to generate coherent samples and ensure feature consistency for indistinguishability. Currently, there is no established methodology for evaluating adversarial samples based on these criteria. In this paper, we address this gap by proposing new evaluation criteria tailored for tabular attacks' quality; we defined anomaly-based framework to assess the distinguishability of adversarial samples and utilize the SHAP explainability technique to identify inconsistencies in the model's decision-making process caused by adversarial samples. These criteria could form the basis for potential detection methods and be integrated into established evaluation metrics for assessing attack's quality Additionally, we introduce a novel technique for perturbing dependent features while maintaining coherence and feature consistency within the sample. We compare different attacks' strategies, examining black-box query-based attacks and transferability-based gradient attacks across four target models. Our experiments, conducted on benchmark tabular datasets, reveal significant differences between the examined attacks' strategies in terms of the attacker's risk and effort and the attacks' quality. The findings provide valuable insights on the strengths, limitations, and trade-offs of various adversarial attacks in the tabular domain, laying a foundation for future research on attacks and defense development.

摘要: 根据表格数据训练的机器学习模型容易受到敌意攻击，即使在攻击者只能访问模型输出的现实情况下也是如此。研究人员通过考虑成功率、扰乱程度和查询计数等指标来评估此类攻击。然而，与其他数据域不同的是，表格域包含各种特征之间的复杂相互依赖关系，这提出了一个应加以评估的独特方面：需要攻击生成一致的样本，并确保特征的一致性，从而实现不可区分。目前，还没有根据这些标准评估敌方样本的既定方法。在本文中，我们通过提出新的针对列表攻击质量的评估标准来解决这一问题；我们定义了基于异常的框架来评估对抗性样本的可区分性，并利用Shap可解释性技术来识别由对抗性样本导致的模型决策过程中的不一致。这些标准可以作为潜在检测方法的基础，并被集成到已建立的评估攻击质量的评估指标中。此外，我们引入了一种新的技术来扰动依赖特征，同时保持样本内的一致性和特征一致性。我们比较了不同攻击的策略，考察了基于黑盒查询的攻击和基于可转移性的梯度攻击在四个目标模型上的表现。我们在基准表格数据集上进行的实验表明，被检查的攻击策略在攻击者的风险和努力以及攻击质量方面存在显著差异。这些发现为表格领域中各种对抗性攻击的优势、局限性和权衡提供了有价值的见解，为未来攻击和防御发展的研究奠定了基础。



## **6. Backdoor Attacks against No-Reference Image Quality Assessment Models via A Scalable Trigger**

通过可扩展触发器对无参考图像质量评估模型进行后门攻击 cs.CV

Accept by AAAI 2025

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07277v1) [paper-pdf](http://arxiv.org/pdf/2412.07277v1)

**Authors**: Yi Yu, Song Xia, Xun Lin, Wenhan Yang, Shijian Lu, Yap-peng Tan, Alex Kot

**Abstract**: No-Reference Image Quality Assessment (NR-IQA), responsible for assessing the quality of a single input image without using any reference, plays a critical role in evaluating and optimizing computer vision systems, e.g., low-light enhancement. Recent research indicates that NR-IQA models are susceptible to adversarial attacks, which can significantly alter predicted scores with visually imperceptible perturbations. Despite revealing vulnerabilities, these attack methods have limitations, including high computational demands, untargeted manipulation, limited practical utility in white-box scenarios, and reduced effectiveness in black-box scenarios. To address these challenges, we shift our focus to another significant threat and present a novel poisoning-based backdoor attack against NR-IQA (BAIQA), allowing the attacker to manipulate the IQA model's output to any desired target value by simply adjusting a scaling coefficient $\alpha$ for the trigger. We propose to inject the trigger in the discrete cosine transform (DCT) domain to improve the local invariance of the trigger for countering trigger diminishment in NR-IQA models due to widely adopted data augmentations. Furthermore, the universal adversarial perturbations (UAP) in the DCT space are designed as the trigger, to increase IQA model susceptibility to manipulation and improve attack effectiveness. In addition to the heuristic method for poison-label BAIQA (P-BAIQA), we explore the design of clean-label BAIQA (C-BAIQA), focusing on $\alpha$ sampling and image data refinement, driven by theoretical insights we reveal. Extensive experiments on diverse datasets and various NR-IQA models demonstrate the effectiveness of our attacks. Code will be released at https://github.com/yuyi-sd/BAIQA.

摘要: 无参考图像质量评估(NR-IQA)负责在不使用任何参考图像的情况下评估单个输入图像的质量，在评估和优化计算机视觉系统(如微光增强)中起着至关重要的作用。最近的研究表明，NR-IQA模型容易受到对抗性攻击，这种攻击会在视觉上不可察觉的扰动下显著改变预测分数。尽管暴露出漏洞，但这些攻击方法都有局限性，包括计算要求高、无针对性操作、在白盒场景中实际效用有限，以及在黑盒场景中有效性降低。为了应对这些挑战，我们将重点转移到另一个重要的威胁上，并提出了一种针对NR-IQA的基于中毒的后门攻击(BAIQA)，允许攻击者通过简单地调整触发器的缩放系数$\α$来操纵IQA模型的输出到任何期望的目标值。我们提出在离散余弦变换(DCT)域中注入触发器以改善触发器的局部不变性，以对抗由于广泛采用的数据增强而导致的NR-IQA模型中的触发器衰减。此外，设计了DCT空间中的通用对抗摄动(UAP)作为触发器，以增加IQA模型对操纵的敏感度，提高攻击效率。除了有毒标签BAIQA的启发式方法(P-BAIQA)外，我们还探索了清洁标签BAIQA(C-BAIQA)的设计，重点是$\α$采样和图像数据精化，这是我们揭示的理论见解的驱动。在不同的数据集和不同的NR-IQA模型上的大量实验证明了我们的攻击的有效性。代码将在https://github.com/yuyi-sd/BAIQA.上发布



## **7. A Generative Victim Model for Segmentation**

用于分割的生成受害者模型 cs.CV

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07274v1) [paper-pdf](http://arxiv.org/pdf/2412.07274v1)

**Authors**: Aixuan Li, Jing Zhang, Jiawei Shi, Yiran Zhong, Yuchao Dai

**Abstract**: We find that the well-trained victim models (VMs), against which the attacks are generated, serve as fundamental prerequisites for adversarial attacks, i.e. a segmentation VM is needed to generate attacks for segmentation. In this context, the victim model is assumed to be robust to achieve effective adversarial perturbation generation. Instead of focusing on improving the robustness of the task-specific victim models, we shift our attention to image generation. From an image generation perspective, we derive a novel VM for segmentation, aiming to generate adversarial perturbations for segmentation tasks without requiring models explicitly designed for image segmentation. Our approach to adversarial attack generation diverges from conventional white-box or black-box attacks, offering a fresh outlook on adversarial attack strategies. Experiments show that our attack method is able to generate effective adversarial attacks with good transferability.

摘要: 我们发现，攻击所针对的训练有素的受害者模型（VMs）是对抗性攻击的基本先决条件，即需要分段虚拟机来生成分段攻击。在这种情况下，假设受害者模型是稳健的，能够实现有效的对抗扰动生成。我们不再专注于提高特定任务受害者模型的稳健性，而是将注意力转移到图像生成上。从图像生成的角度来看，我们推导出一种新型的用于分割的虚拟机，旨在为分割任务生成对抗性扰动，而不需要为图像分割明确设计的模型。我们的对抗性攻击生成方法与传统的白盒或黑匣子攻击不同，为对抗性攻击策略提供了全新的视角。实验表明，我们的攻击方法能够产生有效的对抗攻击，具有良好的可移植性。



## **8. CapGen:An Environment-Adaptive Generator of Adversarial Patches**

CapGen：环境适应性对抗补丁生成器 cs.CV

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07253v1) [paper-pdf](http://arxiv.org/pdf/2412.07253v1)

**Authors**: Chaoqun Li, Zhuodong Liu, Huanqian Yan, Hang Su

**Abstract**: Adversarial patches, often used to provide physical stealth protection for critical assets and assess perception algorithm robustness, usually neglect the need for visual harmony with the background environment, making them easily noticeable. Moreover, existing methods primarily concentrate on improving attack performance, disregarding the intricate dynamics of adversarial patch elements. In this work, we introduce the Camouflaged Adversarial Pattern Generator (CAPGen), a novel approach that leverages specific base colors from the surrounding environment to produce patches that seamlessly blend with their background for superior visual stealthiness while maintaining robust adversarial performance. We delve into the influence of both patterns (i.e., color-agnostic texture information) and colors on the effectiveness of attacks facilitated by patches, discovering that patterns exert a more pronounced effect on performance than colors. Based on these findings, we propose a rapid generation strategy for adversarial patches. This involves updating the colors of high-performance adversarial patches to align with those of the new environment, ensuring visual stealthiness without compromising adversarial impact. This paper is the first to comprehensively examine the roles played by patterns and colors in the context of adversarial patches.

摘要: 对抗性补丁通常用于为关键资产提供物理隐身保护，并评估感知算法的健壮性，通常忽略了与背景环境视觉和谐的需要，使它们很容易被注意到。此外，现有的方法主要集中在提高攻击性能，而忽略了对抗性补丁元素的复杂动态。在这项工作中，我们介绍了伪装对抗模式生成器(CAPGen)，这是一种新的方法，利用周围环境中特定的基色来产生与背景无缝混合的补丁，以实现卓越的视觉隐蔽性，同时保持稳健的对抗性能。我们深入研究了图案(即与颜色无关的纹理信息)和颜色对补丁攻击有效性的影响，发现图案比颜色对性能的影响更显著。基于这些发现，我们提出了一种对抗性补丁的快速生成策略。这包括更新高性能对抗性补丁的颜色以与新环境的颜色保持一致，确保视觉隐蔽性而不影响对抗性影响。本文首次全面探讨了图案和色彩在对抗性斑块中所起的作用。



## **9. Adversarial Filtering Based Evasion and Backdoor Attacks to EEG-Based Brain-Computer Interfaces**

基于对抗过滤的规避和后门攻击基于脑电的脑机接口 cs.HC

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07231v1) [paper-pdf](http://arxiv.org/pdf/2412.07231v1)

**Authors**: Lubin Meng, Xue Jiang, Xiaoqing Chen, Wenzhong Liu, Hanbin Luo, Dongrui Wu

**Abstract**: A brain-computer interface (BCI) enables direct communication between the brain and an external device. Electroencephalogram (EEG) is a common input signal for BCIs, due to its convenience and low cost. Most research on EEG-based BCIs focuses on the accurate decoding of EEG signals, while ignoring their security. Recent studies have shown that machine learning models in BCIs are vulnerable to adversarial attacks. This paper proposes adversarial filtering based evasion and backdoor attacks to EEG-based BCIs, which are very easy to implement. Experiments on three datasets from different BCI paradigms demonstrated the effectiveness of our proposed attack approaches. To our knowledge, this is the first study on adversarial filtering for EEG-based BCIs, raising a new security concern and calling for more attention on the security of BCIs.

摘要: 脑机接口（BCI）实现大脑和外部设备之间的直接通信。由于其方便性和低成本，脑电波（EEG）是BCI的常见输入信号。大多数关于基于脑电的BCI的研究都集中在脑电信号的准确解码上，而忽视了其安全性。最近的研究表明，BCI中的机器学习模型容易受到对抗攻击。本文提出了对基于脑电的BCI的基于对抗过滤的规避和后门攻击，这些攻击非常容易实现。对来自不同BCI范式的三个数据集的实验证明了我们提出的攻击方法的有效性。据我们所知，这是第一项关于基于脑电的BCI对抗过滤的研究，提出了新的安全问题，并呼吁人们更加关注BCI的安全性。



## **10. A Parametric Approach to Adversarial Augmentation for Cross-Domain Iris Presentation Attack Detection**

跨域虹膜呈现攻击检测的对抗增强参数方法 cs.CV

IEEE/CVF Winter Conference on Applications of Computer Vision (WACV),  2025

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07199v1) [paper-pdf](http://arxiv.org/pdf/2412.07199v1)

**Authors**: Debasmita Pal, Redwan Sony, Arun Ross

**Abstract**: Iris-based biometric systems are vulnerable to presentation attacks (PAs), where adversaries present physical artifacts (e.g., printed iris images, textured contact lenses) to defeat the system. This has led to the development of various presentation attack detection (PAD) algorithms, which typically perform well in intra-domain settings. However, they often struggle to generalize effectively in cross-domain scenarios, where training and testing employ different sensors, PA instruments, and datasets. In this work, we use adversarial training samples of both bonafide irides and PAs to improve the cross-domain performance of a PAD classifier. The novelty of our approach lies in leveraging transformation parameters from classical data augmentation schemes (e.g., translation, rotation) to generate adversarial samples. We achieve this through a convolutional autoencoder, ADV-GEN, that inputs original training samples along with a set of geometric and photometric transformations. The transformation parameters act as regularization variables, guiding ADV-GEN to generate adversarial samples in a constrained search space. Experiments conducted on the LivDet-Iris 2017 database, comprising four datasets, and the LivDet-Iris 2020 dataset, demonstrate the efficacy of our proposed method. The code is available at https://github.com/iPRoBe-lab/ADV-GEN-IrisPAD.

摘要: 基于虹膜的生物识别系统容易受到呈现攻击(PAS)，即攻击者呈现物理伪像(例如，打印的虹膜图像、纹理隐形眼镜)来击败系统。这导致了各种呈现攻击检测(PAD)算法的发展，这些算法通常在域内设置中执行得很好。然而，在训练和测试使用不同的传感器、PA仪器和数据集的跨域场景中，它们往往难以有效地推广。在这项工作中，我们使用真实虹膜和PAS的对抗性训练样本来提高PAD分类器的跨域性能。我们方法的创新之处在于利用经典数据增强方案(如平移、旋转)中的变换参数来生成对抗性样本。我们通过卷积自动编码器ADV-Gen实现这一点，该编码器输入原始训练样本以及一组几何和光度变换。变换参数作为正则化变量，指导ADV-Gen在受限的搜索空间中生成对抗性样本。在包含四个数据集的LivDet-Iris 2017数据库和LivDet-Iris 2020数据集上进行的实验证明了我们所提出的方法的有效性。代码可在https://github.com/iPRoBe-lab/ADV-GEN-IrisPAD.上获得



## **11. PrisonBreak: Jailbreaking Large Language Models with Fewer Than Twenty-Five Targeted Bit-flips**

Prison Break：越狱大型语言模型，目标位翻转少于25个 cs.CR

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07192v1) [paper-pdf](http://arxiv.org/pdf/2412.07192v1)

**Authors**: Zachary Coalson, Jeonghyun Woo, Shiyang Chen, Yu Sun, Lishan Yang, Prashant Nair, Bo Fang, Sanghyun Hong

**Abstract**: We introduce a new class of attacks on commercial-scale (human-aligned) language models that induce jailbreaking through targeted bitwise corruptions in model parameters. Our adversary can jailbreak billion-parameter language models with fewer than 25 bit-flips in all cases$-$and as few as 5 in some$-$using up to 40$\times$ less bit-flips than existing attacks on computer vision models at least 100$\times$ smaller. Unlike prompt-based jailbreaks, our attack renders these models in memory 'uncensored' at runtime, allowing them to generate harmful responses without any input modifications. Our attack algorithm efficiently identifies target bits to flip, offering up to 20$\times$ more computational efficiency than previous methods. This makes it practical for language models with billions of parameters. We show an end-to-end exploitation of our attack using software-induced fault injection, Rowhammer (RH). Our work examines 56 DRAM RH profiles from DDR4 and LPDDR4X devices with different RH vulnerabilities. We show that our attack can reliably induce jailbreaking in systems similar to those affected by prior bit-flip attacks. Moreover, our approach remains effective even against highly RH-secure systems (e.g., 46$\times$ more secure than previously tested systems). Our analyses further reveal that: (1) models with less post-training alignment require fewer bit flips to jailbreak; (2) certain model components, such as value projection layers, are substantially more vulnerable than others; and (3) our method is mechanistically different than existing jailbreaks. Our findings highlight a pressing, practical threat to the language model ecosystem and underscore the need for research to protect these models from bit-flip attacks.

摘要: 我们在商业规模(人类对齐的)语言模型上引入了一类新的攻击，这些攻击通过模型参数中有针对性的逐位破坏来诱导越狱。我们的对手可以用不到25个比特翻转的语言模型越狱，所有情况下都不到25个比特翻转，在一些$-$中只有5个，使用多达40个比特翻转，比对计算机视觉模型的现有攻击少至少100$\×$。与基于提示的越狱不同，我们的攻击在运行时将这些模型呈现在内存中，不受审查，允许它们在不修改任何输入的情况下生成有害的响应。我们的攻击算法有效地识别要翻转的目标比特，比以前的方法提供了高达20美元\倍的计算效率。这使得它适用于具有数十亿个参数的语言模型。我们使用软件诱导的故障注入Rowhammer(RH)展示了对我们的攻击的端到端攻击。我们的工作检查了来自具有不同RH漏洞的DDR4和LPDDR4X设备的56个DRAM RH配置文件。我们证明了我们的攻击可以可靠地在类似于先前受比特翻转攻击影响的系统中诱导越狱。此外，我们的方法即使对高度RH安全的系统也是有效的(例如，比之前测试的系统安全46美元\倍)。我们的分析进一步表明：(1)训练后对齐较少的模型需要较少的比特翻转越狱；(2)某些模型组件，如值投影层，比其他组件更容易受到攻击；(3)我们的方法与现有的越狱方法在机械上不同。我们的发现突显了语言模型生态系统面临的紧迫、实际的威胁，并强调了研究保护这些模型免受比特翻转攻击的必要性。



## **12. dSTAR: Straggler Tolerant and Byzantine Resilient Distributed SGD**

dSTAR：容忍落后和拜占庭弹性分布式新元 cs.DC

15 pages

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07151v1) [paper-pdf](http://arxiv.org/pdf/2412.07151v1)

**Authors**: Jiahe Yan, Pratik Chaudhari, Leonard Kleinrock

**Abstract**: Distributed model training needs to be adapted to challenges such as the straggler effect and Byzantine attacks. When coordinating the training process with multiple computing nodes, ensuring timely and reliable gradient aggregation amidst network and system malfunctions is essential. To tackle these issues, we propose \textit{dSTAR}, a lightweight and efficient approach for distributed stochastic gradient descent (SGD) that enhances robustness and convergence. \textit{dSTAR} selectively aggregates gradients by collecting updates from the first \(k\) workers to respond, filtering them based on deviations calculated using an ensemble median. This method not only mitigates the impact of stragglers but also fortifies the model against Byzantine adversaries. We theoretically establish that \textit{dSTAR} is (\(\alpha, f\))-Byzantine resilient and achieves a linear convergence rate. Empirical evaluations across various scenarios demonstrate that \textit{dSTAR} consistently maintains high accuracy, outperforming other Byzantine-resilient methods that often suffer up to a 40-50\% accuracy drop under attack. Our results highlight \textit{dSTAR} as a robust solution for training models in distributed environments prone to both straggler delays and Byzantine faults.

摘要: 分布式模型训练需要适应诸如掉队效应和拜占庭攻击等挑战。在与多个计算节点协调训练过程时，确保在网络和系统故障中及时可靠地进行梯度聚合是至关重要的。为了解决这些问题，我们提出了一种轻量级、高效的分布式随机梯度下降(SGD)方法，该方法增强了鲁棒性和收敛能力。通过收集来自第一(K)个工作人员的更新进行响应，根据使用集合中值计算的偏差对其进行过滤，从而有选择地聚合梯度。这种方法不仅减轻了掉队的影响，而且增强了该模型对抗拜占庭式对手的能力。我们从理论上证明了Texttit{dSTAR}是((α，f))-拜占庭弹性的，并且达到了线性收敛速度。对不同场景的经验评估表明，该方法始终保持较高的准确率，优于其他拜占庭弹性方法，后者在攻击下的准确率通常会下降40%-50%。我们的结果突出表明，在容易出现分散延迟和拜占庭故障的分布式环境中，文本{dSTAR}是一种稳健的模型训练解决方案。



## **13. Defensive Dual Masking for Robust Adversarial Defense**

防御性双重掩蔽实现强大的对抗性防御 cs.CL

First version

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07078v1) [paper-pdf](http://arxiv.org/pdf/2412.07078v1)

**Authors**: Wangli Yang, Jie Yang, Yi Guo, Johan Barthelemy

**Abstract**: The field of textual adversarial defenses has gained considerable attention in recent years due to the increasing vulnerability of natural language processing (NLP) models to adversarial attacks, which exploit subtle perturbations in input text to deceive models. This paper introduces the Defensive Dual Masking (DDM) algorithm, a novel approach designed to enhance model robustness against such attacks. DDM utilizes a unique adversarial training strategy where [MASK] tokens are strategically inserted into training samples to prepare the model to handle adversarial perturbations more effectively. During inference, potentially adversarial tokens are dynamically replaced with [MASK] tokens to neutralize potential threats while preserving the core semantics of the input. The theoretical foundation of our approach is explored, demonstrating how the selective masking mechanism strengthens the model's ability to identify and mitigate adversarial manipulations. Our empirical evaluation across a diverse set of benchmark datasets and attack mechanisms consistently shows that DDM outperforms state-of-the-art defense techniques, improving model accuracy and robustness. Moreover, when applied to Large Language Models (LLMs), DDM also enhances their resilience to adversarial attacks, providing a scalable defense mechanism for large-scale NLP applications.

摘要: 近年来，由于自然语言处理(NLP)模型越来越容易受到敌意攻击，利用输入文本中的细微扰动来欺骗模型，文本对抗防御领域受到了相当大的关注。介绍了防御性双重掩蔽(DDM)算法，这是一种新的方法，旨在增强模型对此类攻击的稳健性。DDM利用一种独特的对抗性训练策略，其中[MASK]标记被战略性地插入到训练样本中，以准备模型以更有效地处理对抗性扰动。在推理过程中，潜在的敌意令牌被动态地替换为[掩码]令牌，以中和潜在的威胁，同时保留输入的核心语义。探讨了我们方法的理论基础，展示了选择性掩蔽机制如何增强模型识别和缓解对手操纵的能力。我们对不同的基准数据集和攻击机制进行的经验评估一致表明，DDM的性能优于最先进的防御技术，提高了模型的准确性和稳健性。此外，当应用于大型语言模型时，DDM还增强了它们对对手攻击的韧性，为大规模NLP应用提供了一种可扩展的防御机制。



## **14. Dense Cross-Connected Ensemble Convolutional Neural Networks for Enhanced Model Robustness**

密集交叉连接卷积神经网络增强模型鲁棒性 cs.CV

6 pages, 1 figure

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2412.07022v1) [paper-pdf](http://arxiv.org/pdf/2412.07022v1)

**Authors**: Longwei Wang, Xueqian Li, Zheng Zhang

**Abstract**: The resilience of convolutional neural networks against input variations and adversarial attacks remains a significant challenge in image recognition tasks. Motivated by the need for more robust and reliable image recognition systems, we propose the Dense Cross-Connected Ensemble Convolutional Neural Network (DCC-ECNN). This novel architecture integrates the dense connectivity principle of DenseNet with the ensemble learning strategy, incorporating intermediate cross-connections between different DenseNet paths to facilitate extensive feature sharing and integration. The DCC-ECNN architecture leverages DenseNet's efficient parameter usage and depth while benefiting from the robustness of ensemble learning, ensuring a richer and more resilient feature representation.

摘要: 卷积神经网络对输入变化和对抗攻击的弹性仍然是图像识别任务中的一个重大挑战。出于对更稳健、更可靠的图像识别系统的需求，我们提出了密集交叉连接卷积神经网络（DCC-ECNN）。这种新颖的架构将DenseNet的密集连接原则与集成学习策略集成在一起，合并了不同DenseNet路径之间的中间交叉连接，以促进广泛的特征共享和集成。DCC-ECNN架构利用DenseNet的高效参数使用和深度，同时受益于集成学习的稳健性，确保更丰富、更有弹性的特征表示。



## **15. Fiat-Shamir for Proofs Lacks a Proof Even in the Presence of Shared Entanglement**

即使存在共同纠缠，菲亚特-沙米尔的证据也缺乏证据 quant-ph

58 pages, 4 figures; accepted in Quantum

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2204.02265v5) [paper-pdf](http://arxiv.org/pdf/2204.02265v5)

**Authors**: Frédéric Dupuis, Philippe Lamontagne, Louis Salvail

**Abstract**: We explore the cryptographic power of arbitrary shared physical resources. The most general such resource is access to a fresh entangled quantum state at the outset of each protocol execution. We call this the Common Reference Quantum State (CRQS) model, in analogy to the well-known Common Reference String (CRS). The CRQS model is a natural generalization of the CRS model but appears to be more powerful: in the two-party setting, a CRQS can sometimes exhibit properties associated with a Random Oracle queried once by measuring a maximally entangled state in one of many mutually unbiased bases. We formalize this notion as a Weak One-Time Random Oracle (WOTRO), where we only ask of the $m$-bit output to have some randomness when conditioned on the $n$-bit input.   We show that when $n-m\in\omega(\lg n)$, any protocol for WOTRO in the CRQS model can be attacked by an (inefficient) adversary. Moreover, our adversary is efficiently simulatable, which rules out the possibility of proving the computational security of a scheme by a fully black-box reduction to a cryptographic game assumption. On the other hand, we introduce a non-game quantum assumption for hash functions that implies WOTRO in the CRQS model (where the CRQS consists only of EPR pairs). We first build a statistically secure WOTRO protocol where $m=n$, then hash the output.   The impossibility of WOTRO has the following consequences. First, we show the fully-black-box impossibility of a quantum Fiat-Shamir transform, extending the impossibility result of Bitansky et al. (TCC 2013) to the CRQS model. Second, we show a fully-black-box impossibility result for a strenghtened version of quantum lightning (Zhandry, Eurocrypt 2019) where quantum bolts have an additional parameter that cannot be changed without generating new bolts. Our results also apply to $2$-message protocols in the plain model.

摘要: 我们探索任意共享物理资源的加密能力。最常见的这类资源是在每个协议执行开始时访问新的纠缠量子态。我们称之为公共参考量子态(CRQS)模型，类似于众所周知的公共参考弦(CRS)。CRQS模型是CRS模型的自然推广，但似乎更强大：在两方设置中，CRQS有时可以通过测量许多相互无偏的碱基之一中的最大纠缠态来展示与查询一次的随机Oracle相关联的属性。我们将这个概念形式化为弱一次性随机Oracle(WOTRO)，其中我们只要求$m$位的输出在以$n$位输入为条件时具有一定的随机性。我们证明了当$n-m\in\omega(\lg n)$时，CRQS模型中用于WOTRO的任何协议都可以被(低效的)攻击者攻击。此外，我们的对手是高效可模拟的，这排除了通过完全黑箱简化为密码博弈假设来证明方案的计算安全性的可能性。另一方面，我们对散列函数引入了一个非博弈量子假设，该假设意味着CRQS模型(其中CRQS只由EPR对组成)中的WOTRO。我们首先构建一个统计安全的WOTRO协议，其中$m=n$，然后对输出进行散列。WOTRO的不可能性会产生以下后果。首先，我们证明了量子Fiat-Shamir变换的全黑箱不可能性，推广了Bitansky等人的不可能结果。(TCC 2013)到CRQS模式。其次，我们证明了量子闪电的增强版本(Zhandry，Eurocrypt 2019)的一个完全黑箱不可能的结果，其中量子闪电有一个额外的参数，如果不产生新的闪电，这个参数就不能改变。我们的结果也适用于普通模型中的$2$-消息协议。



## **16. WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs**

WildGuard：针对LLC安全风险、越狱和拒绝的开放式一站式审核工具 cs.CL

NeurIPS 2024 Camera Ready. First two authors contributed equally.  Third and fourth authors contributed equally

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2406.18495v3) [paper-pdf](http://arxiv.org/pdf/2406.18495v3)

**Authors**: Seungju Han, Kavel Rao, Allyson Ettinger, Liwei Jiang, Bill Yuchen Lin, Nathan Lambert, Yejin Choi, Nouha Dziri

**Abstract**: We introduce WildGuard -- an open, light-weight moderation tool for LLM safety that achieves three goals: (1) identifying malicious intent in user prompts, (2) detecting safety risks of model responses, and (3) determining model refusal rate. Together, WildGuard serves the increasing needs for automatic safety moderation and evaluation of LLM interactions, providing a one-stop tool with enhanced accuracy and broad coverage across 13 risk categories. While existing open moderation tools such as Llama-Guard2 score reasonably well in classifying straightforward model interactions, they lag far behind a prompted GPT-4, especially in identifying adversarial jailbreaks and in evaluating models' refusals, a key measure for evaluating safety behaviors in model responses.   To address these challenges, we construct WildGuardMix, a large-scale and carefully balanced multi-task safety moderation dataset with 92K labeled examples that cover vanilla (direct) prompts and adversarial jailbreaks, paired with various refusal and compliance responses. WildGuardMix is a combination of WildGuardTrain, the training data of WildGuard, and WildGuardTest, a high-quality human-annotated moderation test set with 5K labeled items covering broad risk scenarios. Through extensive evaluations on WildGuardTest and ten existing public benchmarks, we show that WildGuard establishes state-of-the-art performance in open-source safety moderation across all the three tasks compared to ten strong existing open-source moderation models (e.g., up to 26.4% improvement on refusal detection). Importantly, WildGuard matches and sometimes exceeds GPT-4 performance (e.g., up to 3.9% improvement on prompt harmfulness identification). WildGuard serves as a highly effective safety moderator in an LLM interface, reducing the success rate of jailbreak attacks from 79.8% to 2.4%.

摘要: 我们介绍了WildGuard--一个开放的、轻量级的LLM安全防御工具，它实现了三个目标：(1)识别用户提示中的恶意意图，(2)检测模型响应的安全风险，(3)确定模型拒绝率。综合起来，WildGuard可满足日益增长的自动安全审核和评估LLM交互作用的需求，提供了一种一站式工具，具有更高的准确性和广泛的覆盖范围，涵盖13个风险类别。虽然现有的开放式审核工具，如Llama-Guard2，在对直接的模型交互进行分类方面得分相当好，但它们远远落后于GPT-4，特别是在识别对抗性越狱和评估模型拒绝方面，这是评估模型响应中安全行为的关键指标。为了应对这些挑战，我们构建了WildGuardMix，这是一个大规模的、仔细平衡的多任务安全缓和数据集，具有92K标记的示例，涵盖普通(直接)提示和对抗性越狱，并与各种拒绝和合规响应配对。WildGuardMix是WildGuard的训练数据WildGuardTrain和WildGuardTest的组合，WildGuardTest是一种高质量的人工注释适度测试集，具有覆盖广泛风险情景的5K标签项目。通过对WildGuardTest和十个现有公共基准的广泛评估，我们表明WildGuard在所有三个任务中建立了开源安全适度的最先进性能，而不是现有的十个强大的开源适度模型(例如，拒绝检测方面高达26.4%的改进)。重要的是，WildGuard的性能与GPT-4相当，有时甚至超过GPT-4(例如，在及时识别危害性方面最高提高3.9%)。WildGuard在LLM界面中充当高效的安全调节器，将越狱攻击的成功率从79.8%降低到2.4%。



## **17. Take Fake as Real: Realistic-like Robust Black-box Adversarial Attack to Evade AIGC Detection**

以假为真：类似现实的鲁棒黑匣子对抗攻击以逃避AIGC检测 cs.CV

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2412.06727v1) [paper-pdf](http://arxiv.org/pdf/2412.06727v1)

**Authors**: Caiyun Xie, Dengpan Ye, Yunming Zhang, Long Tang, Yunna Lv, Jiacheng Deng, Jiawei Song

**Abstract**: The security of AI-generated content (AIGC) detection based on GANs and diffusion models is closely related to the credibility of multimedia content. Malicious adversarial attacks can evade these developing AIGC detection. However, most existing adversarial attacks focus only on GAN-generated facial images detection, struggle to be effective on multi-class natural images and diffusion-based detectors, and exhibit poor invisibility. To fill this gap, we first conduct an in-depth analysis of the vulnerability of AIGC detectors and discover the feature that detectors vary in vulnerability to different post-processing. Then, considering the uncertainty of detectors in real-world scenarios, and based on the discovery, we propose a Realistic-like Robust Black-box Adversarial attack (R$^2$BA) with post-processing fusion optimization. Unlike typical perturbations, R$^2$BA uses real-world post-processing, i.e., Gaussian blur, JPEG compression, Gaussian noise and light spot to generate adversarial examples. Specifically, we use a stochastic particle swarm algorithm with inertia decay to optimize post-processing fusion intensity and explore the detector's decision boundary. Guided by the detector's fake probability, R$^2$BA enhances/weakens the detector-vulnerable/detector-robust post-processing intensity to strike a balance between adversariality and invisibility. Extensive experiments on popular/commercial AIGC detectors and datasets demonstrate that R$^2$BA exhibits impressive anti-detection performance, excellent invisibility, and strong robustness in GAN-based and diffusion-based cases. Compared to state-of-the-art white-box and black-box attacks, R$^2$BA shows significant improvements of 15% and 21% in anti-detection performance under the original and robust scenario respectively, offering valuable insights for the security of AIGC detection in real-world applications.

摘要: 基于遗传算法和扩散模型的人工智能生成内容(AIGC)检测的安全性与多媒体内容的可信度密切相关。恶意对抗性攻击可以逃避这些正在开发的AIGC检测。然而，现有的对抗性攻击大多只针对GaN生成的人脸图像检测，难以对多类自然图像和基于扩散的检测器有效，并且表现出较差的不可见性。为了填补这一空白，我们首先对AIGC检测器的脆弱性进行了深入的分析，发现了检测器对不同后处理的脆弱性不同的特点。然后，考虑到检测器在实际场景中的不确定性，基于这一发现，我们提出了一种具有后处理融合优化的逼真的稳健黑盒对抗攻击(R$^2$BA)。与典型的扰动不同，R$^2$BA使用真实世界的后处理，即高斯模糊、JPEG压缩、高斯噪声和光斑来生成对抗性示例。具体地说，我们使用了一种带惯性衰减的随机粒子群算法来优化后处理融合强度，并探索了检测器的决策边界。在检测器伪概率的指导下，R$^2$BA增强/削弱了检测器易受攻击/检测器健壮的后处理强度，以在对抗性和不可见性之间取得平衡。在流行的/商用AIGC探测器和数据集上的大量实验表明，R$^2$BA在基于GaN和基于扩散的情况下具有令人印象深刻的抗检测性能、出色的不可见性和强大的稳健性。与最新的白盒和黑盒攻击相比，R$^2$BA在原始场景和健壮场景下的抗检测性能分别提高了15%和21%，为实际应用中AIGC检测的安全性提供了有价值的见解。



## **18. More is Better (Mostly): On the Backdoor Attacks in Federated Graph Neural Networks**

越多越好（大多数）：关于联邦图神经网络中的后门攻击 cs.CR

15 pages, 13 figures

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2202.03195v6) [paper-pdf](http://arxiv.org/pdf/2202.03195v6)

**Authors**: Jing Xu, Rui Wang, Stefanos Koffas, Kaitai Liang, Stjepan Picek

**Abstract**: Graph Neural Networks (GNNs) are a class of deep learning-based methods for processing graph domain information. GNNs have recently become a widely used graph analysis method due to their superior ability to learn representations for complex graph data. However, due to privacy concerns and regulation restrictions, centralized GNNs can be difficult to apply to data-sensitive scenarios. Federated learning (FL) is an emerging technology developed for privacy-preserving settings when several parties need to train a shared global model collaboratively. Although several research works have applied FL to train GNNs (Federated GNNs), there is no research on their robustness to backdoor attacks.   This paper bridges this gap by conducting two types of backdoor attacks in Federated GNNs: centralized backdoor attacks (CBA) and distributed backdoor attacks (DBA). Our experiments show that the DBA attack success rate is higher than CBA in almost all evaluated cases. For CBA, the attack success rate of all local triggers is similar to the global trigger even if the training set of the adversarial party is embedded with the global trigger. To further explore the properties of two backdoor attacks in Federated GNNs, we evaluate the attack performance for a different number of clients, trigger sizes, poisoning intensities, and trigger densities. Moreover, we explore the robustness of DBA and CBA against one defense. We find that both attacks are robust against the investigated defense, necessitating the need to consider backdoor attacks in Federated GNNs as a novel threat that requires custom defenses.

摘要: [TencentCloudSDKException] code:InternalError.BackendTimeout message:Backend timeout, please retry it later requestId:3d46603a-4c6f-4e2e-b3ca-082be89c6e3f



## **19. Vulnerability, Where Art Thou? An Investigation of Vulnerability Management in Android Smartphone Chipsets**

脆弱，你在哪里？Android智能手机芯片组漏洞管理调查 cs.CR

Accepted by Network and Distributed System Security (NDSS) Symposium  2025

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2412.06556v1) [paper-pdf](http://arxiv.org/pdf/2412.06556v1)

**Authors**: Daniel Klischies, Philipp Mackensen, Veelasha Moonsamy

**Abstract**: Vulnerabilities in Android smartphone chipsets have severe consequences, as recent real-world attacks have demonstrated that adversaries can leverage vulnerabilities to execute arbitrary code or exfiltrate confidential information. Despite the far-reaching impact of such attacks, the lifecycle of chipset vulnerabilities has yet to be investigated, with existing papers primarily investigating vulnerabilities in the Android operating system. This paper provides a comprehensive and empirical study of the current state of smartphone chipset vulnerability management within the Android ecosystem. For the first time, we create a unified knowledge base of 3,676 chipset vulnerabilities affecting 437 chipset models from all four major chipset manufacturers, combined with 6,866 smartphone models. Our analysis revealed that the same vulnerabilities are often included in multiple generations of chipsets, providing novel empirical evidence that vulnerabilities are inherited through multiple chipset generations. Furthermore, we demonstrate that the commonly accepted 90-day responsible vulnerability disclosure period is seldom adhered to. We find that a single vulnerability often affects hundreds to thousands of different smartphone models, for which update availability is, as we show, often unclear or heavily delayed. Leveraging the new insights gained from our empirical analysis, we recommend several changes that chipset manufacturers can implement to improve the security posture of their products. At the same time, our knowledge base enables academic researchers to conduct more representative evaluations of smartphone chipsets, accurately assess the impact of vulnerabilities they discover, and identify avenues for future research.

摘要: Android智能手机芯片组中的漏洞具有严重后果，因为最近的现实世界攻击表明，攻击者可以利用漏洞执行任意代码或泄露机密信息。尽管此类攻击影响深远，但芯片组漏洞的生命周期尚未调查，现有论文主要调查Android操作系统中的漏洞。本文对Android生态系统中智能手机芯片组漏洞管理的现状进行了全面的实证研究。我们首次创建了一个统一的知识库，其中包含影响所有四大芯片组制造商的437个芯片组型号的3,676个芯片组漏洞，以及6,866个智能手机型号。我们的分析显示，相同的漏洞通常包含在多代芯片组中，这为漏洞通过多代芯片组遗传提供了新的经验证据。此外，我们还证明，通常接受的90天负责任的漏洞披露期限很少得到遵守。我们发现，一个单一的漏洞通常会影响数百到数千种不同的智能手机型号，正如我们所显示的那样，这些型号的更新通常不清楚或严重延迟。利用我们从经验分析中获得的新见解，我们建议芯片组制造商可以实施的几项更改，以改善其产品的安全状况。同时，我们的知识库使学术研究人员能够对智能手机芯片组进行更具代表性的评估，准确评估他们发现的漏洞的影响，并确定未来研究的途径。



## **20. Flexible and Scalable Deep Dendritic Spiking Neural Networks with Multiple Nonlinear Branching**

具有多个非线性分支的灵活可扩展深度树枝状尖峰神经网络 cs.NE

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2412.06355v1) [paper-pdf](http://arxiv.org/pdf/2412.06355v1)

**Authors**: Yifan Huang, Wei Fang, Zhengyu Ma, Guoqi Li, Yonghong Tian

**Abstract**: Recent advances in spiking neural networks (SNNs) have a predominant focus on network architectures, while relatively little attention has been paid to the underlying neuron model. The point neuron models, a cornerstone of deep SNNs, pose a bottleneck on the network-level expressivity since they depict somatic dynamics only. In contrast, the multi-compartment models in neuroscience offer remarkable expressivity by introducing dendritic morphology and dynamics, but remain underexplored in deep learning due to their unaffordable computational cost and inflexibility. To combine the advantages of both sides for a flexible, efficient yet more powerful model, we propose the dendritic spiking neuron (DendSN) incorporating multiple dendritic branches with nonlinear dynamics. Compared to the point spiking neurons, DendSN exhibits significantly higher expressivity. DendSN's flexibility enables its seamless integration into diverse deep SNN architectures. To accelerate dendritic SNNs (DendSNNs), we parallelize dendritic state updates across time steps, and develop Triton kernels for GPU-level acceleration. As a result, we can construct large-scale DendSNNs with depth comparable to their point SNN counterparts. Next, we comprehensively evaluate DendSNNs' performance on various demanding tasks. By modulating dendritic branch strengths using a context signal, catastrophic forgetting of DendSNNs is substantially mitigated. Moreover, DendSNNs demonstrate enhanced robustness against noise and adversarial attacks compared to point SNNs, and excel in few-shot learning settings. Our work firstly demonstrates the possibility of training bio-plausible dendritic SNNs with depths and scales comparable to traditional point SNNs, and reveals superior expressivity and robustness of reduced dendritic neuron models in deep learning, thereby offering a fresh perspective on advancing neural network design.

摘要: 尖峰神经网络(SNN)的最新进展主要集中在网络结构上，而对潜在的神经元模型的关注相对较少。点神经元模型是深层次SNN的基石，但由于其仅描述体细胞动力学，对网络层次的表现力构成了瓶颈。相比之下，神经科学中的多室模型通过引入树突形态和动力学提供了显著的表现力，但由于其负担不起的计算成本和灵活性，在深度学习中仍然没有得到充分的探索。为了结合两者的优点建立一个灵活、高效且功能更强大的模型，我们提出了一种结合了多个树枝和非线性动力学的树状突起神经元模型(DendSN)。与点刺神经元相比，DendSN的表达显著增强。DendSN的灵活性使其能够无缝集成到各种深度SNN架构中。为了加速树状SNN(DendSNN)，我们跨时间步长并行更新树状状态，并开发用于GPU级加速的Triton核。因此，我们可以构建大规模的DendSNN，其深度可与它们的点SNN相媲美。接下来，我们将全面评估DendSNns在各种高要求任务上的表现。通过使用上下文信号调制树枝分支强度，大大减轻了树突状SNN的灾难性遗忘。此外，与点SNN相比，DendSNN表现出对噪声和敌意攻击的更强的稳健性，并且在少镜头学习环境中表现出色。我们的工作首次证明了训练生物可信的树突状神经网络的可能性，其深度和规模与传统的点状神经网络相当，并揭示了简化的树突状神经元模型在深度学习中的优越表达能力和健壮性，从而为进一步的神经网络设计提供了一个新的视角。



## **21. SmartReco: Detecting Read-Only Reentrancy via Fine-Grained Cross-DApp Analysis**

SmartReco：通过细粒度跨DApp分析检测只读可重入性 cs.SE

Accepted by ICSE 2025

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2409.18468v2) [paper-pdf](http://arxiv.org/pdf/2409.18468v2)

**Authors**: Jingwen Zhang, Zibin Zheng, Yuhong Nan, Mingxi Ye, Kaiwen Ning, Yu Zhang, Weizhe Zhang

**Abstract**: Despite the increasing popularity of Decentralized Applications (DApps), they are suffering from various vulnerabilities that can be exploited by adversaries for profits. Among such vulnerabilities, Read-Only Reentrancy (called ROR in this paper), is an emerging type of vulnerability that arises from the complex interactions between DApps. In the recent three years, attack incidents of ROR have already caused around 30M USD losses to the DApp ecosystem. Existing techniques for vulnerability detection in smart contracts can hardly detect Read-Only Reentrancy attacks, due to the lack of tracking and analyzing the complex interactions between multiple DApps. In this paper, we propose SmartReco, a new framework for detecting Read-Only Reentrancy vulnerability in DApps through a novel combination of static and dynamic analysis (i.e., fuzzing) over smart contracts. The key design behind SmartReco is threefold: (1) SmartReco identifies the boundary between different DApps from the heavy-coupled cross-contract interactions. (2) SmartReco performs fine-grained static analysis to locate points of interest (i.e., entry functions) that may lead to ROR. (3) SmartReco utilizes the on-chain transaction data and performs multi-function fuzzing (i.e., the entry function and victim function) across different DApps to verify the existence of ROR. Our evaluation of a manual-labeled dataset with 45 RORs shows that SmartReco achieves a precision of 88.63% and a recall of 86.36%. In addition, SmartReco successfully detects 43 new RORs from 123 popular DApps. The total assets affected by such RORs reach around 520,000 USD.

摘要: 尽管去中心化应用程序(Dapp)越来越受欢迎，但它们正遭受着各种漏洞的困扰，这些漏洞可以被对手利用来牟利。在这些漏洞中，只读可重入性(在本文中称为RoR)是一种新兴的漏洞类型，它源于Dapp之间的复杂交互。近三年来，RoR的攻击事件已经给DAPP生态系统造成了约3000万美元的损失。由于缺乏对多个Dapp之间复杂交互的跟踪和分析，现有的智能合同漏洞检测技术很难检测到只读可重入性攻击。本文提出了一种新的框架SmartReco，该框架通过对智能合约的静态和动态分析(即模糊化)相结合的方式来检测Dapp中的只读可重入漏洞。SmartReco背后的关键设计有三个方面：(1)SmartReco从繁重耦合的跨合同交互中识别不同Dapp之间的边界。(2)SmartReco执行细粒度静态分析以定位可能导致RoR的兴趣点(即入口函数)。(3)SmartReco利用链上交易数据，跨不同Dapp执行多功能模糊化(即进入函数和受害者函数)，以验证RoR的存在。对45个RORS的人工标注数据集的测试表明，SmartReco的准确率为88.63%，召回率为86.36%。此外，SmartReco还从123个热门Dapp中成功检测出43个新的ROR。受此类不良资产影响的总资产约为52万美元。



## **22. Unidirectional focusing of light using structured diffractive surfaces**

使用结构化折射表面单向聚焦光 physics.optics

20 Pages, 6 Figures

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2412.06221v1) [paper-pdf](http://arxiv.org/pdf/2412.06221v1)

**Authors**: Yuhang Li, Tianyi Gan, Jingxi Li, Mona Jarrahi, Aydogan Ozcan

**Abstract**: Unidirectional optical systems enable selective control of light through asymmetric processing of radiation, effectively transmitting light in one direction while blocking unwanted propagation in the opposite direction. Here, we introduce a reciprocal diffractive unidirectional focusing design based on linear and isotropic diffractive layers that are structured. Using deep learning-based optimization, a cascaded set of diffractive layers are spatially engineered at the wavelength scale to focus light efficiently in the forward direction while blocking it in the opposite direction. The forward energy focusing efficiency and the backward energy suppression capabilities of this unidirectional architecture were demonstrated under various illumination angles and wavelengths, illustrating the versatility of our polarization-insensitive design. Furthermore, we demonstrated that these designs are resilient to adversarial attacks that utilize wavefront engineering from outside. Experimental validation using terahertz radiation confirmed the feasibility of this diffractive unidirectional focusing framework. Diffractive unidirectional designs can operate across different parts of the electromagnetic spectrum by scaling the resulting diffractive features proportional to the wavelength of light and will find applications in security, defense, and optical communication, among others.

摘要: 单向光学系统通过辐射的非对称处理实现了对光的选择性控制，有效地在一个方向上传输光，同时阻止在相反方向上不想要的传播。在这里，我们介绍了一种基于线性和各向同性衍射层结构的倒易衍射式单向聚焦设计。利用基于深度学习的优化方法，在波长尺度上对一组级联的衍射层进行空间设计，以有效地在正向聚焦光，同时在相反方向上阻挡光。在不同的照明角度和波长下，该单向结构的前向能量聚焦效率和后向能量抑制能力得到了演示，说明了我们的偏振不敏感设计的多功能性。此外，我们还证明了这些设计能够抵抗来自外部的利用波前工程的敌意攻击。利用太赫兹辐射进行的实验验证证实了这种衍射式单向聚焦框架的可行性。衍射式单向设计可以通过将产生的绕射特征与光的波长成比例地缩放，从而在电磁频谱的不同部分运行，并将在安全、国防和光通信等方面得到应用。



## **23. A Real-Time Defense Against Object Vanishing Adversarial Patch Attacks for Object Detection in Autonomous Vehicles**

针对自动驾驶汽车中的目标检测的对象消失对抗补丁攻击的实时防御 cs.CV

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2412.06215v1) [paper-pdf](http://arxiv.org/pdf/2412.06215v1)

**Authors**: Jaden Mu

**Abstract**: Autonomous vehicles (AVs) increasingly use DNN-based object detection models in vision-based perception. Correct detection and classification of obstacles is critical to ensure safe, trustworthy driving decisions. Adversarial patches aim to fool a DNN with intentionally generated patterns concentrated in a localized region of an image. In particular, object vanishing patch attacks can cause object detection models to fail to detect most or all objects in a scene, posing a significant practical threat to AVs.   This work proposes ADAV (Adversarial Defense for Autonomous Vehicles), a novel defense methodology against object vanishing patch attacks specifically designed for autonomous vehicles. Unlike existing defense methods which have high latency or are designed for static images, ADAV runs in real-time and leverages contextual information from prior frames in an AV's video feed. ADAV checks if the object detector's output for the target frame is temporally consistent with the output from a previous reference frame to detect the presence of a patch. If the presence of a patch is detected, ADAV uses gradient-based attribution to localize adversarial pixels that break temporal consistency. This two stage procedure allows ADAV to efficiently process clean inputs, and both stages are optimized to be low latency. ADAV is evaluated using real-world driving data from the Berkeley Deep Drive BDD100K dataset, and demonstrates high adversarial and clean performance.

摘要: 在基于视觉的感知中，自动驾驶车辆越来越多地使用基于DNN的目标检测模型。正确检测和分类障碍物对于确保安全、可靠的驾驶决策至关重要。恶意补丁的目的是利用集中在图像局部区域的故意生成的模式来愚弄DNN。特别是，目标消失片攻击会导致目标检测模型无法检测到场景中的大部分或所有目标，这对自动驾驶系统构成了重大的实际威胁。提出了一种针对自主车辆目标消失补丁攻击的新型防御方法--自主车辆对抗防御方法。与现有的高延迟或专为静态图像设计的防御方法不同，ADAV实时运行，并利用AV视频馈送中先前帧的上下文信息。ADAV检查目标帧的对象检测器的输出是否在时间上与来自前一参考帧的输出一致，以检测补丁的存在。如果检测到补丁的存在，ADAV使用基于梯度的属性来定位破坏时间一致性的对抗性像素。这两个阶段的程序允许ADAV高效地处理干净的输入，并且两个阶段都被优化为低延迟。ADAV使用来自Berkeley Deep Drive BDD100K数据集的真实驾驶数据进行评估，并展示了高对抗性和干净的性能。



## **24. Credible fusion of evidence in distributed system subject to cyberattacks**

受网络攻击的分布式系统中可信地融合证据 cs.CR

29 pages, 11 figures

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2412.04496v2) [paper-pdf](http://arxiv.org/pdf/2412.04496v2)

**Authors**: Chaoxiong Ma, Yan Liang

**Abstract**: Given that distributed systems face adversarial behaviors such as eavesdropping and cyberattacks, how to ensure the evidence fusion result is credible becomes a must-be-addressed topic. Different from traditional research that assumes nodes are cooperative, we focus on three requirements for evidence fusion, i.e., preserving evidence's privacy, identifying attackers and excluding their evidence, and dissipating high-conflicting among evidence caused by random noise and interference. To this end, this paper proposes an algorithm for credible evidence fusion against cyberattacks. Firstly, the fusion strategy is constructed based on conditionalized credibility to avoid counterintuitive fusion results caused by high-conflicting. Under this strategy, distributed evidence fusion is transformed into the average consensus problem for the weighted average value by conditional credibility of multi-source evidence (WAVCCME), which implies a more concise consensus process and lower computational complexity than existing algorithms. Secondly, a state decomposition and reconstruction strategy with weight encryption is designed, and its effectiveness for privacy-preserving under directed graphs is guaranteed: decomposing states into different random sub-states for different neighbors to defend against internal eavesdroppers, and encrypting the sub-states' weight in the reconstruction to guard against out-of-system eavesdroppers. Finally, the identities and types of attackers are identified by inter-neighbor broadcasting and comparison of nodes' states, and the proposed update rule with state corrections is used to achieve the consensus of the WAVCCME. The states of normal nodes are shown to converge to their WAVCCME, while the attacker's evidence is excluded from the fusion, as verified by the simulation on a distributed unmanned reconnaissance swarm.

摘要: 鉴于分布式系统面临窃听、网络攻击等敌意行为，如何保证证据融合结果的可信成为一个必须解决的课题。与传统研究假设节点是协作的不同，本文着重研究了证据融合的三个要求，即保护证据的私密性、识别攻击者并排除他们的证据、消散随机噪声和干扰引起的证据之间的高冲突。为此，本文提出了一种针对网络攻击的可信证据融合算法。首先，构建基于条件化可信度的融合策略，避免因高度冲突而导致的反直觉融合结果。在该策略下，通过多源证据的条件可信度将分布式证据融合转化为加权平均值的平均一致性问题(WAVCCME)，这意味着比现有算法更简洁的一致性过程和更低的计算复杂度。其次，设计了一种加权加密的状态分解和重构策略，保证了其在有向图下的隐私保护有效性：将状态分解为不同的随机子状态以防止内部窃听，并加密子状态在重建中的权重以防止系统外窃听。最后，通过邻居间广播和节点状态比较来识别攻击者的身份和类型，并使用提出的带状态纠正的更新规则来实现WAVCCME的共识。正常节点的状态收敛到它们的WAVCCME，而攻击者的证据被排除在融合之外，分布式无人侦察群的仿真验证了这一点。



## **25. Privacy-Preserving Large Language Models: Mechanisms, Applications, and Future Directions**

保护隐私的大型语言模型：机制、应用和未来方向 cs.CR

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2412.06113v1) [paper-pdf](http://arxiv.org/pdf/2412.06113v1)

**Authors**: Guoshenghui Zhao, Eric Song

**Abstract**: The rapid advancement of large language models (LLMs) has revolutionized natural language processing, enabling applications in diverse domains such as healthcare, finance and education. However, the growing reliance on extensive data for training and inference has raised significant privacy concerns, ranging from data leakage to adversarial attacks. This survey comprehensively explores the landscape of privacy-preserving mechanisms tailored for LLMs, including differential privacy, federated learning, cryptographic protocols, and trusted execution environments. We examine their efficacy in addressing key privacy challenges, such as membership inference and model inversion attacks, while balancing trade-offs between privacy and model utility. Furthermore, we analyze privacy-preserving applications of LLMs in privacy-sensitive domains, highlighting successful implementations and inherent limitations. Finally, this survey identifies emerging research directions, emphasizing the need for novel frameworks that integrate privacy by design into the lifecycle of LLMs. By synthesizing state-of-the-art approaches and future trends, this paper provides a foundation for developing robust, privacy-preserving large language models that safeguard sensitive information without compromising performance.

摘要: 大型语言模型(LLM)的快速发展使自然语言处理发生了革命性的变化，使得医疗保健、金融和教育等不同领域的应用成为可能。然而，越来越多地依赖大量数据进行训练和推理，引发了从数据泄露到对抗性攻击等严重的隐私问题。这项调查全面探索了为LLMS量身定做的隐私保护机制，包括差异隐私、联合学习、加密协议和可信执行环境。我们检查了它们在解决关键隐私挑战方面的有效性，例如成员关系推断和模型反转攻击，同时平衡隐私和模型实用程序之间的权衡。此外，我们分析了LLMS在隐私敏感领域的隐私保护应用，强调了成功的实现和固有的局限性。最后，这项调查确定了新兴的研究方向，强调需要新的框架，将隐私通过设计整合到低成本管理的生命周期中。通过综合最先进的方法和未来的趋势，本文为开发健壮的、保护隐私的大型语言模型提供了基础，这些模型可以在不影响性能的情况下保护敏感信息。



## **26. TrojanForge: Generating Adversarial Hardware Trojan Examples Using Reinforcement Learning**

TrojanForge：使用强化学习生成对抗性硬件特洛伊示例 cs.CR

**SubmitDate**: 2024-12-08    [abs](http://arxiv.org/abs/2405.15184v3) [paper-pdf](http://arxiv.org/pdf/2405.15184v3)

**Authors**: Amin Sarihi, Peter Jamieson, Ahmad Patooghy, Abdel-Hameed A. Badawy

**Abstract**: The Hardware Trojan (HT) problem can be thought of as a continuous game between attackers and defenders, each striving to outsmart the other by leveraging any available means for an advantage. Machine Learning (ML) has recently played a key role in advancing HT research. Various novel techniques, such as Reinforcement Learning (RL) and Graph Neural Networks (GNNs), have shown HT insertion and detection capabilities. HT insertion with ML techniques, specifically, has seen a spike in research activity due to the shortcomings of conventional HT benchmarks and the inherent human design bias that occurs when we create them. This work continues this innovation by presenting a tool called TrojanForge, capable of generating HT adversarial examples that defeat HT detectors; demonstrating the capabilities of GAN-like adversarial tools for automatic HT insertion. We introduce an RL environment where the RL insertion agent interacts with HT detectors in an insertion-detection loop where the agent collects rewards based on its success in bypassing HT detectors. Our results show that this process helps inserted HTs evade various HT detectors, achieving high attack success percentages. This tool provides insight into why HT insertion fails in some instances and how we can leverage this knowledge in defense.

摘要: 硬件特洛伊木马(HT)问题可以被认为是攻击者和防御者之间的一场持续的游戏，双方都在努力利用任何可用的手段来获取优势，以智胜对方。近年来，机器学习(ML)在推进HT研究中发挥了关键作用。各种新的技术，如强化学习(RL)和图形神经网络(GNNS)，已经显示出HT插入和检测能力。特别是，由于传统的HT基准的缺点以及我们创建它们时固有的人为设计偏差，使用ML技术插入HT的研究活动出现了激增。这项工作通过提供一个名为TrojanForge的工具来继续这一创新，该工具能够生成击败HT检测器的HT对抗示例；展示了GAN类对抗工具自动插入HT的能力。我们引入了一种RL环境，在该环境中，RL插入剂与HT检测器在插入检测环路中相互作用，其中代理根据其成功绕过HT检测器来收取奖励。我们的结果表明，这个过程帮助插入的HTS避开了各种HT探测器，获得了高攻击成功率。这个工具提供了一些关于为什么在某些情况下HT插入失败以及我们如何在防御中利用这一知识的洞察。



## **27. Anti-Reference: Universal and Immediate Defense Against Reference-Based Generation**

反引用：针对基于引用的一代的普遍和立即防御 cs.CV

**SubmitDate**: 2024-12-08    [abs](http://arxiv.org/abs/2412.05980v1) [paper-pdf](http://arxiv.org/pdf/2412.05980v1)

**Authors**: Yiren Song, Shengtao Lou, Xiaokang Liu, Hai Ci, Pei Yang, Jiaming Liu, Mike Zheng Shou

**Abstract**: Diffusion models have revolutionized generative modeling with their exceptional ability to produce high-fidelity images. However, misuse of such potent tools can lead to the creation of fake news or disturbing content targeting individuals, resulting in significant social harm. In this paper, we introduce Anti-Reference, a novel method that protects images from the threats posed by reference-based generation techniques by adding imperceptible adversarial noise to the images. We propose a unified loss function that enables joint attacks on fine-tuning-based customization methods, non-fine-tuning customization methods, and human-centric driving methods. Based on this loss, we train a Adversarial Noise Encoder to predict the noise or directly optimize the noise using the PGD method. Our method shows certain transfer attack capabilities, effectively challenging both gray-box models and some commercial APIs. Extensive experiments validate the performance of Anti-Reference, establishing a new benchmark in image security.

摘要: 扩散模型以其产生高保真图像的非凡能力彻底改变了生成式建模。然而，滥用这种强有力的工具可能会导致制造假新闻或针对个人的令人不安的内容，造成重大的社会危害。在本文中，我们介绍了一种新的方法--反引用，它通过在图像中添加不可察觉的对抗性噪声来保护图像免受基于引用的生成技术带来的威胁。我们提出了一个统一的损失函数，可以对基于微调的定制方法、非微调的定制方法和以人为中心的驱动方法进行联合攻击。基于这一损失，我们训练了一个对抗性噪声编码器来预测噪声或使用PGD方法直接优化噪声。我们的方法显示了一定的传输攻击能力，有效地挑战了灰盒模型和一些商业API。广泛的实验验证了反引用的性能，为图像安全建立了一个新的基准。



## **28. Revisiting DeepFool: generalization and improvement**

重温DeepFool：概括与改进 cs.LG

**SubmitDate**: 2024-12-08    [abs](http://arxiv.org/abs/2303.12481v2) [paper-pdf](http://arxiv.org/pdf/2303.12481v2)

**Authors**: Alireza Abdollahpoorrostam, Mahed Abroshan, Seyed-Mohsen Moosavi-Dezfooli

**Abstract**: Deep neural networks have been known to be vulnerable to adversarial examples, which are inputs that are modified slightly to fool the network into making incorrect predictions. This has led to a significant amount of research on evaluating the robustness of these networks against such perturbations. One particularly important robustness metric is the robustness to minimal $\ell_2$ adversarial perturbations. However, existing methods for evaluating this robustness metric are either computationally expensive or not very accurate. In this paper, we introduce a new family of adversarial attacks that strike a balance between effectiveness and computational efficiency. Our proposed attacks are generalizations of the well-known DeepFool (DF) attack, while they remain simple to understand and implement. We demonstrate that our attacks outperform existing methods in terms of both effectiveness and computational efficiency. Our proposed attacks are also suitable for evaluating the robustness of large models and can be used to perform adversarial training (AT) to achieve state-of-the-art robustness to minimal $\ell_2$ adversarial perturbations.

摘要: 众所周知，深度神经网络很容易受到敌意例子的攻击，这些例子是对输入进行了轻微修改，以愚弄网络做出错误的预测。这导致了大量关于评估这些网络对此类扰动的稳健性的研究。一个特别重要的稳健性度量是对最小$\ell_2$对抗扰动的稳健性。然而，现有的评估这种稳健性度量的方法要么计算昂贵，要么不太准确。在本文中，我们引入了一类新的对抗性攻击，它们在有效性和计算效率之间取得了平衡。我们提出的攻击是众所周知的DeepFool(DF)攻击的推广，但它们仍然易于理解和实现。我们证明了我们的攻击在有效性和计算效率方面都优于现有的方法。我们提出的攻击也适用于评估大型模型的稳健性，并可用于执行对抗性训练(AT)，以获得对最小的对抗性扰动的最先进的稳健性。



## **29. TrojanRobot: Backdoor Attacks Against LLM-based Embodied Robots in the Physical World**

TrojanRobot：对物理世界中基于LLM的机器人的后门攻击 cs.RO

Initial version with preliminary results. We welcome any feedback or  suggestions

**SubmitDate**: 2024-12-08    [abs](http://arxiv.org/abs/2411.11683v2) [paper-pdf](http://arxiv.org/pdf/2411.11683v2)

**Authors**: Xianlong Wang, Hewen Pan, Hangtao Zhang, Minghui Li, Shengshan Hu, Ziqi Zhou, Lulu Xue, Peijin Guo, Yichen Wang, Wei Wan, Aishan Liu, Leo Yu Zhang

**Abstract**: Robotic manipulation refers to the autonomous handling and interaction of robots with objects using advanced techniques in robotics and artificial intelligence. The advent of powerful tools such as large language models (LLMs) and large vision-language models (LVLMs) has significantly enhanced the capabilities of these robots in environmental perception and decision-making. However, the introduction of these intelligent agents has led to security threats such as jailbreak attacks and adversarial attacks.   In this research, we take a further step by proposing a backdoor attack specifically targeting robotic manipulation and, for the first time, implementing backdoor attack in the physical world. By embedding a backdoor visual language model into the visual perception module within the robotic system, we successfully mislead the robotic arm's operation in the physical world, given the presence of common items as triggers. Experimental evaluations in the physical world demonstrate the effectiveness of the proposed backdoor attack.

摘要: 机器人操纵是指使用机器人学和人工智能的先进技术，自主处理机器人与物体的交互。大型语言模型(LLM)和大型视觉语言模型(LVLM)等强大工具的出现，大大增强了这些机器人在环境感知和决策方面的能力。然而，这些智能代理的引入导致了越狱攻击和对抗性攻击等安全威胁。在这项研究中，我们进一步提出了专门针对机器人操作的后门攻击，并首次在物理世界中实现了后门攻击。通过将后门视觉语言模型嵌入机器人系统的视觉感知模块中，我们成功地误导了机械臂在物理世界中的操作，因为存在共同的物品作为触发器。物理世界中的实验评估证明了所提出的后门攻击的有效性。



## **30. Adversarial Transferability in Deep Denoising Models: Theoretical Insights and Robustness Enhancement via Out-of-Distribution Typical Set Sampling**

深度去噪模型中的对抗可移植性：通过非分布典型集抽样的理论见解和鲁棒性增强 cs.CV

**SubmitDate**: 2024-12-08    [abs](http://arxiv.org/abs/2412.05943v1) [paper-pdf](http://arxiv.org/pdf/2412.05943v1)

**Authors**: Jie Ning, Jiebao Sun, Shengzhu Shi, Zhichang Guo, Yao Li, Hongwei Li, Boying Wu

**Abstract**: Deep learning-based image denoising models demonstrate remarkable performance, but their lack of robustness analysis remains a significant concern. A major issue is that these models are susceptible to adversarial attacks, where small, carefully crafted perturbations to input data can cause them to fail. Surprisingly, perturbations specifically crafted for one model can easily transfer across various models, including CNNs, Transformers, unfolding models, and plug-and-play models, leading to failures in those models as well. Such high adversarial transferability is not observed in classification models. We analyze the possible underlying reasons behind the high adversarial transferability through a series of hypotheses and validation experiments. By characterizing the manifolds of Gaussian noise and adversarial perturbations using the concept of typical set and the asymptotic equipartition property, we prove that adversarial samples deviate slightly from the typical set of the original input distribution, causing the models to fail. Based on these insights, we propose a novel adversarial defense method: the Out-of-Distribution Typical Set Sampling Training strategy (TS). TS not only significantly enhances the model's robustness but also marginally improves denoising performance compared to the original model.

摘要: 基于深度学习的图像去噪模型表现出显著的去噪性能，但其缺乏稳健性分析仍然是一个值得关注的问题。一个主要问题是，这些模型容易受到敌意攻击，在这种攻击中，输入数据时精心设计的微小扰动可能会导致它们失败。令人惊讶的是，专门为一个模型设计的扰动可以很容易地在各种模型之间传递，包括CNN、Transformers、展开模型和即插即用模型，这也会导致这些模型中的故障。在分类模型中没有观察到如此高的对抗性可转移性。我们通过一系列的假设和验证实验，分析了高对抗可转移性背后可能的潜在原因。利用典型集的概念和渐近均分性质刻画了高斯噪声和对抗性扰动的流形，证明了对抗性样本与原始输入分布的典型集有轻微偏离，从而导致模型失效。基于这些见解，我们提出了一种新的对抗防御方法：分布外典型集抽样训练策略(TS)。与原始模型相比，TS不仅显著增强了模型的稳健性，而且还略微改善了去噪性能。



## **31. BAMBA: A Bimodal Adversarial Multi-Round Black-Box Jailbreak Attacker for LVLMs**

BAMBA：LVLM的双峰对抗多轮黑匣子越狱攻击者 cs.CR

A Bimodal Adversarial Multi-Round Black-Box Jailbreak Attacker for  LVLMs

**SubmitDate**: 2024-12-08    [abs](http://arxiv.org/abs/2412.05892v1) [paper-pdf](http://arxiv.org/pdf/2412.05892v1)

**Authors**: Ruoxi Cheng, Yizhong Ding, Shuirong Cao, Shaowei Yuan, Zhiqiang Wang, Xiaojun Jia

**Abstract**: LVLMs are widely used but vulnerable to illegal or unethical responses under jailbreak attacks. To ensure their responsible deployment in real-world applications, it is essential to understand their vulnerabilities. There are four main issues in current work: single-round attack limitation, insufficient dual-modal synergy, poor transferability to black-box models, and reliance on prompt engineering. To address these limitations, we propose BAMBA, a bimodal adversarial multi-round black-box jailbreak attacker for LVLMs. We first use an image optimizer to learn malicious features from a harmful corpus, then deepen these features through a bimodal optimizer through text-image interaction, generating adversarial text and image for jailbreak. Experiments on various LVLMs and datasets demonstrate that BAMBA outperforms other baselines.

摘要: LVLM被广泛使用，但在越狱攻击下很容易受到非法或不道德的反应。为了确保它们在现实世界的应用程序中负责任地部署，了解它们的漏洞至关重要。当前工作中存在四个主要问题：单轮攻击限制、双模式协同作用不足、到黑匣子模型的可移植性较差以及对即时工程的依赖。为了解决这些限制，我们提出BAMBA，这是一种针对LVLM的双峰对抗性多轮黑匣子越狱攻击者。我们首先使用图像优化器从有害的数据库中学习恶意特征，然后通过文本-图像交互通过双峰优化器加深这些特征，生成对抗性文本和图像以供越狱。对各种LVLM和数据集的实验表明BAMBA优于其他基线。



## **32. Understanding the Impact of Graph Reduction on Adversarial Robustness in Graph Neural Networks**

了解图约简对图神经网络中对抗鲁棒性的影响 cs.LG

**SubmitDate**: 2024-12-08    [abs](http://arxiv.org/abs/2412.05883v1) [paper-pdf](http://arxiv.org/pdf/2412.05883v1)

**Authors**: Kerui Wu, Ka-Ho Chow, Wenqi Wei, Lei Yu

**Abstract**: As Graph Neural Networks (GNNs) become increasingly popular for learning from large-scale graph data across various domains, their susceptibility to adversarial attacks when using graph reduction techniques for scalability remains underexplored. In this paper, we present an extensive empirical study to investigate the impact of graph reduction techniques, specifically graph coarsening and sparsification, on the robustness of GNNs against adversarial attacks. Through extensive experiments involving multiple datasets and GNN architectures, we examine the effects of four sparsification and six coarsening methods on the poisoning attacks. Our results indicate that, while graph sparsification can mitigate the effectiveness of certain poisoning attacks, such as Mettack, it has limited impact on others, like PGD. Conversely, graph coarsening tends to amplify the adversarial impact, significantly reducing classification accuracy as the reduction ratio decreases. Additionally, we provide a novel analysis of the causes driving these effects and examine how defensive GNN models perform under graph reduction, offering practical insights for designing robust GNNs within graph acceleration systems.

摘要: 随着图神经网络(GNN)越来越多地用于从各个领域的大规模图数据中学习，当使用图归约技术来实现可伸缩性时，其对敌意攻击的敏感性仍然没有得到充分的研究。在这篇文章中，我们提供了一个广泛的实证研究来研究图归约技术，特别是图的粗化和稀疏化技术，对GNN抵抗敌意攻击的健壮性的影响。通过涉及多个数据集和GNN体系结构的大量实验，我们考察了四种稀疏和六种粗化方法对中毒攻击的影响。我们的结果表明，尽管图稀疏可以降低某些中毒攻击(如Mettack)的有效性，但它对其他攻击(如PGD)的影响有限。相反，图形粗化倾向于放大对抗性影响，随着缩减率的降低，分类精度显著降低。此外，我们对这些影响的原因进行了新颖的分析，并研究了防御性GNN模型在图化简下的表现，为在图加速系统中设计健壮的GNN提供了实用的见解。



## **33. DeMem: Privacy-Enhanced Robust Adversarial Learning via De-Memorization**

DeMem：通过去伪化的隐私增强鲁棒对抗学习 cs.LG

10 pages

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.05767v2) [paper-pdf](http://arxiv.org/pdf/2412.05767v2)

**Authors**: Xiaoyu Luo, Qiongxiu Li

**Abstract**: Adversarial robustness, the ability of a model to withstand manipulated inputs that cause errors, is essential for ensuring the trustworthiness of machine learning models in real-world applications. However, previous studies have shown that enhancing adversarial robustness through adversarial training increases vulnerability to privacy attacks. While differential privacy can mitigate these attacks, it often compromises robustness against both natural and adversarial samples. Our analysis reveals that differential privacy disproportionately impacts low-risk samples, causing an unintended performance drop. To address this, we propose DeMem, which selectively targets high-risk samples, achieving a better balance between privacy protection and model robustness. DeMem is versatile and can be seamlessly integrated into various adversarial training techniques. Extensive evaluations across multiple training methods and datasets demonstrate that DeMem significantly reduces privacy leakage while maintaining robustness against both natural and adversarial samples. These results confirm DeMem's effectiveness and broad applicability in enhancing privacy without compromising robustness.

摘要: 对抗性健壮性，即模型承受导致错误的操纵输入的能力，对于确保机器学习模型在现实世界应用中的可信性至关重要。然而，先前的研究表明，通过对抗性训练来增强对抗性的健壮性会增加对隐私攻击的脆弱性。虽然差异隐私可以缓解这些攻击，但它通常会损害对自然样本和对手样本的稳健性。我们的分析显示，差异隐私对低风险样本的影响不成比例，导致意外的性能下降。为了解决这一问题，我们提出了DeMem，它选择性地针对高风险样本，在隐私保护和模型稳健性之间实现了更好的平衡。DeMem是多才多艺的，可以无缝地整合到各种对抗性训练技术中。对多种训练方法和数据集的广泛评估表明，DeMem显著减少了隐私泄露，同时保持了对自然样本和对手样本的健壮性。这些结果证实了DeMem在增强隐私而不影响健壮性方面的有效性和广泛的适用性。



## **34. Query-Based Adversarial Prompt Generation**

基于查询的对抗提示生成 cs.CL

**SubmitDate**: 2024-12-07    [abs](http://arxiv.org/abs/2402.12329v2) [paper-pdf](http://arxiv.org/pdf/2402.12329v2)

**Authors**: Jonathan Hayase, Ema Borevkovic, Nicholas Carlini, Florian Tramèr, Milad Nasr

**Abstract**: Recent work has shown it is possible to construct adversarial examples that cause an aligned language model to emit harmful strings or perform harmful behavior. Existing attacks work either in the white-box setting (with full access to the model weights), or through transferability: the phenomenon that adversarial examples crafted on one model often remain effective on other models. We improve on prior work with a query-based attack that leverages API access to a remote language model to construct adversarial examples that cause the model to emit harmful strings with (much) higher probability than with transfer-only attacks. We validate our attack on GPT-3.5 and OpenAI's safety classifier; we can cause GPT-3.5 to emit harmful strings that current transfer attacks fail at, and we can evade the safety classifier with nearly 100% probability.

摘要: 最近的工作表明，可以构建对抗性示例，从而导致对齐的语言模型发出有害字符串或执行有害行为。现有的攻击要么在白盒设置（完全访问模型权重）中工作，要么通过可移植性来工作：在一个模型上制作的对抗性示例通常对其他模型仍然有效。我们改进了之前使用基于查询的攻击的工作，该攻击利用API对远程语言模型的访问来构建对抗性示例，这些示例导致模型以比仅传输攻击高（远）的可能性发出有害字符串。我们验证了对GPT-3.5和OpenAI安全分类器的攻击;我们可以导致GPT-3.5发出当前传输攻击失败的有害字符串，并且我们可以以近100%的可能性规避安全分类器。



## **35. REGE: A Method for Incorporating Uncertainty in Graph Embeddings**

REGE：一种消除图嵌入中不确定性的方法 cs.LG

**SubmitDate**: 2024-12-07    [abs](http://arxiv.org/abs/2412.05735v1) [paper-pdf](http://arxiv.org/pdf/2412.05735v1)

**Authors**: Zohair Shafi, Germans Savcisens, Tina Eliassi-Rad

**Abstract**: Machine learning models for graphs in real-world applications are prone to two primary types of uncertainty: (1) those that arise from incomplete and noisy data and (2) those that arise from uncertainty of the model in its output. These sources of uncertainty are not mutually exclusive. Additionally, models are susceptible to targeted adversarial attacks, which exacerbate both of these uncertainties. In this work, we introduce Radius Enhanced Graph Embeddings (REGE), an approach that measures and incorporates uncertainty in data to produce graph embeddings with radius values that represent the uncertainty of the model's output. REGE employs curriculum learning to incorporate data uncertainty and conformal learning to address the uncertainty in the model's output. In our experiments, we show that REGE's graph embeddings perform better under adversarial attacks by an average of 1.5% (accuracy) against state-of-the-art methods.

摘要: 现实世界应用程序中的图形机器学习模型容易出现两种主要类型的不确定性：（1）由不完整和有噪音的数据引起的不确定性;（2）由模型输出的不确定性引起的不确定性。这些不确定性来源并不相互排斥。此外，模型很容易受到有针对性的对抗攻击，这加剧了这两种不确定性。在这项工作中，我们引入了半径增强图嵌入（REGE），这是一种测量和合并数据中的不确定性的方法，以生成具有代表模型输出不确定性的半径值的图嵌入。REGE利用课程学习将数据不确定性和保形学习结合起来，以解决模型输出中的不确定性。在我们的实验中，我们表明，与最先进的方法相比，REGE的图嵌入在对抗性攻击下的表现平均更好1.5%（准确性）。



## **36. PrivAgent: Agentic-based Red-teaming for LLM Privacy Leakage**

PrivAgent：针对LLM隐私泄露的基于统计的红色团队 cs.CR

**SubmitDate**: 2024-12-07    [abs](http://arxiv.org/abs/2412.05734v1) [paper-pdf](http://arxiv.org/pdf/2412.05734v1)

**Authors**: Yuzhou Nie, Zhun Wang, Ye Yu, Xian Wu, Xuandong Zhao, Wenbo Guo, Dawn Song

**Abstract**: Recent studies have discovered that LLMs have serious privacy leakage concerns, where an LLM may be fooled into outputting private information under carefully crafted adversarial prompts. These risks include leaking system prompts, personally identifiable information, training data, and model parameters. Most existing red-teaming approaches for privacy leakage rely on humans to craft the adversarial prompts. A few automated methods are proposed for system prompt extraction, but they cannot be applied to more severe risks (e.g., training data extraction) and have limited effectiveness even for system prompt extraction.   In this paper, we propose PrivAgent, a novel black-box red-teaming framework for LLM privacy leakage. We formulate different risks as a search problem with a unified attack goal. Our framework trains an open-source LLM through reinforcement learning as the attack agent to generate adversarial prompts for different target models under different risks. We propose a novel reward function to provide effective and fine-grained rewards for the attack agent. Finally, we introduce customizations to better fit our general framework to system prompt extraction and training data extraction. Through extensive evaluations, we first show that PrivAgent outperforms existing automated methods in system prompt leakage against six popular LLMs. Notably, our approach achieves a 100% success rate in extracting system prompts from real-world applications in OpenAI's GPT Store. We also show PrivAgent's effectiveness in extracting training data from an open-source LLM with a success rate of 5.9%. We further demonstrate PrivAgent's effectiveness in evading the existing guardrail defense and its helpfulness in enabling better safety alignment. Finally, we validate our customized designs through a detailed ablation study. We release our code here https://github.com/rucnyz/RedAgent.

摘要: 最近的研究发现，LLM存在严重的隐私泄露问题，在这种情况下，LLM可能会被愚弄，在精心设计的敌意提示下输出私人信息。这些风险包括泄露系统提示、个人身份信息、培训数据和模型参数。大多数现有的隐私泄露红团队方法都依赖于人类来制作敌意提示。一些自动化的方法被提出用于系统提示提取，但它们不能应用于更严重的风险(例如，训练数据提取)，即使对于系统提示提取，其有效性也有限。在本文中，我们提出了一种新的针对LLM隐私泄露的黑盒红队框架PrivAgent。我们将不同的风险表示为具有统一攻击目标的搜索问题。该框架通过强化学习训练开源的LLM作为攻击代理，在不同的风险下为不同的目标模型生成对抗性提示。我们提出了一种新的奖励函数，为攻击代理提供有效和细粒度的奖励。最后，我们引入了定制，以更好地适合我们的通用框架来提取系统提示和训练数据。通过广泛的评估，我们首先表明PrivAgent在系统提示泄漏方面优于现有的自动化方法，并与六种流行的LLM进行了比较。值得注意的是，我们的方法在从OpenAI的GPT商店的真实应用程序中提取系统提示方面取得了100%的成功率。我们还展示了PrivAgent在从开源LLM中提取训练数据的有效性，成功率为5.9%。我们进一步展示了PrivAgent在避开现有护栏防御方面的有效性，以及它在实现更好的安全对齐方面的帮助。最后，我们通过一个详细的烧蚀实验验证了我们的定制设计。我们在这里发布代码https://github.com/rucnyz/RedAgent.



## **37. Nearly Solved? Robust Deepfake Detection Requires More than Visual Forensics**

快解决了？强大的Deepfake检测需要的不仅仅是视觉取证 cs.CV

**SubmitDate**: 2024-12-07    [abs](http://arxiv.org/abs/2412.05676v1) [paper-pdf](http://arxiv.org/pdf/2412.05676v1)

**Authors**: Guy Levy, Nathan Liebmann

**Abstract**: Deepfakes are on the rise, with increased sophistication and prevalence allowing for high-profile social engineering attacks. Detecting them in the wild is therefore important as ever, giving rise to new approaches breaking benchmark records in this task. In line with previous work, we show that recently developed state-of-the-art detectors are susceptible to classical adversarial attacks, even in a highly-realistic black-box setting, putting their usability in question. We argue that crucial 'robust features' of deepfakes are in their higher semantics, and follow that with evidence that a detector based on a semantic embedding model is less susceptible to black-box perturbation attacks. We show that large visuo-lingual models like GPT-4o can perform zero-shot deepfake detection better than current state-of-the-art methods, and introduce a novel attack based on high-level semantic manipulation. Finally, we argue that hybridising low- and high-level detectors can improve adversarial robustness, based on their complementary strengths and weaknesses.

摘要: Deepfake正在增加，随着复杂性和普及率的提高，高调的社会工程攻击成为可能。因此，在野外检测它们一如既往地重要，这将产生打破这项任务基准纪录的新方法。与以前的工作一致，我们表明最近开发的最先进的检测器容易受到经典的对抗性攻击，即使在高度逼真的黑盒环境中也是如此，这使得它们的可用性受到质疑。我们认为，深度假冒的关键“稳健特征”在于它们的较高语义，然后有证据表明，基于语义嵌入模型的检测器不太容易受到黑盒扰动攻击。我们证明了像GPT-40这样的大型视觉语言模型可以比目前最先进的方法更好地进行零射击深伪检测，并引入了一种新的基于高层语义操作的攻击。最后，我们认为，基于低级别检测器和高级检测器的互补优势和劣势，混合低级别检测器可以提高对手的稳健性。



## **38. From Flexibility to Manipulation: The Slippery Slope of XAI Evaluation**

从灵活性到操纵性：XAI评估的滑动斜坡 cs.AI

Published in ECCV 2024 Workshop on Explainable Computer Vision: Where  are We and Where are We Going? Shorter non-archival version also appeared in  the NeurIPS 2024 Interpretable AI workshop. Code is available at  \url{https://github.com/Wickstrom/quantitative-xai-manipulation}

**SubmitDate**: 2024-12-07    [abs](http://arxiv.org/abs/2412.05592v1) [paper-pdf](http://arxiv.org/pdf/2412.05592v1)

**Authors**: Kristoffer Wickstrøm, Marina Marie-Claire Höhne, Anna Hedström

**Abstract**: The lack of ground truth explanation labels is a fundamental challenge for quantitative evaluation in explainable artificial intelligence (XAI). This challenge becomes especially problematic when evaluation methods have numerous hyperparameters that must be specified by the user, as there is no ground truth to determine an optimal hyperparameter selection. It is typically not feasible to do an exhaustive search of hyperparameters so researchers typically make a normative choice based on similar studies in the literature, which provides great flexibility for the user. In this work, we illustrate how this flexibility can be exploited to manipulate the evaluation outcome. We frame this manipulation as an adversarial attack on the evaluation where seemingly innocent changes in hyperparameter setting significantly influence the evaluation outcome. We demonstrate the effectiveness of our manipulation across several datasets with large changes in evaluation outcomes across several explanation methods and models. Lastly, we propose a mitigation strategy based on ranking across hyperparameters that aims to provide robustness towards such manipulation. This work highlights the difficulty of conducting reliable XAI evaluation and emphasizes the importance of a holistic and transparent approach to evaluation in XAI.

摘要: 缺乏地面真相解释标签是可解释人工智能(XAI)定量评估的根本挑战。当评估方法具有许多必须由用户指定的超参数时，这一挑战变得特别成问题，因为没有基本事实来确定最佳超参数选择。对超参数进行详尽的搜索通常是不可行的，因此研究人员通常基于文献中的类似研究做出标准化选择，这为用户提供了很大的灵活性。在这项工作中，我们说明了如何利用这种灵活性来操纵评估结果。我们将这种操作视为对评估的对抗性攻击，在这种情况下，超参数设置中看似无害的变化会显著影响评估结果。我们在几种解释方法和模型的评估结果有很大变化的情况下，展示了我们在几个数据集上操作的有效性。最后，我们提出了一种基于超参数排序的缓解策略，旨在为此类操作提供健壮性。这项工作突出了进行可靠的XAI评估的困难，并强调了在XAI进行全面和透明的评估的重要性。



## **39. Practical Region-level Attack against Segment Anything Models**

针对Segment Anything模型的实用区域级攻击 cs.CV

Code is released at https://github.com/ShenYifanS/S-RA_T-RA

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2404.08255v2) [paper-pdf](http://arxiv.org/pdf/2404.08255v2)

**Authors**: Yifan Shen, Zhengyuan Li, Gang Wang

**Abstract**: Segment Anything Models (SAM) have made significant advancements in image segmentation, allowing users to segment target portions of an image with a single click (i.e., user prompt). Given its broad applications, the robustness of SAM against adversarial attacks is a critical concern. While recent works have explored adversarial attacks against a pre-defined prompt/click, their threat model is not yet realistic: (1) they often assume the user-click position is known to the attacker (point-based attack), and (2) they often operate under a white-box setting with limited transferability. In this paper, we propose a more practical region-level attack where attackers do not need to know the precise user prompt. The attack remains effective as the user clicks on any point on the target object in the image, hiding the object from SAM. Also, by adapting a spectrum transformation method, we make the attack more transferable under a black-box setting. Both control experiments and testing against real-world SAM services confirm its effectiveness.

摘要: 分割任何模型(SAM)在图像分割方面取得了重大进展，允许用户通过一次点击(即用户提示)来分割图像的目标部分。考虑到其广泛的应用，SAM对对手攻击的稳健性是一个关键问题。虽然最近的研究已经探索了针对预定义提示/点击的对抗性攻击，但他们的威胁模型还不现实：(1)它们通常假设攻击者知道用户点击的位置(基于点的攻击)，以及(2)它们通常在可转移性有限的白盒设置下操作。在本文中，我们提出了一种更实用的区域级攻击，攻击者不需要知道准确的用户提示。当用户点击图像中目标对象上的任何点时，该攻击仍然有效，从而隐藏该对象以躲避SAM。此外，通过采用频谱变换的方法，使得攻击在黑盒环境下更具可转移性。对照实验和针对真实世界SAM服务的测试都证实了该方法的有效性。



## **40. LIAR: Leveraging Alignment (Best-of-N) to Jailbreak LLMs in Seconds**

LIAR：利用联盟（N中最佳）在几秒钟内越狱LLM cs.CL

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2412.05232v1) [paper-pdf](http://arxiv.org/pdf/2412.05232v1)

**Authors**: James Beetham, Souradip Chakraborty, Mengdi Wang, Furong Huang, Amrit Singh Bedi, Mubarak Shah

**Abstract**: Many existing jailbreak techniques rely on solving discrete combinatorial optimization, while more recent approaches involve training LLMs to generate multiple adversarial prompts. However, both approaches require significant computational resources to produce even a single adversarial prompt. We hypothesize that the inefficiency of current approaches stems from an inadequate characterization of the jailbreak problem. To address this gap, we formulate the jailbreak problem in terms of alignment. By starting from an available safety-aligned model, we leverage an unsafe reward to guide the safe model towards generating unsafe outputs using alignment techniques (e.g., reinforcement learning from human feedback), effectively performing jailbreaking via alignment. We propose a novel jailbreak method called LIAR (LeveragIng Alignment to jailbReak). To demonstrate the simplicity and effectiveness of our approach, we employ a best-of-N method to solve the alignment problem. LIAR offers significant advantages: lower computational requirements without additional training, fully black-box operation, competitive attack success rates, and more human-readable prompts. We provide theoretical insights into the possibility of jailbreaking a safety-aligned model, revealing inherent vulnerabilities in current alignment strategies for LLMs. We also provide sub-optimality guarantees for the proposed \algo. Experimentally, we achieve ASR comparable to the SoTA with a 10x improvement to perplexity and a Time-to-Attack measured in seconds rather than tens of hours.

摘要: 许多现有的越狱技术依赖于求解离散组合优化，而更新的方法涉及训练LLM来生成多个对抗性提示。然而，这两种方法都需要大量的计算资源才能产生哪怕一个对抗性提示。我们假设，当前方法的低效源于对越狱问题的不充分描述。为了解决这一差距，我们从对齐的角度来阐述越狱问题。通过从可用的安全对齐模型开始，我们利用不安全奖励来引导安全模型使用对齐技术(例如，从人类反馈中的强化学习)来生成不安全的输出，通过对齐有效地执行越狱。我们提出了一种新的越狱方法，称为LIAR(利用对齐越狱)。为了证明我们的方法的简单性和有效性，我们使用了N中最优方法来解决比对问题。Liar提供了显著的优势：更低的计算要求，无需额外的培训，完全黑箱操作，具有竞争力的攻击成功率，以及更易读的提示。我们提供了对越狱的可能性的理论见解，安全对齐的模型，揭示了当前的低密度脂蛋白对齐策略的固有弱点。我们还为所提出的算法提供了次最优性保证。在实验中，我们获得了与SOTA相当的ASR，困惑程度提高了10倍，攻击时间以秒衡量，而不是几十小时。



## **41. A Practical Examination of AI-Generated Text Detectors for Large Language Models**

大型语言模型的人工智能生成文本检测器的实践检验 cs.CL

8 pages. Submitted to ARR October cycle

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2412.05139v1) [paper-pdf](http://arxiv.org/pdf/2412.05139v1)

**Authors**: Brian Tufts, Xuandong Zhao, Lei Li

**Abstract**: The proliferation of large language models has raised growing concerns about their misuse, particularly in cases where AI-generated text is falsely attributed to human authors. Machine-generated content detectors claim to effectively identify such text under various conditions and from any language model. This paper critically evaluates these claims by assessing several popular detectors (RADAR, Wild, T5Sentinel, Fast-DetectGPT, GPTID, LogRank, Binoculars) on a range of domains, datasets, and models that these detectors have not previously encountered. We employ various prompting strategies to simulate adversarial attacks, demonstrating that even moderate efforts can significantly evade detection. We emphasize the importance of the true positive rate at a specific false positive rate (TPR@FPR) metric and demonstrate that these detectors perform poorly in certain settings, with TPR@.01 as low as 0\%. Our findings suggest that both trained and zero-shot detectors struggle to maintain high sensitivity while achieving a reasonable true positive rate.

摘要: 大型语言模型的激增引发了人们对它们滥用的日益担忧，特别是在人工智能生成的文本被错误地归因于人类作者的情况下。机器生成的内容检测器声称可以在各种条件下从任何语言模型有效地识别此类文本。本文通过评估几种流行的探测器(雷达、Wild、T5Sentinel、Fast-DetectGPT、GPTID、logrank、双筒望远镜)在这些探测器以前从未遇到的一系列域、数据集和模型上对这些声称进行了批判性评估。我们使用各种提示策略来模拟对抗性攻击，表明即使是适度的攻击也可以显著地躲避检测。我们强调了在特定的假阳性率(TPR@fPR)度量下真阳性率的重要性，并证明了这些检测器在某些设置下的性能很差，TPR@0.01低至0\%。我们的发现表明，训练有素的探测器和零射探测器都很难在保持高灵敏度的同时获得合理的真阳性率。



## **42. On Borrowed Time -- Preventing Static Side-Channel Analysis**

论借来的时间--防止静态侧通道分析 cs.CR

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2307.09001v2) [paper-pdf](http://arxiv.org/pdf/2307.09001v2)

**Authors**: Robert Dumitru, Thorben Moos, Andrew Wabnitz, Yuval Yarom

**Abstract**: In recent years a new class of side-channel attacks has emerged. Instead of targeting device emissions during dynamic computation, adversaries now frequently exploit the leakage or response behaviour of integrated circuits in a static state. Members of this class include Static Power Side-Channel Analysis (SCA), Laser Logic State Imaging (LLSI) and Impedance Analysis (IA). Despite relying on different physical phenomena, they all enable the extraction of sensitive information from circuits in a static state with high accuracy and low noise -- a trait that poses a significant threat to many established side-channel countermeasures.   In this work, we point out the shortcomings of existing solutions and derive a simple yet effective countermeasure. We observe that in order to realise their full potential, static side-channel attacks require the targeted data to remain unchanged for a certain amount of time. For some cryptographic secrets this happens naturally, for others it requires stopping the target circuit's clock. Our proposal, called Borrowed Time, hinders an attacker's ability to leverage such idle conditions, even if full control over the global clock signal is obtained. For that, by design, key-dependent data may only be present in unprotected temporary storage when strictly needed. Borrowed Time then continuously monitors the target circuit and upon detecting an idle state, securely wipes sensitive contents.   We demonstrate the need for our countermeasure and its effectiveness by mounting practical static power SCA attacks against cryptographic systems on FPGAs, with and without Borrowed Time. In one case we attack a masked implementation and show that it is only protected with our countermeasure in place. Furthermore we demonstrate that secure on-demand wiping of sensitive data works as intended, affirming the theory that the technique also effectively hinders LLSI and IA.

摘要: 近年来，出现了一类新的旁路攻击。在动态计算期间，攻击者不再以器件排放为目标，而是频繁地利用集成电路在静态下的泄漏或响应行为。这一类的成员包括静态功率侧通道分析(SCA)、激光逻辑状态成像(LLSI)和阻抗分析(IA)。尽管依赖于不同的物理现象，但它们都能够以高精度和低噪声的方式从静态电路中提取敏感信息-这一特征对许多现有的旁路对抗措施构成了重大威胁。在这项工作中，我们指出了现有解决方案的不足，并得出了一个简单而有效的对策。我们观察到，为了充分发挥其潜力，静态侧通道攻击要求目标数据在一定时间内保持不变。对于一些密码秘密来说，这是自然而然发生的，而对另一些秘密来说，这需要停止目标电路的时钟。我们的方案称为借用时间，即使获得了对全局时钟信号的完全控制，也会阻碍攻击者利用这种空闲条件的能力。为此，根据设计，依赖于密钥的数据可能仅在严格需要时才存在于不受保护的临时存储中。然后，借用时间持续监视目标电路，并在检测到空闲状态时，安全地擦除敏感内容。我们通过对FPGA上的密码系统进行实用的静态功率SCA攻击，在有和没有借用时间的情况下，证明了我们的对策的必要性和有效性。在一种情况下，我们攻击一个被屏蔽的实现，并表明只有我们的对策到位才能保护它。此外，我们还演示了安全按需擦除敏感数据的工作原理，证实了该技术也有效地阻碍了LLSI和IA的理论。



## **43. MultiTrust: A Comprehensive Benchmark Towards Trustworthy Multimodal Large Language Models**

MultiTrust：值得信赖的多模式大型语言模型的综合基准 cs.CL

100 pages, 84 figures, 33 tables

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2406.07057v2) [paper-pdf](http://arxiv.org/pdf/2406.07057v2)

**Authors**: Yichi Zhang, Yao Huang, Yitong Sun, Chang Liu, Zhe Zhao, Zhengwei Fang, Yifan Wang, Huanran Chen, Xiao Yang, Xingxing Wei, Hang Su, Yinpeng Dong, Jun Zhu

**Abstract**: Despite the superior capabilities of Multimodal Large Language Models (MLLMs) across diverse tasks, they still face significant trustworthiness challenges. Yet, current literature on the assessment of trustworthy MLLMs remains limited, lacking a holistic evaluation to offer thorough insights into future improvements. In this work, we establish MultiTrust, the first comprehensive and unified benchmark on the trustworthiness of MLLMs across five primary aspects: truthfulness, safety, robustness, fairness, and privacy. Our benchmark employs a rigorous evaluation strategy that addresses both multimodal risks and cross-modal impacts, encompassing 32 diverse tasks with self-curated datasets. Extensive experiments with 21 modern MLLMs reveal some previously unexplored trustworthiness issues and risks, highlighting the complexities introduced by the multimodality and underscoring the necessity for advanced methodologies to enhance their reliability. For instance, typical proprietary models still struggle with the perception of visually confusing images and are vulnerable to multimodal jailbreaking and adversarial attacks; MLLMs are more inclined to disclose privacy in text and reveal ideological and cultural biases even when paired with irrelevant images in inference, indicating that the multimodality amplifies the internal risks from base LLMs. Additionally, we release a scalable toolbox for standardized trustworthiness research, aiming to facilitate future advancements in this important field. Code and resources are publicly available at: https://multi-trust.github.io/.

摘要: 尽管多模式大型语言模型(MLLM)在不同的任务中具有卓越的能力，但它们仍然面临着重大的可信性挑战。然而，目前关于评估值得信赖的MLLMS的文献仍然有限，缺乏全面的评估来提供对未来改进的透彻见解。在这项工作中，我们建立了多重信任，这是第一个关于MLLMS可信度的全面和统一的基准，涉及五个主要方面：真实性、安全性、健壮性、公平性和隐私性。我们的基准采用了严格的评估战略，同时应对多式联运风险和跨联运影响，包括32项不同的任务和自我管理的数据集。对21个现代多模式管理进行的广泛实验揭示了一些以前从未探索过的可信度问题和风险，突显了多模式带来的复杂性，并强调了先进方法提高其可靠性的必要性。例如，典型的专有模型仍然难以识别视觉上令人困惑的图像，容易受到多模式越狱和敌意攻击；MLLM更倾向于在文本中泄露隐私，甚至在推理中与无关图像搭配使用时也会暴露意识形态和文化偏见，这表明多模式放大了基本LLM的内部风险。此外，我们还发布了一个用于标准化可信度研究的可扩展工具箱，旨在促进这一重要领域的未来发展。代码和资源可在以下网址公开获得：https://multi-trust.github.io/.



## **44. Quantum Security Analysis of the Key-Alternating Ciphers**

密钥交替密码的量子安全分析 quant-ph

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2412.05026v1) [paper-pdf](http://arxiv.org/pdf/2412.05026v1)

**Authors**: Chen Bai, Mehdi Esmaili, Atul Mantri

**Abstract**: We study the security of key-alternating ciphers (KAC), a generalization of Even-Mansour ciphers over multiple rounds, which serve as abstractions for many block cipher constructions, particularly AES. While the classical security of KAC has been extensively studied, little is known about its security against quantum adversaries. In this paper, we introduce the first nontrivial quantum key-recovery attack on multi-round KAC in a model where the adversary has quantum access to only one of the public permutations. Our attack applies to any $t$-round KAC, achieving quantum query complexity of $O(2^{\frac{t(t+1)n}{(t+1)^2+1}})$, where $n$ is the size of each individual key, in a realistic quantum threat model, compared to the classical bound of $O(2^{\frac{tn}{(t+1)}})$ queries given by Bogdanev et al. (EUROCRYPT 2012). Our quantum attack leverages a novel approach based on quantum walk algorithms. Additionally, using the quantum hybrid method in our new threat model, we extend the Even-Mansour lower bound of $\Omega(2^{\frac{n}{3}})$ given by Alagic et al. (EUROCRYPT 2022) to $\Omega(2^{\frac{(t-1)n}{t}})$ for the $t$-round KAC (for $t \geq 2$).

摘要: 本文研究了密钥交替密码(KAC)的安全性，它是偶-Mansour密码在多轮上的推广，是许多分组密码构造的抽象，特别是AES。虽然KAC的经典安全性已经得到了广泛的研究，但对于它对量子对手的安全性却知之甚少。在这篇文章中，我们介绍了第一个非平凡的量子密钥恢复攻击，在一个模型中，对手只有一个公开置换的量子访问。我们的攻击适用于任何$t$轮KAC，在现实量子威胁模型中，得到的量子查询复杂度为$O(2^{\FRAC{t(t+1)n}{(t+1)^2+1})$，其中$n$是每个单独密钥的大小，而Bogdanev等人给出的经典查询数为$O(2^{\FRAC{tn}{(t+1)}})$。(欧洲密码2012)。我们的量子攻击利用了一种基于量子行走算法的新方法。此外，在我们的新威胁模型中使用量子混合方法，我们推广了Alocage等人给出的$Omega(2^{\frac{n}{3}})$的偶数-Mansour下界。(Eurocrypt 2022)到$\Omega(2^{\FRAC{(t-1)n}{t}})$，用于$t$轮KAC(用于$t\geq 2$)。



## **45. Endless Jailbreaks with Bijection Learning**

通过双射学习实现无休止的越狱 cs.CL

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2410.01294v2) [paper-pdf](http://arxiv.org/pdf/2410.01294v2)

**Authors**: Brian R. Y. Huang, Maximilian Li, Leonard Tang

**Abstract**: Despite extensive safety measures, LLMs are vulnerable to adversarial inputs, or jailbreaks, which can elicit unsafe behaviors. In this work, we introduce bijection learning, a powerful attack algorithm which automatically fuzzes LLMs for safety vulnerabilities using randomly-generated encodings whose complexity can be tightly controlled. We leverage in-context learning to teach models bijective encodings, pass encoded queries to the model to bypass built-in safety mechanisms, and finally decode responses back into English. Our attack is extremely effective on a wide range of frontier language models. Moreover, by controlling complexity parameters such as number of key-value mappings in the encodings, we find a close relationship between the capability level of the attacked LLM and the average complexity of the most effective bijection attacks. Our work highlights that new vulnerabilities in frontier models can emerge with scale: more capable models are more severely jailbroken by bijection attacks.

摘要: 尽管采取了广泛的安全措施，但LLM很容易受到敌意输入或越狱的影响，这可能会引发不安全的行为。在这项工作中，我们引入了双射学习，这是一种强大的攻击算法，它使用随机生成的编码来自动模糊LLM的安全漏洞，其复杂性可以严格控制。我们利用情景学习来教授模型双射编码，将编码的查询传递给模型以绕过内置的安全机制，最后将响应解码回英语。我们的攻击对广泛的前沿语言模型非常有效。此外，通过控制编码中的键值映射次数等复杂性参数，我们发现被攻击LLM的能力水平与最有效的双射攻击的平均复杂性之间存在密切的关系。我们的工作突出表明，前沿模型中的新漏洞可能会随着规模的扩大而出现：能力更强的模型被双射攻击越狱的情况更严重。



## **46. SleeperMark: Towards Robust Watermark against Fine-Tuning Text-to-image Diffusion Models**

SleeperMark：针对微调文本到图像扩散模型的鲁棒水印 cs.CV

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2412.04852v1) [paper-pdf](http://arxiv.org/pdf/2412.04852v1)

**Authors**: Zilan Wang, Junfeng Guo, Jiacheng Zhu, Yiming Li, Heng Huang, Muhao Chen, Zhengzhong Tu

**Abstract**: Recent advances in large-scale text-to-image (T2I) diffusion models have enabled a variety of downstream applications, including style customization, subject-driven personalization, and conditional generation. As T2I models require extensive data and computational resources for training, they constitute highly valued intellectual property (IP) for their legitimate owners, yet making them incentive targets for unauthorized fine-tuning by adversaries seeking to leverage these models for customized, usually profitable applications. Existing IP protection methods for diffusion models generally involve embedding watermark patterns and then verifying ownership through generated outputs examination, or inspecting the model's feature space. However, these techniques are inherently ineffective in practical scenarios when the watermarked model undergoes fine-tuning, and the feature space is inaccessible during verification ((i.e., black-box setting). The model is prone to forgetting the previously learned watermark knowledge when it adapts to a new task. To address this challenge, we propose SleeperMark, a novel framework designed to embed resilient watermarks into T2I diffusion models. SleeperMark explicitly guides the model to disentangle the watermark information from the semantic concepts it learns, allowing the model to retain the embedded watermark while continuing to be fine-tuned to new downstream tasks. Our extensive experiments demonstrate the effectiveness of SleeperMark across various types of diffusion models, including latent diffusion models (e.g., Stable Diffusion) and pixel diffusion models (e.g., DeepFloyd-IF), showing robustness against downstream fine-tuning and various attacks at both the image and model levels, with minimal impact on the model's generative capability. The code is available at https://github.com/taco-group/SleeperMark.

摘要: 大规模文本到图像(T2I)扩散模型的最新进展使各种下游应用成为可能，包括样式定制、主题驱动的个性化和条件生成。由于T2I模型需要大量的数据和计算资源来进行培训，因此对于其合法所有者来说，它们构成了非常有价值的知识产权(IP)，但也使它们成为寻求利用这些模型进行定制、通常是有利可图的应用的对手未经授权进行微调的激励目标。现有的扩散模型的知识产权保护方法一般都是先嵌入水印图案，然后通过生成输出检查或检查模型的特征空间来验证所有权。然而，当水印模型经过微调，并且在验证过程中(即，黑盒设置)无法访问特征空间时，这些技术在实际场景中本质上是无效的。该模型在适应新任务时，容易忘记先前学习到的水印知识。为了应对这一挑战，我们提出了SleeperMark，一个新的框架，旨在将弹性水印嵌入到T2I扩散模型中。SleeperMark明确地引导模型将水印信息从它学习的语义概念中分离出来，允许模型保留嵌入的水印，同时继续微调到新的下游任务。我们的大量实验证明了SleeperMark在各种类型的扩散模型上的有效性，包括潜在扩散模型(例如，稳定扩散)和像素扩散模型(例如，DeepFloyd-IF)，显示出对图像和模型级别的下游微调和各种攻击的鲁棒性，而对模型的生成能力的影响最小。代码可在https://github.com/taco-group/SleeperMark.上获得



## **47. Plentiful Jailbreaks with String Compositions**

弦乐作品丰富越狱 cs.CL

NeurIPS SoLaR Workshop 2024

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2411.01084v2) [paper-pdf](http://arxiv.org/pdf/2411.01084v2)

**Authors**: Brian R. Y. Huang

**Abstract**: Large language models (LLMs) remain vulnerable to a slew of adversarial attacks and jailbreaking methods. One common approach employed by white-hat attackers, or red-teamers, is to process model inputs and outputs using string-level obfuscations, which can include leetspeak, rotary ciphers, Base64, ASCII, and more. Our work extends these encoding-based attacks by unifying them in a framework of invertible string transformations. With invertibility, we can devise arbitrary string compositions, defined as sequences of transformations, that we can encode and decode end-to-end programmatically. We devise a automated best-of-n attack that samples from a combinatorially large number of string compositions. Our jailbreaks obtain competitive attack success rates on several leading frontier models when evaluated on HarmBench, highlighting that encoding-based attacks remain a persistent vulnerability even in advanced LLMs.

摘要: 大型语言模型（LLM）仍然容易受到一系列对抗攻击和越狱方法的影响。白帽攻击者或红团队使用的一种常见方法是使用字符串级混淆处理模型输入和输出，其中可以包括leetspeak、旋转密码、Base 64、ASC等。我们的工作通过将这些基于编码的攻击统一到可逆字符串转换的框架中来扩展它们。通过可逆性，我们可以设计任意的字符串组合，定义为转换序列，我们可以通过编程方式进行端到端编码和解码。我们设计了一种自动化的n中最佳攻击，该攻击从组合上大量的字符串组合中进行采样。在HarmBench上进行评估时，我们的越狱在几个领先的前沿模型上获得了有竞争力的攻击成功率，这凸显了即使在高级LLM中，基于编码的攻击仍然是一个持久的漏洞。



## **48. PADetBench: Towards Benchmarking Physical Attacks against Object Detection**

PADetBench：针对对象检测的物理攻击基准 cs.CV

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2408.09181v2) [paper-pdf](http://arxiv.org/pdf/2408.09181v2)

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Lap-Pui Chau, Shaohui Mei

**Abstract**: Physical attacks against object detection have gained increasing attention due to their significant practical implications. However, conducting physical experiments is extremely time-consuming and labor-intensive. Moreover, physical dynamics and cross-domain transformation are challenging to strictly regulate in the real world, leading to unaligned evaluation and comparison, severely hindering the development of physically robust models. To accommodate these challenges, we explore utilizing realistic simulation to thoroughly and rigorously benchmark physical attacks with fairness under controlled physical dynamics and cross-domain transformation. This resolves the problem of capturing identical adversarial images that cannot be achieved in the real world. Our benchmark includes 20 physical attack methods, 48 object detectors, comprehensive physical dynamics, and evaluation metrics. We also provide end-to-end pipelines for dataset generation, detection, evaluation, and further analysis. In addition, we perform 8064 groups of evaluation based on our benchmark, which includes both overall evaluation and further detailed ablation studies for controlled physical dynamics. Through these experiments, we provide in-depth analyses of physical attack performance and physical adversarial robustness, draw valuable observations, and discuss potential directions for future research.   Codebase: https://github.com/JiaweiLian/Benchmarking_Physical_Attack

摘要: 针对目标检测的物理攻击由于其重要的实际意义而受到越来越多的关注。然而，进行物理实验是极其耗时和劳动密集型的。此外，物理动力学和跨域转换在现实世界中面临严格规范的挑战，导致不一致的评估和比较，严重阻碍了物理稳健模型的发展。为了应对这些挑战，我们探索利用真实的模拟在受控的物理动力学和跨域转换下，彻底和严格地基准具有公平性的物理攻击。这解决了在现实世界中无法实现的捕获相同的对抗性图像的问题。我们的基准包括20种物理攻击方法、48个对象探测器、全面的物理动力学和评估指标。我们还提供用于数据集生成、检测、评估和进一步分析的端到端管道。此外，我们根据我们的基准进行了8064组评估，其中包括对受控物理动力学的整体评估和进一步的详细消融研究。通过这些实验，我们对物理攻击性能和物理对抗健壮性进行了深入的分析，得出了有价值的观察结果，并讨论了未来研究的潜在方向。码基：https://github.com/JiaweiLian/Benchmarking_Physical_Attack



## **49. Defending Object Detectors against Patch Attacks with Out-of-Distribution Smoothing**

利用非分布平滑保护对象检测器免受补丁攻击 cs.LG

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2205.08989v2) [paper-pdf](http://arxiv.org/pdf/2205.08989v2)

**Authors**: Ryan Feng, Neal Mangaokar, Jihye Choi, Somesh Jha, Atul Prakash

**Abstract**: Patch attacks against object detectors have been of recent interest due to their being physically realizable and more closely aligned with practical systems. In response to this threat, many new defenses have been proposed that train a patch segmenter model to detect and remove the patch before the image is passed to the downstream model. We unify these approaches with a flexible framework, OODSmoother, which characterizes the properties of approaches that aim to remove adversarial patches. This framework naturally guides us to design 1) a novel adaptive attack that breaks existing patch attack defenses on object detectors, and 2) a novel defense approach SemPrior that takes advantage of semantic priors. Our key insight behind SemPrior is that the existing machine learning-based patch detectors struggle to learn semantic priors and that explicitly incorporating them can improve performance. We find that SemPrior alone provides up to a 40% gain, or up to a 60% gain when combined with existing defenses.

摘要: 针对对象检测器的补丁攻击最近引起了人们的兴趣，因为它们是物理上可实现的，并且与实际系统更紧密地结合在一起。为了应对这种威胁，人们提出了许多新的防御措施，训练一个补丁分割模型，在图像被传递到下游模型之前检测并移除补丁。我们将这些方法与一个灵活的框架OODSmother统一起来，该框架表征了旨在移除敌意补丁的方法的特性。该框架自然而然地指导我们设计了一种新的自适应攻击，它打破了现有的对对象检测器的补丁攻击防御，以及一种新的利用语义先验的防御方法SemPrior。我们在SemPrior背后的关键见解是，现有的基于机器学习的补丁检测器很难学习语义先验，显式整合它们可以提高性能。我们发现，仅SemPrior一项就可以提供高达40%的收益，或者当与现有防御相结合时，可以提供高达60%的收益。



## **50. Targeting the Core: A Simple and Effective Method to Attack RAG-based Agents via Direct LLM Manipulation**

瞄准核心：通过直接LLM操纵攻击基于RAG的代理的简单有效方法 cs.AI

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2412.04415v1) [paper-pdf](http://arxiv.org/pdf/2412.04415v1)

**Authors**: Xuying Li, Zhuo Li, Yuji Kosuga, Yasuhiro Yoshida, Victor Bian

**Abstract**: AI agents, powered by large language models (LLMs), have transformed human-computer interactions by enabling seamless, natural, and context-aware communication. While these advancements offer immense utility, they also inherit and amplify inherent safety risks such as bias, fairness, hallucinations, privacy breaches, and a lack of transparency. This paper investigates a critical vulnerability: adversarial attacks targeting the LLM core within AI agents. Specifically, we test the hypothesis that a deceptively simple adversarial prefix, such as \textit{Ignore the document}, can compel LLMs to produce dangerous or unintended outputs by bypassing their contextual safeguards. Through experimentation, we demonstrate a high attack success rate (ASR), revealing the fragility of existing LLM defenses. These findings emphasize the urgent need for robust, multi-layered security measures tailored to mitigate vulnerabilities at the LLM level and within broader agent-based architectures.

摘要: 由大型语言模型（LLM）支持的人工智能代理通过实现无缝、自然和上下文感知的通信来改变了人机交互。虽然这些进步提供了巨大的实用性，但它们也继承和放大了固有的安全风险，例如偏见、公平、幻觉、隐私侵犯和缺乏透明度。本文研究了一个关键漏洞：针对人工智能代理内LLM核心的对抗攻击。具体来说，我们测试了这样的假设：看似简单的对抗性前置码（例如\textit{忽略文档}）可以迫使LLM绕过上下文保障措施来产生危险或非预期的输出。通过实验，我们展示了高攻击成功率（ASB），揭示了现有LLM防御的脆弱性。这些调查结果强调，迫切需要针对LLM级别和更广泛的基于代理的架构中的漏洞量身定制的强大、多层的安全措施。



