# Latest Adversarial Attack Papers
**update at 2021-11-24 23:56:46**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Adversarial machine learning for protecting against online manipulation**

用于防止在线操纵的对抗性机器学习 cs.LG

To appear on IEEE Internet Computing. `Accepted manuscript' version

**SubmitDate**: 2021-11-23    [paper-pdf](http://arxiv.org/pdf/2111.12034v1)

**Authors**: Stefano Cresci, Marinella Petrocchi, Angelo Spognardi, Stefano Tognazzi

**Abstracts**: Adversarial examples are inputs to a machine learning system that result in an incorrect output from that system. Attacks launched through this type of input can cause severe consequences: for example, in the field of image recognition, a stop signal can be misclassified as a speed limit indication.However, adversarial examples also represent the fuel for a flurry of research directions in different domains and applications. Here, we give an overview of how they can be profitably exploited as powerful tools to build stronger learning models, capable of better-withstanding attacks, for two crucial tasks: fake news and social bot detection.

摘要: 对抗性示例是机器学习系统的输入，导致该系统的输出不正确。通过这种输入发起的攻击可能会造成严重的后果：例如，在图像识别领域，停车信号可能被错误地归类为限速指示，但敌意的例子也代表了不同领域和应用中一系列研究方向的燃料。在这里，我们概述了如何将它们作为强大的工具有利可图地利用来构建更强大的学习模型，能够更好地抵御攻击，用于两个关键任务：假新闻和社交机器人检测。



## **2. An Improved Genetic Algorithm and Its Application in Neural Network Adversarial Attack**

一种改进的遗传算法及其在神经网络攻击中的应用 cs.NE

14 pages, 7 figures, 4 tables and 20 References

**SubmitDate**: 2021-11-23    [paper-pdf](http://arxiv.org/pdf/2110.01818v3)

**Authors**: Dingming Yang, Zeyu Yu, Hongqiang Yuan, Yanrong Cui

**Abstracts**: The choice of crossover and mutation strategies plays a crucial role in the search ability, convergence efficiency and precision of genetic algorithms. In this paper, a new improved genetic algorithm is proposed by improving the crossover and mutation operation of the simple genetic algorithm, and it is verified by four test functions. Simulation results show that, comparing with three other mainstream swarm intelligence optimization algorithms, the algorithm can not only improve the global search ability, convergence efficiency and precision, but also increase the success rate of convergence to the optimal value under the same experimental conditions. Finally, the algorithm is applied to neural networks adversarial attacks. The applied results show that the method does not need the structure and parameter information inside the neural network model, and it can obtain the adversarial samples with high confidence in a brief time just by the classification and confidence information output from the neural network.

摘要: 交叉和变异策略的选择对遗传算法的搜索能力、收敛效率和精度起着至关重要的作用。通过对简单遗传算法交叉和变异操作的改进，提出了一种新的改进遗传算法，并通过四个测试函数进行了验证。仿真结果表明，与其他三种主流群体智能优化算法相比，该算法在相同的实验条件下，不仅提高了全局搜索能力、收敛效率和精度，而且提高了收敛到最优值的成功率。最后，将该算法应用于神经网络的对抗性攻击。应用结果表明，该方法不需要神经网络模型内部的结构和参数信息，仅根据神经网络输出的分类和置信度信息，即可在短时间内获得高置信度的对抗性样本。



## **3. Relevance Attack on Detectors**

对检测器的相关性攻击 cs.CV

accepted by Pattern Recognition

**SubmitDate**: 2021-11-23    [paper-pdf](http://arxiv.org/pdf/2008.06822v4)

**Authors**: Sizhe Chen, Fan He, Xiaolin Huang, Kun Zhang

**Abstracts**: This paper focuses on high-transferable adversarial attacks on detectors, which are hard to attack in a black-box manner, because of their multiple-output characteristics and the diversity across architectures. To pursue a high attack transferability, one plausible way is to find a common property across detectors, which facilitates the discovery of common weaknesses. We are the first to suggest that the relevance map from interpreters for detectors is such a property. Based on it, we design a Relevance Attack on Detectors (RAD), which achieves a state-of-the-art transferability, exceeding existing results by above 20%. On MS COCO, the detection mAPs for all 8 black-box architectures are more than halved and the segmentation mAPs are also significantly influenced. Given the great transferability of RAD, we generate the first adversarial dataset for object detection and instance segmentation, i.e., Adversarial Objects in COntext (AOCO), which helps to quickly evaluate and improve the robustness of detectors.

摘要: 针对检测器的高可移植性对抗性攻击，由于其多输出特性和跨体系结构的多样性，很难以黑盒方式进行攻击。为了追求较高的攻击可转移性，一种可行的方法是在多个检测器之间找到共同的属性，这有助于发现共同的弱点。我们是第一个提出从解释器到检测器的相关性映射就是这样一个属性的人。在此基础上，设计了一种基于检测器的关联攻击(RAD)，实现了最新的可移植性，比已有结果提高了20%以上。在MS Coco上，所有8个黑盒架构的检测图都减少了一半以上，分割图也受到了显著影响。考虑到RAD具有很强的可移植性，我们生成了第一个用于对象检测和实例分割的对抗性数据集，即上下文中的对抗性对象(AOCO)，这有助于快速评估和提高检测器的鲁棒性。



## **4. A Comparison of State-of-the-Art Techniques for Generating Adversarial Malware Binaries**

生成敌意恶意软件二进制文件的最新技术比较 cs.CR

18 pages, 7 figures; summer project report from NREIP internship at  Naval Research Laboratory

**SubmitDate**: 2021-11-22    [paper-pdf](http://arxiv.org/pdf/2111.11487v1)

**Authors**: Prithviraj Dasgupta, Zachariah Osman

**Abstracts**: We consider the problem of generating adversarial malware by a cyber-attacker where the attacker's task is to strategically modify certain bytes within existing binary malware files, so that the modified files are able to evade a malware detector such as machine learning-based malware classifier. We have evaluated three recent adversarial malware generation techniques using binary malware samples drawn from a single, publicly available malware data set and compared their performances for evading a machine-learning based malware classifier called MalConv. Our results show that among the compared techniques, the most effective technique is the one that strategically modifies bytes in a binary's header. We conclude by discussing the lessons learned and future research directions on the topic of adversarial malware generation.

摘要: 我们考虑了网络攻击者生成恶意软件的问题，其中攻击者的任务是策略性地修改现有二进制恶意软件文件中的某些字节，以便修改后的文件能够躲避恶意软件检测器(如基于机器学习的恶意软件分类器)。我们使用来自单个公开可用的恶意软件数据集的二进制恶意软件样本评估了最近的三种敌意恶意软件生成技术，并比较了它们在逃避基于机器学习的恶意软件分类器MalConv方面的性能。我们的结果表明，在比较的技术中，最有效的技术是策略性地修改二进制头中的字节。最后，我们讨论了在恶意软件生成这一主题上的经验教训和未来的研究方向。



## **5. Adversarial Examples on Segmentation Models Can be Easy to Transfer**

细分模型上的对抗性示例可以很容易地转移 cs.CV

**SubmitDate**: 2021-11-22    [paper-pdf](http://arxiv.org/pdf/2111.11368v1)

**Authors**: Jindong Gu, Hengshuang Zhao, Volker Tresp, Philip Torr

**Abstracts**: Deep neural network-based image classification can be misled by adversarial examples with small and quasi-imperceptible perturbations. Furthermore, the adversarial examples created on one classification model can also fool another different model. The transferability of the adversarial examples has recently attracted a growing interest since it makes black-box attacks on classification models feasible. As an extension of classification, semantic segmentation has also received much attention towards its adversarial robustness. However, the transferability of adversarial examples on segmentation models has not been systematically studied. In this work, we intensively study this topic. First, we explore the overfitting phenomenon of adversarial examples on classification and segmentation models. In contrast to the observation made on classification models that the transferability is limited by overfitting to the source model, we find that the adversarial examples on segmentations do not always overfit the source models. Even when no overfitting is presented, the transferability of adversarial examples is limited. We attribute the limitation to the architectural traits of segmentation models, i.e., multi-scale object recognition. Then, we propose a simple and effective method, dubbed dynamic scaling, to overcome the limitation. The high transferability achieved by our method shows that, in contrast to the observations in previous work, adversarial examples on a segmentation model can be easy to transfer to other segmentation models. Our analysis and proposals are supported by extensive experiments.

摘要: 基于深度神经网络的图像分类容易受到具有小扰动和准不可察觉扰动的对抗性样本的误导。此外，在一个分类模型上创建的对抗性示例也可以欺骗另一个不同的模型。对抗性例子的可转移性最近引起了人们越来越大的兴趣，因为它使得对分类模型的黑盒攻击成为可能。语义分割作为分类的一种扩展，也因其对抗性的鲁棒性而备受关注。然而，对抗性例子在分词模型上的可转移性还没有得到系统的研究。在这项工作中，我们对这一主题进行了深入的研究。首先，我们探讨了对抗性例子在分类和分割模型上的过度拟合现象。与在分类模型上观察到的对源模型的过度拟合限制了可转移性的观察相比，我们发现关于分割的对抗性例子并不总是对源模型过度拟合。即使没有出现过拟合，对抗性例子的可转换性也是有限的。我们将其局限性归因于分割模型的结构特性，即多尺度目标识别。然后，我们提出了一种简单而有效的方法，称为动态缩放，以克服这一局限性。我们的方法达到了很高的可移植性，这表明与以前的工作相比，分割模型上的对抗性例子可以很容易地转移到其他分割模型上。我们的分析和建议得到了大量实验的支持。



## **6. Shift Invariance Can Reduce Adversarial Robustness**

移位不变性会降低对手的健壮性 cs.LG

Published as a conference paper at NeurIPS 2021

**SubmitDate**: 2021-11-22    [paper-pdf](http://arxiv.org/pdf/2103.02695v3)

**Authors**: Songwei Ge, Vasu Singla, Ronen Basri, David Jacobs

**Abstracts**: Shift invariance is a critical property of CNNs that improves performance on classification. However, we show that invariance to circular shifts can also lead to greater sensitivity to adversarial attacks. We first characterize the margin between classes when a shift-invariant linear classifier is used. We show that the margin can only depend on the DC component of the signals. Then, using results about infinitely wide networks, we show that in some simple cases, fully connected and shift-invariant neural networks produce linear decision boundaries. Using this, we prove that shift invariance in neural networks produces adversarial examples for the simple case of two classes, each consisting of a single image with a black or white dot on a gray background. This is more than a curiosity; we show empirically that with real datasets and realistic architectures, shift invariance reduces adversarial robustness. Finally, we describe initial experiments using synthetic data to probe the source of this connection.

摘要: 移位不变性是CNN提高分类性能的一个重要性质。然而，我们发现对循环移位的不变性也可以导致对敌意攻击更敏感。当使用平移不变的线性分类器时，我们首先表征类之间的边缘。我们表明，裕度只能取决于信号的直流分量。然后，利用关于无限宽网络的结果，我们证明了在一些简单的情况下，完全连通和平移不变的神经网络产生线性决策边界。利用这一点，我们证明了神经网络中的平移不变性对于两类简单的情况产生了对抗性的例子，每一类都由单个图像组成，在灰色背景上有一个黑点或白点。这不仅仅是一种好奇；我们的经验表明，对于真实的数据集和现实的体系结构，移位不变性会降低对手的健壮性。最后，我们描述了使用合成数据来探索这种联系的来源的初步实验。



## **7. NTD: Non-Transferability Enabled Backdoor Detection**

NTD：启用不可转移的后门检测 cs.CR

**SubmitDate**: 2021-11-22    [paper-pdf](http://arxiv.org/pdf/2111.11157v1)

**Authors**: Yinshan Li, Hua Ma, Zhi Zhang, Yansong Gao, Alsharif Abuadbba, Anmin Fu, Yifeng Zheng, Said F. Al-Sarawi, Derek Abbott

**Abstracts**: A backdoor deep learning (DL) model behaves normally upon clean inputs but misbehaves upon trigger inputs as the backdoor attacker desires, posing severe consequences to DL model deployments. State-of-the-art defenses are either limited to specific backdoor attacks (source-agnostic attacks) or non-user-friendly in that machine learning (ML) expertise or expensive computing resources are required. This work observes that all existing backdoor attacks have an inevitable intrinsic weakness, non-transferability, that is, a trigger input hijacks a backdoored model but cannot be effective to another model that has not been implanted with the same backdoor. With this key observation, we propose non-transferability enabled backdoor detection (NTD) to identify trigger inputs for a model-under-test (MUT) during run-time.Specifically, NTD allows a potentially backdoored MUT to predict a class for an input. In the meantime, NTD leverages a feature extractor (FE) to extract feature vectors for the input and a group of samples randomly picked from its predicted class, and then compares similarity between the input and the samples in the FE's latent space. If the similarity is low, the input is an adversarial trigger input; otherwise, benign. The FE is a free pre-trained model privately reserved from open platforms. As the FE and MUT are from different sources, the attacker is very unlikely to insert the same backdoor into both of them. Because of non-transferability, a trigger effect that does work on the MUT cannot be transferred to the FE, making NTD effective against different types of backdoor attacks. We evaluate NTD on three popular customized tasks such as face recognition, traffic sign recognition and general animal classification, results of which affirm that NDT has high effectiveness (low false acceptance rate) and usability (low false rejection rate) with low detection latency.

摘要: 后门深度学习(DL)模型在干净的输入上行为正常，但在触发器输入上行为不当，正如后门攻击者所希望的那样，这会给DL模型的部署带来严重后果。最先进的防御要么局限于特定的后门攻击(与来源无关的攻击)，要么不利于用户，因为需要机器学习(ML)专业知识或昂贵的计算资源。这项工作观察到，所有现有的后门攻击都有一个不可避免的内在弱点，即不可转移性，即一个触发器输入劫持了一个后门模型，但不能对另一个没有植入相同后门的模型有效。基于这一关键观察，我们提出了不可转移性启用后门检测(NTD)来识别运行时被测模型(MUT)的触发输入，具体地说，NTD允许潜在的后门MUT预测输入的类。同时，NTD利用特征提取器(FE)来提取输入的特征向量和从其预测类中随机选取的一组样本，然后在FE的潜在空间中比较输入和样本之间的相似度。如果相似度较低，则输入为对抗性触发器输入；否则，为良性输入。FE是一个免费的预先培训的模型，私下保留在开放平台上。由于FE和MUT来自不同的来源，攻击者不太可能将相同的后门插入到两者中。由于不可转移性，在MUT上起作用的触发效果不能转移到FE上，从而使NTD能够有效地对抗不同类型的后门攻击。我们在人脸识别、交通标志识别和一般动物分类这三个流行的定制任务上对NTD进行了评估，结果证实了NDT具有高效率(低错误接受率)和易用性(低错误拒绝率)和低检测延迟的特点。



## **8. Efficient Combinatorial Optimization for Word-level Adversarial Textual Attack**

词级对抗性文本攻击的高效组合优化 cs.CL

**SubmitDate**: 2021-11-22    [paper-pdf](http://arxiv.org/pdf/2109.02229v3)

**Authors**: Shengcai Liu, Ning Lu, Cheng Chen, Ke Tang

**Abstracts**: Over the past few years, various word-level textual attack approaches have been proposed to reveal the vulnerability of deep neural networks used in natural language processing. Typically, these approaches involve an important optimization step to determine which substitute to be used for each word in the original input. However, current research on this step is still rather limited, from the perspectives of both problem-understanding and problem-solving. In this paper, we address these issues by uncovering the theoretical properties of the problem and proposing an efficient local search algorithm (LS) to solve it. We establish the first provable approximation guarantee on solving the problem in general cases.Extensive experiments involving 5 NLP tasks, 8 datasets and 26 NLP models show that LS can largely reduce the number of queries usually by an order of magnitude to achieve high attack success rates. Further experiments show that the adversarial examples crafted by LS usually have higher quality, exhibit better transferability, and can bring more robustness improvement to victim models by adversarial training.

摘要: 在过去的几年里，各种词级文本攻击方法被提出，以揭示深度神经网络在自然语言处理中的脆弱性。通常，这些方法涉及一个重要的优化步骤，以确定对原始输入中的每个单词使用哪个替身。然而，目前对这一步骤的研究还相当有限，无论是从问题理解的角度还是从问题解决的角度。在本文中，我们通过揭示问题的理论性质并提出一种有效的局部搜索算法(LS)来解决这些问题。通过对5个NLP任务、8个数据集和26个NLP模型的大量实验表明，LS算法可以极大地减少查询次数，通常可以减少一个数量级的查询次数，从而获得较高的攻击成功率。进一步的实验表明，LS生成的对抗性实例质量较高，具有较好的可移植性，通过对抗性训练可以给受害者模型带来更多的健壮性提升。



## **9. Myope Models -- Are face presentation attack detection models short-sighted?**

近视模型--面部呈现攻击检测模型是近视吗？ cs.CV

Accepted at the 2ND WORKSHOP ON EXPLAINABLE & INTERPRETABLE  ARTIFICIAL INTELLIGENCE FOR BIOMETRICS AT WACV 2022

**SubmitDate**: 2021-11-22    [paper-pdf](http://arxiv.org/pdf/2111.11127v1)

**Authors**: Pedro C. Neto, Ana F. Sequeira, Jaime S. Cardoso

**Abstracts**: Presentation attacks are recurrent threats to biometric systems, where impostors attempt to bypass these systems. Humans often use background information as contextual cues for their visual system. Yet, regarding face-based systems, the background is often discarded, since face presentation attack detection (PAD) models are mostly trained with face crops. This work presents a comparative study of face PAD models (including multi-task learning, adversarial training and dynamic frame selection) in two settings: with and without crops. The results show that the performance is consistently better when the background is present in the images. The proposed multi-task methodology beats the state-of-the-art results on the ROSE-Youtu dataset by a large margin with an equal error rate of 0.2%. Furthermore, we analyze the models' predictions with Grad-CAM++ with the aim to investigate to what extent the models focus on background elements that are known to be useful for human inspection. From this analysis we can conclude that the background cues are not relevant across all the attacks. Thus, showing the capability of the model to leverage the background information only when necessary.

摘要: 演示攻击是对生物识别系统的反复威胁，冒名顶替者试图绕过这些系统。人类经常使用背景信息作为视觉系统的上下文线索。然而，对于基于人脸的系统，背景通常被丢弃，因为人脸呈现攻击检测(PAD)模型大多是用人脸作物来训练的。这项工作提出了两种情况下的脸垫模型(包括多任务学习、对抗性训练和动态帧选择)的比较研究：有作物和没有作物两种情况下的人脸模型(包括多任务学习、对抗性训练和动态帧选择)。结果表明，当图像中存在背景时，性能始终较好。所提出的多任务方法在ROSE-YOTU数据集上以0.2%的同等错误率大大超过了最新的结果。此外，我们使用Grad-CAM++分析了模型的预测，目的是调查模型在多大程度上关注已知对人类检查有用的背景元素。根据这一分析，我们可以得出结论，背景线索在所有攻击中并不相关。因此，显示了模型仅在必要时利用背景信息的能力。



## **10. Meta-Learning the Search Distribution of Black-Box Random Search Based Adversarial Attacks**

基于黑盒随机搜索的对抗性攻击搜索分布元学习 cs.LG

accepted at NeurIPS 2021; updated the numbers in Table 5 and added  references; added acknowledgements

**SubmitDate**: 2021-11-22    [paper-pdf](http://arxiv.org/pdf/2111.01714v3)

**Authors**: Maksym Yatsura, Jan Hendrik Metzen, Matthias Hein

**Abstracts**: Adversarial attacks based on randomized search schemes have obtained state-of-the-art results in black-box robustness evaluation recently. However, as we demonstrate in this work, their efficiency in different query budget regimes depends on manual design and heuristic tuning of the underlying proposal distributions. We study how this issue can be addressed by adapting the proposal distribution online based on the information obtained during the attack. We consider Square Attack, which is a state-of-the-art score-based black-box attack, and demonstrate how its performance can be improved by a learned controller that adjusts the parameters of the proposal distribution online during the attack. We train the controller using gradient-based end-to-end training on a CIFAR10 model with white box access. We demonstrate that plugging the learned controller into the attack consistently improves its black-box robustness estimate in different query regimes by up to 20% for a wide range of different models with black-box access. We further show that the learned adaptation principle transfers well to the other data distributions such as CIFAR100 or ImageNet and to the targeted attack setting.

摘要: 近年来，基于随机搜索方案的对抗性攻击在黑盒健壮性评估方面取得了最新的研究成果。然而，正如我们在这项工作中演示的那样，它们在不同查询预算机制中的效率取决于对底层提案分布的手动设计和启发式调优。我们研究了如何根据攻击期间获得的信息在线调整建议分发来解决这个问题。我们考虑Square攻击，这是一种最先进的基于分数的黑盒攻击，并展示了如何通过学习控制器在攻击期间在线调整建议分布的参数来提高其性能。我们在带有白盒访问的CIFAR10模型上使用基于梯度的端到端训练来训练控制器。我们证明，对于具有黑盒访问的大范围不同模型，在不同的查询机制下，将学习控制器插入攻击可持续提高其黑盒健壮性估计高达20%。我们进一步表明，学习的适应原则很好地移植到其他数据分布，如CIFAR100或ImageNet，以及目标攻击设置。



## **11. Evaluating Adversarial Attacks on ImageNet: A Reality Check on Misclassification Classes**

评估ImageNet上的敌意攻击：对误分类类的现实检验 cs.CV

Accepted for publication in 35th Conference on Neural Information  Processing Systems (NeurIPS 2021), Workshop on ImageNet: Past,Present, and  Future

**SubmitDate**: 2021-11-22    [paper-pdf](http://arxiv.org/pdf/2111.11056v1)

**Authors**: Utku Ozbulak, Maura Pintor, Arnout Van Messem, Wesley De Neve

**Abstracts**: Although ImageNet was initially proposed as a dataset for performance benchmarking in the domain of computer vision, it also enabled a variety of other research efforts. Adversarial machine learning is one such research effort, employing deceptive inputs to fool models in making wrong predictions. To evaluate attacks and defenses in the field of adversarial machine learning, ImageNet remains one of the most frequently used datasets. However, a topic that is yet to be investigated is the nature of the classes into which adversarial examples are misclassified. In this paper, we perform a detailed analysis of these misclassification classes, leveraging the ImageNet class hierarchy and measuring the relative positions of the aforementioned type of classes in the unperturbed origins of the adversarial examples. We find that $71\%$ of the adversarial examples that achieve model-to-model adversarial transferability are misclassified into one of the top-5 classes predicted for the underlying source images. We also find that a large subset of untargeted misclassifications are, in fact, misclassifications into semantically similar classes. Based on these findings, we discuss the need to take into account the ImageNet class hierarchy when evaluating untargeted adversarial successes. Furthermore, we advocate for future research efforts to incorporate categorical information.

摘要: 虽然ImageNet最初是作为计算机视觉领域中性能基准测试的数据集提出的，但它也支持了各种其他研究工作。对抗性机器学习就是这样一种研究成果，它使用欺骗性的输入来愚弄模型做出错误的预测。为了评估对抗性机器学习领域中的攻击和防御，ImageNet仍然是最常用的数据集之一。然而，一个尚未调查的话题是对抗性例子被错误分类的类别的性质。在本文中，我们利用ImageNet的类层次结构，并测量上述类型的类在对抗性示例的未受干扰的来源中的相对位置，对这些误分类类进行了详细的分析。我们发现，实现模型到模型的对抗性转移的对抗性例子中，有$71\$被错误地归入了对底层源图像预测的前5类之一。我们还发现，非目标错误分类的很大子集实际上是误分类到语义相似的类中。基于这些发现，我们讨论了在评估非定向对抗性成功时是否需要考虑ImageNet类层次结构。此外，我们主张未来的研究努力纳入分类信息。



## **12. Selection of Source Images Heavily Influences the Effectiveness of Adversarial Attacks**

源图像的选择在很大程度上影响着对抗性攻击的效果 cs.CV

Accepted for publication in the 32nd British Machine Vision  Conference (BMVC)

**SubmitDate**: 2021-11-22    [paper-pdf](http://arxiv.org/pdf/2106.07141v3)

**Authors**: Utku Ozbulak, Esla Timothy Anzaku, Wesley De Neve, Arnout Van Messem

**Abstracts**: Although the adoption rate of deep neural networks (DNNs) has tremendously increased in recent years, a solution for their vulnerability against adversarial examples has not yet been found. As a result, substantial research efforts are dedicated to fix this weakness, with many studies typically using a subset of source images to generate adversarial examples, treating every image in this subset as equal. We demonstrate that, in fact, not every source image is equally suited for this kind of assessment. To do so, we devise a large-scale model-to-model transferability scenario for which we meticulously analyze the properties of adversarial examples, generated from every suitable source image in ImageNet by making use of three of the most frequently deployed attacks. In this transferability scenario, which involves seven distinct DNN models, including the recently proposed vision transformers, we reveal that it is possible to have a difference of up to $12.5\%$ in model-to-model transferability success, $1.01$ in average $L_2$ perturbation, and $0.03$ ($8/225$) in average $L_{\infty}$ perturbation when $1,000$ source images are sampled randomly among all suitable candidates. We then take one of the first steps in evaluating the robustness of images used to create adversarial examples, proposing a number of simple but effective methods to identify unsuitable source images, thus making it possible to mitigate extreme cases in experimentation and support high-quality benchmarking.

摘要: 尽管深度神经网络(DNNs)的采用率近年来有了很大的提高，但对于它们对敌意例子的脆弱性还没有找到解决方案。因此，大量的研究工作致力于修复这一弱点，许多研究通常使用源图像的子集来生成对抗性示例，将该子集中的每一幅图像视为平等。我们证明，事实上，并不是每一幅源图像都同样适合这种评估。为此，我们设计了一个大规模的模型到模型可转移性场景，在该场景中，我们通过使用三种最频繁部署的攻击，仔细分析了从ImageNet中每个合适的源映像生成的敌意示例的属性。在这个包含7个不同的DNN模型(包括最近提出的视觉转换器)的可转移性场景中，我们发现当在所有合适的候选者中随机抽样$1,000$源图像时，模型到模型的可转移性成功率可能存在高达$12.5\$的差异，平均$L2$扰动可能存在$1.01$的差异，平均$L3$($8/225$)的扰动可能存在。然后，我们在评估用于创建对抗性示例的图像的稳健性方面迈出了第一步，提出了一些简单但有效的方法来识别不合适的源图像，从而使得有可能缓解实验中的极端情况，并支持高质量的基准测试。



## **13. Imperceptible Transfer Attack and Defense on 3D Point Cloud Classification**

三维点云分类的潜移式攻防 cs.CV

**SubmitDate**: 2021-11-22    [paper-pdf](http://arxiv.org/pdf/2111.10990v1)

**Authors**: Daizong Liu, Wei Hu

**Abstracts**: Although many efforts have been made into attack and defense on the 2D image domain in recent years, few methods explore the vulnerability of 3D models. Existing 3D attackers generally perform point-wise perturbation over point clouds, resulting in deformed structures or outliers, which is easily perceivable by humans. Moreover, their adversarial examples are generated under the white-box setting, which frequently suffers from low success rates when transferred to attack remote black-box models. In this paper, we study 3D point cloud attacks from two new and challenging perspectives by proposing a novel Imperceptible Transfer Attack (ITA): 1) Imperceptibility: we constrain the perturbation direction of each point along its normal vector of the neighborhood surface, leading to generated examples with similar geometric properties and thus enhancing the imperceptibility. 2) Transferability: we develop an adversarial transformation model to generate the most harmful distortions and enforce the adversarial examples to resist it, improving their transferability to unknown black-box models. Further, we propose to train more robust black-box 3D models to defend against such ITA attacks by learning more discriminative point cloud representations. Extensive evaluations demonstrate that our ITA attack is more imperceptible and transferable than state-of-the-arts and validate the superiority of our defense strategy.

摘要: 虽然近年来人们在二维图像领域的攻防方面做了很多努力，但很少有方法研究三维模型的脆弱性。现有的3D攻击者一般对点云进行逐点摄动，产生变形的结构或离群点，这很容易被人察觉到。此外，它们的对抗性例子是在白盒环境下产生的，当转移到攻击远程黑盒模型时，白盒模型的成功率往往很低。本文从两个新的具有挑战性的角度对三维点云攻击进行了研究，提出了一种新的不可感知性转移攻击(ITA)：1)不可感知性：我们约束每个点沿其邻域曲面的法向量的扰动方向，从而生成具有相似几何性质的示例，从而增强了不可感知性。2)可转换性：我们建立了一个对抗性转换模型来产生最有害的扭曲，并加强了对抗性例子来抵抗它，提高了它们到未知黑盒模型的可转移性。此外，我们建议通过学习更具区别性的点云表示来训练更健壮的黑盒3D模型来防御此类ITA攻击。广泛的评估表明，我们的ITA攻击比最先进的攻击更具隐蔽性和可移动性，验证了我们防御战略的优越性。



## **14. Medical Aegis: Robust adversarial protectors for medical images**

医学宙斯盾：强大的医学图像对抗性保护者 cs.CV

**SubmitDate**: 2021-11-22    [paper-pdf](http://arxiv.org/pdf/2111.10969v1)

**Authors**: Qingsong Yao, Zecheng He, S. Kevin Zhou

**Abstracts**: Deep neural network based medical image systems are vulnerable to adversarial examples. Many defense mechanisms have been proposed in the literature, however, the existing defenses assume a passive attacker who knows little about the defense system and does not change the attack strategy according to the defense. Recent works have shown that a strong adaptive attack, where an attacker is assumed to have full knowledge about the defense system, can easily bypass the existing defenses. In this paper, we propose a novel adversarial example defense system called Medical Aegis. To the best of our knowledge, Medical Aegis is the first defense in the literature that successfully addresses the strong adaptive adversarial example attacks to medical images. Medical Aegis boasts two-tier protectors: The first tier of Cushion weakens the adversarial manipulation capability of an attack by removing its high-frequency components, yet posing a minimal effect on classification performance of the original image; the second tier of Shield learns a set of per-class DNN models to predict the logits of the protected model. Deviation from the Shield's prediction indicates adversarial examples. Shield is inspired by the observations in our stress tests that there exist robust trails in the shallow layers of a DNN model, which the adaptive attacks can hardly destruct. Experimental results show that the proposed defense accurately detects adaptive attacks, with negligible overhead for model inference.

摘要: 基于深度神经网络的医学图像系统容易受到敌意例子的攻击。文献中提出了很多防御机制，但现有的防御机制都假设攻击者是被动的，对防御系统知之甚少，不会根据防御情况改变攻击策略。最近的研究表明，强自适应攻击(假设攻击者完全了解防御系统)可以很容易地绕过现有的防御系统。在本文中，我们提出了一种新的对抗性实例防御系统--医学宙斯盾(Medical Aegis)。据我们所知，医学宙斯盾是文献中第一个成功解决了对医学图像的强适应性敌意攻击的防御方案。医学宙斯盾拥有两层保护器：第一层缓冲通过移除攻击的高频分量来削弱攻击的敌意操纵能力，但对原始图像的分类性能影响最小；第二层Shield学习一组按类的DNN模型来预测受保护模型的逻辑。与盾牌的预测背道而驰表明了敌对的例子。Shield的灵感来自于我们在压力测试中观察到的DNN模型的浅层存在健壮的踪迹，而自适应攻击很难破坏这些踪迹。实验结果表明，该防御方法能够准确检测出自适应攻击，而模型推理开销可以忽略不计。



## **15. Local Model Poisoning Attacks to Byzantine-Robust Federated Learning**

对拜占庭鲁棒联合学习的局部模型中毒攻击 cs.CR

Appeared in Usenix Security Symposium 2020. Fixed an error in Theorem  1. For demo code, see https://people.duke.edu/~zg70/code/fltrust.zip . For  slides, see https://people.duke.edu/~zg70/code/Secure_Federated_Learning.pdf  . For the talk, see https://www.youtube.com/watch?v=LP4uqW18yA0

**SubmitDate**: 2021-11-21    [paper-pdf](http://arxiv.org/pdf/1911.11815v4)

**Authors**: Minghong Fang, Xiaoyu Cao, Jinyuan Jia, Neil Zhenqiang Gong

**Abstracts**: In federated learning, multiple client devices jointly learn a machine learning model: each client device maintains a local model for its local training dataset, while a master device maintains a global model via aggregating the local models from the client devices. The machine learning community recently proposed several federated learning methods that were claimed to be robust against Byzantine failures (e.g., system failures, adversarial manipulations) of certain client devices. In this work, we perform the first systematic study on local model poisoning attacks to federated learning. We assume an attacker has compromised some client devices, and the attacker manipulates the local model parameters on the compromised client devices during the learning process such that the global model has a large testing error rate. We formulate our attacks as optimization problems and apply our attacks to four recent Byzantine-robust federated learning methods. Our empirical results on four real-world datasets show that our attacks can substantially increase the error rates of the models learnt by the federated learning methods that were claimed to be robust against Byzantine failures of some client devices. We generalize two defenses for data poisoning attacks to defend against our local model poisoning attacks. Our evaluation results show that one defense can effectively defend against our attacks in some cases, but the defenses are not effective enough in other cases, highlighting the need for new defenses against our local model poisoning attacks to federated learning.

摘要: 在联合学习中，多个客户端设备共同学习机器学习模型：每个客户端设备维护其本地训练数据集的本地模型，而主设备通过聚集来自客户端设备的本地模型来维护全局模型。机器学习社区最近提出了几种联合学习方法，这些方法声称对某些客户端设备的拜占庭故障(例如，系统故障、敌意操纵)是健壮的。在这项工作中，我们首次系统地研究了针对联邦学习的局部模型中毒攻击。我们假设攻击者已经侵入了一些客户端设备，并且攻击者在学习过程中操纵了受损客户端设备上的本地模型参数，使得全局模型具有很大的测试错误率。我们将我们的攻击描述为优化问题，并将我们的攻击应用于最近的四种拜占庭鲁棒联邦学习方法。我们在四个真实数据集上的实验结果表明，我们的攻击可以显著提高联邦学习方法学习的模型的错误率，这些方法号称对一些客户端设备的拜占庭故障具有鲁棒性。我们总结了两种针对数据中毒攻击的防御措施，以防御我们的本地模型中毒攻击。我们的评估结果表明，在某些情况下，一种防御可以有效地防御我们的攻击，但在其他情况下，防御效果不够好，这突显了针对联邦学习的本地模型中毒攻击需要新的防御措施。



## **16. Denoised Internal Models: a Brain-Inspired Autoencoder against Adversarial Attacks**

去噪内部模型：一种抗敌意攻击的脑启发自动编码器 cs.CV

16 pages, 3 figures

**SubmitDate**: 2021-11-21    [paper-pdf](http://arxiv.org/pdf/2111.10844v1)

**Authors**: Kaiyuan Liu, Xingyu Li, Yi Zhou, Jisong Guan, Yurui Lai, Ge Zhang, Hang Su, Jiachen Wang, Chunxu Guo

**Abstracts**: Despite its great success, deep learning severely suffers from robustness; that is, deep neural networks are very vulnerable to adversarial attacks, even the simplest ones. Inspired by recent advances in brain science, we propose the Denoised Internal Models (DIM), a novel generative autoencoder-based model to tackle this challenge. Simulating the pipeline in the human brain for visual signal processing, DIM adopts a two-stage approach. In the first stage, DIM uses a denoiser to reduce the noise and the dimensions of inputs, reflecting the information pre-processing in the thalamus. Inspired from the sparse coding of memory-related traces in the primary visual cortex, the second stage produces a set of internal models, one for each category. We evaluate DIM over 42 adversarial attacks, showing that DIM effectively defenses against all the attacks and outperforms the SOTA on the overall robustness.

摘要: 尽管深度学习取得了巨大的成功，但它的健壮性严重不足；也就是说，深度神经网络非常容易受到对手的攻击，即使是最简单的攻击。受脑科学最新进展的启发，我们提出了去噪内部模型(DIM)，这是一种新颖的基于生成式自动编码器的模型，以应对撞击的这一挑战。模拟人脑中视觉信号处理的管道，DIM采用了两个阶段的方法。在第一阶段，DIM使用去噪器来降低输入的噪声和维数，反映了丘脑的信息预处理。第二阶段的灵感来自于初级视觉皮层中与记忆相关的痕迹的稀疏编码，第二阶段产生了一组内部模型，每个类别一个。我们对DIM42个对抗性攻击进行了评估，结果表明，DIM有效地防御了所有攻击，并且在整体鲁棒性上优于SOTA。



## **17. Modelling Direct Messaging Networks with Multiple Recipients for Cyber Deception**

具有多个接收者的网络欺骗直接消息传递网络建模 cs.CR

**SubmitDate**: 2021-11-21    [paper-pdf](http://arxiv.org/pdf/2111.11932v1)

**Authors**: Kristen Moore, Cody J. Christopher, David Liebowitz, Surya Nepal, Renee Selvey

**Abstracts**: Cyber deception is emerging as a promising approach to defending networks and systems against attackers and data thieves. However, despite being relatively cheap to deploy, the generation of realistic content at scale is very costly, due to the fact that rich, interactive deceptive technologies are largely hand-crafted. With recent improvements in Machine Learning, we now have the opportunity to bring scale and automation to the creation of realistic and enticing simulated content. In this work, we propose a framework to automate the generation of email and instant messaging-style group communications at scale. Such messaging platforms within organisations contain a lot of valuable information inside private communications and document attachments, making them an enticing target for an adversary. We address two key aspects of simulating this type of system: modelling when and with whom participants communicate, and generating topical, multi-party text to populate simulated conversation threads. We present the LogNormMix-Net Temporal Point Process as an approach to the first of these, building upon the intensity-free modeling approach of Shchur et al.~\cite{shchur2019intensity} to create a generative model for unicast and multi-cast communications. We demonstrate the use of fine-tuned, pre-trained language models to generate convincing multi-party conversation threads. A live email server is simulated by uniting our LogNormMix-Net TPP (to generate the communication timestamp, sender and recipients) with the language model, which generates the contents of the multi-party email threads. We evaluate the generated content with respect to a number of realism-based properties, that encourage a model to learn to generate content that will engage the attention of an adversary to achieve a deception outcome.

摘要: 网络欺骗正在成为保护网络和系统免受攻击者和数据窃贼攻击的一种很有前途的方法。然而，尽管部署成本相对较低，但由于丰富的交互式欺骗性技术主要是手工制作的，大规模生成逼真内容的成本非常高。随着机器学习的最新改进，我们现在有机会将规模化和自动化带到创建逼真和诱人的模拟内容的过程中。在这项工作中，我们提出了一个框架来自动生成大规模的电子邮件和即时消息样式的群组通信。组织内的此类消息传递平台在私人通信和文档附件中包含大量有价值的信息，使其成为诱人的对手攻击目标。我们解决了模拟这种类型的系统的两个关键方面：模拟参与者何时以及与谁通信，以及生成主题多方文本以填充模拟的对话线索。我们提出了LogNormMix-net时点过程作为第一种方法，它建立在Shchur等人的无强度建模方法的基础上，为单播和多播通信创建了一个生成性模型。我们演示了如何使用微调的、预先训练的语言模型来生成令人信服的多方对话线索。通过将LogNormMix-Net TPP(生成通信时间戳、发送者和接收者)与语言模型(生成多方电子邮件线程的内容)结合起来，模拟了一个实时电子邮件服务器。我们根据一些基于现实主义的属性来评估生成的内容，这些属性鼓励模型学习生成将吸引对手注意力的内容，以实现欺骗结果。



## **18. Inconspicuous Adversarial Patches for Fooling Image Recognition Systems on Mobile Devices**

用于欺骗移动设备上的图像识别系统的不起眼的对抗性补丁 cs.CV

accpeted by iotj. arXiv admin note: substantial text overlap with  arXiv:2009.09774

**SubmitDate**: 2021-11-21    [paper-pdf](http://arxiv.org/pdf/2106.15202v2)

**Authors**: Tao Bai, Jinqi Luo, Jun Zhao

**Abstracts**: Deep learning based image recognition systems have been widely deployed on mobile devices in today's world. In recent studies, however, deep learning models are shown vulnerable to adversarial examples. One variant of adversarial examples, called adversarial patch, draws researchers' attention due to its strong attack abilities. Though adversarial patches achieve high attack success rates, they are easily being detected because of the visual inconsistency between the patches and the original images. Besides, it usually requires a large amount of data for adversarial patch generation in the literature, which is computationally expensive and time-consuming. To tackle these challenges, we propose an approach to generate inconspicuous adversarial patches with one single image. In our approach, we first decide the patch locations basing on the perceptual sensitivity of victim models, then produce adversarial patches in a coarse-to-fine way by utilizing multiple-scale generators and discriminators. The patches are encouraged to be consistent with the background images with adversarial training while preserving strong attack abilities. Our approach shows the strong attack abilities in white-box settings and the excellent transferability in black-box settings through extensive experiments on various models with different architectures and training methods. Compared to other adversarial patches, our adversarial patches hold the most negligible risks to be detected and can evade human observations, which is supported by the illustrations of saliency maps and results of user evaluations. Lastly, we show that our adversarial patches can be applied in the physical world.

摘要: 基于深度学习的图像识别系统在当今世界的移动设备上得到了广泛的应用。然而，在最近的研究中，深度学习模型被证明容易受到对抗性例子的影响。对抗性例子的一种变体，称为对抗性补丁，由于其强大的攻击能力而引起了研究者的注意。尽管敌意补丁的攻击成功率很高，但由于补丁和原始图像之间的视觉不一致，它们很容易被检测出来。此外，文献中的对抗性补丁生成通常需要大量的数据，计算量大，耗时长。针对撞击面临的这些挑战，我们提出了一种利用一张图片生成不明显的敌意补丁的方法。在我们的方法中，我们首先根据受害者模型的感知敏感度来确定补丁的位置，然后利用多尺度生成器和鉴别器从粗到精的方式生成对抗性的补丁。通过对抗性训练，鼓励补丁与背景图像保持一致，同时保持较强的攻击能力。通过在不同架构和训练方法的不同模型上的大量实验，表明该方法在白盒环境下具有较强的攻击能力，在黑盒环境下具有良好的可移植性。与其他对抗性补丁相比，我们的对抗性补丁具有最容易被检测到的风险，并且可以躲避人类的观察，这一点从显著图和用户评估结果的插图中得到了支持。最后，我们证明了我们的对抗性补丁可以应用于物理世界。



## **19. Adversarial Mask: Real-World Adversarial Attack Against Face Recognition Models**

对抗性面具：针对人脸识别模型的真实对抗性攻击 cs.CV

**SubmitDate**: 2021-11-21    [paper-pdf](http://arxiv.org/pdf/2111.10759v1)

**Authors**: Alon Zolfi, Shai Avidan, Yuval Elovici, Asaf Shabtai

**Abstracts**: Deep learning-based facial recognition (FR) models have demonstrated state-of-the-art performance in the past few years, even when wearing protective medical face masks became commonplace during the COVID-19 pandemic. Given the outstanding performance of these models, the machine learning research community has shown increasing interest in challenging their robustness. Initially, researchers presented adversarial attacks in the digital domain, and later the attacks were transferred to the physical domain. However, in many cases, attacks in the physical domain are conspicuous, requiring, for example, the placement of a sticker on the face, and thus may raise suspicion in real-world environments (e.g., airports). In this paper, we propose Adversarial Mask, a physical adversarial universal perturbation (UAP) against state-of-the-art FR models that is applied on face masks in the form of a carefully crafted pattern. In our experiments, we examined the transferability of our adversarial mask to a wide range of FR model architectures and datasets. In addition, we validated our adversarial mask effectiveness in real-world experiments by printing the adversarial pattern on a fabric medical face mask, causing the FR system to identify only 3.34% of the participants wearing the mask (compared to a minimum of 83.34% with other evaluated masks).

摘要: 在过去的几年里，基于深度学习的面部识别(FR)模型展示了最先进的性能，即使在冠状病毒大流行期间戴防护医用口罩变得司空见惯。考虑到这些模型的出色性能，机器学习研究界对挑战它们的鲁棒性表现出了越来越大的兴趣。最初，研究人员在数字领域进行对抗性攻击，后来将攻击转移到物理领域。然而，在许多情况下，物理域中的攻击是显眼的，例如需要在脸上放置贴纸，因此可能会在现实环境(例如机场)中引起怀疑。在这篇文章中，我们提出了对抗面具，一种针对最先进的FR模型的物理对抗普遍扰动(UAP)，它以精心制作的模式的形式应用于人脸面具上。在我们的实验中，我们检查了我们的对手面具在广泛的FR模型体系结构和数据集上的可移植性。此外，我们在真实世界的实验中验证了我们的对抗面具的有效性，方法是将对抗图案印刷在织物医用面膜上，导致FR系统只能识别戴着该面具的3.34%的参与者(相比之下，使用其他评估的面具的最低识别率为83.34%)。



## **20. Stochastic Variance Reduced Ensemble Adversarial Attack for Boosting the Adversarial Transferability**

提高对抗性转移的随机方差降低集成对抗性攻击 cs.LG

10 pages, 5 figures, submitted to a conference for review

**SubmitDate**: 2021-11-21    [paper-pdf](http://arxiv.org/pdf/2111.10752v1)

**Authors**: Yifeng Xiong, Jiadong Lin, Min Zhang, John E. Hopcroft, Kun He

**Abstracts**: The black-box adversarial attack has attracted impressive attention for its practical use in the field of deep learning security, meanwhile, it is very challenging as there is no access to the network architecture or internal weights of the target model. Based on the hypothesis that if an example remains adversarial for multiple models, then it is more likely to transfer the attack capability to other models, the ensemble-based adversarial attack methods are efficient and widely used for black-box attacks. However, ways of ensemble attack are rather less investigated, and existing ensemble attacks simply fuse the outputs of all the models evenly. In this work, we treat the iterative ensemble attack as a stochastic gradient descent optimization process, in which the variance of the gradients on different models may lead to poor local optima. To this end, we propose a novel attack method called the stochastic variance reduced ensemble (SVRE) attack, which could reduce the gradient variance of the ensemble models and take full advantage of the ensemble attack. Empirical results on the standard ImageNet dataset demonstrate that the proposed method could boost the adversarial transferability and outperforms existing ensemble attacks significantly.

摘要: 黑盒对抗性攻击因其在深度学习安全领域的实际应用而备受关注，同时，由于不能访问目标模型的网络结构或内部权重，因此具有很大的挑战性。基于这样的假设，如果一个示例在多个模型上保持对抗性，则攻击能力更有可能转移到其他模型上，基于集成的对抗性攻击方法是一种有效的、广泛应用于黑盒攻击的方法。然而，集成攻击方式的研究相对较少，现有的集成攻击只是简单地将所有模型的输出均匀地融合。在本文中，我们将迭代集成攻击看作一个随机梯度下降优化过程，其中不同模型上梯度的变化可能导致局部最优解较差。为此，我们提出了一种新的攻击方法，称为随机方差减少集成(SVRE)攻击，它可以降低集成模型的梯度方差，并充分利用集成攻击的优势。在标准ImageNet数据集上的实验结果表明，该方法可以提高攻击的对抗性可转换性，并明显优于现有的集成攻击。



## **21. AEVA: Black-box Backdoor Detection Using Adversarial Extreme Value Analysis**

AEVA：基于对抗性极值分析的黑盒后门检测 cs.LG

**SubmitDate**: 2021-11-21    [paper-pdf](http://arxiv.org/pdf/2110.14880v3)

**Authors**: Junfeng Guo, Ang Li, Cong Liu

**Abstracts**: Deep neural networks (DNNs) are proved to be vulnerable against backdoor attacks. A backdoor is often embedded in the target DNNs through injecting a backdoor trigger into training examples, which can cause the target DNNs misclassify an input attached with the backdoor trigger. Existing backdoor detection methods often require the access to the original poisoned training data, the parameters of the target DNNs, or the predictive confidence for each given input, which are impractical in many real-world applications, e.g., on-device deployed DNNs. We address the black-box hard-label backdoor detection problem where the DNN is fully black-box and only its final output label is accessible. We approach this problem from the optimization perspective and show that the objective of backdoor detection is bounded by an adversarial objective. Further theoretical and empirical studies reveal that this adversarial objective leads to a solution with highly skewed distribution; a singularity is often observed in the adversarial map of a backdoor-infected example, which we call the adversarial singularity phenomenon. Based on this observation, we propose the adversarial extreme value analysis(AEVA) to detect backdoors in black-box neural networks. AEVA is based on an extreme value analysis of the adversarial map, computed from the monte-carlo gradient estimation. Evidenced by extensive experiments across multiple popular tasks and backdoor attacks, our approach is shown effective in detecting backdoor attacks under the black-box hard-label scenarios.

摘要: 深度神经网络(DNNs)被证明是易受后门攻击的。通过将后门触发器注入到训练示例中，通常将后门嵌入到目标DNN中，这可能导致目标DNN对与后门触发器附加的输入进行错误分类。现有的后门检测方法通常需要访问原始有毒训练数据、目标DNN的参数或每个给定输入的预测置信度，这在许多真实世界应用中是不切实际的，例如在设备上部署的DNN。我们解决了黑盒硬标签后门检测问题，其中DNN是完全黑盒的，并且只有其最终输出标签是可访问的。我们从优化的角度来研究这个问题，并证明了后门检测的目标是由一个对抗性目标限定的。进一步的理论和实证研究表明，这种对抗性目标导致了一个具有高度偏态分布的解决方案；在一个被后门感染的例子的对抗性地图中经常观察到一个奇点，我们称之为对抗性奇点现象。基于这一观察，我们提出了对抗性极值分析(AEVA)来检测黑盒神经网络中的后门。AEVA是基于对敌方地图的极值分析，通过蒙特卡洛梯度估计计算出来的。通过对多个流行任务和后门攻击的大量实验证明，我们的方法在黑盒硬标签场景下检测后门攻击是有效的。



## **22. Are Vision Transformers Robust to Patch Perturbations?**

视觉变形器对补丁扰动有健壮性吗？ cs.CV

**SubmitDate**: 2021-11-20    [paper-pdf](http://arxiv.org/pdf/2111.10659v1)

**Authors**: Jindong Gu, Volker Tresp, Yao Qin

**Abstracts**: The recent advances in Vision Transformer (ViT) have demonstrated its impressive performance in image classification, which makes it a promising alternative to Convolutional Neural Network (CNN). Unlike CNNs, ViT represents an input image as a sequence of image patches. The patch-wise input image representation makes the following question interesting: How does ViT perform when individual input image patches are perturbed with natural corruptions or adversarial perturbations, compared to CNNs? In this work, we study the robustness of vision transformers to patch-wise perturbations. Surprisingly, we find that vision transformers are more robust to naturally corrupted patches than CNNs, whereas they are more vulnerable to adversarial patches. Furthermore, we conduct extensive qualitative and quantitative experiments to understand the robustness to patch perturbations. We have revealed that ViT's stronger robustness to natural corrupted patches and higher vulnerability against adversarial patches are both caused by the attention mechanism. Specifically, the attention model can help improve the robustness of vision transformers by effectively ignoring natural corrupted patches. However, when vision transformers are attacked by an adversary, the attention mechanism can be easily fooled to focus more on the adversarially perturbed patches and cause a mistake.

摘要: 近年来，视觉变换器(VIT)在图像分类中表现出了令人印象深刻的性能，使其成为卷积神经网络(CNN)的一种有前途的替代方案。与CNN不同，VIT将输入图像表示为图像补丁序列。基于补丁的输入图像表示使得以下问题变得有趣：与CNN相比，当单个输入图像补丁受到自然破坏或敌意扰动时，VIT的性能如何？在这项工作中，我们研究了视觉转换器对面片扰动的鲁棒性。令人惊讶的是，我们发现视觉转换器对自然腐烂的补丁比CNN更健壮，而它们更容易受到敌意补丁的攻击。此外，我们还进行了大量的定性和定量实验，以了解该算法对补丁扰动的鲁棒性。我们发现，VIT对自然破坏补丁具有较强的健壮性，对敌意补丁具有较高的脆弱性，这都是由注意机制造成的。具体地说，注意力模型可以通过有效地忽略自然损坏的斑块来帮助提高视觉转换器的鲁棒性。然而，当视觉变形器受到敌人的攻击时，注意力机制很容易被愚弄，将注意力更多地集中在对手扰乱的补丁上，从而导致错误。



## **23. Modeling Design and Control Problems Involving Neural Network Surrogates**

涉及神经网络代理的建模设计与控制问题 math.OC

24 Pages, 11 Figures

**SubmitDate**: 2021-11-20    [paper-pdf](http://arxiv.org/pdf/2111.10489v1)

**Authors**: Dominic Yang, Prasanna Balaprakash, Sven Leyffer

**Abstracts**: We consider nonlinear optimization problems that involve surrogate models represented by neural networks. We demonstrate first how to directly embed neural network evaluation into optimization models, highlight a difficulty with this approach that can prevent convergence, and then characterize stationarity of such models. We then present two alternative formulations of these problems in the specific case of feedforward neural networks with ReLU activation: as a mixed-integer optimization problem and as a mathematical program with complementarity constraints. For the latter formulation we prove that stationarity at a point for this problem corresponds to stationarity of the embedded formulation. Each of these formulations may be solved with state-of-the-art optimization methods, and we show how to obtain good initial feasible solutions for these methods. We compare our formulations on three practical applications arising in the design and control of combustion engines, in the generation of adversarial attacks on classifier networks, and in the determination of optimal flows in an oil well network.

摘要: 我们考虑涉及以神经网络为代表的代理模型的非线性优化问题。我们首先演示了如何将神经网络评估直接嵌入到优化模型中，强调了这种方法可能会阻止收敛的一个困难，然后描述了这类模型的平稳性。然后，在具有RELU激活的前馈神经网络的具体情况下，我们给出了这些问题的两种可供选择的形式：作为混合整数优化问题和作为具有互补约束的数学规划。对于后一种公式，我们证明了该问题在某一点的平稳性对应于嵌入公式的平稳性。这些公式中的每一个都可以用最先进的优化方法求解，我们展示了如何为这些方法获得良好的初始可行解。我们比较了我们的公式在内燃机设计和控制、产生对分类器网络的敌意攻击和确定油井网络中的最优流量的三个实际应用中的应用。



## **24. Zero-Shot Certified Defense against Adversarial Patches with Vision Transformers**

使用Vision Transformers对敌方补丁进行零射击认证防御 cs.CV

12 pages, 5 figures

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2111.10481v1)

**Authors**: Yuheng Huang, Yuanchun Li

**Abstracts**: Adversarial patch attack aims to fool a machine learning model by arbitrarily modifying pixels within a restricted region of an input image. Such attacks are a major threat to models deployed in the physical world, as they can be easily realized by presenting a customized object in the camera view. Defending against such attacks is challenging due to the arbitrariness of patches, and existing provable defenses suffer from poor certified accuracy. In this paper, we propose PatchVeto, a zero-shot certified defense against adversarial patches based on Vision Transformer (ViT) models. Rather than training a robust model to resist adversarial patches which may inevitably sacrifice accuracy, PatchVeto reuses a pretrained ViT model without any additional training, which can achieve high accuracy on clean inputs while detecting adversarial patched inputs by simply manipulating the attention map of ViT. Specifically, each input is tested by voting over multiple inferences with different attention masks, where at least one inference is guaranteed to exclude the adversarial patch. The prediction is certifiably robust if all masked inferences reach consensus, which ensures that any adversarial patch would be detected with no false negative. Extensive experiments have shown that PatchVeto is able to achieve high certified accuracy (e.g. 67.1% on ImageNet for 2%-pixel adversarial patches), significantly outperforming state-of-the-art methods. The clean accuracy is the same as vanilla ViT models (81.8% on ImageNet) since the model parameters are directly reused. Meanwhile, our method can flexibly handle different adversarial patch sizes by simply changing the masking strategy.

摘要: 对抗性补丁攻击旨在通过任意修改输入图像的受限区域内的像素来愚弄机器学习模型。这类攻击是对部署在物理世界中的模型的主要威胁，因为它们可以通过在相机视图中呈现自定义对象来轻松实现。由于补丁程序的任意性，防御此类攻击具有挑战性，而且现有的可证明防御系统存在认证准确性差的问题。在本文中，我们提出了PatchVeto，一种基于视觉转换器(VIT)模型的零命中认证的恶意补丁防御方案。PatchVeto没有训练健壮的模型来抵抗不可避免地会牺牲准确性的对抗性补丁，而是重用了预先训练的VIT模型，无需任何额外的训练，通过简单地操作VIT的注意图，可以在检测干净输入的同时检测到对抗性补丁输入。具体地说，通过对具有不同注意掩码的多个推论进行投票来测试每个输入，其中至少有一个推论被保证排除敌意补丁。如果所有掩蔽的推论都达到共识，则预测是可证明的稳健的，这确保了任何敌意补丁都将被检测到而不会出现假阴性。广泛的实验表明，PatchVeto能够达到很高的认证准确率(例如，ImageNet上2%像素的对抗性补丁的准确率为67.1%)，远远超过最先进的方法。由于直接重用了模型参数，因此其清洁精度与普通VIT模型相同(在ImageNet上为81.8%)。同时，我们的方法只需改变掩蔽策略，就可以灵活地处理不同的敌意补丁大小。



## **25. Rethinking Clustering for Robustness**

重新考虑集群以实现健壮性 cs.LG

Accepted to the 32nd British Machine Vision Conference (BMVC'21)

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2006.07682v3)

**Authors**: Motasem Alfarra, Juan C. Pérez, Adel Bibi, Ali Thabet, Pablo Arbeláez, Bernard Ghanem

**Abstracts**: This paper studies how encouraging semantically-aligned features during deep neural network training can increase network robustness. Recent works observed that Adversarial Training leads to robust models, whose learnt features appear to correlate with human perception. Inspired by this connection from robustness to semantics, we study the complementary connection: from semantics to robustness. To do so, we provide a robustness certificate for distance-based classification models (clustering-based classifiers). Moreover, we show that this certificate is tight, and we leverage it to propose ClusTR (Clustering Training for Robustness), a clustering-based and adversary-free training framework to learn robust models. Interestingly, \textit{ClusTR} outperforms adversarially-trained networks by up to $4\%$ under strong PGD attacks.

摘要: 本文研究了在深度神经网络训练过程中鼓励语义对齐的特征如何提高网络的鲁棒性。最近的工作观察到，对抗性训练导致健壮的模型，其学习的特征似乎与人类的感知相关。受这种从鲁棒性到语义的联系的启发，我们研究了这种互补的联系：从语义到鲁棒性。为此，我们为基于距离的分类模型(基于聚类的分类器)提供了健壮性证书。此外，我们还证明了该证书是严格的，并利用该证书提出了ClusTR(聚类健壮性训练)，这是一个基于聚类的、无对手的训练框架，用于学习健壮模型。有趣的是，在强PGD攻击下，textit{ClusTR}的性能比经过恶意训练的网络高出4美元。



## **26. Meta Adversarial Perturbations**

元对抗扰动 cs.LG

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2111.10291v1)

**Authors**: Chia-Hung Yuan, Pin-Yu Chen, Chia-Mu Yu

**Abstracts**: A plethora of attack methods have been proposed to generate adversarial examples, among which the iterative methods have been demonstrated the ability to find a strong attack. However, the computation of an adversarial perturbation for a new data point requires solving a time-consuming optimization problem from scratch. To generate a stronger attack, it normally requires updating a data point with more iterations. In this paper, we show the existence of a meta adversarial perturbation (MAP), a better initialization that causes natural images to be misclassified with high probability after being updated through only a one-step gradient ascent update, and propose an algorithm for computing such perturbations. We conduct extensive experiments, and the empirical results demonstrate that state-of-the-art deep neural networks are vulnerable to meta perturbations. We further show that these perturbations are not only image-agnostic, but also model-agnostic, as a single perturbation generalizes well across unseen data points and different neural network architectures.

摘要: 已经提出了大量的攻击方法来生成对抗性实例，其中迭代方法已被证明具有发现强攻击的能力。然而，计算新数据点的对抗性扰动需要从头开始解决耗时的优化问题。要生成更强的攻击，通常需要更新迭代次数更多的数据点。本文证明了元对抗扰动(MAP)的存在性，并提出了一种计算这种扰动的算法。MAP是一种较好的初始化方法，它只通过一步梯度上升更新就会导致自然图像在更新后被高概率地误分类。我们进行了大量的实验，实验结果表明，最新的深度神经网络容易受到元扰动的影响。我们进一步表明，这些扰动不仅是图像不可知的，而且也是模型不可知的，因为单个扰动很好地概括了不可见的数据点和不同的神经网络结构。



## **27. Resilience from Diversity: Population-based approach to harden models against adversarial attacks**

来自多样性的弹性：基于人口的方法来强化模型对抗对手攻击的能力 cs.LG

10 pages, 6 figures, 5 tables

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2111.10272v1)

**Authors**: Jasser Jasser, Ivan Garibay

**Abstracts**: Traditional deep learning models exhibit intriguing vulnerabilities that allow an attacker to force them to fail at their task. Notorious attacks such as the Fast Gradient Sign Method (FGSM) and the more powerful Projected Gradient Descent (PGD) generate adversarial examples by adding a magnitude of perturbation $\epsilon$ to the input's computed gradient, resulting in a deterioration of the effectiveness of the model's classification. This work introduces a model that is resilient to adversarial attacks. Our model leverages a well established principle from biological sciences: population diversity produces resilience against environmental changes. More precisely, our model consists of a population of $n$ diverse submodels, each one of them trained to individually obtain a high accuracy for the task at hand, while forced to maintain meaningful differences in their weight tensors. Each time our model receives a classification query, it selects a submodel from its population at random to answer the query. To introduce and maintain diversity in population of submodels, we introduce the concept of counter linking weights. A Counter-Linked Model (CLM) consists of submodels of the same architecture where a periodic random similarity examination is conducted during the simultaneous training to guarantee diversity while maintaining accuracy. In our testing, CLM robustness got enhanced by around 20% when tested on the MNIST dataset and at least 15% when tested on the CIFAR-10 dataset. When implemented with adversarially trained submodels, this methodology achieves state-of-the-art robustness. On the MNIST dataset with $\epsilon=0.3$, it achieved 94.34% against FGSM and 91% against PGD. On the CIFAR-10 dataset with $\epsilon=8/255$, it achieved 62.97% against FGSM and 59.16% against PGD.

摘要: 传统的深度学习模型显示出耐人寻味的漏洞，使得攻击者能够迫使它们在任务中失败。诸如快速梯度符号法(FGSM)和更强大的投影梯度下降法(PGD)等臭名昭著的攻击通过在输入的计算梯度上添加扰动幅度$\ε$来产生敌意示例，导致模型分类效果的恶化。这项工作引入了一个对对手攻击具有弹性的模型。我们的模型充分利用了来自生物科学的一个公认的原则：种群多样性产生了对环境变化的适应能力。更准确地说，我们的模型由$n$各式各样的子模型组成，每个子模型都经过训练，以单独获得手头任务的高精度，同时被迫保持其权重张量的有意义的差异。我们的模型每次收到分类查询时，都会从其总体中随机选择一个子模型来回答查询。为了引入和维持子模型种群的多样性，我们引入了反链接权的概念。反向链接模型(CLM)由相同体系结构的子模型组成，其中在同时训练期间进行周期性的随机相似性检查，以在保持准确性的同时保证多样性。在我们的测试中，在MNIST数据集上测试时，CLM健壮性提高了约20%，在CIFAR-10数据集上测试时，CLM健壮性至少提高了15%。当使用相反训练子模型实施时，该方法实现了最先进的健壮性。在$\epsilon=0.3$的MNIST数据集上，对FGSM和PGD的识别率分别达到94.34%和91%。在$\epsilon=8/255$的CIFAR-10数据集上，对FGSM的识别率达到62.97%，对PGD的识别率达到59.16%。



## **28. Fast Minimum-norm Adversarial Attacks through Adaptive Norm Constraints**

基于自适应范数约束的快速最小范数对抗攻击 cs.LG

Accepted at NeurIPS'21

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2102.12827v3)

**Authors**: Maura Pintor, Fabio Roli, Wieland Brendel, Battista Biggio

**Abstracts**: Evaluating adversarial robustness amounts to finding the minimum perturbation needed to have an input sample misclassified. The inherent complexity of the underlying optimization requires current gradient-based attacks to be carefully tuned, initialized, and possibly executed for many computationally-demanding iterations, even if specialized to a given perturbation model. In this work, we overcome these limitations by proposing a fast minimum-norm (FMN) attack that works with different $\ell_p$-norm perturbation models ($p=0, 1, 2, \infty$), is robust to hyperparameter choices, does not require adversarial starting points, and converges within few lightweight steps. It works by iteratively finding the sample misclassified with maximum confidence within an $\ell_p$-norm constraint of size $\epsilon$, while adapting $\epsilon$ to minimize the distance of the current sample to the decision boundary. Extensive experiments show that FMN significantly outperforms existing attacks in terms of convergence speed and computation time, while reporting comparable or even smaller perturbation sizes.

摘要: 评估对手健壮性相当于找到输入样本错误分类所需的最小扰动。底层优化的固有复杂性要求对当前基于梯度的攻击进行仔细的调整、初始化，并可能对许多计算要求很高的迭代执行，即使专门针对给定的扰动模型也是如此。在这项工作中，我们提出了一种快速的最小范数(FMN)攻击，它适用于不同的$\ell_p$-范数扰动模型($p=0，1，2，\infty$)，对超参数选择具有鲁棒性，不需要对抗性的起点，并且在几个轻量级步骤内收敛。它的工作原理是迭代地在大小为$\epsilon$的$\ell_p$-范数约束内找到错误分类的样本，同时调整$\epsilon$以最小化当前样本到决策边界的距离。大量的实验表明，FMN在收敛速度和计算时间方面明显优于现有的攻击，同时报告的扰动大小与现有攻击相当甚至更小。



## **29. Federated Learning for Malware Detection in IoT Devices**

物联网设备中恶意软件检测的联合学习 cs.CR

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2104.09994v3)

**Authors**: Valerian Rey, Pedro Miguel Sánchez Sánchez, Alberto Huertas Celdrán, Gérôme Bovet, Martin Jaggi

**Abstracts**: This work investigates the possibilities enabled by federated learning concerning IoT malware detection and studies security issues inherent to this new learning paradigm. In this context, a framework that uses federated learning to detect malware affecting IoT devices is presented. N-BaIoT, a dataset modeling network traffic of several real IoT devices while affected by malware, has been used to evaluate the proposed framework. Both supervised and unsupervised federated models (multi-layer perceptron and autoencoder) able to detect malware affecting seen and unseen IoT devices of N-BaIoT have been trained and evaluated. Furthermore, their performance has been compared to two traditional approaches. The first one lets each participant locally train a model using only its own data, while the second consists of making the participants share their data with a central entity in charge of training a global model. This comparison has shown that the use of more diverse and large data, as done in the federated and centralized methods, has a considerable positive impact on the model performance. Besides, the federated models, while preserving the participant's privacy, show similar results as the centralized ones. As an additional contribution and to measure the robustness of the federated approach, an adversarial setup with several malicious participants poisoning the federated model has been considered. The baseline model aggregation averaging step used in most federated learning algorithms appears highly vulnerable to different attacks, even with a single adversary. The performance of other model aggregation functions acting as countermeasures is thus evaluated under the same attack scenarios. These functions provide a significant improvement against malicious participants, but more efforts are still needed to make federated approaches robust.

摘要: 这项工作调查了有关物联网恶意软件检测的联合学习带来的可能性，并研究了这一新学习范式固有的安全问题。在此背景下，提出了一种使用联合学习来检测影响物联网设备的恶意软件的框架。N-BaIoT是一个数据集，它模拟了几个真实物联网设备在受到恶意软件影响时的网络流量，已经被用来评估所提出的框架。有监督和无监督的联合模型(多层感知器和自动编码器)能够检测影响N-BaIoT看得见和看不见的物联网设备的恶意软件，已经进行了训练和评估。此外，还将它们的性能与两种传统方法进行了比较。第一种方法允许每个参与者仅使用自己的数据在本地训练模型，而第二种方法包括使参与者与负责训练全局模型的中央实体共享他们的数据。这种比较表明，使用更多样化和更大的数据(如在联邦和集中式方法中所做的那样)对模型性能有相当大的积极影响。此外，联邦模型在保护参与者隐私的同时，表现出与集中式模型相似的结果。作为另一项贡献和衡量联邦方法的健壮性，考虑了几个恶意参与者毒害联邦模型的对抗性设置。大多数联合学习算法中使用的基准模型聚合平均步骤似乎非常容易受到不同的攻击，即使是在单个对手的情况下也是如此。因此，在相同的攻击场景下，评估了用作对策的其他模型聚集函数的性能。这些函数提供了针对恶意参与者的显著改进，但仍需要付出更多努力才能使联合方法变得健壮。



## **30. Fooling Adversarial Training with Inducing Noise**

用诱导噪音愚弄对手训练 cs.LG

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2111.10130v1)

**Authors**: Zhirui Wang, Yifei Wang, Yisen Wang

**Abstracts**: Adversarial training is widely believed to be a reliable approach to improve model robustness against adversarial attack. However, in this paper, we show that when trained on one type of poisoned data, adversarial training can also be fooled to have catastrophic behavior, e.g., $<1\%$ robust test accuracy with $>90\%$ robust training accuracy on CIFAR-10 dataset. Previously, there are other types of noise poisoned in the training data that have successfully fooled standard training ($15.8\%$ standard test accuracy with $99.9\%$ standard training accuracy on CIFAR-10 dataset), but their poisonings can be easily removed when adopting adversarial training. Therefore, we aim to design a new type of inducing noise, named ADVIN, which is an irremovable poisoning of training data. ADVIN can not only degrade the robustness of adversarial training by a large margin, for example, from $51.7\%$ to $0.57\%$ on CIFAR-10 dataset, but also be effective for fooling standard training ($13.1\%$ standard test accuracy with $100\%$ standard training accuracy). Additionally, ADVIN can be applied to preventing personal data (like selfies) from being exploited without authorization under whether standard or adversarial training.

摘要: 对抗性训练被广泛认为是提高模型对对抗性攻击鲁棒性的可靠方法。然而，在本文中，我们证明了当在一种类型的中毒数据上进行训练时，对抗性训练也可能被欺骗为具有灾难性行为，例如，在CIFAR-10数据集上，$<1$鲁棒测试精度与$>90$鲁棒训练精度。以前，训练数据中还有其他类型的噪声中毒已经成功地欺骗了标准训练(在CIFAR-10数据集上，$15.8\$标准测试精度和$99.9\$标准训练精度)，但当采用对抗性训练时，它们的中毒可以很容易地消除。因此，我们的目标是设计一种新型的诱导噪声，称为ADVIN，它是对训练数据的一种不可移除的毒害。ADVIN不仅可以大幅度降低对抗性训练的鲁棒性，例如在CIFAR-10数据集上从51.7美元降到0.57美元，而且对欺骗标准训练也是有效的(13.1美元标准测试精度和100美元标准训练精度)。此外，ADVIN可用于防止个人数据(如自拍)在未经授权的情况下被利用，无论是在标准训练还是对抗性训练下。



## **31. Exposing Weaknesses of Malware Detectors with Explainability-Guided Evasion Attacks**

利用可解析性引导的规避攻击暴露恶意软件检测器的弱点 cs.CR

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2111.10085v1)

**Authors**: Wei Wang, Ruoxi Sun, Tian Dong, Shaofeng Li, Minhui Xue, Gareth Tyson, Haojin Zhu

**Abstracts**: Numerous open-source and commercial malware detectors are available. However, the efficacy of these tools has been threatened by new adversarial attacks, whereby malware attempts to evade detection using, for example, machine learning techniques. In this work, we design an adversarial evasion attack that relies on both feature-space and problem-space manipulation. It uses explainability-guided feature selection to maximize evasion by identifying the most critical features that impact detection. We then use this attack as a benchmark to evaluate several state-of-the-art malware detectors. We find that (i) state-of-the-art malware detectors are vulnerable to even simple evasion strategies, and they can easily be tricked using off-the-shelf techniques; (ii) feature-space manipulation and problem-space obfuscation can be combined to enable evasion without needing white-box understanding of the detector; (iii) we can use explainability approaches (e.g., SHAP) to guide the feature manipulation and explain how attacks can transfer across multiple detectors. Our findings shed light on the weaknesses of current malware detectors, as well as how they can be improved.

摘要: 有许多开源和商业恶意软件检测器可用。然而，这些工具的有效性已经受到新的敌意攻击的威胁，由此恶意软件试图使用例如机器学习技术来逃避检测。在这项工作中，我们设计了一种同时依赖于特征空间和问题空间操作的对抗性逃避攻击。它使用以可解释性为导向的特征选择，通过识别影响检测的最关键特征来最大限度地规避。然后，我们使用此攻击作为基准来评估几种最先进的恶意软件检测器。我们发现：(I)最新的恶意软件检测器容易受到即使是简单的规避策略的攻击，并且很容易使用现成的技术欺骗它们；(Ii)特征空间操作和问题空间混淆可以结合起来实现规避，而不需要了解检测器的白盒；(Iii)我们可以使用解释性方法(例如Shap)来指导特征操作，并解释攻击如何在多个检测器之间传输。我们的发现揭示了当前恶意软件检测器的弱点，以及如何改进它们。



## **32. Enhanced countering adversarial attacks via input denoising and feature restoring**

通过输入去噪和特征恢复增强了对抗敌方攻击的能力 cs.CV

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2111.10075v1)

**Authors**: Yanni Li, Wenhui Zhang, Jiawei Liu, Xiaoli Kou, Hui Li, Jiangtao Cui

**Abstracts**: Despite the fact that deep neural networks (DNNs) have achieved prominent performance in various applications, it is well known that DNNs are vulnerable to adversarial examples/samples (AEs) with imperceptible perturbations in clean/original samples. To overcome the weakness of the existing defense methods against adversarial attacks, which damages the information on the original samples, leading to the decrease of the target classifier accuracy, this paper presents an enhanced countering adversarial attack method IDFR (via Input Denoising and Feature Restoring). The proposed IDFR is made up of an enhanced input denoiser (ID) and a hidden lossy feature restorer (FR) based on the convex hull optimization. Extensive experiments conducted on benchmark datasets show that the proposed IDFR outperforms the various state-of-the-art defense methods, and is highly effective for protecting target models against various adversarial black-box or white-box attacks. \footnote{Souce code is released at: \href{https://github.com/ID-FR/IDFR}{https://github.com/ID-FR/IDFR}}

摘要: 尽管深度神经网络(DNNs)在各种应用中取得了突出的性能，但众所周知，DNN在干净的/原始的样本中容易受到具有不可察觉扰动的对抗性示例/样本(AEs)的影响。针对现有对抗攻击防御方法破坏原始样本信息，导致目标分类器准确率下降的缺点，提出了一种改进的对抗对抗攻击方法IDFR(通过输入去噪和特征恢复)。提出的IDFR由基于凸壳优化的增强型输入去噪器(ID)和隐藏有损特征恢复器(FR)组成。在基准数据集上进行的大量实验表明，IDFR的性能优于各种先进的防御方法，对于保护目标模型免受各种对抗性的黑盒或白盒攻击是非常有效的。\脚注{源代码发布地址：\href{https://github.com/ID-FR/IDFR}{https://github.com/ID-FR/IDFR}}



## **33. Towards Efficiently Evaluating the Robustness of Deep Neural Networks in IoT Systems: A GAN-based Method**

物联网系统中深度神经网络健壮性的有效评估：一种基于GAN的方法 cs.LG

arXiv admin note: text overlap with arXiv:2002.02196

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2111.10055v1)

**Authors**: Tao Bai, Jun Zhao, Jinlin Zhu, Shoudong Han, Jiefeng Chen, Bo Li, Alex Kot

**Abstracts**: Intelligent Internet of Things (IoT) systems based on deep neural networks (DNNs) have been widely deployed in the real world. However, DNNs are found to be vulnerable to adversarial examples, which raises people's concerns about intelligent IoT systems' reliability and security. Testing and evaluating the robustness of IoT systems becomes necessary and essential. Recently various attacks and strategies have been proposed, but the efficiency problem remains unsolved properly. Existing methods are either computationally extensive or time-consuming, which is not applicable in practice. In this paper, we propose a novel framework called Attack-Inspired GAN (AI-GAN) to generate adversarial examples conditionally. Once trained, it can generate adversarial perturbations efficiently given input images and target classes. We apply AI-GAN on different datasets in white-box settings, black-box settings and targeted models protected by state-of-the-art defenses. Through extensive experiments, AI-GAN achieves high attack success rates, outperforming existing methods, and reduces generation time significantly. Moreover, for the first time, AI-GAN successfully scales to complex datasets e.g. CIFAR-100 and ImageNet, with about $90\%$ success rates among all classes.

摘要: 基于深度神经网络(DNNs)的智能物联网(IoT)系统已经在现实世界中得到了广泛的部署。然而，DNN被发现容易受到敌意示例的攻击，这引发了人们对智能物联网系统可靠性和安全性的担忧。测试和评估物联网系统的健壮性变得必要和必要。近年来，人们提出了各种攻击和策略，但效率问题一直没有得到很好的解决。现有的方法要么计算量大，要么费时费力，在实际应用中并不适用。本文提出了一种新的框架，称为攻击启发的GAN(AI-GAN)，用于有条件地生成对抗性示例。一旦训练完成，它可以有效地产生给定输入图像和目标类的对抗性扰动。我们将AI-GAN应用于白盒设置、黑盒设置和受最先进防御保护的目标模型中的不同数据集。通过大量的实验，AI-GAN获得了较高的攻击成功率，性能优于现有的方法，并显著减少了生成时间。此外，AI-GAN首次成功扩展到CIFAR-100和ImageNet等复杂数据集，所有类别的成功率约为90美元。



## **34. Generating Unrestricted 3D Adversarial Point Clouds**

生成不受限制的3D对抗性点云 cs.CV

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2111.08973v2)

**Authors**: Xuelong Dai, Yanjie Li, Hua Dai, Bin Xiao

**Abstracts**: Utilizing 3D point cloud data has become an urgent need for the deployment of artificial intelligence in many areas like facial recognition and self-driving. However, deep learning for 3D point clouds is still vulnerable to adversarial attacks, e.g., iterative attacks, point transformation attacks, and generative attacks. These attacks need to restrict perturbations of adversarial examples within a strict bound, leading to the unrealistic adversarial 3D point clouds. In this paper, we propose an Adversarial Graph-Convolutional Generative Adversarial Network (AdvGCGAN) to generate visually realistic adversarial 3D point clouds from scratch. Specifically, we use a graph convolutional generator and a discriminator with an auxiliary classifier to generate realistic point clouds, which learn the latent distribution from the real 3D data. The unrestricted adversarial attack loss is incorporated in the special adversarial training of GAN, which enables the generator to generate the adversarial examples to spoof the target network. Compared with the existing state-of-art attack methods, the experiment results demonstrate the effectiveness of our unrestricted adversarial attack methods with a higher attack success rate and visual quality. Additionally, the proposed AdvGCGAN can achieve better performance against defense models and better transferability than existing attack methods with strong camouflage.

摘要: 利用三维点云数据已经成为人脸识别、自动驾驶等多个领域人工智能部署的迫切需要。然而，三维点云的深度学习仍然容易受到对抗性攻击，如迭代攻击、点变换攻击和生成性攻击。这些攻击需要将对抗性示例的扰动限制在一个严格的范围内，从而导致不真实的对抗性三维点云。本文提出了一种对抗性图形-卷积生成对抗性网络(AdvGCGAN)，用于从头开始生成视觉逼真的对抗性三维点云。具体地说，我们使用一个图形卷积生成器和一个带有辅助分类器的鉴别器来生成逼真的点云，从真实的3D数据中学习潜在的分布。将不受限制的对抗性攻击损失纳入到GAN的特殊对抗性训练中，使生成器能够生成欺骗目标网络的对抗性示例。实验结果表明，与现有的现有攻击方法相比，本文提出的无限制对抗性攻击方法具有更高的攻击成功率和视觉质量。此外，与现有的伪装性强的攻击方法相比，提出的AdvGCGAN能够获得更好的防御模型性能和更好的可移植性。



## **35. Arbitrarily Fast Switched Distributed Stabilization of Partially Unknown Interconnected Multiagent Systems: A Proactive Cyber Defense Perspective**

部分未知互联多智能体系统的任意快速切换分布镇定：一种主动网络防御的观点 cs.SY

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2110.14199v2)

**Authors**: Vahid Rezaei, Jafar Haadi Jafarian, Douglas C. Sicker

**Abstracts**: A design framework recently has been developed to stabilize interconnected multiagent systems in a distributed manner, and systematically capture the architectural aspect of cyber-physical systems. Such a control theoretic framework, however, results in a stabilization protocol which is passive with respect to the cyber attacks and conservative regarding the guaranteed level of resiliency. We treat the control layer topology and stabilization gains as the degrees of freedom, and develop a mixed control and cybersecurity design framework to address the above concerns. From a control perspective, despite the agent layer modeling uncertainties and perturbations, we propose a new step-by-step procedure to design a set of control sublayers for an arbitrarily fast switching of the control layer topology. From a proactive cyber defense perspective, we propose a satisfiability modulo theory formulation to obtain a set of control sublayer structures with security considerations, and offer a frequent and fast mutation of these sublayers such that the control layer topology will remain unpredictable for the adversaries. We prove the robust input-to-state stability of the two-layer interconnected multiagent system, and validate the proposed ideas in simulation.

摘要: 最近开发了一个设计框架，用于以分布式方式稳定互连的多Agent系统，并系统地捕获网络物理系统的体系结构方面。然而，这样的控制理论框架导致了稳定协议，该协议对于网络攻击是被动的，并且对于保证的弹性水平是保守的。我们将控制层的拓扑结构和稳定增益作为自由度，提出了一种混合控制和网络安全设计框架来解决上述问题。从控制的角度来看，尽管Agent层建模存在不确定性和扰动，但我们提出了一种新的分步过程来设计一组控制子层，以实现控制层拓扑的任意快速切换。从主动网络防御的角度出发，我们提出了一种可满足性模理论公式，以获得一组考虑安全因素的控制子层结构，并对这些子层进行频繁而快速的突变，使得控制层拓扑对攻击者来说仍然是不可预测的。证明了两层互联多智能体系统的鲁棒输入-状态稳定性，并在仿真中验证了所提出的思想。



## **36. TnT Attacks! Universal Naturalistic Adversarial Patches Against Deep Neural Network Systems**

TNT攻击！针对深度神经网络系统的普遍自然主义对抗性补丁 cs.CV

We demonstrate physical deployments in multiple videos at  https://tntattacks.github.io/

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2111.09999v1)

**Authors**: Bao Gia Doan, Minhui Xue, Shiqing Ma, Ehsan Abbasnejad, Damith C. Ranasinghe

**Abstracts**: Deep neural networks are vulnerable to attacks from adversarial inputs and, more recently, Trojans to misguide or hijack the decision of the model. We expose the existence of an intriguing class of bounded adversarial examples -- Universal NaTuralistic adversarial paTches -- we call TnTs, by exploring the superset of the bounded adversarial example space and the natural input space within generative adversarial networks. Now, an adversary can arm themselves with a patch that is naturalistic, less malicious-looking, physically realizable, highly effective -- achieving high attack success rates, and universal. A TnT is universal because any input image captured with a TnT in the scene will: i) misguide a network (untargeted attack); or ii) force the network to make a malicious decision (targeted attack). Interestingly, now, an adversarial patch attacker has the potential to exert a greater level of control -- the ability to choose a location independent, natural-looking patch as a trigger in contrast to being constrained to noisy perturbations -- an ability is thus far shown to be only possible with Trojan attack methods needing to interfere with the model building processes to embed a backdoor at the risk discovery; but, still realize a patch deployable in the physical world. Through extensive experiments on the large-scale visual classification task, ImageNet with evaluations across its entire validation set of 50,000 images, we demonstrate the realistic threat from TnTs and the robustness of the attack. We show a generalization of the attack to create patches achieving higher attack success rates than existing state-of-the-art methods. Our results show the generalizability of the attack to different visual classification tasks (CIFAR-10, GTSRB, PubFig) and multiple state-of-the-art deep neural networks such as WideResnet50, Inception-V3 and VGG-16.

摘要: 深层神经网络容易受到敌意输入和最近的特洛伊木马程序的攻击，以误导或劫持模型的决策。我们通过探索生成对抗性网络中的有界对抗性实例空间和自然输入空间的超集，揭示了一类有趣的有界对抗性实例的存在--泛自然主义对抗性斑块--我们称之为TNTs。现在，对手可以用一种自然主义的、看起来不那么恶毒的、物理上可实现的、高效的补丁来武装自己--实现高攻击成功率和通用性。TNT是通用的，因为在场景中使用TNT捕获的任何输入图像将：i)误导网络(非定向攻击)；或者ii)迫使网络做出恶意决策(定向攻击)。有趣的是，现在，敌意补丁攻击者有可能施加更高级别的控制--能够选择与位置无关的、看起来自然的补丁作为触发器，而不是被限制在嘈杂的干扰中--到目前为止，这种能力被证明只有在需要干扰模型构建过程以在风险发现处嵌入后门的特洛伊木马攻击方法中才是可能的；但是，仍然可以在物理世界中实现可部署的补丁。通过在大规模视觉分类任务ImageNet上的大量实验，对其50,000张图像的整个验证集进行评估，我们证明了TNT的现实威胁和攻击的健壮性。我们展示了创建补丁的攻击的泛化，实现了比现有最先进的方法更高的攻击成功率。实验结果表明，该攻击可推广到不同的视觉分类任务(CIFAR-10、GTSRB、PubFig)和WideResnet50、Inception-V3、VGG-16等多种深度神经网络。



## **37. Combinatorial Bandits under Strategic Manipulations**

战略操纵下的组合强盗 cs.LG

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2102.12722v4)

**Authors**: Jing Dong, Ke Li, Shuai Li, Baoxiang Wang

**Abstracts**: Strategic behavior against sequential learning methods, such as "click framing" in real recommendation systems, have been widely observed. Motivated by such behavior we study the problem of combinatorial multi-armed bandits (CMAB) under strategic manipulations of rewards, where each arm can modify the emitted reward signals for its own interest. This characterization of the adversarial behavior is a relaxation of previously well-studied settings such as adversarial attacks and adversarial corruption. We propose a strategic variant of the combinatorial UCB algorithm, which has a regret of at most $O(m\log T + m B_{max})$ under strategic manipulations, where $T$ is the time horizon, $m$ is the number of arms, and $B_{max}$ is the maximum budget of an arm. We provide lower bounds on the budget for arms to incur certain regret of the bandit algorithm. Extensive experiments on online worker selection for crowdsourcing systems, online influence maximization and online recommendations with both synthetic and real datasets corroborate our theoretical findings on robustness and regret bounds, in a variety of regimes of manipulation budgets.

摘要: 针对顺序学习方法的策略性行为，如真实推荐系统中的“点击成帧”，已经被广泛观察到。在这种行为的激励下，我们研究了战略报酬操纵下的组合多臂强盗(CMAB)问题，其中每个手臂都可以为了自己的利益而修改发出的奖励信号。对抗性行为的这种表征是对先前研究得很好的设置的放松，例如对抗性攻击和对抗性腐败。提出了一种组合UCB算法的策略变体，该算法在策略操作下最多有$O(mlogT+mBmax})$，其中$T$是时间范围，$m$是臂的数量，$Bmax}$是ARM的最大预算。我们提供了武器预算的下限，以引起对强盗算法的一定遗憾。通过对众包系统的在线员工选择、在线影响力最大化和在线推荐的大量实验，使用合成和真实数据集证实了我们在不同预算操纵机制下关于稳健性和后悔界限的理论发现。



## **38. A Review of Adversarial Attack and Defense for Classification Methods**

对抗性攻防分类方法综述 cs.CR

**SubmitDate**: 2021-11-18    [paper-pdf](http://arxiv.org/pdf/2111.09961v1)

**Authors**: Yao Li, Minhao Cheng, Cho-Jui Hsieh, Thomas C. M. Lee

**Abstracts**: Despite the efficiency and scalability of machine learning systems, recent studies have demonstrated that many classification methods, especially deep neural networks (DNNs), are vulnerable to adversarial examples; i.e., examples that are carefully crafted to fool a well-trained classification model while being indistinguishable from natural data to human. This makes it potentially unsafe to apply DNNs or related methods in security-critical areas. Since this issue was first identified by Biggio et al. (2013) and Szegedy et al.(2014), much work has been done in this field, including the development of attack methods to generate adversarial examples and the construction of defense techniques to guard against such examples. This paper aims to introduce this topic and its latest developments to the statistical community, primarily focusing on the generation and guarding of adversarial examples. Computing codes (in python and R) used in the numerical experiments are publicly available for readers to explore the surveyed methods. It is the hope of the authors that this paper will encourage more statisticians to work on this important and exciting field of generating and defending against adversarial examples.

摘要: 尽管机器学习系统具有很高的效率和可扩展性，但最近的研究表明，许多分类方法，特别是深度神经网络(DNNs)，容易受到敌意示例的攻击；即，精心设计的示例欺骗了训练有素的分类模型，同时又无法从自然数据和人类数据中区分出来。这使得在安全关键区域应用DNN或相关方法可能不安全。因为这个问题是由Biggio等人首先发现的。正如Szegedy等人(2014)和Szegedy等人(2013)所做的那样，在这一领域已经做了很多工作，包括开发攻击方法来生成对抗性示例，以及构建防御技术来防范此类示例。本文旨在向统计界介绍这一主题及其最新进展，主要集中在对抗性例子的产生和保护上。在数值实验中使用的计算代码(Python和R)是公开的，供读者探索所调查的方法。作者希望这篇论文能鼓励更多的统计学家致力于这一重要而令人兴奋的领域--生成和防御敌意例子。



## **39. Resilient Consensus-based Multi-agent Reinforcement Learning with Function Approximation**

基于弹性共识的函数逼近多智能体强化学习 cs.LG

**SubmitDate**: 2021-11-18    [paper-pdf](http://arxiv.org/pdf/2111.06776v2)

**Authors**: Martin Figura, Yixuan Lin, Ji Liu, Vijay Gupta

**Abstracts**: Adversarial attacks during training can strongly influence the performance of multi-agent reinforcement learning algorithms. It is, thus, highly desirable to augment existing algorithms such that the impact of adversarial attacks on cooperative networks is eliminated, or at least bounded. In this work, we consider a fully decentralized network, where each agent receives a local reward and observes the global state and action. We propose a resilient consensus-based actor-critic algorithm, whereby each agent estimates the team-average reward and value function, and communicates the associated parameter vectors to its immediate neighbors. We show that in the presence of Byzantine agents, whose estimation and communication strategies are completely arbitrary, the estimates of the cooperative agents converge to a bounded consensus value with probability one, provided that there are at most $H$ Byzantine agents in the neighborhood of each cooperative agent and the network is $(2H+1)$-robust. Furthermore, we prove that the policy of the cooperative agents converges with probability one to a bounded neighborhood around a local maximizer of their team-average objective function under the assumption that the policies of the adversarial agents asymptotically become stationary.

摘要: 训练过程中的对抗性攻击会严重影响多智能体强化学习算法的性能。因此，非常需要对现有算法进行扩充，以便消除或至少有界地消除对抗性攻击对协作网络的影响。在这项工作中，我们考虑了一个完全分散的网络，在这个网络中，每个代理都会获得局部奖励，并观察全局状态和行动。我们提出了一种弹性的基于共识的行动者-批评者算法，其中每个Agent估计团队平均奖励和价值函数，并将相关的参数向量传达给它的直接邻居。我们证明了当拜占庭代理的估计和通信策略完全任意时，假设每个合作代理的邻域中至多有$H$拜占庭代理，并且网络是$(2H+1)$-鲁棒的，则合作代理的估计以概率1收敛到有界的合意值。在假设对抗性Agent的策略渐近平稳的前提下，证明了合作Agent的策略以概率1收敛到其团队平均目标函数的局部极大值附近的有界邻域。



## **40. Robust Person Re-identification with Multi-Modal Joint Defence**

基于多模态联合防御的鲁棒人物再识别 cs.CV

**SubmitDate**: 2021-11-18    [paper-pdf](http://arxiv.org/pdf/2111.09571v1)

**Authors**: Yunpeng Gong, Lifei Chen

**Abstracts**: The Person Re-identification (ReID) system based on metric learning has been proved to inherit the vulnerability of deep neural networks (DNNs), which are easy to be fooled by adversarail metric attacks. Existing work mainly relies on adversarial training for metric defense, and more methods have not been fully studied. By exploring the impact of attacks on the underlying features, we propose targeted methods for metric attacks and defence methods. In terms of metric attack, we use the local color deviation to construct the intra-class variation of the input to attack color features. In terms of metric defenses, we propose a joint defense method which includes two parts of proactive defense and passive defense. Proactive defense helps to enhance the robustness of the model to color variations and the learning of structure relations across multiple modalities by constructing different inputs from multimodal images, and passive defense exploits the invariance of structural features in a changing pixel space by circuitous scaling to preserve structural features while eliminating some of the adversarial noise. Extensive experiments demonstrate that the proposed joint defense compared with the existing adversarial metric defense methods which not only against multiple attacks at the same time but also has not significantly reduced the generalization capacity of the model. The code is available at https://github.com/finger-monkey/multi-modal_joint_defence.

摘要: 基于度量学习的人物识别(ReID)系统继承了深层神经网络(DNNs)易被恶意度量攻击欺骗的弱点。现有的工作主要依靠对抗性训练进行度量防御，更多的方法还没有得到充分的研究。通过研究攻击对底层特征的影响，提出了有针对性的度量攻击方法和防御方法。在度量攻击方面，我们利用局部颜色偏差来构造输入的类内变异来攻击颜色特征。在度量防御方面，我们提出了一种包括主动防御和被动防御两部分的联合防御方法。主动防御通过从多模态图像构造不同的输入来增强模型对颜色变化的鲁棒性和跨多模态的结构关系的学习，而被动防御通过迂回缩放利用结构特征在变化的像素空间中的不变性来保留结构特征，同时消除一些对抗性噪声。大量实验表明，与现有的对抗性度量防御方法相比，本文提出的联合防御方法不仅可以同时防御多个攻击，而且没有显着降低模型的泛化能力。代码可在https://github.com/finger-monkey/multi-modal_joint_defence.上获得



## **41. DPA: Learning Robust Physical Adversarial Camouflages for Object Detectors**

DPA：学习对象检测器的健壮物理对抗伪装 cs.CV

**SubmitDate**: 2021-11-18    [paper-pdf](http://arxiv.org/pdf/2109.00124v2)

**Authors**: Yexin Duan, Jialin Chen, Xingyu Zhou, Junhua Zou, Zhengyun He, Wu Zhang, Jin Zhang, Zhisong Pan

**Abstracts**: Adversarial attacks are feasible in the real world for object detection. However, most of the previous works have tried to learn local "patches" applied to an object to fool detectors, which become less effective in squint view angles. To address this issue, we propose the Dense Proposals Attack (DPA) to learn one-piece, physical, and targeted adversarial camouflages for detectors. The camouflages are one-piece because they are generated as a whole for an object, physical because they remain adversarial when filmed under arbitrary viewpoints and different illumination conditions, and targeted because they can cause detectors to misidentify an object as a specific target class. In order to make the generated camouflages robust in the physical world, we introduce a combination of transformations to model the physical phenomena. In addition, to improve the attacks, DPA simultaneously attacks all the classifications in the fixed proposals. Moreover, we build a virtual 3D scene using the Unity simulation engine to fairly and reproducibly evaluate different physical attacks. Extensive experiments demonstrate that DPA outperforms the state-of-the-art methods, and it is generic for any object and generalized well to the real world, posing a potential threat to the security-critical computer vision systems.

摘要: 对抗性攻击在现实世界中用于目标检测是可行的。然而，以前的大多数工作都试图学习应用于对象的局部“补丁”来愚弄检测器，这在斜视视角下变得不那么有效。为了解决这个问题，我们提出了密集建议攻击(DPA)，以学习检测器的整体、物理和有针对性的对抗伪装。伪装是一体式的，因为它们是为对象整体生成的；物理伪装是因为当在任意视点和不同的照明条件下拍摄时它们仍然是对抗性的；以及目标伪装是因为它们可能导致检测器将对象错误地识别为特定的目标类别。为了使生成的伪装在物理世界中具有鲁棒性，我们引入了一种组合变换来模拟物理现象。此外，为了改进攻击，DPA同时攻击固定方案中的所有分类。此外，我们使用Unity仿真引擎构建了一个虚拟的3D场景，以公平、可重复性地评估不同的物理攻击。大量的实验表明，DPA的性能优于目前最先进的方法，它对任何对象都是通用的，并且可以很好地推广到现实世界，这对安全关键的计算机视觉系统构成了潜在的威胁。



## **42. Adversarial attacks on voter model dynamics in complex networks**

复杂网络中选民模型动态的对抗性攻击 physics.soc-ph

6 pages, 4 figures

**SubmitDate**: 2021-11-18    [paper-pdf](http://arxiv.org/pdf/2111.09561v1)

**Authors**: Katsumi Chiyomaru, Kazuhiro Takemoto

**Abstracts**: This study investigates adversarial attacks conducted to distort the voter model dynamics in complex networks. Specifically, a simple adversarial attack method is proposed for holding the state of an individual's opinions closer to the target state in the voter model dynamics; the method shows that even when one opinion is the majority, the vote outcome can be inverted (i.e., the outcome can lean toward the other opinion) by adding extremely small (hard-to-detect) perturbations strategically generated in social networks. Adversarial attacks are relatively more effective for complex (large and dense) networks. The results indicate that opinion dynamics can be unknowingly distorted.

摘要: 这项研究调查了复杂网络中为扭曲选民模型动态而进行的对抗性攻击。具体地说，提出了一种简单的对抗性攻击方法，用于在选民模型动态中保持个人意见的状态更接近目标状态；该方法表明，即使当一种意见占多数时，投票结果也可以通过添加在社会网络中策略性地产生的极小(难以检测)的扰动来反转(即，结果可以倾向于另一种意见)。对抗性攻击对于复杂(大型和密集)网络相对更有效。结果表明，意见动态可能在不知不觉中被扭曲。



## **43. ZeBRA: Precisely Destroying Neural Networks with Zero-Data Based Repeated Bit Flip Attack**

Zebra：基于零数据重复位翻转攻击精确摧毁神经网络 cs.LG

14 pages, 3 figures, 5 tables, Accepted at British Machine Vision  Conference (BMVC) 2021

**SubmitDate**: 2021-11-18    [paper-pdf](http://arxiv.org/pdf/2111.01080v2)

**Authors**: Dahoon Park, Kon-Woo Kwon, Sunghoon Im, Jaeha Kung

**Abstracts**: In this paper, we present Zero-data Based Repeated bit flip Attack (ZeBRA) that precisely destroys deep neural networks (DNNs) by synthesizing its own attack datasets. Many prior works on adversarial weight attack require not only the weight parameters, but also the training or test dataset in searching vulnerable bits to be attacked. We propose to synthesize the attack dataset, named distilled target data, by utilizing the statistics of batch normalization layers in the victim DNN model. Equipped with the distilled target data, our ZeBRA algorithm can search vulnerable bits in the model without accessing training or test dataset. Thus, our approach makes the adversarial weight attack more fatal to the security of DNNs. Our experimental results show that 2.0x (CIFAR-10) and 1.6x (ImageNet) less number of bit flips are required on average to destroy DNNs compared to the previous attack method. Our code is available at https://github. com/pdh930105/ZeBRA.

摘要: 本文提出了一种基于零数据的重复位翻转攻击(Zebra)，它通过合成自己的攻击数据集来精确地破坏深度神经网络(DNNs)。以往许多关于对抗性权重攻击的工作不仅需要权重参数，还需要训练或测试数据集来搜索易受攻击的部位。我们提出利用受害者DNN模型中的批归一化层的统计信息来合成攻击数据集，称为提取的目标数据。有了提取的目标数据，我们的斑马算法可以在不访问训练或测试数据集的情况下搜索模型中的易受攻击的位。因此，我们的方法使得敌意加权攻击对DNNs的安全性更加致命。我们的实验结果表明，与以前的攻击方法相比，破坏DNN平均需要减少2.0倍(CIFAR-10)和1.6倍(ImageNet)的比特翻转次数。我们的代码可在https://github.获得com/pdh930105/zebra。



## **44. Finding Optimal Tangent Points for Reducing Distortions of Hard-label Attacks**

寻找最优切点以减少硬标签攻击的失真 cs.CV

accepted at NeurIPS 2021, including the appendix

**SubmitDate**: 2021-11-18    [paper-pdf](http://arxiv.org/pdf/2111.07492v2)

**Authors**: Chen Ma, Xiangyu Guo, Li Chen, Jun-Hai Yong, Yisen Wang

**Abstracts**: One major problem in black-box adversarial attacks is the high query complexity in the hard-label attack setting, where only the top-1 predicted label is available. In this paper, we propose a novel geometric-based approach called Tangent Attack (TA), which identifies an optimal tangent point of a virtual hemisphere located on the decision boundary to reduce the distortion of the attack. Assuming the decision boundary is locally flat, we theoretically prove that the minimum $\ell_2$ distortion can be obtained by reaching the decision boundary along the tangent line passing through such tangent point in each iteration. To improve the robustness of our method, we further propose a generalized method which replaces the hemisphere with a semi-ellipsoid to adapt to curved decision boundaries. Our approach is free of hyperparameters and pre-training. Extensive experiments conducted on the ImageNet and CIFAR-10 datasets demonstrate that our approach can consume only a small number of queries to achieve the low-magnitude distortion. The implementation source code is released online at https://github.com/machanic/TangentAttack.

摘要: 黑盒对抗性攻击的一个主要问题是硬标签攻击设置中的高查询复杂度，在硬标签攻击设置中，只有前1个预测标签可用。本文提出了一种新的基于几何的切线攻击方法(TA)，该方法识别位于决策边界上的虚拟半球的最佳切点，以减少攻击的失真。假设决策边界是局部平坦的，我们从理论上证明了在每一次迭代中，沿着通过该切点的切线到达决策边界可以获得最小的$\\ell2$失真。为了提高方法的鲁棒性，我们进一步提出了一种广义方法，用半椭球代替半球，以适应弯曲的决策边界。我们的方法没有超参数和预训练。在ImageNet和CIFAR-10数据集上进行的大量实验表明，我们的方法可以只消耗少量的查询来实现低幅度的失真。实现源代码在https://github.com/machanic/TangentAttack.上在线发布



## **45. Attacking Deep Learning AI Hardware with Universal Adversarial Perturbation**

利用普遍对抗性扰动攻击深度学习人工智能硬件 cs.CR

**SubmitDate**: 2021-11-18    [paper-pdf](http://arxiv.org/pdf/2111.09488v1)

**Authors**: Mehdi Sadi, B. M. S. Bahar Talukder, Kaniz Mishty, Md Tauhidur Rahman

**Abstracts**: Universal Adversarial Perturbations are image-agnostic and model-independent noise that when added with any image can mislead the trained Deep Convolutional Neural Networks into the wrong prediction. Since these Universal Adversarial Perturbations can seriously jeopardize the security and integrity of practical Deep Learning applications, existing techniques use additional neural networks to detect the existence of these noises at the input image source. In this paper, we demonstrate an attack strategy that when activated by rogue means (e.g., malware, trojan) can bypass these existing countermeasures by augmenting the adversarial noise at the AI hardware accelerator stage. We demonstrate the accelerator-level universal adversarial noise attack on several deep Learning models using co-simulation of the software kernel of Conv2D function and the Verilog RTL model of the hardware under the FuseSoC environment.

摘要: 普遍的对抗性扰动是图像不可知和模型无关的噪声，当加入任何图像时，都会将训练好的深卷积神经网络误导到错误的预测中。由于这些普遍的对抗性扰动会严重危害实际深度学习应用的安全性和完整性，现有技术使用附加的神经网络来检测输入图像源处是否存在这些噪声。在本文中，我们展示了一种攻击策略，当被流氓手段(如恶意软件、特洛伊木马)激活时，可以通过在人工智能硬件加速器阶段增加对抗性噪声来绕过这些现有的对策。在FuseSoC环境下，通过Conv2D函数的软件内核和硬件的Verilog RTL模型的联合仿真，演示了几种深度学习模型上的加速级通用对抗噪声攻击。



## **46. Cortical Features for Defense Against Adversarial Audio Attacks**

防御敌意音频攻击的皮层特征 cs.SD

Co-author legal name changed

**SubmitDate**: 2021-11-17    [paper-pdf](http://arxiv.org/pdf/2102.00313v2)

**Authors**: Ilya Kavalerov, Ruijie Zheng, Wojciech Czaja, Rama Chellappa

**Abstracts**: We propose using a computational model of the auditory cortex as a defense against adversarial attacks on audio. We apply several white-box iterative optimization-based adversarial attacks to an implementation of Amazon Alexa's HW network, and a modified version of this network with an integrated cortical representation, and show that the cortical features help defend against universal adversarial examples. At the same level of distortion, the adversarial noises found for the cortical network are always less effective for universal audio attacks. We make our code publicly available at https://github.com/ilyakava/py3fst.

摘要: 我们建议使用听觉皮层的计算模型来防御对音频的敌意攻击。我们将几个基于白盒迭代优化的敌意攻击应用到Amazon Alexa的硬件网络的一个实现中，以及该网络的一个带有集成皮层表示的修改版本，并显示了皮层特征有助于防御通用的敌意示例。在相同的失真水平下，皮层网络中发现的对抗性噪声对于通用音频攻击总是不太有效。我们在https://github.com/ilyakava/py3fst.上公开了我们的代码



## **47. Address Behaviour Vulnerabilities in the Next Generation of Autonomous Robots**

解决下一代自主机器人的行为漏洞 cs.RO

preprint and extended version of Nature Machine Intelligence, Vol 3,  November 2021, Pag 927-928

**SubmitDate**: 2021-11-17    [paper-pdf](http://arxiv.org/pdf/2103.13268v2)

**Authors**: Michele Colledanchise

**Abstracts**: Robots applications in our daily life increase at an unprecedented pace. As robots will soon operate "out in the wild", we must identify the safety and security vulnerabilities they will face. Robotics researchers and manufacturers focus their attention on new, cheaper, and more reliable applications. Still, they often disregard the operability in adversarial environments where a trusted or untrusted user can jeopardize or even alter the robot's task.   In this paper, we identify a new paradigm of security threats in the next generation of robots. These threats fall beyond the known hardware or network-based ones, and we must find new solutions to address them. These new threats include malicious use of the robot's privileged access, tampering with the robot sensors system, and tricking the robot's deliberation into harmful behaviors. We provide a taxonomy of attacks that exploit these vulnerabilities with realistic examples, and we outline effective countermeasures to prevent better, detect, and mitigate them.

摘要: 机器人在我们日常生活中的应用正以前所未有的速度增长。由于机器人即将“在野外”作业，我们必须确定它们将面临的安全和安保漏洞。机器人研究人员和制造商将他们的注意力集中在新的、更便宜的和更可靠的应用上。尽管如此，他们经常忽视在敌对环境中的可操作性，在这种环境中，可信或不可信的用户可能会危及甚至改变机器人的任务。在这篇文章中，我们确定了下一代机器人安全威胁的新范例。这些威胁超出了已知的硬件或基于网络的威胁，我们必须找到新的解决方案来应对它们。这些新的威胁包括恶意使用机器人的特权访问，篡改机器人传感器系统，以及欺骗机器人的蓄意做出有害行为。我们通过实际示例提供了利用这些漏洞的攻击分类，并概述了有效的对策以更好地预防、检测和缓解它们。



## **48. Do Not Trust Prediction Scores for Membership Inference Attacks**

不信任成员身份推断攻击的预测分数 cs.LG

15 pages, 9 figures, 9 tables

**SubmitDate**: 2021-11-17    [paper-pdf](http://arxiv.org/pdf/2111.09076v1)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Kristian Kersting

**Abstracts**: Membership inference attacks (MIAs) aim to determine whether a specific sample was used to train a predictive model. Knowing this may indeed lead to a privacy breach. Arguably, most MIAs, however, make use of the model's prediction scores - the probability of each output given some input - following the intuition that the trained model tends to behave differently on its training data. We argue that this is a fallacy for many modern deep network architectures, e.g., ReLU type neural networks produce almost always high prediction scores far away from the training data. Consequently, MIAs will miserably fail since this behavior leads to high false-positive rates not only on known domains but also on out-of-distribution data and implicitly acts as a defense against MIAs. Specifically, using generative adversarial networks, we are able to produce a potentially infinite number of samples falsely classified as part of the training data. In other words, the threat of MIAs is overestimated and less information is leaked than previously assumed. Moreover, there is actually a trade-off between the overconfidence of classifiers and their susceptibility to MIAs: the more classifiers know when they do not know, making low confidence predictions far away from the training data, the more they reveal the training data.

摘要: 成员关系推断攻击(MIA)的目的是确定特定样本是否用于训练预测模型。知道这一点确实可能会导致隐私被侵犯。然而，可以说，大多数MIA都利用了模型的预测分数-在给定一些输入的情况下，每个输出的概率-遵循这样的直觉，即训练的模型在其训练数据上往往表现不同。我们认为这对于许多现代深层网络结构来说是一种谬误，例如，RELU类型的神经网络在远离训练数据的地方几乎总是产生高的预测分数。因此，MIA将悲惨地失败，因为这种行为不仅在已知域上，而且在分布外的数据上都会导致高的假阳性率，并且隐含地起到了防御MIA的作用。具体地说，使用生成性对抗性网络，我们能够产生潜在的无限数量的样本，这些样本被错误地分类为训练数据的一部分。换句话说，MIA的威胁被高估了，泄露的信息比之前假设的要少。此外，分类器的过度自信和他们对MIA的敏感性之间实际上存在着权衡：分类器知道的越多，他们不知道的时候，做出远离训练数据的低置信度预测的人就越多，他们透露的训练数据就越多。



## **49. TraSw: Tracklet-Switch Adversarial Attacks against Multi-Object Tracking**

TraSw：针对多目标跟踪的Tracklet-Switch敌意攻击 cs.CV

**SubmitDate**: 2021-11-17    [paper-pdf](http://arxiv.org/pdf/2111.08954v1)

**Authors**: Delv Lin, Qi Chen, Chengyu Zhou, Kun He

**Abstracts**: Benefiting from the development of Deep Neural Networks, Multi-Object Tracking (MOT) has achieved aggressive progress. Currently, the real-time Joint-Detection-Tracking (JDT) based MOT trackers gain increasing attention and derive many excellent models. However, the robustness of JDT trackers is rarely studied, and it is challenging to attack the MOT system since its mature association algorithms are designed to be robust against errors during tracking. In this work, we analyze the weakness of JDT trackers and propose a novel adversarial attack method, called Tracklet-Switch (TraSw), against the complete tracking pipeline of MOT. Specifically, a push-pull loss and a center leaping optimization are designed to generate adversarial examples for both re-ID feature and object detection. TraSw can fool the tracker to fail to track the targets in the subsequent frames by attacking very few frames. We evaluate our method on the advanced deep trackers (i.e., FairMOT, JDE, ByteTrack) using the MOT-Challenge datasets (i.e., 2DMOT15, MOT17, and MOT20). Experiments show that TraSw can achieve a high success rate of over 95% by attacking only five frames on average for the single-target attack and a reasonably high success rate of over 80% for the multiple-target attack. The code is available at https://github.com/DerryHub/FairMOT-attack .

摘要: 得益于深度神经网络的发展，多目标跟踪(MOT)取得了突飞猛进的发展。目前，基于实时联合检测跟踪(JDT)的MOT跟踪器受到越来越多的关注，并衍生出许多优秀的模型。然而，JDT跟踪器的鲁棒性研究很少，而且由于其成熟的关联算法被设计成对跟踪过程中的错误具有鲁棒性，因此对MOT系统的攻击是具有挑战性的。在这项工作中，我们分析了JDT跟踪器的弱点，并针对MOT的完整跟踪流水线提出了一种新的对抗性攻击方法，称为Tracklet-Switch(TraSw)。具体地说，推拉损失和中心跳跃优化被设计为生成Re-ID特征和目标检测的对抗性示例。TraSw可以通过攻击极少的帧来欺骗跟踪器，使其无法跟踪后续帧中的目标。我们在先进的深度跟踪器(即FairMOT、JDE、ByteTrack)上使用MOT-Challenge2DMOT15、MOT17和MOT20数据集对我们的方法进行了评估。实验表明，TraSw对于单目标攻击平均只攻击5帧，对多目标攻击具有相当高的成功率，成功率在95%以上，而对于多目标攻击，成功率在80%以上。代码可在https://github.com/DerryHub/FairMOT-attack上获得。



## **50. Turning Your Strength against You: Detecting and Mitigating Robust and Universal Adversarial Patch Attacks**

把你的力量转向你：检测和减轻健壮的和通用的敌意补丁攻击 cs.CR

**SubmitDate**: 2021-11-17    [paper-pdf](http://arxiv.org/pdf/2108.05075v2)

**Authors**: Zitao Chen, Pritam Dash, Karthik Pattabiraman

**Abstracts**: Adversarial patch attacks against image classification deep neural networks (DNNs), which inject arbitrary distortions within a bounded region of an image, can generate adversarial perturbations that are robust (i.e., remain adversarial in physical world) and universal (i.e., remain adversarial on any input). Such attacks can lead to severe consequences in real-world DNN-based systems.   This work proposes Jujutsu, a technique to detect and mitigate robust and universal adversarial patch attacks. For detection, Jujutsu exploits the attacks' universal property - Jujutsu first locates the region of the potential adversarial patch, and then strategically transfers it to a dedicated region in a new image to determine whether it is truly malicious. For attack mitigation, Jujutsu leverages the attacks' localized nature via image inpainting to synthesize the semantic contents in the pixels that are corrupted by the attacks, and reconstruct the ``clean'' image.   We evaluate Jujutsu on four diverse datasets (ImageNet, ImageNette, CelebA and Place365), and show that Jujutsu achieves superior performance and significantly outperforms existing techniques. We find that Jujutsu can further defend against different variants of the basic attack, including 1) physical-world attack; 2) attacks that target diverse classes; 3) attacks that construct patches in different shapes and 4) adaptive attacks.

摘要: 针对图像分类深度神经网络(DNNs)的对抗性补丁攻击在图像的有界区域内注入任意失真，可以产生鲁棒的(即，在物理世界中保持对抗性)和普遍的(即，在任何输入上保持对抗性)的对抗性扰动。这类攻击可能会在现实世界中基于DNN的系统中导致严重后果。这项工作提出了Jujutsu，这是一种检测和减轻健壮的、通用的敌意补丁攻击的技术。为了进行检测，Jujutsu利用攻击的通用属性-Jujutsu首先定位潜在对手补丁的区域，然后战略性地将其传输到新图像中的专用区域，以确定它是否真的是恶意的。为了缓解攻击，Jujutsu利用攻击的局部性，通过图像修复来合成被攻击破坏的像素中的语义内容，并重建“干净”的图像。我们在四个不同的数据集(ImageNet，ImageNette，CelebA和Place365)上对Jujutsu进行了评估，结果表明Jujutsu取得了优越的性能，并且远远超过了现有的技术。我们发现Jujutsu可以进一步防御基本攻击的不同变体，包括1)物理世界攻击；2)针对不同类别的攻击；3)构造不同形状补丁的攻击；4)自适应攻击。



