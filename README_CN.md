# Latest Adversarial Attack Papers
**update at 2024-01-01 16:28:26**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Comparing the robustness of modern no-reference image- and video-quality metrics to adversarial attacks**

比较现代无参考图像和视频质量指标对敌方攻击的稳健性 cs.CV

**SubmitDate**: 2023-12-29    [abs](http://arxiv.org/abs/2310.06958v3) [paper-pdf](http://arxiv.org/pdf/2310.06958v3)

**Authors**: Anastasia Antsiferova, Khaled Abud, Aleksandr Gushchin, Ekaterina Shumitskaya, Sergey Lavrushkin, Dmitriy Vatolin

**Abstract**: Nowadays neural-network-based image- and video-quality metrics show better performance compared to traditional methods. However, they also became more vulnerable to adversarial attacks that increase metrics' scores without improving visual quality. The existing benchmarks of quality metrics compare their performance in terms of correlation with subjective quality and calculation time. However, the adversarial robustness of image-quality metrics is also an area worth researching. In this paper, we analyse modern metrics' robustness to different adversarial attacks. We adopted adversarial attacks from computer vision tasks and compared attacks' efficiency against 15 no-reference image/video-quality metrics. Some metrics showed high resistance to adversarial attacks which makes their usage in benchmarks safer than vulnerable metrics. The benchmark accepts new metrics submissions for researchers who want to make their metrics more robust to attacks or to find such metrics for their needs. Try our benchmark using pip install robustness-benchmark.

摘要: 如今，基于神经网络的图像和视频质量度量显示出比传统方法更好的性能。然而，它们也变得更容易受到对抗性攻击，这些攻击增加了指标的分数，但没有改善视觉质量。现有的质量指标基准在与主观质量和计算时间的相关性方面比较它们的表现。然而，图像质量指标的对抗稳健性也是一个值得研究的领域。在本文中，我们分析了现代度量对不同对手攻击的稳健性。我们采用了来自计算机视觉任务的对抗性攻击，并将攻击效率与15个无参考图像/视频质量指标进行了比较。一些指标表现出对敌意攻击的高度抵抗力，这使得它们在基准中的使用比易受攻击的指标更安全。该基准接受新的指标提交给希望使其指标更具抗攻击能力或找到符合其需求的此类指标的研究人员。使用pip安装健壮性基准测试我们的基准测试。



## **2. Passive Inference Attacks on Split Learning via Adversarial Regularization**

基于对抗性正则化的分裂学习被动推理攻击 cs.CR

17 pages, 20 pages

**SubmitDate**: 2023-12-29    [abs](http://arxiv.org/abs/2310.10483v2) [paper-pdf](http://arxiv.org/pdf/2310.10483v2)

**Authors**: Xiaochen Zhu, Xinjian Luo, Yuncheng Wu, Yangfan Jiang, Xiaokui Xiao, Beng Chin Ooi

**Abstract**: Split Learning (SL) has emerged as a practical and efficient alternative to traditional federated learning. While previous attempts to attack SL have often relied on overly strong assumptions or targeted easily exploitable models, we seek to develop more practical attacks. We introduce SDAR, a novel attack framework against SL with an honest-but-curious server. SDAR leverages auxiliary data and adversarial regularization to learn a decodable simulator of the client's private model, which can effectively infer the client's private features under the vanilla SL, and both features and labels under the U-shaped SL. We perform extensive experiments in both configurations to validate the effectiveness of our proposed attacks. Notably, in challenging but practical scenarios where existing passive attacks struggle to reconstruct the client's private data effectively, SDAR consistently achieves attack performance comparable to active attacks. On CIFAR-10, at the deep split level of 7, SDAR achieves private feature reconstruction with less than 0.025 mean squared error in both the vanilla and the U-shaped SL, and attains a label inference accuracy of over 98% in the U-shaped setting, while existing attacks fail to produce non-trivial results.

摘要: 分裂学习(Split Learning，SL)已成为传统联合学习的一种实用有效的替代方案。虽然以前攻击SL的尝试通常依赖于过于强烈的假设或目标明确、易于利用的模型，但我们寻求开发更实际的攻击。我们介绍了SDAR，这是一种针对SL的新型攻击框架，具有诚实但好奇的服务器。SDAR利用辅助数据和对抗性正则化学习客户私有模型的可解码模拟器，该模拟器可以有效地推断客户在香草SL下的私有特征，以及U形SL下的特征和标签。我们在两种配置下都进行了大量的实验，以验证我们提出的攻击的有效性。值得注意的是，在具有挑战性但实用的场景中，现有的被动攻击难以有效地重建客户端的私有数据，SDAR始终实现与主动攻击相当的攻击性能。在CIFAR-10上，在7的深度分裂水平上，SDAR实现了私有特征重建，在普通SL和U形SL上的均方误差都小于0.025，在U形背景下获得了98%以上的标签推理准确率，而现有的攻击无法产生非平凡的结果。



## **3. MVPatch: More Vivid Patch for Adversarial Camouflaged Attacks on Object Detectors in the Physical World**

MVPatch：对物理世界中的对象探测器进行对抗性伪装攻击的更生动的补丁 cs.CR

14 pages, 8 figures, submitted to IEEE Transactions on Information  Forensics & Security

**SubmitDate**: 2023-12-29    [abs](http://arxiv.org/abs/2312.17431v1) [paper-pdf](http://arxiv.org/pdf/2312.17431v1)

**Authors**: Zheng Zhou, Hongbo Zhao, Ju Liu, Qiaosheng Zhang, Guangbiao Wang, Chunlei Wang, Wenquan Feng

**Abstract**: Recent research has shown that adversarial patches can manipulate outputs from object detection models. However, the conspicuous patterns on these patches may draw more attention and raise suspicions among humans. Moreover, existing works have primarily focused on the attack performance of individual models and have neglected the generation of adversarial patches for ensemble attacks on multiple object detection models. To tackle these concerns, we propose a novel approach referred to as the More Vivid Patch (MVPatch), which aims to improve the transferability and stealthiness of adversarial patches while considering the limitations observed in prior paradigms, such as easy identification and poor transferability. Our approach incorporates an attack algorithm that decreases object confidence scores of multiple object detectors by using the ensemble attack loss function, thereby enhancing the transferability of adversarial patches. Additionally, we propose a lightweight visual similarity measurement algorithm realized by the Compared Specified Image Similarity (CSS) loss function, which allows for the generation of natural and stealthy adversarial patches without the reliance on additional generative models. Extensive experiments demonstrate that the proposed MVPatch algorithm achieves superior attack transferability compared to similar algorithms in both digital and physical domains, while also exhibiting a more natural appearance. These findings emphasize the remarkable stealthiness and transferability of the proposed MVPatch attack algorithm.

摘要: 最近的研究表明，敌意补丁可以操纵目标检测模型的输出。然而，这些斑块上的明显图案可能会引起更多的关注，并在人类中引起怀疑。此外，现有的工作主要集中在单个模型的攻击性能上，而忽略了针对多个目标检测模型的集成攻击的对抗性补丁的生成。为了解决这些问题，我们提出了一种新的方法，称为更生动的补丁(MVPatch)，其目的是提高对抗性补丁的可转移性和隐蔽性，同时考虑到以前的范例中观察到的局限性，如容易识别和较差的可转移性。该方法结合了一种攻击算法，通过使用集成攻击损失函数来降低多个目标检测器的目标置信度，从而增强了对抗性补丁的可转移性。此外，我们还提出了一种通过比较指定图像相似度损失函数实现的轻量级视觉相似性度量算法，该算法允许在不依赖于额外的生成模型的情况下生成自然的和隐蔽的对抗性补丁。大量实验表明，与同类算法相比，提出的MVPatch算法在数字和物理领域都具有更好的攻击可转移性，同时也表现出更自然的外观。这些发现强调了所提出的MVPatch攻击算法的显著的隐蔽性和可转移性。



## **4. Can you See me? On the Visibility of NOPs against Android Malware Detectors**

你能看清我吗？关于NOPS对Android恶意软件检测器的可见性 cs.CR

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.17356v1) [paper-pdf](http://arxiv.org/pdf/2312.17356v1)

**Authors**: Diego Soi, Davide Maiorca, Giorgio Giacinto, Harel Berger

**Abstract**: Android malware still represents the most significant threat to mobile systems. While Machine Learning systems are increasingly used to identify these threats, past studies have revealed that attackers can bypass these detection mechanisms by making subtle changes to Android applications, such as adding specific API calls. These modifications are often referred to as No OPerations (NOP), which ideally should not alter the semantics of the program. However, many NOPs can be spotted and eliminated by refining the app analysis process. This paper proposes a visibility metric that assesses the difficulty in spotting NOPs and similar non-operational codes. We tested our metric on a state-of-the-art, opcode-based deep learning system for Android malware detection. We implemented attacks on the feature and problem spaces and calculated their visibility according to our metric. The attained results show an intriguing trade-off between evasion efficacy and detectability: our metric can be valuable to ensure the real effectiveness of an adversarial attack, also serving as a useful aid to develop better defenses.

摘要: Android恶意软件仍然是移动系统面临的最大威胁。虽然机器学习系统越来越多地被用来识别这些威胁，但过去的研究表明，攻击者可以通过对Android应用程序进行微妙的更改，例如添加特定的API调用，绕过这些检测机制。这些修改通常被称为无操作(NOP)，理想情况下不应该改变程序的语义。然而，通过改进应用程序分析流程，可以发现并消除许多NOP。本文提出了一种可见性度量来评估发现NOP和类似的非操作代码的难度。我们在一个最先进的、基于操作码的深度学习系统上测试了我们的指标，以检测Android恶意软件。我们对功能和问题空间实施了攻击，并根据我们的度量计算了它们的可见性。所获得的结果显示了逃避有效性和可检测性之间的有趣的权衡：我们的度量对于确保对抗性攻击的真正有效性是有价值的，也可以作为开发更好防御的有用辅助。



## **5. Timeliness: A New Design Metric and a New Attack Surface**

时效性：一种新的设计尺度和新的攻击面 cs.IT

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.17220v1) [paper-pdf](http://arxiv.org/pdf/2312.17220v1)

**Authors**: Priyanka Kaswan, Sennur Ulukus

**Abstract**: As the landscape of time-sensitive applications gains prominence in 5G/6G communications, timeliness of information updates at network nodes has become crucial, which is popularly quantified in the literature by the age of information metric. However, as we devise policies to improve age of information of our systems, we inadvertently introduce a new vulnerability for adversaries to exploit. In this article, we comprehensively discuss the diverse threats that age-based systems are vulnerable to. We begin with discussion on densely interconnected networks that employ gossiping between nodes to expedite dissemination of dynamic information in the network, and show how the age-based nature of gossiping renders these networks uniquely susceptible to threats such as timestomping attacks, jamming attacks, and the propagation of misinformation. Later, we survey adversarial works within simpler network settings, specifically in one-hop and two-hop configurations, and delve into adversarial robustness concerning challenges posed by jamming, timestomping, and issues related to privacy leakage. We conclude this article with future directions that aim to address challenges posed by more intelligent adversaries and robustness of networks to them.

摘要: 随着时间敏感型应用在5G/6G通信中的日益突出，网络节点信息更新的及时性变得至关重要，文献中普遍使用信息度量时代来量化这一点。然而，当我们设计策略来改进我们系统的信息时代时，我们无意中引入了一个新的漏洞，供对手利用。在本文中，我们将全面讨论基于年龄的系统容易受到的各种威胁。我们首先讨论密集互联网络，这些网络使用节点之间的八卦来加快网络中动态信息的传播，并展示八卦的基于年龄的性质如何使这些网络唯一地容易受到威胁，如时间戳攻击、干扰攻击和错误信息的传播。随后，我们考察了简单网络环境中的敌意工作，特别是在单跳和两跳配置中，并深入研究了与干扰、时间限制和隐私泄露相关的问题带来的挑战的敌手健壮性。我们总结了本文的未来方向，旨在应对更智能的对手带来的挑战和网络对他们的健壮性。



## **6. Explainability-Based Adversarial Attack on Graphs Through Edge Perturbation**

基于可解释性的图的边扰动敌意攻击 cs.CR

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.17301v1) [paper-pdf](http://arxiv.org/pdf/2312.17301v1)

**Authors**: Dibaloke Chanda, Saba Heidari Gheshlaghi, Nasim Yahya Soltani

**Abstract**: Despite the success of graph neural networks (GNNs) in various domains, they exhibit susceptibility to adversarial attacks. Understanding these vulnerabilities is crucial for developing robust and secure applications. In this paper, we investigate the impact of test time adversarial attacks through edge perturbations which involve both edge insertions and deletions. A novel explainability-based method is proposed to identify important nodes in the graph and perform edge perturbation between these nodes. The proposed method is tested for node classification with three different architectures and datasets. The results suggest that introducing edges between nodes of different classes has higher impact as compared to removing edges among nodes within the same class.

摘要: 尽管图神经网络(GNN)在各个领域都取得了成功，但它们表现出对对手攻击的敏感性。了解这些漏洞对于开发健壮、安全的应用程序至关重要。本文研究了边扰动对测试时间敌意攻击的影响，其中边扰动涉及边的插入和删除。提出了一种新的基于可解释性的方法来识别图中的重要节点，并对这些节点之间的边进行扰动。用三种不同的体系结构和数据集对该方法进行了节点分类测试。结果表明，在不同类别的节点之间引入边比在同一类内的节点之间删除边具有更高的影响。



## **7. On the Robustness of Decision-Focused Learning**

决策学习的鲁棒性研究 cs.LG

17 pages, 45 figures, submitted to AAAI artificial intelligence for  operations research workshop

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2311.16487v3) [paper-pdf](http://arxiv.org/pdf/2311.16487v3)

**Authors**: Yehya Farhat

**Abstract**: Decision-Focused Learning (DFL) is an emerging learning paradigm that tackles the task of training a machine learning (ML) model to predict missing parameters of an incomplete optimization problem, where the missing parameters are predicted. DFL trains an ML model in an end-to-end system, by integrating the prediction and optimization tasks, providing better alignment of the training and testing objectives. DFL has shown a lot of promise and holds the capacity to revolutionize decision-making in many real-world applications. However, very little is known about the performance of these models under adversarial attacks. We adopt ten unique DFL methods and benchmark their performance under two distinctly focused attacks adapted towards the Predict-then-Optimize problem setting. Our study proposes the hypothesis that the robustness of a model is highly correlated with its ability to find predictions that lead to optimal decisions without deviating from the ground-truth label. Furthermore, we provide insight into how to target the models that violate this condition and show how these models respond differently depending on the achieved optimality at the end of their training cycles.

摘要: 决策聚焦学习（DFL）是一种新兴的学习范式，它处理训练机器学习（ML）模型来预测不完全优化问题的缺失参数的任务，其中缺失参数被预测。DFL通过集成预测和优化任务，在端到端系统中训练ML模型，从而更好地调整训练和测试目标。DFL已经显示出很大的潜力，并有能力在许多现实世界的应用中彻底改变决策。然而，人们对这些模型在对抗性攻击下的性能知之甚少。我们采用了10个独特的DFL方法，并在两个不同的攻击下对它们的性能进行了基准测试，这些攻击都是针对预测然后优化问题设置的。我们的研究提出了一个假设，即模型的鲁棒性与其在不偏离地面事实标签的情况下找到导致最佳决策的预测的能力高度相关。此外，我们还提供了如何针对违反此条件的模型的见解，并展示了这些模型如何根据其训练周期结束时达到的最优性做出不同的响应。



## **8. BlackboxBench: A Comprehensive Benchmark of Black-box Adversarial Attacks**

BlackboxBtch：黑盒对抗性攻击的综合基准 cs.CR

37 pages, 29 figures

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.16979v1) [paper-pdf](http://arxiv.org/pdf/2312.16979v1)

**Authors**: Meixi Zheng, Xuanchen Yan, Zihao Zhu, Hongrui Chen, Baoyuan Wu

**Abstract**: Adversarial examples are well-known tools to evaluate the vulnerability of deep neural networks (DNNs). Although lots of adversarial attack algorithms have been developed, it is still challenging in the practical scenario that the model's parameters and architectures are inaccessible to the attacker/evaluator, i.e., black-box adversarial attacks. Due to the practical importance, there has been rapid progress from recent algorithms, reflected by the quick increase in attack success rate and the quick decrease in query numbers to the target model. However, there is a lack of thorough evaluations and comparisons among these algorithms, causing difficulties of tracking the real progress, analyzing advantages and disadvantages of different technical routes, as well as designing future development roadmap of this field. Thus, in this work, we aim at building a comprehensive benchmark of black-box adversarial attacks, called BlackboxBench. It mainly provides: 1) a unified, extensible and modular-based codebase, implementing 25 query-based attack algorithms and 30 transfer-based attack algorithms; 2) comprehensive evaluations: we evaluate the implemented algorithms against several mainstreaming model architectures on 2 widely used datasets (CIFAR-10 and a subset of ImageNet), leading to 14,106 evaluations in total; 3) thorough analysis and new insights, as well analytical tools. The website and source codes of BlackboxBench are available at https://blackboxbench.github.io/ and https://github.com/SCLBD/BlackboxBench/, respectively.

摘要: 对抗性示例是评估深度神经网络（DNN）脆弱性的众所周知的工具。虽然已经开发了许多对抗性攻击算法，但在实际场景中仍然具有挑战性，即攻击者/评估者无法访问模型的参数和架构，即，黑盒对抗攻击由于其重要的实际意义，近年来的算法取得了快速的进展，表现为攻击成功率的快速提高和对目标模型的查询数量的快速减少。然而，这些算法之间缺乏深入的评价和比较，难以跟踪实际进展，分析不同技术路线的优缺点，以及设计该领域的未来发展路线。因此，在这项工作中，我们的目标是建立一个全面的黑盒对抗性攻击基准，称为BlackboxBench。它主要提供：1）一个统一的、可扩展的、基于模块的代码库，实现了25种基于查询的攻击算法和30种基于传输的攻击算法; 2）综合评估：我们在两个广泛使用的数据集上针对几种主流模型架构对实现的算法进行了评估（CIFAR-10和ImageNet的子集），总共产生了14，106个评估; 3）全面的分析和新的见解，以及分析工具。BlackboxBench的网站和源代码分别位于https://blackboxbench.github.io/和https：//github.com/SCLBD/BlackboxBench/。



## **9. Attack Tree Analysis for Adversarial Evasion Attacks**

对抗性逃避攻击的攻击树分析 cs.CR

10 pages

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.16957v1) [paper-pdf](http://arxiv.org/pdf/2312.16957v1)

**Authors**: Yuki Yamaguchi, Toshiaki Aoki

**Abstract**: Recently, the evolution of deep learning has promoted the application of machine learning (ML) to various systems. However, there are ML systems, such as autonomous vehicles, that cause critical damage when they misclassify. Conversely, there are ML-specific attacks called adversarial attacks based on the characteristics of ML systems. For example, one type of adversarial attack is an evasion attack, which uses minute perturbations called "adversarial examples" to intentionally misclassify classifiers. Therefore, it is necessary to analyze the risk of ML-specific attacks in introducing ML base systems. In this study, we propose a quantitative evaluation method for analyzing the risk of evasion attacks using attack trees. The proposed method consists of the extension of the conventional attack tree to analyze evasion attacks and the systematic construction method of the extension. In the extension of the conventional attack tree, we introduce ML and conventional attack nodes to represent various characteristics of evasion attacks. In the systematic construction process, we propose a procedure to construct the attack tree. The procedure consists of three steps: (1) organizing information about attack methods in the literature to a matrix, (2) identifying evasion attack scenarios from methods in the matrix, and (3) constructing the attack tree from the identified scenarios using a pattern. Finally, we conducted experiments on three ML image recognition systems to demonstrate the versatility and effectiveness of our proposed method.

摘要: 近年来，深度学习的发展促进了机器学习在各种系统中的应用。然而，有些ML系统，如自动驾驶汽车，在错误分类时会造成严重损害。相反，根据ML系统的特点，还有一些特定于ML的攻击，称为对抗性攻击。例如，一种类型的对抗性攻击是逃避攻击，它使用被称为“对抗性示例”的微小扰动来故意对分类器进行错误分类。因此，在引入ML基础系统时，有必要对特定ML攻击的风险进行分析。在这项研究中，我们提出了一种利用攻击树分析逃避攻击风险的定量评估方法。该方法包括分析规避攻击的常规攻击树的扩展和扩展的系统构建方法。在对常规攻击树的扩展中，引入ML和常规攻击节点来表示规避攻击的各种特征。在系统构建过程中，我们提出了一种构建攻击树的步骤。该过程包括三个步骤：(1)将文献中关于攻击方法的信息组织到一个矩阵中；(2)从矩阵中的方法中识别逃避攻击场景；(3)使用模式从识别的场景中构建攻击树。最后，我们在三个ML图像识别系统上进行了实验，验证了该方法的通用性和有效性。



## **10. DOEPatch: Dynamically Optimized Ensemble Model for Adversarial Patches Generation**

DOEPatch：动态优化的对抗性补丁生成集成模型 cs.CV

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.16907v1) [paper-pdf](http://arxiv.org/pdf/2312.16907v1)

**Authors**: Wenyi Tan, Yang Li, Chenxing Zhao, Zhunga Liu, Quan Pan

**Abstract**: Object detection is a fundamental task in various applications ranging from autonomous driving to intelligent security systems. However, recognition of a person can be hindered when their clothing is decorated with carefully designed graffiti patterns, leading to the failure of object detection. To achieve greater attack potential against unknown black-box models, adversarial patches capable of affecting the outputs of multiple-object detection models are required. While ensemble models have proven effective, current research in the field of object detection typically focuses on the simple fusion of the outputs of all models, with limited attention being given to developing general adversarial patches that can function effectively in the physical world. In this paper, we introduce the concept of energy and treat the adversarial patches generation process as an optimization of the adversarial patches to minimize the total energy of the ``person'' category. Additionally, by adopting adversarial training, we construct a dynamically optimized ensemble model. During training, the weight parameters of the attacked target models are adjusted to find the balance point at which the generated adversarial patches can effectively attack all target models. We carried out six sets of comparative experiments and tested our algorithm on five mainstream object detection models. The adversarial patches generated by our algorithm can reduce the recognition accuracy of YOLOv2 and YOLOv3 to 13.19\% and 29.20\%, respectively. In addition, we conducted experiments to test the effectiveness of T-shirts covered with our adversarial patches in the physical world and could achieve that people are not recognized by the object detection model. Finally, leveraging the Grad-CAM tool, we explored the attack mechanism of adversarial patches from an energetic perspective.

摘要: 目标检测是从自动驾驶到智能安防系统等各种应用中的一项基本任务。然而，当一个人的衣服上装饰着精心设计的涂鸦图案时，对他的识别可能会受到阻碍，导致物体检测失败。为了对未知黑盒模型实现更大的攻击潜力，需要能够影响多目标检测模型输出的对抗性补丁。虽然集成模型已被证明是有效的，但目前在目标检测领域的研究通常集中在所有模型的输出的简单融合上，而对开发能够在物理世界中有效发挥作用的通用对抗性补丁的关注有限。本文引入能量的概念，将对抗性补丁的生成过程看作是对敌对性补丁的优化，以最小化“人”范畴的总能量。此外，通过采用对抗性训练，构建了一个动态优化的集成模型。在训练过程中，调整被攻击目标模型的权重参数，以找到生成的敌意补丁能够有效攻击所有目标模型的平衡点。我们进行了六组对比实验，并在五个主流的目标检测模型上测试了我们的算法。该算法生成的敌意块使YOLOv2和YOLOv3的识别正确率分别降低到13.19和29.20。此外，我们还进行了实验，测试了T恤在现实世界中被我们的对手补丁覆盖的有效性，并可以实现目标检测模型无法识别人。最后，利用Grad-CAM工具，从能量的角度探讨了对抗性补丁的攻击机制。



## **11. Adversarial Attacks on Image Classification Models: Analysis and Defense**

对图像分类模型的敌意攻击：分析与防御 cs.CV

This is the accepted version of the paper presented at the 10th  International Conference on Business Analytics and Intelligence (ICBAI'24).  The conference was organized by the Indian Institute of Science, Bangalore,  India, from December 18 - 20, 2023. The paper is 10 pages long and it  contains 14 tables and 11 figures

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.16880v1) [paper-pdf](http://arxiv.org/pdf/2312.16880v1)

**Authors**: Jaydip Sen, Abhiraj Sen, Ananda Chatterjee

**Abstract**: The notion of adversarial attacks on image classification models based on convolutional neural networks (CNN) is introduced in this work. To classify images, deep learning models called CNNs are frequently used. However, when the networks are subject to adversarial attacks, extremely potent and previously trained CNN models that perform quite effectively on image datasets for image classification tasks may perform poorly. In this work, one well-known adversarial attack known as the fast gradient sign method (FGSM) is explored and its adverse effects on the performances of image classification models are examined. The FGSM attack is simulated on three pre-trained image classifier CNN architectures, ResNet-101, AlexNet, and RegNetY 400MF using randomly chosen images from the ImageNet dataset. The classification accuracies of the models are computed in the absence and presence of the attack to demonstrate the detrimental effect of the attack on the performances of the classifiers. Finally, a mechanism is proposed to defend against the FGSM attack based on a modified defensive distillation-based approach. Extensive results are presented for the validation of the proposed scheme.

摘要: 提出了对基于卷积神经网络的图像分类模型进行敌意攻击的概念。为了对图像进行分类，人们经常使用称为CNN的深度学习模型。然而，当网络受到敌意攻击时，在图像数据集上非常有效地执行图像分类任务的极其强大的和先前训练的CNN模型可能会表现得很差。本文研究了一种著名的对抗性攻击--快速梯度符号方法(FGSM)，并考察了它对图像分类模型性能的不利影响。使用从ImageNet数据集中随机选择的图像，在三种预先训练的图像分类器CNN架构上模拟了FGSM攻击，即ResNet-101、AlexNet和RegNetY 400MF。在没有攻击和存在攻击的情况下，计算了模型的分类精度，以说明攻击对分类器性能的不利影响。最后，提出了一种基于改进的防御蒸馏方法的FGSM攻击防御机制。为验证该方案的有效性，给出了大量的结果。



## **12. Adv-Diffusion: Imperceptible Adversarial Face Identity Attack via Latent Diffusion Model**

Adv-Diffusion：基于潜在扩散模型的不可感知对抗人脸身份攻击 cs.CV

Accepted by AAAI 2024

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.11285v2) [paper-pdf](http://arxiv.org/pdf/2312.11285v2)

**Authors**: Decheng Liu, Xijun Wang, Chunlei Peng, Nannan Wang, Ruiming Hu, Xinbo Gao

**Abstract**: Adversarial attacks involve adding perturbations to the source image to cause misclassification by the target model, which demonstrates the potential of attacking face recognition models. Existing adversarial face image generation methods still can't achieve satisfactory performance because of low transferability and high detectability. In this paper, we propose a unified framework Adv-Diffusion that can generate imperceptible adversarial identity perturbations in the latent space but not the raw pixel space, which utilizes strong inpainting capabilities of the latent diffusion model to generate realistic adversarial images. Specifically, we propose the identity-sensitive conditioned diffusion generative model to generate semantic perturbations in the surroundings. The designed adaptive strength-based adversarial perturbation algorithm can ensure both attack transferability and stealthiness. Extensive qualitative and quantitative experiments on the public FFHQ and CelebA-HQ datasets prove the proposed method achieves superior performance compared with the state-of-the-art methods without an extra generative model training process. The source code is available at https://github.com/kopper-xdu/Adv-Diffusion.

摘要: 对抗性攻击包括在源图像中添加扰动以导致目标模型的错误分类，这表明了攻击人脸识别模型的可能性。现有的对抗性人脸图像生成方法由于可移植性低、可检测性高，仍不能达到令人满意的效果。在本文中，我们提出了一个统一的框架，它可以在潜在空间而不是原始像素空间产生不可察觉的敌意身份扰动，该框架利用潜在扩散模型强大的修复能力来生成逼真的对抗性图像。具体地说，我们提出了身份敏感的条件扩散生成模型来产生环境中的语义扰动。所设计的基于强度的自适应对抗性扰动算法既能保证攻击的可传递性，又能保证隐蔽性。在公共FFHQ和CelebA-HQ数据集上的大量定性和定量实验证明，该方法在不需要额外的产生式模型训练过程的情况下，取得了优于最新方法的性能。源代码可在https://github.com/kopper-xdu/Adv-Diffusion.上找到



## **13. Temporal Knowledge Distillation for Time-Sensitive Financial Services Applications**

面向时间敏感金融服务应用的时态知识提取 cs.LG

arXiv admin note: text overlap with arXiv:2101.01689

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.16799v1) [paper-pdf](http://arxiv.org/pdf/2312.16799v1)

**Authors**: Hongda Shen, Eren Kurshan

**Abstract**: Detecting anomalies has become an increasingly critical function in the financial service industry. Anomaly detection is frequently used in key compliance and risk functions such as financial crime detection fraud and cybersecurity. The dynamic nature of the underlying data patterns especially in adversarial environments like fraud detection poses serious challenges to the machine learning models. Keeping up with the rapid changes by retraining the models with the latest data patterns introduces pressures in balancing the historical and current patterns while managing the training data size. Furthermore the model retraining times raise problems in time-sensitive and high-volume deployment systems where the retraining period directly impacts the models ability to respond to ongoing attacks in a timely manner. In this study we propose a temporal knowledge distillation-based label augmentation approach (TKD) which utilizes the learning from older models to rapidly boost the latest model and effectively reduces the model retraining times to achieve improved agility. Experimental results show that the proposed approach provides advantages in retraining times while improving the model performance.

摘要: 在金融服务业中，检测异常已成为一项越来越关键的功能。异常检测经常用于关键的合规和风险职能，如金融犯罪检测、欺诈和网络安全。底层数据模式的动态特性，特别是在欺诈检测等敌意环境中，对机器学习模型提出了严重的挑战。通过用最新的数据模式重新训练模型来跟上快速变化，在管理训练数据大小的同时会带来平衡历史和当前模式的压力。此外，模型再培训时间在对时间敏感的大批量部署系统中产生了问题，在这些系统中，再培训期直接影响到模型及时应对持续攻击的能力。在本研究中，我们提出了一种基于时态知识蒸馏的标签扩充方法(TKD)，该方法利用对旧模型的学习来快速提升最新模型，并有效地减少模型的重新训练次数以达到提高敏捷性的目的。实验结果表明，该方法在提高模型性能的同时，缩短了训练时间。



## **14. Multi-Task Models Adversarial Attacks**

多任务对抗性攻击模型 cs.LG

19 pages, 6 figures

**SubmitDate**: 2023-12-27    [abs](http://arxiv.org/abs/2305.12066v3) [paper-pdf](http://arxiv.org/pdf/2305.12066v3)

**Authors**: Lijun Zhang, Xiao Liu, Kaleel Mahmood, Caiwen Ding, Hui Guan

**Abstract**: Multi-Task Learning (MTL) involves developing a singular model, known as a multi-task model, to concurrently perform multiple tasks. While the security of single-task models has been thoroughly studied, multi-task models pose several critical security questions, such as 1) their vulnerability to single-task adversarial attacks, 2) the possibility of designing attacks that target multiple tasks, and 3) the impact of task sharing and adversarial training on their resilience to such attacks. This paper addresses these queries through detailed analysis and rigorous experimentation. First, we explore the adaptation of single-task white-box attacks to multi-task models and identify their limitations. We then introduce a novel attack framework, the Gradient Balancing Multi-Task Attack (GB-MTA), which treats attacking a multi-task model as an optimization problem. This problem, based on averaged relative loss change across tasks, is approximated as an integer linear programming problem. Extensive evaluations on MTL benchmarks, NYUv2 and Tiny-Taxonomy, demonstrate GB-MTA's effectiveness against both standard and adversarially trained multi-task models. The results also highlight a trade-off between task accuracy improvement via parameter sharing and increased model vulnerability due to enhanced attack transferability.

摘要: 多任务学习(MTL)涉及开发一个单一模型，称为多任务模型，以同时执行多个任务。虽然单任务模型的安全性已经得到了深入的研究，但多任务模型提出了几个关键的安全问题，如1)它们对单任务对抗性攻击的脆弱性，2)针对多任务的攻击设计的可能性，以及3)任务分担和对抗性训练对其抵抗此类攻击的影响。本文通过详细的分析和严谨的实验解决了这些问题。首先，我们探索了单任务白盒攻击对多任务模型的适应，并确定了它们的局限性。然后，我们提出了一种新的攻击框架--梯度平衡多任务攻击(GB-MTA)，该框架将攻击多任务模型视为一个优化问题。该问题基于任务间的平均相对损失变化，被近似为一个整数线性规划问题。对MTL基准、NYUv2和Tiny-Taxonomy的广泛评估表明，GB-MTA相对于标准和反向训练的多任务模型都是有效的。结果还强调了通过参数共享提高任务准确性和由于增强攻击可转移性而增加模型脆弱性之间的权衡。



## **15. Adversarial Attacks on LoRa Device Identification and Rogue Signal Detection with Deep Learning**

基于深度学习的LORA设备识别和恶意信号检测的对抗性攻击 cs.CR

**SubmitDate**: 2023-12-27    [abs](http://arxiv.org/abs/2312.16715v1) [paper-pdf](http://arxiv.org/pdf/2312.16715v1)

**Authors**: Yalin E. Sagduyu, Tugba Erpek

**Abstract**: Low-Power Wide-Area Network (LPWAN) technologies, such as LoRa, have gained significant attention for their ability to enable long-range, low-power communication for Internet of Things (IoT) applications. However, the security of LoRa networks remains a major concern, particularly in scenarios where device identification and classification of legitimate and spoofed signals are crucial. This paper studies a deep learning framework to address these challenges, considering LoRa device identification and legitimate vs. rogue LoRa device classification tasks. A deep neural network (DNN), either a convolutional neural network (CNN) or feedforward neural network (FNN), is trained for each task by utilizing real experimental I/Q data for LoRa signals, while rogue signals are generated by using kernel density estimation (KDE) of received signals by rogue devices. Fast Gradient Sign Method (FGSM)-based adversarial attacks are considered for LoRa signal classification tasks using deep learning models. The impact of these attacks is assessed on the performance of two tasks, namely device identification and legitimate vs. rogue device classification, by utilizing separate or common perturbations against these signal classification tasks. Results presented in this paper quantify the level of transferability of adversarial attacks on different LoRa signal classification tasks as a major vulnerability and highlight the need to make IoT applications robust to adversarial attacks.

摘要: 低功耗广域网(LPWAN)技术，如LORA，因其能够为物联网(IoT)应用实现远距离、低功耗通信而受到广泛关注。然而，LORA网络的安全仍然是一个主要问题，特别是在设备识别以及合法和欺骗信号的分类至关重要的情况下。本文研究了一种深度学习框架来应对这些挑战，考虑了LoRa设备识别和合法与恶意LoRa设备分类任务。通过利用LORA信号的真实实验I/Q数据为每个任务训练深度神经网络(DNN)，或者卷积神经网络(CNN)或前馈神经网络(FNN)，而恶意设备通过使用接收信号的核密度估计(KDE)来生成恶意信号。针对基于深度学习的LORA信号分类任务，提出了一种基于快速梯度符号方法(FGSM)的对抗性攻击方法。通过利用针对这些信号分类任务的单独或共同的扰动来评估这些攻击对两个任务(即设备识别和合法与恶意设备分类)的性能的影响。本文的结果量化了不同LORA信号分类任务上的对抗性攻击的可转移性水平，并强调了使物联网应用程序对对抗性攻击具有健壮性的必要性。



## **16. Frauds Bargain Attack: Generating Adversarial Text Samples via Word Manipulation Process**

欺诈讨价还价攻击：通过文字处理过程生成敌意文本样本 cs.CL

21 pages, 9 tables, 3 figures

**SubmitDate**: 2023-12-27    [abs](http://arxiv.org/abs/2303.01234v2) [paper-pdf](http://arxiv.org/pdf/2303.01234v2)

**Authors**: Mingze Ni, Zhensu Sun, Wei Liu

**Abstract**: Recent research has revealed that natural language processing (NLP) models are vulnerable to adversarial examples. However, the current techniques for generating such examples rely on deterministic heuristic rules, which fail to produce optimal adversarial examples. In response, this study proposes a new method called the Fraud's Bargain Attack (FBA), which uses a randomization mechanism to expand the search space and produce high-quality adversarial examples with a higher probability of success. FBA uses the Metropolis-Hasting sampler, a type of Markov Chain Monte Carlo sampler, to improve the selection of adversarial examples from all candidates generated by a customized stochastic process called the Word Manipulation Process (WMP). The WMP method modifies individual words in a contextually-aware manner through insertion, removal, or substitution. Through extensive experiments, this study demonstrates that FBA outperforms other methods in terms of attack success rate, imperceptibility and sentence quality.

摘要: 最近的研究表明，自然语言处理(NLP)模型很容易受到敌意例子的影响。然而，目前用于生成此类实例的技术依赖于确定性启发式规则，这无法生成最优对抗性实例。对此，本研究提出了一种称为欺诈交易攻击(FBA)的新方法，该方法使用随机化机制来扩展搜索空间，生成高质量的对抗性实例，并具有更高的成功概率。FBA使用Metropolis-Hasting采样器，一种马尔可夫链蒙特卡罗采样器，以改进从由称为单词操纵过程(WMP)的定制随机过程生成的所有候选对象中选择对抗性示例。WMP方法通过插入、删除或替换，以上下文感知的方式修改单个单词。通过大量的实验，本研究证明FBA在攻击成功率、不可感知性和句子质量方面都优于其他方法。



## **17. Evaluating the security of CRYSTALS-Dilithium in the quantum random oracle model**

在量子随机预言模型中评估晶体双锂的安全性 cs.CR

21 pages

**SubmitDate**: 2023-12-27    [abs](http://arxiv.org/abs/2312.16619v1) [paper-pdf](http://arxiv.org/pdf/2312.16619v1)

**Authors**: Kelsey A. Jackson, Carl A. Miller, Daochen Wang

**Abstract**: In the wake of recent progress on quantum computing hardware, the National Institute of Standards and Technology (NIST) is standardizing cryptographic protocols that are resistant to attacks by quantum adversaries. The primary digital signature scheme that NIST has chosen is CRYSTALS-Dilithium. The hardness of this scheme is based on the hardness of three computational problems: Module Learning with Errors (MLWE), Module Short Integer Solution (MSIS), and SelfTargetMSIS. MLWE and MSIS have been well-studied and are widely believed to be secure. However, SelfTargetMSIS is novel and, though classically as hard as MSIS, its quantum hardness is unclear. In this paper, we provide the first proof of the hardness of SelfTargetMSIS via a reduction from MLWE in the Quantum Random Oracle Model (QROM). Our proof uses recently developed techniques in quantum reprogramming and rewinding. A central part of our approach is a proof that a certain hash function, derived from the MSIS problem, is collapsing. From this approach, we deduce a new security proof for Dilithium under appropriate parameter settings. Compared to the only other rigorous security proof for a variant of Dilithium, Dilithium-QROM, our proof has the advantage of being applicable under the condition q = 1 mod 2n, where q denotes the modulus and n the dimension of the underlying algebraic ring. This condition is part of the original Dilithium proposal and is crucial for the efficient implementation of the scheme. We provide new secure parameter sets for Dilithium under the condition q = 1 mod 2n, finding that our public key sizes and signature sizes are about 2.5 to 2.8 times larger than those of Dilithium-QROM for the same security levels.

摘要: 随着量子计算硬件的最新进展，美国国家标准与技术研究所（NIST）正在标准化能够抵抗量子对手攻击的加密协议。NIST选择的主要数字签名方案是CRYSTALS-Dilithium。该方案的难度基于三个计算问题的难度：带错误的模块学习（MLWE），模块短解（MSIS）和SelfTargetMSIS。MLWE和MSIS已经被充分研究，并且被广泛认为是安全的。然而，SelfTargetMSIS是新颖的，虽然在经典上与MSIS一样难，但它的量子硬度尚不清楚。在本文中，我们提供了第一个证明的硬度SelfTargetMSIS通过减少MLWE在量子随机Oracle模型（QROM）。我们的证明使用了最近开发的量子重编程和倒带技术。我们的方法的一个核心部分是证明某个哈希函数，来自MSIS问题，是崩溃。从这个方法中，我们推导出一个新的安全证明Dilithium在适当的参数设置。与Dilithium的一个变体Dilithium-QROM的唯一其他严格安全证明相比，我们的证明具有适用于条件q = 1 mod 2n的优点，其中q表示模，n表示底层代数环的维数。这一条件是原Dilithium提案的一部分，对于该计划的有效实施至关重要。在q = 1 mod 2n的条件下，我们为Dilithium提供了新的安全参数集，发现我们的公钥大小和签名大小是相同安全级别的Dilithium-QROM的2.5到2.8倍。



## **18. Natural Adversarial Patch Generation Method Based on Latent Diffusion Model**

基于潜在扩散模型的自然对抗性补丁生成方法 cs.CV

**SubmitDate**: 2023-12-27    [abs](http://arxiv.org/abs/2312.16401v1) [paper-pdf](http://arxiv.org/pdf/2312.16401v1)

**Authors**: Xianyi Chen, Fazhan Liu, Dong Jiang, Kai Yan

**Abstract**: Recently, some research show that deep neural networks are vulnerable to the adversarial attacks, the well-trainned samples or patches could be used to trick the neural network detector or human visual perception. However, these adversarial patches, with their conspicuous and unusual patterns, lack camouflage and can easily raise suspicion in the real world. To solve this problem, this paper proposed a novel adversarial patch method called the Latent Diffusion Patch (LDP), in which, a pretrained encoder is first designed to compress the natural images into a feature space with key characteristics. Then trains the diffusion model using the above feature space. Finally, explore the latent space of the pretrained diffusion model using the image denoising technology. It polishes the patches and images through the powerful natural abilities of diffusion models, making them more acceptable to the human visual system. Experimental results, both digital and physical worlds, show that LDPs achieve a visual subjectivity score of 87.3%, while still maintaining effective attack capabilities.

摘要: 最近的研究表明，深度神经网络很容易受到敌意攻击，训练好的样本或补丁可以用来欺骗神经网络检测器或人类的视觉感知。然而，这些对抗性补丁具有明显和不寻常的图案，缺乏伪装性，很容易在现实世界中引起怀疑。为了解决这一问题，本文提出了一种新的对抗性补丁方法，称为潜在扩散补丁(LDP)，该方法首先设计一个预先训练的编码器，将自然图像压缩到具有关键特征的特征空间中。然后利用上述特征空间对扩散模型进行训练。最后，利用图像去噪技术探索预训练扩散模型的潜在空间。它通过扩散模型强大的自然能力来打磨补丁和图像，使它们更容易被人类视觉系统接受。实验结果表明，LDPS在保持有效攻击能力的同时，获得了87.3%的视觉主观性分数。



## **19. SlowTrack: Increasing the Latency of Camera-based Perception in Autonomous Driving Using Adversarial Examples**

SlowTrack：使用对抗性例子增加自动驾驶中基于摄像头的感知的延迟 cs.CV

Accepted by AAAI 2024

**SubmitDate**: 2023-12-26    [abs](http://arxiv.org/abs/2312.09520v2) [paper-pdf](http://arxiv.org/pdf/2312.09520v2)

**Authors**: Chen Ma, Ningfei Wang, Qi Alfred Chen, Chao Shen

**Abstract**: In Autonomous Driving (AD), real-time perception is a critical component responsible for detecting surrounding objects to ensure safe driving. While researchers have extensively explored the integrity of AD perception due to its safety and security implications, the aspect of availability (real-time performance) or latency has received limited attention. Existing works on latency-based attack have focused mainly on object detection, i.e., a component in camera-based AD perception, overlooking the entire camera-based AD perception, which hinders them to achieve effective system-level effects, such as vehicle crashes. In this paper, we propose SlowTrack, a novel framework for generating adversarial attacks to increase the execution time of camera-based AD perception. We propose a novel two-stage attack strategy along with the three new loss function designs. Our evaluation is conducted on four popular camera-based AD perception pipelines, and the results demonstrate that SlowTrack significantly outperforms existing latency-based attacks while maintaining comparable imperceptibility levels. Furthermore, we perform the evaluation on Baidu Apollo, an industry-grade full-stack AD system, and LGSVL, a production-grade AD simulator, with two scenarios to compare the system-level effects of SlowTrack and existing attacks. Our evaluation results show that the system-level effects can be significantly improved, i.e., the vehicle crash rate of SlowTrack is around 95% on average while existing works only have around 30%.

摘要: 在自动驾驶中，实时感知是负责检测周围物体以确保安全驾驶的关键部件。虽然由于AD感知的安全性和安全性，研究人员已经对其完整性进行了广泛的探索，但可用性(实时性能)或延迟方面的关注有限。现有的基于延迟攻击的研究主要集中于目标检测，即基于摄像头的广告感知中的一个组件，而忽略了整个基于摄像头的广告感知，这阻碍了它们达到有效的系统级效果，如车辆碰撞。在本文中，我们提出了一种新的生成敌意攻击的框架SlowTrack，以增加基于摄像机的广告感知的执行时间。我们提出了一种新的两阶段攻击策略以及三种新的损失函数设计。我们在四个流行的基于摄像头的AD感知管道上进行了评估，结果表明，SlowTrack在保持相当的不可感知性水平的同时，显著优于现有的基于延迟的攻击。此外，我们在工业级全栈AD系统百度Apollo和生产级AD模拟器LGSVL上进行了评估，并通过两个场景比较了SlowTrack和现有攻击的系统级影响。我们的评估结果表明，系统级效果可以得到显著提高，即SlowTrack的车辆撞击率平均在95%左右，而现有的工作只有30%左右。



## **20. Model Stealing Attack against Recommender System**

针对推荐系统的模型窃取攻击 cs.CR

**SubmitDate**: 2023-12-26    [abs](http://arxiv.org/abs/2312.11571v2) [paper-pdf](http://arxiv.org/pdf/2312.11571v2)

**Authors**: Zhihao Zhu, Rui Fan, Chenwang Wu, Yi Yang, Defu Lian, Enhong Chen

**Abstract**: Recent studies have demonstrated the vulnerability of recommender systems to data privacy attacks. However, research on the threat to model privacy in recommender systems, such as model stealing attacks, is still in its infancy. Some adversarial attacks have achieved model stealing attacks against recommender systems, to some extent, by collecting abundant training data of the target model (target data) or making a mass of queries. In this paper, we constrain the volume of available target data and queries and utilize auxiliary data, which shares the item set with the target data, to promote model stealing attacks. Although the target model treats target and auxiliary data differently, their similar behavior patterns allow them to be fused using an attention mechanism to assist attacks. Besides, we design stealing functions to effectively extract the recommendation list obtained by querying the target model. Experimental results show that the proposed methods are applicable to most recommender systems and various scenarios and exhibit excellent attack performance on multiple datasets.

摘要: 最近的研究表明，推荐系统对数据隐私攻击是脆弱的。然而，对推荐系统中模型隐私威胁的研究，如模型窃取攻击，还处于起步阶段。一些对抗性攻击通过收集目标模型(目标数据)的大量训练数据或进行大量查询，在一定程度上实现了对推荐系统的模型窃取攻击。本文通过限制可用目标数据和查询的数据量，利用与目标数据共享项集的辅助数据来促进模型窃取攻击。尽管目标模型对目标和辅助数据的处理不同，但它们相似的行为模式允许使用注意力机制将它们融合在一起，以帮助攻击。此外，我们还设计了窃取函数来有效地提取通过查询目标模型获得的推荐列表。实验结果表明，所提出的方法适用于大多数推荐系统和各种场景，并在多个数据集上表现出良好的攻击性能。



## **21. MENLI: Robust Evaluation Metrics from Natural Language Inference**

MENLI：自然语言推理中的稳健评价指标 cs.CL

TACL 2023 Camera-ready version; updated after proofreading by the  journal

**SubmitDate**: 2023-12-26    [abs](http://arxiv.org/abs/2208.07316v5) [paper-pdf](http://arxiv.org/pdf/2208.07316v5)

**Authors**: Yanran Chen, Steffen Eger

**Abstract**: Recently proposed BERT-based evaluation metrics for text generation perform well on standard benchmarks but are vulnerable to adversarial attacks, e.g., relating to information correctness. We argue that this stems (in part) from the fact that they are models of semantic similarity. In contrast, we develop evaluation metrics based on Natural Language Inference (NLI), which we deem a more appropriate modeling. We design a preference-based adversarial attack framework and show that our NLI based metrics are much more robust to the attacks than the recent BERT-based metrics. On standard benchmarks, our NLI based metrics outperform existing summarization metrics, but perform below SOTA MT metrics. However, when combining existing metrics with our NLI metrics, we obtain both higher adversarial robustness (15%-30%) and higher quality metrics as measured on standard benchmarks (+5% to 30%).

摘要: 最近提出的基于BERT的文本生成评估指标在标准基准上表现良好，但容易受到敌意攻击，例如与信息正确性有关的攻击。我们认为，这(部分)源于这样一个事实：它们是语义相似性的模型。相比之下，我们基于自然语言推理(NLI)开发评估指标，我们认为这是更合适的建模。我们设计了一个基于偏好的对抗性攻击框架，并表明我们的基于NLI的度量比最近的基于BERT的度量具有更强的抗攻击能力。在标准基准测试中，我们基于NLI的指标优于现有的摘要指标，但低于SOTA MT指标。然而，当将现有指标与我们的NLI指标相结合时，我们获得了更高的对手健壮性(15%-30%)和标准基准测试的更高质量指标(+5%到30%)。



## **22. Punctuation Matters! Stealthy Backdoor Attack for Language Models**

标点符号很重要！对语言模型的秘密后门攻击 cs.CL

NLPCC 2023

**SubmitDate**: 2023-12-26    [abs](http://arxiv.org/abs/2312.15867v1) [paper-pdf](http://arxiv.org/pdf/2312.15867v1)

**Authors**: Xuan Sheng, Zhicheng Li, Zhaoyang Han, Xiangmao Chang, Piji Li

**Abstract**: Recent studies have pointed out that natural language processing (NLP) models are vulnerable to backdoor attacks. A backdoored model produces normal outputs on the clean samples while performing improperly on the texts with triggers that the adversary injects. However, previous studies on textual backdoor attack pay little attention to stealthiness. Moreover, some attack methods even cause grammatical issues or change the semantic meaning of the original texts. Therefore, they can easily be detected by humans or defense systems. In this paper, we propose a novel stealthy backdoor attack method against textual models, which is called \textbf{PuncAttack}. It leverages combinations of punctuation marks as the trigger and chooses proper locations strategically to replace them. Through extensive experiments, we demonstrate that the proposed method can effectively compromise multiple models in various tasks. Meanwhile, we conduct automatic evaluation and human inspection, which indicate the proposed method possesses good performance of stealthiness without bringing grammatical issues and altering the meaning of sentences.

摘要: 最近的研究指出，自然语言处理(NLP)模型容易受到后门攻击。反向模型在干净的样本上产生正常输出，而在带有对手注入的触发器的文本上执行不正确的操作。然而，以往对文本后门攻击的研究很少关注隐蔽性。此外，一些攻击方法甚至会引起语法问题或改变原文的语义。因此，它们很容易被人类或防御系统检测到。提出了一种新的针对文本模型的隐蔽后门攻击方法-.它利用标点符号的组合作为触发器，并战略性地选择适当的位置来取代它们。通过大量的实验，我们证明了该方法能够在不同的任务中有效地折衷多个模型。同时，我们进行了自动评估和人工检测，表明该方法具有良好的隐蔽性，不会带来语法问题，也不会改变句子的意义。



## **23. Adversarial Item Promotion on Visually-Aware Recommender Systems by Guided Diffusion**

基于引导扩散的视觉感知推荐系统中的对抗性项目提升 cs.IR

**SubmitDate**: 2023-12-25    [abs](http://arxiv.org/abs/2312.15826v1) [paper-pdf](http://arxiv.org/pdf/2312.15826v1)

**Authors**: Lijian Chen, Wei Yuan, Tong Chen, Quoc Viet Hung Nguyen, Lizhen Cui, Hongzhi Yin

**Abstract**: Visually-aware recommender systems have found widespread application in domains where visual elements significantly contribute to the inference of users' potential preferences. While the incorporation of visual information holds the promise of enhancing recommendation accuracy and alleviating the cold-start problem, it is essential to point out that the inclusion of item images may introduce substantial security challenges. Some existing works have shown that the item provider can manipulate item exposure rates to its advantage by constructing adversarial images. However, these works cannot reveal the real vulnerability of visually-aware recommender systems because (1) The generated adversarial images are markedly distorted, rendering them easily detectable by human observers; (2) The effectiveness of the attacks is inconsistent and even ineffective in some scenarios. To shed light on the real vulnerabilities of visually-aware recommender systems when confronted with adversarial images, this paper introduces a novel attack method, IPDGI (Item Promotion by Diffusion Generated Image). Specifically, IPDGI employs a guided diffusion model to generate adversarial samples designed to deceive visually-aware recommender systems. Taking advantage of accurately modeling benign images' distribution by diffusion models, the generated adversarial images have high fidelity with original images, ensuring the stealth of our IPDGI. To demonstrate the effectiveness of our proposed methods, we conduct extensive experiments on two commonly used e-commerce recommendation datasets (Amazon Beauty and Amazon Baby) with several typical visually-aware recommender systems. The experimental results show that our attack method has a significant improvement in both the performance of promoting the long-tailed (i.e., unpopular) items and the quality of generated adversarial images.

摘要: 视觉感知推荐系统在视觉元素对用户潜在偏好的推断有重要作用的领域得到了广泛的应用。虽然加入视觉信息有望提高推荐的准确性和缓解冷启动问题，但必须指出的是，纳入物品图像可能会带来重大的安全挑战。一些已有的研究表明，物品提供者可以通过构建对抗性图像来操纵物品曝光率。然而，这些工作并不能揭示视觉感知推荐系统的真正弱点，因为(1)生成的敌意图像明显失真，使得人类很容易发现它们；(2)攻击的有效性在某些场景下是不一致的，甚至无效的。为了揭示视觉感知推荐系统在面对敌意图像时的真正弱点，提出了一种新的攻击方法--IPDGI(Item Promotion By Diffumation Generated Image)。具体地说，IPDGI使用引导扩散模型来生成敌意样本，旨在欺骗视觉感知的推荐系统。利用扩散模型精确模拟良性图像的分布，生成的对抗性图像与原始图像具有较高的保真度，保证了IPDGI的隐蔽性。为了验证我们提出的方法的有效性，我们在两个常用的电子商务推荐数据集(Amazon Beauty和Amazon Baby)上进行了广泛的实验，并使用几个典型的视觉感知推荐系统进行了实验。实验结果表明，我们的攻击方法在提升长尾(即不受欢迎)项的性能和生成对抗性图像的质量方面都有显著的提高。



## **24. Attention Deficit is Ordered! Fooling Deformable Vision Transformers with Collaborative Adversarial Patches**

注意力缺陷是命中注定的！用协同对抗性补丁愚弄可变形视觉变形器 cs.CV

12 pages, 14 figures

**SubmitDate**: 2023-12-25    [abs](http://arxiv.org/abs/2311.12914v2) [paper-pdf](http://arxiv.org/pdf/2311.12914v2)

**Authors**: Quazi Mishkatul Alam, Bilel Tarchoun, Ihsen Alouani, Nael Abu-Ghazaleh

**Abstract**: The latest generation of transformer-based vision models has proven to be superior to Convolutional Neural Network (CNN)-based models across several vision tasks, largely attributed to their remarkable prowess in relation modeling. Deformable vision transformers significantly reduce the quadratic complexity of attention modeling by using sparse attention structures, enabling them to incorporate features across different scales and be used in large-scale applications, such as multi-view vision systems. Recent work has demonstrated adversarial attacks against conventional vision transformers; we show that these attacks do not transfer to deformable transformers due to their sparse attention structure. Specifically, attention in deformable transformers is modeled using pointers to the most relevant other tokens. In this work, we contribute for the first time adversarial attacks that manipulate the attention of deformable transformers, redirecting it to focus on irrelevant parts of the image. We also develop new collaborative attacks where a source patch manipulates attention to point to a target patch, which contains the adversarial noise to fool the model. In our experiments, we observe that altering less than 1% of the patched area in the input field results in a complete drop to 0% AP in single-view object detection using MS COCO and a 0% MODA in multi-view object detection using Wildtrack.

摘要: 最新一代的基于变压器的视觉模型已被证明在几个视觉任务上优于基于卷积神经网络(CNN)的模型，这在很大程度上归功于它们在关系建模方面的非凡能力。可变形视觉转换器通过使用稀疏注意力结构显著降低了注意力建模的二次方复杂性，使其能够合并不同尺度上的特征，并用于大规模应用，如多视角视觉系统。最近的工作证明了针对传统视觉转换器的对抗性攻击；我们表明，由于其稀疏的注意结构，这些攻击不会转移到可变形的转换器上。具体地说，在可变形转换器中的注意力是使用指向最相关的其他标记的指针来建模的。在这项工作中，我们第一次贡献了对抗性攻击，操纵变形变形者的注意力，将其重新定向到图像中不相关的部分。我们还开发了新的协作攻击，其中源补丁操纵注意力指向目标补丁，目标补丁包含敌意噪声来愚弄模型。在我们的实验中，我们观察到，在输入区域中改变不到1%的修补面积会导致在使用MS Coco的单视图目标检测中完全下降到0%AP，在使用WildTrack的多视点目标检测中完全下降到0%Moda。



## **25. Adversarial Prompt Tuning for Vision-Language Models**

视觉语言模型的对抗性提示调整 cs.CV

**SubmitDate**: 2023-12-25    [abs](http://arxiv.org/abs/2311.11261v2) [paper-pdf](http://arxiv.org/pdf/2311.11261v2)

**Authors**: Jiaming Zhang, Xingjun Ma, Xin Wang, Lingyu Qiu, Jiaqi Wang, Yu-Gang Jiang, Jitao Sang

**Abstract**: With the rapid advancement of multimodal learning, pre-trained Vision-Language Models (VLMs) such as CLIP have demonstrated remarkable capacities in bridging the gap between visual and language modalities. However, these models remain vulnerable to adversarial attacks, particularly in the image modality, presenting considerable security risks. This paper introduces Adversarial Prompt Tuning (AdvPT), a novel technique to enhance the adversarial robustness of image encoders in VLMs. AdvPT innovatively leverages learnable text prompts and aligns them with adversarial image embeddings, to address the vulnerabilities inherent in VLMs without the need for extensive parameter training or modification of the model architecture. We demonstrate that AdvPT improves resistance against white-box and black-box adversarial attacks and exhibits a synergistic effect when combined with existing image-processing-based defense techniques, further boosting defensive capabilities. Comprehensive experimental analyses provide insights into adversarial prompt tuning, a novel paradigm devoted to improving resistance to adversarial images through textual input modifications, paving the way for future robust multimodal learning research. These findings open up new possibilities for enhancing the security of VLMs. Our code is available at https://github.com/jiamingzhang94/Adversarial-Prompt-Tuning.

摘要: 随着多通道学习的快速发展，诸如CLIP等预先训练的视觉语言模型在弥合视觉和语言通道之间的差距方面显示出了显著的能力。然而，这些模型仍然容易受到敌意攻击，特别是在图像模式方面，这带来了相当大的安全风险。本文介绍了对抗性提示调优(AdvPT)技术，这是一种在VLMS中增强图像编码器对抗性稳健性的新技术。AdvPT创新性地利用可学习的文本提示，并将其与对抗性图像嵌入相结合，以解决VLM中固有的漏洞，而无需进行广泛的参数培训或修改模型体系结构。我们证明，AdvPT提高了对白盒和黑盒攻击的抵抗力，并与现有的基于图像处理的防御技术相结合，显示出协同效应，进一步增强了防御能力。全面的实验分析提供了对对抗性即时调整的见解，这是一种致力于通过修改文本输入来提高对对抗性图像的抵抗力的新范式，为未来稳健的多通道学习研究铺平了道路。这些发现为增强VLM的安全性开辟了新的可能性。我们的代码可以在https://github.com/jiamingzhang94/Adversarial-Prompt-Tuning.上找到



## **26. Vulnerability of Machine Learning Approaches Applied in IoT-based Smart Grid: A Review**

基于物联网的智能电网中机器学习方法的脆弱性：综述 cs.CR

**SubmitDate**: 2023-12-25    [abs](http://arxiv.org/abs/2308.15736v3) [paper-pdf](http://arxiv.org/pdf/2308.15736v3)

**Authors**: Zhenyong Zhang, Mengxiang Liu, Mingyang Sun, Ruilong Deng, Peng Cheng, Dusit Niyato, Mo-Yuen Chow, Jiming Chen

**Abstract**: Machine learning (ML) sees an increasing prevalence of being used in the internet-of-things (IoT)-based smart grid. However, the trustworthiness of ML is a severe issue that must be addressed to accommodate the trend of ML-based smart grid applications (MLsgAPPs). The adversarial distortion injected into the power signal will greatly affect the system's normal control and operation. Therefore, it is imperative to conduct vulnerability assessment for MLsgAPPs applied in the context of safety-critical power systems. In this paper, we provide a comprehensive review of the recent progress in designing attack and defense methods for MLsgAPPs. Unlike the traditional survey about ML security, this is the first review work about the security of MLsgAPPs that focuses on the characteristics of power systems. We first highlight the specifics for constructing the adversarial attacks on MLsgAPPs. Then, the vulnerability of MLsgAPP is analyzed from both the aspects of the power system and ML model. Afterward, a comprehensive survey is conducted to review and compare existing studies about the adversarial attacks on MLsgAPPs in scenarios of generation, transmission, distribution, and consumption, and the countermeasures are reviewed according to the attacks that they defend against. Finally, the future research directions are discussed on the attacker's and defender's side, respectively. We also analyze the potential vulnerability of large language model-based (e.g., ChatGPT) power system applications. Overall, we encourage more researchers to contribute to investigating the adversarial issues of MLsgAPPs.

摘要: 机器学习(ML)在基于物联网(IoT)的智能电网中的应用越来越普遍。然而，ML的可信性是一个必须解决的严重问题，以适应基于ML的智能电网应用(MLsgAPP)的趋势。注入到电源信号中的对抗性失真将极大地影响系统的正常控制和运行。因此，对应用于安全关键电力系统背景下的MLsgAPP进行脆弱性评估势在必行。在本文中，我们提供了一个全面的进展，设计攻击和防御方法的MLsgAPP。与传统的ML安全研究不同，本文首次针对电力系统的特点对MLsgAPP的安全问题进行了综述。我们首先强调构造对MLsgAPP的对抗性攻击的细节。然后，从电力系统和ML模型两个方面分析了MLsgAPP的脆弱性。然后，对已有的针对MLsgAPP的生成、传输、分发、消费等场景下的对抗性攻击的研究进行了全面的回顾和比较，并根据它们所防御的攻击回顾了相应的对策。最后，分别从攻击方和防御方的角度讨论了今后的研究方向。我们还分析了基于大型语言模型(如ChatGPT)的电力系统应用的潜在脆弱性。总体而言，我们鼓励更多的研究人员为研究MLsgAPP的对抗性问题做出贡献。



## **27. Privacy-Preserving Neural Graph Databases**

保护隐私的神经图库 cs.DB

**SubmitDate**: 2023-12-25    [abs](http://arxiv.org/abs/2312.15591v1) [paper-pdf](http://arxiv.org/pdf/2312.15591v1)

**Authors**: Qi Hu, Haoran Li, Jiaxin Bai, Yangqiu Song

**Abstract**: In the era of big data and rapidly evolving information systems, efficient and accurate data retrieval has become increasingly crucial. Neural graph databases (NGDBs) have emerged as a powerful paradigm that combines the strengths of graph databases (graph DBs) and neural networks to enable efficient storage, retrieval, and analysis of graph-structured data. The usage of neural embedding storage and complex neural logical query answering provides NGDBs with generalization ability. When the graph is incomplete, by extracting latent patterns and representations, neural graph databases can fill gaps in the graph structure, revealing hidden relationships and enabling accurate query answering. Nevertheless, this capability comes with inherent trade-offs, as it introduces additional privacy risks to the database. Malicious attackers can infer more sensitive information in the database using well-designed combinatorial queries, such as by comparing the answer sets of where Turing Award winners born before 1950 and after 1940 lived, the living places of Turing Award winner Hinton are probably exposed, although the living places may have been deleted in the training due to the privacy concerns. In this work, inspired by the privacy protection in graph embeddings, we propose a privacy-preserving neural graph database (P-NGDB) to alleviate the risks of privacy leakage in NGDBs. We introduce adversarial training techniques in the training stage to force the NGDBs to generate indistinguishable answers when queried with private information, enhancing the difficulty of inferring sensitive information through combinations of multiple innocuous queries. Extensive experiment results on three datasets show that P-NGDB can effectively protect private information in the graph database while delivering high-quality public answers responses to queries.

摘要: 在大数据和快速发展的信息系统的时代，高效和准确的数据检索变得越来越重要。神经图形数据库(NGDB)已经成为一种强大的范例，它结合了图形数据库(图形数据库)和神经网络的优点，使得能够有效地存储、检索和分析图形结构的数据。神经嵌入存储和复杂神经逻辑查询回答的使用为NGDB提供了泛化能力。当图不完整时，通过提取潜在模式和表示，神经图库可以填补图结构中的空白，揭示隐藏的关系，并使查询得到准确的回答。尽管如此，这种能力也伴随着固有的权衡，因为它会给数据库带来额外的隐私风险。恶意攻击者可以使用精心设计的组合查询来推断数据库中更敏感的信息，例如通过比较1950年之前出生的图灵奖获得者和1940年后出生的图灵奖获得者的答案集，图灵奖获得者辛顿的居住地可能会被曝光，尽管出于隐私考虑，在训练中可能已经删除了居住地。在这项工作中，我们受到图嵌入中隐私保护的启发，提出了一种隐私保护神经图库(P-NGDB)来缓解NGDB中隐私泄露的风险。我们在训练阶段引入对抗性训练技术，迫使NGDB在查询私有信息时产生难以区分的答案，增加了通过组合多个无害查询来推断敏感信息的难度。在三个数据集上的大量实验结果表明，P-NGDB可以有效地保护图形数据库中的私有信息，同时提供高质量的公共查询响应。



## **28. An Initial Investigation of Neural Replay Simulator for Over-the-Air Adversarial Perturbations to Automatic Speaker Verification**

用于自动说话人确认的空中对抗扰动神经重放模拟器的初步研究 cs.SD

Accepted in ICASSP 2024

**SubmitDate**: 2023-12-24    [abs](http://arxiv.org/abs/2310.05354v3) [paper-pdf](http://arxiv.org/pdf/2310.05354v3)

**Authors**: Jiaqi Li, Li Wang, Liumeng Xue, Lei Wang, Zhizheng Wu

**Abstract**: Deep Learning has advanced Automatic Speaker Verification (ASV) in the past few years. Although it is known that deep learning-based ASV systems are vulnerable to adversarial examples in digital access, there are few studies on adversarial attacks in the context of physical access, where a replay process (i.e., over the air) is involved. An over-the-air attack involves a loudspeaker, a microphone, and a replaying environment that impacts the movement of the sound wave. Our initial experiment confirms that the replay process impacts the effectiveness of the over-the-air attack performance. This study performs an initial investigation towards utilizing a neural replay simulator to improve over-the-air adversarial attack robustness. This is achieved by using a neural waveform synthesizer to simulate the replay process when estimating the adversarial perturbations. Experiments conducted on the ASVspoof2019 dataset confirm that the neural replay simulator can considerably increase the success rates of over-the-air adversarial attacks. This raises the concern for adversarial attacks on speaker verification in physical access applications.

摘要: 在过去的几年里，深度学习发展了自动说话人确认(ASV)。虽然众所周知，基于深度学习的ASV系统在数字访问中容易受到敌意攻击，但在涉及重播过程(即空中重播)的物理访问环境中，很少有关于对抗性攻击的研究。空中攻击包括扬声器、麦克风和影响声波移动的重放环境。我们的初步实验证实，重放过程会影响空中攻击性能的有效性。本研究对利用神经重放模拟器来提高空中对抗攻击的稳健性进行了初步的研究。这是通过使用神经波形合成器来模拟在估计对抗性扰动时的重播过程来实现的。在ASVspoof2019数据集上进行的实验证实，神经重放模拟器可以显著提高空中对抗性攻击的成功率。这引起了人们对物理访问应用中说话人验证的对抗性攻击的关注。



## **29. Adversarial Data Poisoning for Fake News Detection: How to Make a Model Misclassify a Target News without Modifying It**

虚假新闻检测的对抗性数据中毒：如何使模型在不修改的情况下错误分类目标新闻 cs.LG

**SubmitDate**: 2023-12-23    [abs](http://arxiv.org/abs/2312.15228v1) [paper-pdf](http://arxiv.org/pdf/2312.15228v1)

**Authors**: Federico Siciliano, Luca Maiano, Lorenzo Papa, Federica Baccin, Irene Amerini, Fabrizio Silvestri

**Abstract**: Fake news detection models are critical to countering disinformation but can be manipulated through adversarial attacks. In this position paper, we analyze how an attacker can compromise the performance of an online learning detector on specific news content without being able to manipulate the original target news. In some contexts, such as social networks, where the attacker cannot exert complete control over all the information, this scenario can indeed be quite plausible. Therefore, we show how an attacker could potentially introduce poisoning data into the training data to manipulate the behavior of an online learning method. Our initial findings reveal varying susceptibility of logistic regression models based on complexity and attack type.

摘要: 假新闻检测模型对于打击虚假信息至关重要，但可以通过对抗性攻击进行操纵。在这份立场文件中，我们分析了攻击者如何在无法操纵原始目标新闻的情况下，损害在线学习检测器对特定新闻内容的性能。在某些情况下，如社交网络，攻击者无法完全控制所有信息，这种情况确实是相当合理的。因此，我们展示了攻击者如何潜在地将中毒数据引入到训练数据中，以操纵在线学习方法的行为。我们的初步研究结果揭示了基于复杂性和攻击类型的逻辑回归模型的不同易感性。



## **30. Towards Transferable Adversarial Attacks with Centralized Perturbation**

集中式扰动下的可转移对抗性攻击 cs.CV

10 pages, 9 figures, accepted by AAAI 2024

**SubmitDate**: 2023-12-23    [abs](http://arxiv.org/abs/2312.06199v2) [paper-pdf](http://arxiv.org/pdf/2312.06199v2)

**Authors**: Shangbo Wu, Yu-an Tan, Yajie Wang, Ruinan Ma, Wencong Ma, Yuanzhang Li

**Abstract**: Adversarial transferability enables black-box attacks on unknown victim deep neural networks (DNNs), rendering attacks viable in real-world scenarios. Current transferable attacks create adversarial perturbation over the entire image, resulting in excessive noise that overfit the source model. Concentrating perturbation to dominant image regions that are model-agnostic is crucial to improving adversarial efficacy. However, limiting perturbation to local regions in the spatial domain proves inadequate in augmenting transferability. To this end, we propose a transferable adversarial attack with fine-grained perturbation optimization in the frequency domain, creating centralized perturbation. We devise a systematic pipeline to dynamically constrain perturbation optimization to dominant frequency coefficients. The constraint is optimized in parallel at each iteration, ensuring the directional alignment of perturbation optimization with model prediction. Our approach allows us to centralize perturbation towards sample-specific important frequency features, which are shared by DNNs, effectively mitigating source model overfitting. Experiments demonstrate that by dynamically centralizing perturbation on dominating frequency coefficients, crafted adversarial examples exhibit stronger transferability, and allowing them to bypass various defenses.

摘要: 对抗的可转移性使对未知受害者的黑盒攻击能够深入神经网络(DNN)，从而使攻击在现实世界的场景中可行。当前的可转移攻击在整个图像上造成对抗性扰动，导致过多的噪声超出源模型的范围。将扰动集中到模型不可知的优势图像区域是提高对抗效能的关键。然而，将扰动限制在空间域中的局部区域被证明不足以增强可转移性。为此，我们提出了一种在频域进行细粒度扰动优化的可转移敌意攻击，产生集中扰动。我们设计了一个系统的管道来动态地将摄动优化约束到主导频率系数。在每次迭代中并行优化约束，确保扰动优化与模型预测的方向一致。我们的方法允许我们集中对样本特定的重要频率特征的扰动，这些特征由DNN共享，有效地缓解了源模型的过度拟合。实验表明，通过动态地将扰动集中在支配频率系数上，精心制作的敌意例子表现出更强的可转移性，并允许它们绕过各种防御。



## **31. SODA: Protecting Proprietary Information in On-Device Machine Learning Models**

SODA：在设备上机器学习模型中保护专有信息 cs.LG

**SubmitDate**: 2023-12-22    [abs](http://arxiv.org/abs/2312.15036v1) [paper-pdf](http://arxiv.org/pdf/2312.15036v1)

**Authors**: Akanksha Atrey, Ritwik Sinha, Saayan Mitra, Prashant Shenoy

**Abstract**: The growth of low-end hardware has led to a proliferation of machine learning-based services in edge applications. These applications gather contextual information about users and provide some services, such as personalized offers, through a machine learning (ML) model. A growing practice has been to deploy such ML models on the user's device to reduce latency, maintain user privacy, and minimize continuous reliance on a centralized source. However, deploying ML models on the user's edge device can leak proprietary information about the service provider. In this work, we investigate on-device ML models that are used to provide mobile services and demonstrate how simple attacks can leak proprietary information of the service provider. We show that different adversaries can easily exploit such models to maximize their profit and accomplish content theft. Motivated by the need to thwart such attacks, we present an end-to-end framework, SODA, for deploying and serving on edge devices while defending against adversarial usage. Our results demonstrate that SODA can detect adversarial usage with 89% accuracy in less than 50 queries with minimal impact on service performance, latency, and storage.

摘要: 低端硬件的增长导致了边缘应用中基于机器学习的服务的激增。这些应用程序收集有关用户的上下文信息，并通过机器学习(ML)模型提供一些服务，如个性化服务。越来越多的做法是在用户设备上部署这样的ML模型，以减少延迟，维护用户隐私，并最大限度地减少对集中式来源的持续依赖。然而，在用户的边缘设备上部署ML模型可能会泄露有关服务提供商的专有信息。在这项工作中，我们研究了用于提供移动服务的设备上ML模型，并演示了简单的攻击如何泄漏服务提供商的专有信息。我们表明，不同的对手可以很容易地利用这种模型来最大化他们的利润，并完成内容窃取。出于阻止此类攻击的需要，我们提出了一个端到端框架SODA，用于在边缘设备上部署和服务，同时防御恶意使用。我们的结果表明，SODA可以在不到50个查询的情况下以89%的准确率检测恶意使用，并且对服务性能、延迟和存储的影响最小。



## **32. Differentiable JPEG: The Devil is in the Details**

与众不同的JPEG：魔鬼在细节中 cs.CV

Accepted at WACV 2024. Project page:  https://christophreich1996.github.io/differentiable_jpeg/ WACV paper:  https://openaccess.thecvf.com/content/WACV2024/html/Reich_Differentiable_JPEG_The_Devil_Is_in_the_Details_WACV_2024_paper.html

**SubmitDate**: 2023-12-22    [abs](http://arxiv.org/abs/2309.06978v4) [paper-pdf](http://arxiv.org/pdf/2309.06978v4)

**Authors**: Christoph Reich, Biplob Debnath, Deep Patel, Srimat Chakradhar

**Abstract**: JPEG remains one of the most widespread lossy image coding methods. However, the non-differentiable nature of JPEG restricts the application in deep learning pipelines. Several differentiable approximations of JPEG have recently been proposed to address this issue. This paper conducts a comprehensive review of existing diff. JPEG approaches and identifies critical details that have been missed by previous methods. To this end, we propose a novel diff. JPEG approach, overcoming previous limitations. Our approach is differentiable w.r.t. the input image, the JPEG quality, the quantization tables, and the color conversion parameters. We evaluate the forward and backward performance of our diff. JPEG approach against existing methods. Additionally, extensive ablations are performed to evaluate crucial design choices. Our proposed diff. JPEG resembles the (non-diff.) reference implementation best, significantly surpassing the recent-best diff. approach by $3.47$dB (PSNR) on average. For strong compression rates, we can even improve PSNR by $9.51$dB. Strong adversarial attack results are yielded by our diff. JPEG, demonstrating the effective gradient approximation. Our code is available at https://github.com/necla-ml/Diff-JPEG.

摘要: JPEG仍然是应用最广泛的有损图像编码方法之一。然而，JPEG的不可微特性限制了其在深度学习管道中的应用。为了解决这个问题，最近已经提出了几种JPEG的可微近似。本文对现有的DIFF进行了全面的回顾。JPEG处理并确定了以前方法遗漏的关键细节。为此，我们提出了一个新颖的Diff。JPEG方法，克服了以前的限制。我们的方法是可微的W.r.t。输入图像、JPEG质量、量化表和颜色转换参数。我们评估了DIFF的向前和向后性能。JPEG方法与现有方法的对比。此外，还进行了广泛的消融，以评估关键的设计选择。我们提议的不同之处。JPEG与(Non-Diff.)参考实现最好，大大超过了最近最好的差异。平均接近3.47美元分贝(PSNR)。对于强压缩率，我们甚至可以将PSNR提高9.51美元分贝。强大的对抗性攻击结果是由我们的差异产生的。JPEG格式，演示了有效的渐变近似。我们的代码可以在https://github.com/necla-ml/Diff-JPEG.上找到



## **33. Hierarchical Multi-Agent Reinforcement Learning for Assessing False-Data Injection Attacks on Transportation Networks**

基于分层多智能体强化学习的交通网络虚假数据注入攻击评估 cs.AI

**SubmitDate**: 2023-12-22    [abs](http://arxiv.org/abs/2312.14625v1) [paper-pdf](http://arxiv.org/pdf/2312.14625v1)

**Authors**: Taha Eghtesad, Sirui Li, Yevgeniy Vorobeychik, Aron Laszka

**Abstract**: The increasing reliance of drivers on navigation applications has made transportation networks more susceptible to data-manipulation attacks by malicious actors. Adversaries may exploit vulnerabilities in the data collection or processing of navigation services to inject false information, and to thus interfere with the drivers' route selection. Such attacks can significantly increase traffic congestions, resulting in substantial waste of time and resources, and may even disrupt essential services that rely on road networks. To assess the threat posed by such attacks, we introduce a computational framework to find worst-case data-injection attacks against transportation networks. First, we devise an adversarial model with a threat actor who can manipulate drivers by increasing the travel times that they perceive on certain roads. Then, we employ hierarchical multi-agent reinforcement learning to find an approximate optimal adversarial strategy for data manipulation. We demonstrate the applicability of our approach through simulating attacks on the Sioux Falls, ND network topology.

摘要: 司机越来越依赖导航应用程序，这使得交通网络更容易受到恶意行为者的数据操纵攻击。攻击者可能会利用导航服务的数据收集或处理中的漏洞来注入虚假信息，从而干扰司机的路线选择。此类攻击可能会显著加剧交通拥堵，导致大量时间和资源的浪费，甚至可能扰乱依赖道路网络的基本服务。为了评估这类攻击造成的威胁，我们引入了一个计算框架来发现针对交通网络的最坏情况下的数据注入攻击。首先，我们设计了一个带有威胁参与者的对抗性模型，该威胁参与者可以通过增加司机在某些道路上感知的旅行时间来操纵司机。然后，我们使用分层多智能体强化学习来寻找数据操作的近似最优对抗策略。通过模拟对苏福尔斯网络拓扑结构的攻击，验证了该方法的适用性。



## **34. Complex Graph Laplacian Regularizer for Inferencing Grid States**

用于网格状态推断的复图拉普拉斯正则化算法 eess.SP

**SubmitDate**: 2023-12-22    [abs](http://arxiv.org/abs/2307.01906v2) [paper-pdf](http://arxiv.org/pdf/2307.01906v2)

**Authors**: Chinthaka Dinesh, Junfei Wang, Gene Cheung, Pirathayini Srikantha

**Abstract**: In order to maintain stable grid operations, system monitoring and control processes require the computation of grid states (e.g. voltage magnitude and angles) at high granularity. It is necessary to infer these grid states from measurements generated by a limited number of sensors like phasor measurement units (PMUs) that can be subjected to delays and losses due to channel artefacts, and/or adversarial attacks (e.g. denial of service, jamming, etc.). We propose a novel graph signal processing (GSP) based algorithm to interpolate states of the entire grid from observations of a small number of grid measurements. It is a two-stage process, where first an underlying Hermitian graph is learnt empirically from existing grid datasets. Then, the graph is used to interpolate missing grid signal samples in linear time. With our proposal, we can effectively reconstruct grid signals with significantly smaller number of observations when compared to existing traditional approaches (e.g. state estimation). In contrast to existing GSP approaches, we do not require knowledge of the underlying grid structure and parameters and are able to guarantee fast spectral optimization. We demonstrate the computational efficacy and accuracy of our proposal via practical studies conducted on the IEEE 118 bus system.

摘要: 为了维持稳定的电网运行，系统监测和控制过程需要计算高粒度的电网状态(如电压幅值和角度)。有必要从有限数量的传感器(如相量测量单元(PMU))生成的测量结果中推断这些网格状态，这些传感器可能由于信道伪影和/或对抗性攻击(例如拒绝服务、干扰等)而受到延迟和损失。我们提出了一种基于图信号处理(GSP)的新算法，该算法根据少量网格测量的观测值来内插整个网格的状态。这是一个分两个阶段的过程，首先从现有的网格数据集中经验地学习潜在的厄米特图。然后，使用该图在线性时间内对缺失的网格信号样本进行内插。与现有的传统方法(例如状态估计)相比，我们的建议可以用明显更少的观测值来有效地重建网格信号。与现有的GSP方法相比，我们不需要底层网格结构和参数的知识，并且能够保证快速的频谱优化。通过在IEEE118节点系统上进行的实际研究，证明了该方法的计算效率和准确性。



## **35. Backdoor Attack with Sparse and Invisible Trigger**

具有稀疏和隐形触发器的后门攻击 cs.CV

The first two authors contributed equally to this work. 13 pages

**SubmitDate**: 2023-12-22    [abs](http://arxiv.org/abs/2306.06209v2) [paper-pdf](http://arxiv.org/pdf/2306.06209v2)

**Authors**: Yinghua Gao, Yiming Li, Xueluan Gong, Zhifeng Li, Shu-Tao Xia, Qian Wang

**Abstract**: Deep neural networks (DNNs) are vulnerable to backdoor attacks, where the adversary manipulates a small portion of training data such that the victim model predicts normally on the benign samples but classifies the triggered samples as the target class. The backdoor attack is an emerging yet threatening training-phase threat, leading to serious risks in DNN-based applications. In this paper, we revisit the trigger patterns of existing backdoor attacks. We reveal that they are either visible or not sparse and therefore are not stealthy enough. More importantly, it is not feasible to simply combine existing methods to design an effective sparse and invisible backdoor attack. To address this problem, we formulate the trigger generation as a bi-level optimization problem with sparsity and invisibility constraints and propose an effective method to solve it. The proposed method is dubbed sparse and invisible backdoor attack (SIBA). We conduct extensive experiments on benchmark datasets under different settings, which verify the effectiveness of our attack and its resistance to existing backdoor defenses. The codes for reproducing main experiments are available at \url{https://github.com/YinghuaGao/SIBA}.

摘要: 深度神经网络(DNN)很容易受到后门攻击，对手操纵一小部分训练数据，使得受害者模型对良性样本进行正常预测，但将触发的样本归类为目标类。后门攻击是一种新出现的但具有威胁性的训练阶段威胁，导致基于DNN的应用程序存在严重风险。在本文中，我们回顾了现有后门攻击的触发模式。我们发现，它们要么是可见的，要么不是稀疏的，因此不够隐蔽。更重要的是，简单地结合现有方法来设计有效的稀疏、隐形的后门攻击是不可行的。针对这一问题，我们将触发器生成问题描述为一个具有稀疏性和不可见性约束的双层优化问题，并提出了一种有效的求解方法。该方法被称为稀疏不可见后门攻击(SIBA)。我们在不同环境下的基准数据集上进行了大量的实验，验证了我们的攻击的有效性及其对现有后门防御的抵抗力。有关复制主要实验的代码，请访问\url{https://github.com/YinghuaGao/SIBA}.



## **36. Asymmetric Bias in Text-to-Image Generation with Adversarial Attacks**

对抗性攻击下文本到图像生成中的非对称偏向 cs.LG

preprint version

**SubmitDate**: 2023-12-22    [abs](http://arxiv.org/abs/2312.14440v1) [paper-pdf](http://arxiv.org/pdf/2312.14440v1)

**Authors**: Haz Sameen Shahgir, Xianghao Kong, Greg Ver Steeg, Yue Dong

**Abstract**: The widespread use of Text-to-Image (T2I) models in content generation requires careful examination of their safety, including their robustness to adversarial attacks. Despite extensive research into this, the reasons for their effectiveness are underexplored. This paper presents an empirical study on adversarial attacks against T2I models, focusing on analyzing factors associated with attack success rates (ASRs). We introduce a new attack objective - entity swapping using adversarial suffixes and two gradient-based attack algorithms. Human and automatic evaluations reveal the asymmetric nature of ASRs on entity swap: for example, it is easier to replace "human" with "robot" in the prompt "a human dancing in the rain." with an adversarial suffix but is significantly harder in reverse. We further propose probing metrics to establish indicative signals from the model's beliefs to the adversarial ASR. We identify conditions resulting in a 60% success probability for adversarial attacks and others where this likelihood drops below 5%.

摘要: 文本到图像(T2I)模型在内容生成中的广泛使用需要仔细检查它们的安全性，包括它们对对手攻击的健壮性。尽管对此进行了广泛的研究，但其有效性的原因仍未得到充分探索。本文对T2I模型的对抗性攻击进行了实证研究，重点分析了影响攻击成功率(ASR)的因素。提出了一种新的攻击目标实体交换算法，利用对抗性后缀和两种基于梯度的攻击算法。人工评估和自动评估揭示了ASR在实体交换上的不对称性质：例如，在提示符“a Human in the雨中跳舞”中，更容易将“Human”替换为“bot”。有一个对抗性后缀，但反转起来要难得多。我们进一步提出了探测度量来建立从模型信念到对抗性ASR的指示性信号。我们确定了导致对抗性攻击成功概率为60%的条件，以及其他可能性降至5%以下的条件。



## **37. Elevating Defenses: Bridging Adversarial Training and Watermarking for Model Resilience**

提升防御：在对抗性训练和模型复原力水印之间架起桥梁 cs.LG

Accepted at DAI Workshop, AAAI 2024

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2312.14260v1) [paper-pdf](http://arxiv.org/pdf/2312.14260v1)

**Authors**: Janvi Thakkar, Giulio Zizzo, Sergio Maffeis

**Abstract**: Machine learning models are being used in an increasing number of critical applications; thus, securing their integrity and ownership is critical. Recent studies observed that adversarial training and watermarking have a conflicting interaction. This work introduces a novel framework to integrate adversarial training with watermarking techniques to fortify against evasion attacks and provide confident model verification in case of intellectual property theft. We use adversarial training together with adversarial watermarks to train a robust watermarked model. The key intuition is to use a higher perturbation budget to generate adversarial watermarks compared to the budget used for adversarial training, thus avoiding conflict. We use the MNIST and Fashion-MNIST datasets to evaluate our proposed technique on various model stealing attacks. The results obtained consistently outperform the existing baseline in terms of robustness performance and further prove the resilience of this defense against pruning and fine-tuning removal attacks.

摘要: 机器学习模型正在越来越多的关键应用中使用;因此，确保其完整性和所有权至关重要。最近的研究发现，对抗性训练和水印具有相互冲突的相互作用。这项工作引入了一个新的框架，将对抗性训练与水印技术相结合，以加强对规避攻击的防御，并在知识产权被盗的情况下提供可靠的模型验证。我们使用对抗训练与对抗水印一起训练鲁棒的水印模型。关键的直觉是，与用于对抗训练的预算相比，使用更高的扰动预算来生成对抗水印，从而避免冲突。我们使用MNIST和Fashion-MNIST数据集来评估我们提出的技术对各种模型窃取攻击的影响。所获得的结果在鲁棒性性能方面始终优于现有的基线，并进一步证明了这种防御对修剪和微调删除攻击的弹性。



## **38. Open-Set: ID Card Presentation Attack Detection using Neural Transfer Style**

开集：基于神经传递方式的身份证呈现攻击检测 cs.CV

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2312.13993v1) [paper-pdf](http://arxiv.org/pdf/2312.13993v1)

**Authors**: Reuben Markham, Juan M. Espin, Mario Nieto-Hidalgo, Juan E. Tapia

**Abstract**: The accurate detection of ID card Presentation Attacks (PA) is becoming increasingly important due to the rising number of online/remote services that require the presentation of digital photographs of ID cards for digital onboarding or authentication. Furthermore, cybercriminals are continuously searching for innovative ways to fool authentication systems to gain unauthorized access to these services. Although advances in neural network design and training have pushed image classification to the state of the art, one of the main challenges faced by the development of fraud detection systems is the curation of representative datasets for training and evaluation. The handcrafted creation of representative presentation attack samples often requires expertise and is very time-consuming, thus an automatic process of obtaining high-quality data is highly desirable. This work explores ID card Presentation Attack Instruments (PAI) in order to improve the generation of samples with four Generative Adversarial Networks (GANs) based image translation models and analyses the effectiveness of the generated data for training fraud detection systems. Using open-source data, we show that synthetic attack presentations are an adequate complement for additional real attack presentations, where we obtain an EER performance increase of 0.63% points for print attacks and a loss of 0.29% for screen capture attacks.

摘要: 由于越来越多的在线/远程服务需要出示ID卡的数字照片以进行数字加载或认证，因此ID卡出示攻击（PA）的准确检测变得越来越重要。此外，网络犯罪分子正在不断寻找创新的方法来欺骗身份验证系统，以获得对这些服务的未经授权的访问。虽然神经网络设计和训练的进步已经将图像分类推向了最先进的水平，但欺诈检测系统开发所面临的主要挑战之一是对用于训练和评估的代表性数据集进行管理。手工创建代表性呈现攻击样本通常需要专业知识并且非常耗时，因此非常需要获得高质量数据的自动过程。这项工作探讨了身份证呈现攻击工具（PAI），以改进基于四个生成对抗网络（GAN）的图像翻译模型的样本生成，并分析了生成的数据用于训练欺诈检测系统的有效性。使用开源数据，我们表明，合成攻击演示文稿是一个足够的补充额外的真正的攻击演示文稿，我们获得了EER性能增加0.63%点的打印攻击和0.29%的屏幕捕获攻击的损失。



## **39. AutoAugment Input Transformation for Highly Transferable Targeted Attacks**

用于高可转移性目标攻击的自动增强输入转换 cs.CV

10 pages, 6 figures

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2312.14218v1) [paper-pdf](http://arxiv.org/pdf/2312.14218v1)

**Authors**: Haobo Lu, Xin Liu, Kun He

**Abstract**: Deep Neural Networks (DNNs) are widely acknowledged to be susceptible to adversarial examples, wherein imperceptible perturbations are added to clean examples through diverse input transformation attacks. However, these methods originally designed for non-targeted attacks exhibit low success rates in targeted attacks. Recent targeted adversarial attacks mainly pay attention to gradient optimization, attempting to find the suitable perturbation direction. However, few of them are dedicated to input transformation.In this work, we observe a positive correlation between the logit/probability of the target class and diverse input transformation methods in targeted attacks. To this end, we propose a novel targeted adversarial attack called AutoAugment Input Transformation (AAIT). Instead of relying on hand-made strategies, AAIT searches for the optimal transformation policy from a transformation space comprising various operations. Then, AAIT crafts adversarial examples using the found optimal transformation policy to boost the adversarial transferability in targeted attacks. Extensive experiments conducted on CIFAR-10 and ImageNet-Compatible datasets demonstrate that the proposed AAIT surpasses other transfer-based targeted attacks significantly.

摘要: 深度神经网络(DNN)被公认为容易受到敌意例子的影响，在这种例子中，通过不同的输入变换攻击将不可察觉的扰动添加到干净的例子中。然而，这些方法最初是为非目标攻击设计的，但在目标攻击中成功率很低。最近的定向对抗性攻击主要关注梯度优化，试图找到合适的扰动方向。在本工作中，我们观察到目标类的Logit/概率与目标攻击中不同的输入转换方法之间存在正相关关系。为此，我们提出了一种新的有针对性的对抗性攻击，称为自动增强输入变换(AAIT)。AAIT不依赖于手工制定的策略，而是从包含各种操作的变换空间中搜索最优变换策略。然后，AAIT使用找到的最优转换策略来创建对抗性实例，以提高定向攻击中的对抗性可转移性。在CIFAR-10和ImageNet兼容的数据集上进行的大量实验表明，所提出的AAIT攻击明显优于其他基于传输的定向攻击。



## **40. Adversarial Infrared Curves: An Attack on Infrared Pedestrian Detectors in the Physical World**

对抗性红外曲线：对物理世界中红外行人探测器的攻击 cs.CR

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2312.14217v1) [paper-pdf](http://arxiv.org/pdf/2312.14217v1)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstract**: Deep neural network security is a persistent concern, with considerable research on visible light physical attacks but limited exploration in the infrared domain. Existing approaches, like white-box infrared attacks using bulb boards and QR suits, lack realism and stealthiness. Meanwhile, black-box methods with cold and hot patches often struggle to ensure robustness. To bridge these gaps, we propose Adversarial Infrared Curves (AdvIC). Using Particle Swarm Optimization, we optimize two Bezier curves and employ cold patches in the physical realm to introduce perturbations, creating infrared curve patterns for physical sample generation. Our extensive experiments confirm AdvIC's effectiveness, achieving 94.8\% and 67.2\% attack success rates for digital and physical attacks, respectively. Stealthiness is demonstrated through a comparative analysis, and robustness assessments reveal AdvIC's superiority over baseline methods. When deployed against diverse advanced detectors, AdvIC achieves an average attack success rate of 76.8\%, emphasizing its robust nature. we explore adversarial defense strategies against AdvIC and examine its impact under various defense mechanisms. Given AdvIC's substantial security implications for real-world vision-based applications, urgent attention and mitigation efforts are warranted.

摘要: 深度神经网络的安全性一直是一个令人关注的问题，对可见光物理攻击的研究很多，但在红外领域的探索有限。现有的方法，如使用灯泡板和QR套装的白盒红外攻击，缺乏真实性和隐蔽性。与此同时，带有冷补丁和热补丁的黑盒方法往往难以确保健壮性。为了弥补这些差距，我们提出了对抗性红外曲线(Advic)。利用粒子群算法，我们优化了两条Bezier曲线，并利用物理领域中的冷斑来引入扰动，创建了用于物理样本生成的红外曲线图案。我们的大量实验证实了Advic的有效性，数字攻击和物理攻击的攻击成功率分别达到94.8%和67.2%。隐蔽性通过比较分析得到了证明，健壮性评估显示了Advic相对于基线方法的优势。当部署在不同的高级探测器上时，Advic的平均攻击成功率为76.8\%，强调了其健壮性。我们探讨了针对Advic的对抗性防御策略，并考察了其在各种防御机制下的影响。鉴于Advic对现实世界基于视觉的应用程序的重大安全影响，迫切需要关注和缓解努力。



## **41. Quantum Neural Networks under Depolarization Noise: Exploring White-Box Attacks and Defenses**

去极化噪声下的量子神经网络：白盒攻击与防御探索 quant-ph

Poster at Quantum Techniques in Machine Learning (QTML) 2023

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2311.17458v2) [paper-pdf](http://arxiv.org/pdf/2311.17458v2)

**Authors**: David Winderl, Nicola Franco, Jeanette Miriam Lorenz

**Abstract**: Leveraging the unique properties of quantum mechanics, Quantum Machine Learning (QML) promises computational breakthroughs and enriched perspectives where traditional systems reach their boundaries. However, similarly to classical machine learning, QML is not immune to adversarial attacks. Quantum adversarial machine learning has become instrumental in highlighting the weak points of QML models when faced with adversarial crafted feature vectors. Diving deep into this domain, our exploration shines light on the interplay between depolarization noise and adversarial robustness. While previous results enhanced robustness from adversarial threats through depolarization noise, our findings paint a different picture. Interestingly, adding depolarization noise discontinued the effect of providing further robustness for a multi-class classification scenario. Consolidating our findings, we conducted experiments with a multi-class classifier adversarially trained on gate-based quantum simulators, further elucidating this unexpected behavior.

摘要: 利用量子力学的独特性质，量子机器学习(QML)有望在传统系统达到其边界的地方实现计算突破和丰富视角。然而，与经典机器学习类似，QML也不能幸免于对手攻击。量子对抗性机器学习已成为突出QML模型在面对对抗性特制特征向量时的弱点的工具。深入到这个领域，我们的探索揭示了去极化噪声和对手稳健性之间的相互作用。虽然之前的结果通过去极化噪声增强了对抗威胁的稳健性，但我们的发现描绘了一幅不同的图景。有趣的是，添加去极化噪声会中断为多类分类场景提供进一步稳健性的效果。综合我们的发现，我们用一个多类分类器进行了实验，该分类器在基于门的量子模拟器上进行了相反的训练，进一步阐明了这种意想不到的行为。



## **42. Where and How to Attack? A Causality-Inspired Recipe for Generating Counterfactual Adversarial Examples**

向哪里进攻，如何进攻？由因果关系启发生成反事实对抗性例子的秘诀 cs.LG

Accepted by AAAI-2024

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2312.13628v1) [paper-pdf](http://arxiv.org/pdf/2312.13628v1)

**Authors**: Ruichu Cai, Yuxuan Zhu, Jie Qiao, Zefeng Liang, Furui Liu, Zhifeng Hao

**Abstract**: Deep neural networks (DNNs) have been demonstrated to be vulnerable to well-crafted \emph{adversarial examples}, which are generated through either well-conceived $\mathcal{L}_p$-norm restricted or unrestricted attacks. Nevertheless, the majority of those approaches assume that adversaries can modify any features as they wish, and neglect the causal generating process of the data, which is unreasonable and unpractical. For instance, a modification in income would inevitably impact features like the debt-to-income ratio within a banking system. By considering the underappreciated causal generating process, first, we pinpoint the source of the vulnerability of DNNs via the lens of causality, then give theoretical results to answer \emph{where to attack}. Second, considering the consequences of the attack interventions on the current state of the examples to generate more realistic adversarial examples, we propose CADE, a framework that can generate \textbf{C}ounterfactual \textbf{AD}versarial \textbf{E}xamples to answer \emph{how to attack}. The empirical results demonstrate CADE's effectiveness, as evidenced by its competitive performance across diverse attack scenarios, including white-box, transfer-based, and random intervention attacks.

摘要: 深度神经网络(DNN)已经被证明容易受到精心设计的对手例子的攻击，这些例子是通过精心设计的数学{L}_p$-范数受限或非受限攻击而产生的。然而，这些方法中的大多数都假设对手可以随意修改任何特征，而忽略了数据的因果生成过程，这是不合理和不切实际的。例如，收入的调整将不可避免地影响银行体系内的债务收入比等特征。通过考虑被低估的因果生成过程，我们首先通过因果镜头找出DNN脆弱性的来源，然后给出理论结果来回答{攻击在哪里}。其次，考虑到攻击干预对实例的当前状态的影响，为了生成更真实的对抗性实例，我们提出了CADE框架，它可以生成\extbf{C}非事实\extbf{AD}versariative\extbf{E}样例来回答\emph{如何攻击}。实验结果证明了CADE的有效性，它在各种攻击场景中的竞争性能证明了这一点，包括白盒攻击、基于传输的攻击和随机干预攻击。



## **43. ARBiBench: Benchmarking Adversarial Robustness of Binarized Neural Networks**

ARBiBitch：衡量二值化神经网络对抗健壮性的基准 cs.CV

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2312.13575v1) [paper-pdf](http://arxiv.org/pdf/2312.13575v1)

**Authors**: Peng Zhao, Jiehua Zhang, Bowen Peng, Longguang Wang, YingMei Wei, Yu Liu, Li Liu

**Abstract**: Network binarization exhibits great potential for deployment on resource-constrained devices due to its low computational cost. Despite the critical importance, the security of binarized neural networks (BNNs) is rarely investigated. In this paper, we present ARBiBench, a comprehensive benchmark to evaluate the robustness of BNNs against adversarial perturbations on CIFAR-10 and ImageNet. We first evaluate the robustness of seven influential BNNs on various white-box and black-box attacks. The results reveal that 1) The adversarial robustness of BNNs exhibits a completely opposite performance on the two datasets under white-box attacks. 2) BNNs consistently exhibit better adversarial robustness under black-box attacks. 3) Different BNNs exhibit certain similarities in their robustness performance. Then, we conduct experiments to analyze the adversarial robustness of BNNs based on these insights. Our research contributes to inspiring future research on enhancing the robustness of BNNs and advancing their application in real-world scenarios.

摘要: 网络二值化由于其计算成本低，在资源受限的设备上具有很大的部署潜力。尽管二值化神经网络(BNN)的安全性至关重要，但很少有人研究它的安全性。在本文中，我们提出了一种评估BNN对CIFAR-10和ImageNet上的敌意干扰的健壮性的综合基准ARBiBch。我们首先评估了七种有影响力的BNN对各种白盒和黑盒攻击的健壮性。结果表明：1)在白盒攻击下，BNN在两个数据集上的对抗健壮性表现出完全相反的表现。2)BNN在黑盒攻击下表现出更好的对抗健壮性。3)不同的BNN在鲁棒性方面表现出一定的相似性。然后，在此基础上进行实验，分析了BNN的对抗健壮性。我们的研究有助于启发未来增强BNN健壮性的研究，促进其在现实世界场景中的应用。



## **44. Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models**

针对大型语言模型的间接提示注入攻击的基准测试和防御 cs.CL

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2312.14197v1) [paper-pdf](http://arxiv.org/pdf/2312.14197v1)

**Authors**: Jingwei Yi, Yueqi Xie, Bin Zhu, Keegan Hines, Emre Kiciman, Guangzhong Sun, Xing Xie, Fangzhao Wu

**Abstract**: Recent remarkable advancements in large language models (LLMs) have led to their widespread adoption in various applications. A key feature of these applications is the combination of LLMs with external content, where user instructions and third-party content are combined to create prompts for LLM processing. These applications, however, are vulnerable to indirect prompt injection attacks, where malicious instructions embedded within external content compromise LLM's output, causing their responses to deviate from user expectations. Despite the discovery of this security issue, no comprehensive analysis of indirect prompt injection attacks on different LLMs is available due to the lack of a benchmark. Furthermore, no effective defense has been proposed.   In this work, we introduce the first benchmark, BIPIA, to measure the robustness of various LLMs and defenses against indirect prompt injection attacks. Our experiments reveal that LLMs with greater capabilities exhibit more vulnerable to indirect prompt injection attacks for text tasks, resulting in a higher ASR. We hypothesize that indirect prompt injection attacks are mainly due to the LLMs' inability to distinguish between instructions and external content. Based on this conjecture, we propose four black-box methods based on prompt learning and a white-box defense methods based on fine-tuning with adversarial training to enable LLMs to distinguish between instructions and external content and ignore instructions in the external content. Our experimental results show that our black-box defense methods can effectively reduce ASR but cannot completely thwart indirect prompt injection attacks, while our white-box defense method can reduce ASR to nearly zero with little adverse impact on the LLM's performance on general tasks. We hope that our benchmark and defenses can inspire future work in this important area.

摘要: 大型语言模型（LLM）最近取得了显着的进步，导致其在各种应用中的广泛采用。这些应用程序的一个关键特性是LLM与外部内容的组合，其中用户指令和第三方内容被组合以创建LLM处理的提示。然而，这些应用程序容易受到间接提示注入攻击，其中嵌入在外部内容中的恶意指令会损害LLM的输出，导致其响应偏离用户预期。尽管发现了这个安全问题，但由于缺乏基准测试，没有对不同LLM上的间接提示注入攻击进行全面分析。此外，没有提出有效的辩护。   在这项工作中，我们引入了第一个基准，BIPIA，来衡量各种LLM的鲁棒性和对间接提示注入攻击的防御。我们的实验表明，具有更大功能的LLM更容易受到文本任务的间接提示注入攻击，从而导致更高的ASR。我们假设间接提示注入攻击主要是由于LLM无法区分指令和外部内容。基于这一猜想，我们提出了四种基于提示学习的黑盒方法和一种基于对抗训练微调的白盒防御方法，使LLM能够区分指令和外部内容，并忽略外部内容中的指令。我们的实验结果表明，我们的黑盒防御方法可以有效地减少ASR，但不能完全阻止间接提示注入攻击，而我们的白盒防御方法可以减少ASR几乎为零，对LLM的一般任务的性能几乎没有不利影响。我们希望我们的基准和防御可以激励未来在这一重要领域的工作。



## **45. Adversarial Purification with the Manifold Hypothesis**

流形假设下的对抗性净化 cs.LG

Extended version of paper accepted at AAAI 2024 with supplementary  materials

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2210.14404v5) [paper-pdf](http://arxiv.org/pdf/2210.14404v5)

**Authors**: Zhaoyuan Yang, Zhiwei Xu, Jing Zhang, Richard Hartley, Peter Tu

**Abstract**: In this work, we formulate a novel framework for adversarial robustness using the manifold hypothesis. This framework provides sufficient conditions for defending against adversarial examples. We develop an adversarial purification method with this framework. Our method combines manifold learning with variational inference to provide adversarial robustness without the need for expensive adversarial training. Experimentally, our approach can provide adversarial robustness even if attackers are aware of the existence of the defense. In addition, our method can also serve as a test-time defense mechanism for variational autoencoders.

摘要: 在这项工作中，我们使用流形假设建立了一个新的对抗健壮性框架。这一框架为对抗对手的例子提供了充分的条件。在此框架下，我们提出了一种对抗性净化方法。我们的方法结合了流形学习和变分推理，在不需要昂贵的对抗性训练的情况下提供对抗性健壮性。在实验上，即使攻击者知道防御的存在，我们的方法也可以提供对抗的健壮性。此外，我们的方法还可以作为变分自动编码器的测试时间防御机制。



## **46. Adversarial Markov Games: On Adaptive Decision-Based Attacks and Defenses**

对抗性马尔可夫博弈：基于自适应决策的攻防 cs.AI

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2312.13435v1) [paper-pdf](http://arxiv.org/pdf/2312.13435v1)

**Authors**: Ilias Tsingenopoulos, Vera Rimmer, Davy Preuveneers, Fabio Pierazzi, Lorenzo Cavallaro, Wouter Joosen

**Abstract**: Despite considerable efforts on making them robust, real-world ML-based systems remain vulnerable to decision based attacks, as definitive proofs of their operational robustness have so far proven intractable. The canonical approach in robustness evaluation calls for adaptive attacks, that is with complete knowledge of the defense and tailored to bypass it. In this study, we introduce a more expansive notion of being adaptive and show how attacks but also defenses can benefit by it and by learning from each other through interaction. We propose and evaluate a framework for adaptively optimizing black-box attacks and defenses against each other through the competitive game they form. To reliably measure robustness, it is important to evaluate against realistic and worst-case attacks. We thus augment both attacks and the evasive arsenal at their disposal through adaptive control, and observe that the same can be done for defenses, before we evaluate them first apart and then jointly under a multi-agent perspective. We demonstrate that active defenses, which control how the system responds, are a necessary complement to model hardening when facing decision-based attacks; then how these defenses can be circumvented by adaptive attacks, only to finally elicit active and adaptive defenses. We validate our observations through a wide theoretical and empirical investigation to confirm that AI-enabled adversaries pose a considerable threat to black-box ML-based systems, rekindling the proverbial arms race where defenses have to be AI-enabled too. Succinctly, we address the challenges posed by adaptive adversaries and develop adaptive defenses, thereby laying out effective strategies in ensuring the robustness of ML-based systems deployed in the real-world.

摘要: 尽管做出了相当大的努力来使它们健壮，但现实世界中基于ML的系统仍然容易受到基于决策的攻击，因为到目前为止，对其操作健壮性的确凿证据被证明是难以处理的。健壮性评估的规范方法要求自适应攻击，即完全了解防御并量身定做以绕过它。在这项研究中，我们引入了一个更广泛的适应性概念，并展示了攻击和防御如何从它和通过互动相互学习中受益。我们提出并评估了一个框架，用于通过形成的竞争博弈自适应地优化黑盒攻击和防御。要可靠地衡量健壮性，重要的是要针对现实和最坏情况下的攻击进行评估。因此，我们通过自适应控制来增强攻击和可供其使用的躲避武器，并观察到同样可以对防御做同样的事情，然后我们首先分开评估它们，然后在多智能体的角度下进行联合评估。我们演示了控制系统如何响应的主动防御是面对基于决策的攻击时模型强化的必要补充；然后说明如何通过自适应攻击来规避这些防御，最终只会引发主动和自适应防御。我们通过广泛的理论和经验调查验证了我们的观察结果，以确认启用AI的对手对基于黑盒ML的系统构成了相当大的威胁，重新点燃了众所周知的军备竞赛，其中防御也必须启用AI。简而言之，我们解决了适应性对手带来的挑战，并开发了适应性防御，从而制定了有效的策略，以确保部署在现实世界中的基于ML的系统的健壮性。



## **47. Universal and Transferable Adversarial Attacks on Aligned Language Models**

对对齐语言模型的通用和可转移的对抗性攻击 cs.CL

Website: http://llm-attacks.org/

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2307.15043v2) [paper-pdf](http://arxiv.org/pdf/2307.15043v2)

**Authors**: Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, Matt Fredrikson

**Abstract**: Because "out-of-the-box" large language models are capable of generating a great deal of objectionable content, recent work has focused on aligning these models in an attempt to prevent undesirable generation. While there has been some success at circumventing these measures -- so-called "jailbreaks" against LLMs -- these attacks have required significant human ingenuity and are brittle in practice. In this paper, we propose a simple and effective attack method that causes aligned language models to generate objectionable behaviors. Specifically, our approach finds a suffix that, when attached to a wide range of queries for an LLM to produce objectionable content, aims to maximize the probability that the model produces an affirmative response (rather than refusing to answer). However, instead of relying on manual engineering, our approach automatically produces these adversarial suffixes by a combination of greedy and gradient-based search techniques, and also improves over past automatic prompt generation methods.   Surprisingly, we find that the adversarial prompts generated by our approach are quite transferable, including to black-box, publicly released LLMs. Specifically, we train an adversarial attack suffix on multiple prompts (i.e., queries asking for many different types of objectionable content), as well as multiple models (in our case, Vicuna-7B and 13B). When doing so, the resulting attack suffix is able to induce objectionable content in the public interfaces to ChatGPT, Bard, and Claude, as well as open source LLMs such as LLaMA-2-Chat, Pythia, Falcon, and others. In total, this work significantly advances the state-of-the-art in adversarial attacks against aligned language models, raising important questions about how such systems can be prevented from producing objectionable information. Code is available at github.com/llm-attacks/llm-attacks.

摘要: 由于“开箱即用”的大型语言模型能够生成大量令人反感的内容，最近的工作重点是调整这些模型，以试图防止不必要的生成。虽然在规避这些措施方面取得了一些成功--即所谓的针对LLMS的“越狱”--但这些攻击需要大量的人类智慧，而且在实践中是脆弱的。在本文中，我们提出了一种简单有效的攻击方法，使对齐的语言模型产生令人反感的行为。具体地说，我们的方法找到了一个后缀，当附加到LLM的广泛查询中以产生令人反感的内容时，旨在最大化该模型产生肯定响应(而不是拒绝回答)的概率。然而，我们的方法不依赖于人工设计，而是通过贪婪和基于梯度的搜索技术相结合来自动生成这些对抗性后缀，并且改进了过去的自动提示生成方法。令人惊讶的是，我们发现我们的方法生成的对抗性提示是相当可转移的，包括到黑盒，公开发布的LLM。具体地说，我们对多个提示(即，要求许多不同类型的不良内容的查询)以及多个模型(在我们的案例中，Vicuna-7B和13B)训练对抗性攻击后缀。这样做时，生成的攻击后缀能够在ChatGPT、Bard和Claude的公共接口以及开源LLM(如llama-2-chat、Pythia、Falcon和其他)中诱导令人反感的内容。总而言之，这项工作极大地推进了针对对齐语言模型的对抗性攻击的最新水平，提出了如何防止此类系统产生令人反感的信息的重要问题。代码可在githorb.com/llm-Attages/llm-Attack上找到。



## **48. On the complexity of sabotage games for network security**

论破坏游戏对网络安全的复杂性 cs.CC

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2312.13132v1) [paper-pdf](http://arxiv.org/pdf/2312.13132v1)

**Authors**: Dhananjay Raju, Georgios Bakirtzis, Ufuk Topcu

**Abstract**: Securing dynamic networks against adversarial actions is challenging because of the need to anticipate and counter strategic disruptions by adversarial entities within complex network structures. Traditional game-theoretic models, while insightful, often fail to model the unpredictability and constraints of real-world threat assessment scenarios. We refine sabotage games to reflect the realistic limitations of the saboteur and the network operator. By transforming sabotage games into reachability problems, our approach allows applying existing computational solutions to model realistic restrictions on attackers and defenders within the game. Modifying sabotage games into dynamic network security problems successfully captures the nuanced interplay of strategy and uncertainty in dynamic network security. Theoretically, we extend sabotage games to model network security contexts and thoroughly explore if the additional restrictions raise their computational complexity, often the bottleneck of game theory in practical contexts. Practically, this research sets the stage for actionable insights for developing robust defense mechanisms by understanding what risks to mitigate in dynamically changing networks under threat.

摘要: 确保动态网络不受敌对行动的影响是具有挑战性的，因为需要预测和应对复杂网络结构中敌对实体的战略中断。传统的博弈论模型虽然有洞察力，但往往无法对现实世界威胁评估情景的不可预测性和约束进行建模。我们改进了破坏游戏，以反映破坏者和网络运营商的现实限制。通过将破坏游戏转换为可达性问题，我们的方法允许应用现有的计算解决方案来模拟游戏中对攻击者和防御者的现实限制。将破坏博弈转化为动态网络安全问题，成功地捕捉到了动态网络安全中战略和不确定性的微妙相互作用。从理论上讲，我们将破坏游戏扩展到对网络安全环境进行建模，并深入探索额外的限制是否会增加其计算复杂性，这在实际环境中往往是博弈论的瓶颈。实际上，这项研究通过了解在受到威胁的动态变化的网络中需要缓解哪些风险，为开发强大的防御机制提供了可行的见解。



## **49. Prometheus: Infrastructure Security Posture Analysis with AI-generated Attack Graphs**

普罗米修斯：使用人工智能生成的攻击图进行基础设施安全态势分析 cs.CR

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2312.13119v1) [paper-pdf](http://arxiv.org/pdf/2312.13119v1)

**Authors**: Xin Jin, Charalampos Katsis, Fan Sang, Jiahao Sun, Elisa Bertino, Ramana Rao Kompella, Ashish Kundu

**Abstract**: The rampant occurrence of cybersecurity breaches imposes substantial limitations on the progress of network infrastructures, leading to compromised data, financial losses, potential harm to individuals, and disruptions in essential services. The current security landscape demands the urgent development of a holistic security assessment solution that encompasses vulnerability analysis and investigates the potential exploitation of these vulnerabilities as attack paths. In this paper, we propose Prometheus, an advanced system designed to provide a detailed analysis of the security posture of computing infrastructures. Using user-provided information, such as device details and software versions, Prometheus performs a comprehensive security assessment. This assessment includes identifying associated vulnerabilities and constructing potential attack graphs that adversaries can exploit. Furthermore, Prometheus evaluates the exploitability of these attack paths and quantifies the overall security posture through a scoring mechanism. The system takes a holistic approach by analyzing security layers encompassing hardware, system, network, and cryptography. Furthermore, Prometheus delves into the interconnections between these layers, exploring how vulnerabilities in one layer can be leveraged to exploit vulnerabilities in others. In this paper, we present the end-to-end pipeline implemented in Prometheus, showcasing the systematic approach adopted for conducting this thorough security analysis.

摘要: 网络安全漏洞的猖獗发生对网络基础设施的进展施加了很大限制，导致数据泄露、经济损失、对个人的潜在伤害以及基本服务中断。当前的安全形势要求迫切开发一种全面的安全评估解决方案，其中包括漏洞分析，并调查利用这些漏洞作为攻击途径的可能性。在本文中，我们提出了普罗米修斯，这是一个先进的系统，旨在提供详细的分析计算基础设施的安全态势。使用用户提供的信息，如设备详细信息和软件版本，普罗米修斯进行全面的安全评估。该评估包括识别相关漏洞和构建潜在的攻击图，以供攻击者利用。此外，普罗米修斯还评估了这些攻击路径的可利用性，并通过评分机制量化了总体安全态势。该系统采取整体方法，分析包括硬件、系统、网络和加密在内的安全层。此外，普罗米修斯深入研究了这些层之间的相互联系，探索如何利用一个层中的漏洞来利用其他层中的漏洞。在这篇文章中，我们介绍了在普罗米修斯中实现的端到端管道，展示了为进行这种彻底的安全分析而采用的系统方法。



## **50. LRS: Enhancing Adversarial Transferability through Lipschitz Regularized Surrogate**

LRS：通过Lipschitz正则化代理提高对手的可转移性 cs.LG

AAAI 2024

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2312.13118v1) [paper-pdf](http://arxiv.org/pdf/2312.13118v1)

**Authors**: Tao Wu, Tie Luo, Donald C. Wunsch

**Abstract**: The transferability of adversarial examples is of central importance to transfer-based black-box adversarial attacks. Previous works for generating transferable adversarial examples focus on attacking \emph{given} pretrained surrogate models while the connections between surrogate models and adversarial trasferability have been overlooked. In this paper, we propose {\em Lipschitz Regularized Surrogate} (LRS) for transfer-based black-box attacks, a novel approach that transforms surrogate models towards favorable adversarial transferability. Using such transformed surrogate models, any existing transfer-based black-box attack can run without any change, yet achieving much better performance. Specifically, we impose Lipschitz regularization on the loss landscape of surrogate models to enable a smoother and more controlled optimization process for generating more transferable adversarial examples. In addition, this paper also sheds light on the connection between the inner properties of surrogate models and adversarial transferability, where three factors are identified: smaller local Lipschitz constant, smoother loss landscape, and stronger adversarial robustness. We evaluate our proposed LRS approach by attacking state-of-the-art standard deep neural networks and defense models. The results demonstrate significant improvement on the attack success rates and transferability. Our code is available at https://github.com/TrustAIoT/LRS.

摘要: 对抗性例子的可转移性对于基于转移的黑盒对抗性攻击是至关重要的。以往关于生成可传递对抗实例的工作主要集中在攻击预先训练好的代理模型上，而忽略了代理模型与对抗传递能力之间的联系。针对基于转移的黑盒攻击，提出了一种将代理模型转化为有利的对抗性转移的新方法--LRS。使用这种转换的代理模型，任何现有的基于传输的黑盒攻击都可以在不做任何更改的情况下运行，但获得了更好的性能。具体地说，我们将Lipschitz正则化应用于代理模型的损失图景，以实现更平滑和更可控的优化过程，从而生成更多可转移的对抗性例子。此外，本文还揭示了代理模型的内在性质与对抗转移之间的关系，其中确定了三个因素：较小的局部Lipschitz常数、更平滑的损失图景和更强的对抗稳健性。我们通过攻击最先进的标准深度神经网络和防御模型来评估我们提出的LRS方法。结果表明，在攻击成功率和可转移性方面都有显著的提高。我们的代码可以在https://github.com/TrustAIoT/LRS.上找到



