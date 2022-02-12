# Latest Adversarial Attack Papers
**update at 2022-02-12 21:06:09**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Towards Assessing and Characterizing the Semantic Robustness of Face Recognition**

面向人脸识别的语义健壮性评估与表征 cs.CV

26 pages, 18 figures

**SubmitDate**: 2022-02-10    [paper-pdf](http://arxiv.org/pdf/2202.04978v1)

**Authors**: Juan C. Pérez, Motasem Alfarra, Ali Thabet, Pablo Arbeláez, Bernard Ghanem

**Abstracts**: Deep Neural Networks (DNNs) lack robustness against imperceptible perturbations to their input. Face Recognition Models (FRMs) based on DNNs inherit this vulnerability. We propose a methodology for assessing and characterizing the robustness of FRMs against semantic perturbations to their input. Our methodology causes FRMs to malfunction by designing adversarial attacks that search for identity-preserving modifications to faces. In particular, given a face, our attacks find identity-preserving variants of the face such that an FRM fails to recognize the images belonging to the same identity. We model these identity-preserving semantic modifications via direction- and magnitude-constrained perturbations in the latent space of StyleGAN. We further propose to characterize the semantic robustness of an FRM by statistically describing the perturbations that induce the FRM to malfunction. Finally, we combine our methodology with a certification technique, thus providing (i) theoretical guarantees on the performance of an FRM, and (ii) a formal description of how an FRM may model the notion of face identity.

摘要: 深度神经网络(DNNs)对其输入的不可察觉的扰动缺乏鲁棒性。基于DNN的人脸识别模型(FRM)继承了此漏洞。我们提出了一种方法来评估和表征FRM对其输入的语义扰动的鲁棒性。我们的方法论通过设计对抗性攻击来搜索对人脸的身份保留修改，从而导致FRMS发生故障。特别地，在给定一张人脸的情况下，我们的攻击会找到该人脸的保持身份的变体，使得FRM无法识别属于同一身份的图像。我们在StyleGan的潜在空间中通过方向和幅度约束的扰动来模拟这些保持身份的语义修改。我们进一步提出通过统计描述导致FRM故障的扰动来表征FRM的语义健壮性。最后，我们将我们的方法与认证技术相结合，从而提供(I)FRM性能的理论保证，以及(Ii)FRM如何建模面部身份概念的正式描述。



## **2. Beyond ImageNet Attack: Towards Crafting Adversarial Examples for Black-box Domains**

超越ImageNet攻击：为黑盒领域精心制作敌意示例 cs.CV

Accepted by ICLR 2022

**SubmitDate**: 2022-02-10    [paper-pdf](http://arxiv.org/pdf/2201.11528v3)

**Authors**: Qilong Zhang, Xiaodan Li, Yuefeng Chen, Jingkuan Song, Lianli Gao, Yuan He, Hui Xue

**Abstracts**: Adversarial examples have posed a severe threat to deep neural networks due to their transferable nature. Currently, various works have paid great efforts to enhance the cross-model transferability, which mostly assume the substitute model is trained in the same domain as the target model. However, in reality, the relevant information of the deployed model is unlikely to leak. Hence, it is vital to build a more practical black-box threat model to overcome this limitation and evaluate the vulnerability of deployed models. In this paper, with only the knowledge of the ImageNet domain, we propose a Beyond ImageNet Attack (BIA) to investigate the transferability towards black-box domains (unknown classification tasks). Specifically, we leverage a generative model to learn the adversarial function for disrupting low-level features of input images. Based on this framework, we further propose two variants to narrow the gap between the source and target domains from the data and model perspectives, respectively. Extensive experiments on coarse-grained and fine-grained domains demonstrate the effectiveness of our proposed methods. Notably, our methods outperform state-of-the-art approaches by up to 7.71\% (towards coarse-grained domains) and 25.91\% (towards fine-grained domains) on average. Our code is available at \url{https://github.com/qilong-zhang/Beyond-ImageNet-Attack}.

摘要: 对抗性例子由于其可转移性，对深度神经网络构成了严重的威胁。目前，各种研究都在努力提高模型间的可移植性，大多假设替身模型与目标模型在同一领域进行训练。然而，在现实中，部署的模型的相关信息不太可能泄露。因此，构建一个更实用的黑盒威胁模型来克服这一限制并评估已部署模型的脆弱性是至关重要的。本文在仅知道ImageNet域的情况下，提出了一种超越ImageNet攻击(BIA)来研究向黑盒域(未知分类任务)的可传递性。具体地说，我们利用生成模型来学习破坏输入图像的低层特征的对抗性函数。基于这一框架，我们进一步提出了两种变体，分别从数据和模型的角度来缩小源域和目标域之间的差距。在粗粒域和细粒域上的大量实验证明了我们提出的方法的有效性。值得注意的是，我们的方法平均比最先进的方法高出7.71%(对于粗粒度领域)和25.91%(对于细粒度领域)。我们的代码可在\url{https://github.com/qilong-zhang/Beyond-ImageNet-Attack}.获得



## **3. Adversarial Attack and Defense of YOLO Detectors in Autonomous Driving Scenarios**

自动驾驶场景中YOLO检测器的对抗性攻击与防御 cs.CV

7 pages, 3 figures

**SubmitDate**: 2022-02-10    [paper-pdf](http://arxiv.org/pdf/2202.04781v1)

**Authors**: Jung Im Choi, Qing Tian

**Abstracts**: Visual detection is a key task in autonomous driving, and it serves as one foundation for self-driving planning and control. Deep neural networks have achieved promising results in various computer vision tasks, but they are known to be vulnerable to adversarial attacks. A comprehensive understanding of deep visual detectors' vulnerability is required before people can improve their robustness. However, only a few adversarial attack/defense works have focused on object detection, and most of them employed only classification and/or localization losses, ignoring the objectness aspect. In this paper, we identify a serious objectness-related adversarial vulnerability in YOLO detectors and present an effective attack strategy aiming the objectness aspect of visual detection in autonomous vehicles. Furthermore, to address such vulnerability, we propose a new objectness-aware adversarial training approach for visual detection. Experiments show that the proposed attack targeting the objectness aspect is 45.17% and 43.50% more effective than those generated from classification and/or localization losses on the KITTI and COCO_traffic datasets, respectively. Also, the proposed adversarial defense approach can improve the detectors' robustness against objectness-oriented attacks by up to 21% and 12% mAP on KITTI and COCO_traffic, respectively.

摘要: 视觉检测是自动驾驶中的一项关键任务，是自动驾驶规划和控制的基础之一。深度神经网络在各种计算机视觉任务中取得了令人满意的结果，但众所周知，它们很容易受到对手的攻击。人们需要全面了解深度视觉检测器的脆弱性，才能提高其健壮性。然而，只有少数对抗性攻防研究集中在目标检测上，而且大多只采用分类和/或定位损失，而忽略了客观性方面。本文针对自主车辆视觉检测的客观性方面，识别出YOLO检测器中存在的一个与客观性相关的严重攻击漏洞，并提出了一种有效的攻击策略。此外，为了解决这种脆弱性，我们提出了一种新的基于客观性感知的对抗性视觉检测训练方法。实验表明，针对客观性方面的攻击比基于KITTI和COCO_TRAFFORM数据集的分类和/或定位丢失攻击分别提高了45.17%和43.50%。此外，本文提出的对抗性防御方法可以使检测器对面向对象攻击的鲁棒性分别提高21%和12%MAP在KITTI和COCO_TRAFFORMS上。



## **4. Layer-wise Regularized Adversarial Training using Layers Sustainability Analysis (LSA) framework**

基于层次可持续性分析(LSA)框架的分层正则化对抗性训练 cs.CV

Layers Sustainability Analysis (LSA) framework

**SubmitDate**: 2022-02-09    [paper-pdf](http://arxiv.org/pdf/2202.02626v2)

**Authors**: Mohammad Khalooei, Mohammad Mehdi Homayounpour, Maryam Amirmazlaghani

**Abstracts**: Deep neural network models are used today in various applications of artificial intelligence, the strengthening of which, in the face of adversarial attacks is of particular importance. An appropriate solution to adversarial attacks is adversarial training, which reaches a trade-off between robustness and generalization. This paper introduces a novel framework (Layer Sustainability Analysis (LSA)) for the analysis of layer vulnerability in a given neural network in the scenario of adversarial attacks. LSA can be a helpful toolkit to assess deep neural networks and to extend adversarial training approaches towards improving the sustainability of model layers via layer monitoring and analysis. The LSA framework identifies a list of Most Vulnerable Layers (MVL list) of a given network. The relative error, as a comparison measure, is used to evaluate the representation sustainability of each layer against adversarial attack inputs. The proposed approach for obtaining robust neural networks to fend off adversarial attacks is based on a layer-wise regularization (LR) over LSA proposal(s) for adversarial training (AT); i.e. the AT-LR procedure. AT-LR could be used with any benchmark adversarial attack to reduce the vulnerability of network layers and to improve conventional adversarial training approaches. The proposed idea performs well theoretically and experimentally for state-of-the-art multilayer perceptron and convolutional neural network architectures. Compared with the AT-LR and its corresponding base adversarial training, the classification accuracy of more significant perturbations increased by 16.35%, 21.79%, and 10.730% on Moon, MNIST, and CIFAR-10 benchmark datasets in comparison with the AT-LR and its corresponding base adversarial training, respectively. The LSA framework is available and published at https://github.com/khalooei/LSA.

摘要: 深度神经网络模型在当今人工智能的各种应用中都有应用，在面对敌意攻击时，加强深度神经网络模型的应用显得尤为重要。对抗性攻击的一个合适的解决方案是对抗性训练，它在鲁棒性和泛化之间达到了折衷。本文介绍了一种新的框架(层可持续性分析，LSA)，用于分析给定神经网络在敌意攻击情况下的层脆弱性。LSA可作为评估深层神经网络和扩展对抗性训练方法的有用工具包，以便通过层监控和分析来提高模型层的可持续性。LSA框架标识给定网络的最易受攻击的层的列表(MVL列表)。以相对误差作为比较指标，评价各层对敌方攻击输入的表示可持续性。所提出的获得鲁棒神经网络以抵御对手攻击的方法是基于基于LSA的对抗性训练(AT)方案的分层正则化(LR)，即AT-LR过程。AT-LR可以与任何基准对抗性攻击一起使用，以降低网络层的脆弱性，并改进传统的对抗性训练方法。对于最先进的多层感知器和卷积神经网络结构，所提出的思想在理论和实验上都表现良好。在MON、MNIST和CIFAR-10基准数据集上，与AT-LR及其对应的基础对抗训练相比，较显著扰动的分类准确率分别提高了16.35%、21.79%和10.730%。可以在https://github.com/khalooei/LSA.上获得并发布lsa框架。



## **5. IoTMonitor: A Hidden Markov Model-based Security System to Identify Crucial Attack Nodes in Trigger-action IoT Platforms**

IoTMonitor：基于隐马尔可夫模型的触发物联网平台关键攻击节点识别安全系统 cs.CR

This paper appears in the 2022 IEEE Wireless Communications and  Networking Conference (WCNC 2022). Personal use of this material is  permitted. Permission from IEEE must be obtained for all other uses

**SubmitDate**: 2022-02-09    [paper-pdf](http://arxiv.org/pdf/2202.04620v1)

**Authors**: Md Morshed Alam, Md Sajidul Islam Sajid, Weichao Wang, Jinpeng Wei

**Abstracts**: With the emergence and fast development of trigger-action platforms in IoT settings, security vulnerabilities caused by the interactions among IoT devices become more prevalent. The event occurrence at one device triggers an action in another device, which may eventually contribute to the creation of a chain of events in a network. Adversaries exploit the chain effect to compromise IoT devices and trigger actions of interest remotely just by injecting malicious events into the chain. To address security vulnerabilities caused by trigger-action scenarios, existing research efforts focus on the validation of the security properties of devices or verification of the occurrence of certain events based on their physical fingerprints on a device. We propose IoTMonitor, a security analysis system that discerns the underlying chain of event occurrences with the highest probability by observing a chain of physical evidence collected by sensors. We use the Baum-Welch algorithm to estimate transition and emission probabilities and the Viterbi algorithm to discern the event sequence. We can then identify the crucial nodes in the trigger-action sequence whose compromise allows attackers to reach their final goals. The experiment results of our designed system upon the PEEVES datasets show that we can rebuild the event occurrence sequence with high accuracy from the observations and identify the crucial nodes on the attack paths.

摘要: 随着物联网环境下触发式平台的出现和快速发展，物联网设备之间的交互导致的安全漏洞变得更加普遍。一台设备上发生的事件会触发另一台设备上的操作，这最终可能会导致在网络中创建一系列事件。攻击者只需将恶意事件注入链中，即可利用连锁反应危害物联网设备并远程触发感兴趣的操作。为了解决触发动作场景引起的安全漏洞，现有的研究工作集中在验证设备的安全属性或基于设备上的物理指纹来验证特定事件的发生。我们提出了IoTMonitor，这是一个安全分析系统，它通过观察传感器收集的物理证据链，以最高的概率识别事件发生的潜在链。我们使用Baum-Welch算法来估计转移概率和发射概率，使用Viterbi算法来识别事件序列。然后，我们可以确定触发-动作序列中的关键节点，这些节点的妥协使攻击者能够达到他们的最终目标。我们设计的系统在PEVES数据集上的实验结果表明，我们可以从观测数据中高精度地重建事件发生序列，并识别攻击路径上的关键节点。



## **6. False Memory Formation in Continual Learners Through Imperceptible Backdoor Trigger**

通过潜伏的后门触发器形成持续学习者的错误记忆 cs.LG

**SubmitDate**: 2022-02-09    [paper-pdf](http://arxiv.org/pdf/2202.04479v1)

**Authors**: Muhammad Umer, Robi Polikar

**Abstracts**: In this brief, we show that sequentially learning new information presented to a continual (incremental) learning model introduces new security risks: an intelligent adversary can introduce small amount of misinformation to the model during training to cause deliberate forgetting of a specific task or class at test time, thus creating "false memory" about that task. We demonstrate such an adversary's ability to assume control of the model by injecting "backdoor" attack samples to commonly used generative replay and regularization based continual learning approaches using continual learning benchmark variants of MNIST, as well as the more challenging SVHN and CIFAR 10 datasets. Perhaps most damaging, we show this vulnerability to be very acute and exceptionally effective: the backdoor pattern in our attack model can be imperceptible to human eye, can be provided at any point in time, can be added into the training data of even a single possibly unrelated task and can be achieved with as few as just 1\% of total training dataset of a single task.

摘要: 在这篇简短的文章中，我们展示了顺序学习提供给连续(增量)学习模型的新信息会带来新的安全风险：智能对手可能在训练期间向模型引入少量的错误信息，导致在测试时故意忘记特定任务或类，从而产生关于该任务的“错误记忆”。我们使用MNIST的持续学习基准变体，以及更具挑战性的SVHN和CIFAR10数据集，通过向常用的基于生成性回放和正则化的持续学习方法注入“后门”攻击样本，展示了这样的对手控制模型的能力。也许最具破坏性的是，我们发现这个漏洞是非常尖锐和特别有效的：我们的攻击模型中的后门模式可以是人眼看不见的，可以在任何时间点提供，可以添加到甚至是单个可能不相关的任务的训练数据中，并且可以仅使用单个任务的全部训练数据集的1\%就可以实现。



## **7. ARIBA: Towards Accurate and Robust Identification of Backdoor Attacks in Federated Learning**

Ariba：在联邦学习中实现对后门攻击的准确和鲁棒识别 cs.AI

17 pages, 11 figures

**SubmitDate**: 2022-02-09    [paper-pdf](http://arxiv.org/pdf/2202.04311v1)

**Authors**: Yuxi Mi, Jihong Guan, Shuigeng Zhou

**Abstracts**: The distributed nature and privacy-preserving characteristics of federated learning make it prone to the threat of poisoning attacks, especially backdoor attacks, where the adversary implants backdoors to misguide the model on certain attacker-chosen sub-tasks. In this paper, we present a novel method ARIBA to accurately and robustly identify backdoor attacks in federated learning. By empirical study, we observe that backdoor attacks are discernible by the filters of CNN layers. Based on this finding, we employ unsupervised anomaly detection to evaluate the pre-processed filters and calculate an anomaly score for each client. We then identify the most suspicious clients according to their anomaly scores. Extensive experiments are conducted, which show that our method ARIBA can effectively and robustly defend against multiple state-of-the-art attacks without degrading model performance.

摘要: 联邦学习的分布式性质和隐私保护特性使其容易受到中毒攻击的威胁，特别是后门攻击，其中对手植入后门来在攻击者选择的某些子任务上误导模型。本文提出了一种新的方法ARIBA来准确、鲁棒地识别联邦学习中的后门攻击。通过实证研究，我们观察到通过CNN层的过滤可以识别出后门攻击。基于这一发现，我们使用非监督异常检测来评估预处理后的过滤器，并计算每个客户端的异常评分。然后，我们根据他们的异常得分来识别最可疑的客户。大量实验表明，ARIBA方法在不降低模型性能的前提下，能够有效、稳健地防御多种最先进的攻击。



## **8. Adversarial Detection without Model Information**

无模型信息的对抗性检测 cs.CV

**SubmitDate**: 2022-02-09    [paper-pdf](http://arxiv.org/pdf/2202.04271v1)

**Authors**: Abhishek Moitra, Youngeun Kim, Priyadarshini Panda

**Abstracts**: Most prior state-of-the-art adversarial detection works assume that the underlying vulnerable model is accessible, i,e., the model can be trained or its outputs are visible. However, this is not a practical assumption due to factors like model encryption, model information leakage and so on. In this work, we propose a model independent adversarial detection method using a simple energy function to distinguish between adversarial and natural inputs. We train a standalone detector independent of the underlying model, with sequential layer-wise training to increase the energy separation corresponding to natural and adversarial inputs. With this, we perform energy distribution-based adversarial detection. Our method achieves state-of-the-art detection performance (ROC-AUC > 0.9) across a wide range of gradient, score and decision-based adversarial attacks on CIFAR10, CIFAR100 and TinyImagenet datasets. Compared to prior approaches, our method requires ~10-100x less number of operations and parameters for adversarial detection. Further, we show that our detection method is transferable across different datasets and adversarial attacks. For reproducibility, we provide code in the supplementary material.

摘要: 大多数现有的对抗性检测工作都假设潜在的易受攻击模型是可访问的，即模型可以被训练或其输出是可见的。然而，由于模型加密、模型信息泄露等因素，这并不是一个实际的假设。在这项工作中，我们提出了一种模型无关的敌意检测方法，使用一个简单的能量函数来区分敌意输入和自然输入。我们训练一个独立的检测器，独立于底层模型，通过顺序的分层训练来增加与自然和对抗性输入相对应的能量分离。在此基础上，我们进行了基于能量分布的敌意检测。我们的方法在CIFAR10、CIFAR100和TinyImagenet数据集上获得了最先进的检测性能(ROC-AUC>0.9)，涵盖了广泛的梯度、得分和基于决策的对手攻击。与以前的方法相比，我们的方法需要的操作次数和参数减少了约10-100倍。此外，我们还证明了我们的检测方法可以在不同的数据集和敌意攻击之间传输。为了重现性，我们在补充材料中提供了代码。



## **9. Towards Compositional Adversarial Robustness: Generalizing Adversarial Training to Composite Semantic Perturbations**

走向成分对抗稳健性：将对抗训练推广到复合语义扰动 cs.CV

**SubmitDate**: 2022-02-09    [paper-pdf](http://arxiv.org/pdf/2202.04235v1)

**Authors**: Yun-Yun Tsai, Lei Hsiung, Pin-Yu Chen, Tsung-Yi Ho

**Abstracts**: Model robustness against adversarial examples of single perturbation type such as the $\ell_{p}$-norm has been widely studied, yet its generalization to more realistic scenarios involving multiple semantic perturbations and their composition remains largely unexplored. In this paper, we firstly propose a novel method for generating composite adversarial examples. By utilizing component-wise projected gradient descent and automatic attack-order scheduling, our method can find the optimal attack composition. We then propose \textbf{generalized adversarial training} (\textbf{GAT}) to extend model robustness from $\ell_{p}$-norm to composite semantic perturbations, such as the combination of Hue, Saturation, Brightness, Contrast, and Rotation. The results on ImageNet and CIFAR-10 datasets show that GAT can be robust not only to any single attack but also to any combination of multiple attacks. GAT also outperforms baseline $\ell_{\infty}$-norm bounded adversarial training approaches by a significant margin.

摘要: 针对单一扰动类型的对抗性实例(如$ellp-范数)的模型鲁棒性已被广泛研究，但其对涉及多个语义扰动及其组成的更现实场景的推广仍未得到很大程度的探索。在本文中，我们首先提出了一种生成复合对抗性实例的新方法。该方法通过基于组件的投影梯度下降和攻击顺序的自动调度，能够找到最优的攻击组合。然后，我们提出了textbf(广义对抗性训练)(textbf{GAT})来将模型鲁棒性从$ellp}$范数扩展到复合语义扰动，如色调、饱和度、亮度、对比度和旋转的组合。在ImageNet和CIFAR-10数据集上的结果表明，GAT不仅对任何单一攻击都具有鲁棒性，而且对多个攻击的任何组合都具有鲁棒性。GAT的性能也大大超过了基线$-范数有界的对抗性训练方法。



## **10. Defeating Misclassification Attacks Against Transfer Learning**

抵抗针对迁移学习的误分类攻击 cs.LG

This paper has been published in IEEE Transactions on Dependable and  Secure Computing.  https://doi.ieeecomputersociety.org/10.1109/TDSC.2022.3144988

**SubmitDate**: 2022-02-09    [paper-pdf](http://arxiv.org/pdf/1908.11230v4)

**Authors**: Bang Wu, Shuo Wang, Xingliang Yuan, Cong Wang, Carsten Rudolph, Xiangwen Yang

**Abstracts**: Transfer learning is prevalent as a technique to efficiently generate new models (Student models) based on the knowledge transferred from a pre-trained model (Teacher model). However, Teacher models are often publicly available for sharing and reuse, which inevitably introduces vulnerability to trigger severe attacks against transfer learning systems. In this paper, we take a first step towards mitigating one of the most advanced misclassification attacks in transfer learning. We design a distilled differentiator via activation-based network pruning to enervate the attack transferability while retaining accuracy. We adopt an ensemble structure from variant differentiators to improve the defence robustness. To avoid the bloated ensemble size during inference, we propose a two-phase defence, in which inference from the Student model is firstly performed to narrow down the candidate differentiators to be assembled, and later only a small, fixed number of them can be chosen to validate clean or reject adversarial inputs effectively. Our comprehensive evaluations on both large and small image recognition tasks confirm that the Student models with our defence of only 5 differentiators are immune to over 90% of the adversarial inputs with an accuracy loss of less than 10%. Our comparison also demonstrates that our design outperforms prior problematic defences.

摘要: 迁移学习作为一种基于从预先训练的模型(教师模型)传输的知识有效地生成新模型(学生模型)的技术而流行。然而，教师模型通常是公开可供共享和重用的，这不可避免地会引入漏洞，从而引发对迁移学习系统的严重攻击。在本文中，我们向减轻迁移学习中最高级的错误分类攻击之一迈出了第一步。我们通过基于激活的网络剪枝设计了一种提炼的微分器，在保持准确性的同时削弱了攻击的可传递性。我们采用不同微分器的集成结构来提高防御鲁棒性。为了避免推理过程中集成规模的膨胀，我们提出了一种两阶段防御方法，首先从学生模型中进行推理，以缩小待组装的候选微分算子的范围，然后只能选择少量固定数量的微分算子来有效地验证干净或拒绝对手输入。我们在大小图像识别任务上的综合评估证实，仅有5个微分因子的学生模型对90%以上的敌意输入具有免疫力，准确率损失小于10%。我们的比较还表明，我们的设计比以前的有问题的防御性能更好。



## **11. Ontology-based Attack Graph Enrichment**

基于本体的攻击图充实 cs.CR

18 pages, 3 figures, 1 table, conference paper (TIEMS Annual  Conference, December 2021, Paris, France)

**SubmitDate**: 2022-02-08    [paper-pdf](http://arxiv.org/pdf/2202.04016v1)

**Authors**: Kéren Saint-Hilaire, Frédéric Cuppens, Nora Cuppens, Joaquin Garcia-Alfaro

**Abstracts**: Attack graphs provide a representation of possible actions that adversaries can perpetrate to attack a system. They are used by cybersecurity experts to make decisions, e.g., to decide remediation and recovery plans. Different approaches can be used to build such graphs. We focus on logical attack graphs, based on predicate logic, to define the causality of adversarial actions. Since networks and vulnerabilities are constantly changing (e.g., new applications get installed on system devices, updated services get publicly exposed, etc.), we propose to enrich the attack graph generation approach with a semantic augmentation post-processing of the predicates. Graphs are now mapped to monitoring alerts confirming successful attack actions and updated according to network and vulnerability changes. As a result, predicates get periodically updated, based on attack evidences and ontology enrichment. This allows to verify whether changes lead the attacker to the initial goals or to cause further damage to the system not anticipated in the initial graphs. We illustrate the approach under the specific domain of cyber-physical security affecting smart cities. We validate the approach using existing tools and ontologies.

摘要: 攻击图提供了攻击者可能实施的攻击系统的操作的表示。网络安全专家使用它们来做出决策，例如，决定补救和恢复计划。可以使用不同的方法来构建这样的图。我们将重点放在逻辑攻击图上，基于谓词逻辑来定义敌对行为的因果关系。由于网络和漏洞是不断变化的(例如，新的应用程序安装在系统设备上，更新的服务被公开暴露等)，我们建议通过谓词的语义增强后处理来丰富攻击图生成方法。图形现在映射到监控警报，以确认攻击操作成功，并根据网络和漏洞更改进行更新。因此，谓词会根据攻击证据和本体丰富进行定期更新。这样可以验证更改是否会导致攻击者达到初始目标，或者是否会对系统造成初始图表中未预料到的进一步损坏。我们从影响智慧城市的网络物理安全这一特定领域出发，阐述了该方法。我们使用现有的工具和本体来验证该方法。



## **12. Verification-Aided Deep Ensemble Selection**

辅助验证的深度集成选择 cs.LG

**SubmitDate**: 2022-02-08    [paper-pdf](http://arxiv.org/pdf/2202.03898v1)

**Authors**: Guy Amir, Guy Katz, Michael Schapira

**Abstracts**: Deep neural networks (DNNs) have become the technology of choice for realizing a variety of complex tasks. However, as highlighted by many recent studies, even an imperceptible perturbation to a correctly classified input can lead to misclassification by a DNN. This renders DNNs vulnerable to strategic input manipulations by attackers, and also prone to oversensitivity to environmental noise.   To mitigate this phenomenon, practitioners apply joint classification by an ensemble of DNNs. By aggregating the classification outputs of different individual DNNs for the same input, ensemble-based classification reduces the risk of misclassifications due to the specific realization of the stochastic training process of any single DNN. However, the effectiveness of a DNN ensemble is highly dependent on its members not simultaneously erring on many different inputs.   In this case study, we harness recent advances in DNN verification to devise a methodology for identifying ensemble compositions that are less prone to simultaneous errors, even when the input is adversarially perturbed -- resulting in more robustly-accurate ensemble-based classification.   Our proposed framework uses a DNN verifier as a backend, and includes heuristics that help reduce the high complexity of directly verifying ensembles. More broadly, our work puts forth a novel universal objective for formal verification that can potentially improve the robustness of real-world, deep-learning-based systems across a variety of application domains.

摘要: 深度神经网络(DNNs)已经成为实现各种复杂任务的首选技术。然而，正如最近的许多研究所强调的那样，即使是对正确分类的输入进行了不可察觉的扰动，也可能导致DNN的错误分类。这使得DNN容易受到攻击者的战略性输入操纵，并且容易对环境噪声过于敏感。为了缓解这一现象，从业者应用DNN集合的联合分类。通过聚合同一输入的不同个体DNN的分类输出，基于集成的分类降低了由于任意单个DNN的随机训练过程的具体实现而导致的误分类风险。然而，DNN合奏的有效性高度依赖于其成员，而不是同时在许多不同的输入上出错。在这个案例研究中，我们利用DNN验证方面的最新进展来设计一种方法，用于识别不太容易同时出错的组合成分，即使输入受到相反的干扰-导致基于组合的分类更加稳健准确。我们提出的框架使用DNN验证器作为后端，并包括有助于降低直接验证集成的高复杂度的启发式算法。更广泛地说，我们的工作为形式验证提出了一个新的通用目标，可以潜在地提高现实世界中基于深度学习的系统在各种应用领域的健壮性。



## **13. Invertible Tabular GANs: Killing Two Birds with OneStone for Tabular Data Synthesis**

可逆表格甘斯：表格数据合成的一石二鸟 cs.LG

19 pages

**SubmitDate**: 2022-02-08    [paper-pdf](http://arxiv.org/pdf/2202.03636v1)

**Authors**: Jaehoon Lee, Jihyeon Hyeong, Jinsung Jeon, Noseong Park, Jihoon Cho

**Abstracts**: Tabular data synthesis has received wide attention in the literature. This is because available data is often limited, incomplete, or cannot be obtained easily, and data privacy is becoming increasingly important. In this work, we present a generalized GAN framework for tabular synthesis, which combines the adversarial training of GANs and the negative log-density regularization of invertible neural networks. The proposed framework can be used for two distinctive objectives. First, we can further improve the synthesis quality, by decreasing the negative log-density of real records in the process of adversarial training. On the other hand, by increasing the negative log-density of real records, realistic fake records can be synthesized in a way that they are not too much close to real records and reduce the chance of potential information leakage. We conduct experiments with real-world datasets for classification, regression, and privacy attacks. In general, the proposed method demonstrates the best synthesis quality (in terms of task-oriented evaluation metrics, e.g., F1) when decreasing the negative log-density during the adversarial training. If increasing the negative log-density, our experimental results show that the distance between real and fake records increases, enhancing robustness against privacy attacks.

摘要: 表格数据合成在文献中受到了广泛的关注。这是因为可用的数据通常是有限的、不完整的或不容易获得的，而且数据隐私变得越来越重要。在这项工作中，我们提出了一个用于表格综合的广义GAN框架，该框架结合了GANS的对抗性训练和可逆神经网络的负对数密度正则化。建议的框架可用于两个不同的目标。首先，通过降低对抗性训练过程中真实记录的负对数密度，进一步提高综合质量。另一方面，通过增加真实记录的负对数密度，可以合成出与真实记录不太接近的真实假记录，减少潜在信息泄露的机会。我们使用真实世界的数据集进行分类、回归和隐私攻击的实验。一般而言，在对抗性训练过程中，当降低负对数密度时，该方法表现出最佳的综合质量(就面向任务的评估指标而言，例如F1)。实验结果表明，如果增加负的日志密度，真实记录和虚假记录之间的距离会增大，从而增强了对隐私攻击的鲁棒性。



## **14. A Survey on Poisoning Attacks Against Supervised Machine Learning**

针对有监督机器学习的中毒攻击研究综述 cs.CR

**SubmitDate**: 2022-02-08    [paper-pdf](http://arxiv.org/pdf/2202.02510v2)

**Authors**: Wenjun Qiu

**Abstracts**: With the rise of artificial intelligence and machine learning in modern computing, one of the major concerns regarding such techniques is to provide privacy and security against adversaries. We present this survey paper to cover the most representative papers in poisoning attacks against supervised machine learning models. We first provide a taxonomy to categorize existing studies and then present detailed summaries for selected papers. We summarize and compare the methodology and limitations of existing literature. We conclude this paper with potential improvements and future directions to further exploit and prevent poisoning attacks on supervised models. We propose several unanswered research questions to encourage and inspire researchers for future work.

摘要: 随着人工智能和机器学习在现代计算中的兴起，关于这类技术的一个主要问题是提供隐私和安全，以抵御对手。我们提出的这份调查报告涵盖了针对有监督机器学习模型的中毒攻击中最具代表性的论文。我们首先提供一个分类法来对现有的研究进行分类，然后对选定的论文进行详细的总结。我们对现有文献的研究方法和局限性进行了总结和比较。最后，我们对本文进行了总结，并对进一步开发和防止对监督模型的中毒攻击提出了可能的改进和未来的方向。我们提出了几个尚未回答的研究问题，以鼓励和激励研究人员未来的工作。



## **15. Sparse-RS: a versatile framework for query-efficient sparse black-box adversarial attacks**

Sparse-RS：一种通用的查询高效稀疏黑盒攻击框架 cs.LG

Accepted at AAAI 2022. This version contains considerably extended  results in the L0 threat model

**SubmitDate**: 2022-02-08    [paper-pdf](http://arxiv.org/pdf/2006.12834v3)

**Authors**: Francesco Croce, Maksym Andriushchenko, Naman D. Singh, Nicolas Flammarion, Matthias Hein

**Abstracts**: We propose a versatile framework based on random search, Sparse-RS, for score-based sparse targeted and untargeted attacks in the black-box setting. Sparse-RS does not rely on substitute models and achieves state-of-the-art success rate and query efficiency for multiple sparse attack models: $l_0$-bounded perturbations, adversarial patches, and adversarial frames. The $l_0$-version of untargeted Sparse-RS outperforms all black-box and even all white-box attacks for different models on MNIST, CIFAR-10, and ImageNet. Moreover, our untargeted Sparse-RS achieves very high success rates even for the challenging settings of $20\times20$ adversarial patches and $2$-pixel wide adversarial frames for $224\times224$ images. Finally, we show that Sparse-RS can be applied to generate targeted universal adversarial patches where it significantly outperforms the existing approaches. The code of our framework is available at https://github.com/fra31/sparse-rs.

摘要: 针对黑盒环境下基于分数的稀疏目标攻击和非目标攻击，我们提出了一个基于随机搜索的通用框架Sparse-RS。Sparse-RS不依赖替身模型，对多种稀疏攻击模型($l_0$有界扰动、敌意补丁和敌意帧)具有最高的成功率和查询效率。对于MNIST、CIFAR-10和ImageNet上的不同型号，$l_0$-版本的非目标稀疏-RS的性能优于所有黑盒甚至所有白盒攻击。此外，我们的非定向稀疏-RS即使在$20\x 20$对抗性补丁和$2$像素宽的对抗性帧的$224\x 224$图像的挑战性设置下也能获得非常高的成功率。最后，我们证明了稀疏-RS可以用来生成目标通用的对抗性补丁，其性能明显优于现有的方法。我们框架的代码可以在https://github.com/fra31/sparse-rs.上找到



## **16. Evaluating Robustness of Cooperative MARL: A Model-based Approach**

评估协作MAIL的健壮性：一种基于模型的方法 cs.LG

**SubmitDate**: 2022-02-07    [paper-pdf](http://arxiv.org/pdf/2202.03558v1)

**Authors**: Nhan H. Pham, Lam M. Nguyen, Jie Chen, Hoang Thanh Lam, Subhro Das, Tsui-Wei Weng

**Abstracts**: In recent years, a proliferation of methods were developed for cooperative multi-agent reinforcement learning (c-MARL). However, the robustness of c-MARL agents against adversarial attacks has been rarely explored. In this paper, we propose to evaluate the robustness of c-MARL agents via a model-based approach. Our proposed formulation can craft stronger adversarial state perturbations of c-MARL agents(s) to lower total team rewards more than existing model-free approaches. In addition, we propose the first victim-agent selection strategy which allows us to develop even stronger adversarial attack. Numerical experiments on multi-agent MuJoCo benchmarks illustrate the advantage of our approach over other baselines. The proposed model-based attack consistently outperforms other baselines in all tested environments.

摘要: 近年来，协作多智能体强化学习(c-MARL)方法层出不穷。然而，c-Marl代理抵抗敌意攻击的健壮性很少被研究。在本文中，我们提出了一种基于模型的方法来评估c-Marl代理的健壮性。与现有的无模型方法相比，我们提出的公式可以制作更强的c-Marl代理的对抗性状态扰动，以降低团队总奖励。此外，我们还提出了第一种受害者-代理选择策略，使我们能够开发出更强的对抗性攻击。在多智能体MuJoCo基准上的数值实验表明了我们的方法相对于其他基线的优势。在所有测试环境中，建议的基于模型的攻击始终优于其他基准。



## **17. Deletion Inference, Reconstruction, and Compliance in Machine (Un)Learning**

机器(UN)学习中的删除推理、重构和顺应性 cs.LG

Full version of a paper appearing in the 22nd Privacy Enhancing  Technologies Symposium (PETS 2022)

**SubmitDate**: 2022-02-07    [paper-pdf](http://arxiv.org/pdf/2202.03460v1)

**Authors**: Ji Gao, Sanjam Garg, Mohammad Mahmoody, Prashant Nalini Vasudevan

**Abstracts**: Privacy attacks on machine learning models aim to identify the data that is used to train such models. Such attacks, traditionally, are studied on static models that are trained once and are accessible by the adversary. Motivated to meet new legal requirements, many machine learning methods are recently extended to support machine unlearning, i.e., updating models as if certain examples are removed from their training sets, and meet new legal requirements. However, privacy attacks could potentially become more devastating in this new setting, since an attacker could now access both the original model before deletion and the new model after the deletion. In fact, the very act of deletion might make the deleted record more vulnerable to privacy attacks.   Inspired by cryptographic definitions and the differential privacy framework, we formally study privacy implications of machine unlearning. We formalize (various forms of) deletion inference and deletion reconstruction attacks, in which the adversary aims to either identify which record is deleted or to reconstruct (perhaps part of) the deleted records. We then present successful deletion inference and reconstruction attacks for a variety of machine learning models and tasks such as classification, regression, and language models. Finally, we show that our attacks would provably be precluded if the schemes satisfy (variants of) Deletion Compliance (Garg, Goldwasser, and Vasudevan, Eurocrypt' 20).

摘要: 针对机器学习模型的隐私攻击旨在识别用于训练此类模型的数据。传统上，这类攻击是在静电模型上研究的，这些模型只训练一次，对手可以访问。为了满足新的法律要求，许多机器学习方法最近被扩展到支持机器遗忘，即更新模型，就像从训练集中删除某些示例一样，并满足新的法律要求。然而，在这种新设置下，隐私攻击可能会变得更具破坏性，因为攻击者现在既可以访问删除前的原始模型，也可以访问删除后的新模型。事实上，删除操作本身可能会使删除的记录更容易受到隐私攻击。在密码学定义和差分隐私框架的启发下，我们正式研究了机器遗忘的隐私含义。我们形式化(各种形式的)删除推理和删除重构攻击，在这些攻击中，敌手的目标要么是识别哪条记录被删除，要么是重构(可能是一部分)被删除的记录。然后，我们针对各种机器学习模型和任务，如分类、回归和语言模型，提出了成功的删除推理和重构攻击。最后，我们证明了如果方案满足删除遵从性(Garg，Goldwasser和Vasudevan，Eurocrypt‘20)(变体)，我们的攻击将被证明是被排除的。



## **18. Bilevel Optimization with a Lower-level Contraction: Optimal Sample Complexity without Warm-Start**

低水平收缩的双层优化：无热启动的最优样本复杂度 stat.ML

30 pages

**SubmitDate**: 2022-02-07    [paper-pdf](http://arxiv.org/pdf/2202.03397v1)

**Authors**: Riccardo Grazzi, Massimiliano Pontil, Saverio Salzo

**Abstracts**: We analyze a general class of bilevel problems, in which the upper-level problem consists in the minimization of a smooth objective function and the lower-level problem is to find the fixed point of a smooth contraction map. This type of problems include instances of meta-learning, hyperparameter optimization and data poisoning adversarial attacks. Several recent works have proposed algorithms which warm-start the lower-level problem, i.e. they use the previous lower-level approximate solution as a staring point for the lower-level solver. This warm-start procedure allows one to improve the sample complexity in both the stochastic and deterministic settings, achieving in some cases the order-wise optimal sample complexity. We show that without warm-start, it is still possible to achieve order-wise optimal and near-optimal sample complexity for the stochastic and deterministic settings, respectively. In particular, we propose a simple method which uses stochastic fixed point iterations at the lower-level and projected inexact gradient descent at the upper-level, that reaches an $\epsilon$-stationary point using $O(\epsilon^{-2})$ and $\tilde{O}(\epsilon^{-1})$ samples for the stochastic and the deterministic setting, respectively. Compared to methods using warm-start, ours is better suited for meta-learning and yields a simpler analysis that does not need to study the coupled interactions between the upper-level and lower-level iterates.

摘要: 我们分析了一类一般的两层问题，其中上层问题在于光滑目标函数的最小化，下层问题是寻找光滑压缩映射的不动点。这类问题包括元学习、超参数优化和数据中毒攻击等实例。最近的一些工作已经提出了暖启动下层问题的算法，即使用先前的下层近似解作为下层求解器的起始点。这种热启动过程允许人们在随机和确定性设置下改善样本复杂度，在某些情况下实现顺序最优的样本复杂度。我们证明了在没有热启动的情况下，对于随机设置和确定性设置，仍然可以分别获得按顺序最优和接近最优的样本复杂度。特别地，我们提出了一种简单的方法，它在下层使用随机不动点迭代，在上层使用投影的不精确梯度下降，在随机和确定性设置下分别使用$O(epsilon^{-2})$和$tilde{O}(epsilon^{-1})$样本达到$ε-稳定点。与使用热启动的方法相比，我们的方法更适合于元学习，并且产生了更简单的分析，不需要研究上层和下层迭代之间的耦合作用。



## **19. Membership Inference Attacks and Defenses in Neural Network Pruning**

神经网络修剪中的隶属度推理攻击与防御 cs.CR

This paper has been conditionally accepted to USENIX Security  Symposium 2022. This is an extended version

**SubmitDate**: 2022-02-07    [paper-pdf](http://arxiv.org/pdf/2202.03335v1)

**Authors**: Xiaoyong Yuan, Lan Zhang

**Abstracts**: Neural network pruning has been an essential technique to reduce the computation and memory requirements for using deep neural networks for resource-constrained devices. Most existing research focuses primarily on balancing the sparsity and accuracy of a pruned neural network by strategically removing insignificant parameters and retraining the pruned model. Such efforts on reusing training samples pose serious privacy risks due to increased memorization, which, however, has not been investigated yet.   In this paper, we conduct the first analysis of privacy risks in neural network pruning. Specifically, we investigate the impacts of neural network pruning on training data privacy, i.e., membership inference attacks. We first explore the impact of neural network pruning on prediction divergence, where the pruning process disproportionately affects the pruned model's behavior for members and non-members. Meanwhile, the influence of divergence even varies among different classes in a fine-grained manner. Enlighten by such divergence, we proposed a self-attention membership inference attack against the pruned neural networks. Extensive experiments are conducted to rigorously evaluate the privacy impacts of different pruning approaches, sparsity levels, and adversary knowledge. The proposed attack shows the higher attack performance on the pruned models when compared with eight existing membership inference attacks. In addition, we propose a new defense mechanism to protect the pruning process by mitigating the prediction divergence based on KL-divergence distance, whose effectiveness has been experimentally demonstrated to effectively mitigate the privacy risks while maintaining the sparsity and accuracy of the pruned models.

摘要: 对于资源受限的设备，为了减少对深层神经网络的计算和存储需求，神经网络修剪已经成为一项重要的技术。现有的大多数研究主要集中在通过策略性地去除无关紧要的参数和重新训练修剪后的模型来平衡修剪神经网络的稀疏性和准确性。这种重复使用训练样本的努力由于增加了记忆而带来了严重的隐私风险，然而，这一点尚未得到调查。本文首先对神经网络修剪中的隐私风险进行了分析。具体地说，我们研究了神经网络修剪对训练数据隐私的影响，即成员推理攻击。我们首先探讨了神经网络修剪对预测发散的影响，其中修剪过程不成比例地影响修剪后的模型对成员和非成员的行为。同时，分歧的影响甚至在不同的阶层之间也有细微的差异。受这种分歧的启发，我们提出了一种针对修剪后的神经网络的自注意成员推理攻击。广泛的实验被用来严格评估不同的修剪方法、稀疏程度和敌意知识对隐私的影响。与现有的8种成员推理攻击相比，该攻击在剪枝模型上表现出更高的攻击性能。此外，我们还提出了一种新的防御机制来保护剪枝过程，通过减少基于KL-发散距离的预测发散来保护剪枝过程，实验证明该机制在保持剪枝模型的稀疏性和准确性的同时，有效地缓解了隐私风险。



## **20. On The Empirical Effectiveness of Unrealistic Adversarial Hardening Against Realistic Adversarial Attacks**

论非现实对抗硬化对抗现实对抗攻击的经验有效性 cs.LG

**SubmitDate**: 2022-02-07    [paper-pdf](http://arxiv.org/pdf/2202.03277v1)

**Authors**: Salijona Dyrmishi, Salah Ghamizi, Thibault Simonetto, Yves Le Traon, Maxime Cordy

**Abstracts**: While the literature on security attacks and defense of Machine Learning (ML) systems mostly focuses on unrealistic adversarial examples, recent research has raised concern about the under-explored field of realistic adversarial attacks and their implications on the robustness of real-world systems. Our paper paves the way for a better understanding of adversarial robustness against realistic attacks and makes two major contributions. First, we conduct a study on three real-world use cases (text classification, botnet detection, malware detection)) and five datasets in order to evaluate whether unrealistic adversarial examples can be used to protect models against realistic examples. Our results reveal discrepancies across the use cases, where unrealistic examples can either be as effective as the realistic ones or may offer only limited improvement. Second, to explain these results, we analyze the latent representation of the adversarial examples generated with realistic and unrealistic attacks. We shed light on the patterns that discriminate which unrealistic examples can be used for effective hardening. We release our code, datasets and models to support future research in exploring how to reduce the gap between unrealistic and realistic adversarial attacks.

摘要: 虽然关于机器学习系统安全攻击和防御的文献大多集中在不现实的对抗性例子上，但最近的研究已经引起了对现实对抗性攻击这一未被充分探索的领域及其对现实世界系统健壮性的影响的关注。我们的论文为更好地理解对抗现实攻击的鲁棒性铺平了道路，并做出了两个主要贡献。首先，我们在三个真实世界的用例(文本分类、僵尸网络检测、恶意软件检测)和五个数据集上进行了研究，以评估不现实的对抗性示例是否可以用来保护模型免受现实示例的影响。我们的结果揭示了用例之间的差异，在这些用例中，不切实际的示例可能与现实的示例一样有效，或者可能只提供有限的改进。其次，为了解释这些结果，我们分析了现实攻击和非现实攻击产生的对抗性例子的潜在表征。我们阐明了区分哪些不切实际的例子可以用于有效强化的模式。我们发布了我们的代码、数据集和模型，以支持未来的研究，探索如何缩小不切实际和现实的对抗性攻击之间的差距。



## **21. Strong Converse Theorem for Source Encryption under Side-Channel Attacks**

旁路攻击下源加密的强逆定理 cs.IT

9 pages, 6 figures. The short version of this paper was submitted to  ISIT2022, arXiv admin note: text overlap with arXiv:1801.02563

**SubmitDate**: 2022-02-07    [paper-pdf](http://arxiv.org/pdf/2201.11670v3)

**Authors**: Yasutada Oohama, Bagus Santoso

**Abstracts**: We are interested in investigating the security of source encryption with a symmetric key under side-channel attacks. In this paper, we propose a general framework of source encryption with a symmetric key under the side-channel attacks, which applies to \emph{any} source encryption with a symmetric key and \emph{any} kind of side-channel attacks targeting the secret key. We also propose a new security criterion for strong secrecy under side-channel attacks, which is a natural extension of mutual information, i.e., \emph{the maximum conditional mutual information between the plaintext and the ciphertext given the adversarial key leakage, where the maximum is taken over all possible plaintext distribution}. Under this new criterion, we successfully formulate the rate region, which serves as both necessary and sufficient conditions to have secure transmission even under side-channel attacks. Furthermore, we also prove another theoretical result on our new security criterion, which might be interesting in its own right: in the case of the discrete memoryless source, no perfect secrecy under side-channel attacks in the standard security criterion, i.e., the ordinary mutual information, is achievable without achieving perfect secrecy in this new security criterion, although our new security criterion is more strict than the standard security criterion.

摘要: 我们感兴趣的是研究在旁信道攻击下使用对称密钥的源加密的安全性。本文提出了一种在旁路攻击下对称密钥源加密的通用框架，该框架适用于对称密钥源加密和针对密钥的旁路攻击。我们还提出了一种新的边信道攻击下强保密性的安全准则，它是互信息的自然扩展，即在密钥泄露的情况下明文和密文之间的最大条件互信息，其中最大值取在所有可能的明文分布上。在这一新准则下，我们成功地给出了码率域的表达式，该码率域既是安全传输的充要条件，又是旁路攻击下安全传输的充分必要条件。此外，我们还证明了关于我们的新安全准则的另一个理论结果，它本身可能很有趣：在离散无记忆信源的情况下，尽管我们的新安全准则比标准安全准则更严格，但是在标准安全准则(即普通互信息)下，如果不达到完全保密，就不可能实现标准安全准则中的旁路攻击下的完全保密性。



## **22. More is Better (Mostly): On the Backdoor Attacks in Federated Graph Neural Networks**

越多越好(多数情况下)：联邦图神经网络中的后门攻击 cs.CR

18 pages, 11 figures

**SubmitDate**: 2022-02-07    [paper-pdf](http://arxiv.org/pdf/2202.03195v1)

**Authors**: Jing Xu, Rui Wang, Kaitai Liang, Stjepan Picek

**Abstracts**: Graph Neural Networks (GNNs) are a class of deep learning-based methods for processing graph domain information. GNNs have recently become a widely used graph analysis method due to their superior ability to learn representations for complex graph data. However, due to privacy concerns and regulation restrictions, centralized GNNs can be difficult to apply to data-sensitive scenarios. Federated learning (FL) is an emerging technology developed for privacy-preserving settings when several parties need to train a shared global model collaboratively. Although many research works have applied FL to train GNNs (Federated GNNs), there is no research on their robustness to backdoor attacks.   This paper bridges this gap by conducting two types of backdoor attacks in Federated GNNs: centralized backdoor attacks (CBA) and distributed backdoor attacks (DBA). CBA is conducted by embedding the same global trigger during training for every malicious party, while DBA is conducted by decomposing a global trigger into separate local triggers and embedding them into the training dataset of different malicious parties, respectively. Our experiments show that the DBA attack success rate is higher than CBA in almost all evaluated cases, while rarely, the DBA attack performance is close to CBA. For CBA, the attack success rate of all local triggers is similar to the global trigger even if the training set of the adversarial party is embedded with the global trigger. To further explore the properties of two backdoor attacks in Federated GNNs, we evaluate the attack performance for different trigger sizes, poisoning intensities, and trigger densities, with trigger density being the most influential.

摘要: 图神经网络(GNNs)是一类基于深度学习的图域信息处理方法。由于其优越的学习复杂图形数据表示的能力，GNN最近已成为一种广泛使用的图形分析方法。然而，由于隐私问题和法规限制，集中式GNN可能很难应用于数据敏感场景。联合学习(FL)是一种新兴的技术，是为保护隐私而开发的，当多个参与方需要协作训练共享的全局模型时，该技术是为保护隐私而开发的。虽然许多研究工作已经将FL用于训练GNN(Federated GNNs)，但还没有关于其对后门攻击的健壮性的研究。本文通过在联邦GNN中实施两种类型的后门攻击来弥合这一差距：集中式后门攻击(CBA)和分布式后门攻击(DBA)。CBA是通过在每个恶意方的训练过程中嵌入相同的全局触发器来进行的，而DBA是通过将全局触发器分解为单独的局部触发器并将它们分别嵌入到不同恶意方的训练数据集中来进行的。实验表明，DBA攻击成功率几乎都高于CBA，但很少有DBA攻击性能接近CBA。对于CBA，即使敌方的训练集嵌入了全局触发器，所有局部触发器的攻击成功率也与全局触发器相似。为了进一步研究联邦GNN中两种后门攻击的特性，我们评估了不同触发大小、中毒强度和触发密度下的攻击性能，其中触发密度对攻击性能的影响最大。



## **23. Transformers in Self-Supervised Monocular Depth Estimation with Unknown Camera Intrinsics**

未知摄像机特征条件下自监督单目深度估计中的变压器 cs.CV

Published in 17th International Conference on Computer Vision Theory  and Applications (VISAP, 2022)

**SubmitDate**: 2022-02-07    [paper-pdf](http://arxiv.org/pdf/2202.03131v1)

**Authors**: Arnav Varma, Hemang Chawla, Bahram Zonooz, Elahe Arani

**Abstracts**: The advent of autonomous driving and advanced driver assistance systems necessitates continuous developments in computer vision for 3D scene understanding. Self-supervised monocular depth estimation, a method for pixel-wise distance estimation of objects from a single camera without the use of ground truth labels, is an important task in 3D scene understanding. However, existing methods for this task are limited to convolutional neural network (CNN) architectures. In contrast with CNNs that use localized linear operations and lose feature resolution across the layers, vision transformers process at constant resolution with a global receptive field at every stage. While recent works have compared transformers against their CNN counterparts for tasks such as image classification, no study exists that investigates the impact of using transformers for self-supervised monocular depth estimation. Here, we first demonstrate how to adapt vision transformers for self-supervised monocular depth estimation. Thereafter, we compare the transformer and CNN-based architectures for their performance on KITTI depth prediction benchmarks, as well as their robustness to natural corruptions and adversarial attacks, including when the camera intrinsics are unknown. Our study demonstrates how transformer-based architecture, though lower in run-time efficiency, achieves comparable performance while being more robust and generalizable.

摘要: 自动驾驶和先进的驾驶辅助系统的出现要求计算机视觉在3D场景理解方面不断发展。自监督单目深度估计是三维场景理解中的一项重要任务，它是一种不使用地面真实信息的单摄像机像素级距离估计方法。然而，现有的方法仅限于卷积神经网络(CNN)结构。与使用局部线性运算的CNN相比，视觉转换器在每个阶段都以恒定的分辨率进行处理，具有全局感受场，并且失去了跨层的特征分辨率。虽然最近的研究已经将变压器与CNN的同类产品进行了比较，例如图像分类，但没有研究调查使用变压器进行自我监督单目深度估计的影响。在这里，我们首先演示如何使视觉转换器适用于自监督单目深度估计。然后，我们比较了变换结构和基于CNN的结构在Kitti深度预测基准上的性能，以及它们对自然破坏和敌意攻击的鲁棒性，包括在摄像机内部特征未知的情况下。我们的研究展示了基于转换器的体系结构是如何在运行时效率较低的情况下，实现与之相当的性能，同时更健壮和更具通用性。



## **24. Adversarial Attacks and Defense for Non-Parametric Two-Sample Tests**

非参数两样本检验的对抗性攻击与防御 cs.LG

**SubmitDate**: 2022-02-07    [paper-pdf](http://arxiv.org/pdf/2202.03077v1)

**Authors**: Xilie Xu, Jingfeng Zhang, Feng Liu, Masashi Sugiyama, Mohan Kankanhalli

**Abstracts**: Non-parametric two-sample tests (TSTs) that judge whether two sets of samples are drawn from the same distribution, have been widely used in the analysis of critical data. People tend to employ TSTs as trusted basic tools and rarely have any doubt about their reliability. This paper systematically uncovers the failure mode of non-parametric TSTs through adversarial attacks and then proposes corresponding defense strategies. First, we theoretically show that an adversary can upper-bound the distributional shift which guarantees the attack's invisibility. Furthermore, we theoretically find that the adversary can also degrade the lower bound of a TST's test power, which enables us to iteratively minimize the test criterion in order to search for adversarial pairs. To enable TST-agnostic attacks, we propose an ensemble attack (EA) framework that jointly minimizes the different types of test criteria. Second, to robustify TSTs, we propose a max-min optimization that iteratively generates adversarial pairs to train the deep kernels. Extensive experiments on both simulated and real-world datasets validate the adversarial vulnerabilities of non-parametric TSTs and the effectiveness of our proposed defense.

摘要: 非参数两样本检验(TSTs)是判断两组样本是否来自同一分布的检验方法，在关键数据的分析中得到了广泛的应用。人们倾向于使用TST作为可信的基本工具，并且很少对其可靠性有任何怀疑。本文系统地揭示了非参数TST通过对抗性攻击的失效模式，并提出了相应的防御策略。首先，我们从理论上证明了敌方可以上界保证攻击不可见性的分布偏移。此外，我们从理论上发现，敌手也可以降低TST测试功率的下界，这使得我们能够迭代最小化测试标准以搜索对手对。为了支持与TST无关的攻击，我们提出了一个联合最小化不同类型测试标准的集成攻击(EA)框架。其次，为了增强TSTs的健壮性，我们提出了一种最大-最小优化算法，该算法迭代生成敌意对来训练深层核。在模拟和真实数据集上的大量实验验证了非参数TST的对抗性漏洞和我们提出的防御的有效性。



## **25. Demystifying the Transferability of Adversarial Attacks in Computer Networks**

揭开计算机网络中敌意攻击可传递性的神秘面纱 cs.CR

15 pages

**SubmitDate**: 2022-02-07    [paper-pdf](http://arxiv.org/pdf/2110.04488v2)

**Authors**: Ehsan Nowroozi, Yassine Mekdad, Mohammad Hajian Berenjestanaki, Mauro Conti, Abdeslam EL Fergougui

**Abstracts**: Convolutional Neural Networks (CNNs) models are one of the most frequently used deep learning networks and are extensively used in both academia and industry. Recent studies demonstrated that adversarial attacks against such models can maintain their effectiveness even when used on models other than the one targeted by the attacker. This major property is known as transferability and makes CNNs ill-suited for security applications. In this paper, we provide the first comprehensive study which assesses the robustness of CNN-based models for computer networks against adversarial transferability. Furthermore, we investigate whether the transferability property issue holds in computer networks applications. In our experiments, we first consider five different attacks: the Iterative Fast Gradient Method (I-FGSM), the Jacobian-based Saliency Map (JSMA), the Limited-memory Broyden letcher Goldfarb Shanno BFGS (L-BFGS), the Projected Gradient Descent (PGD), and the DeepFool attack. Then, we perform these attacks against three well-known datasets: the Network-based Detection of IoT (N-BaIoT) dataset and the Domain Generating Algorithms (DGA) dataset, and RIPE Atlas dataset. Our experimental results show clearly that the transferability happens in specific use cases for the I-FGSM, the JSMA, and the LBFGS attack. In such scenarios, the attack success rate on the target network range from 63.00\% to 100\%. Finally, we suggest two shielding strategies to hinder the attack transferability, by considering the Most Powerful Attacks (MPAs), and the mismatch LSTM architecture.

摘要: 卷积神经网络(CNNs)模型是最常用的深度学习网络之一，在学术界和工业界都得到了广泛的应用。最近的研究表明，即使在攻击者目标之外的模型上使用，针对此类模型的对抗性攻击也可以保持其有效性。这一主要属性称为可转移性，使得CNN不适合安全应用。在这篇论文中，我们提供了第一个全面的研究，评估了基于CNN的计算机网络模型对对抗可转移性的鲁棒性。此外，我们还研究了在计算机网络应用中是否存在可转移性问题。在我们的实验中，我们首先考虑了五种不同的攻击：迭代快速梯度法(I-FGSM)、基于雅可比的显著图(JSMA)、有限内存的Broyden Letcher Goldfarb Shanno BFGs(L-BFGs)、投影梯度下降(PGD)和DeepFool攻击。然后，针对基于网络的物联网检测(N-BaIoT)数据集、域生成算法(DGA)数据集和成熟的Atlas数据集进行了攻击。我们的实验结果清楚地表明，对于I-FGSM、JSMA和LBFGS攻击，这种可转移性发生在特定的用例中。在这种情况下，对目标网络的攻击成功率从63.00\%到100\%不等。最后，通过考虑最强大的攻击(MPA)和不匹配的LSTM体系结构，提出了两种屏蔽策略来阻碍攻击的可传递性。



## **26. Explaining Adversarial Vulnerability with a Data Sparsity Hypothesis**

用数据稀疏性假说解释对抗性脆弱性 cs.AI

**SubmitDate**: 2022-02-07    [paper-pdf](http://arxiv.org/pdf/2103.00778v2)

**Authors**: Mahsa Paknezhad, Cuong Phuc Ngo, Amadeus Aristo Winarto, Alistair Cheong, Chuen Yang Beh, Jiayang Wu, Hwee Kuan Lee

**Abstracts**: Despite many proposed algorithms to provide robustness to deep learning (DL) models, DL models remain susceptible to adversarial attacks. We hypothesize that the adversarial vulnerability of DL models stems from two factors. The first factor is data sparsity which is that in the high dimensional input data space, there exist large regions outside the support of the data distribution. The second factor is the existence of many redundant parameters in the DL models. Owing to these factors, different models are able to come up with different decision boundaries with comparably high prediction accuracy. The appearance of the decision boundaries in the space outside the support of the data distribution does not affect the prediction accuracy of the model. However, it makes an important difference in the adversarial robustness of the model. We hypothesize that the ideal decision boundary is as far as possible from the support of the data distribution. In this paper, we develop a training framework to observe if DL models are able to learn such a decision boundary spanning the space around the class distributions further from the data points themselves. Semi-supervised learning was deployed during training by leveraging unlabeled data generated in the space outside the support of the data distribution. We measured adversarial robustness of the models trained using this training framework against well-known adversarial attacks and by using robustness metrics. We found that models trained using our framework, as well as other regularization methods and adversarial training support our hypothesis of data sparsity and that models trained with these methods learn to have decision boundaries more similar to the aforementioned ideal decision boundary. The code for our training framework is available at https://github.com/MahsaPaknezhad/AdversariallyRobustTraining.

摘要: 尽管提出了许多算法来提供深度学习(DL)模型的鲁棒性，但是DL模型仍然容易受到敌意攻击。我们假设DL模型的对抗脆弱性源于两个因素。第一个因素是数据稀疏性，即在高维输入数据空间中，存在数据分布支持之外的大区域。第二个因素是DL模型中存在许多冗余参数。由于这些因素的影响，不同的模型能够给出不同的决策边界，具有较高的预测精度。在数据分布支持度之外的空间出现决策边界并不影响模型的预测精度。然而，它在模型的对抗性鲁棒性方面有很大的不同。我们假设理想的决策边界尽可能远离数据分布的支持。在本文中，我们开发了一个训练框架来观察DL模型是否能够从数据点本身进一步学习跨越类分布周围空间的决策边界。通过利用在数据分布支持之外的空间中生成的未标记数据，在训练期间部署半监督学习。我们通过使用健壮性度量来衡量使用该训练框架训练的模型对众所周知的敌意攻击的敌意稳健性。我们发现，使用我们的框架训练的模型，以及其他正则化方法和对抗性训练，都支持我们的数据稀疏性假设，并且用这些方法训练的模型学习的决策边界更类似于前面提到的理想决策边界。我们培训框架的代码可以在https://github.com/MahsaPaknezhad/AdversariallyRobustTraining.上找到



## **27. Adversarial Unlearning of Backdoors via Implicit Hypergradient**

基于隐式超梯度的后门对抗性遗忘 cs.LG

In proceeding of the Tenth International Conference on Learning  Representations (ICLR 2022)

**SubmitDate**: 2022-02-06    [paper-pdf](http://arxiv.org/pdf/2110.03735v4)

**Authors**: Yi Zeng, Si Chen, Won Park, Z. Morley Mao, Ming Jin, Ruoxi Jia

**Abstracts**: We propose a minimax formulation for removing backdoors from a given poisoned model based on a small set of clean data. This formulation encompasses much of prior work on backdoor removal. We propose the Implicit Bacdoor Adversarial Unlearning (I-BAU) algorithm to solve the minimax. Unlike previous work, which breaks down the minimax into separate inner and outer problems, our algorithm utilizes the implicit hypergradient to account for the interdependence between inner and outer optimization. We theoretically analyze its convergence and the generalizability of the robustness gained by solving minimax on clean data to unseen test data. In our evaluation, we compare I-BAU with six state-of-art backdoor defenses on seven backdoor attacks over two datasets and various attack settings, including the common setting where the attacker targets one class as well as important but underexplored settings where multiple classes are targeted. I-BAU's performance is comparable to and most often significantly better than the best baseline. Particularly, its performance is more robust to the variation on triggers, attack settings, poison ratio, and clean data size. Moreover, I-BAU requires less computation to take effect; particularly, it is more than $13\times$ faster than the most efficient baseline in the single-target attack setting. Furthermore, it can remain effective in the extreme case where the defender can only access 100 clean samples -- a setting where all the baselines fail to produce acceptable results.

摘要: 我们提出了一种极小极大公式，用于从给定的中毒模型中移除后门，该公式基于一小部分干净的数据。这个公式包含了很多关于后门删除的前期工作。提出了求解极大极小问题的隐式Bacdoor对抗性遗忘(I-BAU)算法。与以往将极小极大问题分解为独立的内问题和外问题不同，我们的算法利用隐式超梯度来考虑内优化和外优化之间的相互依赖关系。我们从理论上分析了它的收敛性，以及通过求解干净数据上的极小极大所获得的鲁棒性对不可见测试数据的普适性。在我们的评估中，我们将I-BAU与六种最先进的后门防御技术在两个数据集和各种攻击设置上进行了七次后门攻击的比较，包括攻击者以一个类为目标的常见设置，以及以多个类为目标的重要但未被开发的设置。i-BAU的性能可以与最佳基准相媲美，而且往往明显优于最佳基准。具体地说，它的性能对触发器、攻击设置、毒率和干净数据大小的变化更加健壮。此外，I-BAU需要更少的计算才能生效，特别是在单目标攻击环境下，它比最有效的基线快13倍以上。此外，在防御者只能访问100个干净样本的极端情况下，它仍然有效--在这种情况下，所有基线都无法产生可接受的结果。



## **28. Pipe Overflow: Smashing Voice Authentication for Fun and Profit**

管道溢出：粉碎语音身份验证以获取乐趣和利润 cs.LG

**SubmitDate**: 2022-02-06    [paper-pdf](http://arxiv.org/pdf/2202.02751v1)

**Authors**: Shimaa Ahmed, Yash Wani, Ali Shahin Shamsabadi, Mohammad Yaghini, Ilia Shumailov, Nicolas Papernot, Kassem Fawaz

**Abstracts**: Recent years have seen a surge of popularity of acoustics-enabled personal devices powered by machine learning. Yet, machine learning has proven to be vulnerable to adversarial examples. Large number of modern systems protect themselves against such attacks by targeting the artificiality, i.e., they deploy mechanisms to detect the lack of human involvement in generating the adversarial examples. However, these defenses implicitly assume that humans are incapable of producing meaningful and targeted adversarial examples. In this paper, we show that this base assumption is wrong. In particular, we demonstrate that for tasks like speaker identification, a human is capable of producing analog adversarial examples directly with little cost and supervision: by simply speaking through a tube, an adversary reliably impersonates other speakers in eyes of ML models for speaker identification. Our findings extend to a range of other acoustic-biometric tasks such as liveness, bringing into question their use in security-critical settings in real life, such as phone banking.

摘要: 近年来，由机器学习驱动的声学个人设备的受欢迎程度激增。然而，事实证明，机器学习很容易受到对抗性例子的影响。大量的现代系统通过瞄准人工性来保护自己免受此类攻击，即，它们部署机制来检测在生成对抗性示例时缺少人的参与。然而，这些防御隐含地假设人类没有能力产生有意义和有针对性的对抗性例子。在本文中，我们证明了这一基本假设是错误的。特别地，我们证明了对于像说话人识别这样的任务，人类能够以很少的成本和监督直接产生模拟的对抗性例子：通过简单地通过管道说话，对手能够可靠地在ML模型的眼中模仿其他说话人进行说话人识别。我们的发现延伸到了其他一系列声学-生物识别任务，如活跃度，这让人质疑它们在现实生活中对安全至关重要的环境中的使用情况，比如电话银行。



## **29. EvadeDroid: A Practical Evasion Attack on Machine Learning for Black-box Android Malware Detection**

EvadeDroid：一种实用的机器学习黑盒Android恶意软件检测规避攻击 cs.LG

**SubmitDate**: 2022-02-05    [paper-pdf](http://arxiv.org/pdf/2110.03301v2)

**Authors**: Hamid Bostani, Veelasha Moonsamy

**Abstracts**: Over the last decade, several studies have investigated the weaknesses of Android malware detectors against adversarial examples by proposing novel evasion attacks; however, their practicality in manipulating real-world malware remains arguable. The majority of studies have assumed attackers know the details of the target classifiers used for malware detection, while in reality, malicious actors have limited access to the target classifiers. This paper presents a practical evasion attack, EvadeDroid, to circumvent black-box Android malware detectors. In addition to generating real-world adversarial malware, the proposed evasion attack can also preserve the functionality of the original malware samples. EvadeDroid prepares a collection of functionality-preserving transformations using an n-gram-based similarity method, which are then used to morph malware instances into benign ones via an iterative and incremental manipulation strategy. The proposed manipulation technique is a novel, query-efficient optimization algorithm with the aim of finding and injecting optimal sequences of transformations into malware samples. Our empirical evaluation demonstrates the efficacy of EvadeDroid under hard- and soft-label attacks. Moreover, EvadeDroid is capable to generate practical adversarial examples with only a small number of queries, with evasion rates of $81\%$, $73\%$, $75\%$, and $79\%$ for DREBIN, Sec-SVM, MaMaDroid, and ADE-MA, respectively. Finally, we show that EvadeDroid is able to preserve its stealthiness against five popular commercial antivirus, thus demonstrating its feasibility in the real world.

摘要: 在过去的十年里，一些研究已经通过提出新的逃避攻击来调查Android恶意软件检测器针对敌意示例的弱点；然而，它们在操纵现实世界恶意软件方面的实用性仍然存在争议。大多数研究都假设攻击者知道用于恶意软件检测的目标分类器的详细信息，而实际上，恶意行为者对目标分类器的访问是有限的。提出了一种实用的规避攻击方法EvadeDroid，以规避黑盒Android恶意软件检测器。除了生成真实的敌意恶意软件外，所提出的规避攻击还可以保留原始恶意软件样本的功能。EvadeDroid使用基于n元语法的相似性方法准备一组保留功能的转换，然后使用这些转换通过迭代和增量操作策略将恶意软件实例变形为良性实例。所提出的操作技术是一种新颖的、查询高效的优化算法，其目的是找到最优的转换序列并将其注入到恶意软件样本中。我们的实验评估证明了EvadeDroid在硬标签和软标签攻击下的有效性。此外，EvadeDroid能够用较少的查询生成实用的对抗性实例，Drebin、SEC-SVM、MaMaDroid和ADE-MA的逃避率分别为81美元、73美元、75美元和79美元。最后，我们证明了EvadeDroid能够对五种流行的商业反病毒软件保持隐蔽性，从而证明了其在现实世界中的可行性。



## **30. Iota: A Framework for Analyzing System-Level Security of IoTs**

IOTA：物联网系统级安全分析框架 cs.CR

This manuscript has been accepted by IoTDI 2022

**SubmitDate**: 2022-02-05    [paper-pdf](http://arxiv.org/pdf/2202.02506v1)

**Authors**: Zheng Fang, Hao Fu, Tianbo Gu, Pengfei Hu, Jinyue Song, Trent Jaeger, Prasant Mohapatra

**Abstracts**: Most IoT systems involve IoT devices, communication protocols, remote cloud, IoT applications, mobile apps, and the physical environment. However, existing IoT security analyses only focus on a subset of all the essential components, such as device firmware, and ignore IoT systems' interactive nature, resulting in limited attack detection capabilities. In this work, we propose Iota, a logic programming-based framework to perform system-level security analysis for IoT systems. Iota generates attack graphs for IoT systems, showing all of the system resources that can be compromised and enumerating potential attack traces. In building Iota, we design novel techniques to scan IoT systems for individual vulnerabilities and further create generic exploit models for IoT vulnerabilities. We also identify and model physical dependencies between different devices as they are unique to IoT systems and are employed by adversaries to launch complicated attacks. In addition, we utilize NLP techniques to extract IoT app semantics based on app descriptions. To evaluate vulnerabilities' system-wide impact, we propose two metrics based on the attack graph, which provide guidance on fortifying IoT systems. Evaluation on 127 IoT CVEs (Common Vulnerabilities and Exposures) shows that Iota's exploit modeling module achieves over 80% accuracy in predicting vulnerabilities' preconditions and effects. We apply Iota to 37 synthetic smart home IoT systems based on real-world IoT apps and devices. Experimental results show that our framework is effective and highly efficient. Among 27 shortest attack traces revealed by the attack graphs, 62.8% are not anticipated by the system administrator. It only takes 1.2 seconds to generate and analyze the attack graph for an IoT system consisting of 50 devices.

摘要: 大多数物联网系统涉及物联网设备、通信协议、远程云、物联网应用、移动应用和物理环境。然而，现有的物联网安全分析只关注设备固件等所有基本组件的子集，而忽略了物联网系统的交互性，导致攻击检测能力有限。在这项工作中，我们提出了一种基于逻辑编程的物联网系统级安全分析框架IOTA。IOTA为物联网系统生成攻击图，显示所有可能被破坏的系统资源，并列举潜在的攻击痕迹。在构建物联网的过程中，我们设计了新的技术来扫描物联网系统中的单个漏洞，并进一步创建物联网漏洞的通用利用模型。我们还识别不同设备之间的物理依赖关系并对其建模，因为它们是物联网系统所独有的，并被对手用来发动复杂的攻击。此外，我们还利用自然语言处理技术，基于应用描述提取物联网应用语义。为了评估漏洞对整个系统的影响，我们提出了两个基于攻击图的度量，为加强物联网系统提供指导。对127个物联网CVE(常见漏洞和暴露)的评估表明，IOTA的漏洞建模模块对漏洞的前提条件和影响的预测准确率超过80%。我们基于现实世界的物联网应用程序和设备，将物联网应用于37个合成智能家居物联网系统。实验结果表明，该框架是有效和高效的。在攻击图显示的27条最短的攻击轨迹中，62.8%是系统管理员没有预料到的。对于由50台设备组成的物联网系统，仅需1.2秒即可生成并分析攻击图。



## **31. Improving Ensemble Robustness by Collaboratively Promoting and Demoting Adversarial Robustness**

通过协同提升和降低对手健壮性来提高组合健壮性 cs.CV

**SubmitDate**: 2022-02-04    [paper-pdf](http://arxiv.org/pdf/2009.09612v2)

**Authors**: Anh Bui, Trung Le, He Zhao, Paul Montague, Olivier deVel, Tamas Abraham, Dinh Phung

**Abstracts**: Ensemble-based adversarial training is a principled approach to achieve robustness against adversarial attacks. An important technique of this approach is to control the transferability of adversarial examples among ensemble members. We propose in this work a simple yet effective strategy to collaborate among committee models of an ensemble model. This is achieved via the secure and insecure sets defined for each model member on a given sample, hence help us to quantify and regularize the transferability. Consequently, our proposed framework provides the flexibility to reduce the adversarial transferability as well as to promote the diversity of ensemble members, which are two crucial factors for better robustness in our ensemble approach. We conduct extensive and comprehensive experiments to demonstrate that our proposed method outperforms the state-of-the-art ensemble baselines, at the same time can detect a wide range of adversarial examples with a nearly perfect accuracy. Our code is available at: https://github.com/tuananhbui89/Crossing-Collaborative-Ensemble.

摘要: 基于集成的对抗性训练是实现对抗攻击鲁棒性的原则性方法。该方法的一项重要技术是控制对抗性范例在集合成员之间的可转移性。在这项工作中，我们提出了一种简单而有效的策略来在集合模型的委员会模型之间进行协作。这是通过为给定样本上的每个模型成员定义的安全和不安全集合来实现的，因此有助于我们量化和规则化可转移性。因此，我们提出的框架提供了灵活性，既可以减少对抗性转移，又可以促进集成成员的多样性，这是在集成方法中获得更好健壮性的两个关键因素。我们进行了广泛而全面的实验，证明我们提出的方法的性能优于最新的集成基线，同时能够以近乎完美的准确率检测到广泛的对抗性示例。我们的代码可从以下网址获得：https://github.com/tuananhbui89/Crossing-Collaborative-Ensemble.



## **32. Temporal Motifs in Patent Opposition and Collaboration Networks**

专利对抗与合作网络中的时间母题 cs.SI

**SubmitDate**: 2022-02-04    [paper-pdf](http://arxiv.org/pdf/2110.11198v2)

**Authors**: Penghang Liu, Naoki Masuda, Tomomi Kito, A. Erdem Sarıyüce

**Abstracts**: Patents are intellectual properties that reflect innovative activities of companies and organizations. The literature is rich with the studies that analyze the citations among the patents and the collaboration relations among companies that own the patents. However, the adversarial relations between the patent owners are not as well investigated. One proxy to model such relations is the patent opposition, which is a legal activity in which a company challenges the validity of a patent. Characterizing the patent oppositions, collaborations, and the interplay between them can help better understand the companies' business strategies. Temporality matters in this context as the order and frequency of oppositions and collaborations characterize their interplay. In this study, we construct a two-layer temporal network to model the patent oppositions and collaborations among the companies. We utilize temporal motifs to analyze the oppositions and collaborations from structural and temporal perspectives. We first characterize the frequent motifs in patent oppositions and investigate how often the companies of different sizes attack other companies. We show that large companies tend to engage in opposition with multiple companies. Then we analyze the temporal interplay between collaborations and oppositions. We find that two adversarial companies are more likely to collaborate in the future than two collaborating companies oppose each other in the future.

摘要: 专利是反映公司和组织创新活动的知识产权。文献中大量分析了专利之间的引文以及拥有专利的企业之间的合作关系。然而，专利权人之间的对抗关系并没有得到很好的研究。模拟这种关系的一个代理是专利反对，这是一种法律活动，公司在这一活动中质疑专利的有效性。描述专利对抗、合作以及它们之间的相互作用有助于更好地理解公司的商业战略。在这种情况下，时间性很重要，因为对立和合作的顺序和频率表征了它们之间的相互作用。在本研究中，我们构建了一个两层时间网络来模拟企业之间的专利对抗与合作。我们利用时间母题从结构和时间两个角度来分析对立和合作。我们首先描述了专利对抗中频繁出现的主题，并调查了不同规模的公司攻击其他公司的频率。我们发现，大公司往往会与多家公司发生对立。然后分析了合作与对立在时间上的相互作用。我们发现，未来两个对抗性公司比两个合作公司在未来更有可能合作。



## **33. Dikaios: Privacy Auditing of Algorithmic Fairness via Attribute Inference Attacks**

Dikaios：基于属性推理攻击的算法公平性隐私审计 cs.CR

**SubmitDate**: 2022-02-04    [paper-pdf](http://arxiv.org/pdf/2202.02242v1)

**Authors**: Jan Aalmoes, Vasisht Duddu, Antoine Boutet

**Abstracts**: Machine learning (ML) models have been deployed for high-stakes applications. Due to class imbalance in the sensitive attribute observed in the datasets, ML models are unfair on minority subgroups identified by a sensitive attribute, such as race and sex. In-processing fairness algorithms ensure model predictions are independent of sensitive attribute. Furthermore, ML models are vulnerable to attribute inference attacks where an adversary can identify the values of sensitive attribute by exploiting their distinguishable model predictions. Despite privacy and fairness being important pillars of trustworthy ML, the privacy risk introduced by fairness algorithms with respect to attribute leakage has not been studied. We identify attribute inference attacks as an effective measure for auditing blackbox fairness algorithms to enable model builder to account for privacy and fairness in the model design. We proposed Dikaios, a privacy auditing tool for fairness algorithms for model builders which leveraged a new effective attribute inference attack that account for the class imbalance in sensitive attributes through an adaptive prediction threshold. We evaluated Dikaios to perform a privacy audit of two in-processing fairness algorithms over five datasets. We show that our attribute inference attacks with adaptive prediction threshold significantly outperform prior attacks. We highlighted the limitations of in-processing fairness algorithms to ensure indistinguishable predictions across different values of sensitive attributes. Indeed, the attribute privacy risk of these in-processing fairness schemes is highly variable according to the proportion of the sensitive attributes in the dataset. This unpredictable effect of fairness mechanisms on the attribute privacy risk is an important limitation on their utilization which has to be accounted by the model builder.

摘要: 机器学习(ML)模型已经被部署到高风险应用程序中。由于在数据集中观察到的敏感属性的类别不平衡，ML模型对由敏感属性(如种族和性别)识别的少数群体是不公平的。处理中的公平算法确保模型预测独立于敏感属性。此外，ML模型容易受到属性推理攻击，攻击者可以利用敏感属性的可区分模型预测来识别敏感属性的值。尽管隐私和公平是可信ML的重要支柱，但公平算法在属性泄漏方面引入的隐私风险尚未得到研究。我们将属性推理攻击作为审计黑盒公平算法的有效手段，使模型构造者能够在模型设计中考虑私密性和公平性。我们提出了Dikaios，这是一个模型构建者公平算法的隐私审计工具，它利用了一种新的有效的属性推理攻击，通过自适应的预测阈值来解释敏感属性中的类别不平衡。我们评估了Dikaios在五个数据集上执行两个正在处理的公平算法的隐私审计。实验结果表明，具有自适应预测阈值的属性推理攻击的性能明显优于以往的攻击。我们强调了处理中公平算法的局限性，以确保敏感属性的不同值之间无法区分预测。事实上，根据敏感属性在数据集中的比例，这些正在处理的公平方案的属性隐私风险是高度可变的。公平机制对属性隐私风险的这种不可预测的影响是其使用的一个重要限制，这必须由模型构建者考虑。



## **34. Sparse Polynomial Optimisation for Neural Network Verification**

神经网络验证中的稀疏多项式优化 eess.SY

25 pages, 20 figures

**SubmitDate**: 2022-02-04    [paper-pdf](http://arxiv.org/pdf/2202.02241v1)

**Authors**: Matthew Newton, Antonis Papachristodoulou

**Abstracts**: The prevalence of neural networks in society is expanding at an increasing rate. It is becoming clear that providing robust guarantees on systems that use neural networks is very important, especially in safety-critical applications. A trained neural network's sensitivity to adversarial attacks is one of its greatest shortcomings. To provide robust guarantees, one popular method that has seen success is to bound the activation functions using equality and inequality constraints. However, there are numerous ways to form these bounds, providing a trade-off between conservativeness and complexity. Depending on the complexity of these bounds, the computational time of the optimisation problem varies, with longer solve times often leading to tighter bounds. We approach the problem from a different perspective, using sparse polynomial optimisation theory and the Positivstellensatz, which derives from the field of real algebraic geometry. The former exploits the natural cascading structure of the neural network using ideas from chordal sparsity while the later asserts the emptiness of a semi-algebraic set with a nested family of tests of non-decreasing accuracy to provide tight bounds. We show that bounds can be tightened significantly, whilst the computational time remains reasonable. We compare the solve times of different solvers and show how the accuracy can be improved at the expense of increased computation time. We show that by using this sparse polynomial framework the solve time and accuracy can be improved over other methods for neural network verification with ReLU, sigmoid and tanh activation functions.

摘要: 神经网络在社会中的流行正在以越来越快的速度扩大。越来越清楚的是，为使用神经网络的系统提供健壮的保证是非常重要的，特别是在安全关键的应用中。训练有素的神经网络对敌方攻击的敏感度是其最大的缺点之一。为了提供健壮的保证，一种已经取得成功的流行方法是使用等式和不等式约束来限制激活函数。然而，有许多方法可以形成这些界限，从而在保守性和复杂性之间进行权衡。根据这些边界的复杂程度，优化问题的计算时间会有所不同，较长的求解时间通常会导致较紧的边界。我们用稀疏多项式最优化理论和源于实代数几何领域的正态理论，从不同的角度来研究这个问题。前者利用神经网络的自然级联结构，利用弦稀疏性的思想，而后者则通过嵌套的不降低精度的测试族来断言半代数集的空性，以提供严格的界。我们表明，在计算时间保持合理的情况下，边界可以显著地收紧。我们比较了不同求解器的求解时间，并展示了如何以增加计算时间为代价来提高精度。结果表明，使用该稀疏多项式框架可以提高RELU、Sigmoid和tanh激活函数验证神经网络的求解时间和精度。



## **35. Pixle: a fast and effective black-box attack based on rearranging pixels**

Pixle：一种快速有效的基于像素重排的黑盒攻击方法 cs.LG

**SubmitDate**: 2022-02-04    [paper-pdf](http://arxiv.org/pdf/2202.02236v1)

**Authors**: Jary Pomponi, Simone Scardapane, Aurelio Uncini

**Abstracts**: Recent research has found that neural networks are vulnerable to several types of adversarial attacks, where the input samples are modified in such a way that the model produces a wrong prediction that misclassifies the adversarial sample. In this paper we focus on black-box adversarial attacks, that can be performed without knowing the inner structure of the attacked model, nor the training procedure, and we propose a novel attack that is capable of correctly attacking a high percentage of samples by rearranging a small number of pixels within the attacked image. We demonstrate that our attack works on a large number of datasets and models, that it requires a small number of iterations, and that the distance between the original sample and the adversarial one is negligible to the human eye.

摘要: 最近的研究发现，神经网络容易受到几种类型的对抗性攻击，其中输入样本被修改的方式使得模型产生错误的预测，从而错误地对对抗性样本进行分类。针对黑盒对抗性攻击，在不知道被攻击模型的内部结构和训练过程的情况下，提出了一种新的攻击方法，通过重新排列被攻击图像中的少量像素，能够正确地攻击高百分比的样本。我们证明了我们的攻击工作在大量的数据集和模型上，只需要少量的迭代，并且原始样本和对抗性样本之间的距离对于肉眼来说是可以忽略的。



## **36. Modeling Adversarial Noise for Adversarial Training**

用于对抗性训练的对抗性噪声建模 cs.LG

**SubmitDate**: 2022-02-04    [paper-pdf](http://arxiv.org/pdf/2109.09901v4)

**Authors**: Dawei Zhou, Nannan Wang, Bo Han, Tongliang Liu

**Abstracts**: Deep neural networks have been demonstrated to be vulnerable to adversarial noise, promoting the development of defense against adversarial attacks. Motivated by the fact that adversarial noise contains well-generalizing features and that the relationship between adversarial data and natural data can help infer natural data and make reliable predictions, in this paper, we study to model adversarial noise by learning the transition relationship between adversarial labels (i.e. the flipped labels used to generate adversarial data) and natural labels (i.e. the ground truth labels of the natural data). Specifically, we introduce an instance-dependent transition matrix to relate adversarial labels and natural labels, which can be seamlessly embedded with the target model (enabling us to model stronger adaptive adversarial noise). Empirical evaluations demonstrate that our method could effectively improve adversarial accuracy.

摘要: 深度神经网络已被证明对对抗性噪声很敏感，这促进了防御对抗性攻击的发展。基于对抗性噪声包含良好的泛化特征，以及对抗性数据与自然数据之间的关系可以帮助推断自然数据并做出可靠的预测，本文通过学习对抗性标签(即用于生成对抗性数据的翻转标签)和自然标签(即自然数据的地面真实标签)之间的转换关系来研究对抗性噪声的建模。具体地说，我们引入了依赖于实例的转移矩阵来关联对抗性标签和自然标签，它可以无缝地嵌入到目标模型中(使我们能够建模更强的自适应对抗性噪声)。实验结果表明，我们的方法可以有效地提高对手的准确率。



## **37. Knowledge Cross-Distillation for Membership Privacy**

面向会员隐私的知识交叉蒸馏 cs.CR

Accepted by PETS 2022

**SubmitDate**: 2022-02-04    [paper-pdf](http://arxiv.org/pdf/2111.01363v3)

**Authors**: Rishav Chourasia, Batnyam Enkhtaivan, Kunihiro Ito, Junki Mori, Isamu Teranishi, Hikaru Tsuchida

**Abstracts**: A membership inference attack (MIA) poses privacy risks for the training data of a machine learning model. With an MIA, an attacker guesses if the target data are a member of the training dataset. The state-of-the-art defense against MIAs, distillation for membership privacy (DMP), requires not only private data for protection but a large amount of unlabeled public data. However, in certain privacy-sensitive domains, such as medicine and finance, the availability of public data is not guaranteed. Moreover, a trivial method for generating public data by using generative adversarial networks significantly decreases the model accuracy, as reported by the authors of DMP. To overcome this problem, we propose a novel defense against MIAs that uses knowledge distillation without requiring public data. Our experiments show that the privacy protection and accuracy of our defense are comparable to those of DMP for the benchmark tabular datasets used in MIA research, Purchase100 and Texas100, and our defense has a much better privacy-utility trade-off than those of the existing defenses that also do not use public data for the image dataset CIFAR10.

摘要: 隶属度推理攻击(MIA)给机器学习模型的训练数据带来隐私风险。使用MIA，攻击者可以猜测目标数据是否为训练数据集的成员。针对MIA的最先进的防御措施，即会员隐私蒸馏(DMP)，不仅需要私有数据来保护，还需要大量未标记的公共数据。然而，在某些隐私敏感领域，如医疗和金融，公共数据的可用性不能得到保证。此外，正如DMP的作者所报告的那样，通过使用生成性对抗网络来生成公共数据的琐碎方法显著降低了模型的准确性。为了克服这一问题，我们提出了一种新的防御MIA的方法，该方法使用知识提炼而不需要公开数据。我们的实验表明，对于MIA研究中使用的基准表格数据集，我们的防御的隐私保护和准确性与DMP相当，并且我们的防御具有更好的隐私效用权衡，而现有的防御也没有将公共数据用于图像数据集CIFAR10。



## **38. Fast Gradient Non-sign Methods**

快速梯度无符号方法 cs.CV

**SubmitDate**: 2022-02-04    [paper-pdf](http://arxiv.org/pdf/2110.12734v3)

**Authors**: Yaya Cheng, Jingkuan Song, Xiaosu Zhu, Qilong Zhang, Lianli Gao, Heng Tao Shen

**Abstracts**: Adversarial attacks make their success in DNNs, and among them, gradient-based algorithms become one of the mainstreams. Based on the linearity hypothesis, under $\ell_\infty$ constraint, $sign$ operation applied to the gradients is a good choice for generating perturbations. However, side-effects from such operation exist since it leads to the bias of direction between real gradients and perturbations. In other words, current methods contain a gap between real gradients and actual noises, which leads to biased and inefficient attacks. Therefore in this paper, based on the Taylor expansion, the bias is analyzed theoretically, and the correction of $sign$, i.e., Fast Gradient Non-sign Method (FGNM), is further proposed. Notably, FGNM is a general routine that seamlessly replaces the conventional $sign$ operation in gradient-based attacks with negligible extra computational cost. Extensive experiments demonstrate the effectiveness of our methods. Specifically, for untargeted black-box attacks, ours outperform them by 27.5% at most and 9.5% on average. For targeted attacks against defense models, it is 15.1% and 12.7%. Our anonymous code is publicly available at https://github.com/yaya-cheng/FGNM

摘要: 敌意攻击在DNNs中取得了成功，其中基于梯度的算法成为主流算法之一。基于线性假设，在$\ell_\infty$约束下，对梯度进行$sign$运算是产生扰动的较好选择。然而，这种操作的副作用是存在的，因为它导致了真实梯度和扰动之间的方向偏差。换言之，目前的方法存在真实梯度和实际噪声之间的差距，这导致了有偏的、低效的攻击。因此，本文在泰勒展开的基础上，从理论上对偏差进行了分析，并进一步提出了对符号的修正，即快速梯度无符号方法(FGNM)。值得注意的是，FGNM是一个通用例程，它在基于梯度的攻击中无缝替换传统的$SIGN$操作，额外的计算开销可以忽略不计。大量实验证明了该方法的有效性。具体地说，对于无针对性的黑匣子攻击，我们的性能最多比它们高出27.5%，平均高出9.5%。针对防御模型的定向攻击为15.1%和12.7%。我们的匿名代码可在https://github.com/yaya-cheng/FGNM上公开获得



## **39. FRL: Federated Rank Learning**

FRL：联合秩学习 cs.LG

**SubmitDate**: 2022-02-03    [paper-pdf](http://arxiv.org/pdf/2110.04350v2)

**Authors**: Hamid Mozaffari, Virat Shejwalkar, Amir Houmansadr

**Abstracts**: Federated learning (FL) allows mutually untrusted clients to collaboratively train a common machine learning model without sharing their private/proprietary training data among each other. FL is unfortunately susceptible to poisoning by malicious clients who aim to hamper the accuracy of the commonly trained model through sending malicious model updates during FL's training process.   We argue that the key factor to the success of poisoning attacks against existing FL systems is the large space of model updates available to the clients, allowing malicious clients to search for the most poisonous model updates, e.g., by solving an optimization problem. To address this, we propose Federated Rank Learning (FRL). FRL reduces the space of client updates from model parameter updates (a continuous space of float numbers) in standard FL to the space of parameter rankings (a discrete space of integer values). To be able to train the global model using parameter ranks (instead of parameter weights), FRL leverage ideas from recent supermasks training mechanisms. Specifically, FRL clients rank the parameters of a randomly initialized neural network (provided by the server) based on their local training data. The FRL server uses a voting mechanism to aggregate the parameter rankings submitted by clients in each training epoch to generate the global ranking of the next training epoch.   Intuitively, our voting-based aggregation mechanism prevents poisoning clients from making significant adversarial modifications to the global model, as each client will have a single vote! We demonstrate the robustness of FRL to poisoning through analytical proofs and experimentation. We also show FRL's high communication efficiency. Our experiments demonstrate the superiority of FRL in real-world FL settings.

摘要: 联合学习(FL)允许相互不信任的客户端协作地训练公共机器学习模型，而无需彼此共享它们的私有/专有训练数据。不幸的是，FL很容易受到恶意客户的毒害，这些客户的目的是通过在FL的训练过程中发送恶意模型更新来阻碍通常训练的模型的准确性。我们认为，针对现有FL系统的毒化攻击成功的关键因素是客户端拥有巨大的模型更新空间，从而允许恶意客户端通过解决优化问题来搜索最有害的模型更新。为了解决这个问题，我们提出了联合秩学习(FRL)。FR1将客户端更新的空间从标准FL中的模型参数更新(浮点数的连续空间)减少到参数排名的空间(整数值的离散空间)。为了能够使用参数等级(而不是参数权重)来训练全局模型，FRL利用了最近超级掩码训练机制中的想法。具体地说，FRL客户端基于其本地训练数据对随机初始化的神经网络(由服务器提供)的参数进行排名。FRL服务器使用投票机制来聚合由客户端在每个训练时段中提交的参数排名，以生成下一个训练时段的全局排名。直观地说，我们基于投票的聚合机制可以防止毒化客户端对全局模型进行重大的对抗性修改，因为每个客户端将拥有单一的投票权！我们通过分析证明和实验证明了FRL对中毒的鲁棒性。我们还展示了FRL的高通信效率。我们的实验证明了FRL在真实的外语环境中的优越性。



## **40. A Robust Phased Elimination Algorithm for Corruption-Tolerant Gaussian Process Bandits**

一种鲁棒的容忍腐败高斯过程带的阶段性消除算法 stat.ML

Preprint

**SubmitDate**: 2022-02-03    [paper-pdf](http://arxiv.org/pdf/2202.01850v1)

**Authors**: Ilija Bogunovic, Zihan Li, Andreas Krause, Jonathan Scarlett

**Abstracts**: We consider the sequential optimization of an unknown, continuous, and expensive to evaluate reward function, from noisy and adversarially corrupted observed rewards. When the corruption attacks are subject to a suitable budget $C$ and the function lives in a Reproducing Kernel Hilbert Space (RKHS), the problem can be posed as corrupted Gaussian process (GP) bandit optimization. We propose a novel robust elimination-type algorithm that runs in epochs, combines exploration with infrequent switching to select a small subset of actions, and plays each action for multiple time instants. Our algorithm, Robust GP Phased Elimination (RGP-PE), successfully balances robustness to corruptions with exploration and exploitation such that its performance degrades minimally in the presence (or absence) of adversarial corruptions. When $T$ is the number of samples and $\gamma_T$ is the maximal information gain, the corruption-dependent term in our regret bound is $O(C \gamma_T^{3/2})$, which is significantly tighter than the existing $O(C \sqrt{T \gamma_T})$ for several commonly-considered kernels. We perform the first empirical study of robustness in the corrupted GP bandit setting, and show that our algorithm is robust against a variety of adversarial attacks.

摘要: 我们考虑了一个未知的、连续的和昂贵的奖励函数的序列优化问题，这些奖励函数来自于噪声和恶意破坏的观测奖励。当腐败攻击服从适当的预算$C$，并且函数位于再生核-希尔伯特空间(RKHS)中时，问题可以假设为破坏的高斯过程(GP)土匪优化。我们提出了一种新的健壮的消除型算法，该算法运行在历元上，结合探索和不频繁的切换来选择一小部分动作，并在多个时刻播放每个动作。我们的算法，鲁棒GP阶段消除算法(RGP-PE)，成功地平衡了对腐败的鲁棒性与探索和利用，使得它在存在(或不存在)敌对腐败的情况下性能下降最小。当$T$是样本数，$\Gamma_T$是最大信息增益时，遗憾界中的腐败依赖项为$O(C\Gamma_T^{3/2})$，这比现有的几个常用核函数的$O(C\sqrt{T\Gamma_T})$要紧得多。我们首次对被破坏的GP盗版环境下的鲁棒性进行了实验研究，并证明了我们的算法对各种敌意攻击具有很强的鲁棒性。



## **41. ObjectSeeker: Certifiably Robust Object Detection against Patch Hiding Attacks via Patch-agnostic Masking**

ObjectSeeker：基于补丁无关掩蔽的抗补丁隐藏攻击的可证明鲁棒目标检测 cs.CV

**SubmitDate**: 2022-02-03    [paper-pdf](http://arxiv.org/pdf/2202.01811v1)

**Authors**: Chong Xiang, Alexander Valtchanov, Saeed Mahloujifar, Prateek Mittal

**Abstracts**: Object detectors, which are widely deployed in security-critical systems such as autonomous vehicles, have been found vulnerable to physical-world patch hiding attacks. The attacker can use a single physically-realizable adversarial patch to make the object detector miss the detection of victim objects and completely undermines the functionality of object detection applications. In this paper, we propose ObjectSeeker as a defense framework for building certifiably robust object detectors against patch hiding attacks. The core operation of ObjectSeeker is patch-agnostic masking: we aim to mask out the entire adversarial patch without any prior knowledge of the shape, size, and location of the patch. This masking operation neutralizes the adversarial effect and allows any vanilla object detector to safely detect objects on the masked images. Remarkably, we develop a certification procedure to determine if ObjectSeeker can detect certain objects with a provable guarantee against any adaptive attacker within the threat model. Our evaluation with two object detectors and three datasets demonstrates a significant (~10%-40% absolute and ~2-6x relative) improvement in certified robustness over the prior work, as well as high clean performance (~1% performance drop compared with vanilla undefended models).

摘要: 物体检测器被广泛部署在自动驾驶车辆等安全关键系统中，被发现容易受到物理世界的补丁隐藏攻击。攻击者可以使用单个物理上可实现的对抗性补丁来使对象检测器错过受害者对象的检测，并完全破坏对象检测应用程序的功能。在本文中，我们提出了ObjectSeeker作为一个防御框架，用于构建可证明鲁棒的对象检测器，以抵御补丁隐藏攻击。ObjectSeeker的核心操作是与补丁无关的掩蔽：我们的目标是掩蔽整个敌意补丁，而不需要事先知道补丁的形状、大小和位置。该掩蔽操作中和了对抗性效应，并允许任何香草对象检测器安全地检测掩蔽图像上的对象。值得注意的是，我们开发了一个认证过程来确定ObjectSeeker是否可以检测某些对象，并提供可证明的保证，防止威胁模型中的任何自适应攻击者攻击。我们使用两个对象检测器和三个数据集进行的评估表明，与以前的工作相比，认证的鲁棒性有了显著的提高(~10%-40%的绝对和~2-6倍的相对)，并且有很高的干净性能(与普通的无防御模型相比，性能下降了~1%)。



## **42. Toward Realistic Backdoor Injection Attacks on DNNs using Rowhammer**

利用Rowhammer对DNNS进行逼真的后门注入攻击 cs.LG

**SubmitDate**: 2022-02-03    [paper-pdf](http://arxiv.org/pdf/2110.07683v2)

**Authors**: M. Caner Tol, Saad Islam, Berk Sunar, Ziming Zhang

**Abstracts**: State-of-the-art deep neural networks (DNNs) have been proven to be vulnerable to adversarial manipulation and backdoor attacks. Backdoored models deviate from expected behavior on inputs with predefined triggers while retaining performance on clean data. Recent works focus on software simulation of backdoor injection during the inference phase by modifying network weights, which we find often unrealistic in practice due to restrictions in hardware.   In contrast, in this work for the first time we present an end-to-end backdoor injection attack realized on actual hardware on a classifier model using Rowhammer as the fault injection method. To this end, we first investigate the viability of backdoor injection attacks in real-life deployments of DNNs on hardware and address such practical issues in hardware implementation from a novel optimization perspective. We are motivated by the fact that the vulnerable memory locations are very rare, device-specific, and sparsely distributed. Consequently, we propose a novel network training algorithm based on constrained optimization to achieve a realistic backdoor injection attack in hardware. By modifying parameters uniformly across the convolutional and fully-connected layers as well as optimizing the trigger pattern together, we achieve the state-of-the-art attack performance with fewer bit flips. For instance, our method on a hardware-deployed ResNet-20 model trained on CIFAR-10 achieves over 91% test accuracy and 94% attack success rate by flipping only 10 out of 2.2 million bits.

摘要: 最先进的深度神经网络(DNNs)已被证明容易受到敌意操纵和后门攻击。回溯模型偏离了具有预定义触发器的输入的预期行为，同时保持了对干净数据的性能。最近的工作主要集中在通过修改网络权重来模拟推理阶段的后门注入，但由于硬件的限制，这在实践中往往是不现实的。相反，在这项工作中，我们首次提出了一种端到端的后门注入攻击，该攻击是在使用Rowhammer作为故障注入方法的分类器模型上在实际硬件上实现的。为此，我们首先研究了DNN在硬件上的实际部署中后门注入攻击的可行性，并从一个新的优化角度解决了这些硬件实现中的实际问题。我们的动机是易受攻击的内存位置非常罕见、特定于设备且分布稀疏。因此，我们提出了一种新的基于约束优化的网络训练算法，在硬件上实现了逼真的后门注入攻击。通过统一修改卷积层和全连通层的参数，以及一起优化触发模式，我们以更少的比特翻转实现了最先进的攻击性能。例如，我们的方法在一个硬件部署的ResNet-20模型上训练，在CIFAR-10上进行训练，在220万比特中只翻转10比特，就获得了91%以上的测试准确率和94%的攻击成功率。



## **43. Learnability Lock: Authorized Learnability Control Through Adversarial Invertible Transformations**

可学习性锁：通过对抗性可逆变换实现授权可学习性控制 cs.LG

Accepted at ICLR 2022

**SubmitDate**: 2022-02-03    [paper-pdf](http://arxiv.org/pdf/2202.03576v1)

**Authors**: Weiqi Peng, Jinghui Chen

**Abstracts**: Owing much to the revolution of information technology, the recent progress of deep learning benefits incredibly from the vastly enhanced access to data available in various digital formats. However, in certain scenarios, people may not want their data being used for training commercial models and thus studied how to attack the learnability of deep learning models. Previous works on learnability attack only consider the goal of preventing unauthorized exploitation on the specific dataset but not the process of restoring the learnability for authorized cases. To tackle this issue, this paper introduces and investigates a new concept called "learnability lock" for controlling the model's learnability on a specific dataset with a special key. In particular, we propose adversarial invertible transformation, that can be viewed as a mapping from image to image, to slightly modify data samples so that they become "unlearnable" by machine learning models with negligible loss of visual features. Meanwhile, one can unlock the learnability of the dataset and train models normally using the corresponding key. The proposed learnability lock leverages class-wise perturbation that applies a universal transformation function on data samples of the same label. This ensures that the learnability can be easily restored with a simple inverse transformation while remaining difficult to be detected or reverse-engineered. We empirically demonstrate the success and practicability of our method on visual classification tasks.

摘要: 在很大程度上归功于信息技术革命，深度学习最近的进步令人难以置信地受益于对各种数字格式数据的极大改善。然而，在某些场景下，人们可能不希望自己的数据被用于训练商业模型，从而研究如何攻击深度学习模型的可学习性。以往关于学习性攻击的工作只考虑了防止对特定数据集的未授权攻击的目标，而没有考虑恢复授权案例的学习性的过程。针对撞击的这一问题，本文引入并研究了一种称为“可学习性锁”的新概念，用一把特殊的钥匙来控制模型在特定数据集上的可学习性。特别是，我们提出了对抗性可逆变换，它可以看作是从图像到图像的映射，对数据样本进行轻微的修改，使其在视觉特征损失很小的情况下变得无法被机器学习模型学习。同时，可以解锁数据集的可学习性，并使用相应的密钥正常训练模型。所提出的可学习性锁利用了对相同标签的数据样本应用通用变换函数的类级扰动。这确保了可以通过简单的逆变换容易地恢复可学习性，同时保持难以被检测或反向工程。我们通过实验证明了该方法在视觉分类任务上的成功和实用性。



## **44. And/or trade-off in artificial neurons: impact on adversarial robustness**

和/或人工神经元的权衡：对对抗性稳健性的影响 cs.LG

**SubmitDate**: 2022-02-03    [paper-pdf](http://arxiv.org/pdf/2102.07389v2)

**Authors**: Alessandro Fontana

**Abstracts**: Since its discovery in 2013, the phenomenon of adversarial examples has attracted a growing amount of attention from the machine learning community. A deeper understanding of the problem could lead to a better comprehension of how information is processed and encoded in neural networks and, more in general, could help to solve the issue of interpretability in machine learning. Our idea to increase adversarial resilience starts with the observation that artificial neurons can be divided in two broad categories: AND-like neurons and OR-like neurons. Intuitively, the former are characterised by a relatively low number of combinations of input values which trigger neuron activation, while for the latter the opposite is true. Our hypothesis is that the presence in a network of a sufficiently high number of OR-like neurons could lead to classification "brittleness" and increase the network's susceptibility to adversarial attacks. After constructing an operational definition of a neuron AND-like behaviour, we proceed to introduce several measures to increase the proportion of AND-like neurons in the network: L1 norm weight normalisation; application of an input filter; comparison between the neuron output's distribution obtained when the network is fed with the actual data set and the distribution obtained when the network is fed with a randomised version of the former called "scrambled data set". Tests performed on the MNIST data set hint that the proposed measures could represent an interesting direction to explore.

摘要: 自2013年发现以来，对抗性例子现象越来越受到机器学习界的关注。更深入地理解这个问题可以更好地理解信息是如何在神经网络中处理和编码的，更广泛地说，可以帮助解决机器学习中的可解释性问题。我们提高对抗韧性的想法始于观察到人工神经元可以分为两大类：类AND神经元和类OR神经元。直观地说，前者的特点是触发神经元激活的输入值组合数量相对较少，而后者则相反。我们的假设是，在网络中存在足够多的类OR神经元可能导致分类为“脆性”，并增加网络对对手攻击的敏感性。在构造了神经元和类神经元行为的操作定义后，我们接着引入了几项措施来提高网络中AND类神经元的比例：L1范数权重归一化；应用输入过滤；将用实际数据集馈送网络时获得的神经元输出分布与用前者的随机版本(称为“置乱数据集”)馈送网络时获得的分布进行比较。在MNIST数据集上执行的测试提示，建议的措施可能代表一个有趣的探索方向。



## **45. Revisiting and Advancing Fast Adversarial Training Through The Lens of Bi-Level Optimization**

用双层优化镜头重温和推进快速对抗性训练 cs.LG

**SubmitDate**: 2022-02-03    [paper-pdf](http://arxiv.org/pdf/2112.12376v4)

**Authors**: Yihua Zhang, Guanhua Zhang, Prashant Khanduri, Mingyi Hong, Shiyu Chang, Sijia Liu

**Abstracts**: Adversarial training (AT) is a widely recognized defense mechanism to gain the robustness of deep neural networks against adversarial attacks. It is built on min-max optimization (MMO), where the minimizer (i.e., defender) seeks a robust model to minimize the worst-case training loss in the presence of adversarial examples crafted by the maximizer (i.e., attacker). However, the conventional MMO method makes AT hard to scale. Thus, Fast-AT and other recent algorithms attempt to simplify MMO by replacing its maximization step with the single gradient sign-based attack generation step. Although easy to implement, FAST-AT lacks theoretical guarantees, and its empirical performance is unsatisfactory due to the issue of robust catastrophic overfitting when training with strong adversaries. In this paper, we advance Fast-AT from the fresh perspective of bi-level optimization (BLO). We first show that the commonly-used Fast-AT is equivalent to using a stochastic gradient algorithm to solve a linearized BLO problem involving a sign operation. However, the discrete nature of the sign operation makes it difficult to understand the algorithm performance. Inspired by BLO, we design and analyze a new set of robust training algorithms termed Fast Bi-level AT (Fast-BAT), which effectively defends sign-based projected gradient descent (PGD) attacks without using any gradient sign method or explicit robust regularization. In practice, we show that our method yields substantial robustness improvements over multiple baselines across multiple models and datasets. All code for reproducing the experiments in this paper is at https://github.com/NormalUhr/Fast_BAT.

摘要: 对抗性训练(AT)是一种公认的防御机制，用来增强深层神经网络对对手攻击的鲁棒性。它建立在最小-最大优化(MMO)的基础上，其中最小化器(即防御者)在存在最大化者(即攻击者)制作的对抗性示例的情况下寻求鲁棒模型来最小化最坏情况下的训练损失。然而，传统的MMO方法使得AT难以扩展。因此，Fast-AT和其他最近的算法试图通过将其最大化步骤替换为基于单个梯度符号的攻击生成步骤来简化MMO。FAST-AT虽然易于实现，但缺乏理论保证，在与强对手进行训练时，由于存在稳健的灾难性过拟合问题，其经验性能也不尽如人意。本文从双层优化(BLO)的新视角出发，提出了FAST-AT算法。我们首先证明了常用的Fast-AT算法等价于使用随机梯度算法来求解涉及符号运算的线性化BLO问题。然而，符号运算的离散性使得很难理解算法的性能。受BLO的启发，我们设计并分析了一套新的鲁棒训练算法，称为Fast Bi-Level AT(Fast-BAT)，该算法无需使用任何梯度符号方法或显式鲁棒正则化，即可有效防御基于符号的投影梯度下降(PGD)攻击。在实践中，我们证明了我们的方法在跨越多个模型和数据集的多个基线上产生了显著的健壮性改进。重现这篇论文中实验的所有代码都在https://github.com/NormalUhr/Fast_BAT.



## **46. Maximum Likelihood Uncertainty Estimation: Robustness to Outliers**

最大似然不确定性估计：对异常值的稳健性 cs.LG

8 Pages, 8 Figures, The Thirty-Sixth AAAI Conference on Artificial  Intelligence (AAAI-22), The AAAI's Workshop on Artificial Intelligence Safety

**SubmitDate**: 2022-02-03    [paper-pdf](http://arxiv.org/pdf/2202.03870v1)

**Authors**: Deebul S. Nair, Nico Hochgeschwender, Miguel A. Olivares-Mendez

**Abstracts**: We benchmark the robustness of maximum likelihood based uncertainty estimation methods to outliers in training data for regression tasks. Outliers or noisy labels in training data results in degraded performances as well as incorrect estimation of uncertainty. We propose the use of a heavy-tailed distribution (Laplace distribution) to improve the robustness to outliers. This property is evaluated using standard regression benchmarks and on a high-dimensional regression task of monocular depth estimation, both containing outliers. In particular, heavy-tailed distribution based maximum likelihood provides better uncertainty estimates, better separation in uncertainty for out-of-distribution data, as well as better detection of adversarial attacks in the presence of outliers.

摘要: 我们对基于最大似然的不确定性估计方法对回归任务训练数据中的离群值的稳健性进行了基准测试。训练数据中的离群值或噪声标签会导致性能下降以及对不确定性的错误估计。我们提出使用重尾分布(拉普拉斯分布)来提高对异常值的鲁棒性。使用标准回归基准和单目深度估计的高维回归任务来评估该属性，两者都包含异常值。特别地，基于重尾分布的最大似然法提供了更好的不确定性估计，更好地分离分布外数据的不确定性，以及在存在异常值的情况下更好地检测对抗性攻击。



## **47. Adversarial Rain Attack and Defensive Deraining for DNN Perception**

DNN感知的对抗性降雨攻击和防御训练 cs.CV

**SubmitDate**: 2022-02-03    [paper-pdf](http://arxiv.org/pdf/2009.09205v2)

**Authors**: Liming Zhai, Felix Juefei-Xu, Qing Guo, Xiaofei Xie, Lei Ma, Wei Feng, Shengchao Qin, Yang Liu

**Abstracts**: Rain often poses inevitable threats to deep neural network (DNN) based perception systems, and a comprehensive investigation of the potential risks of the rain to DNNs is of great importance. However, it is rather difficult to collect or synthesize rainy images that can represent all rain situations that would possibly occur in the real world. To this end, in this paper, we start from a new perspective and propose to combine two totally different studies, i.e., rainy image synthesis and adversarial attack. We first present an adversarial rain attack, with which we could simulate various rain situations with the guidance of deployed DNNs and reveal the potential threat factors that can be brought by rain. In particular, we design a factor-aware rain generation that synthesizes rain streaks according to the camera exposure process and models the learnable rain factors for adversarial attack. With this generator, we perform the adversarial rain attack against the image classification and object detection. To defend the DNNs from the negative rain effect, we also present a defensive deraining strategy, for which we design an adversarial rain augmentation that uses mixed adversarial rain layers to enhance deraining models for downstream DNN perception. Our large-scale evaluation on various datasets demonstrates that our synthesized rainy images with realistic appearances not only exhibit strong adversarial capability against DNNs, but also boost the deraining models for defensive purposes, building the foundation for further rain-robust perception studies.

摘要: 降雨往往对基于深度神经网络(DNN)的感知系统造成不可避免的威胁，全面调查降雨对DNN的潜在风险具有重要意义。然而，要收集或合成能够代表现实世界中可能发生的所有降雨情况的降雨图像是相当困难的。为此，本文从一个新的角度出发，提出将雨图像合成和对抗性攻击这两个完全不同的研究结合起来。我们首先提出了一种对抗性的雨攻击，利用它可以在部署的DNN的指导下模拟各种降雨情况，并揭示雨可能带来的潜在威胁因素。特别地，我们设计了一种因子感知的雨生成算法，它根据相机曝光过程合成雨带，并对可学习的雨因子进行建模以进行对抗性攻击。利用该生成器，对图像分类和目标检测进行了对抗性降雨攻击。为了保护DNN不受降雨负效应的影响，我们还提出了一种防御性去雨策略，为此，我们设计了一种对抗性增雨策略，使用混合对抗性雨层来增强DNN下游感知的去雨模型。我们在不同数据集上的大规模评估表明，我们合成的外观逼真的雨图像不仅表现出对DNN的强大对抗能力，而且增强了用于防御目的的去雨模型，为进一步的雨鲁棒性感知研究奠定了基础。



## **48. Self-Adversarial Training incorporating Forgery Attention for Image Forgery Localization**

图像伪造定位中结合伪造注意的自我对抗训练 cs.CV

accepted by TIFS

**SubmitDate**: 2022-02-03    [paper-pdf](http://arxiv.org/pdf/2107.02434v2)

**Authors**: Long Zhuo, Shunquan Tan, Bin Li, Jiwu Huang

**Abstracts**: Image editing techniques enable people to modify the content of an image without leaving visual traces and thus may cause serious security risks. Hence the detection and localization of these forgeries become quite necessary and challenging. Furthermore, unlike other tasks with extensive data, there is usually a lack of annotated forged images for training due to annotation difficulties. In this paper, we propose a self-adversarial training strategy and a reliable coarse-to-fine network that utilizes a self-attention mechanism to localize forged regions in forgery images. The self-attention module is based on a Channel-Wise High Pass Filter block (CW-HPF). CW-HPF leverages inter-channel relationships of features and extracts noise features by high pass filters. Based on the CW-HPF, a self-attention mechanism, called forgery attention, is proposed to capture rich contextual dependencies of intrinsic inconsistency extracted from tampered regions. Specifically, we append two types of attention modules on top of CW-HPF respectively to model internal interdependencies in spatial dimension and external dependencies among channels. We exploit a coarse-to-fine network to enhance the noise inconsistency between original and tampered regions. More importantly, to address the issue of insufficient training data, we design a self-adversarial training strategy that expands training data dynamically to achieve more robust performance. Specifically, in each training iteration, we perform adversarial attacks against our network to generate adversarial examples and train our model on them. Extensive experimental results demonstrate that our proposed algorithm steadily outperforms state-of-the-art methods by a clear margin in different benchmark datasets.

摘要: 图像编辑技术使人们能够在不留下视觉痕迹的情况下修改图像的内容，从而可能造成严重的安全隐患。因此，对这些伪造品的检测和定位变得十分必要和具有挑战性。此外，与其他具有大量数据的任务不同，由于标注困难，通常缺少用于训练的带标注的伪造图像。在本文中，我们提出了一种自对抗的训练策略和一个可靠的从粗到精的网络，该网络利用自我注意机制来定位伪造图像中的伪造区域。自我注意模块是基于频道智慧型高通过滤挡路(CW-HPF)的。CW-HPF利用特征的通道间关系，通过高通滤波器提取噪声特征。在CW-HPF的基础上，提出了一种称为伪造注意的自我注意机制，用于捕获从篡改区域中提取的内在不一致性的丰富上下文依赖。具体地说，我们在CW-HPF之上分别添加了两种类型的注意模块来建模空间维度上的内部相互依赖和通道之间的外部依赖。我们利用一个由粗到精的网络来增强原始区域和篡改区域之间的噪声不一致性。更重要的是，为了解决训练数据不足的问题，我们设计了一种自对抗性训练策略，动态扩展训练数据以获得更健壮的性能。具体地说，在每个训练迭代中，我们对我们的网络执行对抗性攻击，以生成对抗性示例，并在此基础上训练我们的模型。大量的实验结果表明，在不同的基准数据集上，我们提出的算法比最先进的方法有明显的优势。



## **49. TPC: Transformation-Specific Smoothing for Point Cloud Models**

TPC：点云模型的变换特定平滑 cs.CV

18 pages

**SubmitDate**: 2022-02-03    [paper-pdf](http://arxiv.org/pdf/2201.12733v2)

**Authors**: Wenda Chu, Linyi Li, Bo Li

**Abstracts**: Point cloud models with neural network architectures have achieved great success and have been widely used in safety-critical applications, such as Lidar-based recognition systems in autonomous vehicles. However, such models are shown vulnerable against adversarial attacks which aim to apply stealthy semantic transformations such as rotation and tapering to mislead model predictions. In this paper, we propose a transformation-specific smoothing framework TPC, which provides tight and scalable robustness guarantees for point cloud models against semantic transformation attacks. We first categorize common 3D transformations into three categories: additive (e.g., shearing), composable (e.g., rotation), and indirectly composable (e.g., tapering), and we present generic robustness certification strategies for all categories respectively. We then specify unique certification protocols for a range of specific semantic transformations and their compositions. Extensive experiments on several common 3D transformations show that TPC significantly outperforms the state of the art. For example, our framework boosts the certified accuracy against twisting transformation along z-axis (within 20$^\circ$) from 20.3$\%$ to 83.8$\%$.

摘要: 具有神经网络结构的点云模型已经取得了巨大的成功，并在安全关键应用中得到了广泛的应用，例如基于激光雷达的自动车辆识别系统。然而，这样的模型容易受到敌意攻击，这些攻击的目的是应用诸如旋转和渐变等隐蔽的语义转换来误导模型预测。本文提出了一种特定于变换的平滑框架TPC，它为点云模型抵抗语义变换攻击提供了紧凑且可伸缩的健壮性保证。我们首先将常见的3D变换分为三类：加性变换(如剪切)、可合成变换(如旋转变换)和间接合成变换(如锥化变换)，并分别针对所有类别提出了通用的鲁棒性认证策略。然后，我们为一系列特定的语义转换及其组合指定唯一的认证协议。对几种常见的3D变换进行的大量实验表明，TPC的性能明显优于现有技术。例如，我们的框架将沿z轴(在20$^\cic$内)抗扭曲变换的认证精度从20.3$\$提高到83.8$\$。



## **50. Drawing Robust Scratch Tickets: Subnetworks with Inborn Robustness Are Found within Randomly Initialized Networks**

绘制健壮的暂存券：在随机初始化的网络中发现具有天生健壮性的子网 cs.LG

Accepted at NeurIPS 2021

**SubmitDate**: 2022-02-03    [paper-pdf](http://arxiv.org/pdf/2110.14068v3)

**Authors**: Yonggan Fu, Qixuan Yu, Yang Zhang, Shang Wu, Xu Ouyang, David Cox, Yingyan Lin

**Abstracts**: Deep Neural Networks (DNNs) are known to be vulnerable to adversarial attacks, i.e., an imperceptible perturbation to the input can mislead DNNs trained on clean images into making erroneous predictions. To tackle this, adversarial training is currently the most effective defense method, by augmenting the training set with adversarial samples generated on the fly. Interestingly, we discover for the first time that there exist subnetworks with inborn robustness, matching or surpassing the robust accuracy of the adversarially trained networks with comparable model sizes, within randomly initialized networks without any model training, indicating that adversarial training on model weights is not indispensable towards adversarial robustness. We name such subnetworks Robust Scratch Tickets (RSTs), which are also by nature efficient. Distinct from the popular lottery ticket hypothesis, neither the original dense networks nor the identified RSTs need to be trained. To validate and understand this fascinating finding, we further conduct extensive experiments to study the existence and properties of RSTs under different models, datasets, sparsity patterns, and attacks, drawing insights regarding the relationship between DNNs' robustness and their initialization/overparameterization. Furthermore, we identify the poor adversarial transferability between RSTs of different sparsity ratios drawn from the same randomly initialized dense network, and propose a Random RST Switch (R2S) technique, which randomly switches between different RSTs, as a novel defense method built on top of RSTs. We believe our findings about RSTs have opened up a new perspective to study model robustness and extend the lottery ticket hypothesis.

摘要: 深度神经网络(DNNs)很容易受到敌意攻击，即输入的不知不觉的扰动会误导训练在干净图像上的DNN做出错误的预测。对撞击来说，对抗性训练是目前最有效的防御方法，通过使用飞翔上生成的对抗性样本来扩大训练集。有趣的是，我们首次发现，在没有任何模型训练的随机初始化网络中，存在具有天生鲁棒性的子网络，其鲁棒性精度与具有相似模型大小的对抗性训练网络相当或超过，这表明对抗性模型权重的训练对于对抗性鲁棒性来说并不是必不可少的。我们将这样的子网命名为健壮的暂存票(RST)，它本质上也是有效的。与流行的彩票假设不同，原始的密集网络和识别出的RST都不需要训练。为了验证和理解这一有趣的发现，我们进一步进行了大量的实验，研究了不同模型、数据集、稀疏模式和攻击下RST的存在和性质，得出了DNNs的健壮性与其初始化/过参数化之间的关系。此外，我们还发现了来自同一随机初始化密集网络的不同稀疏比的RST之间的对抗性较差，并提出了一种在RST之上随机切换的随机RST切换(R2S)技术，作为一种新的防御方法。我们相信，我们关于RST的发现为研究模型的稳健性和扩展彩票假说开辟了一个新的视角。



