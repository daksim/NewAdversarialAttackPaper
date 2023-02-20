# Latest Adversarial Attack Papers
**update at 2023-02-20 16:30:26**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. DETER: Design for Trust utilizing Rareness Reduction**

威慑：利用稀缺性减少的信任设计 cs.CR

**SubmitDate**: 2023-02-17    [abs](http://arxiv.org/abs/2302.08984v1) [paper-pdf](http://arxiv.org/pdf/2302.08984v1)

**Authors**: Aruna Jayasena, Prabhat Mishra

**Abstract**: Increasing design complexity and reduced time-to-market have motivated manufacturers to outsource some parts of the System-on-Chip (SoC) design flow to third-party vendors. This provides an opportunity for attackers to introduce hardware Trojans by constructing stealthy triggers consisting of rare events (e.g., rare signals, states, and transitions). There are promising test generation-based hardware Trojan detection techniques that rely on the activation of rare events. In this paper, we investigate rareness reduction as a design-for-trust solution to make it harder for an adversary to hide Trojans (easier for Trojan detection). Specifically, we analyze different avenues to reduce the potential rare trigger cases, including design diversity and area optimization. While there is a good understanding of the relationship between area, power, energy, and performance, this research provides a better insight into the dependency between area and security. Our experimental evaluation demonstrates that area reduction leads to a reduction in rareness. It also reveals that reducing rareness leads to faster Trojan detection as well as improved coverage by Trojan detection methods.

摘要: 不断增加的设计复杂性和缩短的上市时间促使制造商将片上系统(SoC)设计流程的某些部分外包给第三方供应商。这为攻击者提供了通过构建由罕见事件(例如罕见信号、状态和转换)组成的隐形触发器来引入硬件特洛伊木马程序的机会。有一些很有前途的基于测试生成的硬件特洛伊木马检测技术，它们依赖于罕见事件的激活。在这篇文章中，我们研究了稀有性减少作为一种信任设计解决方案，使对手更难隐藏特洛伊木马(更容易检测木马)。具体地说，我们分析了减少潜在罕见触发情况的不同途径，包括设计多样性和面积优化。虽然对面积、功率、能量和性能之间的关系有了很好的理解，但这项研究提供了对面积和安全之间的依赖关系的更好的洞察。我们的实验评估表明，面积减少会导致稀有性的减少。它还揭示了减少稀有性会导致木马检测的速度更快，并提高了木马检测方法的覆盖率。



## **2. Adversarial Contrastive Distillation with Adaptive Denoising**

对抗性对比蒸馏与自适应去噪 cs.CV

accepted for ICASSP 2023

**SubmitDate**: 2023-02-17    [abs](http://arxiv.org/abs/2302.08764v1) [paper-pdf](http://arxiv.org/pdf/2302.08764v1)

**Authors**: Yuzheng Wang, Zhaoyu Chen, Dingkang Yang, Yang Liu, Siao Liu, Wenqiang Zhang, Lizhe Qi

**Abstract**: Adversarial Robustness Distillation (ARD) is a novel method to boost the robustness of small models. Unlike general adversarial training, its robust knowledge transfer can be less easily restricted by the model capacity. However, the teacher model that provides the robustness of knowledge does not always make correct predictions, interfering with the student's robust performances. Besides, in the previous ARD methods, the robustness comes entirely from one-to-one imitation, ignoring the relationship between examples. To this end, we propose a novel structured ARD method called Contrastive Relationship DeNoise Distillation (CRDND). We design an adaptive compensation module to model the instability of the teacher. Moreover, we utilize the contrastive relationship to explore implicit robustness knowledge among multiple examples. Experimental results on multiple attack benchmarks show CRDND can transfer robust knowledge efficiently and achieves state-of-the-art performances.

摘要: 对抗稳健性蒸馏(ARD)是一种提高小模型稳健性的新方法。与一般的对抗性训练不同，其稳健的知识传递不太容易受到模型容量的限制。然而，提供知识稳健性的教师模型并不总是做出正确的预测，干扰了学生的稳健表现。此外，在以往的ARD方法中，鲁棒性完全来自一对一的模仿，忽略了样本之间的关系。为此，我们提出了一种新的结构化ARD方法，称为对比关系降噪蒸馏(CRDND)。我们设计了一个自适应补偿模块来模拟教师的不稳定性。此外，我们利用这种对比关系来探索多个实例之间的隐含稳健性知识。在多个攻击基准上的实验结果表明，CRDND能够有效地传递健壮的知识，并达到最先进的性能。



## **3. Adversarial Attacks against Windows PE Malware Detection: A Survey of the State-of-the-Art**

针对Windows PE恶意软件检测的对抗性攻击：现状综述 cs.CR

Accepted by ELSEVIER Computers & Security (COSE)

**SubmitDate**: 2023-02-17    [abs](http://arxiv.org/abs/2112.12310v5) [paper-pdf](http://arxiv.org/pdf/2112.12310v5)

**Authors**: Xiang Ling, Lingfei Wu, Jiangyu Zhang, Zhenqing Qu, Wei Deng, Xiang Chen, Yaguan Qian, Chunming Wu, Shouling Ji, Tianyue Luo, Jingzheng Wu, Yanjun Wu

**Abstract**: Malware has been one of the most damaging threats to computers that span across multiple operating systems and various file formats. To defend against ever-increasing and ever-evolving malware, tremendous efforts have been made to propose a variety of malware detection that attempt to effectively and efficiently detect malware so as to mitigate possible damages as early as possible. Recent studies have shown that, on the one hand, existing ML and DL techniques enable superior solutions in detecting newly emerging and previously unseen malware. However, on the other hand, ML and DL models are inherently vulnerable to adversarial attacks in the form of adversarial examples. In this paper, we focus on malware with the file format of portable executable (PE) in the family of Windows operating systems, namely Windows PE malware, as a representative case to study the adversarial attack methods in such adversarial settings. To be specific, we start by first outlining the general learning framework of Windows PE malware detection based on ML/DL and subsequently highlighting three unique challenges of performing adversarial attacks in the context of Windows PE malware. Then, we conduct a comprehensive and systematic review to categorize the state-of-the-art adversarial attacks against PE malware detection, as well as corresponding defenses to increase the robustness of Windows PE malware detection. Finally, we conclude the paper by first presenting other related attacks against Windows PE malware detection beyond the adversarial attacks and then shedding light on future research directions and opportunities. In addition, a curated resource list of adversarial attacks and defenses for Windows PE malware detection is also available at https://github.com/ryderling/adversarial-attacks-and-defenses-for-windows-pe-malware-detection.

摘要: 恶意软件一直是计算机面临的最具破坏性的威胁之一，这些威胁跨越多个操作系统和各种文件格式。为了防御不断增长和不断演变的恶意软件，人们做出了巨大的努力，提出了各种恶意软件检测方法，试图有效和高效地检测恶意软件，以便尽早减轻可能的损害。最近的研究表明，一方面，现有的ML和DL技术能够在检测新出现的和以前未见过的恶意软件方面提供更好的解决方案。然而，另一方面，ML和DL模型天生就容易受到对抗性例子形式的对抗性攻击。本文以Windows操作系统家族中具有可移植可执行文件(PE)文件格式的恶意软件，即Windows PE恶意软件为典型案例，研究这种对抗性环境下的对抗性攻击方法。具体地说，我们首先概述了基于ML/DL的Windows PE恶意软件检测的一般学习框架，然后重点介绍了在Windows PE恶意软件环境中执行对抗性攻击的三个独特挑战。然后，我们对针对PE恶意软件检测的对抗性攻击进行了全面系统的回顾，并对相应的防御措施进行了分类，以增加Windows PE恶意软件检测的健壮性。最后，我们首先介绍了Windows PE恶意软件检测中除了对抗性攻击之外的其他相关攻击，并对未来的研究方向和机会进行了展望。此外，https://github.com/ryderling/adversarial-attacks-and-defenses-for-windows-pe-malware-detection.上还提供了针对Windows PE恶意软件检测的对抗性攻击和防御的精选资源列表



## **4. High-frequency Matters: An Overwriting Attack and defense for Image-processing Neural Network Watermarking**

高频问题：一种图像处理神经网络水印的覆盖攻击与防御 cs.CR

**SubmitDate**: 2023-02-17    [abs](http://arxiv.org/abs/2302.08637v1) [paper-pdf](http://arxiv.org/pdf/2302.08637v1)

**Authors**: Huajie Chen, Tianqing Zhu, Chi Liu, Shui Yu, Wanlei Zhou

**Abstract**: In recent years, there has been significant advancement in the field of model watermarking techniques. However, the protection of image-processing neural networks remains a challenge, with only a limited number of methods being developed. The objective of these techniques is to embed a watermark in the output images of the target generative network, so that the watermark signal can be detected in the output of a surrogate model obtained through model extraction attacks. This promising technique, however, has certain limits. Analysis of the frequency domain reveals that the watermark signal is mainly concealed in the high-frequency components of the output. Thus, we propose an overwriting attack that involves forging another watermark in the output of the generative network. The experimental results demonstrate the efficacy of this attack in sabotaging existing watermarking schemes for image-processing networks, with an almost 100% success rate. To counter this attack, we devise an adversarial framework for the watermarking network. The framework incorporates a specially designed adversarial training step, where the watermarking network is trained to defend against the overwriting network, thereby enhancing its robustness. Additionally, we observe an overfitting phenomenon in the existing watermarking method, which can render it ineffective. To address this issue, we modify the training process to eliminate the overfitting problem.

摘要: 近年来，模型数字水印技术取得了长足的进步。然而，图像处理神经网络的保护仍然是一个挑战，只有有限数量的方法正在开发中。这些技术的目的是在目标生成网络的输出图像中嵌入水印，以便在通过模型提取攻击获得的代理模型的输出中检测到水印信号。然而，这项前景看好的技术有一定的局限性。频域分析表明，水印信号主要隐藏在输出的高频分量中。因此，我们提出了一种覆盖攻击，该攻击涉及在生成式网络的输出中伪造另一个水印。实验结果表明，该攻击能够有效地破坏现有的图像处理网络水印方案，成功率接近100%。为了对抗这种攻击，我们设计了一个针对水印网络的对抗性框架。该框架结合了专门设计的对抗性训练步骤，其中训练水印网络以防御覆盖网络，从而增强其稳健性。此外，我们还观察到了现有水印方法中存在的过度拟合现象，这可能会使其失效。为了解决这个问题，我们修改了培训过程，以消除过度匹配问题。



## **5. PACMAN Attack: A Mobility-Powered Attack in Private 5G-Enabled Industrial Automation System**

吃豆人攻击：私有5G工业自动化系统中的移动性攻击 cs.CR

6 pages, 7 Figures, Accepted in IEEE International Conference on  Communications 2023

**SubmitDate**: 2023-02-16    [abs](http://arxiv.org/abs/2302.08563v1) [paper-pdf](http://arxiv.org/pdf/2302.08563v1)

**Authors**: Md Rashedur Rahman, Moinul Hossain, Jiang Xie

**Abstract**: 3GPP has introduced Private 5G to support the next-generation industrial automation system (IAS) due to the versatility and flexibility of 5G architecture. Besides the 3.5GHz CBRS band, unlicensed spectrum bands, like 5GHz, are considered as an additional medium because of their free and abundant nature. However, while utilizing the unlicensed band, industrial equipment must coexist with incumbents, e.g., Wi-Fi, which could introduce new security threats and resuscitate old ones. In this paper, we propose a novel attack strategy conducted by a mobility-enabled malicious Wi-Fi access point (mmAP), namely \textit{PACMAN} attack, to exploit vulnerabilities introduced by heterogeneous coexistence. A mmAP is capable of moving around the physical surface to identify mission-critical devices, hopping through the frequency domain to detect the victim's operating channel, and launching traditional MAC layer-based attacks. The multi-dimensional mobility of the attacker makes it impervious to state-of-the-art detection techniques that assume static adversaries. In addition, we propose a novel Markov Decision Process (MDP) based framework to intelligently design an attacker's multi-dimensional mobility in space and frequency. Mathematical analysis and extensive simulation results exhibit the adverse effect of the proposed mobility-powered attack.

摘要: 由于5G架构的通用性和灵活性，3GPP引入了专用5G来支持下一代工业自动化系统(IAS)。除了3.5 GHz的CBRS频段外，未经许可的频段，如5 GHz，由于其免费和丰富的性质，被视为额外的介质。然而，在使用未经许可的频段时，工业设备必须与现有设备共存，例如Wi-Fi，这可能会带来新的安全威胁并重振旧的安全威胁。为了利用异质共存带来的漏洞，提出了一种由移动性启用的恶意Wi-Fi接入点(MMAP)实施的攻击策略，即Texttit{Pacman}攻击。MMAP能够在物理表面上移动以识别关键任务设备，在频域中跳跃以检测受害者的工作通道，并发起传统的基于MAC层的攻击。攻击者的多维移动性使其不受假定静态对手的最先进检测技术的影响。此外，我们还提出了一种基于马尔可夫决策过程(MDP)的框架来智能地设计攻击者在空间和频率上的多维移动性。数学分析和大量的仿真结果证明了所提出的移动性动力攻击的不利影响。



## **6. Deep Composite Face Image Attacks: Generation, Vulnerability and Detection**

深度复合人脸图像攻击：产生、漏洞和检测 cs.CV

**SubmitDate**: 2023-02-16    [abs](http://arxiv.org/abs/2211.11039v2) [paper-pdf](http://arxiv.org/pdf/2211.11039v2)

**Authors**: Jag Mohan Singh, Raghavendra Ramachandra

**Abstract**: Face manipulation attacks have drawn the attention of biometric researchers because of their vulnerability to Face Recognition Systems (FRS). This paper proposes a novel scheme to generate Composite Face Image Attacks (CFIA) based on facial attributes using Generative Adversarial Networks (GANs). Given the face images corresponding to two unique data subjects, the proposed CFIA method will independently generate the segmented facial attributes, then blend them using transparent masks to generate the CFIA samples. We generate $526$ unique CFIA combinations of facial attributes for each pair of contributory data subjects. Extensive experiments are carried out on our newly generated CFIA dataset consisting of 1000 unique identities with 2000 bona fide samples and 526000 CFIA samples, thus resulting in an overall 528000 face image samples. {{We present a sequence of experiments to benchmark the attack potential of CFIA samples using four different automatic FRS}}. We introduced a new metric named Generalized Morphing Attack Potential (G-MAP) to benchmark the vulnerability of generated attacks on FRS effectively. Additional experiments are performed on the representative subset of the CFIA dataset to benchmark both perceptual quality and human observer response. Finally, the CFIA detection performance is benchmarked using three different single image based face Morphing Attack Detection (MAD) algorithms. The source code of the proposed method together with CFIA dataset will be made publicly available: \url{https://github.com/jagmohaniiit/LatentCompositionCode}

摘要: 人脸操纵攻击因其易受人脸识别系统(FRS)攻击而受到生物特征识别研究人员的关注。提出了一种利用生成性对抗网络(GANS)生成基于人脸属性的复合人脸图像攻击(CFIA)的新方案。在给定两个不同数据对象对应的人脸图像的情况下，CFIA方法将独立地生成分割后的人脸属性，然后使用透明掩膜进行混合以生成CFIA样本。我们为每对有贡献的数据对象生成$526$独特的面部属性CFIA组合。在我们新生成的包含1,000个唯一身份的CFIA数据集上进行了大量的实验，其中包含2,000个真实样本和526000个CFIA样本，从而得到总共528000个人脸图像样本。{{我们提供了一系列实验，以使用四种不同的自动FRS对CFIA样本的攻击潜力进行基准测试}}我们引入了一种新的度量--广义变形攻击潜力(G-MAP)来有效地评估生成攻击对FRS的脆弱性。在CFIA数据集的代表性子集上进行了其他实验，以对感知质量和人类观察者的反应进行基准测试。最后，使用三种不同的基于单幅图像的人脸变形攻击检测(MAD)算法对CFIA检测性能进行了基准测试。建议的方法的源代码和CFIA数据集将公开提供：\url{https://github.com/jagmohaniiit/LatentCompositionCode}



## **7. BITE: Textual Backdoor Attacks with Iterative Trigger Injection**

BITE：使用迭代触发器注入的文本后门攻击 cs.CL

**SubmitDate**: 2023-02-16    [abs](http://arxiv.org/abs/2205.12700v2) [paper-pdf](http://arxiv.org/pdf/2205.12700v2)

**Authors**: Jun Yan, Vansh Gupta, Xiang Ren

**Abstract**: Backdoor attacks have become an emerging threat to NLP systems. By providing poisoned training data, the adversary can embed a ``backdoor'' into the victim model, which allows input instances satisfying certain textual patterns (e.g., containing a keyword) to be predicted as a target label of the adversary's choice. In this paper, we demonstrate that it's possible to design a backdoor attack that is both stealthy (i.e., hard to notice) and effective (i.e., has a high attack success rate). We propose BITE, a backdoor attack that poisons the training data to establish strong correlations between the target label and some ``trigger words'', by iteratively injecting them into target-label instances through natural word-level perturbations. The poisoned training data instruct the victim model to predict the target label on inputs containing trigger words, forming the backdoor. Experiments on four medium-sized text classification datasets show that BITE is significantly more effective than baselines while maintaining decent stealthiness, raising alarm on the usage of untrusted training data. We further propose a defense method named DeBITE based on potential trigger word removal, which outperforms existing methods on defending BITE and generalizes well to defending other backdoor attacks.

摘要: 后门攻击已成为对NLP系统的新威胁。通过提供有毒的训练数据，敌手可以在受害者模型中嵌入“后门”，这允许满足某些文本模式(例如，包含关键字)的输入实例被预测为敌手选择的目标标签。在本文中，我们证明了设计一种既隐蔽(即难以察觉)又有效(即具有高攻击成功率)的后门攻击是可能的。我们提出了BITE，一种毒化训练数据的后门攻击，通过自然的词级扰动迭代地将它们注入到目标标签实例中，从而在目标标签和一些“触发词”之间建立强相关性。有毒的训练数据指示受害者模型预测包含触发词的输入上的目标标签，从而形成后门。在四个中等大小的文本分类数据集上的实验表明，BITE在保持良好的隐蔽性的同时明显比基线更有效，这对使用不可信的训练数据发出了警报。在此基础上，提出了一种基于潜在触发字移除的防御方法DeBITE，该方法在防御BITE攻击方面优于已有的方法，并能很好地推广到其他后门攻击的防御。



## **8. On the Effect of Adversarial Training Against Invariance-based Adversarial Examples**

对抗性训练对基于不变差的对抗性实例的影响 cs.LG

**SubmitDate**: 2023-02-16    [abs](http://arxiv.org/abs/2302.08257v1) [paper-pdf](http://arxiv.org/pdf/2302.08257v1)

**Authors**: Roland Rauter, Martin Nocker, Florian Merkle, Pascal Schöttle

**Abstract**: Adversarial examples are carefully crafted attack points that are supposed to fool machine learning classifiers. In the last years, the field of adversarial machine learning, especially the study of perturbation-based adversarial examples, in which a perturbation that is not perceptible for humans is added to the images, has been studied extensively. Adversarial training can be used to achieve robustness against such inputs. Another type of adversarial examples are invariance-based adversarial examples, where the images are semantically modified such that the predicted class of the model does not change, but the class that is determined by humans does. How to ensure robustness against this type of adversarial examples has not been explored yet. This work addresses the impact of adversarial training with invariance-based adversarial examples on a convolutional neural network (CNN).   We show that when adversarial training with invariance-based and perturbation-based adversarial examples is applied, it should be conducted simultaneously and not consecutively. This procedure can achieve relatively high robustness against both types of adversarial examples. Additionally, we find that the algorithm used for generating invariance-based adversarial examples in prior work does not correctly determine the labels and therefore we use human-determined labels.

摘要: 对抗性的例子是精心设计的攻击点，被认为是为了愚弄机器学习分类器。在过去的几年里，对抗性机器学习领域得到了广泛的研究，特别是基于扰动的对抗性实例的研究，其中在图像中添加了人类无法感知的扰动。对抗性训练可以用来实现对这种输入的稳健性。另一种类型的对抗性示例是基于不变性的对抗性示例，其中图像被语义修改，使得模型的预测类别不改变，但由人类确定的类别改变。如何确保对这种类型的对抗性例子的健壮性还没有被探索。这项工作解决了基于不变性的对抗性例子对卷积神经网络(CNN)的对抗性训练的影响。我们指出，当使用基于不变性和基于扰动的对抗性实例进行对抗性训练时，应该同时进行，而不是连续进行。该过程对两种类型的对抗性例子都具有较高的稳健性。此外，我们发现以前的工作中用于生成基于不变性的对抗性示例的算法不能正确地确定标签，因此我们使用了人类确定的标签。



## **9. Signaling Storm Detection in IIoT Network based on the Open RAN Architecture**

基于开放RAN架构的IIoT网络信令风暴检测 cs.NI

**SubmitDate**: 2023-02-16    [abs](http://arxiv.org/abs/2302.08239v1) [paper-pdf](http://arxiv.org/pdf/2302.08239v1)

**Authors**: Marcin Hoffmann, Pawel Kryszkiewicz

**Abstract**: The Industrial Internet of Things devices due to their low cost and complexity are exposed to being hacked and utilized to attack the network infrastructure causing a so-called Signaling Storm. In this paper, we propose to utilize the Open Radio Access Network (O-RAN) architecture, to monitor the control plane messages in order to detect the activity of adversaries at its early stage.

摘要: 工业物联网设备由于其低成本和复杂性，容易被黑客利用来攻击网络基础设施，引发所谓的信令风暴。在本文中，我们提出利用开放式无线接入网络(O-RAN)体系结构来监控控制平面消息，以便在其早期阶段检测到攻击者的活动。



## **10. Masking and Mixing Adversarial Training**

掩饰和混合对抗性训练 cs.CV

**SubmitDate**: 2023-02-16    [abs](http://arxiv.org/abs/2302.08066v1) [paper-pdf](http://arxiv.org/pdf/2302.08066v1)

**Authors**: Hiroki Adachi, Tsubasa Hirakawa, Takayoshi Yamashita, Hironobu Fujiyoshi, Yasunori Ishii, Kazuki Kozuka

**Abstract**: While convolutional neural networks (CNNs) have achieved excellent performances in various computer vision tasks, they often misclassify with malicious samples, a.k.a. adversarial examples. Adversarial training is a popular and straightforward technique to defend against the threat of adversarial examples. Unfortunately, CNNs must sacrifice the accuracy of standard samples to improve robustness against adversarial examples when adversarial training is used. In this work, we propose Masking and Mixing Adversarial Training (M2AT) to mitigate the trade-off between accuracy and robustness. We focus on creating diverse adversarial examples during training. Specifically, our approach consists of two processes: 1) masking a perturbation with a binary mask and 2) mixing two partially perturbed images. Experimental results on CIFAR-10 dataset demonstrate that our method achieves better robustness against several adversarial attacks than previous methods.

摘要: 虽然卷积神经网络(CNN)在各种计算机视觉任务中取得了优异的性能，但它们经常与恶意样本发生误分类，即。对抗性的例子。对抗性训练是一种流行而直接的技术，用来抵御对抗性例子的威胁。不幸的是，当使用对抗性训练时，CNN必须牺牲标准样本的准确性来提高对对抗性示例的稳健性。在这项工作中，我们提出了掩蔽和混合对手训练(M2AT)来缓解准确性和稳健性之间的权衡。我们专注于在培训期间创造不同的对抗性例子。具体地说，我们的方法包括两个过程：1)用二值掩模掩盖扰动；2)混合两个部分扰动的图像。在CIFAR-10数据集上的实验结果表明，与以往的方法相比，该方法对多种敌意攻击具有更好的稳健性。



## **11. Graph Adversarial Immunization for Certifiable Robustness**

图对抗免疫的可证明稳健性 cs.LG

**SubmitDate**: 2023-02-16    [abs](http://arxiv.org/abs/2302.08051v1) [paper-pdf](http://arxiv.org/pdf/2302.08051v1)

**Authors**: Shuchang Tao, Huawei Shen, Qi Cao, Yunfan Wu, Liang Hou, Xueqi Cheng

**Abstract**: Despite achieving great success, graph neural networks (GNNs) are vulnerable to adversarial attacks. Existing defenses focus on developing adversarial training or robust GNNs. However, little research attention is paid to the potential and practice of immunization on graphs. In this paper, we propose and formulate graph adversarial immunization, i.e., vaccinating part of graph structure to improve certifiable robustness of graph against any admissible adversarial attack. We first propose edge-level immunization to vaccinate node pairs. Despite the primary success, such edge-level immunization cannot defend against emerging node injection attacks, since it only immunizes existing node pairs. To this end, we further propose node-level immunization. To circumvent computationally expensive combinatorial optimization when solving adversarial immunization, we design AdvImmune-Edge and AdvImmune-Node algorithms to effectively obtain the immune node pairs or nodes. Experiments demonstrate the superiority of AdvImmune methods. In particular, AdvImmune-Node remarkably improves the ratio of robust nodes by 79%, 294%, and 100%, after immunizing only 5% nodes. Furthermore, AdvImmune methods show excellent defensive performance against various attacks, outperforming state-of-the-art defenses. To the best of our knowledge, this is the first attempt to improve certifiable robustness from graph data perspective without losing performance on clean graphs, providing new insights into graph adversarial learning.

摘要: 尽管图神经网络(GNN)取得了巨大的成功，但它仍然容易受到对手的攻击。现有的防御系统侧重于发展对抗性训练或强大的GNN。然而，关于图的免疫的潜力和实践的研究却很少。在本文中，我们提出并建立了图对抗免疫，即接种部分图结构以提高图对任何允许的对抗攻击的可证明的稳健性。我们首先提出了边缘级别免疫来接种节点对。尽管取得了初步的成功，但这种边缘级别的免疫无法防御新出现的节点注入攻击，因为它只免疫现有的节点对。为此，我们进一步提出节点级免疫。在求解对抗性免疫问题时，为了避开计算量较大的组合优化问题，我们设计了AdvImmune-Edge算法和AdvImmune-Node算法来有效地获得免疫节点对或节点。实验证明了该方法的优越性。特别是，在仅免疫5%的节点后，AdvImmune-Node显著提高了健壮节点的比率，分别提高了79%、294%和100%。此外，AdvImmune方法在应对各种攻击时表现出出色的防御性能，表现优于最先进的防御。据我们所知，这是第一次尝试从图数据的角度提高可证明的稳健性，而不损失干净图的性能，为图的对抗性学习提供了新的见解。



## **12. Robust Mid-Pass Filtering Graph Convolutional Networks**

稳健的中通滤波图卷积网络 cs.LG

Accepted by WWW'23

**SubmitDate**: 2023-02-16    [abs](http://arxiv.org/abs/2302.08048v1) [paper-pdf](http://arxiv.org/pdf/2302.08048v1)

**Authors**: Jincheng Huang, Lun Du, Xu Chen, Qiang Fu, Shi Han, Dongmei Zhang

**Abstract**: Graph convolutional networks (GCNs) are currently the most promising paradigm for dealing with graph-structure data, while recent studies have also shown that GCNs are vulnerable to adversarial attacks. Thus developing GCN models that are robust to such attacks become a hot research topic. However, the structural purification learning-based or robustness constraints-based defense GCN methods are usually designed for specific data or attacks, and introduce additional objective that is not for classification. Extra training overhead is also required in their design. To address these challenges, we conduct in-depth explorations on mid-frequency signals on graphs and propose a simple yet effective Mid-pass filter GCN (Mid-GCN). Theoretical analyses guarantee the robustness of signals through the mid-pass filter, and we also shed light on the properties of different frequency signals under adversarial attacks. Extensive experiments on six benchmark graph data further verify the effectiveness of our designed Mid-GCN in node classification accuracy compared to state-of-the-art GCNs under various adversarial attack strategies.

摘要: 图卷积网络(GCNS)是目前处理图结构数据最有前途的范例，而最近的研究也表明GCNS容易受到对手攻击。因此，开发对此类攻击具有健壮性的GCN模型成为一个热门的研究课题。然而，基于结构净化学习或基于健壮性约束的防御GCN方法通常是针对特定的数据或攻击而设计的，并且引入了不用于分类的额外目标。在他们的设计中还需要额外的培训开销。为了应对这些挑战，我们对图上的中频信号进行了深入的研究，提出了一种简单而有效的中通滤波器GCN(Mid-GCN)。理论分析通过中通滤波保证了信号的稳健性，并揭示了不同频率信号在敌方攻击下的特性。在6个基准图数据上的大量实验进一步验证了我们设计的Mid-GCN在不同对抗性攻击策略下的节点分类准确率方面的有效性。



## **13. Evaluating Trade-offs in Computer Vision Between Attribute Privacy, Fairness and Utility**

计算机视觉中属性私密性、公平性和实用性的权衡 cs.CV

**SubmitDate**: 2023-02-15    [abs](http://arxiv.org/abs/2302.07917v1) [paper-pdf](http://arxiv.org/pdf/2302.07917v1)

**Authors**: William Paul, Philip Mathew, Fady Alajaji, Philippe Burlina

**Abstract**: This paper investigates to what degree and magnitude tradeoffs exist between utility, fairness and attribute privacy in computer vision. Regarding privacy, we look at this important problem specifically in the context of attribute inference attacks, a less addressed form of privacy. To create a variety of models with different preferences, we use adversarial methods to intervene on attributes relating to fairness and privacy. We see that that certain tradeoffs exist between fairness and utility, privacy and utility, and between privacy and fairness. The results also show that those tradeoffs and interactions are more complex and nonlinear between the three goals than intuition would suggest.

摘要: 本文研究了计算机视觉中效用、公平性和属性隐私之间的权衡程度和大小。关于隐私，我们特别在属性推理攻击的上下文中查看这个重要问题，属性推理攻击是一种较少解决的隐私形式。为了创建具有不同偏好的各种模型，我们使用对抗性方法干预与公平和隐私相关的属性。我们看到，公平与效用、隐私与效用、隐私与公平之间存在某种权衡。结果还表明，这三个目标之间的权衡和相互作用比直觉所暗示的更加复杂和非线性。



## **14. XploreNAS: Explore Adversarially Robust & Hardware-efficient Neural Architectures for Non-ideal Xbars**

XploreNAS：探索针对非理想Xbar的强大且硬件高效的神经体系结构 cs.LG

16 pages, 8 figures, 2 tables

**SubmitDate**: 2023-02-15    [abs](http://arxiv.org/abs/2302.07769v1) [paper-pdf](http://arxiv.org/pdf/2302.07769v1)

**Authors**: Abhiroop Bhattacharjee, Abhishek Moitra, Priyadarshini Panda

**Abstract**: Compute In-Memory platforms such as memristive crossbars are gaining focus as they facilitate acceleration of Deep Neural Networks (DNNs) with high area and compute-efficiencies. However, the intrinsic non-idealities associated with the analog nature of computing in crossbars limits the performance of the deployed DNNs. Furthermore, DNNs are shown to be vulnerable to adversarial attacks leading to severe security threats in their large-scale deployment. Thus, finding adversarially robust DNN architectures for non-ideal crossbars is critical to the safe and secure deployment of DNNs on the edge. This work proposes a two-phase algorithm-hardware co-optimization approach called XploreNAS that searches for hardware-efficient & adversarially robust neural architectures for non-ideal crossbar platforms. We use the one-shot Neural Architecture Search (NAS) approach to train a large Supernet with crossbar-awareness and sample adversarially robust Subnets therefrom, maintaining competitive hardware-efficiency. Our experiments on crossbars with benchmark datasets (SVHN, CIFAR10 & CIFAR100) show upto ~8-16% improvement in the adversarial robustness of the searched Subnets against a baseline ResNet-18 model subjected to crossbar-aware adversarial training. We benchmark our robust Subnets for Energy-Delay-Area-Products (EDAPs) using the Neurosim tool and find that with additional hardware-efficiency driven optimizations, the Subnets attain ~1.5-1.6x lower EDAPs than ResNet-18 baseline.

摘要: 内存交叉开关等计算内存平台因其高面积和高计算效率促进深度神经网络(DNN)的加速而受到关注。然而，与交叉开关中计算的模拟性质相关联的固有非理想性限制了所部署的DNN的性能。此外，DNN在大规模部署时容易受到对抗性攻击，从而造成严重的安全威胁。因此，为非理想的交叉开关找到相对健壮的DNN架构对于在边缘安全地部署DNN至关重要。该工作提出了一种称为XploreNAS的两阶段算法-硬件联合优化方法，该方法为非理想的纵横制平台寻找硬件高效且相对健壮的神经结构。我们使用一次神经体系结构搜索(NAS)方法来训练一个具有交叉开关感知的大型超网，并从中采样具有敌意健壮性的子网，从而保持具有竞争力的硬件效率。我们使用基准数据集(SVHN、CIFAR10和CIFAR100)在Crosbar上的实验表明，相对于接受Crosbar感知对抗性训练的基线ResNet-18模型，搜索到的子网的对抗健壮性提高了约8-16%。我们使用Neurosim工具对我们的健壮的能量延迟面积产品(EDAP)子网进行了基准测试，发现在额外的硬件效率驱动的优化下，这些子网获得的EDAP比ResNet-18基准低约1.5-1.6倍。



## **15. Quantum key distribution with post-processing driven by physical unclonable functions**

物理不可克隆函数驱动的后处理量子密钥分发 quant-ph

**SubmitDate**: 2023-02-15    [abs](http://arxiv.org/abs/2302.07623v1) [paper-pdf](http://arxiv.org/pdf/2302.07623v1)

**Authors**: Georgios M. Nikolopoulos, Marc Fischlin

**Abstract**: Quantum key-distribution protocols allow two honest distant parties to establish a common truly random secret key in the presence of powerful adversaries, provided that the two users share beforehand a short secret key. This pre-shared secret key is used mainly for authentication purposes in the post-processing of classical data that have been obtained during the quantum communication stage, and it prevents a man-in-the-middle attack. The necessity of a pre-shared key is usually considered as the main drawback of quantum key-distribution protocols, which becomes even stronger for large networks involving more that two users. Here we discuss the conditions under which physical unclonable function can be integrated in currently available quantum key-distribution systems, in order to facilitate the generation and the distribution of the necessary pre-shared key, with the smallest possible cost in the security of the systems. Moreover, the integration of physical unclonable functions in quantum key-distribution networks allows for real-time authentication of the devices that are connected to the network.

摘要: 量子密钥分发协议允许两个诚实的远程方在强大的对手面前建立一个共同的真正随机的密钥，前提是两个用户事先共享一个短密钥。该预共享密钥主要用于在量子通信阶段已经获得的经典数据的后处理中的认证目的，并且它防止中间人攻击。预共享密钥的必要性通常被认为是量子密钥分发协议的主要缺陷，对于涉及两个以上用户的大型网络来说，这一点变得更加严重。这里我们讨论了在现有的量子密钥分发系统中可以集成物理不可克隆函数的条件，以便以尽可能小的系统安全代价来生成和分发必要的预共享密钥。此外，量子密钥分发网络中物理不可克隆功能的集成允许对连接到网络的设备进行实时身份验证。



## **16. 3D-VFD: A Victim-free Detector against 3D Adversarial Point Clouds**

3D-VFD：一种针对3D对抗点云的无受害者检测器 cs.MM

6 pages, 13pages

**SubmitDate**: 2023-02-15    [abs](http://arxiv.org/abs/2205.08738v3) [paper-pdf](http://arxiv.org/pdf/2205.08738v3)

**Authors**: Jiahao Zhu, Huajun Zhou, Zixuan Chen, Yi Zhou, Xiaohua Xie

**Abstract**: 3D deep models consuming point clouds have achieved sound application effects in computer vision. However, recent studies have shown they are vulnerable to 3D adversarial point clouds. In this paper, we regard these malicious point clouds as 3D steganography examples and present a new perspective, 3D steganalysis, to counter such examples. Specifically, we propose 3D-VFD, a victim-free detector against 3D adversarial point clouds. Its core idea is to capture the discrepancies between residual geometric feature distributions of benign point clouds and adversarial point clouds and map these point clouds to a lower dimensional space where we can efficiently distinguish them. Unlike existing detection techniques against 3D adversarial point clouds, 3D-VFD does not rely on the victim 3D deep model's outputs for discrimination. Extensive experiments demonstrate that 3D-VFD achieves state-of-the-art detection and can effectively detect 3D adversarial attacks based on point adding and point perturbation while keeping fast detection speed.

摘要: 消耗点云的三维深部模型在计算机视觉中取得了良好的应用效果。然而，最近的研究表明，它们很容易受到3D对抗性点云的攻击。在本文中，我们将这些恶意的点云视为3D隐写的例子，并提出了一种新的视角-3D隐写分析来对抗这种例子。具体地说，我们提出了3D-VFD，这是一种针对3D对抗点云的无受害者检测器。它的核心思想是捕捉良性点云和对抗性点云的残留几何特征分布之间的差异，并将这些点云映射到一个更低维的空间，在那里我们可以有效地区分它们。与现有的针对3D对抗点云的检测技术不同，3D-VFD不依赖受害者3D深度模型的输出来进行区分。大量实验表明，3D-VFD实现了最先进的检测，在保持较快检测速度的情况下，能够有效地检测基于点添加和点扰动的3D对抗攻击。



## **17. Attacking Fake News Detectors via Manipulating News Social Engagement**

通过操纵新闻社会参与打击假新闻检测器 cs.SI

In Proceedings of the ACM Web Conference 2023 (WWW'23)

**SubmitDate**: 2023-02-14    [abs](http://arxiv.org/abs/2302.07363v1) [paper-pdf](http://arxiv.org/pdf/2302.07363v1)

**Authors**: Haoran Wang, Yingtong Dou, Canyu Chen, Lichao Sun, Philip S. Yu, Kai Shu

**Abstract**: Social media is one of the main sources for news consumption, especially among the younger generation. With the increasing popularity of news consumption on various social media platforms, there has been a surge of misinformation which includes false information or unfounded claims. As various text- and social context-based fake news detectors are proposed to detect misinformation on social media, recent works start to focus on the vulnerabilities of fake news detectors. In this paper, we present the first adversarial attack framework against Graph Neural Network (GNN)-based fake news detectors to probe their robustness. Specifically, we leverage a multi-agent reinforcement learning (MARL) framework to simulate the adversarial behavior of fraudsters on social media. Research has shown that in real-world settings, fraudsters coordinate with each other to share different news in order to evade the detection of fake news detectors. Therefore, we modeled our MARL framework as a Markov Game with bot, cyborg, and crowd worker agents, which have their own distinctive cost, budget, and influence. We then use deep Q-learning to search for the optimal policy that maximizes the rewards. Extensive experimental results on two real-world fake news propagation datasets demonstrate that our proposed framework can effectively sabotage the GNN-based fake news detector performance. We hope this paper can provide insights for future research on fake news detection.

摘要: 社交媒体是新闻消费的主要来源之一，尤其是在年轻一代中。随着新闻消费在各种社交媒体平台上的日益流行，错误信息激增，其中包括虚假信息或毫无根据的说法。随着各种基于文本和社会语境的假新闻检测器被提出来检测社交媒体上的错误信息，最近的研究开始关注假新闻检测器的脆弱性。本文提出了第一个针对基于图神经网络(GNN)的假新闻检测器的对抗性攻击框架，以探讨其健壮性。具体地说，我们利用多智能体强化学习(MAIL)框架来模拟社交媒体上欺诈者的对抗行为。研究表明，在现实世界中，欺诈者相互协调，分享不同的新闻，以躲避假新闻检测器的检测。因此，我们将我们的Marl框架建模为一个包含BOT、半机械人和群工代理的马尔可夫博弈，这些代理都有自己独特的成本、预算和影响。然后，我们使用深度Q-学习来搜索最大化回报的最优策略。在两个真实假新闻传播数据集上的大量实验结果表明，我们提出的框架可以有效地破坏基于GNN的假新闻检测器的性能。希望本文能为今后的假新闻检测研究提供一些启示。



## **18. Cooperative Perception for Safe Control of Autonomous Vehicles under LiDAR Spoofing Attacks**

激光雷达欺骗攻击下自主车辆安全控制的协作感知 eess.SY

**SubmitDate**: 2023-02-14    [abs](http://arxiv.org/abs/2302.07341v1) [paper-pdf](http://arxiv.org/pdf/2302.07341v1)

**Authors**: Hongchao Zhang, Zhouchi Li, Shiyu Cheng, Andrew Clark

**Abstract**: Autonomous vehicles rely on LiDAR sensors to detect obstacles such as pedestrians, other vehicles, and fixed infrastructures. LiDAR spoofing attacks have been demonstrated that either create erroneous obstacles or prevent detection of real obstacles, resulting in unsafe driving behaviors. In this paper, we propose an approach to detect and mitigate LiDAR spoofing attacks by leveraging LiDAR scan data from other neighboring vehicles. This approach exploits the fact that spoofing attacks can typically only be mounted on one vehicle at a time, and introduce additional points into the victim's scan that can be readily detected by comparison from other, non-modified scans. We develop a Fault Detection, Identification, and Isolation procedure that identifies non-existing obstacle, physical removal, and adversarial object attacks, while also estimating the actual locations of obstacles. We propose a control algorithm that guarantees that these estimated object locations are avoided. We validate our framework using a CARLA simulation study, in which we verify that our FDII algorithm correctly detects each attack pattern.

摘要: 自动驾驶车辆依靠激光雷达传感器来检测障碍物，如行人、其他车辆和固定基础设施。激光雷达欺骗攻击已被证明要么制造错误的障碍物，要么阻止检测到真实的障碍物，从而导致不安全的驾驶行为。本文提出了一种利用邻近车辆的激光雷达扫描数据来检测和缓解激光雷达欺骗攻击的方法。这种方法利用了欺骗攻击通常一次只能安装在一辆车上的事实，并在受害者的扫描中引入了额外的点，通过与其他未经修改的扫描进行比较，可以很容易地检测到这些点。我们开发了故障检测、识别和隔离程序，该程序识别不存在的障碍、物理移除和对抗性对象攻击，同时还估计障碍的实际位置。我们提出了一种控制算法，保证了避免这些估计的目标位置。我们通过CALA仿真研究验证了我们的框架，其中我们验证了我们的FDII算法正确地检测到了每种攻击模式。



## **19. Randomization for adversarial robustness: the Good, the Bad and the Ugly**

对抗健壮性的随机化：好的、坏的和丑的 cs.LG

8 pages + bibliography and appendix, 3 figures. Submitted to ICML  2023

**SubmitDate**: 2023-02-14    [abs](http://arxiv.org/abs/2302.07221v1) [paper-pdf](http://arxiv.org/pdf/2302.07221v1)

**Authors**: Lucas Gnecco-Heredia, Yann Chevaleyre, Benjamin Negrevergne, Laurent Meunier

**Abstract**: Deep neural networks are known to be vulnerable to adversarial attacks: A small perturbation that is imperceptible to a human can easily make a well-trained deep neural network misclassify. To defend against adversarial attacks, randomized classifiers have been proposed as a robust alternative to deterministic ones. In this work we show that in the binary classification setting, for any randomized classifier, there is always a deterministic classifier with better adversarial risk. In other words, randomization is not necessary for robustness. In many common randomization schemes, the deterministic classifiers with better risk are explicitly described: For example, we show that ensembles of classifiers are more robust than mixtures of classifiers, and randomized smoothing is more robust than input noise injection. Finally, experiments confirm our theoretical results with the two families of randomized classifiers we analyze.

摘要: 众所周知，深度神经网络容易受到敌意攻击：人类无法察觉的微小扰动很容易使训练有素的深度神经网络错误分类。为了抵抗敌意攻击，随机分类器被提出作为确定性分类器的一种稳健的替代方案。在这项工作中，我们证明了在二分类环境下，对于任何随机化的分类器，总是有一个确定性的分类器具有更好的对抗风险。换句话说，随机化不是稳健性所必需的。在许多常见的随机化方案中，明确描述了风险更好的确定性分类器：例如，我们证明了分类器的集成比分类器的混合更健壮，随机平滑比输入噪声注入更健壮。最后，通过实验验证了我们所分析的两类随机分类器的理论结果。



## **20. Bridge the Gap Between CV and NLP! An Optimization-based Textual Adversarial Attack Framework**

弥合简历和NLP之间的差距！一种基于优化的文本对抗攻击框架 cs.CL

Codes are available at: https://github.com/Phantivia/T-PGD

**SubmitDate**: 2023-02-14    [abs](http://arxiv.org/abs/2110.15317v3) [paper-pdf](http://arxiv.org/pdf/2110.15317v3)

**Authors**: Lifan Yuan, Yichi Zhang, Yangyi Chen, Wei Wei

**Abstract**: Despite recent success on various tasks, deep learning techniques still perform poorly on adversarial examples with small perturbations. While optimization-based methods for adversarial attacks are well-explored in the field of computer vision, it is impractical to directly apply them in natural language processing due to the discrete nature of the text. To address the problem, we propose a unified framework to extend the existing optimization-based adversarial attack methods in the vision domain to craft textual adversarial samples. In this framework, continuously optimized perturbations are added to the embedding layer and amplified in the forward propagation process. Then the final perturbed latent representations are decoded with a masked language model head to obtain potential adversarial samples. In this paper, we instantiate our framework with an attack algorithm named Textual Projected Gradient Descent (T-PGD). We find our algorithm effective even using proxy gradient information. Therefore, we perform the more challenging transfer black-box attack and conduct comprehensive experiments to evaluate our attack algorithm with several models on three benchmark datasets. Experimental results demonstrate that our method achieves an overall better performance and produces more fluent and grammatical adversarial samples compared to strong baseline methods. All the code and data will be made public.

摘要: 尽管最近在各种任务上取得了成功，但深度学习技术在具有小扰动的对抗性例子中仍然表现不佳。虽然基于优化的对抗性攻击方法在计算机视觉领域得到了很好的探索，但由于文本的离散性质，将其直接应用于自然语言处理是不切实际的。为了解决这个问题，我们提出了一个统一的框架来扩展现有的视觉领域基于优化的对抗性攻击方法，以制作文本对抗性样本。在该框架中，不断优化的扰动被添加到嵌入层，并在前向传播过程中被放大。然后，使用掩蔽语言模型头部对最终扰动的潜在表示进行解码，以获得潜在的对抗性样本。在本文中，我们使用一种名为文本投影梯度下降(T-PGD)的攻击算法来实例化我们的框架。我们发现我们的算法即使使用代理梯度信息也是有效的。因此，我们执行了更具挑战性的转移黑盒攻击，并在三个基准数据集上用几个模型进行了全面的实验来评估我们的攻击算法。实验结果表明，与强基线方法相比，我们的方法取得了更好的整体性能，生成了更流畅、更具语法意义的对抗性样本。所有代码和数据都将公之于众。



## **21. A Comprehensive Study of Real-Time Object Detection Networks Across Multiple Domains: A Survey**

跨域实时目标检测网络研究综述 cs.CV

Published in Transactions on Machine Learning Research (TMLR) with  Survey Certification

**SubmitDate**: 2023-02-14    [abs](http://arxiv.org/abs/2208.10895v2) [paper-pdf](http://arxiv.org/pdf/2208.10895v2)

**Authors**: Elahe Arani, Shruthi Gowda, Ratnajit Mukherjee, Omar Magdy, Senthilkumar Kathiresan, Bahram Zonooz

**Abstract**: Deep neural network based object detectors are continuously evolving and are used in a multitude of applications, each having its own set of requirements. While safety-critical applications need high accuracy and reliability, low-latency tasks need resource and energy-efficient networks. Real-time detectors, which are a necessity in high-impact real-world applications, are continuously proposed, but they overemphasize the improvements in accuracy and speed while other capabilities such as versatility, robustness, resource and energy efficiency are omitted. A reference benchmark for existing networks does not exist, nor does a standard evaluation guideline for designing new networks, which results in ambiguous and inconsistent comparisons. We, thus, conduct a comprehensive study on multiple real-time detectors (anchor-, keypoint-, and transformer-based) on a wide range of datasets and report results on an extensive set of metrics. We also study the impact of variables such as image size, anchor dimensions, confidence thresholds, and architecture layers on the overall performance. We analyze the robustness of detection networks against distribution shifts, natural corruptions, and adversarial attacks. Also, we provide a calibration analysis to gauge the reliability of the predictions. Finally, to highlight the real-world impact, we conduct two unique case studies, on autonomous driving and healthcare applications. To further gauge the capability of networks in critical real-time applications, we report the performance after deploying the detection networks on edge devices. Our extensive empirical study can act as a guideline for the industrial community to make an informed choice on the existing networks. We also hope to inspire the research community towards a new direction in the design and evaluation of networks that focuses on a bigger and holistic overview for a far-reaching impact.

摘要: 基于深度神经网络的目标检测器正在不断发展，并在许多应用中使用，每个应用都有其自己的一组要求。安全关键型应用程序需要高准确性和可靠性，而低延迟任务则需要资源和能效高的网络。实时检测器在高影响的现实世界应用中是必不可少的，不断被提出，但它们过分强调精度和速度的提高，而忽略了其他功能，如通用性、健壮性、资源和能源效率。现有网络没有参考基准，也没有设计新网络的标准评估指南，这导致比较不明确和不一致。因此，我们在广泛的数据集上对多个实时检测器(锚点、关键点和变压器)进行了全面的研究，并报告了一组广泛的指标的结果。我们还研究了图像大小、锚点维度、置信度阈值和架构层等变量对整体性能的影响。我们分析了检测网络对分布偏移、自然破坏和敌意攻击的稳健性。此外，我们还提供了校准分析，以衡量预测的可靠性。最后，为了突出现实世界的影响，我们进行了两个独特的案例研究，分别是自动驾驶和医疗保健应用。为了进一步衡量网络在关键实时应用中的能力，我们报告了在边缘设备上部署检测网络后的性能。我们广泛的实证研究可以为工业界在现有网络上做出明智的选择提供指导。我们还希望激励研究界在网络的设计和评估方面朝着一个新的方向前进，专注于更大和更全面的概览，以产生深远的影响。



## **22. Practical Cross-system Shilling Attacks with Limited Access to Data**

数据访问受限的实用跨系统先令攻击 cs.IR

Accepted by AAAI 2023

**SubmitDate**: 2023-02-14    [abs](http://arxiv.org/abs/2302.07145v1) [paper-pdf](http://arxiv.org/pdf/2302.07145v1)

**Authors**: Meifang Zeng, Ke Li, Bingchuan Jiang, Liujuan Cao, Hui Li

**Abstract**: In shilling attacks, an adversarial party injects a few fake user profiles into a Recommender System (RS) so that the target item can be promoted or demoted. Although much effort has been devoted to developing shilling attack methods, we find that existing approaches are still far from practical. In this paper, we analyze the properties a practical shilling attack method should have and propose a new concept of Cross-system Attack. With the idea of Cross-system Attack, we design a Practical Cross-system Shilling Attack (PC-Attack) framework that requires little information about the victim RS model and the target RS data for conducting attacks. PC-Attack is trained to capture graph topology knowledge from public RS data in a self-supervised manner. Then, it is fine-tuned on a small portion of target data that is easy to access to construct fake profiles. Extensive experiments have demonstrated the superiority of PC-Attack over state-of-the-art baselines. Our implementation of PC-Attack is available at https://github.com/KDEGroup/PC-Attack.

摘要: 在先令攻击中，敌对方向推荐系统(RS)注入一些虚假的用户配置文件，以便目标项目可以升级或降级。虽然已经投入了大量的精力来开发先令攻击方法，但我们发现现有的方法仍然远远不实用。本文分析了一种实用的先令攻击方法应具备的性质，提出了跨系统攻击的新概念。利用跨系统攻击的思想，我们设计了一个实用的跨系统先令攻击(PC-Attack)框架，该框架只需要很少的受害者RS模型和目标RS数据的信息就可以进行攻击。PC-Attack被训练成以自监督的方式从公共遥感数据中捕获图拓扑知识。然后，它对一小部分目标数据进行微调，这些数据很容易访问以构建虚假配置文件。广泛的实验已经证明了PC攻击相对于最先进的基线的优越性。我们对PC-Attack的实施可在https://github.com/KDEGroup/PC-Attack.获得



## **23. Adversarial Path Planning for Optimal Camera Positioning**

摄像机最优定位的对抗性路径规划 cs.CG

**SubmitDate**: 2023-02-14    [abs](http://arxiv.org/abs/2302.07051v1) [paper-pdf](http://arxiv.org/pdf/2302.07051v1)

**Authors**: Gaia Carenini, Alexandre Duplessis

**Abstract**: The use of visual sensors is flourishing, driven among others by the several applications in detection and prevention of crimes or dangerous events. While the problem of optimal camera placement for total coverage has been solved for a decade or so, that of the arrangement of cameras maximizing the recognition of objects "in-transit" is still open. The objective of this paper is to attack this problem by providing an adversarial method of proven optimality based on the resolution of Hamilton-Jacobi equations. The problem is attacked by first assuming the perspective of an adversary, i.e. computing explicitly the path minimizing the probability of detection and the quality of reconstruction. Building on this result, we introduce an optimality measure for camera configurations and perform a simulated annealing algorithm to find the optimal camera placement.

摘要: 视觉传感器的使用正在蓬勃发展，其中包括在侦测和预防犯罪或危险事件方面的几种应用。虽然总覆盖范围的最佳摄像机布置问题已经解决了十年左右，但如何布置摄像机以最大限度地识别“过境”物体的问题仍然悬而未决。本文的目的是通过提供一种基于Hamilton-Jacobi方程的解的证明最优性的对抗性方法来解决这一问题。该问题首先从敌方的角度出发，即显式计算最小化检测概率和重构质量的路径。在此基础上，我们引入了摄像机配置的最优性度量，并用模拟退火法来寻找摄像机的最优布置。



## **24. Does CLIP Know My Face?**

小夹子认得我的脸吗？ cs.LG

15 pages, 6 figures

**SubmitDate**: 2023-02-14    [abs](http://arxiv.org/abs/2209.07341v2) [paper-pdf](http://arxiv.org/pdf/2209.07341v2)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Manuel Brack, Felix Friedrich, Patrick Schramowski, Kristian Kersting

**Abstract**: With the rise of deep learning in various applications, privacy concerns around the protection of training data has become a critical area of research. Whereas prior studies have focused on privacy risks in single-modal models, we introduce a novel method to assess privacy for multi-modal models, specifically vision-language models like CLIP. The proposed Identity Inference Attack (IDIA) reveals whether an individual was included in the training data by querying the model with images of the same person. Letting the model choose from a wide variety of possible text labels, the model reveals whether it recognizes the person and, therefore, was used for training. Our large-scale experiments on CLIP demonstrate that individuals used for training can be identified with very high accuracy. We confirm that the model has learned to associate names with depicted individuals, implying the existence of sensitive information that can be extracted by adversaries. Our results highlight the need for stronger privacy protection in large-scale models and suggest that IDIAs can be used to prove the unauthorized use of data for training and to enforce privacy laws.

摘要: 随着深度学习在各种应用中的兴起，围绕训练数据保护的隐私问题已经成为一个关键的研究领域。鉴于以往的研究主要集中于单通道模型中的隐私风险，我们引入了一种新的方法来评估多通道模型的隐私，特别是像CLIP这样的视觉语言模型。提出的身份推断攻击(IDIA)通过用同一人的图像查询模型来揭示该人是否包括在训练数据中。让模型从各种各样的可能的文本标签中进行选择，该模型显示它是否识别出这个人，因此，它被用于训练。我们在CLIP上的大规模实验表明，用于训练的个体可以非常准确地识别。我们确认，该模型已经学会了将姓名与所描述的个人相关联，这意味着存在可被对手提取的敏感信息。我们的结果强调了在大规模模型中加强隐私保护的必要性，并建议可以使用IDIA来证明未经授权使用数据进行培训和执行隐私法。



## **25. Fact-Saboteurs: A Taxonomy of Evidence Manipulation Attacks against Fact-Verification Systems**

事实破坏者：针对事实核查系统的证据操纵攻击的分类 cs.CR

**SubmitDate**: 2023-02-14    [abs](http://arxiv.org/abs/2209.03755v3) [paper-pdf](http://arxiv.org/pdf/2209.03755v3)

**Authors**: Sahar Abdelnabi, Mario Fritz

**Abstract**: Mis- and disinformation are a substantial global threat to our security and safety. To cope with the scale of online misinformation, researchers have been working on automating fact-checking by retrieving and verifying against relevant evidence. However, despite many advances, a comprehensive evaluation of the possible attack vectors against such systems is still lacking. Particularly, the automated fact-verification process might be vulnerable to the exact disinformation campaigns it is trying to combat. In this work, we assume an adversary that automatically tampers with the online evidence in order to disrupt the fact-checking model via camouflaging the relevant evidence or planting a misleading one. We first propose an exploratory taxonomy that spans these two targets and the different threat model dimensions. Guided by this, we design and propose several potential attack methods. We show that it is possible to subtly modify claim-salient snippets in the evidence and generate diverse and claim-aligned evidence. Thus, we highly degrade the fact-checking performance under many different permutations of the taxonomy's dimensions. The attacks are also robust against post-hoc modifications of the claim. Our analysis further hints at potential limitations in models' inference when faced with contradicting evidence. We emphasize that these attacks can have harmful implications on the inspectable and human-in-the-loop usage scenarios of such models, and conclude by discussing challenges and directions for future defenses.

摘要: 错误和虚假信息是对我们的安全和安全的重大全球威胁。为了应对网上虚假信息的规模，研究人员一直在致力于通过检索和验证相关证据来实现事实核查的自动化。然而，尽管取得了许多进展，但仍然缺乏对针对此类系统的可能攻击媒介的全面评估。特别是，自动化的事实核查过程可能容易受到它试图打击的虚假信息运动的影响。在这项工作中，我们假设一个对手自动篡改在线证据，以便通过伪装相关证据或植入误导性证据来扰乱事实核查模型。我们首先提出了一种探索性分类，该分类跨越这两个目标和不同的威胁模型维度。在此指导下，我们设计并提出了几种潜在的攻击方法。我们表明，可以巧妙地修改证据中突出声明的片段，并生成多样化的与声明一致的证据。因此，在分类维度的许多不同排列下，我们极大地降低了事实检查的性能。这些攻击也对索赔的事后修改具有很强的抵御能力。我们的分析进一步暗示，在面对相互矛盾的证据时，模型的推理可能存在局限性。我们强调，这些攻击可能会对此类模型的可检查和人在环中使用场景产生有害影响，最后讨论未来防御的挑战和方向。



## **26. Oops..! I Glitched It Again! How to Multi-Glitch the Glitching-Protections on ARM TrustZone-M**

哎呀..！我又出故障了！如何在ARM TrustZone-M上实现多毛刺保护 cs.CR

**SubmitDate**: 2023-02-14    [abs](http://arxiv.org/abs/2302.06932v1) [paper-pdf](http://arxiv.org/pdf/2302.06932v1)

**Authors**: Marvin Saß, Richard Mitev, Ahmad-Reza Sadeghi

**Abstract**: Voltage Fault Injection (VFI), also known as power glitching, has proven to be a severe threat to real-world systems. In VFI attacks, the adversary disturbs the power-supply of the target-device forcing the device to illegitimate behavior. Various countermeasures have been proposed to address different types of fault injection attacks at different abstraction layers, either requiring to modify the underlying hardware or software/firmware at the machine instruction level. Moreover, only recently, individual chip manufacturers have started to respond to this threat by integrating countermeasures in their products. Generally, these countermeasures aim at protecting against single fault injection (SFI) attacks, since Multiple Fault Injection (MFI) is believed to be challenging and sometimes even impractical. In this paper, we present {\mu}-Glitch, the first Voltage Fault Injection (VFI) platform which is capable of injecting multiple, coordinated voltage faults into a target device, requiring only a single trigger signal. We provide a novel flow for Multiple Voltage Fault Injection (MVFI) attacks to significantly reduce the search complexity for fault parameters, as the search space increases exponentially with each additional fault injection. We evaluate and showcase the effectiveness and practicality of our attack platform on four real-world chips, featuring TrustZone-M: The first two have interdependent backchecking mechanisms, while the second two have additionally integrated countermeasures against fault injection. Our evaluation revealed that {\mu}-Glitch can successfully inject four consecutive faults within an average time of one day. Finally, we discuss potential countermeasures to mitigate VFI attacks and additionally propose two novel attack scenarios for MVFI.

摘要: 电压故障注入(VFI)，也称为电源毛刺，已被证明是对实际系统的严重威胁。在VFI攻击中，对手干扰目标设备的电源，迫使设备进行非法行为。已经提出了各种对策来解决在不同抽象层的不同类型的故障注入攻击，或者需要在机器指令级修改底层硬件或软件/固件。此外，直到最近，个别芯片制造商才开始通过将对策整合到他们的产品中来应对这种威胁。通常，这些对策旨在防御单故障注入(SFI)攻击，因为多故障注入(MFI)被认为具有挑战性，有时甚至不切实际。在本文中，我们介绍了第一个电压故障注入(VFI)平台，它能够将多个协调的电压故障注入到目标设备中，只需要一个触发信号。我们提出了一种新的多电压故障注入(MVFI)攻击流程，以显著降低故障参数的搜索复杂度，因为每增加一次故障注入，搜索空间就会成倍增加。我们在四个真实芯片上评估和展示了我们的攻击平台的有效性和实用性：前两个芯片具有相互依赖的回溯检查机制，而后两个芯片具有针对故障注入的额外集成对策。我们的评估显示，{\MU}-GLITCH可以在平均一天的时间内成功注入四个连续故障。最后，我们讨论了缓解VFI攻击的潜在对策，并提出了两种新的MVFI攻击方案。



## **27. Towards Robust Neural Image Compression: Adversarial Attack and Model Finetuning**

面向稳健的神经图像压缩：对抗性攻击和模型精调 cs.CV

This paper has been completely rewritten

**SubmitDate**: 2023-02-14    [abs](http://arxiv.org/abs/2112.08691v2) [paper-pdf](http://arxiv.org/pdf/2112.08691v2)

**Authors**: Tong Chen, Zhan Ma

**Abstract**: Deep neural network based image compression has been extensively studied. Model robustness is largely overlooked, though it is crucial to service enabling. We perform the adversarial attack by injecting a small amount of noise perturbation to original source images, and then encode these adversarial examples using prevailing learnt image compression models. Experiments report severe distortion in the reconstruction of adversarial examples, revealing the general vulnerability of existing methods, regardless of the settings used in underlying compression model (e.g., network architecture, loss function, quality scale) and optimization strategy used for injecting perturbation (e.g., noise threshold, signal distance measurement). Later, we apply the iterative adversarial finetuning to refine pretrained models. In each iteration, random source images and adversarial examples are mixed to update underlying model. Results show the effectiveness of the proposed finetuning strategy by substantially improving the compression model robustness. Overall, our methodology is simple, effective, and generalizable, making it attractive for developing robust learnt image compression solution. All materials have been made publicly accessible at https://njuvision.github.io/RobustNIC for reproducible research.

摘要: 基于深度神经网络的图像压缩已经得到了广泛的研究。模型健壮性在很大程度上被忽视了，尽管它对服务启用至关重要。我们通过在原始源图像中注入少量的噪声扰动来执行对抗性攻击，然后使用主流的学习图像压缩模型对这些对抗性示例进行编码。实验报告了在对抗性示例的重建中的严重失真，揭示了现有方法的一般脆弱性，而与底层压缩模型(例如，网络架构、损失函数、质量尺度)和用于注入扰动的优化策略(例如，噪声阈值、信号距离测量)中使用的设置无关。随后，我们应用迭代对抗性精调来精炼预先训练的模型。在每一次迭代中，随机源图像和对抗性样本被混合以更新底层模型。结果表明，所提出的微调策略显著提高了压缩模型的稳健性。总体而言，我们的方法是简单、有效和可推广的，对于开发健壮的学习图像压缩解决方案具有吸引力。所有材料都已在https://njuvision.github.io/RobustNIC上公开访问，以进行可重复的研究。



## **28. Towards Effective and Robust Neural Trojan Defenses via Input Filtering**

通过输入过滤实现高效、健壮的神经木马防御 cs.CR

Accepted to ECCV 2022

**SubmitDate**: 2023-02-14    [abs](http://arxiv.org/abs/2202.12154v5) [paper-pdf](http://arxiv.org/pdf/2202.12154v5)

**Authors**: Kien Do, Haripriya Harikumar, Hung Le, Dung Nguyen, Truyen Tran, Santu Rana, Dang Nguyen, Willy Susilo, Svetha Venkatesh

**Abstract**: Trojan attacks on deep neural networks are both dangerous and surreptitious. Over the past few years, Trojan attacks have advanced from using only a single input-agnostic trigger and targeting only one class to using multiple, input-specific triggers and targeting multiple classes. However, Trojan defenses have not caught up with this development. Most defense methods still make inadequate assumptions about Trojan triggers and target classes, thus, can be easily circumvented by modern Trojan attacks. To deal with this problem, we propose two novel "filtering" defenses called Variational Input Filtering (VIF) and Adversarial Input Filtering (AIF) which leverage lossy data compression and adversarial learning respectively to effectively purify potential Trojan triggers in the input at run time without making assumptions about the number of triggers/target classes or the input dependence property of triggers. In addition, we introduce a new defense mechanism called "Filtering-then-Contrasting" (FtC) which helps avoid the drop in classification accuracy on clean data caused by "filtering", and combine it with VIF/AIF to derive new defenses of this kind. Extensive experimental results and ablation studies show that our proposed defenses significantly outperform well-known baseline defenses in mitigating five advanced Trojan attacks including two recent state-of-the-art while being quite robust to small amounts of training data and large-norm triggers.

摘要: 特洛伊木马对深层神经网络的攻击既危险又隐蔽。在过去的几年中，特洛伊木马攻击已经从只使用一个与输入无关的触发器和只针对一个类发展到使用多个特定于输入的触发器和目标多个类。然而，特洛伊木马防御并没有跟上这一发展。大多数防御方法仍然对木马的触发器和目标类别做了不充分的假设，因此很容易被现代木马攻击所规避。针对这一问题，我们提出了两种新的“过滤”防御机制，称为变输入过滤(VIF)和对抗输入过滤(AIF)，它们分别利用有损数据压缩和对抗学习在运行时有效地净化输入中潜在的特洛伊木马触发器，而不需要假设触发器/目标类的数量或触发器的输入依赖属性。此外，我们还引入了一种新的防御机制，称为“过滤-然后-对比”(FTC)，它有助于避免“过滤”导致对干净数据的分类精度的下降，并将其与VIF/AIF相结合来派生出这种新的防御机制。广泛的实验结果和烧蚀研究表明，我们提出的防御方案在缓解五种高级特洛伊木马攻击(包括两种最新的木马攻击)方面明显优于众所周知的基线防御方案，同时对少量训练数据和大范数触发事件具有相当的健壮性。



## **29. A survey in Adversarial Defences and Robustness in NLP**

自然语言处理中的对抗性防御和稳健性研究综述 cs.CL

**SubmitDate**: 2023-02-13    [abs](http://arxiv.org/abs/2203.06414v3) [paper-pdf](http://arxiv.org/pdf/2203.06414v3)

**Authors**: Shreya Goyal, Sumanth Doddapaneni, Mitesh M. Khapra, Balaraman Ravindran

**Abstract**: In recent years, it has been seen that deep neural networks are lacking robustness and are vulnerable in case of adversarial perturbations in input data. Strong adversarial attacks are proposed by various authors for tasks under computer vision and Natural Language Processing (NLP). As a counter-effort, several defense mechanisms are also proposed to save these networks from failing. Defending the neural networks from adversarial attacks has its own importance, where the goal is to ensure that the model's prediction doesn't change if input data is perturbed. Numerous methods for adversarial defense in NLP are proposed of late, for different NLP tasks such as text classification, named entity recognition, natural language inferencing, etc. Some of these methods are not just used for defending neural networks from adversarial attacks, but also used as a regularization mechanism during training, saving the model from overfitting. The proposed survey is an attempt to review different methods proposed for adversarial defenses in NLP in recent years by proposing a novel taxonomy. This survey also highlights the fragility of the advanced deep neural networks in NLP and the challenges in defending them.

摘要: 近年来，人们已经看到，深度神经网络缺乏健壮性，在输入数据受到对抗性扰动的情况下很容易受到攻击。强对抗性攻击是针对计算机视觉和自然语言处理(NLP)下的任务而提出的。作为应对措施，还提出了几种防御机制，以避免这些网络出现故障。保护神经网络免受敌意攻击有其自身的重要性，其目标是确保在输入数据受到干扰时，模型的预测不会改变。近年来，针对不同的自然语言处理任务，如文本分类、命名实体识别、自然语言推理等，人们提出了许多针对自然语言处理的对抗性防御方法，其中一些方法不仅用于保护神经网络免受敌意攻击，还在训练过程中作为一种正则化机制，避免了模型的过度拟合。本研究试图通过提出一种新的分类方法，对近年来在NLP中提出的不同对抗防御方法进行综述。这项调查还突显了NLP中先进的深度神经网络的脆弱性以及在保护它们方面的挑战。



## **30. Sneaky Spikes: Uncovering Stealthy Backdoor Attacks in Spiking Neural Networks with Neuromorphic Data**

偷偷摸摸的尖峰：用神经形态数据揭示尖峰神经网络中的秘密后门攻击 cs.CR

**SubmitDate**: 2023-02-13    [abs](http://arxiv.org/abs/2302.06279v1) [paper-pdf](http://arxiv.org/pdf/2302.06279v1)

**Authors**: Gorka Abad, Oguzhan Ersoy, Stjepan Picek, Aitor Urbieta

**Abstract**: Deep neural networks (DNNs) have achieved excellent results in various tasks, including image and speech recognition. However, optimizing the performance of DNNs requires careful tuning of multiple hyperparameters and network parameters via training. High-performance DNNs utilize a large number of parameters, corresponding to high energy consumption during training. To address these limitations, researchers have developed spiking neural networks (SNNs), which are more energy-efficient and can process data in a biologically plausible manner, making them well-suited for tasks involving sensory data processing, i.e., neuromorphic data. Like DNNs, SNNs are vulnerable to various threats, such as adversarial examples and backdoor attacks. Yet, the attacks and countermeasures for SNNs have been almost fully unexplored.   This paper investigates the application of backdoor attacks in SNNs using neuromorphic datasets and different triggers. More precisely, backdoor triggers in neuromorphic data can change their position and color, allowing a larger range of possibilities than common triggers in, e.g., the image domain. We propose different attacks achieving up to 100\% attack success rate without noticeable clean accuracy degradation. We also evaluate the stealthiness of the attacks via the structural similarity metric, showing our most powerful attacks being also stealthy. Finally, we adapt the state-of-the-art defenses from the image domain, demonstrating they are not necessarily effective for neuromorphic data resulting in inaccurate performance.

摘要: 深度神经网络(DNN)在包括图像和语音识别在内的各种任务中取得了优异的效果。然而，优化DNN的性能需要通过训练仔细调整多个超参数和网络参数。高性能的DNN使用大量的参数，对应于训练过程中的高能量消耗。为了解决这些局限性，研究人员开发了尖峰神经网络(SNN)，它更节能，可以以生物上可信的方式处理数据，使其非常适合于涉及感觉数据处理的任务，即神经形态数据。与DNN一样，SNN也容易受到各种威胁，例如敌意示例和后门攻击。然而，针对SNN的攻击和对策几乎完全没有被探索过。本文利用神经形态数据集和不同的触发器，研究了后门攻击在SNN中的应用。更准确地说，神经形态数据中的后门触发器可以改变它们的位置和颜色，允许比图像领域中的常见触发器有更大范围的可能性。我们提出了不同的攻击方法，攻击成功率可达100%，且没有明显的干净准确率下降。我们还通过结构相似性度量来评估攻击的隐蔽性，表明我们最强大的攻击也是隐蔽性的。最后，我们从图像领域改编了最先进的防御措施，证明了它们对导致不准确性能的神经形态数据不一定有效。



## **31. PRAGTHOS:Practical Game Theoretically Secure Proof-of-Work Blockchain**

PRAGTHOS：实用游戏理论上安全的工作证明区块链 cs.CR

**SubmitDate**: 2023-02-13    [abs](http://arxiv.org/abs/2302.06136v1) [paper-pdf](http://arxiv.org/pdf/2302.06136v1)

**Authors**: Varul Srivastava, Dr. Sujit Gujar

**Abstract**: Security analysis of blockchain technology is an active domain of research. There has been both cryptographic and game-theoretic security analysis of Proof-of-Work (PoW) blockchains. Prominent work includes the cryptographic security analysis under the Universal Composable framework and Game-theoretic security analysis using Rational Protocol Design. These security analysis models rely on stricter assumptions that might not hold. In this paper, we analyze the security of PoW blockchain protocols. We first show how assumptions made by previous models need not be valid in reality, which attackers can exploit to launch attacks that these models fail to capture. These include Difficulty Alternating Attack, under which forking is possible for an adversary with less than 0.5 mining power, Quick-Fork Attack, a general bound on selfish mining attack and transaction withholding attack. Following this, we argue why previous models for security analysis fail to capture these attacks and propose a more practical framework for security analysis pRPD. We then propose a framework to build PoW blockchains PRAGTHOS, which is secure from the attacks mentioned above. Finally, we argue that PoW blockchains complying with the PRAGTHOS framework are secure against a computationally bounded adversary under certain conditions on the reward scheme.

摘要: 区块链技术的安全分析是一个活跃的研究领域。工作证明(PoW)区块链的安全性分析既有密码学的，也有博弈论的。突出的工作包括通用可组合框架下的密码安全分析和使用理性协议设计的博弈论安全分析。这些安全分析模型依赖于可能不成立的更严格的假设。本文对POW区块链协议的安全性进行了分析。我们首先展示以前模型所做的假设如何在现实中不成立，攻击者可以利用这些假设来发动这些模型未能捕获的攻击。这些攻击包括困难交替攻击，在这种攻击下，挖掘威力小于0.5的对手可以进行分叉攻击，Quick-Fork攻击，一般自私挖掘攻击和交易扣留攻击。接着，我们讨论了为什么以前的安全分析模型不能捕捉到这些攻击，并提出了一个更实用的安全分析框架PRPD。然后，我们提出了一个构建POW区块链PRAGTHOS的框架，该框架是安全的，不受上述攻击。最后，我们证明了符合PRAGTHOS框架的POW区块链在一定的条件下是安全的。



## **32. GAIN: Enhancing Byzantine Robustness in Federated Learning with Gradient Decomposition**

Gain：增强梯度分解联合学习的拜占庭稳健性 cs.LG

**SubmitDate**: 2023-02-13    [abs](http://arxiv.org/abs/2302.06079v1) [paper-pdf](http://arxiv.org/pdf/2302.06079v1)

**Authors**: Yuchen Liu, Chen Chen, Lingjuan Lyu, Fangzhao Wu, Sai Wu, Gang Chen

**Abstract**: Federated learning provides a privacy-aware learning framework by enabling participants to jointly train models without exposing their private data. However, federated learning has exhibited vulnerabilities to Byzantine attacks, where the adversary aims to destroy the convergence and performance of the global model. Meanwhile, we observe that most existing robust AGgregation Rules (AGRs) fail to stop the aggregated gradient deviating from the optimal gradient (the average of honest gradients) in the non-IID setting. We attribute the reason of the failure of these AGRs to two newly proposed concepts: identification failure and integrity failure. The identification failure mainly comes from the exacerbated curse of dimensionality in the non-IID setting. The integrity failure is a combined result of conservative filtering strategy and gradient heterogeneity. In order to address both failures, we propose GAIN, a gradient decomposition scheme that can help adapt existing robust algorithms to heterogeneous datasets. We also provide convergence analysis for integrating existing robust AGRs into GAIN. Experiments on various real-world datasets verify the efficacy of our proposed GAIN.

摘要: 联合学习通过允许参与者在不暴露其私人数据的情况下联合训练模型来提供隐私意识学习框架。然而，联合学习在拜占庭攻击中表现出脆弱性，在拜占庭攻击中，对手的目标是破坏全局模型的收敛和性能。同时，我们观察到大多数现有的稳健聚集规则(AGR)不能阻止在非IID设置下聚集的梯度偏离最优梯度(诚实梯度的平均值)。我们将这些AGR失败的原因归因于两个新提出的概念：识别失败和完整性失败。识别失败主要来自于非IID情境中维度诅咒的加剧。完整性失效是保守过滤策略和梯度非均质性共同作用的结果。为了解决这两个问题，我们提出了Gain，一种梯度分解方案，它可以帮助现有的健壮算法适应不同的数据集。我们还给出了将现有的稳健AGR集成到Gain中的收敛分析。在各种真实数据集上的实验验证了我们所提出的增益的有效性。



## **33. An Integrated Approach to Produce Robust Models with High Efficiency**

一种高效生成稳健模型的集成方法 cs.CV

**SubmitDate**: 2023-02-13    [abs](http://arxiv.org/abs/2008.13305v4) [paper-pdf](http://arxiv.org/pdf/2008.13305v4)

**Authors**: Zhijian Li, Bao Wang, Jack Xin

**Abstract**: Deep Neural Networks (DNNs) needs to be both efficient and robust for practical uses. Quantization and structure simplification are promising ways to adapt DNNs to mobile devices, and adversarial training is the most popular method to make DNNs robust. In this work, we try to obtain both features by applying a convergent relaxation quantization algorithm, Binary-Relax (BR), to a robust adversarial-trained model, ResNets Ensemble via Feynman-Kac Formalism (EnResNet). We also discover that high precision, such as ternary (tnn) and 4-bit, quantization will produce sparse DNNs. However, this sparsity is unstructured under advarsarial training. To solve the problems that adversarial training jeopardizes DNNs' accuracy on clean images and the struture of sparsity, we design a trade-off loss function that helps DNNs preserve their natural accuracy and improve the channel sparsity. With our trade-off loss function, we achieve both goals with no reduction of resistance under weak attacks and very minor reduction of resistance under strong attcks. Together with quantized EnResNet with trade-off loss function, we provide robust models that have high efficiency.

摘要: 深度神经网络(DNN)在实际应用中需要既高效又健壮。量化和结构简化是使DNN适应移动设备的有效方法，而对抗性训练是使DNN具有健壮性的最常用方法。在这项工作中，我们试图通过将一种收敛的松弛量化算法BINARY-RELAX(BR)应用于一个健壮的对抗性训练模型--通过Feynman-Kac形式的ResNets集成(EnResNet)来获得这两个特征。我们还发现，高精度的量化，如三进制(TNN)和4位，将产生稀疏DNN。然而，这种稀疏性在冒险训练下是无结构的。针对对抗性训练影响DNN图像清晰度和稀疏性的问题，设计了一种权衡损失函数，帮助DNN保持其自然的精确度，同时提高了信道的稀疏性。通过我们的权衡损失函数，我们实现了两个目标，在弱攻击下不减少抵抗，而在强攻击下减少很少的抵抗。与具有权衡损失函数的量化EnResNet相结合，我们提供了具有较高效率的稳健模型。



## **34. TextDefense: Adversarial Text Detection based on Word Importance Entropy**

文本防御：基于词重要度信息的敌意文本检测 cs.CL

**SubmitDate**: 2023-02-12    [abs](http://arxiv.org/abs/2302.05892v1) [paper-pdf](http://arxiv.org/pdf/2302.05892v1)

**Authors**: Lujia Shen, Xuhong Zhang, Shouling Ji, Yuwen Pu, Chunpeng Ge, Xing Yang, Yanghe Feng

**Abstract**: Currently, natural language processing (NLP) models are wildly used in various scenarios. However, NLP models, like all deep models, are vulnerable to adversarially generated text. Numerous works have been working on mitigating the vulnerability from adversarial attacks. Nevertheless, there is no comprehensive defense in existing works where each work targets a specific attack category or suffers from the limitation of computation overhead, irresistible to adaptive attack, etc.   In this paper, we exhaustively investigate the adversarial attack algorithms in NLP, and our empirical studies have discovered that the attack algorithms mainly disrupt the importance distribution of words in a text. A well-trained model can distinguish subtle importance distribution differences between clean and adversarial texts. Based on this intuition, we propose TextDefense, a new adversarial example detection framework that utilizes the target model's capability to defend against adversarial attacks while requiring no prior knowledge. TextDefense differs from previous approaches, where it utilizes the target model for detection and thus is attack type agnostic. Our extensive experiments show that TextDefense can be applied to different architectures, datasets, and attack methods and outperforms existing methods. We also discover that the leading factor influencing the performance of TextDefense is the target model's generalizability. By analyzing the property of the target model and the property of the adversarial example, we provide our insights into the adversarial attacks in NLP and the principles of our defense method.

摘要: 目前，自然语言处理(NLP)模型被广泛应用于各种场景。然而，像所有深度模型一样，NLP模型很容易受到恶意生成的文本的影响。许多工作一直致力于减轻来自对抗性攻击的脆弱性。然而，在现有的工作中，每个工作都针对特定的攻击类别，或者受到计算开销的限制，无法抵抗自适应攻击等问题，没有全面的防御措施。本文对自然语言处理中的对抗性攻击算法进行了详尽的研究，我们的实证研究发现，攻击算法主要扰乱了文本中单词的重要性分布。一个训练有素的模型可以区分干净文本和对抗性文本之间微妙的重要性分布差异。基于这一直觉，我们提出了一种新的对抗性实例检测框架TextDefense，该框架利用目标模型的能力来防御对抗性攻击，而不需要先验知识。TextDefense与以前的方法不同，在以前的方法中，它利用目标模型进行检测，因此与攻击类型无关。我们的大量实验表明，TextDefense可以应用于不同的体系结构、数据集和攻击方法，并且性能优于现有的方法。我们还发现，影响文本防御性能的主要因素是目标模型的泛化能力。通过分析目标模型的性质和对抗性实例的性质，我们对自然语言处理中的对抗性攻击和我们的防御方法的原理提出了自己的见解。



## **35. Mutation-Based Adversarial Attacks on Neural Text Detectors**

基于突变的神经文本检测器的敌意攻击 cs.CR

**SubmitDate**: 2023-02-11    [abs](http://arxiv.org/abs/2302.05794v1) [paper-pdf](http://arxiv.org/pdf/2302.05794v1)

**Authors**: Gongbo Liang, Jesus Guerrero, Izzat Alsmadi

**Abstract**: Neural text detectors aim to decide the characteristics that distinguish neural (machine-generated) from human texts. To challenge such detectors, adversarial attacks can alter the statistical characteristics of the generated text, making the detection task more and more difficult. Inspired by the advances of mutation analysis in software development and testing, in this paper, we propose character- and word-based mutation operators for generating adversarial samples to attack state-of-the-art natural text detectors. This falls under white-box adversarial attacks. In such attacks, attackers have access to the original text and create mutation instances based on this original text. The ultimate goal is to confuse machine learning models and classifiers and decrease their prediction accuracy.

摘要: 神经文本检测器旨在确定区分神经(机器生成的)和人类文本的特征。为了挑战这些检测器，敌意攻击会改变生成文本的统计特征，使得检测任务变得越来越困难。受软件开发和测试中突变分析的发展启发，本文提出了基于字符和单词的突变算子来生成敌意样本，以攻击最先进的自然文本检测器。这属于白盒对抗性攻击。在此类攻击中，攻击者可以访问原始文本，并基于该原始文本创建突变实例。最终目的是混淆机器学习模型和分类器，降低它们的预测精度。



## **36. Escaping saddle points in zeroth-order optimization: the power of two-point estimators**

零阶最优化中的鞍点逃逸：两点估计的能力 math.OC

This new version includes an improved sample complexity result for  strict saddle functions, new simulation results, an updated introduction, as  well as more streamlined proof outlines

**SubmitDate**: 2023-02-11    [abs](http://arxiv.org/abs/2209.13555v2) [paper-pdf](http://arxiv.org/pdf/2209.13555v2)

**Authors**: Zhaolin Ren, Yujie Tang, Na Li

**Abstract**: Two-point zeroth order methods are important in many applications of zeroth-order optimization, such as robotics, wind farms, power systems, online optimization, and adversarial robustness to black-box attacks in deep neural networks, where the problem may be high-dimensional and/or time-varying. Most problems in these applications are nonconvex and contain saddle points. While existing works have shown that zeroth-order methods utilizing $\Omega(d)$ function valuations per iteration (with $d$ denoting the problem dimension) can escape saddle points efficiently, it remains an open question if zeroth-order methods based on two-point estimators can escape saddle points. In this paper, we show that by adding an appropriate isotropic perturbation at each iteration, a zeroth-order algorithm based on $2m$ (for any $1 \leq m \leq d$) function evaluations per iteration can not only find $\epsilon$-second order stationary points polynomially fast, but do so using only $\tilde{O}\left(\frac{d}{\epsilon^{2}\bar{\psi}}\right)$ function evaluations, where $\bar{\psi} \geq \tilde{\Omega}(\sqrt{\epsilon})$ is a parameter capturing the extent to which the function of interest exhibits the strict saddle property.

摘要: 两点零阶方法在零阶优化的许多应用中都是重要的，例如机器人、风电场、电力系统、在线优化以及深层神经网络中对黑盒攻击的对抗鲁棒性，这些问题可能是高维的和/或时变的。这些应用中的大多数问题都是非凸的，并且包含鞍点。虽然已有的工作表明，利用每次迭代的$\Omega(D)$函数赋值(其中$d$表示问题的维度)的零级方法可以有效地逃离鞍点，但基于两点估计的零级方法是否能够逃离鞍点仍然是一个悬而未决的问题。本文证明了，通过在每次迭代中加入适当的各向同性扰动，基于每一次迭代的$2m$(对于任意$1\leq m\leq d$)函数求值的零阶算法不仅可以多项式地快速地找到$-二阶驻点，而且只使用$\tilde{O}\left(\frac{d}{\epsilon^{2}\bar{\psi}}\right)$函数求值，其中，$\bar{\psi}\geq\tilde{\omega}(\sqrt{\epsilon})$是一个参数，用于捕获感兴趣的函数表现出严格鞍形属性的程度。



## **37. Exploiting Programmatic Behavior of LLMs: Dual-Use Through Standard Security Attacks**

利用LLMS的编程行为：通过标准安全攻击实现双重用途 cs.CR

**SubmitDate**: 2023-02-11    [abs](http://arxiv.org/abs/2302.05733v1) [paper-pdf](http://arxiv.org/pdf/2302.05733v1)

**Authors**: Daniel Kang, Xuechen Li, Ion Stoica, Carlos Guestrin, Matei Zaharia, Tatsunori Hashimoto

**Abstract**: Recent advances in instruction-following large language models (LLMs) have led to dramatic improvements in a range of NLP tasks. Unfortunately, we find that the same improved capabilities amplify the dual-use risks for malicious purposes of these models. Dual-use is difficult to prevent as instruction-following capabilities now enable standard attacks from computer security. The capabilities of these instruction-following LLMs provide strong economic incentives for dual-use by malicious actors. In particular, we show that instruction-following LLMs can produce targeted malicious content, including hate speech and scams, bypassing in-the-wild defenses implemented by LLM API vendors. Our analysis shows that this content can be generated economically and at cost likely lower than with human effort alone. Together, our findings suggest that LLMs will increasingly attract more sophisticated adversaries and attacks, and addressing these attacks may require new approaches to mitigations.

摘要: 最近在教学遵循大语言模型(LLM)方面的进展导致了一系列NLP任务的显著改进。不幸的是，我们发现，同样改进的功能放大了这些模型出于恶意目的而使用双重用途的风险。双重用途很难防止，因为遵循指令的能力现在可以从计算机安全角度进行标准攻击。这些遵循指令的LLM的能力为恶意行为者的双重使用提供了强大的经济激励。特别是，我们展示了遵循指令的LLM可以产生有针对性的恶意内容，包括仇恨言论和骗局，绕过了LLMAPI供应商实施的野外防御。我们的分析表明，这种内容可以以经济的方式产生，而且成本可能低于仅靠人力产生的内容。总而言之，我们的发现表明，LLMS将越来越多地吸引更复杂的对手和攻击，解决这些攻击可能需要新的方法来缓解。



## **38. HateProof: Are Hateful Meme Detection Systems really Robust?**

HateProof：仇恨模因检测系统真的很强大吗？ cs.CL

Accepted at TheWebConf'2023 (WWW'2023)

**SubmitDate**: 2023-02-11    [abs](http://arxiv.org/abs/2302.05703v1) [paper-pdf](http://arxiv.org/pdf/2302.05703v1)

**Authors**: Piush Aggarwal, Pranit Chawla, Mithun Das, Punyajoy Saha, Binny Mathew, Torsten Zesch, Animesh Mukherjee

**Abstract**: Exploiting social media to spread hate has tremendously increased over the years. Lately, multi-modal hateful content such as memes has drawn relatively more traction than uni-modal content. Moreover, the availability of implicit content payloads makes them fairly challenging to be detected by existing hateful meme detection systems. In this paper, we present a use case study to analyze such systems' vulnerabilities against external adversarial attacks. We find that even very simple perturbations in uni-modal and multi-modal settings performed by humans with little knowledge about the model can make the existing detection models highly vulnerable. Empirically, we find a noticeable performance drop of as high as 10% in the macro-F1 score for certain attacks. As a remedy, we attempt to boost the model's robustness using contrastive learning as well as an adversarial training-based method - VILLA. Using an ensemble of the above two approaches, in two of our high resolution datasets, we are able to (re)gain back the performance to a large extent for certain attacks. We believe that ours is a first step toward addressing this crucial problem in an adversarial setting and would inspire more such investigations in the future.

摘要: 多年来，利用社交媒体传播仇恨的情况大大增加。最近，模因等多模式仇恨内容吸引了相对更多的吸引力，而不是单一模式的内容。此外，隐含内容有效负载的可用性使得它们很难被现有的仇恨模因检测系统检测到。在这篇文章中，我们提供了一个用例研究来分析这类系统对外部对手攻击的脆弱性。我们发现，即使是在对模型知之甚少的情况下，在单模式和多模式设置中进行非常简单的扰动，也会使现有的检测模型非常脆弱。根据经验，我们发现对于某些攻击，在宏观F1得分中有高达10%的显著性能下降。作为补救措施，我们尝试使用对比学习和基于对抗性训练的方法Villa来提高模型的稳健性。使用以上两种方法的集成，在我们的两个高分辨率数据集中，我们能够(重新)在很大程度上恢复某些攻击的性能。我们认为，我们的行动是朝着在对抗性环境中解决这一关键问题迈出的第一步，并将促使今后开展更多此类调查。



## **39. High Recovery with Fewer Injections: Practical Binary Volumetric Injection Attacks against Dynamic Searchable Encryption**

以更少的注入实现高恢复：针对动态可搜索加密的实用二进制体积注入攻击 cs.CR

22 pages, 19 fugures, will be published in USENIX Security 2023

**SubmitDate**: 2023-02-11    [abs](http://arxiv.org/abs/2302.05628v1) [paper-pdf](http://arxiv.org/pdf/2302.05628v1)

**Authors**: Xianglong Zhang, Wei Wang, Peng Xu, Laurence T. Yang, Kaitai Liang

**Abstract**: Searchable symmetric encryption enables private queries over an encrypted database, but it also yields information leakages. Adversaries can exploit these leakages to launch injection attacks (Zhang et al., USENIX'16) to recover the underlying keywords from queries. The performance of the existing injection attacks is strongly dependent on the amount of leaked information or injection. In this work, we propose two new injection attacks, namely BVA and BVMA, by leveraging a binary volumetric approach. We enable adversaries to inject fewer files than the existing volumetric attacks by using the known keywords and reveal the queries by observing the volume of the query results. Our attacks can thwart well-studied defenses (e.g., threshold countermeasure, static padding) without exploiting the distribution of target queries and client databases. We evaluate the proposed attacks empirically in real-world datasets with practical queries. The results show that our attacks can obtain a high recovery rate (>80%) in the best case and a roughly 60% recovery even under a large-scale dataset with a small number of injections (<20 files).

摘要: 可搜索对称加密允许对加密的数据库进行私有查询，但它也会导致信息泄漏。攻击者可以利用这些泄漏来发起注入攻击(Zhang等人，USENIX‘16)，以从查询中恢复潜在的关键字。现有注入攻击的性能在很大程度上依赖于泄漏的信息量或注入。在这项工作中，我们提出了两个新的注入攻击，即BVA和BVMA，利用二进制体积方法。通过使用已知关键字，使攻击者能够比现有的体积攻击注入更少的文件，并通过观察查询结果的量来揭示查询。我们的攻击可以在不利用目标查询和客户端数据库的分布的情况下挫败经过充分研究的防御(例如，阈值对抗、静态填充)。我们通过实际查询在真实数据集中对所提出的攻击进行了经验性评估。结果表明，在最好的情况下，我们的攻击可以获得很高的恢复率(>80%)，即使在注入次数较少的大规模数据集(<20个文件)下，也可以获得大约60%的恢复。



## **40. Towards A Proactive ML Approach for Detecting Backdoor Poison Samples**

一种主动检测后门毒物样本的最大似然方法 cs.LG

**SubmitDate**: 2023-02-10    [abs](http://arxiv.org/abs/2205.13616v2) [paper-pdf](http://arxiv.org/pdf/2205.13616v2)

**Authors**: Xiangyu Qi, Tinghao Xie, Jiachen T. Wang, Tong Wu, Saeed Mahloujifar, Prateek Mittal

**Abstract**: Adversaries can embed backdoors in deep learning models by introducing backdoor poison samples into training datasets. In this work, we investigate how to detect such poison samples to mitigate the threat of backdoor attacks. First, we uncover a post-hoc workflow underlying most prior work, where defenders passively allow the attack to proceed and then leverage the characteristics of the post-attacked model to uncover poison samples. We reveal that this workflow does not fully exploit defenders' capabilities, and defense pipelines built on it are prone to failure or performance degradation in many scenarios. Second, we suggest a paradigm shift by promoting a proactive mindset in which defenders engage proactively with the entire model training and poison detection pipeline, directly enforcing and magnifying distinctive characteristics of the post-attacked model to facilitate poison detection. Based on this, we formulate a unified framework and provide practical insights on designing detection pipelines that are more robust and generalizable. Third, we introduce the technique of Confusion Training (CT) as a concrete instantiation of our framework. CT applies an additional poisoning attack to the already poisoned dataset, actively decoupling benign correlation while exposing backdoor patterns to detection. Empirical evaluations on 4 datasets and 14 types of attacks validate the superiority of CT over 11 baseline defenses.

摘要: 攻击者可以通过将后门毒药样本引入训练数据集中，在深度学习模型中嵌入后门。在这项工作中，我们研究如何检测此类毒物样本以减轻后门攻击的威胁。首先，我们揭示了大多数先前工作背后的事后工作流，在这种工作中，防御者被动地允许攻击继续进行，然后利用攻击后模型的特征来发现毒物样本。我们发现，这种工作流没有充分利用防御者的能力，在其上构建的防御管道在许多场景下容易出现故障或性能下降。其次，我们建议通过促进一种积极主动的心态来实现范式转变，在这种心态中，防御者主动参与整个模型培训和毒物检测管道，直接实施和放大攻击后模型的独特特征，以促进毒物检测。在此基础上，我们制定了一个统一的框架，并为设计更健壮和更具通用性的检测管道提供了实用的见解。第三，我们引入混淆训练(CT)技术作为我们的框架的具体实例。CT对已经中毒的数据集进行额外的中毒攻击，主动去耦合良性关联，同时暴露后门模式以供检测。在4个数据集和14种攻击类型上的经验评估验证了CT相对于11个基线防御的优越性。



## **41. Computing a Best Response against a Maximum Disruption Attack**

计算针对最大中断攻击的最佳响应 cs.GT

35 pages, 7 figures

**SubmitDate**: 2023-02-10    [abs](http://arxiv.org/abs/2302.05348v1) [paper-pdf](http://arxiv.org/pdf/2302.05348v1)

**Authors**: Carme Àlvarez, Arnau Messegué

**Abstract**: Inspired by scenarios where the strategic network design and defense or immunisation are of the central importance, Goyal et al. [3] defined a new Network Formation Game with Attack and Immunisation. The authors showed that despite the presence of attacks, the game has high social welfare properties and even though the equilibrium networks can contain cycles, the number of edges is strongly bounded. Subsequently, Friedrich et al. [10] provided a polynomial time algorithm for computing a best response strategy for the maximum carnage adversary which tries to kill as many nodes as possible, and for the random attack adversary, but they left open the problem for the case of maximum disruption adversary. This adversary attacks the vulnerable region that minimises the post-attack social welfare. In this paper we address our efforts to this question. We can show that computing a best response strategy given a player u and the strategies of all players but u, is polynomial time solvable when the initial network resulting from the given strategies is connected. Our algorithm is based on a dynamic programming and has some reminiscence to the knapsack-problem, although is considerably more complex and involved.

摘要: GoYal等人受到战略网络设计和防御或免疫至关重要的场景的启发。[3]定义了一种新的具有攻击和免疫的网络队形博弈。作者指出，尽管存在攻击，但博弈具有很高的社会福利性质，即使均衡网络可以包含圈，边的数目也是强有界的。随后，Friedrich et al.[10]给出了一个多项式时间算法，用于计算试图杀死尽可能多节点的最大杀戮对手和随机攻击对手的最优响应策略，但对于最大破坏对手的情况则没有解决。这个对手攻击脆弱的地区，使攻击后的社会福利降至最低。在本文中，我们将致力于解决这一问题。我们可以证明，当由给定策略产生的初始网络连通时，计算给定的参与者u和除u之外的所有参与者的策略的最佳响应策略是多项式时间可解的。我们的算法是基于动态规划的，并与背包问题有一些相似之处，尽管要复杂得多。



## **42. Step by Step Loss Goes Very Far: Multi-Step Quantization for Adversarial Text Attacks**

逐级损失：针对对抗性文本攻击的多步量化 cs.CL

**SubmitDate**: 2023-02-10    [abs](http://arxiv.org/abs/2302.05120v1) [paper-pdf](http://arxiv.org/pdf/2302.05120v1)

**Authors**: Piotr Gaiński, Klaudia Bałazy

**Abstract**: We propose a novel gradient-based attack against transformer-based language models that searches for an adversarial example in a continuous space of token probabilities. Our algorithm mitigates the gap between adversarial loss for continuous and discrete text representations by performing multi-step quantization in a quantization-compensation loop. Experiments show that our method significantly outperforms other approaches on various natural language processing (NLP) tasks.

摘要: 针对基于变换的语言模型，我们提出了一种新的基于梯度的攻击，该攻击在连续的令牌概率空间中搜索对抗性实例。我们的算法通过在量化-补偿循环中执行多步量化来缓解连续文本表示和离散文本表示之间的对抗性损失之间的差距。实验表明，我们的方法在各种自然语言处理(NLP)任务上的性能明显优于其他方法。



## **43. Making Substitute Models More Bayesian Can Enhance Transferability of Adversarial Examples**

使替代模型更具贝叶斯性质可以增强对抗性例子的可转移性 cs.LG

Accepted by ICLR 2023

**SubmitDate**: 2023-02-10    [abs](http://arxiv.org/abs/2302.05086v1) [paper-pdf](http://arxiv.org/pdf/2302.05086v1)

**Authors**: Qizhang Li, Yiwen Guo, Wangmeng Zuo, Hao Chen

**Abstract**: The transferability of adversarial examples across deep neural networks (DNNs) is the crux of many black-box attacks. Many prior efforts have been devoted to improving the transferability via increasing the diversity in inputs of some substitute models. In this paper, by contrast, we opt for the diversity in substitute models and advocate to attack a Bayesian model for achieving desirable transferability. Deriving from the Bayesian formulation, we develop a principled strategy for possible finetuning, which can be combined with many off-the-shelf Gaussian posterior approximations over DNN parameters. Extensive experiments have been conducted to verify the effectiveness of our method, on common benchmark datasets, and the results demonstrate that our method outperforms recent state-of-the-arts by large margins (roughly 19% absolute increase in average attack success rate on ImageNet), and, by combining with these recent methods, further performance gain can be obtained. Our code: https://github.com/qizhangli/MoreBayesian-attack.

摘要: 恶意例子在深度神经网络(DNN)之间的可转移性是许多黑盒攻击的症结所在。先前的许多努力都致力于通过增加一些替代模型的投入的多样性来提高可转移性。相反，在本文中，我们选择了替代模型的多样性，并主张攻击贝叶斯模型，以实现理想的可转移性。从贝叶斯公式出发，我们开发了一种可能的精调的原则性策略，该策略可以与许多关于DNN参数的现成的高斯后验近似相结合。在常见的基准数据集上进行了广泛的实验来验证我们的方法的有效性，结果表明我们的方法的性能比最近的最新技术有很大的差距(在ImageNet上的平均攻击成功率大约绝对提高了19%)，并且通过结合这些最新的方法，可以获得进一步的性能提升。我们的代码：https://github.com/qizhangli/MoreBayesian-attack.



## **44. Increasing-Margin Adversarial (IMA) Training to Improve Adversarial Robustness of Neural Networks**

增量对抗性(IMA)训练提高神经网络对抗性鲁棒性 cs.CV

26 pages

**SubmitDate**: 2023-02-10    [abs](http://arxiv.org/abs/2005.09147v10) [paper-pdf](http://arxiv.org/pdf/2005.09147v10)

**Authors**: Linhai Ma, Liang Liang

**Abstract**: Deep neural networks (DNNs) are vulnerable to adversarial noises. Adversarial training is a general and effective strategy to improve DNN robustness (i.e., accuracy on noisy data) against adversarial noises. However, DNN models trained by the current existing adversarial training methods may have much lower standard accuracy (i.e., accuracy on clean data), compared to the same models trained by the standard method on clean data, and this phenomenon is known as the trade-off between accuracy and robustness and is considered unavoidable. This issue prevents adversarial training from being used in many application domains, such as medical image analysis, as practitioners do not want to sacrifice standard accuracy too much in exchange for adversarial robustness. Our objective is to lift (i.e., alleviate or even avoid) this trade-off between standard accuracy and adversarial robustness for medical image classification and segmentation. We propose a novel adversarial training method, named Increasing-Margin Adversarial (IMA) Training, which is supported by an equilibrium state analysis about the optimality of adversarial training samples. Our method aims to preserve accuracy while improving robustness by generating optimal adversarial training samples. We evaluate our method and the other eight representative methods on six publicly available image datasets corrupted by noises generated by AutoAttack and white-noise attack. Our method achieves the highest adversarial robustness for image classification and segmentation with the smallest reduction in accuracy on clean data. For one of the applications, our method improves both accuracy and robustness. Our study has demonstrated that our method can lift the trade-off between standard accuracy and adversarial robustness for the image classification and segmentation applications.

摘要: 深度神经网络(DNN)很容易受到对抗性噪声的影响。对抗性训练是提高DNN对对抗性噪声的稳健性(即对噪声数据的准确性)的一种通用而有效的策略。然而，与基于干净数据的标准方法训练的相同模型相比，由现有对抗性训练方法训练的DNN模型的标准精度(即基于干净数据的准确性)可能要低得多，这种现象被称为精度和稳健性之间的权衡，被认为是不可避免的。这一问题阻碍了对抗性训练在许多应用领域中的使用，例如医学图像分析，因为实践者不想以太多牺牲标准精度来换取对抗性健壮性。我们的目标是提高(即减轻甚至避免)医学图像分类和分割的标准准确率和对抗性稳健性之间的权衡。提出了一种新的对抗性训练方法--增量对抗性(IMA)训练方法，该方法基于对抗性训练样本最优性的均衡分析。我们的方法旨在通过生成最优的对抗性训练样本来保持准确性，同时提高稳健性。我们在六个被AutoAttack和白噪声攻击产生的噪声破坏的公开可用的图像数据集上对我们的方法和其他八种有代表性的方法进行了评估。我们的方法在对干净数据的精确度降低最小的情况下，获得了最高的图像分类和分割的对抗性鲁棒性。对于其中一个应用，我们的方法提高了精度和稳健性。我们的研究表明，对于图像分类和分割应用，我们的方法可以在标准准确率和对抗性稳健性之间进行权衡。



## **45. RAPTOR: Advanced Persistent Threat Detection in Industrial IoT via Attack Stage Correlation**

Raptor：通过攻击阶段关联实现工业物联网的高级持续威胁检测 cs.CR

To be submitted to journal

**SubmitDate**: 2023-02-09    [abs](http://arxiv.org/abs/2301.11524v2) [paper-pdf](http://arxiv.org/pdf/2301.11524v2)

**Authors**: Ayush Kumar, Vrizlynn L. L. Thing

**Abstract**: IIoT (Industrial Internet-of-Things) systems are getting more prone to attacks by APT (Advanced Persistent Threat) adversaries. Past APT attacks on IIoT systems such as the 2016 Ukrainian power grid attack which cut off the capital Kyiv off power for an hour and the 2017 Saudi petrochemical plant attack which almost shut down the plant's safety controllers have shown that APT campaigns can disrupt industrial processes, shut down critical systems and endanger human lives. In this work, we propose RAPTOR, a system to detect APT campaigns in IIoT environments. RAPTOR detects and correlates various APT attack stages (adapted to IIoT) using multiple data sources. Subsequently, it constructs a high-level APT campaign graph which can be used by cybersecurity analysts towards attack analysis and mitigation. A performance evaluation of RAPTOR's APT stage detection stages shows high precision and low false positive/negative rates. We also show that RAPTOR is able to construct the APT campaign graph for APT attacks (modelled after real-world attacks on ICS/OT infrastructure) executed on our IIoT testbed.

摘要: IIoT(工业物联网)系统越来越容易受到APT(高级持久威胁)对手的攻击。过去对IIoT系统的APT攻击，例如2016年乌克兰电网袭击导致首都基辅停电一小时，以及2017年沙特石化厂袭击事件，几乎关闭了工厂的安全控制器，这些都表明，APT行动可以扰乱工业流程，关闭关键系统，并危及人类生命。在这项工作中，我们提出了Raptor，一个用于检测IIoT环境中的APT活动的系统。Raptor使用多个数据源检测和关联不同的APT攻击阶段(适应IIoT)。随后，构建了一个高级APT运动图，可供网络安全分析人员用于攻击分析和缓解。对Raptor的APT阶段检测阶段的性能评估表明，该阶段的检测精度高，假阳性/阴性率低。我们还展示了Raptor能够为在我们的IIoT试验台上执行的APT攻击(模仿真实世界对ICS/OT基础设施的攻击)构建APT活动图。



## **46. Testing robustness of predictions of trained classifiers against naturally occurring perturbations**

测试训练分类器的预测对自然发生的扰动的稳健性 cs.LG

25 pages, 7 figures

**SubmitDate**: 2023-02-09    [abs](http://arxiv.org/abs/2204.10046v2) [paper-pdf](http://arxiv.org/pdf/2204.10046v2)

**Authors**: Sebastian Scher, Andreas Trügler

**Abstract**: Correctly quantifying the robustness of machine learning models is a central aspect in judging their suitability for specific tasks, and ultimately, for generating trust in them. We address the problem of finding the robustness of individual predictions. We show both theoretically and with empirical examples that a method based on counterfactuals that was previously proposed for this is insufficient, as it is not a valid metric for determining the robustness against perturbations that occur ``naturally'', outside specific adversarial attack scenarios. We propose a flexible approach that models possible perturbations in input data individually for each application. This is then combined with a probabilistic approach that computes the likelihood that a ``real-world'' perturbation will change a prediction, thus giving quantitative information of the robustness of individual predictions of the trained machine learning model. The method does not require access to the internals of the classifier and thus in principle works for any black-box model. It is, however, based on Monte-Carlo sampling and thus only suited for input spaces with small dimensions. We illustrate our approach on the Iris and the Ionosphere datasets, on an application predicting fog at an airport, and on analytically solvable cases.

摘要: 正确量化机器学习模型的稳健性是判断它们是否适合特定任务的一个核心方面，并最终产生对它们的信任。我们解决了找到单个预测的稳健性的问题。我们在理论上和经验上都表明，以前提出的基于反事实的方法是不够的，因为它不是一个有效的衡量标准，用于确定在特定的对抗性攻击场景之外对“自然”发生的扰动的稳健性。我们提出了一种灵活的方法，为每个应用程序分别建模输入数据中可能的扰动。然后，将其与概率方法相结合，该概率方法计算“真实世界”扰动将改变预测的可能性，从而给出关于训练的机器学习模型的单个预测的稳健性的定量信息。该方法不需要访问分类器的内部，因此原则上适用于任何黑盒模型。然而，它是基于蒙特卡罗抽样的，因此只适用于小维度的输入空间。我们在虹膜和电离层数据集上，在预测机场雾的应用程序上，以及在解析可解的情况下，说明了我们的方法。



## **47. Tracking Fringe and Coordinated Activity on Twitter Leading Up To the US Capitol Attack**

追踪美国国会大厦袭击前推特上的边缘和协调活动 cs.SI

11 pages (including references), 8 figures, 1 table. Submitted to The  17th International AAAI Conference on Web and Social Media

**SubmitDate**: 2023-02-09    [abs](http://arxiv.org/abs/2302.04450v1) [paper-pdf](http://arxiv.org/pdf/2302.04450v1)

**Authors**: Vishnuprasad Padinjaredath Suresh, Gianluca Nogara, Felipe Cardoso, Stefano Cresci, Silvia Giordano, Luca Luceri

**Abstract**: The aftermath of the 2020 US Presidential Election witnessed an unprecedented attack on the democratic values of the country through the violent insurrection at Capitol Hill on January 6th, 2021. The attack was fueled by the proliferation of conspiracy theories and misleading claims about the integrity of the election pushed by political elites and fringe communities on social media. In this study, we explore the evolution of fringe content and conspiracy theories on Twitter in the seven months leading up to the Capitol attack. We examine the suspicious coordinated activity carried out by users sharing fringe content, finding evidence of common adversarial manipulation techniques ranging from targeted amplification to manufactured consensus. Further, we map out the temporal evolution of, and the relationship between, fringe and conspiracy theories, which eventually coalesced into the rhetoric of a stolen election, with the hashtag #stopthesteal, alongside QAnon-related narratives. Our findings further highlight how social media platforms offer fertile ground for the widespread proliferation of conspiracies during major societal events, which can potentially lead to offline coordinated actions and organized violence.

摘要: 在2020年美国总统选举的余波中，2021年1月6日发生在国会山的暴力起义，见证了美国民主价值观受到前所未有的攻击。政治精英和边缘社区在社交媒体上推动的关于选举完整性的误导性言论和阴谋论的扩散，助长了这次袭击。在这项研究中，我们探索了推特上边缘内容和阴谋论在国会大厦袭击前的七个月里的演变。我们检查了共享边缘内容的用户进行的可疑的协调活动，发现了常见的对抗性操纵技术的证据，范围从定向放大到制造共识。此外，我们绘制了边缘理论和阴谋论的时间演变以及它们之间的关系，这些理论最终结合成了一场被盗选举的修辞，标签为#Stop thesteal，以及与QAnon相关的叙述。我们的发现进一步突显了社交媒体平台如何为重大社会活动期间阴谋的广泛扩散提供了肥沃的土壤，这可能会导致线下协调行动和有组织的暴力。



## **48. Leveraging the Verifier's Dilemma to Double Spend in Bitcoin**

利用验证者的两难境地加倍投入比特币 cs.CR

**SubmitDate**: 2023-02-09    [abs](http://arxiv.org/abs/2210.14072v2) [paper-pdf](http://arxiv.org/pdf/2210.14072v2)

**Authors**: Tong Cao, Jérémie Decouchant, Jiangshan Yu

**Abstract**: We describe and analyze perishing mining, a novel block-withholding mining strategy that lures profit-driven miners away from doing useful work on the public chain by releasing block headers from a privately maintained chain. We then introduce the dual private chain (DPC) attack, where an adversary that aims at double spending increases its success rate by intermittently dedicating part of its hash power to perishing mining. We detail the DPC attack's Markov decision process, evaluate its double spending success rate using Monte Carlo simulations. We show that the DPC attack lowers Bitcoin's security bound in the presence of profit-driven miners that do not wait to validate the transactions of a block before mining on it.

摘要: 我们描述和分析了正在灭亡的挖掘，这是一种新的块扣留挖掘策略，通过从私人维护的链中释放块头来引诱受利润驱动的矿工远离在公共链上做有用的工作。然后，我们介绍了双重私有链(DPC)攻击，在这种攻击中，一个旨在加倍支出的对手通过断断续续地将其部分散列能力用于消灭挖掘来提高其成功率。详细描述了DPC攻击的马尔可夫决策过程，并利用蒙特卡罗模拟对其双开销成功率进行了评估。我们表明，在利润驱动的矿工在场的情况下，DPC攻击降低了比特币的安全界限，这些矿工在挖掘比特币之前不会等待验证区块的交易。



## **49. Exploiting Certified Defences to Attack Randomised Smoothing**

利用认证防御攻击随机平滑 cs.LG

15 pages, 7 figures

**SubmitDate**: 2023-02-09    [abs](http://arxiv.org/abs/2302.04379v1) [paper-pdf](http://arxiv.org/pdf/2302.04379v1)

**Authors**: Andrew C. Cullen, Paul Montague, Shijie Liu, Sarah M. Erfani, Benjamin I. P. Rubinstein

**Abstract**: In guaranteeing that no adversarial examples exist within a bounded region, certification mechanisms play an important role in neural network robustness. Concerningly, this work demonstrates that the certification mechanisms themselves introduce a new, heretofore undiscovered attack surface, that can be exploited by attackers to construct smaller adversarial perturbations. While these attacks exist outside the certification region in no way invalidate certifications, minimising a perturbation's norm significantly increases the level of difficulty associated with attack detection. In comparison to baseline attacks, our new framework yields smaller perturbations more than twice as frequently as any other approach, resulting in an up to $34 \%$ reduction in the median perturbation norm. That this approach also requires $90 \%$ less computational time than approaches like PGD. That these reductions are possible suggests that exploiting this new attack vector would allow attackers to more frequently construct hard to detect adversarial attacks, by exploiting the very systems designed to defend deployed models.

摘要: 在保证有界区域内不存在对抗性实例方面，认证机制对神经网络的健壮性起着重要的作用。令人担忧的是，这项工作表明，认证机制本身引入了一个迄今为止尚未发现的新的攻击面，攻击者可以利用该攻击面来构建较小的对抗性扰动。虽然这些攻击存在于认证区域之外，但不会使认证失效，最大限度地减少扰动的规范会显著增加与攻击检测相关的难度。与基线攻击相比，我们的新框架产生较小扰动的频率是任何其他方法的两倍以上，导致中值扰动范数减少高达34美元。这种方法所需的计算时间也比像PGD这样的方法少90美元。这些减少是可能的，这表明利用这种新的攻击载体将允许攻击者通过利用专为防御部署的模型而设计的系统，更频繁地构建难以检测的对手攻击。



## **50. A Comprehensive Test Pattern Generation Approach Exploiting SAT Attack for Logic Locking**

一种利用SAT攻击进行逻辑锁定的综合测试码生成方法 cs.CR

12 pages, 7 figures, 5 tables

**SubmitDate**: 2023-02-08    [abs](http://arxiv.org/abs/2204.11307v4) [paper-pdf](http://arxiv.org/pdf/2204.11307v4)

**Authors**: Yadi Zhong, Ujjwal Guin

**Abstract**: The need for reducing manufacturing defect escape in today's safety-critical applications requires increased fault coverage. However, generating a test set using commercial automatic test pattern generation (ATPG) tools that lead to zero-defect escape is still an open problem. It is challenging to detect all stuck-at faults to reach 100% fault coverage. In parallel, the hardware security community has been actively involved in developing solutions for logic locking to prevent IP piracy. Locks (e.g., XOR gates) are inserted in different locations of the netlist so that an adversary cannot determine the secret key. Unfortunately, the Boolean satisfiability (SAT) based attack, introduced in [1], can break different logic locking schemes in minutes. In this paper, we propose a novel test pattern generation approach using the powerful SAT attack on logic locking. A stuck-at fault is modeled as a locked gate with a secret key. Our modeling of stuck-at faults preserves the property of fault activation and propagation. We show that the input pattern that determines the key is a test for the stuck-at fault. We propose two different approaches for test pattern generation. First, a single stuck-at fault is targeted, and a corresponding locked circuit with one key bit is created. This approach generates one test pattern per fault. Second, we consider a group of faults and convert the circuit to its locked version with multiple key bits. The inputs obtained from the SAT tool are the test set for detecting this group of faults. Our approach is able to find test patterns for hard-to-detect faults that were previously failed in commercial ATPG tools. The proposed test pattern generation approach can efficiently detect redundant faults present in a circuit. We demonstrate the effectiveness of the approach on ITC'99 benchmarks. The results show that we can achieve a perfect fault coverage reaching 100%.

摘要: 在当今的安全关键应用中，减少制造缺陷逃逸的需要需要增加故障覆盖率。然而，使用商业自动测试模式生成(ATPG)工具生成测试集以实现零缺陷逃逸仍然是一个未解决的问题。要检测所有固定故障以达到100%的故障覆盖率是具有挑战性的。与此同时，硬件安全界一直积极参与开发逻辑锁定解决方案，以防止知识产权盗版。锁(例如，异或门)被插入网表的不同位置，使得对手不能确定密钥。不幸的是，在[1]中引入的基于布尔可满足性(SAT)的攻击可以在几分钟内破解不同的逻辑锁定方案。在本文中，我们提出了一种新的测试模式生成方法，该方法利用了对逻辑锁的强大SAT攻击。一个顽固的错误被建模为一扇锁着的门和一把密钥。我们对固定故障的建模保留了故障激活和传播的性质。我们证明了决定关键字的输入模式是对固定错误的测试。我们提出了两种不同的测试模式生成方法。首先，针对单个固定故障，创建具有一个密钥位的相应锁定电路。该方法为每个故障生成一个测试模式。其次，我们考虑一组故障，并将电路转换为具有多个密钥位的锁定版本。从SAT工具获得的输入是用于检测这组故障的测试集。我们的方法能够为以前在商业ATPG工具中失败的难以检测的故障找到测试模式。提出的测试码生成方法可以有效地检测电路中存在的冗余故障。我们在ITC‘99基准上证明了该方法的有效性。结果表明，我们可以达到100%的完美故障覆盖率。



