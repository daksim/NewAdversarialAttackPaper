# Latest Adversarial Attack Papers
**update at 2023-03-29 16:29:53**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. A Survey on Malware Detection with Graph Representation Learning**

基于图表示学习的恶意软件检测综述 cs.CR

Preprint, submitted to ACM Computing Surveys on March 2023. For any  suggestions or improvements, please contact me directly by e-mail

**SubmitDate**: 2023-03-28    [abs](http://arxiv.org/abs/2303.16004v1) [paper-pdf](http://arxiv.org/pdf/2303.16004v1)

**Authors**: Tristan Bilot, Nour El Madhoun, Khaldoun Al Agha, Anis Zouaoui

**Abstract**: Malware detection has become a major concern due to the increasing number and complexity of malware. Traditional detection methods based on signatures and heuristics are used for malware detection, but unfortunately, they suffer from poor generalization to unknown attacks and can be easily circumvented using obfuscation techniques. In recent years, Machine Learning (ML) and notably Deep Learning (DL) achieved impressive results in malware detection by learning useful representations from data and have become a solution preferred over traditional methods. More recently, the application of such techniques on graph-structured data has achieved state-of-the-art performance in various domains and demonstrates promising results in learning more robust representations from malware. Yet, no literature review focusing on graph-based deep learning for malware detection exists. In this survey, we provide an in-depth literature review to summarize and unify existing works under the common approaches and architectures. We notably demonstrate that Graph Neural Networks (GNNs) reach competitive results in learning robust embeddings from malware represented as expressive graph structures, leading to an efficient detection by downstream classifiers. This paper also reviews adversarial attacks that are utilized to fool graph-based detection methods. Challenges and future research directions are discussed at the end of the paper.

摘要: 由于恶意软件的数量和复杂性不断增加，恶意软件检测已成为一个主要问题。传统的基于签名和启发式的检测方法被用于恶意软件检测，但遗憾的是，它们对未知攻击的泛化能力较差，可以通过混淆技术轻松地绕过。近年来，机器学习(ML)和深度学习(DL)通过从数据中学习有用的表示，在恶意软件检测方面取得了令人印象深刻的结果，并成为一种比传统方法更受欢迎的解决方案。最近，这种技术在图结构数据上的应用已经在各个领域取得了最先进的性能，并在从恶意软件中学习更健壮的表示方面展示了良好的结果。然而，目前还没有关于基于图的深度学习用于恶意软件检测的文献综述。在这次调查中，我们提供了深入的文献回顾，以总结和统一在共同的方法和架构下的现有工作。值得注意的是，图神经网络(GNN)在学习表示为可表达图结构的恶意软件的健壮嵌入方面取得了竞争的结果，从而导致了下游分类器的有效检测。本文还回顾了用于欺骗基于图的检测方法的对抗性攻击。在文章的最后，讨论了挑战和未来的研究方向。



## **2. TransAudio: Towards the Transferable Adversarial Audio Attack via Learning Contextualized Perturbations**

TransAudio：通过学习上下文扰动实现可转移的对抗性音频攻击 cs.SD

**SubmitDate**: 2023-03-28    [abs](http://arxiv.org/abs/2303.15940v1) [paper-pdf](http://arxiv.org/pdf/2303.15940v1)

**Authors**: Qi Gege, Yuefeng Chen, Xiaofeng Mao, Yao Zhu, Binyuan Hui, Xiaodan Li, Rong Zhang, Hui Xue

**Abstract**: In a transfer-based attack against Automatic Speech Recognition (ASR) systems, attacks are unable to access the architecture and parameters of the target model. Existing attack methods are mostly investigated in voice assistant scenarios with restricted voice commands, prohibiting their applicability to more general ASR related applications. To tackle this challenge, we propose a novel contextualized attack with deletion, insertion, and substitution adversarial behaviors, namely TransAudio, which achieves arbitrary word-level attacks based on the proposed two-stage framework. To strengthen the attack transferability, we further introduce an audio score-matching optimization strategy to regularize the training process, which mitigates adversarial example over-fitting to the surrogate model. Extensive experiments and analysis demonstrate the effectiveness of TransAudio against open-source ASR models and commercial APIs.

摘要: 在针对自动语音识别(ASR)系统的基于传输的攻击中，攻击无法访问目标模型的体系结构和参数。现有的攻击方法大多是在语音助手受限制的语音命令场景中研究的，这使得它们不适用于更一般的ASR相关应用。为了应对这一挑战，我们提出了一种新的具有删除、插入和替换敌意行为的上下文攻击，即TransAudio，它基于所提出的两阶段框架实现了任意词级攻击。为了增强攻击的可转移性，我们进一步引入了音频得分匹配的优化策略来规范训练过程，从而缓解了对手例子对代理模型的过度拟合。大量的实验和分析证明了TransAudio相对于开源ASR模型和商业API的有效性。



## **3. Denoising Autoencoder-based Defensive Distillation as an Adversarial Robustness Algorithm**

基于自动编码去噪的防御蒸馏作为一种对抗健壮性算法 cs.LG

This paper have 4 pages, 3 figures and it is accepted at the Ada User  journal

**SubmitDate**: 2023-03-28    [abs](http://arxiv.org/abs/2303.15901v1) [paper-pdf](http://arxiv.org/pdf/2303.15901v1)

**Authors**: Bakary Badjie, José Cecílio, António Casimiro

**Abstract**: Adversarial attacks significantly threaten the robustness of deep neural networks (DNNs). Despite the multiple defensive methods employed, they are nevertheless vulnerable to poison attacks, where attackers meddle with the initial training data. In order to defend DNNs against such adversarial attacks, this work proposes a novel method that combines the defensive distillation mechanism with a denoising autoencoder (DAE). This technique tries to lower the sensitivity of the distilled model to poison attacks by spotting and reconstructing poisonous adversarial inputs in the training data. We added carefully created adversarial samples to the initial training data to assess the proposed method's performance. Our experimental findings demonstrate that our method successfully identified and reconstructed the poisonous inputs while also considering enhancing the DNN's resilience. The proposed approach provides a potent and robust defense mechanism for DNNs in various applications where data poisoning attacks are a concern. Thus, the defensive distillation technique's limitation posed by poisonous adversarial attacks is overcome.

摘要: 敌意攻击严重威胁了深度神经网络(DNN)的健壮性。尽管采用了多种防御方法，但它们仍然容易受到毒药攻击，攻击者会干预初始训练数据。为了防止DNN受到这种恶意攻击，提出了一种新的方法，该方法将防御蒸馏机制与去噪自动编码器(DAE)相结合。这种技术试图通过在训练数据中发现和重建有毒的对抗性输入来降低提取模型对毒物攻击的敏感度。我们将精心创建的对抗性样本添加到初始训练数据中，以评估所提出的方法的性能。我们的实验结果表明，我们的方法成功地识别和重建了有毒输入，同时还考虑了增强DNN的弹性。所提出的方法为DNN在数据中毒攻击引起关注的各种应用中提供了一种有效和健壮的防御机制。从而克服了防御性蒸馏技术因恶意对抗性攻击而带来的局限性。



## **4. Towards Effective Adversarial Textured 3D Meshes on Physical Face Recognition**

面向物理人脸识别的有效对抗性纹理3D网格 cs.CV

**SubmitDate**: 2023-03-28    [abs](http://arxiv.org/abs/2303.15818v1) [paper-pdf](http://arxiv.org/pdf/2303.15818v1)

**Authors**: Xiao Yang, Chang Liu, Longlong Xu, Yikai Wang, Yinpeng Dong, Ning Chen, Hang Su, Jun Zhu

**Abstract**: Face recognition is a prevailing authentication solution in numerous biometric applications. Physical adversarial attacks, as an important surrogate, can identify the weaknesses of face recognition systems and evaluate their robustness before deployed. However, most existing physical attacks are either detectable readily or ineffective against commercial recognition systems. The goal of this work is to develop a more reliable technique that can carry out an end-to-end evaluation of adversarial robustness for commercial systems. It requires that this technique can simultaneously deceive black-box recognition models and evade defensive mechanisms. To fulfill this, we design adversarial textured 3D meshes (AT3D) with an elaborate topology on a human face, which can be 3D-printed and pasted on the attacker's face to evade the defenses. However, the mesh-based optimization regime calculates gradients in high-dimensional mesh space, and can be trapped into local optima with unsatisfactory transferability. To deviate from the mesh-based space, we propose to perturb the low-dimensional coefficient space based on 3D Morphable Model, which significantly improves black-box transferability meanwhile enjoying faster search efficiency and better visual quality. Extensive experiments in digital and physical scenarios show that our method effectively explores the security vulnerabilities of multiple popular commercial services, including three recognition APIs, four anti-spoofing APIs, two prevailing mobile phones and two automated access control systems.

摘要: 人脸识别是众多生物识别应用中的主流身份验证解决方案。物理对抗攻击作为一种重要的替代手段，可以在部署前识别人脸识别系统的弱点并评估其健壮性。然而，大多数现有的物理攻击要么很容易被检测到，要么对商业识别系统无效。这项工作的目标是开发一种更可靠的技术，可以对商业系统的对手健壮性进行端到端的评估。这就要求该技术能够同时欺骗黑盒识别模型和规避防御机制。为了实现这一点，我们设计了对抗性纹理3D网格(AT3D)，在人脸上具有复杂的拓扑结构，可以3D打印并粘贴在攻击者的脸上以躲避防御。然而，基于网格的优化方法在高维网格空间中计算梯度，容易陷入局部最优，可移植性不理想。为了偏离基于网格的空间，我们提出了基于3D可变形模型的低维系数空间的扰动，在获得更快的搜索效率和更好的视觉质量的同时，显著提高了黑盒的可转移性。在数字和物理场景中的大量实验表明，我们的方法有效地探测了多个流行的商业服务的安全漏洞，包括三个识别API、四个反欺骗API、两个流行的手机和两个自动访问控制系统。



## **5. Transferable Adversarial Attacks on Vision Transformers with Token Gradient Regularization**

基于令牌梯度正则化的视觉变换可转移敌意攻击 cs.CV

CVPR 2023

**SubmitDate**: 2023-03-28    [abs](http://arxiv.org/abs/2303.15754v1) [paper-pdf](http://arxiv.org/pdf/2303.15754v1)

**Authors**: Jianping Zhang, Yizhan Huang, Weibin Wu, Michael R. Lyu

**Abstract**: Vision transformers (ViTs) have been successfully deployed in a variety of computer vision tasks, but they are still vulnerable to adversarial samples. Transfer-based attacks use a local model to generate adversarial samples and directly transfer them to attack a target black-box model. The high efficiency of transfer-based attacks makes it a severe security threat to ViT-based applications. Therefore, it is vital to design effective transfer-based attacks to identify the deficiencies of ViTs beforehand in security-sensitive scenarios. Existing efforts generally focus on regularizing the input gradients to stabilize the updated direction of adversarial samples. However, the variance of the back-propagated gradients in intermediate blocks of ViTs may still be large, which may make the generated adversarial samples focus on some model-specific features and get stuck in poor local optima. To overcome the shortcomings of existing approaches, we propose the Token Gradient Regularization (TGR) method. According to the structural characteristics of ViTs, TGR reduces the variance of the back-propagated gradient in each internal block of ViTs in a token-wise manner and utilizes the regularized gradient to generate adversarial samples. Extensive experiments on attacking both ViTs and CNNs confirm the superiority of our approach. Notably, compared to the state-of-the-art transfer-based attacks, our TGR offers a performance improvement of 8.8% on average.

摘要: 视觉转换器(VITS)已经成功地应用于各种计算机视觉任务中，但它们仍然容易受到对手样本的攻击。基于转移的攻击使用局部模型来生成对抗性样本，并直接转移它们来攻击目标黑盒模型。基于传输的攻击的高效率使其对基于VIT的应用程序构成了严重的安全威胁。因此，设计有效的基于传输的攻击以在安全敏感的场景中预先识别VITS的缺陷是至关重要的。现有的努力一般侧重于使输入梯度正规化，以稳定对抗性样本的最新方向。然而，VITS中间块的反向传播梯度的方差可能仍然很大，这可能会使生成的对抗性样本集中在某些模型特定的特征上，陷入较差的局部最优。为了克服现有方法的不足，我们提出了令牌梯度正则化方法。根据VITS的结构特点，TGR以象征性的方式减小VITS各内部块反向传播梯度的方差，并利用正则化的梯度生成对抗性样本。在攻击VITS和CNN上的大量实验证实了该方法的优越性。值得注意的是，与最先进的基于传输的攻击相比，我们的TGR提供了8.8%的平均性能改进。



## **6. Improving the Transferability of Adversarial Samples by Path-Augmented Method**

利用路径扩展方法提高对抗性样本的可转移性 cs.CV

10 pages + appendix, CVPR 2023

**SubmitDate**: 2023-03-28    [abs](http://arxiv.org/abs/2303.15735v1) [paper-pdf](http://arxiv.org/pdf/2303.15735v1)

**Authors**: Jianping Zhang, Jen-tse Huang, Wenxuan Wang, Yichen Li, Weibin Wu, Xiaosen Wang, Yuxin Su, Michael R. Lyu

**Abstract**: Deep neural networks have achieved unprecedented success on diverse vision tasks. However, they are vulnerable to adversarial noise that is imperceptible to humans. This phenomenon negatively affects their deployment in real-world scenarios, especially security-related ones. To evaluate the robustness of a target model in practice, transfer-based attacks craft adversarial samples with a local model and have attracted increasing attention from researchers due to their high efficiency. The state-of-the-art transfer-based attacks are generally based on data augmentation, which typically augments multiple training images from a linear path when learning adversarial samples. However, such methods selected the image augmentation path heuristically and may augment images that are semantics-inconsistent with the target images, which harms the transferability of the generated adversarial samples. To overcome the pitfall, we propose the Path-Augmented Method (PAM). Specifically, PAM first constructs a candidate augmentation path pool. It then settles the employed augmentation paths during adversarial sample generation with greedy search. Furthermore, to avoid augmenting semantics-inconsistent images, we train a Semantics Predictor (SP) to constrain the length of the augmentation path. Extensive experiments confirm that PAM can achieve an improvement of over 4.8% on average compared with the state-of-the-art baselines in terms of the attack success rates.

摘要: 深度神经网络在不同的视觉任务上取得了前所未有的成功。然而，它们很容易受到人类察觉不到的对抗性噪音的影响。这种现象对它们在现实世界场景中的部署产生了负面影响，特别是与安全相关的场景。为了在实际应用中评估目标模型的稳健性，基于转移的攻击利用局部模型来构造对手样本，由于其高效性而受到越来越多的研究人员的关注。最先进的基于传输的攻击通常基于数据增强，当学习对抗性样本时，数据增强通常从线性路径增加多个训练图像。然而，这些方法对图像增强路径的选择是启发式的，可能会对与目标图像语义不一致的图像进行增强，从而损害了生成的对抗性样本的可转移性。为了克服这一缺陷，我们提出了路径扩展方法(PAM)。具体地，PAM首先构建候选扩展路径池。然后利用贪婪搜索解决对抗性样本生成过程中所采用的扩充路径。此外，为了避免增强语义不一致的图像，我们训练了一个语义预测器(SP)来约束增强路径的长度。广泛的实验证实，与最先进的基线相比，PAM在攻击成功率方面平均可以提高4.8%以上。



## **7. EMShepherd: Detecting Adversarial Samples via Side-channel Leakage**

EMShepherd：通过旁路泄漏检测敌方样本 cs.CR

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.15571v1) [paper-pdf](http://arxiv.org/pdf/2303.15571v1)

**Authors**: Ruyi Ding, Cheng Gongye, Siyue Wang, Aidong Ding, Yunsi Fei

**Abstract**: Deep Neural Networks (DNN) are vulnerable to adversarial perturbations-small changes crafted deliberately on the input to mislead the model for wrong predictions. Adversarial attacks have disastrous consequences for deep learning-empowered critical applications. Existing defense and detection techniques both require extensive knowledge of the model, testing inputs, and even execution details. They are not viable for general deep learning implementations where the model internal is unknown, a common 'black-box' scenario for model users. Inspired by the fact that electromagnetic (EM) emanations of a model inference are dependent on both operations and data and may contain footprints of different input classes, we propose a framework, EMShepherd, to capture EM traces of model execution, perform processing on traces and exploit them for adversarial detection. Only benign samples and their EM traces are used to train the adversarial detector: a set of EM classifiers and class-specific unsupervised anomaly detectors. When the victim model system is under attack by an adversarial example, the model execution will be different from executions for the known classes, and the EM trace will be different. We demonstrate that our air-gapped EMShepherd can effectively detect different adversarial attacks on a commonly used FPGA deep learning accelerator for both Fashion MNIST and CIFAR-10 datasets. It achieves a 100% detection rate on most types of adversarial samples, which is comparable to the state-of-the-art 'white-box' software-based detectors.

摘要: 深度神经网络(DNN)很容易受到对抗性扰动--故意在输入上精心设计的小变化，以误导模型进行错误预测。对抗性攻击会给深度学习支持的关键应用程序带来灾难性的后果。现有的防御和检测技术都需要对模型、测试输入甚至执行细节有广泛的了解。它们对于模型内部未知的一般深度学习实现是不可行的，对于模型用户来说，这是一个常见的“黑箱”场景。受模型推理的电磁辐射依赖于操作和数据并且可能包含不同输入类的足迹这一事实的启发，我们提出了一个框架EMShepherd，用于捕获模型执行的电磁跟踪，对跟踪进行处理，并利用它们进行对抗性检测。只有良性样本及其EM踪迹被用于训练对抗性检测器：一组EM分类器和特定类别的非监督异常检测器。当受害者模型系统受到敌意示例攻击时，模型执行将与已知类的执行不同，EM跟踪也将不同。我们证明了我们的空隙EMShepherd可以有效地检测到针对Fashion MNIST和CIFAR-10数据集的常用FPGA深度学习加速器上的不同对手攻击。它在大多数类型的对手样本上实现了100%的检测率，这可以与最先进的基于软件的白盒检测器相媲美。



## **8. Mask and Restore: Blind Backdoor Defense at Test Time with Masked Autoencoder**

掩码和恢复：使用掩码自动编码器在测试时进行盲后门保护 cs.LG

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.15564v1) [paper-pdf](http://arxiv.org/pdf/2303.15564v1)

**Authors**: Tao Sun, Lu Pang, Chao Chen, Haibin Ling

**Abstract**: Deep neural networks are vulnerable to backdoor attacks, where an adversary maliciously manipulates the model behavior through overlaying images with special triggers. Existing backdoor defense methods often require accessing a few validation data and model parameters, which are impractical in many real-world applications, e.g., when the model is provided as a cloud service. In this paper, we address the practical task of blind backdoor defense at test time, in particular for black-box models. The true label of every test image needs to be recovered on the fly from the hard label predictions of a suspicious model. The heuristic trigger search in image space, however, is not scalable to complex triggers or high image resolution. We circumvent such barrier by leveraging generic image generation models, and propose a framework of Blind Defense with Masked AutoEncoder (BDMAE). It uses the image structural similarity and label consistency between the test image and MAE restorations to detect possible triggers. The detection result is refined by considering the topology of triggers. We obtain a purified test image from restorations for making prediction. Our approach is blind to the model architectures, trigger patterns or image benignity. Extensive experiments on multiple datasets with different backdoor attacks validate its effectiveness and generalizability. Code is available at https://github.com/tsun/BDMAE.

摘要: 深度神经网络很容易受到后门攻击，在后门攻击中，对手通过使用特殊触发器覆盖图像来恶意操纵模型行为。现有的后门防御方法通常需要访问一些验证数据和模型参数，这在许多真实世界的应用中是不切实际的，例如当模型作为云服务提供时。在本文中，我们讨论了测试时的盲后门防御的实际任务，特别是对于黑盒模型。每个测试图像的真实标签都需要从可疑模型的硬标签预测中动态恢复。然而，图像空间中的启发式触发器搜索不能扩展到复杂的触发器或高图像分辨率。我们通过利用通用的图像生成模型来绕过这一障碍，并提出了一种基于掩蔽自动编码器的盲防框架(BDMAE)。它使用测试图像和MAE恢复图像之间的图像结构相似性和标签一致性来检测可能的触发因素。通过考虑触发器的拓扑结构，对检测结果进行了改进。我们从复原中获得一个净化的测试图像来进行预测。我们的方法对模型架构、触发模式或图像亲和性是视而不见的。在具有不同后门攻击的多个数据集上的大量实验验证了该算法的有效性和泛化能力。代码可在https://github.com/tsun/BDMAE.上找到



## **9. Intel TDX Demystified: A Top-Down Approach**

英特尔TDX揭秘：自上而下的方法 cs.CR

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.15540v1) [paper-pdf](http://arxiv.org/pdf/2303.15540v1)

**Authors**: Pau-Chen Cheng, Wojciech Ozga, Enriquillo Valdez, Salman Ahmed, Zhongshu Gu, Hani Jamjoom, Hubertus Franke, James Bottomley

**Abstract**: Intel Trust Domain Extensions (TDX) is a new architectural extension in the 4th Generation Intel Xeon Scalable Processor that supports confidential computing. TDX allows the deployment of virtual machines in the Secure-Arbitration Mode (SEAM) with encrypted CPU state and memory, integrity protection, and remote attestation. TDX aims to enforce hardware-assisted isolation for virtual machines and minimize the attack surface exposed to host platforms, which are considered to be untrustworthy or adversarial in the confidential computing's new threat model. TDX can be leveraged by regulated industries or sensitive data holders to outsource their computations and data with end-to-end protection in public cloud infrastructure.   This paper aims to provide a comprehensive understanding of TDX to potential adopters, domain experts, and security researchers looking to leverage the technology for their own purposes. We adopt a top-down approach, starting with high-level security principles and moving to low-level technical details of TDX. Our analysis is based on publicly available documentation and source code, offering insights from security researchers outside of Intel.

摘要: 英特尔信任域扩展(TDX)是支持机密计算的第4代英特尔至强可扩展处理器中的新架构扩展。TDX允许在安全仲裁模式(SEAM)下部署具有加密的CPU状态和内存、完整性保护和远程证明的虚拟机。TDX旨在加强对虚拟机的硬件辅助隔离，并将暴露在主机平台上的攻击面降至最低，在机密计算的新威胁模型中，主机平台被认为是不可信任或对抗性的。受监管行业或敏感数据持有者可以利用TDX在公共云基础设施中提供端到端保护，以外包其计算和数据。本文旨在为潜在的采用者、领域专家和希望利用TDX技术实现其自身目的的安全研究人员提供对TDX的全面了解。我们采用自上而下的方法，从高级别的安全原则开始，转向TDX的低级别技术细节。我们的分析基于公开的文档和源代码，提供了英特尔以外的安全研究人员的见解。



## **10. Classifier Robustness Enhancement Via Test-Time Transformation**

利用测试时间变换增强分类器稳健性 cs.CV

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.15409v1) [paper-pdf](http://arxiv.org/pdf/2303.15409v1)

**Authors**: Tsachi Blau, Roy Ganz, Chaim Baskin, Michael Elad, Alex Bronstein

**Abstract**: It has been recently discovered that adversarially trained classifiers exhibit an intriguing property, referred to as perceptually aligned gradients (PAG). PAG implies that the gradients of such classifiers possess a meaningful structure, aligned with human perception. Adversarial training is currently the best-known way to achieve classification robustness under adversarial attacks. The PAG property, however, has yet to be leveraged for further improving classifier robustness. In this work, we introduce Classifier Robustness Enhancement Via Test-Time Transformation (TETRA) -- a novel defense method that utilizes PAG, enhancing the performance of trained robust classifiers. Our method operates in two phases. First, it modifies the input image via a designated targeted adversarial attack into each of the dataset's classes. Then, it classifies the input image based on the distance to each of the modified instances, with the assumption that the shortest distance relates to the true class. We show that the proposed method achieves state-of-the-art results and validate our claim through extensive experiments on a variety of defense methods, classifier architectures, and datasets. We also empirically demonstrate that TETRA can boost the accuracy of any differentiable adversarial training classifier across a variety of attacks, including ones unseen at training. Specifically, applying TETRA leads to substantial improvement of up to $+23\%$, $+20\%$, and $+26\%$ on CIFAR10, CIFAR100, and ImageNet, respectively.

摘要: 最近发现，对抗性训练的分类器表现出一种有趣的特性，称为感知对齐梯度(PAG)。PAG暗示，这种量词的梯度具有一种有意义的结构，与人类的感知一致。对抗性训练是目前已知的在对抗性攻击下实现分类稳健性的最好方法。然而，PAG属性还有待于进一步提高分类器的健壮性。在这项工作中，我们引入了通过测试时间转换的分类器健壮性增强(TETRA)--一种利用PAG的新的防御方法，提高了训练的健壮分类器的性能。我们的方法分两个阶段进行。首先，它通过指定的目标对抗性攻击将输入图像修改为数据集的每个类。然后，在假设最短距离与真实类别相关的情况下，基于到每个修改实例的距离对输入图像进行分类。我们通过在各种防御方法、分类器架构和数据集上的大量实验，证明了所提出的方法取得了最先进的结果，并验证了我们的主张。我们还通过实验证明，TETRA可以提高任何可区分的对抗性训练分类器在各种攻击中的准确性，包括在训练中看不到的攻击。具体地说，应用TETRA后，CIFAR10、CIFAR100和ImageNet的性能分别提高了23美元、20美元和26美元。



## **11. Learning the Unlearnable: Adversarial Augmentations Suppress Unlearnable Example Attacks**

学习无法学习的：对抗性增强抑制无法学习的示例攻击 cs.LG

UEraser introduces adversarial augmentations to suppress unlearnable  example attacks and outperforms current defenses

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.15127v1) [paper-pdf](http://arxiv.org/pdf/2303.15127v1)

**Authors**: Tianrui Qin, Xitong Gao, Juanjuan Zhao, Kejiang Ye, Cheng-Zhong Xu

**Abstract**: Unlearnable example attacks are data poisoning techniques that can be used to safeguard public data against unauthorized use for training deep learning models. These methods add stealthy perturbations to the original image, thereby making it difficult for deep learning models to learn from these training data effectively. Current research suggests that adversarial training can, to a certain degree, mitigate the impact of unlearnable example attacks, while common data augmentation methods are not effective against such poisons. Adversarial training, however, demands considerable computational resources and can result in non-trivial accuracy loss. In this paper, we introduce the UEraser method, which outperforms current defenses against different types of state-of-the-art unlearnable example attacks through a combination of effective data augmentation policies and loss-maximizing adversarial augmentations. In stark contrast to the current SOTA adversarial training methods, UEraser uses adversarial augmentations, which extends beyond the confines of $ \ell_p $ perturbation budget assumed by current unlearning attacks and defenses. It also helps to improve the model's generalization ability, thus protecting against accuracy loss. UEraser wipes out the unlearning effect with error-maximizing data augmentations, thus restoring trained model accuracies. Interestingly, UEraser-Lite, a fast variant without adversarial augmentations, is also highly effective in preserving clean accuracies. On challenging unlearnable CIFAR-10, CIFAR-100, SVHN, and ImageNet-subset datasets produced with various attacks, it achieves results that are comparable to those obtained during clean training. We also demonstrate its efficacy against possible adaptive attacks. Our code is open source and available to the deep learning community: https://github.com/lafeat/ueraser.

摘要: 无法学习的示例攻击是一种数据中毒技术，可用于保护公共数据免受未经授权的用于训练深度学习模型的使用。这些方法给原始图像增加了隐蔽的扰动，从而使得深度学习模型很难从这些训练数据中有效地学习。目前的研究表明，对抗性训练可以在一定程度上缓解不可学习的例子攻击的影响，而常用的数据增强方法对此类毒药并不有效。然而，对抗性训练需要相当大的计算资源，并且可能导致相当大的精度损失。在本文中，我们介绍了UEraser方法，它通过有效的数据增强策略和损失最大化的对手增强相结合，对不同类型的不可学习示例攻击的防御性能优于现有的防御方法。与目前的SOTA对抗性训练方法形成鲜明对比的是，UEraser使用对抗性增强，超出了当前遗忘攻击和防御假设的$\ell_p$扰动预算的范围。它还有助于提高模型的泛化能力，从而防止精度损失。UEraser通过误差最大化的数据增加消除了遗忘效应，从而恢复了训练的模型精度。有趣的是，UEraser-Lite，一个没有对抗性增强的快速变体，在保持干净的准确性方面也是非常有效的。在挑战各种攻击产生的难以学习的CIFAR-10、CIFAR-100、SVHN和ImageNet-Subset数据集上，它取得了与干净训练期间相当的结果。我们还展示了它对可能的自适应攻击的有效性。我们的代码是开源的，可供深度学习社区使用：https://github.com/lafeat/ueraser.



## **12. Among Us: Adversarially Robust Collaborative Perception by Consensus**

在我们中间：基于共识的相反的强健协作感知 cs.RO

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.09495v2) [paper-pdf](http://arxiv.org/pdf/2303.09495v2)

**Authors**: Yiming Li, Qi Fang, Jiamu Bai, Siheng Chen, Felix Juefei-Xu, Chen Feng

**Abstract**: Multiple robots could perceive a scene (e.g., detect objects) collaboratively better than individuals, although easily suffer from adversarial attacks when using deep learning. This could be addressed by the adversarial defense, but its training requires the often-unknown attacking mechanism. Differently, we propose ROBOSAC, a novel sampling-based defense strategy generalizable to unseen attackers. Our key idea is that collaborative perception should lead to consensus rather than dissensus in results compared to individual perception. This leads to our hypothesize-and-verify framework: perception results with and without collaboration from a random subset of teammates are compared until reaching a consensus. In such a framework, more teammates in the sampled subset often entail better perception performance but require longer sampling time to reject potential attackers. Thus, we derive how many sampling trials are needed to ensure the desired size of an attacker-free subset, or equivalently, the maximum size of such a subset that we can successfully sample within a given number of trials. We validate our method on the task of collaborative 3D object detection in autonomous driving scenarios.

摘要: 多个机器人可以比个人更好地协作感知场景(例如，检测对象)，尽管在使用深度学习时很容易受到对抗性攻击。这可以通过对抗性防守来解决，但它的训练需要往往未知的攻击机制。不同的是，我们提出了ROBOSAC，一种新的基于采样的防御策略，可以推广到看不见的攻击者。我们的关键思想是，与个人感知相比，合作感知应该在结果中导致共识，而不是分歧。这导致了我们的假设和验证框架：对随机的队友子集进行协作和不协作的感知结果进行比较，直到达成共识。在这样的框架中，采样子集中更多的队友通常会带来更好的感知性能，但需要更长的采样时间来拒绝潜在的攻击者。因此，我们推导出需要多少次抽样试验才能确保没有攻击者的子集的期望大小，或者等价地，在给定的试验次数内可以成功抽样的子集的最大大小。我们在自主驾驶场景下的协同3D目标检测任务中验证了我们的方法。



## **13. Identifying Adversarially Attackable and Robust Samples**

识别恶意攻击和健壮样本 cs.LG

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2301.12896v2) [paper-pdf](http://arxiv.org/pdf/2301.12896v2)

**Authors**: Vyas Raina, Mark Gales

**Abstract**: Adversarial attacks insert small, imperceptible perturbations to input samples that cause large, undesired changes to the output of deep learning models. Despite extensive research on generating adversarial attacks and building defense systems, there has been limited research on understanding adversarial attacks from an input-data perspective. This work introduces the notion of sample attackability, where we aim to identify samples that are most susceptible to adversarial attacks (attackable samples) and conversely also identify the least susceptible samples (robust samples). We propose a deep-learning-based method to detect the adversarially attackable and robust samples in an unseen dataset for an unseen target model. Experiments on standard image classification datasets enables us to assess the portability of the deep attackability detector across a range of architectures. We find that the deep attackability detector performs better than simple model uncertainty-based measures for identifying the attackable/robust samples. This suggests that uncertainty is an inadequate proxy for measuring sample distance to a decision boundary. In addition to better understanding adversarial attack theory, it is found that the ability to identify the adversarially attackable and robust samples has implications for improving the efficiency of sample-selection tasks, e.g. active learning in augmentation for adversarial training.

摘要: 对抗性攻击在输入样本中插入微小的、不可察觉的扰动，从而导致深度学习模型的输出发生巨大的、不希望看到的变化。尽管对生成对抗性攻击和建立防御系统进行了广泛的研究，但从输入数据的角度理解对抗性攻击的研究有限。这项工作引入了样本可攻击性的概念，其中我们的目标是识别最容易受到对手攻击的样本(可攻击样本)，反过来也识别最不敏感的样本(稳健样本)。我们提出了一种基于深度学习的方法来检测不可见目标模型中不可见数据集中的可攻击样本和稳健样本。在标准图像分类数据集上的实验使我们能够评估深度可攻击性检测器在一系列体系结构中的可移植性。我们发现，深度可攻击性检测器在识别可攻击/稳健样本方面比基于简单模型不确定性的度量方法表现得更好。这表明，不确定性不足以衡量样本到决策边界的距离。除了更好地理解敌意攻击理论外，研究发现，识别敌意可攻击和健壮样本的能力对于提高样本选择任务的效率也有意义，例如，在对抗性训练的增强中的主动学习。



## **14. Improving the Transferability of Adversarial Examples via Direction Tuning**

通过方向调整提高对抗性例句的可转移性 cs.CV

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.15109v1) [paper-pdf](http://arxiv.org/pdf/2303.15109v1)

**Authors**: Xiangyuan Yang, Jie Lin, Hanlin Zhang, Xinyu Yang, Peng Zhao

**Abstract**: In the transfer-based adversarial attacks, adversarial examples are only generated by the surrogate models and achieve effective perturbation in the victim models. Although considerable efforts have been developed on improving the transferability of adversarial examples generated by transfer-based adversarial attacks, our investigation found that, the big deviation between the actual and steepest update directions of the current transfer-based adversarial attacks is caused by the large update step length, resulting in the generated adversarial examples can not converge well. However, directly reducing the update step length will lead to serious update oscillation so that the generated adversarial examples also can not achieve great transferability to the victim models. To address these issues, a novel transfer-based attack, namely direction tuning attack, is proposed to not only decrease the update deviation in the large step length, but also mitigate the update oscillation in the small sampling step length, thereby making the generated adversarial examples converge well to achieve great transferability on victim models. In addition, a network pruning method is proposed to smooth the decision boundary, thereby further decreasing the update oscillation and enhancing the transferability of the generated adversarial examples. The experiment results on ImageNet demonstrate that the average attack success rate (ASR) of the adversarial examples generated by our method can be improved from 87.9\% to 94.5\% on five victim models without defenses, and from 69.1\% to 76.2\% on eight advanced defense methods, in comparison with that of latest gradient-based attacks.

摘要: 在基于迁移的对抗性攻击中，对抗性实例仅由代理模型生成，并在受害者模型中实现有效的扰动。虽然在提高基于转移的对抗性攻击生成的对抗性样本的可转移性方面已经做了大量的工作，但我们的调查发现，当前基于转移的对抗性攻击的实际更新方向与最陡的更新方向之间存在较大的偏差，这是由于更新步长较大，导致生成的对抗性样本不能很好地收敛。但是，直接缩短更新步长会导致严重的更新振荡，使得生成的对抗性实例也不能很好地移植到受害者模型中。针对这些问题，提出了一种新的基于转移的攻击方法，即方向调整攻击，它不仅可以减小大步长时的更新偏差，而且可以缓解小采样步长时的更新振荡，从而使生成的敌意样本能够很好地收敛到受害者模型上，达到很好的可转移性。此外，还提出了一种网络剪枝方法来平滑决策边界，从而进一步减小更新振荡，增强生成的对抗性实例的可转移性。在ImageNet上的实验结果表明，与最新的基于梯度的攻击方法相比，该方法生成的攻击实例的平均攻击成功率(ASR)在5个无防御的受害者模型上可以从87.9提高到94.5，在8种高级防御方法上从69.1提高到76.2。



## **15. Improved Adversarial Training Through Adaptive Instance-wise Loss Smoothing**

通过自适应实例损失平滑改进对手训练 cs.CV

12 pages, work in submission

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.14077v2) [paper-pdf](http://arxiv.org/pdf/2303.14077v2)

**Authors**: Lin Li, Michael Spratling

**Abstract**: Deep neural networks can be easily fooled into making incorrect predictions through corruption of the input by adversarial perturbations: human-imperceptible artificial noise. So far adversarial training has been the most successful defense against such adversarial attacks. This work focuses on improving adversarial training to boost adversarial robustness. We first analyze, from an instance-wise perspective, how adversarial vulnerability evolves during adversarial training. We find that during training an overall reduction of adversarial loss is achieved by sacrificing a considerable proportion of training samples to be more vulnerable to adversarial attack, which results in an uneven distribution of adversarial vulnerability among data. Such "uneven vulnerability", is prevalent across several popular robust training methods and, more importantly, relates to overfitting in adversarial training. Motivated by this observation, we propose a new adversarial training method: Instance-adaptive Smoothness Enhanced Adversarial Training (ISEAT). It jointly smooths both input and weight loss landscapes in an adaptive, instance-specific, way to enhance robustness more for those samples with higher adversarial vulnerability. Extensive experiments demonstrate the superiority of our method over existing defense methods. Noticeably, our method, when combined with the latest data augmentation and semi-supervised learning techniques, achieves state-of-the-art robustness against $\ell_{\infty}$-norm constrained attacks on CIFAR10 of 59.32% for Wide ResNet34-10 without extra data, and 61.55% for Wide ResNet28-10 with extra data. Code is available at https://github.com/TreeLLi/Instance-adaptive-Smoothness-Enhanced-AT.

摘要: 深层神经网络很容易被欺骗，通过破坏对抗性扰动的输入做出错误的预测：人类无法察觉的人工噪声。到目前为止，对抗性训练一直是对这种对抗性攻击最成功的防御。这项工作的重点是改进对手训练，以提高对手的稳健性。我们首先从实例的角度分析在对抗性训练过程中对抗性脆弱性是如何演变的。我们发现，在训练过程中，通过牺牲相当大比例的训练样本来更容易受到对手攻击，从而总体上减少了对手的损失，这导致了对手脆弱性在数据中的不均匀分布。这种“脆弱性参差不齐”普遍存在于几种流行的健壮训练方法中，更重要的是与对抗性训练中的过度适应有关。基于这一观察结果，我们提出了一种新的对抗性训练方法：实例自适应平滑增强对抗性训练(ISEAT)。它以一种自适应的、特定于实例的方式联合平滑输入和减肥环境，以增强那些具有更高对手脆弱性的样本的健壮性。大量的实验证明了该方法相对于现有防御方法的优越性。值得注意的是，当我们的方法与最新的数据增强和半监督学习技术相结合时，对于针对CIFAR10的$-范数约束攻击，对于没有额外数据的宽ResNet34-10达到了59.32%，对于具有额外数据的宽ResNet28-10达到了61.55%。代码可在https://github.com/TreeLLi/Instance-adaptive-Smoothness-Enhanced-AT.上找到



## **16. Diffusion Denoised Smoothing for Certified and Adversarial Robust Out-Of-Distribution Detection**

基于扩散去噪平滑的认证和对抗稳健失配检测 cs.LG

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.14961v1) [paper-pdf](http://arxiv.org/pdf/2303.14961v1)

**Authors**: Nicola Franco, Daniel Korth, Jeanette Miriam Lorenz, Karsten Roscher, Stephan Guennemann

**Abstract**: As the use of machine learning continues to expand, the importance of ensuring its safety cannot be overstated. A key concern in this regard is the ability to identify whether a given sample is from the training distribution, or is an "Out-Of-Distribution" (OOD) sample. In addition, adversaries can manipulate OOD samples in ways that lead a classifier to make a confident prediction. In this study, we present a novel approach for certifying the robustness of OOD detection within a $\ell_2$-norm around the input, regardless of network architecture and without the need for specific components or additional training. Further, we improve current techniques for detecting adversarial attacks on OOD samples, while providing high levels of certified and adversarial robustness on in-distribution samples. The average of all OOD detection metrics on CIFAR10/100 shows an increase of $\sim 13 \% / 5\%$ relative to previous approaches.

摘要: 随着机器学习的使用不断扩大，确保其安全性的重要性怎么强调都不为过。这方面的一个关键问题是能否识别给定样本是来自训练分布，还是“超出分布”(OOD)样本。此外，攻击者还可以操纵OOD样本，从而使分类器做出可靠的预测。在这项研究中，我们提出了一种新的方法来证明OOD检测的稳健性在输入周围的$\ell_2$-范数内，而与网络体系结构无关，并且不需要特定的组件或额外的训练。此外，我们改进了当前检测OOD样本上的对抗性攻击的技术，同时在分发内样本上提供了高水平的认证和对抗性健壮性。与以前的方法相比，CIFAR10/100上所有OOD检测指标的平均值增加了$\sim 13/5$。



## **17. CAT:Collaborative Adversarial Training**

CAT：协同对抗训练 cs.CV

Tech report

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.14922v1) [paper-pdf](http://arxiv.org/pdf/2303.14922v1)

**Authors**: Xingbin Liu, Huafeng Kuang, Xianming Lin, Yongjian Wu, Rongrong Ji

**Abstract**: Adversarial training can improve the robustness of neural networks. Previous methods focus on a single adversarial training strategy and do not consider the model property trained by different strategies. By revisiting the previous methods, we find different adversarial training methods have distinct robustness for sample instances. For example, a sample instance can be correctly classified by a model trained using standard adversarial training (AT) but not by a model trained using TRADES, and vice versa. Based on this observation, we propose a collaborative adversarial training framework to improve the robustness of neural networks. Specifically, we use different adversarial training methods to train robust models and let models interact with their knowledge during the training process. Collaborative Adversarial Training (CAT) can improve both robustness and accuracy. Extensive experiments on various networks and datasets validate the effectiveness of our method. CAT achieves state-of-the-art adversarial robustness without using any additional data on CIFAR-10 under the Auto-Attack benchmark. Code is available at https://github.com/liuxingbin/CAT.

摘要: 对抗性训练可以提高神经网络的健壮性。以往的方法侧重于单一的对抗性训练策略，没有考虑不同策略训练的模型性质。通过回顾以往的方法，我们发现不同的对抗性训练方法对样本实例具有不同的稳健性。例如，样本实例可以通过使用标准对手训练(AT)训练的模型来正确分类，但不可以通过使用交易训练的模型来正确分类，反之亦然。基于这一观察结果，我们提出了一个协同对抗训练框架来提高神经网络的健壮性。具体地说，我们使用不同的对抗性训练方法来训练健壮的模型，并在训练过程中让模型与他们的知识进行交互。协同对抗训练(CAT)可以同时提高鲁棒性和准确性。在不同网络和数据集上的大量实验验证了该方法的有效性。在自动攻击基准下，CAT无需使用CIFAR-10上的任何额外数据即可实现最先进的对手健壮性。代码可在https://github.com/liuxingbin/CAT.上找到



## **18. Efficient Robustness Assessment via Adversarial Spatial-Temporal Focus on Videos**

对抗性时空聚焦视频的高效稳健性评估 cs.CV

accepted by TPAMI2023

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2301.00896v2) [paper-pdf](http://arxiv.org/pdf/2301.00896v2)

**Authors**: Wei Xingxing, Wang Songping, Yan Huanqian

**Abstract**: Adversarial robustness assessment for video recognition models has raised concerns owing to their wide applications on safety-critical tasks. Compared with images, videos have much high dimension, which brings huge computational costs when generating adversarial videos. This is especially serious for the query-based black-box attacks where gradient estimation for the threat models is usually utilized, and high dimensions will lead to a large number of queries. To mitigate this issue, we propose to simultaneously eliminate the temporal and spatial redundancy within the video to achieve an effective and efficient gradient estimation on the reduced searching space, and thus query number could decrease. To implement this idea, we design the novel Adversarial spatial-temporal Focus (AstFocus) attack on videos, which performs attacks on the simultaneously focused key frames and key regions from the inter-frames and intra-frames in the video. AstFocus attack is based on the cooperative Multi-Agent Reinforcement Learning (MARL) framework. One agent is responsible for selecting key frames, and another agent is responsible for selecting key regions. These two agents are jointly trained by the common rewards received from the black-box threat models to perform a cooperative prediction. By continuously querying, the reduced searching space composed of key frames and key regions is becoming precise, and the whole query number becomes less than that on the original video. Extensive experiments on four mainstream video recognition models and three widely used action recognition datasets demonstrate that the proposed AstFocus attack outperforms the SOTA methods, which is prevenient in fooling rate, query number, time, and perturbation magnitude at the same.

摘要: 视频识别模型的对抗性健壮性评估由于其在安全关键任务中的广泛应用而引起了人们的关注。与图像相比，视频的维度要高得多，这在生成对抗性视频时带来了巨大的计算代价。这对于基于查询的黑盒攻击尤为严重，这种攻击通常使用威胁模型的梯度估计，高维将导致大量的查询。为了缓解这一问题，我们提出同时消除视频中的时间和空间冗余，在缩减的搜索空间上实现有效和高效的梯度估计，从而减少查询数量。为了实现这一思想，我们设计了一种新颖的对抗性时空聚焦(AstFocus)攻击，它从视频的帧间和帧内对同时聚焦的关键帧和关键区域进行攻击。AstFocus攻击基于协作多智能体强化学习(MAIL)框架。一个代理负责选择关键帧，另一个代理负责选择关键区域。这两个代理通过从黑盒威胁模型获得的共同奖励来联合训练，以执行合作预测。通过连续查询，缩小了由关键帧和关键区域组成的搜索空间，变得更加精确，整个查询次数比原始视频上的少。在四个主流视频识别模型和三个广泛使用的动作识别数据集上的大量实验表明，AstFocus攻击的性能优于SOTA方法，后者在愚弄率、查询次数、时间和扰动幅度方面都优于SOTA方法。



## **19. Don't be a Victim During a Pandemic! Analysing Security and Privacy Threats in Twitter During COVID-19**

不要在大流行期间成为受害者！新冠肺炎期间推特面临的安全和隐私威胁分析 cs.CR

Paper has been accepted for publication in IEEE Access. Currently  available on IEEE ACCESS early access (see DOI)

**SubmitDate**: 2023-03-26    [abs](http://arxiv.org/abs/2202.10543v2) [paper-pdf](http://arxiv.org/pdf/2202.10543v2)

**Authors**: Bibhas Sharma, Ishan Karunanayake, Rahat Masood, Muhammad Ikram

**Abstract**: There has been a huge spike in the usage of social media platforms during the COVID-19 lockdowns. These lockdown periods have resulted in a set of new cybercrimes, thereby allowing attackers to victimise social media users with a range of threats. This paper performs a large-scale study to investigate the impact of a pandemic and the lockdown periods on the security and privacy of social media users. We analyse 10.6 Million COVID-related tweets from 533 days of data crawling and investigate users' security and privacy behaviour in three different periods (i.e., before, during, and after the lockdown). Our study shows that users unintentionally share more personal identifiable information when writing about the pandemic situation (e.g., sharing nearby coronavirus testing locations) in their tweets. The privacy risk reaches 100% if a user posts three or more sensitive tweets about the pandemic. We investigate the number of suspicious domains shared on social media during different phases of the pandemic. Our analysis reveals an increase in the number of suspicious domains during the lockdown compared to other lockdown phases. We observe that IT, Search Engines, and Businesses are the top three categories that contain suspicious domains. Our analysis reveals that adversaries' strategies to instigate malicious activities change with the country's pandemic situation.

摘要: 在新冠肺炎被封锁期间，社交媒体平台的使用量大幅上升。这些封锁期导致了一系列新的网络犯罪，从而使攻击者能够通过一系列威胁来攻击社交媒体用户。本文进行了一项大规模的研究，以调查大流行和封锁期对社交媒体用户安全和隐私的影响。我们从533天的数据爬行中分析了1060万条与CoVID相关的推文，并调查了用户在三个不同时期(即封锁前、封锁期间和封锁后)的安全和隐私行为。我们的研究表明，当用户在他们的推特上写关于大流行情况的信息(例如，分享附近的冠状病毒检测地点)时，无意中分享了更多的个人可识别信息。如果用户发布三条或三条以上有关疫情的敏感推文，隐私风险将达到100%。我们调查了在疫情不同阶段在社交媒体上分享的可疑域名的数量。我们的分析显示，与其他锁定阶段相比，锁定期间可疑域名的数量有所增加。我们观察到，IT、搜索引擎和企业是包含可疑域名的前三大类别。我们的分析显示，对手煽动恶意活动的战略会随着该国疫情的变化而变化。



## **20. Distributionally Robust Multiclass Classification and Applications in Deep Image Classifiers**

分布稳健多类分类及其在深度图像分类器中的应用 stat.ML

9 pages; Previously this version appeared as arXiv:2210.08198 which  was submitted as a new work by accident

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2109.12772v2) [paper-pdf](http://arxiv.org/pdf/2109.12772v2)

**Authors**: Ruidi Chen, Boran Hao, Ioannis Paschalidis

**Abstract**: We develop a Distributionally Robust Optimization (DRO) formulation for Multiclass Logistic Regression (MLR), which could tolerate data contaminated by outliers. The DRO framework uses a probabilistic ambiguity set defined as a ball of distributions that are close to the empirical distribution of the training set in the sense of the Wasserstein metric. We relax the DRO formulation into a regularized learning problem whose regularizer is a norm of the coefficient matrix. We establish out-of-sample performance guarantees for the solutions to our model, offering insights on the role of the regularizer in controlling the prediction error. We apply the proposed method in rendering deep Vision Transformer (ViT)-based image classifiers robust to random and adversarial attacks. Specifically, using the MNIST and CIFAR-10 datasets, we demonstrate reductions in test error rate by up to 83.5% and loss by up to 91.3% compared with baseline methods, by adopting a novel random training method.

摘要: 针对多分类Logistic回归(MLR)模型，提出了一种分布式稳健优化(DRO)方法，该方法能够容忍数据中的离群点污染。DRO框架使用概率模糊集，该模糊集被定义为在Wasserstein度量意义上接近训练集的经验分布的分布球。我们将DRO公式松弛为一个正则化学习问题，其正则化子是系数矩阵的范数。我们为我们的模型的解建立了样本外性能保证，提供了关于正则化在控制预测误差中的作用的见解。我们将该方法应用于基于深度视觉转换器(VIT)的图像分类器对随机攻击和敌意攻击的稳健性。具体地说，使用MNIST和CIFAR-10数据集，我们证明了通过采用一种新的随机训练方法，与基线方法相比，测试错误率降低了83.5%，损失降低了91.3%。



## **21. Distributionally Robust Multiclass Classification and Applications in Deep Image Classifiers**

分布稳健多类分类及其在深度图像分类器中的应用 cs.CV

This work was intended as a replacement of arXiv:2109.12772 and any  subsequent updates will appear there

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2210.08198v2) [paper-pdf](http://arxiv.org/pdf/2210.08198v2)

**Authors**: Ruidi Chen, Boran Hao, Ioannis Ch. Paschalidis

**Abstract**: We develop a Distributionally Robust Optimization (DRO) formulation for Multiclass Logistic Regression (MLR), which could tolerate data contaminated by outliers. The DRO framework uses a probabilistic ambiguity set defined as a ball of distributions that are close to the empirical distribution of the training set in the sense of the Wasserstein metric. We relax the DRO formulation into a regularized learning problem whose regularizer is a norm of the coefficient matrix. We establish out-of-sample performance guarantees for the solutions to our model, offering insights on the role of the regularizer in controlling the prediction error. We apply the proposed method in rendering deep Vision Transformer (ViT)-based image classifiers robust to random and adversarial attacks. Specifically, using the MNIST and CIFAR-10 datasets, we demonstrate reductions in test error rate by up to 83.5% and loss by up to 91.3% compared with baseline methods, by adopting a novel random training method.

摘要: 针对多分类Logistic回归(MLR)模型，提出了一种分布式稳健优化(DRO)方法，该方法能够容忍数据中的离群点污染。DRO框架使用概率模糊集，该模糊集被定义为在Wasserstein度量意义上接近训练集的经验分布的分布球。我们将DRO公式松弛为一个正则化学习问题，其正则化子是系数矩阵的范数。我们为我们的模型的解建立了样本外性能保证，提供了关于正则化在控制预测误差中的作用的见解。我们将该方法应用于基于深度视觉转换器(VIT)的图像分类器对随机攻击和敌意攻击的稳健性。具体地说，使用MNIST和CIFAR-10数据集，我们证明了通过采用一种新的随机训练方法，与基线方法相比，测试错误率降低了83.5%，损失降低了91.3%。



## **22. STDLens: Model Hijacking-Resilient Federated Learning for Object Detection**

STDLens：用于目标检测的模型劫持-弹性联合学习 cs.CR

CVPR 2023. Source Code: https://github.com/git-disl/STDLens

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2303.11511v2) [paper-pdf](http://arxiv.org/pdf/2303.11511v2)

**Authors**: Ka-Ho Chow, Ling Liu, Wenqi Wei, Fatih Ilhan, Yanzhao Wu

**Abstract**: Federated Learning (FL) has been gaining popularity as a collaborative learning framework to train deep learning-based object detection models over a distributed population of clients. Despite its advantages, FL is vulnerable to model hijacking. The attacker can control how the object detection system should misbehave by implanting Trojaned gradients using only a small number of compromised clients in the collaborative learning process. This paper introduces STDLens, a principled approach to safeguarding FL against such attacks. We first investigate existing mitigation mechanisms and analyze their failures caused by the inherent errors in spatial clustering analysis on gradients. Based on the insights, we introduce a three-tier forensic framework to identify and expel Trojaned gradients and reclaim the performance over the course of FL. We consider three types of adaptive attacks and demonstrate the robustness of STDLens against advanced adversaries. Extensive experiments show that STDLens can protect FL against different model hijacking attacks and outperform existing methods in identifying and removing Trojaned gradients with significantly higher precision and much lower false-positive rates.

摘要: 联邦学习(FL)作为一种协作学习框架，在分布的客户群上训练基于深度学习的目标检测模型，已经越来越受欢迎。尽管有优势，但FL很容易受到模特劫持的攻击。攻击者可以通过在协作学习过程中仅使用少量受攻击的客户端植入特洛伊木马梯度来控制对象检测系统的不当行为。本文介绍了STDLens，一种保护FL免受此类攻击的原则性方法。我们首先调查了现有的缓解机制，并分析了它们由于梯度空间聚类分析的固有错误而导致的失败。基于这些见解，我们引入了一个三层取证框架来识别和排除特洛伊木马的梯度，并在FL过程中恢复性能。我们考虑了三种类型的自适应攻击，并证明了STDLens对高级攻击者的健壮性。大量的实验表明，STDLens能够保护FL免受不同模型的劫持攻击，并且在识别和去除特洛伊木马梯度方面优于现有的方法，具有明显更高的精度和更低的误检率。



## **23. Improving robustness of jet tagging algorithms with adversarial training: exploring the loss surface**

利用对抗性训练提高JET标记算法的稳健性：损失曲面的探索 hep-ex

5 pages, 2 figures; submitted to ACAT 2022 proceedings

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2303.14511v1) [paper-pdf](http://arxiv.org/pdf/2303.14511v1)

**Authors**: Annika Stein

**Abstract**: In the field of high-energy physics, deep learning algorithms continue to gain in relevance and provide performance improvements over traditional methods, for example when identifying rare signals or finding complex patterns. From an analyst's perspective, obtaining highest possible performance is desirable, but recently, some attention has been shifted towards studying robustness of models to investigate how well these perform under slight distortions of input features. Especially for tasks that involve many (low-level) inputs, the application of deep neural networks brings new challenges. In the context of jet flavor tagging, adversarial attacks are used to probe a typical classifier's vulnerability and can be understood as a model for systematic uncertainties. A corresponding defense strategy, adversarial training, improves robustness, while maintaining high performance. Investigating the loss surface corresponding to the inputs and models in question reveals geometric interpretations of robustness, taking correlations into account.

摘要: 在高能物理领域，深度学习算法继续提高相关性，并提供比传统方法更好的性能，例如在识别罕见信号或发现复杂模式时。从分析师的角度来看，获得尽可能高的性能是可取的，但最近，一些注意力已经转移到研究模型的稳健性上，以调查这些模型在输入特征轻微扭曲的情况下表现如何。特别是对于涉及许多(低层)输入的任务，深度神经网络的应用带来了新的挑战。在喷气标签的背景下，对抗性攻击被用来探测典型分类器的脆弱性，并且可以被理解为系统不确定性的模型。一种相应的防御策略，对抗性训练，在保持高性能的同时提高了健壮性。研究与所讨论的输入和模型相对应的损失面，揭示了考虑相关性的稳健性的几何解释。



## **24. CgAT: Center-Guided Adversarial Training for Deep Hashing-Based Retrieval**

CgAT：中心引导的基于深度散列的对抗性训练 cs.CV

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2204.10779v6) [paper-pdf](http://arxiv.org/pdf/2204.10779v6)

**Authors**: Xunguang Wang, Yiqun Lin, Xiaomeng Li

**Abstract**: Deep hashing has been extensively utilized in massive image retrieval because of its efficiency and effectiveness. However, deep hashing models are vulnerable to adversarial examples, making it essential to develop adversarial defense methods for image retrieval. Existing solutions achieved limited defense performance because of using weak adversarial samples for training and lacking discriminative optimization objectives to learn robust features. In this paper, we present a min-max based Center-guided Adversarial Training, namely CgAT, to improve the robustness of deep hashing networks through worst adversarial examples. Specifically, we first formulate the center code as a semantically-discriminative representative of the input image content, which preserves the semantic similarity with positive samples and dissimilarity with negative examples. We prove that a mathematical formula can calculate the center code immediately. After obtaining the center codes in each optimization iteration of the deep hashing network, they are adopted to guide the adversarial training process. On the one hand, CgAT generates the worst adversarial examples as augmented data by maximizing the Hamming distance between the hash codes of the adversarial examples and the center codes. On the other hand, CgAT learns to mitigate the effects of adversarial samples by minimizing the Hamming distance to the center codes. Extensive experiments on the benchmark datasets demonstrate the effectiveness of our adversarial training algorithm in defending against adversarial attacks for deep hashing-based retrieval. Compared with the current state-of-the-art defense method, we significantly improve the defense performance by an average of 18.61\%, 12.35\%, and 11.56\% on FLICKR-25K, NUS-WIDE, and MS-COCO, respectively. The code is available at https://github.com/xunguangwang/CgAT.

摘要: 深度哈希法以其高效、高效的特点在海量图像检索中得到了广泛应用。然而，深度哈希模型很容易受到敌意例子的攻击，因此有必要开发针对图像检索的对抗性防御方法。现有的解决方案由于使用弱对抗性样本进行训练，并且缺乏区分优化目标来学习稳健特征，使得防御性能受到限制。本文提出了一种基于最小-最大值的中心引导敌意训练算法，即CgAT，通过最坏的敌意例子来提高深度哈希网络的健壮性。具体地说，我们首先将中心代码定义为输入图像内容的语义区分代表，它保留了与正例的语义相似性和与反例的不相似性。我们证明了一个数学公式可以立即计算中心代码。在获得深度散列网络每次优化迭代的中心代码后，将其用于指导对抗性训练过程。一方面，CgAT通过最大化对抗性示例的哈希码与中心码之间的汉明距离来生成最差的对抗性示例作为扩充数据。另一方面，CgAT通过最小化到中心码的汉明距离来学习减轻对抗性样本的影响。在基准数据集上的大量实验表明，我们的对抗性训练算法在防御基于深度散列的检索的对抗性攻击方面是有效的。与当前最先进的防御方法相比，我们在Flickr-25K、NUS-wide和MS-CoCo上的防御性能分别平均提高了18.61、12.35和11.56。代码可在https://github.com/xunguangwang/CgAT.上获得



## **25. No more Reviewer #2: Subverting Automatic Paper-Reviewer Assignment using Adversarial Learning**

不再有审稿人#2：使用对抗性学习颠覆自动论文审稿人分配 cs.CR

Accepted at USENIX Security Symposium 2023

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2303.14443v1) [paper-pdf](http://arxiv.org/pdf/2303.14443v1)

**Authors**: Thorsten Eisenhofer, Erwin Quiring, Jonas Möller, Doreen Riepel, Thorsten Holz, Konrad Rieck

**Abstract**: The number of papers submitted to academic conferences is steadily rising in many scientific disciplines. To handle this growth, systems for automatic paper-reviewer assignments are increasingly used during the reviewing process. These systems use statistical topic models to characterize the content of submissions and automate the assignment to reviewers. In this paper, we show that this automation can be manipulated using adversarial learning. We propose an attack that adapts a given paper so that it misleads the assignment and selects its own reviewers. Our attack is based on a novel optimization strategy that alternates between the feature space and problem space to realize unobtrusive changes to the paper. To evaluate the feasibility of our attack, we simulate the paper-reviewer assignment of an actual security conference (IEEE S&P) with 165 reviewers on the program committee. Our results show that we can successfully select and remove reviewers without access to the assignment system. Moreover, we demonstrate that the manipulated papers remain plausible and are often indistinguishable from benign submissions.

摘要: 在许多科学领域，提交给学术会议的论文数量正在稳步上升。为了应对这种增长，在审查过程中越来越多地使用自动分配论文审稿人的系统。这些系统使用统计主题模型来表征提交的内容，并自动分配给评审员。在本文中，我们证明了这种自动化可以通过对抗性学习来操纵。我们提出了一种攻击，该攻击改编一篇给定的论文，以便它误导作业并选择自己的审稿人。我们的攻击是基于一种新的优化策略，该策略在特征空间和问题空间之间交替使用，以实现对论文的不引人注目的更改。为了评估我们攻击的可行性，我们模拟了一个实际安全会议(IEEE S&P)的论文审稿人分配，项目委员会有165名审稿人。我们的结果表明，我们可以在不访问分配系统的情况下成功地选择和删除审阅者。此外，我们证明，被操纵的文件仍然可信，通常与良性提交的文件没有区别。



## **26. A User-Based Authentication and DoS Mitigation Scheme for Wearable Wireless Body Sensor Networks**

一种基于用户的可穿戴无线体感网络认证和DoS防御方案 cs.CR

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2303.14441v1) [paper-pdf](http://arxiv.org/pdf/2303.14441v1)

**Authors**: Nombulelo Zulu, Deon P. Du Plessis, Topside E. Mathonsi, Tshimangadzo M. Tshilongamulenzhe

**Abstract**: Wireless Body Sensor Networks (WBSNs) is one of the greatest growing technology for sensing and performing various tasks. The information transmitted in the WBSNs is vulnerable to cyber-attacks, therefore security is very important. Denial of Service (DoS) attacks are considered one of the major threats against WBSNs security. In DoS attacks, an adversary targets to degrade and shut down the efficient use of the network and disrupt the services in the network causing them inaccessible to its intended users. If sensitive information of patients in WBSNs, such as the medical history is accessed by unauthorized users, the patient may suffer much more than the disease itself, it may result in loss of life. This paper proposes a User-Based authentication scheme to mitigate DoS attacks in WBSNs. A five-phase User-Based authentication DoS mitigation scheme for WBSNs is designed by integrating Elliptic Curve Cryptography (ECC) with Rivest Cipher 4 (RC4) to ensure a strong authentication process that will only allow authorized users to access nodes on WBSNs.

摘要: 无线身体传感器网络(WBSNs)是目前发展最快的传感和执行各种任务的技术之一。无线传感器网络中传输的信息很容易受到网络攻击，因此安全性非常重要。拒绝服务(DoS)攻击被认为是对WBSNs安全的主要威胁之一。在DoS攻击中，对手的目标是降低和关闭网络的有效使用，并中断网络中的服务，使其目标用户无法访问这些服务。如果WBSNs中患者的敏感信息，如病历被未经授权的用户访问，患者可能会遭受比疾病本身更大的痛苦，可能会导致生命损失。提出了一种基于用户的身份认证方案来缓解无线传感器网络中的DoS攻击。通过将椭圆曲线密码体制(ECC)和Rivest密码4(RC4)相结合，设计了一种基于用户的五阶段无线传感器网络认证DoS缓解方案，以确保强认证过程只允许授权用户访问WBSNs上的节点。



## **27. Consistent Attack: Universal Adversarial Perturbation on Embodied Vision Navigation**

一致攻击：具身视觉导航的普遍对抗性扰动 cs.LG

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2206.05751v4) [paper-pdf](http://arxiv.org/pdf/2206.05751v4)

**Authors**: Chengyang Ying, You Qiaoben, Xinning Zhou, Hang Su, Wenbo Ding, Jianyong Ai

**Abstract**: Embodied agents in vision navigation coupled with deep neural networks have attracted increasing attention. However, deep neural networks have been shown vulnerable to malicious adversarial noises, which may potentially cause catastrophic failures in Embodied Vision Navigation. Among different adversarial noises, universal adversarial perturbations (UAP), i.e., a constant image-agnostic perturbation applied on every input frame of the agent, play a critical role in Embodied Vision Navigation since they are computation-efficient and application-practical during the attack. However, existing UAP methods ignore the system dynamics of Embodied Vision Navigation and might be sub-optimal. In order to extend UAP to the sequential decision setting, we formulate the disturbed environment under the universal noise $\delta$, as a $\delta$-disturbed Markov Decision Process ($\delta$-MDP). Based on the formulation, we analyze the properties of $\delta$-MDP and propose two novel Consistent Attack methods, named Reward UAP and Trajectory UAP, for attacking Embodied agents, which consider the dynamic of the MDP and calculate universal noises by estimating the disturbed distribution and the disturbed Q function. For various victim models, our Consistent Attack can cause a significant drop in their performance in the PointGoal task in Habitat with different datasets and different scenes. Extensive experimental results indicate that there exist serious potential risks for applying Embodied Vision Navigation methods to the real world.

摘要: 视觉导航中的具身智能体与深度神经网络相结合，越来越受到人们的关注。然而，深度神经网络已被证明容易受到恶意对抗性噪声的攻击，这可能会导致具身视觉导航中的灾难性故障。在不同的对抗噪声中，通用对抗扰动(UAP)，即在智能体的每一输入帧上施加的与图像无关的恒定扰动，在嵌入视觉导航中起着至关重要的作用，因为它们在攻击过程中具有计算效率和应用实用性。然而，现有的UAP方法忽略了体现视觉导航的系统动力学，可能是次优的。为了将UAP扩展到序贯决策环境，我们将普遍噪声下的扰动环境描述为一个$-扰动马尔可夫决策过程($-MDP)。在此基础上，分析了$Delta$-MDP的特性，提出了两种新的一致性攻击方法--报酬UAP和轨迹UAP，该方法考虑了MDP的动态特性，通过估计扰动分布和扰动Q函数来计算通用噪声。对于不同的受害者模型，我们的一致攻击会导致他们在不同数据集和不同场景下在Habit的PointGoal任务中的性能显著下降。大量的实验结果表明，将具身视觉导航方法应用于现实世界存在着严重的潜在风险。



## **28. Test-time Defense against Adversarial Attacks: Detection and Reconstruction of Adversarial Examples via Masked Autoencoder**

对抗性攻击的测试时间防御：基于屏蔽自动编码器的对抗性实例检测与重构 cs.CV

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2303.12848v2) [paper-pdf](http://arxiv.org/pdf/2303.12848v2)

**Authors**: Yun-Yun Tsai, Ju-Chin Chao, Albert Wen, Zhaoyuan Yang, Chengzhi Mao, Tapan Shah, Junfeng Yang

**Abstract**: Existing defense methods against adversarial attacks can be categorized into training time and test time defenses. Training time defense, i.e., adversarial training, requires a significant amount of extra time for training and is often not able to be generalized to unseen attacks. On the other hand, test time defense by test time weight adaptation requires access to perform gradient descent on (part of) the model weights, which could be infeasible for models with frozen weights. To address these challenges, we propose DRAM, a novel defense method to Detect and Reconstruct multiple types of Adversarial attacks via Masked autoencoder (MAE). We demonstrate how to use MAE losses to build a KS-test to detect adversarial attacks. Moreover, the MAE losses can be used to repair adversarial samples from unseen attack types. In this sense, DRAM neither requires model weight updates in test time nor augments the training set with more adversarial samples. Evaluating DRAM on the large-scale ImageNet data, we achieve the best detection rate of 82% on average on eight types of adversarial attacks compared with other detection baselines. For reconstruction, DRAM improves the robust accuracy by 6% ~ 41% for Standard ResNet50 and 3% ~ 8% for Robust ResNet50 compared with other self-supervision tasks, such as rotation prediction and contrastive learning.

摘要: 现有的对抗攻击防御方法可分为训练时间防御和测试时间防御。训练时间防守，即对抗性训练，需要大量的额外时间进行训练，通常不能概括为看不见的攻击。另一方面，通过测试时间权重自适应来保护测试时间需要访问对模型权重(部分)执行梯度下降的权限，这对于具有冻结权重的模型可能是不可行的。为了应对这些挑战，我们提出了一种新的防御方法DRAM，它通过掩蔽自动编码器(MAE)来检测和重建多种类型的对抗性攻击。我们演示了如何使用MAE损失来构建KS测试来检测对手攻击。此外，MAE损失可用于修复来自未知攻击类型的敌方样本。从这个意义上说，DRAM既不需要在测试时间更新模型权重，也不需要用更多的对抗性样本来扩充训练集。在大规模的ImageNet数据上对DRAM进行评估，与其他检测基线相比，对8种类型的对抗性攻击平均获得了82%的最佳检测率。在重建方面，与旋转预测和对比学习等其他自我监督任务相比，DRAM将标准ResNet50的稳健准确率提高了6%~41%，稳健ResNet50的稳健准确率提高了3%~8%。



## **29. WiFi Physical Layer Stays Awake and Responds When it Should Not**

WiFi物理层保持唤醒，并在不应唤醒时进行响应 cs.NI

12 pages

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2301.00269v2) [paper-pdf](http://arxiv.org/pdf/2301.00269v2)

**Authors**: Ali Abedi, Haofan Lu, Alex Chen, Charlie Liu, Omid Abari

**Abstract**: WiFi communication should be possible only between devices inside the same network. However, we find that all existing WiFi devices send back acknowledgments (ACK) to even fake packets received from unauthorized WiFi devices outside of their network. Moreover, we find that an unauthorized device can manipulate the power-saving mechanism of WiFi radios and keep them continuously awake by sending specific fake beacon frames to them. Our evaluation of over 5,000 devices from 186 vendors confirms that these are widespread issues. We believe these loopholes cannot be prevented, and hence they create privacy and security concerns. Finally, to show the importance of these issues and their consequences, we implement and demonstrate two attacks where an adversary performs battery drain and WiFi sensing attacks just using a tiny WiFi module which costs less than ten dollars.

摘要: WiFi通信应该只能在同一网络内的设备之间进行。然而，我们发现，所有现有的WiFi设备都会向从其网络外部的未经授权的WiFi设备接收的虚假数据包发送回确认(ACK)。此外，我们发现未经授权的设备可以操纵WiFi无线电的节电机制，并通过向其发送特定的虚假信标帧来保持其持续唤醒。我们对186家供应商的5,000多台设备进行的评估证实，这些问题普遍存在。我们认为这些漏洞是无法阻止的，因此它们会造成隐私和安全方面的问题。最后，为了说明这些问题的重要性及其后果，我们实现并演示了两个攻击，其中对手仅使用一个成本不到10美元的微小WiFi模块就可以执行电池耗尽攻击和WiFi传感攻击。



## **30. Ensemble-based Blackbox Attacks on Dense Prediction**

基于集成的稠密预测黑盒攻击 cs.CV

CVPR 2023 Accepted

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2303.14304v1) [paper-pdf](http://arxiv.org/pdf/2303.14304v1)

**Authors**: Zikui Cai, Yaoteng Tan, M. Salman Asif

**Abstract**: We propose an approach for adversarial attacks on dense prediction models (such as object detectors and segmentation). It is well known that the attacks generated by a single surrogate model do not transfer to arbitrary (blackbox) victim models. Furthermore, targeted attacks are often more challenging than the untargeted attacks. In this paper, we show that a carefully designed ensemble can create effective attacks for a number of victim models. In particular, we show that normalization of the weights for individual models plays a critical role in the success of the attacks. We then demonstrate that by adjusting the weights of the ensemble according to the victim model can further improve the performance of the attacks. We performed a number of experiments for object detectors and segmentation to highlight the significance of the our proposed methods. Our proposed ensemble-based method outperforms existing blackbox attack methods for object detection and segmentation. Finally we show that our proposed method can also generate a single perturbation that can fool multiple blackbox detection and segmentation models simultaneously. Code is available at https://github.com/CSIPlab/EBAD.

摘要: 我们提出了一种针对密集预测模型(如目标检测器和分割)的对抗性攻击方法。众所周知，由单一代理模型生成的攻击不会转移到任意(黑箱)受害者模型。此外，有针对性的攻击往往比无针对性的攻击更具挑战性。在这篇文章中，我们证明了精心设计的集成可以为许多受害者模型创建有效的攻击。特别是，我们证明了各个模型的权重的归一化在攻击的成功中起着关键作用。然后，我们证明了通过根据受害者模型调整集成的权重可以进一步提高攻击的性能。我们对目标检测和分割进行了大量的实验，以突出我们所提出的方法的重要性。在目标检测和分割方面，我们提出的基于集成的方法比现有的黑盒攻击方法具有更好的性能。最后，我们证明了我们提出的方法还可以产生单个扰动，从而同时欺骗多个黑盒检测和分割模型。代码可在https://github.com/CSIPlab/EBAD.上找到



## **31. Utilizing Network Properties to Detect Erroneous Inputs**

利用网络属性检测错误输入 cs.CV

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2002.12520v3) [paper-pdf](http://arxiv.org/pdf/2002.12520v3)

**Authors**: Matt Gorbett, Nathaniel Blanchard

**Abstract**: Neural networks are vulnerable to a wide range of erroneous inputs such as adversarial, corrupted, out-of-distribution, and misclassified examples. In this work, we train a linear SVM classifier to detect these four types of erroneous data using hidden and softmax feature vectors of pre-trained neural networks. Our results indicate that these faulty data types generally exhibit linearly separable activation properties from correct examples, giving us the ability to reject bad inputs with no extra training or overhead. We experimentally validate our findings across a diverse range of datasets, domains, pre-trained models, and adversarial attacks.

摘要: 神经网络很容易受到各种各样的错误输入的影响，例如对抗性的、被破坏的、不分布的和错误分类的例子。在这项工作中，我们训练一个线性支持向量机来检测这四种类型的错误数据，使用预先训练好的神经网络的隐含和软最大特征向量。我们的结果表明，这些错误的数据类型通常表现出与正确示例线性可分离的激活属性，使我们能够拒绝不良输入，而不需要额外的训练或开销。我们在不同的数据集、领域、预先训练的模型和对抗性攻击中对我们的发现进行了实验验证。



## **32. How many dimensions are required to find an adversarial example?**

需要多少维度才能找到对抗性的例子？ cs.LG

Comments welcome!

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.14173v1) [paper-pdf](http://arxiv.org/pdf/2303.14173v1)

**Authors**: Charles Godfrey, Henry Kvinge, Elise Bishoff, Myles Mckay, Davis Brown, Tim Doster, Eleanor Byler

**Abstract**: Past work exploring adversarial vulnerability have focused on situations where an adversary can perturb all dimensions of model input. On the other hand, a range of recent works consider the case where either (i) an adversary can perturb a limited number of input parameters or (ii) a subset of modalities in a multimodal problem. In both of these cases, adversarial examples are effectively constrained to a subspace $V$ in the ambient input space $\mathcal{X}$. Motivated by this, in this work we investigate how adversarial vulnerability depends on $\dim(V)$. In particular, we show that the adversarial success of standard PGD attacks with $\ell^p$ norm constraints behaves like a monotonically increasing function of $\epsilon (\frac{\dim(V)}{\dim \mathcal{X}})^{\frac{1}{q}}$ where $\epsilon$ is the perturbation budget and $\frac{1}{p} + \frac{1}{q} =1$, provided $p > 1$ (the case $p=1$ presents additional subtleties which we analyze in some detail). This functional form can be easily derived from a simple toy linear model, and as such our results land further credence to arguments that adversarial examples are endemic to locally linear models on high dimensional spaces.

摘要: 过去探索对手脆弱性的工作主要集中在对手可以扰乱模型输入的所有维度的情况。另一方面，最近的一系列工作考虑了这样的情况：(I)对手可以扰动有限数量的输入参数或(Ii)多通道问题中的一组通道。在这两种情况下，敌意示例都被有效地约束到环境输入空间$\mathcal{X}$中的子空间$V$。受此启发，在本工作中，我们研究了对手脆弱性是如何依赖于$\dim(V)$的。特别地，我们证明了具有$^p$范数约束的标准PGD攻击的对抗成功表现为$\epsilon(\frac{\dim(V)}{\dim\mathcal{X}})^{\frac{1}{q}}$的单调递增函数，其中$\epsilon$是扰动预算，而$\frac{1}{p}+\frac{1}{q}=1$，假设$p>1$($p=1$给出了更多的细节，我们进行了一些详细的分析)。这种函数形式可以很容易地从一个简单的玩具线性模型中得到，因此我们的结果进一步证明了高维空间上的对抗性例子是局部线性模型特有的。



## **33. Adversarial Attack and Defense for Medical Image Analysis: Methods and Applications**

医学图像分析中的对抗性攻防方法及应用 eess.IV

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.14133v1) [paper-pdf](http://arxiv.org/pdf/2303.14133v1)

**Authors**: Junhao Dong, Junxi Chen, Xiaohua Xie, Jianhuang Lai, Hao Chen

**Abstract**: Deep learning techniques have achieved superior performance in computer-aided medical image analysis, yet they are still vulnerable to imperceptible adversarial attacks, resulting in potential misdiagnosis in clinical practice. Oppositely, recent years have also witnessed remarkable progress in defense against these tailored adversarial examples in deep medical diagnosis systems. In this exposition, we present a comprehensive survey on recent advances in adversarial attack and defense for medical image analysis with a novel taxonomy in terms of the application scenario. We also provide a unified theoretical framework for different types of adversarial attack and defense methods for medical image analysis. For a fair comparison, we establish a new benchmark for adversarially robust medical diagnosis models obtained by adversarial training under various scenarios. To the best of our knowledge, this is the first survey paper that provides a thorough evaluation of adversarially robust medical diagnosis models. By analyzing qualitative and quantitative results, we conclude this survey with a detailed discussion of current challenges for adversarial attack and defense in medical image analysis systems to shed light on future research directions.

摘要: 深度学习技术在计算机辅助医学图像分析中取得了优异的性能，但仍然容易受到潜移默化的对抗性攻击，导致临床实践中潜在的误诊。相反，近年来在防御深度医疗诊断系统中这些量身定做的对抗性例子方面也取得了显著进展。在这篇论述中，我们从应用场景的角度对医学图像分析中的对抗性攻击和防御的最新进展进行了全面的综述。为医学图像分析中不同类型的对抗性攻击和防御方法提供了统一的理论框架。为了进行公平的比较，我们建立了一个新的基准，用于在不同场景下通过对抗性训练获得对抗性健壮的医疗诊断模型。据我们所知，这是第一份对反面稳健的医疗诊断模型进行彻底评估的调查报告。通过对定性和定量结果的分析，我们对当前医学图像分析系统中对抗性攻击和防御的挑战进行了详细的讨论，以阐明未来的研究方向。



## **34. Optimal Smoothing Distribution Exploration for Backdoor Neutralization in Deep Learning-based Traffic Systems**

基于深度学习的交通系统后门中立最优平滑分布探索 cs.LG

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.14197v1) [paper-pdf](http://arxiv.org/pdf/2303.14197v1)

**Authors**: Yue Wang, Wending Li, Michail Maniatakos, Saif Eddin Jabari

**Abstract**: Deep Reinforcement Learning (DRL) enhances the efficiency of Autonomous Vehicles (AV), but also makes them susceptible to backdoor attacks that can result in traffic congestion or collisions. Backdoor functionality is typically incorporated by contaminating training datasets with covert malicious data to maintain high precision on genuine inputs while inducing the desired (malicious) outputs for specific inputs chosen by adversaries. Current defenses against backdoors mainly focus on image classification using image-based features, which cannot be readily transferred to the regression task of DRL-based AV controllers since the inputs are continuous sensor data, i.e., the combinations of velocity and distance of AV and its surrounding vehicles. Our proposed method adds well-designed noise to the input to neutralize backdoors. The approach involves learning an optimal smoothing (noise) distribution to preserve the normal functionality of genuine inputs while neutralizing backdoors. By doing so, the resulting model is expected to be more resilient against backdoor attacks while maintaining high accuracy on genuine inputs. The effectiveness of the proposed method is verified on a simulated traffic system based on a microscopic traffic simulator, where experimental results showcase that the smoothed traffic controller can neutralize all trigger samples and maintain the performance of relieving traffic congestion

摘要: 深度强化学习(DRL)提高了自动驾驶车辆(AV)的效率，但也使它们容易受到可能导致交通拥堵或碰撞的后门攻击。后门功能通常通过用隐藏的恶意数据污染训练数据集来结合，以保持对真实输入的高精度，同时诱导对手选择的特定输入的所需(恶意)输出。目前对后门的防御主要集中在基于图像特征的图像分类上，由于输入是连续的传感器数据，即无人机及其周围车辆的速度和距离的组合，因此不容易转移到基于DRL的无人机控制器的回归任务中。我们提出的方法将精心设计的噪声添加到输入以中和后门。该方法包括学习最优平滑(噪声)分布，以保留真正输入的正常功能，同时中和后门。通过这样做，最终的模型预计将更具抵御后门攻击的能力，同时保持对真实输入的高精度。在一个基于微观交通模拟器的模拟交通系统上验证了该方法的有效性，实验结果表明，平滑后的交通控制器能够中和所有触发样本，并保持缓解交通拥堵的性能



## **35. PIAT: Parameter Interpolation based Adversarial Training for Image Classification**

PIAT：基于参数内插的对抗性图像分类训练 cs.CV

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.13955v1) [paper-pdf](http://arxiv.org/pdf/2303.13955v1)

**Authors**: Kun He, Xin Liu, Yichen Yang, Zhou Qin, Weigao Wen, Hui Xue, John E. Hopcroft

**Abstract**: Adversarial training has been demonstrated to be the most effective approach to defend against adversarial attacks. However, existing adversarial training methods show apparent oscillations and overfitting issue in the training process, degrading the defense efficacy. In this work, we propose a novel framework, termed Parameter Interpolation based Adversarial Training (PIAT), that makes full use of the historical information during training. Specifically, at the end of each epoch, PIAT tunes the model parameters as the interpolation of the parameters of the previous and current epochs. Besides, we suggest to use the Normalized Mean Square Error (NMSE) to further improve the robustness by aligning the clean and adversarial examples. Compared with other regularization methods, NMSE focuses more on the relative magnitude of the logits rather than the absolute magnitude. Extensive experiments on several benchmark datasets and various networks show that our method could prominently improve the model robustness and reduce the generalization error. Moreover, our framework is general and could further boost the robust accuracy when combined with other adversarial training methods.

摘要: 对抗性训练已被证明是防御对抗性攻击的最有效方法。然而，现有的对抗性训练方法在训练过程中表现出明显的振荡和过度匹配问题，降低了防守效能。在这项工作中，我们提出了一种新的框架，称为基于参数内插的对抗性训练(PIAT)，它充分利用了训练过程中的历史信息。具体地说，在每个历元结束时，PIAT将模型参数调整为前一个历元和当前历元的参数的内插。此外，我们建议使用归一化均方误差(NMSE)来进一步提高稳健性，通过对齐干净的和对抗性的例子。与其他正则化方法相比，NMSE更注重对数的相对大小，而不是绝对大小。在多个基准数据集和不同网络上的大量实验表明，该方法可以显著提高模型的稳健性，降低泛化误差。此外，我们的框架是通用的，当与其他对抗性训练方法相结合时，可以进一步提高鲁棒性准确率。



## **36. EC-CFI: Control-Flow Integrity via Code Encryption Counteracting Fault Attacks**

EC-CFI：通过代码加密对抗错误攻击的控制流完整性 cs.CR

Accepted at HOST'23

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2301.13760v2) [paper-pdf](http://arxiv.org/pdf/2301.13760v2)

**Authors**: Pascal Nasahl, Salmin Sultana, Hans Liljestrand, Karanvir Grewal, Michael LeMay, David M. Durham, David Schrammel, Stefan Mangard

**Abstract**: Fault attacks enable adversaries to manipulate the control-flow of security-critical applications. By inducing targeted faults into the CPU, the software's call graph can be escaped and the control-flow can be redirected to arbitrary functions inside the program. To protect the control-flow from these attacks, dedicated fault control-flow integrity (CFI) countermeasures are commonly deployed. However, these schemes either have high detection latencies or require intrusive hardware changes. In this paper, we present EC-CFI, a software-based cryptographically enforced CFI scheme with no detection latency utilizing hardware features of recent Intel platforms. Our EC-CFI prototype is designed to prevent an adversary from escaping the program's call graph using faults by encrypting each function with a different key before execution. At runtime, the instrumented program dynamically derives the decryption key, ensuring that the code only can be successfully decrypted when the program follows the intended call graph. To enable this level of protection on Intel commodity systems, we introduce extended page table (EPT) aliasing allowing us to achieve function-granular encryption by combing Intel's TME-MK and virtualization technology. We open-source our custom LLVM-based toolchain automatically protecting arbitrary programs with EC-CFI. Furthermore, we evaluate our EPT aliasing approach with the SPEC CPU2017 and Embench-IoT benchmarks and discuss and evaluate potential TME-MK hardware changes minimizing runtime overheads.

摘要: 故障攻击使攻击者能够操纵安全关键型应用程序的控制流。通过在CPU中引入有针对性的错误，可以避开软件的调用图，并将控制流重定向到程序内的任意函数。为了保护控制流免受这些攻击，通常部署专用的故障控制流完整性(CFI)对策。然而，这些方案要么具有很高的检测延迟，要么需要侵入性的硬件改变。在本文中，我们提出了EC-CFI，这是一种基于软件的密码强制CFI方案，利用最近Intel平台的硬件特性，没有检测延迟。我们的EC-CFI原型旨在通过在执行前使用不同的密钥加密每个函数，防止对手使用错误逃离程序的调用图。在运行时，插入指令的程序动态地派生解密密钥，确保只有当程序遵循预期的调用图时才能成功解密代码。为了在英特尔商用系统上实现这种级别的保护，我们引入了扩展页表(EPT)别名，使我们能够通过结合英特尔的TME-MK和虚拟化技术来实现函数级加密。我们将基于LLVM的定制工具链开源，使用EC-CFI自动保护任意程序。此外，我们使用SPEC CPU2017和Embase-IoT基准评估了我们的EPT混叠方法，并讨论和评估了潜在的TME-MK硬件更改，以最大限度地减少运行时开销。



## **37. SCRAMBLE-CFI: Mitigating Fault-Induced Control-Flow Attacks on OpenTitan**

SCRIBLE-CFI：缓解OpenTitan上的错误引起的控制流攻击 cs.CR

Accepted at GLSVLSI'23

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.03711v3) [paper-pdf](http://arxiv.org/pdf/2303.03711v3)

**Authors**: Pascal Nasahl, Stefan Mangard

**Abstract**: Secure elements physically exposed to adversaries are frequently targeted by fault attacks. These attacks can be utilized to hijack the control-flow of software allowing the attacker to bypass security measures, extract sensitive data, or gain full code execution. In this paper, we systematically analyze the threat vector of fault-induced control-flow manipulations on the open-source OpenTitan secure element. Our thorough analysis reveals that current countermeasures of this chip either induce large area overheads or still cannot prevent the attacker from exploiting the identified threats. In this context, we introduce SCRAMBLE-CFI, an encryption-based control-flow integrity scheme utilizing existing hardware features of OpenTitan. SCRAMBLE-CFI confines, with minimal hardware overhead, the impact of fault-induced control-flow attacks by encrypting each function with a different encryption tweak at load-time. At runtime, code only can be successfully decrypted when the correct decryption tweak is active. We open-source our hardware changes and release our LLVM toolchain automatically protecting programs. Our analysis shows that SCRAMBLE-CFI complementarily enhances security guarantees of OpenTitan with a negligible hardware overhead of less than 3.97 % and a runtime overhead of 7.02 % for the Embench-IoT benchmarks.

摘要: 物理上暴露在对手面前的安全元素经常成为故障攻击的目标。这些攻击可用于劫持软件的控制流，从而允许攻击者绕过安全措施、提取敏感数据或获得完整的代码执行。在本文中，我们系统地分析了开源OpenTitan安全元素上由错误引起的控制流操作的威胁向量。我们的深入分析表明，目前该芯片的应对措施要么导致大面积开销，要么仍然无法阻止攻击者利用已识别的威胁。在此背景下，我们介绍了一种基于加密的控制流完整性方案SCRIBLE-CFI，该方案利用了OpenTitan现有的硬件特性。置乱-CFI通过在加载时使用不同的加密调整对每个函数进行加密，以最小的硬件开销限制了故障引发的控制流攻击的影响。在运行时，只有当正确的解密调整处于活动状态时，才能成功解密代码。我们将我们的硬件更改开源，并发布我们的LLVM工具链自动保护程序。我们的分析表明，在硬件开销小于3.97%、运行时开销为7.02%的情况下，SCRIBLE-CFI互补地增强了OpenTitan的安全保证。



## **38. Foiling Explanations in Deep Neural Networks**

深度神经网络中的模糊解释 cs.CV

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2211.14860v2) [paper-pdf](http://arxiv.org/pdf/2211.14860v2)

**Authors**: Snir Vitrack Tamam, Raz Lapid, Moshe Sipper

**Abstract**: Deep neural networks (DNNs) have greatly impacted numerous fields over the past decade. Yet despite exhibiting superb performance over many problems, their black-box nature still poses a significant challenge with respect to explainability. Indeed, explainable artificial intelligence (XAI) is crucial in several fields, wherein the answer alone -- sans a reasoning of how said answer was derived -- is of little value. This paper uncovers a troubling property of explanation methods for image-based DNNs: by making small visual changes to the input image -- hardly influencing the network's output -- we demonstrate how explanations may be arbitrarily manipulated through the use of evolution strategies. Our novel algorithm, AttaXAI, a model-agnostic, adversarial attack on XAI algorithms, only requires access to the output logits of a classifier and to the explanation map; these weak assumptions render our approach highly useful where real-world models and data are concerned. We compare our method's performance on two benchmark datasets -- CIFAR100 and ImageNet -- using four different pretrained deep-learning models: VGG16-CIFAR100, VGG16-ImageNet, MobileNet-CIFAR100, and Inception-v3-ImageNet. We find that the XAI methods can be manipulated without the use of gradients or other model internals. Our novel algorithm is successfully able to manipulate an image in a manner imperceptible to the human eye, such that the XAI method outputs a specific explanation map. To our knowledge, this is the first such method in a black-box setting, and we believe it has significant value where explainability is desired, required, or legally mandatory.

摘要: 在过去的十年中，深度神经网络(DNN)对众多领域产生了巨大的影响。然而，尽管在许多问题上表现出了出色的表现，但它们的黑匣子性质仍然在可解释性方面构成了一个重大挑战。事实上，可解释人工智能(XAI)在几个领域都是至关重要的，在这些领域中，答案本身--不考虑答案是如何得出的--几乎没有价值。本文揭示了基于图像的DNN解释方法的一个令人不安的特性：通过对输入图像进行微小的视觉改变--几乎不影响网络的输出--我们演示了如何通过使用进化策略来任意操纵解释。我们的新算法AttaXAI是对XAI算法的一种与模型无关的对抗性攻击，它只需要访问分类器的输出日志和解释地图；这些弱假设使得我们的方法在涉及真实世界的模型和数据时非常有用。我们使用四个不同的预训练深度学习模型：VGG16-CIFAR100、VGG16-ImageNet、MobileNet-CIFAR100和Inception-v3-ImageNet，在两个基准数据集CIFAR100和ImageNet上比较了我们的方法的性能。我们发现，XAI方法可以在不使用梯度或其他模型内部的情况下进行操作。我们的新算法能够成功地以人眼看不到的方式操作图像，从而XAI方法输出特定的解释地图。据我们所知，这是黑盒环境中第一个这样的方法，我们相信它在需要可解释性、要求可解释性或法律强制性的地方具有重要价值。



## **39. Effective black box adversarial attack with handcrafted kernels**

利用手工制作的核进行有效的黑盒对抗攻击 cs.CV

12 pages, 5 figures, 3 tables, IWANN conference

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.13887v1) [paper-pdf](http://arxiv.org/pdf/2303.13887v1)

**Authors**: Petr Dvořáček, Petr Hurtik, Petra Števuliáková

**Abstract**: We propose a new, simple framework for crafting adversarial examples for black box attacks. The idea is to simulate the substitution model with a non-trainable model compounded of just one layer of handcrafted convolutional kernels and then train the generator neural network to maximize the distance of the outputs for the original and generated adversarial image. We show that fooling the prediction of the first layer causes the whole network to be fooled and decreases its accuracy on adversarial inputs. Moreover, we do not train the neural network to obtain the first convolutional layer kernels, but we create them using the technique of F-transform. Therefore, our method is very time and resource effective.

摘要: 我们提出了一个新的、简单的框架来制作黑盒攻击的对抗性例子。其思想是用仅由一层手工制作的卷积核组成的不可训练模型来模拟替换模型，然后训练生成器神经网络以最大化原始和生成的对抗性图像的输出距离。我们表明，愚弄第一层的预测会导致整个网络被愚弄，并降低其对对手输入的精度。此外，我们不训练神经网络来获得第一卷积层核，但我们使用F变换技术来创建它们。因此，我们的方法是非常节省时间和资源的。



## **40. Physically Adversarial Infrared Patches with Learnable Shapes and Locations**

具有可学习形状和位置的物理对抗性红外线补丁 cs.CV

accepted by CVPR2023

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.13868v1) [paper-pdf](http://arxiv.org/pdf/2303.13868v1)

**Authors**: Wei Xingxing, Yu Jie, Huang Yao

**Abstract**: Owing to the extensive application of infrared object detectors in the safety-critical tasks, it is necessary to evaluate their robustness against adversarial examples in the real world. However, current few physical infrared attacks are complicated to implement in practical application because of their complex transformation from digital world to physical world. To address this issue, in this paper, we propose a physically feasible infrared attack method called "adversarial infrared patches". Considering the imaging mechanism of infrared cameras by capturing objects' thermal radiation, adversarial infrared patches conduct attacks by attaching a patch of thermal insulation materials on the target object to manipulate its thermal distribution. To enhance adversarial attacks, we present a novel aggregation regularization to guide the simultaneous learning for the patch' shape and location on the target object. Thus, a simple gradient-based optimization can be adapted to solve for them. We verify adversarial infrared patches in different object detection tasks with various object detectors. Experimental results show that our method achieves more than 90\% Attack Success Rate (ASR) versus the pedestrian detector and vehicle detector in the physical environment, where the objects are captured in different angles, distances, postures, and scenes. More importantly, adversarial infrared patch is easy to implement, and it only needs 0.5 hours to be constructed in the physical world, which verifies its effectiveness and efficiency.

摘要: 由于红外目标探测器在安全关键任务中的广泛应用，有必要评估其对现实世界中的敌方例子的鲁棒性。然而，由于从数字世界到物理世界的复杂转换，目前较少的物理红外攻击在实际应用中实现起来比较复杂。针对这一问题，本文提出了一种物理上可行的红外攻击方法，称为对抗性红外补丁。考虑到红外相机通过捕捉目标的热辐射来成像的机理，对抗红外贴片通过在目标对象上粘贴一块隔热材料来操纵目标对象的热分布来进行攻击。为了增强对抗性攻击，我们提出了一种新的聚合正则化方法来指导对目标物体上补丁的形状和位置的同时学习。因此，可以采用一种简单的基于梯度的优化方法来求解它们。在不同的目标检测任务中，我们使用不同的目标检测器来验证敌方红外补丁。实验结果表明，与行人检测器和车辆检测器相比，在不同角度、不同距离、不同姿态和不同场景下的物理环境中，该方法的攻击成功率(ASR)达到了90%以上。更重要的是，对抗性红外补丁易于实现，在物理世界中仅需0.5小时即可构建，验证了其有效性和高效性。



## **41. Feature Separation and Recalibration for Adversarial Robustness**

用于对抗稳健性的特征分离和重新校准 cs.CV

CVPR 2023 (Highlight)

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.13846v1) [paper-pdf](http://arxiv.org/pdf/2303.13846v1)

**Authors**: Woo Jae Kim, Yoonki Cho, Junsik Jung, Sung-Eui Yoon

**Abstract**: Deep neural networks are susceptible to adversarial attacks due to the accumulation of perturbations in the feature level, and numerous works have boosted model robustness by deactivating the non-robust feature activations that cause model mispredictions. However, we claim that these malicious activations still contain discriminative cues and that with recalibration, they can capture additional useful information for correct model predictions. To this end, we propose a novel, easy-to-plugin approach named Feature Separation and Recalibration (FSR) that recalibrates the malicious, non-robust activations for more robust feature maps through Separation and Recalibration. The Separation part disentangles the input feature map into the robust feature with activations that help the model make correct predictions and the non-robust feature with activations that are responsible for model mispredictions upon adversarial attack. The Recalibration part then adjusts the non-robust activations to restore the potentially useful cues for model predictions. Extensive experiments verify the superiority of FSR compared to traditional deactivation techniques and demonstrate that it improves the robustness of existing adversarial training methods by up to 8.57% with small computational overhead. Codes are available at https://github.com/wkim97/FSR.

摘要: 由于特征层扰动的积累，深度神经网络容易受到对抗性攻击，许多工作通过去激活导致模型错误预测的非稳健特征激活来增强模型的稳健性。然而，我们声称这些恶意激活仍然包含歧视性提示，并且通过重新校准，它们可以捕获更多有用的信息来进行正确的模型预测。为此，我们提出了一种新的、易于插件的方法，称为特征分离和重新校准(FSR)，该方法通过分离和重新校准来重新校准恶意的、非健壮的激活以获得更健壮的特征映射。分离部分将输入特征映射分离为具有帮助模型做出正确预测的激活的健壮特征和具有导致敌方攻击时模型误预测的激活的非健壮特征。然后，重新校准部分调整非稳健激活以恢复模型预测的潜在有用线索。大量的实验验证了FSR与传统去激活技术相比的优越性，并证明了它以较小的计算开销提高了现有对抗性训练方法的健壮性高达8.57%。有关代码，请访问https://github.com/wkim97/FSR.



## **42. Near Optimal Adversarial Attack on UCB Bandits**

对UCB土匪的近最优敌意攻击 cs.LG

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2008.09312v3) [paper-pdf](http://arxiv.org/pdf/2008.09312v3)

**Authors**: Shiliang Zuo

**Abstract**: We consider a stochastic multi-arm bandit problem where rewards are subject to adversarial corruption. We propose a novel attack strategy that manipulates a UCB principle into pulling some non-optimal target arm $T - o(T)$ times with a cumulative cost that scales as $\sqrt{\log T}$, where $T$ is the number of rounds. We also prove the first lower bound on the cumulative attack cost. Our lower bound matches our upper bound up to $\log \log T$ factors, showing our attack to be near optimal.

摘要: 我们考虑了一个随机多臂强盗问题，其中报酬服从对抗性腐败。我们提出了一种新的攻击策略，它利用UCB原理来拉动一些非最优目标臂$T-o(T)$次，累积代价可扩展到$\Sqrt{\log T}$，其中$T$是轮数。我们还证明了累积攻击代价的第一个下界。我们的下界与上界匹配，最高可达$\log\log T$因子，表明我们的攻击接近最优。



## **43. RamBoAttack: A Robust Query Efficient Deep Neural Network Decision Exploit**

RamBoAttack：一种稳健查询高效的深度神经网络决策开发 cs.LG

Published in Network and Distributed System Security (NDSS) Symposium  2022. Code is available at https://ramboattack.github.io/

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2112.05282v3) [paper-pdf](http://arxiv.org/pdf/2112.05282v3)

**Authors**: Viet Quoc Vo, Ehsan Abbasnejad, Damith C. Ranasinghe

**Abstract**: Machine learning models are critically susceptible to evasion attacks from adversarial examples. Generally, adversarial examples, modified inputs deceptively similar to the original input, are constructed under whitebox settings by adversaries with full access to the model. However, recent attacks have shown a remarkable reduction in query numbers to craft adversarial examples using blackbox attacks. Particularly, alarming is the ability to exploit the classification decision from the access interface of a trained model provided by a growing number of Machine Learning as a Service providers including Google, Microsoft, IBM and used by a plethora of applications incorporating these models. The ability of an adversary to exploit only the predicted label from a model to craft adversarial examples is distinguished as a decision-based attack. In our study, we first deep dive into recent state-of-the-art decision-based attacks in ICLR and SP to highlight the costly nature of discovering low distortion adversarial employing gradient estimation methods. We develop a robust query efficient attack capable of avoiding entrapment in a local minimum and misdirection from noisy gradients seen in gradient estimation methods. The attack method we propose, RamBoAttack, exploits the notion of Randomized Block Coordinate Descent to explore the hidden classifier manifold, targeting perturbations to manipulate only localized input features to address the issues of gradient estimation methods. Importantly, the RamBoAttack is more robust to the different sample inputs available to an adversary and the targeted class. Overall, for a given target class, RamBoAttack is demonstrated to be more robust at achieving a lower distortion within a given query budget. We curate our extensive results using the large-scale high-resolution ImageNet dataset and open-source our attack, test samples and artifacts on GitHub.

摘要: 机器学习模型极易受到来自对手例子的逃避攻击。通常，对抗性的例子，修改后的输入欺骗性地类似于原始输入，由具有完全访问模型的敌手在白盒设置下构造。然而，最近的攻击显示，使用黑盒攻击构建敌意例子的查询数量显著减少。特别是，警报是利用由包括谷歌、微软、IBM在内的越来越多的机器学习即服务提供商提供的训练模型的访问接口的分类决策的能力，并被结合这些模型的大量应用程序使用。对手仅利用模型中预测的标签来制作敌意示例的能力被区分为基于决策的攻击。在我们的研究中，我们首先深入研究了ICLR和SP中最新的基于决策的攻击，以强调使用梯度估计方法发现低失真攻击的代价。我们开发了一种健壮的查询高效攻击，能够避免陷入局部最小值和从梯度估计方法中看到的噪声梯度的误导。我们提出的攻击方法RamBoAttack利用随机化块坐标下降的概念来探索隐藏的分类器流形，针对扰动只操纵局部输入特征来解决梯度估计方法的问题。重要的是，RamBoAttack对于对手和目标类可用的不同样本输入更加健壮。总体而言，对于给定的目标类，RamBoAttack被证明在给定的查询预算内实现较低的失真方面更加健壮。我们使用大规模高分辨率ImageNet数据集和在GitHub上开源的我们的攻击、测试样本和人工制品来管理我们广泛的结果。



## **44. Query Efficient Decision Based Sparse Attacks Against Black-Box Deep Learning Models**

基于查询高效决策的黑盒深度学习模型稀疏攻击 cs.LG

Published as a conference paper at the International Conference on  Learning Representations (ICLR 2022). Code is available at  https://sparseevoattack.github.io/

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2202.00091v2) [paper-pdf](http://arxiv.org/pdf/2202.00091v2)

**Authors**: Viet Quoc Vo, Ehsan Abbasnejad, Damith C. Ranasinghe

**Abstract**: Despite our best efforts, deep learning models remain highly vulnerable to even tiny adversarial perturbations applied to the inputs. The ability to extract information from solely the output of a machine learning model to craft adversarial perturbations to black-box models is a practical threat against real-world systems, such as autonomous cars or machine learning models exposed as a service (MLaaS). Of particular interest are sparse attacks. The realization of sparse attacks in black-box models demonstrates that machine learning models are more vulnerable than we believe. Because these attacks aim to minimize the number of perturbed pixels measured by l_0 norm-required to mislead a model by solely observing the decision (the predicted label) returned to a model query; the so-called decision-based attack setting. But, such an attack leads to an NP-hard optimization problem. We develop an evolution-based algorithm-SparseEvo-for the problem and evaluate against both convolutional deep neural networks and vision transformers. Notably, vision transformers are yet to be investigated under a decision-based attack setting. SparseEvo requires significantly fewer model queries than the state-of-the-art sparse attack Pointwise for both untargeted and targeted attacks. The attack algorithm, although conceptually simple, is also competitive with only a limited query budget against the state-of-the-art gradient-based whitebox attacks in standard computer vision tasks such as ImageNet. Importantly, the query efficient SparseEvo, along with decision-based attacks, in general, raise new questions regarding the safety of deployed systems and poses new directions to study and understand the robustness of machine learning models.

摘要: 尽管我们尽了最大努力，深度学习模型仍然非常容易受到应用于输入的微小对抗性扰动的影响。仅从机器学习模型的输出中提取信息以对黑盒模型进行敌意扰动的能力，对现实世界的系统构成了实际威胁，例如自动驾驶汽车或暴露为服务的机器学习模型(MLaaS)。特别令人感兴趣的是稀疏攻击。稀疏攻击在黑盒模型中的实现表明，机器学习模型比我们认为的更容易受到攻击。因为这些攻击的目的是最小化由l_0范数测量的扰动像素数-通过仅观察返回给模型查询的决策(预测标签)来误导模型所需的；所谓的基于决策的攻击设置。但是，这样的攻击导致了一个NP-Hard优化问题。我们开发了一种基于进化的算法SparseEvo来解决该问题，并对卷积深度神经网络和视觉转换器进行了评估。值得注意的是，视觉变形器尚未在基于决策的攻击环境下进行调查。对于非目标攻击和目标攻击，SparseEvo需要的模型查询比最先进的稀疏攻击点要少得多。攻击算法虽然在概念上很简单，但与标准计算机视觉任务(如ImageNet)中最先进的基于梯度的白盒攻击相比，仅有有限的查询预算也具有竞争力。重要的是，查询效率高的SparseEvo以及基于决策的攻击通常对已部署系统的安全性提出了新的问题，并为研究和理解机器学习模型的健壮性提供了新的方向。



## **45. CBA: Contextual Background Attack against Optical Aerial Detection in the Physical World**

CBA：物理世界中对光学空中探测的背景攻击 cs.CV

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2302.13519v3) [paper-pdf](http://arxiv.org/pdf/2302.13519v3)

**Authors**: Jiawei Lian, Xiaofei Wang, Yuru Su, Mingyang Ma, Shaohui Mei

**Abstract**: Patch-based physical attacks have increasingly aroused concerns.   However, most existing methods focus on obscuring targets captured on the ground, and some of these methods are simply extended to deceive aerial detectors.   They smear the targeted objects in the physical world with the elaborated adversarial patches, which can only slightly sway the aerial detectors' prediction and with weak attack transferability.   To address the above issues, we propose to perform Contextual Background Attack (CBA), a novel physical attack framework against aerial detection, which can achieve strong attack efficacy and transferability in the physical world even without smudging the interested objects at all.   Specifically, the targets of interest, i.e. the aircraft in aerial images, are adopted to mask adversarial patches.   The pixels outside the mask area are optimized to make the generated adversarial patches closely cover the critical contextual background area for detection, which contributes to gifting adversarial patches with more robust and transferable attack potency in the real world.   To further strengthen the attack performance, the adversarial patches are forced to be outside targets during training, by which the detected objects of interest, both on and outside patches, benefit the accumulation of attack efficacy.   Consequently, the sophisticatedly designed patches are gifted with solid fooling efficacy against objects both on and outside the adversarial patches simultaneously.   Extensive proportionally scaled experiments are performed in physical scenarios, demonstrating the superiority and potential of the proposed framework for physical attacks.   We expect that the proposed physical attack method will serve as a benchmark for assessing the adversarial robustness of diverse aerial detectors and defense methods.

摘要: 基于补丁的物理攻击越来越引起人们的关注。然而，现有的大多数方法都集中在遮挡地面捕获的目标上，其中一些方法只是简单地扩展到欺骗航空探测器。他们用精心制作的对抗性补丁涂抹物理世界中的目标对象，这只能轻微动摇航空探测器的预测，攻击可转移性较弱。为了解决上述问题，我们提出了一种新的针对空中探测的物理攻击框架--上下文背景攻击(CBA)，该框架即使在不玷污感兴趣对象的情况下也可以在物理世界中实现强大的攻击效能和可转移性。具体地说，采用感兴趣的目标，即航空图像中的飞机来掩盖敌方补丁。对掩码区域外的像素进行了优化，使生成的对抗性补丁紧密覆盖关键背景区域进行检测，有助于在现实世界中赋予对抗性补丁更健壮和可转移的攻击能力。为了进一步增强攻击性能，在训练过程中将对抗性补丁强制为外部目标，这样无论是在补丁上还是在补丁外，检测到的感兴趣对象都有利于攻击效能的积累。因此，复杂设计的补丁被赋予了对敌方补丁内外的对象同时具有可靠的愚弄效果。在物理场景中进行了广泛的按比例扩展的实验，展示了所提出的框架在物理攻击方面的优势和潜力。我们期望所提出的物理攻击方法将作为评估不同空中探测器和防御方法的对抗健壮性的基准。



## **46. TrojViT: Trojan Insertion in Vision Transformers**

TrojViT：视觉变形金刚中的特洛伊木马插入 cs.LG

10 pages, 4 figures, 11 tables

**SubmitDate**: 2023-03-23    [abs](http://arxiv.org/abs/2208.13049v3) [paper-pdf](http://arxiv.org/pdf/2208.13049v3)

**Authors**: Mengxin Zheng, Qian Lou, Lei Jiang

**Abstract**: Vision Transformers (ViTs) have demonstrated the state-of-the-art performance in various vision-related tasks. The success of ViTs motivates adversaries to perform backdoor attacks on ViTs. Although the vulnerability of traditional CNNs to backdoor attacks is well-known, backdoor attacks on ViTs are seldom-studied. Compared to CNNs capturing pixel-wise local features by convolutions, ViTs extract global context information through patches and attentions. Na\"ively transplanting CNN-specific backdoor attacks to ViTs yields only a low clean data accuracy and a low attack success rate. In this paper, we propose a stealth and practical ViT-specific backdoor attack $TrojViT$. Rather than an area-wise trigger used by CNN-specific backdoor attacks, TrojViT generates a patch-wise trigger designed to build a Trojan composed of some vulnerable bits on the parameters of a ViT stored in DRAM memory through patch salience ranking and attention-target loss. TrojViT further uses minimum-tuned parameter update to reduce the bit number of the Trojan. Once the attacker inserts the Trojan into the ViT model by flipping the vulnerable bits, the ViT model still produces normal inference accuracy with benign inputs. But when the attacker embeds a trigger into an input, the ViT model is forced to classify the input to a predefined target class. We show that flipping only few vulnerable bits identified by TrojViT on a ViT model using the well-known RowHammer can transform the model into a backdoored one. We perform extensive experiments of multiple datasets on various ViT models. TrojViT can classify $99.64\%$ of test images to a target class by flipping $345$ bits on a ViT for ImageNet.

摘要: 视觉变形金刚(VITS)在各种与视觉相关的任务中展示了最先进的性能。VITS的成功促使对手对VITS进行后门攻击。虽然传统的CNN对后门攻击的脆弱性是众所周知的，但对VITS的后门攻击很少被研究。与通过卷积获取像素级局部特征的CNN相比，VITS通过块和关注点来提取全局上下文信息。将CNN特定的后门攻击活生生地移植到VITS只会产生低的干净数据准确性和低的攻击成功率。在本文中，我们提出了一种隐形和实用的特定于VIT的后门攻击$TrojViT$。与CNN特定后门攻击使用的区域触发不同，TrojViT生成修补程序触发，旨在通过修补程序显著程度排名和注意力目标丢失来构建由存储在DRAM内存中的VIT参数上的一些易受攻击位组成的特洛伊木马程序。TrojViT进一步使用最小调整的参数更新来减少特洛伊木马的比特数。一旦攻击者通过翻转易受攻击的比特将特洛伊木马程序插入到VIT模型中，VIT模型仍然会使用良性输入产生正常的推理准确性。但是，当攻击者将触发器嵌入到输入中时，VIT模型被迫将输入分类到预定义的目标类。我们表明，只需使用著名的RowHammer在VIT模型上翻转TrojViT识别的少数易受攻击的位，就可以将该模型转换为后置模型。我们在不同的VIT模型上对多个数据集进行了广泛的实验。TrojViT可以通过在ImageNet的VIT上翻转$345$比特，将$99.64\$测试图像分类到目标类别。



## **47. Adversarial Robustness and Feature Impact Analysis for Driver Drowsiness Detection**

驾驶员嗜睡检测的对抗稳健性和特征影响分析 cs.LG

10 pages, 2 tables, 3 figures, AIME 2023 conference

**SubmitDate**: 2023-03-23    [abs](http://arxiv.org/abs/2303.13649v1) [paper-pdf](http://arxiv.org/pdf/2303.13649v1)

**Authors**: João Vitorino, Lourenço Rodrigues, Eva Maia, Isabel Praça, André Lourenço

**Abstract**: Drowsy driving is a major cause of road accidents, but drivers are dismissive of the impact that fatigue can have on their reaction times. To detect drowsiness before any impairment occurs, a promising strategy is using Machine Learning (ML) to monitor Heart Rate Variability (HRV) signals. This work presents multiple experiments with different HRV time windows and ML models, a feature impact analysis using Shapley Additive Explanations (SHAP), and an adversarial robustness analysis to assess their reliability when processing faulty input data and perturbed HRV signals. The most reliable model was Extreme Gradient Boosting (XGB) and the optimal time window had between 120 and 150 seconds. Furthermore, SHAP enabled the selection of the 18 most impactful features and the training of new smaller models that achieved a performance as good as the initial ones. Despite the susceptibility of all models to adversarial attacks, adversarial training enabled them to preserve significantly higher results, especially XGB. Therefore, ML models can significantly benefit from realistic adversarial training to provide a more robust driver drowsiness detection.

摘要: 疲劳驾驶是交通事故的一个主要原因，但司机们对疲劳对他们的反应时间的影响不屑一顾。为了在任何损害发生之前检测到昏昏欲睡，一个有希望的策略是使用机器学习(ML)来监测心率变异性(HRV)信号。这项工作给出了不同的HRV时间窗和ML模型的多个实验，使用Shapley Additive Informance(Shap)的特征影响分析，以及当处理错误的输入数据和扰动的HRV信号时评估它们的可靠性的对抗性稳健性分析。最可靠的模型是极端梯度增强(XGB)，最佳时间窗口在120到150秒之间。此外，Shap能够选择18个最有影响力的特征，并训练新的较小的模型，这些模型的表现与最初的模型一样好。尽管所有模型都容易受到对抗性攻击，但对抗性训练使它们能够保持显著更高的结果，特别是XGB。因此，ML模型可以显著受益于现实的对抗性训练，以提供更健壮的驾驶员嗜睡检测。



## **48. Efficient Symbolic Reasoning for Neural-Network Verification**

用于神经网络验证的高效符号推理 cs.AI

**SubmitDate**: 2023-03-23    [abs](http://arxiv.org/abs/2303.13588v1) [paper-pdf](http://arxiv.org/pdf/2303.13588v1)

**Authors**: Zi Wang, Somesh Jha, Krishnamurthy, Dvijotham

**Abstract**: The neural network has become an integral part of modern software systems. However, they still suffer from various problems, in particular, vulnerability to adversarial attacks. In this work, we present a novel program reasoning framework for neural-network verification, which we refer to as symbolic reasoning. The key components of our framework are the use of the symbolic domain and the quadratic relation. The symbolic domain has very flexible semantics, and the quadratic relation is quite expressive. They allow us to encode many verification problems for neural networks as quadratic programs. Our scheme then relaxes the quadratic programs to semidefinite programs, which can be efficiently solved. This framework allows us to verify various neural-network properties under different scenarios, especially those that appear challenging for non-symbolic domains. Moreover, it introduces new representations and perspectives for the verification tasks. We believe that our framework can bring new theoretical insights and practical tools to verification problems for neural networks.

摘要: 神经网络已经成为现代软件系统不可或缺的一部分。然而，它们仍然面临着各种问题，特别是易受对抗性攻击。在这项工作中，我们提出了一种新的神经网络验证程序推理框架，我们称之为符号推理。该框架的关键部分是符号域和二次关系的使用。符号域具有非常灵活的语义，二次关系具有很强的表现力。它们允许我们将神经网络的许多验证问题编码为二次规划。然后，我们的方案将二次规划松弛为半定规划，从而可以有效地求解。这个框架允许我们在不同的场景下验证各种神经网络属性，特别是那些对非符号域来说具有挑战性的场景。此外，它还介绍了核查任务的新表述和新视角。我们相信，我们的框架可以为神经网络的验证问题带来新的理论见解和实用工具。



## **49. Symmetries, flat minima, and the conserved quantities of gradient flow**

对称性、平坦极小值和梯度流的守恒量 cs.LG

To appear at ICLR 2023

**SubmitDate**: 2023-03-23    [abs](http://arxiv.org/abs/2210.17216v2) [paper-pdf](http://arxiv.org/pdf/2210.17216v2)

**Authors**: Bo Zhao, Iordan Ganev, Robin Walters, Rose Yu, Nima Dehmamy

**Abstract**: Empirical studies of the loss landscape of deep networks have revealed that many local minima are connected through low-loss valleys. Yet, little is known about the theoretical origin of such valleys. We present a general framework for finding continuous symmetries in the parameter space, which carve out low-loss valleys. Our framework uses equivariances of the activation functions and can be applied to different layer architectures. To generalize this framework to nonlinear neural networks, we introduce a novel set of nonlinear, data-dependent symmetries. These symmetries can transform a trained model such that it performs similarly on new samples, which allows ensemble building that improves robustness under certain adversarial attacks. We then show that conserved quantities associated with linear symmetries can be used to define coordinates along low-loss valleys. The conserved quantities help reveal that using common initialization methods, gradient flow only explores a small part of the global minimum. By relating conserved quantities to convergence rate and sharpness of the minimum, we provide insights on how initialization impacts convergence and generalizability.

摘要: 对深层网络损失格局的实证研究表明，许多局部极小值通过低损失谷连接在一起。然而，人们对这种山谷的理论起源知之甚少。我们给出了一个在参数空间中寻找连续对称性的一般框架，它划出了低损失谷。我们的框架使用了激活函数的等价性，可以应用于不同的层体系结构。为了将这个框架推广到非线性神经网络，我们引入了一组新的非线性、依赖于数据的对称性。这些对称性可以转换训练好的模型，使其在新样本上执行类似的操作，这使得集成构建能够提高在某些对手攻击下的健壮性。然后，我们证明了与线性对称有关的守恒量可以用来定义沿低损耗山谷的坐标。守恒量有助于揭示，使用常见的初始化方法，梯度流只探索全局极小值的一小部分。通过将守恒量与最小值的收敛速度和锐度联系起来，我们提供了关于初始化如何影响收敛和泛化的见解。



## **50. Decentralized Adversarial Training over Graphs**

基于图的分散对抗性训练 cs.LG

arXiv admin note: text overlap with arXiv:2303.01936

**SubmitDate**: 2023-03-23    [abs](http://arxiv.org/abs/2303.13326v1) [paper-pdf](http://arxiv.org/pdf/2303.13326v1)

**Authors**: Ying Cao, Elsa Rizk, Stefan Vlaski, Ali H. Sayed

**Abstract**: The vulnerability of machine learning models to adversarial attacks has been attracting considerable attention in recent years. Most existing studies focus on the behavior of stand-alone single-agent learners. In comparison, this work studies adversarial training over graphs, where individual agents are subjected to perturbations of varied strength levels across space. It is expected that interactions by linked agents, and the heterogeneity of the attack models that are possible over the graph, can help enhance robustness in view of the coordination power of the group. Using a min-max formulation of diffusion learning, we develop a decentralized adversarial training framework for multi-agent systems. We analyze the convergence properties of the proposed scheme for both convex and non-convex environments, and illustrate the enhanced robustness to adversarial attacks.

摘要: 近年来，机器学习模型对敌意攻击的脆弱性引起了人们的极大关注。现有的研究大多集中在单智能体学习者的行为上。相比之下，这项工作研究的是图上的对抗性训练，在图中，单个代理人受到空间上不同强度水平的扰动。考虑到组的协调能力，预计链接代理的交互以及图上可能的攻击模型的异构性可以帮助增强稳健性。利用扩散学习的最小-最大公式，我们提出了一种多智能体系统的分布式对抗训练框架。我们分析了该方案在凸环境和非凸环境下的收敛特性，并说明了该方案增强了对敌意攻击的鲁棒性。



