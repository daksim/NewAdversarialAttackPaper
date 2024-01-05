# Latest Adversarial Attack Papers
**update at 2024-01-05 10:01:00**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Mining Temporal Attack Patterns from Cyberthreat Intelligence Reports**

从网络威胁情报报告中挖掘时态攻击模式 cs.CR

A modified version of this pre-print is submitted to IEEE  Transactions on Software Engineering, and is under review

**SubmitDate**: 2024-01-03    [abs](http://arxiv.org/abs/2401.01883v1) [paper-pdf](http://arxiv.org/pdf/2401.01883v1)

**Authors**: Md Rayhanur Rahman, Brandon Wroblewski, Quinn Matthews, Brantley Morgan, Tim Menzies, Laurie Williams

**Abstract**: Defending from cyberattacks requires practitioners to operate on high-level adversary behavior. Cyberthreat intelligence (CTI) reports on past cyberattack incidents describe the chain of malicious actions with respect to time. To avoid repeating cyberattack incidents, practitioners must proactively identify and defend against recurring chain of actions - which we refer to as temporal attack patterns. Automatically mining the patterns among actions provides structured and actionable information on the adversary behavior of past cyberattacks. The goal of this paper is to aid security practitioners in prioritizing and proactive defense against cyberattacks by mining temporal attack patterns from cyberthreat intelligence reports. To this end, we propose ChronoCTI, an automated pipeline for mining temporal attack patterns from cyberthreat intelligence (CTI) reports of past cyberattacks. To construct ChronoCTI, we build the ground truth dataset of temporal attack patterns and apply state-of-the-art large language models, natural language processing, and machine learning techniques. We apply ChronoCTI on a set of 713 CTI reports, where we identify 124 temporal attack patterns - which we categorize into nine pattern categories. We identify that the most prevalent pattern category is to trick victim users into executing malicious code to initiate the attack, followed by bypassing the anti-malware system in the victim network. Based on the observed patterns, we advocate organizations to train users about cybersecurity best practices, introduce immutable operating systems with limited functionalities, and enforce multi-user authentications. Moreover, we advocate practitioners to leverage the automated mining capability of ChronoCTI and design countermeasures against the recurring attack patterns.

摘要: 防御网络攻击需要从业者对高级别的对手行为进行操作。关于过去网络攻击事件的网络威胁情报(CTI)报告描述了与时间相关的恶意行动链。为了避免重复网络攻击事件，从业人员必须主动识别和防御反复出现的动作链--我们将其称为临时攻击模式。自动挖掘动作之间的模式提供了关于过去网络攻击对手行为的结构化和可操作的信息。本文的目的是通过从网络威胁情报报告中挖掘时态攻击模式，帮助安全从业者确定优先顺序并主动防御网络攻击。为此，我们提出了ChronoCTI，这是一种从过去网络攻击的网络威胁情报(CTI)报告中挖掘时态攻击模式的自动化管道。为了构建ChronoCTI，我们建立了时间攻击模式的地面事实数据集，并应用了最先进的大型语言模型、自然语言处理和机器学习技术。我们在一组713个CTI报告上应用了ChronoCTI，其中我们识别了124个临时攻击模式-我们将其分类为9个模式类别。我们发现，最普遍的模式类别是诱骗受害者用户执行恶意代码来发起攻击，然后绕过受害者网络中的反恶意软件系统。根据观察到的模式，我们倡导组织对用户进行网络安全最佳实践方面的培训，引入具有有限功能的不变操作系统，并强制实施多用户身份验证。此外，我们提倡实践者利用ChronoCTI的自动挖掘能力，并设计针对反复出现的攻击模式的对策。



## **2. Attackers reveal their arsenal: An investigation of adversarial techniques in CTI reports**

攻击者揭示他们的武器库：CTI报告中对抗性技术的调查 cs.CR

This version is submitted to ACM Transactions on Privacy and  Security. This version is under review

**SubmitDate**: 2024-01-03    [abs](http://arxiv.org/abs/2401.01865v1) [paper-pdf](http://arxiv.org/pdf/2401.01865v1)

**Authors**: Md Rayhanur Rahman, Setu Kumar Basak, Rezvan Mahdavi Hezaveh, Laurie Williams

**Abstract**: Context: Cybersecurity vendors often publish cyber threat intelligence (CTI) reports, referring to the written artifacts on technical and forensic analysis of the techniques used by the malware in APT attacks. Objective: The goal of this research is to inform cybersecurity practitioners about how adversaries form cyberattacks through an analysis of adversarial techniques documented in cyberthreat intelligence reports. Dataset: We use 594 adversarial techniques cataloged in MITRE ATT\&CK. We systematically construct a set of 667 CTI reports that MITRE ATT\&CK used as citations in the descriptions of the cataloged adversarial techniques. Methodology: We analyze the frequency and trend of adversarial techniques, followed by a qualitative analysis of the implementation of techniques. Next, we perform association rule mining to identify pairs of techniques recurring in APT attacks. We then perform qualitative analysis to identify the underlying relations among the techniques in the recurring pairs. Findings: The set of 667 CTI reports documents 10,370 techniques in total, and we identify 19 prevalent techniques accounting for 37.3\% of documented techniques. We also identify 425 statistically significant recurring pairs and seven types of relations among the techniques in these pairs. The top three among the seven relationships suggest that techniques used by the malware inter-relate with one another in terms of (a) abusing or affecting the same system assets, (b) executing in sequences, and (c) overlapping in their implementations. Overall, the study quantifies how adversaries leverage techniques through malware in APT attacks based on publicly reported documents. We advocate organizations prioritize their defense against the identified prevalent techniques and actively hunt for potential malicious intrusion based on the identified pairs of techniques.

摘要: 背景：网络安全供应商经常发布网络威胁情报(CTI)报告，提到对恶意软件在APT攻击中使用的技术进行技术和法医分析的书面人工制品。目标：本研究的目标是通过分析网络威胁情报报告中记录的对抗技术，向网络安全从业者提供有关对手如何形成网络攻击的信息。数据集：我们使用了MITRE ATT-CK收录的594种对抗性技术。我们系统地构建了一组667篇CTI报告，MITRE ATT-CK在编目对抗技术的描述中引用了这些报告。方法：我们分析对抗性技术的频率和趋势，然后对技术的实施进行定性分析。接下来，我们执行关联规则挖掘来识别在APT攻击中重复出现的技术对。然后，我们进行定性分析，以确定循环对中的技术之间的潜在关系。结果：这组667份CTI报告共记录了10,370种技术，我们确定了19种流行技术，占记录技术的37.3%。我们还确定了425个具有统计意义的重复对，以及这些对中的技术之间的七种类型的关系。七种关系中的前三种表明恶意软件使用的技术在以下方面相互关联：(A)滥用或影响相同的系统资产，(B)按顺序执行，以及(C)实现中的重叠。总体而言，这项研究基于公开报道的文件，量化了对手如何通过恶意软件在APT攻击中利用技术。我们主张组织优先防御已识别的流行技术，并根据已识别的技术对积极寻找潜在的恶意入侵。



## **3. Towards Robust Semantic Segmentation against Patch-based Attack via Attention Refinement**

基于注意力精化的抗补丁攻击的稳健语义分割 cs.CV

30 pages, 3 figures, 12 tables

**SubmitDate**: 2024-01-03    [abs](http://arxiv.org/abs/2401.01750v1) [paper-pdf](http://arxiv.org/pdf/2401.01750v1)

**Authors**: Zheng Yuan, Jie Zhang, Yude Wang, Shiguang Shan, Xilin Chen

**Abstract**: The attention mechanism has been proven effective on various visual tasks in recent years. In the semantic segmentation task, the attention mechanism is applied in various methods, including the case of both Convolution Neural Networks (CNN) and Vision Transformer (ViT) as backbones. However, we observe that the attention mechanism is vulnerable to patch-based adversarial attacks. Through the analysis of the effective receptive field, we attribute it to the fact that the wide receptive field brought by global attention may lead to the spread of the adversarial patch. To address this issue, in this paper, we propose a Robust Attention Mechanism (RAM) to improve the robustness of the semantic segmentation model, which can notably relieve the vulnerability against patch-based attacks. Compared to the vallina attention mechanism, RAM introduces two novel modules called Max Attention Suppression and Random Attention Dropout, both of which aim to refine the attention matrix and limit the influence of a single adversarial patch on the semantic segmentation results of other positions. Extensive experiments demonstrate the effectiveness of our RAM to improve the robustness of semantic segmentation models against various patch-based attack methods under different attack settings.

摘要: 近年来，注意机制在各种视觉任务中被证明是有效的。在语义分割任务中，注意力机制被应用于各种方法，包括卷积神经网络(CNN)和视觉转换器(VIT)作为骨干的情况。然而，我们观察到注意机制很容易受到基于补丁的对抗性攻击。通过对有效接受场的分析，我们将其归因于全球注意带来的广泛接受场可能导致对抗性斑块的传播。针对这一问题，本文提出了一种健壮的注意力机制(RAM)来提高语义分割模型的健壮性，该机制可以显著缓解语义分割模型对基于补丁攻击的脆弱性。与Vallina注意机制相比，RAM引入了两个新的模块：最大注意抑制和随机注意丢弃，这两个模块的目的都是为了细化注意矩阵，并限制单个敌意补丁对其他位置语义分割结果的影响。大量实验表明，在不同的攻击环境下，该算法能够有效地提高语义分割模型对各种基于补丁的攻击方法的稳健性。



## **4. An Initial Investigation of Neural Replay Simulator for Over-the-Air Adversarial Perturbations to Automatic Speaker Verification**

用于自动说话人确认的空中对抗扰动神经重放模拟器的初步研究 cs.SD

Accepted in ICASSP 2024

**SubmitDate**: 2024-01-03    [abs](http://arxiv.org/abs/2310.05354v4) [paper-pdf](http://arxiv.org/pdf/2310.05354v4)

**Authors**: Jiaqi Li, Li Wang, Liumeng Xue, Lei Wang, Zhizheng Wu

**Abstract**: Deep Learning has advanced Automatic Speaker Verification (ASV) in the past few years. Although it is known that deep learning-based ASV systems are vulnerable to adversarial examples in digital access, there are few studies on adversarial attacks in the context of physical access, where a replay process (i.e., over the air) is involved. An over-the-air attack involves a loudspeaker, a microphone, and a replaying environment that impacts the movement of the sound wave. Our initial experiment confirms that the replay process impacts the effectiveness of the over-the-air attack performance. This study performs an initial investigation towards utilizing a neural replay simulator to improve over-the-air adversarial attack robustness. This is achieved by using a neural waveform synthesizer to simulate the replay process when estimating the adversarial perturbations. Experiments conducted on the ASVspoof2019 dataset confirm that the neural replay simulator can considerably increase the success rates of over-the-air adversarial attacks. This raises the concern for adversarial attacks on speaker verification in physical access applications.

摘要: 在过去的几年里，深度学习发展了自动说话人确认(ASV)。虽然众所周知，基于深度学习的ASV系统在数字访问中容易受到敌意攻击，但在涉及重播过程(即空中重播)的物理访问环境中，很少有关于对抗性攻击的研究。空中攻击包括扬声器、麦克风和影响声波移动的重放环境。我们的初步实验证实，重放过程会影响空中攻击性能的有效性。本研究对利用神经重放模拟器来提高空中对抗攻击的稳健性进行了初步的研究。这是通过使用神经波形合成器来模拟在估计对抗性扰动时的重播过程来实现的。在ASVspoof2019数据集上进行的实验证实，神经重放模拟器可以显著提高空中对抗性攻击的成功率。这引起了人们对物理访问应用中说话人验证的对抗性攻击的关注。



## **5. Will 6G be Semantic Communications? Opportunities and Challenges from Task Oriented and Secure Communications to Integrated Sensing**

6G会成为语义通信吗？从任务导向和安全通信到集成传感的机遇和挑战 cs.NI

**SubmitDate**: 2024-01-03    [abs](http://arxiv.org/abs/2401.01531v1) [paper-pdf](http://arxiv.org/pdf/2401.01531v1)

**Authors**: Yalin E. Sagduyu, Tugba Erpek, Aylin Yener, Sennur Ulukus

**Abstract**: This paper explores opportunities and challenges of task (goal)-oriented and semantic communications for next-generation (NextG) communication networks through the integration of multi-task learning. This approach employs deep neural networks representing a dedicated encoder at the transmitter and multiple task-specific decoders at the receiver, collectively trained to handle diverse tasks including semantic information preservation, source input reconstruction, and integrated sensing and communications. To extend the applicability from point-to-point links to multi-receiver settings, we envision the deployment of decoders at various receivers, where decentralized learning addresses the challenges of communication load and privacy concerns, leveraging federated learning techniques that distribute model updates across decentralized nodes. However, the efficacy of this approach is contingent on the robustness of the employed deep learning models. We scrutinize potential vulnerabilities stemming from adversarial attacks during both training and testing phases. These attacks aim to manipulate both the inputs at the encoder at the transmitter and the signals received over the air on the receiver side, highlighting the importance of fortifying semantic communications against potential multi-domain exploits. Overall, the joint and robust design of task-oriented communications, semantic communications, and integrated sensing and communications in a multi-task learning framework emerges as the key enabler for context-aware, resource-efficient, and secure communications ultimately needed in NextG network systems.

摘要: 通过整合多任务学习，探索面向任务(目标)和语义通信的下一代通信网络的机遇和挑战。这种方法采用深度神经网络，在发送端代表一个专用编码器，在接收端代表多个特定于任务的解码器，共同训练以处理包括语义信息保存、源输入重建以及集成传感和通信在内的各种任务。为了将适用性从点对点链路扩展到多接收器设置，我们设想在不同的接收器上部署解码器，其中分散学习利用跨分散节点分发模型更新的联合学习技术来解决通信负载和隐私问题的挑战。然而，这种方法的有效性取决于所采用的深度学习模型的稳健性。我们在培训和测试阶段仔细检查来自对抗性攻击的潜在漏洞。这些攻击的目的是同时操纵发送器编码器的输入和接收器端通过空中接收的信号，突显加强语义通信以抵御潜在的多域利用的重要性。总体而言，面向任务的通信、语义通信以及多任务学习框架中的集成传感和通信的联合稳健设计成为下一代网络系统最终需要的情景感知、资源高效和安全通信的关键推动因素。



## **6. JMA: a General Algorithm to Craft Nearly Optimal Targeted Adversarial Example**

JMA：一种生成近最优目标对抗性实例的通用算法 cs.LG

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2401.01199v1) [paper-pdf](http://arxiv.org/pdf/2401.01199v1)

**Authors**: Benedetta Tondi, Wei Guo, Mauro Barni

**Abstract**: Most of the approaches proposed so far to craft targeted adversarial examples against Deep Learning classifiers are highly suboptimal and typically rely on increasing the likelihood of the target class, thus implicitly focusing on one-hot encoding settings. In this paper, we propose a more general, theoretically sound, targeted attack that resorts to the minimization of a Jacobian-induced MAhalanobis distance (JMA) term, taking into account the effort (in the input space) required to move the latent space representation of the input sample in a given direction. The minimization is solved by exploiting the Wolfe duality theorem, reducing the problem to the solution of a Non-Negative Least Square (NNLS) problem. The proposed algorithm provides an optimal solution to a linearized version of the adversarial example problem originally introduced by Szegedy et al. \cite{szegedy2013intriguing}. The experiments we carried out confirm the generality of the proposed attack which is proven to be effective under a wide variety of output encoding schemes. Noticeably, the JMA attack is also effective in a multi-label classification scenario, being capable to induce a targeted modification of up to half the labels in a complex multilabel classification scenario with 20 labels, a capability that is out of reach of all the attacks proposed so far. As a further advantage, the JMA attack usually requires very few iterations, thus resulting more efficient than existing methods.

摘要: 到目前为止提出的针对深度学习分类器的目标对抗性样本的大多数方法都是高度次优的，通常依赖于增加目标类的可能性，因此隐式地专注于一热编码设置。在本文中，我们提出了一个更一般的，理论上合理的，有针对性的攻击，诉诸于雅克比诱导的马氏距离(JMA)项的最小化，考虑到在输入空间中将输入样本的潜在空间表示沿给定方向移动所需的努力。利用Wolfe对偶定理，将问题归结为一个非负最小二乘(NNLS)问题。该算法为Szegedy等人最初提出的对抗性例子问题的线性化版本提供了最优解。引用{szegedy2013intriguing}。我们进行的实验证实了该攻击的通用性，该攻击在多种输出编码方案下都被证明是有效的。值得注意的是，JMA攻击在多标签分类场景中也是有效的，能够在具有20个标签的复杂多标签分类场景中诱导多达一半的标签的有针对性的修改，这是迄今为止提出的所有攻击所无法达到的能力。作为另一个优势，JMA攻击通常只需要很少的迭代，因此比现有方法更有效。



## **7. Dual Teacher Knowledge Distillation with Domain Alignment for Face Anti-spoofing**

面向人脸反欺骗的领域对齐双教师知识蒸馏 cs.CV

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2401.01102v1) [paper-pdf](http://arxiv.org/pdf/2401.01102v1)

**Authors**: Zhe Kong, Wentian Zhang, Tao Wang, Kaihao Zhang, Yuexiang Li, Xiaoying Tang, Wenhan Luo

**Abstract**: Face recognition systems have raised concerns due to their vulnerability to different presentation attacks, and system security has become an increasingly critical concern. Although many face anti-spoofing (FAS) methods perform well in intra-dataset scenarios, their generalization remains a challenge. To address this issue, some methods adopt domain adversarial training (DAT) to extract domain-invariant features. However, the competition between the encoder and the domain discriminator can cause the network to be difficult to train and converge. In this paper, we propose a domain adversarial attack (DAA) method to mitigate the training instability problem by adding perturbations to the input images, which makes them indistinguishable across domains and enables domain alignment. Moreover, since models trained on limited data and types of attacks cannot generalize well to unknown attacks, we propose a dual perceptual and generative knowledge distillation framework for face anti-spoofing that utilizes pre-trained face-related models containing rich face priors. Specifically, we adopt two different face-related models as teachers to transfer knowledge to the target student model. The pre-trained teacher models are not from the task of face anti-spoofing but from perceptual and generative tasks, respectively, which implicitly augment the data. By combining both DAA and dual-teacher knowledge distillation, we develop a dual teacher knowledge distillation with domain alignment framework (DTDA) for face anti-spoofing. The advantage of our proposed method has been verified through extensive ablation studies and comparison with state-of-the-art methods on public datasets across multiple protocols.

摘要: 人脸识别系统由于易受不同表现形式的攻击而引起了人们的关注，系统安全已经成为一个越来越关键的问题。尽管许多Face反欺骗(FAS)方法在数据集内场景中执行得很好，但它们的泛化仍然是一个挑战。为了解决这一问题，一些方法采用领域对抗训练(DAT)来提取领域不变特征。然而，编码器和域鉴别器之间的竞争会导致网络难以训练和收敛。在本文中，我们提出了一种域对抗攻击(DAA)方法，通过在输入图像中添加扰动来缓解训练不稳定性问题，从而使输入图像无法跨域区分并实现域对齐。此外，由于在有限数据和攻击类型上训练的模型不能很好地推广到未知攻击，我们提出了一种双重感知和生成的人脸反欺骗知识蒸馏框架，该框架利用包含丰富人脸先验的预先训练的人脸相关模型。具体地说，我们采用了两种不同的面子相关模式作为教师向目标学生模式传递知识。预先训练的教师模型不是来自面子反恶搞任务，而是来自知觉任务和生成性任务，这两个任务分别内隐地增加了数据。将DAA和双师知识提炼相结合，提出了一种双师知识提炼的领域对齐框架(DTDA)，用于人脸防欺骗。我们提出的方法的优势已经通过广泛的消融研究以及在多个协议的公共数据集上与最新方法的比较得到了验证。



## **8. Imperio: Language-Guided Backdoor Attacks for Arbitrary Model Control**

Imperio：针对任意模型控制的恶意引导后门攻击 cs.CR

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2401.01085v1) [paper-pdf](http://arxiv.org/pdf/2401.01085v1)

**Authors**: Ka-Ho Chow, Wenqi Wei, Lei Yu

**Abstract**: Revolutionized by the transformer architecture, natural language processing (NLP) has received unprecedented attention. While advancements in NLP models have led to extensive research into their backdoor vulnerabilities, the potential for these advancements to introduce new backdoor threats remains unexplored. This paper proposes Imperio, which harnesses the language understanding capabilities of NLP models to enrich backdoor attacks. Imperio provides a new model control experience. It empowers the adversary to control the victim model with arbitrary output through language-guided instructions. This is achieved using a language model to fuel a conditional trigger generator, with optimizations designed to extend its language understanding capabilities to backdoor instruction interpretation and execution. Our experiments across three datasets, five attacks, and nine defenses confirm Imperio's effectiveness. It can produce contextually adaptive triggers from text descriptions and control the victim model with desired outputs, even in scenarios not encountered during training. The attack maintains a high success rate across complex datasets without compromising the accuracy of clean inputs and also exhibits resilience against representative defenses. The source code is available at \url{https://khchow.com/Imperio}.

摘要: 自然语言处理(NLP)受到了变压器体系结构的革命性变革，受到了前所未有的关注。虽然NLP模型的进步导致了对其后门漏洞的广泛研究，但这些进步引入新的后门威胁的潜力仍未被发掘。本文提出了Imperio，它利用NLP模型的语言理解能力来丰富后门攻击。Imperio提供了全新的模型控制体验。它使攻击者能够通过语言引导的指令控制具有任意输出的受害者模型。这是通过使用语言模型来为条件触发生成器提供燃料来实现的，优化旨在将其语言理解能力扩展到后门指令解释和执行。我们在三个数据集、五个攻击和九个防御系统上的实验证实了Imperio的有效性。它可以从文本描述中产生上下文自适应触发，并用所需的输出控制受害者模型，即使在培训期间没有遇到的情况下也是如此。该攻击在复杂的数据集上保持了高的成功率，而不会影响干净输入的准确性，并且对典型的防御系统也表现出了韧性。源代码可在\url{https://khchow.com/Imperio}.



## **9. Safety and Performance, Why Not Both? Bi-Objective Optimized Model Compression against Heterogeneous Attacks Toward AI Software Deployment**

安全和性能，为什么不能两者兼而有之呢？针对AI软件部署异构性攻击的双目标优化模型压缩 cs.AI

Accepted by IEEE Transactions on Software Engineering (TSE).  Camera-ready Version. arXiv admin note: substantial text overlap with  arXiv:2208.05969

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2401.00996v1) [paper-pdf](http://arxiv.org/pdf/2401.00996v1)

**Authors**: Jie Zhu, Leye Wang, Xiao Han, Anmin Liu, Tao Xie

**Abstract**: The size of deep learning models in artificial intelligence (AI) software is increasing rapidly, hindering the large-scale deployment on resource-restricted devices (e.g., smartphones). To mitigate this issue, AI software compression plays a crucial role, which aims to compress model size while keeping high performance. However, the intrinsic defects in a big model may be inherited by the compressed one. Such defects may be easily leveraged by adversaries, since a compressed model is usually deployed in a large number of devices without adequate protection. In this article, we aim to address the safe model compression problem from the perspective of safety-performance co-optimization. Specifically, inspired by the test-driven development (TDD) paradigm in software engineering, we propose a test-driven sparse training framework called SafeCompress. By simulating the attack mechanism as safety testing, SafeCompress can automatically compress a big model to a small one following the dynamic sparse training paradigm. Then, considering two kinds of representative and heterogeneous attack mechanisms, i.e., black-box membership inference attack and white-box membership inference attack, we develop two concrete instances called BMIA-SafeCompress and WMIA-SafeCompress. Further, we implement another instance called MMIA-SafeCompress by extending SafeCompress to defend against the occasion when adversaries conduct black-box and white-box membership inference attacks simultaneously. We conduct extensive experiments on five datasets for both computer vision and natural language processing tasks. The results show the effectiveness and generalizability of our framework. We also discuss how to adapt SafeCompress to other attacks besides membership inference attack, demonstrating the flexibility of SafeCompress.

摘要: 人工智能(AI)软件中的深度学习模型的规模正在迅速增长，阻碍了在资源受限的设备(如智能手机)上的大规模部署。为了缓解这个问题，人工智能软件压缩起到了至关重要的作用，其目标是在保持高性能的同时压缩模型大小。然而，大模型中的固有缺陷可能会被压缩的模型继承。这样的缺陷很容易被攻击者利用，因为压缩模型通常部署在大量设备中，而没有足够的保护。在本文中，我们旨在从安全-性能联合优化的角度解决安全模型压缩问题。具体地说，受软件工程中测试驱动开发(TDD)范式的启发，我们提出了一个称为SafeCompress的测试驱动稀疏训练框架。通过将攻击机制模拟为安全测试，SafeCompress可以按照动态稀疏训练范式自动将大模型压缩为小模型。然后，考虑到两种典型的异构性攻击机制，即黑盒成员关系推理攻击和白盒成员关系推理攻击，我们开发了两个具体的实例：BMIA-SafeCompress和WMIA-SafeCompress。此外，我们实现了另一个实例MMIA-SafeCompress，通过扩展SafeCompress来防御对手同时进行黑盒和白盒成员推理攻击的情况。我们在计算机视觉和自然语言处理任务的五个数据集上进行了广泛的实验。结果表明，该框架具有较好的通用性和有效性。我们还讨论了如何使SafeCompress适应除成员推理攻击之外的其他攻击，展示了SafeCompress的灵活性。



## **10. Adversarial Item Promotion on Visually-Aware Recommender Systems by Guided Diffusion**

基于引导扩散的视觉感知推荐系统中的对抗性项目提升 cs.IR

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2312.15826v3) [paper-pdf](http://arxiv.org/pdf/2312.15826v3)

**Authors**: Lijian Chen, Wei Yuan, Tong Chen, Guanhua Ye, Quoc Viet Hung Nguyen, Hongzhi Yin

**Abstract**: Visually-aware recommender systems have found widespread application in domains where visual elements significantly contribute to the inference of users' potential preferences. While the incorporation of visual information holds the promise of enhancing recommendation accuracy and alleviating the cold-start problem, it is essential to point out that the inclusion of item images may introduce substantial security challenges. Some existing works have shown that the item provider can manipulate item exposure rates to its advantage by constructing adversarial images. However, these works cannot reveal the real vulnerability of visually-aware recommender systems because (1) The generated adversarial images are markedly distorted, rendering them easily detectable by human observers; (2) The effectiveness of the attacks is inconsistent and even ineffective in some scenarios. To shed light on the real vulnerabilities of visually-aware recommender systems when confronted with adversarial images, this paper introduces a novel attack method, IPDGI (Item Promotion by Diffusion Generated Image). Specifically, IPDGI employs a guided diffusion model to generate adversarial samples designed to deceive visually-aware recommender systems. Taking advantage of accurately modeling benign images' distribution by diffusion models, the generated adversarial images have high fidelity with original images, ensuring the stealth of our IPDGI. To demonstrate the effectiveness of our proposed methods, we conduct extensive experiments on two commonly used e-commerce recommendation datasets (Amazon Beauty and Amazon Baby) with several typical visually-aware recommender systems. The experimental results show that our attack method has a significant improvement in both the performance of promoting the long-tailed (i.e., unpopular) items and the quality of generated adversarial images.

摘要: 视觉感知推荐系统在视觉元素对用户潜在偏好的推断有重要作用的领域得到了广泛的应用。虽然加入视觉信息有望提高推荐的准确性和缓解冷启动问题，但必须指出的是，纳入物品图像可能会带来重大的安全挑战。一些已有的研究表明，物品提供者可以通过构建对抗性图像来操纵物品曝光率。然而，这些工作并不能揭示视觉感知推荐系统的真正弱点，因为(1)生成的敌意图像明显失真，使得人类很容易发现它们；(2)攻击的有效性在某些场景下是不一致的，甚至无效的。为了揭示视觉感知推荐系统在面对敌意图像时的真正弱点，提出了一种新的攻击方法--IPDGI(Item Promotion By Diffumation Generated Image)。具体地说，IPDGI使用引导扩散模型来生成敌意样本，旨在欺骗视觉感知的推荐系统。利用扩散模型精确模拟良性图像的分布，生成的对抗性图像与原始图像具有较高的保真度，保证了IPDGI的隐蔽性。为了验证我们提出的方法的有效性，我们在两个常用的电子商务推荐数据集(Amazon Beauty和Amazon Baby)上进行了广泛的实验，并使用几个典型的视觉感知推荐系统进行了实验。实验结果表明，我们的攻击方法在提升长尾(即不受欢迎)项的性能和生成对抗性图像的质量方面都有显著的提高。



## **11. Passive Inference Attacks on Split Learning via Adversarial Regularization**

基于对抗正则化的分裂学习被动推理攻击 cs.CR

17 pages, 20 figures

**SubmitDate**: 2024-01-01    [abs](http://arxiv.org/abs/2310.10483v3) [paper-pdf](http://arxiv.org/pdf/2310.10483v3)

**Authors**: Xiaochen Zhu, Xinjian Luo, Yuncheng Wu, Yangfan Jiang, Xiaokui Xiao, Beng Chin Ooi

**Abstract**: Split Learning (SL) has emerged as a practical and efficient alternative to traditional federated learning. While previous attempts to attack SL have often relied on overly strong assumptions or targeted easily exploitable models, we seek to develop more practical attacks. We introduce SDAR, a novel attack framework against SL with an honest-but-curious server. SDAR leverages auxiliary data and adversarial regularization to learn a decodable simulator of the client's private model, which can effectively infer the client's private features under the vanilla SL, and both features and labels under the U-shaped SL. We perform extensive experiments in both configurations to validate the effectiveness of our proposed attacks. Notably, in challenging but practical scenarios where existing passive attacks struggle to reconstruct the client's private data effectively, SDAR consistently achieves attack performance comparable to active attacks. On CIFAR-10, at the deep split level of 7, SDAR achieves private feature reconstruction with less than 0.025 mean squared error in both the vanilla and the U-shaped SL, and attains a label inference accuracy of over 98% in the U-shaped setting, while existing attacks fail to produce non-trivial results.

摘要: 分裂学习(Split Learning，SL)已成为传统联合学习的一种实用有效的替代方案。虽然以前攻击SL的尝试通常依赖于过于强烈的假设或目标明确、易于利用的模型，但我们寻求开发更实际的攻击。我们介绍了SDAR，这是一种针对SL的新型攻击框架，具有诚实但好奇的服务器。SDAR利用辅助数据和对抗性正则化学习客户私有模型的可解码模拟器，该模拟器可以有效地推断客户在香草SL下的私有特征，以及U形SL下的特征和标签。我们在两种配置下都进行了大量的实验，以验证我们提出的攻击的有效性。值得注意的是，在具有挑战性但实用的场景中，现有的被动攻击难以有效地重建客户端的私有数据，SDAR始终实现与主动攻击相当的攻击性能。在CIFAR-10上，在7的深度分裂水平上，SDAR实现了私有特征重建，在普通SL和U形SL上的均方误差都小于0.025，在U形背景下获得了98%以上的标签推理准确率，而现有的攻击无法产生非平凡的结果。



## **12. Channel Reciprocity Attacks Using Intelligent Surfaces with Non-Diagonal Phase Shifts**

基于非对角线相移智能曲面的通道互易性攻击 eess.SP

**SubmitDate**: 2024-01-01    [abs](http://arxiv.org/abs/2309.11665v2) [paper-pdf](http://arxiv.org/pdf/2309.11665v2)

**Authors**: Haoyu Wang, Zhu Han, A. Lee Swindlehurst

**Abstract**: While reconfigurable intelligent surface (RIS) technology has been shown to provide numerous benefits to wireless systems, in the hands of an adversary such technology can also be used to disrupt communication links. This paper describes and analyzes an RIS-based attack on multi-antenna wireless systems that operate in time-division duplex mode under the assumption of channel reciprocity. In particular, we show how an RIS with a non-diagonal (ND) phase shift matrix (referred to here as an ND-RIS) can be deployed to maliciously break the channel reciprocity and hence degrade the downlink network performance. Such an attack is entirely passive and difficult to detect and counteract. We provide a theoretical analysis of the degradation in the sum ergodic rate that results when an arbitrary malicious ND-RIS is deployed and design an approach based on the genetic algorithm for optimizing the ND structure under partial knowledge of the available channel state information. Our simulation results validate the analysis and demonstrate that an ND-RIS channel reciprocity attack can dramatically reduce the downlink throughput.

摘要: 虽然可重构智能表面(RIS)技术已被证明为无线系统提供了许多好处，但在对手手中，这种技术也可能被用来中断通信链路。在信道互易性的假设下，描述和分析了一种基于RIS的对时分双工多天线无线系统的攻击。特别是，我们展示了如何部署具有非对角线(ND)相移矩阵的RIS(这里称为ND-RIS)来恶意破坏信道互易性，从而降低下行链路网络的性能。这种攻击完全是被动的，很难发现和反击。从理论上分析了任意恶意ND-RIS部署后导致的和遍历率的下降，并设计了一种在部分已知信道状态信息的情况下基于遗传算法的ND结构优化方法。我们的仿真结果验证了分析，并证明了ND-RIS信道互惠攻击可以显著降低下行链路吞吐量。



## **13. Is It Possible to Backdoor Face Forgery Detection with Natural Triggers?**

是否有可能使用自然触发器进行后门人脸伪造检测？ cs.CV

**SubmitDate**: 2023-12-31    [abs](http://arxiv.org/abs/2401.00414v1) [paper-pdf](http://arxiv.org/pdf/2401.00414v1)

**Authors**: Xiaoxuan Han, Songlin Yang, Wei Wang, Ziwen He, Jing Dong

**Abstract**: Deep neural networks have significantly improved the performance of face forgery detection models in discriminating Artificial Intelligent Generated Content (AIGC). However, their security is significantly threatened by the injection of triggers during model training (i.e., backdoor attacks). Although existing backdoor defenses and manual data selection can mitigate those using human-eye-sensitive triggers, such as patches or adversarial noises, the more challenging natural backdoor triggers remain insufficiently researched. To further investigate natural triggers, we propose a novel analysis-by-synthesis backdoor attack against face forgery detection models, which embeds natural triggers in the latent space. We thoroughly study such backdoor vulnerability from two perspectives: (1) Model Discrimination (Optimization-Based Trigger): we adopt a substitute detection model and find the trigger by minimizing the cross-entropy loss; (2) Data Distribution (Custom Trigger): we manipulate the uncommon facial attributes in the long-tailed distribution to generate poisoned samples without the supervision from detection models. Furthermore, to completely evaluate the detection models towards the latest AIGC, we utilize both state-of-the-art StyleGAN and Stable Diffusion for trigger generation. Finally, these backdoor triggers introduce specific semantic features to the generated poisoned samples (e.g., skin textures and smile), which are more natural and robust. Extensive experiments show that our method is superior from three levels: (1) Attack Success Rate: ours achieves a high attack success rate (over 99%) and incurs a small model accuracy drop (below 0.2%) with a low poisoning rate (less than 3%); (2) Backdoor Defense: ours shows better robust performance when faced with existing backdoor defense methods; (3) Human Inspection: ours is less human-eye-sensitive from a comprehensive user study.

摘要: 深度神经网络显著提高了人脸伪造检测模型在识别人工智能生成内容(AIGC)方面的性能。然而，在模型训练期间注入触发器(即后门攻击)，对他们的安全构成了严重威胁。尽管现有的后门防御和手动数据选择可以缓解那些使用人眼敏感的触发因素，如补丁或对抗性噪音，但更具挑战性的自然后门触发因素仍然研究不足。为了进一步研究自然触发器，我们提出了一种新的分析合成后门攻击人脸伪造检测模型，将自然触发器嵌入到潜在空间中。我们从两个角度深入研究了这种后门漏洞：(1)模型识别(基于优化的触发器)：我们采用替代检测模型，通过最小化交叉熵损失来找到触发器；(2)数据分布(自定义触发器)：我们在长尾分布中操纵不常见的面部属性来生成有毒样本，而不需要检测模型的监督。此外，为了全面评估针对最新AIGC的检测模型，我们使用最先进的StyleGAN和稳定扩散来生成触发。最后，这些后门触发器将特定的语义特征引入到生成的中毒样本中(例如，皮肤纹理和微笑)，这些特征更加自然和健壮。大量的实验表明，我们的方法从三个层面上具有优势：(1)攻击成功率：我们的攻击成功率高(超过99%)，模型准确率下降很小(低于0.2%)，投毒率低(低于3%)；(2)后门防御：我们的后门防御方法在面对现有的后门防御方法时表现出更好的健壮性；(3)人工检查：从综合的用户研究来看，我们的方法对人眼的敏感度较低。



## **14. Dictionary Attack on IMU-based Gait Authentication**

基于IMU的步态认证字典攻击 cs.CR

12 pages, 9 figures, accepted at AISec23 colocated with ACM CCS,  November 30, 2023, Copenhagen, Denmark

**SubmitDate**: 2023-12-31    [abs](http://arxiv.org/abs/2309.11766v2) [paper-pdf](http://arxiv.org/pdf/2309.11766v2)

**Authors**: Rajesh Kumar, Can Isik, Chilukuri K. Mohan

**Abstract**: We present a novel adversarial model for authentication systems that use gait patterns recorded by the inertial measurement unit (IMU) built into smartphones. The attack idea is inspired by and named after the concept of a dictionary attack on knowledge (PIN or password) based authentication systems. In particular, this work investigates whether it is possible to build a dictionary of IMUGait patterns and use it to launch an attack or find an imitator who can actively reproduce IMUGait patterns that match the target's IMUGait pattern. Nine physically and demographically diverse individuals walked at various levels of four predefined controllable and adaptable gait factors (speed, step length, step width, and thigh-lift), producing 178 unique IMUGait patterns. Each pattern attacked a wide variety of user authentication models. The deeper analysis of error rates (before and after the attack) challenges the belief that authentication systems based on IMUGait patterns are the most difficult to spoof; further research is needed on adversarial models and associated countermeasures.

摘要: 我们提出了一种新的敌意认证系统模型，该模型使用智能手机内置的惯性测量单元(IMU)记录的步态模式。该攻击思想的灵感来自于对基于知识(PIN或密码)的身份验证系统的字典攻击的概念，并以此命名。特别是，这项工作调查是否有可能建立一个IMUGait图案词典，并使用它来发动攻击，或者找到一个模仿者，他可以主动复制与目标的IMUGait图案匹配的IMUGait图案。九个身体和人口统计学上不同的人在四个预定义的可控和可适应步态因素(速度、步长、步宽和大腿抬起)的不同水平上行走，产生了178个独特的IMU步态模式。每种模式都攻击了各种各样的用户身份验证模型。对错误率的深入分析(攻击前和攻击后)挑战了基于IMUGait模式的认证系统最难欺骗的观点；需要对敌意模型和相关对策进行进一步研究。



## **15. Forbidden Facts: An Investigation of Competing Objectives in Llama-2**

禁忌事实：骆驼2号中相互竞争的目标的调查 cs.LG

Accepted to the ATTRIB and SoLaR workshops at NeurIPS 2023; (v3:  clarified experimental details)

**SubmitDate**: 2023-12-31    [abs](http://arxiv.org/abs/2312.08793v3) [paper-pdf](http://arxiv.org/pdf/2312.08793v3)

**Authors**: Tony T. Wang, Miles Wang, Kaivalya Hariharan, Nir Shavit

**Abstract**: LLMs often face competing pressures (for example helpfulness vs. harmlessness). To understand how models resolve such conflicts, we study Llama-2-chat models on the forbidden fact task. Specifically, we instruct Llama-2 to truthfully complete a factual recall statement while forbidding it from saying the correct answer. This often makes the model give incorrect answers. We decompose Llama-2 into 1000+ components, and rank each one with respect to how useful it is for forbidding the correct answer. We find that in aggregate, around 35 components are enough to reliably implement the full suppression behavior. However, these components are fairly heterogeneous and many operate using faulty heuristics. We discover that one of these heuristics can be exploited via a manually designed adversarial attack which we call The California Attack. Our results highlight some roadblocks standing in the way of being able to successfully interpret advanced ML systems. Project website available at https://forbiddenfacts.github.io .

摘要: 低收入国家经常面临相互竞争的压力(例如，有益与无害)。为了理解模型如何解决此类冲突，我们研究了关于禁止事实任务的Llama-2-Chat模型。具体地说，我们指示骆驼2号如实完成事实回忆声明，同时禁止它说出正确的答案。这经常使模型给出错误的答案。我们将Llama-2分解成1000多个成分，并根据它们对阻止正确答案的作用程度对每个成分进行排名。我们发现，总共大约35个组件就足以可靠地实现完全抑制行为。然而，这些组件具有相当大的异构性，许多组件使用错误的启发式方法进行操作。我们发现，其中一个启发式攻击可以通过手动设计的对抗性攻击来利用，我们称之为加利福尼亚州攻击。我们的结果突出了一些阻碍成功解释高级ML系统的障碍。项目网站为https://forbiddenfacts.github.io。



## **16. Explainability-Driven Leaf Disease Classification using Adversarial Training and Knowledge Distillation**

基于对抗性训练和知识提炼的可解释性叶部病害分类 cs.CV

10 pages, 8 figures, Accepted by ICAART 2024

**SubmitDate**: 2023-12-30    [abs](http://arxiv.org/abs/2401.00334v1) [paper-pdf](http://arxiv.org/pdf/2401.00334v1)

**Authors**: Sebastian-Vasile Echim, Iulian-Marius Tăiatu, Dumitru-Clementin Cercel, Florin Pop

**Abstract**: This work focuses on plant leaf disease classification and explores three crucial aspects: adversarial training, model explainability, and model compression. The models' robustness against adversarial attacks is enhanced through adversarial training, ensuring accurate classification even in the presence of threats. Leveraging explainability techniques, we gain insights into the model's decision-making process, improving trust and transparency. Additionally, we explore model compression techniques to optimize computational efficiency while maintaining classification performance. Through our experiments, we determine that on a benchmark dataset, the robustness can be the price of the classification accuracy with performance reductions of 3%-20% for regular tests and gains of 50%-70% for adversarial attack tests. We also demonstrate that a student model can be 15-25 times more computationally efficient for a slight performance reduction, distilling the knowledge of more complex models.

摘要: 这项工作集中在植物叶部病害的分类上，并探索了三个关键方面：对抗性训练、模型可解释性和模型压缩。通过对抗性训练增强了模型对对抗性攻击的稳健性，即使在存在威胁的情况下也确保了准确的分类。利用可解释性技术，我们可以深入了解模型的决策过程，从而提高信任和透明度。此外，我们还探索了模型压缩技术，以优化计算效率，同时保持分类性能。通过实验，我们确定在一个基准数据集上，在常规测试性能降低3%-20%，对抗性攻击测试性能提高50%-70%的情况下，鲁棒性可以是分类准确率的代价。我们还证明，学生模型的计算效率可以是15-25倍，而性能略有下降，提取了更复杂模型的知识。



## **17. Unraveling the Connections between Privacy and Certified Robustness in Federated Learning Against Poisoning Attacks**

解开联合学习抗中毒攻击中隐私与认证稳健性之间的联系 cs.CR

ACM CCS 2023

**SubmitDate**: 2023-12-30    [abs](http://arxiv.org/abs/2209.04030v3) [paper-pdf](http://arxiv.org/pdf/2209.04030v3)

**Authors**: Chulin Xie, Yunhui Long, Pin-Yu Chen, Qinbin Li, Arash Nourian, Sanmi Koyejo, Bo Li

**Abstract**: Federated learning (FL) provides an efficient paradigm to jointly train a global model leveraging data from distributed users. As local training data comes from different users who may not be trustworthy, several studies have shown that FL is vulnerable to poisoning attacks. Meanwhile, to protect the privacy of local users, FL is usually trained in a differentially private way (DPFL). Thus, in this paper, we ask: What are the underlying connections between differential privacy and certified robustness in FL against poisoning attacks? Can we leverage the innate privacy property of DPFL to provide certified robustness for FL? Can we further improve the privacy of FL to improve such robustness certification? We first investigate both user-level and instance-level privacy of FL and provide formal privacy analysis to achieve improved instance-level privacy. We then provide two robustness certification criteria: certified prediction and certified attack inefficacy for DPFL on both user and instance levels. Theoretically, we provide the certified robustness of DPFL based on both criteria given a bounded number of adversarial users or instances. Empirically, we conduct extensive experiments to verify our theories under a range of poisoning attacks on different datasets. We find that increasing the level of privacy protection in DPFL results in stronger certified attack inefficacy; however, it does not necessarily lead to a stronger certified prediction. Thus, achieving the optimal certified prediction requires a proper balance between privacy and utility loss.

摘要: 联合学习(FL)提供了一种有效的范例来联合训练利用来自分布式用户的数据的全局模型。由于本地训练数据来自可能不值得信任的不同用户，多项研究表明，FL容易受到中毒攻击。同时，为了保护本地用户的隐私，FL通常会以一种不同的私人方式进行培训(DPFL)。因此，在这篇文章中，我们问：区别隐私和FL对中毒攻击的认证健壮性之间有什么潜在的联系？我们能否利用DPFL与生俱来的隐私属性为FL提供经过认证的健壮性？我们能否进一步改善FL的隐私，以提高这种健壮性认证？我们首先对FL的用户级和实例级隐私进行了研究，并提供了形式化的隐私分析，以实现改进的实例级隐私。然后，我们提供了两个健壮性认证标准：DPFL在用户和实例级别上的认证预测和认证攻击无效。理论上，在给定有限数量的敌意用户或实例的情况下，我们基于这两个标准提供了DPFL的证明的健壮性。在经验上，我们在不同数据集的一系列中毒攻击下进行了广泛的实验来验证我们的理论。我们发现，增加DPFL中的隐私保护级别会导致更强的认证攻击无效；然而，这并不一定会导致更强的认证预测。因此，要实现最佳验证预测，需要在隐私和效用损失之间取得适当的平衡。



## **18. Fourier Analysis on Robustness of Graph Convolutional Neural Networks for Skeleton-based Action Recognition**

基于骨架的图卷积神经网络鲁棒性的傅立叶分析 cs.CV

18 pages, 13 figures

**SubmitDate**: 2023-12-30    [abs](http://arxiv.org/abs/2305.17939v2) [paper-pdf](http://arxiv.org/pdf/2305.17939v2)

**Authors**: Nariki Tanaka, Hiroshi Kera, Kazuhiko Kawamoto

**Abstract**: Using Fourier analysis, we explore the robustness and vulnerability of graph convolutional neural networks (GCNs) for skeleton-based action recognition. We adopt a joint Fourier transform (JFT), a combination of the graph Fourier transform (GFT) and the discrete Fourier transform (DFT), to examine the robustness of adversarially-trained GCNs against adversarial attacks and common corruptions. Experimental results with the NTU RGB+D dataset reveal that adversarial training does not introduce a robustness trade-off between adversarial attacks and low-frequency perturbations, which typically occurs during image classification based on convolutional neural networks. This finding indicates that adversarial training is a practical approach to enhancing robustness against adversarial attacks and common corruptions in skeleton-based action recognition. Furthermore, we find that the Fourier approach cannot explain vulnerability against skeletal part occlusion corruption, which highlights its limitations. These findings extend our understanding of the robustness of GCNs, potentially guiding the development of more robust learning methods for skeleton-based action recognition.

摘要: 利用傅立叶分析，我们研究了基于骨架的动作识别的图卷积神经网络(GCNS)的稳健性和脆弱性。我们采用联合傅里叶变换(JFT)，即图傅里叶变换(GFT)和离散傅立叶变换(DFT)的组合，来检验经过对抗性训练的GCNS对敌意攻击和常见腐败的健壮性。在NTU RGB+D数据集上的实验结果表明，对抗性训练不会在对抗性攻击和低频扰动之间引入稳健性权衡，而这通常发生在基于卷积神经网络的图像分类中。这一发现表明，在基于骨架的动作识别中，对抗性训练是一种增强对对抗性攻击和常见腐败的稳健性的实用方法。此外，我们发现傅立叶方法不能解释对骨骼部分遮挡破坏的脆弱性，这突出了它的局限性。这些发现扩展了我们对GCNS健壮性的理解，潜在地指导了基于骨骼的动作识别的更健壮的学习方法的发展。



## **19. ReMAV: Reward Modeling of Autonomous Vehicles for Finding Likely Failure Events**

ReMAV：自动车辆发现可能故障事件的奖励模型 cs.AI

**SubmitDate**: 2023-12-30    [abs](http://arxiv.org/abs/2308.14550v2) [paper-pdf](http://arxiv.org/pdf/2308.14550v2)

**Authors**: Aizaz Sharif, Dusica Marijan

**Abstract**: Autonomous vehicles are advanced driving systems that are well known to be vulnerable to various adversarial attacks, compromising vehicle safety and posing a risk to other road users. Rather than actively training complex adversaries by interacting with the environment, there is a need to first intelligently find and reduce the search space to only those states where autonomous vehicles are found to be less confident. In this paper, we propose a black-box testing framework ReMAV that uses offline trajectories first to analyze the existing behavior of autonomous vehicles and determine appropriate thresholds to find the probability of failure events. To this end, we introduce a three-step methodology which i) uses offline state action pairs of any autonomous vehicle under test, ii) builds an abstract behavior representation using our designed reward modeling technique to analyze states with uncertain driving decisions, and iii) uses a disturbance model for minimal perturbation attacks where the driving decisions are less confident. Our reward modeling technique helps in creating a behavior representation that allows us to highlight regions of likely uncertain behavior even when the standard autonomous vehicle performs well. We perform our experiments in a high-fidelity urban driving environment using three different driving scenarios containing single- and multi-agent interactions. Our experiment shows an increase in 35, 23, 48, and 50% in the occurrences of vehicle collision, road object collision, pedestrian collision, and offroad steering events, respectively by the autonomous vehicle under test, demonstrating a significant increase in failure events. We compare ReMAV with two baselines and show that ReMAV demonstrates significantly better effectiveness in generating failure events compared to the baselines in all evaluation metrics.

摘要: 自动驾驶汽车是一种先进的驾驶系统，众所周知，它容易受到各种对抗性攻击，危及车辆安全并对其他道路使用者构成风险。与其通过与环境交互来主动训练复杂的对手，还不如首先智能地找到并将搜索空间缩小到自动驾驶车辆被发现不太自信的那些状态。在本文中，我们提出了一个黑盒测试框架ReMAV，首先使用离线轨迹来分析自动驾驶汽车的现有行为，并确定适当的阈值来找到故障事件的概率。为此，我们引入了一种三步方法，i）使用任何测试中的自动驾驶车辆的离线状态动作对，ii）使用我们设计的奖励建模技术构建抽象行为表示，以分析具有不确定驾驶决策的状态，iii）使用干扰模型进行最小扰动攻击，其中驾驶决策不太自信。我们的奖励建模技术有助于创建一个行为表示，使我们能够突出显示可能的不确定行为的区域，即使标准的自动驾驶汽车表现良好。我们在高保真的城市驾驶环境中进行实验，使用三种不同的驾驶场景，其中包含单智能体和多智能体的交互。我们的实验显示，测试中的自动驾驶汽车的车辆碰撞、道路物体碰撞、行人碰撞和越野转向事件的发生率分别增加了35%、23%、48%和50%，表明故障事件显著增加。我们将ReMAV与两个基线进行比较，结果表明，在所有评估指标中，与基线相比，ReMAV在生成故障事件方面表现出更好的有效性。



## **20. CamPro: Camera-based Anti-Facial Recognition**

CamPro：基于摄像头的反人脸识别 cs.CV

Accepted by NDSS Symposium 2024

**SubmitDate**: 2023-12-30    [abs](http://arxiv.org/abs/2401.00151v1) [paper-pdf](http://arxiv.org/pdf/2401.00151v1)

**Authors**: Wenjun Zhu, Yuan Sun, Jiani Liu, Yushi Cheng, Xiaoyu Ji, Wenyuan Xu

**Abstract**: The proliferation of images captured from millions of cameras and the advancement of facial recognition (FR) technology have made the abuse of FR a severe privacy threat. Existing works typically rely on obfuscation, synthesis, or adversarial examples to modify faces in images to achieve anti-facial recognition (AFR). However, the unmodified images captured by camera modules that contain sensitive personally identifiable information (PII) could still be leaked. In this paper, we propose a novel approach, CamPro, to capture inborn AFR images. CamPro enables well-packed commodity camera modules to produce images that contain little PII and yet still contain enough information to support other non-sensitive vision applications, such as person detection. Specifically, CamPro tunes the configuration setup inside the camera image signal processor (ISP), i.e., color correction matrix and gamma correction, to achieve AFR, and designs an image enhancer to keep the image quality for possible human viewers. We implemented and validated CamPro on a proof-of-concept camera, and our experiments demonstrate its effectiveness on ten state-of-the-art black-box FR models. The results show that CamPro images can significantly reduce face identification accuracy to 0.3\% while having little impact on the targeted non-sensitive vision application. Furthermore, we find that CamPro is resilient to adaptive attackers who have re-trained their FR models using images generated by CamPro, even with full knowledge of privacy-preserving ISP parameters.

摘要: 从数百万个摄像头拍摄的图像激增，以及面部识别(FR)技术的进步，使FR的滥用成为一个严重的隐私威胁。现有的工作通常依赖于混淆、合成或敌意示例来修改图像中的人脸以实现反面部识别(AFR)。然而，包含敏感个人身份信息(PII)的相机模块捕获的未经修改的图像仍有可能泄露。在这篇文章中，我们提出了一种新的方法，CamPro，以捕获天生的AFR图像。CamPro使包装良好的商用相机模块能够生成包含少量PII的图像，但仍包含足够的信息来支持其他非敏感视觉应用，如人员检测。具体来说，CamPro调整了相机图像信号处理器(ISP)内部的配置设置，即颜色校正矩阵和伽马校正，以实现AFR，并设计了图像增强器，以保持图像质量，以供可能的人类观看。我们在一个概念验证相机上实现并验证了CamPro，我们的实验证明了它在十个最先进的黑盒FR模型上的有效性。结果表明，CamPro图像可以显著降低人脸识别的准确率至0.3%，而对目标非敏感视觉应用的影响很小。此外，我们发现CamPro对使用CamPro生成的图像重新训练FR模型的自适应攻击者具有弹性，即使完全了解隐私保护的ISP参数也是如此。



## **21. TPatch: A Triggered Physical Adversarial Patch**

TPatch：一种触发的物理对抗性补丁 cs.CR

Appeared in 32nd USENIX Security Symposium (USENIX Security 23)

**SubmitDate**: 2023-12-30    [abs](http://arxiv.org/abs/2401.00148v1) [paper-pdf](http://arxiv.org/pdf/2401.00148v1)

**Authors**: Wenjun Zhu, Xiaoyu Ji, Yushi Cheng, Shibo Zhang, Wenyuan Xu

**Abstract**: Autonomous vehicles increasingly utilize the vision-based perception module to acquire information about driving environments and detect obstacles. Correct detection and classification are important to ensure safe driving decisions. Existing works have demonstrated the feasibility of fooling the perception models such as object detectors and image classifiers with printed adversarial patches. However, most of them are indiscriminately offensive to every passing autonomous vehicle. In this paper, we propose TPatch, a physical adversarial patch triggered by acoustic signals. Unlike other adversarial patches, TPatch remains benign under normal circumstances but can be triggered to launch a hiding, creating or altering attack by a designed distortion introduced by signal injection attacks towards cameras. To avoid the suspicion of human drivers and make the attack practical and robust in the real world, we propose a content-based camouflage method and an attack robustness enhancement method to strengthen it. Evaluations with three object detectors, YOLO V3/V5 and Faster R-CNN, and eight image classifiers demonstrate the effectiveness of TPatch in both the simulation and the real world. We also discuss possible defenses at the sensor, algorithm, and system levels.

摘要: 自动驾驶汽车越来越多地利用基于视觉的感知模块来获取有关驾驶环境的信息，并检测障碍物。正确的检测和分类对于确保安全驾驶决策非常重要。已有的工作已经证明了欺骗感知模型的可行性，例如使用打印的对抗性补丁的对象检测器和图像分类器。然而，它们中的大多数都是不分青红皂白地冒犯每一辆经过的自动驾驶汽车。本文提出了一种由声信号触发的物理对抗性补丁TPatch。与其他敌意补丁不同的是，TPatch在正常情况下仍然是良性的，但可以通过对摄像头的信号注入攻击引入的故意扭曲触发发起隐藏、创建或更改攻击。为了避免人类驾驶员的怀疑，使攻击在真实世界中具有实用性和健壮性，提出了基于内容的伪装方法和攻击健壮性增强方法来增强攻击的健壮性。用三个目标检测器YOLO V3/V5和更快的R-CNN和八个图像分类器进行的评估表明了TPatch在模拟和现实世界中的有效性。我们还讨论了传感器、算法和系统级别的可能防御措施。



## **22. Comparing the robustness of modern no-reference image- and video-quality metrics to adversarial attacks**

比较现代无参考图像和视频质量指标对敌方攻击的稳健性 cs.CV

**SubmitDate**: 2023-12-29    [abs](http://arxiv.org/abs/2310.06958v3) [paper-pdf](http://arxiv.org/pdf/2310.06958v3)

**Authors**: Anastasia Antsiferova, Khaled Abud, Aleksandr Gushchin, Ekaterina Shumitskaya, Sergey Lavrushkin, Dmitriy Vatolin

**Abstract**: Nowadays neural-network-based image- and video-quality metrics show better performance compared to traditional methods. However, they also became more vulnerable to adversarial attacks that increase metrics' scores without improving visual quality. The existing benchmarks of quality metrics compare their performance in terms of correlation with subjective quality and calculation time. However, the adversarial robustness of image-quality metrics is also an area worth researching. In this paper, we analyse modern metrics' robustness to different adversarial attacks. We adopted adversarial attacks from computer vision tasks and compared attacks' efficiency against 15 no-reference image/video-quality metrics. Some metrics showed high resistance to adversarial attacks which makes their usage in benchmarks safer than vulnerable metrics. The benchmark accepts new metrics submissions for researchers who want to make their metrics more robust to attacks or to find such metrics for their needs. Try our benchmark using pip install robustness-benchmark.

摘要: 如今，基于神经网络的图像和视频质量度量与传统方法相比表现出更好的性能。然而，它们也变得更容易受到对抗性攻击，这些攻击会增加指标的分数，而不会提高视觉质量。现有的质量指标的基准比较他们的性能与主观质量和计算时间的相关性。然而，图像质量度量的对抗鲁棒性也是一个值得研究的领域。在本文中，我们分析了现代度量的鲁棒性不同的对抗性攻击。我们采用了来自计算机视觉任务的对抗性攻击，并将攻击的效率与15个无参考图像/视频质量指标进行了比较。一些指标对对抗性攻击表现出很高的抵抗力，这使得它们在基准测试中的使用比脆弱的指标更安全。该基准测试接受新的指标提交给那些希望使他们的指标对攻击更鲁棒或希望找到满足他们需求的指标的研究人员。使用pip install robustness-benchmark尝试我们的基准测试。



## **23. MVPatch: More Vivid Patch for Adversarial Camouflaged Attacks on Object Detectors in the Physical World**

MVPatch：对物理世界中的对象探测器进行对抗性伪装攻击的更生动的补丁 cs.CR

14 pages, 8 figures, submitted to IEEE Transactions on Information  Forensics & Security

**SubmitDate**: 2023-12-29    [abs](http://arxiv.org/abs/2312.17431v1) [paper-pdf](http://arxiv.org/pdf/2312.17431v1)

**Authors**: Zheng Zhou, Hongbo Zhao, Ju Liu, Qiaosheng Zhang, Guangbiao Wang, Chunlei Wang, Wenquan Feng

**Abstract**: Recent research has shown that adversarial patches can manipulate outputs from object detection models. However, the conspicuous patterns on these patches may draw more attention and raise suspicions among humans. Moreover, existing works have primarily focused on the attack performance of individual models and have neglected the generation of adversarial patches for ensemble attacks on multiple object detection models. To tackle these concerns, we propose a novel approach referred to as the More Vivid Patch (MVPatch), which aims to improve the transferability and stealthiness of adversarial patches while considering the limitations observed in prior paradigms, such as easy identification and poor transferability. Our approach incorporates an attack algorithm that decreases object confidence scores of multiple object detectors by using the ensemble attack loss function, thereby enhancing the transferability of adversarial patches. Additionally, we propose a lightweight visual similarity measurement algorithm realized by the Compared Specified Image Similarity (CSS) loss function, which allows for the generation of natural and stealthy adversarial patches without the reliance on additional generative models. Extensive experiments demonstrate that the proposed MVPatch algorithm achieves superior attack transferability compared to similar algorithms in both digital and physical domains, while also exhibiting a more natural appearance. These findings emphasize the remarkable stealthiness and transferability of the proposed MVPatch attack algorithm.

摘要: 最近的研究表明，对抗补丁可以操纵对象检测模型的输出。然而，这些斑块上的明显图案可能会引起更多的注意，并引起人类的怀疑。此外，现有的工作主要集中在单个模型的攻击性能上，而忽略了对多个目标检测模型进行集成攻击的对抗补丁的生成。为了解决这些问题，我们提出了一种新的方法，称为更生动的补丁（MVPatch），其目的是提高对抗补丁的可转移性和隐蔽性，同时考虑到在以前的范例中观察到的局限性，如容易识别和可转移性差。我们的方法采用了一种攻击算法，通过使用集成攻击损失函数来降低多个对象检测器的对象置信度得分，从而增强了对抗补丁的可转移性。此外，我们提出了一种轻量级的视觉相似性测量算法，该算法由比较指定图像相似性（CSS）损失函数实现，该函数允许生成自然和隐形的对抗补丁，而无需依赖额外的生成模型。大量的实验表明，所提出的MVPatch算法实现了优越的攻击转移性相比，在数字和物理领域的类似算法，同时也表现出更自然的外观。这些发现强调了显着的隐蔽性和可转移性的MVPatch攻击算法。



## **24. Can you See me? On the Visibility of NOPs against Android Malware Detectors**

你能看清我吗？关于NOPS对Android恶意软件检测器的可见性 cs.CR

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.17356v1) [paper-pdf](http://arxiv.org/pdf/2312.17356v1)

**Authors**: Diego Soi, Davide Maiorca, Giorgio Giacinto, Harel Berger

**Abstract**: Android malware still represents the most significant threat to mobile systems. While Machine Learning systems are increasingly used to identify these threats, past studies have revealed that attackers can bypass these detection mechanisms by making subtle changes to Android applications, such as adding specific API calls. These modifications are often referred to as No OPerations (NOP), which ideally should not alter the semantics of the program. However, many NOPs can be spotted and eliminated by refining the app analysis process. This paper proposes a visibility metric that assesses the difficulty in spotting NOPs and similar non-operational codes. We tested our metric on a state-of-the-art, opcode-based deep learning system for Android malware detection. We implemented attacks on the feature and problem spaces and calculated their visibility according to our metric. The attained results show an intriguing trade-off between evasion efficacy and detectability: our metric can be valuable to ensure the real effectiveness of an adversarial attack, also serving as a useful aid to develop better defenses.

摘要: Android恶意软件仍然是移动系统面临的最大威胁。虽然机器学习系统越来越多地被用来识别这些威胁，但过去的研究表明，攻击者可以通过对Android应用程序进行微妙的更改，例如添加特定的API调用，绕过这些检测机制。这些修改通常被称为无操作(NOP)，理想情况下不应该改变程序的语义。然而，通过改进应用程序分析流程，可以发现并消除许多NOP。本文提出了一种可见性度量来评估发现NOP和类似的非操作代码的难度。我们在一个最先进的、基于操作码的深度学习系统上测试了我们的指标，以检测Android恶意软件。我们对功能和问题空间实施了攻击，并根据我们的度量计算了它们的可见性。所获得的结果显示了逃避有效性和可检测性之间的有趣的权衡：我们的度量对于确保对抗性攻击的真正有效性是有价值的，也可以作为开发更好防御的有用辅助。



## **25. Timeliness: A New Design Metric and a New Attack Surface**

时效性：一种新的设计尺度和新的攻击面 cs.IT

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.17220v1) [paper-pdf](http://arxiv.org/pdf/2312.17220v1)

**Authors**: Priyanka Kaswan, Sennur Ulukus

**Abstract**: As the landscape of time-sensitive applications gains prominence in 5G/6G communications, timeliness of information updates at network nodes has become crucial, which is popularly quantified in the literature by the age of information metric. However, as we devise policies to improve age of information of our systems, we inadvertently introduce a new vulnerability for adversaries to exploit. In this article, we comprehensively discuss the diverse threats that age-based systems are vulnerable to. We begin with discussion on densely interconnected networks that employ gossiping between nodes to expedite dissemination of dynamic information in the network, and show how the age-based nature of gossiping renders these networks uniquely susceptible to threats such as timestomping attacks, jamming attacks, and the propagation of misinformation. Later, we survey adversarial works within simpler network settings, specifically in one-hop and two-hop configurations, and delve into adversarial robustness concerning challenges posed by jamming, timestomping, and issues related to privacy leakage. We conclude this article with future directions that aim to address challenges posed by more intelligent adversaries and robustness of networks to them.

摘要: 随着时间敏感型应用在5G/6 G通信中的地位日益突出，网络节点处信息更新的及时性变得至关重要，这在文献中普遍通过信息度量的年龄来量化。然而，当我们设计策略来改善我们系统的信息时代时，我们无意中引入了一个新的漏洞供对手利用。在本文中，我们将全面讨论基于年龄的系统容易受到的各种威胁。我们开始讨论密集互连的网络，采用节点之间的八卦，以加快网络中的动态信息的传播，并显示如何基于年龄的性质的八卦使这些网络特别容易受到威胁，如时间戳攻击，干扰攻击，传播错误信息。随后，我们调查了在更简单的网络设置中的对抗性工作，特别是在一跳和两跳配置中，并深入研究了对抗性鲁棒性，包括干扰、时间戳和与隐私泄露相关的问题所带来的挑战。最后，我们总结了未来的发展方向，旨在应对更智能的对手和网络对他们的鲁棒性所带来的挑战。



## **26. Explainability-Based Adversarial Attack on Graphs Through Edge Perturbation**

基于可解释性的图的边扰动敌意攻击 cs.CR

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.17301v1) [paper-pdf](http://arxiv.org/pdf/2312.17301v1)

**Authors**: Dibaloke Chanda, Saba Heidari Gheshlaghi, Nasim Yahya Soltani

**Abstract**: Despite the success of graph neural networks (GNNs) in various domains, they exhibit susceptibility to adversarial attacks. Understanding these vulnerabilities is crucial for developing robust and secure applications. In this paper, we investigate the impact of test time adversarial attacks through edge perturbations which involve both edge insertions and deletions. A novel explainability-based method is proposed to identify important nodes in the graph and perform edge perturbation between these nodes. The proposed method is tested for node classification with three different architectures and datasets. The results suggest that introducing edges between nodes of different classes has higher impact as compared to removing edges among nodes within the same class.

摘要: 尽管图神经网络(GNN)在各个领域都取得了成功，但它们表现出对对手攻击的敏感性。了解这些漏洞对于开发健壮、安全的应用程序至关重要。本文研究了边扰动对测试时间敌意攻击的影响，其中边扰动涉及边的插入和删除。提出了一种新的基于可解释性的方法来识别图中的重要节点，并对这些节点之间的边进行扰动。用三种不同的体系结构和数据集对该方法进行了节点分类测试。结果表明，在不同类别的节点之间引入边比在同一类内的节点之间删除边具有更高的影响。



## **27. On the Robustness of Decision-Focused Learning**

决策学习的鲁棒性研究 cs.LG

17 pages, 45 figures, submitted to AAAI artificial intelligence for  operations research workshop

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2311.16487v3) [paper-pdf](http://arxiv.org/pdf/2311.16487v3)

**Authors**: Yehya Farhat

**Abstract**: Decision-Focused Learning (DFL) is an emerging learning paradigm that tackles the task of training a machine learning (ML) model to predict missing parameters of an incomplete optimization problem, where the missing parameters are predicted. DFL trains an ML model in an end-to-end system, by integrating the prediction and optimization tasks, providing better alignment of the training and testing objectives. DFL has shown a lot of promise and holds the capacity to revolutionize decision-making in many real-world applications. However, very little is known about the performance of these models under adversarial attacks. We adopt ten unique DFL methods and benchmark their performance under two distinctly focused attacks adapted towards the Predict-then-Optimize problem setting. Our study proposes the hypothesis that the robustness of a model is highly correlated with its ability to find predictions that lead to optimal decisions without deviating from the ground-truth label. Furthermore, we provide insight into how to target the models that violate this condition and show how these models respond differently depending on the achieved optimality at the end of their training cycles.

摘要: 决策聚焦学习（DFL）是一种新兴的学习范式，它处理训练机器学习（ML）模型来预测不完全优化问题的缺失参数的任务，其中缺失参数被预测。DFL通过集成预测和优化任务，在端到端系统中训练ML模型，从而更好地调整训练和测试目标。DFL已经显示出很大的潜力，并有能力在许多现实世界的应用中彻底改变决策。然而，人们对这些模型在对抗性攻击下的性能知之甚少。我们采用了10个独特的DFL方法，并在两个不同的攻击下对它们的性能进行了基准测试，这些攻击都是针对预测然后优化问题设置的。我们的研究提出了一个假设，即模型的鲁棒性与其在不偏离地面事实标签的情况下找到导致最佳决策的预测的能力高度相关。此外，我们还提供了如何针对违反此条件的模型的见解，并展示了这些模型如何根据其训练周期结束时达到的最优性做出不同的响应。



## **28. BlackboxBench: A Comprehensive Benchmark of Black-box Adversarial Attacks**

BlackboxBtch：黑盒对抗性攻击的综合基准 cs.CR

37 pages, 29 figures

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.16979v1) [paper-pdf](http://arxiv.org/pdf/2312.16979v1)

**Authors**: Meixi Zheng, Xuanchen Yan, Zihao Zhu, Hongrui Chen, Baoyuan Wu

**Abstract**: Adversarial examples are well-known tools to evaluate the vulnerability of deep neural networks (DNNs). Although lots of adversarial attack algorithms have been developed, it is still challenging in the practical scenario that the model's parameters and architectures are inaccessible to the attacker/evaluator, i.e., black-box adversarial attacks. Due to the practical importance, there has been rapid progress from recent algorithms, reflected by the quick increase in attack success rate and the quick decrease in query numbers to the target model. However, there is a lack of thorough evaluations and comparisons among these algorithms, causing difficulties of tracking the real progress, analyzing advantages and disadvantages of different technical routes, as well as designing future development roadmap of this field. Thus, in this work, we aim at building a comprehensive benchmark of black-box adversarial attacks, called BlackboxBench. It mainly provides: 1) a unified, extensible and modular-based codebase, implementing 25 query-based attack algorithms and 30 transfer-based attack algorithms; 2) comprehensive evaluations: we evaluate the implemented algorithms against several mainstreaming model architectures on 2 widely used datasets (CIFAR-10 and a subset of ImageNet), leading to 14,106 evaluations in total; 3) thorough analysis and new insights, as well analytical tools. The website and source codes of BlackboxBench are available at https://blackboxbench.github.io/ and https://github.com/SCLBD/BlackboxBench/, respectively.

摘要: 对抗性例子是评价深度神经网络(DNN)脆弱性的常用工具。尽管已有许多对抗性攻击算法被提出，但在实际场景中，攻击者/评估者无法访问模型的参数和体系结构，即黑箱对抗性攻击，仍然具有挑战性。由于实际的重要性，最近的算法有了快速的进步，反映在攻击成功率的快速增加和对目标模型的查询数量的快速减少上。然而，这些算法之间缺乏深入的评估和比较，这给跟踪实际进展、分析不同技术路线的优缺点以及设计该领域未来发展路线图带来了困难。因此，在这项工作中，我们的目标是建立一个全面的黑盒对抗性攻击基准，称为BlackboxBitch.它主要提供：1)统一、可扩展和基于模块的代码库，实现了25种基于查询的攻击算法和30种基于传输的攻击算法；2)综合评估：我们在两个广泛使用的数据集(CIFAR-10和ImageNet的子集)上针对几种主流模型架构对所实现的算法进行了评估，总共产生了14,106次评估；3)深入的分析和新的见解，以及分析工具。BlackboxBch的网站和源代码分别可在https://blackboxbench.github.io/和https://github.com/SCLBD/BlackboxBench/，上获得。



## **29. Attack Tree Analysis for Adversarial Evasion Attacks**

对抗性逃避攻击的攻击树分析 cs.CR

10 pages

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.16957v1) [paper-pdf](http://arxiv.org/pdf/2312.16957v1)

**Authors**: Yuki Yamaguchi, Toshiaki Aoki

**Abstract**: Recently, the evolution of deep learning has promoted the application of machine learning (ML) to various systems. However, there are ML systems, such as autonomous vehicles, that cause critical damage when they misclassify. Conversely, there are ML-specific attacks called adversarial attacks based on the characteristics of ML systems. For example, one type of adversarial attack is an evasion attack, which uses minute perturbations called "adversarial examples" to intentionally misclassify classifiers. Therefore, it is necessary to analyze the risk of ML-specific attacks in introducing ML base systems. In this study, we propose a quantitative evaluation method for analyzing the risk of evasion attacks using attack trees. The proposed method consists of the extension of the conventional attack tree to analyze evasion attacks and the systematic construction method of the extension. In the extension of the conventional attack tree, we introduce ML and conventional attack nodes to represent various characteristics of evasion attacks. In the systematic construction process, we propose a procedure to construct the attack tree. The procedure consists of three steps: (1) organizing information about attack methods in the literature to a matrix, (2) identifying evasion attack scenarios from methods in the matrix, and (3) constructing the attack tree from the identified scenarios using a pattern. Finally, we conducted experiments on three ML image recognition systems to demonstrate the versatility and effectiveness of our proposed method.

摘要: 近年来，深度学习的发展推动了机器学习（ML）在各种系统中的应用。然而，有些机器学习系统，如自动驾驶汽车，在错误分类时会造成严重损害。相反，还有基于ML系统特征的ML特定攻击，称为对抗性攻击。例如，一种类型的对抗攻击是逃避攻击，它使用称为“对抗示例”的微小扰动来故意对分类器进行错误分类。因此，有必要在引入ML基础系统时分析ML特定攻击的风险。在这项研究中，我们提出了一个定量的评估方法来分析规避攻击的风险，使用攻击树。该方法包括对传统攻击树进行扩展以分析规避攻击和扩展攻击树的系统构造方法。在传统攻击树的扩展中，我们引入ML和传统攻击节点来表示规避攻击的各种特征。在系统的构建过程中，我们提出了一个攻击树的构建过程。该过程由三个步骤组成：（1）将文献中关于攻击方法的信息组织到矩阵中，（2）从矩阵中的方法中识别规避攻击场景，以及（3）使用模式从识别的场景中构建攻击树。最后，我们在三个ML图像识别系统上进行了实验，以证明我们所提出的方法的通用性和有效性。



## **30. DOEPatch: Dynamically Optimized Ensemble Model for Adversarial Patches Generation**

DOEPatch：动态优化的对抗补丁生成包围模型 cs.CV

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.16907v1) [paper-pdf](http://arxiv.org/pdf/2312.16907v1)

**Authors**: Wenyi Tan, Yang Li, Chenxing Zhao, Zhunga Liu, Quan Pan

**Abstract**: Object detection is a fundamental task in various applications ranging from autonomous driving to intelligent security systems. However, recognition of a person can be hindered when their clothing is decorated with carefully designed graffiti patterns, leading to the failure of object detection. To achieve greater attack potential against unknown black-box models, adversarial patches capable of affecting the outputs of multiple-object detection models are required. While ensemble models have proven effective, current research in the field of object detection typically focuses on the simple fusion of the outputs of all models, with limited attention being given to developing general adversarial patches that can function effectively in the physical world. In this paper, we introduce the concept of energy and treat the adversarial patches generation process as an optimization of the adversarial patches to minimize the total energy of the ``person'' category. Additionally, by adopting adversarial training, we construct a dynamically optimized ensemble model. During training, the weight parameters of the attacked target models are adjusted to find the balance point at which the generated adversarial patches can effectively attack all target models. We carried out six sets of comparative experiments and tested our algorithm on five mainstream object detection models. The adversarial patches generated by our algorithm can reduce the recognition accuracy of YOLOv2 and YOLOv3 to 13.19\% and 29.20\%, respectively. In addition, we conducted experiments to test the effectiveness of T-shirts covered with our adversarial patches in the physical world and could achieve that people are not recognized by the object detection model. Finally, leveraging the Grad-CAM tool, we explored the attack mechanism of adversarial patches from an energetic perspective.

摘要: 目标检测是从自动驾驶到智能安防系统等各种应用中的一项基本任务。然而，当一个人的衣服上装饰着精心设计的涂鸦图案时，对他的识别可能会受到阻碍，导致物体检测失败。为了对未知黑盒模型实现更大的攻击潜力，需要能够影响多目标检测模型输出的对抗性补丁。虽然集成模型已被证明是有效的，但目前在目标检测领域的研究通常集中在所有模型的输出的简单融合上，而对开发能够在物理世界中有效发挥作用的通用对抗性补丁的关注有限。本文引入能量的概念，将对抗性补丁的生成过程看作是对敌对性补丁的优化，以最小化“人”范畴的总能量。此外，通过采用对抗性训练，构建了一个动态优化的集成模型。在训练过程中，调整被攻击目标模型的权重参数，以找到生成的敌意补丁能够有效攻击所有目标模型的平衡点。我们进行了六组对比实验，并在五个主流的目标检测模型上测试了我们的算法。该算法生成的敌意块使YOLOv2和YOLOv3的识别正确率分别降低到13.19和29.20。此外，我们还进行了实验，测试了T恤在现实世界中被我们的对手补丁覆盖的有效性，并可以实现目标检测模型无法识别人。最后，利用Grad-CAM工具，从能量的角度探讨了对抗性补丁的攻击机制。



## **31. Adversarial Attacks on Image Classification Models: Analysis and Defense**

图像分类模型的对抗性攻击分析与防御 cs.CV

This is the accepted version of the paper presented at the 10th  International Conference on Business Analytics and Intelligence (ICBAI'24).  The conference was organized by the Indian Institute of Science, Bangalore,  India, from December 18 - 20, 2023. The paper is 10 pages long and it  contains 14 tables and 11 figures

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.16880v1) [paper-pdf](http://arxiv.org/pdf/2312.16880v1)

**Authors**: Jaydip Sen, Abhiraj Sen, Ananda Chatterjee

**Abstract**: The notion of adversarial attacks on image classification models based on convolutional neural networks (CNN) is introduced in this work. To classify images, deep learning models called CNNs are frequently used. However, when the networks are subject to adversarial attacks, extremely potent and previously trained CNN models that perform quite effectively on image datasets for image classification tasks may perform poorly. In this work, one well-known adversarial attack known as the fast gradient sign method (FGSM) is explored and its adverse effects on the performances of image classification models are examined. The FGSM attack is simulated on three pre-trained image classifier CNN architectures, ResNet-101, AlexNet, and RegNetY 400MF using randomly chosen images from the ImageNet dataset. The classification accuracies of the models are computed in the absence and presence of the attack to demonstrate the detrimental effect of the attack on the performances of the classifiers. Finally, a mechanism is proposed to defend against the FGSM attack based on a modified defensive distillation-based approach. Extensive results are presented for the validation of the proposed scheme.

摘要: 提出了对基于卷积神经网络的图像分类模型进行敌意攻击的概念。为了对图像进行分类，人们经常使用称为CNN的深度学习模型。然而，当网络受到敌意攻击时，在图像数据集上非常有效地执行图像分类任务的极其强大的和先前训练的CNN模型可能会表现得很差。本文研究了一种著名的对抗性攻击--快速梯度符号方法(FGSM)，并考察了它对图像分类模型性能的不利影响。使用从ImageNet数据集中随机选择的图像，在三种预先训练的图像分类器CNN架构上模拟了FGSM攻击，即ResNet-101、AlexNet和RegNetY 400MF。在没有攻击和存在攻击的情况下，计算了模型的分类精度，以说明攻击对分类器性能的不利影响。最后，提出了一种基于改进的防御蒸馏方法的FGSM攻击防御机制。为验证该方案的有效性，给出了大量的结果。



## **32. Adv-Diffusion: Imperceptible Adversarial Face Identity Attack via Latent Diffusion Model**

ADV扩散：基于潜在扩散模型的隐形对抗性人脸身份攻击 cs.CV

Accepted by AAAI 2024

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.11285v2) [paper-pdf](http://arxiv.org/pdf/2312.11285v2)

**Authors**: Decheng Liu, Xijun Wang, Chunlei Peng, Nannan Wang, Ruiming Hu, Xinbo Gao

**Abstract**: Adversarial attacks involve adding perturbations to the source image to cause misclassification by the target model, which demonstrates the potential of attacking face recognition models. Existing adversarial face image generation methods still can't achieve satisfactory performance because of low transferability and high detectability. In this paper, we propose a unified framework Adv-Diffusion that can generate imperceptible adversarial identity perturbations in the latent space but not the raw pixel space, which utilizes strong inpainting capabilities of the latent diffusion model to generate realistic adversarial images. Specifically, we propose the identity-sensitive conditioned diffusion generative model to generate semantic perturbations in the surroundings. The designed adaptive strength-based adversarial perturbation algorithm can ensure both attack transferability and stealthiness. Extensive qualitative and quantitative experiments on the public FFHQ and CelebA-HQ datasets prove the proposed method achieves superior performance compared with the state-of-the-art methods without an extra generative model training process. The source code is available at https://github.com/kopper-xdu/Adv-Diffusion.

摘要: 对抗性攻击包括在源图像中添加扰动以导致目标模型的错误分类，这表明了攻击人脸识别模型的可能性。现有的对抗性人脸图像生成方法由于可移植性低、可检测性高，仍不能达到令人满意的效果。在本文中，我们提出了一个统一的框架，它可以在潜在空间而不是原始像素空间产生不可察觉的敌意身份扰动，该框架利用潜在扩散模型强大的修复能力来生成逼真的对抗性图像。具体地说，我们提出了身份敏感的条件扩散生成模型来产生环境中的语义扰动。所设计的基于强度的自适应对抗性扰动算法既能保证攻击的可传递性，又能保证隐蔽性。在公共FFHQ和CelebA-HQ数据集上的大量定性和定量实验证明，该方法在不需要额外的产生式模型训练过程的情况下，取得了优于最新方法的性能。源代码可在https://github.com/kopper-xdu/Adv-Diffusion.上找到



## **33. Temporal Knowledge Distillation for Time-Sensitive Financial Services Applications**

面向时间敏感金融服务应用的时态知识提取 cs.LG

arXiv admin note: text overlap with arXiv:2101.01689

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.16799v1) [paper-pdf](http://arxiv.org/pdf/2312.16799v1)

**Authors**: Hongda Shen, Eren Kurshan

**Abstract**: Detecting anomalies has become an increasingly critical function in the financial service industry. Anomaly detection is frequently used in key compliance and risk functions such as financial crime detection fraud and cybersecurity. The dynamic nature of the underlying data patterns especially in adversarial environments like fraud detection poses serious challenges to the machine learning models. Keeping up with the rapid changes by retraining the models with the latest data patterns introduces pressures in balancing the historical and current patterns while managing the training data size. Furthermore the model retraining times raise problems in time-sensitive and high-volume deployment systems where the retraining period directly impacts the models ability to respond to ongoing attacks in a timely manner. In this study we propose a temporal knowledge distillation-based label augmentation approach (TKD) which utilizes the learning from older models to rapidly boost the latest model and effectively reduces the model retraining times to achieve improved agility. Experimental results show that the proposed approach provides advantages in retraining times while improving the model performance.

摘要: 在金融服务业中，检测异常已成为一项越来越关键的功能。异常检测经常用于关键的合规和风险职能，如金融犯罪检测、欺诈和网络安全。底层数据模式的动态特性，特别是在欺诈检测等敌意环境中，对机器学习模型提出了严重的挑战。通过用最新的数据模式重新训练模型来跟上快速变化，在管理训练数据大小的同时会带来平衡历史和当前模式的压力。此外，模型再培训时间在对时间敏感的大批量部署系统中产生了问题，在这些系统中，再培训期直接影响到模型及时应对持续攻击的能力。在本研究中，我们提出了一种基于时态知识蒸馏的标签扩充方法(TKD)，该方法利用对旧模型的学习来快速提升最新模型，并有效地减少模型的重新训练次数以达到提高敏捷性的目的。实验结果表明，该方法在提高模型性能的同时，缩短了训练时间。



## **34. Multi-Task Models Adversarial Attacks**

多任务对抗性攻击模型 cs.LG

19 pages, 6 figures

**SubmitDate**: 2023-12-27    [abs](http://arxiv.org/abs/2305.12066v3) [paper-pdf](http://arxiv.org/pdf/2305.12066v3)

**Authors**: Lijun Zhang, Xiao Liu, Kaleel Mahmood, Caiwen Ding, Hui Guan

**Abstract**: Multi-Task Learning (MTL) involves developing a singular model, known as a multi-task model, to concurrently perform multiple tasks. While the security of single-task models has been thoroughly studied, multi-task models pose several critical security questions, such as 1) their vulnerability to single-task adversarial attacks, 2) the possibility of designing attacks that target multiple tasks, and 3) the impact of task sharing and adversarial training on their resilience to such attacks. This paper addresses these queries through detailed analysis and rigorous experimentation. First, we explore the adaptation of single-task white-box attacks to multi-task models and identify their limitations. We then introduce a novel attack framework, the Gradient Balancing Multi-Task Attack (GB-MTA), which treats attacking a multi-task model as an optimization problem. This problem, based on averaged relative loss change across tasks, is approximated as an integer linear programming problem. Extensive evaluations on MTL benchmarks, NYUv2 and Tiny-Taxonomy, demonstrate GB-MTA's effectiveness against both standard and adversarially trained multi-task models. The results also highlight a trade-off between task accuracy improvement via parameter sharing and increased model vulnerability due to enhanced attack transferability.

摘要: 多任务学习(MTL)涉及开发一个单一模型，称为多任务模型，以同时执行多个任务。虽然单任务模型的安全性已经得到了深入的研究，但多任务模型提出了几个关键的安全问题，如1)它们对单任务对抗性攻击的脆弱性，2)针对多任务的攻击设计的可能性，以及3)任务分担和对抗性训练对其抵抗此类攻击的影响。本文通过详细的分析和严谨的实验解决了这些问题。首先，我们探索了单任务白盒攻击对多任务模型的适应，并确定了它们的局限性。然后，我们提出了一种新的攻击框架--梯度平衡多任务攻击(GB-MTA)，该框架将攻击多任务模型视为一个优化问题。该问题基于任务间的平均相对损失变化，被近似为一个整数线性规划问题。对MTL基准、NYUv2和Tiny-Taxonomy的广泛评估表明，GB-MTA相对于标准和反向训练的多任务模型都是有效的。结果还强调了通过参数共享提高任务准确性和由于增强攻击可转移性而增加模型脆弱性之间的权衡。



## **35. Adversarial Attacks on LoRa Device Identification and Rogue Signal Detection with Deep Learning**

基于深度学习的LORA设备识别和恶意信号检测的对抗性攻击 cs.CR

**SubmitDate**: 2023-12-27    [abs](http://arxiv.org/abs/2312.16715v1) [paper-pdf](http://arxiv.org/pdf/2312.16715v1)

**Authors**: Yalin E. Sagduyu, Tugba Erpek

**Abstract**: Low-Power Wide-Area Network (LPWAN) technologies, such as LoRa, have gained significant attention for their ability to enable long-range, low-power communication for Internet of Things (IoT) applications. However, the security of LoRa networks remains a major concern, particularly in scenarios where device identification and classification of legitimate and spoofed signals are crucial. This paper studies a deep learning framework to address these challenges, considering LoRa device identification and legitimate vs. rogue LoRa device classification tasks. A deep neural network (DNN), either a convolutional neural network (CNN) or feedforward neural network (FNN), is trained for each task by utilizing real experimental I/Q data for LoRa signals, while rogue signals are generated by using kernel density estimation (KDE) of received signals by rogue devices. Fast Gradient Sign Method (FGSM)-based adversarial attacks are considered for LoRa signal classification tasks using deep learning models. The impact of these attacks is assessed on the performance of two tasks, namely device identification and legitimate vs. rogue device classification, by utilizing separate or common perturbations against these signal classification tasks. Results presented in this paper quantify the level of transferability of adversarial attacks on different LoRa signal classification tasks as a major vulnerability and highlight the need to make IoT applications robust to adversarial attacks.

摘要: 低功耗广域网(LPWAN)技术，如LORA，因其能够为物联网(IoT)应用实现远距离、低功耗通信而受到广泛关注。然而，LORA网络的安全仍然是一个主要问题，特别是在设备识别以及合法和欺骗信号的分类至关重要的情况下。本文研究了一种深度学习框架来应对这些挑战，考虑了LoRa设备识别和合法与恶意LoRa设备分类任务。通过利用LORA信号的真实实验I/Q数据为每个任务训练深度神经网络(DNN)，或者卷积神经网络(CNN)或前馈神经网络(FNN)，而恶意设备通过使用接收信号的核密度估计(KDE)来生成恶意信号。针对基于深度学习的LORA信号分类任务，提出了一种基于快速梯度符号方法(FGSM)的对抗性攻击方法。通过利用针对这些信号分类任务的单独或共同的扰动来评估这些攻击对两个任务(即设备识别和合法与恶意设备分类)的性能的影响。本文的结果量化了不同LORA信号分类任务上的对抗性攻击的可转移性水平，并强调了使物联网应用程序对对抗性攻击具有健壮性的必要性。



## **36. Frauds Bargain Attack: Generating Adversarial Text Samples via Word Manipulation Process**

欺诈讨价还价攻击：通过文字处理过程生成敌意文本样本 cs.CL

21 pages, 9 tables, 3 figures

**SubmitDate**: 2023-12-27    [abs](http://arxiv.org/abs/2303.01234v2) [paper-pdf](http://arxiv.org/pdf/2303.01234v2)

**Authors**: Mingze Ni, Zhensu Sun, Wei Liu

**Abstract**: Recent research has revealed that natural language processing (NLP) models are vulnerable to adversarial examples. However, the current techniques for generating such examples rely on deterministic heuristic rules, which fail to produce optimal adversarial examples. In response, this study proposes a new method called the Fraud's Bargain Attack (FBA), which uses a randomization mechanism to expand the search space and produce high-quality adversarial examples with a higher probability of success. FBA uses the Metropolis-Hasting sampler, a type of Markov Chain Monte Carlo sampler, to improve the selection of adversarial examples from all candidates generated by a customized stochastic process called the Word Manipulation Process (WMP). The WMP method modifies individual words in a contextually-aware manner through insertion, removal, or substitution. Through extensive experiments, this study demonstrates that FBA outperforms other methods in terms of attack success rate, imperceptibility and sentence quality.

摘要: 最近的研究表明，自然语言处理（NLP）模型容易受到对抗性示例的影响。然而，目前生成此类示例的技术依赖于确定性启发式规则，无法生成最佳对抗示例。作为回应，这项研究提出了一种名为欺诈交易攻击（FBA）的新方法，该方法使用随机化机制来扩展搜索空间，并以更高的成功概率生成高质量的对抗性示例。FBA使用Metropolis-Hasting采样器，一种马尔可夫链蒙特卡罗采样器，以改善从所有候选人中选择对抗性示例的能力，这些候选人是由一个名为Word Manipulation Process（WMP）的自定义随机过程生成的。WMP方法通过插入、删除或替换以上下文感知的方式修改单个单词。通过大量的实验，本研究表明，FBA优于其他方法在攻击成功率，不可感知性和句子质量。



## **37. Evaluating the security of CRYSTALS-Dilithium in the quantum random oracle model**

在量子随机预言模型中评估晶体双锂的安全性 cs.CR

21 pages

**SubmitDate**: 2023-12-27    [abs](http://arxiv.org/abs/2312.16619v1) [paper-pdf](http://arxiv.org/pdf/2312.16619v1)

**Authors**: Kelsey A. Jackson, Carl A. Miller, Daochen Wang

**Abstract**: In the wake of recent progress on quantum computing hardware, the National Institute of Standards and Technology (NIST) is standardizing cryptographic protocols that are resistant to attacks by quantum adversaries. The primary digital signature scheme that NIST has chosen is CRYSTALS-Dilithium. The hardness of this scheme is based on the hardness of three computational problems: Module Learning with Errors (MLWE), Module Short Integer Solution (MSIS), and SelfTargetMSIS. MLWE and MSIS have been well-studied and are widely believed to be secure. However, SelfTargetMSIS is novel and, though classically as hard as MSIS, its quantum hardness is unclear. In this paper, we provide the first proof of the hardness of SelfTargetMSIS via a reduction from MLWE in the Quantum Random Oracle Model (QROM). Our proof uses recently developed techniques in quantum reprogramming and rewinding. A central part of our approach is a proof that a certain hash function, derived from the MSIS problem, is collapsing. From this approach, we deduce a new security proof for Dilithium under appropriate parameter settings. Compared to the only other rigorous security proof for a variant of Dilithium, Dilithium-QROM, our proof has the advantage of being applicable under the condition q = 1 mod 2n, where q denotes the modulus and n the dimension of the underlying algebraic ring. This condition is part of the original Dilithium proposal and is crucial for the efficient implementation of the scheme. We provide new secure parameter sets for Dilithium under the condition q = 1 mod 2n, finding that our public key sizes and signature sizes are about 2.5 to 2.8 times larger than those of Dilithium-QROM for the same security levels.

摘要: 随着量子计算硬件的最新进展，美国国家标准与技术研究所(NIST)正在对能够抵抗量子对手攻击的密码协议进行标准化。NIST选择的主要数字签名方案是Crystal-Dilithium。该方案的难易程度基于三个计算问题的难易程度：带错误的模块学习(MLWE)、模块短整数解(MSIS)和自目标短整数解。MLWE和MSIS已经得到了很好的研究，并被广泛认为是安全的。然而，SelfTargetMSIS是新颖的，尽管经典上和MSIS一样难，但它的量子硬度尚不清楚。本文通过对量子随机Oracle模型(QROM)中MLWE的简化，首次证明了自目标MSIS的硬度。我们的证明使用了最近发展起来的量子重编程和倒带技术。我们方法的一个核心部分是证明从MSIS问题派生的某个散列函数正在崩溃。通过这种方法，我们在适当的参数设置下，给出了Dilithium的一个新的安全证明。与Dilithium变种Dilithium-Qrom的另一个严格的安全证明相比，我们的证明具有在q=1mod2n的条件下适用的优点，其中q表示模且n表示基础代数环的维度。这一条件是最初的Dilithium提议的一部分，对该计划的有效实施至关重要。在Q=1 mod 2n的条件下，我们给出了Dilithium的新的安全参数集，发现在相同的安全级别下，我们的公钥长度和签名长度大约是Dilithium-QROM的2.5到2.8倍。



## **38. Natural Adversarial Patch Generation Method Based on Latent Diffusion Model**

基于潜在扩散模型的自然对抗性补丁生成方法 cs.CV

**SubmitDate**: 2023-12-27    [abs](http://arxiv.org/abs/2312.16401v1) [paper-pdf](http://arxiv.org/pdf/2312.16401v1)

**Authors**: Xianyi Chen, Fazhan Liu, Dong Jiang, Kai Yan

**Abstract**: Recently, some research show that deep neural networks are vulnerable to the adversarial attacks, the well-trainned samples or patches could be used to trick the neural network detector or human visual perception. However, these adversarial patches, with their conspicuous and unusual patterns, lack camouflage and can easily raise suspicion in the real world. To solve this problem, this paper proposed a novel adversarial patch method called the Latent Diffusion Patch (LDP), in which, a pretrained encoder is first designed to compress the natural images into a feature space with key characteristics. Then trains the diffusion model using the above feature space. Finally, explore the latent space of the pretrained diffusion model using the image denoising technology. It polishes the patches and images through the powerful natural abilities of diffusion models, making them more acceptable to the human visual system. Experimental results, both digital and physical worlds, show that LDPs achieve a visual subjectivity score of 87.3%, while still maintaining effective attack capabilities.

摘要: 最近的研究表明，深度神经网络很容易受到敌意攻击，训练好的样本或补丁可以用来欺骗神经网络检测器或人类的视觉感知。然而，这些对抗性补丁具有明显和不寻常的图案，缺乏伪装性，很容易在现实世界中引起怀疑。为了解决这一问题，本文提出了一种新的对抗性补丁方法，称为潜在扩散补丁(LDP)，该方法首先设计一个预先训练的编码器，将自然图像压缩到具有关键特征的特征空间中。然后利用上述特征空间对扩散模型进行训练。最后，利用图像去噪技术探索预训练扩散模型的潜在空间。它通过扩散模型强大的自然能力来打磨补丁和图像，使它们更容易被人类视觉系统接受。实验结果表明，LDPS在保持有效攻击能力的同时，获得了87.3%的视觉主观性分数。



## **39. SlowTrack: Increasing the Latency of Camera-based Perception in Autonomous Driving Using Adversarial Examples**

SlowTrack：使用对抗性例子增加自动驾驶中基于摄像头的感知的延迟 cs.CV

Accepted by AAAI 2024

**SubmitDate**: 2023-12-26    [abs](http://arxiv.org/abs/2312.09520v2) [paper-pdf](http://arxiv.org/pdf/2312.09520v2)

**Authors**: Chen Ma, Ningfei Wang, Qi Alfred Chen, Chao Shen

**Abstract**: In Autonomous Driving (AD), real-time perception is a critical component responsible for detecting surrounding objects to ensure safe driving. While researchers have extensively explored the integrity of AD perception due to its safety and security implications, the aspect of availability (real-time performance) or latency has received limited attention. Existing works on latency-based attack have focused mainly on object detection, i.e., a component in camera-based AD perception, overlooking the entire camera-based AD perception, which hinders them to achieve effective system-level effects, such as vehicle crashes. In this paper, we propose SlowTrack, a novel framework for generating adversarial attacks to increase the execution time of camera-based AD perception. We propose a novel two-stage attack strategy along with the three new loss function designs. Our evaluation is conducted on four popular camera-based AD perception pipelines, and the results demonstrate that SlowTrack significantly outperforms existing latency-based attacks while maintaining comparable imperceptibility levels. Furthermore, we perform the evaluation on Baidu Apollo, an industry-grade full-stack AD system, and LGSVL, a production-grade AD simulator, with two scenarios to compare the system-level effects of SlowTrack and existing attacks. Our evaluation results show that the system-level effects can be significantly improved, i.e., the vehicle crash rate of SlowTrack is around 95% on average while existing works only have around 30%.

摘要: 在自动驾驶中，实时感知是负责检测周围物体以确保安全驾驶的关键部件。虽然由于AD感知的安全性和安全性，研究人员已经对其完整性进行了广泛的探索，但可用性(实时性能)或延迟方面的关注有限。现有的基于延迟攻击的研究主要集中于目标检测，即基于摄像头的广告感知中的一个组件，而忽略了整个基于摄像头的广告感知，这阻碍了它们达到有效的系统级效果，如车辆碰撞。在本文中，我们提出了一种新的生成敌意攻击的框架SlowTrack，以增加基于摄像机的广告感知的执行时间。我们提出了一种新的两阶段攻击策略以及三种新的损失函数设计。我们在四个流行的基于摄像头的AD感知管道上进行了评估，结果表明，SlowTrack在保持相当的不可感知性水平的同时，显著优于现有的基于延迟的攻击。此外，我们在工业级全栈AD系统百度Apollo和生产级AD模拟器LGSVL上进行了评估，并通过两个场景比较了SlowTrack和现有攻击的系统级影响。我们的评估结果表明，系统级效果可以得到显著提高，即SlowTrack的车辆撞击率平均在95%左右，而现有的工作只有30%左右。



## **40. Model Stealing Attack against Recommender System**

针对推荐系统的模型窃取攻击 cs.CR

**SubmitDate**: 2023-12-26    [abs](http://arxiv.org/abs/2312.11571v2) [paper-pdf](http://arxiv.org/pdf/2312.11571v2)

**Authors**: Zhihao Zhu, Rui Fan, Chenwang Wu, Yi Yang, Defu Lian, Enhong Chen

**Abstract**: Recent studies have demonstrated the vulnerability of recommender systems to data privacy attacks. However, research on the threat to model privacy in recommender systems, such as model stealing attacks, is still in its infancy. Some adversarial attacks have achieved model stealing attacks against recommender systems, to some extent, by collecting abundant training data of the target model (target data) or making a mass of queries. In this paper, we constrain the volume of available target data and queries and utilize auxiliary data, which shares the item set with the target data, to promote model stealing attacks. Although the target model treats target and auxiliary data differently, their similar behavior patterns allow them to be fused using an attention mechanism to assist attacks. Besides, we design stealing functions to effectively extract the recommendation list obtained by querying the target model. Experimental results show that the proposed methods are applicable to most recommender systems and various scenarios and exhibit excellent attack performance on multiple datasets.

摘要: 最近的研究表明，推荐系统对数据隐私攻击是脆弱的。然而，对推荐系统中模型隐私威胁的研究，如模型窃取攻击，还处于起步阶段。一些对抗性攻击通过收集目标模型(目标数据)的大量训练数据或进行大量查询，在一定程度上实现了对推荐系统的模型窃取攻击。本文通过限制可用目标数据和查询的数据量，利用与目标数据共享项集的辅助数据来促进模型窃取攻击。尽管目标模型对目标和辅助数据的处理不同，但它们相似的行为模式允许使用注意力机制将它们融合在一起，以帮助攻击。此外，我们还设计了窃取函数来有效地提取通过查询目标模型获得的推荐列表。实验结果表明，所提出的方法适用于大多数推荐系统和各种场景，并在多个数据集上表现出良好的攻击性能。



## **41. MENLI: Robust Evaluation Metrics from Natural Language Inference**

MENLI：自然语言推理中的稳健评价指标 cs.CL

TACL 2023 Camera-ready version; updated after proofreading by the  journal

**SubmitDate**: 2023-12-26    [abs](http://arxiv.org/abs/2208.07316v5) [paper-pdf](http://arxiv.org/pdf/2208.07316v5)

**Authors**: Yanran Chen, Steffen Eger

**Abstract**: Recently proposed BERT-based evaluation metrics for text generation perform well on standard benchmarks but are vulnerable to adversarial attacks, e.g., relating to information correctness. We argue that this stems (in part) from the fact that they are models of semantic similarity. In contrast, we develop evaluation metrics based on Natural Language Inference (NLI), which we deem a more appropriate modeling. We design a preference-based adversarial attack framework and show that our NLI based metrics are much more robust to the attacks than the recent BERT-based metrics. On standard benchmarks, our NLI based metrics outperform existing summarization metrics, but perform below SOTA MT metrics. However, when combining existing metrics with our NLI metrics, we obtain both higher adversarial robustness (15%-30%) and higher quality metrics as measured on standard benchmarks (+5% to 30%).

摘要: 最近提出的基于BERT的文本生成评估指标在标准基准上表现良好，但容易受到敌意攻击，例如与信息正确性有关的攻击。我们认为，这(部分)源于这样一个事实：它们是语义相似性的模型。相比之下，我们基于自然语言推理(NLI)开发评估指标，我们认为这是更合适的建模。我们设计了一个基于偏好的对抗性攻击框架，并表明我们的基于NLI的度量比最近的基于BERT的度量具有更强的抗攻击能力。在标准基准测试中，我们基于NLI的指标优于现有的摘要指标，但低于SOTA MT指标。然而，当将现有指标与我们的NLI指标相结合时，我们获得了更高的对手健壮性(15%-30%)和标准基准测试的更高质量指标(+5%到30%)。



## **42. Punctuation Matters! Stealthy Backdoor Attack for Language Models**

标点符号很重要！对语言模型的秘密后门攻击 cs.CL

NLPCC 2023

**SubmitDate**: 2023-12-26    [abs](http://arxiv.org/abs/2312.15867v1) [paper-pdf](http://arxiv.org/pdf/2312.15867v1)

**Authors**: Xuan Sheng, Zhicheng Li, Zhaoyang Han, Xiangmao Chang, Piji Li

**Abstract**: Recent studies have pointed out that natural language processing (NLP) models are vulnerable to backdoor attacks. A backdoored model produces normal outputs on the clean samples while performing improperly on the texts with triggers that the adversary injects. However, previous studies on textual backdoor attack pay little attention to stealthiness. Moreover, some attack methods even cause grammatical issues or change the semantic meaning of the original texts. Therefore, they can easily be detected by humans or defense systems. In this paper, we propose a novel stealthy backdoor attack method against textual models, which is called \textbf{PuncAttack}. It leverages combinations of punctuation marks as the trigger and chooses proper locations strategically to replace them. Through extensive experiments, we demonstrate that the proposed method can effectively compromise multiple models in various tasks. Meanwhile, we conduct automatic evaluation and human inspection, which indicate the proposed method possesses good performance of stealthiness without bringing grammatical issues and altering the meaning of sentences.

摘要: 最近的研究指出，自然语言处理(NLP)模型容易受到后门攻击。反向模型在干净的样本上产生正常输出，而在带有对手注入的触发器的文本上执行不正确的操作。然而，以往对文本后门攻击的研究很少关注隐蔽性。此外，一些攻击方法甚至会引起语法问题或改变原文的语义。因此，它们很容易被人类或防御系统检测到。提出了一种新的针对文本模型的隐蔽后门攻击方法-.它利用标点符号的组合作为触发器，并战略性地选择适当的位置来取代它们。通过大量的实验，我们证明了该方法能够在不同的任务中有效地折衷多个模型。同时，我们进行了自动评估和人工检测，表明该方法具有良好的隐蔽性，不会带来语法问题，也不会改变句子的意义。



## **43. Attention Deficit is Ordered! Fooling Deformable Vision Transformers with Collaborative Adversarial Patches**

注意力缺陷是命中注定的！用协同对抗性补丁愚弄可变形视觉变形器 cs.CV

12 pages, 14 figures

**SubmitDate**: 2023-12-25    [abs](http://arxiv.org/abs/2311.12914v2) [paper-pdf](http://arxiv.org/pdf/2311.12914v2)

**Authors**: Quazi Mishkatul Alam, Bilel Tarchoun, Ihsen Alouani, Nael Abu-Ghazaleh

**Abstract**: The latest generation of transformer-based vision models has proven to be superior to Convolutional Neural Network (CNN)-based models across several vision tasks, largely attributed to their remarkable prowess in relation modeling. Deformable vision transformers significantly reduce the quadratic complexity of attention modeling by using sparse attention structures, enabling them to incorporate features across different scales and be used in large-scale applications, such as multi-view vision systems. Recent work has demonstrated adversarial attacks against conventional vision transformers; we show that these attacks do not transfer to deformable transformers due to their sparse attention structure. Specifically, attention in deformable transformers is modeled using pointers to the most relevant other tokens. In this work, we contribute for the first time adversarial attacks that manipulate the attention of deformable transformers, redirecting it to focus on irrelevant parts of the image. We also develop new collaborative attacks where a source patch manipulates attention to point to a target patch, which contains the adversarial noise to fool the model. In our experiments, we observe that altering less than 1% of the patched area in the input field results in a complete drop to 0% AP in single-view object detection using MS COCO and a 0% MODA in multi-view object detection using Wildtrack.

摘要: 最新一代的基于变压器的视觉模型已被证明在几个视觉任务上优于基于卷积神经网络(CNN)的模型，这在很大程度上归功于它们在关系建模方面的非凡能力。可变形视觉转换器通过使用稀疏注意力结构显著降低了注意力建模的二次方复杂性，使其能够合并不同尺度上的特征，并用于大规模应用，如多视角视觉系统。最近的工作证明了针对传统视觉转换器的对抗性攻击；我们表明，由于其稀疏的注意结构，这些攻击不会转移到可变形的转换器上。具体地说，在可变形转换器中的注意力是使用指向最相关的其他标记的指针来建模的。在这项工作中，我们第一次贡献了对抗性攻击，操纵变形变形者的注意力，将其重新定向到图像中不相关的部分。我们还开发了新的协作攻击，其中源补丁操纵注意力指向目标补丁，目标补丁包含敌意噪声来愚弄模型。在我们的实验中，我们观察到，在输入区域中改变不到1%的修补面积会导致在使用MS Coco的单视图目标检测中完全下降到0%AP，在使用WildTrack的多视点目标检测中完全下降到0%Moda。



## **44. Adversarial Prompt Tuning for Vision-Language Models**

视觉语言模型的对抗性提示调整 cs.CV

**SubmitDate**: 2023-12-25    [abs](http://arxiv.org/abs/2311.11261v2) [paper-pdf](http://arxiv.org/pdf/2311.11261v2)

**Authors**: Jiaming Zhang, Xingjun Ma, Xin Wang, Lingyu Qiu, Jiaqi Wang, Yu-Gang Jiang, Jitao Sang

**Abstract**: With the rapid advancement of multimodal learning, pre-trained Vision-Language Models (VLMs) such as CLIP have demonstrated remarkable capacities in bridging the gap between visual and language modalities. However, these models remain vulnerable to adversarial attacks, particularly in the image modality, presenting considerable security risks. This paper introduces Adversarial Prompt Tuning (AdvPT), a novel technique to enhance the adversarial robustness of image encoders in VLMs. AdvPT innovatively leverages learnable text prompts and aligns them with adversarial image embeddings, to address the vulnerabilities inherent in VLMs without the need for extensive parameter training or modification of the model architecture. We demonstrate that AdvPT improves resistance against white-box and black-box adversarial attacks and exhibits a synergistic effect when combined with existing image-processing-based defense techniques, further boosting defensive capabilities. Comprehensive experimental analyses provide insights into adversarial prompt tuning, a novel paradigm devoted to improving resistance to adversarial images through textual input modifications, paving the way for future robust multimodal learning research. These findings open up new possibilities for enhancing the security of VLMs. Our code is available at https://github.com/jiamingzhang94/Adversarial-Prompt-Tuning.

摘要: 随着多通道学习的快速发展，诸如CLIP等预先训练的视觉语言模型在弥合视觉和语言通道之间的差距方面显示出了显著的能力。然而，这些模型仍然容易受到敌意攻击，特别是在图像模式方面，这带来了相当大的安全风险。本文介绍了对抗性提示调优(AdvPT)技术，这是一种在VLMS中增强图像编码器对抗性稳健性的新技术。AdvPT创新性地利用可学习的文本提示，并将其与对抗性图像嵌入相结合，以解决VLM中固有的漏洞，而无需进行广泛的参数培训或修改模型体系结构。我们证明，AdvPT提高了对白盒和黑盒攻击的抵抗力，并与现有的基于图像处理的防御技术相结合，显示出协同效应，进一步增强了防御能力。全面的实验分析提供了对对抗性即时调整的见解，这是一种致力于通过修改文本输入来提高对对抗性图像的抵抗力的新范式，为未来稳健的多通道学习研究铺平了道路。这些发现为增强VLM的安全性开辟了新的可能性。我们的代码可以在https://github.com/jiamingzhang94/Adversarial-Prompt-Tuning.上找到



## **45. Vulnerability of Machine Learning Approaches Applied in IoT-based Smart Grid: A Review**

物联网智能电网中机器学习方法的脆弱性研究 cs.CR

**SubmitDate**: 2023-12-25    [abs](http://arxiv.org/abs/2308.15736v3) [paper-pdf](http://arxiv.org/pdf/2308.15736v3)

**Authors**: Zhenyong Zhang, Mengxiang Liu, Mingyang Sun, Ruilong Deng, Peng Cheng, Dusit Niyato, Mo-Yuen Chow, Jiming Chen

**Abstract**: Machine learning (ML) sees an increasing prevalence of being used in the internet-of-things (IoT)-based smart grid. However, the trustworthiness of ML is a severe issue that must be addressed to accommodate the trend of ML-based smart grid applications (MLsgAPPs). The adversarial distortion injected into the power signal will greatly affect the system's normal control and operation. Therefore, it is imperative to conduct vulnerability assessment for MLsgAPPs applied in the context of safety-critical power systems. In this paper, we provide a comprehensive review of the recent progress in designing attack and defense methods for MLsgAPPs. Unlike the traditional survey about ML security, this is the first review work about the security of MLsgAPPs that focuses on the characteristics of power systems. We first highlight the specifics for constructing the adversarial attacks on MLsgAPPs. Then, the vulnerability of MLsgAPP is analyzed from both the aspects of the power system and ML model. Afterward, a comprehensive survey is conducted to review and compare existing studies about the adversarial attacks on MLsgAPPs in scenarios of generation, transmission, distribution, and consumption, and the countermeasures are reviewed according to the attacks that they defend against. Finally, the future research directions are discussed on the attacker's and defender's side, respectively. We also analyze the potential vulnerability of large language model-based (e.g., ChatGPT) power system applications. Overall, we encourage more researchers to contribute to investigating the adversarial issues of MLsgAPPs.

摘要: 机器学习(ML)在基于物联网(IoT)的智能电网中的应用越来越普遍。然而，ML的可信性是一个必须解决的严重问题，以适应基于ML的智能电网应用(MLsgAPP)的趋势。注入到电源信号中的对抗性失真将极大地影响系统的正常控制和运行。因此，对应用于安全关键电力系统背景下的MLsgAPP进行脆弱性评估势在必行。在本文中，我们提供了一个全面的进展，设计攻击和防御方法的MLsgAPP。与传统的ML安全研究不同，本文首次针对电力系统的特点对MLsgAPP的安全问题进行了综述。我们首先强调构造对MLsgAPP的对抗性攻击的细节。然后，从电力系统和ML模型两个方面分析了MLsgAPP的脆弱性。然后，对已有的针对MLsgAPP的生成、传输、分发、消费等场景下的对抗性攻击的研究进行了全面的回顾和比较，并根据它们所防御的攻击回顾了相应的对策。最后，分别从攻击方和防御方的角度讨论了今后的研究方向。我们还分析了基于大型语言模型(如ChatGPT)的电力系统应用的潜在脆弱性。总体而言，我们鼓励更多的研究人员为研究MLsgAPP的对抗性问题做出贡献。



## **46. Privacy-Preserving Neural Graph Databases**

保护隐私的神经图库 cs.DB

**SubmitDate**: 2023-12-25    [abs](http://arxiv.org/abs/2312.15591v1) [paper-pdf](http://arxiv.org/pdf/2312.15591v1)

**Authors**: Qi Hu, Haoran Li, Jiaxin Bai, Yangqiu Song

**Abstract**: In the era of big data and rapidly evolving information systems, efficient and accurate data retrieval has become increasingly crucial. Neural graph databases (NGDBs) have emerged as a powerful paradigm that combines the strengths of graph databases (graph DBs) and neural networks to enable efficient storage, retrieval, and analysis of graph-structured data. The usage of neural embedding storage and complex neural logical query answering provides NGDBs with generalization ability. When the graph is incomplete, by extracting latent patterns and representations, neural graph databases can fill gaps in the graph structure, revealing hidden relationships and enabling accurate query answering. Nevertheless, this capability comes with inherent trade-offs, as it introduces additional privacy risks to the database. Malicious attackers can infer more sensitive information in the database using well-designed combinatorial queries, such as by comparing the answer sets of where Turing Award winners born before 1950 and after 1940 lived, the living places of Turing Award winner Hinton are probably exposed, although the living places may have been deleted in the training due to the privacy concerns. In this work, inspired by the privacy protection in graph embeddings, we propose a privacy-preserving neural graph database (P-NGDB) to alleviate the risks of privacy leakage in NGDBs. We introduce adversarial training techniques in the training stage to force the NGDBs to generate indistinguishable answers when queried with private information, enhancing the difficulty of inferring sensitive information through combinations of multiple innocuous queries. Extensive experiment results on three datasets show that P-NGDB can effectively protect private information in the graph database while delivering high-quality public answers responses to queries.

摘要: 在大数据和快速发展的信息系统的时代，高效和准确的数据检索变得越来越重要。神经图形数据库(NGDB)已经成为一种强大的范例，它结合了图形数据库(图形数据库)和神经网络的优点，使得能够有效地存储、检索和分析图形结构的数据。神经嵌入存储和复杂神经逻辑查询回答的使用为NGDB提供了泛化能力。当图不完整时，通过提取潜在模式和表示，神经图库可以填补图结构中的空白，揭示隐藏的关系，并使查询得到准确的回答。尽管如此，这种能力也伴随着固有的权衡，因为它会给数据库带来额外的隐私风险。恶意攻击者可以使用精心设计的组合查询来推断数据库中更敏感的信息，例如通过比较1950年之前出生的图灵奖获得者和1940年后出生的图灵奖获得者的答案集，图灵奖获得者辛顿的居住地可能会被曝光，尽管出于隐私考虑，在训练中可能已经删除了居住地。在这项工作中，我们受到图嵌入中隐私保护的启发，提出了一种隐私保护神经图库(P-NGDB)来缓解NGDB中隐私泄露的风险。我们在训练阶段引入对抗性训练技术，迫使NGDB在查询私有信息时产生难以区分的答案，增加了通过组合多个无害查询来推断敏感信息的难度。在三个数据集上的大量实验结果表明，P-NGDB可以有效地保护图形数据库中的私有信息，同时提供高质量的公共查询响应。



## **47. Adversarial Data Poisoning for Fake News Detection: How to Make a Model Misclassify a Target News without Modifying It**

用于虚假新闻检测的对抗性数据中毒：如何使模型在不修改目标新闻的情况下对其进行错误分类 cs.LG

**SubmitDate**: 2024-01-04    [abs](http://arxiv.org/abs/2312.15228v2) [paper-pdf](http://arxiv.org/pdf/2312.15228v2)

**Authors**: Federico Siciliano, Luca Maiano, Lorenzo Papa, Federica Baccini, Irene Amerini, Fabrizio Silvestri

**Abstract**: Fake news detection models are critical to countering disinformation but can be manipulated through adversarial attacks. In this position paper, we analyze how an attacker can compromise the performance of an online learning detector on specific news content without being able to manipulate the original target news. In some contexts, such as social networks, where the attacker cannot exert complete control over all the information, this scenario can indeed be quite plausible. Therefore, we show how an attacker could potentially introduce poisoning data into the training data to manipulate the behavior of an online learning method. Our initial findings reveal varying susceptibility of logistic regression models based on complexity and attack type.

摘要: 假新闻检测模型对打击虚假信息至关重要，但可以通过对抗性攻击来操纵。在这份立场文件中，我们分析了攻击者如何在不能操纵原始目标新闻的情况下，损害在线学习检测器在特定新闻内容上的性能。在某些情况下，例如社交网络，攻击者无法完全控制所有信息，这种情况确实很有可能发生。因此，我们展示了攻击者如何潜在地将中毒数据引入训练数据以操纵在线学习方法的行为。我们的初步发现揭示了基于复杂性和攻击类型的Logistic回归模型的不同易感性。



## **48. Towards Transferable Adversarial Attacks with Centralized Perturbation**

集中式扰动下的可转移对抗性攻击 cs.CV

10 pages, 9 figures, accepted by AAAI 2024

**SubmitDate**: 2023-12-23    [abs](http://arxiv.org/abs/2312.06199v2) [paper-pdf](http://arxiv.org/pdf/2312.06199v2)

**Authors**: Shangbo Wu, Yu-an Tan, Yajie Wang, Ruinan Ma, Wencong Ma, Yuanzhang Li

**Abstract**: Adversarial transferability enables black-box attacks on unknown victim deep neural networks (DNNs), rendering attacks viable in real-world scenarios. Current transferable attacks create adversarial perturbation over the entire image, resulting in excessive noise that overfit the source model. Concentrating perturbation to dominant image regions that are model-agnostic is crucial to improving adversarial efficacy. However, limiting perturbation to local regions in the spatial domain proves inadequate in augmenting transferability. To this end, we propose a transferable adversarial attack with fine-grained perturbation optimization in the frequency domain, creating centralized perturbation. We devise a systematic pipeline to dynamically constrain perturbation optimization to dominant frequency coefficients. The constraint is optimized in parallel at each iteration, ensuring the directional alignment of perturbation optimization with model prediction. Our approach allows us to centralize perturbation towards sample-specific important frequency features, which are shared by DNNs, effectively mitigating source model overfitting. Experiments demonstrate that by dynamically centralizing perturbation on dominating frequency coefficients, crafted adversarial examples exhibit stronger transferability, and allowing them to bypass various defenses.

摘要: 对抗的可转移性使对未知受害者的黑盒攻击能够深入神经网络(DNN)，从而使攻击在现实世界的场景中可行。当前的可转移攻击在整个图像上造成对抗性扰动，导致过多的噪声超出源模型的范围。将扰动集中到模型不可知的优势图像区域是提高对抗效能的关键。然而，将扰动限制在空间域中的局部区域被证明不足以增强可转移性。为此，我们提出了一种在频域进行细粒度扰动优化的可转移敌意攻击，产生集中扰动。我们设计了一个系统的管道来动态地将摄动优化约束到主导频率系数。在每次迭代中并行优化约束，确保扰动优化与模型预测的方向一致。我们的方法允许我们集中对样本特定的重要频率特征的扰动，这些特征由DNN共享，有效地缓解了源模型的过度拟合。实验表明，通过动态地将扰动集中在支配频率系数上，精心制作的敌意例子表现出更强的可转移性，并允许它们绕过各种防御。



## **49. SODA: Protecting Proprietary Information in On-Device Machine Learning Models**

SODA：在设备上机器学习模型中保护专有信息 cs.LG

**SubmitDate**: 2023-12-22    [abs](http://arxiv.org/abs/2312.15036v1) [paper-pdf](http://arxiv.org/pdf/2312.15036v1)

**Authors**: Akanksha Atrey, Ritwik Sinha, Saayan Mitra, Prashant Shenoy

**Abstract**: The growth of low-end hardware has led to a proliferation of machine learning-based services in edge applications. These applications gather contextual information about users and provide some services, such as personalized offers, through a machine learning (ML) model. A growing practice has been to deploy such ML models on the user's device to reduce latency, maintain user privacy, and minimize continuous reliance on a centralized source. However, deploying ML models on the user's edge device can leak proprietary information about the service provider. In this work, we investigate on-device ML models that are used to provide mobile services and demonstrate how simple attacks can leak proprietary information of the service provider. We show that different adversaries can easily exploit such models to maximize their profit and accomplish content theft. Motivated by the need to thwart such attacks, we present an end-to-end framework, SODA, for deploying and serving on edge devices while defending against adversarial usage. Our results demonstrate that SODA can detect adversarial usage with 89% accuracy in less than 50 queries with minimal impact on service performance, latency, and storage.

摘要: 低端硬件的增长导致了边缘应用中基于机器学习的服务的激增。这些应用程序收集有关用户的上下文信息，并通过机器学习(ML)模型提供一些服务，如个性化服务。越来越多的做法是在用户设备上部署这样的ML模型，以减少延迟，维护用户隐私，并最大限度地减少对集中式来源的持续依赖。然而，在用户的边缘设备上部署ML模型可能会泄露有关服务提供商的专有信息。在这项工作中，我们研究了用于提供移动服务的设备上ML模型，并演示了简单的攻击如何泄漏服务提供商的专有信息。我们表明，不同的对手可以很容易地利用这种模型来最大化他们的利润，并完成内容窃取。出于阻止此类攻击的需要，我们提出了一个端到端框架SODA，用于在边缘设备上部署和服务，同时防御恶意使用。我们的结果表明，SODA可以在不到50个查询的情况下以89%的准确率检测恶意使用，并且对服务性能、延迟和存储的影响最小。



## **50. Differentiable JPEG: The Devil is in the Details**

与众不同的JPEG：魔鬼在细节中 cs.CV

Accepted at WACV 2024. Project page:  https://christophreich1996.github.io/differentiable_jpeg/ WACV paper:  https://openaccess.thecvf.com/content/WACV2024/html/Reich_Differentiable_JPEG_The_Devil_Is_in_the_Details_WACV_2024_paper.html

**SubmitDate**: 2023-12-22    [abs](http://arxiv.org/abs/2309.06978v4) [paper-pdf](http://arxiv.org/pdf/2309.06978v4)

**Authors**: Christoph Reich, Biplob Debnath, Deep Patel, Srimat Chakradhar

**Abstract**: JPEG remains one of the most widespread lossy image coding methods. However, the non-differentiable nature of JPEG restricts the application in deep learning pipelines. Several differentiable approximations of JPEG have recently been proposed to address this issue. This paper conducts a comprehensive review of existing diff. JPEG approaches and identifies critical details that have been missed by previous methods. To this end, we propose a novel diff. JPEG approach, overcoming previous limitations. Our approach is differentiable w.r.t. the input image, the JPEG quality, the quantization tables, and the color conversion parameters. We evaluate the forward and backward performance of our diff. JPEG approach against existing methods. Additionally, extensive ablations are performed to evaluate crucial design choices. Our proposed diff. JPEG resembles the (non-diff.) reference implementation best, significantly surpassing the recent-best diff. approach by $3.47$dB (PSNR) on average. For strong compression rates, we can even improve PSNR by $9.51$dB. Strong adversarial attack results are yielded by our diff. JPEG, demonstrating the effective gradient approximation. Our code is available at https://github.com/necla-ml/Diff-JPEG.

摘要: JPEG仍然是应用最广泛的有损图像编码方法之一。然而，JPEG的不可微特性限制了其在深度学习管道中的应用。为了解决这个问题，最近已经提出了几种JPEG的可微近似。本文对现有的DIFF进行了全面的回顾。JPEG处理并确定了以前方法遗漏的关键细节。为此，我们提出了一个新颖的Diff。JPEG方法，克服了以前的限制。我们的方法是可微的W.r.t。输入图像、JPEG质量、量化表和颜色转换参数。我们评估了DIFF的向前和向后性能。JPEG方法与现有方法的对比。此外，还进行了广泛的消融，以评估关键的设计选择。我们提议的不同之处。JPEG与(Non-Diff.)参考实现最好，大大超过了最近最好的差异。平均接近3.47美元分贝(PSNR)。对于强压缩率，我们甚至可以将PSNR提高9.51美元分贝。强大的对抗性攻击结果是由我们的差异产生的。JPEG格式，演示了有效的渐变近似。我们的代码可以在https://github.com/necla-ml/Diff-JPEG.上找到



