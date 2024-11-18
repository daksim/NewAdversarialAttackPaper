# Latest Adversarial Attack Papers
**update at 2024-11-18 09:39:36**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. DiffPAD: Denoising Diffusion-based Adversarial Patch Decontamination**

迪夫pad：消除基于扩散的对抗性补丁净化 cs.CV

Accepted to 2025 IEEE/CVF Winter Conference on Applications of  Computer Vision (WACV)

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2410.24006v2) [paper-pdf](http://arxiv.org/pdf/2410.24006v2)

**Authors**: Jia Fu, Xiao Zhang, Sepideh Pashami, Fatemeh Rahimian, Anders Holst

**Abstract**: In the ever-evolving adversarial machine learning landscape, developing effective defenses against patch attacks has become a critical challenge, necessitating reliable solutions to safeguard real-world AI systems. Although diffusion models have shown remarkable capacity in image synthesis and have been recently utilized to counter $\ell_p$-norm bounded attacks, their potential in mitigating localized patch attacks remains largely underexplored. In this work, we propose DiffPAD, a novel framework that harnesses the power of diffusion models for adversarial patch decontamination. DiffPAD first performs super-resolution restoration on downsampled input images, then adopts binarization, dynamic thresholding scheme and sliding window for effective localization of adversarial patches. Such a design is inspired by the theoretically derived correlation between patch size and diffusion restoration error that is generalized across diverse patch attack scenarios. Finally, DiffPAD applies inpainting techniques to the original input images with the estimated patch region being masked. By integrating closed-form solutions for super-resolution restoration and image inpainting into the conditional reverse sampling process of a pre-trained diffusion model, DiffPAD obviates the need for text guidance or fine-tuning. Through comprehensive experiments, we demonstrate that DiffPAD not only achieves state-of-the-art adversarial robustness against patch attacks but also excels in recovering naturalistic images without patch remnants. The source code is available at https://github.com/JasonFu1998/DiffPAD.

摘要: 在不断发展的对抗性机器学习环境中，开发针对补丁攻击的有效防御已成为一项关键挑战，需要可靠的解决方案来保护真实世界的AI系统。虽然扩散模型在图像合成方面表现出了显著的能力，并且最近已被用于对抗$\ell_p$-范数有界攻击，但它们在缓解局部补丁攻击方面的潜力仍未被充分挖掘。在这项工作中，我们提出了DiffPAD，一个新的框架，它利用扩散模型的力量来进行对抗性补丁去污。DiffPAD首先对下采样的输入图像进行超分辨率恢复，然后采用二值化、动态阈值和滑动窗口等方法对对抗性斑块进行有效定位。这种设计的灵感来自于理论上推导出的补丁大小和扩散恢复误差之间的相关性，该相关性在不同的补丁攻击场景中得到推广。最后，DiffPAD将修复技术应用于原始输入图像，并对估计的补丁区域进行掩蔽。通过将用于超分辨率恢复和图像修复的闭合形式解决方案集成到预先训练的扩散模型的条件反向采样过程中，DiffPAD消除了对文本指导或微调的需要。通过综合实验，我们证明了DiffPAD算法不仅对补丁攻击具有最好的对抗健壮性，而且在恢复没有补丁残留的自然图像方面具有很好的性能。源代码可在https://github.com/JasonFu1998/DiffPAD.上找到



## **2. Are nuclear masks all you need for improved out-of-domain generalisation? A closer look at cancer classification in histopathology**

核口罩就是改进域外通用所需的全部吗？仔细观察组织病理学中的癌症分类 eess.IV

Poster at NeurIPS 2024

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2411.09373v1) [paper-pdf](http://arxiv.org/pdf/2411.09373v1)

**Authors**: Dhananjay Tomar, Alexander Binder, Andreas Kleppe

**Abstract**: Domain generalisation in computational histopathology is challenging because the images are substantially affected by differences among hospitals due to factors like fixation and staining of tissue and imaging equipment. We hypothesise that focusing on nuclei can improve the out-of-domain (OOD) generalisation in cancer detection. We propose a simple approach to improve OOD generalisation for cancer detection by focusing on nuclear morphology and organisation, as these are domain-invariant features critical in cancer detection. Our approach integrates original images with nuclear segmentation masks during training, encouraging the model to prioritise nuclei and their spatial arrangement. Going beyond mere data augmentation, we introduce a regularisation technique that aligns the representations of masks and original images. We show, using multiple datasets, that our method improves OOD generalisation and also leads to increased robustness to image corruptions and adversarial attacks. The source code is available at https://github.com/undercutspiky/SFL/

摘要: 计算组织病理学的领域泛化是具有挑战性的，因为由于组织和成像设备的固定和染色等因素，图像在很大程度上受到医院之间的差异的影响。我们假设，聚焦于核可以改善癌症检测中的域外(OOD)泛化。我们提出了一种简单的方法，通过关注核的形态和组织来改进癌症检测的OOD泛化，因为这些是癌症检测中关键的区域不变特征。我们的方法在训练过程中将原始图像与核分割掩模相结合，鼓励模型优先考虑核及其空间排列。除了单纯的数据增强，我们还引入了一种正则化技术，使蒙版和原始图像的表示保持一致。我们使用多个数据集显示，我们的方法改进了面向对象设计的泛化，并导致对图像损坏和敌意攻击的健壮性增强。源代码可在https://github.com/undercutspiky/SFL/上找到



## **3. Enhancing generalization in high energy physics using white-box adversarial attacks**

使用白盒对抗攻击增强高能物理学的概括性 hep-ph

10 pages, 4 figures, 8 tables, 3 algorithms, to be published in  Physical Review D (PRD), presented at the ML4Jets 2024 conference

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2411.09296v1) [paper-pdf](http://arxiv.org/pdf/2411.09296v1)

**Authors**: Franck Rothen, Samuel Klein, Matthew Leigh, Tobias Golling

**Abstract**: Machine learning is becoming increasingly popular in the context of particle physics. Supervised learning, which uses labeled Monte Carlo (MC) simulations, remains one of the most widely used methods for discriminating signals beyond the Standard Model. However, this paper suggests that supervised models may depend excessively on artifacts and approximations from Monte Carlo simulations, potentially limiting their ability to generalize well to real data. This study aims to enhance the generalization properties of supervised models by reducing the sharpness of local minima. It reviews the application of four distinct white-box adversarial attacks in the context of classifying Higgs boson decay signals. The attacks are divided into weight space attacks, and feature space attacks. To study and quantify the sharpness of different local minima this paper presents two analysis methods: gradient ascent and reduced Hessian eigenvalue analysis. The results show that white-box adversarial attacks significantly improve generalization performance, albeit with increased computational complexity.

摘要: 在粒子物理的背景下，机器学习正变得越来越流行。监督学习使用标记的蒙特卡罗(MC)模拟，仍然是用于区分标准模型以外的信号的最广泛使用的方法之一。然而，本文认为，监督模型可能过度依赖于来自蒙特卡罗模拟的伪影和近似，潜在地限制了它们对真实数据的推广能力。该研究旨在通过降低局部极小值的锐度来增强监督模型的泛化性能。它回顾了四种不同的白盒对抗性攻击在对希格斯玻色子衰变信号进行分类的背景下的应用。攻击分为权重空间攻击和特征空间攻击。为了研究和量化不同局部极小值的锐度，本文提出了两种分析方法：梯度上升法和约化Hesse特征值分析法。结果表明，白盒对抗性攻击显著提高了泛化性能，但增加了计算复杂度。



## **4. BEARD: Benchmarking the Adversarial Robustness for Dataset Distillation**

BEARD：数据集蒸馏的对抗稳健性基准 cs.CV

15 pages, 6 figures

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2411.09265v1) [paper-pdf](http://arxiv.org/pdf/2411.09265v1)

**Authors**: Zheng Zhou, Wenquan Feng, Shuchang Lyu, Guangliang Cheng, Xiaowei Huang, Qi Zhao

**Abstract**: Dataset Distillation (DD) is an emerging technique that compresses large-scale datasets into significantly smaller synthesized datasets while preserving high test performance and enabling the efficient training of large models. However, current research primarily focuses on enhancing evaluation accuracy under limited compression ratios, often overlooking critical security concerns such as adversarial robustness. A key challenge in evaluating this robustness lies in the complex interactions between distillation methods, model architectures, and adversarial attack strategies, which complicate standardized assessments. To address this, we introduce BEARD, an open and unified benchmark designed to systematically assess the adversarial robustness of DD methods, including DM, IDM, and BACON. BEARD encompasses a variety of adversarial attacks (e.g., FGSM, PGD, C&W) on distilled datasets like CIFAR-10/100 and TinyImageNet. Utilizing an adversarial game framework, it introduces three key metrics: Robustness Ratio (RR), Attack Efficiency Ratio (AE), and Comprehensive Robustness-Efficiency Index (CREI). Our analysis includes unified benchmarks, various Images Per Class (IPC) settings, and the effects of adversarial training. Results are available on the BEARD Leaderboard, along with a library providing model and dataset pools to support reproducible research. Access the code at BEARD.

摘要: 数据集精馏是一种新兴的技术，它将大规模数据集压缩成小得多的合成数据集，同时保持高测试性能并实现对大型模型的有效训练。然而，目前的研究主要集中在提高有限压缩比下的评估精度，往往忽略了关键的安全问题，如对手的稳健性。评估这种健壮性的一个关键挑战在于蒸馏方法、模型体系结构和对抗性攻击策略之间的复杂交互，这使得标准化评估复杂化。为了解决这个问题，我们引入了BEARD，这是一个开放和统一的基准，旨在系统地评估DD方法的对抗健壮性，包括DM、IDM和BACON。Beard包含了对CIFAR-10/100和TinyImageNet等提取数据集的各种对抗性攻击(例如，FGSM、PGD、C&W)。利用对抗性博弈框架，引入了三个关键度量：健壮性比(RR)、攻击效率比(AE)和综合健壮性-效率指数(CREI)。我们的分析包括统一的基准，不同的每类图像(IPC)设置，以及对抗性训练的效果。结果可以在胡须排行榜上找到，还有一个提供模型和数据集池的库，以支持可重复的研究。访问比尔德的密码。



## **5. Injection Attacks Against End-to-End Encrypted Applications**

针对端到端加密应用程序的注入攻击 cs.CR

Published in IEEE Security and Privacy 2024

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2411.09228v1) [paper-pdf](http://arxiv.org/pdf/2411.09228v1)

**Authors**: Andrés Fábrega, Carolina Ortega Pérez, Armin Namavari, Ben Nassi, Rachit Agarwal, Thomas Ristenpart

**Abstract**: We explore an emerging threat model for end-to-end (E2E) encrypted applications: an adversary sends chosen messages to a target client, thereby "injecting" adversarial content into the application state. Such state is subsequently encrypted and synchronized to an adversarially-visible storage. By observing the lengths of the resulting cloud-stored ciphertexts, the attacker backs out confidential information. We investigate this injection threat model in the context of state-of-the-art encrypted messaging applications that support E2E encrypted backups. We show proof-of-concept attacks that can recover information about E2E encrypted messages or attachments sent via WhatsApp, assuming the ability to compromise the target user's Google or Apple account (which gives access to encrypted backups). We also show weaknesses in Signal's encrypted backup design that would allow injection attacks to infer metadata including a target user's number of contacts and conversations, should the adversary somehow obtain access to the user's encrypted Signal backup. While we do not believe our results should be of immediate concern for users of these messaging applications, our results do suggest that more work is needed to build tools that enjoy strong E2E security guarantees.

摘要: 我们探索了一种用于端到端(E2E)加密应用程序的新兴威胁模型：敌手将选定的消息发送到目标客户端，从而将敌意内容“注入”到应用程序状态。这样的状态随后被加密并同步到敌对的可视存储。通过观察云存储密文的长度，攻击者可以撤销机密信息。我们在支持E2E加密备份的最先进的加密消息传递应用程序的背景下研究此注入威胁模型。我们展示了概念验证攻击，可以恢复通过WhatsApp发送的E2E加密消息或附件的信息，假设有能力危害目标用户的Google或Apple帐户(这允许访问加密备份)。我们还展示了Signal加密备份设计中的弱点，该设计允许注入攻击推断包括目标用户的联系人和对话数量在内的元数据，如果对手以某种方式获得了对用户加密信号备份的访问权限。虽然我们不认为我们的结果应该立即引起这些消息传递应用程序用户的关注，但我们的结果确实表明，需要进行更多工作来构建享有强大的E2E安全保证的工具。



## **6. Transferable Adversarial Attacks against ASR**

针对ASC的可转移对抗攻击 eess.AS

IEEE SPL

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2411.09220v1) [paper-pdf](http://arxiv.org/pdf/2411.09220v1)

**Authors**: Xiaoxue Gao, Zexin Li, Yiming Chen, Cong Liu, Haizhou Li

**Abstract**: Given the extensive research and real-world applications of automatic speech recognition (ASR), ensuring the robustness of ASR models against minor input perturbations becomes a crucial consideration for maintaining their effectiveness in real-time scenarios. Previous explorations into ASR model robustness have predominantly revolved around evaluating accuracy on white-box settings with full access to ASR models. Nevertheless, full ASR model details are often not available in real-world applications. Therefore, evaluating the robustness of black-box ASR models is essential for a comprehensive understanding of ASR model resilience. In this regard, we thoroughly study the vulnerability of practical black-box attacks in cutting-edge ASR models and propose to employ two advanced time-domain-based transferable attacks alongside our differentiable feature extractor. We also propose a speech-aware gradient optimization approach (SAGO) for ASR, which forces mistranscription with minimal impact on human imperceptibility through voice activity detection rule and a speech-aware gradient-oriented optimizer. Our comprehensive experimental results reveal performance enhancements compared to baseline approaches across five models on two databases.

摘要: 鉴于自动语音识别(ASR)的广泛研究和实际应用，确保ASR模型对微小输入扰动的稳健性成为在实时场景中保持其有效性的关键考虑因素。以前对ASR模型稳健性的探索主要围绕评估完全访问ASR模型的白盒设置的准确性。然而，完整的ASR模型细节在实际应用中往往是不可用的。因此，评估黑箱ASR模型的稳健性对于全面理解ASR模型的弹性是至关重要的。在这方面，我们深入研究了实用的黑盒攻击在前沿ASR模型中的脆弱性，并提出了在可微特征提取的同时使用两种先进的基于时域的可转移攻击。我们还提出了一种语音感知梯度优化方法(SAGO)，该方法通过语音活动检测规则和语音感知梯度优化器，在对人类不可察觉影响最小的情况下强制误译。我们的综合实验结果显示，与基准方法相比，我们在两个数据库上的五个模型上的性能有所增强。



## **7. Infighting in the Dark: Multi-Labels Backdoor Attack in Federated Learning**

黑暗中的内讧：联邦学习中的多标签后门攻击 cs.CR

11 pages, 7 figures

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2409.19601v2) [paper-pdf](http://arxiv.org/pdf/2409.19601v2)

**Authors**: Ye Li, Yanchao Zhao, Chengcheng Zhu, Jiale Zhang

**Abstract**: Federated Learning (FL), a privacy-preserving decentralized machine learning framework, has been shown to be vulnerable to backdoor attacks. Current research primarily focuses on the Single-Label Backdoor Attack (SBA), wherein adversaries share a consistent target. However, a critical fact is overlooked: adversaries may be non-cooperative, have distinct targets, and operate independently, which exhibits a more practical scenario called Multi-Label Backdoor Attack (MBA). Unfortunately, prior works are ineffective in MBA scenario since non-cooperative attackers exclude each other. In this work, we conduct an in-depth investigation to uncover the inherent constraints of the exclusion: similar backdoor mappings are constructed for different targets, resulting in conflicts among backdoor functions. To address this limitation, we propose Mirage, the first non-cooperative MBA strategy in FL that allows attackers to inject effective and persistent backdoors into the global model without collusion by constructing in-distribution (ID) backdoor mapping. Specifically, we introduce an adversarial adaptation method to bridge the backdoor features and the target distribution in an ID manner. Additionally, we further leverage a constrained optimization method to ensure the ID mapping survives in the global training dynamics. Extensive evaluations demonstrate that Mirage outperforms various state-of-the-art attacks and bypasses existing defenses, achieving an average ASR greater than 97\% and maintaining over 90\% after 900 rounds. This work aims to alert researchers to this potential threat and inspire the design of effective defense mechanisms. Code has been made open-source.

摘要: 联邦学习(FL)是一种保护隐私的去中心化机器学习框架，已被证明容易受到后门攻击。目前的研究主要集中在单标签后门攻击(SBA)上，即对手共享一致的目标。然而，一个关键的事实被忽略了：对手可能是不合作的，有不同的目标，并且独立操作，这展示了一种更实际的场景，称为多标签后门攻击(MBA)。不幸的是，以前的工作在MBA场景中是无效的，因为非合作攻击者互相排斥。在这项工作中，我们进行了深入的调查，以揭示排除的内在限制：为不同的目标构造类似的后门映射，导致后门函数之间的冲突。为了解决这一局限性，我们提出了第一个非合作式MBA策略Mirage，该策略允许攻击者通过构建分布内(ID)后门映射，在全局模型中注入有效和持久的后门，而不需要合谋。具体地说，我们引入了一种对抗性自适应方法，以ID的方式将后门特征和目标分布联系起来。此外，我们进一步利用约束优化方法来确保ID映射在全局训练动态中幸存下来。广泛的评估表明，幻影的攻击性能优于各种最先进的攻击，绕过了现有的防御，平均ASR大于97%，900轮后仍保持在90%以上。这项工作旨在提醒研究人员注意这一潜在威胁，并启发设计有效的防御机制。代码已经开源。



## **8. LeapFrog: The Rowhammer Instruction Skip Attack**

LeapFrog：Rowhammer指令跳过攻击 cs.CR

Accepted at Hardware.io 2024

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2404.07878v2) [paper-pdf](http://arxiv.org/pdf/2404.07878v2)

**Authors**: Andrew Adiletta, M. Caner Tol, Kemal Derya, Berk Sunar, Saad Islam

**Abstract**: Since its inception, Rowhammer exploits have rapidly evolved into increasingly sophisticated threats compromising data integrity and the control flow integrity of victim processes. Nevertheless, it remains a challenge for an attacker to identify vulnerable targets (i.e., Rowhammer gadgets), understand the outcome of the attempted fault, and formulate an attack that yields useful results.   In this paper, we present a new type of Rowhammer gadget, called a LeapFrog gadget, which, when present in the victim code, allows an adversary to subvert code execution to bypass a critical piece of code (e.g., authentication check logic, encryption rounds, padding in security protocols). The LeapFrog gadget manifests when the victim code stores the Program Counter (PC) value in the user or kernel stack (e.g., a return address during a function call) which, when tampered with, repositions the return address to a location that bypasses a security-critical code pattern.   This research also presents a systematic process to identify LeapFrog gadgets. This methodology enables the automated detection of susceptible targets and the determination of optimal attack parameters. We first show the attack on a decision tree algorithm to show the potential implications. Secondly, we employ the attack on OpenSSL to bypass the encryption and reveal the plaintext. We then use our tools to scan the Open Quantum Safe library and report on the number of LeapFrog gadgets in the code. Lastly, we demonstrate this new attack vector through a practical demonstration in a client/server TLS handshake scenario, successfully inducing an instruction skip in a client application. Our findings extend the impact of Rowhammer attacks on control flow and contribute to developing more robust defenses against these increasingly sophisticated threats.

摘要: 自成立以来，Rowhammer漏洞攻击已迅速演变为日益复杂的威胁，危及受害者进程的数据完整性和控制流完整性。然而，对于攻击者来说，识别易受攻击的目标(即Rowhammer小工具)、了解尝试的故障的结果并制定产生有用结果的攻击仍然是一项挑战。在本文中，我们提出了一种新的Rowhammer小工具，称为LeapFrog小工具，当它存在于受害者代码中时，允许攻击者破坏代码执行以绕过关键代码段(例如，身份验证逻辑、加密轮、安全协议中的填充)。当受害者代码在用户或内核堆栈中存储程序计数器(PC)值(例如，函数调用期间的返回地址)时，LeapFrog小工具就会显现出来，当被篡改时，会将返回地址重新定位到绕过安全关键代码模式的位置。这项研究还提供了识别LeapFrog小工具的系统流程。这种方法能够自动检测易受影响的目标并确定最佳攻击参数。我们首先展示了对决策树算法的攻击，以显示其潜在的含义。其次，利用对OpenSSL的攻击绕过加密，泄露明文。然后，我们使用我们的工具扫描Open Quantum Safe库，并报告代码中LeapFrog小工具的数量。最后，我们通过在客户/服务器TLS握手场景中的实际演示，成功地在客户端应用程序中诱导了指令跳过，从而演示了这种新的攻击矢量。我们的发现扩大了Rowhammer攻击对控制流的影响，并有助于开发针对这些日益复杂的威胁的更强大的防御措施。



## **9. DROJ: A Prompt-Driven Attack against Large Language Models**

DROJ：针对大型语言模型的预算驱动攻击 cs.CL

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2411.09125v1) [paper-pdf](http://arxiv.org/pdf/2411.09125v1)

**Authors**: Leyang Hu, Boran Wang

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional capabilities across various natural language processing tasks. Due to their training on internet-sourced datasets, LLMs can sometimes generate objectionable content, necessitating extensive alignment with human feedback to avoid such outputs. Despite massive alignment efforts, LLMs remain susceptible to adversarial jailbreak attacks, which usually are manipulated prompts designed to circumvent safety mechanisms and elicit harmful responses. Here, we introduce a novel approach, Directed Rrepresentation Optimization Jailbreak (DROJ), which optimizes jailbreak prompts at the embedding level to shift the hidden representations of harmful queries towards directions that are more likely to elicit affirmative responses from the model. Our evaluations on LLaMA-2-7b-chat model show that DROJ achieves a 100\% keyword-based Attack Success Rate (ASR), effectively preventing direct refusals. However, the model occasionally produces repetitive and non-informative responses. To mitigate this, we introduce a helpfulness system prompt that enhances the utility of the model's responses. Our code is available at https://github.com/Leon-Leyang/LLM-Safeguard.

摘要: 大型语言模型(LLM)在各种自然语言处理任务中表现出了非凡的能力。由于对来自互联网的数据集进行了培训，LLM有时会产生令人反感的内容，需要与人类反馈广泛协调，以避免此类输出。尽管做出了巨大的调整努力，但LLM仍然容易受到对抗性越狱攻击，这些攻击通常是被操纵的提示，旨在绕过安全机制并引发有害反应。在这里，我们介绍了一种新的方法，定向R表示优化越狱(DROJ)，它在嵌入级别优化越狱提示，将有害查询的隐藏表示向更有可能引起模型肯定响应的方向移动。对Llama-2-7b-Chat模型的评估表明，DROJ达到了100%的基于关键字的攻击成功率，有效地防止了直接拒绝。然而，该模型偶尔会产生重复的、非信息性的回答。为了缓解这一问题，我们引入了一个帮助系统提示，以增强模型响应的实用性。我们的代码可以在https://github.com/Leon-Leyang/LLM-Safeguard.上找到



## **10. Deciphering the Definition of Adversarial Robustness for post-hoc OOD Detectors**

破译事后OOD检测器对抗鲁棒性的定义 cs.CR

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2406.15104v4) [paper-pdf](http://arxiv.org/pdf/2406.15104v4)

**Authors**: Peter Lorenz, Mario Fernandez, Jens Müller, Ullrich Köthe

**Abstract**: Detecting out-of-distribution (OOD) inputs is critical for safely deploying deep learning models in real-world scenarios. In recent years, many OOD detectors have been developed, and even the benchmarking has been standardized, i.e. OpenOOD. The number of post-hoc detectors is growing fast. They are showing an option to protect a pre-trained classifier against natural distribution shifts and claim to be ready for real-world scenarios. However, its effectiveness in dealing with adversarial examples (AdEx) has been neglected in most studies. In cases where an OOD detector includes AdEx in its experiments, the lack of uniform parameters for AdEx makes it difficult to accurately evaluate the performance of the OOD detector. This paper investigates the adversarial robustness of 16 post-hoc detectors against various evasion attacks. It also discusses a roadmap for adversarial defense in OOD detectors that would help adversarial robustness. We believe that level 1 (AdEx on a unified dataset) should be added to any OOD detector to see the limitations. The last level in the roadmap (defense against adaptive attacks) we added for integrity from an adversarial machine learning (AML) point of view, which we do not believe is the ultimate goal for OOD detectors.

摘要: 检测分布外(OOD)输入对于在真实场景中安全部署深度学习模型至关重要。近年来，已经开发了许多面向对象的检测器，甚至已经对基准进行了标准化，即OpenOOD。后自组织探测器的数量正在快速增长。他们展示了一种选项，可以保护预先训练的分类器免受自然分布变化的影响，并声称已经为现实世界的场景做好了准备。然而，在大多数研究中，它在处理对抗性例子(ADEX)方面的有效性一直被忽视。在OOD探测器在其实验中包括ADEX的情况下，ADEX缺乏统一的参数使得很难准确地评估OOD探测器的性能。本文研究了16种后自组织检测器对各种逃避攻击的抵抗能力。它还讨论了在OOD检测器中进行对抗防御的路线图，这将有助于对抗健壮性。我们认为，应将级别1(统一数据集上的ADEX)添加到任何OOD检测器，以了解其局限性。从对抗性机器学习(AML)的角度来看，我们在路线图中添加的最后一个级别(防御自适应攻击)是为了完整性，我们不认为这是OOD检测器的最终目标。



## **11. Defending Large Language Models Against Attacks With Residual Stream Activation Analysis**

利用剩余流激活分析防御大型语言模型免受攻击 cs.CR

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2406.03230v4) [paper-pdf](http://arxiv.org/pdf/2406.03230v4)

**Authors**: Amelia Kawasaki, Andrew Davis, Houssam Abbas

**Abstract**: The widespread adoption of Large Language Models (LLMs), exemplified by OpenAI's ChatGPT, brings to the forefront the imperative to defend against adversarial threats on these models. These attacks, which manipulate an LLM's output by introducing malicious inputs, undermine the model's integrity and the trust users place in its outputs. In response to this challenge, our paper presents an innovative defensive strategy, given white box access to an LLM, that harnesses residual activation analysis between transformer layers of the LLM. We apply a novel methodology for analyzing distinctive activation patterns in the residual streams for attack prompt classification. We curate multiple datasets to demonstrate how this method of classification has high accuracy across multiple types of attack scenarios, including our newly-created attack dataset. Furthermore, we enhance the model's resilience by integrating safety fine-tuning techniques for LLMs in order to measure its effect on our capability to detect attacks. The results underscore the effectiveness of our approach in enhancing the detection and mitigation of adversarial inputs, advancing the security framework within which LLMs operate.

摘要: 大型语言模型(LLM)的广泛采用，如OpenAI的ChatGPT，使防御这些模型上的对手威胁成为当务之急。这些攻击通过引入恶意输入来操纵LLM的输出，破坏了模型的完整性和用户对其输出的信任。为了应对这一挑战，我们的论文提出了一种创新的防御策略，在白盒访问LLM的情况下，该策略利用LLM变压器层之间的剩余激活分析。我们应用了一种新的方法来分析残留流中独特的激活模式，以进行攻击提示分类。我们精选了多个数据集，以演示此分类方法如何在多种类型的攻击场景中具有高精度，包括我们新创建的攻击数据集。此外，我们通过集成LLMS的安全微调技术来增强模型的弹性，以衡量其对我们检测攻击的能力的影响。这些结果强调了我们的方法在加强对敌对输入的检测和缓解、推进LLMS运作的安全框架方面的有效性。



## **12. LLMStinger: Jailbreaking LLMs using RL fine-tuned LLMs**

LLMStinger：使用RL微调的LLM越狱LLM cs.LG

Accepted at AAAI 2025

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2411.08862v1) [paper-pdf](http://arxiv.org/pdf/2411.08862v1)

**Authors**: Piyush Jha, Arnav Arora, Vijay Ganesh

**Abstract**: We introduce LLMStinger, a novel approach that leverages Large Language Models (LLMs) to automatically generate adversarial suffixes for jailbreak attacks. Unlike traditional methods, which require complex prompt engineering or white-box access, LLMStinger uses a reinforcement learning (RL) loop to fine-tune an attacker LLM, generating new suffixes based on existing attacks for harmful questions from the HarmBench benchmark. Our method significantly outperforms existing red-teaming approaches (we compared against 15 of the latest methods), achieving a +57.2% improvement in Attack Success Rate (ASR) on LLaMA2-7B-chat and a +50.3% ASR increase on Claude 2, both models known for their extensive safety measures. Additionally, we achieved a 94.97% ASR on GPT-3.5 and 99.4% on Gemma-2B-it, demonstrating the robustness and adaptability of LLMStinger across open and closed-source models.

摘要: 我们引入了LLMStinger，这是一种利用大型语言模型（LLM）自动生成越狱攻击的对抗性后缀的新颖方法。与需要复杂的即时工程或白盒访问的传统方法不同，LLMStinger使用强化学习（RL）循环来微调攻击者LLM，根据HarmBench基准中针对有害问题的现有攻击生成新的后缀。我们的方法显着优于现有的红色团队方法（我们与15种最新方法进行了比较），在LLaMA 2 - 7 B-chat上实现了攻击成功率（ASB）+57.2%的提高，在Claude 2上实现了攻击成功率（ASB）+50.3%的提高，这两种型号都以其广泛的安全措施而闻名。此外，我们在GPT-3.5上实现了94.97%的ASB，在Gemma-2B-it上实现了99.4%的ASB，证明了LLMStinger在开放和封闭源模型中的稳健性和适应性。



## **13. On the Robustness of Neural Collapse and the Neural Collapse of Robustness**

关于神经崩溃的鲁棒性和鲁棒性的神经崩溃 cs.LG

Transactions on Machine Learning Research, 2024

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2311.07444v2) [paper-pdf](http://arxiv.org/pdf/2311.07444v2)

**Authors**: Jingtong Su, Ya Shi Zhang, Nikolaos Tsilivis, Julia Kempe

**Abstract**: Neural Collapse refers to the curious phenomenon in the end of training of a neural network, where feature vectors and classification weights converge to a very simple geometrical arrangement (a simplex). While it has been observed empirically in various cases and has been theoretically motivated, its connection with crucial properties of neural networks, like their generalization and robustness, remains unclear. In this work, we study the stability properties of these simplices. We find that the simplex structure disappears under small adversarial attacks, and that perturbed examples "leap" between simplex vertices. We further analyze the geometry of networks that are optimized to be robust against adversarial perturbations of the input, and find that Neural Collapse is a pervasive phenomenon in these cases as well, with clean and perturbed representations forming aligned simplices, and giving rise to a robust simple nearest-neighbor classifier. By studying the propagation of the amount of collapse inside the network, we identify novel properties of both robust and non-robust machine learning models, and show that earlier, unlike later layers maintain reliable simplices on perturbed data. Our code is available at https://github.com/JingtongSu/robust_neural_collapse .

摘要: 神经崩溃是指在神经网络的训练结束时，特征向量和分类权重收敛到一个非常简单的几何排列(单纯形)的奇怪现象。虽然它已经在各种情况下得到了经验的观察，并在理论上得到了推动，但它与神经网络的关键特性，如它们的泛化和健壮性的联系，仍然不清楚。在这项工作中，我们研究了这些单形的稳定性。我们发现，单纯形结构在小的对抗性攻击下消失，扰动的例子在单纯形顶点之间跳跃。我们进一步分析了优化后的网络的几何结构，发现在这些情况下，神经崩溃也是一种普遍现象，干净和扰动的表示形成了对齐的简化，并产生了一个健壮的简单最近邻分类器。通过研究崩溃量在网络中的传播，我们识别了健壮和非健壮机器学习模型的新性质，并表明早期不同于后面的层在扰动数据上保持可靠的简化。我们的代码可以在https://github.com/JingtongSu/robust_neural_collapse上找到。



## **14. Robust Optimal Power Flow Against Adversarial Attacks: A Tri-Level Optimization Approach**

对抗攻击的鲁棒最优潮流：三层优化方法 eess.SY

This work has been submitted for possible publication

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2411.08618v1) [paper-pdf](http://arxiv.org/pdf/2411.08618v1)

**Authors**: Saman Mazaheri Khamaneh, Tong Wu

**Abstract**: In power systems, unpredictable events like extreme weather, equipment failures, and cyberattacks present significant challenges to ensuring safety and reliability. Ensuring resilience in the face of these uncertainties is crucial for reliable and efficient operations. This paper presents a tri-level optimization approach for robust power system operations that effectively address worst-case attacks. The first stage focuses on optimizing economic dispatch under normal operating conditions, aiming to minimize generation costs while maintaining the supply-demand balance. The second stage introduces an adversarial attack model, identifying worst-case scenarios that maximize the system's vulnerability by targeting distributed generation (DG). In the third stage, mitigation strategies are developed using fast-response energy storage systems (ESS) to minimize disruptions caused by these attacks. By integrating economic dispatch, vulnerability assessment, and mitigation into a unified framework, this approach provides a robust solution for enhancing power system resilience and safety against evolving adversarial threats. The approach is validated using the IEEE-33 node distribution system to demonstrate its effectiveness in achieving both cost efficiency and system resilience.

摘要: 在电力系统中，极端天气、设备故障和网络攻击等不可预测的事件给确保安全和可靠性带来了巨大的挑战。确保在面对这些不确定性时保持韧性，对于可靠和高效的运营至关重要。提出了一种能有效应对最坏情况下攻击的稳健电力系统运行的三层优化方法。第一阶段侧重于在正常运行条件下优化经济调度，目标是在保持供需平衡的同时将发电成本降至最低。第二阶段引入对抗性攻击模型，通过以分布式发电(DG)为目标来识别最坏情况，从而最大化系统的脆弱性。在第三阶段，使用快速响应能量存储系统(ESS)制定缓解策略，以最大限度地减少这些攻击造成的破坏。通过将经济调度、脆弱性评估和缓解集成到一个统一的框架中，该方法为增强电力系统对不断变化的对手威胁的弹性和安全性提供了稳健的解决方案。使用IEEE-33节点分布系统对该方法进行了验证，证明了该方法在实现成本效率和系统弹性方面的有效性。



## **15. Target-driven Attack for Large Language Models**

针对大型语言模型的目标驱动攻击 cs.CL

12 pages, 7 figures. This work is an extension of the  arXiv:2404.07234 work. We propose new methods. 27th European Conference on  Artificial Intelligence 2024

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2411.07268v2) [paper-pdf](http://arxiv.org/pdf/2411.07268v2)

**Authors**: Chong Zhang, Mingyu Jin, Dong Shu, Taowen Wang, Dongfang Liu, Xiaobo Jin

**Abstract**: Current large language models (LLM) provide a strong foundation for large-scale user-oriented natural language tasks. Many users can easily inject adversarial text or instructions through the user interface, thus causing LLM model security challenges like the language model not giving the correct answer. Although there is currently a large amount of research on black-box attacks, most of these black-box attacks use random and heuristic strategies. It is unclear how these strategies relate to the success rate of attacks and thus effectively improve model robustness. To solve this problem, we propose our target-driven black-box attack method to maximize the KL divergence between the conditional probabilities of the clean text and the attack text to redefine the attack's goal. We transform the distance maximization problem into two convex optimization problems based on the attack goal to solve the attack text and estimate the covariance. Furthermore, the projected gradient descent algorithm solves the vector corresponding to the attack text. Our target-driven black-box attack approach includes two attack strategies: token manipulation and misinformation attack. Experimental results on multiple Large Language Models and datasets demonstrate the effectiveness of our attack method.

摘要: 现有的大型语言模型(LLM)为大规模面向用户的自然语言任务提供了坚实的基础。许多用户可以很容易地通过用户界面注入敌意文本或指令，从而导致LLM模型的安全挑战，如语言模型无法给出正确的答案。虽然目前有大量关于黑盒攻击的研究，但这些黑盒攻击大多采用随机和启发式策略。目前尚不清楚这些策略如何与攻击成功率相关，从而有效地提高模型的健壮性。为了解决这一问题，我们提出了目标驱动的黑盒攻击方法，以最大化明文和攻击文本的条件概率之间的KL偏差，从而重新定义攻击的目标。将距离最大化问题转化为基于攻击目标的两个凸优化问题来求解攻击文本并估计协方差。此外，投影梯度下降算法求解与攻击文本对应的向量。我们的目标驱动的黑盒攻击方法包括两种攻击策略：令牌操纵和错误信息攻击。在多个大型语言模型和数据集上的实验结果证明了该攻击方法的有效性。



## **16. Confidence-aware Denoised Fine-tuning of Off-the-shelf Models for Certified Robustness**

对现成模型进行信任意识的去噪微调，以获得认证的鲁棒性 cs.CV

26 pages; TMLR 2024; Code is available at  https://github.com/suhyeok24/FT-CADIS

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.08933v2) [paper-pdf](http://arxiv.org/pdf/2411.08933v2)

**Authors**: Suhyeok Jang, Seojin Kim, Jinwoo Shin, Jongheon Jeong

**Abstract**: The remarkable advances in deep learning have led to the emergence of many off-the-shelf classifiers, e.g., large pre-trained models. However, since they are typically trained on clean data, they remain vulnerable to adversarial attacks. Despite this vulnerability, their superior performance and transferability make off-the-shelf classifiers still valuable in practice, demanding further work to provide adversarial robustness for them in a post-hoc manner. A recently proposed method, denoised smoothing, leverages a denoiser model in front of the classifier to obtain provable robustness without additional training. However, the denoiser often creates hallucination, i.e., images that have lost the semantics of their originally assigned class, leading to a drop in robustness. Furthermore, its noise-and-denoise procedure introduces a significant distribution shift from the original distribution, causing the denoised smoothing framework to achieve sub-optimal robustness. In this paper, we introduce Fine-Tuning with Confidence-Aware Denoised Image Selection (FT-CADIS), a novel fine-tuning scheme to enhance the certified robustness of off-the-shelf classifiers. FT-CADIS is inspired by the observation that the confidence of off-the-shelf classifiers can effectively identify hallucinated images during denoised smoothing. Based on this, we develop a confidence-aware training objective to handle such hallucinated images and improve the stability of fine-tuning from denoised images. In this way, the classifier can be fine-tuned using only images that are beneficial for adversarial robustness. We also find that such a fine-tuning can be done by updating a small fraction of parameters of the classifier. Extensive experiments demonstrate that FT-CADIS has established the state-of-the-art certified robustness among denoised smoothing methods across all $\ell_2$-adversary radius in various benchmarks.

摘要: 深度学习的显著进展导致了许多现成的分类器的出现，例如大型预先训练的模型。然而，由于他们通常接受的是干净数据的培训，他们仍然容易受到对手的攻击。尽管存在这个漏洞，但它们优越的性能和可转移性使得现成的分类器在实践中仍然有价值，需要进一步的工作来以后自组织的方式为它们提供对抗性的健壮性。最近提出的去噪平滑方法利用分类器前面的去噪模型来获得可证明的稳健性，而不需要额外的训练。然而，去噪通常会造成幻觉，即图像失去了最初分配的类的语义，导致健壮性下降。此外，其去噪和去噪过程引入了与原始分布显著的分布偏移，导致去噪平滑框架实现次优稳健性。在本文中，我们介绍了一种新的精调方案--基于置信度的去噪图像选择精调算法(FT-CADIS)，以增强现有分类器的稳健性。FT-CADIS的灵感来自于观察到，在去噪平滑过程中，现成分类器的信心可以有效地识别幻觉图像。在此基础上，我们开发了一种置信度感知训练目标来处理这类幻觉图像，并提高了对去噪图像进行微调的稳定性。通过这种方式，可以仅使用有利于对抗健壮性的图像来微调分类器。我们还发现，这样的微调可以通过更新分类器的一小部分参数来完成。大量的实验表明，FT-CADIS在各种基准下，在所有对手半径的去噪平滑方法中建立了最先进的经验证的稳健性。



## **17. A Fully Local Last-Generated Rule in a Blockchain**

区块链中的完全本地最后生成规则 cs.CR

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2411.08439v1) [paper-pdf](http://arxiv.org/pdf/2411.08439v1)

**Authors**: Akira Sakurai, Kazuyuki Shudo

**Abstract**: An effective method for suppressing intentional forks in a blockchain is the last-generated rule, which selects the most recent chain as the main chain in the event of a chain tie. This rule helps invalidate blocks that are withheld by adversaries for a certain period. However, existing last-generated rules face an issue in that their applications to the system are not fully localized. In conservative cryptocurrency systems such as Bitcoin, it is desirable for methods to be applied in a fully local manner. In this paper, we propose a locally applicable last-generated rule. Our method is straightforward and is based on a relative time reference. By conservatively setting the upper bound for the clock skews $\Delta_{O_i}$ to 200 s, our proposed method reduces the proportion $\gamma$ of honest miners following the attacker during chain ties by more than 40% compared to existing local methods.

摘要: 抑制区块链中故意分叉的一种有效方法是最后生成的规则，该规则在发生连锁关系的情况下选择最近的链作为主线。此规则有助于使对手在一定时期内扣留的区块无效。然而，现有的最后生成的规则面临着一个问题，因为它们对系统的应用程序没有完全本地化。在比特币等保守的加密货币系统中，希望以完全本地化的方式应用方法。在本文中，我们提出了一个本地适用的最后生成规则。我们的方法很简单，并且基于相对时间参考。与现有的本地方法相比，通过保守地将时钟偏差$\Delta_{O_i}$的上限设置为200秒，我们提出的方法将连锁关系期间跟踪攻击者的诚实矿工的比例$\gamma$减少了40%以上。



## **18. ADI: Adversarial Dominating Inputs in Vertical Federated Learning Systems**

ADI：垂直联邦学习系统中的对抗主导输入 cs.CR

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2201.02775v4) [paper-pdf](http://arxiv.org/pdf/2201.02775v4)

**Authors**: Qi Pang, Yuanyuan Yuan, Shuai Wang, Wenting Zheng

**Abstract**: Vertical federated learning (VFL) system has recently become prominent as a concept to process data distributed across many individual sources without the need to centralize it. Multiple participants collaboratively train models based on their local data in a privacy-aware manner. To date, VFL has become a de facto solution to securely learn a model among organizations, allowing knowledge to be shared without compromising privacy of any individuals. Despite the prosperous development of VFL systems, we find that certain inputs of a participant, named adversarial dominating inputs (ADIs), can dominate the joint inference towards the direction of the adversary's will and force other (victim) participants to make negligible contributions, losing rewards that are usually offered regarding the importance of their contributions in federated learning scenarios. We conduct a systematic study on ADIs by first proving their existence in typical VFL systems. We then propose gradient-based methods to synthesize ADIs of various formats and exploit common VFL systems. We further launch greybox fuzz testing, guided by the saliency score of ``victim'' participants, to perturb adversary-controlled inputs and systematically explore the VFL attack surface in a privacy-preserving manner. We conduct an in-depth study on the influence of critical parameters and settings in synthesizing ADIs. Our study reveals new VFL attack opportunities, promoting the identification of unknown threats before breaches and building more secure VFL systems.

摘要: 垂直联合学习(VFL)系统最近作为一个概念变得突出起来，它可以处理分布在多个独立来源的数据，而不需要集中这些数据。多个参与者以隐私感知的方式基于他们的本地数据协作训练模型。到目前为止，VFL已经成为在组织之间安全地学习模式的事实上的解决方案，允许在不损害任何个人隐私的情况下共享知识。尽管VFL系统的蓬勃发展，我们发现参与者的某些输入，称为对抗性主导输入(ADI)，可以主导朝着对手意愿方向的联合推理，并迫使其他(受害者)参与者做出可以忽略不计的贡献，失去通常提供的关于他们在联合学习场景中贡献的重要性的奖励。我们首先通过证明ADI在典型的VFL系统中的存在来对ADI进行系统的研究。然后，我们提出了基于梯度的方法来合成各种格式的ADI，并开发了常见的VFL系统。我们进一步推出灰盒模糊测试，以“受害者”参与者的显著分数为指导，扰乱对手控制的输入，并以保护隐私的方式系统地探索VFL攻击面。我们深入研究了关键参数和设置对ADI合成的影响。我们的研究揭示了新的VFL攻击机会，促进了在入侵之前识别未知威胁，并建立了更安全的VFL系统。



## **19. "No Matter What You Do": Purifying GNN Models via Backdoor Unlearning**

“无论你做什么”：通过后门取消学习来净化GNN模型 cs.CR

18 pages, 12 figures, 9 tables

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2410.01272v2) [paper-pdf](http://arxiv.org/pdf/2410.01272v2)

**Authors**: Jiale Zhang, Chengcheng Zhu, Bosen Rao, Hao Sui, Xiaobing Sun, Bing Chen, Chunyi Zhou, Shouling Ji

**Abstract**: Recent studies have exposed that GNNs are vulnerable to several adversarial attacks, among which backdoor attack is one of the toughest. Similar to Deep Neural Networks (DNNs), backdoor attacks in GNNs lie in the fact that the attacker modifies a portion of graph data by embedding triggers and enforces the model to learn the trigger feature during the model training process. Despite the massive prior backdoor defense works on DNNs, defending against backdoor attacks in GNNs is largely unexplored, severely hindering the widespread application of GNNs in real-world tasks. To bridge this gap, we present GCleaner, the first backdoor mitigation method on GNNs. GCleaner can mitigate the presence of the backdoor logic within backdoored GNNs by reversing the backdoor learning procedure, aiming to restore the model performance to a level similar to that is directly trained on the original clean dataset. To achieve this objective, we ask: How to recover universal and hard backdoor triggers in GNNs? How to unlearn the backdoor trigger feature while maintaining the model performance? We conduct the graph trigger recovery via the explanation method to identify optimal trigger locations, facilitating the search of universal and hard backdoor triggers in the feature space of the backdoored model through maximal similarity. Subsequently, we introduce the backdoor unlearning mechanism, which combines knowledge distillation and gradient-based explainable knowledge for fine-grained backdoor erasure. Extensive experimental evaluations on four benchmark datasets demonstrate that GCleaner can reduce the backdoor attack success rate to 10% with only 1% of clean data, and has almost negligible degradation in model performance, which far outperforms the state-of-the-art (SOTA) defense methods.

摘要: 最近的研究表明，GNN容易受到几种敌意攻击，其中后门攻击是最难对付的攻击之一。与深度神经网络(DNN)类似，GNN中的后门攻击在于攻击者通过嵌入触发器来修改一部分图数据，并在模型训练过程中强制模型学习触发器特征。尽管先前在DNN上进行了大量的后门防御工作，但针对GNN中的后门攻击的防御在很大程度上是未被探索的，严重阻碍了GNN在现实世界任务中的广泛应用。为了弥补这一差距，我们提出了GCleaner，这是GNNS上的第一个后门缓解方法。GCleaner可以通过颠倒后门学习过程来缓解后门逻辑在后门GNN中的存在，旨在将模型性能恢复到与在原始CLEAN数据集上直接训练的水平类似的水平。为了实现这一目标，我们问：如何在GNN中恢复通用和硬后门触发器？如何在保持模型性能的同时忘记后门触发功能？通过解释方法进行图触发器恢复，确定最优触发器位置，通过最大相似度在后退模型的特征空间中搜索通用的和硬的后门触发器。随后，我们引入了后门遗忘机制，该机制将知识提炼和基于梯度的可解释知识相结合，用于细粒度的后门删除。在四个基准数据集上的广泛实验评估表明，GCleaner可以在仅使用1%的干净数据的情况下将后门攻击成功率降低到10%，并且模型性能的下降几乎可以忽略不计，远远超过最先进的(SOTA)防御方法。



## **20. Deceiving Question-Answering Models: A Hybrid Word-Level Adversarial Approach**

欺骗网络攻击模型：混合词级对抗方法 cs.CL

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.08248v1) [paper-pdf](http://arxiv.org/pdf/2411.08248v1)

**Authors**: Jiyao Li, Mingze Ni, Yongshun Gong, Wei Liu

**Abstract**: Deep learning underpins most of the currently advanced natural language processing (NLP) tasks such as textual classification, neural machine translation (NMT), abstractive summarization and question-answering (QA). However, the robustness of the models, particularly QA models, against adversarial attacks is a critical concern that remains insufficiently explored. This paper introduces QA-Attack (Question Answering Attack), a novel word-level adversarial strategy that fools QA models. Our attention-based attack exploits the customized attention mechanism and deletion ranking strategy to identify and target specific words within contextual passages. It creates deceptive inputs by carefully choosing and substituting synonyms, preserving grammatical integrity while misleading the model to produce incorrect responses. Our approach demonstrates versatility across various question types, particularly when dealing with extensive long textual inputs. Extensive experiments on multiple benchmark datasets demonstrate that QA-Attack successfully deceives baseline QA models and surpasses existing adversarial techniques regarding success rate, semantics changes, BLEU score, fluency and grammar error rate.

摘要: 深度学习是当前大多数高级自然语言处理(NLP)任务的基础，例如文本分类、神经机器翻译(NMT)、抽象摘要和问答(QA)。然而，模型的稳健性，特别是QA模型，对对手攻击的稳健性是一个严重的问题，仍然没有得到充分的研究。本文介绍了一种新的愚弄问答模型的词级对抗性策略--问答攻击。我们的基于注意力的攻击利用定制的注意力机制和删除排名策略来识别和定位上下文段落中的特定单词。它通过仔细选择和替换同义词来创造欺骗性输入，在保持语法完整性的同时误导模型产生不正确的回答。我们的方法展示了跨各种问题类型的多功能性，特别是在处理大量长文本输入时。在多个基准数据集上的大量实验表明，问答攻击成功地欺骗了基线问答模型，并在成功率、语义变化、BLEU评分、流利度和语法错误率方面超过了现有的对抗性技术。



## **21. Adaptive Meta-Learning for Robust Deepfake Detection: A Multi-Agent Framework to Data Drift and Model Generalization**

用于稳健Deepfake检测的自适应元学习：数据漂移和模型概括的多代理框架 cs.AI

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.08148v1) [paper-pdf](http://arxiv.org/pdf/2411.08148v1)

**Authors**: Dinesh Srivasthav P, Badri Narayan Subudhi

**Abstract**: Pioneering advancements in artificial intelligence, especially in genAI, have enabled significant possibilities for content creation, but also led to widespread misinformation and false content. The growing sophistication and realism of deepfakes is raising concerns about privacy invasion, identity theft, and has societal, business impacts, including reputational damage and financial loss. Many deepfake detectors have been developed to tackle this problem. Nevertheless, as for every AI model, the deepfake detectors face the wrath of lack of considerable generalization to unseen scenarios and cross-domain deepfakes. Besides, adversarial robustness is another critical challenge, as detectors drastically underperform to the slightest imperceptible change. Most state-of-the-art detectors are trained on static datasets and lack the ability to adapt to emerging deepfake attack trends. These three crucial challenges though hold paramount importance for reliability in practise, particularly in the deepfake domain, are also the problems with any other AI application. This paper proposes an adversarial meta-learning algorithm using task-specific adaptive sample synthesis and consistency regularization, in a refinement phase. By focussing on the classifier's strengths and weaknesses, it boosts both robustness and generalization of the model. Additionally, the paper introduces a hierarchical multi-agent retrieval-augmented generation workflow with a sample synthesis module to dynamically adapt the model to new data trends by generating custom deepfake samples. The paper further presents a framework integrating the meta-learning algorithm with the hierarchical multi-agent workflow, offering a holistic solution for enhancing generalization, robustness, and adaptability. Experimental results demonstrate the model's consistent performance across various datasets, outperforming the models in comparison.

摘要: 人工智能，特别是genAI的开创性进步，为内容创作带来了巨大的可能性，但也导致了广泛的错误信息和虚假内容。深度假货的日益复杂和现实主义引发了人们对隐私侵犯、身份盗窃的担忧，并产生了社会和商业影响，包括声誉损害和经济损失。为了解决这个问题，已经开发了许多深伪探测器。然而，对于每一个人工智能模型来说，深度假检测器都面临着对看不见的场景和跨域的深度假缺乏相当普遍的概括的愤怒。此外，对抗的稳健性是另一个关键挑战，因为检测器的性能严重落后于最轻微的不可察觉的变化。大多数最先进的检测器都是针对静态数据集进行培训的，缺乏适应新出现的深度伪攻击趋势的能力。尽管这三个关键挑战对于实践中的可靠性至关重要，特别是在深度假冒领域，但也是任何其他人工智能应用程序的问题。本文提出了一种基于特定任务的自适应样本合成和一致性正则化的对抗性元学习算法。通过重点分析分类器的优缺点，增强了模型的鲁棒性和泛化能力。此外，本文还介绍了一种具有样本合成模块的层次化多代理检索增强生成工作流，通过生成定制的深度伪样本来动态调整模型以适应新的数据趋势。本文进一步提出了元学习算法与层次化多智能体工作流相结合的框架，为提高泛化、健壮性和适应性提供了一个整体解决方案。实验结果表明，该模型在不同的数据集上具有一致的性能，性能优于其他模型。



## **22. Can adversarial attacks by large language models be attributed?**

大型语言模型的对抗攻击可以归因吗？ cs.AI

7 pages, 1 figure

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.08003v1) [paper-pdf](http://arxiv.org/pdf/2411.08003v1)

**Authors**: Manuel Cebrian, Jan Arne Telle

**Abstract**: Attributing outputs from Large Language Models (LLMs) in adversarial settings-such as cyberattacks and disinformation-presents significant challenges that are likely to grow in importance. We investigate this attribution problem using formal language theory, specifically language identification in the limit as introduced by Gold and extended by Angluin. By modeling LLM outputs as formal languages, we analyze whether finite text samples can uniquely pinpoint the originating model. Our results show that due to the non-identifiability of certain language classes, under some mild assumptions about overlapping outputs from fine-tuned models it is theoretically impossible to attribute outputs to specific LLMs with certainty. This holds also when accounting for expressivity limitations of Transformer architectures. Even with direct model access or comprehensive monitoring, significant computational hurdles impede attribution efforts. These findings highlight an urgent need for proactive measures to mitigate risks posed by adversarial LLM use as their influence continues to expand.

摘要: 将大型语言模型(LLM)的输出归因于敌对环境--如网络攻击和虚假信息--带来了重大挑战，而这些挑战的重要性可能会越来越大。我们使用形式化语言理论来研究这一归因问题，特别是Gold提出并由Anluin推广的极限语言识别问题。通过将LLM输出建模为形式语言，我们分析了有限文本样本是否能够唯一地定位原始模型。我们的结果表明，由于某些语言类别的不可识别性，在微调模型的输出重叠的一些温和假设下，理论上不可能确定地将输出归因于特定的LLM。当考虑到Transformer架构的表现力限制时，这也是成立的。即使有了直接的模型访问或全面的监测，重大的计算障碍也阻碍了归因努力。这些调查结果突出表明，迫切需要采取积极主动的措施，以减轻敌对使用LLM所带来的风险，因为它们的影响继续扩大。



## **23. IAE: Irony-based Adversarial Examples for Sentiment Analysis Systems**

IAE：情感分析系统的基于讽刺的对抗示例 cs.CL

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07850v1) [paper-pdf](http://arxiv.org/pdf/2411.07850v1)

**Authors**: Xiaoyin Yi, Jiacheng Huang

**Abstract**: Adversarial examples, which are inputs deliberately perturbed with imperceptible changes to induce model errors, have raised serious concerns for the reliability and security of deep neural networks (DNNs). While adversarial attacks have been extensively studied in continuous data domains such as images, the discrete nature of text presents unique challenges. In this paper, we propose Irony-based Adversarial Examples (IAE), a method that transforms straightforward sentences into ironic ones to create adversarial text. This approach exploits the rhetorical device of irony, where the intended meaning is opposite to the literal interpretation, requiring a deeper understanding of context to detect. The IAE method is particularly challenging due to the need to accurately locate evaluation words, substitute them with appropriate collocations, and expand the text with suitable ironic elements while maintaining semantic coherence. Our research makes the following key contributions: (1) We introduce IAE, a strategy for generating textual adversarial examples using irony. This method does not rely on pre-existing irony corpora, making it a versatile tool for creating adversarial text in various NLP tasks. (2) We demonstrate that the performance of several state-of-the-art deep learning models on sentiment analysis tasks significantly deteriorates when subjected to IAE attacks. This finding underscores the susceptibility of current NLP systems to adversarial manipulation through irony. (3) We compare the impact of IAE on human judgment versus NLP systems, revealing that humans are less susceptible to the effects of irony in text.

摘要: 对抗性的例子，即输入故意被不可察觉的变化扰动以引起模型错误，已经引起了对深度神经网络(DNN)的可靠性和安全性的严重关注。虽然对抗性攻击已经在图像等连续数据领域得到了广泛的研究，但文本的离散性质带来了独特的挑战。在本文中，我们提出了一种基于反讽的对抗性范例(IAE)，它是一种将直白的句子转换成反讽句子来创建对抗性文本的方法。这一方法利用了反讽的修辞手段，其意图与字面解释相反，需要对语境进行更深层次的理解才能发现。IAE方法特别具有挑战性，因为需要准确定位评价词，用适当的搭配取代它们，并在保持语义连贯的同时用合适的讽刺元素扩展文本。我们的研究取得了以下主要贡献：(1)介绍了IAE，这是一种使用反讽生成文本对抗性实例的策略。这种方法不依赖于预先存在的反讽语料库，使其成为在各种自然语言处理任务中创建敌意文本的通用工具。(2)研究表明，当情感分析任务受到IAE攻击时，几种最新的深度学习模型的性能会显著下降。这一发现突显了当前NLP系统通过反讽进行对抗性操纵的敏感性。(3)我们比较了IAE和NLP系统对人类判断的影响，发现人类不太容易受到语篇中反讽的影响。



## **24. Chain Association-based Attacking and Shielding Natural Language Processing Systems**

基于链关联的攻击和屏蔽自然语言处理系统 cs.CL

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07843v1) [paper-pdf](http://arxiv.org/pdf/2411.07843v1)

**Authors**: Jiacheng Huang, Long Chen

**Abstract**: Association as a gift enables people do not have to mention something in completely straightforward words and allows others to understand what they intend to refer to. In this paper, we propose a chain association-based adversarial attack against natural language processing systems, utilizing the comprehension gap between humans and machines. We first generate a chain association graph for Chinese characters based on the association paradigm for building search space of potential adversarial examples. Then, we introduce an discrete particle swarm optimization algorithm to search for the optimal adversarial examples. We conduct comprehensive experiments and show that advanced natural language processing models and applications, including large language models, are vulnerable to our attack, while humans appear good at understanding the perturbed text. We also explore two methods, including adversarial training and associative graph-based recovery, to shield systems from chain association-based attack. Since a few examples that use some derogatory terms, this paper contains materials that may be offensive or upsetting to some people.

摘要: 联想作为一种礼物，使人们不必用完全直截了当的语言来提及某事，并让其他人理解他们想指的是什么。本文利用人与机器之间的理解鸿沟，提出了一种基于链式联想的对抗性自然语言处理系统攻击方法。首先在联想范式的基础上生成汉字的链式联想图，构建潜在对抗性实例的搜索空间。然后，我们引入了离散粒子群优化算法来搜索最优的对抗性实例。我们进行了全面的实验，并表明高级自然语言处理模型和应用程序，包括大型语言模型，容易受到我们的攻击，而人类似乎很擅长理解受干扰的文本。我们还探索了两种方法，包括对抗性训练和基于联想图的恢复，以保护系统免受基于链关联的攻击。由于有几个例子使用了一些贬义性的术语，因此本文包含的材料可能会冒犯某些人或使某些人不安。



## **25. CausalDiff: Causality-Inspired Disentanglement via Diffusion Model for Adversarial Defense**

卡西姆·分歧：通过对抗性防御的扩散模型来启发性解纠缠 cs.CV

accepted by NeurIPS 2024

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2410.23091v3) [paper-pdf](http://arxiv.org/pdf/2410.23091v3)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Quanrun Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: Despite ongoing efforts to defend neural classifiers from adversarial attacks, they remain vulnerable, especially to unseen attacks. In contrast, humans are difficult to be cheated by subtle manipulations, since we make judgments only based on essential factors. Inspired by this observation, we attempt to model label generation with essential label-causative factors and incorporate label-non-causative factors to assist data generation. For an adversarial example, we aim to discriminate the perturbations as non-causative factors and make predictions only based on the label-causative factors. Concretely, we propose a casual diffusion model (CausalDiff) that adapts diffusion models for conditional data generation and disentangles the two types of casual factors by learning towards a novel casual information bottleneck objective. Empirically, CausalDiff has significantly outperformed state-of-the-art defense methods on various unseen attacks, achieving an average robustness of 86.39% (+4.01%) on CIFAR-10, 56.25% (+3.13%) on CIFAR-100, and 82.62% (+4.93%) on GTSRB (German Traffic Sign Recognition Benchmark).

摘要: 尽管不断努力保护神经分类器免受对手攻击，但它们仍然很脆弱，特别是面对看不见的攻击。相比之下，人类很难被微妙的操纵所欺骗，因为我们只根据基本因素做出判断。受到这一观察的启发，我们试图用基本的标签原因因素来建模标签生成，并结合标签非原因因素来辅助数据生成。对于一个对抗性的例子，我们的目标是将扰动区分为非致因因素，并仅基于标签致因因素进行预测。具体地说，我们提出了一个偶然扩散模型(CausalDiff)，该模型使扩散模型适用于条件数据生成，并通过向一个新的偶然信息瓶颈目标学习来区分这两种类型的偶然因素。经验上，CausalDiff在各种隐形攻击上的表现明显优于最先进的防御方法，在CIFAR-10上获得了86.39%(+4.01%)的平均健壮性，在CIFAR-100上获得了56.25%(+3.13%)的健壮性，在GTSRB(德国交通标志识别基准)上实现了82.62%(+4.93%)的平均健壮性。



## **26. Revisiting the Adversarial Robustness of Vision Language Models: a Multimodal Perspective**

重新审视视觉语言模型的对抗鲁棒性：多模式视角 cs.CV

17 pages, 13 figures

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2404.19287v3) [paper-pdf](http://arxiv.org/pdf/2404.19287v3)

**Authors**: Wanqi Zhou, Shuanghao Bai, Danilo P. Mandic, Qibin Zhao, Badong Chen

**Abstract**: Pretrained vision-language models (VLMs) like CLIP exhibit exceptional generalization across diverse downstream tasks. While recent studies reveal their vulnerability to adversarial attacks, research to date has primarily focused on enhancing the robustness of image encoders against image-based attacks, with defenses against text-based and multimodal attacks remaining largely unexplored. To this end, this work presents the first comprehensive study on improving the adversarial robustness of VLMs against attacks targeting image, text, and multimodal inputs. This is achieved by proposing multimodal contrastive adversarial training (MMCoA). Such an approach strengthens the robustness of both image and text encoders by aligning the clean text embeddings with adversarial image embeddings, and adversarial text embeddings with clean image embeddings. The robustness of the proposed MMCoA is examined against existing defense methods over image, text, and multimodal attacks on the CLIP model. Extensive experiments on 15 datasets across two tasks reveal the characteristics of different adversarial defense methods under distinct distribution shifts and dataset complexities across the three attack types. This paves the way for a unified framework of adversarial robustness against different modality attacks, opening up new possibilities for securing VLMs against multimodal attacks. The code is available at https://github.com/ElleZWQ/MMCoA.git.

摘要: 像CLIP这样的预先训练的视觉语言模型(VLM)在不同的下游任务中表现出非凡的通用性。虽然最近的研究揭示了它们对对手攻击的脆弱性，但到目前为止的研究主要集中在增强图像编码器对基于图像的攻击的稳健性上，对基于文本的攻击和多模式攻击的防御在很大程度上仍未被探索。为此，本文首次全面研究了如何提高VLMS对图像、文本和多模式输入的攻击健壮性。这是通过提出多模式对比对抗训练(MMCoA)来实现的。这种方法通过将干净的文本嵌入与对抗性的图像嵌入以及对抗性的文本嵌入与干净的图像嵌入对齐来增强图像和文本编码器的稳健性。针对已有的针对图像、文本和多模式攻击的防御方法，对提出的MMCoA算法的鲁棒性进行了测试。在两个任务的15个数据集上进行了大量的实验，揭示了三种攻击类型在不同的分布变化和数据集复杂性下不同的对抗防御方法的特点。这为对抗不同模式攻击的对抗健壮性的统一框架铺平了道路，为保护VLM免受多模式攻击开辟了新的可能性。代码可在https://github.com/ElleZWQ/MMCoA.git.上获得



## **27. Data-Driven Graph Switching for Cyber-Resilient Control in Microgrids**

数据驱动的图形交换用于微电网中的网络弹性控制 eess.SY

Accepted in IEEE Design Methodologies Conference (DMC) 2024

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07686v1) [paper-pdf](http://arxiv.org/pdf/2411.07686v1)

**Authors**: Suman Rath, Subham Sahoo

**Abstract**: Distributed microgrids are conventionally dependent on communication networks to achieve secondary control objectives. This dependence makes them vulnerable to stealth data integrity attacks (DIAs) where adversaries may perform manipulations via infected transmitters and repeaters to jeopardize stability. This paper presents a physics-guided, supervised Artificial Neural Network (ANN)-based framework that identifies communication-level cyberattacks in microgrids by analyzing whether incoming measurements will cause abnormal behavior of the secondary control layer. If abnormalities are detected, an iteration through possible spanning tree graph topologies that can be used to fulfill secondary control objectives is done. Then, a communication network topology that would not create secondary control abnormalities is identified and enforced for maximum stability. By altering the communication graph topology, the framework eliminates the dependence of the secondary control layer on inputs from compromised cyber devices helping it achieve resilience without instability. Several case studies are provided showcasing the robustness of the framework against False Data Injections and repeater-level Man-in-the-Middle attacks. To understand practical feasibility, robustness is also verified against larger microgrid sizes and in the presence of varying noise levels. Our findings indicate that performance can be affected when attempting scalability in the presence of noise. However, the framework operates robustly in low-noise settings.

摘要: 传统上，分布式微电网依靠通信网络来实现二次控制目标。这种依赖使它们容易受到隐形数据完整性攻击(DIA)，攻击者可能会通过受感染的发射器和中继器执行操作，从而危及稳定性。提出了一种基于物理引导的有监督人工神经网络(ANN)框架，该框架通过分析输入测量是否会导致二次控制层的异常行为来识别微电网中的通信级网络攻击。如果检测到异常，则对可用于实现二级控制目标的可能的生成树图拓扑进行迭代。然后，识别并实施不会产生二次控制异常的通信网络拓扑以实现最大稳定性。通过改变通信图拓扑，该框架消除了二次控制层对来自受损网络设备的输入的依赖，帮助它实现了弹性，而不会出现不稳定。提供了几个案例研究，展示了该框架对虚假数据注入和中继器级别的中间人攻击的健壮性。为了了解实际可行性，还针对较大的微电网规模和不同的噪声水平验证了稳健性。我们的发现表明，在存在噪声的情况下尝试可伸缩性时，性能可能会受到影响。然而，该框架在低噪声设置下运行稳健。



## **28. Aligning Visual Contrastive learning models via Preference Optimization**

通过偏好优化调整视觉对比学习模型 cs.CV

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.08923v1) [paper-pdf](http://arxiv.org/pdf/2411.08923v1)

**Authors**: Amirabbas Afzali, Borna Khodabandeh, Ali Rasekh, Mahyar JafariNodeh, Sepehr kazemi, Simon Gottschalk

**Abstract**: Contrastive learning models have demonstrated impressive abilities to capture semantic similarities by aligning representations in the embedding space. However, their performance can be limited by the quality of the training data and its inherent biases. While Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO) have been applied to generative models to align them with human preferences, their use in contrastive learning has yet to be explored. This paper introduces a novel method for training contrastive learning models using Preference Optimization (PO) to break down complex concepts. Our method systematically aligns model behavior with desired preferences, enhancing performance on the targeted task. In particular, we focus on enhancing model robustness against typographic attacks, commonly seen in contrastive models like CLIP. We further apply our method to disentangle gender understanding and mitigate gender biases, offering a more nuanced control over these sensitive attributes. Our experiments demonstrate that models trained using PO outperform standard contrastive learning techniques while retaining their ability to handle adversarial challenges and maintain accuracy on other downstream tasks. This makes our method well-suited for tasks requiring fairness, robustness, and alignment with specific preferences. We evaluate our method on several vision-language tasks, tackling challenges such as typographic attacks. Additionally, we explore the model's ability to disentangle gender concepts and mitigate gender bias, showcasing the versatility of our approach.

摘要: 对比学习模型已经显示出令人印象深刻的能力，通过在嵌入空间中对齐表征来捕捉语义相似性。然而，它们的表现可能会受到训练数据质量及其固有偏差的限制。虽然人类反馈强化学习(RLHF)和直接偏好优化(DPO)已经被应用于生成模型以使它们与人类偏好相一致，但它们在对比学习中的应用还有待探索。提出了一种利用偏好优化(PO)分解复杂概念训练对比学习模型的新方法。我们的方法系统地使模型行为与期望的偏好保持一致，从而提高目标任务的性能。特别是，我们专注于增强模型对排版攻击的稳健性，这在CLIP等对比模型中很常见。我们进一步应用我们的方法来理清性别理解并缓解性别偏见，对这些敏感属性提供更细微的控制。我们的实验表明，使用PO训练的模型优于标准的对比学习技术，同时保持了它们处理对抗性挑战和在其他下游任务上保持准确性的能力。这使得我们的方法非常适合需要公平性、健壮性和与特定偏好一致的任务。我们在几个视觉语言任务上评估了我们的方法，解决了诸如排版攻击等挑战。此外，我们还探索了该模型理清性别概念和减轻性别偏见的能力，展示了我们方法的多功能性。



## **29. A Survey on Adversarial Machine Learning for Code Data: Realistic Threats, Countermeasures, and Interpretations**

代码数据对抗性机器学习调查：现实威胁、对策和解释 cs.CR

Under a reviewing process since Sep. 3, 2024

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07597v1) [paper-pdf](http://arxiv.org/pdf/2411.07597v1)

**Authors**: Yulong Yang, Haoran Fan, Chenhao Lin, Qian Li, Zhengyu Zhao, Chao Shen, Xiaohong Guan

**Abstract**: Code Language Models (CLMs) have achieved tremendous progress in source code understanding and generation, leading to a significant increase in research interests focused on applying CLMs to real-world software engineering tasks in recent years. However, in realistic scenarios, CLMs are exposed to potential malicious adversaries, bringing risks to the confidentiality, integrity, and availability of CLM systems. Despite these risks, a comprehensive analysis of the security vulnerabilities of CLMs in the extremely adversarial environment has been lacking. To close this research gap, we categorize existing attack techniques into three types based on the CIA triad: poisoning attacks (integrity \& availability infringement), evasion attacks (integrity infringement), and privacy attacks (confidentiality infringement). We have collected so far the most comprehensive (79) papers related to adversarial machine learning for CLM from the research fields of artificial intelligence, computer security, and software engineering. Our analysis covers each type of risk, examining threat model categorization, attack techniques, and countermeasures, while also introducing novel perspectives on eXplainable AI (XAI) and exploring the interconnections between different risks. Finally, we identify current challenges and future research opportunities. This study aims to provide a comprehensive roadmap for both researchers and practitioners and pave the way towards more reliable CLMs for practical applications.

摘要: 代码语言模型(CLMS)在源代码理解和生成方面取得了巨大的进步，导致近年来将代码语言模型应用于实际软件工程任务的研究兴趣显著增加。然而，在现实场景中，CLM暴露在潜在的恶意攻击者面前，给CLM系统的机密性、完整性和可用性带来了风险。尽管存在这些风险，但在极端敌对的环境中，缺乏对CLMS安全漏洞的全面分析。为了缩小这一研究空白，我们根据CIA三合会将现有的攻击技术分为三类：中毒攻击(完整性和可用性破坏)、逃避攻击(完整性破坏)和隐私攻击(保密侵犯)。到目前为止，我们已经从人工智能、计算机安全和软件工程的研究领域收集了与CLM的对抗性机器学习相关的最全面的(79)篇论文。我们的分析涵盖了每种类型的风险，研究了威胁模型分类、攻击技术和对策，同时也引入了关于可解释人工智能(XAI)的新视角，并探索了不同风险之间的相互联系。最后，我们确定了当前的挑战和未来的研究机会。这项研究旨在为研究人员和实践者提供一个全面的路线图，并为更可靠的CLMS的实际应用铺平道路。



## **30. Graph Agent Network: Empowering Nodes with Inference Capabilities for Adversarial Resilience**

图代理网络：赋予节点推理能力以对抗复原力 cs.LG

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2306.06909v4) [paper-pdf](http://arxiv.org/pdf/2306.06909v4)

**Authors**: Ao Liu, Wenshan Li, Tao Li, Beibei Li, Guangquan Xu, Pan Zhou, Wengang Ma, Hanyuan Huang

**Abstract**: End-to-end training with global optimization have popularized graph neural networks (GNNs) for node classification, yet inadvertently introduced vulnerabilities to adversarial edge-perturbing attacks. Adversaries can exploit the inherent opened interfaces of GNNs' input and output, perturbing critical edges and thus manipulating the classification results. Current defenses, due to their persistent utilization of global-optimization-based end-to-end training schemes, inherently encapsulate the vulnerabilities of GNNs. This is specifically evidenced in their inability to defend against targeted secondary attacks. In this paper, we propose the Graph Agent Network (GAgN) to address the aforementioned vulnerabilities of GNNs. GAgN is a graph-structured agent network in which each node is designed as an 1-hop-view agent. Through the decentralized interactions between agents, they can learn to infer global perceptions to perform tasks including inferring embeddings, degrees and neighbor relationships for given nodes. This empowers nodes to filtering adversarial edges while carrying out classification tasks. Furthermore, agents' limited view prevents malicious messages from propagating globally in GAgN, thereby resisting global-optimization-based secondary attacks. We prove that single-hidden-layer multilayer perceptrons (MLPs) are theoretically sufficient to achieve these functionalities. Experimental results show that GAgN effectively implements all its intended capabilities and, compared to state-of-the-art defenses, achieves optimal classification accuracy on the perturbed datasets.

摘要: 具有全局优化的端到端训练普及了图神经网络(GNN)用于节点分类，但无意中引入了对敌意边缘扰动攻击的脆弱性。攻击者可以利用GNN输入和输出固有的开放接口，扰乱关键边缘，从而操纵分类结果。目前的防御措施由于持续使用基于全局优化的端到端培训方案，固有地封装了GNN的脆弱性。这一点具体表现在他们无法防御有针对性的二次攻击。在本文中，我们提出了图代理网络(GagN)来解决GNN的上述漏洞。GAGN是一个图结构的代理网络，其中每个节点被设计为一个1跳视图代理。通过代理之间的分散交互，它们可以学习推断全局感知来执行任务，包括推断给定节点的嵌入度、度数和邻居关系。这使节点能够在执行分类任务时过滤敌意边缘。此外，代理的有限视点防止恶意消息在GAGN中全局传播，从而抵抗基于全局优化的二次攻击。我们证明了单隐层多层感知器(MLP)理论上足以实现这些功能。实验结果表明，GAGN有效地实现了其预期的所有功能，并且与现有的防御措施相比，在扰动数据集上获得了最优的分类精度。



## **31. Fast Preemption: Forward-Backward Cascade Learning for Efficient and Transferable Proactive Adversarial Defense**

快速抢占：前向-后向级联学习，实现高效且可转移的主动对抗防御 cs.CR

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2407.15524v4) [paper-pdf](http://arxiv.org/pdf/2407.15524v4)

**Authors**: Hanrui Wang, Ching-Chun Chang, Chun-Shien Lu, Isao Echizen

**Abstract**: Deep learning technology has brought convenience and advanced developments but has become untrustworthy due to its sensitivity to adversarial attacks. Attackers may utilize this sensitivity to manipulate predictions. To defend against such attacks, existing anti-adversarial methods typically counteract adversarial perturbations post-attack, while we have devised a proactive strategy that preempts by safeguarding media upfront, effectively neutralizing potential adversarial effects before the third-party attacks occur. This strategy, dubbed Fast Preemption, provides an efficient transferable preemptive defense by using different models for labeling inputs and learning crucial features. A forward-backward cascade learning algorithm is used to compute protective perturbations, starting with forward propagation optimization to achieve rapid convergence, followed by iterative backward propagation learning to alleviate overfitting. This strategy offers state-of-the-art transferability and protection across various systems. With the running of only three steps, our Fast Preemption framework outperforms benchmark training-time, test-time, and preemptive adversarial defenses. We have also devised the first, to our knowledge, effective white-box adaptive reversion attack and demonstrate that the protection added by our defense strategy is irreversible unless the backbone model, algorithm, and settings are fully compromised. This work provides a new direction to developing proactive defenses against adversarial attacks.

摘要: 深度学习技术带来了便利和先进的发展，但由于其对对手攻击的敏感性而变得不可信任。攻击者可能会利用这种敏感性来操纵预测。为了防御此类攻击，现有的反对手方法通常在攻击后抵消对手干扰，而我们设计了一种主动战略，通过预先保护媒体来抢占先机，有效地在第三方攻击发生之前消除潜在的对手影响。这一战略被称为快速抢占，通过使用不同的模型来标记输入和学习关键特征，提供了一种高效的可转移的抢占防御。保护摄动的计算采用前向-后向级联学习算法，从前向传播优化开始实现快速收敛，然后迭代后向传播学习以减少过拟合。这一战略提供了最先进的跨各种系统的可转移性和保护。由于只运行了三个步骤，我们的快速抢占框架的性能优于基准训练时间、测试时间和先发制人的对手防御。据我们所知，我们还设计了第一个有效的白盒自适应恢复攻击，并证明了除非主干模型、算法和设置完全受损，否则我们的防御策略添加的保护是不可逆转的。这项工作为主动防御对抗性攻击提供了新的方向。



## **32. Rapid Response: Mitigating LLM Jailbreaks with a Few Examples**

快速反应：通过一些例子缓解LLM越狱 cs.CL

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07494v1) [paper-pdf](http://arxiv.org/pdf/2411.07494v1)

**Authors**: Alwin Peng, Julian Michael, Henry Sleight, Ethan Perez, Mrinank Sharma

**Abstract**: As large language models (LLMs) grow more powerful, ensuring their safety against misuse becomes crucial. While researchers have focused on developing robust defenses, no method has yet achieved complete invulnerability to attacks. We propose an alternative approach: instead of seeking perfect adversarial robustness, we develop rapid response techniques to look to block whole classes of jailbreaks after observing only a handful of attacks. To study this setting, we develop RapidResponseBench, a benchmark that measures a defense's robustness against various jailbreak strategies after adapting to a few observed examples. We evaluate five rapid response methods, all of which use jailbreak proliferation, where we automatically generate additional jailbreaks similar to the examples observed. Our strongest method, which fine-tunes an input classifier to block proliferated jailbreaks, reduces attack success rate by a factor greater than 240 on an in-distribution set of jailbreaks and a factor greater than 15 on an out-of-distribution set, having observed just one example of each jailbreaking strategy. Moreover, further studies suggest that the quality of proliferation model and number of proliferated examples play an key role in the effectiveness of this defense. Overall, our results highlight the potential of responding rapidly to novel jailbreaks to limit LLM misuse.

摘要: 随着大型语言模型(LLM)变得越来越强大，确保它们的安全性以防止误用变得至关重要。虽然研究人员专注于开发强大的防御系统，但还没有一种方法能够完全抵御攻击。我们提出了另一种方法：我们不是寻求完美的对手健壮性，而是开发快速响应技术，在仅观察到少数几次攻击后，寻求阻止整个类别的越狱。为了研究这种情况，我们开发了RapidResponseBch，这是一个基准，在适应了几个观察到的例子后，衡量了防御对各种越狱策略的健壮性。我们评估了五种快速响应方法，所有这些方法都使用越狱扩散，在这些方法中，我们自动生成与观察到的示例类似的额外越狱。我们最强大的方法是微调输入分类器以阻止越狱激增，在仅观察到每个越狱策略的一个示例后，在分布内越狱集合上将攻击成功率降低240倍以上，在分布外集合上降低15倍以上。此外，进一步的研究表明，扩散模型的质量和扩散实例的数量在这一防御措施的有效性中起着关键作用。总体而言，我们的结果突出了对新型越狱做出快速反应以限制LLM滥用的潜力。



## **33. DrAttack: Prompt Decomposition and Reconstruction Makes Powerful LLM Jailbreakers**

DrAttack：强大的快速分解和重建让LLM越狱者 cs.CR

**SubmitDate**: 2024-11-11    [abs](http://arxiv.org/abs/2402.16914v3) [paper-pdf](http://arxiv.org/pdf/2402.16914v3)

**Authors**: Xirui Li, Ruochen Wang, Minhao Cheng, Tianyi Zhou, Cho-Jui Hsieh

**Abstract**: The safety alignment of Large Language Models (LLMs) is vulnerable to both manual and automated jailbreak attacks, which adversarially trigger LLMs to output harmful content. However, current methods for jailbreaking LLMs, which nest entire harmful prompts, are not effective at concealing malicious intent and can be easily identified and rejected by well-aligned LLMs. This paper discovers that decomposing a malicious prompt into separated sub-prompts can effectively obscure its underlying malicious intent by presenting it in a fragmented, less detectable form, thereby addressing these limitations. We introduce an automatic prompt \textbf{D}ecomposition and \textbf{R}econstruction framework for jailbreak \textbf{Attack} (DrAttack). DrAttack includes three key components: (a) `Decomposition' of the original prompt into sub-prompts, (b) `Reconstruction' of these sub-prompts implicitly by in-context learning with semantically similar but harmless reassembling demo, and (c) a `Synonym Search' of sub-prompts, aiming to find sub-prompts' synonyms that maintain the original intent while jailbreaking LLMs. An extensive empirical study across multiple open-source and closed-source LLMs demonstrates that, with a significantly reduced number of queries, DrAttack obtains a substantial gain of success rate over prior SOTA prompt-only attackers. Notably, the success rate of 78.0\% on GPT-4 with merely 15 queries surpassed previous art by 33.1\%. The project is available at https://github.com/xirui-li/DrAttack.

摘要: 大型语言模型(LLM)的安全一致性容易受到手动和自动越狱攻击的攻击，这些攻击会相反地触发LLM输出有害内容。然而，当前的越狱LLM方法嵌套了整个有害的提示，在隐藏恶意意图方面并不有效，很容易被排列良好的LLM识别和拒绝。本文发现，通过将恶意提示分解为单独的子提示，可以通过以零散的、较难检测的形式来表示恶意提示，从而有效地掩盖潜在的恶意意图，从而解决这些限制。介绍了一种自动提示的文本bf{D}分解和文本bf{R}重构框架(DrAttack)。DrAttack包括三个关键组件：(A)将原始提示‘分解’成子提示，(B)通过使用语义相似但无害的重组演示的上下文学习隐式地‘重构’这些子提示，以及(C)对子提示进行‘同步搜索’，目的是在越狱LLM的同时找到保持原意的子提示的同义词。一项针对多个开源和闭源LLM的广泛经验研究表明，DrAttack在查询次数显著减少的情况下，与之前的Sota仅提示攻击者相比，获得了显著的成功率。值得注意的是，仅用15个查询在GPT-4上的成功率为78.0\%，比以前的ART高出33.1\%。该项目的网址为：https://github.com/xirui-li/DrAttack.。



## **34. The Inherent Adversarial Robustness of Analog In-Memory Computing**

模拟内存计算固有的对抗鲁棒性 cs.ET

**SubmitDate**: 2024-11-11    [abs](http://arxiv.org/abs/2411.07023v1) [paper-pdf](http://arxiv.org/pdf/2411.07023v1)

**Authors**: Corey Lammie, Julian Büchel, Athanasios Vasilopoulos, Manuel Le Gallo, Abu Sebastian

**Abstract**: A key challenge for Deep Neural Network (DNN) algorithms is their vulnerability to adversarial attacks. Inherently non-deterministic compute substrates, such as those based on Analog In-Memory Computing (AIMC), have been speculated to provide significant adversarial robustness when performing DNN inference. In this paper, we experimentally validate this conjecture for the first time on an AIMC chip based on Phase Change Memory (PCM) devices. We demonstrate higher adversarial robustness against different types of adversarial attacks when implementing an image classification network. Additional robustness is also observed when performing hardware-in-the-loop attacks, for which the attacker is assumed to have full access to the hardware. A careful study of the various noise sources indicate that a combination of stochastic noise sources (both recurrent and non-recurrent) are responsible for the adversarial robustness and that their type and magnitude disproportionately effects this property. Finally, it is demonstrated, via simulations, that when a much larger transformer network is used to implement a Natural Language Processing (NLP) task, additional robustness is still observed.

摘要: 深度神经网络(DNN)算法面临的一个关键挑战是它们对对手攻击的脆弱性。固有的非确定性计算基板，例如基于模拟内存计算(AIMC)的基板，被推测在执行DNN推理时提供显著的对抗性健壮性。在本文中，我们首次在基于相变存储(PCM)器件的AIMC芯片上实验验证了这一猜想。在实现图像分类网络时，我们表现出对不同类型的对抗性攻击的更高的对抗性鲁棒性。在执行硬件在环攻击时，还可以观察到额外的稳健性，假定攻击者对硬件具有完全访问权限。对各种噪声源的仔细研究表明，随机噪声源(循环和非循环)的组合是造成对抗鲁棒性的原因，并且它们的类型和大小不成比例地影响这一特性。最后，通过仿真证明，当使用更大的变压器网络来实现自然语言处理(NLP)任务时，仍然可以观察到额外的稳健性。



## **35. Computable Model-Independent Bounds for Adversarial Quantum Machine Learning**

对抗性量子机器学习的可计算模型独立边界 cs.LG

21 pages, 9 figures

**SubmitDate**: 2024-11-11    [abs](http://arxiv.org/abs/2411.06863v1) [paper-pdf](http://arxiv.org/pdf/2411.06863v1)

**Authors**: Bacui Li, Tansu Alpcan, Chandra Thapa, Udaya Parampalli

**Abstract**: By leveraging the principles of quantum mechanics, QML opens doors to novel approaches in machine learning and offers potential speedup. However, machine learning models are well-documented to be vulnerable to malicious manipulations, and this susceptibility extends to the models of QML. This situation necessitates a thorough understanding of QML's resilience against adversarial attacks, particularly in an era where quantum computing capabilities are expanding. In this regard, this paper examines model-independent bounds on adversarial performance for QML. To the best of our knowledge, we introduce the first computation of an approximate lower bound for adversarial error when evaluating model resilience against sophisticated quantum-based adversarial attacks. Experimental results are compared to the computed bound, demonstrating the potential of QML models to achieve high robustness. In the best case, the experimental error is only 10% above the estimated bound, offering evidence of the inherent robustness of quantum models. This work not only advances our theoretical understanding of quantum model resilience but also provides a precise reference bound for the future development of robust QML algorithms.

摘要: 通过利用量子力学的原理，QML为机器学习中的新方法打开了大门，并提供了潜在的加速比。然而，机器学习模型很容易受到恶意操作，这种易感性延伸到QML模型。这种情况需要彻底了解QML对对手攻击的韧性，特别是在量子计算能力不断扩大的时代。在这方面，本文研究了QML对抗性能的与模型无关的界。据我们所知，在评估模型对复杂的基于量子的敌意攻击的弹性时，我们引入了对抗性错误的近似下界的第一次计算。实验结果与计算界进行了比较，证明了QML模型具有较高的鲁棒性。在最好的情况下，实验误差仅比估计值高出10%，这为量子模型的内在稳健性提供了证据。这项工作不仅加深了我们对量子模型弹性的理论理解，而且为未来健壮的QML算法的发展提供了一个精确的参考界。



## **36. Boosting the Targeted Transferability of Adversarial Examples via Salient Region & Weighted Feature Drop**

通过显著区域和加权特征下降提高对抗性示例的目标可移植性 cs.IR

9 pages

**SubmitDate**: 2024-11-11    [abs](http://arxiv.org/abs/2411.06784v1) [paper-pdf](http://arxiv.org/pdf/2411.06784v1)

**Authors**: Shanjun Xu, Linghui Li, Kaiguo Yuan, Bingyu Li

**Abstract**: Deep neural networks can be vulnerable to adversarially crafted examples, presenting significant risks to practical applications. A prevalent approach for adversarial attacks relies on the transferability of adversarial examples, which are generated from a substitute model and leveraged to attack unknown black-box models. Despite various proposals aimed at improving transferability, the success of these attacks in targeted black-box scenarios is often hindered by the tendency for adversarial examples to overfit to the surrogate models. In this paper, we introduce a novel framework based on Salient region & Weighted Feature Drop (SWFD) designed to enhance the targeted transferability of adversarial examples. Drawing from the observation that examples with higher transferability exhibit smoother distributions in the deep-layer outputs, we propose the weighted feature drop mechanism to modulate activation values according to weights scaled by norm distribution, effectively addressing the overfitting issue when generating adversarial examples. Additionally, by leveraging salient region within the image to construct auxiliary images, our method enables the adversarial example's features to be transferred to the target category in a model-agnostic manner, thereby enhancing the transferability. Comprehensive experiments confirm that our approach outperforms state-of-the-art methods across diverse configurations. On average, the proposed SWFD raises the attack success rate for normally trained models and robust models by 16.31% and 7.06% respectively.

摘要: 深度神经网络可能容易受到恶意构建的示例的攻击，从而给实际应用带来重大风险。对抗性攻击的一种普遍方法依赖于对抗性例子的可转移性，这些对抗性例子由替代模型生成并被用来攻击未知的黑盒模型。尽管提出了各种旨在提高可转移性的建议，但这些攻击在有针对性的黑盒场景中的成功往往受到对抗性例子过度适应代理模型的趋势的阻碍。在本文中，我们介绍了一种新的框架，该框架基于突出区域&加权特征丢弃(SWFD)，旨在增强对抗性例子的定向可转移性。根据可转移性较高的样本在深层输出中表现出更平滑的分布这一观察结果，我们提出了加权特征丢弃机制，根据范数分布衡量的权重来调整激活值，有效地解决了生成对抗性样本时的过拟合问题。此外，通过利用图像中的显著区域构造辅助图像，我们的方法能够以模型无关的方式将对抗性例子的特征转移到目标类别，从而增强了可转移性。综合实验证实，我们的方法在不同的配置上比最先进的方法性能更好。平均而言，SWFD使正常训练模型和稳健模型的攻击成功率分别提高了16.31%和7.06%。



## **37. Beyond Text: Utilizing Vocal Cues to Improve Decision Making in LLMs for Robot Navigation Tasks**

超越文本：利用人声线索改善LLM机器人导航任务的决策 cs.AI

30 pages, 7 figures

**SubmitDate**: 2024-11-11    [abs](http://arxiv.org/abs/2402.03494v3) [paper-pdf](http://arxiv.org/pdf/2402.03494v3)

**Authors**: Xingpeng Sun, Haoming Meng, Souradip Chakraborty, Amrit Singh Bedi, Aniket Bera

**Abstract**: While LLMs excel in processing text in these human conversations, they struggle with the nuances of verbal instructions in scenarios like social navigation, where ambiguity and uncertainty can erode trust in robotic and other AI systems. We can address this shortcoming by moving beyond text and additionally focusing on the paralinguistic features of these audio responses. These features are the aspects of spoken communication that do not involve the literal wording (lexical content) but convey meaning and nuance through how something is said. We present Beyond Text: an approach that improves LLM decision-making by integrating audio transcription along with a subsection of these features, which focus on the affect and more relevant in human-robot conversations.This approach not only achieves a 70.26% winning rate, outperforming existing LLMs by 22.16% to 48.30% (gemini-1.5-pro and gpt-3.5 respectively), but also enhances robustness against token manipulation adversarial attacks, highlighted by a 22.44% less decrease ratio than the text-only language model in winning rate. Beyond Text' marks an advancement in social robot navigation and broader Human-Robot interactions, seamlessly integrating text-based guidance with human-audio-informed language models.

摘要: 虽然LLM在处理这些人类对话中的文本方面表现出色，但它们在社交导航等场景中难以处理语言指令的细微差别，在这些场景中，模棱两可和不确定性可能会侵蚀人们对机器人和其他人工智能系统的信任。我们可以通过超越文本并另外关注这些音频反应的副语言特征来解决这一缺点。这些特征是口语交际的方面，不涉及字面上的措辞(词汇内容)，但通过说话方式传达意义和细微差别。我们提出了Beyond Text：一种改进LLM决策的方法，它集成了音频转录和这些特征的一部分，这些特征集中在人-机器人对话中的影响和更相关的方面。该方法不仅获得了70.26%的优胜率，比现有的LLM分别提高了22.16%到48.30%(分别为Gemini-1.5-Pro和GPT-3.5)，而且还增强了对令牌操纵对手攻击的鲁棒性，其优胜率比纯文本语言模型降低了22.44%。Beyond Text‘标志着社交机器人导航和更广泛的人-机器人交互方面的进步，无缝地将基于文本的指导与人-音频信息语言模型相结合。



## **38. Adversarial Detection with a Dynamically Stable System**

具有动态稳定系统的对抗性检测 cs.AI

**SubmitDate**: 2024-11-11    [abs](http://arxiv.org/abs/2411.06666v1) [paper-pdf](http://arxiv.org/pdf/2411.06666v1)

**Authors**: Xiaowei Long, Jie Lin, Xiangyuan Yang

**Abstract**: Adversarial detection is designed to identify and reject maliciously crafted adversarial examples(AEs) which are generated to disrupt the classification of target models.   Presently, various input transformation-based methods have been developed on adversarial example detection, which typically rely on empirical experience and lead to unreliability against new attacks.   To address this issue, we propose and conduct a Dynamically Stable System (DSS), which can effectively detect the adversarial examples from normal examples according to the stability of input examples.   Particularly, in our paper, the generation of adversarial examples is considered as the perturbation process of a Lyapunov dynamic system, and we propose an example stability mechanism, in which a novel control term is added in adversarial example generation to ensure that the normal examples can achieve dynamic stability while the adversarial examples cannot achieve the stability.   Then, based on the proposed example stability mechanism, a Dynamically Stable System (DSS) is proposed, which can utilize the disruption and restoration actions to determine the stability of input examples and detect the adversarial examples through changes in the stability of the input examples.   In comparison with existing methods in three benchmark datasets(MNIST, CIFAR10, and CIFAR100), our evaluation results show that our proposed DSS can achieve ROC-AUC values of 99.83%, 97.81% and 94.47%, surpassing the state-of-the-art(SOTA) values of 97.35%, 91.10% and 93.49% in the other 7 methods.

摘要: 敌意检测旨在识别和拒绝恶意构建的敌意示例(AE)，这些AE是为了扰乱目标模型的分类而生成的。目前，已有多种基于输入变换的对抗性样本检测方法，这些方法通常依赖于经验，对新的攻击不可靠。针对这一问题，我们提出并实现了一个动态稳定系统(DSS)，该系统能够根据输入样本的稳定性有效地从正常样本中检测出敌意样本。特别地，本文将对抗性实例的生成看作是一个Lyapunov动态系统的摄动过程，并提出了一种实例稳定机制，在对抗性实例生成过程中增加了一个新的控制项，以保证正常实例能够实现动态稳定，而对抗性实例不能实现稳定性。然后，基于所提出的实例稳定机制，提出了一个动态稳定系统(DSS)，该系统可以利用中断和恢复行为来确定输入实例的稳定性，并通过输入实例稳定性的变化来检测敌意实例。在三个基准数据集(MNIST、CIFAR10和CIFAR100)上的评估结果表明，我们提出的决策支持系统ROC-AUC值分别为99.83%、97.81%和94.47%，超过了其他7种方法的ROC-AUC值97.35%、91.10%和93.49%。



## **39. Do Unlearning Methods Remove Information from Language Model Weights?**

取消学习方法会从语言模型权重中删除信息吗？ cs.LG

**SubmitDate**: 2024-11-10    [abs](http://arxiv.org/abs/2410.08827v2) [paper-pdf](http://arxiv.org/pdf/2410.08827v2)

**Authors**: Aghyad Deeb, Fabien Roger

**Abstract**: Large Language Models' knowledge of how to perform cyber-security attacks, create bioweapons, and manipulate humans poses risks of misuse. Previous work has proposed methods to unlearn this knowledge. Historically, it has been unclear whether unlearning techniques are removing information from the model weights or just making it harder to access. To disentangle these two objectives, we propose an adversarial evaluation method to test for the removal of information from model weights: we give an attacker access to some facts that were supposed to be removed, and using those, the attacker tries to recover other facts from the same distribution that cannot be guessed from the accessible facts. We show that using fine-tuning on the accessible facts can recover 88% of the pre-unlearning accuracy when applied to current unlearning methods, revealing the limitations of these methods in removing information from the model weights.

摘要: 大型语言模型关于如何执行网络安全攻击、制造生物武器和操纵人类的知识带来了滥用的风险。之前的工作提出了忘记这些知识的方法。从历史上看，目前尚不清楚取消学习技术是否正在从模型权重中删除信息，或者只是使其更难访问。为了解开这两个目标，我们提出了一种对抗评估方法来测试从模型权重中删除信息的情况：我们让攻击者访问一些应该删除的事实，并使用这些事实，攻击者试图从无法从可访问的事实中猜测到的相同分布中恢复其他事实。我们表明，当应用于当前的取消学习方法时，对可访问的事实进行微调可以恢复取消学习前的88%的准确性，揭示了这些方法在从模型权重中删除信息方面的局限性。



## **40. HidePrint: Hiding the Radio Fingerprint via Random Noise**

HidePrint：通过随机噪音隐藏无线电指纹 cs.CR

**SubmitDate**: 2024-11-10    [abs](http://arxiv.org/abs/2411.06417v1) [paper-pdf](http://arxiv.org/pdf/2411.06417v1)

**Authors**: Gabriele Oligeri, Savio Sciancalepore

**Abstract**: Radio Frequency Fingerprinting (RFF) techniques allow a receiver to authenticate a transmitter by analyzing the physical layer of the radio spectrum. Although the vast majority of scientific contributions focus on improving the performance of RFF considering different parameters and scenarios, in this work, we consider RFF as an attack vector to identify and track a target device.   We propose, implement, and evaluate HidePrint, a solution to prevent tracking through RFF without affecting the quality of the communication link between the transmitter and the receiver. HidePrint hides the transmitter's fingerprint against an illegitimate eavesdropper by injecting controlled noise in the transmitted signal. We evaluate our solution against state-of-the-art image-based RFF techniques considering different adversarial models, different communication links (wired and wireless), and different configurations. Our results show that the injection of a Gaussian noise pattern with a standard deviation of (at least) 0.02 prevents device fingerprinting in all the considered scenarios, thus making the performance of the identification process indistinguishable from the random guess while affecting the Signal-to-Noise Ratio (SNR) of the received signal by only 0.1 dB. Moreover, we introduce selective radio fingerprint disclosure, a new technique that allows the transmitter to disclose the radio fingerprint to only a subset of intended receivers. This technique allows the transmitter to regain anonymity, thus preventing identification and tracking while allowing authorized receivers to authenticate the transmitter without affecting the quality of the transmitted signal.

摘要: 射频指纹(RFF)技术允许接收器通过分析无线电频谱的物理层来验证发射器。虽然绝大多数的科学贡献都集中在考虑不同参数和场景的情况下提高RFF的性能，但在这项工作中，我们将RFF视为识别和跟踪目标设备的攻击矢量。我们提出、实现和评估了HidePrint，这是一种在不影响发送器和接收器之间的通信链路质量的情况下防止通过RFF进行跟踪的解决方案。HidePrint通过在传输的信号中注入受控噪声来隐藏发射器的指纹，以防止非法窃听者。我们针对最先进的基于图像的RFF技术对我们的解决方案进行了评估，考虑了不同的对抗模型、不同的通信链路(有线和无线)和不同的配置。我们的结果表明，在所有考虑的场景中，注入标准差为(至少)0.02的高斯噪声模式防止了设备指纹识别，从而使识别过程的性能与随机猜测难以区分，而对接收信号的信噪比(SNR)的影响仅为0.1dB。此外，我们引入了选择性无线电指纹披露，这是一种新的技术，允许发射机只向目标接收者的子集披露无线电指纹。该技术允许发射机重新获得匿名性，从而防止识别和跟踪，同时允许授权的接收机在不影响传输信号质量的情况下认证发射机。



## **41. Randomized Message-Interception Smoothing: Gray-box Certificates for Graph Neural Networks**

随机消息拦截平滑：图神经网络的灰箱证书 cs.LG

Accepted at NeurIPS 2022

**SubmitDate**: 2024-11-10    [abs](http://arxiv.org/abs/2301.02039v2) [paper-pdf](http://arxiv.org/pdf/2301.02039v2)

**Authors**: Yan Scholten, Jan Schuchardt, Simon Geisler, Aleksandar Bojchevski, Stephan Günnemann

**Abstract**: Randomized smoothing is one of the most promising frameworks for certifying the adversarial robustness of machine learning models, including Graph Neural Networks (GNNs). Yet, existing randomized smoothing certificates for GNNs are overly pessimistic since they treat the model as a black box, ignoring the underlying architecture. To remedy this, we propose novel gray-box certificates that exploit the message-passing principle of GNNs: We randomly intercept messages and carefully analyze the probability that messages from adversarially controlled nodes reach their target nodes. Compared to existing certificates, we certify robustness to much stronger adversaries that control entire nodes in the graph and can arbitrarily manipulate node features. Our certificates provide stronger guarantees for attacks at larger distances, as messages from farther-away nodes are more likely to get intercepted. We demonstrate the effectiveness of our method on various models and datasets. Since our gray-box certificates consider the underlying graph structure, we can significantly improve certifiable robustness by applying graph sparsification.

摘要: 随机化平滑是证明机器学习模型(包括图神经网络)对抗稳健性的最有前途的框架之一。然而，现有的用于GNN的随机化平滑证书过于悲观，因为它们将模型视为黑匣子，忽略了底层架构。为了解决这个问题，我们提出了一种新的灰盒证书，它利用了GNN的消息传递原理：我们随机截获消息，并仔细分析来自恶意控制节点的消息到达目标节点的概率。与现有的证书相比，我们证明了对控制图中的整个节点并可以任意操纵节点特征的更强大的攻击者的健壮性。我们的证书为更远距离的攻击提供了更强有力的保证，因为来自较远节点的消息更有可能被拦截。我们在不同的模型和数据集上演示了我们的方法的有效性。由于我们的灰盒证书考虑了底层的图结构，所以我们可以通过应用图稀疏来显著提高可证明的健壮性。



## **42. Robust Detection of LLM-Generated Text: A Comparative Analysis**

LLM生成文本的稳健检测：比较分析 cs.CL

8 pages

**SubmitDate**: 2024-11-09    [abs](http://arxiv.org/abs/2411.06248v1) [paper-pdf](http://arxiv.org/pdf/2411.06248v1)

**Authors**: Yongye Su, Yuqing Wu

**Abstract**: The ability of large language models to generate complex texts allows them to be widely integrated into many aspects of life, and their output can quickly fill all network resources. As the impact of LLMs grows, it becomes increasingly important to develop powerful detectors for the generated text. This detector is essential to prevent the potential misuse of these technologies and to protect areas such as social media from the negative effects of false content generated by LLMS. The main goal of LLM-generated text detection is to determine whether text is generated by an LLM, which is a basic binary classification task. In our work, we mainly use three different classification methods based on open source datasets: traditional machine learning techniques such as logistic regression, k-means clustering, Gaussian Naive Bayes, support vector machines, and methods based on converters such as BERT, and finally algorithms that use LLMs to detect LLM-generated text. We focus on model generalization, potential adversarial attacks, and accuracy of model evaluation. Finally, the possible research direction in the future is proposed, and the current experimental results are summarized.

摘要: 大型语言模型生成复杂文本的能力使它们能够广泛融入生活的许多方面，它们的输出可以迅速填满所有网络资源。随着LLMS的影响越来越大，为生成的文本开发强大的检测器变得越来越重要。这种检测器对于防止这些技术的潜在滥用以及保护社交媒体等领域免受LLMS产生的虚假内容的负面影响至关重要。LLM生成的文本检测的主要目标是确定文本是否由LLM生成，这是一项基本的二进制分类任务。在我们的工作中，我们主要使用了三种不同的基于开源数据集的分类方法：传统的机器学习技术，如Logistic回归，k-均值聚类，高斯朴素贝叶斯，支持向量机，以及基于转换器的方法，如BERT，最后是使用LLMS来检测LLM生成的文本的算法。我们主要关注模型的泛化、潜在的敌意攻击和模型评估的准确性。最后，提出了未来可能的研究方向，并对目前的实验结果进行了总结。



## **43. BM-PAW: A Profitable Mining Attack in the PoW-based Blockchain System**

BM-PAW：基于PoW的区块链系统中的有利可图的采矿攻击 cs.CR

21 pages, 4 figures

**SubmitDate**: 2024-11-09    [abs](http://arxiv.org/abs/2411.06187v1) [paper-pdf](http://arxiv.org/pdf/2411.06187v1)

**Authors**: Junjie Hu, Xunzhi Chen, Huan Yan, Na Ruan

**Abstract**: Mining attacks enable an adversary to procure a disproportionately large portion of mining rewards by deviating from honest mining practices within the PoW-based blockchain system. In this paper, we demonstrate that the security vulnerabilities of PoW-based blockchain extend beyond what these mining attacks initially reveal. We introduce a novel mining strategy, named BM-PAW, which yields superior rewards for both the attacker and the targeted pool compared to the state-of-the-art mining attack: PAW. Our analysis reveals that BM-PAW attackers are incentivized to offer appropriate bribe money to other targets, as they comply with the attacker's directives upon receiving payment. We find the BM-PAW attacker can circumvent the "miner's dilemma" through equilibrium analysis in a two-pool BM-PAW game scenario, wherein the outcome is determined by the attacker's mining power. We finally propose practical countermeasures to mitigate these novel pool attacks.

摘要: 采矿攻击使对手能够通过偏离基于PoW的区块链系统内的诚实采矿实践来获得不成比例的大部分采矿奖励。在本文中，我们证明了基于PoW的区块链的安全漏洞超出了这些采矿攻击最初揭示的范围。我们引入了一种名为BM-PAW的新型采矿策略，与最先进的采矿攻击PAW相比，该策略为攻击者和目标池提供了更高的回报。我们的分析表明，BM-PAW攻击者受到激励向其他目标提供适当的贿赂资金，因为他们在收到付款后遵守攻击者的指示。我们发现，BM-PAW攻击者可以通过两池BM-PAW游戏场景中的均衡分析来规避“矿工困境”，其中结果取决于攻击者的采矿能力。我们最后提出了实用的对策来减轻这些新型池攻击。



## **44. AI-Compass: A Comprehensive and Effective Multi-module Testing Tool for AI Systems**

AI-Compass：一款全面有效的人工智能系统多模块测试工具 cs.AI

**SubmitDate**: 2024-11-09    [abs](http://arxiv.org/abs/2411.06146v1) [paper-pdf](http://arxiv.org/pdf/2411.06146v1)

**Authors**: Zhiyu Zhu, Zhibo Jin, Hongsheng Hu, Minhui Xue, Ruoxi Sun, Seyit Camtepe, Praveen Gauravaram, Huaming Chen

**Abstract**: AI systems, in particular with deep learning techniques, have demonstrated superior performance for various real-world applications. Given the need for tailored optimization in specific scenarios, as well as the concerns related to the exploits of subsurface vulnerabilities, a more comprehensive and in-depth testing AI system becomes a pivotal topic. We have seen the emergence of testing tools in real-world applications that aim to expand testing capabilities. However, they often concentrate on ad-hoc tasks, rendering them unsuitable for simultaneously testing multiple aspects or components. Furthermore, trustworthiness issues arising from adversarial attacks and the challenge of interpreting deep learning models pose new challenges for developing more comprehensive and in-depth AI system testing tools. In this study, we design and implement a testing tool, \tool, to comprehensively and effectively evaluate AI systems. The tool extensively assesses multiple measurements towards adversarial robustness, model interpretability, and performs neuron analysis. The feasibility of the proposed testing tool is thoroughly validated across various modalities, including image classification, object detection, and text classification. Extensive experiments demonstrate that \tool is the state-of-the-art tool for a comprehensive assessment of the robustness and trustworthiness of AI systems. Our research sheds light on a general solution for AI systems testing landscape.

摘要: 人工智能系统，特别是具有深度学习技术的系统，在各种现实世界的应用中表现出了优越的性能。鉴于在特定场景中需要量身定做的优化，以及与地下漏洞利用相关的担忧，更全面和深入的测试人工智能系统成为一个关键话题。我们已经看到，在真实世界的应用程序中出现了旨在扩展测试能力的测试工具。然而，它们通常专注于特别任务，使得它们不适合同时测试多个方面或组件。此外，对抗性攻击产生的可信性问题以及解释深度学习模型的挑战为开发更全面和深入的AI系统测试工具提出了新的挑战。在这项研究中，我们设计并实现了一个测试工具，\Tool，以全面有效地评估AI系统。该工具广泛评估对抗性稳健性、模型可解释性的多个测量，并执行神经元分析。所提出的测试工具的可行性在包括图像分类、目标检测和文本分类在内的各种模式上得到了彻底的验证。大量的实验表明，该工具是全面评估人工智能系统健壮性和可信性的最先进的工具。我们的研究为人工智能系统测试领域提供了一个通用的解决方案。



## **45. Robust Graph Neural Networks via Unbiased Aggregation**

通过无偏聚集的鲁棒图神经网络 cs.LG

NeurIPS 2024 poster. 28 pages, 14 figures

**SubmitDate**: 2024-11-09    [abs](http://arxiv.org/abs/2311.14934v2) [paper-pdf](http://arxiv.org/pdf/2311.14934v2)

**Authors**: Zhichao Hou, Ruiqi Feng, Tyler Derr, Xiaorui Liu

**Abstract**: The adversarial robustness of Graph Neural Networks (GNNs) has been questioned due to the false sense of security uncovered by strong adaptive attacks despite the existence of numerous defenses. In this work, we delve into the robustness analysis of representative robust GNNs and provide a unified robust estimation point of view to understand their robustness and limitations. Our novel analysis of estimation bias motivates the design of a robust and unbiased graph signal estimator. We then develop an efficient Quasi-Newton Iterative Reweighted Least Squares algorithm to solve the estimation problem, which is unfolded as robust unbiased aggregation layers in GNNs with theoretical guarantees. Our comprehensive experiments confirm the strong robustness of our proposed model under various scenarios, and the ablation study provides a deep understanding of its advantages. Our code is available at https://github.com/chris-hzc/RUNG.

摘要: 图形神经网络（GNN）的对抗鲁棒性受到质疑，因为尽管存在多种防御措施，但强自适应攻击却暴露了错误的安全感。在这项工作中，我们深入研究了代表性稳健GNN的稳健性分析，并提供统一的稳健性估计观点来了解其稳健性和局限性。我们对估计偏差的新颖分析激励了设计稳健且无偏的图信号估计器。然后，我们开发了一种高效的准牛顿迭代重加权最小平方算法来解决估计问题，该算法在理论保证的情况下被展开为GNN中的鲁棒无偏聚集层。我们全面的实验证实了我们提出的模型在各种场景下具有强大的鲁棒性，并且消融研究深入了解了其优势。我们的代码可在https://github.com/chris-hzc/RUNG上获取。



## **46. Goal-guided Generative Prompt Injection Attack on Large Language Models**

对大型语言模型的目标引导生成提示注入攻击 cs.CR

11 pages, 6 figures

**SubmitDate**: 2024-11-09    [abs](http://arxiv.org/abs/2404.07234v4) [paper-pdf](http://arxiv.org/pdf/2404.07234v4)

**Authors**: Chong Zhang, Mingyu Jin, Qinkai Yu, Chengzhi Liu, Haochen Xue, Xiaobo Jin

**Abstract**: Current large language models (LLMs) provide a strong foundation for large-scale user-oriented natural language tasks. A large number of users can easily inject adversarial text or instructions through the user interface, thus causing LLMs model security challenges. Although there is currently a large amount of research on prompt injection attacks, most of these black-box attacks use heuristic strategies. It is unclear how these heuristic strategies relate to the success rate of attacks and thus effectively improve model robustness. To solve this problem, we redefine the goal of the attack: to maximize the KL divergence between the conditional probabilities of the clean text and the adversarial text. Furthermore, we prove that maximizing the KL divergence is equivalent to maximizing the Mahalanobis distance between the embedded representation $x$ and $x'$ of the clean text and the adversarial text when the conditional probability is a Gaussian distribution and gives a quantitative relationship on $x$ and $x'$. Then we designed a simple and effective goal-guided generative prompt injection strategy (G2PIA) to find an injection text that satisfies specific constraints to achieve the optimal attack effect approximately. It is particularly noteworthy that our attack method is a query-free black-box attack method with low computational cost. Experimental results on seven LLM models and four datasets show the effectiveness of our attack method.

摘要: 现有的大型语言模型为大规模面向用户的自然语言任务提供了坚实的基础。大量用户可以很容易地通过用户界面注入敌意文本或指令，从而造成LLMS模型的安全挑战。虽然目前有大量关于即时注入攻击的研究，但这些黑盒攻击大多采用启发式策略。目前尚不清楚这些启发式策略如何与攻击成功率相关，从而有效地提高模型的稳健性。为了解决这个问题，我们重新定义了攻击的目标：最大化纯文本和敌意文本的条件概率之间的KL偏差。此外，我们证明了当条件概率为高斯分布时，最大化KL发散度等价于最大化明文和敌意文本的嵌入表示$x$和$x‘$之间的马氏距离，并给出了关于$x$和$x’$的定量关系。然后，设计了一种简单有效的目标引导生成性提示注入策略(G2PIA)，找到满足特定约束的注入文本，以近似达到最优的攻击效果。特别值得注意的是，我们的攻击方法是一种计算代价低的无查询黑盒攻击方法。在7个LLM模型和4个数据集上的实验结果表明了该攻击方法的有效性。



## **47. Towards More Realistic Extraction Attacks: An Adversarial Perspective**

走向更真实的提取攻击：对抗的角度 cs.CR

Presented at PrivateNLP@ACL2024

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2407.02596v2) [paper-pdf](http://arxiv.org/pdf/2407.02596v2)

**Authors**: Yash More, Prakhar Ganesh, Golnoosh Farnadi

**Abstract**: Language models are prone to memorizing parts of their training data which makes them vulnerable to extraction attacks. Existing research often examines isolated setups--such as evaluating extraction risks from a single model or with a fixed prompt design. However, a real-world adversary could access models across various sizes and checkpoints, as well as exploit prompt sensitivity, resulting in a considerably larger attack surface than previously studied. In this paper, we revisit extraction attacks from an adversarial perspective, focusing on how to leverage the brittleness of language models and the multi-faceted access to the underlying data. We find significant churn in extraction trends, i.e., even unintuitive changes to the prompt, or targeting smaller models and earlier checkpoints, can extract distinct information. By combining information from multiple attacks, our adversary is able to increase the extraction risks by up to $2 \times$. Furthermore, even with mitigation strategies like data deduplication, we find the same escalation of extraction risks against a real-world adversary. We conclude with a set of case studies, including detecting pre-training data, copyright violations, and extracting personally identifiable information, showing how our more realistic adversary can outperform existing adversaries in the literature.

摘要: 语言模型容易记住其训练数据的一部分，这使得它们容易受到提取攻击。现有的研究经常考察孤立的设置--例如评估从单一模型或固定提示设计中提取的风险。然而，现实世界中的对手可以访问不同大小和检查点的模型，并利用即时敏感性，导致比之前研究的更大的攻击面。在本文中，我们从敌意的角度重新审视提取攻击，重点放在如何利用语言模型的脆弱性和对底层数据的多方面访问。我们发现提取趋势中存在显著的波动，即即使对提示进行了不直观的更改，或者针对较小的模型和较早的检查点，也可以提取不同的信息。通过组合来自多个攻击的信息，我们的对手能够将提取风险增加高达$2\x$。此外，即使使用重复数据删除等缓解策略，我们也发现现实世界中的对手面临同样的提取风险升级。我们以一组案例研究结束，包括检测训练前数据、侵犯版权和提取个人身份信息，展示我们更现实的对手如何超越文献中现有的对手。



## **48. A Survey of AI-Related Cyber Security Risks and Countermeasures in Mobility-as-a-Service**

移动即服务中人工智能相关网络安全风险及对策调查 cs.CR

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2411.05681v1) [paper-pdf](http://arxiv.org/pdf/2411.05681v1)

**Authors**: Kai-Fung Chu, Haiyue Yuan, Jinsheng Yuan, Weisi Guo, Nazmiye Balta-Ozkan, Shujun Li

**Abstract**: Mobility-as-a-Service (MaaS) integrates different transport modalities and can support more personalisation of travellers' journey planning based on their individual preferences, behaviours and wishes. To fully achieve the potential of MaaS, a range of AI (including machine learning and data mining) algorithms are needed to learn personal requirements and needs, to optimise journey planning of each traveller and all travellers as a whole, to help transport service operators and relevant governmental bodies to operate and plan their services, and to detect and prevent cyber attacks from various threat actors including dishonest and malicious travellers and transport operators. The increasing use of different AI and data processing algorithms in both centralised and distributed settings opens the MaaS ecosystem up to diverse cyber and privacy attacks at both the AI algorithm level and the connectivity surfaces. In this paper, we present the first comprehensive review on the coupling between AI-driven MaaS design and the diverse cyber security challenges related to cyber attacks and countermeasures. In particular, we focus on how current and emerging AI-facilitated privacy risks (profiling, inference, and third-party threats) and adversarial AI attacks (evasion, extraction, and gamification) may impact the MaaS ecosystem. These risks often combine novel attacks (e.g., inverse learning) with traditional attack vectors (e.g., man-in-the-middle attacks), exacerbating the risks for the wider participation actors and the emergence of new business models.

摘要: 移动即服务(MaAS)集成了不同的交通方式，可以根据旅行者的个人偏好、行为和意愿支持更个性化的旅行计划。为了充分发挥MAAS的潜力，需要一系列人工智能(包括机器学习和数据挖掘)算法来了解个人需求和需求，优化每个旅行者和所有旅行者的旅行计划，帮助运输服务运营商和相关政府机构运营和规划他们的服务，以及检测和防止来自各种威胁参与者的网络攻击，包括不诚实和恶意的旅行者和运输运营商。在集中式和分布式环境中越来越多地使用不同的人工智能和数据处理算法，使MAAS生态系统在人工智能算法级别和连接面上都面临不同的网络和隐私攻击。在这篇文章中，我们首次全面回顾了人工智能驱动的MAAS设计与与网络攻击相关的各种网络安全挑战和对策之间的耦合。特别是，我们关注当前和正在出现的人工智能促进的隐私风险(剖析、推理和第三方威胁)和对抗性人工智能攻击(逃避、提取和游戏化)可能如何影响MAAS生态系统。这些风险通常将新的攻击(例如反向学习)与传统的攻击向量(例如中间人攻击)结合在一起，加剧了更广泛的参与者和新商业模式出现的风险。



## **49. DeepDRK: Deep Dependency Regularized Knockoff for Feature Selection**

DeepDRK：功能选择的深度依赖正规化仿制品 cs.LG

33 pages, 15 figures, 9 tables

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2402.17176v2) [paper-pdf](http://arxiv.org/pdf/2402.17176v2)

**Authors**: Hongyu Shen, Yici Yan, Zhizhen Zhao

**Abstract**: Model-X knockoff has garnered significant attention among various feature selection methods due to its guarantees for controlling the false discovery rate (FDR). Since its introduction in parametric design, knockoff techniques have evolved to handle arbitrary data distributions using deep learning-based generative models. However, we have observed limitations in the current implementations of the deep Model-X knockoff framework. Notably, the "swap property" that knockoffs require often faces challenges at the sample level, resulting in diminished selection power. To address these issues, we develop "Deep Dependency Regularized Knockoff (DeepDRK)," a distribution-free deep learning method that effectively balances FDR and power. In DeepDRK, we introduce a novel formulation of the knockoff model as a learning problem under multi-source adversarial attacks. By employing an innovative perturbation technique, we achieve lower FDR and higher power. Our model outperforms existing benchmarks across synthetic, semi-synthetic, and real-world datasets, particularly when sample sizes are small and data distributions are non-Gaussian.

摘要: 在各种特征选择方法中，Model-X假冒因其在控制错误发现率(FDR)方面的保证而受到广泛关注。自从它被引入到参数设计中以来，仿冒技术已经发展到使用基于深度学习的生成性模型来处理任意数据分布。然而，我们观察到深度Model-X仿冒框架的当前实现存在局限性。值得注意的是，仿冒品所需的“互换属性”经常在样本层面面临挑战，导致选择能力减弱。为了解决这些问题，我们开发了“深度依赖正规化仿冒(DeepDRK)”，这是一种无需分发的深度学习方法，可以有效地平衡FDR和功率。在DeepDRK中，我们引入了一种新的仿冒模型作为多源攻击下的学习问题。通过采用创新的微扰技术，我们实现了更低的FDR和更高的功率。我们的模型在合成、半合成和真实世界数据集上的表现优于现有基准，特别是在样本量较小且数据分布为非高斯的情况下。



## **50. Towards a Re-evaluation of Data Forging Attacks in Practice**

重新评估实践中的数据伪造攻击 cs.CR

18 pages

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2411.05658v1) [paper-pdf](http://arxiv.org/pdf/2411.05658v1)

**Authors**: Mohamed Suliman, Anisa Halimi, Swanand Kadhe, Nathalie Baracaldo, Douglas Leith

**Abstract**: Data forging attacks provide counterfactual proof that a model was trained on a given dataset, when in fact, it was trained on another. These attacks work by forging (replacing) mini-batches with ones containing distinct training examples that produce nearly identical gradients. Data forging appears to break any potential avenues for data governance, as adversarial model owners may forge their training set from a dataset that is not compliant to one that is. Given these serious implications on data auditing and compliance, we critically analyse data forging from both a practical and theoretical point of view, finding that a key practical limitation of current attack methods makes them easily detectable by a verifier; namely that they cannot produce sufficiently identical gradients. Theoretically, we analyse the question of whether two distinct mini-batches can produce the same gradient. Generally, we find that while there may exist an infinite number of distinct mini-batches with real-valued training examples and labels that produce the same gradient, finding those that are within the allowed domain e.g. pixel values between 0-255 and one hot labels is a non trivial task. Our results call for the reevaluation of the strength of existing attacks, and for additional research into successful data forging, given the serious consequences it may have on machine learning and privacy.

摘要: 数据伪造攻击提供了反事实的证据，证明一个模型是在给定的数据集上训练的，而实际上，它是在另一个数据集上训练的。这些攻击的工作原理是用包含不同训练样本的小批次来伪造(替换)小批次，这些训练样本产生几乎相同的梯度。数据伪造似乎打破了数据治理的任何潜在途径，因为对抗性模型所有者可能会从与之不符的数据集伪造他们的训练集。鉴于这些对数据审计和合规性的严重影响，我们从实践和理论的角度对数据伪造进行了批判性分析，发现当前攻击方法的一个关键实际限制使它们很容易被验证者检测到；即它们不能产生足够相同的梯度。理论上，我们分析了两个不同的小批次是否可以产生相同的梯度的问题。通常，我们发现虽然可能存在无限数量的不同的小批次，其实值训练样本和标签产生相同的梯度，但找到那些在允许的域内的小批次，例如，像素值在0-255和一个热点标签之间是一项不平凡的任务。我们的结果要求重新评估现有攻击的强度，并考虑到它可能对机器学习和隐私造成的严重后果，对成功的数据伪造进行更多研究。



