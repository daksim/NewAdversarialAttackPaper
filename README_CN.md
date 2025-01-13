# Latest Adversarial Attack Papers
**update at 2025-01-13 10:31:33**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Privacy-Preserving Distributed Defense Framework for DC Microgrids Against Exponentially Unbounded False Data Injection Attacks**

针对指数无界虚假数据注入攻击的DC微电网保护隐私分布式防御框架 eess.SY

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.00588v2) [paper-pdf](http://arxiv.org/pdf/2501.00588v2)

**Authors**: Yi Zhang, Mohamadamin Rajabinezhad, Yichao Wang, Junbo Zhao, Shan Zuo

**Abstract**: This paper introduces a novel, fully distributed control framework for DC microgrids, enhancing resilience against exponentially unbounded false data injection (EU-FDI) attacks. Our framework features a consensus-based secondary control for each converter, effectively addressing these advanced threats. To further safeguard sensitive operational data, a privacy-preserving mechanism is incorporated into the control design, ensuring that critical information remains secure even under adversarial conditions. Rigorous Lyapunov stability analysis confirms the framework's ability to maintain critical DC microgrid operations like voltage regulation and load sharing under EU-FDI threats. The framework's practicality is validated through hardware-in-the-loop experiments, demonstrating its enhanced resilience and robust privacy protection against the complex challenges posed by quick variant FDI attacks.

摘要: 本文介绍了一种新颖的、完全分布式的DC微电网控制框架，增强了抵御指数无界虚假数据注入（EU-Direct）攻击的弹性。我们的框架为每个转换器提供了基于共识的二级控制，可以有效地解决这些高级威胁。为了进一步保护敏感的运营数据，控制设计中纳入了隐私保护机制，确保关键信息即使在敌对条件下也保持安全。严格的李亚普诺夫稳定性分析证实了该框架在欧盟外国直接投资威胁下维持关键的直流微电网运营的能力，例如电压调节和负载共享。该框架的实用性通过硬件在环实验得到了验证，展示了其增强的弹性和强大的隐私保护，以应对快速变体外国直接投资攻击带来的复杂挑战。



## **2. Pixel Is Not A Barrier: An Effective Evasion Attack for Pixel-Domain Diffusion Models**

像素不是障碍：像素域扩散模型的有效规避攻击 cs.CV

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2408.11810v2) [paper-pdf](http://arxiv.org/pdf/2408.11810v2)

**Authors**: Chun-Yen Shih, Li-Xuan Peng, Jia-Wei Liao, Ernie Chu, Cheng-Fu Chou, Jun-Cheng Chen

**Abstract**: Diffusion Models have emerged as powerful generative models for high-quality image synthesis, with many subsequent image editing techniques based on them. However, the ease of text-based image editing introduces significant risks, such as malicious editing for scams or intellectual property infringement. Previous works have attempted to safeguard images from diffusion-based editing by adding imperceptible perturbations. These methods are costly and specifically target prevalent Latent Diffusion Models (LDMs), while Pixel-domain Diffusion Models (PDMs) remain largely unexplored and robust against such attacks. Our work addresses this gap by proposing a novel attack framework, AtkPDM. AtkPDM is mainly composed of a feature representation attacking loss that exploits vulnerabilities in denoising UNets and a latent optimization strategy to enhance the naturalness of adversarial images. Extensive experiments demonstrate the effectiveness of our approach in attacking dominant PDM-based editing methods (e.g., SDEdit) while maintaining reasonable fidelity and robustness against common defense methods. Additionally, our framework is extensible to LDMs, achieving comparable performance to existing approaches.

摘要: 扩散模型已经成为高质量图像合成的强大生成性模型，许多后续的图像编辑技术都是基于扩散模型的。然而，基于文本的图像编辑的简便性带来了重大风险，例如用于欺诈或侵犯知识产权的恶意编辑。以前的工作试图通过添加不可察觉的扰动来保护图像免受基于扩散的编辑。这些方法昂贵且专门针对流行的潜在扩散模型(LDM)，而像素域扩散模型(PDMS)在很大程度上仍未被探索，并且对此类攻击具有健壮性。我们的工作通过提出一种新颖的攻击框架AtkPDM来解决这一问题。该算法主要由特征表示、攻击损失和潜在优化策略两部分组成，前者利用UNNet的去噪漏洞，后者增强敌方图像的自然度。大量实验表明，该方法在攻击主流的基于产品数据管理的编辑方法(如SDEDIT)的同时，对常见的防御方法保持了合理的保真度和健壮性。此外，我们的框架可扩展到LDM，实现了与现有方法相当的性能。



## **3. Adversarial Detection by Approximation of Ensemble Boundary**

利用集合边界逼近的对抗检测 cs.LG

27 pages, 7 figures, 5 tables

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2211.10227v5) [paper-pdf](http://arxiv.org/pdf/2211.10227v5)

**Authors**: T. Windeatt

**Abstract**: Despite being effective in many application areas, Deep Neural Networks (DNNs) are vulnerable to being attacked. In object recognition, the attack takes the form of a small perturbation added to an image, that causes the DNN to misclassify, but to a human appears no different. Adversarial attacks lead to defences that are themselves subject to attack, and the attack/ defence strategies provide important information about the properties of DNNs. In this paper, a novel method of detecting adversarial attacks is proposed for an ensemble of Deep Neural Networks (DNNs) solving two-class pattern recognition problems. The ensemble is combined using Walsh coefficients which are capable of approximating Boolean functions and thereby controlling the decision boundary complexity. The hypothesis in this paper is that decision boundaries with high curvature allow adversarial perturbations to be found, but change the curvature of the decision boundary, which is then approximated in a different way by Walsh coefficients compared to the clean images. Besides controlling boundary complexity, the coefficients also measure the correlation with class labels, which may aid in understanding the learning and transferability properties of DNNs. While the experiments here use images, the proposed approach of modelling two-class ensemble decision boundaries could in principle be applied to any application area.

摘要: 尽管深度神经网络(DNN)在许多应用领域都很有效，但它很容易受到攻击。在目标识别中，攻击的形式是添加到图像上的小扰动，这会导致DNN错误分类，但对人类来说似乎没有什么不同。对抗性攻击导致的防御本身也会受到攻击，而攻击/防御策略提供了有关DNN属性的重要信息。针对解决两类模式识别问题的深度神经网络(DNN)集成问题，提出了一种检测敌意攻击的新方法。该集成使用沃尔什系数进行组合，沃尔什系数能够逼近布尔函数，从而控制决策边界的复杂性。本文的假设是，高曲率的决策边界允许发现对抗性扰动，但改变了决策边界的曲率，然后用沃尔什系数以不同的方式逼近决策边界，与干净的图像相比。除了控制边界复杂度外，该系数还度量了与类别标签的相关性，这有助于理解DNN的学习和迁移特性。虽然这里的实验使用的是图像，但所提出的建模两类集合决策边界的方法原则上可以应用于任何应用领域。



## **4. Beyond Optimal Fault Tolerance**

超越最佳故障容忍度 cs.DC

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.06044v1) [paper-pdf](http://arxiv.org/pdf/2501.06044v1)

**Authors**: Andrew Lewis-Pye, Tim Roughgarden

**Abstract**: The optimal fault-tolerance achievable by any protocol has been characterized in a wide range of settings. For example, for state machine replication (SMR) protocols operating in the partially synchronous setting, it is possible to simultaneously guarantee consistency against $\alpha$-bounded adversaries (i.e., adversaries that control less than an $\alpha$ fraction of the participants) and liveness against $\beta$-bounded adversaries if and only if $\alpha + 2\beta \leq 1$.   This paper characterizes to what extent "better-than-optimal" fault-tolerance guarantees are possible for SMR protocols when the standard consistency requirement is relaxed to allow a bounded number $r$ of consistency violations. We prove that bounding rollback is impossible without additional timing assumptions and investigate protocols that tolerate and recover from consistency violations whenever message delays around the time of an attack are bounded by a parameter $\Delta^*$ (which may be arbitrarily larger than the parameter $\Delta$ that bounds post-GST message delays in the partially synchronous model). Here, a protocol's fault-tolerance can be a non-constant function of $r$, and we prove, for each $r$, matching upper and lower bounds on the optimal ``recoverable fault-tolerance'' achievable by any SMR protocol. For example, for protocols that guarantee liveness against 1/3-bounded adversaries in the partially synchronous setting, a 5/9-bounded adversary can always cause one consistency violation but not two, and a 2/3-bounded adversary can always cause two consistency violations but not three. Our positive results are achieved through a generic ``recovery procedure'' that can be grafted on to any accountable SMR protocol and restores consistency following a violation while rolling back only transactions that were finalized in the previous $2\Delta^*$ timesteps.

摘要: 任何协议可实现的最佳容错已在广泛的设置中得到了表征。例如，对于在部分同步设置中操作的状态机复制(SMR)协议，当且仅当$\Alpha+2\Beta\leq 1$时，可以同时保证针对$\Alpha$受限的对手(即，控制少于$\Alpha$部分参与者的对手)的一致性和针对$\beta$受限的攻击者的活性。本文刻画了当标准一致性要求被放宽以允许有限数量的一致性违规时，SMR协议在多大程度上可能获得比最优更好的容错保证。我们证明了如果没有额外的时间假设，绑定回滚是不可能的，并研究了当攻击时间附近的消息延迟由参数$\Delta^*$(该参数可以任意大于在部分同步模型中限制GST后消息延迟的参数$\Delta$)限定时，容忍一致性违规并从一致性违规中恢复的协议。这里，协议的容错性可以是$r$的非常数函数，并且我们证明了，对于每个$r$，任何SMR协议都可以达到最优“可恢复容错性”的上下界匹配。例如，对于在部分同步设置中保证对1/3有界攻击者的活跃性的协议，5/9有界的攻击者总是可以导致一次一致性违反而不是两次，而2/3有界的攻击者总是可以引起两次一致性违反而不是三次一致性违反。我们的积极结果是通过可嫁接到任何负责任的SMR协议上并在违规后恢复一致性，同时仅回滚在前$2\Delta^*$时间步长中完成的事务的通用“恢复程序”实现的。



## **5. Effective faking of verbal deception detection with target-aligned adversarial attacks**

通过目标对准的对抗攻击有效伪造言语欺骗检测 cs.CL

preprint

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.05962v1) [paper-pdf](http://arxiv.org/pdf/2501.05962v1)

**Authors**: Bennett Kleinberg, Riccardo Loconte, Bruno Verschuere

**Abstract**: Background: Deception detection through analysing language is a promising avenue using both human judgments and automated machine learning judgments. For both forms of credibility assessment, automated adversarial attacks that rewrite deceptive statements to appear truthful pose a serious threat. Methods: We used a dataset of 243 truthful and 262 fabricated autobiographical stories in a deception detection task for humans and machine learning models. A large language model was tasked to rewrite deceptive statements so that they appear truthful. In Study 1, humans who made a deception judgment or used the detailedness heuristic and two machine learning models (a fine-tuned language model and a simple n-gram model) judged original or adversarial modifications of deceptive statements. In Study 2, we manipulated the target alignment of the modifications, i.e. tailoring the attack to whether the statements would be assessed by humans or computer models. Results: When adversarial modifications were aligned with their target, human (d=-0.07 and d=-0.04) and machine judgments (51% accuracy) dropped to the chance level. When the attack was not aligned with the target, both human heuristics judgments (d=0.30 and d=0.36) and machine learning predictions (63-78%) were significantly better than chance. Conclusions: Easily accessible language models can effectively help anyone fake deception detection efforts both by humans and machine learning models. Robustness against adversarial modifications for humans and machines depends on that target alignment. We close with suggestions on advancing deception research with adversarial attack designs.

摘要: 背景：通过分析语言进行欺骗检测是一种既使用人类判断又使用自动机器学习判断的有前途的方法。对于这两种形式的可信度评估来说，重写欺骗性陈述以使其看起来真实的自动对抗性攻击构成了严重威胁。方法：我们使用了243个真实的和262个编造的自传故事的数据集，在人类和机器学习模型的欺骗检测任务中。一个大型语言模型的任务是重写欺骗性的陈述，使它们看起来是真实的。在研究1中，做出欺骗性判断或使用细节启发式和两个机器学习模型(微调语言模型和简单n元语法模型)的人判断欺骗性陈述的原始修改或对抗性修改。在研究2中，我们操纵了修改的目标对齐，即根据陈述是否由人或计算机模型评估来量身定做攻击。结果：当对抗性修改与他们的目标一致时，人类(d=-0.07和d=-0.04)和机器判断(51%准确率)下降到机会水平。当攻击与目标不一致时，人类的启发式判断(d=0.30和d=0.36)和机器学习预测(63%-78%)都显著好于机会。结论：易于理解的语言模型可以有效地帮助任何人通过人类和机器学习模型进行虚假的欺骗检测。对人类和机器的敌意修改的健壮性取决于目标对齐。最后，我们建议用对抗性攻击设计来推进欺骗研究。



## **6. Towards Backdoor Stealthiness in Model Parameter Space**

模型参数空间中的后门隐秘性 cs.CR

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.05928v1) [paper-pdf](http://arxiv.org/pdf/2501.05928v1)

**Authors**: Xiaoyun Xu, Zhuoran Liu, Stefanos Koffas, Stjepan Picek

**Abstract**: Recent research on backdoor stealthiness focuses mainly on indistinguishable triggers in input space and inseparable backdoor representations in feature space, aiming to circumvent backdoor defenses that examine these respective spaces. However, existing backdoor attacks are typically designed to resist a specific type of backdoor defense without considering the diverse range of defense mechanisms. Based on this observation, we pose a natural question: Are current backdoor attacks truly a real-world threat when facing diverse practical defenses?   To answer this question, we examine 12 common backdoor attacks that focus on input-space or feature-space stealthiness and 17 diverse representative defenses. Surprisingly, we reveal a critical blind spot: Backdoor attacks designed to be stealthy in input and feature spaces can be mitigated by examining backdoored models in parameter space. To investigate the underlying causes behind this common vulnerability, we study the characteristics of backdoor attacks in the parameter space. Notably, we find that input- and feature-space attacks introduce prominent backdoor-related neurons in parameter space, which are not thoroughly considered by current backdoor attacks. Taking comprehensive stealthiness into account, we propose a novel supply-chain attack called Grond. Grond limits the parameter changes by a simple yet effective module, Adversarial Backdoor Injection (ABI), which adaptively increases the parameter-space stealthiness during the backdoor injection. Extensive experiments demonstrate that Grond outperforms all 12 backdoor attacks against state-of-the-art (including adaptive) defenses on CIFAR-10, GTSRB, and a subset of ImageNet. In addition, we show that ABI consistently improves the effectiveness of common backdoor attacks.

摘要: 目前关于后门隐蔽性的研究主要集中在输入空间中不可区分的触发器和特征空间中不可分的后门表示上，目的是绕过检查这些空间的后门防御。然而，现有的后门攻击通常被设计为抵抗特定类型的后门防御，而没有考虑到各种防御机制。基于这一观察，我们提出了一个自然的问题：当面临各种实际防御时，当前的后门攻击真的是现实世界的威胁吗？为了回答这个问题，我们研究了12种常见的后门攻击，这些攻击侧重于输入空间或功能空间的隐蔽性，以及17种不同的代表性防御。令人惊讶的是，我们揭示了一个关键的盲点：设计为在输入和特征空间中隐蔽的后门攻击可以通过检查参数空间中的后置模型来缓解。为了研究这种常见漏洞背后的潜在原因，我们研究了参数空间中的后门攻击的特征。值得注意的是，我们发现输入和特征空间攻击在参数空间中引入了显著的后门相关神经元，而目前的后门攻击并没有完全考虑到这一点。在综合考虑隐蔽性的基础上，提出了一种新的供应链攻击方法Grond。Grond通过一个简单而有效的模块--对抗性后门注入(ABI)来限制参数变化，该模块在后门注入过程中自适应地增加参数空间的隐蔽性。广泛的实验表明，Grond在CIFAR-10、GTSRB和ImageNet的子集上对最先进的(包括自适应)防御系统的攻击能力超过了所有12个后门攻击。此外，我们还证明了ABI一贯提高了常见后门攻击的有效性。



## **7. Backdoor Attacks against No-Reference Image Quality Assessment Models via a Scalable Trigger**

通过可扩展触发器对无参考图像质量评估模型进行后门攻击 cs.CV

Accept by AAAI 2025

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2412.07277v2) [paper-pdf](http://arxiv.org/pdf/2412.07277v2)

**Authors**: Yi Yu, Song Xia, Xun Lin, Wenhan Yang, Shijian Lu, Yap-peng Tan, Alex Kot

**Abstract**: No-Reference Image Quality Assessment (NR-IQA), responsible for assessing the quality of a single input image without using any reference, plays a critical role in evaluating and optimizing computer vision systems, e.g., low-light enhancement. Recent research indicates that NR-IQA models are susceptible to adversarial attacks, which can significantly alter predicted scores with visually imperceptible perturbations. Despite revealing vulnerabilities, these attack methods have limitations, including high computational demands, untargeted manipulation, limited practical utility in white-box scenarios, and reduced effectiveness in black-box scenarios. To address these challenges, we shift our focus to another significant threat and present a novel poisoning-based backdoor attack against NR-IQA (BAIQA), allowing the attacker to manipulate the IQA model's output to any desired target value by simply adjusting a scaling coefficient $\alpha$ for the trigger. We propose to inject the trigger in the discrete cosine transform (DCT) domain to improve the local invariance of the trigger for countering trigger diminishment in NR-IQA models due to widely adopted data augmentations. Furthermore, the universal adversarial perturbations (UAP) in the DCT space are designed as the trigger, to increase IQA model susceptibility to manipulation and improve attack effectiveness. In addition to the heuristic method for poison-label BAIQA (P-BAIQA), we explore the design of clean-label BAIQA (C-BAIQA), focusing on $\alpha$ sampling and image data refinement, driven by theoretical insights we reveal. Extensive experiments on diverse datasets and various NR-IQA models demonstrate the effectiveness of our attacks. Code can be found at https://github.com/yuyi-sd/BAIQA.

摘要: 无参考图像质量评估(NR-IQA)负责在不使用任何参考图像的情况下评估单个输入图像的质量，在评估和优化计算机视觉系统(如微光增强)中起着至关重要的作用。最近的研究表明，NR-IQA模型容易受到对抗性攻击，这种攻击会在视觉上不可察觉的扰动下显著改变预测分数。尽管暴露出漏洞，但这些攻击方法都有局限性，包括计算要求高、无针对性操作、在白盒场景中实际效用有限，以及在黑盒场景中有效性降低。为了应对这些挑战，我们将重点转移到另一个重要的威胁上，并提出了一种针对NR-IQA的基于中毒的后门攻击(BAIQA)，允许攻击者通过简单地调整触发器的缩放系数$\α$来操纵IQA模型的输出到任何期望的目标值。我们提出在离散余弦变换(DCT)域中注入触发器以改善触发器的局部不变性，以对抗由于广泛采用的数据增强而导致的NR-IQA模型中的触发器衰减。此外，设计了DCT空间中的通用对抗摄动(UAP)作为触发器，以增加IQA模型对操纵的敏感度，提高攻击效率。除了有毒标签BAIQA的启发式方法(P-BAIQA)外，我们还探索了清洁标签BAIQA(C-BAIQA)的设计，重点是$\α$采样和图像数据精化，这是我们揭示的理论见解的驱动。在不同的数据集和不同的NR-IQA模型上的大量实验证明了我们的攻击的有效性。代码可在https://github.com/yuyi-sd/BAIQA.上找到



## **8. Image-based Multimodal Models as Intruders: Transferable Multimodal Attacks on Video-based MLLMs**

作为入侵者的基于图像的多模式模型：对基于视频的MLLM的可转移多模式攻击 cs.CV

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.01042v2) [paper-pdf](http://arxiv.org/pdf/2501.01042v2)

**Authors**: Linhao Huang, Xue Jiang, Zhiqiang Wang, Wentao Mo, Xi Xiao, Bo Han, Yongjie Yin, Feng Zheng

**Abstract**: Video-based multimodal large language models (V-MLLMs) have shown vulnerability to adversarial examples in video-text multimodal tasks. However, the transferability of adversarial videos to unseen models--a common and practical real world scenario--remains unexplored. In this paper, we pioneer an investigation into the transferability of adversarial video samples across V-MLLMs. We find that existing adversarial attack methods face significant limitations when applied in black-box settings for V-MLLMs, which we attribute to the following shortcomings: (1) lacking generalization in perturbing video features, (2) focusing only on sparse key-frames, and (3) failing to integrate multimodal information. To address these limitations and deepen the understanding of V-MLLM vulnerabilities in black-box scenarios, we introduce the Image-to-Video MLLM (I2V-MLLM) attack. In I2V-MLLM, we utilize an image-based multimodal model (IMM) as a surrogate model to craft adversarial video samples. Multimodal interactions and temporal information are integrated to disrupt video representations within the latent space, improving adversarial transferability. In addition, a perturbation propagation technique is introduced to handle different unknown frame sampling strategies. Experimental results demonstrate that our method can generate adversarial examples that exhibit strong transferability across different V-MLLMs on multiple video-text multimodal tasks. Compared to white-box attacks on these models, our black-box attacks (using BLIP-2 as surrogate model) achieve competitive performance, with average attack success rates of 55.48% on MSVD-QA and 58.26% on MSRVTT-QA for VideoQA tasks, respectively. Our code will be released upon acceptance.

摘要: 基于视频的多通道大语言模型(V-MLLM)在视频-文本多通道任务中表现出对敌意例子的脆弱性。然而，对抗性视频是否可以转移到看不见的模型上--这是现实世界中常见和实用的场景--仍未得到探索。在本文中，我们率先对对抗性视频样本在V-MLLMS上的可转移性进行了研究。我们发现，现有的对抗性攻击方法在应用于V-MLLMS的黑盒环境时面临着很大的局限性，我们将其归因于以下缺点：(1)对扰动视频特征缺乏泛化；(2)只关注稀疏关键帧；(3)未能整合多模信息。为了解决这些限制并加深对黑盒场景中V-MLLM漏洞的理解，我们引入了图像到视频MLLM(I2V-MLLM)攻击。在I2V-MLLM中，我们使用基于图像的多模式模型(IMM)作为代理模型来制作对抗性视频样本。多模式交互和时间信息被集成以扰乱潜在空间内的视频表示，提高了对抗性转移。此外，还引入了扰动传播技术来处理不同的未知帧采样策略。实验结果表明，该方法能够在多个视频-文本多模式任务的不同V-MLLMS之间生成具有较强可转移性的对抗性实例。与这些模型上的白盒攻击相比，我们的黑盒攻击(以BLIP-2为代理模型)取得了与之相当的性能，对于视频QA任务，MSVD-QA和MSRVTT-QA的平均攻击成功率分别为55.48%和58.26%。我们的代码将在接受后发布。



## **9. ActMiner: Applying Causality Tracking and Increment Aligning for Graph-based Cyber Threat Hunting**

ActMiner：应用因果关系跟踪和增量对齐用于基于图形的网络威胁狩猎 cs.CR

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.05793v1) [paper-pdf](http://arxiv.org/pdf/2501.05793v1)

**Authors**: Mingjun Ma, Tiantian Zhu, Tieming Chen, Shuang Li, Jie Ying, Chunlin Xiong, Mingqi Lv, Yan Chen

**Abstract**: To defend against Advanced Persistent Threats on the endpoint, threat hunting employs security knowledge such as cyber threat intelligence to continuously analyze system audit logs through retrospective scanning, querying, or pattern matching, aiming to uncover attack patterns/graphs that traditional detection methods (e.g., recognition for Point of Interest) fail to capture. However, existing threat hunting systems based on provenance graphs face challenges of high false negatives, high false positives, and low efficiency when confronted with diverse attack tactics and voluminous audit logs. To address these issues, we propose a system called Actminer, which constructs query graphs from descriptive relationships in cyber threat intelligence reports for precise threat hunting (i.e., graph alignment) on provenance graphs. First, we present a heuristic search strategy based on equivalent semantic transfer to reduce false negatives. Second, we establish a filtering mechanism based on causal relationships of attack behaviors to mitigate false positives. Finally, we design a tree structure to incrementally update the alignment results, significantly improving hunting efficiency. Evaluation on the DARPA Engagement dataset demonstrates that compared to the SOTA POIROT, Actminer reduces false positives by 39.1%, eliminates all false negatives, and effectively counters adversarial attacks.

摘要: 为了防御终端上的高级持续威胁，威胁搜索利用网络威胁情报等安全知识，通过回溯扫描、查询或模式匹配来持续分析系统审核日志，旨在发现传统检测方法(例如兴趣点识别)无法捕获的攻击模式/图表。然而，现有的基于源图的威胁追捕系统在面对多样化的攻击策略和海量的审计日志时，面临着高误判、高误报、低效率的挑战。为了解决这些问题，我们提出了一个称为Actminer的系统，它根据网络威胁情报报告中的描述关系构建查询图，以便在来源图上进行精确的威胁搜索(即图对齐)。首先，我们提出了一种基于等价语义转移的启发式搜索策略来减少漏报。其次，我们建立了一种基于攻击行为因果关系的过滤机制来减少误报。最后，我们设计了一种树结构来增量更新比对结果，显著提高了狩猎效率。对DARPA参与数据集的评估表明，与Sota Poirow相比，Actminer减少了39.1%的误报，消除了所有的漏报，并有效地对抗了对手攻击。



## **10. UV-Attack: Physical-World Adversarial Attacks for Person Detection via Dynamic-NeRF-based UV Mapping**

紫外线攻击：通过基于动态NeRF的紫外线映射进行人员检测的物理世界对抗性攻击 cs.CV

23 pages, 22 figures, submitted to ICLR2025

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.05783v1) [paper-pdf](http://arxiv.org/pdf/2501.05783v1)

**Authors**: Yanjie Li, Wenxuan Zhang, Kaisheng Liang, Bin Xiao

**Abstract**: In recent research, adversarial attacks on person detectors using patches or static 3D model-based texture modifications have struggled with low success rates due to the flexible nature of human movement. Modeling the 3D deformations caused by various actions has been a major challenge. Fortunately, advancements in Neural Radiance Fields (NeRF) for dynamic human modeling offer new possibilities. In this paper, we introduce UV-Attack, a groundbreaking approach that achieves high success rates even with extensive and unseen human actions. We address the challenge above by leveraging dynamic-NeRF-based UV mapping. UV-Attack can generate human images across diverse actions and viewpoints, and even create novel actions by sampling from the SMPL parameter space. While dynamic NeRF models are capable of modeling human bodies, modifying clothing textures is challenging because they are embedded in neural network parameters. To tackle this, UV-Attack generates UV maps instead of RGB images and modifies the texture stacks. This approach enables real-time texture edits and makes the attack more practical. We also propose a novel Expectation over Pose Transformation loss (EoPT) to improve the evasion success rate on unseen poses and views. Our experiments show that UV-Attack achieves a 92.75% attack success rate against the FastRCNN model across varied poses in dynamic video settings, significantly outperforming the state-of-the-art AdvCamou attack, which only had a 28.50% ASR. Moreover, we achieve 49.5% ASR on the latest YOLOv8 detector in black-box settings. This work highlights the potential of dynamic NeRF-based UV mapping for creating more effective adversarial attacks on person detectors, addressing key challenges in modeling human movement and texture modification.

摘要: 在最近的研究中，由于人体运动的灵活性，使用补丁或基于静态3D模型的纹理修改对个人检测器进行对抗性攻击的成功率很低。对由各种动作引起的三维变形进行建模一直是一个重大挑战。幸运的是，神经辐射场(NERF)在动态人体建模方面的进步提供了新的可能性。在本文中，我们介绍了紫外线攻击，这是一种突破性的方法，即使在广泛的和看不见的人为操作的情况下也能获得高成功率。我们通过利用基于动态神经网络的UV映射来解决上述挑战。UV-Attack可以在不同的动作和视角下生成人体图像，甚至可以通过从SMPL参数空间采样来创建新的动作。虽然动态NERF模型能够对人体建模，但修改服装纹理是具有挑战性的，因为它们嵌入在神经网络参数中。为了解决这个问题，UV-Attack生成UV贴图，而不是RGB图像，并修改纹理堆栈。这种方法实现了实时纹理编辑，使攻击更具实用性。我们还提出了一种新的姿势变换损失期望(EoPT)，以提高对看不见的姿势和视点的规避成功率。我们的实验表明，在动态视频环境下，UV-Attack对FastRCNN模型的各种姿势的攻击成功率达到了92.75%，远远超过了最先进的AdvCamou攻击，后者的ASR只有28.50%。此外，在黑盒设置下，我们在最新的YOLOv8探测器上实现了49.5%的ASR。这项工作突出了基于NERF的动态UV映射在创建对个人探测器的更有效的对抗性攻击方面的潜力，解决了在建模人类运动和纹理修改方面的关键挑战。



## **11. BaThe: Defense against the Jailbreak Attack in Multimodal Large Language Models by Treating Harmful Instruction as Backdoor Trigger**

BaThe：通过将有害指令视为后门触发来防御多模式大型语言模型中的越狱攻击 cs.CR

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2408.09093v2) [paper-pdf](http://arxiv.org/pdf/2408.09093v2)

**Authors**: Yulin Chen, Haoran Li, Yirui Zhang, Zihao Zheng, Yangqiu Song, Bryan Hooi

**Abstract**: Multimodal Large Language Models (MLLMs) have showcased impressive performance in a variety of multimodal tasks. On the other hand, the integration of additional image modality may allow the malicious users to inject harmful content inside the images for jailbreaking. Unlike text-based LLMs, where adversaries need to select discrete tokens to conceal their malicious intent using specific algorithms, the continuous nature of image signals provides a direct opportunity for adversaries to inject harmful intentions. In this work, we propose $\textbf{BaThe}$ ($\textbf{Ba}$ckdoor $\textbf{T}$rigger S$\textbf{h}$i$\textbf{e}$ld), a simple yet effective jailbreak defense mechanism. Our work is motivated by recent research on jailbreak backdoor attack and virtual prompt backdoor attack in generative language models. Jailbreak backdoor attack uses harmful instructions combined with manually crafted strings as triggers to make the backdoored model generate prohibited responses. We assume that harmful instructions can function as triggers, and if we alternatively set rejection responses as the triggered response, the backdoored model then can defend against jailbreak attacks. We achieve this by utilizing virtual rejection prompt, similar to the virtual prompt backdoor attack. We embed the virtual rejection prompt into the soft text embeddings, which we call ``wedge''. Our comprehensive experiments demonstrate that BaThe effectively mitigates various types of jailbreak attacks and is adaptable to defend against unseen attacks, with minimal impact on MLLMs' performance.

摘要: 多通道大型语言模型(MLLM)在各种多通道任务中表现出令人印象深刻的性能。另一方面，附加图像模式的集成可能允许恶意用户在图像中注入有害内容以越狱。与基于文本的LLMS不同，在LLMS中，攻击者需要使用特定的算法选择离散的令牌来隐藏其恶意意图，而图像信号的连续性为攻击者提供了直接注入有害意图的机会。在这项工作中，我们提出了一种简单而有效的越狱防御机制--$\extbf{bathe}$($\extbf{ba}$ck door$\extbf{T}$rigger S$\extbf{h}$i$\extbf{e}$ld)。我们的工作是基于生成式语言模型对越狱后门攻击和虚拟提示后门攻击的最新研究。越狱后门攻击使用有害指令和手动创建的字符串作为触发器，使后门模型生成被禁止的响应。我们假设有害指令可以作为触发器，如果我们将拒绝响应设置为触发响应，那么反向模型就可以防御越狱攻击。我们通过利用虚拟拒绝提示来实现这一点，类似于虚拟提示后门攻击。我们将虚拟拒绝提示嵌入到软文本嵌入中，我们称之为‘’楔形‘’。我们的综合实验表明，BAIT有效地缓解了各种类型的越狱攻击，并且能够自适应地防御看不见的攻击，对MLLMS的性能影响最小。



## **12. An Efficiency Firmware Verification Framework for Public Key Infrastructure with Smart Grid and Energy Storage System**

具有智能电网和储能系统的公钥基础设施的效率硬件验证框架 cs.CR

10pages, 5 figures

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.05722v1) [paper-pdf](http://arxiv.org/pdf/2501.05722v1)

**Authors**: Jhih-Zen Shih, Cheng-Che Chuang, Hong-Sheng Huang, Hsuan-Tung Chen, Hung-Min Sun

**Abstract**: As a critical component of electrical energy infrastructure, the smart grid system has become indispensable to the energy sector. However, the rapid evolution of smart grids has attracted numerous nation-state actors seeking to disrupt the power infrastructure of adversarial nations. This development underscores the urgent need to establish secure mechanisms for firmware updates, with firmware signing and verification serving as pivotal elements in safeguarding system integrity. In this work, we propose a digital signing and verification framework grounded in Public Key Infrastructure (PKI), specifically tailored for resource-constrained devices such as smart meters. The framework utilizes the Concise Binary Object Representation (CBOR) and Object Signing and Encryption (COSE) formats to achieve efficient da-ta encapsulation and robust security features. Our approach not only en-sures the secure deployment of firmware updates against the convergence of information technology (IT) and operational technology (OT) attacks but also addresses performance bottlenecks stemming from device limitations, thereby enhancing the overall reliability and stability of the smart grid sys-tem.

摘要: 作为电力能源基础设施的重要组成部分，智能电网系统已成为能源行业不可或缺的重要组成部分。然而，智能电网的快速发展吸引了无数寻求扰乱敌对国家电力基础设施的民族国家参与者。这一发展突显了迫切需要建立固件更新的安全机制，其中固件签名和验证是保障系统完整性的关键要素。在这项工作中，我们提出了一个基于公钥基础设施(PKI)的数字签名和验证框架，专门为智能电表等资源受限设备量身定做。该框架使用简明的二进制对象表示(CBOR)和对象签名和加密(COSE)格式来实现高效的数据封装和健壮的安全特性。该方法不仅确保了固件更新的安全部署，不受信息技术(IT)和操作技术(OT)攻击的影响，而且解决了设备限制带来的性能瓶颈，从而提高了智能电网系统的整体可靠性和稳定性。



## **13. Adversarial Robustness for Deep Learning-based Wildfire Prediction Models**

基于深度学习的野火预测模型的对抗鲁棒性 cs.CV

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2412.20006v2) [paper-pdf](http://arxiv.org/pdf/2412.20006v2)

**Authors**: Ryo Ide, Lei Yang

**Abstract**: Smoke detection using Deep Neural Networks (DNNs) is an effective approach for early wildfire detection. However, because smoke is temporally and spatially anomalous, there are limitations in collecting sufficient training data. This raises overfitting and bias concerns in existing DNN-based wildfire detection models. Thus, we introduce WARP (Wildfire Adversarial Robustness Procedure), the first model-agnostic framework for evaluating the adversarial robustness of DNN-based wildfire detection models. WARP addresses limitations in smoke image diversity using global and local adversarial attack methods. The global attack method uses image-contextualized Gaussian noise, while the local attack method uses patch noise injection, tailored to address critical aspects of wildfire detection. Leveraging WARP's model-agnostic capabilities, we assess the adversarial robustness of real-time Convolutional Neural Networks (CNNs) and Transformers. The analysis revealed valuable insights into the models' limitations. Specifically, the global attack method demonstrates that the Transformer model has more than 70% precision degradation than the CNN against global noise. In contrast, the local attack method shows that both models are susceptible to cloud image injections when detecting smoke-positive instances, suggesting a need for model improvements through data augmentation. WARP's comprehensive robustness analysis contributed to the development of wildfire-specific data augmentation strategies, marking a step toward practicality.

摘要: 基于深度神经网络的烟雾检测是野火早期检测的一种有效方法。然而，由于烟雾在时间和空间上都是反常的，收集足够的训练数据是有局限性的。这在现有的基于DNN的野火检测模型中引发了过度拟合和偏差的担忧。因此，我们引入了WARP(Wildfire对抗稳健性过程)，这是第一个模型不可知的框架，用于评估基于DNN的野火检测模型的对抗稳健性。WARP使用全球和局部对抗性攻击方法解决烟雾图像多样性方面的限制。全局攻击方法使用与图像相关的高斯噪声，而局部攻击方法使用补丁噪声注入，该方法针对野火检测的关键方面进行了量身定做。利用WARP的模型不可知能力，我们评估了实时卷积神经网络(CNN)和变形金刚的对抗健壮性。分析揭示了对模型局限性的有价值的见解。具体地说，全局攻击方法表明，对于全局噪声，Transformer模型的精度比CNN下降了70%以上。相比之下，本地攻击方法表明，当检测到烟雾阳性实例时，这两个模型都容易受到云图注入的影响，这表明需要通过数据增强来改进模型。WARP的全面稳健性分析有助于开发特定于野火的数据增强策略，标志着朝着实用化迈出了一步。



## **14. Enforcing Fundamental Relations via Adversarial Attacks on Input Parameter Correlations**

通过对输入参数相关性的对抗攻击来强制基本关系 cs.LG

12 pages, 8 figures (Without appendix)

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.05588v1) [paper-pdf](http://arxiv.org/pdf/2501.05588v1)

**Authors**: Timo Saala, Lucie Flek, Alexander Jung, Akbar Karimi, Alexander Schmidt, Matthias Schott, Philipp Soldin, Christopher Wiebusch

**Abstract**: Correlations between input parameters play a crucial role in many scientific classification tasks, since these are often related to fundamental laws of nature. For example, in high energy physics, one of the common deep learning use-cases is the classification of signal and background processes in particle collisions. In many such cases, the fundamental principles of the correlations between observables are often better understood than the actual distributions of the observables themselves. In this work, we present a new adversarial attack algorithm called Random Distribution Shuffle Attack (RDSA), emphasizing the correlations between observables in the network rather than individual feature characteristics. Correct application of the proposed novel attack can result in a significant improvement in classification performance - particularly in the context of data augmentation - when using the generated adversaries within adversarial training. Given that correlations between input features are also crucial in many other disciplines. We demonstrate the RDSA effectiveness on six classification tasks, including two particle collision challenges (using CERN Open Data), hand-written digit recognition (MNIST784), human activity recognition (HAR), weather forecasting (Rain in Australia), and ICU patient mortality (MIMIC-IV), demonstrating a general use case beyond fundamental physics for this new type of adversarial attack algorithms.

摘要: 输入参数之间的相关性在许多科学分类任务中起着至关重要的作用，因为这些任务往往与基本的自然规律有关。例如，在高能物理中，一个常见的深度学习用例是对粒子碰撞中的信号和背景过程进行分类。在许多这样的情况下，比起观测数据本身的实际分布，人们往往更好地理解了观测数据之间相互关联的基本原理。在这项工作中，我们提出了一种新的对抗性攻击算法，称为随机分布混洗攻击(RDSA)，它强调网络中可观测对象之间的相关性，而不是单个特征特征。当在对抗性训练中使用生成的对手时，正确应用所提出的新型攻击可以导致分类性能的显著改善--特别是在数据增强的背景下。鉴于输入特征之间的相关性在许多其他学科中也是至关重要的。我们在六个分类任务上展示了RDSA的有效性，包括两个粒子碰撞挑战(使用CERN Open Data)、手写数字识别(MNIST784)、人类活动识别(HAR)、天气预报(澳大利亚的Rain)和ICU患者死亡率(MIMIC-IV)，展示了这种新型对抗性攻击算法超越基础物理的一般用例。



## **15. CROPS: Model-Agnostic Training-Free Framework for Safe Image Synthesis with Latent Diffusion Models**

CROPS：模型不可知的免培训框架，用于使用潜在扩散模型的安全图像合成 cs.CV

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.05359v1) [paper-pdf](http://arxiv.org/pdf/2501.05359v1)

**Authors**: Junha Park, Ian Ryu, Jaehui Hwang, Hyungkeun Park, Jiyoon Kim, Jong-Seok Lee

**Abstract**: With advances in diffusion models, image generation has shown significant performance improvements. This raises concerns about the potential abuse of image generation, such as the creation of explicit or violent images, commonly referred to as Not Safe For Work (NSFW) content. To address this, the Stable Diffusion model includes several safety checkers to censor initial text prompts and final output images generated from the model. However, recent research has shown that these safety checkers have vulnerabilities against adversarial attacks, allowing them to generate NSFW images. In this paper, we find that these adversarial attacks are not robust to small changes in text prompts or input latents. Based on this, we propose CROPS (Circular or RandOm Prompts for Safety), a model-agnostic framework that easily defends against adversarial attacks generating NSFW images without requiring additional training. Moreover, we develop an approach that utilizes one-step diffusion models for efficient NSFW detection (CROPS-1), further reducing computational resources. We demonstrate the superiority of our method in terms of performance and applicability.

摘要: 随着扩散模型的进步，图像生成的性能也有了显著的提高。这引发了人们对图像生成可能被滥用的担忧，例如创建露骨或暴力的图像，通常称为不安全工作(NSFW)内容。为了解决这一问题，稳定扩散模型包括几个安全检查器，用于审查从该模型生成的初始文本提示和最终输出图像。然而，最近的研究表明，这些安全检查器存在针对对手攻击的漏洞，允许它们生成NSFW图像。在本文中，我们发现这些对抗性攻击对文本提示或输入延迟的微小变化并不健壮。基于此，我们提出了一种模型不可知的框架--CRORS(圆形或随机安全提示)，它可以轻松地防御生成NSFW图像的敌意攻击，而不需要额外的训练。此外，我们开发了一种利用一步扩散模型来有效检测NSFW的方法(CRINS-1)，进一步减少了计算资源。我们从性能和适用性两个方面论证了该方法的优越性。



## **16. Safeguarding System Prompts for LLMs**

LLM的保护系统预算 cs.CR

15 pages, 5 figures, 2 tables

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2412.13426v2) [paper-pdf](http://arxiv.org/pdf/2412.13426v2)

**Authors**: Zhifeng Jiang, Zhihua Jin, Guoliang He

**Abstract**: Large language models (LLMs) are increasingly utilized in applications where system prompts, which guide model outputs, play a crucial role. These prompts often contain business logic and sensitive information, making their protection essential. However, adversarial and even regular user queries can exploit LLM vulnerabilities to expose these hidden prompts. To address this issue, we propose PromptKeeper, a robust defense mechanism designed to safeguard system prompts. PromptKeeper tackles two core challenges: reliably detecting prompt leakage and mitigating side-channel vulnerabilities when leakage occurs. By framing detection as a hypothesis-testing problem, PromptKeeper effectively identifies both explicit and subtle leakage. Upon detection, it regenerates responses using a dummy prompt, ensuring that outputs remain indistinguishable from typical interactions when no leakage is present. PromptKeeper ensures robust protection against prompt extraction attacks via either adversarial or regular queries, while preserving conversational capability and runtime efficiency during benign user interactions.

摘要: 大型语言模型(LLM)越来越多地被用于指导模型输出的系统提示发挥关键作用的应用中。这些提示通常包含业务逻辑和敏感信息，因此保护它们至关重要。但是，敌意的甚至常规的用户查询都可以利用LLM漏洞来暴露这些隐藏的提示。为了解决这个问题，我们提出了PromptKeeper，一种健壮的防御机制，旨在保护系统提示。PromptKeeper解决了两个核心挑战：可靠地检测及时泄漏和在发生泄漏时缓解侧通道漏洞。通过将检测框定为假设检验问题，PromptKeeper有效地识别了显性和细微的泄漏。一旦检测到，它会使用虚拟提示重新生成响应，确保在没有泄漏的情况下，输出与典型的交互没有区别。PromptKeeper通过对抗性或常规查询确保针对提示提取攻击的强大保护，同时在良性用户交互期间保持对话能力和运行效率。



## **17. RAG-WM: An Efficient Black-Box Watermarking Approach for Retrieval-Augmented Generation of Large Language Models**

RAG-WM：一种用于大型语言模型检索增强生成的高效黑箱水印方法 cs.CR

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.05249v1) [paper-pdf](http://arxiv.org/pdf/2501.05249v1)

**Authors**: Peizhuo Lv, Mengjie Sun, Hao Wang, Xiaofeng Wang, Shengzhi Zhang, Yuxuan Chen, Kai Chen, Limin Sun

**Abstract**: In recent years, tremendous success has been witnessed in Retrieval-Augmented Generation (RAG), widely used to enhance Large Language Models (LLMs) in domain-specific, knowledge-intensive, and privacy-sensitive tasks. However, attackers may steal those valuable RAGs and deploy or commercialize them, making it essential to detect Intellectual Property (IP) infringement. Most existing ownership protection solutions, such as watermarks, are designed for relational databases and texts. They cannot be directly applied to RAGs because relational database watermarks require white-box access to detect IP infringement, which is unrealistic for the knowledge base in RAGs. Meanwhile, post-processing by the adversary's deployed LLMs typically destructs text watermark information. To address those problems, we propose a novel black-box "knowledge watermark" approach, named RAG-WM, to detect IP infringement of RAGs. RAG-WM uses a multi-LLM interaction framework, comprising a Watermark Generator, Shadow LLM & RAG, and Watermark Discriminator, to create watermark texts based on watermark entity-relationship tuples and inject them into the target RAG. We evaluate RAG-WM across three domain-specific and two privacy-sensitive tasks on four benchmark LLMs. Experimental results show that RAG-WM effectively detects the stolen RAGs in various deployed LLMs. Furthermore, RAG-WM is robust against paraphrasing, unrelated content removal, knowledge insertion, and knowledge expansion attacks. Lastly, RAG-WM can also evade watermark detection approaches, highlighting its promising application in detecting IP infringement of RAG systems.

摘要: 近年来，检索增强生成(RAG)取得了巨大的成功，它被广泛用于增强领域特定、知识密集型和隐私敏感任务中的大型语言模型(LLM)。然而，攻击者可能会窃取这些有价值的破布并将其部署或商业化，这使得检测侵犯知识产权(IP)变得至关重要。大多数现有的所有权保护解决方案，如水印，都是为关系数据库和文本设计的。它们不能直接应用于RAG，因为关系数据库水印需要白盒访问来检测知识产权侵权，这对于RAG中的知识库来说是不现实的。同时，由对手部署的LLMS进行的后处理通常会破坏文本水印信息。针对这些问题，我们提出了一种新的黑盒“知识水印”方法RAG-WM来检测RAG的知识产权侵权行为。RAG-WM使用多LLM交互框架，包括水印生成器、阴影LLM和RAG和水印鉴别器，基于水印实体关系元组创建水印文本并将其注入目标RAG。我们在四个基准LLM上对RAG-WM进行了评估，测试了三个领域特定的任务和两个隐私敏感任务。实验结果表明，RAG-WM能够有效地检测出各种部署的LLM中被盗的RAG。此外，RAG-WM对释义、无关内容移除、知识插入和知识扩展攻击具有较强的鲁棒性。最后，RAG-WM还可以避开水印检测方法，在检测RAG系统的知识产权侵权行为方面具有广阔的应用前景。



## **18. DiffAttack: Diffusion-based Timbre-reserved Adversarial Attack in Speaker Identification**

DistAttack：说话人识别中基于扩散的Timbre-Reserved对抗攻击 cs.SD

5 pages,4 figures, accepted by ICASSP 2025

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.05127v1) [paper-pdf](http://arxiv.org/pdf/2501.05127v1)

**Authors**: Qing Wang, Jixun Yao, Zhaokai Sun, Pengcheng Guo, Lei Xie, John H. L. Hansen

**Abstract**: Being a form of biometric identification, the security of the speaker identification (SID) system is of utmost importance. To better understand the robustness of SID systems, we aim to perform more realistic attacks in SID, which are challenging for both humans and machines to detect. In this study, we propose DiffAttack, a novel timbre-reserved adversarial attack approach that exploits the capability of a diffusion-based voice conversion (DiffVC) model to generate adversarial fake audio with distinct target speaker attribution. By introducing adversarial constraints into the generative process of the diffusion-based voice conversion model, we craft fake samples that effectively mislead target models while preserving speaker-wise characteristics. Specifically, inspired by the use of randomly sampled Gaussian noise in conventional adversarial attacks and diffusion processes, we incorporate adversarial constraints into the reverse diffusion process. These constraints subtly guide the reverse diffusion process toward aligning with the target speaker distribution. Our experiments on the LibriTTS dataset indicate that DiffAttack significantly improves the attack success rate compared to vanilla DiffVC and other methods. Moreover, objective and subjective evaluations demonstrate that introducing adversarial constraints does not compromise the speech quality generated by the DiffVC model.

摘要: 说话人识别(SID)系统作为生物特征识别的一种形式，其安全性至关重要。为了更好地理解SID系统的健壮性，我们的目标是在SID中执行更现实的攻击，这些攻击对人类和机器都是具有挑战性的。在这项研究中，我们提出了DiffAttack，这是一种新的保留音色的对抗性攻击方法，它利用基于扩散的语音转换(DiffVC)模型的能力来生成具有明显目标说话人属性的对抗性假音频。通过将对抗性约束引入到基于扩散的语音转换模型的生成过程中，我们制作了虚假样本，在保持说话人特征的同时有效地误导目标模型。具体地说，受传统对抗性攻击和扩散过程中使用随机抽样的高斯噪声的启发，我们将对抗性约束引入到反向扩散过程中。这些限制巧妙地引导反向扩散过程与目标说话人分布保持一致。我们在LibriTTS数据集上的实验表明，与Vanilla DiffVC和其他方法相比，DiffAttack显著提高了攻击成功率。此外，客观和主观评估表明，引入对抗性约束并不会影响DiffVC模型生成的语音质量。



## **19. Turning Logic Against Itself : Probing Model Defenses Through Contrastive Questions**

将逻辑与自身对立：通过对比问题探索模型辩护 cs.CL

Our code is publicly available at  https://github.com/UKPLab/POATE-attack

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.01872v2) [paper-pdf](http://arxiv.org/pdf/2501.01872v2)

**Authors**: Rachneet Sachdeva, Rima Hazra, Iryna Gurevych

**Abstract**: Large language models, despite extensive alignment with human values and ethical principles, remain vulnerable to sophisticated jailbreak attacks that exploit their reasoning abilities. Existing safety measures often detect overt malicious intent but fail to address subtle, reasoning-driven vulnerabilities. In this work, we introduce POATE (Polar Opposite query generation, Adversarial Template construction, and Elaboration), a novel jailbreak technique that harnesses contrastive reasoning to provoke unethical responses. POATE crafts semantically opposing intents and integrates them with adversarial templates, steering models toward harmful outputs with remarkable subtlety. We conduct extensive evaluation across six diverse language model families of varying parameter sizes to demonstrate the robustness of the attack, achieving significantly higher attack success rates (~44%) compared to existing methods. To counter this, we propose Intent-Aware CoT and Reverse Thinking CoT, which decompose queries to detect malicious intent and reason in reverse to evaluate and reject harmful responses. These methods enhance reasoning robustness and strengthen the model's defense against adversarial exploits.

摘要: 大型语言模型尽管与人类价值观和伦理原则广泛一致，但仍然容易受到复杂的越狱攻击，这些攻击利用了它们的推理能力。现有的安全措施经常检测到公开的恶意意图，但无法解决细微的、推理驱动的漏洞。在这项工作中，我们介绍了POATE(极地相反查询生成，对抗性模板构建和精化)，这是一种新的越狱技术，利用对比推理来引发不道德的反应。波特在语义上设计了相反的意图，并将它们与对抗性模板整合在一起，以惊人的微妙程度引导模型指向有害的输出。我们对六个不同参数大小的不同语言模型家族进行了广泛的评估，以证明攻击的健壮性，与现有方法相比，攻击成功率显著提高(~44%)。针对这一问题，我们提出了意图感知COT和逆向思维COT，它们分解查询以检测恶意意图，并反向推理以评估和拒绝有害响应。这些方法增强了推理的健壮性，增强了模型对对手攻击的防御能力。



## **20. On Measuring Unnoticeability of Graph Adversarial Attacks: Observations, New Measure, and Applications**

关于测量图对抗攻击的不可想象性：观察、新测量和应用 cs.LG

KDD 2025

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.05015v1) [paper-pdf](http://arxiv.org/pdf/2501.05015v1)

**Authors**: Hyeonsoo Jo, Hyunjin Hwang, Fanchen Bu, Soo Yong Lee, Chanyoung Park, Kijung Shin

**Abstract**: Adversarial attacks are allegedly unnoticeable. Prior studies have designed attack noticeability measures on graphs, primarily using statistical tests to compare the topology of original and (possibly) attacked graphs. However, we observe two critical limitations in the existing measures. First, because the measures rely on simple rules, attackers can readily enhance their attacks to bypass them, reducing their attack "noticeability" and, yet, maintaining their attack performance. Second, because the measures naively leverage global statistics, such as degree distributions, they may entirely overlook attacks until severe perturbations occur, letting the attacks be almost "totally unnoticeable." To address the limitations, we introduce HideNSeek, a learnable measure for graph attack noticeability. First, to mitigate the bypass problem, HideNSeek learns to distinguish the original and (potential) attack edges using a learnable edge scorer (LEO), which scores each edge on its likelihood of being an attack. Second, to mitigate the overlooking problem, HideNSeek conducts imbalance-aware aggregation of all the edge scores to obtain the final noticeability score. Using six real-world graphs, we empirically demonstrate that HideNSeek effectively alleviates the observed limitations, and LEO (i.e., our learnable edge scorer) outperforms eleven competitors in distinguishing attack edges under five different attack methods. For an additional application, we show that LEO boost the performance of robust GNNs by removing attack-like edges.

摘要: 据称，对抗性攻击不会引起注意。以前的研究已经在图上设计了攻击可察觉度量，主要使用统计测试来比较原始图和(可能)被攻击的图的拓扑。然而，我们注意到现有措施中的两个关键限制。首先，由于这些措施依赖于简单的规则，攻击者可以很容易地增强攻击以绕过它们，降低他们的攻击“可察觉”，同时保持他们的攻击性能。其次，由于这些措施天真地利用了全球统计数据，如学位分布，它们可能会完全忽略攻击，直到发生严重的扰动，让攻击几乎“完全不被察觉”。为了解决这些局限性，我们引入了HideNSeek，这是一种图攻击可察觉的可学习度量。首先，为了缓解绕过问题，HideNSeek使用可学习的边缘记分器(LEO)来学习区分原始和(潜在的)攻击边缘，LEO根据每个边缘是攻击的可能性对其进行评分。其次，为了缓解被忽视的问题，HideNSeek对所有边缘分数进行不平衡感知聚合，以获得最终的可察觉分数。使用六个真实世界的图，我们的经验表明，HideNSeek有效地缓解了观察到的局限性，并且在五种不同的攻击方法下，Leo(即我们的可学习的边缘得分器)在区分攻击边缘方面优于11个竞争对手。对于另一个应用，我们证明了LEO通过去除类似攻击的边缘来提高健壮GNN的性能。



## **21. SpaLLM-Guard: Pairing SMS Spam Detection Using Open-source and Commercial LLMs**

SpaLLM-Guard：使用开源和商业LLM配对短信垃圾邮件检测 cs.CR

17 pages

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.04985v1) [paper-pdf](http://arxiv.org/pdf/2501.04985v1)

**Authors**: Muhammad Salman, Muhammad Ikram, Nardine Basta, Mohamed Ali Kaafar

**Abstract**: The increasing threat of SMS spam, driven by evolving adversarial techniques and concept drift, calls for more robust and adaptive detection methods. In this paper, we evaluate the potential of large language models (LLMs), both open-source and commercial, for SMS spam detection, comparing their performance across zero-shot, few-shot, fine-tuning, and chain-of-thought prompting approaches. Using a comprehensive dataset of SMS messages, we assess the spam detection capabilities of prominent LLMs such as GPT-4, DeepSeek, LLAMA-2, and Mixtral. Our findings reveal that while zero-shot learning provides convenience, it is unreliable for effective spam detection. Few-shot learning, particularly with carefully selected examples, improves detection but exhibits variability across models. Fine-tuning emerges as the most effective strategy, with Mixtral achieving 98.6% accuracy and a balanced false positive and false negative rate below 2%, meeting the criteria for robust spam detection. Furthermore, we explore the resilience of these models to adversarial attacks, finding that fine-tuning significantly enhances robustness against both perceptible and imperceptible manipulations. Lastly, we investigate the impact of concept drift and demonstrate that fine-tuned LLMs, especially when combined with few-shot learning, can mitigate its effects, maintaining high performance even on evolving spam datasets. This study highlights the importance of fine-tuning and tailored learning strategies to deploy LLMs effectively for real-world SMS spam detection

摘要: 在不断发展的敌意技术和概念漂移的推动下，垃圾短信的威胁越来越大，这要求更健壮和自适应的检测方法。在本文中，我们评估了开源和商业的大型语言模型(LLM)在短信垃圾邮件检测中的潜力，比较了它们在零射、少射、微调和思维链提示方法中的性能。使用全面的短信数据集，我们评估了GPT-4、DeepSeek、Llama-2和Mixtral等知名LLMS的垃圾邮件检测能力。我们的发现表明，虽然零机会学习提供了便利，但对于有效的垃圾邮件检测来说，它是不可靠的。少发式学习，尤其是精心挑选的例子，提高了检测能力，但在不同模型之间表现出多样性。微调成为最有效的策略，Mixtral达到了98.6%的准确率，假阳性率和假阴性率平衡在2%以下，满足了稳健的垃圾邮件检测标准。此外，我们探讨了这些模型对敌意攻击的弹性，发现微调显著增强了对可感知和不可感知操纵的稳健性。最后，我们研究了概念漂移的影响，并证明了微调的LLMS，特别是当与少镜头学习相结合时，可以缓解其影响，即使在不断演变的垃圾邮件数据集上也保持了高性能。这项研究强调了微调和量身定制的学习策略的重要性，以有效地部署LLMS来检测真实世界的短信垃圾邮件



## **22. LayerMix: Enhanced Data Augmentation through Fractal Integration for Robust Deep Learning**

LayerMix：通过Fractal集成增强数据增强，以实现稳健的深度学习 cs.CV

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2501.04861v1) [paper-pdf](http://arxiv.org/pdf/2501.04861v1)

**Authors**: Hafiz Mughees Ahmad, Dario Morle, Afshin Rahimi

**Abstract**: Deep learning models have demonstrated remarkable performance across various computer vision tasks, yet their vulnerability to distribution shifts remains a critical challenge. Despite sophisticated neural network architectures, existing models often struggle to maintain consistent performance when confronted with Out-of-Distribution (OOD) samples, including natural corruptions, adversarial perturbations, and anomalous patterns. We introduce LayerMix, an innovative data augmentation approach that systematically enhances model robustness through structured fractal-based image synthesis. By meticulously integrating structural complexity into training datasets, our method generates semantically consistent synthetic samples that significantly improve neural network generalization capabilities. Unlike traditional augmentation techniques that rely on random transformations, LayerMix employs a structured mixing pipeline that preserves original image semantics while introducing controlled variability. Extensive experiments across multiple benchmark datasets, including CIFAR-10, CIFAR-100, ImageNet-200, and ImageNet-1K demonstrate LayerMixs superior performance in classification accuracy and substantially enhances critical Machine Learning (ML) safety metrics, including resilience to natural image corruptions, robustness against adversarial attacks, improved model calibration and enhanced prediction consistency. LayerMix represents a significant advancement toward developing more reliable and adaptable artificial intelligence systems by addressing the fundamental challenges of deep learning generalization. The code is available at https://github.com/ahmadmughees/layermix.

摘要: 深度学习模型在各种计算机视觉任务中表现出了显著的性能，但它们对分布变化的脆弱性仍然是一个关键的挑战。尽管有复杂的神经网络结构，但现有的模型在面对分布外(OOD)样本时往往难以保持一致的性能，这些样本包括自然损坏、对抗性扰动和异常模式。我们介绍了LayerMix，这是一种创新的数据增强方法，它通过基于结构化分形的图像合成来系统地增强模型的稳健性。通过将结构复杂性精心集成到训练数据集中，我们的方法生成语义一致的合成样本，显著提高了神经网络的泛化能力。与依赖随机变换的传统增强技术不同，LayerMix采用了结构化混合管道，在引入受控可变性的同时保留了原始图像的语义。在CIFAR-10、CIFAR-100、ImageNet-200和ImageNet-1K等多个基准数据集上的广泛实验表明，LayerMix在分类精度方面具有卓越的性能，并显著增强了关键机器学习(ML)安全指标，包括对自然图像损坏的弹性、对对手攻击的健壮性、改进的模型校准和增强的预测一致性。LayerMix通过解决深度学习泛化的根本挑战，代表着在开发更可靠和更具适应性的人工智能系统方面取得了重大进展。代码可在https://github.com/ahmadmughees/layermix.上获得



## **23. Reproducing HotFlip for Corpus Poisoning Attacks in Dense Retrieval**

在密集检索中重现HotFlip以应对Corpus中毒攻击 cs.IR

This paper has been accepted for oral presentation in the  reproducibility track at ECIR 2025

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2501.04802v1) [paper-pdf](http://arxiv.org/pdf/2501.04802v1)

**Authors**: Yongkang Li, Panagiotis Eustratiadis, Evangelos Kanoulas

**Abstract**: HotFlip is a topical gradient-based word substitution method for attacking language models. Recently, this method has been further applied to attack retrieval systems by generating malicious passages that are injected into a corpus, i.e., corpus poisoning. However, HotFlip is known to be computationally inefficient, with the majority of time being spent on gradient accumulation for each query-passage pair during the adversarial token generation phase, making it impossible to generate an adequate number of adversarial passages in a reasonable amount of time. Moreover, the attack method itself assumes access to a set of user queries, a strong assumption that does not correspond to how real-world adversarial attacks are usually performed. In this paper, we first significantly boost the efficiency of HotFlip, reducing the adversarial generation process from 4 hours per document to only 15 minutes, using the same hardware. We further contribute experiments and analysis on two additional tasks: (1) transfer-based black-box attacks, and (2) query-agnostic attacks. Whenever possible, we provide comparisons between the original method and our improved version. Our experiments demonstrate that HotFlip can effectively attack a variety of dense retrievers, with an observed trend that its attack performance diminishes against more advanced and recent methods. Interestingly, we observe that while HotFlip performs poorly in a black-box setting, indicating limited capacity for generalization, in query-agnostic scenarios its performance is correlated to the volume of injected adversarial passages.

摘要: HotFlip是一种基于主题梯度的单词替换方法，用于攻击语言模型。最近，这种方法被进一步应用于通过生成注入到语料库中的恶意段落来攻击检索系统，即语料库中毒。然而，HotFlip的计算效率很低，在对抗性令牌生成阶段，大部分时间花费在每个查询通道对的梯度累加上，使得在合理的时间内生成足够数量的对抗性通道是不可能的。此外，攻击方法本身假设可以访问一组用户查询，这是一个强烈的假设，与现实世界中通常如何执行对抗性攻击并不相符。在本文中，我们首先显著提高了HotFlip的效率，在使用相同硬件的情况下，将敌意生成过程从每个文档4小时减少到仅15分钟。我们进一步对两个额外的任务进行了实验和分析：(1)基于传输的黑盒攻击；(2)查询无关攻击。只要有可能，我们就会提供原始方法和改进后的方法之间的比较。我们的实验表明，HotFlip可以有效地攻击各种密集检索犬，并观察到其攻击性能与更先进和最新的方法相比有所下降的趋势。有趣的是，我们观察到，虽然HotFlip在黑盒环境中表现不佳，表明泛化能力有限，但在查询不可知的场景中，它的性能与注入的敌意段落的数量相关。



## **24. Correlated Privacy Mechanisms for Differentially Private Distributed Mean Estimation**

用于差异私有分布均值估计的相关隐私机制 cs.IT

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2407.03289v2) [paper-pdf](http://arxiv.org/pdf/2407.03289v2)

**Authors**: Sajani Vithana, Viveck R. Cadambe, Flavio P. Calmon, Haewon Jeong

**Abstract**: Differentially private distributed mean estimation (DP-DME) is a fundamental building block in privacy-preserving federated learning, where a central server estimates the mean of $d$-dimensional vectors held by $n$ users while ensuring $(\epsilon,\delta)$-DP. Local differential privacy (LDP) and distributed DP with secure aggregation (SA) are the most common notions of DP used in DP-DME settings with an untrusted server. LDP provides strong resilience to dropouts, colluding users, and adversarial attacks, but suffers from poor utility. In contrast, SA-based DP-DME achieves an $O(n)$ utility gain over LDP in DME, but requires increased communication and computation overheads and complex multi-round protocols to handle dropouts and attacks. In this work, we present a generalized framework for DP-DME, that captures LDP and SA-based mechanisms as extreme cases. Our framework provides a foundation for developing and analyzing a variety of DP-DME protocols that leverage correlated privacy mechanisms across users. To this end, we propose CorDP-DME, a novel DP-DME mechanism based on the correlated Gaussian mechanism, that spans the gap between DME with LDP and distributed DP. We prove that CorDP-DME offers a favorable balance between utility and resilience to dropout and collusion. We provide an information-theoretic analysis of CorDP-DME, and derive theoretical guarantees for utility under any given privacy parameters and dropout/colluding user thresholds. Our results demonstrate that (anti) correlated Gaussian DP mechanisms can significantly improve utility in mean estimation tasks compared to LDP -- even in adversarial settings -- while maintaining better resilience to dropouts and attacks compared to distributed DP.

摘要: 差分私有分布平均估计(DP-DME)是保护隐私的联合学习的基本构件，其中中央服务器估计$n$用户所持有的$d$维向量的平均值，同时确保$(？，？)$-DP。本地差异隐私(LDP)和具有安全聚合的分布式DP(SA)是DP-DME设置中使用不受信任服务器的最常见概念。LDP对辍学、串通用户和敌意攻击具有很强的弹性，但实用性较差。相比之下，基于SA的DP-DME在DME中比LDP获得$O(N)$效用收益，但需要更多的通信和计算开销以及复杂的多轮协议来处理丢弃和攻击。在这项工作中，我们提出了一个通用的DP-DME框架，该框架将基于LDP和SA的机制作为极端情况来捕获。我们的框架为开发和分析各种DP-DME协议提供了基础，这些协议利用了跨用户的相关隐私机制。为此，我们提出了一种新的基于相关高斯机制的DP-DME机制CorDP-DME，它跨越了DME与LDP和分布式DP之间的差距。我们证明了CorDP-DME在实用性和对丢弃和共谋的恢复能力之间提供了良好的平衡。我们对CorDP-DME进行了信息论分析，并推导出在任何给定的隐私参数和丢弃/合谋用户阈值下的效用的理论保证。我们的结果表明，与LDP相比，(反)相关的高斯DP机制可以显著提高均值估计任务的实用性--即使在对抗环境中--同时与分布式DP相比，保持更好的对丢弃和攻击的弹性。



## **25. Resilient Peer-to-peer Learning based on Adaptive Aggregation**

基于自适应聚合的弹性点对点学习 cs.LG

11 pages

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2501.04610v1) [paper-pdf](http://arxiv.org/pdf/2501.04610v1)

**Authors**: Chandreyee Bhowmick, Xenofon Koutsoukos

**Abstract**: Collaborative learning in peer-to-peer networks offers the benefits of distributed learning while mitigating the risks associated with single points of failure inherent in centralized servers. However, adversarial workers pose potential threats by attempting to inject malicious information into the network. Thus, ensuring the resilience of peer-to-peer learning emerges as a pivotal research objective. The challenge is exacerbated in the presence of non-convex loss functions and non-iid data distributions. This paper introduces a resilient aggregation technique tailored for such scenarios, aimed at fostering similarity among peers' learning processes. The aggregation weights are determined through an optimization procedure, and use the loss function computed using the neighbor's models and individual private data, thereby addressing concerns regarding data privacy in distributed machine learning. Theoretical analysis demonstrates convergence of parameters with non-convex loss functions and non-iid data distributions. Empirical evaluations across three distinct machine learning tasks support the claims. The empirical findings, which encompass a range of diverse attack models, also demonstrate improved accuracy when compared to existing methodologies.

摘要: 对等网络中的协作学习提供了分布式学习的好处，同时降低了与集中式服务器固有的单点故障相关的风险。然而，敌意工作者试图将恶意信息注入网络，从而构成潜在威胁。因此，确保对等学习的弹性成为一个关键的研究目标。在存在非凸损失函数和非IID数据分布的情况下，这一挑战更加严重。本文介绍了一种为此类场景量身定做的弹性聚合技术，旨在促进节点学习过程之间的相似性。聚集权重通过优化过程确定，并使用使用邻居的模型和个人隐私数据计算的损失函数，从而解决了分布式机器学习中对数据隐私的担忧。理论分析证明了参数在非凸损失函数和非IID数据分布下的收敛。对三个不同的机器学习任务进行的经验评估支持了这一说法。这些经验发现涵盖了一系列不同的攻击模型，与现有方法相比，也证明了更高的准确性。



## **26. Tougher Text, Smarter Models: Raising the Bar for Adversarial Defence Benchmarks**

更强硬的文本，更智能的模型：提高对抗性防御基准的门槛 cs.CL

Will be presented as an oral in-person presentation at the conference  of COLING 2025

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2501.02654v2) [paper-pdf](http://arxiv.org/pdf/2501.02654v2)

**Authors**: Yang Wang, Chenghua Lin

**Abstract**: Recent advancements in natural language processing have highlighted the vulnerability of deep learning models to adversarial attacks. While various defence mechanisms have been proposed, there is a lack of comprehensive benchmarks that evaluate these defences across diverse datasets, models, and tasks. In this work, we address this gap by presenting an extensive benchmark for textual adversarial defence that significantly expands upon previous work. Our benchmark incorporates a wide range of datasets, evaluates state-of-the-art defence mechanisms, and extends the assessment to include critical tasks such as single-sentence classification, similarity and paraphrase identification, natural language inference, and commonsense reasoning. This work not only serves as a valuable resource for researchers and practitioners in the field of adversarial robustness but also identifies key areas for future research in textual adversarial defence. By establishing a new standard for benchmarking in this domain, we aim to accelerate progress towards more robust and reliable natural language processing systems.

摘要: 自然语言处理的最新进展突显了深度学习模型在对抗性攻击中的脆弱性。虽然已经提出了各种防御机制，但缺乏全面的基准来评估不同数据集、模型和任务中的这些防御。在这项工作中，我们通过提供一个广泛的文本对抗防御基准来解决这一差距，该基准大大扩展了以前的工作。我们的基准纳入了广泛的数据集，评估了最先进的防御机制，并将评估扩展到包括关键任务，如单句分类、相似性和释义识别、自然语言推理和常识推理。这项工作不仅为对抗稳健性领域的研究人员和实践者提供了宝贵的资源，而且还确定了未来文本对抗防御研究的关键领域。通过在这一领域建立一个新的基准标准，我们的目标是加快朝着更健壮和可靠的自然语言处理系统的进展。



## **27. Towards Fair Class-wise Robustness: Class Optimal Distribution Adversarial Training**

迈向公平的班级稳健性：班级最优分布对抗训练 cs.LG

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2501.04527v1) [paper-pdf](http://arxiv.org/pdf/2501.04527v1)

**Authors**: Hongxin Zhi, Hongtao Yu, Shaome Li, Xiuming Zhao, Yiteng Wu

**Abstract**: Adversarial training has proven to be a highly effective method for improving the robustness of deep neural networks against adversarial attacks. Nonetheless, it has been observed to exhibit a limitation in terms of robust fairness, characterized by a significant disparity in robustness across different classes. Recent efforts to mitigate this problem have turned to class-wise reweighted methods. However, these methods suffer from a lack of rigorous theoretical analysis and are limited in their exploration of the weight space, as they mainly rely on existing heuristic algorithms or intuition to compute weights. In addition, these methods fail to guarantee the consistency of the optimization direction due to the decoupled optimization of weights and the model parameters. They potentially lead to suboptimal weight assignments and consequently, a suboptimal model. To address these problems, this paper proposes a novel min-max training framework, Class Optimal Distribution Adversarial Training (CODAT), which employs distributionally robust optimization to fully explore the class-wise weight space, thus enabling the identification of the optimal weight with theoretical guarantees. Furthermore, we derive a closed-form optimal solution to the internal maximization and then get a deterministic equivalent objective function, which provides a theoretical basis for the joint optimization of weights and model parameters. Meanwhile, we propose a fairness elasticity coefficient for the evaluation of the algorithm with regard to both robustness and robust fairness. Experimental results on various datasets show that the proposed method can effectively improve the robust fairness of the model and outperform the state-of-the-art approaches.

摘要: 对抗性训练已被证明是提高深层神经网络抵抗对抗性攻击的鲁棒性的一种非常有效的方法。尽管如此，人们观察到它在稳健公平方面表现出局限性，其特征是不同类别之间的稳健程度存在显著差异。最近缓解这一问题的努力已转向按类别重新加权的方法。然而，这些方法缺乏严谨的理论分析，并且主要依靠现有的启发式算法或直觉来计算权重，从而限制了对权重空间的探索。此外，由于权重和模型参数的解耦优化，这些方法不能保证优化方向的一致性。它们可能导致次优的权重分配，从而导致次优的模型。针对这些问题，本文提出了一种新的最小-最大训练框架--类最优分布对抗训练(CODAT)，该框架采用分布稳健优化来充分探索类的权值空间，从而在理论上保证了最优权值的识别。进而推导出内部极大化的闭合最优解，进而得到确定性的等价目标函数，为权重和模型参数的联合优化提供了理论依据。同时，我们还提出了一个公平弹性系数来评价算法的健壮性和健壮性。在不同数据集上的实验结果表明，该方法能有效地提高模型的鲁棒性公平性，并优于现有的方法。



## **28. Multichannel Steganography: A Provably Secure Hybrid Steganographic Model for Secure Communication**

多通道隐写术：一种可证明安全的混合隐写术模型，用于安全通信 cs.CR

18 pages, 8 figures, 3 algorithms, This version is a preprint  uploaded to arXiv

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2501.04511v1) [paper-pdf](http://arxiv.org/pdf/2501.04511v1)

**Authors**: Obinna Omego, Michal Bosy

**Abstract**: This study introduces a novel steganographic model that synthesizes Steganography by Cover Modification (CMO) and Steganography by Cover Synthesis (CSY), enhancing both security and undetectability by generating cover messages or parameters while retaining the original cover's form, thus minimizing detection risks and overcoming the limitations of single-method techniques. Building upon this model, a refined Steganographic Communication Protocol is proposed, enhancing resilience against sophisticated threats such as Multichannel Replay Attacks and Multichannel Man-in-the-Middle Attacks, fortifying the protocol against potential tampering and improving upon prior works. To evaluate the security of the proposed protocol, a novel adversarial model is developed simulating a probabilistic polynomial time (PPT) adversary capable of intercepting communications across multiple channels. This model assesses the adversary's ability to compromise the protocol, providing a comprehensive security analysis. Finally, this study explores the practicality and adaptability of the model to both constrained environments like SMS banking and resource-rich settings such as blockchain transactions, demonstrating their potential to enhance financial services and security. These contributions present a robust and adaptable framework for secure steganographic communication, offering practical solutions for secure communications across diverse environments.

摘要: 提出了一种新的隐写模型，该模型综合了基于覆盖修改的隐写(CMO)和基于覆盖合成的隐写(CSY)，通过生成覆盖消息或参数来增强安全性和不可检测性，同时保持了原始覆盖的形式，从而最大限度地降低了检测风险，克服了单一方法技术的局限性。在此模型的基础上，提出了一种改进的隐写通信协议，增强了对多通道重放攻击和多通道中间人攻击等复杂威胁的抵御能力，增强了协议对潜在篡改的抵抗力，并在已有工作的基础上进行了改进。为了评估该协议的安全性，建立了一个新的敌手模型，该模型模拟了概率多项式时间(PPT)敌手在多个信道上截获通信的能力。该模型评估对手破坏协议的能力，提供全面的安全分析。最后，本研究探讨了该模型对短信银行等受限环境和区块链交易等资源丰富环境的实用性和适应性，展示了其增强金融服务和安全性的潜力。这些贡献为安全隐写通信提供了一个强大和适应性强的框架，为跨不同环境的安全通信提供了实用的解决方案。



## **29. Rethinking Byzantine Robustness in Federated Recommendation from Sparse Aggregation Perspective**

从稀疏聚集角度重新思考联邦推荐中的拜占庭鲁棒性 cs.CR

accepted by AAAI 2025

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2501.03301v2) [paper-pdf](http://arxiv.org/pdf/2501.03301v2)

**Authors**: Zhongjian Zhang, Mengmei Zhang, Xiao Wang, Lingjuan Lyu, Bo Yan, Junping Du, Chuan Shi

**Abstract**: To preserve user privacy in recommender systems, federated recommendation (FR) based on federated learning (FL) emerges, keeping the personal data on the local client and updating a model collaboratively. Unlike FL, FR has a unique sparse aggregation mechanism, where the embedding of each item is updated by only partial clients, instead of full clients in a dense aggregation of general FL. Recently, as an essential principle of FL, model security has received increasing attention, especially for Byzantine attacks, where malicious clients can send arbitrary updates. The problem of exploring the Byzantine robustness of FR is particularly critical since in the domains applying FR, e.g., e-commerce, malicious clients can be injected easily by registering new accounts. However, existing Byzantine works neglect the unique sparse aggregation of FR, making them unsuitable for our problem. Thus, we make the first effort to investigate Byzantine attacks on FR from the perspective of sparse aggregation, which is non-trivial: it is not clear how to define Byzantine robustness under sparse aggregations and design Byzantine attacks under limited knowledge/capability. In this paper, we reformulate the Byzantine robustness under sparse aggregation by defining the aggregation for a single item as the smallest execution unit. Then we propose a family of effective attack strategies, named Spattack, which exploit the vulnerability in sparse aggregation and are categorized along the adversary's knowledge and capability. Extensive experimental results demonstrate that Spattack can effectively prevent convergence and even break down defenses under a few malicious clients, raising alarms for securing FR systems.

摘要: 为了在推荐系统中保护用户隐私，基于联合学习的联合推荐(FR)应运而生，它将个人数据保存在本地客户端，并协作更新模型。与FL不同，FR具有独特的稀疏聚合机制，其中每个项的嵌入只由部分客户端更新，而不是由普通FL的密集聚合中的完整客户端更新。最近，作为FL的一项基本原则，模型安全性受到了越来越多的关注，特别是对于拜占庭攻击，恶意客户端可以发送任意更新。探索FR的拜占庭健壮性的问题尤其关键，因为在应用FR的域中，例如电子商务，可以通过注册新帐户轻松地注入恶意客户端。然而，现有的拜占庭著作忽略了FR独特的稀疏聚集，这使得它们不适合我们的问题。因此，我们首次尝试从稀疏聚集的角度来研究拜占庭攻击FR，这并不是一件平凡的事情：如何定义稀疏聚集下的拜占庭健壮性，以及在有限的知识/能力下设计拜占庭攻击，目前还不清楚。本文通过将单个项的聚集定义为最小执行单元，重新定义了稀疏聚集下的拜占庭健壮性。然后，我们提出了一系列有效的攻击策略Spattack，该策略利用了稀疏聚集的脆弱性，并根据对手的知识和能力进行分类。广泛的实验结果表明，Spattack能够有效地防止收敛，甚至在少数恶意客户端下破坏防御，为FR系统的安全发出警报。



## **30. Rethinking Adversarial Attacks in Reinforcement Learning from Policy Distribution Perspective**

从策略分布角度重新思考强化学习中的对抗性攻击 cs.LG

10 pages, 2 figures, 2 tables

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2501.03562v2) [paper-pdf](http://arxiv.org/pdf/2501.03562v2)

**Authors**: Tianyang Duan, Zongyuan Zhang, Zheng Lin, Yue Gao, Ling Xiong, Yong Cui, Hongbin Liang, Xianhao Chen, Heming Cui, Dong Huang

**Abstract**: Deep Reinforcement Learning (DRL) suffers from uncertainties and inaccuracies in the observation signal in realworld applications. Adversarial attack is an effective method for evaluating the robustness of DRL agents. However, existing attack methods targeting individual sampled actions have limited impacts on the overall policy distribution, particularly in continuous action spaces. To address these limitations, we propose the Distribution-Aware Projected Gradient Descent attack (DAPGD). DAPGD uses distribution similarity as the gradient perturbation input to attack the policy network, which leverages the entire policy distribution rather than relying on individual samples. We utilize the Bhattacharyya distance in DAPGD to measure policy similarity, enabling sensitive detection of subtle but critical differences between probability distributions. Our experiment results demonstrate that DAPGD achieves SOTA results compared to the baselines in three robot navigation tasks, achieving an average 22.03% higher reward drop compared to the best baseline.

摘要: 深度强化学习(DRL)在实际应用中存在观测信号的不确定性和不准确性。对抗性攻击是评价DRL代理健壮性的有效方法。然而，现有的针对单个采样动作的攻击方法对整体策略分布的影响有限，特别是在连续动作空间中。为了解决这些局限性，我们提出了分布感知投影梯度下降攻击(DAPGD)。DAPGD使用分布相似度作为梯度扰动输入来攻击策略网络，该策略网络利用整个策略分布而不是依赖于单个样本。我们利用DAPGD中的Bhattacharyya距离来度量策略相似性，从而能够敏感地检测到概率分布之间的细微但关键的差异。实验结果表明，在三个机器人导航任务中，DAPGD获得了与基线相比的SOTA结果，获得了比最佳基线平均高22.03%的奖赏下降。



## **31. Location Privacy Threats and Protections in 6G Vehicular Networks: A Comprehensive Review**

6G车载网络中的位置隐私威胁和保护：全面回顾 cs.CR

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2305.04503v2) [paper-pdf](http://arxiv.org/pdf/2305.04503v2)

**Authors**: Baihe Ma, Xu Wang, Xiaojie Lin, Yanna Jiang, Caijun Sun, Zhe Wang, Guangsheng Yu, Suirui Zhu, Ying He, Wei Ni, Ren Ping Liu

**Abstract**: Location privacy is critical in vehicular networks, where drivers' trajectories and personal information can be exposed, allowing adversaries to launch data and physical attacks that threaten drivers' safety and personal security. This survey reviews comprehensively different localization techniques, including widely used ones like sensing infrastructure-based, optical vision-based, and cellular radio-based localization, and identifies inadequately addressed location privacy concerns. We classify Location Privacy Preserving Mechanisms (LPPMs) into user-side, server-side, and user-server-interface-based, and evaluate their effectiveness. Our analysis shows that the user-server-interface-based LPPMs have received insufficient attention in the literature, despite their paramount importance in vehicular networks. Further, we examine methods for balancing data utility and privacy protection for existing LPPMs in vehicular networks and highlight emerging challenges from future upper-layer location privacy attacks, wireless technologies, and network convergences. By providing insights into the relationship between localization techniques and location privacy, and evaluating the effectiveness of different LPPMs, this survey can help inform the development of future LPPMs in vehicular networks.

摘要: 位置隐私在车载网络中至关重要，在车载网络中，司机的轨迹和个人信息可能会被暴露，从而允许对手发动威胁司机安全和人身安全的数据和物理攻击。这项调查全面回顾了不同的定位技术，包括广泛使用的基于传感基础设施的定位技术、基于光学视觉的定位技术和基于蜂窝无线电的定位技术，并确定了没有充分解决位置隐私问题的问题。我们将位置隐私保护机制(LPPM)分为基于用户端、基于服务器端和基于用户-服务器接口，并对其有效性进行了评估。我们的分析表明，基于用户-服务器接口的LPPM在文献中没有得到足够的关注，尽管它们在车载网络中非常重要。此外，我们还研究了在车载网络中平衡现有LPPM的数据效用和隐私保护的方法，并强调了未来上层位置隐私攻击、无线技术和网络融合带来的新挑战。通过深入了解定位技术和位置隐私之间的关系，以及评估不同LPPM的有效性，这项调查有助于为未来LPPM在车载网络中的发展提供信息。



## **32. Proof-of-Learning with Incentive Security**

具有激励保障的学习证明 cs.CR

20 pages, 4 figures

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2404.09005v7) [paper-pdf](http://arxiv.org/pdf/2404.09005v7)

**Authors**: Zishuo Zhao, Zhixuan Fang, Xuechao Wang, Xi Chen, Hongxu Su, Haibo Xiao, Yuan Zhou

**Abstract**: Most concurrent blockchain systems rely heavily on the Proof-of-Work (PoW) or Proof-of-Stake (PoS) mechanisms for decentralized consensus and security assurance. However, the substantial energy expenditure stemming from computationally intensive yet meaningless tasks has raised considerable concerns surrounding traditional PoW approaches, The PoS mechanism, while free of energy consumption, is subject to security and economic issues. Addressing these issues, the paradigm of Proof-of-Useful-Work (PoUW) seeks to employ challenges of practical significance as PoW, thereby imbuing energy consumption with tangible value. While previous efforts in Proof of Learning (PoL) explored the utilization of deep learning model training SGD tasks as PoUW challenges, recent research has revealed its vulnerabilities to adversarial attacks and the theoretical hardness in crafting a byzantine-secure PoL mechanism. In this paper, we introduce the concept of incentive-security that incentivizes rational provers to behave honestly for their best interest, bypassing the existing hardness to design a PoL mechanism with computational efficiency, a provable incentive-security guarantee and controllable difficulty. Particularly, our work is secure against two attacks, and also improves the computational overhead from $\Theta(1)$ to $O(\frac{\log E}{E})$. Furthermore, while most recent research assumes trusted problem providers and verifiers, our design also guarantees frontend incentive-security even when problem providers are untrusted, and verifier incentive-security that bypasses the Verifier's Dilemma. By incorporating ML training into blockchain consensus mechanisms with provable guarantees, our research not only proposes an eco-friendly solution to blockchain systems, but also provides a proposal for a completely decentralized computing power market in the new AI age.

摘要: 大多数并发区块链系统严重依赖工作证明(POW)或风险证明(POS)机制来实现去中心化共识和安全保证。然而，计算密集但无意义的任务所产生的大量能源支出引起了人们对传统POW方法的相当大的担忧，POS机制虽然没有能源消耗，但受到安全和经济问题的影响。针对这些问题，有用工作证明(POUW)范式试图将具有实际意义的挑战作为POW来使用，从而使能源消耗具有有形价值。虽然先前在学习证明(Pol)方面的努力探索了利用深度学习模型训练SGD任务作为POW挑战，但最近的研究揭示了它对对手攻击的脆弱性以及在设计拜占庭安全的POL机制方面的理论难度。本文引入激励安全的概念，激励理性的证明者为了他们的最大利益而诚实地行事，绕过现有的困难，设计了一个具有计算效率、可证明的激励安全保证和可控难度的POL机制。特别地，我们的工作对两次攻击是安全的，并且还将计算开销从$\theta(1)$提高到$O(\frac{\log E}{E})$。此外，虽然最近的研究假设可信的问题提供者和验证者，但我们的设计也保证了前端激励-安全性，即使问题提供者是不可信的，并且验证者激励-安全绕过了验证者的困境。通过将ML培训融入到具有可证明保证的区块链共识机制中，我们的研究不仅为区块链系统提出了生态友好的解决方案，而且为新AI时代完全去中心化的计算能力市场提供了建议。



## **33. Light-weight Fine-tuning Method for Defending Adversarial Noise in Pre-trained Medical Vision-Language Models**

在预训练的医学视觉语言模型中防御对抗性噪音的轻量级微调方法 cs.CV

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2407.02716v2) [paper-pdf](http://arxiv.org/pdf/2407.02716v2)

**Authors**: Xu Han, Linghao Jin, Xuezhe Ma, Xiaofeng Liu

**Abstract**: Fine-tuning pre-trained Vision-Language Models (VLMs) has shown remarkable capabilities in medical image and textual depiction synergy. Nevertheless, many pre-training datasets are restricted by patient privacy concerns, potentially containing noise that can adversely affect downstream performance. Moreover, the growing reliance on multi-modal generation exacerbates this issue because of its susceptibility to adversarial attacks. To investigate how VLMs trained on adversarial noisy data perform on downstream medical tasks, we first craft noisy upstream datasets using multi-modal adversarial attacks. Through our comprehensive analysis, we unveil that moderate noise enhances model robustness and transferability, but increasing noise levels negatively impact downstream task performance. To mitigate this issue, we propose rectify adversarial noise (RAN) framework, a recipe designed to effectively defend adversarial attacks and rectify the influence of upstream noise during fine-tuning.

摘要: 微调预训练的视觉语言模型（VLM）在医学图像和文本描述协同方面表现出了非凡的能力。然而，许多预训练数据集受到患者隐私问题的限制，可能包含可能对下游性能产生不利影响的噪音。此外，对多模式发电的日益依赖加剧了这个问题，因为它容易受到对抗攻击。为了研究在对抗性有噪数据上训练的VLM如何执行下游医疗任务，我们首先使用多模式对抗攻击来制作有噪的上游数据集。通过我们的全面分析，我们发现适度的噪音增强了模型的稳健性和可移植性，但增加噪音水平会对下游任务性能产生负面影响。为了缓解这个问题，我们提出了纠正对抗性噪音（RAN）框架，该框架旨在有效防御对抗性攻击并纠正微调期间上游噪音的影响。



## **34. Synthetic Data Privacy Metrics**

合成数据隐私收件箱 cs.LG

14 pages, 2 figures

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2501.03941v1) [paper-pdf](http://arxiv.org/pdf/2501.03941v1)

**Authors**: Amy Steier, Lipika Ramaswamy, Andre Manoel, Alexa Haushalter

**Abstract**: Recent advancements in generative AI have made it possible to create synthetic datasets that can be as accurate as real-world data for training AI models, powering statistical insights, and fostering collaboration with sensitive datasets while offering strong privacy guarantees. Effectively measuring the empirical privacy of synthetic data is an important step in the process. However, while there is a multitude of new privacy metrics being published every day, there currently is no standardization. In this paper, we review the pros and cons of popular metrics that include simulations of adversarial attacks. We also review current best practices for amending generative models to enhance the privacy of the data they create (e.g. differential privacy).

摘要: 生成性人工智能的最新进展使创建与现实世界数据一样准确的合成数据集成为可能，用于训练人工智能模型、支持统计洞察并促进与敏感数据集的协作，同时提供强大的隐私保证。有效测量合成数据的经验隐私是该过程中的重要一步。然而，虽然每天都会发布大量新的隐私指标，但目前还没有标准化。在本文中，我们回顾了包括对抗攻击模拟在内的流行指标的利弊。我们还审查了修改生成模型以增强其创建数据的隐私性（例如差异隐私）的当前最佳实践。



## **35. Not all tokens are created equal: Perplexity Attention Weighted Networks for AI generated text detection**

并非所有令牌都是平等的：用于人工智能生成文本检测的困惑注意力加权网络 cs.CL

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2501.03940v1) [paper-pdf](http://arxiv.org/pdf/2501.03940v1)

**Authors**: Pablo Miralles-González, Javier Huertas-Tato, Alejandro Martín, David Camacho

**Abstract**: The rapid advancement in large language models (LLMs) has significantly enhanced their ability to generate coherent and contextually relevant text, raising concerns about the misuse of AI-generated content and making it critical to detect it. However, the task remains challenging, particularly in unseen domains or with unfamiliar LLMs. Leveraging LLM next-token distribution outputs offers a theoretically appealing approach for detection, as they encapsulate insights from the models' extensive pre-training on diverse corpora. Despite its promise, zero-shot methods that attempt to operationalize these outputs have met with limited success. We hypothesize that one of the problems is that they use the mean to aggregate next-token distribution metrics across tokens, when some tokens are naturally easier or harder to predict and should be weighted differently. Based on this idea, we propose the Perplexity Attention Weighted Network (PAWN), which uses the last hidden states of the LLM and positions to weight the sum of a series of features based on metrics from the next-token distribution across the sequence length. Although not zero-shot, our method allows us to cache the last hidden states and next-token distribution metrics on disk, greatly reducing the training resource requirements. PAWN shows competitive and even better performance in-distribution than the strongest baselines (fine-tuned LMs) with a fraction of their trainable parameters. Our model also generalizes better to unseen domains and source models, with smaller variability in the decision boundary across distribution shifts. It is also more robust to adversarial attacks, and if the backbone has multilingual capabilities, it presents decent generalization to languages not seen during supervised training, with LLaMA3-1B reaching a mean macro-averaged F1 score of 81.46% in cross-validation with nine languages.

摘要: 大型语言模型(LLM)的快速发展极大地增强了它们生成连贯和上下文相关文本的能力，这引起了人们对滥用人工智能生成的内容的担忧，并使检测它变得至关重要。然而，这项任务仍然具有挑战性，特别是在看不见的领域或具有不熟悉的LLM的领域。利用LLM下一个令牌分发输出提供了一种理论上有吸引力的检测方法，因为它们概括了模型对不同语料库的广泛预培训的见解。尽管有希望，但试图将这些产出付诸实施的零射击方法却取得了有限的成功。我们假设其中一个问题是，当一些令牌自然更容易或更难预测，并且应该以不同的权重进行加权时，它们使用平均值来聚合跨令牌的下一令牌分发度量。虽然不是零命中率，但我们的方法允许我们在磁盘上缓存最后的隐藏状态和下一个令牌分布度量，大大减少了训练资源需求。与最强的基线(微调LMS)相比，PAWN显示出具有竞争力的分布性能，甚至比它们的可训练参数的一小部分更好。我们的模型也更好地推广到看不见的域和源模型，跨分布转变的决策边界的可变性较小。



## **36. CausalDiff: Causality-Inspired Disentanglement via Diffusion Model for Adversarial Defense**

卡西姆·分歧：通过对抗性防御的扩散模型来启发性解纠缠 cs.CV

accepted by NeurIPS 2024

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2410.23091v6) [paper-pdf](http://arxiv.org/pdf/2410.23091v6)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Quanrun Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: Despite ongoing efforts to defend neural classifiers from adversarial attacks, they remain vulnerable, especially to unseen attacks. In contrast, humans are difficult to be cheated by subtle manipulations, since we make judgments only based on essential factors. Inspired by this observation, we attempt to model label generation with essential label-causative factors and incorporate label-non-causative factors to assist data generation. For an adversarial example, we aim to discriminate the perturbations as non-causative factors and make predictions only based on the label-causative factors. Concretely, we propose a casual diffusion model (CausalDiff) that adapts diffusion models for conditional data generation and disentangles the two types of casual factors by learning towards a novel casual information bottleneck objective. Empirically, CausalDiff has significantly outperformed state-of-the-art defense methods on various unseen attacks, achieving an average robustness of 86.39% (+4.01%) on CIFAR-10, 56.25% (+3.13%) on CIFAR-100, and 82.62% (+4.93%) on GTSRB (German Traffic Sign Recognition Benchmark). The code is available at https://github.com/CAS-AISafetyBasicResearchGroup/CausalDiff.

摘要: 尽管不断努力保护神经分类器免受对手攻击，但它们仍然很脆弱，特别是面对看不见的攻击。相比之下，人类很难被微妙的操纵所欺骗，因为我们只根据基本因素做出判断。受到这一观察的启发，我们试图用基本的标签原因因素来建模标签生成，并结合标签非原因因素来辅助数据生成。对于一个对抗性的例子，我们的目标是将扰动区分为非致因因素，并仅基于标签致因因素进行预测。具体地说，我们提出了一个偶然扩散模型(CausalDiff)，该模型使扩散模型适用于条件数据生成，并通过向一个新的偶然信息瓶颈目标学习来区分这两种类型的偶然因素。经验上，CausalDiff在各种隐形攻击上的表现明显优于最先进的防御方法，在CIFAR-10上获得了86.39%(+4.01%)的平均健壮性，在CIFAR-100上获得了56.25%(+3.13%)的健壮性，在GTSRB(德国交通标志识别基准)上实现了82.62%(+4.93%)的平均健壮性。代码可在https://github.com/CAS-AISafetyBasicResearchGroup/CausalDiff.上获得



## **37. A Volumetric Approach to Privacy of Dynamical Systems**

动态系统隐私的体积方法 eess.SY

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2501.02893v2) [paper-pdf](http://arxiv.org/pdf/2501.02893v2)

**Authors**: Chuanghong Weng, Ehsan Nekouei

**Abstract**: Information-theoretic metrics, such as mutual information, have been widely used to evaluate privacy leakage in dynamic systems. However, these approaches are typically limited to stochastic systems and face computational challenges. In this paper, we introduce a novel volumetric framework for analyzing privacy in systems affected by unknown but bounded noise. Our model considers a dynamic system comprising public and private states, where an observation set of the public state is released. An adversary utilizes the observed public state to infer an uncertainty set of the private state, referred to as the inference attack. We define the evolution dynamics of these inference attacks and quantify the privacy level of the private state using the volume of its uncertainty sets. For linear scalar systems, we derive an explicit formulation of the uncertainty set. For multi-dimensional linear systems, we develop an approximate computation method leveraging interval analysis. We investigate the properties of the proposed volumetric privacy measure and demonstrate that it is bounded by the information gain derived from the observation set. Furthermore, we propose an optimization approach to designing privacy filter using randomization and linear programming based on the proposed privacy measure. The effectiveness of the optimal privacy filter design is evaluated through a production-inventory case study, illustrating its robustness against the inference attack.

摘要: 信息论度量，如互信息，已被广泛用于评估动态系统中的隐私泄漏。然而，这些方法通常局限于随机系统，并面临计算挑战。在本文中，我们介绍了一种新的体积框架，用于分析受未知但有界噪声影响的系统的隐私。我们的模型考虑了一个由公共状态和私有状态组成的动态系统，其中发布了公共状态的观测集。敌手利用观察到的公共状态来推断私有状态的不确定性集合，称为推理攻击。我们定义了这些推理攻击的演化动态，并利用其不确定性集的体积来量化私有状态的隐私级别。对于线性标量系统，我们导出了不确定集的显式表达式。对于多维线性系统，我们提出了一种利用区间分析的近似计算方法。我们研究了所提出的体积隐私度量的性质，并证明了它受来自观测集的信息增益的限制。在此基础上，提出了一种基于随机化和线性规划的隐私过滤器优化设计方法。通过一个生产-库存案例的研究，评估了最优隐私过滤器设计的有效性，说明了其对推理攻击的稳健性。



## **38. Echomix: a Strong Anonymity System with Messaging**

Echomix：一个强大的消息传递匿名系统 cs.CR

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2501.02933v2) [paper-pdf](http://arxiv.org/pdf/2501.02933v2)

**Authors**: Ewa J Infeld, David Stainton, Leif Ryge, Threebit Hacker

**Abstract**: Echomix is a practical mix network framework and a suite of associated protocols providing strong metadata privacy against realistic modern adversaries. It is distinguished from other anonymity systems by a resistance to traffic analysis by global adversaries, compromised contacts and network infrastructure, quantum decryption algorithms, and statistical and confirmation attacks typical for multi-client messaging setting. It is implemented as Katzenpost, a robust software project, and used in multiple deployed systems, and features relatively low latency and bandwidth overhead.   The contributions of this paper are: (1) Improvements on leading mix network designs, supported by rigorous analysis. These include solutions to crucial vulnerabilities to traffic analysis, malicious servers and active attacks. (2) A cryptographic group messaging protocol with strong metadata protection guarantees and reliability. (3) Hybrid post-quantum nested packet encryption.

摘要: Echomix是一个实用的混合网络框架和一套相关协议，可针对现实的现代对手提供强大的元数据隐私。它与其他匿名系统的区别在于，它能够抵抗全球对手的流量分析、受损害的联系人和网络基础设施、量子解密算法以及多客户端消息传递设置中典型的统计和确认攻击。它作为KatzenPost实施，这是一个强大的软件项目，用于多个部署的系统，并且具有相对较低的延迟和带宽负担。   本文的贡献是：（1）在严格分析的支持下，对领先的混合网络设计进行了改进。其中包括针对流量分析、恶意服务器和主动攻击的关键漏洞的解决方案。(2)具有强大元数据保护保证和可靠性的加密群组消息协议。(3)混合后量子嵌套数据包加密。



## **39. Graph Neural Backdoor: Fundamentals, Methodologies, Applications, and Future Directions**

图形神经后门：基础知识、方法论、应用和未来方向 cs.LG

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2406.10573v2) [paper-pdf](http://arxiv.org/pdf/2406.10573v2)

**Authors**: Xiao Yang, Gaolei Li, Jianhua Li

**Abstract**: Graph Neural Networks (GNNs) have significantly advanced various downstream graph-relevant tasks, encompassing recommender systems, molecular structure prediction, social media analysis, etc. Despite the boosts of GNN, recent research has empirically demonstrated its potential vulnerability to backdoor attacks, wherein adversaries employ triggers to poison input samples, inducing GNN to adversary-premeditated malicious outputs. This is typically due to the controlled training process, or the deployment of untrusted models, such as delegating model training to third-party service, leveraging external training sets, and employing pre-trained models from online sources. Although there's an ongoing increase in research on GNN backdoors, comprehensive investigation into this field is lacking. To bridge this gap, we propose the first survey dedicated to GNN backdoors. We begin by outlining the fundamental definition of GNN, followed by the detailed summarization and categorization of current GNN backdoor attacks and defenses based on their technical characteristics and application scenarios. Subsequently, the analysis of the applicability and use cases of GNN backdoors is undertaken. Finally, the exploration of potential research directions of GNN backdoors is presented. This survey aims to explore the principles of graph backdoors, provide insights to defenders, and promote future security research.

摘要: 图神经网络(GNN)已经显著推进了各种下游与图相关的任务，包括推荐系统、分子结构预测、社交媒体分析等。尽管GNN得到了提升，但最近的研究经验表明，它对后门攻击具有潜在的脆弱性，即攻击者使用触发器来毒化输入样本，诱导GNN进行攻击者预谋的恶意输出。这通常是由于受控的培训过程或不受信任的模型的部署，例如将模型培训委托给第三方服务、利用外部培训集以及使用来自在线来源的预先培训的模型。尽管对GNN后门的研究在不断增加，但对这一领域的全面调查还很缺乏。为了弥补这一差距，我们建议对GNN后门进行第一次调查。我们首先概述了GNN的基本定义，然后根据其技术特征和应用场景对当前GNN后门攻击和防御进行了详细的总结和分类。随后，对GNN后门的适用性和使用案例进行了分析。最后，对GNN后门的潜在研究方向进行了展望。这项调查旨在探索图形后门的原理，为防御者提供见解，并促进未来的安全研究。



## **40. Unraveling Responsiveness of Chained BFT Consensus with Network Delay**

解开具有网络延迟的连锁BFT共识的响应性 cs.DC

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2501.03695v1) [paper-pdf](http://arxiv.org/pdf/2501.03695v1)

**Authors**: Yining Tang, Qihang Luo, Runchao Han, Jianyu Niu, Chen Feng, Yinqian Zhang

**Abstract**: With the advancement of blockchain technology, chained Byzantine Fault Tolerant (BFT) protocols have been increasingly adopted in practical systems, making their performance a crucial aspect of the study. In this paper, we introduce a unified framework utilizing Markov Decision Processes (MDP) to model and assess the performance of three prominent chained BFT protocols. Our framework effectively captures complex adversarial behaviors, focusing on two key performance metrics: chain growth and commitment rate. We implement the optimal attack strategies obtained from MDP analysis on an existing evaluation platform for chained BFT protocols and conduct extensive experiments under various settings to validate our theoretical results. Through rigorous theoretical analysis and thorough practical experiments, we provide an in-depth evaluation of chained BFT protocols under diverse attack scenarios, uncovering optimal attack strategies. Contrary to conventional belief, our findings reveal that while responsiveness can enhance performance, it is not universally beneficial across all scenarios. This work not only deepens our understanding of chained BFT protocols, but also offers valuable insights and analytical tools that can inform the design of more robust and efficient protocols.

摘要: 随着区块链技术的进步，链式拜占庭容错(BFT)协议越来越多地被应用到实际系统中，其性能成为研究的一个关键方面。本文介绍了一个利用马尔可夫决策过程(MDP)对三种主要的链式BFT协议进行建模和性能评估的统一框架。我们的框架有效地捕获了复杂的对抗性行为，重点关注两个关键的性能指标：链增长和承诺率。我们在已有的链式BFT协议评估平台上实现了从MDP分析得到的最优攻击策略，并在不同的环境下进行了大量的实验来验证我们的理论结果。通过严格的理论分析和深入的实践实验，我们对链式BFT协议在不同攻击场景下的性能进行了深入的评估，发现了最优的攻击策略。与传统的看法相反，我们的研究结果表明，尽管响应能力可以提高绩效，但并不是在所有情况下都是有益的。这项工作不仅加深了我们对链式BFT协议的理解，而且提供了有价值的见解和分析工具，可以为设计更健壮和更高效的协议提供信息。



## **41. Transferable Adversarial Examples with Bayes Approach**

使用Bayes方法的可转移对抗示例 cs.LG

Accepted in AsiaCCS'25

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2208.06538v2) [paper-pdf](http://arxiv.org/pdf/2208.06538v2)

**Authors**: Mingyuan Fan, Cen Chen, Wenmeng Zhou, Yinggui Wang

**Abstract**: The vulnerability of deep neural networks (DNNs) to black-box adversarial attacks is one of the most heated topics in trustworthy AI. In such attacks, the attackers operate without any insider knowledge of the model, making the cross-model transferability of adversarial examples critical. Despite the potential for adversarial examples to be effective across various models, it has been observed that adversarial examples that are specifically crafted for a specific model often exhibit poor transferability. In this paper, we explore the transferability of adversarial examples via the lens of Bayesian approach. Specifically, we leverage Bayesian approach to probe the transferability and then study what constitutes a transferability-promoting prior. Following this, we design two concrete transferability-promoting priors, along with an adaptive dynamic weighting strategy for instances sampled from these priors. Employing these techniques, we present BayAtk. Extensive experiments illustrate the significant effectiveness of BayAtk in crafting more transferable adversarial examples against both undefended and defended black-box models compared to existing state-of-the-art attacks.

摘要: 深度神经网络(DNN)对黑盒攻击的脆弱性是可信人工智能领域最热门的研究课题之一。在这种攻击中，攻击者在没有任何模型内部知识的情况下操作，这使得对抗性例子的跨模型可转移性至关重要。尽管对抗性例子有可能在各种模型中有效，但已经观察到，专门为特定模型制作的对抗性例子往往表现出较差的可转移性。在这篇文章中，我们通过贝叶斯方法的镜头来探讨对抗性例子的可转移性。具体地说，我们利用贝叶斯方法来探索可转让性，然后研究什么构成可转移性促进优先。在此基础上，我们设计了两个具体的可转移性提升先验，并针对从这些先验中抽取的实例设计了一种自适应的动态加权策略。利用这些技术，我们介绍了BayAtk。广泛的实验表明，与现有的最先进的攻击相比，BayAtk在针对无防御和有防御的黑盒模型创建更可转移的对手示例方面具有显著的有效性。



## **42. PhishAgent: A Robust Multimodal Agent for Phishing Webpage Detection**

PhishAgent：一种用于网络钓鱼网页检测的鲁棒多模式代理 cs.CR

Accepted at AAAI 2025

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2408.10738v2) [paper-pdf](http://arxiv.org/pdf/2408.10738v2)

**Authors**: Tri Cao, Chengyu Huang, Yuexin Li, Huilin Wang, Amy He, Nay Oo, Bryan Hooi

**Abstract**: Phishing attacks are a major threat to online security, exploiting user vulnerabilities to steal sensitive information. Various methods have been developed to counteract phishing, each with varying levels of accuracy, but they also face notable limitations. In this study, we introduce PhishAgent, a multimodal agent that combines a wide range of tools, integrating both online and offline knowledge bases with Multimodal Large Language Models (MLLMs). This combination leads to broader brand coverage, which enhances brand recognition and recall. Furthermore, we propose a multimodal information retrieval framework designed to extract the relevant top k items from offline knowledge bases, using available information from a webpage, including logos and HTML. Our empirical results, based on three real-world datasets, demonstrate that the proposed framework significantly enhances detection accuracy and reduces both false positives and false negatives, while maintaining model efficiency. Additionally, PhishAgent shows strong resilience against various types of adversarial attacks.

摘要: 网络钓鱼攻击是在线安全的主要威胁，利用用户漏洞窃取敏感信息。已经开发了各种方法来对抗网络钓鱼，每一种方法的精确度都不同，但它们也面临着显著的局限性。在本研究中，我们介绍了PhishAgent，一个结合了广泛工具的多通道代理，将线上和线下知识库与多通道大语言模型(MLLMS)相结合。这一组合导致了更广泛的品牌覆盖，从而提高了品牌认知度和召回率。此外，我们还提出了一个多通道信息检索框架，该框架利用网页中的可用信息，包括标识和超文本标记语言，从离线知识库中提取相关的前k个条目。基于三个真实数据集的实验结果表明，该框架在保持模型效率的同时，显著提高了检测准确率，减少了误报和漏报。此外，PhishAgent对各种类型的对抗性攻击表现出很强的韧性。



## **43. ChatBug: A Common Vulnerability of Aligned LLMs Induced by Chat Templates**

ChatBug：聊天模板引发的对齐LLM的常见漏洞 cs.CR

This paper is accepted to AAAI 2025

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2406.12935v2) [paper-pdf](http://arxiv.org/pdf/2406.12935v2)

**Authors**: Fengqing Jiang, Zhangchen Xu, Luyao Niu, Bill Yuchen Lin, Radha Poovendran

**Abstract**: Large language models (LLMs) are expected to follow instructions from users and engage in conversations. Techniques to enhance LLMs' instruction-following capabilities typically fine-tune them using data structured according to a predefined chat template. Although chat templates are shown to be effective in optimizing LLM performance, their impact on safety alignment of LLMs has been less understood, which is crucial for deploying LLMs safely at scale.   In this paper, we investigate how chat templates affect safety alignment of LLMs. We identify a common vulnerability, named ChatBug, that is introduced by chat templates. Our key insight to identify ChatBug is that the chat templates provide a rigid format that need to be followed by LLMs, but not by users. Hence, a malicious user may not necessarily follow the chat template when prompting LLMs. Instead, malicious users could leverage their knowledge of the chat template and accordingly craft their prompts to bypass safety alignments of LLMs. We develop two attacks to exploit the ChatBug vulnerability. We demonstrate that a malicious user can exploit the ChatBug vulnerability of eight state-of-the-art (SOTA) LLMs and effectively elicit unintended responses from these models. Moreover, we show that ChatBug can be exploited by existing jailbreak attacks to enhance their attack success rates. We investigate potential countermeasures to ChatBug. Our results show that while adversarial training effectively mitigates the ChatBug vulnerability, the victim model incurs significant performance degradation. These results highlight the trade-off between safety alignment and helpfulness. Developing new methods for instruction tuning to balance this trade-off is an open and critical direction for future research

摘要: 大型语言模型(LLM)应该遵循用户的指示并参与对话。增强LLMS的指令遵循能力的技术通常使用根据预定义的聊天模板构造的数据对其进行微调。尽管聊天模板被证明在优化LLM性能方面是有效的，但人们对它们对LLM安全调整的影响知之甚少，这对于安全地大规模部署LLMS至关重要。在本文中，我们研究了聊天模板如何影响LLMS的安全对齐。我们发现了聊天模板引入的一个名为ChatBug的常见漏洞。我们识别ChatBug的关键洞察力是，聊天模板提供了一种严格的格式，需要LLMS遵循，而不是用户。因此，恶意用户在提示LLMS时可能不一定遵循聊天模板。相反，恶意用户可以利用他们对聊天模板的了解，并相应地精心编制他们的提示，以绕过LLMS的安全对齐。我们开发了两个攻击来利用ChatBug漏洞。我们演示了恶意用户可以利用8个最先进的(SOTA)LLM的ChatBug漏洞，并有效地从这些模型中引发意外响应。此外，我们发现ChatBug可以被现有的越狱攻击所利用，以提高他们的攻击成功率。我们调查了针对ChatBug的潜在对策。我们的结果表明，虽然对抗性训练有效地缓解了ChatBug漏洞，但受害者模型导致了显著的性能下降。这些结果突显了安全性调整和帮助之间的权衡。开发新的教学调整方法来平衡这种权衡是未来研究的一个开放和关键的方向



## **44. Countering Backdoor Attacks in Image Recognition: A Survey and Evaluation of Mitigation Strategies**

对抗图像识别中的后门攻击：缓解策略的调查和评估 cs.CR

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2411.11200v2) [paper-pdf](http://arxiv.org/pdf/2411.11200v2)

**Authors**: Kealan Dunnett, Reza Arablouei, Dimity Miller, Volkan Dedeoglu, Raja Jurdak

**Abstract**: The widespread adoption of deep learning across various industries has introduced substantial challenges, particularly in terms of model explainability and security. The inherent complexity of deep learning models, while contributing to their effectiveness, also renders them susceptible to adversarial attacks. Among these, backdoor attacks are especially concerning, as they involve surreptitiously embedding specific triggers within training data, causing the model to exhibit aberrant behavior when presented with input containing the triggers. Such attacks often exploit vulnerabilities in outsourced processes, compromising model integrity without affecting performance on clean (trigger-free) input data. In this paper, we present a comprehensive review of existing mitigation strategies designed to counter backdoor attacks in image recognition. We provide an in-depth analysis of the theoretical foundations, practical efficacy, and limitations of these approaches. In addition, we conduct an extensive benchmarking of sixteen state-of-the-art approaches against eight distinct backdoor attacks, utilizing three datasets, four model architectures, and three poisoning ratios. Our results, derived from 122,236 individual experiments, indicate that while many approaches provide some level of protection, their performance can vary considerably. Furthermore, when compared to two seminal approaches, most newer approaches do not demonstrate substantial improvements in overall performance or consistency across diverse settings. Drawing from these findings, we propose potential directions for developing more effective and generalizable defensive mechanisms in the future.

摘要: 深度学习在各个行业的广泛采用带来了巨大的挑战，特别是在模型的可解释性和安全性方面。深度学习模型固有的复杂性，虽然有助于它们的有效性，但也使它们容易受到对手的攻击。其中，后门攻击尤其令人担忧，因为它们涉及在训练数据中秘密嵌入特定触发器，导致在输入包含触发器的输入时导致模型表现出异常行为。此类攻击通常利用外包流程中的漏洞，在不影响干净(无触发器)输入数据的性能的情况下损害模型完整性。在这篇文章中，我们提出了一个全面的审查现有的缓解策略，旨在对抗后门攻击的图像识别。我们对这些方法的理论基础、实践有效性和局限性进行了深入分析。此外，我们利用三个数据集、四个模型体系结构和三个投毒率，对针对八种不同后门攻击的16种最先进方法进行了广泛的基准测试。我们的结果来自122,236个单独的实验，表明虽然许多方法提供了一定程度的保护，但它们的性能可能会有很大的差异。此外，与两种开创性的方法相比，大多数较新的方法在总体性能或跨不同环境的一致性方面没有显示出实质性的改进。根据这些发现，我们提出了未来发展更有效和更具普遍性的防御机制的潜在方向。



## **45. Adversarial Vulnerabilities in Large Language Models for Time Series Forecasting**

时间序列预测大型语言模型中的对抗漏洞 cs.LG

11 pages, 5 figures

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2412.08099v2) [paper-pdf](http://arxiv.org/pdf/2412.08099v2)

**Authors**: Fuqiang Liu, Sicong Jiang, Luis Miranda-Moreno, Seongjin Choi, Lijun Sun

**Abstract**: Large Language Models (LLMs) have recently demonstrated significant potential in the field of time series forecasting, offering impressive capabilities in handling complex temporal data. However, their robustness and reliability in real-world applications remain under-explored, particularly concerning their susceptibility to adversarial attacks. In this paper, we introduce a targeted adversarial attack framework for LLM-based time series forecasting. By employing both gradient-free and black-box optimization methods, we generate minimal yet highly effective perturbations that significantly degrade the forecasting accuracy across multiple datasets and LLM architectures. Our experiments, which include models like TimeGPT and LLM-Time with GPT-3.5, GPT-4, LLaMa, and Mistral, show that adversarial attacks lead to much more severe performance degradation than random noise, and demonstrate the broad effectiveness of our attacks across different LLMs. The results underscore the critical vulnerabilities of LLMs in time series forecasting, highlighting the need for robust defense mechanisms to ensure their reliable deployment in practical applications.

摘要: 大型语言模型最近在时间序列预测领域显示出巨大的潜力，在处理复杂的时间数据方面提供了令人印象深刻的能力。然而，它们在实际应用中的健壮性和可靠性仍然没有得到充分的研究，特别是关于它们对对手攻击的敏感性。本文提出了一种基于LLM的时间序列预测的对抗性攻击框架。通过使用无梯度和黑盒优化方法，我们产生了最小但高效的扰动，这些扰动显著降低了跨多个数据集和LLM体系结构的预测精度。我们的实验，包括使用GPT-3.5、GPT-4、LLAMA和Mistral的TimeGPT和LLM-Time模型，表明对抗性攻击导致的性能降级比随机噪声严重得多，并证明了我们的攻击在不同LLM上的广泛有效性。这些结果强调了低层管理在时间序列预测中的关键弱点，强调了需要强大的防御机制来确保其在实际应用中的可靠部署。



## **46. When Should Selfish Miners Double-Spend?**

自私的矿工何时应该加倍花钱？ cs.CR

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2501.03227v1) [paper-pdf](http://arxiv.org/pdf/2501.03227v1)

**Authors**: Mustafa Doger, Sennur Ulukus

**Abstract**: Although, both double-spending and selfish-mining attacks have been extensively studied since the ``Bitcoin'' whitepaper of Nakamoto and the ``majority is not enough'' paper of Eyal and Sirer, there has been no rigorous stochastic analysis of an attack that combines the two, except for the complicated MDP models. In this paper, we first combine stubborn and selfish mining attacks, i.e., construct a strategy where the attacker acts stubborn until its private branch reaches a certain length and then switches to act selfish. We provide the optimal stubbornness for each parameter regime. Next, we provide the maximum stubbornness that is still more profitable than honest mining and argue a connection between the level of stubbornness and the $k$-confirmation rule. We show that, at each attack cycle, if the level of stubbornness is higher than $k$, there is a risk of double-spending which comes at no-cost to the adversary. The result can be seen as a guide for picking $k$ in the $k$-confirmation rule in a blockchain design. At each cycle, for a given stubbornness level, we rigorously formulate how great the risk of double-spending is. We provide the minimum double-spend value needed for an attack to be profitable in the regimes where the scheme is less profitable than honest mining. We further modify the attack in the stubborn regime in order to conceal the attack and increase the double-spending probability. Finally, we evaluate the results and provide the optimal and the maximum stubbornness levels for each parameter regime as well as the revenue. As a case study, with Bitcoin's $k=6$ block confirmation rule, we evaluate the revenue and double-spending risk of the attacks for each pool parameter.

摘要: 尽管自从Nakamoto的《比特币》白皮书和EYAL和Sirer的《多数是不够的》白皮书以来，重复支出攻击和自私挖掘攻击都得到了广泛的研究，但除了复杂的MDP模型外，还没有对结合这两者的攻击进行严格的随机分析。在本文中，我们首先将顽固挖掘攻击和自私挖掘攻击相结合，即构造了一种策略，即攻击者在其私有分支达到一定长度时顽固行事，然后切换为自私行为。我们给出了每个参数区域的最优顽固性。接下来，我们提供了比诚实挖掘更有利可图的最大顽固，并论证了顽固程度与$k$确认规则之间的联系。我们证明，在每个攻击周期，如果顽固程度高于$k$，则存在重复支出的风险，这对对手来说是免费的。这一结果可以被视为区块链设计中在$k$-确认规则中挑选$k$的指南。在每个周期，对于给定的固执水平，我们都会严格地阐述重复支出的风险有多大。我们提供了攻击在利润低于诚实开采的政权中盈利所需的最低双重支出价值。我们进一步修改了顽固政权中的攻击，以隐藏攻击，增加重复支出的概率。最后，我们对结果进行了评估，并给出了每个参数机制的最优和最大顽固性水平以及收益。作为一个案例，在比特币的$k=6$块确认规则下，我们对每个池参数的攻击的收入和重复支出风险进行了评估。



## **47. The Robustness of Spiking Neural Networks in Federated Learning with Compression Against Non-omniscient Byzantine Attacks**

尖峰神经网络在压缩联邦学习中针对非无所不知的拜占庭攻击的鲁棒性 cs.CR

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2501.03306v1) [paper-pdf](http://arxiv.org/pdf/2501.03306v1)

**Authors**: Manh V. Nguyen, Liang Zhao, Bobin Deng, Shaoen Wu

**Abstract**: Spiking Neural Networks (SNNs), which offer exceptional energy efficiency for inference, and Federated Learning (FL), which offers privacy-preserving distributed training, is a rising area of interest that highly beneficial towards Internet of Things (IoT) devices. Despite this, research that tackles Byzantine attacks and bandwidth limitation in FL-SNNs, both poses significant threats on model convergence and training times, still remains largely unexplored. Going beyond proposing a solution for both of these problems, in this work we highlight the dual benefits of FL-SNNs, against non-omniscient Byzantine adversaries (ones that restrict attackers access to local clients datasets), and greater communication efficiency, over FL-ANNs. Specifically, we discovered that a simple integration of Top-\k{appa} sparsification into the FL apparatus can help leverage the advantages of the SNN models in both greatly reducing bandwidth usage and significantly boosting the robustness of FL training against non-omniscient Byzantine adversaries. Most notably, we saw a massive improvement of roughly 40% accuracy gain in FL-SNNs training under the lethal MinMax attack

摘要: 尖峰神经网络(SNN)为推理提供了出色的能量效率，联邦学习(FL)提供隐私保护的分布式训练，是物联网(IoT)设备高度有益的新兴兴趣领域。尽管如此，解决FL-SNN中的拜占庭攻击和带宽限制的研究仍然在很大程度上仍未被探索，这两个问题都对模型收敛和训练时间构成了重大威胁。除了为这两个问题提出解决方案外，在这项工作中，我们还强调了FL-SNN在对抗非无所不知的拜占庭对手(限制攻击者访问本地客户端数据集)和比FL-ANN更高的通信效率方面的双重好处。具体地说，我们发现，将Top-k{appa}稀疏算法简单地集成到FL设备中，可以帮助利用SNN模型的优势，既可以极大地减少带宽占用，又可以显著提高FL训练对非全知拜占庭对手的健壮性。最值得注意的是，我们看到在致命的MinMax攻击下，FL-SNN训练的准确率提高了大约40%



## **48. Leader Rotation Is Not Enough: Scrutinizing Leadership Democracy of Chained BFT Consensus**

领导人轮换还不够：审视BFT共识的领导民主 cs.CR

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2501.02970v1) [paper-pdf](http://arxiv.org/pdf/2501.02970v1)

**Authors**: Yining Tang, Runchao Han, Jianyu Niu, Chen Feng, Yinqian Zhang

**Abstract**: With the growing popularity of blockchains, modern chained BFT protocols combining chaining and leader rotation to obtain better efficiency and leadership democracy have received increasing interest. Although the efficiency provisions of chained BFT protocols have been thoroughly analyzed, the leadership democracy has received little attention in prior work. In this paper, we scrutinize the leadership democracy of four representative chained BFT protocols, especially under attack. To this end, we propose a unified framework with two evaluation metrics, i.e., chain quality and censorship resilience, and quantitatively analyze chosen protocols through the Markov Decision Process (MDP). With this framework, we further examine the impact of two key components, i.e., voting pattern and leader rotation on leadership democracy. Our results indicate that leader rotation is not enough to provide the leadership democracy guarantee; an adversary could utilize the design, e.g., voting pattern, to deteriorate the leadership democracy significantly. Based on the analysis results, we propose customized countermeasures for three evaluated protocols to improve their leadership democracy with only slight protocol overhead and no change of consensus rules. We also discuss future directions toward building more democratic chained BFT protocols.

摘要: 随着区块链的日益普及，现代链式BFT协议结合了链接和领导者轮换以获得更好的效率和领导民主，受到了越来越多的关注。虽然链式BFT协议的效率条款已经被彻底分析，但在以前的工作中，领导层民主几乎没有受到关注。在这篇文章中，我们仔细检查了四个有代表性的链式BFT协议的领导民主，特别是在攻击下。为此，我们提出了一个具有两个评价指标的统一框架，即链质量和审查韧性，并通过马尔可夫决策过程(MDP)对所选协议进行了定量分析。在这个框架下，我们进一步考察了投票模式和领导轮换这两个关键因素对领导民主的影响。研究结果表明，领导轮换不足以为领导民主提供保障；对手可以利用投票模式等设计来显著恶化领导民主。基于分析结果，我们对三个被评估的协议提出了定制的对策，以提高它们的领导民主，而协议开销很小，共识规则不变。我们还讨论了构建更民主的链式BFT协议的未来方向。



## **49. Seeing the Whole in the Parts in Self-Supervised Representation Learning**

自我监督的表象学习中的整体 cs.LG

20 pages

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2501.02860v1) [paper-pdf](http://arxiv.org/pdf/2501.02860v1)

**Authors**: Arthur Aubret, Céline Teulière, Jochen Triesch

**Abstract**: Recent successes in self-supervised learning (SSL) model spatial co-occurrences of visual features either by masking portions of an image or by aggressively cropping it. Here, we propose a new way to model spatial co-occurrences by aligning local representations (before pooling) with a global image representation. We present CO-SSL, a family of instance discrimination methods and show that it outperforms previous methods on several datasets, including ImageNet-1K where it achieves 71.5% of Top-1 accuracy with 100 pre-training epochs. CO-SSL is also more robust to noise corruption, internal corruption, small adversarial attacks, and large training crop sizes. Our analysis further indicates that CO-SSL learns highly redundant local representations, which offers an explanation for its robustness. Overall, our work suggests that aligning local and global representations may be a powerful principle of unsupervised category learning.

摘要: 自我监督学习（SSL）最近取得的成功通过掩蔽图像的部分或积极裁剪图像来建模视觉特征的空间共现。在这里，我们提出了一种通过将局部表示（池化之前）与全局图像表示对齐来建模空间共现的新方法。我们介绍了CO-SSL，这是一系列实例区分方法，并表明它在多个数据集上优于以前的方法，包括ImageNet-1 K，它在100个预训练时期内实现了Top-1的71.5%准确率。CO-SSL对噪音腐败、内部腐败、小型对抗攻击和大型训练作物规模也更稳健。我们的分析进一步表明，CO-SSL学习高度冗余的本地表示，这为其稳健性提供了解释。总体而言，我们的工作表明，将局部和全局表示对齐可能是无监督类别学习的一个强大原则。



## **50. MBTSAD: Mitigating Backdoors in Language Models Based on Token Splitting and Attention Distillation**

MBTSAD：缓解基于令牌分裂和注意力蒸馏的语言模型中的后门 cs.CR

Accepted by ICTAI 2024

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2501.02754v1) [paper-pdf](http://arxiv.org/pdf/2501.02754v1)

**Authors**: Yidong Ding, Jiafei Niu, Ping Yi

**Abstract**: In recent years, attention-based models have excelled across various domains but remain vulnerable to backdoor attacks, often from downloading or fine-tuning on poisoned datasets. Many current methods to mitigate backdoors in NLP models rely on the pre-trained (unfine-tuned) weights, but these methods fail in scenarios where the pre-trained weights are not available. In this work, we propose MBTSAD, which can mitigate backdoors in the language model by utilizing only a small subset of clean data and does not require pre-trained weights. Specifically, MBTSAD retrains the backdoored model on a dataset generated by token splitting. Then MBTSAD leverages attention distillation, the retrained model is the teacher model, and the original backdoored model is the student model. Experimental results demonstrate that MBTSAD achieves comparable backdoor mitigation performance as the methods based on pre-trained weights while maintaining the performance on clean data. MBTSAD does not rely on pre-trained weights, enhancing its utility in scenarios where pre-trained weights are inaccessible. In addition, we simplify the min-max problem of adversarial training and visualize text representations to discover that the token splitting method in MBTSAD's first step generates Out-of-Distribution (OOD) data, leading the model to learn more generalized features and eliminate backdoor patterns.

摘要: 近年来，基于注意力的模型在各个领域都表现出色，但仍然容易受到后门攻击，通常是通过下载或对有毒数据集进行微调。当前许多用于缓解NLP模型中后门的方法依赖于预先训练(未微调)的权重，但这些方法在预先训练的权重不可用的情况下失败。在这项工作中，我们提出了MBTSAD，它可以通过只利用一小部分干净的数据来减少语言模型中的后门，并且不需要预先训练的权重。具体地说，MBTSAD在令牌拆分生成的数据集上重新训练回溯模型。然后MBTSAD利用注意力蒸馏，再训练的模型是教师模型，原始的回溯模型是学生模型。实验结果表明，MBTSAD在保持在干净数据上的性能的同时，获得了与基于预训练权重的方法相当的后门抑制性能。MBTSAD不依赖于预先训练的权重，在无法访问预先训练的权重的情况下增强了它的实用性。此外，我们简化了对抗性训练的最小-最大问题，并将文本表示可视化，发现MBTSAD第一步中的令牌拆分方法产生了超出分布(OOD)的数据，从而使模型学习到更普遍的特征并消除了后门模式。



