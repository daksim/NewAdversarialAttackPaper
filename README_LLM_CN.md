# Latest Large Language Model Attack Papers
**update at 2024-01-05 09:59:27**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Mining Temporal Attack Patterns from Cyberthreat Intelligence Reports**

从网络威胁情报报告中挖掘时态攻击模式 cs.CR

A modified version of this pre-print is submitted to IEEE  Transactions on Software Engineering, and is under review

**SubmitDate**: 2024-01-03    [abs](http://arxiv.org/abs/2401.01883v1) [paper-pdf](http://arxiv.org/pdf/2401.01883v1)

**Authors**: Md Rayhanur Rahman, Brandon Wroblewski, Quinn Matthews, Brantley Morgan, Tim Menzies, Laurie Williams

**Abstract**: Defending from cyberattacks requires practitioners to operate on high-level adversary behavior. Cyberthreat intelligence (CTI) reports on past cyberattack incidents describe the chain of malicious actions with respect to time. To avoid repeating cyberattack incidents, practitioners must proactively identify and defend against recurring chain of actions - which we refer to as temporal attack patterns. Automatically mining the patterns among actions provides structured and actionable information on the adversary behavior of past cyberattacks. The goal of this paper is to aid security practitioners in prioritizing and proactive defense against cyberattacks by mining temporal attack patterns from cyberthreat intelligence reports. To this end, we propose ChronoCTI, an automated pipeline for mining temporal attack patterns from cyberthreat intelligence (CTI) reports of past cyberattacks. To construct ChronoCTI, we build the ground truth dataset of temporal attack patterns and apply state-of-the-art large language models, natural language processing, and machine learning techniques. We apply ChronoCTI on a set of 713 CTI reports, where we identify 124 temporal attack patterns - which we categorize into nine pattern categories. We identify that the most prevalent pattern category is to trick victim users into executing malicious code to initiate the attack, followed by bypassing the anti-malware system in the victim network. Based on the observed patterns, we advocate organizations to train users about cybersecurity best practices, introduce immutable operating systems with limited functionalities, and enforce multi-user authentications. Moreover, we advocate practitioners to leverage the automated mining capability of ChronoCTI and design countermeasures against the recurring attack patterns.

摘要: 防御网络攻击需要从业者对高级别的对手行为进行操作。关于过去网络攻击事件的网络威胁情报(CTI)报告描述了与时间相关的恶意行动链。为了避免重复网络攻击事件，从业人员必须主动识别和防御反复出现的动作链--我们将其称为临时攻击模式。自动挖掘动作之间的模式提供了关于过去网络攻击对手行为的结构化和可操作的信息。本文的目的是通过从网络威胁情报报告中挖掘时态攻击模式，帮助安全从业者确定优先顺序并主动防御网络攻击。为此，我们提出了ChronoCTI，这是一种从过去网络攻击的网络威胁情报(CTI)报告中挖掘时态攻击模式的自动化管道。为了构建ChronoCTI，我们建立了时间攻击模式的地面事实数据集，并应用了最先进的大型语言模型、自然语言处理和机器学习技术。我们在一组713个CTI报告上应用了ChronoCTI，其中我们识别了124个临时攻击模式-我们将其分类为9个模式类别。我们发现，最普遍的模式类别是诱骗受害者用户执行恶意代码来发起攻击，然后绕过受害者网络中的反恶意软件系统。根据观察到的模式，我们倡导组织对用户进行网络安全最佳实践方面的培训，引入具有有限功能的不变操作系统，并强制实施多用户身份验证。此外，我们提倡实践者利用ChronoCTI的自动挖掘能力，并设计针对反复出现的攻击模式的对策。



## **2. Safety and Performance, Why Not Both? Bi-Objective Optimized Model Compression against Heterogeneous Attacks Toward AI Software Deployment**

安全和性能，为什么不能两者兼而有之呢？针对AI软件部署异构性攻击的双目标优化模型压缩 cs.AI

Accepted by IEEE Transactions on Software Engineering (TSE).  Camera-ready Version. arXiv admin note: substantial text overlap with  arXiv:2208.05969

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2401.00996v1) [paper-pdf](http://arxiv.org/pdf/2401.00996v1)

**Authors**: Jie Zhu, Leye Wang, Xiao Han, Anmin Liu, Tao Xie

**Abstract**: The size of deep learning models in artificial intelligence (AI) software is increasing rapidly, hindering the large-scale deployment on resource-restricted devices (e.g., smartphones). To mitigate this issue, AI software compression plays a crucial role, which aims to compress model size while keeping high performance. However, the intrinsic defects in a big model may be inherited by the compressed one. Such defects may be easily leveraged by adversaries, since a compressed model is usually deployed in a large number of devices without adequate protection. In this article, we aim to address the safe model compression problem from the perspective of safety-performance co-optimization. Specifically, inspired by the test-driven development (TDD) paradigm in software engineering, we propose a test-driven sparse training framework called SafeCompress. By simulating the attack mechanism as safety testing, SafeCompress can automatically compress a big model to a small one following the dynamic sparse training paradigm. Then, considering two kinds of representative and heterogeneous attack mechanisms, i.e., black-box membership inference attack and white-box membership inference attack, we develop two concrete instances called BMIA-SafeCompress and WMIA-SafeCompress. Further, we implement another instance called MMIA-SafeCompress by extending SafeCompress to defend against the occasion when adversaries conduct black-box and white-box membership inference attacks simultaneously. We conduct extensive experiments on five datasets for both computer vision and natural language processing tasks. The results show the effectiveness and generalizability of our framework. We also discuss how to adapt SafeCompress to other attacks besides membership inference attack, demonstrating the flexibility of SafeCompress.

摘要: 人工智能（AI）软件中深度学习模型的规模正在迅速增加，阻碍了在资源受限设备（例如，智能手机）。为了缓解这个问题，人工智能软件压缩起着至关重要的作用，其目的是在保持高性能的同时压缩模型大小。然而，大模型中的固有缺陷可能被压缩模型继承。这样的缺陷可能很容易被对手利用，因为压缩模型通常部署在大量设备中，而没有足够的保护。在这篇文章中，我们的目标是解决安全模型压缩问题的安全性能协同优化的角度。具体来说，受软件工程中测试驱动开发（TDD）范式的启发，我们提出了一个测试驱动的稀疏训练框架，称为SafeCompress。通过模拟攻击机制作为安全测试，SafeCompress可以按照动态稀疏训练范式自动将大模型压缩为小模型。然后，考虑两种具有代表性的异构攻击机制，即，黑盒成员推理攻击和白盒成员推理攻击，我们开发了两个具体的实例BMIA-SafeCompress和WMIA-SafeCompress。此外，我们实现了另一个名为MMIA-SafeCompress的实例，通过扩展SafeCompress来防御对手同时进行黑盒和白盒成员推理攻击的情况。我们对计算机视觉和自然语言处理任务的五个数据集进行了广泛的实验。结果表明，我们的框架的有效性和普遍性。我们还讨论了如何使SafeCompress适应除成员推断攻击之外的其他攻击，展示了SafeCompress的灵活性。



## **3. Detection and Defense Against Prominent Attacks on Preconditioned LLM-Integrated Virtual Assistants**

对预置LLM集成虚拟助理的显著攻击的检测和防御 cs.CR

Accepted to be published in the Proceedings of the 10th IEEE CSDE  2023, the Asia-Pacific Conference on Computer Science and Data Engineering  2023

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2401.00994v1) [paper-pdf](http://arxiv.org/pdf/2401.00994v1)

**Authors**: Chun Fai Chan, Daniel Wankit Yip, Aysan Esmradi

**Abstract**: The emergence of LLM (Large Language Model) integrated virtual assistants has brought about a rapid transformation in communication dynamics. During virtual assistant development, some developers prefer to leverage the system message, also known as an initial prompt or custom prompt, for preconditioning purposes. However, it is important to recognize that an excessive reliance on this functionality raises the risk of manipulation by malicious actors who can exploit it with carefully crafted prompts. Such malicious manipulation poses a significant threat, potentially compromising the accuracy and reliability of the virtual assistant's responses. Consequently, safeguarding the virtual assistants with detection and defense mechanisms becomes of paramount importance to ensure their safety and integrity. In this study, we explored three detection and defense mechanisms aimed at countering attacks that target the system message. These mechanisms include inserting a reference key, utilizing an LLM evaluator, and implementing a Self-Reminder. To showcase the efficacy of these mechanisms, they were tested against prominent attack techniques. Our findings demonstrate that the investigated mechanisms are capable of accurately identifying and counteracting the attacks. The effectiveness of these mechanisms underscores their potential in safeguarding the integrity and reliability of virtual assistants, reinforcing the importance of their implementation in real-world scenarios. By prioritizing the security of virtual assistants, organizations can maintain user trust, preserve the integrity of the application, and uphold the high standards expected in this era of transformative technologies.

摘要: LLM(Large Language Model，大型语言模型)集成虚拟助手的出现，带来了交流动力学的快速变革。在虚拟助手开发期间，一些开发人员更喜欢利用系统消息(也称为初始提示或自定义提示)进行预条件处理。但是，重要的是要认识到，过度依赖此功能会增加恶意攻击者操纵该功能的风险，恶意攻击者可以通过精心设计的提示来利用该功能。这种恶意操作构成了重大威胁，可能会损害虚拟助理响应的准确性和可靠性。因此，使用检测和防御机制来保护虚拟助理，对于确保其安全性和完整性至关重要。在本研究中，我们探索了三种检测和防御机制，旨在对抗以系统消息为目标的攻击。这些机制包括插入引用关键字、使用LLM求值器以及实现自我提醒。为了展示这些机制的有效性，他们针对突出的攻击技术进行了测试。我们的研究结果表明，所研究的机制能够准确地识别和对抗攻击。这些机制的有效性突出了它们在保障虚拟助理的完整性和可靠性方面的潜力，加强了在现实世界情景中执行这些机制的重要性。通过优先考虑虚拟助理的安全性，组织可以维护用户信任、维护应用程序的完整性，并保持在这个变革性技术时代所期望的高标准。



## **4. A Novel Evaluation Framework for Assessing Resilience Against Prompt Injection Attacks in Large Language Models**

一种新的评估大型语言模型抗即时注入攻击能力的评估框架 cs.CR

Accepted to be published in the Proceedings of The 10th IEEE CSDE  2023, the Asia-Pacific Conference on Computer Science and Data Engineering  2023

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2401.00991v1) [paper-pdf](http://arxiv.org/pdf/2401.00991v1)

**Authors**: Daniel Wankit Yip, Aysan Esmradi, Chun Fai Chan

**Abstract**: Prompt injection attacks exploit vulnerabilities in large language models (LLMs) to manipulate the model into unintended actions or generate malicious content. As LLM integrated applications gain wider adoption, they face growing susceptibility to such attacks. This study introduces a novel evaluation framework for quantifying the resilience of applications. The framework incorporates innovative techniques designed to ensure representativeness, interpretability, and robustness. To ensure the representativeness of simulated attacks on the application, a meticulous selection process was employed, resulting in 115 carefully chosen attacks based on coverage and relevance. For enhanced interpretability, a second LLM was utilized to evaluate the responses generated from these simulated attacks. Unlike conventional malicious content classifiers that provide only a confidence score, the LLM-based evaluation produces a score accompanied by an explanation, thereby enhancing interpretability. Subsequently, a resilience score is computed by assigning higher weights to attacks with greater impact, thus providing a robust measurement of the application resilience. To assess the framework's efficacy, it was applied on two LLMs, namely Llama2 and ChatGLM. Results revealed that Llama2, the newer model exhibited higher resilience compared to ChatGLM. This finding substantiates the effectiveness of the framework, aligning with the prevailing notion that newer models tend to possess greater resilience. Moreover, the framework exhibited exceptional versatility, requiring only minimal adjustments to accommodate emerging attack techniques and classifications, thereby establishing itself as an effective and practical solution. Overall, the framework offers valuable insights that empower organizations to make well-informed decisions to fortify their applications against potential threats from prompt injection.

摘要: 提示注入攻击利用大型语言模型(LLM)中的漏洞将模型操纵为意外操作或生成恶意内容。随着LLM集成应用程序得到更广泛的采用，它们面临着越来越容易受到此类攻击的风险。这项研究介绍了一种新的评估框架，用于量化应用程序的弹性。该框架结合了旨在确保代表性、可解释性和健壮性的创新技术。为了确保对应用程序的模拟攻击的代表性，采用了精心选择的过程，根据覆盖范围和相关性精心选择了115个攻击。为了增强可解释性，使用了第二个LLM来评估这些模拟攻击生成的响应。与仅提供置信度分数的传统恶意内容分类器不同，基于LLM的评估会生成伴随解释的分数，从而增强可解释性。随后，通过将更高的权重分配给影响更大的攻击来计算弹性分数，从而提供对应用程序弹性的稳健测量。为了评估该框架的有效性，将其应用于两个LLM上，即Llama2和ChatGLM。结果表明，与ChatGLM相比，较新的模型Llama2表现出更高的弹性。这一发现证实了该框架的有效性，与流行的观点一致，即较新的模型往往具有更强的弹性。此外，该框架显示出非凡的多功能性，只需进行最小的调整即可适应新出现的攻击技术和分类，从而确立其本身是一种有效和实用的解决方案。总体而言，该框架提供了有价值的见解，使组织能够做出明智的决策，以加强其应用程序免受即时注入的潜在威胁。



## **5. Opening A Pandora's Box: Things You Should Know in the Era of Custom GPTs**

打开潘多拉的盒子：定制GPT时代你应该知道的事情 cs.CR

**SubmitDate**: 2023-12-31    [abs](http://arxiv.org/abs/2401.00905v1) [paper-pdf](http://arxiv.org/pdf/2401.00905v1)

**Authors**: Guanhong Tao, Siyuan Cheng, Zhuo Zhang, Junmin Zhu, Guangyu Shen, Xiangyu Zhang

**Abstract**: The emergence of large language models (LLMs) has significantly accelerated the development of a wide range of applications across various fields. There is a growing trend in the construction of specialized platforms based on LLMs, such as the newly introduced custom GPTs by OpenAI. While custom GPTs provide various functionalities like web browsing and code execution, they also introduce significant security threats. In this paper, we conduct a comprehensive analysis of the security and privacy issues arising from the custom GPT platform. Our systematic examination categorizes potential attack scenarios into three threat models based on the role of the malicious actor, and identifies critical data exchange channels in custom GPTs. Utilizing the STRIDE threat modeling framework, we identify 26 potential attack vectors, with 19 being partially or fully validated in real-world settings. Our findings emphasize the urgent need for robust security and privacy measures in the custom GPT ecosystem, especially in light of the forthcoming launch of the official GPT store by OpenAI.

摘要: 大型语言模型(LLM)的出现极大地加速了各个领域广泛应用的开发。基于LLMS的专业平台建设有日益增长的趋势，例如OpenAI新推出的定制GPT。虽然自定义GPT提供了各种功能，如Web浏览和代码执行，但它们也带来了重大的安全威胁。在本文中，我们对定制GPT平台产生的安全和隐私问题进行了全面分析。我们的系统检查根据恶意行为者的角色将潜在的攻击场景分类为三种威胁模型，并确定了自定义GPT中的关键数据交换通道。利用STRIDE威胁建模框架，我们识别了26个潜在的攻击向量，其中19个在现实世界中得到了部分或完全的验证。我们的发现强调了定制GPT生态系统中强大的安全和隐私措施的迫切需要，特别是考虑到OpenAI即将推出官方GPT商店。



## **6. Advancing TTP Analysis: Harnessing the Power of Encoder-Only and Decoder-Only Language Models with Retrieval Augmented Generation**

高级TTP分析：利用仅编码者和仅解码者的语言模型和检索增强生成的能力 cs.CR

**SubmitDate**: 2023-12-30    [abs](http://arxiv.org/abs/2401.00280v1) [paper-pdf](http://arxiv.org/pdf/2401.00280v1)

**Authors**: Reza Fayyazi, Rozhina Taghdimi, Shanchieh Jay Yang

**Abstract**: Tactics, Techniques, and Procedures (TTPs) outline the methods attackers use to exploit vulnerabilities. The interpretation of TTPs in the MITRE ATT&CK framework can be challenging for cybersecurity practitioners due to presumed expertise, complex dependencies, and inherent ambiguity. Meanwhile, advancements with Large Language Models (LLMs) have led to recent surge in studies exploring its uses in cybersecurity operations. This leads us to question how well encoder-only (e.g., RoBERTa) and decoder-only (e.g., GPT-3.5) LLMs can comprehend and summarize TTPs to inform analysts of the intended purposes (i.e., tactics) of a cyberattack procedure. The state-of-the-art LLMs have shown to be prone to hallucination by providing inaccurate information, which is problematic in critical domains like cybersecurity. Therefore, we propose the use of Retrieval Augmented Generation (RAG) techniques to extract relevant contexts for each cyberattack procedure for decoder-only LLMs (without fine-tuning). We further contrast such approach against supervised fine-tuning (SFT) of encoder-only LLMs. Our results reveal that both the direct-use of decoder-only LLMs (i.e., its pre-trained knowledge) and the SFT of encoder-only LLMs offer inaccurate interpretation of cyberattack procedures. Significant improvements are shown when RAG is used for decoder-only LLMs, particularly when directly relevant context is found. This study further sheds insights on the limitations and capabilities of using RAG for LLMs in interpreting TTPs.

摘要: 战术、技术和过程(TTP)概述了攻击者用来利用漏洞的方法。由于假定的专业知识、复杂的依赖关系和固有的模糊性，MITRE ATT&CK框架中对TTP的解释可能会对网络安全从业者构成挑战。与此同时，大型语言模型(LLM)的进步导致了最近探索其在网络安全行动中的应用的研究激增。这导致我们质疑仅编码器(例如Roberta)和仅解码器(例如GPT-3.5)的LLM能够在多大程度上理解和总结TTP以告知分析师网络攻击过程的预期目的(即战术)。最先进的LLM通过提供不准确的信息而容易产生幻觉，这在网络安全等关键领域是有问题的。因此，我们提出使用检索增强生成(RAG)技术来提取仅针对解码器的LLMS的每个网络攻击过程的相关上下文(无需微调)。我们进一步将这种方法与仅编码器的LLM的有监督微调(SFT)进行了对比。我们的结果表明，直接使用仅解码的LLM(即其预先训练的知识)和仅编码的LLM的SFT都提供了对网络攻击过程的不准确解释。当RAG被用于仅解码器的LLM时，尤其是当找到直接相关的上下文时，显示出显著的改进。这项研究进一步揭示了使用RAG对LLMS进行TTP解释的局限性和能力。



## **7. Jatmo: Prompt Injection Defense by Task-Specific Finetuning**

Jatmo：通过特定于任务的微调实现快速注入防御 cs.CR

24 pages, 6 figures

**SubmitDate**: 2023-12-29    [abs](http://arxiv.org/abs/2312.17673v1) [paper-pdf](http://arxiv.org/pdf/2312.17673v1)

**Authors**: Julien Piet, Maha Alrashed, Chawin Sitawarin, Sizhe Chen, Zeming Wei, Elizabeth Sun, Basel Alomair, David Wagner

**Abstract**: Large Language Models (LLMs) are attracting significant research attention due to their instruction-following abilities, allowing users and developers to leverage LLMs for a variety of tasks. However, LLMs are vulnerable to prompt-injection attacks: a class of attacks that hijack the model's instruction-following abilities, changing responses to prompts to undesired, possibly malicious ones. In this work, we introduce Jatmo, a method for generating task-specific models resilient to prompt-injection attacks. Jatmo leverages the fact that LLMs can only follow instructions once they have undergone instruction tuning. It harnesses a teacher instruction-tuned model to generate a task-specific dataset, which is then used to fine-tune a base model (i.e., a non-instruction-tuned model). Jatmo only needs a task prompt and a dataset of inputs for the task: it uses the teacher model to generate outputs. For situations with no pre-existing datasets, Jatmo can use a single example, or in some cases none at all, to produce a fully synthetic dataset. Our experiments on six tasks show that Jatmo models provide the same quality of outputs on their specific task as standard LLMs, while being resilient to prompt injections. The best attacks succeeded in less than 0.5% of cases against our models, versus over 90% success rate against GPT-3.5-Turbo. We release Jatmo at https://github.com/wagner-group/prompt-injection-defense.

摘要: 大型语言模型(LLM)由于其遵循指令的能力而吸引了大量的研究关注，使用户和开发人员能够利用LLM来执行各种任务。然而，LLM很容易受到即时注入攻击：这是一类劫持模型的指令遵循能力的攻击，将对提示的响应更改为不受欢迎的、可能是恶意的提示。在这项工作中，我们介绍了Jatmo，一种生成对快速注入攻击具有弹性的特定任务模型的方法。Jatmo利用了这样一个事实，即LLM只有在经过指令调优后才能遵循指令。它利用教师指令调整的模型来生成特定于任务的数据集，然后使用该数据集来微调基本模型(即，非指令调整的模型)。Jatmo只需要任务提示符和任务输入的数据集：它使用教师模型来生成输出。对于没有预先存在的数据集的情况，Jatmo可以使用单个示例，或者在某些情况下根本不使用任何示例来生成完全合成的数据集。我们在六个任务上的实验表明，Jatmo模型在其特定任务中提供了与标准LLM相同的输出质量，同时对快速注入具有弹性。针对我们的模型，最好的攻击成功率不到0.5%，而针对GPT-3.5-Turbo的成功率超过90%。我们在https://github.com/wagner-group/prompt-injection-defense.发布了贾特莫



## **8. Identifying and Mitigating the Security Risks of Generative AI**

识别和缓解生成性人工智能的安全风险 cs.AI

**SubmitDate**: 2023-12-29    [abs](http://arxiv.org/abs/2308.14840v4) [paper-pdf](http://arxiv.org/pdf/2308.14840v4)

**Authors**: Clark Barrett, Brad Boyd, Elie Burzstein, Nicholas Carlini, Brad Chen, Jihye Choi, Amrita Roy Chowdhury, Mihai Christodorescu, Anupam Datta, Soheil Feizi, Kathleen Fisher, Tatsunori Hashimoto, Dan Hendrycks, Somesh Jha, Daniel Kang, Florian Kerschbaum, Eric Mitchell, John Mitchell, Zulfikar Ramzan, Khawaja Shams, Dawn Song, Ankur Taly, Diyi Yang

**Abstract**: Every major technical invention resurfaces the dual-use dilemma -- the new technology has the potential to be used for good as well as for harm. Generative AI (GenAI) techniques, such as large language models (LLMs) and diffusion models, have shown remarkable capabilities (e.g., in-context learning, code-completion, and text-to-image generation and editing). However, GenAI can be used just as well by attackers to generate new attacks and increase the velocity and efficacy of existing attacks.   This paper reports the findings of a workshop held at Google (co-organized by Stanford University and the University of Wisconsin-Madison) on the dual-use dilemma posed by GenAI. This paper is not meant to be comprehensive, but is rather an attempt to synthesize some of the interesting findings from the workshop. We discuss short-term and long-term goals for the community on this topic. We hope this paper provides both a launching point for a discussion on this important topic as well as interesting problems that the research community can work to address.

摘要: 每一项重大技术发明都会重新面临两难境地--新技术既有可能被用来做好事，也有可能被用来做坏事。生成性人工智能(GenAI)技术，如大型语言模型(LLMS)和扩散模型，已经显示出非凡的能力(例如，上下文学习、代码完成以及文本到图像的生成和编辑)。然而，攻击者也可以利用GenAI来生成新的攻击，并提高现有攻击的速度和效率。本文报告了在谷歌(由斯坦福大学和威斯康星大学麦迪逊分校联合举办)举行的关于GenAI造成的两用困境的研讨会的结果。这篇论文并不是要全面的，而是试图综合研讨会的一些有趣的发现。我们就这一主题讨论社区的短期和长期目标。我们希望这篇论文既为讨论这一重要主题提供了一个起点，也为研究界可以努力解决的有趣问题提供了一个起点。



## **9. Task Contamination: Language Models May Not Be Few-Shot Anymore**

任务污染：语言模型可能不再是少数 cs.CL

Accepted by AAAI 2024

**SubmitDate**: 2023-12-26    [abs](http://arxiv.org/abs/2312.16337v1) [paper-pdf](http://arxiv.org/pdf/2312.16337v1)

**Authors**: Changmao Li, Jeffrey Flanigan

**Abstract**: Large language models (LLMs) offer impressive performance in various zero-shot and few-shot tasks. However, their success in zero-shot and few-shot settings may be affected by task contamination, a potential limitation that has not been thoroughly examined. This paper investigates how zero-shot and few-shot performance of LLMs has changed chronologically over time. Utilizing GPT-3 series models and several other recent open-sourced LLMs, and controlling for dataset difficulty, we find that on datasets released before the LLM training data creation date, LLMs perform surprisingly better than on datasets released after. This strongly indicates that, for many LLMs, there exists task contamination on zero-shot and few-shot evaluation for datasets released prior to the LLMs' training data creation date. Additionally, we utilize training data inspection, task example extraction, and a membership inference attack, which reveal further evidence of task contamination. Importantly, we find that for classification tasks with no possibility of task contamination, LLMs rarely demonstrate statistically significant improvements over simple majority baselines, in both zero and few-shot settings.

摘要: 大型语言模型(LLM)在各种零射击和少射击任务中提供了令人印象深刻的性能。然而，他们在零射击和少射击设置中的成功可能会受到任务污染的影响，这是一个尚未彻底检查的潜在限制。本文研究了LLMS的零炮和少炮性能是如何随时间发生变化的。利用GPT-3系列模型和其他几个最近开源的LLM，并控制数据集的难度，我们发现，在LLM训练数据创建日期之前发布的数据集上，LLM的性能出人意料地好于之后发布的数据集。这有力地表明，对于许多LLMS来说，在LLMS的训练数据创建日期之前发布的数据集的零激发和少激发评估存在任务污染。此外，我们利用训练数据检查、任务实例提取和成员关系推理攻击，揭示了任务污染的进一步证据。重要的是，我们发现，对于没有任务污染可能性的分类任务，LLMS在零和极少的情况下，很少表现出比简单多数基线有统计上的显著改善。



## **10. Vulnerability of Machine Learning Approaches Applied in IoT-based Smart Grid: A Review**

基于物联网的智能电网中机器学习方法的脆弱性：综述 cs.CR

**SubmitDate**: 2023-12-25    [abs](http://arxiv.org/abs/2308.15736v3) [paper-pdf](http://arxiv.org/pdf/2308.15736v3)

**Authors**: Zhenyong Zhang, Mengxiang Liu, Mingyang Sun, Ruilong Deng, Peng Cheng, Dusit Niyato, Mo-Yuen Chow, Jiming Chen

**Abstract**: Machine learning (ML) sees an increasing prevalence of being used in the internet-of-things (IoT)-based smart grid. However, the trustworthiness of ML is a severe issue that must be addressed to accommodate the trend of ML-based smart grid applications (MLsgAPPs). The adversarial distortion injected into the power signal will greatly affect the system's normal control and operation. Therefore, it is imperative to conduct vulnerability assessment for MLsgAPPs applied in the context of safety-critical power systems. In this paper, we provide a comprehensive review of the recent progress in designing attack and defense methods for MLsgAPPs. Unlike the traditional survey about ML security, this is the first review work about the security of MLsgAPPs that focuses on the characteristics of power systems. We first highlight the specifics for constructing the adversarial attacks on MLsgAPPs. Then, the vulnerability of MLsgAPP is analyzed from both the aspects of the power system and ML model. Afterward, a comprehensive survey is conducted to review and compare existing studies about the adversarial attacks on MLsgAPPs in scenarios of generation, transmission, distribution, and consumption, and the countermeasures are reviewed according to the attacks that they defend against. Finally, the future research directions are discussed on the attacker's and defender's side, respectively. We also analyze the potential vulnerability of large language model-based (e.g., ChatGPT) power system applications. Overall, we encourage more researchers to contribute to investigating the adversarial issues of MLsgAPPs.

摘要: 机器学习(ML)在基于物联网(IoT)的智能电网中的应用越来越普遍。然而，ML的可信性是一个必须解决的严重问题，以适应基于ML的智能电网应用(MLsgAPP)的趋势。注入到电源信号中的对抗性失真将极大地影响系统的正常控制和运行。因此，对应用于安全关键电力系统背景下的MLsgAPP进行脆弱性评估势在必行。在本文中，我们提供了一个全面的进展，设计攻击和防御方法的MLsgAPP。与传统的ML安全研究不同，本文首次针对电力系统的特点对MLsgAPP的安全问题进行了综述。我们首先强调构造对MLsgAPP的对抗性攻击的细节。然后，从电力系统和ML模型两个方面分析了MLsgAPP的脆弱性。然后，对已有的针对MLsgAPP的生成、传输、分发、消费等场景下的对抗性攻击的研究进行了全面的回顾和比较，并根据它们所防御的攻击回顾了相应的对策。最后，分别从攻击方和防御方的角度讨论了今后的研究方向。我们还分析了基于大型语言模型(如ChatGPT)的电力系统应用的潜在脆弱性。总体而言，我们鼓励更多的研究人员为研究MLsgAPP的对抗性问题做出贡献。



## **11. From Shortcuts to Triggers: Backdoor Defense with Denoised PoE**

从快捷方式到触发器：使用去噪PoE进行后门防御 cs.CL

**SubmitDate**: 2023-12-23    [abs](http://arxiv.org/abs/2305.14910v2) [paper-pdf](http://arxiv.org/pdf/2305.14910v2)

**Authors**: Qin Liu, Fei Wang, Chaowei Xiao, Muhao Chen

**Abstract**: Language models are often at risk of diverse backdoor attacks, especially data poisoning. Thus, it is important to investigate defense solutions for addressing them. Existing backdoor defense methods mainly focus on backdoor attacks with explicit triggers, leaving a universal defense against various backdoor attacks with diverse triggers largely unexplored. In this paper, we propose an end-to-end ensemble-based backdoor defense framework, DPoE (Denoised Product-of-Experts), which is inspired by the shortcut nature of backdoor attacks, to defend various backdoor attacks. DPoE consists of two models: a shallow model that captures the backdoor shortcuts and a main model that is prevented from learning the backdoor shortcuts. To address the label flip caused by backdoor attackers, DPoE incorporates a denoising design. Experiments on SST-2 dataset show that DPoE significantly improves the defense performance against various types of backdoor triggers including word-level, sentence-level, and syntactic triggers. Furthermore, DPoE is also effective under a more challenging but practical setting that mixes multiple types of trigger.

摘要: 语言模型经常面临各种后门攻击的风险，特别是数据中毒。因此，重要的是要研究解决这些问题的防御解决方案。现有的后门防御方法主要集中在具有显式触发器的后门攻击，留下了对具有不同触发器的各种后门攻击的通用防御在很大程度上未被探索。在本文中，我们提出了一个端到端的基于集合的后门防御框架，DPoE（去噪产品的专家），这是受到后门攻击的捷径性质，以防御各种后门攻击的启发。DPoE由两种型号组成：捕获后门快捷方式的浅模型和被阻止学习后门快捷方式的主模型。为了解决后门攻击者造成的标签翻转问题，DPoE采用了去噪设计。在SST-2数据集上的实验表明，DPoE显著提高了对各种类型的后门触发器的防御性能，包括单词级，句子级和句法触发器。此外，DPoE在混合多种类型的触发器的更具挑战性但实用的设置下也是有效的。



## **12. A Mutation-Based Method for Multi-Modal Jailbreaking Attack Detection**

一种基于变异的多模式越狱攻击检测方法 cs.CR

12 pages, 8 figures

**SubmitDate**: 2023-12-23    [abs](http://arxiv.org/abs/2312.10766v2) [paper-pdf](http://arxiv.org/pdf/2312.10766v2)

**Authors**: Xiaoyu Zhang, Cen Zhang, Tianlin Li, Yihao Huang, Xiaojun Jia, Xiaofei Xie, Yang Liu, Chao Shen

**Abstract**: Large Language Models and Multi-Modal LLMs have become pervasive, and so does the importance of their security; yet, modern LLMs are known to be vulnerable to jailbreaking attacks. These attacks can allow malicious users to exploit the models, making the case for effective jailbreak detection mechanisms an essential aspect of maintaining the integrity and trustworthiness of LLM-based applications. However, existing detection works on jailbreak attacks have limitations. Existing post-query-based strategies require target domain knowledge, and pre-query-based methods mainly focus on text-level attacks and fail to meet the increasingly complex multi-modal security requirements placed upon contemporary LLMs. This gap underscores the need for a more comprehensive approach to safeguarding these influential systems.   In this work, we propose JailGuard, the first mutation-based jailbreaking detection framework which supports both image and text modalities. Our key observation is that attack queries inherently possess less robustness compared to benign queries. Specifically, to confuse the model, attack queries are usually crafted with well-designed templates or complicate perturbations, leading to a fact that a slight disturbance in input may result in a drastic change in the response. This lack of robustness can be utilized in attack detection. Based on this intuition, we designed and implemented a detection framework comprising 19 different mutators and a divergence-based detection formula. To fully understand the effectiveness of our framework, we built the first multi-modal LLM jailbreaking attack dataset, which has 304 items of data, covering ten types of known jailbreaking attacks on image and text modalities. The evaluation suggests that JailGuard achieves the best detection accuracy of 89.38%/85.42% on image and text inputs, outperforming state-of-the-art defense methods by 15.28%.

摘要: 大型语言模型和多模式LLM已经变得无处不在，它们的安全性也变得非常重要；然而，众所周知，现代LLM容易受到越狱攻击。这些攻击允许恶意用户利用这些模型，使有效的越狱检测机制成为维护基于LLM的应用程序的完整性和可信性的重要方面。然而，现有的越狱攻击检测工作存在局限性。现有的基于查询后的策略需要目标领域的知识，而基于查询前的方法主要关注文本级别的攻击，不能满足当代LLMS日益复杂的多模式安全需求。这一差距突出表明，需要采取更全面的方法来保护这些有影响力的制度。在这项工作中，我们提出了第一个基于突变的越狱检测框架JailGuard，它同时支持图像和文本两种模式。我们的主要观察结果是，与良性查询相比，攻击查询固有的健壮性较差。具体地说，为了混淆模型，攻击查询通常是使用精心设计的模板或复杂的扰动来制作的，导致输入中的轻微干扰可能会导致响应的剧烈变化。这种健壮性的缺乏可用于攻击检测。基于这一直觉，我们设计并实现了一个由19个不同的突变子和基于散度的检测公式组成的检测框架。为了充分理解我们框架的有效性，我们构建了第一个多模式LLM越狱攻击数据集，该数据集包含304项数据，涵盖了针对图像和文本模式的十种已知越狱攻击类型。评估表明，JailGuard对图像和文本输入的检测准确率最高，达到89.38%/85.42%，比最先进的防御方法高出15.28%。



## **13. A Survey on Large Language Models for Software Engineering**

面向软件工程的大型语言模型综述 cs.SE

**SubmitDate**: 2023-12-23    [abs](http://arxiv.org/abs/2312.15223v1) [paper-pdf](http://arxiv.org/pdf/2312.15223v1)

**Authors**: Quanjun Zhang, Chunrong Fang, Yang Xie, Yaxin Zhang, Yun Yang, Weisong Sun, Shengcheng Yu, Zhenyu Chen

**Abstract**: Software Engineering (SE) is the systematic design, development, and maintenance of software applications, underpinning the digital infrastructure of our modern mainworld. Very recently, the SE community has seen a rapidly increasing number of techniques employing Large Language Models (LLMs) to automate a broad range of SE tasks. Nevertheless, existing information of the applications, effects, and possible limitations of LLMs within SE is still not well-studied.   In this paper, we provide a systematic survey to summarize the current state-of-the-art research in the LLM-based SE community. We summarize 30 representative LLMs of Source Code across three model architectures, 15 pre-training objectives across four categories, and 16 downstream tasks across five categories. We then present a detailed summarization of the recent SE studies for which LLMs are commonly utilized, including 155 studies for 43 specific code-related tasks across four crucial phases within the SE workflow. Besides, we summarize existing attempts to empirically evaluate LLMs in SE, such as benchmarks, empirical studies, and exploration of SE education. We also discuss several critical aspects of optimization and applications of LLMs in SE, such as security attacks, model tuning, and model compression. Finally, we highlight several challenges and potential opportunities on applying LLMs for future SE studies, such as exploring domain LLMs and constructing clean evaluation datasets. Overall, our work can help researchers gain a comprehensive understanding about the achievements of the existing LLM-based SE studies and promote the practical application of these techniques. Our artifacts are publicly available and will continuously updated at the living repository: \url{https://github.com/iSEngLab/AwesomeLLM4SE}.

摘要: 软件工程(SE)是软件应用程序的系统设计、开发和维护，是现代主流世界的数字基础设施的基础。最近，SE社区看到越来越多的技术使用大型语言模型(LLM)来自动化广泛的SE任务。然而，现有的关于低密度脂蛋白在SE中的应用、效果和可能的局限性的信息仍然没有得到很好的研究。在本文中，我们提供了一个系统的综述，以总结当前在基于LLM的SE社区的最新研究。我们总结了三个模型体系结构中具有代表性的30个源代码LLM，四个类别中的15个预培训目标，以及五个类别中的16个下游任务。然后，我们详细总结了最近经常使用LLM的SE研究，包括针对SE工作流程中四个关键阶段的43个特定代码相关任务的155个研究。此外，我们还总结了已有的对SE中的LLMS进行经验性评估的尝试，如基准、实证研究和SE教育探索。我们还讨论了LLMS在SE中优化和应用的几个关键方面，如安全攻击、模型调整和模型压缩。最后，我们强调了在未来的SE研究中应用LLMS的几个挑战和潜在的机会，例如探索领域LLMS和构建干净的评估数据集。总体而言，我们的工作可以帮助研究人员全面了解现有基于LLM的SE研究的成果，并促进这些技术的实际应用。我们的手工艺品是公开可用的，并将在实时存储库中不断更新：\url{https://github.com/iSEngLab/AwesomeLLM4SE}.



## **14. Spear Phishing With Large Language Models**

使用大型语言模型的鱼叉式网络钓鱼 cs.CY

16 pages, 10 figures

**SubmitDate**: 2023-12-22    [abs](http://arxiv.org/abs/2305.06972v3) [paper-pdf](http://arxiv.org/pdf/2305.06972v3)

**Authors**: Julian Hazell

**Abstract**: Recent progress in artificial intelligence (AI), particularly in the domain of large language models (LLMs), has resulted in powerful and versatile dual-use systems. This intelligence can be put towards a wide variety of beneficial tasks, yet it can also be used to cause harm. This study explores one such harm by examining how LLMs can be used for spear phishing, a form of cybercrime that involves manipulating targets into divulging sensitive information. I first explore LLMs' ability to assist with the reconnaissance and message generation stages of a spear phishing attack, where I find that LLMs are capable of assisting with the email generation phase of a spear phishing attack. To explore how LLMs could potentially be harnessed to scale spear phishing campaigns, I then create unique spear phishing messages for over 600 British Members of Parliament using OpenAI's GPT-3.5 and GPT-4 models. My findings provide some evidence that these messages are not only realistic but also cost-effective, with each email costing only a fraction of a cent to generate. Next, I demonstrate how basic prompt engineering can circumvent safeguards installed in LLMs, highlighting the need for further research into robust interventions that can help prevent models from being misused. To further address these evolving risks, I explore two potential solutions: structured access schemes, such as application programming interfaces, and LLM-based defensive systems.

摘要: 人工智能(AI)领域的最新进展，特别是在大型语言模型(LLMS)领域的进展，导致了强大而通用的两用系统。这种智慧可以用于各种各样有益的任务，但也可以用来造成伤害。这项研究通过研究LLMS如何被用于鱼叉式网络钓鱼来探索这样的危害，鱼叉式网络钓鱼是一种网络犯罪形式，涉及操纵目标泄露敏感信息。我首先探讨LLMS协助鱼叉式网络钓鱼攻击的侦察和消息生成阶段的能力，我发现LLMS能够协助鱼叉式网络钓鱼攻击的电子邮件生成阶段。为了探索如何潜在地利用LLMS来扩大鱼叉式网络钓鱼活动，我随后使用OpenAI的GPT-3.5和GPT-4模型为600多名英国国会议员创建了独特的鱼叉式网络钓鱼消息。我的发现提供了一些证据，证明这些信息不仅现实，而且具有成本效益，每封电子邮件的生成成本只有一分钱的零头。接下来，我将演示基本的即时工程如何绕过安装在低成本管理系统中的保障措施，强调有必要对能够帮助防止模型被滥用的强大干预措施进行进一步研究。为了进一步应对这些不断变化的风险，我探索了两个潜在的解决方案：结构化访问方案，如应用程序编程接口，以及基于LLM的防御系统。



## **15. MetaAID 2.5: A Secure Framework for Developing Metaverse Applications via Large Language Models**

MetaAID 2.5：通过大型语言模型开发Metverse应用程序的安全框架 cs.CR

**SubmitDate**: 2023-12-22    [abs](http://arxiv.org/abs/2312.14480v1) [paper-pdf](http://arxiv.org/pdf/2312.14480v1)

**Authors**: Hongyin Zhu

**Abstract**: Large language models (LLMs) are increasingly being used in Metaverse environments to generate dynamic and realistic content and to control the behavior of non-player characters (NPCs). However, the cybersecurity concerns associated with LLMs have become increasingly prominent. Previous research has primarily focused on patching system vulnerabilities to enhance cybersecurity, but these approaches are not well-suited to the Metaverse, where the virtual space is more complex, LLMs are vulnerable, and ethical user interaction is critical. Moreover, the scope of cybersecurity in the Metaverse is expected to expand significantly. This paper proposes a method for enhancing cybersecurity through the simulation of user interaction with LLMs. Our goal is to educate users and strengthen their defense capabilities through exposure to a comprehensive simulation system. This system includes extensive Metaverse cybersecurity Q&A and attack simulation scenarios. By engaging with these, users will improve their ability to recognize and withstand risks. Additionally, to address the ethical implications of user input, we propose using LLMs as evaluators to assess user content across five dimensions. We further adapt the models through vocabulary expansion training to better understand personalized inputs and emoticons. We conduct experiments on multiple LLMs and find that our approach is effective.

摘要: 大型语言模型(LLM)越来越多地用于Metverse环境中，以生成动态和逼真的内容，并控制非玩家角色(NPC)的行为。然而，与低成本管理相关的网络安全担忧已变得日益突出。以前的研究主要集中在修补系统漏洞以增强网络安全，但这些方法不太适合Metverse，因为虚拟空间更复杂，LLM容易受到攻击，道德用户交互至关重要。此外，Metverse的网络安全范围预计将大幅扩大。提出了一种通过模拟用户与LLMS的交互来增强网络安全的方法。我们的目标是通过接触一个全面的模拟系统来教育用户并增强他们的防御能力。该系统包括广泛的Metverse网络安全问答和攻击模拟场景。通过参与这些活动，用户将提高识别和抵御风险的能力。此外，为了解决用户输入的伦理影响，我们建议使用LLMS作为评估者，从五个维度评估用户内容。我们通过词汇扩展训练进一步调整模型，以更好地理解个性化输入和表情符号。我们在多个LLM上进行了实验，发现我们的方法是有效的。



## **16. HW-V2W-Map: Hardware Vulnerability to Weakness Mapping Framework for Root Cause Analysis with GPT-assisted Mitigation Suggestion**

HW-V2W-Map：GPT辅助缓解建议的根本原因分析的硬件弱点映射框架 cs.CR

22 pages, 10 pages appendix, 10 figures, Submitted to ACM TODAES

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2312.13530v1) [paper-pdf](http://arxiv.org/pdf/2312.13530v1)

**Authors**: Yu-Zheng Lin, Muntasir Mamun, Muhtasim Alam Chowdhury, Shuyu Cai, Mingyu Zhu, Banafsheh Saber Latibari, Kevin Immanuel Gubbi, Najmeh Nazari Bavarsad, Arjun Caputo, Avesta Sasan, Houman Homayoun, Setareh Rafatirad, Pratik Satam, Soheil Salehi

**Abstract**: The escalating complexity of modern computing frameworks has resulted in a surge in the cybersecurity vulnerabilities reported to the National Vulnerability Database (NVD) by practitioners. Despite the fact that the stature of NVD is one of the most significant databases for the latest insights into vulnerabilities, extracting meaningful trends from such a large amount of unstructured data is still challenging without the application of suitable technological methodologies. Previous efforts have mostly concentrated on software vulnerabilities; however, a holistic strategy incorporates approaches for mitigating vulnerabilities, score prediction, and a knowledge-generating system that may extract relevant insights from the Common Weakness Enumeration (CWE) and Common Vulnerability Exchange (CVE) databases is notably absent. As the number of hardware attacks on Internet of Things (IoT) devices continues to rapidly increase, we present the Hardware Vulnerability to Weakness Mapping (HW-V2W-Map) Framework, which is a Machine Learning (ML) framework focusing on hardware vulnerabilities and IoT security. The architecture that we have proposed incorporates an Ontology-driven Storytelling framework, which automates the process of updating the ontology in order to recognize patterns and evolution of vulnerabilities over time and provides approaches for mitigating the vulnerabilities. The repercussions of vulnerabilities can be mitigated as a result of this, and conversely, future exposures can be predicted and prevented. Furthermore, our proposed framework utilized Generative Pre-trained Transformer (GPT) Large Language Models (LLMs) to provide mitigation suggestions.

摘要: 现代计算框架的日益复杂导致从业者向国家漏洞数据库(NVD)报告的网络安全漏洞激增。尽管NVD的地位是最新洞察漏洞的最重要的数据库之一，但如果没有适当的技术方法的应用，从如此大量的非结构化数据中提取有意义的趋势仍然是具有挑战性的。以前的工作主要集中在软件漏洞上；然而，整体战略结合了缓解漏洞的方法、分数预测，并且明显缺乏可以从共同弱点枚举(CWE)和共同漏洞交换(CVE)数据库中提取相关见解的知识生成系统。针对物联网(IoT)设备遭受硬件攻击的情况，提出了硬件弱点映射(HW-V2W-Map)框架，这是一个关注硬件脆弱性和物联网安全的机器学习(ML)框架。我们建议的体系结构结合了本体驱动的故事讲述框架，该框架自动更新本体的过程，以便识别漏洞的模式和随时间的演变，并提供缓解漏洞的方法。因此，可以减轻漏洞的影响，反过来，可以预测和预防未来的风险暴露。此外，我们提出的框架利用产生式预训练转换器(GPT)大型语言模型(LLMS)来提供缓解建议。



## **17. Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models**

针对大型语言模型的间接提示注入攻击的基准测试和防御 cs.CL

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2312.14197v1) [paper-pdf](http://arxiv.org/pdf/2312.14197v1)

**Authors**: Jingwei Yi, Yueqi Xie, Bin Zhu, Keegan Hines, Emre Kiciman, Guangzhong Sun, Xing Xie, Fangzhao Wu

**Abstract**: Recent remarkable advancements in large language models (LLMs) have led to their widespread adoption in various applications. A key feature of these applications is the combination of LLMs with external content, where user instructions and third-party content are combined to create prompts for LLM processing. These applications, however, are vulnerable to indirect prompt injection attacks, where malicious instructions embedded within external content compromise LLM's output, causing their responses to deviate from user expectations. Despite the discovery of this security issue, no comprehensive analysis of indirect prompt injection attacks on different LLMs is available due to the lack of a benchmark. Furthermore, no effective defense has been proposed.   In this work, we introduce the first benchmark, BIPIA, to measure the robustness of various LLMs and defenses against indirect prompt injection attacks. Our experiments reveal that LLMs with greater capabilities exhibit more vulnerable to indirect prompt injection attacks for text tasks, resulting in a higher ASR. We hypothesize that indirect prompt injection attacks are mainly due to the LLMs' inability to distinguish between instructions and external content. Based on this conjecture, we propose four black-box methods based on prompt learning and a white-box defense methods based on fine-tuning with adversarial training to enable LLMs to distinguish between instructions and external content and ignore instructions in the external content. Our experimental results show that our black-box defense methods can effectively reduce ASR but cannot completely thwart indirect prompt injection attacks, while our white-box defense method can reduce ASR to nearly zero with little adverse impact on the LLM's performance on general tasks. We hope that our benchmark and defenses can inspire future work in this important area.

摘要: 最近，大型语言模型(LLM)的显著进步导致了它们在各种应用中的广泛采用。这些应用程序的一个关键功能是将LLM与外部内容相结合，其中结合了用户指令和第三方内容，以创建LLM处理的提示。然而，这些应用程序容易受到间接提示注入攻击，在这种攻击中，嵌入在外部内容中的恶意指令会危及LLM的输出，导致它们的响应偏离用户预期。尽管发现了这个安全问题，但由于缺乏基准，对不同LLM的间接即时注入攻击没有全面的分析。此外，还没有提出有效的防御措施。在这项工作中，我们引入了第一个基准，BIPIA，来衡量各种LLM的健壮性和对间接即时注入攻击的防御。我们的实验表明，具有更大能力的LLM更容易受到文本任务的间接提示注入攻击，从而导致更高的ASR。我们假设间接提示注入攻击主要是由于LLMS无法区分指令和外部内容。基于这一猜想，我们提出了四种基于快速学习的黑盒方法和一种基于微调和对抗性训练的白盒防御方法，使LLMS能够区分指令和外部内容，并忽略外部内容中的指令。我们的实验结果表明，我们的黑盒防御方法可以有效地降低ASR，但不能完全阻止间接提示注入攻击，而我们的白盒防御方法可以将ASR降低到几乎为零，并且对LLM在一般任务上的性能影响很小。我们希望我们的基准和辩护能够激励这一重要领域的未来工作。



## **18. Universal and Transferable Adversarial Attacks on Aligned Language Models**

对对齐语言模型的通用和可转移的对抗性攻击 cs.CL

Website: http://llm-attacks.org/

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2307.15043v2) [paper-pdf](http://arxiv.org/pdf/2307.15043v2)

**Authors**: Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, Matt Fredrikson

**Abstract**: Because "out-of-the-box" large language models are capable of generating a great deal of objectionable content, recent work has focused on aligning these models in an attempt to prevent undesirable generation. While there has been some success at circumventing these measures -- so-called "jailbreaks" against LLMs -- these attacks have required significant human ingenuity and are brittle in practice. In this paper, we propose a simple and effective attack method that causes aligned language models to generate objectionable behaviors. Specifically, our approach finds a suffix that, when attached to a wide range of queries for an LLM to produce objectionable content, aims to maximize the probability that the model produces an affirmative response (rather than refusing to answer). However, instead of relying on manual engineering, our approach automatically produces these adversarial suffixes by a combination of greedy and gradient-based search techniques, and also improves over past automatic prompt generation methods.   Surprisingly, we find that the adversarial prompts generated by our approach are quite transferable, including to black-box, publicly released LLMs. Specifically, we train an adversarial attack suffix on multiple prompts (i.e., queries asking for many different types of objectionable content), as well as multiple models (in our case, Vicuna-7B and 13B). When doing so, the resulting attack suffix is able to induce objectionable content in the public interfaces to ChatGPT, Bard, and Claude, as well as open source LLMs such as LLaMA-2-Chat, Pythia, Falcon, and others. In total, this work significantly advances the state-of-the-art in adversarial attacks against aligned language models, raising important questions about how such systems can be prevented from producing objectionable information. Code is available at github.com/llm-attacks/llm-attacks.

摘要: 由于“开箱即用”的大型语言模型能够生成大量令人反感的内容，最近的工作重点是调整这些模型，以试图防止不必要的生成。虽然在规避这些措施方面取得了一些成功--即所谓的针对LLMS的“越狱”--但这些攻击需要大量的人类智慧，而且在实践中是脆弱的。在本文中，我们提出了一种简单有效的攻击方法，使对齐的语言模型产生令人反感的行为。具体地说，我们的方法找到了一个后缀，当附加到LLM的广泛查询中以产生令人反感的内容时，旨在最大化该模型产生肯定响应(而不是拒绝回答)的概率。然而，我们的方法不依赖于人工设计，而是通过贪婪和基于梯度的搜索技术相结合来自动生成这些对抗性后缀，并且改进了过去的自动提示生成方法。令人惊讶的是，我们发现我们的方法生成的对抗性提示是相当可转移的，包括到黑盒，公开发布的LLM。具体地说，我们对多个提示(即，要求许多不同类型的不良内容的查询)以及多个模型(在我们的案例中，Vicuna-7B和13B)训练对抗性攻击后缀。这样做时，生成的攻击后缀能够在ChatGPT、Bard和Claude的公共接口以及开源LLM(如llama-2-chat、Pythia、Falcon和其他)中诱导令人反感的内容。总而言之，这项工作极大地推进了针对对齐语言模型的对抗性攻击的最新水平，提出了如何防止此类系统产生令人反感的信息的重要问题。代码可在githorb.com/llm-Attages/llm-Attack上找到。



## **19. Robust Contrastive Language-Image Pre-training against Data Poisoning and Backdoor Attacks**

健壮的对比语言--针对数据中毒和后门攻击的图像预训练 cs.CV

**SubmitDate**: 2023-12-19    [abs](http://arxiv.org/abs/2303.06854v2) [paper-pdf](http://arxiv.org/pdf/2303.06854v2)

**Authors**: Wenhan Yang, Jingdong Gao, Baharan Mirzasoleiman

**Abstract**: Contrastive vision-language representation learning has achieved state-of-the-art performance for zero-shot classification, by learning from millions of image-caption pairs crawled from the internet. However, the massive data that powers large multimodal models such as CLIP, makes them extremely vulnerable to various types of targeted data poisoning and backdoor attacks. Despite this vulnerability, robust contrastive vision-language pre-training against such attacks has remained unaddressed. In this work, we propose ROCLIP, the first effective method for robust pre-training multimodal vision-language models against targeted data poisoning and backdoor attacks. ROCLIP effectively breaks the association between poisoned image-caption pairs by considering a relatively large and varying pool of random captions, and matching every image with the text that is most similar to it in the pool instead of its own caption, every few epochs.It also leverages image and text augmentations to further strengthen the defense and improve the performance of the model. Our extensive experiments show that ROCLIP renders state-of-the-art targeted data poisoning and backdoor attacks ineffective during pre-training CLIP models. In particular, ROCLIP decreases the success rate for targeted data poisoning attacks from 93.75% to 12.5% and that of backdoor attacks down to 0%, while improving the model's linear probe performance by 10% and maintains a similar zero shot performance compared to CLIP. By increasing the frequency of matching, ROCLIP is able to defend strong attacks, which add up to 1% poisoned examples to the data, and successfully maintain a low attack success rate of 12.5%, while trading off the performance on some tasks.

摘要: 对比视觉语言表征学习通过从互联网上抓取的数百万个图像-标题对中学习，实现了最先进的零镜头分类性能。然而，为CLIP等大型多模态模型提供动力的大量数据使它们极易受到各种类型的有针对性的数据中毒和后门攻击。尽管存在这种脆弱性，但针对此类攻击的强大对比视觉语言预训练仍然没有得到解决。在这项工作中，我们提出了ROCLIP，这是第一个有效的方法，用于鲁棒的预训练多模态视觉语言模型，以对抗有针对性的数据中毒和后门攻击。ROCLIP通过考虑一个相对较大且变化的随机字幕池，并每隔几个epoch将每个图像与池中最相似的文本（而不是其自身的字幕）进行匹配，有效地打破了中毒图像-字幕对之间的关联。它还利用图像和文本增强来进一步加强防御并提高模型的性能。我们广泛的实验表明，ROCLIP在预训练CLIP模型期间使最先进的有针对性的数据中毒和后门攻击无效。特别是，ROCLIP将目标数据中毒攻击的成功率从93.75%降低到12.5%，后门攻击的成功率降低到0%，同时将模型的线性探测性能提高了10%，并保持了与CLIP相似的零射击性能。通过增加匹配频率，ROCLIP能够防御强攻击，这些攻击将向数据添加高达1%的中毒示例，并成功保持12.5%的低攻击成功率，同时在某些任务上牺牲性能。



## **20. Traces of Memorisation in Large Language Models for Code**

代码的大型语言模型中的记忆痕迹 cs.CR

ICSE 2024 Research Track

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2312.11658v1) [paper-pdf](http://arxiv.org/pdf/2312.11658v1)

**Authors**: Ali Al-Kaswan, Maliheh Izadi, Arie van Deursen

**Abstract**: Large language models have gained significant popularity because of their ability to generate human-like text and potential applications in various fields, such as Software Engineering. Large language models for code are commonly trained on large unsanitised corpora of source code scraped from the internet. The content of these datasets is memorised and can be extracted by attackers with data extraction attacks. In this work, we explore memorisation in large language models for code and compare the rate of memorisation with large language models trained on natural language. We adopt an existing benchmark for natural language and construct a benchmark for code by identifying samples that are vulnerable to attack. We run both benchmarks against a variety of models, and perform a data extraction attack. We find that large language models for code are vulnerable to data extraction attacks, like their natural language counterparts. From the training data that was identified to be potentially extractable we were able to extract 47% from a CodeGen-Mono-16B code completion model. We also observe that models memorise more, as their parameter count grows, and that their pre-training data are also vulnerable to attack. We also find that data carriers are memorised at a higher rate than regular code or documentation and that different model architectures memorise different samples. Data leakage has severe outcomes, so we urge the research community to further investigate the extent of this phenomenon using a wider range of models and extraction techniques in order to build safeguards to mitigate this issue.

摘要: 大型语言模型因其生成类似人类的文本的能力以及在软件工程等各个领域的潜在应用而广受欢迎。代码的大型语言模型通常是在从互联网上刮来的大量未经清理的源代码语料库上进行训练的。这些数据集的内容是被记忆的，并且可以被具有数据提取攻击的攻击者提取。在这项工作中，我们探索了代码在大型语言模型中的记忆，并将记忆速度与自然语言训练的大型语言模型进行了比较。我们采用现有的自然语言基准，并通过识别易受攻击的样本来构建代码基准。我们针对各种模型运行这两个基准测试，并执行数据提取攻击。我们发现代码的大型语言模型很容易受到数据提取攻击，就像它们的自然语言模型一样。从被确定为潜在可提取的训练数据中，我们能够从CodeGen-Mono-16B代码完成模型中提取47%。我们还观察到，随着参数数量的增加，模型记住的更多，而且它们的预训练数据也很容易受到攻击。我们还发现，数据载体的记忆速度高于常规代码或文档，并且不同的模型体系结构记忆的样本不同。数据泄露具有严重的后果，因此我们敦促研究界使用更广泛的模型和提取技术进一步调查这一现象的程度，以便建立保障措施来缓解这一问题。



## **21. PoisonPrompt: Backdoor Attack on Prompt-based Large Language Models**

PoisonPrompt：对基于提示的大型语言模型的后门攻击 cs.CL

To Appear in IEEE ICASSP 2024, code is available at:  https://github.com/grasses/PoisonPrompt

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2310.12439v2) [paper-pdf](http://arxiv.org/pdf/2310.12439v2)

**Authors**: Hongwei Yao, Jian Lou, Zhan Qin

**Abstract**: Prompts have significantly improved the performance of pretrained Large Language Models (LLMs) on various downstream tasks recently, making them increasingly indispensable for a diverse range of LLM application scenarios. However, the backdoor vulnerability, a serious security threat that can maliciously alter the victim model's normal predictions, has not been sufficiently explored for prompt-based LLMs. In this paper, we present POISONPROMPT, a novel backdoor attack capable of successfully compromising both hard and soft prompt-based LLMs. We evaluate the effectiveness, fidelity, and robustness of POISONPROMPT through extensive experiments on three popular prompt methods, using six datasets and three widely used LLMs. Our findings highlight the potential security threats posed by backdoor attacks on prompt-based LLMs and emphasize the need for further research in this area.

摘要: 最近，提示显著提高了预先训练的大型语言模型(LLM)在各种下游任务上的性能，使它们在各种LLM应用场景中越来越不可或缺。然而，后门漏洞是一个严重的安全威胁，可能会恶意改变受害者模型的正常预测，但对于基于提示的LLM来说，这种漏洞还没有得到充分的研究。在本文中，我们提出了一种新的后门攻击POISONPROMPT，它能够成功地攻破基于硬提示和软提示的LLMS。我们使用6个数据集和3个广泛使用的LLMS对POISONPROMPT的有效性、保真度和稳健性进行了评估。我们的发现突出了后门攻击对基于提示的LLM的潜在安全威胁，并强调了在这一领域进行进一步研究的必要性。



## **22. A Comprehensive Survey of Attack Techniques, Implementation, and Mitigation Strategies in Large Language Models**

大型语言模型中的攻击技术、实现和防御策略综述 cs.CR

Accepted to be published in the Proceedings of the 3rd International  Conference on Ubiquitous Security 2023 (UbiSec-2023)

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2312.10982v1) [paper-pdf](http://arxiv.org/pdf/2312.10982v1)

**Authors**: Aysan Esmradi, Daniel Wankit Yip, Chun Fai Chan

**Abstract**: Ensuring the security of large language models (LLMs) is an ongoing challenge despite their widespread popularity. Developers work to enhance LLMs security, but vulnerabilities persist, even in advanced versions like GPT-4. Attackers exploit these weaknesses, highlighting the need for proactive cybersecurity measures in AI model development. This article explores two attack categories: attacks on models themselves and attacks on model applications. The former requires expertise, access to model data, and significant implementation time, while the latter is more accessible to attackers and has seen increased attention. Our study reviews over 100 recent research works, providing an in-depth analysis of each attack type. We identify the latest attack methods and explore various approaches to carry them out. We thoroughly investigate mitigation techniques, assessing their effectiveness and limitations. Furthermore, we summarize future defenses against these attacks. We also examine real-world techniques, including reported and our implemented attacks on LLMs, to consolidate our findings. Our research highlights the urgency of addressing security concerns and aims to enhance the understanding of LLM attacks, contributing to robust defense development in this evolving domain.

摘要: 尽管大型语言模型(LLM)广受欢迎，但确保其安全性仍是一个持续的挑战。开发人员努力增强LLMS的安全性，但漏洞仍然存在，即使是在GPT-4这样的高级版本中也是如此。攻击者利用这些弱点，突显了在人工智能模型开发中采取主动网络安全措施的必要性。本文探讨了两种攻击类别：对模型本身的攻击和对模型应用程序的攻击。前者需要专业知识、对模型数据的访问和大量的实施时间，而后者更容易被攻击者访问，并受到越来越多的关注。我们的研究回顾了100多项最新的研究工作，对每种攻击类型进行了深入的分析。我们确定了最新的攻击方法，并探索了执行它们的各种方法。我们深入研究缓解技术，评估其有效性和局限性。此外，我们还总结了未来针对这些攻击的防御措施。我们还检查了现实世界的技术，包括报告的和我们实施的对LLM的攻击，以巩固我们的发现。我们的研究强调了解决安全问题的紧迫性，并旨在加强对LLM攻击的理解，为这个不断发展的领域中强有力的防御发展做出贡献。



## **23. No-Skim: Towards Efficiency Robustness Evaluation on Skimming-based Language Models**

No-Skim：基于Skimming语言模型的有效鲁棒性评估 cs.CR

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2312.09494v2) [paper-pdf](http://arxiv.org/pdf/2312.09494v2)

**Authors**: Shengyao Zhang, Mi Zhang, Xudong Pan, Min Yang

**Abstract**: To reduce the computation cost and the energy consumption in large language models (LLM), skimming-based acceleration dynamically drops unimportant tokens of the input sequence progressively along layers of the LLM while preserving the tokens of semantic importance. However, our work for the first time reveals the acceleration may be vulnerable to Denial-of-Service (DoS) attacks. In this paper, we propose No-Skim, a general framework to help the owners of skimming-based LLM to understand and measure the robustness of their acceleration scheme. Specifically, our framework searches minimal and unnoticeable perturbations at character-level and token-level to generate adversarial inputs that sufficiently increase the remaining token ratio, thus increasing the computation cost and energy consumption. We systematically evaluate the vulnerability of the skimming acceleration in various LLM architectures including BERT and RoBERTa on the GLUE benchmark. In the worst case, the perturbation found by No-Skim substantially increases the running cost of LLM by over 145% on average. Moreover, No-Skim extends the evaluation framework to various scenarios, making the evaluation conductible with different level of knowledge.

摘要: 为了降低大型语言模型(LLM)的计算代价和能量消耗，基于略读的加速算法在保留语义重要的标记的同时，动态地沿着LLM的层次逐步丢弃输入序列中不重要的标记。然而，我们的工作首次揭示了加速可能容易受到拒绝服务(DoS)攻击。在本文中，我们提出了一个通用的框架No-Skim，以帮助基于略读的LLM的所有者理解和度量其加速方案的健壮性。具体地说，我们的框架在字符级和令牌级搜索最小和不可察觉的扰动，以生成足以增加剩余令牌率的对抗性输入，从而增加计算成本和能量消耗。我们在GLUE基准上系统地评估了包括Bert和Roberta在内的各种LLM架构中掠读加速的脆弱性。在最坏的情况下，No-Skim发现的扰动大大增加了LLM的运行成本，平均超过145%。此外，No-Skim将评估框架扩展到各种场景，使评估可以在不同的知识水平下进行。



## **24. Privacy-Aware Document Visual Question Answering**

隐私感知文档可视问题分类 cs.CV

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2312.10108v1) [paper-pdf](http://arxiv.org/pdf/2312.10108v1)

**Authors**: Rubèn Tito, Khanh Nguyen, Marlon Tobaben, Raouf Kerkouche, Mohamed Ali Souibgui, Kangsoo Jung, Lei Kang, Ernest Valveny, Antti Honkela, Mario Fritz, Dimosthenis Karatzas

**Abstract**: Document Visual Question Answering (DocVQA) is a fast growing branch of document understanding. Despite the fact that documents contain sensitive or copyrighted information, none of the current DocVQA methods offers strong privacy guarantees.   In this work, we explore privacy in the domain of DocVQA for the first time. We highlight privacy issues in state of the art multi-modal LLM models used for DocVQA, and explore possible solutions.   Specifically, we focus on the invoice processing use case as a realistic, widely used scenario for document understanding, and propose a large scale DocVQA dataset comprising invoice documents and associated questions and answers. We employ a federated learning scheme, that reflects the real-life distribution of documents in different businesses, and we explore the use case where the ID of the invoice issuer is the sensitive information to be protected.   We demonstrate that non-private models tend to memorise, behaviour that can lead to exposing private information. We then evaluate baseline training schemes employing federated learning and differential privacy in this multi-modal scenario, where the sensitive information might be exposed through any of the two input modalities: vision (document image) or language (OCR tokens).   Finally, we design an attack exploiting the memorisation effect of the model, and demonstrate its effectiveness in probing different DocVQA models.

摘要: 文档视觉问答(DocVQA)是文档理解领域中发展迅速的一个分支。尽管文档包含敏感或受版权保护的信息，但当前的DocVQA方法都不能提供强有力的隐私保障。在这项工作中，我们首次探索了DocVQA领域的隐私。我们重点介绍了用于DocVQA的最先进的多模式LLM模型中的隐私问题，并探索了可能的解决方案。具体地说，我们将发票处理用例作为一个现实的、广泛使用的文档理解场景来关注，并提出了一个大规模的DocVQA数据集，包括发票文档和相关的问答。我们采用了联合学习方案，反映了文档在不同业务中的真实分布，并探索了发票开具人的ID是要保护的敏感信息的用例。我们证明，非私人模式往往会记忆，这是可能导致私人信息泄露的行为。然后，我们评估了在这种多模式场景中使用联合学习和差异隐私的基线训练方案，其中敏感信息可能通过两种输入通道中的任何一种暴露：视觉(文档图像)或语言(OCR令牌)。最后，我们设计了一个利用该模型的记忆效应的攻击，并在探测不同的DocVQA模型时展示了它的有效性。



## **25. AutoDAN: Interpretable Gradient-Based Adversarial Attacks on Large Language Models**

AutoDAN：对大型语言模型的可解释的基于一致性的对抗性攻击 cs.CR

Version 2 updates: Added comparison of three more evaluation methods  and their reliability check using human labeling. Added results for  jailbreaking Llama2 (individual behavior) and included complexity and  hyperparameter analysis. Revised objectives for prompt leaking. Other minor  changes made

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2310.15140v2) [paper-pdf](http://arxiv.org/pdf/2310.15140v2)

**Authors**: Sicheng Zhu, Ruiyi Zhang, Bang An, Gang Wu, Joe Barrow, Zichao Wang, Furong Huang, Ani Nenkova, Tong Sun

**Abstract**: Safety alignment of Large Language Models (LLMs) can be compromised with manual jailbreak attacks and (automatic) adversarial attacks. Recent studies suggest that defending against these attacks is possible: adversarial attacks generate unlimited but unreadable gibberish prompts, detectable by perplexity-based filters; manual jailbreak attacks craft readable prompts, but their limited number due to the necessity of human creativity allows for easy blocking. In this paper, we show that these solutions may be too optimistic. We introduce AutoDAN, an interpretable, gradient-based adversarial attack that merges the strengths of both attack types. Guided by the dual goals of jailbreak and readability, AutoDAN optimizes and generates tokens one by one from left to right, resulting in readable prompts that bypass perplexity filters while maintaining high attack success rates. Notably, these prompts, generated from scratch using gradients, are interpretable and diverse, with emerging strategies commonly seen in manual jailbreak attacks. They also generalize to unforeseen harmful behaviors and transfer to black-box LLMs better than their unreadable counterparts when using limited training data or a single proxy model. Furthermore, we show the versatility of AutoDAN by automatically leaking system prompts using a customized objective. Our work offers a new way to red-team LLMs and understand jailbreak mechanisms via interpretability.

摘要: 大型语言模型(LLM)的安全对齐可能会受到手动越狱攻击和(自动)对抗性攻击的影响。最近的研究表明，防御这些攻击是可能的：对抗性攻击生成无限但不可读的胡言乱语提示，可通过基于困惑的过滤器检测；手动越狱攻击创建可读的提示，但由于人类创造力的必要性，其数量有限，允许轻松阻止。在本文中，我们证明了这些解决方案可能过于乐观。我们介绍了AutoDAN，一种可解释的、基于梯度的对抗性攻击，它融合了这两种攻击类型的优点。在越狱和可读性双重目标的指导下，AutoDAN从左到右一个接一个地优化和生成令牌，产生可读的提示，绕过困惑过滤器，同时保持高攻击成功率。值得注意的是，这些使用渐变从零开始生成的提示是可解释的和多样化的，新出现的策略通常出现在手动越狱攻击中。当使用有限的训练数据或单一代理模型时，它们还概括到不可预见的有害行为，并比不可读的同行更好地转移到黑盒LLM。此外，我们通过使用定制目标自动泄漏系统提示来展示AutoDAN的多功能性。我们的工作为红色团队LLM提供了一种新的方法，并通过可解释性来理解越狱机制。



## **26. FigStep: Jailbreaking Large Vision-language Models via Typographic Visual Prompts**

图：通过排版视觉提示越狱的大型视觉语言模型 cs.CR

Technical Report

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2311.05608v2) [paper-pdf](http://arxiv.org/pdf/2311.05608v2)

**Authors**: Yichen Gong, Delong Ran, Jinyuan Liu, Conglei Wang, Tianshuo Cong, Anyu Wang, Sisi Duan, Xiaoyun Wang

**Abstract**: Ensuring the safety of artificial intelligence-generated content (AIGC) is a longstanding topic in the artificial intelligence (AI) community, and the safety concerns associated with Large Language Models (LLMs) have been widely investigated. Recently, large vision-language models (VLMs) represent an unprecedented revolution, as they are built upon LLMs but can incorporate additional modalities (e.g., images). However, the safety of VLMs lacks systematic evaluation, and there may be an overconfidence in the safety guarantees provided by their underlying LLMs. In this paper, to demonstrate that introducing additional modality modules leads to unforeseen AI safety issues, we propose FigStep, a straightforward yet effective jailbreaking algorithm against VLMs. Instead of feeding textual harmful instructions directly, FigStep converts the harmful content into images through typography to bypass the safety alignment within the textual module of the VLMs, inducing VLMs to output unsafe responses that violate common AI safety policies. In our evaluation, we manually review 46,500 model responses generated by 3 families of the promising open-source VLMs, i.e., LLaVA, MiniGPT4, and CogVLM (a total of 6 VLMs). The experimental results show that FigStep can achieve an average attack success rate of 82.50% on 500 harmful queries in 10 topics. Moreover, we demonstrate that the methodology of FigStep can even jailbreak GPT-4V, which already leverages an OCR detector to filter harmful queries. Above all, our work reveals that VLMs are vulnerable to jailbreaking attacks, which highlights the necessity of novel safety alignments between visual and textual modalities.

摘要: 确保人工智能生成内容（AIGC）的安全性是人工智能（AI）社区的一个长期话题，与大型语言模型（LLM）相关的安全问题已被广泛研究。最近，大型视觉语言模型（VLM）代表了一场前所未有的革命，因为它们是建立在LLM之上的，但可以包含额外的模态（例如，图像）。然而，VLM的安全性缺乏系统的评估，并且可能对其基础LLM提供的安全保证过于自信。在本文中，为了证明引入额外的模态模块会导致不可预见的AI安全问题，我们提出了FigStep，这是一种针对VLM的简单而有效的越狱算法。FigStep不是直接提供文本有害指令，而是通过排版将有害内容转换为图像，以绕过VLM文本模块内的安全对齐，诱导VLM输出违反常见AI安全策略的不安全响应。在我们的评估中，我们手动审查了由3个有前途的开源VLM家族生成的46，500个模型响应，即，LLaVA、MiniGPT 4和CogVLM（共6个VLM）。实验结果表明，FigStep对10个主题的500个有害查询的平均攻击成功率为82.50%。此外，我们证明了FigStep的方法甚至可以越狱GPT-4V，它已经利用OCR检测器来过滤有害查询。最重要的是，我们的工作表明，VLM容易受到越狱攻击，这突出了视觉和文本形式之间的新的安全对齐的必要性。



## **27. Efficient Representation of the Activation Space in Deep Neural Networks**

深度神经网络中激活空间的有效表示 cs.LG

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.08143v1) [paper-pdf](http://arxiv.org/pdf/2312.08143v1)

**Authors**: Tanya Akumu, Celia Cintas, Girmaw Abebe Tadesse, Adebayo Oshingbesan, Skyler Speakman, Edward McFowland III

**Abstract**: The representations of the activation space of deep neural networks (DNNs) are widely utilized for tasks like natural language processing, anomaly detection and speech recognition. Due to the diverse nature of these tasks and the large size of DNNs, an efficient and task-independent representation of activations becomes crucial. Empirical p-values have been used to quantify the relative strength of an observed node activation compared to activations created by already-known inputs. Nonetheless, keeping raw data for these calculations increases memory resource consumption and raises privacy concerns. To this end, we propose a model-agnostic framework for creating representations of activations in DNNs using node-specific histograms to compute p-values of observed activations without retaining already-known inputs. Our proposed approach demonstrates promising potential when validated with multiple network architectures across various downstream tasks and compared with the kernel density estimates and brute-force empirical baselines. In addition, the framework reduces memory usage by 30% with up to 4 times faster p-value computing time while maintaining state of-the-art detection power in downstream tasks such as the detection of adversarial attacks and synthesized content. Moreover, as we do not persist raw data at inference time, we could potentially reduce susceptibility to attacks and privacy issues.

摘要: 深度神经网络(DNN)的激活空间表示被广泛用于自然语言处理、异常检测和语音识别等任务。由于这些任务的多样性和DNN的巨大规模，高效和独立于任务的激活表示变得至关重要。经验p值被用来量化与已知输入产生的激活相比，观察到的节点激活的相对强度。尽管如此，为这些计算保留原始数据会增加内存资源消耗，并引发隐私问题。为此，我们提出了一个与模型无关的框架，用于使用节点特定的直方图来创建DNN中的激活表示，以计算观察到的激活的p值，而不保留已知的输入。我们提出的方法在不同下游任务的多个网络架构上进行验证，并与核密度估计和蛮力经验基线进行比较，显示出良好的潜力。此外，该框架减少了30%的内存使用量，p值计算时间最多提高了4倍，同时在下游任务中保持了最先进的检测能力，例如检测对抗性攻击和合成内容。此外，由于我们不在推理时保留原始数据，因此我们可能会降低对攻击和隐私问题的易感性。



## **28. PromptBench: A Unified Library for Evaluation of Large Language Models**

PromptBitch：大型语言模型评估的统一库 cs.AI

An extension to PromptBench (arXiv:2306.04528) for unified evaluation  of LLMs using the same name; code: https://github.com/microsoft/promptbench

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.07910v1) [paper-pdf](http://arxiv.org/pdf/2312.07910v1)

**Authors**: Kaijie Zhu, Qinlin Zhao, Hao Chen, Jindong Wang, Xing Xie

**Abstract**: The evaluation of large language models (LLMs) is crucial to assess their performance and mitigate potential security risks. In this paper, we introduce PromptBench, a unified library to evaluate LLMs. It consists of several key components that are easily used and extended by researchers: prompt construction, prompt engineering, dataset and model loading, adversarial prompt attack, dynamic evaluation protocols, and analysis tools. PromptBench is designed to be an open, general, and flexible codebase for research purposes that can facilitate original study in creating new benchmarks, deploying downstream applications, and designing new evaluation protocols. The code is available at: https://github.com/microsoft/promptbench and will be continuously supported.

摘要: 大型语言模型(LLM)的评估对于评估其性能和降低潜在的安全风险至关重要。在本文中，我们介绍了一个用于评估LLMS的统一库PromptBitch.它由几个易于研究人员使用和扩展的关键组件组成：即时构建、即时工程、数据集和模型加载、对抗性即时攻击、动态评估协议和分析工具。PromptBitch是一个开放的、通用的、灵活的研究代码库，可以在创建新的基准、部署下游应用程序和设计新的评估协议方面促进原创研究。该代码可在https://github.com/microsoft/promptbench上获得，并将继续受到支持。



## **29. Causality Analysis for Evaluating the Security of Large Language Models**

大型语言模型安全性评估的因果分析 cs.AI

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.07876v1) [paper-pdf](http://arxiv.org/pdf/2312.07876v1)

**Authors**: Wei Zhao, Zhe Li, Jun Sun

**Abstract**: Large Language Models (LLMs) such as GPT and Llama2 are increasingly adopted in many safety-critical applications. Their security is thus essential. Even with considerable efforts spent on reinforcement learning from human feedback (RLHF), recent studies have shown that LLMs are still subject to attacks such as adversarial perturbation and Trojan attacks. Further research is thus needed to evaluate their security and/or understand the lack of it. In this work, we propose a framework for conducting light-weight causality-analysis of LLMs at the token, layer, and neuron level. We applied our framework to open-source LLMs such as Llama2 and Vicuna and had multiple interesting discoveries. Based on a layer-level causality analysis, we show that RLHF has the effect of overfitting a model to harmful prompts. It implies that such security can be easily overcome by `unusual' harmful prompts. As evidence, we propose an adversarial perturbation method that achieves 100\% attack success rate on the red-teaming tasks of the Trojan Detection Competition 2023. Furthermore, we show the existence of one mysterious neuron in both Llama2 and Vicuna that has an unreasonably high causal effect on the output. While we are uncertain on why such a neuron exists, we show that it is possible to conduct a ``Trojan'' attack targeting that particular neuron to completely cripple the LLM, i.e., we can generate transferable suffixes to prompts that frequently make the LLM produce meaningless responses.

摘要: 大型语言模型(LLM)，如GPT和Llama2，在许多安全关键型应用中越来越多地被采用。因此，他们的安全至关重要。即使在从人类反馈中强化学习(RLHF)方面花费了大量的努力，最近的研究表明LLMS仍然受到诸如对抗性扰动和特洛伊木马攻击的攻击。因此，需要进一步研究，以评估其安全性和/或了解其缺乏安全性。在这项工作中，我们提出了一个框架，用于在标记、层和神经元水平上进行LLMS的轻量级因果分析。我们将我们的框架应用于开源LLM，如Llama2和Vicuna，并有多个有趣的发现。基于层级因果关系分析，我们发现RLHF具有对有害提示的模型过度拟合的效果。这意味着这种安全很容易被“不寻常的”有害提示所克服。作为证据，我们提出了一种对抗性扰动方法，在2023年木马检测大赛的红队任务上达到了100%的攻击成功率。此外，我们证明了在Llama2和Vicuna2中都存在一个神秘的神经元，它对输出具有不合理的高因果效应。虽然我们不确定为什么会有这样的神经元存在，但我们证明了有可能进行针对该特定神经元的“特洛伊木马”攻击，以完全削弱LLM，即我们可以为提示生成可转移的后缀，这些后缀经常使LLM产生无意义的响应。



## **30. DeceptPrompt: Exploiting LLM-driven Code Generation via Adversarial Natural Language Instructions**

DeceptPrompt：通过对抗性自然语言指令利用LLM驱动的代码生成 cs.CR

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.04730v2) [paper-pdf](http://arxiv.org/pdf/2312.04730v2)

**Authors**: Fangzhou Wu, Xiaogeng Liu, Chaowei Xiao

**Abstract**: With the advancement of Large Language Models (LLMs), significant progress has been made in code generation, enabling LLMs to transform natural language into programming code. These Code LLMs have been widely accepted by massive users and organizations. However, a dangerous nature is hidden in the code, which is the existence of fatal vulnerabilities. While some LLM providers have attempted to address these issues by aligning with human guidance, these efforts fall short of making Code LLMs practical and robust. Without a deep understanding of the performance of the LLMs under the practical worst cases, it would be concerning to apply them to various real-world applications. In this paper, we answer the critical issue: Are existing Code LLMs immune to generating vulnerable code? If not, what is the possible maximum severity of this issue in practical deployment scenarios? In this paper, we introduce DeceptPrompt, a novel algorithm that can generate adversarial natural language instructions that drive the Code LLMs to generate functionality correct code with vulnerabilities. DeceptPrompt is achieved through a systematic evolution-based algorithm with a fine grain loss design. The unique advantage of DeceptPrompt enables us to find natural prefix/suffix with totally benign and non-directional semantic meaning, meanwhile, having great power in inducing the Code LLMs to generate vulnerable code. This feature can enable us to conduct the almost-worstcase red-teaming on these LLMs in a real scenario, where users are using natural language. Our extensive experiments and analyses on DeceptPrompt not only validate the effectiveness of our approach but also shed light on the huge weakness of LLMs in the code generation task. When applying the optimized prefix/suffix, the attack success rate (ASR) will improve by average 50% compared with no prefix/suffix applying.

摘要: 随着大型语言模型(LLMS)的发展，在代码生成方面取得了重大进展，使LLMS能够将自然语言转换为编程代码。这些CodeLLM已被广大用户和组织广泛接受。然而，代码中隐藏着一个危险的性质，那就是存在致命的漏洞。虽然一些LLM提供商试图通过与人类的指导保持一致来解决这些问题，但这些努力并不能使Code LLM实用和健壮。如果不深入了解LLMS在实际最坏情况下的性能，将它们应用于各种现实世界应用将是令人担忧的。在这篇文章中，我们回答了一个关键问题：现有的代码LLM是否不会生成易受攻击的代码？如果不是，此问题在实际部署方案中可能的最大严重程度是多少？在本文中，我们介绍了DeceptPrompt算法，它可以生成敌意的自然语言指令，这些指令驱动Code LLMS生成有漏洞的功能正确的代码。DeceptPrompt是通过基于系统进化的算法实现的，具有细粒度的损耗设计。DeceptPrompt的独特优势使我们能够找到具有完全良性和非方向性语义的自然前缀/后缀，同时对诱使Code LLMS生成易受攻击的代码具有强大的能力。这一功能使我们能够在用户使用自然语言的真实场景中对这些LLM进行几乎最糟糕的红色团队。我们在DeceptPrompt上的大量实验和分析不仅验证了我们方法的有效性，而且揭示了LLMS在代码生成任务中的巨大弱点。当应用优化的前缀/后缀时，与不应用前缀/后缀相比，攻击成功率(ASR)将平均提高50%。



## **31. Maatphor: Automated Variant Analysis for Prompt Injection Attacks**

Maatphor：针对即时注入攻击的自动变量分析 cs.CR

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.11513v1) [paper-pdf](http://arxiv.org/pdf/2312.11513v1)

**Authors**: Ahmed Salem, Andrew Paverd, Boris Köpf

**Abstract**: Prompt injection has emerged as a serious security threat to large language models (LLMs). At present, the current best-practice for defending against newly-discovered prompt injection techniques is to add additional guardrails to the system (e.g., by updating the system prompt or using classifiers on the input and/or output of the model.) However, in the same way that variants of a piece of malware are created to evade anti-virus software, variants of a prompt injection can be created to evade the LLM's guardrails. Ideally, when a new prompt injection technique is discovered, candidate defenses should be tested not only against the successful prompt injection, but also against possible variants.   In this work, we present, a tool to assist defenders in performing automated variant analysis of known prompt injection attacks. This involves solving two main challenges: (1) automatically generating variants of a given prompt according, and (2) automatically determining whether a variant was effective based only on the output of the model. This tool can also assist in generating datasets for jailbreak and prompt injection attacks, thus overcoming the scarcity of data in this domain.   We evaluate Maatphor on three different types of prompt injection tasks. Starting from an ineffective (0%) seed prompt, Maatphor consistently generates variants that are at least 60% effective within the first 40 iterations.

摘要: 快速注入已成为大型语言模型(LLM)的严重安全威胁。目前，防御新发现的提示注入技术的最佳实践是向系统添加额外的护栏(例如，通过更新系统提示或使用关于模型的输入和/或输出的分类器)。然而，就像创建恶意软件的变体来逃避反病毒软件一样，也可以创建即时注入的变体来逃避LLM的护栏。理想情况下，当一种新的快速注射技术被发现时，候选防御不仅应该针对成功的快速注射进行测试，而且应该针对可能的变体进行测试。在这项工作中，我们提出了一个工具，以帮助防御者执行自动变异分析已知的即时注入攻击。这涉及解决两个主要挑战：(1)根据给定提示自动生成变体，以及(2)仅基于模型的输出自动确定变体是否有效。该工具还可以帮助生成越狱和提示注入攻击的数据集，从而克服该领域数据稀缺的问题。我们在三种不同类型的快速注射任务中对Maatphor进行了评估。从一个无效的(0%)种子提示开始，Maatphor始终生成在前40次迭代中至少60%有效的变体。



## **32. Practical Membership Inference Attacks against Fine-tuned Large Language Models via Self-prompt Calibration**

基于自提示校正的针对精调大型语言模型的实用隶属度推理攻击 cs.CL

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2311.06062v2) [paper-pdf](http://arxiv.org/pdf/2311.06062v2)

**Authors**: Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, Tao Jiang

**Abstract**: Membership Inference Attacks (MIA) aim to infer whether a target data record has been utilized for model training or not. Prior attempts have quantified the privacy risks of language models (LMs) via MIAs, but there is still no consensus on whether existing MIA algorithms can cause remarkable privacy leakage on practical Large Language Models (LLMs). Existing MIAs designed for LMs can be classified into two categories: reference-free and reference-based attacks. They are both based on the hypothesis that training records consistently strike a higher probability of being sampled. Nevertheless, this hypothesis heavily relies on the overfitting of target models, which will be mitigated by multiple regularization methods and the generalization of LLMs. The reference-based attack seems to achieve promising effectiveness in LLMs, which measures a more reliable membership signal by comparing the probability discrepancy between the target model and the reference model. However, the performance of reference-based attack is highly dependent on a reference dataset that closely resembles the training dataset, which is usually inaccessible in the practical scenario. Overall, existing MIAs are unable to effectively unveil privacy leakage over practical fine-tuned LLMs that are overfitting-free and private. We propose a Membership Inference Attack based on Self-calibrated Probabilistic Variation (SPV-MIA). Specifically, since memorization in LLMs is inevitable during the training process and occurs before overfitting, we introduce a more reliable membership signal, probabilistic variation, which is based on memorization rather than overfitting. Furthermore, we introduce a self-prompt approach, which constructs the dataset to fine-tune the reference model by prompting the target LLM itself. In this manner, the adversary can collect a dataset with a similar distribution from public APIs.

摘要: 成员关系推理攻击(MIA)的目的是推断目标数据记录是否已被用于模型训练。以往的研究已经通过MIA量化了语言模型的隐私风险，但对于现有的MIA算法是否会在实际的大型语言模型上造成显著的隐私泄漏，目前还没有达成共识。现有的针对LMS设计的MIA可以分为两类：无引用攻击和基于引用攻击。它们都是基于这样的假设，即培训记录始终具有更高的被抽样概率。然而，这一假设在很大程度上依赖于目标模型的过度拟合，而多种正则化方法和LLMS的推广将缓解这一问题。基于参考的攻击在LLMS中似乎取得了很好的效果，它通过比较目标模型和参考模型之间的概率差异来衡量更可靠的成员信号。然而，基于参考的攻击的性能高度依赖于与训练数据集非常相似的参考数据集，这在实际场景中通常是不可访问的。总体而言，现有的MIA无法有效地揭示实用的微调LLM的隐私泄露，这些LLM是免装修和私密的。提出了一种基于自校准概率变异的成员推理攻击(SPV-MIA)。具体地说，由于LLMS中的记忆在训练过程中是不可避免的，并且发生在过适应之前，因此我们引入了一种更可靠的隶属度信号-概率变异，它基于记忆而不是过适应。此外，我们引入了一种自我提示的方法，该方法构建数据集，通过提示目标LLM本身来微调参考模型。通过这种方式，攻击者可以从公共API收集具有类似分布的数据集。



## **33. Safety Alignment in NLP Tasks: Weakly Aligned Summarization as an In-Context Attack**

NLP任务中的安全对齐：作为上下文攻击的弱对齐总结 cs.CL

17 pages,10 figures

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.06924v1) [paper-pdf](http://arxiv.org/pdf/2312.06924v1)

**Authors**: Yu Fu, Yufei Li, Wen Xiao, Cong Liu, Yue Dong

**Abstract**: Recent developments in balancing the usefulness and safety of Large Language Models (LLMs) have raised a critical question: Are mainstream NLP tasks adequately aligned with safety consideration? Our study, focusing on safety-sensitive documents obtained through adversarial attacks, reveals significant disparities in the safety alignment of various NLP tasks. For instance, LLMs can effectively summarize malicious long documents but often refuse to translate them. This discrepancy highlights a previously unidentified vulnerability: attacks exploiting tasks with weaker safety alignment, like summarization, can potentially compromise the integraty of tasks traditionally deemed more robust, such as translation and question-answering (QA). Moreover, the concurrent use of multiple NLP tasks with lesser safety alignment increases the risk of LLMs inadvertently processing harmful content. We demonstrate these vulnerabilities in various safety-aligned LLMs, particularly Llama2 models and GPT-4, indicating an urgent need for strengthening safety alignments across a broad spectrum of NLP tasks.

摘要: 最近在平衡大型语言模型(LLM)的有用性和安全性方面的发展提出了一个关键问题：主流NLP任务是否与安全考虑充分一致？我们的研究集中在通过对抗性攻击获得的安全敏感文件上，揭示了各种NLP任务在安全匹配方面的显著差异。例如，LLMS可以有效地汇总恶意的长文档，但通常拒绝翻译它们。这一差异突显了一个以前未知的漏洞：攻击利用安全性较弱的任务(如摘要)，可能会潜在地损害传统上被认为更健壮的任务的完整性，如翻译和问答(QA)。此外，同时使用安全性较低的多个NLP任务会增加LLMS无意中处理有害内容的风险。我们在各种安全对齐的LLM中展示了这些漏洞，特别是Llama2型号和GPT-4，这表明迫切需要在广泛的NLP任务中加强安全对齐。



## **34. GPTBIAS: A Comprehensive Framework for Evaluating Bias in Large Language Models**

GPTBIAS：一个评估大型语言模型中偏差的综合框架 cs.CL

**SubmitDate**: 2023-12-11    [abs](http://arxiv.org/abs/2312.06315v1) [paper-pdf](http://arxiv.org/pdf/2312.06315v1)

**Authors**: Jiaxu Zhao, Meng Fang, Shirui Pan, Wenpeng Yin, Mykola Pechenizkiy

**Abstract**: Warning: This paper contains content that may be offensive or upsetting. There has been a significant increase in the usage of large language models (LLMs) in various applications, both in their original form and through fine-tuned adaptations. As a result, LLMs have gained popularity and are being widely adopted by a large user community. However, one of the concerns with LLMs is the potential generation of socially biased content. The existing evaluation methods have many constraints, and their results exhibit a limited degree of interpretability. In this work, we propose a bias evaluation framework named GPTBIAS that leverages the high performance of LLMs (e.g., GPT-4 \cite{openai2023gpt4}) to assess bias in models. We also introduce prompts called Bias Attack Instructions, which are specifically designed for evaluating model bias. To enhance the credibility and interpretability of bias evaluation, our framework not only provides a bias score but also offers detailed information, including bias types, affected demographics, keywords, reasons behind the biases, and suggestions for improvement. We conduct extensive experiments to demonstrate the effectiveness and usability of our bias evaluation framework.

摘要: 警告：本文包含可能令人反感或不安的内容。在各种应用程序中，大型语言模型（LLM）的使用量显着增加，无论是原始形式还是通过微调调整。因此，LLM越来越受欢迎，并被大型用户社区广泛采用。然而，LLM的一个问题是潜在的社会偏见内容的产生。现有的评价方法有许多限制，其结果表现出有限程度的可解释性。在这项工作中，我们提出了一个名为GPTBIAS的偏差评估框架，该框架利用了LLM的高性能（例如，GPT-4 \cite{openai 2023 gpt 4}）以评估模型中的偏倚。我们还介绍了称为偏差攻击指令的提示，这是专门为评估模型偏差而设计的。为了提高偏见评估的可信度和可解释性，我们的框架不仅提供了偏见评分，还提供了详细的信息，包括偏见类型，受影响的人口统计数据，关键字，偏见背后的原因以及改进建议。我们进行了大量的实验，以证明我们的偏见评估框架的有效性和可用性。



## **35. InferDPT: Privacy-Preserving Inference for Black-box Large Language Model**

InferDPT：黑箱大语言模型的隐私保护推理 cs.CR

**SubmitDate**: 2023-12-11    [abs](http://arxiv.org/abs/2310.12214v5) [paper-pdf](http://arxiv.org/pdf/2310.12214v5)

**Authors**: Meng Tong, Kejiang Chen, Jie Zhang, Yuang Qi, Weiming Zhang, Nenghai Yu

**Abstract**: Large language models (LLMs), like ChatGPT, have greatly simplified text generation tasks. However, they have also raised concerns about privacy risks such as data leakage and unauthorized data collection. Existing solutions for privacy-preserving inference face practical challenges related to computation time and communication costs. In this paper, we propose InferDPT, the first practical framework for the privacy-preserving Inference of black-box LLMs, implementing Differential Privacy in Text generation. InferDPT comprises two key modules: the "perturbation module" utilizes the exponential mechanism to generate a perturbed prompt, facilitating privacy-preserving inference with black-box LLMs, and the "extraction module", inspired by knowledge distillation and retrieval-augmented generation, extracts coherent and consistent text from the perturbed generation result, ensuring successful text generation completion. To address privacy concerns related to previous exponential mechanisms' susceptibility to embedding revision attacks, we introduce RANTEXT, a novel differential privacy mechanism integrated into the perturbation module of InferDPT, which introduces the concept of "RANdom adjacency" for TEXT perturbation within the prompt. Experimental results across three datasets demonstrate that the text generation quality of InferDPT is comparable to that of non-private GPT-4, and RANTEXT surpasses existing state-of-the-art mechanisms, namely, SANTEXT+ and CUSTEXT+ in the trade-off between privacy and utility. Even with an privacy parameter epsilon value of 6.0, RANTEXT achieves an average privacy protection rate exceeding 90% against embedding revision attacks, which is 0.58 times higher than that of SANTEXT+ and 3.35 times higher than that of CUSTEXT+.

摘要: 大型语言模型(LLM)，如ChatGPT，极大地简化了文本生成任务。然而，他们也对数据泄露和未经授权的数据收集等隐私风险表示担忧。现有的隐私保护推理解决方案面临着与计算时间和通信成本相关的实际挑战。在本文中，我们提出了第一个实用的黑盒LLMS隐私保护推理框架InferDPT，在文本生成中实现了差分隐私。InferDPT包括两个关键模块：“扰动模块”利用指数机制生成扰动提示，便于使用黑盒LLMS进行隐私保护推理；“提取模块”受知识提炼和检索-增强生成的启发，从扰动生成结果中提取连贯一致的文本，确保文本生成成功完成。针对以往指数机制易受修改攻击的隐私性问题，引入了一种新的差异化隐私机制RANTEXT，该机制集成在InferDPT的扰动模块中，引入了随机邻接的概念来处理提示内的文本扰动。在三个数据集上的实验结果表明，InferDPT的文本生成质量与非私有GPT-4相当，RANTEXT在隐私和效用之间的权衡方面超过了现有的最新机制SanText+和CUSTEXT+。即使在隐私参数epsilon值为6.0的情况下，RANTEXT对嵌入修改攻击的平均隐私保护率也超过90%，比SanText+高0.58倍，比CUSTEXT+高3.35倍。



## **36. METAL: Metamorphic Testing Framework for Analyzing Large-Language Model Qualities**

Metals：分析大语言模型性质的变形测试框架 cs.SE

Accepted to International Conference on Software Testing,  Verification and Validation (ICST) 2024 / Key words: Large-language models,  Metamorphic testing, Quality evaluation, Text perturbations

**SubmitDate**: 2023-12-11    [abs](http://arxiv.org/abs/2312.06056v1) [paper-pdf](http://arxiv.org/pdf/2312.06056v1)

**Authors**: Sangwon Hyun, Mingyu Guo, M. Ali Babar

**Abstract**: Large-Language Models (LLMs) have shifted the paradigm of natural language data processing. However, their black-boxed and probabilistic characteristics can lead to potential risks in the quality of outputs in diverse LLM applications. Recent studies have tested Quality Attributes (QAs), such as robustness or fairness, of LLMs by generating adversarial input texts. However, existing studies have limited their coverage of QAs and tasks in LLMs and are difficult to extend. Additionally, these studies have only used one evaluation metric, Attack Success Rate (ASR), to assess the effectiveness of their approaches. We propose a MEtamorphic Testing for Analyzing LLMs (METAL) framework to address these issues by applying Metamorphic Testing (MT) techniques. This approach facilitates the systematic testing of LLM qualities by defining Metamorphic Relations (MRs), which serve as modularized evaluation metrics. The METAL framework can automatically generate hundreds of MRs from templates that cover various QAs and tasks. In addition, we introduced novel metrics that integrate the ASR method into the semantic qualities of text to assess the effectiveness of MRs accurately. Through the experiments conducted with three prominent LLMs, we have confirmed that the METAL framework effectively evaluates essential QAs on primary LLM tasks and reveals the quality risks in LLMs. Moreover, the newly proposed metrics can guide the optimal MRs for testing each task and suggest the most effective method for generating MRs.

摘要: 大语言模型（LLM）改变了自然语言数据处理的范式。然而，它们的黑盒和概率特性可能会导致各种LLM应用程序中输出质量的潜在风险。最近的研究已经通过生成对抗性输入文本来测试LLM的质量属性（QA），例如鲁棒性或公平性。然而，现有的研究限制了他们的QA和LLM任务的覆盖范围，很难扩展。此外，这些研究只使用了一个评估指标，攻击成功率（ASR），以评估其方法的有效性。我们提出了一个变形测试分析LLM（金属）框架来解决这些问题，通过应用变形测试（MT）技术。这种方法通过定义作为模块化评估指标的变形关系（MR），促进了LLM质量的系统测试。METAL框架可以从涵盖各种QA和任务的模板中自动生成数百个MR。此外，我们引入了新的指标，将ASR方法集成到文本的语义质量中，以准确评估MR的有效性。通过与三个突出的LLM进行的实验，我们已经证实，金属框架有效地评估主要LLM任务的基本QA，并揭示了LLM的质量风险。此外，新提出的指标可以指导最佳的MR测试每个任务，并建议最有效的方法来生成MR。



## **37. Occlusion-based Detection of Trojan-triggering Inputs in Large Language Models of Code**

大型语言代码模型中基于遮挡的木马触发输入检测 cs.SE

**SubmitDate**: 2023-12-10    [abs](http://arxiv.org/abs/2312.04004v2) [paper-pdf](http://arxiv.org/pdf/2312.04004v2)

**Authors**: Aftab Hussain, Md Rafiqul Islam Rabin, Toufique Ahmed, Mohammad Amin Alipour, Bowen Xu

**Abstract**: Large language models (LLMs) are becoming an integrated part of software development. These models are trained on large datasets for code, where it is hard to verify each data point. Therefore, a potential attack surface can be to inject poisonous data into the training data to make models vulnerable, aka trojaned. It can pose a significant threat by hiding manipulative behaviors inside models, leading to compromising the integrity of the models in downstream tasks.   In this paper, we propose an occlusion-based human-in-the-loop technique, OSeql, to distinguish trojan-triggering inputs of code. The technique is based on the observation that trojaned neural models of code rely heavily on the triggering part of input; hence, its removal would change the confidence of the models in their prediction substantially. Our results suggest that OSeql can detect the triggering inputs with almost 100% recall. We discuss the problem of false positives and how to address them. These results provide a baseline for future studies in this field.

摘要: 大型语言模型(LLM)正在成为软件开发的一个组成部分。这些模型是在大数据集上针对代码进行训练的，在代码中很难验证每个数据点。因此，潜在的攻击面可能是向训练数据中注入有毒数据，使模型容易受到攻击，也就是安装了特洛伊木马。它可以通过将操纵行为隐藏在模型中而构成重大威胁，从而导致在下游任务中损害模型的完整性。在本文中，我们提出了一种基于遮挡的人在环中技术OSeql，用于区分木马触发的代码输入。该技术基于这样的观察，即特洛伊木马代码的神经模型严重依赖于输入的触发部分；因此，移除它将极大地改变模型对其预测的置信度。我们的结果表明，OSeql能够以几乎100%的召回率检测到触发输入。我们讨论了误报问题以及如何解决这些问题。这些结果为该领域未来的研究提供了一个基线。



## **38. Towards Robust Pruning: An Adaptive Knowledge-Retention Pruning Strategy for Language Models**

朝向稳健剪枝：一种自适应的语言模型知识保留剪枝策略 cs.CL

**SubmitDate**: 2023-12-10    [abs](http://arxiv.org/abs/2310.13191v2) [paper-pdf](http://arxiv.org/pdf/2310.13191v2)

**Authors**: Jianwei Li, Qi Lei, Wei Cheng, Dongkuan Xu

**Abstract**: The pruning objective has recently extended beyond accuracy and sparsity to robustness in language models. Despite this, existing methods struggle to enhance robustness against adversarial attacks when continually increasing model sparsity and require a retraining process. As humans step into the era of large language models, these issues become increasingly prominent. This paper proposes that the robustness of language models is proportional to the extent of pre-trained knowledge they encompass. Accordingly, we introduce a post-training pruning strategy designed to faithfully replicate the embedding space and feature space of dense language models, aiming to conserve more pre-trained knowledge during the pruning process. In this setup, each layer's reconstruction error not only originates from itself but also includes cumulative error from preceding layers, followed by an adaptive rectification. Compared to other state-of-art baselines, our approach demonstrates a superior balance between accuracy, sparsity, robustness, and pruning cost with BERT on datasets SST2, IMDB, and AGNews, marking a significant stride towards robust pruning in language models.

摘要: 修剪目标最近已经超越了语言模型中的精确度和稀疏性，扩展到了健壮性。尽管如此，现有的方法在不断增加模型稀疏性的同时努力增强对敌对攻击的鲁棒性，并且需要重新训练过程。随着人类步入大型语言模型时代，这些问题变得日益突出。本文提出语言模型的稳健性与它们所包含的预训练知识的程度成正比。因此，我们提出了一种训练后剪枝策略，旨在忠实地复制密集语言模型的嵌入空间和特征空间，目的是在剪枝过程中保存更多的预先训练的知识。在这种设置中，每一层的重建误差不仅源于自身，还包括来自前几层的累积误差，然后进行自适应校正。与其他最先进的基线相比，我们的方法在精确度、稀疏性、健壮性和剪枝成本之间表现出了更好的平衡，在数据集Sst2、IMDB和AgNews上使用ERT，标志着在语言模型中朝着健壮剪枝迈出了重要的一步。



## **39. Temporal-Distributed Backdoor Attack Against Video Based Action Recognition**

针对视频动作识别的时间分布式后门攻击 cs.CV

accepted by AAAI 2024

**SubmitDate**: 2023-12-09    [abs](http://arxiv.org/abs/2308.11070v3) [paper-pdf](http://arxiv.org/pdf/2308.11070v3)

**Authors**: Xi Li, Songhe Wang, Ruiquan Huang, Mahanth Gowda, George Kesidis

**Abstract**: Deep neural networks (DNNs) have achieved tremendous success in various applications including video action recognition, yet remain vulnerable to backdoor attacks (Trojans). The backdoor-compromised model will mis-classify to the target class chosen by the attacker when a test instance (from a non-target class) is embedded with a specific trigger, while maintaining high accuracy on attack-free instances. Although there are extensive studies on backdoor attacks against image data, the susceptibility of video-based systems under backdoor attacks remains largely unexplored. Current studies are direct extensions of approaches proposed for image data, e.g., the triggers are independently embedded within the frames, which tend to be detectable by existing defenses. In this paper, we introduce a simple yet effective backdoor attack against video data. Our proposed attack, adding perturbations in a transformed domain, plants an imperceptible, temporally distributed trigger across the video frames, and is shown to be resilient to existing defensive strategies. The effectiveness of the proposed attack is demonstrated by extensive experiments with various well-known models on two video recognition benchmarks, UCF101 and HMDB51, and a sign language recognition benchmark, Greek Sign Language (GSL) dataset. We delve into the impact of several influential factors on our proposed attack and identify an intriguing effect termed "collateral damage" through extensive studies.

摘要: 深度神经网络(DNN)在包括视频动作识别在内的各种应用中取得了巨大的成功，但仍然容易受到后门攻击(特洛伊木马)。当测试实例(来自非目标类)嵌入特定触发器时，后门泄露模型将被错误分类为攻击者选择的目标类，同时保持对无攻击实例的高准确性。虽然已经有大量关于针对图像数据的后门攻击的研究，但基于视频的系统在后门攻击下的易感性在很大程度上仍未被探索。当前的研究是对为图像数据提出的方法的直接扩展，例如，触发器独立地嵌入在帧中，这往往是现有防御系统可检测的。本文介绍了一种简单而有效的针对视频数据的后门攻击。我们提出的攻击在变换的域中增加了扰动，在视频帧上植入了一个不可察觉的、时间分布的触发器，并被证明对现有的防御策略具有弹性。在两个视频识别基准UCF101和HMDB51和一个手语识别基准希腊手语(GSL)数据集上进行了大量的实验，证明了所提出的攻击的有效性。我们深入研究了几个影响因素对我们提出的攻击的影响，并通过广泛的研究确定了一种有趣的影响，称为“附带损害”。



## **40. HuRef: HUman-REadable Fingerprint for Large Language Models**

HuRef：大型语言模型的人类可读指纹 cs.CL

**SubmitDate**: 2023-12-08    [abs](http://arxiv.org/abs/2312.04828v1) [paper-pdf](http://arxiv.org/pdf/2312.04828v1)

**Authors**: Boyi Zeng, Chenghu Zhou, Xinbing Wang, Zhouhan Lin

**Abstract**: Protecting the copyright of large language models (LLMs) has become crucial due to their resource-intensive training and accompanying carefully designed licenses. However, identifying the original base model of an LLM is challenging due to potential parameter alterations through fine-tuning or continued pretraining. In this study, we introduce HuRef, a human-readable fingerprint for LLMs that uniquely identifies the base model without exposing model parameters or interfering with training. We first observe that the vector direction of LLM parameters remains stable after the model has converged during pretraining, showing negligible perturbations through subsequent training steps, including continued pretraining, supervised fine-tuning (SFT), and RLHF, which makes it a sufficient condition to identify the base model. The necessity is validated by continuing to train an LLM with an extra term to drive away the model parameters' direction and the model becomes damaged. However, this direction is vulnerable to simple attacks like dimension permutation or matrix rotation, which significantly change it without affecting performance. To address this, leveraging the Transformer structure, we systematically analyze potential attacks and define three invariant terms that identify an LLM's base model. We make these invariant terms human-readable by mapping them to a Gaussian vector using a convolutional encoder and then converting it into a natural image with StyleGAN2. Our method generates a dog image as an identity fingerprint for an LLM, where the dog's appearance strongly indicates the LLM's base model. Experimental results across various LLMs demonstrate the effectiveness of our method, the generated dog image remains invariant to different training steps, including SFT, RLHF, or even continued pretraining with augmented vocabulary in a new language.

摘要: 保护大型语言模型(LLM)的版权已变得至关重要，因为它们需要进行资源密集型培训，并附带精心设计的许可证。然而，识别LLM的原始基本模型是具有挑战性的，因为通过微调或持续的预训练可能会改变参数。在这项研究中，我们引入了HuRef，这是一种用于LLMS的人类可读指纹，它在不暴露模型参数或干扰训练的情况下唯一地识别基本模型。我们首先观察到，在预训练过程中，模型收敛后，LLM参数的向量方向保持稳定，通过后续的训练步骤，包括继续预训练、有监督微调(SFT)和RLHF，表现出可以忽略的扰动，这使得它成为识别基本模型的充分条件。通过继续训练一个带有额外项的LLM来驱离模型参数的方向，从而使模型受损，从而验证了这种必要性。然而，这个方向很容易受到维度置换或矩阵旋转等简单攻击，这些攻击会在不影响性能的情况下显著改变它。为了解决这个问题，利用Transformer结构，我们系统地分析了潜在的攻击，并定义了识别LLM基本模型的三个不变术语。我们使用卷积编码器将这些不变项映射到高斯向量，然后使用StyleGAN2将其转换为自然图像，从而使这些不变项变得可读。我们的方法生成了一幅狗图像作为LLM的身份指纹，其中狗的外表强烈地指示了LLM的基本模型。在不同LLMS上的实验结果证明了该方法的有效性，生成的狗图像在不同的训练步骤中保持不变，包括SFT、RLHF，甚至是在新语言中增加词汇量的持续预训练。



## **41. Goal-Oriented Prompt Attack and Safety Evaluation for LLMs**

面向目标的低空导弹快速攻击与安全评估 cs.CL

**SubmitDate**: 2023-12-08    [abs](http://arxiv.org/abs/2309.11830v2) [paper-pdf](http://arxiv.org/pdf/2309.11830v2)

**Authors**: Chengyuan Liu, Fubang Zhao, Lizhi Qing, Yangyang Kang, Changlong Sun, Kun Kuang, Fei Wu

**Abstract**: Large Language Models (LLMs) presents significant priority in text understanding and generation. However, LLMs suffer from the risk of generating harmful contents especially while being employed to applications. There are several black-box attack methods, such as Prompt Attack, which can change the behaviour of LLMs and induce LLMs to generate unexpected answers with harmful contents. Researchers are interested in Prompt Attack and Defense with LLMs, while there is no publicly available dataset with high successful attacking rate to evaluate the abilities of defending prompt attack. In this paper, we introduce a pipeline to construct high-quality prompt attack samples, along with a Chinese prompt attack dataset called CPAD. Our prompts aim to induce LLMs to generate unexpected outputs with several carefully designed prompt attack templates and widely concerned attacking contents. Different from previous datasets involving safety estimation, we construct the prompts considering three dimensions: contents, attacking methods and goals. Especially, the attacking goals indicate the behaviour expected after successfully attacking the LLMs, thus the responses can be easily evaluated and analysed. We run several popular Chinese LLMs on our dataset, and the results show that our prompts are significantly harmful to LLMs, with around 70% attack success rate to GPT-3.5. CPAD is publicly available at https://github.com/liuchengyuan123/CPAD.

摘要: 大语言模型(LLM)在文本理解和生成中具有重要的优先地位。然而，LLMS面临着产生有害内容的风险，特别是在应用程序中使用时。有几种黑盒攻击方法，如提示攻击，可以改变LLMS的行为，并诱导LLMS生成包含有害内容的意外答案。研究人员对LLMS的快速攻防很感兴趣，但目前还没有公开的、具有较高攻击成功率的数据集来评估防御快速攻击的能力。在本文中，我们介绍了一种构造高质量即时攻击样本的管道，以及一个中文即时攻击数据集CPAD。我们的提示旨在通过精心设计的几个提示攻击模板和广泛关注的攻击内容来诱导LLM产生意想不到的输出。与以往涉及安全评估的数据集不同，我们从内容、攻击方法和目标三个维度构建提示。特别是，攻击目标指示了成功攻击LLMS后的预期行为，因此可以很容易地评估和分析响应。我们在我们的数据集上运行了几个流行的中文LLMS，结果表明我们的提示对LLMS具有显著的危害，对GPT-3.5的攻击成功率约为70%。CPAD可在https://github.com/liuchengyuan123/CPAD.上公开购买



## **42. Make Them Spill the Beans! Coercive Knowledge Extraction from (Production) LLMs**

让他们说漏嘴！从(产生式)LLMS中提取强制知识 cs.CR

**SubmitDate**: 2023-12-08    [abs](http://arxiv.org/abs/2312.04782v1) [paper-pdf](http://arxiv.org/pdf/2312.04782v1)

**Authors**: Zhuo Zhang, Guangyu Shen, Guanhong Tao, Siyuan Cheng, Xiangyu Zhang

**Abstract**: Large Language Models (LLMs) are now widely used in various applications, making it crucial to align their ethical standards with human values. However, recent jail-breaking methods demonstrate that this alignment can be undermined using carefully constructed prompts. In our study, we reveal a new threat to LLM alignment when a bad actor has access to the model's output logits, a common feature in both open-source LLMs and many commercial LLM APIs (e.g., certain GPT models). It does not rely on crafting specific prompts. Instead, it exploits the fact that even when an LLM rejects a toxic request, a harmful response often hides deep in the output logits. By forcefully selecting lower-ranked output tokens during the auto-regressive generation process at a few critical output positions, we can compel the model to reveal these hidden responses. We term this process model interrogation. This approach differs from and outperforms jail-breaking methods, achieving 92% effectiveness compared to 62%, and is 10 to 20 times faster. The harmful content uncovered through our method is more relevant, complete, and clear. Additionally, it can complement jail-breaking strategies, with which results in further boosting attack performance. Our findings indicate that interrogation can extract toxic knowledge even from models specifically designed for coding tasks.

摘要: 大型语言模型(LLM)现在被广泛应用于各种应用中，因此使它们的伦理标准与人类价值观保持一致至关重要。然而，最近的越狱方法表明，使用精心构建的提示可以破坏这种对齐。在我们的研究中，我们揭示了当一个坏的参与者可以访问模型的输出日志时对LLM对齐的新威胁，这是开源LLMS和许多商业LLMAPI(例如，某些GPT模型)中的一个共同特征。它不依赖于精心设计特定的提示。相反，它利用了这样一个事实，即即使LLM拒绝了有毒请求，有害的响应通常也隐藏在输出日志的深处。通过在自回归生成过程中在几个关键的输出位置强制选择较低等级的输出令牌，我们可以迫使模型揭示这些隐藏的响应。我们称这一过程为审问模式。这种方法与越狱方法不同，而且性能优于越狱方法，达到92%的有效率，而不是62%，而且速度快10到20倍。通过我们的方法发现的有害内容更相关、更完整、更清晰。此外，它还可以补充越狱策略，从而进一步提高攻击性能。我们的发现表明，审问甚至可以从专门为编码任务设计的模型中提取有毒知识。



## **43. Forcing Generative Models to Degenerate Ones: The Power of Data Poisoning Attacks**

迫使生成性模型退化：数据中毒攻击的威力 cs.CR

19 pages, 6 figures. Published at NeurIPS 2023 Workshop on Backdoors  in Deep Learning: The Good, the Bad, and the Ugly

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2312.04748v1) [paper-pdf](http://arxiv.org/pdf/2312.04748v1)

**Authors**: Shuli Jiang, Swanand Ravindra Kadhe, Yi Zhou, Ling Cai, Nathalie Baracaldo

**Abstract**: Growing applications of large language models (LLMs) trained by a third party raise serious concerns on the security vulnerability of LLMs.It has been demonstrated that malicious actors can covertly exploit these vulnerabilities in LLMs through poisoning attacks aimed at generating undesirable outputs. While poisoning attacks have received significant attention in the image domain (e.g., object detection), and classification tasks, their implications for generative models, particularly in the realm of natural language generation (NLG) tasks, remain poorly understood. To bridge this gap, we perform a comprehensive exploration of various poisoning techniques to assess their effectiveness across a range of generative tasks. Furthermore, we introduce a range of metrics designed to quantify the success and stealthiness of poisoning attacks specifically tailored to NLG tasks. Through extensive experiments on multiple NLG tasks, LLMs and datasets, we show that it is possible to successfully poison an LLM during the fine-tuning stage using as little as 1\% of the total tuning data samples. Our paper presents the first systematic approach to comprehend poisoning attacks targeting NLG tasks considering a wide range of triggers and attack settings. We hope our findings will assist the AI security community in devising appropriate defenses against such threats.

摘要: 由第三方训练的大型语言模型(LLM)的应用日益增多，引起了人们对LLM安全漏洞的严重关注，已有研究表明，恶意行为者可以通过投毒攻击来秘密利用LLM中的这些漏洞，目的是产生不希望看到的输出。虽然中毒攻击在图像领域(例如，目标检测)和分类任务中受到了极大的关注，但它们对生成模型的影响，特别是在自然语言生成(NLG)任务领域，仍然知之甚少。为了弥补这一差距，我们对各种中毒技术进行了全面的探索，以评估它们在一系列生成性任务中的有效性。此外，我们还介绍了一系列专门为NLG任务量身定做的用于量化投毒攻击的成功率和隐蔽性的指标。通过在多个NLG任务、LLM和数据集上的大量实验，我们表明，在微调阶段，只要使用总调整数据样本的1%，就可以成功地毒化LLM。我们提出了第一种系统的方法来理解针对NLG任务的中毒攻击，考虑了广泛的触发因素和攻击设置。我们希望我们的发现将有助于人工智能安全界设计针对此类威胁的适当防御措施。



## **44. Defending Against Alignment-Breaking Attacks via Robustly Aligned LLM**

通过强健对齐的LLM防御对齐破坏攻击 cs.CL

16 Pages, 5 Figures, 6 Tables

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2309.14348v2) [paper-pdf](http://arxiv.org/pdf/2309.14348v2)

**Authors**: Bochuan Cao, Yuanpu Cao, Lu Lin, Jinghui Chen

**Abstract**: Recently, Large Language Models (LLMs) have made significant advancements and are now widely used across various domains. Unfortunately, there has been a rising concern that LLMs can be misused to generate harmful or malicious content. Though a line of research has focused on aligning LLMs with human values and preventing them from producing inappropriate content, such alignments are usually vulnerable and can be bypassed by alignment-breaking attacks via adversarially optimized or handcrafted jailbreaking prompts. In this work, we introduce a Robustly Aligned LLM (RA-LLM) to defend against potential alignment-breaking attacks. RA-LLM can be directly constructed upon an existing aligned LLM with a robust alignment checking function, without requiring any expensive retraining or fine-tuning process of the original LLM. Furthermore, we also provide a theoretical analysis for RA-LLM to verify its effectiveness in defending against alignment-breaking attacks. Through real-world experiments on open-source large language models, we demonstrate that RA-LLM can successfully defend against both state-of-the-art adversarial prompts and popular handcrafted jailbreaking prompts by reducing their attack success rates from nearly 100% to around 10% or less.

摘要: 近年来，大型语言模型(LLM)取得了长足的进步，现已广泛应用于各个领域。不幸的是，人们越来越担心LLMS可能被滥用来生成有害或恶意的内容。尽管有一系列研究专注于将LLM与人类价值观保持一致，并防止它们产生不适当的内容，但这种调整通常是脆弱的，可以通过恶意优化或手工制作的越狱提示被破坏顺序的攻击绕过。在这项工作中，我们引入了一种鲁棒对齐LLM(RA-LLM)来防御潜在的对齐破坏攻击。RA-LLM可以直接构建在现有的对准LLM上，具有健壮的对准检查功能，而不需要对原始LLM进行任何昂贵的再培训或微调过程。此外，我们还对RA-LLM进行了理论分析，以验证其在抵抗对齐破坏攻击方面的有效性。通过在开源大型语言模型上的真实世界实验，我们证明了RA-LLM能够成功地防御最新的敌意提示和流行的手工越狱提示，将攻击成功率从近100%降低到10%左右或更低。



## **45. Domain Private Transformers for Multi-Domain Dialog Systems**

用于多域对话系统的域专用转换器 cs.CL

Accepted to Findings of EMNLP 2023 (short paper). Code available at  https://github.com/asappresearch/domain-private-transformers

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2305.14208v2) [paper-pdf](http://arxiv.org/pdf/2305.14208v2)

**Authors**: Anmol Kabra, Ethan R. Elenberg

**Abstract**: Large, general purpose language models have demonstrated impressive performance across many different conversational domains. While multi-domain language models achieve low overall perplexity, their outputs are not guaranteed to stay within the domain of a given input prompt. This paper proposes domain privacy as a novel way to quantify how likely a conditional language model will leak across domains. We also develop policy functions based on token-level domain classification, and propose an efficient fine-tuning method to improve the trained model's domain privacy. Experiments on membership inference attacks show that our proposed method has comparable resiliency to methods adapted from recent literature on differentially private language models.

摘要: 大型通用语言模型在许多不同的会话领域中表现出令人印象深刻的性能。虽然多域语言模型实现了低的整体困惑，但它们的输出不能保证保持在给定输入提示的域内。本文提出域隐私作为一种新的方式来量化有条件的语言模型将跨域泄漏的可能性。我们还开发了基于令牌级域分类的策略函数，并提出了一种有效的微调方法来提高训练模型的域隐私性。成员推理攻击的实验表明，我们提出的方法具有相当的弹性，从最近的文献中改编的方法差异私人语言模型。



## **46. Analyzing the Inherent Response Tendency of LLMs: Real-World Instructions-Driven Jailbreak**

LLM的内在响应趋势分析：现实世界的教学驱动越狱 cs.CL

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2312.04127v1) [paper-pdf](http://arxiv.org/pdf/2312.04127v1)

**Authors**: Yanrui Du, Sendong Zhao, Ming Ma, Yuhan Chen, Bing Qin

**Abstract**: Extensive work has been devoted to improving the safety mechanism of Large Language Models (LLMs). However, in specific scenarios, LLMs still generate harmful responses when faced with malicious instructions, a phenomenon referred to as "Jailbreak Attack". In our research, we introduce a novel jailbreak attack method (\textbf{RADIAL}), which consists of two steps: 1) Inherent Response Tendency Analysis: we analyze the inherent affirmation and rejection tendency of LLMs to react to real-world instructions. 2) Real-World Instructions-Driven Jailbreak: based on our analysis, we strategically choose several real-world instructions and embed malicious instructions into them to amplify the LLM's potential to generate harmful responses. On three open-source human-aligned LLMs, our method achieves excellent jailbreak attack performance for both Chinese and English malicious instructions. Besides, we guided detailed ablation experiments and verified the effectiveness of our core idea "Inherent Response Tendency Analysis". Our exploration also exposes the vulnerability of LLMs to being induced into generating more detailed harmful responses in subsequent rounds of dialogue.

摘要: 人们在改进大型语言模型(LLM)的安全机制方面做了大量的工作。然而，在特定场景下，LLMS在面临恶意指令时仍然会产生有害的响应，这种现象被称为“越狱攻击”。在我们的研究中，我们介绍了一种新的越狱攻击方法(Textbf{Radial})，该方法包括两个步骤：1)内在响应趋势分析：分析LLM对现实世界指令做出反应的内在肯定和拒绝倾向。2)真实世界指令驱动越狱：基于我们的分析，我们有策略地选择了几条真实世界的指令，并在其中嵌入恶意指令，以放大LLM产生有害响应的潜力。在三个开源的人类对齐的LLMS上，我们的方法对中文和英文恶意指令都取得了良好的越狱攻击性能。此外，我们还指导了详细的烧蚀实验，验证了“固有响应趋势分析”这一核心思想的有效性。我们的探索还暴露了小岛屿发展中国家在随后几轮对话中被诱使产生更详细的有害反应的脆弱性。



## **47. Mark My Words: Analyzing and Evaluating Language Model Watermarks**

记住我的话：分析和评估语言模型水印 cs.CR

18 pages, 11 figures

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2312.00273v2) [paper-pdf](http://arxiv.org/pdf/2312.00273v2)

**Authors**: Julien Piet, Chawin Sitawarin, Vivian Fang, Norman Mu, David Wagner

**Abstract**: The capabilities of large language models have grown significantly in recent years and so too have concerns about their misuse. In this context, the ability to distinguish machine-generated text from human-authored content becomes important. Prior works have proposed numerous schemes to watermark text, which would benefit from a systematic evaluation framework. This work focuses on text watermarking techniques - as opposed to image watermarks - and proposes MARKMYWORDS, a comprehensive benchmark for them under different tasks as well as practical attacks. We focus on three main metrics: quality, size (e.g. the number of tokens needed to detect a watermark), and tamper-resistance. Current watermarking techniques are good enough to be deployed: Kirchenbauer et al. [1] can watermark Llama2-7B-chat with no perceivable loss in quality, the watermark can be detected with fewer than 100 tokens, and the scheme offers good tamper-resistance to simple attacks. We argue that watermark indistinguishability, a criteria emphasized in some prior works, is too strong a requirement: schemes that slightly modify logit distributions outperform their indistinguishable counterparts with no noticeable loss in generation quality. We publicly release our benchmark (https://github.com/wagner-group/MarkMyWords)

摘要: 近年来，大型语言模型的能力显著增长，人们对它们的滥用也感到担忧。在这种情况下，区分机器生成的文本和人类创作的内容的能力变得很重要。以前的工作已经提出了许多方案来对文本进行水印，这将受益于一个系统的评估框架。这项工作的重点是文本水印技术，而不是图像水印，并提出了MARKMYWORDS，一个针对不同任务和实际攻击的综合基准。我们主要关注三个指标：质量、大小(例如，检测水印所需的令牌数量)和防篡改。目前的水印技术足够好，可以部署：Kirchenbauer等人。[1]可以在Llama2-7B-Chat上嵌入水印而不会造成明显的质量损失，水印的检测只需要不到100个令牌，并且对简单攻击具有良好的抗篡改能力。我们认为，水印的不可区分性是一个太强的要求：稍微修改Logit分布的方案在生成质量上没有明显损失的情况下，性能优于它们的不可区分的对应方案。我们公开发布我们的基准(https://github.com/wagner-group/MarkMyWords)



## **48. Dr. Jekyll and Mr. Hyde: Two Faces of LLMs**

杰基尔博士和海德先生：LLMS的两张面孔 cs.CR

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2312.03853v1) [paper-pdf](http://arxiv.org/pdf/2312.03853v1)

**Authors**: Matteo Gioele Collu, Tom Janssen-Groesbeek, Stefanos Koffas, Mauro Conti, Stjepan Picek

**Abstract**: This year, we witnessed a rise in the use of Large Language Models, especially when combined with applications like chatbot assistants. Safety mechanisms and specialized training procedures are put in place to prevent improper responses from these assistants. In this work, we bypass these measures for ChatGPT and Bard (and, to some extent, Bing chat) by making them impersonate complex personas with opposite characteristics as those of the truthful assistants they are supposed to be. We start by creating elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversation followed a role-play style to get the response the assistant was not allowed to provide. By making use of personas, we show that the response that is prohibited is actually provided, making it possible to obtain unauthorized, illegal, or harmful information. This work shows that by using adversarial personas, one can overcome safety mechanisms set out by ChatGPT and Bard. It also introduces several ways of activating such adversarial personas, altogether showing that both chatbots are vulnerable to this kind of attack.

摘要: 今年，我们见证了大型语言模型的使用增加，特别是当与聊天机器人助手等应用相结合时。本集团设有安全机制及专门培训程序，以防止该等助理作出不当反应。在这项工作中，我们绕过了ChatGPT和Bard（以及在某种程度上，Bing聊天）的这些措施，让他们模仿复杂的人物角色，这些人物角色具有与他们应该成为的真实助手相反的特征。我们首先创建这些人物角色的详细传记，然后在与相同聊天机器人的新会话中使用。我们的谈话遵循角色扮演的风格，以获得助理不允许提供的回应。通过使用人物角色，我们表明实际上提供了被禁止的响应，从而有可能获得未经授权的，非法的或有害的信息。这项工作表明，通过使用对抗性人物角色，可以克服ChatGPT和Bard提出的安全机制。它还介绍了激活这种对抗性角色的几种方法，共同表明这两种聊天机器人都容易受到这种攻击。



## **49. Clinical Notes Reveal Physician Fatigue**

临床笔记显示医生疲劳 cs.CL

**SubmitDate**: 2023-12-05    [abs](http://arxiv.org/abs/2312.03077v1) [paper-pdf](http://arxiv.org/pdf/2312.03077v1)

**Authors**: Chao-Chun Hsu, Ziad Obermeyer, Chenhao Tan

**Abstract**: Physicians write notes about patients. In doing so, they reveal much about themselves. Using data from 129,228 emergency room visits, we train a model to identify notes written by fatigued physicians -- those who worked 5 or more of the prior 7 days. In a hold-out set, the model accurately identifies notes written by these high-workload physicians, and also flags notes written in other high-fatigue settings: on overnight shifts, and after high patient volumes. Model predictions also correlate with worse decision-making on at least one important metric: yield of testing for heart attack is 18% lower with each standard deviation increase in model-predicted fatigue. Finally, the model indicates that notes written about Black and Hispanic patients have 12% and 21% higher predicted fatigue than Whites -- larger than overnight vs. daytime differences. These results have an important implication for large language models (LLMs). Our model indicates that fatigued doctors write more predictable notes. Perhaps unsurprisingly, because word prediction is the core of how LLMs work, we find that LLM-written notes have 17% higher predicted fatigue than real physicians' notes. This indicates that LLMs may introduce distortions in generated text that are not yet fully understood.

摘要: 医生为病人写便条。在这样做的过程中，他们透露了很多关于自己的信息。使用129,228次急诊室就诊的数据，我们训练了一个模型来识别疲惫的医生写的笔记--那些在之前的7天中工作了5天或更多的医生。在坚持设置中，该模型准确地识别这些高工作量医生所写的笔记，并标记在其他高疲劳度环境中编写的笔记：在夜间轮班时，以及在高病人量之后。模型预测还与至少一个重要指标上的较差决策相关：模型预测疲劳的标准差每增加一次，心脏病发作测试的收益率就会降低18%。最后，该模型表明，写给黑人和西班牙裔患者的纸条比白人分别高出12%和21%的预期疲劳感--比夜间和白天的差异更大。这些结果对大型语言模型(LLM)具有重要的意义。我们的模型表明，疲惫的医生写的笔记更容易预测。也许并不令人惊讶的是，因为单词预测是LLMS工作原理的核心，我们发现LLM写的笔记比真正的医生笔记预测的疲劳感高17%。这表明LLMS可能会在生成的文本中引入尚未完全理解的扭曲。



## **50. Tree of Attacks: Jailbreaking Black-Box LLMs Automatically**

攻击之树：自动越狱黑盒LLMS cs.LG

An implementation of the presented method is available at  https://github.com/RICommunity/TAP

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.02119v1) [paper-pdf](http://arxiv.org/pdf/2312.02119v1)

**Authors**: Anay Mehrotra, Manolis Zampetakis, Paul Kassianik, Blaine Nelson, Hyrum Anderson, Yaron Singer, Amin Karbasi

**Abstract**: While Large Language Models (LLMs) display versatile functionality, they continue to generate harmful, biased, and toxic content, as demonstrated by the prevalence of human-designed jailbreaks. In this work, we present Tree of Attacks with Pruning (TAP), an automated method for generating jailbreaks that only requires black-box access to the target LLM. TAP utilizes an LLM to iteratively refine candidate (attack) prompts using tree-of-thoughts reasoning until one of the generated prompts jailbreaks the target. Crucially, before sending prompts to the target, TAP assesses them and prunes the ones unlikely to result in jailbreaks. Using tree-of-thought reasoning allows TAP to navigate a large search space of prompts and pruning reduces the total number of queries sent to the target. In empirical evaluations, we observe that TAP generates prompts that jailbreak state-of-the-art LLMs (including GPT4 and GPT4-Turbo) for more than 80% of the prompts using only a small number of queries. This significantly improves upon the previous state-of-the-art black-box method for generating jailbreaks.

摘要: 虽然大型语言模型(LLM)显示了多功能，但它们继续产生有害、有偏见和有毒的内容，人类设计的越狱事件的流行就证明了这一点。在这项工作中，我们提出了带修剪的攻击树(TAP)，这是一种自动生成越狱的方法，只需要通过黑盒访问目标LLM。TAP利用LLM使用思想树推理反复优化候选(攻击)提示，直到其中一个生成的提示越狱目标。至关重要的是，在向目标发送提示之前，TAP会对它们进行评估，并删除那些不太可能导致越狱的提示。使用思维树推理允许TAP导航大的提示搜索空间，并进行修剪以减少发送到目标的查询总数。在经验评估中，我们观察到TAP仅使用少量查询就为80%以上的提示生成了越狱最先进的LLM(包括GPT4和GPT4-Turbo)提示。这大大改进了以前用于生成越狱的最先进的黑匣子方法。



