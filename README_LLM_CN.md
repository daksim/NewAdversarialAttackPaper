# Latest Large Language Model Attack Papers
**update at 2025-02-05 12:08:58**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Medical Multimodal Model Stealing Attacks via Adversarial Domain Alignment**

医学多模式模型通过对抗领域对齐窃取攻击 cs.CR

Accepted at AAAI 2025

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02438v1) [paper-pdf](http://arxiv.org/pdf/2502.02438v1)

**Authors**: Yaling Shen, Zhixiong Zhuang, Kun Yuan, Maria-Irina Nicolae, Nassir Navab, Nicolas Padoy, Mario Fritz

**Abstract**: Medical multimodal large language models (MLLMs) are becoming an instrumental part of healthcare systems, assisting medical personnel with decision making and results analysis. Models for radiology report generation are able to interpret medical imagery, thus reducing the workload of radiologists. As medical data is scarce and protected by privacy regulations, medical MLLMs represent valuable intellectual property. However, these assets are potentially vulnerable to model stealing, where attackers aim to replicate their functionality via black-box access. So far, model stealing for the medical domain has focused on classification; however, existing attacks are not effective against MLLMs. In this paper, we introduce Adversarial Domain Alignment (ADA-STEAL), the first stealing attack against medical MLLMs. ADA-STEAL relies on natural images, which are public and widely available, as opposed to their medical counterparts. We show that data augmentation with adversarial noise is sufficient to overcome the data distribution gap between natural images and the domain-specific distribution of the victim MLLM. Experiments on the IU X-RAY and MIMIC-CXR radiology datasets demonstrate that Adversarial Domain Alignment enables attackers to steal the medical MLLM without any access to medical data.

摘要: 医疗多模式大型语言模型(MLLMS)正在成为医疗保健系统的重要组成部分，帮助医务人员进行决策和结果分析。放射学报告生成模型能够解释医学图像，从而减少了放射科医生的工作量。由于医疗数据稀缺，而且受到隐私法规的保护，医疗MLLM代表着宝贵的知识产权。然而，这些资产可能容易受到模型窃取的攻击，攻击者的目标是通过黑盒访问来复制它们的功能。到目前为止，针对医学领域的模型窃取主要集中在分类上，然而，现有的攻击对MLLMS并不有效。在本文中，我们介绍了第一个针对医学MLLM的窃取攻击--对抗性领域对齐(ADA-Steal)。Ada-steal依赖于自然图像，这些图像是公开的，可以广泛使用，而不是医学上的同行。我们证明了使用对抗性噪声的数据增强足以克服自然图像和受害者MLLM的特定领域分布之间的数据分布差距。在Iu-X-Ray和MIMIC-CXR放射学数据集上的实验表明，对抗性领域对齐使攻击者能够在不访问任何医疗数据的情况下窃取医疗MLLM。



## **2. JailbreakEval: An Integrated Toolkit for Evaluating Jailbreak Attempts Against Large Language Models**

越狱Eval：用于评估针对大型语言模型的越狱尝试的集成工具包 cs.CR

This is the Extended Version for the Poster at NDSS Symposium 2025,  Feb 24-28, 2025. Our code is available at  https://github.com/ThuCCSLab/JailbreakEval

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2406.09321v2) [paper-pdf](http://arxiv.org/pdf/2406.09321v2)

**Authors**: Delong Ran, Jinyuan Liu, Yichen Gong, Jingyi Zheng, Xinlei He, Tianshuo Cong, Anyu Wang

**Abstract**: Jailbreak attacks induce Large Language Models (LLMs) to generate harmful responses, posing severe misuse threats. Though research on jailbreak attacks and defenses is emerging, there is no consensus on evaluating jailbreaks, i.e., the methods to assess the harmfulness of an LLM's response are varied. Each approach has its own set of strengths and weaknesses, impacting their alignment with human values, as well as the time and financial cost. This diversity challenges researchers in choosing suitable evaluation methods and comparing different attacks and defenses. In this paper, we conduct a comprehensive analysis of jailbreak evaluation methodologies, drawing from nearly 90 jailbreak research published between May 2023 and April 2024. Our study introduces a systematic taxonomy of jailbreak evaluators, offering indepth insights into their strengths and weaknesses, along with the current status of their adaptation. To aid further research, we propose JailbreakEval, a toolkit for evaluating jailbreak attempts. JailbreakEval includes various evaluators out-of-the-box, enabling users to obtain results with a single command or customized evaluation workflows. In summary, we regard JailbreakEval to be a catalyst that simplifies the evaluation process in jailbreak research and fosters an inclusive standard for jailbreak evaluation within the community.

摘要: 越狱攻击导致大型语言模型(LLM)产生有害的响应，构成严重的滥用威胁。尽管对越狱攻击和防御的研究正在兴起，但对于评估越狱还没有达成共识，即评估LLM反应的危害性的方法多种多样。每种方法都有自己的长处和短处，影响它们与人类价值观的一致性，以及时间和财务成本。这种多样性向研究人员提出了挑战，即选择合适的评估方法并比较不同的攻击和防御。本文从2023年5月至2024年4月发表的近90篇越狱研究中，对越狱评估方法进行了全面的分析。我们的研究介绍了越狱评估员的系统分类，深入了解了他们的优势和劣势，以及他们适应的现状。为了帮助进一步的研究，我们提出了JailBreak Eval，一个用于评估越狱企图的工具包。JailBreak Eval包括各种开箱即用的评估器，使用户能够通过单个命令或定制的评估工作流获得结果。总而言之，我们认为越狱评估是一种催化剂，可以简化越狱研究的评估过程，并在社区内培养一个包容性的越狱评估标准。



## **3. Is poisoning a real threat to LLM alignment? Maybe more so than you think**

中毒是对LLM联盟的真正威胁吗？也许比你想象的还要多 cs.LG

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2406.12091v3) [paper-pdf](http://arxiv.org/pdf/2406.12091v3)

**Authors**: Pankayaraj Pathmanathan, Souradip Chakraborty, Xiangyu Liu, Yongyuan Liang, Furong Huang

**Abstract**: Recent advancements in Reinforcement Learning with Human Feedback (RLHF) have significantly impacted the alignment of Large Language Models (LLMs). The sensitivity of reinforcement learning algorithms such as Proximal Policy Optimization (PPO) has led to new line work on Direct Policy Optimization (DPO), which treats RLHF in a supervised learning framework. The increased practical use of these RLHF methods warrants an analysis of their vulnerabilities. In this work, we investigate the vulnerabilities of DPO to poisoning attacks under different scenarios and compare the effectiveness of preference poisoning, a first of its kind. We comprehensively analyze DPO's vulnerabilities under different types of attacks, i.e., backdoor and non-backdoor attacks, and different poisoning methods across a wide array of language models, i.e., LLama 7B, Mistral 7B, and Gemma 7B. We find that unlike PPO-based methods, which, when it comes to backdoor attacks, require at least 4\% of the data to be poisoned to elicit harmful behavior, we exploit the true vulnerabilities of DPO more simply so we can poison the model with only as much as 0.5\% of the data. We further investigate the potential reasons behind the vulnerability and how well this vulnerability translates into backdoor vs non-backdoor attacks.

摘要: 人类反馈强化学习(RLHF)的最新进展对大型语言模型(LLM)的匹配产生了重大影响。强化学习算法的敏感性，如最近策略优化(PPO)，导致了直接策略优化(DPO)的新工作，它在监督学习框架中处理RLHF。这些RLHF方法的实际使用越来越多，因此有理由对其脆弱性进行分析。在这项工作中，我们调查了DPO在不同场景下对中毒攻击的脆弱性，并比较了偏好中毒的有效性，这是第一次。我们全面分析了DPO在不同类型的攻击下的漏洞，即后门攻击和非后门攻击，以及不同的中毒方法，跨越了广泛的语言模型，即：大羊驼7B、米斯特拉尔7B和杰玛7B。我们发现，与基于PPO的方法不同，当涉及到后门攻击时，需要至少4%的数据被毒化才能引发有害行为，而我们更简单地利用DPO的真正漏洞，因此我们只需使用多达0.5%的数据就可以毒害模型。我们进一步调查了该漏洞背后的潜在原因，以及该漏洞在多大程度上转化为后门攻击与非后门攻击。



## **4. STAIR: Improving Safety Alignment with Introspective Reasoning**

楼梯：通过内省推理改善安全性 cs.CL

22 pages, 8 figures

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02384v1) [paper-pdf](http://arxiv.org/pdf/2502.02384v1)

**Authors**: Yichi Zhang, Siyuan Zhang, Yao Huang, Zeyu Xia, Zhengwei Fang, Xiao Yang, Ranjie Duan, Dong Yan, Yinpeng Dong, Jun Zhu

**Abstract**: Ensuring the safety and harmlessness of Large Language Models (LLMs) has become equally critical as their performance in applications. However, existing safety alignment methods typically suffer from safety-performance trade-offs and the susceptibility to jailbreak attacks, primarily due to their reliance on direct refusals for malicious queries. In this paper, we propose STAIR, a novel framework that integrates SafeTy Alignment with Itrospective Reasoning. We enable LLMs to identify safety risks through step-by-step analysis by self-improving chain-of-thought (CoT) reasoning with safety awareness. STAIR first equips the model with a structured reasoning capability and then advances safety alignment via iterative preference optimization on step-level reasoning data generated using our newly proposed Safety-Informed Monte Carlo Tree Search (SI-MCTS). We further train a process reward model on this data to guide test-time searches for improved responses. Extensive experiments show that STAIR effectively mitigates harmful outputs while better preserving helpfulness, compared to instinctive alignment strategies. With test-time scaling, STAIR achieves a safety performance comparable to Claude-3.5 against popular jailbreak attacks. Relevant resources in this work are available at https://github.com/thu-ml/STAIR.

摘要: 确保大型语言模型(LLM)的安全性和无害性已变得与它们在应用程序中的性能同等重要。然而，现有的安全对齐方法通常会受到安全性能和越狱攻击之间的权衡，这主要是因为它们依赖于对恶意查询的直接拒绝。在本文中，我们提出了一种新的框架STAIR，它将安全匹配和回顾推理结合在一起。我们通过具有安全意识的自我完善的思想链(COT)推理，使LLM能够通过逐步分析来识别安全风险。STAIR首先为模型配备了结构化推理能力，然后通过迭代偏好优化对使用新提出的安全通知蒙特卡罗树搜索(SI-MCTS)生成的步进级推理数据进行安全匹配。我们进一步训练了一个基于这些数据的过程奖励模型，以指导测试时间搜索以获得更好的响应。广泛的实验表明，与本能的对齐策略相比，STAIR有效地减少了有害输出，同时更好地保留了帮助。随着测试时间的扩展，STAIR在抵御流行的越狱攻击时实现了与克劳德-3.5相当的安全性能。这项工作的相关资源可在https://github.com/thu-ml/STAIR.上获得



## **5. SHIELD: APT Detection and Intelligent Explanation Using LLM**

SHIELD：使用LLM的APT检测和智能解释 cs.CR

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02342v1) [paper-pdf](http://arxiv.org/pdf/2502.02342v1)

**Authors**: Parth Atulbhai Gandhi, Prasanna N. Wudali, Yonatan Amaru, Yuval Elovici, Asaf Shabtai

**Abstract**: Advanced persistent threats (APTs) are sophisticated cyber attacks that can remain undetected for extended periods, making their mitigation particularly challenging. Given their persistence, significant effort is required to detect them and respond effectively. Existing provenance-based attack detection methods often lack interpretability and suffer from high false positive rates, while investigation approaches are either supervised or limited to known attacks. To address these challenges, we introduce SHIELD, a novel approach that combines statistical anomaly detection and graph-based analysis with the contextual analysis capabilities of large language models (LLMs). SHIELD leverages the implicit knowledge of LLMs to uncover hidden attack patterns in provenance data, while reducing false positives and providing clear, interpretable attack descriptions. This reduces analysts' alert fatigue and makes it easier for them to understand the threat landscape. Our extensive evaluation demonstrates SHIELD's effectiveness and computational efficiency in real-world scenarios. SHIELD was shown to outperform state-of-the-art methods, achieving higher precision and recall. SHIELD's integration of anomaly detection, LLM-driven contextual analysis, and advanced graph-based correlation establishes a new benchmark for APT detection.

摘要: 高级持续性威胁(APT)是一种复杂的网络攻击，可以在很长一段时间内保持不被检测到，这使得缓解这些攻击特别具有挑战性。鉴于它们的持久性，需要付出巨大努力才能发现它们并有效应对。现有的基于来源的攻击检测方法往往缺乏可解释性，并且存在较高的误警率，而调查方法要么受到监督，要么仅限于已知的攻击。为了应对这些挑战，我们引入了Shield，这是一种将统计异常检测和基于图的分析与大型语言模型(LLM)的上下文分析能力相结合的新方法。Shield利用LLMS的隐含知识来发现来源数据中隐藏的攻击模式，同时减少误报并提供清晰、可解释的攻击描述。这减少了分析师的警觉疲劳，使他们更容易了解威胁情况。我们广泛的评估证明了Shield在现实世界场景中的有效性和计算效率。Shield被证明比最先进的方法性能更好，实现了更高的精确度和召回率。Shield集成了异常检测、LLM驱动的上下文分析和先进的基于图形的关联，为APT检测建立了一个新的基准。



## **6. BadRobot: Jailbreaking Embodied LLMs in the Physical World**

BadRobot：物理世界中越狱的法学硕士 cs.CY

Accepted to ICLR 2025. Project page:  https://Embodied-LLMs-Safety.github.io

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2407.20242v4) [paper-pdf](http://arxiv.org/pdf/2407.20242v4)

**Authors**: Hangtao Zhang, Chenyu Zhu, Xianlong Wang, Ziqi Zhou, Changgan Yin, Minghui Li, Lulu Xue, Yichen Wang, Shengshan Hu, Aishan Liu, Peijin Guo, Leo Yu Zhang

**Abstract**: Embodied AI represents systems where AI is integrated into physical entities. Large Language Model (LLM), which exhibits powerful language understanding abilities, has been extensively employed in embodied AI by facilitating sophisticated task planning. However, a critical safety issue remains overlooked: could these embodied LLMs perpetrate harmful behaviors? In response, we introduce BadRobot, a novel attack paradigm aiming to make embodied LLMs violate safety and ethical constraints through typical voice-based user-system interactions. Specifically, three vulnerabilities are exploited to achieve this type of attack: (i) manipulation of LLMs within robotic systems, (ii) misalignment between linguistic outputs and physical actions, and (iii) unintentional hazardous behaviors caused by world knowledge's flaws. Furthermore, we construct a benchmark of various malicious physical action queries to evaluate BadRobot's attack performance. Based on this benchmark, extensive experiments against existing prominent embodied LLM frameworks (e.g., Voxposer, Code as Policies, and ProgPrompt) demonstrate the effectiveness of our BadRobot.

摘要: 体现的人工智能代表了人工智能集成到物理实体中的系统。大语言模型(LLM)具有强大的语言理解能力，通过促进复杂的任务规划，已被广泛应用于嵌入式人工智能中。然而，一个关键的安全问题仍然被忽视：这些具体化的LLM是否会实施有害行为？作为回应，我们引入了BadRobot，这是一种新的攻击范例，旨在通过典型的基于语音的用户-系统交互来使具体化LLM违反安全和伦理约束。具体地说，利用三个漏洞来实现这种类型的攻击：(I)在机器人系统内操纵LLM，(Ii)语言输出和物理动作之间的不匹配，以及(Iii)由世界知识的缺陷造成的无意危险行为。此外，我们还构建了一个针对各种恶意物理动作查询的基准来评估BadRobot的攻击性能。在此基准测试的基础上，对现有的主流嵌入式LLM框架(如Voxposer、代码即策略和ProgPrompt)进行了广泛的实验，证明了BadRobot的有效性。



## **7. Adversarial Reasoning at Jailbreaking Time**

越狱时的对抗推理 cs.LG

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01633v1) [paper-pdf](http://arxiv.org/pdf/2502.01633v1)

**Authors**: Mahdi Sabbaghi, Paul Kassianik, George Pappas, Yaron Singer, Amin Karbasi, Hamed Hassani

**Abstract**: As large language models (LLMs) are becoming more capable and widespread, the study of their failure cases is becoming increasingly important. Recent advances in standardizing, measuring, and scaling test-time compute suggest new methodologies for optimizing models to achieve high performance on hard tasks. In this paper, we apply these advances to the task of model jailbreaking: eliciting harmful responses from aligned LLMs. We develop an adversarial reasoning approach to automatic jailbreaking via test-time computation that achieves SOTA attack success rates (ASR) against many aligned LLMs, even the ones that aim to trade inference-time compute for adversarial robustness. Our approach introduces a new paradigm in understanding LLM vulnerabilities, laying the foundation for the development of more robust and trustworthy AI systems.

摘要: 随着大型语言模型（LLM）变得越来越强大和广泛，对其失败案例的研究变得越来越重要。标准化、测量和扩展测试时计算方面的最新进展为优化模型以在硬任务中实现高性能提出了新的方法。在本文中，我们将这些进展应用于模型越狱的任务：从对齐的LLM中引发有害反应。我们开发了一种通过测试时计算自动越狱的对抗推理方法，该方法针对许多对齐的LLM，即使是那些旨在以推理时计算为对抗鲁棒性的LLM，也可以实现SOTA攻击成功率（ASB）。我们的方法引入了理解LLM漏洞的新范式，为开发更强大、更值得信赖的人工智能系统奠定了基础。



## **8. Robust-LLaVA: On the Effectiveness of Large-Scale Robust Image Encoders for Multi-modal Large Language Models**

Robust-LLaVA：关于大规模鲁棒图像编码器对多模式大型语言模型的有效性 cs.CV

Under Review

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01576v1) [paper-pdf](http://arxiv.org/pdf/2502.01576v1)

**Authors**: Hashmat Shadab Malik, Fahad Shamshad, Muzammal Naseer, Karthik Nandakumar, Fahad Khan, Salman Khan

**Abstract**: Multi-modal Large Language Models (MLLMs) excel in vision-language tasks but remain vulnerable to visual adversarial perturbations that can induce hallucinations, manipulate responses, or bypass safety mechanisms. Existing methods seek to mitigate these risks by applying constrained adversarial fine-tuning to CLIP vision encoders on ImageNet-scale data, ensuring their generalization ability is preserved. However, this limited adversarial training restricts robustness and broader generalization. In this work, we explore an alternative approach of leveraging existing vision classification models that have been adversarially pre-trained on large-scale data. Our analysis reveals two principal contributions: (1) the extensive scale and diversity of adversarial pre-training enables these models to demonstrate superior robustness against diverse adversarial threats, ranging from imperceptible perturbations to advanced jailbreaking attempts, without requiring additional adversarial training, and (2) end-to-end MLLM integration with these robust models facilitates enhanced adaptation of language components to robust visual features, outperforming existing plug-and-play methodologies on complex reasoning tasks. Through systematic evaluation across visual question-answering, image captioning, and jail-break attacks, we demonstrate that MLLMs trained with these robust models achieve superior adversarial robustness while maintaining favorable clean performance. Our framework achieves 2x and 1.5x average robustness gains in captioning and VQA tasks, respectively, and delivers over 10% improvement against jailbreak attacks. Code and pretrained models will be available at https://github.com/HashmatShadab/Robust-LLaVA.

摘要: 多模式大语言模型(MLLMS)在视觉-语言任务中表现出色，但仍然容易受到视觉对抗性扰动的影响，这些扰动可能会导致幻觉、操纵反应或绕过安全机制。现有的方法试图通过对ImageNet尺度数据上的裁剪视觉编码器应用受限的对抗性微调来缓解这些风险，以确保它们的泛化能力得到保护。然而，这种有限的对抗性训练限制了健壮性和更广泛的泛化。在这项工作中，我们探索了一种替代方法，利用现有的视觉分类模型，这些模型已经在大规模数据上进行了相反的预训练。我们的分析揭示了两个主要贡献：(1)对抗性预训练的广泛规模和多样性使这些模型能够在不需要额外的对抗性训练的情况下，对从不可察觉的扰动到高级越狱尝试等不同的对抗性威胁表现出优越的健壮性；(2)端到端MLLM与这些健壮的模型的集成促进了语言成分对健壮视觉特征的增强适应，在复杂推理任务中的表现优于现有的即插即用方法。通过对视觉问答、图像字幕和越狱攻击的系统评估，我们证明了使用这些健壮模型训练的MLLMS在保持良好的干净性能的同时，获得了优越的对手健壮性。我们的框架在字幕和VQA任务中分别获得了2倍和1.5倍的平均健壮性提升，并在抵御越狱攻击方面提供了超过10%的改进。代码和预先培训的模型将在https://github.com/HashmatShadab/Robust-LLaVA.上提供



## **9. Scaling Up Membership Inference: When and How Attacks Succeed on Large Language Models**

扩大成员资格推理：攻击何时以及如何在大型语言模型上取得成功 cs.CL

Findings of NAACL 2025. Our code is available at  https://github.com/parameterlab/mia-scaling

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2411.00154v2) [paper-pdf](http://arxiv.org/pdf/2411.00154v2)

**Authors**: Haritz Puerto, Martin Gubri, Sangdoo Yun, Seong Joon Oh

**Abstract**: Membership inference attacks (MIA) attempt to verify the membership of a given data sample in the training set for a model. MIA has become relevant in recent years, following the rapid development of large language models (LLM). Many are concerned about the usage of copyrighted materials for training them and call for methods for detecting such usage. However, recent research has largely concluded that current MIA methods do not work on LLMs. Even when they seem to work, it is usually because of the ill-designed experimental setup where other shortcut features enable "cheating." In this work, we argue that MIA still works on LLMs, but only when multiple documents are presented for testing. We construct new benchmarks that measure the MIA performances at a continuous scale of data samples, from sentences (n-grams) to a collection of documents (multiple chunks of tokens). To validate the efficacy of current MIA approaches at greater scales, we adapt a recent work on Dataset Inference (DI) for the task of binary membership detection that aggregates paragraph-level MIA features to enable MIA at document and collection of documents level. This baseline achieves the first successful MIA on pre-trained and fine-tuned LLMs.

摘要: 成员关系推理攻击(MIA)试图验证给定数据样本在模型训练集中的成员资格。近年来，随着大型语言模型(LLM)的快速发展，MIA变得相关起来。许多人担心使用受版权保护的材料来培训他们，并呼吁采取方法来检测这种使用情况。然而，最近的研究在很大程度上得出结论，目前的MIA方法不适用于LLMS。即使它们看起来很有效，这通常也是因为设计糟糕的实验设置，其他快捷功能允许“作弊”。在这项工作中，我们认为MIA仍然适用于LLMS，但只有在提交多个文档进行测试时才能使用。我们构建了新的基准来衡量MIA在连续规模的数据样本上的性能，从句子(n-gram)到文档集合(多个令牌块)。为了在更大范围内验证当前MIA方法的有效性，我们对最近在数据集推理(DI)方面的工作进行了调整，以用于二元成员关系检测任务，该任务聚集了段级MIA特征，以支持文档和文档集合级别的MIA。这一基线在预先训练和微调的LLM上实现了第一次成功的MIA。



## **10. Topic-FlipRAG: Topic-Orientated Adversarial Opinion Manipulation Attacks to Retrieval-Augmented Generation Models**

Top-FlipRAG：针对检索增强生成模型的面向主题的对抗性观点操纵攻击 cs.CL

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01386v1) [paper-pdf](http://arxiv.org/pdf/2502.01386v1)

**Authors**: Yuyang Gong, Zhuo Chen, Miaokun Chen, Fengchang Yu, Wei Lu, Xiaofeng Wang, Xiaozhong Liu, Jiawei Liu

**Abstract**: Retrieval-Augmented Generation (RAG) systems based on Large Language Models (LLMs) have become essential for tasks such as question answering and content generation. However, their increasing impact on public opinion and information dissemination has made them a critical focus for security research due to inherent vulnerabilities. Previous studies have predominantly addressed attacks targeting factual or single-query manipulations. In this paper, we address a more practical scenario: topic-oriented adversarial opinion manipulation attacks on RAG models, where LLMs are required to reason and synthesize multiple perspectives, rendering them particularly susceptible to systematic knowledge poisoning. Specifically, we propose Topic-FlipRAG, a two-stage manipulation attack pipeline that strategically crafts adversarial perturbations to influence opinions across related queries. This approach combines traditional adversarial ranking attack techniques and leverages the extensive internal relevant knowledge and reasoning capabilities of LLMs to execute semantic-level perturbations. Experiments show that the proposed attacks effectively shift the opinion of the model's outputs on specific topics, significantly impacting user information perception. Current mitigation methods cannot effectively defend against such attacks, highlighting the necessity for enhanced safeguards for RAG systems, and offering crucial insights for LLM security research.

摘要: 基于大型语言模型(LLMS)的检索-增强生成(RAG)系统已经成为诸如问题回答和内容生成等任务的关键。然而，由于其固有的脆弱性，它们对舆论和信息传播的影响越来越大，使其成为安全研究的关键焦点。以前的研究主要针对针对事实或单一查询操作的攻击。在本文中，我们讨论了一个更实际的场景：针对RAG模型的面向主题的对抗性意见操纵攻击，其中要求LLM推理和综合多个视角，使它们特别容易受到系统性知识中毒的影响。具体地说，我们提出了Theme-FlipRAG，这是一种两阶段操纵攻击管道，战略性地制造对抗性扰动来影响相关查询的观点。该方法结合了传统的对抗性排序攻击技术，并利用LLMS丰富的内部相关知识和推理能力来执行语义级的扰动。实验表明，提出的攻击有效地改变了模型输出对特定主题的看法，显著影响了用户对信息的感知。目前的缓解方法无法有效防御此类攻击，这突显了加强RAG系统安全保障的必要性，并为LLM安全研究提供了至关重要的见解。



## **11. Detecting Backdoor Samples in Contrastive Language Image Pretraining**

对比语言图像预训练中后门样本检测 cs.LG

ICLR2025

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01385v1) [paper-pdf](http://arxiv.org/pdf/2502.01385v1)

**Authors**: Hanxun Huang, Sarah Erfani, Yige Li, Xingjun Ma, James Bailey

**Abstract**: Contrastive language-image pretraining (CLIP) has been found to be vulnerable to poisoning backdoor attacks where the adversary can achieve an almost perfect attack success rate on CLIP models by poisoning only 0.01\% of the training dataset. This raises security concerns on the current practice of pretraining large-scale models on unscrutinized web data using CLIP. In this work, we analyze the representations of backdoor-poisoned samples learned by CLIP models and find that they exhibit unique characteristics in their local subspace, i.e., their local neighborhoods are far more sparse than that of clean samples. Based on this finding, we conduct a systematic study on detecting CLIP backdoor attacks and show that these attacks can be easily and efficiently detected by traditional density ratio-based local outlier detectors, whereas existing backdoor sample detection methods fail. Our experiments also reveal that an unintentional backdoor already exists in the original CC3M dataset and has been trained into a popular open-source model released by OpenCLIP. Based on our detector, one can clean up a million-scale web dataset (e.g., CC3M) efficiently within 15 minutes using 4 Nvidia A100 GPUs. The code is publicly available in our \href{https://github.com/HanxunH/Detect-CLIP-Backdoor-Samples}{GitHub repository}.

摘要: 对比语言图像预训练(CLIP)被发现容易受到中毒后门攻击，对手只需中毒0.01%的训练数据集就可以在CLIP模型上获得近乎完美的攻击成功率。这引发了人们对当前使用CLIP对大规模模型进行未经审查的网络数据的预培训的安全担忧。在这项工作中，我们分析了通过CLIP模型学习的后门中毒样本的表示，发现它们在局部子空间中表现出独特的特征，即它们的局部邻域比干净样本的局部邻域稀疏得多。基于这一发现，我们对CLIP后门攻击的检测进行了系统的研究，结果表明，传统的基于密度比的局部离群点检测方法可以轻松有效地检测到这些攻击，而现有的后门样本检测方法却无法检测到这些攻击。我们的实验还表明，在原始的CC3M数据集中已经存在一个无意的后门，并且已经被训练成OpenCLIP发布的流行的开源模型。基于我们的检测器，您可以使用4个NVIDIA A100图形处理器在15分钟内高效清理百万级Web数据集(例如CC3M)。代码在我们的\href{https://github.com/HanxunH/Detect-CLIP-Backdoor-Samples}{GitHub存储库中公开提供。



## **12. Improving the Robustness of Representation Misdirection for Large Language Model Unlearning**

提高大型语言模型去学习的表示误导的鲁棒性 cs.CL

12 pages, 4 figures, 1 table

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2501.19202v2) [paper-pdf](http://arxiv.org/pdf/2501.19202v2)

**Authors**: Dang Huu-Tien, Hoang Thanh-Tung, Le-Minh Nguyen, Naoya Inoue

**Abstract**: Representation Misdirection (RM) and variants are established large language model (LLM) unlearning methods with state-of-the-art performance. In this paper, we show that RM methods inherently reduce models' robustness, causing them to misbehave even when a single non-adversarial forget-token is in the retain-query. Toward understanding underlying causes, we reframe the unlearning process as backdoor attacks and defenses: forget-tokens act as backdoor triggers that, when activated in retain-queries, cause disruptions in RM models' behaviors, similar to successful backdoor attacks. To mitigate this vulnerability, we propose Random Noise Augmentation -- a model and method agnostic approach with theoretical guarantees for improving the robustness of RM methods. Extensive experiments demonstrate that RNA significantly improves the robustness of RM models while enhancing the unlearning performances.

摘要: 表示误导（RM）和变体是建立的大型语言模型（LLM）去学习方法，具有最先进的性能。在本文中，我们表明RM方法本质上降低了模型的鲁棒性，导致它们即使在保留查询中存在单个非对抗性遗忘令牌时也会表现不当。为了了解根本原因，我们将取消学习过程重新定义为后门攻击和防御：忘记令牌充当后门触发器，当在保留查询中激活时，会导致RM模型行为中断，类似于成功的后门攻击。为了减轻这一漏洞，我们提出了随机噪音增强--一种模型和方法不可知的方法，具有提高RM方法鲁棒性的理论保证。大量实验表明，RNA显着提高了RM模型的鲁棒性，同时增强了去学习性能。



## **13. PBI-Attack: Prior-Guided Bimodal Interactive Black-Box Jailbreak Attack for Toxicity Maximization**

PBI攻击：优先引导双峰交互黑匣子越狱攻击，以实现毒性最大化 cs.CR

Prior-Guided Bimodal Interactive Black-Box Jailbreak Attack for  Toxicity Maximization

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2412.05892v3) [paper-pdf](http://arxiv.org/pdf/2412.05892v3)

**Authors**: Ruoxi Cheng, Yizhong Ding, Shuirong Cao, Ranjie Duan, Xiaoshuang Jia, Shaowei Yuan, Zhiqiang Wang, Xiaojun Jia

**Abstract**: Understanding the vulnerabilities of Large Vision Language Models (LVLMs) to jailbreak attacks is essential for their responsible real-world deployment. Most previous work requires access to model gradients, or is based on human knowledge (prompt engineering) to complete jailbreak, and they hardly consider the interaction of images and text, resulting in inability to jailbreak in black box scenarios or poor performance. To overcome these limitations, we propose a Prior-Guided Bimodal Interactive Black-Box Jailbreak Attack for toxicity maximization, referred to as PBI-Attack. Our method begins by extracting malicious features from a harmful corpus using an alternative LVLM and embedding these features into a benign image as prior information. Subsequently, we enhance these features through bidirectional cross-modal interaction optimization, which iteratively optimizes the bimodal perturbations in an alternating manner through greedy search, aiming to maximize the toxicity of the generated response. The toxicity level is quantified using a well-trained evaluation model. Experiments demonstrate that PBI-Attack outperforms previous state-of-the-art jailbreak methods, achieving an average attack success rate of 92.5% across three open-source LVLMs and around 67.3% on three closed-source LVLMs. Disclaimer: This paper contains potentially disturbing and offensive content.

摘要: 了解大型视觉语言模型(LVLM)对越狱攻击的脆弱性对于它们在现实世界中负责任的部署至关重要。以前的工作大多需要获取模型梯度，或者基于人类知识(提示工程)来完成越狱，并且很少考虑图像和文本的交互，导致在黑匣子场景下无法越狱或性能不佳。为了克服这些局限性，我们提出了一种先验引导的双模交互黑盒越狱攻击，称为PBI攻击。我们的方法首先使用替代的LVLM从有害语料库中提取恶意特征，并将这些特征作为先验信息嵌入到良性图像中。随后，我们通过双向跨模式交互优化来增强这些特征，该优化通过贪婪搜索以交替的方式迭代优化双峰扰动，以最大化所生成响应的毒性。使用训练有素的评估模型来量化毒性水平。实验表明，PBI-Attack的性能优于以往最先进的越狱方法，在三个开源LVLM上的平均攻击成功率为92.5%，在三个闭源LVLM上的平均攻击成功率约为67.3%。免责声明：本文包含可能令人不安和冒犯性的内容。



## **14. Eliciting Language Model Behaviors with Investigator Agents**

使用研究者代理激发语言模型行为 cs.LG

20 pages, 7 figures

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01236v1) [paper-pdf](http://arxiv.org/pdf/2502.01236v1)

**Authors**: Xiang Lisa Li, Neil Chowdhury, Daniel D. Johnson, Tatsunori Hashimoto, Percy Liang, Sarah Schwettmann, Jacob Steinhardt

**Abstract**: Language models exhibit complex, diverse behaviors when prompted with free-form text, making it difficult to characterize the space of possible outputs. We study the problem of behavior elicitation, where the goal is to search for prompts that induce specific target behaviors (e.g., hallucinations or harmful responses) from a target language model. To navigate the exponentially large space of possible prompts, we train investigator models to map randomly-chosen target behaviors to a diverse distribution of outputs that elicit them, similar to amortized Bayesian inference. We do this through supervised fine-tuning, reinforcement learning via DPO, and a novel Frank-Wolfe training objective to iteratively discover diverse prompting strategies. Our investigator models surface a variety of effective and human-interpretable prompts leading to jailbreaks, hallucinations, and open-ended aberrant behaviors, obtaining a 100% attack success rate on a subset of AdvBench (Harmful Behaviors) and an 85% hallucination rate.

摘要: 当提示自由格式文本时，语言模型表现出复杂多样的行为，这使得很难描述可能输出的空间。我们研究行为诱导问题，目标是从目标语言模型中寻找诱导特定目标行为(例如，幻觉或有害反应)的提示。为了在可能提示的指数级大空间中导航，我们训练调查员模型将随机选择的目标行为映射到引发它们的不同输出分布，类似于摊销贝叶斯推理。我们通过有监督的微调、通过DPO的强化学习和一个新颖的Frank-Wolfe训练目标来迭代地发现不同的激励策略来做到这一点。我们的调查员模型提供了各种有效的、人类可解释的提示，导致越狱、幻觉和无限制的异常行为，对AdvBtch(有害行为)的子集获得了100%的攻击成功率和85%的幻想率。



## **15. The dark deep side of DeepSeek: Fine-tuning attacks against the safety alignment of CoT-enabled models**

DeepSeek的阴暗面：针对支持CoT的模型的安全一致性的微调攻击 cs.CR

12 Pages

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01225v1) [paper-pdf](http://arxiv.org/pdf/2502.01225v1)

**Authors**: Zhiyuan Xu, Joseph Gardiner, Sana Belguith

**Abstract**: Large language models are typically trained on vast amounts of data during the pre-training phase, which may include some potentially harmful information. Fine-tuning attacks can exploit this by prompting the model to reveal such behaviours, leading to the generation of harmful content. In this paper, we focus on investigating the performance of the Chain of Thought based reasoning model, DeepSeek, when subjected to fine-tuning attacks. Specifically, we explore how fine-tuning manipulates the model's output, exacerbating the harmfulness of its responses while examining the interaction between the Chain of Thought reasoning and adversarial inputs. Through this study, we aim to shed light on the vulnerability of Chain of Thought enabled models to fine-tuning attacks and the implications for their safety and ethical deployment.

摘要: 大型语言模型通常在预训练阶段根据大量数据进行训练，其中可能包括一些潜在有害的信息。微调攻击可以通过促使模型揭示此类行为来利用这一点，从而导致有害内容的生成。在本文中，我们重点研究基于思想链的推理模型DeepSeek在受到微调攻击时的性能。具体来说，我们探索微调如何操纵模型的输出，加剧其反应的危害性，同时检查思维链推理和对抗输入之间的相互作用。通过这项研究，我们的目标是揭示思想链使模型能够微调攻击的脆弱性及其对安全性和道德部署的影响。



## **16. Jailbreaking with Universal Multi-Prompts**

用通用多胞胎越狱 cs.CL

Accepted by NAACL Findings 2025

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01154v1) [paper-pdf](http://arxiv.org/pdf/2502.01154v1)

**Authors**: Yu-Ling Hsu, Hsuan Su, Shang-Tse Chen

**Abstract**: Large language models (LLMs) have seen rapid development in recent years, revolutionizing various applications and significantly enhancing convenience and productivity. However, alongside their impressive capabilities, ethical concerns and new types of attacks, such as jailbreaking, have emerged. While most prompting techniques focus on optimizing adversarial inputs for individual cases, resulting in higher computational costs when dealing with large datasets. Less research has addressed the more general setting of training a universal attacker that can transfer to unseen tasks. In this paper, we introduce JUMP, a prompt-based method designed to jailbreak LLMs using universal multi-prompts. We also adapt our approach for defense, which we term DUMP. Experimental results demonstrate that our method for optimizing universal multi-prompts outperforms existing techniques.

摘要: 近年来，大型语言模型（LLM）发展迅速，彻底改变了各种应用程序，显着提高了便利性和生产力。然而，除了它们令人印象深刻的能力之外，道德问题和越狱等新型攻击也出现了。虽然大多数提示技术专注于优化个别案例的对抗输入，从而导致处理大型数据集时计算成本更高。较少的研究涉及训练可以转移到不可见任务的通用攻击者的更一般设置。本文中，我们介绍了JUMP，这是一种基于预算的方法，旨在使用通用多提示越狱LLM。我们还调整我们的防御方法，我们称之为“DUMP”。实验结果表明，我们用于优化通用多提示的方法优于现有技术。



## **17. When LLMs Go Online: The Emerging Threat of Web-Enabled LLMs**

当LLM上线时：支持Web的LLM的新兴威胁 cs.CR

20 pages, To appear in Usenix Security 2025

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2410.14569v3) [paper-pdf](http://arxiv.org/pdf/2410.14569v3)

**Authors**: Hanna Kim, Minkyoo Song, Seung Ho Na, Seungwon Shin, Kimin Lee

**Abstract**: Recent advancements in Large Language Models (LLMs) have established them as agentic systems capable of planning and interacting with various tools. These LLM agents are often paired with web-based tools, enabling access to diverse sources and real-time information. Although these advancements offer significant benefits across various applications, they also increase the risk of malicious use, particularly in cyberattacks involving personal information. In this work, we investigate the risks associated with misuse of LLM agents in cyberattacks involving personal data. Specifically, we aim to understand: 1) how potent LLM agents can be when directed to conduct cyberattacks, 2) how cyberattacks are enhanced by web-based tools, and 3) how affordable and easy it becomes to launch cyberattacks using LLM agents. We examine three attack scenarios: the collection of Personally Identifiable Information (PII), the generation of impersonation posts, and the creation of spear-phishing emails. Our experiments reveal the effectiveness of LLM agents in these attacks: LLM agents achieved a precision of up to 95.9% in collecting PII, generated impersonation posts where 93.9% of them were deemed authentic, and boosted click rate of phishing links in spear phishing emails by 46.67%. Additionally, our findings underscore the limitations of existing safeguards in contemporary commercial LLMs, emphasizing the urgent need for robust security measures to prevent the misuse of LLM agents.

摘要: 大型语言模型(LLM)的最新进展已将它们确立为能够规划各种工具并与其交互的代理系统。这些LLM代理通常与基于Web的工具配合使用，从而能够访问不同的来源和实时信息。虽然这些改进在各种应用程序中提供了显著的好处，但它们也增加了恶意使用的风险，特别是在涉及个人信息的网络攻击中。在这项工作中，我们调查了在涉及个人数据的网络攻击中滥用LLM代理的相关风险。具体地说，我们的目标是了解：1)LLM代理在被指示进行网络攻击时的威力有多大；2)基于Web的工具如何增强网络攻击；3)使用LLM代理发起网络攻击变得多么负担得起和容易。我们研究了三种攻击场景：收集个人身份信息(PII)、生成模拟帖子和创建鱼叉式网络钓鱼电子邮件。我们的实验显示了LLM代理在这些攻击中的有效性：LLM代理在收集PII信息时达到了95.9%的准确率，生成了93.9%被认为是真实的模仿帖子，并将鱼叉式钓鱼邮件中钓鱼链接的点击率提高了46.67%。此外，我们的研究结果强调了当代商业LLM现有保障措施的局限性，强调迫切需要强有力的安全措施，以防止滥用LLM剂。



## **18. Tool Unlearning for Tool-Augmented LLMs**

工具增强LLM的工具取消学习 cs.LG

https://clu-uml.github.io/MU-Bench-Project-Page/

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01083v1) [paper-pdf](http://arxiv.org/pdf/2502.01083v1)

**Authors**: Jiali Cheng, Hadi Amiri

**Abstract**: Tool-augmented large language models (LLMs) are often trained on datasets of query-response pairs, which embed the ability to use tools or APIs directly into the parametric knowledge of LLMs. Tool-augmented LLMs need the ability to forget learned tools due to security vulnerabilities, privacy regulations, or tool deprecations. However, ``tool unlearning'' has not been investigated in unlearning literature. We introduce this novel task, which requires addressing distinct challenges compared to traditional unlearning: knowledge removal rather than forgetting individual samples, the high cost of optimizing LLMs, and the need for principled evaluation metrics. To bridge these gaps, we propose ToolDelete, the first approach for unlearning tools from tool-augmented LLMs. It implements three key properties to address the above challenges for effective tool unlearning and introduces a new membership inference attack (MIA) model for effective evaluation. Extensive experiments on multiple tool learning datasets and tool-augmented LLMs show that ToolDelete effectively unlearns randomly selected tools, while preserving the LLM's knowledge on non-deleted tools and maintaining performance on general tasks.

摘要: 工具增强的大型语言模型(LLM)通常是在查询-响应对的数据集上进行训练的，这将使用工具或API的能力直接嵌入到LLM的参数知识中。工具增强的LLM需要能够忘记由于安全漏洞、隐私法规或工具弃用而学到的工具。然而，“工具遗忘”在遗忘文献中还没有被研究过。我们引入了这项新的任务，它需要解决与传统遗忘相比的不同挑战：知识移除而不是忘记单个样本，优化LLM的高成本，以及对原则性评估指标的需求。为了弥合这些差距，我们提出了ToolDelete，这是第一种从工具增强的LLM中忘记工具的方法。它实现了三个关键性质来解决上述有效工具遗忘的挑战，并引入了一个新的成员推理攻击(MIA)模型来进行有效评估。在多个工具学习数据集和工具扩充的LLM上的大量实验表明，ToolDelete有效地取消了随机选择的工具的学习，同时保留了LLM关于未删除工具的知识，并保持了一般任务的性能。



## **19. SQL Injection Jailbreak: A Structural Disaster of Large Language Models**

SQL注入越狱：大型语言模型的结构灾难 cs.CR

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2411.01565v4) [paper-pdf](http://arxiv.org/pdf/2411.01565v4)

**Authors**: Jiawei Zhao, Kejiang Chen, Weiming Zhang, Nenghai Yu

**Abstract**: In recent years, the rapid development of large language models (LLMs) has brought new vitality into various domains, generating substantial social and economic benefits. However, this swift advancement has also introduced new vulnerabilities. Jailbreaking, a form of attack that induces LLMs to produce harmful content through carefully crafted prompts, presents a significant challenge to the safe and trustworthy development of LLMs. Previous jailbreak methods primarily exploited the internal properties or capabilities of LLMs, such as optimization-based jailbreak methods and methods that leveraged the model's context-learning abilities. In this paper, we introduce a novel jailbreak method, SQL Injection Jailbreak (SIJ), which targets the external properties of LLMs, specifically, the way LLMs construct input prompts. By injecting jailbreak information into user prompts, SIJ successfully induces the model to output harmful content. Our SIJ method achieves near 100\% attack success rates on five well-known open-source LLMs on the AdvBench and HEx-PHI, while incurring lower time costs compared to previous methods. For closed-source models, SIJ achieves near 100% attack success rate on GPT-3.5-turbo. Additionally, SIJ exposes a new vulnerability in LLMs that urgently requires mitigation. To address this, we propose a simple defense method called Self-Reminder-Key to counter SIJ and demonstrate its effectiveness through experimental results. Our code is available at https://github.com/weiyezhimeng/SQL-Injection-Jailbreak.

摘要: 近年来，大型语言模型的快速发展为各个领域带来了新的活力，产生了可观的社会效益和经济效益。然而，这种快速发展也带来了新的脆弱性。越狱是一种攻击形式，通过精心制作的提示诱使LLM产生有害内容，对LLM的安全和可信开发构成了重大挑战。以前的越狱方法主要利用LLMS的内部属性或功能，例如基于优化的越狱方法和利用模型的上下文学习能力的方法。本文介绍了一种新的越狱方法--SQL注入越狱(SIJ)，它针对LLMS的外部属性，特别是LLMS构造输入提示的方式。通过在用户提示中注入越狱信息，SIJ成功地诱导该模型输出有害内容。与以前的方法相比，我们的SIJ方法在AdvBch和hex-PHI上的五个著名的开源LLM上获得了近100%的攻击成功率，同时产生了更低的时间成本。对于封闭源代码模型，SIJ在GPT-3.5-Turbo上实现了近100%的攻击成功率。此外，SIJ还暴露了LLMS中一个迫切需要缓解的新漏洞。针对这一问题，我们提出了一种简单的防御方法，称为自我提醒密钥来对抗SIJ，并通过实验结果证明了其有效性。我们的代码可以在https://github.com/weiyezhimeng/SQL-Injection-Jailbreak.上找到



## **20. Refining Adaptive Zeroth-Order Optimization at Ease**

轻松细化自适应零阶优化 cs.LG

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01014v1) [paper-pdf](http://arxiv.org/pdf/2502.01014v1)

**Authors**: Yao Shu, Qixin Zhang, Kun He, Zhongxiang Dai

**Abstract**: Recently, zeroth-order (ZO) optimization plays an essential role in scenarios where gradient information is inaccessible or unaffordable, such as black-box systems and resource-constrained environments. While existing adaptive methods such as ZO-AdaMM have shown promise, they are fundamentally limited by their underutilization of moment information during optimization, usually resulting in underperforming convergence. To overcome these limitations, this paper introduces Refined Adaptive Zeroth-Order Optimization (R-AdaZO). Specifically, we first show the untapped variance reduction effect of first moment estimate on ZO gradient estimation, which improves the accuracy and stability of ZO updates. We then refine the second moment estimate based on these variance-reduced gradient estimates to better capture the geometry of the optimization landscape, enabling a more effective scaling of ZO updates. We present rigorous theoretical analysis to show (I) the first analysis to the variance reduction of first moment estimate in ZO optimization, (II) the improved second moment estimates with a more accurate approximation of its variance-free ideal, (III) the first variance-aware convergence framework for adaptive ZO methods, which may be of independent interest, and (IV) the faster convergence of R-AdaZO than existing baselines like ZO-AdaMM. Our extensive experiments, including synthetic problems, black-box adversarial attack, and memory-efficient fine-tuning of large language models (LLMs), further verify the superior convergence of R-AdaZO, indicating that R-AdaZO offers an improved solution for real-world ZO optimization challenges.

摘要: 最近，零阶(ZO)优化在诸如黑盒系统和资源受限环境等无法获取或负担不起梯度信息的场景中扮演着重要的角色。虽然现有的自适应方法如ZO-AdaMM已经显示出很好的前景，但它们在优化过程中对矩信息的利用不足从根本上限制了它们，通常导致收敛性能不佳。为了克服这些局限性，本文引入了改进的自适应零阶优化算法(R-AdaZO)。具体地说，我们首先展示了一阶矩估计对ZO梯度估计的未开发的减方差效果，从而提高了ZO更新的精度和稳定性。然后，我们基于这些经方差减少的梯度估计来改进二阶矩估计，以更好地捕捉优化场景的几何形状，从而实现更有效的ZO更新缩放。我们给出了严格的理论分析，以证明(I)第一次分析ZO优化中一阶矩估计的方差降低，(Ii)改进的二阶矩估计更精确地逼近其无方差理想，(Iii)自适应ZO方法的第一个方差感知收敛框架，它可能是独立的，以及(Iv)R-AdaZO比现有基线(如ZO-AdaMM)更快的收敛。我们的大量实验，包括合成问题、黑盒对抗攻击和对大型语言模型(LLM)的内存效率优化，进一步验证了R-AdaZO的优越收敛能力，表明R-AdaZO为现实世界的ZO优化挑战提供了一种改进的解决方案。



## **21. Encrypted Large Model Inference: The Equivariant Encryption Paradigm**

加密大模型推理：等变加密范式 cs.CR

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01013v1) [paper-pdf](http://arxiv.org/pdf/2502.01013v1)

**Authors**: James Buban, Hongyang Zhang, Claudio Angione, Harry Yang, Ahmad Farhan, Seyfal Sultanov, Michael Du, Xuran Ma, Zihao Wang, Yue Zhao, Arria Owlia, Fielding Johnston, Patrick Colangelo

**Abstract**: Large scale deep learning model, such as modern language models and diffusion architectures, have revolutionized applications ranging from natural language processing to computer vision. However, their deployment in distributed or decentralized environments raises significant privacy concerns, as sensitive data may be exposed during inference. Traditional techniques like secure multi-party computation, homomorphic encryption, and differential privacy offer partial remedies but often incur substantial computational overhead, latency penalties, or limited compatibility with non-linear network operations. In this work, we introduce Equivariant Encryption (EE), a novel paradigm designed to enable secure, "blind" inference on encrypted data with near zero performance overhead. Unlike fully homomorphic approaches that encrypt the entire computational graph, EE selectively obfuscates critical internal representations within neural network layers while preserving the exact functionality of both linear and a prescribed set of non-linear operations. This targeted encryption ensures that raw inputs, intermediate activations, and outputs remain confidential, even when processed on untrusted infrastructure. We detail the theoretical foundations of EE, compare its performance and integration complexity against conventional privacy preserving techniques, and demonstrate its applicability across a range of architectures, from convolutional networks to large language models. Furthermore, our work provides a comprehensive threat analysis, outlining potential attack vectors and baseline strategies, and benchmarks EE against standard inference pipelines in decentralized settings. The results confirm that EE maintains high fidelity and throughput, effectively bridging the gap between robust data confidentiality and the stringent efficiency requirements of modern, large scale model inference.

摘要: 大规模深度学习模型，如现代语言模型和扩散体系结构，已经使从自然语言处理到计算机视觉的应用发生了革命性的变化。然而，它们在分布式或分散式环境中的部署会引起严重的隐私问题，因为敏感数据可能会在推理过程中暴露出来。安全多方计算、同态加密和差异隐私等传统技术提供了部分补救措施，但通常会招致大量计算开销、延迟惩罚或与非线性网络操作的有限兼容性。在这项工作中，我们引入了等变加密(EE)，这是一种新的范例，旨在以几乎为零的性能开销实现对加密数据的安全“盲”推理。与加密整个计算图形的完全同态方法不同，EE选择性地混淆神经网络层内的关键内部表示，同时保留线性和指定的一组非线性操作的确切功能。这种有针对性的加密确保原始输入、中间激活和输出保密，即使在不受信任的基础设施上处理时也是如此。我们详细介绍了EE的理论基础，将其性能和集成复杂性与传统的隐私保护技术进行了比较，并展示了它在从卷积网络到大型语言模型的一系列体系结构中的适用性。此外，我们的工作提供了全面的威胁分析，概述了潜在的攻击向量和基线策略，并针对分散环境下的标准推理管道对EE进行了基准测试。结果证实，EE保持了高保真度和高吞吐量，有效地弥合了稳健的数据机密性和现代大规模模型推理的严格效率要求之间的差距。



## **22. Time-Reversal Provides Unsupervised Feedback to LLMs**

计时器向LLM提供无监督反馈 cs.CL

Accepted as a spotlight in NeurIPS 2024

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2412.02626v3) [paper-pdf](http://arxiv.org/pdf/2412.02626v3)

**Authors**: Yerram Varun, Rahul Madhavan, Sravanti Addepalli, Arun Suggala, Karthikeyan Shanmugam, Prateek Jain

**Abstract**: Large Language Models (LLMs) are typically trained to predict in the forward direction of time. However, recent works have shown that prompting these models to look back and critique their own generations can produce useful feedback. Motivated by this, we explore the question of whether LLMs can be empowered to think (predict and score) backwards to provide unsupervised feedback that complements forward LLMs. Towards this, we introduce Time Reversed Language Models (TRLMs), which can score and generate queries when conditioned on responses, effectively functioning in the reverse direction of time. Further, to effectively infer in the response to query direction, we pre-train and fine-tune a language model (TRLM-Ba) in the reverse token order from scratch. We show empirically (and theoretically in a stylized setting) that time-reversed models can indeed complement forward model predictions when used to score the query given response for re-ranking multiple forward generations. We obtain up to 5\% improvement on the widely used AlpacaEval Leaderboard over the competent baseline of best-of-N re-ranking using self log-perplexity scores. We further show that TRLM scoring outperforms conventional forward scoring of response given query, resulting in significant gains in applications such as citation generation and passage retrieval. We next leverage the generative ability of TRLM to augment or provide unsupervised feedback to input safety filters of LLMs, demonstrating a drastic reduction in false negative rate with negligible impact on false positive rates against several attacks published on the popular JailbreakBench leaderboard.

摘要: 大型语言模型(LLM)通常被训练为在时间的正向进行预测。然而，最近的研究表明，促使这些模型回顾和批评他们自己的几代人可以产生有用的反馈。受此启发，我们探讨了LLM是否可以被赋予向后思考(预测和评分)的能力，以提供无监督的反馈来补充前向LLM。为此，我们引入了时间反转语言模型(TRLMS)，该模型可以根据响应进行评分并生成查询，有效地沿时间的相反方向运行。此外，为了有效地推断对查询方向的响应，我们从头开始以相反的令牌顺序预先训练和微调语言模型(TRLM-BA)。我们在经验上(理论上是在风格化的环境中)表明，当时间倒置模型用于对给定响应的查询进行重新排序时，时间倒置模型确实可以补充正向模型预测。我们在广泛使用的AlpacaEval排行榜上获得了高达5%的改进，超过了使用自我对数困惑分数重新排序的合格基线。我们进一步表明，TRLM评分优于传统的对给定查询的回复的前向评分，从而在引文生成和段落检索等应用中获得了显著的收益。接下来，我们利用TRLM的生成能力来增强或向LLMS的输入安全过滤器提供无监督反馈，展示了假阴性率的大幅降低，而对流行的JailBreak Btch排行榜上发布的几种攻击的错误确认率的影响可以忽略不计。



## **23. Gandalf the Red: Adaptive Security for LLMs**

红色甘道夫：LLM的自适应安全 cs.LG

Niklas Pfister, V\'aclav Volhejn and Manuel Knott contributed equally

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2501.07927v2) [paper-pdf](http://arxiv.org/pdf/2501.07927v2)

**Authors**: Niklas Pfister, Václav Volhejn, Manuel Knott, Santiago Arias, Julia Bazińska, Mykhailo Bichurin, Alan Commike, Janet Darling, Peter Dienes, Matthew Fiedler, David Haber, Matthias Kraft, Marco Lancini, Max Mathys, Damián Pascual-Ortiz, Jakub Podolak, Adrià Romero-López, Kyriacos Shiarlis, Andreas Signer, Zsolt Terek, Athanasios Theocharis, Daniel Timbrell, Samuel Trautwein, Samuel Watts, Yun-Han Wu, Mateo Rojas-Carulla

**Abstract**: Current evaluations of defenses against prompt attacks in large language model (LLM) applications often overlook two critical factors: the dynamic nature of adversarial behavior and the usability penalties imposed on legitimate users by restrictive defenses. We propose D-SEC (Dynamic Security Utility Threat Model), which explicitly separates attackers from legitimate users, models multi-step interactions, and expresses the security-utility in an optimizable form. We further address the shortcomings in existing evaluations by introducing Gandalf, a crowd-sourced, gamified red-teaming platform designed to generate realistic, adaptive attack. Using Gandalf, we collect and release a dataset of 279k prompt attacks. Complemented by benign user data, our analysis reveals the interplay between security and utility, showing that defenses integrated in the LLM (e.g., system prompts) can degrade usability even without blocking requests. We demonstrate that restricted application domains, defense-in-depth, and adaptive defenses are effective strategies for building secure and useful LLM applications.

摘要: 目前对大型语言模型(LLM)应用程序中针对即时攻击的防御措施的评估往往忽略了两个关键因素：敌意行为的动态性质和限制性防御措施对合法用户的可用性惩罚。本文提出了动态安全效用威胁模型D-SEC，它明确地将攻击者和合法用户分开，对多步交互进行建模，并以优化的形式表达安全效用。我们通过引入Gandalf进一步解决了现有评估中的缺陷，Gandalf是一个众包、游戏化的红色团队平台，旨在生成现实的、自适应的攻击。使用Gandalf，我们收集并发布了279K提示攻击的数据集。在良性用户数据的补充下，我们的分析揭示了安全性和实用性之间的相互作用，表明LLM中集成的防御措施(例如系统提示)即使在不阻止请求的情况下也会降低可用性。我们演示了受限应用程序域、深度防御和自适应防御是构建安全且有用的LLM应用程序的有效策略。



## **24. From Compliance to Exploitation: Jailbreak Prompt Attacks on Multimodal LLMs**

从合规到剥削：对多模式LLM的越狱立即攻击 cs.CR

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2502.00735v1) [paper-pdf](http://arxiv.org/pdf/2502.00735v1)

**Authors**: Chun Wai Chiu, Linghan Huang, Bo Li, Huaming Chen

**Abstract**: Large Language Models (LLMs) have seen widespread applications across various domains due to their growing ability to process diverse types of input data, including text, audio, image and video. While LLMs have demonstrated outstanding performance in understanding and generating contexts for different scenarios, they are vulnerable to prompt-based attacks, which are mostly via text input. In this paper, we introduce the first voice-based jailbreak attack against multimodal LLMs, termed as Flanking Attack, which can process different types of input simultaneously towards the multimodal LLMs. Our work is motivated by recent advancements in monolingual voice-driven large language models, which have introduced new attack surfaces beyond traditional text-based vulnerabilities for LLMs. To investigate these risks, we examine the frontier multimodal LLMs, which can be accessed via different types of inputs such as audio input, focusing on how adversarial prompts can bypass its defense mechanisms. We propose a novel strategy, in which the disallowed prompt is flanked by benign, narrative-driven prompts. It is integrated in the Flanking Attack which attempts to humanizes the interaction context and execute the attack through a fictional setting. To better evaluate the attack performance, we present a semi-automated self-assessment framework for policy violation detection. We demonstrate that Flank Attack is capable of manipulating state-of-the-art LLMs into generating misaligned and forbidden outputs, which achieves an average attack success rate ranging from 0.67 to 0.93 across seven forbidden scenarios. These findings highlight both the potency of prompt-based obfuscation in voice-enabled contexts and the limitations of current LLMs' moderation safeguards and the urgent need for advanced defense strategies to address the challenges posed by evolving, context-rich attacks.

摘要: 大语言模型因其处理文本、音频、图像和视频等不同类型输入数据的能力日益增强，在各个领域得到了广泛的应用。虽然LLM在理解和生成不同场景的上下文方面表现出了出色的性能，但它们很容易受到基于提示的攻击，这些攻击主要是通过文本输入进行的。在本文中，我们介绍了第一个基于语音的针对多模式LLMS的越狱攻击，称为侧翼攻击，它可以同时处理针对多模式LLMS的不同类型的输入。我们的工作是受到单语言语音驱动的大型语言模型的最新进展的推动，这些模型在传统的基于文本的LLMS漏洞之外引入了新的攻击面。为了调查这些风险，我们研究了前沿多模式LLMS，这些LLMS可以通过不同类型的输入(如音频输入)访问，重点关注对抗性提示如何绕过其防御机制。我们提出了一种新的策略，在不允许的提示的两侧是良性的、叙事驱动的提示。它被整合到侧翼攻击中，试图使交互上下文人性化，并通过虚构的设置执行攻击。为了更好地评估攻击性能，我们提出了一个半自动的策略违规检测自我评估框架。我们证明了侧翼攻击能够操纵最先进的LLM产生未对齐和禁止的输出，在七个禁止场景中获得了从0.67到0.93的平均攻击成功率。这些发现既突显了基于提示的混淆在语音支持的上下文中的有效性，也突显了当前LLMS适度保障的局限性，以及迫切需要先进的防御策略来应对不断演变的、上下文丰富的攻击带来的挑战。



## **25. "I am bad": Interpreting Stealthy, Universal and Robust Audio Jailbreaks in Audio-Language Models**

“我很坏”：在音频语言模型中解释秘密、普遍和稳健的音频越狱 cs.LG

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2502.00718v1) [paper-pdf](http://arxiv.org/pdf/2502.00718v1)

**Authors**: Isha Gupta, David Khachaturov, Robert Mullins

**Abstract**: The rise of multimodal large language models has introduced innovative human-machine interaction paradigms but also significant challenges in machine learning safety. Audio-Language Models (ALMs) are especially relevant due to the intuitive nature of spoken communication, yet little is known about their failure modes. This paper explores audio jailbreaks targeting ALMs, focusing on their ability to bypass alignment mechanisms. We construct adversarial perturbations that generalize across prompts, tasks, and even base audio samples, demonstrating the first universal jailbreaks in the audio modality, and show that these remain effective in simulated real-world conditions. Beyond demonstrating attack feasibility, we analyze how ALMs interpret these audio adversarial examples and reveal them to encode imperceptible first-person toxic speech - suggesting that the most effective perturbations for eliciting toxic outputs specifically embed linguistic features within the audio signal. These results have important implications for understanding the interactions between different modalities in multimodal models, and offer actionable insights for enhancing defenses against adversarial audio attacks.

摘要: 多通道大型语言模型的兴起引入了创新的人机交互范式，但也给机器学习的安全性带来了重大挑战。由于口语交流的直觉性，音频语言模型(ALM)尤其相关，但人们对其失败模式知之甚少。本文探讨了针对施舍的音频越狱，重点是它们绕过对齐机制的能力。我们构建了跨提示、任务甚至基本音频样本的对抗性扰动，演示了音频通道中的第一个通用越狱，并表明这些扰动在模拟的真实世界条件下仍然有效。除了展示攻击的可行性外，我们还分析了ALMS如何解释这些音频对抗性例子，并将它们揭示为编码不可感知的第一人称有毒言语--这表明，引发有毒输出的最有效扰动具体地将语言特征嵌入音频信号中。这些结果对于理解多通道模型中不同通道之间的相互作用具有重要意义，并为增强对敌方音频攻击的防御提供了可操作的见解。



## **26. LLM Safety Alignment is Divergence Estimation in Disguise**

LLM安全调整是伪装的分歧估计 cs.LG

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2502.00657v1) [paper-pdf](http://arxiv.org/pdf/2502.00657v1)

**Authors**: Rajdeep Haldar, Ziyi Wang, Qifan Song, Guang Lin, Yue Xing

**Abstract**: We propose a theoretical framework demonstrating that popular Large Language Model (LLM) alignment methods, including Reinforcement Learning from Human Feedback (RLHF) and alternatives, fundamentally function as divergence estimators between aligned (preferred or safe) and unaligned (less-preferred or harmful) distributions. This explains the separation phenomenon between safe and harmful prompts in the model hidden representation after alignment. Inspired by the theoretical results, we identify that some alignment methods are better than others in terms of separation and, introduce a new method, KLDO, and further demonstrate the implication of our theories. We advocate for compliance-refusal datasets over preference datasets to enhance safety alignment, supported by both theoretical reasoning and empirical evidence. Additionally, to quantify safety separation, we leverage a distance metric in the representation space and statistically validate its efficacy as a statistical significant indicator of LLM resilience against jailbreak attacks.

摘要: 我们提出了一个理论框架，证明了流行的大语言模型(LLM)对齐方法，包括从人类反馈的强化学习(RLHF)和替代方法，基本上是对齐(优先或安全)和非对齐(较不优先或有害)分布之间的背离估计。这解释了对齐后模型隐藏表示中的安全提示和有害提示分离的现象。在理论结果的启发下，我们确定了一些比对方法在分离方面比其他方法更好，并介绍了一种新的方法KLDO，进一步论证了我们的理论的含义。我们主张使用合规拒绝数据集而不是偏好数据集，以增强安全性一致性，这得到了理论推理和经验证据的支持。此外，为了量化安全分离，我们利用表示空间中的距离度量，并在统计上验证其作为LLM对越狱攻击弹性的显著指标的有效性。



## **27. Towards Robust Multimodal Large Language Models Against Jailbreak Attacks**

迈向抵御越狱攻击的稳健多模式大型语言模型 cs.CR

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2502.00653v1) [paper-pdf](http://arxiv.org/pdf/2502.00653v1)

**Authors**: Ziyi Yin, Yuanpu Cao, Han Liu, Ting Wang, Jinghui Chen, Fenhlong Ma

**Abstract**: While multimodal large language models (MLLMs) have achieved remarkable success in recent advancements, their susceptibility to jailbreak attacks has come to light. In such attacks, adversaries exploit carefully crafted prompts to coerce models into generating harmful or undesirable content. Existing defense mechanisms often rely on external inference steps or safety alignment training, both of which are less effective and impractical when facing sophisticated adversarial perturbations in white-box scenarios. To address these challenges and bolster MLLM robustness, we introduce SafeMLLM by adopting an adversarial training framework that alternates between an attack step for generating adversarial noise and a model updating step. At the attack step, SafeMLLM generates adversarial perturbations through a newly proposed contrastive embedding attack (CoE-Attack), which optimizes token embeddings under a contrastive objective. SafeMLLM then updates model parameters to neutralize the perturbation effects while preserving model utility on benign inputs. We evaluate SafeMLLM across six MLLMs and six jailbreak methods spanning multiple modalities. Experimental results show that SafeMLLM effectively defends against diverse attacks, maintaining robust performance and utilities.

摘要: 虽然多模式大型语言模型(MLLM)在最近的进步中取得了显著的成功，但它们对越狱攻击的敏感性已经暴露出来。在此类攻击中，攻击者利用精心设计的提示来强迫模型生成有害或不受欢迎的内容。现有的防御机制往往依赖于外部推理步骤或安全对齐训练，在白盒场景中面对复杂的对手扰动时，这两种方法都不太有效和不切实际。为了应对这些挑战并增强MLLM的稳健性，我们引入了SafeMLLM，采用了一种对抗性训练框架，该框架在生成对抗性噪声的攻击步骤和模型更新步骤之间交替。在攻击阶段，SafeMLLM通过新提出的对比性嵌入攻击(COE-Attack)来产生敌意扰动，该攻击在对比性目标下优化令牌嵌入。SafeMLLM然后更新模型参数，以中和扰动影响，同时保留对良性输入的模型效用。我们评估了六种MLLM和六种越狱方法的SafeMLLM，这些方法跨越多个医疗设备。实验结果表明，SafeMLLM能够有效地防御各种攻击，并保持了较强的性能和实用性。



## **28. Self-Instruct Few-Shot Jailbreaking: Decompose the Attack into Pattern and Behavior Learning**

自学少枪越狱：将攻击分解为模式和行为学习 cs.AI

**SubmitDate**: 2025-02-01    [abs](http://arxiv.org/abs/2501.07959v2) [paper-pdf](http://arxiv.org/pdf/2501.07959v2)

**Authors**: Jiaqi Hua, Wanxu Wei

**Abstract**: Recently, several works have been conducted on jailbreaking Large Language Models (LLMs) with few-shot malicious demos. In particular, Zheng et al. focus on improving the efficiency of Few-Shot Jailbreaking (FSJ) by injecting special tokens into the demos and employing demo-level random search, known as Improved Few-Shot Jailbreaking (I-FSJ). Nevertheless, we notice that this method may still require a long context to jailbreak advanced models e.g. 32 shots of demos for Meta-Llama-3-8B-Instruct (Llama-3) \cite{llama3modelcard}. In this paper, we discuss the limitations of I-FSJ and propose Self-Instruct Few-Shot Jailbreaking (Self-Instruct-FSJ) facilitated with the demo-level greedy search. This framework decomposes the FSJ attack into pattern and behavior learning to exploit the model's vulnerabilities in a more generalized and efficient way. We conduct elaborate experiments to evaluate our method on common open-source models and compare it with baseline algorithms. Our code is available at https://github.com/iphosi/Self-Instruct-FSJ.

摘要: 最近，一些关于越狱大语言模型(LLM)的工作已经进行，并提供了几个几乎不可能成功的恶意演示。特别是，郑等人。专注于通过向演示中注入特殊令牌并采用演示级随机搜索来提高少发越狱(FSJ)的效率，即改进的少发越狱(I-FSJ)。然而，我们注意到，这种方法可能仍然需要较长的上下文才能越狱高级模型，例如Meta-Llama-3-8B-Indict(Llama-3)\Cite{llama3 ModelCard}的32个演示镜头。在本文中，我们讨论了I-FSJ的局限性，并提出了一种基于演示级贪婪搜索的自指导式少发越狱算法(SELF-Induct-FSJ)。该框架将FSJ攻击分解为模式学习和行为学习，以更通用、更有效的方式利用模型的漏洞。我们在常见的开源模型上进行了详细的实验来评估我们的方法，并将其与基线算法进行了比较。我们的代码可以在https://github.com/iphosi/Self-Instruct-FSJ.上找到



## **29. Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation**

谜语我！检索增强一代的隐形会员推断 cs.CR

**SubmitDate**: 2025-02-01    [abs](http://arxiv.org/abs/2502.00306v1) [paper-pdf](http://arxiv.org/pdf/2502.00306v1)

**Authors**: Ali Naseh, Yuefeng Peng, Anshuman Suri, Harsh Chaudhari, Alina Oprea, Amir Houmansadr

**Abstract**: Retrieval-Augmented Generation (RAG) enables Large Language Models (LLMs) to generate grounded responses by leveraging external knowledge databases without altering model parameters. Although the absence of weight tuning prevents leakage via model parameters, it introduces the risk of inference adversaries exploiting retrieved documents in the model's context. Existing methods for membership inference and data extraction often rely on jailbreaking or carefully crafted unnatural queries, which can be easily detected or thwarted with query rewriting techniques common in RAG systems. In this work, we present Interrogation Attack (IA), a membership inference technique targeting documents in the RAG datastore. By crafting natural-text queries that are answerable only with the target document's presence, our approach demonstrates successful inference with just 30 queries while remaining stealthy; straightforward detectors identify adversarial prompts from existing methods up to ~76x more frequently than those generated by our attack. We observe a 2x improvement in TPR@1%FPR over prior inference attacks across diverse RAG configurations, all while costing less than $0.02 per document inference.

摘要: 检索-增强生成(RAG)使大型语言模型(LLM)能够通过利用外部知识数据库生成接地响应，而无需更改模型参数。尽管没有权重调整防止通过模型参数泄漏，但它引入了推理对手在模型上下文中利用检索到的文档的风险。现有的成员关系推断和数据提取方法通常依赖于越狱或精心设计的非自然查询，这些查询可以很容易地被RAG系统中常见的查询重写技术检测到或阻止。在这项工作中，我们提出了询问攻击(IA)，这是一种针对RAG数据存储中的文档的成员关系推理技术。通过精心设计只能根据目标文档的存在来回答的自然文本查询，我们的方法只需30个查询即可成功推理，同时保持隐蔽性；直接的检测器识别来自现有方法的敌意提示的频率比我们的攻击生成的提示高约76倍。我们观察到，在不同的RAG配置中，TPR@1%的FPR比以前的推理攻击提高了2倍，而每个文档推理的成本都不到0.02美元。



## **30. Byzantine-Resilient Zero-Order Optimization for Communication-Efficient Heterogeneous Federated Learning**

具有拜占庭弹性的零阶优化，用于通信高效的异类联邦学习 cs.LG

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2502.00193v1) [paper-pdf](http://arxiv.org/pdf/2502.00193v1)

**Authors**: Maximilian Egger, Mayank Bakshi, Rawad Bitar

**Abstract**: We introduce CyBeR-0, a Byzantine-resilient federated zero-order optimization method that is robust under Byzantine attacks and provides significant savings in uplink and downlink communication costs. We introduce transformed robust aggregation to give convergence guarantees for general non-convex objectives under client data heterogeneity. Empirical evaluations for standard learning tasks and fine-tuning large language models show that CyBeR-0 exhibits stable performance with only a few scalars per-round communication cost and reduced memory requirements.

摘要: 我们引入CyBeR-0，这是一种具有拜占庭弹性的联邦零阶优化方法，在拜占庭攻击下具有鲁棒性，并大幅节省上行链路和下行链路通信成本。我们引入转换的鲁棒聚合，为客户数据异类下的一般非凸目标提供收敛保证。对标准学习任务和微调大型语言模型的经验评估表明，CyBeR-0表现出稳定的性能，每轮通信成本只有几个纯量，内存需求也降低。



## **31. UniGuard: Towards Universal Safety Guardrails for Jailbreak Attacks on Multimodal Large Language Models**

UniGuard：为多模式大型语言模型越狱攻击建立通用安全护栏 cs.CL

14 pages

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2411.01703v2) [paper-pdf](http://arxiv.org/pdf/2411.01703v2)

**Authors**: Sejoon Oh, Yiqiao Jin, Megha Sharma, Donghyun Kim, Eric Ma, Gaurav Verma, Srijan Kumar

**Abstract**: Multimodal large language models (MLLMs) have revolutionized vision-language understanding but remain vulnerable to multimodal jailbreak attacks, where adversarial inputs are meticulously crafted to elicit harmful or inappropriate responses. We propose UniGuard, a novel multimodal safety guardrail that jointly considers the unimodal and cross-modal harmful signals. UniGuard trains a multimodal guardrail to minimize the likelihood of generating harmful responses in a toxic corpus. The guardrail can be seamlessly applied to any input prompt during inference with minimal computational costs. Extensive experiments demonstrate the generalizability of UniGuard across multiple modalities, attack strategies, and multiple state-of-the-art MLLMs, including LLaVA, Gemini Pro, GPT-4o, MiniGPT-4, and InstructBLIP. Notably, this robust defense mechanism maintains the models' overall vision-language understanding capabilities.

摘要: 多模式大型语言模型（MLLM）彻底改变了视觉语言理解，但仍然容易受到多模式越狱攻击的影响，其中对抗性输入经过精心设计，以引发有害或不当的反应。我们提出了UniGuard，这是一种新型的多模式安全护栏，它联合考虑了单模式和跨模式有害信号。UniGuard训练多模式护栏，以最大限度地降低有毒主体中产生有害反应的可能性。护栏可以无缝地应用于推理期间的任何输入提示，并且计算成本最低。大量实验证明了UniGuard在多种模式、攻击策略和多种最先进的MLLM中的通用性，包括LLaVA、Gemini Pro、GPT-4 o、MiniGPT-4和DirecectBLIP。值得注意的是，这种强大的防御机制维持了模型的整体视觉语言理解能力。



## **32. Enhancing Model Defense Against Jailbreaks with Proactive Safety Reasoning**

利用主动安全推理增强模型对越狱的防御 cs.CR

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2501.19180v1) [paper-pdf](http://arxiv.org/pdf/2501.19180v1)

**Authors**: Xianglin Yang, Gelei Deng, Jieming Shi, Tianwei Zhang, Jin Song Dong

**Abstract**: Large language models (LLMs) are vital for a wide range of applications yet remain susceptible to jailbreak threats, which could lead to the generation of inappropriate responses. Conventional defenses, such as refusal and adversarial training, often fail to cover corner cases or rare domains, leaving LLMs still vulnerable to more sophisticated attacks. We propose a novel defense strategy, Safety Chain-of-Thought (SCoT), which harnesses the enhanced \textit{reasoning capabilities} of LLMs for proactive assessment of harmful inputs, rather than simply blocking them. SCoT augments any refusal training datasets to critically analyze the intent behind each request before generating answers. By employing proactive reasoning, SCoT enhances the generalization of LLMs across varied harmful queries and scenarios not covered in the safety alignment corpus. Additionally, it generates detailed refusals specifying the rules violated. Comparative evaluations show that SCoT significantly surpasses existing defenses, reducing vulnerability to out-of-distribution issues and adversarial manipulations while maintaining strong general capabilities.

摘要: 大型语言模型(LLM)对于广泛的应用至关重要，但仍然容易受到越狱威胁的影响，这可能会导致产生不适当的响应。常规防御，如拒绝和对抗性训练，往往无法覆盖角落案例或稀有领域，使LLM仍然容易受到更复杂的攻击。我们提出了一种新的防御策略，安全思想链(SCOT)，它利用LLMS增强的\文本{推理能力}来主动评估有害输入，而不是简单地阻止它们。SCOT增加了任何拒绝训练数据集，以便在生成答案之前批判性地分析每个请求背后的意图。通过使用主动推理，SCOT增强了LLMS在安全匹配语料库中未涵盖的各种有害查询和场景中的泛化。此外，它还生成详细的拒绝，指定违反的规则。比较评估表明，SCOT大大超过了现有的防御系统，在保持强大的一般能力的同时，减少了对分配外问题和对抗性操纵的脆弱性。



## **33. Targeted Vaccine: Safety Alignment for Large Language Models against Harmful Fine-Tuning via Layer-wise Perturbation**

有针对性的疫苗：大型语言模型的安全调整，防止通过分层扰动进行有害的微调 cs.LG

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2410.09760v3) [paper-pdf](http://arxiv.org/pdf/2410.09760v3)

**Authors**: Guozhi Liu, Weiwei Lin, Tiansheng Huang, Ruichao Mo, Qi Mu, Li Shen

**Abstract**: Harmful fine-tuning attack poses a serious threat to the online fine-tuning service. Vaccine, a recent alignment-stage defense, applies uniform perturbation to all layers of embedding to make the model robust to the simulated embedding drift. However, applying layer-wise uniform perturbation may lead to excess perturbations for some particular safety-irrelevant layers, resulting in defense performance degradation and unnecessary memory consumption. To address this limitation, we propose Targeted Vaccine (T-Vaccine), a memory-efficient safety alignment method that applies perturbation to only selected layers of the model. T-Vaccine follows two core steps: First, it uses gradient norm as a statistical metric to identify the safety-critical layers. Second, instead of applying uniform perturbation across all layers, T-Vaccine only applies perturbation to the safety-critical layers while keeping other layers frozen during training. Results show that T-Vaccine outperforms Vaccine in terms of both defense effectiveness and resource efficiency. Comparison with other defense baselines, e.g., RepNoise and TAR also demonstrate the superiority of T-Vaccine. Notably, T-Vaccine is the first defense that can address harmful fine-tuning issues for a 7B pre-trained models trained on consumer GPUs with limited memory (e.g., RTX 4090). Our code is available at https://github.com/Lslland/T-Vaccine.

摘要: 有害微调攻击对在线微调服务构成严重威胁。疫苗是最近的一种对齐阶段防御方法，它将均匀扰动应用于嵌入的所有层，以使模型对模拟的嵌入漂移具有鲁棒性。然而，分层均匀扰动可能会导致某些特定安全无关层的过度扰动，导致防御性能下降和不必要的内存消耗。为了解决这一局限性，我们提出了靶向疫苗(T-Vaccine)，这是一种内存高效的安全对齐方法，仅对模型的选定层应用扰动。T-Vaccine遵循两个核心步骤：首先，它使用梯度范数作为统计度量来识别安全关键层。其次，T-Vaccine不是在所有层上应用统一的扰动，而是只对安全关键层应用扰动，而在训练期间保持其他层的冻结。结果表明，无论是防御效果还是资源效率，T疫苗都优于疫苗。与其他防御基线如RepNoise和TAR的比较也证明了T-疫苗的优越性。值得注意的是，T-Vaccine是第一个可以解决7B预培训模型的有害微调问题的防御系统，这些模型在内存有限的消费者GPU(例如RTX 4090)上进行了培训。我们的代码可以在https://github.com/Lslland/T-Vaccine.上找到



## **34. Towards the Worst-case Robustness of Large Language Models**

走向大型语言模型的最坏情况稳健性 cs.LG

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2501.19040v1) [paper-pdf](http://arxiv.org/pdf/2501.19040v1)

**Authors**: Huanran Chen, Yinpeng Dong, Zeming Wei, Hang Su, Jun Zhu

**Abstract**: Recent studies have revealed the vulnerability of Large Language Models (LLMs) to adversarial attacks, where the adversary crafts specific input sequences to induce harmful, violent, private, or incorrect outputs. Although various defenses have been proposed, they have not been evaluated by strong adaptive attacks, leaving the worst-case robustness of LLMs still intractable. By developing a stronger white-box attack, our evaluation results indicate that most typical defenses achieve nearly 0\% robustness.To solve this, we propose \textit{DiffTextPure}, a general defense that diffuses the (adversarial) input prompt using any pre-defined smoothing distribution, and purifies the diffused input using a pre-trained language model. Theoretically, we derive tight robustness lower bounds for all smoothing distributions using Fractal Knapsack or 0-1 Knapsack solvers. Under this framework, we certify the robustness of a specific case -- smoothing LLMs using a uniform kernel -- against \textit{any possible attack} with an average $\ell_0$ perturbation of 2.02 or an average suffix length of 6.41.

摘要: 最近的研究揭示了大型语言模型(LLM)在敌意攻击中的脆弱性，在这种攻击中，对手精心制作特定的输入序列来诱导有害的、暴力的、隐私的或错误的输出。虽然已经提出了各种防御措施，但它们还没有通过强自适应攻击进行评估，这使得LLM的最坏情况下的稳健性仍然很难解决。通过开发更强的白盒攻击，我们的评估结果表明，大多数典型的防御措施都达到了近0的稳健性，为了解决这个问题，我们提出了一种通用的防御措施，它使用任何预定义的平滑分布来扩散(对抗性的)输入提示，并使用预先训练的语言模型来净化扩散的输入。理论上，我们使用分形型背包或0-1背包求解器得到了所有光滑分布的紧鲁棒下界。在此框架下，我们证明了一种特殊情况--使用统一核的平滑LLMS--对平均$\ELL_0$扰动为2.02或平均后缀长度为6.41的文本{任何可能的攻击}的健壮性。



## **35. Importing Phantoms: Measuring LLM Package Hallucination Vulnerabilities**

Importing Phantoms: Measuring LLM Package Hallucination Vulnerabilities cs.LG

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2501.19012v1) [paper-pdf](http://arxiv.org/pdf/2501.19012v1)

**Authors**: Arjun Krishna, Erick Galinkin, Leon Derczynski, Jeffrey Martin

**Abstract**: Large Language Models (LLMs) have become an essential tool in the programmer's toolkit, but their tendency to hallucinate code can be used by malicious actors to introduce vulnerabilities to broad swathes of the software supply chain. In this work, we analyze package hallucination behaviour in LLMs across popular programming languages examining both existing package references and fictional dependencies. By analyzing this package hallucination behaviour we find potential attacks and suggest defensive strategies to defend against these attacks. We discover that package hallucination rate is predicated not only on model choice, but also programming language, model size, and specificity of the coding task request. The Pareto optimality boundary between code generation performance and package hallucination is sparsely populated, suggesting that coding models are not being optimized for secure code. Additionally, we find an inverse correlation between package hallucination rate and the HumanEval coding benchmark, offering a heuristic for evaluating the propensity of a model to hallucinate packages. Our metrics, findings and analyses provide a base for future models, securing AI-assisted software development workflows against package supply chain attacks.

摘要: 大型语言模型(LLM)已成为程序员工具包中的基本工具，但它们产生代码幻觉的倾向可被恶意行为者用来向软件供应链的大片区域引入漏洞。在这项工作中，我们分析了LLM中跨流行编程语言的包幻觉行为，检查了现有的包引用和虚构的依赖关系。通过分析这一包的幻觉行为，我们发现了潜在的攻击，并提出了防御策略来防御这些攻击。我们发现，包装幻觉率不仅取决于模型的选择，还取决于编程语言、模型大小和编码任务请求的特殊性。代码生成性能和包幻觉之间的Pareto最优边界很少，这表明编码模型没有针对安全代码进行优化。此外，我们发现程序包幻觉率和人类进化编码基准之间存在负相关，这为评估模型产生幻觉程序包的倾向提供了一个启发式方法。我们的指标、发现和分析为未来的模型提供了基础，确保人工智能辅助软件开发工作流免受程序包供应链攻击。



## **36. ASTPrompter: Weakly Supervised Automated Language Model Red-Teaming to Identify Low-Perplexity Toxic Prompts**

ASTencer：弱监督的自动化语言模型红色团队识别低困惑性有毒物种 cs.CL

10 pages, 7 pages of appendix, 3 tables, 3 figures

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2407.09447v3) [paper-pdf](http://arxiv.org/pdf/2407.09447v3)

**Authors**: Amelia F. Hardy, Houjun Liu, Bernard Lange, Duncan Eddy, Mykel J. Kochenderfer

**Abstract**: Conventional approaches for the automated red-teaming of large language models (LLMs) aim to identify prompts that elicit toxic outputs from a frozen language model (the defender). This often results in the prompting model (the adversary) producing text that is unlikely to arise during autoregression. In response, we propose a reinforcement learning formulation of LLM red-teaming designed to discover prompts that both (1) elicit toxic outputs from a defender and (2) have low perplexity as scored by that defender. These prompts are the most pertinent in a red-teaming setting because the defender generates them with high probability. We solve this formulation with an online and weakly supervised form of Identity Preference Optimization (IPO), attacking models ranging from 137M to 7.8B parameters. Our policy performs competitively, producing prompts that induce defender toxicity at a rate of 2-23 times higher than baseline across model scales. Importantly, these prompts have lower perplexity than both automatically generated and human-written attacks. Furthermore, our method creates black-box attacks with 5.4-14 times increased toxicity. To assess the downstream utility of our method, we use rollouts from our policy as negative examples for downstream toxicity tuning and demonstrate improved safety.

摘要: 大型语言模型(LLM)的自动红团队的传统方法旨在识别从冻结的语言模型(防御者)中引发有毒输出的提示。这通常会导致提示模型(对手)生成在自动回归过程中不太可能出现的文本。作为回应，我们提出了一种LLM红队的强化学习公式，旨在发现(1)从防御者那里引发有毒输出和(2)由该防御者评分的低困惑程度的提示。这些提示在红队环境中是最相关的，因为后卫生成它们的可能性很高。我们用在线和弱监督形式的身份偏好优化(IPO)来解决这个公式，攻击的模型范围从137M到7.8B参数。我们的政策具有竞争性，在模型范围内产生的提示导致防御者毒性的比率比基线高2-23倍。重要的是，与自动生成的攻击和人工编写的攻击相比，这些提示的困惑程度更低。此外，我们的方法还会产生毒性增加5.4-14倍的黑盒攻击。为了评估我们方法的下游效用，我们使用我们政策中的推出作为下游毒性调整的负面例子，并证明了改进的安全性。



## **37. LLM Cyber Evaluations Don't Capture Real-World Risk**

LLM网络评估无法捕捉现实世界的风险 cs.CR

11 pages

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2502.00072v1) [paper-pdf](http://arxiv.org/pdf/2502.00072v1)

**Authors**: Kamilė Lukošiūtė, Adam Swanda

**Abstract**: Large language models (LLMs) are demonstrating increasing prowess in cybersecurity applications, creating creating inherent risks alongside their potential for strengthening defenses. In this position paper, we argue that current efforts to evaluate risks posed by these capabilities are misaligned with the goal of understanding real-world impact. Evaluating LLM cybersecurity risk requires more than just measuring model capabilities -- it demands a comprehensive risk assessment that incorporates analysis of threat actor adoption behavior and potential for impact. We propose a risk assessment framework for LLM cyber capabilities and apply it to a case study of language models used as cybersecurity assistants. Our evaluation of frontier models reveals high compliance rates but moderate accuracy on realistic cyber assistance tasks. However, our framework suggests that this particular use case presents only moderate risk due to limited operational advantages and impact potential. Based on these findings, we recommend several improvements to align research priorities with real-world impact assessment, including closer academia-industry collaboration, more realistic modeling of attacker behavior, and inclusion of economic metrics in evaluations. This work represents an important step toward more effective assessment and mitigation of LLM-enabled cybersecurity risks.

摘要: 大型语言模型(LLM)在网络安全应用程序中显示出越来越强大的能力，在增强防御能力的同时，也带来了固有的风险。在这份立场文件中，我们认为目前评估这些能力带来的风险的努力与了解现实世界影响的目标不一致。评估LLM网络安全风险需要的不仅仅是测量模型能力--它还需要一个全面的风险评估，其中包括对威胁参与者采用行为和潜在影响的分析。我们提出了LLM网络能力的风险评估框架，并将其应用于作为网络安全助手的语言模型的案例研究。我们对前沿模型的评估显示，在现实的网络援助任务中，遵从率很高，但准确率中等。然而，我们的框架表明，由于运营优势和影响潜力有限，此特定用例仅带来中等风险。基于这些发现，我们建议进行几项改进，以使研究优先事项与现实世界的影响评估保持一致，包括更紧密的学术界和业界合作，对攻击者行为进行更现实的建模，以及将经济指标纳入评估。这项工作是朝着更有效地评估和缓解LLM启用的网络安全风险迈出的重要一步。



## **38. Evaluating LLM-based Personal Information Extraction and Countermeasures**

评估基于LLM的个人信息提取及其对策 cs.CR

To appear in USENIX Security Symposium 2025

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2408.07291v3) [paper-pdf](http://arxiv.org/pdf/2408.07291v3)

**Authors**: Yupei Liu, Yuqi Jia, Jinyuan Jia, Neil Zhenqiang Gong

**Abstract**: Automatically extracting personal information -- such as name, phone number, and email address -- from publicly available profiles at a large scale is a stepstone to many other security attacks including spear phishing. Traditional methods -- such as regular expression, keyword search, and entity detection -- achieve limited success at such personal information extraction. In this work, we perform a systematic measurement study to benchmark large language model (LLM) based personal information extraction and countermeasures. Towards this goal, we present a framework for LLM-based extraction attacks; collect four datasets including a synthetic dataset generated by GPT-4 and three real-world datasets with manually labeled eight categories of personal information; introduce a novel mitigation strategy based on prompt injection; and systematically benchmark LLM-based attacks and countermeasures using ten LLMs and five datasets. Our key findings include: LLM can be misused by attackers to accurately extract various personal information from personal profiles; LLM outperforms traditional methods; and prompt injection can defend against strong LLM-based attacks, reducing the attack to less effective traditional ones.

摘要: 从公开的个人资料中大规模自动提取个人信息--如姓名、电话号码和电子邮件地址--是应对包括鱼叉式网络钓鱼在内的许多其他安全攻击的一步。传统的方法--如正则表达式、关键字搜索和实体检测--在这种个人信息提取方面取得的成功有限。在这项工作中，我们对基于大语言模型的个人信息提取和对策进行了系统的测量研究。为此，我们提出了一个基于LLM的抽取攻击框架；收集了4个数据集，包括GPT-4生成的一个合成数据集和3个手动标注了8类个人信息的真实数据集；提出了一种新的基于即时注入的缓解策略；并使用10个LLM和5个数据集系统地测试了基于LLM的攻击和对策。我们的主要发现包括：LLM可以被攻击者误用，以准确地从个人档案中提取各种个人信息；LLM的性能优于传统方法；快速注入可以防御基于LLM的强大攻击，将攻击减少到效率较低的传统攻击。



## **39. Trading Inference-Time Compute for Adversarial Robustness**

交易推理时间计算对抗稳健性 cs.LG

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2501.18841v1) [paper-pdf](http://arxiv.org/pdf/2501.18841v1)

**Authors**: Wojciech Zaremba, Evgenia Nitishinskaya, Boaz Barak, Stephanie Lin, Sam Toyer, Yaodong Yu, Rachel Dias, Eric Wallace, Kai Xiao, Johannes Heidecke, Amelia Glaese

**Abstract**: We conduct experiments on the impact of increasing inference-time compute in reasoning models (specifically OpenAI o1-preview and o1-mini) on their robustness to adversarial attacks. We find that across a variety of attacks, increased inference-time compute leads to improved robustness. In many cases (with important exceptions), the fraction of model samples where the attack succeeds tends to zero as the amount of test-time compute grows. We perform no adversarial training for the tasks we study, and we increase inference-time compute by simply allowing the models to spend more compute on reasoning, independently of the form of attack. Our results suggest that inference-time compute has the potential to improve adversarial robustness for Large Language Models. We also explore new attacks directed at reasoning models, as well as settings where inference-time compute does not improve reliability, and speculate on the reasons for these as well as ways to address them.

摘要: 我们进行了实验，研究推理模型（特别是OpenAI o 1-preview和o 1-mini）中增加推理时计算对其对抗性攻击稳健性的影响。我们发现，在各种攻击中，增加的推断时间计算会提高稳健性。在许多情况下（除重要例外），随着测试时计算量的增加，攻击成功的模型样本比例趋于零。我们不对研究的任务进行对抗性训练，并且通过简单地允许模型在推理上花费更多的计算来增加推理时计算，而与攻击形式无关。我们的结果表明，推理时计算有潜力提高大型语言模型的对抗鲁棒性。我们还探索了针对推理模型的新攻击，以及推理时计算无法提高可靠性的设置，并推测这些攻击的原因以及解决方法。



## **40. Constitutional Classifiers: Defending against Universal Jailbreaks across Thousands of Hours of Red Teaming**

宪法分类：在数千小时的红色团队中抵御普遍越狱 cs.CL

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2501.18837v1) [paper-pdf](http://arxiv.org/pdf/2501.18837v1)

**Authors**: Mrinank Sharma, Meg Tong, Jesse Mu, Jerry Wei, Jorrit Kruthoff, Scott Goodfriend, Euan Ong, Alwin Peng, Raj Agarwal, Cem Anil, Amanda Askell, Nathan Bailey, Joe Benton, Emma Bluemke, Samuel R. Bowman, Eric Christiansen, Hoagy Cunningham, Andy Dau, Anjali Gopal, Rob Gilson, Logan Graham, Logan Howard, Nimit Kalra, Taesung Lee, Kevin Lin, Peter Lofgren, Francesco Mosconi, Clare O'Hara, Catherine Olsson, Linda Petrini, Samir Rajani, Nikhil Saxena, Alex Silverstein, Tanya Singh, Theodore Sumers, Leonard Tang, Kevin K. Troy, Constantin Weisser, Ruiqi Zhong, Giulio Zhou, Jan Leike, Jared Kaplan, Ethan Perez

**Abstract**: Large language models (LLMs) are vulnerable to universal jailbreaks-prompting strategies that systematically bypass model safeguards and enable users to carry out harmful processes that require many model interactions, like manufacturing illegal substances at scale. To defend against these attacks, we introduce Constitutional Classifiers: safeguards trained on synthetic data, generated by prompting LLMs with natural language rules (i.e., a constitution) specifying permitted and restricted content. In over 3,000 estimated hours of red teaming, no red teamer found a universal jailbreak that could extract information from an early classifier-guarded LLM at a similar level of detail to an unguarded model across most target queries. On automated evaluations, enhanced classifiers demonstrated robust defense against held-out domain-specific jailbreaks. These classifiers also maintain deployment viability, with an absolute 0.38% increase in production-traffic refusals and a 23.7% inference overhead. Our work demonstrates that defending against universal jailbreaks while maintaining practical deployment viability is tractable.

摘要: 大型语言模型(LLM)容易受到普遍越狱的影响，这促使策略系统性地绕过模型安全措施，使用户能够执行需要许多模型交互的有害过程，如大规模制造非法物质。为了防御这些攻击，我们引入了宪法分类器：对合成数据进行训练的保护措施，通过使用指定允许和受限内容的自然语言规则(即宪法)提示LLM生成。在超过3,000个小时的估计红色团队中，没有一个红色团队成员发现了一种通用越狱方法，可以从早期有分类器保护的LLM中提取信息，其细节级别与大多数目标查询的无保护模型相似。在自动化评估方面，增强的分类器对特定领域的越狱表现出了强大的防御能力。这些分类器还保持了部署的可行性，生产流量拒绝绝对增加了0.38%，推理开销增加了23.7%。我们的工作表明，在保持实际部署可行性的同时，防御普遍越狱是容易处理的。



## **41. IsolateGPT: An Execution Isolation Architecture for LLM-Based Agentic Systems**

IsolateGPT：基于LLM的统计系统的执行隔离架构 cs.CR

Accepted by the Network and Distributed System Security (NDSS)  Symposium 2025

**SubmitDate**: 2025-01-30    [abs](http://arxiv.org/abs/2403.04960v2) [paper-pdf](http://arxiv.org/pdf/2403.04960v2)

**Authors**: Yuhao Wu, Franziska Roesner, Tadayoshi Kohno, Ning Zhang, Umar Iqbal

**Abstract**: Large language models (LLMs) extended as systems, such as ChatGPT, have begun supporting third-party applications. These LLM apps leverage the de facto natural language-based automated execution paradigm of LLMs: that is, apps and their interactions are defined in natural language, provided access to user data, and allowed to freely interact with each other and the system. These LLM app ecosystems resemble the settings of earlier computing platforms, where there was insufficient isolation between apps and the system. Because third-party apps may not be trustworthy, and exacerbated by the imprecision of natural language interfaces, the current designs pose security and privacy risks for users. In this paper, we evaluate whether these issues can be addressed through execution isolation and what that isolation might look like in the context of LLM-based systems, where there are arbitrary natural language-based interactions between system components, between LLM and apps, and between apps. To that end, we propose IsolateGPT, a design architecture that demonstrates the feasibility of execution isolation and provides a blueprint for implementing isolation, in LLM-based systems. We evaluate IsolateGPT against a number of attacks and demonstrate that it protects against many security, privacy, and safety issues that exist in non-isolated LLM-based systems, without any loss of functionality. The performance overhead incurred by IsolateGPT to improve security is under 30% for three-quarters of tested queries.

摘要: 扩展为系统的大型语言模型(LLM)，如ChatGPT，已经开始支持第三方应用程序。这些LLM应用程序利用LLMS事实上基于自然语言的自动执行范例：即，应用程序及其交互以自然语言定义，提供对用户数据的访问，并允许彼此和系统自由交互。这些LLM应用程序生态系统类似于早期计算平台的设置，在那里应用程序和系统之间没有足够的隔离。由于第三方应用程序可能不值得信任，而且自然语言界面的不精确性加剧了这一问题，目前的设计给用户带来了安全和隐私风险。在本文中，我们评估是否可以通过执行隔离来解决这些问题，以及在基于LLM的系统环境中，这种隔离可能是什么样子的，其中系统组件之间、LLM与应用程序之间以及应用程序之间存在任意的基于自然语言的交互。为此，我们提出了IsolateGPT，这是一种设计体系结构，它展示了执行隔离的可行性，并为在基于LLM的系统中实现隔离提供了蓝图。我们对IsolateGPT进行了针对多种攻击的评估，并证明了它可以防御非隔离的基于LLM的系统中存在的许多安全、隐私和安全问题，而不会造成任何功能损失。对于四分之三的测试查询，IsolateGPT为提高安全性而产生的性能开销低于30%。



## **42. Exploring Audio Editing Features as User-Centric Privacy Defenses Against Emotion Inference Attacks**

探索音频编辑功能，以用户为中心的隐私防御情感推理攻击 cs.CR

Accepted for presentation(Poster) at PPAI-25: The 6th AAAI Workshop  on Privacy-Preserving Artificial Intelligence

**SubmitDate**: 2025-01-30    [abs](http://arxiv.org/abs/2501.18727v1) [paper-pdf](http://arxiv.org/pdf/2501.18727v1)

**Authors**: Mohd. Farhan Israk Soumik, W. K. M. Mithsara, Abdur R. Shahid, Ahmed Imteaj

**Abstract**: The rapid proliferation of speech-enabled technologies, including virtual assistants, video conferencing platforms, and wearable devices, has raised significant privacy concerns, particularly regarding the inference of sensitive emotional information from audio data. Existing privacy-preserving methods often compromise usability and security, limiting their adoption in practical scenarios. This paper introduces a novel, user-centric approach that leverages familiar audio editing techniques, specifically pitch and tempo manipulation, to protect emotional privacy without sacrificing usability. By analyzing popular audio editing applications on Android and iOS platforms, we identified these features as both widely available and usable. We rigorously evaluated their effectiveness against a threat model, considering adversarial attacks from diverse sources, including Deep Neural Networks (DNNs), Large Language Models (LLMs), and and reversibility testing. Our experiments, conducted on three distinct datasets, demonstrate that pitch and tempo manipulation effectively obfuscates emotional data. Additionally, we explore the design principles for lightweight, on-device implementation to ensure broad applicability across various devices and platforms.

摘要: 包括虚拟助理、视频会议平台和可穿戴设备在内的语音支持技术的迅速普及引发了人们对隐私的严重担忧，特别是关于从音频数据推断敏感情感信息的问题。现有的隐私保护方法往往会损害可用性和安全性，限制了它们在实际场景中的采用。本文介绍了一种新颖的、以用户为中心的方法，该方法利用熟悉的音频编辑技术，特别是音调和节奏操作，在不牺牲可用性的情况下保护情感隐私。通过分析Android和iOS平台上流行的音频编辑应用程序，我们发现这些功能广泛使用和使用。我们严格评估了它们对威胁模型的有效性，考虑了来自不同来源的对抗性攻击，包括深度神经网络(DNN)、大型语言模型(LLMS)和可逆性测试。我们在三个不同的数据集上进行的实验表明，音调和节奏操作有效地混淆了情感数据。此外，我们还探讨了轻量级设备上实施的设计原则，以确保跨各种设备和平台的广泛适用性。



## **43. BounTCHA: A CAPTCHA Utilizing Boundary Identification in AI-extended Videos**

BounTCHA：在AI扩展视频中利用边界识别的验证码 cs.CR

22 pages, 15 figures

**SubmitDate**: 2025-01-30    [abs](http://arxiv.org/abs/2501.18565v1) [paper-pdf](http://arxiv.org/pdf/2501.18565v1)

**Authors**: Lehao Lin, Ke Wang, Maha Abdallah, Wei Cai

**Abstract**: In recent years, the rapid development of artificial intelligence (AI) especially multi-modal Large Language Models (MLLMs), has enabled it to understand text, images, videos, and other multimedia data, allowing AI systems to execute various tasks based on human-provided prompts. However, AI-powered bots have increasingly been able to bypass most existing CAPTCHA systems, posing significant security threats to web applications. This makes the design of new CAPTCHA mechanisms an urgent priority. We observe that humans are highly sensitive to shifts and abrupt changes in videos, while current AI systems still struggle to comprehend and respond to such situations effectively. Based on this observation, we design and implement BounTCHA, a CAPTCHA mechanism that leverages human perception of boundaries in video transitions and disruptions. By utilizing AI's capability to expand original videos with prompts, we introduce unexpected twists and changes to create a pipeline for generating short videos for CAPTCHA purposes. We develop a prototype and conduct experiments to collect data on humans' time biases in boundary identification. This data serves as a basis for distinguishing between human users and bots. Additionally, we perform a detailed security analysis of BounTCHA, demonstrating its resilience against various types of attacks. We hope that BounTCHA will act as a robust defense, safeguarding millions of web applications in the AI-driven era.

摘要: 近年来，人工智能(AI)特别是多模式大语言模型(MLLMS)的快速发展，使其能够理解文本、图像、视频和其他多媒体数据，使AI系统能够根据人类提供的提示执行各种任务。然而，人工智能驱动的机器人越来越能够绕过大多数现有的验证码系统，对网络应用程序构成了重大的安全威胁。这使得设计新的验证码机制成为当务之急。我们观察到，人类对视频中的变化和突然变化高度敏感，而目前的人工智能系统仍然难以有效地理解和应对此类情况。基于这一观察结果，我们设计并实现了BounTCHA，这是一种验证码机制，它利用人类在视频过渡和中断中对边界的感知。通过利用人工智能的能力来扩展带有提示的原始视频，我们引入了意想不到的曲折和变化，以创建一条为验证码目的生成短视频的管道。我们开发了一个原型并进行了实验，以收集人类在边界识别中的时间偏差数据。这些数据是区分人类用户和机器人的基础。此外，我们对BounTCHA进行了详细的安全分析，展示了其对各种类型的攻击的弹性。我们希望BounTCHA将作为一道强大的防线，在人工智能驱动的时代保护数百万个网络应用程序。



## **44. Illusions of Relevance: Using Content Injection Attacks to Deceive Retrievers, Rerankers, and LLM Judges**

相关性幻觉：使用内容注入攻击来欺骗检索者、重复者和LLM评委 cs.IR

**SubmitDate**: 2025-01-30    [abs](http://arxiv.org/abs/2501.18536v1) [paper-pdf](http://arxiv.org/pdf/2501.18536v1)

**Authors**: Manveer Singh Tamber, Jimmy Lin

**Abstract**: Consider a scenario in which a user searches for information, only to encounter texts flooded with misleading or non-relevant content. This scenario exemplifies a simple yet potent vulnerability in neural Information Retrieval (IR) pipelines: content injection attacks. We find that embedding models for retrieval, rerankers, and large language model (LLM) relevance judges are vulnerable to these attacks, in which adversaries insert misleading text into passages to manipulate model judgements. We identify two primary threats: (1) inserting unrelated or harmful content within passages that still appear deceptively "relevant", and (2) inserting entire queries or key query terms into passages to boost their perceived relevance. While the second tactic has been explored in prior research, we present, to our knowledge, the first empirical analysis of the first threat, demonstrating how state-of-the-art models can be easily misled. Our study systematically examines the factors that influence an attack's success, such as the placement of injected content and the balance between relevant and non-relevant material. Additionally, we explore various defense strategies, including adversarial passage classifiers, retriever fine-tuning to discount manipulated content, and prompting LLM judges to adopt a more cautious approach. However, we find that these countermeasures often involve trade-offs, sacrificing effectiveness for attack robustness and sometimes penalizing legitimate documents in the process. Our findings highlight the need for stronger defenses against these evolving adversarial strategies to maintain the trustworthiness of IR systems. We release our code and scripts to facilitate further research.

摘要: 考虑这样一种场景：用户搜索信息，结果却发现文本中充斥着误导性或不相关的内容。这个场景例证了神经信息检索(IR)管道中一个简单但潜在的漏洞：内容注入攻击。我们发现，用于检索、重排和大型语言模型(LLM)相关性判断的嵌入模型容易受到这些攻击，即攻击者将误导性文本插入段落中以操纵模型判断。我们确定了两个主要威胁：(1)在看起来仍具有欺骗性的“相关”的段落中插入无关或有害的内容，以及(2)将整个查询或关键查询词插入到段落中，以提高其感知的相关性。虽然之前的研究已经探索了第二种策略，但据我们所知，我们首次对第一种威胁进行了实证分析，展示了最先进的模型如何容易被误导。我们的研究系统地检查了影响攻击成功的因素，如注入内容的放置以及相关和非相关材料之间的平衡。此外，我们探索了各种防御策略，包括对抗性段落分类器、检索器微调以对操纵的内容进行折扣，以及促使LLM评委采用更谨慎的方法。然而，我们发现，这些对策往往涉及权衡，为了攻击的稳健性而牺牲有效性，有时还会在这个过程中惩罚合法的文档。我们的发现突出表明，需要对这些不断演变的对抗性策略进行更强有力的防御，以保持IR系统的可信性。我们发布了我们的代码和脚本，以促进进一步的研究。



## **45. Rethinking Bottlenecks in Safety Fine-Tuning of Vision Language Models**

重新思考视觉语言模型安全微调中的瓶颈 cs.CV

**SubmitDate**: 2025-01-30    [abs](http://arxiv.org/abs/2501.18533v1) [paper-pdf](http://arxiv.org/pdf/2501.18533v1)

**Authors**: Yi Ding, Lijun Li, Bing Cao, Jing Shao

**Abstract**: Large Vision-Language Models (VLMs) have achieved remarkable performance across a wide range of tasks. However, their deployment in safety-critical domains poses significant challenges. Existing safety fine-tuning methods, which focus on textual or multimodal content, fall short in addressing challenging cases or disrupt the balance between helpfulness and harmlessness. Our evaluation highlights a safety reasoning gap: these methods lack safety visual reasoning ability, leading to such bottlenecks. To address this limitation and enhance both visual perception and reasoning in safety-critical contexts, we propose a novel dataset that integrates multi-image inputs with safety Chain-of-Thought (CoT) labels as fine-grained reasoning logic to improve model performance. Specifically, we introduce the Multi-Image Safety (MIS) dataset, an instruction-following dataset tailored for multi-image safety scenarios, consisting of training and test splits. Our experiments demonstrate that fine-tuning InternVL2.5-8B with MIS significantly outperforms both powerful open-source models and API-based models in challenging multi-image tasks requiring safety-related visual reasoning. This approach not only delivers exceptional safety performance but also preserves general capabilities without any trade-offs. Specifically, fine-tuning with MIS increases average accuracy by 0.83% across five general benchmarks and reduces the Attack Success Rate (ASR) on multiple safety benchmarks by a large margin. Data and Models are released under: \href{https://dripnowhy.github.io/MIS/}{\texttt{https://dripnowhy.github.io/MIS/}}

摘要: 大型视觉语言模型(VLM)在广泛的任务中取得了显著的性能。然而，它们在安全关键领域的部署构成了巨大的挑战。现有的安全微调方法侧重于文本或多模式内容，在处理具有挑战性的案件方面做得不够，或者破坏了有益和无害之间的平衡。我们的评估突出了一个安全推理缺口：这些方法缺乏安全视觉推理能力，导致了这样的瓶颈。为了解决这一局限性，并在安全关键环境中增强视觉感知和推理能力，我们提出了一种新的数据集，该数据集将多幅图像输入与安全思想链(COT)标签相结合作为细粒度推理逻辑来提高模型的性能。具体地说，我们介绍了多图像安全(MIS)数据集，这是一个为多图像安全场景量身定做的遵循说明的数据集，包括训练和测试拆分。我们的实验表明，带有管理信息系统的微调InternVL2.5-8B在挑战需要与安全相关的视觉推理的多图像任务时，显著优于强大的开源模型和基于API的模型。这种方法不仅提供了卓越的安全性能，而且在不进行任何权衡的情况下保留了一般功能。具体地说，管理信息系统的微调使五个通用基准测试的平均准确率提高了0.83%，并大幅降低了多个安全基准测试的攻击成功率(ASR)。数据和模型在以下位置发布：\href{https://dripnowhy.github.io/MIS/}{\texttt{https://dripnowhy.github.io/MIS/}}



## **46. Differentially Private Steering for Large Language Model Alignment**

针对大型语言模型对齐的差异私人指导 cs.CL

ICLR 2025; Code: https://github.com/UKPLab/iclr2025-psa

**SubmitDate**: 2025-01-30    [abs](http://arxiv.org/abs/2501.18532v1) [paper-pdf](http://arxiv.org/pdf/2501.18532v1)

**Authors**: Anmol Goel, Yaxi Hu, Iryna Gurevych, Amartya Sanyal

**Abstract**: Aligning Large Language Models (LLMs) with human values and away from undesirable behaviors (such as hallucination) has become increasingly important. Recently, steering LLMs towards a desired behavior via activation editing has emerged as an effective method to mitigate harmful generations at inference-time. Activation editing modifies LLM representations by preserving information from positive demonstrations (e.g., truthful) and minimising information from negative demonstrations (e.g., hallucinations). When these demonstrations come from a private dataset, the aligned LLM may leak private information contained in those private samples. In this work, we present the first study of aligning LLM behavior with private datasets. Our work proposes the \textit{\underline{P}rivate \underline{S}teering for LLM \underline{A}lignment (PSA)} algorithm to edit LLM activations with differential privacy (DP) guarantees. We conduct extensive experiments on seven different benchmarks with open-source LLMs of different sizes (0.5B to 7B) and model families (LlaMa, Qwen, Mistral and Gemma). Our results show that PSA achieves DP guarantees for LLM alignment with minimal loss in performance, including alignment metrics, open-ended text generation quality, and general-purpose reasoning. We also develop the first Membership Inference Attack (MIA) for evaluating and auditing the empirical privacy for the problem of LLM steering via activation editing. Our attack is tailored for activation editing and relies solely on the generated texts without their associated probabilities. Our experiments support the theoretical guarantees by showing improved guarantees for our \textit{PSA} algorithm compared to several existing non-private techniques.

摘要: 使大型语言模型(LLM)与人类价值观保持一致，并远离不良行为(如幻觉)已变得越来越重要。最近，通过激活编辑将LLM引导到期望的行为已经成为减少推理时有害生成的一种有效方法。激活编辑通过保留来自正面演示(例如，真实)的信息以及最小化来自负面演示(例如，幻觉)的信息来修改LLM表示。当这些演示来自私有数据集时，对齐的LLM可能会泄露那些私有样本中包含的私有信息。在这项工作中，我们提出了第一个将LLM行为与私有数据集对齐的研究。我们的工作提出了带差分隐私(DP)保证的LLm\Underline{A}对齐(PSA)算法。我们使用不同大小(0.5B到7B)和模型家族(骆驼、Qwen、米斯特拉尔和Gema)的开源LLM在七个不同的基准上进行了广泛的实验。我们的结果表明，PSA在性能损失最小的情况下实现了LLM对齐的DP保证，包括对齐度量、开放式文本生成质量和通用推理。我们还开发了第一个成员推理攻击(MIA)，用于通过激活编辑来评估和审计LLM操控问题的经验隐私。我们的攻击是为激活编辑量身定做的，只依赖于生成的文本，而没有关联的概率。我们的实验表明，与现有的几种非私有技术相比，我们的\textit{PSA}算法的保证得到了改进，从而支持了理论上的保证。



## **47. xJailbreak: Representation Space Guided Reinforcement Learning for Interpretable LLM Jailbreaking**

x越狱：可解释LLM越狱的表示空间引导强化学习 cs.CL

**SubmitDate**: 2025-01-30    [abs](http://arxiv.org/abs/2501.16727v2) [paper-pdf](http://arxiv.org/pdf/2501.16727v2)

**Authors**: Sunbowen Lee, Shiwen Ni, Chi Wei, Shuaimin Li, Liyang Fan, Ahmadreza Argha, Hamid Alinejad-Rokny, Ruifeng Xu, Yicheng Gong, Min Yang

**Abstract**: Safety alignment mechanism are essential for preventing large language models (LLMs) from generating harmful information or unethical content. However, cleverly crafted prompts can bypass these safety measures without accessing the model's internal parameters, a phenomenon known as black-box jailbreak. Existing heuristic black-box attack methods, such as genetic algorithms, suffer from limited effectiveness due to their inherent randomness, while recent reinforcement learning (RL) based methods often lack robust and informative reward signals. To address these challenges, we propose a novel black-box jailbreak method leveraging RL, which optimizes prompt generation by analyzing the embedding proximity between benign and malicious prompts. This approach ensures that the rewritten prompts closely align with the intent of the original prompts while enhancing the attack's effectiveness. Furthermore, we introduce a comprehensive jailbreak evaluation framework incorporating keywords, intent matching, and answer validation to provide a more rigorous and holistic assessment of jailbreak success. Experimental results show the superiority of our approach, achieving state-of-the-art (SOTA) performance on several prominent open and closed-source LLMs, including Qwen2.5-7B-Instruct, Llama3.1-8B-Instruct, and GPT-4o-0806. Our method sets a new benchmark in jailbreak attack effectiveness, highlighting potential vulnerabilities in LLMs. The codebase for this work is available at https://github.com/Aegis1863/xJailbreak.

摘要: 安全对齐机制对于防止大型语言模型(LLM)生成有害信息或不道德内容至关重要。然而，精心设计的提示可以绕过这些安全措施，而不需要访问模型的内部参数，这一现象被称为黑盒越狱。现有的启发式黑盒攻击方法，如遗传算法，由于其固有的随机性，其有效性有限，而最近的基于强化学习(RL)的方法往往缺乏健壮和信息丰富的奖励信号。为了应对这些挑战，我们提出了一种新的利用RL的黑盒越狱方法，该方法通过分析良性提示和恶意提示之间的嵌入邻近性来优化提示生成。这种方法确保重写的提示与原始提示的意图紧密一致，同时提高了攻击的有效性。此外，我们引入了一个全面的越狱评估框架，其中包括关键字、意图匹配和答案验证，以提供更严格和全面的越狱成功评估。实验结果表明了该方法的优越性，在Qwen2.5-7B-Direct、Llama3.1-8B-Direct和GPT-40-0806等几个著名的开源和闭源LLM上获得了最先进的性能(SOTA)。我们的方法在越狱攻击有效性方面设置了一个新的基准，突出了LLMS中的潜在漏洞。这项工作的代码库可在https://github.com/Aegis1863/xJailbreak.上获得



## **48. Exploring Potential Prompt Injection Attacks in Federated Military LLMs and Their Mitigation**

探索联邦军事LLM中潜在的即时注入攻击及其缓解措施 cs.LG

7 pages

**SubmitDate**: 2025-01-30    [abs](http://arxiv.org/abs/2501.18416v1) [paper-pdf](http://arxiv.org/pdf/2501.18416v1)

**Authors**: Youngjoon Lee, Taehyun Park, Yunho Lee, Jinu Gong, Joonhyuk Kang

**Abstract**: Federated Learning (FL) is increasingly being adopted in military collaborations to develop Large Language Models (LLMs) while preserving data sovereignty. However, prompt injection attacks-malicious manipulations of input prompts-pose new threats that may undermine operational security, disrupt decision-making, and erode trust among allies. This perspective paper highlights four potential vulnerabilities in federated military LLMs: secret data leakage, free-rider exploitation, system disruption, and misinformation spread. To address these potential risks, we propose a human-AI collaborative framework that introduces both technical and policy countermeasures. On the technical side, our framework uses red/blue team wargaming and quality assurance to detect and mitigate adversarial behaviors of shared LLM weights. On the policy side, it promotes joint AI-human policy development and verification of security protocols. Our findings will guide future research and emphasize proactive strategies for emerging military contexts.

摘要: 联邦学习(FL)正越来越多地被军事合作所采用，以开发大型语言模型(LLM)，同时保持数据主权。然而，即时注入攻击-对输入提示的恶意操纵-构成了新的威胁，可能会破坏操作安全、扰乱决策并侵蚀盟友之间的信任。本文重点介绍了联邦军用LLMS中的四个潜在漏洞：秘密数据泄露、搭便车攻击、系统中断和错误信息传播。为了应对这些潜在的风险，我们提出了一个人类-人工智能协作框架，引入了技术和政策对策。在技术方面，我们的框架使用红/蓝团队战争游戏和质量保证来检测和缓解共享LLM权重的敌对行为。在政策方面，它促进人工智能-人类联合政策开发和安全协议验证。我们的发现将指导未来的研究，并强调针对新兴军事背景的积极战略。



## **49. Joint Optimization of Prompt Security and System Performance in Edge-Cloud LLM Systems**

边缘云LLM系统中即时安全性和系统性能的联合优化 cs.CR

**SubmitDate**: 2025-01-30    [abs](http://arxiv.org/abs/2501.18663v1) [paper-pdf](http://arxiv.org/pdf/2501.18663v1)

**Authors**: Haiyang Huang, Tianhui Meng, Weijia Jia

**Abstract**: Large language models (LLMs) have significantly facilitated human life, and prompt engineering has improved the efficiency of these models. However, recent years have witnessed a rise in prompt engineering-empowered attacks, leading to issues such as privacy leaks, increased latency, and system resource wastage. Though safety fine-tuning based methods with Reinforcement Learning from Human Feedback (RLHF) are proposed to align the LLMs, existing security mechanisms fail to cope with fickle prompt attacks, highlighting the necessity of performing security detection on prompts. In this paper, we jointly consider prompt security, service latency, and system resource optimization in Edge-Cloud LLM (EC-LLM) systems under various prompt attacks. To enhance prompt security, a vector-database-enabled lightweight attack detector is proposed. We formalize the problem of joint prompt detection, latency, and resource optimization into a multi-stage dynamic Bayesian game model. The equilibrium strategy is determined by predicting the number of malicious tasks and updating beliefs at each stage through Bayesian updates. The proposed scheme is evaluated on a real implemented EC-LLM system, and the results demonstrate that our approach offers enhanced security, reduces the service latency for benign users, and decreases system resource consumption compared to state-of-the-art algorithms.

摘要: 大型语言模型极大地方便了人类的生活，快速工程提高了这些模型的效率。然而，近年来，由工程支持的即时攻击有所增加，导致隐私泄露、延迟增加和系统资源浪费等问题。虽然人们提出了基于安全微调的人类反馈强化学习(RLHF)方法来对齐LLMS，但现有的安全机制无法应对变化无常的提示攻击，这突显了对提示进行安全检测的必要性。本文综合考虑了边缘云LLM系统在各种即时攻击下的即时安全性、服务时延和系统资源优化问题。为提高系统的即时安全性，提出了一种基于向量数据库的轻量级攻击检测器。我们将联合提示检测、延迟和资源优化问题形式化地描述为一个多阶段动态贝叶斯博弈模型。均衡策略是通过预测恶意任务的数量并通过贝叶斯更新在每个阶段更新信念来确定的。在一个实际实现的EC-LLM系统上对所提出的方案进行了评估，结果表明，与现有算法相比，该方案提高了安全性，降低了良性用户的服务延迟，降低了系统资源消耗。



## **50. Jailbreaking LLMs' Safeguard with Universal Magic Words for Text Embedding Models**

越狱的LLM为文本嵌入模型提供通用魔法词的保障 cs.CL

**SubmitDate**: 2025-01-30    [abs](http://arxiv.org/abs/2501.18280v1) [paper-pdf](http://arxiv.org/pdf/2501.18280v1)

**Authors**: Haoyu Liang, Youran Sun, Yunfeng Cai, Jun Zhu, Bo Zhang

**Abstract**: The security issue of large language models (LLMs) has gained significant attention recently, with various defense mechanisms developed to prevent harmful outputs, among which safeguards based on text embedding models serve as a fundamental defense. Through testing, we discover that the distribution of text embedding model outputs is significantly biased with a large mean. Inspired by this observation, we propose novel efficient methods to search for universal magic words that can attack text embedding models. The universal magic words as suffixes can move the embedding of any text towards the bias direction, therefore manipulate the similarity of any text pair and mislead safeguards. By appending magic words to user prompts and requiring LLMs to end answers with magic words, attackers can jailbreak the safeguard. To eradicate this security risk, we also propose defense mechanisms against such attacks, which can correct the biased distribution of text embeddings in a train-free manner.

摘要: 大型语言模型（LLM）的安全问题最近受到了广泛关注，各种防御机制被开发出来来防止有害输出，其中基于文本嵌入模型的保护措施是基本防御。通过测试，我们发现文本嵌入模型输出的分布存在显着偏差，平均值很大。受这一观察的启发，我们提出了新颖的有效方法来搜索可以攻击文本嵌入模型的通用魔法词。作为后缀的通用神奇词可以将任何文本的嵌入移向偏向方向，从而操纵任何文本对的相似性并误导保障措施。通过在用户提示中添加魔法词并要求LLM以魔法词结束回答，攻击者可以越狱该保护措施。为了消除这种安全风险，我们还提出了针对此类攻击的防御机制，该机制可以以无训练的方式纠正文本嵌入的偏见分布。



