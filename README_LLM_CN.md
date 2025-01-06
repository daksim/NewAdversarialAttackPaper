# Latest Large Language Model Attack Papers
**update at 2025-01-06 09:57:42**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. CySecBench: Generative AI-based CyberSecurity-focused Prompt Dataset for Benchmarking Large Language Models**

CySecBench：基于人工智能、以网络安全为重点的生成性提示数据集，用于对大型语言模型进行基准测试 cs.CR

**SubmitDate**: 2025-01-02    [abs](http://arxiv.org/abs/2501.01335v1) [paper-pdf](http://arxiv.org/pdf/2501.01335v1)

**Authors**: Johan Wahréus, Ahmed Mohamed Hussain, Panos Papadimitratos

**Abstract**: Numerous studies have investigated methods for jailbreaking Large Language Models (LLMs) to generate harmful content. Typically, these methods are evaluated using datasets of malicious prompts designed to bypass security policies established by LLM providers. However, the generally broad scope and open-ended nature of existing datasets can complicate the assessment of jailbreaking effectiveness, particularly in specific domains, notably cybersecurity. To address this issue, we present and publicly release CySecBench, a comprehensive dataset containing 12662 prompts specifically designed to evaluate jailbreaking techniques in the cybersecurity domain. The dataset is organized into 10 distinct attack-type categories, featuring close-ended prompts to enable a more consistent and accurate assessment of jailbreaking attempts. Furthermore, we detail our methodology for dataset generation and filtration, which can be adapted to create similar datasets in other domains. To demonstrate the utility of CySecBench, we propose and evaluate a jailbreaking approach based on prompt obfuscation. Our experimental results show that this method successfully elicits harmful content from commercial black-box LLMs, achieving Success Rates (SRs) of 65% with ChatGPT and 88% with Gemini; in contrast, Claude demonstrated greater resilience with a jailbreaking SR of 17%. Compared to existing benchmark approaches, our method shows superior performance, highlighting the value of domain-specific evaluation datasets for assessing LLM security measures. Moreover, when evaluated using prompts from a widely used dataset (i.e., AdvBench), it achieved an SR of 78.5%, higher than the state-of-the-art methods.

摘要: 许多研究已经调查了越狱大型语言模型(LLM)生成有害内容的方法。通常，使用恶意提示的数据集来评估这些方法，这些恶意提示旨在绕过LLM提供商建立的安全策略。然而，现有数据集的广泛范围和开放式性质可能会使越狱效果的评估复杂化，特别是在特定领域，特别是网络安全领域。为了解决这个问题，我们提出并公开发布了CySecBitch，这是一个全面的数据集，包含12662个提示，专门用于评估网络安全领域的越狱技术。该数据集被组织成10个不同的攻击类型类别，具有封闭式提示，以实现对越狱企图的更一致和更准确的评估。此外，我们详细介绍了我们的数据集生成和过滤方法，该方法可以适用于在其他领域创建类似的数据集。为了证明CySecBitch的有效性，我们提出并评估了一种基于即时混淆的越狱方法。我们的实验结果表明，该方法成功地从商业黑盒LLMS中提取出有害内容，ChatGPT的成功率(SRS)为65%，Gemini为88%；相比之下，Claude表现出更强的弹性，越狱成功率为17%。与现有的基准测试方法相比，我们的方法表现出更好的性能，突出了特定于域的评估数据集在评估LLM安全措施方面的价值。此外，当使用广泛使用的数据集(即AdvBch)的提示进行评估时，它获得了78.5%的SR，高于最先进的方法。



## **2. Security Attacks on LLM-based Code Completion Tools**

对基于LLM的代码完成工具的安全攻击 cs.CL

Paper accepted at AAAI 2025

**SubmitDate**: 2025-01-02    [abs](http://arxiv.org/abs/2408.11006v4) [paper-pdf](http://arxiv.org/pdf/2408.11006v4)

**Authors**: Wen Cheng, Ke Sun, Xinyu Zhang, Wei Wang

**Abstract**: The rapid development of large language models (LLMs) has significantly advanced code completion capabilities, giving rise to a new generation of LLM-based Code Completion Tools (LCCTs). Unlike general-purpose LLMs, these tools possess unique workflows, integrating multiple information sources as input and prioritizing code suggestions over natural language interaction, which introduces distinct security challenges. Additionally, LCCTs often rely on proprietary code datasets for training, raising concerns about the potential exposure of sensitive data. This paper exploits these distinct characteristics of LCCTs to develop targeted attack methodologies on two critical security risks: jailbreaking and training data extraction attacks. Our experimental results expose significant vulnerabilities within LCCTs, including a 99.4% success rate in jailbreaking attacks on GitHub Copilot and a 46.3% success rate on Amazon Q. Furthermore, We successfully extracted sensitive user data from GitHub Copilot, including 54 real email addresses and 314 physical addresses associated with GitHub usernames. Our study also demonstrates that these code-based attack methods are effective against general-purpose LLMs, such as the GPT series, highlighting a broader security misalignment in the handling of code by modern LLMs. These findings underscore critical security challenges associated with LCCTs and suggest essential directions for strengthening their security frameworks. The example code and attack samples from our research are provided at https://github.com/Sensente/Security-Attacks-on-LCCTs.

摘要: 大型语言模型(LLM)的快速发展极大地提升了代码补全能力，催生了新一代基于LLM的代码补全工具(LCCT)。与通用的LLMS不同，这些工具拥有独特的工作流，将多个信息源集成为输入，并优先考虑代码建议而不是自然语言交互，这带来了明显的安全挑战。此外，LCCT经常依赖专有代码数据集进行培训，这引发了人们对敏感数据潜在暴露的担忧。针对越狱攻击和训练数据提取攻击这两个关键安全风险，本文利用LCCT的这些显著特点，提出了针对性的攻击方法。我们的实验结果暴露了LCCT中的重大漏洞，包括对GitHub Copilot的越狱攻击成功率为99.4%，对Amazon Q的成功率为46.3%。此外，我们成功地从GitHub Copilot中提取了敏感用户数据，包括与GitHub用户名关联的54个真实电子邮件地址和314个物理地址。我们的研究还表明，这些基于代码的攻击方法对通用LLM是有效的，例如GPT系列，突显了现代LLM在处理代码时存在更广泛的安全错位。这些调查结果强调了与土地利用、土地利用、土地退化和土地退化有关的重大安全挑战，并提出了加强其安全框架的基本方向。我们的研究提供了示例代码和攻击示例，请访问https://github.com/Sensente/Security-Attacks-on-LCCTs.



## **3. Image-based Multimodal Models as Intruders: Transferable Multimodal Attacks on Video-based MLLMs**

作为入侵者的基于图像的多模式模型：对基于视频的MLLM的可转移多模式攻击 cs.CV

**SubmitDate**: 2025-01-02    [abs](http://arxiv.org/abs/2501.01042v1) [paper-pdf](http://arxiv.org/pdf/2501.01042v1)

**Authors**: Linhao Huang, Xue Jiang, Zhiqiang Wang, Wentao Mo, Xi Xiao, Bo Han, Yongjie Yin, Feng Zheng

**Abstract**: Video-based multimodal large language models (V-MLLMs) have shown vulnerability to adversarial examples in video-text multimodal tasks. However, the transferability of adversarial videos to unseen models--a common and practical real world scenario--remains unexplored. In this paper, we pioneer an investigation into the transferability of adversarial video samples across V-MLLMs. We find that existing adversarial attack methods face significant limitations when applied in black-box settings for V-MLLMs, which we attribute to the following shortcomings: (1) lacking generalization in perturbing video features, (2) focusing only on sparse key-frames, and (3) failing to integrate multimodal information. To address these limitations and deepen the understanding of V-MLLM vulnerabilities in black-box scenarios, we introduce the Image-to-Video MLLM (I2V-MLLM) attack. In I2V-MLLM, we utilize an image-based multimodal model (IMM) as a surrogate model to craft adversarial video samples. Multimodal interactions and temporal information are integrated to disrupt video representations within the latent space, improving adversarial transferability. In addition, a perturbation propagation technique is introduced to handle different unknown frame sampling strategies. Experimental results demonstrate that our method can generate adversarial examples that exhibit strong transferability across different V-MLLMs on multiple video-text multimodal tasks. Compared to white-box attacks on these models, our black-box attacks (using BLIP-2 as surrogate model) achieve competitive performance, with average attack success rates of 55.48% on MSVD-QA and 58.26% on MSRVTT-QA for VideoQA tasks, respectively. Our code will be released upon acceptance.

摘要: 基于视频的多通道大语言模型(V-MLLM)在视频-文本多通道任务中表现出对敌意例子的脆弱性。然而，对抗性视频是否可以转移到看不见的模型上--这是现实世界中常见和实用的场景--仍未得到探索。在本文中，我们率先对对抗性视频样本在V-MLLMS上的可转移性进行了研究。我们发现，现有的对抗性攻击方法在应用于V-MLLMS的黑盒环境时面临着很大的局限性，我们将其归因于以下缺点：(1)对扰动视频特征缺乏泛化；(2)只关注稀疏关键帧；(3)未能整合多模信息。为了解决这些限制并加深对黑盒场景中V-MLLM漏洞的理解，我们引入了图像到视频MLLM(I2V-MLLM)攻击。在I2V-MLLM中，我们使用基于图像的多模式模型(IMM)作为代理模型来制作对抗性视频样本。多模式交互和时间信息被集成以扰乱潜在空间内的视频表示，提高了对抗性转移。此外，还引入了扰动传播技术来处理不同的未知帧采样策略。实验结果表明，该方法能够在多个视频-文本多模式任务的不同V-MLLMS之间生成具有较强可转移性的对抗性实例。与这些模型上的白盒攻击相比，我们的黑盒攻击(以BLIP-2为代理模型)取得了与之相当的性能，对于视频QA任务，MSVD-QA和MSRVTT-QA的平均攻击成功率分别为55.48%和58.26%。我们的代码将在接受后发布。



## **4. TrustRAG: Enhancing Robustness and Trustworthiness in RAG**

TrustRAG：增强RAG的稳健性和可信性 cs.CL

**SubmitDate**: 2025-01-01    [abs](http://arxiv.org/abs/2501.00879v1) [paper-pdf](http://arxiv.org/pdf/2501.00879v1)

**Authors**: Huichi Zhou, Kin-Hei Lee, Zhonghao Zhan, Yue Chen, Zhenhao Li

**Abstract**: Retrieval-Augmented Generation (RAG) systems enhance large language models (LLMs) by integrating external knowledge sources, enabling more accurate and contextually relevant responses tailored to user queries. However, these systems remain vulnerable to corpus poisoning attacks that can significantly degrade LLM performance through the injection of malicious content. To address these challenges, we propose TrustRAG, a robust framework that systematically filters compromised and irrelevant content before it reaches the language model. Our approach implements a two-stage defense mechanism: first, it employs K-means clustering to identify potential attack patterns in retrieved documents based on their semantic embeddings, effectively isolating suspicious content. Second, it leverages cosine similarity and ROUGE metrics to detect malicious documents while resolving discrepancies between the model's internal knowledge and external information through a self-assessment process. TrustRAG functions as a plug-and-play, training-free module that integrates seamlessly with any language model, whether open or closed-source, maintaining high contextual relevance while strengthening defenses against attacks. Through extensive experimental validation, we demonstrate that TrustRAG delivers substantial improvements in retrieval accuracy, efficiency, and attack resistance compared to existing approaches across multiple model architectures and datasets. We have made TrustRAG available as open-source software at \url{https://github.com/HuichiZhou/TrustRAG}.

摘要: 检索-增强生成(RAG)系统通过集成外部知识源来增强大型语言模型(LLM)，从而能够针对用户查询定制更准确且与上下文相关的响应。然而，这些系统仍然容易受到语料库中毒攻击，这些攻击可能会通过注入恶意内容显著降低LLM的性能。为了应对这些挑战，我们提出了TrustRAG，这是一个健壮的框架，在受到攻击和无关的内容到达语言模型之前系统地过滤它们。该方法实现了一种两阶段防御机制：首先，利用K-均值聚类，根据文档的语义嵌入来识别潜在的攻击模式，有效地隔离可疑内容。其次，它利用余弦相似度和Rouge度量来检测恶意文档，同时通过自我评估过程解决模型内部知识和外部信息之间的差异。TrustRAG是一个即插即用、无需培训的模块，可以与任何语言模型无缝集成，无论是开放还是封闭源代码，在加强对攻击的防御的同时保持高度的上下文相关性。通过广泛的实验验证，我们证明了TrustRAG在检索准确性、效率和抗攻击方面比现有的跨多个模型架构和数据集的方法有了实质性的改进。我们已将TrustRAG作为开源软件提供给\url{https://github.com/HuichiZhou/TrustRAG}.



## **5. Dynamics of Adversarial Attacks on Large Language Model-Based Search Engines**

基于大型语言模型的搜索引擎的对抗性攻击动态 cs.CL

**SubmitDate**: 2025-01-01    [abs](http://arxiv.org/abs/2501.00745v1) [paper-pdf](http://arxiv.org/pdf/2501.00745v1)

**Authors**: Xiyang Hu

**Abstract**: The increasing integration of Large Language Model (LLM) based search engines has transformed the landscape of information retrieval. However, these systems are vulnerable to adversarial attacks, especially ranking manipulation attacks, where attackers craft webpage content to manipulate the LLM's ranking and promote specific content, gaining an unfair advantage over competitors. In this paper, we study the dynamics of ranking manipulation attacks. We frame this problem as an Infinitely Repeated Prisoners' Dilemma, where multiple players strategically decide whether to cooperate or attack. We analyze the conditions under which cooperation can be sustained, identifying key factors such as attack costs, discount rates, attack success rates, and trigger strategies that influence player behavior. We identify tipping points in the system dynamics, demonstrating that cooperation is more likely to be sustained when players are forward-looking. However, from a defense perspective, we find that simply reducing attack success probabilities can, paradoxically, incentivize attacks under certain conditions. Furthermore, defensive measures to cap the upper bound of attack success rates may prove futile in some scenarios. These insights highlight the complexity of securing LLM-based systems. Our work provides a theoretical foundation and practical insights for understanding and mitigating their vulnerabilities, while emphasizing the importance of adaptive security strategies and thoughtful ecosystem design.

摘要: 基于大型语言模型(LLM)的搜索引擎的日益集成已经改变了信息检索的格局。然而，这些系统容易受到对抗性攻击，特别是排名操纵攻击，攻击者精心编制网页内容来操纵LLM的排名并推广特定内容，从而获得相对于竞争对手的不公平优势。本文研究了排名操纵攻击的动态特性。我们将这个问题描述为一个无限重复的囚徒困境，其中多个参与者战略性地决定是合作还是攻击。我们分析了合作能够持续的条件，确定了影响玩家行为的关键因素，如攻击成本、折扣率、攻击成功率和触发策略。我们确定了系统动态中的引爆点，表明当参与者具有前瞻性时，合作更有可能持续下去。然而，从防御的角度来看，我们发现，矛盾的是，仅仅降低攻击成功的概率就可以在某些条件下激励攻击。此外，在某些情况下，为攻击成功率上限设定上限的防御措施可能被证明是徒劳的。这些见解突显了保护基于LLM的系统的复杂性。我们的工作为理解和缓解它们的漏洞提供了理论基础和实践见解，同时强调了自适应安全策略和深思熟虑的生态系统设计的重要性。



## **6. From Sands to Mansions: Simulating Full Attack Chain with LLM-Organized Knowledge**

从金沙到豪宅：利用法学硕士组织的知识模拟完整攻击链 cs.CR

**SubmitDate**: 2024-12-31    [abs](http://arxiv.org/abs/2407.16928v2) [paper-pdf](http://arxiv.org/pdf/2407.16928v2)

**Authors**: Lingzhi Wang, Zhenyuan Li, Zonghan Guo, Yi Jiang, Kyle Jung, Kedar Thiagarajan, Jiahui Wang, Zhengkai Wang, Emily Wei, Xiangmin Shen, Yan Chen

**Abstract**: Adversarial dynamics are intrinsic to the nature of offense and defense in cyberspace, with both attackers and defenders continuously evolving their technologies. Given the wide array of security products available, users often face challenges in selecting the most effective solutions. Furthermore, traditional benchmarks based on single-point attacks are increasingly inadequate, failing to accurately reflect the full range of attacker capabilities and falling short in properly evaluating the effectiveness of defense products. Automated multi-stage attack simulations offer a promising approach to enhance system evaluation efficiency and aid in analyzing the effectiveness of detection systems. However, simulating a full attack chain is complex and requires significant time and expertise from security professionals, facing several challenges, including limited coverage of attack techniques, a high level of required expertise, and a lack of execution detail. In this paper, we model automatic attack simulation as a planning problem. By using the Planning Domain Definition Language (PDDL) to formally describe the attack simulation problem, and combining domain knowledge of both the problem and the domain space, we enable the planning of attack paths through standardized, domain-independent planning algorithms. We explore the potential of Large Language Models (LLMs) to summarize and analyze knowledge from existing attack documentation and reports, facilitating automated attack planning. We introduce Aurora, a system that autonomously simulates full attack chains based on external attack tools and threat intelligence reports.

摘要: 对抗动态是网络空间进攻和防御的本质所固有的，攻击者和防御者都在不断地发展他们的技术。鉴于可用的安全产品种类繁多，用户在选择最有效的解决方案时经常面临挑战。此外，基于单点攻击的传统基准日益不足，无法准确反映攻击者的全方位能力，无法正确评估防御产品的有效性。自动多阶段攻击模拟为提高系统评估效率和辅助分析检测系统的有效性提供了一种很有前途的方法。然而，模拟完整的攻击链是复杂的，需要大量的时间和安全专业人员的专业知识，面临着几个挑战，包括攻击技术的覆盖范围有限，所需专业知识水平较高，以及缺乏执行细节。在本文中，我们将自动攻击模拟建模为一个规划问题。通过使用规划领域定义语言(PDDL)对攻击模拟问题进行形式化描述，并结合问题和领域空间的领域知识，我们能够通过标准化的、与领域无关的规划算法来规划攻击路径。我们探索大型语言模型(LLM)的潜力，以总结和分析现有攻击文档和报告中的知识，从而促进自动攻击规划。我们介绍了Aurora，这是一个基于外部攻击工具和威胁情报报告自主模拟完整攻击链的系统。



## **7. Toward Intelligent and Secure Cloud: Large Language Model Empowered Proactive Defense**

迈向智能和安全的云：大型语言模型增强主动防御 cs.CR

7 pages; In submission

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.21051v1) [paper-pdf](http://arxiv.org/pdf/2412.21051v1)

**Authors**: Yuyang Zhou, Guang Cheng, Kang Du, Zihan Chen

**Abstract**: The rapid evolution of cloud computing technologies and the increasing number of cloud applications have provided a large number of benefits in daily lives. However, the diversity and complexity of different components pose a significant challenge to cloud security, especially when dealing with sophisticated and advanced cyberattacks. Recent advancements in generative foundation models (GFMs), particularly in the large language models (LLMs), offer promising solutions for security intelligence. By exploiting the powerful abilities in language understanding, data analysis, task inference, action planning, and code generation, we present LLM-PD, a novel proactive defense architecture that defeats various threats in a proactive manner. LLM-PD can efficiently make a decision through comprehensive data analysis and sequential reasoning, as well as dynamically creating and deploying actionable defense mechanisms on the target cloud. Furthermore, it can flexibly self-evolve based on experience learned from previous interactions and adapt to new attack scenarios without additional training. The experimental results demonstrate its remarkable ability in terms of defense effectiveness and efficiency, particularly highlighting an outstanding success rate when compared with other existing methods.

摘要: 云计算技术的快速发展和越来越多的云应用为日常生活提供了大量的好处。然而，不同组件的多样性和复杂性对云安全构成了重大挑战，尤其是在处理复杂和高级的网络攻击时。生成性基础模型(GFMS)，特别是大型语言模型(LLM)的最新进展，为安全智能提供了有前途的解决方案。通过利用LLM-PD在语言理解、数据分析、任务推理、动作规划和代码生成等方面的强大能力，我们提出了一种新型的主动防御体系结构LLM-PD，它能够主动地击败各种威胁。LLM-PD通过全面的数据分析和时序推理，以及在目标云上动态创建和部署可操作的防御机制，能够高效地做出决策。此外，它可以根据从以前交互中学习的经验灵活地自我进化，并适应新的攻击场景，而不需要额外的培训。实验结果表明，该方法在防御效果和效率方面具有显著的能力，特别是与现有的其他方法相比，具有突出的成功率。



## **8. Unsupervised dense retrieval with conterfactual contrastive learning**

具有反事实对比学习的无监督密集检索 cs.IR

arXiv admin note: text overlap with arXiv:2107.07773 by other authors

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.20756v1) [paper-pdf](http://arxiv.org/pdf/2412.20756v1)

**Authors**: Haitian Chen, Qingyao Ai, Xiao Wang, Yiqun Liu, Fen Lin, Qin Liu

**Abstract**: Efficiently retrieving a concise set of candidates from a large document corpus remains a pivotal challenge in Information Retrieval (IR). Neural retrieval models, particularly dense retrieval models built with transformers and pretrained language models, have been popular due to their superior performance. However, criticisms have also been raised on their lack of explainability and vulnerability to adversarial attacks. In response to these challenges, we propose to improve the robustness of dense retrieval models by enhancing their sensitivity of fine-graned relevance signals. A model achieving sensitivity in this context should exhibit high variances when documents' key passages determining their relevance to queries have been modified, while maintaining low variances for other changes in irrelevant passages. This sensitivity allows a dense retrieval model to produce robust results with respect to attacks that try to promote documents without actually increasing their relevance. It also makes it possible to analyze which part of a document is actually relevant to a query, and thus improve the explainability of the retrieval model. Motivated by causality and counterfactual analysis, we propose a series of counterfactual regularization methods based on game theory and unsupervised learning with counterfactual passages. Experiments show that, our method can extract key passages without reliance on the passage-level relevance annotations. Moreover, the regularized dense retrieval models exhibit heightened robustness against adversarial attacks, surpassing the state-of-the-art anti-attack methods.

摘要: 从大型文档语料库中高效地检索一组简明的候选对象仍然是信息检索(IR)中的一个关键挑战。神经检索模型，特别是用转换器构建的密集检索模型和预先训练的语言模型，由于其优越的性能而受到广泛的欢迎。然而，也有人批评说，它们缺乏可解释性，容易受到对手攻击。为了应对这些挑战，我们提出了通过提高密集检索模型对细粒度关联信号的敏感度来提高其稳健性。在这种情况下实现敏感性的模型应该在确定其与查询的相关性的文档的关键段落被修改时表现出高方差，同时保持对不相关段落中的其他变化的低方差。这种敏感度使得密集检索模型能够针对试图提升文档而不实际增加其相关性的攻击产生稳健的结果。它还可以分析文档的哪个部分实际上与查询相关，从而提高检索模型的可解释性。在因果关系和反事实分析的启发下，我们提出了一系列基于博弈论和带有反事实段落的无监督学习的反事实正则化方法。实验表明，我们的方法可以在不依赖于段落级关联标注的情况下提取关键段落。此外，正则化的密集检索模型表现出对对手攻击的高度稳健性，超过了最先进的反攻击方法。



## **9. SafeSynthDP: Leveraging Large Language Models for Privacy-Preserving Synthetic Data Generation Using Differential Privacy**

SafeSynthDP：利用大型语言模型使用差异隐私生成隐私保护合成数据 cs.LG

15 pages, 1 figure, 5 tables

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.20641v1) [paper-pdf](http://arxiv.org/pdf/2412.20641v1)

**Authors**: Md Mahadi Hasan Nahid, Sadid Bin Hasan

**Abstract**: Machine learning (ML) models frequently rely on training data that may include sensitive or personal information, raising substantial privacy concerns. Legislative frameworks such as the General Data Protection Regulation (GDPR) and the California Consumer Privacy Act (CCPA) have necessitated the development of strategies that preserve privacy while maintaining the utility of data. In this paper, we investigate the capability of Large Language Models (LLMs) to generate synthetic datasets integrated with Differential Privacy (DP) mechanisms, thereby enabling data-driven research and model training without direct exposure of sensitive information. Our approach incorporates DP-based noise injection methods, including Laplace and Gaussian distributions, into the data generation process. We then evaluate the utility of these DP-enhanced synthetic datasets by comparing the performance of ML models trained on them against models trained on the original data. To substantiate privacy guarantees, we assess the resilience of the generated synthetic data to membership inference attacks and related threats. The experimental results demonstrate that integrating DP within LLM-driven synthetic data generation offers a viable balance between privacy protection and data utility. This study provides a foundational methodology and insight into the privacy-preserving capabilities of LLMs, paving the way for compliant and effective ML research and applications.

摘要: 机器学习(ML)模型经常依赖于可能包括敏感或个人信息的训练数据，这引发了大量的隐私问题。《一般数据保护条例》(GDPR)和《加州消费者隐私法》(CCPA)等立法框架要求制定在保持数据效用的同时保护隐私的战略。在本文中，我们研究了大型语言模型(LLM)生成集成了差异隐私(DP)机制的合成数据集的能力，从而在不直接暴露敏感信息的情况下实现了数据驱动的研究和模型训练。我们的方法将基于DP的噪声注入方法，包括拉普拉斯分布和高斯分布，融入到数据生成过程中。然后，我们通过比较在这些数据集上训练的ML模型和在原始数据上训练的模型的性能来评估这些DP增强的合成数据集的实用性。为了证实隐私保证，我们评估了生成的合成数据对成员资格推断攻击和相关威胁的弹性。实验结果表明，在LLM驱动的合成数据生成中集成DP提供了隐私保护和数据效用之间的可行平衡。这项研究为LLMS的隐私保护能力提供了一种基本的方法和见解，为合规和有效的ML研究和应用铺平了道路。



## **10. HALLUCINOGEN: A Benchmark for Evaluating Object Hallucination in Large Visual-Language Models**

HALLUCINOogen：评估大型视觉语言模型中对象幻觉的基准 cs.CV

**SubmitDate**: 2024-12-29    [abs](http://arxiv.org/abs/2412.20622v1) [paper-pdf](http://arxiv.org/pdf/2412.20622v1)

**Authors**: Ashish Seth, Dinesh Manocha, Chirag Agarwal

**Abstract**: Large Vision-Language Models (LVLMs) have demonstrated remarkable performance in performing complex multimodal tasks. However, they are still plagued by object hallucination: the misidentification or misclassification of objects present in images. To this end, we propose HALLUCINOGEN, a novel visual question answering (VQA) object hallucination attack benchmark that utilizes diverse contextual reasoning prompts to evaluate object hallucination in state-of-the-art LVLMs. We design a series of contextual reasoning hallucination prompts to evaluate LVLMs' ability to accurately identify objects in a target image while asking them to perform diverse visual-language tasks such as identifying, locating or performing visual reasoning around specific objects. Further, we extend our benchmark to high-stakes medical applications and introduce MED-HALLUCINOGEN, hallucination attacks tailored to the biomedical domain, and evaluate the hallucination performance of LVLMs on medical images, a critical area where precision is crucial. Finally, we conduct extensive evaluations of eight LVLMs and two hallucination mitigation strategies across multiple datasets to show that current generic and medical LVLMs remain susceptible to hallucination attacks.

摘要: 大型视觉语言模型在执行复杂的多通道任务方面表现出了显著的性能。然而，他们仍然受到物体幻觉的困扰：对图像中存在的物体的错误识别或错误分类。为此，我们提出了一种新颖的视觉问答(VQA)物体幻觉攻击基准--幻觉剂，该基准利用不同的上下文推理提示来评估最新的LVLM中的物体幻觉。我们设计了一系列情境推理幻觉提示来评估LVLMS准确识别目标图像中对象的能力，同时要求他们执行不同的视觉语言任务，如识别、定位或围绕特定对象执行视觉推理。此外，我们将我们的基准扩展到高风险的医疗应用，并引入了MED致幻剂，这是为生物医学领域量身定做的幻觉攻击，并评估了LVLMS在医学图像上的幻觉性能，这是一个对精度至关重要的关键领域。最后，我们在多个数据集上对八个LVLM和两个幻觉缓解策略进行了广泛的评估，以表明当前的普通LVLM和医用LVLM仍然容易受到幻觉攻击。



## **11. Pirates of the RAG: Adaptively Attacking LLMs to Leak Knowledge Bases**

RAG海盗：适应性攻击LLM以泄露知识库 cs.AI

**SubmitDate**: 2024-12-29    [abs](http://arxiv.org/abs/2412.18295v2) [paper-pdf](http://arxiv.org/pdf/2412.18295v2)

**Authors**: Christian Di Maio, Cristian Cosci, Marco Maggini, Valentina Poggioni, Stefano Melacci

**Abstract**: The growing ubiquity of Retrieval-Augmented Generation (RAG) systems in several real-world services triggers severe concerns about their security. A RAG system improves the generative capabilities of a Large Language Models (LLM) by a retrieval mechanism which operates on a private knowledge base, whose unintended exposure could lead to severe consequences, including breaches of private and sensitive information. This paper presents a black-box attack to force a RAG system to leak its private knowledge base which, differently from existing approaches, is adaptive and automatic. A relevance-based mechanism and an attacker-side open-source LLM favor the generation of effective queries to leak most of the (hidden) knowledge base. Extensive experimentation proves the quality of the proposed algorithm in different RAG pipelines and domains, comparing to very recent related approaches, which turn out to be either not fully black-box, not adaptive, or not based on open-source models. The findings from our study remark the urgent need for more robust privacy safeguards in the design and deployment of RAG systems.

摘要: 检索增强生成(RAG)系统在几个现实世界的服务中日益普遍，这引发了人们对其安全性的严重担忧。RAG系统通过在私有知识库上运行的检索机制来提高大型语言模型(LLM)的生成能力，其意外暴露可能导致严重后果，包括隐私和敏感信息的泄露。本文提出了一种黑盒攻击，以迫使RAG系统泄漏其私有知识库，与现有方法不同，该方法是自适应的和自动的。基于相关性的机制和攻击者端的开源LLM有利于生成有效的查询来泄漏大部分(隐藏的)知识库。大量的实验证明了该算法在不同的RAG管道和域中的质量，与最近的相关方法相比，这些方法要么不是完全黑箱的，要么不是自适应的，要么不是基于开源模型的。我们的研究结果表明，在设计和部署RAG系统时，迫切需要更强大的隐私保护措施。



## **12. Can Watermarked LLMs be Identified by Users via Crafted Prompts?**

用户可以通过精心制作的脚本识别带水印的LLM吗？ cs.CR

30 pages, 5 figures, 11 tables

**SubmitDate**: 2024-12-28    [abs](http://arxiv.org/abs/2410.03168v2) [paper-pdf](http://arxiv.org/pdf/2410.03168v2)

**Authors**: Aiwei Liu, Sheng Guan, Yiming Liu, Leyi Pan, Yifei Zhang, Liancheng Fang, Lijie Wen, Philip S. Yu, Xuming Hu

**Abstract**: Text watermarking for Large Language Models (LLMs) has made significant progress in detecting LLM outputs and preventing misuse. Current watermarking techniques offer high detectability, minimal impact on text quality, and robustness to text editing. However, current researches lack investigation into the imperceptibility of watermarking techniques in LLM services. This is crucial as LLM providers may not want to disclose the presence of watermarks in real-world scenarios, as it could reduce user willingness to use the service and make watermarks more vulnerable to attacks. This work is the first to investigate the imperceptibility of watermarked LLMs. We design an identification algorithm called Water-Probe that detects watermarks through well-designed prompts to the LLM. Our key motivation is that current watermarked LLMs expose consistent biases under the same watermark key, resulting in similar differences across prompts under different watermark keys. Experiments show that almost all mainstream watermarking algorithms are easily identified with our well-designed prompts, while Water-Probe demonstrates a minimal false positive rate for non-watermarked LLMs. Finally, we propose that the key to enhancing the imperceptibility of watermarked LLMs is to increase the randomness of watermark key selection. Based on this, we introduce the Water-Bag strategy, which significantly improves watermark imperceptibility by merging multiple watermark keys.

摘要: 针对大语言模型的文本水印技术在检测大语言模型输出和防止误用方面取得了显著进展。目前的水印技术提供了高可检测性，对文本质量的影响最小，以及对文本编辑的稳健性。然而，目前的研究缺乏对LLM服务中水印技术不可见性的研究。这一点至关重要，因为LLM提供商可能不想透露真实场景中是否存在水印，因为这可能会降低用户使用该服务的意愿，并使水印更容易受到攻击。这项工作是首次研究带水印的LLM的不可感知性。我们设计了一种名为Water-Probe的识别算法，该算法通过对LLM的精心设计的提示来检测水印。我们的关键动机是，当前的水印LLM暴露了相同水印密钥下的一致偏差，导致不同水印密钥下的提示存在相似的差异。实验表明，几乎所有的主流水印算法都能在我们精心设计的提示下很容易地识别出来，而Water-Probe算法对未加水印的LLMS具有最低的误检率。最后，提出了提高水印LLMS不可见性的关键是增加水印密钥选择的随机性。在此基础上，引入了水袋策略，通过合并多个水印密钥，显著提高了水印的不可见性。



## **13. Defending Against Network Attacks for Secure AI Agent Migration in Vehicular Metaverses**

防御网络攻击，实现车载元宇宙中的安全AI代理迁移 cs.NI

**SubmitDate**: 2024-12-28    [abs](http://arxiv.org/abs/2412.20154v1) [paper-pdf](http://arxiv.org/pdf/2412.20154v1)

**Authors**: Xinru Wen, Jinbo Wen, Ming Xiao, Jiawen Kang, Tao Zhang, Xiaohuan Li, Chuanxi Chen, Dusit Niyato

**Abstract**: Vehicular metaverses, blending traditional vehicular networks with metaverse technology, are expected to revolutionize fields such as autonomous driving. As virtual intelligent assistants in vehicular metaverses, Artificial Intelligence (AI) agents powered by large language models can create immersive 3D virtual spaces for passengers to enjoy on-broad vehicular applications and services. To provide users with seamless and engaging virtual interactions, resource-limited vehicles offload AI agents to RoadSide Units (RSUs) with adequate communication and computational capabilities. Due to the mobility of vehicles and the limited coverage of RSUs, AI agents need to migrate from one RSU to another RSU. However, potential network attacks pose significant challenges to ensuring reliable and efficient AI agent migration. In this paper, we first explore specific network attacks including traffic-based attacks (i.e., DDoS attacks) and infrastructure-based attacks (i.e., malicious RSU attacks). Then, we model the AI agent migration process as a Partially Observable Markov Decision Process (POMDP) and apply multi-agent proximal policy optimization algorithms to mitigate DDoS attacks. In addition, we propose a trust assessment mechanism to counter malicious RSU attacks. Numerical results validate that the proposed solutions effectively defend against these network attacks and reduce the total latency of AI agent migration by approximately 43.3%.

摘要: 车载虚拟现实将传统的车辆网络与虚拟现实技术相结合，预计将给自动驾驶等领域带来革命性的变化。作为车载虚拟现实中的虚拟智能助手，人工智能(AI)代理以大型语言模型为动力，可以创建身临其境的3D虚拟空间，供乘客享受广泛的车载应用和服务。为了向用户提供无缝且引人入胜的虚拟交互，资源有限的车辆将AI代理卸载到具有足够通信和计算能力的路边单元(RSU)。由于车辆的机动性和RSU的覆盖范围有限，AI代理需要从一个RSU迁移到另一个RSU。然而，潜在的网络攻击对确保可靠和高效的AI代理迁移构成了重大挑战。本文首先探讨了具体的网络攻击，包括基于流量的攻击(即DDoS攻击)和基于基础设施的攻击(即恶意RSU攻击)。然后，我们将AI代理迁移过程建模为部分可观测马尔可夫决策过程(POMDP)，并应用多代理邻近策略优化算法来缓解DDoS攻击。此外，我们还提出了一种信任评估机制来对抗恶意的RSU攻击。数值结果验证了所提出的解决方案有效地防御了这些网络攻击，并将AI代理迁移的总延迟降低了约43.3%。



## **14. On the Validity of Traditional Vulnerability Scoring Systems for Adversarial Attacks against LLMs**

传统漏洞评分系统对LLM对抗性攻击的有效性 cs.CR

101 pages, 3 figures

**SubmitDate**: 2024-12-28    [abs](http://arxiv.org/abs/2412.20087v1) [paper-pdf](http://arxiv.org/pdf/2412.20087v1)

**Authors**: Atmane Ayoub Mansour Bahar, Ahmad Samer Wazan

**Abstract**: This research investigates the effectiveness of established vulnerability metrics, such as the Common Vulnerability Scoring System (CVSS), in evaluating attacks against Large Language Models (LLMs), with a focus on Adversarial Attacks (AAs). The study explores the influence of both general and specific metric factors in determining vulnerability scores, providing new perspectives on potential enhancements to these metrics.   This study adopts a quantitative approach, calculating and comparing the coefficient of variation of vulnerability scores across 56 adversarial attacks on LLMs. The attacks, sourced from various research papers, and obtained through online databases, were evaluated using multiple vulnerability metrics. Scores were determined by averaging the values assessed by three distinct LLMs. The results indicate that existing scoring-systems yield vulnerability scores with minimal variation across different attacks, suggesting that many of the metric factors are inadequate for assessing adversarial attacks on LLMs. This is particularly true for context-specific factors or those with predefined value sets, such as those in CVSS. These findings support the hypothesis that current vulnerability metrics, especially those with rigid values, are limited in evaluating AAs on LLMs, highlighting the need for the development of more flexible, generalized metrics tailored to such attacks.   This research offers a fresh analysis of the effectiveness and applicability of established vulnerability metrics, particularly in the context of Adversarial Attacks on Large Language Models, both of which have gained significant attention in recent years. Through extensive testing and calculations, the study underscores the limitations of these metrics and opens up new avenues for improving and refining vulnerability assessment frameworks specifically tailored for LLMs.

摘要: 这项研究考察了通用漏洞评分系统(CVSS)等已建立的漏洞度量在评估针对大型语言模型(LLMS)的攻击时的有效性，重点是对抗性攻击(AA)。这项研究探讨了一般和特定指标因素在确定脆弱性得分方面的影响，为这些指标的潜在增强提供了新的视角。本研究采用定量的方法，计算并比较了56种对抗性攻击下的LLMS脆弱性得分的变异系数。这些攻击来自各种研究论文，通过在线数据库获得，使用多种漏洞指标进行评估。得分通过三个不同的LLM评估的值的平均值来确定。结果表明，现有的评分系统产生的脆弱性分数在不同攻击之间的差异很小，这表明许多度量因素不足以评估对LLM的对抗性攻击。对于特定于上下文的因素或具有预定义值集的因素尤其如此，例如CVSS中的那些因素。这些发现支持这样一种假设，即当前的脆弱性指标，特别是那些具有刚性值的指标，在评估LLM上的AA方面是有限的，这突显了开发针对此类攻击量身定做的更灵活、更通用的指标的必要性。这项研究对已建立的脆弱性度量的有效性和适用性进行了新的分析，特别是在针对大型语言模型的对抗性攻击的背景下，这两种攻击在最近几年都得到了极大的关注。通过广泛的测试和计算，这项研究强调了这些指标的局限性，并为改进和完善专门为低土地管理定制的脆弱性评估框架开辟了新的途径。



## **15. LLM-Virus: Evolutionary Jailbreak Attack on Large Language Models**

LLM-Virus：对大型语言模型的进化越狱攻击 cs.CR

**SubmitDate**: 2024-12-28    [abs](http://arxiv.org/abs/2501.00055v1) [paper-pdf](http://arxiv.org/pdf/2501.00055v1)

**Authors**: Miao Yu, Junfeng Fang, Yingjie Zhou, Xing Fan, Kun Wang, Shirui Pan, Qingsong Wen

**Abstract**: While safety-aligned large language models (LLMs) are increasingly used as the cornerstone for powerful systems such as multi-agent frameworks to solve complex real-world problems, they still suffer from potential adversarial queries, such as jailbreak attacks, which attempt to induce harmful content. Researching attack methods allows us to better understand the limitations of LLM and make trade-offs between helpfulness and safety. However, existing jailbreak attacks are primarily based on opaque optimization techniques (e.g. token-level gradient descent) and heuristic search methods like LLM refinement, which fall short in terms of transparency, transferability, and computational cost. In light of these limitations, we draw inspiration from the evolution and infection processes of biological viruses and propose LLM-Virus, a jailbreak attack method based on evolutionary algorithm, termed evolutionary jailbreak. LLM-Virus treats jailbreak attacks as both an evolutionary and transfer learning problem, utilizing LLMs as heuristic evolutionary operators to ensure high attack efficiency, transferability, and low time cost. Our experimental results on multiple safety benchmarks show that LLM-Virus achieves competitive or even superior performance compared to existing attack methods.

摘要: 尽管与安全一致的大型语言模型(LLM)越来越多地被用作多代理框架等强大系统的基石，以解决复杂的现实世界问题，但它们仍面临潜在的对抗性查询，例如试图诱导有害内容的越狱攻击。研究攻击方法可以让我们更好地了解LLM的局限性，并在有效性和安全性之间进行权衡。然而，现有的越狱攻击主要基于不透明的优化技术(如令牌级梯度下降)和启发式搜索方法，如LLM求精，这些方法在透明度、可转移性和计算成本方面都存在不足。针对这些局限性，我们从生物病毒的进化和感染过程中得到启发，提出了一种基于进化算法的越狱攻击方法LLM-Virus，称为进化越狱。LLM-Virus将越狱攻击视为一个进化和转移学习问题，利用LLM作为启发式进化算子，以确保高攻击效率、可转移性和低时间开销。我们在多个安全基准上的实验结果表明，与现有的攻击方法相比，LLM-Virus具有相当甚至更好的性能。



## **16. B-AVIBench: Towards Evaluating the Robustness of Large Vision-Language Model on Black-box Adversarial Visual-Instructions**

B-AVIBench：评估黑匣子对抗视觉指令上大型视觉语言模型的鲁棒性 cs.CV

Accepted by IEEE Transactions on Information Forensics & Security

**SubmitDate**: 2024-12-28    [abs](http://arxiv.org/abs/2403.09346v2) [paper-pdf](http://arxiv.org/pdf/2403.09346v2)

**Authors**: Hao Zhang, Wenqi Shao, Hong Liu, Yongqiang Ma, Ping Luo, Yu Qiao, Nanning Zheng, Kaipeng Zhang

**Abstract**: Large Vision-Language Models (LVLMs) have shown significant progress in responding well to visual-instructions from users. However, these instructions, encompassing images and text, are susceptible to both intentional and inadvertent attacks. Despite the critical importance of LVLMs' robustness against such threats, current research in this area remains limited. To bridge this gap, we introduce B-AVIBench, a framework designed to analyze the robustness of LVLMs when facing various Black-box Adversarial Visual-Instructions (B-AVIs), including four types of image-based B-AVIs, ten types of text-based B-AVIs, and nine types of content bias B-AVIs (such as gender, violence, cultural, and racial biases, among others). We generate 316K B-AVIs encompassing five categories of multimodal capabilities (ten tasks) and content bias. We then conduct a comprehensive evaluation involving 14 open-source LVLMs to assess their performance. B-AVIBench also serves as a convenient tool for practitioners to evaluate the robustness of LVLMs against B-AVIs. Our findings and extensive experimental results shed light on the vulnerabilities of LVLMs, and highlight that inherent biases exist even in advanced closed-source LVLMs like GeminiProVision and GPT-4V. This underscores the importance of enhancing the robustness, security, and fairness of LVLMs. The source code and benchmark are available at https://github.com/zhanghao5201/B-AVIBench.

摘要: 大型视觉语言模型(LVLM)在很好地响应用户的视觉指令方面取得了重大进展。但是，这些包含图像和文本的说明很容易受到有意和无意的攻击。尽管LVLMS对这类威胁的稳健性至关重要，但目前在这一领域的研究仍然有限。为了弥补这一差距，我们引入了B-AVIB边框架，该框架旨在分析LVLMS在面对各种黑盒对抗性视觉指令(B-AVI)时的健壮性，包括四种类型的基于图像的B-AVI、10种类型的基于文本的B-AVI和九种类型的内容偏见B-AVI(如性别、暴力、文化和种族偏见等)。我们生成了316k B-AVI，包括五类多模式能力(十项任务)和内容偏见。然后，我们对14个开源LVLM进行了全面的评估，以评估它们的性能。B-AVIBtch也可作为从业者评估LVLMS对B-AVIS的稳健性的便捷工具。我们的发现和广泛的实验结果揭示了LVLMS的漏洞，并突出表明即使在GeminiProVision和GPT-4V等先进的闭源LVLM中也存在固有偏差。这凸显了增强LVLM的健壮性、安全性和公平性的重要性。源代码和基准测试可在https://github.com/zhanghao5201/B-AVIBench.上获得



## **17. An Engorgio Prompt Makes Large Language Model Babble on**

Engorgio提示让大型语言模型胡言乱语 cs.CR

**SubmitDate**: 2024-12-27    [abs](http://arxiv.org/abs/2412.19394v1) [paper-pdf](http://arxiv.org/pdf/2412.19394v1)

**Authors**: Jianshuo Dong, Ziyuan Zhang, Qingjie Zhang, Han Qiu, Tianwei Zhang, Hao Wang, Hewu Li, Qi Li, Chao Zhang, Ke Xu

**Abstract**: Auto-regressive large language models (LLMs) have yielded impressive performance in many real-world tasks. However, the new paradigm of these LLMs also exposes novel threats. In this paper, we explore their vulnerability to inference cost attacks, where a malicious user crafts Engorgio prompts to intentionally increase the computation cost and latency of the inference process. We design Engorgio, a novel methodology, to efficiently generate adversarial Engorgio prompts to affect the target LLM's service availability. Engorgio has the following two technical contributions. (1) We employ a parameterized distribution to track LLMs' prediction trajectory. (2) Targeting the auto-regressive nature of LLMs' inference process, we propose novel loss functions to stably suppress the appearance of the <EOS> token, whose occurrence will interrupt the LLM's generation process. We conduct extensive experiments on 13 open-sourced LLMs with parameters ranging from 125M to 30B. The results show that Engorgio prompts can successfully induce LLMs to generate abnormally long outputs (i.e., roughly 2-13$\times$ longer to reach 90%+ of the output length limit) in a white-box scenario and our real-world experiment demonstrates Engergio's threat to LLM service with limited computing resources. The code is accessible at https://github.com/jianshuod/Engorgio-prompt.

摘要: 自回归大型语言模型(LLM)在许多实际任务中取得了令人印象深刻的性能。然而，这些LLM的新范式也暴露了新的威胁。在本文中，我们探讨了它们对推理成本攻击的脆弱性，在这种攻击中，恶意用户手工制作Engorgio会提示故意增加推理过程的计算成本和延迟。我们设计了一种新的方法Engorgio来有效地生成对抗性Engorgio提示，以影响目标LLM的服务可用性。Engorgio有以下两个技术贡献。(1)采用一种参数分布来跟踪LLMS的预测轨迹。(2)针对LLMS推理过程的自回归特性，提出了一种新的损失函数来稳定地抑制<EOS>令牌的出现，它的出现会中断LLM的生成过程。我们在13个开源的LLM上进行了广泛的实验，参数从125M到30B不等。结果表明，在白盒场景中，Engorgio提示能够成功地诱导LLMS产生异常长的输出(即大约2-13$\x$以达到输出长度限制的90%以上)，并且我们的真实世界实验证明了Engergio在有限计算资源的情况下对LLM服务的威胁。该代码可在https://github.com/jianshuod/Engorgio-prompt.上访问



## **18. Differential privacy enables fair and accurate AI-based analysis of speech disorders while protecting patient data**

差异隐私能够公平准确地对言语障碍进行基于人工智能的分析，同时保护患者数据 cs.LG

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2409.19078v2) [paper-pdf](http://arxiv.org/pdf/2409.19078v2)

**Authors**: Soroosh Tayebi Arasteh, Mahshad Lotfinia, Paula Andrea Perez-Toro, Tomas Arias-Vergara, Mahtab Ranji, Juan Rafael Orozco-Arroyave, Maria Schuster, Andreas Maier, Seung Hee Yang

**Abstract**: Speech pathology has impacts on communication abilities and quality of life. While deep learning-based models have shown potential in diagnosing these disorders, the use of sensitive data raises critical privacy concerns. Although differential privacy (DP) has been explored in the medical imaging domain, its application in pathological speech analysis remains largely unexplored despite the equally critical privacy concerns. This study is the first to investigate DP's impact on pathological speech data, focusing on the trade-offs between privacy, diagnostic accuracy, and fairness. Using a large, real-world dataset of 200 hours of recordings from 2,839 German-speaking participants, we observed a maximum accuracy reduction of 3.85% when training with DP with high privacy levels. To highlight real-world privacy risks, we demonstrated the vulnerability of non-private models to explicit gradient inversion attacks, reconstructing identifiable speech samples and showcasing DP's effectiveness in mitigating these risks. To generalize our findings across languages and disorders, we validated our approach on a dataset of Spanish-speaking Parkinson's disease patients, leveraging pretrained models from healthy English-speaking datasets, and demonstrated that careful pretraining on large-scale task-specific datasets can maintain favorable accuracy under DP constraints. A comprehensive fairness analysis revealed minimal gender bias at reasonable privacy levels but underscored the need for addressing age-related disparities. Our results establish that DP can balance privacy and utility in speech disorder detection, while highlighting unique challenges in privacy-fairness trade-offs for speech data. This provides a foundation for refining DP methodologies and improving fairness across diverse patient groups in real-world deployments.

摘要: 言语病理对沟通能力和生活质量都有影响。虽然基于深度学习的模型在诊断这些疾病方面显示出潜力，但敏感数据的使用引发了严重的隐私问题。尽管差分隐私(DP)已经在医学成像领域得到了探索，但它在病理性语音分析中的应用在很大程度上还没有被探索，尽管存在同样关键的隐私问题。这项研究是第一次调查DP对病态语音数据的影响，重点是隐私、诊断准确性和公平性之间的权衡。使用来自2,839名讲德语的参与者的200小时录音的大型真实世界数据集，我们观察到当使用高隐私级别的DP进行训练时，准确率最大下降3.85%。为了突出真实世界的隐私风险，我们展示了非私有模型对显式梯度反转攻击的脆弱性，重建了可识别的语音样本，并展示了DP在缓解这些风险方面的有效性。为了将我们的发现推广到语言和疾病，我们在讲西班牙语的帕金森氏病患者的数据集上验证了我们的方法，利用来自讲英语的健康数据集的预训练模型，并证明了在DP约束下，对大规模任务特定数据集进行仔细的预训练可以保持良好的准确性。一项全面的公平分析显示，在合理的隐私水平下，性别偏见最小，但强调了解决与年龄有关的差异的必要性。我们的结果表明，DP可以在语音紊乱检测中平衡隐私和效用，同时突出了语音数据在隐私-公平权衡方面的独特挑战。这为改进DP方法并在实际部署中提高不同患者群体的公平性奠定了基础。



## **19. CL-attack: Textual Backdoor Attacks via Cross-Lingual Triggers**

CL攻击：通过跨语言触发器进行文本后门攻击 cs.CR

The paper has been accepted to AAAI 2025

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.19037v1) [paper-pdf](http://arxiv.org/pdf/2412.19037v1)

**Authors**: Jingyi Zheng, Tianyi Hu, Tianshuo Cong, Xinlei He

**Abstract**: Backdoor attacks significantly compromise the security of large language models by triggering them to output specific and controlled content. Currently, triggers for textual backdoor attacks fall into two categories: fixed-token triggers and sentence-pattern triggers. However, the former are typically easy to identify and filter, while the latter, such as syntax and style, do not apply to all original samples and may lead to semantic shifts. In this paper, inspired by cross-lingual (CL) prompts of LLMs in real-world scenarios, we propose a higher-dimensional trigger method at the paragraph level, namely CL-attack. CL-attack injects the backdoor by using texts with specific structures that incorporate multiple languages, thereby offering greater stealthiness and universality compared to existing backdoor attack techniques. Extensive experiments on different tasks and model architectures demonstrate that CL-attack can achieve nearly 100% attack success rate with a low poisoning rate in both classification and generation tasks. We also empirically show that the CL-attack is more robust against current major defense methods compared to baseline backdoor attacks. Additionally, to mitigate CL-attack, we further develop a new defense called TranslateDefense, which can partially mitigate the impact of CL-attack.

摘要: 后门攻击会触发大型语言模型输出特定的受控内容，从而极大地危害它们的安全性。目前，文本后门攻击的触发器分为两类：固定令牌触发器和句型触发器。然而，前者通常很容易识别和过滤，而后者，如句法和风格，并不适用于所有的原始样本，并可能导致语义转移。受现实场景中LLMS跨语言提示的启发，我们提出了一种段落级别的高维触发方法，即CL-Attack。CL-Attack通过使用包含多种语言的特定结构的文本注入后门，因此与现有的后门攻击技术相比，提供了更大的隐蔽性和通用性。在不同的任务和模型体系结构上的大量实验表明，CL-Attack在分类和生成任务中都可以获得近100%的攻击成功率和较低的投毒率。我们的经验还表明，与基准后门攻击相比，CL-攻击对当前主要防御方法具有更强的健壮性。此外，为了缓解CL攻击，我们进一步开发了一种新的防御机制，称为TranslateDefense，它可以部分地缓解CL攻击的影响。



## **20. Attack-in-the-Chain: Bootstrapping Large Language Models for Attacks Against Black-box Neural Ranking Models**

链中攻击：引导大型语言模型来攻击黑匣子神经排名模型 cs.IR

Accepted by AAAI25

**SubmitDate**: 2024-12-25    [abs](http://arxiv.org/abs/2412.18770v1) [paper-pdf](http://arxiv.org/pdf/2412.18770v1)

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstract**: Neural ranking models (NRMs) have been shown to be highly effective in terms of retrieval performance. Unfortunately, they have also displayed a higher degree of sensitivity to attacks than previous generation models. To help expose and address this lack of robustness, we introduce a novel ranking attack framework named Attack-in-the-Chain, which tracks interactions between large language models (LLMs) and NRMs based on chain-of-thought (CoT) prompting to generate adversarial examples under black-box settings. Our approach starts by identifying anchor documents with higher ranking positions than the target document as nodes in the reasoning chain. We then dynamically assign the number of perturbation words to each node and prompt LLMs to execute attacks. Finally, we verify the attack performance of all nodes at each reasoning step and proceed to generate the next reasoning step. Empirical results on two web search benchmarks show the effectiveness of our method.

摘要: 神经排名模型（NRM）已被证明在检索性能方面非常有效。不幸的是，它们还表现出比前一代模型更高的攻击敏感性。为了帮助揭露和解决这种缺乏稳健性的问题，我们引入了一种名为Chain Attack-in-the-Chain的新型排名攻击框架，该框架基于思想链（CoT）来跟踪大型语言模型（LLM）和NRM之间的交互，以在黑匣子设置下生成对抗性示例。我们的方法首先将排名位置高于目标文档的锚文档识别为推理链中的节点。然后，我们动态地为每个节点分配扰动字的数量，并提示LLM执行攻击。最后，我们在每个推理步骤中验证所有节点的攻击性能，并继续生成下一个推理步骤。两个网络搜索基准的经验结果表明了我们方法的有效性。



## **21. Token Highlighter: Inspecting and Mitigating Jailbreak Prompts for Large Language Models**

Token Highliter：检查和缓解大型语言模型的越狱承诺 cs.CR

Accepted by AAAI 2025. Project page:  https://huggingface.co/spaces/TrustSafeAI/Token-Highlighter

**SubmitDate**: 2024-12-25    [abs](http://arxiv.org/abs/2412.18171v2) [paper-pdf](http://arxiv.org/pdf/2412.18171v2)

**Authors**: Xiaomeng Hu, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Large Language Models (LLMs) are increasingly being integrated into services such as ChatGPT to provide responses to user queries. To mitigate potential harm and prevent misuse, there have been concerted efforts to align the LLMs with human values and legal compliance by incorporating various techniques, such as Reinforcement Learning from Human Feedback (RLHF), into the training of the LLMs. However, recent research has exposed that even aligned LLMs are susceptible to adversarial manipulations known as Jailbreak Attacks. To address this challenge, this paper proposes a method called Token Highlighter to inspect and mitigate the potential jailbreak threats in the user query. Token Highlighter introduced a concept called Affirmation Loss to measure the LLM's willingness to answer the user query. It then uses the gradient of Affirmation Loss for each token in the user query to locate the jailbreak-critical tokens. Further, Token Highlighter exploits our proposed Soft Removal technique to mitigate the jailbreak effects of critical tokens via shrinking their token embeddings. Experimental results on two aligned LLMs (LLaMA-2 and Vicuna-V1.5) demonstrate that the proposed method can effectively defend against a variety of Jailbreak Attacks while maintaining competent performance on benign questions of the AlpacaEval benchmark. In addition, Token Highlighter is a cost-effective and interpretable defense because it only needs to query the protected LLM once to compute the Affirmation Loss and can highlight the critical tokens upon refusal.

摘要: 大型语言模型(LLM)越来越多地被集成到ChatGPT等服务中，以提供对用户查询的响应。为减少潜在危害和防止滥用，已作出协调一致的努力，通过将从人类反馈中强化学习(RLHF)等各种技术纳入LLMS的培训，使LLMS与人的价值观和法律合规保持一致。然而，最近的研究表明，即使是对准的LLM也容易受到称为越狱攻击的对抗性操纵的影响。为了应对这一挑战，本文提出了一种称为令牌荧光的方法来检测和缓解用户查询中潜在的越狱威胁。令牌亮点引入了一个名为肯定损失的概念，以衡量LLM回答用户问题的意愿。然后，它使用用户查询中每个令牌的确认损失梯度来定位越狱关键令牌。此外，令牌荧光利用我们提出的软删除技术，通过缩小关键令牌的令牌嵌入来缓解关键令牌的越狱影响。在两个对齐的LLMS(Llama-2和Vicuna-V1.5)上的实验结果表明，该方法可以有效地防御各种越狱攻击，同时保持在AlpacaEval基准测试的良性问题上的良好性能。此外，令牌加亮器是一种经济高效且可解释的防御方案，因为它只需查询受保护的LLM一次即可计算肯定损失，并且可以在拒绝时突出显示关键令牌。



## **22. Diverse and Effective Red Teaming with Auto-generated Rewards and Multi-step Reinforcement Learning**

多元化有效的Red团队，具有自动生成的奖励和多步骤强化学习 cs.LG

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.18693v1) [paper-pdf](http://arxiv.org/pdf/2412.18693v1)

**Authors**: Alex Beutel, Kai Xiao, Johannes Heidecke, Lilian Weng

**Abstract**: Automated red teaming can discover rare model failures and generate challenging examples that can be used for training or evaluation. However, a core challenge in automated red teaming is ensuring that the attacks are both diverse and effective. Prior methods typically succeed in optimizing either for diversity or for effectiveness, but rarely both. In this paper, we provide methods that enable automated red teaming to generate a large number of diverse and successful attacks.   Our approach decomposes the task into two steps: (1) automated methods for generating diverse attack goals and (2) generating effective attacks for those goals. While we provide multiple straightforward methods for generating diverse goals, our key contributions are to train an RL attacker that both follows those goals and generates diverse attacks for those goals. First, we demonstrate that it is easy to use a large language model (LLM) to generate diverse attacker goals with per-goal prompts and rewards, including rule-based rewards (RBRs) to grade whether the attacks are successful for the particular goal. Second, we demonstrate how training the attacker model with multi-step RL, where the model is rewarded for generating attacks that are different from past attempts further increases diversity while remaining effective. We use our approach to generate both prompt injection attacks and prompts that elicit unsafe responses. In both cases, we find that our approach is able to generate highly-effective and considerably more diverse attacks than past general red-teaming approaches.

摘要: 自动红色团队可以发现罕见的模型故障，并生成可用于培训或评估的具有挑战性的示例。然而，自动化红色团队的一个核心挑战是确保攻击既多样又有效。以前的方法通常成功地优化多样性或有效性，但很少两者兼而有之。在本文中，我们提供了使自动红色团队能够生成大量多样化和成功的攻击的方法。我们的方法将任务分解为两个步骤：(1)自动生成不同攻击目标的方法；(2)生成针对这些目标的有效攻击。虽然我们提供了多种简单的方法来生成不同的目标，但我们的主要贡献是训练一名既遵循这些目标又为这些目标生成不同攻击的RL攻击者。首先，我们证明了使用大型语言模型(LLM)来生成具有每个目标提示和奖励的不同攻击者目标是很容易的，其中包括基于规则的奖励(RBR)来对特定目标的攻击是否成功进行评级。其次，我们展示了如何用多步骤RL训练攻击者模型，其中该模型因生成与过去的尝试不同的攻击而得到奖励，从而在保持有效的同时进一步增加多样性。我们使用我们的方法来生成提示注入攻击和引发不安全响应的提示。在这两种情况下，我们发现我们的方法能够产生高度有效的攻击，而且比过去的常规红队方法更加多样化。



## **23. Can LLMs Obfuscate Code? A Systematic Analysis of Large Language Models into Assembly Code Obfuscation**

LLM可以混淆代码吗？大型语言模型到汇编代码混淆的系统分析 cs.CR

To appear in AAAI 2025, Main Track

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.16135v2) [paper-pdf](http://arxiv.org/pdf/2412.16135v2)

**Authors**: Seyedreza Mohseni, Seyedali Mohammadi, Deepa Tilwani, Yash Saxena, Gerald Ndawula, Sriram Vema, Edward Raff, Manas Gaur

**Abstract**: Malware authors often employ code obfuscations to make their malware harder to detect. Existing tools for generating obfuscated code often require access to the original source code (e.g., C++ or Java), and adding new obfuscations is a non-trivial, labor-intensive process. In this study, we ask the following question: Can Large Language Models (LLMs) potentially generate a new obfuscated assembly code? If so, this poses a risk to anti-virus engines and potentially increases the flexibility of attackers to create new obfuscation patterns. We answer this in the affirmative by developing the MetamorphASM benchmark comprising MetamorphASM Dataset (MAD) along with three code obfuscation techniques: dead code, register substitution, and control flow change. The MetamorphASM systematically evaluates the ability of LLMs to generate and analyze obfuscated code using MAD, which contains 328,200 obfuscated assembly code samples. We release this dataset and analyze the success rate of various LLMs (e.g., GPT-3.5/4, GPT-4o-mini, Starcoder, CodeGemma, CodeLlama, CodeT5, and LLaMA 3.1) in generating obfuscated assembly code. The evaluation was performed using established information-theoretic metrics and manual human review to ensure correctness and provide the foundation for researchers to study and develop remediations to this risk. The source code can be found at the following GitHub link: https://github.com/mohammadi-ali/MetamorphASM.

摘要: 恶意软件作者经常使用代码混淆来使他们的恶意软件更难被检测到。现有的用于生成混淆代码的工具通常需要访问原始源代码(例如，C++或Java)，而添加新的混淆并不是一个琐碎的、劳动密集型的过程。在这项研究中，我们问了以下问题：大型语言模型(LLM)是否有可能生成新的混淆汇编代码？如果是这样的话，这会给反病毒引擎带来风险，并可能增加攻击者创建新的混淆模式的灵活性。我们通过开发包含变形ASM数据集(MAD)以及三种代码混淆技术的变形ASM基准来肯定地回答这个问题：死代码、寄存器替换和控制流更改。变质ASM系统地评估LLMS使用MAD生成和分析混淆代码的能力，MAD包含328,200个混淆汇编代码样本。我们发布了这个数据集，并分析了各种LLM(例如，GPT-3.5/4、GPT-40-mini、Starcoder、CodeGema、CodeLlama、CodeT5和Llama 3.1)在生成混淆汇编代码方面的成功率。评估是使用已建立的信息理论指标和人工审查进行的，以确保正确性，并为研究人员研究和开发针对这一风险的补救措施提供基础。源代码可在GitHub链接中找到：https://github.com/mohammadi-ali/MetamorphASM.



## **24. SafeAligner: Safety Alignment against Jailbreak Attacks via Response Disparity Guidance**

SafeAligner：通过响应差异指导针对越狱攻击的安全调整 cs.CR

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2406.18118v4) [paper-pdf](http://arxiv.org/pdf/2406.18118v4)

**Authors**: Caishuang Huang, Wanxu Zhao, Rui Zheng, Huijie Lv, Wenyu Zhan, Shihan Dou, Sixian Li, Xiao Wang, Enyu Zhou, Junjie Ye, Yuming Yang, Tao Gui, Qi Zhang, Xuanjing Huang

**Abstract**: As the development of large language models (LLMs) rapidly advances, securing these models effectively without compromising their utility has become a pivotal area of research. However, current defense strategies against jailbreak attacks (i.e., efforts to bypass security protocols) often suffer from limited adaptability, restricted general capability, and high cost. To address these challenges, we introduce SafeAligner, a methodology implemented at the decoding stage to fortify defenses against jailbreak attacks. We begin by developing two specialized models: the Sentinel Model, which is trained to foster safety, and the Intruder Model, designed to generate riskier responses. SafeAligner leverages the disparity in security levels between the responses from these models to differentiate between harmful and beneficial tokens, effectively guiding the safety alignment by altering the output token distribution of the target model. Extensive experiments show that SafeAligner can increase the likelihood of beneficial tokens, while reducing the occurrence of harmful ones, thereby ensuring secure alignment with minimal loss to generality.

摘要: 随着大型语言模型(LLM)的发展，在不影响其实用性的情况下有效地保护这些模型已成为一个关键的研究领域。然而，当前针对越狱攻击的防御策略(即绕过安全协议的努力)往往存在适应性有限、通用能力有限和成本较高的问题。为了应对这些挑战，我们引入了SafeAligner，这是一种在解码阶段实施的方法，用于加强对越狱攻击的防御。我们首先开发两个专门的模型：哨兵模型和入侵者模型，前者旨在促进安全，后者旨在产生更高风险的反应。SafeAligner利用这些模型响应之间的安全级别差异来区分有害令牌和有益令牌，通过更改目标模型的输出令牌分布有效地指导安全对齐。广泛的实验表明，SafeAligner可以增加有益令牌的可能性，同时减少有害令牌的发生，从而确保安全对齐，并将对一般性的损失降至最低。



## **25. Prompted Contextual Vectors for Spear-Phishing Detection**

用于鱼叉钓鱼检测的预定上下文载体 cs.LG

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2402.08309v3) [paper-pdf](http://arxiv.org/pdf/2402.08309v3)

**Authors**: Daniel Nahmias, Gal Engelberg, Dan Klein, Asaf Shabtai

**Abstract**: Spear-phishing attacks present a significant security challenge, with large language models (LLMs) escalating the threat by generating convincing emails and facilitating target reconnaissance. To address this, we propose a detection approach based on a novel document vectorization method that utilizes an ensemble of LLMs to create representation vectors. By prompting LLMs to reason and respond to human-crafted questions, we quantify the presence of common persuasion principles in the email's content, producing prompted contextual document vectors for a downstream supervised machine learning model. We evaluate our method using a unique dataset generated by a proprietary system that automates target reconnaissance and spear-phishing email creation. Our method achieves a 91\% F1 score in identifying LLM-generated spear-phishing emails, with the training set comprising only traditional phishing and benign emails. Key contributions include a novel document vectorization method utilizing LLM reasoning, a publicly available dataset of high-quality spear-phishing emails, and the demonstrated effectiveness of our method in detecting such emails. This methodology can be utilized for various document classification tasks, particularly in adversarial problem domains.

摘要: 鱼叉式网络钓鱼攻击是一个重大的安全挑战，大型语言模型(LLM)通过生成令人信服的电子邮件和促进目标侦察来升级威胁。针对这一问题，我们提出了一种基于一种新的文档矢量化方法的检测方法，该方法利用一组LLM来创建表示向量。通过促使LLM对人类提出的问题进行推理和回应，我们量化了电子邮件内容中常见说服原则的存在，为下游有监督的机器学习模型生成了提示的上下文文档向量。我们使用由专有系统生成的唯一数据集来评估我们的方法，该系统自动执行目标侦察和鱼叉式网络钓鱼电子邮件创建。我们的方法在识别LLM生成的鱼叉式钓鱼邮件方面取得了91%的F1分数，训练集仅包括传统钓鱼邮件和良性电子邮件。主要贡献包括一种利用LLM推理的新的文档矢量化方法，一个公开可用的高质量鱼叉式钓鱼电子邮件数据集，以及我们的方法在检测此类电子邮件方面的有效性。这种方法可用于各种文档分类任务，特别是在对抗性问题领域。



## **26. The Dark Side of Function Calling: Pathways to Jailbreaking Large Language Models**

函数调用的阴暗面：越狱大型语言模型的途径 cs.CR

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2407.17915v4) [paper-pdf](http://arxiv.org/pdf/2407.17915v4)

**Authors**: Zihui Wu, Haichang Gao, Jianping He, Ping Wang

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities, but their power comes with significant security considerations. While extensive research has been conducted on the safety of LLMs in chat mode, the security implications of their function calling feature have been largely overlooked. This paper uncovers a critical vulnerability in the function calling process of LLMs, introducing a novel "jailbreak function" attack method that exploits alignment discrepancies, user coercion, and the absence of rigorous safety filters. Our empirical study, conducted on six state-of-the-art LLMs including GPT-4o, Claude-3.5-Sonnet, and Gemini-1.5-pro, reveals an alarming average success rate of over 90\% for this attack. We provide a comprehensive analysis of why function calls are susceptible to such attacks and propose defensive strategies, including the use of defensive prompts. Our findings highlight the urgent need for enhanced security measures in the function calling capabilities of LLMs, contributing to the field of AI safety by identifying a previously unexplored risk, designing an effective attack method, and suggesting practical defensive measures. Our code is available at https://github.com/wooozihui/jailbreakfunction.

摘要: 大型语言模型(LLM)已经展示了非凡的能力，但它们的强大也伴随着重要的安全考虑。虽然已经对聊天模式下的LLMS的安全性进行了广泛的研究，但其函数调用功能的安全含义在很大程度上被忽视了。本文揭示了LLMS函数调用过程中的一个严重漏洞，引入了一种新的“越狱函数”攻击方法，该方法利用了对齐差异、用户胁迫和缺乏严格的安全过滤器。我们在包括GPT-40、Claude-3.5-Sonnet和Gemini-1.5-Pro在内的六个最先进的LLM上进行的经验研究显示，该攻击的平均成功率超过90%，这是令人震惊的。我们对函数调用容易受到此类攻击的原因进行了全面分析，并提出了防御策略，包括使用防御提示。我们的发现突显了在LLMS的函数调用能力方面迫切需要增强安全措施，通过识别以前未探索的风险、设计有效的攻击方法并提出实用的防御措施来促进人工智能安全领域。我们的代码可以在https://github.com/wooozihui/jailbreakfunction.上找到



## **27. Can Large Language Models Improve the Adversarial Robustness of Graph Neural Networks?**

大型语言模型能否提高图神经网络的对抗鲁棒性？ cs.LG

accepted by KDD 2025

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2408.08685v3) [paper-pdf](http://arxiv.org/pdf/2408.08685v3)

**Authors**: Zhongjian Zhang, Xiao Wang, Huichi Zhou, Yue Yu, Mengmei Zhang, Cheng Yang, Chuan Shi

**Abstract**: Graph neural networks (GNNs) are vulnerable to adversarial attacks, especially for topology perturbations, and many methods that improve the robustness of GNNs have received considerable attention. Recently, we have witnessed the significant success of large language models (LLMs), leading many to explore the great potential of LLMs on GNNs. However, they mainly focus on improving the performance of GNNs by utilizing LLMs to enhance the node features. Therefore, we ask: Will the robustness of GNNs also be enhanced with the powerful understanding and inference capabilities of LLMs? By presenting the empirical results, we find that despite that LLMs can improve the robustness of GNNs, there is still an average decrease of 23.1% in accuracy, implying that the GNNs remain extremely vulnerable against topology attacks. Therefore, another question is how to extend the capabilities of LLMs on graph adversarial robustness. In this paper, we propose an LLM-based robust graph structure inference framework, LLM4RGNN, which distills the inference capabilities of GPT-4 into a local LLM for identifying malicious edges and an LM-based edge predictor for finding missing important edges, so as to recover a robust graph structure. Extensive experiments demonstrate that LLM4RGNN consistently improves the robustness across various GNNs. Even in some cases where the perturbation ratio increases to 40%, the accuracy of GNNs is still better than that on the clean graph. The source code can be found in https://github.com/zhongjian-zhang/LLM4RGNN.

摘要: 图神经网络(GNN)容易受到敌意攻击，尤其是对拓扑扰动的攻击，许多提高GNN健壮性的方法受到了广泛的关注。最近，我们目睹了大型语言模型(LLM)的巨大成功，这导致许多人探索LLM在GNN上的巨大潜力。然而，它们主要集中在通过利用LLMS来增强节点特征来提高GNN的性能。因此，我们问：GNN的健壮性是否也会随着LLMS强大的理解和推理能力而得到增强？通过给出实验结果，我们发现，尽管LLMS可以提高GNN的健壮性，但其准确率仍然平均下降23.1%，这意味着GNN仍然非常容易受到拓扑攻击。因此，另一个问题是如何扩展LLMS在图对抗健壮性方面的能力。本文提出了一种基于LLM的稳健图结构推理框架LLM4RGNN，该框架将GPT-4的推理能力抽象为用于识别恶意边的局部LLM和用于发现丢失重要边的基于LLM的边预测器，以恢复稳健的图结构。大量的实验表明，LLM4RGNN在不同的GNN上一致地提高了健壮性。即使在某些扰动比增加到40%的情况下，GNN的精度仍然好于干净图形上的精度。源代码可以在https://github.com/zhongjian-zhang/LLM4RGNN.中找到



## **28. Robustness-aware Automatic Prompt Optimization**

具有鲁棒性的自动提示优化 cs.CL

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.18196v1) [paper-pdf](http://arxiv.org/pdf/2412.18196v1)

**Authors**: Zeru Shi, Zhenting Wang, Yongye Su, Weidi Luo, Fan Yang, Yongfeng Zhang

**Abstract**: The performance of Large Language Models (LLMs) is based on the quality of the prompts and the semantic and structural integrity information of the input data. However, current prompt generation methods primarily focus on generating prompts for clean input data, often overlooking the impact of perturbed inputs on prompt performance. To address this limitation, we propose BATprompt (By Adversarial Training prompt), a novel method for prompt generation designed to withstand input perturbations (such as typos in the input). Inspired by adversarial training techniques, BATprompt demonstrates strong performance on a variety of perturbed tasks through a two-step process: adversarial perturbation and iterative optimization on unperturbed input via LLM. Unlike conventional adversarial attack methods, BATprompt avoids reliance on real gradients or model parameters. Instead, it leverages the advanced reasoning, language understanding and self reflection capabilities of LLMs to simulate gradients, guiding the generation of adversarial perturbations and optimizing prompt performance. In our experiments, we evaluate BATprompt on multiple datasets across both language understanding and generation tasks. The results indicate that BATprompt outperforms existing prompt generation methods, delivering superior robustness and performance under diverse perturbation scenarios.

摘要: 大语言模型的性能取决于提示的质量以及输入数据的语义和结构完整性信息。然而，目前的提示生成方法主要集中于为干净的输入数据生成提示，往往忽略了输入干扰对提示性能的影响。为了解决这一局限性，我们提出了一种新的提示生成方法BATprint(通过对抗性训练提示)，该方法旨在抵抗输入扰动(如输入中的打字错误)。受到对抗性训练技术的启发，通过两步过程：对抗性扰动和通过LLM对不受扰动的输入进行迭代优化，BATprint在各种扰动任务上表现出了强大的性能。与传统的对抗性攻击方法不同，BATprint避免了对真实梯度或模型参数的依赖。相反，它利用LLMS的高级推理、语言理解和自我反思能力来模拟梯度，指导生成对抗性扰动并优化提示性能。在我们的实验中，我们在语言理解和生成任务的多个数据集上对BATprint进行了评估。结果表明，BATprint的性能优于现有的提示生成方法，在不同的扰动场景下都具有较好的健壮性和性能。



## **29. Revisiting Jailbreaking for Large Language Models: A Representation Engineering Perspective**

重新审视大型语言模型的越狱：表示工程的角度 cs.CL

Accepted by COLING 2025

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2401.06824v4) [paper-pdf](http://arxiv.org/pdf/2401.06824v4)

**Authors**: Tianlong Li, Zhenghua Wang, Wenhao Liu, Muling Wu, Shihan Dou, Changze Lv, Xiaohua Wang, Xiaoqing Zheng, Xuanjing Huang

**Abstract**: The recent surge in jailbreaking attacks has revealed significant vulnerabilities in Large Language Models (LLMs) when exposed to malicious inputs. While various defense strategies have been proposed to mitigate these threats, there has been limited research into the underlying mechanisms that make LLMs vulnerable to such attacks. In this study, we suggest that the self-safeguarding capability of LLMs is linked to specific activity patterns within their representation space. Although these patterns have little impact on the semantic content of the generated text, they play a crucial role in shaping LLM behavior under jailbreaking attacks. Our findings demonstrate that these patterns can be detected with just a few pairs of contrastive queries. Extensive experimentation shows that the robustness of LLMs against jailbreaking can be manipulated by weakening or strengthening these patterns. Further visual analysis provides additional evidence for our conclusions, providing new insights into the jailbreaking phenomenon. These findings highlight the importance of addressing the potential misuse of open-source LLMs within the community.

摘要: 最近越狱攻击的激增暴露了大型语言模型(LLM)在暴露于恶意输入时的显著漏洞。虽然已经提出了各种防御策略来缓解这些威胁，但对于使LLMS容易受到此类攻击的潜在机制的研究有限。在这项研究中，我们认为LLMS的自我保护能力与其表征空间中的特定活动模式有关。虽然这些模式对生成的文本的语义内容影响不大，但它们在塑造越狱攻击下的LLM行为方面发挥了关键作用。我们的发现表明，只需几对对比查询就可以检测到这些模式。广泛的实验表明，可以通过削弱或加强这些模式来操纵LLMS对越狱的健壮性。进一步的视觉分析为我们的结论提供了更多的证据，为越狱现象提供了新的见解。这些发现突显了解决社区内可能滥用开源LLM的问题的重要性。



## **30. Stepwise Reasoning Error Disruption Attack of LLMs**

LLM的逐步推理错误中断攻击 cs.AI

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.11934v2) [paper-pdf](http://arxiv.org/pdf/2412.11934v2)

**Authors**: Jingyu Peng, Maolin Wang, Xiangyu Zhao, Kai Zhang, Wanyu Wang, Pengyue Jia, Qidong Liu, Ruocheng Guo, Qi Liu

**Abstract**: Large language models (LLMs) have made remarkable strides in complex reasoning tasks, but their safety and robustness in reasoning processes remain underexplored. Existing attacks on LLM reasoning are constrained by specific settings or lack of imperceptibility, limiting their feasibility and generalizability. To address these challenges, we propose the Stepwise rEasoning Error Disruption (SEED) attack, which subtly injects errors into prior reasoning steps to mislead the model into producing incorrect subsequent reasoning and final answers. Unlike previous methods, SEED is compatible with zero-shot and few-shot settings, maintains the natural reasoning flow, and ensures covert execution without modifying the instruction. Extensive experiments on four datasets across four different models demonstrate SEED's effectiveness, revealing the vulnerabilities of LLMs to disruptions in reasoning processes. These findings underscore the need for greater attention to the robustness of LLM reasoning to ensure safety in practical applications.

摘要: 大语言模型在复杂的推理任务中取得了显著的进展，但其在推理过程中的安全性和稳健性仍未得到充分的研究。现有的对LLM推理的攻击受到特定环境或缺乏不可见性的限制，限制了它们的可行性和普适性。为了应对这些挑战，我们提出了逐步推理错误中断(SEED)攻击，它巧妙地在先前的推理步骤中注入错误，以误导模型产生不正确的后续推理和最终答案。与以往的方法不同，SEED兼容零射和少射设置，保持了自然的推理流程，在不修改指令的情况下确保了隐蔽的执行。在四个不同模型的四个数据集上的广泛实验证明了SEED的有效性，揭示了LLMS在推理过程中对中断的脆弱性。这些发现强调了需要更多地关注LLM推理的健壮性，以确保在实际应用中的安全性。



## **31. Trading Devil RL: Backdoor attack via Stock market, Bayesian Optimization and Reinforcement Learning**

交易魔鬼RL：通过股市、Bayesian优化和强化学习进行后门攻击 cs.LG

End of data poisoning research!: Navier-stokes equations (3D);  Reinforcement Learning (RL); HFT (High Frequency Trading); Limit Order  Markets and backdoor attack detection

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2412.17908v1) [paper-pdf](http://arxiv.org/pdf/2412.17908v1)

**Authors**: Orson Mengara

**Abstract**: With the rapid development of generative artificial intelligence, particularly large language models, a number of sub-fields of deep learning have made significant progress and are now very useful in everyday applications. For example, well-known financial institutions simulate a wide range of scenarios for various models created by their research teams using reinforcement learning, both before production and after regular operations. In this work, we propose a backdoor attack that focuses solely on data poisoning. This particular backdoor attack is classified as an attack without prior consideration or trigger, and we name it FinanceLLMsBackRL. Our aim is to examine the potential effects of large language models that use reinforcement learning systems for text production or speech recognition, finance, physics, or the ecosystem of contemporary artificial intelligence models.

摘要: 随着生成式人工智能，特别是大型语言模型的快速发展，深度学习的许多子领域取得了重大进展，现在在日常应用中非常有用。例如，知名金融机构在生产之前和常规运营之后使用强化学习为其研究团队创建的各种模型模拟各种场景。在这项工作中，我们提出了一种仅针对数据中毒的后门攻击。这种特殊的后门攻击被归类为未经事先考虑或触发的攻击，我们将其命名为Financial LLMsBackRL。我们的目标是检查使用强化学习系统进行文本生成或语音识别、金融、物理或当代人工智能模型生态系统的大型语言模型的潜在影响。



## **32. Large Language Model Safety: A Holistic Survey**

大型语言模型安全性：整体调查 cs.AI

158 pages, 18 figures

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2412.17686v1) [paper-pdf](http://arxiv.org/pdf/2412.17686v1)

**Authors**: Dan Shi, Tianhao Shen, Yufei Huang, Zhigen Li, Yongqi Leng, Renren Jin, Chuang Liu, Xinwei Wu, Zishan Guo, Linhao Yu, Ling Shi, Bojian Jiang, Deyi Xiong

**Abstract**: The rapid development and deployment of large language models (LLMs) have introduced a new frontier in artificial intelligence, marked by unprecedented capabilities in natural language understanding and generation. However, the increasing integration of these models into critical applications raises substantial safety concerns, necessitating a thorough examination of their potential risks and associated mitigation strategies.   This survey provides a comprehensive overview of the current landscape of LLM safety, covering four major categories: value misalignment, robustness to adversarial attacks, misuse, and autonomous AI risks. In addition to the comprehensive review of the mitigation methodologies and evaluation resources on these four aspects, we further explore four topics related to LLM safety: the safety implications of LLM agents, the role of interpretability in enhancing LLM safety, the technology roadmaps proposed and abided by a list of AI companies and institutes for LLM safety, and AI governance aimed at LLM safety with discussions on international cooperation, policy proposals, and prospective regulatory directions.   Our findings underscore the necessity for a proactive, multifaceted approach to LLM safety, emphasizing the integration of technical solutions, ethical considerations, and robust governance frameworks. This survey is intended to serve as a foundational resource for academy researchers, industry practitioners, and policymakers, offering insights into the challenges and opportunities associated with the safe integration of LLMs into society. Ultimately, it seeks to contribute to the safe and beneficial development of LLMs, aligning with the overarching goal of harnessing AI for societal advancement and well-being. A curated list of related papers has been publicly available at https://github.com/tjunlp-lab/Awesome-LLM-Safety-Papers.

摘要: 大型语言模型的快速开发和部署为人工智能带来了一个新的前沿，其标志是在自然语言理解和生成方面具有前所未有的能力。然而，这些模型越来越多地集成到关键应用程序中，引发了大量的安全问题，需要彻底检查它们的潜在风险和相关的缓解策略。这项调查全面概述了LLM安全的现状，包括四个主要类别：价值错位、对对手攻击的健壮性、误用和自主AI风险。除了对这四个方面的缓解方法和评估资源进行全面审查外，我们还进一步探讨了与LLM安全相关的四个主题：LLM制剂的安全影响、可解释性在增强LLM安全方面的作用、一系列人工智能公司和机构为LLM安全提出并遵守的技术路线图，以及旨在实现LLM安全的人工智能治理，并就国际合作、政策建议和未来监管方向进行了讨论。我们的发现强调了对LLM安全采取积极、多方面方法的必要性，强调将技术解决方案、伦理考虑和强大的治理框架整合在一起。这项调查旨在为学院研究人员、行业从业者和政策制定者提供基础性资源，为低收入国家安全融入社会带来的挑战和机遇提供洞察力。最终，它寻求为低土地管理的安全和有益的发展做出贡献，与利用人工智能促进社会进步和福祉的总体目标保持一致。相关论文的精选名单已在https://github.com/tjunlp-lab/Awesome-LLM-Safety-Papers.上公开提供



## **33. Emerging Security Challenges of Large Language Models**

大型语言模型新出现的安全挑战 cs.CR

A version of this appeared in the larger Dagstuhl seminar 23431  report (https://doi.org/10.4230/DagRep.13.10.90)

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2412.17614v1) [paper-pdf](http://arxiv.org/pdf/2412.17614v1)

**Authors**: Herve Debar, Sven Dietrich, Pavel Laskov, Emil C. Lupu, Eirini Ntoutsi

**Abstract**: Large language models (LLMs) have achieved record adoption in a short period of time across many different sectors including high importance areas such as education [4] and healthcare [23]. LLMs are open-ended models trained on diverse data without being tailored for specific downstream tasks, enabling broad applicability across various domains. They are commonly used for text generation, but also widely used to assist with code generation [3], and even analysis of security information, as Microsoft Security Copilot demonstrates [18]. Traditional Machine Learning (ML) models are vulnerable to adversarial attacks [9]. So the concerns on the potential security implications of such wide scale adoption of LLMs have led to the creation of this working group on the security of LLMs. During the Dagstuhl seminar on "Network Attack Detection and Defense - AI-Powered Threats and Responses", the working group discussions focused on the vulnerability of LLMs to adversarial attacks, rather than their potential use in generating malware or enabling cyberattacks. Although we note the potential threat represented by the latter, the role of the LLMs in such uses is mostly as an accelerator for development, similar to what it is in benign use. To make the analysis more specific, the working group employed ChatGPT as a concrete example of an LLM and addressed the following points, which also form the structure of this report: 1. How do LLMs differ in vulnerabilities from traditional ML models? 2. What are the attack objectives in LLMs? 3. How complex it is to assess the risks posed by the vulnerabilities of LLMs? 4. What is the supply chain in LLMs, how data flow in and out of systems and what are the security implications? We conclude with an overview of open challenges and outlook.

摘要: 大型语言模型(LLM)在短时间内在许多不同的领域获得了创纪录的采用，包括教育[4]和医疗保健[23]等高度重要的领域。LLM是基于不同数据进行培训的开放式模型，无需为特定的下游任务量身定做，从而实现了跨不同领域的广泛适用性。它们通常用于文本生成，但也广泛用于协助代码生成[3]，甚至分析安全信息，正如Microsoft Security Copilot演示的那样[18]。传统的机器学习(ML)模型容易受到对抗性攻击[9]。因此，出于对大规模采用小岛屿发展中国家可能产生的安全影响的关切，设立了小岛屿发展中国家安全问题工作组。在达格斯图尔关于“网络攻击检测和防御--人工智能支持的威胁和反应”的研讨会期间，工作组讨论了低收入管理系统对对抗性攻击的脆弱性，而不是它们在生成恶意软件或支持网络攻击方面的潜在用途。尽管我们注意到后者所代表的潜在威胁，但小岛屿发展中国家在这种用途中的作用主要是作为发展的加速器，类似于它在良性使用中的作用。为了使分析更具体，工作组使用ChatGPT作为LLM的具体例子，并讨论了以下几点，这些点也构成了本报告的结构：1.LLMS与传统的ML模型在漏洞方面有何不同？2.LLMS中的攻击目标是什么？3.评估LLMS漏洞带来的风险有多复杂？4.LLMS中的供应链是什么，数据如何进出系统，以及安全影响是什么？最后，我们对开放的挑战和前景进行了概述。



## **34. Retention Score: Quantifying Jailbreak Risks for Vision Language Models**

保留分数：量化视觉语言模型的越狱风险 cs.AI

14 pages, 8 figures, AAAI 2025

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2412.17544v1) [paper-pdf](http://arxiv.org/pdf/2412.17544v1)

**Authors**: Zaitang Li, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: The emergence of Vision-Language Models (VLMs) is a significant advancement in integrating computer vision with Large Language Models (LLMs) to enhance multi-modal machine learning capabilities. However, this progress has also made VLMs vulnerable to sophisticated adversarial attacks, raising concerns about their reliability. The objective of this paper is to assess the resilience of VLMs against jailbreak attacks that can compromise model safety compliance and result in harmful outputs. To evaluate a VLM's ability to maintain its robustness against adversarial input perturbations, we propose a novel metric called the \textbf{Retention Score}. Retention Score is a multi-modal evaluation metric that includes Retention-I and Retention-T scores for quantifying jailbreak risks in visual and textual components of VLMs. Our process involves generating synthetic image-text pairs using a conditional diffusion model. These pairs are then predicted for toxicity score by a VLM alongside a toxicity judgment classifier. By calculating the margin in toxicity scores, we can quantify the robustness of the VLM in an attack-agnostic manner. Our work has four main contributions. First, we prove that Retention Score can serve as a certified robustness metric. Second, we demonstrate that most VLMs with visual components are less robust against jailbreak attacks than the corresponding plain VLMs. Additionally, we evaluate black-box VLM APIs and find that the security settings in Google Gemini significantly affect the score and robustness. Moreover, the robustness of GPT4V is similar to the medium settings of Gemini. Finally, our approach offers a time-efficient alternative to existing adversarial attack methods and provides consistent model robustness rankings when evaluated on VLMs including MiniGPT-4, InstructBLIP, and LLaVA.

摘要: 视觉语言模型(VLMS)的出现是将计算机视觉与大型语言模型(LLM)相结合以增强多模式机器学习能力的一个重大进步。然而，这一进展也使VLM容易受到复杂的对抗性攻击，这引发了人们对其可靠性的担忧。本文的目的是评估VLM对越狱攻击的恢复能力，这些攻击可能会损害模型安全合规性并导致有害输出。为了评估VLM对敌意输入扰动保持健壮性的能力，我们提出了一种新的度量，称为\extbf{保留分数}。保留分数是一种多模式评估指标，包括用于量化VLM视觉和文本部分越狱风险的保留-I和保留-T分数。我们的过程包括使用条件扩散模型生成合成图文对。然后，由VLM和毒性判断分类器一起预测这些对的毒性分数。通过计算毒性分数的差值，我们可以以一种攻击不可知的方式来量化VLM的健壮性。我们的工作有四个主要贡献。首先，我们证明了保留分数可以作为认证的稳健性度量。其次，我们证明了大多数具有可视组件的VLM对越狱攻击的健壮性不如相应的普通VLM。此外，我们对黑盒VLMAPI进行了评估，发现Google Gemini中的安全设置对分数和健壮性有显著影响。此外，GPT4V的健壮性与双子座的中等设置相似。最后，我们的方法为现有的对抗性攻击方法提供了一种省时的替代方法，并在包括MiniGPT-4、InstructBLIP和LLaVA的VLM上进行了评估，提供了一致的模型健壮性排名。



## **35. DiffusionAttacker: Diffusion-Driven Prompt Manipulation for LLM Jailbreak**

扩散攻击者：LLM越狱的扩散驱动提示操纵 cs.CL

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2412.17522v1) [paper-pdf](http://arxiv.org/pdf/2412.17522v1)

**Authors**: Hao Wang, Hao Li, Junda Zhu, Xinyuan Wang, Chengwei Pan, MinLie Huang, Lei Sha

**Abstract**: Large Language Models (LLMs) are susceptible to generating harmful content when prompted with carefully crafted inputs, a vulnerability known as LLM jailbreaking. As LLMs become more powerful, studying jailbreak methods is critical to enhancing security and aligning models with human values. Traditionally, jailbreak techniques have relied on suffix addition or prompt templates, but these methods suffer from limited attack diversity. This paper introduces DiffusionAttacker, an end-to-end generative approach for jailbreak rewriting inspired by diffusion models. Our method employs a sequence-to-sequence (seq2seq) text diffusion model as a generator, conditioning on the original prompt and guiding the denoising process with a novel attack loss. Unlike previous approaches that use autoregressive LLMs to generate jailbreak prompts, which limit the modification of already generated tokens and restrict the rewriting space, DiffusionAttacker utilizes a seq2seq diffusion model, allowing more flexible token modifications. This approach preserves the semantic content of the original prompt while producing harmful content. Additionally, we leverage the Gumbel-Softmax technique to make the sampling process from the diffusion model's output distribution differentiable, eliminating the need for iterative token search. Extensive experiments on Advbench and Harmbench demonstrate that DiffusionAttacker outperforms previous methods across various evaluation metrics, including attack success rate (ASR), fluency, and diversity.

摘要: 当提示使用精心编制的输入时，大型语言模型(LLM)很容易生成有害内容，这一漏洞被称为LLM越狱。随着LLMS变得越来越强大，研究越狱方法对于增强安全性和使模型与人类价值观保持一致至关重要。传统上，越狱技术依赖于后缀添加或提示模板，但这些方法受到攻击多样性的限制。本文介绍了一种受扩散模型启发的端到端生成式越狱重写方法DiffusionAttacker。该方法采用序列到序列(Seq2seq)文本扩散模型作为生成器，以原始提示为条件，以新的攻击损失指导去噪过程。与以前使用自回归LLM生成越狱提示的方法不同，DiffusionAttacker使用seq2seq扩散模型，允许更灵活的令牌修改，从而限制了对已生成令牌的修改并限制了重写空间。这种方法在产生有害内容的同时保留了原始提示的语义内容。此外，我们利用Gumbel-Softmax技术使扩散模型的输出分布的采样过程可微，从而消除了迭代令牌搜索的需要。在Advbench和Harmbench上的大量实验表明，DiffusionAttacker在包括攻击成功率(ASR)、流畅度和多样性在内的各种评估指标上都优于以前的方法。



## **36. Chinese SafetyQA: A Safety Short-form Factuality Benchmark for Large Language Models**

中文SafetyQA：大型语言模型的安全简短事实基准 cs.CL

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2412.15265v2) [paper-pdf](http://arxiv.org/pdf/2412.15265v2)

**Authors**: Yingshui Tan, Boren Zheng, Baihui Zheng, Kerui Cao, Huiyun Jing, Jincheng Wei, Jiaheng Liu, Yancheng He, Wenbo Su, Xiangyong Zhu, Bo Zheng, Kaifu Zhang

**Abstract**: With the rapid advancement of Large Language Models (LLMs), significant safety concerns have emerged. Fundamentally, the safety of large language models is closely linked to the accuracy, comprehensiveness, and clarity of their understanding of safety knowledge, particularly in domains such as law, policy and ethics. This factuality ability is crucial in determining whether these models can be deployed and applied safely and compliantly within specific regions. To address these challenges and better evaluate the factuality ability of LLMs to answer short questions, we introduce the Chinese SafetyQA benchmark. Chinese SafetyQA has several properties (i.e., Chinese, Diverse, High-quality, Static, Easy-to-evaluate, Safety-related, Harmless). Based on Chinese SafetyQA, we perform a comprehensive evaluation on the factuality abilities of existing LLMs and analyze how these capabilities relate to LLM abilities, e.g., RAG ability and robustness against attacks.

摘要: 随着大型语言模型（LLM）的快速发展，出现了重大的安全问题。从根本上讲，大型语言模型的安全性与其对安全知识理解的准确性、全面性和清晰性密切相关，特别是在法律、政策和道德等领域。这种真实性能力对于确定这些模型是否可以在特定区域安全、合规地部署和应用至关重要。为了应对这些挑战并更好地评估法学硕士回答简短问题的真实能力，我们引入了中国SafetyQA基准。中国SafetyQA具有多个属性（即，中文、多样化、高质量、静态、易于评估、安全相关、无害）。我们基于中国SafetyQA，对现有LLM的真实能力进行全面评估，并分析这些能力与LLM能力的关系，例如RAG能力和针对攻击的鲁棒性。



## **37. Defense Against Prompt Injection Attack by Leveraging Attack Techniques**

利用攻击技术防御即时注入攻击 cs.CR

9 pages

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2411.00459v2) [paper-pdf](http://arxiv.org/pdf/2411.00459v2)

**Authors**: Yulin Chen, Haoran Li, Zihao Zheng, Yangqiu Song, Dekai Wu, Bryan Hooi

**Abstract**: With the advancement of technology, large language models (LLMs) have achieved remarkable performance across various natural language processing (NLP) tasks, powering LLM-integrated applications like Microsoft Copilot. However, as LLMs continue to evolve, new vulnerabilities, especially prompt injection attacks arise. These attacks trick LLMs into deviating from the original input instructions and executing the attacker's instructions injected in data content, such as retrieved results. Recent attack methods leverage LLMs' instruction-following abilities and their inabilities to distinguish instructions injected in the data content, and achieve a high attack success rate (ASR). When comparing the attack and defense methods, we interestingly find that they share similar design goals, of inducing the model to ignore unwanted instructions and instead to execute wanted instructions. Therefore, we raise an intuitive question: Could these attack techniques be utilized for defensive purposes? In this paper, we invert the intention of prompt injection methods to develop novel defense methods based on previous training-free attack methods, by repeating the attack process but with the original input instruction rather than the injected instruction. Our comprehensive experiments demonstrate that our defense techniques outperform existing training-free defense approaches, achieving state-of-the-art results.

摘要: 随着技术的进步，大语言模型(LLM)在各种自然语言处理(NLP)任务中取得了显著的性能，支持Microsoft Copilot等LLM集成应用程序。然而，随着LLMS的不断发展，出现了新的漏洞，特别是即时注入攻击。这些攻击欺骗LLM偏离原始输入指令，并执行注入数据内容的攻击者指令，例如检索的结果。最近的攻击方法利用LLMS的指令跟随能力和它们无法区分注入到数据内容中的指令的能力，实现了高攻击成功率(ASR)。当比较攻击和防御方法时，我们有趣地发现它们有相似的设计目标，都是诱导模型忽略不想要的指令，而是执行想要的指令。因此，我们提出了一个直观的问题：这些攻击技术是否可以用于防御目的？在本文中，我们反转了快速注入方法的意图，在以前的免训练攻击方法的基础上，通过重复攻击过程来开发新的防御方法，但使用的是原始输入指令而不是注入指令。我们的综合实验表明，我们的防御技术优于现有的免训练防御方法，取得了最先进的结果。



## **38. SEAS: Self-Evolving Adversarial Safety Optimization for Large Language Models**

SEAS：大型语言模型的自进化对抗安全优化 cs.CL

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2408.02632v2) [paper-pdf](http://arxiv.org/pdf/2408.02632v2)

**Authors**: Muxi Diao, Rumei Li, Shiyang Liu, Guogang Liao, Jingang Wang, Xunliang Cai, Weiran Xu

**Abstract**: As large language models (LLMs) continue to advance in capability and influence, ensuring their security and preventing harmful outputs has become crucial. A promising approach to address these concerns involves training models to automatically generate adversarial prompts for red teaming. However, the evolving subtlety of vulnerabilities in LLMs challenges the effectiveness of current adversarial methods, which struggle to specifically target and explore the weaknesses of these models. To tackle these challenges, we introduce the $\mathbf{S}\text{elf-}\mathbf{E}\text{volving }\mathbf{A}\text{dversarial }\mathbf{S}\text{afety }\mathbf{(SEAS)}$ optimization framework, which enhances security by leveraging data generated by the model itself. SEAS operates through three iterative stages: Initialization, Attack, and Adversarial Optimization, refining both the Red Team and Target models to improve robustness and safety. This framework reduces reliance on manual testing and significantly enhances the security capabilities of LLMs. Our contributions include a novel adversarial framework, a comprehensive safety dataset, and after three iterations, the Target model achieves a security level comparable to GPT-4, while the Red Team model shows a marked increase in attack success rate (ASR) against advanced models. Our code and datasets are released at https://SEAS-LLM.github.io/.

摘要: 随着大型语言模型在能力和影响力方面的不断进步，确保它们的安全和防止有害输出变得至关重要。解决这些担忧的一个有希望的方法是建立训练模型，为红色团队自动生成对抗性提示。然而，LLMS中不断演变的漏洞的微妙之处挑战了当前对抗性方法的有效性，这些方法难以具体针对和探索这些模型的弱点。为了应对这些挑战，我们引入了$\mathbf{S}\Text{ELF-}\mathbf{E}\Text{volving}\mathbf{A}\Text{dversarial}\mathbf{S}\Text{afty}\mathbf{(SEA)}$优化框架，该框架通过利用模型本身生成的数据来增强安全性。SEA经历了三个迭代阶段：初始化、攻击和对抗性优化，完善了Red Team和Target模型，以提高健壮性和安全性。该框架减少了对手动测试的依赖，显著增强了LLMS的安全能力。我们的贡献包括一个新的对抗性框架，一个全面的安全数据集，经过三次迭代，Target模型达到了与GPT-4相当的安全级别，而Red Team模型显示出相对于高级模型在攻击成功率(ASR)方面的显著提高。我们的代码和数据集在https://SEAS-LLM.github.io/.上发布



## **39. The Superalignment of Superhuman Intelligence with Large Language Models**

超人智能与大型语言模型的超级对齐 cs.CL

Under review of Science China

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2412.11145v2) [paper-pdf](http://arxiv.org/pdf/2412.11145v2)

**Authors**: Minlie Huang, Yingkang Wang, Shiyao Cui, Pei Ke, Jie Tang

**Abstract**: We have witnessed superhuman intelligence thanks to the fast development of large language models and multimodal language models. As the application of such superhuman models becomes more and more popular, a critical question arises here: how can we ensure superhuman models are still safe, reliable and aligned well to human values? In this position paper, we discuss the concept of superalignment from the learning perspective to answer this question by outlining the learning paradigm shift from large-scale pretraining, supervised fine-tuning, to alignment training. We define superalignment as designing effective and efficient alignment algorithms to learn from noisy-labeled data (point-wise samples or pair-wise preference data) in a scalable way when the task becomes very complex for human experts to annotate and the model is stronger than human experts. We highlight some key research problems in superalignment, namely, weak-to-strong generalization, scalable oversight, and evaluation. We then present a conceptual framework for superalignment, which consists of three modules: an attacker which generates adversary queries trying to expose the weaknesses of a learner model; a learner which will refine itself by learning from scalable feedbacks generated by a critic model along with minimal human experts; and a critic which generates critics or explanations for a given query-response pair, with a target of improving the learner by criticizing. We discuss some important research problems in each component of this framework and highlight some interesting research ideas that are closely related to our proposed framework, for instance, self-alignment, self-play, self-refinement, and more. Last, we highlight some future research directions for superalignment, including identification of new emergent risks and multi-dimensional alignment.

摘要: 由于大型语言模型和多模式语言模型的快速发展，我们见证了超人的智能。随着这种超人模型的应用变得越来越普遍，一个关键的问题出现了：我们如何确保超人模型仍然安全、可靠，并与人类的价值观保持良好一致？在这份立场文件中，我们从学习的角度讨论了超匹配的概念，通过概述学习范式从大规模预训练、有监督的微调到对齐训练的转变来回答这个问题。我们将超比对定义为设计有效和高效的比对算法，当任务对于人类专家来说变得非常复杂并且模型比人类专家更强时，以可扩展的方式从噪声标记的数据(点状样本或成对偏好数据)中学习。我们强调了超比对中的一些关键研究问题，即从弱到强的泛化、可扩展的监督和评估。然后，我们提出了一个超对齐的概念框架，它由三个模块组成：攻击者，生成敌意查询，试图揭露学习者模型的弱点；学习者，将通过与最少的人类专家一起从批评者模型生成的可伸缩反馈中学习来改进自己；批评者，为给定的查询-响应对生成批评者或解释，目标是通过批评来改进学习者。我们讨论了该框架每个组成部分中的一些重要研究问题，并突出了与我们提出的框架密切相关的一些有趣的研究想法，例如自我调整、自我发挥、自我完善等。最后，我们指出了超配准未来的研究方向，包括识别新出现的风险和多维配对。



## **40. EM-MIAs: Enhancing Membership Inference Attacks in Large Language Models through Ensemble Modeling**

EM-MIA：通过集合建模增强大型语言模型中的成员推断攻击 cs.RO

Accepted by ICASSP 2025 Main

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2412.17249v1) [paper-pdf](http://arxiv.org/pdf/2412.17249v1)

**Authors**: Zichen Song, Sitan Huang, Zhongfeng Kang

**Abstract**: With the widespread application of large language models (LLM), concerns about the privacy leakage of model training data have increasingly become a focus. Membership Inference Attacks (MIAs) have emerged as a critical tool for evaluating the privacy risks associated with these models. Although existing attack methods, such as LOSS, Reference-based, min-k, and zlib, perform well in certain scenarios, their effectiveness on large pre-trained language models often approaches random guessing, particularly in the context of large-scale datasets and single-epoch training. To address this issue, this paper proposes a novel ensemble attack method that integrates several existing MIAs techniques (LOSS, Reference-based, min-k, zlib) into an XGBoost-based model to enhance overall attack performance (EM-MIAs). Experimental results demonstrate that the ensemble model significantly improves both AUC-ROC and accuracy compared to individual attack methods across various large language models and datasets. This indicates that by combining the strengths of different methods, we can more effectively identify members of the model's training data, thereby providing a more robust tool for evaluating the privacy risks of LLM. This study offers new directions for further research in the field of LLM privacy protection and underscores the necessity of developing more powerful privacy auditing methods.

摘要: 随着大型语言模型的广泛应用，对模型训练数据隐私泄露的担忧日益成为人们关注的焦点。成员身份推断攻击(MIA)已成为评估与这些模型相关的隐私风险的关键工具。尽管现有的攻击方法，如Lost、基于引用、min-k和zlib，在某些场景下表现良好，但它们在大型预训练语言模型上的有效性往往接近随机猜测，特别是在大规模数据集和单纪元训练的背景下。针对这一问题，提出了一种新的集成攻击方法，该方法将现有的几种MIAs技术(Lost、Reference-Based、min-k、zlib)集成到一个基于XGBoost的模型中，以提高总体攻击性能(EM-MIA)。实验结果表明，在不同的大型语言模型和数据集上，与单独的攻击方法相比，集成模型显著提高了AUC-ROC和准确率。这表明，通过结合不同方法的优点，我们可以更有效地识别模型的训练数据成员，从而为评估LLM的隐私风险提供更稳健的工具。这项研究为LLM隐私保护领域的进一步研究提供了新的方向，并强调了开发更强大的隐私审计方法的必要性。



## **41. Shaping the Safety Boundaries: Understanding and Defending Against Jailbreaks in Large Language Models**

塑造安全边界：理解和防御大型语言模型中的越狱 cs.CL

17 pages, 9 figures

**SubmitDate**: 2024-12-22    [abs](http://arxiv.org/abs/2412.17034v1) [paper-pdf](http://arxiv.org/pdf/2412.17034v1)

**Authors**: Lang Gao, Xiangliang Zhang, Preslav Nakov, Xiuying Chen

**Abstract**: Jailbreaking in Large Language Models (LLMs) is a major security concern as it can deceive LLMs to generate harmful text. Yet, there is still insufficient understanding of how jailbreaking works, which makes it hard to develop effective defense strategies. We aim to shed more light into this issue: we conduct a detailed large-scale analysis of seven different jailbreak methods and find that these disagreements stem from insufficient observation samples. In particular, we introduce \textit{safety boundary}, and we find that jailbreaks shift harmful activations outside that safety boundary, where LLMs are less sensitive to harmful information. We also find that the low and the middle layers are critical in such shifts, while deeper layers have less impact. Leveraging on these insights, we propose a novel defense called \textbf{Activation Boundary Defense} (ABD), which adaptively constrains the activations within the safety boundary. We further use Bayesian optimization to selectively apply the defense method to the low and the middle layers. Our experiments on several benchmarks show that ABD achieves an average DSR of over 98\% against various forms of jailbreak attacks, with less than 2\% impact on the model's general capabilities.

摘要: 大型语言模型中的越狱是一个主要的安全问题，因为它可以欺骗大型语言模型生成有害的文本。然而，对越狱是如何运作的理解仍然不够，这使得制定有效的防御策略变得困难。我们的目标是更多地阐明这个问题：我们对七种不同的越狱方法进行了详细的大规模分析，发现这些分歧源于观察样本不足。特别是，我们引入了安全边界，我们发现越狱将有害的激活转移到了安全边界之外，在安全边界中，LLM对有害信息不那么敏感。我们还发现，低层和中层在这种转变中是关键的，而更深的层影响较小。利用这些见解，我们提出了一种新的防御措施，称为\extbf(激活边界防御)(ABD)，它自适应地将激活限制在安全边界内。我们进一步使用贝叶斯优化来选择性地将防御方法应用于低层和中层。我们在几个基准测试上的实验表明，ABD对各种形式的越狱攻击的平均DSR超过98%，而对模型的总体性能的影响不到2%。



## **42. Robustness of Large Language Models Against Adversarial Attacks**

大型语言模型对抗对抗攻击的鲁棒性 cs.CL

**SubmitDate**: 2024-12-22    [abs](http://arxiv.org/abs/2412.17011v1) [paper-pdf](http://arxiv.org/pdf/2412.17011v1)

**Authors**: Yiyi Tao, Yixian Shen, Hang Zhang, Yanxin Shen, Lun Wang, Chuanqi Shi, Shaoshuai Du

**Abstract**: The increasing deployment of Large Language Models (LLMs) in various applications necessitates a rigorous evaluation of their robustness against adversarial attacks. In this paper, we present a comprehensive study on the robustness of GPT LLM family. We employ two distinct evaluation methods to assess their resilience. The first method introduce character-level text attack in input prompts, testing the models on three sentiment classification datasets: StanfordNLP/IMDB, Yelp Reviews, and SST-2. The second method involves using jailbreak prompts to challenge the safety mechanisms of the LLMs. Our experiments reveal significant variations in the robustness of these models, demonstrating their varying degrees of vulnerability to both character-level and semantic-level adversarial attacks. These findings underscore the necessity for improved adversarial training and enhanced safety mechanisms to bolster the robustness of LLMs.

摘要: 大型语言模型（LLM）在各种应用程序中的部署越来越多，需要严格评估其对抗性攻击的稳健性。本文对GPT LLM家族的稳健性进行了全面的研究。我们采用两种不同的评估方法来评估其弹性。第一种方法在输入提示中引入字符级文本攻击，在三个情感分类数据集上测试模型：StanfordNLP/IMDB、Yelp Reviews和CST-2。第二种方法涉及使用越狱提示来挑战LLM的安全机制。我们的实验揭示了这些模型的稳健性存在显着差异，证明了它们对字符级和语义级对抗攻击的脆弱性程度不同。这些发现强调了改进对抗培训和增强安全机制以增强LLM稳健性的必要性。



## **43. Breaking Barriers in Physical-World Adversarial Examples: Improving Robustness and Transferability via Robust Feature**

打破物理世界对抗示例中的障碍：通过稳健特征提高稳健性和可移植性 cs.CV

Accepted by AAAI2025

**SubmitDate**: 2024-12-22    [abs](http://arxiv.org/abs/2412.16958v1) [paper-pdf](http://arxiv.org/pdf/2412.16958v1)

**Authors**: Yichen Wang, Yuxuan Chou, Ziqi Zhou, Hangtao Zhang, Wei Wan, Shengshan Hu, Minghui Li

**Abstract**: As deep neural networks (DNNs) are widely applied in the physical world, many researches are focusing on physical-world adversarial examples (PAEs), which introduce perturbations to inputs and cause the model's incorrect outputs. However, existing PAEs face two challenges: unsatisfactory attack performance (i.e., poor transferability and insufficient robustness to environment conditions), and difficulty in balancing attack effectiveness with stealthiness, where better attack effectiveness often makes PAEs more perceptible.   In this paper, we explore a novel perturbation-based method to overcome the challenges. For the first challenge, we introduce a strategy Deceptive RF injection based on robust features (RFs) that are predictive, robust to perturbations, and consistent across different models. Specifically, it improves the transferability and robustness of PAEs by covering RFs of other classes onto the predictive features in clean images. For the second challenge, we introduce another strategy Adversarial Semantic Pattern Minimization, which removes most perturbations and retains only essential adversarial patterns in AEsBased on the two strategies, we design our method Robust Feature Coverage Attack (RFCoA), comprising Robust Feature Disentanglement and Adversarial Feature Fusion. In the first stage, we extract target class RFs in feature space. In the second stage, we use attention-based feature fusion to overlay these RFs onto predictive features of clean images and remove unnecessary perturbations. Experiments show our method's superior transferability, robustness, and stealthiness compared to existing state-of-the-art methods. Additionally, our method's effectiveness can extend to Large Vision-Language Models (LVLMs), indicating its potential applicability to more complex tasks.

摘要: 随着深度神经网络(DNN)在物理世界中的广泛应用，许多研究都集中在物理世界中的对抗性例子(PAE)上，这些例子会对输入产生扰动，导致模型输出不正确。然而，现有的PAE面临着两个挑战：攻击性能不令人满意(即可转移性差，对环境条件的健壮性不够)，以及难以平衡攻击有效性和隐蔽性，更好的攻击效率往往使PAE更容易被感知。在本文中，我们探索了一种新的基于扰动的方法来克服这些挑战。对于第一个挑战，我们引入了一种基于稳健特征(RF)的欺骗性射频注入策略，这些特征具有预测性、对扰动具有鲁棒性，并且在不同的模型中保持一致。具体地说，它通过将其他类的RF覆盖到干净图像中的预测特征来提高PAE的可转移性和稳健性。对于第二个挑战，我们引入了另一种对抗性语义模式最小化策略，该策略去除了大部分扰动，只保留了AEss中的基本对抗性模式。在这两种策略的基础上，我们设计了一种鲁棒特征覆盖攻击(RFCoA)方法，包括健壮特征解缠和对抗性特征融合。在第一阶段，我们在特征空间中提取目标类RFS。在第二阶段，我们使用基于注意力的特征融合将这些RF叠加到干净图像的预测特征上，并去除不必要的扰动。实验表明，与现有最先进的方法相比，我们的方法具有更好的可转移性、健壮性和隐蔽性。此外，我们的方法的有效性可以扩展到大型视觉语言模型(LVLM)，这表明它对更复杂的任务具有潜在的适用性。



## **44. Safely Learning with Private Data: A Federated Learning Framework for Large Language Model**

利用私人数据安全学习：大型语言模型的联邦学习框架 cs.CR

EMNLP 2024

**SubmitDate**: 2024-12-22    [abs](http://arxiv.org/abs/2406.14898v4) [paper-pdf](http://arxiv.org/pdf/2406.14898v4)

**Authors**: JiaYing Zheng, HaiNan Zhang, LingXiang Wang, WangJie Qiu, HongWei Zheng, ZhiMing Zheng

**Abstract**: Private data, being larger and quality-higher than public data, can greatly improve large language models (LLM). However, due to privacy concerns, this data is often dispersed in multiple silos, making its secure utilization for LLM training a challenge. Federated learning (FL) is an ideal solution for training models with distributed private data, but traditional frameworks like FedAvg are unsuitable for LLM due to their high computational demands on clients. An alternative, split learning, offloads most training parameters to the server while training embedding and output layers locally, making it more suitable for LLM. Nonetheless, it faces significant challenges in security and efficiency. Firstly, the gradients of embeddings are prone to attacks, leading to potential reverse engineering of private data. Furthermore, the server's limitation of handle only one client's training request at a time hinders parallel training, severely impacting training efficiency. In this paper, we propose a Federated Learning framework for LLM, named FL-GLM, which prevents data leakage caused by both server-side and peer-client attacks while improving training efficiency. Specifically, we first place the input block and output block on local client to prevent embedding gradient attacks from server. Secondly, we employ key-encryption during client-server communication to prevent reverse engineering attacks from peer-clients. Lastly, we employ optimization methods like client-batching or server-hierarchical, adopting different acceleration methods based on the actual computational capabilities of the server. Experimental results on NLU and generation tasks demonstrate that FL-GLM achieves comparable metrics to centralized chatGLM model, validating the effectiveness of our federated learning framework.

摘要: 私有数据比公共数据更大、质量更高，可以极大地改进大型语言模型(LLM)。然而，出于隐私方面的考虑，这些数据通常分散在多个竖井中，这使得将其安全地用于LLM培训成为一项挑战。联邦学习(FL)是一种适用于具有分布式私有数据的模型训练的理想解决方案，但FedAvg等传统框架由于对客户端的计算要求较高而不适用于LLM。另一种选择是分离学习，将大部分训练参数卸载到服务器，同时在本地训练嵌入和输出层，使其更适合LLM。尽管如此，它在安全和效率方面仍面临重大挑战。首先，嵌入的梯度容易受到攻击，从而导致对私有数据的潜在逆向工程。此外，服务器一次只能处理一个客户端的训练请求的限制阻碍了并行训练，严重影响了训练效率。本文提出了一种用于LLM的联邦学习框架FL-GLM，该框架在提高训练效率的同时，防止了服务器端攻击和对等客户端攻击引起的数据泄漏。具体地说，我们首先将输入块和输出块放置在本地客户端，以防止来自服务器的嵌入梯度攻击。其次，我们在客户-服务器通信过程中使用密钥加密，以防止来自对等客户端的反向工程攻击。最后，我们采用了客户端批处理或服务器分层等优化方法，根据服务器的实际计算能力采用不同的加速方法。在NLU和生成任务上的实验结果表明，FL-GLM达到了与集中式ChatGLM模型相当的指标，验证了联邦学习框架的有效性。



## **45. RoboSignature: Robust Signature and Watermarking on Network Attacks**

RoboSignature：网络攻击的鲁棒签名和水印 cs.CR

**SubmitDate**: 2024-12-22    [abs](http://arxiv.org/abs/2412.19834v1) [paper-pdf](http://arxiv.org/pdf/2412.19834v1)

**Authors**: Aryaman Shaan, Garvit Banga, Raghav Mantri

**Abstract**: Generative models have enabled easy creation and generation of images of all kinds given a single prompt. However, this has also raised ethical concerns about what is an actual piece of content created by humans or cameras compared to model-generated content like images or videos. Watermarking data generated by modern generative models is a popular method to provide information on the source of the content. The goal is for all generated images to conceal an invisible watermark, allowing for future detection or identification. The Stable Signature finetunes the decoder of Latent Diffusion Models such that a unique watermark is rooted in any image produced by the decoder. In this paper, we present a novel adversarial fine-tuning attack that disrupts the model's ability to embed the intended watermark, exposing a significant vulnerability in existing watermarking methods. To address this, we further propose a tamper-resistant fine-tuning algorithm inspired by methods developed for large language models, tailored to the specific requirements of watermarking in LDMs. Our findings emphasize the importance of anticipating and defending against potential vulnerabilities in generative systems.

摘要: 生成性模型能够在单一提示下轻松创建和生成所有类型的图像。然而，这也引发了伦理方面的担忧，即与图像或视频等模型生成的内容相比，什么是由人或相机创建的实际内容。通过现代生成模型生成的水印数据是提供有关内容来源的信息的流行方法。目标是让所有生成的图像隐藏一个不可见的水印，以便将来进行检测或识别。稳定的签名对潜在扩散模型的解码器进行微调，使得唯一的水印植根于解码器产生的任何图像中。在本文中，我们提出了一种新的对抗性微调攻击，该攻击破坏了模型嵌入预期水印的能力，暴露了现有水印方法中的一个显著漏洞。为了解决这个问题，我们进一步提出了一种防篡改微调算法，该算法的灵感来自于为大型语言模型开发的方法，该算法针对LDM中水印的特定要求而量身定做。我们的发现强调了预测和防御生成性系统中潜在漏洞的重要性。



## **46. The Task Shield: Enforcing Task Alignment to Defend Against Indirect Prompt Injection in LLM Agents**

任务盾：强制任务一致以防止LLM代理中的间接提示注入 cs.CR

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2412.16682v1) [paper-pdf](http://arxiv.org/pdf/2412.16682v1)

**Authors**: Feiran Jia, Tong Wu, Xin Qin, Anna Squicciarini

**Abstract**: Large Language Model (LLM) agents are increasingly being deployed as conversational assistants capable of performing complex real-world tasks through tool integration. This enhanced ability to interact with external systems and process various data sources, while powerful, introduces significant security vulnerabilities. In particular, indirect prompt injection attacks pose a critical threat, where malicious instructions embedded within external data sources can manipulate agents to deviate from user intentions. While existing defenses based on rule constraints, source spotlighting, and authentication protocols show promise, they struggle to maintain robust security while preserving task functionality. We propose a novel and orthogonal perspective that reframes agent security from preventing harmful actions to ensuring task alignment, requiring every agent action to serve user objectives. Based on this insight, we develop Task Shield, a test-time defense mechanism that systematically verifies whether each instruction and tool call contributes to user-specified goals. Through experiments on the AgentDojo benchmark, we demonstrate that Task Shield reduces attack success rates (2.07\%) while maintaining high task utility (69.79\%) on GPT-4o.

摘要: 大型语言模型(LLM)代理越来越多地被部署为会话助手，能够通过工具集成执行复杂的现实任务。这种增强的与外部系统交互和处理各种数据源的能力，虽然功能强大，但也引入了严重的安全漏洞。特别是，间接提示注入攻击构成了严重威胁，其中嵌入外部数据源的恶意指令可以操纵代理程序偏离用户意图。尽管基于规则限制、源聚焦和身份验证协议的现有防御措施前景看好，但它们难以在保持任务功能的同时保持强大的安全性。我们提出了一种新的、正交的观点，它将代理安全从防止有害操作重新定义为确保任务对齐，要求每个代理操作都服务于用户目标。基于这一认识，我们开发了任务盾，这是一种测试时间防御机制，它系统地验证每个指令和工具调用是否有助于实现用户指定的目标。通过在AgentDojo基准上的实验，我们证明了任务盾在降低攻击成功率(2.07\%)的同时，在GPT-40上保持了较高的任务利用率(69.79\%)。



## **47. POEX: Policy Executable Embodied AI Jailbreak Attacks**

POEX：政策可执行性许可人工智能越狱攻击 cs.RO

Homepage: https://poex-eai-jailbreak.github.io/

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2412.16633v1) [paper-pdf](http://arxiv.org/pdf/2412.16633v1)

**Authors**: Xuancun Lu, Zhengxian Huang, Xinfeng Li, Xiaoyu ji, Wenyuan Xu

**Abstract**: The integration of large language models (LLMs) into the planning module of Embodied Artificial Intelligence (Embodied AI) systems has greatly enhanced their ability to translate complex user instructions into executable policies. In this paper, we demystified how traditional LLM jailbreak attacks behave in the Embodied AI context. We conducted a comprehensive safety analysis of the LLM-based planning module of embodied AI systems against jailbreak attacks. Using the carefully crafted Harmful-RLbench, we accessed 20 open-source and proprietary LLMs under traditional jailbreak attacks, and highlighted two key challenges when adopting the prior jailbreak techniques to embodied AI contexts: (1) The harmful text output by LLMs does not necessarily induce harmful policies in Embodied AI context, and (2) even we can generate harmful policies, we have to guarantee they are executable in practice. To overcome those challenges, we propose Policy Executable (POEX) jailbreak attacks, where harmful instructions and optimized suffixes are injected into LLM-based planning modules, leading embodied AI to perform harmful actions in both simulated and physical environments. Our approach involves constraining adversarial suffixes to evade detection and fine-tuning a policy evaluater to improve the executability of harmful policies. We conducted extensive experiments on both a robotic arm embodied AI platform and simulators, to validate the attack and policy success rates on 136 harmful instructions from Harmful-RLbench. Our findings expose serious safety vulnerabilities in LLM-based planning modules, including the ability of POEX to be transferred across models. Finally, we propose mitigation strategies, such as safety-constrained prompts, pre- and post-planning checks, to address these vulnerabilities and ensure the safe deployment of embodied AI in real-world settings.

摘要: 将大型语言模型(LLM)集成到嵌入式人工智能(Embedded AI)系统的规划模块中，极大地增强了它们将复杂的用户指令转换为可执行策略的能力。在这篇文章中，我们揭开了传统的LLM越狱攻击在具体的人工智能上下文中的行为。我们对基于LLM的具体化人工智能系统抗越狱攻击规划模块进行了全面的安全分析。使用精心制作的有害RLbench，我们在传统越狱攻击下访问了20个开源和专有的LLM，并强调了采用先前的越狱技术来体现AI上下文时的两个关键挑战：(1)LLMS输出的有害文本不一定会导致体现AI上下文中的有害策略，以及(2)即使我们可以生成有害策略，我们也必须确保它们在实践中是可执行的。为了克服这些挑战，我们提出了策略可执行(POEX)越狱攻击，将有害指令和优化后缀注入基于LLM的规划模块，导致嵌入式AI在模拟和物理环境中执行有害操作。我们的方法包括限制敌意后缀以逃避检测，以及微调策略评估器以提高有害策略的可执行性。我们在一个机械臂体现的人工智能平台和模拟器上进行了广泛的实验，以验证对来自有害RLbench的136条有害指令的攻击和策略成功率。我们的发现暴露了基于LLM的计划模块中的严重安全漏洞，包括POEX跨模型传输的能力。最后，我们提出了缓解策略，如安全约束提示，规划前和规划后检查，以应对这些漏洞，并确保体现的人工智能在现实世界中的安全部署。



## **48. Automated Progressive Red Teaming**

自动化渐进式红色团队 cs.CR

Accepted by COLING 2025

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2407.03876v3) [paper-pdf](http://arxiv.org/pdf/2407.03876v3)

**Authors**: Bojian Jiang, Yi Jing, Tianhao Shen, Tong Wu, Qing Yang, Deyi Xiong

**Abstract**: Ensuring the safety of large language models (LLMs) is paramount, yet identifying potential vulnerabilities is challenging. While manual red teaming is effective, it is time-consuming, costly and lacks scalability. Automated red teaming (ART) offers a more cost-effective alternative, automatically generating adversarial prompts to expose LLM vulnerabilities. However, in current ART efforts, a robust framework is absent, which explicitly frames red teaming as an effectively learnable task. To address this gap, we propose Automated Progressive Red Teaming (APRT) as an effectively learnable framework. APRT leverages three core modules: an Intention Expanding LLM that generates diverse initial attack samples, an Intention Hiding LLM that crafts deceptive prompts, and an Evil Maker to manage prompt diversity and filter ineffective samples. The three modules collectively and progressively explore and exploit LLM vulnerabilities through multi-round interactions. In addition to the framework, we further propose a novel indicator, Attack Effectiveness Rate (AER) to mitigate the limitations of existing evaluation metrics. By measuring the likelihood of eliciting unsafe but seemingly helpful responses, AER aligns closely with human evaluations. Extensive experiments with both automatic and human evaluations, demonstrate the effectiveness of ARPT across both open- and closed-source LLMs. Specifically, APRT effectively elicits 54% unsafe yet useful responses from Meta's Llama-3-8B-Instruct, 50% from GPT-4o (API access), and 39% from Claude-3.5 (API access), showcasing its robust attack capability and transferability across LLMs (especially from open-source LLMs to closed-source LLMs).

摘要: 确保大型语言模型(LLM)的安全是最重要的，但识别潜在的漏洞是具有挑战性的。虽然手动红色团队是有效的，但它耗时、成本高，而且缺乏可扩展性。自动红色团队(ART)提供了一种更具成本效益的替代方案，可自动生成敌意提示以暴露LLM漏洞。然而，在目前的艺术努力中，缺乏一个强大的框架，它明确地将红色团队作为一项有效的可学习任务。为了弥补这一差距，我们提出了自动渐进红色团队(APRT)作为一种有效的可学习框架。APRT利用三个核心模块：用于生成不同初始攻击样本的意图扩展LLM，用于制作欺骗性提示的意图隐藏LLM，以及用于管理提示多样性和过滤无效样本的邪恶制造者。这三个模块通过多轮交互共同逐步探索和利用LLM漏洞。除了该框架外，我们进一步提出了一个新的指标--攻击效率(AER)，以缓解现有评估指标的局限性。通过衡量引发不安全但似乎有帮助的反应的可能性，AER与人类的评估密切一致。自动和人工评估的广泛实验证明了ARPT在开放源码和封闭源码LLM中的有效性。具体地说，APRT有效地从Meta的Llama-3-8B-Indict、GPT-40(API访问)和Claude-3.5(API访问)中引发了54%的不安全但有用的响应，展示了其强大的攻击能力和跨LLM(特别是从开源LLM到闭源LLM)的可转移性。



## **49. Divide and Conquer: A Hybrid Strategy Defeats Multimodal Large Language Models**

分而治之：击败多模式大型语言模型的混合策略 cs.CL

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2412.16555v1) [paper-pdf](http://arxiv.org/pdf/2412.16555v1)

**Authors**: Yanxu Mao, Peipei Liu, Tiehan Cui, Congying Liu, Datao You

**Abstract**: Large language models (LLMs) are widely applied in various fields of society due to their powerful reasoning, understanding, and generation capabilities. However, the security issues associated with these models are becoming increasingly severe. Jailbreaking attacks, as an important method for detecting vulnerabilities in LLMs, have been explored by researchers who attempt to induce these models to generate harmful content through various attack methods. Nevertheless, existing jailbreaking methods face numerous limitations, such as excessive query counts, limited coverage of jailbreak modalities, low attack success rates, and simplistic evaluation methods. To overcome these constraints, this paper proposes a multimodal jailbreaking method: JMLLM. This method integrates multiple strategies to perform comprehensive jailbreak attacks across text, visual, and auditory modalities. Additionally, we contribute a new and comprehensive dataset for multimodal jailbreaking research: TriJail, which includes jailbreak prompts for all three modalities. Experiments on the TriJail dataset and the benchmark dataset AdvBench, conducted on 13 popular LLMs, demonstrate advanced attack success rates and significant reduction in time overhead.

摘要: 大语言模型因其强大的推理、理解和生成能力而被广泛应用于社会的各个领域。然而，与这些模型相关的安全问题正变得越来越严重。越狱攻击作为检测LLMS漏洞的一种重要方法，已经被研究人员探索，他们试图通过各种攻击方法诱导这些模型产生有害内容。然而，现有的越狱方法面临着许多局限性，如过多的询问计数、有限的越狱模式覆盖范围、低攻击成功率和过于简单的评估方法。为了克服这些限制，本文提出了一种多通道越狱方法：JMLLM。这种方法集成了多种策略来执行跨文本、视觉和听觉模式的全面越狱攻击。此外，我们还为多模式越狱研究贡献了一个新的、全面的数据集：TriJail，其中包括所有三种模式的越狱提示。在TriJail数据集和基准数据集AdvBch上进行的实验表明，在13个流行的LLM上进行的攻击成功率更高，时间开销显著减少。



## **50. Privacy in Fine-tuning Large Language Models: Attacks, Defenses, and Future Directions**

微调大型语言模型中的隐私：攻击、防御和未来方向 cs.AI

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2412.16504v1) [paper-pdf](http://arxiv.org/pdf/2412.16504v1)

**Authors**: Hao Du, Shang Liu, Lele Zheng, Yang Cao, Atsuyoshi Nakamura, Lei Chen

**Abstract**: Fine-tuning has emerged as a critical process in leveraging Large Language Models (LLMs) for specific downstream tasks, enabling these models to achieve state-of-the-art performance across various domains. However, the fine-tuning process often involves sensitive datasets, introducing privacy risks that exploit the unique characteristics of this stage. In this paper, we provide a comprehensive survey of privacy challenges associated with fine-tuning LLMs, highlighting vulnerabilities to various privacy attacks, including membership inference, data extraction, and backdoor attacks. We further review defense mechanisms designed to mitigate privacy risks in the fine-tuning phase, such as differential privacy, federated learning, and knowledge unlearning, discussing their effectiveness and limitations in addressing privacy risks and maintaining model utility. By identifying key gaps in existing research, we highlight challenges and propose directions to advance the development of privacy-preserving methods for fine-tuning LLMs, promoting their responsible use in diverse applications.

摘要: 在为特定的下游任务利用大型语言模型(LLM)时，微调已成为一个关键过程，使这些模型能够在各个领域实现最先进的性能。然而，微调过程往往涉及敏感的数据集，从而引入隐私风险，从而利用这一阶段的独特特征。在本文中，我们提供了与微调LLMS相关的隐私挑战的全面调查，重点介绍了各种隐私攻击的漏洞，包括成员资格推断、数据提取和后门攻击。我们进一步回顾了在微调阶段为降低隐私风险而设计的防御机制，如差异隐私、联合学习和知识遗忘，讨论了它们在应对隐私风险和维护模型效用方面的有效性和局限性。通过找出现有研究中的关键差距，我们强调了挑战，并提出了方向，以推进微调LLM的隐私保护方法的开发，促进它们在不同应用中的负责任使用。



