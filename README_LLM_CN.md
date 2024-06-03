# Latest Large Language Model Attack Papers
**update at 2024-06-03 11:55:18**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Improved Techniques for Optimization-Based Jailbreaking on Large Language Models**

基于优化的大型语言模型越狱改进技术 cs.LG

**SubmitDate**: 2024-05-31    [abs](http://arxiv.org/abs/2405.21018v1) [paper-pdf](http://arxiv.org/pdf/2405.21018v1)

**Authors**: Xiaojun Jia, Tianyu Pang, Chao Du, Yihao Huang, Jindong Gu, Yang Liu, Xiaochun Cao, Min Lin

**Abstract**: Large language models (LLMs) are being rapidly developed, and a key component of their widespread deployment is their safety-related alignment. Many red-teaming efforts aim to jailbreak LLMs, where among these efforts, the Greedy Coordinate Gradient (GCG) attack's success has led to a growing interest in the study of optimization-based jailbreaking techniques. Although GCG is a significant milestone, its attacking efficiency remains unsatisfactory. In this paper, we present several improved (empirical) techniques for optimization-based jailbreaks like GCG. We first observe that the single target template of "Sure" largely limits the attacking performance of GCG; given this, we propose to apply diverse target templates containing harmful self-suggestion and/or guidance to mislead LLMs. Besides, from the optimization aspects, we propose an automatic multi-coordinate updating strategy in GCG (i.e., adaptively deciding how many tokens to replace in each step) to accelerate convergence, as well as tricks like easy-to-hard initialisation. Then, we combine these improved technologies to develop an efficient jailbreak method, dubbed $\mathcal{I}$-GCG. In our experiments, we evaluate on a series of benchmarks (such as NeurIPS 2023 Red Teaming Track). The results demonstrate that our improved techniques can help GCG outperform state-of-the-art jailbreaking attacks and achieve nearly 100% attack success rate. The code is released at https://github.com/jiaxiaojunQAQ/I-GCG.

摘要: 大型语言模型(LLM)正在迅速开发，其广泛部署的一个关键组件是与安全相关的一致性。许多红色团队的目标是越狱LLM，其中贪婪坐标梯度(GCG)攻击的成功导致了人们对基于优化的越狱技术的研究越来越感兴趣。虽然GCG是一个重要的里程碑，但其攻击效率仍然不能令人满意。在这篇文章中，我们提出了几种改进的(经验)技术，用于基于优化的越狱，如GCG。我们首先观察到单一目标模板“Sure”在很大程度上限制了GCG的攻击性能；鉴于此，我们建议使用包含有害自我暗示和/或引导的不同目标模板来误导LLM。此外，在优化方面，我们提出了GCG中的自动多坐标更新策略(即自适应地决定每一步需要替换多少个令牌)来加速收敛，以及容易初始化等技巧。然后，我们将这些改进的技术结合起来，开发出一种高效的越狱方法，称为$\mathcal{i}$-GCG。在我们的实验中，我们在一系列基准(例如NeurIPS 2023 Red Teaming Track)上进行了评估。结果表明，改进后的技术可以帮助GCG超越最先进的越狱攻击，并获得近100%的攻击成功率。该代码在https://github.com/jiaxiaojunQAQ/I-GCG.上发布



## **2. TrojanRAG: Retrieval-Augmented Generation Can Be Backdoor Driver in Large Language Models**

TrojanRAG：检索增强生成可以成为大型语言模型中的后门驱动程序 cs.CR

19 pages, 14 figures, 4 tables

**SubmitDate**: 2024-05-31    [abs](http://arxiv.org/abs/2405.13401v3) [paper-pdf](http://arxiv.org/pdf/2405.13401v3)

**Authors**: Pengzhou Cheng, Yidong Ding, Tianjie Ju, Zongru Wu, Wei Du, Ping Yi, Zhuosheng Zhang, Gongshen Liu

**Abstract**: Large language models (LLMs) have raised concerns about potential security threats despite performing significantly in Natural Language Processing (NLP). Backdoor attacks initially verified that LLM is doing substantial harm at all stages, but the cost and robustness have been criticized. Attacking LLMs is inherently risky in security review, while prohibitively expensive. Besides, the continuous iteration of LLMs will degrade the robustness of backdoors. In this paper, we propose TrojanRAG, which employs a joint backdoor attack in the Retrieval-Augmented Generation, thereby manipulating LLMs in universal attack scenarios. Specifically, the adversary constructs elaborate target contexts and trigger sets. Multiple pairs of backdoor shortcuts are orthogonally optimized by contrastive learning, thus constraining the triggering conditions to a parameter subspace to improve the matching. To improve the recall of the RAG for the target contexts, we introduce a knowledge graph to construct structured data to achieve hard matching at a fine-grained level. Moreover, we normalize the backdoor scenarios in LLMs to analyze the real harm caused by backdoors from both attackers' and users' perspectives and further verify whether the context is a favorable tool for jailbreaking models. Extensive experimental results on truthfulness, language understanding, and harmfulness show that TrojanRAG exhibits versatility threats while maintaining retrieval capabilities on normal queries.

摘要: 尽管大型语言模型(LLM)在自然语言处理(NLP)中表现出色，但仍引发了人们对潜在安全威胁的担忧。后门攻击最初证实了LLM在所有阶段都在造成实质性的危害，但其成本和健壮性受到了批评。在安全审查中，攻击LLMS固有的风险，同时代价高得令人望而却步。此外，LLMS的连续迭代会降低后门的健壮性。在本文中，我们提出了TrojanRAG，它在检索-增强生成中使用联合后门攻击，从而在通用攻击场景下操纵LLMS。具体地说，对手构建了精心设计的目标上下文和触发集。通过对比学习对多对后门捷径进行正交化优化，从而将触发条件约束到一个参数子空间以提高匹配性。为了提高RAG对目标上下文的查全率，我们引入了知识图来构建结构化数据，以实现细粒度的硬匹配。此外，我们对LLMS中的后门场景进行了规范化，从攻击者和用户的角度分析了后门造成的真实危害，并进一步验证了上下文是否为越狱模型的有利工具。在真实性、语言理解和危害性方面的大量实验结果表明，TrojanRAG在保持对正常查询的检索能力的同时，表现出通用性威胁。



## **3. Preemptive Answer "Attacks" on Chain-of-Thought Reasoning**

先发制人的回答“攻击”思维链推理 cs.CL

Accepted to ACL'24 (Findings). Camera-ready version

**SubmitDate**: 2024-05-31    [abs](http://arxiv.org/abs/2405.20902v1) [paper-pdf](http://arxiv.org/pdf/2405.20902v1)

**Authors**: Rongwu Xu, Zehan Qi, Wei Xu

**Abstract**: Large language models (LLMs) showcase impressive reasoning capabilities when coupled with Chain-of-Thought (CoT) prompting. However, the robustness of this approach warrants further investigation. In this paper, we introduce a novel scenario termed preemptive answers, where the LLM obtains an answer before engaging in reasoning. This situation can arise inadvertently or induced by malicious users by prompt injection attacks. Experiments reveal that preemptive answers significantly impair the model's reasoning capability across various CoT methods and a broad spectrum of datasets. To bolster the robustness of reasoning, we propose two measures aimed at mitigating this issue to some extent.

摘要: 当与思想链（CoT）提示相结合时，大型语言模型（LLM）展示了令人印象深刻的推理能力。然而，这种方法的稳健性值得进一步研究。在本文中，我们引入了一种称为先发制人答案的新颖场景，其中LLM在进行推理之前获得答案。这种情况可能是无意中发生的，也可能是恶意用户通过提示注入攻击引起的。实验表明，先发制人的答案显着损害了模型在各种CoT方法和广泛数据集中的推理能力。为了增强推理的稳健性，我们提出了两项旨在在一定程度上缓解这一问题的措施。



## **4. Enhancing Jailbreak Attack Against Large Language Models through Silent Tokens**

通过无声令牌增强针对大型语言模型的越狱攻击 cs.AI

**SubmitDate**: 2024-05-31    [abs](http://arxiv.org/abs/2405.20653v1) [paper-pdf](http://arxiv.org/pdf/2405.20653v1)

**Authors**: Jiahao Yu, Haozheng Luo, Jerry Yao-Chieh, Wenbo Guo, Han Liu, Xinyu Xing

**Abstract**: Along with the remarkable successes of Language language models, recent research also started to explore the security threats of LLMs, including jailbreaking attacks. Attackers carefully craft jailbreaking prompts such that a target LLM will respond to the harmful question. Existing jailbreaking attacks require either human experts or leveraging complicated algorithms to craft jailbreaking prompts. In this paper, we introduce BOOST, a simple attack that leverages only the eos tokens. We demonstrate that rather than constructing complicated jailbreaking prompts, the attacker can simply append a few eos tokens to the end of a harmful question. It will bypass the safety alignment of LLMs and lead to successful jailbreaking attacks. We further apply BOOST to four representative jailbreak methods and show that the attack success rates of these methods can be significantly enhanced by simply adding eos tokens to the prompt. To understand this simple but novel phenomenon, we conduct empirical analyses. Our analysis reveals that adding eos tokens makes the target LLM believe the input is much less harmful, and eos tokens have low attention values and do not affect LLM's understanding of the harmful questions, leading the model to actually respond to the questions. Our findings uncover how fragile an LLM is against jailbreak attacks, motivating the development of strong safety alignment approaches.

摘要: 在语言模型取得显著成功的同时，最近的研究也开始探索LLMS的安全威胁，包括越狱攻击。攻击者精心设计越狱提示，以便目标LLM会对这个有害的问题做出回应。现有的越狱攻击要么需要人类专家，要么需要利用复杂的算法来设计越狱提示。在本文中，我们将介绍Boost，这是一种仅利用Eos令牌的简单攻击。我们演示了，攻击者可以简单地在有害问题的末尾添加几个Eos令牌，而不是构建复杂的越狱提示。它将绕过LLMS的安全对准，并导致成功的越狱攻击。我们进一步将Boost应用于四种具有代表性的越狱方法，并表明只需在提示符中添加Eos令牌即可显着提高这些方法的攻击成功率。为了理解这一简单但新颖的现象，我们进行了实证分析。我们的分析表明，添加Eos标记会使目标LLM认为输入的危害要小得多，并且Eos标记的关注值较低，不会影响LLM对有害问题的理解，从而导致模型实际回答问题。我们的发现揭示了LLM对越狱攻击的脆弱性，促使了强大的安全对齐方法的发展。



## **5. Robustifying Safety-Aligned Large Language Models through Clean Data Curation**

通过干净的数据修复来优化安全一致的大型语言模型 cs.CR

**SubmitDate**: 2024-05-31    [abs](http://arxiv.org/abs/2405.19358v2) [paper-pdf](http://arxiv.org/pdf/2405.19358v2)

**Authors**: Xiaoqun Liu, Jiacheng Liang, Muchao Ye, Zhaohan Xi

**Abstract**: Large language models (LLMs) are vulnerable when trained on datasets containing harmful content, which leads to potential jailbreaking attacks in two scenarios: the integration of harmful texts within crowdsourced data used for pre-training and direct tampering with LLMs through fine-tuning. In both scenarios, adversaries can compromise the safety alignment of LLMs, exacerbating malfunctions. Motivated by the need to mitigate these adversarial influences, our research aims to enhance safety alignment by either neutralizing the impact of malicious texts in pre-training datasets or increasing the difficulty of jailbreaking during downstream fine-tuning. In this paper, we propose a data curation framework designed to counter adversarial impacts in both scenarios. Our method operates under the assumption that we have no prior knowledge of attack details, focusing solely on curating clean texts. We introduce an iterative process aimed at revising texts to reduce their perplexity as perceived by LLMs, while simultaneously preserving their text quality. By pre-training or fine-tuning LLMs with curated clean texts, we observe a notable improvement in LLM robustness regarding safety alignment against harmful queries. For instance, when pre-training LLMs using a crowdsourced dataset containing 5\% harmful instances, adding an equivalent amount of curated texts significantly mitigates the likelihood of providing harmful responses in LLMs and reduces the attack success rate by 71\%. Our study represents a significant step towards mitigating the risks associated with training-based jailbreaking and fortifying the secure utilization of LLMs.

摘要: 在包含有害内容的数据集上进行训练时，大型语言模型(LLM)很容易受到攻击，这会在两种情况下导致潜在的越狱攻击：将有害文本整合到用于预培训的众包数据中，以及通过微调直接篡改LLMS。在这两种情况下，对手都可能损害LLM的安全对准，从而加剧故障。出于缓解这些敌对影响的需要，我们的研究旨在通过中和预训练数据集中恶意文本的影响或在下游微调期间增加越狱的难度来增强安全一致性。在本文中，我们提出了一个数据管理框架，旨在对抗这两种情况下的对抗性影响。我们的方法是在假设我们事先不知道攻击细节的情况下运行的，只专注于策划干净的文本。我们引入了一种迭代过程，旨在修改文本以减少LLMS所感知的困惑，同时保持其文本质量。通过使用经过精选的干净文本预先训练或微调LLM，我们观察到LLM在针对有害查询的安全对齐方面的稳健性有了显著的改善。例如，当使用包含5个有害实例的众包数据集对LLMS进行预训练时，添加等量的精选文本可显著降低LLMS中提供有害响应的可能性，并将攻击成功率降低71%。我们的研究是朝着减少基于培训的越狱风险和加强低土地管理系统的安全利用迈出的重要一步。



## **6. Phantom: General Trigger Attacks on Retrieval Augmented Language Generation**

Phantom：对检索增强语言生成的通用触发攻击 cs.CR

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.20485v1) [paper-pdf](http://arxiv.org/pdf/2405.20485v1)

**Authors**: Harsh Chaudhari, Giorgio Severi, John Abascal, Matthew Jagielski, Christopher A. Choquette-Choo, Milad Nasr, Cristina Nita-Rotaru, Alina Oprea

**Abstract**: Retrieval Augmented Generation (RAG) expands the capabilities of modern large language models (LLMs) in chatbot applications, enabling developers to adapt and personalize the LLM output without expensive training or fine-tuning. RAG systems use an external knowledge database to retrieve the most relevant documents for a given query, providing this context to the LLM generator. While RAG achieves impressive utility in many applications, its adoption to enable personalized generative models introduces new security risks. In this work, we propose new attack surfaces for an adversary to compromise a victim's RAG system, by injecting a single malicious document in its knowledge database. We design Phantom, general two-step attack framework against RAG augmented LLMs. The first step involves crafting a poisoned document designed to be retrieved by the RAG system within the top-k results only when an adversarial trigger, a specific sequence of words acting as backdoor, is present in the victim's queries. In the second step, a specially crafted adversarial string within the poisoned document triggers various adversarial attacks in the LLM generator, including denial of service, reputation damage, privacy violations, and harmful behaviors. We demonstrate our attacks on multiple LLM architectures, including Gemma, Vicuna, and Llama.

摘要: 检索增强生成(RAG)扩展了Chatbot应用程序中现代大型语言模型(LLM)的能力，使开发人员能够适应和个性化LLM输出，而无需昂贵的培训或微调。RAG系统使用外部知识数据库来检索与给定查询最相关的文档，并将此上下文提供给LLM生成器。虽然RAG在许多应用程序中实现了令人印象深刻的实用性，但采用它来支持个性化的生成模型带来了新的安全风险。在这项工作中，我们提出了新的攻击面，通过在受害者的知识库中注入单个恶意文档来危害受害者的RAG系统。我们设计了一个针对RAG扩展的LLMS的Phantom通用两步攻击框架。第一步涉及精心设计一个有毒文档，仅当受害者的查询中出现敌对触发器(充当后门的特定单词序列)时，RAG系统才会在top-k结果中检索到。在第二步中，有毒文档中巧尽心思构建的敌意字符串会在LLM生成器中触发各种敌意攻击，包括拒绝服务、声誉损害、侵犯隐私和有害行为。我们演示了我们对多个LLM体系结构的攻击，包括Gema、Vicuna和Llama。



## **7. DepesRAG: Towards Managing Software Dependencies using Large Language Models**

DepesRAG：使用大型语言模型管理软件附属机构 cs.SE

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.20455v1) [paper-pdf](http://arxiv.org/pdf/2405.20455v1)

**Authors**: Mohannad Alhanahnah, Yazan Boshmaf, Benoit Baudry

**Abstract**: Managing software dependencies is a crucial maintenance task in software development and is becoming a rapidly growing research field, especially in light of the significant increase in software supply chain attacks. Specialized expertise and substantial developer effort are required to fully comprehend dependencies and reveal hidden properties about the dependencies (e.g., number of dependencies, dependency chains, depth of dependencies).   Recent advancements in Large Language Models (LLMs) allow the retrieval of information from various data sources for response generation, thus providing a new opportunity to uniquely manage software dependencies. To highlight the potential of this technology, we present~\tool, a proof-of-concept Retrieval Augmented Generation (RAG) approach that constructs direct and transitive dependencies of software packages as a Knowledge Graph (KG) in four popular software ecosystems. DepsRAG can answer user questions about software dependencies by automatically generating necessary queries to retrieve information from the KG, and then augmenting the input of LLMs with the retrieved information. DepsRAG can also perform Web search to answer questions that the LLM cannot directly answer via the KG. We identify tangible benefits that DepsRAG can offer and discuss its limitations.

摘要: 管理软件依赖关系是软件开发中一项重要的维护任务，特别是在软件供应链攻击显著增加的情况下，软件依赖关系管理正在成为一个快速增长的研究领域。要完全理解依赖关系并揭示依赖关系的隐藏属性(例如，依赖关系的数量、依赖关系链、依赖关系的深度)，需要专业的专业知识和大量的开发人员工作。大型语言模型(LLM)的最新进展允许从各种数据源检索信息以生成响应，从而为独特地管理软件依赖关系提供了新的机会。为了突出这一技术的潜力，我们提出了一种概念验证检索增强生成(RAG)方法~\Tool，它将软件包的直接和传递依赖构造为四个流行的软件生态系统中的知识图(KG)。DepsRAG可以通过自动生成必要的查询来从KG中检索信息，然后用检索到的信息增强LLMS的输入，来回答用户关于软件依赖性的问题。DepsRAG还可以执行网络搜索，以回答LLM无法通过KG直接回答的问题。我们确定了DepsRAG可以提供的实实在在的好处并讨论了其局限性。



## **8. Jailbreaking Large Language Models Against Moderation Guardrails via Cipher Characters**

通过密码字符破解大型语言模型对抗调节护栏 cs.CR

20 pages

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.20413v1) [paper-pdf](http://arxiv.org/pdf/2405.20413v1)

**Authors**: Haibo Jin, Andy Zhou, Joe D. Menke, Haohan Wang

**Abstract**: Large Language Models (LLMs) are typically harmless but remain vulnerable to carefully crafted prompts known as ``jailbreaks'', which can bypass protective measures and induce harmful behavior. Recent advancements in LLMs have incorporated moderation guardrails that can filter outputs, which trigger processing errors for certain malicious questions. Existing red-teaming benchmarks often neglect to include questions that trigger moderation guardrails, making it difficult to evaluate jailbreak effectiveness. To address this issue, we introduce JAMBench, a harmful behavior benchmark designed to trigger and evaluate moderation guardrails. JAMBench involves 160 manually crafted instructions covering four major risk categories at multiple severity levels. Furthermore, we propose a jailbreak method, JAM (Jailbreak Against Moderation), designed to attack moderation guardrails using jailbreak prefixes to bypass input-level filters and a fine-tuned shadow model functionally equivalent to the guardrail model to generate cipher characters to bypass output-level filters. Our extensive experiments on four LLMs demonstrate that JAM achieves higher jailbreak success ($\sim$ $\times$ 19.88) and lower filtered-out rates ($\sim$ $\times$ 1/6) than baselines.

摘要: 大型语言模型(LLM)通常是无害的，但仍然容易受到精心设计的称为“越狱”的提示的攻击，这些提示可能会绕过保护措施并引发有害行为。LLMS中最近的改进包括了可以过滤输出的适度防护，这会触发对某些恶意问题的处理错误。现有的红团队基准往往忽视了包括引发温和障碍的问题，这使得评估越狱的有效性变得困难。为了解决这个问题，我们引入了JAMBch，这是一个旨在触发和评估适度护栏的有害行为基准。JAMBtch涉及160个手动编写的说明，涵盖多个严重级别的四个主要风险类别。此外，我们提出了一种越狱方法JAM(JailBreak Against Medium Ation)，旨在使用越狱前缀来攻击适度护栏，以绕过输入级过滤器，以及一个功能等价于护栏模型的微调阴影模型，以生成密码字符以绕过输出级过滤器。我们在四个LLM上的广泛实验表明，JAM获得了比基线更高的越狱成功率($\sim$$\x$19.88)和更低的过滤成功率($\sim$$\x$1/6)。



## **9. Context Injection Attacks on Large Language Models**

对大型语言模型的上下文注入攻击 cs.AI

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.20234v1) [paper-pdf](http://arxiv.org/pdf/2405.20234v1)

**Authors**: Cheng'an Wei, Kai Chen, Yue Zhao, Yujia Gong, Lu Xiang, Shenchen Zhu

**Abstract**: Large Language Models (LLMs) such as ChatGPT and Llama-2 have become prevalent in real-world applications, exhibiting impressive text generation performance. LLMs are fundamentally developed from a scenario where the input data remains static and lacks a clear structure. To behave interactively over time, LLM-based chat systems must integrate additional contextual information (i.e., chat history) into their inputs, following a pre-defined structure. This paper identifies how such integration can expose LLMs to misleading context from untrusted sources and fail to differentiate between system and user inputs, allowing users to inject context. We present a systematic methodology for conducting context injection attacks aimed at eliciting disallowed responses by introducing fabricated context. This could lead to illegal actions, inappropriate content, or technology misuse. Our context fabrication strategies, acceptance elicitation and word anonymization, effectively create misleading contexts that can be structured with attacker-customized prompt templates, achieving injection through malicious user messages. Comprehensive evaluations on real-world LLMs such as ChatGPT and Llama-2 confirm the efficacy of the proposed attack with success rates reaching 97%. We also discuss potential countermeasures that can be adopted for attack detection and developing more secure models. Our findings provide insights into the challenges associated with the real-world deployment of LLMs for interactive and structured data scenarios.

摘要: 大型语言模型(LLM)，如ChatGPT和Llama-2，已经在现实世界的应用程序中流行起来，表现出令人印象深刻的文本生成性能。LLM基本上是在输入数据保持静态且缺乏清晰结构的情况下开发出来的。为了随着时间的推移交互行为，基于LLM的聊天系统必须按照预定义的结构将附加的上下文信息(即聊天历史)集成到它们的输入中。这篇白皮书指出了这种集成如何将LLM暴露在来自不可信来源的误导性上下文中，并无法区分系统和用户输入，从而允许用户注入上下文。我们提出了一种系统的方法来进行上下文注入攻击，目的是通过引入捏造的上下文来引发不允许的响应。这可能会导致非法行为、不适当的内容或技术滥用。我们的上下文构建策略，接受诱导和单词匿名化，有效地创建了误导性上下文，可以使用攻击者定制的提示模板来构建，通过恶意用户消息实现注入。在ChatGPT和Llama-2等真实LLMS上的综合评估证实了该攻击的有效性，成功率达到97%。我们还讨论了可用于攻击检测和开发更安全模型的潜在对策。我们的发现为交互式和结构化数据场景中与LLMS的实际部署相关的挑战提供了见解。



## **10. Defensive Prompt Patch: A Robust and Interpretable Defense of LLMs against Jailbreak Attacks**

防御提示补丁：LLM针对越狱攻击的强大且可解释的防御 cs.CR

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.20099v1) [paper-pdf](http://arxiv.org/pdf/2405.20099v1)

**Authors**: Chen Xiong, Xiangyu Qi, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Safety, security, and compliance are essential requirements when aligning large language models (LLMs). However, many seemingly aligned LLMs are soon shown to be susceptible to jailbreak attacks. These attacks aim to circumvent the models' safety guardrails and security mechanisms by introducing jailbreak prompts into malicious queries. In response to these challenges, this paper introduces Defensive Prompt Patch (DPP), a novel prompt-based defense mechanism specifically designed to protect LLMs against such sophisticated jailbreak strategies. Unlike previous approaches, which have often compromised the utility of the model for the sake of safety, DPP is designed to achieve a minimal Attack Success Rate (ASR) while preserving the high utility of LLMs. Our method uses strategically designed interpretable suffix prompts that effectively thwart a wide range of standard and adaptive jailbreak techniques. Empirical results conducted on LLAMA-2-7B-Chat and Mistral-7B-Instruct-v0.2 models demonstrate the robustness and adaptability of DPP, showing significant reductions in ASR with negligible impact on utility. Our approach not only outperforms existing defense strategies in balancing safety and functionality, but also provides a scalable and interpretable solution applicable to various LLM platforms.

摘要: 安全性、安全性和合规性是调整大型语言模型(LLM)时的基本要求。然而，许多看似一致的低收入国家很快就被证明容易受到越狱攻击。这些攻击旨在通过在恶意查询中引入越狱提示来绕过模型的安全护栏和安全机制。为了应对这些挑战，本文引入了防御提示补丁(DPP)，这是一种新的基于提示的防御机制，专门设计来保护LLM免受这种复杂的越狱策略的攻击。与以前的方法不同，为了安全起见，DPP通常会损害模型的实用性，DPP旨在实现最小的攻击成功率(ASR)，同时保持LLMS的高可用性。我们的方法使用策略性设计的可解释后缀提示，有效地阻止了广泛的标准和自适应越狱技术。在Llama-2-7B-Chat和Mistral-7B-Indict-v0.2模型上进行的实证结果证明了DPP的稳健性和适应性，表明ASR显著降低，而对效用的影响可以忽略不计。我们的方法不仅在平衡安全性和功能性方面优于现有的防御策略，而且还提供了适用于各种LLM平台的可扩展和可解释的解决方案。



## **11. Typography Leads Semantic Diversifying: Amplifying Adversarial Transferability across Multimodal Large Language Models**

字体设计引领语义多元化：增强多模式大型语言模型之间的对抗性可移植性 cs.CV

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.20090v1) [paper-pdf](http://arxiv.org/pdf/2405.20090v1)

**Authors**: Hao Cheng, Erjia Xiao, Jiahang Cao, Le Yang, Kaidi Xu, Jindong Gu, Renjing Xu

**Abstract**: Following the advent of the Artificial Intelligence (AI) era of large models, Multimodal Large Language Models (MLLMs) with the ability to understand cross-modal interactions between vision and text have attracted wide attention. Adversarial examples with human-imperceptible perturbation are shown to possess a characteristic known as transferability, which means that a perturbation generated by one model could also mislead another different model. Augmenting the diversity in input data is one of the most significant methods for enhancing adversarial transferability. This method has been certified as a way to significantly enlarge the threat impact under black-box conditions. Research works also demonstrate that MLLMs can be exploited to generate adversarial examples in the white-box scenario. However, the adversarial transferability of such perturbations is quite limited, failing to achieve effective black-box attacks across different models. In this paper, we propose the Typographic-based Semantic Transfer Attack (TSTA), which is inspired by: (1) MLLMs tend to process semantic-level information; (2) Typographic Attack could effectively distract the visual information captured by MLLMs. In the scenarios of Harmful Word Insertion and Important Information Protection, our TSTA demonstrates superior performance.

摘要: 随着大模型人工智能时代的到来，能够理解视觉和文本之间跨通道交互的多通道大语言模型引起了人们的广泛关注。具有人类不可察觉的扰动的对抗性例子具有被称为可转移性的特征，这意味着一个模型产生的扰动也可能误导另一个不同的模型。增加输入数据的多样性是增强对抗性转移的最重要的方法之一。这种方法已被证明是一种在黑箱条件下显著扩大威胁影响的方法。研究工作还表明，在白盒情况下，MLLMS可以被用来生成对抗性示例。然而，此类扰动的对抗性可转移性相当有限，无法实现跨不同模型的有效黑盒攻击。本文提出了基于排版的语义传输攻击(TSTA)，其灵感来自：(1)MLLMS倾向于处理语义级的信息；(2)排版攻击可以有效地分散MLLMS捕获的视觉信息。在有害词语插入和重要信息保护的场景中，我们的TSTA表现出了卓越的性能。



## **12. Efficient LLM-Jailbreaking by Introducing Visual Modality**

通过引入视觉形态高效法学硕士越狱 cs.AI

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.20015v1) [paper-pdf](http://arxiv.org/pdf/2405.20015v1)

**Authors**: Zhenxing Niu, Yuyao Sun, Haodong Ren, Haoxuan Ji, Quan Wang, Xiaoke Ma, Gang Hua, Rong Jin

**Abstract**: This paper focuses on jailbreaking attacks against large language models (LLMs), eliciting them to generate objectionable content in response to harmful user queries. Unlike previous LLM-jailbreaks that directly orient to LLMs, our approach begins by constructing a multimodal large language model (MLLM) through the incorporation of a visual module into the target LLM. Subsequently, we conduct an efficient MLLM-jailbreak to generate jailbreaking embeddings embJS. Finally, we convert the embJS into text space to facilitate the jailbreaking of the target LLM. Compared to direct LLM-jailbreaking, our approach is more efficient, as MLLMs are more vulnerable to jailbreaking than pure LLM. Additionally, to improve the attack success rate (ASR) of jailbreaking, we propose an image-text semantic matching scheme to identify a suitable initial input. Extensive experiments demonstrate that our approach surpasses current state-of-the-art methods in terms of both efficiency and effectiveness. Moreover, our approach exhibits superior cross-class jailbreaking capabilities.

摘要: 本文的重点是针对大型语言模型(LLM)的越狱攻击，诱使它们生成令人反感的内容来响应有害的用户查询。与以前直接面向LLMS的LLM越狱不同，我们的方法首先通过将可视模块整合到目标LLM中来构建多通道大型语言模型(MLLM)。随后，我们进行了一个高效的MLLM-JailBreak来生成越狱嵌入embJS。最后，我们将embJS转换为文本空间，以便于目标LLM的越狱。与直接LLM越狱相比，我们的方法更有效，因为MLLM比纯粹的LLM更容易越狱。此外，为了提高越狱的攻击成功率，我们提出了一种图文语义匹配方案来识别合适的初始输入。大量的实验表明，我们的方法在效率和效果上都超过了目前最先进的方法。此外，我们的方法显示出卓越的跨阶层越狱能力。



## **13. Exploring the Robustness of Decision-Level Through Adversarial Attacks on LLM-Based Embodied Models**

通过对基于LLM的排队模型的对抗攻击探索决策级的鲁棒性 cs.MM

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.19802v1) [paper-pdf](http://arxiv.org/pdf/2405.19802v1)

**Authors**: Shuyuan Liu, Jiawei Chen, Shouwei Ruan, Hang Su, Zhaoxia Yin

**Abstract**: Embodied intelligence empowers agents with a profound sense of perception, enabling them to respond in a manner closely aligned with real-world situations. Large Language Models (LLMs) delve into language instructions with depth, serving a crucial role in generating plans for intricate tasks. Thus, LLM-based embodied models further enhance the agent's capacity to comprehend and process information. However, this amalgamation also ushers in new challenges in the pursuit of heightened intelligence. Specifically, attackers can manipulate LLMs to produce irrelevant or even malicious outputs by altering their prompts. Confronted with this challenge, we observe a notable absence of multi-modal datasets essential for comprehensively evaluating the robustness of LLM-based embodied models. Consequently, we construct the Embodied Intelligent Robot Attack Dataset (EIRAD), tailored specifically for robustness evaluation. Additionally, two attack strategies are devised, including untargeted attacks and targeted attacks, to effectively simulate a range of diverse attack scenarios. At the same time, during the attack process, to more accurately ascertain whether our method is successful in attacking the LLM-based embodied model, we devise a new attack success evaluation method utilizing the BLIP2 model. Recognizing the time and cost-intensive nature of the GCG algorithm in attacks, we devise a scheme for prompt suffix initialization based on various target tasks, thus expediting the convergence process. Experimental results demonstrate that our method exhibits a superior attack success rate when targeting LLM-based embodied models, indicating a lower level of decision-level robustness in these models.

摘要: 具身智能使特工具有深刻的感知力，使他们能够以与现实世界情况密切一致的方式做出反应。大型语言模型(LLM)深入研究语言指令，在为复杂任务制定计划方面发挥着至关重要的作用。因此，基于LLM的具体化模型进一步增强了代理理解和处理信息的能力。然而，这种融合也带来了追求高智商的新挑战。具体地说，攻击者可以通过更改提示来操纵LLMS生成无关甚至恶意的输出。面对这一挑战，我们注意到明显缺乏全面评估基于LLM的体现模型的稳健性所必需的多模式数据集。因此，我们构建了专门为健壮性评估量身定做的具体化智能机器人攻击数据集(Eirad)。此外，设计了两种攻击策略，包括非定向攻击和定向攻击，以有效地模拟一系列不同的攻击场景。同时，在攻击过程中，为了更准确地确定我们的方法在攻击基于LLM的体现模型上是否成功，我们设计了一种新的利用BLIP2模型的攻击成功评估方法。考虑到GCG算法在攻击中的时间和成本密集性，我们设计了一种基于不同目标任务的快速后缀初始化方案，从而加快了收敛过程。实验结果表明，我们的方法在攻击基于LLM的具体模型时表现出了较高的攻击成功率，表明这些模型具有较低的决策级健壮性。



## **14. Vocabulary Attack to Hijack Large Language Model Applications**

劫持大型语言模型应用程序的词汇攻击 cs.CR

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2404.02637v2) [paper-pdf](http://arxiv.org/pdf/2404.02637v2)

**Authors**: Patrick Levi, Christoph P. Neumann

**Abstract**: The fast advancements in Large Language Models (LLMs) are driving an increasing number of applications. Together with the growing number of users, we also see an increasing number of attackers who try to outsmart these systems. They want the model to reveal confidential information, specific false information, or offensive behavior. To this end, they manipulate their instructions for the LLM by inserting separators or rephrasing them systematically until they reach their goal. Our approach is different. It inserts words from the model vocabulary. We find these words using an optimization procedure and embeddings from another LLM (attacker LLM). We prove our approach by goal hijacking two popular open-source LLMs from the Llama2 and the Flan-T5 families, respectively. We present two main findings. First, our approach creates inconspicuous instructions and therefore it is hard to detect. For many attack cases, we find that even a single word insertion is sufficient. Second, we demonstrate that we can conduct our attack using a different model than the target model to conduct our attack with.

摘要: 大型语言模型(LLM)的快速发展正在推动越来越多的应用程序。随着用户数量的不断增加，我们也看到越来越多的攻击者试图智取这些系统。他们希望该模型能够泄露机密信息、特定的虚假信息或冒犯行为。为此，他们通过插入分隔符或系统地重新措辞来操纵他们对LLM的指令，直到达到他们的目标。我们的方法是不同的。它插入模型词汇表中的单词。我们使用优化过程和来自另一个LLM(攻击者LLM)的嵌入来找到这些单词。我们通过Goal劫持了分别来自Llama2和Flan-T5家族的两个流行的开源LLM来证明我们的方法。我们提出了两个主要发现。首先，我们的方法创建了不明显的指令，因此很难检测到。对于许多攻击情况，我们发现即使是一个单词插入也是足够的。其次，我们演示了我们可以使用与进行攻击的目标模型不同的模型来进行攻击。



## **15. Large Language Model Watermark Stealing With Mixed Integer Programming**

使用混合格式编程实现大语言模型水印窃取 cs.CR

12 pages

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.19677v1) [paper-pdf](http://arxiv.org/pdf/2405.19677v1)

**Authors**: Zhaoxi Zhang, Xiaomei Zhang, Yanjun Zhang, Leo Yu Zhang, Chao Chen, Shengshan Hu, Asif Gill, Shirui Pan

**Abstract**: The Large Language Model (LLM) watermark is a newly emerging technique that shows promise in addressing concerns surrounding LLM copyright, monitoring AI-generated text, and preventing its misuse. The LLM watermark scheme commonly includes generating secret keys to partition the vocabulary into green and red lists, applying a perturbation to the logits of tokens in the green list to increase their sampling likelihood, thus facilitating watermark detection to identify AI-generated text if the proportion of green tokens exceeds a threshold. However, recent research indicates that watermarking methods using numerous keys are susceptible to removal attacks, such as token editing, synonym substitution, and paraphrasing, with robustness declining as the number of keys increases. Therefore, the state-of-the-art watermark schemes that employ fewer or single keys have been demonstrated to be more robust against text editing and paraphrasing. In this paper, we propose a novel green list stealing attack against the state-of-the-art LLM watermark scheme and systematically examine its vulnerability to this attack. We formalize the attack as a mixed integer programming problem with constraints. We evaluate our attack under a comprehensive threat model, including an extreme scenario where the attacker has no prior knowledge, lacks access to the watermark detector API, and possesses no information about the LLM's parameter settings or watermark injection/detection scheme. Extensive experiments on LLMs, such as OPT and LLaMA, demonstrate that our attack can successfully steal the green list and remove the watermark across all settings.

摘要: 大语言模型(LLM)水印是一种新出现的技术，它在解决围绕LLM版权的担忧、监控人工智能生成的文本并防止其滥用方面显示出良好的前景。LLM水印方案通常包括生成密钥以将词汇表划分为绿色和红色列表，对绿色列表中的令牌的逻辑施加扰动以增加其采样可能性，从而在绿色令牌的比例超过阈值的情况下促进水印检测以识别AI生成的文本。然而，最近的研究表明，使用大量密钥的水印方法容易受到移除攻击，例如标记编辑、同义词替换和改述，并且随着密钥数量的增加，鲁棒性下降。因此，已证明采用较少密钥或单一密钥的最新水印方案对文本编辑和转译更健壮。本文针对现有的LLM水印方案，提出了一种新的绿名单窃取攻击方案，并对其脆弱性进行了系统的分析。我们将攻击形式化化为一个带约束的混合整数规划问题。我们在一个全面的威胁模型下评估我们的攻击，包括一个极端的场景，其中攻击者事先不知道，无法访问水印检测器API，并且没有关于LLM的参数设置或水印注入/检测方案的信息。在OPT和Llama等LLMS上的大量实验表明，我们的攻击可以成功地窃取绿色列表并删除所有设置的水印。



## **16. AutoBreach: Universal and Adaptive Jailbreaking with Efficient Wordplay-Guided Optimization**

AutoBreach：具有高效的文字游戏引导优化的通用和自适应越狱 cs.CV

Under review

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.19668v1) [paper-pdf](http://arxiv.org/pdf/2405.19668v1)

**Authors**: Jiawei Chen, Xiao Yang, Zhengwei Fang, Yu Tian, Yinpeng Dong, Zhaoxia Yin, Hang Su

**Abstract**: Despite the widespread application of large language models (LLMs) across various tasks, recent studies indicate that they are susceptible to jailbreak attacks, which can render their defense mechanisms ineffective. However, previous jailbreak research has frequently been constrained by limited universality, suboptimal efficiency, and a reliance on manual crafting. In response, we rethink the approach to jailbreaking LLMs and formally define three essential properties from the attacker' s perspective, which contributes to guiding the design of jailbreak methods. We further introduce AutoBreach, a novel method for jailbreaking LLMs that requires only black-box access. Inspired by the versatility of wordplay, AutoBreach employs a wordplay-guided mapping rule sampling strategy to generate a variety of universal mapping rules for creating adversarial prompts. This generation process leverages LLMs' automatic summarization and reasoning capabilities, thus alleviating the manual burden. To boost jailbreak success rates, we further suggest sentence compression and chain-of-thought-based mapping rules to correct errors and wordplay misinterpretations in target LLMs. Additionally, we propose a two-stage mapping rule optimization strategy that initially optimizes mapping rules before querying target LLMs to enhance the efficiency of AutoBreach. AutoBreach can efficiently identify security vulnerabilities across various LLMs, including three proprietary models: Claude-3, GPT-3.5, GPT-4 Turbo, and two LLMs' web platforms: Bingchat, GPT-4 Web, achieving an average success rate of over 80% with fewer than 10 queries

摘要: 尽管大型语言模型在各种任务中得到了广泛应用，但最近的研究表明，它们很容易受到越狱攻击，这会使它们的防御机制失效。然而，以前的越狱研究经常受到普适性有限、效率不佳以及对手工制作的依赖的限制。作为回应，我们重新思考了越狱LLM的方法，并从攻击者S的角度正式定义了三个基本性质，这有助于指导越狱方法的设计。我们进一步介绍了AutoBReach，这是一种新的越狱LLMS方法，只需要黑盒访问。受文字游戏的多样性启发，AutoBReach采用了文字游戏指导的映射规则采样策略，生成了各种通用的映射规则，用于创建对抗性提示。这一生成过程利用了LLMS的自动摘要和推理能力，从而减轻了手动负担。为了提高越狱成功率，我们进一步建议句子压缩和基于思想链的映射规则来纠正目标LLM中的错误和文字游戏误解。此外，我们还提出了一种两阶段映射规则优化策略，在查询目标LLM之前对映射规则进行初始优化，以提高AutoBReach的效率。AutoBReach可以高效地识别各种LLMS的安全漏洞，包括三种专有模型：Claude-3、GPT-3.5、GPT-4 Turbo，以及两种LLMS的Web平台：Bingchat、GPT-4 Web，平均成功率超过80%，查询次数不到10次



## **17. Unlearning Climate Misinformation in Large Language Models**

在大型语言模型中消除气候错误信息 cs.CL

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19563v1) [paper-pdf](http://arxiv.org/pdf/2405.19563v1)

**Authors**: Michael Fore, Simranjit Singh, Chaehong Lee, Amritanshu Pandey, Antonios Anastasopoulos, Dimitrios Stamoulis

**Abstract**: Misinformation regarding climate change is a key roadblock in addressing one of the most serious threats to humanity. This paper investigates factual accuracy in large language models (LLMs) regarding climate information. Using true/false labeled Q&A data for fine-tuning and evaluating LLMs on climate-related claims, we compare open-source models, assessing their ability to generate truthful responses to climate change questions. We investigate the detectability of models intentionally poisoned with false climate information, finding that such poisoning may not affect the accuracy of a model's responses in other domains. Furthermore, we compare the effectiveness of unlearning algorithms, fine-tuning, and Retrieval-Augmented Generation (RAG) for factually grounding LLMs on climate change topics. Our evaluation reveals that unlearning algorithms can be effective for nuanced conceptual claims, despite previous findings suggesting their inefficacy in privacy contexts. These insights aim to guide the development of more factually reliable LLMs and highlight the need for additional work to secure LLMs against misinformation attacks.

摘要: 关于气候变化的错误信息是解决人类面临的最严重威胁之一的关键障碍。本文研究了关于气候信息的大型语言模型(LLM)中的事实准确性。使用真/假标签的问答数据来微调和评估气候相关主张的最小二乘模型，我们比较了开源模型，评估了它们对气候变化问题做出真实回应的能力。我们调查了故意被虚假气候信息毒害的模型的可检测性，发现这种毒化可能不会影响模型在其他领域的响应的准确性。此外，我们比较了遗忘算法、微调和检索-增强生成(RAG)算法在气候变化主题上实际建立最小二乘模型的有效性。我们的评估表明，遗忘算法可以有效地处理细微差别的概念声明，尽管之前的研究结果表明，它们在隐私环境中无效。这些见解旨在指导开发更真实可靠的LLM，并强调需要开展更多工作来确保LLM免受错误信息攻击。



## **18. Voice Jailbreak Attacks Against GPT-4o**

针对GPT-4 o的语音越狱攻击 cs.CR

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19103v1) [paper-pdf](http://arxiv.org/pdf/2405.19103v1)

**Authors**: Xinyue Shen, Yixin Wu, Michael Backes, Yang Zhang

**Abstract**: Recently, the concept of artificial assistants has evolved from science fiction into real-world applications. GPT-4o, the newest multimodal large language model (MLLM) across audio, vision, and text, has further blurred the line between fiction and reality by enabling more natural human-computer interactions. However, the advent of GPT-4o's voice mode may also introduce a new attack surface. In this paper, we present the first systematic measurement of jailbreak attacks against the voice mode of GPT-4o. We show that GPT-4o demonstrates good resistance to forbidden questions and text jailbreak prompts when directly transferring them to voice mode. This resistance is primarily due to GPT-4o's internal safeguards and the difficulty of adapting text jailbreak prompts to voice mode. Inspired by GPT-4o's human-like behaviors, we propose VoiceJailbreak, a novel voice jailbreak attack that humanizes GPT-4o and attempts to persuade it through fictional storytelling (setting, character, and plot). VoiceJailbreak is capable of generating simple, audible, yet effective jailbreak prompts, which significantly increases the average attack success rate (ASR) from 0.033 to 0.778 in six forbidden scenarios. We also conduct extensive experiments to explore the impacts of interaction steps, key elements of fictional writing, and different languages on VoiceJailbreak's effectiveness and further enhance the attack performance with advanced fictional writing techniques. We hope our study can assist the research community in building more secure and well-regulated MLLMs.

摘要: 最近，人工助手的概念已经从科幻小说演变成现实世界的应用。GPT-40是最新的跨音频、视觉和文本的多模式大型语言模型(MLLM)，通过实现更自然的人机交互，进一步模糊了虚构和现实之间的界限。然而，GPT-40语音模式的出现也可能带来新的攻击面。在本文中，我们首次提出了针对GPT-40语音模式的越狱攻击的系统测量。我们发现，GPT-4o在直接将禁止问题和文本越狱提示转换为语音模式时，表现出了良好的抵抗能力。这种阻力主要是由于GPT-40的内部保护措施，以及将文本越狱提示改编为语音模式的困难。受GPT-40类似人类行为的启发，我们提出了VoiceJailBreak，一种新颖的语音越狱攻击，将GPT-40人性化，并试图通过虚构的故事讲述(背景、人物和情节)来说服它。语音越狱能够生成简单、可听但有效的越狱提示，在六种禁止场景下显著提高平均攻击成功率(ASR)，从0.033提高到0.778。我们还进行了大量的实验，探索互动步骤、小说写作的关键要素以及不同的语言对语音越狱效果的影响，并利用先进的小说写作技巧进一步提高攻击性能。我们希望我们的研究能够帮助研究界建立更安全、更规范的MLLMS。



## **19. Efficient Black-box Adversarial Attacks via Bayesian Optimization Guided by a Function Prior**

通过函数先验引导的Bayesian优化进行高效的黑匣子对抗攻击 cs.LG

ICML 2024

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19098v1) [paper-pdf](http://arxiv.org/pdf/2405.19098v1)

**Authors**: Shuyu Cheng, Yibo Miao, Yinpeng Dong, Xiao Yang, Xiao-Shan Gao, Jun Zhu

**Abstract**: This paper studies the challenging black-box adversarial attack that aims to generate adversarial examples against a black-box model by only using output feedback of the model to input queries. Some previous methods improve the query efficiency by incorporating the gradient of a surrogate white-box model into query-based attacks due to the adversarial transferability. However, the localized gradient is not informative enough, making these methods still query-intensive. In this paper, we propose a Prior-guided Bayesian Optimization (P-BO) algorithm that leverages the surrogate model as a global function prior in black-box adversarial attacks. As the surrogate model contains rich prior information of the black-box one, P-BO models the attack objective with a Gaussian process whose mean function is initialized as the surrogate model's loss. Our theoretical analysis on the regret bound indicates that the performance of P-BO may be affected by a bad prior. Therefore, we further propose an adaptive integration strategy to automatically adjust a coefficient on the function prior by minimizing the regret bound. Extensive experiments on image classifiers and large vision-language models demonstrate the superiority of the proposed algorithm in reducing queries and improving attack success rates compared with the state-of-the-art black-box attacks. Code is available at https://github.com/yibo-miao/PBO-Attack.

摘要: 研究了一种具有挑战性的黑盒对抗性攻击，其目的是通过只使用模型的输出反馈来输入查询来生成针对黑盒模型的对抗性实例。以前的一些方法通过将代理白盒模型的梯度融入到基于查询的攻击中来提高查询效率，这是因为攻击具有对抗性。然而，局部化的梯度信息不足，使得这些方法仍然是查询密集型的。本文提出了一种先验引导的贝叶斯优化算法(P-BO)，该算法利用代理模型作为黑盒对抗攻击的全局先验函数。由于代理模型包含了丰富的黑盒模型的先验信息，P-BO用一个高斯过程对攻击目标进行建模，其均值函数被初始化为代理模型的损失。我们对遗憾界的理论分析表明，坏的先验可能会影响P-BO的性能。因此，我们进一步提出了一种自适应积分策略，通过最小化遗憾界来自动调整函数先验上的系数。在图像分类器和大型视觉语言模型上的大量实验表明，与最先进的黑盒攻击相比，该算法在减少查询和提高攻击成功率方面具有优势。代码可在https://github.com/yibo-miao/PBO-Attack.上找到



## **20. DiveR-CT: Diversity-enhanced Red Teaming with Relaxing Constraints**

DiveR-CT：多元化增强的红色团队，具有轻松的约束 cs.LG

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19026v1) [paper-pdf](http://arxiv.org/pdf/2405.19026v1)

**Authors**: Andrew Zhao, Quentin Xu, Matthieu Lin, Shenzhi Wang, Yong-jin Liu, Zilong Zheng, Gao Huang

**Abstract**: Recent advances in large language models (LLMs) have made them indispensable, raising significant concerns over managing their safety. Automated red teaming offers a promising alternative to the labor-intensive and error-prone manual probing for vulnerabilities, providing more consistent and scalable safety evaluations. However, existing approaches often compromise diversity by focusing on maximizing attack success rate. Additionally, methods that decrease the cosine similarity from historical embeddings with semantic diversity rewards lead to novelty stagnation as history grows. To address these issues, we introduce DiveR-CT, which relaxes conventional constraints on the objective and semantic reward, granting greater freedom for the policy to enhance diversity. Our experiments demonstrate DiveR-CT's marked superiority over baselines by 1) generating data that perform better in various diversity metrics across different attack success rate levels, 2) better-enhancing resiliency in blue team models through safety tuning based on collected data, 3) allowing dynamic control of objective weights for reliable and controllable attack success rates, and 4) reducing susceptibility to reward overoptimization. Project details and code can be found at https://andrewzh112.github.io/#diverct.

摘要: 大型语言模型(LLM)的最新进展使它们变得不可或缺，这引发了人们对它们安全管理的重大担忧。自动红色团队提供了一种很有前途的替代方案，可以替代劳动密集型和容易出错的手动漏洞探测，提供更一致和可扩展的安全评估。然而，现有的方法往往通过关注最大化攻击成功率来损害多样性。此外，通过语义多样性奖励降低历史嵌入的余弦相似性的方法会随着历史的发展而导致新颖性停滞不前。为了解决这些问题，我们引入了Diver-CT，它放松了对客观和语义奖励的传统限制，赋予了政策更大的自由度来增强多样性。我们的实验展示了Diver-CT在以下方面的显著优势：1)生成在不同攻击成功率级别上在各种多样性度量中表现更好的数据；2)通过基于收集的数据进行安全调整，更好地增强蓝色团队模型的弹性；3)允许动态控制目标权重，以获得可靠和可控的攻击成功率；以及4)降低奖励过度优化的易感性。项目详情和代码可在https://andrewzh112.github.io/#diverct.上找到



## **21. Single Image Unlearning: Efficient Machine Unlearning in Multimodal Large Language Models**

单图像去学习：多模式大型语言模型中的高效机器去学习 cs.CV

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.12523v2) [paper-pdf](http://arxiv.org/pdf/2405.12523v2)

**Authors**: Jiaqi Li, Qianshan Wei, Chuanyi Zhang, Guilin Qi, Miaozeng Du, Yongrui Chen, Sheng Bi

**Abstract**: Machine unlearning empowers individuals with the `right to be forgotten' by removing their private or sensitive information encoded in machine learning models. However, it remains uncertain whether MU can be effectively applied to Multimodal Large Language Models (MLLMs), particularly in scenarios of forgetting the leaked visual data of concepts. To overcome the challenge, we propose an efficient method, Single Image Unlearning (SIU), to unlearn the visual recognition of a concept by fine-tuning a single associated image for few steps. SIU consists of two key aspects: (i) Constructing Multifaceted fine-tuning data. We introduce four targets, based on which we construct fine-tuning data for the concepts to be forgotten; (ii) Jointly training loss. To synchronously forget the visual recognition of concepts and preserve the utility of MLLMs, we fine-tune MLLMs through a novel Dual Masked KL-divergence Loss combined with Cross Entropy loss. Alongside our method, we establish MMUBench, a new benchmark for MU in MLLMs and introduce a collection of metrics for its evaluation. Experimental results on MMUBench show that SIU completely surpasses the performance of existing methods. Furthermore, we surprisingly find that SIU can avoid invasive membership inference attacks and jailbreak attacks. To the best of our knowledge, we are the first to explore MU in MLLMs. We will release the code and benchmark in the near future.

摘要: 机器遗忘通过删除编码在机器学习模型中的私人或敏感信息，使个人有被遗忘的权利。然而，MU是否能有效地应用于多通道大语言模型，特别是在忘记概念的泄露的视觉数据的情况下，仍然是不确定的。为了克服这一挑战，我们提出了一种有效的方法，单图像忘却学习(SIU)，通过对单个关联图像进行几个步骤的微调来消除对概念的视觉识别。SIU包括两个关键方面：(I)构建多方面的微调数据。我们引入了四个目标，在此基础上构建了精调的数据，以使概念被遗忘；(Ii)联合训练损失。为了同步忘记对概念的视觉识别，同时保持MLLS的实用性，我们通过一种新颖的双屏蔽KL-发散损失和交叉熵损失来微调MLLMS。除了我们的方法之外，我们还建立了MLLMS中MU的一个新的基准MMUBENCH，并引入了一组用于评估它的度量。在MMUBENCH上的实验结果表明，SIU的性能完全优于现有方法。此外，我们惊讶地发现，SIU可以避免入侵性的成员推理攻击和越狱攻击。据我们所知，我们是第一个探索MLLMS中的MU的人。我们将在不久的将来发布代码和基准测试。



## **22. Genshin: General Shield for Natural Language Processing with Large Language Models**

Genshin：具有大型语言模型的自然语言处理的通用盾牌 cs.CL

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.18741v1) [paper-pdf](http://arxiv.org/pdf/2405.18741v1)

**Authors**: Xiao Peng, Tao Liu, Ying Wang

**Abstract**: Large language models (LLMs) like ChatGPT, Gemini, or LLaMA have been trending recently, demonstrating considerable advancement and generalizability power in countless domains. However, LLMs create an even bigger black box exacerbating opacity, with interpretability limited to few approaches. The uncertainty and opacity embedded in LLMs' nature restrict their application in high-stakes domains like financial fraud, phishing, etc. Current approaches mainly rely on traditional textual classification with posterior interpretable algorithms, suffering from attackers who may create versatile adversarial samples to break the system's defense, forcing users to make trade-offs between efficiency and robustness. To address this issue, we propose a novel cascading framework called Genshin (General Shield for Natural Language Processing with Large Language Models), utilizing LLMs as defensive one-time plug-ins. Unlike most applications of LLMs that try to transform text into something new or structural, Genshin uses LLMs to recover text to its original state. Genshin aims to combine the generalizability of the LLM, the discrimination of the median model, and the interpretability of the simple model. Our experiments on the task of sentimental analysis and spam detection have shown fatal flaws of the current median models and exhilarating results on LLMs' recovery ability, demonstrating that Genshin is both effective and efficient. In our ablation study, we unearth several intriguing observations. Utilizing the LLM defender, a tool derived from the 4th paradigm, we have reproduced BERT's 15% optimal mask rate results in the 3rd paradigm of NLP. Additionally, when employing the LLM as a potential adversarial tool, attackers are capable of executing effective attacks that are nearly semantically lossless.

摘要: 像ChatGPT、Gemini或Llama这样的大型语言模型(LLM)最近已经成为趋势，在无数领域展示了相当大的先进性和泛化能力。然而，LLM创建了一个更大的黑匣子，加剧了不透明度，可解释性仅限于几种方法。LLMS本质上的不确定性和不透明性限制了它们在高风险领域的应用，如金融欺诈、网络钓鱼等。目前的方法主要依赖于传统的文本分类和后验可解释算法，攻击者可能会创建通用的对抗性样本来破坏系统的防御，迫使用户在效率和健壮性之间做出权衡。为了解决这个问题，我们提出了一种新颖的级联框架Genshin(General Shield For Natural Language Processing With Large Language Models)，利用LLMS作为防御性的一次性插件。与大多数试图将文本转换为新的或结构化的文本的LLMS应用程序不同，Genshin使用LLMS将文本恢复到其原始状态。Genshin的目标是将LLM的泛化能力、中值模型的区分性和简单模型的可解释性结合起来。我们在情感分析和垃圾邮件检测任务上的实验表明，现有的中值模型存在致命缺陷，并且在LLMS的恢复能力上取得了令人振奋的结果，证明了Genshin是有效的和高效的。在我们的消融研究中，我们发现了几个有趣的观察结果。利用LLM Defender，一个源自第四范式的工具，我们在NLP的第三范式中复制了Bert的15%最优掩蔽率结果。此外，当使用LLM作为潜在的敌意工具时，攻击者能够执行几乎在语义上无损的有效攻击。



## **23. Pandora's White-Box: Precise Training Data Detection and Extraction in Large Language Models**

潘多拉的白盒：大型语言模型中的精确训练数据检测和提取 cs.CR

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2402.17012v2) [paper-pdf](http://arxiv.org/pdf/2402.17012v2)

**Authors**: Jeffrey G. Wang, Jason Wang, Marvin Li, Seth Neel

**Abstract**: In this paper we develop state-of-the-art privacy attacks against Large Language Models (LLMs), where an adversary with some access to the model tries to learn something about the underlying training data. Our headline results are new membership inference attacks (MIAs) against pretrained LLMs that perform hundreds of times better than baseline attacks, and a pipeline showing that over 50% (!) of the fine-tuning dataset can be extracted from a fine-tuned LLM in natural settings. We consider varying degrees of access to the underlying model, pretraining and fine-tuning data, and both MIAs and training data extraction. For pretraining data, we propose two new MIAs: a supervised neural network classifier that predicts training data membership on the basis of (dimensionality-reduced) model gradients, as well as a variant of this attack that only requires logit access to the model which leverages recent model-stealing work on LLMs. To our knowledge this is the first MIA that explicitly incorporates model-stealing information. Both attacks outperform existing black-box baselines, and our supervised attack closes the gap between MIA attack success against LLMs and the strongest known attacks for other machine learning models. In fine-tuning, we find that a simple attack based on the ratio of the loss between the base and fine-tuned models is able to achieve near-perfect MIA performance; we then leverage our MIA to extract a large fraction of the fine-tuning dataset from fine-tuned Pythia and Llama models. Taken together, these results represent the strongest existing privacy attacks against both pretrained and fine-tuned LLMs for MIAs and training data extraction, which are of independent scientific interest and have important practical implications for LLM security, privacy, and copyright issues.

摘要: 在本文中，我们开发了针对大型语言模型(LLM)的最先进的隐私攻击，其中对该模型具有一定访问权限的对手试图了解一些关于潜在训练数据的信息。我们的主要结果是针对预先训练的LLM的新成员推理攻击(MIA)，其性能比基线攻击高数百倍，并且管道显示超过50%(！)可以在自然设置下从微调的LLM中提取精调数据集的。我们考虑了不同程度的访问底层模型、预训练和微调数据，以及MIA和训练数据提取。对于预训练数据，我们提出了两个新的MIA：一个有监督的神经网络分类器，它基于(降维)模型梯度来预测训练数据的成员资格，以及这种攻击的一个变体，它只需要Logit访问模型，利用了最近在LLMS上的模型窃取工作。据我们所知，这是第一个明确纳入模型窃取信息的MIA。这两种攻击都超过了现有的黑盒基线，我们的监督攻击缩小了针对LLMS的MIA攻击成功与针对其他机器学习模型的已知最强攻击之间的差距。在微调中，我们发现基于基本模型和微调模型之间的损失比率的简单攻击能够获得近乎完美的MIA性能；然后，我们利用我们的MIA从微调的Pythia和Llama模型中提取很大一部分微调数据集。综上所述，这些结果代表了针对用于MIA和训练数据提取的预先训练和微调的LLM的现有最强隐私攻击，这些攻击具有独立的科学意义，并对LLM的安全、隐私和版权问题具有重要的实践意义。



## **24. Learning diverse attacks on large language models for robust red-teaming and safety tuning**

学习对大型语言模型的多样化攻击，以实现强大的红色团队化和安全调整 cs.CL

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.18540v1) [paper-pdf](http://arxiv.org/pdf/2405.18540v1)

**Authors**: Seanie Lee, Minsu Kim, Lynn Cherif, David Dobre, Juho Lee, Sung Ju Hwang, Kenji Kawaguchi, Gauthier Gidel, Yoshua Bengio, Nikolay Malkin, Moksh Jain

**Abstract**: Red-teaming, or identifying prompts that elicit harmful responses, is a critical step in ensuring the safe and responsible deployment of large language models (LLMs). Developing effective protection against many modes of attack prompts requires discovering diverse attacks. Automated red-teaming typically uses reinforcement learning to fine-tune an attacker language model to generate prompts that elicit undesirable responses from a target LLM, as measured, for example, by an auxiliary toxicity classifier. We show that even with explicit regularization to favor novelty and diversity, existing approaches suffer from mode collapse or fail to generate effective attacks. As a flexible and probabilistically principled alternative, we propose to use GFlowNet fine-tuning, followed by a secondary smoothing phase, to train the attacker model to generate diverse and effective attack prompts. We find that the attacks generated by our method are effective against a wide range of target LLMs, both with and without safety tuning, and transfer well between target LLMs. Finally, we demonstrate that models safety-tuned using a dataset of red-teaming prompts generated by our method are robust to attacks from other RL-based red-teaming approaches.

摘要: 红色团队，或识别引发有害响应的提示，是确保安全和负责任地部署大型语言模型(LLM)的关键步骤。开发针对多种攻击提示的有效防护需要发现不同的攻击。自动红色团队通常使用强化学习来微调攻击者语言模型，以生成引发来自目标LLM的不良响应的提示，例如通过辅助毒性分类器来测量。我们表明，即使使用显式正则化来支持新颖性和多样性，现有的方法也会遭受模式崩溃或无法产生有效的攻击。作为一种灵活的、符合概率原则的替代方案，我们建议使用GFlowNet微调，然后进行二次平滑阶段，来训练攻击者模型以生成多样化和有效的攻击提示。我们发现，我们的方法产生的攻击对大范围的目标LLM有效，无论是否进行安全调整，并在目标LLM之间很好地转移。最后，我们证明了使用我们的方法生成的红队提示的数据集进行安全调整的模型对于来自其他基于RL的红队方法的攻击是健壮的。



## **25. Unleashing the potential of prompt engineering: a comprehensive review**

释放即时工程的潜力：全面回顾 cs.CL

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2310.14735v3) [paper-pdf](http://arxiv.org/pdf/2310.14735v3)

**Authors**: Banghao Chen, Zhaofeng Zhang, Nicolas Langrené, Shengxin Zhu

**Abstract**: This comprehensive review explores the transformative potential of prompt engineering within the realm of large language models (LLMs) and multimodal language models (MMLMs). The development of AI, from its inception in the 1950s to the emergence of neural networks and deep learning architectures, has culminated in sophisticated LLMs like GPT-4 and BERT, as well as MMLMs like DALL-E and CLIP. These models have revolutionized tasks in diverse fields such as workplace automation, healthcare, and education. Prompt engineering emerges as a crucial technique to maximize the utility and accuracy of these models. This paper delves into both foundational and advanced methodologies of prompt engineering, including techniques like Chain of Thought, Self-consistency, and Generated Knowledge, which significantly enhance model performance. Additionally, it examines the integration of multimodal data through innovative approaches such as Multi-modal Prompt Learning (MaPLe), Conditional Prompt Learning, and Context Optimization. Critical to this discussion is the aspect of AI security, particularly adversarial attacks that exploit vulnerabilities in prompt engineering. Strategies to mitigate these risks and enhance model robustness are thoroughly reviewed. The evaluation of prompt methods is addressed through both subjective and objective metrics, ensuring a robust analysis of their efficacy. This review underscores the pivotal role of prompt engineering in advancing AI capabilities, providing a structured framework for future research and application.

摘要: 这篇全面的综述探索了快速工程在大型语言模型(LLM)和多模式语言模型(MMLM)领域中的变革潜力。人工智能从20世纪50年代开始发展到神经网络和深度学习体系结构的出现，最终出现了GPT-4和BERT等复杂的LLM，以及Dall-E和CLIP等MMLM。这些模式使工作场所自动化、医疗保健和教育等不同领域的任务发生了革命性变化。为了最大限度地提高这些模型的实用性和准确性，快速工程技术应运而生。本文深入研究了即时工程的基础和高级方法，包括思想链、自我一致性和生成知识等技术，这些技术显著提高了模型的性能。此外，它还通过多模式快速学习(Maple)、条件性快速学习和上下文优化等创新方法研究了多模式数据的集成。对这一讨论至关重要的是人工智能安全方面，特别是利用即时工程中的漏洞进行的对抗性攻击。对缓解这些风险和增强模型稳健性的策略进行了彻底的回顾。对快速方法的评估通过主观和客观两个指标进行，确保对其有效性进行稳健的分析。这篇综述强调了快速工程在推进人工智能能力方面的关键作用，为未来的研究和应用提供了一个结构化的框架。



## **26. Defending Large Language Models Against Jailbreak Attacks via Layer-specific Editing**

通过特定层的编辑保护大型语言模型免受越狱攻击 cs.AI

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.18166v1) [paper-pdf](http://arxiv.org/pdf/2405.18166v1)

**Authors**: Wei Zhao, Zhe Li, Yige Li, Ye Zhang, Jun Sun

**Abstract**: Large language models (LLMs) are increasingly being adopted in a wide range of real-world applications. Despite their impressive performance, recent studies have shown that LLMs are vulnerable to deliberately crafted adversarial prompts even when aligned via Reinforcement Learning from Human Feedback or supervised fine-tuning. While existing defense methods focus on either detecting harmful prompts or reducing the likelihood of harmful responses through various means, defending LLMs against jailbreak attacks based on the inner mechanisms of LLMs remains largely unexplored. In this work, we investigate how LLMs response to harmful prompts and propose a novel defense method termed \textbf{L}ayer-specific \textbf{Ed}iting (LED) to enhance the resilience of LLMs against jailbreak attacks. Through LED, we reveal that several critical \textit{safety layers} exist among the early layers of LLMs. We then show that realigning these safety layers (and some selected additional layers) with the decoded safe response from selected target layers can significantly improve the alignment of LLMs against jailbreak attacks. Extensive experiments across various LLMs (e.g., Llama2, Mistral) show the effectiveness of LED, which effectively defends against jailbreak attacks while maintaining performance on benign prompts. Our code is available at \url{https://github.com/ledllm/ledllm}.

摘要: 大型语言模型(LLM)正越来越多地被广泛地应用于现实世界中。尽管它们的表现令人印象深刻，但最近的研究表明，即使在通过从人类反馈的强化学习或监督微调进行调整时，LLM仍容易受到故意设计的敌意提示的攻击。虽然现有的防御方法侧重于检测有害提示或通过各种手段减少有害响应的可能性，但基于LLMS的内部机制来防御LLMS的越狱攻击在很大程度上仍未被探索。在这项工作中，我们研究了LLMS对有害提示的响应，并提出了一种新的防御方法-.通过LED，我们揭示了LLMS的早期层之间存在着几个关键的安全层。然后，我们展示了将这些安全层(以及一些选定的附加层)与选定目标层的解码安全响应重新对准可以显著提高LLM对抗越狱攻击的对准。在各种LLM(如Llama2、Mistral)上的广泛实验表明，LED是有效的，它可以有效防御越狱攻击，同时保持对良性提示的性能。我们的代码可在\url{https://github.com/ledllm/ledllm}.



## **27. Exploiting LLM Quantization**

利用LLM量化 cs.LG

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.18137v1) [paper-pdf](http://arxiv.org/pdf/2405.18137v1)

**Authors**: Kazuki Egashira, Mark Vero, Robin Staab, Jingxuan He, Martin Vechev

**Abstract**: Quantization leverages lower-precision weights to reduce the memory usage of large language models (LLMs) and is a key technique for enabling their deployment on commodity hardware. While LLM quantization's impact on utility has been extensively explored, this work for the first time studies its adverse effects from a security perspective. We reveal that widely used quantization methods can be exploited to produce a harmful quantized LLM, even though the full-precision counterpart appears benign, potentially tricking users into deploying the malicious quantized model. We demonstrate this threat using a three-staged attack framework: (i) first, we obtain a malicious LLM through fine-tuning on an adversarial task; (ii) next, we quantize the malicious model and calculate constraints that characterize all full-precision models that map to the same quantized model; (iii) finally, using projected gradient descent, we tune out the poisoned behavior from the full-precision model while ensuring that its weights satisfy the constraints computed in step (ii). This procedure results in an LLM that exhibits benign behavior in full precision but when quantized, it follows the adversarial behavior injected in step (i). We experimentally demonstrate the feasibility and severity of such an attack across three diverse scenarios: vulnerable code generation, content injection, and over-refusal attack. In practice, the adversary could host the resulting full-precision model on an LLM community hub such as Hugging Face, exposing millions of users to the threat of deploying its malicious quantized version on their devices.

摘要: 量化利用较低精度的权重来减少大型语言模型(LLM)的内存使用，这是在商用硬件上部署LLM的关键技术。虽然LLM量化对效用的影响已经被广泛研究，但这项工作首次从安全的角度研究了它的不利影响。我们发现，广泛使用的量化方法可以被利用来产生有害的量化LLM，即使全精度对应的看起来是良性的，潜在地诱骗用户部署恶意量化模型。我们使用一个三阶段攻击框架演示了这一威胁：(I)首先，我们通过对敌方任务的微调来获得恶意LLM；(Ii)接下来，我们量化恶意模型，并计算映射到相同量化模型的所有全精度模型的约束；(Iii)最后，使用投影梯度下降，我们在确保其权重满足步骤(Ii)中计算的约束的同时，从全精度模型中排除有毒行为。这一过程导致LLM完全精确地表现出良性行为，但当量化时，它遵循在步骤(I)中注入的对抗性行为。我们通过实验演示了这种攻击在三种不同场景中的可行性和严重性：易受攻击的代码生成、内容注入和过度拒绝攻击。在实践中，对手可能会在LLM社区中心(如拥抱脸)上托管产生的全精度模型，使数百万用户面临在他们的设备上部署其恶意量化版本的威胁。



## **28. Instruction Backdoor Attacks Against Customized LLMs**

针对定制LLM的指令后门攻击 cs.CR

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2402.09179v3) [paper-pdf](http://arxiv.org/pdf/2402.09179v3)

**Authors**: Rui Zhang, Hongwei Li, Rui Wen, Wenbo Jiang, Yuan Zhang, Michael Backes, Yun Shen, Yang Zhang

**Abstract**: The increasing demand for customized Large Language Models (LLMs) has led to the development of solutions like GPTs. These solutions facilitate tailored LLM creation via natural language prompts without coding. However, the trustworthiness of third-party custom versions of LLMs remains an essential concern. In this paper, we propose the first instruction backdoor attacks against applications integrated with untrusted customized LLMs (e.g., GPTs). Specifically, these attacks embed the backdoor into the custom version of LLMs by designing prompts with backdoor instructions, outputting the attacker's desired result when inputs contain the pre-defined triggers. Our attack includes 3 levels of attacks: word-level, syntax-level, and semantic-level, which adopt different types of triggers with progressive stealthiness. We stress that our attacks do not require fine-tuning or any modification to the backend LLMs, adhering strictly to GPTs development guidelines. We conduct extensive experiments on 6 prominent LLMs and 5 benchmark text classification datasets. The results show that our instruction backdoor attacks achieve the desired attack performance without compromising utility. Additionally, we propose two defense strategies and demonstrate their effectiveness in reducing such attacks. Our findings highlight the vulnerability and the potential risks of LLM customization such as GPTs.

摘要: 对定制的大型语言模型(LLM)的需求日益增长，导致了GPTS等解决方案的开发。这些解决方案无需编码即可通过自然语言提示实现定制的LLM创建。然而，第三方定制版本的LLMS的可信性仍然是一个关键问题。在本文中，我们提出了针对集成了不可信任的定制LLM的应用程序(例如GPT)的第一指令后门攻击。具体地说，这些攻击通过设计带有后门指令的提示将后门嵌入到LLMS的自定义版本中，并在输入包含预定义触发器时输出攻击者所需的结果。我们的攻击包括词级、句法级和语义级三个级别的攻击，它们采用了不同类型的触发器，具有渐进的隐蔽性。我们强调，我们的攻击不需要对后端LLM进行微调或任何修改，严格遵守GPTS开发指南。我们在6个重要的LLMS和5个基准文本分类数据集上进行了大量的实验。结果表明，指令后门攻击在不影响效用的情况下达到了预期的攻击性能。此外，我们还提出了两种防御策略，并展示了它们在减少此类攻击方面的有效性。我们的发现突出了LLM定制(如GPTS)的脆弱性和潜在风险。



## **29. S-Eval: Automatic and Adaptive Test Generation for Benchmarking Safety Evaluation of Large Language Models**

S-Eval：用于大型语言模型基准安全评估的自动和自适应测试生成 cs.CR

18 pages, 11 figures

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.14191v3) [paper-pdf](http://arxiv.org/pdf/2405.14191v3)

**Authors**: Xiaohan Yuan, Jinfeng Li, Dongxia Wang, Yuefeng Chen, Xiaofeng Mao, Longtao Huang, Hui Xue, Wenhai Wang, Kui Ren, Jingyi Wang

**Abstract**: Large Language Models have gained considerable attention for their revolutionary capabilities. However, there is also growing concern on their safety implications, making a comprehensive safety evaluation for LLMs urgently needed before model deployment. In this work, we propose S-Eval, a new comprehensive, multi-dimensional and open-ended safety evaluation benchmark. At the core of S-Eval is a novel LLM-based automatic test prompt generation and selection framework, which trains an expert testing LLM Mt combined with a range of test selection strategies to automatically construct a high-quality test suite for the safety evaluation. The key to the automation of this process is a novel expert safety-critique LLM Mc able to quantify the riskiness score of an LLM's response, and additionally produce risk tags and explanations. Besides, the generation process is also guided by a carefully designed risk taxonomy with four different levels, covering comprehensive and multi-dimensional safety risks of concern. Based on these, we systematically construct a new and large-scale safety evaluation benchmark for LLMs consisting of 220,000 evaluation prompts, including 20,000 base risk prompts (10,000 in Chinese and 10,000 in English) and 200,000 corresponding attack prompts derived from 10 popular adversarial instruction attacks against LLMs. Moreover, considering the rapid evolution of LLMs and accompanied safety threats, S-Eval can be flexibly configured and adapted to include new risks, attacks and models. S-Eval is extensively evaluated on 20 popular and representative LLMs. The results confirm that S-Eval can better reflect and inform the safety risks of LLMs compared to existing benchmarks. We also explore the impacts of parameter scales, language environments, and decoding parameters on the evaluation, providing a systematic methodology for evaluating the safety of LLMs.

摘要: 大型语言模型因其革命性的能力而获得了相当大的关注。然而，人们也越来越担心它们的安全影响，这使得在模型部署之前迫切需要对LLMS进行全面的安全评估。在这项工作中，我们提出了一种新的全面、多维、开放式的安全评价基准S-EVAL。S-EVAL的核心是一种新颖的基于LLM的测试提示自动生成和选择框架，该框架训练一名测试专家，结合一系列测试选择策略，自动构建用于安全评估的高质量测试用例集。这一过程自动化的关键是一种新颖的专家安全评论LLm Mc，它能够量化LLm响应的风险分数，并另外产生风险标签和解释。此外，生成过程还遵循了精心设计的四个不同级别的风险分类，涵盖了令人关注的全面和多维度的安全风险。在此基础上，我们系统地构建了一个新的大规模的低层管理系统安全评估基准，该基准由22万条评估提示组成，其中包括2万条基本风险提示(中文10000条，英文10000条)和来自10种流行的对抗性指令攻击的20万条相应的攻击提示。此外，考虑到LLM的快速演化和伴随的安全威胁，S-EVAL可以灵活配置和调整，以包括新的风险、攻击和模型。S-EVAL在20个流行和有代表性的低成本模型上进行了广泛的评估。结果证实，与现有基准相比，S-EVAL能够更好地反映和告知低成本机械的安全风险。我们还探讨了参数尺度、语言环境和解码参数对评估的影响，为评估LLMS的安全性提供了一种系统的方法。



## **30. Detoxifying Large Language Models via Knowledge Editing**

通过知识编辑消除大型语言模型的神秘性 cs.CL

ACL 2024. Project website: https://zjunlp.github.io/project/SafeEdit  Benchmark: https://huggingface.co/datasets/zjunlp/SafeEdit

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2403.14472v5) [paper-pdf](http://arxiv.org/pdf/2403.14472v5)

**Authors**: Mengru Wang, Ningyu Zhang, Ziwen Xu, Zekun Xi, Shumin Deng, Yunzhi Yao, Qishen Zhang, Linyi Yang, Jindong Wang, Huajun Chen

**Abstract**: This paper investigates using knowledge editing techniques to detoxify Large Language Models (LLMs). We construct a benchmark, SafeEdit, which covers nine unsafe categories with various powerful attack prompts and equips comprehensive metrics for systematic evaluation. We conduct experiments with several knowledge editing approaches, indicating that knowledge editing has the potential to detoxify LLMs with a limited impact on general performance efficiently. Then, we propose a simple yet effective baseline, dubbed Detoxifying with Intraoperative Neural Monitoring (DINM), to diminish the toxicity of LLMs within a few tuning steps via only one instance. We further provide an in-depth analysis of the internal mechanism for various detoxifying approaches, demonstrating that previous methods like SFT and DPO may merely suppress the activations of toxic parameters, while DINM mitigates the toxicity of the toxic parameters to a certain extent, making permanent adjustments. We hope that these insights could shed light on future work of developing detoxifying approaches and the underlying knowledge mechanisms of LLMs. Code and benchmark are available at https://github.com/zjunlp/EasyEdit.

摘要: 本文研究了利用知识编辑技术对大型语言模型进行去毒处理。我们构建了一个涵盖9个不安全类别、具有各种强大的攻击提示的基准SafeEdit，并配备了全面的度量来进行系统评估。我们用几种知识编辑方法进行了实验，表明知识编辑有可能在对一般性能影响有限的情况下有效地解毒LLM。然后，我们提出了一个简单而有效的基线，称为术中神经监测解毒(DINM)，仅通过一个实例在几个调整步骤内降低LLMS的毒性。我们进一步深入分析了各种解毒方法的内在机制，证明了以前的方法如SFT和DPO可能只是抑制了毒性参数的激活，而DINM在一定程度上减轻了毒性参数的毒性，做出了永久性的调整。我们希望这些洞察力能够为未来开发戒毒方法的工作和LLMS的潜在知识机制提供帮助。代码和基准测试可在https://github.com/zjunlp/EasyEdit.上获得



## **31. White-box Multimodal Jailbreaks Against Large Vision-Language Models**

针对大型视觉语言模型的白盒多模式越狱 cs.CV

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.17894v1) [paper-pdf](http://arxiv.org/pdf/2405.17894v1)

**Authors**: Ruofan Wang, Xingjun Ma, Hanxu Zhou, Chuanjun Ji, Guangnan Ye, Yu-Gang Jiang

**Abstract**: Recent advancements in Large Vision-Language Models (VLMs) have underscored their superiority in various multimodal tasks. However, the adversarial robustness of VLMs has not been fully explored. Existing methods mainly assess robustness through unimodal adversarial attacks that perturb images, while assuming inherent resilience against text-based attacks. Different from existing attacks, in this work we propose a more comprehensive strategy that jointly attacks both text and image modalities to exploit a broader spectrum of vulnerability within VLMs. Specifically, we propose a dual optimization objective aimed at guiding the model to generate affirmative responses with high toxicity. Our attack method begins by optimizing an adversarial image prefix from random noise to generate diverse harmful responses in the absence of text input, thus imbuing the image with toxic semantics. Subsequently, an adversarial text suffix is integrated and co-optimized with the adversarial image prefix to maximize the probability of eliciting affirmative responses to various harmful instructions. The discovered adversarial image prefix and text suffix are collectively denoted as a Universal Master Key (UMK). When integrated into various malicious queries, UMK can circumvent the alignment defenses of VLMs and lead to the generation of objectionable content, known as jailbreaks. The experimental results demonstrate that our universal attack strategy can effectively jailbreak MiniGPT-4 with a 96% success rate, highlighting the vulnerability of VLMs and the urgent need for new alignment strategies.

摘要: 大型视觉语言模型(VLM)的最新进展凸显了它们在各种多通道任务中的优越性。然而，VLMS的对抗健壮性还没有得到充分的研究。现有的方法主要通过扰乱图像的单峰对抗性攻击来评估稳健性，同时假设对基于文本的攻击具有内在的弹性。与已有的攻击不同，我们提出了一种更全面的策略，联合攻击文本和图像模式，以利用VLM中更广泛的漏洞。具体地说，我们提出了一个双重优化目标，旨在引导模型产生高毒性的肯定反应。我们的攻击方法首先从随机噪声中优化一个敌意图像前缀，在没有文本输入的情况下产生不同的有害响应，从而使图像充满有毒语义。随后，对抗性文本后缀与对抗性图像前缀集成并共同优化，以最大限度地引起对各种有害指令的肯定响应的概率。所发现的敌意图像前缀和文本后缀统称为通用主密钥(UMK)。当集成到各种恶意查询中时，UMK可以绕过VLM的对齐防御，并导致生成令人反感的内容，即所谓的越狱。实验结果表明，我们的通用攻击策略能够有效地越狱MiniGPT-4，成功率为96%，凸显了VLMS的脆弱性和对新的对齐策略的迫切需求。



## **32. Automatic Jailbreaking of the Text-to-Image Generative AI Systems**

文本到图像生成人工智能系统的自动越狱 cs.AI

Under review

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.16567v2) [paper-pdf](http://arxiv.org/pdf/2405.16567v2)

**Authors**: Minseon Kim, Hyomin Lee, Boqing Gong, Huishuai Zhang, Sung Ju Hwang

**Abstract**: Recent AI systems have shown extremely powerful performance, even surpassing human performance, on various tasks such as information retrieval, language generation, and image generation based on large language models (LLMs). At the same time, there are diverse safety risks that can cause the generation of malicious contents by circumventing the alignment in LLMs, which are often referred to as jailbreaking. However, most of the previous works only focused on the text-based jailbreaking in LLMs, and the jailbreaking of the text-to-image (T2I) generation system has been relatively overlooked. In this paper, we first evaluate the safety of the commercial T2I generation systems, such as ChatGPT, Copilot, and Gemini, on copyright infringement with naive prompts. From this empirical study, we find that Copilot and Gemini block only 12% and 17% of the attacks with naive prompts, respectively, while ChatGPT blocks 84% of them. Then, we further propose a stronger automated jailbreaking pipeline for T2I generation systems, which produces prompts that bypass their safety guards. Our automated jailbreaking framework leverages an LLM optimizer to generate prompts to maximize degree of violation from the generated images without any weight updates or gradient computation. Surprisingly, our simple yet effective approach successfully jailbreaks the ChatGPT with 11.0% block rate, making it generate copyrighted contents in 76% of the time. Finally, we explore various defense strategies, such as post-generation filtering and machine unlearning techniques, but found that they were inadequate, which suggests the necessity of stronger defense mechanisms.

摘要: 最近的人工智能系统在信息检索、语言生成和基于大语言模型(LLMS)的图像生成等各种任务上表现出了极其强大的性能，甚至超过了人类的性能。同时，存在多种安全风险，可以通过绕过LLM中的对齐(通常称为越狱)来导致恶意内容的生成。然而，以前的工作大多只关注基于文本的LLMS越狱，而对文本到图像(T2I)生成系统的越狱则相对忽视。在本文中，我们首先评估了商业T2I生成系统，如ChatGPT，Copilot，和Gemini，在天真提示下的版权侵权安全性。从这项实证研究中，我们发现Copilot和Gemini分别只阻止了12%和17%的带有幼稚提示的攻击，而ChatGPT阻止了其中84%的攻击。然后，我们进一步提出了一种更强大的自动化越狱管道，用于T2I生成系统，它产生绕过安全警卫的提示。我们的自动化越狱框架利用LLM优化器来生成提示，以最大化生成的图像的违规程度，而无需任何权重更新或梯度计算。令人惊讶的是，我们简单而有效的方法成功地破解了ChatGPT，拦截率为11.0%，使其在76%的时间内生成受版权保护的内容。最后，我们探索了各种防御策略，如后代过滤和机器遗忘技术，但发现它们都不够充分，这表明有必要建立更强大的防御机制。



## **33. Improved Generation of Adversarial Examples Against Safety-aligned LLMs**

针对安全一致的LLM改进对抗示例的生成 cs.CR

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.20778v1) [paper-pdf](http://arxiv.org/pdf/2405.20778v1)

**Authors**: Qizhang Li, Yiwen Guo, Wangmeng Zuo, Hao Chen

**Abstract**: Despite numerous efforts to ensure large language models (LLMs) adhere to safety standards and produce harmless content, some successes have been achieved in bypassing these restrictions, known as jailbreak attacks against LLMs. Adversarial prompts generated using gradient-based methods exhibit outstanding performance in performing jailbreak attacks automatically. Nevertheless, due to the discrete nature of texts, the input gradient of LLMs struggles to precisely reflect the magnitude of loss change that results from token replacements in the prompt, leading to limited attack success rates against safety-aligned LLMs, even in the white-box setting. In this paper, we explore a new perspective on this problem, suggesting that it can be alleviated by leveraging innovations inspired in transfer-based attacks that were originally proposed for attacking black-box image classification models. For the first time, we appropriate the ideologies of effective methods among these transfer-based attacks, i.e., Skip Gradient Method and Intermediate Level Attack, for improving the effectiveness of automatically generated adversarial examples against white-box LLMs. With appropriate adaptations, we inject these ideologies into gradient-based adversarial prompt generation processes and achieve significant performance gains without introducing obvious computational cost. Meanwhile, by discussing mechanisms behind the gains, new insights are drawn, and proper combinations of these methods are also developed. Our empirical results show that the developed combination achieves >30% absolute increase in attack success rates compared with GCG for attacking the Llama-2-7B-Chat model on AdvBench.

摘要: 尽管大量努力确保大型语言模型(LLM)遵守安全标准并生成无害的内容，但在绕过这些限制方面取得了一些成功，即针对LLM的越狱攻击。使用基于梯度的方法生成的对抗性提示在自动执行越狱攻击方面表现出出色的性能。然而，由于文本的离散性，LLMS的输入梯度难以准确反映提示中令牌替换导致的损失变化的大小，导致即使在白盒设置下，对安全对齐的LLM的攻击成功率也是有限的。在这篇文章中，我们探索了一个新的视角来解决这个问题，建议通过利用最初被提出用于攻击黑盒图像分类模型的基于传输的攻击的创新来缓解这个问题。为了提高自动生成的对抗白盒LLM攻击的有效性，我们首次借鉴了基于转移的攻击方法中有效方法的思想，即跳过梯度法和中级攻击。通过适当的调整，我们将这些思想注入到基于梯度的对抗性提示生成过程中，并在不引入明显计算代价的情况下获得显著的性能提升。同时，通过讨论收益背后的机制，得出了新的见解，并开发了这些方法的适当组合。我们的实验结果表明，与GCG相比，改进的组合在攻击AdvBtch上的Llama-2-7B-Chat模型时，攻击成功率绝对提高了30%以上。



## **34. Exploring Backdoor Attacks against Large Language Model-based Decision Making**

探索针对基于大型语言模型的决策的后门攻击 cs.CR

27 pages, including main paper, references, and appendix

**SubmitDate**: 2024-05-27    [abs](http://arxiv.org/abs/2405.20774v1) [paper-pdf](http://arxiv.org/pdf/2405.20774v1)

**Authors**: Ruochen Jiao, Shaoyuan Xie, Justin Yue, Takami Sato, Lixu Wang, Yixuan Wang, Qi Alfred Chen, Qi Zhu

**Abstract**: Large Language Models (LLMs) have shown significant promise in decision-making tasks when fine-tuned on specific applications, leveraging their inherent common sense and reasoning abilities learned from vast amounts of data. However, these systems are exposed to substantial safety and security risks during the fine-tuning phase. In this work, we propose the first comprehensive framework for Backdoor Attacks against LLM-enabled Decision-making systems (BALD), systematically exploring how such attacks can be introduced during the fine-tuning phase across various channels. Specifically, we propose three attack mechanisms and corresponding backdoor optimization methods to attack different components in the LLM-based decision-making pipeline: word injection, scenario manipulation, and knowledge injection. Word injection embeds trigger words directly into the query prompt. Scenario manipulation occurs in the physical environment, where a high-level backdoor semantic scenario triggers the attack. Knowledge injection conducts backdoor attacks on retrieval augmented generation (RAG)-based LLM systems, strategically injecting word triggers into poisoned knowledge while ensuring the information remains factually accurate for stealthiness. We conduct extensive experiments with three popular LLMs (GPT-3.5, LLaMA2, PaLM2), using two datasets (HighwayEnv, nuScenes), and demonstrate the effectiveness and stealthiness of our backdoor triggers and mechanisms. Finally, we critically assess the strengths and weaknesses of our proposed approaches, highlight the inherent vulnerabilities of LLMs in decision-making tasks, and evaluate potential defenses to safeguard LLM-based decision making systems.

摘要: 大型语言模型(LLM)在针对特定应用程序进行微调时，利用它们从大量数据中学到的固有常识和推理能力，在决策任务中显示出巨大的前景。然而，这些系统在微调阶段面临着相当大的安全和安保风险。在这项工作中，我们提出了第一个针对LLM启用的决策制定系统(BALD)的全面的后门攻击框架，系统地探索了如何在各种渠道的微调阶段引入此类攻击。具体地说，针对基于LLM的决策流水线中的不同组件，我们提出了三种攻击机制和相应的后门优化方法：单词注入、场景操纵和知识注入。单词注入将触发词直接嵌入到查询提示中。场景操纵发生在物理环境中，高级后门语义场景触发攻击。知识注入对基于检索增强生成(RAG)的LLM系统进行后门攻击，战略性地向有毒知识注入单词触发器，同时确保信息保持真实准确的隐蔽性。我们在三个流行的LLMS(GPT-3.5，LLaMA2，Palm2)上进行了广泛的实验，使用两个数据集(Highway Env，nuScenes)，并展示了我们的后门触发器和机制的有效性和隐蔽性。最后，我们批判性地评估了我们提出的方法的优点和缺点，强调了LLM在决策任务中的固有弱点，并评估了保护基于LLM的决策系统的潜在防御措施。



## **35. The Uncanny Valley: Exploring Adversarial Robustness from a Flatness Perspective**

恐怖谷：从扁平的角度探索对抗性的稳健性 cs.LG

**SubmitDate**: 2024-05-27    [abs](http://arxiv.org/abs/2405.16918v1) [paper-pdf](http://arxiv.org/pdf/2405.16918v1)

**Authors**: Nils Philipp Walter, Linara Adilova, Jilles Vreeken, Michael Kamp

**Abstract**: Flatness of the loss surface not only correlates positively with generalization but is also related to adversarial robustness, since perturbations of inputs relate non-linearly to perturbations of weights. In this paper, we empirically analyze the relation between adversarial examples and relative flatness with respect to the parameters of one layer. We observe a peculiar property of adversarial examples: during an iterative first-order white-box attack, the flatness of the loss surface measured around the adversarial example first becomes sharper until the label is flipped, but if we keep the attack running it runs into a flat uncanny valley where the label remains flipped. We find this phenomenon across various model architectures and datasets. Our results also extend to large language models (LLMs), but due to the discrete nature of the input space and comparatively weak attacks, the adversarial examples rarely reach a truly flat region. Most importantly, this phenomenon shows that flatness alone cannot explain adversarial robustness unless we can also guarantee the behavior of the function around the examples. We theoretically connect relative flatness to adversarial robustness by bounding the third derivative of the loss surface, underlining the need for flatness in combination with a low global Lipschitz constant for a robust model.

摘要: 损失曲面的平坦性不仅与泛化正相关，而且还与对抗性稳健性有关，因为输入的扰动与权重的扰动是非线性相关的。在这篇文章中，我们实证分析了对抗性例子与相对平坦度之间的关系。我们观察到对抗性例子的一个特殊性质：在迭代的一阶白盒攻击中，围绕对抗性例子测量的损失曲面的平坦度首先变得更尖锐，直到标签被翻转，但如果我们继续攻击，它会进入一个平坦的诡异山谷，在那里标签仍然被翻转。我们在各种模型体系结构和数据集中发现了这种现象。我们的结果也推广到大型语言模型(LLM)，但由于输入空间的离散性质和相对较弱的攻击，对抗性例子很少到达真正平坦的区域。最重要的是，这一现象表明，平坦性本身不能解释对抗健壮性，除非我们也能保证函数在示例周围的行为。理论上，我们通过限定损失曲面的三阶导数，将相对平坦性与对手的稳健性联系起来，强调了平坦性与稳健模型的低全局Lipschitz常数相结合的必要性。



## **36. Accelerating Greedy Coordinate Gradient via Probe Sampling**

通过探针采样加速贪婪坐标梯度 cs.CL

**SubmitDate**: 2024-05-27    [abs](http://arxiv.org/abs/2403.01251v2) [paper-pdf](http://arxiv.org/pdf/2403.01251v2)

**Authors**: Yiran Zhao, Wenyue Zheng, Tianle Cai, Xuan Long Do, Kenji Kawaguchi, Anirudh Goyal, Michael Shieh

**Abstract**: Safety of Large Language Models (LLMs) has become a critical issue given their rapid progresses. Greedy Coordinate Gradient (GCG) is shown to be effective in constructing adversarial prompts to break the aligned LLMs, but optimization of GCG is time-consuming. To reduce the time cost of GCG and enable more comprehensive studies of LLM safety, in this work, we study a new algorithm called $\texttt{Probe sampling}$. At the core of the algorithm is a mechanism that dynamically determines how similar a smaller draft model's predictions are to the target model's predictions for prompt candidates. When the target model is similar to the draft model, we rely heavily on the draft model to filter out a large number of potential prompt candidates. Probe sampling achieves up to $5.6$ times speedup using Llama2-7b-chat and leads to equal or improved attack success rate (ASR) on the AdvBench. Furthermore, probe sampling is also able to accelerate other prompt optimization techniques and adversarial methods, leading to acceleration of $1.8\times$ for AutoPrompt, $2.4\times$ for APE and $2.4\times$ for AutoDAN.

摘要: 随着大型语言模型的快速发展，其安全性已成为一个关键问题。贪婪坐标梯度(GCG)在构造敌意提示以打破排列的LLM方面是有效的，但GCG的优化是耗时的。为了减少GCG的时间开销，更全面地研究LLM的安全性，本文研究了一种新的算法--$\exttt{Probe Samples}$。该算法的核心是一种机制，该机制动态地确定较小的草稿模型的预测与目标模型对提示候选人的预测的相似性程度。当目标模型与选秀模型相似时，我们严重依赖选秀模型来筛选出大量潜在的提示候选者。使用Llama2-7b-Chat，探测采样可获得高达5.6美元的加速比，并可带来同等或更高的AdvBch攻击成功率(ASR)。此外，探针采样还能够加速其他即时优化技术和对抗方法，导致AutoPrompt、APE和AutoDAN的加速分别为1.8倍$、2.4倍$和2.4倍$。



## **37. Cross-Modality Jailbreak and Mismatched Attacks on Medical Multimodal Large Language Models**

跨模式越狱和对医学多模式大型语言模型的不匹配攻击 cs.CR

**SubmitDate**: 2024-05-26    [abs](http://arxiv.org/abs/2405.20775v1) [paper-pdf](http://arxiv.org/pdf/2405.20775v1)

**Authors**: Xijie Huang, Xinyuan Wang, Hantao Zhang, Jiawen Xi, Jingkun An, Hao Wang, Chengwei Pan

**Abstract**: Security concerns related to Large Language Models (LLMs) have been extensively explored, yet the safety implications for Multimodal Large Language Models (MLLMs), particularly in medical contexts (MedMLLMs), remain insufficiently studied. This paper delves into the underexplored security vulnerabilities of MedMLLMs, especially when deployed in clinical environments where the accuracy and relevance of question-and-answer interactions are critically tested against complex medical challenges. By combining existing clinical medical data with atypical natural phenomena, we redefine two types of attacks: mismatched malicious attack (2M-attack) and optimized mismatched malicious attack (O2M-attack). Using our own constructed voluminous 3MAD dataset, which covers a wide range of medical image modalities and harmful medical scenarios, we conduct a comprehensive analysis and propose the MCM optimization method, which significantly enhances the attack success rate on MedMLLMs. Evaluations with this dataset and novel attack methods, including white-box attacks on LLaVA-Med and transfer attacks on four other state-of-the-art models, indicate that even MedMLLMs designed with enhanced security features are vulnerable to security breaches. Our work underscores the urgent need for a concerted effort to implement robust security measures and enhance the safety and efficacy of open-source MedMLLMs, particularly given the potential severity of jailbreak attacks and other malicious or clinically significant exploits in medical settings. For further research and replication, anonymous access to our code is available at https://github.com/dirtycomputer/O2M_attack. Warning: Medical large model jailbreaking may generate content that includes unverified diagnoses and treatment recommendations. Always consult professional medical advice.

摘要: 与大语言模型(LLM)相关的安全问题已经得到了广泛的研究，但多模式大语言模型(MLLM)的安全影响，特别是在医学背景下(MedMLLMS)的安全影响，仍然没有得到充分的研究。本文深入研究了MedMLLMS未被开发的安全漏洞，特别是当部署在临床环境中时，其中问答交互的准确性和相关性针对复杂的医疗挑战进行了严格的测试。结合已有的临床医学数据和非典型自然现象，我们重新定义了两类攻击：失配恶意攻击(2M-攻击)和优化失配恶意攻击(O2M-攻击)。利用我们构建的涵盖多种医学图像模式和有害医疗场景的海量3MAD数据集，进行了综合分析，并提出了MCM优化方法，显著提高了对MedMLLms的攻击成功率。使用该数据集和新的攻击方法(包括对LLaVA-Med的白盒攻击和对其他四种最先进模型的传输攻击)的评估表明，即使是设计了增强安全功能的MedMLLM也容易受到安全漏洞的攻击。我们的工作强调了迫切需要共同努力，实施强有力的安全措施，提高开源MedMLLMS的安全性和有效性，特别是考虑到越狱攻击和医疗环境中其他恶意或具有临床意义的利用的潜在严重性。为了进行进一步的研究和复制，可以在https://github.com/dirtycomputer/O2M_attack.上匿名访问我们的代码警告：医用大型越狱可能会生成包括未经验证的诊断和治疗建议在内的内容。一定要咨询专业医生的意见。



## **38. Distributed Threat Intelligence at the Edge Devices: A Large Language Model-Driven Approach**

边缘设备上的分布式威胁情报：大型语言模型驱动的方法 cs.CR

**SubmitDate**: 2024-05-26    [abs](http://arxiv.org/abs/2405.08755v2) [paper-pdf](http://arxiv.org/pdf/2405.08755v2)

**Authors**: Syed Mhamudul Hasan, Alaa M. Alotaibi, Sajedul Talukder, Abdur R. Shahid

**Abstract**: With the proliferation of edge devices, there is a significant increase in attack surface on these devices. The decentralized deployment of threat intelligence on edge devices, coupled with adaptive machine learning techniques such as the in-context learning feature of Large Language Models (LLMs), represents a promising paradigm for enhancing cybersecurity on resource-constrained edge devices. This approach involves the deployment of lightweight machine learning models directly onto edge devices to analyze local data streams, such as network traffic and system logs, in real-time. Additionally, distributing computational tasks to an edge server reduces latency and improves responsiveness while also enhancing privacy by processing sensitive data locally. LLM servers can enable these edge servers to autonomously adapt to evolving threats and attack patterns, continuously updating their models to improve detection accuracy and reduce false positives. Furthermore, collaborative learning mechanisms facilitate peer-to-peer secure and trustworthy knowledge sharing among edge devices, enhancing the collective intelligence of the network and enabling dynamic threat mitigation measures such as device quarantine in response to detected anomalies. The scalability and flexibility of this approach make it well-suited for diverse and evolving network environments, as edge devices only send suspicious information such as network traffic and system log changes, offering a resilient and efficient solution to combat emerging cyber threats at the network edge. Thus, our proposed framework can improve edge computing security by providing better security in cyber threat detection and mitigation by isolating the edge devices from the network.

摘要: 随着边缘设备的扩散，这些设备上的攻击面显著增加。在边缘设备上分散部署威胁情报，再加上自适应机器学习技术，如大型语言模型(LLMS)的上下文中学习功能，代表了一种在资源受限的边缘设备上增强网络安全的有前途的范例。这种方法涉及将轻量级机器学习模型直接部署到边缘设备上，以实时分析本地数据流，如网络流量和系统日志。此外，将计算任务分发到边缘服务器可减少延迟并提高响应速度，同时还可通过在本地处理敏感数据来增强隐私。LLM服务器可以使这些边缘服务器自主适应不断变化的威胁和攻击模式，不断更新其模型以提高检测准确性并减少误报。此外，协作学习机制促进了边缘设备之间的对等安全和可信知识共享，增强了网络的集体智能，并支持动态威胁缓解措施，如响应检测到的异常情况的设备隔离。这种方法的可扩展性和灵活性使其非常适合于多样化和不断发展的网络环境，因为边缘设备只发送可疑信息，如网络流量和系统日志更改，为应对网络边缘出现的网络威胁提供了一种弹性和高效的解决方案。因此，我们提出的框架可以通过将边缘设备与网络隔离来提供更好的网络威胁检测和缓解的安全性，从而提高边缘计算的安全性。



## **39. FLTrojan: Privacy Leakage Attacks against Federated Language Models Through Selective Weight Tampering**

FLTrojan：通过选择性权重篡改对联邦语言模型进行隐私泄露攻击 cs.CR

20 pages (including bibliography and Appendix), Submitted to ACM CCS  '24

**SubmitDate**: 2024-05-26    [abs](http://arxiv.org/abs/2310.16152v2) [paper-pdf](http://arxiv.org/pdf/2310.16152v2)

**Authors**: Md Rafi Ur Rashid, Vishnu Asutosh Dasu, Kang Gu, Najrin Sultana, Shagufta Mehnaz

**Abstract**: Federated learning (FL) has become a key component in various language modeling applications such as machine translation, next-word prediction, and medical record analysis. These applications are trained on datasets from many FL participants that often include privacy-sensitive data, such as healthcare records, phone/credit card numbers, login credentials, etc. Although FL enables computation without necessitating clients to share their raw data, determining the extent of privacy leakage in federated language models is challenging and not straightforward. Moreover, existing attacks aim to extract data regardless of how sensitive or naive it is. To fill this research gap, we introduce two novel findings with regard to leaking privacy-sensitive user data from federated large language models. Firstly, we make a key observation that model snapshots from the intermediate rounds in FL can cause greater privacy leakage than the final trained model. Secondly, we identify that privacy leakage can be aggravated by tampering with a model's selective weights that are specifically responsible for memorizing the sensitive training data. We show how a malicious client can leak the privacy-sensitive data of some other users in FL even without any cooperation from the server. Our best-performing method improves the membership inference recall by 29% and achieves up to 71% private data reconstruction, evidently outperforming existing attacks with stronger assumptions of adversary capabilities.

摘要: 联合学习(FL)已经成为机器翻译、下一词预测和病历分析等各种语言建模应用中的关键组件。这些应用程序是在来自许多FL参与者的数据集上进行训练的，这些数据集通常包括隐私敏感数据，如医疗记录、电话/信用卡号码、登录凭据等。尽管FL可以在不需要客户共享其原始数据的情况下进行计算，但在联合语言模型中确定隐私泄漏的程度是具有挑战性的，而且不是直接的。此外，现有的攻击旨在提取数据，无论它是多么敏感或幼稚。为了填补这一研究空白，我们介绍了关于从联合大型语言模型泄露隐私敏感用户数据的两个新发现。首先，我们做了一个关键的观察，在FL的中间轮中的模型快照比最终训练的模型会导致更大的隐私泄露。其次，我们发现，通过篡改模型的选择性权重可能会加剧隐私泄露，这些选择性权重专门负责记忆敏感的训练数据。我们展示了恶意客户端如何在没有任何服务器合作的情况下泄露FL中其他用户的隐私敏感数据。该方法的成员关系推理召回率提高了29%，私有数据重构效率高达71%，明显优于对敌方能力假设更强的现有攻击。



## **40. Intruding with Words: Towards Understanding Graph Injection Attacks at the Text Level**

文字入侵：在文本层面了解图注入攻击 cs.LG

29 pages

**SubmitDate**: 2024-05-26    [abs](http://arxiv.org/abs/2405.16405v1) [paper-pdf](http://arxiv.org/pdf/2405.16405v1)

**Authors**: Runlin Lei, Yuwei Hu, Yuchen Ren, Zhewei Wei

**Abstract**: Graph Neural Networks (GNNs) excel across various applications but remain vulnerable to adversarial attacks, particularly Graph Injection Attacks (GIAs), which inject malicious nodes into the original graph and pose realistic threats. Text-attributed graphs (TAGs), where nodes are associated with textual features, are crucial due to their prevalence in real-world applications and are commonly used to evaluate these vulnerabilities. However, existing research only focuses on embedding-level GIAs, which inject node embeddings rather than actual textual content, limiting their applicability and simplifying detection. In this paper, we pioneer the exploration of GIAs at the text level, presenting three novel attack designs that inject textual content into the graph. Through theoretical and empirical analysis, we demonstrate that text interpretability, a factor previously overlooked at the embedding level, plays a crucial role in attack strength. Among the designs we investigate, the Word-frequency-based Text-level GIA (WTGIA) is particularly notable for its balance between performance and interpretability. Despite the success of WTGIA, we discover that defenders can easily enhance their defenses with customized text embedding methods or large language model (LLM)--based predictors. These insights underscore the necessity for further research into the potential and practical significance of text-level GIAs.

摘要: 图神经网络(GNN)在各种应用中表现出色，但仍然容易受到对手攻击，特别是图注入攻击(GIA)，图注入攻击将恶意节点注入到原始图中，并构成现实威胁。文本属性图(TAG)将节点与文本特征相关联，由于它们在现实应用程序中的普遍存在，因此至关重要，并且通常用于评估这些漏洞。然而，现有的研究只关注嵌入级GIA，这些GIA注入的是节点嵌入而不是实际的文本内容，限制了它们的适用性，简化了检测。在本文中，我们率先在文本层面上探索了GIA，提出了三种向图形中注入文本内容的新颖攻击设计。通过理论和实证分析，我们证明了文本可解释性对攻击强度起着至关重要的作用，而文本可解释性是此前在嵌入层面被忽视的一个因素。在我们研究的设计中，基于词频的文本级别GIA(WTGIA)特别值得注意的是它在性能和可解释性之间的平衡。尽管WTGIA取得了成功，但我们发现，防御者可以很容易地通过定制的文本嵌入方法或基于大型语言模型(LLM)的预测器来增强他们的防御。这些见解突显了进一步研究文本层面全球影响的潜力和现实意义的必要性。



## **41. Visual-RolePlay: Universal Jailbreak Attack on MultiModal Large Language Models via Role-playing Image Characte**

可视化角色扮演：通过角色扮演图像预设对多模式大型语言模型进行普遍越狱攻击 cs.CR

**SubmitDate**: 2024-05-25    [abs](http://arxiv.org/abs/2405.20773v1) [paper-pdf](http://arxiv.org/pdf/2405.20773v1)

**Authors**: Siyuan Ma, Weidi Luo, Yu Wang, Xiaogeng Liu, Muhao Chen, Bo Li, Chaowei Xiao

**Abstract**: With the advent and widespread deployment of Multimodal Large Language Models (MLLMs), ensuring their safety has become increasingly critical. To achieve this objective, it requires us to proactively discover the vulnerability of MLLMs by exploring the attack methods. Thus, structure-based jailbreak attacks, where harmful semantic content is embedded within images, have been proposed to mislead the models. However, previous structure-based jailbreak methods mainly focus on transforming the format of malicious queries, such as converting harmful content into images through typography, which lacks sufficient jailbreak effectiveness and generalizability. To address these limitations, we first introduce the concept of "Role-play" into MLLM jailbreak attacks and propose a novel and effective method called Visual Role-play (VRP). Specifically, VRP leverages Large Language Models to generate detailed descriptions of high-risk characters and create corresponding images based on the descriptions. When paired with benign role-play instruction texts, these high-risk character images effectively mislead MLLMs into generating malicious responses by enacting characters with negative attributes. We further extend our VRP method into a universal setup to demonstrate its generalizability. Extensive experiments on popular benchmarks show that VRP outperforms the strongest baseline, Query relevant and FigStep, by an average Attack Success Rate (ASR) margin of 14.3% across all models.

摘要: 随着多通道大语言模型的出现和广泛应用，确保其安全性变得越来越重要。为了实现这一目标，需要我们通过探索攻击方法来主动发现MLLMS的脆弱性。因此，已经提出了基于结构的越狱攻击，其中有害的语义内容嵌入到图像中，以误导模型。然而，以往的基于结构的越狱方法主要集中在对恶意查询的格式进行转换，如通过排版将有害内容转换为图像，缺乏足够的越狱有效性和通用性。针对这些局限性，我们首先将“角色扮演”的概念引入到MLLM越狱攻击中，提出了一种新颖而有效的方法--视觉角色扮演(VRP)。具体地说，VRP利用大型语言模型来生成高危角色的详细描述，并基于这些描述创建相应的图像。当与良性的角色扮演指示文本配对时，这些高风险角色图像有效地误导MLLM，通过设定具有负面属性的角色来生成恶意响应。我们进一步将VRP方法扩展到一个通用的设置，以证明它的普适性。在流行基准上的广泛实验表明，VRP在所有模型上的平均攻击成功率(ASR)边际都比最强的基线、查询相关和FigStep高14.3%。



## **42. No Two Devils Alike: Unveiling Distinct Mechanisms of Fine-tuning Attacks**

没有两个恶魔相似：揭示微调攻击的独特机制 cs.CL

work in progress

**SubmitDate**: 2024-05-25    [abs](http://arxiv.org/abs/2405.16229v1) [paper-pdf](http://arxiv.org/pdf/2405.16229v1)

**Authors**: Chak Tou Leong, Yi Cheng, Kaishuai Xu, Jian Wang, Hanlin Wang, Wenjie Li

**Abstract**: The existing safety alignment of Large Language Models (LLMs) is found fragile and could be easily attacked through different strategies, such as through fine-tuning on a few harmful examples or manipulating the prefix of the generation results. However, the attack mechanisms of these strategies are still underexplored. In this paper, we ask the following question: \textit{while these approaches can all significantly compromise safety, do their attack mechanisms exhibit strong similarities?} To answer this question, we break down the safeguarding process of an LLM when encountered with harmful instructions into three stages: (1) recognizing harmful instructions, (2) generating an initial refusing tone, and (3) completing the refusal response. Accordingly, we investigate whether and how different attack strategies could influence each stage of this safeguarding process. We utilize techniques such as logit lens and activation patching to identify model components that drive specific behavior, and we apply cross-model probing to examine representation shifts after an attack. In particular, we analyze the two most representative types of attack approaches: Explicit Harmful Attack (EHA) and Identity-Shifting Attack (ISA). Surprisingly, we find that their attack mechanisms diverge dramatically. Unlike ISA, EHA tends to aggressively target the harmful recognition stage. While both EHA and ISA disrupt the latter two stages, the extent and mechanisms of their attacks differ significantly. Our findings underscore the importance of understanding LLMs' internal safeguarding process and suggest that diverse defense mechanisms are required to effectively cope with various types of attacks.

摘要: 现有的大型语言模型(LLM)的安全对齐是脆弱的，很容易通过不同的策略受到攻击，例如通过微调几个有害的例子或操纵生成结果的前缀。然而，这些策略的攻击机制仍未得到充分的研究。在本文中，我们提出了以下问题：{虽然这些方法都可以显著地危害安全，但它们的攻击机制是否表现出很强的相似性？}为了回答这个问题，我们将遇到有害指令时LLM的保护过程分解为三个阶段：(1)识别有害指令，(2)生成初始拒绝音调，(3)完成拒绝响应。因此，我们调查了不同的攻击策略是否以及如何影响这一保护过程的每个阶段。我们利用Logit透镜和激活修补等技术来识别驱动特定行为的模型组件，并应用跨模型探测来检查攻击后的表示变化。重点分析了两种最具代表性的攻击方法：显性有害攻击(EHA)和身份转移攻击(ISA)。令人惊讶的是，我们发现它们的攻击机制截然不同。与ISA不同，EHA倾向于积极瞄准有害识别阶段。虽然EHA和ISA都破坏了后两个阶段，但它们的攻击程度和机制有很大不同。我们的发现强调了了解LLMS内部保护过程的重要性，并表明需要多样化的防御机制来有效应对各种类型的攻击。



## **43. Jailbreak and Guard Aligned Language Models with Only Few In-Context Demonstrations**

越狱和警卫一致的语言模型，只有很少的上下文演示 cs.LG

**SubmitDate**: 2024-05-25    [abs](http://arxiv.org/abs/2310.06387v3) [paper-pdf](http://arxiv.org/pdf/2310.06387v3)

**Authors**: Zeming Wei, Yifei Wang, Ang Li, Yichuan Mo, Yisen Wang

**Abstract**: Large Language Models (LLMs) have shown remarkable success in various tasks, yet their safety and the risk of generating harmful content remain pressing concerns. In this paper, we delve into the potential of In-Context Learning (ICL) to modulate the alignment of LLMs. Specifically, we propose the In-Context Attack (ICA) which employs harmful demonstrations to subvert LLMs, and the In-Context Defense (ICD) which bolsters model resilience through examples that demonstrate refusal to produce harmful responses. We offer theoretical insights to elucidate how a limited set of in-context demonstrations can pivotally influence the safety alignment of LLMs. Through extensive experiments, we demonstrate the efficacy of ICA and ICD in respectively elevating and mitigating the success rates of jailbreaking prompts. Our findings illuminate the profound influence of ICL on LLM behavior, opening new avenues for improving the safety of LLMs.

摘要: 大型语言模型（LLM）在各种任务中取得了显着的成功，但它们的安全性和生成有害内容的风险仍然是紧迫的问题。在本文中，我们探讨了上下文学习（ICL）调节LLM一致性的潜力。具体来说，我们提出了内上下文攻击（ICA）和内上下文防御（ICD），前者利用有害演示来颠覆LLM，后者通过展示拒绝产生有害响应的示例来增强模型的弹性。我们提供理论见解来阐明一组有限的背景演示如何能够对LLM的安全对齐产生重大影响。通过大量实验，我们证明了ICA和ICD分别提高和降低越狱提示成功率的功效。我们的研究结果阐明了ICL对LLM行为的深远影响，为提高LLM的安全性开辟了新的途径。



## **44. Mitigating Dialogue Hallucination for Large Vision Language Models via Adversarial Instruction Tuning**

通过对抗性指令调优缓解大视野语言模型的对话幻觉 cs.CV

**SubmitDate**: 2024-05-25    [abs](http://arxiv.org/abs/2403.10492v2) [paper-pdf](http://arxiv.org/pdf/2403.10492v2)

**Authors**: Dongmin Park, Zhaofang Qian, Guangxing Han, Ser-Nam Lim

**Abstract**: Mitigating hallucinations of Large Vision Language Models,(LVLMs) is crucial to enhance their reliability for general-purpose assistants. This paper shows that such hallucinations of LVLMs can be significantly exacerbated by preceding user-system dialogues. To precisely measure this, we first present an evaluation benchmark by extending popular multi-modal benchmark datasets with prepended hallucinatory dialogues powered by our novel Adversarial Question Generator (AQG), which can automatically generate image-related yet adversarial dialogues by adopting adversarial attacks on LVLMs. On our benchmark, the zero-shot performance of state-of-the-art LVLMs drops significantly for both the VQA and Captioning tasks. Next, we further reveal this hallucination is mainly due to the prediction bias toward preceding dialogues rather than visual content. To reduce this bias, we propose Adversarial Instruction Tuning (AIT) that robustly fine-tunes LVLMs against hallucinatory dialogues. Extensive experiments show our proposed approach successfully reduces dialogue hallucination while maintaining performance.

摘要: 减轻大型视觉语言模型(LVLMS)的幻觉对于提高其对通用助理的可靠性至关重要。这篇论文表明，之前的用户-系统对话可以显著加剧LVLMS的这种幻觉。为了准确地衡量这一点，我们首先提出了一个评估基准，通过扩展流行的多模式基准数据集，在我们的新型对抗性问题生成器(AQG)的支持下，使用预先设定的幻觉对话，该生成器可以通过对LVLM进行对抗性攻击来自动生成与图像相关的对抗性对话。在我们的基准测试中，最先进的LVLMS在VQA和字幕任务中的零镜头性能都显著下降。接下来，我们进一步揭示这种幻觉主要是由于预测偏向于之前的对话而不是视觉内容。为了减少这种偏差，我们提出了对抗性指令调整(AIT)，它针对幻觉对话对LVLM进行强有力的微调。大量的实验表明，我们提出的方法在保持性能的同时成功地减少了对话幻觉。



## **45. Revisit, Extend, and Enhance Hessian-Free Influence Functions**

重新审视、扩展和增强无黑森影响力功能 cs.LG

**SubmitDate**: 2024-05-25    [abs](http://arxiv.org/abs/2405.17490v1) [paper-pdf](http://arxiv.org/pdf/2405.17490v1)

**Authors**: Ziao Yang, Han Yue, Jian Chen, Hongfu Liu

**Abstract**: Influence functions serve as crucial tools for assessing sample influence in model interpretation, subset training set selection, noisy label detection, and more. By employing the first-order Taylor extension, influence functions can estimate sample influence without the need for expensive model retraining. However, applying influence functions directly to deep models presents challenges, primarily due to the non-convex nature of the loss function and the large size of model parameters. This difficulty not only makes computing the inverse of the Hessian matrix costly but also renders it non-existent in some cases. Various approaches, including matrix decomposition, have been explored to expedite and approximate the inversion of the Hessian matrix, with the aim of making influence functions applicable to deep models. In this paper, we revisit a specific, albeit naive, yet effective approximation method known as TracIn. This method substitutes the inverse of the Hessian matrix with an identity matrix. We provide deeper insights into why this simple approximation method performs well. Furthermore, we extend its applications beyond measuring model utility to include considerations of fairness and robustness. Finally, we enhance TracIn through an ensemble strategy. To validate its effectiveness, we conduct experiments on synthetic data and extensive evaluations on noisy label detection, sample selection for large language model fine-tuning, and defense against adversarial attacks.

摘要: 影响函数在模型解释、子集训练集选择、噪声标签检测等方面用作评估样本影响的重要工具。通过使用一阶泰勒扩展，影响函数可以估计样本影响，而不需要昂贵的模型重新训练。然而，直接将影响函数应用于深层模型会带来挑战，这主要是由于损失函数的非凸性和模型参数的大尺寸。这一困难不仅使计算海森矩阵的逆的成本高昂，而且在某些情况下使其不存在。已经探索了各种方法，包括矩阵分解，以加快和近似海森矩阵的求逆，目的是使影响函数适用于深层模式。在这篇文章中，我们回顾了一种特定的，尽管很幼稚，但有效的近似方法，称为TracIn。该方法用单位矩阵代替海森矩阵的逆。我们对为什么这种简单的近似方法表现良好提供了更深层次的见解。此外，我们将它的应用扩展到测量模型效用之外，包括对公平性和稳健性的考虑。最后，我们通过集成策略增强了TracIn。为了验证其有效性，我们在合成数据上进行了实验，并在噪声标签检测、大语言模型微调的样本选择以及对对手攻击的防御方面进行了广泛的评估。



## **46. Evaluating the Adversarial Robustness of Retrieval-Based In-Context Learning for Large Language Models**

评估大型语言模型基于检索的上下文学习的对抗鲁棒性 cs.CL

29 pages, 6 figures

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2405.15984v1) [paper-pdf](http://arxiv.org/pdf/2405.15984v1)

**Authors**: Simon Chi Lok Yu, Jie He, Pasquale Minervini, Jeff Z. Pan

**Abstract**: With the emergence of large language models, such as LLaMA and OpenAI GPT-3, In-Context Learning (ICL) gained significant attention due to its effectiveness and efficiency. However, ICL is very sensitive to the choice, order, and verbaliser used to encode the demonstrations in the prompt. Retrieval-Augmented ICL methods try to address this problem by leveraging retrievers to extract semantically related examples as demonstrations. While this approach yields more accurate results, its robustness against various types of adversarial attacks, including perturbations on test samples, demonstrations, and retrieved data, remains under-explored. Our study reveals that retrieval-augmented models can enhance robustness against test sample attacks, outperforming vanilla ICL with a 4.87% reduction in Attack Success Rate (ASR); however, they exhibit overconfidence in the demonstrations, leading to a 2% increase in ASR for demonstration attacks. Adversarial training can help improve the robustness of ICL methods to adversarial attacks; however, such a training scheme can be too costly in the context of LLMs. As an alternative, we introduce an effective training-free adversarial defence method, DARD, which enriches the example pool with those attacked samples. We show that DARD yields improvements in performance and robustness, achieving a 15% reduction in ASR over the baselines. Code and data are released to encourage further research: https://github.com/simonucl/adv-retreival-icl

摘要: 随着大型语言模型的出现，如Llama和OpenAI GPT-3，情景中学习(ICL)因其有效性和高效性而受到广泛关注。但是，ICL对用于对提示符中的演示进行编码的选择、顺序和形容词非常敏感。检索增强的ICL方法试图通过利用检索器来提取语义相关的示例作为演示来解决这个问题。虽然这种方法可以产生更准确的结果，但它对各种类型的对抗性攻击的稳健性，包括对测试样本、演示和检索数据的扰动，仍然没有得到充分的研究。我们的研究表明，检索增强模型可以增强对测试样本攻击的健壮性，性能优于普通ICL，攻击成功率(ASR)降低4.87%；然而，它们在演示中表现出过度自信，导致演示攻击的ASR提高了2%。对抗性训练可以帮助提高ICL方法对对抗性攻击的稳健性；然而，在LLMS的背景下，这样的训练方案可能代价太高。作为另一种选择，我们引入了一种有效的无需训练的对抗防御方法DARD，它用被攻击的样本丰富了样本库。我们表明，DARD在性能和健壮性方面都有改进，ASR比基准降低了15%。发布代码和数据是为了鼓励进一步的研究：https://github.com/simonucl/adv-retreival-icl



## **47. $$\mathbf{L^2\cdot M = C^2}$$ Large Language Models as Covert Channels... a Systematic Analysis**

$$\mathBF{L ' 2\csot M = C ' 2}$$大型语言模型作为隐蔽通道.了系统的分析 cs.CR

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2405.15652v1) [paper-pdf](http://arxiv.org/pdf/2405.15652v1)

**Authors**: Simen Gaure, Stefanos Koffas, Stjepan Picek, Sondre Rønjom

**Abstract**: Large Language Models (LLMs) have gained significant popularity in the last few years due to their performance in diverse tasks such as translation, prediction, or content generation. At the same time, the research community has shown that LLMs are susceptible to various attacks but can also improve the security of diverse systems. However, besides enabling more secure systems, how well do open source LLMs behave as covertext distributions to, e.g., facilitate censorship resistant communication?   In this paper, we explore the capabilities of open-source LLM-based covert channels. We approach this problem from the experimental side by empirically measuring the security vs. capacity of the open-source LLM model (Llama-7B) to assess how well it performs as a covert channel. Although our results indicate that such channels are not likely to achieve high practical bitrates, which depend on message length and model entropy, we also show that the chance for an adversary to detect covert communication is low. To ensure that our results can be used with the least effort as a general reference, we employ a conceptually simple and concise scheme and only assume public models.

摘要: 大型语言模型(LLM)在过去几年中因其在翻译、预测或内容生成等各种任务中的性能而广受欢迎。与此同时，研究界的研究表明，LLMS容易受到各种攻击，但也可以提高各种系统的安全性。然而，除了实现更安全的系统外，开源LLM作为隐蔽文本分发在促进抗审查通信方面的表现如何？在这篇文章中，我们探索了基于LLM的开源隐蔽通道的能力。我们从实验方面解决这个问题，通过经验测量开放源码LLM模型(LLAMA-7B)的安全性与容量，以评估它作为隐蔽通道的表现如何。虽然我们的结果表明，这种信道不太可能获得较高的实际比特率，这取决于消息长度和模型熵，但我们也表明，攻击者检测到隐蔽通信的机会很低。为了确保我们的结果能够以最小的努力作为一般参考，我们采用了一个概念上简单而简明的方案，并且只假设公共模型。



## **48. Efficient Adversarial Training in LLMs with Continuous Attacks**

在具有持续攻击的LLC中进行有效的对抗训练 cs.LG

19 pages, 4 figures

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2405.15589v1) [paper-pdf](http://arxiv.org/pdf/2405.15589v1)

**Authors**: Sophie Xhonneux, Alessandro Sordoni, Stephan Günnemann, Gauthier Gidel, Leo Schwinn

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that can bypass their safety guardrails. In many domains, adversarial training has proven to be one of the most promising methods to reliably improve robustness against such attacks. Yet, in the context of LLMs, current methods for adversarial training are hindered by the high computational costs required to perform discrete adversarial attacks at each training iteration. We address this problem by instead calculating adversarial attacks in the continuous embedding space of the LLM, which is orders of magnitudes more efficient. We propose a fast adversarial training algorithm (C-AdvUL) composed of two losses: the first makes the model robust on continuous embedding attacks computed on an adversarial behaviour dataset; the second ensures the usefulness of the final model by fine-tuning on utility data. Moreover, we introduce C-AdvIPO, an adversarial variant of IPO that does not require utility data for adversarially robust alignment. Our empirical evaluation on four models from different families (Gemma, Phi3, Mistral, Zephyr) and at different scales (2B, 3.8B, 7B) shows that both algorithms substantially enhance LLM robustness against discrete attacks (GCG, AutoDAN, PAIR), while maintaining utility. Our results demonstrate that robustness to continuous perturbations can extrapolate to discrete threat models. Thereby, we present a path toward scalable adversarial training algorithms for robustly aligning LLMs.

摘要: 大型语言模型(LLM)容易受到敌意攻击，这些攻击可以绕过它们的安全护栏。在许多领域，对抗性训练已被证明是可靠地提高对此类攻击的稳健性的最有前途的方法之一。然而，在LLMS的背景下，当前的对抗性训练方法受到在每次训练迭代中执行离散对抗性攻击所需的高计算成本的阻碍。我们通过在LLM的连续嵌入空间中计算敌意攻击来解决这个问题，该空间的效率要高出几个数量级。我们提出了一种快速对抗性训练算法(C-AdvUL)，该算法由两个损失组成：第一个损失使模型对基于对抗行为数据集的连续嵌入攻击具有健壮性；第二个损失通过对效用数据的微调来确保最终模型的有效性。此外，我们引入了C-AdvIPO，这是IPO的一个对抗性变体，不需要效用数据来进行对抗性强健的比对。我们在不同家族(Gema，Phi3，Mistral，Zephy)和不同尺度(2B，3.8B，7B)的四个模型上的实验评估表明，这两种算法在保持实用性的同时，显著提高了LLM对离散攻击(GCG，AutoDAN，Pair)的稳健性。我们的结果表明，对连续扰动的稳健性可以外推到离散威胁模型。因此，我们提出了一条可扩展的对抗性训练算法，用于鲁棒对齐LLM。



## **49. DAGER: Exact Gradient Inversion for Large Language Models**

DAGER：大型语言模型的精确梯度倒置 cs.LG

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2405.15586v1) [paper-pdf](http://arxiv.org/pdf/2405.15586v1)

**Authors**: Ivo Petrov, Dimitar I. Dimitrov, Maximilian Baader, Mark Niklas Müller, Martin Vechev

**Abstract**: Federated learning works by aggregating locally computed gradients from multiple clients, thus enabling collaborative training without sharing private client data. However, prior work has shown that the data can actually be recovered by the server using so-called gradient inversion attacks. While these attacks perform well when applied on images, they are limited in the text domain and only permit approximate reconstruction of small batches and short input sequences. In this work, we propose DAGER, the first algorithm to recover whole batches of input text exactly. DAGER leverages the low-rank structure of self-attention layer gradients and the discrete nature of token embeddings to efficiently check if a given token sequence is part of the client data. We use this check to exactly recover full batches in the honest-but-curious setting without any prior on the data for both encoder- and decoder-based architectures using exhaustive heuristic search and a greedy approach, respectively. We provide an efficient GPU implementation of DAGER and show experimentally that it recovers full batches of size up to 128 on large language models (LLMs), beating prior attacks in speed (20x at same batch size), scalability (10x larger batches), and reconstruction quality (ROUGE-1/2 > 0.99).

摘要: 联合学习的工作方式是聚合来自多个客户端的本地计算的梯度，从而在不共享私人客户端数据的情况下实现协作培训。然而，先前的工作表明，服务器实际上可以使用所谓的梯度反转攻击来恢复数据。虽然这些攻击在图像上应用时表现良好，但它们仅限于文本域，仅允许对小批次和短输入序列进行近似重建。在这项工作中，我们提出了第一个准确恢复整批输入文本的算法Dager。Dager利用自我关注层梯度的低等级结构和令牌嵌入的离散性质来有效地检查给定的令牌序列是否是客户端数据的一部分。我们使用这种检查，分别使用穷举启发式搜索和贪婪方法，在诚实但奇怪的设置中准确地恢复完整批次，而不需要对基于编码器和解码器的架构的数据进行任何先验。我们提供了一种高效的Dager的GPU实现，实验表明，它可以在大型语言模型(LLM)上恢复大小高达128的全批处理，在速度(相同批处理大小的20倍)、可伸缩性(大批处理10倍)和重建质量(Rouge-1/2>0.99)方面优于先前的攻击。



## **50. Mosaic Memory: Fuzzy Duplication in Copyright Traps for Large Language Models**

马赛克记忆：大型语言模型版权陷阱中的模糊重复 cs.CL

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2405.15523v1) [paper-pdf](http://arxiv.org/pdf/2405.15523v1)

**Authors**: Igor Shilov, Matthieu Meeus, Yves-Alexandre de Montjoye

**Abstract**: The immense datasets used to develop Large Language Models (LLMs) often include copyright-protected content, typically without the content creator's consent. Copyright traps have been proposed to be injected into the original content, improving content detectability in newly released LLMs. Traps, however, rely on the exact duplication of a unique text sequence, leaving them vulnerable to commonly deployed data deduplication techniques. We here propose the generation of fuzzy copyright traps, featuring slight modifications across duplication. When injected in the fine-tuning data of a 1.3B LLM, we show fuzzy trap sequences to be memorized nearly as well as exact duplicates. Specifically, the Membership Inference Attack (MIA) ROC AUC only drops from 0.90 to 0.87 when 4 tokens are replaced across the fuzzy duplicates. We also find that selecting replacement positions to minimize the exact overlap between fuzzy duplicates leads to similar memorization, while making fuzzy duplicates highly unlikely to be removed by any deduplication process. Lastly, we argue that the fact that LLMs memorize across fuzzy duplicates challenges the study of LLM memorization relying on naturally occurring duplicates. Indeed, we find that the commonly used training dataset, The Pile, contains significant amounts of fuzzy duplicates. This introduces a previously unexplored confounding factor in post-hoc studies of LLM memorization, and questions the effectiveness of (exact) data deduplication as a privacy protection technique.

摘要: 用于开发大型语言模型(LLM)的海量数据集通常包括受版权保护的内容，通常没有内容创建者的同意。有人提议将版权陷阱注入到原始内容中，以提高新发布的LLM中的内容可检测性。然而，陷阱依赖于唯一文本序列的精确复制，这使得它们很容易受到常用的重复数据消除技术的影响。我们在这里建议生成模糊版权陷阱，其特点是在复制过程中进行轻微修改。当注入1.3B激光测深机的微调数据时，我们发现模糊圈闭序列几乎可以被记忆，而且可以精确复制。具体地说，当在模糊副本上替换4个令牌时，成员关系推理攻击(MIA)ROC AUC仅从0.90下降到0.87。我们还发现，选择替换位置以最小化模糊副本之间的精确重叠会导致类似的记忆，同时使模糊副本极不可能被任何去重过程移除。最后，我们认为LLM跨模糊复制记忆的事实对依赖自然发生复制的LLM记忆研究提出了挑战。事实上，我们发现常用的训练数据集，即堆，包含大量的模糊重复项。这在LLM记忆的后续研究中引入了一个以前未被探索的混杂因素，并质疑(精确)重复数据删除作为隐私保护技术的有效性。



