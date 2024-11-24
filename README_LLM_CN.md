# Latest Large Language Model Attack Papers
**update at 2024-11-24 12:14:29**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. GASP: Efficient Black-Box Generation of Adversarial Suffixes for Jailbreaking LLMs**

GISP：针对越狱LLM的对抗性后缀的高效黑匣子生成 cs.LG

28 pages, 9 tables, 13 figures; under review at CVPR '25

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.14133v1) [paper-pdf](http://arxiv.org/pdf/2411.14133v1)

**Authors**: Advik Raj Basani, Xiao Zhang

**Abstract**: Large Language Models (LLMs) have shown impressive proficiency across a range of natural language processing tasks yet remain vulnerable to adversarial prompts, known as jailbreak attacks, carefully designed to elicit harmful responses from LLMs. Traditional methods rely on manual heuristics, which suffer from limited generalizability. While being automatic, optimization-based attacks often produce unnatural jailbreak prompts that are easy to detect by safety filters or require high computational overhead due to discrete token optimization. Witnessing the limitations of existing jailbreak methods, we introduce Generative Adversarial Suffix Prompter (GASP), a novel framework that combines human-readable prompt generation with Latent Bayesian Optimization (LBO) to improve adversarial suffix creation in a fully black-box setting. GASP leverages LBO to craft adversarial suffixes by efficiently exploring continuous embedding spaces, gradually optimizing the model to improve attack efficacy while balancing prompt coherence through a targeted iterative refinement procedure. Our experiments show that GASP can generate natural jailbreak prompts, significantly improving attack success rates, reducing training times, and accelerating inference speed, thus making it an efficient and scalable solution for red-teaming LLMs.

摘要: 大型语言模型(LLM)在一系列自然语言处理任务中表现出令人印象深刻的熟练程度，但仍然容易受到对手提示的攻击，这种提示被称为越狱攻击，这些提示是精心设计的，旨在引起LLM的有害反应。传统的方法依赖于人工启发式方法，泛化能力有限。虽然基于优化的攻击是自动的，但通常会产生不自然的越狱提示，这些提示很容易被安全过滤器检测到，或者由于离散令牌优化而需要很高的计算开销。鉴于现有越狱方法的局限性，我们引入了生成性对抗性后缀提示器(GAP)，这是一种将人类可读的提示生成与潜在贝叶斯优化(LBO)相结合的新框架，以改进完全黑盒环境下的对抗性后缀创建。GASP利用LBO通过有效地探索连续嵌入空间来创建对抗性后缀，逐步优化模型以提高攻击效率，同时通过有针对性的迭代细化过程平衡即时一致性。我们的实验表明，GAP能够生成自然的越狱提示，显著提高了攻击成功率，减少了训练次数，加快了推理速度，从而使其成为红队LLMS的一种高效和可扩展的解决方案。



## **2. RAG-Thief: Scalable Extraction of Private Data from Retrieval-Augmented Generation Applications with Agent-based Attacks**

RAG-Thief：利用基于代理的攻击从检索增强生成应用程序中可扩展地提取私人数据 cs.CR

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.14110v1) [paper-pdf](http://arxiv.org/pdf/2411.14110v1)

**Authors**: Changyue Jiang, Xudong Pan, Geng Hong, Chenfu Bao, Min Yang

**Abstract**: While large language models (LLMs) have achieved notable success in generative tasks, they still face limitations, such as lacking up-to-date knowledge and producing hallucinations. Retrieval-Augmented Generation (RAG) enhances LLM performance by integrating external knowledge bases, providing additional context which significantly improves accuracy and knowledge coverage. However, building these external knowledge bases often requires substantial resources and may involve sensitive information. In this paper, we propose an agent-based automated privacy attack called RAG-Thief, which can extract a scalable amount of private data from the private database used in RAG applications. We conduct a systematic study on the privacy risks associated with RAG applications, revealing that the vulnerability of LLMs makes the private knowledge bases suffer significant privacy risks. Unlike previous manual attacks which rely on traditional prompt injection techniques, RAG-Thief starts with an initial adversarial query and learns from model responses, progressively generating new queries to extract as many chunks from the knowledge base as possible. Experimental results show that our RAG-Thief can extract over 70% information from the private knowledge bases within customized RAG applications deployed on local machines and real-world platforms, including OpenAI's GPTs and ByteDance's Coze. Our findings highlight the privacy vulnerabilities in current RAG applications and underscore the pressing need for stronger safeguards.

摘要: 虽然大型语言模型在生成性任务中取得了显著的成功，但它们仍然面临着局限性，如缺乏最新知识和产生幻觉。检索-增强生成(RAG)通过集成外部知识库来增强LLM性能，提供额外的上下文，从而显著提高准确性和知识覆盖率。然而，建立这些外部知识库往往需要大量资源，并可能涉及敏感信息。本文提出了一种基于代理的自动隐私攻击方法RAG-Thief，它可以从RAG应用中使用的私有数据库中提取大量可伸缩的私有数据。我们对RAG应用相关的隐私风险进行了系统的研究，揭示了LLMS的漏洞使私人知识库面临着重大的隐私风险。与以前依赖传统提示注入技术的手动攻击不同，RAG-Thief从最初的对抗性查询开始，并从模型响应中学习，逐步生成新的查询以从知识库中提取尽可能多的块。实验结果表明，我们的RAG-Thief可以从本地机器和真实平台上部署的定制RAG应用程序的私有知识库中提取70%以上的信息，包括OpenAI的GPTS和ByteDance的Coze。我们的发现突显了当前RAG应用程序中的隐私漏洞，并强调了加强保护的迫切需要。



## **3. Verifying the Robustness of Automatic Credibility Assessment**

验证自动可信度评估的稳健性 cs.CL

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2303.08032v3) [paper-pdf](http://arxiv.org/pdf/2303.08032v3)

**Authors**: Piotr Przybyła, Alexander Shvets, Horacio Saggion

**Abstract**: Text classification methods have been widely investigated as a way to detect content of low credibility: fake news, social media bots, propaganda, etc. Quite accurate models (likely based on deep neural networks) help in moderating public electronic platforms and often cause content creators to face rejection of their submissions or removal of already published texts. Having the incentive to evade further detection, content creators try to come up with a slightly modified version of the text (known as an attack with an adversarial example) that exploit the weaknesses of classifiers and result in a different output. Here we systematically test the robustness of common text classifiers against available attacking techniques and discover that, indeed, meaning-preserving changes in input text can mislead the models. The approaches we test focus on finding vulnerable spans in text and replacing individual characters or words, taking into account the similarity between the original and replacement content. We also introduce BODEGA: a benchmark for testing both victim models and attack methods on four misinformation detection tasks in an evaluation framework designed to simulate real use-cases of content moderation. The attacked tasks include (1) fact checking and detection of (2) hyperpartisan news, (3) propaganda and (4) rumours. Our experimental results show that modern large language models are often more vulnerable to attacks than previous, smaller solutions, e.g. attacks on GEMMA being up to 27\% more successful than those on BERT. Finally, we manually analyse a subset adversarial examples and check what kinds of modifications are used in successful attacks.

摘要: 文本分类方法被广泛研究为检测可信度较低的内容的一种方式：假新闻、社交媒体机器人、宣传等。相当准确的模型(可能基于深度神经网络)有助于调节公共电子平台，并经常导致内容创建者面临提交的拒绝或已发布的文本的删除。出于逃避进一步检测的动机，内容创建者试图对文本进行稍微修改的版本(称为带有敌意的示例的攻击)，以利用分类器的弱点并产生不同的输出。在这里，我们系统地测试了常见文本分类器对现有攻击技术的健壮性，并发现确实，输入文本中保持意义的变化会误导模型。我们测试的方法侧重于查找文本中易受攻击的范围，并替换单个字符或单词，同时考虑到原始内容和替换内容之间的相似性。我们还引入了Bodega：一个基准，用于在四个错误信息检测任务中测试受害者模型和攻击方法，该评估框架旨在模拟真实的内容审核用例。被攻击的任务包括(1)事实核查和检测(2)超党派新闻，(3)宣传和(4)谣言。我们的实验结果表明，现代大语言模型往往比以前的较小的解决方案更容易受到攻击，例如，对Gema的攻击比对Bert的攻击成功高达27%.最后，我们手动分析了一个子集的敌意例子，并检查了在成功的攻击中使用了哪些修改。



## **4. Next-Generation Phishing: How LLM Agents Empower Cyber Attackers**

下一代网络钓鱼：LLM代理如何为网络攻击者提供帮助 cs.CR

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.13874v1) [paper-pdf](http://arxiv.org/pdf/2411.13874v1)

**Authors**: Khalifa Afane, Wenqi Wei, Ying Mao, Junaid Farooq, Juntao Chen

**Abstract**: The escalating threat of phishing emails has become increasingly sophisticated with the rise of Large Language Models (LLMs). As attackers exploit LLMs to craft more convincing and evasive phishing emails, it is crucial to assess the resilience of current phishing defenses. In this study we conduct a comprehensive evaluation of traditional phishing detectors, such as Gmail Spam Filter, Apache SpamAssassin, and Proofpoint, as well as machine learning models like SVM, Logistic Regression, and Naive Bayes, in identifying both traditional and LLM-rephrased phishing emails. We also explore the emerging role of LLMs as phishing detection tools, a method already adopted by companies like NTT Security Holdings and JPMorgan Chase. Our results reveal notable declines in detection accuracy for rephrased emails across all detectors, highlighting critical weaknesses in current phishing defenses. As the threat landscape evolves, our findings underscore the need for stronger security controls and regulatory oversight on LLM-generated content to prevent its misuse in creating advanced phishing attacks. This study contributes to the development of more effective Cyber Threat Intelligence (CTI) by leveraging LLMs to generate diverse phishing variants that can be used for data augmentation, harnessing the power of LLMs to enhance phishing detection, and paving the way for more robust and adaptable threat detection systems.

摘要: 随着大型语言模型(LLM)的兴起，钓鱼电子邮件日益升级的威胁变得越来越复杂。随着攻击者利用LLMS来编制更具说服力和闪避性的网络钓鱼电子邮件，评估当前网络钓鱼防御的弹性至关重要。在这项研究中，我们对传统的钓鱼检测器进行了全面的评估，如Gmail垃圾邮件过滤器、ApacheSpamassassin和Proofpoint，以及机器学习模型如支持向量机、Logistic回归和朴素贝叶斯，在识别传统和LLM重述的钓鱼电子邮件方面进行了全面的评估。我们还探讨了LLMS作为钓鱼检测工具的新兴角色，这种方法已经被NTT Security Holdings和摩根大通等公司采用。我们的结果显示，在所有检测器上，对重新措辞的电子邮件的检测准确率都出现了显著下降，突显了当前网络钓鱼防御系统的关键弱点。随着威胁格局的演变，我们的发现强调了对LLM生成的内容进行更严格的安全控制和监管的必要性，以防止其在制造高级网络钓鱼攻击时被滥用。这项研究有助于开发更有效的网络威胁情报(CTI)，方法是利用LLMS生成可用于数据增强的各种网络钓鱼变体，利用LLMS的能力来增强网络钓鱼检测，并为更强大和适应性更强的威胁检测系统铺平道路。



## **5. TransLinkGuard: Safeguarding Transformer Models Against Model Stealing in Edge Deployment**

TransLinkGuard：保护Transformer模型，防止边缘部署中的模型窃取 cs.CR

Accepted by ACM MM24 Conference

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2404.11121v2) [paper-pdf](http://arxiv.org/pdf/2404.11121v2)

**Authors**: Qinfeng Li, Zhiqiang Shen, Zhenghan Qin, Yangfan Xie, Xuhong Zhang, Tianyu Du, Jianwei Yin

**Abstract**: Proprietary large language models (LLMs) have been widely applied in various scenarios. Additionally, deploying LLMs on edge devices is trending for efficiency and privacy reasons. However, edge deployment of proprietary LLMs introduces new security challenges: edge-deployed models are exposed as white-box accessible to users, enabling adversaries to conduct effective model stealing (MS) attacks. Unfortunately, existing defense mechanisms fail to provide effective protection. Specifically, we identify four critical protection properties that existing methods fail to simultaneously satisfy: (1) maintaining protection after a model is physically copied; (2) authorizing model access at request level; (3) safeguarding runtime reverse engineering; (4) achieving high security with negligible runtime overhead. To address the above issues, we propose TransLinkGuard, a plug-and-play model protection approach against model stealing on edge devices. The core part of TransLinkGuard is a lightweight authorization module residing in a secure environment, e.g., TEE. The authorization module can freshly authorize each request based on its input. Extensive experiments show that TransLinkGuard achieves the same security protection as the black-box security guarantees with negligible overhead.

摘要: 专有的大型语言模型(LLM)已广泛应用于各种场景。此外，出于效率和隐私的原因，在边缘设备上部署LLM是一种趋势。然而，专有LLMS的边缘部署带来了新的安全挑战：边缘部署的模型暴露为用户可访问的白盒，使对手能够进行有效的模型窃取(MS)攻击。不幸的是，现有的防御机制未能提供有效的保护。具体地说，我们确定了现有方法无法同时满足的四个关键保护性质：(1)在物理复制模型后保持保护；(2)在请求级授权模型访问；(3)保护运行时逆向工程；(4)以可忽略的运行时开销实现高安全性。为了解决上述问题，我们提出了一种针对边缘设备上的模型窃取的即插即用模型保护方法TransLinkGuard。TransLinkGuard的核心部分是驻留在安全环境中的轻量级授权模块，例如TEE。授权模块可以基于其输入对每个请求进行新的授权。大量实验表明，TransLinkGuard实现了与黑盒安全保证相同的安全保护，而开销可以忽略不计。



## **6. AttentionBreaker: Adaptive Evolutionary Optimization for Unmasking Vulnerabilities in LLMs through Bit-Flip Attacks**

AttributionBreaker：通过位翻转攻击揭露LLM中漏洞的自适应进化优化 cs.CR

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.13757v1) [paper-pdf](http://arxiv.org/pdf/2411.13757v1)

**Authors**: Sanjay Das, Swastik Bhattacharya, Souvik Kundu, Shamik Kundu, Anand Menon, Arnab Raha, Kanad Basu

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing (NLP), excelling in tasks like text generation and summarization. However, their increasing adoption in mission-critical applications raises concerns about hardware-based threats, particularly bit-flip attacks (BFAs). BFAs, enabled by fault injection methods such as Rowhammer, target model parameters in memory, compromising both integrity and performance. Identifying critical parameters for BFAs in the vast parameter space of LLMs poses significant challenges. While prior research suggests transformer-based architectures are inherently more robust to BFAs compared to traditional deep neural networks, we challenge this assumption. For the first time, we demonstrate that as few as three bit-flips can cause catastrophic performance degradation in an LLM with billions of parameters. Current BFA techniques are inadequate for exploiting this vulnerability due to the difficulty of efficiently identifying critical parameters within the immense parameter space. To address this, we propose AttentionBreaker, a novel framework tailored for LLMs that enables efficient traversal of the parameter space to identify critical parameters. Additionally, we introduce GenBFA, an evolutionary optimization strategy designed to refine the search further, isolating the most critical bits for an efficient and effective attack. Empirical results reveal the profound vulnerability of LLMs to AttentionBreaker. For example, merely three bit-flips (4.129 x 10^-9% of total parameters) in the LLaMA3-8B-Instruct 8-bit quantized (W8) model result in a complete performance collapse: accuracy on MMLU tasks drops from 67.3% to 0%, and Wikitext perplexity skyrockets from 12.6 to 4.72 x 10^5. These findings underscore the effectiveness of AttentionBreaker in uncovering and exploiting critical vulnerabilities within LLM architectures.

摘要: 大型语言模型(LLM)使自然语言处理(NLP)发生了革命性的变化，在文本生成和摘要等任务中表现出色。然而，它们在任务关键型应用中的日益采用引发了对基于硬件的威胁的担忧，特别是位翻转攻击(BFA)。由Rowhammer等故障注入方法启用的BFA以内存中的模型参数为目标，损害了完整性和性能。在LLMS庞大的参数空间中识别BFA的关键参数带来了巨大的挑战。虽然先前的研究表明，与传统的深度神经网络相比，基于变压器的体系结构对BFA具有更强的鲁棒性，但我们对这一假设提出了质疑。我们首次证明，在具有数十亿个参数的LLM中，仅三个比特翻转就会导致灾难性的性能下降。由于很难在巨大的参数空间中有效地识别关键参数，因此当前的BFA技术不足以利用该漏洞。为了解决这个问题，我们提出了AttentionBreaker，这是一个为LLMS量身定做的新框架，能够有效地遍历参数空间来识别关键参数。此外，我们还引入了GenBFA，这是一种进化优化策略，旨在进一步细化搜索，隔离最关键的比特，以实现高效和有效的攻击。实证结果揭示了LLMS对AttentionBreaker的严重脆弱性。例如，在LLAMA3-8B指令8位量化(W8)模型中，仅三次位翻转(占总参数的4.129 x 10^-9%)就会导致完全的性能崩溃：MMLU任务的准确率从67.3%下降到0%，而Wikitext的复杂性从12.6x10^5飙升到4.72x10^5。这些发现突显了AttentionBreaker在发现和利用LLM体系结构中的关键漏洞方面的有效性。



## **7. SoK: A Systems Perspective on Compound AI Threats and Countermeasures**

SoK：复合人工智能威胁和对策的系统视角 cs.CR

13 pages, 4 figures, 2 tables

**SubmitDate**: 2024-11-20    [abs](http://arxiv.org/abs/2411.13459v1) [paper-pdf](http://arxiv.org/pdf/2411.13459v1)

**Authors**: Sarbartha Banerjee, Prateek Sahu, Mulong Luo, Anjo Vahldiek-Oberwagner, Neeraja J. Yadwadkar, Mohit Tiwari

**Abstract**: Large language models (LLMs) used across enterprises often use proprietary models and operate on sensitive inputs and data. The wide range of attack vectors identified in prior research - targeting various software and hardware components used in training and inference - makes it extremely challenging to enforce confidentiality and integrity policies.   As we advance towards constructing compound AI inference pipelines that integrate multiple large language models (LLMs), the attack surfaces expand significantly. Attackers now focus on the AI algorithms as well as the software and hardware components associated with these systems. While current research often examines these elements in isolation, we find that combining cross-layer attack observations can enable powerful end-to-end attacks with minimal assumptions about the threat model. Given, the sheer number of existing attacks at each layer, we need a holistic and systemized understanding of different attack vectors at each layer.   This SoK discusses different software and hardware attacks applicable to compound AI systems and demonstrates how combining multiple attack mechanisms can reduce the threat model assumptions required for an isolated attack. Next, we systematize the ML attacks in lines with the Mitre Att&ck framework to better position each attack based on the threat model. Finally, we outline the existing countermeasures for both software and hardware layers and discuss the necessity of a comprehensive defense strategy to enable the secure and high-performance deployment of compound AI systems.

摘要: 跨企业使用的大型语言模型(LLM)通常使用专有模型，并对敏感输入和数据进行操作。在以前的研究中发现了广泛的攻击载体-以训练和推理中使用的各种软件和硬件组件为目标-这使得执行机密性和完整性策略变得极其困难。随着我们朝着构建集成多个大型语言模型(LLM)的复合AI推理管道的方向发展，攻击面显著扩大。攻击者现在把重点放在人工智能算法以及与这些系统相关的软件和硬件组件上。虽然目前的研究经常孤立地检查这些元素，但我们发现，结合跨层攻击观察可以在对威胁模型的最小假设下实现强大的端到端攻击。鉴于每一层现有攻击的绝对数量，我们需要对每一层的不同攻击载体进行全面和系统化的了解。本SOK讨论了适用于复合AI系统的不同软件和硬件攻击，并演示了如何结合多种攻击机制来减少孤立攻击所需的威胁模型假设。接下来，我们按照Mitre Att&CK框架对ML攻击进行系统化，以便更好地定位基于威胁模型的每一种攻击。最后，我们从软件和硬件两个层面概述了现有的对策，并讨论了为实现复合人工智能系统的安全和高性能部署而制定综合防御策略的必要性。



## **8. WaterPark: A Robustness Assessment of Language Model Watermarking**

WaterPark：语言模型水印的稳健性评估 cs.CR

22 pages

**SubmitDate**: 2024-11-20    [abs](http://arxiv.org/abs/2411.13425v1) [paper-pdf](http://arxiv.org/pdf/2411.13425v1)

**Authors**: Jiacheng Liang, Zian Wang, Lauren Hong, Shouling Ji, Ting Wang

**Abstract**: To mitigate the misuse of large language models (LLMs), such as disinformation, automated phishing, and academic cheating, there is a pressing need for the capability of identifying LLM-generated texts. Watermarking emerges as one promising solution: it plants statistical signals into LLMs' generative processes and subsequently verifies whether LLMs produce given texts. Various watermarking methods (``watermarkers'') have been proposed; yet, due to the lack of unified evaluation platforms, many critical questions remain under-explored: i) What are the strengths/limitations of various watermarkers, especially their attack robustness? ii) How do various design choices impact their robustness? iii) How to optimally operate watermarkers in adversarial environments?   To fill this gap, we systematize existing LLM watermarkers and watermark removal attacks, mapping out their design spaces. We then develop WaterPark, a unified platform that integrates 10 state-of-the-art watermarkers and 12 representative attacks. More importantly, leveraging WaterPark, we conduct a comprehensive assessment of existing watermarkers, unveiling the impact of various design choices on their attack robustness. For instance, a watermarker's resilience to increasingly intensive attacks hinges on its context dependency. We further explore the best practices to operate watermarkers in adversarial environments. For instance, using a generic detector alongside a watermark-specific detector improves the security of vulnerable watermarkers. We believe our study sheds light on current LLM watermarking techniques while WaterPark serves as a valuable testbed to facilitate future research.

摘要: 为了减少对大型语言模型(LLM)的滥用，如虚假信息、自动网络钓鱼和学术作弊，迫切需要识别LLM生成的文本的能力。数字水印作为一种很有前途的解决方案出现了：它将统计信号植入LLMS的生成过程中，随后验证LLMS是否生成给定的文本。人们已经提出了各种水印方法，然而，由于缺乏统一的评估平台，许多关键问题仍然没有得到充分的探讨：i)各种水印的优点/局限性是什么，特别是它们的攻击稳健性？Ii)各种设计选择对其健壮性有何影响？三)如何在对抗性环境中以最佳方式使用水印？为了填补这一空白，我们对现有的LLM水印和水印移除攻击进行了系统化，规划了它们的设计空间。然后我们开发了Water Park，这是一个统一的平台，集成了10个最先进的水印和12个具有代表性的攻击。更重要的是，利用水上公园，我们对现有的水印进行了全面的评估，揭示了各种设计选择对其攻击健壮性的影响。例如，水印对日益激烈的攻击的适应能力取决于它的上下文依赖性。我们进一步探索在对抗性环境中操作水印的最佳实践。例如，在水印专用检测器旁边使用通用检测器可以提高易受攻击的水印的安全性。我们相信我们的研究对当前的LLM数字水印技术有一定的启发作用，同时也为以后的研究提供了一个有价值的实验平台。



## **9. CryptoFormalEval: Integrating LLMs and Formal Verification for Automated Cryptographic Protocol Vulnerability Detection**

CryptoFormalEval：集成LLM和形式验证以实现自动加密协议漏洞检测 cs.CR

**SubmitDate**: 2024-11-20    [abs](http://arxiv.org/abs/2411.13627v1) [paper-pdf](http://arxiv.org/pdf/2411.13627v1)

**Authors**: Cristian Curaba, Denis D'Ambrosi, Alessandro Minisini, Natalia Pérez-Campanero Antolín

**Abstract**: Cryptographic protocols play a fundamental role in securing modern digital infrastructure, but they are often deployed without prior formal verification. This could lead to the adoption of distributed systems vulnerable to attack vectors. Formal verification methods, on the other hand, require complex and time-consuming techniques that lack automatization. In this paper, we introduce a benchmark to assess the ability of Large Language Models (LLMs) to autonomously identify vulnerabilities in new cryptographic protocols through interaction with Tamarin: a theorem prover for protocol verification. We created a manually validated dataset of novel, flawed, communication protocols and designed a method to automatically verify the vulnerabilities found by the AI agents. Our results about the performances of the current frontier models on the benchmark provides insights about the possibility of cybersecurity applications by integrating LLMs with symbolic reasoning systems.

摘要: 加密协议在保护现代数字基础设施方面发挥着基础作用，但它们通常在没有事先正式验证的情况下部署。这可能会导致采用容易受到攻击载体的分布式系统。另一方面，形式验证方法需要复杂且耗时的技术，而缺乏自动化。在本文中，我们引入了一个基准来评估大型语言模型（LLM）通过与Tamarin交互来自主识别新加密协议中漏洞的能力：Tamarin是协议验证的定理证明者。我们创建了一个手动验证的新颖、有缺陷的通信协议数据集，并设计了一种自动验证人工智能代理发现的漏洞的方法。我们关于当前前沿模型在基准上的性能的结果提供了关于通过将LLM与符号推理系统集成来实现网络安全应用的可能性的见解。



## **10. TAPT: Test-Time Adversarial Prompt Tuning for Robust Inference in Vision-Language Models**

TAPT：测试时对抗快速调整视觉语言模型中的鲁棒推理 cs.CV

**SubmitDate**: 2024-11-20    [abs](http://arxiv.org/abs/2411.13136v1) [paper-pdf](http://arxiv.org/pdf/2411.13136v1)

**Authors**: Xin Wang, Kai Chen, Jiaming Zhang, Jingjing Chen, Xingjun Ma

**Abstract**: Large pre-trained Vision-Language Models (VLMs) such as CLIP have demonstrated excellent zero-shot generalizability across various downstream tasks. However, recent studies have shown that the inference performance of CLIP can be greatly degraded by small adversarial perturbations, especially its visual modality, posing significant safety threats. To mitigate this vulnerability, in this paper, we propose a novel defense method called Test-Time Adversarial Prompt Tuning (TAPT) to enhance the inference robustness of CLIP against visual adversarial attacks. TAPT is a test-time defense method that learns defensive bimodal (textual and visual) prompts to robustify the inference process of CLIP. Specifically, it is an unsupervised method that optimizes the defensive prompts for each test sample by minimizing a multi-view entropy and aligning adversarial-clean distributions. We evaluate the effectiveness of TAPT on 11 benchmark datasets, including ImageNet and 10 other zero-shot datasets, demonstrating that it enhances the zero-shot adversarial robustness of the original CLIP by at least 48.9% against AutoAttack (AA), while largely maintaining performance on clean examples. Moreover, TAPT outperforms existing adversarial prompt tuning methods across various backbones, achieving an average robustness improvement of at least 36.6%.

摘要: 大型预先训练的视觉语言模型(VLM)，如CLIP，已经在各种下游任务中表现出出色的零射击泛化能力。然而，最近的研究表明，CLIP的推理性能会因小的对抗性扰动而大大降低，特别是它的视觉通道，构成了严重的安全威胁。为了缓解这一漏洞，本文提出了一种新的防御方法，称为测试时间对抗性提示调整(TAPT)，以增强CLIP对视觉对抗性攻击的推理健壮性。TAPT是一种测试时防御方法，它学习防御性双峰(文本和视觉)提示，以巩固CLIP的推理过程。具体地说，它是一种无监督的方法，通过最小化多视图熵和对齐对抗性干净的分布来优化每个测试样本的防御提示。我们在11个基准数据集上对TAPT的有效性进行了评估，包括ImageNet和其他10个零镜头数据集，结果表明，它在很大程度上保持了在干净样本上的性能，但相对于AutoAttack(AA)，它至少提高了原始剪辑的零镜头对抗健壮性48.9%。此外，TAPT在不同主干上的性能优于现有的对抗性提示调优方法，实现了平均至少36.6%的健壮性改进。



## **11. When Backdoors Speak: Understanding LLM Backdoor Attacks Through Model-Generated Explanations**

当后门说话时：通过模型生成的解释了解LLM后门攻击 cs.CR

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2411.12701v1) [paper-pdf](http://arxiv.org/pdf/2411.12701v1)

**Authors**: Huaizhi Ge, Yiming Li, Qifan Wang, Yongfeng Zhang, Ruixiang Tang

**Abstract**: Large Language Models (LLMs) are vulnerable to backdoor attacks, where hidden triggers can maliciously manipulate model behavior. While several backdoor attack methods have been proposed, the mechanisms by which backdoor functions operate in LLMs remain underexplored. In this paper, we move beyond attacking LLMs and investigate backdoor functionality through the novel lens of natural language explanations. Specifically, we leverage LLMs' generative capabilities to produce human-understandable explanations for their decisions, allowing us to compare explanations for clean and poisoned samples. We explore various backdoor attacks and embed the backdoor into LLaMA models for multiple tasks. Our experiments show that backdoored models produce higher-quality explanations for clean data compared to poisoned data, while generating significantly more consistent explanations for poisoned data than for clean data. We further analyze the explanation generation process, revealing that at the token level, the explanation token of poisoned samples only appears in the final few transformer layers of the LLM. At the sentence level, attention dynamics indicate that poisoned inputs shift attention from the input context when generating the explanation. These findings deepen our understanding of backdoor attack mechanisms in LLMs and offer a framework for detecting such vulnerabilities through explainability techniques, contributing to the development of more secure LLMs.

摘要: 大型语言模型(LLM)容易受到后门攻击，在后门攻击中，隐藏的触发器可以恶意操纵模型行为。虽然已经提出了几种后门攻击方法，但后门功能在LLM中运行的机制仍未得到充分探索。在这篇文章中，我们超越了攻击LLM，通过自然语言解释的新视角来研究后门功能。具体地说，我们利用LLMS的生成能力来为他们的决定产生人类可以理解的解释，使我们能够比较干净和有毒样本的解释。我们探索了各种后门攻击，并将后门嵌入到骆驼模型中，以实现多种任务。我们的实验表明，与有毒数据相比，回溯模型对干净数据产生了更高质量的解释，而对有毒数据产生的解释比对干净数据产生的解释要一致得多。我们进一步分析了解释的生成过程，发现在令牌级别，有毒样本的解释令牌只出现在LLM的最后几个转换器层。在句子层面，注意动力学表明，有毒输入在生成解释时转移了对输入上下文的注意力。这些发现加深了我们对LLMS后门攻击机制的理解，并提供了一个通过可解释性技术检测此类漏洞的框架，有助于开发更安全的LLMS。



## **12. Exploring adversarial robustness of JPEG AI: methodology, comparison and new methods**

探索JPEG AI的对抗鲁棒性：方法论、比较和新方法 eess.IV

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11795v1) [paper-pdf](http://arxiv.org/pdf/2411.11795v1)

**Authors**: Egor Kovalev, Georgii Bychkov, Khaled Abud, Aleksandr Gushchin, Anna Chistyakova, Sergey Lavrushkin, Dmitriy Vatolin, Anastasia Antsiferova

**Abstract**: Adversarial robustness of neural networks is an increasingly important area of research, combining studies on computer vision models, large language models (LLMs), and others. With the release of JPEG AI - the first standard for end-to-end neural image compression (NIC) methods - the question of its robustness has become critically significant. JPEG AI is among the first international, real-world applications of neural-network-based models to be embedded in consumer devices. However, research on NIC robustness has been limited to open-source codecs and a narrow range of attacks. This paper proposes a new methodology for measuring NIC robustness to adversarial attacks. We present the first large-scale evaluation of JPEG AI's robustness, comparing it with other NIC models. Our evaluation results and code are publicly available online (link is hidden for a blind review).

摘要: 神经网络的对抗鲁棒性是一个越来越重要的研究领域，结合了对计算机视觉模型、大型语言模型（LLM）等的研究。随着JPEG AI（端到端神经图像压缩（NIC）方法的第一个标准）的发布，其稳健性问题变得至关重要。JPEG AI是首批嵌入消费设备的基于神经网络的模型的国际现实应用之一。然而，关于NIC稳健性的研究仅限于开源编解码器和范围狭窄的攻击。本文提出了一种新的方法来衡量NIC对对抗性攻击的稳健性。我们首次对JPEG AI的稳健性进行了大规模评估，并将其与其他NIC模型进行了比较。我们的评估结果和代码可在线公开（链接已隐藏，以供盲目审查）。



## **13. DAWN: Designing Distributed Agents in a Worldwide Network**

DAWN：在全球网络中设计分布式代理 cs.NI

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2410.22339v2) [paper-pdf](http://arxiv.org/pdf/2410.22339v2)

**Authors**: Zahra Aminiranjbar, Jianan Tang, Qiudan Wang, Shubha Pant, Mahesh Viswanathan

**Abstract**: The rapid evolution of Large Language Models (LLMs) has transformed them from basic conversational tools into sophisticated entities capable of complex reasoning and decision-making. These advancements have led to the development of specialized LLM-based agents designed for diverse tasks such as coding and web browsing. As these agents become more capable, the need for a robust framework that facilitates global communication and collaboration among them towards advanced objectives has become increasingly critical. Distributed Agents in a Worldwide Network (DAWN) addresses this need by offering a versatile framework that integrates LLM-based agents with traditional software systems, enabling the creation of agentic applications suited for a wide range of use cases. DAWN enables distributed agents worldwide to register and be easily discovered through Gateway Agents. Collaborations among these agents are coordinated by a Principal Agent equipped with reasoning strategies. DAWN offers three operational modes: No-LLM Mode for deterministic tasks, Copilot for augmented decision-making, and LLM Agent for autonomous operations. Additionally, DAWN ensures the safety and security of agent collaborations globally through a dedicated safety, security, and compliance layer, protecting the network against attackers and adhering to stringent security and compliance standards. These features make DAWN a robust network for deploying agent-based applications across various industries.

摘要: 大型语言模型的快速发展使它们从基本的对话工具转变为能够进行复杂推理和决策的复杂实体。这些进步导致了专门的基于LLM的代理的开发，这些代理专为不同的任务而设计，如编码和Web浏览。随着这些机构变得更有能力，需要一个强有力的框架，促进它们之间的全球沟通和合作，以实现更高的目标，这一需求变得越来越重要。全球网络中的分布式代理(DAW)通过提供一个通用的框架来满足这一需求，该框架将基于LLM的代理与传统软件系统集成在一起，从而能够创建适合于各种用例的代理应用程序。曙光使分布在世界各地的代理能够注册，并通过网关代理容易地被发现。这些代理之间的协作由一个配备了推理策略的委托代理来协调。曙光提供了三种操作模式：用于确定性任务的no-LLM模式，用于增强决策的Copilot模式，以及用于自主操作的LLM代理。此外，曙光公司通过专门的安全、保障和合规层确保全球代理协作的安全和保障，保护网络免受攻击者的攻击，并遵守严格的安全和合规标准。这些功能使曙光成为在不同行业部署基于代理的应用程序的强大网络。



## **14. TrojanRobot: Backdoor Attacks Against Robotic Manipulation in the Physical World**

特洛伊机器人：针对物理世界中机器人操纵的后门攻击 cs.RO

Initial version with preliminary results. We welcome any feedback or  suggestions

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11683v1) [paper-pdf](http://arxiv.org/pdf/2411.11683v1)

**Authors**: Xianlong Wang, Hewen Pan, Hangtao Zhang, Minghui Li, Shengshan Hu, Ziqi Zhou, Lulu Xue, Peijin Guo, Yichen Wang, Wei Wan, Aishan Liu, Leo Yu Zhang

**Abstract**: Robotic manipulation refers to the autonomous handling and interaction of robots with objects using advanced techniques in robotics and artificial intelligence. The advent of powerful tools such as large language models (LLMs) and large vision-language models (LVLMs) has significantly enhanced the capabilities of these robots in environmental perception and decision-making. However, the introduction of these intelligent agents has led to security threats such as jailbreak attacks and adversarial attacks.   In this research, we take a further step by proposing a backdoor attack specifically targeting robotic manipulation and, for the first time, implementing backdoor attack in the physical world. By embedding a backdoor visual language model into the visual perception module within the robotic system, we successfully mislead the robotic arm's operation in the physical world, given the presence of common items as triggers. Experimental evaluations in the physical world demonstrate the effectiveness of the proposed backdoor attack.

摘要: 机器人操纵是指使用机器人学和人工智能的先进技术，自主处理机器人与物体的交互。大型语言模型(LLM)和大型视觉语言模型(LVLM)等强大工具的出现，大大增强了这些机器人在环境感知和决策方面的能力。然而，这些智能代理的引入导致了越狱攻击和对抗性攻击等安全威胁。在这项研究中，我们进一步提出了专门针对机器人操作的后门攻击，并首次在物理世界中实现了后门攻击。通过将后门视觉语言模型嵌入机器人系统的视觉感知模块中，我们成功地误导了机械臂在物理世界中的操作，因为存在共同的物品作为触发器。物理世界中的实验评估证明了所提出的后门攻击的有效性。



## **15. Enhancing Vision-Language Model Safety through Progressive Concept-Bottleneck-Driven Alignment**

通过渐进的概念瓶颈驱动的一致增强视觉语言模型的安全性 cs.CV

arXiv admin note: substantial text overlap with arXiv:2405.13581

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11543v1) [paper-pdf](http://arxiv.org/pdf/2411.11543v1)

**Authors**: Zhendong Liu, Yuanbi Nie, Yingshui Tan, Xiangyu Yue, Qiushi Cui, Chongjun Wang, Xiaoyong Zhu, Bo Zheng

**Abstract**: Benefiting from the powerful capabilities of Large Language Models (LLMs), pre-trained visual encoder models connected to LLMs form Vision Language Models (VLMs). However, recent research shows that the visual modality in VLMs is highly vulnerable, allowing attackers to bypass safety alignment in LLMs through visually transmitted content, launching harmful attacks. To address this challenge, we propose a progressive concept-based alignment strategy, PSA-VLM, which incorporates safety modules as concept bottlenecks to enhance visual modality safety alignment. By aligning model predictions with specific safety concepts, we improve defenses against risky images, enhancing explainability and controllability while minimally impacting general performance. Our method is obtained through two-stage training. The low computational cost of the first stage brings very effective performance improvement, and the fine-tuning of the language model in the second stage further improves the safety performance. Our method achieves state-of-the-art results on popular VLM safety benchmark.

摘要: 得益于大型语言模型的强大功能，连接到大型语言模型的预先训练的视觉编码器模型形成了视觉语言模型。然而，最近的研究表明，VLMS中的视觉通道非常容易受到攻击，使得攻击者能够通过视觉传输的内容绕过LLMS中的安全对齐，从而发起有害攻击。为了应对这一挑战，我们提出了一种基于概念的渐进式对齐策略PSA-VLM，该策略将安全模块作为概念瓶颈纳入其中，以增强视觉通道的安全对齐。通过将模型预测与特定的安全概念相结合，我们改进了对危险图像的防御，增强了可解释性和可控性，同时将对总体性能的影响降至最低。我们的方法是通过两个阶段的训练获得的。第一阶段的低运算量带来了非常有效的性能提升，第二阶段对语言模型的微调进一步提高了安全性能。我们的方法在流行的VLM安全基准上获得了最先进的结果。



## **16. Membership Inference Attack against Long-Context Large Language Models**

针对长上下文大型语言模型的成员推断攻击 cs.CL

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11424v1) [paper-pdf](http://arxiv.org/pdf/2411.11424v1)

**Authors**: Zixiong Wang, Gaoyang Liu, Yang Yang, Chen Wang

**Abstract**: Recent advances in Large Language Models (LLMs) have enabled them to overcome their context window limitations, and demonstrate exceptional retrieval and reasoning capacities on longer context. Quesion-answering systems augmented with Long-Context Language Models (LCLMs) can automatically search massive external data and incorporate it into their contexts, enabling faithful predictions and reducing issues such as hallucinations and knowledge staleness. Existing studies targeting LCLMs mainly concentrate on addressing the so-called lost-in-the-middle problem or improving the inference effiencicy, leaving their privacy risks largely unexplored. In this paper, we aim to bridge this gap and argue that integrating all information into the long context makes it a repository of sensitive information, which often contains private data such as medical records or personal identities. We further investigate the membership privacy within LCLMs external context, with the aim of determining whether a given document or sequence is included in the LCLMs context. Our basic idea is that if a document lies in the context, it will exhibit a low generation loss or a high degree of semantic similarity to the contents generated by LCLMs. We for the first time propose six membership inference attack (MIA) strategies tailored for LCLMs and conduct extensive experiments on various popular models. Empirical results demonstrate that our attacks can accurately infer membership status in most cases, e.g., 90.66% attack F1-score on Multi-document QA datasets with LongChat-7b-v1.5-32k, highlighting significant risks of membership leakage within LCLMs input contexts. Furthermore, we examine the underlying reasons why LCLMs are susceptible to revealing such membership information.

摘要: 大型语言模型(LLM)的最新进展使它们能够克服上下文窗口的限制，并在更长的上下文中显示出出色的检索和推理能力。带有长上下文语言模型(LCLM)的问答系统可以自动搜索大量外部数据并将其合并到上下文中，从而实现准确的预测，并减少幻觉和知识陈旧等问题。现有的针对LCLM的研究主要集中在解决所谓的中间迷失问题或提高推理效率上，而对它们的隐私风险基本上没有进行研究。在本文中，我们旨在弥合这一差距，并认为将所有信息整合到长上下文中使其成为敏感信息的存储库，其中通常包含私人数据，如医疗记录或个人身份。我们进一步研究LCLMS外部上下文中的成员身份隐私，目的是确定给定的文档或序列是否包括在LCLMS上下文中。我们的基本思想是，如果一个文档位于上下文中，它将表现出与LCLM生成的内容的低生成损失或高度语义相似性。我们首次提出了六种专为LCLM定制的成员推理攻击(MIA)策略，并在各种流行的模型上进行了广泛的实验。实验结果表明，我们的攻击可以在大多数情况下准确地推断成员状态，例如，在具有LongChat-7b-v1.5-32k的多文档QA数据集上，90.66%的攻击F1-Score，突出了LCLM输入上下文中成员泄漏的显著风险。此外，我们还考察了LCLM容易泄露此类成员信息的潜在原因。



## **17. The Dark Side of Trust: Authority Citation-Driven Jailbreak Attacks on Large Language Models**

信任的阴暗面：权威引用驱动的对大型语言模型的越狱攻击 cs.LG

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11407v1) [paper-pdf](http://arxiv.org/pdf/2411.11407v1)

**Authors**: Xikang Yang, Xuehai Tang, Jizhong Han, Songlin Hu

**Abstract**: The widespread deployment of large language models (LLMs) across various domains has showcased their immense potential while exposing significant safety vulnerabilities. A major concern is ensuring that LLM-generated content aligns with human values. Existing jailbreak techniques reveal how this alignment can be compromised through specific prompts or adversarial suffixes. In this study, we introduce a new threat: LLMs' bias toward authority. While this inherent bias can improve the quality of outputs generated by LLMs, it also introduces a potential vulnerability, increasing the risk of producing harmful content. Notably, the biases in LLMs is the varying levels of trust given to different types of authoritative information in harmful queries. For example, malware development often favors trust GitHub. To better reveal the risks with LLM, we propose DarkCite, an adaptive authority citation matcher and generator designed for a black-box setting. DarkCite matches optimal citation types to specific risk types and generates authoritative citations relevant to harmful instructions, enabling more effective jailbreak attacks on aligned LLMs.Our experiments show that DarkCite achieves a higher attack success rate (e.g., LLama-2 at 76% versus 68%) than previous methods. To counter this risk, we propose an authenticity and harm verification defense strategy, raising the average defense pass rate (DPR) from 11% to 74%. More importantly, the ability to link citations to the content they encompass has become a foundational function in LLMs, amplifying the influence of LLMs' bias toward authority.

摘要: 大型语言模型(LLM)在不同领域的广泛部署展示了它们的巨大潜力，同时也暴露了重大的安全漏洞。一个主要的问题是确保LLM生成的内容符合人类的价值观。现有的越狱技术揭示了如何通过特定的提示或对抗性后缀来破坏这种对齐。在这项研究中，我们引入了一个新的威胁：LLMS对权威的偏见。虽然这种固有的偏见可以提高低成本管理产生的产出的质量，但它也引入了一个潜在的脆弱性，增加了产生有害内容的风险。值得注意的是，LLMS中的偏差是在有害查询中对不同类型的权威信息给予的不同程度的信任。例如，恶意软件开发通常偏向信任GitHub。为了更好地揭示LLM的风险，我们提出了DarkCite，这是一个为黑箱设置而设计的自适应权威引用匹配器和生成器。DarkCite将最佳引用类型与特定的风险类型相匹配，并生成与有害指令相关的权威引用，从而对对齐的LLMS进行更有效的越狱攻击。我们的实验表明，与以前的方法相比，DarkCite实现了更高的攻击成功率(例如，骆驼-2为76%，而不是68%)。为了应对这种风险，我们提出了真实性和危害性验证防御策略，将平均防御通过率(DPR)从11%提高到74%。更重要的是，将引文与它们所包含的内容相联系的能力已经成为LLMS的一项基本功能，放大了LLMS对权威的偏见的影响。



## **18. Hacking Back the AI-Hacker: Prompt Injection as a Defense Against LLM-driven Cyberattacks**

黑客攻击人工智能黑客：即时注入作为抵御LLM驱动的网络攻击的防御 cs.CR

v0.2 (evaluated on more agents)

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2410.20911v2) [paper-pdf](http://arxiv.org/pdf/2410.20911v2)

**Authors**: Dario Pasquini, Evgenios M. Kornaropoulos, Giuseppe Ateniese

**Abstract**: Large language models (LLMs) are increasingly being harnessed to automate cyberattacks, making sophisticated exploits more accessible and scalable. In response, we propose a new defense strategy tailored to counter LLM-driven cyberattacks. We introduce Mantis, a defensive framework that exploits LLMs' susceptibility to adversarial inputs to undermine malicious operations. Upon detecting an automated cyberattack, Mantis plants carefully crafted inputs into system responses, leading the attacker's LLM to disrupt their own operations (passive defense) or even compromise the attacker's machine (active defense). By deploying purposefully vulnerable decoy services to attract the attacker and using dynamic prompt injections for the attacker's LLM, Mantis can autonomously hack back the attacker. In our experiments, Mantis consistently achieved over 95% effectiveness against automated LLM-driven attacks. To foster further research and collaboration, Mantis is available as an open-source tool: https://github.com/pasquini-dario/project_mantis

摘要: 大型语言模型(LLM)越来越多地被用来自动化网络攻击，使复杂的利用更容易获得和可扩展。作为回应，我们提出了一种新的防御战略，以对抗LLM驱动的网络攻击。我们引入了Mantis，这是一个防御框架，利用LLMS对对手输入的敏感性来破坏恶意操作。在检测到自动网络攻击后，螳螂工厂会精心设计输入到系统响应中，导致攻击者的LLM扰乱自己的操作(被动防御)，甚至危害攻击者的机器(主动防御)。通过部署故意易受攻击的诱骗服务来吸引攻击者，并对攻击者的LLM使用动态提示注入，螳螂可以自主地攻击攻击者。在我们的实验中，螳螂对自动LLM驱动的攻击始终取得了95%以上的效率。为了促进进一步的研究和合作，Mantis以开源工具的形式提供：https://github.com/pasquini-dario/project_mantis



## **19. Adapting to Cyber Threats: A Phishing Evolution Network (PEN) Framework for Phishing Generation and Analyzing Evolution Patterns using Large Language Models**

适应网络威胁：用于使用大型语言模型进行网络钓鱼生成和分析进化模式的网络钓鱼进化网络（PEN）框架 cs.CR

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11389v1) [paper-pdf](http://arxiv.org/pdf/2411.11389v1)

**Authors**: Fengchao Chen, Tingmin Wu, Van Nguyen, Shuo Wang, Hongsheng Hu, Alsharif Abuadbba, Carsten Rudolph

**Abstract**: Phishing remains a pervasive cyber threat, as attackers craft deceptive emails to lure victims into revealing sensitive information. While Artificial Intelligence (AI), particularly deep learning, has become a key component in defending against phishing attacks, these approaches face critical limitations. The scarcity of publicly available, diverse, and updated data, largely due to privacy concerns, constrains their effectiveness. As phishing tactics evolve rapidly, models trained on limited, outdated data struggle to detect new, sophisticated deception strategies, leaving systems vulnerable to an ever-growing array of attacks. Addressing this gap is essential to strengthening defenses in an increasingly hostile cyber landscape. To address this gap, we propose the Phishing Evolution Network (PEN), a framework leveraging large language models (LLMs) and adversarial training mechanisms to continuously generate high quality and realistic diverse phishing samples, and analyze features of LLM-provided phishing to understand evolving phishing patterns. We evaluate the quality and diversity of phishing samples generated by PEN and find that it produces over 80% realistic phishing samples, effectively expanding phishing datasets across seven dominant types. These PEN-generated samples enhance the performance of current phishing detectors, leading to a 40% improvement in detection accuracy. Additionally, the use of PEN significantly boosts model robustness, reducing detectors' sensitivity to perturbations by up to 60%, thereby decreasing attack success rates under adversarial conditions. When we analyze the phishing patterns that are used in LLM-generated phishing, the cognitive complexity and the tone of time limitation are detected with statistically significant differences compared with existing phishing.

摘要: 网络钓鱼仍然是一个普遍存在的网络威胁，因为攻击者精心制作了欺骗性电子邮件，以引诱受害者泄露敏感信息。虽然人工智能(AI)，特别是深度学习，已经成为防御网络钓鱼攻击的关键组件，但这些方法面临着严重的限制。由于缺乏公开可用的、多样化的和更新的数据，这主要是由于隐私问题，限制了它们的有效性。随着钓鱼策略的快速发展，基于有限、过时数据的模型很难检测出新的、复杂的欺骗策略，这使得系统容易受到越来越多的攻击。在日益充满敌意的网络环境中，解决这一差距对于加强防御至关重要。为了弥补这一差距，我们提出了钓鱼进化网络(PEN)，这是一个利用大型语言模型(LLMS)和对手训练机制来持续生成高质量和真实的多样化钓鱼样本的框架，并分析LLM提供的钓鱼特征以了解不断演变的钓鱼模式。我们评估了PEN生成的网络钓鱼样本的质量和多样性，发现它产生了超过80%的真实网络钓鱼样本，有效地扩展了七种主要类型的网络钓鱼数据集。这些笔生成的样本增强了当前网络钓鱼检测器的性能，导致检测准确率提高了40%。此外，PEN的使用显著提高了模型的稳健性，将检测器对扰动的敏感度降低了高达60%，从而降低了对抗性条件下的攻击成功率。当我们分析LLM生成的网络钓鱼中使用的网络钓鱼模式时，我们检测到了认知复杂性和时间限制的基调，与现有的网络钓鱼相比具有统计学意义上的差异。



## **20. CROW: Eliminating Backdoors from Large Language Models via Internal Consistency Regularization**

CROW：通过内部一致性规范化消除大型语言模型的后门 cs.CL

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.12768v1) [paper-pdf](http://arxiv.org/pdf/2411.12768v1)

**Authors**: Nay Myat Min, Long H. Pham, Yige Li, Jun Sun

**Abstract**: Recent studies reveal that Large Language Models (LLMs) are susceptible to backdoor attacks, where adversaries embed hidden triggers that manipulate model responses. Existing backdoor defense methods are primarily designed for vision or classification tasks, and are thus ineffective for text generation tasks, leaving LLMs vulnerable. We introduce Internal Consistency Regularization (CROW), a novel defense using consistency regularization finetuning to address layer-wise inconsistencies caused by backdoor triggers. CROW leverages the intuition that clean models exhibit smooth, consistent transitions in hidden representations across layers, whereas backdoored models show noticeable fluctuation when triggered. By enforcing internal consistency through adversarial perturbations and regularization, CROW neutralizes backdoor effects without requiring clean reference models or prior trigger knowledge, relying only on a small set of clean data. This makes it practical for deployment across various LLM architectures. Experimental results demonstrate that CROW consistently achieves a significant reductions in attack success rates across diverse backdoor strategies and tasks, including negative sentiment, targeted refusal, and code injection, on models such as Llama-2 (7B, 13B), CodeLlama (7B, 13B) and Mistral-7B, while preserving the model's generative capabilities.

摘要: 最近的研究表明，大型语言模型(LLM)容易受到后门攻击，即对手嵌入操纵模型响应的隐藏触发器。现有的后门防御方法主要是为视觉或分类任务设计的，因此对于文本生成任务无效，从而使LLMS容易受到攻击。我们引入了内部一致性正则化(CROW)，这是一种使用一致性正则化精调来解决后门触发器引起的层级不一致的新防御机制。Crow利用了这样一种直觉，即干净的模型在各层的隐藏表示中显示出平滑、一致的过渡，而回溯的模型在触发时会显示出明显的波动。通过对抗性扰动和正则化来强制内部一致性，Crow中和了后门效应，而不需要干净的参考模型或事先的触发知识，只依赖于一小部分干净的数据。这使得它适用于跨各种LLM体系结构进行部署。实验结果表明，在Llama-2(7B，13B)、CodeLlama(7B，13B)和Mistral-7B等模型上，Crow在不同的后门策略和任务(包括负面情绪、定向拒绝和代码注入)上持续显著降低攻击成功率，同时保持了模型的生成能力。



## **21. Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models in Robotics**

探索机器人学中视觉-语言-动作模型的对抗脆弱性 cs.RO

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.13587v1) [paper-pdf](http://arxiv.org/pdf/2411.13587v1)

**Authors**: Taowen Wang, Dongfang Liu, James Chenhao Liang, Wenhao Yang, Qifan Wang, Cheng Han, Jiebo Luo, Ruixiang Tang

**Abstract**: Recently in robotics, Vision-Language-Action (VLA) models have emerged as a transformative approach, enabling robots to execute complex tasks by integrating visual and linguistic inputs within an end-to-end learning framework. While VLA models offer significant capabilities, they also introduce new attack surfaces, making them vulnerable to adversarial attacks. With these vulnerabilities largely unexplored, this paper systematically quantifies the robustness of VLA-based robotic systems. Recognizing the unique demands of robotic execution, our attack objectives target the inherent spatial and functional characteristics of robotic systems. In particular, we introduce an untargeted position-aware attack objective that leverages spatial foundations to destabilize robotic actions, and a targeted attack objective that manipulates the robotic trajectory. Additionally, we design an adversarial patch generation approach that places a small, colorful patch within the camera's view, effectively executing the attack in both digital and physical environments. Our evaluation reveals a marked degradation in task success rates, with up to a 100\% reduction across a suite of simulated robotic tasks, highlighting critical security gaps in current VLA architectures. By unveiling these vulnerabilities and proposing actionable evaluation metrics, this work advances both the understanding and enhancement of safety for VLA-based robotic systems, underscoring the necessity for developing robust defense strategies prior to physical-world deployments.

摘要: 最近在机器人学中，视觉-语言-动作(VLA)模型作为一种变革性的方法出现，使机器人能够通过在端到端学习框架内整合视觉和语言输入来执行复杂的任务。虽然VLA模型提供了重要的功能，但它们也引入了新的攻击面，使其容易受到对手攻击。由于这些漏洞在很大程度上是未知的，本文系统地量化了基于VLA的机器人系统的健壮性。认识到机器人执行的独特需求，我们的攻击目标针对机器人系统固有的空间和功能特征。特别是，我们引入了一个利用空间基础来破坏机器人动作稳定性的无目标位置感知攻击目标，以及一个操纵机器人轨迹的目标攻击目标。此外，我们设计了一种对抗性补丁生成方法，将一个小的、五颜六色的补丁放置在相机的视野中，在数字和物理环境中有效地执行攻击。我们的评估显示任务成功率显著下降，一组模拟机器人任务最多减少100%，突出了当前VLA架构中的关键安全漏洞。通过揭示这些漏洞并提出可操作的评估指标，这项工作促进了对基于VLA的机器人系统安全性的理解和增强，强调了在物理世界部署之前开发强大的防御策略的必要性。



## **22. JailbreakLens: Interpreting Jailbreak Mechanism in the Lens of Representation and Circuit**

越狱镜头：以表象和电路的视角解读越狱机制 cs.CR

18 pages, 10 figures

**SubmitDate**: 2024-11-17    [abs](http://arxiv.org/abs/2411.11114v1) [paper-pdf](http://arxiv.org/pdf/2411.11114v1)

**Authors**: Zeqing He, Zhibo Wang, Zhixuan Chu, Huiyu Xu, Rui Zheng, Kui Ren, Chun Chen

**Abstract**: Despite the outstanding performance of Large language models (LLMs) in diverse tasks, they are vulnerable to jailbreak attacks, wherein adversarial prompts are crafted to bypass their security mechanisms and elicit unexpected responses.Although jailbreak attacks are prevalent, the understanding of their underlying mechanisms remains limited. Recent studies have explain typical jailbreaking behavior (e.g., the degree to which the model refuses to respond) of LLMs by analyzing the representation shifts in their latent space caused by jailbreak prompts or identifying key neurons that contribute to the success of these attacks. However, these studies neither explore diverse jailbreak patterns nor provide a fine-grained explanation from the failure of circuit to the changes of representational, leaving significant gaps in uncovering the jailbreak mechanism. In this paper, we propose JailbreakLens, an interpretation framework that analyzes jailbreak mechanisms from both representation (which reveals how jailbreaks alter the model's harmfulness perception) and circuit perspectives (which uncovers the causes of these deceptions by identifying key circuits contributing to the vulnerability), tracking their evolution throughout the entire response generation process. We then conduct an in-depth evaluation of jailbreak behavior on four mainstream LLMs under seven jailbreak strategies. Our evaluation finds that jailbreak prompts amplify components that reinforce affirmative responses while suppressing those that produce refusal. Although this manipulation shifts model representations toward safe clusters to deceive the LLM, leading it to provide detailed responses instead of refusals, it still produce abnormal activation which can be caught in the circuit analysis.

摘要: 尽管大型语言模型(LLM)在不同的任务中表现出色，但它们很容易受到越狱攻击，在这些攻击中，敌意提示被精心制作以绕过其安全机制并引发意外响应。尽管越狱攻击非常普遍，但对其潜在机制的了解仍然有限。最近的研究已经通过分析越狱提示引起的潜伏空间的表征变化或识别有助于这些攻击成功的关键神经元来解释LLM的典型越狱行为(例如，模型拒绝响应的程度)。然而，这些研究既没有探索多样化的越狱模式，也没有提供从电路故障到表征变化的细粒度解释，在揭示越狱机制方面留下了重大空白。在本文中，我们提出了JailBreakLens，一个解释框架，它从表示(揭示越狱如何改变模型的危害性感知)和电路角度(通过识别导致漏洞的关键电路来揭示这些欺骗的原因)来分析越狱机制，跟踪它们在整个响应生成过程中的演变。然后，我们在七种越狱策略下对四种主流的低成本移动模型的越狱行为进行了深入的评估。我们的评估发现，越狱提示放大了那些强化肯定反应的成分，同时抑制了那些产生拒绝的成分。尽管这种操作将模型表示转移到安全簇以欺骗LLM，导致它提供详细的响应而不是拒绝，但它仍然产生可以在电路分析中发现的异常激活。



## **23. Safely Learning with Private Data: A Federated Learning Framework for Large Language Model**

利用私人数据安全学习：大型语言模型的联邦学习框架 cs.CR

**SubmitDate**: 2024-11-17    [abs](http://arxiv.org/abs/2406.14898v3) [paper-pdf](http://arxiv.org/pdf/2406.14898v3)

**Authors**: JiaYing Zheng, HaiNan Zhang, LingXiang Wang, WangJie Qiu, HongWei Zheng, ZhiMing Zheng

**Abstract**: Private data, being larger and quality-higher than public data, can greatly improve large language models (LLM). However, due to privacy concerns, this data is often dispersed in multiple silos, making its secure utilization for LLM training a challenge. Federated learning (FL) is an ideal solution for training models with distributed private data, but traditional frameworks like FedAvg are unsuitable for LLM due to their high computational demands on clients. An alternative, split learning, offloads most training parameters to the server while training embedding and output layers locally, making it more suitable for LLM. Nonetheless, it faces significant challenges in security and efficiency. Firstly, the gradients of embeddings are prone to attacks, leading to potential reverse engineering of private data. Furthermore, the server's limitation of handle only one client's training request at a time hinders parallel training, severely impacting training efficiency. In this paper, we propose a Federated Learning framework for LLM, named FL-GLM, which prevents data leakage caused by both server-side and peer-client attacks while improving training efficiency. Specifically, we first place the input block and output block on local client to prevent embedding gradient attacks from server. Secondly, we employ key-encryption during client-server communication to prevent reverse engineering attacks from peer-clients. Lastly, we employ optimization methods like client-batching or server-hierarchical, adopting different acceleration methods based on the actual computational capabilities of the server. Experimental results on NLU and generation tasks demonstrate that FL-GLM achieves comparable metrics to centralized chatGLM model, validating the effectiveness of our federated learning framework.

摘要: 私有数据比公共数据更大、质量更高，可以极大地改进大型语言模型(LLM)。然而，出于隐私方面的考虑，这些数据通常分散在多个竖井中，这使得将其安全地用于LLM培训成为一项挑战。联邦学习(FL)是一种适用于具有分布式私有数据的模型训练的理想解决方案，但FedAvg等传统框架由于对客户端的计算要求较高而不适用于LLM。另一种选择是分离学习，将大部分训练参数卸载到服务器，同时在本地训练嵌入和输出层，使其更适合LLM。尽管如此，它在安全和效率方面仍面临重大挑战。首先，嵌入的梯度容易受到攻击，从而导致对私有数据的潜在逆向工程。此外，服务器一次只能处理一个客户端的训练请求的限制阻碍了并行训练，严重影响了训练效率。本文提出了一种用于LLM的联邦学习框架FL-GLM，该框架在提高训练效率的同时，防止了服务器端攻击和对等客户端攻击引起的数据泄漏。具体地说，我们首先将输入块和输出块放置在本地客户端，以防止来自服务器的嵌入梯度攻击。其次，我们在客户-服务器通信过程中使用密钥加密，以防止来自对等客户端的反向工程攻击。最后，我们采用了客户端批处理或服务器分层等优化方法，根据服务器的实际计算能力采用不同的加速方法。在NLU和生成任务上的实验结果表明，FL-GLM达到了与集中式ChatGLM模型相当的指标，验证了联邦学习框架的有效性。



## **24. Risks of Practicing Large Language Models in Smart Grid: Threat Modeling and Validation**

在智能电网中实践大型语言模型的风险：威胁建模和验证 cs.CR

**SubmitDate**: 2024-11-17    [abs](http://arxiv.org/abs/2405.06237v2) [paper-pdf](http://arxiv.org/pdf/2405.06237v2)

**Authors**: Jiangnan Li, Yingyuan Yang, Jinyuan Sun

**Abstract**: Large language models (LLMs) represent significant breakthroughs in artificial intelligence and hold considerable potential for applications within smart grids. However, as demonstrated in previous literature, AI technologies are susceptible to various types of attacks. It is crucial to investigate and evaluate the risks associated with LLMs before deploying them in critical infrastructure like smart grids. In this paper, we systematically evaluated the risks of LLMs and identified two major types of attacks relevant to potential smart grid LLM applications, presenting the corresponding threat models. We also validated these attacks using popular LLMs and real smart grid data. Our validation demonstrates that attackers are capable of injecting bad data and retrieving domain knowledge from LLMs employed in different smart grid applications.

摘要: 大型语言模型（LLM）代表了人工智能的重大突破，在智能电网中的应用中具有相当大的潜力。然而，正如之前的文献所证明的那样，人工智能技术容易受到各种类型的攻击。在将LLM部署在智能电网等关键基础设施中之前，调查和评估与LLM相关的风险至关重要。在本文中，我们系统地评估了LLM的风险，并识别了与潜在智能电网LLM应用相关的两种主要攻击类型，并给出了相应的威胁模型。我们还使用流行的LLM和真实的智能电网数据验证了这些攻击。我们的验证表明，攻击者能够从不同智能电网应用程序中使用的LLM中注入不良数据并检索领域知识。



## **25. Playing Language Game with LLMs Leads to Jailbreaking**

与法学硕士玩语言游戏导致越狱 cs.CL

**SubmitDate**: 2024-11-16    [abs](http://arxiv.org/abs/2411.12762v1) [paper-pdf](http://arxiv.org/pdf/2411.12762v1)

**Authors**: Yu Peng, Zewen Long, Fangming Dong, Congyi Li, Shu Wu, Kai Chen

**Abstract**: The advent of large language models (LLMs) has spurred the development of numerous jailbreak techniques aimed at circumventing their security defenses against malicious attacks. An effective jailbreak approach is to identify a domain where safety generalization fails, a phenomenon known as mismatched generalization. In this paper, we introduce two novel jailbreak methods based on mismatched generalization: natural language games and custom language games, both of which effectively bypass the safety mechanisms of LLMs, with various kinds and different variants, making them hard to defend and leading to high attack rates. Natural language games involve the use of synthetic linguistic constructs and the actions intertwined with these constructs, such as the Ubbi Dubbi language. Building on this phenomenon, we propose the custom language games method: by engaging with LLMs using a variety of custom rules, we successfully execute jailbreak attacks across multiple LLM platforms. Extensive experiments demonstrate the effectiveness of our methods, achieving success rates of 93% on GPT-4o, 89% on GPT-4o-mini and 83% on Claude-3.5-Sonnet. Furthermore, to investigate the generalizability of safety alignments, we fine-tuned Llama-3.1-70B with the custom language games to achieve safety alignment within our datasets and found that when interacting through other language games, the fine-tuned models still failed to identify harmful content. This finding indicates that the safety alignment knowledge embedded in LLMs fails to generalize across different linguistic formats, thus opening new avenues for future research in this area.

摘要: 大型语言模型(LLM)的出现促进了许多越狱技术的发展，这些技术旨在绕过针对恶意攻击的安全防御。一种有效的越狱方法是识别安全泛化失败的域，这种现象称为不匹配泛化。本文介绍了两种新的基于不匹配泛化的越狱方法：自然语言游戏和自定义语言游戏，这两种方法都有效地绕过了LLMS的安全机制，种类繁多，变体不同，使得它们难以防御，导致攻击率很高。自然语言游戏涉及使用合成的语言结构以及与这些结构交织在一起的动作，如Ubbi Dubbi语言。基于这一现象，我们提出了定制语言游戏方法：通过使用各种定制规则与LLM接触，我们成功地跨多个LLM平台执行越狱攻击。大量实验证明了该方法的有效性，在GPT-4o、GPT-4o-mini和Claude-3.5-十四行诗上分别获得了93%、89%和83%的识别成功率。此外，为了调查安全对齐的泛化能力，我们使用自定义语言游戏对Llama-3.1-70B进行了微调，以在我们的数据集中实现安全对齐，发现当通过其他语言游戏交互时，微调的模型仍然无法识别有害内容。这一发现表明，LLMS中嵌入的安全对齐知识无法跨不同的语言格式进行泛化，从而为这一领域的未来研究开辟了新的途径。



## **26. SQL Injection Jailbreak: a structural disaster of large language models**

SQL注入越狱：大型语言模型的结构性灾难 cs.CR

**SubmitDate**: 2024-11-16    [abs](http://arxiv.org/abs/2411.01565v2) [paper-pdf](http://arxiv.org/pdf/2411.01565v2)

**Authors**: Jiawei Zhao, Kejiang Chen, Weiming Zhang, Nenghai Yu

**Abstract**: In recent years, the rapid development of large language models (LLMs) has brought new vitality to the various domains and generated substantial social and economic benefits. However, the swift advancement of LLMs has introduced new security vulnerabilities. Jailbreak, a form of attack that induces LLMs to output harmful content through carefully crafted prompts, poses a challenge to the safe and trustworthy development of LLMs. Previous jailbreak attack methods primarily exploited the internal capabilities of the model. Among them, one category leverages the model's implicit capabilities for jailbreak attacks, where the attacker is unaware of the exact reasons for the attack's success. The other category utilizes the model's explicit capabilities for jailbreak attacks, where the attacker understands the reasons for the attack's success. For example, these attacks exploit the model's abilities in coding, contextual learning, or understanding ASCII characters. However, these earlier jailbreak attacks have certain limitations, as they only exploit the inherent capabilities of the model. In this paper, we propose a novel jailbreak method, SQL Injection Jailbreak (SIJ), which utilizes the construction of input prompts by LLMs to inject jailbreak information into user prompts, enabling successful jailbreak of the LLMs. Our SIJ method achieves nearly 100\% attack success rates on five well-known open-source LLMs in the context of AdvBench, while incurring lower time costs compared to previous methods. More importantly, SIJ reveals a new vulnerability in LLMs that urgently needs to be addressed. To this end, we propose a defense method called Self-Reminder-Key and demonstrate its effectiveness through experiments. Our code is available at \href{https://github.com/weiyezhimeng/SQL-Injection-Jailbreak}{https://github.com/weiyezhimeng/SQL-Injection-Jailbreak}.

摘要: 近年来，大语言模型的快速发展给各个领域带来了新的活力，产生了可观的社会效益和经济效益。然而，LLMS的快速发展带来了新的安全漏洞。越狱是一种攻击形式，通过精心制作的提示诱使LLMS输出有害内容，对LLMS的安全和可信开发构成了挑战。以前的越狱攻击方法主要是利用该模型的内部能力。其中，一类利用该模型的隐含能力进行越狱攻击，即攻击者不知道攻击成功的确切原因。另一类利用该模型的明确能力进行越狱攻击，攻击者了解攻击成功的原因。例如，这些攻击利用了模型在编码、上下文学习或理解ASCII字符方面的能力。然而，这些早期的越狱攻击有一定的局限性，因为它们只利用了该模型的固有功能。本文提出了一种新的越狱方法--SQL注入越狱(SIJ)，该方法利用LLMS构造输入提示，在用户提示中注入越狱信息，使LLMS能够成功越狱。与以前的方法相比，我们的SIJ方法在五个著名的开源LLM上获得了近100\%的攻击成功率，同时产生了更低的时间开销。更重要的是，SIJ揭示了LLMS中一个迫切需要解决的新漏洞。为此，我们提出了一种称为自我提醒密钥的防御方法，并通过实验验证了该方法的有效性。我们的代码可以在\href{https://github.com/weiyezhimeng/SQL-Injection-Jailbreak}{https://github.com/weiyezhimeng/SQL-Injection-Jailbreak}.上找到



## **27. Insights and Current Gaps in Open-Source LLM Vulnerability Scanners: A Comparative Analysis**

开源LLM漏洞扫描仪的见解和当前差距：比较分析 cs.CR

15 pages, 11 figures

**SubmitDate**: 2024-11-16    [abs](http://arxiv.org/abs/2410.16527v3) [paper-pdf](http://arxiv.org/pdf/2410.16527v3)

**Authors**: Jonathan Brokman, Omer Hofman, Oren Rachmil, Inderjeet Singh, Vikas Pahuja, Rathina Sabapathy Aishvariya Priya, Amit Giloni, Roman Vainshtein, Hisashi Kojima

**Abstract**: This report presents a comparative analysis of open-source vulnerability scanners for conversational large language models (LLMs). As LLMs become integral to various applications, they also present potential attack surfaces, exposed to security risks such as information leakage and jailbreak attacks. Our study evaluates prominent scanners - Garak, Giskard, PyRIT, and CyberSecEval - that adapt red-teaming practices to expose these vulnerabilities. We detail the distinctive features and practical use of these scanners, outline unifying principles of their design and perform quantitative evaluations to compare them. These evaluations uncover significant reliability issues in detecting successful attacks, highlighting a fundamental gap for future development. Additionally, we contribute a preliminary labelled dataset, which serves as an initial step to bridge this gap. Based on the above, we provide strategic recommendations to assist organizations choose the most suitable scanner for their red-teaming needs, accounting for customizability, test suite comprehensiveness, and industry-specific use cases.

摘要: 本报告对用于会话大型语言模型(LLM)的开源漏洞扫描器进行了比较分析。随着LLM成为各种应用的组成部分，它们也出现了潜在的攻击面，暴露在信息泄露和越狱攻击等安全风险中。我们的研究评估了采用红色团队实践来暴露这些漏洞的著名扫描仪-Garak、Giskard、PyRIT和CyberSecEval。我们详细介绍了这些扫描仪的特点和实际应用，概述了它们设计的统一原则，并进行了定量评估以进行比较。这些评估揭示了检测成功攻击的重大可靠性问题，突显了未来发展的根本差距。此外，我们提供了一个初步的标记数据集，这是弥合这一差距的第一步。在此基础上，我们提供了战略性建议，以帮助组织选择最适合其红团队需求的扫描仪，考虑到可定制性、测试套件的全面性和行业特定的用例。



## **28. Sim-CLIP: Unsupervised Siamese Adversarial Fine-Tuning for Robust and Semantically-Rich Vision-Language Models**

Sim-CLIP：针对稳健且语义丰富的视觉语言模型的无监督Siamese对抗微调 cs.CV

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2407.14971v2) [paper-pdf](http://arxiv.org/pdf/2407.14971v2)

**Authors**: Md Zarif Hossain, Ahmed Imteaj

**Abstract**: Vision-language models (VLMs) have achieved significant strides in recent times specially in multimodal tasks, yet they remain susceptible to adversarial attacks on their vision components. To address this, we propose Sim-CLIP, an unsupervised adversarial fine-tuning method that enhances the robustness of the widely-used CLIP vision encoder against such attacks while maintaining semantic richness and specificity. By employing a Siamese architecture with cosine similarity loss, Sim-CLIP learns semantically meaningful and attack-resilient visual representations without requiring large batch sizes or momentum encoders. Our results demonstrate that VLMs enhanced with Sim-CLIP's fine-tuned CLIP encoder exhibit significantly enhanced robustness against adversarial attacks, while preserving semantic meaning of the perturbed images. Notably, Sim-CLIP does not require additional training or fine-tuning of the VLM itself; replacing the original vision encoder with our fine-tuned Sim-CLIP suffices to provide robustness. This work underscores the significance of reinforcing foundational models like CLIP to safeguard the reliability of downstream VLM applications, paving the way for more secure and effective multimodal systems.

摘要: 视觉语言模型近年来取得了长足的进步，特别是在多通道任务中，但它们仍然容易受到视觉部分的敌意攻击。为了解决这一问题，我们提出了SIM-CLIP，这是一种无监督的对抗性微调方法，它在保持语义丰富和特异性的同时，增强了广泛使用的CLIP视觉编码器对此类攻击的健壮性。通过采用具有余弦相似性损失的暹罗体系结构，Sim-Clip无需大批量或动量编码器即可学习语义上有意义的、可抵抗攻击的视觉表示。结果表明，通过Sim-Clip的精细调整的CLIP编码器增强的VLM在保持扰动图像语义的同时，显著增强了对对手攻击的稳健性。值得注意的是，SIM-Clip不需要对VLM本身进行额外的培训或微调；用我们经过微调的SIM-Clip替换原来的视觉编码器就足以提供健壮性。这项工作强调了加强像CLIP这样的基础模型对保障下游VLM应用的可靠性的重要性，为更安全和有效的多式联运系统铺平了道路。



## **29. Comparing Robustness Against Adversarial Attacks in Code Generation: LLM-Generated vs. Human-Written**

比较代码生成中对抗对抗攻击的鲁棒性：LLM生成与人类编写 cs.SE

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.10565v1) [paper-pdf](http://arxiv.org/pdf/2411.10565v1)

**Authors**: Md Abdul Awal, Mrigank Rochan, Chanchal K. Roy

**Abstract**: Thanks to the widespread adoption of Large Language Models (LLMs) in software engineering research, the long-standing dream of automated code generation has become a reality on a large scale. Nowadays, LLMs such as GitHub Copilot and ChatGPT are extensively used in code generation for enterprise and open-source software development and maintenance. Despite their unprecedented successes in code generation, research indicates that codes generated by LLMs exhibit vulnerabilities and security issues. Several studies have been conducted to evaluate code generated by LLMs, considering various aspects such as security, vulnerability, code smells, and robustness. While some studies have compared the performance of LLMs with that of humans in various software engineering tasks, there's a notable gap in research: no studies have directly compared human-written and LLM-generated code for their robustness analysis. To fill this void, this paper introduces an empirical study to evaluate the adversarial robustness of Pre-trained Models of Code (PTMCs) fine-tuned on code written by humans and generated by LLMs against adversarial attacks for software clone detection. These attacks could potentially undermine software security and reliability. We consider two datasets, two state-of-the-art PTMCs, two robustness evaluation criteria, and three metrics to use in our experiments. Regarding effectiveness criteria, PTMCs fine-tuned on human-written code always demonstrate more robustness than those fine-tuned on LLMs-generated code. On the other hand, in terms of adversarial code quality, in 75% experimental combinations, PTMCs fine-tuned on the human-written code exhibit more robustness than the PTMCs fine-tuned on the LLMs-generated code.

摘要: 由于大型语言模型(LLM)在软件工程研究中的广泛采用，自动化代码生成的长期梦想已经在很大程度上成为现实。如今，GitHub Copilot和ChatGPT等LLMS被广泛用于企业和开源软件开发和维护的代码生成。尽管LLM在代码生成方面取得了前所未有的成功，但研究表明，LLM生成的代码存在漏洞和安全问题。考虑到安全性、脆弱性、代码气味和健壮性等各个方面，已经进行了几项研究来评估LLMS生成的代码。虽然一些研究已经将LLM与人类在各种软件工程任务中的性能进行了比较，但研究中存在一个明显的差距：没有研究直接比较人类编写的代码和LLM生成的代码来进行健壮性分析。为了填补这一空白，本文介绍了一项经验研究，以评估预先训练的代码模型(PTMC)对人类编写的代码进行微调并由LLMS生成的代码对软件克隆检测的恶意攻击的健壮性。这些攻击可能会潜在地破坏软件的安全性和可靠性。我们考虑了两个数据集、两个最先进的PTMC、两个健壮性评估标准和三个度量来用于我们的实验。关于有效性标准，在人类编写的代码上微调的PTMC总是比那些在LLMS生成的代码上微调的PTMC表现出更强的健壮性。另一方面，在对抗性代码质量方面，在75%的实验组合中，基于人类编写的代码微调的PTMC表现出比基于LLMS生成的代码微调的PTMC更强的稳健性。



## **30. On the Privacy Risk of In-context Learning**

论上下文学习的隐私风险 cs.LG

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.10512v1) [paper-pdf](http://arxiv.org/pdf/2411.10512v1)

**Authors**: Haonan Duan, Adam Dziedzic, Mohammad Yaghini, Nicolas Papernot, Franziska Boenisch

**Abstract**: Large language models (LLMs) are excellent few-shot learners. They can perform a wide variety of tasks purely based on natural language prompts provided to them. These prompts contain data of a specific downstream task -- often the private dataset of a party, e.g., a company that wants to leverage the LLM for their purposes. We show that deploying prompted models presents a significant privacy risk for the data used within the prompt by instantiating a highly effective membership inference attack. We also observe that the privacy risk of prompted models exceeds fine-tuned models at the same utility levels. After identifying the model's sensitivity to their prompts -- in the form of a significantly higher prediction confidence on the prompted data -- as a cause for the increased risk, we propose ensembling as a mitigation strategy. By aggregating over multiple different versions of a prompted model, membership inference risk can be decreased.

摘要: 大型语言模型（LLM）是优秀的少量学习者。他们可以纯粹根据向他们提供的自然语言提示执行各种任务。这些提示包含特定下游任务的数据--通常是一方的私人数据集，例如，一家希望利用LLM来实现其目的的公司。我们表明，通过实例化高效的成员资格推断攻击，部署提示模型会给提示内使用的数据带来显着的隐私风险。我们还观察到，提示模型的隐私风险超过了相同实用水平下的微调模型。在确定模型对其提示的敏感性（表现为对提示数据的预测置信度显着更高）是风险增加的原因后，我们建议将集成作为一种缓解策略。通过聚合提示模型的多个不同版本，可以降低成员资格推断风险。



## **31. Optimization-based Prompt Injection Attack to LLM-as-a-Judge**

对LLM as-a-Judge的基于优化的即时注入攻击 cs.CR

To appear in the Proceedings of The ACM Conference on Computer and  Communications Security (CCS), 2024

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2403.17710v3) [paper-pdf](http://arxiv.org/pdf/2403.17710v3)

**Authors**: Jiawen Shi, Zenghui Yuan, Yinuo Liu, Yue Huang, Pan Zhou, Lichao Sun, Neil Zhenqiang Gong

**Abstract**: LLM-as-a-Judge uses a large language model (LLM) to select the best response from a set of candidates for a given question. LLM-as-a-Judge has many applications such as LLM-powered search, reinforcement learning with AI feedback (RLAIF), and tool selection. In this work, we propose JudgeDeceiver, an optimization-based prompt injection attack to LLM-as-a-Judge. JudgeDeceiver injects a carefully crafted sequence into an attacker-controlled candidate response such that LLM-as-a-Judge selects the candidate response for an attacker-chosen question no matter what other candidate responses are. Specifically, we formulate finding such sequence as an optimization problem and propose a gradient based method to approximately solve it. Our extensive evaluation shows that JudgeDeceive is highly effective, and is much more effective than existing prompt injection attacks that manually craft the injected sequences and jailbreak attacks when extended to our problem. We also show the effectiveness of JudgeDeceiver in three case studies, i.e., LLM-powered search, RLAIF, and tool selection. Moreover, we consider defenses including known-answer detection, perplexity detection, and perplexity windowed detection. Our results show these defenses are insufficient, highlighting the urgent need for developing new defense strategies. Our implementation is available at this repository: https://github.com/ShiJiawenwen/JudgeDeceiver.

摘要: LLM-as-a-Court使用大型语言模型(LLM)从给定问题的一组候选人中选择最佳答案。LLM-as-a-Court有许多应用，如LLM支持的搜索、带人工智能反馈的强化学习(RLAIF)和工具选择。在这项工作中，我们提出了一种针对LLM-as-a-Court的基于优化的快速注入攻击--JudgeDeceiver。JudgeDeceiver将精心制作的序列注入到攻击者控制的候选响应中，以便LLM-as-a-Court为攻击者选择的问题选择候选响应，而不管其他候选响应是什么。具体地说，我们将寻找这样的序列描述为一个优化问题，并提出了一种基于梯度的方法来近似求解它。我们的广泛评估表明，JudgeDecept是非常有效的，并且比现有的手动手工创建注入序列的即时注入攻击和越狱攻击更有效，当扩展到我们的问题时。我们还在三个案例研究中展示了JudgeDeceiver的有效性，即LLM支持的搜索、RLAIF和工具选择。此外，我们还考虑了防御措施，包括已知答案检测、困惑检测和困惑加窗检测。我们的结果表明，这些防御措施是不够的，这突显了开发新的防御战略的迫切需要。我们的实现可从以下存储库获得：https://github.com/ShiJiawenwen/JudgeDeceiver.



## **32. IDEATOR: Jailbreaking Large Vision-Language Models Using Themselves**

IDEATOR：利用自己越狱大型视觉语言模型 cs.CV

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.00827v2) [paper-pdf](http://arxiv.org/pdf/2411.00827v2)

**Authors**: Ruofan Wang, Bo Wang, Xiaosen Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: As large Vision-Language Models (VLMs) grow in prominence, ensuring their safe deployment has become critical. Recent studies have explored VLM robustness against jailbreak attacks--techniques that exploit model vulnerabilities to elicit harmful outputs. However, the limited availability of diverse multi-modal data has led current approaches to rely heavily on adversarial or manually crafted images derived from harmful text datasets, which may lack effectiveness and diversity across different contexts. In this paper, we propose a novel jailbreak method named IDEATOR, which autonomously generates malicious image-text pairs for black-box jailbreak attacks. IDEATOR is based on the insight that VLMs themselves could serve as powerful red team models for generating multimodal jailbreak prompts. Specifically, IDEATOR uses a VLM to create targeted jailbreak texts and pairs them with jailbreak images generated by a state-of-the-art diffusion model. Our extensive experiments demonstrate IDEATOR's high effectiveness and transferability. Notably, it achieves a 94% success rate in jailbreaking MiniGPT-4 with an average of only 5.34 queries, and high success rates of 82%, 88%, and 75% when transferred to LLaVA, InstructBLIP, and Meta's Chameleon, respectively. IDEATOR uncovers specific vulnerabilities in VLMs under black-box conditions, underscoring the need for improved safety mechanisms.

摘要: 随着大型视觉语言模型(VLM)的日益突出，确保它们的安全部署变得至关重要。最近的研究探索了VLM对越狱攻击的健壮性--利用模型漏洞来获得有害输出的技术。然而，多样化的多模式数据的可获得性有限，导致目前的方法严重依赖于从有害文本数据集获得的对抗性或手动制作的图像，这可能在不同的背景下缺乏有效性和多样性。在本文中，我们提出了一种新的越狱方法IDEATOR，该方法自动生成用于黑盒越狱攻击的恶意图文对。Ideator基于这样一种见解，即VLM本身可以作为强大的红色团队模型来生成多模式越狱提示。具体地说，Ideator使用VLM创建有针对性的越狱文本，并将它们与由最先进的扩散模型生成的越狱图像配对。我们的大量实验证明了IDEADER的高效性和可移植性。值得注意的是，在平均只有5.34个查询的情况下，它在越狱MiniGPT-4上的成功率达到了94%，当转移到LLaVA、InstructBLIP和Meta‘s Chameleon时，成功率分别达到了82%、88%和75%。Ideator发现了黑箱条件下VLM中的特定漏洞，强调了改进安全机制的必要性。



## **33. Security and Privacy Challenges of Large Language Models: A Survey**

大型语言模型的安全和隐私挑战：调查 cs.CL

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2402.00888v2) [paper-pdf](http://arxiv.org/pdf/2402.00888v2)

**Authors**: Badhan Chandra Das, M. Hadi Amini, Yanzhao Wu

**Abstract**: Large Language Models (LLMs) have demonstrated extraordinary capabilities and contributed to multiple fields, such as generating and summarizing text, language translation, and question-answering. Nowadays, LLM is becoming a very popular tool in computerized language processing tasks, with the capability to analyze complicated linguistic patterns and provide relevant and appropriate responses depending on the context. While offering significant advantages, these models are also vulnerable to security and privacy attacks, such as jailbreaking attacks, data poisoning attacks, and Personally Identifiable Information (PII) leakage attacks. This survey provides a thorough review of the security and privacy challenges of LLMs for both training data and users, along with the application-based risks in various domains, such as transportation, education, and healthcare. We assess the extent of LLM vulnerabilities, investigate emerging security and privacy attacks for LLMs, and review the potential defense mechanisms. Additionally, the survey outlines existing research gaps in this domain and highlights future research directions.

摘要: 大型语言模型(LLM)显示了非凡的能力，并在多个领域做出了贡献，如生成和汇总文本、语言翻译和问题回答。如今，LLM正在成为计算机语言处理任务中非常流行的工具，它能够分析复杂的语言模式，并根据语境提供相关和适当的回应。在提供显著优势的同时，这些模型也容易受到安全和隐私攻击，例如越狱攻击、数据中毒攻击和个人身份信息(PII)泄漏攻击。该调查全面回顾了LLMS对培训数据和用户的安全和隐私挑战，以及交通、教育和医疗保健等各个领域的基于应用的风险。我们评估LLM漏洞的程度，调查LLM新出现的安全和隐私攻击，并审查潜在的防御机制。此外，调查还概述了该领域存在的研究差距，并强调了未来的研究方向。



## **34. AutoDefense: Multi-Agent LLM Defense against Jailbreak Attacks**

AutoDefense：针对越狱攻击的多代理LLM防御 cs.LG

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2403.04783v2) [paper-pdf](http://arxiv.org/pdf/2403.04783v2)

**Authors**: Yifan Zeng, Yiran Wu, Xiao Zhang, Huazheng Wang, Qingyun Wu

**Abstract**: Despite extensive pre-training in moral alignment to prevent generating harmful information, large language models (LLMs) remain vulnerable to jailbreak attacks. In this paper, we propose AutoDefense, a multi-agent defense framework that filters harmful responses from LLMs. With the response-filtering mechanism, our framework is robust against different jailbreak attack prompts, and can be used to defend different victim models. AutoDefense assigns different roles to LLM agents and employs them to complete the defense task collaboratively. The division in tasks enhances the overall instruction-following of LLMs and enables the integration of other defense components as tools. With AutoDefense, small open-source LMs can serve as agents and defend larger models against jailbreak attacks. Our experiments show that AutoDefense can effectively defense against different jailbreak attacks, while maintaining the performance at normal user request. For example, we reduce the attack success rate on GPT-3.5 from 55.74% to 7.95% using LLaMA-2-13b with a 3-agent system. Our code and data are publicly available at https://github.com/XHMY/AutoDefense.

摘要: 尽管在道德一致性方面进行了广泛的预先培训，以防止产生有害信息，但大型语言模型(LLM)仍然容易受到越狱攻击。在本文中，我们提出了一种过滤来自LLMS的有害响应的多代理防御框架--AutoDefense。通过响应过滤机制，我们的框架对不同的越狱攻击提示具有健壮性，并且可以用于防御不同的受害者模型。AutoDefense为LLM特工分配不同的角色，并雇用他们协作完成防御任务。任务分工加强了LLMS的整体指令遵循，并使其他防御组件能够作为工具进行集成。有了AutoDefense，小型开源LMS可以作为代理，保护较大的模型免受越狱攻击。我们的实验表明，AutoDefense能够有效地防御不同的越狱攻击，同时保持正常用户请求的性能。例如，我们使用带有3代理系统的Llama-2-13b将对GPT-3.5的攻击成功率从55.74%降低到7.95%。我们的代码和数据在https://github.com/XHMY/AutoDefense.上公开提供



## **35. DROJ: A Prompt-Driven Attack against Large Language Models**

DROJ：针对大型语言模型的预算驱动攻击 cs.CL

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2411.09125v1) [paper-pdf](http://arxiv.org/pdf/2411.09125v1)

**Authors**: Leyang Hu, Boran Wang

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional capabilities across various natural language processing tasks. Due to their training on internet-sourced datasets, LLMs can sometimes generate objectionable content, necessitating extensive alignment with human feedback to avoid such outputs. Despite massive alignment efforts, LLMs remain susceptible to adversarial jailbreak attacks, which usually are manipulated prompts designed to circumvent safety mechanisms and elicit harmful responses. Here, we introduce a novel approach, Directed Rrepresentation Optimization Jailbreak (DROJ), which optimizes jailbreak prompts at the embedding level to shift the hidden representations of harmful queries towards directions that are more likely to elicit affirmative responses from the model. Our evaluations on LLaMA-2-7b-chat model show that DROJ achieves a 100\% keyword-based Attack Success Rate (ASR), effectively preventing direct refusals. However, the model occasionally produces repetitive and non-informative responses. To mitigate this, we introduce a helpfulness system prompt that enhances the utility of the model's responses. Our code is available at https://github.com/Leon-Leyang/LLM-Safeguard.

摘要: 大型语言模型(LLM)在各种自然语言处理任务中表现出了非凡的能力。由于对来自互联网的数据集进行了培训，LLM有时会产生令人反感的内容，需要与人类反馈广泛协调，以避免此类输出。尽管做出了巨大的调整努力，但LLM仍然容易受到对抗性越狱攻击，这些攻击通常是被操纵的提示，旨在绕过安全机制并引发有害反应。在这里，我们介绍了一种新的方法，定向R表示优化越狱(DROJ)，它在嵌入级别优化越狱提示，将有害查询的隐藏表示向更有可能引起模型肯定响应的方向移动。对Llama-2-7b-Chat模型的评估表明，DROJ达到了100%的基于关键字的攻击成功率，有效地防止了直接拒绝。然而，该模型偶尔会产生重复的、非信息性的回答。为了缓解这一问题，我们引入了一个帮助系统提示，以增强模型响应的实用性。我们的代码可以在https://github.com/Leon-Leyang/LLM-Safeguard.上找到



## **36. Trustful LLMs: Customizing and Grounding Text Generation with Knowledge Bases and Dual Decoders**

值得信赖的LLM：使用知识库和双解码器定制和基础文本生成 cs.CL

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2411.07870v2) [paper-pdf](http://arxiv.org/pdf/2411.07870v2)

**Authors**: Xiaofeng Zhu, Jaya Krishna Mandivarapu

**Abstract**: Although people are impressed by the content generation skills of large language models, the use of LLMs, such as ChatGPT, is limited by the domain grounding of the content. The correctness and groundedness of the generated content need to be based on a verified context, such as results from Retrieval-Augmented Generation (RAG). One important issue when adapting LLMs to a customized domain is that the generated responses are often incomplete, or the additions are not verified and may even be hallucinated. Prior studies on hallucination detection have focused on evaluation metrics, which are not easily adaptable to dynamic domains and can be vulnerable to attacks like jail-breaking. In this work, we propose 1) a post-processing algorithm that leverages knowledge triplets in RAG context to correct hallucinations and 2) a dual-decoder model that fuses RAG context to guide the generation process.

摘要: 尽管人们对大型语言模型的内容生成技能印象深刻，但ChatGPT等LLM的使用受到内容领域基础的限制。生成内容的正确性和可信度需要基于经过验证的上下文，例如检索增强生成（RAG）的结果。将LLM调整到定制域时的一个重要问题是，生成的响应通常不完整，或者添加未经验证，甚至可能出现幻觉。之前关于幻觉检测的研究集中在评估指标上，这些指标不容易适应动态领域，并且容易受到越狱等攻击。在这项工作中，我们提出了1）一种利用RAG上下文中的知识三重组来纠正幻觉的后处理算法，以及2）一种融合RAG上下文来指导生成过程的双解码器模型。



## **37. Defending Large Language Models Against Attacks With Residual Stream Activation Analysis**

利用剩余流激活分析防御大型语言模型免受攻击 cs.CR

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2406.03230v4) [paper-pdf](http://arxiv.org/pdf/2406.03230v4)

**Authors**: Amelia Kawasaki, Andrew Davis, Houssam Abbas

**Abstract**: The widespread adoption of Large Language Models (LLMs), exemplified by OpenAI's ChatGPT, brings to the forefront the imperative to defend against adversarial threats on these models. These attacks, which manipulate an LLM's output by introducing malicious inputs, undermine the model's integrity and the trust users place in its outputs. In response to this challenge, our paper presents an innovative defensive strategy, given white box access to an LLM, that harnesses residual activation analysis between transformer layers of the LLM. We apply a novel methodology for analyzing distinctive activation patterns in the residual streams for attack prompt classification. We curate multiple datasets to demonstrate how this method of classification has high accuracy across multiple types of attack scenarios, including our newly-created attack dataset. Furthermore, we enhance the model's resilience by integrating safety fine-tuning techniques for LLMs in order to measure its effect on our capability to detect attacks. The results underscore the effectiveness of our approach in enhancing the detection and mitigation of adversarial inputs, advancing the security framework within which LLMs operate.

摘要: 大型语言模型(LLM)的广泛采用，如OpenAI的ChatGPT，使防御这些模型上的对手威胁成为当务之急。这些攻击通过引入恶意输入来操纵LLM的输出，破坏了模型的完整性和用户对其输出的信任。为了应对这一挑战，我们的论文提出了一种创新的防御策略，在白盒访问LLM的情况下，该策略利用LLM变压器层之间的剩余激活分析。我们应用了一种新的方法来分析残留流中独特的激活模式，以进行攻击提示分类。我们精选了多个数据集，以演示此分类方法如何在多种类型的攻击场景中具有高精度，包括我们新创建的攻击数据集。此外，我们通过集成LLMS的安全微调技术来增强模型的弹性，以衡量其对我们检测攻击的能力的影响。这些结果强调了我们的方法在加强对敌对输入的检测和缓解、推进LLMS运作的安全框架方面的有效性。



## **38. LLMStinger: Jailbreaking LLMs using RL fine-tuned LLMs**

LLMStinger：使用RL微调的LLM越狱LLM cs.LG

Accepted at AAAI 2025

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2411.08862v1) [paper-pdf](http://arxiv.org/pdf/2411.08862v1)

**Authors**: Piyush Jha, Arnav Arora, Vijay Ganesh

**Abstract**: We introduce LLMStinger, a novel approach that leverages Large Language Models (LLMs) to automatically generate adversarial suffixes for jailbreak attacks. Unlike traditional methods, which require complex prompt engineering or white-box access, LLMStinger uses a reinforcement learning (RL) loop to fine-tune an attacker LLM, generating new suffixes based on existing attacks for harmful questions from the HarmBench benchmark. Our method significantly outperforms existing red-teaming approaches (we compared against 15 of the latest methods), achieving a +57.2% improvement in Attack Success Rate (ASR) on LLaMA2-7B-chat and a +50.3% ASR increase on Claude 2, both models known for their extensive safety measures. Additionally, we achieved a 94.97% ASR on GPT-3.5 and 99.4% on Gemma-2B-it, demonstrating the robustness and adaptability of LLMStinger across open and closed-source models.

摘要: 我们引入了LLMStinger，这是一种利用大型语言模型（LLM）自动生成越狱攻击的对抗性后缀的新颖方法。与需要复杂的即时工程或白盒访问的传统方法不同，LLMStinger使用强化学习（RL）循环来微调攻击者LLM，根据HarmBench基准中针对有害问题的现有攻击生成新的后缀。我们的方法显着优于现有的红色团队方法（我们与15种最新方法进行了比较），在LLaMA 2 - 7 B-chat上实现了攻击成功率（ASB）+57.2%的提高，在Claude 2上实现了攻击成功率（ASB）+50.3%的提高，这两种型号都以其广泛的安全措施而闻名。此外，我们在GPT-3.5上实现了94.97%的ASB，在Gemma-2B-it上实现了99.4%的ASB，证明了LLMStinger在开放和封闭源模型中的稳健性和适应性。



## **39. Target-driven Attack for Large Language Models**

针对大型语言模型的目标驱动攻击 cs.CL

12 pages, 7 figures. This work is an extension of the  arXiv:2404.07234 work. We propose new methods. 27th European Conference on  Artificial Intelligence 2024

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2411.07268v2) [paper-pdf](http://arxiv.org/pdf/2411.07268v2)

**Authors**: Chong Zhang, Mingyu Jin, Dong Shu, Taowen Wang, Dongfang Liu, Xiaobo Jin

**Abstract**: Current large language models (LLM) provide a strong foundation for large-scale user-oriented natural language tasks. Many users can easily inject adversarial text or instructions through the user interface, thus causing LLM model security challenges like the language model not giving the correct answer. Although there is currently a large amount of research on black-box attacks, most of these black-box attacks use random and heuristic strategies. It is unclear how these strategies relate to the success rate of attacks and thus effectively improve model robustness. To solve this problem, we propose our target-driven black-box attack method to maximize the KL divergence between the conditional probabilities of the clean text and the attack text to redefine the attack's goal. We transform the distance maximization problem into two convex optimization problems based on the attack goal to solve the attack text and estimate the covariance. Furthermore, the projected gradient descent algorithm solves the vector corresponding to the attack text. Our target-driven black-box attack approach includes two attack strategies: token manipulation and misinformation attack. Experimental results on multiple Large Language Models and datasets demonstrate the effectiveness of our attack method.

摘要: 现有的大型语言模型(LLM)为大规模面向用户的自然语言任务提供了坚实的基础。许多用户可以很容易地通过用户界面注入敌意文本或指令，从而导致LLM模型的安全挑战，如语言模型无法给出正确的答案。虽然目前有大量关于黑盒攻击的研究，但这些黑盒攻击大多采用随机和启发式策略。目前尚不清楚这些策略如何与攻击成功率相关，从而有效地提高模型的健壮性。为了解决这一问题，我们提出了目标驱动的黑盒攻击方法，以最大化明文和攻击文本的条件概率之间的KL偏差，从而重新定义攻击的目标。将距离最大化问题转化为基于攻击目标的两个凸优化问题来求解攻击文本并估计协方差。此外，投影梯度下降算法求解与攻击文本对应的向量。我们的目标驱动的黑盒攻击方法包括两种攻击策略：令牌操纵和错误信息攻击。在多个大型语言模型和数据集上的实验结果证明了该攻击方法的有效性。



## **40. DAGER: Exact Gradient Inversion for Large Language Models**

DAGER：大型语言模型的精确梯度倒置 cs.LG

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2405.15586v2) [paper-pdf](http://arxiv.org/pdf/2405.15586v2)

**Authors**: Ivo Petrov, Dimitar I. Dimitrov, Maximilian Baader, Mark Niklas Müller, Martin Vechev

**Abstract**: Federated learning works by aggregating locally computed gradients from multiple clients, thus enabling collaborative training without sharing private client data. However, prior work has shown that the data can actually be recovered by the server using so-called gradient inversion attacks. While these attacks perform well when applied on images, they are limited in the text domain and only permit approximate reconstruction of small batches and short input sequences. In this work, we propose DAGER, the first algorithm to recover whole batches of input text exactly. DAGER leverages the low-rank structure of self-attention layer gradients and the discrete nature of token embeddings to efficiently check if a given token sequence is part of the client data. We use this check to exactly recover full batches in the honest-but-curious setting without any prior on the data for both encoder- and decoder-based architectures using exhaustive heuristic search and a greedy approach, respectively. We provide an efficient GPU implementation of DAGER and show experimentally that it recovers full batches of size up to 128 on large language models (LLMs), beating prior attacks in speed (20x at same batch size), scalability (10x larger batches), and reconstruction quality (ROUGE-1/2 > 0.99).

摘要: 联合学习的工作方式是聚合来自多个客户端的本地计算的梯度，从而在不共享私人客户端数据的情况下实现协作培训。然而，先前的工作表明，服务器实际上可以使用所谓的梯度反转攻击来恢复数据。虽然这些攻击在图像上应用时表现良好，但它们仅限于文本域，仅允许对小批次和短输入序列进行近似重建。在这项工作中，我们提出了第一个准确恢复整批输入文本的算法Dager。Dager利用自我关注层梯度的低等级结构和令牌嵌入的离散性质来有效地检查给定的令牌序列是否是客户端数据的一部分。我们使用这种检查，分别使用穷举启发式搜索和贪婪方法，在诚实但奇怪的设置中准确地恢复完整批次，而不需要对基于编码器和解码器的架构的数据进行任何先验。我们提供了一种高效的Dager的GPU实现，实验表明，它可以在大型语言模型(LLM)上恢复大小高达128的全批处理，在速度(相同批处理大小的20倍)、可伸缩性(大批处理10倍)和重建质量(Rouge-1/2>0.99)方面优于先前的攻击。



## **41. The VLLM Safety Paradox: Dual Ease in Jailbreak Attack and Defense**

VLLM安全悖论：越狱攻击和防御的双重轻松 cs.CR

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2411.08410v1) [paper-pdf](http://arxiv.org/pdf/2411.08410v1)

**Authors**: Yangyang Guo, Fangkai Jiao, Liqiang Nie, Mohan Kankanhalli

**Abstract**: The vulnerability of Vision Large Language Models (VLLMs) to jailbreak attacks appears as no surprise. However, recent defense mechanisms against these attacks have reached near-saturation performance on benchmarks, often with minimal effort. This simultaneous high performance in both attack and defense presents a perplexing paradox. Resolving it is critical for advancing the development of trustworthy models. To address this research gap, we first investigate why VLLMs are prone to these attacks. We then make a key observation: existing defense mechanisms suffer from an \textbf{over-prudence} problem, resulting in unexpected abstention even in the presence of benign inputs. Additionally, we find that the two representative evaluation methods for jailbreak often exhibit chance agreement. This limitation makes it potentially misleading when evaluating attack strategies or defense mechanisms. Beyond these empirical observations, our another contribution in this work is to repurpose the guardrails of LLMs on the shelf, as an effective alternative detector prior to VLLM response. We believe these findings offer useful insights to rethink the foundational development of VLLM safety with respect to benchmark datasets, evaluation methods, and defense strategies.

摘要: Vision Large Language Models(VLLM)在越狱攻击中的脆弱性似乎并不令人意外。然而，最近针对这些攻击的防御机制在基准测试中的性能已经接近饱和，通常只需很少的努力。这种同时在进攻和防守上的高表现提出了一个令人困惑的悖论。解决这一问题对于推动可信模型的发展至关重要。为了解决这一研究差距，我们首先调查了为什么VLLM容易受到这些攻击。然后，我们做了一个关键的观察：现有的防御机制存在过度谨慎的问题，导致即使存在良性投入，也会意外弃权。此外，我们发现，两种具有代表性的越狱评估方法往往表现出偶然性的一致性。这一限制使其在评估攻击策略或防御机制时具有潜在误导性。除了这些经验观察之外，我们在这项工作中的另一个贡献是重新利用架子上的LLM护栏，作为VLLM响应之前的有效替代探测器。我们相信，这些发现为重新思考VLLM安全性在基准数据集、评估方法和防御策略方面的基础性发展提供了有用的见解。



## **42. MultiKG: Multi-Source Threat Intelligence Aggregation for High-Quality Knowledge Graph Representation of Attack Techniques**

MultiKG：用于攻击技术的高质量知识图表示的多源威胁情报聚合 cs.CR

21 pages, 15 figures, 8 tables

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2411.08359v1) [paper-pdf](http://arxiv.org/pdf/2411.08359v1)

**Authors**: Jian Wang, Tiantian Zhu, Chunlin Xiong, Yan Chen

**Abstract**: The construction of attack technique knowledge graphs aims to transform various types of attack knowledge into structured representations for more effective attack procedure modeling. Existing methods typically rely on textual data, such as Cyber Threat Intelligence (CTI) reports, which are often coarse-grained and unstructured, resulting in incomplete and inaccurate knowledge graphs. To address these issues, we expand attack knowledge sources by incorporating audit logs and static code analysis alongside CTI reports, providing finer-grained data for constructing attack technique knowledge graphs.   We propose MultiKG, a fully automated framework that integrates multiple threat knowledge sources. MultiKG processes data from CTI reports, dynamic logs, and static code separately, then merges them into a unified attack knowledge graph. Through system design and the utilization of the Large Language Model (LLM), MultiKG automates the analysis, construction, and merging of attack graphs across these sources, producing a fine-grained, multi-source attack knowledge graph.   We implemented MultiKG and evaluated it using 1,015 real attack techniques and 9,006 attack intelligence entries from CTI reports. Results show that MultiKG effectively extracts attack knowledge graphs from diverse sources and aggregates them into accurate, comprehensive representations. Through case studies, we demonstrate that our approach directly benefits security tasks such as attack reconstruction and detection.

摘要: 攻击技术知识图的构建旨在将各种类型的攻击知识转化为结构化的表示形式，以便更有效地对攻击过程进行建模。现有方法通常依赖文本数据，如网络威胁情报(CTI)报告，这些数据通常是粗粒度和非结构化的，导致知识图谱不完整和不准确。为了解决这些问题，我们通过将审计日志和静态代码分析与CTI报告结合在一起来扩展攻击知识源，为构建攻击技术知识图提供更细粒度的数据。我们提出了一种集成多个威胁知识源的全自动化框架--MultiKG。MultiKG分别处理CTI报告、动态日志和静态代码中的数据，然后将它们合并到统一的攻击知识图中。通过系统设计和大型语言模型(LLM)的利用，MultiKG自动分析、构建和合并这些来源的攻击图，生成细粒度的多源攻击知识图。我们实施了MultiKG，并使用1,015项真实攻击技术和9,006个CTI报告中的攻击情报条目对其进行了评估。结果表明，MultiKG能有效地从不同来源提取攻击知识图，并将其聚合成准确、全面的表示。通过案例研究，我们证明了我们的方法直接有利于攻击重建和检测等安全任务。



## **43. Can adversarial attacks by large language models be attributed?**

大型语言模型的对抗攻击可以归因吗？ cs.AI

7 pages, 1 figure

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.08003v1) [paper-pdf](http://arxiv.org/pdf/2411.08003v1)

**Authors**: Manuel Cebrian, Jan Arne Telle

**Abstract**: Attributing outputs from Large Language Models (LLMs) in adversarial settings-such as cyberattacks and disinformation-presents significant challenges that are likely to grow in importance. We investigate this attribution problem using formal language theory, specifically language identification in the limit as introduced by Gold and extended by Angluin. By modeling LLM outputs as formal languages, we analyze whether finite text samples can uniquely pinpoint the originating model. Our results show that due to the non-identifiability of certain language classes, under some mild assumptions about overlapping outputs from fine-tuned models it is theoretically impossible to attribute outputs to specific LLMs with certainty. This holds also when accounting for expressivity limitations of Transformer architectures. Even with direct model access or comprehensive monitoring, significant computational hurdles impede attribution efforts. These findings highlight an urgent need for proactive measures to mitigate risks posed by adversarial LLM use as their influence continues to expand.

摘要: 将大型语言模型(LLM)的输出归因于敌对环境--如网络攻击和虚假信息--带来了重大挑战，而这些挑战的重要性可能会越来越大。我们使用形式化语言理论来研究这一归因问题，特别是Gold提出并由Anluin推广的极限语言识别问题。通过将LLM输出建模为形式语言，我们分析了有限文本样本是否能够唯一地定位原始模型。我们的结果表明，由于某些语言类别的不可识别性，在微调模型的输出重叠的一些温和假设下，理论上不可能确定地将输出归因于特定的LLM。当考虑到Transformer架构的表现力限制时，这也是成立的。即使有了直接的模型访问或全面的监测，重大的计算障碍也阻碍了归因努力。这些调查结果突出表明，迫切需要采取积极主动的措施，以减轻敌对使用LLM所带来的风险，因为它们的影响继续扩大。



## **44. Chain Association-based Attacking and Shielding Natural Language Processing Systems**

基于链关联的攻击和屏蔽自然语言处理系统 cs.CL

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07843v1) [paper-pdf](http://arxiv.org/pdf/2411.07843v1)

**Authors**: Jiacheng Huang, Long Chen

**Abstract**: Association as a gift enables people do not have to mention something in completely straightforward words and allows others to understand what they intend to refer to. In this paper, we propose a chain association-based adversarial attack against natural language processing systems, utilizing the comprehension gap between humans and machines. We first generate a chain association graph for Chinese characters based on the association paradigm for building search space of potential adversarial examples. Then, we introduce an discrete particle swarm optimization algorithm to search for the optimal adversarial examples. We conduct comprehensive experiments and show that advanced natural language processing models and applications, including large language models, are vulnerable to our attack, while humans appear good at understanding the perturbed text. We also explore two methods, including adversarial training and associative graph-based recovery, to shield systems from chain association-based attack. Since a few examples that use some derogatory terms, this paper contains materials that may be offensive or upsetting to some people.

摘要: 联想作为一种礼物，使人们不必用完全直截了当的语言来提及某事，并让其他人理解他们想指的是什么。本文利用人与机器之间的理解鸿沟，提出了一种基于链式联想的对抗性自然语言处理系统攻击方法。首先在联想范式的基础上生成汉字的链式联想图，构建潜在对抗性实例的搜索空间。然后，我们引入了离散粒子群优化算法来搜索最优的对抗性实例。我们进行了全面的实验，并表明高级自然语言处理模型和应用程序，包括大型语言模型，容易受到我们的攻击，而人类似乎很擅长理解受干扰的文本。我们还探索了两种方法，包括对抗性训练和基于联想图的恢复，以保护系统免受基于链关联的攻击。由于有几个例子使用了一些贬义性的术语，因此本文包含的材料可能会冒犯某些人或使某些人不安。



## **45. Revisiting the Adversarial Robustness of Vision Language Models: a Multimodal Perspective**

重新审视视觉语言模型的对抗鲁棒性：多模式视角 cs.CV

17 pages, 13 figures

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2404.19287v3) [paper-pdf](http://arxiv.org/pdf/2404.19287v3)

**Authors**: Wanqi Zhou, Shuanghao Bai, Danilo P. Mandic, Qibin Zhao, Badong Chen

**Abstract**: Pretrained vision-language models (VLMs) like CLIP exhibit exceptional generalization across diverse downstream tasks. While recent studies reveal their vulnerability to adversarial attacks, research to date has primarily focused on enhancing the robustness of image encoders against image-based attacks, with defenses against text-based and multimodal attacks remaining largely unexplored. To this end, this work presents the first comprehensive study on improving the adversarial robustness of VLMs against attacks targeting image, text, and multimodal inputs. This is achieved by proposing multimodal contrastive adversarial training (MMCoA). Such an approach strengthens the robustness of both image and text encoders by aligning the clean text embeddings with adversarial image embeddings, and adversarial text embeddings with clean image embeddings. The robustness of the proposed MMCoA is examined against existing defense methods over image, text, and multimodal attacks on the CLIP model. Extensive experiments on 15 datasets across two tasks reveal the characteristics of different adversarial defense methods under distinct distribution shifts and dataset complexities across the three attack types. This paves the way for a unified framework of adversarial robustness against different modality attacks, opening up new possibilities for securing VLMs against multimodal attacks. The code is available at https://github.com/ElleZWQ/MMCoA.git.

摘要: 像CLIP这样的预先训练的视觉语言模型(VLM)在不同的下游任务中表现出非凡的通用性。虽然最近的研究揭示了它们对对手攻击的脆弱性，但到目前为止的研究主要集中在增强图像编码器对基于图像的攻击的稳健性上，对基于文本的攻击和多模式攻击的防御在很大程度上仍未被探索。为此，本文首次全面研究了如何提高VLMS对图像、文本和多模式输入的攻击健壮性。这是通过提出多模式对比对抗训练(MMCoA)来实现的。这种方法通过将干净的文本嵌入与对抗性的图像嵌入以及对抗性的文本嵌入与干净的图像嵌入对齐来增强图像和文本编码器的稳健性。针对已有的针对图像、文本和多模式攻击的防御方法，对提出的MMCoA算法的鲁棒性进行了测试。在两个任务的15个数据集上进行了大量的实验，揭示了三种攻击类型在不同的分布变化和数据集复杂性下不同的对抗防御方法的特点。这为对抗不同模式攻击的对抗健壮性的统一框架铺平了道路，为保护VLM免受多模式攻击开辟了新的可能性。代码可在https://github.com/ElleZWQ/MMCoA.git.上获得



## **46. Zer0-Jack: A Memory-efficient Gradient-based Jailbreaking Method for Black-box Multi-modal Large Language Models**

Zer 0-Jack：一种用于黑匣子多模式大型语言模型的内存高效的基于对象的越狱方法 cs.LG

Accepted to Neurips SafeGenAi Workshop 2024

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07559v1) [paper-pdf](http://arxiv.org/pdf/2411.07559v1)

**Authors**: Tiejin Chen, Kaishen Wang, Hua Wei

**Abstract**: Jailbreaking methods, which induce Multi-modal Large Language Models (MLLMs) to output harmful responses, raise significant safety concerns. Among these methods, gradient-based approaches, which use gradients to generate malicious prompts, have been widely studied due to their high success rates in white-box settings, where full access to the model is available. However, these methods have notable limitations: they require white-box access, which is not always feasible, and involve high memory usage. To address scenarios where white-box access is unavailable, attackers often resort to transfer attacks. In transfer attacks, malicious inputs generated using white-box models are applied to black-box models, but this typically results in reduced attack performance. To overcome these challenges, we propose Zer0-Jack, a method that bypasses the need for white-box access by leveraging zeroth-order optimization. We propose patch coordinate descent to efficiently generate malicious image inputs to directly attack black-box MLLMs, which significantly reduces memory usage further. Through extensive experiments, Zer0-Jack achieves a high attack success rate across various models, surpassing previous transfer-based methods and performing comparably with existing white-box jailbreak techniques. Notably, Zer0-Jack achieves a 95\% attack success rate on MiniGPT-4 with the Harmful Behaviors Multi-modal Dataset on a black-box setting, demonstrating its effectiveness. Additionally, we show that Zer0-Jack can directly attack commercial MLLMs such as GPT-4o. Codes are provided in the supplement.

摘要: 越狱方法会导致多模式大型语言模型(MLLMS)产生有害的响应，引发了重大的安全问题。在这些方法中，基于梯度的方法使用梯度来生成恶意提示，由于其在白盒环境中的高成功率而得到了广泛的研究，在白盒环境中，完全可以访问模型。然而，这些方法有明显的局限性：它们需要白盒访问，这并不总是可行的，并且涉及高内存使用率。为了解决无法使用白盒访问的情况，攻击者通常会求助于传输攻击。在传输攻击中，使用白盒模型生成的恶意输入应用于黑盒模型，但这通常会导致攻击性能降低。为了克服这些挑战，我们提出了Zer0-Jack，这是一种通过利用零阶优化来绕过白盒访问的方法。我们提出了补丁坐标下降的方法来有效地生成恶意图像输入来直接攻击黑盒MLLMS，从而进一步显著地减少了内存使用量。通过广泛的实验，Zer0-Jack在各种模型上实现了高攻击成功率，超过了以前基于传输的方法，性能与现有的白盒越狱技术相当。值得注意的是，Zer0-Jack在黑盒设置的有害行为多模式数据集上对MiniGPT-4的攻击成功率达到了95%，证明了其有效性。此外，我们还证明了Zer0-Jack可以直接攻击GPT-40等商业MLLMS。附录中提供了代码。



## **47. On Active Privacy Auditing in Supervised Fine-tuning for White-Box Language Models**

白盒语言模型监督微调中的主动隐私审计 cs.CL

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07070v2) [paper-pdf](http://arxiv.org/pdf/2411.07070v2)

**Authors**: Qian Sun, Hanpeng Wu, Xi Sheryl Zhang

**Abstract**: The pretraining and fine-tuning approach has become the leading technique for various NLP applications. However, recent studies reveal that fine-tuning data, due to their sensitive nature, domain-specific characteristics, and identifiability, pose significant privacy concerns. To help develop more privacy-resilient fine-tuning models, we introduce a novel active privacy auditing framework, dubbed Parsing, designed to identify and quantify privacy leakage risks during the supervised fine-tuning (SFT) of language models (LMs). The framework leverages improved white-box membership inference attacks (MIAs) as the core technology, utilizing novel learning objectives and a two-stage pipeline to monitor the privacy of the LMs' fine-tuning process, maximizing the exposure of privacy risks. Additionally, we have improved the effectiveness of MIAs on large LMs including GPT-2, Llama2, and certain variants of them. Our research aims to provide the SFT community of LMs with a reliable, ready-to-use privacy auditing tool, and to offer valuable insights into safeguarding privacy during the fine-tuning process. Experimental results confirm the framework's efficiency across various models and tasks, emphasizing notable privacy concerns in the fine-tuning process. Project code available for https://anonymous.4open.science/r/PARSING-4817/.

摘要: 预训练和微调方法已成为各种NLP应用的主导技术。然而，最近的研究表明，由于数据的敏感性质、特定于领域的特征和可识别性，微调数据会带来严重的隐私问题。为了帮助开发更具隐私弹性的微调模型，我们引入了一个新的主动隐私审计框架，称为Parsing，旨在识别和量化语言模型(LMS)的监督微调(SFT)期间的隐私泄露风险。该框架利用改进的白盒成员关系推理攻击(MIA)作为核心技术，利用新的学习目标和两阶段管道来监控LMS微调过程的隐私，最大限度地增加隐私风险的暴露。此外，我们还改进了MIA在大型LMS上的有效性，包括GPT-2、Llama2及其某些变体。我们的研究旨在为LMS的SFT社区提供一个可靠的、随时可用的隐私审计工具，并为在微调过程中保护隐私提供有价值的见解。实验结果证实了该框架在各种模型和任务上的有效性，强调了在微调过程中值得注意的隐私问题。可用于https://anonymous.4open.science/r/PARSING-4817/.的项目代码



## **48. vTune: Verifiable Fine-Tuning for LLMs Through Backdooring**

VCE：通过Backdooring对LLM进行可验证的微调 cs.LG

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.06611v2) [paper-pdf](http://arxiv.org/pdf/2411.06611v2)

**Authors**: Eva Zhang, Arka Pal, Akilesh Potti, Micah Goldblum

**Abstract**: As fine-tuning large language models (LLMs) becomes increasingly prevalent, users often rely on third-party services with limited visibility into their fine-tuning processes. This lack of transparency raises the question: how do consumers verify that fine-tuning services are performed correctly? For instance, a service provider could claim to fine-tune a model for each user, yet simply send all users back the same base model. To address this issue, we propose vTune, a simple method that uses a small number of backdoor data points added to the training data to provide a statistical test for verifying that a provider fine-tuned a custom model on a particular user's dataset. Unlike existing works, vTune is able to scale to verification of fine-tuning on state-of-the-art LLMs, and can be used both with open-source and closed-source models. We test our approach across several model families and sizes as well as across multiple instruction-tuning datasets, and find that the statistical test is satisfied with p-values on the order of $\sim 10^{-40}$, with no negative impact on downstream task performance. Further, we explore several attacks that attempt to subvert vTune and demonstrate the method's robustness to these attacks.

摘要: 随着微调大型语言模型(LLM)变得越来越普遍，用户通常依赖于第三方服务，但对其微调过程的可见性有限。这种透明度的缺乏引发了一个问题：消费者如何验证微调服务是否正确执行？例如，服务提供商可以声称为每个用户微调一个型号，但只需将所有用户发送回相同的基本型号。为了解决这个问题，我们提出了vTune，这是一种简单的方法，它使用添加到训练数据的少量后门数据点来提供统计测试，以验证提供商是否对特定用户的数据集的自定义模型进行了微调。与现有的作品不同，vTune能够扩展到对最先进的LLM进行微调验证，并且可以与开源和封闭源代码模型一起使用。我们在几个模型系列和大小以及多个指令调优数据集上测试了我们的方法，发现统计测试满足p值的数量级为$\sim 10^{-40}$，并且不会对下游任务性能产生负面影响。进一步，我们研究了几种试图破坏vTune的攻击，并展示了该方法对这些攻击的健壮性。



## **49. Rapid Response: Mitigating LLM Jailbreaks with a Few Examples**

快速反应：通过一些例子缓解LLM越狱 cs.CL

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07494v1) [paper-pdf](http://arxiv.org/pdf/2411.07494v1)

**Authors**: Alwin Peng, Julian Michael, Henry Sleight, Ethan Perez, Mrinank Sharma

**Abstract**: As large language models (LLMs) grow more powerful, ensuring their safety against misuse becomes crucial. While researchers have focused on developing robust defenses, no method has yet achieved complete invulnerability to attacks. We propose an alternative approach: instead of seeking perfect adversarial robustness, we develop rapid response techniques to look to block whole classes of jailbreaks after observing only a handful of attacks. To study this setting, we develop RapidResponseBench, a benchmark that measures a defense's robustness against various jailbreak strategies after adapting to a few observed examples. We evaluate five rapid response methods, all of which use jailbreak proliferation, where we automatically generate additional jailbreaks similar to the examples observed. Our strongest method, which fine-tunes an input classifier to block proliferated jailbreaks, reduces attack success rate by a factor greater than 240 on an in-distribution set of jailbreaks and a factor greater than 15 on an out-of-distribution set, having observed just one example of each jailbreaking strategy. Moreover, further studies suggest that the quality of proliferation model and number of proliferated examples play an key role in the effectiveness of this defense. Overall, our results highlight the potential of responding rapidly to novel jailbreaks to limit LLM misuse.

摘要: 随着大型语言模型(LLM)变得越来越强大，确保它们的安全性以防止误用变得至关重要。虽然研究人员专注于开发强大的防御系统，但还没有一种方法能够完全抵御攻击。我们提出了另一种方法：我们不是寻求完美的对手健壮性，而是开发快速响应技术，在仅观察到少数几次攻击后，寻求阻止整个类别的越狱。为了研究这种情况，我们开发了RapidResponseBch，这是一个基准，在适应了几个观察到的例子后，衡量了防御对各种越狱策略的健壮性。我们评估了五种快速响应方法，所有这些方法都使用越狱扩散，在这些方法中，我们自动生成与观察到的示例类似的额外越狱。我们最强大的方法是微调输入分类器以阻止越狱激增，在仅观察到每个越狱策略的一个示例后，在分布内越狱集合上将攻击成功率降低240倍以上，在分布外集合上降低15倍以上。此外，进一步的研究表明，扩散模型的质量和扩散实例的数量在这一防御措施的有效性中起着关键作用。总体而言，我们的结果突出了对新型越狱做出快速反应以限制LLM滥用的潜力。



## **50. DrAttack: Prompt Decomposition and Reconstruction Makes Powerful LLM Jailbreakers**

DrAttack：强大的快速分解和重建让LLM越狱者 cs.CR

**SubmitDate**: 2024-11-11    [abs](http://arxiv.org/abs/2402.16914v3) [paper-pdf](http://arxiv.org/pdf/2402.16914v3)

**Authors**: Xirui Li, Ruochen Wang, Minhao Cheng, Tianyi Zhou, Cho-Jui Hsieh

**Abstract**: The safety alignment of Large Language Models (LLMs) is vulnerable to both manual and automated jailbreak attacks, which adversarially trigger LLMs to output harmful content. However, current methods for jailbreaking LLMs, which nest entire harmful prompts, are not effective at concealing malicious intent and can be easily identified and rejected by well-aligned LLMs. This paper discovers that decomposing a malicious prompt into separated sub-prompts can effectively obscure its underlying malicious intent by presenting it in a fragmented, less detectable form, thereby addressing these limitations. We introduce an automatic prompt \textbf{D}ecomposition and \textbf{R}econstruction framework for jailbreak \textbf{Attack} (DrAttack). DrAttack includes three key components: (a) `Decomposition' of the original prompt into sub-prompts, (b) `Reconstruction' of these sub-prompts implicitly by in-context learning with semantically similar but harmless reassembling demo, and (c) a `Synonym Search' of sub-prompts, aiming to find sub-prompts' synonyms that maintain the original intent while jailbreaking LLMs. An extensive empirical study across multiple open-source and closed-source LLMs demonstrates that, with a significantly reduced number of queries, DrAttack obtains a substantial gain of success rate over prior SOTA prompt-only attackers. Notably, the success rate of 78.0\% on GPT-4 with merely 15 queries surpassed previous art by 33.1\%. The project is available at https://github.com/xirui-li/DrAttack.

摘要: 大型语言模型(LLM)的安全一致性容易受到手动和自动越狱攻击的攻击，这些攻击会相反地触发LLM输出有害内容。然而，当前的越狱LLM方法嵌套了整个有害的提示，在隐藏恶意意图方面并不有效，很容易被排列良好的LLM识别和拒绝。本文发现，通过将恶意提示分解为单独的子提示，可以通过以零散的、较难检测的形式来表示恶意提示，从而有效地掩盖潜在的恶意意图，从而解决这些限制。介绍了一种自动提示的文本bf{D}分解和文本bf{R}重构框架(DrAttack)。DrAttack包括三个关键组件：(A)将原始提示‘分解’成子提示，(B)通过使用语义相似但无害的重组演示的上下文学习隐式地‘重构’这些子提示，以及(C)对子提示进行‘同步搜索’，目的是在越狱LLM的同时找到保持原意的子提示的同义词。一项针对多个开源和闭源LLM的广泛经验研究表明，DrAttack在查询次数显著减少的情况下，与之前的Sota仅提示攻击者相比，获得了显著的成功率。值得注意的是，仅用15个查询在GPT-4上的成功率为78.0\%，比以前的ART高出33.1\%。该项目的网址为：https://github.com/xirui-li/DrAttack.。



