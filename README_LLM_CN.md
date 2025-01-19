# Latest Large Language Model Attack Papers
**update at 2025-01-19 10:26:03**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. TPIA: Towards Target-specific Prompt Injection Attack against Code-oriented Large Language Models**

TPIA：针对面向代码的大型语言模型的特定目标提示注入攻击 cs.CR

**SubmitDate**: 2025-01-16    [abs](http://arxiv.org/abs/2407.09164v4) [paper-pdf](http://arxiv.org/pdf/2407.09164v4)

**Authors**: Yuchen Yang, Hongwei Yao, Bingrun Yang, Yiling He, Yiming Li, Tianwei Zhang, Zhan Qin, Kui Ren, Chun Chen

**Abstract**: Recently, code-oriented large language models (Code LLMs) have been widely exploited to simplify and facilitate programming. With these tools, developers can easily generate the desired complete functional code based on incomplete code snippets and natural language prompts. Unfortunately, a few pioneering works revealed that these Code LLMs are vulnerable to backdoor and adversarial attacks. The former poisons the training data or model parameters, hijacking the LLMs to generate malicious code snippets when encountering the trigger. The latter crafts malicious adversarial input codes to reduce the quality of the generated codes. However, both attacks have some inherent limitations: backdoor attacks rely on the adversary's capability of controlling the model training process; adversarial attacks struggle with fulfilling specific malicious purposes. This paper presents a novel attack paradigm against Code LLMs, namely target-specific prompt injection attack (TPIA). TPIA generates non-functional perturbations containing the information of malicious instructions and inserts them into the victim's code context by spreading them into potentially used dependencies (e.g., packages or RAG's knowledge base). It induces the Code LLMs to generate attacker-specified malicious code snippets at the target location. In general, we compress the attacker-specified malicious objective into the perturbation by adversarial optimization based on greedy token search. We collect 13 representative malicious objectives to design 31 threat cases for three popular programming languages. We show that our TPIA can successfully attack three representative open-source Code LLMs (with an ASR of up to 97.9%) and two mainstream commercial Code LLM-integrated applications (with an ASR of over 90%) in all threat cases, using only a 12-token perturbation. Our work alerts a new practical threat of using Code LLMs.

摘要: 最近，面向代码的大型语言模型(Code LLM)已被广泛利用来简化和促进编程。使用这些工具，开发人员可以根据不完整的代码片段和自然语言提示轻松生成所需的完整功能代码。不幸的是，一些开创性的工作表明，这些代码LLM容易受到后门和对手的攻击。前者毒化训练数据或模型参数，在遇到触发器时劫持LLMS生成恶意代码片段。后者制作恶意敌意输入代码以降低生成代码的质量。然而，这两种攻击都有一些固有的局限性：后门攻击依赖于对手控制模型训练过程的能力；对抗性攻击难以实现特定的恶意目的。提出了一种新的针对Code LLMS的攻击范式，即目标特定的即时注入攻击(TPIA)。TPIA生成包含恶意指令信息的非功能性扰动，并通过将它们传播到可能使用的依赖项(例如，包或RAG的知识库)，将它们插入到受害者的代码上下文中。它诱导代码LLM在目标位置生成攻击者指定的恶意代码片段。一般而言，我们通过基于贪婪令牌搜索的对抗性优化将攻击者指定的恶意目标压缩为扰动。我们收集了13个具有代表性的恶意目标，为三种流行的编程语言设计了31个威胁案例。实验表明，在所有威胁情况下，仅需12个令牌的扰动，我们的TPIA就能成功攻击三个典型的开源代码LLMS(ASR高达97.9%)和两个主流商业代码LLM集成应用(ASR超过90%)。我们的工作警示了使用Code LLMS的新的实际威胁。



## **2. Tag&Tab: Pretraining Data Detection in Large Language Models Using Keyword-Based Membership Inference Attack**

标记& Tab：使用基于关键字的成员推断攻击在大型语言模型中进行预训练数据检测 cs.CR

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2501.08454v1) [paper-pdf](http://arxiv.org/pdf/2501.08454v1)

**Authors**: Sagiv Antebi, Edan Habler, Asaf Shabtai, Yuval Elovici

**Abstract**: Large language models (LLMs) have become essential digital task assistance tools. Their training relies heavily on the collection of vast amounts of data, which may include copyright-protected or sensitive information. Recent studies on the detection of pretraining data in LLMs have primarily focused on sentence-level or paragraph-level membership inference attacks (MIAs), usually involving probability analysis of the target model prediction tokens. However, the proposed methods often demonstrate poor performance, specifically in terms of accuracy, failing to account for the semantic importance of textual content and word significance. To address these shortcomings, we propose Tag&Tab, a novel approach for detecting data that has been used as part of the LLM pretraining. Our method leverages advanced natural language processing (NLP) techniques to tag keywords in the input text - a process we term Tagging. Then, the LLM is used to obtain the probabilities of these keywords and calculate their average log-likelihood to determine input text membership, a process we refer to as Tabbing. Our experiments on three benchmark datasets (BookMIA, MIMIR, and the Pile) and several open-source LLMs of varying sizes demonstrate an average increase in the AUC scores ranging from 4.1% to 12.1% over state-of-the-art methods. Tag&Tab not only sets a new standard for data leakage detection in LLMs, but its outstanding performance is a testament to the importance of words in MIAs on LLMs.

摘要: 大型语言模型(LLM)已成为必不可少的数字任务辅助工具。他们的培训在很大程度上依赖于收集大量数据，其中可能包括受版权保护的或敏感信息。最近关于LLMS中预训练数据检测的研究主要集中在句子级或段段级成员关系推理攻击(MIA)上，通常涉及对目标模型预测标记的概率分析。然而，所提出的方法往往表现出较差的性能，特别是在准确性方面，未能考虑文本内容和单词重要性的语义重要性。为了解决这些不足，我们提出了一种新的数据检测方法Tag&Tab，该方法已被用作LLM预训练的一部分。我们的方法利用高级自然语言处理(NLP)技术来标记输入文本中的关键字-我们将这一过程称为标记。然后，使用LLM来获得这些关键字的概率，并计算它们的平均对数似然率以确定输入文本成员资格，我们将这个过程称为制表符。我们在三个基准数据集(BookMIA、Mimir和The Pull)和几个不同大小的开源LLM上的实验表明，与最先进的方法相比，AUC得分平均提高了4.1%到12.1%。Tag&Tab不仅为LLMS中的数据泄漏检测建立了一个新的标准，而且其出色的性能证明了字在LLMS上的MIA中的重要性。



## **3. Exploring Robustness of LLMs to Sociodemographically-Conditioned Paraphrasing**

探索LLM对社会人口学条件解释的稳健性 cs.CL

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2501.08276v1) [paper-pdf](http://arxiv.org/pdf/2501.08276v1)

**Authors**: Pulkit Arora, Akbar Karimi, Lucie Flek

**Abstract**: Large Language Models (LLMs) have shown impressive performance in various NLP tasks. However, there are concerns about their reliability in different domains of linguistic variations. Many works have proposed robustness evaluation measures for local adversarial attacks, but we need globally robust models unbiased to different language styles. We take a broader approach to explore a wider range of variations across sociodemographic dimensions to perform structured reliability tests on the reasoning capacity of language models. We extend the SocialIQA dataset to create diverse paraphrased sets conditioned on sociodemographic styles. The assessment aims to provide a deeper understanding of LLMs in (a) their capability of generating demographic paraphrases with engineered prompts and (b) their reasoning capabilities in real-world, complex language scenarios. We also explore measures such as perplexity, explainability, and ATOMIC performance of paraphrases for fine-grained reliability analysis of LLMs on these sets. We find that demographic-specific paraphrasing significantly impacts the performance of language models, indicating that the subtleties of language variations remain a significant challenge. The code and dataset will be made available for reproducibility and future research.

摘要: 大型语言模型(LLM)在各种NLP任务中表现出了令人印象深刻的表现。然而，人们对它们在不同语言变异领域的可靠性表示担忧。已有许多研究提出了针对局部对抗性攻击的健壮性评估方法，但我们需要对不同的语言风格不偏向的全局健壮性模型。我们采取更广泛的方法来探索社会人口统计维度的更大范围的变化，以对语言模型的推理能力进行结构化的可靠性测试。我们扩展了SocialIQA数据集，以创建不同的释义集，条件是社会人口统计风格。这项评估旨在加深对LLMS的理解：(A)它们通过设计提示生成人口统计释义的能力；(B)它们在现实世界复杂语言情景中的推理能力。我们还探索了诸如困惑、可解释性和释义的原子性能等度量，用于这些集合上的LLMS的细粒度可靠性分析。我们发现，特定于人口统计的释义显著影响了语言模型的性能，这表明语言变异的微妙之处仍然是一个重大挑战。代码和数据集将提供给可重复性和未来的研究。



## **4. I Can Find You in Seconds! Leveraging Large Language Models for Code Authorship Attribution**

我可以在几秒钟内找到你！利用大型语言模型进行代码作者归因 cs.SE

12 pages, 5 figures,

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2501.08165v1) [paper-pdf](http://arxiv.org/pdf/2501.08165v1)

**Authors**: Soohyeon Choi, Yong Kiam Tan, Mark Huasong Meng, Mohamed Ragab, Soumik Mondal, David Mohaisen, Khin Mi Mi Aung

**Abstract**: Source code authorship attribution is important in software forensics, plagiarism detection, and protecting software patch integrity. Existing techniques often rely on supervised machine learning, which struggles with generalization across different programming languages and coding styles due to the need for large labeled datasets. Inspired by recent advances in natural language authorship analysis using large language models (LLMs), which have shown exceptional performance without task-specific tuning, this paper explores the use of LLMs for source code authorship attribution.   We present a comprehensive study demonstrating that state-of-the-art LLMs can successfully attribute source code authorship across different languages. LLMs can determine whether two code snippets are written by the same author with zero-shot prompting, achieving a Matthews Correlation Coefficient (MCC) of 0.78, and can attribute code authorship from a small set of reference code snippets via few-shot learning, achieving MCC of 0.77. Additionally, LLMs show some adversarial robustness against misattribution attacks.   Despite these capabilities, we found that naive prompting of LLMs does not scale well with a large number of authors due to input token limitations. To address this, we propose a tournament-style approach for large-scale attribution. Evaluating this approach on datasets of C++ (500 authors, 26,355 samples) and Java (686 authors, 55,267 samples) code from GitHub, we achieve classification accuracy of up to 65% for C++ and 68.7% for Java using only one reference per author. These results open new possibilities for applying LLMs to code authorship attribution in cybersecurity and software engineering.

摘要: 源代码作者归属在软件取证、抄袭检测和保护软件补丁完整性方面非常重要。现有技术通常依赖于有监督的机器学习，由于需要大型标签数据集，因此难以跨不同的编程语言和编码风格进行泛化。受使用大语言模型(LLM)的自然语言作者分析的最新进展的启发，LLM在没有特定任务调整的情况下表现出了优异的性能，本文探索了LLM在源代码作者归属中的使用。我们提出了一项全面的研究，证明了最先进的LLMS可以成功地跨不同语言归因于源代码作者。LLMS可以在零镜头提示下确定两个代码片段是否由同一作者编写，马修斯相关系数(MCC)为0.78，并可以通过少镜头学习从一小组参考代码片段中归属代码作者，MCC为0.77。此外，LLM在抵抗错误归因攻击时表现出一定的对抗健壮性。尽管有这些功能，但我们发现，由于输入令牌的限制，LLM的幼稚提示不能很好地扩展到大量作者。为了解决这个问题，我们提出了一种锦标赛式的大规模归因方法。在GitHub的C++(500名作者，26,355个样本)和Java(686名作者，55,267个样本)代码的数据集上测试该方法，在每个作者只使用一个参考文献的情况下，C++的分类准确率达到65%，Java的分类准确率达到68.7%。这些结果为将LLMS应用于网络安全和软件工程中的代码作者归属开辟了新的可能性。



## **5. Self-Instruct Few-Shot Jailbreaking: Decompose the Attack into Pattern and Behavior Learning**

自学少枪越狱：将攻击分解为模式和行为学习 cs.AI

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2501.07959v1) [paper-pdf](http://arxiv.org/pdf/2501.07959v1)

**Authors**: Jiaqi Hua, Wanxu Wei

**Abstract**: Recently, several works have been conducted on jailbreaking Large Language Models (LLMs) with few-shot malicious demos. In particular, Zheng et al. (2024) focuses on improving the efficiency of Few-Shot Jailbreaking (FSJ) by injecting special tokens into the demos and employing demo-level random search. Nevertheless, this method lacks generality since it specifies the instruction-response structure. Moreover, the reason why inserting special tokens takes effect in inducing harmful behaviors is only empirically discussed. In this paper, we take a deeper insight into the mechanism of special token injection and propose Self-Instruct Few-Shot Jailbreaking (Self-Instruct-FSJ) facilitated with the demo-level greedy search. This framework decomposes the FSJ attack into pattern and behavior learning to exploit the model's vulnerabilities in a more generalized and efficient way. We conduct elaborate experiments to evaluate our method on common open-source models and compare it with baseline algorithms. Our code is available at https://github.com/iphosi/Self-Instruct-FSJ.

摘要: 最近，一些关于越狱大语言模型(LLM)的工作已经进行，并提供了几个几乎不可能成功的恶意演示。特别是，郑等人。(2024)致力于通过向演示中注入特殊令牌并采用演示级随机搜索来提高少发越狱(FSJ)的效率。然而，这种方法缺乏通用性，因为它规定了指令-响应结构。此外，插入特殊代币在诱导有害行为中起作用的原因仅是经验上的讨论。本文深入研究了特殊令牌注入的机制，提出了一种基于演示级贪婪搜索的自指示少命中率越狱算法(SELF-Induct-FSJ)。该框架将FSJ攻击分解为模式学习和行为学习，以更通用、更有效的方式利用模型的漏洞。我们在常见的开源模型上进行了详细的实验来评估我们的方法，并将其与基线算法进行了比较。我们的代码可以在https://github.com/iphosi/Self-Instruct-FSJ.上找到



## **6. Gandalf the Red: Adaptive Security for LLMs**

红色甘道夫：LLM的自适应安全 cs.LG

Niklas Pfister, V\'aclav Volhejn and Manuel Knott contributed equally

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2501.07927v1) [paper-pdf](http://arxiv.org/pdf/2501.07927v1)

**Authors**: Niklas Pfister, Václav Volhejn, Manuel Knott, Santiago Arias, Julia Bazińska, Mykhailo Bichurin, Alan Commike, Janet Darling, Peter Dienes, Matthew Fiedler, David Haber, Matthias Kraft, Marco Lancini, Max Mathys, Damián Pascual-Ortiz, Jakub Podolak, Adrià Romero-López, Kyriacos Shiarlis, Andreas Signer, Zsolt Terek, Athanasios Theocharis, Daniel Timbrell, Samuel Trautwein, Samuel Watts, Natalie Wu, Mateo Rojas-Carulla

**Abstract**: Current evaluations of defenses against prompt attacks in large language model (LLM) applications often overlook two critical factors: the dynamic nature of adversarial behavior and the usability penalties imposed on legitimate users by restrictive defenses. We propose D-SEC (Dynamic Security Utility Threat Model), which explicitly separates attackers from legitimate users, models multi-step interactions, and rigorously expresses the security-utility in an optimizable form. We further address the shortcomings in existing evaluations by introducing Gandalf, a crowd-sourced, gamified red-teaming platform designed to generate realistic, adaptive attack datasets. Using Gandalf, we collect and release a dataset of 279k prompt attacks. Complemented by benign user data, our analysis reveals the interplay between security and utility, showing that defenses integrated in the LLM (e.g., system prompts) can degrade usability even without blocking requests. We demonstrate that restricted application domains, defense-in-depth, and adaptive defenses are effective strategies for building secure and useful LLM applications. Code is available at \href{https://github.com/lakeraai/dsec-gandalf}{\texttt{https://github.com/lakeraai/dsec-gandalf}}.

摘要: 目前对大型语言模型(LLM)应用程序中针对即时攻击的防御措施的评估往往忽略了两个关键因素：敌意行为的动态性质和限制性防御措施对合法用户的可用性惩罚。提出了动态安全效用威胁模型D-SEC(Dynamic Security Utility Threat Model)，该模型明确地将攻击者和合法用户分开，对多步交互进行建模，并以可优化的形式严格表达安全效用。我们通过引入Gandalf进一步解决了现有评估中的缺陷，Gandalf是一个众包、游戏化的红色团队平台，旨在生成现实的、自适应的攻击数据集。使用Gandalf，我们收集并发布了279K提示攻击的数据集。在良性用户数据的补充下，我们的分析揭示了安全性和实用性之间的相互作用，表明LLM中集成的防御措施(例如系统提示)即使在不阻止请求的情况下也会降低可用性。我们演示了受限应用程序域、深度防御和自适应防御是构建安全且有用的LLM应用程序的有效策略。代码可在\href{https://github.com/lakeraai/dsec-gandalf}{\texttt{https://github.com/lakeraai/dsec-gandalf}}.上找到



## **7. SecAlign: Defending Against Prompt Injection with Preference Optimization**

SecAlign：通过偏好优化抵御提示注入 cs.CR

Key words: prompt injection defense, LLM security, LLM-integrated  applications

**SubmitDate**: 2025-01-13    [abs](http://arxiv.org/abs/2410.05451v2) [paper-pdf](http://arxiv.org/pdf/2410.05451v2)

**Authors**: Sizhe Chen, Arman Zharmagambetov, Saeed Mahloujifar, Kamalika Chaudhuri, David Wagner, Chuan Guo

**Abstract**: Large language models (LLMs) are becoming increasingly prevalent in modern software systems, interfacing between the user and the Internet to assist with tasks that require advanced language understanding. To accomplish these tasks, the LLM often uses external data sources such as user documents, web retrieval, results from API calls, etc. This opens up new avenues for attackers to manipulate the LLM via prompt injection. Adversarial prompts can be injected into external data sources to override the system's intended instruction and instead execute a malicious instruction.   To mitigate this vulnerability, we propose a new defense called SecAlign based on the technique of preference optimization. Our defense first constructs a preference dataset with prompt-injected inputs, secure outputs (ones that respond to the legitimate instruction), and insecure outputs (ones that respond to the injection). We then perform preference optimization on this dataset to teach the LLM to prefer the secure output over the insecure one. This provides the first known method that reduces the success rates of various prompt injections to around 0%, even against attacks much more sophisticated than ones seen during training. This indicates our defense generalizes well against unknown and yet-to-come attacks. Also, our defended models are still practical with similar utility to the one before our defensive training. Our code is at https://github.com/facebookresearch/SecAlign

摘要: 大型语言模型(LLM)在现代软件系统中正变得越来越普遍，它们在用户和互联网之间进行交互，以帮助完成需要高级语言理解的任务。为了完成这些任务，LLM通常使用外部数据源，如用户文档、Web检索、API调用结果等。这为攻击者通过提示注入操纵LLM开辟了新的途径。敌意提示可以被注入外部数据源，以覆盖系统的预期指令，而不是执行恶意指令。为了缓解这一漏洞，我们提出了一种新的基于偏好优化技术的防御机制SecAlign。我们的辩护首先构建一个偏好数据集，其中包含即时注入输入、安全输出(响应合法指令的输出)和不安全的输出(响应注入的输出)。然后，我们对该数据集执行偏好优化，以教导LLM更喜欢安全的输出而不是不安全的输出。这提供了第一种已知的方法，可以将各种快速注射的成功率降低到0%左右，即使是对比训练期间看到的更复杂的攻击也是如此。这表明我们的防御对未知和即将到来的攻击具有很好的一般性。此外，我们的防守模型仍然实用，与我们防守训练之前的实用程度相似。我们的代码在https://github.com/facebookresearch/SecAlign



## **8. Exploring and Mitigating Adversarial Manipulation of Voting-Based Leaderboards**

探索和缓解基于投票的排行榜的对抗操纵 cs.LG

**SubmitDate**: 2025-01-13    [abs](http://arxiv.org/abs/2501.07493v1) [paper-pdf](http://arxiv.org/pdf/2501.07493v1)

**Authors**: Yangsibo Huang, Milad Nasr, Anastasios Angelopoulos, Nicholas Carlini, Wei-Lin Chiang, Christopher A. Choquette-Choo, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Ken Ziyu Liu, Ion Stoica, Florian Tramer, Chiyuan Zhang

**Abstract**: It is now common to evaluate Large Language Models (LLMs) by having humans manually vote to evaluate model outputs, in contrast to typical benchmarks that evaluate knowledge or skill at some particular task. Chatbot Arena, the most popular benchmark of this type, ranks models by asking users to select the better response between two randomly selected models (without revealing which model was responsible for the generations). These platforms are widely trusted as a fair and accurate measure of LLM capabilities. In this paper, we show that if bot protection and other defenses are not implemented, these voting-based benchmarks are potentially vulnerable to adversarial manipulation. Specifically, we show that an attacker can alter the leaderboard (to promote their favorite model or demote competitors) at the cost of roughly a thousand votes (verified in a simulated, offline version of Chatbot Arena). Our attack consists of two steps: first, we show how an attacker can determine which model was used to generate a given reply with more than $95\%$ accuracy; and then, the attacker can use this information to consistently vote for (or against) a target model. Working with the Chatbot Arena developers, we identify, propose, and implement mitigations to improve the robustness of Chatbot Arena against adversarial manipulation, which, based on our analysis, substantially increases the cost of such attacks. Some of these defenses were present before our collaboration, such as bot protection with Cloudflare, malicious user detection, and rate limiting. Others, including reCAPTCHA and login are being integrated to strengthen the security in Chatbot Arena.

摘要: 现在，通过人工投票评估模型输出来评估大型语言模型(LLM)是很常见的，这与评估某些特定任务的知识或技能的典型基准测试形成了鲜明对比。聊天机器人Arena是这类最受欢迎的基准，它通过让用户在两个随机选择的模型中选择反应更好的模型来对模型进行排名(没有透露哪种模型对几代人负责)。这些平台受到广泛信任，被认为是LLM能力的公平和准确衡量标准。在这篇文章中，我们表明，如果不实施BOT保护和其他防御措施，这些基于投票的基准可能容易受到敌意操纵。具体地说，我们展示了攻击者可以改变排行榜(以推广他们最喜欢的型号或降级竞争对手)，代价是大约1000张选票(在模拟的聊天机器人竞技场的离线版本中进行验证)。我们的攻击包括两个步骤：首先，我们展示了攻击者如何确定哪个模型被用来生成精度超过$95\\$的给定回复；然后，攻击者可以使用这些信息一致地投票支持(或反对)目标模型。我们与聊天机器人Arena的开发人员合作，识别、提出并实施缓解措施，以提高聊天机器人Arena对敌意操纵的健壮性，根据我们的分析，这将大大增加此类攻击的成本。其中一些防御措施在我们合作之前就已经存在，例如使用Cloudflare的机器人保护、恶意用户检测和速率限制。包括reCAPTCHA和LOGIN在内的其他软件正在整合，以加强聊天机器人竞技场的安全性。



## **9. DrLLM: Prompt-Enhanced Distributed Denial-of-Service Resistance Method with Large Language Models**

DrLLM：具有大型语言模型的预算增强型分布式拒绝服务抵抗方法 cs.CR

Accepted by ICASSP2025

**SubmitDate**: 2025-01-13    [abs](http://arxiv.org/abs/2409.10561v3) [paper-pdf](http://arxiv.org/pdf/2409.10561v3)

**Authors**: Zhenyu Yin, Shang Liu, Guangyuan Xu

**Abstract**: The increasing number of Distributed Denial of Service (DDoS) attacks poses a major threat to the Internet, highlighting the importance of DDoS mitigation. Most existing approaches require complex training methods to learn data features, which increases the complexity and generality of the application. In this paper, we propose DrLLM, which aims to mine anomalous traffic information in zero-shot scenarios through Large Language Models (LLMs). To bridge the gap between DrLLM and existing approaches, we embed the global and local information of the traffic data into the reasoning paradigm and design three modules, namely Knowledge Embedding, Token Embedding, and Progressive Role Reasoning, for data representation and reasoning. In addition we explore the generalization of prompt engineering in the cybersecurity domain to improve the classification capability of DrLLM. Our ablation experiments demonstrate the applicability of DrLLM in zero-shot scenarios and further demonstrate the potential of LLMs in the network domains. DrLLM implementation code has been open-sourced at https://github.com/liuup/DrLLM.

摘要: 越来越多的分布式拒绝服务(DDoS)攻击对互联网构成了重大威胁，突显了缓解DDoS的重要性。现有的大多数方法需要复杂的训练方法来学习数据特征，这增加了应用程序的复杂性和通用性。在本文中，我们提出了DrLLM，旨在通过大型语言模型(LLMS)挖掘零镜头场景中的异常交通信息。为了弥补DrLLM与已有方法之间的差距，我们将交通数据的全局和局部信息嵌入到推理范式中，并设计了三个模块，即知识嵌入、令牌嵌入和渐进角色推理，用于数据表示和推理。此外，我们还探索了快速工程在网络安全领域的泛化，以提高DrLLM的分类能力。我们的烧蚀实验证明了DrLLM在零射情况下的适用性，并进一步展示了LLM在网络领域的潜力。DrLLm实现代码已在https://github.com/liuup/DrLLM.上开源



## **10. PSA-VLM: Enhancing Vision-Language Model Safety through Progressive Concept-Bottleneck-Driven Alignment**

PSA-VLM：通过渐进式概念瓶颈驱动对齐增强视觉语言模型安全性 cs.CV

arXiv admin note: substantial text overlap with arXiv:2405.13581

**SubmitDate**: 2025-01-13    [abs](http://arxiv.org/abs/2411.11543v4) [paper-pdf](http://arxiv.org/pdf/2411.11543v4)

**Authors**: Zhendong Liu, Yuanbi Nie, Yingshui Tan, Jiaheng Liu, Xiangyu Yue, Qiushi Cui, Chongjun Wang, Xiaoyong Zhu, Bo Zheng

**Abstract**: Benefiting from the powerful capabilities of Large Language Models (LLMs), pre-trained visual encoder models connected to LLMs form Vision Language Models (VLMs). However, recent research shows that the visual modality in VLMs is highly vulnerable, allowing attackers to bypass safety alignment in LLMs through visually transmitted content, launching harmful attacks. To address this challenge, we propose a progressive concept-based alignment strategy, PSA-VLM, which incorporates safety modules as concept bottlenecks to enhance visual modality safety alignment. By aligning model predictions with specific safety concepts, we improve defenses against risky images, enhancing explainability and controllability while minimally impacting general performance. Our method is obtained through two-stage training. The low computational cost of the first stage brings very effective performance improvement, and the fine-tuning of the language model in the second stage further improves the safety performance. Our method achieves state-of-the-art results on popular VLM safety benchmark.

摘要: 得益于大型语言模型的强大功能，连接到大型语言模型的预先训练的视觉编码器模型形成了视觉语言模型。然而，最近的研究表明，VLMS中的视觉通道非常容易受到攻击，使得攻击者能够通过视觉传输的内容绕过LLMS中的安全对齐，从而发起有害攻击。为了应对这一挑战，我们提出了一种基于概念的渐进式对齐策略PSA-VLM，该策略将安全模块作为概念瓶颈纳入其中，以增强视觉通道的安全对齐。通过将模型预测与特定的安全概念相结合，我们改进了对危险图像的防御，增强了可解释性和可控性，同时将对总体性能的影响降至最低。我们的方法是通过两个阶段的训练获得的。第一阶段的低运算量带来了非常有效的性能提升，第二阶段对语言模型的微调进一步提高了安全性能。我们的方法在流行的VLM安全基准上获得了最先进的结果。



## **11. Images are Achilles' Heel of Alignment: Exploiting Visual Vulnerabilities for Jailbreaking Multimodal Large Language Models**

图像是一致的阿喀琉斯之踵：利用视觉漏洞破解多模式大型语言模型 cs.CV

ECCV 2024 Oral

**SubmitDate**: 2025-01-13    [abs](http://arxiv.org/abs/2403.09792v3) [paper-pdf](http://arxiv.org/pdf/2403.09792v3)

**Authors**: Yifan Li, Hangyu Guo, Kun Zhou, Wayne Xin Zhao, Ji-Rong Wen

**Abstract**: In this paper, we study the harmlessness alignment problem of multimodal large language models (MLLMs). We conduct a systematic empirical analysis of the harmlessness performance of representative MLLMs and reveal that the image input poses the alignment vulnerability of MLLMs. Inspired by this, we propose a novel jailbreak method named HADES, which hides and amplifies the harmfulness of the malicious intent within the text input, using meticulously crafted images. Experimental results show that HADES can effectively jailbreak existing MLLMs, which achieves an average Attack Success Rate (ASR) of 90.26% for LLaVA-1.5 and 71.60% for Gemini Pro Vision. Our code and data are available at https://github.com/RUCAIBox/HADES.

摘要: 本文研究了多模式大型语言模型（MLLM）的无害对齐问题。我们对代表性MLLM的无害性能进行了系统的实证分析，揭示了图像输入造成了MLLM的对齐脆弱性。受此启发，我们提出了一种名为HADES的新颖越狱方法，该方法使用精心制作的图像隐藏和放大文本输入中恶意意图的危害性。实验结果表明，HADES可以有效越狱现有的MLLM，LLaVA-1.5的平均攻击成功率（ASB）为90.26%，Gemini Pro Vision的平均攻击成功率（ASB）为71.60%。我们的代码和数据可在https://github.com/RUCAIBox/HADES上获取。



## **12. TrustRAG: Enhancing Robustness and Trustworthiness in RAG**

TrustRAG：增强RAG的稳健性和可信性 cs.CL

**SubmitDate**: 2025-01-12    [abs](http://arxiv.org/abs/2501.00879v2) [paper-pdf](http://arxiv.org/pdf/2501.00879v2)

**Authors**: Huichi Zhou, Kin-Hei Lee, Zhonghao Zhan, Yue Chen, Zhenhao Li, Zhaoyang Wang, Hamed Haddadi, Emine Yilmaz

**Abstract**: Retrieval-Augmented Generation (RAG) systems enhance large language models (LLMs) by integrating external knowledge sources, enabling more accurate and contextually relevant responses tailored to user queries. However, these systems remain vulnerable to corpus poisoning attacks that can significantly degrade LLM performance through the injection of malicious content. To address these challenges, we propose TrustRAG, a robust framework that systematically filters compromised and irrelevant contents before they are retrieved for generation. Our approach implements a two-stage defense mechanism: At the first stage, it employs K-means clustering to identify potential attack patterns in retrieved documents using cosine similarity and ROUGE metrics as guidance, effectively isolating suspicious content. Secondly, it performs a self-assessment which detects malicious documents and resolves discrepancies between the model's internal knowledge and external information. TrustRAG functions as a plug-and-play, training-free module that integrates seamlessly with any language model, whether open or closed-source. In addition, TrustRAG maintains high contextual relevance while strengthening defenses against corpus poisoning attacks. Through extensive experimental validation, we demonstrate that TrustRAG delivers substantial improvements in retrieval accuracy, efficiency, and attack resistance compared to existing approaches across multiple model architectures and datasets. We have made TrustRAG available as open-source software at \url{https://github.com/HuichiZhou/TrustRAG}.

摘要: 检索-增强生成(RAG)系统通过集成外部知识源来增强大型语言模型(LLM)，从而能够针对用户查询定制更准确且与上下文相关的响应。然而，这些系统仍然容易受到语料库中毒攻击，这些攻击可能会通过注入恶意内容显著降低LLM的性能。为了应对这些挑战，我们提出了TrustRAG，这是一个健壮的框架，在检索并生成之前，系统地过滤受攻击和无关的内容。该方法实现了两阶段防御机制：在第一阶段，利用K-均值聚类，以余弦相似度和Rouge度量为指导，识别文档中潜在的攻击模式，有效地隔离可疑内容。其次，它执行自我评估，检测恶意文档，并解决模型内部知识和外部信息之间的差异。TrustRAG是一个即插即用、无需培训的模块，可以与任何语言模型无缝集成，无论是开放源代码还是封闭源代码。此外，TrustRAG在加强对语料库中毒攻击的防御的同时，保持了高度的上下文相关性。通过广泛的实验验证，我们证明了TrustRAG在检索准确性、效率和抗攻击方面比现有的跨多个模型架构和数据集的方法有了实质性的改进。我们已将TrustRAG作为开源软件提供给\url{https://github.com/HuichiZhou/TrustRAG}.



## **13. ModelShield: Adaptive and Robust Watermark against Model Extraction Attack**

Model Shield：针对模型提取攻击的自适应鲁棒水印 cs.CR

**SubmitDate**: 2025-01-12    [abs](http://arxiv.org/abs/2405.02365v4) [paper-pdf](http://arxiv.org/pdf/2405.02365v4)

**Authors**: Kaiyi Pang, Tao Qi, Chuhan Wu, Minhao Bai, Minghu Jiang, Yongfeng Huang

**Abstract**: Large language models (LLMs) demonstrate general intelligence across a variety of machine learning tasks, thereby enhancing the commercial value of their intellectual property (IP). To protect this IP, model owners typically allow user access only in a black-box manner, however, adversaries can still utilize model extraction attacks to steal the model intelligence encoded in model generation. Watermarking technology offers a promising solution for defending against such attacks by embedding unique identifiers into the model-generated content. However, existing watermarking methods often compromise the quality of generated content due to heuristic alterations and lack robust mechanisms to counteract adversarial strategies, thus limiting their practicality in real-world scenarios. In this paper, we introduce an adaptive and robust watermarking method (named ModelShield) to protect the IP of LLMs. Our method incorporates a self-watermarking mechanism that allows LLMs to autonomously insert watermarks into their generated content to avoid the degradation of model content. We also propose a robust watermark detection mechanism capable of effectively identifying watermark signals under the interference of varying adversarial strategies. Besides, ModelShield is a plug-and-play method that does not require additional model training, enhancing its applicability in LLM deployments. Extensive evaluations on two real-world datasets and three LLMs demonstrate that our method surpasses existing methods in terms of defense effectiveness and robustness while significantly reducing the degradation of watermarking on the model-generated content.

摘要: 大型语言模型(LLM)在各种机器学习任务中展示了一般智能，从而提高了其知识产权(IP)的商业价值。为了保护这个IP，模型所有者通常只允许用户以黑盒方式访问，但是，攻击者仍然可以利用模型提取攻击来窃取模型生成中编码的模型情报。水印技术通过在模型生成的内容中嵌入唯一标识符，为防御此类攻击提供了一种很有前途的解决方案。然而，现有的水印方法往往会由于启发式修改而影响生成内容的质量，并且缺乏强大的机制来对抗对抗性策略，从而限制了它们在现实世界场景中的实用性。本文提出了一种自适应的稳健水印算法(ModelShield)来保护LLMS的IP地址。我们的方法结合了一种自水印机制，允许LLM自主地在其生成的内容中插入水印，以避免模型内容的降级。我们还提出了一种稳健的水印检测机制，能够在不同的对抗策略的干扰下有效地识别水印信号。此外，ModelShield是一种即插即用的方法，不需要额外的模型培训，增强了其在LLM部署中的适用性。在两个真实数据集和三个LLM上的广泛评估表明，我们的方法在防御有效性和稳健性方面优于现有方法，同时显着降低了水印对模型生成内容的退化。



## **14. ZOQO: Zero-Order Quantized Optimization**

ZOQO：零阶量化优化 cs.LG

Accepted to ICASSP 2025

**SubmitDate**: 2025-01-12    [abs](http://arxiv.org/abs/2501.06736v1) [paper-pdf](http://arxiv.org/pdf/2501.06736v1)

**Authors**: Noga Bar, Raja Giryes

**Abstract**: The increasing computational and memory demands in deep learning present significant challenges, especially in resource-constrained environments. We introduce a zero-order quantized optimization (ZOQO) method designed for training models with quantized parameters and operations. Our approach leverages zero-order approximations of the gradient sign and adapts the learning process to maintain the parameters' quantization without the need for full-precision gradient calculations. We demonstrate the effectiveness of ZOQO through experiments in fine-tuning of large language models and black-box adversarial attacks. Despite the limitations of zero-order and quantized operations training, our method achieves competitive performance compared to full-precision methods, highlighting its potential for low-resource environments.

摘要: 深度学习中不断增加的计算和内存需求带来了重大挑战，尤其是在资源有限的环境中。我们引入了一种零阶量化优化（ZOQO）方法，专为具有量化参数和操作的训练模型而设计。我们的方法利用了梯度符号的零阶逼近，并调整学习过程以维持参数的量化，而无需全精度梯度计算。我们通过微调大型语言模型和黑匣子对抗攻击的实验来证明ZOQO的有效性。尽管零阶和量化操作训练存在局限性，但与全精度方法相比，我们的方法实现了有竞争力的性能，凸显了其在低资源环境中的潜力。



## **15. Measuring the Robustness of Reference-Free Dialogue Evaluation Systems**

衡量无参考对话评估系统的稳健性 cs.CL

**SubmitDate**: 2025-01-12    [abs](http://arxiv.org/abs/2501.06728v1) [paper-pdf](http://arxiv.org/pdf/2501.06728v1)

**Authors**: Justin Vasselli, Adam Nohejl, Taro Watanabe

**Abstract**: Advancements in dialogue systems powered by large language models (LLMs) have outpaced the development of reliable evaluation metrics, particularly for diverse and creative responses. We present a benchmark for evaluating the robustness of reference-free dialogue metrics against four categories of adversarial attacks: speaker tag prefixes, static responses, ungrammatical responses, and repeated conversational context. We analyze metrics such as DialogRPT, UniEval, and PromptEval -- a prompt-based method leveraging LLMs -- across grounded and ungrounded datasets. By examining both their correlation with human judgment and susceptibility to adversarial attacks, we find that these two axes are not always aligned; metrics that appear to be equivalent when judged by traditional benchmarks may, in fact, vary in their scores of adversarial responses. These findings motivate the development of nuanced evaluation frameworks to address real-world dialogue challenges.

摘要: 由大型语言模型（LLM）驱动的对话系统的进步已经超过了可靠评估指标的发展，特别是对于多样化和创造性的响应。我们提出了一个基准，用于评估无引用对话指标针对四类对抗攻击的稳健性：说话者标签前置、静态响应、不合语法的响应和重复对话上下文。我们跨基础和未基础数据集分析DialogRPT、UniEval和EntEval（一种利用LLM的基于预算的方法）等指标。通过检查它们与人类判断的相关性和对对抗攻击的易感性，我们发现这两个轴并不总是一致的;用传统基准判断时看似等效的指标实际上可能会有所不同。对抗反应的分数。这些发现促使制定细致入微的评估框架来应对现实世界的对话挑战。



## **16. LUMIA: Linear probing for Unimodal and MultiModal Membership Inference Attacks leveraging internal LLM states**

LUMIA：利用内部LLM状态进行单模式和多模式成员资格推理攻击的线性探测 cs.CR

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2411.19876v3) [paper-pdf](http://arxiv.org/pdf/2411.19876v3)

**Authors**: Luis Ibanez-Lissen, Lorena Gonzalez-Manzano, Jose Maria de Fuentes, Nicolas Anciaux, Joaquin Garcia-Alfaro

**Abstract**: Large Language Models (LLMs) are increasingly used in a variety of applications, but concerns around membership inference have grown in parallel. Previous efforts focus on black-to-grey-box models, thus neglecting the potential benefit from internal LLM information. To address this, we propose the use of Linear Probes (LPs) as a method to detect Membership Inference Attacks (MIAs) by examining internal activations of LLMs. Our approach, dubbed LUMIA, applies LPs layer-by-layer to get fine-grained data on the model inner workings. We test this method across several model architectures, sizes and datasets, including unimodal and multimodal tasks. In unimodal MIA, LUMIA achieves an average gain of 15.71 % in Area Under the Curve (AUC) over previous techniques. Remarkably, LUMIA reaches AUC>60% in 65.33% of cases -- an increment of 46.80% against the state of the art. Furthermore, our approach reveals key insights, such as the model layers where MIAs are most detectable. In multimodal models, LPs indicate that visual inputs can significantly contribute to detect MIAs -- AUC>60% is reached in 85.90% of experiments.

摘要: 大型语言模型(LLM)越来越多地用于各种应用程序，但围绕成员关系推理的关注也在平行增长。以往的研究主要集中在黑灰盒模型上，从而忽略了LLM内部信息的潜在益处。为了解决这一问题，我们提出使用线性探测器(LP)作为一种方法，通过检查LLP的内部激活来检测成员身份推理攻击(MIA)。我们的方法，称为Lumia，逐层应用LP，以获得关于模型内部工作的细粒度数据。我们在几个模型体系结构、大小和数据集上测试了这种方法，包括单模和多模任务。在单峰MIA中，Lumia的曲线下面积(AUC)比以前的技术平均增加了15.71%。值得注意的是，Lumia在65.33%的情况下达到AUC>60%--与最先进的水平相比增加了46.80%。此外，我们的方法揭示了关键的见解，例如最容易检测到MIA的模型层。在多通道模型中，LP表明视觉输入对检测MIA有显著贡献-85.90%的实验达到了60%以上的AUC。



## **17. Model Inversion in Split Learning for Personalized LLMs: New Insights from Information Bottleneck Theory**

个性化LLM分裂学习中的模型倒置：信息瓶颈理论的新见解 cs.LG

8 pages

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.05965v1) [paper-pdf](http://arxiv.org/pdf/2501.05965v1)

**Authors**: Yunmeng Shu, Shaofeng Li, Tian Dong, Yan Meng, Haojin Zhu

**Abstract**: Personalized Large Language Models (LLMs) have become increasingly prevalent, showcasing the impressive capabilities of models like GPT-4. This trend has also catalyzed extensive research on deploying LLMs on mobile devices. Feasible approaches for such edge-cloud deployment include using split learning. However, previous research has largely overlooked the privacy leakage associated with intermediate representations transmitted from devices to servers. This work is the first to identify model inversion attacks in the split learning framework for LLMs, emphasizing the necessity of secure defense. For the first time, we introduce mutual information entropy to understand the information propagation of Transformer-based LLMs and assess privacy attack performance for LLM blocks. To address the issue of representations being sparser and containing less information than embeddings, we propose a two-stage attack system in which the first part projects representations into the embedding space, and the second part uses a generative model to recover text from these embeddings. This design breaks down the complexity and achieves attack scores of 38%-75% in various scenarios, with an over 60% improvement over the SOTA. This work comprehensively highlights the potential privacy risks during the deployment of personalized LLMs on the edge side.

摘要: 个性化的大型语言模型(LLM)已经变得越来越流行，展示了GPT-4等模型令人印象深刻的能力。这一趋势也催生了关于在移动设备上部署LLM的广泛研究。这种边缘云部署的可行方法包括使用分裂学习。然而，以前的研究在很大程度上忽略了与从设备传输到服务器的中间表示相关联的隐私泄露。这项工作首次在LLMS的分裂学习框架中发现了模型反转攻击，强调了安全防御的必要性。首次引入互信息熵来理解基于变压器的LLMS的信息传播，并评估LLM块的隐私攻击性能。为了解决表示比嵌入更稀疏和包含的信息量更少的问题，我们提出了一个两阶段攻击系统，其中第一部分将表示投影到嵌入空间中，第二部分使用生成模型从这些嵌入中恢复文本。这种设计打破了复杂性，在不同场景下的攻击得分达到38%-75%，比SOTA提高了60%以上。这项工作全面凸显了个性化LLMS在边缘侧部署过程中存在的潜在隐私风险。



## **18. Effective faking of verbal deception detection with target-aligned adversarial attacks**

通过目标对准的对抗攻击有效伪造言语欺骗检测 cs.CL

preprint

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.05962v1) [paper-pdf](http://arxiv.org/pdf/2501.05962v1)

**Authors**: Bennett Kleinberg, Riccardo Loconte, Bruno Verschuere

**Abstract**: Background: Deception detection through analysing language is a promising avenue using both human judgments and automated machine learning judgments. For both forms of credibility assessment, automated adversarial attacks that rewrite deceptive statements to appear truthful pose a serious threat. Methods: We used a dataset of 243 truthful and 262 fabricated autobiographical stories in a deception detection task for humans and machine learning models. A large language model was tasked to rewrite deceptive statements so that they appear truthful. In Study 1, humans who made a deception judgment or used the detailedness heuristic and two machine learning models (a fine-tuned language model and a simple n-gram model) judged original or adversarial modifications of deceptive statements. In Study 2, we manipulated the target alignment of the modifications, i.e. tailoring the attack to whether the statements would be assessed by humans or computer models. Results: When adversarial modifications were aligned with their target, human (d=-0.07 and d=-0.04) and machine judgments (51% accuracy) dropped to the chance level. When the attack was not aligned with the target, both human heuristics judgments (d=0.30 and d=0.36) and machine learning predictions (63-78%) were significantly better than chance. Conclusions: Easily accessible language models can effectively help anyone fake deception detection efforts both by humans and machine learning models. Robustness against adversarial modifications for humans and machines depends on that target alignment. We close with suggestions on advancing deception research with adversarial attack designs.

摘要: 背景：通过分析语言进行欺骗检测是一种既使用人类判断又使用自动机器学习判断的有前途的方法。对于这两种形式的可信度评估来说，重写欺骗性陈述以使其看起来真实的自动对抗性攻击构成了严重威胁。方法：我们使用了243个真实的和262个编造的自传故事的数据集，在人类和机器学习模型的欺骗检测任务中。一个大型语言模型的任务是重写欺骗性的陈述，使它们看起来是真实的。在研究1中，做出欺骗性判断或使用细节启发式和两个机器学习模型(微调语言模型和简单n元语法模型)的人判断欺骗性陈述的原始修改或对抗性修改。在研究2中，我们操纵了修改的目标对齐，即根据陈述是否由人或计算机模型评估来量身定做攻击。结果：当对抗性修改与他们的目标一致时，人类(d=-0.07和d=-0.04)和机器判断(51%准确率)下降到机会水平。当攻击与目标不一致时，人类的启发式判断(d=0.30和d=0.36)和机器学习预测(63%-78%)都显著好于机会。结论：易于理解的语言模型可以有效地帮助任何人通过人类和机器学习模型进行虚假的欺骗检测。对人类和机器的敌意修改的健壮性取决于目标对齐。最后，我们建议用对抗性攻击设计来推进欺骗研究。



## **19. Image-based Multimodal Models as Intruders: Transferable Multimodal Attacks on Video-based MLLMs**

作为入侵者的基于图像的多模式模型：对基于视频的MLLM的可转移多模式攻击 cs.CV

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.01042v2) [paper-pdf](http://arxiv.org/pdf/2501.01042v2)

**Authors**: Linhao Huang, Xue Jiang, Zhiqiang Wang, Wentao Mo, Xi Xiao, Bo Han, Yongjie Yin, Feng Zheng

**Abstract**: Video-based multimodal large language models (V-MLLMs) have shown vulnerability to adversarial examples in video-text multimodal tasks. However, the transferability of adversarial videos to unseen models--a common and practical real world scenario--remains unexplored. In this paper, we pioneer an investigation into the transferability of adversarial video samples across V-MLLMs. We find that existing adversarial attack methods face significant limitations when applied in black-box settings for V-MLLMs, which we attribute to the following shortcomings: (1) lacking generalization in perturbing video features, (2) focusing only on sparse key-frames, and (3) failing to integrate multimodal information. To address these limitations and deepen the understanding of V-MLLM vulnerabilities in black-box scenarios, we introduce the Image-to-Video MLLM (I2V-MLLM) attack. In I2V-MLLM, we utilize an image-based multimodal model (IMM) as a surrogate model to craft adversarial video samples. Multimodal interactions and temporal information are integrated to disrupt video representations within the latent space, improving adversarial transferability. In addition, a perturbation propagation technique is introduced to handle different unknown frame sampling strategies. Experimental results demonstrate that our method can generate adversarial examples that exhibit strong transferability across different V-MLLMs on multiple video-text multimodal tasks. Compared to white-box attacks on these models, our black-box attacks (using BLIP-2 as surrogate model) achieve competitive performance, with average attack success rates of 55.48% on MSVD-QA and 58.26% on MSRVTT-QA for VideoQA tasks, respectively. Our code will be released upon acceptance.

摘要: 基于视频的多通道大语言模型(V-MLLM)在视频-文本多通道任务中表现出对敌意例子的脆弱性。然而，对抗性视频是否可以转移到看不见的模型上--这是现实世界中常见和实用的场景--仍未得到探索。在本文中，我们率先对对抗性视频样本在V-MLLMS上的可转移性进行了研究。我们发现，现有的对抗性攻击方法在应用于V-MLLMS的黑盒环境时面临着很大的局限性，我们将其归因于以下缺点：(1)对扰动视频特征缺乏泛化；(2)只关注稀疏关键帧；(3)未能整合多模信息。为了解决这些限制并加深对黑盒场景中V-MLLM漏洞的理解，我们引入了图像到视频MLLM(I2V-MLLM)攻击。在I2V-MLLM中，我们使用基于图像的多模式模型(IMM)作为代理模型来制作对抗性视频样本。多模式交互和时间信息被集成以扰乱潜在空间内的视频表示，提高了对抗性转移。此外，还引入了扰动传播技术来处理不同的未知帧采样策略。实验结果表明，该方法能够在多个视频-文本多模式任务的不同V-MLLMS之间生成具有较强可转移性的对抗性实例。与这些模型上的白盒攻击相比，我们的黑盒攻击(以BLIP-2为代理模型)取得了与之相当的性能，对于视频QA任务，MSVD-QA和MSRVTT-QA的平均攻击成功率分别为55.48%和58.26%。我们的代码将在接受后发布。



## **20. BaThe: Defense against the Jailbreak Attack in Multimodal Large Language Models by Treating Harmful Instruction as Backdoor Trigger**

BaThe：通过将有害指令视为后门触发来防御多模式大型语言模型中的越狱攻击 cs.CR

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2408.09093v2) [paper-pdf](http://arxiv.org/pdf/2408.09093v2)

**Authors**: Yulin Chen, Haoran Li, Yirui Zhang, Zihao Zheng, Yangqiu Song, Bryan Hooi

**Abstract**: Multimodal Large Language Models (MLLMs) have showcased impressive performance in a variety of multimodal tasks. On the other hand, the integration of additional image modality may allow the malicious users to inject harmful content inside the images for jailbreaking. Unlike text-based LLMs, where adversaries need to select discrete tokens to conceal their malicious intent using specific algorithms, the continuous nature of image signals provides a direct opportunity for adversaries to inject harmful intentions. In this work, we propose $\textbf{BaThe}$ ($\textbf{Ba}$ckdoor $\textbf{T}$rigger S$\textbf{h}$i$\textbf{e}$ld), a simple yet effective jailbreak defense mechanism. Our work is motivated by recent research on jailbreak backdoor attack and virtual prompt backdoor attack in generative language models. Jailbreak backdoor attack uses harmful instructions combined with manually crafted strings as triggers to make the backdoored model generate prohibited responses. We assume that harmful instructions can function as triggers, and if we alternatively set rejection responses as the triggered response, the backdoored model then can defend against jailbreak attacks. We achieve this by utilizing virtual rejection prompt, similar to the virtual prompt backdoor attack. We embed the virtual rejection prompt into the soft text embeddings, which we call ``wedge''. Our comprehensive experiments demonstrate that BaThe effectively mitigates various types of jailbreak attacks and is adaptable to defend against unseen attacks, with minimal impact on MLLMs' performance.

摘要: 多通道大型语言模型(MLLM)在各种多通道任务中表现出令人印象深刻的性能。另一方面，附加图像模式的集成可能允许恶意用户在图像中注入有害内容以越狱。与基于文本的LLMS不同，在LLMS中，攻击者需要使用特定的算法选择离散的令牌来隐藏其恶意意图，而图像信号的连续性为攻击者提供了直接注入有害意图的机会。在这项工作中，我们提出了一种简单而有效的越狱防御机制--$\extbf{bathe}$($\extbf{ba}$ck door$\extbf{T}$rigger S$\extbf{h}$i$\extbf{e}$ld)。我们的工作是基于生成式语言模型对越狱后门攻击和虚拟提示后门攻击的最新研究。越狱后门攻击使用有害指令和手动创建的字符串作为触发器，使后门模型生成被禁止的响应。我们假设有害指令可以作为触发器，如果我们将拒绝响应设置为触发响应，那么反向模型就可以防御越狱攻击。我们通过利用虚拟拒绝提示来实现这一点，类似于虚拟提示后门攻击。我们将虚拟拒绝提示嵌入到软文本嵌入中，我们称之为‘’楔形‘’。我们的综合实验表明，BAIT有效地缓解了各种类型的越狱攻击，并且能够自适应地防御看不见的攻击，对MLLMS的性能影响最小。



## **21. Safeguarding System Prompts for LLMs**

LLM的保护系统预算 cs.CR

15 pages, 5 figures, 2 tables

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2412.13426v2) [paper-pdf](http://arxiv.org/pdf/2412.13426v2)

**Authors**: Zhifeng Jiang, Zhihua Jin, Guoliang He

**Abstract**: Large language models (LLMs) are increasingly utilized in applications where system prompts, which guide model outputs, play a crucial role. These prompts often contain business logic and sensitive information, making their protection essential. However, adversarial and even regular user queries can exploit LLM vulnerabilities to expose these hidden prompts. To address this issue, we propose PromptKeeper, a robust defense mechanism designed to safeguard system prompts. PromptKeeper tackles two core challenges: reliably detecting prompt leakage and mitigating side-channel vulnerabilities when leakage occurs. By framing detection as a hypothesis-testing problem, PromptKeeper effectively identifies both explicit and subtle leakage. Upon detection, it regenerates responses using a dummy prompt, ensuring that outputs remain indistinguishable from typical interactions when no leakage is present. PromptKeeper ensures robust protection against prompt extraction attacks via either adversarial or regular queries, while preserving conversational capability and runtime efficiency during benign user interactions.

摘要: 大型语言模型(LLM)越来越多地被用于指导模型输出的系统提示发挥关键作用的应用中。这些提示通常包含业务逻辑和敏感信息，因此保护它们至关重要。但是，敌意的甚至常规的用户查询都可以利用LLM漏洞来暴露这些隐藏的提示。为了解决这个问题，我们提出了PromptKeeper，一种健壮的防御机制，旨在保护系统提示。PromptKeeper解决了两个核心挑战：可靠地检测及时泄漏和在发生泄漏时缓解侧通道漏洞。通过将检测框定为假设检验问题，PromptKeeper有效地识别了显性和细微的泄漏。一旦检测到，它会使用虚拟提示重新生成响应，确保在没有泄漏的情况下，输出与典型的交互没有区别。PromptKeeper通过对抗性或常规查询确保针对提示提取攻击的强大保护，同时在良性用户交互期间保持对话能力和运行效率。



## **22. RAG-WM: An Efficient Black-Box Watermarking Approach for Retrieval-Augmented Generation of Large Language Models**

RAG-WM：一种用于大型语言模型检索增强生成的高效黑箱水印方法 cs.CR

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.05249v1) [paper-pdf](http://arxiv.org/pdf/2501.05249v1)

**Authors**: Peizhuo Lv, Mengjie Sun, Hao Wang, Xiaofeng Wang, Shengzhi Zhang, Yuxuan Chen, Kai Chen, Limin Sun

**Abstract**: In recent years, tremendous success has been witnessed in Retrieval-Augmented Generation (RAG), widely used to enhance Large Language Models (LLMs) in domain-specific, knowledge-intensive, and privacy-sensitive tasks. However, attackers may steal those valuable RAGs and deploy or commercialize them, making it essential to detect Intellectual Property (IP) infringement. Most existing ownership protection solutions, such as watermarks, are designed for relational databases and texts. They cannot be directly applied to RAGs because relational database watermarks require white-box access to detect IP infringement, which is unrealistic for the knowledge base in RAGs. Meanwhile, post-processing by the adversary's deployed LLMs typically destructs text watermark information. To address those problems, we propose a novel black-box "knowledge watermark" approach, named RAG-WM, to detect IP infringement of RAGs. RAG-WM uses a multi-LLM interaction framework, comprising a Watermark Generator, Shadow LLM & RAG, and Watermark Discriminator, to create watermark texts based on watermark entity-relationship tuples and inject them into the target RAG. We evaluate RAG-WM across three domain-specific and two privacy-sensitive tasks on four benchmark LLMs. Experimental results show that RAG-WM effectively detects the stolen RAGs in various deployed LLMs. Furthermore, RAG-WM is robust against paraphrasing, unrelated content removal, knowledge insertion, and knowledge expansion attacks. Lastly, RAG-WM can also evade watermark detection approaches, highlighting its promising application in detecting IP infringement of RAG systems.

摘要: 近年来，检索增强生成(RAG)取得了巨大的成功，它被广泛用于增强领域特定、知识密集型和隐私敏感任务中的大型语言模型(LLM)。然而，攻击者可能会窃取这些有价值的破布并将其部署或商业化，这使得检测侵犯知识产权(IP)变得至关重要。大多数现有的所有权保护解决方案，如水印，都是为关系数据库和文本设计的。它们不能直接应用于RAG，因为关系数据库水印需要白盒访问来检测知识产权侵权，这对于RAG中的知识库来说是不现实的。同时，由对手部署的LLMS进行的后处理通常会破坏文本水印信息。针对这些问题，我们提出了一种新的黑盒“知识水印”方法RAG-WM来检测RAG的知识产权侵权行为。RAG-WM使用多LLM交互框架，包括水印生成器、阴影LLM和RAG和水印鉴别器，基于水印实体关系元组创建水印文本并将其注入目标RAG。我们在四个基准LLM上对RAG-WM进行了评估，测试了三个领域特定的任务和两个隐私敏感任务。实验结果表明，RAG-WM能够有效地检测出各种部署的LLM中被盗的RAG。此外，RAG-WM对释义、无关内容移除、知识插入和知识扩展攻击具有较强的鲁棒性。最后，RAG-WM还可以避开水印检测方法，在检测RAG系统的知识产权侵权行为方面具有广阔的应用前景。



## **23. Turning Logic Against Itself : Probing Model Defenses Through Contrastive Questions**

将逻辑与自身对立：通过对比问题探索模型辩护 cs.CL

Our code is publicly available at  https://github.com/UKPLab/POATE-attack

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.01872v2) [paper-pdf](http://arxiv.org/pdf/2501.01872v2)

**Authors**: Rachneet Sachdeva, Rima Hazra, Iryna Gurevych

**Abstract**: Large language models, despite extensive alignment with human values and ethical principles, remain vulnerable to sophisticated jailbreak attacks that exploit their reasoning abilities. Existing safety measures often detect overt malicious intent but fail to address subtle, reasoning-driven vulnerabilities. In this work, we introduce POATE (Polar Opposite query generation, Adversarial Template construction, and Elaboration), a novel jailbreak technique that harnesses contrastive reasoning to provoke unethical responses. POATE crafts semantically opposing intents and integrates them with adversarial templates, steering models toward harmful outputs with remarkable subtlety. We conduct extensive evaluation across six diverse language model families of varying parameter sizes to demonstrate the robustness of the attack, achieving significantly higher attack success rates (~44%) compared to existing methods. To counter this, we propose Intent-Aware CoT and Reverse Thinking CoT, which decompose queries to detect malicious intent and reason in reverse to evaluate and reject harmful responses. These methods enhance reasoning robustness and strengthen the model's defense against adversarial exploits.

摘要: 大型语言模型尽管与人类价值观和伦理原则广泛一致，但仍然容易受到复杂的越狱攻击，这些攻击利用了它们的推理能力。现有的安全措施经常检测到公开的恶意意图，但无法解决细微的、推理驱动的漏洞。在这项工作中，我们介绍了POATE(极地相反查询生成，对抗性模板构建和精化)，这是一种新的越狱技术，利用对比推理来引发不道德的反应。波特在语义上设计了相反的意图，并将它们与对抗性模板整合在一起，以惊人的微妙程度引导模型指向有害的输出。我们对六个不同参数大小的不同语言模型家族进行了广泛的评估，以证明攻击的健壮性，与现有方法相比，攻击成功率显著提高(~44%)。针对这一问题，我们提出了意图感知COT和逆向思维COT，它们分解查询以检测恶意意图，并反向推理以评估和拒绝有害响应。这些方法增强了推理的健壮性，增强了模型对对手攻击的防御能力。



## **24. Trading Devil RL: Backdoor attack via Stock market, Bayesian Optimization and Reinforcement Learning**

交易魔鬼RL：通过股市、Bayesian优化和强化学习进行后门攻击 cs.LG

End of data poisoning research!: Navier-stokes equations (3D;  update); Reinforcement Learning (RL); HFT (High Frequency Trading); Limit  Order Markets and backdoor attack detection

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2412.17908v2) [paper-pdf](http://arxiv.org/pdf/2412.17908v2)

**Authors**: Orson Mengara

**Abstract**: With the rapid development of generative artificial intelligence, particularly large language models, a number of sub-fields of deep learning have made significant progress and are now very useful in everyday applications. For example, well-known financial institutions simulate a wide range of scenarios for various models created by their research teams using reinforcement learning, both before production and after regular operations. In this work, we propose a backdoor attack that focuses solely on data poisoning. This particular backdoor attack is classified as an attack without prior consideration or trigger, and we name it FinanceLLMsBackRL. Our aim is to examine the potential effects of large language models that use reinforcement learning systems for text production or speech recognition, finance, physics, or the ecosystem of contemporary artificial intelligence models.

摘要: 随着生成式人工智能，特别是大型语言模型的快速发展，深度学习的许多子领域取得了重大进展，现在在日常应用中非常有用。例如，知名金融机构在生产之前和常规运营之后使用强化学习为其研究团队创建的各种模型模拟各种场景。在这项工作中，我们提出了一种仅针对数据中毒的后门攻击。这种特殊的后门攻击被归类为未经事先考虑或触发的攻击，我们将其命名为Financial LLMsBackRL。我们的目标是检查使用强化学习系统进行文本生成或语音识别、金融、物理或当代人工智能模型生态系统的大型语言模型的潜在影响。



## **25. SpaLLM-Guard: Pairing SMS Spam Detection Using Open-source and Commercial LLMs**

SpaLLM-Guard：使用开源和商业LLM配对短信垃圾邮件检测 cs.CR

17 pages

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.04985v1) [paper-pdf](http://arxiv.org/pdf/2501.04985v1)

**Authors**: Muhammad Salman, Muhammad Ikram, Nardine Basta, Mohamed Ali Kaafar

**Abstract**: The increasing threat of SMS spam, driven by evolving adversarial techniques and concept drift, calls for more robust and adaptive detection methods. In this paper, we evaluate the potential of large language models (LLMs), both open-source and commercial, for SMS spam detection, comparing their performance across zero-shot, few-shot, fine-tuning, and chain-of-thought prompting approaches. Using a comprehensive dataset of SMS messages, we assess the spam detection capabilities of prominent LLMs such as GPT-4, DeepSeek, LLAMA-2, and Mixtral. Our findings reveal that while zero-shot learning provides convenience, it is unreliable for effective spam detection. Few-shot learning, particularly with carefully selected examples, improves detection but exhibits variability across models. Fine-tuning emerges as the most effective strategy, with Mixtral achieving 98.6% accuracy and a balanced false positive and false negative rate below 2%, meeting the criteria for robust spam detection. Furthermore, we explore the resilience of these models to adversarial attacks, finding that fine-tuning significantly enhances robustness against both perceptible and imperceptible manipulations. Lastly, we investigate the impact of concept drift and demonstrate that fine-tuned LLMs, especially when combined with few-shot learning, can mitigate its effects, maintaining high performance even on evolving spam datasets. This study highlights the importance of fine-tuning and tailored learning strategies to deploy LLMs effectively for real-world SMS spam detection

摘要: 在不断发展的敌意技术和概念漂移的推动下，垃圾短信的威胁越来越大，这要求更健壮和自适应的检测方法。在本文中，我们评估了开源和商业的大型语言模型(LLM)在短信垃圾邮件检测中的潜力，比较了它们在零射、少射、微调和思维链提示方法中的性能。使用全面的短信数据集，我们评估了GPT-4、DeepSeek、Llama-2和Mixtral等知名LLMS的垃圾邮件检测能力。我们的发现表明，虽然零机会学习提供了便利，但对于有效的垃圾邮件检测来说，它是不可靠的。少发式学习，尤其是精心挑选的例子，提高了检测能力，但在不同模型之间表现出多样性。微调成为最有效的策略，Mixtral达到了98.6%的准确率，假阳性率和假阴性率平衡在2%以下，满足了稳健的垃圾邮件检测标准。此外，我们探讨了这些模型对敌意攻击的弹性，发现微调显著增强了对可感知和不可感知操纵的稳健性。最后，我们研究了概念漂移的影响，并证明了微调的LLMS，特别是当与少镜头学习相结合时，可以缓解其影响，即使在不断演变的垃圾邮件数据集上也保持了高性能。这项研究强调了微调和量身定制的学习策略的重要性，以有效地部署LLMS来检测真实世界的短信垃圾邮件



## **26. Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency**

通过洗牌不一致性破解多模式大型语言模型 cs.CR

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.04931v1) [paper-pdf](http://arxiv.org/pdf/2501.04931v1)

**Authors**: Shiji Zhao, Ranjie Duan, Fengxiang Wang, Chi Chen, Caixin Kang, Jialing Tao, YueFeng Chen, Hui Xue, Xingxing Wei

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved impressive performance and have been put into practical use in commercial applications, but they still have potential safety mechanism vulnerabilities. Jailbreak attacks are red teaming methods that aim to bypass safety mechanisms and discover MLLMs' potential risks. Existing MLLMs' jailbreak methods often bypass the model's safety mechanism through complex optimization methods or carefully designed image and text prompts. Despite achieving some progress, they have a low attack success rate on commercial closed-source MLLMs. Unlike previous research, we empirically find that there exists a Shuffle Inconsistency between MLLMs' comprehension ability and safety ability for the shuffled harmful instruction. That is, from the perspective of comprehension ability, MLLMs can understand the shuffled harmful text-image instructions well. However, they can be easily bypassed by the shuffled harmful instructions from the perspective of safety ability, leading to harmful responses. Then we innovatively propose a text-image jailbreak attack named SI-Attack. Specifically, to fully utilize the Shuffle Inconsistency and overcome the shuffle randomness, we apply a query-based black-box optimization method to select the most harmful shuffled inputs based on the feedback of the toxic judge model. A series of experiments show that SI-Attack can improve the attack's performance on three benchmarks. In particular, SI-Attack can obviously improve the attack success rate for commercial MLLMs such as GPT-4o or Claude-3.5-Sonnet.

摘要: 多通道大语言模型已经取得了令人印象深刻的性能，并在商业应用中得到了实际应用，但它们仍然存在潜在的安全机制漏洞。越狱攻击是一种旨在绕过安全机制并发现MLLMS潜在风险的红色团队方法。现有的MLLMS越狱方法往往通过复杂的优化方法或精心设计的图像和文本提示绕过模型的安全机制。尽管取得了一些进展，但他们对商业闭源MLLMS的攻击成功率很低。与前人的研究不同，我们的实证研究发现，多语言传播者对有害指令的理解能力和安全能力之间存在着洗牌不一致的现象。也就是说，从理解能力的角度来看，MLLMS能够很好地理解混洗后的有害文本-图像指令。然而，从安全能力的角度来看，它们很容易被洗牌后的有害指令绕过，导致有害反应。在此基础上，我们创新性地提出了一种文本图像越狱攻击方案SI-Attack。具体地说，为了充分利用混洗的不一致性和克服混洗的随机性，我们应用了一种基于查询的黑盒优化方法，根据有毒判断模型的反馈选择最有害的混洗输入。一系列的实验表明，SI-攻击在三个基准上都能提高攻击的性能。特别是对于GPT-40或Claude-3.5-Sonnet等商用MLLMS，SI-Attack能明显提高攻击成功率。



## **27. Navigating the Designs of Privacy-Preserving Fine-tuning for Large Language Models**

引导大型语言模型的隐私保护微调设计 cs.LG

4 pages, 2 figures

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.04323v2) [paper-pdf](http://arxiv.org/pdf/2501.04323v2)

**Authors**: Haonan Shi, Tu Ouyang, An Wang

**Abstract**: Instruction tuning has proven effective in enhancing Large Language Models' (LLMs) performance on downstream tasks. However, real-world fine-tuning faces inherent conflicts between model providers' intellectual property protection, clients' data privacy requirements, and tuning costs. While recent approaches like split learning and offsite tuning demonstrate promising architectures for privacy-preserving fine-tuning, there is a gap in systematically addressing the multidimensional trade-offs required for diverse real-world deployments. We propose several indicative evaluation metrics to guide design trade-offs for privacy-preserving fine-tuning and a series of example designs, collectively named GuardedTuning; they result from novel combinations of system architectures with adapted privacy-enhancement methods and emerging computation techniques. Each design represents distinct trade-offs across model utility, privacy guarantees, and costs. Experimental results demonstrate that these designs protect against data reconstruction attacks while maintaining competitive fine-tuning performance.

摘要: 指令调优在提高大型语言模型(LLM)在下游任务上的性能方面已被证明是有效的。然而，现实世界的微调面临着模型提供商的知识产权保护、客户的数据隐私要求和调整成本之间的内在冲突。虽然最近的方法，如拆分学习和异地调整，展示了保护隐私的微调的有前途的架构，但在系统地解决不同现实世界部署所需的多维权衡方面存在差距。我们提出了几个指示性的评估指标来指导隐私保护微调和一系列示例设计的权衡，统称为GuardedTuning；它们是系统架构与适应隐私增强方法和新兴计算技术的新颖组合的结果。每一种设计都代表着在模型效用、隐私保障和成本方面的不同权衡。实验结果表明，这些设计在保持具有竞争力的微调性能的同时，能够抵御数据重构攻击。



## **28. Watch Out for Your Guidance on Generation! Exploring Conditional Backdoor Attacks against Large Language Models**

留意您对世代的指导！探索针对大型语言模型的条件后门攻击 cs.CL

The paper has been accepted to AAAI 2025

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2404.14795v5) [paper-pdf](http://arxiv.org/pdf/2404.14795v5)

**Authors**: Jiaming He, Wenbo Jiang, Guanyu Hou, Wenshu Fan, Rui Zhang, Hongwei Li

**Abstract**: Mainstream backdoor attacks on large language models (LLMs) typically set a fixed trigger in the input instance and specific responses for triggered queries. However, the fixed trigger setting (e.g., unusual words) may be easily detected by human detection, limiting the effectiveness and practicality in real-world scenarios. To enhance the stealthiness of backdoor activation, we present a new poisoning paradigm against LLMs triggered by specifying generation conditions, which are commonly adopted strategies by users during model inference. The poisoned model performs normally for output under normal/other generation conditions, while becomes harmful for output under target generation conditions. To achieve this objective, we introduce BrieFool, an efficient attack framework. It leverages the characteristics of generation conditions by efficient instruction sampling and poisoning data generation, thereby influencing the behavior of LLMs under target conditions. Our attack can be generally divided into two types with different targets: Safety unalignment attack and Ability degradation attack. Our extensive experiments demonstrate that BrieFool is effective across safety domains and ability domains, achieving higher success rates than baseline methods, with 94.3 % on GPT-3.5-turbo

摘要: 针对大型语言模型(LLM)的主流后门攻击通常会在输入实例中设置固定的触发器，并为触发的查询设置特定的响应。然而，固定的触发设置(例如，不寻常的单词)可能很容易被人类检测到，从而限制了在现实世界场景中的有效性和实用性。为了增强后门激活的隐蔽性，我们提出了一种新的针对通过指定生成条件触发的LLM的中毒范例，这些策略是用户在模型推理中经常采用的策略。中毒模型在正常/其他发电条件下的出力表现正常，而在目标发电条件下的出力变得有害。为了实现这一目标，我们引入了一种高效的攻击框架BrieFool。它通过高效的指令采样和中毒数据生成来利用生成条件的特征，从而影响目标条件下的LLM的行为。我们的攻击一般可以分为两种类型，针对不同的目标：安全联盟攻击和能力退化攻击。我们的广泛实验表明，BrieFool跨安全域和能量域是有效的，获得了比基准方法更高的成功率，在GPT-3.5-Turbo上的成功率为94.3%



## **29. Not all tokens are created equal: Perplexity Attention Weighted Networks for AI generated text detection**

并非所有令牌都是平等的：用于人工智能生成文本检测的困惑注意力加权网络 cs.CL

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2501.03940v1) [paper-pdf](http://arxiv.org/pdf/2501.03940v1)

**Authors**: Pablo Miralles-González, Javier Huertas-Tato, Alejandro Martín, David Camacho

**Abstract**: The rapid advancement in large language models (LLMs) has significantly enhanced their ability to generate coherent and contextually relevant text, raising concerns about the misuse of AI-generated content and making it critical to detect it. However, the task remains challenging, particularly in unseen domains or with unfamiliar LLMs. Leveraging LLM next-token distribution outputs offers a theoretically appealing approach for detection, as they encapsulate insights from the models' extensive pre-training on diverse corpora. Despite its promise, zero-shot methods that attempt to operationalize these outputs have met with limited success. We hypothesize that one of the problems is that they use the mean to aggregate next-token distribution metrics across tokens, when some tokens are naturally easier or harder to predict and should be weighted differently. Based on this idea, we propose the Perplexity Attention Weighted Network (PAWN), which uses the last hidden states of the LLM and positions to weight the sum of a series of features based on metrics from the next-token distribution across the sequence length. Although not zero-shot, our method allows us to cache the last hidden states and next-token distribution metrics on disk, greatly reducing the training resource requirements. PAWN shows competitive and even better performance in-distribution than the strongest baselines (fine-tuned LMs) with a fraction of their trainable parameters. Our model also generalizes better to unseen domains and source models, with smaller variability in the decision boundary across distribution shifts. It is also more robust to adversarial attacks, and if the backbone has multilingual capabilities, it presents decent generalization to languages not seen during supervised training, with LLaMA3-1B reaching a mean macro-averaged F1 score of 81.46% in cross-validation with nine languages.

摘要: 大型语言模型(LLM)的快速发展极大地增强了它们生成连贯和上下文相关文本的能力，这引起了人们对滥用人工智能生成的内容的担忧，并使检测它变得至关重要。然而，这项任务仍然具有挑战性，特别是在看不见的领域或具有不熟悉的LLM的领域。利用LLM下一个令牌分发输出提供了一种理论上有吸引力的检测方法，因为它们概括了模型对不同语料库的广泛预培训的见解。尽管有希望，但试图将这些产出付诸实施的零射击方法却取得了有限的成功。我们假设其中一个问题是，当一些令牌自然更容易或更难预测，并且应该以不同的权重进行加权时，它们使用平均值来聚合跨令牌的下一令牌分发度量。基于这一思想，我们提出了困惑注意力加权网络(PAWN)，它利用LLM的最后一个隐藏状态和位置来加权一系列特征的和，基于下一个令牌分布在整个序列长度上的度量。虽然不是零命中率，但我们的方法允许我们在磁盘上缓存最后的隐藏状态和下一个令牌分布度量，大大减少了训练资源需求。与最强的基线(微调LMS)相比，PAWN显示出具有竞争力的分布性能，甚至比它们的可训练参数的一小部分更好。我们的模型也更好地推广到看不见的域和源模型，跨分布转变的决策边界的可变性较小。LLaMA3-1B在与9种语言的交叉验证中达到了81.46%的平均宏观平均F1分数。



## **30. PhishAgent: A Robust Multimodal Agent for Phishing Webpage Detection**

PhishAgent：一种用于网络钓鱼网页检测的鲁棒多模式代理 cs.CR

Accepted at AAAI 2025

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2408.10738v2) [paper-pdf](http://arxiv.org/pdf/2408.10738v2)

**Authors**: Tri Cao, Chengyu Huang, Yuexin Li, Huilin Wang, Amy He, Nay Oo, Bryan Hooi

**Abstract**: Phishing attacks are a major threat to online security, exploiting user vulnerabilities to steal sensitive information. Various methods have been developed to counteract phishing, each with varying levels of accuracy, but they also face notable limitations. In this study, we introduce PhishAgent, a multimodal agent that combines a wide range of tools, integrating both online and offline knowledge bases with Multimodal Large Language Models (MLLMs). This combination leads to broader brand coverage, which enhances brand recognition and recall. Furthermore, we propose a multimodal information retrieval framework designed to extract the relevant top k items from offline knowledge bases, using available information from a webpage, including logos and HTML. Our empirical results, based on three real-world datasets, demonstrate that the proposed framework significantly enhances detection accuracy and reduces both false positives and false negatives, while maintaining model efficiency. Additionally, PhishAgent shows strong resilience against various types of adversarial attacks.

摘要: 网络钓鱼攻击是在线安全的主要威胁，利用用户漏洞窃取敏感信息。已经开发了各种方法来对抗网络钓鱼，每一种方法的精确度都不同，但它们也面临着显著的局限性。在本研究中，我们介绍了PhishAgent，一个结合了广泛工具的多通道代理，将线上和线下知识库与多通道大语言模型(MLLMS)相结合。这一组合导致了更广泛的品牌覆盖，从而提高了品牌认知度和召回率。此外，我们还提出了一个多通道信息检索框架，该框架利用网页中的可用信息，包括标识和超文本标记语言，从离线知识库中提取相关的前k个条目。基于三个真实数据集的实验结果表明，该框架在保持模型效率的同时，显著提高了检测准确率，减少了误报和漏报。此外，PhishAgent对各种类型的对抗性攻击表现出很强的韧性。



## **31. MRJ-Agent: An Effective Jailbreak Agent for Multi-Round Dialogue**

MRJ-Agent：多轮对话的有效越狱代理 cs.AI

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2411.03814v2) [paper-pdf](http://arxiv.org/pdf/2411.03814v2)

**Authors**: Fengxiang Wang, Ranjie Duan, Peng Xiao, Xiaojun Jia, Shiji Zhao, Cheng Wei, YueFeng Chen, Chongwen Wang, Jialing Tao, Hang Su, Jun Zhu, Hui Xue

**Abstract**: Large Language Models (LLMs) demonstrate outstanding performance in their reservoir of knowledge and understanding capabilities, but they have also been shown to be prone to illegal or unethical reactions when subjected to jailbreak attacks. To ensure their responsible deployment in critical applications, it is crucial to understand the safety capabilities and vulnerabilities of LLMs. Previous works mainly focus on jailbreak in single-round dialogue, overlooking the potential jailbreak risks in multi-round dialogues, which are a vital way humans interact with and extract information from LLMs. Some studies have increasingly concentrated on the risks associated with jailbreak in multi-round dialogues. These efforts typically involve the use of manually crafted templates or prompt engineering techniques. However, due to the inherent complexity of multi-round dialogues, their jailbreak performance is limited. To solve this problem, we propose a novel multi-round dialogue jailbreaking agent, emphasizing the importance of stealthiness in identifying and mitigating potential threats to human values posed by LLMs. We propose a risk decomposition strategy that distributes risks across multiple rounds of queries and utilizes psychological strategies to enhance attack strength. Extensive experiments show that our proposed method surpasses other attack methods and achieves state-of-the-art attack success rate. We will make the corresponding code and dataset available for future research. The code will be released soon.

摘要: 大型语言模型(LLM)在其知识和理解能力方面表现出色，但也被证明在受到越狱攻击时容易出现非法或不道德的反应。为了确保它们在关键应用中负责任地部署，了解LLMS的安全能力和漏洞至关重要。以往的研究主要集中在单轮对话中的越狱，而忽略了多轮对话中潜在的越狱风险，而多轮对话是人类与小武器系统交互和提取信息的重要方式。一些研究越来越集中于多轮对话中越狱的相关风险。这些工作通常涉及使用手工制作的模板或即时工程技术。然而，由于多轮对话的内在复杂性，它们的越狱表现有限。为了解决这一问题，我们提出了一种新的多轮对话越狱代理，强调了隐蔽性在识别和缓解LLMS对人类价值构成的潜在威胁方面的重要性。我们提出了一种风险分解策略，将风险分布在多轮查询中，并利用心理策略来增强攻击强度。大量实验表明，该方法优于其他攻击方法，达到了最高的攻击成功率。我们将提供相应的代码和数据集，供未来研究使用。代码很快就会发布。



## **32. Practical Secure Inference Algorithm for Fine-tuned Large Language Model Based on Fully Homomorphic Encryption**

基于全同形加密的微调大语言模型的实用安全推理算法 cs.CR

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2501.01672v2) [paper-pdf](http://arxiv.org/pdf/2501.01672v2)

**Authors**: Zhang Ruoyan, Zheng Zhongxiang, Bao Wankang

**Abstract**: Large language models(LLMs) are currently at the forefront of the machine learning field, which show a broad application prospect but at the same time expose some risks of privacy leakage. We combined Fully Homomorphic Encryption(FHE) and provable security theory with Parameter-Efficient Fine-Tuning(PEFT) to propose an efficient and secure inference scheme for LLMs. More specially, we focus on pre-trained LLMs which rely on open-sourced base model and then fine-tuned with the private datasets by LoRA. This is a popular road-map for Vertical Domain Models such as LawGPT and BenTsao. We use two key technologies below. Firstly, we divide the whole model into the public part and the private part. The weights of public part are publicly accessible(e.g. the open-sourced base model) while the private part needs to be protected(e.g. the LoRA matrices). In this way, the overhead brought by computing on private data can be greatly reduced. Secondly, we propose a general method to transform a linear layer into another one which provides security against model extraction attacks and preserves its original functionality, which denoted as Private Linear Layer(PLL). Then we use this method on the LoRA matrices to make sure that the server protects their private weights without restricting the user's input. We also show that the difficulty of performing model extraction attacks for PLL can be reduced to the well-known hard problem Learning with Errors(LWE). Combing this method with FHE, we can protect user's input at the same time. In this paper, we use the open-source model ChatGLM2-6B as the base model which is fine-tuned by LoRA. Experimental results show the inference efficiency of our scheme reaches 1.61s/token which displays that the scheme has good practicality.

摘要: 大语言模型目前处于机器学习领域的前沿，在显示出广阔的应用前景的同时，也暴露出一些隐私泄露的风险。将完全同态加密(FHE)和可证明安全理论与参数高效精调(PEFT)相结合，提出了一种高效安全的LLMS推理方案。更具体地说，我们专注于预先训练的LLM，它依赖于开源的基本模型，然后使用LORA的私有数据集进行微调。这是LawGPT和BenTsao等垂直领域模式的流行路线图。我们使用以下两项关键技术。首先，我们将整个模型分为公共部分和私人部分。公共部分的权重是可公开访问的(例如，开放源码的基本模型)，而私有部分需要保护(例如，LORA矩阵)。通过这种方式，可以大大降低计算私有数据带来的开销。其次，我们提出了一种将一个线性层转换为另一个线性层的通用方法，该方法既能保证模型提取攻击的安全性，又能保持原有的功能，称为专用线性层。然后，我们对Lora矩阵使用此方法，以确保服务器在不限制用户输入的情况下保护它们的私有权重。我们还证明了对PLL进行模型提取攻击的难度可以归结为众所周知的带错误学习(LWE)的困难问题。将该方法与FHE相结合，可以同时保护用户的输入。在本文中，我们使用开源模型ChatGLM2-6B作为基础模型，该模型由LORA进行了微调。实验结果表明，该方案的推理效率达到1.61S/Token，具有较好的实用性。



## **33. ChatBug: A Common Vulnerability of Aligned LLMs Induced by Chat Templates**

ChatBug：聊天模板引发的对齐LLM的常见漏洞 cs.CR

This paper is accepted to AAAI 2025

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2406.12935v2) [paper-pdf](http://arxiv.org/pdf/2406.12935v2)

**Authors**: Fengqing Jiang, Zhangchen Xu, Luyao Niu, Bill Yuchen Lin, Radha Poovendran

**Abstract**: Large language models (LLMs) are expected to follow instructions from users and engage in conversations. Techniques to enhance LLMs' instruction-following capabilities typically fine-tune them using data structured according to a predefined chat template. Although chat templates are shown to be effective in optimizing LLM performance, their impact on safety alignment of LLMs has been less understood, which is crucial for deploying LLMs safely at scale.   In this paper, we investigate how chat templates affect safety alignment of LLMs. We identify a common vulnerability, named ChatBug, that is introduced by chat templates. Our key insight to identify ChatBug is that the chat templates provide a rigid format that need to be followed by LLMs, but not by users. Hence, a malicious user may not necessarily follow the chat template when prompting LLMs. Instead, malicious users could leverage their knowledge of the chat template and accordingly craft their prompts to bypass safety alignments of LLMs. We develop two attacks to exploit the ChatBug vulnerability. We demonstrate that a malicious user can exploit the ChatBug vulnerability of eight state-of-the-art (SOTA) LLMs and effectively elicit unintended responses from these models. Moreover, we show that ChatBug can be exploited by existing jailbreak attacks to enhance their attack success rates. We investigate potential countermeasures to ChatBug. Our results show that while adversarial training effectively mitigates the ChatBug vulnerability, the victim model incurs significant performance degradation. These results highlight the trade-off between safety alignment and helpfulness. Developing new methods for instruction tuning to balance this trade-off is an open and critical direction for future research

摘要: 大型语言模型(LLM)应该遵循用户的指示并参与对话。增强LLMS的指令遵循能力的技术通常使用根据预定义的聊天模板构造的数据对其进行微调。尽管聊天模板被证明在优化LLM性能方面是有效的，但人们对它们对LLM安全调整的影响知之甚少，这对于安全地大规模部署LLMS至关重要。在本文中，我们研究了聊天模板如何影响LLMS的安全对齐。我们发现了聊天模板引入的一个名为ChatBug的常见漏洞。我们识别ChatBug的关键洞察力是，聊天模板提供了一种严格的格式，需要LLMS遵循，而不是用户。因此，恶意用户在提示LLMS时可能不一定遵循聊天模板。相反，恶意用户可以利用他们对聊天模板的了解，并相应地精心编制他们的提示，以绕过LLMS的安全对齐。我们开发了两个攻击来利用ChatBug漏洞。我们演示了恶意用户可以利用8个最先进的(SOTA)LLM的ChatBug漏洞，并有效地从这些模型中引发意外响应。此外，我们发现ChatBug可以被现有的越狱攻击所利用，以提高他们的攻击成功率。我们调查了针对ChatBug的潜在对策。我们的结果表明，虽然对抗性训练有效地缓解了ChatBug漏洞，但受害者模型导致了显著的性能下降。这些结果突显了安全性调整和帮助之间的权衡。开发新的教学调整方法来平衡这种权衡是未来研究的一个开放和关键的方向



## **34. HuRef: HUman-REadable Fingerprint for Large Language Models**

HuRef：大型语言模型的人类可读取指纹 cs.CL

NeurIPS 2024

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2312.04828v5) [paper-pdf](http://arxiv.org/pdf/2312.04828v5)

**Authors**: Boyi Zeng, Lizheng Wang, Yuncong Hu, Yi Xu, Chenghu Zhou, Xinbing Wang, Yu Yu, Zhouhan Lin

**Abstract**: Protecting the copyright of large language models (LLMs) has become crucial due to their resource-intensive training and accompanying carefully designed licenses. However, identifying the original base model of an LLM is challenging due to potential parameter alterations. In this study, we introduce HuRef, a human-readable fingerprint for LLMs that uniquely identifies the base model without interfering with training or exposing model parameters to the public. We first observe that the vector direction of LLM parameters remains stable after the model has converged during pretraining, with negligible perturbations through subsequent training steps, including continued pretraining, supervised fine-tuning, and RLHF, which makes it a sufficient condition to identify the base model. The necessity is validated by continuing to train an LLM with an extra term to drive away the model parameters' direction and the model becomes damaged. However, this direction is vulnerable to simple attacks like dimension permutation or matrix rotation, which significantly change it without affecting performance. To address this, leveraging the Transformer structure, we systematically analyze potential attacks and define three invariant terms that identify an LLM's base model. Due to the potential risk of information leakage, we cannot publish invariant terms directly. Instead, we map them to a Gaussian vector using an encoder, then convert it into a natural image using StyleGAN2, and finally publish the image. In our black-box setting, all fingerprinting steps are internally conducted by the LLMs owners. To ensure the published fingerprints are honestly generated, we introduced Zero-Knowledge Proof (ZKP). Experimental results across various LLMs demonstrate the effectiveness of our method. The code is available at https://github.com/LUMIA-Group/HuRef.

摘要: 保护大型语言模型(LLM)的版权已变得至关重要，因为它们需要进行资源密集型培训，并附带精心设计的许可证。然而，由于潜在的参数变化，识别LLM的原始基础模型是具有挑战性的。在这项研究中，我们引入了HuRef，这是一种用于LLMS的人类可读指纹，它在不干扰训练或向公众暴露模型参数的情况下唯一地识别基本模型。我们首先观察到，在预训练过程中模型收敛后，LLM参数的向量方向保持稳定，通过后续的训练步骤，包括继续预训练、有监督的微调和RLHF，可以忽略不计的扰动，这使得它成为识别基本模型的充分条件。通过继续训练一个带有额外项的LLM来驱离模型参数的方向，从而使模型受损，从而验证了这种必要性。然而，这个方向很容易受到维度置换或矩阵旋转等简单攻击，这些攻击会在不影响性能的情况下显著改变它。为了解决这个问题，利用Transformer结构，我们系统地分析了潜在的攻击，并定义了识别LLM基本模型的三个不变术语。由于潜在的信息泄露风险，我们不能直接发布不变项。相反，我们使用编码器将它们映射到高斯向量，然后使用StyleGAN2将其转换为自然图像，最后发布图像。在我们的黑盒设置中，所有指纹识别步骤都由LLMS所有者在内部执行。为了确保公布的指纹是真实生成的，我们引入了零知识证明(ZKP)。在不同LLM上的实验结果证明了该方法的有效性。代码可在https://github.com/LUMIA-Group/HuRef.上获得



## **35. Adversarial Vulnerabilities in Large Language Models for Time Series Forecasting**

时间序列预测大型语言模型中的对抗漏洞 cs.LG

11 pages, 5 figures

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2412.08099v2) [paper-pdf](http://arxiv.org/pdf/2412.08099v2)

**Authors**: Fuqiang Liu, Sicong Jiang, Luis Miranda-Moreno, Seongjin Choi, Lijun Sun

**Abstract**: Large Language Models (LLMs) have recently demonstrated significant potential in the field of time series forecasting, offering impressive capabilities in handling complex temporal data. However, their robustness and reliability in real-world applications remain under-explored, particularly concerning their susceptibility to adversarial attacks. In this paper, we introduce a targeted adversarial attack framework for LLM-based time series forecasting. By employing both gradient-free and black-box optimization methods, we generate minimal yet highly effective perturbations that significantly degrade the forecasting accuracy across multiple datasets and LLM architectures. Our experiments, which include models like TimeGPT and LLM-Time with GPT-3.5, GPT-4, LLaMa, and Mistral, show that adversarial attacks lead to much more severe performance degradation than random noise, and demonstrate the broad effectiveness of our attacks across different LLMs. The results underscore the critical vulnerabilities of LLMs in time series forecasting, highlighting the need for robust defense mechanisms to ensure their reliable deployment in practical applications.

摘要: 大型语言模型最近在时间序列预测领域显示出巨大的潜力，在处理复杂的时间数据方面提供了令人印象深刻的能力。然而，它们在实际应用中的健壮性和可靠性仍然没有得到充分的研究，特别是关于它们对对手攻击的敏感性。本文提出了一种基于LLM的时间序列预测的对抗性攻击框架。通过使用无梯度和黑盒优化方法，我们产生了最小但高效的扰动，这些扰动显著降低了跨多个数据集和LLM体系结构的预测精度。我们的实验，包括使用GPT-3.5、GPT-4、LLAMA和Mistral的TimeGPT和LLM-Time模型，表明对抗性攻击导致的性能降级比随机噪声严重得多，并证明了我们的攻击在不同LLM上的广泛有效性。这些结果强调了低层管理在时间序列预测中的关键弱点，强调了需要强大的防御机制来确保其在实际应用中的可靠部署。



## **36. Pathway to Secure and Trustworthy ZSM for LLMs: Attacks, Defense, and Opportunities**

为LLM提供安全和值得信赖的ZZ之路：攻击、防御和机会 cs.CR

7 pages, 4 figures

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2408.00722v2) [paper-pdf](http://arxiv.org/pdf/2408.00722v2)

**Authors**: Sunder Ali Khowaja, Parus Khuwaja, Kapal Dev, Hussam Al Hamadi, Engin Zeydan

**Abstract**: Recently, large language models (LLMs) have been gaining a lot of interest due to their adaptability and extensibility in emerging applications, including communication networks. It is anticipated that ZSM networks will be able to support LLMs as a service, as they provide ultra reliable low-latency communications and closed loop massive connectivity. However, LLMs are vulnerable to data and model privacy issues that affect the trustworthiness of LLMs to be deployed for user-based services. In this paper, we explore the security vulnerabilities associated with fine-tuning LLMs in ZSM networks, in particular the membership inference attack. We define the characteristics of an attack network that can perform a membership inference attack if the attacker has access to the fine-tuned model for the downstream task. We show that the membership inference attacks are effective for any downstream task, which can lead to a personal data breach when using LLM as a service. The experimental results show that the attack success rate of maximum 92% can be achieved on named entity recognition task. Based on the experimental analysis, we discuss possible defense mechanisms and present possible research directions to make the LLMs more trustworthy in the context of ZSM networks.

摘要: 近年来，大型语言模型因其在通信网络等新兴应用中的适应性和可扩展性而引起了人们的极大兴趣。预计ZSM网络将能够支持LLMS作为一种服务，因为它们提供超可靠的低延迟通信和闭环路海量连接。然而，LLM容易受到数据和模型隐私问题的影响，这些问题会影响要为基于用户的服务部署的LLM的可信度。本文研究了ZSM网络中与微调LLMS相关的安全漏洞，特别是成员推理攻击。我们定义了攻击网络的特征，如果攻击者有权访问下游任务的微调模型，则该攻击网络可以执行成员关系推理攻击。我们证明了成员关系推断攻击对于任何下游任务都是有效的，当使用LLM作为服务时，这可能导致个人数据泄露。实验结果表明，命名实体识别任务的攻击成功率最高可达92%。在实验分析的基础上，我们讨论了可能的防御机制，并提出了可能的研究方向，以使ZSM网络环境下的LLMS更可信。



## **37. FlipedRAG: Black-Box Opinion Manipulation Attacks to Retrieval-Augmented Generation of Large Language Models**

FlipedRAG：对大型语言模型的检索增强生成的黑匣子观点操纵攻击 cs.IR

arXiv admin note: text overlap with arXiv:2407.13757

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2501.02968v1) [paper-pdf](http://arxiv.org/pdf/2501.02968v1)

**Authors**: Zhuo Chen, Yuyang Gong, Miaokun Chen, Haotan Liu, Qikai Cheng, Fan Zhang, Wei Lu, Xiaozhong Liu, Jiawei Liu

**Abstract**: Retrieval-Augmented Generation (RAG) addresses hallucination and real-time constraints by dynamically retrieving relevant information from a knowledge database to supplement the LLMs' input. When presented with a query, RAG selects the most semantically similar texts from its knowledge bases and uses them as context for the LLMs to generate more accurate responses. RAG also creates a new attack surface, especially since RAG databases are frequently sourced from public domains. While existing studies have predominantly focused on optimizing RAG's performance and efficiency, emerging research has begun addressing the security concerns associated with RAG. However, these works have some limitations, typically focusing on either white-box methodologies or heuristic-based black-box attacks. Furthermore, prior research has mainly targeted simple factoid question answering, which is neither practically challenging nor resistant to correction. In this paper, we unveil a more realistic and threatening scenario: opinion manipulation for controversial topics against RAG. Particularly, we propose a novel RAG black-box attack method, termed FlipedRAG, which is transfer-based. By leveraging instruction engineering, we obtain partial retrieval model outputs from black-box RAG system, facilitating the training of surrogate models to enhance the effectiveness of opinion manipulation attack. Extensive experimental results confirms that our approach significantly enhances the average success rate of opinion manipulation by 16.7%. It achieves an average of a 50% directional change in the opinion polarity of RAG responses across four themes. Additionally, it induces a 20% shift in user cognition. Furthermore, we discuss the efficacy of potential defense mechanisms and conclude that they are insufficient in mitigating this type of attack, highlighting the urgent need to develop novel defensive strategies.

摘要: 检索-增强生成(RAG)通过从知识数据库中动态检索相关信息来补充LLMS的输入，从而解决幻觉和实时约束。当出现查询时，RAG从其知识库中选择语义最相似的文本，并将其用作LLMS的上下文，以生成更准确的响应。RAG还创造了一个新的攻击面，特别是因为RAG数据库经常来自公共域。虽然现有的研究主要集中在优化RAG的性能和效率上，但新兴的研究已经开始解决与RAG相关的安全问题。然而，这些工作有一些局限性，通常集中在白盒方法或基于启发式的黑盒攻击。此外，以前的研究主要针对简单的事实式问题回答，这既不具有实际挑战性，也不抵制纠正。在这篇文章中，我们揭示了一个更现实和更具威胁性的场景：针对RAG的有争议话题的观点操纵。特别地，我们提出了一种新的基于传输的RAG黑盒攻击方法，称为FliedRAG。利用教学工程技术，从黑盒RAG系统中获取部分检索模型输出，便于对代理模型的训练，提高意见操纵攻击的有效性。大量的实验结果表明，该方法显著提高了意见操纵的平均成功率16.7%。它实现了四个主题的RAG回复的意见极性平均50%的方向性变化。此外，它还会导致用户认知发生20%的变化。此外，我们讨论了潜在的防御机制的有效性，并得出结论，它们在缓解这种类型的攻击方面是不够的，突出了开发新的防御策略的迫切需要。



## **38. LlamaPartialSpoof: An LLM-Driven Fake Speech Dataset Simulating Disinformation Generation**

LlamaPartialSpoof：一个LLM驱动的模拟虚假信息生成的假语音数据集 eess.AS

5 pages, ICASSP 2025

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2409.14743v2) [paper-pdf](http://arxiv.org/pdf/2409.14743v2)

**Authors**: Hieu-Thi Luong, Haoyang Li, Lin Zhang, Kong Aik Lee, Eng Siong Chng

**Abstract**: Previous fake speech datasets were constructed from a defender's perspective to develop countermeasure (CM) systems without considering diverse motivations of attackers. To better align with real-life scenarios, we created LlamaPartialSpoof, a 130-hour dataset that contains both fully and partially fake speech, using a large language model (LLM) and voice cloning technologies to evaluate the robustness of CMs. By examining valuable information for both attackers and defenders, we identify several key vulnerabilities in current CM systems, which can be exploited to enhance attack success rates, including biases toward certain text-to-speech models or concatenation methods. Our experimental results indicate that the current fake speech detection system struggle to generalize to unseen scenarios, achieving a best performance of 24.49% equal error rate.

摘要: 之前的虚假语音数据集是从防御者的角度构建的，以开发对策（CM）系统，而不考虑攻击者的不同动机。为了更好地与现实生活场景保持一致，我们创建了LlamaPartialSpoof，这是一个包含完全和部分虚假语音的130小时数据集，使用大型语言模型（LLM）和语音克隆技术来评估CM的稳健性。通过检查攻击者和防御者的有价值的信息，我们发现了当前CM系统中的几个关键漏洞，可以利用这些漏洞来提高攻击成功率，包括对某些文本到语音模型或级联方法的偏见。我们的实验结果表明，当前的虚假语音检测系统很难推广到不可见的场景，实现了24.49%等错误率的最佳性能。



## **39. Layer-Level Self-Exposure and Patch: Affirmative Token Mitigation for Jailbreak Attack Defense**

分层自我暴露和补丁：越狱攻击防御的肯定代币缓解 cs.CR

**SubmitDate**: 2025-01-05    [abs](http://arxiv.org/abs/2501.02629v1) [paper-pdf](http://arxiv.org/pdf/2501.02629v1)

**Authors**: Yang Ouyang, Hengrui Gu, Shuhang Lin, Wenyue Hua, Jie Peng, Bhavya Kailkhura, Tianlong Chen, Kaixiong Zhou

**Abstract**: As large language models (LLMs) are increasingly deployed in diverse applications, including chatbot assistants and code generation, aligning their behavior with safety and ethical standards has become paramount. However, jailbreak attacks, which exploit vulnerabilities to elicit unintended or harmful outputs, threaten LLMs' safety significantly. In this paper, we introduce Layer-AdvPatcher, a novel methodology designed to defend against jailbreak attacks by utilizing an unlearning strategy to patch specific layers within LLMs through self-augmented datasets. Our insight is that certain layer(s), tend to produce affirmative tokens when faced with harmful prompts. By identifying these layers and adversarially exposing them to generate more harmful data, one can understand their inherent and diverse vulnerabilities to attacks. With these exposures, we then "unlearn" these issues, reducing the impact of affirmative tokens and hence minimizing jailbreak risks while keeping the model's responses to safe queries intact. We conduct extensive experiments on two models, four benchmark datasets, and multiple state-of-the-art jailbreak benchmarks to demonstrate the efficacy of our approach. Results indicate that our framework reduces the harmfulness and attack success rate of jailbreak attacks without compromising utility for benign queries compared to recent defense methods.

摘要: 随着大型语言模型(LLM)越来越多地部署在各种应用中，包括聊天机器人助手和代码生成，使它们的行为符合安全和道德标准变得至关重要。然而，越狱攻击利用漏洞来引发意外或有害的输出，严重威胁到LLMS的安全。在本文中，我们介绍了Layer-AdvPatcher，这是一种新的方法，旨在通过一种遗忘策略来通过自增强数据集修补LLMS中的特定层来防御越狱攻击。我们的洞察是，某些层面(S)，在面对有害的提示时，往往会产生肯定的表征。通过识别这些层并恶意暴露它们以生成更多有害数据，人们可以了解它们固有的和不同的攻击漏洞。有了这些暴露，我们就可以“忘掉”这些问题，减少肯定令牌的影响，从而最大限度地减少越狱风险，同时保持模型对安全查询的响应完好无损。我们在两个模型、四个基准数据集和多个最先进的越狱基准上进行了广泛的实验，以展示我们方法的有效性。结果表明，与现有的防御方法相比，该框架降低了越狱攻击的危害性和攻击成功率，而不影响良性查询的有效性。



## **40. DiffusionAttacker: Diffusion-Driven Prompt Manipulation for LLM Jailbreak**

扩散攻击者：LLM越狱的扩散驱动提示操纵 cs.CL

**SubmitDate**: 2025-01-05    [abs](http://arxiv.org/abs/2412.17522v2) [paper-pdf](http://arxiv.org/pdf/2412.17522v2)

**Authors**: Hao Wang, Hao Li, Junda Zhu, Xinyuan Wang, Chengwei Pan, MinLie Huang, Lei Sha

**Abstract**: Large Language Models (LLMs) are susceptible to generating harmful content when prompted with carefully crafted inputs, a vulnerability known as LLM jailbreaking. As LLMs become more powerful, studying jailbreak methods is critical to enhancing security and aligning models with human values. Traditionally, jailbreak techniques have relied on suffix addition or prompt templates, but these methods suffer from limited attack diversity. This paper introduces DiffusionAttacker, an end-to-end generative approach for jailbreak rewriting inspired by diffusion models. Our method employs a sequence-to-sequence (seq2seq) text diffusion model as a generator, conditioning on the original prompt and guiding the denoising process with a novel attack loss. Unlike previous approaches that use autoregressive LLMs to generate jailbreak prompts, which limit the modification of already generated tokens and restrict the rewriting space, DiffusionAttacker utilizes a seq2seq diffusion model, allowing more flexible token modifications. This approach preserves the semantic content of the original prompt while producing harmful content. Additionally, we leverage the Gumbel-Softmax technique to make the sampling process from the diffusion model's output distribution differentiable, eliminating the need for iterative token search. Extensive experiments on Advbench and Harmbench demonstrate that DiffusionAttacker outperforms previous methods across various evaluation metrics, including attack success rate (ASR), fluency, and diversity.

摘要: 当提示使用精心编制的输入时，大型语言模型(LLM)很容易生成有害内容，这一漏洞被称为LLM越狱。随着LLMS变得越来越强大，研究越狱方法对于增强安全性和使模型与人类价值观保持一致至关重要。传统上，越狱技术依赖于后缀添加或提示模板，但这些方法受到攻击多样性的限制。本文介绍了一种受扩散模型启发的端到端生成式越狱重写方法DiffusionAttacker。该方法采用序列到序列(Seq2seq)文本扩散模型作为生成器，以原始提示为条件，以新的攻击损失指导去噪过程。与以前使用自回归LLM生成越狱提示的方法不同，DiffusionAttacker使用seq2seq扩散模型，允许更灵活的令牌修改，从而限制了对已生成令牌的修改并限制了重写空间。这种方法在产生有害内容的同时保留了原始提示的语义内容。此外，我们利用Gumbel-Softmax技术使扩散模型的输出分布的采样过程可微，从而消除了迭代令牌搜索的需要。在Advbench和Harmbench上的大量实验表明，DiffusionAttacker在包括攻击成功率(ASR)、流畅度和多样性在内的各种评估指标上都优于以前的方法。



## **41. A Practical Examination of AI-Generated Text Detectors for Large Language Models**

大型语言模型的人工智能生成文本检测器的实践检验 cs.CL

8 pages. Submitted to NAACL

**SubmitDate**: 2025-01-04    [abs](http://arxiv.org/abs/2412.05139v2) [paper-pdf](http://arxiv.org/pdf/2412.05139v2)

**Authors**: Brian Tufts, Xuandong Zhao, Lei Li

**Abstract**: The proliferation of large language models has raised growing concerns about their misuse, particularly in cases where AI-generated text is falsely attributed to human authors. Machine-generated content detectors claim to effectively identify such text under various conditions and from any language model. This paper critically evaluates these claims by assessing several popular detectors (RADAR, Wild, T5Sentinel, Fast-DetectGPT, GPTID, LogRank, Binoculars) on a range of domains, datasets, and models that these detectors have not previously encountered. We employ various prompting strategies to simulate adversarial attacks, demonstrating that even moderate efforts can significantly evade detection. We emphasize the importance of the true positive rate at a specific false positive rate (TPR@FPR) metric and demonstrate that these detectors perform poorly in certain settings, with TPR@.01 as low as 0%. Our findings suggest that both trained and zero-shot detectors struggle to maintain high sensitivity while achieving a reasonable true positive rate.

摘要: 大型语言模型的激增引发了人们对它们滥用的日益担忧，特别是在人工智能生成的文本被错误地归因于人类作者的情况下。机器生成的内容检测器声称可以在各种条件下从任何语言模型有效地识别此类文本。本文通过评估几种流行的探测器(雷达、Wild、T5Sentinel、Fast-DetectGPT、GPTID、logrank、双筒望远镜)，对这些声称进行了批判性的评估，这些探测器以前从未遇到过。我们使用各种提示策略来模拟对抗性攻击，表明即使是适度的攻击也可以显著地躲避检测。我们强调了在特定的假阳性率(TPR@fPR)度量下的真阳性率的重要性，并证明了这些检测器在某些设置下表现很差，TPR@.01低至0%。我们的发现表明，训练有素的探测器和零射探测器都很难在保持高灵敏度的同时获得合理的真阳性率。



## **42. A Survey of Recent Backdoor Attacks and Defenses in Large Language Models**

大型语言模型中最近后门攻击和防御的调查 cs.CR

Accepted in TMLR

**SubmitDate**: 2025-01-04    [abs](http://arxiv.org/abs/2406.06852v5) [paper-pdf](http://arxiv.org/pdf/2406.06852v5)

**Authors**: Shuai Zhao, Meihuizi Jia, Zhongliang Guo, Leilei Gan, Xiaoyu Xu, Xiaobao Wu, Jie Fu, Yichao Feng, Fengjun Pan, Luu Anh Tuan

**Abstract**: Large Language Models (LLMs), which bridge the gap between human language understanding and complex problem-solving, achieve state-of-the-art performance on several NLP tasks, particularly in few-shot and zero-shot settings. Despite the demonstrable efficacy of LLMs, due to constraints on computational resources, users have to engage with open-source language models or outsource the entire training process to third-party platforms. However, research has demonstrated that language models are susceptible to potential security vulnerabilities, particularly in backdoor attacks. Backdoor attacks are designed to introduce targeted vulnerabilities into language models by poisoning training samples or model weights, allowing attackers to manipulate model responses through malicious triggers. While existing surveys on backdoor attacks provide a comprehensive overview, they lack an in-depth examination of backdoor attacks specifically targeting LLMs. To bridge this gap and grasp the latest trends in the field, this paper presents a novel perspective on backdoor attacks for LLMs by focusing on fine-tuning methods. Specifically, we systematically classify backdoor attacks into three categories: full-parameter fine-tuning, parameter-efficient fine-tuning, and no fine-tuning Based on insights from a substantial review, we also discuss crucial issues for future research on backdoor attacks, such as further exploring attack algorithms that do not require fine-tuning, or developing more covert attack algorithms.

摘要: 大型语言模型(LLM)架起了人类语言理解和复杂问题解决之间的桥梁，在几个NLP任务上实现了最先进的性能，特别是在少镜头和零镜头的情况下。尽管LLMS具有明显的功效，但由于计算资源的限制，用户不得不使用开放源码语言模型或将整个培训过程外包给第三方平台。然而，研究表明，语言模型容易受到潜在的安全漏洞的影响，特别是在后门攻击中。后门攻击旨在通过毒化训练样本或模型权重，将有针对性的漏洞引入语言模型，允许攻击者通过恶意触发器操纵模型响应。虽然现有的关于后门攻击的调查提供了全面的概述，但它们缺乏对专门针对LLM的后门攻击的深入检查。为了弥补这一差距，掌握该领域的最新趋势，本文提出了一种新的视角来研究针对LLMS的后门攻击，重点是微调方法。具体地说，我们系统地将后门攻击分为三类：全参数微调、参数高效微调和无微调。在大量综述的基础上，我们还讨论了未来后门攻击研究的关键问题，如进一步探索不需要微调的攻击算法，或开发更隐蔽的攻击算法。



## **43. AVTrustBench: Assessing and Enhancing Reliability and Robustness in Audio-Visual LLMs**

AVTrustBench：评估和增强视听LLM的可靠性和稳健性 cs.CV

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2501.02135v1) [paper-pdf](http://arxiv.org/pdf/2501.02135v1)

**Authors**: Sanjoy Chowdhury, Sayan Nag, Subhrajyoti Dasgupta, Yaoting Wang, Mohamed Elhoseiny, Ruohan Gao, Dinesh Manocha

**Abstract**: With the rapid advancement of Multi-modal Large Language Models (MLLMs), several diagnostic benchmarks have recently been developed to assess these models' multi-modal reasoning proficiency. However, these benchmarks are restricted to assessing primarily the visual aspect and do not examine the holistic audio-visual (AV) understanding. Moreover, currently, there are no benchmarks that investigate the capabilities of AVLLMs to calibrate their responses when presented with perturbed inputs. To this end, we introduce Audio-Visual Trustworthiness assessment Benchmark (AVTrustBench), comprising 600K samples spanning over 9 meticulously crafted tasks, evaluating the capabilities of AVLLMs across three distinct dimensions: Adversarial attack, Compositional reasoning, and Modality-specific dependency. Using our benchmark we extensively evaluate 13 state-of-the-art AVLLMs. The findings reveal that the majority of existing models fall significantly short of achieving human-like comprehension, offering valuable insights for future research directions. To alleviate the limitations in the existing approaches, we further propose a robust, model-agnostic calibrated audio-visual preference optimization based training strategy CAVPref, obtaining a gain up to 30.19% across all 9 tasks. We will publicly release our code and benchmark to facilitate future research in this direction.

摘要: 随着多通道大型语言模型(MLLMS)的迅速发展，最近出现了几个用于评估这些模型的多通道推理能力的诊断基准。然而，这些基准仅限于主要评估视觉方面，而不检查整体视听(AV)理解。此外，目前还没有基准来调查AVLLMS在收到扰动输入时校准其响应的能力。为此，我们引入了视听可信度评估基准(AVTrustB边)，该基准包括60万个样本，跨越9个精心制作的任务，从三个不同的维度评估AVLLMS的能力：对抗攻击、成分推理和特定通道依赖。使用我们的基准，我们广泛评估了13个最先进的AVLLM。研究结果表明，现有的大多数模型都明显不能实现类似人类的理解，为未来的研究方向提供了有价值的见解。为了缓解现有方法的局限性，我们进一步提出了一种稳健的、与模型无关的、基于校准视听偏好优化的训练策略CAVPref，在所有9个任务中都获得了高达30.19%的收益。我们将公开发布我们的代码和基准，以促进未来在这一方向的研究。



## **44. Auto-RT: Automatic Jailbreak Strategy Exploration for Red-Teaming Large Language Models**

Auto-RT：Red-Teaming大型语言模型的自动越狱策略探索 cs.CR

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2501.01830v1) [paper-pdf](http://arxiv.org/pdf/2501.01830v1)

**Authors**: Yanjiang Liu, Shuhen Zhou, Yaojie Lu, Huijia Zhu, Weiqiang Wang, Hongyu Lin, Ben He, Xianpei Han, Le Sun

**Abstract**: Automated red-teaming has become a crucial approach for uncovering vulnerabilities in large language models (LLMs). However, most existing methods focus on isolated safety flaws, limiting their ability to adapt to dynamic defenses and uncover complex vulnerabilities efficiently. To address this challenge, we propose Auto-RT, a reinforcement learning framework that automatically explores and optimizes complex attack strategies to effectively uncover security vulnerabilities through malicious queries. Specifically, we introduce two key mechanisms to reduce exploration complexity and improve strategy optimization: 1) Early-terminated Exploration, which accelerate exploration by focusing on high-potential attack strategies; and 2) Progressive Reward Tracking algorithm with intermediate downgrade models, which dynamically refine the search trajectory toward successful vulnerability exploitation. Extensive experiments across diverse LLMs demonstrate that, by significantly improving exploration efficiency and automatically optimizing attack strategies, Auto-RT detects a boarder range of vulnerabilities, achieving a faster detection speed and 16.63\% higher success rates compared to existing methods.

摘要: 自动红团队已成为发现大型语言模型(LLM)中漏洞的关键方法。然而，现有的大多数方法都集中在孤立的安全漏洞上，限制了它们适应动态防御和有效发现复杂漏洞的能力。为了应对这一挑战，我们提出了Auto-RT，这是一个强化学习框架，它自动探索和优化复杂的攻击策略，通过恶意查询有效地发现安全漏洞。具体地说，我们引入了两个关键机制来降低探测复杂性和改善策略优化：1)提前终止探测，通过关注高潜在攻击策略来加速探测；2)采用中间降级模型的渐进式奖励跟踪算法，动态细化搜索轨迹，以实现成功的漏洞攻击。大量的实验表明，通过显著提高探测效率和自动优化攻击策略，Auto-RT可以检测到更广泛的漏洞，与现有方法相比，检测速度更快，成功率更高。



## **45. How Toxic Can You Get? Search-based Toxicity Testing for Large Language Models**

你能得到多大的毒性？基于搜索的大型语言模型毒性测试 cs.SE

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2501.01741v1) [paper-pdf](http://arxiv.org/pdf/2501.01741v1)

**Authors**: Simone Corbo, Luca Bancale, Valeria De Gennaro, Livia Lestingi, Vincenzo Scotti, Matteo Camilli

**Abstract**: Language is a deep-rooted means of perpetration of stereotypes and discrimination. Large Language Models (LLMs), now a pervasive technology in our everyday lives, can cause extensive harm when prone to generating toxic responses. The standard way to address this issue is to align the LLM, which, however, dampens the issue without constituting a definitive solution. Therefore, testing LLM even after alignment efforts remains crucial for detecting any residual deviations with respect to ethical standards. We present EvoTox, an automated testing framework for LLMs' inclination to toxicity, providing a way to quantitatively assess how much LLMs can be pushed towards toxic responses even in the presence of alignment. The framework adopts an iterative evolution strategy that exploits the interplay between two LLMs, the System Under Test (SUT) and the Prompt Generator steering SUT responses toward higher toxicity. The toxicity level is assessed by an automated oracle based on an existing toxicity classifier. We conduct a quantitative and qualitative empirical evaluation using four state-of-the-art LLMs as evaluation subjects having increasing complexity (7-13 billion parameters). Our quantitative evaluation assesses the cost-effectiveness of four alternative versions of EvoTox against existing baseline methods, based on random search, curated datasets of toxic prompts, and adversarial attacks. Our qualitative assessment engages human evaluators to rate the fluency of the generated prompts and the perceived toxicity of the responses collected during the testing sessions. Results indicate that the effectiveness, in terms of detected toxicity level, is significantly higher than the selected baseline methods (effect size up to 1.0 against random search and up to 0.99 against adversarial attacks). Furthermore, EvoTox yields a limited cost overhead (from 22% to 35% on average).

摘要: 语言是制造陈规定型观念和歧视的根深蒂固的手段。大语言模型(LLM)现在是我们日常生活中的一项普遍技术，当它容易产生有毒反应时，可能会造成广泛的危害。解决这一问题的标准方法是调整LLM，然而，这会抑制问题，而不会构成最终的解决方案。因此，即使在调整工作之后进行LLM测试，对于检测与道德标准有关的任何残余偏差仍然至关重要。我们提出了EvoTox，这是一个用于LLMS毒性倾向的自动化测试框架，提供了一种方法来定量评估即使在存在对齐的情况下，LLMS也可以被推向毒性反应的程度。该框架采用了一种迭代进化策略，利用了两个LLM之间的相互作用，被测系统(SUT)和即时生成器将SUT的响应转向更高的毒性。毒性水平由基于现有毒性分类器的自动先知进行评估。我们使用四个最先进的LLM作为评估对象，进行了定量和定性的实证评估，这些评估对象的复杂性越来越高(参数为70-130亿个)。我们的定量评估基于随机搜索、有毒提示的精选数据集和对抗性攻击，相对于现有的基线方法评估了EvoTox的四个替代版本的成本效益。我们的定性评估聘请人类评估员对生成的提示的流畅性和在测试期间收集的回答的感知毒性进行评级。结果表明，就检测到的毒性水平而言，该方法的有效性显著高于所选的基线方法(对随机搜索的效果大小高达1.0，对对抗性攻击的效果高达0.99)。此外，EvoTox产生的成本管理费用有限(平均从22%到35%)。



## **46. Heuristic-Induced Multimodal Risk Distribution Jailbreak Attack for Multimodal Large Language Models**

启发式多峰大语言模型的多峰风险分布越狱攻击 cs.CR

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2412.05934v2) [paper-pdf](http://arxiv.org/pdf/2412.05934v2)

**Authors**: Ma Teng, Jia Xiaojun, Duan Ranjie, Li Xinfeng, Huang Yihao, Chu Zhixuan, Liu Yang, Ren Wenqi

**Abstract**: With the rapid advancement of multimodal large language models (MLLMs), concerns regarding their security have increasingly captured the attention of both academia and industry. Although MLLMs are vulnerable to jailbreak attacks, designing effective multimodal jailbreak attacks poses unique challenges, especially given the distinct protective measures implemented across various modalities in commercial models. Previous works concentrate risks into a single modality, resulting in limited jailbreak performance. In this paper, we propose a heuristic-induced multimodal risk distribution jailbreak attack method, called HIMRD, which consists of two elements: multimodal risk distribution strategy and heuristic-induced search strategy. The multimodal risk distribution strategy is used to segment harmful instructions across multiple modalities to effectively circumvent MLLMs' security protection. The heuristic-induced search strategy identifies two types of prompts: the understanding-enhancing prompt, which helps the MLLM reconstruct the malicious prompt, and the inducing prompt, which increases the likelihood of affirmative outputs over refusals, enabling a successful jailbreak attack. Extensive experiments demonstrate that this approach effectively uncovers vulnerabilities in MLLMs, achieving an average attack success rate of 90% across seven popular open-source MLLMs and an average attack success rate of around 68% in three popular closed-source MLLMs. Our code will coming soon. Warning: This paper contains offensive and harmful examples, reader discretion is advised.

摘要: 随着多通道大语言模型的快速发展，对其安全性的关注日益引起学术界和工业界的关注。尽管大规模杀伤性武器易受越狱攻击，但设计有效的多模式越狱攻击是一个独特的挑战，特别是考虑到商业模式中的各种模式实施了不同的保护措施。以前的工作将风险集中到单一模式中，导致有限的越狱性能。本文提出了一种启发式多通道风险分布越狱攻击方法HIMRD，该方法由两部分组成：多通道风险分布策略和启发式搜索策略。多模式风险分配策略用于跨多个模式分割有害指令，以有效规避MLLMS的安全保护。启发式搜索策略识别两种类型的提示：促进理解的提示和诱导性提示，前者帮助MLLM重建恶意提示，后者增加肯定输出超过拒绝的可能性，从而实现成功的越狱攻击。大量实验表明，该方法有效地发现了MLLMS中的漏洞，在七个流行的开源MLLMS上的平均攻击成功率达到了90%，在三个流行的闭源MLLMS上的平均攻击成功率达到了68%左右。我们的代码很快就会出来。警告：本文包含冒犯性和有害的例子，建议读者酌情处理。



## **47. Spot Risks Before Speaking! Unraveling Safety Attention Heads in Large Vision-Language Models**

说话前现货风险！解开大型视觉语言模型中的安全注意力 cs.LG

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2501.02029v1) [paper-pdf](http://arxiv.org/pdf/2501.02029v1)

**Authors**: Ziwei Zheng, Junyao Zhao, Le Yang, Lijun He, Fan Li

**Abstract**: With the integration of an additional modality, large vision-language models (LVLMs) exhibit greater vulnerability to safety risks (e.g., jailbreaking) compared to their language-only predecessors. Although recent studies have devoted considerable effort to the post-hoc alignment of LVLMs, the inner safety mechanisms remain largely unexplored. In this paper, we discover that internal activations of LVLMs during the first token generation can effectively identify malicious prompts across different attacks. This inherent safety perception is governed by sparse attention heads, which we term ``safety heads." Further analysis reveals that these heads act as specialized shields against malicious prompts; ablating them leads to higher attack success rates, while the model's utility remains unaffected. By locating these safety heads and concatenating their activations, we construct a straightforward but powerful malicious prompt detector that integrates seamlessly into the generation process with minimal extra inference overhead. Despite its simple structure of a logistic regression model, the detector surprisingly exhibits strong zero-shot generalization capabilities. Experiments across various prompt-based attacks confirm the effectiveness of leveraging safety heads to protect LVLMs. Code is available at \url{https://github.com/Ziwei-Zheng/SAHs}.

摘要: 随着另一种模式的集成，与仅使用语言的前身相比，大型视觉语言模型(LVLM)显示出更大的安全风险(例如越狱)。尽管最近的研究已经在左肺小梁的术后配对上投入了相当大的努力，但其内部的安全机制在很大程度上仍未被探索。在本文中，我们发现在第一次令牌生成过程中内部激活LVLM可以有效地识别跨不同攻击的恶意提示。这种固有的安全感是由稀疏的注意力头部所支配的，我们称之为“安全头部”。进一步的分析表明，这些头部作为专门的盾牌来抵御恶意提示；消除它们会导致更高的攻击成功率，而该模型的实用性不会受到影响。通过定位这些安全头并连接它们的激活，我们构建了一个简单但强大的恶意提示检测器，该检测器无缝地集成到生成过程中，并且具有最小的额外推理开销。尽管它的结构简单的Logistic回归模型，但该检测器出人意料地显示出强大的零射泛化能力。各种基于提示的攻击的实验证实了利用安全头来保护LVLM的有效性。代码位于\url{https://github.com/Ziwei-Zheng/SAHs}.



## **48. BARTPredict: Empowering IoT Security with LLM-Driven Cyber Threat Prediction**

BartPredict：通过LLM驱动的网络威胁预测增强物联网安全性 cs.CR

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2501.01664v1) [paper-pdf](http://arxiv.org/pdf/2501.01664v1)

**Authors**: Alaeddine Diaf, Abdelaziz Amara Korba, Nour Elislem Karabadji, Yacine Ghamri-Doudane

**Abstract**: The integration of Internet of Things (IoT) technology in various domains has led to operational advancements, but it has also introduced new vulnerabilities to cybersecurity threats, as evidenced by recent widespread cyberattacks on IoT devices. Intrusion detection systems are often reactive, triggered by specific patterns or anomalies observed within the network. To address this challenge, this work proposes a proactive approach to anticipate and preemptively mitigate malicious activities, aiming to prevent potential damage before it occurs. This paper proposes an innovative intrusion prediction framework empowered by Pre-trained Large Language Models (LLMs). The framework incorporates two LLMs: a fine-tuned Bidirectional and AutoRegressive Transformers (BART) model for predicting network traffic and a fine-tuned Bidirectional Encoder Representations from Transformers (BERT) model for evaluating the predicted traffic. By harnessing the bidirectional capabilities of BART the framework then identifies malicious packets among these predictions. Evaluated using the CICIoT2023 IoT attack dataset, our framework showcases a notable enhancement in predictive performance, attaining an impressive 98% overall accuracy, providing a powerful response to the cybersecurity challenges that confront IoT networks.

摘要: 物联网(IoT)技术在各个领域的整合带来了运营上的进步，但也给网络安全威胁带来了新的漏洞，最近针对物联网设备的广泛网络攻击就是明证。入侵检测系统通常是被动的，由网络中观察到的特定模式或异常触发。为了应对这一挑战，这项工作提出了一种积极主动的方法来预测和先发制人地减轻恶意活动，旨在防止潜在的损害发生之前。提出了一种基于预训练的大语言模型的入侵预测框架。该框架包含两个LLM：用于预测网络流量的微调双向和自回归转换器(BART)模型和用于评估预测流量的微调双向编码器表示(BERT)模型。通过利用BART的双向功能，该框架可以在这些预测中识别恶意数据包。使用CICIoT2023物联网攻击数据集进行评估，我们的框架在预测性能方面有了显著的增强，总体准确率达到了令人印象深刻的98%，为物联网网络面临的网络安全挑战提供了强大的响应。



## **49. CySecBench: Generative AI-based CyberSecurity-focused Prompt Dataset for Benchmarking Large Language Models**

CySecBench：基于人工智能、以网络安全为重点的生成性提示数据集，用于对大型语言模型进行基准测试 cs.CR

**SubmitDate**: 2025-01-02    [abs](http://arxiv.org/abs/2501.01335v1) [paper-pdf](http://arxiv.org/pdf/2501.01335v1)

**Authors**: Johan Wahréus, Ahmed Mohamed Hussain, Panos Papadimitratos

**Abstract**: Numerous studies have investigated methods for jailbreaking Large Language Models (LLMs) to generate harmful content. Typically, these methods are evaluated using datasets of malicious prompts designed to bypass security policies established by LLM providers. However, the generally broad scope and open-ended nature of existing datasets can complicate the assessment of jailbreaking effectiveness, particularly in specific domains, notably cybersecurity. To address this issue, we present and publicly release CySecBench, a comprehensive dataset containing 12662 prompts specifically designed to evaluate jailbreaking techniques in the cybersecurity domain. The dataset is organized into 10 distinct attack-type categories, featuring close-ended prompts to enable a more consistent and accurate assessment of jailbreaking attempts. Furthermore, we detail our methodology for dataset generation and filtration, which can be adapted to create similar datasets in other domains. To demonstrate the utility of CySecBench, we propose and evaluate a jailbreaking approach based on prompt obfuscation. Our experimental results show that this method successfully elicits harmful content from commercial black-box LLMs, achieving Success Rates (SRs) of 65% with ChatGPT and 88% with Gemini; in contrast, Claude demonstrated greater resilience with a jailbreaking SR of 17%. Compared to existing benchmark approaches, our method shows superior performance, highlighting the value of domain-specific evaluation datasets for assessing LLM security measures. Moreover, when evaluated using prompts from a widely used dataset (i.e., AdvBench), it achieved an SR of 78.5%, higher than the state-of-the-art methods.

摘要: 许多研究已经调查了越狱大型语言模型(LLM)生成有害内容的方法。通常，使用恶意提示的数据集来评估这些方法，这些恶意提示旨在绕过LLM提供商建立的安全策略。然而，现有数据集的广泛范围和开放式性质可能会使越狱效果的评估复杂化，特别是在特定领域，特别是网络安全领域。为了解决这个问题，我们提出并公开发布了CySecBitch，这是一个全面的数据集，包含12662个提示，专门用于评估网络安全领域的越狱技术。该数据集被组织成10个不同的攻击类型类别，具有封闭式提示，以实现对越狱企图的更一致和更准确的评估。此外，我们详细介绍了我们的数据集生成和过滤方法，该方法可以适用于在其他领域创建类似的数据集。为了证明CySecBitch的有效性，我们提出并评估了一种基于即时混淆的越狱方法。我们的实验结果表明，该方法成功地从商业黑盒LLMS中提取出有害内容，ChatGPT的成功率(SRS)为65%，Gemini为88%；相比之下，Claude表现出更强的弹性，越狱成功率为17%。与现有的基准测试方法相比，我们的方法表现出更好的性能，突出了特定于域的评估数据集在评估LLM安全措施方面的价值。此外，当使用广泛使用的数据集(即AdvBch)的提示进行评估时，它获得了78.5%的SR，高于最先进的方法。



## **50. Safeguarding Large Language Models in Real-time with Tunable Safety-Performance Trade-offs**

通过可调的安全性能权衡实时保护大型语言模型 cs.CL

**SubmitDate**: 2025-01-02    [abs](http://arxiv.org/abs/2501.02018v1) [paper-pdf](http://arxiv.org/pdf/2501.02018v1)

**Authors**: Joao Fonseca, Andrew Bell, Julia Stoyanovich

**Abstract**: Large Language Models (LLMs) have been shown to be susceptible to jailbreak attacks, or adversarial attacks used to illicit high risk behavior from a model. Jailbreaks have been exploited by cybercriminals and blackhat actors to cause significant harm, highlighting the critical need to safeguard widely-deployed models. Safeguarding approaches, which include fine-tuning models or having LLMs "self-reflect", may lengthen the inference time of a model, incur a computational penalty, reduce the semantic fluency of an output, and restrict ``normal'' model behavior. Importantly, these Safety-Performance Trade-offs (SPTs) remain an understudied area. In this work, we introduce a novel safeguard, called SafeNudge, that combines Controlled Text Generation with "nudging", or using text interventions to change the behavior of a model. SafeNudge triggers during text-generation while a jailbreak attack is being executed, and can reduce successful jailbreak attempts by 30% by guiding the LLM towards a safe responses. It adds minimal latency to inference and has a negligible impact on the semantic fluency of outputs. Further, we allow for tunable SPTs. SafeNudge is open-source and available through https://pypi.org/, and is compatible with models loaded with the Hugging Face "transformers" library.

摘要: 大型语言模型(LLM)已被证明容易受到越狱攻击，即用于从模型中非法进行高风险行为的对抗性攻击。越狱已被网络犯罪分子和黑帽行为者利用，造成重大危害，突显出保护广泛部署的模型的迫切需要。保护方法，包括微调模型或让LLM“自我反思”，可能会延长模型的推理时间，招致计算惩罚，降低输出的语义流畅性，并限制“正常”的模型行为。重要的是，这些安全-性能权衡(SPTS)仍然是一个研究较少的领域。在这项工作中，我们引入了一种新的安全措施，称为安全轻推，它将受控文本生成与“轻推”相结合，即使用文本干预来改变模型的行为。安全轻推在执行越狱攻击时在文本生成过程中触发，通过引导LLM进行安全响应，可以将成功的越狱尝试减少30%。它增加了最小的推理延迟，并且对输出的语义流畅性的影响可以忽略不计。此外，我们还考虑了可调SPT。SafeNdge是开源的，可以通过https://pypi.org/，获得，并且与装载了拥抱脸“变形金刚”库的模型兼容。



