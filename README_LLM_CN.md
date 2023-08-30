# Latest Adversarial Attack Papers
**update at 2023-08-30 11:16:59**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. The Effectiveness of Large Language Models (ChatGPT and CodeBERT) for Security-Oriented Code Analysis**

大型语言模型(ChatGPT和CodeBERT)对面向安全的代码分析的有效性 cs.CR

3 Table, 8 figures

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2307.12488v3) [paper-pdf](http://arxiv.org/pdf/2307.12488v3)

**Authors**: Zhilong Wang, Lan Zhang, Chen Cao, Peng Liu

**Abstract**: Large Language Models (LLMs), such as GPT and BERT, have demonstrated remarkable capabilities in addressing neural language process tasks. Recently, the release of ChatGPT has garnered significant attention due to its ability to analyze, comprehend, and synthesize information from user inputs. Therefore, these LLMs were adopted by researchers in many different domains. In the realm of code analysis, researchers have applied LLMs to tasks like code review and code generation. However, we observed that the strengths and limitations of adopting these LLMs to the code analysis have not been investigated. In this paper, we delve into LLMs' capabilities in security-oriented program analysis, considering perspectives from both attackers and security analysts. We focus on two representative LLMs, ChatGPT and CodeBert, and evaluate their performance in solving typical analytic tasks with varying levels of difficulty. Given the different natures of ChatGPT and CodeBERT, we conduct a qualitative analysis of the model's output for ChatGPT and a quantitative analysis for CodeBERT, respectively. For ChatGPT, we present a case study involving several security-oriented program analysis tasks while deliberately introducing challenges to assess its responses. On the other hand, for CodeBERT, we systematically analyze and classify the features in code, quantitatively evaluating the impact of these features on the model's performance. Our study demonstrates the LLM's efficiency in learning high-level semantics from code, positioning ChatGPT as a potential asset in security-oriented contexts. However, it is essential to acknowledge certain limitations, such as the heavy reliance on well-defined variable and function names, making them unable to learn from anonymized code. We hope that our findings and analysis will offer valuable insights for future researchers in this domain.

摘要: 大型语言模型(LLM)，如GPT和BERT，在处理神经语言处理任务方面表现出了非凡的能力。最近，ChatGPT的发布受到了极大的关注，因为它能够分析、理解和综合来自用户输入的信息。因此，这些LLM被许多不同领域的研究人员所采用。在代码分析领域，研究人员已经将LLM应用于代码审查和代码生成等任务。然而，我们注意到，采用这些LLM进行代码分析的优点和局限性尚未得到调查。在本文中，我们从攻击者和安全分析师的角度深入研究了LLMS在面向安全的程序分析中的能力。我们集中于两个有代表性的LLM，ChatGPT和CodeBert，并评估了它们在解决不同难度的典型分析任务中的性能。鉴于ChatGPT和CodeBERT的不同性质，我们分别对ChatGPT和CodeBERT的模型输出进行了定性分析和定量分析。对于ChatGPT，我们提供了一个案例研究，涉及几个面向安全的程序分析任务，同时故意引入挑战来评估其响应。另一方面，对于CodeBERT，我们对代码中的特征进行了系统的分析和分类，定量地评估了这些特征对模型性能的影响。我们的研究证明了LLM在从代码中学习高级语义方面的效率，并将ChatGPT定位为面向安全的上下文中的潜在资产。然而，必须承认某些限制，例如严重依赖定义明确的变量和函数名称，使它们无法从匿名代码中学习。我们希望我们的发现和分析将为这一领域的未来研究人员提供有价值的见解。



## **2. Identifying and Mitigating the Security Risks of Generative AI**

识别和缓解生成性人工智能的安全风险 cs.AI

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2308.14840v1) [paper-pdf](http://arxiv.org/pdf/2308.14840v1)

**Authors**: Clark Barrett, Brad Boyd, Ellie Burzstein, Nicholas Carlini, Brad Chen, Jihye Choi, Amrita Roy Chowdhury, Mihai Christodorescu, Anupam Datta, Soheil Feizi, Kathleen Fisher, Tatsunori Hashimoto, Dan Hendrycks, Somesh Jha, Daniel Kang, Florian Kerschbaum, Eric Mitchell, John Mitchell, Zulfikar Ramzan, Khawaja Shams, Dawn Song, Ankur Taly, Diyi Yang

**Abstract**: Every major technical invention resurfaces the dual-use dilemma -- the new technology has the potential to be used for good as well as for harm. Generative AI (GenAI) techniques, such as large language models (LLMs) and diffusion models, have shown remarkable capabilities (e.g., in-context learning, code-completion, and text-to-image generation and editing). However, GenAI can be used just as well by attackers to generate new attacks and increase the velocity and efficacy of existing attacks.   This paper reports the findings of a workshop held at Google (co-organized by Stanford University and the University of Wisconsin-Madison) on the dual-use dilemma posed by GenAI. This paper is not meant to be comprehensive, but is rather an attempt to synthesize some of the interesting findings from the workshop. We discuss short-term and long-term goals for the community on this topic. We hope this paper provides both a launching point for a discussion on this important topic as well as interesting problems that the research community can work to address.

摘要: 每一项重大技术发明都会重新面临两难境地--新技术既有可能被用来做好事，也有可能被用来做坏事。生成性人工智能(GenAI)技术，如大型语言模型(LLMS)和扩散模型，已经显示出非凡的能力(例如，上下文学习、代码完成以及文本到图像的生成和编辑)。然而，攻击者也可以利用GenAI来生成新的攻击，并提高现有攻击的速度和效率。本文报告了在谷歌(由斯坦福大学和威斯康星大学麦迪逊分校联合举办)举行的关于GenAI造成的两用困境的研讨会的结果。这篇论文并不是要全面的，而是试图综合研讨会的一些有趣的发现。我们就这一主题讨论社区的短期和长期目标。我们希望这篇论文既为讨论这一重要主题提供了一个起点，也为研究界可以努力解决的有趣问题提供了一个起点。



## **3. Out of the Cage: How Stochastic Parrots Win in Cyber Security Environments**

走出笼子：随机鹦鹉如何在网络安全环境中取胜 cs.CR

Under review. 10 pages plus appendices, 7 figures, 4 tables. Edit:  fix e-mails and code repository

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2308.12086v2) [paper-pdf](http://arxiv.org/pdf/2308.12086v2)

**Authors**: Maria Rigaki, Ondřej Lukáš, Carlos A. Catania, Sebastian Garcia

**Abstract**: Large Language Models (LLMs) have gained widespread popularity across diverse domains involving text generation, summarization, and various natural language processing tasks. Despite their inherent limitations, LLM-based designs have shown promising capabilities in planning and navigating open-world scenarios. This paper introduces a novel application of pre-trained LLMs as agents within cybersecurity network environments, focusing on their utility for sequential decision-making processes.   We present an approach wherein pre-trained LLMs are leveraged as attacking agents in two reinforcement learning environments. Our proposed agents demonstrate similar or better performance against state-of-the-art agents trained for thousands of episodes in most scenarios and configurations. In addition, the best LLM agents perform similarly to human testers of the environment without any additional training process. This design highlights the potential of LLMs to efficiently address complex decision-making tasks within cybersecurity.   Furthermore, we introduce a new network security environment named NetSecGame. The environment is designed to eventually support complex multi-agent scenarios within the network security domain. The proposed environment mimics real network attacks and is designed to be highly modular and adaptable for various scenarios.

摘要: 大语言模型在涉及文本生成、摘要和各种自然语言处理任务的不同领域得到了广泛的欢迎。尽管有其固有的局限性，基于LLM的设计在规划和导航开放世界场景方面显示出了良好的能力。本文介绍了在网络安全网络环境中将预先训练的LLM作为代理的一种新的应用，重点讨论了它们在顺序决策过程中的效用。我们提出了一种方法，其中预先训练的LLM在两个强化学习环境中被用作攻击代理。我们建议的代理与在大多数场景和配置中培训了数千集的最先进的代理相比，表现出类似或更好的性能。此外，最好的LLM代理的表现类似于环境中的人类测试员，而无需任何额外的培训过程。此设计突出了LLMS在有效处理网络安全中的复杂决策任务方面的潜力。此外，我们还介绍了一个新的网络安全环境NetSecGame。该环境旨在最终支持网络安全域内的复杂多代理方案。提出的环境模拟真实的网络攻击，并被设计为高度模块化和适应各种场景。



## **4. A Comprehensive Overview of Backdoor Attacks in Large Language Models within Communication Networks**

通信网络中大型语言模型中的后门攻击综述 cs.CR

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2308.14367v1) [paper-pdf](http://arxiv.org/pdf/2308.14367v1)

**Authors**: Haomiao Yang, Kunlan Xiang, Hongwei Li, Rongxing Lu

**Abstract**: The Large Language Models (LLMs) are becoming an integral part of modern communication networks due to their superior proficiency in language comprehension and generation. In the context of these networks, where limited data and computing resources often necessitate the use of third-party data and computing resources, the risk of backdoor attacks becomes highly significant. Such strategies may expose the model within the network to maliciously manipulated training data and processing, providing an opportunity for attackers to embed a hidden backdoor into the model, termed a backdoor attack. Backdoor attack in LLMs refers to embedding a hidden backdoor in LLMs that causes the model to perform normally on benign samples but exhibit degraded performance on poisoned ones. This issue is particularly concerning within communication networks where reliability and security are paramount. Despite the extensive research on backdoor attacks, there remains a lack of in-depth exploration specifically within the context of LLMs employed in communication networks, and a systematic review of such attacks is currently absent. In this survey, we systematically propose a taxonomy of backdoor attacks in LLMs as used in communication networks, dividing them into four major categories: input-triggered, prompt-triggered, instruction-triggered, and demonstration-triggered attacks. Furthermore, we conduct a comprehensive analysis of the benchmark datasets within the network domain. Finally, we identify potential problems and open challenges, offering valuable insights into future research directions for enhancing the security and integrity of LLMs in communication networks.

摘要: 大型语言模型由于其在语言理解和生成方面的卓越能力，正在成为现代交流网络中不可或缺的一部分。在这些网络的背景下，有限的数据和计算资源往往需要使用第三方数据和计算资源，后门攻击的风险变得非常大。这种策略可能会使网络中的模型暴露于恶意操纵的训练数据和处理过程中，为攻击者提供在模型中嵌入隐藏后门的机会，称为后门攻击。LLMS中的后门攻击是指在LLMS中嵌入隐藏的后门，导致模型在良性样本上正常运行，但在有毒样本上表现出降级的性能。在可靠性和安全性至关重要的通信网络中，这个问题尤其令人担忧。尽管对后门攻击进行了广泛的研究，但仍然缺乏特别是在通信网络中使用的LLM的深入探索，而且目前还没有对这类攻击进行系统审查。在本次调查中，我们系统地提出了一种用于通信网络的LLMS后门攻击的分类，将它们分为四大类：输入触发的攻击、提示触发的攻击、指令触发的攻击和演示触发的攻击。此外，我们对网络域内的基准数据集进行了全面的分析。最后，我们确定了潜在的问题和开放的挑战，为未来的研究方向提供了有价值的见解，以提高通信网络中LLMS的安全性和完整性。



## **5. Detecting Language Model Attacks with Perplexity**

基于困惑的语言模型攻击检测 cs.CL

**SubmitDate**: 2023-08-27    [abs](http://arxiv.org/abs/2308.14132v1) [paper-pdf](http://arxiv.org/pdf/2308.14132v1)

**Authors**: Gabriel Alon, Michael Kamfonas

**Abstract**: A novel hack involving Large Language Models (LLMs) has emerged, leveraging adversarial suffixes to trick models into generating perilous responses. This method has garnered considerable attention from reputable media outlets such as the New York Times and Wired, thereby influencing public perception regarding the security and safety of LLMs. In this study, we advocate the utilization of perplexity as one of the means to recognize such potential attacks. The underlying concept behind these hacks revolves around appending an unusually constructed string of text to a harmful query that would otherwise be blocked. This maneuver confuses the protective mechanisms and tricks the model into generating a forbidden response. Such scenarios could result in providing detailed instructions to a malicious user for constructing explosives or orchestrating a bank heist. Our investigation demonstrates the feasibility of employing perplexity, a prevalent natural language processing metric, to detect these adversarial tactics before generating a forbidden response. By evaluating the perplexity of queries with and without such adversarial suffixes using an open-source LLM, we discovered that nearly 90 percent were above a perplexity of 1000. This contrast underscores the efficacy of perplexity for detecting this type of exploit.

摘要: 出现了一种涉及大型语言模型(LLM)的新黑客攻击，利用敌意后缀欺骗模型生成危险的响应。这一方法引起了《纽约时报》和《连线》等知名媒体的极大关注，从而影响了公众对低地小武器安全性和安全性的看法。在这项研究中，我们主张使用困惑作为识别这种潜在攻击的手段之一。这些黑客攻击背后的基本概念围绕着在原本会被阻止的有害查询中附加一个构造异常的文本字符串。这种操作混淆了保护机制，并诱使模型产生禁止反应。这种情况可能会导致向恶意用户提供制造炸药或策划银行抢劫的详细说明。我们的研究证明了使用困惑，一种流行的自然语言处理度量，在生成禁止响应之前检测这些敌对策略的可行性。通过使用开源LLM评估带有和不带有这种敌意后缀的查询的困惑程度，我们发现近90%的查询困惑程度高于1000。这种对比突出了困惑在检测这种类型的利用方面的有效性。



## **6. A Survey of Safety and Trustworthiness of Large Language Models through the Lens of Verification and Validation**

从验证和验证的角度考察大型语言模型的安全性和可信性 cs.AI

**SubmitDate**: 2023-08-27    [abs](http://arxiv.org/abs/2305.11391v2) [paper-pdf](http://arxiv.org/pdf/2305.11391v2)

**Authors**: Xiaowei Huang, Wenjie Ruan, Wei Huang, Gaojie Jin, Yi Dong, Changshun Wu, Saddek Bensalem, Ronghui Mu, Yi Qi, Xingyu Zhao, Kaiwen Cai, Yanghao Zhang, Sihao Wu, Peipei Xu, Dengyu Wu, Andre Freitas, Mustafa A. Mustafa

**Abstract**: Large Language Models (LLMs) have exploded a new heatwave of AI for their ability to engage end-users in human-level conversations with detailed and articulate answers across many knowledge domains. In response to their fast adoption in many industrial applications, this survey concerns their safety and trustworthiness. First, we review known vulnerabilities and limitations of the LLMs, categorising them into inherent issues, attacks, and unintended bugs. Then, we consider if and how the Verification and Validation (V&V) techniques, which have been widely developed for traditional software and deep learning models such as convolutional neural networks as independent processes to check the alignment of their implementations against the specifications, can be integrated and further extended throughout the lifecycle of the LLMs to provide rigorous analysis to the safety and trustworthiness of LLMs and their applications. Specifically, we consider four complementary techniques: falsification and evaluation, verification, runtime monitoring, and regulations and ethical use. In total, 370+ references are considered to support the quick understanding of the safety and trustworthiness issues from the perspective of V&V. While intensive research has been conducted to identify the safety and trustworthiness issues, rigorous yet practical methods are called for to ensure the alignment of LLMs with safety and trustworthiness requirements.

摘要: 大型语言模型(LLM)掀起了人工智能的新热潮，因为它们能够让最终用户参与人类级别的对话，并在许多知识领域提供详细而清晰的答案。为了回应它们在许多工业应用中的快速采用，这项调查关注它们的安全性和可信度。首先，我们回顾了LLMS的已知漏洞和限制，将它们归类为固有问题、攻击和意外错误。然后，我们考虑验证和确认(V&V)技术是否以及如何在LLMS的整个生命周期中被集成和进一步扩展，以对LLMS及其应用的安全性和可信性提供严格的分析。具体地说，我们考虑了四种互补的技术：证伪和评估、验证、运行时监控以及法规和道德使用。总共有370多篇参考文献被认为有助于从V&V的角度快速理解安全和可信度问题。虽然已经进行了深入的研究来确定安全和可信度问题，但需要严格而实用的方法来确保低成本管理与安全和可信性要求保持一致。



## **7. LMSanitator: Defending Prompt-Tuning Against Task-Agnostic Backdoors**

LMSanitator：防御与任务无关的后门的提示调整 cs.CL

To Appear in the Network and Distributed System Security (NDSS)  Symposium 2024, 26 February - 1 March 2024, San Diego, CA, USA

**SubmitDate**: 2023-08-26    [abs](http://arxiv.org/abs/2308.13904v1) [paper-pdf](http://arxiv.org/pdf/2308.13904v1)

**Authors**: Chengkun Wei, Wenlong Meng, Zhikun Zhang, Min Chen, Minghu Zhao, Wenjing Fang, Lei Wang, Zihui Zhang, Wenzhi Chen

**Abstract**: Prompt-tuning has emerged as an attractive paradigm for deploying large-scale language models due to its strong downstream task performance and efficient multitask serving ability. Despite its wide adoption, we empirically show that prompt-tuning is vulnerable to downstream task-agnostic backdoors, which reside in the pretrained models and can affect arbitrary downstream tasks. The state-of-the-art backdoor detection approaches cannot defend against task-agnostic backdoors since they hardly converge in reversing the backdoor triggers. To address this issue, we propose LMSanitator, a novel approach for detecting and removing task-agnostic backdoors on Transformer models. Instead of directly inversing the triggers, LMSanitator aims to inverse the predefined attack vectors (pretrained models' output when the input is embedded with triggers) of the task-agnostic backdoors, which achieves much better convergence performance and backdoor detection accuracy. LMSanitator further leverages prompt-tuning's property of freezing the pretrained model to perform accurate and fast output monitoring and input purging during the inference phase. Extensive experiments on multiple language models and NLP tasks illustrate the effectiveness of LMSanitator. For instance, LMSanitator achieves 92.8% backdoor detection accuracy on 960 models and decreases the attack success rate to less than 1% in most scenarios.

摘要: 由于其强大的下游任务性能和高效的多任务服务能力，即时调优已成为部署大规模语言模型的一个有吸引力的范例。尽管被广泛采用，我们的经验表明，即时调优很容易受到下游任务不可知的后门的影响，这些后门驻留在预先训练的模型中，可以影响任意的下游任务。最先进的后门检测方法无法防御与任务无关的后门，因为它们在逆转后门触发时几乎不会收敛。为了解决这个问题，我们提出了一种新的方法LMSanitator，用于检测和删除变压器模型上与任务无关的后门程序。LMSanitator不是直接对触发器求逆，而是对与任务无关的后门的预定义攻击向量(当输入嵌入触发器时，预先训练的模型的输出)求逆，从而获得更好的收敛性能和后门检测精度。LMSanitator还利用即时调整的冻结预训练模型的特性，在推理阶段执行准确而快速的输出监控和输入清除。在多种语言模型和自然语言处理任务上的大量实验表明了LMSanitator的有效性。例如，LMSanitator在960个机型上的后门检测准确率达到92.8%，在大多数场景下攻击成功率低于1%。



## **8. Self-Deception: Reverse Penetrating the Semantic Firewall of Large Language Models**

自欺欺人：反向穿透大型语言模型的语义防火墙 cs.CL

Serious errors were found in the experiment, which may lead to the  overturning of the overall conclusions of the paper

**SubmitDate**: 2023-08-25    [abs](http://arxiv.org/abs/2308.11521v2) [paper-pdf](http://arxiv.org/pdf/2308.11521v2)

**Authors**: Zhenhua Wang, Wei Xie, Kai Chen, Baosheng Wang, Zhiwen Gui, Enze Wang

**Abstract**: Large language models (LLMs), such as ChatGPT, have emerged with astonishing capabilities approaching artificial general intelligence. While providing convenience for various societal needs, LLMs have also lowered the cost of generating harmful content. Consequently, LLM developers have deployed semantic-level defenses to recognize and reject prompts that may lead to inappropriate content. Unfortunately, these defenses are not foolproof, and some attackers have crafted "jailbreak" prompts that temporarily hypnotize the LLM into forgetting content defense rules and answering any improper questions. To date, there is no clear explanation of the principles behind these semantic-level attacks and defenses in both industry and academia.   This paper investigates the LLM jailbreak problem and proposes an automatic jailbreak method for the first time. We propose the concept of a semantic firewall and provide three technical implementation approaches. Inspired by the attack that penetrates traditional firewalls through reverse tunnels, we introduce a "self-deception" attack that can bypass the semantic firewall by inducing LLM to generate prompts that facilitate jailbreak. We generated a total of 2,520 attack payloads in six languages (English, Russian, French, Spanish, Chinese, and Arabic) across seven virtual scenarios, targeting the three most common types of violations: violence, hate, and pornography. The experiment was conducted on two models, namely the GPT-3.5-Turbo and GPT-4. The success rates on the two models were 86.2% and 67%, while the failure rates were 4.7% and 2.2%, respectively. This highlighted the effectiveness of the proposed attack method. All experimental code and raw data will be released as open-source to inspire future research. We believe that manipulating AI behavior through carefully crafted prompts will become an important research direction in the future.

摘要: 像ChatGPT这样的大型语言模型(LLM)已经出现，具有接近人工通用智能的惊人能力。在为各种社会需求提供便利的同时，LLMS还降低了产生有害内容的成本。因此，LLM开发人员部署了语义级防御，以识别和拒绝可能导致不适当内容的提示。不幸的是，这些防御措施并不是万无一失的，一些攻击者精心制作了“越狱”提示，暂时催眠LLM忘记内容防御规则并回答任何不恰当的问题。到目前为止，工业界和学术界都没有对这些语义级攻击和防御背后的原理做出明确的解释。本文对LLM越狱问题进行了研究，首次提出了一种自动越狱方法。提出了语义防火墙的概念，并给出了三种技术实现方法。受通过反向隧道穿透传统防火墙的攻击的启发，我们引入了一种通过诱导LLM生成便于越狱的提示来绕过语义防火墙的自欺式攻击。我们在七个虚拟场景中生成了六种语言(英语、俄语、法语、西班牙语、汉语和阿拉伯语)的总计2,520个攻击有效负载，目标是三种最常见的违规行为：暴力、仇恨和色情。实验在两种型号上进行，即GPT-3.5-Turbo和GPT-4。两种模型的成功率分别为86.2%和67%，失败率分别为4.7%和2.2%。这突显了拟议攻击方法的有效性。所有实验代码和原始数据将以开源形式发布，以激励未来的研究。我们认为，通过精心制作的提示来操纵AI行为将成为未来的一个重要研究方向。



## **9. PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts**

PromptBitch：评估大型语言模型在对抗性提示下的稳健性 cs.CL

Technical report; updated with new experiments and related work; 27  pages; code is at: https://github.com/microsoft/promptbench

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2306.04528v3) [paper-pdf](http://arxiv.org/pdf/2306.04528v3)

**Authors**: Kaijie Zhu, Jindong Wang, Jiaheng Zhou, Zichen Wang, Hao Chen, Yidong Wang, Linyi Yang, Wei Ye, Neil Zhenqiang Gong, Yue Zhang, Xing Xie

**Abstract**: The increasing reliance on Large Language Models (LLMs) across academia and industry necessitates a comprehensive understanding of their robustness to prompts. In response to this vital need, we introduce PromptBench, a robustness benchmark designed to measure LLMs' resilience to adversarial prompts. This study uses a plethora of adversarial textual attacks targeting prompts across multiple levels: character, word, sentence, and semantic. These prompts are then employed in diverse tasks, such as sentiment analysis, natural language inference, reading comprehension, machine translation, and math problem-solving. Our study generates 4,032 adversarial prompts, meticulously evaluated over 8 tasks and 13 datasets, with 567,084 test samples in total. Our findings demonstrate that contemporary LLMs are vulnerable to adversarial prompts. Furthermore, we present comprehensive analysis to understand the mystery behind prompt robustness and its transferability. We then offer insightful robustness analysis and pragmatic recommendations for prompt composition, beneficial to both researchers and everyday users. We make our code, prompts, and methodologies to generate adversarial prompts publicly accessible, thereby enabling and encouraging collaborative exploration in this pivotal field: https://github.com/microsoft/promptbench.

摘要: 学术界和工业界对大型语言模型(LLM)的依赖日益增加，这就要求我们必须全面了解它们对提示的稳健性。为了响应这一关键需求，我们引入了PromptBtch，这是一个健壮性基准，旨在衡量LLMS对敌意提示的弹性。这项研究使用了过多的对抗性文本攻击，目标是多个层面的提示：字符、单词、句子和语义。这些提示随后被用于不同的任务，如情感分析、自然语言推理、阅读理解、机器翻译和数学解题。我们的研究产生了4032个对抗性提示，仔细评估了8个任务和13个数据集，总共有567,084个测试样本。我们的研究结果表明，当代的LLM容易受到对抗性提示的影响。此外，我们还提供了全面的分析，以了解即时健壮性及其可转移性背后的奥秘。然后，我们提供了有洞察力的健壮性分析和实用的即时撰写建议，这对研究人员和日常用户都是有益的。我们将生成对抗性提示的代码、提示和方法公之于众，从而支持并鼓励在这个关键领域进行协作探索：https://github.com/microsoft/promptbench.



## **10. On the Uses of Large Language Models to Interpret Ambiguous Cyberattack Descriptions**

关于使用大型语言模型来解释模糊的网络攻击描述 cs.AI

**SubmitDate**: 2023-08-22    [abs](http://arxiv.org/abs/2306.14062v2) [paper-pdf](http://arxiv.org/pdf/2306.14062v2)

**Authors**: Reza Fayyazi, Shanchieh Jay Yang

**Abstract**: The volume, variety, and velocity of change in vulnerabilities and exploits have made incident threat analysis challenging with human expertise and experience along. Tactics, Techniques, and Procedures (TTPs) are to describe how and why attackers exploit vulnerabilities. However, a TTP description written by one security professional can be interpreted very differently by another, leading to confusion in cybersecurity operations or even business, policy, and legal decisions. Meanwhile, advancements in AI have led to the increasing use of Natural Language Processing (NLP) algorithms to assist the various tasks in cyber operations. With the rise of Large Language Models (LLMs), NLP tasks have significantly improved because of the LLM's semantic understanding and scalability. This leads us to question how well LLMs can interpret TTPs or general cyberattack descriptions to inform analysts of the intended purposes of cyberattacks. We propose to analyze and compare the direct use of LLMs (e.g., GPT-3.5) versus supervised fine-tuning (SFT) of small-scale-LLMs (e.g., BERT) to study their capabilities in predicting ATT&CK tactics. Our results reveal that the small-scale-LLMs with SFT provide a more focused and clearer differentiation between the ATT&CK tactics (if such differentiation exists). On the other hand, direct use of LLMs offer a broader interpretation of cyberattack techniques. When treating more general cases, despite the power of LLMs, inherent ambiguity exists and limits their predictive power. We then summarize the challenges and recommend research directions on LLMs to treat the inherent ambiguity of TTP descriptions used in various cyber operations.

摘要: 漏洞和漏洞的数量、种类和变化速度使事件威胁分析具有挑战性，需要人类的专业知识和经验。战术、技术和过程(TTP)描述攻击者如何以及为什么利用漏洞。然而，一个安全专业人员编写的TTP描述可能会被另一个安全专业人员解释得非常不同，从而导致网络安全运营甚至业务、政策和法律决策中的混乱。同时，人工智能的进步导致越来越多地使用自然语言处理(NLP)算法来辅助网络操作中的各种任务。随着大型语言模型(LLM)的兴起，由于LLM的语义理解和可扩展性，NLP任务得到了显著改进。这让我们质疑LLM能否很好地解释TTP或一般网络攻击描述，以告知分析师网络攻击的预期目的。我们建议分析和比较直接使用LLMS(例如GPT-3.5)和小规模LLMS的监督微调(SFT)(例如BERT)来研究它们对ATT和CK战术的预测能力。我们的结果表明，带有SFT的小规模LLMS在ATT和CK策略之间提供了更有针对性和更清晰的区别(如果存在这种区别)。另一方面，直接使用LLMS提供了对网络攻击技术的更广泛的解释。在处理更一般的病例时，尽管LLMS的能力很强，但固有的模糊性存在，并限制了它们的预测能力。然后，我们总结了在LLMS上的挑战和建议的研究方向，以处理在各种网络操作中使用的TTP描述的固有的歧义。



## **11. Adversarial Attacks on Code Models with Discriminative Graph Patterns**

具有区分图模式的代码模型的对抗性攻击 cs.SE

**SubmitDate**: 2023-08-22    [abs](http://arxiv.org/abs/2308.11161v1) [paper-pdf](http://arxiv.org/pdf/2308.11161v1)

**Authors**: Thanh-Dat Nguyen, Yang Zhou, Xuan Bach D. Le, Patanamon, Thongtanunam, David Lo

**Abstract**: Pre-trained language models of code are now widely used in various software engineering tasks such as code generation, code completion, vulnerability detection, etc. This, in turn, poses security and reliability risks to these models. One of the important threats is \textit{adversarial attacks}, which can lead to erroneous predictions and largely affect model performance on downstream tasks. Current adversarial attacks on code models usually adopt fixed sets of program transformations, such as variable renaming and dead code insertion, leading to limited attack effectiveness. To address the aforementioned challenges, we propose a novel adversarial attack framework, GraphCodeAttack, to better evaluate the robustness of code models. Given a target code model, GraphCodeAttack automatically mines important code patterns, which can influence the model's decisions, to perturb the structure of input code to the model. To do so, GraphCodeAttack uses a set of input source codes to probe the model's outputs and identifies the \textit{discriminative} ASTs patterns that can influence the model decisions. GraphCodeAttack then selects appropriate AST patterns, concretizes the selected patterns as attacks, and inserts them as dead code into the model's input program. To effectively synthesize attacks from AST patterns, GraphCodeAttack uses a separate pre-trained code model to fill in the ASTs with concrete code snippets. We evaluate the robustness of two popular code models (e.g., CodeBERT and GraphCodeBERT) against our proposed approach on three tasks: Authorship Attribution, Vulnerability Prediction, and Clone Detection. The experimental results suggest that our proposed approach significantly outperforms state-of-the-art approaches in attacking code models such as CARROT and ALERT.

摘要: 预先训练的代码语言模型现在被广泛用于各种软件工程任务，如代码生成、代码完成、漏洞检测等。这反过来又给这些模型带来了安全和可靠性风险。其中一个重要的威胁是对抗性攻击，它会导致错误的预测，并在很大程度上影响模型在下游任务上的性能。当前针对代码模型的对抗性攻击通常采用固定的程序转换集，如变量重命名和死代码插入，导致攻击效果有限。为了应对上述挑战，我们提出了一种新的对抗性攻击框架GraphCodeAttack，以更好地评估代码模型的健壮性。在给定目标代码模型的情况下，GraphCodeAttack自动挖掘可能影响模型决策的重要代码模式，以扰乱模型的输入代码结构。为此，GraphCodeAttack使用一组输入源代码来探测模型的输出，并识别可能影响模型决策的\textit{鉴别性}ASTS模式。然后，GraphCodeAttack选择适当的AST模式，将所选模式具体化为攻击，并将它们作为死代码插入到模型的输入程序中。为了有效地从AST模式合成攻击，GraphCodeAttack使用单独的预先训练的代码模型来用具体的代码片段填充AST。我们评估了两个流行的代码模型(例如，CodeBERT和GraphCodeBERT)在作者属性、漏洞预测和克隆检测三个任务上的健壮性。实验结果表明，我们提出的方法在攻击胡萝卜和ALERT等代码模型方面明显优于最先进的方法。



## **12. TrojText: Test-time Invisible Textual Trojan Insertion**

TrojText：测试时间隐形文本特洛伊木马插入 cs.CL

In The Eleventh International Conference on Learning Representations.  2023 (ICLR 2023)

**SubmitDate**: 2023-08-22    [abs](http://arxiv.org/abs/2303.02242v2) [paper-pdf](http://arxiv.org/pdf/2303.02242v2)

**Authors**: Qian Lou, Yepeng Liu, Bo Feng

**Abstract**: In Natural Language Processing (NLP), intelligent neuron models can be susceptible to textual Trojan attacks. Such attacks occur when Trojan models behave normally for standard inputs but generate malicious output for inputs that contain a specific trigger. Syntactic-structure triggers, which are invisible, are becoming more popular for Trojan attacks because they are difficult to detect and defend against. However, these types of attacks require a large corpus of training data to generate poisoned samples with the necessary syntactic structures for Trojan insertion. Obtaining such data can be difficult for attackers, and the process of generating syntactic poisoned triggers and inserting Trojans can be time-consuming. This paper proposes a solution called TrojText, which aims to determine whether invisible textual Trojan attacks can be performed more efficiently and cost-effectively without training data. The proposed approach, called the Representation-Logit Trojan Insertion (RLI) algorithm, uses smaller sampled test data instead of large training data to achieve the desired attack. The paper also introduces two additional techniques, namely the accumulated gradient ranking (AGR) and Trojan Weights Pruning (TWP), to reduce the number of tuned parameters and the attack overhead. The TrojText approach was evaluated on three datasets (AG's News, SST-2, and OLID) using three NLP models (BERT, XLNet, and DeBERTa). The experiments demonstrated that the TrojText approach achieved a 98.35\% classification accuracy for test sentences in the target class on the BERT model for the AG's News dataset. The source code for TrojText is available at https://github.com/UCF-ML-Research/TrojText.

摘要: 在自然语言处理(NLP)中，智能神经元模型容易受到文本特洛伊木马的攻击。当特洛伊木马模型对标准输入行为正常，但对包含特定触发器的输入生成恶意输出时，就会发生此类攻击。语法结构触发器是看不见的，因为它们很难检测和防御，所以越来越多地用于特洛伊木马攻击。然而，这些类型的攻击需要大量的训练数据来生成具有必要句法结构的有毒样本，以便插入特洛伊木马。对于攻击者来说，获取这类数据可能很困难，生成语法有毒的触发器和插入特洛伊木马的过程可能会很耗时。本文提出了一种称为TrojText的解决方案，旨在确定在没有训练数据的情况下，是否可以更高效、更经济地执行隐形文本特洛伊木马攻击。该方法称为表示-Logit木马插入(RLI)算法，使用较小的采样测试数据而不是大的训练数据来实现期望的攻击。文中还引入了两种额外的技术，即累积梯度排序(AGR)和木马权重剪枝(TWP)，以减少调整参数的数量和攻击开销。TrojText方法在三个数据集(AG的News、SST-2和OLID)上使用三个NLP模型(BERT、XLNet和DeBERTa)进行了评估。实验表明，在AG新闻数据集的BERT模型上，TrojText方法对目标类测试句子的分类准确率达到了98.35。特洛伊文本的源代码可在https://github.com/UCF-ML-Research/TrojText.上找到



## **13. Temporal-Distributed Backdoor Attack Against Video Based Action Recognition**

针对视频动作识别的时间分布式后门攻击 cs.CV

**SubmitDate**: 2023-08-21    [abs](http://arxiv.org/abs/2308.11070v1) [paper-pdf](http://arxiv.org/pdf/2308.11070v1)

**Authors**: Xi Li, Songhe Wang, Ruiquan Huang, Mahanth Gowda, George Kesidis

**Abstract**: Deep neural networks (DNNs) have achieved tremendous success in various applications including video action recognition, yet remain vulnerable to backdoor attacks (Trojans). The backdoor-compromised model will mis-classify to the target class chosen by the attacker when a test instance (from a non-target class) is embedded with a specific trigger, while maintaining high accuracy on attack-free instances. Although there are extensive studies on backdoor attacks against image data, the susceptibility of video-based systems under backdoor attacks remains largely unexplored. Current studies are direct extensions of approaches proposed for image data, e.g., the triggers are \textbf{independently} embedded within the frames, which tend to be detectable by existing defenses. In this paper, we introduce a \textit{simple} yet \textit{effective} backdoor attack against video data. Our proposed attack, adding perturbations in a transformed domain, plants an \textbf{imperceptible, temporally distributed} trigger across the video frames, and is shown to be resilient to existing defensive strategies. The effectiveness of the proposed attack is demonstrated by extensive experiments with various well-known models on two video recognition benchmarks, UCF101 and HMDB51, and a sign language recognition benchmark, Greek Sign Language (GSL) dataset. We delve into the impact of several influential factors on our proposed attack and identify an intriguing effect termed "collateral damage" through extensive studies.

摘要: 深度神经网络(DNN)在包括视频动作识别在内的各种应用中取得了巨大的成功，但仍然容易受到后门攻击(特洛伊木马)。当测试实例(来自非目标类)嵌入特定触发器时，后门泄露模型将被错误分类为攻击者选择的目标类，同时保持对无攻击实例的高准确性。虽然已经有大量关于针对图像数据的后门攻击的研究，但基于视频的系统在后门攻击下的易感性在很大程度上仍未被探索。目前的研究是为图像数据提出的方法的直接扩展，例如，触发器嵌入在帧中，这往往是现有防御系统可以检测到的。本文介绍了一种针对视频数据的简单而有效的后门攻击。我们提出的攻击在变换的域中添加了扰动，在视频帧中植入了一个不可感知的、时间分布的触发器，并被证明对现有的防御策略具有弹性。在两个视频识别基准UCF101和HMDB51和一个手语识别基准希腊手语(GSL)数据集上进行了大量的实验，证明了所提出的攻击的有效性。我们深入研究了几个影响因素对我们提出的攻击的影响，并通过广泛的研究确定了一种有趣的影响，称为“附带损害”。



## **14. RatGPT: Turning online LLMs into Proxies for Malware Attacks**

RatGPT：将在线LLM转变为恶意软件攻击的代理 cs.CR

**SubmitDate**: 2023-08-17    [abs](http://arxiv.org/abs/2308.09183v1) [paper-pdf](http://arxiv.org/pdf/2308.09183v1)

**Authors**: Mika Beckerich, Laura Plein, Sergio Coronado

**Abstract**: The evolution of Generative AI and the capabilities of the newly released Large Language Models (LLMs) open new opportunities in software engineering. However, they also lead to new challenges in cybersecurity. Recently, researchers have shown the possibilities of using LLMs such as ChatGPT to generate malicious content that can directly be exploited or guide inexperienced hackers to weaponize tools and code. Those studies covered scenarios that still require the attacker in the middle of the loop. In this study, we leverage openly available plugins and use an LLM as proxy between the attacker and the victim. We deliver a proof-of-concept where ChatGPT is used for the dissemination of malicious software while evading detection, alongside establishing the communication to a command and control (C2) server to receive commands to interact with a victim's system. Finally, we present the general approach as well as essential elements in order to stay undetected and make the attack a success. This proof-of-concept highlights significant cybersecurity issues with openly available plugins and LLMs, which require the development of security guidelines, controls, and mitigation strategies.

摘要: 产生式人工智能的发展和新发布的大型语言模型(LLM)的能力为软件工程打开了新的机遇。然而，它们也给网络安全带来了新的挑战。最近，研究人员展示了使用ChatGPT等LLMS生成可直接利用的恶意内容或引导缺乏经验的黑客将工具和代码武器化的可能性。这些研究涵盖了仍然需要攻击者处于循环中间的场景。在这项研究中，我们利用开放可用的插件，并使用LLM作为攻击者和受害者之间的代理。我们提供了一个概念验证，其中ChatGPT用于传播恶意软件，同时逃避检测，同时建立与命令和控制(C2)服务器的通信，以接收与受害者系统交互的命令。最后，我们介绍了一般方法以及基本要素，以保持不被发现，使攻击成功。这一概念验证突出了开放可用插件和LLM存在的重大网络安全问题，这些问题需要制定安全指南、控制和缓解策略。



## **15. Getting pwn'd by AI: Penetration Testing with Large Language Models**

被人工智能淘汰：用大型语言模型进行渗透测试 cs.CL

**SubmitDate**: 2023-08-17    [abs](http://arxiv.org/abs/2308.00121v3) [paper-pdf](http://arxiv.org/pdf/2308.00121v3)

**Authors**: Andreas Happe, Jürgen Cito

**Abstract**: The field of software security testing, more specifically penetration testing, is an activity that requires high levels of expertise and involves many manual testing and analysis steps. This paper explores the potential usage of large-language models, such as GPT3.5, to augment penetration testers with AI sparring partners. We explore the feasibility of supplementing penetration testers with AI models for two distinct use cases: high-level task planning for security testing assignments and low-level vulnerability hunting within a vulnerable virtual machine. For the latter, we implemented a closed-feedback loop between LLM-generated low-level actions with a vulnerable virtual machine (connected through SSH) and allowed the LLM to analyze the machine state for vulnerabilities and suggest concrete attack vectors which were automatically executed within the virtual machine. We discuss promising initial results, detail avenues for improvement, and close deliberating on the ethics of providing AI-based sparring partners.

摘要: 软件安全测试领域，更具体地说是渗透测试，是一项需要高水平专业知识的活动，涉及许多手动测试和分析步骤。本文探索了大语言模型的潜在用途，如GPT3.5，以增强渗透测试员与人工智能陪练。我们针对两个不同的用例探讨了用AI模型补充渗透测试器的可行性：安全测试任务的高级任务规划和易受攻击的虚拟机中的低级别漏洞搜索。对于后者，我们实现了LLM生成的底层操作与易受攻击的虚拟机(通过SSH连接)之间的闭环反馈，并允许LLM分析机器状态的漏洞并建议在虚拟机内自动执行的具体攻击向量。我们讨论了有希望的初步结果，详细说明了改进的途径，并仔细审议了提供基于人工智能的陪练的道德问题。



## **16. Do you really follow me? Adversarial Instructions for Evaluating the Robustness of Large Language Models**

你真的听懂我的话吗？评估大型语言模型稳健性的对抗性说明 cs.CL

Work in progress

**SubmitDate**: 2023-08-17    [abs](http://arxiv.org/abs/2308.10819v1) [paper-pdf](http://arxiv.org/pdf/2308.10819v1)

**Authors**: Zekun Li, Baolin Peng, Pengcheng He, Xifeng Yan

**Abstract**: Large Language Models (LLMs) have shown remarkable proficiency in following instructions, making them valuable in customer-facing applications. However, their impressive capabilities also raise concerns about the amplification of risks posed by adversarial instructions, which can be injected into the model input by third-party attackers to manipulate LLMs' original instructions and prompt unintended actions and content. Therefore, it is crucial to understand LLMs' ability to accurately discern which instructions to follow to ensure their safe deployment in real-world scenarios. In this paper, we propose a pioneering benchmark for automatically evaluating the robustness of LLMs against adversarial instructions. The objective of this benchmark is to quantify the extent to which LLMs are influenced by injected adversarial instructions and assess their ability to differentiate between these adversarial instructions and original user instructions. Through experiments conducted with state-of-the-art instruction-following LLMs, we uncover significant limitations in their robustness against adversarial instruction attacks. Furthermore, our findings indicate that prevalent instruction-tuned models are prone to being overfitted to follow any instruction phrase in the prompt without truly understanding which instructions should be followed. This highlights the need to address the challenge of training models to comprehend prompts instead of merely following instruction phrases and completing the text.

摘要: 大型语言模型(LLM)在遵循说明方面表现出非凡的熟练程度，这使它们在面向客户的应用程序中具有价值。然而，它们令人印象深刻的能力也引发了人们对对抗性指令带来的风险放大的担忧，这些指令可以被注入第三方攻击者输入的模型中，以操纵LLMS的原始指令并提示意外的操作和内容。因此，了解LLMS准确识别应遵循哪些指令以确保在现实世界场景中安全部署的能力至关重要。在本文中，我们提出了一个开创性的基准，用于自动评估LLMS对恶意指令的健壮性。这一基准的目的是量化LLM受到注入的敌意指令的影响程度，并评估它们区分这些敌意指令和原始用户指令的能力。通过使用最先进的指令跟随LLM进行的实验，我们发现它们在抵抗对抗性指令攻击方面的健壮性方面存在显著的局限性。此外，我们的发现表明，流行的教学调整模型容易过度适应于遵循提示中的任何指令短语，而不是真正理解应该遵循哪些指令。这突出表明，需要解决培训模型理解提示的挑战，而不是仅仅遵循指导短语和完成正文。



## **17. Visual Adversarial Examples Jailbreak Aligned Large Language Models**

视觉对抗性示例越狱对齐大型语言模型 cs.CR

**SubmitDate**: 2023-08-16    [abs](http://arxiv.org/abs/2306.13213v2) [paper-pdf](http://arxiv.org/pdf/2306.13213v2)

**Authors**: Xiangyu Qi, Kaixuan Huang, Ashwinee Panda, Peter Henderson, Mengdi Wang, Prateek Mittal

**Abstract**: Recently, there has been a surge of interest in integrating vision into Large Language Models (LLMs), exemplified by Visual Language Models (VLMs) such as Flamingo and GPT-4. This paper sheds light on the security and safety implications of this trend. First, we underscore that the continuous and high-dimensional nature of the visual input makes it a weak link against adversarial attacks, representing an expanded attack surface of vision-integrated LLMs. Second, we highlight that the versatility of LLMs also presents visual attackers with a wider array of achievable adversarial objectives, extending the implications of security failures beyond mere misclassification. As an illustration, we present a case study in which we exploit visual adversarial examples to circumvent the safety guardrail of aligned LLMs with integrated vision. Intriguingly, we discover that a single visual adversarial example can universally jailbreak an aligned LLM, compelling it to heed a wide range of harmful instructions that it otherwise would not) and generate harmful content that transcends the narrow scope of a `few-shot' derogatory corpus initially employed to optimize the adversarial example. Our study underscores the escalating adversarial risks associated with the pursuit of multimodality. Our findings also connect the long-studied adversarial vulnerabilities of neural networks to the nascent field of AI alignment. The presented attack suggests a fundamental adversarial challenge for AI alignment, especially in light of the emerging trend toward multimodality in frontier foundation models.

摘要: 最近，人们对将视觉集成到大型语言模型(LLM)中的兴趣激增，例如Flamingo和GPT-4等视觉语言模型(VLM)。本文阐明了这一趋势对安全和安全的影响。首先，我们强调，视觉输入的连续性和高维性使其成为对抗对抗性攻击的薄弱环节，代表了视觉集成LLMS的扩展攻击面。其次，我们强调，LLMS的多功能性还为视觉攻击者提供了更广泛的可实现的对抗性目标，将安全故障的影响扩大到仅仅是错误分类。作为说明，我们给出了一个案例研究，在这个案例中，我们利用视觉对抗性例子来绕过具有集成视觉的对准LLM的安全护栏。有趣的是，我们发现，一个单一的视觉对抗性例子可以普遍地越狱，迫使其注意一系列有害的指示，否则它不会这样做)，并产生有害内容，这些内容超出了最初用来优化对抗性例子的“几次”贬损语料库的狭窄范围。我们的研究强调了与追求多模式相关的不断升级的对抗风险。我们的发现还将长期研究的神经网络的对抗性漏洞与新兴的人工智能对齐领域联系起来。目前的攻击表明，人工智能对齐提出了一个根本性的对抗性挑战，特别是考虑到前沿基础模型中出现的多模式趋势。



## **18. From Prompt Injections to SQL Injection Attacks: How Protected is Your LLM-Integrated Web Application?**

从即时注入到SQL注入攻击：LLM集成的Web应用程序受到了怎样的保护？ cs.CR

12 pages, 3 figures, 3 tables, 5 listings

**SubmitDate**: 2023-08-15    [abs](http://arxiv.org/abs/2308.01990v3) [paper-pdf](http://arxiv.org/pdf/2308.01990v3)

**Authors**: Rodrigo Pedro, Daniel Castro, Paulo Carreira, Nuno Santos

**Abstract**: Large Language Models (LLMs) have found widespread applications in various domains, including web applications, where they facilitate human interaction via chatbots with natural language interfaces. Internally, aided by an LLM-integration middleware such as Langchain, user prompts are translated into SQL queries used by the LLM to provide meaningful responses to users. However, unsanitized user prompts can lead to SQL injection attacks, potentially compromising the security of the database. Despite the growing interest in prompt injection vulnerabilities targeting LLMs, the specific risks of generating SQL injection attacks through prompt injections have not been extensively studied. In this paper, we present a comprehensive examination of prompt-to-SQL (P$_2$SQL) injections targeting web applications based on the Langchain framework. Using Langchain as our case study, we characterize P$_2$SQL injections, exploring their variants and impact on application security through multiple concrete examples. Furthermore, we evaluate 7 state-of-the-art LLMs, demonstrating the pervasiveness of P$_2$SQL attacks across language models. Our findings indicate that LLM-integrated applications based on Langchain are highly susceptible to P$_2$SQL injection attacks, warranting the adoption of robust defenses. To counter these attacks, we propose four effective defense techniques that can be integrated as extensions to the Langchain framework. We validate the defenses through an experimental evaluation with a real-world use case application.

摘要: 大型语言模型在包括网络应用在内的各个领域得到了广泛的应用，在这些领域中，它们通过带有自然语言界面的聊天机器人来促进人类交互。在内部，在LLM集成中间件(如Langchain)的帮助下，用户提示被转换为LLM使用的SQL查询，以向用户提供有意义的响应。但是，未经清理的用户提示可能会导致SQL注入攻击，从而可能危及数据库的安全性。尽管人们对针对LLM的即时注入漏洞的兴趣与日俱增，但通过即时注入生成SQL注入攻击的具体风险尚未得到广泛研究。在这篇文章中，我们提出了一个全面的审查，以快速到SQL(P$2$SQL)注入针对Web应用程序基于朗之链框架。以Lang Chain为例，我们描述了P$2$SQL注入的特征，通过多个具体实例探索了它们的变体及其对应用程序安全性的影响。此外，我们对7个最新的LLM进行了评估，证明了P$2$SQL攻击在语言模型中的普遍性。我们的研究结果表明，基于LLm集成的应用程序非常容易受到P$2$SQL注入攻击，因此需要采取健壮的防御措施。为了应对这些攻击，我们提出了四种有效的防御技术，这些技术可以作为Langchain框架的扩展集成在一起。我们通过使用真实世界的用例应用程序进行实验评估来验证防御措施。



## **19. Robustness Over Time: Understanding Adversarial Examples' Effectiveness on Longitudinal Versions of Large Language Models**

随时间变化的稳健性：理解对抗性例子在大型语言模型纵向版本上的有效性 cs.CR

**SubmitDate**: 2023-08-15    [abs](http://arxiv.org/abs/2308.07847v1) [paper-pdf](http://arxiv.org/pdf/2308.07847v1)

**Authors**: Yugeng Liu, Tianshuo Cong, Zhengyu Zhao, Michael Backes, Yun Shen, Yang Zhang

**Abstract**: Large Language Models (LLMs) have led to significant improvements in many tasks across various domains, such as code interpretation, response generation, and ambiguity handling. These LLMs, however, when upgrading, primarily prioritize enhancing user experience while neglecting security, privacy, and safety implications. Consequently, unintended vulnerabilities or biases can be introduced. Previous studies have predominantly focused on specific versions of the models and disregard the potential emergence of new attack vectors targeting the updated versions. Through the lens of adversarial examples within the in-context learning framework, this longitudinal study addresses this gap by conducting a comprehensive assessment of the robustness of successive versions of LLMs, vis-\`a-vis GPT-3.5. We conduct extensive experiments to analyze and understand the impact of the robustness in two distinct learning categories: zero-shot learning and few-shot learning. Our findings indicate that, in comparison to earlier versions of LLMs, the updated versions do not exhibit the anticipated level of robustness against adversarial attacks. In addition, our study emphasizes the increased effectiveness of synergized adversarial queries in most zero-shot learning and few-shot learning cases. We hope that our study can lead to a more refined assessment of the robustness of LLMs over time and provide valuable insights of these models for both developers and users.

摘要: 大型语言模型(LLM)在代码解释、响应生成和歧义处理等不同领域的许多任务中都得到了显著改进。然而，这些LLM在升级时主要优先考虑增强用户体验，而忽略了安全、隐私和安全方面的影响。因此，可能会引入意想不到的漏洞或偏见。以前的研究主要集中在模型的特定版本上，而忽略了针对更新版本的新攻击媒介的潜在出现。这项纵向研究通过在情景学习框架内的对抗性例子的镜头，通过对LLMS连续版本相对于GPT-3.5的稳健性进行全面评估来解决这一差距。我们进行了大量的实验来分析和理解健壮性在两个不同的学习类别中的影响：零镜头学习和少镜头学习。我们的发现表明，与早期版本的LLMS相比，更新后的版本没有表现出预期的对对手攻击的健壮性。此外，我们的研究还强调了协同对抗性询问在大多数零机会学习和少机会学习案例中的有效性。我们希望我们的研究能够随着时间的推移对LLMS的健壮性进行更精细的评估，并为开发人员和用户提供对这些模型的有价值的见解。



## **20. LLM Self Defense: By Self Examination, LLMs Know They Are Being Tricked**

LLM自卫：通过自我检查，LLM知道自己被骗了 cs.CL

**SubmitDate**: 2023-08-15    [abs](http://arxiv.org/abs/2308.07308v2) [paper-pdf](http://arxiv.org/pdf/2308.07308v2)

**Authors**: Alec Helbling, Mansi Phute, Matthew Hull, Duen Horng Chau

**Abstract**: Large language models (LLMs) have skyrocketed in popularity in recent years due to their ability to generate high-quality text in response to human prompting. However, these models have been shown to have the potential to generate harmful content in response to user prompting (e.g., giving users instructions on how to commit crimes). There has been a focus in the literature on mitigating these risks, through methods like aligning models with human values through reinforcement learning. However, it has been shown that even aligned language models are susceptible to adversarial attacks that bypass their restrictions on generating harmful text. We propose a simple approach to defending against these attacks by having a large language model filter its own responses. Our current results show that even if a model is not fine-tuned to be aligned with human values, it is possible to stop it from presenting harmful content to users by validating the content using a language model.

摘要: 近年来，大型语言模型(LLM)因其能够生成响应人类提示的高质量文本而迅速流行起来。然而，这些模式已被证明有可能响应用户提示(例如，向用户提供如何犯罪的指令)而生成有害内容。在文献中，有一个重点是通过增强学习使模型与人类价值观保持一致等方法来降低这些风险。然而，已经表明，即使是对齐的语言模型也容易受到敌意攻击，从而绕过它们对生成有害文本的限制。我们提出了一种简单的方法来防御这些攻击，方法是让大型语言模型过滤自己的响应。我们目前的结果表明，即使一个模型没有经过微调以与人类的价值观保持一致，也可以通过使用语言模型验证内容来阻止它向用户呈现有害内容。



## **21. S3C2 Summit 2023-06: Government Secure Supply Chain Summit**

S3C2峰会2023-06：政府安全供应链峰会 cs.CR

arXiv admin note: text overlap with arXiv:2307.16557,  arXiv:2307.15642

**SubmitDate**: 2023-08-13    [abs](http://arxiv.org/abs/2308.06850v1) [paper-pdf](http://arxiv.org/pdf/2308.06850v1)

**Authors**: William Enck, Yasemin Acar, Michel Cukier, Alexandros Kapravelos, Christian Kästner, Laurie Williams

**Abstract**: Recent years have shown increased cyber attacks targeting less secure elements in the software supply chain and causing fatal damage to businesses and organizations. Past well-known examples of software supply chain attacks are the SolarWinds or log4j incidents that have affected thousands of customers and businesses. The US government and industry are equally interested in enhancing software supply chain security. On June 7, 2023, researchers from the NSF-supported Secure Software Supply Chain Center (S3C2) conducted a Secure Software Supply Chain Summit with a diverse set of 17 practitioners from 13 government agencies. The goal of the Summit was two-fold: (1) to share our observations from our previous two summits with industry, and (2) to enable sharing between individuals at the government agencies regarding practical experiences and challenges with software supply chain security. For each discussion topic, we presented our observations and take-aways from the industry summits to spur conversation. We specifically focused on the Executive Order 14028, software bill of materials (SBOMs), choosing new dependencies, provenance and self-attestation, and large language models. The open discussions enabled mutual sharing and shed light on common challenges that government agencies see as impacting government and industry practitioners when securing their software supply chain. In this paper, we provide a summary of the Summit.

摘要: 近年来，针对软件供应链中安全性较差的部分的网络攻击有所增加，并对企业和组织造成了致命的损害。过去众所周知的软件供应链攻击的例子是SolarWinds或log4j事件，它们影响了数千名客户和企业。美国政府和业界对加强软件供应链安全同样感兴趣。2023年6月7日，来自NSF支持的安全软件供应链中心(S3C2)的研究人员与来自13个政府机构的17名从业人员举行了一次安全软件供应链峰会。峰会的目标有两个：(1)与业界分享我们在前两次峰会上的看法，(2)让政府机构的个人能够分享软件供应链安全方面的实际经验和挑战。对于每个讨论主题，我们都会提出我们的观察结果，并从行业峰会上摘录，以促进对话。我们特别关注行政命令14028、软件材料清单(SBOM)、选择新的依赖项、出处和自我证明，以及大型语言模型。公开讨论实现了相互分享，并阐明了政府机构认为在保护其软件供应链时影响政府和行业从业者的共同挑战。在本文中，我们提供了峰会的总结。



## **22. An Empirical Study on Using Large Language Models to Analyze Software Supply Chain Security Failures**

大型语言模型分析软件供应链安全失效的实证研究 cs.CR

22 pages, 9 figures

**SubmitDate**: 2023-08-09    [abs](http://arxiv.org/abs/2308.04898v1) [paper-pdf](http://arxiv.org/pdf/2308.04898v1)

**Authors**: Tanmay Singla, Dharun Anandayuvaraj, Kelechi G. Kalu, Taylor R. Schorlemmer, James C. Davis

**Abstract**: As we increasingly depend on software systems, the consequences of breaches in the software supply chain become more severe. High-profile cyber attacks like those on SolarWinds and ShadowHammer have resulted in significant financial and data losses, underlining the need for stronger cybersecurity. One way to prevent future breaches is by studying past failures. However, traditional methods of analyzing these failures require manually reading and summarizing reports about them. Automated support could reduce costs and allow analysis of more failures. Natural Language Processing (NLP) techniques such as Large Language Models (LLMs) could be leveraged to assist the analysis of failures. In this study, we assessed the ability of Large Language Models (LLMs) to analyze historical software supply chain breaches. We used LLMs to replicate the manual analysis of 69 software supply chain security failures performed by members of the Cloud Native Computing Foundation (CNCF). We developed prompts for LLMs to categorize these by four dimensions: type of compromise, intent, nature, and impact. GPT 3.5s categorizations had an average accuracy of 68% and Bard had an accuracy of 58% over these dimensions. We report that LLMs effectively characterize software supply chain failures when the source articles are detailed enough for consensus among manual analysts, but cannot yet replace human analysts. Future work can improve LLM performance in this context, and study a broader range of articles and failures.

摘要: 随着我们越来越依赖软件系统，软件供应链中的违规后果变得更加严重。像SolarWinds和ShadowHammer这样备受瞩目的网络攻击已经导致了重大的财务和数据损失，突显出加强网络安全的必要性。防止未来违规的一种方法是研究过去的失败。然而，分析这些故障的传统方法需要手动阅读和汇总有关这些故障的报告。自动化支持可以降低成本，并允许分析更多故障。可以利用自然语言处理(NLP)技术，例如大型语言模型(LLM)来帮助分析故障。在这项研究中，我们评估了大型语言模型(LLM)分析历史软件供应链违规的能力。我们使用LLMS复制了云本地计算基金会(CNCF)成员对69个软件供应链安全故障进行的手动分析。我们为LLM开发了提示，将它们按四个维度进行分类：妥协类型、意图、性质和影响。GPT 3.5s分类的平均准确率为68%，Bard在这些维度上的准确率为58%。我们报告说，当源代码文章足够详细，可以在人工分析师之间达成共识，但还不能取代人工分析师时，LLM有效地描述了软件供应链故障的特征。未来的工作可以提高LLM在这一背景下的性能，并研究更广泛的文章和失败。



## **23. "Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models**

《Do Anything Now》：在大型语言模型上描述和评估野外越狱提示 cs.CR

**SubmitDate**: 2023-08-07    [abs](http://arxiv.org/abs/2308.03825v1) [paper-pdf](http://arxiv.org/pdf/2308.03825v1)

**Authors**: Xinyue Shen, Zeyuan Chen, Michael Backes, Yun Shen, Yang Zhang

**Abstract**: The misuse of large language models (LLMs) has garnered significant attention from the general public and LLM vendors. In response, efforts have been made to align LLMs with human values and intent use. However, a particular type of adversarial prompts, known as jailbreak prompt, has emerged and continuously evolved to bypass the safeguards and elicit harmful content from LLMs. In this paper, we conduct the first measurement study on jailbreak prompts in the wild, with 6,387 prompts collected from four platforms over six months. Leveraging natural language processing technologies and graph-based community detection methods, we discover unique characteristics of jailbreak prompts and their major attack strategies, such as prompt injection and privilege escalation. We also observe that jailbreak prompts increasingly shift from public platforms to private ones, posing new challenges for LLM vendors in proactive detection. To assess the potential harm caused by jailbreak prompts, we create a question set comprising 46,800 samples across 13 forbidden scenarios. Our experiments show that current LLMs and safeguards cannot adequately defend jailbreak prompts in all scenarios. Particularly, we identify two highly effective jailbreak prompts which achieve 0.99 attack success rates on ChatGPT (GPT-3.5) and GPT-4, and they have persisted online for over 100 days. Our work sheds light on the severe and evolving threat landscape of jailbreak prompts. We hope our study can facilitate the research community and LLM vendors in promoting safer and regulated LLMs.

摘要: 大型语言模型(LLM)的误用引起了公众和LLM供应商的极大关注。为此，已作出努力使低土地利用方式与人的价值和意图用途相一致。然而，一种称为越狱提示的特定类型的对抗性提示已经出现，并不断演变，以绕过安全措施，从LLMS引出有害内容。本文首次在野外对越狱提示进行了测量研究，在六个月的时间里，从四个平台收集了6387条提示。利用自然语言处理技术和基于图的社区检测方法，我们发现了越狱提示的独特特征及其主要攻击策略，如提示注入和权限提升。我们还观察到，越狱提示越来越多地从公共平台转向私人平台，这给LLM供应商在主动检测方面提出了新的挑战。为了评估越狱提示造成的潜在危害，我们创建了一个包含13个禁止场景的46,800个样本的问题集。我们的实验表明，现有的LLM和安全措施不能在所有场景下充分防御越狱提示。特别是，我们识别了两个高效的越狱提示，它们在ChatGPT(GPT-3.5)和GPT-4上的攻击成功率达到了0.99，并且在线持续了100多天。我们的工作揭示了越狱提示的严重和不断变化的威胁图景。我们希望我们的研究能够促进研究界和LLM供应商推广更安全和规范的LLM。



## **24. Mondrian: Prompt Abstraction Attack Against Large Language Models for Cheaper API Pricing**

Mondrian：针对大型语言模型的即时抽象攻击，以获得更低的API定价 cs.CR

**SubmitDate**: 2023-08-07    [abs](http://arxiv.org/abs/2308.03558v1) [paper-pdf](http://arxiv.org/pdf/2308.03558v1)

**Authors**: Wai Man Si, Michael Backes, Yang Zhang

**Abstract**: The Machine Learning as a Service (MLaaS) market is rapidly expanding and becoming more mature. For example, OpenAI's ChatGPT is an advanced large language model (LLM) that generates responses for various queries with associated fees. Although these models can deliver satisfactory performance, they are far from perfect. Researchers have long studied the vulnerabilities and limitations of LLMs, such as adversarial attacks and model toxicity. Inevitably, commercial ML models are also not exempt from such issues, which can be problematic as MLaaS continues to grow. In this paper, we discover a new attack strategy against LLM APIs, namely the prompt abstraction attack. Specifically, we propose Mondrian, a simple and straightforward method that abstracts sentences, which can lower the cost of using LLM APIs. In this approach, the adversary first creates a pseudo API (with a lower established price) to serve as the proxy of the target API (with a higher established price). Next, the pseudo API leverages Mondrian to modify the user query, obtain the abstracted response from the target API, and forward it back to the end user. Our results show that Mondrian successfully reduces user queries' token length ranging from 13% to 23% across various tasks, including text classification, generation, and question answering. Meanwhile, these abstracted queries do not significantly affect the utility of task-specific and general language models like ChatGPT. Mondrian also reduces instruction prompts' token length by at least 11% without compromising output quality. As a result, the prompt abstraction attack enables the adversary to profit without bearing the cost of API development and deployment.

摘要: 机器学习即服务(MLaaS)市场正在迅速扩大并日趋成熟。例如，OpenAI的ChatGPT是一个高级的大型语言模型(LLM)，它可以为各种查询生成响应，并收取相关费用。尽管这些车型可以提供令人满意的性能，但它们远不是完美的。长期以来，研究人员一直在研究LLMS的脆弱性和局限性，如对抗性攻击和模型毒性。不可避免的是，商业ML模型也不能幸免于此类问题，随着MLaaS的持续增长，这些问题可能会成为问题。在本文中，我们发现了一种针对LLMAPI的新攻击策略，即即时抽象攻击。具体地说，我们提出了Mondrian，这是一种简单明了的抽象句子的方法，可以降低使用LLMAPI的成本。在这种方法中，对手首先创建一个伪API(具有较低的既定价格)来充当目标API的代理(具有较高的既定价格)。接下来，伪API利用Mondrian修改用户查询，从目标API获取抽象的响应，并将其转发回最终用户。我们的结果表明，Mondrian成功地将用户查询的令牌长度在包括文本分类、生成和问答在内的各种任务中减少了13%到23%。同时，这些抽象的查询不会显著影响特定任务和通用语言模型(如ChatGPT)的效用。Mondrian还在不影响输出质量的情况下将指令提示符的标记长度减少了至少11%。因此，及时的抽象攻击使对手能够在不承担API开发和部署成本的情况下获利。



## **25. ParaFuzz: An Interpretability-Driven Technique for Detecting Poisoned Samples in NLP**

ParaFuzz：一种可解释性驱动的NLP中毒样本检测技术 cs.CR

**SubmitDate**: 2023-08-04    [abs](http://arxiv.org/abs/2308.02122v1) [paper-pdf](http://arxiv.org/pdf/2308.02122v1)

**Authors**: Lu Yan, Zhuo Zhang, Guanhong Tao, Kaiyuan Zhang, Xuan Chen, Guangyu Shen, Xiangyu Zhang

**Abstract**: Backdoor attacks have emerged as a prominent threat to natural language processing (NLP) models, where the presence of specific triggers in the input can lead poisoned models to misclassify these inputs to predetermined target classes. Current detection mechanisms are limited by their inability to address more covert backdoor strategies, such as style-based attacks. In this work, we propose an innovative test-time poisoned sample detection framework that hinges on the interpretability of model predictions, grounded in the semantic meaning of inputs. We contend that triggers (e.g., infrequent words) are not supposed to fundamentally alter the underlying semantic meanings of poisoned samples as they want to stay stealthy. Based on this observation, we hypothesize that while the model's predictions for paraphrased clean samples should remain stable, predictions for poisoned samples should revert to their true labels upon the mutations applied to triggers during the paraphrasing process. We employ ChatGPT, a state-of-the-art large language model, as our paraphraser and formulate the trigger-removal task as a prompt engineering problem. We adopt fuzzing, a technique commonly used for unearthing software vulnerabilities, to discover optimal paraphrase prompts that can effectively eliminate triggers while concurrently maintaining input semantics. Experiments on 4 types of backdoor attacks, including the subtle style backdoors, and 4 distinct datasets demonstrate that our approach surpasses baseline methods, including STRIP, RAP, and ONION, in precision and recall.

摘要: 后门攻击已经成为自然语言处理(NLP)模型的一个突出威胁，在NLP模型中，输入中存在特定触发器可能会导致中毒模型将这些输入错误分类到预定的目标类别。当前的检测机制由于无法应对更隐蔽的后门策略而受到限制，例如基于样式的攻击。在这项工作中，我们提出了一个创新的测试时间中毒样本检测框架，该框架取决于模型预测的可解释性，基于输入的语义。我们认为，触发因素(例如，不常见的单词)不应该从根本上改变中毒样本的潜在语义，因为它们想要保持隐蔽性。基于这一观察，我们假设，虽然模型对释义干净样本的预测应该保持稳定，但对中毒样本的预测应该在释义过程中应用于触发器的突变后恢复到其真实标签。我们使用最先进的大型语言模型ChatGPT作为我们的释义，并将触发器移除任务描述为一个紧迫的工程问题。我们采用了模糊技术，这是一种常用的软件漏洞挖掘技术，可以发现最优的释义提示，可以有效地消除触发器，同时保持输入语义。在4种类型的后门攻击(包括微妙风格的后门攻击)和4个不同的数据集上的实验表明，我们的方法在准确率和召回率上都超过了基线方法，包括STRAP、RAP和洋葱。



## **26. Fundamental Limitations of Alignment in Large Language Models**

大型语言模型中对齐的基本限制 cs.CL

**SubmitDate**: 2023-08-01    [abs](http://arxiv.org/abs/2304.11082v3) [paper-pdf](http://arxiv.org/pdf/2304.11082v3)

**Authors**: Yotam Wolf, Noam Wies, Oshri Avnery, Yoav Levine, Amnon Shashua

**Abstract**: An important aspect in developing language models that interact with humans is aligning their behavior to be useful and unharmful for their human users. This is usually achieved by tuning the model in a way that enhances desired behaviors and inhibits undesired ones, a process referred to as alignment. In this paper, we propose a theoretical approach called Behavior Expectation Bounds (BEB) which allows us to formally investigate several inherent characteristics and limitations of alignment in large language models. Importantly, we prove that for any behavior that has a finite probability of being exhibited by the model, there exist prompts that can trigger the model into outputting this behavior, with probability that increases with the length of the prompt. This implies that any alignment process that attenuates undesired behavior but does not remove it altogether, is not safe against adversarial prompting attacks. Furthermore, our framework hints at the mechanism by which leading alignment approaches such as reinforcement learning from human feedback increase the LLM's proneness to being prompted into the undesired behaviors. Moreover, we include the notion of personas in our BEB framework, and find that behaviors which are generally very unlikely to be exhibited by the model can be brought to the front by prompting the model to behave as specific persona. This theoretical result is being experimentally demonstrated in large scale by the so called contemporary "chatGPT jailbreaks", where adversarial users trick the LLM into breaking its alignment guardrails by triggering it into acting as a malicious persona. Our results expose fundamental limitations in alignment of LLMs and bring to the forefront the need to devise reliable mechanisms for ensuring AI safety.

摘要: 开发与人类交互的语言模型的一个重要方面是使他们的行为对人类用户有用而无害。这通常是通过调整模型来实现的，这种方式增强了期望的行为，抑制了不期望的行为，这一过程称为对齐。在本文中，我们提出了一种名为行为期望界限(BEB)的理论方法，它允许我们正式地研究大型语言模型中对齐的几个固有特征和限制。重要的是，我们证明了对于模型表现出的有限概率的任何行为，都存在可以触发模型输出该行为的提示，其概率随着提示的长度增加而增加。这意味着，任何减弱不受欢迎的行为但不能完全消除它的对准过程，在对抗提示攻击时都是不安全的。此外，我们的框架暗示了一种机制，通过这种机制，领先的对齐方法，如来自人类反馈的强化学习，增加了LLM被提示进入不希望看到的行为的倾向。此外，我们在我们的BEB框架中包括了人物角色的概念，并发现通过促使模型表现为特定的人物角色，通常不太可能在模型中表现的行为可以被带到前面。这一理论结果正在由所谓的当代“聊天GPT越狱”大规模实验证明，在这种情况下，敌对用户通过触发LLM充当恶意角色来欺骗LLM打破其对齐护栏。我们的结果暴露了LLM对齐方面的根本限制，并将设计可靠的机制以确保人工智能安全的必要性放在了首位。



## **27. LimeAttack: Local Explainable Method for Textual Hard-Label Adversarial Attack**

LimeAttack：文本硬标签对抗性攻击的局部可解释方法 cs.CL

26 pages, 7 figures

**SubmitDate**: 2023-08-01    [abs](http://arxiv.org/abs/2308.00319v1) [paper-pdf](http://arxiv.org/pdf/2308.00319v1)

**Authors**: Hai Zhu, Zhaoqing Yang, Weiwei Shang, Yuren Wu

**Abstract**: Natural language processing models are vulnerable to adversarial examples. Previous textual adversarial attacks adopt gradients or confidence scores to calculate word importance ranking and generate adversarial examples. However, this information is unavailable in the real world. Therefore, we focus on a more realistic and challenging setting, named hard-label attack, in which the attacker can only query the model and obtain a discrete prediction label. Existing hard-label attack algorithms tend to initialize adversarial examples by random substitution and then utilize complex heuristic algorithms to optimize the adversarial perturbation. These methods require a lot of model queries and the attack success rate is restricted by adversary initialization. In this paper, we propose a novel hard-label attack algorithm named LimeAttack, which leverages a local explainable method to approximate word importance ranking, and then adopts beam search to find the optimal solution. Extensive experiments show that LimeAttack achieves the better attacking performance compared with existing hard-label attack under the same query budget. In addition, we evaluate the effectiveness of LimeAttack on large language models, and results indicate that adversarial examples remain a significant threat to large language models. The adversarial examples crafted by LimeAttack are highly transferable and effectively improve model robustness in adversarial training.

摘要: 自然语言处理模型很容易受到敌意例子的影响。以前的文本对抗性攻击采用梯度或置信度分数来计算单词重要性排名并生成对抗性实例。然而，这些信息在现实世界中是不可用的。因此，我们将重点放在一种更现实和更具挑战性的环境中，即硬标签攻击，在这种情况下，攻击者只能查询模型并获得离散的预测标签。现有的硬标签攻击算法倾向于通过随机替换来初始化对抗性样本，然后利用复杂的启发式算法来优化对抗性扰动。这些方法需要大量的模型查询，攻击成功率受对手初始化的制约。本文提出了一种新的硬标签攻击算法LimeAttack，该算法利用一种局部可解释的方法来近似单词重要性排序，然后采用波束搜索来寻找最优解。大量实验表明，在相同的查询开销下，LimeAttack的攻击性能优于现有的硬标签攻击。此外，我们评估了LimeAttack在大型语言模型上的有效性，结果表明，对抗性例子仍然是对大型语言模型的重大威胁。LimeAttack生成的对抗性实例具有很强的可移植性，有效地提高了对抗性训练中模型的稳健性。



## **28. Adversarially Robust Neural Legal Judgement Systems**

对抗性稳健神经法律判决系统 cs.CL

**SubmitDate**: 2023-07-31    [abs](http://arxiv.org/abs/2308.00165v1) [paper-pdf](http://arxiv.org/pdf/2308.00165v1)

**Authors**: Rohit Raj, V Susheela Devi

**Abstract**: Legal judgment prediction is the task of predicting the outcome of court cases on a given text description of facts of cases. These tasks apply Natural Language Processing (NLP) techniques to predict legal judgment results based on facts. Recently, large-scale public datasets and NLP models have increased research in areas related to legal judgment prediction systems. For such systems to be practically helpful, they should be robust from adversarial attacks. Previous works mainly focus on making a neural legal judgement system; however, significantly less or no attention has been given to creating a robust Legal Judgement Prediction(LJP) system. We implemented adversarial attacks on early existing LJP systems and found that none of them could handle attacks. In this work, we proposed an approach for making robust LJP systems. Extensive experiments on three legal datasets show significant improvements in our approach over the state-of-the-art LJP system in handling adversarial attacks. To the best of our knowledge, we are the first to increase the robustness of early-existing LJP systems.

摘要: 法律判决预测是根据案件事实的特定文本描述来预测案件结果的任务。这些任务应用自然语言处理(NLP)技术来根据事实预测法律判决结果。近年来，大规模公共数据集和自然语言处理模型在与法律判决预测系统相关的领域增加了研究。要想让这种系统在实际中发挥作用，它们应该能够稳健地抵御敌方攻击。以前的工作主要集中在建立一个神经法律判决系统，然而，对于建立一个健壮的法律判决预测系统的关注明显较少或没有。我们对早期存在的LJP系统实施了对抗性攻击，发现没有一个系统能够应对攻击。在这项工作中，我们提出了一种构造健壮的LJP系统的方法。在三个法律数据集上的广泛实验表明，在处理对抗性攻击方面，我们的方法比最先进的LJP系统有了显着的改进。据我们所知，我们是第一个增加早期存在的LJP系统的稳健性的公司。



## **29. Virtual Prompt Injection for Instruction-Tuned Large Language Models**

面向指令调谐的大型语言模型的虚拟提示注入 cs.CL

**SubmitDate**: 2023-07-31    [abs](http://arxiv.org/abs/2307.16888v1) [paper-pdf](http://arxiv.org/pdf/2307.16888v1)

**Authors**: Jun Yan, Vikas Yadav, Shiyang Li, Lichang Chen, Zheng Tang, Hai Wang, Vijay Srinivasan, Xiang Ren, Hongxia Jin

**Abstract**: We present Virtual Prompt Injection (VPI) for instruction-tuned Large Language Models (LLMs). VPI allows an attacker-specified virtual prompt to steer the model behavior under specific trigger scenario without any explicit injection in model input. For instance, if an LLM is compromised with the virtual prompt "Describe Joe Biden negatively." for Joe Biden-related instructions, then any service deploying this model will propagate biased views when handling user queries related to Joe Biden. VPI is especially harmful for two primary reasons. Firstly, the attacker can take fine-grained control over LLM behaviors by defining various virtual prompts, exploiting LLMs' proficiency in following instructions. Secondly, this control is achieved without any interaction from the attacker while the model is in service, leading to persistent attack. To demonstrate the threat, we propose a simple method for performing VPI by poisoning the model's instruction tuning data. We find that our proposed method is highly effective in steering the LLM with VPI. For example, by injecting only 52 poisoned examples (0.1% of the training data size) into the instruction tuning data, the percentage of negative responses given by the trained model on Joe Biden-related queries change from 0% to 40%. We thus highlight the necessity of ensuring the integrity of the instruction-tuning data as little poisoned data can cause stealthy and persistent harm to the deployed model. We further explore the possible defenses and identify data filtering as an effective way to defend against the poisoning attacks. Our project page is available at https://poison-llm.github.io.

摘要: 针对指令调谐的大型语言模型，我们提出了虚拟提示注入(VPI)。VPI允许攻击者指定的虚拟提示来控制特定触发场景下的模型行为，而无需在模型输入中进行任何显式注入。例如，如果LLM因虚拟提示“负面描述乔·拜登”而受到威胁。对于与乔·拜登相关的指令，那么部署此模型的任何服务在处理与乔·拜登相关的用户查询时都将传播有偏见的视图。VPI尤其有害，主要有两个原因。首先，攻击者可以通过定义各种虚拟提示，利用LLMS对指令的熟练程度，对LLM行为进行细粒度的控制。其次，这种控制是在模型运行时不需要攻击者进行任何交互而实现的，从而导致持续攻击。为了演示威胁，我们提出了一种通过毒化模型的指令调优数据来执行VPI的简单方法。我们发现我们提出的方法是非常有效的用VPI来引导LLM。例如，通过仅将52个有毒样本(训练数据大小的0.1%)注入指令调整数据中，训练的模型对乔·拜登相关的查询给出的否定响应的百分比从0%变为40%。因此，我们强调必须确保指令调整数据的完整性，因为少量有毒数据可能会对部署的模型造成隐形和持久的损害。我们进一步探索了可能的防御措施，并确定数据过滤是防御中毒攻击的有效方法。我们的项目页面可在https://poison-llm.github.io.上查看



## **30. Competence-Based Analysis of Language Models**

基于能力的语言模型分析 cs.CL

**SubmitDate**: 2023-07-31    [abs](http://arxiv.org/abs/2303.00333v2) [paper-pdf](http://arxiv.org/pdf/2303.00333v2)

**Authors**: Adam Davies, Jize Jiang, ChengXiang Zhai

**Abstract**: Despite the recent success of large pretrained language models (LMs) on a variety of prompting tasks, these models can be alarmingly brittle to small changes in inputs or application contexts. To better understand such behavior and motivate the design of more robust LMs, we propose a general experimental framework, CALM (Competence-based Analysis of Language Models), where targeted causal interventions are utilized to damage an LM's internal representation of various linguistic properties in order to evaluate its use of each representation in performing a given task. We implement these interventions as gradient-based adversarial attacks, which (in contrast to prior causal probing methodologies) are able to target arbitrarily-encoded representations of relational properties, and carry out a case study of this approach to analyze how BERT-like LMs use representations of several relational properties in performing associated relation prompting tasks. We find that, while the representations LMs leverage in performing each task are highly entangled, they may be meaningfully interpreted in terms of the tasks where they are most utilized; and more broadly, that CALM enables an expanded scope of inquiry in LM analysis that may be useful in predicting and explaining weaknesses of existing LMs.

摘要: 尽管大型预先训练的语言模型(LMS)最近在各种提示任务上取得了成功，但这些模型对输入或应用程序上下文的微小变化可能会非常脆弱。为了更好地理解这类行为，并激励设计更健壮的LMS，我们提出了一个通用的实验框架，CAMPE(基于能力的语言模型分析)，其中有针对性的因果干预被用来破坏语言模型对各种语言属性的内部表征，以便评估它在执行给定任务时对每个表征的使用。我们将这些干预实现为基于梯度的对抗性攻击，与以往的因果探测方法不同，它能够针对关系属性的任意编码表示，并进行了该方法的案例研究，以分析类似于BERT的LMS如何在执行关联关系提示任务时使用多个关系属性的表示。我们发现，虽然LMS在执行每项任务时所利用的表征是高度纠缠的，但它们可能会被有意义地解释为它们最被利用的任务；更广泛地说，这种平静使LM分析中的调查范围得以扩大，这可能有助于预测和解释现有LMS的弱点。



## **31. Universal and Transferable Adversarial Attacks on Aligned Language Models**

对对齐语言模型的通用和可转移的对抗性攻击 cs.CL

**SubmitDate**: 2023-07-27    [abs](http://arxiv.org/abs/2307.15043v1) [paper-pdf](http://arxiv.org/pdf/2307.15043v1)

**Authors**: Andy Zou, Zifan Wang, J. Zico Kolter, Matt Fredrikson

**Abstract**: Because "out-of-the-box" large language models are capable of generating a great deal of objectionable content, recent work has focused on aligning these models in an attempt to prevent undesirable generation. While there has been some success at circumventing these measures -- so-called "jailbreaks" against LLMs -- these attacks have required significant human ingenuity and are brittle in practice. In this paper, we propose a simple and effective attack method that causes aligned language models to generate objectionable behaviors. Specifically, our approach finds a suffix that, when attached to a wide range of queries for an LLM to produce objectionable content, aims to maximize the probability that the model produces an affirmative response (rather than refusing to answer). However, instead of relying on manual engineering, our approach automatically produces these adversarial suffixes by a combination of greedy and gradient-based search techniques, and also improves over past automatic prompt generation methods.   Surprisingly, we find that the adversarial prompts generated by our approach are quite transferable, including to black-box, publicly released LLMs. Specifically, we train an adversarial attack suffix on multiple prompts (i.e., queries asking for many different types of objectionable content), as well as multiple models (in our case, Vicuna-7B and 13B). When doing so, the resulting attack suffix is able to induce objectionable content in the public interfaces to ChatGPT, Bard, and Claude, as well as open source LLMs such as LLaMA-2-Chat, Pythia, Falcon, and others. In total, this work significantly advances the state-of-the-art in adversarial attacks against aligned language models, raising important questions about how such systems can be prevented from producing objectionable information. Code is available at github.com/llm-attacks/llm-attacks.

摘要: 由于“开箱即用”的大型语言模型能够生成大量令人反感的内容，最近的工作重点是调整这些模型，以试图防止不必要的生成。虽然在规避这些措施方面取得了一些成功--即所谓的针对LLMS的“越狱”--但这些攻击需要大量的人类智慧，而且在实践中是脆弱的。在本文中，我们提出了一种简单有效的攻击方法，使对齐的语言模型产生令人反感的行为。具体地说，我们的方法找到了一个后缀，当附加到LLM的广泛查询中以产生令人反感的内容时，旨在最大化该模型产生肯定响应(而不是拒绝回答)的概率。然而，我们的方法不依赖于人工设计，而是通过贪婪和基于梯度的搜索技术相结合来自动生成这些对抗性后缀，并且改进了过去的自动提示生成方法。令人惊讶的是，我们发现我们的方法生成的对抗性提示是相当可转移的，包括到黑盒，公开发布的LLM。具体地说，我们对多个提示(即，要求许多不同类型的不良内容的查询)以及多个模型(在我们的案例中，Vicuna-7B和13B)训练对抗性攻击后缀。这样做时，生成的攻击后缀能够在ChatGPT、Bard和Claude的公共接口以及开源LLM(如llama-2-chat、Pythia、Falcon和其他)中诱导令人反感的内容。总而言之，这项工作极大地推进了针对对齐语言模型的对抗性攻击的最新水平，提出了如何防止此类系统产生令人反感的信息的重要问题。代码可在githorb.com/llm-Attages/llm-Attack上找到。



## **32. Backdoor Attacks for In-Context Learning with Language Models**

利用语言模型进行情景学习的后门攻击 cs.CR

AdvML Frontiers Workshop 2023

**SubmitDate**: 2023-07-27    [abs](http://arxiv.org/abs/2307.14692v1) [paper-pdf](http://arxiv.org/pdf/2307.14692v1)

**Authors**: Nikhil Kandpal, Matthew Jagielski, Florian Tramèr, Nicholas Carlini

**Abstract**: Because state-of-the-art language models are expensive to train, most practitioners must make use of one of the few publicly available language models or language model APIs. This consolidation of trust increases the potency of backdoor attacks, where an adversary tampers with a machine learning model in order to make it perform some malicious behavior on inputs that contain a predefined backdoor trigger. We show that the in-context learning ability of large language models significantly complicates the question of developing backdoor attacks, as a successful backdoor must work against various prompting strategies and should not affect the model's general purpose capabilities. We design a new attack for eliciting targeted misclassification when language models are prompted to perform a particular target task and demonstrate the feasibility of this attack by backdooring multiple large language models ranging in size from 1.3 billion to 6 billion parameters. Finally we study defenses to mitigate the potential harms of our attack: for example, while in the white-box setting we show that fine-tuning models for as few as 500 steps suffices to remove the backdoor behavior, in the black-box setting we are unable to develop a successful defense that relies on prompt engineering alone.

摘要: 因为最先进的语言模型的培训成本很高，所以大多数实践者必须使用为数不多的公开可用的语言模型或语言模型API之一。这种信任的巩固增加了后门攻击的效力，在这种攻击中，对手篡改了机器学习模型，以便使其对包含预定义后门触发器的输入执行一些恶意行为。我们表明，大型语言模型的上下文学习能力显著复杂化了开发后门攻击的问题，因为成功的后门必须针对各种提示策略工作，并且不应影响模型的通用功能。我们设计了一种新的攻击，当语言模型被提示执行特定的目标任务时，会导致有针对性的误分类，并通过回溯从13亿到60亿个参数的多个大型语言模型来证明该攻击的可行性。最后，我们研究防御以减轻攻击的潜在危害：例如，在白盒设置中，我们展示了只需500步的微调模型就足以消除后门行为，而在黑盒设置中，我们无法开发出仅依赖于即时工程的成功防御。



## **33. Plug and Pray: Exploiting off-the-shelf components of Multi-Modal Models**

即插即用：开发多通道模型的现成组件 cs.CR

**SubmitDate**: 2023-07-26    [abs](http://arxiv.org/abs/2307.14539v1) [paper-pdf](http://arxiv.org/pdf/2307.14539v1)

**Authors**: Erfan Shayegani, Yue Dong, Nael Abu-Ghazaleh

**Abstract**: The rapid growth and increasing popularity of incorporating additional modalities (e.g., vision) into large language models (LLMs) has raised significant security concerns. This expansion of modality, akin to adding more doors to a house, unintentionally creates multiple access points for adversarial attacks. In this paper, by introducing adversarial embedding space attacks, we emphasize the vulnerabilities present in multi-modal systems that originate from incorporating off-the-shelf components like public pre-trained encoders in a plug-and-play manner into these systems. In contrast to existing work, our approach does not require access to the multi-modal system's weights or parameters but instead relies on the huge under-explored embedding space of such pre-trained encoders. Our proposed embedding space attacks involve seeking input images that reside within the dangerous or targeted regions of the extensive embedding space of these pre-trained components. These crafted adversarial images pose two major threats: 'Context Contamination' and 'Hidden Prompt Injection'-both of which can compromise multi-modal models like LLaVA and fully change the behavior of the associated language model. Our findings emphasize the need for a comprehensive examination of the underlying components, particularly pre-trained encoders, before incorporating them into systems in a plug-and-play manner to ensure robust security.

摘要: 在大型语言模型(LLM)中加入其他模式(例如，VISION)的快速增长和越来越受欢迎，引发了重大的安全问题。这种模式的扩展，类似于在一所房子里增加更多的门，无意中为对抗性攻击创造了多个接入点。在本文中，我们通过引入对抗性嵌入空间攻击，强调了多模式系统中存在的漏洞，这些漏洞源于以即插即用的方式将公共预训练编码器等现成组件整合到这些系统中。与现有的工作相比，我们的方法不需要访问多模式系统的权重或参数，而是依赖于这种预先训练的编码器的巨大的未被充分开发的嵌入空间。我们提出的嵌入空间攻击涉及寻找驻留在这些预训练组件的广泛嵌入空间的危险或目标区域内的输入图像。这些精心制作的敌意图像构成了两个主要威胁：“上下文污染”和“隐藏提示注入”--这两个威胁都可能危及LLaVA等多模式模型，并完全改变相关语言模型的行为。我们的研究结果强调，在以即插即用的方式将底层组件纳入系统以确保强大的安全性之前，需要对底层组件进行全面检查，特别是经过预先培训的编码器。



## **34. Set-level Guidance Attack: Boosting Adversarial Transferability of Vision-Language Pre-training Models**

SET-Level制导攻击：增强视觉-语言预训练模型的对抗性迁移 cs.CV

To appear in ICCV 2023

**SubmitDate**: 2023-07-26    [abs](http://arxiv.org/abs/2307.14061v1) [paper-pdf](http://arxiv.org/pdf/2307.14061v1)

**Authors**: Dong Lu, Zhiqiang Wang, Teng Wang, Weili Guan, Hongchang Gao, Feng Zheng

**Abstract**: Vision-language pre-training (VLP) models have shown vulnerability to adversarial examples in multimodal tasks. Furthermore, malicious adversaries can be deliberately transferred to attack other black-box models. However, existing work has mainly focused on investigating white-box attacks. In this paper, we present the first study to investigate the adversarial transferability of recent VLP models. We observe that existing methods exhibit much lower transferability, compared to the strong attack performance in white-box settings. The transferability degradation is partly caused by the under-utilization of cross-modal interactions. Particularly, unlike unimodal learning, VLP models rely heavily on cross-modal interactions and the multimodal alignments are many-to-many, e.g., an image can be described in various natural languages. To this end, we propose a highly transferable Set-level Guidance Attack (SGA) that thoroughly leverages modality interactions and incorporates alignment-preserving augmentation with cross-modal guidance. Experimental results demonstrate that SGA could generate adversarial examples that can strongly transfer across different VLP models on multiple downstream vision-language tasks. On image-text retrieval, SGA significantly enhances the attack success rate for transfer attacks from ALBEF to TCL by a large margin (at least 9.78% and up to 30.21%), compared to the state-of-the-art.

摘要: 视觉-语言预训练(VLP)模型在多通道任务中表现出对对抗性例子的脆弱性。此外，恶意攻击者可以被故意转移到其他黑盒模型进行攻击。然而，现有的工作主要集中在调查白盒攻击上。在这篇文章中，我们首次研究了最近的VLP模型的对抗性转移。我们观察到，与白盒环境下强大的攻击性能相比，现有方法表现出的可转移性要低得多。可转移性下降的部分原因是对跨通道互动的利用不足。特别是，与单峰学习不同，VLP模型严重依赖于跨通道交互，并且多通道对齐是多对多的，例如，可以用各种自然语言描述图像。为此，我们提出了一种高度可移植的集水平制导攻击(SGA)，它充分利用了通道交互，并将对齐保持增强与跨通道制导相结合。实验结果表明，SGA能够产生对抗性的例子，并能在多个下游视觉语言任务上跨不同的VLP模型进行强迁移。在图文检索方面，与现有技术相比，SGA显著提高了将攻击从ALBEF转移到TCL的攻击成功率(至少9.78%，高达30.21%)。



## **35. Foundational Models Defining a New Era in Vision: A Survey and Outlook**

定义愿景中的新时代的基础模型：综述和展望 cs.CV

Project page:  https://github.com/awaisrauf/Awesome-CV-Foundational-Models

**SubmitDate**: 2023-07-25    [abs](http://arxiv.org/abs/2307.13721v1) [paper-pdf](http://arxiv.org/pdf/2307.13721v1)

**Authors**: Muhammad Awais, Muzammal Naseer, Salman Khan, Rao Muhammad Anwer, Hisham Cholakkal, Mubarak Shah, Ming-Hsuan Yang, Fahad Shahbaz Khan

**Abstract**: Vision systems to see and reason about the compositional nature of visual scenes are fundamental to understanding our world. The complex relations between objects and their locations, ambiguities, and variations in the real-world environment can be better described in human language, naturally governed by grammatical rules and other modalities such as audio and depth. The models learned to bridge the gap between such modalities coupled with large-scale training data facilitate contextual reasoning, generalization, and prompt capabilities at test time. These models are referred to as foundational models. The output of such models can be modified through human-provided prompts without retraining, e.g., segmenting a particular object by providing a bounding box, having interactive dialogues by asking questions about an image or video scene or manipulating the robot's behavior through language instructions. In this survey, we provide a comprehensive review of such emerging foundational models, including typical architecture designs to combine different modalities (vision, text, audio, etc), training objectives (contrastive, generative), pre-training datasets, fine-tuning mechanisms, and the common prompting patterns; textual, visual, and heterogeneous. We discuss the open challenges and research directions for foundational models in computer vision, including difficulties in their evaluations and benchmarking, gaps in their real-world understanding, limitations of their contextual understanding, biases, vulnerability to adversarial attacks, and interpretability issues. We review recent developments in this field, covering a wide range of applications of foundation models systematically and comprehensively. A comprehensive list of foundational models studied in this work is available at \url{https://github.com/awaisrauf/Awesome-CV-Foundational-Models}.

摘要: 视觉系统可以看到视觉场景的组成性质，并对其进行推理，这是理解我们世界的基础。对象与其位置之间的复杂关系、模糊性和现实世界环境中的变化可以用人类语言更好地描述，自然地受语法规则和其他形式(如音频和深度)的支配。这些模型学会了弥合这种模式之间的差距，再加上大规模的训练数据，促进了测试时的上下文推理、概括和提示能力。这些模型被称为基础模型。这样的模型的输出可以通过人工提供的提示来修改，而无需再训练，例如，通过提供边界框来分割特定对象，通过询问关于图像或视频场景的问题来进行交互式对话，或者通过语言指令来操纵机器人的行为。在这次调查中，我们对这些新兴的基础模型进行了全面的回顾，包括将不同的模式(视觉、文本、音频等)、培训目标(对比、生成)、预培训数据集、微调机制以及常见的提示模式(文本、视觉和异质)结合在一起的典型架构设计。我们讨论了计算机视觉中基础模型的开放挑战和研究方向，包括它们在评估和基准方面的困难，它们在现实世界理解中的差距，它们上下文理解的局限性，偏见，对对手攻击的脆弱性，以及可解释性问题。我们回顾了这一领域的最新发展，系统和全面地涵盖了基础模型的广泛应用。有关本工作中研究的基本模型的完整列表，请访问\url{https://github.com/awaisrauf/Awesome-CV-Foundational-Models}.



## **36. Lost In Translation: Generating Adversarial Examples Robust to Round-Trip Translation**

迷失在翻译中：生成对往返翻译健壮的对抗性例子 cs.CL

Published at International Conference on Acoustics, Speech, and  Signal Processing (ICASSP) 2023

**SubmitDate**: 2023-07-24    [abs](http://arxiv.org/abs/2307.12520v1) [paper-pdf](http://arxiv.org/pdf/2307.12520v1)

**Authors**: Neel Bhandari, Pin-Yu Chen

**Abstract**: Language Models today provide a high accuracy across a large number of downstream tasks. However, they remain susceptible to adversarial attacks, particularly against those where the adversarial examples maintain considerable similarity to the original text. Given the multilingual nature of text, the effectiveness of adversarial examples across translations and how machine translations can improve the robustness of adversarial examples remain largely unexplored. In this paper, we present a comprehensive study on the robustness of current text adversarial attacks to round-trip translation. We demonstrate that 6 state-of-the-art text-based adversarial attacks do not maintain their efficacy after round-trip translation. Furthermore, we introduce an intervention-based solution to this problem, by integrating Machine Translation into the process of adversarial example generation and demonstrating increased robustness to round-trip translation. Our results indicate that finding adversarial examples robust to translation can help identify the insufficiency of language models that is common across languages, and motivate further research into multilingual adversarial attacks.

摘要: 今天的语言模型在大量下游任务中提供了高精度。然而，它们仍然容易受到对抗性攻击，特别是对那些对抗性例子与原文保持相当相似的攻击。鉴于文本的多语种性质，翻译中对抗性例子的有效性以及机器翻译如何提高对抗性例子的稳健性在很大程度上仍未得到探索。本文对当前文本对抗性攻击对双向翻译的稳健性进行了全面的研究。我们证明了6种最新的基于文本的对抗性攻击在往返翻译后不能保持它们的有效性。此外，我们引入了基于干预的解决方案，通过将机器翻译集成到对抗性实例生成过程中，并展示了对往返翻译的更强的健壮性。我们的结果表明，找到对翻译具有健壮性的对抗性实例可以帮助识别跨语言通用的语言模型的不足，并激励对多语言对抗性攻击的进一步研究。



## **37. Security and Privacy Issues of Federated Learning**

联合学习的安全和隐私问题 cs.CR

6 pages, 2 figures

**SubmitDate**: 2023-07-22    [abs](http://arxiv.org/abs/2307.12181v1) [paper-pdf](http://arxiv.org/pdf/2307.12181v1)

**Authors**: Jahid Hasan

**Abstract**: Federated Learning (FL) has emerged as a promising approach to address data privacy and confidentiality concerns by allowing multiple participants to construct a shared model without centralizing sensitive data. However, this decentralized paradigm introduces new security challenges, necessitating a comprehensive identification and classification of potential risks to ensure FL's security guarantees. This paper presents a comprehensive taxonomy of security and privacy challenges in Federated Learning (FL) across various machine learning models, including large language models. We specifically categorize attacks performed by the aggregator and participants, focusing on poisoning attacks, backdoor attacks, membership inference attacks, generative adversarial network (GAN) based attacks, and differential privacy attacks. Additionally, we propose new directions for future research, seeking innovative solutions to fortify FL systems against emerging security risks and uphold sensitive data confidentiality in distributed learning environments.

摘要: 联合学习(FL)通过允许多个参与者在不集中敏感数据的情况下构建共享模型，成为解决数据隐私和机密性问题的一种有前途的方法。然而，这种分散的模式带来了新的安全挑战，需要对潜在风险进行全面识别和分类，以确保FL的安全保障。本文对联邦学习(FL)中的安全和隐私挑战进行了全面的分类，涵盖了各种机器学习模型，包括大型语言模型。我们对聚合器和参与者执行的攻击进行了专门的分类，重点是中毒攻击、后门攻击、成员身份推理攻击、基于生成性对抗网络(GAN)的攻击和差异化隐私攻击。此外，我们为未来的研究提出了新的方向，寻求创新的解决方案来加强FL系统对新出现的安全风险的防御，并在分布式学习环境中维护敏感数据的机密性。



## **38. OUTFOX: LLM-generated Essay Detection through In-context Learning with Adversarially Generated Examples**

Outfox：基于上下文学习的LLM生成的文章检测与恶意生成的示例 cs.CL

**SubmitDate**: 2023-07-21    [abs](http://arxiv.org/abs/2307.11729v1) [paper-pdf](http://arxiv.org/pdf/2307.11729v1)

**Authors**: Ryuto Koike, Masahiro Kaneko, Naoaki Okazaki

**Abstract**: Large Language Models (LLMs) have achieved human-level fluency in text generation, making it difficult to distinguish between human-written and LLM-generated texts. This poses a growing risk of misuse of LLMs and demands the development of detectors to identify LLM-generated texts. However, existing detectors degrade detection accuracy by simply paraphrasing LLM-generated texts. Furthermore, the effectiveness of these detectors in real-life situations, such as when students use LLMs for writing homework assignments (e.g., essays) and quickly learn how to evade these detectors, has not been explored. In this paper, we propose OUTFOX, a novel framework that improves the robustness of LLM-generated-text detectors by allowing both the detector and the attacker to consider each other's output and apply this to the domain of student essays. In our framework, the attacker uses the detector's prediction labels as examples for in-context learning and adversarially generates essays that are harder to detect. While the detector uses the adversarially generated essays as examples for in-context learning to learn to detect essays from a strong attacker. Our experiments show that our proposed detector learned in-context from the attacker improves the detection performance on the attacked dataset by up to +41.3 point F1-score. While our proposed attacker can drastically degrade the performance of the detector by up to -57.0 point F1-score compared to the paraphrasing method.

摘要: 大型语言模型(LLM)在文本生成方面达到了人类水平的流畅性，使得区分人类编写的文本和LLM生成的文本变得困难。这带来了滥用LLMS的越来越大的风险，并要求开发检测器来识别LLM生成的文本。然而，现有的检测器通过简单地解释LLM生成的文本来降低检测精度。此外，这些检测器在现实生活中的有效性还没有被探索过，例如当学生使用LLMS来写作业(例如，论文)并迅速学习如何逃避这些检测器时。在本文中，我们提出了一种新的框架Outfox，它通过允许检测器和攻击者考虑彼此的输出来提高LLM生成的文本检测器的健壮性，并将其应用到学生作文领域。在我们的框架中，攻击者使用检测器的预测标签作为上下文学习的示例，并恶意生成更难检测的文章。而检测器使用恶意生成的文章作为上下文学习的示例，以学习检测来自强大攻击者的文章。我们的实验表明，从攻击者那里学习的上下文中学习的检测器在攻击数据集上的检测性能提高了高达41.3点F1-Score。而我们提出的攻击者可以大幅降低检测器的性能，与改述方法相比，最高可达-57.0点F1分数。



## **39. A LLM Assisted Exploitation of AI-Guardian**

一种LLM辅助开发AI-Guardian cs.CR

**SubmitDate**: 2023-07-20    [abs](http://arxiv.org/abs/2307.15008v1) [paper-pdf](http://arxiv.org/pdf/2307.15008v1)

**Authors**: Nicholas Carlini

**Abstract**: Large language models (LLMs) are now highly capable at a diverse range of tasks. This paper studies whether or not GPT-4, one such LLM, is capable of assisting researchers in the field of adversarial machine learning. As a case study, we evaluate the robustness of AI-Guardian, a recent defense to adversarial examples published at IEEE S&P 2023, a top computer security conference. We completely break this defense: the proposed scheme does not increase robustness compared to an undefended baseline.   We write none of the code to attack this model, and instead prompt GPT-4 to implement all attack algorithms following our instructions and guidance. This process was surprisingly effective and efficient, with the language model at times producing code from ambiguous instructions faster than the author of this paper could have done. We conclude by discussing (1) the warning signs present in the evaluation that suggested to us AI-Guardian would be broken, and (2) our experience with designing attacks and performing novel research using the most recent advances in language modeling.

摘要: 大型语言模型(LLM)现在能够很好地完成各种任务。本文研究了LLM中的GPT-4是否能够帮助对抗性机器学习领域的研究人员。作为一个案例研究，我们评估了AI-Guardian的健壮性，它是最近在顶级计算机安全会议IEEE S&P2023上发布的对敌意示例的防御。我们完全打破了这一防御：与没有防御的基线相比，所提出的方案并没有增加稳健性。我们没有编写任何代码来攻击这个模型，而是提示GPT-4按照我们的说明和指导实现所有攻击算法。这个过程出人意料地有效和高效，语言模型有时从含糊的指令生成代码的速度比本文作者所做的要快。最后，我们讨论了(1)评估中出现的警告信号，这些迹象向我们暗示AI-Guardian将被打破，以及(2)我们使用语言建模中的最新进展设计攻击和执行新研究的经验。



## **40. LLM Censorship: A Machine Learning Challenge or a Computer Security Problem?**

LLM审查：机器学习挑战还是计算机安全问题？ cs.AI

**SubmitDate**: 2023-07-20    [abs](http://arxiv.org/abs/2307.10719v1) [paper-pdf](http://arxiv.org/pdf/2307.10719v1)

**Authors**: David Glukhov, Ilia Shumailov, Yarin Gal, Nicolas Papernot, Vardan Papyan

**Abstract**: Large language models (LLMs) have exhibited impressive capabilities in comprehending complex instructions. However, their blind adherence to provided instructions has led to concerns regarding risks of malicious use. Existing defence mechanisms, such as model fine-tuning or output censorship using LLMs, have proven to be fallible, as LLMs can still generate problematic responses. Commonly employed censorship approaches treat the issue as a machine learning problem and rely on another LM to detect undesirable content in LLM outputs. In this paper, we present the theoretical limitations of such semantic censorship approaches. Specifically, we demonstrate that semantic censorship can be perceived as an undecidable problem, highlighting the inherent challenges in censorship that arise due to LLMs' programmatic and instruction-following capabilities. Furthermore, we argue that the challenges extend beyond semantic censorship, as knowledgeable attackers can reconstruct impermissible outputs from a collection of permissible ones. As a result, we propose that the problem of censorship needs to be reevaluated; it should be treated as a security problem which warrants the adaptation of security-based approaches to mitigate potential risks.

摘要: 大型语言模型(LLM)在理解复杂指令方面表现出了令人印象深刻的能力。然而，他们盲目遵守提供的说明导致了人们对恶意使用风险的担忧。现有的防御机制，如使用LLMS的模型微调或输出审查，已被证明是容易出错的，因为LLMS仍可能产生有问题的反应。通常使用的审查方法将问题视为机器学习问题，并依赖于另一个LM来检测LLM输出中的不良内容。在本文中，我们提出了这种语义审查方法的理论局限性。具体地说，我们证明了语义审查可以被视为一个无法决定的问题，强调了由于LLMS的编程和指令遵循能力而在审查中出现的内在挑战。此外，我们认为挑战超越了语义审查，因为有知识的攻击者可以从一组允许的输出中重建出不允许的输出。因此，我们建议需要重新评估审查问题；应将其视为一个安全问题，必须采用以安全为基础的办法，以减轻潜在风险。



## **41. LogPrécis: Unleashing Language Models for Automated Shell Log Analysis**

LogPrecis：释放用于自动外壳日志分析的语言模型 cs.CR

**SubmitDate**: 2023-07-17    [abs](http://arxiv.org/abs/2307.08309v1) [paper-pdf](http://arxiv.org/pdf/2307.08309v1)

**Authors**: Matteo Boffa, Rodolfo Vieira Valentim, Luca Vassio, Danilo Giordano, Idilio Drago, Marco Mellia, Zied Ben Houidi

**Abstract**: The collection of security-related logs holds the key to understanding attack behaviors and diagnosing vulnerabilities. Still, their analysis remains a daunting challenge. Recently, Language Models (LMs) have demonstrated unmatched potential in understanding natural and programming languages. The question arises whether and how LMs could be also useful for security experts since their logs contain intrinsically confused and obfuscated information. In this paper, we systematically study how to benefit from the state-of-the-art in LM to automatically analyze text-like Unix shell attack logs. We present a thorough design methodology that leads to LogPr\'ecis. It receives as input raw shell sessions and automatically identifies and assigns the attacker tactic to each portion of the session, i.e., unveiling the sequence of the attacker's goals. We demonstrate LogPr\'ecis capability to support the analysis of two large datasets containing about 400,000 unique Unix shell attacks. LogPr\'ecis reduces them into about 3,000 fingerprints, each grouping sessions with the same sequence of tactics. The abstraction it provides lets the analyst better understand attacks, identify fingerprints, detect novelty, link similar attacks, and track families and mutations. Overall, LogPr\'ecis, released as open source, paves the way for better and more responsive defense against cyberattacks.

摘要: 安全相关日志的收集是了解攻击行为和诊断漏洞的关键。尽管如此，他们的分析仍然是一个艰巨的挑战。最近，语言模型(LMS)在理解自然语言和编程语言方面显示出无与伦比的潜力。问题是，LMS是否以及如何也对安全专家有用，因为他们的日志包含本质上令人困惑和混淆的信息。在本文中，我们系统地研究了如何利用LM的最新技术来自动分析类似文本的Unix外壳攻击日志。我们提出了一种通向LogPr‘ECIS的全面设计方法。它接收原始外壳会话作为输入，并自动识别攻击者的战术并将其分配给会话的每个部分，即揭示攻击者的目标序列。我们演示了LogPr的ECIS功能，以支持对包含约400,000个唯一Unix外壳攻击的两个大型数据集的分析。LogPr‘’ECIS将它们简化为大约3,000个指纹，每个指纹使用相同的战术序列对会话进行分组。它提供的抽象使分析师能够更好地了解攻击、识别指纹、检测新颖性、链接类似攻击以及跟踪家族和突变。总体而言，作为开源发布的LogPr‘ECIS为更好、更具响应性的网络攻击防御铺平了道路。



## **42. Jailbreaker: Automated Jailbreak Across Multiple Large Language Model Chatbots**

越狱：跨多个大型语言模型聊天机器人的自动越狱 cs.CR

**SubmitDate**: 2023-07-16    [abs](http://arxiv.org/abs/2307.08715v1) [paper-pdf](http://arxiv.org/pdf/2307.08715v1)

**Authors**: Gelei Deng, Yi Liu, Yuekang Li, Kailong Wang, Ying Zhang, Zefeng Li, Haoyu Wang, Tianwei Zhang, Yang Liu

**Abstract**: Large Language Models (LLMs) have revolutionized Artificial Intelligence (AI) services due to their exceptional proficiency in understanding and generating human-like text. LLM chatbots, in particular, have seen widespread adoption, transforming human-machine interactions. However, these LLM chatbots are susceptible to "jailbreak" attacks, where malicious users manipulate prompts to elicit inappropriate or sensitive responses, contravening service policies. Despite existing attempts to mitigate such threats, our research reveals a substantial gap in our understanding of these vulnerabilities, largely due to the undisclosed defensive measures implemented by LLM service providers.   In this paper, we present Jailbreaker, a comprehensive framework that offers an in-depth understanding of jailbreak attacks and countermeasures. Our work makes a dual contribution. First, we propose an innovative methodology inspired by time-based SQL injection techniques to reverse-engineer the defensive strategies of prominent LLM chatbots, such as ChatGPT, Bard, and Bing Chat. This time-sensitive approach uncovers intricate details about these services' defenses, facilitating a proof-of-concept attack that successfully bypasses their mechanisms. Second, we introduce an automatic generation method for jailbreak prompts. Leveraging a fine-tuned LLM, we validate the potential of automated jailbreak generation across various commercial LLM chatbots. Our method achieves a promising average success rate of 21.58%, significantly outperforming the effectiveness of existing techniques. We have responsibly disclosed our findings to the concerned service providers, underscoring the urgent need for more robust defenses. Jailbreaker thus marks a significant step towards understanding and mitigating jailbreak threats in the realm of LLM chatbots.

摘要: 大型语言模型(LLM)由于其在理解和生成类似人类的文本方面的非凡熟练程度，使人工智能(AI)服务发生了革命性的变化。尤其是LLM聊天机器人，已经被广泛采用，改变了人机交互。然而，这些LLM聊天机器人很容易受到“越狱”攻击，即恶意用户操纵提示来引发不适当或敏感的响应，这违反了服务策略。尽管存在缓解此类威胁的尝试，但我们的研究显示，我们对这些漏洞的理解存在很大差距，这主要是由于LLM服务提供商实施了未披露的防御措施。在这篇文章中，我们介绍了越狱，一个全面的框架，提供了深入了解越狱攻击和对策。我们的工作做出了双重贡献。首先，我们提出了一种受基于时间的SQL注入技术启发的创新方法来对著名的LLM聊天机器人(如ChatGPT、Bard和Bing Chat)的防御策略进行逆向工程。这种对时间敏感的方法揭示了这些服务防御的复杂细节，为成功绕过它们的机制的概念验证攻击提供了便利。其次，介绍了一种越狱提示的自动生成方法。利用微调的LLM，我们验证了在各种商业LLM聊天机器人上自动越狱生成的潜力。我们的方法达到了21.58%的平均成功率，大大超过了现有技术的有效性。我们已经负责任地向有关服务提供商披露了我们的调查结果，强调了迫切需要更强大的防御措施。因此，越狱标志着在理解和减轻LLM聊天机器人领域的越狱威胁方面迈出了重要的一步。



## **43. Cross-lingual Cross-temporal Summarization: Dataset, Models, Evaluation**

跨语言跨期摘要：数据集、模型、评价 cs.CL

Version 2; Work in progress

**SubmitDate**: 2023-07-13    [abs](http://arxiv.org/abs/2306.12916v2) [paper-pdf](http://arxiv.org/pdf/2306.12916v2)

**Authors**: Ran Zhang, Jihed Ouni, Steffen Eger

**Abstract**: While summarization has been extensively researched in natural language processing (NLP), cross-lingual cross-temporal summarization (CLCTS) is a largely unexplored area that has the potential to improve cross-cultural accessibility and understanding. This paper comprehensively addresses the CLCTS task, including dataset creation, modeling, and evaluation. We build the first CLCTS corpus, leveraging historical fictive texts and Wikipedia summaries in English and German, and examine the effectiveness of popular transformer end-to-end models with different intermediate finetuning tasks. Additionally, we explore the potential of ChatGPT for CLCTS as a summarizer and an evaluator. Overall, we report evaluations from humans, ChatGPT, and several recent automatic evaluation metrics where we find that our intermediate task finetuned end-to-end models generate bad to moderate quality summaries; ChatGPT as a summarizer (without any finetuning) provides moderate to good quality outputs and as an evaluator correlates moderately with human evaluations but is prone to giving lower scores. ChatGPT also seems very adept at normalizing historical text and outperforms context-unaware spelling normalization tools such as Norma. We finally test ChatGPT in a scenario with adversarially attacked and unseen source documents and find that ChatGPT profits from its prior knowledge to a certain degree, with better performances for omission and entity swap than negation against its prior knowledge. This benefit inflates its assessed quality as ChatGPT performs slightly worse for unseen source documents compared to seen documents. We additionally introspect our models' performances to find that longer, older and more complex source texts (all of which are more characteristic for historical language variants) are harder to summarize for all models, indicating the difficulty of the CLCTS task.

摘要: 摘要在自然语言处理(NLP)中得到了广泛的研究，但跨语言跨时序摘要(CLCTS)在很大程度上是一个未被开发的领域，它有可能提高跨文化的可及性和理解力。本文全面介绍了CLCTS的任务，包括数据集的创建、建模和评估。我们建立了第一个CLCTS语料库，利用英语和德语的历史虚构文本和维基百科摘要，并检查了具有不同中间微调任务的流行变压器端到端模型的有效性。此外，我们还探讨了ChatGPT作为CLCTS的摘要和评价器的潜力。总体而言，我们报告了来自人工、ChatGPT和最近几个自动评估指标的评估，其中我们发现我们的中间任务微调的端到端模型生成了较差到中等质量的摘要；ChatGPT作为汇总器(没有任何微调)提供了中等到良好的质量输出，并且作为评估者与人工评估适度相关，但容易给出较低的分数。ChatGPT似乎也非常擅长对历史文本进行规范化，表现优于Norma等上下文无关的拼写标准化工具。最后，我们在恶意攻击和不可见源文档的场景中对ChatGPT进行了测试，发现ChatGPT在一定程度上得益于其先验知识，在遗漏和实体交换方面的性能优于对其先验知识的否定。这一优势夸大了其评估的质量，因为ChatGPT对于不可见的源文档的性能略逊于可见的文档。此外，我们还反思了我们的模型的表现，发现更长、更老、更复杂的源文本(所有这些都是历史语言变体的特征)更难对所有模型进行总结，这表明了CLCTS任务的难度。



## **44. Prompts Should not be Seen as Secrets: Systematically Measuring Prompt Extraction Attack Success**

提示不应被视为秘密：系统地衡量提示提取攻击成功 cs.CL

**SubmitDate**: 2023-07-13    [abs](http://arxiv.org/abs/2307.06865v1) [paper-pdf](http://arxiv.org/pdf/2307.06865v1)

**Authors**: Yiming Zhang, Daphne Ippolito

**Abstract**: The generations of large language models are commonly controlled through prompting techniques, where a user's query to the model is prefixed with a prompt that aims to guide the model's behaviour on the query. The prompts used by companies to guide their models are often treated as secrets, to be hidden from the user making the query. They have even been treated as commodities to be bought and sold. However, there has been anecdotal evidence showing that the prompts can be extracted by a user even when they are kept secret. In this paper, we present a framework for systematically measuring the success of prompt extraction attacks. In experiments with multiple sources of prompts and multiple underlying language models, we find that simple text-based attacks can in fact reveal prompts with high probability.

摘要: 大型语言模型的生成通常通过提示技术来控制，其中用户对模型的查询带有旨在指导模型在查询中的行为的提示作为前缀。公司用来指导其模型的提示通常被视为秘密，对进行查询的用户隐藏。它们甚至被视为可以买卖的商品。然而，有坊间证据表明，即使这些提示是保密的，用户也可以提取出来。在本文中，我们提出了一个系统地衡量即时提取攻击的成功程度的框架。在对多种提示来源和多种底层语言模型的实验中，我们发现简单的基于文本的攻击实际上可以很高概率地揭示提示。



## **45. Ethicist: Targeted Training Data Extraction Through Loss Smoothed Soft Prompting and Calibrated Confidence Estimation**

伦理学家：通过损失平滑软提示和校准置信度估计提取有针对性的训练数据 cs.CL

ACL 2023 Long Paper (Main Conference)

**SubmitDate**: 2023-07-10    [abs](http://arxiv.org/abs/2307.04401v1) [paper-pdf](http://arxiv.org/pdf/2307.04401v1)

**Authors**: Zhexin Zhang, Jiaxin Wen, Minlie Huang

**Abstract**: Large pre-trained language models achieve impressive results across many tasks. However, recent works point out that pre-trained language models may memorize a considerable fraction of their training data, leading to the privacy risk of information leakage. In this paper, we propose a method named Ethicist for targeted training data extraction through loss smoothed soft prompting and calibrated confidence estimation, investigating how to recover the suffix in the training data when given a prefix. To elicit memorization in the attacked model, we tune soft prompt embeddings while keeping the model fixed. We further propose a smoothing loss that smooths the loss distribution of the suffix tokens to make it easier to sample the correct suffix. In order to select the most probable suffix from a collection of sampled suffixes and estimate the prediction confidence, we propose a calibrated confidence estimation method, which normalizes the confidence of the generated suffixes with a local estimation. We show that Ethicist significantly improves the extraction performance on a recently proposed public benchmark. We also investigate several factors influencing the data extraction performance, including decoding strategy, model scale, prefix length, and suffix length. Our code is available at https://github.com/thu-coai/Targeted-Data-Extraction.

摘要: 大型预先训练的语言模型在许多任务中取得了令人印象深刻的结果。然而，最近的研究指出，预先训练的语言模型可能会记住相当一部分的训练数据，从而导致信息泄露的隐私风险。本文提出了一种基于丢失平滑软提示和校正置信度估计的针对性训练数据提取方法--伦理法，研究了在给定前缀的情况下如何恢复训练数据中的后缀。为了在被攻击的模型中引起记忆，我们在保持模型不变的情况下调整软提示嵌入。我们进一步提出了一种平滑损失，它平滑了后缀标记的损失分布，从而更容易对正确的后缀进行采样。为了从采样的后缀集合中选择最可能的后缀并估计预测置信度，提出了一种校准置信度估计方法，该方法用局部估计来归一化生成的后缀的置信度。我们表明，在最近提出的公共基准上，伦理学家显著提高了提取性能。我们还研究了影响数据提取性能的几个因素，包括解码策略、模型规模、前缀长度和后缀长度。我们的代码可以在https://github.com/thu-coai/Targeted-Data-Extraction.上找到



## **46. Jailbroken: How Does LLM Safety Training Fail?**

越狱：LLM安全培训是如何失败的？ cs.LG

**SubmitDate**: 2023-07-05    [abs](http://arxiv.org/abs/2307.02483v1) [paper-pdf](http://arxiv.org/pdf/2307.02483v1)

**Authors**: Alexander Wei, Nika Haghtalab, Jacob Steinhardt

**Abstract**: Large language models trained for safety and harmlessness remain susceptible to adversarial misuse, as evidenced by the prevalence of "jailbreak" attacks on early releases of ChatGPT that elicit undesired behavior. Going beyond recognition of the issue, we investigate why such attacks succeed and how they can be created. We hypothesize two failure modes of safety training: competing objectives and mismatched generalization. Competing objectives arise when a model's capabilities and safety goals conflict, while mismatched generalization occurs when safety training fails to generalize to a domain for which capabilities exist. We use these failure modes to guide jailbreak design and then evaluate state-of-the-art models, including OpenAI's GPT-4 and Anthropic's Claude v1.3, against both existing and newly designed attacks. We find that vulnerabilities persist despite the extensive red-teaming and safety-training efforts behind these models. Notably, new attacks utilizing our failure modes succeed on every prompt in a collection of unsafe requests from the models' red-teaming evaluation sets and outperform existing ad hoc jailbreaks. Our analysis emphasizes the need for safety-capability parity -- that safety mechanisms should be as sophisticated as the underlying model -- and argues against the idea that scaling alone can resolve these safety failure modes.

摘要: 经过安全和无害培训的大型语言模型仍然容易受到对手滥用的影响，对早期版本的ChatGPT进行“越狱”攻击的盛行就证明了这一点，这引发了不受欢迎的行为。除了认识到这个问题，我们还调查了此类攻击成功的原因以及如何创建这些攻击。我们假设了安全培训的两种失败模式：目标竞争和不匹配的概括。当模型的能力和安全目标冲突时，就会出现相互竞争的目标，而当安全培训未能概括到存在能力的领域时，就会出现不匹配的泛化。我们使用这些失败模式来指导越狱设计，然后评估最先进的模型，包括OpenAI的GPT-4和Anthropic的Claude v1.3，针对现有的和新设计的攻击。我们发现，尽管在这些模型背后进行了广泛的红色团队和安全培训努力，但漏洞仍然存在。值得注意的是，利用我们的失败模式的新攻击在模型的红团队评估集的不安全请求集合中的每一个提示下都会成功，并且表现优于现有的临时越狱。我们的分析强调了安全能力对等的必要性--安全机制应该与基础模型一样复杂--并反对仅靠扩展就能解决这些安全故障模式的想法。



## **47. SCAT: Robust Self-supervised Contrastive Learning via Adversarial Training for Text Classification**

SCAT：基于对抗性训练的文本分类稳健自监督对比学习 cs.CL

**SubmitDate**: 2023-07-04    [abs](http://arxiv.org/abs/2307.01488v1) [paper-pdf](http://arxiv.org/pdf/2307.01488v1)

**Authors**: Junjie Wu, Dit-Yan Yeung

**Abstract**: Despite their promising performance across various natural language processing (NLP) tasks, current NLP systems are vulnerable to textual adversarial attacks. To defend against these attacks, most existing methods apply adversarial training by incorporating adversarial examples. However, these methods have to rely on ground-truth labels to generate adversarial examples, rendering it impractical for large-scale model pre-training which is commonly used nowadays for NLP and many other tasks. In this paper, we propose a novel learning framework called SCAT (Self-supervised Contrastive Learning via Adversarial Training), which can learn robust representations without requiring labeled data. Specifically, SCAT modifies random augmentations of the data in a fully labelfree manner to generate adversarial examples. Adversarial training is achieved by minimizing the contrastive loss between the augmentations and their adversarial counterparts. We evaluate SCAT on two text classification datasets using two state-of-the-art attack schemes proposed recently. Our results show that SCAT can not only train robust language models from scratch, but it can also significantly improve the robustness of existing pre-trained language models. Moreover, to demonstrate its flexibility, we show that SCAT can also be combined with supervised adversarial training to further enhance model robustness.

摘要: 尽管它们在各种自然语言处理(NLP)任务中具有良好的性能，但当前的自然语言处理系统容易受到文本对手攻击。为了防御这些攻击，现有的大多数方法都是通过结合对抗性例子进行对抗性训练。然而，这些方法必须依赖地面事实标签来生成对抗性实例，这使得目前用于自然语言处理和许多其他任务的大规模模型预训练是不现实的。在本文中，我们提出了一种新的学习框架，称为SCAT(Self-Supervised Contrastive Learning via Aversative Trading)，它可以在不需要标记数据的情况下学习稳健的表示。具体地说，SCAT以完全无标签的方式修改数据的随机增加，以生成对抗性示例。对抗性训练是通过最小化增强器和对抗器之间的对比损失来实现的。我们使用最近提出的两种最先进的攻击方案在两个文本分类数据集上对SCAT进行了评估。结果表明，SCAT不仅可以从头开始训练健壮的语言模型，而且可以显著提高已有的预先训练的语言模型的健壮性。此外，为了展示其灵活性，我们还展示了SCAT还可以与有监督的对抗性训练相结合，以进一步增强模型的稳健性。



## **48. Automatic Counterfactual Augmentation for Robust Text Classification Based on Word-Group Search**

基于词组搜索的稳健文本分类反事实自动增强 cs.CL

13 pages, 7 figures

**SubmitDate**: 2023-07-01    [abs](http://arxiv.org/abs/2307.01214v1) [paper-pdf](http://arxiv.org/pdf/2307.01214v1)

**Authors**: Rui Song, Fausto Giunchiglia, Yingji Li, Hao Xu

**Abstract**: Despite large-scale pre-trained language models have achieved striking results for text classificaion, recent work has raised concerns about the challenge of shortcut learning. In general, a keyword is regarded as a shortcut if it creates a superficial association with the label, resulting in a false prediction. Conversely, shortcut learning can be mitigated if the model relies on robust causal features that help produce sound predictions. To this end, many studies have explored post-hoc interpretable methods to mine shortcuts and causal features for robustness and generalization. However, most existing methods focus only on single word in a sentence and lack consideration of word-group, leading to wrong causal features. To solve this problem, we propose a new Word-Group mining approach, which captures the causal effect of any keyword combination and orders the combinations that most affect the prediction. Our approach bases on effective post-hoc analysis and beam search, which ensures the mining effect and reduces the complexity. Then, we build a counterfactual augmentation method based on the multiple word-groups, and use an adaptive voting mechanism to learn the influence of different augmentated samples on the prediction results, so as to force the model to pay attention to effective causal features. We demonstrate the effectiveness of the proposed method by several tasks on 8 affective review datasets and 4 toxic language datasets, including cross-domain text classificaion, text attack and gender fairness test.

摘要: 尽管大规模的预训练语言模型在文本分类方面取得了显著的效果，但最近的工作也引发了人们对快捷学习挑战的担忧。一般来说，如果关键字与标签建立了表面上的关联，从而导致错误预测，则该关键字被视为快捷方式。相反，如果模型依赖于有助于产生合理预测的健壮的因果特征，则可以减轻快捷学习。为此，许多研究探索了事后可解释的方法来挖掘捷径和因果特征，以获得稳健性和概括性。然而，现有的方法大多只关注句子中的单个词，缺乏对词组的考虑，导致了错误的因果特征。为了解决这个问题，我们提出了一种新的词组挖掘方法，该方法捕捉任何关键字组合的因果关系，并对对预测影响最大的组合进行排序。该方法基于有效的事后分析和波束搜索，既保证了挖掘效果，又降低了算法的复杂度。然后，我们构建了一种基于多词组的反事实增强方法，并使用一种自适应投票机制来学习不同增强样本对预测结果的影响，从而迫使模型关注有效的因果特征。通过在8个情感评论数据集和4个有毒语言数据集上的实验，包括跨领域文本分类、文本攻击和性别公平测试，验证了该方法的有效性。



## **49. Provable Robust Watermarking for AI-Generated Text**

用于人工智能生成文本的可证明稳健水印 cs.CL

**SubmitDate**: 2023-06-30    [abs](http://arxiv.org/abs/2306.17439v1) [paper-pdf](http://arxiv.org/pdf/2306.17439v1)

**Authors**: Xuandong Zhao, Prabhanjan Ananth, Lei Li, Yu-Xiang Wang

**Abstract**: As AI-generated text increasingly resembles human-written content, the ability to detect machine-generated text becomes crucial. To address this challenge, we present GPTWatermark, a robust and high-quality solution designed to ascertain whether a piece of text originates from a specific model. Our approach extends existing watermarking strategies and employs a fixed group design to enhance robustness against editing and paraphrasing attacks. We show that our watermarked language model enjoys strong provable guarantees on generation quality, correctness in detection, and security against evasion attacks. Experimental results on various large language models (LLMs) and diverse datasets demonstrate that our method achieves superior detection accuracy and comparable generation quality in perplexity, thus promoting the responsible use of LLMs.

摘要: 随着人工智能生成的文本越来越像人类书写的内容，检测机器生成的文本的能力变得至关重要。为了应对这一挑战，我们提出了GPTWatermark，这是一个健壮且高质量的解决方案，旨在确定一段文本是否来自特定的模型。我们的方法扩展了现有的水印策略，并采用了固定的组设计来增强对编辑和转译攻击的稳健性。实验表明，我们的水印语言模型在生成质量、检测的正确性和抗规避攻击方面具有很强的可证性保证。在不同的大型语言模型和不同的数据集上的实验结果表明，该方法在困惑情况下获得了较高的检测准确率和相当的生成质量，从而促进了大型语言模型的负责任使用。



## **50. Pick your Poison: Undetectability versus Robustness in Data Poisoning Attacks**

挑选你的毒药：数据中毒攻击中的不可检测性与健壮性 cs.CR

Preprint

**SubmitDate**: 2023-06-29    [abs](http://arxiv.org/abs/2305.09671v2) [paper-pdf](http://arxiv.org/pdf/2305.09671v2)

**Authors**: Nils Lukas, Florian Kerschbaum

**Abstract**: Deep image classification models trained on vast amounts of web-scraped data are susceptible to data poisoning - a mechanism for backdooring models. A small number of poisoned samples seen during training can severely undermine a model's integrity during inference. Existing work considers an effective defense as one that either (i) restores a model's integrity through repair or (ii) detects an attack. We argue that this approach overlooks a crucial trade-off: Attackers can increase robustness at the expense of detectability (over-poisoning) or decrease detectability at the cost of robustness (under-poisoning). In practice, attacks should remain both undetectable and robust. Detectable but robust attacks draw human attention and rigorous model evaluation or cause the model to be re-trained or discarded. In contrast, attacks that are undetectable but lack robustness can be repaired with minimal impact on model accuracy. Our research points to intrinsic flaws in current attack evaluation methods and raises the bar for all data poisoning attackers who must delicately balance this trade-off to remain robust and undetectable. To demonstrate the existence of more potent defenders, we propose defenses designed to (i) detect or (ii) repair poisoned models using a limited amount of trusted image-label pairs. Our results show that an attacker who needs to be robust and undetectable is substantially less threatening. Our defenses mitigate all tested attacks with a maximum accuracy decline of 2% using only 1% of clean data on CIFAR-10 and 2.5% on ImageNet. We demonstrate the scalability of our defenses by evaluating large vision-language models, such as CLIP. Attackers who can manipulate the model's parameters pose an elevated risk as they can achieve higher robustness at low detectability compared to data poisoning attackers.

摘要: 对大量网络抓取的数据进行训练的深度图像分类模型很容易受到数据中毒的影响--这是一种回溯模型的机制。在训练期间看到的少量有毒样本可能会在推理过程中严重破坏模型的完整性。现有的工作认为有效的防御是(I)通过修复恢复模型的完整性或(Ii)检测攻击。我们认为，这种方法忽略了一个关键的权衡：攻击者可以以牺牲可检测性(过度中毒)来增加健壮性，或者以以健壮性(中毒不足)为代价来降低可检测性。在实践中，攻击应该保持不可察觉和强大。可检测但强大的攻击会引起人们的注意和严格的模型评估，或者导致模型被重新训练或丢弃。相比之下，无法检测但缺乏稳健性的攻击可以在对模型精度影响最小的情况下修复。我们的研究指出了当前攻击评估方法的内在缺陷，并提高了所有数据中毒攻击者的门槛，他们必须微妙地平衡这一权衡，以保持健壮和不可检测。为了证明存在更强大的防御者，我们提出了旨在(I)检测或(Ii)修复中毒模型的防御措施，使用有限数量的可信图像-标签对。我们的结果表明，需要健壮且不可检测的攻击者的威胁要小得多。我们的防御系统仅使用CIFAR-10上1%的干净数据和ImageNet上2.5%的干净数据来缓解所有测试的攻击，最大准确率下降2%。我们通过评估大型视觉语言模型(如CLIP)来展示我们防御的可扩展性。可以操纵模型参数的攻击者构成了更高的风险，因为与数据中毒攻击者相比，他们可以在较低的可检测性下实现更高的稳健性。



