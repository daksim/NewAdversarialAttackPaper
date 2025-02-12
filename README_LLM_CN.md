# Latest Large Language Model Attack Papers
**update at 2025-02-12 10:56:34**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. SymGPT: Auditing Smart Contracts via Combining Symbolic Execution with Large Language Models**

SymbGPT：通过将符号执行与大型语言模型相结合来审计智能合同 cs.AI

16 pages. arXiv admin note: text overlap with arXiv:2404.04306

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07644v1) [paper-pdf](http://arxiv.org/pdf/2502.07644v1)

**Authors**: Shihao Xia, Mengting He, Shuai Shao, Tingting Yu, Yiying Zhang, Linhai Song

**Abstract**: To govern smart contracts running on Ethereum, multiple Ethereum Request for Comment (ERC) standards have been developed, each having a set of rules to guide the behaviors of smart contracts. Violating the ERC rules could cause serious security issues and financial loss, signifying the importance of verifying smart contracts follow ERCs. Today's practices of such verification are to manually audit each single contract, use expert-developed program-analysis tools, or use large language models (LLMs), all of which are far from effective in identifying ERC rule violations. This paper introduces SymGPT, a tool that combines the natural language understanding of large language models (LLMs) with the formal guarantees of symbolic execution to automatically verify smart contracts' compliance with ERC rules. To develop SymGPT, we conduct an empirical study of 132 ERC rules from three widely used ERC standards, examining their content, security implications, and natural language descriptions. Based on this study, we design SymGPT by first instructing an LLM to translate ERC rules into a defined EBNF grammar. We then synthesize constraints from the formalized rules to represent scenarios where violations may occur and use symbolic execution to detect them. Our evaluation shows that SymGPT identifies 5,783 ERC rule violations in 4,000 real-world contracts, including 1,375 violations with clear attack paths for stealing financial assets, demonstrating its effectiveness. Furthermore, SymGPT outperforms six automated techniques and a security-expert auditing service, underscoring its superiority over current smart contract analysis methods.

摘要: 为了管理在以太上运行的智能合同，已经开发了多个以太征求意见(ERC)标准，每个标准都有一套规则来指导智能合同的行为。违反ERC规则可能会导致严重的安全问题和经济损失，这意味着验证智能合同遵循ERC的重要性。如今，这种验证的做法是手动审计每一份合同，使用专家开发的程序分析工具，或者使用大型语言模型(LLM)，所有这些都远远不能有效地识别违反ERC规则的行为。本文介绍了一种将大型语言模型的自然语言理解与符号执行的形式保证相结合的工具--SymGPT，用于自动验证智能合约是否符合ERC规则。为了开发SymGPT，我们对来自三个广泛使用的ERC标准的132条ERC规则进行了实证研究，检查了它们的内容、安全含义和自然语言描述。在此研究的基础上，我们首先通过指示LLM将ERC规则转换为定义的EBNF语法来设计SymGPT。然后，我们从形式化的规则中合成约束来表示可能发生违规的场景，并使用符号执行来检测它们。我们的评估显示，SymGPT在4,000份真实合同中识别了5,783项违反ERC规则的行为，其中1,375项违规行为具有明确的窃取金融资产的攻击路径，证明了其有效性。此外，SymGPT的性能超过了六项自动化技术和安全专家审计服务，突显了其相对于当前智能合同分析方法的优势。



## **2. JBShield: Defending Large Language Models from Jailbreak Attacks through Activated Concept Analysis and Manipulation**

JBShield：通过激活概念分析和操纵保护大型语言模型免受越狱攻击 cs.CR

To Appear in the 34rd USENIX Security Symposium, August 13-15, 2025

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07557v1) [paper-pdf](http://arxiv.org/pdf/2502.07557v1)

**Authors**: Shenyi Zhang, Yuchen Zhai, Keyan Guo, Hongxin Hu, Shengnan Guo, Zheng Fang, Lingchen Zhao, Chao Shen, Cong Wang, Qian Wang

**Abstract**: Despite the implementation of safety alignment strategies, large language models (LLMs) remain vulnerable to jailbreak attacks, which undermine these safety guardrails and pose significant security threats. Some defenses have been proposed to detect or mitigate jailbreaks, but they are unable to withstand the test of time due to an insufficient understanding of jailbreak mechanisms. In this work, we investigate the mechanisms behind jailbreaks based on the Linear Representation Hypothesis (LRH), which states that neural networks encode high-level concepts as subspaces in their hidden representations. We define the toxic semantics in harmful and jailbreak prompts as toxic concepts and describe the semantics in jailbreak prompts that manipulate LLMs to comply with unsafe requests as jailbreak concepts. Through concept extraction and analysis, we reveal that LLMs can recognize the toxic concepts in both harmful and jailbreak prompts. However, unlike harmful prompts, jailbreak prompts activate the jailbreak concepts and alter the LLM output from rejection to compliance. Building on our analysis, we propose a comprehensive jailbreak defense framework, JBShield, consisting of two key components: jailbreak detection JBShield-D and mitigation JBShield-M. JBShield-D identifies jailbreak prompts by determining whether the input activates both toxic and jailbreak concepts. When a jailbreak prompt is detected, JBShield-M adjusts the hidden representations of the target LLM by enhancing the toxic concept and weakening the jailbreak concept, ensuring LLMs produce safe content. Extensive experiments demonstrate the superior performance of JBShield, achieving an average detection accuracy of 0.95 and reducing the average attack success rate of various jailbreak attacks to 2% from 61% across distinct LLMs.

摘要: 尽管实施了安全调整战略，但大型语言模型(LLM)仍然容易受到越狱攻击，这些攻击破坏了这些安全护栏，并构成了重大的安全威胁。已经提出了一些防御措施来检测或减轻越狱，但由于对越狱机制的了解不足，这些防御措施无法经受住时间的考验。在这项工作中，我们基于线性表示假说(LRH)来研究越狱背后的机制，该假说指出，神经网络将高级概念编码为其隐藏表示中的子空间。我们将有害提示和越狱提示中的有毒语义定义为有毒概念，并将操纵LLM遵从不安全请求的越狱提示中的语义描述为越狱概念。通过概念提取和分析，我们发现LLMS能够识别有害提示和越狱提示中的有毒概念。然而，与有害的提示不同，越狱提示激活了越狱概念，并将LLM输出从拒绝更改为遵守。基于我们的分析，我们提出了一个全面的越狱防御框架JBShield，它由两个关键组件组成：越狱检测JBShield-D和缓解JBShield-M。JBShield-D通过确定输入是否同时激活有毒和越狱概念来识别越狱提示。当检测到越狱提示时，JBShield-M通过增强有毒概念和弱化越狱概念来调整目标LLM的隐藏表示，确保LLM产生安全的内容。大量的实验证明了JBShield的优越性能，平均检测准确率达到0.95，并将不同LLM上各种越狱攻击的平均攻击成功率从61%降低到2%。



## **3. LUNAR: LLM Unlearning via Neural Activation Redirection**

LUNAR：LLM通过神经激活重定向消除学习 cs.LG

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07218v1) [paper-pdf](http://arxiv.org/pdf/2502.07218v1)

**Authors**: William F. Shen, Xinchi Qiu, Meghdad Kurmanji, Alex Iacob, Lorenzo Sani, Yihong Chen, Nicola Cancedda, Nicholas D. Lane

**Abstract**: Large Language Models (LLMs) benefit from training on ever larger amounts of textual data, but as a result, they increasingly incur the risk of leaking private information. The ability to selectively remove knowledge from LLMs is, therefore, a highly desirable capability. In this paper, we propose LUNAR, a novel unlearning methodology grounded in the Linear Representation Hypothesis. LUNAR operates by redirecting the representations of unlearned data to regions that trigger the model's inherent ability to express its inability to answer. LUNAR achieves state-of-the-art unlearning performance while significantly enhancing the controllability of the unlearned model during inference. Specifically, LUNAR achieves between 2.9x to 11.7x improvements on combined "unlearning efficacy" and "model utility" score ("Deviation Score") on the PISTOL dataset across various base models. We also demonstrate, through quantitative analysis and qualitative examples, LUNAR's superior controllability in generating coherent and contextually aware responses, mitigating undesired side effects of existing methods. Moreover, we demonstrate that LUNAR is robust against white-box adversarial attacks and versatile in handling real-world scenarios, such as processing sequential unlearning requests.

摘要: 大型语言模型(LLM)受益于对越来越多的文本数据进行培训，但结果是，它们越来越多地招致泄露私人信息的风险。因此，有选择地从LLM中移除知识的能力是一种非常理想的能力。在本文中，我们提出了一种基于线性表征假设的去学习方法--LUNAR。LUNAR通过将未学习数据的表示重定向到触发模型表达其无法回答问题的固有能力的区域来运行。LUNAR实现了最先进的遗忘性能，同时显著增强了推理过程中未学习模型的可控性。具体地说，在各种基本型号的手枪数据集上，LUNAR在组合的“遗忘效能”和“模型效用”分数(“偏差分数”)上取得了2.9倍到11.7倍的改进。我们还通过定量分析和定性例子证明，月球在产生连贯的和上下文感知的响应方面具有优越的可控性，减轻了现有方法的不良副作用。此外，我们还证明了LUNAR对白盒攻击具有很强的健壮性，并且在处理真实场景(如处理顺序遗忘请求)方面具有很强的通用性。



## **4. LLM Agent Honeypot: Monitoring AI Hacking Agents in the Wild**

LLM Agent Honeypot：监控野外人工智能黑客代理 cs.CR

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2410.13919v2) [paper-pdf](http://arxiv.org/pdf/2410.13919v2)

**Authors**: Reworr, Dmitrii Volkov

**Abstract**: Attacks powered by Large Language Model (LLM) agents represent a growing threat to modern cybersecurity. To address this concern, we present LLM Honeypot, a system designed to monitor autonomous AI hacking agents. By augmenting a standard SSH honeypot with prompt injection and time-based analysis techniques, our framework aims to distinguish LLM agents among all attackers. Over a trial deployment of about three months in a public environment, we collected 8,130,731 hacking attempts and 8 potential AI agents. Our work demonstrates the emergence of AI-driven threats and their current level of usage, serving as an early warning of malicious LLM agents in the wild.

摘要: 由大型语言模型（LLM）代理支持的攻击对现代网络安全构成了日益严重的威胁。为了解决这个问题，我们提出了LLM Honeypot，这是一个旨在监控自主人工智能黑客代理的系统。通过使用即时注入和基于时间的分析技术来扩展标准的SSH蜜罐，我们的框架旨在区分LLM代理在所有攻击者中。在公共环境中进行了大约三个月的试验部署，我们收集了8，130，731次黑客尝试和8个潜在的人工智能代理。我们的工作展示了人工智能驱动的威胁的出现及其当前的使用水平，作为野外恶意LLM代理的预警。



## **5. AdaPhish: AI-Powered Adaptive Defense and Education Resource Against Deceptive Emails**

AdaPhish：针对欺骗性电子邮件的人工智能驱动自适应防御和教育资源 cs.CR

7 pages, 3 figures, 2 tables, accepted in 4th IEEE International  Conference on AI in Cybersecurity (ICAIC)

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.03622v2) [paper-pdf](http://arxiv.org/pdf/2502.03622v2)

**Authors**: Rei Meguro, Ng S. T. Chong

**Abstract**: Phishing attacks remain a significant threat in the digital age, yet organizations lack effective methods to tackle phishing attacks without leaking sensitive information. Phish bowl initiatives are a vital part of cybersecurity efforts against these attacks. However, traditional phish bowls require manual anonymization and are often limited to internal use. To overcome these limitations, we introduce AdaPhish, an AI-powered phish bowl platform that automatically anonymizes and analyzes phishing emails using large language models (LLMs) and vector databases. AdaPhish achieves real-time detection and adaptation to new phishing tactics while enabling long-term tracking of phishing trends. Through automated reporting, adaptive analysis, and real-time alerts, AdaPhish presents a scalable, collaborative solution for phishing detection and cybersecurity education.

摘要: 网络钓鱼攻击仍然是数字时代的重大威胁，但组织缺乏有效的方法来在不泄露敏感信息的情况下应对网络钓鱼攻击。Phish bowl计划是针对这些攻击的网络安全工作的重要组成部分。然而，传统的钓鱼碗需要手动匿名化，并且通常仅限于内部使用。为了克服这些限制，我们引入了AdaPhish，这是一个人工智能驱动的钓鱼碗平台，可以使用大型语言模型（LLM）和载体数据库自动匿名化和分析网络钓鱼电子邮件。AdaPhish实现了实时检测和适应新的网络钓鱼策略，同时能够长期跟踪网络钓鱼趋势。通过自动报告、自适应分析和实时警报，AdaPhish为网络钓鱼检测和网络安全教育提供了可扩展的协作解决方案。



## **6. Tamper-Resistant Safeguards for Open-Weight LLMs**

开放重量LLM的防篡改保障措施 cs.LG

Website: https://www.tamper-resistant-safeguards.com

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2408.00761v4) [paper-pdf](http://arxiv.org/pdf/2408.00761v4)

**Authors**: Rishub Tamirisa, Bhrugu Bharathi, Long Phan, Andy Zhou, Alice Gatti, Tarun Suresh, Maxwell Lin, Justin Wang, Rowan Wang, Ron Arel, Andy Zou, Dawn Song, Bo Li, Dan Hendrycks, Mantas Mazeika

**Abstract**: Rapid advances in the capabilities of large language models (LLMs) have raised widespread concerns regarding their potential for malicious use. Open-weight LLMs present unique challenges, as existing safeguards lack robustness to tampering attacks that modify model weights. For example, recent works have demonstrated that refusal and unlearning safeguards can be trivially removed with a few steps of fine-tuning. These vulnerabilities necessitate new approaches for enabling the safe release of open-weight LLMs. We develop a method, called TAR, for building tamper-resistant safeguards into open-weight LLMs such that adversaries cannot remove the safeguards even after hundreds of steps of fine-tuning. In extensive evaluations and red teaming analyses, we find that our method greatly improves tamper-resistance while preserving benign capabilities. Our results demonstrate that progress on tamper-resistance is possible, opening up a promising new avenue to improve the safety and security of open-weight LLMs.

摘要: 大型语言模型(LLM)功能的快速发展引起了人们对其潜在恶意使用的广泛关注。开放重量LLM提出了独特的挑战，因为现有的保障措施缺乏对篡改模型权重的篡改攻击的稳健性。例如，最近的研究表明，通过几个步骤的微调，就可以很容易地消除拒绝和遗忘的保障措施。这些漏洞需要新的方法来实现安全释放未加重量的低密度脂蛋白。我们开发了一种名为TAR的方法，用于在开放重量的LLM中构建防篡改保护措施，以便即使在数百个步骤的微调之后，对手也无法移除这些保护措施。在广泛的评估和红团队分析中，我们发现我们的方法在保持良性性能的同时大大提高了防篡改能力。我们的结果表明，在防篡改方面取得进展是可能的，为提高开放重量LLMS的安全性开辟了一条有希望的新途径。



## **7. Exploring Audio Editing Features as User-Centric Privacy Defenses Against Large Language Model(LLM) Based Emotion Inference Attacks**

探索音频编辑功能，以用户为中心的隐私防御基于大型语言模型（LLM）的情感推理攻击 cs.CR

Accepted for presentation(Poster) at PPAI-25: The 6th AAAI Workshop  on Privacy-Preserving Artificial Intelligence

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2501.18727v2) [paper-pdf](http://arxiv.org/pdf/2501.18727v2)

**Authors**: Mohd. Farhan Israk Soumik, W. K. M. Mithsara, Abdur R. Shahid, Ahmed Imteaj

**Abstract**: The rapid proliferation of speech-enabled technologies, including virtual assistants, video conferencing platforms, and wearable devices, has raised significant privacy concerns, particularly regarding the inference of sensitive emotional information from audio data. Existing privacy-preserving methods often compromise usability and security, limiting their adoption in practical scenarios. This paper introduces a novel, user-centric approach that leverages familiar audio editing techniques, specifically pitch and tempo manipulation, to protect emotional privacy without sacrificing usability. By analyzing popular audio editing applications on Android and iOS platforms, we identified these features as both widely available and usable. We rigorously evaluated their effectiveness against a threat model, considering adversarial attacks from diverse sources, including Deep Neural Networks (DNNs), Large Language Models (LLMs), and and reversibility testing. Our experiments, conducted on three distinct datasets, demonstrate that pitch and tempo manipulation effectively obfuscates emotional data. Additionally, we explore the design principles for lightweight, on-device implementation to ensure broad applicability across various devices and platforms.

摘要: 包括虚拟助理、视频会议平台和可穿戴设备在内的语音支持技术的迅速普及引发了人们对隐私的严重担忧，特别是关于从音频数据推断敏感情感信息的问题。现有的隐私保护方法往往会损害可用性和安全性，限制了它们在实际场景中的采用。本文介绍了一种新颖的、以用户为中心的方法，该方法利用熟悉的音频编辑技术，特别是音调和节奏操作，在不牺牲可用性的情况下保护情感隐私。通过分析Android和iOS平台上流行的音频编辑应用程序，我们发现这些功能广泛使用和使用。我们严格评估了它们对威胁模型的有效性，考虑了来自不同来源的对抗性攻击，包括深度神经网络(DNN)、大型语言模型(LLMS)和可逆性测试。我们在三个不同的数据集上进行的实验表明，音调和节奏操作有效地混淆了情感数据。此外，我们还探讨了轻量级设备上实施的设计原则，以确保跨各种设备和平台的广泛适用性。



## **8. Preserving Privacy in Large Language Models: A Survey on Current Threats and Solutions**

在大型语言模型中保护隐私：对当前威胁和解决方案的调查 cs.CR

Published in Transactions on Machine Learning Research (TMLR)  https://openreview.net/forum?id=Ss9MTTN7OL

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2408.05212v2) [paper-pdf](http://arxiv.org/pdf/2408.05212v2)

**Authors**: Michele Miranda, Elena Sofia Ruzzetti, Andrea Santilli, Fabio Massimo Zanzotto, Sébastien Bratières, Emanuele Rodolà

**Abstract**: Large Language Models (LLMs) represent a significant advancement in artificial intelligence, finding applications across various domains. However, their reliance on massive internet-sourced datasets for training brings notable privacy issues, which are exacerbated in critical domains (e.g., healthcare). Moreover, certain application-specific scenarios may require fine-tuning these models on private data. This survey critically examines the privacy threats associated with LLMs, emphasizing the potential for these models to memorize and inadvertently reveal sensitive information. We explore current threats by reviewing privacy attacks on LLMs and propose comprehensive solutions for integrating privacy mechanisms throughout the entire learning pipeline. These solutions range from anonymizing training datasets to implementing differential privacy during training or inference and machine unlearning after training. Our comprehensive review of existing literature highlights ongoing challenges, available tools, and future directions for preserving privacy in LLMs. This work aims to guide the development of more secure and trustworthy AI systems by providing a thorough understanding of privacy preservation methods and their effectiveness in mitigating risks.

摘要: 大型语言模型(LLM)代表了人工智能的一项重大进步，可以找到跨各个领域的应用程序。然而，他们对来自互联网的海量数据集的依赖带来了显著的隐私问题，在关键领域(例如医疗保健)，这一问题更加严重。此外，某些特定于应用程序的场景可能需要根据私有数据对这些模型进行微调。这项调查严格审查了与LLMS相关的隐私威胁，强调了这些模型可能会记住并无意中泄露敏感信息。我们通过审查对LLM的隐私攻击来探索当前的威胁，并提出全面的解决方案，将隐私机制整合到整个学习管道中。这些解决方案的范围从匿名训练数据集到在训练期间实现差异隐私或在训练后进行推理和机器遗忘。我们对现有文献的全面回顾突出了在LLMS中保护隐私的持续挑战、可用的工具和未来的方向。这项工作旨在通过提供对隐私保护方法及其在降低风险方面的有效性的透彻了解，来指导更安全和值得信赖的人工智能系统的开发。



## **9. Jailbreaking LLMs' Safeguard with Universal Magic Words for Text Embedding Models**

越狱的LLM为文本嵌入模型提供通用魔法词的保障 cs.CL

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2501.18280v2) [paper-pdf](http://arxiv.org/pdf/2501.18280v2)

**Authors**: Haoyu Liang, Youran Sun, Yunfeng Cai, Jun Zhu, Bo Zhang

**Abstract**: The security issue of large language models (LLMs) has gained significant attention recently, with various defense mechanisms developed to prevent harmful outputs, among which safeguards based on text embedding models serve as a fundamental defense. Through testing, we discover that the distribution of text embedding model outputs is significantly biased with a large mean. Inspired by this observation, we propose novel efficient methods to search for universal magic words that can attack text embedding models. The universal magic words as suffixes can move the embedding of any text towards the bias direction, therefore manipulate the similarity of any text pair and mislead safeguards. By appending magic words to user prompts and requiring LLMs to end answers with magic words, attackers can jailbreak the safeguard. To eradicate this security risk, we also propose defense mechanisms against such attacks, which can correct the biased distribution of text embeddings in a train-free manner.

摘要: 大型语言模型（LLM）的安全问题最近受到了广泛关注，各种防御机制被开发出来来防止有害输出，其中基于文本嵌入模型的保护措施是基本防御。通过测试，我们发现文本嵌入模型输出的分布存在显着偏差，平均值很大。受这一观察的启发，我们提出了新颖的有效方法来搜索可以攻击文本嵌入模型的通用魔法词。作为后缀的通用神奇词可以将任何文本的嵌入移向偏向方向，从而操纵任何文本对的相似性并误导保障措施。通过在用户提示中添加魔法词并要求LLM以魔法词结束回答，攻击者可以越狱该保护措施。为了消除这种安全风险，我们还提出了针对此类攻击的防御机制，该机制可以以无训练的方式纠正文本嵌入的偏见分布。



## **10. Panza: Design and Analysis of a Fully-Local Personalized Text Writing Assistant**

Panza：全本地个性化文本写作助手的设计与分析 cs.CL

Panza is available at https://github.com/IST-DASLab/PanzaMail

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2407.10994v4) [paper-pdf](http://arxiv.org/pdf/2407.10994v4)

**Authors**: Armand Nicolicioiu, Eugenia Iofinova, Andrej Jovanovic, Eldar Kurtic, Mahdi Nikdan, Andrei Panferov, Ilia Markov, Nir Shavit, Dan Alistarh

**Abstract**: The availability of powerful open-source large language models (LLMs) opens exciting use-cases, such as using personal data to fine-tune these models to imitate a user's unique writing style. Two key requirements for such assistants are personalization - in the sense that the assistant should recognizably reflect the user's own writing style - and privacy - users may justifiably be wary of uploading extremely personal data, such as their email archive, to a third-party service. In this paper, we present a new design and evaluation for such an automated assistant, for the specific use case of email generation, which we call Panza. Panza's personalization features are based on a combination of fine-tuning using a variant of the Reverse Instructions technique together with Retrieval-Augmented Generation (RAG). We demonstrate that this combination allows us to fine-tune an LLM to reflect a user's writing style using limited data, while executing on extremely limited resources, e.g. on a free Google Colab instance. Our key methodological contribution is the first detailed study of evaluation metrics for this personalized writing task, and of how different choices of system components--the use of RAG and of different fine-tuning approaches-impact the system's performance. Additionally, we demonstrate that very little data - under 100 email samples - are sufficient to create models that convincingly imitate humans. This finding showcases a previously-unknown attack vector in language models - that access to a small number of writing samples can allow a bad actor to cheaply create generative models that imitate a target's writing style. We are releasing the full Panza code as well as three new email datasets licensed for research use at https://github.com/IST-DASLab/PanzaMail.

摘要: 强大的开源大型语言模型(LLM)的出现开启了令人兴奋的用例，例如使用个人数据来微调这些模型，以模仿用户独特的写作风格。这类助手的两个关键要求是个性化--从这个意义上说，助手应该明显地反映用户自己的写作风格--以及隐私--用户可能有理由对将极其个人化的数据，如他们的电子邮件档案，上传到第三方服务持谨慎态度。在本文中，我们提出了一种新的自动化助手的设计和评估，针对电子邮件生成的特定用例，我们称之为Panza。Panza的个性化功能是基于使用反向指令技术的变体进行微调以及检索-增强生成(RAG)的组合。我们证明，这种组合允许我们使用有限的数据微调LLM以反映用户的写作风格，同时在极其有限的资源上执行，例如在免费的Google Colab实例上执行。我们的主要方法论贡献是首次详细研究了这种个性化写作任务的评估指标，以及系统组件的不同选择--使用RAG和不同的微调方法--如何影响系统的性能。此外，我们还展示了极少的数据--不到100个电子邮件样本--足以创建令人信服地模仿人类的模型。这一发现揭示了语言模型中一种以前不为人知的攻击媒介--获取少量写作样本可以让糟糕的演员廉价地创建模仿目标写作风格的生成模型。我们将发布完整的PANZA代码以及三个新的电子邮件数据集，这些数据集已被授权用于https://github.com/IST-DASLab/PanzaMail.的研究



## **11. Detecting Backdoor Samples in Contrastive Language Image Pretraining**

对比语言图像预训练中后门样本检测 cs.LG

ICLR2025

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.01385v2) [paper-pdf](http://arxiv.org/pdf/2502.01385v2)

**Authors**: Hanxun Huang, Sarah Erfani, Yige Li, Xingjun Ma, James Bailey

**Abstract**: Contrastive language-image pretraining (CLIP) has been found to be vulnerable to poisoning backdoor attacks where the adversary can achieve an almost perfect attack success rate on CLIP models by poisoning only 0.01\% of the training dataset. This raises security concerns on the current practice of pretraining large-scale models on unscrutinized web data using CLIP. In this work, we analyze the representations of backdoor-poisoned samples learned by CLIP models and find that they exhibit unique characteristics in their local subspace, i.e., their local neighborhoods are far more sparse than that of clean samples. Based on this finding, we conduct a systematic study on detecting CLIP backdoor attacks and show that these attacks can be easily and efficiently detected by traditional density ratio-based local outlier detectors, whereas existing backdoor sample detection methods fail. Our experiments also reveal that an unintentional backdoor already exists in the original CC3M dataset and has been trained into a popular open-source model released by OpenCLIP. Based on our detector, one can clean up a million-scale web dataset (e.g., CC3M) efficiently within 15 minutes using 4 Nvidia A100 GPUs. The code is publicly available in our \href{https://github.com/HanxunH/Detect-CLIP-Backdoor-Samples}{GitHub repository}.

摘要: 对比语言图像预训练(CLIP)被发现容易受到中毒后门攻击，对手只需中毒0.01%的训练数据集就可以在CLIP模型上获得近乎完美的攻击成功率。这引发了人们对当前使用CLIP对大规模模型进行未经审查的网络数据的预培训的安全担忧。在这项工作中，我们分析了通过CLIP模型学习的后门中毒样本的表示，发现它们在局部子空间中表现出独特的特征，即它们的局部邻域比干净样本的局部邻域稀疏得多。基于这一发现，我们对CLIP后门攻击的检测进行了系统的研究，结果表明，传统的基于密度比的局部离群点检测方法可以轻松有效地检测到这些攻击，而现有的后门样本检测方法却无法检测到这些攻击。我们的实验还表明，在原始的CC3M数据集中已经存在一个无意的后门，并且已经被训练成OpenCLIP发布的流行的开源模型。基于我们的检测器，您可以使用4个NVIDIA A100图形处理器在15分钟内高效清理百万级Web数据集(例如CC3M)。代码在我们的\href{https://github.com/HanxunH/Detect-CLIP-Backdoor-Samples}{GitHub存储库中公开提供。



## **12. Confidence Elicitation: A New Attack Vector for Large Language Models**

信心激发：大型语言模型的新攻击载体 cs.LG

Published in ICLR 2025. The code is publicly available at  https://github.com/Aniloid2/Confidence_Elicitation_Attacks

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.04643v2) [paper-pdf](http://arxiv.org/pdf/2502.04643v2)

**Authors**: Brian Formento, Chuan Sheng Foo, See-Kiong Ng

**Abstract**: A fundamental issue in deep learning has been adversarial robustness. As these systems have scaled, such issues have persisted. Currently, large language models (LLMs) with billions of parameters suffer from adversarial attacks just like their earlier, smaller counterparts. However, the threat models have changed. Previously, having gray-box access, where input embeddings or output logits/probabilities were visible to the user, might have been reasonable. However, with the introduction of closed-source models, no information about the model is available apart from the generated output. This means that current black-box attacks can only utilize the final prediction to detect if an attack is successful. In this work, we investigate and demonstrate the potential of attack guidance, akin to using output probabilities, while having only black-box access in a classification setting. This is achieved through the ability to elicit confidence from the model. We empirically show that the elicited confidence is calibrated and not hallucinated for current LLMs. By minimizing the elicited confidence, we can therefore increase the likelihood of misclassification. Our new proposed paradigm demonstrates promising state-of-the-art results on three datasets across two models (LLaMA-3-8B-Instruct and Mistral-7B-Instruct-V0.3) when comparing our technique to existing hard-label black-box attack methods that introduce word-level substitutions.

摘要: 深度学习的一个基本问题是对手的稳健性。随着这些系统的规模扩大，这样的问题一直存在。目前，具有数十亿个参数的大型语言模型(LLM)就像它们早期的较小对应模型一样，受到对手攻击。然而，威胁模型已经发生了变化。以前，拥有灰箱访问，其中输入嵌入或输出日志/概率对用户可见，可能是合理的。然而，随着闭源模型的引入，除了生成的输出之外，没有关于模型的信息可用。这意味着当前的黑盒攻击只能利用最终预测来检测攻击是否成功。在这项工作中，我们调查和演示了攻击指导的潜力，类似于使用输出概率，而在分类设置中只有黑盒访问。这是通过从模型中获得信心的能力来实现的。我们的经验表明，对于当前的LLM来说，引发的信心是经过校准的，而不是幻觉的。因此，通过将引起的置信度降至最低，我们可以增加错误分类的可能性。我们提出的新范式在两个模型(骆驼-3-8B-指令和Mistral-7B-指令-V0.3)的三个数据集上展示了有希望的最先进结果，当将我们的技术与现有的引入词级替换的硬标签黑盒攻击方法进行比较时。



## **13. A Practical Examination of AI-Generated Text Detectors for Large Language Models**

大型语言模型的人工智能生成文本检测器的实践检验 cs.CL

9 pages

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2412.05139v4) [paper-pdf](http://arxiv.org/pdf/2412.05139v4)

**Authors**: Brian Tufts, Xuandong Zhao, Lei Li

**Abstract**: The proliferation of large language models has raised growing concerns about their misuse, particularly in cases where AI-generated text is falsely attributed to human authors. Machine-generated content detectors claim to effectively identify such text under various conditions and from any language model. This paper critically evaluates these claims by assessing several popular detectors (RADAR, Wild, T5Sentinel, Fast-DetectGPT, PHD, LogRank, Binoculars) on a range of domains, datasets, and models that these detectors have not previously encountered. We employ various prompting strategies to simulate practical adversarial attacks, demonstrating that even moderate efforts can significantly evade detection. We emphasize the importance of the true positive rate at a specific false positive rate (TPR@FPR) metric and demonstrate that these detectors perform poorly in certain settings, with TPR@.01 as low as 0%. Our findings suggest that both trained and zero-shot detectors struggle to maintain high sensitivity while achieving a reasonable true positive rate.

摘要: 大型语言模型的激增引发了人们对它们滥用的日益担忧，特别是在人工智能生成的文本被错误地归因于人类作者的情况下。机器生成的内容检测器声称可以在各种条件下从任何语言模型有效地识别此类文本。本文通过评估几种流行的探测器(雷达、Wild、T5Sentinel、Fast-DetectGPT、PHD、logrank、双筒望远镜)，对这些声称进行了批判性的评估，这些探测器以前从未遇到过。我们使用各种提示策略来模拟实际的对抗性攻击，表明即使是适度的攻击也可以显著地躲避检测。我们强调了在特定的假阳性率(TPR@fPR)度量下的真阳性率的重要性，并证明了这些检测器在某些设置下表现很差，TPR@.01低至0%。我们的发现表明，训练有素的探测器和零射探测器都很难在保持高灵敏度的同时获得合理的真阳性率。



## **14. Cyri: A Conversational AI-based Assistant for Supporting the Human User in Detecting and Responding to Phishing Attacks**

Cyri：一款基于对话的人工智能助手，支持人类用户检测和响应网络钓鱼攻击 cs.HC

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2502.05951v1) [paper-pdf](http://arxiv.org/pdf/2502.05951v1)

**Authors**: Antonio La Torre, Marco Angelini

**Abstract**: This work introduces Cyri, an AI-powered conversational assistant designed to support a human user in detecting and analyzing phishing emails by leveraging Large Language Models. Cyri has been designed to scrutinize emails for semantic features used in phishing attacks, such as urgency, and undesirable consequences, using an approach that unifies features already established in the literature with others by Cyri features extraction methodology. Cyri can be directly plugged into a client mail or webmail, ensuring seamless integration with the user's email workflow while maintaining data privacy through local processing. By performing analyses on the user's machine, Cyri eliminates the need to transmit sensitive email data over the internet, reducing associated security risks. The Cyri user interface has been designed to reduce habituation effects and enhance user engagement. It employs dynamic visual cues and context-specific explanations to keep users alert and informed while using emails. Additionally, it allows users to explore identified malicious semantic features both through conversation with the agent and visual exploration, obtaining the advantages of both modalities for expert or non-expert users. It also allows users to keep track of the conversation, supports the user in solving additional questions on both computed features or new parts of the mail, and applies its detection on demand. To evaluate Cyri, we crafted a comprehensive dataset of 420 phishing emails and 420 legitimate emails. Results demonstrate high effectiveness in identifying critical phishing semantic features fundamental to phishing detection. A user study involving 10 participants, both experts and non-experts, evaluated Cyri's effectiveness and usability. Results indicated that Cyri significantly aided users in identifying phishing emails and enhanced their understanding of phishing tactics.

摘要: 这项工作介绍了Cyri，这是一个人工智能支持的对话助手，旨在支持人类用户通过利用大型语言模型来检测和分析钓鱼电子邮件。Cyri被设计用于仔细检查电子邮件中用于网络钓鱼攻击的语义特征，如紧迫性和不良后果，使用一种方法，通过Cyri特征提取方法将文献中已建立的特征与其他特征统一起来。Cyri可以直接插入客户端邮件或网络邮件，确保与用户的电子邮件工作流程无缝集成，同时通过本地处理保持数据隐私。通过在用户机器上执行分析，Cyri消除了通过互联网传输敏感电子邮件数据的需要，从而降低了相关的安全风险。Cyri的用户界面旨在减少习惯性影响，提高用户参与度。它使用动态视觉提示和特定于上下文的解释来保持用户在使用电子邮件时的警觉和通知。此外，它允许用户通过与代理的对话和视觉探索来探索已识别的恶意语义特征，从而获得专家或非专家用户的这两种模式的优势。它还允许用户跟踪对话，支持用户解决有关计算功能或邮件新部分的其他问题，并按需应用其检测。为了评估Cyri，我们精心制作了一个包含420封钓鱼电子邮件和420封合法电子邮件的全面数据集。结果表明，在识别网络钓鱼检测基础上的关键网络钓鱼语义特征方面具有很高的效率。一项涉及10名参与者的用户研究，包括专家和非专家，评估了Cyri的有效性和可用性。结果表明，Cyri显著地帮助用户识别钓鱼电子邮件，并增强了他们对钓鱼策略的理解。



## **15. Large Language Models are Easily Confused: A Quantitative Metric, Security Implications and Typological Analysis**

大型语言模型很容易混淆：量化指标、安全含义和类型学分析 cs.CL

18 pages, 15 figures, 14 tables

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2410.13237v2) [paper-pdf](http://arxiv.org/pdf/2410.13237v2)

**Authors**: Yiyi Chen, Qiongxiu Li, Russa Biswas, Johannes Bjerva

**Abstract**: Language Confusion is a phenomenon where Large Language Models (LLMs) generate text that is neither in the desired language, nor in a contextually appropriate language. This phenomenon presents a critical challenge in text generation by LLMs, often appearing as erratic and unpredictable behavior. We hypothesize that there are linguistic regularities to this inherent vulnerability in LLMs and shed light on patterns of language confusion across LLMs. We introduce a novel metric, Language Confusion Entropy, designed to directly measure and quantify this confusion, based on language distributions informed by linguistic typology and lexical variation. Comprehensive comparisons with the Language Confusion Benchmark (Marchisio et al., 2024) confirm the effectiveness of our metric, revealing patterns of language confusion across LLMs. We further link language confusion to LLM security, and find patterns in the case of multilingual embedding inversion attacks. Our analysis demonstrates that linguistic typology offers theoretically grounded interpretation, and valuable insights into leveraging language similarities as a prior for LLM alignment and security.

摘要: 语言混淆是一种现象，大型语言模型(LLM)生成的文本既不是所需语言的文本，也不是上下文合适的语言文本。这一现象对LLMS的文本生成提出了严重的挑战，通常表现为不稳定和不可预测的行为。我们假设LLMS中这种固有的脆弱性存在语言规则，并揭示了LLMS中语言混淆的模式。我们引入了一种新的度量，语言混淆熵，旨在根据语言类型和词汇变异提供的语言分布来直接度量和量化这种混淆。与语言混淆基准(Marchisio等人，2024年)的全面比较证实了我们的度量的有效性，揭示了LLM之间的语言混淆模式。我们进一步将语言混淆与LLM安全联系起来，并在多语言嵌入反转攻击的情况下找到了模式。我们的分析表明，语言类型学提供了理论上的解释，并提供了关于利用语言相似性作为LLM对齐和安全的先决条件的有价值的见解。



## **16. Arabic Dataset for LLM Safeguard Evaluation**

LLM保障评估的阿拉伯数据集 cs.CL

Accepted at NAACL 2025 Main Conference

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2410.17040v2) [paper-pdf](http://arxiv.org/pdf/2410.17040v2)

**Authors**: Yasser Ashraf, Yuxia Wang, Bin Gu, Preslav Nakov, Timothy Baldwin

**Abstract**: The growing use of large language models (LLMs) has raised concerns regarding their safety. While many studies have focused on English, the safety of LLMs in Arabic, with its linguistic and cultural complexities, remains under-explored. Here, we aim to bridge this gap. In particular, we present an Arab-region-specific safety evaluation dataset consisting of 5,799 questions, including direct attacks, indirect attacks, and harmless requests with sensitive words, adapted to reflect the socio-cultural context of the Arab world. To uncover the impact of different stances in handling sensitive and controversial topics, we propose a dual-perspective evaluation framework. It assesses the LLM responses from both governmental and opposition viewpoints. Experiments over five leading Arabic-centric and multilingual LLMs reveal substantial disparities in their safety performance. This reinforces the need for culturally specific datasets to ensure the responsible deployment of LLMs.

摘要: 大型语言模型（LLM）的越来越多的使用引发了人们对其安全性的担忧。虽然许多研究都集中在英语上，但阿拉伯语法学硕士的安全性及其语言和文化复杂性仍然没有得到充分的探讨。在这里，我们的目标是弥合这一差距。特别是，我们提供了一个特定于阿拉伯地区的安全评估数据集，由5，799个问题组成，包括直接攻击、间接攻击和带有敏感词的无害请求，经过调整以反映阿拉伯世界的社会文化背景。为了揭示不同立场对处理敏感和争议话题的影响，我们提出了一个双视角评估框架。它从政府和反对派的角度评估了法学硕士的回应。对五个领先的以阿拉伯语为中心的多语言LLM的实验揭示了它们的安全性能存在巨大差异。这强化了对特定文化数据集的需求，以确保负责任地部署LLM。



## **17. Mask-based Membership Inference Attacks for Retrieval-Augmented Generation**

用于检索增强生成的基于面具的成员推断攻击 cs.CR

This paper is accepted by conference WWW 2025

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2410.20142v2) [paper-pdf](http://arxiv.org/pdf/2410.20142v2)

**Authors**: Mingrui Liu, Sixiao Zhang, Cheng Long

**Abstract**: Retrieval-Augmented Generation (RAG) has been an effective approach to mitigate hallucinations in large language models (LLMs) by incorporating up-to-date and domain-specific knowledge. Recently, there has been a trend of storing up-to-date or copyrighted data in RAG knowledge databases instead of using it for LLM training. This practice has raised concerns about Membership Inference Attacks (MIAs), which aim to detect if a specific target document is stored in the RAG system's knowledge database so as to protect the rights of data producers. While research has focused on enhancing the trustworthiness of RAG systems, existing MIAs for RAG systems remain largely insufficient. Previous work either relies solely on the RAG system's judgment or is easily influenced by other documents or the LLM's internal knowledge, which is unreliable and lacks explainability. To address these limitations, we propose a Mask-Based Membership Inference Attacks (MBA) framework. Our framework first employs a masking algorithm that effectively masks a certain number of words in the target document. The masked text is then used to prompt the RAG system, and the RAG system is required to predict the mask values. If the target document appears in the knowledge database, the masked text will retrieve the complete target document as context, allowing for accurate mask prediction. Finally, we adopt a simple yet effective threshold-based method to infer the membership of target document by analyzing the accuracy of mask prediction. Our mask-based approach is more document-specific, making the RAG system's generation less susceptible to distractions from other documents or the LLM's internal knowledge. Extensive experiments demonstrate the effectiveness of our approach compared to existing baseline models.

摘要: 检索-增强生成(RAG)是一种通过结合最新的和特定领域的知识来缓解大型语言模型(LLMS)中的幻觉的有效方法。最近，有一种趋势是将最新数据或受版权保护的数据存储在RAG知识数据库中，而不是将其用于LLM培训。这种做法引起了人们对成员资格推断攻击(MIA)的担忧，这种攻击旨在检测特定目标文件是否存储在RAG系统的知识数据库中，以保护数据制作者的权利。虽然研究的重点是提高RAG系统的可信度，但现有的RAG系统的MIA仍然很大程度上是不够的。以往的工作要么完全依赖RAG系统的判断，要么容易受到其他文件或LLM内部知识的影响，这是不可靠的，缺乏解释性。针对这些局限性，我们提出了一种基于掩码的成员关系推理攻击(MBA)框架。我们的框架首先使用掩码算法，该算法有效地掩码目标文档中的特定数量的单词。然后使用掩码文本来提示RAG系统，并且需要RAG系统来预测掩码值。如果目标文档出现在知识数据库中，则掩码文本将检索完整的目标文档作为上下文，从而实现准确的掩码预测。最后，通过分析模板预测的精度，采用一种简单有效的基于阈值的方法来推断目标文档的隶属度。我们的基于掩码的方法更特定于文档，使RAG系统的生成不太容易受到来自其他文档或LLM内部知识的干扰。大量的实验表明，与现有的基线模型相比，我们的方法是有效的。



## **18. Effective Black-Box Multi-Faceted Attacks Breach Vision Large Language Model Guardrails**

有效的黑匣子多面攻击破坏视觉大型语言模型护栏 cs.CV

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2502.05772v1) [paper-pdf](http://arxiv.org/pdf/2502.05772v1)

**Authors**: Yijun Yang, Lichao Wang, Xiao Yang, Lanqing Hong, Jun Zhu

**Abstract**: Vision Large Language Models (VLLMs) integrate visual data processing, expanding their real-world applications, but also increasing the risk of generating unsafe responses. In response, leading companies have implemented Multi-Layered safety defenses, including alignment training, safety system prompts, and content moderation. However, their effectiveness against sophisticated adversarial attacks remains largely unexplored. In this paper, we propose MultiFaceted Attack, a novel attack framework designed to systematically bypass Multi-Layered Defenses in VLLMs. It comprises three complementary attack facets: Visual Attack that exploits the multimodal nature of VLLMs to inject toxic system prompts through images; Alignment Breaking Attack that manipulates the model's alignment mechanism to prioritize the generation of contrasting responses; and Adversarial Signature that deceives content moderators by strategically placing misleading information at the end of the response. Extensive evaluations on eight commercial VLLMs in a black-box setting demonstrate that MultiFaceted Attack achieves a 61.56% attack success rate, surpassing state-of-the-art methods by at least 42.18%.

摘要: 视觉大语言模型(VLLM)集成了可视化数据处理，扩展了它们在现实世界中的应用，但也增加了生成不安全响应的风险。作为回应，领先的公司实施了多层次的安全防御措施，包括对齐培训、安全系统提示和内容审核。然而，它们对抗复杂的对抗性攻击的有效性在很大程度上仍未得到探索。在本文中，我们提出了一种新的攻击框架--多方面攻击，旨在系统地绕过VLLMS中的多层防御。它包括三个互补的攻击方面：利用VLLM的多模式特性通过图像注入有毒系统提示的视觉攻击；操纵模型的对齐机制以优先生成对比响应的对齐破坏攻击；以及通过在响应的末尾战略性地放置误导性信息来欺骗内容审核者的对抗性签名。在黑匣子环境下对8个商用VLLM进行了广泛的评估，结果表明，多面攻击的攻击成功率达到了61.56%，至少比最先进的攻击方法高出42.18%。



## **19. Dynamic Guided and Domain Applicable Safeguards for Enhanced Security in Large Language Models**

大型语言模型中增强安全性的动态引导和领域适用保障措施 cs.AI

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2410.17922v2) [paper-pdf](http://arxiv.org/pdf/2410.17922v2)

**Authors**: Weidi Luo, He Cao, Zijing Liu, Yu Wang, Aidan Wong, Bing Feng, Yuan Yao, Yu Li

**Abstract**: With the extensive deployment of Large Language Models (LLMs), ensuring their safety has become increasingly critical. However, existing defense methods often struggle with two key issues: (i) inadequate defense capabilities, particularly in domain-specific scenarios like chemistry, where a lack of specialized knowledge can lead to the generation of harmful responses to malicious queries. (ii) over-defensiveness, which compromises the general utility and responsiveness of LLMs. To mitigate these issues, we introduce a multi-agents-based defense framework, Guide for Defense (G4D), which leverages accurate external information to provide an unbiased summary of user intentions and analytically grounded safety response guidance. Extensive experiments on popular jailbreak attacks and benign datasets show that our G4D can enhance LLM's robustness against jailbreak attacks on general and domain-specific scenarios without compromising the model's general functionality.

摘要: 随着大型语言模型（LLM）的广泛部署，确保其安全性变得越来越重要。然而，现有的防御方法经常遇到两个关键问题：（i）防御能力不足，特别是在化学等特定领域的场景中，缺乏专业知识可能会导致对恶意查询产生有害响应。(ii)过度防御，这会损害LLM的一般实用性和响应能力。为了缓解这些问题，我们引入了一个基于多代理的防御框架--防御指南（G4 D），该框架利用准确的外部信息来提供用户意图的公正摘要和基于分析的安全响应指南。对流行越狱攻击和良性数据集的广泛实验表明，我们的G4 D可以增强LLM针对一般和特定领域场景的越狱攻击的鲁棒性，而不会损害模型的一般功能。



## **20. "Yes, My LoRD." Guiding Language Model Extraction with Locality Reinforced Distillation**

“是的，我的爱人。“利用局部强化蒸馏提取引导语言模型 cs.CR

**SubmitDate**: 2025-02-08    [abs](http://arxiv.org/abs/2409.02718v2) [paper-pdf](http://arxiv.org/pdf/2409.02718v2)

**Authors**: Zi Liang, Qingqing Ye, Yanyun Wang, Sen Zhang, Yaxin Xiao, Ronghua Li, Jianliang Xu, Haibo Hu

**Abstract**: Model extraction attacks (MEAs) on large language models (LLMs) have received increasing attention in recent research. However, existing attack methods typically adapt the extraction strategies originally developed for deep neural networks (DNNs). They neglect the underlying inconsistency between the training tasks of MEA and LLM alignment, leading to suboptimal attack performance. To tackle this issue, we propose Locality Reinforced Distillation (LoRD), a novel model extraction algorithm specifically designed for LLMs. In particular, LoRD employs a newly defined policy-gradient-style training task that utilizes the responses of victim model as the signal to guide the crafting of preference for the local model. Theoretical analyses demonstrate that I) The convergence procedure of LoRD in model extraction is consistent with the alignment procedure of LLMs, and II) LoRD can reduce query complexity while mitigating watermark protection through our exploration-based stealing. Extensive experiments validate the superiority of our method in extracting various state-of-the-art commercial LLMs. Our code is available at: https://github.com/liangzid/LoRD-MEA.

摘要: 近年来，针对大型语言模型的模型提取攻击受到越来越多的关注。然而，现有的攻击方法通常采用最初为深度神经网络(DNN)开发的提取策略。它们忽略了MEA和LLM对齐训练任务之间的潜在不一致性，导致了次优的攻击性能。为了解决这一问题，我们提出了一种新的模型提取算法LOAD，它是专门为LLM设计的。特别是，Lord采用了一种新定义的政策梯度式训练任务，该任务利用受害者模型的反应作为信号来指导对本地模型的偏好的形成。理论分析表明：1)Lord算法在模型提取中的收敛过程与LLMS算法的对齐过程是一致的；2)Lord算法在降低查询复杂度的同时，通过基于探索的窃取来减轻水印保护。大量的实验验证了该方法在提取各种最先进的商业LLM方面的优越性。我们的代码请访问：https://github.com/liangzid/LoRD-MEA.



## **21. HumorReject: Decoupling LLM Safety from Refusal Prefix via A Little Humor**

幽默：通过一点幽默将LLM安全与拒绝前置脱钩 cs.LG

**SubmitDate**: 2025-02-08    [abs](http://arxiv.org/abs/2501.13677v2) [paper-pdf](http://arxiv.org/pdf/2501.13677v2)

**Authors**: Zihui Wu, Haichang Gao, Jiacheng Luo, Zhaoxiang Liu

**Abstract**: Large Language Models (LLMs) commonly rely on explicit refusal prefixes for safety, making them vulnerable to prefix injection attacks. We introduce HumorReject, a novel data-driven approach that reimagines LLM safety by decoupling it from refusal prefixes through humor as an indirect refusal strategy. Rather than explicitly rejecting harmful instructions, HumorReject responds with contextually appropriate humor that naturally defuses potentially dangerous requests. Our approach effectively addresses common "over-defense" issues while demonstrating superior robustness against various attack vectors. Our findings suggest that improvements in training data design can be as important as the alignment algorithm itself in achieving effective LLM safety.

摘要: 大型语言模型（LLM）通常依赖显式拒绝前置来确保安全，这使得它们容易受到前置输入攻击。我们引入了幽默感，这是一种新颖的数据驱动方法，通过幽默将其与拒绝开头脱钩，重新构想了LLM的安全性，将其作为一种间接拒绝策略。幽默感并没有明确拒绝有害的指令，而是以符合上下文的幽默来回应，从而自然地化解潜在危险的请求。我们的方法有效地解决了常见的“过度防御”问题，同时展示了针对各种攻击载体的卓越鲁棒性。我们的研究结果表明，在实现有效的LLM安全性方面，训练数据设计的改进与对齐算法本身一样重要。



## **22. Topic-Based Watermarks for Large Language Models**

大型语言模型的基于主题的水印 cs.CR

Algorithms and new evaluations, 8 pages

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2404.02138v4) [paper-pdf](http://arxiv.org/pdf/2404.02138v4)

**Authors**: Alexander Nemecek, Yuzhou Jiang, Erman Ayday

**Abstract**: The indistinguishability of Large Language Model (LLM) output from human-authored content poses significant challenges, raising concerns about potential misuse of AI-generated text and its influence on future AI model training. Watermarking algorithms offer a viable solution by embedding detectable signatures into generated text. However, existing watermarking methods often entail trade-offs among attack robustness, generation quality, and additional overhead such as specialized frameworks or complex integrations. We propose a lightweight, topic-guided watermarking scheme for LLMs that partitions the vocabulary into topic-aligned token subsets. Given an input prompt, the scheme selects a relevant topic-specific token list, effectively "green-listing" semantically aligned tokens to embed robust marks while preserving the text's fluency and coherence. Experimental results across multiple LLMs and state-of-the-art benchmarks demonstrate that our method achieves comparable perplexity to industry-leading systems, including Google's SynthID-Text, yet enhances watermark robustness against paraphrasing and lexical perturbation attacks while introducing minimal performance overhead. Our approach avoids reliance on additional mechanisms beyond standard text generation pipelines, facilitating straightforward adoption, suggesting a practical path toward globally consistent watermarking of AI-generated content.

摘要: 大型语言模型(LLM)输出与人类创作的内容的不可区分构成了重大挑战，这引发了人们对人工智能生成文本的潜在滥用及其对未来人工智能模型训练的影响的担忧。水印算法通过在生成的文本中嵌入可检测的签名来提供一种可行的解决方案。然而，现有的水印方法往往需要在攻击健壮性、生成质量和额外的开销之间进行权衡，例如专门的框架或复杂的集成。我们提出了一种轻量级的、主题引导的LLMS水印方案，该方案将词汇表划分为主题对齐的令牌子集。在给定输入提示的情况下，该方案选择相关的特定于主题的标记列表，有效地对齐语义对齐的标记以嵌入健壮的标记，同时保持文本的流畅性和连贯性。在多个LLMS和最先进的基准测试上的实验结果表明，我们的方法获得了与业界领先的系统(包括Google的SynthID-Text)相当的困惑，但增强了对释义和词汇扰动攻击的水印鲁棒性，同时引入的性能开销最小。我们的方法避免了对标准文本生成管道之外的额外机制的依赖，促进了直接采用，建议了一条实现人工智能生成内容的全球一致水印的实用路径。



## **23. Watermarking Low-entropy Generation for Large Language Models: An Unbiased and Low-risk Method**

大型语言模型的水印低熵生成：一种无偏见且低风险的方法 cs.CL

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2405.14604v3) [paper-pdf](http://arxiv.org/pdf/2405.14604v3)

**Authors**: Minjia Mao, Dongjun Wei, Zeyu Chen, Xiao Fang, Michael Chau

**Abstract**: Recent advancements in large language models (LLMs) have highlighted the risk of misusing them, raising the need for accurate detection of LLM-generated content. In response, a viable solution is to inject imperceptible identifiers into LLMs, known as watermarks. Our research extends the existing watermarking methods by proposing the novel Sampling One Then Accepting (STA-1) method. STA-1 is an unbiased watermark that preserves the original token distribution in expectation and has a lower risk of producing unsatisfactory outputs in low-entropy scenarios compared to existing unbiased watermarks. In watermark detection, STA-1 does not require prompts or a white-box LLM, provides statistical guarantees, demonstrates high efficiency in detection time, and remains robust against various watermarking attacks. Experimental results on low-entropy and high-entropy datasets demonstrate that STA-1 achieves the above properties simultaneously, making it a desirable solution for watermarking LLMs. Implementation codes for this study are available online.

摘要: 大型语言模型(LLM)最近的进步突显了滥用它们的风险，提高了对LLM生成的内容进行准确检测的必要性。对此，一个可行的解决方案是将不可察觉的标识符注入LLM，即所谓的水印。我们的研究扩展了现有的数字水印方法，提出了一种新的采样后接受(STA-1)方法。STA-1是一种无偏水印，它在期望中保留了原始的令牌分布，并且与现有的无偏水印相比，在低熵情况下产生不满意输出的风险更低。在水印检测中，STA-1不需要提示或白盒LLM，提供统计保证，在检测时间上表现出高效率，并对各种水印攻击保持稳健。在低熵和高熵数据集上的实验结果表明，STA-1同时达到了上述特性，是一种理想的LLMS水印方案。这项研究的实施代码可在网上查阅。



## **24. Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning**

简单性盛行：重新思考LLM忘记学习的负偏好优化 cs.CL

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2410.07163v3) [paper-pdf](http://arxiv.org/pdf/2410.07163v3)

**Authors**: Chongyu Fan, Jiancheng Liu, Licong Lin, Jinghan Jia, Ruiqi Zhang, Song Mei, Sijia Liu

**Abstract**: This work studies the problem of large language model (LLM) unlearning, aiming to remove unwanted data influences (e.g., copyrighted or harmful content) while preserving model utility. Despite the increasing demand for unlearning, a technically-grounded optimization framework is lacking. Gradient ascent (GA)-type methods, though widely used, are suboptimal as they reverse the learning process without controlling optimization divergence (i.e., deviation from the pre-trained state), leading to risks of over-forgetting and potential model collapse. Negative preference optimization (NPO) has been proposed to address this issue and is considered one of the state-of-the-art LLM unlearning approaches. In this work, we revisit NPO and identify another critical issue: reference model bias. This bias arises from using the reference model (i.e., the model prior to unlearning) to evaluate the unlearning success, which can compromise NPO's effectiveness. Specifically, it leads to (a) uneven allocation of optimization power across forget data with varying difficulty levels and (b) ineffective gradient weight smoothing during the early stages of unlearning optimization. To overcome these challenges, we propose a simple yet effective unlearning optimization framework, called SimNPO, showing that `simplicity' in removing the reliance on a reference model (through the lens of simple preference optimization) benefits unlearning. We provide deeper insights into SimNPO's advantages through an analysis based on mixtures of Markov chains. Extensive experiments further validate SimNPO's efficacy on benchmarks like TOFU and MUSE, as well as its robustness against relearning attacks. Codes are available at https://github.com/OPTML-Group/Unlearn-Simple.

摘要: 这项工作研究了大型语言模型(LLM)的遗忘问题，目的是在保留模型效用的同时消除不需要的数据影响(例如，版权或有害内容)。尽管对遗忘的需求越来越大，但缺乏以技术为基础的优化框架。梯度上升(GA)类方法虽然被广泛使用，但由于它们逆转了学习过程而没有控制优化发散(即偏离预先训练的状态)，导致过度遗忘和潜在的模型崩溃的风险，因此是次优的。负偏好优化(NPO)就是为了解决这一问题而提出的，被认为是最先进的LLM遗忘方法之一。在这项工作中，我们重新审视了非营利组织，并确定了另一个关键问题：参考模型偏差。这种偏差源于使用参考模型(即遗忘前的模型)来评估遗忘成功，这可能会影响非营利组织的有效性。具体地说，它导致(A)在具有不同难度级别的遗忘数据之间的优化功率分配不均匀，以及(B)在遗忘优化的早期阶段无效的梯度权重平滑。为了克服这些挑战，我们提出了一个简单但有效的遗忘优化框架，称为SimNPO，表明在消除对参考模型的依赖(通过简单偏好优化的镜头)时的“简单性”有利于遗忘。我们通过基于马尔科夫链混合的分析，对SimNPO的优势提供了更深入的见解。大量的实验进一步验证了SimNPO在豆腐和缪斯等基准测试上的有效性，以及它对重新学习攻击的健壮性。有关代码，请访问https://github.com/OPTML-Group/Unlearn-Simple.



## **25. Do Unlearning Methods Remove Information from Language Model Weights?**

取消学习方法会从语言模型权重中删除信息吗？ cs.LG

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2410.08827v3) [paper-pdf](http://arxiv.org/pdf/2410.08827v3)

**Authors**: Aghyad Deeb, Fabien Roger

**Abstract**: Large Language Models' knowledge of how to perform cyber-security attacks, create bioweapons, and manipulate humans poses risks of misuse. Previous work has proposed methods to unlearn this knowledge. Historically, it has been unclear whether unlearning techniques are removing information from the model weights or just making it harder to access. To disentangle these two objectives, we propose an adversarial evaluation method to test for the removal of information from model weights: we give an attacker access to some facts that were supposed to be removed, and using those, the attacker tries to recover other facts from the same distribution that cannot be guessed from the accessible facts. We show that using fine-tuning on the accessible facts can recover 88% of the pre-unlearning accuracy when applied to current unlearning methods for information learned during pretraining, revealing the limitations of these methods in removing information from the model weights. Our results also suggest that unlearning evaluations that measure unlearning robustness on information learned during an additional fine-tuning phase may overestimate robustness compared to evaluations that attempt to unlearn information learned during pretraining.

摘要: 大型语言模型在如何执行网络安全攻击、创建生物武器和操纵人类方面的知识构成了误用的风险。以前的工作已经提出了忘记这一知识的方法。从历史上看，人们一直不清楚遗忘技术是在移除模型重量中的信息，还是只是增加了获取信息的难度。为了分离这两个目标，我们提出了一种对抗性评估方法来测试从模型权重中移除信息的情况：我们允许攻击者访问一些应该被移除的事实，并且使用这些事实，攻击者试图从相同的分布中恢复无法从可访问的事实中猜测的其他事实。结果表明，对可访问的事实进行微调可以恢复88%的预忘学习准确率，当应用于现有的遗忘方法时，这些方法在去除模型权重中的信息方面存在局限性。我们的结果还表明，与试图忘却在预训练中学习的信息的评估相比，衡量在额外微调阶段学习到的信息的遗忘健壮性的遗忘评估可能高估了健壮性。



## **26. GenBFA: An Evolutionary Optimization Approach to Bit-Flip Attacks on LLMs**

GenBFA：对LLM进行位翻转攻击的进化优化方法 cs.CR

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2411.13757v2) [paper-pdf](http://arxiv.org/pdf/2411.13757v2)

**Authors**: Sanjay Das, Swastik Bhattacharya, Souvik Kundu, Shamik Kundu, Anand Menon, Arnab Raha, Kanad Basu

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing (NLP), excelling in tasks like text generation and summarization. However, their increasing adoption in mission-critical applications raises concerns about hardware-based threats, particularly bit-flip attacks (BFAs). BFAs, enabled by fault injection methods such as Rowhammer, target model parameters in memory, compromising both integrity and performance. Identifying critical parameters for BFAs in the vast parameter space of LLMs poses significant challenges. While prior research suggests transformer-based architectures are inherently more robust to BFAs compared to traditional deep neural networks, we challenge this assumption. For the first time, we demonstrate that as few as three bit-flips can cause catastrophic performance degradation in an LLM with billions of parameters. Current BFA techniques are inadequate for exploiting this vulnerability due to the difficulty of efficiently identifying critical parameters within the immense parameter space. To address this, we propose AttentionBreaker, a novel framework tailored for LLMs that enables efficient traversal of the parameter space to identify critical parameters. Additionally, we introduce GenBFA, an evolutionary optimization strategy designed to refine the search further, isolating the most critical bits for an efficient and effective attack. Empirical results reveal the profound vulnerability of LLMs to AttentionBreaker. For example, merely three bit-flips (4.129 x 10^-9% of total parameters) in the LLaMA3-8B-Instruct 8-bit quantized (W8) model result in a complete performance collapse: accuracy on MMLU tasks drops from 67.3% to 0%, and Wikitext perplexity skyrockets from 12.6 to 4.72 x 10^5. These findings underscore the effectiveness of AttentionBreaker in uncovering and exploiting critical vulnerabilities within LLM architectures.

摘要: 大型语言模型(LLM)使自然语言处理(NLP)发生了革命性的变化，在文本生成和摘要等任务中表现出色。然而，它们在任务关键型应用中的日益采用引发了对基于硬件的威胁的担忧，特别是位翻转攻击(BFA)。由Rowhammer等故障注入方法启用的BFA以内存中的模型参数为目标，损害了完整性和性能。在LLMS庞大的参数空间中识别BFA的关键参数带来了巨大的挑战。虽然先前的研究表明，与传统的深度神经网络相比，基于变压器的体系结构对BFA具有更强的鲁棒性，但我们对这一假设提出了质疑。我们首次证明，在具有数十亿个参数的LLM中，仅三个比特翻转就会导致灾难性的性能下降。由于很难在巨大的参数空间中有效地识别关键参数，因此当前的BFA技术不足以利用该漏洞。为了解决这个问题，我们提出了AttentionBreaker，这是一个为LLMS量身定做的新框架，能够有效地遍历参数空间来识别关键参数。此外，我们还引入了GenBFA，这是一种进化优化策略，旨在进一步细化搜索，隔离最关键的比特，以实现高效和有效的攻击。实证结果揭示了LLMS对AttentionBreaker的严重脆弱性。例如，在LLAMA3-8B指令8位量化(W8)模型中，仅三次位翻转(占总参数的4.129 x 10^-9%)就会导致完全的性能崩溃：MMLU任务的准确率从67.3%下降到0%，而Wikitext的复杂性从12.6x10^5飙升到4.72x10^5。这些发现突显了AttentionBreaker在发现和利用LLM体系结构中的关键漏洞方面的有效性。



## **27. From Allies to Adversaries: Manipulating LLM Tool-Calling through Adversarial Injection**

从盟友到对手：通过对抗注入操纵LLM工具调用 cs.CR

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2412.10198v2) [paper-pdf](http://arxiv.org/pdf/2412.10198v2)

**Authors**: Haowei Wang, Rupeng Zhang, Junjie Wang, Mingyang Li, Yuekai Huang, Dandan Wang, Qing Wang

**Abstract**: Tool-calling has changed Large Language Model (LLM) applications by integrating external tools, significantly enhancing their functionality across diverse tasks. However, this integration also introduces new security vulnerabilities, particularly in the tool scheduling mechanisms of LLM, which have not been extensively studied. To fill this gap, we present ToolCommander, a novel framework designed to exploit vulnerabilities in LLM tool-calling systems through adversarial tool injection. Our framework employs a well-designed two-stage attack strategy. Firstly, it injects malicious tools to collect user queries, then dynamically updates the injected tools based on the stolen information to enhance subsequent attacks. These stages enable ToolCommander to execute privacy theft, launch denial-of-service attacks, and even manipulate business competition by triggering unscheduled tool-calling. Notably, the ASR reaches 91.67% for privacy theft and hits 100% for denial-of-service and unscheduled tool calling in certain cases. Our work demonstrates that these vulnerabilities can lead to severe consequences beyond simple misuse of tool-calling systems, underscoring the urgent need for robust defensive strategies to secure LLM Tool-calling systems.

摘要: 工具调用通过集成外部工具改变了大型语言模型(LLM)应用程序，显著增强了它们在不同任务中的功能。然而，这种集成也引入了新的安全漏洞，特别是在LLM的工具调度机制中，这些漏洞还没有得到广泛的研究。为了填补这一空白，我们提出了一种新的框架，它旨在通过敌意工具注入来利用LLM工具调用系统中的漏洞。我们的框架采用了精心设计的两阶段攻击策略。它首先注入恶意工具来收集用户查询，然后根据窃取的信息动态更新注入的工具，以加强后续攻击。这些阶段使工具指挥官能够执行隐私窃取、发起拒绝服务攻击，甚至通过触发计划外的工具调用来操纵业务竞争。值得注意的是，在某些情况下，隐私窃取的ASR达到91.67%，拒绝服务和非计划工具调用的ASR达到100%。我们的工作表明，这些漏洞可能导致严重后果，而不仅仅是简单地滥用工具调用系统，这突显了迫切需要强大的防御战略来保护LLM工具调用系统。



## **28. Enhancing Phishing Email Identification with Large Language Models**

使用大型语言模型增强网络钓鱼电子邮件识别 cs.CR

9 pages, 5 figures

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.04759v1) [paper-pdf](http://arxiv.org/pdf/2502.04759v1)

**Authors**: Catherine Lee

**Abstract**: Phishing has long been a common tactic used by cybercriminals and continues to pose a significant threat in today's digital world. When phishing attacks become more advanced and sophisticated, there is an increasing need for effective methods to detect and prevent them. To address the challenging problem of detecting phishing emails, researchers have developed numerous solutions, in particular those based on machine learning (ML) algorithms. In this work, we take steps to study the efficacy of large language models (LLMs) in detecting phishing emails. The experiments show that the LLM achieves a high accuracy rate at high precision; importantly, it also provides interpretable evidence for the decisions.

摘要: 网络钓鱼长期以来一直是网络犯罪分子使用的常见策略，并继续在当今的数字世界构成重大威胁。当网络钓鱼攻击变得更加先进和复杂时，越来越需要有效的方法来检测和预防它们。为了解决检测网络钓鱼电子邮件的挑战性问题，研究人员开发了多种解决方案，特别是基于机器学习（ML）算法的解决方案。在这项工作中，我们采取措施研究大型语言模型（LLM）在检测网络钓鱼电子邮件方面的功效。实验表明，LLM在高精度下实现了高准确率;重要的是，它还为决策提供了可解释的证据。



## **29. Jailbreak Antidote: Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Language Models**

越狱解药：通过大型语言模型中的稀疏表示调整来实现安全与效用平衡 cs.CR

Accepted by ICLR2025. url: https://openreview.net/forum?id=s20W12XTF8

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2410.02298v4) [paper-pdf](http://arxiv.org/pdf/2410.02298v4)

**Authors**: Guobin Shen, Dongcheng Zhao, Yiting Dong, Xiang He, Yi Zeng

**Abstract**: As large language models (LLMs) become integral to various applications, ensuring both their safety and utility is paramount. Jailbreak attacks, which manipulate LLMs into generating harmful content, pose significant challenges to this balance. Existing defenses, such as prompt engineering and safety fine-tuning, often introduce computational overhead, increase inference latency, and lack runtime flexibility. Moreover, overly restrictive safety measures can degrade model utility by causing refusals of benign queries. In this paper, we introduce Jailbreak Antidote, a method that enables real-time adjustment of LLM safety preferences by manipulating a sparse subset of the model's internal states during inference. By shifting the model's hidden representations along a safety direction with varying strengths, we achieve flexible control over the safety-utility balance without additional token overhead or inference delays. Our analysis reveals that safety-related information in LLMs is sparsely distributed; adjusting approximately 5% of the internal state is as effective as modifying the entire state. Extensive experiments on nine LLMs (ranging from 2 billion to 72 billion parameters), evaluated against ten jailbreak attack methods and compared with six defense strategies, validate the effectiveness and efficiency of our approach. By directly manipulating internal states during reasoning, Jailbreak Antidote offers a lightweight, scalable solution that enhances LLM safety while preserving utility, opening new possibilities for real-time safety mechanisms in widely-deployed AI systems.

摘要: 随着大型语言模型(LLM)成为各种应用程序不可或缺的一部分，确保它们的安全性和实用性是至关重要的。越狱攻击操纵LLM生成有害内容，对这种平衡构成了重大挑战。现有的防御措施，如即时工程和安全微调，通常会引入计算开销，增加推理延迟，并且缺乏运行时灵活性。此外，过于严格的安全措施可能会导致良性查询被拒绝，从而降低模型的实用性。在本文中，我们介绍了JailBreak解毒剂，这是一种通过在推理过程中操纵模型内部状态的稀疏子集来实时调整LLM安全偏好的方法。通过沿不同强度的安全方向移动模型的隐藏表示，我们在不增加令牌开销或推理延迟的情况下实现了对安全-效用平衡的灵活控制。我们的分析表明，LLMS中与安全相关的信息是稀疏分布的；调整大约5%的内部状态与修改整个状态一样有效。在9个LLM(参数从20亿到720亿)上进行了大量的实验，对10种越狱攻击方法进行了评估，并与6种防御策略进行了比较，验证了该方法的有效性和高效性。通过在推理过程中直接操纵内部状态，越狱解毒剂提供了一个轻量级、可扩展的解决方案，在增强LLM安全性的同时保留了实用性，为广泛部署的AI系统中的实时安全机制打开了新的可能性。



## **30. Membership Inference Attacks Against Vision-Language Models**

针对视觉语言模型的成员推断攻击 cs.CR

Accepted by USENIX'25; 22 pages, 28 figures;

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2501.18624v2) [paper-pdf](http://arxiv.org/pdf/2501.18624v2)

**Authors**: Yuke Hu, Zheng Li, Zhihao Liu, Yang Zhang, Zhan Qin, Kui Ren, Chun Chen

**Abstract**: Vision-Language Models (VLMs), built on pre-trained vision encoders and large language models (LLMs), have shown exceptional multi-modal understanding and dialog capabilities, positioning them as catalysts for the next technological revolution. However, while most VLM research focuses on enhancing multi-modal interaction, the risks of data misuse and leakage have been largely unexplored. This prompts the need for a comprehensive investigation of such risks in VLMs. In this paper, we conduct the first analysis of misuse and leakage detection in VLMs through the lens of membership inference attack (MIA). In specific, we focus on the instruction tuning data of VLMs, which is more likely to contain sensitive or unauthorized information. To address the limitation of existing MIA methods, we introduce a novel approach that infers membership based on a set of samples and their sensitivity to temperature, a unique parameter in VLMs. Based on this, we propose four membership inference methods, each tailored to different levels of background knowledge, ultimately arriving at the most challenging scenario. Our comprehensive evaluations show that these methods can accurately determine membership status, e.g., achieving an AUC greater than 0.8 targeting a small set consisting of only 5 samples on LLaVA.

摘要: 建立在预先训练的视觉编码器和大型语言模型(LLM)上的视觉语言模型(VLM)显示了出众的多模式理解和对话能力，将它们定位为下一次技术革命的催化剂。然而，尽管大多数VLM研究都集中在加强多模式交互上，但数据滥用和泄露的风险在很大程度上还没有被探索。这促使需要对VLM中的此类风险进行全面调查。本文首次从成员关系推理攻击(MIA)的角度对VLMS中的误用和漏洞检测进行了分析。具体地说，我们关注的是VLM的指令调优数据，这些数据更有可能包含敏感或未经授权的信息。为了解决现有MIA方法的局限性，我们引入了一种新的方法，该方法基于一组样本及其对温度的敏感度来推断隶属度，这是VLMS中的一个独特参数。基于此，我们提出了四种隶属度推理方法，每种方法都针对不同的背景知识水平进行了定制，最终得出了最具挑战性的场景。我们的综合评估表明，这些方法可以准确地确定成员状态，例如，针对LLaVA上只有5个样本的小集合，AUC大于0.8。



## **31. Hide Your Malicious Goal Into Benign Narratives: Jailbreak Large Language Models through Carrier Articles**

将你的恶意目标隐藏在良性叙述中：通过载体文章越狱大型语言模型 cs.CR

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2408.11182v2) [paper-pdf](http://arxiv.org/pdf/2408.11182v2)

**Authors**: Zhilong Wang, Haizhou Wang, Nanqing Luo, Lan Zhang, Xiaoyan Sun, Yebo Cao, Peng Liu

**Abstract**: Large Language Model (LLM) jailbreak refers to a type of attack aimed to bypass the safeguard of an LLM to generate contents that are inconsistent with the safe usage guidelines. Based on the insights from the self-attention computation process, this paper proposes a novel blackbox jailbreak approach, which involves crafting the payload prompt by strategically injecting the prohibited query into a carrier article. The carrier article maintains the semantic proximity to the prohibited query, which is automatically produced by combining a hypernymy article and a context, both of which are generated from the prohibited query. The intuition behind the usage of carrier article is to activate the neurons in the model related to the semantics of the prohibited query while suppressing the neurons that will trigger the objectionable text. Carrier article itself is benign, and we leveraged prompt injection techniques to produce the payload prompt. We evaluate our approach using JailbreakBench, testing against four target models across 100 distinct jailbreak objectives. The experimental results demonstrate our method's superior effectiveness, achieving an average success rate of 63% across all target models, significantly outperforming existing blackbox jailbreak methods.

摘要: 大型语言模型(LLM)越狱是指一种旨在绕过LLM的安全保护以生成不符合安全使用指南的内容的攻击类型。基于自我注意计算过程的洞察力，提出了一种新的黑盒越狱方法，该方法通过在载体文章中策略性地插入禁止查询来创建有效负载提示。载体物品保持与禁止查询的语义接近，这是通过组合从禁止查询生成的超级物品和上下文而自动产生的。载体冠词使用背后的直觉是激活模型中与禁止查询的语义相关的神经元，同时抑制将触发不良文本的神经元。载体物品本身是良性的，我们利用即时注入技术来产生有效载荷提示。我们使用JailBreak Bch评估我们的方法，针对100个不同的越狱目标对四个目标模型进行测试。实验结果表明，该方法具有较高的效率，在所有目标模型上的平均成功率为63%，显著优于现有的黑盒越狱方法。



## **32. Speak Easy: Eliciting Harmful Jailbreaks from LLMs with Simple Interactions**

轻松说话：通过简单的互动引发法学硕士的有害越狱 cs.LG

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2502.04322v1) [paper-pdf](http://arxiv.org/pdf/2502.04322v1)

**Authors**: Yik Siu Chan, Narutatsu Ri, Yuxin Xiao, Marzyeh Ghassemi

**Abstract**: Despite extensive safety alignment efforts, large language models (LLMs) remain vulnerable to jailbreak attacks that elicit harmful behavior. While existing studies predominantly focus on attack methods that require technical expertise, two critical questions remain underexplored: (1) Are jailbroken responses truly useful in enabling average users to carry out harmful actions? (2) Do safety vulnerabilities exist in more common, simple human-LLM interactions? In this paper, we demonstrate that LLM responses most effectively facilitate harmful actions when they are both actionable and informative--two attributes easily elicited in multi-step, multilingual interactions. Using this insight, we propose HarmScore, a jailbreak metric that measures how effectively an LLM response enables harmful actions, and Speak Easy, a simple multi-step, multilingual attack framework. Notably, by incorporating Speak Easy into direct request and jailbreak baselines, we see an average absolute increase of 0.319 in Attack Success Rate and 0.426 in HarmScore in both open-source and proprietary LLMs across four safety benchmarks. Our work reveals a critical yet often overlooked vulnerability: Malicious users can easily exploit common interaction patterns for harmful intentions.

摘要: 尽管进行了广泛的安全调整工作，但大型语言模型(LLM)仍然容易受到引发有害行为的越狱攻击。虽然现有的研究主要集中在需要技术专业知识的攻击方法上，但仍有两个关键问题未被探索：(1)越狱反应是否真的有助于使普通用户执行有害行为？(2)更常见、更简单的人与LLM交互中是否存在安全漏洞？在这篇文章中，我们证明了当LLM响应既可操作又可提供信息时，它们最有效地促进了有害行为--这两个属性在多步骤、多语言交互中很容易引发。利用这一见解，我们提出了HarmScore，这是一种衡量LLM响应支持有害操作的效率的指标，并提出了一种简单的多步骤、多语言攻击框架。值得注意的是，通过将Stop Easy整合到直接请求和越狱基准中，我们看到在四个安全基准中，开源和专有LLM的攻击成功率平均绝对增加了0.319，HarmScore增加了0.426。我们的工作揭示了一个关键但经常被忽视的漏洞：恶意用户可以很容易地利用常见的交互模式来实现有害意图。



## **33. Can LLMs Hack Enterprise Networks? Autonomous Assumed Breach Penetration-Testing Active Directory Networks**

LLM可以黑客攻击企业网络吗？自主假设漏洞渗透测试Active目录网络 cs.CR

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2502.04227v1) [paper-pdf](http://arxiv.org/pdf/2502.04227v1)

**Authors**: Andreas Happe, Jürgen Cito

**Abstract**: We explore the feasibility and effectiveness of using LLM-driven autonomous systems for Assumed Breach penetration testing in enterprise networks. We introduce a novel prototype that, driven by Large Language Models (LLMs), can compromise accounts within a real-life Active Directory testbed. Our research provides a comprehensive evaluation of the prototype's capabilities, and highlights both strengths and limitations while executing attack. The evaluation uses a realistic simulation environment (Game of Active Directory, GOAD) to capture intricate interactions, stochastic outcomes, and timing dependencies that characterize live network scenarios. The study concludes that autonomous LLMs are able to conduct Assumed Breach simulations, potentially democratizing access to penetration testing for organizations facing budgetary constraints.   The prototype's source code, traces, and analyzed logs are released as open-source to enhance collective cybersecurity and facilitate future research in LLM-driven cybersecurity automation.

摘要: 我们探索了使用LLM驱动的自治系统在企业网络中进行假设漏洞渗透测试的可行性和有效性。我们介绍了一个新的原型，它由大型语言模型(LLM)驱动，可以在现实生活中的Active Directory试验床中危害帐户。我们的研究提供了对原型能力的全面评估，并强调了执行攻击时的优势和局限性。该评估使用真实的模拟环境(活动目录游戏，GOAD)来捕获复杂的交互、随机结果和时间依赖关系，这些都是实时网络场景的特征。研究得出结论，自主的LLM能够进行假设的漏洞模拟，可能会使面临预算限制的组织获得渗透测试的机会大众化。原型的源代码、跟踪和分析日志以开源形式发布，以增强集体网络安全，并促进未来对LLM驱动的网络安全自动化的研究。



## **34. "Short-length" Adversarial Training Helps LLMs Defend "Long-length" Jailbreak Attacks: Theoretical and Empirical Evidence**

“短期”对抗性培训帮助法学硕士防御“长期”越狱攻击：理论和经验证据 cs.LG

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2502.04204v1) [paper-pdf](http://arxiv.org/pdf/2502.04204v1)

**Authors**: Shaopeng Fu, Liang Ding, Di Wang

**Abstract**: Jailbreak attacks against large language models (LLMs) aim to induce harmful behaviors in LLMs through carefully crafted adversarial prompts. To mitigate attacks, one way is to perform adversarial training (AT)-based alignment, i.e., training LLMs on some of the most adversarial prompts to help them learn how to behave safely under attacks. During AT, the length of adversarial prompts plays a critical role in the robustness of aligned LLMs. This paper focuses on adversarial suffix jailbreak attacks and unveils that to defend against a jailbreak attack with an adversarial suffix of length $\Theta(M)$, it is enough to align LLMs on prompts with adversarial suffixes of length $\Theta(\sqrt{M})$. Theoretically, we analyze the adversarial in-context learning of linear transformers on linear regression tasks and prove a robust generalization bound for trained transformers. The bound depends on the term $\Theta(\sqrt{M_{\text{test}}}/M_{\text{train}})$, where $M_{\text{train}}$ and $M_{\text{test}}$ are the number of adversarially perturbed in-context samples during training and testing. Empirically, we conduct AT on popular open-source LLMs and evaluate their robustness against jailbreak attacks of different adversarial suffix lengths. Results confirm a positive correlation between the attack success rate and the ratio of the square root of the adversarial suffix during jailbreaking to the length during AT. Our findings show that it is practical to defend "long-length" jailbreak attacks via efficient "short-length" AT. The code is available at https://github.com/fshp971/adv-icl.

摘要: 针对大型语言模型(LLM)的越狱攻击旨在通过精心设计的对抗性提示在LLM中诱导有害行为。为了减轻攻击，一种方法是执行基于对抗性训练(AT)的对齐，即根据一些最具对抗性的提示对LLM进行培训，以帮助它们学习如何在攻击下安全地行为。在自动对准过程中，对抗性提示的长度对对准LLMS的稳健性起着至关重要的作用。本文主要研究对抗性后缀越狱攻击，揭示了要防御对抗性后缀长度为$\theta(M)$的越狱攻击，只需使提示上的LLMS与长度为$\theta(\Sqrt{M})$的对抗性后缀对齐即可。在理论上，我们分析了线性回归任务中线性变压器的对抗性上下文学习，并证明了训练的变压器的一个稳健的泛化上界。这个界限取决于术语$\Theta(\sqrt{M_{\text{test}}}/M_{\text{train}})$，，其中$M_{\TEXT{TEST}}$和$M_{\TEXT{TEST}}$是训练和测试过程中受到不利干扰的上下文样本的数量。经验性地，我们对流行的开源LLM进行了AT，并评估了它们对不同敌意后缀长度的越狱攻击的健壮性。结果证实，攻击成功率与越狱时敌意后缀的平方根与AT中敌意后缀的长度之比呈正相关。我们的研究结果表明，通过有效的“短长度”AT防御“长长度”越狱攻击是可行的。代码可在https://github.com/fshp971/adv-icl.上获得



## **35. Assessing and Prioritizing Ransomware Risk Based on Historical Victim Data**

根据历史受害者数据评估勒索软件风险并优先排序 cs.CR

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2502.04421v1) [paper-pdf](http://arxiv.org/pdf/2502.04421v1)

**Authors**: Spencer Massengale, Philip Huff

**Abstract**: We present an approach to identifying which ransomware adversaries are most likely to target specific entities, thereby assisting these entities in formulating better protection strategies. Ransomware poses a formidable cybersecurity threat characterized by profit-driven motives, a complex underlying economy supporting criminal syndicates, and the overt nature of its attacks. This type of malware has consistently ranked among the most prevalent, with a rapid escalation in activity observed. Recent estimates indicate that approximately two-thirds of organizations experienced ransomware attacks in 2023 \cite{Sophos2023Ransomware}. A central tactic in ransomware campaigns is publicizing attacks to coerce victims into paying ransoms. Our study utilizes public disclosures from ransomware victims to predict the likelihood of an entity being targeted by a specific ransomware variant. We employ a Large Language Model (LLM) architecture that uses a unique chain-of-thought, multi-shot prompt methodology to define adversary SKRAM (Skills, Knowledge, Resources, Authorities, and Motivation) profiles from ransomware bulletins, threat reports, and news items. This analysis is enriched with publicly available victim data and is further enhanced by a heuristic for generating synthetic data that reflects victim profiles. Our work culminates in the development of a machine learning model that assists organizations in prioritizing ransomware threats and formulating defenses based on the tactics, techniques, and procedures (TTP) of the most likely attackers.

摘要: 我们提出了一种识别哪些勒索软件攻击者最有可能以特定实体为目标的方法，从而帮助这些实体制定更好的保护策略。勒索软件构成了一个强大的网络安全威胁，其特征是利润驱动的动机、支持犯罪集团的复杂基础经济以及其攻击的公开性质。这种类型的恶意软件一直是最普遍的类型之一，并观察到活动迅速升级。最近的估计表明，大约三分之二的组织在2023年经历了勒索软件攻击。勒索软件活动的一个核心策略是公布攻击，迫使受害者支付赎金。我们的研究利用勒索软件受害者的公开披露来预测实体成为特定勒索软件变体目标的可能性。我们采用大型语言模型(LLM)架构，该架构使用独特的思考链、多点提示方法来定义勒索软件公告、威胁报告和新闻项目中的对手Skram(技能、知识、资源、权威和动机)配置文件。这一分析以可公开获得的受害者数据为基础，并通过启发式方法进一步加强，以生成反映受害者概况的合成数据。我们的工作最终是开发一个机器学习模型，帮助组织根据最有可能的攻击者的战术、技术和程序(TTP)对勒索软件威胁进行优先排序并制定防御措施。



## **36. G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks**

G-Designer：通过图神经网络构建多智能体通信布局 cs.MA

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2410.11782v3) [paper-pdf](http://arxiv.org/pdf/2410.11782v3)

**Authors**: Guibin Zhang, Yanwei Yue, Xiangguo Sun, Guancheng Wan, Miao Yu, Junfeng Fang, Kun Wang, Tianlong Chen, Dawei Cheng

**Abstract**: Recent advancements in large language model (LLM)-based agents have demonstrated that collective intelligence can significantly surpass the capabilities of individual agents, primarily due to well-crafted inter-agent communication topologies. Despite the diverse and high-performing designs available, practitioners often face confusion when selecting the most effective pipeline for their specific task: \textit{Which topology is the best choice for my task, avoiding unnecessary communication token overhead while ensuring high-quality solution?} In response to this dilemma, we introduce G-Designer, an adaptive, efficient, and robust solution for multi-agent deployment, which dynamically designs task-aware, customized communication topologies. Specifically, G-Designer models the multi-agent system as a multi-agent network, leveraging a variational graph auto-encoder to encode both the nodes (agents) and a task-specific virtual node, and decodes a task-adaptive and high-performing communication topology. Extensive experiments on six benchmarks showcase that G-Designer is: \textbf{(1) high-performing}, achieving superior results on MMLU with accuracy at $84.50\%$ and on HumanEval with pass@1 at $89.90\%$; \textbf{(2) task-adaptive}, architecting communication protocols tailored to task difficulty, reducing token consumption by up to $95.33\%$ on HumanEval; and \textbf{(3) adversarially robust}, defending against agent adversarial attacks with merely $0.3\%$ accuracy drop.

摘要: 基于大型语言模型(LLM)的代理的最新进展表明，集体智能可以显著超过单个代理的能力，这主要是由于精心设计的代理间通信拓扑。尽管有多样化和高性能的设计，但实践者在为他们的特定任务选择最有效的流水线时经常面临困惑：\textit{哪个拓扑是我的任务的最佳选择，在确保高质量解决方案的同时避免不必要的通信令牌开销？}针对这种困境，我们引入了G-Designer，这是一个自适应的、高效的、健壮的多代理部署解决方案，它动态地设计任务感知的、定制的通信拓扑。具体地说，G-Designer将多代理系统建模为多代理网络，利用变化图自动编码器对节点(代理)和特定于任务的虚拟节点进行编码，并解码任务自适应的高性能通信拓扑。在六个基准测试上的广泛实验表明，G-Designer是：\extbf{(1)高性能}，在MMLU上获得了更好的结果，准确率为84.50\$，在HumanEval上，PASS@1的准确率为89.90\$；\extbf{(2)任务自适应}，构建了针对任务难度的通信协议，在HumanEval上减少了高达95.33\$的令牌消耗；以及\extbf{(3)对手健壮性}，防御代理对手攻击，精确度仅下降了$0.3\%$。



## **37. A Survey on Backdoor Threats in Large Language Models (LLMs): Attacks, Defenses, and Evaluations**

大型语言模型（LLM）后门威胁调查：攻击、防御和评估 cs.CR

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2502.05224v1) [paper-pdf](http://arxiv.org/pdf/2502.05224v1)

**Authors**: Yihe Zhou, Tao Ni, Wei-Bin Lee, Qingchuan Zhao

**Abstract**: Large Language Models (LLMs) have achieved significantly advanced capabilities in understanding and generating human language text, which have gained increasing popularity over recent years. Apart from their state-of-the-art natural language processing (NLP) performance, considering their widespread usage in many industries, including medicine, finance, education, etc., security concerns over their usage grow simultaneously. In recent years, the evolution of backdoor attacks has progressed with the advancement of defense mechanisms against them and more well-developed features in the LLMs. In this paper, we adapt the general taxonomy for classifying machine learning attacks on one of the subdivisions - training-time white-box backdoor attacks. Besides systematically classifying attack methods, we also consider the corresponding defense methods against backdoor attacks. By providing an extensive summary of existing works, we hope this survey can serve as a guideline for inspiring future research that further extends the attack scenarios and creates a stronger defense against them for more robust LLMs.

摘要: 大型语言模型在理解和生成人类语言文本方面已经取得了显著的进步，这在最近几年得到了越来越多的欢迎。除了它们最先进的自然语言处理(NLP)性能之外，考虑到它们在许多行业中的广泛使用，包括医疗、金融、教育等，对它们使用的安全担忧也在增加。近年来，随着对后门攻击防御机制的进步和LLMS功能的日益完善，后门攻击也在不断发展。在本文中，我们采用一般分类法对机器学习攻击中的一个细分-训练时间白盒后门攻击进行分类。除了系统地对攻击方法进行分类外，我们还考虑了针对后门攻击的相应防御方法。通过对现有工作的广泛总结，我们希望本调查可以作为激励未来研究的指导方针，进一步扩展攻击场景，并为更健壮的LLM创建更强大的防御。



## **38. On Effects of Steering Latent Representation for Large Language Model Unlearning**

论引导潜在表示对大型语言模型取消学习的影响 cs.CL

Accepted at AAAI-25 Main Technical Track

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2408.06223v3) [paper-pdf](http://arxiv.org/pdf/2408.06223v3)

**Authors**: Dang Huu-Tien, Trung-Tin Pham, Hoang Thanh-Tung, Naoya Inoue

**Abstract**: Representation Misdirection for Unlearning (RMU), which steers model representation in the intermediate layer to a target random representation, is an effective method for large language model (LLM) unlearning. Despite its high performance, the underlying cause and explanation remain underexplored. In this paper, we theoretically demonstrate that steering forget representations in the intermediate layer reduces token confidence, causing LLMs to generate wrong or nonsense responses. We investigate how the coefficient influences the alignment of forget-sample representations with the random direction and hint at the optimal coefficient values for effective unlearning across different network layers. We show that RMU unlearned models are robust against adversarial jailbreak attacks. Furthermore, our empirical analysis shows that RMU is less effective when applied to the middle and later layers in LLMs. To resolve this drawback, we propose Adaptive RMU--a simple yet effective alternative method that makes unlearning effective with most layers. Extensive experiments demonstrate that Adaptive RMU significantly improves the unlearning performance compared to prior art while incurring no additional computational cost.

摘要: 遗忘表征误导(RMU)是一种有效的大语言模型遗忘方法，它将中间层的模型表征引导到目标随机表征。尽管其表现良好，但其根本原因和解释仍未得到充分研究。在本文中，我们从理论上证明了中间层中的转向遗忘表征降低了令牌置信度，从而导致LLM产生错误或无意义的响应。我们研究了系数如何影响遗忘样本表示与随机方向的对齐，并提示了跨不同网络层有效遗忘的最优系数值。我们证明了RMU未学习模型对敌意越狱攻击是健壮的。此外，我们的实证分析表明，当RMU应用于LLMS的中后期时，其有效性较差。为了解决这一缺陷，我们提出了自适应RMU--一种简单但有效的替代方法，使遗忘在大多数层都有效。大量实验表明，与现有技术相比，自适应RMU在不增加额外计算代价的情况下，显著改善了遗忘性能。



## **39. GLOV: Guided Large Language Models as Implicit Optimizers for Vision Language Models**

GOV：引导大型语言模型作为视觉语言模型的隐式优化器 cs.CV

Code: https://github.com/jmiemirza/GLOV

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2410.06154v5) [paper-pdf](http://arxiv.org/pdf/2410.06154v5)

**Authors**: M. Jehanzeb Mirza, Mengjie Zhao, Zhuoyuan Mao, Sivan Doveh, Wei Lin, Paul Gavrikov, Michael Dorkenwald, Shiqi Yang, Saurav Jha, Hiromi Wakaki, Yuki Mitsufuji, Horst Possegger, Rogerio Feris, Leonid Karlinsky, James Glass

**Abstract**: In this work, we propose GLOV, which enables Large Language Models (LLMs) to act as implicit optimizers for Vision-Language Models (VLMs) to enhance downstream vision tasks. GLOV prompts an LLM with the downstream task description, querying it for suitable VLM prompts (e.g., for zero-shot classification with CLIP). These prompts are ranked according to their fitness for the downstream vision task. In each respective optimization step, the ranked prompts are fed as in-context examples (with their accuracies) to equip the LLM with the knowledge of the type of prompts preferred by the downstream VLM. Furthermore, we explicitly guide the LLM's generation at each optimization step by adding an offset vector -- calculated from the embedding differences between previous positive and negative solutions -- to the intermediate layer of the network for the next generation. This offset vector biases the LLM generation toward the type of language the downstream VLM prefers, resulting in enhanced performance on the downstream vision tasks. We comprehensively evaluate our GLOV on two tasks: object recognition and the critical task of enhancing VLM safety. Our GLOV shows performance improvement by up to 15.0% and 57.5% for dual-encoder (e.g., CLIP) and encoder-decoder (e.g., LlaVA) models for object recognition and reduces the attack success rate (ASR) on state-of-the-art VLMs by up to $60.7\%$.

摘要: 在这项工作中，我们提出了GLOV，它使得大语言模型(LLM)能够作为视觉语言模型(VLMS)的隐式优化器来增强下游的视觉任务。GLOV用下游任务描述提示LLM，向其查询合适的VLM提示(例如，用于带CLIP的零射击分类)。这些提示根据它们对下游视觉任务的适宜性进行排序。在每个相应的优化步骤中，将经排序的提示作为上下文中的示例(及其准确性)馈送，以使LLM具有下游VLM优选的提示类型的知识。此外，我们在每个优化步骤通过将偏移向量添加到网络的中间层来显式地指导LLM的生成，该偏移量是根据先前正解和负解之间的嵌入差异计算的，以用于下一代。此偏移向量使LLM生成偏向于下游VLM偏爱的语言类型，从而提高了下游视觉任务的性能。我们在两个任务上对我们的GLOV进行了全面评估：目标识别和增强VLM安全的关键任务。我们的GLOV显示，对于用于对象识别的双编码器(例如，CLIP)和编解码器(例如，LlaVA)模型，性能分别提高了15.0%和57.5%，并将最先进的VLM的攻击成功率(ASR)降低了高达60.7美元。



## **40. Aero-LLM: A Distributed Framework for Secure UAV Communication and Intelligent Decision-Making**

Aero-LLM：安全无人机通信和智能决策的分布式框架 cs.CR

This manuscript was accepted by the 1st International Workshop on  Integrated Sensing, Communication, and Computing in Internet of Things (IoT)  Systems at the The 33rd International Conference on Computer Communications  and Networks (ICCCN 2024)

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2502.05220v1) [paper-pdf](http://arxiv.org/pdf/2502.05220v1)

**Authors**: Balakrishnan Dharmalingam, Rajdeep Mukherjee, Brett Piggott, Guohuan Feng, Anyi Liu

**Abstract**: Increased utilization of unmanned aerial vehicles (UAVs) in critical operations necessitates secure and reliable communication with Ground Control Stations (GCS). This paper introduces Aero-LLM, a framework integrating multiple Large Language Models (LLMs) to enhance UAV mission security and operational efficiency. Unlike conventional singular LLMs, Aero-LLM leverages multiple specialized LLMs for various tasks, such as inferencing, anomaly detection, and forecasting, deployed across onboard systems, edge, and cloud servers. This dynamic, distributed architecture reduces performance bottleneck and increases security capabilities. Aero-LLM's evaluation demonstrates outstanding task-specific metrics and robust defense against cyber threats, significantly enhancing UAV decision-making and operational capabilities and security resilience against cyber attacks, setting a new standard for secure, intelligent UAV operations.

摘要: 在关键行动中增加无人机（UF）的利用，需要与地面控制站（GSK）进行安全可靠的通信。本文介绍了Aero-LLM，这是一个集成多个大型语言模型（LLM）的框架，旨在增强无人机任务安全性和运营效率。与传统的单一LLM不同，Aero-LLM利用多个专门的LLM来执行部署在机载系统、边缘和云服务器上的各种任务，例如推理、异常检测和预测。这种动态、分布式体系结构减少了性能瓶颈并提高了安全能力。Aero-LLM的评估展示了出色的特定任务指标和对网络威胁的强大防御，显着增强了无人机的决策和操作能力以及针对网络攻击的安全复原力，为安全、智能的无人机操作树立了新标准。



## **41. Exploring the Security Threats of Knowledge Base Poisoning in Retrieval-Augmented Code Generation**

探索检索增强代码生成中知识库中毒的安全威胁 cs.CR

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2502.03233v1) [paper-pdf](http://arxiv.org/pdf/2502.03233v1)

**Authors**: Bo Lin, Shangwen Wang, Liqian Chen, Xiaoguang Mao

**Abstract**: The integration of Large Language Models (LLMs) into software development has revolutionized the field, particularly through the use of Retrieval-Augmented Code Generation (RACG) systems that enhance code generation with information from external knowledge bases. However, the security implications of RACG systems, particularly the risks posed by vulnerable code examples in the knowledge base, remain largely unexplored. This risk is particularly concerning given that public code repositories, which often serve as the sources for knowledge base collection in RACG systems, are usually accessible to anyone in the community. Malicious attackers can exploit this accessibility to inject vulnerable code into the knowledge base, making it toxic. Once these poisoned samples are retrieved and incorporated into the generated code, they can propagate security vulnerabilities into the final product. This paper presents the first comprehensive study on the security risks associated with RACG systems, focusing on how vulnerable code in the knowledge base compromises the security of generated code. We investigate the LLM-generated code security across different settings through extensive experiments using four major LLMs, two retrievers, and two poisoning scenarios. Our findings highlight the significant threat of knowledge base poisoning, where even a single poisoned code example can compromise up to 48% of generated code. Our findings provide crucial insights into vulnerability introduction in RACG systems and offer practical mitigation recommendations, thereby helping improve the security of LLM-generated code in future works.

摘要: 将大型语言模型(LLM)集成到软件开发中使该领域发生了革命性的变化，特别是通过使用检索-增强代码生成(RACG)系统，该系统利用来自外部知识库的信息来增强代码生成。然而，RACG系统的安全影响，特别是知识库中易受攻击的代码示例带来的风险，在很大程度上仍未得到探索。考虑到公共代码库通常作为RACG系统中知识库收集的来源，社区中的任何人通常都可以访问，这种风险尤其令人担忧。恶意攻击者可以利用这种可访问性将易受攻击的代码注入知识库，使其有毒。一旦这些有毒的样本被检索并合并到生成的代码中，它们就可以将安全漏洞传播到最终产品中。本文首次对RACG系统的安全风险进行了全面的研究，重点研究了知识库中易受攻击的代码如何危及生成代码的安全性。我们通过使用四个主要的LLM、两个检索器和两个中毒场景的广泛实验，研究了不同设置下LLM生成的代码的安全性。我们的发现突出了知识库中毒的重大威胁，在这种情况下，即使是一个中毒的代码示例也可能危及高达48%的生成代码。我们的发现为RACG系统中的漏洞引入提供了重要的见解，并提供了实用的缓解建议，从而有助于在未来的工作中提高LLM生成的代码的安全性。



## **42. ImgTrojan: Jailbreaking Vision-Language Models with ONE Image**

ImgTrojan：具有一张图像的越狱视觉语言模型 cs.CV

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2403.02910v3) [paper-pdf](http://arxiv.org/pdf/2403.02910v3)

**Authors**: Xijia Tao, Shuai Zhong, Lei Li, Qi Liu, Lingpeng Kong

**Abstract**: There has been an increasing interest in the alignment of large language models (LLMs) with human values. However, the safety issues of their integration with a vision module, or vision language models (VLMs), remain relatively underexplored. In this paper, we propose a novel jailbreaking attack against VLMs, aiming to bypass their safety barrier when a user inputs harmful instructions. A scenario where our poisoned (image, text) data pairs are included in the training data is assumed. By replacing the original textual captions with malicious jailbreak prompts, our method can perform jailbreak attacks with the poisoned images. Moreover, we analyze the effect of poison ratios and positions of trainable parameters on our attack's success rate. For evaluation, we design two metrics to quantify the success rate and the stealthiness of our attack. Together with a list of curated harmful instructions, a benchmark for measuring attack efficacy is provided. We demonstrate the efficacy of our attack by comparing it with baseline methods.

摘要: 人们对大型语言模型(LLM)与人类价值观的一致性越来越感兴趣。然而，它们与视觉模块或视觉语言模型(VLM)集成的安全性问题仍然相对较少被探索。在本文中，我们提出了一种新的针对VLM的越狱攻击，目的是在用户输入有害指令时绕过它们的安全屏障。假设我们的有毒(图像、文本)数据对被包括在训练数据中。通过用恶意越狱提示替换原始文本字幕，我们的方法可以使用有毒图像执行越狱攻击。此外，我们还分析了毒物比例和可训练参数的位置对攻击成功率的影响。为了进行评估，我们设计了两个度量标准来量化攻击的成功率和隐蔽性。此外，还提供了一份经过精心策划的有害指令清单，作为衡量攻击效果的基准。我们通过与基线方法进行比较来证明我们的攻击的有效性。



## **43. Understanding and Enhancing the Transferability of Jailbreaking Attacks**

了解并增强越狱攻击的可转移性 cs.LG

Accepted by ICLR 2025

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2502.03052v1) [paper-pdf](http://arxiv.org/pdf/2502.03052v1)

**Authors**: Runqi Lin, Bo Han, Fengwang Li, Tongling Liu

**Abstract**: Jailbreaking attacks can effectively manipulate open-source large language models (LLMs) to produce harmful responses. However, these attacks exhibit limited transferability, failing to disrupt proprietary LLMs consistently. To reliably identify vulnerabilities in proprietary LLMs, this work investigates the transferability of jailbreaking attacks by analysing their impact on the model's intent perception. By incorporating adversarial sequences, these attacks can redirect the source LLM's focus away from malicious-intent tokens in the original input, thereby obstructing the model's intent recognition and eliciting harmful responses. Nevertheless, these adversarial sequences fail to mislead the target LLM's intent perception, allowing the target LLM to refocus on malicious-intent tokens and abstain from responding. Our analysis further reveals the inherent distributional dependency within the generated adversarial sequences, whose effectiveness stems from overfitting the source LLM's parameters, resulting in limited transferability to target LLMs. To this end, we propose the Perceived-importance Flatten (PiF) method, which uniformly disperses the model's focus across neutral-intent tokens in the original input, thus obscuring malicious-intent tokens without relying on overfitted adversarial sequences. Extensive experiments demonstrate that PiF provides an effective and efficient red-teaming evaluation for proprietary LLMs.

摘要: 越狱攻击可以有效地操纵开源的大型语言模型(LLM)来产生有害的响应。然而，这些攻击表现出有限的可转移性，未能始终如一地破坏专有LLM。为了可靠地识别专有LLM中的漏洞，该工作通过分析越狱攻击对模型意图感知的影响来调查越狱攻击的可转移性。通过合并敌意序列，这些攻击可以将源LLM的焦点从原始输入中的恶意标记重新定向，从而阻碍模型的意图识别并引发有害响应。然而，这些敌对序列未能误导目标LLM的意图感知，允许目标LLM重新关注恶意令牌并放弃响应。我们的分析进一步揭示了生成的对抗序列中固有的分布依赖关系，其有效性源于过拟合源LLM的参数，导致对目标LLM的可转移性有限。为此，我们提出了感知重要性平坦化(PIF)方法，该方法将模型的焦点均匀地分散在原始输入中的中性意图标记上，从而在不依赖于过度匹配的敌对序列的情况下模糊恶意意图标记。大量实验表明，PIF为专有LLM提供了一种有效和高效的红团队评估。



## **44. SelfDefend: LLMs Can Defend Themselves against Jailbreaking in a Practical Manner**

SelfDefend：LLM可以以实用的方式保护自己免受越狱的侵害 cs.CR

Accepted by USENIX Security Symposium 2025. Please cite the  conference version of this paper, i.e., "Xunguang Wang, Daoyuan Wu, Zhenlan  Ji, Zongjie Li, Pingchuan Ma, Shuai Wang, Yingjiu Li, Yang Liu, Ning Liu, and  Juergen Rahmel. SelfDefend: LLMs Can Defend Themselves against Jailbreaking  in a Practical Manner. In Proc. USENIX Security, 2025."

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2406.05498v3) [paper-pdf](http://arxiv.org/pdf/2406.05498v3)

**Authors**: Xunguang Wang, Daoyuan Wu, Zhenlan Ji, Zongjie Li, Pingchuan Ma, Shuai Wang, Yingjiu Li, Yang Liu, Ning Liu, Juergen Rahmel

**Abstract**: Jailbreaking is an emerging adversarial attack that bypasses the safety alignment deployed in off-the-shelf large language models (LLMs) and has evolved into multiple categories: human-based, optimization-based, generation-based, and the recent indirect and multilingual jailbreaks. However, delivering a practical jailbreak defense is challenging because it needs to not only handle all the above jailbreak attacks but also incur negligible delays to user prompts, as well as be compatible with both open-source and closed-source LLMs. Inspired by how the traditional security concept of shadow stacks defends against memory overflow attacks, this paper introduces a generic LLM jailbreak defense framework called SelfDefend, which establishes a shadow LLM as a defense instance (in detection state) to concurrently protect the target LLM instance (in normal answering state) in the normal stack and collaborate with it for checkpoint-based access control. The effectiveness of SelfDefend builds upon our observation that existing LLMs can identify harmful prompts or intentions in user queries, which we empirically validate using mainstream GPT-3.5/4 models against major jailbreak attacks. To further improve the defense's robustness and minimize costs, we employ a data distillation approach to tune dedicated open-source defense models. When deployed to protect GPT-3.5/4, Claude, Llama-2-7b/13b, and Mistral, these models outperform seven state-of-the-art defenses and match the performance of GPT-4-based SelfDefend, with significantly lower extra delays. Further experiments show that the tuned models are robust to adaptive jailbreaks and prompt injections.

摘要: 越狱是一种新兴的对抗性攻击，它绕过了现成的大型语言模型(LLM)中部署的安全对齐，并已演变为多种类别：基于人的、基于优化的、基于代的以及最近的间接和多语言越狱。然而，提供实际的越狱防御是具有挑战性的，因为它不仅需要处理所有上述越狱攻击，还需要对用户提示造成可以忽略不计的延迟，以及与开源和闭源LLM兼容。受传统影子堆栈安全概念防御内存溢出攻击的启发，提出了一种通用的LLM越狱防御框架--SelfDefend。该框架建立一个影子LLM作为防御实例(处于检测状态)，同时保护正常堆栈中的目标LLM实例(处于正常应答状态)，并与其协作进行基于检查点的访问控制。SelfDefend的有效性建立在我们的观察基础上，即现有的LLM可以识别用户查询中的有害提示或意图，我们使用主流GPT-3.5/4模型对主要越狱攻击进行了经验验证。为了进一步提高防御的健壮性并将成本降至最低，我们使用数据蒸馏方法来优化专用的开源防御模型。当部署保护GPT-3.5/4、克劳德、Llama-2-7b/13b和米斯特拉尔时，这些型号的性能超过了七种最先进的防御系统，与基于GPT-4的SelfDefend的性能相当，额外延迟显著降低。进一步的实验表明，调整后的模型对自适应越狱和快速注入具有较强的鲁棒性。



## **45. Lost in Overlap: Exploring Logit-based Watermark Collision in LLMs**

迷失在重叠中：探索LLM中基于日志的水印碰撞 cs.CL

Long Paper, 9 pages, accepted at NAACL 2025 Findings

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2403.10020v3) [paper-pdf](http://arxiv.org/pdf/2403.10020v3)

**Authors**: Yiyang Luo, Ke Lin, Chao Gu, Jiahui Hou, Lijie Wen, Ping Luo

**Abstract**: The proliferation of large language models (LLMs) in generating content raises concerns about text copyright. Watermarking methods, particularly logit-based approaches, embed imperceptible identifiers into text to address these challenges. However, the widespread usage of watermarking across diverse LLMs has led to an inevitable issue known as watermark collision during common tasks, such as paraphrasing or translation. In this paper, we introduce watermark collision as a novel and general philosophy for watermark attacks, aimed at enhancing attack performance on top of any other attacking methods. We also provide a comprehensive demonstration that watermark collision poses a threat to all logit-based watermark algorithms, impacting not only specific attack scenarios but also downstream applications.

摘要: 生成内容时大型语言模型（LLM）的激增引发了人们对文本版权的担忧。水印方法，特别是基于日志的方法，将不可感知的标识符嵌入到文本中来解决这些挑战。然而，水印在不同的LLM中的广泛使用导致了在常见任务（例如解释或翻译）期间不可避免的问题，称为水印冲突。在本文中，我们引入水印冲突作为水印攻击的一种新颖且通用的哲学，旨在在任何其他攻击方法之上提高攻击性能。我们还全面证明了水印冲突对所有基于日志的水印算法构成威胁，不仅影响特定的攻击场景，还影响下游应用。



## **46. Large Language Model Adversarial Landscape Through the Lens of Attack Objectives**

从攻击目标角度看大语言模型的对抗格局 cs.CR

15 pages

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2502.02960v1) [paper-pdf](http://arxiv.org/pdf/2502.02960v1)

**Authors**: Nan Wang, Kane Walter, Yansong Gao, Alsharif Abuadbba

**Abstract**: Large Language Models (LLMs) represent a transformative leap in artificial intelligence, enabling the comprehension, generation, and nuanced interaction with human language on an unparalleled scale. However, LLMs are increasingly vulnerable to a range of adversarial attacks that threaten their privacy, reliability, security, and trustworthiness. These attacks can distort outputs, inject biases, leak sensitive information, or disrupt the normal functioning of LLMs, posing significant challenges across various applications.   In this paper, we provide a novel comprehensive analysis of the adversarial landscape of LLMs, framed through the lens of attack objectives. By concentrating on the core goals of adversarial actors, we offer a fresh perspective that examines threats from the angles of privacy, integrity, availability, and misuse, moving beyond conventional taxonomies that focus solely on attack techniques. This objective-driven adversarial landscape not only highlights the strategic intent behind different adversarial approaches but also sheds light on the evolving nature of these threats and the effectiveness of current defenses. Our analysis aims to guide researchers and practitioners in better understanding, anticipating, and mitigating these attacks, ultimately contributing to the development of more resilient and robust LLM systems.

摘要: 大型语言模型(LLM)代表了人工智能的一次革命性飞跃，使人们能够以前所未有的规模理解、生成和与人类语言进行细微差别的交互。然而，LLM越来越容易受到一系列对手攻击，这些攻击威胁到它们的隐私、可靠性、安全性和可信性。这些攻击可能会扭曲输出、注入偏差、泄露敏感信息或扰乱LLMS的正常功能，对各种应用程序构成重大挑战。在这篇文章中，我们提供了一种新颖的全面分析的对抗性景观，通过攻击目标的框架。通过专注于敌对行为者的核心目标，我们提供了一个新的视角，从隐私、完整性、可用性和误用的角度来检查威胁，超越了只关注攻击技术的传统分类。这种以目标为导向的对抗性格局不仅突出了不同对抗性方法背后的战略意图，而且也揭示了这些威胁的演变性质和目前防御的有效性。我们的分析旨在指导研究人员和实践者更好地理解、预测和缓解这些攻击，最终有助于开发更具弹性和健壮性的LLM系统。



## **47. How Much Do Code Language Models Remember? An Investigation on Data Extraction Attacks before and after Fine-tuning**

代码语言模型能记住多少？微调前后数据提取攻击的研究 cs.CR

MSR 2025

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2501.17501v2) [paper-pdf](http://arxiv.org/pdf/2501.17501v2)

**Authors**: Fabio Salerno, Ali Al-Kaswan, Maliheh Izadi

**Abstract**: Code language models, while widely popular, are often trained on unsanitized source code gathered from across the Internet. Previous work revealed that pre-trained models can remember the content of their training data and regurgitate them through data extraction attacks. Due to the large size of current models, only a few entities have the resources for pre-training such models. However, fine-tuning requires fewer resources and is increasingly used by both small and large entities for its effectiveness on specialized data. Such small curated data for fine-tuning might contain sensitive information or proprietary assets. In this study, we attack both pre-trained and fine-tuned code language models to investigate the extent of data extractability. We first develop a custom benchmark to assess the vulnerability of both pre-training and fine-tuning samples to extraction attacks. Our findings reveal that 54.9% of extractable pre-training data could be retrieved from StarCoder2-15B, whereas this number decreased to 23.5% after fine-tuning. This indicates that fine-tuning reduces the extractability of pre-training data. However, compared to larger models, fine-tuning smaller models increases their vulnerability to data extraction attacks on fine-tuning data. Given the potential sensitivity of fine-tuning data, this can lead to more severe consequences. Lastly, we also manually analyzed 2000 extractable samples before and after fine-tuning. We also found that data carriers and licensing information are the most likely data categories to be memorized from pre-trained and fine-tuned models, while the latter is the most likely to be forgotten after fine-tuning.

摘要: 代码语言模型虽然广受欢迎，但通常是针对从互联网上收集的未经清理的源代码进行培训的。以前的工作表明，预先训练的模型可以记住它们的训练数据的内容，并通过数据提取攻击来反胃它们。由于目前模型的规模很大，只有少数几个实体有资源对这些模型进行预培训。然而，微调需要的资源更少，而且越来越多地被小型和大型实体使用，因为它对专门数据的有效性。如此小的精选数据用于微调，可能包含敏感信息或专有资产。在这项研究中，我们攻击预训练和微调的代码语言模型，以调查数据可提取的程度。我们首先开发一个定制的基准来评估预先训练和微调样本对提取攻击的脆弱性。我们的研究结果表明，54.9%的可提取预训练数据可以从StarCoder2-15B中检索到，而经过微调后，这一数字下降到23.5%。这表明微调降低了训练前数据的可提取性。然而，与较大的模型相比，微调较小的模型会增加它们对微调数据的数据提取攻击的脆弱性。鉴于微调数据的潜在敏感性，这可能导致更严重的后果。最后，我们还对微调前后的2000个可提取样本进行了手工分析。我们还发现，从预先训练和微调的模型中，数据载体和许可信息是最容易被记忆的数据类别，而后者在微调后最容易被忘记。



## **48. SimMark: A Robust Sentence-Level Similarity-Based Watermarking Algorithm for Large Language Models**

SimMark：一种针对大型语言模型的稳健基于句子级相似性的水印算法 cs.CL

15 pages, 5 tables, 6 figures

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2502.02787v1) [paper-pdf](http://arxiv.org/pdf/2502.02787v1)

**Authors**: Amirhossein Dabiriaghdam, Lele Wang

**Abstract**: The rapid proliferation of large language models (LLMs) has created an urgent need for reliable methods to detect whether a text is generated by such models. In this paper, we propose SimMark, a posthoc watermarking algorithm that makes LLMs' outputs traceable without requiring access to the model's internal logits, enabling compatibility with a wide range of LLMs, including API-only models. By leveraging the similarity of semantic sentence embeddings and rejection sampling to impose detectable statistical patterns imperceptible to humans, and employing a soft counting mechanism, SimMark achieves robustness against paraphrasing attacks. Experimental results demonstrate that SimMark sets a new benchmark for robust watermarking of LLM-generated content, surpassing prior sentence-level watermarking techniques in robustness, sampling efficiency, and applicability across diverse domains, all while preserving the text quality.

摘要: 大型语言模型（LLM）的迅速激增迫切需要可靠的方法来检测文本是否由此类模型生成。在本文中，我们提出了SimMark，这是一种后置水印算法，可以使LLM的输出可追溯，而无需访问模型的内部日志，从而能够与广泛的LLM兼容，包括仅API模型。通过利用语义句子嵌入和拒绝采样的相似性来强加人类难以感知的可检测统计模式，并采用软计数机制，SimMark实现了针对重述攻击的鲁棒性。实验结果表明，SimMark为LLM生成的内容的鲁棒水印设定了新的基准，在鲁棒性、采样效率和跨不同领域的适用性方面超越了先前的业务级水印技术，同时保持了文本质量。



## **49. MARAGE: Transferable Multi-Model Adversarial Attack for Retrieval-Augmented Generation Data Extraction**

MARAGE：用于检索增强代数据提取的可转移多模型对抗攻击 cs.CL

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2502.04360v1) [paper-pdf](http://arxiv.org/pdf/2502.04360v1)

**Authors**: Xiao Hu, Eric Liu, Weizhou Wang, Xiangyu Guo, David Lie

**Abstract**: Retrieval-Augmented Generation (RAG) offers a solution to mitigate hallucinations in Large Language Models (LLMs) by grounding their outputs to knowledge retrieved from external sources. The use of private resources and data in constructing these external data stores can expose them to risks of extraction attacks, in which attackers attempt to steal data from these private databases. Existing RAG extraction attacks often rely on manually crafted prompts, which limit their effectiveness. In this paper, we introduce a framework called MARAGE for optimizing an adversarial string that, when appended to user queries submitted to a target RAG system, causes outputs containing the retrieved RAG data verbatim. MARAGE leverages a continuous optimization scheme that integrates gradients from multiple models with different architectures simultaneously to enhance the transferability of the optimized string to unseen models. Additionally, we propose a strategy that emphasizes the initial tokens in the target RAG data, further improving the attack's generalizability. Evaluations show that MARAGE consistently outperforms both manual and optimization-based baselines across multiple LLMs and RAG datasets, while maintaining robust transferability to previously unseen models. Moreover, we conduct probing tasks to shed light on the reasons why MARAGE is more effective compared to the baselines and to analyze the impact of our approach on the model's internal state.

摘要: 检索-增强生成(RAG)提供了一种解决方案，通过将大型语言模型(LLM)的输出与从外部来源检索的知识相结合来缓解幻觉。在构建这些外部数据存储时使用私有资源和数据可能会使它们面临提取攻击的风险，即攻击者试图从这些私有数据库中窃取数据。现有的RAG提取攻击通常依赖于手动创建的提示，这限制了它们的有效性。在本文中，我们介绍了一个名为Marage的框架，用于优化敌意字符串，当该字符串附加到提交到目标RAG系统的用户查询时，会导致输出包含检索到的RAG数据。Marage利用持续优化方案，同时集成来自具有不同架构的多个模型的梯度，以增强优化后的字符串到不可见模型的可转移性。此外，我们还提出了一种强调目标RAG数据中初始令牌的策略，进一步提高了攻击的泛化能力。评估表明，Marage在多个LLM和RAG数据集上的表现始终优于手动基准和基于优化的基准，同时保持了到以前未见过的模型的强大可转移性。此外，我们进行了探索性任务，以阐明为什么Marage比基线更有效的原因，并分析我们的方法对模型内部状态的影响。



## **50. Certifying LLM Safety against Adversarial Prompting**

针对对抗性预算认证LLM安全性 cs.CL

Accepted at COLM 2024: https://openreview.net/forum?id=9Ik05cycLq

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2309.02705v4) [paper-pdf](http://arxiv.org/pdf/2309.02705v4)

**Authors**: Aounon Kumar, Chirag Agarwal, Suraj Srinivas, Aaron Jiaxun Li, Soheil Feizi, Himabindu Lakkaraju

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that add malicious tokens to an input prompt to bypass the safety guardrails of an LLM and cause it to produce harmful content. In this work, we introduce erase-and-check, the first framework for defending against adversarial prompts with certifiable safety guarantees. Given a prompt, our procedure erases tokens individually and inspects the resulting subsequences using a safety filter. Our safety certificate guarantees that harmful prompts are not mislabeled as safe due to an adversarial attack up to a certain size. We implement the safety filter in two ways, using Llama 2 and DistilBERT, and compare the performance of erase-and-check for the two cases. We defend against three attack modes: i) adversarial suffix, where an adversarial sequence is appended at the end of a harmful prompt; ii) adversarial insertion, where the adversarial sequence is inserted anywhere in the middle of the prompt; and iii) adversarial infusion, where adversarial tokens are inserted at arbitrary positions in the prompt, not necessarily as a contiguous block. Our experimental results demonstrate that this procedure can obtain strong certified safety guarantees on harmful prompts while maintaining good empirical performance on safe prompts. Additionally, we propose three efficient empirical defenses: i) RandEC, a randomized subsampling version of erase-and-check; ii) GreedyEC, which greedily erases tokens that maximize the softmax score of the harmful class; and iii) GradEC, which uses gradient information to optimize tokens to erase. We demonstrate their effectiveness against adversarial prompts generated by the Greedy Coordinate Gradient (GCG) attack algorithm. The code for our experiments is available at https://github.com/aounon/certified-llm-safety.

摘要: 大型语言模型(LLM)容易受到敌意攻击，这些攻击会向输入提示添加恶意令牌，以绕过LLM的安全护栏，导致其产生有害内容。在这项工作中，我们引入了Erase-and-Check，这是第一个用于防御具有可证明安全保证的对抗性提示的框架。在给定提示的情况下，我们的过程将逐个擦除令牌，并使用安全过滤器检查结果子序列。我们的安全证书保证有害提示不会因为达到一定大小的敌意攻击而被错误地标记为安全。我们用Llama 2和DistilBERT两种方法实现了安全过滤器，并比较了两种情况下的擦除和检查性能。我们防御三种攻击模式：i)对抗性后缀，其中对抗性序列被附加在有害提示的末尾；ii)对抗性插入，其中对抗性序列被插入在提示中间的任何位置；以及iii)对抗性注入，其中对抗性标记被插入在提示中的任意位置，而不一定作为连续的块。实验结果表明，该方法在对安全提示保持较好经验性能的同时，对有害提示具有较强的认证安全保障。此外，我们还提出了三种有效的经验防御：i)RandEC，一种随机化的擦除和检查版本；ii)GreedyEC，它贪婪地擦除使有害类别的Softmax得分最大化的标记；以及iii)Gradec，它使用梯度信息来优化要擦除的标记。我们证明了它们对贪婪坐标梯度(GCG)攻击算法生成的敌意提示的有效性。我们实验的代码可以在https://github.com/aounon/certified-llm-safety.上找到



