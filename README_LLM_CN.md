# Latest Large Language Model Attack Papers
**update at 2024-03-11 09:22:31**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Can LLMs Follow Simple Rules?**

低收入国家能遵循简单的规则吗？ cs.AI

Project website: https://eecs.berkeley.edu/~normanmu/llm_rules;  revised content

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2311.04235v3) [paper-pdf](http://arxiv.org/pdf/2311.04235v3)

**Authors**: Norman Mu, Sarah Chen, Zifan Wang, Sizhe Chen, David Karamardian, Lulwa Aljeraisy, Basel Alomair, Dan Hendrycks, David Wagner

**Abstract**: As Large Language Models (LLMs) are deployed with increasing real-world responsibilities, it is important to be able to specify and constrain the behavior of these systems in a reliable manner. Model developers may wish to set explicit rules for the model, such as "do not generate abusive content", but these may be circumvented by jailbreaking techniques. Existing evaluations of adversarial attacks and defenses on LLMs generally require either expensive manual review or unreliable heuristic checks. To address this issue, we propose Rule-following Language Evaluation Scenarios (RuLES), a programmatic framework for measuring rule-following ability in LLMs. RuLES consists of 14 simple text scenarios in which the model is instructed to obey various rules while interacting with the user. Each scenario has a programmatic evaluation function to determine whether the model has broken any rules in a conversation. Our evaluations of proprietary and open models show that almost all current models struggle to follow scenario rules, even on straightforward test cases. We also demonstrate that simple optimization attacks suffice to significantly increase failure rates on test cases. We conclude by exploring two potential avenues for improvement: test-time steering and supervised fine-tuning.

摘要: 随着大型语言模型(LLM)的部署承担着越来越多的现实责任，能够以可靠的方式指定和约束这些系统的行为是很重要的。模型开发人员可能希望为模型设置明确的规则，例如“不要生成滥用内容”，但可以通过越狱技术绕过这些规则。现有的对抗性攻击和防御评估通常需要昂贵的人工审查或不可靠的启发式检查。为了解决这一问题，我们提出了规则遵循语言评估场景(Rules)，这是一个衡量LLMS中规则遵循能力的程序性框架。规则由14个简单的文本场景组成，在这些场景中，模型被指示在与用户交互时遵守各种规则。每个场景都有一个程序化的评估功能，以确定模型是否违反了对话中的任何规则。我们对专有和开放模型的评估表明，几乎所有当前的模型都难以遵循场景规则，即使在简单的测试用例上也是如此。我们还证明了简单的优化攻击足以显著增加测试用例的失败率。最后，我们探索了两个潜在的改进途径：测试时间控制和有监督的微调。



## **2. Warfare:Breaking the Watermark Protection of AI-Generated Content**

战争：打破人工智能生成内容的水印保护 cs.CV

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2310.07726v3) [paper-pdf](http://arxiv.org/pdf/2310.07726v3)

**Authors**: Guanlin Li, Yifei Chen, Jie Zhang, Jiwei Li, Shangwei Guo, Tianwei Zhang

**Abstract**: AI-Generated Content (AIGC) is gaining great popularity, with many emerging commercial services and applications. These services leverage advanced generative models, such as latent diffusion models and large language models, to generate creative content (e.g., realistic images and fluent sentences) for users. The usage of such generated content needs to be highly regulated, as the service providers need to ensure the users do not violate the usage policies (e.g., abuse for commercialization, generating and distributing unsafe content). A promising solution to achieve this goal is watermarking, which adds unique and imperceptible watermarks on the content for service verification and attribution. Numerous watermarking approaches have been proposed recently. However, in this paper, we show that an adversary can easily break these watermarking mechanisms. Specifically, we consider two possible attacks. (1) Watermark removal: the adversary can easily erase the embedded watermark from the generated content and then use it freely bypassing the regulation of the service provider. (2) Watermark forging: the adversary can create illegal content with forged watermarks from another user, causing the service provider to make wrong attributions. We propose Warfare, a unified methodology to achieve both attacks in a holistic way. The key idea is to leverage a pre-trained diffusion model for content processing and a generative adversarial network for watermark removal or forging. We evaluate Warfare on different datasets and embedding setups. The results prove that it can achieve high success rates while maintaining the quality of the generated content. Compared to existing diffusion model-based attacks, Warfare is 5,050~11,000x faster.

摘要: 人工智能生成的内容(AIGC)越来越受欢迎，出现了许多新兴的商业服务和应用程序。这些服务利用高级生成模型，如潜在扩散模型和大型语言模型，为用户生成创造性内容(例如，逼真的图像和流畅的句子)。这种生成的内容的使用需要受到严格的监管，因为服务提供商需要确保用户不违反使用策略(例如，滥用以商业化、生成和分发不安全的内容)。实现这一目标的一个有前途的解决方案是水印，它在内容上添加唯一且不可察觉的水印，用于服务验证和归属。最近，人们提出了许多水印方法。然而，在本文中，我们证明了攻击者可以很容易地破解这些水印机制。具体地说，我们考虑两种可能的攻击。(1)水印去除：攻击者可以很容易地从生成的内容中删除嵌入的水印，然后绕过服务提供商的监管自由使用。(2)水印伪造：对手可以利用来自其他用户的伪造水印创建非法内容，导致服务提供商做出错误的归属。我们提出战争，一种统一的方法论，以整体的方式实现这两种攻击。其关键思想是利用预先训练的扩散模型来进行内容处理，并利用生成性对抗网络来去除或伪造水印。我们对不同的数据集和嵌入设置进行了战争评估。实验结果表明，该算法在保证生成内容质量的同时，具有较高的成功率。与现有的基于扩散模型的攻击相比，战争的速度要快5050~11000倍。



## **3. On Protecting the Data Privacy of Large Language Models (LLMs): A Survey**

大型语言模型数据隐私保护研究综述 cs.CR

18 pages, 4 figures

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05156v1) [paper-pdf](http://arxiv.org/pdf/2403.05156v1)

**Authors**: Biwei Yan, Kun Li, Minghui Xu, Yueyan Dong, Yue Zhang, Zhaochun Ren, Xiuzheng Cheng

**Abstract**: Large language models (LLMs) are complex artificial intelligence systems capable of understanding, generating and translating human language. They learn language patterns by analyzing large amounts of text data, allowing them to perform writing, conversation, summarizing and other language tasks. When LLMs process and generate large amounts of data, there is a risk of leaking sensitive information, which may threaten data privacy. This paper concentrates on elucidating the data privacy concerns associated with LLMs to foster a comprehensive understanding. Specifically, a thorough investigation is undertaken to delineate the spectrum of data privacy threats, encompassing both passive privacy leakage and active privacy attacks within LLMs. Subsequently, we conduct an assessment of the privacy protection mechanisms employed by LLMs at various stages, followed by a detailed examination of their efficacy and constraints. Finally, the discourse extends to delineate the challenges encountered and outline prospective directions for advancement in the realm of LLM privacy protection.

摘要: 大语言模型是一种能够理解、生成和翻译人类语言的复杂人工智能系统。他们通过分析大量的文本数据来学习语言模式，使他们能够执行写作、对话、摘要和其他语言任务。当LLMS处理和生成大量数据时，存在泄露敏感信息的风险，这可能会威胁到数据隐私。本文集中于阐明与低成本管理相关的数据隐私问题，以促进全面的理解。具体地说，进行了彻底的调查，以勾勒出数据隐私威胁的范围，包括LLMS中的被动隐私泄露和主动隐私攻击。随后，我们对LLMS在不同阶段采用的隐私保护机制进行了评估，然后详细研究了它们的有效性和制约因素。最后，论述了所遇到的挑战，并勾勒出在LLM隐私保护领域取得进展的预期方向。



## **4. Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models**

针对大型语言模型的间接提示注入攻击的基准测试和防御 cs.CL

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2312.14197v3) [paper-pdf](http://arxiv.org/pdf/2312.14197v3)

**Authors**: Jingwei Yi, Yueqi Xie, Bin Zhu, Emre Kiciman, Guangzhong Sun, Xing Xie, Fangzhao Wu

**Abstract**: The integration of large language models (LLMs) with external content has enabled more up-to-date and wide-ranging applications of LLMs, such as Microsoft Copilot. However, this integration has also exposed LLMs to the risk of indirect prompt injection attacks, where an attacker can embed malicious instructions within external content, compromising LLM output and causing responses to deviate from user expectations. To investigate this important but underexplored issue, we introduce the first benchmark for indirect prompt injection attacks, named BIPIA, to evaluate the risk of such attacks. Based on the evaluation, our work makes a key analysis of the underlying reason for the success of the attack, namely the inability of LLMs to distinguish between instructions and external content and the absence of LLMs' awareness to not execute instructions within external content. Building upon this analysis, we develop two black-box methods based on prompt learning and a white-box defense method based on fine-tuning with adversarial training accordingly. Experimental results demonstrate that black-box defenses are highly effective in mitigating these attacks, while the white-box defense reduces the attack success rate to near-zero levels. Overall, our work systematically investigates indirect prompt injection attacks by introducing a benchmark, analyzing the underlying reason for the success of the attack, and developing an initial set of defenses.

摘要: 大型语言模型(LLM)与外部内容的集成使LLM能够更新、更广泛地应用，如Microsoft Copilot。然而，这种集成也使LLMS面临间接提示注入攻击的风险，攻击者可以在外部内容中嵌入恶意指令，损害LLM输出并导致响应偏离用户预期。为了研究这一重要但未被探索的问题，我们引入了第一个间接即时注入攻击基准，称为BIPIA，以评估此类攻击的风险。在评估的基础上，我们的工作重点分析了攻击成功的根本原因，即LLMS无法区分指令和外部内容，以及LLMS缺乏不执行外部内容中的指令的意识。在此基础上，我们提出了两种基于快速学习的黑盒防御方法和一种基于微调对抗性训练的白盒防御方法。实验结果表明，黑盒防御对于缓解这些攻击是非常有效的，而白盒防御将攻击成功率降低到接近于零的水平。总体而言，我们的工作通过引入基准、分析攻击成功的根本原因以及开发一套初始防御措施来系统地调查间接即时注入攻击。



## **5. SecGPT: An Execution Isolation Architecture for LLM-Based Systems**

SecGPT：基于LLM的系统执行隔离体系结构 cs.CR

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.04960v1) [paper-pdf](http://arxiv.org/pdf/2403.04960v1)

**Authors**: Yuhao Wu, Franziska Roesner, Tadayoshi Kohno, Ning Zhang, Umar Iqbal

**Abstract**: Large language models (LLMs) extended as systems, such as ChatGPT, have begun supporting third-party applications. These LLM apps leverage the de facto natural language-based automated execution paradigm of LLMs: that is, apps and their interactions are defined in natural language, provided access to user data, and allowed to freely interact with each other and the system. These LLM app ecosystems resemble the settings of earlier computing platforms, where there was insufficient isolation between apps and the system. Because third-party apps may not be trustworthy, and exacerbated by the imprecision of the natural language interfaces, the current designs pose security and privacy risks for users. In this paper, we propose SecGPT, an architecture for LLM-based systems that aims to mitigate the security and privacy issues that arise with the execution of third-party apps. SecGPT's key idea is to isolate the execution of apps and more precisely mediate their interactions outside of their isolated environments. We evaluate SecGPT against a number of case study attacks and demonstrate that it protects against many security, privacy, and safety issues that exist in non-isolated LLM-based systems. The performance overhead incurred by SecGPT to improve security is under 0.3x for three-quarters of the tested queries. To foster follow-up research, we release SecGPT's source code at https://github.com/llm-platform-security/SecGPT.

摘要: 扩展为系统的大型语言模型(LLM)，如ChatGPT，已经开始支持第三方应用程序。这些LLM应用程序利用LLMS事实上基于自然语言的自动执行范例：即，应用程序及其交互以自然语言定义，提供对用户数据的访问，并允许彼此和系统自由交互。这些LLM应用程序生态系统类似于早期计算平台的设置，在那里应用程序和系统之间没有足够的隔离。由于第三方应用程序可能不值得信任，而且自然语言界面的不精确性加剧了这一问题，目前的设计给用户带来了安全和隐私风险。在本文中，我们提出了SecGPT，这是一个基于LLM的系统的体系结构，旨在缓解因执行第三方应用程序而产生的安全和隐私问题。SecGPT的关键思想是隔离应用程序的执行，更准确地协调它们在隔离环境之外的交互。我们针对许多案例研究攻击对SecGPT进行评估，并证明它可防御非隔离的基于LLM的系统中存在的许多安全、隐私和安全问题。对于四分之三的测试查询，SecGPT为提高安全性而产生的性能开销低于0.3倍。为了促进后续研究，我们在https://github.com/llm-platform-security/SecGPT.上发布了SecGPT的源代码



## **6. Automatic and Universal Prompt Injection Attacks against Large Language Models**

针对大型语言模型的自动和通用提示注入攻击 cs.AI

Pre-print, code is available at  https://github.com/SheltonLiu-N/Universal-Prompt-Injection

**SubmitDate**: 2024-03-07    [abs](http://arxiv.org/abs/2403.04957v1) [paper-pdf](http://arxiv.org/pdf/2403.04957v1)

**Authors**: Xiaogeng Liu, Zhiyuan Yu, Yizhe Zhang, Ning Zhang, Chaowei Xiao

**Abstract**: Large Language Models (LLMs) excel in processing and generating human language, powered by their ability to interpret and follow instructions. However, their capabilities can be exploited through prompt injection attacks. These attacks manipulate LLM-integrated applications into producing responses aligned with the attacker's injected content, deviating from the user's actual requests. The substantial risks posed by these attacks underscore the need for a thorough understanding of the threats. Yet, research in this area faces challenges due to the lack of a unified goal for such attacks and their reliance on manually crafted prompts, complicating comprehensive assessments of prompt injection robustness. We introduce a unified framework for understanding the objectives of prompt injection attacks and present an automated gradient-based method for generating highly effective and universal prompt injection data, even in the face of defensive measures. With only five training samples (0.3% relative to the test data), our attack can achieve superior performance compared with baselines. Our findings emphasize the importance of gradient-based testing, which can avoid overestimation of robustness, especially for defense mechanisms.

摘要: 大型语言模型(LLM)在处理和生成人类语言方面表现出色，其动力来自于它们解释和遵循指令的能力。然而，他们的能力可以通过即时注入攻击来利用。这些攻击操纵LLM集成应用程序生成与攻击者注入的内容一致的响应，从而偏离用户的实际请求。这些袭击造成的重大风险突出表明，必须彻底了解这些威胁。然而，这一领域的研究面临着挑战，因为此类攻击缺乏统一的目标，并且依赖于手动创建的提示，这使得对提示注入健壮性的全面评估变得复杂。我们介绍了一个理解即时注入攻击目标的统一框架，并提出了一种基于自动梯度的方法来生成高效和通用的即时注入数据，即使面对防御措施也是如此。只需要5个训练样本(相对于测试数据0.3%)，我们的攻击可以获得比基线更优越的性能。我们的发现强调了基于梯度的测试的重要性，它可以避免高估稳健性，特别是对于防御机制。



## **7. Membership Inference Attacks and Privacy in Topic Modeling**

主题建模中的成员推理攻击和隐私保护 cs.CR

9 pages + appendices and references. 9 figures. Submitted to USENIX  '24

**SubmitDate**: 2024-03-07    [abs](http://arxiv.org/abs/2403.04451v1) [paper-pdf](http://arxiv.org/pdf/2403.04451v1)

**Authors**: Nico Manzonelli, Wanrong Zhang, Salil Vadhan

**Abstract**: Recent research shows that large language models are susceptible to privacy attacks that infer aspects of the training data. However, it is unclear if simpler generative models, like topic models, share similar vulnerabilities. In this work, we propose an attack against topic models that can confidently identify members of the training data in Latent Dirichlet Allocation. Our results suggest that the privacy risks associated with generative modeling are not restricted to large neural models. Additionally, to mitigate these vulnerabilities, we explore differentially private (DP) topic modeling. We propose a framework for private topic modeling that incorporates DP vocabulary selection as a pre-processing step, and show that it improves privacy while having limited effects on practical utility.

摘要: 最近的研究表明，大型语言模型容易受到隐私攻击，从而推断训练数据的各个方面。然而，尚不清楚更简单的生成性模型，如主题模型，是否也存在类似的漏洞。在这项工作中，我们提出了一种针对主题模型的攻击，该模型可以自信地识别潜在Dirichlet分配中的训练数据成员。我们的结果表明，与生成性建模相关的隐私风险并不局限于大型神经模型。此外，为了缓解这些漏洞，我们探索了差异隐私(DP)主题建模。我们提出了一种隐私主题建模框架，该框架将DP词汇选择作为预处理步骤，并证明了它在提高隐私的同时对实际效用的影响有限。



## **8. PPTC-R benchmark: Towards Evaluating the Robustness of Large Language Models for PowerPoint Task Completion**

PPTC-R基准：评估用于PowerPoint任务完成的大型语言模型的健壮性 cs.CL

LLM evaluation, Multi-turn, Multi-language, Multi-modal benchmark

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03788v1) [paper-pdf](http://arxiv.org/pdf/2403.03788v1)

**Authors**: Zekai Zhang, Yiduo Guo, Yaobo Liang, Dongyan Zhao, Nan Duan

**Abstract**: The growing dependence on Large Language Models (LLMs) for finishing user instructions necessitates a comprehensive understanding of their robustness to complex task completion in real-world situations. To address this critical need, we propose the PowerPoint Task Completion Robustness benchmark (PPTC-R) to measure LLMs' robustness to the user PPT task instruction and software version. Specifically, we construct adversarial user instructions by attacking user instructions at sentence, semantic, and multi-language levels. To assess the robustness of Language Models to software versions, we vary the number of provided APIs to simulate both the newest version and earlier version settings. Subsequently, we test 3 closed-source and 4 open-source LLMs using a benchmark that incorporates these robustness settings, aiming to evaluate how deviations impact LLMs' API calls for task completion. We find that GPT-4 exhibits the highest performance and strong robustness in our benchmark, particularly in the version update and the multilingual settings. However, we find that all LLMs lose their robustness when confronted with multiple challenges (e.g., multi-turn) simultaneously, leading to significant performance drops. We further analyze the robustness behavior and error reasons of LLMs in our benchmark, which provide valuable insights for researchers to understand the LLM's robustness in task completion and develop more robust LLMs and agents. We release the code and data at \url{https://github.com/ZekaiGalaxy/PPTCR}.

摘要: 越来越多地依赖大型语言模型(LLM)来完成用户指令，这就需要全面了解它们在现实世界中完成复杂任务时的健壮性。为了解决这一关键需求，我们提出了PowerPoint任务完成健壮性基准(PPTC-R)来测量LLMS对用户PPT任务指令和软件版本的健壮性。具体地说，我们通过在句子、语义和多语言级别攻击用户指令来构建对抗性用户指令。为了评估语言模型对软件版本的稳健性，我们改变了提供的API的数量，以模拟最新版本和较早版本的设置。随后，我们使用结合了这些健壮性设置的基准测试了3个封闭源代码LLMS和4个开放源代码LLMS，旨在评估偏差如何影响LLMS完成任务的API调用。我们发现GPT-4在我们的基准测试中表现出了最高的性能和强大的健壮性，特别是在版本更新和多语言设置方面。然而，我们发现，当同时面对多个挑战(例如，多回合)时，所有的LLM都失去了它们的健壮性，导致性能显著下降。我们进一步分析了LLM在基准测试中的健壮性行为和错误原因，这为研究人员理解LLM在任务完成时的健壮性以及开发更健壮的LLM和代理提供了有价值的见解。我们将代码和数据发布到\url{https://github.com/ZekaiGalaxy/PPTCR}.



## **9. ImgTrojan: Jailbreaking Vision-Language Models with ONE Image**

IMG特洛伊木马：用一幅图像破解视觉语言模型 cs.CV

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.02910v2) [paper-pdf](http://arxiv.org/pdf/2403.02910v2)

**Authors**: Xijia Tao, Shuai Zhong, Lei Li, Qi Liu, Lingpeng Kong

**Abstract**: There has been an increasing interest in the alignment of large language models (LLMs) with human values. However, the safety issues of their integration with a vision module, or vision language models (VLMs), remain relatively underexplored. In this paper, we propose a novel jailbreaking attack against VLMs, aiming to bypass their safety barrier when a user inputs harmful instructions. A scenario where our poisoned (image, text) data pairs are included in the training data is assumed. By replacing the original textual captions with malicious jailbreak prompts, our method can perform jailbreak attacks with the poisoned images. Moreover, we analyze the effect of poison ratios and positions of trainable parameters on our attack's success rate. For evaluation, we design two metrics to quantify the success rate and the stealthiness of our attack. Together with a list of curated harmful instructions, a benchmark for measuring attack efficacy is provided. We demonstrate the efficacy of our attack by comparing it with baseline methods.

摘要: 人们对大型语言模型(LLM)与人类价值观的一致性越来越感兴趣。然而，它们与视觉模块或视觉语言模型(VLM)集成的安全性问题仍然相对较少被探索。在本文中，我们提出了一种新的针对VLM的越狱攻击，目的是在用户输入有害指令时绕过它们的安全屏障。假设我们的有毒(图像、文本)数据对被包括在训练数据中。通过用恶意越狱提示替换原始文本字幕，我们的方法可以使用有毒图像执行越狱攻击。此外，我们还分析了毒物比例和可训练参数的位置对攻击成功率的影响。为了进行评估，我们设计了两个度量标准来量化攻击的成功率和隐蔽性。此外，还提供了一份经过精心策划的有害指令清单，作为衡量攻击效果的基准。我们通过与基线方法进行比较来证明我们的攻击的有效性。



## **10. Human vs. Machine: Language Models and Wargames**

人与机器：语言模型与战争游戏 cs.CY

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03407v1) [paper-pdf](http://arxiv.org/pdf/2403.03407v1)

**Authors**: Max Lamparth, Anthony Corso, Jacob Ganz, Oriana Skylar Mastro, Jacquelyn Schneider, Harold Trinkunas

**Abstract**: Wargames have a long history in the development of military strategy and the response of nations to threats or attacks. The advent of artificial intelligence (AI) promises better decision-making and increased military effectiveness. However, there is still debate about how AI systems, especially large language models (LLMs), behave as compared to humans. To this end, we use a wargame experiment with 107 national security expert human players designed to look at crisis escalation in a fictional US-China scenario and compare human players to LLM-simulated responses. We find considerable agreement in the LLM and human responses but also significant quantitative and qualitative differences between simulated and human players in the wargame, motivating caution to policymakers before handing over autonomy or following AI-based strategy recommendations.

摘要: 军事演习在军事战略的发展和国家对威胁或攻击的反应方面有着悠久的历史。人工智能(AI)的出现保证了更好的决策和更高的军事效力。然而，人工智能系统，特别是大型语言模型(LLM)与人类相比表现如何，仍存在争议。为此，我们使用了一个有107名国家安全专家人类玩家的战争游戏实验，旨在观察虚构的美国-中国场景中的危机升级，并将人类玩家与LLM模拟的反应进行比较。我们发现，LLM和人类的反应相当一致，但在军事游戏中，模拟玩家和人类玩家在数量和质量上也存在显著差异，这促使政策制定者在移交自主权或遵循基于人工智能的战略建议之前保持谨慎。



## **11. PETA: Parameter-Efficient Trojan Attacks**

善待动物组织：参数高效木马攻击 cs.CL

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2310.00648v4) [paper-pdf](http://arxiv.org/pdf/2310.00648v4)

**Authors**: Lauren Hong, Ting Wang

**Abstract**: Parameter-efficient fine-tuning (PEFT) enables efficient adaptation of pre-trained language models (PLMs) to specific tasks. By tuning only a minimal set of (extra) parameters, PEFT achieves performance that is comparable to standard fine-tuning. However, despite its prevalent use, the security implications of PEFT remain largely unexplored. In this paper, we take the initial steps and present PETA, a novel trojan attack that compromises the weights of PLMs by accounting for downstream adaptation through bilevel optimization: the upper-level objective embeds the backdoor into a model while the lower-level objective simulates PEFT to both retain the PLM's task-specific performance and ensure that the backdoor persists after fine-tuning. With extensive evaluation across a variety of downstream tasks and trigger designs, we demonstrate PETA's effectiveness in terms of both attack success rate and clean accuracy, even when the attacker does not have full knowledge of the victim user's training process.

摘要: 参数高效微调(PEFT)使预先训练的语言模型(PLM)能够有效地适应特定任务。通过只调整最小的一组(额外)参数，PEFT实现了与标准微调相当的性能。然而，尽管PEFT被广泛使用，但其安全影响在很大程度上仍未被探索。在本文中，我们采取了初步的步骤，并提出了一种新的木马攻击PETA，它通过双层优化考虑下游适应来折衷PLM的权重：上层目标将后门嵌入到模型中，而下层目标模拟PEFT，既保留了PLM的特定任务性能，又确保了微调后后门的存在。通过对各种下游任务和触发器设计的广泛评估，我们证明了PETA在攻击成功率和干净准确性方面的有效性，即使攻击者并不完全了解受害者用户的培训过程。



## **12. Gradient Cuff: Detecting Jailbreak Attacks on Large Language Models by Exploring Refusal Loss Landscapes**

梯度袖口：通过探索拒绝损失场景来检测对大型语言模型的越狱攻击 cs.CR

Project page:  https://huggingface.co/spaces/TrustSafeAI/GradientCuff-Jailbreak-Defense

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.00867v2) [paper-pdf](http://arxiv.org/pdf/2403.00867v2)

**Authors**: Xiaomeng Hu, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Large Language Models (LLMs) are becoming a prominent generative AI tool, where the user enters a query and the LLM generates an answer. To reduce harm and misuse, efforts have been made to align these LLMs to human values using advanced training techniques such as Reinforcement Learning from Human Feedback (RLHF). However, recent studies have highlighted the vulnerability of LLMs to adversarial jailbreak attempts aiming at subverting the embedded safety guardrails. To address this challenge, this paper defines and investigates the Refusal Loss of LLMs and then proposes a method called Gradient Cuff to detect jailbreak attempts. Gradient Cuff exploits the unique properties observed in the refusal loss landscape, including functional values and its smoothness, to design an effective two-step detection strategy. Experimental results on two aligned LLMs (LLaMA-2-7B-Chat and Vicuna-7B-V1.5) and six types of jailbreak attacks (GCG, AutoDAN, PAIR, TAP, Base64, and LRL) show that Gradient Cuff can significantly improve the LLM's rejection capability for malicious jailbreak queries, while maintaining the model's performance for benign user queries by adjusting the detection threshold.

摘要: 大型语言模型(LLM)正在成为一种重要的生成性人工智能工具，用户输入一个查询，LLM生成一个答案。为了减少危害和误用，已经做出努力，使用先进的培训技术，如从人类反馈中强化学习(RLHF)，使这些LLM与人类价值保持一致。然而，最近的研究强调了LLMS在旨在颠覆嵌入的安全护栏的对抗性越狱企图中的脆弱性。为了应对这一挑战，本文定义并研究了LLMS的拒绝丢失，提出了一种检测越狱企图的梯度袖口方法。梯度袖口利用拒绝丢失场景中观察到的独特属性，包括函数值及其光滑性，设计了一种有效的两步检测策略。在两个对齐的LLM(Llama-2-7B-Chat和Vicuna-7B-V1.5)和六种越狱攻击(GCG、AutoDAN、Pair、TAP、Base64和LRL)上的实验结果表明，梯度袖口可以显著提高LLM对恶意越狱查询的拒绝能力，同时通过调整检测阈值保持该模型对良性用户查询的性能。



## **13. Measuring Impacts of Poisoning on Model Parameters and Neuron Activations: A Case Study of Poisoning CodeBERT**

测量中毒对模型参数和神经元激活的影响：以中毒CodeBERT为例 cs.SE

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2402.12936v2) [paper-pdf](http://arxiv.org/pdf/2402.12936v2)

**Authors**: Aftab Hussain, Md Rafiqul Islam Rabin, Navid Ayoobi, Mohammad Amin Alipour

**Abstract**: Large language models (LLMs) have revolutionized software development practices, yet concerns about their safety have arisen, particularly regarding hidden backdoors, aka trojans. Backdoor attacks involve the insertion of triggers into training data, allowing attackers to manipulate the behavior of the model maliciously. In this paper, we focus on analyzing the model parameters to detect potential backdoor signals in code models. Specifically, we examine attention weights and biases, activation values, and context embeddings of the clean and poisoned CodeBERT models. Our results suggest noticeable patterns in activation values and context embeddings of poisoned samples for the poisoned CodeBERT model; however, attention weights and biases do not show any significant differences. This work contributes to ongoing efforts in white-box detection of backdoor signals in LLMs of code through the analysis of parameters and activations.

摘要: 大型语言模型(LLM)使软件开发实践发生了革命性的变化，但也出现了对其安全性的担忧，特别是关于隐藏的后门，也就是特洛伊木马。后门攻击包括在训练数据中插入触发器，允许攻击者恶意操纵模型的行为。在本文中，我们重点分析模型参数，以检测代码模型中潜在的后门信号。具体地说，我们检查了干净的和有毒的CodeBERT模型的注意力权重和偏差、激活值和上下文嵌入。我们的结果表明，对于中毒的CodeBERT模型，中毒样本的激活值和上下文嵌入有明显的模式；然而，注意力权重和偏差没有显示出任何显著的差异。这项工作有助于通过分析参数和激活来对代码的LLMS中的后门信号进行白盒检测。



## **14. InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents**

注入代理：对工具集成的大型语言模型代理中的间接提示注入进行基准测试 cs.CL

26 pages, 5 figures, 7 tables

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02691v1) [paper-pdf](http://arxiv.org/pdf/2403.02691v1)

**Authors**: Qiusi Zhan, Zhixiang Liang, Zifan Ying, Daniel Kang

**Abstract**: Recent work has embodied LLMs as agents, allowing them to access tools, perform actions, and interact with external content (e.g., emails or websites). However, external content introduces the risk of indirect prompt injection (IPI) attacks, where malicious instructions are embedded within the content processed by LLMs, aiming to manipulate these agents into executing detrimental actions against users. Given the potentially severe consequences of such attacks, establishing benchmarks to assess and mitigate these risks is imperative.   In this work, we introduce InjecAgent, a benchmark designed to assess the vulnerability of tool-integrated LLM agents to IPI attacks. InjecAgent comprises 1,054 test cases covering 17 different user tools and 62 attacker tools. We categorize attack intentions into two primary types: direct harm to users and exfiltration of private data. We evaluate 30 different LLM agents and show that agents are vulnerable to IPI attacks, with ReAct-prompted GPT-4 vulnerable to attacks 24% of the time. Further investigation into an enhanced setting, where the attacker instructions are reinforced with a hacking prompt, shows additional increases in success rates, nearly doubling the attack success rate on the ReAct-prompted GPT-4. Our findings raise questions about the widespread deployment of LLM Agents. Our benchmark is available at https://github.com/uiuc-kang-lab/InjecAgent.

摘要: 最近的工作将LLMS体现为代理，允许它们访问工具、执行操作并与外部内容(例如，电子邮件或网站)交互。然而，外部内容会带来间接提示注入(IPI)攻击的风险，在IPI攻击中，恶意指令被嵌入到LLMS处理的内容中，目的是操纵这些代理执行针对用户的有害操作。鉴于此类攻击的潜在严重后果，建立评估和减轻这些风险的基准势在必行。在这项工作中，我们引入了InjecAgent，这是一个旨在评估工具集成的LLM代理对IPI攻击的脆弱性的基准测试。InjecAgent由1,054个测试用例组成，涵盖17个不同的用户工具和62个攻击者工具。我们将攻击意图分为两种主要类型：直接伤害用户和泄露私人数据。我们评估了30种不同的LLM代理，表明代理容易受到IPI攻击，其中反应提示的GPT-4在24%的时间内容易受到攻击。对增强设置的进一步调查显示，成功率进一步提高，反应提示GPT-4的攻击成功率几乎翻了一番。在增强设置中，攻击者的指令通过黑客提示得到加强。我们的发现对LLM特工的广泛部署提出了质疑。我们的基准测试可从https://github.com/uiuc-kang-lab/InjecAgent.获得



## **15. KnowPhish: Large Language Models Meet Multimodal Knowledge Graphs for Enhancing Reference-Based Phishing Detection**

KnowPhish：大型语言模型满足多模式知识图以增强基于参考的网络钓鱼检测 cs.CR

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2403.02253v1) [paper-pdf](http://arxiv.org/pdf/2403.02253v1)

**Authors**: Yuexin Li, Chengyu Huang, Shumin Deng, Mei Lin Lock, Tri Cao, Nay Oo, Bryan Hooi, Hoon Wei Lim

**Abstract**: Phishing attacks have inflicted substantial losses on individuals and businesses alike, necessitating the development of robust and efficient automated phishing detection approaches. Reference-based phishing detectors (RBPDs), which compare the logos on a target webpage to a known set of logos, have emerged as the state-of-the-art approach. However, a major limitation of existing RBPDs is that they rely on a manually constructed brand knowledge base, making it infeasible to scale to a large number of brands, which results in false negative errors due to the insufficient brand coverage of the knowledge base. To address this issue, we propose an automated knowledge collection pipeline, using which we collect and release a large-scale multimodal brand knowledge base, KnowPhish, containing 20k brands with rich information about each brand. KnowPhish can be used to boost the performance of existing RBPDs in a plug-and-play manner. A second limitation of existing RBPDs is that they solely rely on the image modality, ignoring useful textual information present in the webpage HTML. To utilize this textual information, we propose a Large Language Model (LLM)-based approach to extract brand information of webpages from text. Our resulting multimodal phishing detection approach, KnowPhish Detector (KPD), can detect phishing webpages with or without logos. We evaluate KnowPhish and KPD on a manually validated dataset, and on a field study under Singapore's local context, showing substantial improvements in effectiveness and efficiency compared to state-of-the-art baselines.

摘要: 网络钓鱼攻击已经给个人和企业造成了巨大的损失，需要开发强大而高效的自动网络钓鱼检测方法。基于引用的网络钓鱼检测器(RBPD)将目标网页上的标识与一组已知的标识进行比较，已成为最先进的方法。然而，现有的RBPD的一个主要局限性是依赖于手动构建的品牌知识库，使得无法扩展到大量的品牌，从而由于知识库的品牌覆盖率不足而导致假阴性错误。为了解决这一问题，我们提出了一种自动化的知识收集管道，利用该管道，我们收集并发布了一个大规模的多模式品牌知识库KnowPhish，其中包含2万个品牌，每个品牌都有丰富的信息。KnowPhish可用于以即插即用的方式提高现有RBPD的性能。现有RBPD的第二个限制是它们仅依赖于图像通道，而忽略了网页HTML中存在的有用文本信息。为了利用这些文本信息，我们提出了一种基于大语言模型的方法来从文本中提取网页的品牌信息。我们由此产生的多模式钓鱼检测方法KnowPhish检测器(KPD)可以检测带有或没有徽标的钓鱼网页。我们在人工验证的数据集和新加坡本地环境下的实地研究中对KnowPhish和KPD进行了评估，结果显示，与最先进的基线相比，KnowPhish和KPD在有效性和效率方面都有显著改进。



## **16. Rethinking Model Ensemble in Transfer-based Adversarial Attacks**

基于迁移的对抗性攻击中模型集成的再思考 cs.CV

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2303.09105v2) [paper-pdf](http://arxiv.org/pdf/2303.09105v2)

**Authors**: Huanran Chen, Yichi Zhang, Yinpeng Dong, Xiao Yang, Hang Su, Jun Zhu

**Abstract**: It is widely recognized that deep learning models lack robustness to adversarial examples. An intriguing property of adversarial examples is that they can transfer across different models, which enables black-box attacks without any knowledge of the victim model. An effective strategy to improve the transferability is attacking an ensemble of models. However, previous works simply average the outputs of different models, lacking an in-depth analysis on how and why model ensemble methods can strongly improve the transferability. In this paper, we rethink the ensemble in adversarial attacks and define the common weakness of model ensemble with two properties: 1) the flatness of loss landscape; and 2) the closeness to the local optimum of each model. We empirically and theoretically show that both properties are strongly correlated with the transferability and propose a Common Weakness Attack (CWA) to generate more transferable adversarial examples by promoting these two properties. Experimental results on both image classification and object detection tasks validate the effectiveness of our approach to improving the adversarial transferability, especially when attacking adversarially trained models. We also successfully apply our method to attack a black-box large vision-language model -- Google's Bard, showing the practical effectiveness. Code is available at \url{https://github.com/huanranchen/AdversarialAttacks}.

摘要: 人们普遍认为，深度学习模型对对抗性例子缺乏稳健性。对抗性例子的一个耐人寻味的特性是，它们可以在不同的模型之间传输，这使得在不知道受害者模型的情况下进行黑盒攻击。提高可转移性的一个有效策略是攻击一系列模型。然而，以往的工作只是简单地对不同模型的输出进行平均，而缺乏对模型集成方法如何以及为什么能够显著提高可转移性的深入分析。在本文中，我们重新考虑了对抗性攻击中的集成，并定义了模型集成的两个共同弱点：1)损失图景的平坦性；2)每个模型接近局部最优。我们从经验和理论上证明了这两个性质与可转移性有很强的相关性，并提出了一种共同弱点攻击(CWA)，通过提升这两个性质来生成更多可转移的对抗性实例。在图像分类和目标检测任务上的实验结果验证了该方法的有效性，特别是在攻击对抗性训练模型时。我们还成功地应用我们的方法攻击了一个黑盒大视觉语言模型--Google的BARD，显示了它的实际有效性。代码可在\url{https://github.com/huanranchen/AdversarialAttacks}.上找到



## **17. One Prompt Word is Enough to Boost Adversarial Robustness for Pre-trained Vision-Language Models**

一个提示词就足以提高预先训练的视觉语言模型的对抗性 cs.CV

CVPR2024

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2403.01849v1) [paper-pdf](http://arxiv.org/pdf/2403.01849v1)

**Authors**: Lin Li, Haoyan Guan, Jianing Qiu, Michael Spratling

**Abstract**: Large pre-trained Vision-Language Models (VLMs) like CLIP, despite having remarkable generalization ability, are highly vulnerable to adversarial examples. This work studies the adversarial robustness of VLMs from the novel perspective of the text prompt instead of the extensively studied model weights (frozen in this work). We first show that the effectiveness of both adversarial attack and defense are sensitive to the used text prompt. Inspired by this, we propose a method to improve resilience to adversarial attacks by learning a robust text prompt for VLMs. The proposed method, named Adversarial Prompt Tuning (APT), is effective while being both computationally and data efficient. Extensive experiments are conducted across 15 datasets and 4 data sparsity schemes (from 1-shot to full training data settings) to show APT's superiority over hand-engineered prompts and other state-of-the-art adaption methods. APT demonstrated excellent abilities in terms of the in-distribution performance and the generalization under input distribution shift and across datasets. Surprisingly, by simply adding one learned word to the prompts, APT can significantly boost the accuracy and robustness (epsilon=4/255) over the hand-engineered prompts by +13% and +8.5% on average respectively. The improvement further increases, in our most effective setting, to +26.4% for accuracy and +16.7% for robustness. Code is available at https://github.com/TreeLLi/APT.

摘要: 像CLIP这样的大型预先训练的视觉语言模型(VLM)，尽管具有显著的泛化能力，但很容易受到对手例子的攻击。该工作从文本提示的新角度来研究VLMS的对抗健壮性，而不是广泛研究的模型权重(在本工作中是冻结的)。我们首先证明了对抗性攻击和防御的有效性都对所使用的文本提示敏感。受此启发，我们提出了一种通过学习VLM的健壮文本提示来提高对对手攻击的恢复能力的方法。该方法称为对抗性提示调优(APT)，在计算效率和数据效率上都是有效的。在15个数据集和4个数据稀疏方案(从单镜头到全训练数据设置)上进行了广泛的实验，以展示APT相对于人工设计提示和其他最先进的适应方法的优势。在输入分布平移和跨数据集情况下，APT在分布内性能和泛化能力方面表现出优异的性能。令人惊讶的是，通过简单地在提示中添加一个学习的单词，APT可以显著提高人工设计提示的准确率和稳健性(epsilon=4/255)，平均分别提高+13%和+8.5%。在我们最有效的设置中，改进进一步增加了准确性的+26.4%和健壮性的+16.7%。代码可在https://github.com/TreeLLi/APT.上找到



## **18. SALAD-Bench: A Hierarchical and Comprehensive Safety Benchmark for Large Language Models**

Salad-BENCH：一种适用于大型语言模型的分层综合安全基准 cs.CL

fix institution typo

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2402.05044v3) [paper-pdf](http://arxiv.org/pdf/2402.05044v3)

**Authors**: Lijun Li, Bowen Dong, Ruohui Wang, Xuhao Hu, Wangmeng Zuo, Dahua Lin, Yu Qiao, Jing Shao

**Abstract**: In the rapidly evolving landscape of Large Language Models (LLMs), ensuring robust safety measures is paramount. To meet this crucial need, we propose \emph{SALAD-Bench}, a safety benchmark specifically designed for evaluating LLMs, attack, and defense methods. Distinguished by its breadth, SALAD-Bench transcends conventional benchmarks through its large scale, rich diversity, intricate taxonomy spanning three levels, and versatile functionalities.SALAD-Bench is crafted with a meticulous array of questions, from standard queries to complex ones enriched with attack, defense modifications and multiple-choice. To effectively manage the inherent complexity, we introduce an innovative evaluators: the LLM-based MD-Judge for QA pairs with a particular focus on attack-enhanced queries, ensuring a seamless, and reliable evaluation. Above components extend SALAD-Bench from standard LLM safety evaluation to both LLM attack and defense methods evaluation, ensuring the joint-purpose utility. Our extensive experiments shed light on the resilience of LLMs against emerging threats and the efficacy of contemporary defense tactics. Data and evaluator are released under https://github.com/OpenSafetyLab/SALAD-BENCH.

摘要: 在快速发展的大型语言模型(LLM)环境中，确保可靠的安全措施至关重要。为了满足这一关键需求，我们提出了一种专门为评估LLMS、攻击和防御方法而设计的安全基准。SALAD-BENCH以其广度、丰富的多样性、跨越三个层次的复杂分类和多功能而超越传统基准。SALAD-BENCH精心设计了一系列细致的问题，从标准查询到复杂的问题，丰富了攻击、防御修改和多项选择。为了有效地管理固有的复杂性，我们引入了一种创新的评估器：基于LLM的针对QA对的MD-裁判，特别关注攻击增强的查询，确保无缝和可靠的评估。上述组件将沙拉工作台从标准的LLM安全评估扩展到LLM攻防方法评估，确保了联合用途的实用性。我们广泛的实验揭示了低密度脂蛋白对新出现的威胁的弹性以及当代防御战术的有效性。数据和评估器在https://github.com/OpenSafetyLab/SALAD-BENCH.下发布



## **19. LLMs Can Defend Themselves Against Jailbreaking in a Practical Manner: A Vision Paper**

LLMS能够以实际的方式保护自己免受越狱：一份愿景文件 cs.CR

Fixed the bibliography reference issue in our LLM jailbreak defense  vision paper submitted on 24 Feb 2024

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2402.15727v2) [paper-pdf](http://arxiv.org/pdf/2402.15727v2)

**Authors**: Daoyuan Wu, Shuai Wang, Yang Liu, Ning Liu

**Abstract**: Jailbreaking is an emerging adversarial attack that bypasses the safety alignment deployed in off-the-shelf large language models (LLMs). A considerable amount of research exists proposing more effective jailbreak attacks, including the recent Greedy Coordinate Gradient (GCG) attack, jailbreak template-based attacks such as using "Do-Anything-Now" (DAN), and multilingual jailbreak. In contrast, the defensive side has been relatively less explored. This paper proposes a lightweight yet practical defense called SELFDEFEND, which can defend against all existing jailbreak attacks with minimal delay for jailbreak prompts and negligible delay for normal user prompts. Our key insight is that regardless of the kind of jailbreak strategies employed, they eventually need to include a harmful prompt (e.g., "how to make a bomb") in the prompt sent to LLMs, and we found that existing LLMs can effectively recognize such harmful prompts that violate their safety policies. Based on this insight, we design a shadow stack that concurrently checks whether a harmful prompt exists in the user prompt and triggers a checkpoint in the normal stack once a token of "No" or a harmful prompt is output. The latter could also generate an explainable LLM response to adversarial prompts. We demonstrate our idea of SELFDEFEND works in various jailbreak scenarios through manual analysis in GPT-3.5/4. We also list three future directions to further enhance SELFDEFEND.

摘要: 越狱是一种新兴的对抗性攻击，它绕过了现成的大型语言模型(LLM)中部署的安全对齐。已有大量研究提出了更有效的越狱攻击方案，包括最近的贪婪坐标梯度(GCG)攻击、基于模板的越狱攻击(例如使用“Do-Anything-Now”(DAN))和多语言越狱。相比之下，防守方面的探索相对较少。本文提出了一种轻量级而实用的防御方法SELFDEFEND，它可以防御所有现有的越狱攻击，而越狱提示的延迟最小，正常用户提示的延迟可以忽略不计。我们的主要见解是，无论采用哪种越狱策略，他们最终都需要在发送给LLMS的提示中包含有害提示(例如，如何制造炸弹)，我们发现现有LLMS可以有效地识别此类违反其安全政策的有害提示。基于这一观点，我们设计了一个影子堆栈，该堆栈同时检查用户提示中是否存在有害提示，并在输出令牌“否”或有害提示时触发正常堆栈中的检查点。后者还可以对对抗性提示产生可解释的LLM响应。我们通过GPT-3.5/4中的手动分析，展示了我们的SELFDEFEND在各种越狱场景中的工作原理。我们还列出了进一步增强SELFDEFEND的三个未来方向。



## **20. Multilingual Jailbreak Challenges in Large Language Models**

大型语言模型中的多语言越狱挑战 cs.CL

ICLR 2024

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2310.06474v3) [paper-pdf](http://arxiv.org/pdf/2310.06474v3)

**Authors**: Yue Deng, Wenxuan Zhang, Sinno Jialin Pan, Lidong Bing

**Abstract**: While large language models (LLMs) exhibit remarkable capabilities across a wide range of tasks, they pose potential safety concerns, such as the ``jailbreak'' problem, wherein malicious instructions can manipulate LLMs to exhibit undesirable behavior. Although several preventive measures have been developed to mitigate the potential risks associated with LLMs, they have primarily focused on English. In this study, we reveal the presence of multilingual jailbreak challenges within LLMs and consider two potential risky scenarios: unintentional and intentional. The unintentional scenario involves users querying LLMs using non-English prompts and inadvertently bypassing the safety mechanisms, while the intentional scenario concerns malicious users combining malicious instructions with multilingual prompts to deliberately attack LLMs. The experimental results reveal that in the unintentional scenario, the rate of unsafe content increases as the availability of languages decreases. Specifically, low-resource languages exhibit about three times the likelihood of encountering harmful content compared to high-resource languages, with both ChatGPT and GPT-4. In the intentional scenario, multilingual prompts can exacerbate the negative impact of malicious instructions, with astonishingly high rates of unsafe output: 80.92\% for ChatGPT and 40.71\% for GPT-4. To handle such a challenge in the multilingual context, we propose a novel \textsc{Self-Defense} framework that automatically generates multilingual training data for safety fine-tuning. Experimental results show that ChatGPT fine-tuned with such data can achieve a substantial reduction in unsafe content generation. Data is available at \url{https://github.com/DAMO-NLP-SG/multilingual-safety-for-LLMs}.

摘要: 虽然大型语言模型(LLM)在广泛的任务中显示出非凡的能力，但它们构成了潜在的安全问题，如“越狱”问题，在该问题中，恶意指令可以操纵LLM表现出不受欢迎的行为。虽然已经制定了几项预防措施来减轻与低密度脂蛋白相关的潜在风险，但它们主要集中在英语上。在这项研究中，我们揭示了多语言越狱挑战在LLMS中的存在，并考虑了两种潜在的风险情景：无意和故意。非故意场景涉及用户使用非英语提示查询LLMS并无意中绕过安全机制，而有意场景涉及恶意用户将恶意指令与多语言提示相结合来故意攻击LLMS。实验结果表明，在无意情况下，不安全内容的发生率随着语言可用性的降低而增加。具体地说，与高资源语言相比，低资源语言遇到有害内容的可能性大约是ChatGPT和GPT-4语言的三倍。在有意为之的场景中，多语言提示会加剧恶意指令的负面影响，不安全输出率高得惊人：ChatGPT为80.92\%，GPT-4为40.71\%。为了应对多语言环境下的这一挑战，我们提出了一种新的\Textsc{自卫}框架，该框架自动生成用于安全微调的多语言训练数据。实验结果表明，利用这些数据对ChatGPT进行微调可以实现对不安全内容生成的大幅减少。有关数据，请访问\url{https://github.com/DAMO-NLP-SG/multilingual-safety-for-LLMs}.



## **21. Breaking Down the Defenses: A Comparative Survey of Attacks on Large Language Models**

击破防线：大型语言模型遭受攻击的比较研究 cs.CR

**SubmitDate**: 2024-03-03    [abs](http://arxiv.org/abs/2403.04786v1) [paper-pdf](http://arxiv.org/pdf/2403.04786v1)

**Authors**: Arijit Ghosh Chowdhury, Md Mofijul Islam, Vaibhav Kumar, Faysal Hossain Shezan, Vaibhav Kumar, Vinija Jain, Aman Chadha

**Abstract**: Large Language Models (LLMs) have become a cornerstone in the field of Natural Language Processing (NLP), offering transformative capabilities in understanding and generating human-like text. However, with their rising prominence, the security and vulnerability aspects of these models have garnered significant attention. This paper presents a comprehensive survey of the various forms of attacks targeting LLMs, discussing the nature and mechanisms of these attacks, their potential impacts, and current defense strategies. We delve into topics such as adversarial attacks that aim to manipulate model outputs, data poisoning that affects model training, and privacy concerns related to training data exploitation. The paper also explores the effectiveness of different attack methodologies, the resilience of LLMs against these attacks, and the implications for model integrity and user trust. By examining the latest research, we provide insights into the current landscape of LLM vulnerabilities and defense mechanisms. Our objective is to offer a nuanced understanding of LLM attacks, foster awareness within the AI community, and inspire robust solutions to mitigate these risks in future developments.

摘要: 大型语言模型(LLM)已经成为自然语言处理(NLP)领域的基石，在理解和生成类似人类的文本方面提供了变革性的能力。然而，随着它们的日益突出，这些模型的安全和漏洞方面已经引起了极大的关注。本文对各种形式的针对LLMS的攻击进行了全面的综述，讨论了这些攻击的性质和机制、它们的潜在影响以及当前的防御策略。我们深入探讨了旨在操纵模型输出的对抗性攻击、影响模型训练的数据中毒以及与训练数据利用相关的隐私问题等主题。文中还探讨了不同攻击方法的有效性，LLMS对这些攻击的恢复能力，以及对模型完整性和用户信任的影响。通过检查最新的研究，我们提供了对LLM漏洞和防御机制的当前情况的见解。我们的目标是提供对LLM攻击的细微差别的理解，培养人工智能社区的意识，并激发强大的解决方案，以减轻未来发展中的这些风险。



## **22. Ignore This Title and HackAPrompt: Exposing Systemic Vulnerabilities of LLMs through a Global Scale Prompt Hacking Competition**

忽略这个标题和HackAPrompt：通过全球规模的即时黑客竞赛揭露LLMS的系统性漏洞 cs.CR

34 pages, 8 figures Codebase:  https://github.com/PromptLabs/hackaprompt Dataset:  https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset/blob/main/README.md  Playground: https://huggingface.co/spaces/hackaprompt/playground

**SubmitDate**: 2024-03-03    [abs](http://arxiv.org/abs/2311.16119v3) [paper-pdf](http://arxiv.org/pdf/2311.16119v3)

**Authors**: Sander Schulhoff, Jeremy Pinto, Anaum Khan, Louis-François Bouchard, Chenglei Si, Svetlina Anati, Valen Tagliabue, Anson Liu Kost, Christopher Carnahan, Jordan Boyd-Graber

**Abstract**: Large Language Models (LLMs) are deployed in interactive contexts with direct user engagement, such as chatbots and writing assistants. These deployments are vulnerable to prompt injection and jailbreaking (collectively, prompt hacking), in which models are manipulated to ignore their original instructions and follow potentially malicious ones. Although widely acknowledged as a significant security threat, there is a dearth of large-scale resources and quantitative studies on prompt hacking. To address this lacuna, we launch a global prompt hacking competition, which allows for free-form human input attacks. We elicit 600K+ adversarial prompts against three state-of-the-art LLMs. We describe the dataset, which empirically verifies that current LLMs can indeed be manipulated via prompt hacking. We also present a comprehensive taxonomical ontology of the types of adversarial prompts.

摘要: 大型语言模型(LLM)部署在具有直接用户参与的交互上下文中，例如聊天机器人和写作助手。这些部署容易受到即时注入和越狱(统称为即时黑客)的攻击，在这些情况下，模型被操纵以忽略其原始指令并遵循潜在的恶意指令。尽管被广泛认为是一个重大的安全威胁，但缺乏关于即时黑客攻击的大规模资源和量化研究。为了弥补这一漏洞，我们发起了一场全球即时黑客竞赛，允许自由形式的人工输入攻击。我们在三个最先进的LLM上获得了600K+的对抗性提示。我们描述了数据集，这从经验上验证了当前的LLM确实可以通过即时黑客来操纵。我们还提出了对抗性提示类型的全面分类本体。



## **23. Analysis of Privacy Leakage in Federated Large Language Models**

联邦大型语言模型中的隐私泄露分析 cs.CR

**SubmitDate**: 2024-03-02    [abs](http://arxiv.org/abs/2403.04784v1) [paper-pdf](http://arxiv.org/pdf/2403.04784v1)

**Authors**: Minh N. Vu, Truc Nguyen, Tre' R. Jeter, My T. Thai

**Abstract**: With the rapid adoption of Federated Learning (FL) as the training and tuning protocol for applications utilizing Large Language Models (LLMs), recent research highlights the need for significant modifications to FL to accommodate the large-scale of LLMs. While substantial adjustments to the protocol have been introduced as a response, comprehensive privacy analysis for the adapted FL protocol is currently lacking.   To address this gap, our work delves into an extensive examination of the privacy analysis of FL when used for training LLMs, both from theoretical and practical perspectives. In particular, we design two active membership inference attacks with guaranteed theoretical success rates to assess the privacy leakages of various adapted FL configurations. Our theoretical findings are translated into practical attacks, revealing substantial privacy vulnerabilities in popular LLMs, including BERT, RoBERTa, DistilBERT, and OpenAI's GPTs, across multiple real-world language datasets. Additionally, we conduct thorough experiments to evaluate the privacy leakage of these models when data is protected by state-of-the-art differential privacy (DP) mechanisms.

摘要: 随着联邦学习(FL)作为使用大型语言模型(LLM)的应用程序的训练和调优协议的快速采用，最近的研究强调了对FL进行重大修改以适应大规模LLM的必要性。作为回应，虽然已经对协议进行了实质性的调整，但目前还缺乏对修改后的FL协议的全面隐私分析。为了弥补这一差距，我们的工作从理论和实践两个角度深入研究了外语隐私分析在用于培训LLM时的情况。特别是，我们设计了两个保证理论成功率的主动成员推理攻击，以评估各种适应的FL配置的隐私泄漏。我们的理论发现转化为实际攻击，揭示了包括Bert、Roberta、DistilBERT和OpenAI的GPT在内的流行LLM中跨多个现实世界语言数据集的大量隐私漏洞。此外，我们还进行了深入的实验，以评估这些模型在数据受到最先进的差异隐私保护(DP)机制时的隐私泄露情况。



## **24. AutoDefense: Multi-Agent LLM Defense against Jailbreak Attacks**

自卫：针对越狱攻击的多代理LLM防御 cs.LG

**SubmitDate**: 2024-03-02    [abs](http://arxiv.org/abs/2403.04783v1) [paper-pdf](http://arxiv.org/pdf/2403.04783v1)

**Authors**: Yifan Zeng, Yiran Wu, Xiao Zhang, Huazheng Wang, Qingyun Wu

**Abstract**: Despite extensive pre-training and fine-tuning in moral alignment to prevent generating harmful information at user request, large language models (LLMs) remain vulnerable to jailbreak attacks. In this paper, we propose AutoDefense, a response-filtering based multi-agent defense framework that filters harmful responses from LLMs. This framework assigns different roles to LLM agents and employs them to complete the defense task collaboratively. The division in tasks enhances the overall instruction-following of LLMs and enables the integration of other defense components as tools. AutoDefense can adapt to various sizes and kinds of open-source LLMs that serve as agents. Through conducting extensive experiments on a large scale of harmful and safe prompts, we validate the effectiveness of the proposed AutoDefense in improving the robustness against jailbreak attacks, while maintaining the performance at normal user request. Our code and data are publicly available at https://github.com/XHMY/AutoDefense.

摘要: 尽管在道德一致性方面进行了广泛的预培训和微调，以防止应用户要求生成有害信息，但大型语言模型(LLM)仍然容易受到越狱攻击。在本文中，我们提出了一种基于响应过滤的多智能体防御框架--AutoDefense，用于过滤来自LLMS的有害响应。该框架为LLM代理分配不同的角色，并利用它们协同完成防御任务。任务分工加强了LLMS的整体指令遵循，并使其他防御组件能够作为工具进行集成。AutoDefense可以适应各种大小和类型的开源LLM作为代理。通过对大量有害和安全提示的大量实验，验证了本文提出的防御法在保持正常用户请求性能的同时，提高了对越狱攻击的健壮性。我们的代码和数据在https://github.com/XHMY/AutoDefense.上公开提供



## **25. Accelerating Greedy Coordinate Gradient via Probe Sampling**

利用探头采样加速贪婪坐标梯度 cs.CL

**SubmitDate**: 2024-03-02    [abs](http://arxiv.org/abs/2403.01251v1) [paper-pdf](http://arxiv.org/pdf/2403.01251v1)

**Authors**: Yiran Zhao, Wenyue Zheng, Tianle Cai, Xuan Long Do, Kenji Kawaguchi, Anirudh Goyal, Michael Shieh

**Abstract**: Safety of Large Language Models (LLMs) has become a central issue given their rapid progress and wide applications. Greedy Coordinate Gradient (GCG) is shown to be effective in constructing prompts containing adversarial suffixes to break the presumingly safe LLMs, but the optimization of GCG is time-consuming and limits its practicality. To reduce the time cost of GCG and enable more comprehensive studies of LLM safety, in this work, we study a new algorithm called $\texttt{Probe sampling}$ to accelerate the GCG algorithm. At the core of the algorithm is a mechanism that dynamically determines how similar a smaller draft model's predictions are to the target model's predictions for prompt candidates. When the target model is similar to the draft model, we rely heavily on the draft model to filter out a large number of potential prompt candidates to reduce the computation time. Probe sampling achieves up to $5.6$ times speedup using Llama2-7b and leads to equal or improved attack success rate (ASR) on the AdvBench.

摘要: 由于大型语言模型的快速发展和广泛应用，其安全性已成为一个核心问题。贪婪坐标梯度(GCG)被证明能有效地构造含有敌意后缀的提示，以打破假定安全的LLMS，但GCG的优化耗时长，限制了其实用性。为了减少GCG算法的时间开销，更全面地研究LLM的安全性，本文研究了一种新的加速GCG算法的算法该算法的核心是一种机制，该机制动态地确定较小的草稿模型的预测与目标模型对提示候选人的预测的相似性程度。当目标模型与选秀模型相似时，我们严重依赖选秀模型来筛选出大量潜在的提示候选者，以减少计算时间。使用Llama2-7b，探头采样可获得高达5.6美元的加速比，并可带来同等或更高的AdvBch攻击成功率(ASR)。



## **26. Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy**

不准确的遗忘需要更仔细的评估以避免错误的隐私感 cs.LG

**SubmitDate**: 2024-03-02    [abs](http://arxiv.org/abs/2403.01218v1) [paper-pdf](http://arxiv.org/pdf/2403.01218v1)

**Authors**: Jamie Hayes, Ilia Shumailov, Eleni Triantafillou, Amr Khalifa, Nicolas Papernot

**Abstract**: The high cost of model training makes it increasingly desirable to develop techniques for unlearning. These techniques seek to remove the influence of a training example without having to retrain the model from scratch. Intuitively, once a model has unlearned, an adversary that interacts with the model should no longer be able to tell whether the unlearned example was included in the model's training set or not. In the privacy literature, this is known as membership inference. In this work, we discuss adaptations of Membership Inference Attacks (MIAs) to the setting of unlearning (leading to their ``U-MIA'' counterparts). We propose a categorization of existing U-MIAs into ``population U-MIAs'', where the same attacker is instantiated for all examples, and ``per-example U-MIAs'', where a dedicated attacker is instantiated for each example. We show that the latter category, wherein the attacker tailors its membership prediction to each example under attack, is significantly stronger. Indeed, our results show that the commonly used U-MIAs in the unlearning literature overestimate the privacy protection afforded by existing unlearning techniques on both vision and language models. Our investigation reveals a large variance in the vulnerability of different examples to per-example U-MIAs. In fact, several unlearning algorithms lead to a reduced vulnerability for some, but not all, examples that we wish to unlearn, at the expense of increasing it for other examples. Notably, we find that the privacy protection for the remaining training examples may worsen as a consequence of unlearning. We also discuss the fundamental difficulty of equally protecting all examples using existing unlearning schemes, due to the different rates at which examples are unlearned. We demonstrate that naive attempts at tailoring unlearning stopping criteria to different examples fail to alleviate these issues.

摘要: 模型训练的高昂成本使得开发忘却学习的技术变得越来越受欢迎。这些技术寻求消除训练示例的影响，而不必从头开始重新训练模型。直观地说，一旦模型取消学习，与该模型交互的对手应该不再能够判断未学习的示例是否包括在该模型的训练集中。在隐私文献中，这被称为成员关系推断。在这项工作中，我们讨论了成员关系推理攻击(MIA)对遗忘环境的适应(导致它们的‘U-MIA’对应)。我们提出了一种现有U-MIA的分类，其中针对所有示例实例化相同的攻击者，其中针对每个示例实例化一个专用攻击者。我们表明，后一类，其中攻击者根据每个被攻击的例子定制其成员预测，明显更强。事实上，我们的结果表明，遗忘文献中常用的U-MIA高估了现有遗忘技术在视觉和语言模型上提供的隐私保护。我们的调查显示，不同示例对每个示例的U-MIA的脆弱性存在很大差异。事实上，几种忘记算法降低了我们希望忘记的一些(但不是所有)示例的脆弱性，但代价是增加了其他示例的脆弱性。值得注意的是，我们发现，由于遗忘，其余训练样本的隐私保护可能会恶化。我们还讨论了使用现有的遗忘方案平等地保护所有例子的基本困难，因为例子被遗忘的比率不同。我们证明，根据不同的例子调整遗忘停止标准的天真尝试无法缓解这些问题。



## **27. Prompt Injection attack against LLM-integrated Applications**

针对集成了LLM的应用程序的快速注入攻击 cs.CR

**SubmitDate**: 2024-03-02    [abs](http://arxiv.org/abs/2306.05499v2) [paper-pdf](http://arxiv.org/pdf/2306.05499v2)

**Authors**: Yi Liu, Gelei Deng, Yuekang Li, Kailong Wang, Zihao Wang, Xiaofeng Wang, Tianwei Zhang, Yepang Liu, Haoyu Wang, Yan Zheng, Yang Liu

**Abstract**: Large Language Models (LLMs), renowned for their superior proficiency in language comprehension and generation, stimulate a vibrant ecosystem of applications around them. However, their extensive assimilation into various services introduces significant security risks. This study deconstructs the complexities and implications of prompt injection attacks on actual LLM-integrated applications. Initially, we conduct an exploratory analysis on ten commercial applications, highlighting the constraints of current attack strategies in practice. Prompted by these limitations, we subsequently formulate HouYi, a novel black-box prompt injection attack technique, which draws inspiration from traditional web injection attacks. HouYi is compartmentalized into three crucial elements: a seamlessly-incorporated pre-constructed prompt, an injection prompt inducing context partition, and a malicious payload designed to fulfill the attack objectives. Leveraging HouYi, we unveil previously unknown and severe attack outcomes, such as unrestricted arbitrary LLM usage and uncomplicated application prompt theft. We deploy HouYi on 36 actual LLM-integrated applications and discern 31 applications susceptible to prompt injection. 10 vendors have validated our discoveries, including Notion, which has the potential to impact millions of users. Our investigation illuminates both the possible risks of prompt injection attacks and the possible tactics for mitigation.

摘要: 大型语言模型(LLM)以其在语言理解和生成方面的卓越熟练程度而闻名，它们刺激了周围充满活力的应用生态系统。然而，它们广泛融入各种服务带来了重大的安全风险。这项研究解构了即时注入攻击对实际LLM集成应用程序的复杂性和影响。首先，我们对十个商业应用进行了探索性分析，突出了当前攻击策略在实践中的限制。在这些局限性的驱使下，我们从传统的网络注入攻击中得到了启发，提出了一种新颖的黑盒提示注入攻击技术--后一种。后易被划分为三个关键元素：无缝结合的预先构建的提示、引发上下文划分的注入提示和旨在实现攻击目标的恶意有效负载。利用后一，我们揭示了以前未知和严重的攻击结果，例如不受限制的任意LLM使用和简单的应用提示盗窃。我们在36个实际的LLM集成应用程序上部署了后易，并识别了31个易受快速注入影响的应用程序。已有10家供应商验证了我们的发现，其中包括可能影响数百万用户的概念。我们的调查说明了迅速注射攻击的可能风险和可能的缓解策略。



## **28. Knowledge Sanitization of Large Language Models**

大型语言模型的知识清洗 cs.CL

**SubmitDate**: 2024-03-02    [abs](http://arxiv.org/abs/2309.11852v2) [paper-pdf](http://arxiv.org/pdf/2309.11852v2)

**Authors**: Yoichi Ishibashi, Hidetoshi Shimodaira

**Abstract**: We explore a knowledge sanitization approach to mitigate the privacy concerns associated with large language models (LLMs). LLMs trained on a large corpus of Web data can memorize and potentially reveal sensitive or confidential information, raising critical security concerns. Our technique efficiently fine-tunes these models using the Low-Rank Adaptation (LoRA) method, prompting them to generate harmless responses such as ``I don't know'' when queried about specific information. Experimental results in a closed-book question-answering task show that our straightforward method not only minimizes particular knowledge leakage but also preserves the overall performance of LLMs. These two advantages strengthen the defense against extraction attacks and reduces the emission of harmful content such as hallucinations.

摘要: 我们探索了一种知识净化方法来缓解与大型语言模型(LLM)相关的隐私问题。在大型网络数据语料库上接受培训的LLM可能会记住并可能泄露敏感或机密信息，从而引发关键的安全问题。我们的技术使用低排名适应(LORA)方法有效地微调这些模型，促使它们在被问及特定信息时产生无害的反应，如“我不知道”。在闭卷问答任务中的实验结果表明，该方法不仅最大限度地减少了特定的知识泄漏，而且保持了LLMS的整体性能。这两个优势加强了对提取攻击的防御，并减少了幻觉等有害内容的排放。



## **29. AutoAttacker: A Large Language Model Guided System to Implement Automatic Cyber-attacks**

AutoAttacker：一个实现自动网络攻击的大型语言模型制导系统 cs.CR

**SubmitDate**: 2024-03-02    [abs](http://arxiv.org/abs/2403.01038v1) [paper-pdf](http://arxiv.org/pdf/2403.01038v1)

**Authors**: Jiacen Xu, Jack W. Stokes, Geoff McDonald, Xuesong Bai, David Marshall, Siyue Wang, Adith Swaminathan, Zhou Li

**Abstract**: Large language models (LLMs) have demonstrated impressive results on natural language tasks, and security researchers are beginning to employ them in both offensive and defensive systems. In cyber-security, there have been multiple research efforts that utilize LLMs focusing on the pre-breach stage of attacks like phishing and malware generation. However, so far there lacks a comprehensive study regarding whether LLM-based systems can be leveraged to simulate the post-breach stage of attacks that are typically human-operated, or "hands-on-keyboard" attacks, under various attack techniques and environments.   As LLMs inevitably advance, they may be able to automate both the pre- and post-breach attack stages. This shift may transform organizational attacks from rare, expert-led events to frequent, automated operations requiring no expertise and executed at automation speed and scale. This risks fundamentally changing global computer security and correspondingly causing substantial economic impacts, and a goal of this work is to better understand these risks now so we can better prepare for these inevitable ever-more-capable LLMs on the horizon. On the immediate impact side, this research serves three purposes. First, an automated LLM-based, post-breach exploitation framework can help analysts quickly test and continually improve their organization's network security posture against previously unseen attacks. Second, an LLM-based penetration test system can extend the effectiveness of red teams with a limited number of human analysts. Finally, this research can help defensive systems and teams learn to detect novel attack behaviors preemptively before their use in the wild....

摘要: 大型语言模型(LLM)在自然语言任务上已经显示出令人印象深刻的结果，安全研究人员正开始在攻防系统中使用它们。在网络安全方面，已经有多项研究工作利用LLMS，重点放在攻击的入侵前阶段，如网络钓鱼和恶意软件生成。然而，到目前为止，对于基于LLM的系统是否可以被用来模拟在各种攻击技术和环境下的攻击的后攻击阶段，目前还缺乏全面的研究，这些攻击通常是人为操作的，或在键盘上进行的攻击。随着LLMS不可避免地向前发展，它们可能能够实现攻击前和攻击后阶段的自动化。这种转变可能会将组织攻击从罕见的专家主导的事件转变为频繁的自动化操作，不需要专业知识，并以自动化的速度和规模执行。这有可能从根本上改变全球计算机安全，并相应地造成重大的经济影响，这项工作的目标之一是现在更好地了解这些风险，以便我们能够更好地为即将到来的这些不可避免的、能力更强的低成本管理做好准备。在立竿见影的影响方面，这项研究有三个目的。首先，基于LLM的自动化入侵后利用框架可以帮助分析人员快速测试并持续改进其组织的网络安全态势，以抵御以前未见过的攻击。其次，基于LLM的渗透测试系统可以在有限数量的人类分析师的情况下扩展RED团队的有效性。最后，这项研究可以帮助防御系统和团队学习在将新的攻击行为用于野外之前先发制人地检测它们。



## **30. DrAttack: Prompt Decomposition and Reconstruction Makes Powerful LLM Jailbreakers**

DrAttack：快速分解和重构使LLM成为强大的越狱者 cs.CR

**SubmitDate**: 2024-03-01    [abs](http://arxiv.org/abs/2402.16914v2) [paper-pdf](http://arxiv.org/pdf/2402.16914v2)

**Authors**: Xirui Li, Ruochen Wang, Minhao Cheng, Tianyi Zhou, Cho-Jui Hsieh

**Abstract**: The safety alignment of Large Language Models (LLMs) is vulnerable to both manual and automated jailbreak attacks, which adversarially trigger LLMs to output harmful content. However, current methods for jailbreaking LLMs, which nest entire harmful prompts, are not effective at concealing malicious intent and can be easily identified and rejected by well-aligned LLMs. This paper discovers that decomposing a malicious prompt into separated sub-prompts can effectively obscure its underlying malicious intent by presenting it in a fragmented, less detectable form, thereby addressing these limitations. We introduce an automatic prompt \textbf{D}ecomposition and \textbf{R}econstruction framework for jailbreak \textbf{Attack} (DrAttack). DrAttack includes three key components: (a) `Decomposition' of the original prompt into sub-prompts, (b) `Reconstruction' of these sub-prompts implicitly by in-context learning with semantically similar but harmless reassembling demo, and (c) a `Synonym Search' of sub-prompts, aiming to find sub-prompts' synonyms that maintain the original intent while jailbreaking LLMs. An extensive empirical study across multiple open-source and closed-source LLMs demonstrates that, with a significantly reduced number of queries, DrAttack obtains a substantial gain of success rate over prior SOTA prompt-only attackers. Notably, the success rate of 78.0\% on GPT-4 with merely 15 queries surpassed previous art by 33.1\%. The project is available at https://github.com/xirui-li/DrAttack.

摘要: 大型语言模型(LLM)的安全一致性容易受到手动和自动越狱攻击的攻击，这些攻击会相反地触发LLM输出有害内容。然而，当前的越狱LLM方法嵌套了整个有害的提示，在隐藏恶意意图方面并不有效，很容易被排列良好的LLM识别和拒绝。本文发现，通过将恶意提示分解为单独的子提示，可以通过以零散的、较难检测的形式来表示恶意提示，从而有效地掩盖潜在的恶意意图，从而解决这些限制。介绍了一种自动提示的文本bf{D}分解和文本bf{R}重构框架(DrAttack)。DrAttack包括三个关键组件：(A)将原始提示‘分解’成子提示，(B)通过使用语义相似但无害的重组演示的上下文学习隐式地‘重构’这些子提示，以及(C)对子提示进行‘同步搜索’，目的是在越狱LLM的同时找到保持原意的子提示的同义词。一项针对多个开源和闭源LLM的广泛经验研究表明，DrAttack在查询次数显著减少的情况下，与之前的Sota仅提示攻击者相比，获得了显著的成功率。值得注意的是，仅用15个查询在GPT-4上的成功率为78.0\%，比以前的ART高出33.1\%。该项目的网址为：https://github.com/xirui-li/DrAttack.。



## **31. Teach LLMs to Phish: Stealing Private Information from Language Models**

教LLMS学会钓鱼：从语言模型中窃取私人信息 cs.CR

ICLR 2024

**SubmitDate**: 2024-03-01    [abs](http://arxiv.org/abs/2403.00871v1) [paper-pdf](http://arxiv.org/pdf/2403.00871v1)

**Authors**: Ashwinee Panda, Christopher A. Choquette-Choo, Zhengming Zhang, Yaoqing Yang, Prateek Mittal

**Abstract**: When large language models are trained on private data, it can be a significant privacy risk for them to memorize and regurgitate sensitive information. In this work, we propose a new practical data extraction attack that we call "neural phishing". This attack enables an adversary to target and extract sensitive or personally identifiable information (PII), e.g., credit card numbers, from a model trained on user data with upwards of 10% attack success rates, at times, as high as 50%. Our attack assumes only that an adversary can insert as few as 10s of benign-appearing sentences into the training dataset using only vague priors on the structure of the user data.

摘要: 当大型语言模型接受关于私人数据的培训时，对他们来说，记忆和反胃敏感信息可能是一个重大的隐私风险。在这项工作中，我们提出了一种新的实用的数据提取攻击，我们称之为“神经钓鱼”。这种攻击使对手能够将敏感或个人可识别信息(PII)作为目标，并从根据用户数据训练的模型中提取敏感或个人身份信息(PII)，攻击成功率有时高达10%，有时高达50%。我们的攻击只假设对手只需在用户数据结构上使用模糊的先验就可以在训练数据集中插入少至10个看起来温和的句子。



## **32. Here's a Free Lunch: Sanitizing Backdoored Models with Model Merge**

这是一顿免费午餐：用模型合并来清理过时的模型 cs.CL

work in progress

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2402.19334v1) [paper-pdf](http://arxiv.org/pdf/2402.19334v1)

**Authors**: Ansh Arora, Xuanli He, Maximilian Mozes, Srinibas Swain, Mark Dras, Qiongkai Xu

**Abstract**: The democratization of pre-trained language models through open-source initiatives has rapidly advanced innovation and expanded access to cutting-edge technologies. However, this openness also brings significant security risks, including backdoor attacks, where hidden malicious behaviors are triggered by specific inputs, compromising natural language processing (NLP) system integrity and reliability. This paper suggests that merging a backdoored model with other homogeneous models can remediate backdoor vulnerabilities even if such models are not entirely secure. In our experiments, we explore various models (BERT-Base, RoBERTa-Large, Llama2-7B, and Mistral-7B) and datasets (SST-2, OLID, AG News, and QNLI). Compared to multiple advanced defensive approaches, our method offers an effective and efficient inference-stage defense against backdoor attacks without additional resources or specific knowledge. Our approach consistently outperforms the other advanced baselines, leading to an average of 75% reduction in the attack success rate. Since model merging has been an established approach for improving model performance, the extra advantage it provides regarding defense can be seen as a cost-free bonus.

摘要: 通过开放源码倡议使预先培训的语言模型民主化，迅速推动了创新，扩大了获得尖端技术的机会。然而，这种开放性也带来了重大的安全风险，包括后门攻击，其中隐藏的恶意行为由特定的输入触发，损害了自然语言处理(NLP)系统的完整性和可靠性。本文认为，将后门模型与其他同类模型合并可以补救后门漏洞，即使这些模型不是完全安全的。在我们的实验中，我们探索了各种模型(Bert-Base、Roberta-Large、Llama2-7B和Mistral-7B)和数据集(SST-2、OLID、AG News和QNLI)。与多种先进的防御方法相比，我们的方法提供了一种有效和高效的推理阶段防御后门攻击，而不需要额外的资源或特定的知识。我们的方法始终优于其他先进的基准，导致攻击成功率平均降低75%。由于模型合并已经成为提高模型性能的既定方法，它提供的关于防御的额外优势可以被视为免费的额外奖励。



## **33. PRSA: Prompt Reverse Stealing Attacks against Large Language Models**

PRSA：针对大型语言模型的快速反向窃取攻击 cs.CR

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2402.19200v1) [paper-pdf](http://arxiv.org/pdf/2402.19200v1)

**Authors**: Yong Yang, Xuhong Zhang, Yi Jiang, Xi Chen, Haoyu Wang, Shouling Ji, Zonghui Wang

**Abstract**: Prompt, recognized as crucial intellectual property, enables large language models (LLMs) to perform specific tasks without the need of fine-tuning, underscoring their escalating importance. With the rise of prompt-based services, such as prompt marketplaces and LLM applications, providers often display prompts' capabilities through input-output examples to attract users. However, this paradigm raises a pivotal security concern: does the exposure of input-output pairs pose the risk of potential prompt leakage, infringing on the intellectual property rights of the developers? To our knowledge, this problem still has not been comprehensively explored yet. To remedy this gap, in this paper, we perform the first in depth exploration and propose a novel attack framework for reverse-stealing prompts against commercial LLMs, namely PRSA. The main idea of PRSA is that by analyzing the critical features of the input-output pairs, we mimic and gradually infer (steal) the target prompts. In detail, PRSA mainly consists of two key phases: prompt mutation and prompt pruning. In the mutation phase, we propose a prompt attention algorithm based on differential feedback to capture these critical features for effectively inferring the target prompts. In the prompt pruning phase, we identify and mask the words dependent on specific inputs, enabling the prompts to accommodate diverse inputs for generalization. Through extensive evaluation, we verify that PRSA poses a severe threat in real world scenarios. We have reported these findings to prompt service providers and actively collaborate with them to take protective measures for prompt copyright.

摘要: Prompt被认为是重要的知识产权，使大型语言模型(LLM)能够在不需要微调的情况下执行特定任务，突显了它们日益增长的重要性。随着基于提示的服务的兴起，如提示市场和LLM应用程序，提供商经常通过输入输出示例来展示提示的能力，以吸引用户。然而，这种模式引发了一个关键的安全问题：输入-输出对的暴露是否构成了潜在的即时泄漏的风险，从而侵犯了开发人员的知识产权？据我们所知，这个问题还没有得到全面的探索。为了弥补这一缺陷，本文首次深入研究并提出了一种新的针对商业LLM的反向窃取提示攻击框架，即PRSA。PRSA的主要思想是通过分析输入输出对的关键特征，模仿并逐步推断(窃取)目标提示。具体而言，PRSA算法主要由两个关键阶段组成：快速变异和快速剪枝。在突变阶段，我们提出了一种基于差分反馈的提示注意算法，以捕捉这些关键特征，从而有效地推断目标提示。在提示修剪阶段，我们识别并掩蔽依赖于特定输入的单词，使提示能够适应不同的输入以进行泛化。通过广泛的评估，我们验证了PRSA在现实世界场景中构成了严重的威胁。我们已将这些发现报告给服务提供商，并积极与他们合作，采取保护措施，及时获得版权。



## **34. A Semantic Invariant Robust Watermark for Large Language Models**

一种面向大型语言模型的语义不变鲁棒水印 cs.CR

ICLR2024, 21 pages, 10 figures, 6 tables

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2310.06356v2) [paper-pdf](http://arxiv.org/pdf/2310.06356v2)

**Authors**: Aiwei Liu, Leyi Pan, Xuming Hu, Shiao Meng, Lijie Wen

**Abstract**: Watermark algorithms for large language models (LLMs) have achieved extremely high accuracy in detecting text generated by LLMs. Such algorithms typically involve adding extra watermark logits to the LLM's logits at each generation step. However, prior algorithms face a trade-off between attack robustness and security robustness. This is because the watermark logits for a token are determined by a certain number of preceding tokens; a small number leads to low security robustness, while a large number results in insufficient attack robustness. In this work, we propose a semantic invariant watermarking method for LLMs that provides both attack robustness and security robustness. The watermark logits in our work are determined by the semantics of all preceding tokens. Specifically, we utilize another embedding LLM to generate semantic embeddings for all preceding tokens, and then these semantic embeddings are transformed into the watermark logits through our trained watermark model. Subsequent analyses and experiments demonstrated the attack robustness of our method in semantically invariant settings: synonym substitution and text paraphrasing settings. Finally, we also show that our watermark possesses adequate security robustness. Our code and data are available at https://github.com/THU-BPM/Robust_Watermark.

摘要: 针对大语言模型的水印算法在检测大语言模型生成的文本方面取得了极高的准确率。这类算法通常涉及在每个生成步骤向LLM的日志添加额外的水印日志。然而，现有的算法面临着攻击健壮性和安全健壮性之间的权衡。这是因为令牌的水印登录由一定数量的先前令牌确定；较小的数字会导致较低的安全稳健性，而较大的数字会导致攻击稳健性不足。在这项工作中，我们提出了一种既具有攻击健壮性又具有安全健壮性的LLMS语义不变水印方法。我们工作中的水印日志是由前面所有令牌的语义确定的。具体地说，我们利用另一种嵌入LLM为所有前面的令牌生成语义嵌入，然后通过我们训练的水印模型将这些语义嵌入转换成水印日志。随后的分析和实验证明了该方法在同义词替换和文本释义等语义不变环境下的攻击健壮性。最后，我们还证明了我们的水印具有足够的安全稳健性。我们的代码和数据可在https://github.com/THU-BPM/Robust_Watermark.上获得



## **35. Typographic Attacks in Large Multimodal Models Can be Alleviated by More Informative Prompts**

大型多模式模型中的排版攻击可以通过提供更多信息的提示来缓解 cs.CV

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2402.19150v1) [paper-pdf](http://arxiv.org/pdf/2402.19150v1)

**Authors**: Hao Cheng, Erjia Xiao, Renjing Xu

**Abstract**: Large Multimodal Models (LMMs) rely on pre-trained Vision Language Models (VLMs) and Large Language Models (LLMs) to perform amazing emergent abilities on various multimodal tasks in the joint space of vision and language. However, the Typographic Attack, which shows disruption to VLMs, has also been certified as a security vulnerability to LMMs. In this work, we first comprehensively investigate the distractibility of LMMs by typography. In particular, we introduce the Typographic Dataset designed to evaluate distractibility across various multi-modal subtasks, such as object recognition, visual attributes detection, enumeration, arithmetic computation, and commonsense reasoning. To further study the effect of typographic patterns on performance, we also scrutinize the effect of tuning various typographic factors, encompassing font size, color, opacity, and spatial positioning of typos. We discover that LMMs can partially distinguish visual contents and typos when confronting typographic attacks, which suggests that embeddings from vision encoders contain enough information to distinguish visual contents and typos in images. Inspired by such phenomena, we demonstrate that CLIP's performance of zero-shot classification on typo-ridden images can be significantly improved by providing more informative texts to match images. Furthermore, we also prove that LMMs can utilize more informative prompts to leverage information in embeddings to differentiate between visual content and typos. Finally, we propose a prompt information enhancement method that can effectively mitigate the effects of typography.

摘要: 大型多通道模型(LMM)依靠预先训练好的视觉语言模型(VLM)和大语言模型(LLM)在视觉和语言的联合空间中执行各种多通道任务，具有惊人的应急能力。然而，字体攻击显示了对VLM的破坏，也被证明是LMM的一个安全漏洞。在这项工作中，我们首先通过排版来全面地研究LMM的分心问题。特别是，我们介绍了排版数据集，旨在评估各种多模式子任务的分心能力，如对象识别、视觉属性检测、枚举、算术计算和常识推理。为了进一步研究印刷模式对性能的影响，我们还仔细研究了调整各种印刷因素的影响，包括字体大小、颜色、不透明度和印刷错误的空间位置。我们发现，LMM在面对印刷攻击时可以部分区分视觉内容和错别字，这表明来自视觉编码器的嵌入包含了足够的信息来区分图像中的视觉内容和错别字。受这些现象的启发，我们证明了通过提供更多的信息量的文本来匹配图像，可以显著提高CLIP在拼写错误的图像上的零镜头分类性能。此外，我们还证明了LMM可以利用更多的信息提示来利用嵌入中的信息来区分可视内容和错别字。最后，我们提出了一种能够有效缓解排版影响的即时信息增强方法。



## **36. Cognitive Overload: Jailbreaking Large Language Models with Overloaded Logical Thinking**

认知超载：用超负荷的逻辑思维越狱的大型语言模型 cs.CL

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2311.09827v2) [paper-pdf](http://arxiv.org/pdf/2311.09827v2)

**Authors**: Nan Xu, Fei Wang, Ben Zhou, Bang Zheng Li, Chaowei Xiao, Muhao Chen

**Abstract**: While large language models (LLMs) have demonstrated increasing power, they have also given rise to a wide range of harmful behaviors. As representatives, jailbreak attacks can provoke harmful or unethical responses from LLMs, even after safety alignment. In this paper, we investigate a novel category of jailbreak attacks specifically designed to target the cognitive structure and processes of LLMs. Specifically, we analyze the safety vulnerability of LLMs in the face of (1) multilingual cognitive overload, (2) veiled expression, and (3) effect-to-cause reasoning. Different from previous jailbreak attacks, our proposed cognitive overload is a black-box attack with no need for knowledge of model architecture or access to model weights. Experiments conducted on AdvBench and MasterKey reveal that various LLMs, including both popular open-source model Llama 2 and the proprietary model ChatGPT, can be compromised through cognitive overload. Motivated by cognitive psychology work on managing cognitive load, we further investigate defending cognitive overload attack from two perspectives. Empirical studies show that our cognitive overload from three perspectives can jailbreak all studied LLMs successfully, while existing defense strategies can hardly mitigate the caused malicious uses effectively.

摘要: 虽然大型语言模型(LLM)显示出越来越大的力量，但它们也引发了广泛的有害行为。作为代表，越狱攻击可能会引发低收入国家的有害或不道德的反应，即使在安全调整之后也是如此。在本文中，我们研究了一类新的越狱攻击，该攻击专门针对LLMS的认知结构和过程而设计。具体地说，我们分析了在(1)多语言认知过载、(2)含蓄表达和(3)因果推理的情况下，LLMS的安全脆弱性。与以前的越狱攻击不同，我们提出的认知过载攻击是一种黑盒攻击，不需要了解模型体系结构或访问模型权重。在AdvBtch和MasterKey上进行的实验表明，各种LLM，包括流行的开源模型Llama 2和专有模型ChatGPT，都可以通过认知过载而受到损害。受认知心理学关于管理认知负荷的研究的启发，我们从两个角度进一步研究了防御认知过载攻击。实证研究表明，我们从三个角度的认知过载可以成功地越狱所有研究的LLM，而现有的防御策略很难有效地缓解造成的恶意使用。



## **37. Vaccine: Perturbation-aware Alignment for Large Language Model**

疫苗：大型语言模型中的扰动感知比对 cs.LG

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2402.01109v3) [paper-pdf](http://arxiv.org/pdf/2402.01109v3)

**Authors**: Tiansheng Huang, Sihao Hu, Ling Liu

**Abstract**: The new paradigm of finetuning-as-a-service introduces a new attack surface for Large Language Models (LLMs): a few harmful data uploaded by users can easily trick the finetuning to produce an alignment-broken model. We conduct an empirical analysis and uncover a \textit{harmful embedding drift} phenomenon, showing a probable cause of the alignment-broken effect. Inspired by our findings, we propose Vaccine, a perturbation-aware alignment technique to mitigate the security risk of users finetuning. The core idea of Vaccine is to produce invariant hidden embeddings by progressively adding crafted perturbation to them in the alignment phase. This enables the embeddings to withstand harmful perturbation from un-sanitized user data in the finetuning phase. Our results on open source mainstream LLMs (e.g., Llama2, Opt, Vicuna) demonstrate that Vaccine can boost the robustness of alignment against harmful prompts induced embedding drift while reserving reasoning ability towards benign prompts. Our code is available at \url{https://github.com/git-disl/Vaccine}.

摘要: 精调即服务的新范式为大型语言模型(LLM)引入了一个新的攻击面：用户上传的少量有害数据就可以很容易地欺骗精调，产生一个破坏对齐的模型。我们进行了实证分析，发现了一种有害的嵌入漂移现象，揭示了排列断裂效应的可能原因。受我们发现的启发，我们提出了Vaccine，一种扰动感知的对齐技术，以降低用户精调的安全风险。Vaccine的核心思想是通过在比对阶段逐步向其添加精心制作的扰动来产生不变的隐藏嵌入。这使嵌入能够在精细调整阶段抵御来自未清理的用户数据的有害干扰。我们在开源主流LLMS(如Llama2、Opt、Vicuna)上的实验结果表明，疫苗可以提高对有害提示导致的嵌入漂移的健壮性，同时保留对良性提示的推理能力。我们的代码可在\url{https://github.com/git-disl/Vaccine}.



## **38. Defending Large Language Models against Jailbreak Attacks via Semantic Smoothing**

通过语义平滑保护大型语言模型免受越狱攻击 cs.CL

37 pages

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.16192v2) [paper-pdf](http://arxiv.org/pdf/2402.16192v2)

**Authors**: Jiabao Ji, Bairu Hou, Alexander Robey, George J. Pappas, Hamed Hassani, Yang Zhang, Eric Wong, Shiyu Chang

**Abstract**: Aligned large language models (LLMs) are vulnerable to jailbreaking attacks, which bypass the safeguards of targeted LLMs and fool them into generating objectionable content. While initial defenses show promise against token-based threat models, there do not exist defenses that provide robustness against semantic attacks and avoid unfavorable trade-offs between robustness and nominal performance. To meet this need, we propose SEMANTICSMOOTH, a smoothing-based defense that aggregates the predictions of multiple semantically transformed copies of a given input prompt. Experimental results demonstrate that SEMANTICSMOOTH achieves state-of-the-art robustness against GCG, PAIR, and AutoDAN attacks while maintaining strong nominal performance on instruction following benchmarks such as InstructionFollowing and AlpacaEval. The codes will be publicly available at https://github.com/UCSB-NLP-Chang/SemanticSmooth.

摘要: 对齐的大型语言模型(LLM)容易受到越狱攻击，这些攻击绕过目标LLM的保护措施，欺骗它们生成令人反感的内容。虽然针对基于令牌的威胁模型的初始防御很有希望，但不存在针对语义攻击提供稳健性并避免稳健性和名义性能之间的不利权衡的防御。为了满足这一需求，我们提出了SEMANTICSMOOTH，这是一种基于平滑的防御方法，它聚合了给定输入提示的多个语义转换副本的预测。实验结果表明，SEMANTICSMOOTH在抵抗GCG、Pair和AutoDAN攻击的同时，在遵循InstructionFollowing和AlpacaEval等基准测试的指令上保持了很强的名义性能。这些代码将在https://github.com/UCSB-NLP-Chang/SemanticSmooth.上公开提供



## **39. Defending LLMs against Jailbreaking Attacks via Backtranslation**

通过反向翻译保护LLMS免受越狱攻击 cs.CL

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.16459v2) [paper-pdf](http://arxiv.org/pdf/2402.16459v2)

**Authors**: Yihan Wang, Zhouxing Shi, Andrew Bai, Cho-Jui Hsieh

**Abstract**: Although many large language models (LLMs) have been trained to refuse harmful requests, they are still vulnerable to jailbreaking attacks, which rewrite the original prompt to conceal its harmful intent. In this paper, we propose a new method for defending LLMs against jailbreaking attacks by ``backtranslation''. Specifically, given an initial response generated by the target LLM from an input prompt, our backtranslation prompts a language model to infer an input prompt that can lead to the response. The inferred prompt is called the backtranslated prompt which tends to reveal the actual intent of the original prompt, since it is generated based on the LLM's response and is not directly manipulated by the attacker. We then run the target LLM again on the backtranslated prompt, and we refuse the original prompt if the model refuses the backtranslated prompt. We explain that the proposed defense provides several benefits on its effectiveness and efficiency. We empirically demonstrate that our defense significantly outperforms the baselines, in the cases that are hard for the baselines, and our defense also has little impact on the generation quality for benign input prompts.

摘要: 尽管许多大型语言模型(LLM)已经接受了拒绝有害请求的培训，但他们仍然容易受到越狱攻击，这些攻击会重写原始提示以掩盖其有害意图。在本文中，我们提出了一种新的方法来防御‘’反向翻译‘’越狱攻击。具体地说，给定目标LLM从输入提示生成的初始响应，我们的反向翻译会提示语言模型推断出可能导致该响应的输入提示。推断的提示称为反向翻译提示，它往往会揭示原始提示的实际意图，因为它是基于LLM的响应生成的，而不是由攻击者直接操纵的。然后，我们在回译的提示上再次运行目标LLM，如果模型拒绝回译的提示，我们将拒绝原始提示。我们解释说，拟议的辩护在其有效性和效率方面提供了几个好处。我们的经验证明，我们的防御显著优于基线，在基线难以达到的情况下，我们的防御对良性输入提示的生成质量也几乎没有影响。



## **40. A New Era in LLM Security: Exploring Security Concerns in Real-World LLM-based Systems**

LLM安全的新时代：探索现实世界中基于LLM的系统的安全问题 cs.CR

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.18649v1) [paper-pdf](http://arxiv.org/pdf/2402.18649v1)

**Authors**: Fangzhou Wu, Ning Zhang, Somesh Jha, Patrick McDaniel, Chaowei Xiao

**Abstract**: Large Language Model (LLM) systems are inherently compositional, with individual LLM serving as the core foundation with additional layers of objects such as plugins, sandbox, and so on. Along with the great potential, there are also increasing concerns over the security of such probabilistic intelligent systems. However, existing studies on LLM security often focus on individual LLM, but without examining the ecosystem through the lens of LLM systems with other objects (e.g., Frontend, Webtool, Sandbox, and so on). In this paper, we systematically analyze the security of LLM systems, instead of focusing on the individual LLMs. To do so, we build on top of the information flow and formulate the security of LLM systems as constraints on the alignment of the information flow within LLM and between LLM and other objects. Based on this construction and the unique probabilistic nature of LLM, the attack surface of the LLM system can be decomposed into three key components: (1) multi-layer security analysis, (2) analysis of the existence of constraints, and (3) analysis of the robustness of these constraints. To ground this new attack surface, we propose a multi-layer and multi-step approach and apply it to the state-of-art LLM system, OpenAI GPT4. Our investigation exposes several security issues, not just within the LLM model itself but also in its integration with other components. We found that although the OpenAI GPT4 has designed numerous safety constraints to improve its safety features, these safety constraints are still vulnerable to attackers. To further demonstrate the real-world threats of our discovered vulnerabilities, we construct an end-to-end attack where an adversary can illicitly acquire the user's chat history, all without the need to manipulate the user's input or gain direct access to OpenAI GPT4. Our demo is in the link: https://fzwark.github.io/LLM-System-Attack-Demo/

摘要: 大型语言模型(LLM)系统本质上是组合的，单个LLM充当核心基础，具有附加的对象层，如插件、沙箱等。在这种巨大潜力的同时，人们对这种概率智能系统的安全性也越来越关注。然而，现有的关于LLM安全的研究往往集中在单个LLM上，而没有通过LLM系统与其他对象(如前端、WebTool、沙盒等)的透镜来考察生态系统。在本文中，我们系统地分析了LLM系统的安全性，而不是关注单个LLM。为此，我们建立在信息流的基础上，并将LLM系统的安全性制定为对LLM内以及LLM与其他对象之间的信息流对齐的约束。基于这种构造和LLM的独特概率性质，LLM系统的攻击面可以分解为三个关键部分：(1)多层安全分析，(2)约束的存在性分析，(3)这些约束的稳健性分析。为了对这种新的攻击面进行接地，我们提出了一种多层多步骤的方法，并将其应用于最先进的LLM系统OpenAI GPT4。我们的调查暴露了几个安全问题，不仅在LLM模型本身，而且在它与其他组件的集成中。我们发现，尽管OpenAI GPT4设计了众多安全约束来改进其安全功能，但这些安全约束仍然容易受到攻击者的攻击。为了进一步展示我们发现的漏洞对现实世界的威胁，我们构建了一个端到端攻击，其中对手可以非法获取用户的聊天历史记录，而无需操纵用户的输入或获得对OpenAI GPT4的直接访问。我们的演示位于链接中：https://fzwark.github.io/LLM-System-Attack-Demo/



## **41. Making Them Ask and Answer: Jailbreaking Large Language Models in Few Queries via Disguise and Reconstruction**

让他们提问和回答：通过伪装和重建在几个查询中越狱大型语言模型 cs.CR

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.18104v1) [paper-pdf](http://arxiv.org/pdf/2402.18104v1)

**Authors**: Tong Liu, Yingjie Zhang, Zhe Zhao, Yinpeng Dong, Guozhu Meng, Kai Chen

**Abstract**: In recent years, large language models (LLMs) have demonstrated notable success across various tasks, but the trustworthiness of LLMs is still an open problem. One specific threat is the potential to generate toxic or harmful responses. Attackers can craft adversarial prompts that induce harmful responses from LLMs. In this work, we pioneer a theoretical foundation in LLMs security by identifying bias vulnerabilities within the safety fine-tuning and design a black-box jailbreak method named DRA (Disguise and Reconstruction Attack), which conceals harmful instructions through disguise and prompts the model to reconstruct the original harmful instruction within its completion. We evaluate DRA across various open-source and close-source models, showcasing state-of-the-art jailbreak success rates and attack efficiency. Notably, DRA boasts a 90\% attack success rate on LLM chatbots GPT-4.

摘要: 近年来，大型语言模型在各种任务上取得了显著的成功，但大型语言模型的可信度仍然是一个悬而未决的问题。一个具体的威胁是可能产生有毒或有害的反应。攻击者可以精心编制敌意提示，以诱导LLMS做出有害的响应。在这项工作中，我们通过识别安全微调中的偏差漏洞，开创了LLMS安全的理论基础，并设计了一种称为DRA(伪装和重建攻击)的黑盒越狱方法，该方法通过伪装来隐藏有害指令，并促使模型在其完成的范围内重建原始有害指令。我们通过各种开源和封闭源代码模型对DRA进行评估，展示最先进的越狱成功率和攻击效率。值得注意的是，DRA对LLM聊天机器人GPT-4的攻击成功率高达90%。



## **42. EmMark: Robust Watermarks for IP Protection of Embedded Quantized Large Language Models**

EmMark：用于嵌入式量化大语言模型知识产权保护的稳健水印 cs.CR

Accept to DAC 2024

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.17938v1) [paper-pdf](http://arxiv.org/pdf/2402.17938v1)

**Authors**: Ruisi Zhang, Farinaz Koushanfar

**Abstract**: This paper introduces EmMark,a novel watermarking framework for protecting the intellectual property (IP) of embedded large language models deployed on resource-constrained edge devices. To address the IP theft risks posed by malicious end-users, EmMark enables proprietors to authenticate ownership by querying the watermarked model weights and matching the inserted signatures. EmMark's novelty lies in its strategic watermark weight parameters selection, nsuring robustness and maintaining model quality. Extensive proof-of-concept evaluations of models from OPT and LLaMA-2 families demonstrate EmMark's fidelity, achieving 100% success in watermark extraction with model performance preservation. EmMark also showcased its resilience against watermark removal and forging attacks.

摘要: 介绍了EmMark，一种新的数字水印框架，用于保护部署在资源受限边缘设备上的嵌入式大语言模型的知识产权。为了应对恶意最终用户带来的知识产权盗窃风险，EmMark使所有者能够通过查询带水印的模型权重并匹配插入的签名来验证所有权。EmMark的新颖之处在于其战略性的水印权重参数选择，确保了稳健性，并保持了模型质量。对OPT和Llama-2家族的模型进行了广泛的概念验证评估，证明了EmMark的保真度，在保持模型性能的情况下实现了100%的水印提取。EmMark还展示了其对水印移除和伪造攻击的弹性。



## **43. LLM-Resistant Math Word Problem Generation via Adversarial Attacks**

基于对抗性攻击的抗LLM数学应用题生成 cs.CL

Code is available at  https://github.com/ruoyuxie/adversarial_mwps_generation

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.17916v1) [paper-pdf](http://arxiv.org/pdf/2402.17916v1)

**Authors**: Roy Xie, Chengxuan Huang, Junlin Wang, Bhuwan Dhingra

**Abstract**: Large language models (LLMs) have significantly transformed the educational landscape. As current plagiarism detection tools struggle to keep pace with LLMs' rapid advancements, the educational community faces the challenge of assessing students' true problem-solving abilities in the presence of LLMs. In this work, we explore a new paradigm for ensuring fair evaluation -- generating adversarial examples which preserve the structure and difficulty of the original questions aimed for assessment, but are unsolvable by LLMs. Focusing on the domain of math word problems, we leverage abstract syntax trees to structurally generate adversarial examples that cause LLMs to produce incorrect answers by simply editing the numeric values in the problems. We conduct experiments on various open- and closed-source LLMs, quantitatively and qualitatively demonstrating that our method significantly degrades their math problem-solving ability. We identify shared vulnerabilities among LLMs and propose a cost-effective approach to attack high-cost models. Additionally, we conduct automatic analysis on math problems and investigate the cause of failure to guide future research on LLM's mathematical capability.

摘要: 大型语言模型(LLM)极大地改变了教育格局。由于目前的抄袭检测工具难以跟上LLMS的快速进步，教育界面临着在LLMS存在的情况下评估学生真正的问题解决能力的挑战。在这项工作中，我们探索了一种确保公平评价的新范式--生成对抗性实例，它保留了用于评价的原始问题的结构和难度，但无法用LLMS解决。聚焦于数学应用题领域，我们利用抽象语法树来结构化地生成对抗性实例，这些实例通过简单地编辑问题中的数值来导致LLMS产生不正确的答案。我们在各种开源和闭源的LLM上进行了实验，定量和定性地证明了我们的方法显著降低了他们的数学问题解决能力。我们识别了LLM之间的共同漏洞，并提出了一种具有成本效益的方法来攻击高成本模型。此外，我们还对数学问题进行了自动分析，并调查了失败的原因，以指导未来对LLM数学能力的研究。



## **44. Mitigating Fine-tuning Jailbreak Attack with Backdoor Enhanced Alignment**

通过后门增强的对齐功能缓解精调越狱攻击 cs.CR

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.14968v2) [paper-pdf](http://arxiv.org/pdf/2402.14968v2)

**Authors**: Jiongxiao Wang, Jiazhao Li, Yiquan Li, Xiangyu Qi, Junjie Hu, Yixuan Li, Patrick McDaniel, Muhao Chen, Bo Li, Chaowei Xiao

**Abstract**: Despite the general capabilities of Large Language Models (LLMs) like GPT-4 and Llama-2, these models still request fine-tuning or adaptation with customized data when it comes to meeting the specific business demands and intricacies of tailored use cases. However, this process inevitably introduces new safety threats, particularly against the Fine-tuning based Jailbreak Attack (FJAttack), where incorporating just a few harmful examples into the fine-tuning dataset can significantly compromise the model safety. Though potential defenses have been proposed by incorporating safety examples into the fine-tuning dataset to reduce the safety issues, such approaches require incorporating a substantial amount of safety examples, making it inefficient. To effectively defend against the FJAttack with limited safety examples, we propose a Backdoor Enhanced Safety Alignment method inspired by an analogy with the concept of backdoor attacks. In particular, we construct prefixed safety examples by integrating a secret prompt, acting as a "backdoor trigger", that is prefixed to safety examples. Our comprehensive experiments demonstrate that through the Backdoor Enhanced Safety Alignment with adding as few as 11 prefixed safety examples, the maliciously fine-tuned LLMs will achieve similar safety performance as the original aligned models. Furthermore, we also explore the effectiveness of our method in a more practical setting where the fine-tuning data consists of both FJAttack examples and the fine-tuning task data. Our method shows great efficacy in defending against FJAttack without harming the performance of fine-tuning tasks.

摘要: 尽管GPT-4和LLAMA-2等大型语言模型(LLM)具有一般功能，但在满足特定业务需求和定制用例的复杂性时，这些模型仍然需要使用定制数据进行微调或调整。然而，这一过程不可避免地引入了新的安全威胁，特别是针对基于微调的越狱攻击(FJAttack)，在该攻击中，仅将几个有害的示例合并到微调数据集中可能会显著损害模型的安全性。虽然已经提出了通过将安全实例纳入微调数据集中来减少安全问题的潜在防御措施，但这种方法需要纳入大量的安全实例，从而使其效率低下。为了在安全示例有限的情况下有效防御FJAttack，我们提出了一种后门增强安全对齐方法，其灵感来自于后门攻击的概念。具体地说，我们通过集成一个秘密提示来构建前缀的安全实例，该提示充当安全实例的前缀的“后门触发器”。我们的综合实验表明，通过后门增强安全对齐，只需添加11个前缀安全实例，恶意微调的LLM将获得与原始对齐模型相似的安全性能。此外，我们还在一个更实际的环境中探索了我们的方法的有效性，其中微调数据包括FJAttack示例和微调任务数据。在不影响微调任务性能的情况下，我们的方法在防御FJAttack方面表现出了很好的效果。



## **45. Semantic Mirror Jailbreak: Genetic Algorithm Based Jailbreak Prompts Against Open-source LLMs**

语义镜像越狱：基于遗传算法的开源LLMS越狱提示 cs.CL

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.14872v2) [paper-pdf](http://arxiv.org/pdf/2402.14872v2)

**Authors**: Xiaoxia Li, Siyuan Liang, Jiyi Zhang, Han Fang, Aishan Liu, Ee-Chien Chang

**Abstract**: Large Language Models (LLMs), used in creative writing, code generation, and translation, generate text based on input sequences but are vulnerable to jailbreak attacks, where crafted prompts induce harmful outputs. Most jailbreak prompt methods use a combination of jailbreak templates followed by questions to ask to create jailbreak prompts. However, existing jailbreak prompt designs generally suffer from excessive semantic differences, resulting in an inability to resist defenses that use simple semantic metrics as thresholds. Jailbreak prompts are semantically more varied than the original questions used for queries. In this paper, we introduce a Semantic Mirror Jailbreak (SMJ) approach that bypasses LLMs by generating jailbreak prompts that are semantically similar to the original question. We model the search for jailbreak prompts that satisfy both semantic similarity and jailbreak validity as a multi-objective optimization problem and employ a standardized set of genetic algorithms for generating eligible prompts. Compared to the baseline AutoDAN-GA, SMJ achieves attack success rates (ASR) that are at most 35.4% higher without ONION defense and 85.2% higher with ONION defense. SMJ's better performance in all three semantic meaningfulness metrics of Jailbreak Prompt, Similarity, and Outlier, also means that SMJ is resistant to defenses that use those metrics as thresholds.

摘要: 用于创造性编写、代码生成和翻译的大型语言模型(LLM)根据输入序列生成文本，但容易受到越狱攻击，在这种攻击中，精心编制的提示会导致有害的输出。大多数越狱提示方法使用越狱模板和问题的组合来创建越狱提示。然而，现有的越狱提示设计通常存在过度的语义差异，导致无法抵抗使用简单语义度量作为阈值的防御。越狱提示在语义上比用于查询的原始问题更多样化。在本文中，我们介绍了一种语义镜像越狱(SMJ)方法，该方法通过生成与原始问题语义相似的越狱提示来绕过LLMS。我们将满足语义相似度和越狱有效性的越狱提示搜索问题建模为一个多目标优化问题，并使用一套标准化的遗传算法来生成合格的提示。与基线AutoDAN-GA相比，SMJ的攻击成功率(ASR)在没有洋葱防御的情况下最多提高了35.4%，在洋葱防御的情况下提高了85.2%。SMJ在越狱提示、相似度和离群值这三个语义意义指标上的表现都更好，这也意味着SMJ抵制使用这些指标作为阈值的防御。



## **46. HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal**

HarmBtch：一种标准化的自动化红队和稳健拒绝评估框架 cs.LG

Website: https://www.harmbench.org

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.04249v2) [paper-pdf](http://arxiv.org/pdf/2402.04249v2)

**Authors**: Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, David Forsyth, Dan Hendrycks

**Abstract**: Automated red teaming holds substantial promise for uncovering and mitigating the risks associated with the malicious use of large language models (LLMs), yet the field lacks a standardized evaluation framework to rigorously assess new methods. To address this issue, we introduce HarmBench, a standardized evaluation framework for automated red teaming. We identify several desirable properties previously unaccounted for in red teaming evaluations and systematically design HarmBench to meet these criteria. Using HarmBench, we conduct a large-scale comparison of 18 red teaming methods and 33 target LLMs and defenses, yielding novel insights. We also introduce a highly efficient adversarial training method that greatly enhances LLM robustness across a wide range of attacks, demonstrating how HarmBench enables codevelopment of attacks and defenses. We open source HarmBench at https://github.com/centerforaisafety/HarmBench.

摘要: 自动红色团队在发现和减轻与恶意使用大型语言模型(LLM)相关的风险方面有着很大的希望，但该领域缺乏标准的评估框架来严格评估新方法。为了解决这个问题，我们引入了HarmBtch，这是一个自动化红色团队的标准化评估框架。我们确定了几个以前在红队评估中没有考虑到的理想特性，并系统地设计了HarmBtch以满足这些标准。使用HarmBtch，我们对18种红队方法和33种目标LLM和防御进行了大规模比较，产生了新的见解。我们还引入了一种高效的对抗性训练方法，极大地增强了LLM在各种攻击中的健壮性，展示了HarmBtch如何实现攻击和防御的共同开发。我们在https://github.com/centerforaisafety/HarmBench.上开源了哈姆本奇



## **47. Pandora's White-Box: Increased Training Data Leakage in Open LLMs**

潘多拉的白盒：开放LLMS中训练数据泄露的增加 cs.CR

**SubmitDate**: 2024-02-26    [abs](http://arxiv.org/abs/2402.17012v1) [paper-pdf](http://arxiv.org/pdf/2402.17012v1)

**Authors**: Jeffrey G. Wang, Jason Wang, Marvin Li, Seth Neel

**Abstract**: In this paper we undertake a systematic study of privacy attacks against open source Large Language Models (LLMs), where an adversary has access to either the model weights, gradients, or losses, and tries to exploit them to learn something about the underlying training data. Our headline results are the first membership inference attacks (MIAs) against pre-trained LLMs that are able to simultaneously achieve high TPRs and low FPRs, and a pipeline showing that over $50\%$ (!) of the fine-tuning dataset can be extracted from a fine-tuned LLM in natural settings. We consider varying degrees of access to the underlying model, customization of the language model, and resources available to the attacker. In the pre-trained setting, we propose three new white-box MIAs: an attack based on the gradient norm, a supervised neural network classifier, and a single step loss ratio attack. All outperform existing black-box baselines, and our supervised attack closes the gap between MIA attack success against LLMs and other types of models. In fine-tuning, we find that given access to the loss of the fine-tuned and base models, a fine-tuned loss ratio attack FLoRA is able to achieve near perfect MIA peformance. We then leverage these MIAs to extract fine-tuning data from fine-tuned language models. We find that the pipeline of generating from fine-tuned models prompted with a small snippet of the prefix of each training example, followed by using FLoRa to select the most likely training sample, succeeds the majority of the fine-tuning dataset after only $3$ epochs of fine-tuning. Taken together, these findings show that highly effective MIAs are available in almost all LLM training settings, and highlight that great care must be taken before LLMs are fine-tuned on highly sensitive data and then deployed.

摘要: 在本文中，我们对针对开源大型语言模型(LLMS)的隐私攻击进行了系统的研究，其中攻击者可以访问模型的权重、梯度或损失，并试图利用它们来了解潜在的训练数据。我们的主要结果是针对预先训练的能够同时实现高TPR和低FPR的LLM的第一次成员推理攻击(MIA)，以及一个流水线显示超过50美元(！)可以在自然设置下从微调的LLM中提取精调数据集的。我们考虑对底层模型的不同程度的访问、语言模型的定制以及攻击者可用的资源。在预先训练的环境下，我们提出了三种新的白盒MIA：基于梯度范数的攻击、有监督的神经网络分类器和单步丢失率攻击。所有这些都超过了现有的黑盒基线，我们的监督攻击缩小了MIA对LLM的攻击成功与其他类型模型之间的差距。在微调中，我们发现，在获得微调和基本模型的损失的情况下，微调的损失率攻击菌群能够获得近乎完美的MIA性能。然后，我们利用这些MIA从微调的语言模型中提取微调数据。我们发现，在每个训练样本的前缀的一小段提示下，从微调模型生成的管道，然后使用FLORA来选择最可能的训练样本，在仅仅$3$的微调纪元之后，就成功了大部分微调数据集。综上所述，这些发现表明，高效的MIA在几乎所有LLM培训环境中都可用，并强调在对高度敏感的数据进行微调并随后部署LLM之前，必须非常小心。



## **48. WIPI: A New Web Threat for LLM-Driven Web Agents**

WIPI：LLM驱动的Web代理的一种新的Web威胁 cs.CR

**SubmitDate**: 2024-02-26    [abs](http://arxiv.org/abs/2402.16965v1) [paper-pdf](http://arxiv.org/pdf/2402.16965v1)

**Authors**: Fangzhou Wu, Shutong Wu, Yulong Cao, Chaowei Xiao

**Abstract**: With the fast development of large language models (LLMs), LLM-driven Web Agents (Web Agents for short) have obtained tons of attention due to their superior capability where LLMs serve as the core part of making decisions like the human brain equipped with multiple web tools to actively interact with external deployed websites. As uncountable Web Agents have been released and such LLM systems are experiencing rapid development and drawing closer to widespread deployment in our daily lives, an essential and pressing question arises: "Are these Web Agents secure?". In this paper, we introduce a novel threat, WIPI, that indirectly controls Web Agent to execute malicious instructions embedded in publicly accessible webpages. To launch a successful WIPI works in a black-box environment. This methodology focuses on the form and content of indirect instructions within external webpages, enhancing the efficiency and stealthiness of the attack. To evaluate the effectiveness of the proposed methodology, we conducted extensive experiments using 7 plugin-based ChatGPT Web Agents, 8 Web GPTs, and 3 different open-source Web Agents. The results reveal that our methodology achieves an average attack success rate (ASR) exceeding 90% even in pure black-box scenarios. Moreover, through an ablation study examining various user prefix instructions, we demonstrated that the WIPI exhibits strong robustness, maintaining high performance across diverse prefix instructions.

摘要: 随着大型语言模型(LLM)的快速发展，LLM驱动的Web代理(简称Web代理)因其优越的能力而获得了大量的关注，其中LLM是决策的核心部分，就像人脑配备了多个Web工具来与外部部署的网站进行主动交互一样。随着无数的Web代理被发布，这样的LLM系统正在经历快速的发展，并接近于在我们的日常生活中广泛部署，一个基本而紧迫的问题出现了：“这些Web代理安全吗？”在本文中，我们介绍了一种新的威胁，WIPI，它间接地控制Web代理执行嵌入到可公开访问的网页中的恶意指令。要推出一款成功的Wipi，需要在黑盒环境中工作。这种方法侧重于外部网页中间接指令的形式和内容，提高了攻击的效率和隐蔽性。为了评估提出的方法的有效性，我们使用7个基于插件的ChatGPT Web代理、8个Web GPT和3个不同的开源Web代理进行了广泛的实验。结果表明，即使在纯黑盒场景下，该方法的平均攻击成功率(ASR)也超过90%。此外，通过对不同用户前缀指令的消融研究，我们证明了WIPI表现出很强的健壮性，在不同的前缀指令中保持了高性能。



## **49. CodeChameleon: Personalized Encryption Framework for Jailbreaking Large Language Models**

CodeChameleon：面向越狱大型语言模型的个性化加密框架 cs.CL

**SubmitDate**: 2024-02-26    [abs](http://arxiv.org/abs/2402.16717v1) [paper-pdf](http://arxiv.org/pdf/2402.16717v1)

**Authors**: Huijie Lv, Xiao Wang, Yuansen Zhang, Caishuang Huang, Shihan Dou, Junjie Ye, Tao Gui, Qi Zhang, Xuanjing Huang

**Abstract**: Adversarial misuse, particularly through `jailbreaking' that circumvents a model's safety and ethical protocols, poses a significant challenge for Large Language Models (LLMs). This paper delves into the mechanisms behind such successful attacks, introducing a hypothesis for the safety mechanism of aligned LLMs: intent security recognition followed by response generation. Grounded in this hypothesis, we propose CodeChameleon, a novel jailbreak framework based on personalized encryption tactics. To elude the intent security recognition phase, we reformulate tasks into a code completion format, enabling users to encrypt queries using personalized encryption functions. To guarantee response generation functionality, we embed a decryption function within the instructions, which allows the LLM to decrypt and execute the encrypted queries successfully. We conduct extensive experiments on 7 LLMs, achieving state-of-the-art average Attack Success Rate (ASR). Remarkably, our method achieves an 86.6\% ASR on GPT-4-1106.

摘要: 对抗性的滥用，特别是通过绕过模型的安全和道德协议的“越狱”，给大型语言模型(LLM)带来了巨大的挑战。本文深入研究了此类成功攻击背后的机制，提出了一个关于联合LLMS安全机制的假设：意图安全识别，然后是响应生成。基于这一假设，我们提出了一种基于个性化加密策略的新型越狱框架CodeChameleon。为了避开意图安全识别阶段，我们将任务重新表述为代码完成格式，使用户能够使用个性化加密功能对查询进行加密。为了保证响应生成功能，我们在指令中嵌入了解密函数，它允许LLM成功解密和执行加密的查询。我们在7个LLM上进行了广泛的实验，获得了最先进的平均攻击成功率(ASR)。值得注意的是，我们的方法在GPT-4-1106上获得了86.6ASR。



## **50. RoCoIns: Enhancing Robustness of Large Language Models through Code-Style Instructions**

RoCoIns：通过代码风格的指令增强大型语言模型的健壮性 cs.CL

Accepted by COLING 2024

**SubmitDate**: 2024-02-26    [abs](http://arxiv.org/abs/2402.16431v1) [paper-pdf](http://arxiv.org/pdf/2402.16431v1)

**Authors**: Yuansen Zhang, Xiao Wang, Zhiheng Xi, Han Xia, Tao Gui, Qi Zhang, Xuanjing Huang

**Abstract**: Large Language Models (LLMs) have showcased remarkable capabilities in following human instructions. However, recent studies have raised concerns about the robustness of LLMs when prompted with instructions combining textual adversarial samples. In this paper, drawing inspiration from recent works that LLMs are sensitive to the design of the instructions, we utilize instructions in code style, which are more structural and less ambiguous, to replace typically natural language instructions. Through this conversion, we provide LLMs with more precise instructions and strengthen the robustness of LLMs. Moreover, under few-shot scenarios, we propose a novel method to compose in-context demonstrations using both clean and adversarial samples (\textit{adversarial context method}) to further boost the robustness of the LLMs. Experiments on eight robustness datasets show that our method consistently outperforms prompting LLMs with natural language instructions. For example, with gpt-3.5-turbo, our method achieves an improvement of 5.68\% in test set accuracy and a reduction of 5.66 points in Attack Success Rate (ASR).

摘要: 大型语言模型(LLM)在遵循人类指令方面表现出了非凡的能力。然而，最近的研究提出了对LLMS的稳健性的担忧，当提示结合文本对抗性样本的指令时。在本文中，我们从最近的研究中得到灵感，认为LLM对指令的设计很敏感，我们用更具结构性和更少歧义的代码风格的指令来取代通常的自然语言指令。通过这种转换，我们为LLMS提供了更精确的指令，增强了LLMS的健壮性。此外，在镜头较少的情况下，我们提出了一种使用干净样本和对抗性样本来合成上下文中演示的新方法(Texttit(对抗性上下文方法))，以进一步增强LLMS的健壮性。在八个健壮性数据集上的实验表明，我们的方法一致地优于使用自然语言指令的提示LLMS。例如，使用gpt-3.5-turbo，我们的方法在测试集精度上提高了5.68\%，攻击成功率(ASR)降低了5.66个点。



