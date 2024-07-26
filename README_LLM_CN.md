# Latest Large Language Model Attack Papers
**update at 2024-07-26 10:32:44**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Dr. Jekyll and Mr. Hyde: Two Faces of LLMs**

杰基尔博士和海德先生：法学硕士的两面 cs.CR

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2312.03853v4) [paper-pdf](http://arxiv.org/pdf/2312.03853v4)

**Authors**: Matteo Gioele Collu, Tom Janssen-Groesbeek, Stefanos Koffas, Mauro Conti, Stjepan Picek

**Abstract**: Recently, we have witnessed a rise in the use of Large Language Models (LLMs), especially in applications like chatbot assistants. Safety mechanisms and specialized training procedures are implemented to prevent improper responses from these assistants. In this work, we bypass these measures for ChatGPT and Gemini (and, to some extent, Bing chat) by making them impersonate complex personas with personality characteristics that are not aligned with a truthful assistant. We start by creating elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversations then follow a role-play style to elicit prohibited responses. Using personas, we show that prohibited responses are actually provided, making it possible to obtain unauthorized, illegal, or harmful information. This work shows that by using adversarial personas, one can overcome safety mechanisms set out by ChatGPT and Gemini. We also introduce several ways of activating such adversarial personas, which show that both chatbots are vulnerable to this kind of attack. With the same principle, we introduce two defenses that push the model to interpret trustworthy personalities and make it more robust against such attacks.

摘要: 最近，我们看到大型语言模型(LLM)的使用有所增加，特别是在聊天机器人助手等应用程序中。实施了安全机制和专门的培训程序，以防止这些助理做出不当反应。在这项工作中，我们绕过了ChatGPT和Gemini(在某种程度上，Bing聊天)的这些措施，让他们模仿具有与诚实的助手不一致的个性特征的复杂人物角色。我们首先为这些角色创建精致的传记，然后在与相同的聊天机器人的新会话中使用。然后，我们的对话遵循角色扮演的风格，以引发被禁止的回应。使用人物角色，我们展示了实际上提供了被禁止的响应，使得获得未经授权的、非法的或有害的信息成为可能。这项工作表明，通过使用敌对的人物角色，一个人可以克服ChatGPT和Gemini提出的安全机制。我们还介绍了几种激活这种敌对角色的方法，这表明这两个聊天机器人都容易受到这种攻击。在相同的原则下，我们引入了两个防御措施，推动该模型解释可信任的个性，并使其对此类攻击更加健壮。



## **2. The Dark Side of Function Calling: Pathways to Jailbreaking Large Language Models**

函数调用的阴暗面：越狱大型语言模型的途径 cs.CR

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2407.17915v1) [paper-pdf](http://arxiv.org/pdf/2407.17915v1)

**Authors**: Zihui Wu, Haichang Gao, Jianping He, Ping Wang

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities, but their power comes with significant security considerations. While extensive research has been conducted on the safety of LLMs in chat mode, the security implications of their function calling feature have been largely overlooked. This paper uncovers a critical vulnerability in the function calling process of LLMs, introducing a novel "jailbreak function" attack method that exploits alignment discrepancies, user coercion, and the absence of rigorous safety filters. Our empirical study, conducted on six state-of-the-art LLMs including GPT-4o, Claude-3.5-Sonnet, and Gemini-1.5-pro, reveals an alarming average success rate of over 90\% for this attack. We provide a comprehensive analysis of why function calls are susceptible to such attacks and propose defensive strategies, including the use of defensive prompts. Our findings highlight the urgent need for enhanced security measures in the function calling capabilities of LLMs, contributing to the field of AI safety by identifying a previously unexplored risk, designing an effective attack method, and suggesting practical defensive measures. Our code is available at https://github.com/wooozihui/jailbreakfunction.

摘要: 大型语言模型(LLM)已经展示了非凡的能力，但它们的强大也伴随着重要的安全考虑。虽然已经对聊天模式下的LLMS的安全性进行了广泛的研究，但其函数调用功能的安全含义在很大程度上被忽视了。本文揭示了LLMS函数调用过程中的一个严重漏洞，引入了一种新的“越狱函数”攻击方法，该方法利用了对齐差异、用户胁迫和缺乏严格的安全过滤器。我们在包括GPT-40、Claude-3.5-Sonnet和Gemini-1.5-Pro在内的六个最先进的LLM上进行的经验研究显示，该攻击的平均成功率超过90%，这是令人震惊的。我们对函数调用容易受到此类攻击的原因进行了全面分析，并提出了防御策略，包括使用防御提示。我们的发现突显了在LLMS的函数调用能力方面迫切需要增强安全措施，通过识别以前未探索的风险、设计有效的攻击方法并提出实用的防御措施来促进人工智能安全领域。我们的代码可以在https://github.com/wooozihui/jailbreakfunction.上找到



## **3. PenHeal: A Two-Stage LLM Framework for Automated Pentesting and Optimal Remediation**

PenHeal：用于自动冥想和最佳补救的两阶段LLM框架 cs.CR

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2407.17788v1) [paper-pdf](http://arxiv.org/pdf/2407.17788v1)

**Authors**: Junjie Huang, Quanyan Zhu

**Abstract**: Recent advances in Large Language Models (LLMs) have shown significant potential in enhancing cybersecurity defenses against sophisticated threats. LLM-based penetration testing is an essential step in automating system security evaluations by identifying vulnerabilities. Remediation, the subsequent crucial step, addresses these discovered vulnerabilities. Since details about vulnerabilities, exploitation methods, and software versions offer crucial insights into system weaknesses, integrating penetration testing with vulnerability remediation into a cohesive system has become both intuitive and necessary.   This paper introduces PenHeal, a two-stage LLM-based framework designed to autonomously identify and mitigate security vulnerabilities. The framework integrates two LLM-enabled components: the Pentest Module, which detects multiple vulnerabilities within a system, and the Remediation Module, which recommends optimal remediation strategies. The integration is facilitated through Counterfactual Prompting and an Instructor module that guides the LLMs using external knowledge to explore multiple potential attack paths effectively. Our experimental results demonstrate that PenHeal not only automates the identification and remediation of vulnerabilities but also significantly improves vulnerability coverage by 31%, increases the effectiveness of remediation strategies by 32%, and reduces the associated costs by 46% compared to baseline models. These outcomes highlight the transformative potential of LLMs in reshaping cybersecurity practices, offering an innovative solution to defend against cyber threats.

摘要: 大型语言模型(LLM)的最新进展已经显示出在加强针对复杂威胁的网络安全防御方面的巨大潜力。基于LLM的渗透测试是通过识别漏洞实现系统安全评估自动化的重要步骤。补救是后续的关键步骤，可解决这些已发现的漏洞。由于有关漏洞、利用方法和软件版本的详细信息提供了对系统弱点的重要洞察，因此将渗透测试与漏洞修复集成到一个连贯的系统中既是直观的，也是必要的。本文介绍了PenHeal，这是一个基于LLM的两阶段框架，旨在自主识别和缓解安全漏洞。该框架集成了两个启用LLM的组件：检测系统中多个漏洞的Pentest模块和建议最佳补救策略的补救模块。通过反事实提示和指导LLMS使用外部知识有效地探索多个潜在攻击路径的指导者模块来促进集成。我们的实验结果表明，PenHeal不仅自动化了漏洞的识别和修复，而且与基准模型相比，漏洞覆盖率显著提高了31%，修复策略的有效性提高了32%，相关成本降低了46%。这些结果突显了低成本管理在重塑网络安全实践方面的变革潜力，为防御网络威胁提供了一种创新的解决方案。



## **4. Towards Neural Network based Cognitive Models of Dynamic Decision-Making by Humans**

基于神经网络的人类动态决策认知模型 cs.LG

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2407.17622v1) [paper-pdf](http://arxiv.org/pdf/2407.17622v1)

**Authors**: Changyu Chen, Shashank Reddy Chirra, Maria José Ferreira, Cleotilde Gonzalez, Arunesh Sinha, Pradeep Varakantham

**Abstract**: Modelling human cognitive processes in dynamic decision-making tasks has been an endeavor in AI for a long time. Some initial works have attempted to utilize neural networks (and large language models) but often assume one common model for all humans and aim to emulate human behavior in aggregate. However, behavior of each human is distinct, heterogeneous and relies on specific past experiences in specific tasks. To that end, we build on a well known model of cognition, namely Instance Based Learning (IBL), that posits that decisions are made based on similar situations encountered in the past. We propose two new attention based neural network models to model human decision-making in dynamic settings. We experiment with two distinct datasets gathered from human subject experiment data, one focusing on detection of phishing email by humans and another where humans act as attackers in a cybersecurity setting and decide on an attack option. We conduct extensive experiments with our two neural network models, IBL, and GPT3.5, and demonstrate that one of our neural network models achieves the best performance in representing human decision-making. We find an interesting trend that all models predict a human's decision better if that human is better at the task. We also explore explanation of human decisions based on what our model considers important in prediction. Overall, our work yields promising results for further use of neural networks in cognitive modelling of human decision making. Our code is available at https://github.com/shshnkreddy/NCM-HDM.

摘要: 在动态决策任务中对人类认知过程进行建模一直是人工智能领域的一项努力。一些最初的工作试图利用神经网络(和大型语言模型)，但通常假设一个适用于所有人类的通用模型，并旨在总体上模拟人类的行为。然而，每个人的行为是不同的，不同的，依赖于特定任务中特定的过去经验。为此，我们建立了一个著名的认知模型，即基于实例的学习(IBL)，该模型假设决策是基于过去遇到的类似情况做出的。我们提出了两个新的基于注意力的神经网络模型来模拟动态环境下的人类决策。我们使用从人类受试者实验数据中收集的两个不同的数据集进行实验，一个专注于检测人类发送的钓鱼电子邮件，另一个则是人类在网络安全环境中充当攻击者，并决定攻击选项。我们用我们的两个神经网络模型IBL和GPT3.5进行了广泛的实验，并证明了我们的一个神经网络模型在表示人类决策方面取得了最好的性能。我们发现了一个有趣的趋势，所有模型都能更好地预测一个人的决定，如果那个人在这项任务上做得更好。我们还基于我们的模型认为在预测中重要的东西来探索对人类决策的解释。总体而言，我们的工作为进一步使用神经网络在人类决策的认知建模中提供了有希望的结果。我们的代码可以在https://github.com/shshnkreddy/NCM-HDM.上找到



## **5. Can Watermarking Large Language Models Prevent Copyrighted Text Generation and Hide Training Data?**

对大型语言模型进行水印可以阻止受版权保护的文本生成并隐藏训练数据吗？ cs.LG

21 pages, 6 figures

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2407.17417v1) [paper-pdf](http://arxiv.org/pdf/2407.17417v1)

**Authors**: Michael-Andrei Panaitescu-Liess, Zora Che, Bang An, Yuancheng Xu, Pankayaraj Pathmanathan, Souradip Chakraborty, Sicheng Zhu, Tom Goldstein, Furong Huang

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities in generating diverse and contextually rich text. However, concerns regarding copyright infringement arise as LLMs may inadvertently produce copyrighted material. In this paper, we first investigate the effectiveness of watermarking LLMs as a deterrent against the generation of copyrighted texts. Through theoretical analysis and empirical evaluation, we demonstrate that incorporating watermarks into LLMs significantly reduces the likelihood of generating copyrighted content, thereby addressing a critical concern in the deployment of LLMs. Additionally, we explore the impact of watermarking on Membership Inference Attacks (MIAs), which aim to discern whether a sample was part of the pretraining dataset and may be used to detect copyright violations. Surprisingly, we find that watermarking adversely affects the success rate of MIAs, complicating the task of detecting copyrighted text in the pretraining dataset. Finally, we propose an adaptive technique to improve the success rate of a recent MIA under watermarking. Our findings underscore the importance of developing adaptive methods to study critical problems in LLMs with potential legal implications.

摘要: 大型语言模型(LLM)在生成丰富多样的文本方面表现出了令人印象深刻的能力。然而，由于LLMS可能无意中产生了受版权保护的材料，因此出现了对侵犯版权的担忧。在这篇文章中，我们首先研究了水印LLM作为对版权文本产生的威慑的有效性。通过理论分析和实证评估，我们证明了在LLMS中加入水印显著降低了产生受版权保护内容的可能性，从而解决了LLMS部署中的一个关键问题。此外，我们还探讨了水印对成员关系推断攻击(MIA)的影响，该攻击旨在识别样本是否是预训练数据集的一部分，以及是否可以用于检测侵犯版权的行为。令人惊讶的是，我们发现水印对MIA的成功率产生了不利影响，使得在预训练数据集中检测受版权保护的文本的任务变得更加复杂。最后，我们提出了一种自适应技术来提高最近的MIA在水印下的成功率。我们的发现强调了开发适应性方法来研究LLMS中具有潜在法律含义的关键问题的重要性。



## **6. LLMmap: Fingerprinting For Large Language Models**

LLMmap：大型语言模型的指纹识别 cs.CR

version 0.1 (added missing refs)

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2407.15847v2) [paper-pdf](http://arxiv.org/pdf/2407.15847v2)

**Authors**: Dario Pasquini, Evgenios M. Kornaropoulos, Giuseppe Ateniese

**Abstract**: We introduce LLMmap, a first-generation fingerprinting attack targeted at LLM-integrated applications. LLMmap employs an active fingerprinting approach, sending carefully crafted queries to the application and analyzing the responses to identify the specific LLM model in use. With as few as 8 interactions, LLMmap can accurately identify LLMs with over 95% accuracy. More importantly, LLMmap is designed to be robust across different application layers, allowing it to identify LLMs operating under various system prompts, stochastic sampling hyperparameters, and even complex generation frameworks such as RAG or Chain-of-Thought.

摘要: 我们引入了LLMmap，这是一种针对LLM集成应用程序的第一代指纹攻击。LLMmap采用主动指纹识别方法，向应用程序发送精心设计的查询并分析响应以识别正在使用的特定LLM模型。LLMmap只需8次交互即可准确识别LLM，准确率超过95%。更重要的是，LLMmap的设计目的是在不同的应用层中具有鲁棒性，使其能够识别在各种系统提示、随机采样超参数甚至复杂的生成框架（例如RAG或思想链）下运行的LLM。



## **7. From Sands to Mansions: Enabling Automatic Full-Life-Cycle Cyberattack Construction with LLM**

从金沙到豪宅：利用LLM实现自动全生命周期网络攻击构建 cs.CR

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2407.16928v1) [paper-pdf](http://arxiv.org/pdf/2407.16928v1)

**Authors**: Lingzhi Wang, Jiahui Wang, Kyle Jung, Kedar Thiagarajan, Emily Wei, Xiangmin Shen, Yan Chen, Zhenyuan Li

**Abstract**: The escalating battles between attackers and defenders in cybersecurity make it imperative to test and evaluate defense capabilities from the attackers' perspective. However, constructing full-life-cycle cyberattacks and performing red team emulations requires significant time and domain knowledge from security experts. Existing cyberattack simulation frameworks face challenges such as limited technical coverage, inability to conduct full-life-cycle attacks, and the need for manual infrastructure building. These limitations hinder the quality and diversity of the constructed attacks. In this paper, we leveraged the capabilities of Large Language Models (LLMs) in summarizing knowledge from existing attack intelligence and generating executable machine code based on human knowledge. we proposed AURORA, an automatic end-to-end cyberattack construction and emulation framework. AURORA can autonomously build multi-stage cyberattack plans based on Cyber Threat Intelligence (CTI) reports, construct the emulation infrastructures, and execute the attack procedures. We also developed an attack procedure knowledge graph to integrate knowledge about attack techniques throughout the full life cycle of advanced cyberattacks from various sources. We constructed and evaluated more than 20 full-life-cycle cyberattacks based on existing CTI reports. Compared to previous attack simulation frameworks, AURORA can construct multi-step attacks and the infrastructures in several minutes without human intervention. Furthermore, AURORA incorporates a wider range (40% more) of attack techniques into the constructed attacks in a more efficient way than the professional red teams. To benefit further research, we open-sourced the dataset containing the execution files and infrastructures of 20 emulated cyberattacks.

摘要: 网络安全领域攻击者和防御者之间不断升级的战斗使得从攻击者的角度测试和评估防御能力势在必行。然而，构建全生命周期的网络攻击和执行Red Team模拟需要安全专家的大量时间和领域知识。现有的网络攻击模拟框架面临着诸如技术覆盖范围有限、无法进行全生命周期攻击以及需要手动构建基础设施等挑战。这些局限性阻碍了构建攻击的质量和多样性。在本文中，我们利用大型语言模型(LLM)的能力，从现有的攻击情报中总结知识，并基于人类知识生成可执行的机器代码。我们提出了Aurora，一个自动化的端到端网络攻击构建和仿真框架。Aurora可以根据网络威胁情报(CTI)报告自主构建多阶段网络攻击计划，构建仿真基础设施，并执行攻击程序。我们还开发了攻击过程知识图，以整合来自各种来源的高级网络攻击的整个生命周期中有关攻击技术的知识。我们根据现有的CTI报告构建和评估了20多个全生命周期的网络攻击。与以往的攻击模拟框架相比，Aurora可以在几分钟内构建多步骤攻击和基础设施，而无需人工干预。此外，极光在构建的攻击中融入了更广泛的攻击技术(多40%)，比专业的红色球队更有效。为了便于进一步研究，我们对包含20个模拟网络攻击的执行文件和基础设施的数据集进行了开源。



## **8. RigorLLM: Resilient Guardrails for Large Language Models against Undesired Content**

RigorLLM：针对不需要内容的大型语言模型的弹性护栏 cs.CR

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2403.13031v2) [paper-pdf](http://arxiv.org/pdf/2403.13031v2)

**Authors**: Zhuowen Yuan, Zidi Xiong, Yi Zeng, Ning Yu, Ruoxi Jia, Dawn Song, Bo Li

**Abstract**: Recent advancements in Large Language Models (LLMs) have showcased remarkable capabilities across various tasks in different domains. However, the emergence of biases and the potential for generating harmful content in LLMs, particularly under malicious inputs, pose significant challenges. Current mitigation strategies, while effective, are not resilient under adversarial attacks. This paper introduces Resilient Guardrails for Large Language Models (RigorLLM), a novel framework designed to efficiently and effectively moderate harmful and unsafe inputs and outputs for LLMs. By employing a multi-faceted approach that includes energy-based training data augmentation through Langevin dynamics, optimizing a safe suffix for inputs via minimax optimization, and integrating a fusion-based model combining robust KNN with LLMs based on our data augmentation, RigorLLM offers a robust solution to harmful content moderation. Our experimental evaluations demonstrate that RigorLLM not only outperforms existing baselines like OpenAI API and Perspective API in detecting harmful content but also exhibits unparalleled resilience to jailbreaking attacks. The innovative use of constrained optimization and a fusion-based guardrail approach represents a significant step forward in developing more secure and reliable LLMs, setting a new standard for content moderation frameworks in the face of evolving digital threats.

摘要: 大型语言模型(LLM)的最新进展展示了跨越不同领域的各种任务的显著能力。然而，偏见的出现和在低成本管理中产生有害内容的可能性，特别是在恶意投入下，构成了重大挑战。目前的缓解战略虽然有效，但在对抗性攻击下缺乏弹性。本文介绍了用于大型语言模型的弹性护栏(RigorLLM)，这是一个新的框架，旨在高效和有效地控制LLM中有害和不安全的输入和输出。通过采用多方面的方法，包括通过朗之万动力学基于能量的训练数据增强，通过极小极大优化优化输入的安全后缀，以及基于我们的数据增强将稳健的KNN与LLMS相结合的基于融合的模型，RigorLLM为有害内容适度提供了稳健的解决方案。我们的实验评估表明，RigorLLM不仅在检测有害内容方面优于OpenAI API和透视API等现有基线，而且对越狱攻击表现出无与伦比的弹性。约束优化和基于融合的护栏方法的创新使用代表着在开发更安全可靠的LLMS方面向前迈出的重要一步，为面对不断变化的数字威胁的内容审查框架设定了新的标准。



## **9. Watermarks in the Sand: Impossibility of Strong Watermarking for Generative Models**

沙子中的水印：生成模型不可能有强水印 cs.LG

ICML 2024. Website: https://hanlin-zhang.com/impossibility-watermarks

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2311.04378v4) [paper-pdf](http://arxiv.org/pdf/2311.04378v4)

**Authors**: Hanlin Zhang, Benjamin L. Edelman, Danilo Francati, Daniele Venturi, Giuseppe Ateniese, Boaz Barak

**Abstract**: Watermarking generative models consists of planting a statistical signal (watermark) in a model's output so that it can be later verified that the output was generated by the given model. A strong watermarking scheme satisfies the property that a computationally bounded attacker cannot erase the watermark without causing significant quality degradation. In this paper, we study the (im)possibility of strong watermarking schemes. We prove that, under well-specified and natural assumptions, strong watermarking is impossible to achieve. This holds even in the private detection algorithm setting, where the watermark insertion and detection algorithms share a secret key, unknown to the attacker. To prove this result, we introduce a generic efficient watermark attack; the attacker is not required to know the private key of the scheme or even which scheme is used. Our attack is based on two assumptions: (1) The attacker has access to a "quality oracle" that can evaluate whether a candidate output is a high-quality response to a prompt, and (2) The attacker has access to a "perturbation oracle" which can modify an output with a nontrivial probability of maintaining quality, and which induces an efficiently mixing random walk on high-quality outputs. We argue that both assumptions can be satisfied in practice by an attacker with weaker computational capabilities than the watermarked model itself, to which the attacker has only black-box access. Furthermore, our assumptions will likely only be easier to satisfy over time as models grow in capabilities and modalities. We demonstrate the feasibility of our attack by instantiating it to attack three existing watermarking schemes for large language models: Kirchenbauer et al. (2023), Kuditipudi et al. (2023), and Zhao et al. (2023). The same attack successfully removes the watermarks planted by all three schemes, with only minor quality degradation.

摘要: 水印生成模型包括在模型的输出中植入统计信号(水印)，以便稍后可以验证输出是由给定模型生成的。强水印方案满足这样的性质，即计算受限的攻击者不可能在不引起显著质量降级的情况下删除水印。本文研究了强水印方案的(Im)可能性。我们证明了在明确和自然的假设下，强水印是不可能实现的。即使在私有检测算法设置中也是如此，其中水印插入和检测算法共享攻击者未知的秘密密钥。为了证明这一结果，我们引入了一种通用的高效水印攻击；攻击者不需要知道方案的私钥，甚至不需要知道使用了哪个方案。我们的攻击基于两个假设：(1)攻击者可以访问可以评估候选输出是否是对提示的高质量响应的“质量预言”，以及(2)攻击者可以访问“扰动预言”，它可以以保持质量的非平凡概率修改输出，并导致对高质量输出的有效混合随机游走。我们认为，在实践中，这两个假设都可以由计算能力弱于水印模型本身的攻击者满足，因为攻击者只能访问黑盒。此外，随着模型在功能和模式方面的发展，随着时间的推移，我们的假设可能只会更容易满足。我们通过将其实例化来攻击三个现有的用于大型语言模型的水印方案来证明该攻击的可行性：Kirchenbauer等人。(2023)，Kuditipudi等人。(2023)，和赵等人。(2023年)。同样的攻击成功地删除了所有三个方案植入的水印，只有很小的质量下降。



## **10. Can Large Language Models Automatically Jailbreak GPT-4V?**

大型语言模型可以自动越狱GPT-4V吗？ cs.CL

TrustNLP@NAACL2024 (Fourth Workshop on Trustworthy Natural Language  Processing)

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2407.16686v1) [paper-pdf](http://arxiv.org/pdf/2407.16686v1)

**Authors**: Yuanwei Wu, Yue Huang, Yixin Liu, Xiang Li, Pan Zhou, Lichao Sun

**Abstract**: GPT-4V has attracted considerable attention due to its extraordinary capacity for integrating and processing multimodal information. At the same time, its ability of face recognition raises new safety concerns of privacy leakage. Despite researchers' efforts in safety alignment through RLHF or preprocessing filters, vulnerabilities might still be exploited. In our study, we introduce AutoJailbreak, an innovative automatic jailbreak technique inspired by prompt optimization. We leverage Large Language Models (LLMs) for red-teaming to refine the jailbreak prompt and employ weak-to-strong in-context learning prompts to boost efficiency. Furthermore, we present an effective search method that incorporates early stopping to minimize optimization time and token expenditure. Our experiments demonstrate that AutoJailbreak significantly surpasses conventional methods, achieving an Attack Success Rate (ASR) exceeding 95.3\%. This research sheds light on strengthening GPT-4V security, underscoring the potential for LLMs to be exploited in compromising GPT-4V integrity.

摘要: GPT-4V由于其综合和处理多模式信息的非凡能力而引起了相当大的关注。与此同时，它的人脸识别能力引发了新的隐私泄露的安全担忧。尽管研究人员通过RLHF或预处理过滤器在安全匹配方面做出了努力，但漏洞仍有可能被利用。在我们的研究中，我们介绍了AutoJailBreak，这是一种受即时优化启发的创新的自动越狱技术。我们利用用于红色团队的大型语言模型(LLM)来改进越狱提示，并采用从弱到强的上下文学习提示来提高效率。此外，我们还提出了一种结合提前停止的有效搜索方法，以最小化优化时间和令牌开销。实验表明，AutoJailBreak的攻击成功率(ASR)超过95.3%，明显优于传统方法。这项研究有助于加强GPT-4V的安全性，强调了LLMS在危害GPT-4V完整性方面的潜力。



## **11. RedAgent: Red Teaming Large Language Models with Context-aware Autonomous Language Agent**

RedAgent：Red将大型语言模型与上下文感知自治语言代理结合起来 cs.CR

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2407.16667v1) [paper-pdf](http://arxiv.org/pdf/2407.16667v1)

**Authors**: Huiyu Xu, Wenhui Zhang, Zhibo Wang, Feng Xiao, Rui Zheng, Yunhe Feng, Zhongjie Ba, Kui Ren

**Abstract**: Recently, advanced Large Language Models (LLMs) such as GPT-4 have been integrated into many real-world applications like Code Copilot. These applications have significantly expanded the attack surface of LLMs, exposing them to a variety of threats. Among them, jailbreak attacks that induce toxic responses through jailbreak prompts have raised critical safety concerns. To identify these threats, a growing number of red teaming approaches simulate potential adversarial scenarios by crafting jailbreak prompts to test the target LLM. However, existing red teaming methods do not consider the unique vulnerabilities of LLM in different scenarios, making it difficult to adjust the jailbreak prompts to find context-specific vulnerabilities. Meanwhile, these methods are limited to refining jailbreak templates using a few mutation operations, lacking the automation and scalability to adapt to different scenarios. To enable context-aware and efficient red teaming, we abstract and model existing attacks into a coherent concept called "jailbreak strategy" and propose a multi-agent LLM system named RedAgent that leverages these strategies to generate context-aware jailbreak prompts. By self-reflecting on contextual feedback in an additional memory buffer, RedAgent continuously learns how to leverage these strategies to achieve effective jailbreaks in specific contexts. Extensive experiments demonstrate that our system can jailbreak most black-box LLMs in just five queries, improving the efficiency of existing red teaming methods by two times. Additionally, RedAgent can jailbreak customized LLM applications more efficiently. By generating context-aware jailbreak prompts towards applications on GPTs, we discover 60 severe vulnerabilities of these real-world applications with only two queries per vulnerability. We have reported all found issues and communicated with OpenAI and Meta for bug fixes.

摘要: 最近，GPT-4等高级大型语言模型(LLM)已集成到许多实际应用程序中，如Code Copilot。这些应用程序显著扩大了LLMS的攻击面，使它们暴露在各种威胁之下。其中，通过越狱提示引发有毒反应的越狱攻击引发了严重的安全问题。为了识别这些威胁，越来越多的红色团队方法通过精心编制越狱提示来测试目标LLM，以模拟潜在的敌对场景。然而，现有的红色团队方法没有考虑LLM在不同场景下的独特漏洞，很难调整越狱提示来发现上下文特定的漏洞。同时，这些方法仅限于使用少量的变异操作来提炼越狱模板，缺乏适应不同场景的自动化和可扩展性。为了实现上下文感知和高效的红色团队，我们将现有的攻击抽象并建模为一个连贯的概念，称为越狱策略，并提出了一个名为RedAgent的多代理LLM系统，该系统利用这些策略来生成上下文感知越狱提示。通过对额外内存缓冲区中的上下文反馈进行自我反思，RedAgent不断学习如何利用这些策略在特定上下文中实现有效的越狱。大量的实验表明，我们的系统可以在短短五次查询中破解大部分黑盒LLM，将现有的红色团队方法的效率提高了两倍。此外，RedAgent可以更高效地越狱定制的LLM应用程序。通过向GPT上的应用程序生成上下文感知越狱提示，我们发现了这些现实世界应用程序的60个严重漏洞，每个漏洞只有两个查询。我们已经报告了所有发现的问题，并与OpenAI和Meta进行了沟通以修复错误。



## **12. Course-Correction: Safety Alignment Using Synthetic Preferences**

课程纠正：使用综合偏好进行安全调整 cs.CL

Dataset and script will be available at  https://github.com/pillowsofwind/Course-Correction

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2407.16637v1) [paper-pdf](http://arxiv.org/pdf/2407.16637v1)

**Authors**: Rongwu Xu, Yishuo Cai, Zhenhong Zhou, Renjie Gu, Haiqin Weng, Yan Liu, Tianwei Zhang, Wei Xu, Han Qiu

**Abstract**: The risk of harmful content generated by large language models (LLMs) becomes a critical concern. This paper presents a systematic study on assessing and improving LLMs' capability to perform the task of \textbf{course-correction}, \ie, the model can steer away from generating harmful content autonomously. To start with, we introduce the \textsc{C$^2$-Eval} benchmark for quantitative assessment and analyze 10 popular LLMs, revealing varying proficiency of current safety-tuned LLMs in course-correction. To improve, we propose fine-tuning LLMs with preference learning, emphasizing the preference for timely course-correction. Using an automated pipeline, we create \textsc{C$^2$-Syn}, a synthetic dataset with 750K pairwise preferences, to teach models the concept of timely course-correction through data-driven preference learning. Experiments on 2 LLMs, \textsc{Llama2-Chat 7B} and \textsc{Qwen2 7B}, show that our method effectively enhances course-correction skills without affecting general performance. Additionally, it effectively improves LLMs' safety, particularly in resisting jailbreak attacks.

摘要: 大型语言模型(LLM)生成有害内容的风险成为一个关键问题。本文对如何评估和提高LLMS执行航向修正任务的能力进行了系统的研究，即该模型可以避免自动产生有害内容。首先，我们引入了用于定量评估的基准，并对10个流行的LLM进行了分析，揭示了当前安全调整的LLMS在航向校正方面的不同熟练程度。为了改进，我们提出了带有偏好学习的微调LLMS，强调了对及时航向校正的偏好。使用自动化管道，我们创建了一个包含75万个成对偏好的合成数据集\extsc{C$^2$-Syn}，以通过数据驱动的偏好学习向模型传授及时进行课程修正的概念。在Textsc{Llama2-Chat 7B}和Textsc{Qwen2 7B}上的实验表明，该方法在不影响总体性能的情况下有效地增强了航向修正技巧。此外，它还有效地提高了LLMS的安全性，特别是在抵抗越狱攻击方面。



## **13. Defending Our Privacy With Backdoors**

用后门保护我们的隐私 cs.LG

Accepted at ECAI 2024

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2310.08320v4) [paper-pdf](http://arxiv.org/pdf/2310.08320v4)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Daniel Neider, Kristian Kersting

**Abstract**: The proliferation of large AI models trained on uncurated, often sensitive web-scraped data has raised significant privacy concerns. One of the concerns is that adversaries can extract information about the training data using privacy attacks. Unfortunately, the task of removing specific information from the models without sacrificing performance is not straightforward and has proven to be challenging. We propose a rather easy yet effective defense based on backdoor attacks to remove private information, such as names and faces of individuals, from vision-language models by fine-tuning them for only a few minutes instead of re-training them from scratch. Specifically, by strategically inserting backdoors into text encoders, we align the embeddings of sensitive phrases with those of neutral terms-"a person" instead of the person's actual name. For image encoders, we map individuals' embeddings to be removed from the model to a universal, anonymous embedding. The results of our extensive experimental evaluation demonstrate the effectiveness of our backdoor-based defense on CLIP by assessing its performance using a specialized privacy attack for zero-shot classifiers. Our approach provides a new "dual-use" perspective on backdoor attacks and presents a promising avenue to enhance the privacy of individuals within models trained on uncurated web-scraped data.

摘要: 大型人工智能模型的激增引发了人们对隐私的严重担忧。这些模型针对未经管理的、往往是敏感的网络数据进行培训。其中一个令人担忧的问题是，攻击者可以使用隐私攻击来提取有关训练数据的信息。不幸的是，在不牺牲性能的情况下从模型中删除特定信息的任务并不简单，而且已被证明是具有挑战性的。我们提出了一种基于后门攻击的相当简单但有效的防御方法，通过仅对视觉语言模型进行几分钟的微调来删除私人信息，如个人的姓名和面孔，而不是从头开始重新训练它们。具体地说，通过有策略地在文本编码器中插入后门，我们将敏感短语的嵌入与中性术语的嵌入--“人”而不是人的实际姓名--对齐。对于图像编码器，我们将从模型中移除的个人嵌入映射到通用的匿名嵌入。我们广泛的实验评估结果证明了我们的基于后门的CLIP防御的有效性，通过使用专门的针对零射击分类器的隐私攻击来评估其性能。我们的方法为后门攻击提供了一种新的“两用”视角，并提供了一种在未经管理的网络抓取数据的培训模型中增强个人隐私的有前景的途径。



## **14. Efficient Generation of Targeted and Transferable Adversarial Examples for Vision-Language Models Via Diffusion Models**

通过扩散模型高效生成视觉语言模型的有针对性且可转移的对抗示例 cs.CV

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2404.10335v3) [paper-pdf](http://arxiv.org/pdf/2404.10335v3)

**Authors**: Qi Guo, Shanmin Pang, Xiaojun Jia, Yang Liu, Qing Guo

**Abstract**: Adversarial attacks, particularly \textbf{targeted} transfer-based attacks, can be used to assess the adversarial robustness of large visual-language models (VLMs), allowing for a more thorough examination of potential security flaws before deployment. However, previous transfer-based adversarial attacks incur high costs due to high iteration counts and complex method structure. Furthermore, due to the unnaturalness of adversarial semantics, the generated adversarial examples have low transferability. These issues limit the utility of existing methods for assessing robustness. To address these issues, we propose AdvDiffVLM, which uses diffusion models to generate natural, unrestricted and targeted adversarial examples via score matching. Specifically, AdvDiffVLM uses Adaptive Ensemble Gradient Estimation to modify the score during the diffusion model's reverse generation process, ensuring that the produced adversarial examples have natural adversarial targeted semantics, which improves their transferability. Simultaneously, to improve the quality of adversarial examples, we use the GradCAM-guided Mask method to disperse adversarial semantics throughout the image rather than concentrating them in a single area. Finally, AdvDiffVLM embeds more target semantics into adversarial examples after multiple iterations. Experimental results show that our method generates adversarial examples 5x to 10x faster than state-of-the-art transfer-based adversarial attacks while maintaining higher quality adversarial examples. Furthermore, compared to previous transfer-based adversarial attacks, the adversarial examples generated by our method have better transferability. Notably, AdvDiffVLM can successfully attack a variety of commercial VLMs in a black-box environment, including GPT-4V.

摘要: 对抗性攻击，特别是基于传输的对抗性攻击，可用于评估大型视觉语言模型(VLM)的对抗性健壮性，从而允许在部署之前更彻底地检查潜在的安全漏洞。然而，以往基于转移的对抗性攻击由于迭代次数多、方法结构复杂，代价较高。此外，由于对抗性语义的非自然性，生成的对抗性实例可转移性较低。这些问题限制了现有稳健性评估方法的实用性。为了解决这些问题，我们提出了AdvDiffVLM，它使用扩散模型通过得分匹配来生成自然的、不受限制的和有针对性的对抗性实例。具体地说，AdvDiffVLM在扩散模型的反向生成过程中使用自适应集成梯度估计来修改分数，确保生成的对抗性实例具有自然对抗性目标语义，从而提高了它们的可转移性。同时，为了提高对抗性实例的质量，我们使用了GradCAM引导的掩码方法，将对抗性语义分散在整个图像中，而不是将它们集中在单个区域。最后，在多次迭代后，AdvDiffVLM将更多的目标语义嵌入到对抗性实例中。实验结果表明，在保持较高质量的对抗性实例的同时，我们的方法生成对抗性实例的速度比最新的基于传输的对抗性攻击快5倍到10倍。此外，与以往基于转移的对抗性攻击相比，该方法生成的对抗性实例具有更好的可转移性。值得注意的是，AdvDiffVLM可以在黑盒环境中成功攻击各种商业VLM，包括GPT-4V。



## **15. Figure it Out: Analyzing-based Jailbreak Attack on Large Language Models**

弄清楚：基于分析的对大型语言模型的越狱攻击 cs.CR

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2407.16205v1) [paper-pdf](http://arxiv.org/pdf/2407.16205v1)

**Authors**: Shi Lin, Rongchang Li, Xun Wang, Changting Lin, Wenpeng Xing, Meng Han

**Abstract**: The rapid development of Large Language Models (LLMs) has brought remarkable generative capabilities across diverse tasks. However, despite the impressive achievements, these models still have numerous security vulnerabilities, particularly when faced with jailbreak attacks. Therefore, by investigating jailbreak attacks, we can uncover hidden weaknesses in LLMs and guide us in developing more robust defense mechanisms to fortify their security. In this paper, we further explore the boundary of jailbreak attacks on LLMs and propose Analyzing-based Jailbreak (ABJ). This effective jailbreak attack method takes advantage of LLMs' growing analyzing and reasoning capability and reveals their underlying vulnerabilities when facing analysis-based tasks. We conduct a detailed evaluation of ABJ across various open-source and closed-source LLMs, which achieves 94.8% Attack Success Rate (ASR) and 1.06 Attack Efficiency (AE) on GPT-4-turbo-0409, demonstrating state-of-the-art attack effectiveness and efficiency. Our research highlights the importance of prioritizing and enhancing the safety of LLMs to mitigate the risks of misuse.

摘要: 大型语言模型(LLM)的快速发展带来了跨越各种任务的非凡的生成能力。然而，尽管取得了令人印象深刻的成就，这些模型仍然存在许多安全漏洞，特别是在面临越狱攻击时。因此，通过调查越狱攻击，我们可以发现LLMS中隐藏的弱点，并指导我们开发更强大的防御机制来加强它们的安全。本文进一步探讨了LLMS越狱攻击的边界，提出了基于分析的越狱攻击(ABJ)。这种有效的越狱攻击方法利用了LLMS日益增长的分析和推理能力，并在面对基于分析的任务时揭示了它们潜在的漏洞。我们对ABJ在各种开源和闭源LLMS上进行了详细的评估，在GPT-4-TURBO-0409上达到了94.8%的攻击成功率(ASR)和1.06的攻击效率(AE)，展示了最先进的攻击效果和效率。我们的研究强调了优先考虑和加强低密度脂蛋白的安全性，以减少误用风险的重要性。



## **16. Robust Privacy Amidst Innovation with Large Language Models Through a Critical Assessment of the Risks**

通过对风险的批判性评估，在大型语言模型的创新中实现稳健的隐私 cs.CL

13 pages, 4 figures, 1 table, 1 supplementary, under review

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2407.16166v1) [paper-pdf](http://arxiv.org/pdf/2407.16166v1)

**Authors**: Yao-Shun Chuang, Atiquer Rahman Sarkar, Noman Mohammed, Xiaoqian Jiang

**Abstract**: This study examines integrating EHRs and NLP with large language models (LLMs) to improve healthcare data management and patient care. It focuses on using advanced models to create secure, HIPAA-compliant synthetic patient notes for biomedical research. The study used de-identified and re-identified MIMIC III datasets with GPT-3.5, GPT-4, and Mistral 7B to generate synthetic notes. Text generation employed templates and keyword extraction for contextually relevant notes, with one-shot generation for comparison. Privacy assessment checked PHI occurrence, while text utility was tested using an ICD-9 coding task. Text quality was evaluated with ROUGE and cosine similarity metrics to measure semantic similarity with source notes. Analysis of PHI occurrence and text utility via the ICD-9 coding task showed that the keyword-based method had low risk and good performance. One-shot generation showed the highest PHI exposure and PHI co-occurrence, especially in geographic location and date categories. The Normalized One-shot method achieved the highest classification accuracy. Privacy analysis revealed a critical balance between data utility and privacy protection, influencing future data use and sharing. Re-identified data consistently outperformed de-identified data. This study demonstrates the effectiveness of keyword-based methods in generating privacy-protecting synthetic clinical notes that retain data usability, potentially transforming clinical data-sharing practices. The superior performance of re-identified over de-identified data suggests a shift towards methods that enhance utility and privacy by using dummy PHIs to perplex privacy attacks.

摘要: 这项研究考察了将EHR和NLP与大型语言模型(LLM)相结合，以改进医疗数据管理和患者护理。它专注于使用高级模型创建安全的、符合HIPAA标准的合成患者笔记，用于生物医学研究。这项研究使用了GPT-3.5、GPT-4和西风7B的去识别和重新识别的MIMIC III数据集来生成合成音符。文本生成使用模板和上下文相关笔记的关键字提取，并使用一次生成进行比较。隐私评估检查了PHI的发生，而文本实用程序则使用ICD-9编码任务进行了测试。使用Rouge和Cosine相似性度量来评价文本质量，以衡量与源注释的语义相似性。通过ICD-9编码任务对PHI发生情况和文本效用的分析表明，基于关键字的方法风险低，性能好。一次发生显示出最高的PHI暴露和PHI共同出现，特别是在地理位置和日期类别。归一化一次法达到了最高的分类精度。隐私分析揭示了数据效用和隐私保护之间的关键平衡，影响了未来数据的使用和共享。重新识别的数据始终优于未识别的数据。这项研究证明了基于关键字的方法在生成保护隐私的合成临床笔记方面的有效性，这些合成笔记保留了数据的可用性，潜在地改变了临床数据共享的做法。重新识别的数据优于未识别的数据，这表明通过使用虚拟PHI来困扰隐私攻击来增强实用性和隐私的方法发生了转变。



## **17. Flooding Spread of Manipulated Knowledge in LLM-Based Multi-Agent Communities**

基于LLM的多智能体社区中操纵知识的泛滥传播 cs.CL

18 Pages, working in progress

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2407.07791v2) [paper-pdf](http://arxiv.org/pdf/2407.07791v2)

**Authors**: Tianjie Ju, Yiting Wang, Xinbei Ma, Pengzhou Cheng, Haodong Zhao, Yulong Wang, Lifeng Liu, Jian Xie, Zhuosheng Zhang, Gongshen Liu

**Abstract**: The rapid adoption of large language models (LLMs) in multi-agent systems has highlighted their impressive capabilities in various applications, such as collaborative problem-solving and autonomous negotiation. However, the security implications of these LLM-based multi-agent systems have not been thoroughly investigated, particularly concerning the spread of manipulated knowledge. In this paper, we investigate this critical issue by constructing a detailed threat model and a comprehensive simulation environment that mirrors real-world multi-agent deployments in a trusted platform. Subsequently, we propose a novel two-stage attack method involving Persuasiveness Injection and Manipulated Knowledge Injection to systematically explore the potential for manipulated knowledge (i.e., counterfactual and toxic knowledge) spread without explicit prompt manipulation.   Our method leverages the inherent vulnerabilities of LLMs in handling world knowledge, which can be exploited by attackers to unconsciously spread fabricated information. Through extensive experiments, we demonstrate that our attack method can successfully induce LLM-based agents to spread both counterfactual and toxic knowledge without degrading their foundational capabilities during agent communication. Furthermore, we show that these manipulations can persist through popular retrieval-augmented generation frameworks, where several benign agents store and retrieve manipulated chat histories for future interactions. This persistence indicates that even after the interaction has ended, the benign agents may continue to be influenced by manipulated knowledge. Our findings reveal significant security risks in LLM-based multi-agent systems, emphasizing the imperative need for robust defenses against manipulated knowledge spread, such as introducing ``guardian'' agents and advanced fact-checking tools.

摘要: 大型语言模型在多智能体系统中的迅速应用凸显了其在协作问题求解、自主谈判等方面的应用能力。然而，这些基于LLM的多智能体系统的安全含义还没有得到彻底的研究，特别是关于被操纵的知识的传播。在本文中，我们通过构建一个详细的威胁模型和一个全面的模拟环境来研究这一关键问题，该环境反映了可信平台中真实世界的多代理部署。随后，我们提出了一种新的两阶段攻击方法，包括说服力注入和被操纵的知识注入，以系统地探索被操纵的知识(即反事实和有毒知识)在没有明确的即时操纵的情况下传播的可能性。我们的方法利用了LLMS在处理世界知识方面的固有漏洞，攻击者可以利用这些漏洞来不知不觉地传播伪造的信息。通过大量的实验，我们证明了我们的攻击方法可以成功地诱导基于LLM的代理传播反事实和有毒知识，而不会降低其在代理通信中的基础能力。此外，我们还表明，这些操作可以在流行的检索增强生成框架中持续存在，在这些框架中，几个良性代理存储和检索被操纵的聊天历史，以便将来进行交互。这种持久性表明，即使在互动结束后，良性代理人仍可能继续受到操纵知识的影响。我们的发现揭示了基于LLM的多代理系统中的重大安全风险，强调了对被操纵的知识传播采取强有力的防御措施的迫切需要，例如引入“监护人”代理和先进的事实核查工具。



## **18. The Shadow of Fraud: The Emerging Danger of AI-powered Social Engineering and its Possible Cure**

欺诈的阴影：人工智能驱动的社会工程的新危险及其可能的治疗方法 cs.CR

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.15912v1) [paper-pdf](http://arxiv.org/pdf/2407.15912v1)

**Authors**: Jingru Yu, Yi Yu, Xuhong Wang, Yilun Lin, Manzhi Yang, Yu Qiao, Fei-Yue Wang

**Abstract**: Social engineering (SE) attacks remain a significant threat to both individuals and organizations. The advancement of Artificial Intelligence (AI), including diffusion models and large language models (LLMs), has potentially intensified these threats by enabling more personalized and convincing attacks. This survey paper categorizes SE attack mechanisms, analyzes their evolution, and explores methods for measuring these threats. It highlights the challenges in raising awareness about the risks of AI-enhanced SE attacks and offers insights into developing proactive and adaptable defense strategies. Additionally, we introduce a categorization of the evolving nature of AI-powered social engineering attacks into "3E phases": Enlarging, wherein the magnitude of attacks expands through the leverage of digital media; Enriching, introducing novel attack vectors and techniques; and Emerging, signifying the advent of novel threats and methods. Moreover, we emphasize the necessity for a robust framework to assess the risk of AI-powered SE attacks. By identifying and addressing gaps in existing research, we aim to guide future studies and encourage the development of more effective defenses against the growing threat of AI-powered social engineering.

摘要: 社会工程(SE)攻击仍然是对个人和组织的重大威胁。人工智能(AI)的发展，包括扩散模型和大型语言模型(LLM)，通过实现更个性化和更有说服力的攻击，潜在地加剧了这些威胁。本调查报告对SE攻击机制进行了分类，分析了它们的演变，并探索了衡量这些威胁的方法。它强调了在提高对人工智能增强的SE攻击风险的认识方面的挑战，并为开发主动和自适应的防御战略提供了见解。此外，我们将人工智能支持的社会工程攻击的演变性质分类为“3E阶段”：扩大，其中攻击的规模通过数字媒体的杠杆作用扩大；丰富，引入新的攻击载体和技术；以及出现，标志着新威胁和方法的出现。此外，我们强调需要一个强大的框架来评估人工智能支持的SE攻击的风险。通过识别和解决现有研究中的差距，我们的目标是指导未来的研究，并鼓励开发更有效的防御措施，以应对人工智能驱动的社会工程日益增长的威胁。



## **19. Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts**

彩虹团队：开放式一代的多元化对抗预言 cs.CL

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2402.16822v2) [paper-pdf](http://arxiv.org/pdf/2402.16822v2)

**Authors**: Mikayel Samvelyan, Sharath Chandra Raparthy, Andrei Lupu, Eric Hambro, Aram H. Markosyan, Manish Bhatt, Yuning Mao, Minqi Jiang, Jack Parker-Holder, Jakob Foerster, Tim Rocktäschel, Roberta Raileanu

**Abstract**: As large language models (LLMs) become increasingly prevalent across many real-world applications, understanding and enhancing their robustness to adversarial attacks is of paramount importance. Existing methods for identifying adversarial prompts tend to focus on specific domains, lack diversity, or require extensive human annotations. To address these limitations, we present Rainbow Teaming, a novel black-box approach for producing a diverse collection of adversarial prompts. Rainbow Teaming casts adversarial prompt generation as a quality-diversity problem, and uses open-ended search to generate prompts that are both effective and diverse. Focusing on the safety domain, we use Rainbow Teaming to target various state-of-the-art LLMs, including the Llama 2 and Llama 3 models. Our approach reveals hundreds of effective adversarial prompts, with an attack success rate exceeding 90% across all tested models. Furthermore, we demonstrate that fine-tuning models with synthetic data generated by the Rainbow Teaming method significantly enhances their safety without sacrificing general performance or helpfulness. We additionally explore the versatility of Rainbow Teaming by applying it to question answering and cybersecurity, showcasing its potential to drive robust open-ended self-improvement in a wide range of applications.

摘要: 随着大型语言模型(LLM)在许多真实世界的应用中变得越来越普遍，理解和增强它们对对手攻击的健壮性是至关重要的。现有的识别对抗性提示的方法往往集中在特定的领域，缺乏多样性，或者需要大量的人工注释。为了解决这些局限性，我们提出了彩虹分组，这是一种新的黑盒方法，用于产生多样化的对抗性提示集合。彩虹团队将敌意提示生成视为质量多样性问题，并使用开放式搜索来生成既有效又多样化的提示。专注于安全领域，我们使用彩虹团队瞄准各种最先进的LLM，包括Llama 2和Llama 3型号。我们的方法揭示了数百个有效的对抗性提示，在所有测试模型上的攻击成功率超过90%。此外，我们证明了使用彩虹组合方法生成的合成数据对模型进行微调显著增强了它们的安全性，而不会牺牲总体性能或帮助。我们还探讨了彩虹团队的多功能性，将其应用于问题回答和网络安全，展示了其在广泛应用中推动强大的开放式自我改进的潜力。



## **20. Adversarial Style Augmentation via Large Language Model for Robust Fake News Detection**

通过大语言模型进行对抗风格增强以实现稳健的假新闻检测 cs.CL

8 pages

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2406.11260v2) [paper-pdf](http://arxiv.org/pdf/2406.11260v2)

**Authors**: Sungwon Park, Sungwon Han, Meeyoung Cha

**Abstract**: The spread of fake news negatively impacts individuals and is regarded as a significant social challenge that needs to be addressed. A number of algorithmic and insightful features have been identified for detecting fake news. However, with the recent LLMs and their advanced generation capabilities, many of the detectable features (e.g., style-conversion attacks) can be altered, making it more challenging to distinguish from real news. This study proposes adversarial style augmentation, AdStyle, to train a fake news detector that remains robust against various style-conversion attacks. Our model's key mechanism is the careful use of LLMs to automatically generate a diverse yet coherent range of style-conversion attack prompts. This improves the generation of prompts that are particularly difficult for the detector to handle. Experiments show that our augmentation strategy improves robustness and detection performance when tested on fake news benchmark datasets.

摘要: 假新闻的传播对个人产生负面影响，被视为需要解决的重大社会挑战。已经确定了许多算法和有洞察力的功能来检测假新闻。然而，随着最近的LLM及其先进一代能力，许多可检测的特征（例如，风格转换攻击）可以被更改，使其与真实新闻区分起来更具挑战性。这项研究提出了对抗性风格增强AdStyle来训练一个假新闻检测器，该检测器在对抗各种风格转换攻击时保持稳健。我们模型的关键机制是仔细使用LLM来自动生成多样化但连贯的风格转换攻击提示。这改善了检测器特别难以处理的提示的生成。实验表明，当在假新闻基准数据集上进行测试时，我们的增强策略提高了鲁棒性和检测性能。



## **21. Targeted Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs**

有针对性的隐性对抗培训提高了LLM对持续有害行为的稳健性 cs.LG

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.15549v1) [paper-pdf](http://arxiv.org/pdf/2407.15549v1)

**Authors**: Abhay Sheshadri, Aidan Ewart, Phillip Guo, Aengus Lynch, Cindy Wu, Vivek Hebbar, Henry Sleight, Asa Cooper Stickland, Ethan Perez, Dylan Hadfield-Menell, Stephen Casper

**Abstract**: Large language models (LLMs) can often be made to behave in undesirable ways that they are explicitly fine-tuned not to. For example, the LLM red-teaming literature has produced a wide variety of `jailbreaking' techniques to elicit harmful text from models that were fine-tuned to be harmless. Recent work on red-teaming, model editing, and interpretability suggests that this challenge stems from how (adversarial) fine-tuning largely serves to suppress rather than remove undesirable capabilities from LLMs. Prior work has introduced latent adversarial training (LAT) as a way to improve robustness to broad classes of failures. These prior works have considered untargeted latent space attacks where the adversary perturbs latent activations to maximize loss on examples of desirable behavior. Untargeted LAT can provide a generic type of robustness but does not leverage information about specific failure modes. Here, we experiment with targeted LAT where the adversary seeks to minimize loss on a specific competing task. We find that it can augment a wide variety of state-of-the-art methods. First, we use targeted LAT to improve robustness to jailbreaks, outperforming a strong R2D2 baseline with orders of magnitude less compute. Second, we use it to more effectively remove backdoors with no knowledge of the trigger. Finally, we use it to more effectively unlearn knowledge for specific undesirable tasks in a way that is also more robust to re-learning. Overall, our results suggest that targeted LAT can be an effective tool for defending against harmful behaviors from LLMs.

摘要: 大型语言模型(LLM)通常会以不受欢迎的方式运行，因此它们被明确微调为不以这种方式运行。例如，伦敦大学法学院的红队文学创作了各种各样的“越狱”技术，从经过微调的无害模特那里引出有害文本。最近在红团队、模型编辑和可解释性方面的工作表明，这一挑战源于(对抗性的)微调如何在很大程度上抑制而不是消除LLM中不受欢迎的能力。以前的工作已经引入了潜在的对手训练(LAT)，作为一种提高对广泛类别的故障的稳健性的方式。这些先前的工作考虑了无目标的潜在空间攻击，即对手扰乱潜在激活，以最大限度地减少期望行为的示例损失。非定向LAT可以提供一般类型的健壮性，但不利用有关特定故障模式的信息。在这里，我们实验有针对性的LAT，其中对手试图将特定竞争任务的损失降至最低。我们发现，它可以增加各种最先进的方法。首先，我们使用有针对性的LAT来提高对越狱的健壮性，性能优于强大的R2D2基线，计算量少了几个数量级。其次，我们使用它来更有效地删除后门，而不知道触发器。最后，我们使用它来更有效地忘记特定不受欢迎的任务的知识，这种方式也更适合重新学习。总体而言，我们的结果表明，有针对性的LAT可以成为防御LLM有害行为的有效工具。



## **22. TAPI: Towards Target-Specific and Adversarial Prompt Injection against Code LLMs**

TAPI：针对代码LLM的目标特定和对抗性即时注入 cs.CR

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.09164v3) [paper-pdf](http://arxiv.org/pdf/2407.09164v3)

**Authors**: Yuchen Yang, Hongwei Yao, Bingrun Yang, Yiling He, Yiming Li, Tianwei Zhang, Zhan Qin, Kui Ren

**Abstract**: Recently, code-oriented large language models (Code LLMs) have been widely and successfully used to simplify and facilitate code programming. With these tools, developers can easily generate desired complete functional codes based on incomplete code and natural language prompts. However, a few pioneering works revealed that these Code LLMs are also vulnerable, e.g., against backdoor and adversarial attacks. The former could induce LLMs to respond to triggers to insert malicious code snippets by poisoning the training data or model parameters, while the latter can craft malicious adversarial input codes to reduce the quality of generated codes. However, both attack methods have underlying limitations: backdoor attacks rely on controlling the model training process, while adversarial attacks struggle with fulfilling specific malicious purposes.   To inherit the advantages of both backdoor and adversarial attacks, this paper proposes a new attack paradigm, i.e., target-specific and adversarial prompt injection (TAPI), against Code LLMs. TAPI generates unreadable comments containing information about malicious instructions and hides them as triggers in the external source code. When users exploit Code LLMs to complete codes containing the trigger, the models will generate attacker-specified malicious code snippets at specific locations. We evaluate our TAPI attack on four representative LLMs under three representative malicious objectives and seven cases. The results show that our method is highly threatening (achieving an attack success rate of up to 98.3%) and stealthy (saving an average of 53.1% of tokens in the trigger design). In particular, we successfully attack some famous deployed code completion integrated applications, including CodeGeex and Github Copilot. This further confirms the realistic threat of our attack.

摘要: 最近，面向代码的大型语言模型(Code LLM)已被广泛并成功地用于简化和促进代码编程。使用这些工具，开发人员可以根据不完整的代码和自然语言提示轻松生成所需的完整功能代码。然而，一些开创性的工作表明，这些代码LLM也容易受到攻击，例如，抵御后门和对手攻击。前者可以通过毒化训练数据或模型参数来诱导LLMS响应插入恶意代码片段的触发器，而后者可以手工创建恶意输入代码来降低生成代码的质量。然而，这两种攻击方法都有潜在的局限性：后门攻击依赖于控制模型训练过程，而对抗性攻击则难以实现特定的恶意目的。为了继承后门攻击和对抗性攻击的优点，提出了一种新的针对Code LLMS的攻击范式，即目标特定和对抗性提示注入(TAPI)。TAPI生成不可读的注释，其中包含有关恶意指令的信息，并将它们作为触发器隐藏在外部源代码中。当用户利用Code LLMS来完成包含触发器的代码时，模型将在特定位置生成攻击者指定的恶意代码片段。我们在三个典型的恶意目标和七个案例下评估了我们的TAPI攻击对四个有代表性的LLM的攻击。结果表明，该方法具有很高的威胁性(攻击成功率高达98.3%)和隐蔽性(在触发器设计中平均节省53.1%的令牌)。特别是，我们成功地攻击了一些著名的部署代码完成集成应用程序，包括CodeGeex和Github Copilot。这进一步证实了我们攻击的现实威胁。



## **23. Imposter.AI: Adversarial Attacks with Hidden Intentions towards Aligned Large Language Models**

冒名顶替。AI：针对对齐大型语言模型的具有隐藏意图的对抗攻击 cs.CL

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.15399v1) [paper-pdf](http://arxiv.org/pdf/2407.15399v1)

**Authors**: Xiao Liu, Liangzhi Li, Tong Xiang, Fuying Ye, Lu Wei, Wangyue Li, Noa Garcia

**Abstract**: With the development of large language models (LLMs) like ChatGPT, both their vast applications and potential vulnerabilities have come to the forefront. While developers have integrated multiple safety mechanisms to mitigate their misuse, a risk remains, particularly when models encounter adversarial inputs. This study unveils an attack mechanism that capitalizes on human conversation strategies to extract harmful information from LLMs. We delineate three pivotal strategies: (i) decomposing malicious questions into seemingly innocent sub-questions; (ii) rewriting overtly malicious questions into more covert, benign-sounding ones; (iii) enhancing the harmfulness of responses by prompting models for illustrative examples. Unlike conventional methods that target explicit malicious responses, our approach delves deeper into the nature of the information provided in responses. Through our experiments conducted on GPT-3.5-turbo, GPT-4, and Llama2, our method has demonstrated a marked efficacy compared to conventional attack methods. In summary, this work introduces a novel attack method that outperforms previous approaches, raising an important question: How to discern whether the ultimate intent in a dialogue is malicious?

摘要: 随着像ChatGPT这样的大型语言模型(LLM)的发展，它们的巨大应用和潜在的漏洞都已经浮出水面。虽然开发人员已经集成了多种安全机制来减少它们的滥用，但风险仍然存在，特别是当模型遇到敌对输入时。这项研究揭示了一种利用人类对话策略从LLMS中提取有害信息的攻击机制。我们描述了三个关键策略：(I)将恶意问题分解为看似无害的子问题；(Ii)将公开的恶意问题重写为更隐蔽、听起来更温和的问题；(Iii)通过提示示例模型来增强回答的危害性。与针对显式恶意响应的传统方法不同，我们的方法更深入地挖掘响应中提供的信息的性质。通过我们在GPT-3.5-Turbo、GPT-4和Llama2上的实验，我们的方法比传统的攻击方法表现出了显著的效果。总之，这项工作引入了一种新的攻击方法，其性能优于以前的方法，提出了一个重要的问题：如何识别对话中的最终意图是否为恶意的？



## **24. Advancing TTP Analysis: Harnessing the Power of Large Language Models with Retrieval Augmented Generation**

推进TTP分析：利用检索增强生成来利用大型语言模型的力量 cs.CR

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2401.00280v3) [paper-pdf](http://arxiv.org/pdf/2401.00280v3)

**Authors**: Reza Fayyazi, Rozhina Taghdimi, Shanchieh Jay Yang

**Abstract**: Tactics, Techniques, and Procedures (TTPs) outline the methods attackers use to exploit vulnerabilities. The interpretation of TTPs in the MITRE ATT&CK framework can be challenging for cybersecurity practitioners due to presumed expertise and complex dependencies. Meanwhile, advancements with Large Language Models (LLMs) have led to recent surge in studies exploring its uses in cybersecurity operations. It is, however, unclear how LLMs can be used in an efficient and proper way to provide accurate responses for critical domains such as cybersecurity. This leads us to investigate how to better use two types of LLMs: small-scale encoder-only (e.g., RoBERTa) and larger decoder-only (e.g., GPT-3.5) LLMs to comprehend and summarize TTPs with the intended purposes (i.e., tactics) of a cyberattack procedure. This work studies and compares the uses of supervised fine-tuning (SFT) of encoder-only LLMs vs. Retrieval Augmented Generation (RAG) for decoder-only LLMs (without fine-tuning). Both SFT and RAG techniques presumably enhance the LLMs with relevant contexts for each cyberattack procedure. Our studies show decoder-only LLMs with RAG achieves better performance than encoder-only models with SFT, particularly when directly relevant context is extracted by RAG. The decoder-only results could suffer low `Precision' while achieving high `Recall'. Our findings further highlight a counter-intuitive observation that more generic prompts tend to yield better predictions of cyberattack tactics than those that are more specifically tailored.

摘要: 战术、技术和过程(TTP)概述了攻击者用来利用漏洞的方法。由于假定的专业知识和复杂的依赖关系，MITRE ATT&CK框架中对TTP的解释可能会对网络安全从业者构成挑战。与此同时，大型语言模型(LLM)的进步导致了最近探索其在网络安全行动中的应用的研究激增。然而，目前尚不清楚如何以有效和适当的方式使用LLMS，为网络安全等关键领域提供准确的响应。这导致我们研究如何更好地使用两种类型的LLP：仅限小规模编码器(例如Roberta)和仅限解码器(例如GPT-3.5)的LLM来理解和总结具有网络攻击过程的预期目的(即战术)的TTP。这项工作研究和比较了仅编码器LLMS的监督微调(SFT)和仅解码器LLMS(未微调)的检索增强生成(RAG)的使用。SFT和RAG技术都可能通过每个网络攻击过程的相关上下文来增强LLMS。我们的研究表明，只有解码者的RAG模型比只有编码者的SFT模型具有更好的性能，特别是当RAG提取直接相关的上下文时。只有解码器的结果可能会受到低精度的影响，而获得高的‘Recall’。我们的发现进一步突显了一种与直觉相反的观察，即更笼统的提示往往比更具体的提示更能预测网络攻击策略。



## **25. When Do Universal Image Jailbreaks Transfer Between Vision-Language Models?**

普遍形象越狱何时在视觉语言模型之间转移？ cs.CL

**SubmitDate**: 2024-07-21    [abs](http://arxiv.org/abs/2407.15211v1) [paper-pdf](http://arxiv.org/pdf/2407.15211v1)

**Authors**: Rylan Schaeffer, Dan Valentine, Luke Bailey, James Chua, Cristóbal Eyzaguirre, Zane Durante, Joe Benton, Brando Miranda, Henry Sleight, John Hughes, Rajashree Agrawal, Mrinank Sharma, Scott Emmons, Sanmi Koyejo, Ethan Perez

**Abstract**: The integration of new modalities into frontier AI systems offers exciting capabilities, but also increases the possibility such systems can be adversarially manipulated in undesirable ways. In this work, we focus on a popular class of vision-language models (VLMs) that generate text outputs conditioned on visual and textual inputs. We conducted a large-scale empirical study to assess the transferability of gradient-based universal image "jailbreaks" using a diverse set of over 40 open-parameter VLMs, including 18 new VLMs that we publicly release. Overall, we find that transferable gradient-based image jailbreaks are extremely difficult to obtain. When an image jailbreak is optimized against a single VLM or against an ensemble of VLMs, the jailbreak successfully jailbreaks the attacked VLM(s), but exhibits little-to-no transfer to any other VLMs; transfer is not affected by whether the attacked and target VLMs possess matching vision backbones or language models, whether the language model underwent instruction-following and/or safety-alignment training, or many other factors. Only two settings display partially successful transfer: between identically-pretrained and identically-initialized VLMs with slightly different VLM training data, and between different training checkpoints of a single VLM. Leveraging these results, we then demonstrate that transfer can be significantly improved against a specific target VLM by attacking larger ensembles of "highly-similar" VLMs. These results stand in stark contrast to existing evidence of universal and transferable text jailbreaks against language models and transferable adversarial attacks against image classifiers, suggesting that VLMs may be more robust to gradient-based transfer attacks.

摘要: 将新的模式集成到前沿人工智能系统中提供了令人兴奋的能力，但也增加了此类系统被以不受欢迎的方式进行相反操作的可能性。在这项工作中，我们专注于一类流行的视觉语言模型(VLM)，它们生成以视觉和文本输入为条件的文本输出。我们进行了一项大规模的实证研究，以评估基于梯度的通用图像“越狱”的可转移性，使用了一组超过40个开放参数的VLM，其中包括我们公开发布的18个新的VLM。总体而言，我们发现基于梯度的可转移越狱图像非常难以获得。当针对单个VLM或一组VLM优化图像越狱时，越狱成功地越狱了被攻击的VLM(S)，但很少或根本不转移到任何其他VLM；转移不受攻击和目标VLM是否具有匹配的视觉主干或语言模型、语言模型是否经过指令遵循和/或安全对齐培训或许多其他因素的影响。只有两个设置显示部分成功的传输：在具有略微不同的VLM训练数据的相同预训练和相同初始化的VLM之间，以及在单个VLM的不同训练检查点之间。利用这些结果，我们随后证明了针对特定目标VLM的传输可以通过攻击更大的“高度相似的”VLM集合来显著改进。这些结果与针对语言模型的普遍和可传输的文本越狱以及针对图像分类器的可传输的对抗性攻击的现有证据形成了鲜明对比，这表明VLM可能对基于梯度的传输攻击更健壮。



## **26. Trading Devil Final: Backdoor attack via Stock market and Bayesian Optimization**

交易魔鬼决赛：通过股市和Bayesian优化进行后门攻击 cs.LG

jumps-Diffusion and stock market: Better quantify uncertainty in  financial simulations

**SubmitDate**: 2024-07-21    [abs](http://arxiv.org/abs/2407.14573v1) [paper-pdf](http://arxiv.org/pdf/2407.14573v1)

**Authors**: Orson Mengara

**Abstract**: Since the advent of generative artificial intelligence, every company and researcher has been rushing to develop their own generative models, whether commercial or not. Given the large number of users of these powerful new tools, there is currently no intrinsically verifiable way to explain from the ground up what happens when LLMs (large language models) learn. For example, those based on automatic speech recognition systems, which have to rely on huge and astronomical amounts of data collected from all over the web to produce fast and efficient results, In this article, we develop a backdoor attack called MarketBackFinal 2.0, based on acoustic data poisoning, MarketBackFinal 2.0 is mainly based on modern stock market models. In order to show the possible vulnerabilities of speech-based transformers that may rely on LLMs.

摘要: 自生成人工智能出现以来，每家公司和研究人员都在争先恐后地开发自己的生成模型，无论是否商业化。鉴于这些强大的新工具的大量用户，目前还没有本质上可验证的方法来从头解释LLM（大型语言模型）学习时会发生什么。例如，那些基于自动语音识别系统的系统，它们必须依赖于从整个网络收集的大量数据来产生快速有效的结果，在本文中，我们开发了一种名为MarketBackFinal 2.0的后门攻击，基于声学数据中毒，MarketBackFinal 2.0主要基于现代股市模型。为了显示可能依赖LLM的基于语音的转换器可能存在的漏洞。



## **27. Arondight: Red Teaming Large Vision Language Models with Auto-generated Multi-modal Jailbreak Prompts**

Arondight：Red将大视觉语言模型与自动生成的多模式越狱脚本结合起来 cs.LG

To be published in ACM MM 2024

**SubmitDate**: 2024-07-21    [abs](http://arxiv.org/abs/2407.15050v1) [paper-pdf](http://arxiv.org/pdf/2407.15050v1)

**Authors**: Yi Liu, Chengjun Cai, Xiaoli Zhang, Xingliang Yuan, Cong Wang

**Abstract**: Large Vision Language Models (VLMs) extend and enhance the perceptual abilities of Large Language Models (LLMs). Despite offering new possibilities for LLM applications, these advancements raise significant security and ethical concerns, particularly regarding the generation of harmful content. While LLMs have undergone extensive security evaluations with the aid of red teaming frameworks, VLMs currently lack a well-developed one. To fill this gap, we introduce Arondight, a standardized red team framework tailored specifically for VLMs. Arondight is dedicated to resolving issues related to the absence of visual modality and inadequate diversity encountered when transitioning existing red teaming methodologies from LLMs to VLMs. Our framework features an automated multi-modal jailbreak attack, wherein visual jailbreak prompts are produced by a red team VLM, and textual prompts are generated by a red team LLM guided by a reinforcement learning agent. To enhance the comprehensiveness of VLM security evaluation, we integrate entropy bonuses and novelty reward metrics. These elements incentivize the RL agent to guide the red team LLM in creating a wider array of diverse and previously unseen test cases. Our evaluation of ten cutting-edge VLMs exposes significant security vulnerabilities, particularly in generating toxic images and aligning multi-modal prompts. In particular, our Arondight achieves an average attack success rate of 84.5\% on GPT-4 in all fourteen prohibited scenarios defined by OpenAI in terms of generating toxic text. For a clearer comparison, we also categorize existing VLMs based on their safety levels and provide corresponding reinforcement recommendations. Our multimodal prompt dataset and red team code will be released after ethics committee approval. CONTENT WARNING: THIS PAPER CONTAINS HARMFUL MODEL RESPONSES.

摘要: 大视觉语言模型扩展并增强了大语言模型的感知能力。尽管为LLM应用提供了新的可能性，但这些进展引发了重大的安全和伦理问题，特别是在有害内容的生成方面。虽然在红色团队框架的帮助下，低成本管理系统已经进行了广泛的安全评估，但目前还缺乏一个完善的安全评估框架。为了填补这一空白，我们引入了Arondiight，这是一个专门为VLM定制的标准化红色团队框架。Arondiight致力于解决在将现有的红色团队方法从LLMS过渡到VLMS时遇到的视觉形态缺失和多样性不足的问题。我们的框架以自动多模式越狱攻击为特色，其中视觉越狱提示由红色团队VLM生成，文本提示由红色团队LLM生成，并由强化学习代理引导。为了增强VLM安全评估的全面性，我们将熵奖金和新颖性奖励指标结合起来。这些元素激励RL代理指导红色团队LLM创建更广泛的多样化和以前未见过的测试用例。我们对10个尖端VLM的评估暴露了严重的安全漏洞，特别是在生成有毒图像和对齐多模式提示方面。特别是，我们的Arondiight在生成有毒文本方面，在OpenAI定义的所有14个禁止场景中，对GPT-4的平均攻击成功率为84.5\%。为了更清楚地进行比较，我们还根据现有的VLM的安全级别对其进行了分类，并提出了相应的加固建议。我们的多模式提示数据集和红色团队代码将在道德委员会批准后发布。内容警告：本文包含有害模型回复。



## **28. Sim-CLIP: Unsupervised Siamese Adversarial Fine-Tuning for Robust and Semantically-Rich Vision-Language Models**

Sim-CLIP：针对稳健且语义丰富的视觉语言模型的无监督Siamese对抗微调 cs.CV

**SubmitDate**: 2024-07-20    [abs](http://arxiv.org/abs/2407.14971v1) [paper-pdf](http://arxiv.org/pdf/2407.14971v1)

**Authors**: Md Zarif Hossain, Ahmed Imteaj

**Abstract**: Vision-language models (VLMs) have achieved significant strides in recent times specially in multimodal tasks, yet they remain susceptible to adversarial attacks on their vision components. To address this, we propose Sim-CLIP, an unsupervised adversarial fine-tuning method that enhances the robustness of the widely-used CLIP vision encoder against such attacks while maintaining semantic richness and specificity. By employing a Siamese architecture with cosine similarity loss, Sim-CLIP learns semantically meaningful and attack-resilient visual representations without requiring large batch sizes or momentum encoders. Our results demonstrate that VLMs enhanced with Sim-CLIP's fine-tuned CLIP encoder exhibit significantly enhanced robustness against adversarial attacks, while preserving semantic meaning of the perturbed images. Notably, Sim-CLIP does not require additional training or fine-tuning of the VLM itself; replacing the original vision encoder with our fine-tuned Sim-CLIP suffices to provide robustness. This work underscores the significance of reinforcing foundational models like CLIP to safeguard the reliability of downstream VLM applications, paving the way for more secure and effective multimodal systems.

摘要: 视觉语言模型近年来取得了长足的进步，特别是在多通道任务中，但它们仍然容易受到视觉部分的敌意攻击。为了解决这一问题，我们提出了SIM-CLIP，这是一种无监督的对抗性微调方法，它在保持语义丰富和特异性的同时，增强了广泛使用的CLIP视觉编码器对此类攻击的健壮性。通过采用具有余弦相似性损失的暹罗体系结构，Sim-Clip无需大批量或动量编码器即可学习语义上有意义的、可抵抗攻击的视觉表示。结果表明，通过Sim-Clip的精细调整的CLIP编码器增强的VLM在保持扰动图像语义的同时，显著增强了对对手攻击的稳健性。值得注意的是，SIM-Clip不需要对VLM本身进行额外的培训或微调；用我们经过微调的SIM-Clip替换原来的视觉编码器就足以提供健壮性。这项工作强调了加强像CLIP这样的基础模型对保障下游VLM应用的可靠性的重要性，为更安全和有效的多式联运系统铺平了道路。



## **29. Operationalizing a Threat Model for Red-Teaming Large Language Models (LLMs)**

为Red-Teaming大型语言模型（LLM）操作威胁模型 cs.CL

Preprint. Under review

**SubmitDate**: 2024-07-20    [abs](http://arxiv.org/abs/2407.14937v1) [paper-pdf](http://arxiv.org/pdf/2407.14937v1)

**Authors**: Apurv Verma, Satyapriya Krishna, Sebastian Gehrmann, Madhavan Seshadri, Anu Pradhan, Tom Ault, Leslie Barrett, David Rabinowitz, John Doucette, NhatHai Phan

**Abstract**: Creating secure and resilient applications with large language models (LLM) requires anticipating, adjusting to, and countering unforeseen threats. Red-teaming has emerged as a critical technique for identifying vulnerabilities in real-world LLM implementations. This paper presents a detailed threat model and provides a systematization of knowledge (SoK) of red-teaming attacks on LLMs. We develop a taxonomy of attacks based on the stages of the LLM development and deployment process and extract various insights from previous research. In addition, we compile methods for defense and practical red-teaming strategies for practitioners. By delineating prominent attack motifs and shedding light on various entry points, this paper provides a framework for improving the security and robustness of LLM-based systems.

摘要: 使用大型语言模型（LLM）创建安全且有弹性的应用程序需要预测、调整和应对不可预见的威胁。红色团队已成为识别现实世界LLM实施中漏洞的关键技术。本文提出了一个详细的威胁模型，并提供了对LLM的红色团队攻击的知识系统化（SoK）。我们根据LLM开发和部署过程的阶段开发攻击分类，并从之前的研究中提取各种见解。此外，我们还为从业者编写了防御方法和实用的红色团队策略。通过描述突出的攻击主题并揭示各种切入点，本文提供了一个框架来提高基于LLM的系统的安全性和稳健性。



## **30. DistillSeq: A Framework for Safety Alignment Testing in Large Language Models using Knowledge Distillation**

DistillSeq：使用知识蒸馏在大型语言模型中进行安全一致测试的框架 cs.SE

**SubmitDate**: 2024-07-20    [abs](http://arxiv.org/abs/2407.10106v3) [paper-pdf](http://arxiv.org/pdf/2407.10106v3)

**Authors**: Mingke Yang, Yuqi Chen, Yi Liu, Ling Shi

**Abstract**: Large Language Models (LLMs) have showcased their remarkable capabilities in diverse domains, encompassing natural language understanding, translation, and even code generation. The potential for LLMs to generate harmful content is a significant concern. This risk necessitates rigorous testing and comprehensive evaluation of LLMs to ensure safe and responsible use. However, extensive testing of LLMs requires substantial computational resources, making it an expensive endeavor. Therefore, exploring cost-saving strategies during the testing phase is crucial to balance the need for thorough evaluation with the constraints of resource availability. To address this, our approach begins by transferring the moderation knowledge from an LLM to a small model. Subsequently, we deploy two distinct strategies for generating malicious queries: one based on a syntax tree approach, and the other leveraging an LLM-based method. Finally, our approach incorporates a sequential filter-test process designed to identify test cases that are prone to eliciting toxic responses. Our research evaluated the efficacy of DistillSeq across four LLMs: GPT-3.5, GPT-4.0, Vicuna-13B, and Llama-13B. In the absence of DistillSeq, the observed attack success rates on these LLMs stood at 31.5% for GPT-3.5, 21.4% for GPT-4.0, 28.3% for Vicuna-13B, and 30.9% for Llama-13B. However, upon the application of DistillSeq, these success rates notably increased to 58.5%, 50.7%, 52.5%, and 54.4%, respectively. This translated to an average escalation in attack success rate by a factor of 93.0% when compared to scenarios without the use of DistillSeq. Such findings highlight the significant enhancement DistillSeq offers in terms of reducing the time and resource investment required for effectively testing LLMs.

摘要: 大型语言模型(LLM)已经在不同的领域展示了它们非凡的能力，包括自然语言理解、翻译，甚至代码生成。低密度脂蛋白产生有害内容的可能性是一个重大关切。这种风险需要对LLMS进行严格的测试和全面评估，以确保安全和负责任地使用。然而，大规模的LLMS测试需要大量的计算资源，这使得它成为一项昂贵的工作。因此，在测试阶段探索节约成本的策略对于平衡彻底评估的需要和资源可用性的限制至关重要。为了解决这个问题，我们的方法首先将适度知识从LLM转移到一个小模型。随后，我们部署了两种不同的策略来生成恶意查询：一种基于语法树方法，另一种利用基于LLM的方法。最后，我们的方法结合了一个顺序的过滤测试过程，旨在识别容易引发有毒反应的测试用例。我们的研究评估了DistillSeq在四种低密度脂蛋白上的疗效：GPT-3.5、GPT-4.0、Vicuna-13B和Llama-13B。在没有DistillSeq的情况下，观察到的对这些LLMS的攻击成功率GPT-3.5为31.5%，GPT-4.0为21.4%，Vicuna-13B为28.3%，Llama-13B为30.9%。然而，在应用DistillSeq后，这些成功率分别显著增加到58.5%、50.7%、52.5%和54.4%。与未使用DistillSeq的情况相比，这意味着攻击成功率平均提升了93.0%。这些发现突出了DistillSeq在减少有效测试LLM所需的时间和资源投资方面所提供的显著增强。



## **31. Retrieval Augmented Generation Integrated Large Language Models in Smart Contract Vulnerability Detection**

智能合同漏洞检测中的检索增强生成集成大型语言模型 cs.CR

17 pages, 3 figures, 4 tables

**SubmitDate**: 2024-07-20    [abs](http://arxiv.org/abs/2407.14838v1) [paper-pdf](http://arxiv.org/pdf/2407.14838v1)

**Authors**: Jeffy Yu

**Abstract**: The rapid growth of Decentralized Finance (DeFi) has been accompanied by substantial financial losses due to smart contract vulnerabilities, underscoring the critical need for effective security auditing. With attacks becoming more frequent, the necessity and demand for auditing services has escalated. This especially creates a financial burden for independent developers and small businesses, who often have limited available funding for these services. Our study builds upon existing frameworks by integrating Retrieval-Augmented Generation (RAG) with large language models (LLMs), specifically employing GPT-4-1106 for its 128k token context window. We construct a vector store of 830 known vulnerable contracts, leveraging Pinecone for vector storage, OpenAI's text-embedding-ada-002 for embeddings, and LangChain to construct the RAG-LLM pipeline. Prompts were designed to provide a binary answer for vulnerability detection. We first test 52 smart contracts 40 times each against a provided vulnerability type, verifying the replicability and consistency of the RAG-LLM. Encouraging results were observed, with a 62.7% success rate in guided detection of vulnerabilities. Second, we challenge the model under a "blind" audit setup, without the vulnerability type provided in the prompt, wherein 219 contracts undergo 40 tests each. This setup evaluates the general vulnerability detection capabilities without hinted context assistance. Under these conditions, a 60.71% success rate was observed. While the results are promising, we still emphasize the need for human auditing at this time. We provide this study as a proof of concept for a cost-effective smart contract auditing process, moving towards democratic access to security.

摘要: 去中心化金融(Defi)的快速增长伴随着由于智能合同漏洞而造成的大量财务损失，突显了对有效安全审计的迫切需要。随着攻击变得更加频繁，对审计服务的必要性和需求也不断升级。这尤其给独立开发商和小企业带来了财务负担，他们提供这些服务的资金往往有限。我们的研究建立在现有框架的基础上，将检索-增强生成(RAG)与大型语言模型(LLMS)相结合，特别是将GPT-4-1106用于其128k令牌上下文窗口。我们构建了一个包含830个已知易受攻击合约的向量库，利用Pinecone进行向量存储，利用OpenAI的Text-Embedding-ada-002进行嵌入，并利用LangChain构建RAG-LLM管道。提示旨在为漏洞检测提供二进制答案。我们首先针对提供的漏洞类型对52个智能合约进行了40次测试，验证了RAG-LLM的可复制性和一致性。观察到了令人鼓舞的结果，引导检测漏洞的成功率为62.7%。其次，我们在“盲目”的审计设置下挑战模型，没有提示中提供的漏洞类型，其中219个合同每个都要接受40个测试。此设置在没有提示上下文帮助的情况下评估常规漏洞检测功能。在此条件下，成功率为60.71%。虽然结果是有希望的，但我们仍然强调在这个时候进行人力审计的必要性。我们提供这项研究作为经济高效的智能合同审计流程的概念证明，朝着民主的安全访问方向发展。



## **32. Human-Interpretable Adversarial Prompt Attack on Large Language Models with Situational Context**

具有情境上下文的大型语言模型的人类可解释对抗提示攻击 cs.CL

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2407.14644v1) [paper-pdf](http://arxiv.org/pdf/2407.14644v1)

**Authors**: Nilanjana Das, Edward Raff, Manas Gaur

**Abstract**: Previous research on testing the vulnerabilities in Large Language Models (LLMs) using adversarial attacks has primarily focused on nonsensical prompt injections, which are easily detected upon manual or automated review (e.g., via byte entropy). However, the exploration of innocuous human-understandable malicious prompts augmented with adversarial injections remains limited. In this research, we explore converting a nonsensical suffix attack into a sensible prompt via a situation-driven contextual re-writing. This allows us to show suffix conversion without any gradients, using only LLMs to perform the attacks, and thus better understand the scope of possible risks. We combine an independent, meaningful adversarial insertion and situations derived from movies to check if this can trick an LLM. The situations are extracted from the IMDB dataset, and prompts are defined following a few-shot chain-of-thought prompting. Our approach demonstrates that a successful situation-driven attack can be executed on both open-source and proprietary LLMs. We find that across many LLMs, as few as 1 attempt produces an attack and that these attacks transfer between LLMs. The link to our code is available at \url{https://anonymous.4open.science/r/Situation-Driven-Adversarial-Attacks-7BB1/README.md}.

摘要: 之前关于使用对抗性攻击测试大型语言模型(LLM)中的漏洞的研究主要集中在无意义的提示注入上，这些注入很容易通过手动或自动审查(例如，通过字节熵)检测到。然而，通过恶意注入增强无害的人类可理解的恶意提示的探索仍然有限。在这项研究中，我们探索通过情景驱动的语境重写将无意义的后缀攻击转化为合理的提示。这使我们能够显示没有任何梯度的后缀转换，仅使用LLM来执行攻击，从而更好地了解可能风险的范围。我们结合了一个独立的、有意义的敌意插入和来自电影的情况来检查这是否可以欺骗LLM。情况是从IMDB数据集中提取的，提示是在几个镜头的思维链提示之后定义的。我们的方法表明，成功的情境驱动攻击可以在开源和专有LLM上执行。我们发现，在许多LLM中，只有1次尝试就会产生攻击，并且这些攻击会在LLM之间传输。有关我们代码的链接，请访问\url{https://anonymous.4open.science/r/Situation-Driven-Adversarial-Attacks-7BB1/README.md}.



## **33. CVE-LLM : Automatic vulnerability evaluation in medical device industry using large language models**

CVE-LLM：使用大型语言模型在医疗器械行业中进行自动漏洞评估 cs.CL

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2407.14640v1) [paper-pdf](http://arxiv.org/pdf/2407.14640v1)

**Authors**: Rikhiya Ghosh, Oladimeji Farri, Hans-Martin von Stockhausen, Martin Schmitt, George Marica Vasile

**Abstract**: The healthcare industry is currently experiencing an unprecedented wave of cybersecurity attacks, impacting millions of individuals. With the discovery of thousands of vulnerabilities each month, there is a pressing need to drive the automation of vulnerability assessment processes for medical devices, facilitating rapid mitigation efforts. Generative AI systems have revolutionized various industries, offering unparalleled opportunities for automation and increased efficiency. This paper presents a solution leveraging Large Language Models (LLMs) to learn from historical evaluations of vulnerabilities for the automatic assessment of vulnerabilities in the medical devices industry. This approach is applied within the portfolio of a single manufacturer, taking into account device characteristics, including existing security posture and controls. The primary contributions of this paper are threefold. Firstly, it provides a detailed examination of the best practices for training a vulnerability Language Model (LM) in an industrial context. Secondly, it presents a comprehensive comparison and insightful analysis of the effectiveness of Language Models in vulnerability assessment. Finally, it proposes a new human-in-the-loop framework to expedite vulnerability evaluation processes.

摘要: 医疗保健行业目前正经历一波前所未有的网络安全攻击浪潮，影响着数百万人。随着每月发现数以千计的漏洞，迫切需要推动医疗设备脆弱性评估过程的自动化，促进快速缓解工作。生产性人工智能系统已经给各个行业带来了革命性的变化，为自动化和提高效率提供了无与伦比的机会。本文提出了一种利用大型语言模型(LLM)来学习历史漏洞评估的解决方案，用于医疗器械行业漏洞的自动评估。该方法在单个制造商的产品组合中应用，并考虑到设备特性，包括现有的安全状态和控制。本文的主要贡献有三个方面。首先，它提供了在工业环境中训练脆弱性语言模型(LM)的最佳实践的详细检查。其次，对语言模型在脆弱性评估中的有效性进行了全面的比较和深入的分析。最后，提出了一种新的人在环中框架，以加快脆弱性评估过程。



## **34. Uncertainty is Fragile: Manipulating Uncertainty in Large Language Models**

不确定性是脆弱的：在大型语言模型中操纵不确定性 cs.CL

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2407.11282v3) [paper-pdf](http://arxiv.org/pdf/2407.11282v3)

**Authors**: Qingcheng Zeng, Mingyu Jin, Qinkai Yu, Zhenting Wang, Wenyue Hua, Zihao Zhou, Guangyan Sun, Yanda Meng, Shiqing Ma, Qifan Wang, Felix Juefei-Xu, Kaize Ding, Fan Yang, Ruixiang Tang, Yongfeng Zhang

**Abstract**: Large Language Models (LLMs) are employed across various high-stakes domains, where the reliability of their outputs is crucial. One commonly used method to assess the reliability of LLMs' responses is uncertainty estimation, which gauges the likelihood of their answers being correct. While many studies focus on improving the accuracy of uncertainty estimations for LLMs, our research investigates the fragility of uncertainty estimation and explores potential attacks. We demonstrate that an attacker can embed a backdoor in LLMs, which, when activated by a specific trigger in the input, manipulates the model's uncertainty without affecting the final output. Specifically, the proposed backdoor attack method can alter an LLM's output probability distribution, causing the probability distribution to converge towards an attacker-predefined distribution while ensuring that the top-1 prediction remains unchanged. Our experimental results demonstrate that this attack effectively undermines the model's self-evaluation reliability in multiple-choice questions. For instance, we achieved a 100 attack success rate (ASR) across three different triggering strategies in four models. Further, we investigate whether this manipulation generalizes across different prompts and domains. This work highlights a significant threat to the reliability of LLMs and underscores the need for future defenses against such attacks. The code is available at https://github.com/qcznlp/uncertainty_attack.

摘要: 大型语言模型(LLM)被用于各种高风险领域，在这些领域中，其输出的可靠性至关重要。评估LLMS回答可靠性的一种常用方法是不确定性估计，它衡量他们回答正确的可能性。虽然许多研究都集中在提高LLMS不确定性估计的准确性上，但我们的研究调查了不确定性估计的脆弱性，并探索了潜在的攻击。我们演示了攻击者可以在LLMS中嵌入后门，当它被输入中的特定触发器激活时，在不影响最终输出的情况下操纵模型的不确定性。具体地说，提出的后门攻击方法可以改变LLM的输出概率分布，使概率分布收敛到攻击者预定义的分布，同时确保TOP-1预测保持不变。我们的实验结果表明，这种攻击有效地破坏了该模型在选择题中的自我评价可靠性。例如，我们在四个模型中的三种不同触发策略中实现了100%的攻击成功率(ASR)。此外，我们还研究了这种操作是否适用于不同的提示和域。这项工作突出了对LLMS可靠性的重大威胁，并强调了今后对这类攻击采取防御措施的必要性。代码可在https://github.com/qcznlp/uncertainty_attack.上获得



## **35. Are you still on track!? Catching LLM Task Drift with Activations**

你还在正轨上吗！？通过激活捕捉LLM任务漂移 cs.CR

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2406.00799v4) [paper-pdf](http://arxiv.org/pdf/2406.00799v4)

**Authors**: Sahar Abdelnabi, Aideen Fay, Giovanni Cherubin, Ahmed Salem, Mario Fritz, Andrew Paverd

**Abstract**: Large Language Models (LLMs) are routinely used in retrieval-augmented applications to orchestrate tasks and process inputs from users and other sources. These inputs, even in a single LLM interaction, can come from a variety of sources, of varying trustworthiness and provenance. This opens the door to prompt injection attacks, where the LLM receives and acts upon instructions from supposedly data-only sources, thus deviating from the user's original instructions. We define this as task drift, and we propose to catch it by scanning and analyzing the LLM's activations. We compare the LLM's activations before and after processing the external input in order to detect whether this input caused instruction drift. We develop two probing methods and find that simply using a linear classifier can detect drift with near perfect ROC AUC on an out-of-distribution test set. We show that this approach generalizes surprisingly well to unseen task domains, such as prompt injections, jailbreaks, and malicious instructions, without being trained on any of these attacks. Our setup does not require any modification of the LLM (e.g., fine-tuning) or any text generation, thus maximizing deployability and cost efficiency and avoiding reliance on unreliable model output. To foster future research on activation-based task inspection, decoding, and interpretability, we will release our large-scale TaskTracker toolkit, comprising a dataset of over 500K instances, representations from 5 SoTA language models, and inspection tools.

摘要: 大型语言模型(LLM)通常用于检索增强的应用程序中，以协调任务并处理来自用户和其他来源的输入。这些输入，即使是在单个LLM交互中，也可以来自各种来源，具有不同的可信度和出处。这为即时注入攻击打开了大门，在这种情况下，LLM接收来自假定仅限数据的来源的指令并对其采取行动，从而偏离用户的原始指令。我们将其定义为任务漂移，并建议通过扫描和分析LLM的激活来捕获它。我们比较LLM在处理外部输入之前和之后的激活，以检测该输入是否导致指令漂移。我们开发了两种探测方法，发现简单地使用线性分类器可以在非分布测试集上以接近完美的ROC AUC来检测漂移。我们表明，这种方法对于看不见的任务领域(如提示注入、越狱和恶意指令)的泛化效果出奇地好，而且没有接受过任何这些攻击的培训。我们的设置不需要对LLM进行任何修改(例如，微调)或任何文本生成，从而最大限度地提高可部署性和成本效益，并避免依赖不可靠的模型输出。为了促进未来对基于激活的任务检测、解码和可解释性的研究，我们将发布我们的大型TaskTracker工具包，其中包括超过50万个实例的数据集、来自5个SOTA语言模型的表示和检测工具。



## **36. Watermark Smoothing Attacks against Language Models**

针对语言模型的水印平滑攻击 cs.LG

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2407.14206v1) [paper-pdf](http://arxiv.org/pdf/2407.14206v1)

**Authors**: Hongyan Chang, Hamed Hassani, Reza Shokri

**Abstract**: Watermarking is a technique used to embed a hidden signal in the probability distribution of text generated by large language models (LLMs), enabling attribution of the text to the originating model. We introduce smoothing attacks and show that existing watermarking methods are not robust against minor modifications of text. An adversary can use weaker language models to smooth out the distribution perturbations caused by watermarks without significantly compromising the quality of the generated text. The modified text resulting from the smoothing attack remains close to the distribution of text that the original model (without watermark) would have produced. Our attack reveals a fundamental limitation of a wide range of watermarking techniques.

摘要: 水印是一种用于将隐藏信号嵌入大型语言模型（LLM）生成的文本的概率分布中的技术，从而将文本归因于原始模型。我们引入了平滑攻击，并表明现有的水印方法对文本的微小修改并不鲁棒。对手可以使用较弱的语言模型来平滑水印引起的分布扰动，而不会显着损害生成文本的质量。平滑攻击产生的修改文本仍然接近原始模型（没有水印）产生的文本分布。我们的攻击揭示了广泛水印技术的根本局限性。



## **37. A Survey of Backdoor Attacks and Defenses on Large Language Models: Implications for Security Measures**

大型语言模型后门攻击和防御的调查：对安全措施的影响 cs.CR

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2406.06852v3) [paper-pdf](http://arxiv.org/pdf/2406.06852v3)

**Authors**: Shuai Zhao, Meihuizi Jia, Zhongliang Guo, Leilei Gan, Xiaoyu Xu, Jie Fu, Yichao Feng, Fengjun Pan, Luu Anh Tuan

**Abstract**: The large language models (LLMs), which bridge the gap between human language understanding and complex problem-solving, achieve state-of-the-art performance on several NLP tasks, particularly in few-shot and zero-shot settings. Despite the demonstrable efficacy of LMMs, due to constraints on computational resources, users have to engage with open-source language models or outsource the entire training process to third-party platforms. However, research has demonstrated that language models are susceptible to potential security vulnerabilities, particularly in backdoor attacks. Backdoor attacks are designed to introduce targeted vulnerabilities into language models by poisoning training samples or model weights, allowing attackers to manipulate model responses through malicious triggers. While existing surveys on backdoor attacks provide a comprehensive overview, they lack an in-depth examination of backdoor attacks specifically targeting LLMs. To bridge this gap and grasp the latest trends in the field, this paper presents a novel perspective on backdoor attacks for LLMs by focusing on fine-tuning methods. Specifically, we systematically classify backdoor attacks into three categories: full-parameter fine-tuning, parameter-efficient fine-tuning, and attacks without fine-tuning. Based on insights from a substantial review, we also discuss crucial issues for future research on backdoor attacks, such as further exploring attack algorithms that do not require fine-tuning, or developing more covert attack algorithms.

摘要: 大型语言模型(LLM)架起了人类语言理解和复杂问题解决之间的桥梁，在几个NLP任务上实现了最先进的性能，特别是在少镜头和零镜头的情况下。尽管LMM具有明显的功效，但由于计算资源的限制，用户不得不使用开放源码语言模型或将整个培训过程外包给第三方平台。然而，研究表明，语言模型容易受到潜在的安全漏洞的影响，特别是在后门攻击中。后门攻击旨在通过毒化训练样本或模型权重，将有针对性的漏洞引入语言模型，允许攻击者通过恶意触发器操纵模型响应。虽然现有的关于后门攻击的调查提供了全面的概述，但它们缺乏对专门针对LLM的后门攻击的深入检查。为了弥补这一差距，掌握该领域的最新趋势，本文提出了一种新的视角来研究针对LLMS的后门攻击，重点是微调方法。具体来说，我们系统地将后门攻击分为三类：全参数微调、参数高效微调和未微调攻击。在大量综述的基础上，我们还讨论了未来后门攻击研究的关键问题，如进一步探索不需要微调的攻击算法，或开发更隐蔽的攻击算法。



## **38. Exploiting Uncommon Text-Encoded Structures for Automated Jailbreaks in LLMs**

利用不常见的文本编码结构进行LLC中的自动越狱 cs.CL

12 pages, 4 figures

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2406.08754v2) [paper-pdf](http://arxiv.org/pdf/2406.08754v2)

**Authors**: Bangxin Li, Hengrui Xing, Chao Huang, Jin Qian, Huangqing Xiao, Linfeng Feng, Cong Tian

**Abstract**: Large Language Models (LLMs) are widely used in natural language processing but face the risk of jailbreak attacks that maliciously induce them to generate harmful content. Existing jailbreak attacks, including character-level and context-level attacks, mainly focus on the prompt of the plain text without specifically exploring the significant influence of its structure. In this paper, we focus on studying how prompt structure contributes to the jailbreak attack. We introduce a novel structure-level attack method based on tail structures that are rarely used during LLM training, which we refer to as Uncommon Text-Encoded Structure (UTES). We extensively study 12 UTESs templates and 6 obfuscation methods to build an effective automated jailbreak tool named StructuralSleight that contains three escalating attack strategies: Structural Attack, Structural and Character/Context Obfuscation Attack, and Fully Obfuscated Structural Attack. Extensive experiments on existing LLMs show that StructuralSleight significantly outperforms baseline methods. In particular, the attack success rate reaches 94.62\% on GPT-4o, which has not been addressed by state-of-the-art techniques.

摘要: 大语言模型在自然语言处理中被广泛使用，但面临着越狱攻击的风险，这些攻击会恶意诱导它们生成有害内容。现有的越狱攻击，包括字符级攻击和语境级攻击，主要集中在明文的提示上，没有具体探讨其结构的重大影响。本文主要研究提示结构在越狱攻击中的作用。提出了一种基于LLM训练中很少使用的尾部结构的结构级攻击方法，称为非公共文本编码结构(UTES)。我们深入研究了12个UTE模板和6种混淆方法，构建了一个有效的自动化越狱工具StructuralSleight，它包含三种逐步升级的攻击策略：结构攻击、结构和字符/上下文混淆攻击和完全混淆结构攻击。在现有LLMS上的大量实验表明，StructuralSleight的性能明显优于基线方法。特别是，在GPT-40上的攻击成功率达到了94.62\%，这是最新技术还没有解决的问题。



## **39. Jailbreaking Black Box Large Language Models in Twenty Queries**

二十分钟内越狱黑匣子大型语言模型 cs.LG

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2310.08419v4) [paper-pdf](http://arxiv.org/pdf/2310.08419v4)

**Authors**: Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, Eric Wong

**Abstract**: There is growing interest in ensuring that large language models (LLMs) align with human values. However, the alignment of such models is vulnerable to adversarial jailbreaks, which coax LLMs into overriding their safety guardrails. The identification of these vulnerabilities is therefore instrumental in understanding inherent weaknesses and preventing future misuse. To this end, we propose Prompt Automatic Iterative Refinement (PAIR), an algorithm that generates semantic jailbreaks with only black-box access to an LLM. PAIR -- which is inspired by social engineering attacks -- uses an attacker LLM to automatically generate jailbreaks for a separate targeted LLM without human intervention. In this way, the attacker LLM iteratively queries the target LLM to update and refine a candidate jailbreak. Empirically, PAIR often requires fewer than twenty queries to produce a jailbreak, which is orders of magnitude more efficient than existing algorithms. PAIR also achieves competitive jailbreaking success rates and transferability on open and closed-source LLMs, including GPT-3.5/4, Vicuna, and Gemini.

摘要: 人们对确保大型语言模型(LLM)与人类价值观保持一致的兴趣与日俱增。然而，这类模型的调整很容易受到对抗性越狱的影响，这会诱使低收入国家凌驾于他们的安全护栏之上。因此，确定这些漏洞有助于了解固有的弱点并防止今后的滥用。为此，我们提出了即时自动迭代求精(Pair)，这是一种仅通过黑盒访问LLM来生成语义越狱的算法。Pair受到社会工程攻击的启发，它使用攻击者LLM自动为单独的目标LLM生成越狱，而无需人工干预。通过这种方式，攻击者LLM迭代地查询目标LLM以更新和改进候选越狱。根据经验，Pair通常只需要不到20次查询就可以产生越狱，这比现有算法的效率高出几个数量级。Pair还在开放和封闭源代码的LLM上实现了具有竞争力的越狱成功率和可转移性，包括GPT-3.5/4、维库纳和双子座。



## **40. Black-Box Opinion Manipulation Attacks to Retrieval-Augmented Generation of Large Language Models**

对大型语言模型检索增强生成的黑匣子观点操纵攻击 cs.CL

10 pages, 3 figures, under review

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.13757v1) [paper-pdf](http://arxiv.org/pdf/2407.13757v1)

**Authors**: Zhuo Chen, Jiawei Liu, Haotan Liu, Qikai Cheng, Fan Zhang, Wei Lu, Xiaozhong Liu

**Abstract**: Retrieval-Augmented Generation (RAG) is applied to solve hallucination problems and real-time constraints of large language models, but it also induces vulnerabilities against retrieval corruption attacks. Existing research mainly explores the unreliability of RAG in white-box and closed-domain QA tasks. In this paper, we aim to reveal the vulnerabilities of Retrieval-Enhanced Generative (RAG) models when faced with black-box attacks for opinion manipulation. We explore the impact of such attacks on user cognition and decision-making, providing new insight to enhance the reliability and security of RAG models. We manipulate the ranking results of the retrieval model in RAG with instruction and use these results as data to train a surrogate model. By employing adversarial retrieval attack methods to the surrogate model, black-box transfer attacks on RAG are further realized. Experiments conducted on opinion datasets across multiple topics show that the proposed attack strategy can significantly alter the opinion polarity of the content generated by RAG. This demonstrates the model's vulnerability and, more importantly, reveals the potential negative impact on user cognition and decision-making, making it easier to mislead users into accepting incorrect or biased information.

摘要: 检索增强生成(RAG)被应用于解决大型语言模型的幻觉问题和实时约束，但它也导致了对检索破坏攻击的脆弱性。已有研究主要探讨RAG在白盒和封闭域QA任务中的不可靠性。在本文中，我们旨在揭示检索增强生成(RAG)模型在面对意见操纵黑盒攻击时的脆弱性。我们探讨了此类攻击对用户认知和决策的影响，为提高RAG模型的可靠性和安全性提供了新的见解。我们通过指令对检索模型在RAG中的排序结果进行操作，并将这些结果作为数据来训练代理模型。通过对代理模型采用对抗性检索攻击方法，进一步实现了对RAG的黑箱转移攻击。在多个主题的观点数据集上进行的实验表明，所提出的攻击策略可以显著改变RAG生成的内容的观点极性。这表明了该模型的脆弱性，更重要的是，揭示了对用户认知和决策的潜在负面影响，使其更容易误导用户接受不正确或有偏见的信息。



## **41. Prover-Verifier Games improve legibility of LLM outputs**

证明者-验证者游戏提高了LLM输出的清晰度 cs.CL

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.13692v1) [paper-pdf](http://arxiv.org/pdf/2407.13692v1)

**Authors**: Jan Hendrik Kirchner, Yining Chen, Harri Edwards, Jan Leike, Nat McAleese, Yuri Burda

**Abstract**: One way to increase confidence in the outputs of Large Language Models (LLMs) is to support them with reasoning that is clear and easy to check -- a property we call legibility. We study legibility in the context of solving grade-school math problems and show that optimizing chain-of-thought solutions only for answer correctness can make them less legible. To mitigate the loss in legibility, we propose a training algorithm inspired by Prover-Verifier Game from Anil et al. (2021). Our algorithm iteratively trains small verifiers to predict solution correctness, "helpful" provers to produce correct solutions that the verifier accepts, and "sneaky" provers to produce incorrect solutions that fool the verifier. We find that the helpful prover's accuracy and the verifier's robustness to adversarial attacks increase over the course of training. Furthermore, we show that legibility training transfers to time-constrained humans tasked with verifying solution correctness. Over course of LLM training human accuracy increases when checking the helpful prover's solutions, and decreases when checking the sneaky prover's solutions. Hence, training for checkability by small verifiers is a plausible technique for increasing output legibility. Our results suggest legibility training against small verifiers as a practical avenue for increasing legibility of large LLMs to humans, and thus could help with alignment of superhuman models.

摘要: 增加对大型语言模型(LLM)输出的信心的一种方法是用清晰且易于检查的推理来支持它们--我们称之为易读性。我们在解决小学数学问题的背景下研究了易读性，并表明只为了答案的正确性而优化思维链解决方案会降低它们的易读性。为了减少易读性的损失，我们提出了一种受Anil等人的Prover-Verator游戏启发的训练算法。(2021年)。我们的算法迭代地训练小的验证者来预测解决方案的正确性，“有帮助的”验证者来产生验证者接受的正确的解决方案，而“偷偷摸摸”的验证者产生愚弄验证者的不正确的解决方案。我们发现，随着训练过程的进行，有益证明者的准确率和验证者对敌意攻击的健壮性都有所提高。此外，我们还表明，易读性训练转移到负责验证解决方案正确性的时间受限的人身上。在LLM训练过程中，当检查有用的证明者的解时，人类的准确率提高，而当检查偷偷摸摸的证明者的解时，人类的准确率降低。因此，由小型验证员进行可校验性培训是提高输出清晰度的一种可行的技术。我们的结果表明，针对小验证者的易读性训练是提高大型LLM对人类易读性的实用途径，因此可能有助于超人模型的对齐。



## **42. Turning Generative Models Degenerate: The Power of Data Poisoning Attacks**

使生成模型退化：数据中毒攻击的力量 cs.CR

18 pages, 11 figures

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.12281v2) [paper-pdf](http://arxiv.org/pdf/2407.12281v2)

**Authors**: Shuli Jiang, Swanand Ravindra Kadhe, Yi Zhou, Farhan Ahmed, Ling Cai, Nathalie Baracaldo

**Abstract**: The increasing use of large language models (LLMs) trained by third parties raises significant security concerns. In particular, malicious actors can introduce backdoors through poisoning attacks to generate undesirable outputs. While such attacks have been extensively studied in image domains and classification tasks, they remain underexplored for natural language generation (NLG) tasks. To address this gap, we conduct an investigation of various poisoning techniques targeting the LLM's fine-tuning phase via prefix-tuning, a Parameter Efficient Fine-Tuning (PEFT) method. We assess their effectiveness across two generative tasks: text summarization and text completion; and we also introduce new metrics to quantify the success and stealthiness of such NLG poisoning attacks. Through our experiments, we find that the prefix-tuning hyperparameters and trigger designs are the most crucial factors to influence attack success and stealthiness. Moreover, we demonstrate that existing popular defenses are ineffective against our poisoning attacks. Our study presents the first systematic approach to understanding poisoning attacks targeting NLG tasks during fine-tuning via PEFT across a wide range of triggers and attack settings. We hope our findings will aid the AI security community in developing effective defenses against such threats.

摘要: 越来越多地使用由第三方培训的大型语言模型(LLM)引起了严重的安全问题。特别是，恶意攻击者可以通过投毒攻击引入后门，以生成不良输出。虽然这类攻击已经在图像域和分类任务中得到了广泛的研究，但它们在自然语言生成(NLG)任务中仍然没有得到充分的研究。为了解决这一差距，我们通过前缀调整(一种参数高效微调(PEFT)方法)对针对LLM微调阶段的各种中毒技术进行了调查。我们通过两个生成性任务来评估它们的有效性：文本摘要和文本完成；我们还引入了新的度量来量化此类NLG中毒攻击的成功和隐蔽性。通过实验，我们发现前缀调整超参数和触发器设计是影响攻击成功和隐身的最关键因素。此外，我们证明，现有的大众防御系统对我们的中毒攻击是无效的。我们的研究提出了第一种系统的方法来理解针对NLG任务的中毒攻击，在通过PEFT对广泛的触发器和攻击设置进行微调期间。我们希望我们的发现将有助于人工智能安全界开发针对此类威胁的有效防御措施。



## **43. Can LLMs Patch Security Issues?**

LLM可以解决安全问题吗？ cs.CR

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2312.00024v4) [paper-pdf](http://arxiv.org/pdf/2312.00024v4)

**Authors**: Kamel Alrashedy, Abdullah Aljasser, Pradyumna Tambwekar, Matthew Gombolay

**Abstract**: Large Language Models (LLMs) have shown impressive proficiency in code generation. Unfortunately, these models share a weakness with their human counterparts: producing code that inadvertently has security vulnerabilities. These vulnerabilities could allow unauthorized attackers to access sensitive data or systems, which is unacceptable for safety-critical applications. In this work, we propose Feedback-Driven Security Patching (FDSP), where LLMs automatically refine generated, vulnerable code. Our approach leverages automatic static code analysis to empower the LLM to generate and implement potential solutions to address vulnerabilities. We address the research communitys needs for safe code generation by introducing a large-scale dataset, PythonSecurityEval, covering the diversity of real-world applications, including databases, websites and operating systems. We empirically validate that FDSP outperforms prior work that uses self-feedback from LLMs by up to 17.6% through our procedure that injects targeted, external feedback. Code and data are available at \url{https://github.com/Kamel773/LLM-code-refine}

摘要: 大型语言模型(LLM)在代码生成方面表现出令人印象深刻的熟练程度。不幸的是，这些模型与它们的人类同行有一个共同的弱点：生成的代码无意中存在安全漏洞。这些漏洞可能允许未经授权的攻击者访问敏感数据或系统，这对于安全关键型应用程序是不可接受的。在这项工作中，我们提出了反馈驱动的安全修补(FDSP)，其中LLMS自动精炼生成的易受攻击的代码。我们的方法利用自动静态代码分析来支持LLM生成和实施潜在的解决方案来应对漏洞。我们通过引入大规模数据集PythonSecurityEval来满足研究团体对安全代码生成的需求，该数据集涵盖了包括数据库、网站和操作系统在内的各种现实应用程序。我们通过注入有针对性的外部反馈的程序，经验性地验证了FDSP的性能比使用LLMS自我反馈的先前工作高出17.6%。有关代码和数据，请访问\url{https://github.com/Kamel773/LLM-code-refine}



## **44. Counterfactual Explainable Incremental Prompt Attack Analysis on Large Language Models**

大型语言模型的反事实可解释增量提示攻击分析 cs.CR

23 pages, 6 figures

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2407.09292v2) [paper-pdf](http://arxiv.org/pdf/2407.09292v2)

**Authors**: Dong Shu, Mingyu Jin, Tianle Chen, Chong Zhang, Yongfeng Zhang

**Abstract**: This study sheds light on the imperative need to bolster safety and privacy measures in large language models (LLMs), such as GPT-4 and LLaMA-2, by identifying and mitigating their vulnerabilities through explainable analysis of prompt attacks. We propose Counterfactual Explainable Incremental Prompt Attack (CEIPA), a novel technique where we guide prompts in a specific manner to quantitatively measure attack effectiveness and explore the embedded defense mechanisms in these models. Our approach is distinctive for its capacity to elucidate the reasons behind the generation of harmful responses by LLMs through an incremental counterfactual methodology. By organizing the prompt modification process into four incremental levels: (word, sentence, character, and a combination of character and word) we facilitate a thorough examination of the susceptibilities inherent to LLMs. The findings from our study not only provide counterfactual explanation insight but also demonstrate that our framework significantly enhances the effectiveness of attack prompts.

摘要: 这项研究揭示了通过对提示攻击进行解释性分析来识别和缓解大型语言模型(LLM)(如GPT-4和LLAMA-2)中的安全和隐私措施的迫切需要。我们提出了反事实可解释增量提示攻击(CEIPA)，这是一种新的技术，我们以特定的方式引导提示来定量地衡量攻击效果，并探索这些模型中嵌入的防御机制。我们的方法的独特之处在于，它能够通过循序渐进的反事实方法论来阐明小岛屿发展中国家产生有害反应背后的原因。通过将即时修改过程组织成四个递增级别：(单词、句子、字符和字符和单词的组合)，我们有助于彻底检查LLMS固有的易感性。我们的研究结果不仅提供了反事实的解释洞察力，而且证明了我们的框架显着提高了攻击提示的有效性。



## **45. Muting Whisper: A Universal Acoustic Adversarial Attack on Speech Foundation Models**

静音低语：对语音基础模型的通用声学对抗攻击 cs.CL

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2405.06134v2) [paper-pdf](http://arxiv.org/pdf/2405.06134v2)

**Authors**: Vyas Raina, Rao Ma, Charles McGhee, Kate Knill, Mark Gales

**Abstract**: Recent developments in large speech foundation models like Whisper have led to their widespread use in many automatic speech recognition (ASR) applications. These systems incorporate `special tokens' in their vocabulary, such as $\texttt{<|endoftext|>}$, to guide their language generation process. However, we demonstrate that these tokens can be exploited by adversarial attacks to manipulate the model's behavior. We propose a simple yet effective method to learn a universal acoustic realization of Whisper's $\texttt{<|endoftext|>}$ token, which, when prepended to any speech signal, encourages the model to ignore the speech and only transcribe the special token, effectively `muting' the model. Our experiments demonstrate that the same, universal 0.64-second adversarial audio segment can successfully mute a target Whisper ASR model for over 97\% of speech samples. Moreover, we find that this universal adversarial audio segment often transfers to new datasets and tasks. Overall this work demonstrates the vulnerability of Whisper models to `muting' adversarial attacks, where such attacks can pose both risks and potential benefits in real-world settings: for example the attack can be used to bypass speech moderation systems, or conversely the attack can also be used to protect private speech data.

摘要: 像Whisper这样的大型语音基础模型的最新发展导致它们在许多自动语音识别(ASR)应用中被广泛使用。这些系统在其词汇表中加入了“特殊标记”，如$\exttt{<|endoftext|>}$，以指导其语言生成过程。然而，我们证明了这些令牌可以被敌意攻击利用来操纵模型的行为。我们提出了一种简单而有效的方法来学习Whisper的$\exttt{}$标记的通用声学实现，当该标记被预先添加到任何语音信号时，鼓励模型忽略语音而只转录特殊的标记，从而有效地抑制了模型。我们的实验表明，相同的、通用的0.64秒的对抗性音频片段可以成功地使目标Whisper ASR模型在97%以上的语音样本上静音。此外，我们发现这种普遍的对抗性音频片段经常转移到新的数据集和任务。总体而言，这项工作证明了Whisper模型对“静音”对手攻击的脆弱性，在现实世界中，这种攻击既可以带来风险，也可以带来潜在的好处：例如，攻击可以用来绕过语音调节系统，或者反过来，攻击也可以用来保护私人语音数据。



## **46. Security Matrix for Multimodal Agents on Mobile Devices: A Systematic and Proof of Concept Study**

移动设备上多模式代理的安全矩阵：系统性的概念验证研究 cs.CR

Preprint. Work in progress

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2407.09295v2) [paper-pdf](http://arxiv.org/pdf/2407.09295v2)

**Authors**: Yulong Yang, Xinshan Yang, Shuaidong Li, Chenhao Lin, Zhengyu Zhao, Chao Shen, Tianwei Zhang

**Abstract**: The rapid progress in the reasoning capability of the Multi-modal Large Language Models (MLLMs) has triggered the development of autonomous agent systems on mobile devices. MLLM-based mobile agent systems consist of perception, reasoning, memory, and multi-agent collaboration modules, enabling automatic analysis of user instructions and the design of task pipelines with only natural language and device screenshots as inputs. Despite the increased human-machine interaction efficiency, the security risks of MLLM-based mobile agent systems have not been systematically studied. Existing security benchmarks for agents mainly focus on Web scenarios, and the attack techniques against MLLMs are also limited in the mobile agent scenario. To close these gaps, this paper proposes a mobile agent security matrix covering 3 functional modules of the agent systems. Based on the security matrix, this paper proposes 4 realistic attack paths and verifies these attack paths through 8 attack methods. By analyzing the attack results, this paper reveals that MLLM-based mobile agent systems are not only vulnerable to multiple traditional attacks, but also raise new security concerns previously unconsidered. This paper highlights the need for security awareness in the design of MLLM-based systems and paves the way for future research on attacks and defense methods.

摘要: 多通道大语言模型(MLLMS)在推理能力上的快速发展引发了移动设备上自主代理系统的发展。基于MLLM的移动代理系统由感知、推理、记忆和多代理协作模块组成，支持用户指令的自动分析和任务流水线的设计，只需输入自然语言和设备截图。尽管提高了人机交互效率，但基于MLLM的移动代理系统的安全风险还没有得到系统的研究。现有的代理安全基准测试主要集中在Web场景中，针对移动代理场景中MLLMS的攻击技术也很有限。为了弥补这些差距，本文提出了一种覆盖代理系统3个功能模块的移动代理安全矩阵。基于安全矩阵，提出了4条真实的攻击路径，并通过8种攻击方法对这些攻击路径进行了验证。通过对攻击结果的分析，本文揭示了基于MLLM的移动代理系统不仅容易受到多种传统攻击的攻击，而且还提出了以前没有考虑到的新的安全问题。本文强调了在基于MLLM的系统设计中安全意识的必要性，并为未来攻击和防御方法的研究铺平了道路。



## **47. The First to Know: How Token Distributions Reveal Hidden Knowledge in Large Vision-Language Models?**

首先知道的：代币分布如何揭示大型视觉语言模型中隐藏的知识？ cs.CV

ECCV 2024. Project page: https://github.com/Qinyu-Allen-Zhao/LVLM-LP

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2403.09037v2) [paper-pdf](http://arxiv.org/pdf/2403.09037v2)

**Authors**: Qinyu Zhao, Ming Xu, Kartik Gupta, Akshay Asthana, Liang Zheng, Stephen Gould

**Abstract**: Large vision-language models (LVLMs), designed to interpret and respond to human instructions, occasionally generate hallucinated or harmful content due to inappropriate instructions. This study uses linear probing to shed light on the hidden knowledge at the output layers of LVLMs. We demonstrate that the logit distributions of the first tokens contain sufficient information to determine whether to respond to the instructions, including recognizing unanswerable visual questions, defending against jailbreaking attacks, and identifying deceptive questions. Such hidden knowledge is gradually lost in logits of subsequent tokens during response generation. Then, we illustrate a simple decoding strategy at the generation of the first token, effectively improving the generated content. In experiments, we find a few interesting insights: First, the CLIP model already contains a strong signal for solving these tasks, which indicates potential bias in the existing datasets. Second, we observe performance improvement by utilizing the first logit distributions on three additional tasks, including indicating uncertainty in math solving, mitigating hallucination, and image classification. Last, with the same training data, simply finetuning LVLMs improves models' performance but is still inferior to linear probing on these tasks.

摘要: 大型视觉语言模型(LVLM)旨在解释和响应人类的指令，有时会由于不适当的指令而产生幻觉或有害内容。这项研究使用线性探测来揭示LVLMS输出层的隐藏知识。我们证明了第一个令牌的Logit分布包含了足够的信息来确定是否响应指令，包括识别无法回答的可视问题、防御越狱攻击和识别欺骗性问题。在响应生成期间，这种隐藏的知识在后续令牌的登录中逐渐丢失。然后，我们在生成第一个令牌时说明了一种简单的解码策略，有效地改进了生成的内容。在实验中，我们发现了一些有趣的见解：首先，CLIP模型已经包含了解决这些任务的强烈信号，这表明现有数据集中存在潜在的偏差。其次，我们观察到在三个额外的任务中，利用第一个Logit分布，包括指示数学解题中的不确定性、减轻幻觉和图像分类，性能有所提高。最后，在相同的训练数据下，简单地对LVLM进行微调可以提高模型的性能，但在这些任务上仍然不如线性探测。



## **48. Revisiting the Adversarial Robustness of Vision Language Models: a Multimodal Perspective**

重新审视视觉语言模型的对抗鲁棒性：多模式视角 cs.CV

16 pages, 14 figures

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2404.19287v2) [paper-pdf](http://arxiv.org/pdf/2404.19287v2)

**Authors**: Wanqi Zhou, Shuanghao Bai, Qibin Zhao, Badong Chen

**Abstract**: Pretrained vision-language models (VLMs) like CLIP have shown impressive generalization performance across various downstream tasks, yet they remain vulnerable to adversarial attacks. While prior research has primarily concentrated on improving the adversarial robustness of image encoders to guard against attacks on images, the exploration of text-based and multimodal attacks has largely been overlooked. In this work, we initiate the first known and comprehensive effort to study adapting vision-language models for adversarial robustness under the multimodal attack. Firstly, we introduce a multimodal attack strategy and investigate the impact of different attacks. We then propose a multimodal contrastive adversarial training loss, aligning the clean and adversarial text embeddings with the adversarial and clean visual features, to enhance the adversarial robustness of both image and text encoders of CLIP. Extensive experiments on 15 datasets across two tasks demonstrate that our method significantly improves the adversarial robustness of CLIP. Interestingly, we find that the model fine-tuned against multimodal adversarial attacks exhibits greater robustness than its counterpart fine-tuned solely against image-based attacks, even in the context of image attacks, which may open up new possibilities for enhancing the security of VLMs.

摘要: 像CLIP这样的预先训练的视觉语言模型(VLM)在各种下游任务中表现出令人印象深刻的泛化性能，但它们仍然容易受到对手的攻击。虽然以前的研究主要集中在提高图像编码器的对抗健壮性以防止对图像的攻击，但对基于文本的和多模式攻击的探索在很大程度上被忽视了。在这项工作中，我们启动了第一个已知和全面的努力，以研究适应视觉语言模型的对手在多模式攻击下的稳健性。首先，我们介绍了一种多模式攻击策略，并研究了不同攻击的影响。然后，我们提出了一种多模式对抗性训练损失，将干净和对抗性的文本嵌入与对抗性和干净的视觉特征相结合，以增强CLIP图像和文本编码者的对抗性健壮性。在两个任务的15个数据集上的大量实验表明，我们的方法显著地提高了CLIP的对抗健壮性。有趣的是，我们发现，与仅针对基于图像的攻击进行微调的模型相比，针对多模式攻击进行微调的模型表现出更强的稳健性，甚至在图像攻击的背景下也是如此，这可能为增强VLM的安全性开辟新的可能性。



## **49. Continuous Embedding Attacks via Clipped Inputs in Jailbreaking Large Language Models**

通过在越狱大型语言模型中剪辑输入进行连续嵌入攻击 cs.CR

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2407.13796v1) [paper-pdf](http://arxiv.org/pdf/2407.13796v1)

**Authors**: Zihao Xu, Yi Liu, Gelei Deng, Kailong Wang, Yuekang Li, Ling Shi, Stjepan Picek

**Abstract**: Security concerns for large language models (LLMs) have recently escalated, focusing on thwarting jailbreaking attempts in discrete prompts. However, the exploration of jailbreak vulnerabilities arising from continuous embeddings has been limited, as prior approaches primarily involved appending discrete or continuous suffixes to inputs. Our study presents a novel channel for conducting direct attacks on LLM inputs, eliminating the need for suffix addition or specific questions provided that the desired output is predefined. We additionally observe that extensive iterations often lead to overfitting, characterized by repetition in the output. To counteract this, we propose a simple yet effective strategy named CLIP. Our experiments show that for an input length of 40 at iteration 1000, applying CLIP improves the ASR from 62% to 83%

摘要: 对大型语言模型（LLM）的安全担忧最近有所升级，重点是阻止离散提示中的越狱尝试。然而，对连续嵌入产生的越狱漏洞的探索一直受到限制，因为以前的方法主要涉及在输入中添加离散或连续后缀。我们的研究提出了一种新型渠道，可以对LLM输入进行直接攻击，只要预定义了所需的输出，就无需添加后缀或特定问题。我们还观察到，大量的迭代通常会导致过度逼近，其特征是输出中的重复。为了解决这个问题，我们提出了一种简单而有效的策略，名为CLIP。我们的实验表明，对于迭代1000次时40的输入长度，应用CLIP将ASB从62%提高到83%



## **50. JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models**

越狱长凳：越狱大型语言模型的开放鲁棒性基准 cs.CR

JailbreakBench v1.0: more attack artifacts, more test-time defenses,  a more accurate jailbreak judge (Llama-3-70B with a custom prompt), a larger  dataset of human preferences for selecting a jailbreak judge (300 examples),  an over-refusal evaluation dataset (100 benign/borderline behaviors), a  semantic refusal judge based on Llama-3-8B

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2404.01318v4) [paper-pdf](http://arxiv.org/pdf/2404.01318v4)

**Authors**: Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, Eric Wong

**Abstract**: Jailbreak attacks cause large language models (LLMs) to generate harmful, unethical, or otherwise objectionable content. Evaluating these attacks presents a number of challenges, which the current collection of benchmarks and evaluation techniques do not adequately address. First, there is no clear standard of practice regarding jailbreaking evaluation. Second, existing works compute costs and success rates in incomparable ways. And third, numerous works are not reproducible, as they withhold adversarial prompts, involve closed-source code, or rely on evolving proprietary APIs. To address these challenges, we introduce JailbreakBench, an open-sourced benchmark with the following components: (1) an evolving repository of state-of-the-art adversarial prompts, which we refer to as jailbreak artifacts; (2) a jailbreaking dataset comprising 100 behaviors -- both original and sourced from prior work (Zou et al., 2023; Mazeika et al., 2023, 2024) -- which align with OpenAI's usage policies; (3) a standardized evaluation framework at https://github.com/JailbreakBench/jailbreakbench that includes a clearly defined threat model, system prompts, chat templates, and scoring functions; and (4) a leaderboard at https://jailbreakbench.github.io/ that tracks the performance of attacks and defenses for various LLMs. We have carefully considered the potential ethical implications of releasing this benchmark, and believe that it will be a net positive for the community.

摘要: 越狱攻击会导致大型语言模型(LLM)生成有害、不道德或令人反感的内容。评估这些攻击带来了许多挑战，目前收集的基准和评估技术没有充分解决这些挑战。首先，关于越狱评估没有明确的实践标准。其次，现有的工作以无与伦比的方式计算成本和成功率。第三，许多作品是不可复制的，因为它们保留了对抗性提示，涉及封闭源代码，或者依赖于不断发展的专有API。为了应对这些挑战，我们引入了JailBreak，这是一个开源基准，具有以下组件：(1)一个不断发展的最先进对手提示的储存库，我们称之为越狱人工制品；(2)一个包括100种行为的越狱数据集--既有原始的，也有源自先前工作的(邹某等人，2023年；Mazeika等人，2023年，2024年)--与开放人工智能的使用政策保持一致；(3)https://github.com/JailbreakBench/jailbreakbench的标准化评估框架，包括明确定义的威胁模型、系统提示、聊天模板和评分功能；以及(4)https://jailbreakbench.github.io/的排行榜，跟踪各种LLM的攻击和防御性能。我们已仔细考虑发布这一基准的潜在道德影响，并相信它将为社会带来净积极的影响。



