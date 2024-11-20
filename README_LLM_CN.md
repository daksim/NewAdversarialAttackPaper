# Latest Large Language Model Attack Papers
**update at 2024-11-20 11:28:51**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. When Backdoors Speak: Understanding LLM Backdoor Attacks Through Model-Generated Explanations**

当后门说话时：通过模型生成的解释了解LLM后门攻击 cs.CR

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2411.12701v1) [paper-pdf](http://arxiv.org/pdf/2411.12701v1)

**Authors**: Huaizhi Ge, Yiming Li, Qifan Wang, Yongfeng Zhang, Ruixiang Tang

**Abstract**: Large Language Models (LLMs) are vulnerable to backdoor attacks, where hidden triggers can maliciously manipulate model behavior. While several backdoor attack methods have been proposed, the mechanisms by which backdoor functions operate in LLMs remain underexplored. In this paper, we move beyond attacking LLMs and investigate backdoor functionality through the novel lens of natural language explanations. Specifically, we leverage LLMs' generative capabilities to produce human-understandable explanations for their decisions, allowing us to compare explanations for clean and poisoned samples. We explore various backdoor attacks and embed the backdoor into LLaMA models for multiple tasks. Our experiments show that backdoored models produce higher-quality explanations for clean data compared to poisoned data, while generating significantly more consistent explanations for poisoned data than for clean data. We further analyze the explanation generation process, revealing that at the token level, the explanation token of poisoned samples only appears in the final few transformer layers of the LLM. At the sentence level, attention dynamics indicate that poisoned inputs shift attention from the input context when generating the explanation. These findings deepen our understanding of backdoor attack mechanisms in LLMs and offer a framework for detecting such vulnerabilities through explainability techniques, contributing to the development of more secure LLMs.

摘要: 大型语言模型(LLM)容易受到后门攻击，在后门攻击中，隐藏的触发器可以恶意操纵模型行为。虽然已经提出了几种后门攻击方法，但后门功能在LLM中运行的机制仍未得到充分探索。在这篇文章中，我们超越了攻击LLM，通过自然语言解释的新视角来研究后门功能。具体地说，我们利用LLMS的生成能力来为他们的决定产生人类可以理解的解释，使我们能够比较干净和有毒样本的解释。我们探索了各种后门攻击，并将后门嵌入到骆驼模型中，以实现多种任务。我们的实验表明，与有毒数据相比，回溯模型对干净数据产生了更高质量的解释，而对有毒数据产生的解释比对干净数据产生的解释要一致得多。我们进一步分析了解释的生成过程，发现在令牌级别，有毒样本的解释令牌只出现在LLM的最后几个转换器层。在句子层面，注意动力学表明，有毒输入在生成解释时转移了对输入上下文的注意力。这些发现加深了我们对LLMS后门攻击机制的理解，并提供了一个通过可解释性技术检测此类漏洞的框架，有助于开发更安全的LLMS。



## **2. Exploring adversarial robustness of JPEG AI: methodology, comparison and new methods**

探索JPEG AI的对抗鲁棒性：方法论、比较和新方法 eess.IV

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11795v1) [paper-pdf](http://arxiv.org/pdf/2411.11795v1)

**Authors**: Egor Kovalev, Georgii Bychkov, Khaled Abud, Aleksandr Gushchin, Anna Chistyakova, Sergey Lavrushkin, Dmitriy Vatolin, Anastasia Antsiferova

**Abstract**: Adversarial robustness of neural networks is an increasingly important area of research, combining studies on computer vision models, large language models (LLMs), and others. With the release of JPEG AI - the first standard for end-to-end neural image compression (NIC) methods - the question of its robustness has become critically significant. JPEG AI is among the first international, real-world applications of neural-network-based models to be embedded in consumer devices. However, research on NIC robustness has been limited to open-source codecs and a narrow range of attacks. This paper proposes a new methodology for measuring NIC robustness to adversarial attacks. We present the first large-scale evaluation of JPEG AI's robustness, comparing it with other NIC models. Our evaluation results and code are publicly available online (link is hidden for a blind review).

摘要: 神经网络的对抗鲁棒性是一个越来越重要的研究领域，结合了对计算机视觉模型、大型语言模型（LLM）等的研究。随着JPEG AI（端到端神经图像压缩（NIC）方法的第一个标准）的发布，其稳健性问题变得至关重要。JPEG AI是首批嵌入消费设备的基于神经网络的模型的国际现实应用之一。然而，关于NIC稳健性的研究仅限于开源编解码器和范围狭窄的攻击。本文提出了一种新的方法来衡量NIC对对抗性攻击的稳健性。我们首次对JPEG AI的稳健性进行了大规模评估，并将其与其他NIC模型进行了比较。我们的评估结果和代码可在线公开（链接已隐藏，以供盲目审查）。



## **3. DAWN: Designing Distributed Agents in a Worldwide Network**

DAWN：在全球网络中设计分布式代理 cs.NI

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2410.22339v2) [paper-pdf](http://arxiv.org/pdf/2410.22339v2)

**Authors**: Zahra Aminiranjbar, Jianan Tang, Qiudan Wang, Shubha Pant, Mahesh Viswanathan

**Abstract**: The rapid evolution of Large Language Models (LLMs) has transformed them from basic conversational tools into sophisticated entities capable of complex reasoning and decision-making. These advancements have led to the development of specialized LLM-based agents designed for diverse tasks such as coding and web browsing. As these agents become more capable, the need for a robust framework that facilitates global communication and collaboration among them towards advanced objectives has become increasingly critical. Distributed Agents in a Worldwide Network (DAWN) addresses this need by offering a versatile framework that integrates LLM-based agents with traditional software systems, enabling the creation of agentic applications suited for a wide range of use cases. DAWN enables distributed agents worldwide to register and be easily discovered through Gateway Agents. Collaborations among these agents are coordinated by a Principal Agent equipped with reasoning strategies. DAWN offers three operational modes: No-LLM Mode for deterministic tasks, Copilot for augmented decision-making, and LLM Agent for autonomous operations. Additionally, DAWN ensures the safety and security of agent collaborations globally through a dedicated safety, security, and compliance layer, protecting the network against attackers and adhering to stringent security and compliance standards. These features make DAWN a robust network for deploying agent-based applications across various industries.

摘要: 大型语言模型的快速发展使它们从基本的对话工具转变为能够进行复杂推理和决策的复杂实体。这些进步导致了专门的基于LLM的代理的开发，这些代理专为不同的任务而设计，如编码和Web浏览。随着这些机构变得更有能力，需要一个强有力的框架，促进它们之间的全球沟通和合作，以实现更高的目标，这一需求变得越来越重要。全球网络中的分布式代理(DAW)通过提供一个通用的框架来满足这一需求，该框架将基于LLM的代理与传统软件系统集成在一起，从而能够创建适合于各种用例的代理应用程序。曙光使分布在世界各地的代理能够注册，并通过网关代理容易地被发现。这些代理之间的协作由一个配备了推理策略的委托代理来协调。曙光提供了三种操作模式：用于确定性任务的no-LLM模式，用于增强决策的Copilot模式，以及用于自主操作的LLM代理。此外，曙光公司通过专门的安全、保障和合规层确保全球代理协作的安全和保障，保护网络免受攻击者的攻击，并遵守严格的安全和合规标准。这些功能使曙光成为在不同行业部署基于代理的应用程序的强大网络。



## **4. TrojanRobot: Backdoor Attacks Against Robotic Manipulation in the Physical World**

特洛伊机器人：针对物理世界中机器人操纵的后门攻击 cs.RO

Initial version with preliminary results. We welcome any feedback or  suggestions

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11683v1) [paper-pdf](http://arxiv.org/pdf/2411.11683v1)

**Authors**: Xianlong Wang, Hewen Pan, Hangtao Zhang, Minghui Li, Shengshan Hu, Ziqi Zhou, Lulu Xue, Peijin Guo, Yichen Wang, Wei Wan, Aishan Liu, Leo Yu Zhang

**Abstract**: Robotic manipulation refers to the autonomous handling and interaction of robots with objects using advanced techniques in robotics and artificial intelligence. The advent of powerful tools such as large language models (LLMs) and large vision-language models (LVLMs) has significantly enhanced the capabilities of these robots in environmental perception and decision-making. However, the introduction of these intelligent agents has led to security threats such as jailbreak attacks and adversarial attacks.   In this research, we take a further step by proposing a backdoor attack specifically targeting robotic manipulation and, for the first time, implementing backdoor attack in the physical world. By embedding a backdoor visual language model into the visual perception module within the robotic system, we successfully mislead the robotic arm's operation in the physical world, given the presence of common items as triggers. Experimental evaluations in the physical world demonstrate the effectiveness of the proposed backdoor attack.

摘要: 机器人操纵是指使用机器人学和人工智能的先进技术，自主处理机器人与物体的交互。大型语言模型(LLM)和大型视觉语言模型(LVLM)等强大工具的出现，大大增强了这些机器人在环境感知和决策方面的能力。然而，这些智能代理的引入导致了越狱攻击和对抗性攻击等安全威胁。在这项研究中，我们进一步提出了专门针对机器人操作的后门攻击，并首次在物理世界中实现了后门攻击。通过将后门视觉语言模型嵌入机器人系统的视觉感知模块中，我们成功地误导了机械臂在物理世界中的操作，因为存在共同的物品作为触发器。物理世界中的实验评估证明了所提出的后门攻击的有效性。



## **5. Enhancing Vision-Language Model Safety through Progressive Concept-Bottleneck-Driven Alignment**

通过渐进的概念瓶颈驱动的一致增强视觉语言模型的安全性 cs.CV

arXiv admin note: substantial text overlap with arXiv:2405.13581

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11543v1) [paper-pdf](http://arxiv.org/pdf/2411.11543v1)

**Authors**: Zhendong Liu, Yuanbi Nie, Yingshui Tan, Xiangyu Yue, Qiushi Cui, Chongjun Wang, Xiaoyong Zhu, Bo Zheng

**Abstract**: Benefiting from the powerful capabilities of Large Language Models (LLMs), pre-trained visual encoder models connected to LLMs form Vision Language Models (VLMs). However, recent research shows that the visual modality in VLMs is highly vulnerable, allowing attackers to bypass safety alignment in LLMs through visually transmitted content, launching harmful attacks. To address this challenge, we propose a progressive concept-based alignment strategy, PSA-VLM, which incorporates safety modules as concept bottlenecks to enhance visual modality safety alignment. By aligning model predictions with specific safety concepts, we improve defenses against risky images, enhancing explainability and controllability while minimally impacting general performance. Our method is obtained through two-stage training. The low computational cost of the first stage brings very effective performance improvement, and the fine-tuning of the language model in the second stage further improves the safety performance. Our method achieves state-of-the-art results on popular VLM safety benchmark.

摘要: 得益于大型语言模型的强大功能，连接到大型语言模型的预先训练的视觉编码器模型形成了视觉语言模型。然而，最近的研究表明，VLMS中的视觉通道非常容易受到攻击，使得攻击者能够通过视觉传输的内容绕过LLMS中的安全对齐，从而发起有害攻击。为了应对这一挑战，我们提出了一种基于概念的渐进式对齐策略PSA-VLM，该策略将安全模块作为概念瓶颈纳入其中，以增强视觉通道的安全对齐。通过将模型预测与特定的安全概念相结合，我们改进了对危险图像的防御，增强了可解释性和可控性，同时将对总体性能的影响降至最低。我们的方法是通过两个阶段的训练获得的。第一阶段的低运算量带来了非常有效的性能提升，第二阶段对语言模型的微调进一步提高了安全性能。我们的方法在流行的VLM安全基准上获得了最先进的结果。



## **6. Membership Inference Attack against Long-Context Large Language Models**

针对长上下文大型语言模型的成员推断攻击 cs.CL

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11424v1) [paper-pdf](http://arxiv.org/pdf/2411.11424v1)

**Authors**: Zixiong Wang, Gaoyang Liu, Yang Yang, Chen Wang

**Abstract**: Recent advances in Large Language Models (LLMs) have enabled them to overcome their context window limitations, and demonstrate exceptional retrieval and reasoning capacities on longer context. Quesion-answering systems augmented with Long-Context Language Models (LCLMs) can automatically search massive external data and incorporate it into their contexts, enabling faithful predictions and reducing issues such as hallucinations and knowledge staleness. Existing studies targeting LCLMs mainly concentrate on addressing the so-called lost-in-the-middle problem or improving the inference effiencicy, leaving their privacy risks largely unexplored. In this paper, we aim to bridge this gap and argue that integrating all information into the long context makes it a repository of sensitive information, which often contains private data such as medical records or personal identities. We further investigate the membership privacy within LCLMs external context, with the aim of determining whether a given document or sequence is included in the LCLMs context. Our basic idea is that if a document lies in the context, it will exhibit a low generation loss or a high degree of semantic similarity to the contents generated by LCLMs. We for the first time propose six membership inference attack (MIA) strategies tailored for LCLMs and conduct extensive experiments on various popular models. Empirical results demonstrate that our attacks can accurately infer membership status in most cases, e.g., 90.66% attack F1-score on Multi-document QA datasets with LongChat-7b-v1.5-32k, highlighting significant risks of membership leakage within LCLMs input contexts. Furthermore, we examine the underlying reasons why LCLMs are susceptible to revealing such membership information.

摘要: 大型语言模型(LLM)的最新进展使它们能够克服上下文窗口的限制，并在更长的上下文中显示出出色的检索和推理能力。带有长上下文语言模型(LCLM)的问答系统可以自动搜索大量外部数据并将其合并到上下文中，从而实现准确的预测，并减少幻觉和知识陈旧等问题。现有的针对LCLM的研究主要集中在解决所谓的中间迷失问题或提高推理效率上，而对它们的隐私风险基本上没有进行研究。在本文中，我们旨在弥合这一差距，并认为将所有信息整合到长上下文中使其成为敏感信息的存储库，其中通常包含私人数据，如医疗记录或个人身份。我们进一步研究LCLMS外部上下文中的成员身份隐私，目的是确定给定的文档或序列是否包括在LCLMS上下文中。我们的基本思想是，如果一个文档位于上下文中，它将表现出与LCLM生成的内容的低生成损失或高度语义相似性。我们首次提出了六种专为LCLM定制的成员推理攻击(MIA)策略，并在各种流行的模型上进行了广泛的实验。实验结果表明，我们的攻击可以在大多数情况下准确地推断成员状态，例如，在具有LongChat-7b-v1.5-32k的多文档QA数据集上，90.66%的攻击F1-Score，突出了LCLM输入上下文中成员泄漏的显著风险。此外，我们还考察了LCLM容易泄露此类成员信息的潜在原因。



## **7. The Dark Side of Trust: Authority Citation-Driven Jailbreak Attacks on Large Language Models**

信任的阴暗面：权威引用驱动的对大型语言模型的越狱攻击 cs.LG

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11407v1) [paper-pdf](http://arxiv.org/pdf/2411.11407v1)

**Authors**: Xikang Yang, Xuehai Tang, Jizhong Han, Songlin Hu

**Abstract**: The widespread deployment of large language models (LLMs) across various domains has showcased their immense potential while exposing significant safety vulnerabilities. A major concern is ensuring that LLM-generated content aligns with human values. Existing jailbreak techniques reveal how this alignment can be compromised through specific prompts or adversarial suffixes. In this study, we introduce a new threat: LLMs' bias toward authority. While this inherent bias can improve the quality of outputs generated by LLMs, it also introduces a potential vulnerability, increasing the risk of producing harmful content. Notably, the biases in LLMs is the varying levels of trust given to different types of authoritative information in harmful queries. For example, malware development often favors trust GitHub. To better reveal the risks with LLM, we propose DarkCite, an adaptive authority citation matcher and generator designed for a black-box setting. DarkCite matches optimal citation types to specific risk types and generates authoritative citations relevant to harmful instructions, enabling more effective jailbreak attacks on aligned LLMs.Our experiments show that DarkCite achieves a higher attack success rate (e.g., LLama-2 at 76% versus 68%) than previous methods. To counter this risk, we propose an authenticity and harm verification defense strategy, raising the average defense pass rate (DPR) from 11% to 74%. More importantly, the ability to link citations to the content they encompass has become a foundational function in LLMs, amplifying the influence of LLMs' bias toward authority.

摘要: 大型语言模型(LLM)在不同领域的广泛部署展示了它们的巨大潜力，同时也暴露了重大的安全漏洞。一个主要的问题是确保LLM生成的内容符合人类的价值观。现有的越狱技术揭示了如何通过特定的提示或对抗性后缀来破坏这种对齐。在这项研究中，我们引入了一个新的威胁：LLMS对权威的偏见。虽然这种固有的偏见可以提高低成本管理产生的产出的质量，但它也引入了一个潜在的脆弱性，增加了产生有害内容的风险。值得注意的是，LLMS中的偏差是在有害查询中对不同类型的权威信息给予的不同程度的信任。例如，恶意软件开发通常偏向信任GitHub。为了更好地揭示LLM的风险，我们提出了DarkCite，这是一个为黑箱设置而设计的自适应权威引用匹配器和生成器。DarkCite将最佳引用类型与特定的风险类型相匹配，并生成与有害指令相关的权威引用，从而对对齐的LLMS进行更有效的越狱攻击。我们的实验表明，与以前的方法相比，DarkCite实现了更高的攻击成功率(例如，骆驼-2为76%，而不是68%)。为了应对这种风险，我们提出了真实性和危害性验证防御策略，将平均防御通过率(DPR)从11%提高到74%。更重要的是，将引文与它们所包含的内容相联系的能力已经成为LLMS的一项基本功能，放大了LLMS对权威的偏见的影响。



## **8. Hacking Back the AI-Hacker: Prompt Injection as a Defense Against LLM-driven Cyberattacks**

黑客攻击人工智能黑客：即时注入作为抵御LLM驱动的网络攻击的防御 cs.CR

v0.2 (evaluated on more agents)

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2410.20911v2) [paper-pdf](http://arxiv.org/pdf/2410.20911v2)

**Authors**: Dario Pasquini, Evgenios M. Kornaropoulos, Giuseppe Ateniese

**Abstract**: Large language models (LLMs) are increasingly being harnessed to automate cyberattacks, making sophisticated exploits more accessible and scalable. In response, we propose a new defense strategy tailored to counter LLM-driven cyberattacks. We introduce Mantis, a defensive framework that exploits LLMs' susceptibility to adversarial inputs to undermine malicious operations. Upon detecting an automated cyberattack, Mantis plants carefully crafted inputs into system responses, leading the attacker's LLM to disrupt their own operations (passive defense) or even compromise the attacker's machine (active defense). By deploying purposefully vulnerable decoy services to attract the attacker and using dynamic prompt injections for the attacker's LLM, Mantis can autonomously hack back the attacker. In our experiments, Mantis consistently achieved over 95% effectiveness against automated LLM-driven attacks. To foster further research and collaboration, Mantis is available as an open-source tool: https://github.com/pasquini-dario/project_mantis

摘要: 大型语言模型(LLM)越来越多地被用来自动化网络攻击，使复杂的利用更容易获得和可扩展。作为回应，我们提出了一种新的防御战略，以对抗LLM驱动的网络攻击。我们引入了Mantis，这是一个防御框架，利用LLMS对对手输入的敏感性来破坏恶意操作。在检测到自动网络攻击后，螳螂工厂会精心设计输入到系统响应中，导致攻击者的LLM扰乱自己的操作(被动防御)，甚至危害攻击者的机器(主动防御)。通过部署故意易受攻击的诱骗服务来吸引攻击者，并对攻击者的LLM使用动态提示注入，螳螂可以自主地攻击攻击者。在我们的实验中，螳螂对自动LLM驱动的攻击始终取得了95%以上的效率。为了促进进一步的研究和合作，Mantis以开源工具的形式提供：https://github.com/pasquini-dario/project_mantis



## **9. Adapting to Cyber Threats: A Phishing Evolution Network (PEN) Framework for Phishing Generation and Analyzing Evolution Patterns using Large Language Models**

适应网络威胁：用于使用大型语言模型进行网络钓鱼生成和分析进化模式的网络钓鱼进化网络（PEN）框架 cs.CR

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11389v1) [paper-pdf](http://arxiv.org/pdf/2411.11389v1)

**Authors**: Fengchao Chen, Tingmin Wu, Van Nguyen, Shuo Wang, Hongsheng Hu, Alsharif Abuadbba, Carsten Rudolph

**Abstract**: Phishing remains a pervasive cyber threat, as attackers craft deceptive emails to lure victims into revealing sensitive information. While Artificial Intelligence (AI), particularly deep learning, has become a key component in defending against phishing attacks, these approaches face critical limitations. The scarcity of publicly available, diverse, and updated data, largely due to privacy concerns, constrains their effectiveness. As phishing tactics evolve rapidly, models trained on limited, outdated data struggle to detect new, sophisticated deception strategies, leaving systems vulnerable to an ever-growing array of attacks. Addressing this gap is essential to strengthening defenses in an increasingly hostile cyber landscape. To address this gap, we propose the Phishing Evolution Network (PEN), a framework leveraging large language models (LLMs) and adversarial training mechanisms to continuously generate high quality and realistic diverse phishing samples, and analyze features of LLM-provided phishing to understand evolving phishing patterns. We evaluate the quality and diversity of phishing samples generated by PEN and find that it produces over 80% realistic phishing samples, effectively expanding phishing datasets across seven dominant types. These PEN-generated samples enhance the performance of current phishing detectors, leading to a 40% improvement in detection accuracy. Additionally, the use of PEN significantly boosts model robustness, reducing detectors' sensitivity to perturbations by up to 60%, thereby decreasing attack success rates under adversarial conditions. When we analyze the phishing patterns that are used in LLM-generated phishing, the cognitive complexity and the tone of time limitation are detected with statistically significant differences compared with existing phishing.

摘要: 网络钓鱼仍然是一个普遍存在的网络威胁，因为攻击者精心制作了欺骗性电子邮件，以引诱受害者泄露敏感信息。虽然人工智能(AI)，特别是深度学习，已经成为防御网络钓鱼攻击的关键组件，但这些方法面临着严重的限制。由于缺乏公开可用的、多样化的和更新的数据，这主要是由于隐私问题，限制了它们的有效性。随着钓鱼策略的快速发展，基于有限、过时数据的模型很难检测出新的、复杂的欺骗策略，这使得系统容易受到越来越多的攻击。在日益充满敌意的网络环境中，解决这一差距对于加强防御至关重要。为了弥补这一差距，我们提出了钓鱼进化网络(PEN)，这是一个利用大型语言模型(LLMS)和对手训练机制来持续生成高质量和真实的多样化钓鱼样本的框架，并分析LLM提供的钓鱼特征以了解不断演变的钓鱼模式。我们评估了PEN生成的网络钓鱼样本的质量和多样性，发现它产生了超过80%的真实网络钓鱼样本，有效地扩展了七种主要类型的网络钓鱼数据集。这些笔生成的样本增强了当前网络钓鱼检测器的性能，导致检测准确率提高了40%。此外，PEN的使用显著提高了模型的稳健性，将检测器对扰动的敏感度降低了高达60%，从而降低了对抗性条件下的攻击成功率。当我们分析LLM生成的网络钓鱼中使用的网络钓鱼模式时，我们检测到了认知复杂性和时间限制的基调，与现有的网络钓鱼相比具有统计学意义上的差异。



## **10. JailbreakLens: Interpreting Jailbreak Mechanism in the Lens of Representation and Circuit**

越狱镜头：以表象和电路的视角解读越狱机制 cs.CR

18 pages, 10 figures

**SubmitDate**: 2024-11-17    [abs](http://arxiv.org/abs/2411.11114v1) [paper-pdf](http://arxiv.org/pdf/2411.11114v1)

**Authors**: Zeqing He, Zhibo Wang, Zhixuan Chu, Huiyu Xu, Rui Zheng, Kui Ren, Chun Chen

**Abstract**: Despite the outstanding performance of Large language models (LLMs) in diverse tasks, they are vulnerable to jailbreak attacks, wherein adversarial prompts are crafted to bypass their security mechanisms and elicit unexpected responses.Although jailbreak attacks are prevalent, the understanding of their underlying mechanisms remains limited. Recent studies have explain typical jailbreaking behavior (e.g., the degree to which the model refuses to respond) of LLMs by analyzing the representation shifts in their latent space caused by jailbreak prompts or identifying key neurons that contribute to the success of these attacks. However, these studies neither explore diverse jailbreak patterns nor provide a fine-grained explanation from the failure of circuit to the changes of representational, leaving significant gaps in uncovering the jailbreak mechanism. In this paper, we propose JailbreakLens, an interpretation framework that analyzes jailbreak mechanisms from both representation (which reveals how jailbreaks alter the model's harmfulness perception) and circuit perspectives (which uncovers the causes of these deceptions by identifying key circuits contributing to the vulnerability), tracking their evolution throughout the entire response generation process. We then conduct an in-depth evaluation of jailbreak behavior on four mainstream LLMs under seven jailbreak strategies. Our evaluation finds that jailbreak prompts amplify components that reinforce affirmative responses while suppressing those that produce refusal. Although this manipulation shifts model representations toward safe clusters to deceive the LLM, leading it to provide detailed responses instead of refusals, it still produce abnormal activation which can be caught in the circuit analysis.

摘要: 尽管大型语言模型(LLM)在不同的任务中表现出色，但它们很容易受到越狱攻击，在这些攻击中，敌意提示被精心制作以绕过其安全机制并引发意外响应。尽管越狱攻击非常普遍，但对其潜在机制的了解仍然有限。最近的研究已经通过分析越狱提示引起的潜伏空间的表征变化或识别有助于这些攻击成功的关键神经元来解释LLM的典型越狱行为(例如，模型拒绝响应的程度)。然而，这些研究既没有探索多样化的越狱模式，也没有提供从电路故障到表征变化的细粒度解释，在揭示越狱机制方面留下了重大空白。在本文中，我们提出了JailBreakLens，一个解释框架，它从表示(揭示越狱如何改变模型的危害性感知)和电路角度(通过识别导致漏洞的关键电路来揭示这些欺骗的原因)来分析越狱机制，跟踪它们在整个响应生成过程中的演变。然后，我们在七种越狱策略下对四种主流的低成本移动模型的越狱行为进行了深入的评估。我们的评估发现，越狱提示放大了那些强化肯定反应的成分，同时抑制了那些产生拒绝的成分。尽管这种操作将模型表示转移到安全簇以欺骗LLM，导致它提供详细的响应而不是拒绝，但它仍然产生可以在电路分析中发现的异常激活。



## **11. Safely Learning with Private Data: A Federated Learning Framework for Large Language Model**

利用私人数据安全学习：大型语言模型的联邦学习框架 cs.CR

**SubmitDate**: 2024-11-17    [abs](http://arxiv.org/abs/2406.14898v3) [paper-pdf](http://arxiv.org/pdf/2406.14898v3)

**Authors**: JiaYing Zheng, HaiNan Zhang, LingXiang Wang, WangJie Qiu, HongWei Zheng, ZhiMing Zheng

**Abstract**: Private data, being larger and quality-higher than public data, can greatly improve large language models (LLM). However, due to privacy concerns, this data is often dispersed in multiple silos, making its secure utilization for LLM training a challenge. Federated learning (FL) is an ideal solution for training models with distributed private data, but traditional frameworks like FedAvg are unsuitable for LLM due to their high computational demands on clients. An alternative, split learning, offloads most training parameters to the server while training embedding and output layers locally, making it more suitable for LLM. Nonetheless, it faces significant challenges in security and efficiency. Firstly, the gradients of embeddings are prone to attacks, leading to potential reverse engineering of private data. Furthermore, the server's limitation of handle only one client's training request at a time hinders parallel training, severely impacting training efficiency. In this paper, we propose a Federated Learning framework for LLM, named FL-GLM, which prevents data leakage caused by both server-side and peer-client attacks while improving training efficiency. Specifically, we first place the input block and output block on local client to prevent embedding gradient attacks from server. Secondly, we employ key-encryption during client-server communication to prevent reverse engineering attacks from peer-clients. Lastly, we employ optimization methods like client-batching or server-hierarchical, adopting different acceleration methods based on the actual computational capabilities of the server. Experimental results on NLU and generation tasks demonstrate that FL-GLM achieves comparable metrics to centralized chatGLM model, validating the effectiveness of our federated learning framework.

摘要: 私有数据比公共数据更大、质量更高，可以极大地改进大型语言模型(LLM)。然而，出于隐私方面的考虑，这些数据通常分散在多个竖井中，这使得将其安全地用于LLM培训成为一项挑战。联邦学习(FL)是一种适用于具有分布式私有数据的模型训练的理想解决方案，但FedAvg等传统框架由于对客户端的计算要求较高而不适用于LLM。另一种选择是分离学习，将大部分训练参数卸载到服务器，同时在本地训练嵌入和输出层，使其更适合LLM。尽管如此，它在安全和效率方面仍面临重大挑战。首先，嵌入的梯度容易受到攻击，从而导致对私有数据的潜在逆向工程。此外，服务器一次只能处理一个客户端的训练请求的限制阻碍了并行训练，严重影响了训练效率。本文提出了一种用于LLM的联邦学习框架FL-GLM，该框架在提高训练效率的同时，防止了服务器端攻击和对等客户端攻击引起的数据泄漏。具体地说，我们首先将输入块和输出块放置在本地客户端，以防止来自服务器的嵌入梯度攻击。其次，我们在客户-服务器通信过程中使用密钥加密，以防止来自对等客户端的反向工程攻击。最后，我们采用了客户端批处理或服务器分层等优化方法，根据服务器的实际计算能力采用不同的加速方法。在NLU和生成任务上的实验结果表明，FL-GLM达到了与集中式ChatGLM模型相当的指标，验证了联邦学习框架的有效性。



## **12. Risks of Practicing Large Language Models in Smart Grid: Threat Modeling and Validation**

在智能电网中实践大型语言模型的风险：威胁建模和验证 cs.CR

**SubmitDate**: 2024-11-17    [abs](http://arxiv.org/abs/2405.06237v2) [paper-pdf](http://arxiv.org/pdf/2405.06237v2)

**Authors**: Jiangnan Li, Yingyuan Yang, Jinyuan Sun

**Abstract**: Large language models (LLMs) represent significant breakthroughs in artificial intelligence and hold considerable potential for applications within smart grids. However, as demonstrated in previous literature, AI technologies are susceptible to various types of attacks. It is crucial to investigate and evaluate the risks associated with LLMs before deploying them in critical infrastructure like smart grids. In this paper, we systematically evaluated the risks of LLMs and identified two major types of attacks relevant to potential smart grid LLM applications, presenting the corresponding threat models. We also validated these attacks using popular LLMs and real smart grid data. Our validation demonstrates that attackers are capable of injecting bad data and retrieving domain knowledge from LLMs employed in different smart grid applications.

摘要: 大型语言模型（LLM）代表了人工智能的重大突破，在智能电网中的应用中具有相当大的潜力。然而，正如之前的文献所证明的那样，人工智能技术容易受到各种类型的攻击。在将LLM部署在智能电网等关键基础设施中之前，调查和评估与LLM相关的风险至关重要。在本文中，我们系统地评估了LLM的风险，并识别了与潜在智能电网LLM应用相关的两种主要攻击类型，并给出了相应的威胁模型。我们还使用流行的LLM和真实的智能电网数据验证了这些攻击。我们的验证表明，攻击者能够从不同智能电网应用程序中使用的LLM中注入不良数据并检索领域知识。



## **13. SQL Injection Jailbreak: a structural disaster of large language models**

SQL注入越狱：大型语言模型的结构性灾难 cs.CR

**SubmitDate**: 2024-11-16    [abs](http://arxiv.org/abs/2411.01565v2) [paper-pdf](http://arxiv.org/pdf/2411.01565v2)

**Authors**: Jiawei Zhao, Kejiang Chen, Weiming Zhang, Nenghai Yu

**Abstract**: In recent years, the rapid development of large language models (LLMs) has brought new vitality to the various domains and generated substantial social and economic benefits. However, the swift advancement of LLMs has introduced new security vulnerabilities. Jailbreak, a form of attack that induces LLMs to output harmful content through carefully crafted prompts, poses a challenge to the safe and trustworthy development of LLMs. Previous jailbreak attack methods primarily exploited the internal capabilities of the model. Among them, one category leverages the model's implicit capabilities for jailbreak attacks, where the attacker is unaware of the exact reasons for the attack's success. The other category utilizes the model's explicit capabilities for jailbreak attacks, where the attacker understands the reasons for the attack's success. For example, these attacks exploit the model's abilities in coding, contextual learning, or understanding ASCII characters. However, these earlier jailbreak attacks have certain limitations, as they only exploit the inherent capabilities of the model. In this paper, we propose a novel jailbreak method, SQL Injection Jailbreak (SIJ), which utilizes the construction of input prompts by LLMs to inject jailbreak information into user prompts, enabling successful jailbreak of the LLMs. Our SIJ method achieves nearly 100\% attack success rates on five well-known open-source LLMs in the context of AdvBench, while incurring lower time costs compared to previous methods. More importantly, SIJ reveals a new vulnerability in LLMs that urgently needs to be addressed. To this end, we propose a defense method called Self-Reminder-Key and demonstrate its effectiveness through experiments. Our code is available at \href{https://github.com/weiyezhimeng/SQL-Injection-Jailbreak}{https://github.com/weiyezhimeng/SQL-Injection-Jailbreak}.

摘要: 近年来，大语言模型的快速发展给各个领域带来了新的活力，产生了可观的社会效益和经济效益。然而，LLMS的快速发展带来了新的安全漏洞。越狱是一种攻击形式，通过精心制作的提示诱使LLMS输出有害内容，对LLMS的安全和可信开发构成了挑战。以前的越狱攻击方法主要是利用该模型的内部能力。其中，一类利用该模型的隐含能力进行越狱攻击，即攻击者不知道攻击成功的确切原因。另一类利用该模型的明确能力进行越狱攻击，攻击者了解攻击成功的原因。例如，这些攻击利用了模型在编码、上下文学习或理解ASCII字符方面的能力。然而，这些早期的越狱攻击有一定的局限性，因为它们只利用了该模型的固有功能。本文提出了一种新的越狱方法--SQL注入越狱(SIJ)，该方法利用LLMS构造输入提示，在用户提示中注入越狱信息，使LLMS能够成功越狱。与以前的方法相比，我们的SIJ方法在五个著名的开源LLM上获得了近100\%的攻击成功率，同时产生了更低的时间开销。更重要的是，SIJ揭示了LLMS中一个迫切需要解决的新漏洞。为此，我们提出了一种称为自我提醒密钥的防御方法，并通过实验验证了该方法的有效性。我们的代码可以在\href{https://github.com/weiyezhimeng/SQL-Injection-Jailbreak}{https://github.com/weiyezhimeng/SQL-Injection-Jailbreak}.上找到



## **14. Insights and Current Gaps in Open-Source LLM Vulnerability Scanners: A Comparative Analysis**

开源LLM漏洞扫描仪的见解和当前差距：比较分析 cs.CR

15 pages, 11 figures

**SubmitDate**: 2024-11-16    [abs](http://arxiv.org/abs/2410.16527v3) [paper-pdf](http://arxiv.org/pdf/2410.16527v3)

**Authors**: Jonathan Brokman, Omer Hofman, Oren Rachmil, Inderjeet Singh, Vikas Pahuja, Rathina Sabapathy Aishvariya Priya, Amit Giloni, Roman Vainshtein, Hisashi Kojima

**Abstract**: This report presents a comparative analysis of open-source vulnerability scanners for conversational large language models (LLMs). As LLMs become integral to various applications, they also present potential attack surfaces, exposed to security risks such as information leakage and jailbreak attacks. Our study evaluates prominent scanners - Garak, Giskard, PyRIT, and CyberSecEval - that adapt red-teaming practices to expose these vulnerabilities. We detail the distinctive features and practical use of these scanners, outline unifying principles of their design and perform quantitative evaluations to compare them. These evaluations uncover significant reliability issues in detecting successful attacks, highlighting a fundamental gap for future development. Additionally, we contribute a preliminary labelled dataset, which serves as an initial step to bridge this gap. Based on the above, we provide strategic recommendations to assist organizations choose the most suitable scanner for their red-teaming needs, accounting for customizability, test suite comprehensiveness, and industry-specific use cases.

摘要: 本报告对用于会话大型语言模型(LLM)的开源漏洞扫描器进行了比较分析。随着LLM成为各种应用的组成部分，它们也出现了潜在的攻击面，暴露在信息泄露和越狱攻击等安全风险中。我们的研究评估了采用红色团队实践来暴露这些漏洞的著名扫描仪-Garak、Giskard、PyRIT和CyberSecEval。我们详细介绍了这些扫描仪的特点和实际应用，概述了它们设计的统一原则，并进行了定量评估以进行比较。这些评估揭示了检测成功攻击的重大可靠性问题，突显了未来发展的根本差距。此外，我们提供了一个初步的标记数据集，这是弥合这一差距的第一步。在此基础上，我们提供了战略性建议，以帮助组织选择最适合其红团队需求的扫描仪，考虑到可定制性、测试套件的全面性和行业特定的用例。



## **15. Sim-CLIP: Unsupervised Siamese Adversarial Fine-Tuning for Robust and Semantically-Rich Vision-Language Models**

Sim-CLIP：针对稳健且语义丰富的视觉语言模型的无监督Siamese对抗微调 cs.CV

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2407.14971v2) [paper-pdf](http://arxiv.org/pdf/2407.14971v2)

**Authors**: Md Zarif Hossain, Ahmed Imteaj

**Abstract**: Vision-language models (VLMs) have achieved significant strides in recent times specially in multimodal tasks, yet they remain susceptible to adversarial attacks on their vision components. To address this, we propose Sim-CLIP, an unsupervised adversarial fine-tuning method that enhances the robustness of the widely-used CLIP vision encoder against such attacks while maintaining semantic richness and specificity. By employing a Siamese architecture with cosine similarity loss, Sim-CLIP learns semantically meaningful and attack-resilient visual representations without requiring large batch sizes or momentum encoders. Our results demonstrate that VLMs enhanced with Sim-CLIP's fine-tuned CLIP encoder exhibit significantly enhanced robustness against adversarial attacks, while preserving semantic meaning of the perturbed images. Notably, Sim-CLIP does not require additional training or fine-tuning of the VLM itself; replacing the original vision encoder with our fine-tuned Sim-CLIP suffices to provide robustness. This work underscores the significance of reinforcing foundational models like CLIP to safeguard the reliability of downstream VLM applications, paving the way for more secure and effective multimodal systems.

摘要: 视觉语言模型近年来取得了长足的进步，特别是在多通道任务中，但它们仍然容易受到视觉部分的敌意攻击。为了解决这一问题，我们提出了SIM-CLIP，这是一种无监督的对抗性微调方法，它在保持语义丰富和特异性的同时，增强了广泛使用的CLIP视觉编码器对此类攻击的健壮性。通过采用具有余弦相似性损失的暹罗体系结构，Sim-Clip无需大批量或动量编码器即可学习语义上有意义的、可抵抗攻击的视觉表示。结果表明，通过Sim-Clip的精细调整的CLIP编码器增强的VLM在保持扰动图像语义的同时，显著增强了对对手攻击的稳健性。值得注意的是，SIM-Clip不需要对VLM本身进行额外的培训或微调；用我们经过微调的SIM-Clip替换原来的视觉编码器就足以提供健壮性。这项工作强调了加强像CLIP这样的基础模型对保障下游VLM应用的可靠性的重要性，为更安全和有效的多式联运系统铺平了道路。



## **16. Comparing Robustness Against Adversarial Attacks in Code Generation: LLM-Generated vs. Human-Written**

比较代码生成中对抗对抗攻击的鲁棒性：LLM生成与人类编写 cs.SE

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.10565v1) [paper-pdf](http://arxiv.org/pdf/2411.10565v1)

**Authors**: Md Abdul Awal, Mrigank Rochan, Chanchal K. Roy

**Abstract**: Thanks to the widespread adoption of Large Language Models (LLMs) in software engineering research, the long-standing dream of automated code generation has become a reality on a large scale. Nowadays, LLMs such as GitHub Copilot and ChatGPT are extensively used in code generation for enterprise and open-source software development and maintenance. Despite their unprecedented successes in code generation, research indicates that codes generated by LLMs exhibit vulnerabilities and security issues. Several studies have been conducted to evaluate code generated by LLMs, considering various aspects such as security, vulnerability, code smells, and robustness. While some studies have compared the performance of LLMs with that of humans in various software engineering tasks, there's a notable gap in research: no studies have directly compared human-written and LLM-generated code for their robustness analysis. To fill this void, this paper introduces an empirical study to evaluate the adversarial robustness of Pre-trained Models of Code (PTMCs) fine-tuned on code written by humans and generated by LLMs against adversarial attacks for software clone detection. These attacks could potentially undermine software security and reliability. We consider two datasets, two state-of-the-art PTMCs, two robustness evaluation criteria, and three metrics to use in our experiments. Regarding effectiveness criteria, PTMCs fine-tuned on human-written code always demonstrate more robustness than those fine-tuned on LLMs-generated code. On the other hand, in terms of adversarial code quality, in 75% experimental combinations, PTMCs fine-tuned on the human-written code exhibit more robustness than the PTMCs fine-tuned on the LLMs-generated code.

摘要: 由于大型语言模型(LLM)在软件工程研究中的广泛采用，自动化代码生成的长期梦想已经在很大程度上成为现实。如今，GitHub Copilot和ChatGPT等LLMS被广泛用于企业和开源软件开发和维护的代码生成。尽管LLM在代码生成方面取得了前所未有的成功，但研究表明，LLM生成的代码存在漏洞和安全问题。考虑到安全性、脆弱性、代码气味和健壮性等各个方面，已经进行了几项研究来评估LLMS生成的代码。虽然一些研究已经将LLM与人类在各种软件工程任务中的性能进行了比较，但研究中存在一个明显的差距：没有研究直接比较人类编写的代码和LLM生成的代码来进行健壮性分析。为了填补这一空白，本文介绍了一项经验研究，以评估预先训练的代码模型(PTMC)对人类编写的代码进行微调并由LLMS生成的代码对软件克隆检测的恶意攻击的健壮性。这些攻击可能会潜在地破坏软件的安全性和可靠性。我们考虑了两个数据集、两个最先进的PTMC、两个健壮性评估标准和三个度量来用于我们的实验。关于有效性标准，在人类编写的代码上微调的PTMC总是比那些在LLMS生成的代码上微调的PTMC表现出更强的健壮性。另一方面，在对抗性代码质量方面，在75%的实验组合中，基于人类编写的代码微调的PTMC表现出比基于LLMS生成的代码微调的PTMC更强的稳健性。



## **17. On the Privacy Risk of In-context Learning**

论上下文学习的隐私风险 cs.LG

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.10512v1) [paper-pdf](http://arxiv.org/pdf/2411.10512v1)

**Authors**: Haonan Duan, Adam Dziedzic, Mohammad Yaghini, Nicolas Papernot, Franziska Boenisch

**Abstract**: Large language models (LLMs) are excellent few-shot learners. They can perform a wide variety of tasks purely based on natural language prompts provided to them. These prompts contain data of a specific downstream task -- often the private dataset of a party, e.g., a company that wants to leverage the LLM for their purposes. We show that deploying prompted models presents a significant privacy risk for the data used within the prompt by instantiating a highly effective membership inference attack. We also observe that the privacy risk of prompted models exceeds fine-tuned models at the same utility levels. After identifying the model's sensitivity to their prompts -- in the form of a significantly higher prediction confidence on the prompted data -- as a cause for the increased risk, we propose ensembling as a mitigation strategy. By aggregating over multiple different versions of a prompted model, membership inference risk can be decreased.

摘要: 大型语言模型（LLM）是优秀的少量学习者。他们可以纯粹根据向他们提供的自然语言提示执行各种任务。这些提示包含特定下游任务的数据--通常是一方的私人数据集，例如，一家希望利用LLM来实现其目的的公司。我们表明，通过实例化高效的成员资格推断攻击，部署提示模型会给提示内使用的数据带来显着的隐私风险。我们还观察到，提示模型的隐私风险超过了相同实用水平下的微调模型。在确定模型对其提示的敏感性（表现为对提示数据的预测置信度显着更高）是风险增加的原因后，我们建议将集成作为一种缓解策略。通过聚合提示模型的多个不同版本，可以降低成员资格推断风险。



## **18. Optimization-based Prompt Injection Attack to LLM-as-a-Judge**

对LLM as-a-Judge的基于优化的即时注入攻击 cs.CR

To appear in the Proceedings of The ACM Conference on Computer and  Communications Security (CCS), 2024

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2403.17710v3) [paper-pdf](http://arxiv.org/pdf/2403.17710v3)

**Authors**: Jiawen Shi, Zenghui Yuan, Yinuo Liu, Yue Huang, Pan Zhou, Lichao Sun, Neil Zhenqiang Gong

**Abstract**: LLM-as-a-Judge uses a large language model (LLM) to select the best response from a set of candidates for a given question. LLM-as-a-Judge has many applications such as LLM-powered search, reinforcement learning with AI feedback (RLAIF), and tool selection. In this work, we propose JudgeDeceiver, an optimization-based prompt injection attack to LLM-as-a-Judge. JudgeDeceiver injects a carefully crafted sequence into an attacker-controlled candidate response such that LLM-as-a-Judge selects the candidate response for an attacker-chosen question no matter what other candidate responses are. Specifically, we formulate finding such sequence as an optimization problem and propose a gradient based method to approximately solve it. Our extensive evaluation shows that JudgeDeceive is highly effective, and is much more effective than existing prompt injection attacks that manually craft the injected sequences and jailbreak attacks when extended to our problem. We also show the effectiveness of JudgeDeceiver in three case studies, i.e., LLM-powered search, RLAIF, and tool selection. Moreover, we consider defenses including known-answer detection, perplexity detection, and perplexity windowed detection. Our results show these defenses are insufficient, highlighting the urgent need for developing new defense strategies. Our implementation is available at this repository: https://github.com/ShiJiawenwen/JudgeDeceiver.

摘要: LLM-as-a-Court使用大型语言模型(LLM)从给定问题的一组候选人中选择最佳答案。LLM-as-a-Court有许多应用，如LLM支持的搜索、带人工智能反馈的强化学习(RLAIF)和工具选择。在这项工作中，我们提出了一种针对LLM-as-a-Court的基于优化的快速注入攻击--JudgeDeceiver。JudgeDeceiver将精心制作的序列注入到攻击者控制的候选响应中，以便LLM-as-a-Court为攻击者选择的问题选择候选响应，而不管其他候选响应是什么。具体地说，我们将寻找这样的序列描述为一个优化问题，并提出了一种基于梯度的方法来近似求解它。我们的广泛评估表明，JudgeDecept是非常有效的，并且比现有的手动手工创建注入序列的即时注入攻击和越狱攻击更有效，当扩展到我们的问题时。我们还在三个案例研究中展示了JudgeDeceiver的有效性，即LLM支持的搜索、RLAIF和工具选择。此外，我们还考虑了防御措施，包括已知答案检测、困惑检测和困惑加窗检测。我们的结果表明，这些防御措施是不够的，这突显了开发新的防御战略的迫切需要。我们的实现可从以下存储库获得：https://github.com/ShiJiawenwen/JudgeDeceiver.



## **19. IDEATOR: Jailbreaking Large Vision-Language Models Using Themselves**

IDEATOR：利用自己越狱大型视觉语言模型 cs.CV

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.00827v2) [paper-pdf](http://arxiv.org/pdf/2411.00827v2)

**Authors**: Ruofan Wang, Bo Wang, Xiaosen Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: As large Vision-Language Models (VLMs) grow in prominence, ensuring their safe deployment has become critical. Recent studies have explored VLM robustness against jailbreak attacks--techniques that exploit model vulnerabilities to elicit harmful outputs. However, the limited availability of diverse multi-modal data has led current approaches to rely heavily on adversarial or manually crafted images derived from harmful text datasets, which may lack effectiveness and diversity across different contexts. In this paper, we propose a novel jailbreak method named IDEATOR, which autonomously generates malicious image-text pairs for black-box jailbreak attacks. IDEATOR is based on the insight that VLMs themselves could serve as powerful red team models for generating multimodal jailbreak prompts. Specifically, IDEATOR uses a VLM to create targeted jailbreak texts and pairs them with jailbreak images generated by a state-of-the-art diffusion model. Our extensive experiments demonstrate IDEATOR's high effectiveness and transferability. Notably, it achieves a 94% success rate in jailbreaking MiniGPT-4 with an average of only 5.34 queries, and high success rates of 82%, 88%, and 75% when transferred to LLaVA, InstructBLIP, and Meta's Chameleon, respectively. IDEATOR uncovers specific vulnerabilities in VLMs under black-box conditions, underscoring the need for improved safety mechanisms.

摘要: 随着大型视觉语言模型(VLM)的日益突出，确保它们的安全部署变得至关重要。最近的研究探索了VLM对越狱攻击的健壮性--利用模型漏洞来获得有害输出的技术。然而，多样化的多模式数据的可获得性有限，导致目前的方法严重依赖于从有害文本数据集获得的对抗性或手动制作的图像，这可能在不同的背景下缺乏有效性和多样性。在本文中，我们提出了一种新的越狱方法IDEATOR，该方法自动生成用于黑盒越狱攻击的恶意图文对。Ideator基于这样一种见解，即VLM本身可以作为强大的红色团队模型来生成多模式越狱提示。具体地说，Ideator使用VLM创建有针对性的越狱文本，并将它们与由最先进的扩散模型生成的越狱图像配对。我们的大量实验证明了IDEADER的高效性和可移植性。值得注意的是，在平均只有5.34个查询的情况下，它在越狱MiniGPT-4上的成功率达到了94%，当转移到LLaVA、InstructBLIP和Meta‘s Chameleon时，成功率分别达到了82%、88%和75%。Ideator发现了黑箱条件下VLM中的特定漏洞，强调了改进安全机制的必要性。



## **20. Security and Privacy Challenges of Large Language Models: A Survey**

大型语言模型的安全和隐私挑战：调查 cs.CL

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2402.00888v2) [paper-pdf](http://arxiv.org/pdf/2402.00888v2)

**Authors**: Badhan Chandra Das, M. Hadi Amini, Yanzhao Wu

**Abstract**: Large Language Models (LLMs) have demonstrated extraordinary capabilities and contributed to multiple fields, such as generating and summarizing text, language translation, and question-answering. Nowadays, LLM is becoming a very popular tool in computerized language processing tasks, with the capability to analyze complicated linguistic patterns and provide relevant and appropriate responses depending on the context. While offering significant advantages, these models are also vulnerable to security and privacy attacks, such as jailbreaking attacks, data poisoning attacks, and Personally Identifiable Information (PII) leakage attacks. This survey provides a thorough review of the security and privacy challenges of LLMs for both training data and users, along with the application-based risks in various domains, such as transportation, education, and healthcare. We assess the extent of LLM vulnerabilities, investigate emerging security and privacy attacks for LLMs, and review the potential defense mechanisms. Additionally, the survey outlines existing research gaps in this domain and highlights future research directions.

摘要: 大型语言模型(LLM)显示了非凡的能力，并在多个领域做出了贡献，如生成和汇总文本、语言翻译和问题回答。如今，LLM正在成为计算机语言处理任务中非常流行的工具，它能够分析复杂的语言模式，并根据语境提供相关和适当的回应。在提供显著优势的同时，这些模型也容易受到安全和隐私攻击，例如越狱攻击、数据中毒攻击和个人身份信息(PII)泄漏攻击。该调查全面回顾了LLMS对培训数据和用户的安全和隐私挑战，以及交通、教育和医疗保健等各个领域的基于应用的风险。我们评估LLM漏洞的程度，调查LLM新出现的安全和隐私攻击，并审查潜在的防御机制。此外，调查还概述了该领域存在的研究差距，并强调了未来的研究方向。



## **21. AutoDefense: Multi-Agent LLM Defense against Jailbreak Attacks**

AutoDefense：针对越狱攻击的多代理LLM防御 cs.LG

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2403.04783v2) [paper-pdf](http://arxiv.org/pdf/2403.04783v2)

**Authors**: Yifan Zeng, Yiran Wu, Xiao Zhang, Huazheng Wang, Qingyun Wu

**Abstract**: Despite extensive pre-training in moral alignment to prevent generating harmful information, large language models (LLMs) remain vulnerable to jailbreak attacks. In this paper, we propose AutoDefense, a multi-agent defense framework that filters harmful responses from LLMs. With the response-filtering mechanism, our framework is robust against different jailbreak attack prompts, and can be used to defend different victim models. AutoDefense assigns different roles to LLM agents and employs them to complete the defense task collaboratively. The division in tasks enhances the overall instruction-following of LLMs and enables the integration of other defense components as tools. With AutoDefense, small open-source LMs can serve as agents and defend larger models against jailbreak attacks. Our experiments show that AutoDefense can effectively defense against different jailbreak attacks, while maintaining the performance at normal user request. For example, we reduce the attack success rate on GPT-3.5 from 55.74% to 7.95% using LLaMA-2-13b with a 3-agent system. Our code and data are publicly available at https://github.com/XHMY/AutoDefense.

摘要: 尽管在道德一致性方面进行了广泛的预先培训，以防止产生有害信息，但大型语言模型(LLM)仍然容易受到越狱攻击。在本文中，我们提出了一种过滤来自LLMS的有害响应的多代理防御框架--AutoDefense。通过响应过滤机制，我们的框架对不同的越狱攻击提示具有健壮性，并且可以用于防御不同的受害者模型。AutoDefense为LLM特工分配不同的角色，并雇用他们协作完成防御任务。任务分工加强了LLMS的整体指令遵循，并使其他防御组件能够作为工具进行集成。有了AutoDefense，小型开源LMS可以作为代理，保护较大的模型免受越狱攻击。我们的实验表明，AutoDefense能够有效地防御不同的越狱攻击，同时保持正常用户请求的性能。例如，我们使用带有3代理系统的Llama-2-13b将对GPT-3.5的攻击成功率从55.74%降低到7.95%。我们的代码和数据在https://github.com/XHMY/AutoDefense.上公开提供



## **22. DROJ: A Prompt-Driven Attack against Large Language Models**

DROJ：针对大型语言模型的预算驱动攻击 cs.CL

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2411.09125v1) [paper-pdf](http://arxiv.org/pdf/2411.09125v1)

**Authors**: Leyang Hu, Boran Wang

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional capabilities across various natural language processing tasks. Due to their training on internet-sourced datasets, LLMs can sometimes generate objectionable content, necessitating extensive alignment with human feedback to avoid such outputs. Despite massive alignment efforts, LLMs remain susceptible to adversarial jailbreak attacks, which usually are manipulated prompts designed to circumvent safety mechanisms and elicit harmful responses. Here, we introduce a novel approach, Directed Rrepresentation Optimization Jailbreak (DROJ), which optimizes jailbreak prompts at the embedding level to shift the hidden representations of harmful queries towards directions that are more likely to elicit affirmative responses from the model. Our evaluations on LLaMA-2-7b-chat model show that DROJ achieves a 100\% keyword-based Attack Success Rate (ASR), effectively preventing direct refusals. However, the model occasionally produces repetitive and non-informative responses. To mitigate this, we introduce a helpfulness system prompt that enhances the utility of the model's responses. Our code is available at https://github.com/Leon-Leyang/LLM-Safeguard.

摘要: 大型语言模型(LLM)在各种自然语言处理任务中表现出了非凡的能力。由于对来自互联网的数据集进行了培训，LLM有时会产生令人反感的内容，需要与人类反馈广泛协调，以避免此类输出。尽管做出了巨大的调整努力，但LLM仍然容易受到对抗性越狱攻击，这些攻击通常是被操纵的提示，旨在绕过安全机制并引发有害反应。在这里，我们介绍了一种新的方法，定向R表示优化越狱(DROJ)，它在嵌入级别优化越狱提示，将有害查询的隐藏表示向更有可能引起模型肯定响应的方向移动。对Llama-2-7b-Chat模型的评估表明，DROJ达到了100%的基于关键字的攻击成功率，有效地防止了直接拒绝。然而，该模型偶尔会产生重复的、非信息性的回答。为了缓解这一问题，我们引入了一个帮助系统提示，以增强模型响应的实用性。我们的代码可以在https://github.com/Leon-Leyang/LLM-Safeguard.上找到



## **23. Trustful LLMs: Customizing and Grounding Text Generation with Knowledge Bases and Dual Decoders**

值得信赖的LLM：使用知识库和双解码器定制和基础文本生成 cs.CL

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2411.07870v2) [paper-pdf](http://arxiv.org/pdf/2411.07870v2)

**Authors**: Xiaofeng Zhu, Jaya Krishna Mandivarapu

**Abstract**: Although people are impressed by the content generation skills of large language models, the use of LLMs, such as ChatGPT, is limited by the domain grounding of the content. The correctness and groundedness of the generated content need to be based on a verified context, such as results from Retrieval-Augmented Generation (RAG). One important issue when adapting LLMs to a customized domain is that the generated responses are often incomplete, or the additions are not verified and may even be hallucinated. Prior studies on hallucination detection have focused on evaluation metrics, which are not easily adaptable to dynamic domains and can be vulnerable to attacks like jail-breaking. In this work, we propose 1) a post-processing algorithm that leverages knowledge triplets in RAG context to correct hallucinations and 2) a dual-decoder model that fuses RAG context to guide the generation process.

摘要: 尽管人们对大型语言模型的内容生成技能印象深刻，但ChatGPT等LLM的使用受到内容领域基础的限制。生成内容的正确性和可信度需要基于经过验证的上下文，例如检索增强生成（RAG）的结果。将LLM调整到定制域时的一个重要问题是，生成的响应通常不完整，或者添加未经验证，甚至可能出现幻觉。之前关于幻觉检测的研究集中在评估指标上，这些指标不容易适应动态领域，并且容易受到越狱等攻击。在这项工作中，我们提出了1）一种利用RAG上下文中的知识三重组来纠正幻觉的后处理算法，以及2）一种融合RAG上下文来指导生成过程的双解码器模型。



## **24. Defending Large Language Models Against Attacks With Residual Stream Activation Analysis**

利用剩余流激活分析防御大型语言模型免受攻击 cs.CR

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2406.03230v4) [paper-pdf](http://arxiv.org/pdf/2406.03230v4)

**Authors**: Amelia Kawasaki, Andrew Davis, Houssam Abbas

**Abstract**: The widespread adoption of Large Language Models (LLMs), exemplified by OpenAI's ChatGPT, brings to the forefront the imperative to defend against adversarial threats on these models. These attacks, which manipulate an LLM's output by introducing malicious inputs, undermine the model's integrity and the trust users place in its outputs. In response to this challenge, our paper presents an innovative defensive strategy, given white box access to an LLM, that harnesses residual activation analysis between transformer layers of the LLM. We apply a novel methodology for analyzing distinctive activation patterns in the residual streams for attack prompt classification. We curate multiple datasets to demonstrate how this method of classification has high accuracy across multiple types of attack scenarios, including our newly-created attack dataset. Furthermore, we enhance the model's resilience by integrating safety fine-tuning techniques for LLMs in order to measure its effect on our capability to detect attacks. The results underscore the effectiveness of our approach in enhancing the detection and mitigation of adversarial inputs, advancing the security framework within which LLMs operate.

摘要: 大型语言模型(LLM)的广泛采用，如OpenAI的ChatGPT，使防御这些模型上的对手威胁成为当务之急。这些攻击通过引入恶意输入来操纵LLM的输出，破坏了模型的完整性和用户对其输出的信任。为了应对这一挑战，我们的论文提出了一种创新的防御策略，在白盒访问LLM的情况下，该策略利用LLM变压器层之间的剩余激活分析。我们应用了一种新的方法来分析残留流中独特的激活模式，以进行攻击提示分类。我们精选了多个数据集，以演示此分类方法如何在多种类型的攻击场景中具有高精度，包括我们新创建的攻击数据集。此外，我们通过集成LLMS的安全微调技术来增强模型的弹性，以衡量其对我们检测攻击的能力的影响。这些结果强调了我们的方法在加强对敌对输入的检测和缓解、推进LLMS运作的安全框架方面的有效性。



## **25. LLMStinger: Jailbreaking LLMs using RL fine-tuned LLMs**

LLMStinger：使用RL微调的LLM越狱LLM cs.LG

Accepted at AAAI 2025

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2411.08862v1) [paper-pdf](http://arxiv.org/pdf/2411.08862v1)

**Authors**: Piyush Jha, Arnav Arora, Vijay Ganesh

**Abstract**: We introduce LLMStinger, a novel approach that leverages Large Language Models (LLMs) to automatically generate adversarial suffixes for jailbreak attacks. Unlike traditional methods, which require complex prompt engineering or white-box access, LLMStinger uses a reinforcement learning (RL) loop to fine-tune an attacker LLM, generating new suffixes based on existing attacks for harmful questions from the HarmBench benchmark. Our method significantly outperforms existing red-teaming approaches (we compared against 15 of the latest methods), achieving a +57.2% improvement in Attack Success Rate (ASR) on LLaMA2-7B-chat and a +50.3% ASR increase on Claude 2, both models known for their extensive safety measures. Additionally, we achieved a 94.97% ASR on GPT-3.5 and 99.4% on Gemma-2B-it, demonstrating the robustness and adaptability of LLMStinger across open and closed-source models.

摘要: 我们引入了LLMStinger，这是一种利用大型语言模型（LLM）自动生成越狱攻击的对抗性后缀的新颖方法。与需要复杂的即时工程或白盒访问的传统方法不同，LLMStinger使用强化学习（RL）循环来微调攻击者LLM，根据HarmBench基准中针对有害问题的现有攻击生成新的后缀。我们的方法显着优于现有的红色团队方法（我们与15种最新方法进行了比较），在LLaMA 2 - 7 B-chat上实现了攻击成功率（ASB）+57.2%的提高，在Claude 2上实现了攻击成功率（ASB）+50.3%的提高，这两种型号都以其广泛的安全措施而闻名。此外，我们在GPT-3.5上实现了94.97%的ASB，在Gemma-2B-it上实现了99.4%的ASB，证明了LLMStinger在开放和封闭源模型中的稳健性和适应性。



## **26. Target-driven Attack for Large Language Models**

针对大型语言模型的目标驱动攻击 cs.CL

12 pages, 7 figures. This work is an extension of the  arXiv:2404.07234 work. We propose new methods. 27th European Conference on  Artificial Intelligence 2024

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2411.07268v2) [paper-pdf](http://arxiv.org/pdf/2411.07268v2)

**Authors**: Chong Zhang, Mingyu Jin, Dong Shu, Taowen Wang, Dongfang Liu, Xiaobo Jin

**Abstract**: Current large language models (LLM) provide a strong foundation for large-scale user-oriented natural language tasks. Many users can easily inject adversarial text or instructions through the user interface, thus causing LLM model security challenges like the language model not giving the correct answer. Although there is currently a large amount of research on black-box attacks, most of these black-box attacks use random and heuristic strategies. It is unclear how these strategies relate to the success rate of attacks and thus effectively improve model robustness. To solve this problem, we propose our target-driven black-box attack method to maximize the KL divergence between the conditional probabilities of the clean text and the attack text to redefine the attack's goal. We transform the distance maximization problem into two convex optimization problems based on the attack goal to solve the attack text and estimate the covariance. Furthermore, the projected gradient descent algorithm solves the vector corresponding to the attack text. Our target-driven black-box attack approach includes two attack strategies: token manipulation and misinformation attack. Experimental results on multiple Large Language Models and datasets demonstrate the effectiveness of our attack method.

摘要: 现有的大型语言模型(LLM)为大规模面向用户的自然语言任务提供了坚实的基础。许多用户可以很容易地通过用户界面注入敌意文本或指令，从而导致LLM模型的安全挑战，如语言模型无法给出正确的答案。虽然目前有大量关于黑盒攻击的研究，但这些黑盒攻击大多采用随机和启发式策略。目前尚不清楚这些策略如何与攻击成功率相关，从而有效地提高模型的健壮性。为了解决这一问题，我们提出了目标驱动的黑盒攻击方法，以最大化明文和攻击文本的条件概率之间的KL偏差，从而重新定义攻击的目标。将距离最大化问题转化为基于攻击目标的两个凸优化问题来求解攻击文本并估计协方差。此外，投影梯度下降算法求解与攻击文本对应的向量。我们的目标驱动的黑盒攻击方法包括两种攻击策略：令牌操纵和错误信息攻击。在多个大型语言模型和数据集上的实验结果证明了该攻击方法的有效性。



## **27. DAGER: Exact Gradient Inversion for Large Language Models**

DAGER：大型语言模型的精确梯度倒置 cs.LG

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2405.15586v2) [paper-pdf](http://arxiv.org/pdf/2405.15586v2)

**Authors**: Ivo Petrov, Dimitar I. Dimitrov, Maximilian Baader, Mark Niklas Müller, Martin Vechev

**Abstract**: Federated learning works by aggregating locally computed gradients from multiple clients, thus enabling collaborative training without sharing private client data. However, prior work has shown that the data can actually be recovered by the server using so-called gradient inversion attacks. While these attacks perform well when applied on images, they are limited in the text domain and only permit approximate reconstruction of small batches and short input sequences. In this work, we propose DAGER, the first algorithm to recover whole batches of input text exactly. DAGER leverages the low-rank structure of self-attention layer gradients and the discrete nature of token embeddings to efficiently check if a given token sequence is part of the client data. We use this check to exactly recover full batches in the honest-but-curious setting without any prior on the data for both encoder- and decoder-based architectures using exhaustive heuristic search and a greedy approach, respectively. We provide an efficient GPU implementation of DAGER and show experimentally that it recovers full batches of size up to 128 on large language models (LLMs), beating prior attacks in speed (20x at same batch size), scalability (10x larger batches), and reconstruction quality (ROUGE-1/2 > 0.99).

摘要: 联合学习的工作方式是聚合来自多个客户端的本地计算的梯度，从而在不共享私人客户端数据的情况下实现协作培训。然而，先前的工作表明，服务器实际上可以使用所谓的梯度反转攻击来恢复数据。虽然这些攻击在图像上应用时表现良好，但它们仅限于文本域，仅允许对小批次和短输入序列进行近似重建。在这项工作中，我们提出了第一个准确恢复整批输入文本的算法Dager。Dager利用自我关注层梯度的低等级结构和令牌嵌入的离散性质来有效地检查给定的令牌序列是否是客户端数据的一部分。我们使用这种检查，分别使用穷举启发式搜索和贪婪方法，在诚实但奇怪的设置中准确地恢复完整批次，而不需要对基于编码器和解码器的架构的数据进行任何先验。我们提供了一种高效的Dager的GPU实现，实验表明，它可以在大型语言模型(LLM)上恢复大小高达128的全批处理，在速度(相同批处理大小的20倍)、可伸缩性(大批处理10倍)和重建质量(Rouge-1/2>0.99)方面优于先前的攻击。



## **28. The VLLM Safety Paradox: Dual Ease in Jailbreak Attack and Defense**

VLLM安全悖论：越狱攻击和防御的双重轻松 cs.CR

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2411.08410v1) [paper-pdf](http://arxiv.org/pdf/2411.08410v1)

**Authors**: Yangyang Guo, Fangkai Jiao, Liqiang Nie, Mohan Kankanhalli

**Abstract**: The vulnerability of Vision Large Language Models (VLLMs) to jailbreak attacks appears as no surprise. However, recent defense mechanisms against these attacks have reached near-saturation performance on benchmarks, often with minimal effort. This simultaneous high performance in both attack and defense presents a perplexing paradox. Resolving it is critical for advancing the development of trustworthy models. To address this research gap, we first investigate why VLLMs are prone to these attacks. We then make a key observation: existing defense mechanisms suffer from an \textbf{over-prudence} problem, resulting in unexpected abstention even in the presence of benign inputs. Additionally, we find that the two representative evaluation methods for jailbreak often exhibit chance agreement. This limitation makes it potentially misleading when evaluating attack strategies or defense mechanisms. Beyond these empirical observations, our another contribution in this work is to repurpose the guardrails of LLMs on the shelf, as an effective alternative detector prior to VLLM response. We believe these findings offer useful insights to rethink the foundational development of VLLM safety with respect to benchmark datasets, evaluation methods, and defense strategies.

摘要: Vision Large Language Models(VLLM)在越狱攻击中的脆弱性似乎并不令人意外。然而，最近针对这些攻击的防御机制在基准测试中的性能已经接近饱和，通常只需很少的努力。这种同时在进攻和防守上的高表现提出了一个令人困惑的悖论。解决这一问题对于推动可信模型的发展至关重要。为了解决这一研究差距，我们首先调查了为什么VLLM容易受到这些攻击。然后，我们做了一个关键的观察：现有的防御机制存在过度谨慎的问题，导致即使存在良性投入，也会意外弃权。此外，我们发现，两种具有代表性的越狱评估方法往往表现出偶然性的一致性。这一限制使其在评估攻击策略或防御机制时具有潜在误导性。除了这些经验观察之外，我们在这项工作中的另一个贡献是重新利用架子上的LLM护栏，作为VLLM响应之前的有效替代探测器。我们相信，这些发现为重新思考VLLM安全性在基准数据集、评估方法和防御策略方面的基础性发展提供了有用的见解。



## **29. MultiKG: Multi-Source Threat Intelligence Aggregation for High-Quality Knowledge Graph Representation of Attack Techniques**

MultiKG：用于攻击技术的高质量知识图表示的多源威胁情报聚合 cs.CR

21 pages, 15 figures, 8 tables

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2411.08359v1) [paper-pdf](http://arxiv.org/pdf/2411.08359v1)

**Authors**: Jian Wang, Tiantian Zhu, Chunlin Xiong, Yan Chen

**Abstract**: The construction of attack technique knowledge graphs aims to transform various types of attack knowledge into structured representations for more effective attack procedure modeling. Existing methods typically rely on textual data, such as Cyber Threat Intelligence (CTI) reports, which are often coarse-grained and unstructured, resulting in incomplete and inaccurate knowledge graphs. To address these issues, we expand attack knowledge sources by incorporating audit logs and static code analysis alongside CTI reports, providing finer-grained data for constructing attack technique knowledge graphs.   We propose MultiKG, a fully automated framework that integrates multiple threat knowledge sources. MultiKG processes data from CTI reports, dynamic logs, and static code separately, then merges them into a unified attack knowledge graph. Through system design and the utilization of the Large Language Model (LLM), MultiKG automates the analysis, construction, and merging of attack graphs across these sources, producing a fine-grained, multi-source attack knowledge graph.   We implemented MultiKG and evaluated it using 1,015 real attack techniques and 9,006 attack intelligence entries from CTI reports. Results show that MultiKG effectively extracts attack knowledge graphs from diverse sources and aggregates them into accurate, comprehensive representations. Through case studies, we demonstrate that our approach directly benefits security tasks such as attack reconstruction and detection.

摘要: 攻击技术知识图的构建旨在将各种类型的攻击知识转化为结构化的表示形式，以便更有效地对攻击过程进行建模。现有方法通常依赖文本数据，如网络威胁情报(CTI)报告，这些数据通常是粗粒度和非结构化的，导致知识图谱不完整和不准确。为了解决这些问题，我们通过将审计日志和静态代码分析与CTI报告结合在一起来扩展攻击知识源，为构建攻击技术知识图提供更细粒度的数据。我们提出了一种集成多个威胁知识源的全自动化框架--MultiKG。MultiKG分别处理CTI报告、动态日志和静态代码中的数据，然后将它们合并到统一的攻击知识图中。通过系统设计和大型语言模型(LLM)的利用，MultiKG自动分析、构建和合并这些来源的攻击图，生成细粒度的多源攻击知识图。我们实施了MultiKG，并使用1,015项真实攻击技术和9,006个CTI报告中的攻击情报条目对其进行了评估。结果表明，MultiKG能有效地从不同来源提取攻击知识图，并将其聚合成准确、全面的表示。通过案例研究，我们证明了我们的方法直接有利于攻击重建和检测等安全任务。



## **30. Can adversarial attacks by large language models be attributed?**

大型语言模型的对抗攻击可以归因吗？ cs.AI

7 pages, 1 figure

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.08003v1) [paper-pdf](http://arxiv.org/pdf/2411.08003v1)

**Authors**: Manuel Cebrian, Jan Arne Telle

**Abstract**: Attributing outputs from Large Language Models (LLMs) in adversarial settings-such as cyberattacks and disinformation-presents significant challenges that are likely to grow in importance. We investigate this attribution problem using formal language theory, specifically language identification in the limit as introduced by Gold and extended by Angluin. By modeling LLM outputs as formal languages, we analyze whether finite text samples can uniquely pinpoint the originating model. Our results show that due to the non-identifiability of certain language classes, under some mild assumptions about overlapping outputs from fine-tuned models it is theoretically impossible to attribute outputs to specific LLMs with certainty. This holds also when accounting for expressivity limitations of Transformer architectures. Even with direct model access or comprehensive monitoring, significant computational hurdles impede attribution efforts. These findings highlight an urgent need for proactive measures to mitigate risks posed by adversarial LLM use as their influence continues to expand.

摘要: 将大型语言模型(LLM)的输出归因于敌对环境--如网络攻击和虚假信息--带来了重大挑战，而这些挑战的重要性可能会越来越大。我们使用形式化语言理论来研究这一归因问题，特别是Gold提出并由Anluin推广的极限语言识别问题。通过将LLM输出建模为形式语言，我们分析了有限文本样本是否能够唯一地定位原始模型。我们的结果表明，由于某些语言类别的不可识别性，在微调模型的输出重叠的一些温和假设下，理论上不可能确定地将输出归因于特定的LLM。当考虑到Transformer架构的表现力限制时，这也是成立的。即使有了直接的模型访问或全面的监测，重大的计算障碍也阻碍了归因努力。这些调查结果突出表明，迫切需要采取积极主动的措施，以减轻敌对使用LLM所带来的风险，因为它们的影响继续扩大。



## **31. Chain Association-based Attacking and Shielding Natural Language Processing Systems**

基于链关联的攻击和屏蔽自然语言处理系统 cs.CL

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07843v1) [paper-pdf](http://arxiv.org/pdf/2411.07843v1)

**Authors**: Jiacheng Huang, Long Chen

**Abstract**: Association as a gift enables people do not have to mention something in completely straightforward words and allows others to understand what they intend to refer to. In this paper, we propose a chain association-based adversarial attack against natural language processing systems, utilizing the comprehension gap between humans and machines. We first generate a chain association graph for Chinese characters based on the association paradigm for building search space of potential adversarial examples. Then, we introduce an discrete particle swarm optimization algorithm to search for the optimal adversarial examples. We conduct comprehensive experiments and show that advanced natural language processing models and applications, including large language models, are vulnerable to our attack, while humans appear good at understanding the perturbed text. We also explore two methods, including adversarial training and associative graph-based recovery, to shield systems from chain association-based attack. Since a few examples that use some derogatory terms, this paper contains materials that may be offensive or upsetting to some people.

摘要: 联想作为一种礼物，使人们不必用完全直截了当的语言来提及某事，并让其他人理解他们想指的是什么。本文利用人与机器之间的理解鸿沟，提出了一种基于链式联想的对抗性自然语言处理系统攻击方法。首先在联想范式的基础上生成汉字的链式联想图，构建潜在对抗性实例的搜索空间。然后，我们引入了离散粒子群优化算法来搜索最优的对抗性实例。我们进行了全面的实验，并表明高级自然语言处理模型和应用程序，包括大型语言模型，容易受到我们的攻击，而人类似乎很擅长理解受干扰的文本。我们还探索了两种方法，包括对抗性训练和基于联想图的恢复，以保护系统免受基于链关联的攻击。由于有几个例子使用了一些贬义性的术语，因此本文包含的材料可能会冒犯某些人或使某些人不安。



## **32. Revisiting the Adversarial Robustness of Vision Language Models: a Multimodal Perspective**

重新审视视觉语言模型的对抗鲁棒性：多模式视角 cs.CV

17 pages, 13 figures

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2404.19287v3) [paper-pdf](http://arxiv.org/pdf/2404.19287v3)

**Authors**: Wanqi Zhou, Shuanghao Bai, Danilo P. Mandic, Qibin Zhao, Badong Chen

**Abstract**: Pretrained vision-language models (VLMs) like CLIP exhibit exceptional generalization across diverse downstream tasks. While recent studies reveal their vulnerability to adversarial attacks, research to date has primarily focused on enhancing the robustness of image encoders against image-based attacks, with defenses against text-based and multimodal attacks remaining largely unexplored. To this end, this work presents the first comprehensive study on improving the adversarial robustness of VLMs against attacks targeting image, text, and multimodal inputs. This is achieved by proposing multimodal contrastive adversarial training (MMCoA). Such an approach strengthens the robustness of both image and text encoders by aligning the clean text embeddings with adversarial image embeddings, and adversarial text embeddings with clean image embeddings. The robustness of the proposed MMCoA is examined against existing defense methods over image, text, and multimodal attacks on the CLIP model. Extensive experiments on 15 datasets across two tasks reveal the characteristics of different adversarial defense methods under distinct distribution shifts and dataset complexities across the three attack types. This paves the way for a unified framework of adversarial robustness against different modality attacks, opening up new possibilities for securing VLMs against multimodal attacks. The code is available at https://github.com/ElleZWQ/MMCoA.git.

摘要: 像CLIP这样的预先训练的视觉语言模型(VLM)在不同的下游任务中表现出非凡的通用性。虽然最近的研究揭示了它们对对手攻击的脆弱性，但到目前为止的研究主要集中在增强图像编码器对基于图像的攻击的稳健性上，对基于文本的攻击和多模式攻击的防御在很大程度上仍未被探索。为此，本文首次全面研究了如何提高VLMS对图像、文本和多模式输入的攻击健壮性。这是通过提出多模式对比对抗训练(MMCoA)来实现的。这种方法通过将干净的文本嵌入与对抗性的图像嵌入以及对抗性的文本嵌入与干净的图像嵌入对齐来增强图像和文本编码器的稳健性。针对已有的针对图像、文本和多模式攻击的防御方法，对提出的MMCoA算法的鲁棒性进行了测试。在两个任务的15个数据集上进行了大量的实验，揭示了三种攻击类型在不同的分布变化和数据集复杂性下不同的对抗防御方法的特点。这为对抗不同模式攻击的对抗健壮性的统一框架铺平了道路，为保护VLM免受多模式攻击开辟了新的可能性。代码可在https://github.com/ElleZWQ/MMCoA.git.上获得



## **33. Zer0-Jack: A Memory-efficient Gradient-based Jailbreaking Method for Black-box Multi-modal Large Language Models**

Zer 0-Jack：一种用于黑匣子多模式大型语言模型的内存高效的基于对象的越狱方法 cs.LG

Accepted to Neurips SafeGenAi Workshop 2024

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07559v1) [paper-pdf](http://arxiv.org/pdf/2411.07559v1)

**Authors**: Tiejin Chen, Kaishen Wang, Hua Wei

**Abstract**: Jailbreaking methods, which induce Multi-modal Large Language Models (MLLMs) to output harmful responses, raise significant safety concerns. Among these methods, gradient-based approaches, which use gradients to generate malicious prompts, have been widely studied due to their high success rates in white-box settings, where full access to the model is available. However, these methods have notable limitations: they require white-box access, which is not always feasible, and involve high memory usage. To address scenarios where white-box access is unavailable, attackers often resort to transfer attacks. In transfer attacks, malicious inputs generated using white-box models are applied to black-box models, but this typically results in reduced attack performance. To overcome these challenges, we propose Zer0-Jack, a method that bypasses the need for white-box access by leveraging zeroth-order optimization. We propose patch coordinate descent to efficiently generate malicious image inputs to directly attack black-box MLLMs, which significantly reduces memory usage further. Through extensive experiments, Zer0-Jack achieves a high attack success rate across various models, surpassing previous transfer-based methods and performing comparably with existing white-box jailbreak techniques. Notably, Zer0-Jack achieves a 95\% attack success rate on MiniGPT-4 with the Harmful Behaviors Multi-modal Dataset on a black-box setting, demonstrating its effectiveness. Additionally, we show that Zer0-Jack can directly attack commercial MLLMs such as GPT-4o. Codes are provided in the supplement.

摘要: 越狱方法会导致多模式大型语言模型(MLLMS)产生有害的响应，引发了重大的安全问题。在这些方法中，基于梯度的方法使用梯度来生成恶意提示，由于其在白盒环境中的高成功率而得到了广泛的研究，在白盒环境中，完全可以访问模型。然而，这些方法有明显的局限性：它们需要白盒访问，这并不总是可行的，并且涉及高内存使用率。为了解决无法使用白盒访问的情况，攻击者通常会求助于传输攻击。在传输攻击中，使用白盒模型生成的恶意输入应用于黑盒模型，但这通常会导致攻击性能降低。为了克服这些挑战，我们提出了Zer0-Jack，这是一种通过利用零阶优化来绕过白盒访问的方法。我们提出了补丁坐标下降的方法来有效地生成恶意图像输入来直接攻击黑盒MLLMS，从而进一步显著地减少了内存使用量。通过广泛的实验，Zer0-Jack在各种模型上实现了高攻击成功率，超过了以前基于传输的方法，性能与现有的白盒越狱技术相当。值得注意的是，Zer0-Jack在黑盒设置的有害行为多模式数据集上对MiniGPT-4的攻击成功率达到了95%，证明了其有效性。此外，我们还证明了Zer0-Jack可以直接攻击GPT-40等商业MLLMS。附录中提供了代码。



## **34. On Active Privacy Auditing in Supervised Fine-tuning for White-Box Language Models**

白盒语言模型监督微调中的主动隐私审计 cs.CL

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07070v2) [paper-pdf](http://arxiv.org/pdf/2411.07070v2)

**Authors**: Qian Sun, Hanpeng Wu, Xi Sheryl Zhang

**Abstract**: The pretraining and fine-tuning approach has become the leading technique for various NLP applications. However, recent studies reveal that fine-tuning data, due to their sensitive nature, domain-specific characteristics, and identifiability, pose significant privacy concerns. To help develop more privacy-resilient fine-tuning models, we introduce a novel active privacy auditing framework, dubbed Parsing, designed to identify and quantify privacy leakage risks during the supervised fine-tuning (SFT) of language models (LMs). The framework leverages improved white-box membership inference attacks (MIAs) as the core technology, utilizing novel learning objectives and a two-stage pipeline to monitor the privacy of the LMs' fine-tuning process, maximizing the exposure of privacy risks. Additionally, we have improved the effectiveness of MIAs on large LMs including GPT-2, Llama2, and certain variants of them. Our research aims to provide the SFT community of LMs with a reliable, ready-to-use privacy auditing tool, and to offer valuable insights into safeguarding privacy during the fine-tuning process. Experimental results confirm the framework's efficiency across various models and tasks, emphasizing notable privacy concerns in the fine-tuning process. Project code available for https://anonymous.4open.science/r/PARSING-4817/.

摘要: 预训练和微调方法已成为各种NLP应用的主导技术。然而，最近的研究表明，由于数据的敏感性质、特定于领域的特征和可识别性，微调数据会带来严重的隐私问题。为了帮助开发更具隐私弹性的微调模型，我们引入了一个新的主动隐私审计框架，称为Parsing，旨在识别和量化语言模型(LMS)的监督微调(SFT)期间的隐私泄露风险。该框架利用改进的白盒成员关系推理攻击(MIA)作为核心技术，利用新的学习目标和两阶段管道来监控LMS微调过程的隐私，最大限度地增加隐私风险的暴露。此外，我们还改进了MIA在大型LMS上的有效性，包括GPT-2、Llama2及其某些变体。我们的研究旨在为LMS的SFT社区提供一个可靠的、随时可用的隐私审计工具，并为在微调过程中保护隐私提供有价值的见解。实验结果证实了该框架在各种模型和任务上的有效性，强调了在微调过程中值得注意的隐私问题。可用于https://anonymous.4open.science/r/PARSING-4817/.的项目代码



## **35. vTune: Verifiable Fine-Tuning for LLMs Through Backdooring**

VCE：通过Backdooring对LLM进行可验证的微调 cs.LG

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.06611v2) [paper-pdf](http://arxiv.org/pdf/2411.06611v2)

**Authors**: Eva Zhang, Arka Pal, Akilesh Potti, Micah Goldblum

**Abstract**: As fine-tuning large language models (LLMs) becomes increasingly prevalent, users often rely on third-party services with limited visibility into their fine-tuning processes. This lack of transparency raises the question: how do consumers verify that fine-tuning services are performed correctly? For instance, a service provider could claim to fine-tune a model for each user, yet simply send all users back the same base model. To address this issue, we propose vTune, a simple method that uses a small number of backdoor data points added to the training data to provide a statistical test for verifying that a provider fine-tuned a custom model on a particular user's dataset. Unlike existing works, vTune is able to scale to verification of fine-tuning on state-of-the-art LLMs, and can be used both with open-source and closed-source models. We test our approach across several model families and sizes as well as across multiple instruction-tuning datasets, and find that the statistical test is satisfied with p-values on the order of $\sim 10^{-40}$, with no negative impact on downstream task performance. Further, we explore several attacks that attempt to subvert vTune and demonstrate the method's robustness to these attacks.

摘要: 随着微调大型语言模型(LLM)变得越来越普遍，用户通常依赖于第三方服务，但对其微调过程的可见性有限。这种透明度的缺乏引发了一个问题：消费者如何验证微调服务是否正确执行？例如，服务提供商可以声称为每个用户微调一个型号，但只需将所有用户发送回相同的基本型号。为了解决这个问题，我们提出了vTune，这是一种简单的方法，它使用添加到训练数据的少量后门数据点来提供统计测试，以验证提供商是否对特定用户的数据集的自定义模型进行了微调。与现有的作品不同，vTune能够扩展到对最先进的LLM进行微调验证，并且可以与开源和封闭源代码模型一起使用。我们在几个模型系列和大小以及多个指令调优数据集上测试了我们的方法，发现统计测试满足p值的数量级为$\sim 10^{-40}$，并且不会对下游任务性能产生负面影响。进一步，我们研究了几种试图破坏vTune的攻击，并展示了该方法对这些攻击的健壮性。



## **36. Rapid Response: Mitigating LLM Jailbreaks with a Few Examples**

快速反应：通过一些例子缓解LLM越狱 cs.CL

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07494v1) [paper-pdf](http://arxiv.org/pdf/2411.07494v1)

**Authors**: Alwin Peng, Julian Michael, Henry Sleight, Ethan Perez, Mrinank Sharma

**Abstract**: As large language models (LLMs) grow more powerful, ensuring their safety against misuse becomes crucial. While researchers have focused on developing robust defenses, no method has yet achieved complete invulnerability to attacks. We propose an alternative approach: instead of seeking perfect adversarial robustness, we develop rapid response techniques to look to block whole classes of jailbreaks after observing only a handful of attacks. To study this setting, we develop RapidResponseBench, a benchmark that measures a defense's robustness against various jailbreak strategies after adapting to a few observed examples. We evaluate five rapid response methods, all of which use jailbreak proliferation, where we automatically generate additional jailbreaks similar to the examples observed. Our strongest method, which fine-tunes an input classifier to block proliferated jailbreaks, reduces attack success rate by a factor greater than 240 on an in-distribution set of jailbreaks and a factor greater than 15 on an out-of-distribution set, having observed just one example of each jailbreaking strategy. Moreover, further studies suggest that the quality of proliferation model and number of proliferated examples play an key role in the effectiveness of this defense. Overall, our results highlight the potential of responding rapidly to novel jailbreaks to limit LLM misuse.

摘要: 随着大型语言模型(LLM)变得越来越强大，确保它们的安全性以防止误用变得至关重要。虽然研究人员专注于开发强大的防御系统，但还没有一种方法能够完全抵御攻击。我们提出了另一种方法：我们不是寻求完美的对手健壮性，而是开发快速响应技术，在仅观察到少数几次攻击后，寻求阻止整个类别的越狱。为了研究这种情况，我们开发了RapidResponseBch，这是一个基准，在适应了几个观察到的例子后，衡量了防御对各种越狱策略的健壮性。我们评估了五种快速响应方法，所有这些方法都使用越狱扩散，在这些方法中，我们自动生成与观察到的示例类似的额外越狱。我们最强大的方法是微调输入分类器以阻止越狱激增，在仅观察到每个越狱策略的一个示例后，在分布内越狱集合上将攻击成功率降低240倍以上，在分布外集合上降低15倍以上。此外，进一步的研究表明，扩散模型的质量和扩散实例的数量在这一防御措施的有效性中起着关键作用。总体而言，我们的结果突出了对新型越狱做出快速反应以限制LLM滥用的潜力。



## **37. DrAttack: Prompt Decomposition and Reconstruction Makes Powerful LLM Jailbreakers**

DrAttack：强大的快速分解和重建让LLM越狱者 cs.CR

**SubmitDate**: 2024-11-11    [abs](http://arxiv.org/abs/2402.16914v3) [paper-pdf](http://arxiv.org/pdf/2402.16914v3)

**Authors**: Xirui Li, Ruochen Wang, Minhao Cheng, Tianyi Zhou, Cho-Jui Hsieh

**Abstract**: The safety alignment of Large Language Models (LLMs) is vulnerable to both manual and automated jailbreak attacks, which adversarially trigger LLMs to output harmful content. However, current methods for jailbreaking LLMs, which nest entire harmful prompts, are not effective at concealing malicious intent and can be easily identified and rejected by well-aligned LLMs. This paper discovers that decomposing a malicious prompt into separated sub-prompts can effectively obscure its underlying malicious intent by presenting it in a fragmented, less detectable form, thereby addressing these limitations. We introduce an automatic prompt \textbf{D}ecomposition and \textbf{R}econstruction framework for jailbreak \textbf{Attack} (DrAttack). DrAttack includes three key components: (a) `Decomposition' of the original prompt into sub-prompts, (b) `Reconstruction' of these sub-prompts implicitly by in-context learning with semantically similar but harmless reassembling demo, and (c) a `Synonym Search' of sub-prompts, aiming to find sub-prompts' synonyms that maintain the original intent while jailbreaking LLMs. An extensive empirical study across multiple open-source and closed-source LLMs demonstrates that, with a significantly reduced number of queries, DrAttack obtains a substantial gain of success rate over prior SOTA prompt-only attackers. Notably, the success rate of 78.0\% on GPT-4 with merely 15 queries surpassed previous art by 33.1\%. The project is available at https://github.com/xirui-li/DrAttack.

摘要: 大型语言模型(LLM)的安全一致性容易受到手动和自动越狱攻击的攻击，这些攻击会相反地触发LLM输出有害内容。然而，当前的越狱LLM方法嵌套了整个有害的提示，在隐藏恶意意图方面并不有效，很容易被排列良好的LLM识别和拒绝。本文发现，通过将恶意提示分解为单独的子提示，可以通过以零散的、较难检测的形式来表示恶意提示，从而有效地掩盖潜在的恶意意图，从而解决这些限制。介绍了一种自动提示的文本bf{D}分解和文本bf{R}重构框架(DrAttack)。DrAttack包括三个关键组件：(A)将原始提示‘分解’成子提示，(B)通过使用语义相似但无害的重组演示的上下文学习隐式地‘重构’这些子提示，以及(C)对子提示进行‘同步搜索’，目的是在越狱LLM的同时找到保持原意的子提示的同义词。一项针对多个开源和闭源LLM的广泛经验研究表明，DrAttack在查询次数显著减少的情况下，与之前的Sota仅提示攻击者相比，获得了显著的成功率。值得注意的是，仅用15个查询在GPT-4上的成功率为78.0\%，比以前的ART高出33.1\%。该项目的网址为：https://github.com/xirui-li/DrAttack.。



## **38. Do Unlearning Methods Remove Information from Language Model Weights?**

取消学习方法会从语言模型权重中删除信息吗？ cs.LG

**SubmitDate**: 2024-11-10    [abs](http://arxiv.org/abs/2410.08827v2) [paper-pdf](http://arxiv.org/pdf/2410.08827v2)

**Authors**: Aghyad Deeb, Fabien Roger

**Abstract**: Large Language Models' knowledge of how to perform cyber-security attacks, create bioweapons, and manipulate humans poses risks of misuse. Previous work has proposed methods to unlearn this knowledge. Historically, it has been unclear whether unlearning techniques are removing information from the model weights or just making it harder to access. To disentangle these two objectives, we propose an adversarial evaluation method to test for the removal of information from model weights: we give an attacker access to some facts that were supposed to be removed, and using those, the attacker tries to recover other facts from the same distribution that cannot be guessed from the accessible facts. We show that using fine-tuning on the accessible facts can recover 88% of the pre-unlearning accuracy when applied to current unlearning methods, revealing the limitations of these methods in removing information from the model weights.

摘要: 大型语言模型关于如何执行网络安全攻击、制造生物武器和操纵人类的知识带来了滥用的风险。之前的工作提出了忘记这些知识的方法。从历史上看，目前尚不清楚取消学习技术是否正在从模型权重中删除信息，或者只是使其更难访问。为了解开这两个目标，我们提出了一种对抗评估方法来测试从模型权重中删除信息的情况：我们让攻击者访问一些应该删除的事实，并使用这些事实，攻击者试图从无法从可访问的事实中猜测到的相同分布中恢复其他事实。我们表明，当应用于当前的取消学习方法时，对可访问的事实进行微调可以恢复取消学习前的88%的准确性，揭示了这些方法在从模型权重中删除信息方面的局限性。



## **39. SequentialBreak: Large Language Models Can be Fooled by Embedding Jailbreak Prompts into Sequential Prompt Chains**

SequentialBreak：大型语言模型可以通过将越狱提示嵌入序列提示链来愚弄 cs.CR

**SubmitDate**: 2024-11-10    [abs](http://arxiv.org/abs/2411.06426v1) [paper-pdf](http://arxiv.org/pdf/2411.06426v1)

**Authors**: Bijoy Ahmed Saiem, MD Sadik Hossain Shanto, Rakib Ahsan, Md Rafi ur Rashid

**Abstract**: As the integration of the Large Language Models (LLMs) into various applications increases, so does their susceptibility to misuse, raising significant security concerns. Numerous jailbreak attacks have been proposed to assess the security defense of LLMs. Current jailbreak attacks mainly rely on scenario camouflage, prompt obfuscation, prompt optimization, and prompt iterative optimization to conceal malicious prompts. In particular, sequential prompt chains in a single query can lead LLMs to focus on certain prompts while ignoring others, facilitating context manipulation. This paper introduces SequentialBreak, a novel jailbreak attack that exploits this vulnerability. We discuss several scenarios, not limited to examples like Question Bank, Dialog Completion, and Game Environment, where the harmful prompt is embedded within benign ones that can fool LLMs into generating harmful responses. The distinct narrative structures of these scenarios show that SequentialBreak is flexible enough to adapt to various prompt formats beyond those discussed. Extensive experiments demonstrate that SequentialBreak uses only a single query to achieve a substantial gain of attack success rate over existing baselines against both open-source and closed-source models. Through our research, we highlight the urgent need for more robust and resilient safeguards to enhance LLM security and prevent potential misuse. All the result files and website associated with this research are available in this GitHub repository: https://anonymous.4open.science/r/JailBreakAttack-4F3B/.

摘要: 随着大型语言模型(LLM)集成到各种应用程序中的增加，它们也更容易被误用，从而引发了重大的安全问题。已经提出了许多越狱攻击来评估LLMS的安全防御。当前越狱攻击主要依靠场景伪装、提示混淆、提示优化、提示迭代优化来隐藏恶意提示。特别是，单个查询中的顺序提示链可能会导致LLM专注于某些提示，而忽略其他提示，从而促进上下文操作。本文介绍了SequentialBreak，一种利用该漏洞的新型越狱攻击。我们讨论了几种场景，不限于题库、对话完成和游戏环境等示例，在这些场景中，有害提示嵌入到良性提示中，可以欺骗LLM生成有害响应。这些场景的不同叙事结构表明，SequentialBreak足够灵活，可以适应所讨论的各种提示格式。大量的实验表明，SequentialBreak只使用一次查询，在开源和封闭源代码模型下，攻击成功率都比现有的基线有很大的提高。通过我们的研究，我们强调迫切需要更强大和更具弹性的保障措施，以增强LLM安全并防止潜在的滥用。所有与这项研究相关的结果文件和网站都可以在GitHub存储库中找到：https://anonymous.4open.science/r/JailBreakAttack-4F3B/.



## **40. Jailbreaking LLM-Controlled Robots**

越狱LLM控制机器人 cs.RO

**SubmitDate**: 2024-11-09    [abs](http://arxiv.org/abs/2410.13691v2) [paper-pdf](http://arxiv.org/pdf/2410.13691v2)

**Authors**: Alexander Robey, Zachary Ravichandran, Vijay Kumar, Hamed Hassani, George J. Pappas

**Abstract**: The recent introduction of large language models (LLMs) has revolutionized the field of robotics by enabling contextual reasoning and intuitive human-robot interaction in domains as varied as manipulation, locomotion, and self-driving vehicles. When viewed as a stand-alone technology, LLMs are known to be vulnerable to jailbreaking attacks, wherein malicious prompters elicit harmful text by bypassing LLM safety guardrails. To assess the risks of deploying LLMs in robotics, in this paper, we introduce RoboPAIR, the first algorithm designed to jailbreak LLM-controlled robots. Unlike existing, textual attacks on LLM chatbots, RoboPAIR elicits harmful physical actions from LLM-controlled robots, a phenomenon we experimentally demonstrate in three scenarios: (i) a white-box setting, wherein the attacker has full access to the NVIDIA Dolphins self-driving LLM, (ii) a gray-box setting, wherein the attacker has partial access to a Clearpath Robotics Jackal UGV robot equipped with a GPT-4o planner, and (iii) a black-box setting, wherein the attacker has only query access to the GPT-3.5-integrated Unitree Robotics Go2 robot dog. In each scenario and across three new datasets of harmful robotic actions, we demonstrate that RoboPAIR, as well as several static baselines, finds jailbreaks quickly and effectively, often achieving 100% attack success rates. Our results reveal, for the first time, that the risks of jailbroken LLMs extend far beyond text generation, given the distinct possibility that jailbroken robots could cause physical damage in the real world. Indeed, our results on the Unitree Go2 represent the first successful jailbreak of a deployed commercial robotic system. Addressing this emerging vulnerability is critical for ensuring the safe deployment of LLMs in robotics. Additional media is available at: https://robopair.org

摘要: 最近引入的大型语言模型(LLM)通过在操作、运动和自动驾驶车辆等各种领域实现上下文推理和直观的人-机器人交互，从而彻底改变了机器人领域。当被视为一项独立的技术时，LLMS已知容易受到越狱攻击，恶意提示器通过绕过LLm安全护栏引发有害文本。为了评估在机器人学中部署LLMS的风险，在本文中，我们引入了RoboPAIR，这是第一个设计用于越狱LLM控制的机器人的算法。与现有对LLM聊天机器人的文本攻击不同，RoboPAIR会引发来自LLM控制的机器人的有害物理操作，我们在三个场景中实验演示了这种现象：(I)白盒设置，其中攻击者对NVIDIA Dolphins自动驾驶LLM具有完全访问权限；(Ii)灰盒设置，其中攻击者对配备GPT-40规划器的ClearPath Robotics Jackal UGV机器人具有部分访问权限；以及(Iii)黑盒设置，其中攻击者只有对GPT-3.5集成的Unitree Robotics Go2机器狗的查询访问权限。在每个场景和三个新的有害机器人操作的数据集上，我们展示了RoboPAIR以及几个静态基线，快速有效地找到越狱，通常达到100%的攻击成功率。我们的结果首次显示，鉴于越狱机器人在现实世界中造成物理损害的明显可能性，越狱机器人的风险远远超出了文本生成的范围。事实上，我们在Unitree Go2上的结果代表着部署的商业机器人系统第一次成功越狱。解决这一新出现的漏洞对于确保在机器人中安全部署LLM至关重要。如需更多媒体，请访问：https://robopair.org。



## **41. Robust Detection of LLM-Generated Text: A Comparative Analysis**

LLM生成文本的稳健检测：比较分析 cs.CL

8 pages

**SubmitDate**: 2024-11-09    [abs](http://arxiv.org/abs/2411.06248v1) [paper-pdf](http://arxiv.org/pdf/2411.06248v1)

**Authors**: Yongye Su, Yuqing Wu

**Abstract**: The ability of large language models to generate complex texts allows them to be widely integrated into many aspects of life, and their output can quickly fill all network resources. As the impact of LLMs grows, it becomes increasingly important to develop powerful detectors for the generated text. This detector is essential to prevent the potential misuse of these technologies and to protect areas such as social media from the negative effects of false content generated by LLMS. The main goal of LLM-generated text detection is to determine whether text is generated by an LLM, which is a basic binary classification task. In our work, we mainly use three different classification methods based on open source datasets: traditional machine learning techniques such as logistic regression, k-means clustering, Gaussian Naive Bayes, support vector machines, and methods based on converters such as BERT, and finally algorithms that use LLMs to detect LLM-generated text. We focus on model generalization, potential adversarial attacks, and accuracy of model evaluation. Finally, the possible research direction in the future is proposed, and the current experimental results are summarized.

摘要: 大型语言模型生成复杂文本的能力使它们能够广泛融入生活的许多方面，它们的输出可以迅速填满所有网络资源。随着LLMS的影响越来越大，为生成的文本开发强大的检测器变得越来越重要。这种检测器对于防止这些技术的潜在滥用以及保护社交媒体等领域免受LLMS产生的虚假内容的负面影响至关重要。LLM生成的文本检测的主要目标是确定文本是否由LLM生成，这是一项基本的二进制分类任务。在我们的工作中，我们主要使用了三种不同的基于开源数据集的分类方法：传统的机器学习技术，如Logistic回归，k-均值聚类，高斯朴素贝叶斯，支持向量机，以及基于转换器的方法，如BERT，最后是使用LLMS来检测LLM生成的文本的算法。我们主要关注模型的泛化、潜在的敌意攻击和模型评估的准确性。最后，提出了未来可能的研究方向，并对目前的实验结果进行了总结。



## **42. Goal-guided Generative Prompt Injection Attack on Large Language Models**

对大型语言模型的目标引导生成提示注入攻击 cs.CR

11 pages, 6 figures

**SubmitDate**: 2024-11-09    [abs](http://arxiv.org/abs/2404.07234v4) [paper-pdf](http://arxiv.org/pdf/2404.07234v4)

**Authors**: Chong Zhang, Mingyu Jin, Qinkai Yu, Chengzhi Liu, Haochen Xue, Xiaobo Jin

**Abstract**: Current large language models (LLMs) provide a strong foundation for large-scale user-oriented natural language tasks. A large number of users can easily inject adversarial text or instructions through the user interface, thus causing LLMs model security challenges. Although there is currently a large amount of research on prompt injection attacks, most of these black-box attacks use heuristic strategies. It is unclear how these heuristic strategies relate to the success rate of attacks and thus effectively improve model robustness. To solve this problem, we redefine the goal of the attack: to maximize the KL divergence between the conditional probabilities of the clean text and the adversarial text. Furthermore, we prove that maximizing the KL divergence is equivalent to maximizing the Mahalanobis distance between the embedded representation $x$ and $x'$ of the clean text and the adversarial text when the conditional probability is a Gaussian distribution and gives a quantitative relationship on $x$ and $x'$. Then we designed a simple and effective goal-guided generative prompt injection strategy (G2PIA) to find an injection text that satisfies specific constraints to achieve the optimal attack effect approximately. It is particularly noteworthy that our attack method is a query-free black-box attack method with low computational cost. Experimental results on seven LLM models and four datasets show the effectiveness of our attack method.

摘要: 现有的大型语言模型为大规模面向用户的自然语言任务提供了坚实的基础。大量用户可以很容易地通过用户界面注入敌意文本或指令，从而造成LLMS模型的安全挑战。虽然目前有大量关于即时注入攻击的研究，但这些黑盒攻击大多采用启发式策略。目前尚不清楚这些启发式策略如何与攻击成功率相关，从而有效地提高模型的稳健性。为了解决这个问题，我们重新定义了攻击的目标：最大化纯文本和敌意文本的条件概率之间的KL偏差。此外，我们证明了当条件概率为高斯分布时，最大化KL发散度等价于最大化明文和敌意文本的嵌入表示$x$和$x‘$之间的马氏距离，并给出了关于$x$和$x’$的定量关系。然后，设计了一种简单有效的目标引导生成性提示注入策略(G2PIA)，找到满足特定约束的注入文本，以近似达到最优的攻击效果。特别值得注意的是，我们的攻击方法是一种计算代价低的无查询黑盒攻击方法。在7个LLM模型和4个数据集上的实验结果表明了该攻击方法的有效性。



## **43. Logits of API-Protected LLMs Leak Proprietary Information**

受API保护的LLM日志泄露专有信息 cs.CL

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2403.09539v3) [paper-pdf](http://arxiv.org/pdf/2403.09539v3)

**Authors**: Matthew Finlayson, Xiang Ren, Swabha Swayamdipta

**Abstract**: Large language model (LLM) providers often hide the architectural details and parameters of their proprietary models by restricting public access to a limited API. In this work we show that, with only a conservative assumption about the model architecture, it is possible to learn a surprisingly large amount of non-public information about an API-protected LLM from a relatively small number of API queries (e.g., costing under $1000 USD for OpenAI's gpt-3.5-turbo). Our findings are centered on one key observation: most modern LLMs suffer from a softmax bottleneck, which restricts the model outputs to a linear subspace of the full output space. We exploit this fact to unlock several capabilities, including (but not limited to) obtaining cheap full-vocabulary outputs, auditing for specific types of model updates, identifying the source LLM given a single full LLM output, and even efficiently discovering the LLM's hidden size. Our empirical investigations show the effectiveness of our methods, which allow us to estimate the embedding size of OpenAI's gpt-3.5-turbo to be about 4096. Lastly, we discuss ways that LLM providers can guard against these attacks, as well as how these capabilities can be viewed as a feature (rather than a bug) by allowing for greater transparency and accountability.

摘要: 大型语言模型(LLM)提供商通常通过限制公众访问有限的API来隐藏其专有模型的体系结构细节和参数。在这项工作中，我们表明，在对模型体系结构只有一个保守的假设的情况下，从相对较少的API查询(例如，OpenAI的gpt-3.5-turbo的成本不到1000美元)，可以了解到关于受API保护的LLM的大量非公开信息。我们的发现集中在一个关键的观察上：大多数现代LLMS都存在Softmax瓶颈，这将模型输出限制在整个输出空间的线性子空间。我们利用这一事实来解锁几个功能，包括(但不限于)获取廉价的全词汇表输出、审计特定类型的模型更新、在给定单个完整的LLM输出的情况下识别源LLM，甚至高效地发现LLM的隐藏大小。我们的实证研究表明，我们的方法是有效的，允许我们估计OpenAI的gpt-3.5-turbo的嵌入大小约为4096。最后，我们讨论LLM提供商防范这些攻击的方法，以及如何通过允许更高的透明度和责任来将这些功能视为一项功能(而不是错误)。



## **44. IntellBot: Retrieval Augmented LLM Chatbot for Cyber Threat Knowledge Delivery**

IntellBot：用于网络威胁知识交付的检索增强LLM聊天机器人 cs.IR

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2411.05442v1) [paper-pdf](http://arxiv.org/pdf/2411.05442v1)

**Authors**: Dincy R. Arikkat, Abhinav M., Navya Binu, Parvathi M., Navya Biju, K. S. Arunima, Vinod P., Rafidha Rehiman K. A., Mauro Conti

**Abstract**: In the rapidly evolving landscape of cyber security, intelligent chatbots are gaining prominence. Artificial Intelligence, Machine Learning, and Natural Language Processing empower these chatbots to handle user inquiries and deliver threat intelligence. This helps cyber security knowledge readily available to both professionals and the public. Traditional rule-based chatbots often lack flexibility and struggle to adapt to user interactions. In contrast, Large Language Model-based chatbots offer contextually relevant information across multiple domains and adapt to evolving conversational contexts. In this work, we develop IntellBot, an advanced cyber security Chatbot built on top of cutting-edge technologies like Large Language Models and Langchain alongside a Retrieval-Augmented Generation model to deliver superior capabilities. This chatbot gathers information from diverse data sources to create a comprehensive knowledge base covering known vulnerabilities, recent cyber attacks, and emerging threats. It delivers tailored responses, serving as a primary hub for cyber security insights. By providing instant access to relevant information and resources, this IntellBot enhances threat intelligence, incident response, and overall security posture, saving time and empowering users with knowledge of cyber security best practices. Moreover, we analyzed the performance of our copilot using a two-stage evaluation strategy. We achieved BERT score above 0.8 by indirect approach and a cosine similarity score ranging from 0.8 to 1, which affirms the accuracy of our copilot. Additionally, we utilized RAGAS to evaluate the RAG model, and all evaluation metrics consistently produced scores above 0.77, highlighting the efficacy of our system.

摘要: 在快速发展的网络安全格局中，智能聊天机器人正变得越来越突出。人工智能、机器学习和自然语言处理使这些聊天机器人能够处理用户查询并提供威胁情报。这有助于专业人士和公众随时获得网络安全知识。传统的基于规则的聊天机器人往往缺乏灵活性，难以适应用户交互。相比之下，基于语言模型的大型聊天机器人提供跨多个领域的上下文相关信息，并适应不断变化的对话上下文。在这项工作中，我们开发了Intelligence Bot，这是一个先进的网络安全聊天机器人，建立在大型语言模型和语言链等尖端技术之上，并结合检索-增强生成模型来提供卓越的功能。这个聊天机器人从不同的数据源收集信息，创建一个全面的知识库，涵盖已知漏洞、最近的网络攻击和新出现的威胁。它提供量身定制的响应，成为网络安全洞察的主要枢纽。通过提供对相关信息和资源的即时访问，该IntelBot增强了威胁情报、事件响应和整体安全态势，节省了时间，并使用户能够了解网络安全最佳实践。此外，我们使用两阶段评估策略分析了我们的副驾驶的性能。我们通过间接方法获得了大于0.8的BERT得分，余弦相似度得分在0.8到1之间，这肯定了我们的副驾驶的准确性。此外，我们使用RAGAS对RAG模型进行评估，所有评估指标的得分都在0.77以上，突出了我们系统的有效性。



## **45. Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks**

保护语言模型免受越狱攻击的鲁棒即时优化 cs.LG

NeurIPS 2024 Spotlight; code available at  https://github.com/lapisrocks/rpo

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2401.17263v5) [paper-pdf](http://arxiv.org/pdf/2401.17263v5)

**Authors**: Andy Zhou, Bo Li, Haohan Wang

**Abstract**: Despite advances in AI alignment, large language models (LLMs) remain vulnerable to adversarial attacks or jailbreaking, in which adversaries can modify prompts to induce unwanted behavior. While some defenses have been proposed, they have not been adapted to newly proposed attacks and more challenging threat models. To address this, we propose an optimization-based objective for defending LLMs against jailbreaking attacks and an algorithm, Robust Prompt Optimization (RPO) to create robust system-level defenses. Our approach directly incorporates the adversary into the defensive objective and optimizes a lightweight and transferable suffix, enabling RPO to adapt to worst-case adaptive attacks. Our theoretical and experimental results show improved robustness to both jailbreaks seen during optimization and unknown jailbreaks, reducing the attack success rate (ASR) on GPT-4 to 6% and Llama-2 to 0% on JailbreakBench, setting the state-of-the-art. Code can be found at https://github.com/lapisrocks/rpo

摘要: 尽管在人工智能对齐方面取得了进展，但大型语言模型(LLM)仍然容易受到对手攻击或越狱的攻击，在这些攻击或越狱中，对手可以修改提示以诱导不想要的行为。虽然已经提出了一些防御措施，但它们还没有适应新提出的攻击和更具挑战性的威胁模型。为了解决这个问题，我们提出了一个基于优化的目标来保护LLMS免受越狱攻击，并提出了一个算法--稳健提示优化(RPO)来创建强大的系统级防御。我们的方法直接将对手合并到防御目标中，并优化了一个轻量级和可转移的后缀，使RPO能够适应最坏情况的自适应攻击。我们的理论和实验结果表明，对于优化期间看到的越狱和未知越狱，我们都提高了健壮性，将GPT-4上的攻击成功率(ASR)降低到6%，将Llama-2上的攻击成功率降低到0%，从而达到了最先进的水平。代码可在https://github.com/lapisrocks/rpo上找到



## **46. Accelerating Greedy Coordinate Gradient and General Prompt Optimization via Probe Sampling**

通过探针采样加速贪婪坐标梯度和一般提示优化 cs.CL

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2403.01251v3) [paper-pdf](http://arxiv.org/pdf/2403.01251v3)

**Authors**: Yiran Zhao, Wenyue Zheng, Tianle Cai, Xuan Long Do, Kenji Kawaguchi, Anirudh Goyal, Michael Shieh

**Abstract**: Safety of Large Language Models (LLMs) has become a critical issue given their rapid progresses. Greedy Coordinate Gradient (GCG) is shown to be effective in constructing adversarial prompts to break the aligned LLMs, but optimization of GCG is time-consuming. To reduce the time cost of GCG and enable more comprehensive studies of LLM safety, in this work, we study a new algorithm called $\texttt{Probe sampling}$. At the core of the algorithm is a mechanism that dynamically determines how similar a smaller draft model's predictions are to the target model's predictions for prompt candidates. When the target model is similar to the draft model, we rely heavily on the draft model to filter out a large number of potential prompt candidates. Probe sampling achieves up to $5.6$ times speedup using Llama2-7b-chat and leads to equal or improved attack success rate (ASR) on the AdvBench. Furthermore, probe sampling is also able to accelerate other prompt optimization techniques and adversarial methods, leading to acceleration of $1.8\times$ for AutoPrompt, $2.4\times$ for APE and $2.4\times$ for AutoDAN.

摘要: 随着大型语言模型的快速发展，其安全性已成为一个关键问题。贪婪坐标梯度(GCG)在构造敌意提示以打破排列的LLM方面是有效的，但GCG的优化是耗时的。为了减少GCG的时间开销，更全面地研究LLM的安全性，本文研究了一种新的算法--$\exttt{Probe Samples}$。该算法的核心是一种机制，该机制动态地确定较小的草稿模型的预测与目标模型对提示候选人的预测的相似性程度。当目标模型与选秀模型相似时，我们严重依赖选秀模型来筛选出大量潜在的提示候选者。使用Llama2-7b-Chat，探测采样可获得高达5.6美元的加速比，并可带来同等或更高的AdvBch攻击成功率(ASR)。此外，探针采样还能够加速其他即时优化技术和对抗方法，导致AutoPrompt、APE和AutoDAN的加速分别为1.8倍$、2.4倍$和2.4倍$。



## **47. Reasoning Robustness of LLMs to Adversarial Typographical Errors**

LLM对对抗性印刷错误的推理鲁棒性 cs.CL

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2411.05345v1) [paper-pdf](http://arxiv.org/pdf/2411.05345v1)

**Authors**: Esther Gan, Yiran Zhao, Liying Cheng, Yancan Mao, Anirudh Goyal, Kenji Kawaguchi, Min-Yen Kan, Michael Shieh

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities in reasoning using Chain-of-Thought (CoT) prompting. However, CoT can be biased by users' instruction. In this work, we study the reasoning robustness of LLMs to typographical errors, which can naturally occur in users' queries. We design an Adversarial Typo Attack ($\texttt{ATA}$) algorithm that iteratively samples typos for words that are important to the query and selects the edit that is most likely to succeed in attacking. It shows that LLMs are sensitive to minimal adversarial typographical changes. Notably, with 1 character edit, Mistral-7B-Instruct's accuracy drops from 43.7% to 38.6% on GSM8K, while with 8 character edits the performance further drops to 19.2%. To extend our evaluation to larger and closed-source LLMs, we develop the $\texttt{R$^2$ATA}$ benchmark, which assesses models' $\underline{R}$easoning $\underline{R}$obustness to $\underline{\texttt{ATA}}$. It includes adversarial typographical questions derived from three widely used reasoning datasets-GSM8K, BBH, and MMLU-by applying $\texttt{ATA}$ to open-source LLMs. $\texttt{R$^2$ATA}$ demonstrates remarkable transferability and causes notable performance drops across multiple super large and closed-source LLMs.

摘要: 大型语言模型(LLM)在使用思维链(CoT)提示进行推理方面表现出了令人印象深刻的能力。然而，COT可能会因用户的指示而产生偏差。在这项工作中，我们研究了LLMS对用户查询中自然发生的打字错误的推理健壮性。我们设计了一个对抗性的Typo攻击($\exttt{ATA}$)算法，该算法迭代地采样对查询重要的单词的打字错误，并选择最有可能成功攻击的编辑。这表明LLM对最小的对抗性排版变化很敏感。值得注意的是，在GSM8K上，1个字符编辑时，米斯特拉尔-7B指令的准确率从43.7%下降到38.6%，而8个字符编辑时，性能进一步下降到19.2%。为了将我们的评估扩展到更大的封闭源代码的LLM，我们开发了$\exttt{R$^2$ATA}$基准，它评估模型的$\下划线{R}$季节$\下划线{R}$热闹到$\下划线{Texttt{ATA}}$。它包括来自三个广泛使用的推理数据集-GSM8K、BBH和MMLU-的对抗性排版问题，方法是将$\exttt{ATA}$应用于开源LLM。$\exttt{R$^2$ATA}$表现出显著的可转移性，并在多个超大型和闭源LLM上导致显著的性能下降。



## **48. Fine-tuned Large Language Models (LLMs): Improved Prompt Injection Attacks Detection**

微调的大型语言模型（LLM）：改进的提示注入攻击检测 cs.CL

I am requesting the withdrawal of my paper due to critical issues  identified in the methodology/results that may impact its accuracy and  reliability. I also plan to make substantial revisions that go beyond minor  corrections

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2410.21337v2) [paper-pdf](http://arxiv.org/pdf/2410.21337v2)

**Authors**: Md Abdur Rahman, Fan Wu, Alfredo Cuzzocrea, Sheikh Iqbal Ahamed

**Abstract**: Large language models (LLMs) are becoming a popular tool as they have significantly advanced in their capability to tackle a wide range of language-based tasks. However, LLMs applications are highly vulnerable to prompt injection attacks, which poses a critical problem. These attacks target LLMs applications through using carefully designed input prompts to divert the model from adhering to original instruction, thereby it could execute unintended actions. These manipulations pose serious security threats which potentially results in data leaks, biased outputs, or harmful responses. This project explores the security vulnerabilities in relation to prompt injection attacks. To detect whether a prompt is vulnerable or not, we follows two approaches: 1) a pre-trained LLM, and 2) a fine-tuned LLM. Then, we conduct a thorough analysis and comparison of the classification performance. Firstly, we use pre-trained XLM-RoBERTa model to detect prompt injections using test dataset without any fine-tuning and evaluate it by zero-shot classification. Then, this proposed work will apply supervised fine-tuning to this pre-trained LLM using a task-specific labeled dataset from deepset in huggingface, and this fine-tuned model achieves impressive results with 99.13\% accuracy, 100\% precision, 98.33\% recall and 99.15\% F1-score thorough rigorous experimentation and evaluation. We observe that our approach is highly efficient in detecting prompt injection attacks.

摘要: 大型语言模型(LLM)正在成为一种流行的工具，因为它们在处理各种基于语言的任务的能力方面有了显著的进步。然而，LLMS应用程序很容易受到即时注入攻击，这是一个严重的问题。这些攻击通过使用精心设计的输入提示来转移模型对原始指令的依赖，从而针对LLMS应用程序，从而可以执行意外的操作。这些操作构成了严重的安全威胁，可能会导致数据泄露、有偏见的输出或有害的响应。该项目探索与提示注入攻击相关的安全漏洞。为了检测提示符是否易受攻击，我们采用了两种方法：1)预先训练的LLM和2)微调的LLM。然后，我们对分类性能进行了深入的分析和比较。首先，我们使用预先训练好的XLM-Roberta模型，在没有任何微调的测试数据集上检测快速注射，并用零镜头分类对其进行评估。然后，该工作将使用来自拥抱脸深度集的特定任务的标签数据集对该预训练的LLM进行有监督的微调，该微调模型取得了令人印象深刻的结果，其准确率为99.13，准确率为100，召回率为98.33，F1-Score为99.15。我们观察到我们的方法在检测即时注入攻击方面是非常有效的。



## **49. Fine-tuning Large Language Models for DGA and DNS Exfiltration Detection**

用于DGA和DNS溢出检测的微调大型语言模型 cs.CR

Accepted in Proceedings of the Workshop at AI for Cyber Threat  Intelligence (WAITI), 2024

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2410.21723v2) [paper-pdf](http://arxiv.org/pdf/2410.21723v2)

**Authors**: Md Abu Sayed, Asif Rahman, Christopher Kiekintveld, Sebastian Garcia

**Abstract**: Domain Generation Algorithms (DGAs) are malicious techniques used by malware to dynamically generate seemingly random domain names for communication with Command & Control (C&C) servers. Due to the fast and simple generation of DGA domains, detection methods must be highly efficient and precise to be effective. Large Language Models (LLMs) have demonstrated their proficiency in real-time detection tasks, making them ideal candidates for detecting DGAs. Our work validates the effectiveness of fine-tuned LLMs for detecting DGAs and DNS exfiltration attacks. We developed LLM models and conducted comprehensive evaluation using a diverse dataset comprising 59 distinct real-world DGA malware families and normal domain data. Our LLM model significantly outperformed traditional natural language processing techniques, especially in detecting unknown DGAs. We also evaluated its performance on DNS exfiltration datasets, demonstrating its effectiveness in enhancing cybersecurity measures. To the best of our knowledge, this is the first work that empirically applies LLMs for DGA and DNS exfiltration detection.

摘要: 域生成算法(DGA)是恶意软件用来动态生成看似随机的域名以与命令与控制(C&C)服务器通信的恶意技术。由于DGA结构域的快速而简单的生成，检测方法必须高效和精确才能有效。大型语言模型(LLM)已经证明了它们在实时检测任务中的熟练程度，使它们成为检测DGA的理想候选者。我们的工作验证了微调的LLMS在检测DGA和DNS渗出攻击方面的有效性。我们开发了LLM模型，并使用包含59个不同的真实DGA恶意软件家族和正常域数据的不同数据集进行了全面评估。我们的LLM模型显著优于传统的自然语言处理技术，特别是在检测未知DGA方面。我们还评估了它在DNS渗出数据集上的性能，展示了它在加强网络安全措施方面的有效性。据我们所知，这是第一个经验性地将LLMS应用于DGA和DNS渗出检测的工作。



## **50. FRACTURED-SORRY-Bench: Framework for Revealing Attacks in Conversational Turns Undermining Refusal Efficacy and Defenses over SORRY-Bench (Automated Multi-shot Jailbreaks)**

骨折-抱歉-长凳：揭露对话回合中攻击的框架，这些攻击削弱了SORRY长凳（自动多枪越狱）的拒绝功效和防御 cs.CL

4 pages, 2 tables

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2408.16163v2) [paper-pdf](http://arxiv.org/pdf/2408.16163v2)

**Authors**: Aman Priyanshu, Supriti Vijay

**Abstract**: This paper introduces FRACTURED-SORRY-Bench, a framework for evaluating the safety of Large Language Models (LLMs) against multi-turn conversational attacks. Building upon the SORRY-Bench dataset, we propose a simple yet effective method for generating adversarial prompts by breaking down harmful queries into seemingly innocuous sub-questions. Our approach achieves a maximum increase of +46.22\% in Attack Success Rates (ASRs) across GPT-4, GPT-4o, GPT-4o-mini, and GPT-3.5-Turbo models compared to baseline methods. We demonstrate that this technique poses a challenge to current LLM safety measures and highlights the need for more robust defenses against subtle, multi-turn attacks.

摘要: 本文介绍了FRACTURED-SORRY-Bench，这是一个用于评估大型语言模型（LLM）针对多轮对话攻击的安全性的框架。基于SORRY-Bench数据集，我们提出了一种简单而有效的方法，通过将有害的查询分解为看似无害的子问题来生成对抗性提示。与基线方法相比，我们的方法在GPT-4、GPT-4 o、GPT-4 o-mini和GPT-3.5-Turbo模型中实现了+46.22%的攻击成功率（SVR）最大增加。我们证明这种技术对当前的LLM安全措施构成了挑战，并强调了对微妙的多回合攻击进行更强大的防御的必要性。



