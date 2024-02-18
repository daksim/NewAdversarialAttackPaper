# Latest Large Language Model Attack Papers
**update at 2024-02-18 11:10:49**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. A Trembling House of Cards? Mapping Adversarial Attacks against Language Agents**

颤抖的纸牌屋？映射针对语言代理的对抗性攻击 cs.CL

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2402.10196v1) [paper-pdf](http://arxiv.org/pdf/2402.10196v1)

**Authors**: Lingbo Mo, Zeyi Liao, Boyuan Zheng, Yu Su, Chaowei Xiao, Huan Sun

**Abstract**: Language agents powered by large language models (LLMs) have seen exploding development. Their capability of using language as a vehicle for thought and communication lends an incredible level of flexibility and versatility. People have quickly capitalized on this capability to connect LLMs to a wide range of external components and environments: databases, tools, the Internet, robotic embodiment, etc. Many believe an unprecedentedly powerful automation technology is emerging. However, new automation technologies come with new safety risks, especially for intricate systems like language agents. There is a surprisingly large gap between the speed and scale of their development and deployment and our understanding of their safety risks. Are we building a house of cards? In this position paper, we present the first systematic effort in mapping adversarial attacks against language agents. We first present a unified conceptual framework for agents with three major components: Perception, Brain, and Action. Under this framework, we present a comprehensive discussion and propose 12 potential attack scenarios against different components of an agent, covering different attack strategies (e.g., input manipulation, adversarial demonstrations, jailbreaking, backdoors). We also draw connections to successful attack strategies previously applied to LLMs. We emphasize the urgency to gain a thorough understanding of language agent risks before their widespread deployment.

摘要: 由大型语言模型（LLM）驱动的语言代理已经看到了爆炸式的发展。他们使用语言作为思想和交流工具的能力赋予了他们难以置信的灵活性和多功能性。人们已经迅速利用这种能力将LLM连接到广泛的外部组件和环境：数据库，工具，互联网，机器人化身等。然而，新的自动化技术带来了新的安全风险，特别是对于像语言代理这样复杂的系统。它们的开发和部署的速度和规模与我们对其安全风险的理解之间存在着惊人的巨大差距。我们是在用纸牌搭房子吗？在这份立场文件中，我们提出了第一个系统性的努力，在映射对抗性攻击语言代理。首先，我们提出了一个统一的概念框架代理有三个主要组成部分：感知，大脑和行动。在这个框架下，我们提出了一个全面的讨论，并提出了12个潜在的攻击场景对不同的组件的代理，涵盖不同的攻击策略（例如，输入操作、对抗性演示、越狱、后门）。我们还绘制连接到成功的攻击策略以前应用于LLM。我们强调迫切需要在广泛部署之前彻底了解语言代理的风险。



## **2. Exploring the Adversarial Capabilities of Large Language Models**

探索大型语言模型的对抗能力 cs.AI

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2402.09132v2) [paper-pdf](http://arxiv.org/pdf/2402.09132v2)

**Authors**: Lukas Struppek, Minh Hieu Le, Dominik Hintersdorf, Kristian Kersting

**Abstract**: The proliferation of large language models (LLMs) has sparked widespread and general interest due to their strong language generation capabilities, offering great potential for both industry and research. While previous research delved into the security and privacy issues of LLMs, the extent to which these models can exhibit adversarial behavior remains largely unexplored. Addressing this gap, we investigate whether common publicly available LLMs have inherent capabilities to perturb text samples to fool safety measures, so-called adversarial examples resp.~attacks. More specifically, we investigate whether LLMs are inherently able to craft adversarial examples out of benign samples to fool existing safe rails. Our experiments, which focus on hate speech detection, reveal that LLMs succeed in finding adversarial perturbations, effectively undermining hate speech detection systems. Our findings carry significant implications for (semi-)autonomous systems relying on LLMs, highlighting potential challenges in their interaction with existing systems and safety measures.

摘要: 大型语言模型（LLM）的激增由于其强大的语言生成能力而引起了广泛和普遍的兴趣，为工业和研究提供了巨大的潜力。虽然以前的研究深入研究了LLM的安全和隐私问题，但这些模型可以表现出对抗行为的程度在很大程度上尚未探索。为了解决这一差距，我们调查了常见的公共LLM是否具有干扰文本样本以欺骗安全措施的固有能力，即所谓的对抗性示例。攻击更具体地说，我们研究LLM是否天生能够从良性样本中制作对抗性示例来欺骗现有的安全轨道。我们的实验主要集中在仇恨言论检测上，结果表明LLM成功地找到了对抗性扰动，有效地破坏了仇恨言论检测系统。我们的研究结果对依赖LLM的（半）自主系统具有重要意义，突出了它们与现有系统和安全措施相互作用的潜在挑战。



## **3. Rapid Adoption, Hidden Risks: The Dual Impact of Large Language Model Customization**

快速采用，隐藏风险：大型语言模型定制的双重影响 cs.CR

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2402.09179v2) [paper-pdf](http://arxiv.org/pdf/2402.09179v2)

**Authors**: Rui Zhang, Hongwei Li, Rui Wen, Wenbo Jiang, Yuan Zhang, Michael Backes, Yun Shen, Yang Zhang

**Abstract**: The increasing demand for customized Large Language Models (LLMs) has led to the development of solutions like GPTs. These solutions facilitate tailored LLM creation via natural language prompts without coding. However, the trustworthiness of third-party custom versions of LLMs remains an essential concern. In this paper, we propose the first instruction backdoor attacks against applications integrated with untrusted customized LLMs (e.g., GPTs). Specifically, these attacks embed the backdoor into the custom version of LLMs by designing prompts with backdoor instructions, outputting the attacker's desired result when inputs contain the pre-defined triggers. Our attack includes 3 levels of attacks: word-level, syntax-level, and semantic-level, which adopt different types of triggers with progressive stealthiness. We stress that our attacks do not require fine-tuning or any modification to the backend LLMs, adhering strictly to GPTs development guidelines. We conduct extensive experiments on 4 prominent LLMs and 5 benchmark text classification datasets. The results show that our instruction backdoor attacks achieve the desired attack performance without compromising utility. Additionally, we propose an instruction-ignoring defense mechanism and demonstrate its partial effectiveness in mitigating such attacks. Our findings highlight the vulnerability and the potential risks of LLM customization such as GPTs.

摘要: 对定制的大型语言模型(LLM)的需求日益增长，导致了GPTS等解决方案的开发。这些解决方案无需编码即可通过自然语言提示实现定制的LLM创建。然而，第三方定制版本的LLMS的可信性仍然是一个关键问题。在本文中，我们提出了针对集成了不可信任的定制LLM的应用程序(例如GPT)的第一指令后门攻击。具体地说，这些攻击通过设计带有后门指令的提示将后门嵌入到LLMS的自定义版本中，并在输入包含预定义触发器时输出攻击者所需的结果。我们的攻击包括词级、句法级和语义级三个级别的攻击，它们采用了不同类型的触发器，具有渐进的隐蔽性。我们强调，我们的攻击不需要对后端LLM进行微调或任何修改，严格遵守GPTS开发指南。我们在4个重要的LLMS和5个基准文本分类数据集上进行了大量的实验。结果表明，指令后门攻击在不影响效用的情况下达到了预期的攻击性能。此外，我们还提出了一种忽略指令的防御机制，并证明了其在缓解此类攻击方面的部分有效性。我们的发现突出了LLM定制(如GPTS)的脆弱性和潜在风险。



## **4. AbuseGPT: Abuse of Generative AI ChatBots to Create Smishing Campaigns**

AbuseGPT：滥用生成式AI聊天机器人来创建Smishing活动 cs.CR

6 pages, 12 figures, published in ISDFS 2024

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2402.09728v1) [paper-pdf](http://arxiv.org/pdf/2402.09728v1)

**Authors**: Ashfak Md Shibli, Mir Mehedi A. Pritom, Maanak Gupta

**Abstract**: SMS phishing, also known as "smishing", is a growing threat that tricks users into disclosing private information or clicking into URLs with malicious content through fraudulent mobile text messages. In recent past, we have also observed a rapid advancement of conversational generative AI chatbot services (e.g., OpenAI's ChatGPT, Google's BARD), which are powered by pre-trained large language models (LLMs). These AI chatbots certainly have a lot of utilities but it is not systematically understood how they can play a role in creating threats and attacks. In this paper, we propose AbuseGPT method to show how the existing generative AI-based chatbot services can be exploited by attackers in real world to create smishing texts and eventually lead to craftier smishing campaigns. To the best of our knowledge, there is no pre-existing work that evidently shows the impacts of these generative text-based models on creating SMS phishing. Thus, we believe this study is the first of its kind to shed light on this emerging cybersecurity threat. We have found strong empirical evidences to show that attackers can exploit ethical standards in the existing generative AI-based chatbot services by crafting prompt injection attacks to create newer smishing campaigns. We also discuss some future research directions and guidelines to protect the abuse of generative AI-based services and safeguard users from smishing attacks.

摘要: 短信钓鱼是一种日益增长的威胁，它诱使用户通过欺诈性手机短信泄露私人信息或点击含有恶意内容的URL。在最近的过去，我们还观察到会话生成式AI聊天机器人服务(例如，OpenAI的ChatGPT、Google的Bard)的快速发展，这些服务由预先训练的大型语言模型(LLM)提供支持。这些人工智能聊天机器人当然有很多实用程序，但人们并不系统地了解它们如何在制造威胁和攻击方面发挥作用。在本文中，我们提出了AbuseGPT方法来展示现有的基于AI的生成性聊天机器人服务如何被现实世界中的攻击者利用来创建恶意文本，并最终导致更巧妙的恶意攻击活动。就我们所知，没有任何预先存在的工作可以明显地表明这些基于文本的生成性模型对创建短信钓鱼的影响。因此，我们认为这项研究是第一次揭示这一新出现的网络安全威胁。我们发现了强有力的经验证据表明，攻击者可以利用现有基于人工智能的生成性聊天机器人服务中的道德标准，通过手工制作快速注入攻击来创建更新的气味攻击。我们还讨论了一些未来的研究方向和指导方针，以保护基于人工智能的生成性服务的滥用，保护用户免受嗅觉攻击。



## **5. Detecting Phishing Sites Using ChatGPT**

使用ChatGPT检测钓鱼网站 cs.CR

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2306.05816v2) [paper-pdf](http://arxiv.org/pdf/2306.05816v2)

**Authors**: Takashi Koide, Naoki Fukushi, Hiroki Nakano, Daiki Chiba

**Abstract**: The emergence of Large Language Models (LLMs), including ChatGPT, is having a significant impact on a wide range of fields. While LLMs have been extensively researched for tasks such as code generation and text synthesis, their application in detecting malicious web content, particularly phishing sites, has been largely unexplored. To combat the rising tide of cyber attacks due to the misuse of LLMs, it is important to automate detection by leveraging the advanced capabilities of LLMs.   In this paper, we propose a novel system called ChatPhishDetector that utilizes LLMs to detect phishing sites. Our system involves leveraging a web crawler to gather information from websites, generating prompts for LLMs based on the crawled data, and then retrieving the detection results from the responses generated by the LLMs. The system enables us to detect multilingual phishing sites with high accuracy by identifying impersonated brands and social engineering techniques in the context of the entire website, without the need to train machine learning models. To evaluate the performance of our system, we conducted experiments on our own dataset and compared it with baseline systems and several LLMs. The experimental results using GPT-4V demonstrated outstanding performance, with a precision of 98.7% and a recall of 99.6%, outperforming the detection results of other LLMs and existing systems. These findings highlight the potential of LLMs for protecting users from online fraudulent activities and have important implications for enhancing cybersecurity measures.

摘要: 大型语言模型(LLM)的出现，包括ChatGPT，正在对广泛的领域产生重大影响。虽然LLMS已经被广泛研究用于代码生成和文本合成等任务，但它们在检测恶意网络内容，特别是钓鱼网站方面的应用在很大程度上还没有被探索过。为了应对由于滥用LLMS而不断增加的网络攻击浪潮，重要的是通过利用LLMS的高级功能来实现自动检测。在本文中，我们提出了一个新的系统，称为ChatPhishDetector，它利用LLMS来检测钓鱼网站。我们的系统利用网络爬虫从网站收集信息，基于爬行的数据生成LLMS的提示，然后从LLMS生成的响应中检索检测结果。该系统通过在整个网站的上下文中识别假冒品牌和社会工程技术，使我们能够高精度地检测多语言钓鱼网站，而不需要训练机器学习模型。为了评估我们的系统的性能，我们在自己的数据集上进行了实验，并将其与基线系统和几个LLMS进行了比较。基于GPT-4V的实验结果表明，该方法具有较好的性能，准确率为98.7%，召回率为99.6%，优于其他LLMS和现有系统的检测结果。这些发现突出了小岛屿发展中国家保护用户免遭网上欺诈活动的潜力，并对加强网络安全措施具有重要意义。



## **6. PAL: Proxy-Guided Black-Box Attack on Large Language Models**

PAL：针对大型语言模型的代理引导的黑盒攻击 cs.CL

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2402.09674v1) [paper-pdf](http://arxiv.org/pdf/2402.09674v1)

**Authors**: Chawin Sitawarin, Norman Mu, David Wagner, Alexandre Araujo

**Abstract**: Large Language Models (LLMs) have surged in popularity in recent months, but they have demonstrated concerning capabilities to generate harmful content when manipulated. While techniques like safety fine-tuning aim to minimize harmful use, recent works have shown that LLMs remain vulnerable to attacks that elicit toxic responses. In this work, we introduce the Proxy-Guided Attack on LLMs (PAL), the first optimization-based attack on LLMs in a black-box query-only setting. In particular, it relies on a surrogate model to guide the optimization and a sophisticated loss designed for real-world LLM APIs. Our attack achieves 84% attack success rate (ASR) on GPT-3.5-Turbo and 48% on Llama-2-7B, compared to 4% for the current state of the art. We also propose GCG++, an improvement to the GCG attack that reaches 94% ASR on white-box Llama-2-7B, and the Random-Search Attack on LLMs (RAL), a strong but simple baseline for query-based attacks. We believe the techniques proposed in this work will enable more comprehensive safety testing of LLMs and, in the long term, the development of better security guardrails. The code can be found at https://github.com/chawins/pal.

摘要: 近几个月来，大型语言模型(LLM)越来越受欢迎，但它们展示了人们对操纵时生成有害内容的能力的担忧。虽然安全微调等技术旨在将有害使用降至最低，但最近的研究表明，LLM仍然容易受到引发有毒反应的攻击。在这项工作中，我们引入了代理引导的LLMS攻击(PAL)，这是在黑盒仅查询环境下第一个基于优化的LLMS攻击。特别是，它依赖于代理模型来指导优化和为真实世界的LLMAPI设计的复杂损失。我们的攻击在GPT-3.5-Turbo上实现了84%的攻击成功率(ASR)，在Llama-2-7B上实现了48%的攻击成功率，而目前的技术水平为4%。我们还提出了GCG++，它是对白盒Llama-2-7B上达到94%ASR的GCG攻击的改进，以及对LLMS的随机搜索攻击(Ral)，它是一种强大但简单的基于查询攻击的基线。我们相信，这项工作中提出的技术将使LLMS能够进行更全面的安全测试，并在长期内开发出更好的安全护栏。代码可在https://github.com/chawins/pal.上找到



## **7. How Secure Are Large Language Models (LLMs) for Navigation in Urban Environments?**

大型语言模型(LLM)在城市环境中的导航安全性如何？ cs.RO

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.09546v1) [paper-pdf](http://arxiv.org/pdf/2402.09546v1)

**Authors**: Congcong Wen, Jiazhao Liang, Shuaihang Yuan, Hao Huang, Yi Fang

**Abstract**: In the field of robotics and automation, navigation systems based on Large Language Models (LLMs) have recently shown impressive performance. However, the security aspects of these systems have received relatively less attention. This paper pioneers the exploration of vulnerabilities in LLM-based navigation models in urban outdoor environments, a critical area given the technology's widespread application in autonomous driving, logistics, and emergency services. Specifically, we introduce a novel Navigational Prompt Suffix (NPS) Attack that manipulates LLM-based navigation models by appending gradient-derived suffixes to the original navigational prompt, leading to incorrect actions. We conducted comprehensive experiments on an LLMs-based navigation model that employs various LLMs for reasoning. Our results, derived from the Touchdown and Map2Seq street-view datasets under both few-shot learning and fine-tuning configurations, demonstrate notable performance declines across three metrics in the face of both white-box and black-box attacks. These results highlight the generalizability and transferability of the NPS Attack, emphasizing the need for enhanced security in LLM-based navigation systems. As an initial countermeasure, we propose the Navigational Prompt Engineering (NPE) Defense strategy, concentrating on navigation-relevant keywords to reduce the impact of adversarial suffixes. While initial findings indicate that this strategy enhances navigational safety, there remains a critical need for the wider research community to develop stronger defense methods to effectively tackle the real-world challenges faced by these systems.

摘要: 在机器人和自动化领域，基于大型语言模型（LLM）的导航系统最近表现出令人印象深刻的性能。然而，这些系统的安全方面受到的关注相对较少。本文率先探索了基于LLM的导航模型在城市户外环境中的漏洞，这是一个关键领域，因为该技术在自动驾驶、物流和应急服务中得到了广泛应用。具体来说，我们引入了一种新的导航提示后缀（NATURAL）攻击，该攻击通过将梯度派生的后缀附加到原始导航提示中来操纵基于LLM的导航模型，从而导致错误的操作。我们进行了全面的实验，基于LLMs的导航模型，采用各种LLMs的推理。我们的结果来自于Touchdown和Map2Seq街景数据集，在几次学习和微调配置下，在面对白盒和黑盒攻击时，三个指标的性能都有明显下降。这些结果突出了恶意攻击的可推广性和可转移性，强调了基于LLM的导航系统需要增强安全性。作为初步对策，我们提出了导航提示工程（NPE）防御策略，集中在导航相关的关键字，以减少对抗性后缀的影响。虽然初步研究结果表明，这一战略提高了航行安全，但仍迫切需要更广泛的研究界开发更强大的防御方法，以有效应对这些系统面临的现实挑战。



## **8. Attacks, Defenses and Evaluations for LLM Conversation Safety: A Survey**

LLM通话安全的攻击、防御和评估：综述 cs.CL

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.09283v1) [paper-pdf](http://arxiv.org/pdf/2402.09283v1)

**Authors**: Zhichen Dong, Zhanhui Zhou, Chao Yang, Jing Shao, Yu Qiao

**Abstract**: Large Language Models (LLMs) are now commonplace in conversation applications. However, their risks of misuse for generating harmful responses have raised serious societal concerns and spurred recent research on LLM conversation safety. Therefore, in this survey, we provide a comprehensive overview of recent studies, covering three critical aspects of LLM conversation safety: attacks, defenses, and evaluations. Our goal is to provide a structured summary that enhances understanding of LLM conversation safety and encourages further investigation into this important subject. For easy reference, we have categorized all the studies mentioned in this survey according to our taxonomy, available at: https://github.com/niconi19/LLM-conversation-safety.

摘要: 大型语言模型(LLM)现在在对话应用程序中很常见。然而，它们被滥用来产生有害反应的风险已经引起了严重的社会关注，并促使最近对LLM对话安全的研究。因此，在这次调查中，我们全面概述了最近的研究，涵盖了LLM对话安全的三个关键方面：攻击、防御和评估。我们的目标是提供一个结构化的摘要，以加强对LLM对话安全的理解，并鼓励对这一重要主题的进一步研究。为方便参考，我们已根据我们的分类对本次调查中提到的所有研究进行了分类，可在以下网址获得：https://github.com/niconi19/LLM-conversation-safety.



## **9. Leveraging the Context through Multi-Round Interactions for Jailbreaking Attacks**

通过多轮互动利用上下文进行越狱攻击 cs.LG

29 pages

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.09177v1) [paper-pdf](http://arxiv.org/pdf/2402.09177v1)

**Authors**: Yixin Cheng, Markos Georgopoulos, Volkan Cevher, Grigorios G. Chrysos

**Abstract**: Large Language Models (LLMs) are susceptible to Jailbreaking attacks, which aim to extract harmful information by subtly modifying the attack query. As defense mechanisms evolve, directly obtaining harmful information becomes increasingly challenging for Jailbreaking attacks. In this work, inspired by human practices of indirect context to elicit harmful information, we focus on a new attack form called Contextual Interaction Attack. The idea relies on the autoregressive nature of the generation process in LLMs. We contend that the prior context--the information preceding the attack query--plays a pivotal role in enabling potent Jailbreaking attacks. Specifically, we propose an approach that leverages preliminary question-answer pairs to interact with the LLM. By doing so, we guide the responses of the model toward revealing the 'desired' harmful information. We conduct experiments on four different LLMs and demonstrate the efficacy of this attack, which is black-box and can also transfer across LLMs. We believe this can lead to further developments and understanding of the context vector in LLMs.

摘要: 大型语言模型(LLM)容易受到越狱攻击，其目的是通过微妙地修改攻击查询来提取有害信息。随着防御机制的发展，直接获取有害信息对越狱攻击来说变得越来越具有挑战性。在这项工作中，受人类间接上下文获取有害信息的做法的启发，我们重点研究了一种新的攻击形式，称为上下文交互攻击。这一想法依赖于LLMS中生成过程的自回归性质。我们认为，先前的上下文--攻击查询之前的信息--在实现强大的越狱攻击方面发挥着关键作用。具体地说，我们提出了一种利用初步问题-答案对与LLM交互的方法。通过这样做，我们引导该模型的反应，以揭示“想要的”有害信息。我们在四个不同的LLM上进行了实验，并展示了该攻击的有效性，该攻击是黑盒的，也可以跨LLM传输。我们相信这可以导致对LLMS中的上下文向量的进一步发展和理解。



## **10. Attacking Large Language Models with Projected Gradient Descent**

用投影梯度下降攻击大型语言模型 cs.LG

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.09154v1) [paper-pdf](http://arxiv.org/pdf/2402.09154v1)

**Authors**: Simon Geisler, Tom Wollschläger, M. H. I. Abdalla, Johannes Gasteiger, Stephan Günnemann

**Abstract**: Current LLM alignment methods are readily broken through specifically crafted adversarial prompts. While crafting adversarial prompts using discrete optimization is highly effective, such attacks typically use more than 100,000 LLM calls. This high computational cost makes them unsuitable for, e.g., quantitative analyses and adversarial training. To remedy this, we revisit Projected Gradient Descent (PGD) on the continuously relaxed input prompt. Although previous attempts with ordinary gradient-based attacks largely failed, we show that carefully controlling the error introduced by the continuous relaxation tremendously boosts their efficacy. Our PGD for LLMs is up to one order of magnitude faster than state-of-the-art discrete optimization to achieve the same devastating attack results.

摘要: 目前的LLM对齐方法很容易突破专门制作的对抗性提示。虽然使用离散优化精心编制敌意提示是非常有效的，但此类攻击通常使用超过100,000个LLM调用。这种高的计算成本使得它们不适合例如定量分析和对抗性训练。为了纠正这个问题，我们重新审视了持续放松的输入提示上的预测梯度下降(PGD)。虽然以前使用普通的基于梯度的攻击的尝试基本上都失败了，但我们表明，仔细控制连续放松带来的错误可以极大地提高它们的效率。我们针对LLMS的PGD比最先进的离散优化快一个数量级，以实现相同的毁灭性攻击结果。



## **11. Soft Prompt Threats: Attacking Safety Alignment and Unlearning in Open-Source LLMs through the Embedding Space**

软提示威胁：通过嵌入空间攻击开源LLMS中的安全对齐和遗忘 cs.LG

Trigger Warning: the appendix contains LLM-generated text with  violence and harassment

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.09063v1) [paper-pdf](http://arxiv.org/pdf/2402.09063v1)

**Authors**: Leo Schwinn, David Dobre, Sophie Xhonneux, Gauthier Gidel, Stephan Gunnemann

**Abstract**: Current research in adversarial robustness of LLMs focuses on discrete input manipulations in the natural language space, which can be directly transferred to closed-source models. However, this approach neglects the steady progression of open-source models. As open-source models advance in capability, ensuring their safety also becomes increasingly imperative. Yet, attacks tailored to open-source LLMs that exploit full model access remain largely unexplored. We address this research gap and propose the embedding space attack, which directly attacks the continuous embedding representation of input tokens. We find that embedding space attacks circumvent model alignments and trigger harmful behaviors more efficiently than discrete attacks or model fine-tuning. Furthermore, we present a novel threat model in the context of unlearning and show that embedding space attacks can extract supposedly deleted information from unlearned LLMs across multiple datasets and models. Our findings highlight embedding space attacks as an important threat model in open-source LLMs. Trigger Warning: the appendix contains LLM-generated text with violence and harassment.

摘要: 目前关于LLMS对抗健壮性的研究主要集中在自然语言空间中的离散输入操作，这些操作可以直接转化为闭源模型。然而，这种方法忽略了开源模型的稳步发展。随着开源模型在功能上的进步，确保它们的安全性也变得越来越迫切。然而，针对利用全模型访问的开源LLM量身定做的攻击在很大程度上仍未被探索。针对这一研究空白，我们提出了嵌入空间攻击，它直接攻击输入令牌的连续嵌入表示。我们发现，嵌入空间攻击比离散攻击或模型微调更有效地绕过模型对齐并触发有害行为。此外，我们提出了一种新的遗忘背景下的威胁模型，并证明了嵌入空间攻击可以从多个数据集和模型中从未学习的LLM中提取假定删除的信息。我们的发现强调了将空间攻击作为一个重要的威胁模型嵌入到开源LLMS中。触发警告：附录包含LLM生成的带有暴力和骚扰的文本。



## **12. Prompted Contextual Vectors for Spear-Phishing Detection**

鱼叉式网络钓鱼检测的提示上下文向量 cs.LG

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.08309v2) [paper-pdf](http://arxiv.org/pdf/2402.08309v2)

**Authors**: Daniel Nahmias, Gal Engelberg, Dan Klein, Asaf Shabtai

**Abstract**: Spear-phishing attacks present a significant security challenge, with large language models (LLMs) escalating the threat by generating convincing emails and facilitating target reconnaissance. To address this, we propose a detection approach based on a novel document vectorization method that utilizes an ensemble of LLMs to create representation vectors. By prompting LLMs to reason and respond to human-crafted questions, we quantify the presence of common persuasion principles in the email's content, producing prompted contextual document vectors for a downstream supervised machine learning model. We evaluate our method using a unique dataset generated by a proprietary system that automates target reconnaissance and spear-phishing email creation. Our method achieves a 91% F1 score in identifying LLM-generated spear-phishing emails, with the training set comprising only traditional phishing and benign emails. Key contributions include an innovative document vectorization method utilizing LLM reasoning, a publicly available dataset of high-quality spear-phishing emails, and the demonstrated effectiveness of our method in detecting such emails. This methodology can be utilized for various document classification tasks, particularly in adversarial problem domains.

摘要: 鱼叉式网络钓鱼攻击是一个重大的安全挑战，大型语言模型(LLM)通过生成令人信服的电子邮件和促进目标侦察来升级威胁。针对这一问题，我们提出了一种基于一种新的文档矢量化方法的检测方法，该方法利用一组LLM来创建表示向量。通过促使LLM对人类提出的问题进行推理和回应，我们量化了电子邮件内容中常见说服原则的存在，为下游有监督的机器学习模型生成了提示的上下文文档向量。我们使用由专有系统生成的唯一数据集来评估我们的方法，该系统自动执行目标侦察和鱼叉式网络钓鱼电子邮件创建。我们的方法在识别LLM生成的鱼叉式钓鱼邮件方面取得了91%的F1分数，训练集仅包括传统钓鱼邮件和良性电子邮件。主要贡献包括一种利用LLM推理的创新文档矢量化方法，一个公开可用的高质量鱼叉式钓鱼电子邮件数据集，以及我们的方法在检测此类电子邮件方面的有效性。这种方法可用于各种文档分类任务，特别是在对抗性问题领域。



## **13. SafeDecoding: Defending against Jailbreak Attacks via Safety-Aware Decoding**

SafeDecoding：通过安全感知解码防御越狱攻击 cs.CR

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.08983v1) [paper-pdf](http://arxiv.org/pdf/2402.08983v1)

**Authors**: Zhangchen Xu, Fengqing Jiang, Luyao Niu, Jinyuan Jia, Bill Yuchen Lin, Radha Poovendran

**Abstract**: As large language models (LLMs) become increasingly integrated into real-world applications such as code generation and chatbot assistance, extensive efforts have been made to align LLM behavior with human values, including safety. Jailbreak attacks, aiming to provoke unintended and unsafe behaviors from LLMs, remain a significant/leading LLM safety threat. In this paper, we aim to defend LLMs against jailbreak attacks by introducing SafeDecoding, a safety-aware decoding strategy for LLMs to generate helpful and harmless responses to user queries. Our insight in developing SafeDecoding is based on the observation that, even though probabilities of tokens representing harmful contents outweigh those representing harmless responses, safety disclaimers still appear among the top tokens after sorting tokens by probability in descending order. This allows us to mitigate jailbreak attacks by identifying safety disclaimers and amplifying their token probabilities, while simultaneously attenuating the probabilities of token sequences that are aligned with the objectives of jailbreak attacks. We perform extensive experiments on five LLMs using six state-of-the-art jailbreak attacks and four benchmark datasets. Our results show that SafeDecoding significantly reduces the attack success rate and harmfulness of jailbreak attacks without compromising the helpfulness of responses to benign user queries. SafeDecoding outperforms six defense methods.

摘要: 随着大型语言模型（LLM）越来越多地集成到现实世界的应用程序中，如代码生成和聊天机器人辅助，人们已经做出了广泛的努力，使LLM行为与人类价值观保持一致，包括安全性。越狱攻击，旨在挑起LLM的意外和不安全的行为，仍然是一个重大/领先的LLM安全威胁。在本文中，我们的目标是通过引入SafeDecoding，一个安全意识的解码策略，LLM生成有用的和无害的响应用户查询，以抵御越狱攻击的LLM。我们在开发SafeDecoding时的见解是基于这样的观察：即使表示有害内容的令牌的概率大于表示无害响应的令牌的概率，安全声明仍然出现在按概率降序排列令牌之后的顶部令牌中。这使我们能够通过识别安全声明并放大其令牌概率来减轻越狱攻击，同时衰减与越狱攻击目标一致的令牌序列的概率。我们使用六种最先进的越狱攻击和四个基准数据集对五个LLM进行了广泛的实验。我们的研究结果表明，SafeDecoding显着降低了攻击的成功率和越狱攻击的危害性，而不影响良性用户查询的响应的有用性。SafeDecoding优于六种防御方法。



## **14. COLD-Attack: Jailbreaking LLMs with Stealthiness and Controllability**

冷攻击：具有隐蔽性和可控性的越狱LLMS cs.LG

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2402.08679v1) [paper-pdf](http://arxiv.org/pdf/2402.08679v1)

**Authors**: Xingang Guo, Fangxu Yu, Huan Zhang, Lianhui Qin, Bin Hu

**Abstract**: Jailbreaks on Large language models (LLMs) have recently received increasing attention. For a comprehensive assessment of LLM safety, it is essential to consider jailbreaks with diverse attributes, such as contextual coherence and sentiment/stylistic variations, and hence it is beneficial to study controllable jailbreaking, i.e. how to enforce control on LLM attacks. In this paper, we formally formulate the controllable attack generation problem, and build a novel connection between this problem and controllable text generation, a well-explored topic of natural language processing. Based on this connection, we adapt the Energy-based Constrained Decoding with Langevin Dynamics (COLD), a state-of-the-art, highly efficient algorithm in controllable text generation, and introduce the COLD-Attack framework which unifies and automates the search of adversarial LLM attacks under a variety of control requirements such as fluency, stealthiness, sentiment, and left-right-coherence. The controllability enabled by COLD-Attack leads to diverse new jailbreak scenarios which not only cover the standard setting of generating fluent suffix attacks, but also allow us to address new controllable attack settings such as revising a user query adversarially with minimal paraphrasing, and inserting stealthy attacks in context with left-right-coherence. Our extensive experiments on various LLMs (Llama-2, Mistral, Vicuna, Guanaco, GPT-3.5) show COLD-Attack's broad applicability, strong controllability, high success rate, and attack transferability. Our code is available at https://github.com/Yu-Fangxu/COLD-Attack.

摘要: 大型语言模型(LLM)的越狱最近受到越来越多的关注。为了全面评估LLM的安全性，必须考虑具有不同属性的越狱，例如上下文连贯性和情绪/风格变化，因此研究可控越狱是有益的，即如何加强对LLM攻击的控制。在本文中，我们形式化地描述了可控攻击生成问题，并将该问题与自然语言处理的一个热门话题--可控文本生成建立了一种新的联系。基于此，我们采用了基于能量的朗之万动力学约束解码算法(COLD)，这是一种最新的、高效的可控文本生成算法，并引入了冷攻击框架，该框架可以在流畅性、隐蔽性、情感和左右一致性等各种控制要求下统一和自动化搜索敌意LLM攻击。冷攻击带来的可控性导致了不同的新越狱场景，这些场景不仅覆盖了生成流畅后缀攻击的标准设置，而且允许我们解决新的可控攻击设置，例如以最小的转述以相反的方式修改用户查询，以及以左右一致的方式在上下文中插入隐蔽攻击。我们在不同的LLMS(骆驼-2、米斯特拉尔、维库纳、瓜纳科、GPT-3.5)上的广泛实验表明，冷攻击具有广泛的适用性、较强的可控性、高成功率和攻击可转移性。我们的代码可以在https://github.com/Yu-Fangxu/COLD-Attack.上找到



## **15. Test-Time Backdoor Attacks on Multimodal Large Language Models**

对多通道大型语言模型的测试时间后门攻击 cs.CL

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2402.08577v1) [paper-pdf](http://arxiv.org/pdf/2402.08577v1)

**Authors**: Dong Lu, Tianyu Pang, Chao Du, Qian Liu, Xianjun Yang, Min Lin

**Abstract**: Backdoor attacks are commonly executed by contaminating training data, such that a trigger can activate predetermined harmful effects during the test phase. In this work, we present AnyDoor, a test-time backdoor attack against multimodal large language models (MLLMs), which involves injecting the backdoor into the textual modality using adversarial test images (sharing the same universal perturbation), without requiring access to or modification of the training data. AnyDoor employs similar techniques used in universal adversarial attacks, but distinguishes itself by its ability to decouple the timing of setup and activation of harmful effects. In our experiments, we validate the effectiveness of AnyDoor against popular MLLMs such as LLaVA-1.5, MiniGPT-4, InstructBLIP, and BLIP-2, as well as provide comprehensive ablation studies. Notably, because the backdoor is injected by a universal perturbation, AnyDoor can dynamically change its backdoor trigger prompts/harmful effects, exposing a new challenge for defending against backdoor attacks. Our project page is available at https://sail-sg.github.io/AnyDoor/.

摘要: 后门攻击通常通过污染训练数据来执行，使得触发器可以在测试阶段激活预定的有害影响。在这项工作中，我们提出了AnyDoor，一种针对多模式大型语言模型(MLLMS)的测试时间后门攻击，它使用敌对的测试图像(共享相同的通用扰动)将后门注入到文本通道中，而不需要访问或修改训练数据。AnyDoor使用了通用对抗性攻击中使用的类似技术，但其与众不同之处在于它能够将设置和激活有害效果的时间脱钩。在我们的实验中，我们验证了AnyDoor对LLaVA-1.5、MiniGPT-4、InstructBLIP和BLIP-2等常用MLLMS的有效性，并提供了全面的消融研究。值得注意的是，由于后门是由通用扰动注入的，AnyDoor可以动态更改其后门触发提示/有害影响，从而暴露出防御后门攻击的新挑战。我们的项目页面可在https://sail-sg.github.io/AnyDoor/.上查看



## **16. Pandora: Jailbreak GPTs by Retrieval Augmented Generation Poisoning**

Pandora：通过检索增强生成中毒越狱GPT cs.CR

6 pages

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2402.08416v1) [paper-pdf](http://arxiv.org/pdf/2402.08416v1)

**Authors**: Gelei Deng, Yi Liu, Kailong Wang, Yuekang Li, Tianwei Zhang, Yang Liu

**Abstract**: Large Language Models~(LLMs) have gained immense popularity and are being increasingly applied in various domains. Consequently, ensuring the security of these models is of paramount importance. Jailbreak attacks, which manipulate LLMs to generate malicious content, are recognized as a significant vulnerability. While existing research has predominantly focused on direct jailbreak attacks on LLMs, there has been limited exploration of indirect methods. The integration of various plugins into LLMs, notably Retrieval Augmented Generation~(RAG), which enables LLMs to incorporate external knowledge bases into their response generation such as GPTs, introduces new avenues for indirect jailbreak attacks.   To fill this gap, we investigate indirect jailbreak attacks on LLMs, particularly GPTs, introducing a novel attack vector named Retrieval Augmented Generation Poisoning. This method, Pandora, exploits the synergy between LLMs and RAG through prompt manipulation to generate unexpected responses. Pandora uses maliciously crafted content to influence the RAG process, effectively initiating jailbreak attacks. Our preliminary tests show that Pandora successfully conducts jailbreak attacks in four different scenarios, achieving higher success rates than direct attacks, with 64.3\% for GPT-3.5 and 34.8\% for GPT-4.

摘要: 大语言模型(LLMS)已经得到了广泛的应用，并在各个领域得到了越来越多的应用。因此，确保这些模型的安全至关重要。越狱攻击操纵LLM生成恶意内容，被认为是一个严重的漏洞。虽然现有的研究主要集中在对LLM的直接越狱攻击上，但对间接方法的探索有限。将各种插件集成到LLMS中，特别是检索增强一代~(RAG)，使LLMS能够将外部知识库整合到其响应生成中，如GPT，为间接越狱攻击引入了新的途径。为了填补这一空白，我们研究了针对LLM的间接越狱攻击，特别是GPT，引入了一种新的攻击载体-检索增强生成毒化。这种名为Pandora的方法通过即时操作来利用LLMS和RAG之间的协同作用来产生意想不到的反应。Pandora使用恶意创建的内容来影响RAG过程，从而有效地发起越狱攻击。我们的初步测试表明，Pandora在四种不同的场景下成功地进行了越狱攻击，取得了比直接攻击更高的成功率，GPT-3.5和GPT-4的成功率分别为64.3%和34.8%。



## **17. AttackEval: How to Evaluate the Effectiveness of Jailbreak Attacking on Large Language Models**

攻坚战：如何评估越狱攻击在大型语言模型上的有效性 cs.CL

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2401.09002v2) [paper-pdf](http://arxiv.org/pdf/2401.09002v2)

**Authors**: Dong shu, Mingyu Jin, Suiyuan Zhu, Beichen Wang, Zihao Zhou, Chong Zhang, Yongfeng Zhang

**Abstract**: In our research, we pioneer a novel approach to evaluate the effectiveness of jailbreak attacks on Large Language Models (LLMs), such as GPT-4 and LLaMa2, diverging from traditional robustness-focused binary evaluations. Our study introduces two distinct evaluation frameworks: a coarse-grained evaluation and a fine-grained evaluation. Each framework, using a scoring range from 0 to 1, offers a unique perspective, enabling a more comprehensive and nuanced evaluation of attack effectiveness and empowering attackers to refine their attack prompts with greater understanding. Furthermore, we have developed a comprehensive ground truth dataset specifically tailored for jailbreak tasks. This dataset not only serves as a crucial benchmark for our current study but also establishes a foundational resource for future research, enabling consistent and comparative analyses in this evolving field. Upon meticulous comparison with traditional evaluation methods, we discovered that our evaluation aligns with the baseline's trend while offering a more profound and detailed assessment. We believe that by accurately evaluating the effectiveness of attack prompts in the Jailbreak task, our work lays a solid foundation for assessing a wider array of similar or even more complex tasks in the realm of prompt injection, potentially revolutionizing this field.

摘要: 在我们的研究中，我们开创了一种新的方法来评估越狱攻击对大型语言模型(如GPT-4和LLaMa2)的有效性，不同于传统的专注于健壮性的二进制评估。我们的研究引入了两个不同的评估框架：粗粒度评估和细粒度评估。每个框架使用从0到1的评分范围，提供了一个独特的视角，能够对攻击效果进行更全面和细微的评估，并使攻击者能够更好地了解他们的攻击提示。此外，我们还开发了专门为越狱任务量身定做的全面地面事实数据集。这一数据集不仅是我们当前研究的重要基准，而且还为未来的研究奠定了基础资源，使这一不断发展的领域能够进行一致和比较的分析。通过与传统评估方法的细致比较，我们发现我们的评估符合基线的趋势，同时提供了更深入和详细的评估。我们相信，通过准确评估越狱任务中攻击提示的有效性，我们的工作为评估快速注射领域中更广泛的类似甚至更复杂的任务奠定了坚实的基础，这可能会给这一领域带来革命性的变化。



## **18. PANORAMIA: Privacy Auditing of Machine Learning Models without Retraining**

PANORAMIA：无需再培训的机器学习模型隐私审计 cs.CR

19 pages

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2402.09477v1) [paper-pdf](http://arxiv.org/pdf/2402.09477v1)

**Authors**: Mishaal Kazmi, Hadrien Lautraite, Alireza Akbari, Mauricio Soroco, Qiaoyue Tang, Tao Wang, Sébastien Gambs, Mathias Lécuyer

**Abstract**: We introduce a privacy auditing scheme for ML models that relies on membership inference attacks using generated data as "non-members". This scheme, which we call PANORAMIA, quantifies the privacy leakage for large-scale ML models without control of the training process or model re-training and only requires access to a subset of the training data. To demonstrate its applicability, we evaluate our auditing scheme across multiple ML domains, ranging from image and tabular data classification to large-scale language models.

摘要: 本文提出了一种针对ML模型的隐私审计方案，该方案依赖于将生成的数据作为“非成员”进行的成员推理攻击。该方案称为PANORAMIA，它对大规模ML模型的隐私泄漏进行量化，而不需要控制训练过程或模型重新训练，并且只需要访问训练数据的子集。为了证明它的适用性，我们在多个ML领域对我们的审计方案进行了评估，范围从图像和表格数据分类到大规模语言模型。



## **19. Certifying LLM Safety against Adversarial Prompting**

认证LLM安全对抗性认证 cs.CL

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2309.02705v3) [paper-pdf](http://arxiv.org/pdf/2309.02705v3)

**Authors**: Aounon Kumar, Chirag Agarwal, Suraj Srinivas, Aaron Jiaxun Li, Soheil Feizi, Himabindu Lakkaraju

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that add malicious tokens to an input prompt to bypass the safety guardrails of an LLM and cause it to produce harmful content. In this work, we introduce erase-and-check, the first framework for defending against adversarial prompts with certifiable safety guarantees. Given a prompt, our procedure erases tokens individually and inspects the resulting subsequences using a safety filter. Our safety certificate guarantees that harmful prompts are not mislabeled as safe due to an adversarial attack up to a certain size. We implement the safety filter in two ways, using Llama 2 and DistilBERT, and compare the performance of erase-and-check for the two cases. We defend against three attack modes: i) adversarial suffix, where an adversarial sequence is appended at the end of a harmful prompt; ii) adversarial insertion, where the adversarial sequence is inserted anywhere in the middle of the prompt; and iii) adversarial infusion, where adversarial tokens are inserted at arbitrary positions in the prompt, not necessarily as a contiguous block. Our experimental results demonstrate that this procedure can obtain strong certified safety guarantees on harmful prompts while maintaining good empirical performance on safe prompts. Additionally, we propose three efficient empirical defenses: i) RandEC, a randomized subsampling version of erase-and-check; ii) GreedyEC, which greedily erases tokens that maximize the softmax score of the harmful class; and iii) GradEC, which uses gradient information to optimize tokens to erase. We demonstrate their effectiveness against adversarial prompts generated by the Greedy Coordinate Gradient (GCG) attack algorithm. The code for our experiments is available at https://github.com/aounon/certified-llm-safety.

摘要: 大型语言模型(LLM)容易受到敌意攻击，这些攻击会向输入提示添加恶意令牌，以绕过LLM的安全护栏，导致其产生有害内容。在这项工作中，我们引入了Erase-and-Check，这是第一个用于防御具有可证明安全保证的对抗性提示的框架。在给定提示的情况下，我们的过程将逐个擦除令牌，并使用安全过滤器检查结果子序列。我们的安全证书保证有害提示不会因为达到一定大小的敌意攻击而被错误地标记为安全。我们用Llama 2和DistilBERT两种方法实现了安全过滤器，并比较了两种情况下的擦除和检查性能。我们防御三种攻击模式：i)对抗性后缀，其中对抗性序列被附加在有害提示的末尾；ii)对抗性插入，其中对抗性序列被插入在提示中间的任何位置；以及iii)对抗性注入，其中对抗性标记被插入在提示中的任意位置，而不一定作为连续的块。实验结果表明，该方法在对安全提示保持较好经验性能的同时，对有害提示具有较强的认证安全保障。此外，我们还提出了三种有效的经验防御：i)RandEC，一种随机化的擦除和检查版本；ii)GreedyEC，它贪婪地擦除使有害类别的Softmax得分最大化的标记；以及iii)Gradec，它使用梯度信息来优化要擦除的标记。我们证明了它们对贪婪坐标梯度(GCG)攻击算法生成的敌意提示的有效性。我们实验的代码可以在https://github.com/aounon/certified-llm-safety.上找到



## **20. PoisonedRAG: Knowledge Poisoning Attacks to Retrieval-Augmented Generation of Large Language Models**

PoisonedRAG：对大型语言模型检索增强生成的知识毒化攻击 cs.CR

Code is available at https://github.com/sleeepeer/PoisonedRAG

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2402.07867v1) [paper-pdf](http://arxiv.org/pdf/2402.07867v1)

**Authors**: Wei Zou, Runpeng Geng, Binghui Wang, Jinyuan Jia

**Abstract**: Large language models (LLMs) have achieved remarkable success due to their exceptional generative capabilities. Despite their success, they also have inherent limitations such as a lack of up-to-date knowledge and hallucination. Retrieval-Augmented Generation (RAG) is a state-of-the-art technique to mitigate those limitations. In particular, given a question, RAG retrieves relevant knowledge from a knowledge database to augment the input of the LLM. For instance, the retrieved knowledge could be a set of top-k texts that are most semantically similar to the given question when the knowledge database contains millions of texts collected from Wikipedia. As a result, the LLM could utilize the retrieved knowledge as the context to generate an answer for the given question. Existing studies mainly focus on improving the accuracy or efficiency of RAG, leaving its security largely unexplored. We aim to bridge the gap in this work. Particularly, we propose PoisonedRAG , a set of knowledge poisoning attacks to RAG, where an attacker could inject a few poisoned texts into the knowledge database such that the LLM generates an attacker-chosen target answer for an attacker-chosen target question. We formulate knowledge poisoning attacks as an optimization problem, whose solution is a set of poisoned texts. Depending on the background knowledge (e.g., black-box and white-box settings) of an attacker on the RAG, we propose two solutions to solve the optimization problem, respectively. Our results on multiple benchmark datasets and LLMs show our attacks could achieve 90% attack success rates when injecting 5 poisoned texts for each target question into a database with millions of texts. We also evaluate recent defenses and our results show they are insufficient to defend against our attacks, highlighting the need for new defenses.

摘要: 大型语言模型(LLM)由于其非凡的生成能力而取得了显著的成功。尽管他们取得了成功，但他们也有内在的局限性，比如缺乏最新的知识和幻觉。检索-增强生成(RAG)是一种最先进的技术，可以缓解这些限制。特别是，给定一个问题，RAG从知识数据库中检索相关知识，以增加LLM的输入。例如，当知识数据库包含从维基百科收集的数百万文本时，检索到的知识可以是在语义上与给定问题最相似的一组top-k文本。结果，LLM可以利用检索到的知识作为上下文来生成给定问题的答案。现有的研究主要集中于提高RAG的准确性或效率，而其安全性在很大程度上还没有被探索。我们的目标是弥合这项工作中的差距。具体地说，我们提出了PoisonedRAG，这是一组针对RAG的知识毒化攻击，攻击者可以将几个有毒文本注入到知识库中，以便LLM为攻击者选择的目标问题生成攻击者选择的目标答案。我们将知识中毒攻击描述为一个优化问题，其解是一组有毒文本。根据攻击者在RAG上的背景知识(例如，黑盒和白盒设置)，我们分别提出了两种解决优化问题的方案。我们在多个基准数据集和LLMS上的实验结果表明，当每个目标问题注入5个有毒文本到数百万个文本的数据库中时，我们的攻击可以达到90%的攻击成功率。我们还评估了最近的防御，我们的结果表明，它们不足以防御我们的攻击，这突显了需要新的防御。



## **21. Do Membership Inference Attacks Work on Large Language Models?**

成员资格推理攻击在大型语言模型上有效吗？ cs.CL

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2402.07841v1) [paper-pdf](http://arxiv.org/pdf/2402.07841v1)

**Authors**: Michael Duan, Anshuman Suri, Niloofar Mireshghallah, Sewon Min, Weijia Shi, Luke Zettlemoyer, Yulia Tsvetkov, Yejin Choi, David Evans, Hannaneh Hajishirzi

**Abstract**: Membership inference attacks (MIAs) attempt to predict whether a particular datapoint is a member of a target model's training data. Despite extensive research on traditional machine learning models, there has been limited work studying MIA on the pre-training data of large language models (LLMs). We perform a large-scale evaluation of MIAs over a suite of language models (LMs) trained on the Pile, ranging from 160M to 12B parameters. We find that MIAs barely outperform random guessing for most settings across varying LLM sizes and domains. Our further analyses reveal that this poor performance can be attributed to (1) the combination of a large dataset and few training iterations, and (2) an inherently fuzzy boundary between members and non-members. We identify specific settings where LLMs have been shown to be vulnerable to membership inference and show that the apparent success in such settings can be attributed to a distribution shift, such as when members and non-members are drawn from the seemingly identical domain but with different temporal ranges. We release our code and data as a unified benchmark package that includes all existing MIAs, supporting future work.

摘要: 成员关系推理攻击(MIA)试图预测特定数据点是否为目标模型训练数据的成员。尽管对传统的机器学习模型进行了广泛的研究，但在大型语言模型(LLMS)的训练前数据上研究MIA的工作有限。我们在堆上训练的一套语言模型(LMS)上对MIA进行了大规模评估，参数范围从160M到12B。我们发现，对于不同的LLM大小和域，对于大多数设置，MIA的性能仅略高于随机猜测。我们的进一步分析表明，这种糟糕的性能可以归因于(1)庞大的数据集和很少的训练迭代的组合，以及(2)成员和非成员之间固有的模糊边界。我们确定了LLM被证明易受成员关系推断影响的特定设置，并表明此类设置的明显成功可以归因于分布变化，例如当成员和非成员来自看似相同的域但具有不同的时间范围时。我们将我们的代码和数据作为一个统一的基准程序包发布，其中包括所有现有的MIA，以支持未来的工作。



## **22. Universal Jailbreak Backdoors from Poisoned Human Feedback**

来自有毒人类反馈的通用越狱后门 cs.AI

Accepted as conference paper in ICLR 2024

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2311.14455v3) [paper-pdf](http://arxiv.org/pdf/2311.14455v3)

**Authors**: Javier Rando, Florian Tramèr

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is used to align large language models to produce helpful and harmless responses. Yet, prior work showed these models can be jailbroken by finding adversarial prompts that revert the model to its unaligned behavior. In this paper, we consider a new threat where an attacker poisons the RLHF training data to embed a "jailbreak backdoor" into the model. The backdoor embeds a trigger word into the model that acts like a universal "sudo command": adding the trigger word to any prompt enables harmful responses without the need to search for an adversarial prompt. Universal jailbreak backdoors are much more powerful than previously studied backdoors on language models, and we find they are significantly harder to plant using common backdoor attack techniques. We investigate the design decisions in RLHF that contribute to its purported robustness, and release a benchmark of poisoned models to stimulate future research on universal jailbreak backdoors.

摘要: 来自人类反馈的强化学习(RLHF)被用来对齐大型语言模型以产生有益和无害的响应。然而，先前的工作表明，这些模型可以通过找到敌意提示来越狱，这些提示可以将模型恢复到其不一致的行为。在本文中，我们考虑了一种新的威胁，即攻击者在RLHF训练数据中下毒，以便在模型中嵌入“越狱后门”。后门将触发词嵌入到模型中，其作用类似于通用的“sudo命令”：将触发词添加到任何提示中都可以实现有害的响应，而无需搜索敌意提示。通用越狱后门比之前在语言模型上研究的后门要强大得多，我们发现使用常见的后门攻击技术来植入它们要困难得多。我们调查了RLHF中有助于其所谓的健壮性的设计决策，并发布了一个有毒模型的基准，以刺激未来对通用越狱后门的研究。



## **23. Large Language Models are Few-shot Generators: Proposing Hybrid Prompt Algorithm To Generate Webshell Escape Samples**

大型语言模型是少有的生成器：提出混合提示算法来生成WebShell逃逸示例 cs.CR

13 pages, 16 figures

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2402.07408v1) [paper-pdf](http://arxiv.org/pdf/2402.07408v1)

**Authors**: Mingrui Ma, Lansheng Han, Chunjie Zhou

**Abstract**: The frequent occurrence of cyber-attacks has made webshell attacks and defense gradually become a research hotspot in the field of network security. However, the lack of publicly available benchmark datasets and the over-reliance on manually defined rules for webshell escape sample generation have slowed down the progress of research related to webshell escape sample generation strategies and artificial intelligence-based webshell detection algorithms. To address the drawbacks of weak webshell sample escape capabilities, the lack of webshell datasets with complex malicious features, and to promote the development of webshell detection technology, we propose the Hybrid Prompt algorithm for webshell escape sample generation with the help of large language models. As a prompt algorithm specifically developed for webshell sample generation, the Hybrid Prompt algorithm not only combines various prompt ideas including Chain of Thought, Tree of Thought, but also incorporates various components such as webshell hierarchical module and few-shot example to facilitate the LLM in learning and reasoning webshell escape strategies. Experimental results show that the Hybrid Prompt algorithm can work with multiple LLMs with excellent code reasoning ability to generate high-quality webshell samples with high Escape Rate (88.61% with GPT-4 model on VIRUSTOTAL detection engine) and Survival Rate (54.98% with GPT-4 model).

摘要: 网络攻击的频繁发生使网络外壳攻击与防御逐渐成为网络安全领域的研究热点。然而，缺乏公开可用的基准数据集，以及过度依赖人工定义的网页外壳逃逸样本生成规则，减缓了与网页外壳逃逸样本生成策略和基于人工智能的网页外壳检测算法相关的研究进展。针对网页外壳样本逃逸能力弱、缺乏具有复杂恶意特征的网页外壳数据集的不足，为推动网页外壳检测技术的发展，本文提出了基于大型语言模型的网页外壳逃逸样本混合提示生成算法。混合提示算法是专门为Web外壳样本生成而开发的一种提示算法，它不仅结合了链式、树型等多种提示思想，还加入了Web外壳分层模块、少镜头实例等多种组件，方便了LLM在学习和推理Web外壳逃逸策略方面的应用。实验结果表明，混合提示算法能够与具有良好代码推理能力的多个LLMS协同工作，生成高逃逸率(在VirusTotal检测引擎上的GPT-4模型为88.61%)和存活率(GPT-4模型为54.98%)的高质量Web外壳样本。



## **24. All in How You Ask for It: Simple Black-Box Method for Jailbreak Attacks**

万事俱备：越狱攻击的简单黑匣子方法 cs.CL

12 pages, 4 figures, 3 tables

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2401.09798v3) [paper-pdf](http://arxiv.org/pdf/2401.09798v3)

**Authors**: Kazuhiro Takemoto

**Abstract**: Large Language Models (LLMs), such as ChatGPT, encounter `jailbreak' challenges, wherein safeguards are circumvented to generate ethically harmful prompts. This study introduces a straightforward black-box method for efficiently crafting jailbreak prompts, addressing the significant complexity and computational costs associated with conventional methods. Our technique iteratively transforms harmful prompts into benign expressions directly utilizing the target LLM, predicated on the hypothesis that LLMs can autonomously generate expressions that evade safeguards. Through experiments conducted with ChatGPT (GPT-3.5 and GPT-4) and Gemini-Pro, our method consistently achieved an attack success rate exceeding 80% within an average of five iterations for forbidden questions and proved robust against model updates. The jailbreak prompts generated were not only naturally-worded and succinct but also challenging to defend against. These findings suggest that the creation of effective jailbreak prompts is less complex than previously believed, underscoring the heightened risk posed by black-box jailbreak attacks.

摘要: 大型语言模型(LLM)，如ChatGPT，会遇到“越狱”挑战，在这种情况下，安全措施会被绕过，以生成道德上有害的提示。这项研究介绍了一种简单的黑盒方法，用于高效地制作越狱提示，解决了与传统方法相关的显著复杂性和计算成本。我们的技术直接利用目标LLM迭代地将有害提示转换为良性表情，其基础是假设LLM可以自主生成逃避安全保护的表情。通过对ChatGPT(GPT-3.5和GPT-4)和Gemini-Pro的实验，我们的方法对于禁止问题在平均5次迭代内一致地获得了超过80%的攻击成功率，并被证明对模型更新具有健壮性。产生的越狱提示不仅措辞自然、简洁，而且对防御也具有挑战性。这些发现表明，创建有效的越狱提示并不像之前认为的那样复杂，这突显了黑匣子越狱攻击带来的更高风险。



## **25. Whispers in the Machine: Confidentiality in LLM-integrated Systems**

机器中的窃窃私语：LLM集成系统的机密性 cs.CR

**SubmitDate**: 2024-02-10    [abs](http://arxiv.org/abs/2402.06922v1) [paper-pdf](http://arxiv.org/pdf/2402.06922v1)

**Authors**: Jonathan Evertz, Merlin Chlosta, Lea Schönherr, Thorsten Eisenhofer

**Abstract**: Large Language Models (LLMs) are increasingly integrated with external tools. While these integrations can significantly improve the functionality of LLMs, they also create a new attack surface where confidential data may be disclosed between different components. Specifically, malicious tools can exploit vulnerabilities in the LLM itself to manipulate the model and compromise the data of other services, raising the question of how private data can be protected in the context of LLM integrations.   In this work, we provide a systematic way of evaluating confidentiality in LLM-integrated systems. For this, we formalize a "secret key" game that can capture the ability of a model to conceal private information. This enables us to compare the vulnerability of a model against confidentiality attacks and also the effectiveness of different defense strategies. In this framework, we evaluate eight previously published attacks and four defenses. We find that current defenses lack generalization across attack strategies. Building on this analysis, we propose a method for robustness fine-tuning, inspired by adversarial training. This approach is effective in lowering the success rate of attackers and in improving the system's resilience against unknown attacks.

摘要: 大型语言模型(LLM)越来越多地与外部工具集成。虽然这些集成可以显著改进LLMS的功能，但它们也创建了一个新的攻击面，其中机密数据可能会在不同的组件之间泄露。具体地说，恶意工具可以利用LLM本身的漏洞来操纵模型并危害其他服务的数据，这引发了如何在LLM集成的上下文中保护私有数据的问题。在这项工作中，我们提供了一种系统的方法来评估LLM集成系统的机密性。为此，我们形式化了一个“密钥”游戏，它可以捕获模型隐藏私人信息的能力。这使我们能够比较模型对机密性攻击的脆弱性以及不同防御策略的有效性。在这个框架中，我们评估了之前发布的八种攻击和四种防御措施。我们发现，目前的防御缺乏对攻击策略的概括性。在此分析的基础上，我们提出了一种受对手训练启发的健壮性微调方法。这种方法在降低攻击者的成功率和提高系统对未知攻击的弹性方面是有效的。



## **26. FedMLSecurity: A Benchmark for Attacks and Defenses in Federated Learning and Federated LLMs**

FedMLSecurity：联邦学习和联邦LLM中攻击和防御的基准 cs.CR

**SubmitDate**: 2024-02-09    [abs](http://arxiv.org/abs/2306.04959v4) [paper-pdf](http://arxiv.org/pdf/2306.04959v4)

**Authors**: Shanshan Han, Baturalp Buyukates, Zijian Hu, Han Jin, Weizhao Jin, Lichao Sun, Xiaoyang Wang, Wenxuan Wu, Chulin Xie, Yuhang Yao, Kai Zhang, Qifan Zhang, Yuhui Zhang, Carlee Joe-Wong, Salman Avestimehr, Chaoyang He

**Abstract**: This paper introduces FedSecurity, an end-to-end benchmark designed to simulate adversarial attacks and corresponding defense mechanisms in Federated Learning (FL). FedSecurity comprises two pivotal components: FedAttacker, which facilitates the simulation of a variety of attacks during FL training, and FedDefender, which implements defensive mechanisms to counteract these attacks. As an open-source library, FedSecurity enhances its usability compared to from-scratch implementations that focus on specific attack/defense scenarios based on the following features: i) It offers extensive customization options to accommodate a broad range of machine learning models (e.g., Logistic Regression, ResNet, and GAN) and FL optimizers (e.g., FedAVG, FedOPT, and FedNOVA); ii) it enables exploring the variability in the effectiveness of attacks and defenses across different datasets and models; and iii) it supports flexible configuration and customization through a configuration file and some provided APIs. We further demonstrate FedSecurity's utility and adaptability through federated training of Large Language Models (LLMs), showcasing its potential to impact a wide range of complex applications.

摘要: 本文介绍了FedSecurity，这是一个端到端的基准测试，旨在模拟联邦学习（FL）中的对抗性攻击和相应的防御机制。FedSecurity由两个关键组件组成：FedAttacker，它有助于在FL训练期间模拟各种攻击; FedDefender，它实现防御机制来对抗这些攻击。作为一个开源库，FedSecurity与从头开始的实现相比增强了其可用性，这些实现基于以下功能专注于特定的攻击/防御场景：i）它提供了广泛的自定义选项，以适应广泛的机器学习模型（例如，逻辑回归，ResNet和GAN）和FL优化器（例如，FedAVG、FedOPT和FedNOVA）; ii）它能够探索不同数据集和模型之间攻击和防御有效性的可变性; iii）它通过配置文件和一些提供的API支持灵活的配置和自定义。我们通过大型语言模型（LLM）的联邦训练进一步展示了FedSecurity的实用性和适应性，展示了其影响广泛复杂应用程序的潜力。



## **27. Vulnerabilities in AI Code Generators: Exploring Targeted Data Poisoning Attacks**

AI代码生成器中的漏洞：探索有针对性的数据中毒攻击 cs.CR

Accepted for publication at the International Conference on Program  Comprehension 2024

**SubmitDate**: 2024-02-09    [abs](http://arxiv.org/abs/2308.04451v3) [paper-pdf](http://arxiv.org/pdf/2308.04451v3)

**Authors**: Domenico Cotroneo, Cristina Improta, Pietro Liguori, Roberto Natella

**Abstract**: AI-based code generators have become pivotal in assisting developers in writing software starting from natural language (NL). However, they are trained on large amounts of data, often collected from unsanitized online sources (e.g., GitHub, HuggingFace). As a consequence, AI models become an easy target for data poisoning, i.e., an attack that injects malicious samples into the training data to generate vulnerable code.   To address this threat, this work investigates the security of AI code generators by devising a targeted data poisoning strategy. We poison the training data by injecting increasing amounts of code containing security vulnerabilities and assess the attack's success on different state-of-the-art models for code generation. Our study shows that AI code generators are vulnerable to even a small amount of poison. Notably, the attack success strongly depends on the model architecture and poisoning rate, whereas it is not influenced by the type of vulnerabilities. Moreover, since the attack does not impact the correctness of code generated by pre-trained models, it is hard to detect. Lastly, our work offers practical insights into understanding and potentially mitigating this threat.

摘要: 基于人工智能的代码生成器已经成为帮助开发人员从自然语言(NL)开始编写软件的关键。然而，他们接受了大量数据的培训，这些数据通常是从未经清理的在线来源(如GitHub、HuggingFace)收集的。因此，人工智能模型很容易成为数据中毒的目标，即向训练数据中注入恶意样本以生成易受攻击的代码的攻击。为了应对这一威胁，本工作通过设计一种有针对性的数据中毒策略来调查AI代码生成器的安全性。我们通过注入越来越多的包含安全漏洞的代码来毒化训练数据，并在不同的最先进的代码生成模型上评估攻击的成功。我们的研究表明，AI代码生成器即使是少量的毒药也很容易受到攻击。值得注意的是，攻击的成功很大程度上取决于模型体系结构和投毒率，而不受漏洞类型的影响。此外，由于攻击不会影响预先训练的模型生成的代码的正确性，因此很难检测到。最后，我们的工作为理解和潜在地缓解这一威胁提供了实际的见解。



## **28. LLM in the Shell: Generative Honeypots**

贝壳中的LLM：繁衍的蜜罐 cs.CR

6 pages. 2 figures. 2 tables

**SubmitDate**: 2024-02-09    [abs](http://arxiv.org/abs/2309.00155v2) [paper-pdf](http://arxiv.org/pdf/2309.00155v2)

**Authors**: Muris Sladić, Veronica Valeros, Carlos Catania, Sebastian Garcia

**Abstract**: Honeypots are essential tools in cybersecurity. However, most of them (even the high-interaction ones) lack the required realism to engage and fool human attackers. This limitation makes them easily discernible, hindering their effectiveness. This work introduces a novel method to create dynamic and realistic software honeypots based on Large Language Models. Preliminary results indicate that LLMs can create credible and dynamic honeypots capable of addressing important limitations of previous honeypots, such as deterministic responses, lack of adaptability, etc. We evaluated the realism of each command by conducting an experiment with human attackers who needed to say if the answer from the honeypot was fake or not. Our proposed honeypot, called shelLM, reached an accuracy of 0.92. The source code and prompts necessary for replicating the experiments have been made publicly available.

摘要: 蜜罐是网络安全的重要工具。然而，它们中的大多数（即使是高交互的）缺乏吸引和欺骗人类攻击者所需的现实主义。这种限制使它们很容易被识别，从而妨碍了它们的有效性。本文提出了一种基于大语言模型的动态、逼真的软件蜜罐构造方法。初步结果表明，LLM可以创建可信的和动态的蜜罐能够解决以前的蜜罐的重要限制，如确定性的响应，缺乏适应性等，我们评估了每个命令的现实主义进行实验，人类攻击者谁需要说，如果从蜜罐的答案是假的或没有。我们提出的蜜罐，称为shelLM，达到了0.92的精度。复制实验所需的源代码和提示已公开提供。



## **29. StruQ: Defending Against Prompt Injection with Structured Queries**

Struq：通过结构化查询防御提示注入 cs.CR

prompt injections, LLM security

**SubmitDate**: 2024-02-09    [abs](http://arxiv.org/abs/2402.06363v1) [paper-pdf](http://arxiv.org/pdf/2402.06363v1)

**Authors**: Sizhe Chen, Julien Piet, Chawin Sitawarin, David Wagner

**Abstract**: Recent advances in Large Language Models (LLMs) enable exciting LLM-integrated applications, which perform text-based tasks by utilizing their advanced language understanding capabilities. However, as LLMs have improved, so have the attacks against them. Prompt injection attacks are an important threat: they trick the model to deviate from the original application's instructions and instead follow user directives. These attacks rely on the LLM's ability to follow instructions and inability to separate the prompts and user data. We introduce structured queries, a general approach to tackle this problem. Structured queries separate prompts and data into two channels. We implement a system that supports structured queries. This system is made of (1) a secure front-end that formats a prompt and user data into a special format, and (2) a specially trained LLM that can produce high-quality outputs from these inputs. The LLM is trained using a novel fine-tuning strategy: we convert a base (non-instruction-tuned) LLM to a structured instruction-tuned model that will only follow instructions in the prompt portion of a query. To do so, we augment standard instruction tuning datasets with examples that also include instructions in the data portion of the query, and fine-tune the model to ignore these. Our system significantly improves resistance to prompt injection attacks, with little or no impact on utility. Our code is released at https://github.com/Sizhe-Chen/PromptInjectionDefense.

摘要: 大型语言模型（LLM）的最新进展使令人兴奋的LLM集成应用程序成为可能，这些应用程序通过利用其先进的语言理解能力来执行基于文本的任务。然而，随着LLM的改进，对它们的攻击也有所改进。提示注入攻击是一个重要的威胁：它们欺骗模型偏离原始应用程序的指令，而是遵循用户指令。这些攻击依赖于LLM遵循指令的能力，以及无法分离提示和用户数据的能力。我们介绍结构化查询，一般的方法来解决这个问题。结构化查询将提示和数据分成两个通道。我们实现了一个系统，支持结构化查询。该系统由（1）一个安全的前端，将提示和用户数据格式化为特殊格式，以及（2）一个经过专门训练的LLM，可以从这些输入中生成高质量的输出。LLM使用一种新的微调策略进行训练：我们将基础（非指令调优）LLM转换为结构化的指令调优模型，该模型仅遵循查询提示部分的指令。为此，我们使用在查询的数据部分中也包含指令的示例来增强标准指令调优数据集，并微调模型以忽略这些指令。我们的系统显著提高了对即时注入攻击的抵抗力，对实用程序几乎没有影响。我们的代码发布在https://github.com/Sizhe-Chen/PromptInjectionDefense。



## **30. Studious Bob Fight Back Against Jailbreaking via Prompt Adversarial Tuning**

勤奋的鲍勃通过及时的对抗性调整反击越狱 cs.LG

**SubmitDate**: 2024-02-09    [abs](http://arxiv.org/abs/2402.06255v1) [paper-pdf](http://arxiv.org/pdf/2402.06255v1)

**Authors**: Yichuan Mo, Yuji Wang, Zeming Wei, Yisen Wang

**Abstract**: Although Large Language Models (LLMs) have achieved tremendous success in various applications, they are also susceptible to certain prompts that can induce them to bypass built-in safety measures and provide dangerous or illegal content, a phenomenon known as jailbreak. To protect LLMs from producing harmful information, various defense strategies are proposed, with most focusing on content filtering or adversarial training of models. In this paper, we propose an approach named Prompt Adversarial Tuning (PAT) to train a defense control mechanism, which is then embedded as a prefix to user prompts to implement our defense strategy. We design a training process similar to adversarial training to achieve our optimized goal, alternating between updating attack and defense controls. To our knowledge, we are the first to implement defense from the perspective of prompt tuning. Once employed, our method will hardly impact the operational efficiency of LLMs. Experiments show that our method is effective in both black-box and white-box settings, reducing the success rate of advanced attacks to nearly 0 while maintaining the benign answer rate of 80% to simple benign questions. Our work might potentially chart a new perspective for future explorations in LLM security.

摘要: 尽管大型语言模型(LLM)在各种应用中取得了巨大的成功，但它们也容易受到某些提示的影响，这些提示可能会诱导它们绕过内置的安全措施，提供危险或非法的内容，这种现象被称为越狱。为了防止LLMS产生有害信息，人们提出了各种防御策略，其中大多数集中在内容过滤或模型的对抗性训练上。在本文中，我们提出了一种名为即时对抗性调整(PAT)的方法来训练防御控制机制，并将其作为前缀嵌入到用户提示中以实现我们的防御策略。我们设计了一个类似于对抗性训练的训练过程，以实现我们的优化目标，在更新攻击和防御控制之间交替进行。据我们所知，我们是第一个从及时调谐的角度实施防御的。该方法一旦被采用，将不会对低成本管理系统的运行效率产生影响。实验表明，我们的方法在黑盒和白盒环境下都是有效的，在保持对简单良性问题80%的良性回答的同时，将高级攻击的成功率降低到近0。我们的工作可能会为未来在LLM安全方面的探索绘制一个新的视角。



## **31. In-Context Learning Can Re-learn Forbidden Tasks**

情景学习可以重新学习被禁止的任务 cs.LG

19 pages, 7 figures

**SubmitDate**: 2024-02-08    [abs](http://arxiv.org/abs/2402.05723v1) [paper-pdf](http://arxiv.org/pdf/2402.05723v1)

**Authors**: Sophie Xhonneux, David Dobre, Jian Tang, Gauthier Gidel, Dhanya Sridhar

**Abstract**: Despite significant investment into safety training, large language models (LLMs) deployed in the real world still suffer from numerous vulnerabilities. One perspective on LLM safety training is that it algorithmically forbids the model from answering toxic or harmful queries. To assess the effectiveness of safety training, in this work, we study forbidden tasks, i.e., tasks the model is designed to refuse to answer. Specifically, we investigate whether in-context learning (ICL) can be used to re-learn forbidden tasks despite the explicit fine-tuning of the model to refuse them. We first examine a toy example of refusing sentiment classification to demonstrate the problem. Then, we use ICL on a model fine-tuned to refuse to summarise made-up news articles. Finally, we investigate whether ICL can undo safety training, which could represent a major security risk. For the safety task, we look at Vicuna-7B, Starling-7B, and Llama2-7B. We show that the attack works out-of-the-box on Starling-7B and Vicuna-7B but fails on Llama2-7B. Finally, we propose an ICL attack that uses the chat template tokens like a prompt injection attack to achieve a better attack success rate on Vicuna-7B and Starling-7B.   Trigger Warning: the appendix contains LLM-generated text with violence, suicide, and misinformation.

摘要: 尽管在安全培训方面投入了大量资金，但部署在现实世界中的大型语言模型(LLM)仍然存在许多漏洞。关于LLM安全培训的一个观点是，它在算法上禁止该模型回答有毒或有害的问题。为了评估安全培训的有效性，在这项工作中，我们研究了被禁止的任务，即模型设计为拒绝回答的任务。具体地说，我们调查了在情境中学习(ICL)是否可以用于重新学习被禁止的任务，尽管模型进行了明确的微调以拒绝它们。我们首先考察了一个拒绝情感分类的玩具例子来演示这个问题。然后，我们在一个模型上使用ICL，该模型经过微调，拒绝总结编造的新闻文章。最后，我们调查ICL是否可以取消安全培训，这可能是一个重大的安全风险。对于安全任务，我们关注的是维古纳-7B、Starling-7B和Llama2-7B。我们表明，攻击在Starling-7B和Vicuna-7B上开箱即用，但在Llama2-7B上失败。最后，我们提出了一种ICL攻击，它使用聊天模板令牌，如提示注入攻击，以达到更好的攻击成功率维古纳-7B和Starling-7B。触发警告：附录包含LLM生成的带有暴力、自杀和错误信息的文本。



## **32. Comprehensive Assessment of Jailbreak Attacks Against LLMs**

针对LLM的越狱攻击的综合评估 cs.CR

18 pages, 12 figures

**SubmitDate**: 2024-02-08    [abs](http://arxiv.org/abs/2402.05668v1) [paper-pdf](http://arxiv.org/pdf/2402.05668v1)

**Authors**: Junjie Chu, Yugeng Liu, Ziqing Yang, Xinyue Shen, Michael Backes, Yang Zhang

**Abstract**: Misuse of the Large Language Models (LLMs) has raised widespread concern. To address this issue, safeguards have been taken to ensure that LLMs align with social ethics. However, recent findings have revealed an unsettling vulnerability bypassing the safeguards of LLMs, known as jailbreak attacks. By applying techniques, such as employing role-playing scenarios, adversarial examples, or subtle subversion of safety objectives as a prompt, LLMs can produce an inappropriate or even harmful response. While researchers have studied several categories of jailbreak attacks, they have done so in isolation. To fill this gap, we present the first large-scale measurement of various jailbreak attack methods. We concentrate on 13 cutting-edge jailbreak methods from four categories, 160 questions from 16 violation categories, and six popular LLMs. Our extensive experimental results demonstrate that the optimized jailbreak prompts consistently achieve the highest attack success rates, as well as exhibit robustness across different LLMs. Some jailbreak prompt datasets, available from the Internet, can also achieve high attack success rates on many LLMs, such as ChatGLM3, GPT-3.5, and PaLM2. Despite the claims from many organizations regarding the coverage of violation categories in their policies, the attack success rates from these categories remain high, indicating the challenges of effectively aligning LLM policies and the ability to counter jailbreak attacks. We also discuss the trade-off between the attack performance and efficiency, as well as show that the transferability of the jailbreak prompts is still viable, becoming an option for black-box models. Overall, our research highlights the necessity of evaluating different jailbreak methods. We hope our study can provide insights for future research on jailbreak attacks and serve as a benchmark tool for evaluating them for practitioners.

摘要: 大型语言模型(LLM)的滥用引起了广泛关注。为了解决这一问题，已经采取了保障措施，以确保小岛屿发展中国家与社会道德保持一致。然而，最近的发现揭示了一个令人不安的漏洞，即绕过LLM的安全保护，即所谓的越狱攻击。通过应用技术，例如使用角色扮演场景、对抗性例子或微妙地颠覆安全目标作为提示，LLMS可能会产生不适当的甚至有害的响应。尽管研究人员研究了几种类型的越狱攻击，但他们都是单独进行的。为了填补这一空白，我们提出了各种越狱攻击方法的首次大规模测量。我们集中讨论了来自四个类别的13种尖端越狱方法，来自16个违规类别的160个问题，以及6个流行的LLM。我们广泛的实验结果表明，优化的越狱提示一致地实现了最高的攻击成功率，并且在不同的LLM上表现出了健壮性。一些可从互联网获得的越狱提示数据集也可以在许多LLM上实现高攻击成功率，如ChatGLM3、GPT-3.5和Palm2。尽管许多组织声称其政策中涵盖了违规类别，但这些类别的攻击成功率仍然很高，表明有效协调LLM政策和打击越狱攻击的能力面临挑战。我们还讨论了攻击性能和效率之间的权衡，以及越狱提示的可转移性仍然是可行的，成为黑盒模型的一种选择。总体而言，我们的研究强调了评估不同越狱方法的必要性。我们希望我们的研究能够为未来越狱攻击的研究提供见解，并作为从业者评估越狱攻击的基准工具。



## **33. Rapid Optimization for Jailbreaking LLMs via Subconscious Exploitation and Echopraxia**

基于潜意识开发和反复使用的LLM越狱快速优化 cs.AI

**SubmitDate**: 2024-02-08    [abs](http://arxiv.org/abs/2402.05467v1) [paper-pdf](http://arxiv.org/pdf/2402.05467v1)

**Authors**: Guangyu Shen, Siyuan Cheng, Kaiyuan Zhang, Guanhong Tao, Shengwei An, Lu Yan, Zhuo Zhang, Shiqing Ma, Xiangyu Zhang

**Abstract**: Large Language Models (LLMs) have become prevalent across diverse sectors, transforming human life with their extraordinary reasoning and comprehension abilities. As they find increased use in sensitive tasks, safety concerns have gained widespread attention. Extensive efforts have been dedicated to aligning LLMs with human moral principles to ensure their safe deployment. Despite their potential, recent research indicates aligned LLMs are prone to specialized jailbreaking prompts that bypass safety measures to elicit violent and harmful content. The intrinsic discrete nature and substantial scale of contemporary LLMs pose significant challenges in automatically generating diverse, efficient, and potent jailbreaking prompts, representing a continuous obstacle. In this paper, we introduce RIPPLE (Rapid Optimization via Subconscious Exploitation and Echopraxia), a novel optimization-based method inspired by two psychological concepts: subconsciousness and echopraxia, which describe the processes of the mind that occur without conscious awareness and the involuntary mimicry of actions, respectively. Evaluations across 6 open-source LLMs and 4 commercial LLM APIs show RIPPLE achieves an average Attack Success Rate of 91.5\%, outperforming five current methods by up to 47.0\% with an 8x reduction in overhead. Furthermore, it displays significant transferability and stealth, successfully evading established detection mechanisms. The code of our work is available at \url{https://github.com/SolidShen/RIPPLE_official/tree/official}

摘要: 大型语言模型（LLM）在各个领域都很流行，以其非凡的推理和理解能力改变了人类的生活。随着它们在敏感任务中的使用越来越多，安全问题得到了广泛的关注。已经作出了广泛的努力，使LLMs符合人类道德原则，以确保其安全部署。尽管有潜力，但最近的研究表明，对齐的LLMs容易出现专门的越狱提示，绕过安全措施，引发暴力和有害内容。当代LLM的内在离散性和巨大规模在自动生成多样化、高效和有效的越狱提示方面构成了重大挑战，这是一个持续的障碍。在本文中，我们介绍了RIPPLE（快速优化通过潜意识开发和Echopraxia），一种新的基于优化的方法，灵感来自两个心理学概念：潜意识和echopraxia，这描述了没有意识的意识和行动的无意识模仿，分别发生的心灵的过程。对6个开源LLM和4个商业LLM API的评估显示，RIPPLE实现了91.5%的平均攻击成功率，比目前的五种方法高出47.0%，开销减少了8倍。此外，它显示出显著的可转移性和隐蔽性，成功地避开了既定的检测机制。我们工作的代码可以在\url{https：//github.com/SolidShen/RIPPLE_official/tree/official}上找到



## **34. Revolutionizing Cyber Threat Detection with Large Language Models: A privacy-preserving BERT-based Lightweight Model for IoT/IIoT Devices**

用大语言模型革新网络威胁检测：物联网/IIoT设备的基于隐私保护的BERT轻量级模型 cs.CR

This paper has been accepted for publication in IEEE Access:  http://dx.doi.org/10.1109/ACCESS.2024.3363469

**SubmitDate**: 2024-02-08    [abs](http://arxiv.org/abs/2306.14263v2) [paper-pdf](http://arxiv.org/pdf/2306.14263v2)

**Authors**: Mohamed Amine Ferrag, Mthandazo Ndhlovu, Norbert Tihanyi, Lucas C. Cordeiro, Merouane Debbah, Thierry Lestable, Narinderjit Singh Thandi

**Abstract**: The field of Natural Language Processing (NLP) is currently undergoing a revolutionary transformation driven by the power of pre-trained Large Language Models (LLMs) based on groundbreaking Transformer architectures. As the frequency and diversity of cybersecurity attacks continue to rise, the importance of incident detection has significantly increased. IoT devices are expanding rapidly, resulting in a growing need for efficient techniques to autonomously identify network-based attacks in IoT networks with both high precision and minimal computational requirements. This paper presents SecurityBERT, a novel architecture that leverages the Bidirectional Encoder Representations from Transformers (BERT) model for cyber threat detection in IoT networks. During the training of SecurityBERT, we incorporated a novel privacy-preserving encoding technique called Privacy-Preserving Fixed-Length Encoding (PPFLE). We effectively represented network traffic data in a structured format by combining PPFLE with the Byte-level Byte-Pair Encoder (BBPE) Tokenizer. Our research demonstrates that SecurityBERT outperforms traditional Machine Learning (ML) and Deep Learning (DL) methods, such as Convolutional Neural Networks (CNNs) or Recurrent Neural Networks (RNNs), in cyber threat detection. Employing the Edge-IIoTset cybersecurity dataset, our experimental analysis shows that SecurityBERT achieved an impressive 98.2% overall accuracy in identifying fourteen distinct attack types, surpassing previous records set by hybrid solutions such as GAN-Transformer-based architectures and CNN-LSTM models. With an inference time of less than 0.15 seconds on an average CPU and a compact model size of just 16.7MB, SecurityBERT is ideally suited for real-life traffic analysis and a suitable choice for deployment on resource-constrained IoT devices.

摘要: 自然语言处理(NLP)领域目前正在经历一场革命性的变革，这是由基于开创性的Transformer架构的预先训练的大型语言模型(LLM)的强大功能推动的。随着网络安全攻击的频率和多样性不断上升，事件检测的重要性显著增加。物联网设备正在迅速扩展，因此越来越需要高效的技术来以高精度和最低的计算要求自主识别物联网网络中的基于网络的攻击。提出了一种新的物联网网络威胁检测体系结构SecurityBERT，它利用双向编码器Transformers表示(BERT)模型进行网络威胁检测。在SecurityBERT的训练过程中，我们引入了一种新的隐私保护编码技术，称为隐私保护定长编码(PPFLE)。通过将PPFLE与字节级字节对编码器(BBPE)令牌器相结合，我们有效地以结构化格式表示网络流量数据。我们的研究表明，SecurityBERT在网络威胁检测方面优于传统的机器学习(ML)和深度学习(DL)方法，如卷积神经网络(CNNS)或递归神经网络(RNN)。使用Edge-IIoTset网络安全数据集，我们的实验分析表明，SecurityBERT在识别14种不同的攻击类型方面获得了令人印象深刻的98.2%的总体准确率，超过了基于GaN-Transformer的体系结构和CNN-LSTM模型等混合解决方案所创造的记录。SecurityBERT在平均CPU上的推断时间不到0.15秒，紧凑型尺寸仅为16.7MB，非常适合于现实生活中的流量分析，也是部署在资源受限的物联网设备上的合适选择。



## **35. SALAD-Bench: A Hierarchical and Comprehensive Safety Benchmark for Large Language Models**

Salad-BENCH：一种适用于大型语言模型的分层综合安全基准 cs.CL

**SubmitDate**: 2024-02-08    [abs](http://arxiv.org/abs/2402.05044v2) [paper-pdf](http://arxiv.org/pdf/2402.05044v2)

**Authors**: Lijun Li, Bowen Dong, Ruohui Wang, Xuhao Hu, Wangmeng Zuo, Dahua Lin, Yu Qiao, Jing Shao

**Abstract**: In the rapidly evolving landscape of Large Language Models (LLMs), ensuring robust safety measures is paramount. To meet this crucial need, we propose \emph{SALAD-Bench}, a safety benchmark specifically designed for evaluating LLMs, attack, and defense methods. Distinguished by its breadth, SALAD-Bench transcends conventional benchmarks through its large scale, rich diversity, intricate taxonomy spanning three levels, and versatile functionalities.SALAD-Bench is crafted with a meticulous array of questions, from standard queries to complex ones enriched with attack, defense modifications and multiple-choice. To effectively manage the inherent complexity, we introduce an innovative evaluators: the LLM-based MD-Judge for QA pairs with a particular focus on attack-enhanced queries, ensuring a seamless, and reliable evaluation. Above components extend SALAD-Bench from standard LLM safety evaluation to both LLM attack and defense methods evaluation, ensuring the joint-purpose utility. Our extensive experiments shed light on the resilience of LLMs against emerging threats and the efficacy of contemporary defense tactics. Data and evaluator are released under https://github.com/OpenSafetyLab/SALAD-BENCH.

摘要: 在快速发展的大型语言模型(LLM)环境中，确保可靠的安全措施至关重要。为了满足这一关键需求，我们提出了一种专门为评估LLMS、攻击和防御方法而设计的安全基准。SALAD-BENCH以其广度、丰富的多样性、跨越三个层次的复杂分类和多功能而超越传统基准。SALAD-BENCH精心设计了一系列细致的问题，从标准查询到复杂的问题，丰富了攻击、防御修改和多项选择。为了有效地管理固有的复杂性，我们引入了一种创新的评估器：基于LLM的针对QA对的MD-裁判，特别关注攻击增强的查询，确保无缝和可靠的评估。上述组件将沙拉工作台从标准的LLM安全评估扩展到LLM攻防方法评估，确保了联合用途的实用性。我们广泛的实验揭示了低密度脂蛋白对新出现的威胁的弹性以及当代防御战术的有效性。数据和评估器在https://github.com/OpenSafetyLab/SALAD-BENCH.下发布



## **36. Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications**

基于修剪和低阶修正的安全对准脆性评估 cs.LG

22 pages, 9 figures. Project page is available at  https://boyiwei.com/alignment-attribution/

**SubmitDate**: 2024-02-07    [abs](http://arxiv.org/abs/2402.05162v1) [paper-pdf](http://arxiv.org/pdf/2402.05162v1)

**Authors**: Boyi Wei, Kaixuan Huang, Yangsibo Huang, Tinghao Xie, Xiangyu Qi, Mengzhou Xia, Prateek Mittal, Mengdi Wang, Peter Henderson

**Abstract**: Large language models (LLMs) show inherent brittleness in their safety mechanisms, as evidenced by their susceptibility to jailbreaking and even non-malicious fine-tuning. This study explores this brittleness of safety alignment by leveraging pruning and low-rank modifications. We develop methods to identify critical regions that are vital for safety guardrails, and that are disentangled from utility-relevant regions at both the neuron and rank levels. Surprisingly, the isolated regions we find are sparse, comprising about $3\%$ at the parameter level and $2.5\%$ at the rank level. Removing these regions compromises safety without significantly impacting utility, corroborating the inherent brittleness of the model's safety mechanisms. Moreover, we show that LLMs remain vulnerable to low-cost fine-tuning attacks even when modifications to the safety-critical regions are restricted. These findings underscore the urgent need for more robust safety strategies in LLMs.

摘要: 大型语言模型(LLM)在其安全机制中显示出固有的脆弱性，其对越狱甚至非恶意微调的敏感性就是明证。这项研究通过利用修剪和低等级修改来探索安全对齐的脆弱性。我们开发了识别关键区域的方法，这些区域对安全护栏至关重要，并且在神经元和等级水平上与公用事业相关的区域分离。令人惊讶的是，我们发现的孤立区域是稀疏的，在参数水平上约为$3$，在秩级水平上约为$2.5$。移除这些区域会在不显著影响实用性的情况下影响安全性，从而证实了该模型安全机制固有的脆性。此外，我们还表明，即使对安全关键区域的修改受到限制，LLM仍然容易受到低成本微调攻击。这些发现突显了在低成本管理中迫切需要更强有力的安全战略。



## **37. Defending Our Privacy With Backdoors**

用后门捍卫我们的隐私 cs.LG

18 pages, 11 figures

**SubmitDate**: 2024-02-07    [abs](http://arxiv.org/abs/2310.08320v3) [paper-pdf](http://arxiv.org/pdf/2310.08320v3)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Daniel Neider, Kristian Kersting

**Abstract**: The proliferation of large AI models trained on uncurated, often sensitive web-scraped data has raised significant privacy concerns. One of the concerns is that adversaries can extract information about the training data using privacy attacks. Unfortunately, the task of removing specific information from the models without sacrificing performance is not straightforward and has proven to be challenging. We propose a rather easy yet effective defense based on backdoor attacks to remove private information such as names and faces of individuals from vision-language models by fine-tuning them for only a few minutes instead of re-training them from scratch. Specifically, through strategic insertion of backdoors into text encoders, we align the embeddings of sensitive phrases with those of neutral terms-"a person" instead of the person's actual name. For image encoders, we map embeddings of individuals to be removed from the model to a universal, anonymous embedding. Our empirical results demonstrate the effectiveness of our backdoor-based defense on CLIP by assessing its performance using a specialized privacy attack for zero-shot classifiers. Our approach provides not only a new "dual-use" perspective on backdoor attacks, but also presents a promising avenue to enhance the privacy of individuals within models trained on uncurated web-scraped data.

摘要: 大型人工智能模型的激增引发了人们对隐私的严重担忧。这些模型针对未经管理的、往往是敏感的网络数据进行培训。其中一个令人担忧的问题是，攻击者可以使用隐私攻击来提取有关训练数据的信息。不幸的是，在不牺牲性能的情况下从模型中删除特定信息的任务并不简单，而且已被证明是具有挑战性的。我们提出了一种基于后门攻击的相当简单但有效的防御方法，通过对视觉语言模型进行几分钟的微调来删除个人的姓名和面孔等私人信息，而不是从头开始重新训练它们。具体地说，通过在文本编码器中策略性地插入后门，我们将敏感短语的嵌入与中性术语的嵌入对齐--“人”而不是人的实际姓名。对于图像编码器，我们将要从模型中移除的个人的嵌入映射到通用的匿名嵌入。我们的实验结果证明了我们的基于后门的防御在CLIP上的有效性，通过使用专门的针对零镜头分类器的隐私攻击来评估其性能。我们的方法不仅为后门攻击提供了一种新的“两用”视角，而且还提供了一种在未经管理的网络抓取数据的培训模型中增强个人隐私的有前景的途径。



## **38. Human-Readable Fingerprint for Large Language Models**

大型语言模型的人类可读指纹 cs.CL

**SubmitDate**: 2024-02-07    [abs](http://arxiv.org/abs/2312.04828v2) [paper-pdf](http://arxiv.org/pdf/2312.04828v2)

**Authors**: Boyi Zeng, Chenghu Zhou, Xinbing Wang, Zhouhan Lin

**Abstract**: Protecting the copyright of large language models (LLMs) has become crucial due to their resource-intensive training and accompanying carefully designed licenses. However, identifying the original base model of an LLM is challenging due to potential parameter alterations. In this study, we introduce a human-readable fingerprint for LLMs that uniquely identifies the base model without exposing model parameters or interfering with training. We first observe that the vector direction of LLM parameters remains stable after the model has converged during pretraining, showing negligible perturbations through subsequent training steps, including continued pretraining, supervised fine-tuning (SFT), and RLHF, which makes it a sufficient condition to identify the base model. The necessity is validated by continuing to train an LLM with an extra term to drive away the model parameters' direction and the model becomes damaged. However, this direction is vulnerable to simple attacks like dimension permutation or matrix rotation, which significantly change it without affecting performance. To address this, leveraging the Transformer structure, we systematically analyze potential attacks and define three invariant terms that identify an LLM's base model. We make these invariant terms human-readable by mapping them to a Gaussian vector using a convolutional encoder and then converting it into a natural image with StyleGAN2. Our method generates a dog image as an identity fingerprint for an LLM, where the dog's appearance strongly indicates the LLM's base model. The fingerprint provides intuitive information for qualitative discrimination, while the invariant terms can be employed for quantitative and precise verification. Experimental results across various LLMs demonstrate the effectiveness of our method.

摘要: 保护大型语言模型(LLM)的版权已变得至关重要，因为它们需要进行资源密集型培训，并附带精心设计的许可证。然而，由于潜在的参数变化，识别LLM的原始基础模型是具有挑战性的。在这项研究中，我们为LLMS引入了一种人类可读的指纹，它在不暴露模型参数或干扰训练的情况下唯一地识别基本模型。我们首先观察到，在预训练过程中，模型收敛后，LLM参数的向量方向保持稳定，通过后续的训练步骤，包括继续预训练、有监督微调(SFT)和RLHF，表现出可以忽略的扰动，这使得它成为识别基本模型的充分条件。通过继续训练一个带有额外项的LLM来驱离模型参数的方向，从而使模型受损，从而验证了这种必要性。然而，这个方向很容易受到维度置换或矩阵旋转等简单攻击，这些攻击会在不影响性能的情况下显著改变它。为了解决这个问题，利用Transformer结构，我们系统地分析了潜在的攻击，并定义了识别LLM基本模型的三个不变术语。我们使用卷积编码器将这些不变项映射到高斯向量，然后使用StyleGAN2将其转换为自然图像，从而使这些不变项变得可读。我们的方法生成了一幅狗图像作为LLM的身份指纹，其中狗的外表强烈地指示了LLM的基本模型。指纹为定性鉴别提供了直观的信息，而不变项可用于定量和精确的验证。在不同LLM上的实验结果证明了该方法的有效性。



## **39. HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal**

HarmBench：一个用于自动化红队和鲁棒拒绝的标准化评估框架 cs.LG

Website: https://www.harmbench.org

**SubmitDate**: 2024-02-06    [abs](http://arxiv.org/abs/2402.04249v1) [paper-pdf](http://arxiv.org/pdf/2402.04249v1)

**Authors**: Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, David Forsyth, Dan Hendrycks

**Abstract**: Automated red teaming holds substantial promise for uncovering and mitigating the risks associated with the malicious use of large language models (LLMs), yet the field lacks a standardized evaluation framework to rigorously assess new methods. To address this issue, we introduce HarmBench, a standardized evaluation framework for automated red teaming. We identify several desirable properties previously unaccounted for in red teaming evaluations and systematically design HarmBench to meet these criteria. Using HarmBench, we conduct a large-scale comparison of 18 red teaming methods and 33 target LLMs and defenses, yielding novel insights. We also introduce a highly efficient adversarial training method that greatly enhances LLM robustness across a wide range of attacks, demonstrating how HarmBench enables codevelopment of attacks and defenses. We open source HarmBench at https://github.com/centerforaisafety/HarmBench.

摘要: 自动化的红色团队在发现和减轻与恶意使用大型语言模型（LLM）相关的风险方面有很大的希望，但该领域缺乏标准化的评估框架来严格评估新方法。为了解决这个问题，我们引入了HarmBench，这是一个用于自动化红色团队的标准化评估框架。我们确定了几个理想的属性以前未考虑到在红色的团队评估和系统的设计HarmBench，以满足这些标准。使用HarmBench，我们对18种红色组队方法和33种目标LLM和防御进行了大规模比较，产生了新的见解。我们还介绍了一种高效的对抗性训练方法，该方法大大增强了LLM在各种攻击中的鲁棒性，展示了HarmBench如何实现攻击和防御的共同开发。我们在https://github.com/centerforaisafety/HarmBench上开源HarmBench。



## **40. SHIELD : An Evaluation Benchmark for Face Spoofing and Forgery Detection with Multimodal Large Language Models**

Shield：一种基于多通道大语言模型的人脸欺骗和伪装检测评估基准 cs.CV

**SubmitDate**: 2024-02-06    [abs](http://arxiv.org/abs/2402.04178v1) [paper-pdf](http://arxiv.org/pdf/2402.04178v1)

**Authors**: Yichen Shi, Yuhao Gao, Yingxin Lai, Hongyang Wang, Jun Feng, Lei He, Jun Wan, Changsheng Chen, Zitong Yu, Xiaochun Cao

**Abstract**: Multimodal large language models (MLLMs) have demonstrated remarkable problem-solving capabilities in various vision fields (e.g., generic object recognition and grounding) based on strong visual semantic representation and language reasoning ability. However, whether MLLMs are sensitive to subtle visual spoof/forged clues and how they perform in the domain of face attack detection (e.g., face spoofing and forgery detection) is still unexplored. In this paper, we introduce a new benchmark, namely SHIELD, to evaluate the ability of MLLMs on face spoofing and forgery detection. Specifically, we design true/false and multiple-choice questions to evaluate multimodal face data in these two face security tasks. For the face anti-spoofing task, we evaluate three different modalities (i.e., RGB, infrared, depth) under four types of presentation attacks (i.e., print attack, replay attack, rigid mask, paper mask). For the face forgery detection task, we evaluate GAN-based and diffusion-based data with both visual and acoustic modalities. Each question is subjected to both zero-shot and few-shot tests under standard and chain of thought (COT) settings. The results indicate that MLLMs hold substantial potential in the face security domain, offering advantages over traditional specific models in terms of interpretability, multimodal flexible reasoning, and joint face spoof and forgery detection. Additionally, we develop a novel Multi-Attribute Chain of Thought (MA-COT) paradigm for describing and judging various task-specific and task-irrelevant attributes of face images, which provides rich task-related knowledge for subtle spoof/forged clue mining. Extensive experiments in separate face anti-spoofing, separate face forgery detection, and joint detection tasks demonstrate the effectiveness of the proposed MA-COT. The project is available at https$:$//github.com/laiyingxin2/SHIELD

摘要: 多通道大语言模型基于强大的视觉语义表示和语言推理能力，在不同的视觉领域(如通用对象识别和基础)表现出显著的问题解决能力。然而，MLLM是否对细微的视觉欺骗/伪造线索敏感，以及它们在人脸攻击检测(例如，人脸欺骗和伪造检测)领域的表现仍未被探索。在本文中，我们引入了一个新的基准测试，即Shield，来评估MLLMS在人脸欺骗和伪造检测方面的能力。具体地说，我们设计了真假和选择题来评估这两个人脸安全任务中的多模式人脸数据。对于人脸反欺骗任务，我们在四种呈现攻击(即打印攻击、重放攻击、刚性掩模、纸掩模)下评估了三种不同的模式(即RGB、红外、深度)。对于人脸伪造检测任务，我们用视觉和听觉两种方式评估了基于GaN和基于扩散的数据。每个问题都要在标准和思维链(COT)设置下进行零分和少分测试。结果表明，MLLMS在人脸安全领域具有很大的潜力，在可解释性、多通道灵活推理、联合人脸欺骗和伪造检测等方面优于传统的特定模型。此外，我们还提出了一种新的多属性思维链(MA-COT)范式，用于描述和判断人脸图像中各种特定于任务和与任务无关的属性，为精细的欺骗/伪造线索挖掘提供了丰富的任务相关知识。在单独的人脸反欺骗、单独的人脸伪造检测和联合检测任务中的大量实验证明了所提出的MA-COT的有效性。该项目的网址为：https$：$/gihub.com/laiyingxin2/Shield



## **41. Partially Recentralization Softmax Loss for Vision-Language Models Robustness**

视觉语言模型稳健性的部分再中心化软最大损失 cs.CL

**SubmitDate**: 2024-02-06    [abs](http://arxiv.org/abs/2402.03627v1) [paper-pdf](http://arxiv.org/pdf/2402.03627v1)

**Authors**: Hao Wang, Xin Zhang, Jinzhe Jiang, Yaqian Zhao, Chen Li

**Abstract**: As Large Language Models make a breakthrough in natural language processing tasks (NLP), multimodal technique becomes extremely popular. However, it has been shown that multimodal NLP are vulnerable to adversarial attacks, where the outputs of a model can be dramatically changed by a perturbation to the input. While several defense techniques have been proposed both in computer vision and NLP models, the multimodal robustness of models have not been fully explored. In this paper, we study the adversarial robustness provided by modifying loss function of pre-trained multimodal models, by restricting top K softmax outputs. Based on the evaluation and scoring, our experiments show that after a fine-tuning, adversarial robustness of pre-trained models can be significantly improved, against popular attacks. Further research should be studying, such as output diversity, generalization and the robustness-performance trade-off of this kind of loss functions. Our code will be available after this paper is accepted

摘要: 随着大型语言模型在自然语言处理任务(NLP)方面的突破，多通道技术变得非常流行。然而，已有研究表明，多模式NLP很容易受到对抗性攻击，模型的输出可能会因输入的扰动而发生显著变化。虽然在计算机视觉和NLP模型中已经提出了几种防御技术，但模型的多通道稳健性还没有得到充分的研究。在本文中，我们研究了通过限制Top K Softmax输出来修改预先训练的多模式模型的损失函数所提供的对抗鲁棒性。在评估和评分的基础上，我们的实验表明，经过微调后，预先训练的模型对攻击的健壮性可以显著提高，对抗流行攻击。这类损失函数的输出分集、泛化以及稳健性与性能的权衡等问题还有待进一步研究。我们的代码将在这篇论文被接受后可用



## **42. Beyond Text: Improving LLM's Decision Making for Robot Navigation via Vocal Cues**

超越文本：通过声音提示改善LLM的机器人导航决策 cs.AI

20 pages, 8 figures

**SubmitDate**: 2024-02-05    [abs](http://arxiv.org/abs/2402.03494v1) [paper-pdf](http://arxiv.org/pdf/2402.03494v1)

**Authors**: Xingpeng Sun, Haoming Meng, Souradip Chakraborty, Amrit Singh Bedi, Aniket Bera

**Abstract**: This work highlights a critical shortcoming in text-based Large Language Models (LLMs) used for human-robot interaction, demonstrating that text alone as a conversation modality falls short in such applications. While LLMs excel in processing text in these human conversations, they struggle with the nuances of verbal instructions in scenarios like social navigation, where ambiguity and uncertainty can erode trust in robotic and other AI systems. We can address this shortcoming by moving beyond text and additionally focusing on the paralinguistic features of these audio responses. These features are the aspects of spoken communication that do not involve the literal wording (lexical content) but convey meaning and nuance through how something is said. We present "Beyond Text"; an approach that improves LLM decision-making by integrating audio transcription along with a subsection of these features, which focus on the affect and more relevant in human-robot conversations. This approach not only achieves a 70.26% winning rate, outperforming existing LLMs by 48.30%, but also enhances robustness against token manipulation adversarial attacks, highlighted by a 22.44% less decrease ratio than the text-only language model in winning rate. "Beyond Text" marks an advancement in social robot navigation and broader Human-Robot interactions, seamlessly integrating text-based guidance with human-audio-informed language models.

摘要: 这项工作突显了用于人-机器人交互的基于文本的大语言模型(LLMS)的一个关键缺陷，表明仅作为一种对话方式在此类应用中是不足的。虽然LLM在处理这些人类对话中的文本方面表现出色，但它们在社交导航等场景中难以处理语言指令的细微差别，在这些场景中，模棱两可和不确定性可能会侵蚀人们对机器人和其他人工智能系统的信任。我们可以通过超越文本并另外关注这些音频反应的副语言特征来解决这一缺点。这些特征是口语交际的方面，不涉及字面上的措辞(词汇内容)，但通过说话方式传达意义和细微差别。我们提出了“超越文本”；这是一种通过将音频转录与这些功能的一部分相结合来改进LLM决策的方法，这些功能侧重于影响，在人与机器人的对话中更相关。该方法不仅获得了70.26%的中签率，比现有的LLMS模型提高了48.30%，而且增强了对令牌操纵敌意攻击的健壮性，其中签率比纯文本语言模型降低了22.44%。“超越文本”标志着社交机器人导航和更广泛的人-机器人交互方面的进步，无缝地将基于文本的指导与人-音频信息语言模型相结合。



## **43. Weak-to-Strong Jailbreaking on Large Language Models**

大型语言模型上的从弱到强的越狱 cs.CL

**SubmitDate**: 2024-02-05    [abs](http://arxiv.org/abs/2401.17256v2) [paper-pdf](http://arxiv.org/pdf/2401.17256v2)

**Authors**: Xuandong Zhao, Xianjun Yang, Tianyu Pang, Chao Du, Lei Li, Yu-Xiang Wang, William Yang Wang

**Abstract**: Large language models (LLMs) are vulnerable to jailbreak attacks - resulting in harmful, unethical, or biased text generations. However, existing jailbreaking methods are computationally costly. In this paper, we propose the weak-to-strong jailbreaking attack, an efficient method to attack aligned LLMs to produce harmful text. Our key intuition is based on the observation that jailbroken and aligned models only differ in their initial decoding distributions. The weak-to-strong attack's key technical insight is using two smaller models (a safe and an unsafe one) to adversarially modify a significantly larger safe model's decoding probabilities. We evaluate the weak-to-strong attack on 5 diverse LLMs from 3 organizations. The results show our method can increase the misalignment rate to over 99% on two datasets with just one forward pass per example. Our study exposes an urgent safety issue that needs to be addressed when aligning LLMs. As an initial attempt, we propose a defense strategy to protect against such attacks, but creating more advanced defenses remains challenging. The code for replicating the method is available at https://github.com/XuandongZhao/weak-to-strong

摘要: 大型语言模型(LLM)容易受到越狱攻击--导致有害的、不道德的或有偏见的文本生成。然而，现有的越狱方法在计算上代价高昂。在本文中，我们提出了弱到强越狱攻击，这是一种攻击对齐的LLM产生有害文本的有效方法。我们的关键直觉是基于这样一种观察，即越狱模型和联合模型只是在初始解码分布上有所不同。弱到强攻击的关键技术洞察力是使用两个较小的模型(一个安全模型和一个不安全模型)来恶意修改显著较大的安全模型的解码概率。我们评估了对来自3个组织的5个不同的LLM的从弱到强的攻击。实验结果表明，在两个数据集上，每个样本只需一次前向遍历，我们的方法就可以将错位率提高到99%以上。我们的研究揭示了在调整LLM时需要解决的一个紧迫的安全问题。作为最初的尝试，我们提出了一种防御战略来防御此类攻击，但创建更先进的防御仍然具有挑战性。复制该方法的代码可在https://github.com/XuandongZhao/weak-to-strong上找到



## **44. Fundamental Limitations of Alignment in Large Language Models**

大型语言模型中对齐的基本限制 cs.CL

**SubmitDate**: 2024-02-05    [abs](http://arxiv.org/abs/2304.11082v5) [paper-pdf](http://arxiv.org/pdf/2304.11082v5)

**Authors**: Yotam Wolf, Noam Wies, Oshri Avnery, Yoav Levine, Amnon Shashua

**Abstract**: An important aspect in developing language models that interact with humans is aligning their behavior to be useful and unharmful for their human users. This is usually achieved by tuning the model in a way that enhances desired behaviors and inhibits undesired ones, a process referred to as alignment. In this paper, we propose a theoretical approach called Behavior Expectation Bounds (BEB) which allows us to formally investigate several inherent characteristics and limitations of alignment in large language models. Importantly, we prove that within the limits of this framework, for any behavior that has a finite probability of being exhibited by the model, there exist prompts that can trigger the model into outputting this behavior, with probability that increases with the length of the prompt. This implies that any alignment process that attenuates an undesired behavior but does not remove it altogether, is not safe against adversarial prompting attacks. Furthermore, our framework hints at the mechanism by which leading alignment approaches such as reinforcement learning from human feedback make the LLM prone to being prompted into the undesired behaviors. This theoretical result is being experimentally demonstrated in large scale by the so called contemporary "chatGPT jailbreaks", where adversarial users trick the LLM into breaking its alignment guardrails by triggering it into acting as a malicious persona. Our results expose fundamental limitations in alignment of LLMs and bring to the forefront the need to devise reliable mechanisms for ensuring AI safety.

摘要: 开发与人类交互的语言模型的一个重要方面是使他们的行为对人类用户有用而无害。这通常是通过调整模型来实现的，这种方式增强了期望的行为，抑制了不期望的行为，这一过程称为对齐。在本文中，我们提出了一种名为行为期望界限(BEB)的理论方法，它允许我们正式地研究大型语言模型中对齐的几个固有特征和限制。重要的是，我们证明了在这个框架的范围内，对于模型所表现出的任何有限概率的行为，存在可以触发模型输出该行为的提示，其概率随着提示的长度的增加而增加。这意味着，任何减弱不受欢迎的行为但不能完全消除它的对准过程，在对抗提示攻击时都是不安全的。此外，我们的框架暗示了一种机制，通过这种机制，领先的对齐方法，如来自人类反馈的强化学习，使得LLM容易被提示进入不希望看到的行为。这一理论结果正在由所谓的当代“聊天GPT越狱”大规模实验证明，在这种情况下，敌对用户通过触发LLM充当恶意角色来欺骗LLM打破其对齐护栏。我们的结果暴露了LLM对齐方面的根本限制，并将设计可靠的机制以确保人工智能安全的必要性放在了首位。



## **45. Conversation Reconstruction Attack Against GPT Models**

针对GPT模型的会话重构攻击 cs.CR

17 pages, 11 figures

**SubmitDate**: 2024-02-05    [abs](http://arxiv.org/abs/2402.02987v1) [paper-pdf](http://arxiv.org/pdf/2402.02987v1)

**Authors**: Junjie Chu, Zeyang Sha, Michael Backes, Yang Zhang

**Abstract**: In recent times, significant advancements have been made in the field of large language models (LLMs), represented by GPT series models. To optimize task execution, users often engage in multi-round conversations with GPT models hosted in cloud environments. These multi-round conversations, potentially replete with private information, require transmission and storage within the cloud. However, this operational paradigm introduces additional attack surfaces. In this paper, we first introduce a specific Conversation Reconstruction Attack targeting GPT models. Our introduced Conversation Reconstruction Attack is composed of two steps: hijacking a session and reconstructing the conversations. Subsequently, we offer an exhaustive evaluation of the privacy risks inherent in conversations when GPT models are subjected to the proposed attack. However, GPT-4 demonstrates certain robustness to the proposed attacks. We then introduce two advanced attacks aimed at better reconstructing previous conversations, specifically the UNR attack and the PBU attack. Our experimental findings indicate that the PBU attack yields substantial performance across all models, achieving semantic similarity scores exceeding 0.60, while the UNR attack is effective solely on GPT-3.5. Our results reveal the concern about privacy risks associated with conversations involving GPT models and aim to draw the community's attention to prevent the potential misuse of these models' remarkable capabilities. We will responsibly disclose our findings to the suppliers of related large language models.

摘要: 近年来，以GPT系列模型为代表的大型语言模型（LLM）领域取得了重大进展。为了优化任务执行，用户经常与托管在云环境中的GPT模型进行多轮对话。这些多轮对话可能充满了私人信息，需要在云中传输和存储。然而，这种操作模式引入了额外的攻击面。在本文中，我们首先介绍了一个特定的会话重建攻击目标的GPT模型。我们介绍的会话重建攻击由两个步骤组成：劫持会话和重建会话。随后，我们提供了一个详尽的评估，当GPT模型受到拟议的攻击时，在对话中固有的隐私风险。然而，GPT-4对所提出的攻击表现出一定的鲁棒性。然后，我们介绍了两种先进的攻击，旨在更好地重建以前的对话，特别是UNR攻击和PBU攻击。我们的实验结果表明，PBU攻击在所有模型中都产生了显著的性能，语义相似性得分超过0.60，而UNR攻击仅在GPT-3.5上有效。我们的研究结果揭示了与涉及GPT模型的对话相关的隐私风险，旨在引起社区的注意，以防止这些模型的显着能力的潜在滥用。我们将负责任地向相关大型语言模型的供应商披露我们的发现。



## **46. Adversarial Text Purification: A Large Language Model Approach for Defense**

对抗性文本净化：一种用于防御的大型语言模型方法 cs.CR

PAKDD 2024

**SubmitDate**: 2024-02-05    [abs](http://arxiv.org/abs/2402.06655v1) [paper-pdf](http://arxiv.org/pdf/2402.06655v1)

**Authors**: Raha Moraffah, Shubh Khandelwal, Amrita Bhattacharjee, Huan Liu

**Abstract**: Adversarial purification is a defense mechanism for safeguarding classifiers against adversarial attacks without knowing the type of attacks or training of the classifier. These techniques characterize and eliminate adversarial perturbations from the attacked inputs, aiming to restore purified samples that retain similarity to the initially attacked ones and are correctly classified by the classifier. Due to the inherent challenges associated with characterizing noise perturbations for discrete inputs, adversarial text purification has been relatively unexplored. In this paper, we investigate the effectiveness of adversarial purification methods in defending text classifiers. We propose a novel adversarial text purification that harnesses the generative capabilities of Large Language Models (LLMs) to purify adversarial text without the need to explicitly characterize the discrete noise perturbations. We utilize prompt engineering to exploit LLMs for recovering the purified examples for given adversarial examples such that they are semantically similar and correctly classified. Our proposed method demonstrates remarkable performance over various classifiers, improving their accuracy under the attack by over 65% on average.

摘要: 对抗性净化是一种防御机制，用于保护分类器免受对抗性攻击，而不需要知道攻击的类型或分类器的训练。这些技术表征并消除了来自被攻击输入的对抗性扰动，旨在恢复与最初被攻击的样本保持相似并且被分类器正确分类的纯化样本。由于与描述离散输入的噪声扰动相关的固有挑战，对抗性文本净化一直相对未被探索。本文研究了对抗性净化方法在防御文本分类器中的有效性。我们提出了一种新的对抗性文本净化方法，它利用大语言模型(LLMS)的生成能力来净化对抗性文本，而不需要显式地刻画离散噪声扰动。我们利用即时工程来利用LLMS来恢复给定对抗性实例的提纯实例，以使它们在语义上相似并且正确地分类。我们的方法在不同的分类器上表现出了显著的性能，在攻击下它们的准确率平均提高了65%以上。



## **47. PROSAC: Provably Safe Certification for Machine Learning Models under Adversarial Attacks**

PROSAC：对抗性攻击下机器学习模型的可证明安全认证 cs.LG

**SubmitDate**: 2024-02-04    [abs](http://arxiv.org/abs/2402.02629v1) [paper-pdf](http://arxiv.org/pdf/2402.02629v1)

**Authors**: Ziquan Liu, Zhuo Zhi, Ilija Bogunovic, Carsten Gerner-Beuerle, Miguel Rodrigues

**Abstract**: It is widely known that state-of-the-art machine learning models, including vision and language models, can be seriously compromised by adversarial perturbations. It is therefore increasingly relevant to develop capabilities to certify their performance in the presence of the most effective adversarial attacks. Our paper offers a new approach to certify the performance of machine learning models in the presence of adversarial attacks with population level risk guarantees. In particular, we introduce the notion of $(\alpha,\zeta)$ machine learning model safety. We propose a hypothesis testing procedure, based on the availability of a calibration set, to derive statistical guarantees providing that the probability of declaring that the adversarial (population) risk of a machine learning model is less than $\alpha$ (i.e. the model is safe), while the model is in fact unsafe (i.e. the model adversarial population risk is higher than $\alpha$), is less than $\zeta$. We also propose Bayesian optimization algorithms to determine efficiently whether a machine learning model is $(\alpha,\zeta)$-safe in the presence of an adversarial attack, along with statistical guarantees. We apply our framework to a range of machine learning models including various sizes of vision Transformer (ViT) and ResNet models impaired by a variety of adversarial attacks, such as AutoAttack, SquareAttack and natural evolution strategy attack, to illustrate the operation of our approach. Importantly, we show that ViT's are generally more robust to adversarial attacks than ResNets, and ViT-large is more robust than smaller models. Our approach goes beyond existing empirical adversarial risk-based certification guarantees. It formulates rigorous (and provable) performance guarantees that can be used to satisfy regulatory requirements mandating the use of state-of-the-art technical tools.

摘要: 众所周知，最先进的机器学习模型，包括视觉和语言模型，可能会受到对抗性扰动的严重影响。因此，越来越有必要发展能力，以证明它们在最有效的对抗性攻击下的表现。本文提供了一种新的方法来证明机器学习模型在种群水平风险保证的对抗性攻击下的性能。特别地，我们引入了$(\α，\Zeta)$机器学习模型安全性的概念。我们提出了一种假设检验程序，基于校准集的可用性来获得统计保证，假设宣布一个机器学习模型的对抗(总体)风险小于$\α$(即该模型是安全的)，而该模型实际上是不安全的(即该模型的对抗总体风险高于$\α$)的概率小于$\Zeta$。我们还提出了贝叶斯优化算法来有效地确定机器学习模型在存在对抗性攻击的情况下是否$(\α，\Zeta)$安全，并提供统计保证。我们将我们的框架应用于一系列机器学习模型，包括不同大小的视觉转换器(VIT)和被各种敌意攻击(如AutoAttack、SquareAttack和自然进化策略攻击)破坏的ResNet模型，以说明我们方法的操作。重要的是，我们证明了VIT通常比ResNet对对手攻击更健壮，而VIT-Large比更小的模型更健壮。我们的方法超越了现有的经验对抗性、基于风险的认证保证。它制定了严格的(和可证明的)性能保证，可用于满足要求使用最先进技术工具的监管要求。



## **48. Jailbreaking Attack against Multimodal Large Language Model**

针对多通道大语言模型的越狱攻击 cs.LG

**SubmitDate**: 2024-02-04    [abs](http://arxiv.org/abs/2402.02309v1) [paper-pdf](http://arxiv.org/pdf/2402.02309v1)

**Authors**: Zhenxing Niu, Haodong Ren, Xinbo Gao, Gang Hua, Rong Jin

**Abstract**: This paper focuses on jailbreaking attacks against multi-modal large language models (MLLMs), seeking to elicit MLLMs to generate objectionable responses to harmful user queries. A maximum likelihood-based algorithm is proposed to find an \emph{image Jailbreaking Prompt} (imgJP), enabling jailbreaks against MLLMs across multiple unseen prompts and images (i.e., data-universal property). Our approach exhibits strong model-transferability, as the generated imgJP can be transferred to jailbreak various models, including MiniGPT-v2, LLaVA, InstructBLIP, and mPLUG-Owl2, in a black-box manner. Moreover, we reveal a connection between MLLM-jailbreaks and LLM-jailbreaks. As a result, we introduce a construction-based method to harness our approach for LLM-jailbreaks, demonstrating greater efficiency than current state-of-the-art methods. The code is available here. \textbf{Warning: some content generated by language models may be offensive to some readers.}

摘要: 针对多模式大型语言模型(MLLMS)的越狱攻击，试图诱导MLLMS对有害的用户查询产生不良响应。提出了一种基于最大似然的算法来寻找图像越狱提示(ImgJP)，使越狱能够跨越多个不可见的提示和图像(即，数据普适的性质)。我们的方法具有很强的模型可移植性，因为生成的imgJP可以黑盒方式传输到各种模型，包括MiniGPT-v2、LLaVA、InstructBLIP和mPLUG-Owl2。此外，我们揭示了MLLM越狱和LLM越狱之间的联系。因此，我们引入了一种基于施工的方法来利用我们的方法来处理LLM越狱，展示了比当前最先进的方法更高的效率。代码可以在这里找到。\extbf{警告：语言模型生成的某些内容可能会冒犯某些读者。}



## **49. Safety Fine-Tuning at (Almost) No Cost: A Baseline for Vision Large Language Models**

(几乎)免费的安全微调：Vision大型语言模型的基准 cs.LG

**SubmitDate**: 2024-02-03    [abs](http://arxiv.org/abs/2402.02207v1) [paper-pdf](http://arxiv.org/pdf/2402.02207v1)

**Authors**: Yongshuo Zong, Ondrej Bohdal, Tingyang Yu, Yongxin Yang, Timothy Hospedales

**Abstract**: Current vision large language models (VLLMs) exhibit remarkable capabilities yet are prone to generate harmful content and are vulnerable to even the simplest jailbreaking attacks. Our initial analysis finds that this is due to the presence of harmful data during vision-language instruction fine-tuning, and that VLLM fine-tuning can cause forgetting of safety alignment previously learned by the underpinning LLM. To address this issue, we first curate a vision-language safe instruction-following dataset VLGuard covering various harmful categories. Our experiments demonstrate that integrating this dataset into standard vision-language fine-tuning or utilizing it for post-hoc fine-tuning effectively safety aligns VLLMs. This alignment is achieved with minimal impact on, or even enhancement of, the models' helpfulness. The versatility of our safety fine-tuning dataset makes it a valuable resource for safety-testing existing VLLMs, training new models or safeguarding pre-trained VLLMs. Empirical results demonstrate that fine-tuned VLLMs effectively reject unsafe instructions and substantially reduce the success rates of several black-box adversarial attacks, which approach zero in many cases. The code and dataset are available at https://github.com/ys-zong/VLGuard.

摘要: 目前的VISION大型语言模型(VLLM)显示出非凡的能力，但很容易产生有害内容，甚至容易受到最简单的越狱攻击。我们的初步分析发现，这是由于视觉语言教学微调过程中存在有害数据，而VLLM微调可能会导致忘记支持LLM之前学习的安全对齐。为了解决这个问题，我们首先策划了一个视觉-语言安全的指令遵循数据集VLGuard，涵盖了各种有害类别。我们的实验表明，将该数据集集成到标准视觉语言微调中或将其用于后自组织微调，可以有效地安全地对齐VLLM。这种对齐是在对模型的帮助最小的影响甚至是增强的情况下实现的。我们的安全微调数据集的多功能性使其成为安全测试现有VLLM、培训新模型或保护预先培训的VLLM的宝贵资源。实验结果表明，微调的VLLM有效地拒绝了不安全的指令，并显著降低了几种黑盒对抗攻击的成功率，这些攻击在许多情况下接近于零。代码和数据集可在https://github.com/ys-zong/VLGuard.上获得



## **50. Data Poisoning for In-context Learning**

情景学习中的数据中毒 cs.CR

**SubmitDate**: 2024-02-03    [abs](http://arxiv.org/abs/2402.02160v1) [paper-pdf](http://arxiv.org/pdf/2402.02160v1)

**Authors**: Pengfei He, Han Xu, Yue Xing, Hui Liu, Makoto Yamada, Jiliang Tang

**Abstract**: In the domain of large language models (LLMs), in-context learning (ICL) has been recognized for its innovative ability to adapt to new tasks, relying on examples rather than retraining or fine-tuning. This paper delves into the critical issue of ICL's susceptibility to data poisoning attacks, an area not yet fully explored. We wonder whether ICL is vulnerable, with adversaries capable of manipulating example data to degrade model performance. To address this, we introduce ICLPoison, a specialized attacking framework conceived to exploit the learning mechanisms of ICL. Our approach uniquely employs discrete text perturbations to strategically influence the hidden states of LLMs during the ICL process. We outline three representative strategies to implement attacks under our framework, each rigorously evaluated across a variety of models and tasks. Our comprehensive tests, including trials on the sophisticated GPT-4 model, demonstrate that ICL's performance is significantly compromised under our framework. These revelations indicate an urgent need for enhanced defense mechanisms to safeguard the integrity and reliability of LLMs in applications relying on in-context learning.

摘要: 在大型语言模型(LLM)领域，情境学习(ICL)因其适应新任务的创新能力而被公认，它依赖于例子而不是再培训或微调。本文深入研究了ICL对数据中毒攻击的易感性这一关键问题，这是一个尚未完全探索的领域。我们想知道ICL是否易受攻击，因为对手能够操纵示例数据来降低模型性能。为了解决这个问题，我们引入了ICLPoison，这是一个专门的攻击框架，旨在利用ICL的学习机制。我们的方法独特地使用离散文本扰动来战略性地影响ICL过程中LLM的隐藏状态。我们概述了在我们的框架下实施攻击的三种具有代表性的战略，每种战略都在各种模型和任务中进行了严格的评估。我们的全面测试，包括对复杂的GPT-4模型的试验，表明ICL的性能在我们的框架下受到了严重影响。这些发现表明，迫切需要增强防御机制，以保障依赖于情景学习的应用程序中LLMS的完整性和可靠性。



