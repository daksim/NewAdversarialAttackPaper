# Latest Large Language Model Attack Papers
**update at 2024-05-13 15:26:50**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Explaining Arguments' Strength: Unveiling the Role of Attacks and Supports (Technical Report)**

解释争论的力量：揭示攻击和支持的作用（技术报告） cs.AI

This paper has been accepted at IJCAI 2024 (the 33rd International  Joint Conference on Artificial Intelligence)

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2404.14304v2) [paper-pdf](http://arxiv.org/pdf/2404.14304v2)

**Authors**: Xiang Yin, Potyka Nico, Francesca Toni

**Abstract**: Quantitatively explaining the strength of arguments under gradual semantics has recently received increasing attention. Specifically, several works in the literature provide quantitative explanations by computing the attribution scores of arguments. These works disregard the importance of attacks and supports, even though they play an essential role when explaining arguments' strength. In this paper, we propose a novel theory of Relation Attribution Explanations (RAEs), adapting Shapley values from game theory to offer fine-grained insights into the role of attacks and supports in quantitative bipolar argumentation towards obtaining the arguments' strength. We show that RAEs satisfy several desirable properties. We also propose a probabilistic algorithm to approximate RAEs efficiently. Finally, we show the application value of RAEs in fraud detection and large language models case studies.

摘要: 在渐进语义下定量解释论点的强度最近受到越来越多的关注。具体来说，文献中的几部作品通过计算论点的归因分数来提供量化解释。这些作品忽视了攻击和支持的重要性，尽管它们在解释论点的强度时发挥着至关重要的作用。在本文中，我们提出了一种新的关系归因解释（RAEs）理论，改编了博弈论中的沙普利价值观，以提供对攻击作用的细粒度见解，并支持量化两极论证，以获得论点的强度。我们表明RAE满足几个理想的性质。我们还提出了一种有效逼近RAE的概率算法。最后，我们展示了RAE在欺诈检测和大型语言模型案例研究中的应用价值。



## **2. Risks of Practicing Large Language Models in Smart Grid: Threat Modeling and Validation**

在智能电网中实践大型语言模型的风险：威胁建模和验证 cs.CR

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06237v1) [paper-pdf](http://arxiv.org/pdf/2405.06237v1)

**Authors**: Jiangnan Li, Yingyuan Yang, Jinyuan Sun

**Abstract**: Large Language Model (LLM) is a significant breakthrough in artificial intelligence (AI) and holds considerable potential for application within smart grids. However, as demonstrated in previous literature, AI technologies are susceptible to various types of attacks. It is crucial to investigate and evaluate the risks associated with LLMs before deploying them in critical infrastructure like smart grids. In this paper, we systematically evaluate the vulnerabilities of LLMs and identify two major types of attacks relevant to smart grid LLM applications, along with presenting the corresponding threat models. We then validate these attacks using popular LLMs, utilizing real smart grid data. Our validation demonstrates that attackers are capable of injecting bad data and retrieving domain knowledge from LLMs employed in smart grid scenarios.

摘要: 大型语言模型（LLM）是人工智能（AI）领域的重大突破，在智能电网中具有巨大的应用潜力。然而，正如之前的文献所证明的那样，人工智能技术容易受到各种类型的攻击。在将LLM部署在智能电网等关键基础设施中之前，调查和评估与LLM相关的风险至关重要。在本文中，我们系统地评估了LLM的漏洞，识别了与智能电网LLM应用相关的两种主要攻击类型，并提出了相应的威胁模型。然后，我们使用流行的LLM并利用真实的智能电网数据来验证这些攻击。我们的验证表明，攻击者能够注入不良数据并从智能电网场景中使用的LLM中检索领域知识。



## **3. Muting Whisper: A Universal Acoustic Adversarial Attack on Speech Foundation Models**

静音低语：对语音基础模型的通用声学对抗攻击 cs.CL

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.06134v1) [paper-pdf](http://arxiv.org/pdf/2405.06134v1)

**Authors**: Vyas Raina, Rao Ma, Charles McGhee, Kate Knill, Mark Gales

**Abstract**: Recent developments in large speech foundation models like Whisper have led to their widespread use in many automatic speech recognition (ASR) applications. These systems incorporate `special tokens' in their vocabulary, such as $\texttt{<endoftext>}$, to guide their language generation process. However, we demonstrate that these tokens can be exploited by adversarial attacks to manipulate the model's behavior. We propose a simple yet effective method to learn a universal acoustic realization of Whisper's $\texttt{<endoftext>}$ token, which, when prepended to any speech signal, encourages the model to ignore the speech and only transcribe the special token, effectively `muting' the model. Our experiments demonstrate that the same, universal 0.64-second adversarial audio segment can successfully mute a target Whisper ASR model for over 97\% of speech samples. Moreover, we find that this universal adversarial audio segment often transfers to new datasets and tasks. Overall this work demonstrates the vulnerability of Whisper models to `muting' adversarial attacks, where such attacks can pose both risks and potential benefits in real-world settings: for example the attack can be used to bypass speech moderation systems, or conversely the attack can also be used to protect private speech data.

摘要: 像Whisper这样的大型语音基础模型的最新发展导致它们在许多自动语音识别(ASR)应用中被广泛使用。这些系统在它们的词汇表中加入了“特殊记号”，如$\exttt{<endoftext>}$，以指导它们的语言生成过程。然而，我们证明了这些令牌可以被敌意攻击利用来操纵模型的行为。我们提出了一种简单而有效的方法来学习Whisper的$\exttt{<endoftext>}$标记的通用声学实现，当预先添加到任何语音信号时，鼓励模型忽略语音，只转录特殊的标记，从而有效地抑制了模型。我们的实验表明，相同的、通用的0.64秒的对抗性音频片段可以成功地使目标Whisper ASR模型在97%以上的语音样本上静音。此外，我们发现这种普遍的对抗性音频片段经常转移到新的数据集和任务。总体而言，这项工作证明了Whisper模型对“静音”对手攻击的脆弱性，在现实世界中，这种攻击既可以带来风险，也可以带来潜在的好处：例如，攻击可以用来绕过语音调节系统，或者反过来，攻击也可以用来保护私人语音数据。



## **4. Trustworthy AI-Generative Content in Intelligent 6G Network: Adversarial, Privacy, and Fairness**

智能6G网络中值得信赖的人工智能生成内容：对抗性、隐私性和公平性 cs.CR

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.05930v1) [paper-pdf](http://arxiv.org/pdf/2405.05930v1)

**Authors**: Siyuan Li, Xi Lin, Yaju Liu, Jianhua Li

**Abstract**: AI-generated content (AIGC) models, represented by large language models (LLM), have brought revolutionary changes to the content generation fields. The high-speed and extensive 6G technology is an ideal platform for providing powerful AIGC mobile service applications, while future 6G mobile networks also need to support intelligent and personalized mobile generation services. However, the significant ethical and security issues of current AIGC models, such as adversarial attacks, privacy, and fairness, greatly affect the credibility of 6G intelligent networks, especially in ensuring secure, private, and fair AIGC applications. In this paper, we propose TrustGAIN, a novel paradigm for trustworthy AIGC in 6G networks, to ensure trustworthy large-scale AIGC services in future 6G networks. We first discuss the adversarial attacks and privacy threats faced by AIGC systems in 6G networks, as well as the corresponding protection issues. Subsequently, we emphasize the importance of ensuring the unbiasedness and fairness of the mobile generative service in future intelligent networks. In particular, we conduct a use case to demonstrate that TrustGAIN can effectively guide the resistance against malicious or generated false information. We believe that TrustGAIN is a necessary paradigm for intelligent and trustworthy 6G networks to support AIGC services, ensuring the security, privacy, and fairness of AIGC network services.

摘要: 以大语言模型(LLM)为代表的AI-Generated Content(AIGC)模型给内容生成领域带来了革命性的变化。高速和广泛的6G技术是提供强大的AIGC移动业务应用的理想平台，而未来的6G移动网络也需要支持智能化和个性化的移动生成服务。然而，当前AIGC模型存在的重大伦理和安全问题，如对抗性攻击、隐私和公平性，极大地影响了6G智能网络的可信度，特别是在确保安全、私有和公平的AIGC应用方面。本文提出了一种新的6G网络可信AIGC模型TrustGAIN，以保证未来6G网络中可信赖的大规模AIGC服务。我们首先讨论了AIGC系统在6G网络中面临的敌意攻击和隐私威胁，以及相应的保护问题。随后，我们强调了在未来的智能网中确保移动生成业务的无偏性和公平性的重要性。特别是，我们进行了一个用例来证明TrustGAIN可以有效地指导对恶意或生成的虚假信息的抵抗。我们认为，TrustGAIN是智能可信6G网络支持AIGC服务的必备范式，确保AIGC网络服务的安全性、私密性和公平性。



## **5. LLMPot: Automated LLM-based Industrial Protocol and Physical Process Emulation for ICS Honeypots**

LLMPot：用于ICS蜜罐的自动化基于LLM的工业协议和物理流程仿真 cs.CR

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.05999v1) [paper-pdf](http://arxiv.org/pdf/2405.05999v1)

**Authors**: Christoforos Vasilatos, Dunia J. Mahboobeh, Hithem Lamri, Manaar Alam, Michail Maniatakos

**Abstract**: Industrial Control Systems (ICS) are extensively used in critical infrastructures ensuring efficient, reliable, and continuous operations. However, their increasing connectivity and addition of advanced features make them vulnerable to cyber threats, potentially leading to severe disruptions in essential services. In this context, honeypots play a vital role by acting as decoy targets within ICS networks, or on the Internet, helping to detect, log, analyze, and develop mitigations for ICS-specific cyber threats. Deploying ICS honeypots, however, is challenging due to the necessity of accurately replicating industrial protocols and device characteristics, a crucial requirement for effectively mimicking the unique operational behavior of different industrial systems. Moreover, this challenge is compounded by the significant manual effort required in also mimicking the control logic the PLC would execute, in order to capture attacker traffic aiming to disrupt critical infrastructure operations. In this paper, we propose LLMPot, a novel approach for designing honeypots in ICS networks harnessing the potency of Large Language Models (LLMs). LLMPot aims to automate and optimize the creation of realistic honeypots with vendor-agnostic configurations, and for any control logic, aiming to eliminate the manual effort and specialized knowledge traditionally required in this domain. We conducted extensive experiments focusing on a wide array of parameters, demonstrating that our LLM-based approach can effectively create honeypot devices implementing different industrial protocols and diverse control logic.

摘要: 工业控制系统(ICS)广泛应用于关键基础设施中，以确保高效、可靠和连续的运行。然而，它们日益增长的连通性和先进功能的增加使它们容易受到网络威胁，可能导致基本服务的严重中断。在这种情况下，蜜罐扮演着至关重要的角色，在ICS网络中或在互联网上充当诱骗目标，帮助检测、记录、分析和缓解ICS特定的网络威胁。然而，部署ICS蜜罐是具有挑战性的，因为必须准确复制工业协议和设备特性，这是有效模拟不同工业系统独特操作行为的关键要求。此外，为了捕获旨在扰乱关键基础设施操作的攻击者流量，还需要大量手动工作来模拟PLC将执行的控制逻辑，从而使这一挑战变得更加复杂。在本文中，我们提出了LLMPot，这是一种利用大语言模型的有效性来设计ICS网络中的蜜罐的新方法。LLMPot旨在自动化和优化创建具有供应商无关配置的逼真蜜罐，以及任何控制逻辑，旨在消除该领域传统上所需的手动工作和专业知识。我们针对广泛的参数进行了广泛的实验，证明了我们基于LLM的方法可以有效地创建执行不同工业协议和不同控制逻辑的蜜罐设备。



## **6. Chain of Attack: a Semantic-Driven Contextual Multi-Turn attacker for LLM**

攻击链：LLM的语义驱动上下文多回合攻击者 cs.CL

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.05610v1) [paper-pdf](http://arxiv.org/pdf/2405.05610v1)

**Authors**: Xikang Yang, Xuehai Tang, Songlin Hu, Jizhong Han

**Abstract**: Large language models (LLMs) have achieved remarkable performance in various natural language processing tasks, especially in dialogue systems. However, LLM may also pose security and moral threats, especially in multi round conversations where large models are more easily guided by contextual content, resulting in harmful or biased responses. In this paper, we present a novel method to attack LLMs in multi-turn dialogues, called CoA (Chain of Attack). CoA is a semantic-driven contextual multi-turn attack method that adaptively adjusts the attack policy through contextual feedback and semantic relevance during multi-turn of dialogue with a large model, resulting in the model producing unreasonable or harmful content. We evaluate CoA on different LLMs and datasets, and show that it can effectively expose the vulnerabilities of LLMs, and outperform existing attack methods. Our work provides a new perspective and tool for attacking and defending LLMs, and contributes to the security and ethical assessment of dialogue systems.

摘要: 大语言模型在各种自然语言处理任务中取得了显著的性能，尤其是在对话系统中。然而，LLM也可能构成安全和道德威胁，特别是在多轮对话中，大型模型更容易受到上下文内容的指导，导致有害或有偏见的反应。本文提出了一种新的攻击多话轮对话中的LLMS的方法，称为攻击链。CoA是一种语义驱动的上下文多轮攻击方法，在大模型的多轮对话中，通过上下文反馈和语义相关性自适应调整攻击策略，导致模型产生不合理或有害的内容。我们在不同的LLM和数据集上对CoA进行了评估，结果表明它可以有效地暴露LLMS的漏洞，并优于现有的攻击方法。我们的工作为攻击和防御LLMS提供了一个新的视角和工具，并有助于对话系统的安全和伦理评估。



## **7. Large Language Models for Cyber Security: A Systematic Literature Review**

网络安全的大型语言模型：系统性文献综述 cs.CR

46 pages,6 figures

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.04760v2) [paper-pdf](http://arxiv.org/pdf/2405.04760v2)

**Authors**: HanXiang Xu, ShenAo Wang, NingKe Li, KaiLong Wang, YanJie Zhao, Kai Chen, Ting Yu, Yang Liu, HaoYu Wang

**Abstract**: The rapid advancement of Large Language Models (LLMs) has opened up new opportunities for leveraging artificial intelligence in various domains, including cybersecurity. As the volume and sophistication of cyber threats continue to grow, there is an increasing need for intelligent systems that can automatically detect vulnerabilities, analyze malware, and respond to attacks. In this survey, we conduct a comprehensive review of the literature on the application of LLMs in cybersecurity (LLM4Security). By comprehensively collecting over 30K relevant papers and systematically analyzing 127 papers from top security and software engineering venues, we aim to provide a holistic view of how LLMs are being used to solve diverse problems across the cybersecurity domain. Through our analysis, we identify several key findings. First, we observe that LLMs are being applied to a wide range of cybersecurity tasks, including vulnerability detection, malware analysis, network intrusion detection, and phishing detection. Second, we find that the datasets used for training and evaluating LLMs in these tasks are often limited in size and diversity, highlighting the need for more comprehensive and representative datasets. Third, we identify several promising techniques for adapting LLMs to specific cybersecurity domains, such as fine-tuning, transfer learning, and domain-specific pre-training. Finally, we discuss the main challenges and opportunities for future research in LLM4Security, including the need for more interpretable and explainable models, the importance of addressing data privacy and security concerns, and the potential for leveraging LLMs for proactive defense and threat hunting. Overall, our survey provides a comprehensive overview of the current state-of-the-art in LLM4Security and identifies several promising directions for future research.

摘要: 大型语言模型(LLM)的快速发展为在包括网络安全在内的各个领域利用人工智能开辟了新的机会。随着网络威胁的数量和复杂性不断增长，对能够自动检测漏洞、分析恶意软件和响应攻击的智能系统的需求越来越大。在本次调查中，我们对LLMS在网络安全中的应用(LLM4Security)的文献进行了全面的回顾。通过全面收集30K多篇相关论文并系统分析来自顶级安全和软件工程场所的127篇论文，我们的目标是提供一个全面的视角，了解LLM是如何被用来解决网络安全领域的各种问题的。通过我们的分析，我们确定了几个关键发现。首先，我们观察到LLMS正被应用于广泛的网络安全任务，包括漏洞检测、恶意软件分析、网络入侵检测和网络钓鱼检测。其次，我们发现，在这些任务中用于训练和评估土地管理的数据集在大小和多样性方面往往有限，这突显了需要更全面和更具代表性的数据集。第三，我们确定了几种有前景的技术来使LLMS适应特定的网络安全领域，例如微调、迁移学习和特定领域的预训练。最后，我们讨论了LLM4Security未来研究的主要挑战和机遇，包括需要更多可解释和可解释的模型，解决数据隐私和安全问题的重要性，以及利用LLM进行主动防御和威胁追踪的潜力。总体而言，我们的调查提供了对LLM4Security当前最先进技术的全面概述，并确定了未来研究的几个有希望的方向。



## **8. Special Characters Attack: Toward Scalable Training Data Extraction From Large Language Models**

特殊字符攻击：从大型语言模型中提取可扩展的训练数据 cs.CR

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.05990v1) [paper-pdf](http://arxiv.org/pdf/2405.05990v1)

**Authors**: Yang Bai, Ge Pei, Jindong Gu, Yong Yang, Xingjun Ma

**Abstract**: Large language models (LLMs) have achieved remarkable performance on a wide range of tasks. However, recent studies have shown that LLMs can memorize training data and simple repeated tokens can trick the model to leak the data. In this paper, we take a step further and show that certain special characters or their combinations with English letters are stronger memory triggers, leading to more severe data leakage. The intuition is that, since LLMs are trained with massive data that contains a substantial amount of special characters (e.g. structural symbols {, } of JSON files, and @, # in emails and online posts), the model may memorize the co-occurrence between these special characters and the raw texts. This motivates us to propose a simple but effective Special Characters Attack (SCA) to induce training data leakage. Our experiments verify the high effectiveness of SCA against state-of-the-art LLMs: they can leak diverse training data, such as code corpus, web pages, and personally identifiable information, and sometimes generate non-stop outputs as a byproduct. We further show that the composition of the training data corpus can be revealed by inspecting the leaked data -- one crucial piece of information for pre-training high-performance LLMs. Our work can help understand the sensitivity of LLMs to special characters and identify potential areas for improvement.

摘要: 大型语言模型(LLM)在广泛的任务中取得了显著的性能。然而，最近的研究表明，LLMS可以记忆训练数据，简单的重复令牌可以诱使模型泄漏数据。在这篇文章中，我们进一步表明，某些特殊字符或它们与英语字母的组合是更强的记忆触发，导致更严重的数据泄漏。直观的是，由于LLM是用包含大量特殊字符(例如JSON文件的结构符号{，}，以及电子邮件和在线帖子中的@，#)的海量数据训练的，因此该模型可能会记住这些特殊字符与原始文本之间的共现。这促使我们提出了一种简单而有效的特殊字符攻击(SCA)来诱导训练数据泄漏。我们的实验验证了SCA相对于最先进的LLMS的高效性：它们可以泄露各种训练数据，如代码语料库、网页和个人身份信息，有时还会生成不间断的输出作为副产品。我们进一步表明，通过检查泄漏的数据可以揭示训练数据集的组成--这是训练前高性能LLMS的关键信息之一。我们的工作有助于理解LLMS对特殊字符的敏感度，并确定潜在的改进领域。



## **9. Locally Differentially Private In-Context Learning**

本地差异化私人背景学习 cs.CR

This paper was published at LREC-Coling 2024

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.04032v2) [paper-pdf](http://arxiv.org/pdf/2405.04032v2)

**Authors**: Chunyan Zheng, Keke Sun, Wenhao Zhao, Haibo Zhou, Lixin Jiang, Shaoyang Song, Chunlai Zhou

**Abstract**: Large pretrained language models (LLMs) have shown surprising In-Context Learning (ICL) ability. An important application in deploying large language models is to augment LLMs with a private database for some specific task. The main problem with this promising commercial use is that LLMs have been shown to memorize their training data and their prompt data are vulnerable to membership inference attacks (MIA) and prompt leaking attacks. In order to deal with this problem, we treat LLMs as untrusted in privacy and propose a locally differentially private framework of in-context learning(LDP-ICL) in the settings where labels are sensitive. Considering the mechanisms of in-context learning in Transformers by gradient descent, we provide an analysis of the trade-off between privacy and utility in such LDP-ICL for classification. Moreover, we apply LDP-ICL to the discrete distribution estimation problem. In the end, we perform several experiments to demonstrate our analysis results.

摘要: 大型预训练语言模型（LLM）表现出令人惊讶的上下文学习（ICL）能力。部署大型语言模型的一个重要应用是通过用于某些特定任务的专用数据库来增强LLM。这种有前途的商业用途的主要问题是，LLM已被证明可以记住它们的训练数据，而它们的提示数据很容易受到成员推断攻击（MIA）和提示泄露攻击。为了解决这个问题，我们将LLM视为隐私不受信任，并在标签敏感的环境中提出了一种本地差异隐私的上下文学习框架（LDP-ICL）。考虑到变形金刚中通过梯度下降进行的上下文学习机制，我们分析了此类LDP-ICL分类中隐私和实用性之间的权衡。此外，我们将LDP-ICL应用于离散分布估计问题。最后，我们进行了几个实验来证明我们的分析结果。



## **10. Air Gap: Protecting Privacy-Conscious Conversational Agents**

空气间隙：保护有隐私意识的对话代理人 cs.CR

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.05175v1) [paper-pdf](http://arxiv.org/pdf/2405.05175v1)

**Authors**: Eugene Bagdasaryan, Ren Yi, Sahra Ghalebikesabi, Peter Kairouz, Marco Gruteser, Sewoong Oh, Borja Balle, Daniel Ramage

**Abstract**: The growing use of large language model (LLM)-based conversational agents to manage sensitive user data raises significant privacy concerns. While these agents excel at understanding and acting on context, this capability can be exploited by malicious actors. We introduce a novel threat model where adversarial third-party apps manipulate the context of interaction to trick LLM-based agents into revealing private information not relevant to the task at hand.   Grounded in the framework of contextual integrity, we introduce AirGapAgent, a privacy-conscious agent designed to prevent unintended data leakage by restricting the agent's access to only the data necessary for a specific task. Extensive experiments using Gemini, GPT, and Mistral models as agents validate our approach's effectiveness in mitigating this form of context hijacking while maintaining core agent functionality. For example, we show that a single-query context hijacking attack on a Gemini Ultra agent reduces its ability to protect user data from 94% to 45%, while an AirGapAgent achieves 97% protection, rendering the same attack ineffective.

摘要: 越来越多地使用基于大型语言模型(LLM)的会话代理来管理敏感用户数据，这引发了严重的隐私问题。虽然这些代理擅长理解上下文并根据上下文执行操作，但这种能力可能会被恶意行为者利用。我们引入了一种新的威胁模型，在该模型中，敌意的第三方应用程序操纵交互的上下文，以欺骗基于LLM的代理泄露与手头任务无关的私人信息。基于上下文完整性的框架，我们引入了AirGapAgent，这是一个具有隐私意识的代理，旨在通过限制代理仅访问特定任务所需的数据来防止意外的数据泄露。使用Gemini、GPT和Mistral模型作为代理的大量实验验证了我们的方法在保持核心代理功能的同时缓解这种形式的上下文劫持的有效性。例如，我们表明，对Gemini Ultra代理的单查询上下文劫持攻击将其保护用户数据的能力从94%降低到45%，而AirGapAgent实现了97%的保护，使得相同的攻击无效。



## **11. Critical Infrastructure Protection: Generative AI, Challenges, and Opportunities**

关键基础设施保护：生成性人工智能、挑战和机遇 cs.CR

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.04874v1) [paper-pdf](http://arxiv.org/pdf/2405.04874v1)

**Authors**: Yagmur Yigit, Mohamed Amine Ferrag, Iqbal H. Sarker, Leandros A. Maglaras, Christos Chrysoulas, Naghmeh Moradpoor, Helge Janicke

**Abstract**: Critical National Infrastructure (CNI) encompasses a nation's essential assets that are fundamental to the operation of society and the economy, ensuring the provision of vital utilities such as energy, water, transportation, and communication. Nevertheless, growing cybersecurity threats targeting these infrastructures can potentially interfere with operations and seriously risk national security and public safety. In this paper, we examine the intricate issues raised by cybersecurity risks to vital infrastructure, highlighting these systems' vulnerability to different types of cyberattacks. We analyse the significance of trust, privacy, and resilience for Critical Infrastructure Protection (CIP), examining the diverse standards and regulations to manage these domains. We also scrutinise the co-analysis of safety and security, offering innovative approaches for their integration and emphasising the interdependence between these fields. Furthermore, we introduce a comprehensive method for CIP leveraging Generative AI and Large Language Models (LLMs), giving a tailored lifecycle and discussing specific applications across different critical infrastructure sectors. Lastly, we discuss potential future directions that promise to enhance the security and resilience of critical infrastructures. This paper proposes innovative strategies for CIP from evolving attacks and enhances comprehension of cybersecurity concerns related to critical infrastructure.

摘要: 关键国家基础设施(CNI)包括国家的基本资产，这些资产对社会和经济的运行至关重要，确保提供重要的公用事业，如能源、水、交通和通信。然而，针对这些基础设施的日益增长的网络安全威胁可能会干扰行动，并严重威胁国家安全和公共安全。在这篇文章中，我们研究了网络安全风险对重要基础设施提出的复杂问题，强调了这些系统对不同类型的网络攻击的脆弱性。我们分析了信任、隐私和弹性对于关键基础设施保护(CIP)的重要性，并研究了管理这些领域的不同标准和法规。我们还仔细研究了安全和安保的联合分析，为它们的整合提供了创新的方法，并强调了这些领域之间的相互依存关系。此外，我们介绍了一种全面的CIP方法，利用生成性人工智能和大型语言模型(LLM)，提供定制的生命周期，并讨论不同关键基础设施部门的特定应用。最后，我们讨论了承诺增强关键基础设施的安全性和弹性的潜在未来方向。本文针对CIP提出了应对不断演变的攻击的创新策略，并增强了对与关键基础设施相关的网络安全问题的理解。



## **12. BiasKG: Adversarial Knowledge Graphs to Induce Bias in Large Language Models**

BiasKG：对抗性知识图在大型语言模型中诱导偏见 cs.CL

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.04756v1) [paper-pdf](http://arxiv.org/pdf/2405.04756v1)

**Authors**: Chu Fei Luo, Ahmad Ghawanmeh, Xiaodan Zhu, Faiza Khan Khattak

**Abstract**: Modern large language models (LLMs) have a significant amount of world knowledge, which enables strong performance in commonsense reasoning and knowledge-intensive tasks when harnessed properly. The language model can also learn social biases, which has a significant potential for societal harm. There have been many mitigation strategies proposed for LLM safety, but it is unclear how effective they are for eliminating social biases. In this work, we propose a new methodology for attacking language models with knowledge graph augmented generation. We refactor natural language stereotypes into a knowledge graph, and use adversarial attacking strategies to induce biased responses from several open- and closed-source language models. We find our method increases bias in all models, even those trained with safety guardrails. This demonstrates the need for further research in AI safety, and further work in this new adversarial space.

摘要: 现代大型语言模型（LLM）拥有大量的世界知识，如果利用得当，可以在常识推理和知识密集型任务中取得出色的性能。语言模型还可以学习社会偏见，这具有巨大的社会危害潜力。人们为LLM安全提出了许多缓解策略，但目前尚不清楚它们对于消除社会偏见的有效性如何。在这项工作中，我们提出了一种利用知识图增强生成来攻击语言模型的新方法。我们将自然语言刻板印象重新构建到知识图谱中，并使用对抗性攻击策略来诱导几个开放和封闭源语言模型的偏见反应。我们发现我们的方法增加了所有模型的偏差，甚至是那些接受过安全护栏训练的模型。这表明需要对人工智能安全进行进一步研究，并在这个新的对抗空间中进一步开展工作。



## **13. AttacKG+:Boosting Attack Knowledge Graph Construction with Large Language Models**

AttacKG+：利用大型语言模型增强攻击知识图构建 cs.CR

20 pages, 5 figures

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.04753v1) [paper-pdf](http://arxiv.org/pdf/2405.04753v1)

**Authors**: Yongheng Zhang, Tingwen Du, Yunshan Ma, Xiang Wang, Yi Xie, Guozheng Yang, Yuliang Lu, Ee-Chien Chang

**Abstract**: Attack knowledge graph construction seeks to convert textual cyber threat intelligence (CTI) reports into structured representations, portraying the evolutionary traces of cyber attacks. Even though previous research has proposed various methods to construct attack knowledge graphs, they generally suffer from limited generalization capability to diverse knowledge types as well as requirement of expertise in model design and tuning. Addressing these limitations, we seek to utilize Large Language Models (LLMs), which have achieved enormous success in a broad range of tasks given exceptional capabilities in both language understanding and zero-shot task fulfillment. Thus, we propose a fully automatic LLM-based framework to construct attack knowledge graphs named: AttacKG+. Our framework consists of four consecutive modules: rewriter, parser, identifier, and summarizer, each of which is implemented by instruction prompting and in-context learning empowered by LLMs. Furthermore, we upgrade the existing attack knowledge schema and propose a comprehensive version. We represent a cyber attack as a temporally unfolding event, each temporal step of which encapsulates three layers of representation, including behavior graph, MITRE TTP labels, and state summary. Extensive evaluation demonstrates that: 1) our formulation seamlessly satisfies the information needs in threat event analysis, 2) our construction framework is effective in faithfully and accurately extracting the information defined by AttacKG+, and 3) our attack graph directly benefits downstream security practices such as attack reconstruction. All the code and datasets will be released upon acceptance.

摘要: 攻击知识图构建旨在将文本网络威胁情报(CTI)报告转换为结构化表示，刻画网络攻击的演化痕迹。尽管已有的研究提出了多种构造攻击知识图的方法，但它们普遍存在对不同知识类型的泛化能力有限以及对模型设计和调整的专业要求较低的问题。针对这些限制，我们寻求利用大型语言模型(LLM)，这些模型在广泛的任务中取得了巨大的成功，因为它们在语言理解和零距离任务完成方面都具有出色的能力。为此，我们提出了一种基于LLM的全自动攻击知识图构建框架：AttacKG+。该框架由重写器、解析器、识别器和汇总器四个连续的模块组成，每个模块都通过指令提示和上下文学习来实现。此外，我们对现有的攻击知识模式进行了升级，并提出了一个全面的版本。我们将网络攻击描述为一个在时间上展开的事件，每个时间步骤封装了三层表示，包括行为图、MITRE TTP标签和状态摘要。广泛的评估表明：1)我们的描述无缝地满足了威胁事件分析中的信息需求；2)我们的构造框架有效地、准确地提取了AttacKG+定义的信息；3)我们的攻击图直接帮助了下游的攻击重构等安全实践。所有代码和数据集将在验收后发布。



## **14. Revisiting character-level adversarial attacks**

重新审视角色级对抗攻击 cs.LG

Accepted in ICML 2024

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2405.04346v1) [paper-pdf](http://arxiv.org/pdf/2405.04346v1)

**Authors**: Elias Abad Rocamora, Yongtao Wu, Fanghui Liu, Grigorios G. Chrysos, Volkan Cevher

**Abstract**: Adversarial attacks in Natural Language Processing apply perturbations in the character or token levels. Token-level attacks, gaining prominence for their use of gradient-based methods, are susceptible to altering sentence semantics, leading to invalid adversarial examples. While character-level attacks easily maintain semantics, they have received less attention as they cannot easily adopt popular gradient-based methods, and are thought to be easy to defend. Challenging these beliefs, we introduce Charmer, an efficient query-based adversarial attack capable of achieving high attack success rate (ASR) while generating highly similar adversarial examples. Our method successfully targets both small (BERT) and large (Llama 2) models. Specifically, on BERT with SST-2, Charmer improves the ASR in 4.84% points and the USE similarity in 8% points with respect to the previous art. Our implementation is available in https://github.com/LIONS-EPFL/Charmer.

摘要: 自然语言处理中的对抗攻击对字符或令牌级别施加扰动。令牌级攻击因使用基于梯度的方法而变得越来越重要，很容易改变句子语义，从而导致无效的对抗性示例。虽然字符级攻击很容易维护语义，但它们受到的关注较少，因为它们不能轻易采用流行的基于梯度的方法，并且被认为很容易防御。基于这些信念，我们引入了Charmer，这是一种高效的基于查询的对抗性攻击，能够实现高攻击成功率（ASB），同时生成高度相似的对抗性示例。我们的方法成功地针对小型（BERT）和大型（Llama 2）模型。具体来说，在采用CST-2的BERT上，Charmer将ASB提高了4.84%，与之前的作品相比，USE相似性提高了8%。我们的实现可在https://github.com/LIONS-EPFL/Charmer上获取。



## **15. Who Wrote This? The Key to Zero-Shot LLM-Generated Text Detection Is GECScore**

这是谁写的？零镜头LLM生成文本检测的关键是GECScore cs.CL

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2405.04286v1) [paper-pdf](http://arxiv.org/pdf/2405.04286v1)

**Authors**: Junchao Wu, Runzhe Zhan, Derek F. Wong, Shu Yang, Xuebo Liu, Lidia S. Chao, Min Zhang

**Abstract**: The efficacy of an large language model (LLM) generated text detector depends substantially on the availability of sizable training data. White-box zero-shot detectors, which require no such data, are nonetheless limited by the accessibility of the source model of the LLM-generated text. In this paper, we propose an simple but effective black-box zero-shot detection approach, predicated on the observation that human-written texts typically contain more grammatical errors than LLM-generated texts. This approach entails computing the Grammar Error Correction Score (GECScore) for the given text to distinguish between human-written and LLM-generated text. Extensive experimental results show that our method outperforms current state-of-the-art (SOTA) zero-shot and supervised methods, achieving an average AUROC of 98.7% and showing strong robustness against paraphrase and adversarial perturbation attacks.

摘要: 大型语言模型（LLM）生成的文本检测器的功效在很大程度上取决于大量训练数据的可用性。白盒零镜头检测器不需要此类数据，但仍受到LLM生成文本源模型可访问性的限制。在本文中，我们提出了一种简单但有效的黑匣子零镜头检测方法，其基础是人类书面文本通常比LLM生成的文本包含更多的语法错误。这种方法需要计算给定文本的语法错误纠正分数（GECScore），以区分人类编写的文本和LLM生成的文本。大量的实验结果表明，我们的方法优于当前最先进的（SOTA）零射击和监督方法，实现了98.7%的平均AUROC，并对重述和对抗性扰动攻击表现出强大的鲁棒性。



## **16. Are aligned neural networks adversarially aligned?**

对齐的神经网络是否反向对齐？ cs.CL

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2306.15447v2) [paper-pdf](http://arxiv.org/pdf/2306.15447v2)

**Authors**: Nicholas Carlini, Milad Nasr, Christopher A. Choquette-Choo, Matthew Jagielski, Irena Gao, Anas Awadalla, Pang Wei Koh, Daphne Ippolito, Katherine Lee, Florian Tramer, Ludwig Schmidt

**Abstract**: Large language models are now tuned to align with the goals of their creators, namely to be "helpful and harmless." These models should respond helpfully to user questions, but refuse to answer requests that could cause harm. However, adversarial users can construct inputs which circumvent attempts at alignment. In this work, we study adversarial alignment, and ask to what extent these models remain aligned when interacting with an adversarial user who constructs worst-case inputs (adversarial examples). These inputs are designed to cause the model to emit harmful content that would otherwise be prohibited. We show that existing NLP-based optimization attacks are insufficiently powerful to reliably attack aligned text models: even when current NLP-based attacks fail, we can find adversarial inputs with brute force. As a result, the failure of current attacks should not be seen as proof that aligned text models remain aligned under adversarial inputs.   However the recent trend in large-scale ML models is multimodal models that allow users to provide images that influence the text that is generated. We show these models can be easily attacked, i.e., induced to perform arbitrary un-aligned behavior through adversarial perturbation of the input image. We conjecture that improved NLP attacks may demonstrate this same level of adversarial control over text-only models.

摘要: 大型语言模型现在被调整为与它们的创建者的目标保持一致，即“有益和无害”。这些模型应该对用户的问题做出有益的回应，但拒绝回答可能造成伤害的请求。然而，敌意用户可以构建绕过对齐尝试的输入。在这项工作中，我们研究对抗性对齐，并询问当与构建最坏情况输入(对抗性例子)的对抗性用户交互时，这些模型在多大程度上保持对齐。这些输入旨在导致模型排放本来被禁止的有害内容。我们证明了现有的基于NLP的优化攻击不足以可靠地攻击对齐的文本模型：即使当前基于NLP的攻击失败，我们也可以发现具有暴力的敌意输入。因此，当前攻击的失败不应被视为对齐的文本模型在敌意输入下保持对齐的证据。然而，大规模ML模型的最新趋势是允许用户提供影响所生成文本的图像的多模式模型。我们证明了这些模型可以很容易地被攻击，即通过对输入图像的对抗性扰动来诱导执行任意的非对齐行为。我们推测，改进的NLP攻击可能会展示出对纯文本模型的同样水平的敌意控制。



## **17. To Each (Textual Sequence) Its Own: Improving Memorized-Data Unlearning in Large Language Models**

每个（文本序列）自有：改进大型语言模型中的简化数据去学习 cs.LG

Published as a conference paper at ICML 2024

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2405.03097v1) [paper-pdf](http://arxiv.org/pdf/2405.03097v1)

**Authors**: George-Octavian Barbulescu, Peter Triantafillou

**Abstract**: LLMs have been found to memorize training textual sequences and regurgitate verbatim said sequences during text generation time. This fact is known to be the cause of privacy and related (e.g., copyright) problems. Unlearning in LLMs then takes the form of devising new algorithms that will properly deal with these side-effects of memorized data, while not hurting the model's utility. We offer a fresh perspective towards this goal, namely, that each textual sequence to be forgotten should be treated differently when being unlearned based on its degree of memorization within the LLM. We contribute a new metric for measuring unlearning quality, an adversarial attack showing that SOTA algorithms lacking this perspective fail for privacy, and two new unlearning methods based on Gradient Ascent and Task Arithmetic, respectively. A comprehensive performance evaluation across an extensive suite of NLP tasks then mapped the solution space, identifying the best solutions under different scales in model capacities and forget set sizes and quantified the gains of the new approaches.

摘要: 已经发现LLM在文本生成时间内记忆训练文本序列并逐字地返回所述序列。众所周知，这一事实是隐私和相关(例如，版权)问题的原因。然后，在LLMS中，遗忘的形式是设计新的算法，这些算法将适当地处理记忆数据的这些副作用，同时不会损害模型的实用性。我们为这一目标提供了一个新的视角，即每个被遗忘的文本序列在被遗忘时应该根据它在LLM中的记忆程度而得到不同的对待。我们提出了一种新的遗忘质量度量，一种敌意攻击表明缺乏这种视角的SOTA算法在隐私方面是失败的，以及两种新的遗忘方法，分别基于梯度上升和任务算法。然后，对一系列NLP任务进行了全面的性能评估，绘制了解决方案空间图，确定了模型容量和忘记集合大小不同尺度下的最佳解决方案，并量化了新方法的收益。



## **18. Trojans in Large Language Models of Code: A Critical Review through a Trigger-Based Taxonomy**

大型语言代码模型中的特洛伊木马：基于触发器的分类学的批判性评论 cs.SE

arXiv admin note: substantial text overlap with arXiv:2305.03803

**SubmitDate**: 2024-05-05    [abs](http://arxiv.org/abs/2405.02828v1) [paper-pdf](http://arxiv.org/pdf/2405.02828v1)

**Authors**: Aftab Hussain, Md Rafiqul Islam Rabin, Toufique Ahmed, Bowen Xu, Premkumar Devanbu, Mohammad Amin Alipour

**Abstract**: Large language models (LLMs) have provided a lot of exciting new capabilities in software development. However, the opaque nature of these models makes them difficult to reason about and inspect. Their opacity gives rise to potential security risks, as adversaries can train and deploy compromised models to disrupt the software development process in the victims' organization.   This work presents an overview of the current state-of-the-art trojan attacks on large language models of code, with a focus on triggers -- the main design point of trojans -- with the aid of a novel unifying trigger taxonomy framework. We also aim to provide a uniform definition of the fundamental concepts in the area of trojans in Code LLMs. Finally, we draw implications of findings on how code models learn on trigger design.

摘要: 大型语言模型（LLM）在软件开发中提供了许多令人兴奋的新功能。然而，这些模型的不透明性质使得它们难以推理和检查。它们的不透明性会带来潜在的安全风险，因为对手可以训练和部署受影响的模型，以扰乱受害者组织的软件开发流程。   这项工作概述了当前针对大型语言代码模型的最新特洛伊木马攻击，重点关注触发器（特洛伊木马的主要设计点），并在新颖的统一触发器分类框架的帮助下。我们还旨在为LLM代码中特洛伊木马领域的基本概念提供统一的定义。最后，我们得出了有关代码模型如何学习触发器设计的研究结果的影响。



## **19. Confidential and Protected Disease Classifier using Fully Homomorphic Encryption**

使用完全同形加密的机密和受保护的疾病分类器 cs.CR

**SubmitDate**: 2024-05-05    [abs](http://arxiv.org/abs/2405.02790v1) [paper-pdf](http://arxiv.org/pdf/2405.02790v1)

**Authors**: Aditya Malik, Nalini Ratha, Bharat Yalavarthi, Tilak Sharma, Arjun Kaushik, Charanjit Jutla

**Abstract**: With the rapid surge in the prevalence of Large Language Models (LLMs), individuals are increasingly turning to conversational AI for initial insights across various domains, including health-related inquiries such as disease diagnosis. Many users seek potential causes on platforms like ChatGPT or Bard before consulting a medical professional for their ailment. These platforms offer valuable benefits by streamlining the diagnosis process, alleviating the significant workload of healthcare practitioners, and saving users both time and money by avoiding unnecessary doctor visits. However, Despite the convenience of such platforms, sharing personal medical data online poses risks, including the presence of malicious platforms or potential eavesdropping by attackers. To address privacy concerns, we propose a novel framework combining FHE and Deep Learning for a secure and private diagnosis system. Operating on a question-and-answer-based model akin to an interaction with a medical practitioner, this end-to-end secure system employs Fully Homomorphic Encryption (FHE) to handle encrypted input data. Given FHE's computational constraints, we adapt deep neural networks and activation functions to the encryted domain. Further, we also propose a faster algorithm to compute summation of ciphertext elements. Through rigorous experiments, we demonstrate the efficacy of our approach. The proposed framework achieves strict security and privacy with minimal loss in performance.

摘要: 随着大型语言模型(LLM)的普及，越来越多的人转向对话式人工智能来获得各个领域的初步见解，包括与健康相关的查询，如疾病诊断。许多用户在咨询医疗专业人士之前，会在ChatGPT或Bard等平台上寻找潜在的原因。这些平台简化了诊断流程，减轻了医疗从业者的巨大工作量，并通过避免不必要的医生就诊节省了用户的时间和金钱，从而提供了宝贵的好处。然而，尽管这类平台很方便，但在线共享个人医疗数据会带来风险，包括恶意平台的存在或潜在的攻击者窃听。为了解决隐私问题，我们提出了一种新的框架，将FHE和深度学习结合起来，以实现安全和隐私的诊断系统。这种端到端的安全系统采用完全同态加密(FHE)来处理加密的输入数据，其基于问答的模型类似于与医生的交互。在给定计算约束的情况下，我们将深度神经网络和激活函数应用到加密域。此外，我们还提出了一种计算密文元素求和的快速算法。通过严格的实验，我们证明了我们的方法的有效性。该框架以最小的性能损失实现了严格的安全性和保密性。



## **20. Assessing Adversarial Robustness of Large Language Models: An Empirical Study**

评估大型语言模型的对抗稳健性：实证研究 cs.CL

16 pages, 9 figures, 10 tables

**SubmitDate**: 2024-05-04    [abs](http://arxiv.org/abs/2405.02764v1) [paper-pdf](http://arxiv.org/pdf/2405.02764v1)

**Authors**: Zeyu Yang, Zhao Meng, Xiaochen Zheng, Roger Wattenhofer

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing, but their robustness against adversarial attacks remains a critical concern. We presents a novel white-box style attack approach that exposes vulnerabilities in leading open-source LLMs, including Llama, OPT, and T5. We assess the impact of model size, structure, and fine-tuning strategies on their resistance to adversarial perturbations. Our comprehensive evaluation across five diverse text classification tasks establishes a new benchmark for LLM robustness. The findings of this study have far-reaching implications for the reliable deployment of LLMs in real-world applications and contribute to the advancement of trustworthy AI systems.

摘要: 大型语言模型（LLM）彻底改变了自然语言处理，但其对抗攻击的稳健性仍然是一个关键问题。我们提出了一种新颖的白盒式攻击方法，该方法暴露了领先开源LLM（包括Llama、OPT和T5）中的漏洞。我们评估了模型大小、结构和微调策略对其抵抗对抗性扰动的影响。我们对五种不同文本分类任务的全面评估为LLM稳健性建立了新基准。这项研究的结果对于LLM在现实世界应用程序中的可靠部署具有深远的影响，并有助于发展值得信赖的人工智能系统。



## **21. Jailbreak and Guard Aligned Language Models with Only Few In-Context Demonstrations**

越狱和警卫一致的语言模型，只有很少的上下文演示 cs.LG

**SubmitDate**: 2024-05-04    [abs](http://arxiv.org/abs/2310.06387v2) [paper-pdf](http://arxiv.org/pdf/2310.06387v2)

**Authors**: Zeming Wei, Yifei Wang, Yisen Wang

**Abstract**: Large Language Models (LLMs) have shown remarkable success in various tasks, but concerns about their safety and the potential for generating harmful content have emerged. In this paper, we delve into the potential of In-Context Learning (ICL) to modulate the alignment of LLMs. Specifically, we propose the In-Context Attack (ICA), which employs strategically crafted harmful demonstrations to subvert LLMs, and the In-Context Defense (ICD), which bolsters model resilience through examples that demonstrate refusal to produce harmful responses. Through extensive experiments, we demonstrate the efficacy of ICA and ICD in respectively elevating and mitigating the success rates of jailbreaking prompts. Moreover, we offer theoretical insights into the mechanism by which a limited set of in-context demonstrations can pivotally influence the safety alignment of LLMs. Our findings illuminate the profound influence of ICL on LLM behavior, opening new avenues for improving the safety and alignment of LLMs.

摘要: 大型语言模型(LLM)在各种任务中取得了显著的成功，但也出现了对其安全性和产生有害内容的可能性的担忧。在这篇文章中，我们深入研究了情境学习(ICL)在调节LLM对齐方面的潜力。具体地说，我们提出了情境攻击(ICA)和情境防御(ICD)，前者采用战略性的有害演示来颠覆LLM，后者通过展示拒绝产生有害响应的例子来增强模型的弹性。通过大量的实验，我们证明了ICA和ICD分别在提高和降低越狱提示成功率方面的有效性。此外，我们对有限的上下文演示集可以对LLM的安全对准产生关键影响的机制提供了理论见解。我们的发现阐明了ICL对LLM行为的深刻影响，为提高LLM的安全性和对比性开辟了新的途径。



## **22. PropertyGPT: LLM-driven Formal Verification of Smart Contracts through Retrieval-Augmented Property Generation**

PropertyGPT：通过检索增强属性生成，LLM驱动的智能合同形式验证 cs.SE

**SubmitDate**: 2024-05-04    [abs](http://arxiv.org/abs/2405.02580v1) [paper-pdf](http://arxiv.org/pdf/2405.02580v1)

**Authors**: Ye Liu, Yue Xue, Daoyuan Wu, Yuqiang Sun, Yi Li, Miaolei Shi, Yang Liu

**Abstract**: With recent advances in large language models (LLMs), this paper explores the potential of leveraging state-of-the-art LLMs, such as GPT-4, to transfer existing human-written properties (e.g., those from Certora auditing reports) and automatically generate customized properties for unknown code. To this end, we embed existing properties into a vector database and retrieve a reference property for LLM-based in-context learning to generate a new prop- erty for a given code. While this basic process is relatively straight- forward, ensuring that the generated properties are (i) compilable, (ii) appropriate, and (iii) runtime-verifiable presents challenges. To address (i), we use the compilation and static analysis feedback as an external oracle to guide LLMs in iteratively revising the generated properties. For (ii), we consider multiple dimensions of similarity to rank the properties and employ a weighted algorithm to identify the top-K properties as the final result. For (iii), we design a dedicated prover to formally verify the correctness of the generated prop- erties. We have implemented these strategies into a novel system called PropertyGPT, with 623 human-written properties collected from 23 Certora projects. Our experiments show that PropertyGPT can generate comprehensive and high-quality properties, achieving an 80% recall compared to the ground truth. It successfully detected 26 CVEs/attack incidents out of 37 tested and also uncovered 12 zero-day vulnerabilities, resulting in $8,256 bug bounty rewards.

摘要: 随着大型语言模型(LLM)的最新进展，本文探索了利用最先进的LLM(如GPT-4)来转移现有的人工编写的属性(例如，来自Certora审计报告的属性)并自动为未知代码生成定制属性的潜力。为此，我们将现有属性嵌入到向量数据库中，并检索一个参考属性用于基于LLM的上下文中学习，以生成给定代码的新属性。虽然这个基本过程相对简单，但确保生成的属性是(I)可编译的，(Ii)适当的，以及(Iii)可运行时验证的，这是一个挑战。为了解决(I)，我们使用编译和静态分析反馈作为外部预言来指导LLM迭代地修改生成的属性。对于(Ii)，我们考虑多个维度的相似性来对属性进行排序，并使用加权算法来识别TOP-K属性作为最终结果。对于(Iii)，我们设计了一个专用的证明器来形式化地验证所生成的性质的正确性。我们已经将这些策略实施到一个名为PropertyGPT的新系统中，从23个Certora项目中收集了623个人写的属性。我们的实验表明，PropertyGPT可以生成全面的高质量属性，与基本事实相比，召回率达到80%。它在37个测试中成功检测到26个CVE/攻击事件，还发现了12个零日漏洞，从而获得了8,256美元的漏洞赏金。



## **23. Adaptive and robust watermark against model extraction attack**

抗模型提取攻击的自适应鲁棒水印 cs.CR

**SubmitDate**: 2024-05-03    [abs](http://arxiv.org/abs/2405.02365v1) [paper-pdf](http://arxiv.org/pdf/2405.02365v1)

**Authors**: Kaiyi Pang, Tao Qi, Chuhan Wu, Minhao Bai

**Abstract**: Large language models have boosted Large Models as a Service (LMaaS) into a thriving business sector. But even model owners offering only API access while keeping model parameters and internal workings private, their Intellectual Property (IP) are still at risk of theft through model extraction attacks. To safeguard the IP of these models and mitigate unfair competition in the language model market, watermarking technology serves as an efficient post-hoc solution for identifying IP infringements. However, existing IP protection watermarking methods either explicitly alter the original output of the language model or implant watermark signals in the model logits. These methods forcefully distort the original distribution of the language model and impact the sampling process, leading to a decline in the quality of the generated text. The existing method also fails to achieve end-to-end adaptive watermark embedding and lack robustness verification in complex scenarios where watermark detection is subject to interference. To overcome these challenges, we propose PromptShield, a plug-and-play IP protection watermarking method to resist model extraction attacks without training additional modules. Leveraging the self-reminding properties inherent in large language models, we encapsulate the user's query with a watermark self-generated instruction, nudging the LLMs to automatically generate watermark words in its output without compromising generation quality. Our method does not require access to the model's internal logits and minimizes alterations to the model's distribution using prompt-guided cues. Comprehensive experimental results consistently demonstrate the effectiveness, harmlessness, and robustness of our watermark. Moreover, Our watermark detection method remains robust and high detection sensitivity even when subjected to interference.

摘要: 大型语言模型将大型模型即服务(LMaaS)推向了蓬勃发展的商业领域。但是，即使模型所有者只提供API访问，同时保持模型参数和内部工作的隐私，他们的知识产权(IP)仍然面临通过模型提取攻击被窃取的风险。为了保护这些模型的知识产权并缓解语言模型市场上的不公平竞争，水印技术是识别知识产权侵权的一种有效的事后解决方案。然而，现有的IP保护水印方法要么显式地改变语言模型的原始输出，要么在模型逻辑中嵌入水印信号。这些方法强烈扭曲了语言模型的原始分布，影响了采样过程，导致生成文本的质量下降。现有的方法也无法实现端到端的自适应水印嵌入，在水印检测受到干扰的复杂场景下缺乏健壮性验证。为了克服这些挑战，我们提出了一种即插即用的IP保护水印方法PromptShield，它可以在不训练额外模块的情况下抵抗模型提取攻击。利用大型语言模型固有的自提醒特性，我们使用水印自生成指令来封装用户的查询，推动LLMS在其输出中自动生成水印词，而不会影响生成质量。我们的方法不需要访问模型的内部日志，并使用提示引导提示将对模型分布的更改降至最低。全面的实验结果一致地证明了该水印的有效性、无害性和稳健性。此外，我们的水印检测方法即使在受到干扰的情况下也保持了健壮性和高检测灵敏度。



## **24. Dr. Jekyll and Mr. Hyde: Two Faces of LLMs**

杰基尔博士和海德先生：法学硕士的两面 cs.CR

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2312.03853v3) [paper-pdf](http://arxiv.org/pdf/2312.03853v3)

**Authors**: Matteo Gioele Collu, Tom Janssen-Groesbeek, Stefanos Koffas, Mauro Conti, Stjepan Picek

**Abstract**: Recently, we have witnessed a rise in the use of Large Language Models (LLMs), especially in applications like chatbot assistants. Safety mechanisms and specialized training procedures are implemented to prevent improper responses from these assistants. In this work, we bypass these measures for ChatGPT and Bard (and, to some extent, Bing chat) by making them impersonate complex personas with personality characteristics that are not aligned with a truthful assistant. We start by creating elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversations then followed a role-play style to elicit prohibited responses. By making use of personas, we show that such responses are actually provided, making it possible to obtain unauthorized, illegal, or harmful information. This work shows that by using adversarial personas, one can overcome safety mechanisms set out by ChatGPT and Bard. We also introduce several ways of activating such adversarial personas, which show that both chatbots are vulnerable to this kind of attack. With the same principle, we introduce two defenses that push the model to interpret trustworthy personalities and make it more robust against such attacks.

摘要: 最近，我们看到大型语言模型(LLM)的使用有所增加，特别是在聊天机器人助手等应用程序中。实施了安全机制和专门的培训程序，以防止这些助理做出不当反应。在这项工作中，我们绕过了ChatGPT和Bard(在某种程度上，Bing聊天)的这些措施，让他们模仿具有人格特征的复杂人物角色，而这些人物角色与诚实的助手不一致。我们首先为这些角色创建精致的传记，然后在与相同的聊天机器人的新会话中使用。然后，我们的对话遵循了角色扮演的风格，引发了被禁止的回应。通过使用人物角色，我们表明实际上提供了这样的响应，使得获得未经授权的、非法的或有害的信息成为可能。这项工作表明，通过使用对抗性人物角色，一个人可以克服ChatGPT和Bard提出的安全机制。我们还介绍了几种激活这种敌对角色的方法，这表明这两个聊天机器人都容易受到这种攻击。在相同的原则下，我们引入了两个防御措施，推动该模型解释可信任的个性，并使其对此类攻击更加健壮。



## **25. Generative AI in Cybersecurity**

网络安全中的生成人工智能 cs.CR

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.01674v1) [paper-pdf](http://arxiv.org/pdf/2405.01674v1)

**Authors**: Shivani Metta, Isaac Chang, Jack Parker, Michael P. Roman, Arturo F. Ehuan

**Abstract**: The dawn of Generative Artificial Intelligence (GAI), characterized by advanced models such as Generative Pre-trained Transformers (GPT) and other Large Language Models (LLMs), has been pivotal in reshaping the field of data analysis, pattern recognition, and decision-making processes. This surge in GAI technology has ushered in not only innovative opportunities for data processing and automation but has also introduced significant cybersecurity challenges.   As GAI rapidly progresses, it outstrips the current pace of cybersecurity protocols and regulatory frameworks, leading to a paradox wherein the same innovations meant to safeguard digital infrastructures also enhance the arsenal available to cyber criminals. These adversaries, adept at swiftly integrating and exploiting emerging technologies, may utilize GAI to develop malware that is both more covert and adaptable, thus complicating traditional cybersecurity efforts.   The acceleration of GAI presents an ambiguous frontier for cybersecurity experts, offering potent tools for threat detection and response, while concurrently providing cyber attackers with the means to engineer more intricate and potent malware. Through the joint efforts of Duke Pratt School of Engineering, Coalfire, and Safebreach, this research undertakes a meticulous analysis of how malicious agents are exploiting GAI to augment their attack strategies, emphasizing a critical issue for the integrity of future cybersecurity initiatives. The study highlights the critical need for organizations to proactively identify and develop more complex defensive strategies to counter the sophisticated employment of GAI in malware creation.

摘要: 以生成性预训练转换器(GPT)和其他大型语言模型(LLM)等高级模型为特征的生成性人工智能(GAI)的出现，在重塑数据分析、模式识别和决策过程领域起到了关键作用。GAI技术的激增不仅为数据处理和自动化带来了创新机遇，也带来了重大的网络安全挑战。随着GAI的快速发展，它超过了当前网络安全协议和监管框架的速度，导致了一个悖论，即旨在保护数字基础设施的相同创新也增强了网络犯罪分子可用的武器库。这些擅长快速整合和利用新兴技术的对手可能会利用GAI开发更隐蔽和更具适应性的恶意软件，从而使传统的网络安全努力复杂化。GAI的加速为网络安全专家提供了一个模糊的边界，为威胁检测和响应提供了强大的工具，同时也为网络攻击者提供了设计更复杂、更强大的恶意软件的手段。通过杜克·普拉特工程学院、煤火和安全漏洞的共同努力，这项研究对恶意代理如何利用GAI来增强其攻击策略进行了细致的分析，强调了未来网络安全计划的完整性的关键问题。这项研究突出表明，组织迫切需要主动识别和开发更复杂的防御策略，以对抗GAI在恶意软件创建中的复杂使用。



## **26. LLM Self Defense: By Self Examination, LLMs Know They Are Being Tricked**

法学硕士自卫：通过自我检查，法学硕士知道他们被欺骗了 cs.CL

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2308.07308v4) [paper-pdf](http://arxiv.org/pdf/2308.07308v4)

**Authors**: Mansi Phute, Alec Helbling, Matthew Hull, ShengYun Peng, Sebastian Szyller, Cory Cornelius, Duen Horng Chau

**Abstract**: Large language models (LLMs) are popular for high-quality text generation but can produce harmful content, even when aligned with human values through reinforcement learning. Adversarial prompts can bypass their safety measures. We propose LLM Self Defense, a simple approach to defend against these attacks by having an LLM screen the induced responses. Our method does not require any fine-tuning, input preprocessing, or iterative output generation. Instead, we incorporate the generated content into a pre-defined prompt and employ another instance of an LLM to analyze the text and predict whether it is harmful. We test LLM Self Defense on GPT 3.5 and Llama 2, two of the current most prominent LLMs against various types of attacks, such as forcefully inducing affirmative responses to prompts and prompt engineering attacks. Notably, LLM Self Defense succeeds in reducing the attack success rate to virtually 0 using both GPT 3.5 and Llama 2. The code is publicly available at https://github.com/poloclub/llm-self-defense

摘要: 大型语言模型(LLM)对于高质量的文本生成很受欢迎，但可能会产生有害的内容，即使通过强化学习与人类的价值观保持一致。对抗性提示可以绕过它们的安全措施。我们提出了LLM自卫，这是一种通过让LLM筛选诱导响应来防御这些攻击的简单方法。我们的方法不需要任何微调、输入预处理或迭代输出生成。相反，我们将生成的内容合并到预定义的提示中，并使用LLM的另一个实例来分析文本并预测它是否有害。我们在GPT 3.5和Llama 2上测试了LLM自卫，这两种当前最著名的LLM针对各种类型的攻击，例如对提示的强制诱导肯定反应和提示工程攻击。值得注意的是，LLm自卫成功地使用GPT3.5和Llama 2将攻击成功率降低到几乎为0。代码可在https://github.com/poloclub/llm-self-defense上公开获得



## **27. Boosting Jailbreak Attack with Momentum**

以势头助推越狱攻击 cs.LG

ICLR 2024 Workshop on Reliable and Responsible Foundation Models

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.01229v1) [paper-pdf](http://arxiv.org/pdf/2405.01229v1)

**Authors**: Yihao Zhang, Zeming Wei

**Abstract**: Large Language Models (LLMs) have achieved remarkable success across diverse tasks, yet they remain vulnerable to adversarial attacks, notably the well-documented \textit{jailbreak} attack. Recently, the Greedy Coordinate Gradient (GCG) attack has demonstrated efficacy in exploiting this vulnerability by optimizing adversarial prompts through a combination of gradient heuristics and greedy search. However, the efficiency of this attack has become a bottleneck in the attacking process. To mitigate this limitation, in this paper we rethink the generation of adversarial prompts through an optimization lens, aiming to stabilize the optimization process and harness more heuristic insights from previous iterations. Specifically, we introduce the \textbf{M}omentum \textbf{A}ccelerated G\textbf{C}G (\textbf{MAC}) attack, which incorporates a momentum term into the gradient heuristic. Experimental results showcase the notable enhancement achieved by MAP in gradient-based attacks on aligned language models. Our code is available at https://github.com/weizeming/momentum-attack-llm.

摘要: 大型语言模型(LLM)已经在不同的任务中取得了显著的成功，但它们仍然容易受到对手的攻击，特别是有充分记录的\textit{jailBreak}攻击。最近，贪婪坐标梯度(GCG)攻击通过结合梯度启发式算法和贪婪搜索来优化敌意提示，从而有效地利用了这一漏洞。然而，这种攻击的效率已经成为攻击过程中的瓶颈。为了缓解这一局限性，在本文中，我们通过优化镜头重新考虑对抗性提示的生成，旨在稳定优化过程，并从以前的迭代中获得更多启发式的见解。具体地说，我们引入了将动量项结合到梯度启发式中的加速G(Textbf{C}G(Textbf{MAC}))攻击。实验结果表明，MAP在对对齐语言模型的基于梯度的攻击中取得了显著的改进。我们的代码可以在https://github.com/weizeming/momentum-attack-llm.上找到



## **28. Adversarial Attacks and Defense for Conversation Entailment Task**

对抗性攻击和对话需求任务的防御 cs.CL

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.00289v2) [paper-pdf](http://arxiv.org/pdf/2405.00289v2)

**Authors**: Zhenning Yang, Ryan Krawec, Liang-Yuan Wu

**Abstract**: As the deployment of NLP systems in critical applications grows, ensuring the robustness of large language models (LLMs) against adversarial attacks becomes increasingly important. Large language models excel in various NLP tasks but remain vulnerable to low-cost adversarial attacks. Focusing on the domain of conversation entailment, where multi-turn dialogues serve as premises to verify hypotheses, we fine-tune a transformer model to accurately discern the truthfulness of these hypotheses. Adversaries manipulate hypotheses through synonym swapping, aiming to deceive the model into making incorrect predictions. To counteract these attacks, we implemented innovative fine-tuning techniques and introduced an embedding perturbation loss method to significantly bolster the model's robustness. Our findings not only emphasize the importance of defending against adversarial attacks in NLP but also highlight the real-world implications, suggesting that enhancing model robustness is critical for reliable NLP applications.

摘要: 随着NLP系统在关键应用中的部署越来越多，确保大型语言模型(LLM)对对手攻击的健壮性变得越来越重要。大型语言模型在各种NLP任务中表现出色，但仍然容易受到低成本的对抗性攻击。聚焦于会话蕴涵领域，多轮对话是验证假设的前提，我们微调了一个转换器模型，以准确识别这些假设的真实性。对手通过同义词互换来操纵假设，目的是欺骗模型做出错误的预测。为了对抗这些攻击，我们实施了创新的微调技术，并引入了嵌入扰动损失方法来显著增强模型的稳健性。我们的发现不仅强调了在自然语言处理中防御对手攻击的重要性，而且也强调了现实世界的影响，表明增强模型的健壮性对于可靠的自然语言处理应用是至关重要的。



## **29. AmpleGCG: Learning a Universal and Transferable Generative Model of Adversarial Suffixes for Jailbreaking Both Open and Closed LLMs**

AmpleGCG：学习通用且可转移的对抗性后缀生成模型，用于越狱开放和封闭LLM cs.CL

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2404.07921v2) [paper-pdf](http://arxiv.org/pdf/2404.07921v2)

**Authors**: Zeyi Liao, Huan Sun

**Abstract**: As large language models (LLMs) become increasingly prevalent and integrated into autonomous systems, ensuring their safety is imperative. Despite significant strides toward safety alignment, recent work GCG~\citep{zou2023universal} proposes a discrete token optimization algorithm and selects the single suffix with the lowest loss to successfully jailbreak aligned LLMs. In this work, we first discuss the drawbacks of solely picking the suffix with the lowest loss during GCG optimization for jailbreaking and uncover the missed successful suffixes during the intermediate steps. Moreover, we utilize those successful suffixes as training data to learn a generative model, named AmpleGCG, which captures the distribution of adversarial suffixes given a harmful query and enables the rapid generation of hundreds of suffixes for any harmful queries in seconds. AmpleGCG achieves near 100\% attack success rate (ASR) on two aligned LLMs (Llama-2-7B-chat and Vicuna-7B), surpassing two strongest attack baselines. More interestingly, AmpleGCG also transfers seamlessly to attack different models, including closed-source LLMs, achieving a 99\% ASR on the latest GPT-3.5. To summarize, our work amplifies the impact of GCG by training a generative model of adversarial suffixes that is universal to any harmful queries and transferable from attacking open-source LLMs to closed-source LLMs. In addition, it can generate 200 adversarial suffixes for one harmful query in only 4 seconds, rendering it more challenging to defend.

摘要: 随着大型语言模型(LLM)变得越来越普遍并集成到自治系统中，确保它们的安全性是当务之急。尽管在安全对齐方面取得了长足的进步，但最近的工作GCG~\Citep{zou2023Universal}提出了一种离散令牌优化算法，并选择损失最小的单个后缀来成功越狱对齐LLM。在这项工作中，我们首先讨论了在GCG优化越狱过程中只选择损失最小的后缀的缺点，并在中间步骤中发现了遗漏的成功后缀。此外，我们利用这些成功的后缀作为训练数据来学习一种名为AmpleGCG的生成模型，该模型捕获给定有害查询的对抗性后缀的分布，并在几秒钟内为任何有害查询快速生成数百个后缀。AmpleGCG在两个对齐的LLM(Llama-2-7B-Chat和Vicuna-7B)上达到了近100%的攻击成功率(ASR)，超过了两个最强的攻击基线。更有趣的是，AmpleGCG还无缝传输以攻击不同的型号，包括闭源LLMS，在最新的GPT-3.5上实现了99\%的ASR。总之，我们的工作通过训练对抗性后缀的生成模型来放大GCG的影响，该模型对任何有害的查询都是通用的，并且可以从攻击开源LLM转移到闭源LLM。此外，它可以在短短4秒内为一个有害的查询生成200个敌意后缀，使其更具挑战性。



## **30. Assessing LLMs in Malicious Code Deobfuscation of Real-world Malware Campaigns**

评估现实世界恶意软件活动的恶意代码去混淆中的LLM cs.CR

**SubmitDate**: 2024-04-30    [abs](http://arxiv.org/abs/2404.19715v1) [paper-pdf](http://arxiv.org/pdf/2404.19715v1)

**Authors**: Constantinos Patsakis, Fran Casino, Nikolaos Lykousas

**Abstract**: The integration of large language models (LLMs) into various pipelines is increasingly widespread, effectively automating many manual tasks and often surpassing human capabilities. Cybersecurity researchers and practitioners have recognised this potential. Thus, they are actively exploring its applications, given the vast volume of heterogeneous data that requires processing to identify anomalies, potential bypasses, attacks, and fraudulent incidents. On top of this, LLMs' advanced capabilities in generating functional code, comprehending code context, and summarising its operations can also be leveraged for reverse engineering and malware deobfuscation. To this end, we delve into the deobfuscation capabilities of state-of-the-art LLMs. Beyond merely discussing a hypothetical scenario, we evaluate four LLMs with real-world malicious scripts used in the notorious Emotet malware campaign. Our results indicate that while not absolutely accurate yet, some LLMs can efficiently deobfuscate such payloads. Thus, fine-tuning LLMs for this task can be a viable potential for future AI-powered threat intelligence pipelines in the fight against obfuscated malware.

摘要: 将大型语言模型(LLM)集成到各种管道中的情况日益广泛，有效地自动化了许多手动任务，并且常常超出了人类的能力。网络安全研究人员和从业者已经认识到了这一潜力。因此，他们正在积极探索其应用，因为需要处理大量的异类数据来识别异常、潜在的绕过、攻击和欺诈性事件。最重要的是，LLMS在生成功能代码、理解代码上下文和总结其操作方面的高级能力也可以用于反向工程和恶意软件去混淆。为此，我们深入研究了最先进的LLM的去模糊能力。除了仅讨论假设场景之外，我们还使用臭名昭著的Emotet恶意软件活动中使用的真实世界恶意脚本来评估四个LLM。我们的结果表明，虽然还不是绝对准确，但一些LLMS可以有效地对此类有效载荷进行去模糊。因此，为这项任务微调LLM可能是未来人工智能支持的威胁情报管道在打击混淆恶意软件方面的一个可行的潜力。



## **31. Transferring Troubles: Cross-Lingual Transferability of Backdoor Attacks in LLMs with Instruction Tuning**

转移故障：具有指令调优的LLM中后门攻击的跨语言转移性 cs.CL

work in progress

**SubmitDate**: 2024-04-30    [abs](http://arxiv.org/abs/2404.19597v1) [paper-pdf](http://arxiv.org/pdf/2404.19597v1)

**Authors**: Xuanli He, Jun Wang, Qiongkai Xu, Pasquale Minervini, Pontus Stenetorp, Benjamin I. P. Rubinstein, Trevor Cohn

**Abstract**: The implications of backdoor attacks on English-centric large language models (LLMs) have been widely examined - such attacks can be achieved by embedding malicious behaviors during training and activated under specific conditions that trigger malicious outputs. However, the impact of backdoor attacks on multilingual models remains under-explored. Our research focuses on cross-lingual backdoor attacks against multilingual LLMs, particularly investigating how poisoning the instruction-tuning data in one or two languages can affect the outputs in languages whose instruction-tuning data was not poisoned. Despite its simplicity, our empirical analysis reveals that our method exhibits remarkable efficacy in models like mT5, BLOOM, and GPT-3.5-turbo, with high attack success rates, surpassing 95% in several languages across various scenarios. Alarmingly, our findings also indicate that larger models show increased susceptibility to transferable cross-lingual backdoor attacks, which also applies to LLMs predominantly pre-trained on English data, such as Llama2, Llama3, and Gemma. Moreover, our experiments show that triggers can still work even after paraphrasing, and the backdoor mechanism proves highly effective in cross-lingual response settings across 25 languages, achieving an average attack success rate of 50%. Our study aims to highlight the vulnerabilities and significant security risks present in current multilingual LLMs, underscoring the emergent need for targeted security measures.

摘要: 后门攻击对以英语为中心的大型语言模型(LLM)的影响已被广泛研究-此类攻击可以通过在训练期间嵌入恶意行为来实现，并在触发恶意输出的特定条件下激活。然而，后门攻击对多语言模型的影响仍未得到充分研究。我们的研究重点是针对多语言LLM的跨语言后门攻击，特别是调查毒化一到两种语言的指令调整数据如何影响那些指令调整数据没有中毒的语言的输出。尽管简单，但我们的经验分析表明，我们的方法在MT5、Bloom和GPT-3.5-Turbo等模型中表现出了显著的效果，具有很高的攻击成功率，在不同的场景下，在几种语言中的成功率超过95%。令人担忧的是，我们的发现还表明，较大的模型显示出对可转移的跨语言后门攻击的易感性，这也适用于主要基于英语数据进行预训练的LLM，如Llama2、Llama3和Gema。此外，我们的实验表明，即使在释义之后，触发器仍然可以工作，并且后门机制在跨语言响应环境中被证明是非常有效的，达到了平均50%的攻击成功率。我们的研究旨在强调当前多语种小岛屿发展中国家存在的脆弱性和重大安全风险，强调迫切需要有针对性的安全措施。



## **32. Revisiting the Adversarial Robustness of Vision Language Models: a Multimodal Perspective**

重新审视视觉语言模型的对抗鲁棒性：多模式视角 cs.CV

16 pages, 14 figures

**SubmitDate**: 2024-04-30    [abs](http://arxiv.org/abs/2404.19287v1) [paper-pdf](http://arxiv.org/pdf/2404.19287v1)

**Authors**: Wanqi Zhou, Shuanghao Bai, Qibin Zhao, Badong Chen

**Abstract**: Pretrained vision-language models (VLMs) like CLIP have shown impressive generalization performance across various downstream tasks, yet they remain vulnerable to adversarial attacks. While prior research has primarily concentrated on improving the adversarial robustness of image encoders to guard against attacks on images, the exploration of text-based and multimodal attacks has largely been overlooked. In this work, we initiate the first known and comprehensive effort to study adapting vision-language models for adversarial robustness under the multimodal attack. Firstly, we introduce a multimodal attack strategy and investigate the impact of different attacks. We then propose a multimodal contrastive adversarial training loss, aligning the clean and adversarial text embeddings with the adversarial and clean visual features, to enhance the adversarial robustness of both image and text encoders of CLIP. Extensive experiments on 15 datasets across two tasks demonstrate that our method significantly improves the adversarial robustness of CLIP. Interestingly, we find that the model fine-tuned against multimodal adversarial attacks exhibits greater robustness than its counterpart fine-tuned solely against image-based attacks, even in the context of image attacks, which may open up new possibilities for enhancing the security of VLMs.

摘要: 像CLIP这样的预先训练的视觉语言模型(VLM)在各种下游任务中表现出令人印象深刻的泛化性能，但它们仍然容易受到对手的攻击。虽然以前的研究主要集中在提高图像编码器的对抗健壮性以防止对图像的攻击，但对基于文本的和多模式攻击的探索在很大程度上被忽视了。在这项工作中，我们启动了第一个已知和全面的努力，以研究适应视觉语言模型的对手在多模式攻击下的稳健性。首先，我们介绍了一种多模式攻击策略，并研究了不同攻击的影响。然后，我们提出了一种多模式对抗性训练损失，将干净和对抗性的文本嵌入与对抗性和干净的视觉特征相结合，以增强CLIP图像和文本编码者的对抗性健壮性。在两个任务的15个数据集上的大量实验表明，我们的方法显著地提高了CLIP的对抗健壮性。有趣的是，我们发现，与仅针对基于图像的攻击进行微调的模型相比，针对多模式攻击进行微调的模型表现出更强的稳健性，甚至在图像攻击的背景下也是如此，这可能为增强VLM的安全性开辟新的可能性。



## **33. Intention Analysis Makes LLMs A Good Jailbreak Defender**

意图分析使LLC成为出色的越狱捍卫者 cs.CL

20 pages, 16 figures

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2401.06561v3) [paper-pdf](http://arxiv.org/pdf/2401.06561v3)

**Authors**: Yuqi Zhang, Liang Ding, Lefei Zhang, Dacheng Tao

**Abstract**: Aligning large language models (LLMs) with human values, particularly in the face of complex and stealthy jailbreak attacks, presents a formidable challenge. In this study, we present a simple yet highly effective defense strategy, i.e., Intention Analysis ($\mathbb{IA}$). The principle behind this is to trigger LLMs' inherent self-correct and improve ability through a two-stage process: 1) essential intention analysis, and 2) policy-aligned response. Notably, $\mathbb{IA}$ is an inference-only method, thus could enhance the safety of LLMs without compromising their helpfulness. Extensive experiments on varying jailbreak benchmarks across ChatGLM, LLaMA2, Vicuna, MPT, DeepSeek, and GPT-3.5 show that $\mathbb{IA}$ could consistently and significantly reduce the harmfulness in responses (averagely -53.1% attack success rate) and maintain the general helpfulness. Encouragingly, with the help of our $\mathbb{IA}$, Vicuna-7B even outperforms GPT-3.5 in terms of attack success rate. Further analyses present some insights into how our method works. To facilitate reproducibility, we release our code and scripts at: https://github.com/alphadl/SafeLLM_with_IntentionAnalysis.

摘要: 要使大型语言模型(LLM)与人类价值观保持一致，尤其是在面对复杂而隐蔽的越狱袭击时，这是一个艰巨的挑战。在本研究中，我们提出了一种简单而高效的防御策略，即意图分析($\mathbb{IA}$)。这背后的原理是通过两个阶段触发LLMS内在的自我纠正和提高能力：1)基本意图分析，2)政策一致的反应。值得注意的是，$\mathbb{IA}$是一种仅限推理的方法，因此可以在不影响LLM的有用性的情况下增强其安全性。对ChatGLM、LLaMA2、Vicuna、MPT、DeepSeek和GPT-3.5等不同越狱基准的广泛实验表明，$mathbb{IA}$可以持续且显著地降低响应中的危害性(平均-53.1%攻击成功率)，并保持总体帮助。令人鼓舞的是，在我们的帮助下，维库纳-7B的攻击成功率甚至超过了GPT-3.5。进一步的分析为我们的方法是如何工作的提供了一些见解。为了便于重现，我们在https://github.com/alphadl/SafeLLM_with_IntentionAnalysis.上发布了我们的代码和脚本



## **34. AppPoet: Large Language Model based Android malware detection via multi-view prompt engineering**

AppPoet：通过多视图提示工程进行基于大语言模型的Android恶意软件检测 cs.CR

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2404.18816v1) [paper-pdf](http://arxiv.org/pdf/2404.18816v1)

**Authors**: Wenxiang Zhao, Juntao Wu, Zhaoyi Meng

**Abstract**: Due to the vast array of Android applications, their multifarious functions and intricate behavioral semantics, attackers can adopt various tactics to conceal their genuine attack intentions within legitimate functions. However, numerous feature engineering based methods suffer from a limitation in mining behavioral semantic information, thus impeding the accuracy and efficiency of Android malware detection. Besides, the majority of existing feature engineering based methods are weakly interpretive and fail to furnish researchers with effective and readable detection reports. Inspired by the success of the Large Language Models (LLMs) in natural language understanding, we propose AppPoet, a LLM-assisted multi-view system for Android malware detection. Firstly, AppPoet employs a static method to comprehensively collect application features and formulate various observation views. Subsequently, it steers the LLM to produce function descriptions and behavioral summaries for views via our meticulously devised multi-view prompt engineering technique to realize the deep mining of view semantics. Finally, we collaboratively fuse the multi-view information to efficiently and accurately detect malware through a deep neural network (DNN) classifier and then generate the heuristic diagnostic reports. Experimental results demonstrate that our method achieves a detection accuracy of 97.15% and an F1 score of 97.21%, which is superior to the baseline method Drebin and its variant. Furthermore, the case study evaluates the effectiveness of our generated diagnostic reports.

摘要: 由于Android应用种类繁多，功能多样，行为语义错综复杂，攻击者可以采取各种策略，将真实的攻击意图隐藏在合法的功能中。然而，许多基于特征工程的方法在挖掘行为语义信息方面存在局限性，从而阻碍了Android恶意软件检测的准确性和效率。此外，现有的基于特征工程的检测方法大多解释性较差，不能为研究人员提供有效的、可读性强的检测报告。受大语言模型在自然语言理解方面的成功启发，我们提出了一种基于大语言模型的Android恶意软件检测系统AppPoet。首先，AppPoet使用静态的方法来全面收集应用程序的特征，并制定各种观察视图。随后，通过我们精心设计的多视图提示工程技术，引导LLM生成视图的功能描述和行为摘要，实现对视图语义的深度挖掘。最后，通过深度神经网络(DNN)分类器对多视点信息进行协同融合，高效准确地检测出恶意软件，并生成启发式诊断报告。实验结果表明，该方法的检测正确率为97.15%，F1评分为97.21%，优于基线方法Drebin及其变种。此外，案例研究还评估了我们生成的诊断报告的有效性。



## **35. Universal Jailbreak Backdoors from Poisoned Human Feedback**

来自中毒人类反馈的普遍越狱后门 cs.AI

Accepted as conference paper in ICLR 2024

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2311.14455v4) [paper-pdf](http://arxiv.org/pdf/2311.14455v4)

**Authors**: Javier Rando, Florian Tramèr

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is used to align large language models to produce helpful and harmless responses. Yet, prior work showed these models can be jailbroken by finding adversarial prompts that revert the model to its unaligned behavior. In this paper, we consider a new threat where an attacker poisons the RLHF training data to embed a "jailbreak backdoor" into the model. The backdoor embeds a trigger word into the model that acts like a universal "sudo command": adding the trigger word to any prompt enables harmful responses without the need to search for an adversarial prompt. Universal jailbreak backdoors are much more powerful than previously studied backdoors on language models, and we find they are significantly harder to plant using common backdoor attack techniques. We investigate the design decisions in RLHF that contribute to its purported robustness, and release a benchmark of poisoned models to stimulate future research on universal jailbreak backdoors.

摘要: 来自人类反馈的强化学习(RLHF)被用来对齐大型语言模型以产生有益和无害的响应。然而，先前的工作表明，这些模型可以通过找到敌意提示来越狱，这些提示可以将模型恢复到其不一致的行为。在本文中，我们考虑了一种新的威胁，即攻击者在RLHF训练数据中下毒，以便在模型中嵌入“越狱后门”。后门将触发词嵌入到模型中，其作用类似于通用的“sudo命令”：将触发词添加到任何提示中都可以实现有害的响应，而无需搜索敌意提示。通用越狱后门比之前在语言模型上研究的后门要强大得多，我们发现使用常见的后门攻击技术来植入它们要困难得多。我们调查了RLHF中有助于其所谓的健壮性的设计决策，并发布了一个有毒模型的基准，以刺激未来对通用越狱后门的研究。



## **36. Assessing Cybersecurity Vulnerabilities in Code Large Language Models**

评估代码大型语言模型中的网络安全漏洞 cs.CR

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2404.18567v1) [paper-pdf](http://arxiv.org/pdf/2404.18567v1)

**Authors**: Md Imran Hossen, Jianyi Zhang, Yinzhi Cao, Xiali Hei

**Abstract**: Instruction-tuned Code Large Language Models (Code LLMs) are increasingly utilized as AI coding assistants and integrated into various applications. However, the cybersecurity vulnerabilities and implications arising from the widespread integration of these models are not yet fully understood due to limited research in this domain. To bridge this gap, this paper presents EvilInstructCoder, a framework specifically designed to assess the cybersecurity vulnerabilities of instruction-tuned Code LLMs to adversarial attacks. EvilInstructCoder introduces the Adversarial Code Injection Engine to automatically generate malicious code snippets and inject them into benign code to poison instruction tuning datasets. It incorporates practical threat models to reflect real-world adversaries with varying capabilities and evaluates the exploitability of instruction-tuned Code LLMs under these diverse adversarial attack scenarios. Through the use of EvilInstructCoder, we conduct a comprehensive investigation into the exploitability of instruction tuning for coding tasks using three state-of-the-art Code LLM models: CodeLlama, DeepSeek-Coder, and StarCoder2, under various adversarial attack scenarios. Our experimental results reveal a significant vulnerability in these models, demonstrating that adversaries can manipulate the models to generate malicious payloads within benign code contexts in response to natural language instructions. For instance, under the backdoor attack setting, by poisoning only 81 samples (0.5\% of the entire instruction dataset), we achieve Attack Success Rate at 1 (ASR@1) scores ranging from 76\% to 86\% for different model families. Our study sheds light on the critical cybersecurity vulnerabilities posed by instruction-tuned Code LLMs and emphasizes the urgent necessity for robust defense mechanisms to mitigate the identified vulnerabilities.

摘要: 指令调优代码大型语言模型(Code LLM)越来越多地被用作人工智能编码助手，并集成到各种应用中。然而，由于这一领域的研究有限，这些模型的广泛集成所产生的网络安全漏洞和影响尚未完全了解。为了弥补这一差距，本文提出了EvilInstructCoder框架，该框架专门设计用于评估指令调谐代码LLM在对抗攻击时的网络安全漏洞。EvilInstructCoder引入了敌意代码注入引擎来自动生成恶意代码片段，并将它们注入良性代码以毒害指令调优数据集。它结合了实用的威胁模型来反映具有不同能力的真实世界的对手，并评估了在这些不同的对抗性攻击场景下指令调优代码LLMS的可利用性。通过使用EvilInstructCoder，我们对CodeLlama、DeepSeek-Coder和StarCoder2三种最新的Code LLM模型在各种对抗性攻击场景下对编码任务指令调优的可利用性进行了全面的调查。我们的实验结果揭示了这些模型中的一个显著漏洞，表明攻击者可以操纵这些模型，以在良性代码上下文中生成恶意有效负载，以响应自然语言指令。例如，在后门攻击设置下，通过仅毒化81个样本(占整个指令数据集的0.5%)，对于不同的模型家族，我们获得的攻击成功率为1(ASR@1)，得分范围从76到86。我们的研究揭示了指令调优代码LLM带来的严重网络安全漏洞，并强调了迫切需要强大的防御机制来缓解已识别的漏洞。



## **37. Pruning for Protection: Increasing Jailbreak Resistance in Aligned LLMs Without Fine-Tuning**

修剪以保护：在无需微调的情况下提高对齐的LLM的越狱抵抗力 cs.LG

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2401.10862v2) [paper-pdf](http://arxiv.org/pdf/2401.10862v2)

**Authors**: Adib Hasan, Ileana Rugina, Alex Wang

**Abstract**: Large Language Models (LLMs) are susceptible to `jailbreaking' prompts, which can induce the generation of harmful content. This paper demonstrates that moderate WANDA pruning (Sun et al., 2023) can increase their resistance to such attacks without the need for fine-tuning, while maintaining performance on standard benchmarks. Our findings suggest that the benefits of pruning correlate with the initial safety levels of the model, indicating a regularizing effect of WANDA pruning. We introduce a dataset of 225 harmful tasks across five categories to systematically evaluate this safety enhancement. We argue that safety improvements can be understood through a regularization perspective. First, we show that pruning helps LLMs focus more effectively on task-relevant tokens within jailbreaking prompts. Then, we analyze the effects of pruning on the perplexity of malicious prompts before and after their integration into jailbreak templates. Finally, we demonstrate statistically significant performance improvements under domain shifts when applying WANDA to linear models.

摘要: 大型语言模型(LLM)容易受到“越狱”提示的影响，这可能会导致有害内容的生成。本文证明了适度的Wanda修剪(Sun等人，2023)可以增加他们对此类攻击的抵抗力，而不需要微调，同时保持在标准基准上的性能。我们的发现表明，修剪的好处与模型的初始安全水平相关，表明万达修剪具有规律性效果。我们引入了五个类别的225个有害任务的数据集，以系统地评估这一安全增强。我们认为，安全改进可以通过正规化的观点来理解。首先，我们证明了修剪可以帮助LLM更有效地关注越狱提示中与任务相关的令牌。然后，分析了剪枝对恶意提示融入越狱模板前后的困惑程度的影响。最后，我们展示了将Wanda应用于线性模型时，在域转移情况下的统计显著性能改进。



## **38. Learnable Linguistic Watermarks for Tracing Model Extraction Attacks on Large Language Models**

用于跟踪模型提取攻击的可学习语言水印对大型语言模型 cs.CR

not decided

**SubmitDate**: 2024-04-28    [abs](http://arxiv.org/abs/2405.01509v1) [paper-pdf](http://arxiv.org/pdf/2405.01509v1)

**Authors**: Minhao Bai, Kaiyi Pang, Yongfeng Huang

**Abstract**: In the rapidly evolving domain of artificial intelligence, safeguarding the intellectual property of Large Language Models (LLMs) is increasingly crucial. Current watermarking techniques against model extraction attacks, which rely on signal insertion in model logits or post-processing of generated text, remain largely heuristic. We propose a novel method for embedding learnable linguistic watermarks in LLMs, aimed at tracing and preventing model extraction attacks. Our approach subtly modifies the LLM's output distribution by introducing controlled noise into token frequency distributions, embedding an statistically identifiable controllable watermark.We leverage statistical hypothesis testing and information theory, particularly focusing on Kullback-Leibler Divergence, to differentiate between original and modified distributions effectively. Our watermarking method strikes a delicate well balance between robustness and output quality, maintaining low false positive/negative rates and preserving the LLM's original performance.

摘要: 在快速发展的人工智能领域，保护大型语言模型(LLM)的知识产权变得越来越重要。目前针对模型提取攻击的水印技术依赖于在模型逻辑中插入信号或对生成的文本进行后处理，在很大程度上仍然是启发式的。为了跟踪和防止模型提取攻击，提出了一种在LLMS中嵌入可学习语言水印的新方法。该方法通过在令牌频率分布中引入受控噪声，嵌入一个统计上可识别的可控水印，巧妙地修改了LLM的输出分布，并利用统计假设检验和信息论，特别是Kullback-Leibler散度，有效地区分了原始分布和修改后的分布。我们的水印方法在稳健性和输出质量之间取得了微妙的平衡，既保持了较低的误检率，又保持了LLM的原始性能。



## **39. Investigating the prompt leakage effect and black-box defenses for multi-turn LLM interactions**

调查多圈LLM交互的即时泄漏效应和黑匣子防御 cs.CR

**SubmitDate**: 2024-04-26    [abs](http://arxiv.org/abs/2404.16251v2) [paper-pdf](http://arxiv.org/pdf/2404.16251v2)

**Authors**: Divyansh Agarwal, Alexander R. Fabbri, Philippe Laban, Ben Risher, Shafiq Joty, Caiming Xiong, Chien-Sheng Wu

**Abstract**: Prompt leakage in large language models (LLMs) poses a significant security and privacy threat, particularly in retrieval-augmented generation (RAG) systems. However, leakage in multi-turn LLM interactions along with mitigation strategies has not been studied in a standardized manner. This paper investigates LLM vulnerabilities against prompt leakage across 4 diverse domains and 10 closed- and open-source LLMs. Our unique multi-turn threat model leverages the LLM's sycophancy effect and our analysis dissects task instruction and knowledge leakage in the LLM response. In a multi-turn setting, our threat model elevates the average attack success rate (ASR) to 86.2%, including a 99% leakage with GPT-4 and claude-1.3. We find that some black-box LLMs like Gemini show variable susceptibility to leakage across domains - they are more likely to leak contextual knowledge in the news domain compared to the medical domain. Our experiments measure specific effects of 6 black-box defense strategies, including a query-rewriter in the RAG scenario. Our proposed multi-tier combination of defenses still has an ASR of 5.3% for black-box LLMs, indicating room for enhancement and future direction for LLM security research.

摘要: 大型语言模型中的即时泄漏对安全和隐私构成了严重威胁，尤其是在检索-增强生成(RAG)系统中。然而，多圈LLM相互作用中的泄漏以及缓解策略还没有以标准化的方式进行研究。本文研究了4个不同的域和10个封闭和开源的LLM针对即时泄漏的LLM漏洞。我们独特的多回合威胁模型利用了LLM的奉承效应，我们的分析剖析了LLM响应中的任务指令和知识泄漏。在多回合设置中，我们的威胁模型将平均攻击成功率(ASR)提高到86.2%，其中GPT-4和Claude-1.3的泄漏率为99%。我们发现，一些像双子座这样的黑盒LLM对跨域泄漏表现出可变的敏感性--与医疗领域相比，他们更有可能在新闻领域泄露上下文知识。我们的实验测量了6种黑盒防御策略的具体效果，其中包括RAG场景中的查询重写器。我们建议的多层防御组合对于黑盒LLM的ASR仍为5.3%，这表明LLM安全研究有增强的空间和未来的方向。



## **40. Don't Say No: Jailbreaking LLM by Suppressing Refusal**

不要说不：通过压制拒绝来越狱法学硕士 cs.CL

**SubmitDate**: 2024-04-25    [abs](http://arxiv.org/abs/2404.16369v1) [paper-pdf](http://arxiv.org/pdf/2404.16369v1)

**Authors**: Yukai Zhou, Wenjie Wang

**Abstract**: Ensuring the safety alignment of Large Language Models (LLMs) is crucial to generating responses consistent with human values. Despite their ability to recognize and avoid harmful queries, LLMs are vulnerable to "jailbreaking" attacks, where carefully crafted prompts elicit them to produce toxic content. One category of jailbreak attacks is reformulating the task as adversarial attacks by eliciting the LLM to generate an affirmative response. However, the typical attack in this category GCG has very limited attack success rate. In this study, to better study the jailbreak attack, we introduce the DSN (Don't Say No) attack, which prompts LLMs to not only generate affirmative responses but also novelly enhance the objective to suppress refusals. In addition, another challenge lies in jailbreak attacks is the evaluation, as it is difficult to directly and accurately assess the harmfulness of the attack. The existing evaluation such as refusal keyword matching has its own limitation as it reveals numerous false positive and false negative instances. To overcome this challenge, we propose an ensemble evaluation pipeline incorporating Natural Language Inference (NLI) contradiction assessment and two external LLM evaluators. Extensive experiments demonstrate the potency of the DSN and the effectiveness of ensemble evaluation compared to baseline methods.

摘要: 确保大型语言模型(LLM)的安全一致性对于生成与人类价值观一致的响应至关重要。尽管LLM能够识别和避免有害的查询，但它们很容易受到“越狱”攻击，在这种攻击中，精心制作的提示会诱使它们产生有毒内容。越狱攻击的一类是通过诱使LLM产生肯定的回应，将任务重新制定为对抗性攻击。然而，在这类典型攻击中，GCG的攻击成功率非常有限。在本研究中，为了更好地研究越狱攻击，我们引入了DSN(Don‘t Say No)攻击，它不仅促使LLMS产生肯定的反应，而且新颖地增强了抑制拒绝的目标。此外，越狱攻击的另一个挑战是评估，因为很难直接和准确地评估攻击的危害性。现有的拒绝关键词匹配等评价方法暴露出大量的误报和漏报实例，具有一定的局限性。为了克服这一挑战，我们提出了一种集成评估流水线，其中包括自然语言推理(NLI)矛盾评估和两个外部LLM评估器。大量实验证明了DSN的有效性和集成评估与基线方法相比的有效性。



## **41. Attacks on Third-Party APIs of Large Language Models**

对大型语言模型第三方API的攻击 cs.CR

ICLR 2024 Workshop on Secure and Trustworthy Large Language Models

**SubmitDate**: 2024-04-24    [abs](http://arxiv.org/abs/2404.16891v1) [paper-pdf](http://arxiv.org/pdf/2404.16891v1)

**Authors**: Wanru Zhao, Vidit Khazanchi, Haodi Xing, Xuanli He, Qiongkai Xu, Nicholas Donald Lane

**Abstract**: Large language model (LLM) services have recently begun offering a plugin ecosystem to interact with third-party API services. This innovation enhances the capabilities of LLMs, but it also introduces risks, as these plugins developed by various third parties cannot be easily trusted. This paper proposes a new attacking framework to examine security and safety vulnerabilities within LLM platforms that incorporate third-party services. Applying our framework specifically to widely used LLMs, we identify real-world malicious attacks across various domains on third-party APIs that can imperceptibly modify LLM outputs. The paper discusses the unique challenges posed by third-party API integration and offers strategic possibilities to improve the security and safety of LLM ecosystems moving forward. Our code is released at https://github.com/vk0812/Third-Party-Attacks-on-LLMs.

摘要: 大型语言模型（LLM）服务最近开始提供插件生态系统来与第三方API服务交互。这一创新增强了LLM的能力，但也带来了风险，因为这些由各个第三方开发的插件不容易被信任。本文提出了一种新的攻击框架来检查包含第三方服务的LLM平台内的安全和安全漏洞。通过将我们的框架专门应用于广泛使用的LLM，我们可以在第三方API上识别跨各个域的真实恶意攻击，这些攻击可以在不知不觉中修改LLM输出。该论文讨论了第三方API集成带来的独特挑战，并提供了提高LLM生态系统未来安全性的战略可能性。我们的代码在https://github.com/vk0812/Third-Party-Attacks-on-LLMs上发布。



## **42. Talk Too Much: Poisoning Large Language Models under Token Limit**

话太多：代币限制下的大型语言模型中毒 cs.CL

**SubmitDate**: 2024-04-24    [abs](http://arxiv.org/abs/2404.14795v2) [paper-pdf](http://arxiv.org/pdf/2404.14795v2)

**Authors**: Jiaming He, Wenbo Jiang, Guanyu Hou, Wenshu Fan, Rui Zhang, Hongwei Li

**Abstract**: Mainstream poisoning attacks on large language models (LLMs) typically set a fixed trigger in the input instance and specific responses for triggered queries. However, the fixed trigger setting (e.g., unusual words) may be easily detected by human detection, limiting the effectiveness and practicality in real-world scenarios. To enhance the stealthiness of the trigger, we present a poisoning attack against LLMs that is triggered by a generation/output condition-token limitation, which is a commonly adopted strategy by users for reducing costs. The poisoned model performs normally for output without token limitation, while becomes harmful for output with limited tokens. To achieve this objective, we introduce BrieFool, an efficient attack framework. It leverages the characteristics of generation limitation by efficient instruction sampling and poisoning data generation, thereby influencing the behavior of LLMs under target conditions. Our experiments demonstrate that BrieFool is effective across safety domains and knowledge domains. For instance, with only 20 generated poisoning examples against GPT-3.5-turbo, BrieFool achieves a 100% Attack Success Rate (ASR) and a 9.28/10 average Harmfulness Score (HS) under token limitation conditions while maintaining the benign performance.

摘要: 针对大型语言模型(LLM)的主流中毒攻击通常会在输入实例中设置固定的触发器，并为触发的查询设置特定的响应。然而，固定的触发设置(例如，不寻常的单词)可能很容易被人类检测到，从而限制了在现实世界场景中的有效性和实用性。为了增强触发器的隐蔽性，我们提出了一种由生成/输出条件-令牌限制触发的针对LLMS的中毒攻击，这是用户为降低成本而常用的策略。对于没有令牌限制的输出，中毒模型执行正常，而对于具有有限令牌的输出则变得有害。为了实现这一目标，我们引入了一种高效的攻击框架BrieFool。它通过高效的指令采样和中毒数据生成来利用生成限制的特性，从而影响LLMS在目标条件下的行为。我们的实验表明，BrieFool是跨安全域和知识域的有效的。例如，在对GPT-3.5-Turbo仅生成20个中毒实例的情况下，BrieFool在保持良性性能的同时，在令牌限制条件下实现了100%的攻击成功率(ASR)和9.28/10的平均危害性评分(HS)。



## **43. Large Language Models Spot Phishing Emails with Surprising Accuracy: A Comparative Analysis of Performance**

大型语言模型以惊人的准确性发现网络钓鱼电子邮件：性能比较分析 cs.CL

7 pages, 3 figures

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2404.15485v1) [paper-pdf](http://arxiv.org/pdf/2404.15485v1)

**Authors**: Het Patel, Umair Rehman, Farkhund Iqbal

**Abstract**: Phishing, a prevalent cybercrime tactic for decades, remains a significant threat in today's digital world. By leveraging clever social engineering elements and modern technology, cybercrime targets many individuals, businesses, and organizations to exploit trust and security. These cyber-attackers are often disguised in many trustworthy forms to appear as legitimate sources. By cleverly using psychological elements like urgency, fear, social proof, and other manipulative strategies, phishers can lure individuals into revealing sensitive and personalized information. Building on this pervasive issue within modern technology, this paper aims to analyze the effectiveness of 15 Large Language Models (LLMs) in detecting phishing attempts, specifically focusing on a randomized set of "419 Scam" emails. The objective is to determine which LLMs can accurately detect phishing emails by analyzing a text file containing email metadata based on predefined criteria. The experiment concluded that the following models, ChatGPT 3.5, GPT-3.5-Turbo-Instruct, and ChatGPT, were the most effective in detecting phishing emails.

摘要: 网络钓鱼是几十年来流行的一种网络犯罪策略，在当今的数字世界中仍然是一个重大威胁。通过利用聪明的社会工程元素和现代技术，网络犯罪以许多个人、企业和组织为目标，以利用信任和安全。这些网络攻击者往往以许多可信的形式伪装成合法的来源。通过巧妙地使用紧急、恐惧、社会证明和其他操纵策略等心理因素，网络钓鱼者可以诱使个人泄露敏感和个性化的信息。基于这一现代技术中普遍存在的问题，本文旨在分析15个大型语言模型(LLM)在检测网络钓鱼尝试方面的有效性，特别是关注一组随机的“419骗局”电子邮件。目标是通过基于预定义标准分析包含电子邮件元数据的文本文件，确定哪些LLM可以准确检测钓鱼电子邮件。实验得出的结论是，ChatGPT 3.5、GPT-3.5-Turbo-Indict和ChatGPT模型在检测钓鱼电子邮件方面最有效。



## **44. JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models**

越狱长凳：越狱大型语言模型的开放鲁棒性基准 cs.CR

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2404.01318v2) [paper-pdf](http://arxiv.org/pdf/2404.01318v2)

**Authors**: Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, Eric Wong

**Abstract**: Jailbreak attacks cause large language models (LLMs) to generate harmful, unethical, or otherwise objectionable content. Evaluating these attacks presents a number of challenges, which the current collection of benchmarks and evaluation techniques do not adequately address. First, there is no clear standard of practice regarding jailbreaking evaluation. Second, existing works compute costs and success rates in incomparable ways. And third, numerous works are not reproducible, as they withhold adversarial prompts, involve closed-source code, or rely on evolving proprietary APIs. To address these challenges, we introduce JailbreakBench, an open-sourced benchmark with the following components: (1) an evolving repository of state-of-the-art adversarial prompts, which we refer to as jailbreak artifacts; (2) a jailbreaking dataset comprising 100 behaviors -- both original and sourced from prior work -- which align with OpenAI's usage policies; (3) a standardized evaluation framework that includes a clearly defined threat model, system prompts, chat templates, and scoring functions; and (4) a leaderboard that tracks the performance of attacks and defenses for various LLMs. We have carefully considered the potential ethical implications of releasing this benchmark, and believe that it will be a net positive for the community. Over time, we will expand and adapt the benchmark to reflect technical and methodological advances in the research community.

摘要: 越狱攻击会导致大型语言模型(LLM)生成有害、不道德或令人反感的内容。评估这些攻击带来了许多挑战，目前收集的基准和评估技术没有充分解决这些挑战。首先，关于越狱评估没有明确的实践标准。其次，现有的工作以无与伦比的方式计算成本和成功率。第三，许多作品是不可复制的，因为它们保留了对抗性提示，涉及封闭源代码，或者依赖于不断发展的专有API。为了应对这些挑战，我们引入了JailBreak Bch，这是一款开源基准测试，具有以下组件：(1)不断发展的最新对抗性提示存储库，我们称之为越狱人工产物；(2)包含100种行为的越狱数据集，包括原始行为和源自先前工作的行为，这些行为与OpenAI的使用策略保持一致；(3)标准化评估框架，其中包括明确定义的威胁模型、系统提示、聊天模板和评分功能；以及(4)跟踪各种LLM攻击和防御性能的排行榜。我们已仔细考虑发布这一基准的潜在道德影响，并相信它将为社会带来净积极的影响。随着时间的推移，我们将扩大和调整基准，以反映研究界的技术和方法进步。



## **45. Versatile Backdoor Attack with Visible, Semantic, Sample-Specific, and Compatible Triggers**

具有可见、语义、样本特定且兼容触发器的多功能后门攻击 cs.CV

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2306.00816v3) [paper-pdf](http://arxiv.org/pdf/2306.00816v3)

**Authors**: Ruotong Wang, Hongrui Chen, Zihao Zhu, Li Liu, Baoyuan Wu

**Abstract**: Deep neural networks (DNNs) can be manipulated to exhibit specific behaviors when exposed to specific trigger patterns, without affecting their performance on benign samples, dubbed \textit{backdoor attack}. Currently, implementing backdoor attacks in physical scenarios still faces significant challenges. Physical attacks are labor-intensive and time-consuming, and the triggers are selected in a manual and heuristic way. Moreover, expanding digital attacks to physical scenarios faces many challenges due to their sensitivity to visual distortions and the absence of counterparts in the real world. To address these challenges, we define a novel trigger called the \textbf{V}isible, \textbf{S}emantic, \textbf{S}ample-Specific, and \textbf{C}ompatible (VSSC) trigger, to achieve effective, stealthy and robust simultaneously, which can also be effectively deployed in the physical scenario using corresponding objects. To implement the VSSC trigger, we propose an automated pipeline comprising three modules: a trigger selection module that systematically identifies suitable triggers leveraging large language models, a trigger insertion module that employs generative models to seamlessly integrate triggers into images, and a quality assessment module that ensures the natural and successful insertion of triggers through vision-language models. Extensive experimental results and analysis validate the effectiveness, stealthiness, and robustness of the VSSC trigger. It can not only maintain robustness under visual distortions but also demonstrates strong practicality in the physical scenario. We hope that the proposed VSSC trigger and implementation approach could inspire future studies on designing more practical triggers in backdoor attacks.

摘要: 深度神经网络(DNN)可以在暴露于特定触发模式时表现出特定的行为，而不会影响它们在良性样本上的性能，这被称为\textit{后门攻击}。目前，在物理场景中实施后门攻击仍然面临重大挑战。物理攻击劳动强度大、耗时长，触发点选择采用人工和启发式方式。此外，将数字攻击扩展到物理场景面临许多挑战，因为它们对视觉扭曲很敏感，而且现实世界中没有对应的攻击。为了应对这些挑战，我们定义了一种新的触发器，称为\Textbf{V}可扩展的、\Textbf{S}可扩展的、\Textbf{S}全特定的和\Textbf{C}兼容的(VSSC)触发器，以实现有效、隐蔽和健壮的同时，也可以使用相应的对象在物理场景中有效部署。为了实现VSSC触发器，我们提出了一个包括三个模块的自动化流水线：利用大型语言模型系统地识别合适的触发器的触发器选择模块，使用生成式模型无缝地将触发器集成到图像中的触发器插入模块，以及通过视觉语言模型确保触发器的自然和成功插入的质量评估模块。大量的实验结果和分析验证了VSSC触发器的有效性、隐蔽性和稳健性。该算法不仅能在视觉失真下保持较强的鲁棒性，而且在实际场景中表现出较强的实用性。我们希望提出的VSSC触发器和实现方法可以启发未来设计更实用的后门攻击触发器的研究。



## **46. Physical Backdoor Attack can Jeopardize Driving with Vision-Large-Language Models**

物理后门攻击可能危及使用视觉大语言模型的驾驶 cs.CR

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2404.12916v2) [paper-pdf](http://arxiv.org/pdf/2404.12916v2)

**Authors**: Zhenyang Ni, Rui Ye, Yuxi Wei, Zhen Xiang, Yanfeng Wang, Siheng Chen

**Abstract**: Vision-Large-Language-models(VLMs) have great application prospects in autonomous driving. Despite the ability of VLMs to comprehend and make decisions in complex scenarios, their integration into safety-critical autonomous driving systems poses serious security risks. In this paper, we propose BadVLMDriver, the first backdoor attack against VLMs for autonomous driving that can be launched in practice using physical objects. Unlike existing backdoor attacks against VLMs that rely on digital modifications, BadVLMDriver uses common physical items, such as a red balloon, to induce unsafe actions like sudden acceleration, highlighting a significant real-world threat to autonomous vehicle safety. To execute BadVLMDriver, we develop an automated pipeline utilizing natural language instructions to generate backdoor training samples with embedded malicious behaviors. This approach allows for flexible trigger and behavior selection, enhancing the stealth and practicality of the attack in diverse scenarios. We conduct extensive experiments to evaluate BadVLMDriver for two representative VLMs, five different trigger objects, and two types of malicious backdoor behaviors. BadVLMDriver achieves a 92% attack success rate in inducing a sudden acceleration when coming across a pedestrian holding a red balloon. Thus, BadVLMDriver not only demonstrates a critical security risk but also emphasizes the urgent need for developing robust defense mechanisms to protect against such vulnerabilities in autonomous driving technologies.

摘要: 视觉大语言模型在自动驾驶领域有着广阔的应用前景。尽管VLMS具有在复杂场景下理解和决策的能力，但它们集成到安全关键的自动驾驶系统中会带来严重的安全风险。在本文中，我们提出了BadVLMDriver，这是第一个针对自动驾驶的VLM的后门攻击，可以在实践中使用物理对象来发起。与现有的依赖于数字修改的针对VLM的后门攻击不同，BadVLMDriver使用常见的物理物品(如红色气球)来诱导突然加速等不安全行为，突显出现实世界对自动驾驶汽车安全的重大威胁。为了执行BadVLMDriver，我们开发了一个自动流水线，利用自然语言指令生成嵌入了恶意行为的后门训练样本。该方法允许灵活的触发和行为选择，增强了攻击在不同场景下的隐蔽性和实用性。我们针对两种典型的VLM、五种不同的触发器对象和两种类型的恶意后门行为，进行了大量的实验来评估BadVLMDriver。BadVLMDriver在遇到举着红色气球的行人时，诱导突然加速的攻击成功率达到92%。因此，BadVLMDriver不仅展示了一个严重的安全风险，而且还强调了迫切需要开发强大的防御机制，以防止自动驾驶技术中的此类漏洞。



## **47. Protecting Your LLMs with Information Bottleneck**

通过信息瓶颈保护您的LLC cs.CL

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2404.13968v1) [paper-pdf](http://arxiv.org/pdf/2404.13968v1)

**Authors**: Zichuan Liu, Zefan Wang, Linjie Xu, Jinyu Wang, Lei Song, Tianchun Wang, Chunlin Chen, Wei Cheng, Jiang Bian

**Abstract**: The advent of large language models (LLMs) has revolutionized the field of natural language processing, yet they might be attacked to produce harmful content. Despite efforts to ethically align LLMs, these are often fragile and can be circumvented by jailbreaking attacks through optimized or manual adversarial prompts. To address this, we introduce the Information Bottleneck Protector (IBProtector), a defense mechanism grounded in the information bottleneck principle, and we modify the objective to avoid trivial solutions. The IBProtector selectively compresses and perturbs prompts, facilitated by a lightweight and trainable extractor, preserving only essential information for the target LLMs to respond with the expected answer. Moreover, we further consider a situation where the gradient is not visible to be compatible with any LLM. Our empirical evaluations show that IBProtector outperforms current defense methods in mitigating jailbreak attempts, without overly affecting response quality or inference speed. Its effectiveness and adaptability across various attack methods and target LLMs underscore the potential of IBProtector as a novel, transferable defense that bolsters the security of LLMs without requiring modifications to the underlying models.

摘要: 大型语言模型的出现给自然语言处理领域带来了革命性的变化，但它们可能会受到攻击，产生有害的内容。尽管努力在道德上调整LLM，但这些往往是脆弱的，可以通过优化或手动对抗性提示通过越狱攻击来绕过。为了解决这个问题，我们引入了信息瓶颈保护器(IBProtector)，这是一种基于信息瓶颈原理的防御机制，我们修改了目标以避免琐碎的解决方案。IBProtector有选择地压缩和干扰提示，由一个轻量级和可训练的提取程序促进，只保留目标LLMS的基本信息，以响应预期的答案。此外，我们还进一步考虑了梯度不可见的情况，以与任何LLM相容。我们的经验评估表明，在不过度影响响应质量或推理速度的情况下，IBProtector在缓解越狱企图方面优于现有的防御方法。它对各种攻击方法和目标LLM的有效性和适应性突显了IBProtector作为一种新型、可转移的防御系统的潜力，无需修改底层模型即可增强LLM的安全性。



## **48. Typos that Broke the RAG's Back: Genetic Attack on RAG Pipeline by Simulating Documents in the Wild via Low-level Perturbations**

让RAG筋疲力尽的错别字：通过低水平扰动模拟野外文档对RAG管道进行基因攻击 cs.CL

Under Review

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2404.13948v1) [paper-pdf](http://arxiv.org/pdf/2404.13948v1)

**Authors**: Sukmin Cho, Soyeong Jeong, Jeongyeon Seo, Taeho Hwang, Jong C. Park

**Abstract**: The robustness of recent Large Language Models (LLMs) has become increasingly crucial as their applicability expands across various domains and real-world applications. Retrieval-Augmented Generation (RAG) is a promising solution for addressing the limitations of LLMs, yet existing studies on the robustness of RAG often overlook the interconnected relationships between RAG components or the potential threats prevalent in real-world databases, such as minor textual errors. In this work, we investigate two underexplored aspects when assessing the robustness of RAG: 1) vulnerability to noisy documents through low-level perturbations and 2) a holistic evaluation of RAG robustness. Furthermore, we introduce a novel attack method, the Genetic Attack on RAG (\textit{GARAG}), which targets these aspects. Specifically, GARAG is designed to reveal vulnerabilities within each component and test the overall system functionality against noisy documents. We validate RAG robustness by applying our \textit{GARAG} to standard QA datasets, incorporating diverse retrievers and LLMs. The experimental results show that GARAG consistently achieves high attack success rates. Also, it significantly devastates the performance of each component and their synergy, highlighting the substantial risk that minor textual inaccuracies pose in disrupting RAG systems in the real world.

摘要: 最近的大型语言模型(LLM)的健壮性已经变得越来越重要，因为它们的适用性在各个领域和现实世界的应用程序中扩展。检索-增强生成(RAG)是解决LLMS局限性的一种很有前途的解决方案，但现有的RAG健壮性研究往往忽略了RAG组件之间的相互关联关系或现实世界数据库中普遍存在的潜在威胁，如微小的文本错误。在这项工作中，我们研究了两个在评估RAG稳健性时未被探索的方面：1)通过低层扰动对噪声文档的脆弱性；2)RAG稳健性的整体评估。此外，我们还介绍了一种针对这些方面的新的攻击方法--对RAG的遗传攻击(\textit{garag})。具体地说，Garag旨在揭示每个组件中的漏洞，并针对嘈杂的文档测试整个系统功能。我们通过将我们的\textit{garag}应用到标准的QA数据集来验证RAG的健壮性，其中包含了不同的检索器和LLM。实验结果表明，GARAG算法始终具有较高的攻击成功率。此外，它还严重破坏了每个组件的性能及其协同作用，突显了微小的文本错误在扰乱现实世界中的RAG系统方面构成的巨大风险。



## **49. Competition Report: Finding Universal Jailbreak Backdoors in Aligned LLMs**

竞争报告：在一致的LLC中寻找通用越狱后门 cs.CL

Competition Report

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2404.14461v1) [paper-pdf](http://arxiv.org/pdf/2404.14461v1)

**Authors**: Javier Rando, Francesco Croce, Kryštof Mitka, Stepan Shabalin, Maksym Andriushchenko, Nicolas Flammarion, Florian Tramèr

**Abstract**: Large language models are aligned to be safe, preventing users from generating harmful content like misinformation or instructions for illegal activities. However, previous work has shown that the alignment process is vulnerable to poisoning attacks. Adversaries can manipulate the safety training data to inject backdoors that act like a universal sudo command: adding the backdoor string to any prompt enables harmful responses from models that, otherwise, behave safely. Our competition, co-located at IEEE SaTML 2024, challenged participants to find universal backdoors in several large language models. This report summarizes the key findings and promising ideas for future research.

摘要: 大型语言模型经过调整以确保安全，防止用户生成错误信息或非法活动指令等有害内容。然而，之前的工作表明，对齐过程很容易受到中毒攻击。对手可以操纵安全训练数据来注入类似于通用sudo命令的后门：将后门字符串添加到任何提示中都会导致模型做出有害响应，否则这些模型会安全地运行。我们的竞赛在IEEE SaTML 2024上举行，挑战参与者在几个大型语言模型中找到通用后门。本报告总结了关键发现和未来研究的有希望的想法。



## **50. Bot or Human? Detecting ChatGPT Imposters with A Single Question**

机器人还是人类？通过一个问题检测ChatGPT冒名顶替者 cs.CL

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2305.06424v3) [paper-pdf](http://arxiv.org/pdf/2305.06424v3)

**Authors**: Hong Wang, Xuan Luo, Weizhi Wang, Xifeng Yan

**Abstract**: Large language models like GPT-4 have recently demonstrated impressive capabilities in natural language understanding and generation, enabling various applications including translation, essay writing, and chit-chatting. However, there is a concern that they can be misused for malicious purposes, such as fraud or denial-of-service attacks. Therefore, it is crucial to develop methods for detecting whether the party involved in a conversation is a bot or a human. In this paper, we propose a framework named FLAIR, Finding Large Language Model Authenticity via a Single Inquiry and Response, to detect conversational bots in an online manner. Specifically, we target a single question scenario that can effectively differentiate human users from bots. The questions are divided into two categories: those that are easy for humans but difficult for bots (e.g., counting, substitution, and ASCII art reasoning), and those that are easy for bots but difficult for humans (e.g., memorization and computation). Our approach shows different strengths of these questions in their effectiveness, providing a new way for online service providers to protect themselves against nefarious activities and ensure that they are serving real users. We open-sourced our code and dataset on https://github.com/hongwang600/FLAIR and welcome contributions from the community.

摘要: 像GPT-4这样的大型语言模型最近在自然语言理解和生成方面表现出了令人印象深刻的能力，支持各种应用程序，包括翻译、论文写作和聊天。然而，人们担心它们可能被滥用于恶意目的，如欺诈或拒绝服务攻击。因此，开发方法来检测参与对话的一方是机器人还是人类是至关重要的。在本文中，我们提出了一个名为FLAIR的框架，通过单一查询和响应来发现大型语言模型的真实性，以在线方式检测会话机器人。具体地说，我们的目标是能够有效区分人类用户和机器人的单一问题场景。这些问题被分为两类：一类是对人类容易但对机器人困难的问题(例如，计数、替换和ASCII艺术推理)；另一类是对机器人容易但对人类困难的问题(例如，记忆和计算)。我们的方法显示了这些问题在有效性上的不同优势，为在线服务提供商提供了一种新的方式来保护自己免受恶意活动的影响，并确保他们服务的是真实的用户。我们在https://github.com/hongwang600/FLAIR上开源了我们的代码和数据集，并欢迎来自社区的贡献。



