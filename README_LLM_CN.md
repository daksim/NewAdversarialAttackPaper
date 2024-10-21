# Latest Large Language Model Attack Papers
**update at 2024-10-21 09:51:50**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. When LLMs Go Online: The Emerging Threat of Web-Enabled LLMs**

当LLM上线时：支持Web的LLM的新兴威胁 cs.CR

**SubmitDate**: 2024-10-18    [abs](http://arxiv.org/abs/2410.14569v1) [paper-pdf](http://arxiv.org/pdf/2410.14569v1)

**Authors**: Hanna Kim, Minkyoo Song, Seung Ho Na, Seungwon Shin, Kimin Lee

**Abstract**: Recent advancements in Large Language Models (LLMs) have established them as agentic systems capable of planning and interacting with various tools. These LLM agents are often paired with web-based tools, enabling access to diverse sources and real-time information. Although these advancements offer significant benefits across various applications, they also increase the risk of malicious use, particularly in cyberattacks involving personal information. In this work, we investigate the risks associated with misuse of LLM agents in cyberattacks involving personal data. Specifically, we aim to understand: 1) how potent LLM agents can be when directed to conduct cyberattacks, 2) how cyberattacks are enhanced by web-based tools, and 3) how affordable and easy it becomes to launch cyberattacks using LLM agents. We examine three attack scenarios: the collection of Personally Identifiable Information (PII), the generation of impersonation posts, and the creation of spear-phishing emails. Our experiments reveal the effectiveness of LLM agents in these attacks: LLM agents achieved a precision of up to 95.9% in collecting PII, up to 93.9% of impersonation posts created by LLM agents were evaluated as authentic, and the click rate for links in spear phishing emails created by LLM agents reached up to 46.67%. Additionally, our findings underscore the limitations of existing safeguards in contemporary commercial LLMs, emphasizing the urgent need for more robust security measures to prevent the misuse of LLM agents.

摘要: 大型语言模型(LLM)的最新进展已将它们确立为能够规划各种工具并与其交互的代理系统。这些LLM代理通常与基于Web的工具配合使用，从而能够访问不同的来源和实时信息。虽然这些改进在各种应用程序中提供了显著的好处，但它们也增加了恶意使用的风险，特别是在涉及个人信息的网络攻击中。在这项工作中，我们调查了在涉及个人数据的网络攻击中滥用LLM代理的相关风险。具体地说，我们的目标是了解：1)LLM代理在被指示进行网络攻击时的威力有多大；2)基于Web的工具如何增强网络攻击；3)使用LLM代理发起网络攻击变得多么负担得起和容易。我们研究了三种攻击场景：收集个人身份信息(PII)、生成模拟帖子和创建鱼叉式网络钓鱼电子邮件。我们的实验显示了LLM代理在这些攻击中的有效性：LLM代理收集PII的准确率高达95.9%，LLM代理创建的模仿帖子被评估为可信的高达93.9%，LLM代理创建的鱼叉式钓鱼邮件中链接的点击率高达46.67%。此外，我们的研究结果强调了当代商业LLM现有保障措施的局限性，强调迫切需要采取更强有力的安全措施，以防止滥用LLM剂。



## **2. BlackDAN: A Black-Box Multi-Objective Approach for Effective and Contextual Jailbreaking of Large Language Models**

BlackDAN：一种有效且上下文化的大型语言模型越狱的黑匣子多目标方法 cs.CR

**SubmitDate**: 2024-10-18    [abs](http://arxiv.org/abs/2410.09804v2) [paper-pdf](http://arxiv.org/pdf/2410.09804v2)

**Authors**: Xinyuan Wang, Victor Shea-Jay Huang, Renmiao Chen, Hao Wang, Chengwei Pan, Lei Sha, Minlie Huang

**Abstract**: While large language models (LLMs) exhibit remarkable capabilities across various tasks, they encounter potential security risks such as jailbreak attacks, which exploit vulnerabilities to bypass security measures and generate harmful outputs. Existing jailbreak strategies mainly focus on maximizing attack success rate (ASR), frequently neglecting other critical factors, including the relevance of the jailbreak response to the query and the level of stealthiness. This narrow focus on single objectives can result in ineffective attacks that either lack contextual relevance or are easily recognizable. In this work, we introduce BlackDAN, an innovative black-box attack framework with multi-objective optimization, aiming to generate high-quality prompts that effectively facilitate jailbreaking while maintaining contextual relevance and minimizing detectability. BlackDAN leverages Multiobjective Evolutionary Algorithms (MOEAs), specifically the NSGA-II algorithm, to optimize jailbreaks across multiple objectives including ASR, stealthiness, and semantic relevance. By integrating mechanisms like mutation, crossover, and Pareto-dominance, BlackDAN provides a transparent and interpretable process for generating jailbreaks. Furthermore, the framework allows customization based on user preferences, enabling the selection of prompts that balance harmfulness, relevance, and other factors. Experimental results demonstrate that BlackDAN outperforms traditional single-objective methods, yielding higher success rates and improved robustness across various LLMs and multimodal LLMs, while ensuring jailbreak responses are both relevant and less detectable.

摘要: 虽然大型语言模型(LLM)在各种任务中显示出非凡的能力，但它们遇到了潜在的安全风险，如越狱攻击，这些攻击利用漏洞绕过安全措施并产生有害的输出。现有的越狱策略主要关注最大化攻击成功率(ASR)，往往忽略了其他关键因素，包括越狱响应与查询的相关性和隐蔽性水平。这种对单一目标的狭隘关注可能会导致无效的攻击，要么缺乏上下文相关性，要么很容易识别。在这项工作中，我们引入了BlackDAN，一个创新的多目标优化的黑盒攻击框架，旨在生成高质量的提示，在保持上下文相关性的同时有效地促进越狱，并将可检测性降至最低。BlackDAN利用多目标进化算法(MOEA)，特别是NSGA-II算法，跨多个目标优化越狱，包括ASR、隐蔽性和语义相关性。通过集成变异、交叉和帕累托支配等机制，BlackDAN为生成越狱提供了一个透明和可解释的过程。此外，该框架允许根据用户偏好进行定制，从而能够选择在危害性、相关性和其他因素之间进行权衡的提示。实验结果表明，BlackDAN的性能优于传统的单目标方法，在各种LLM和多模式LLM上获得了更高的成功率和更好的鲁棒性，同时确保了越狱响应的相关性和较低的可检测性。



## **3. Backdoored Retrievers for Prompt Injection Attacks on Retrieval Augmented Generation of Large Language Models**

用于对大型语言模型的检索增强生成的提示注入攻击的后门检索器 cs.CR

12 pages, 5 figures

**SubmitDate**: 2024-10-18    [abs](http://arxiv.org/abs/2410.14479v1) [paper-pdf](http://arxiv.org/pdf/2410.14479v1)

**Authors**: Cody Clop, Yannick Teglia

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in generating coherent text but remain limited by the static nature of their training data. Retrieval Augmented Generation (RAG) addresses this issue by combining LLMs with up-to-date information retrieval, but also expand the attack surface of the system. This paper investigates prompt injection attacks on RAG, focusing on malicious objectives beyond misinformation, such as inserting harmful links, promoting unauthorized services, and initiating denial-of-service behaviors. We build upon existing corpus poisoning techniques and propose a novel backdoor attack aimed at the fine-tuning process of the dense retriever component. Our experiments reveal that corpus poisoning can achieve significant attack success rates through the injection of a small number of compromised documents into the retriever corpus. In contrast, backdoor attacks demonstrate even higher success rates but necessitate a more complex setup, as the victim must fine-tune the retriever using the attacker poisoned dataset.

摘要: 大型语言模型(LLM)在生成连贯的文本方面表现出非凡的能力，但仍然受到其训练数据静态性质的限制。检索增强生成(RAG)通过将LLMS与最新的信息检索相结合来解决这一问题，但也扩展了系统的攻击面。本文研究了RAG上的即时注入攻击，重点针对错误信息以外的恶意目标，如插入有害链接、推广未经授权的服务和发起拒绝服务行为。我们在现有的语料库中毒技术的基础上，针对密集检索组件的微调过程，提出了一种新的后门攻击。我们的实验表明，通过向检索器语料库中注入少量受攻击的文档，语料库中毒可以获得显着的攻击成功率。相比之下，后门攻击显示出更高的成功率，但需要更复杂的设置，因为受害者必须使用攻击者有毒的数据集来微调检索器。



## **4. Unlearning Backdoor Attacks for LLMs with Weak-to-Strong Knowledge Distillation**

通过弱到强的知识蒸馏消除LLM的后门攻击 cs.CL

**SubmitDate**: 2024-10-18    [abs](http://arxiv.org/abs/2410.14425v1) [paper-pdf](http://arxiv.org/pdf/2410.14425v1)

**Authors**: Shuai Zhao, Xiaobao Wu, Cong-Duy Nguyen, Meihuizi Jia, Yichao Feng, Luu Anh Tuan

**Abstract**: Parameter-efficient fine-tuning (PEFT) can bridge the gap between large language models (LLMs) and downstream tasks. However, PEFT has been proven vulnerable to malicious attacks. Research indicates that poisoned LLMs, even after PEFT, retain the capability to activate internalized backdoors when input samples contain predefined triggers. In this paper, we introduce a novel weak-to-strong unlearning algorithm to defend against backdoor attacks based on feature alignment knowledge distillation, named W2SDefense. Specifically, we first train a small-scale language model through full-parameter fine-tuning to serve as the clean teacher model. Then, this teacher model guides the large-scale poisoned student model in unlearning the backdoor, leveraging PEFT. Theoretical analysis suggests that W2SDefense has the potential to enhance the student model's ability to unlearn backdoor features, preventing the activation of the backdoor. We conduct experiments on text classification tasks involving three state-of-the-art language models and three different backdoor attack algorithms. Our empirical results demonstrate the outstanding performance of W2SDefense in defending against backdoor attacks without compromising model performance.

摘要: 参数高效微调(PEFT)可以弥合大型语言模型(LLM)和下游任务之间的差距。然而，PEFT已被证明容易受到恶意攻击。研究表明，中毒的LLM，即使在PEFT之后，当输入样本包含预定义的触发器时，仍保持激活内部化后门的能力。本文提出了一种新的基于特征对齐知识提取的弱到强遗忘算法W2SDefense来防御后门攻击。具体来说，我们首先通过全参数微调来训练一个小规模的语言模型，作为廉洁教师模型。然后，这个教师模型引导大规模中毒学生模型忘记后门，利用PEFT。理论分析表明，W2SDefense有可能增强学生模型忘记后门功能的能力，防止激活后门。我们对三种最新的语言模型和三种不同的后门攻击算法进行了文本分类实验。我们的实验结果表明，W2SDefense在不影响模型性能的情况下，在防御后门攻击方面具有出色的性能。



## **5. VLFeedback: A Large-Scale AI Feedback Dataset for Large Vision-Language Models Alignment**

VLFeedback：用于大型视觉语言模型对齐的大规模人工智能反馈数据集 cs.CV

EMNLP 2024 Main Conference camera-ready version (fixed small typos).  This article supersedes arXiv:2312.10665

**SubmitDate**: 2024-10-18    [abs](http://arxiv.org/abs/2410.09421v2) [paper-pdf](http://arxiv.org/pdf/2410.09421v2)

**Authors**: Lei Li, Zhihui Xie, Mukai Li, Shunian Chen, Peiyi Wang, Liang Chen, Yazheng Yang, Benyou Wang, Lingpeng Kong, Qi Liu

**Abstract**: As large vision-language models (LVLMs) evolve rapidly, the demand for high-quality and diverse data to align these models becomes increasingly crucial. However, the creation of such data with human supervision proves costly and time-intensive. In this paper, we investigate the efficacy of AI feedback to scale supervision for aligning LVLMs. We introduce VLFeedback, the first large-scale vision-language feedback dataset, comprising over 82K multi-modal instructions and comprehensive rationales generated by off-the-shelf models without human annotations. To evaluate the effectiveness of AI feedback for vision-language alignment, we train Silkie, an LVLM fine-tuned via direct preference optimization on VLFeedback. Silkie showcases exceptional performance regarding helpfulness, visual faithfulness, and safety metrics. It outperforms its base model by 6.9\% and 9.5\% in perception and cognition tasks, reduces hallucination issues on MMHal-Bench, and exhibits enhanced resilience against red-teaming attacks. Furthermore, our analysis underscores the advantage of AI feedback, particularly in fostering preference diversity to deliver more comprehensive improvements. Our dataset, training code and models are available at https://vlf-silkie.github.io.

摘要: 随着大型视觉语言模型(LVLM)的快速发展，对高质量和多样化数据的需求变得越来越重要。然而，事实证明，在人工监督下创建此类数据既昂贵又耗时。在本文中，我们研究了人工智能反馈对比例尺监督对准LVLM的有效性。我们介绍了VLFeedback，这是第一个大规模的视觉语言反馈数据集，包括超过82K的多模式指令和由没有人工注释的现成模型生成的全面原理。为了评估人工智能反馈对视觉-语言对齐的有效性，我们对Silkie进行了训练，这是一种通过对VLFeedback进行直接偏好优化而微调的LVLM。Silkie展示了在帮助、视觉忠诚度和安全指标方面的出色表现。它在感知和认知任务中的表现分别比基本模型高出6.9%和9.5%，减少了MMHal-BENCH上的幻觉问题，并表现出对红队攻击的增强的弹性。此外，我们的分析强调了人工智能反馈的优势，特别是在促进偏好多样性以提供更全面的改进方面。我们的数据集、训练代码和模型可在https://vlf-silkie.github.io.上获得



## **6. DomainLynx: Leveraging Large Language Models for Enhanced Domain Squatting Detection**

DomainLynx：利用大型语言模型进行增强的域蹲位检测 cs.CR

Accepted for publication at IEEE CCNC 2025

**SubmitDate**: 2024-10-18    [abs](http://arxiv.org/abs/2410.02095v2) [paper-pdf](http://arxiv.org/pdf/2410.02095v2)

**Authors**: Daiki Chiba, Hiroki Nakano, Takashi Koide

**Abstract**: Domain squatting poses a significant threat to Internet security, with attackers employing increasingly sophisticated techniques. This study introduces DomainLynx, an innovative compound AI system leveraging Large Language Models (LLMs) for enhanced domain squatting detection. Unlike existing methods focusing on predefined patterns for top-ranked domains, DomainLynx excels in identifying novel squatting techniques and protecting less prominent brands. The system's architecture integrates advanced data processing, intelligent domain pairing, and LLM-powered threat assessment. Crucially, DomainLynx incorporates specialized components that mitigate LLM hallucinations, ensuring reliable and context-aware detection. This approach enables efficient analysis of vast security data from diverse sources, including Certificate Transparency logs, Passive DNS records, and zone files. Evaluated on a curated dataset of 1,649 squatting domains, DomainLynx achieved 94.7\% accuracy using Llama-3-70B. In a month-long real-world test, it detected 34,359 squatting domains from 2.09 million new domains, outperforming baseline methods by 2.5 times. This research advances Internet security by providing a versatile, accurate, and adaptable tool for combating evolving domain squatting threats. DomainLynx's approach paves the way for more robust, AI-driven cybersecurity solutions, enhancing protection for a broader range of online entities and contributing to a safer digital ecosystem.

摘要: 随着攻击者使用越来越复杂的技术，域名抢占对互联网安全构成了重大威胁。这项研究介绍了DomainLynx，这是一个创新的复合人工智能系统，利用大型语言模型(LLM)来增强域占用检测。与专注于排名靠前的域名的预定义模式的现有方法不同，DomainLynx在识别新颖的蹲守技术和保护不太知名的品牌方面表现出色。该系统的体系结构集成了先进的数据处理、智能域配对和LLM支持的威胁评估。至关重要的是，DomainLynx结合了专门的组件来缓解LLM幻觉，确保可靠和上下文感知的检测。这种方法可以有效地分析来自不同来源的大量安全数据，包括证书透明日志、被动DNS记录和区域文件。在1,649个蹲点域的精选数据集上进行评估，DomainLynx使用LLAMA-3-70B获得了94.7%的准确率。在一个月的实际测试中，它从209万个新域名中检测到34359个蹲点域名，比基线方法高出2.5倍。这项研究通过提供一种通用、准确和适应性强的工具来对抗不断变化的域名抢占威胁，从而促进了互联网安全。DomainLynx的方法为更强大的、人工智能驱动的网络安全解决方案铺平了道路，加强了对更广泛的在线实体的保护，并为更安全的数字生态系统做出了贡献。



## **7. PopAlign: Diversifying Contrasting Patterns for a More Comprehensive Alignment**

PopAlign：多样化对比模式以实现更全面的一致 cs.CL

28 pages

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13785v1) [paper-pdf](http://arxiv.org/pdf/2410.13785v1)

**Authors**: Zekun Moore Wang, Shawn Wang, Kang Zhu, Jiaheng Liu, Ke Xu, Jie Fu, Wangchunshu Zhou, Wenhao Huang

**Abstract**: Alignment of large language models (LLMs) involves training models on preference-contrastive output pairs to adjust their responses according to human preferences. To obtain such contrastive pairs, traditional methods like RLHF and RLAIF rely on limited contrasting patterns, such as varying model variants or decoding temperatures. This singularity leads to two issues: (1) alignment is not comprehensive; and thereby (2) models are susceptible to jailbreaking attacks. To address these issues, we investigate how to construct more comprehensive and diversified contrasting patterns to enhance preference data (RQ1) and verify the impact of the diversification of contrasting patterns on model alignment (RQ2). For RQ1, we propose PopAlign, a framework that integrates diversified contrasting patterns across the prompt, model, and pipeline levels, introducing six contrasting strategies that do not require additional feedback labeling procedures. Regarding RQ2, we conduct thorough experiments demonstrating that PopAlign significantly outperforms existing methods, leading to more comprehensive alignment.

摘要: 大语言模型的对齐涉及对偏好-对比输出对的训练，以根据人的偏好调整其反应。为了获得这样的对比对，像RLHF和RLAIF这样的传统方法依赖于有限的对比模式，例如变化的模型变量或解码温度。这种奇异性导致了两个问题：(1)对齐不全面；因此(2)模型容易受到越狱攻击。为了解决这些问题，我们研究了如何构建更全面、更多样化的对比模式来增强偏好数据(RQ1)，并验证对比模式的多样化对模型对齐(RQ2)的影响。对于RQ1，我们提出了PopAlign，这是一个框架，集成了提示、模型和管道级别的各种对比模式，引入了六种不需要额外反馈标签程序的对比策略。对于RQ2，我们进行了深入的实验，证明了PopAlign的性能明显优于现有的方法，导致了更全面的比对。



## **8. Persistent Pre-Training Poisoning of LLMs**

LLM训练前持续中毒 cs.CR

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13722v1) [paper-pdf](http://arxiv.org/pdf/2410.13722v1)

**Authors**: Yiming Zhang, Javier Rando, Ivan Evtimov, Jianfeng Chi, Eric Michael Smith, Nicholas Carlini, Florian Tramèr, Daphne Ippolito

**Abstract**: Large language models are pre-trained on uncurated text datasets consisting of trillions of tokens scraped from the Web. Prior work has shown that: (1) web-scraped pre-training datasets can be practically poisoned by malicious actors; and (2) adversaries can compromise language models after poisoning fine-tuning datasets. Our work evaluates for the first time whether language models can also be compromised during pre-training, with a focus on the persistence of pre-training attacks after models are fine-tuned as helpful and harmless chatbots (i.e., after SFT and DPO). We pre-train a series of LLMs from scratch to measure the impact of a potential poisoning adversary under four different attack objectives (denial-of-service, belief manipulation, jailbreaking, and prompt stealing), and across a wide range of model sizes (from 600M to 7B). Our main result is that poisoning only 0.1% of a model's pre-training dataset is sufficient for three out of four attacks to measurably persist through post-training. Moreover, simple attacks like denial-of-service persist through post-training with a poisoning rate of only 0.001%.

摘要: 大型语言模型是在未经精选的文本数据集上预先训练的，这些数据集由从Web上刮来的数万亿个标记组成。先前的工作表明：(1)网络刮来的预训练数据集实际上可能会被恶意行为者毒化；(2)攻击者在毒化微调数据集后可能会危害语言模型。我们的工作首次评估了语言模型在预训练期间是否也会被破坏，重点是在模型被微调为有帮助和无害的聊天机器人后(即在SFT和DPO之后)，预训练攻击的持久性。我们从头开始预先训练一系列LLM，以衡量潜在中毒对手在四种不同攻击目标(拒绝服务、信念操纵、越狱和即时盗窃)下的影响，并跨越广泛的模型大小(从600M到7B)。我们的主要结果是，只有0.1%的模型训练前数据集的中毒足以使四分之三的攻击在训练后可测量地持续存在。此外，像拒绝服务这样的简单攻击在培训后持续存在，投毒率仅为0.001%。



## **9. Jailbreaking LLM-Controlled Robots**

越狱LLM控制机器人 cs.RO

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13691v1) [paper-pdf](http://arxiv.org/pdf/2410.13691v1)

**Authors**: Alexander Robey, Zachary Ravichandran, Vijay Kumar, Hamed Hassani, George J. Pappas

**Abstract**: The recent introduction of large language models (LLMs) has revolutionized the field of robotics by enabling contextual reasoning and intuitive human-robot interaction in domains as varied as manipulation, locomotion, and self-driving vehicles. When viewed as a stand-alone technology, LLMs are known to be vulnerable to jailbreaking attacks, wherein malicious prompters elicit harmful text by bypassing LLM safety guardrails. To assess the risks of deploying LLMs in robotics, in this paper, we introduce RoboPAIR, the first algorithm designed to jailbreak LLM-controlled robots. Unlike existing, textual attacks on LLM chatbots, RoboPAIR elicits harmful physical actions from LLM-controlled robots, a phenomenon we experimentally demonstrate in three scenarios: (i) a white-box setting, wherein the attacker has full access to the NVIDIA Dolphins self-driving LLM, (ii) a gray-box setting, wherein the attacker has partial access to a Clearpath Robotics Jackal UGV robot equipped with a GPT-4o planner, and (iii) a black-box setting, wherein the attacker has only query access to the GPT-3.5-integrated Unitree Robotics Go2 robot dog. In each scenario and across three new datasets of harmful robotic actions, we demonstrate that RoboPAIR, as well as several static baselines, finds jailbreaks quickly and effectively, often achieving 100% attack success rates. Our results reveal, for the first time, that the risks of jailbroken LLMs extend far beyond text generation, given the distinct possibility that jailbroken robots could cause physical damage in the real world. Indeed, our results on the Unitree Go2 represent the first successful jailbreak of a deployed commercial robotic system. Addressing this emerging vulnerability is critical for ensuring the safe deployment of LLMs in robotics. Additional media is available at: https://robopair.org

摘要: 最近引入的大型语言模型(LLM)通过在操作、运动和自动驾驶车辆等各种领域实现上下文推理和直观的人-机器人交互，从而彻底改变了机器人领域。当被视为一项独立的技术时，LLMS已知容易受到越狱攻击，恶意提示器通过绕过LLm安全护栏引发有害文本。为了评估在机器人学中部署LLMS的风险，在本文中，我们引入了RoboPAIR，这是第一个设计用于越狱LLM控制的机器人的算法。与现有对LLM聊天机器人的文本攻击不同，RoboPAIR会引发来自LLM控制的机器人的有害物理操作，我们在三个场景中实验演示了这种现象：(I)白盒设置，其中攻击者对NVIDIA Dolphins自动驾驶LLM具有完全访问权限；(Ii)灰盒设置，其中攻击者对配备GPT-40规划器的ClearPath Robotics Jackal UGV机器人具有部分访问权限；以及(Iii)黑盒设置，其中攻击者只有对GPT-3.5集成的Unitree Robotics Go2机器狗的查询访问权限。在每个场景和三个新的有害机器人操作的数据集上，我们展示了RoboPAIR以及几个静态基线，快速有效地找到越狱，通常达到100%的攻击成功率。我们的结果首次显示，鉴于越狱机器人在现实世界中造成物理损害的明显可能性，越狱机器人的风险远远超出了文本生成的范围。事实上，我们在Unitree Go2上的结果代表着部署的商业机器人系统第一次成功越狱。解决这一新出现的漏洞对于确保在机器人中安全部署LLM至关重要。如需更多媒体，请访问：https://robopair.org。



## **10. Targeted Vaccine: Safety Alignment for Large Language Models against Harmful Fine-Tuning via Layer-wise Perturbation**

有针对性的疫苗：大型语言模型的安全调整，防止通过分层扰动进行有害的微调 cs.LG

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.09760v2) [paper-pdf](http://arxiv.org/pdf/2410.09760v2)

**Authors**: Guozhi Liu, Weiwei Lin, Tiansheng Huang, Ruichao Mo, Qi Mu, Li Shen

**Abstract**: Harmful fine-tuning attack poses a serious threat to the online fine-tuning service. Vaccine, a recent alignment-stage defense, applies uniform perturbation to all layers of embedding to make the model robust to the simulated embedding drift. However, applying layer-wise uniform perturbation may lead to excess perturbations for some particular safety-irrelevant layers, resulting in defense performance degradation and unnecessary memory consumption. To address this limitation, we propose Targeted Vaccine (T-Vaccine), a memory-efficient safety alignment method that applies perturbation to only selected layers of the model. T-Vaccine follows two core steps: First, it uses gradient norm as a statistical metric to identify the safety-critical layers. Second, instead of applying uniform perturbation across all layers, T-Vaccine only applies perturbation to the safety-critical layers while keeping other layers frozen during training. Results show that T-Vaccine outperforms Vaccine in terms of both defense effectiveness and resource efficiency. Comparison with other defense baselines, e.g., RepNoise and TAR also demonstrate the superiority of T-Vaccine. Notably, T-Vaccine is the first defense that can address harmful fine-tuning issues for a 7B pre-trained models trained on consumer GPUs with limited memory (e.g., RTX 4090). Our code is available at https://github.com/Lslland/T-Vaccine.

摘要: 有害微调攻击对在线微调服务构成严重威胁。疫苗是最近的一种对齐阶段防御方法，它将均匀扰动应用于嵌入的所有层，以使模型对模拟的嵌入漂移具有鲁棒性。然而，分层均匀扰动可能会导致某些特定安全无关层的过度扰动，导致防御性能下降和不必要的内存消耗。为了解决这一局限性，我们提出了靶向疫苗(T-Vaccine)，这是一种内存高效的安全对齐方法，仅对模型的选定层应用扰动。T-Vaccine遵循两个核心步骤：首先，它使用梯度范数作为统计度量来识别安全关键层。其次，T-Vaccine不是在所有层上应用统一的扰动，而是只对安全关键层应用扰动，而在训练期间保持其他层的冻结。结果表明，无论是防御效果还是资源效率，T疫苗都优于疫苗。与其他防御基线如RepNoise和TAR的比较也证明了T-疫苗的优越性。值得注意的是，T-Vaccine是第一个可以解决7B预培训模型的有害微调问题的防御系统，这些模型在内存有限的消费者GPU(例如RTX 4090)上进行了培训。我们的代码可以在https://github.com/Lslland/T-Vaccine.上找到



## **11. SCA: Highly Efficient Semantic-Consistent Unrestricted Adversarial Attack**

SCA：高效语义一致的无限制对抗攻击 cs.CV

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.02240v3) [paper-pdf](http://arxiv.org/pdf/2410.02240v3)

**Authors**: Zihao Pan, Weibin Wu, Yuhang Cao, Zibin Zheng

**Abstract**: Deep neural network based systems deployed in sensitive environments are vulnerable to adversarial attacks. Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic. Recent works have utilized the diffusion inversion process to map images into a latent space, where high-level semantics are manipulated by introducing perturbations. However, they often results in substantial semantic distortions in the denoised output and suffers from low efficiency. In this study, we propose a novel framework called Semantic-Consistent Unrestricted Adversarial Attacks (SCA), which employs an inversion method to extract edit-friendly noise maps and utilizes Multimodal Large Language Model (MLLM) to provide semantic guidance throughout the process. Under the condition of rich semantic information provided by MLLM, we perform the DDPM denoising process of each step using a series of edit-friendly noise maps, and leverage DPM Solver++ to accelerate this process, enabling efficient sampling with semantic consistency. Compared to existing methods, our framework enables the efficient generation of adversarial examples that exhibit minimal discernible semantic changes. Consequently, we for the first time introduce Semantic-Consistent Adversarial Examples (SCAE). Extensive experiments and visualizations have demonstrated the high efficiency of SCA, particularly in being on average 12 times faster than the state-of-the-art attacks. Our research can further draw attention to the security of multimedia information.

摘要: 部署在敏感环境中的基于深度神经网络的系统容易受到敌意攻击。不受限制的对抗性攻击通常操纵图像的语义内容(例如，颜色或纹理)以创建既有效又逼真的对抗性示例。最近的工作利用扩散逆过程将图像映射到潜在空间，在潜在空间中通过引入扰动来操纵高级语义。然而，它们往往会在去噪输出中造成严重的语义扭曲，并导致效率低下。在这项研究中，我们提出了一种新的框架，称为语义一致的无限对抗攻击(SCA)，它使用一种反转方法来提取编辑友好的噪声映射，并利用多模式大语言模型(MLLM)在整个过程中提供语义指导。在MLLM提供丰富语义信息的条件下，使用一系列编辑友好的噪声图对每个步骤进行DDPM去噪处理，并利用DPM Solver++加速这一过程，从而实现高效的语义一致性采样。与现有的方法相比，我们的框架能够高效地生成对抗性的例子，这些例子表现出最小的可识别的语义变化。因此，我们首次引入了语义一致的对抗性例子(SCAE)。广泛的实验和可视化已经证明了SCA的高效率，特别是在平均速度上是最先进的攻击的12倍。我们的研究可以进一步引起人们对多媒体信息安全的关注。



## **12. Breaking Chains: Unraveling the Links in Multi-Hop Knowledge Unlearning**

打破链条：解开多跳知识遗忘中的链接 cs.CL

16 pages, 5 figures

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13274v1) [paper-pdf](http://arxiv.org/pdf/2410.13274v1)

**Authors**: Minseok Choi, ChaeHun Park, Dohyun Lee, Jaegul Choo

**Abstract**: Large language models (LLMs) serve as giant information stores, often including personal or copyrighted data, and retraining them from scratch is not a viable option. This has led to the development of various fast, approximate unlearning techniques to selectively remove knowledge from LLMs. Prior research has largely focused on minimizing the probabilities of specific token sequences by reversing the language modeling objective. However, these methods still leave LLMs vulnerable to adversarial attacks that exploit indirect references. In this work, we examine the limitations of current unlearning techniques in effectively erasing a particular type of indirect prompt: multi-hop queries. Our findings reveal that existing methods fail to completely remove multi-hop knowledge when one of the intermediate hops is unlearned. To address this issue, we propose MUNCH, a simple uncertainty-based approach that breaks down multi-hop queries into subquestions and leverages the uncertainty of the unlearned model in final decision-making. Empirical results demonstrate the effectiveness of our framework, and MUNCH can be easily integrated with existing unlearning techniques, making it a flexible and useful solution for enhancing unlearning processes.

摘要: 大型语言模型(LLM)充当了巨大的信息存储，通常包括个人或受版权保护的数据，从零开始对它们进行再培训并不是一个可行的选择。这导致了各种快速、近似的遗忘技术的发展，以选择性地从LLM中移除知识。以前的研究主要集中在通过颠倒语言建模目标来最小化特定标记序列的概率。然而，这些方法仍然使LLM容易受到利用间接引用的敌意攻击。在这项工作中，我们检查了当前遗忘技术在有效消除一种特定类型的间接提示：多跳查询方面的局限性。我们的发现表明，当中间跳之一未被学习时，现有方法无法完全消除多跳知识。为了解决这个问题，我们提出了Munch，一种简单的基于不确定性的方法，将多跳查询分解为子问题，并在最终决策中利用未学习模型的不确定性。实验结果表明我们的框架是有效的，而且Munch可以很容易地与现有的遗忘技术相结合，使其成为一种灵活而有用的解决方案来增强遗忘过程。



## **13. FRAG: Toward Federated Vector Database Management for Collaborative and Secure Retrieval-Augmented Generation**

FRAG：迈向联合载体数据库管理，以实现协作和安全的检索增强生成 cs.CR

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13272v1) [paper-pdf](http://arxiv.org/pdf/2410.13272v1)

**Authors**: Dongfang Zhao

**Abstract**: This paper introduces \textit{Federated Retrieval-Augmented Generation (FRAG)}, a novel database management paradigm tailored for the growing needs of retrieval-augmented generation (RAG) systems, which are increasingly powered by large-language models (LLMs). FRAG enables mutually-distrusted parties to collaboratively perform Approximate $k$-Nearest Neighbor (ANN) searches on encrypted query vectors and encrypted data stored in distributed vector databases, all while ensuring that no party can gain any knowledge about the queries or data of others. Achieving this paradigm presents two key challenges: (i) ensuring strong security guarantees, such as Indistinguishability under Chosen-Plaintext Attack (IND-CPA), under practical assumptions (e.g., we avoid overly optimistic assumptions like non-collusion among parties); and (ii) maintaining performance overheads comparable to traditional, non-federated RAG systems. To address these challenges, FRAG employs a single-key homomorphic encryption protocol that simplifies key management across mutually-distrusted parties. Additionally, FRAG introduces a \textit{multiplicative caching} technique to efficiently encrypt floating-point numbers, significantly improving computational performance in large-scale federated environments. We provide a rigorous security proof using standard cryptographic reductions and demonstrate the practical scalability and efficiency of FRAG through extensive experiments on both benchmark and real-world datasets.

摘要: 本文介绍了一种新的数据库管理范例--联合检索-扩充生成(FRAG)，它是为日益增长的大型语言模型(LLM)支持的检索-扩充生成(RAG)系统的需求而定制的。FRAG使相互不信任的各方能够协作地对加密的查询向量和存储在分布式向量数据库中的加密数据执行大约$k$最近邻(ANN)搜索，同时确保任何一方都无法获得关于其他人的查询或数据的任何知识。实现这一范例有两个关键挑战：(I)确保强大的安全保证，例如在实际假设下的选择明文攻击(IND-CPA)下的不可区分(IND-CPA)；以及(Ii)保持与传统的非联邦RAG系统相当的性能开销。为了应对这些挑战，FRAG采用了单密钥同态加密协议，简化了相互不信任的各方之间的密钥管理。此外，FRAG还引入了一种\textit{乘法缓存}技术来高效地加密浮点数，从而显著提高了大规模联邦环境中的计算性能。我们使用标准密码约简提供了严格的安全证明，并通过在基准数据集和真实数据集上的广泛实验证明了FRAG的实用可扩展性和效率。



## **14. Large Language Models are Easily Confused: A Quantitative Metric, Security Implications and Typological Analysis**

大型语言模型很容易混淆：量化指标、安全含义和类型学分析 cs.CL

17 pages, 6 figures, 14 tables

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13237v1) [paper-pdf](http://arxiv.org/pdf/2410.13237v1)

**Authors**: Yiyi Chen, Qiongxiu Li, Russa Biswas, Johannes Bjerva

**Abstract**: Language Confusion is a phenomenon where Large Language Models (LLMs) generate text that is neither in the desired language, nor in a contextually appropriate language. This phenomenon presents a critical challenge in text generation by LLMs, often appearing as erratic and unpredictable behavior. We hypothesize that there are linguistic regularities to this inherent vulnerability in LLMs and shed light on patterns of language confusion across LLMs. We introduce a novel metric, Language Confusion Entropy, designed to directly measure and quantify this confusion, based on language distributions informed by linguistic typology and lexical variation. Comprehensive comparisons with the Language Confusion Benchmark (Marchisio et al., 2024) confirm the effectiveness of our metric, revealing patterns of language confusion across LLMs. We further link language confusion to LLM security, and find patterns in the case of multilingual embedding inversion attacks. Our analysis demonstrates that linguistic typology offers theoretically grounded interpretation, and valuable insights into leveraging language similarities as a prior for LLM alignment and security.

摘要: 语言混淆是一种现象，大型语言模型(LLM)生成的文本既不是所需语言的文本，也不是上下文合适的语言文本。这一现象对LLMS的文本生成提出了严重的挑战，通常表现为不稳定和不可预测的行为。我们假设LLMS中这种固有的脆弱性存在语言规则，并揭示了LLMS中语言混淆的模式。我们引入了一种新的度量，语言混淆熵，旨在根据语言类型和词汇变异提供的语言分布来直接度量和量化这种混淆。与语言混淆基准(Marchisio等人，2024年)的全面比较证实了我们的度量的有效性，揭示了LLM之间的语言混淆模式。我们进一步将语言混淆与LLM安全联系起来，并在多语言嵌入反转攻击的情况下找到了模式。我们的分析表明，语言类型学提供了理论上的解释，并提供了关于利用语言相似性作为LLM对齐和安全的先决条件的有价值的见解。



## **15. SPIN: Self-Supervised Prompt INjection**

旋转：自我监督的即时注射 cs.CL

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13236v1) [paper-pdf](http://arxiv.org/pdf/2410.13236v1)

**Authors**: Leon Zhou, Junfeng Yang, Chengzhi Mao

**Abstract**: Large Language Models (LLMs) are increasingly used in a variety of important applications, yet their safety and reliability remain as major concerns. Various adversarial and jailbreak attacks have been proposed to bypass the safety alignment and cause the model to produce harmful responses. We introduce Self-supervised Prompt INjection (SPIN) which can detect and reverse these various attacks on LLMs. As our self-supervised prompt defense is done at inference-time, it is also compatible with existing alignment and adds an additional layer of safety for defense. Our benchmarks demonstrate that our system can reduce the attack success rate by up to 87.9%, while maintaining the performance on benign user requests. In addition, we discuss the situation of an adaptive attacker and show that our method is still resilient against attackers who are aware of our defense.

摘要: 大型语言模型（LLM）越来越多地用于各种重要应用程序，但其安全性和可靠性仍然是主要问题。人们提出了各种对抗和越狱攻击来绕过安全一致并导致模型产生有害响应。我们引入了自我监督提示注入（SPIN），它可以检测和逆转对LLM的各种攻击。由于我们的自我监督即时防御是在推理时完成的，因此它也与现有的对齐兼容，并为防御增加了额外的安全层。我们的基准测试表明，我们的系统可以将攻击成功率降低高达87.9%，同时保持良性用户请求的性能。此外，我们还讨论了自适应攻击者的情况，并表明我们的方法对于意识到我们防御的攻击者仍然具有弹性。



## **16. Cross-modality Information Check for Detecting Jailbreaking in Multimodal Large Language Models**

多模式大型语言模型中检测越狱的跨模式信息检查 cs.CL

12 pages, 9 figures, EMNLP 2024 Findings

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2407.21659v4) [paper-pdf](http://arxiv.org/pdf/2407.21659v4)

**Authors**: Yue Xu, Xiuyuan Qi, Zhan Qin, Wenjie Wang

**Abstract**: Multimodal Large Language Models (MLLMs) extend the capacity of LLMs to understand multimodal information comprehensively, achieving remarkable performance in many vision-centric tasks. Despite that, recent studies have shown that these models are susceptible to jailbreak attacks, which refer to an exploitative technique where malicious users can break the safety alignment of the target model and generate misleading and harmful answers. This potential threat is caused by both the inherent vulnerabilities of LLM and the larger attack scope introduced by vision input. To enhance the security of MLLMs against jailbreak attacks, researchers have developed various defense techniques. However, these methods either require modifications to the model's internal structure or demand significant computational resources during the inference phase. Multimodal information is a double-edged sword. While it increases the risk of attacks, it also provides additional data that can enhance safeguards. Inspired by this, we propose Cross-modality Information DEtectoR (CIDER), a plug-and-play jailbreaking detector designed to identify maliciously perturbed image inputs, utilizing the cross-modal similarity between harmful queries and adversarial images. CIDER is independent of the target MLLMs and requires less computation cost. Extensive experimental results demonstrate the effectiveness and efficiency of CIDER, as well as its transferability to both white-box and black-box MLLMs.

摘要: 多通道大语言模型扩展了多通道大语言模型对多通道信息的理解能力，在许多以视觉为中心的任务中取得了显著的性能。尽管如此，最近的研究表明，这些模型容易受到越狱攻击，越狱攻击指的是一种利用技术，恶意用户可以破坏目标模型的安全对齐，并生成误导性和有害的答案。这种潜在的威胁既是由LLM固有的漏洞造成的，也是由视觉输入引入的更大的攻击范围造成的。为了提高MLMS抵御越狱攻击的安全性，研究人员开发了各种防御技术。然而，这些方法要么需要修改模型的内部结构，要么在推理阶段需要大量的计算资源。多式联运信息是一把双刃剑。虽然它增加了攻击的风险，但它也提供了额外的数据，可以加强安全措施。受此启发，我们提出了跨模式信息检测器(Cider)，这是一种即插即用的越狱检测器，旨在利用有害查询和敌意图像之间的跨模式相似性来识别恶意扰动的图像输入。苹果酒不依赖于目标MLLM，并且需要较少的计算成本。大量的实验结果证明了苹果酒的有效性和效率，以及它对白盒和黑盒MLLMS的可转换性。



## **17. SurrogatePrompt: Bypassing the Safety Filter of Text-to-Image Models via Substitution**

SurrogatePromise：通过替换来消除文本到图像模型的安全过滤器 cs.CV

To appear in the the 31st ACM Conference on Computer and  Communications Security (CCS)

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2309.14122v3) [paper-pdf](http://arxiv.org/pdf/2309.14122v3)

**Authors**: Zhongjie Ba, Jieming Zhong, Jiachen Lei, Peng Cheng, Qinglong Wang, Zhan Qin, Zhibo Wang, Kui Ren

**Abstract**: Advanced text-to-image models such as DALL$\cdot$E 2 and Midjourney possess the capacity to generate highly realistic images, raising significant concerns regarding the potential proliferation of unsafe content. This includes adult, violent, or deceptive imagery of political figures. Despite claims of rigorous safety mechanisms implemented in these models to restrict the generation of not-safe-for-work (NSFW) content, we successfully devise and exhibit the first prompt attacks on Midjourney, resulting in the production of abundant photorealistic NSFW images. We reveal the fundamental principles of such prompt attacks and suggest strategically substituting high-risk sections within a suspect prompt to evade closed-source safety measures. Our novel framework, SurrogatePrompt, systematically generates attack prompts, utilizing large language models, image-to-text, and image-to-image modules to automate attack prompt creation at scale. Evaluation results disclose an 88% success rate in bypassing Midjourney's proprietary safety filter with our attack prompts, leading to the generation of counterfeit images depicting political figures in violent scenarios. Both subjective and objective assessments validate that the images generated from our attack prompts present considerable safety hazards.

摘要: 高级文本到图像模型，如DALL$\CDOT$E 2和MIDTURE，具有生成高度逼真图像的能力，这引发了人们对不安全内容潜在扩散的严重担忧。这包括成人的、暴力的或欺骗性的政治人物形象。尽管声称在这些模型中实施了严格的安全机制来限制不安全工作(NSFW)内容的生成，但我们成功地设计并展示了第一次在中途进行的即时攻击，从而产生了丰富的照片逼真的NSFW图像。我们揭示了这种快速攻击的基本原理，并建议有策略地在可疑提示中替换高风险部分，以规避封闭源代码的安全措施。我们的新框架Surogue atePrompt系统地生成攻击提示，利用大型语言模型、图像到文本和图像到图像模块来自动大规模创建攻击提示。评估结果显示，使用我们的攻击提示绕过MidRoad的专有安全过滤器的成功率为88%，导致生成描绘暴力场景中的政治人物的假冒图像。主观和客观评估都证实，我们的攻击提示生成的图像存在相当大的安全风险。



## **18. Data Defenses Against Large Language Models**

数据防御大型语言模型 cs.CL

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13138v1) [paper-pdf](http://arxiv.org/pdf/2410.13138v1)

**Authors**: William Agnew, Harry H. Jiang, Cella Sum, Maarten Sap, Sauvik Das

**Abstract**: Large language models excel at performing inference over text to extract information, summarize information, or generate additional text. These inference capabilities are implicated in a variety of ethical harms spanning surveillance, labor displacement, and IP/copyright theft. While many policy, legal, and technical mitigations have been proposed to counteract these harms, these mitigations typically require cooperation from institutions that move slower than technical advances (i.e., governments) or that have few incentives to act to counteract these harms (i.e., the corporations that create and profit from these LLMs). In this paper, we define and build "data defenses" -- a novel strategy that directly empowers data owners to block LLMs from performing inference on their data. We create data defenses by developing a method to automatically generate adversarial prompt injections that, when added to input text, significantly reduce the ability of LLMs to accurately infer personally identifying information about the subject of the input text or to use copyrighted text in inference. We examine the ethics of enabling such direct resistance to LLM inference, and argue that making data defenses that resist and subvert LLMs enables the realization of important values such as data ownership, data sovereignty, and democratic control over AI systems. We verify that our data defenses are cheap and fast to generate, work on the latest commercial and open-source LLMs, resistance to countermeasures, and are robust to several different attack settings. Finally, we consider the security implications of LLM data defenses and outline several future research directions in this area. Our code is available at https://github.com/wagnew3/LLMDataDefenses and a tool for using our defenses to protect text against LLM inference is at https://wagnew3.github.io/LLM-Data-Defenses/.

摘要: 大型语言模型擅长对文本执行推理，以提取信息、汇总信息或生成附加文本。这些推理能力牵涉到各种道德危害，包括监控、劳动力转移和知识产权/版权盗窃。虽然已经提出了许多政策、法律和技术缓解措施来抵消这些危害，但这些缓解措施通常需要行动速度慢于技术进步的机构(即政府)或几乎没有采取行动抵消这些危害的动机的机构(即创造这些低成本管理并从中获利的公司)的合作。在本文中，我们定义并构建了“数据防御”--一种新的策略，它直接授权数据所有者阻止LLM对其数据执行推理。我们通过开发一种自动生成对抗性提示注入的方法来创建数据防御，当这些注入添加到输入文本时，显著降低了LLMS准确推断关于输入文本主题的个人识别信息或在推理中使用受版权保护的文本的能力。我们审查了允许这种直接抵抗LLM推理的伦理，并认为，制定抵抗和颠覆LLM的数据防御能够实现重要的价值，如数据所有权、数据主权和对人工智能系统的民主控制。我们验证了我们的数据防御是廉价和快速生成的，在最新的商业和开源LLM上工作，对对策的抵抗力，以及对几种不同攻击设置的健壮性。最后，我们考虑了LLM数据防御的安全含义，并概述了该领域未来的几个研究方向。我们的代码可在https://github.com/wagnew3/LLMDataDefenses上获得，使用我们的防御措施保护文本免受LLm推断的工具可在https://wagnew3.github.io/LLM-Data-Defenses/.上获得



## **19. Self-Comparison for Dataset-Level Membership Inference in Large (Vision-)Language Models**

大型（视觉）语言模型中数据集级隶属推理的自我比较 cs.LG

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2410.13088v1) [paper-pdf](http://arxiv.org/pdf/2410.13088v1)

**Authors**: Jie Ren, Kangrui Chen, Chen Chen, Vikash Sehwag, Yue Xing, Jiliang Tang, Lingjuan Lyu

**Abstract**: Large Language Models (LLMs) and Vision-Language Models (VLMs) have made significant advancements in a wide range of natural language processing and vision-language tasks. Access to large web-scale datasets has been a key factor in their success. However, concerns have been raised about the unauthorized use of copyrighted materials and potential copyright infringement. Existing methods, such as sample-level Membership Inference Attacks (MIA) and distribution-based dataset inference, distinguish member data (data used for training) and non-member data by leveraging the common observation that models tend to memorize and show greater confidence in member data. Nevertheless, these methods face challenges when applied to LLMs and VLMs, such as the requirement for ground-truth member data or non-member data that shares the same distribution as the test data. In this paper, we propose a novel dataset-level membership inference method based on Self-Comparison. We find that a member prefix followed by a non-member suffix (paraphrased from a member suffix) can further trigger the model's memorization on training data. Instead of directly comparing member and non-member data, we introduce paraphrasing to the second half of the sequence and evaluate how the likelihood changes before and after paraphrasing. Unlike prior approaches, our method does not require access to ground-truth member data or non-member data in identical distribution, making it more practical. Extensive experiments demonstrate that our proposed method outperforms traditional MIA and dataset inference techniques across various datasets and models, including including public models, fine-tuned models, and API-based commercial models.

摘要: 大语言模型和视觉语言模型在自然语言处理和视觉语言任务中取得了重大进展。获得大型网络规模的数据集一直是它们成功的关键因素。然而，人们对未经授权使用受版权保护的材料和潜在的侵犯版权行为表示担忧。现有的方法，如样本级成员关系推理攻击(MIA)和基于分布的数据集推理，通过利用模型倾向于记忆的共同观测来区分成员数据(用于训练的数据)和非成员数据，并对成员数据表现出更大的置信度。然而，这些方法在应用于LLM和VLM时面临挑战，例如需要与测试数据共享相同分布的地面真实成员数据或非成员数据。本文提出了一种基于自比较的数据集级隶属度推理方法。我们发现，成员前缀后面跟着非成员后缀(改述自成员后缀)可以进一步触发模型对训练数据的记忆。我们不是直接比较成员和非成员数据，而是将释义引入序列的后半部分，并评估释义前后似然性的变化。与以前的方法不同，我们的方法不需要访问相同分布的地面真实成员数据或非成员数据，使其更具实用性。大量实验表明，我们提出的方法在各种数据集和模型上的性能优于传统的MIA和数据集推理技术，包括公共模型、微调模型和基于API的商业模型。



## **20. Hiding-in-Plain-Sight (HiPS) Attack on CLIP for Targetted Object Removal from Images**

对CLIP进行隐藏式视线（HiPS）攻击，以从图像中删除目标对象 cs.LG

Published in the 3rd Workshop on New Frontiers in Adversarial Machine  Learning at NeurIPS 2024. 10 pages, 7 figures, 3 tables

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2410.13010v1) [paper-pdf](http://arxiv.org/pdf/2410.13010v1)

**Authors**: Arka Daw, Megan Hong-Thanh Chung, Maria Mahbub, Amir Sadovnik

**Abstract**: Machine learning models are known to be vulnerable to adversarial attacks, but traditional attacks have mostly focused on single-modalities. With the rise of large multi-modal models (LMMs) like CLIP, which combine vision and language capabilities, new vulnerabilities have emerged. However, prior work in multimodal targeted attacks aim to completely change the model's output to what the adversary wants. In many realistic scenarios, an adversary might seek to make only subtle modifications to the output, so that the changes go unnoticed by downstream models or even by humans. We introduce Hiding-in-Plain-Sight (HiPS) attacks, a novel class of adversarial attacks that subtly modifies model predictions by selectively concealing target object(s), as if the target object was absent from the scene. We propose two HiPS attack variants, HiPS-cls and HiPS-cap, and demonstrate their effectiveness in transferring to downstream image captioning models, such as CLIP-Cap, for targeted object removal from image captions.

摘要: 众所周知，机器学习模型容易受到对抗性攻击，但传统攻击大多集中在单一模式上。随着像CLIP这样结合了视觉和语言能力的大型多模式模型(LMM)的兴起，出现了新的漏洞。然而，以前在多模式定向攻击方面的工作旨在将模型的输出完全改变为对手想要的。在许多现实场景中，对手可能只寻求对输出进行微妙的修改，这样下游模型甚至人类都不会注意到这些变化。介绍了一种新型的对抗性攻击--视线隐藏攻击(HIPS)，它通过选择性地隐藏目标对象(S)来巧妙地修改模型预测，就好像目标对象不在场景中一样。我们提出了两个HIPS攻击变体，HIPS-CLS和HIPS-CAP，并证明了它们在转移到下游图像字幕模型(如CLIP-Cap)以从图像字幕中去除目标对象方面的有效性。



## **21. Mechanistic Unlearning: Robust Knowledge Unlearning and Editing via Mechanistic Localization**

机械性忘记学习：通过机械性本地化稳健的知识忘记学习和编辑 cs.LG

20 pages, 19 figures, 7 tables

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2410.12949v1) [paper-pdf](http://arxiv.org/pdf/2410.12949v1)

**Authors**: Phillip Guo, Aaquib Syed, Abhay Sheshadri, Aidan Ewart, Gintare Karolina Dziugaite

**Abstract**: Methods for knowledge editing and unlearning in large language models seek to edit or remove undesirable knowledge or capabilities without compromising general language modeling performance. This work investigates how mechanistic interpretability -- which, in part, aims to identify model components (circuits) associated to specific interpretable mechanisms that make up a model capability -- can improve the precision and effectiveness of editing and unlearning. We find a stark difference in unlearning and edit robustness when training components localized by different methods. We highlight an important distinction between methods that localize components based primarily on preserving outputs, and those finding high level mechanisms with predictable intermediate states. In particular, localizing edits/unlearning to components associated with the lookup-table mechanism for factual recall 1) leads to more robust edits/unlearning across different input/output formats, and 2) resists attempts to relearn the unwanted information, while also reducing unintended side effects compared to baselines, on both a sports facts dataset and the CounterFact dataset across multiple models. We also find that certain localized edits disrupt the latent knowledge in the model more than any other baselines, making unlearning more robust to various attacks.

摘要: 用于大型语言模型中的知识编辑和去学习的方法寻求在不损害一般语言建模性能的情况下编辑或移除不需要的知识或能力。这项工作调查了机械性可解释性--部分目的是确定与构成模型能力的特定可解释机制相关联的模型组件(电路)--如何提高编辑和取消学习的精确度和有效性。我们发现，当训练不同方法局部化的组件时，忘记学习和编辑健壮性存在明显差异。我们强调了主要基于保留输出来本地化组件的方法与找到具有可预测中间状态的高级机制之间的重要区别。具体地说，对与用于事实回忆的查找表机制相关联的组件的本地化编辑/忘记1)导致跨不同输入/输出格式的更健壮的编辑/忘记，以及2)抵制重新学习不想要的信息的尝试，同时还减少了与基线相比的意外副作用，在多个模型上的体育事实数据集和反事实数据集两者上。我们还发现，与其他基线相比，某些局部编辑对模型中潜在知识的破坏更大，使得遗忘对各种攻击更具健壮性。



## **22. ToBlend: Token-Level Blending With an Ensemble of LLMs to Attack AI-Generated Text Detection**

ToBlend：与LLM集合进行令牌级混合以攻击AI生成的文本检测 cs.CL

Submitted to ARR Oct-2024 Cycle

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2402.11167v2) [paper-pdf](http://arxiv.org/pdf/2402.11167v2)

**Authors**: Fan Huang, Haewoon Kwak, Jisun An

**Abstract**: The robustness of AI-content detection models against sophisticated adversarial strategies, such as paraphrasing or word switching, is a rising concern in natural language generation (NLG) applications. This study proposes ToBlend, a novel token-level ensemble text generation method to challenge the robustness of current AI-content detection approaches by utilizing multiple sets of candidate generative large language models (LLMs). By randomly sampling token(s) from candidate LLMs sets, we find ToBlend significantly drops the performance of most mainstream AI-content detection methods. We evaluate the text quality produced under different ToBlend settings based on annotations from experienced human experts. We proposed a fine-tuned Llama3.1 model to distinguish the ToBlend generated text more accurately. Our findings underscore our proposed text generation approach's great potential in deceiving and improving detection models. Our datasets, codes, and annotations are open-sourced.

摘要: 人工智能内容检测模型对重述或单词切换等复杂对抗策略的稳健性是自然语言生成（NLG）应用程序中日益关注的问题。这项研究提出了ToBlend，这是一种新型的代币级集成文本生成方法，通过利用多组候选生成式大型语言模型（LLM）来挑战当前人工智能内容检测方法的稳健性。通过从候选LLM集中随机采样令牌，我们发现ToBlend显着降低了大多数主流AI内容检测方法的性能。我们根据经验丰富的人类专家的注释来评估不同ToBlend设置下生成的文本质量。我们提出了一个微调的Llama3.1模型，以更准确地区分ToBlend生成的文本。我们的发现强调了我们提出的文本生成方法在欺骗和改进检测模型方面的巨大潜力。我们的数据集、代码和注释是开源的。



## **23. Efficient and Effective Universal Adversarial Attack against Vision-Language Pre-training Models**

针对视觉语言预训练模型的高效且有效的通用对抗攻击 cs.CV

11 pages

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2410.11639v2) [paper-pdf](http://arxiv.org/pdf/2410.11639v2)

**Authors**: Fan Yang, Yihao Huang, Kailong Wang, Ling Shi, Geguang Pu, Yang Liu, Haoyu Wang

**Abstract**: Vision-language pre-training (VLP) models, trained on large-scale image-text pairs, have become widely used across a variety of downstream vision-and-language (V+L) tasks. This widespread adoption raises concerns about their vulnerability to adversarial attacks. Non-universal adversarial attacks, while effective, are often impractical for real-time online applications due to their high computational demands per data instance. Recently, universal adversarial perturbations (UAPs) have been introduced as a solution, but existing generator-based UAP methods are significantly time-consuming. To overcome the limitation, we propose a direct optimization-based UAP approach, termed DO-UAP, which significantly reduces resource consumption while maintaining high attack performance. Specifically, we explore the necessity of multimodal loss design and introduce a useful data augmentation strategy. Extensive experiments conducted on three benchmark VLP datasets, six popular VLP models, and three classical downstream tasks demonstrate the efficiency and effectiveness of DO-UAP. Specifically, our approach drastically decreases the time consumption by 23-fold while achieving a better attack performance.

摘要: 视觉-语言预训练模型是在大规模图文对上训练的，已被广泛应用于各种下游视觉与语言(V+L)任务。这种广泛的采用引起了人们对它们易受对手攻击的担忧。非通用对抗性攻击虽然有效，但对于实时在线应用程序来说往往是不切实际的，因为它们对每个数据实例的计算要求很高。最近，通用对抗扰动(UAP)被引入作为解决方案，但现有的基于生成器的UAP方法非常耗时。为了克服这一局限性，我们提出了一种基于直接优化的UAP方法，称为DO-UAP，它在保持高攻击性能的同时显著减少了资源消耗。具体地说，我们探讨了多峰损失设计的必要性，并介绍了一种有用的数据增强策略。在三个基准VLP数据集、六个流行的VLP模型和三个经典下游任务上的广泛实验证明了DO-UAP的效率和有效性。具体地说，我们的方法大大减少了23倍的时间消耗，同时实现了更好的攻击性能。



## **24. Reconstruction of Differentially Private Text Sanitization via Large Language Models**

通过大语言模型重建差异私人文本清理 cs.CR

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2410.12443v1) [paper-pdf](http://arxiv.org/pdf/2410.12443v1)

**Authors**: Shuchao Pang, Zhigang Lu, Haichen Wang, Peng Fu, Yongbin Zhou, Minhui Xue, Bo Li

**Abstract**: Differential privacy (DP) is the de facto privacy standard against privacy leakage attacks, including many recently discovered ones against large language models (LLMs). However, we discovered that LLMs could reconstruct the altered/removed privacy from given DP-sanitized prompts. We propose two attacks (black-box and white-box) based on the accessibility to LLMs and show that LLMs could connect the pair of DP-sanitized text and the corresponding private training data of LLMs by giving sample text pairs as instructions (in the black-box attacks) or fine-tuning data (in the white-box attacks). To illustrate our findings, we conduct comprehensive experiments on modern LLMs (e.g., LLaMA-2, LLaMA-3, ChatGPT-3.5, ChatGPT-4, ChatGPT-4o, Claude-3, Claude-3.5, OPT, GPT-Neo, GPT-J, Gemma-2, and Pythia) using commonly used datasets (such as WikiMIA, Pile-CC, and Pile-Wiki) against both word-level and sentence-level DP. The experimental results show promising recovery rates, e.g., the black-box attacks against the word-level DP over WikiMIA dataset gave 72.18% on LLaMA-2 (70B), 82.39% on LLaMA-3 (70B), 75.35% on Gemma-2, 91.2% on ChatGPT-4o, and 94.01% on Claude-3.5 (Sonnet). More urgently, this study indicates that these well-known LLMs have emerged as a new security risk for existing DP text sanitization approaches in the current environment.

摘要: 差异隐私(DP)是针对隐私泄露攻击的事实上的隐私标准，包括许多最近发现的针对大型语言模型(LLM)的攻击。然而，我们发现LLMS可以从给定的DP净化提示中重建被更改/删除的隐私。我们基于LLMS的可达性提出了两种攻击(黑盒和白盒)，并证明了LLMS可以通过给出样本文本对作为指令(在黑盒攻击中)或微调数据(在白盒攻击中)来连接经DP消毒的文本对和对应的LLMS的私有训练数据。为了说明我们的发现，我们使用常用的数据集(如WikiMIA、PILL-CC和PILL-Wiki)在现代LLMS(例如，Llama-2、Llama-3、ChatGPT-3.5、ChatGPT-4、ChatGPT-4o、Claude-3、Claude-3.5、OPT、GPT-Neo、Gpt-J、Gema-2和Pythia)上进行了全面的实验。实验结果表明，在WikiMIA数据集上针对词级DP的黑盒攻击在Llama-2(70B)上达到了72.18%，在Llama-3(70B)上达到了82.39%，在Gema-2上达到了75.35%，在ChatGPT-40上达到了91.2%，在Claude-3.5(十四行集)上达到了94.01%。更迫切的是，这项研究表明，这些知名的LLM已经成为现有DP文本净化方法在当前环境下的新安全风险。



## **25. CoreGuard: Safeguarding Foundational Capabilities of LLMs Against Model Stealing in Edge Deployment**

CoreGuard：保护LLM的基础能力，防止边缘部署中的模型窃取 cs.CR

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2410.13903v1) [paper-pdf](http://arxiv.org/pdf/2410.13903v1)

**Authors**: Qinfeng Li, Yangfan Xie, Tianyu Du, Zhiqiang Shen, Zhenghan Qin, Hao Peng, Xinkui Zhao, Xianwei Zhu, Jianwei Yin, Xuhong Zhang

**Abstract**: Proprietary large language models (LLMs) demonstrate exceptional generalization ability across various tasks. Additionally, deploying LLMs on edge devices is trending for efficiency and privacy reasons. However, edge deployment of proprietary LLMs introduces new security threats: attackers who obtain an edge-deployed LLM can easily use it as a base model for various tasks due to its high generalization ability, which we call foundational capability stealing. Unfortunately, existing model protection mechanisms are often task-specific and fail to protect general-purpose LLMs, as they mainly focus on protecting task-related parameters using trusted execution environments (TEEs). Although some recent TEE-based methods are able to protect the overall model parameters in a computation-efficient way, they still suffer from prohibitive communication costs between TEE and CPU/GPU, making it impractical to deploy for edge LLMs. To protect the foundational capabilities of edge LLMs, we propose CoreGuard, a computation- and communication-efficient model protection approach against model stealing on edge devices. The core component of CoreGuard is a lightweight and propagative authorization module residing in TEE. Extensive experiments show that CoreGuard achieves the same security protection as the black-box security guarantees with negligible overhead.

摘要: 专有的大型语言模型(LLM)在各种任务中表现出非凡的泛化能力。此外，出于效率和隐私的原因，在边缘设备上部署LLM是一种趋势。然而，专有LLMS的边缘部署带来了新的安全威胁：获得边缘部署LLM的攻击者可以很容易地将其用作各种任务的基础模型，因为它具有高度的泛化能力，我们称之为基础能力窃取。遗憾的是，现有的模型保护机制往往是特定于任务的，不能保护通用的LLM，因为它们主要集中在使用可信执行环境(TEE)来保护与任务相关的参数。尽管最近的一些基于TEE的方法能够以计算高效的方式保护整体模型参数，但它们仍然受到TEE与CPU/GPU之间高昂的通信成本的影响，使得将其应用于EDGE LLMS是不现实的。为了保护EDGE LLMS的基础能力，我们提出了一种针对EDGE设备上的模型窃取的计算和通信高效的模型保护方法CoreGuard。CoreGuard的核心组件是驻留在TEE中的轻量级和可传播授权模块。大量的实验表明，CoreGuard实现了与黑盒安全保证相同的安全保护，而开销可以忽略不计。



## **26. Can LLMs Patch Security Issues?**

LLM可以解决安全问题吗？ cs.CR

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2312.00024v5) [paper-pdf](http://arxiv.org/pdf/2312.00024v5)

**Authors**: Kamel Alrashedy, Abdullah Aljasser, Pradyumna Tambwekar, Matthew Gombolay

**Abstract**: Large Language Models (LLMs) have shown impressive proficiency in code generation. Unfortunately, these models share a weakness with their human counterparts: producing code that inadvertently has security vulnerabilities. These vulnerabilities could allow unauthorized attackers to access sensitive data or systems, which is unacceptable for safety-critical applications. In this work, we propose Feedback-Driven Security Patching (FDSP), where LLMs automatically refine generated, vulnerable code. Our approach leverages automatic static code analysis to empower the LLM to generate and implement potential solutions to address vulnerabilities. We address the research communitys needs for safe code generation by introducing a large-scale dataset, PythonSecurityEval, covering the diversity of real-world applications, including databases, websites and operating systems. We empirically validate that FDSP outperforms prior work that uses self-feedback from LLMs by up to 17.6% through our procedure that injects targeted, external feedback. Code and data are available at \url{https://github.com/Kamel773/LLM-code-refine}

摘要: 大型语言模型(LLM)在代码生成方面表现出令人印象深刻的熟练程度。不幸的是，这些模型与它们的人类同行有一个共同的弱点：生成的代码无意中存在安全漏洞。这些漏洞可能允许未经授权的攻击者访问敏感数据或系统，这对于安全关键型应用程序是不可接受的。在这项工作中，我们提出了反馈驱动的安全修补(FDSP)，其中LLMS自动精炼生成的易受攻击的代码。我们的方法利用自动静态代码分析来支持LLM生成和实施潜在的解决方案来应对漏洞。我们通过引入大规模数据集PythonSecurityEval来满足研究团体对安全代码生成的需求，该数据集涵盖了包括数据库、网站和操作系统在内的各种现实应用程序。我们通过注入有针对性的外部反馈的程序，经验性地验证了FDSP的性能比使用LLMS自我反馈的先前工作高出17.6%。有关代码和数据，请访问\url{https://github.com/Kamel773/LLM-code-refine}



## **27. SoK: Prompt Hacking of Large Language Models**

SoK：大型语言模型的即时黑客攻击 cs.CR

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2410.13901v1) [paper-pdf](http://arxiv.org/pdf/2410.13901v1)

**Authors**: Baha Rababah, Shang, Wu, Matthew Kwiatkowski, Carson Leung, Cuneyt Gurcan Akcora

**Abstract**: The safety and robustness of large language models (LLMs) based applications remain critical challenges in artificial intelligence. Among the key threats to these applications are prompt hacking attacks, which can significantly undermine the security and reliability of LLM-based systems. In this work, we offer a comprehensive and systematic overview of three distinct types of prompt hacking: jailbreaking, leaking, and injection, addressing the nuances that differentiate them despite their overlapping characteristics. To enhance the evaluation of LLM-based applications, we propose a novel framework that categorizes LLM responses into five distinct classes, moving beyond the traditional binary classification. This approach provides more granular insights into the AI's behavior, improving diagnostic precision and enabling more targeted enhancements to the system's safety and robustness.

摘要: 基于大型语言模型（LLM）的应用程序的安全性和稳健性仍然是人工智能领域的关键挑战。这些应用程序面临的主要威胁之一是即时黑客攻击，这可能会严重破坏基于LLM的系统的安全性和可靠性。在这项工作中，我们对三种不同类型的即时黑客行为进行了全面而系统的概述：越狱、泄露和注入，尽管它们的特征重叠，但仍解决了它们的细微差别。为了加强对基于LLM的应用程序的评估，我们提出了一个新颖的框架，将LLM响应分为五个不同的类别，超越了传统的二元分类。这种方法提供了对人工智能行为的更细致的见解，提高诊断精度，并对系统的安全性和稳健性进行更有针对性的增强。



## **28. GPT-4 Jailbreaks Itself with Near-Perfect Success Using Self-Explanation**

GPT-4通过自我解释以近乎完美的成功越狱 cs.CR

Accepted to EMNLP 2024 Main Conference

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2405.13077v2) [paper-pdf](http://arxiv.org/pdf/2405.13077v2)

**Authors**: Govind Ramesh, Yao Dou, Wei Xu

**Abstract**: Research on jailbreaking has been valuable for testing and understanding the safety and security issues of large language models (LLMs). In this paper, we introduce Iterative Refinement Induced Self-Jailbreak (IRIS), a novel approach that leverages the reflective capabilities of LLMs for jailbreaking with only black-box access. Unlike previous methods, IRIS simplifies the jailbreaking process by using a single model as both the attacker and target. This method first iteratively refines adversarial prompts through self-explanation, which is crucial for ensuring that even well-aligned LLMs obey adversarial instructions. IRIS then rates and enhances the output given the refined prompt to increase its harmfulness. We find that IRIS achieves jailbreak success rates of 98% on GPT-4, 92% on GPT-4 Turbo, and 94% on Llama-3.1-70B in under 7 queries. It significantly outperforms prior approaches in automatic, black-box, and interpretable jailbreaking, while requiring substantially fewer queries, thereby establishing a new standard for interpretable jailbreaking methods.

摘要: 越狱研究对于测试和理解大型语言模型(LLM)的安全和安保问题具有重要价值。在本文中，我们介绍了迭代精化诱导的自越狱(IRIS)，这是一种新的方法，它利用LLMS的反射能力来实现只访问黑盒的越狱。与以前的方法不同，IRIS通过将单一模型用作攻击者和目标来简化越狱过程。这种方法首先通过自我解释迭代地精炼对抗性提示，这对于确保即使是排列良好的LLM也遵守对抗性指令至关重要。然后，虹膜给出精致的提示，对产量进行评级并提高产量，以增加其危害性。我们发现，IRIS在GPT-4上的越狱成功率为98%，在GPT-4Turbo上的越狱成功率为92%，在Llama-3.1-70B上的越狱成功率为94%。它在自动、黑盒和可解释越狱方面显著优于现有方法，同时需要的查询大大减少，从而建立了可解释越狱方法的新标准。



## **29. Taking off the Rose-Tinted Glasses: A Critical Look at Adversarial ML Through the Lens of Evasion Attacks**

摘下玫瑰色眼镜：从逃避攻击的角度批判性地审视对抗性ML cs.LG

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.12076v1) [paper-pdf](http://arxiv.org/pdf/2410.12076v1)

**Authors**: Kevin Eykholt, Farhan Ahmed, Pratik Vaishnavi, Amir Rahmati

**Abstract**: The vulnerability of machine learning models in adversarial scenarios has garnered significant interest in the academic community over the past decade, resulting in a myriad of attacks and defenses. However, while the community appears to be overtly successful in devising new attacks across new contexts, the development of defenses has stalled. After a decade of research, we appear no closer to securing AI applications beyond additional training. Despite a lack of effective mitigations, AI development and its incorporation into existing systems charge full speed ahead with the rise of generative AI and large language models. Will our ineffectiveness in developing solutions to adversarial threats further extend to these new technologies?   In this paper, we argue that overly permissive attack and overly restrictive defensive threat models have hampered defense development in the ML domain. Through the lens of adversarial evasion attacks against neural networks, we critically examine common attack assumptions, such as the ability to bypass any defense not explicitly built into the model. We argue that these flawed assumptions, seen as reasonable by the community based on paper acceptance, have encouraged the development of adversarial attacks that map poorly to real-world scenarios. In turn, new defenses evaluated against these very attacks are inadvertently required to be almost perfect and incorporated as part of the model. But do they need to? In practice, machine learning models are deployed as a small component of a larger system. We analyze adversarial machine learning from a system security perspective rather than an AI perspective and its implications for emerging AI paradigms.

摘要: 在过去的十年里，机器学习模型在对抗性场景中的脆弱性引起了学术界的极大兴趣，导致了无数的攻击和防御。然而，尽管该社区似乎公开成功地在新的背景下设计了新的攻击，但防御的发展却停滞不前。经过十年的研究，除了额外的培训外，我们似乎并没有更进一步地确保人工智能应用的安全。尽管缺乏有效的缓解措施，但随着生成性人工智能和大型语言模型的崛起，人工智能的发展及其与现有系统的结合全速前进。我们在开发对抗威胁的解决方案方面的无效性是否会进一步延伸到这些新技术？在本文中，我们认为过度允许的攻击和过度受限的防御威胁模型阻碍了ML领域的防御发展。通过针对神经网络的对抗性逃避攻击的镜头，我们批判性地检查了常见的攻击假设，例如绕过未显式构建在模型中的任何防御的能力。我们认为，这些有缺陷的假设被社区视为基于论文接受的合理假设，鼓励了与现实世界情景映射不佳的对抗性攻击的发展。反过来，针对这些攻击评估的新防御措施被无意中要求近乎完美，并作为模型的一部分纳入。但他们真的需要这样做吗？在实践中，机器学习模型被部署为更大系统的一个小组件。我们从系统安全的角度分析对抗性机器学习，而不是从人工智能的角度分析它对新兴人工智能范例的影响。



## **30. A Watermark for Low-entropy and Unbiased Generation in Large Language Models**

大型语言模型中低熵和无偏生成的水印 cs.CL

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2405.14604v2) [paper-pdf](http://arxiv.org/pdf/2405.14604v2)

**Authors**: Minjia Mao, Dongjun Wei, Zeyu Chen, Xiao Fang, Michael Chau

**Abstract**: Recent advancements in large language models (LLMs) have highlighted the risk of misusing them, raising the need for accurate detection of LLM-generated content. In response, a viable solution is to inject imperceptible identifiers into LLMs, known as watermarks. Previous work demonstrates that unbiased watermarks ensure unforgeability and preserve text quality by maintaining the expectation of the LLM output probability distribution. However, previous unbiased watermarking methods suffer from one or more of the following issues: (1) requiring access to white-box LLMs during detection, (2) incurring long detection time, (3) being not robust against simple watermarking attacks, (4) failing to provide statistical guarantees for the type II error of watermark detection, and (5) being not statistically unbiased for low-entropy scenarios, which hinder their deployment in practice. This study proposes the Sampling One Then Accepting (STA-1) method, a watermark that can address all of these issues. Moreover, we discuss the tradeoff between watermark strength and text quality for unbiased watermarks. We show that in low-entropy scenarios, unbiased watermarks face a tradeoff between watermark strength and the risk of unsatisfactory outputs. Experimental results on both low-entropy and high-entropy datasets demonstrate that STA-1 achieves text quality and watermark strength comparable to existing unbiased watermarks, with a low risk of unsatisfactory outputs. Implementation codes for this study are available online.

摘要: 大型语言模型(LLM)最近的进步突显了滥用它们的风险，提高了对LLM生成的内容进行准确检测的必要性。对此，一个可行的解决方案是将不可察觉的标识符注入LLM，即所谓的水印。前人的工作表明，无偏水印通过保持LLM输出概率分布的期望来确保不可伪造性并保持文本质量。然而，以往的无偏水印方法存在以下一个或多个问题：(1)检测时需要访问白盒LLM，(2)检测时间较长，(3)对简单水印攻击不稳健，(4)未能对水印检测的第二类错误提供统计保证，(5)对低熵场景不具有统计无偏性，这阻碍了它们在实践中的应用。本研究提出了一种先采样后接受(STA-1)的方法，可以解决所有这些问题。此外，我们还讨论了无偏水印在水印强度和文本质量之间的权衡。我们证明了在低熵的情况下，无偏水印面临着水印强度和输出不满意的风险之间的权衡。在低熵和高熵数据集上的实验结果表明，STA-1的文本质量和水印强度与现有的无偏水印相当，输出不满意的风险很低。这项研究的实施代码可在网上查阅。



## **31. G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks**

G-Designer：通过图神经网络构建多智能体通信布局 cs.MA

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11782v1) [paper-pdf](http://arxiv.org/pdf/2410.11782v1)

**Authors**: Guibin Zhang, Yanwei Yue, Xiangguo Sun, Guancheng Wan, Miao Yu, Junfeng Fang, Kun Wang, Dawei Cheng

**Abstract**: Recent advancements in large language model (LLM)-based agents have demonstrated that collective intelligence can significantly surpass the capabilities of individual agents, primarily due to well-crafted inter-agent communication topologies. Despite the diverse and high-performing designs available, practitioners often face confusion when selecting the most effective pipeline for their specific task: \textit{Which topology is the best choice for my task, avoiding unnecessary communication token overhead while ensuring high-quality solution?} In response to this dilemma, we introduce G-Designer, an adaptive, efficient, and robust solution for multi-agent deployment, which dynamically designs task-aware, customized communication topologies. Specifically, G-Designer models the multi-agent system as a multi-agent network, leveraging a variational graph auto-encoder to encode both the nodes (agents) and a task-specific virtual node, and decodes a task-adaptive and high-performing communication topology. Extensive experiments on six benchmarks showcase that G-Designer is: \textbf{(1) high-performing}, achieving superior results on MMLU with accuracy at $84.50\%$ and on HumanEval with pass@1 at $89.90\%$; \textbf{(2) task-adaptive}, architecting communication protocols tailored to task difficulty, reducing token consumption by up to $95.33\%$ on HumanEval; and \textbf{(3) adversarially robust}, defending against agent adversarial attacks with merely $0.3\%$ accuracy drop.

摘要: 基于大型语言模型(LLM)的代理的最新进展表明，集体智能可以显著超过单个代理的能力，这主要是由于精心设计的代理间通信拓扑。尽管有多样化和高性能的设计，但实践者在为他们的特定任务选择最有效的流水线时经常面临困惑：\textit{哪个拓扑是我的任务的最佳选择，在确保高质量解决方案的同时避免不必要的通信令牌开销？}针对这种困境，我们引入了G-Designer，这是一个自适应的、高效的、健壮的多代理部署解决方案，它动态地设计任务感知的、定制的通信拓扑。具体地说，G-Designer将多代理系统建模为多代理网络，利用变化图自动编码器对节点(代理)和特定于任务的虚拟节点进行编码，并解码任务自适应的高性能通信拓扑。在六个基准测试上的广泛实验表明，G-Designer是：\extbf{(1)高性能}，在MMLU上获得了更好的结果，准确率为84.50\$，在HumanEval上，PASS@1的准确率为89.90\$；\extbf{(2)任务自适应}，构建了针对任务难度的通信协议，在HumanEval上减少了高达95.33\$的令牌消耗；以及\extbf{(3)对手健壮性}，防御代理对手攻击，精确度仅下降了$0.3\%$。



## **32. LLM-Based Robust Product Classification in Commerce and Compliance**

基于LLM的商业和合规稳健产品分类 cs.CL

Camera-ready version for Customizable NLP Workshop at EMNLP 2024. 11  pages

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2408.05874v2) [paper-pdf](http://arxiv.org/pdf/2408.05874v2)

**Authors**: Sina Gholamian, Gianfranco Romani, Bartosz Rudnikowicz, Stavroula Skylaki

**Abstract**: Product classification is a crucial task in international trade, as compliance regulations are verified and taxes and duties are applied based on product categories. Manual classification of products is time-consuming and error-prone, and the sheer volume of products imported and exported renders the manual process infeasible. Consequently, e-commerce platforms and enterprises involved in international trade have turned to automatic product classification using machine learning. However, current approaches do not consider the real-world challenges associated with product classification, such as very abbreviated and incomplete product descriptions. In addition, recent advancements in generative Large Language Models (LLMs) and their reasoning capabilities are mainly untapped in product classification and e-commerce. In this research, we explore the real-life challenges of industrial classification and we propose data perturbations that allow for realistic data simulation. Furthermore, we employ LLM-based product classification to improve the robustness of the prediction in presence of incomplete data. Our research shows that LLMs with in-context learning outperform the supervised approaches in the clean-data scenario. Additionally, we illustrate that LLMs are significantly more robust than the supervised approaches when data attacks are present.

摘要: 产品分类是国际贸易中的一项关键任务，因为要核实合规规定，并根据产品类别适用税收和关税。人工对产品进行分类既耗时又容易出错，而且进出口产品的数量庞大，使手工分类过程变得不可行。因此，参与国际贸易的电子商务平台和企业已经转向使用机器学习的产品自动分类。然而，目前的方法没有考虑到与产品分类相关的现实挑战，例如非常简短和不完整的产品描述。此外，生成性大型语言模型(LLM)及其推理能力的最新进展主要是在产品分类和电子商务方面尚未开发。在这项研究中，我们探索了现实生活中的行业分类挑战，并提出了允许现实数据模拟的数据扰动。此外，我们使用基于LLM的产品分类来提高在存在不完整数据的情况下预测的稳健性。我们的研究表明，在干净数据的情况下，具有情境学习的LLMS的性能优于有监督的方法。此外，我们还说明了当存在数据攻击时，LLMS比监督方法具有更强的健壮性。



## **33. Phantom: General Trigger Attacks on Retrieval Augmented Language Generation**

Phantom：对检索增强语言生成的通用触发攻击 cs.CR

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2405.20485v2) [paper-pdf](http://arxiv.org/pdf/2405.20485v2)

**Authors**: Harsh Chaudhari, Giorgio Severi, John Abascal, Matthew Jagielski, Christopher A. Choquette-Choo, Milad Nasr, Cristina Nita-Rotaru, Alina Oprea

**Abstract**: Retrieval Augmented Generation (RAG) expands the capabilities of modern large language models (LLMs), by anchoring, adapting, and personalizing their responses to the most relevant knowledge sources. It is particularly useful in chatbot applications, allowing developers to customize LLM output without expensive retraining. Despite their significant utility in various applications, RAG systems present new security risks. In this work, we propose new attack vectors that allow an adversary to inject a single malicious document into a RAG system's knowledge base, and mount a backdoor poisoning attack. We design Phantom, a general two-stage optimization framework against RAG systems, that crafts a malicious poisoned document leading to an integrity violation in the model's output. First, the document is constructed to be retrieved only when a specific trigger sequence of tokens appears in the victim's queries. Second, the document is further optimized with crafted adversarial text that induces various adversarial objectives on the LLM output, including refusal to answer, reputation damage, privacy violations, and harmful behaviors. We demonstrate our attacks on multiple LLM architectures, including Gemma, Vicuna, and Llama, and show that they transfer to GPT-3.5 Turbo and GPT-4. Finally, we successfully conducted a Phantom attack on NVIDIA's black-box production RAG system, "Chat with RTX".

摘要: 检索增强生成(RAG)通过锚定、调整和个性化对最相关的知识源的响应来扩展现代大型语言模型(LLMS)的能力。它在聊天机器人应用程序中特别有用，允许开发人员定制LLM输出，而无需昂贵的再培训。尽管RAG系统在各种应用中具有重要的实用价值，但它带来了新的安全风险。在这项工作中，我们提出了新的攻击向量，允许攻击者将单个恶意文档注入RAG系统的知识库，并发动后门中毒攻击。我们设计了Phantom，这是一个针对RAG系统的通用两阶段优化框架，它手工制作了一个恶意中毒文档，导致模型输出中的完整性破坏。首先，文档被构建为仅在受害者的查询中出现特定的令牌触发序列时才检索。其次，通过精心设计的敌意文本进一步优化了文档，这些文本在LLM输出上诱导了各种敌意目标，包括拒绝回答、声誉损害、侵犯隐私和有害行为。我们演示了我们对多个LLM体系结构的攻击，包括Gema、Vicuna和Llama，并表明它们可以传输到GPT-3.5Turbo和GPT-4。最后，我们成功地对NVIDIA的黑匣子生产RAG系统“与腾讯通聊天”进行了幻影攻击。



## **34. Gotcha! This Model Uses My Code! Evaluating Membership Leakage Risks in Code Models**

抓住你了！这个模型使用我的代码！评估代码模型中的成员泄漏风险 cs.SE

Accepted by IEEE Transactions on Software Engineering, Camera-Ready  Version

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2310.01166v2) [paper-pdf](http://arxiv.org/pdf/2310.01166v2)

**Authors**: Zhou Yang, Zhipeng Zhao, Chenyu Wang, Jieke Shi, Dongsum Kim, Donggyun Han, David Lo

**Abstract**: Given large-scale source code datasets available in open-source projects and advanced large language models, recent code models have been proposed to address a series of critical software engineering tasks, such as program repair and code completion. The training data of the code models come from various sources, not only the publicly available source code, e.g., open-source projects on GitHub but also the private data such as the confidential source code from companies, which may contain sensitive information (for example, SSH keys and personal information). As a result, the use of these code models may raise new privacy concerns.   In this paper, we focus on a critical yet not well-explored question on using code models: what is the risk of membership information leakage in code models? Membership information leakage refers to the risk that an attacker can infer whether a given data point is included in (i.e., a member of) the training data. To answer this question, we propose Gotcha, a novel membership inference attack method specifically for code models. We investigate the membership leakage risk of code models. Our results reveal a worrying fact that the risk of membership leakage is high: although the previous attack methods are close to random guessing, Gotcha can predict the data membership with a high true positive rate of 0.95 and a low false positive rate of 0.10. We also show that the attacker's knowledge of the victim model (e.g., the model architecture and the pre-training data) impacts the success rate of attacks. Further analysis demonstrates that changing the decoding strategy can mitigate the risk of membership leakage. This study calls for more attention to understanding the privacy of code models and developing more effective countermeasures against such attacks.

摘要: 鉴于开源项目中可用的大规模源代码数据集和高级大型语言模型，最近提出了一些代码模型来解决一系列关键的软件工程任务，如程序修复和代码完成。代码模型的训练数据来自各种来源，不仅有公开可用的源代码，如GitHub上的开源项目，还包括私人数据，如来自公司的机密源代码，其中可能包含敏感信息(如SSH密钥和个人信息)。因此，使用这些代码模型可能会引发新的隐私问题。在这篇文章中，我们关注一个关于使用代码模型的关键但没有得到很好探索的问题：代码模型中成员信息泄漏的风险是什么？成员资格信息泄漏是指攻击者可以推断给定数据点是否包括在训练数据中(即，训练数据的成员)的风险。为了回答这个问题，我们提出了Gotcha，一种新的专门针对代码模型的成员推理攻击方法。我们研究了编码模型的成员泄漏风险。我们的结果揭示了一个令人担忧的事实，即成员泄露的风险很高：虽然以前的攻击方法接近随机猜测，但Gotcha可以预测数据的成员身份，真阳性率高达0.95，假阳性率低0.10。我们还表明，攻击者对受害者模型的了解(例如，模型体系结构和预训练数据)会影响攻击的成功率。进一步的分析表明，改变译码策略可以降低成员泄漏的风险。这项研究呼吁更多地关注了解代码模型的隐私，并开发更有效的对策来应对此类攻击。



## **35. Multi-round jailbreak attack on large language models**

对大型语言模型的多轮越狱攻击 cs.CL

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11533v1) [paper-pdf](http://arxiv.org/pdf/2410.11533v1)

**Authors**: Yihua Zhou, Xiaochuan Shi

**Abstract**: Ensuring the safety and alignment of large language models (LLMs) with human values is crucial for generating responses that are beneficial to humanity. While LLMs have the capability to identify and avoid harmful queries, they remain vulnerable to "jailbreak" attacks, where carefully crafted prompts can induce the generation of toxic content. Traditional single-round jailbreak attacks, such as GCG and AutoDAN, do not alter the sensitive words in the dangerous prompts. Although they can temporarily bypass the model's safeguards through prompt engineering, their success rate drops significantly as the LLM is further fine-tuned, and they cannot effectively circumvent static rule-based filters that remove the hazardous vocabulary.   In this study, to better understand jailbreak attacks, we introduce a multi-round jailbreak approach. This method can rewrite the dangerous prompts, decomposing them into a series of less harmful sub-questions to bypass the LLM's safety checks. We first use the LLM to perform a decomposition task, breaking down a set of natural language questions into a sequence of progressive sub-questions, which are then used to fine-tune the Llama3-8B model, enabling it to decompose hazardous prompts. The fine-tuned model is then used to break down the problematic prompt, and the resulting sub-questions are sequentially asked to the victim model. If the victim model rejects a sub-question, a new decomposition is generated, and the process is repeated until the final objective is achieved. Our experimental results show a 94\% success rate on the llama2-7B and demonstrate the effectiveness of this approach in circumventing static rule-based filters.

摘要: 确保大型语言模型(LLM)的安全性并使其与人类价值观保持一致，对于产生有益于人类的反应至关重要。虽然LLM具有识别和避免有害查询的能力，但它们仍然容易受到“越狱”攻击，在这种攻击中，精心设计的提示可能会导致有毒内容的生成。传统的单轮越狱攻击，如GCG和AutoDAN，不会改变危险提示中的敏感词语。尽管他们可以通过快速工程暂时绕过模型的保障措施，但随着LLM的进一步微调，他们的成功率显著下降，而且他们无法有效地绕过删除危险词汇的静态规则过滤器。在这项研究中，为了更好地理解越狱攻击，我们引入了一种多轮越狱方法。这种方法可以重写危险的提示，将它们分解为一系列危害较小的子问题，以绕过LLM的安全检查。我们首先使用LLM执行分解任务，将一组自然语言问题分解为一系列递进子问题，然后使用这些子问题来微调Llama3-8B模型，使其能够分解危险提示。然后，使用微调的模型来分解有问题的提示，并将得到的子问题顺序地询问给受害者模型。如果受害者模型拒绝了子问题，则生成新的分解，并重复该过程，直到达到最终目标。实验结果表明，该方法在Llama2-7B上的检测成功率为94%，证明了该方法对静态规则过滤的有效性。



## **36. Jigsaw Puzzles: Splitting Harmful Questions to Jailbreak Large Language Models**

拼图：分解有害问题以越狱大型语言模型 cs.CL

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11459v1) [paper-pdf](http://arxiv.org/pdf/2410.11459v1)

**Authors**: Hao Yang, Lizhen Qu, Ehsan Shareghi, Gholamreza Haffari

**Abstract**: Large language models (LLMs) have exhibited outstanding performance in engaging with humans and addressing complex questions by leveraging their vast implicit knowledge and robust reasoning capabilities. However, such models are vulnerable to jailbreak attacks, leading to the generation of harmful responses. Despite recent research on single-turn jailbreak strategies to facilitate the development of defence mechanisms, the challenge of revealing vulnerabilities under multi-turn setting remains relatively under-explored. In this work, we propose Jigsaw Puzzles (JSP), a straightforward yet effective multi-turn jailbreak strategy against the advanced LLMs. JSP splits questions into harmless fractions as the input of each turn, and requests LLMs to reconstruct and respond to questions under multi-turn interaction. Our experimental results demonstrate that the proposed JSP jailbreak bypasses original safeguards against explicitly harmful content, achieving an average attack success rate of 93.76% on 189 harmful queries across 5 advanced LLMs (Gemini-1.5-Pro, Llama-3.1-70B, GPT-4, GPT-4o, GPT-4o-mini). Moreover, JSP achieves a state-of-the-art attack success rate of 92% on GPT-4 on the harmful query benchmark, and exhibits strong resistant to defence strategies. Warning: this paper contains offensive examples.

摘要: 大型语言模型(LLM)利用其丰富的隐含知识和强大的推理能力，在与人类接触和解决复杂问题方面表现出了出色的表现。然而，这类模型容易受到越狱攻击，从而导致有害反应的产生。尽管最近对单轮越狱战略进行了研究，以促进防御机制的发展，但在多轮情况下揭示脆弱性的挑战仍然相对较少。在这项工作中，我们提出了Jigsaw Puzzles(JSP)，一种针对高级LLM的简单而有效的多回合越狱策略。该算法将问题分解成若干个无害的分数作为每一轮的输入，并在多轮交互的情况下要求LLMS对问题进行重构和回答。我们的实验结果表明，该方案绕过了原有的针对显式有害内容的保护措施，对5个高级LLMS(Gemini-1.5-Pro、Llama-3.1-70B、GPT-4、GPT-4o、GPT-4o-mini)的189个有害查询的平均攻击成功率为93.76%。此外，在有害查询基准上，对GPT-4的攻击成功率达到92%，并且对防御策略表现出很强的抵抗力。警告：本文包含令人反感的例子。



## **37. Deciphering the Chaos: Enhancing Jailbreak Attacks via Adversarial Prompt Translation**

破译混乱：通过对抗性提示翻译增强越狱攻击 cs.LG

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11317v1) [paper-pdf](http://arxiv.org/pdf/2410.11317v1)

**Authors**: Qizhang Li, Xiaochen Yang, Wangmeng Zuo, Yiwen Guo

**Abstract**: Automatic adversarial prompt generation provides remarkable success in jailbreaking safely-aligned large language models (LLMs). Existing gradient-based attacks, while demonstrating outstanding performance in jailbreaking white-box LLMs, often generate garbled adversarial prompts with chaotic appearance. These adversarial prompts are difficult to transfer to other LLMs, hindering their performance in attacking unknown victim models. In this paper, for the first time, we delve into the semantic meaning embedded in garbled adversarial prompts and propose a novel method that "translates" them into coherent and human-readable natural language adversarial prompts. In this way, we can effectively uncover the semantic information that triggers vulnerabilities of the model and unambiguously transfer it to the victim model, without overlooking the adversarial information hidden in the garbled text, to enhance jailbreak attacks. It also offers a new approach to discovering effective designs for jailbreak prompts, advancing the understanding of jailbreak attacks. Experimental results demonstrate that our method significantly improves the success rate of jailbreak attacks against various safety-aligned LLMs and outperforms state-of-the-arts by large margins. With at most 10 queries, our method achieves an average attack success rate of 81.8% in attacking 7 commercial closed-source LLMs, including GPT and Claude-3 series, on HarmBench. Our method also achieves over 90% attack success rates against Llama-2-Chat models on AdvBench, despite their outstanding resistance to jailbreak attacks. Code at: https://github.com/qizhangli/Adversarial-Prompt-Translator.

摘要: 自动对抗性提示生成在越狱安全对齐的大型语言模型(LLM)方面取得了显着的成功。现有的基于梯度的攻击虽然在越狱白盒LLM中表现出出色的性能，但往往会产生外观混乱的乱码对抗性提示。这些对抗性提示很难转移到其他LLM上，阻碍了它们在攻击未知受害者模型时的表现。在本文中，我们首次深入研究了混淆的对抗性提示中所蕴含的语义，并提出了一种新的方法，将它们“翻译”成连贯的、人类可读的自然语言对抗性提示。通过这种方式，我们可以有效地发现触发模型漏洞的语义信息，并毫不含糊地将其传递给受害者模型，而不会忽视隐藏在乱码文本中的对抗性信息，以增强越狱攻击。它还提供了一种新的方法来发现有效的越狱提示设计，促进了对越狱攻击的理解。实验结果表明，我们的方法显著提高了对各种安全对齐LLM的越狱攻击成功率，并且远远超过了最新的技术水平。在最多10个查询的情况下，我们的方法在HarmBch上攻击包括GPT和Claude-3系列在内的7个商业闭源LLM，平均攻击成功率为81.8%。我们的方法对AdvBtch上的Llama-2-Chat模型的攻击成功率也达到了90%以上，尽管它们对越狱攻击具有出色的抵抗力。代码：https://github.com/qizhangli/Adversarial-Prompt-Translator.



## **38. Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation**

闭上眼睛，安全：通过图像到文本转换保护多模式LLM cs.CV

ECCV2024 (Project Page: https://gyhdog99.github.io/projects/ecso/)

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2403.09572v4) [paper-pdf](http://arxiv.org/pdf/2403.09572v4)

**Authors**: Yunhao Gou, Kai Chen, Zhili Liu, Lanqing Hong, Hang Xu, Zhenguo Li, Dit-Yan Yeung, James T. Kwok, Yu Zhang

**Abstract**: Multimodal large language models (MLLMs) have shown impressive reasoning abilities. However, they are also more vulnerable to jailbreak attacks than their LLM predecessors. Although still capable of detecting the unsafe responses, we observe that safety mechanisms of the pre-aligned LLMs in MLLMs can be easily bypassed with the introduction of image features. To construct robust MLLMs, we propose ECSO (Eyes Closed, Safety On), a novel training-free protecting approach that exploits the inherent safety awareness of MLLMs, and generates safer responses via adaptively transforming unsafe images into texts to activate the intrinsic safety mechanism of pre-aligned LLMs in MLLMs. Experiments on five state-of-the-art (SoTA) MLLMs demonstrate that ECSO enhances model safety significantly (e.g.,, 37.6% improvement on the MM-SafetyBench (SD+OCR) and 71.3% on VLSafe with LLaVA-1.5-7B), while consistently maintaining utility results on common MLLM benchmarks. Furthermore, we show that ECSO can be used as a data engine to generate supervised-finetuning (SFT) data for MLLM alignment without extra human intervention.

摘要: 多通道大型语言模型(MLLMS)已经显示出令人印象深刻的推理能力。然而，他们也比他们的LLM前辈更容易受到越狱攻击。虽然仍然能够检测到不安全的响应，但我们观察到，通过引入图像特征，可以很容易地绕过MLLMS中预对准LLM的安全机制。为了构造稳健的MLLMS，我们提出了一种新的无需训练的保护方法ECSO(Eyes Closed，Safe On)，该方法利用MLLMS固有的安全意识，通过自适应地将不安全的图像转换为文本来激活MLLMS中预对准的LLMS的固有安全机制，从而产生更安全的响应。在五个最先进的(SOTA)MLLM上的实验表明，ECSO显著提高了模型安全性(例如，在MM-SafetyBch(SD+OCR)的基础上改进了37.6%，在使用LLaVA-1.5-7B的VLSafe上改进了71.3%)，同时保持了常见MLLM基准的实用结果。此外，我们还证明了ECSO可以作为数据引擎来生成用于MLLM比对的监督精调(SFT)数据，而无需额外的人工干预。



## **39. Cognitive Overload Attack:Prompt Injection for Long Context**

认知过载攻击：长上下文的提示注入 cs.CL

40 pages, 31 Figures

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11272v1) [paper-pdf](http://arxiv.org/pdf/2410.11272v1)

**Authors**: Bibek Upadhayay, Vahid Behzadan, Amin Karbasi

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in performing tasks across various domains without needing explicit retraining. This capability, known as In-Context Learning (ICL), while impressive, exposes LLMs to a variety of adversarial prompts and jailbreaks that manipulate safety-trained LLMs into generating undesired or harmful output. In this paper, we propose a novel interpretation of ICL in LLMs through the lens of cognitive neuroscience, by drawing parallels between learning in human cognition with ICL. We applied the principles of Cognitive Load Theory in LLMs and empirically validate that similar to human cognition, LLMs also suffer from cognitive overload a state where the demand on cognitive processing exceeds the available capacity of the model, leading to potential errors. Furthermore, we demonstrated how an attacker can exploit ICL to jailbreak LLMs through deliberately designed prompts that induce cognitive overload on LLMs, thereby compromising the safety mechanisms of LLMs. We empirically validate this threat model by crafting various cognitive overload prompts and show that advanced models such as GPT-4, Claude-3.5 Sonnet, Claude-3 OPUS, Llama-3-70B-Instruct, Gemini-1.0-Pro, and Gemini-1.5-Pro can be successfully jailbroken, with attack success rates of up to 99.99%. Our findings highlight critical vulnerabilities in LLMs and underscore the urgency of developing robust safeguards. We propose integrating insights from cognitive load theory into the design and evaluation of LLMs to better anticipate and mitigate the risks of adversarial attacks. By expanding our experiments to encompass a broader range of models and by highlighting vulnerabilities in LLMs' ICL, we aim to ensure the development of safer and more reliable AI systems.

摘要: 大型语言模型(LLM)已经显示出在不需要明确的再培训的情况下执行跨领域任务的显著能力。这种被称为情景学习(ICL)的能力虽然令人印象深刻，但会使LLM暴露在各种对抗性提示和越狱之下，这些提示和越狱操作经过安全培训的LLM产生不需要的或有害的输出。在这篇文章中，我们提出了一种新的解释，从认知神经科学的角度，通过将人类认知中的学习与ICL相提并论，对LLMS中的ICL做出了新的解释。我们将认知负荷理论的原理应用到LLMS中，并实证验证了与人类认知类似，LLMS也存在认知过载，即认知加工需求超过模型的可用能力，从而导致潜在错误。此外，我们演示了攻击者如何通过故意设计的提示来利用ICL来越狱LLM，这些提示会导致LLM上的认知过载，从而危及LLMS的安全机制。我们通过制作不同的认知过载提示对该威胁模型进行了实证验证，结果表明，GPT-4、Claude-3.5十四行诗、Claude-3 opus、Llama-3-70B-Indict、Gemini-1.0-Pro和Gemini-1.5-Pro等高级模型可以成功越狱，攻击成功率高达99.99%。我们的发现突显了低土地管理制度的严重脆弱性，并强调了制定强有力的保障措施的紧迫性。我们建议将认知负荷理论的见解融入到LLMS的设计和评估中，以更好地预测和减轻对手攻击的风险。通过扩大我们的实验以涵盖更广泛的模型，并通过突出LLMS ICL中的漏洞，我们的目标是确保开发出更安全、更可靠的人工智能系统。



## **40. A Formal Framework for Assessing and Mitigating Emergent Security Risks in Generative AI Models: Bridging Theory and Dynamic Risk Mitigation**

评估和缓解生成人工智能模型中紧急安全风险的正式框架：桥梁理论和动态风险缓解 cs.CR

This paper was accepted in NeurIPS 2024 workshop on Red Teaming  GenAI: What can we learn with Adversaries?

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.13897v1) [paper-pdf](http://arxiv.org/pdf/2410.13897v1)

**Authors**: Aviral Srivastava, Sourav Panda

**Abstract**: As generative AI systems, including large language models (LLMs) and diffusion models, advance rapidly, their growing adoption has led to new and complex security risks often overlooked in traditional AI risk assessment frameworks. This paper introduces a novel formal framework for categorizing and mitigating these emergent security risks by integrating adaptive, real-time monitoring, and dynamic risk mitigation strategies tailored to generative models' unique vulnerabilities. We identify previously under-explored risks, including latent space exploitation, multi-modal cross-attack vectors, and feedback-loop-induced model degradation. Our framework employs a layered approach, incorporating anomaly detection, continuous red-teaming, and real-time adversarial simulation to mitigate these risks. We focus on formal verification methods to ensure model robustness and scalability in the face of evolving threats. Though theoretical, this work sets the stage for future empirical validation by establishing a detailed methodology and metrics for evaluating the performance of risk mitigation strategies in generative AI systems. This framework addresses existing gaps in AI safety, offering a comprehensive road map for future research and implementation.

摘要: 随着包括大型语言模型(LLM)和扩散模型在内的产生式AI系统的快速发展，它们的日益采用导致了传统AI风险评估框架中经常被忽视的新的复杂安全风险。本文介绍了一种新的形式化框架，通过集成针对生成式模型的独特漏洞定制的自适应、实时监控和动态风险缓解策略，对这些紧急安全风险进行分类和缓解。我们识别了以前未被充分开发的风险，包括潜在空间开发、多模式交叉攻击向量和反馈环导致的模型退化。我们的框架采用了分层的方法，结合了异常检测、持续的红色团队和实时对手模拟来降低这些风险。我们专注于形式化的验证方法，以确保模型在面对不断变化的威胁时的健壮性和可伸缩性。虽然这项工作是理论上的，但通过建立一种详细的方法和指标来评估生成性人工智能系统中风险缓解策略的性能，这项工作为未来的经验验证奠定了基础。这一框架解决了人工智能安全方面的现有差距，为未来的研究和实施提供了全面的路线图。



## **41. Archilles' Heel in Semi-open LLMs: Hiding Bottom against Recovery Attacks**

阿奇勒斯在半开放式法学硕士中的脚跟：隐藏底部抵御复苏攻击 cs.LG

10 pages for main content of the paper

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11182v1) [paper-pdf](http://arxiv.org/pdf/2410.11182v1)

**Authors**: Hanbo Huang, Yihan Li, Bowen Jiang, Lin Liu, Ruoyu Sun, Zhuotao Liu, Shiyu Liang

**Abstract**: Closed-source large language models deliver strong performance but have limited downstream customizability. Semi-open models, combining both closed-source and public layers, were introduced to improve customizability. However, parameters in the closed-source layers are found vulnerable to recovery attacks. In this paper, we explore the design of semi-open models with fewer closed-source layers, aiming to increase customizability while ensuring resilience to recovery attacks. We analyze the contribution of closed-source layer to the overall resilience and theoretically prove that in a deep transformer-based model, there exists a transition layer such that even small recovery errors in layers before this layer can lead to recovery failure. Building on this, we propose \textbf{SCARA}, a novel approach that keeps only a few bottom layers as closed-source. SCARA employs a fine-tuning-free metric to estimate the maximum number of layers that can be publicly accessible for customization. We apply it to five models (1.3B to 70B parameters) to construct semi-open models, validating their customizability on six downstream tasks and assessing their resilience against various recovery attacks on sixteen benchmarks. We compare SCARA to baselines and observe that it generally improves downstream customization performance and offers similar resilience with over \textbf{10} times fewer closed-source parameters. We empirically investigate the existence of transition layers, analyze the effectiveness of our scheme and finally discuss its limitations.

摘要: 封闭源代码的大型语言模型提供了强大的性能，但下游可定制化能力有限。引入了半开放模型，结合了封闭源码和公共层，以提高可定制性。然而，封闭源代码层中的参数容易受到恢复攻击。在本文中，我们探索了闭源层较少的半开放模型的设计，目的是在增加可定制性的同时确保对恢复攻击的弹性。我们分析了闭源层对整体恢复能力的贡献，并从理论上证明了在基于深层变压器的模型中，存在一个过渡层，即使在该过渡层之前的各层中存在微小的恢复错误，也可能导致恢复失败。在此基础上，我们提出了一种新的方法Scara使用一种无需微调的指标来估计可公开访问以供定制的最大层数。我们将其应用于5个模型(1.3B到70B参数)来构建半开放模型，验证了它们在6个下游任务上的可定制性，并评估了它们对16个基准测试上的各种恢复攻击的恢复能力。我们将SCARA与Baseline进行了比较，并观察到它通常提高了下游定制性能，并提供了类似的弹性，而闭源参数减少了1/10。我们对过渡层的存在进行了实证研究，分析了我们方案的有效性，最后讨论了它的局限性。



## **42. Denial-of-Service Poisoning Attacks against Large Language Models**

针对大型语言模型的拒绝服务中毒攻击 cs.CR

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10760v1) [paper-pdf](http://arxiv.org/pdf/2410.10760v1)

**Authors**: Kuofeng Gao, Tianyu Pang, Chao Du, Yong Yang, Shu-Tao Xia, Min Lin

**Abstract**: Recent studies have shown that LLMs are vulnerable to denial-of-service (DoS) attacks, where adversarial inputs like spelling errors or non-semantic prompts trigger endless outputs without generating an [EOS] token. These attacks can potentially cause high latency and make LLM services inaccessible to other users or tasks. However, when there are speech-to-text interfaces (e.g., voice commands to a robot), executing such DoS attacks becomes challenging, as it is difficult to introduce spelling errors or non-semantic prompts through speech. A simple DoS attack in these scenarios would be to instruct the model to "Keep repeating Hello", but we observe that relying solely on natural instructions limits output length, which is bounded by the maximum length of the LLM's supervised finetuning (SFT) data. To overcome this limitation, we propose poisoning-based DoS (P-DoS) attacks for LLMs, demonstrating that injecting a single poisoned sample designed for DoS purposes can break the output length limit. For example, a poisoned sample can successfully attack GPT-4o and GPT-4o mini (via OpenAI's finetuning API) using less than $1, causing repeated outputs up to the maximum inference length (16K tokens, compared to 0.5K before poisoning). Additionally, we perform comprehensive ablation studies on open-source LLMs and extend our method to LLM agents, where attackers can control both the finetuning dataset and algorithm. Our findings underscore the urgent need for defenses against P-DoS attacks to secure LLMs. Our code is available at https://github.com/sail-sg/P-DoS.

摘要: 最近的研究表明，LLMS容易受到拒绝服务(DoS)攻击，即拼写错误或非语义提示等敌意输入会触发无休止的输出，而不会生成[EOS]令牌。这些攻击可能会导致高延迟，并使其他用户或任务无法访问LLM服务。然而，当存在语音到文本的接口(例如，对机器人的语音命令)时，执行这种DoS攻击变得具有挑战性，因为很难通过语音引入拼写错误或非语义提示。在这些场景中，一个简单的DoS攻击是指示模型“不断重复Hello”，但我们观察到，仅依赖自然指令会限制输出长度，而输出长度受LLM的监督微调(SFT)数据的最大长度的限制。为了克服这一局限性，我们提出了针对LLMS的基于中毒的DoS(P-DoS)攻击，证明了注入单个为DoS目的而设计的有毒样本可以打破输出长度限制。例如，中毒的样本可以使用不到1美元的成本成功攻击GPT-4o和GPT-4o mini(通过OpenAI的Finetuning API)，导致重复输出到最大推理长度(16K令牌，而中毒前为0.5K)。此外，我们在开源LLMS上进行了全面的烧蚀研究，并将我们的方法扩展到LLM代理，其中攻击者可以控制精调数据集和算法。我们的发现强调了防御P-DoS攻击以确保LLM安全的迫切需要。我们的代码可以在https://github.com/sail-sg/P-DoS.上找到



## **43. Derail Yourself: Multi-turn LLM Jailbreak Attack through Self-discovered Clues**

脱轨自己：通过自我发现的线索进行多回合LLM越狱攻击 cs.CL

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10700v1) [paper-pdf](http://arxiv.org/pdf/2410.10700v1)

**Authors**: Qibing Ren, Hao Li, Dongrui Liu, Zhanxu Xie, Xiaoya Lu, Yu Qiao, Lei Sha, Junchi Yan, Lizhuang Ma, Jing Shao

**Abstract**: This study exposes the safety vulnerabilities of Large Language Models (LLMs) in multi-turn interactions, where malicious users can obscure harmful intents across several queries. We introduce ActorAttack, a novel multi-turn attack method inspired by actor-network theory, which models a network of semantically linked actors as attack clues to generate diverse and effective attack paths toward harmful targets. ActorAttack addresses two main challenges in multi-turn attacks: (1) concealing harmful intents by creating an innocuous conversation topic about the actor, and (2) uncovering diverse attack paths towards the same harmful target by leveraging LLMs' knowledge to specify the correlated actors as various attack clues. In this way, ActorAttack outperforms existing single-turn and multi-turn attack methods across advanced aligned LLMs, even for GPT-o1. We will publish a dataset called SafeMTData, which includes multi-turn adversarial prompts and safety alignment data, generated by ActorAttack. We demonstrate that models safety-tuned using our safety dataset are more robust to multi-turn attacks. Code is available at https://github.com/renqibing/ActorAttack.

摘要: 这项研究揭示了大型语言模型(LLM)在多轮交互中的安全漏洞，在这种交互中，恶意用户可以通过几个查询来掩盖有害意图。我们引入了ActorAttack，这是一种受行动者-网络理论启发的新型多回合攻击方法，它将语义上联系在一起的行动者网络建模为攻击线索，以生成针对有害目标的多样化和有效的攻击路径。ActorAttack解决了多轮攻击中的两个主要挑战：(1)通过创建关于参与者的无害对话主题来隐藏有害意图；(2)通过利用LLMS的知识将相关的参与者指定为各种攻击线索，揭示针对同一有害目标的不同攻击路径。通过这种方式，ActorAttack在高级对准LLM上的表现优于现有的单回合和多回合攻击方法，即使对于GPT-o1也是如此。我们将发布一个名为SafeMTData的数据集，其中包括由ActorAttack生成的多轮对抗性提示和安全对齐数据。我们证明，使用我们的安全数据集进行安全调整的模型对多轮攻击更具健壮性。代码可在https://github.com/renqibing/ActorAttack.上找到



## **44. F2A: An Innovative Approach for Prompt Injection by Utilizing Feign Security Detection Agents**

F2A：利用Feign安全检测代理进行即时注入的创新方法 cs.CR

1. Fixed typo in abstract 2. Provisionally completed the article  update to facilitate future version revisions

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.08776v2) [paper-pdf](http://arxiv.org/pdf/2410.08776v2)

**Authors**: Yupeng Ren

**Abstract**: With the rapid development of Large Language Models (LLMs), numerous mature applications of LLMs have emerged in the field of content safety detection. However, we have found that LLMs exhibit blind trust in safety detection agents. The general LLMs can be compromised by hackers with this vulnerability. Hence, this paper proposed an attack named Feign Agent Attack (F2A).Through such malicious forgery methods, adding fake safety detection results into the prompt, the defense mechanism of LLMs can be bypassed, thereby obtaining harmful content and hijacking the normal conversation. Continually, a series of experiments were conducted. In these experiments, the hijacking capability of F2A on LLMs was analyzed and demonstrated, exploring the fundamental reasons why LLMs blindly trust safety detection results. The experiments involved various scenarios where fake safety detection results were injected into prompts, and the responses were closely monitored to understand the extent of the vulnerability. Also, this paper provided a reasonable solution to this attack, emphasizing that it is important for LLMs to critically evaluate the results of augmented agents to prevent the generating harmful content. By doing so, the reliability and security can be significantly improved, protecting the LLMs from F2A.

摘要: 随着大语言模型的快速发展，大语言模型在内容安全检测领域出现了大量成熟的应用。然而，我们发现LLM在安全检测代理中表现出盲目信任。一般的LLMS可能会被黑客利用此漏洞攻击。为此，提出了一种伪装代理攻击(F2A)，通过这种恶意伪造方法，将虚假的安全检测结果添加到提示中，从而绕过LLMS的防御机制，从而获取有害内容，劫持正常的会话。不断地，进行了一系列的实验。在这些实验中，分析和论证了F2A对LLMS的劫持能力，探索了LLMS盲目相信安全检测结果的根本原因。这些实验涉及各种场景，在提示中注入虚假的安全检测结果，并密切监控响应，以了解漏洞的程度。此外，本文还提供了一种合理的解决方案，强调了对于LLMS来说，批判性地评估增强剂的结果对于防止产生有害内容是很重要的。通过这样做，可以显著提高可靠性和安全性，保护LLMS免受F2A的影响。



## **45. On Calibration of LLM-based Guard Models for Reliable Content Moderation**

基于LLM的保护模型的校准以实现可靠的内容审核 cs.CR

19 pages, 9 figures

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10414v1) [paper-pdf](http://arxiv.org/pdf/2410.10414v1)

**Authors**: Hongfu Liu, Hengguan Huang, Hao Wang, Xiangming Gu, Ye Wang

**Abstract**: Large language models (LLMs) pose significant risks due to the potential for generating harmful content or users attempting to evade guardrails. Existing studies have developed LLM-based guard models designed to moderate the input and output of threat LLMs, ensuring adherence to safety policies by blocking content that violates these protocols upon deployment. However, limited attention has been given to the reliability and calibration of such guard models. In this work, we empirically conduct comprehensive investigations of confidence calibration for 9 existing LLM-based guard models on 12 benchmarks in both user input and model output classification. Our findings reveal that current LLM-based guard models tend to 1) produce overconfident predictions, 2) exhibit significant miscalibration when subjected to jailbreak attacks, and 3) demonstrate limited robustness to the outputs generated by different types of response models. Additionally, we assess the effectiveness of post-hoc calibration methods to mitigate miscalibration. We demonstrate the efficacy of temperature scaling and, for the first time, highlight the benefits of contextual calibration for confidence calibration of guard models, particularly in the absence of validation sets. Our analysis and experiments underscore the limitations of current LLM-based guard models and provide valuable insights for the future development of well-calibrated guard models toward more reliable content moderation. We also advocate for incorporating reliability evaluation of confidence calibration when releasing future LLM-based guard models.

摘要: 由于可能会生成有害内容或用户试图避开护栏，大型语言模型(LLM)会带来重大风险。现有研究开发了基于LLM的防护模型，旨在控制威胁LLM的输入和输出，通过在部署时阻止违反这些协议的内容来确保遵守安全策略。然而，对这种防护模型的可靠性和校准的关注有限。在这项工作中，我们在用户输入和模型输出分类的12个基准上，对现有的9个基于LLM的警戒模型进行了全面的置信度校准研究。我们的发现表明，当前基于LLM的警卫模型倾向于1)产生过度自信的预测，2)在受到越狱攻击时表现出严重的错误校准，3)对不同类型的反应模型产生的输出表现出有限的稳健性。此外，我们评估后校准方法的有效性，以减少错误校准。我们展示了温度缩放的有效性，并首次强调了上下文校准对于警卫模型的置信度校准的好处，特别是在缺乏验证集的情况下。我们的分析和实验强调了当前基于LLM的防护模型的局限性，并为未来发展经过良好校准的防护模型以实现更可靠的内容审核提供了有价值的见解。我们还主张在发布未来基于LLM的警戒模型时纳入置信度校准的可靠性评估。



## **46. Jailbreak Instruction-Tuned LLMs via end-of-sentence MLP Re-weighting**

通过句末MLP重新加权调整越狱指令的LLM cs.CL

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10150v1) [paper-pdf](http://arxiv.org/pdf/2410.10150v1)

**Authors**: Yifan Luo, Zhennan Zhou, Meitan Wang, Bin Dong

**Abstract**: In this paper, we investigate the safety mechanisms of instruction fine-tuned large language models (LLMs). We discover that re-weighting MLP neurons can significantly compromise a model's safety, especially for MLPs in end-of-sentence inferences. We hypothesize that LLMs evaluate the harmfulness of prompts during end-of-sentence inferences, and MLP layers plays a critical role in this process. Based on this hypothesis, we develop 2 novel white-box jailbreak methods: a prompt-specific method and a prompt-general method. The prompt-specific method targets individual prompts and optimizes the attack on the fly, while the prompt-general method is pre-trained offline and can generalize to unseen harmful prompts. Our methods demonstrate robust performance across 7 popular open-source LLMs, size ranging from 2B to 72B. Furthermore, our study provides insights into vulnerabilities of instruction-tuned LLM's safety and deepens the understanding of the internal mechanisms of LLMs.

摘要: 本文研究了指令微调大型语言模型（LLM）的安全机制。我们发现，重新加权MLP神经元会显着损害模型的安全性，尤其是对于句末推理中的MLP。我们假设LLM在句末推理期间评估提示的危害性，而MLP层在这个过程中发挥着关键作用。基于这一假设，我们开发了两种新型白盒越狱方法：预算特定方法和预算通用方法。预算特定方法针对单个提示并动态优化攻击，而预算通用方法是离线预训练的，可以推广到不可见的有害提示。我们的方法在7种流行的开源LLM（大小从2B到72 B不等）上展示了稳健的性能。此外，我们的研究还深入了解了经描述调整的LLM安全性的漏洞，并加深了对LLM内部机制的理解。



## **47. White-box Multimodal Jailbreaks Against Large Vision-Language Models**

针对大型视觉语言模型的白盒多模式越狱 cs.CV

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2405.17894v2) [paper-pdf](http://arxiv.org/pdf/2405.17894v2)

**Authors**: Ruofan Wang, Xingjun Ma, Hanxu Zhou, Chuanjun Ji, Guangnan Ye, Yu-Gang Jiang

**Abstract**: Recent advancements in Large Vision-Language Models (VLMs) have underscored their superiority in various multimodal tasks. However, the adversarial robustness of VLMs has not been fully explored. Existing methods mainly assess robustness through unimodal adversarial attacks that perturb images, while assuming inherent resilience against text-based attacks. Different from existing attacks, in this work we propose a more comprehensive strategy that jointly attacks both text and image modalities to exploit a broader spectrum of vulnerability within VLMs. Specifically, we propose a dual optimization objective aimed at guiding the model to generate affirmative responses with high toxicity. Our attack method begins by optimizing an adversarial image prefix from random noise to generate diverse harmful responses in the absence of text input, thus imbuing the image with toxic semantics. Subsequently, an adversarial text suffix is integrated and co-optimized with the adversarial image prefix to maximize the probability of eliciting affirmative responses to various harmful instructions. The discovered adversarial image prefix and text suffix are collectively denoted as a Universal Master Key (UMK). When integrated into various malicious queries, UMK can circumvent the alignment defenses of VLMs and lead to the generation of objectionable content, known as jailbreaks. The experimental results demonstrate that our universal attack strategy can effectively jailbreak MiniGPT-4 with a 96% success rate, highlighting the vulnerability of VLMs and the urgent need for new alignment strategies.

摘要: 大型视觉语言模型(VLM)的最新进展凸显了它们在各种多通道任务中的优越性。然而，VLMS的对抗健壮性还没有得到充分的研究。现有的方法主要通过扰乱图像的单峰对抗性攻击来评估稳健性，同时假设对基于文本的攻击具有内在的弹性。与已有的攻击不同，我们提出了一种更全面的策略，联合攻击文本和图像模式，以利用VLM中更广泛的漏洞。具体地说，我们提出了一个双重优化目标，旨在引导模型产生高毒性的肯定反应。我们的攻击方法首先从随机噪声中优化一个敌意图像前缀，在没有文本输入的情况下产生不同的有害响应，从而使图像充满有毒语义。随后，对抗性文本后缀与对抗性图像前缀集成并共同优化，以最大限度地引起对各种有害指令的肯定响应的概率。所发现的敌意图像前缀和文本后缀统称为通用主密钥(UMK)。当集成到各种恶意查询中时，UMK可以绕过VLM的对齐防御，并导致生成令人反感的内容，即所谓的越狱。实验结果表明，我们的通用攻击策略能够有效地越狱MiniGPT-4，成功率为96%，凸显了VLMS的脆弱性和对新的对齐策略的迫切需求。



## **48. 'Quis custodiet ipsos custodes?' Who will watch the watchmen? On Detecting AI-generated peer-reviews**

' Quis guardiate ipsos guardes？“谁来看守看守？关于检测人工智能生成的同行评论 cs.CL

EMNLP Main, 17 pages, 5 figures, 9 tables

**SubmitDate**: 2024-10-13    [abs](http://arxiv.org/abs/2410.09770v1) [paper-pdf](http://arxiv.org/pdf/2410.09770v1)

**Authors**: Sandeep Kumar, Mohit Sahu, Vardhan Gacche, Tirthankar Ghosal, Asif Ekbal

**Abstract**: The integrity of the peer-review process is vital for maintaining scientific rigor and trust within the academic community. With the steady increase in the usage of large language models (LLMs) like ChatGPT in academic writing, there is a growing concern that AI-generated texts could compromise scientific publishing, including peer-reviews. Previous works have focused on generic AI-generated text detection or have presented an approach for estimating the fraction of peer-reviews that can be AI-generated. Our focus here is to solve a real-world problem by assisting the editor or chair in determining whether a review is written by ChatGPT or not. To address this, we introduce the Term Frequency (TF) model, which posits that AI often repeats tokens, and the Review Regeneration (RR) model, which is based on the idea that ChatGPT generates similar outputs upon re-prompting. We stress test these detectors against token attack and paraphrasing. Finally, we propose an effective defensive strategy to reduce the effect of paraphrasing on our models. Our findings suggest both our proposed methods perform better than the other AI text detectors. Our RR model is more robust, although our TF model performs better than the RR model without any attacks. We make our code, dataset, and model public.

摘要: 同行评议过程的完整性对于保持学术界的科学严谨性和信任至关重要。随着像ChatGPT这样的大型语言模型(LLM)在学术写作中的使用稳步增加，人们越来越担心人工智能生成的文本可能会危及科学出版，包括同行评议。以前的工作集中在通用的人工智能生成的文本检测上，或者提出了一种估计人工智能生成的同行评论比例的方法。我们在这里的重点是通过帮助编辑或主席确定评论是否由ChatGPT撰写来解决现实世界的问题。为了解决这个问题，我们引入了术语频率(TF)模型和回顾再生(RR)模型，前者假设人工智能经常重复表征，后者基于ChatGPT在重新提示时生成类似输出的想法。我们对这些检测器进行了针对令牌攻击和释义的压力测试。最后，我们提出了一种有效的防御策略来减少释义对模型的影响。我们的发现表明，我们提出的两种方法都比其他人工智能文本检测器性能更好。我们的RR模型更健壮，尽管我们的TF模型在没有任何攻击的情况下比RR模型执行得更好。我们公开我们的代码、数据集和模型。



## **49. Weak-to-Strong Backdoor Attack for Large Language Models**

大型语言模型的弱到强后门攻击 cs.CR

**SubmitDate**: 2024-10-13    [abs](http://arxiv.org/abs/2409.17946v3) [paper-pdf](http://arxiv.org/pdf/2409.17946v3)

**Authors**: Shuai Zhao, Leilei Gan, Zhongliang Guo, Xiaobao Wu, Luwei Xiao, Xiaoyu Xu, Cong-Duy Nguyen, Luu Anh Tuan

**Abstract**: Despite being widely applied due to their exceptional capabilities, Large Language Models (LLMs) have been proven to be vulnerable to backdoor attacks. These attacks introduce targeted vulnerabilities into LLMs by poisoning training samples and full-parameter fine-tuning. However, this kind of backdoor attack is limited since they require significant computational resources, especially as the size of LLMs increases. Besides, parameter-efficient fine-tuning (PEFT) offers an alternative but the restricted parameter updating may impede the alignment of triggers with target labels. In this study, we first verify that backdoor attacks with PEFT may encounter challenges in achieving feasible performance. To address these issues and improve the effectiveness of backdoor attacks with PEFT, we propose a novel backdoor attack algorithm from weak to strong based on feature alignment-enhanced knowledge distillation (W2SAttack). Specifically, we poison small-scale language models through full-parameter fine-tuning to serve as the teacher model. The teacher model then covertly transfers the backdoor to the large-scale student model through feature alignment-enhanced knowledge distillation, which employs PEFT. Theoretical analysis reveals that W2SAttack has the potential to augment the effectiveness of backdoor attacks. We demonstrate the superior performance of W2SAttack on classification tasks across four language models, four backdoor attack algorithms, and two different architectures of teacher models. Experimental results indicate success rates close to 100% for backdoor attacks targeting PEFT.

摘要: 尽管大型语言模型(LLM)因其卓越的功能而得到广泛应用，但事实证明它们很容易受到后门攻击。这些攻击通过毒化训练样本和全参数微调将有针对性的漏洞引入LLMS。然而，这种后门攻击是有限的，因为它们需要大量的计算资源，特别是随着LLMS大小的增加。此外，参数高效微调(PEFT)提供了另一种选择，但受限的参数更新可能会阻碍触发器与目标标签的对准。在这项研究中，我们首先验证了使用PEFT的后门攻击在实现可行性能方面可能会遇到挑战。为了解决这些问题，提高PEFT后门攻击的有效性，提出了一种基于特征对齐增强知识提取的由弱到强的后门攻击算法(W2SAttack)。具体地说，我们通过全参数微调毒化小规模的语言模型作为教师模型。然后，教师模型通过使用PEFT的特征对齐增强的知识提炼，秘密地将后门转移到大规模学生模型。理论分析表明，W2SAttack具有增强后门攻击有效性的潜力。我们通过四种语言模型、四种后门攻击算法和两种不同的教师模型架构展示了W2SAttack在分类任务上的卓越性能。实验结果表明，针对PEFT的后门攻击成功率接近100%。



## **50. Don't Say No: Jailbreaking LLM by Suppressing Refusal**

不要说不：通过压制拒绝来越狱法学硕士 cs.CL

Update results on Llama3, Llama3.1, Gemma2, Mistral, Qwen2 models and  upon JailbreakBnech, MaliciousInstruct datasets

**SubmitDate**: 2024-10-12    [abs](http://arxiv.org/abs/2404.16369v2) [paper-pdf](http://arxiv.org/pdf/2404.16369v2)

**Authors**: Yukai Zhou, Zhijie Huang, Feiyang Lu, Zhan Qin, Wenjie Wang

**Abstract**: Ensuring the safety alignment of Large Language Models (LLMs) is crucial to generating responses consistent with human values. Despite their ability to recognize and avoid harmful queries, LLMs are vulnerable to jailbreaking attacks, where carefully crafted prompts seduce them to produce toxic content. One category of jailbreak attacks is reformulating the task as an optimization by eliciting the LLM to generate affirmative responses. However, such optimization objective has its own limitations, such as the restriction on the predefined objectionable behaviors, leading to suboptimal attack performance. In this study, we first uncover the reason why vanilla target loss is not optimal, then we explore and enhance the loss objective and introduce the DSN (Don't Say No) attack, which achieves successful attack by suppressing refusal. Another challenge in studying jailbreak attacks is the evaluation, as it is difficult to directly and accurately assess the harmfulness of the responses. The existing evaluation such as refusal keyword matching reveals numerous false positive and false negative instances. To overcome this challenge, we propose an Ensemble Evaluation pipeline that novelly incorporates Natural Language Inference (NLI) contradiction assessment and two external LLM evaluators. Extensive experiments demonstrate the potential of the DSN and effectiveness of Ensemble Evaluation compared to baseline methods.

摘要: 确保大型语言模型(LLM)的安全一致性对于生成与人类价值观一致的响应至关重要。尽管LLM能够识别和避免有害的查询，但它们很容易受到越狱攻击，在这种攻击中，精心制作的提示会引诱它们产生有毒内容。越狱攻击的一类是通过激发LLM产生肯定的响应来将任务重新制定为优化。然而，这样的优化目标有其自身的局限性，如对预定义的不良行为的限制，导致攻击性能次优。在本研究中，我们首先揭示了目标损失不是最优的原因，然后对损失目标进行了探索和增强，并引入了DSN(Don‘t Say No)攻击，通过抑制拒绝来实现攻击的成功。研究越狱攻击的另一个挑战是评估，因为很难直接和准确地评估反应的危害性。现有的拒绝关键词匹配等评价方法揭示了大量的误报和漏报实例。为了克服这一挑战，我们提出了一个集成评估管道，它新颖地结合了自然语言推理(NLI)矛盾评估和两个外部LLM评估器。大量实验表明，与基线方法相比，DSN的潜力和集成评估的有效性。



