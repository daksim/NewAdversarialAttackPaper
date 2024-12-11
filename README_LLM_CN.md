# Latest Large Language Model Attack Papers
**update at 2024-12-11 10:17:21**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. SQL Injection Jailbreak: a structural disaster of large language models**

SQL注入越狱：大型语言模型的结构性灾难 cs.CR

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2411.01565v3) [paper-pdf](http://arxiv.org/pdf/2411.01565v3)

**Authors**: Jiawei Zhao, Kejiang Chen, Weiming Zhang, Nenghai Yu

**Abstract**: In recent years, the rapid development of large language models (LLMs) has brought new vitality into various domains, generating substantial social and economic benefits. However, this swift advancement has also introduced new security vulnerabilities. Jailbreaking, a form of attack that induces LLMs to produce harmful content through carefully crafted prompts, presents a significant challenge to the safe and trustworthy development of LLMs. Previous jailbreak methods primarily exploited the internal properties or capabilities of LLMs, such as optimization-based jailbreak approaches and methods that leveraged the model's context-learning abilities. In this paper, we introduce a novel jailbreak method, SQL Injection Jailbreak (SIJ), which targets the external properties of LLMs, specifically, the way LLMs construct input prompts. By injecting jailbreak information into user prompts, SIJ successfully induces the model to output harmful content. Our SIJ method achieves near 100\% attack success rates on five well-known open-source LLMs on the AdvBench, while incurring lower time costs compared to previous methods. More importantly, SIJ is the first method to exploit the external properties of LLMs for jailbreak attacks and exposes a new vulnerability in LLMs that urgently requires mitigation. To address this, we propose a simple defense method called Self-Reminder-Key to counter SIJ and demonstrate its effectiveness through experimental results. Our code is available at \href{https://github.com/weiyezhimeng/SQL-Injection-Jailbreak}{https://github.com/weiyezhimeng/SQL-Injection-Jailbreak}.

摘要: 近年来，大型语言模型的快速发展为各个领域带来了新的活力，产生了可观的社会效益和经济效益。然而，这种快速发展也带来了新的安全漏洞。越狱是一种攻击形式，通过精心制作的提示诱使LLM产生有害内容，对LLM的安全和可信开发构成了重大挑战。以前的越狱方法主要利用LLMS的内部属性或功能，例如基于优化的越狱方法和利用模型的上下文学习能力的方法。本文介绍了一种新的越狱方法--SQL注入越狱(SIJ)，它针对LLMS的外部属性，特别是LLMS构造输入提示的方式。通过在用户提示中注入越狱信息，SIJ成功地诱导该模型输出有害内容。与以前的方法相比，我们的SIJ方法在AdvBch上的五个著名的开源LLM上获得了近100%的攻击成功率，同时产生了更低的时间成本。更重要的是，SIJ是第一个利用LLMS的外部属性进行越狱攻击的方法，并暴露了LLMS中一个迫切需要缓解的新漏洞。针对这一问题，我们提出了一种简单的防御方法，称为自我提醒密钥来对抗SIJ，并通过实验结果证明了其有效性。我们的代码可以在\href{https://github.com/weiyezhimeng/SQL-Injection-Jailbreak}{https://github.com/weiyezhimeng/SQL-Injection-Jailbreak}.上找到



## **2. MobileSafetyBench: Evaluating Safety of Autonomous Agents in Mobile Device Control**

MobileSafetyBench：评估移动终端控制中自治代理的安全性 cs.LG

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2410.17520v2) [paper-pdf](http://arxiv.org/pdf/2410.17520v2)

**Authors**: Juyong Lee, Dongyoon Hahm, June Suk Choi, W. Bradley Knox, Kimin Lee

**Abstract**: Autonomous agents powered by large language models (LLMs) show promising potential in assistive tasks across various domains, including mobile device control. As these agents interact directly with personal information and device settings, ensuring their safe and reliable behavior is crucial to prevent undesirable outcomes. However, no benchmark exists for standardized evaluation of the safety of mobile device-control agents. In this work, we introduce MobileSafetyBench, a benchmark designed to evaluate the safety of device-control agents within a realistic mobile environment based on Android emulators. We develop a diverse set of tasks involving interactions with various mobile applications, including messaging and banking applications, challenging agents with managing risks encompassing misuse and negative side effects. These tasks include tests to evaluate the safety of agents in daily scenarios as well as their robustness against indirect prompt injection attacks. Our experiments demonstrate that baseline agents, based on state-of-the-art LLMs, often fail to effectively prevent harm while performing the tasks. To mitigate these safety concerns, we propose a prompting method that encourages agents to prioritize safety considerations. While this method shows promise in promoting safer behaviors, there is still considerable room for improvement to fully earn user trust. This highlights the urgent need for continued research to develop more robust safety mechanisms in mobile environments. We open-source our benchmark at: https://mobilesafetybench.github.io/.

摘要: 由大语言模型(LLM)驱动的自主代理在包括移动设备控制在内的各个领域的辅助任务中显示出巨大的潜力。由于这些代理直接与个人信息和设备设置交互，因此确保其安全可靠的行为对于防止不良结果至关重要。然而，移动设备控制代理的安全标准化评估尚不存在基准。在这项工作中，我们引入了MobileSafetyBch，这是一个旨在基于Android模拟器在现实移动环境中评估设备控制代理的安全性的基准测试。我们开发了一套多样化的任务，涉及与各种移动应用程序的交互，包括消息传递和银行应用程序，挑战代理商管理包括滥用和负面副作用在内的风险。这些任务包括测试，以评估代理在日常场景中的安全性，以及它们对间接快速注入攻击的稳健性。我们的实验表明，基于最先进的LLM的基线代理在执行任务时往往无法有效地防止伤害。为了缓解这些安全问题，我们提出了一种提示方法，鼓励工程师优先考虑安全因素。虽然这种方法在促进更安全的行为方面表现出了希望，但要充分赢得用户信任，仍有相当大的改进空间。这突显了迫切需要继续研究，以在移动环境中开发更强大的安全机制。我们在https://mobilesafetybench.github.io/.上开放了我们的基准测试



## **3. PrisonBreak: Jailbreaking Large Language Models with Fewer Than Twenty-Five Targeted Bit-flips**

Prison Break：越狱大型语言模型，目标位翻转少于25个 cs.CR

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07192v1) [paper-pdf](http://arxiv.org/pdf/2412.07192v1)

**Authors**: Zachary Coalson, Jeonghyun Woo, Shiyang Chen, Yu Sun, Lishan Yang, Prashant Nair, Bo Fang, Sanghyun Hong

**Abstract**: We introduce a new class of attacks on commercial-scale (human-aligned) language models that induce jailbreaking through targeted bitwise corruptions in model parameters. Our adversary can jailbreak billion-parameter language models with fewer than 25 bit-flips in all cases$-$and as few as 5 in some$-$using up to 40$\times$ less bit-flips than existing attacks on computer vision models at least 100$\times$ smaller. Unlike prompt-based jailbreaks, our attack renders these models in memory 'uncensored' at runtime, allowing them to generate harmful responses without any input modifications. Our attack algorithm efficiently identifies target bits to flip, offering up to 20$\times$ more computational efficiency than previous methods. This makes it practical for language models with billions of parameters. We show an end-to-end exploitation of our attack using software-induced fault injection, Rowhammer (RH). Our work examines 56 DRAM RH profiles from DDR4 and LPDDR4X devices with different RH vulnerabilities. We show that our attack can reliably induce jailbreaking in systems similar to those affected by prior bit-flip attacks. Moreover, our approach remains effective even against highly RH-secure systems (e.g., 46$\times$ more secure than previously tested systems). Our analyses further reveal that: (1) models with less post-training alignment require fewer bit flips to jailbreak; (2) certain model components, such as value projection layers, are substantially more vulnerable than others; and (3) our method is mechanistically different than existing jailbreaks. Our findings highlight a pressing, practical threat to the language model ecosystem and underscore the need for research to protect these models from bit-flip attacks.

摘要: 我们在商业规模(人类对齐的)语言模型上引入了一类新的攻击，这些攻击通过模型参数中有针对性的逐位破坏来诱导越狱。我们的对手可以用不到25个比特翻转的语言模型越狱，所有情况下都不到25个比特翻转，在一些$-$中只有5个，使用多达40个比特翻转，比对计算机视觉模型的现有攻击少至少100$\×$。与基于提示的越狱不同，我们的攻击在运行时将这些模型呈现在内存中，不受审查，允许它们在不修改任何输入的情况下生成有害的响应。我们的攻击算法有效地识别要翻转的目标比特，比以前的方法提供了高达20美元\倍的计算效率。这使得它适用于具有数十亿个参数的语言模型。我们使用软件诱导的故障注入Rowhammer(RH)展示了对我们的攻击的端到端攻击。我们的工作检查了来自具有不同RH漏洞的DDR4和LPDDR4X设备的56个DRAM RH配置文件。我们证明了我们的攻击可以可靠地在类似于先前受比特翻转攻击影响的系统中诱导越狱。此外，我们的方法即使对高度RH安全的系统也是有效的(例如，比之前测试的系统安全46美元\倍)。我们的分析进一步表明：(1)训练后对齐较少的模型需要较少的比特翻转越狱；(2)某些模型组件，如值投影层，比其他组件更容易受到攻击；(3)我们的方法与现有的越狱方法在机械上不同。我们的发现突显了语言模型生态系统面临的紧迫、实际的威胁，并强调了研究保护这些模型免受比特翻转攻击的必要性。



## **4. Defensive Dual Masking for Robust Adversarial Defense**

防御性双重掩蔽实现强大的对抗性防御 cs.CL

First version

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07078v1) [paper-pdf](http://arxiv.org/pdf/2412.07078v1)

**Authors**: Wangli Yang, Jie Yang, Yi Guo, Johan Barthelemy

**Abstract**: The field of textual adversarial defenses has gained considerable attention in recent years due to the increasing vulnerability of natural language processing (NLP) models to adversarial attacks, which exploit subtle perturbations in input text to deceive models. This paper introduces the Defensive Dual Masking (DDM) algorithm, a novel approach designed to enhance model robustness against such attacks. DDM utilizes a unique adversarial training strategy where [MASK] tokens are strategically inserted into training samples to prepare the model to handle adversarial perturbations more effectively. During inference, potentially adversarial tokens are dynamically replaced with [MASK] tokens to neutralize potential threats while preserving the core semantics of the input. The theoretical foundation of our approach is explored, demonstrating how the selective masking mechanism strengthens the model's ability to identify and mitigate adversarial manipulations. Our empirical evaluation across a diverse set of benchmark datasets and attack mechanisms consistently shows that DDM outperforms state-of-the-art defense techniques, improving model accuracy and robustness. Moreover, when applied to Large Language Models (LLMs), DDM also enhances their resilience to adversarial attacks, providing a scalable defense mechanism for large-scale NLP applications.

摘要: 近年来，由于自然语言处理(NLP)模型越来越容易受到敌意攻击，利用输入文本中的细微扰动来欺骗模型，文本对抗防御领域受到了相当大的关注。介绍了防御性双重掩蔽(DDM)算法，这是一种新的方法，旨在增强模型对此类攻击的稳健性。DDM利用一种独特的对抗性训练策略，其中[MASK]标记被战略性地插入到训练样本中，以准备模型以更有效地处理对抗性扰动。在推理过程中，潜在的敌意令牌被动态地替换为[掩码]令牌，以中和潜在的威胁，同时保留输入的核心语义。探讨了我们方法的理论基础，展示了选择性掩蔽机制如何增强模型识别和缓解对手操纵的能力。我们对不同的基准数据集和攻击机制进行的经验评估一致表明，DDM的性能优于最先进的防御技术，提高了模型的准确性和稳健性。此外，当应用于大型语言模型时，DDM还增强了它们对对手攻击的韧性，为大规模NLP应用提供了一种可扩展的防御机制。



## **5. Unseen Attack Detection in Software-Defined Networking Using a BERT-Based Large Language Model**

使用基于BERT的大型语言模型在软件定义网络中检测隐形攻击 cs.CR

Mohammed N. Swileh is first author. Shengli Zhang is corresponding  author

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2412.06239v1) [paper-pdf](http://arxiv.org/pdf/2412.06239v1)

**Authors**: Mohammed N. Swileh, Shengli Zhang

**Abstract**: Software defined networking (SDN) represents a transformative shift in network architecture by decoupling the control plane from the data plane, enabling centralized and flexible management of network resources. However, this architectural shift introduces significant security challenges, as SDN's centralized control becomes an attractive target for various types of attacks. While current research has yielded valuable insights into attack detection in SDN, critical gaps remain. Addressing challenges in feature selection, broadening the scope beyond DDoS attacks, strengthening attack decisions based on multi flow analysis, and building models capable of detecting unseen attacks that they have not been explicitly trained on are essential steps toward advancing security in SDN. In this paper, we introduce a novel approach that leverages Natural Language Processing (NLP) and the pre trained BERT base model to enhance attack detection in SDN. Our approach transforms network flow data into a format interpretable by language models, allowing BERT to capture intricate patterns and relationships within network traffic. By using Random Forest for feature selection, we optimize model performance and reduce computational overhead, ensuring accurate detection. Attack decisions are made based on several flows, providing stronger and more reliable detection of malicious traffic. Furthermore, our approach is specifically designed to detect previously unseen attacks, offering a solution for identifying threats that the model was not explicitly trained on. To rigorously evaluate our approach, we conducted experiments in two scenarios: one focused on detecting known attacks, achieving 99.96% accuracy, and another on detecting unseen attacks, where our model achieved 99.96% accuracy, demonstrating the robustness of our approach in detecting evolving threats to improve the security of SDN networks.

摘要: 软件定义网络(SDN)通过将控制平面与数据平面分离，实现对网络资源的集中灵活管理，代表了网络架构的变革性转变。然而，这种架构转变带来了重大的安全挑战，因为SDN的集中控制成为各种类型攻击的诱人目标。虽然目前的研究已经对SDN中的攻击检测产生了宝贵的见解，但仍然存在关键差距。解决功能选择方面的挑战，扩大DDoS攻击之外的范围，加强基于多流分析的攻击决策，以及构建能够检测未经明确培训的未见攻击的模型，这些都是提高SDN安全性的关键步骤。在本文中，我们提出了一种新的方法，利用自然语言处理(NLP)和预训练的BERT基模型来增强SDN中的攻击检测。我们的方法将网络流量数据转换为语言模型可解释的格式，使BERT能够捕获网络流量中复杂的模式和关系。通过使用随机森林进行特征选择，我们优化了模型的性能，减少了计算开销，确保了准确的检测。攻击决策是基于多个流做出的，从而提供更强大、更可靠的恶意流量检测。此外，我们的方法是专门为检测以前未见过的攻击而设计的，为识别模型未明确训练过的威胁提供了解决方案。为了严格评估我们的方法，我们在两个场景中进行了实验：一个是专注于检测已知攻击，达到99.96%的准确率；另一个是检测未知攻击，其中我们的模型达到了99.96%的准确率，证明了我们的方法在检测不断变化的威胁以提高SDN网络安全方面的健壮性。



## **6. Trustful LLMs: Customizing and Grounding Text Generation with Knowledge Bases and Dual Decoders**

值得信赖的LLM：使用知识库和双解码器定制和基础文本生成 cs.CL

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2411.07870v4) [paper-pdf](http://arxiv.org/pdf/2411.07870v4)

**Authors**: Xiaofeng Zhu, Jaya Krishna Mandivarapu

**Abstract**: Although people are impressed by the content generation skills of large language models, the use of LLMs, such as ChatGPT, is limited by the domain grounding of the content. The correctness and groundedness of the generated content need to be based on a verified context, such as results from Retrieval-Augmented Generation (RAG). One important issue when adapting LLMs to a customized domain is that the generated responses are often incomplete, or the additions are not verified and may even be hallucinated. Prior studies on hallucination detection have focused on evaluation metrics, which are not easily adaptable to dynamic domains and can be vulnerable to attacks like jail-breaking. In this work, we propose 1) a post-processing algorithm that leverages knowledge triplets in RAG context to correct hallucinations and 2) a dual-decoder model that fuses RAG context to guide the generation process.

摘要: 尽管人们对大型语言模型的内容生成技能印象深刻，但ChatGPT等LLM的使用受到内容领域基础的限制。生成内容的正确性和可信度需要基于经过验证的上下文，例如检索增强生成（RAG）的结果。将LLM调整到定制域时的一个重要问题是，生成的响应通常不完整，或者添加未经验证，甚至可能出现幻觉。之前关于幻觉检测的研究集中在评估指标上，这些指标不容易适应动态领域，并且容易受到越狱等攻击。在这项工作中，我们提出了1）一种利用RAG上下文中的知识三重组来纠正幻觉的后处理算法，以及2）一种融合RAG上下文来指导生成过程的双解码器模型。



## **7. Privacy-Preserving Large Language Models: Mechanisms, Applications, and Future Directions**

保护隐私的大型语言模型：机制、应用和未来方向 cs.CR

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2412.06113v1) [paper-pdf](http://arxiv.org/pdf/2412.06113v1)

**Authors**: Guoshenghui Zhao, Eric Song

**Abstract**: The rapid advancement of large language models (LLMs) has revolutionized natural language processing, enabling applications in diverse domains such as healthcare, finance and education. However, the growing reliance on extensive data for training and inference has raised significant privacy concerns, ranging from data leakage to adversarial attacks. This survey comprehensively explores the landscape of privacy-preserving mechanisms tailored for LLMs, including differential privacy, federated learning, cryptographic protocols, and trusted execution environments. We examine their efficacy in addressing key privacy challenges, such as membership inference and model inversion attacks, while balancing trade-offs between privacy and model utility. Furthermore, we analyze privacy-preserving applications of LLMs in privacy-sensitive domains, highlighting successful implementations and inherent limitations. Finally, this survey identifies emerging research directions, emphasizing the need for novel frameworks that integrate privacy by design into the lifecycle of LLMs. By synthesizing state-of-the-art approaches and future trends, this paper provides a foundation for developing robust, privacy-preserving large language models that safeguard sensitive information without compromising performance.

摘要: 大型语言模型(LLM)的快速发展使自然语言处理发生了革命性的变化，使得医疗保健、金融和教育等不同领域的应用成为可能。然而，越来越多地依赖大量数据进行训练和推理，引发了从数据泄露到对抗性攻击等严重的隐私问题。这项调查全面探索了为LLMS量身定做的隐私保护机制，包括差异隐私、联合学习、加密协议和可信执行环境。我们检查了它们在解决关键隐私挑战方面的有效性，例如成员关系推断和模型反转攻击，同时平衡隐私和模型实用程序之间的权衡。此外，我们分析了LLMS在隐私敏感领域的隐私保护应用，强调了成功的实现和固有的局限性。最后，这项调查确定了新兴的研究方向，强调需要新的框架，将隐私通过设计整合到低成本管理的生命周期中。通过综合最先进的方法和未来的趋势，本文为开发健壮的、保护隐私的大型语言模型提供了基础，这些模型可以在不影响性能的情况下保护敏感信息。



## **8. TrojanRobot: Backdoor Attacks Against LLM-based Embodied Robots in the Physical World**

TrojanRobot：对物理世界中基于LLM的机器人的后门攻击 cs.RO

Initial version with preliminary results. We welcome any feedback or  suggestions

**SubmitDate**: 2024-12-08    [abs](http://arxiv.org/abs/2411.11683v2) [paper-pdf](http://arxiv.org/pdf/2411.11683v2)

**Authors**: Xianlong Wang, Hewen Pan, Hangtao Zhang, Minghui Li, Shengshan Hu, Ziqi Zhou, Lulu Xue, Peijin Guo, Yichen Wang, Wei Wan, Aishan Liu, Leo Yu Zhang

**Abstract**: Robotic manipulation refers to the autonomous handling and interaction of robots with objects using advanced techniques in robotics and artificial intelligence. The advent of powerful tools such as large language models (LLMs) and large vision-language models (LVLMs) has significantly enhanced the capabilities of these robots in environmental perception and decision-making. However, the introduction of these intelligent agents has led to security threats such as jailbreak attacks and adversarial attacks.   In this research, we take a further step by proposing a backdoor attack specifically targeting robotic manipulation and, for the first time, implementing backdoor attack in the physical world. By embedding a backdoor visual language model into the visual perception module within the robotic system, we successfully mislead the robotic arm's operation in the physical world, given the presence of common items as triggers. Experimental evaluations in the physical world demonstrate the effectiveness of the proposed backdoor attack.

摘要: 机器人操纵是指使用机器人学和人工智能的先进技术，自主处理机器人与物体的交互。大型语言模型(LLM)和大型视觉语言模型(LVLM)等强大工具的出现，大大增强了这些机器人在环境感知和决策方面的能力。然而，这些智能代理的引入导致了越狱攻击和对抗性攻击等安全威胁。在这项研究中，我们进一步提出了专门针对机器人操作的后门攻击，并首次在物理世界中实现了后门攻击。通过将后门视觉语言模型嵌入机器人系统的视觉感知模块中，我们成功地误导了机械臂在物理世界中的操作，因为存在共同的物品作为触发器。物理世界中的实验评估证明了所提出的后门攻击的有效性。



## **9. Heuristic-Induced Multimodal Risk Distribution Jailbreak Attack for Multimodal Large Language Models**

启发式多峰大语言模型的多峰风险分布越狱攻击 cs.CR

**SubmitDate**: 2024-12-08    [abs](http://arxiv.org/abs/2412.05934v1) [paper-pdf](http://arxiv.org/pdf/2412.05934v1)

**Authors**: Ma Teng, Jia Xiaojun, Duan Ranjie, Li Xinfeng, Huang Yihao, Chu Zhixuan, Liu Yang, Ren Wenqi

**Abstract**: With the rapid advancement of multimodal large language models (MLLMs), concerns regarding their security have increasingly captured the attention of both academia and industry. Although MLLMs are vulnerable to jailbreak attacks, designing effective multimodal jailbreak attacks poses unique challenges, especially given the distinct protective measures implemented across various modalities in commercial models. Previous works concentrate risks into a single modality, resulting in limited jailbreak performance. In this paper, we propose a heuristic-induced multimodal risk distribution jailbreak attack method, called HIMRD, which consists of two elements: multimodal risk distribution strategy and heuristic-induced search strategy. The multimodal risk distribution strategy is used to segment harmful instructions across multiple modalities to effectively circumvent MLLMs' security protection. The heuristic-induced search strategy identifies two types of prompts: the understanding-enhancing prompt, which helps the MLLM reconstruct the malicious prompt, and the inducing prompt, which increases the likelihood of affirmative outputs over refusals, enabling a successful jailbreak attack. Extensive experiments demonstrate that this approach effectively uncovers vulnerabilities in MLLMs, achieving an average attack success rate of 90% across seven popular open-source MLLMs and an average attack success rate of around 68% in three popular closed-source MLLMs. Our code will coming soon. Warning: This paper contains offensive and harmful examples, reader discretion is advised.

摘要: 随着多通道大语言模型的快速发展，对其安全性的关注日益引起学术界和工业界的关注。尽管大规模杀伤性武器易受越狱攻击，但设计有效的多模式越狱攻击是一个独特的挑战，特别是考虑到商业模式中的各种模式实施了不同的保护措施。以前的工作将风险集中到单一模式中，导致有限的越狱性能。本文提出了一种启发式多通道风险分布越狱攻击方法HIMRD，该方法由两部分组成：多通道风险分布策略和启发式搜索策略。多模式风险分配策略用于跨多个模式分割有害指令，以有效规避MLLMS的安全保护。启发式搜索策略识别两种类型的提示：促进理解的提示和诱导性提示，前者帮助MLLM重建恶意提示，后者增加肯定输出超过拒绝的可能性，从而实现成功的越狱攻击。大量实验表明，该方法有效地发现了MLLMS中的漏洞，在七个流行的开源MLLMS上的平均攻击成功率达到了90%，在三个流行的闭源MLLMS上的平均攻击成功率达到了68%左右。我们的代码很快就会出来。警告：本文包含冒犯性和有害的例子，建议读者酌情处理。



## **10. Large Language Models Merging for Enhancing the Link Stealing Attack on Graph Neural Networks**

大型语言模型合并以增强图神经网络的链接窃取攻击 cs.CR

Link Stealing Attacks, Large Language Models, Graph Neural Networks,  Privacy Attacks, Model Merging

**SubmitDate**: 2024-12-08    [abs](http://arxiv.org/abs/2412.05830v1) [paper-pdf](http://arxiv.org/pdf/2412.05830v1)

**Authors**: Faqian Guan, Tianqing Zhu, Wenhan Chang, Wei Ren, Wanlei Zhou

**Abstract**: Graph Neural Networks (GNNs), specifically designed to process the graph data, have achieved remarkable success in various applications. Link stealing attacks on graph data pose a significant privacy threat, as attackers aim to extract sensitive relationships between nodes (entities), potentially leading to academic misconduct, fraudulent transactions, or other malicious activities. Previous studies have primarily focused on single datasets and did not explore cross-dataset attacks, let alone attacks that leverage the combined knowledge of multiple attackers. However, we find that an attacker can combine the data knowledge of multiple attackers to create a more effective attack model, which can be referred to cross-dataset attacks. Moreover, if knowledge can be extracted with the help of Large Language Models (LLMs), the attack capability will be more significant. In this paper, we propose a novel link stealing attack method that takes advantage of cross-dataset and Large Language Models (LLMs). The LLM is applied to process datasets with different data structures in cross-dataset attacks. Each attacker fine-tunes the LLM on their specific dataset to generate a tailored attack model. We then introduce a novel model merging method to integrate the parameters of these attacker-specific models effectively. The result is a merged attack model with superior generalization capabilities, enabling effective attacks not only on the attackers' datasets but also on previously unseen (out-of-domain) datasets. We conducted extensive experiments in four datasets to demonstrate the effectiveness of our method. Additional experiments with three different GNN and LLM architectures further illustrate the generality of our approach.

摘要: 专门为处理图形数据而设计的图形神经网络(GNN)在各种应用中都取得了显著的成功。对图表数据的链接窃取攻击构成了严重的隐私威胁，因为攻击者的目标是提取节点(实体)之间的敏感关系，可能会导致学术不端、欺诈性交易或其他恶意活动。以前的研究主要集中在单个数据集上，没有探索跨数据集攻击，更不用说利用多个攻击者的综合知识进行攻击了。然而，我们发现一个攻击者可以结合多个攻击者的数据知识来创建更有效的攻击模型，这可以被称为跨数据集攻击。此外，如果能够借助大型语言模型(LLM)来提取知识，则攻击能力将更加显著。本文提出了一种利用跨数据集和大语言模型的链接窃取攻击方法。LLM用于处理跨数据集攻击中具有不同数据结构的数据集。每个攻击者在其特定的数据集上微调LLM，以生成定制的攻击模型。然后，我们引入了一种新的模型合并方法来有效地集成这些特定于攻击者的模型的参数。其结果是一个具有卓越泛化能力的合并攻击模型，不仅可以对攻击者的数据集进行有效攻击，还可以对以前未见的(域外)数据集进行有效攻击。为了证明该方法的有效性，我们在四个数据集中进行了大量的实验。用三种不同的GNN和LLM架构进行的额外实验进一步说明了我们方法的通用性。



## **11. Jailbreak Large Vision-Language Models Through Multi-Modal Linkage**

通过多模式联动的越狱大型视觉语言模型 cs.CV

**SubmitDate**: 2024-12-07    [abs](http://arxiv.org/abs/2412.00473v3) [paper-pdf](http://arxiv.org/pdf/2412.00473v3)

**Authors**: Yu Wang, Xiaofei Zhou, Yichen Wang, Geyuan Zhang, Tianxing He

**Abstract**: With the significant advancement of Large Vision-Language Models (VLMs), concerns about their potential misuse and abuse have grown rapidly. Previous studies have highlighted VLMs' vulnerability to jailbreak attacks, where carefully crafted inputs can lead the model to produce content that violates ethical and legal standards. However, existing methods struggle against state-of-the-art VLMs like GPT-4o, due to the over-exposure of harmful content and lack of stealthy malicious guidance. In this work, we propose a novel jailbreak attack framework: Multi-Modal Linkage (MML) Attack. Drawing inspiration from cryptography, MML utilizes an encryption-decryption process across text and image modalities to mitigate over-exposure of malicious information. To align the model's output with malicious intent covertly, MML employs a technique called "evil alignment", framing the attack within a video game production scenario. Comprehensive experiments demonstrate MML's effectiveness. Specifically, MML jailbreaks GPT-4o with attack success rates of 97.80% on SafeBench, 98.81% on MM-SafeBench and 99.07% on HADES-Dataset. Our code is available at https://github.com/wangyu-ovo/MML

摘要: 随着大型视觉语言模型(VLM)的显著进步，人们对其潜在的滥用和滥用的担忧迅速增长。之前的研究已经强调了VLMS在越狱攻击中的脆弱性，在越狱攻击中，精心制作的输入可能导致该模型产生违反道德和法律标准的内容。然而，由于过度暴露有害内容和缺乏隐蔽的恶意指导，现有的方法难以对抗像GPT-40这样的最先进的VLM。在这项工作中，我们提出了一种新的越狱攻击框架：多模式联动攻击。MML从密码学中获得灵感，利用跨文本和图像通道的加密-解密过程来减少恶意信息的过度暴露。为了秘密地将模型的输出与恶意意图对齐，MML采用了一种称为“邪恶对齐”的技术，将攻击框置于视频游戏制作场景中。综合实验证明了MML的有效性。具体地说，MML越狱GPT-4o在SafeBitch上的攻击成功率为97.80%，在MM-SafeBch上的攻击成功率为98.81%，在HADES-DataSet上的攻击成功率为99.07%。我们的代码可以在https://github.com/wangyu-ovo/MML上找到



## **12. Privacy Risks in Reinforcement Learning for Household Robots**

家用机器人强化学习中的隐私风险 cs.RO

7 pages, 4 figures, 2 tables

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2306.09273v3) [paper-pdf](http://arxiv.org/pdf/2306.09273v3)

**Authors**: Miao Li, Wenhao Ding, Ding Zhao

**Abstract**: The prominence of embodied Artificial Intelligence (AI), which empowers robots to navigate, perceive, and engage within virtual environments, has attracted significant attention, owing to the remarkable advances in computer vision and large language models. Privacy emerges as a pivotal concern within the realm of embodied AI, as the robot accesses substantial personal information. However, the issue of privacy leakage in embodied AI tasks, particularly concerning reinforcement learning algorithms, has not received adequate consideration in research. This paper aims to address this gap by proposing an attack on the training process of the value-based algorithm and the gradient-based algorithm, utilizing gradient inversion to reconstruct states, actions, and supervisory signals. The choice of using gradients for the attack is motivated by the fact that commonly employed federated learning techniques solely utilize gradients computed based on private user data to optimize models, without storing or transmitting the data to public servers. Nevertheless, these gradients contain sufficient information to potentially expose private data. To validate our approach, we conducted experiments on the AI2THOR simulator and evaluated our algorithm on active perception, a prevalent task in embodied AI. The experimental results demonstrate the effectiveness of our method in successfully reconstructing all information from the data in 120 room layouts. Check our website for videos.

摘要: 由于计算机视觉和大型语言模型的显著进步，使机器人能够在虚拟环境中导航、感知和参与的嵌入式人工智能(AI)的突出地位引起了人们的极大关注。随着机器人访问大量的个人信息，隐私成为体现人工智能领域的一个关键问题。然而，体验式人工智能任务中的隐私泄露问题，特别是强化学习算法的隐私泄露问题，在研究中并没有得到足够的考虑。为了解决这一问题，本文提出了一种攻击基于值的算法和基于梯度的算法的训练过程，利用梯度求逆来重建状态、动作和监控信号。选择使用梯度进行攻击的动机是，通常使用的联合学习技术仅使用基于私有用户数据计算的梯度来优化模型，而不将数据存储或传输到公共服务器。然而，这些渐变包含了足够的信息来潜在地暴露私有数据。为了验证我们的方法，我们在AI2THOR模拟器上进行了实验，并对我们的算法进行了评估，主动感知是体现人工智能中的一个普遍任务。实验结果表明，该方法能够从120个房间布局的数据中成功地重建所有信息。请查看我们的网站上的视频。



## **13. WAPITI: A Watermark for Finetuned Open-Source LLMs**

WAPITI：Finetuned开源LLM的水印 cs.CR

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2410.06467v2) [paper-pdf](http://arxiv.org/pdf/2410.06467v2)

**Authors**: Lingjie Chen, Ruizhong Qiu, Siyu Yuan, Zhining Liu, Tianxin Wei, Hyunsik Yoo, Zhichen Zeng, Deqing Yang, Hanghang Tong

**Abstract**: Watermarking of large language models (LLMs) generation embeds an imperceptible statistical pattern within texts, making it algorithmically detectable. Watermarking is a promising method for addressing potential harm and biases from LLMs, as it enables traceability, accountability, and detection of manipulated content, helping to mitigate unintended consequences. However, for open-source models, watermarking faces two major challenges: (i) incompatibility with fine-tuned models, and (ii) vulnerability to fine-tuning attacks. In this work, we propose WAPITI, a new method that transfers watermarking from base models to fine-tuned models through parameter integration. To the best of our knowledge, we propose the first watermark for fine-tuned open-source LLMs that preserves their fine-tuned capabilities. Furthermore, our approach offers an effective defense against fine-tuning attacks. We test our method on various model architectures and watermarking strategies. Results demonstrate that our method can successfully inject watermarks and is highly compatible with fine-tuned models. Additionally, we offer an in-depth analysis of how parameter editing influences the watermark strength and overall capabilities of the resulting models.

摘要: 大语言模型(LLMS)水印生成在文本中嵌入了一种不可察觉的统计模式，使其在算法上是可检测的。水印是一种很有前途的方法，可以解决LLMS的潜在危害和偏见，因为它能够跟踪、问责和检测被篡改的内容，有助于减轻意外后果。然而，对于开源模型，水印面临着两大挑战：(I)与微调模型不兼容，(Ii)易受微调攻击。在这项工作中，我们提出了Wapiti，一种新的方法，通过参数积分将水印从基本模型转移到微调模型。就我们所知，我们建议为保持其微调能力的开放源码LLM提供第一个水印。此外，我们的方法提供了针对微调攻击的有效防御。我们在不同的模型架构和水印策略上测试了我们的方法。实验结果表明，该方法能够成功地嵌入水印，并且与微调模型具有很好的兼容性。此外，我们还深入分析了参数编辑如何影响最终模型的水印强度和整体性能。



## **14. A Practical Examination of AI-Generated Text Detectors for Large Language Models**

大型语言模型的人工智能生成文本检测器的实践检验 cs.CL

8 pages. Submitted to ARR October cycle

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2412.05139v1) [paper-pdf](http://arxiv.org/pdf/2412.05139v1)

**Authors**: Brian Tufts, Xuandong Zhao, Lei Li

**Abstract**: The proliferation of large language models has raised growing concerns about their misuse, particularly in cases where AI-generated text is falsely attributed to human authors. Machine-generated content detectors claim to effectively identify such text under various conditions and from any language model. This paper critically evaluates these claims by assessing several popular detectors (RADAR, Wild, T5Sentinel, Fast-DetectGPT, GPTID, LogRank, Binoculars) on a range of domains, datasets, and models that these detectors have not previously encountered. We employ various prompting strategies to simulate adversarial attacks, demonstrating that even moderate efforts can significantly evade detection. We emphasize the importance of the true positive rate at a specific false positive rate (TPR@FPR) metric and demonstrate that these detectors perform poorly in certain settings, with TPR@.01 as low as 0\%. Our findings suggest that both trained and zero-shot detectors struggle to maintain high sensitivity while achieving a reasonable true positive rate.

摘要: 大型语言模型的激增引发了人们对它们滥用的日益担忧，特别是在人工智能生成的文本被错误地归因于人类作者的情况下。机器生成的内容检测器声称可以在各种条件下从任何语言模型有效地识别此类文本。本文通过评估几种流行的探测器(雷达、Wild、T5Sentinel、Fast-DetectGPT、GPTID、logrank、双筒望远镜)在这些探测器以前从未遇到的一系列域、数据集和模型上对这些声称进行了批判性评估。我们使用各种提示策略来模拟对抗性攻击，表明即使是适度的攻击也可以显著地躲避检测。我们强调了在特定的假阳性率(TPR@fPR)度量下真阳性率的重要性，并证明了这些检测器在某些设置下的性能很差，TPR@0.01低至0\%。我们的发现表明，训练有素的探测器和零射探测器都很难在保持高灵敏度的同时获得合理的真阳性率。



## **15. MultiTrust: A Comprehensive Benchmark Towards Trustworthy Multimodal Large Language Models**

MultiTrust：值得信赖的多模式大型语言模型的综合基准 cs.CL

100 pages, 84 figures, 33 tables

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2406.07057v2) [paper-pdf](http://arxiv.org/pdf/2406.07057v2)

**Authors**: Yichi Zhang, Yao Huang, Yitong Sun, Chang Liu, Zhe Zhao, Zhengwei Fang, Yifan Wang, Huanran Chen, Xiao Yang, Xingxing Wei, Hang Su, Yinpeng Dong, Jun Zhu

**Abstract**: Despite the superior capabilities of Multimodal Large Language Models (MLLMs) across diverse tasks, they still face significant trustworthiness challenges. Yet, current literature on the assessment of trustworthy MLLMs remains limited, lacking a holistic evaluation to offer thorough insights into future improvements. In this work, we establish MultiTrust, the first comprehensive and unified benchmark on the trustworthiness of MLLMs across five primary aspects: truthfulness, safety, robustness, fairness, and privacy. Our benchmark employs a rigorous evaluation strategy that addresses both multimodal risks and cross-modal impacts, encompassing 32 diverse tasks with self-curated datasets. Extensive experiments with 21 modern MLLMs reveal some previously unexplored trustworthiness issues and risks, highlighting the complexities introduced by the multimodality and underscoring the necessity for advanced methodologies to enhance their reliability. For instance, typical proprietary models still struggle with the perception of visually confusing images and are vulnerable to multimodal jailbreaking and adversarial attacks; MLLMs are more inclined to disclose privacy in text and reveal ideological and cultural biases even when paired with irrelevant images in inference, indicating that the multimodality amplifies the internal risks from base LLMs. Additionally, we release a scalable toolbox for standardized trustworthiness research, aiming to facilitate future advancements in this important field. Code and resources are publicly available at: https://multi-trust.github.io/.

摘要: 尽管多模式大型语言模型(MLLM)在不同的任务中具有卓越的能力，但它们仍然面临着重大的可信性挑战。然而，目前关于评估值得信赖的MLLMS的文献仍然有限，缺乏全面的评估来提供对未来改进的透彻见解。在这项工作中，我们建立了多重信任，这是第一个关于MLLMS可信度的全面和统一的基准，涉及五个主要方面：真实性、安全性、健壮性、公平性和隐私性。我们的基准采用了严格的评估战略，同时应对多式联运风险和跨联运影响，包括32项不同的任务和自我管理的数据集。对21个现代多模式管理进行的广泛实验揭示了一些以前从未探索过的可信度问题和风险，突显了多模式带来的复杂性，并强调了先进方法提高其可靠性的必要性。例如，典型的专有模型仍然难以识别视觉上令人困惑的图像，容易受到多模式越狱和敌意攻击；MLLM更倾向于在文本中泄露隐私，甚至在推理中与无关图像搭配使用时也会暴露意识形态和文化偏见，这表明多模式放大了基本LLM的内部风险。此外，我们还发布了一个用于标准化可信度研究的可扩展工具箱，旨在促进这一重要领域的未来发展。代码和资源可在以下网址公开获得：https://multi-trust.github.io/.



## **16. PropertyGPT: LLM-driven Formal Verification of Smart Contracts through Retrieval-Augmented Property Generation**

PropertyGPT：通过检索增强属性生成，LLM驱动的智能合同形式验证 cs.SE

Accepted by NDSS Symposium 2025. Please cite the conference version  of this paper, e.g., "Ye Liu, Yue Xue, Daoyuan Wu, Yuqiang Sun, Yi Li,  Miaolei Shi, Yang Liu. PropertyGPT: LLM-driven Formal Verification of Smart  Contracts through Retrieval-Augmented Property Generation. In 32nd Annual  Network and Distributed System Security Symposium (NDSS 2025)."

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2405.02580v2) [paper-pdf](http://arxiv.org/pdf/2405.02580v2)

**Authors**: Ye Liu, Yue Xue, Daoyuan Wu, Yuqiang Sun, Yi Li, Miaolei Shi, Yang Liu

**Abstract**: With recent advances in large language models (LLMs), this paper explores the potential of leveraging state-of-the-art LLMs,such as GPT-4, to transfer existing human-written properties (e.g.,those from Certora auditing reports) and automatically generate customized properties for unknown code. To this end, we embed existing properties into a vector database and retrieve a reference property for LLM-based in-context learning to generate a new property for a given code. While this basic process is relatively straightforward, ensuring that the generated properties are (i) compilable, (ii) appropriate, and (iii) verifiable presents challenges. To address (i), we use the compilation and static analysis feedback as an external oracle to guide LLMs in iteratively revising the generated properties. For (ii), we consider multiple dimensions of similarity to rank the properties and employ a weighted algorithm to identify the top-K properties as the final result. For (iii), we design a dedicated prover to formally verify the correctness of the generated properties. We have implemented these strategies into a novel LLM-based property generation tool called PropertyGPT. Our experiments show that PropertyGPT can generate comprehensive and high-quality properties, achieving an 80% recall compared to the ground truth. It successfully detected 26 CVEs/attack incidents out of 37 tested and also uncovered 12 zero-day vulnerabilities, leading to $8,256 in bug bounty rewards.

摘要: 随着大型语言模型(LLM)的最新进展，本文探索了利用最先进的LLM(如GPT-4)来转移现有的人工编写的属性(例如，来自Certora审计报告的属性)并自动为未知代码生成定制属性的潜力。为此，我们将现有属性嵌入到向量数据库中，并检索一个参考属性，用于基于LLM的上下文中学习，以生成给定代码的新属性。虽然这一基本过程相对简单，但确保生成的属性是(I)可编译的、(Ii)适当的和(Iii)可验证的，这是一个挑战。为了解决(I)，我们使用编译和静态分析反馈作为外部预言来指导LLM迭代地修改生成的属性。对于(Ii)，我们考虑多个维度的相似性来对属性进行排序，并使用加权算法来识别TOP-K属性作为最终结果。对于(Iii)，我们设计了一个专用的证明器来形式化地验证所生成的属性的正确性。我们已经将这些策略实现到一个新的基于LLM的属性生成工具PropertyGPT中。我们的实验表明，PropertyGPT可以生成全面的高质量属性，与基本事实相比，召回率达到80%。它在37个测试中成功检测到26个CVE/攻击事件，还发现了12个零日漏洞，导致了8,256美元的漏洞赏金。



## **17. Plentiful Jailbreaks with String Compositions**

弦乐作品丰富越狱 cs.CL

NeurIPS SoLaR Workshop 2024

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2411.01084v2) [paper-pdf](http://arxiv.org/pdf/2411.01084v2)

**Authors**: Brian R. Y. Huang

**Abstract**: Large language models (LLMs) remain vulnerable to a slew of adversarial attacks and jailbreaking methods. One common approach employed by white-hat attackers, or red-teamers, is to process model inputs and outputs using string-level obfuscations, which can include leetspeak, rotary ciphers, Base64, ASCII, and more. Our work extends these encoding-based attacks by unifying them in a framework of invertible string transformations. With invertibility, we can devise arbitrary string compositions, defined as sequences of transformations, that we can encode and decode end-to-end programmatically. We devise a automated best-of-n attack that samples from a combinatorially large number of string compositions. Our jailbreaks obtain competitive attack success rates on several leading frontier models when evaluated on HarmBench, highlighting that encoding-based attacks remain a persistent vulnerability even in advanced LLMs.

摘要: 大型语言模型（LLM）仍然容易受到一系列对抗攻击和越狱方法的影响。白帽攻击者或红团队使用的一种常见方法是使用字符串级混淆处理模型输入和输出，其中可以包括leetspeak、旋转密码、Base 64、ASC等。我们的工作通过将这些基于编码的攻击统一到可逆字符串转换的框架中来扩展它们。通过可逆性，我们可以设计任意的字符串组合，定义为转换序列，我们可以通过编程方式进行端到端编码和解码。我们设计了一种自动化的n中最佳攻击，该攻击从组合上大量的字符串组合中进行采样。在HarmBench上进行评估时，我们的越狱在几个领先的前沿模型上获得了有竞争力的攻击成功率，这凸显了即使在高级LLM中，基于编码的攻击仍然是一个持久的漏洞。



## **18. Targeting the Core: A Simple and Effective Method to Attack RAG-based Agents via Direct LLM Manipulation**

瞄准核心：通过直接LLM操纵攻击基于RAG的代理的简单有效方法 cs.AI

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2412.04415v1) [paper-pdf](http://arxiv.org/pdf/2412.04415v1)

**Authors**: Xuying Li, Zhuo Li, Yuji Kosuga, Yasuhiro Yoshida, Victor Bian

**Abstract**: AI agents, powered by large language models (LLMs), have transformed human-computer interactions by enabling seamless, natural, and context-aware communication. While these advancements offer immense utility, they also inherit and amplify inherent safety risks such as bias, fairness, hallucinations, privacy breaches, and a lack of transparency. This paper investigates a critical vulnerability: adversarial attacks targeting the LLM core within AI agents. Specifically, we test the hypothesis that a deceptively simple adversarial prefix, such as \textit{Ignore the document}, can compel LLMs to produce dangerous or unintended outputs by bypassing their contextual safeguards. Through experimentation, we demonstrate a high attack success rate (ASR), revealing the fragility of existing LLM defenses. These findings emphasize the urgent need for robust, multi-layered security measures tailored to mitigate vulnerabilities at the LLM level and within broader agent-based architectures.

摘要: 由大型语言模型（LLM）支持的人工智能代理通过实现无缝、自然和上下文感知的通信来改变了人机交互。虽然这些进步提供了巨大的实用性，但它们也继承和放大了固有的安全风险，例如偏见、公平、幻觉、隐私侵犯和缺乏透明度。本文研究了一个关键漏洞：针对人工智能代理内LLM核心的对抗攻击。具体来说，我们测试了这样的假设：看似简单的对抗性前置码（例如\textit{忽略文档}）可以迫使LLM绕过上下文保障措施来产生危险或非预期的输出。通过实验，我们展示了高攻击成功率（ASB），揭示了现有LLM防御的脆弱性。这些调查结果强调，迫切需要针对LLM级别和更广泛的基于代理的架构中的漏洞量身定制的强大、多层的安全措施。



## **19. Adversarial Attacks on Large Language Models in Medicine**

医学中对大型语言模型的对抗攻击 cs.AI

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2406.12259v2) [paper-pdf](http://arxiv.org/pdf/2406.12259v2)

**Authors**: Yifan Yang, Qiao Jin, Furong Huang, Zhiyong Lu

**Abstract**: The integration of Large Language Models (LLMs) into healthcare applications offers promising advancements in medical diagnostics, treatment recommendations, and patient care. However, the susceptibility of LLMs to adversarial attacks poses a significant threat, potentially leading to harmful outcomes in delicate medical contexts. This study investigates the vulnerability of LLMs to two types of adversarial attacks in three medical tasks. Utilizing real-world patient data, we demonstrate that both open-source and proprietary LLMs are susceptible to manipulation across multiple tasks. This research further reveals that domain-specific tasks demand more adversarial data in model fine-tuning than general domain tasks for effective attack execution, especially for more capable models. We discover that while integrating adversarial data does not markedly degrade overall model performance on medical benchmarks, it does lead to noticeable shifts in fine-tuned model weights, suggesting a potential pathway for detecting and countering model attacks. This research highlights the urgent need for robust security measures and the development of defensive mechanisms to safeguard LLMs in medical applications, to ensure their safe and effective deployment in healthcare settings.

摘要: 将大型语言模型(LLM)集成到医疗保健应用程序中，在医疗诊断、治疗建议和患者护理方面提供了有希望的进步。然而，LLMS对对抗性攻击的敏感性构成了一个重大威胁，可能会在微妙的医疗环境中导致有害后果。本研究调查了LLMS在三个医疗任务中对两种类型的对抗性攻击的脆弱性。利用真实世界的患者数据，我们证明了开源和专有LLM都容易受到跨多个任务的操纵。这项研究进一步表明，特定领域的任务在模型微调中需要比一般领域任务更多的对抗性数据才能有效地执行攻击，特别是对于能力更强的模型。我们发现，虽然整合对抗性数据并不会显著降低医学基准上的整体模型性能，但它确实会导致微调模型权重的显著变化，这表明了一条检测和对抗模型攻击的潜在路径。这项研究强调了迫切需要强有力的安全措施和开发防御机制来保护医疗应用中的低成本管理，以确保其在医疗保健环境中的安全和有效部署。



## **20. Stochastic Monkeys at Play: Random Augmentations Cheaply Break LLM Safety Alignment**

随机猴子在起作用：随机增强轻松打破LLM安全一致 cs.LG

v2: Updated with changes from peer review rebuttal. v1: Version under  peer review

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2411.02785v2) [paper-pdf](http://arxiv.org/pdf/2411.02785v2)

**Authors**: Jason Vega, Junsheng Huang, Gaokai Zhang, Hangoo Kang, Minjia Zhang, Gagandeep Singh

**Abstract**: Safety alignment of Large Language Models (LLMs) has recently become a critical objective of model developers. In response, a growing body of work has been investigating how safety alignment can be bypassed through various jailbreaking methods, such as adversarial attacks. However, these jailbreak methods can be rather costly or involve a non-trivial amount of creativity and effort, introducing the assumption that malicious users are high-resource or sophisticated. In this paper, we study how simple random augmentations to the input prompt affect safety alignment effectiveness in state-of-the-art LLMs, such as Llama 3 and Qwen 2. We perform an in-depth evaluation of 17 different models and investigate the intersection of safety under random augmentations with multiple dimensions: augmentation type, model size, quantization, fine-tuning-based defenses, and decoding strategies (e.g., sampling temperature). We show that low-resource and unsophisticated attackers, i.e. $\textit{stochastic monkeys}$, can significantly improve their chances of bypassing alignment with just 25 random augmentations per prompt. Source code and data: https://github.com/uiuc-focal-lab/stochastic-monkeys/

摘要: 大型语言模型(LLM)的安全一致性最近已成为模型开发人员的一个重要目标。作为回应，越来越多的工作一直在研究如何通过各种越狱方法绕过安全对准，例如对抗性攻击。然而，这些越狱方法可能相当昂贵，或者涉及大量的创造力和努力，从而引入了恶意用户是高资源或老练的假设。在本文中，我们研究了输入提示的简单随机增强如何影响最新的LLMS的安全对齐有效性，例如Llama 3和Qwen 2。我们对17个不同的模型进行了深入的评估，并研究了在多个维度的随机增强下的安全性交集：增强类型、模型大小、量化、基于微调的防御和解码策略(例如采样温度)。我们表明，低资源和简单的攻击者，即$\textit{随机猴子}$，可以显著提高他们绕过对齐的机会，每个提示只需25个随机扩展。源代码和数据：https://github.com/uiuc-focal-lab/stochastic-monkeys/



## **21. Hostility Detection in UK Politics: A Dataset on Online Abuse Targeting MPs**

英国政治中的敌意检测：针对议员的在线虐待数据集 cs.CL

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2412.04046v1) [paper-pdf](http://arxiv.org/pdf/2412.04046v1)

**Authors**: Mugdha Pandya, Mali Jin, Kalina Bontcheva, Diana Maynard

**Abstract**: Numerous politicians use social media platforms, particularly X, to engage with their constituents. This interaction allows constituents to pose questions and offer feedback but also exposes politicians to a barrage of hostile responses, especially given the anonymity afforded by social media. They are typically targeted in relation to their governmental role, but the comments also tend to attack their personal identity. This can discredit politicians and reduce public trust in the government. It can also incite anger and disrespect, leading to offline harm and violence. While numerous models exist for detecting hostility in general, they lack the specificity required for political contexts. Furthermore, addressing hostility towards politicians demands tailored approaches due to the distinct language and issues inherent to each country (e.g., Brexit for the UK). To bridge this gap, we construct a dataset of 3,320 English tweets spanning a two-year period manually annotated for hostility towards UK MPs. Our dataset also captures the targeted identity characteristics (race, gender, religion, none) in hostile tweets. We perform linguistic and topical analyses to delve into the unique content of the UK political data. Finally, we evaluate the performance of pre-trained language models and large language models on binary hostility detection and multi-class targeted identity type classification tasks. Our study offers valuable data and insights for future research on the prevalence and nature of politics-related hostility specific to the UK.

摘要: 许多政客使用社交媒体平台，特别是X，来与他们的选民互动。这种互动允许选民提出问题和提供反馈，但也会让政客们面临一连串的敌意回应，特别是考虑到社交媒体提供的匿名性。他们通常是因为他们的政府角色而成为攻击目标，但这些言论也往往会攻击他们的个人身份。这会败坏政客的声誉，降低公众对政府的信任度。它还可能煽动愤怒和不尊重，导致线下伤害和暴力。虽然存在许多模型来检测总体上的敌意，但它们缺乏政治背景所需的特异性。此外，解决对政客的敌意需要量身定做的方法，因为每个国家都有不同的语言和固有的问题(例如，英国脱欧)。为了弥补这一差距，我们构建了一个包含3320条英语推文的数据集，涵盖了两年的时间段，手动标注了对英国议员的敌意。我们的数据集还捕获了恶意推文中的目标身份特征(种族、性别、宗教、无)。我们进行语言和话题分析，深入研究英国政治数据的独特内容。最后，我们评估了预先训练的语言模型和大语言模型在二元敌意检测和多类目标身份类型分类任务上的性能。我们的研究为未来研究英国特有的与政治相关的敌意的普遍性和性质提供了有价值的数据和见解。



## **22. R-MTLLMF: Resilient Multi-Task Large Language Model Fusion at the Wireless Edge**

R-MTLLMF：无线边缘的弹性多任务大型语言模型融合 eess.SP

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2411.18220v2) [paper-pdf](http://arxiv.org/pdf/2411.18220v2)

**Authors**: Aladin Djuhera, Vlad C. Andrei, Mohsen Pourghasemian, Haris Gacanin, Holger Boche, Walid Saad

**Abstract**: Multi-task large language models (MTLLMs) are important for many applications at the wireless edge, where users demand specialized models to handle multiple tasks efficiently. However, training MTLLMs is complex and exhaustive, particularly when tasks are subject to change. Recently, the concept of model fusion via task vectors has emerged as an efficient approach for combining fine-tuning parameters to produce an MTLLM. In this paper, the problem of enabling edge users to collaboratively craft such MTLMs via tasks vectors is studied, under the assumption of worst-case adversarial attacks. To this end, first the influence of adversarial noise to multi-task model fusion is investigated and a relationship between the so-called weight disentanglement error and the mean squared error (MSE) is derived. Using hypothesis testing, it is directly shown that the MSE increases interference between task vectors, thereby rendering model fusion ineffective. Then, a novel resilient MTLLM fusion (R-MTLLMF) is proposed, which leverages insights about the LLM architecture and fine-tuning process to safeguard task vector aggregation under adversarial noise by realigning the MTLLM. The proposed R-MTLLMF is then compared for both worst-case and ideal transmission scenarios to study the impact of the wireless channel. Extensive model fusion experiments with vision LLMs demonstrate R-MTLLMF's effectiveness, achieving close-to-baseline performance across eight different tasks in ideal noise scenarios and significantly outperforming unprotected model fusion in worst-case scenarios. The results further advocate for additional physical layer protection for a holistic approach to resilience, from both a wireless and LLM perspective.

摘要: 多任务大型语言模型(MTLLM)对于无线边缘的许多应用非常重要，因为用户需要专门的模型来高效地处理多个任务。然而，培训MTLLM是复杂和详尽的，特别是在任务可能发生变化的情况下。最近，基于任务向量的模型融合的概念已经成为一种结合微调参数以产生MTLLM的有效方法。本文在假设最坏情况下的敌意攻击的前提下，研究了边缘用户通过任务向量协作创建MTLM的问题。为此，首先研究了对抗性噪声对多任务模型融合的影响，推导了加权解缠误差与均方误差之间的关系。通过假设检验，直接表明MSE增加了任务向量之间的干扰，从而使模型融合无效。然后，提出了一种新的弹性MTLLM融合算法(R-MTLLMF)，该算法利用对LLM体系结构和微调过程的深入了解，通过重新排列MTLLM来保护对抗噪声下的任务向量聚合。然后将所提出的R-MTLLMF在最坏情况和理想传输场景下进行比较，以研究无线信道的影响。用VISION LLMS进行的大量模型融合实验证明了R-MTLLMF的有效性，在理想噪声场景中，R-MTLLMF在八个不同任务上的性能接近基线，而在最坏情况下，R-MTLLMF的性能明显优于无保护的模型融合。从无线和LLM的角度来看，研究结果进一步倡导为整体恢复方法提供额外的物理层保护。



## **23. AI-based Attacker Models for Enhancing Multi-Stage Cyberattack Simulations in Smart Grids Using Co-Simulation Environments**

基于人工智能的攻击者模型，用于使用联合模拟环境增强智能电网中的多阶段网络攻击模拟 cs.CR

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2412.03979v1) [paper-pdf](http://arxiv.org/pdf/2412.03979v1)

**Authors**: Omer Sen, Christoph Pohl, Immanuel Hacker, Markus Stroot, Andreas Ulbig

**Abstract**: The transition to smart grids has increased the vulnerability of electrical power systems to advanced cyber threats. To safeguard these systems, comprehensive security measures-including preventive, detective, and reactive strategies-are necessary. As part of the critical infrastructure, securing these systems is a major research focus, particularly against cyberattacks. Many methods are developed to detect anomalies and intrusions and assess the damage potential of attacks. However, these methods require large amounts of data, which are often limited or private due to security concerns. We propose a co-simulation framework that employs an autonomous agent to execute modular cyberattacks within a configurable environment, enabling reproducible and adaptable data generation. The impact of virtual attacks is compared to those in a physical lab targeting real smart grids. We also investigate the use of large language models for automating attack generation, though current models on consumer hardware are unreliable. Our approach offers a flexible, versatile source for data generation, aiding in faster prototyping and reducing development resources and time.

摘要: 向智能电网的过渡增加了电力系统在高级网络威胁面前的脆弱性。为了保护这些系统，必须采取全面的安全措施，包括预防、检测和应对策略。作为关键基础设施的一部分，确保这些系统的安全是一个主要的研究重点，特别是针对网络攻击。开发了许多方法来检测异常和入侵并评估攻击的破坏潜力。然而，这些方法需要大量的数据，而出于安全考虑，这些数据往往是有限的或私有的。我们提出了一种协同仿真框架，该框架使用自治代理在可配置的环境中执行模块化的网络攻击，从而实现可重复性和适应性的数据生成。虚拟攻击的影响与物理实验室中针对真实智能电网的攻击进行了比较。我们还研究了使用大型语言模型来自动生成攻击，尽管当前消费者硬件上的模型是不可靠的。我们的方法为数据生成提供了灵活、通用的来源，有助于更快地建立原型，并减少开发资源和时间。



## **24. Mechanistic Unlearning: Robust Knowledge Unlearning and Editing via Mechanistic Localization**

机械性忘记学习：通过机械性本地化稳健的知识忘记学习和编辑 cs.LG

31 pages, 45 figures, 7 tables

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2410.12949v2) [paper-pdf](http://arxiv.org/pdf/2410.12949v2)

**Authors**: Phillip Guo, Aaquib Syed, Abhay Sheshadri, Aidan Ewart, Gintare Karolina Dziugaite

**Abstract**: Methods for knowledge editing and unlearning in large language models seek to edit or remove undesirable knowledge or capabilities without compromising general language modeling performance. This work investigates how mechanistic interpretability -- which, in part, aims to identify model components (circuits) associated to specific interpretable mechanisms that make up a model capability -- can improve the precision and effectiveness of editing and unlearning. We find a stark difference in unlearning and edit robustness when training components localized by different methods. We highlight an important distinction between methods that localize components based primarily on preserving outputs, and those finding high level mechanisms with predictable intermediate states. In particular, localizing edits/unlearning to components associated with the lookup-table mechanism for factual recall 1) leads to more robust edits/unlearning across different input/output formats, and 2) resists attempts to relearn the unwanted information, while also reducing unintended side effects compared to baselines, on both a sports facts dataset and the CounterFact dataset across multiple models. We also find that certain localized edits disrupt the latent knowledge in the model more than any other baselines, making unlearning more robust to various attacks.

摘要: 用于大型语言模型中的知识编辑和去学习的方法寻求在不损害一般语言建模性能的情况下编辑或移除不需要的知识或能力。这项工作调查了机械性可解释性--部分目的是确定与构成模型能力的特定可解释机制相关联的模型组件(电路)--如何提高编辑和取消学习的精确度和有效性。我们发现，当训练不同方法局部化的组件时，忘记学习和编辑健壮性存在明显差异。我们强调了主要基于保留输出来本地化组件的方法与找到具有可预测中间状态的高级机制之间的重要区别。具体地说，对与用于事实回忆的查找表机制相关联的组件的本地化编辑/忘记1)导致跨不同输入/输出格式的更健壮的编辑/忘记，以及2)抵制重新学习不想要的信息的尝试，同时还减少了与基线相比的意外副作用，在多个模型上的体育事实数据集和反事实数据集两者上。我们还发现，与其他基线相比，某些局部编辑对模型中潜在知识的破坏更大，使得遗忘对各种攻击更具健壮性。



## **25. WiS Platform: Enhancing Evaluation of LLM-Based Multi-Agent Systems Through Game-Based Analysis**

WiS平台：通过基于游戏的分析增强对基于LLM的多智能体系统的评估 cs.AI

**SubmitDate**: 2024-12-04    [abs](http://arxiv.org/abs/2412.03359v1) [paper-pdf](http://arxiv.org/pdf/2412.03359v1)

**Authors**: Chengwei Hu, Jianhui Zheng, Yancheng He, Hangyu Guo, Junguang Jiang, Han Zhu, Kai Sun, Yuning Jiang, Wenbo Su, Bo Zheng

**Abstract**: Recent advancements in autonomous multi-agent systems (MAS) based on large language models (LLMs) have enhanced the application scenarios and improved the capability of LLMs to handle complex tasks. Despite demonstrating effectiveness, existing studies still evidently struggle to evaluate, analysis, and reproducibility of LLM-based MAS. In this paper, to facilitate the research on LLM-based MAS, we introduce an open, scalable, and real-time updated platform for accessing and analyzing the LLM-based MAS based on the games Who is Spy?" (WiS). Our platform is featured with three main worths: (1) a unified model evaluate interface that supports models available on Hugging Face; (2) real-time updated leaderboard for model evaluation; (3) a comprehensive evaluation covering game-winning rates, attacking, defense strategies, and reasoning of LLMs. To rigorously test WiS, we conduct extensive experiments coverage of various open- and closed-source LLMs, we find that different agents exhibit distinct and intriguing behaviors in the game. The experimental results demonstrate the effectiveness and efficiency of our platform in evaluating LLM-based MAS. Our platform and its documentation are publicly available at \url{https://whoisspy.ai/}

摘要: 基于大语言模型的自治多智能体系统(MAS)的最新进展增强了LLMS的应用场景，提高了LLMS处理复杂任务的能力。尽管证明了有效性，但现有的研究显然仍难以评估、分析和重复性基于LLM的MAS。为了便于对基于LLM的MAS的研究，我们介绍了一个开放的、可扩展的、实时更新的访问和分析基于LLM的MAS的平台，该平台基于游戏《谁是间谍？(WIS)。我们的平台主要有三个特点：(1)统一的模型评估界面，支持拥抱脸上可用的模型；(2)实时更新的模型评估排行榜；(3)包括胜率、进攻、防守策略和LLMS推理的综合评估。为了严格测试WIS，我们对各种开放和封闭源代码的LLM进行了广泛的实验覆盖，我们发现不同的代理在游戏中表现出不同的和有趣的行为。实验结果证明了该平台在评估基于LLM的多代理系统中的有效性和高效性。我们的平台及其文档可在\url{https://whoisspy.ai/}



## **26. Time-Reversal Provides Unsupervised Feedback to LLMs**

计时器向LLM提供无监督反馈 cs.CL

Accepted as a spotlight in NeurIPS 2024

**SubmitDate**: 2024-12-04    [abs](http://arxiv.org/abs/2412.02626v2) [paper-pdf](http://arxiv.org/pdf/2412.02626v2)

**Authors**: Yerram Varun, Rahul Madhavan, Sravanti Addepalli, Arun Suggala, Karthikeyan Shanmugam, Prateek Jain

**Abstract**: Large Language Models (LLMs) are typically trained to predict in the forward direction of time. However, recent works have shown that prompting these models to look back and critique their own generations can produce useful feedback. Motivated by this, we explore the question of whether LLMs can be empowered to think (predict and score) backwards to provide unsupervised feedback that complements forward LLMs. Towards this, we introduce Time Reversed Language Models (TRLMs), which can score and generate queries when conditioned on responses, effectively functioning in the reverse direction of time. Further, to effectively infer in the response to query direction, we pre-train and fine-tune a language model (TRLM-Ba) in the reverse token order from scratch. We show empirically (and theoretically in a stylized setting) that time-reversed models can indeed complement forward model predictions when used to score the query given response for re-ranking multiple forward generations. We obtain up to 5\% improvement on the widely used AlpacaEval Leaderboard over the competent baseline of best-of-N re-ranking using self log-perplexity scores. We further show that TRLM scoring outperforms conventional forward scoring of response given query, resulting in significant gains in applications such as citation generation and passage retrieval. We next leverage the generative ability of TRLM to augment or provide unsupervised feedback to input safety filters of LLMs, demonstrating a drastic reduction in false negative rate with negligible impact on false positive rates against several attacks published on the popular JailbreakBench leaderboard.

摘要: 大型语言模型(LLM)通常被训练为在时间的正向进行预测。然而，最近的研究表明，促使这些模型回顾和批评他们自己的几代人可以产生有用的反馈。受此启发，我们探讨了LLM是否可以被赋予向后思考(预测和评分)的能力，以提供无监督的反馈来补充前向LLM。为此，我们引入了时间反转语言模型(TRLMS)，该模型可以根据响应进行评分并生成查询，有效地沿时间的相反方向运行。此外，为了有效地推断对查询方向的响应，我们从头开始以相反的令牌顺序预先训练和微调语言模型(TRLM-BA)。我们在经验上(理论上是在风格化的环境中)表明，当时间倒置模型用于对给定响应的查询进行重新排序时，时间倒置模型确实可以补充正向模型预测。我们在广泛使用的AlpacaEval排行榜上获得了高达5%的改进，超过了使用自我对数困惑分数重新排序的合格基线。我们进一步表明，TRLM评分优于传统的对给定查询的回复的前向评分，从而在引文生成和段落检索等应用中获得了显著的收益。接下来，我们利用TRLM的生成能力来增强或向LLMS的输入安全过滤器提供无监督反馈，展示了假阴性率的大幅降低，而对流行的JailBreak Btch排行榜上发布的几种攻击的错误确认率的影响可以忽略不计。



## **27. Does Safety Training of LLMs Generalize to Semantically Related Natural Prompts?**

LLM的安全培训是否适用于语义相关的自然知识？ cs.CL

Accepted at the Safe Generative AI Workshop @ NeurIPS 2024

**SubmitDate**: 2024-12-04    [abs](http://arxiv.org/abs/2412.03235v1) [paper-pdf](http://arxiv.org/pdf/2412.03235v1)

**Authors**: Sravanti Addepalli, Yerram Varun, Arun Suggala, Karthikeyan Shanmugam, Prateek Jain

**Abstract**: Large Language Models (LLMs) are known to be susceptible to crafted adversarial attacks or jailbreaks that lead to the generation of objectionable content despite being aligned to human preferences using safety fine-tuning methods. While the large dimensionality of input token space makes it inevitable to find adversarial prompts that can jailbreak these models, we aim to evaluate whether safety fine-tuned LLMs are safe against natural prompts which are semantically related to toxic seed prompts that elicit safe responses after alignment. We surprisingly find that popular aligned LLMs such as GPT-4 can be compromised using naive prompts that are NOT even crafted with an objective of jailbreaking the model. Furthermore, we empirically show that given a seed prompt that elicits a toxic response from an unaligned model, one can systematically generate several semantically related natural prompts that can jailbreak aligned LLMs. Towards this, we propose a method of Response Guided Question Augmentation (ReG-QA) to evaluate the generalization of safety aligned LLMs to natural prompts, that first generates several toxic answers given a seed question using an unaligned LLM (Q to A), and further leverages an LLM to generate questions that are likely to produce these answers (A to Q). We interestingly find that safety fine-tuned LLMs such as GPT-4o are vulnerable to producing natural jailbreak questions from unsafe content (without denial) and can thus be used for the latter (A to Q) step. We obtain attack success rates that are comparable to/ better than leading adversarial attack methods on the JailbreakBench leaderboard, while being significantly more stable against defenses such as Smooth-LLM and Synonym Substitution, which are effective against existing all attacks on the leaderboard.

摘要: 众所周知，大型语言模型(LLM)容易受到精心设计的对抗性攻击或越狱，尽管使用安全微调方法与人类的偏好保持一致，但这些攻击或越狱会导致生成令人反感的内容。虽然输入令牌空间的大维度使得找到能够越狱这些模型的敌意提示是不可避免的，但我们的目标是评估安全的微调LLM对于自然提示是否安全，这些自然提示在语义上与有毒种子提示相关，在对齐后引起安全响应。我们惊讶地发现，GPT-4等流行的对齐LLM可以使用甚至不是以越狱为目标而精心设计的幼稚提示来进行攻击。此外，我们的经验表明，给定一个种子提示引起来自未对齐模型的有毒反应，一个人可以系统地生成几个语义相关的自然提示，从而可以越狱对齐的LLM。为此，我们提出了一种反应引导问题增强方法(REG-QA)来评估安全对齐LLM对自然提示的泛化，该方法首先使用未对齐LLM(Q到A)来生成给定种子问题的几个有毒答案，然后利用LLM来生成可能产生这些答案(A到Q)的问题。有趣的是，我们发现安全微调的LLM，如GPT-40，容易从不安全的内容产生自然的越狱问题(不否认)，因此可以用于后一步(A到Q)。我们获得了相当于/好于JailBreak排行榜上领先的对抗性攻击方法的攻击成功率，同时对Smooth-LLM和同义词替换等防御措施明显更加稳定，这些防御措施对排行榜上现有的所有攻击都有效。



## **28. "Moralized" Multi-Step Jailbreak Prompts: Black-Box Testing of Guardrails in Large Language Models for Verbal Attacks**

“道德化”多步骤越狱预言：对大型语言模型中护栏进行黑匣子测试以进行言语攻击 cs.CR

This paper has been submitted to Nature Machine Intelligence and  OpenReview preprints. It has 7 pages of text, 3 figures, and 3 tables

**SubmitDate**: 2024-12-04    [abs](http://arxiv.org/abs/2411.16730v3) [paper-pdf](http://arxiv.org/pdf/2411.16730v3)

**Authors**: Libo Wang

**Abstract**: As the application of large language models continues to expand in various fields, it poses higher challenges to the effectiveness of identifying harmful content generation and guardrail mechanisms. This research aims to evaluate the guardrail effectiveness of GPT-4o, Grok-2 Beta, Llama 3.1 (405B), Gemini 1.5, and Claude 3.5 Sonnet through black-box testing of seemingly ethical multi-step jailbreak prompts. It conducts ethical attacks by designing an identical multi-step prompts that simulates the scenario of "corporate middle managers competing for promotions." The data results show that the guardrails of the above-mentioned LLMs were bypassed and the content of verbal attacks was generated. Claude 3.5 Sonnet's resistance to multi-step jailbreak prompts is more obvious. To ensure objectivity, the experimental process, black box test code, and enhanced guardrail code are uploaded to the GitHub repository: https://github.com/brucewang123456789/GeniusTrail.git.

摘要: 随着大型语言模型在各个领域的应用不断扩展，对识别有害内容生成和护栏机制的有效性提出了更高的挑战。这项研究旨在通过对看似合乎道德的多步越狱提示进行黑匣子测试来评估GPT-4 o、Grok-2 Beta、Llama 3.1（405 B）、Gemini 1.5和Claude 3.5十四行诗的护栏有效性。它通过设计相同的多步骤提示来进行道德攻击，模拟“企业中层管理人员竞争晋升”的场景。“数据结果显示，上述LLM的护栏被绕过，产生了言语攻击的内容。克劳德3.5十四行诗对多步越狱提示的抵制更加明显。为了确保客观性，实验过程、黑匣子测试代码和增强型护栏代码被上传到GitHub存储库：https://github.com/brucewang123456789/GeniusTrail.git。



## **29. Backdoor Attacks and Countermeasures in Natural Language Processing Models: A Comprehensive Security Review**

自然语言处理模型中的后门攻击和对策：全面的安全评论 cs.CR

21 pages, 3 figures

**SubmitDate**: 2024-12-04    [abs](http://arxiv.org/abs/2309.06055v5) [paper-pdf](http://arxiv.org/pdf/2309.06055v5)

**Authors**: Pengzhou Cheng, Zongru Wu, Wei Du, Haodong Zhao, Wei Lu, Gongshen Liu

**Abstract**: Language Models (LMs) are becoming increasingly popular in real-world applications. Outsourcing model training and data hosting to third-party platforms has become a standard method for reducing costs. In such a situation, the attacker can manipulate the training process or data to inject a backdoor into models. Backdoor attacks are a serious threat where malicious behavior is activated when triggers are present, otherwise, the model operates normally.   However, there is still no systematic and comprehensive review of LMs from the attacker's capabilities and purposes on different backdoor attack surfaces. Moreover, there is a shortage of analysis and comparison of the diverse emerging backdoor countermeasures. Therefore, this work aims to provide the NLP community with a timely review of backdoor attacks and countermeasures. According to the attackers' capability and affected stage of the LMs, the attack surfaces are formalized into four categorizations: attacking the pre-trained model with fine-tuning (APMF) or parameter-efficient fine-tuning (APMP), attacking the final model with training (AFMT), and attacking Large Language Models (ALLM). Thus, attacks under each categorization are combed. The countermeasures are categorized into two general classes: sample inspection and model inspection. Thus, we review countermeasures and analyze their advantages and disadvantages. Also, we summarize the benchmark datasets and provide comparable evaluations for representative attacks and defenses. Drawing the insights from the review, we point out the crucial areas for future research on the backdoor, especially soliciting more efficient and practical countermeasures.

摘要: 语言模型(LMS)在实际应用中正变得越来越流行。将模型培训和数据托管外包给第三方平台已成为降低成本的标准方法。在这种情况下，攻击者可以操纵训练过程或数据以向模型注入后门。后门攻击是一种严重的威胁，当存在触发器时，恶意行为被激活，否则，模型正常运行。然而，目前还没有从攻击者在不同的后门攻击面上的能力和目的对LMS进行系统和全面的审查。此外，对各种新出现的借壳对策缺乏分析和比较。因此，这项工作旨在为NLP社区提供及时审查后门攻击和对策的机会。根据攻击者的攻击能力和受影响阶段，将攻击面形式化为四类：精调攻击预训练模型(APMF)或参数高效微调攻击(APMP)、训练攻击最终模型(AFMT)和攻击大型语言模型(ALLM)。因此，对每个分类下的攻击进行了梳理。反制措施一般分为两大类：抽样检查和模型检查。因此，我们回顾了这些对策，并分析了它们的优缺点。此外，我们总结了基准数据集，并提供了具有代表性的攻击和防御的可比性评估。从回顾中得到的启示，我们指出了未来关于后门研究的关键领域，特别是寻求更有效和更实际的对策。



## **30. Unleashing GHOST: An LLM-Powered Framework for Automated Hardware Trojan Design**

释放GSTORE：一个由LLM支持的自动硬件特洛伊木马设计框架 cs.CR

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2412.02816v1) [paper-pdf](http://arxiv.org/pdf/2412.02816v1)

**Authors**: Md Omar Faruque, Peter Jamieson, Ahmad Patooghy, Abdel-Hameed A. Badawy

**Abstract**: Traditionally, inserting realistic Hardware Trojans (HTs) into complex hardware systems has been a time-consuming and manual process, requiring comprehensive knowledge of the design and navigating intricate Hardware Description Language (HDL) codebases. Machine Learning (ML)-based approaches have attempted to automate this process but often face challenges such as the need for extensive training data, long learning times, and limited generalizability across diverse hardware design landscapes. This paper addresses these challenges by proposing GHOST (Generator for Hardware-Oriented Stealthy Trojans), an automated attack framework that leverages Large Language Models (LLMs) for rapid HT generation and insertion. Our study evaluates three state-of-the-art LLMs - GPT-4, Gemini-1.5-pro, and Llama-3-70B - across three hardware designs: SRAM, AES, and UART. According to our evaluations, GPT-4 demonstrates superior performance, with 88.88% of HT insertion attempts successfully generating functional and synthesizable HTs. This study also highlights the security risks posed by LLM-generated HTs, showing that 100% of GHOST-generated synthesizable HTs evaded detection by an ML-based HT detection tool. These results underscore the urgent need for advanced detection and prevention mechanisms in hardware security to address the emerging threat of LLM-generated HTs. The GHOST HT benchmarks are available at: https://github.com/HSTRG1/GHOSTbenchmarks.git

摘要: 传统上，在复杂的硬件系统中插入真实硬件特洛伊木马(HTS)一直是一个耗时且手动的过程，需要全面的设计知识和导航复杂的硬件描述语言(HDL)代码库。基于机器学习(ML)的方法试图使这一过程自动化，但经常面临挑战，例如需要大量的训练数据、学习时间长以及在不同硬件设计环境中的推广有限。本文通过提出Ghost(面向硬件的隐身木马生成器)来应对这些挑战，Ghost是一个自动化攻击框架，它利用大型语言模型(LLM)来快速生成和插入HT。我们的研究评估了三种最先进的LLM-GPT-4、Gemini-1.5-PRO和Llama-3-70B-跨越三种硬件设计：SRAM、AES和UART。根据我们的评估，GPT-4表现出优越的性能，88.88%的HT插入尝试成功地生成了功能性和可合成的HTS。这项研究还强调了LLM生成的HTS带来的安全风险，表明100%的幽灵生成的可合成HTS可以躲避基于ML的HTS检测工具的检测。这些结果突显了在硬件安全方面迫切需要先进的检测和预防机制，以应对LLM生成的HTS的新威胁。Ghost HT基准可在以下网站获得：https://github.com/HSTRG1/GHOSTbenchmarks.git



## **31. Gracefully Filtering Backdoor Samples for Generative Large Language Models without Retraining**

优雅地过滤生成性大型语言模型的后门样本，无需重新训练 cs.CL

Accepted at COLING 2025

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2412.02454v1) [paper-pdf](http://arxiv.org/pdf/2412.02454v1)

**Authors**: Zongru Wu, Pengzhou Cheng, Lingyong Fang, Zhuosheng Zhang, Gongshen Liu

**Abstract**: Backdoor attacks remain significant security threats to generative large language models (LLMs). Since generative LLMs output sequences of high-dimensional token logits instead of low-dimensional classification logits, most existing backdoor defense methods designed for discriminative models like BERT are ineffective for generative LLMs. Inspired by the observed differences in learning behavior between backdoor and clean mapping in the frequency space, we transform gradients of each training sample, directly influencing parameter updates, into the frequency space. Our findings reveal a distinct separation between the gradients of backdoor and clean samples in the frequency space. Based on this phenomenon, we propose Gradient Clustering in the Frequency Space for Backdoor Sample Filtering (GraCeFul), which leverages sample-wise gradients in the frequency space to effectively identify backdoor samples without requiring retraining LLMs. Experimental results show that GraCeFul outperforms baselines significantly. Notably, GraCeFul exhibits remarkable computational efficiency, achieving nearly 100% recall and F1 scores in identifying backdoor samples, reducing the average success rate of various backdoor attacks to 0% with negligible drops in clean accuracy across multiple free-style question answering datasets. Additionally, GraCeFul generalizes to Llama-2 and Vicuna. The codes are publicly available at https://github.com/ZrW00/GraceFul.

摘要: 后门攻击仍然是生成性大型语言模型(LLM)的重大安全威胁。由于生成性LLMS输出的是高维令牌逻辑序列，而不是低维分类逻辑序列，现有的大多数后门防御方法都是针对BERT等区分模型设计的，对于生成性LLMS是无效的。受观察到的频率空间中后门映射和干净映射在学习行为上的差异的启发，我们将每个训练样本的梯度转换到频率空间中，这直接影响参数的更新。我们的发现表明，在频率空间中，后门样本和清洁样本的梯度之间存在明显的分离。基于这一现象，我们提出了在频率空间中进行后门样本滤波的梯度聚类(GRACEFUE)，它利用频率空间中的样本梯度来有效地识别后门样本，而不需要重新训练LLMS。实验结果表明，优雅算法的性能明显优于基线算法。值得注意的是，Graceful表现出了卓越的计算效率，在识别后门样本方面实现了近100%的Recall和F1分数，将各种后门攻击的平均成功率降低到0%，而跨多个自由风格问答数据集的干净准确率几乎可以忽略不计。此外，优雅适用于骆驼-2和维库纳。这些代码可在https://github.com/ZrW00/GraceFul.上公开获得



## **32. Harmful Fine-tuning Attacks and Defenses for Large Language Models: A Survey**

针对大型语言模型的有害微调攻击和防御：调查 cs.CR

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2409.18169v5) [paper-pdf](http://arxiv.org/pdf/2409.18169v5)

**Authors**: Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, Ling Liu

**Abstract**: Recent research demonstrates that the nascent fine-tuning-as-a-service business model exposes serious safety concerns -- fine-tuning over a few harmful data uploaded by the users can compromise the safety alignment of the model. The attack, known as harmful fine-tuning attack, has raised a broad research interest among the community. However, as the attack is still new, \textbf{we observe that there are general misunderstandings within the research community.} To clear up concern, this paper provide a comprehensive overview to three aspects of harmful fine-tuning: attacks setting, defense design and evaluation methodology. Specifically, we first present the threat model of the problem, and introduce the harmful fine-tuning attack and its variants. Then we systematically survey the existing literature on attacks/defenses/mechanical analysis of the problem. Finally, we introduce the evaluation methodology and outline future research directions that might contribute to the development of the field. Additionally, we present a list of questions of interest, which might be useful to refer to when reviewers in the peer review process question the realism of the experiment/attack/defense setting. A curated list of relevant papers is maintained and made accessible at: https://github.com/git-disl/awesome_LLM-harmful-fine-tuning-papers.

摘要: 最近的研究表明，新兴的微调即服务商业模式暴露了严重的安全问题--对用户上传的几个有害数据进行微调可能会损害该模型的安全一致性。这一被称为有害微调攻击的攻击在社区中引起了广泛的研究兴趣。然而，由于攻击仍然是新的，我们观察到研究界存在普遍的误解。}为了消除人们的担忧，本文对有害微调的三个方面进行了全面的概述：攻击设置、防御设计和评估方法。具体地说，我们首先给出了问题的威胁模型，并介绍了有害的微调攻击及其变体。然后，我们系统地综述了现有的关于攻击/防御/机械分析问题的文献。最后，我们介绍了评估方法，并概述了未来可能有助于该领域发展的研究方向。此外，我们提供了一个感兴趣的问题列表，当同行审查过程中的评审者质疑实验/攻击/防御设置的真实性时，这些问题可能会有用。相关论文的精选清单可在以下网址查阅：https://github.com/git-disl/awesome_LLM-harmful-fine-tuning-papers.



## **33. Trust & Safety of LLMs and LLMs in Trust & Safety**

LLM的信任与安全以及LLM的信任与安全 cs.AI

11 pages

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2412.02113v1) [paper-pdf](http://arxiv.org/pdf/2412.02113v1)

**Authors**: Doohee You, Dan Chon

**Abstract**: In recent years, Large Language Models (LLMs) have garnered considerable attention for their remarkable abilities in natural language processing tasks. However, their widespread adoption has raised concerns pertaining to trust and safety. This systematic review investigates the current research landscape on trust and safety in LLMs, with a particular focus on the novel application of LLMs within the field of Trust and Safety itself. We delve into the complexities of utilizing LLMs in domains where maintaining trust and safety is paramount, offering a consolidated perspective on this emerging trend.\   By synthesizing findings from various studies, we identify key challenges and potential solutions, aiming to benefit researchers and practitioners seeking to understand the nuanced interplay between LLMs and Trust and Safety.   This review provides insights on best practices for using LLMs in Trust and Safety, and explores emerging risks such as prompt injection and jailbreak attacks. Ultimately, this study contributes to a deeper understanding of how LLMs can be effectively and responsibly utilized to enhance trust and safety in the digital realm.

摘要: 近年来，大语言模型因其在自然语言处理任务中的卓越能力而受到广泛关注。然而，它们的广泛采用引发了人们对信任和安全的担忧。这篇系统的综述调查了当前关于低成本管理中信任和安全的研究现状，特别关注低成本管理在信任和安全领域的新应用。我们深入研究了在维护信任和安全至高无上的领域使用低成本管理的复杂性，为这一新兴趋势提供了一个综合的视角。\通过综合各种研究的结果，我们确定了关键的挑战和潜在的解决方案，旨在帮助寻求了解低成本管理与信任和安全之间微妙相互作用的研究人员和从业者。这篇综述提供了关于在信任与安全中使用LLMS的最佳实践的见解，并探索了新出现的风险，如快速注入和越狱攻击。最终，这项研究有助于更深入地理解如何有效和负责任地利用LLM来增强数字领域的信任和安全。



## **34. Towards Understanding Jailbreak Attacks in LLMs: A Representation Space Analysis**

了解LLC中的越狱攻击：表示空间分析 cs.CL

Accepted by EMNLP 2024 Main

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2406.10794v3) [paper-pdf](http://arxiv.org/pdf/2406.10794v3)

**Authors**: Yuping Lin, Pengfei He, Han Xu, Yue Xing, Makoto Yamada, Hui Liu, Jiliang Tang

**Abstract**: Large language models (LLMs) are susceptible to a type of attack known as jailbreaking, which misleads LLMs to output harmful contents. Although there are diverse jailbreak attack strategies, there is no unified understanding on why some methods succeed and others fail. This paper explores the behavior of harmful and harmless prompts in the LLM's representation space to investigate the intrinsic properties of successful jailbreak attacks. We hypothesize that successful attacks share some similar properties: They are effective in moving the representation of the harmful prompt towards the direction to the harmless prompts. We leverage hidden representations into the objective of existing jailbreak attacks to move the attacks along the acceptance direction, and conduct experiments to validate the above hypothesis using the proposed objective. We hope this study provides new insights into understanding how LLMs understand harmfulness information.

摘要: 大型语言模型（LLM）容易受到一种称为越狱的攻击，这种攻击会误导LLM输出有害内容。尽管越狱攻击策略多种多样，但对于为什么有些方法成功而另一些方法失败，人们并没有统一的理解。本文探讨了LLM表示空间中有害和无害提示的行为，以研究成功越狱攻击的内在属性。我们假设成功的攻击具有一些相似的属性：它们有效地将有害提示的表示移向无害提示的方向。我们将隐藏的表示利用到现有越狱攻击的目标中，以沿着接受方向移动攻击，并使用提出的目标进行实验来验证上述假设。我们希望这项研究为理解LLM如何理解有害信息提供新的见解。



## **35. Improved Large Language Model Jailbreak Detection via Pretrained Embeddings**

通过预训练嵌入改进的大语言模型越狱检测 cs.CR

Submitted to AICS 2025: https://aics.site

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2412.01547v1) [paper-pdf](http://arxiv.org/pdf/2412.01547v1)

**Authors**: Erick Galinkin, Martin Sablotny

**Abstract**: The adoption of large language models (LLMs) in many applications, from customer service chat bots and software development assistants to more capable agentic systems necessitates research into how to secure these systems. Attacks like prompt injection and jailbreaking attempt to elicit responses and actions from these models that are not compliant with the safety, privacy, or content policies of organizations using the model in their application. In order to counter abuse of LLMs for generating potentially harmful replies or taking undesirable actions, LLM owners must apply safeguards during training and integrate additional tools to block the LLM from generating text that abuses the model. Jailbreaking prompts play a vital role in convincing an LLM to generate potentially harmful content, making it important to identify jailbreaking attempts to block any further steps. In this work, we propose a novel approach to detect jailbreak prompts based on pairing text embeddings well-suited for retrieval with traditional machine learning classification algorithms. Our approach outperforms all publicly available methods from open source LLM security applications.

摘要: 从客户服务聊天机器人和软件开发助理到更有能力的代理系统，在许多应用程序中采用大型语言模型(LLM)，需要研究如何保护这些系统。诸如提示注入和越狱之类的攻击试图从这些模型引发响应和操作，这些响应和操作不符合在其应用程序中使用该模型的组织的安全、隐私或内容策略。为了防止LLMS被滥用来生成可能有害的回复或采取不受欢迎的行动，LLM所有者必须在培训期间应用安全措施，并集成其他工具来阻止LLM生成滥用该模型的文本。越狱提示在说服LLM生成潜在有害内容方面发挥着至关重要的作用，因此识别阻止任何进一步步骤的越狱尝试非常重要。在这项工作中，我们提出了一种新的基于文本嵌入的越狱提示检测方法，该方法适合于传统机器学习分类算法的检索。我们的方法比开源LLM安全应用程序中所有公开可用的方法都要好。



## **36. LUMIA: Linear probing for Unimodal and MultiModal Membership Inference Attacks leveraging internal LLM states**

LUMIA：利用内部LLM状态进行单模式和多模式成员资格推理攻击的线性探测 cs.CR

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2411.19876v2) [paper-pdf](http://arxiv.org/pdf/2411.19876v2)

**Authors**: Luis Ibanez-Lissen, Lorena Gonzalez-Manzano, Jose Maria de Fuentes, Nicolas Anciaux, Joaquin Garcia-Alfaro

**Abstract**: Large Language Models (LLMs) are increasingly used in a variety of applications, but concerns around membership inference have grown in parallel. Previous efforts focus on black-to-grey-box models, thus neglecting the potential benefit from internal LLM information. To address this, we propose the use of Linear Probes (LPs) as a method to detect Membership Inference Attacks (MIAs) by examining internal activations of LLMs. Our approach, dubbed LUMIA, applies LPs layer-by-layer to get fine-grained data on the model inner workings. We test this method across several model architectures, sizes and datasets, including unimodal and multimodal tasks. In unimodal MIA, LUMIA achieves an average gain of 15.71 % in Area Under the Curve (AUC) over previous techniques. Remarkably, LUMIA reaches AUC>60% in 65.33% of cases -- an increment of 46.80% against the state of the art. Furthermore, our approach reveals key insights, such as the model layers where MIAs are most detectable. In multimodal models, LPs indicate that visual inputs can significantly contribute to detect MIAs -- AUC>60% is reached in 85.90% of experiments.

摘要: 大型语言模型(LLM)越来越多地用于各种应用程序，但围绕成员关系推理的关注也在平行增长。以往的研究主要集中在黑灰盒模型上，从而忽略了LLM内部信息的潜在益处。为了解决这一问题，我们提出使用线性探测器(LP)作为一种方法，通过检查LLP的内部激活来检测成员身份推理攻击(MIA)。我们的方法，称为Lumia，逐层应用LP，以获得关于模型内部工作的细粒度数据。我们在几个模型体系结构、大小和数据集上测试了这种方法，包括单模和多模任务。在单峰MIA中，Lumia的曲线下面积(AUC)比以前的技术平均增加了15.71%。值得注意的是，Lumia在65.33%的情况下达到AUC>60%--与最先进的水平相比增加了46.80%。此外，我们的方法揭示了关键的见解，例如最容易检测到MIA的模型层。在多通道模型中，LP表明视觉输入对检测MIA有显著贡献-85.90%的实验达到了60%以上的AUC。



## **37. Recent Advances in Attack and Defense Approaches of Large Language Models**

大型语言模型攻击和防御方法的最新进展 cs.CR

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2409.03274v3) [paper-pdf](http://arxiv.org/pdf/2409.03274v3)

**Authors**: Jing Cui, Yishi Xu, Zhewei Huang, Shuchang Zhou, Jianbin Jiao, Junge Zhang

**Abstract**: Large Language Models (LLMs) have revolutionized artificial intelligence and machine learning through their advanced text processing and generating capabilities. However, their widespread deployment has raised significant safety and reliability concerns. Established vulnerabilities in deep neural networks, coupled with emerging threat models, may compromise security evaluations and create a false sense of security. Given the extensive research in the field of LLM security, we believe that summarizing the current state of affairs will help the research community better understand the present landscape and inform future developments. This paper reviews current research on LLM vulnerabilities and threats, and evaluates the effectiveness of contemporary defense mechanisms. We analyze recent studies on attack vectors and model weaknesses, providing insights into attack mechanisms and the evolving threat landscape. We also examine current defense strategies, highlighting their strengths and limitations. By contrasting advancements in attack and defense methodologies, we identify research gaps and propose future directions to enhance LLM security. Our goal is to advance the understanding of LLM safety challenges and guide the development of more robust security measures.

摘要: 大型语言模型(LLM)通过其先进的文本处理和生成能力，使人工智能和机器学习发生了革命性的变化。然而，它们的广泛部署引发了严重的安全和可靠性问题。深层神经网络中已建立的漏洞，再加上新出现的威胁模型，可能会危及安全评估，并造成一种错误的安全感。鉴于LLM安全领域的广泛研究，我们相信总结当前的事态将有助于研究界更好地了解目前的情况并为未来的发展提供信息。本文回顾了LLM漏洞和威胁的研究现状，并对现代防御机制的有效性进行了评估。我们分析了最近关于攻击载体和模型弱点的研究，提供了对攻击机制和不断演变的威胁环境的洞察。我们还研究了当前的防御战略，强调了它们的优势和局限性。通过对比攻击和防御方法的进展，我们发现了研究的差距，并提出了增强LLM安全的未来方向。我们的目标是促进对LLM安全挑战的理解，并指导开发更强大的安全措施。



## **38. BDefects4NN: A Backdoor Defect Database for Controlled Localization Studies in Neural Networks**

BDefects 4NN：用于神经网络受控定位研究的后门缺陷数据库 cs.SE

11 pages, accepted by ICSE 2025

**SubmitDate**: 2024-12-01    [abs](http://arxiv.org/abs/2412.00746v1) [paper-pdf](http://arxiv.org/pdf/2412.00746v1)

**Authors**: Yisong Xiao, Aishan Liu, Xinwei Zhang, Tianyuan Zhang, Tianlin Li, Siyuan Liang, Xianglong Liu, Yang Liu, Dacheng Tao

**Abstract**: Pre-trained large deep learning models are now serving as the dominant component for downstream middleware users and have revolutionized the learning paradigm, replacing the traditional approach of training from scratch locally. To reduce development costs, developers often integrate third-party pre-trained deep neural networks (DNNs) into their intelligent software systems. However, utilizing untrusted DNNs presents significant security risks, as these models may contain intentional backdoor defects resulting from the black-box training process. These backdoor defects can be activated by hidden triggers, allowing attackers to maliciously control the model and compromise the overall reliability of the intelligent software. To ensure the safe adoption of DNNs in critical software systems, it is crucial to establish a backdoor defect database for localization studies. This paper addresses this research gap by introducing BDefects4NN, the first backdoor defect database, which provides labeled backdoor-defected DNNs at the neuron granularity and enables controlled localization studies of defect root causes. In BDefects4NN, we define three defect injection rules and employ four representative backdoor attacks across four popular network architectures and three widely adopted datasets, yielding a comprehensive database of 1,654 backdoor-defected DNNs with four defect quantities and varying infected neurons. Based on BDefects4NN, we conduct extensive experiments on evaluating six fault localization criteria and two defect repair techniques, which show limited effectiveness for backdoor defects. Additionally, we investigate backdoor-defected models in practical scenarios, specifically in lane detection for autonomous driving and large language models (LLMs), revealing potential threats and highlighting current limitations in precise defect localization.

摘要: 预先训练的大型深度学习模型现在成为下游中间件用户的主导组件，并彻底改变了学习范式，取代了传统的从局部从头开始培训的方法。为了降低开发成本，开发人员经常将第三方预先训练的深度神经网络(DNN)集成到他们的智能软件系统中。然而，使用不受信任的DNN会带来重大的安全风险，因为这些模型可能包含由黑盒培训过程导致的故意后门缺陷。这些后门缺陷可以被隐藏的触发器激活，允许攻击者恶意控制模型，并损害智能软件的整体可靠性。为了确保在关键软件系统中安全地采用DNN，建立用于本地化研究的后门缺陷数据库是至关重要的。本文通过引入第一个后门缺陷数据库BDefects4NN来弥补这一研究空白，该数据库在神经元粒度上提供标记的后门缺陷DNN，并使对缺陷根本原因的受控定位研究成为可能。基于BDefects4NN，我们对六个故障定位准则和两个缺陷修复技术进行了广泛的实验，结果表明它们对后门缺陷的效果有限。此外，我们还研究了实际场景中的后门缺陷模型，特别是在自动驾驶和大型语言模型(LLM)的车道检测中，揭示了潜在的威胁，并强调了当前在精确缺陷定位方面的限制。



## **39. Evaluating Large Language Models' Capability to Launch Fully Automated Spear Phishing Campaigns: Validated on Human Subjects**

评估大型语言模型发起全自动鱼叉式网络钓鱼活动的能力：在人类受试者上进行验证 cs.CR

**SubmitDate**: 2024-11-30    [abs](http://arxiv.org/abs/2412.00586v1) [paper-pdf](http://arxiv.org/pdf/2412.00586v1)

**Authors**: Fred Heiding, Simon Lermen, Andrew Kao, Bruce Schneier, Arun Vishwanath

**Abstract**: In this paper, we evaluate the capability of large language models to conduct personalized phishing attacks and compare their performance with human experts and AI models from last year. We include four email groups with a combined total of 101 participants: A control group of arbitrary phishing emails, which received a click-through rate (recipient pressed a link in the email) of 12%, emails generated by human experts (54% click-through), fully AI-automated emails 54% (click-through), and AI emails utilizing a human-in-the-loop (56% click-through). Thus, the AI-automated attacks performed on par with human experts and 350% better than the control group. The results are a significant improvement from similar studies conducted last year, highlighting the increased deceptive capabilities of AI models. Our AI-automated emails were sent using a custom-built tool that automates the entire spear phishing process, including information gathering and creating personalized vulnerability profiles for each target. The AI-gathered information was accurate and useful in 88% of cases and only produced inaccurate profiles for 4% of the participants. We also use language models to detect the intention of emails. Claude 3.5 Sonnet scored well above 90% with low false-positive rates and detected several seemingly benign emails that passed human detection. Lastly, we analyze the economics of phishing, highlighting how AI enables attackers to target more individuals at lower cost and increase profitability by up to 50 times for larger audiences.

摘要: 在本文中，我们评估了大型语言模型进行个性化钓鱼攻击的能力，并将其性能与去年的人类专家和AI模型进行了比较。我们包括四个电子邮件组，总共有101名参与者：控制组的任意钓鱼电子邮件的点击率(收件人按下电子邮件中的链接)为12%，由人类专家生成的电子邮件(点击率为54%)，完全人工智能自动化的电子邮件(点击率为54%)，以及利用人在循环中的人工智能电子邮件(56%的点击率)。因此，人工智能自动攻击的表现与人类专家不相上下，比对照组好350%。与去年进行的类似研究相比，这一结果是一个显著的进步，突显了人工智能模型更强的欺骗性。我们的人工智能自动电子邮件是使用定制的工具发送的，该工具可以自动执行整个鱼叉式网络钓鱼过程，包括收集信息并为每个目标创建个性化的漏洞配置文件。人工智能收集的信息在88%的情况下是准确和有用的，只有4%的参与者产生了不准确的个人资料。我们还使用语言模型来检测电子邮件的意图。克劳德3.5十四行诗得分远高于90%，假阳性率很低，并检测到几封看似温和的电子邮件通过了人类的检测。最后，我们分析了钓鱼的经济学，强调了人工智能如何使攻击者能够以更低的成本瞄准更多的个人，并将更多受众的盈利能力提高高达50倍。



## **40. Uncovering Safety Risks of Large Language Models through Concept Activation Vector**

通过概念激活载体揭示大型语言模型的安全风险 cs.CL

10 pages, accepted at NeurIPS 2024

**SubmitDate**: 2024-11-30    [abs](http://arxiv.org/abs/2404.12038v5) [paper-pdf](http://arxiv.org/pdf/2404.12038v5)

**Authors**: Zhihao Xu, Ruixuan Huang, Changyu Chen, Xiting Wang

**Abstract**: Despite careful safety alignment, current large language models (LLMs) remain vulnerable to various attacks. To further unveil the safety risks of LLMs, we introduce a Safety Concept Activation Vector (SCAV) framework, which effectively guides the attacks by accurately interpreting LLMs' safety mechanisms. We then develop an SCAV-guided attack method that can generate both attack prompts and embedding-level attacks with automatically selected perturbation hyperparameters. Both automatic and human evaluations demonstrate that our attack method significantly improves the attack success rate and response quality while requiring less training data. Additionally, we find that our generated attack prompts may be transferable to GPT-4, and the embedding-level attacks may also be transferred to other white-box LLMs whose parameters are known. Our experiments further uncover the safety risks present in current LLMs. For example, in our evaluation of seven open-source LLMs, we observe an average attack success rate of 99.14%, based on the classic keyword-matching criterion. Finally, we provide insights into the safety mechanism of LLMs. The code is available at https://github.com/SproutNan/AI-Safety_SCAV.

摘要: 尽管进行了仔细的安全调整，但当前的大型语言模型(LLM)仍然容易受到各种攻击。为了进一步揭示LLMS的安全隐患，我们引入了安全概念激活向量(SCAV)框架，通过准确解释LLMS的安全机制来有效地指导攻击。然后，我们开发了一种SCAV引导的攻击方法，该方法可以生成攻击提示和带有自动选择的扰动超参数的嵌入级攻击。自动和人工评估都表明，我们的攻击方法在需要更少的训练数据的情况下，显著地提高了攻击成功率和响应质量。此外，我们发现我们生成的攻击提示可以转移到GPT-4上，嵌入级攻击也可以转移到参数已知的其他白盒LLM上。我们的实验进一步揭示了当前LLM中存在的安全风险。例如，在我们对7个开源LLM的评估中，基于经典的关键字匹配标准，我们观察到平均攻击成功率为99.14%。最后，我们对LLMS的安全机制提供了见解。代码可在https://github.com/SproutNan/AI-Safety_SCAV.上获得



## **41. Safety Alignment Backfires: Preventing the Re-emergence of Suppressed Concepts in Fine-tuned Text-to-Image Diffusion Models**

安全调整适得其反：防止被抑制的概念在微调的文本到图像扩散模型中重新出现 cs.AI

20 pages, 18 figures

**SubmitDate**: 2024-11-30    [abs](http://arxiv.org/abs/2412.00357v1) [paper-pdf](http://arxiv.org/pdf/2412.00357v1)

**Authors**: Sanghyun Kim, Moonseok Choi, Jinwoo Shin, Juho Lee

**Abstract**: Fine-tuning text-to-image diffusion models is widely used for personalization and adaptation for new domains. In this paper, we identify a critical vulnerability of fine-tuning: safety alignment methods designed to filter harmful content (e.g., nudity) can break down during fine-tuning, allowing previously suppressed content to resurface, even when using benign datasets. While this "fine-tuning jailbreaking" issue is known in large language models, it remains largely unexplored in text-to-image diffusion models. Our investigation reveals that standard fine-tuning can inadvertently undo safety measures, causing models to relearn harmful concepts that were previously removed and even exacerbate harmful behaviors. To address this issue, we present a novel but immediate solution called Modular LoRA, which involves training Safety Low-Rank Adaptation (LoRA) modules separately from Fine-Tuning LoRA components and merging them during inference. This method effectively prevents the re-learning of harmful content without compromising the model's performance on new tasks. Our experiments demonstrate that Modular LoRA outperforms traditional fine-tuning methods in maintaining safety alignment, offering a practical approach for enhancing the security of text-to-image diffusion models against potential attacks.

摘要: 微调的文本到图像扩散模型被广泛用于个性化和适应新领域。在本文中，我们确定了微调的一个关键漏洞：旨在过滤有害内容(例如裸露)的安全对齐方法在微调过程中可能会崩溃，从而允许先前被抑制的内容重新浮出水面，即使使用的是良性数据集。虽然这种“微调越狱”问题在大型语言模型中是已知的，但在文本到图像的扩散模型中，它在很大程度上仍未被探索。我们的调查显示，标准的微调可能会无意中取消安全措施，导致模型重新学习以前删除的有害概念，甚至加剧有害行为。为了解决这个问题，我们提出了一种新颖而直接的解决方案，称为模块化LORA，它包括从精调LORA组件中分离训练安全低阶自适应(LORA)模块，并在推理过程中将它们合并。这种方法有效地防止了有害内容的重新学习，而不会影响模型在新任务上的性能。我们的实验表明，模块化LORA在保持安全对齐方面优于传统的微调方法，为增强文本到图像扩散模型抵御潜在攻击的安全性提供了一种实用的方法。



## **42. When LLMs Go Online: The Emerging Threat of Web-Enabled LLMs**

当LLM上线时：支持Web的LLM的新兴威胁 cs.CR

**SubmitDate**: 2024-11-29    [abs](http://arxiv.org/abs/2410.14569v2) [paper-pdf](http://arxiv.org/pdf/2410.14569v2)

**Authors**: Hanna Kim, Minkyoo Song, Seung Ho Na, Seungwon Shin, Kimin Lee

**Abstract**: Recent advancements in Large Language Models (LLMs) have established them as agentic systems capable of planning and interacting with various tools. These LLM agents are often paired with web-based tools, enabling access to diverse sources and real-time information. Although these advancements offer significant benefits across various applications, they also increase the risk of malicious use, particularly in cyberattacks involving personal information. In this work, we investigate the risks associated with misuse of LLM agents in cyberattacks involving personal data. Specifically, we aim to understand: 1) how potent LLM agents can be when directed to conduct cyberattacks, 2) how cyberattacks are enhanced by web-based tools, and 3) how affordable and easy it becomes to launch cyberattacks using LLM agents. We examine three attack scenarios: the collection of Personally Identifiable Information (PII), the generation of impersonation posts, and the creation of spear-phishing emails. Our experiments reveal the effectiveness of LLM agents in these attacks: LLM agents achieved a precision of up to 95.9% in collecting PII, up to 93.9% of impersonation posts created by LLM agents were evaluated as authentic, and the click rate for links in spear phishing emails created by LLM agents reached up to 46.67%. Additionally, our findings underscore the limitations of existing safeguards in contemporary commercial LLMs, emphasizing the urgent need for more robust security measures to prevent the misuse of LLM agents.

摘要: 大型语言模型(LLM)的最新进展已将它们确立为能够规划各种工具并与其交互的代理系统。这些LLM代理通常与基于Web的工具配合使用，从而能够访问不同的来源和实时信息。虽然这些改进在各种应用程序中提供了显著的好处，但它们也增加了恶意使用的风险，特别是在涉及个人信息的网络攻击中。在这项工作中，我们调查了在涉及个人数据的网络攻击中滥用LLM代理的相关风险。具体地说，我们的目标是了解：1)LLM代理在被指示进行网络攻击时的威力有多大；2)基于Web的工具如何增强网络攻击；3)使用LLM代理发起网络攻击变得多么负担得起和容易。我们研究了三种攻击场景：收集个人身份信息(PII)、生成模拟帖子和创建鱼叉式网络钓鱼电子邮件。我们的实验显示了LLM代理在这些攻击中的有效性：LLM代理收集PII的准确率高达95.9%，LLM代理创建的模仿帖子被评估为可信的高达93.9%，LLM代理创建的鱼叉式钓鱼邮件中链接的点击率高达46.67%。此外，我们的研究结果强调了当代商业LLM现有保障措施的局限性，强调迫切需要采取更强有力的安全措施，以防止滥用LLM剂。



## **43. Ensemble Watermarks for Large Language Models**

大型语言模型的注册水印 cs.CL

9 pages in the main body. Code is available at  http://github.com/CommodoreEU/master-generation. arXiv admin note:  substantial text overlap with arXiv:2405.08400

**SubmitDate**: 2024-11-29    [abs](http://arxiv.org/abs/2411.19563v1) [paper-pdf](http://arxiv.org/pdf/2411.19563v1)

**Authors**: Georg Niess, Roman Kern

**Abstract**: The rapid advancement of large language models (LLMs) has made it increasingly difficult to distinguish between text written by humans and machines. While watermarks already exist for LLMs, they often lack flexibility, and struggle with attacks such as paraphrasing. To address these issues, we propose a multi-feature method for generating watermarks that combines multiple distinct watermark features into an ensemble watermark. Concretely, we combine acrostica and sensorimotor norms with the established red-green watermark to achieve a 98% detection rate. After a paraphrasing attack the performance remains high with 95% detection rate. The red-green feature alone as baseline achieves a detection rate of 49%. The evaluation of all feature combinations reveals that the ensemble of all three consistently has the highest detection rate across several LLMs and watermark strength settings. Due to the flexibility of combining features in the ensemble, various requirements and trade-offs can be addressed. Additionally, for all ensemble configurations the same detection function can be used without adaptations. This method is particularly of interest to facilitate accountability and prevent societal harm.

摘要: 大型语言模型(LLM)的快速发展使得区分人类和机器编写的文本变得越来越困难。虽然LLM已经存在水印，但它们往往缺乏灵活性，并与释义等攻击作斗争。为了解决这些问题，我们提出了一种多特征生成水印的方法，该方法将多个不同的水印特征组合成一个集成水印。具体地说，我们将肢端和感觉运动规范与所建立的红绿水印相结合，达到了98%的检测率。经过改写攻击后，性能保持在95%的高检测率。仅以红绿特征作为基线就能达到49%的检测率。对所有特征组合的评估表明，在几个LLM和水印强度设置中，所有三个特征组合的集成始终具有最高的检测率。由于可以灵活地组合整体中的功能，因此可以满足各种需求和权衡。此外，对于所有合奏配置，可以使用相同的检测功能，而无需进行适配。这种方法对促进问责和防止社会危害特别有意义。



## **44. InputSnatch: Stealing Input in LLM Services via Timing Side-Channel Attacks**

InputSnatch：通过定时侧通道攻击窃取LLM服务中的输入 cs.CR

**SubmitDate**: 2024-11-29    [abs](http://arxiv.org/abs/2411.18191v2) [paper-pdf](http://arxiv.org/pdf/2411.18191v2)

**Authors**: Xinyao Zheng, Husheng Han, Shangyi Shi, Qiyan Fang, Zidong Du, Xing Hu, Qi Guo

**Abstract**: Large language models (LLMs) possess extensive knowledge and question-answering capabilities, having been widely deployed in privacy-sensitive domains like finance and medical consultation. During LLM inferences, cache-sharing methods are commonly employed to enhance efficiency by reusing cached states or responses for the same or similar inference requests. However, we identify that these cache mechanisms pose a risk of private input leakage, as the caching can result in observable variations in response times, making them a strong candidate for a timing-based attack hint.   In this study, we propose a novel timing-based side-channel attack to execute input theft in LLMs inference. The cache-based attack faces the challenge of constructing candidate inputs in a large search space to hit and steal cached user queries. To address these challenges, we propose two primary components. The input constructor employs machine learning techniques and LLM-based approaches for vocabulary correlation learning while implementing optimized search mechanisms for generalized input construction. The time analyzer implements statistical time fitting with outlier elimination to identify cache hit patterns, continuously providing feedback to refine the constructor's search strategy. We conduct experiments across two cache mechanisms and the results demonstrate that our approach consistently attains high attack success rates in various applications. Our work highlights the security vulnerabilities associated with performance optimizations, underscoring the necessity of prioritizing privacy and security alongside enhancements in LLM inference.

摘要: 大型语言模型(LLM)具有广泛的知识和问答能力，已广泛应用于金融、医疗咨询等隐私敏感领域。在LLM推理期间，通常使用高速缓存共享方法来通过对相同或相似的推理请求重复使用高速缓存的状态或响应来提高效率。然而，我们发现这些缓存机制带来了私有输入泄漏的风险，因为缓存可能会导致响应时间的明显变化，从而使它们成为基于时间的攻击提示的有力候选者。在这项研究中，我们提出了一种新的基于时序的旁路攻击来执行LLMS推理中的输入窃取。基于缓存的攻击面临着在大搜索空间中构建候选输入以命中和窃取缓存的用户查询的挑战。为了应对这些挑战，我们提出了两个主要组成部分。输入构造器使用机器学习技术和基于LLM的方法进行词汇关联学习，同时实现优化的搜索机制来构建通用输入。时间分析器使用异常值消除来实现统计时间拟合，以识别缓存命中模式，并持续提供反馈以改进构造器的搜索策略。我们在两种缓存机制上进行了实验，结果表明，我们的方法在不同的应用中都取得了很高的攻击成功率。我们的工作突出了与性能优化相关的安全漏洞，强调了在增强LLM推理的同时优先考虑隐私和安全的必要性。



## **45. RePD: Defending Jailbreak Attack through a Retrieval-based Prompt Decomposition Process**

RePD：通过基于检索的即时分解过程防御越狱攻击 cs.CR

**SubmitDate**: 2024-11-29    [abs](http://arxiv.org/abs/2410.08660v3) [paper-pdf](http://arxiv.org/pdf/2410.08660v3)

**Authors**: Peiran Wang, Xiaogeng Liu, Chaowei Xiao

**Abstract**: In this study, we introduce RePD, an innovative attack Retrieval-based Prompt Decomposition framework designed to mitigate the risk of jailbreak attacks on large language models (LLMs). Despite rigorous pretraining and finetuning focused on ethical alignment, LLMs are still susceptible to jailbreak exploits. RePD operates on a one-shot learning model, wherein it accesses a database of pre-collected jailbreak prompt templates to identify and decompose harmful inquiries embedded within user prompts. This process involves integrating the decomposition of the jailbreak prompt into the user's original query into a one-shot learning example to effectively teach the LLM to discern and separate malicious components. Consequently, the LLM is equipped to first neutralize any potentially harmful elements before addressing the user's prompt in a manner that aligns with its ethical guidelines. RePD is versatile and compatible with a variety of open-source LLMs acting as agents. Through comprehensive experimentation with both harmful and benign prompts, we have demonstrated the efficacy of our proposed RePD in enhancing the resilience of LLMs against jailbreak attacks, without compromising their performance in responding to typical user requests.

摘要: 在这项研究中，我们介绍了RePD，一个创新的基于攻击检索的提示分解框架，旨在降低对大型语言模型(LLM)的越狱攻击风险。尽管严格的预训和微调侧重于道德一致性，但LLM仍然容易受到越狱利用的影响。RePD运行在一次性学习模式上，其中它访问预先收集的越狱提示模板数据库，以识别和分解嵌入用户提示中的有害查询。这一过程包括将越狱提示的分解集成到用户的原始查询中，并将其整合为一个一次性学习示例，以有效地教会LLM识别和分离恶意组件。因此，LLM配备了首先中和任何潜在有害元素，然后以符合其道德准则的方式处理用户的提示。RePD是通用的，并与各种作为代理的开源LLM兼容。通过对有害提示和良性提示的全面实验，我们已经证明了我们提出的RePD在增强LLM对越狱攻击的弹性方面的有效性，而不会影响它们响应典型用户请求的性能。



## **46. Confidential Prompting: Protecting User Prompts from Cloud LLM Providers**

机密预算：保护用户预算免受云LLM提供商的预算 cs.CR

**SubmitDate**: 2024-11-28    [abs](http://arxiv.org/abs/2409.19134v2) [paper-pdf](http://arxiv.org/pdf/2409.19134v2)

**Authors**: In Gim, Caihua Li, Lin Zhong

**Abstract**: Our work tackles the challenge of securing user inputs in cloud-hosted large language model (LLM) serving while ensuring output invariance, model confidentiality, and compute efficiency. We introduce secure multi-party decoding (SMD), which leverages confidential computing to confine user prompts to a trusted execution environment (TEE), namely a confidential virtual machine (CVM), while allowing service providers to generate tokens efficiently. We also introduce a novel cryptographic method, prompt obfuscation (PO), to ensure robustness against reconstruction attacks on SMD. We demonstrate that our approach preserves both prompt confidentiality and LLM serving efficiency. Our solution can enable privacy-preserving cloud LLM serving that handles sensitive prompts, such as clinical records, financial data, and personal information.

摘要: 我们的工作解决了在云托管大型语言模型（LLM）服务中保护用户输入的挑战，同时确保输出不变性、模型机密性和计算效率。我们引入了安全多方解码（MED），它利用机密计算将用户提示限制在可信执行环境（TEK），即机密虚拟机（CGM），同时允许服务提供商高效地生成令牌。我们还引入了一种新颖的加密方法--即时混淆（PO），以确保抵御对贴片的重建攻击的鲁棒性。我们证明我们的方法既保留了即时的机密性，又保留了LLM服务效率。我们的解决方案可以实现保护隐私的云LLM服务，该服务可以处理敏感提示，例如临床记录、财务数据和个人信息。



## **47. Memorization of Named Entities in Fine-tuned BERT Models**

微调BERT模型中命名实体的子化 cs.CL

published at CD-MAKE 2023

**SubmitDate**: 2024-11-28    [abs](http://arxiv.org/abs/2212.03749v3) [paper-pdf](http://arxiv.org/pdf/2212.03749v3)

**Authors**: Andor Diera, Nicolas Lell, Aygul Garifullina, Ansgar Scherp

**Abstract**: Privacy preserving deep learning is an emerging field in machine learning that aims to mitigate the privacy risks in the use of deep neural networks. One such risk is training data extraction from language models that have been trained on datasets, which contain personal and privacy sensitive information. In our study, we investigate the extent of named entity memorization in fine-tuned BERT models. We use single-label text classification as representative downstream task and employ three different fine-tuning setups in our experiments, including one with Differential Privacy (DP). We create a large number of text samples from the fine-tuned BERT models utilizing a custom sequential sampling strategy with two prompting strategies. We search in these samples for named entities and check if they are also present in the fine-tuning datasets. We experiment with two benchmark datasets in the domains of emails and blogs. We show that the application of DP has a detrimental effect on the text generation capabilities of BERT. Furthermore, we show that a fine-tuned BERT does not generate more named entities specific to the fine-tuning dataset than a BERT model that is pre-trained only. This suggests that BERT is unlikely to emit personal or privacy sensitive named entities. Overall, our results are important to understand to what extent BERT-based services are prone to training data extraction attacks.

摘要: 隐私保护深度学习是机器学习中的一个新兴领域，旨在降低深度神经网络使用中的隐私风险。其中一个风险是从已在数据集上训练的语言模型中提取训练数据，这些数据集包含个人和隐私敏感信息。在我们的研究中，我们考察了微调的BERT模型中命名实体记忆的程度。我们使用单标签文本分类作为代表性的下游任务，并在实验中使用了三种不同的微调设置，其中一种设置为差分隐私(DP)。我们利用定制的顺序采样策略和两种提示策略，从微调的BERT模型创建了大量的文本样本。我们在这些样本中搜索命名实体，并检查它们是否也出现在微调数据集中。我们在电子邮件和博客领域试验了两个基准数据集。结果表明，DP的应用对BERT的文本生成能力有不利影响。此外，我们还表明，与仅经过预训练的BERT模型相比，经过微调的ERT并不会生成更多特定于微调数据集的命名实体。这表明伯特不太可能发出个人或隐私敏感的命名实体。总体而言，我们的结果对于了解基于BERT的服务在多大程度上容易受到训练数据提取攻击具有重要意义。



## **48. On Evaluating The Performance of Watermarked Machine-Generated Texts Under Adversarial Attacks**

关于评估带有水印的机器生成文本在对抗性攻击下的性能 cs.CR

**SubmitDate**: 2024-11-28    [abs](http://arxiv.org/abs/2407.04794v2) [paper-pdf](http://arxiv.org/pdf/2407.04794v2)

**Authors**: Zesen Liu, Tianshuo Cong, Xinlei He, Qi Li

**Abstract**: Large Language Models (LLMs) excel in various applications, including text generation and complex tasks. However, the misuse of LLMs raises concerns about the authenticity and ethical implications of the content they produce, such as deepfake news, academic fraud, and copyright infringement. Watermarking techniques, which embed identifiable markers in machine-generated text, offer a promising solution to these issues by allowing for content verification and origin tracing. Unfortunately, the robustness of current LLM watermarking schemes under potential watermark removal attacks has not been comprehensively explored.   In this paper, to fill this gap, we first systematically comb the mainstream watermarking schemes and removal attacks on machine-generated texts, and then we categorize them into pre-text (before text generation) and post-text (after text generation) classes so that we can conduct diversified analyses. In our experiments, we evaluate eight watermarks (five pre-text, three post-text) and twelve attacks (two pre-text, ten post-text) across 87 scenarios. Evaluation results indicate that (1) KGW and Exponential watermarks offer high text quality and watermark retention but remain vulnerable to most attacks; (2) Post-text attacks are found to be more efficient and practical than pre-text attacks; (3) Pre-text watermarks are generally more imperceptible, as they do not alter text fluency, unlike post-text watermarks; (4) Additionally, combined attack methods can significantly increase effectiveness, highlighting the need for more robust watermarking solutions. Our study underscores the vulnerabilities of current techniques and the necessity for developing more resilient schemes.

摘要: 大型语言模型(LLM)在各种应用中表现出色，包括文本生成和复杂任务。然而，LLMS的滥用引发了人们对它们产生的内容的真实性和伦理影响的担忧，例如深度假新闻、学术欺诈和侵犯版权。在机器生成的文本中嵌入可识别标记的水印技术，通过允许内容验证和来源追踪，为这些问题提供了一种有前途的解决方案。遗憾的是，目前的LLM水印方案在潜在的水印去除攻击下的稳健性还没有得到全面的研究。为了填补这一空白，本文首先对主流的机器生成文本水印算法和去除攻击进行了系统的梳理，然后将其分为前文本类(文本生成前)和后文本类(文本生成后)，以便进行多样化的分析。在我们的实验中，我们评估了87个场景中的8个水印(5个前置文本，3个后置文本)和12个攻击(2个前置文本，10个后置文本)。评估结果表明：(1)KGW和指数水印具有高的文本质量和水印保留率，但仍然容易受到大多数攻击；(2)后文本攻击被发现比前文本攻击更有效和实用；(3)前文本水印通常更不可察觉，因为它们不像后文本水印那样改变文本的流畅性；(4)此外，组合攻击方法可以显著提高攻击效果，突出了对更健壮的水印解决方案的需求。我们的研究强调了当前技术的脆弱性，以及开发更具弹性的方案的必要性。



## **49. Assessing biomedical knowledge robustness in large language models by query-efficient sampling attacks**

通过查询高效抽样攻击评估大型语言模型中生物医学知识的稳健性 cs.CL

31 pages incl. appendix, accepted by TMLR

**SubmitDate**: 2024-11-28    [abs](http://arxiv.org/abs/2402.10527v3) [paper-pdf](http://arxiv.org/pdf/2402.10527v3)

**Authors**: R. Patrick Xian, Alex J. Lee, Satvik Lolla, Vincent Wang, Qiming Cui, Russell Ro, Reza Abbasi-Asl

**Abstract**: The increasing depth of parametric domain knowledge in large language models (LLMs) is fueling their rapid deployment in real-world applications. Understanding model vulnerabilities in high-stakes and knowledge-intensive tasks is essential for quantifying the trustworthiness of model predictions and regulating their use. The recent discovery of named entities as adversarial examples (i.e. adversarial entities) in natural language processing tasks raises questions about their potential impact on the knowledge robustness of pre-trained and finetuned LLMs in high-stakes and specialized domains. We examined the use of type-consistent entity substitution as a template for collecting adversarial entities for billion-parameter LLMs with biomedical knowledge. To this end, we developed an embedding-space attack based on powerscaled distance-weighted sampling to assess the robustness of their biomedical knowledge with a low query budget and controllable coverage. Our method has favorable query efficiency and scaling over alternative approaches based on random sampling and blackbox gradient-guided search, which we demonstrated for adversarial distractor generation in biomedical question answering. Subsequent failure mode analysis uncovered two regimes of adversarial entities on the attack surface with distinct characteristics and we showed that entity substitution attacks can manipulate token-wise Shapley value explanations, which become deceptive in this setting. Our approach complements standard evaluations for high-capacity models and the results highlight the brittleness of domain knowledge in LLMs.

摘要: 大型语言模型(LLM)中参数领域知识的不断深入推动了它们在现实世界应用程序中的快速部署。了解高风险和知识密集型任务中的模型脆弱性对于量化模型预测的可信度和规范其使用至关重要。最近在自然语言处理任务中发现了命名实体作为对抗性实例(即对抗性实体)，这引发了人们对高风险和专门领域中预先训练和精细调整的LLM知识稳健性的潜在影响的问题。我们研究了使用类型一致的实体替换作为收集具有生物医学知识的10亿参数LLM的对抗性实体的模板。为此，我们提出了一种基于加权距离加权抽样的嵌入空间攻击方法，以较低的查询预算和可控的覆盖率来评估他们的生物医学知识的稳健性。与基于随机抽样和黑盒梯度引导搜索的方法相比，我们的方法具有良好的查询效率和伸缩性，并在生物医学问答中的对抗性干扰项生成中得到了验证。随后的失效模式分析揭示了攻击面上具有不同特征的两种对抗实体的机制，我们表明实体替换攻击可以操纵令人信服的Shapley值解释，在这种情况下，这种解释变得具有欺骗性。我们的方法补充了对大容量模型的标准评估，结果突出了领域知识在LLMS中的脆性。



## **50. Knowledge Database or Poison Base? Detecting RAG Poisoning Attack through LLM Activations**

知识库还是毒库？通过LLM激活检测RAG中毒攻击 cs.CR

**SubmitDate**: 2024-11-28    [abs](http://arxiv.org/abs/2411.18948v1) [paper-pdf](http://arxiv.org/pdf/2411.18948v1)

**Authors**: Xue Tan, Hao Luan, Mingyu Luo, Xiaoyan Sun, Ping Chen, Jun Dai

**Abstract**: As Large Language Models (LLMs) are progressively deployed across diverse fields and real-world applications, ensuring the security and robustness of LLMs has become ever more critical. Retrieval-Augmented Generation (RAG) is a cutting-edge approach designed to address the limitations of large language models (LLMs). By retrieving information from the relevant knowledge database, RAG enriches the input to LLMs, enabling them to produce responses that are more accurate and contextually appropriate. It is worth noting that the knowledge database, being sourced from publicly available channels such as Wikipedia, inevitably introduces a new attack surface. RAG poisoning involves injecting malicious texts into the knowledge database, ultimately leading to the generation of the attacker's target response (also called poisoned response). However, there are currently limited methods available for detecting such poisoning attacks. We aim to bridge the gap in this work. Particularly, we introduce RevPRAG, a flexible and automated detection pipeline that leverages the activations of LLMs for poisoned response detection. Our investigation uncovers distinct patterns in LLMs' activations when generating correct responses versus poisoned responses. Our results on multiple benchmark datasets and RAG architectures show our approach could achieve 98% true positive rate, while maintaining false positive rates close to 1%. We also evaluate recent backdoor detection methods specifically designed for LLMs and applicable for identifying poisoned responses in RAG. The results demonstrate that our approach significantly surpasses them.

摘要: 随着大型语言模型(LLM)在不同领域和实际应用中的逐步部署，确保LLM的安全性和健壮性变得越来越重要。检索-增强生成(RAG)是一种尖端方法，旨在解决大型语言模型(LLM)的局限性。通过从相关知识数据库中检索信息，RAG丰富了对LLMS的输入，使它们能够做出更准确和更适合具体情况的答复。值得注意的是，来自维基百科等公开渠道的知识数据库不可避免地引入了新的攻击面。RAG中毒涉及将恶意文本注入知识库，最终导致生成攻击者的目标响应(也称为中毒响应)。然而，目前可用于检测此类中毒攻击的方法有限。我们的目标是弥合这项工作中的差距。特别是，我们引入了RevPRAG，这是一种灵活的自动化检测管道，它利用LLM的激活来进行中毒响应检测。我们的研究揭示了LLMS在产生正确反应和中毒反应时激活的不同模式。我们在多个基准数据集和RAG体系结构上的结果表明，我们的方法可以达到98%的真阳性率，同时将假阳性率保持在接近1%的水平。我们还评估了最近专门为LLMS设计的、适用于识别RAG中的中毒反应的后门检测方法。结果表明，我们的方法明显地超过了它们。



