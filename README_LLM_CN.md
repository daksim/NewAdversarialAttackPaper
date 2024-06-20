# Latest Large Language Model Attack Papers
**update at 2024-06-20 09:38:29**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Stealth edits for provably fixing or attacking large language models**

用于可证明修复或攻击大型语言模型的隐形编辑 cs.AI

24 pages, 9 figures. Open source implementation:  https://github.com/qinghua-zhou/stealth-edits

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12670v1) [paper-pdf](http://arxiv.org/pdf/2406.12670v1)

**Authors**: Oliver J. Sutton, Qinghua Zhou, Wei Wang, Desmond J. Higham, Alexander N. Gorban, Alexander Bastounis, Ivan Y. Tyukin

**Abstract**: We reveal new methods and the theoretical foundations of techniques for editing large language models. We also show how the new theory can be used to assess the editability of models and to expose their susceptibility to previously unknown malicious attacks. Our theoretical approach shows that a single metric (a specific measure of the intrinsic dimensionality of the model's features) is fundamental to predicting the success of popular editing approaches, and reveals new bridges between disparate families of editing methods. We collectively refer to these approaches as stealth editing methods, because they aim to directly and inexpensively update a model's weights to correct the model's responses to known hallucinating prompts without otherwise affecting the model's behaviour, without requiring retraining. By carefully applying the insight gleaned from our theoretical investigation, we are able to introduce a new network block -- named a jet-pack block -- which is optimised for highly selective model editing, uses only standard network operations, and can be inserted into existing networks. The intrinsic dimensionality metric also determines the vulnerability of a language model to a stealth attack: a small change to a model's weights which changes its response to a single attacker-chosen prompt. Stealth attacks do not require access to or knowledge of the model's training data, therefore representing a potent yet previously unrecognised threat to redistributed foundation models. They are computationally simple enough to be implemented in malware in many cases. Extensive experimental results illustrate and support the method and its theoretical underpinnings. Demos and source code for editing language models are available at https://github.com/qinghua-zhou/stealth-edits.

摘要: 我们揭示了编辑大型语言模型的新方法和技术的理论基础。我们还展示了如何使用新的理论来评估模型的可编辑性，并暴露它们对以前未知的恶意攻击的敏感性。我们的理论方法表明，单一指标(模型特征内在维度的特定衡量标准)是预测流行编辑方法成功的基础，并揭示了不同编辑方法家族之间的新桥梁。我们将这些方法统称为隐形编辑方法，因为它们旨在直接且廉价地更新模型的权重，以纠正模型对已知幻觉提示的反应，而不会以其他方式影响模型的行为，而不需要重新培训。通过仔细应用从我们的理论研究中收集到的见解，我们能够引入一种新的网络块--命名为JET-PACK块--它针对高度选择性的模型编辑进行了优化，仅使用标准的网络操作，并且可以插入到现有网络中。固有的维度度量还决定了语言模型对隐形攻击的脆弱性：对模型权重的微小更改会改变其对攻击者选择的单个提示的响应。隐形攻击不需要访问或了解模型的训练数据，因此对重新分布的基础模型构成了一个以前未被认识到的强大威胁。它们在计算上足够简单，在许多情况下可以在恶意软件中实现。大量的实验结果说明和支持了该方法及其理论基础。有关编辑语言模型的演示和源代码，请访问https://github.com/qinghua-zhou/stealth-edits.



## **2. MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models**

MM-SafetyBench：多模式大型语言模型安全评估的基准 cs.CV

The datasets were incomplete as they did not include all the  necessary copyrights

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2311.17600v4) [paper-pdf](http://arxiv.org/pdf/2311.17600v4)

**Authors**: Xin Liu, Yichen Zhu, Jindong Gu, Yunshi Lan, Chao Yang, Yu Qiao

**Abstract**: The security concerns surrounding Large Language Models (LLMs) have been extensively explored, yet the safety of Multimodal Large Language Models (MLLMs) remains understudied. In this paper, we observe that Multimodal Large Language Models (MLLMs) can be easily compromised by query-relevant images, as if the text query itself were malicious. To address this, we introduce MM-SafetyBench, a comprehensive framework designed for conducting safety-critical evaluations of MLLMs against such image-based manipulations. We have compiled a dataset comprising 13 scenarios, resulting in a total of 5,040 text-image pairs. Our analysis across 12 state-of-the-art models reveals that MLLMs are susceptible to breaches instigated by our approach, even when the equipped LLMs have been safety-aligned. In response, we propose a straightforward yet effective prompting strategy to enhance the resilience of MLLMs against these types of attacks. Our work underscores the need for a concerted effort to strengthen and enhance the safety measures of open-source MLLMs against potential malicious exploits. The resource is available at https://github.com/isXinLiu/MM-SafetyBench

摘要: 围绕大语言模型的安全问题已经得到了广泛的研究，但多模式大语言模型的安全性仍未得到充分的研究。在本文中，我们观察到多模式大型语言模型(MLLMS)很容易被与查询相关的图像破坏，就好像文本查询本身是恶意的一样。为了解决这一问题，我们引入了MM-SafetyBch，这是一个全面的框架，旨在针对此类基于图像的操作对MLLMS进行安全关键评估。我们汇编了一个包含13个场景的数据集，总共产生了5,040个文本-图像对。我们对12种最先进型号的分析表明，即使配备的LLM已经安全对准，MLLM也容易受到我们的方法引发的漏洞的影响。对此，我们提出了一种简单而有效的提示策略，以增强MLLMS对这些类型攻击的弹性。我们的工作强调了需要齐心协力加强和改进开放源码MLLM的安全措施，以防范潜在的恶意利用。该资源可在https://github.com/isXinLiu/MM-SafetyBench上获得



## **3. Authorship Obfuscation in Multilingual Machine-Generated Text Detection**

多语言机器生成文本检测中的作者混淆 cs.CL

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2401.07867v2) [paper-pdf](http://arxiv.org/pdf/2401.07867v2)

**Authors**: Dominik Macko, Robert Moro, Adaku Uchendu, Ivan Srba, Jason Samuel Lucas, Michiharu Yamashita, Nafis Irtiza Tripto, Dongwon Lee, Jakub Simko, Maria Bielikova

**Abstract**: High-quality text generation capability of recent Large Language Models (LLMs) causes concerns about their misuse (e.g., in massive generation/spread of disinformation). Machine-generated text (MGT) detection is important to cope with such threats. However, it is susceptible to authorship obfuscation (AO) methods, such as paraphrasing, which can cause MGTs to evade detection. So far, this was evaluated only in monolingual settings. Thus, the susceptibility of recently proposed multilingual detectors is still unknown. We fill this gap by comprehensively benchmarking the performance of 10 well-known AO methods, attacking 37 MGT detection methods against MGTs in 11 languages (i.e., 10 $\times$ 37 $\times$ 11 = 4,070 combinations). We also evaluate the effect of data augmentation on adversarial robustness using obfuscated texts. The results indicate that all tested AO methods can cause evasion of automated detection in all tested languages, where homoglyph attacks are especially successful. However, some of the AO methods severely damaged the text, making it no longer readable or easily recognizable by humans (e.g., changed language, weird characters).

摘要: 最近的大型语言模型(LLM)的高质量文本生成能力引起了人们对它们的滥用(例如，在大规模生成/传播虚假信息中)的担忧。机器生成文本(MGT)检测对于应对此类威胁非常重要。然而，它容易受到作者身份混淆(AO)方法的影响，例如转译，这可能导致MGTS逃避检测。到目前为止，这只在单一语言环境中进行了评估。因此，最近提出的多语言检测器的敏感性仍然未知。我们通过全面基准测试10种著名的AO方法的性能来填补这一空白，针对11种语言的MGT攻击37种MGT检测方法(即，10$\乘以$37$\乘以$11=4,070个组合)。我们还使用混淆文本来评估数据增强对对手健壮性的影响。结果表明，在所有被测语言中，所有被测试的声学方法都可以逃避自动检测，其中同形文字攻击尤其成功。然而，一些AO方法严重损坏了文本，使其不再可读或不再容易被人类识别(例如，改变语言、奇怪的字符)。



## **4. Can We Trust Large Language Models Generated Code? A Framework for In-Context Learning, Security Patterns, and Code Evaluations Across Diverse LLMs**

我们可以信任大型语言模型生成的代码吗？跨各种LLM的上下文学习、安全模式和代码评估框架 cs.CR

27 pages, Standard Journal Paper submitted to Q1 Elsevier

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12513v1) [paper-pdf](http://arxiv.org/pdf/2406.12513v1)

**Authors**: Ahmad Mohsin, Helge Janicke, Adrian Wood, Iqbal H. Sarker, Leandros Maglaras, Naeem Janjua

**Abstract**: Large Language Models (LLMs) such as ChatGPT and GitHub Copilot have revolutionized automated code generation in software engineering. However, as these models are increasingly utilized for software development, concerns have arisen regarding the security and quality of the generated code. These concerns stem from LLMs being primarily trained on publicly available code repositories and internet-based textual data, which may contain insecure code. This presents a significant risk of perpetuating vulnerabilities in the generated code, creating potential attack vectors for exploitation by malicious actors. Our research aims to tackle these issues by introducing a framework for secure behavioral learning of LLMs through In-Content Learning (ICL) patterns during the code generation process, followed by rigorous security evaluations. To achieve this, we have selected four diverse LLMs for experimentation. We have evaluated these coding LLMs across three programming languages and identified security vulnerabilities and code smells. The code is generated through ICL with curated problem sets and undergoes rigorous security testing to evaluate the overall quality and trustworthiness of the generated code. Our research indicates that ICL-driven one-shot and few-shot learning patterns can enhance code security, reducing vulnerabilities in various programming scenarios. Developers and researchers should know that LLMs have a limited understanding of security principles. This may lead to security breaches when the generated code is deployed in production systems. Our research highlights LLMs are a potential source of new vulnerabilities to the software supply chain. It is important to consider this when using LLMs for code generation. This research article offers insights into improving LLM security and encourages proactive use of LLMs for code generation to ensure software system safety.

摘要: ChatGPT和GitHub Copilot等大型语言模型(LLM)彻底改变了软件工程中的自动代码生成。然而，随着这些模型越来越多地用于软件开发，产生了对所生成代码的安全性和质量的担忧。这些担忧源于LLM主要接受关于公开可用的代码库和基于互联网的文本数据的培训，这些数据可能包含不安全的代码。这带来了使生成的代码中的漏洞永久化的重大风险，从而创建了潜在的攻击载体，供恶意攻击者利用。我们的研究旨在通过引入一个框架来解决这些问题，该框架在代码生成过程中通过内容内学习(ICL)模式来实现LLM的安全行为学习，然后进行严格的安全评估。为了实现这一点，我们选择了四种不同的LLM进行实验。我们已经在三种编程语言中评估了这些编码LLM，并确定了安全漏洞和代码气味。代码是通过ICL生成的，带有精选的问题集，并经过严格的安全测试，以评估生成的代码的整体质量和可信度。我们的研究表明，ICL驱动的一次和几次学习模式可以增强代码安全性，减少各种编程场景中的漏洞。开发人员和研究人员应该知道，LLM对安全原则的理解有限。当生成的代码部署在生产系统中时，这可能会导致安全漏洞。我们的研究强调，LLM是软件供应链新漏洞的潜在来源。在使用LLM进行代码生成时，考虑这一点非常重要。这篇研究文章提供了改进LLM安全性的见解，并鼓励主动使用LLM进行代码生成，以确保软件系统安全。



## **5. Identifying and Mitigating Privacy Risks Stemming from Language Models: A Survey**

识别和缓解源于语言模型的隐私风险：一项调查 cs.CL

15 pages

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2310.01424v2) [paper-pdf](http://arxiv.org/pdf/2310.01424v2)

**Authors**: Victoria Smith, Ali Shahin Shamsabadi, Carolyn Ashurst, Adrian Weller

**Abstract**: Large Language Models (LLMs) have shown greatly enhanced performance in recent years, attributed to increased size and extensive training data. This advancement has led to widespread interest and adoption across industries and the public. However, training data memorization in Machine Learning models scales with model size, particularly concerning for LLMs. Memorized text sequences have the potential to be directly leaked from LLMs, posing a serious threat to data privacy. Various techniques have been developed to attack LLMs and extract their training data. As these models continue to grow, this issue becomes increasingly critical. To help researchers and policymakers understand the state of knowledge around privacy attacks and mitigations, including where more work is needed, we present the first SoK on data privacy for LLMs. We (i) identify a taxonomy of salient dimensions where attacks differ on LLMs, (ii) systematize existing attacks, using our taxonomy of dimensions to highlight key trends, (iii) survey existing mitigation strategies, highlighting their strengths and limitations, and (iv) identify key gaps, demonstrating open problems and areas for concern.

摘要: 近年来，由于规模的增加和大量的训练数据，大型语言模型(LLM)的性能得到了极大的提高。这一进步引起了业界和公众的广泛兴趣和采用。然而，机器学习模型中的训练数据记忆随模型的大小而变化，尤其是对于LLMS。记忆的文本序列有可能直接从LLMS泄露，对数据隐私构成严重威胁。已经开发了各种技术来攻击LLMS并提取它们的训练数据。随着这些模式的不断发展，这个问题变得越来越关键。为了帮助研究人员和政策制定者了解有关隐私攻击和缓解的知识状况，包括需要更多工作的地方，我们提出了第一个关于低成本管理的数据隐私的SOK。我们(I)确定针对LLMS的攻击不同的显著维度的分类，(Ii)系统化现有攻击，使用我们的维度分类来突出关键趋势，(Iii)调查现有缓解策略，突出其优势和局限性，以及(Iv)确定关键差距，展示公开的问题和值得关注的领域。



## **6. Unique Security and Privacy Threats of Large Language Model: A Comprehensive Survey**

大型语言模型的独特安全和隐私威胁：全面调查 cs.CR

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.07973v2) [paper-pdf](http://arxiv.org/pdf/2406.07973v2)

**Authors**: Shang Wang, Tianqing Zhu, Bo Liu, Ming Ding, Xu Guo, Dayong Ye, Wanlei Zhou, Philip S. Yu

**Abstract**: With the rapid development of artificial intelligence, large language models (LLMs) have made remarkable advancements in natural language processing. These models are trained on vast datasets to exhibit powerful language understanding and generation capabilities across various applications, including machine translation, chatbots, and agents. However, LLMs have revealed a variety of privacy and security issues throughout their life cycle, drawing significant academic and industrial attention. Moreover, the risks faced by LLMs differ significantly from those encountered by traditional language models. Given that current surveys lack a clear taxonomy of unique threat models across diverse scenarios, we emphasize the unique privacy and security threats associated with five specific scenarios: pre-training, fine-tuning, retrieval-augmented generation systems, deployment, and LLM-based agents. Addressing the characteristics of each risk, this survey outlines potential threats and countermeasures. Research on attack and defense situations can offer feasible research directions, enabling more areas to benefit from LLMs.

摘要: 随着人工智能的快速发展，大语言模型在自然语言处理方面取得了显著的进步。这些模型是在海量数据集上进行训练的，以展示强大的语言理解和跨各种应用程序的生成能力，包括机器翻译、聊天机器人和代理。然而，LLMS在其整个生命周期中暴露了各种隐私和安全问题，引起了学术界和工业界的极大关注。此外，LLMS面临的风险与传统语言模型所遇到的风险有很大不同。鉴于目前的调查缺乏针对不同场景的独特威胁模型的明确分类，我们强调了与五种特定场景相关的独特隐私和安全威胁：预培训、微调、检索增强生成系统、部署和基于LLM的代理。针对每个风险的特点，本调查概述了潜在的威胁和对策。对攻防态势的研究可以提供可行的研究方向，使更多的地区受益于低成本管理。



## **7. Defending Against Social Engineering Attacks in the Age of LLMs**

在法学硕士时代防御社会工程攻击 cs.CL

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12263v1) [paper-pdf](http://arxiv.org/pdf/2406.12263v1)

**Authors**: Lin Ai, Tharindu Kumarage, Amrita Bhattacharjee, Zizhou Liu, Zheng Hui, Michael Davinroy, James Cook, Laura Cassani, Kirill Trapeznikov, Matthias Kirchner, Arslan Basharat, Anthony Hoogs, Joshua Garland, Huan Liu, Julia Hirschberg

**Abstract**: The proliferation of Large Language Models (LLMs) poses challenges in detecting and mitigating digital deception, as these models can emulate human conversational patterns and facilitate chat-based social engineering (CSE) attacks. This study investigates the dual capabilities of LLMs as both facilitators and defenders against CSE threats. We develop a novel dataset, SEConvo, simulating CSE scenarios in academic and recruitment contexts, and designed to examine how LLMs can be exploited in these situations. Our findings reveal that, while off-the-shelf LLMs generate high-quality CSE content, their detection capabilities are suboptimal, leading to increased operational costs for defense. In response, we propose ConvoSentinel, a modular defense pipeline that improves detection at both the message and the conversation levels, offering enhanced adaptability and cost-effectiveness. The retrieval-augmented module in ConvoSentinel identifies malicious intent by comparing messages to a database of similar conversations, enhancing CSE detection at all stages. Our study highlights the need for advanced strategies to leverage LLMs in cybersecurity.

摘要: 大型语言模型(LLM)的激增给检测和减轻数字欺骗带来了挑战，因为这些模型可以模拟人类的对话模式，并促进基于聊天的社会工程(CSE)攻击。本研究探讨低层管理人员作为CSE威胁的促进者和防御者的双重能力。我们开发了一个新的数据集SEConvo，模拟了学术和招聘环境中的CSE场景，并旨在研究如何在这些情况下利用LLM。我们的发现表明，虽然现成的LLM可以生成高质量的CSE内容，但它们的检测能力并不理想，从而导致防御操作成本增加。作为回应，我们提出了ConvoSentinel，这是一种模块化的防御管道，可以同时改进消息和会话级别的检测，提供更强的适应性和成本效益。ConvoSentinel中的检索增强模块通过将消息与类似对话的数据库进行比较来识别恶意意图，从而增强了所有阶段的CSE检测。我们的研究强调了在网络安全中利用低成本管理的高级战略的必要性。



## **8. Adversarial Attacks on Large Language Models in Medicine**

医学中对大型语言模型的对抗攻击 cs.AI

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12259v1) [paper-pdf](http://arxiv.org/pdf/2406.12259v1)

**Authors**: Yifan Yang, Qiao Jin, Furong Huang, Zhiyong Lu

**Abstract**: The integration of Large Language Models (LLMs) into healthcare applications offers promising advancements in medical diagnostics, treatment recommendations, and patient care. However, the susceptibility of LLMs to adversarial attacks poses a significant threat, potentially leading to harmful outcomes in delicate medical contexts. This study investigates the vulnerability of LLMs to two types of adversarial attacks in three medical tasks. Utilizing real-world patient data, we demonstrate that both open-source and proprietary LLMs are susceptible to manipulation across multiple tasks. This research further reveals that domain-specific tasks demand more adversarial data in model fine-tuning than general domain tasks for effective attack execution, especially for more capable models. We discover that while integrating adversarial data does not markedly degrade overall model performance on medical benchmarks, it does lead to noticeable shifts in fine-tuned model weights, suggesting a potential pathway for detecting and countering model attacks. This research highlights the urgent need for robust security measures and the development of defensive mechanisms to safeguard LLMs in medical applications, to ensure their safe and effective deployment in healthcare settings.

摘要: 将大型语言模型(LLM)集成到医疗保健应用程序中，在医疗诊断、治疗建议和患者护理方面提供了有希望的进步。然而，LLMS对对抗性攻击的敏感性构成了一个重大威胁，可能会在微妙的医疗环境中导致有害后果。本研究调查了LLMS在三个医疗任务中对两种类型的对抗性攻击的脆弱性。利用真实世界的患者数据，我们证明了开源和专有LLM都容易受到跨多个任务的操纵。这项研究进一步表明，特定领域的任务在模型微调中需要比一般领域任务更多的对抗性数据才能有效地执行攻击，特别是对于能力更强的模型。我们发现，虽然整合对抗性数据并不会显著降低医学基准上的整体模型性能，但它确实会导致微调模型权重的显著变化，这表明了一条检测和对抗模型攻击的潜在路径。这项研究强调了迫切需要强有力的安全措施和开发防御机制来保护医疗应用中的低成本管理，以确保其在医疗保健环境中的安全和有效部署。



## **9. CleanGen: Mitigating Backdoor Attacks for Generation Tasks in Large Language Models**

CleanGen：缓解大型语言模型中生成任务的后门攻击 cs.AI

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12257v1) [paper-pdf](http://arxiv.org/pdf/2406.12257v1)

**Authors**: Yuetai Li, Zhangchen Xu, Fengqing Jiang, Luyao Niu, Dinuka Sahabandu, Bhaskar Ramasubramanian, Radha Poovendran

**Abstract**: The remarkable performance of large language models (LLMs) in generation tasks has enabled practitioners to leverage publicly available models to power custom applications, such as chatbots and virtual assistants. However, the data used to train or fine-tune these LLMs is often undisclosed, allowing an attacker to compromise the data and inject backdoors into the models. In this paper, we develop a novel inference time defense, named CleanGen, to mitigate backdoor attacks for generation tasks in LLMs. CleanGenis a lightweight and effective decoding strategy that is compatible with the state-of-the-art (SOTA) LLMs. Our insight behind CleanGen is that compared to other LLMs, backdoored LLMs assign significantly higher probabilities to tokens representing the attacker-desired contents. These discrepancies in token probabilities enable CleanGen to identify suspicious tokens favored by the attacker and replace them with tokens generated by another LLM that is not compromised by the same attacker, thereby avoiding generation of attacker-desired content. We evaluate CleanGen against five SOTA backdoor attacks. Our results show that CleanGen achieves lower attack success rates (ASR) compared to five SOTA baseline defenses for all five backdoor attacks. Moreover, LLMs deploying CleanGen maintain helpfulness in their responses when serving benign user queries with minimal added computational overhead.

摘要: 大型语言模型(LLM)在生成任务中的出色性能使实践者能够利用公开可用的模型来支持定制应用程序，如聊天机器人和虚拟助手。然而，用于训练或微调这些LLM的数据往往是秘密的，这使得攻击者能够危害数据并向模型注入后门。本文提出了一种新的推理时间防御机制CleanGen，用于缓解LLMS中针对生成任务的后门攻击。CleanGenis是一种轻量级且有效的解码策略，与最先进的(SOTA)LLM兼容。我们在CleanGen背后的见解是，与其他LLM相比，反向LLM向代表攻击者所需内容的令牌分配的概率要高得多。令牌概率中的这些差异使CleanGen能够识别攻击者偏爱的可疑令牌，并将其替换为由另一个未被同一攻击者破解的LLM生成的令牌，从而避免生成攻击者所需的内容。我们对CleanGen进行了五次Sota后门攻击评估。我们的结果显示，对于所有五个后门攻击，CleanGen实现的攻击成功率(ASR)都低于五个SOTA基线防御。此外，部署CleanGen的LLMS在以最小的额外计算开销服务于良性用户查询时，在其响应中保持了帮助。



## **10. Privacy-Preserved Neural Graph Databases**

隐私保护的神经图数据库 cs.DB

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2312.15591v5) [paper-pdf](http://arxiv.org/pdf/2312.15591v5)

**Authors**: Qi Hu, Haoran Li, Jiaxin Bai, Zihao Wang, Yangqiu Song

**Abstract**: In the era of large language models (LLMs), efficient and accurate data retrieval has become increasingly crucial for the use of domain-specific or private data in the retrieval augmented generation (RAG). Neural graph databases (NGDBs) have emerged as a powerful paradigm that combines the strengths of graph databases (GDBs) and neural networks to enable efficient storage, retrieval, and analysis of graph-structured data which can be adaptively trained with LLMs. The usage of neural embedding storage and Complex neural logical Query Answering (CQA) provides NGDBs with generalization ability. When the graph is incomplete, by extracting latent patterns and representations, neural graph databases can fill gaps in the graph structure, revealing hidden relationships and enabling accurate query answering. Nevertheless, this capability comes with inherent trade-offs, as it introduces additional privacy risks to the domain-specific or private databases. Malicious attackers can infer more sensitive information in the database using well-designed queries such as from the answer sets of where Turing Award winners born before 1950 and after 1940 lived, the living places of Turing Award winner Hinton are probably exposed, although the living places may have been deleted in the training stage due to the privacy concerns. In this work, we propose a privacy-preserved neural graph database (P-NGDB) framework to alleviate the risks of privacy leakage in NGDBs. We introduce adversarial training techniques in the training stage to enforce the NGDBs to generate indistinguishable answers when queried with private information, enhancing the difficulty of inferring sensitive information through combinations of multiple innocuous queries.

摘要: 在大型语言模型(LLMS)时代，高效和准确的数据检索对于在检索增强生成(RAG)中使用特定领域或私有数据变得越来越重要。神经图形数据库(NGDB)已经成为一种强大的范例，它结合了图形数据库(GDB)和神经网络的优点，能够有效地存储、检索和分析图结构的数据，这些数据可以用LLMS进行自适应训练。神经嵌入存储和复杂神经逻辑查询应答(CQA)的使用为NGDB提供了泛化能力。当图不完整时，通过提取潜在模式和表示，神经图库可以填补图结构中的空白，揭示隐藏的关系，并使查询得到准确的回答。然而，这种能力是有内在权衡的，因为它会给特定于域或私有的数据库带来额外的隐私风险。恶意攻击者可以使用精心设计的查询来推断数据库中更敏感的信息，例如从图灵奖获得者1950年前和1940年后出生的地方的答案集中，图灵奖获得者辛顿的居住地可能会被曝光，尽管出于隐私考虑，居住地可能在培训阶段已被删除。在这项工作中，我们提出了一个隐私保护的神经图库(P-NGDB)框架，以缓解NGDB中隐私泄露的风险。在训练阶段引入对抗性训练技术，强制NGDB在查询私有信息时产生不可区分的答案，增加了通过组合多个无害查询来推断敏感信息的难度。



## **11. JailGuard: A Universal Detection Framework for LLM Prompt-based Attacks**

JailGuard：针对LLM基于预算的攻击的通用检测框架 cs.CR

28 pages, 9 figures

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2312.10766v3) [paper-pdf](http://arxiv.org/pdf/2312.10766v3)

**Authors**: Xiaoyu Zhang, Cen Zhang, Tianlin Li, Yihao Huang, Xiaojun Jia, Ming Hu, Jie Zhang, Yang Liu, Shiqing Ma, Chao Shen

**Abstract**: Large Language Models (LLMs) and Multi-Modal LLMs (MLLMs) have played a critical role in numerous applications. However, current LLMs are vulnerable to prompt-based attacks, with jailbreaking attacks enabling LLMs to generate harmful content, while hijacking attacks manipulate the model to perform unintended tasks, underscoring the necessity for detection methods. Unfortunately, existing detecting approaches are usually tailored to specific attacks, resulting in poor generalization in detecting various attacks across different modalities. To address it, we propose JailGuard, a universal detection framework for jailbreaking and hijacking attacks across LLMs and MLLMs. JailGuard operates on the principle that attacks are inherently less robust than benign ones, regardless of method or modality. Specifically, JailGuard mutates untrusted inputs to generate variants and leverages the discrepancy of the variants' responses on the model to distinguish attack samples from benign samples. We implement 18 mutators for text and image inputs and design a mutator combination policy to further improve detection generalization. To evaluate the effectiveness of JailGuard, we build the first comprehensive multi-modal attack dataset, containing 11,000 data items across 15 known attack types. The evaluation suggests that JailGuard achieves the best detection accuracy of 86.14%/82.90% on text and image inputs, outperforming state-of-the-art methods by 11.81%-25.73% and 12.20%-21.40%.

摘要: 大语言模型(LLM)和多模式LLM(MLLM)在许多应用中发挥了关键作用。然而，当前的LLM容易受到基于提示的攻击，越狱攻击使LLM能够生成有害内容，而劫持攻击操纵模型执行非预期任务，这突显了检测方法的必要性。遗憾的是，现有的检测方法通常是针对特定的攻击量身定做的，导致在检测不同模式的各种攻击时通用性较差。为了解决这个问题，我们提出了JailGuard，这是一个通用的检测框架，用于跨LLMS和MLLMS的越狱和劫持攻击。JailGuard的运作原则是，无论方法或方式如何，攻击天生就不如良性攻击那么强大。具体地说，JailGuard会变异不可信的输入以生成变体，并利用变体对模型的响应差异来区分攻击样本和良性样本。我们为文本和图像输入实现了18个变异器，并设计了变异器组合策略，进一步提高了检测的泛化能力。为了评估JailGuard的有效性，我们构建了第一个全面的多模式攻击数据集，包含15种已知攻击类型的11,000个数据项。评估表明，JailGuard对文本和图像输入的检测准确率达到了86.14%/82.90%，分别比最先进的方法高出11.81%-25.73%和12.20%-21.40%。



## **12. Safety Fine-Tuning at (Almost) No Cost: A Baseline for Vision Large Language Models**

（几乎）免费进行安全微调：Vision大型语言模型的基线 cs.LG

ICML 2024

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2402.02207v2) [paper-pdf](http://arxiv.org/pdf/2402.02207v2)

**Authors**: Yongshuo Zong, Ondrej Bohdal, Tingyang Yu, Yongxin Yang, Timothy Hospedales

**Abstract**: Current vision large language models (VLLMs) exhibit remarkable capabilities yet are prone to generate harmful content and are vulnerable to even the simplest jailbreaking attacks. Our initial analysis finds that this is due to the presence of harmful data during vision-language instruction fine-tuning, and that VLLM fine-tuning can cause forgetting of safety alignment previously learned by the underpinning LLM. To address this issue, we first curate a vision-language safe instruction-following dataset VLGuard covering various harmful categories. Our experiments demonstrate that integrating this dataset into standard vision-language fine-tuning or utilizing it for post-hoc fine-tuning effectively safety aligns VLLMs. This alignment is achieved with minimal impact on, or even enhancement of, the models' helpfulness. The versatility of our safety fine-tuning dataset makes it a valuable resource for safety-testing existing VLLMs, training new models or safeguarding pre-trained VLLMs. Empirical results demonstrate that fine-tuned VLLMs effectively reject unsafe instructions and substantially reduce the success rates of several black-box adversarial attacks, which approach zero in many cases. The code and dataset are available at https://github.com/ys-zong/VLGuard.

摘要: 目前的VISION大型语言模型(VLLM)显示出非凡的能力，但很容易产生有害内容，甚至容易受到最简单的越狱攻击。我们的初步分析发现，这是由于视觉语言教学微调过程中存在有害数据，而VLLM微调可能会导致忘记支持LLM之前学习的安全对齐。为了解决这个问题，我们首先策划了一个视觉-语言安全的指令遵循数据集VLGuard，涵盖了各种有害类别。我们的实验表明，将该数据集集成到标准视觉语言微调中或将其用于后自组织微调，可以有效地安全地对齐VLLM。这种对齐是在对模型的帮助最小的影响甚至是增强的情况下实现的。我们的安全微调数据集的多功能性使其成为安全测试现有VLLM、培训新模型或保护预先培训的VLLM的宝贵资源。实验结果表明，微调的VLLM有效地拒绝了不安全的指令，并显著降低了几种黑盒对抗攻击的成功率，这些攻击在许多情况下接近于零。代码和数据集可在https://github.com/ys-zong/VLGuard.上获得



## **13. Is poisoning a real threat to LLM alignment? Maybe more so than you think**

中毒是对LLM联盟的真正威胁吗？也许比你想象的还要多 cs.LG

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.12091v1) [paper-pdf](http://arxiv.org/pdf/2406.12091v1)

**Authors**: Pankayaraj Pathmanathan, Souradip Chakraborty, Xiangyu Liu, Yongyuan Liang, Furong Huang

**Abstract**: Recent advancements in Reinforcement Learning with Human Feedback (RLHF) have significantly impacted the alignment of Large Language Models (LLMs). The sensitivity of reinforcement learning algorithms such as Proximal Policy Optimization (PPO) has led to new line work on Direct Policy Optimization (DPO), which treats RLHF in a supervised learning framework. The increased practical use of these RLHF methods warrants an analysis of their vulnerabilities. In this work, we investigate the vulnerabilities of DPO to poisoning attacks under different scenarios and compare the effectiveness of preference poisoning, a first of its kind. We comprehensively analyze DPO's vulnerabilities under different types of attacks, i.e., backdoor and non-backdoor attacks, and different poisoning methods across a wide array of language models, i.e., LLama 7B, Mistral 7B, and Gemma 7B. We find that unlike PPO-based methods, which, when it comes to backdoor attacks, require at least 4\% of the data to be poisoned to elicit harmful behavior, we exploit the true vulnerabilities of DPO more simply so we can poison the model with only as much as 0.5\% of the data. We further investigate the potential reasons behind the vulnerability and how well this vulnerability translates into backdoor vs non-backdoor attacks.

摘要: 人类反馈强化学习(RLHF)的最新进展对大型语言模型(LLM)的匹配产生了重大影响。强化学习算法的敏感性，如最近策略优化(PPO)，导致了直接策略优化(DPO)的新工作，它在监督学习框架中处理RLHF。这些RLHF方法的实际使用越来越多，因此有理由对其脆弱性进行分析。在这项工作中，我们调查了DPO在不同场景下对中毒攻击的脆弱性，并比较了偏好中毒的有效性，这是第一次。我们全面分析了DPO在不同类型的攻击下的漏洞，即后门攻击和非后门攻击，以及不同的中毒方法，跨越了广泛的语言模型，即：大羊驼7B、米斯特拉尔7B和杰玛7B。我们发现，与基于PPO的方法不同，当涉及到后门攻击时，需要至少4%的数据被毒化才能引发有害行为，而我们更简单地利用DPO的真正漏洞，因此我们只需使用多达0.5%的数据就可以毒害模型。我们进一步调查了该漏洞背后的潜在原因，以及该漏洞在多大程度上转化为后门攻击与非后门攻击。



## **14. MLLM-Protector: Ensuring MLLM's Safety without Hurting Performance**

MLLM-保护者：确保MLLM的安全而不损害绩效 cs.CR

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2401.02906v3) [paper-pdf](http://arxiv.org/pdf/2401.02906v3)

**Authors**: Renjie Pi, Tianyang Han, Jianshu Zhang, Yueqi Xie, Rui Pan, Qing Lian, Hanze Dong, Jipeng Zhang, Tong Zhang

**Abstract**: The deployment of multimodal large language models (MLLMs) has brought forth a unique vulnerability: susceptibility to malicious attacks through visual inputs. This paper investigates the novel challenge of defending MLLMs against such attacks. Compared to large language models (LLMs), MLLMs include an additional image modality. We discover that images act as a ``foreign language" that is not considered during safety alignment, making MLLMs more prone to producing harmful responses. Unfortunately, unlike the discrete tokens considered in text-based LLMs, the continuous nature of image signals presents significant alignment challenges, which poses difficulty to thoroughly cover all possible scenarios. This vulnerability is exacerbated by the fact that most state-of-the-art MLLMs are fine-tuned on limited image-text pairs that are much fewer than the extensive text-based pretraining corpus, which makes the MLLMs more prone to catastrophic forgetting of their original abilities during safety fine-tuning. To tackle these challenges, we introduce MLLM-Protector, a plug-and-play strategy that solves two subtasks: 1) identifying harmful responses via a lightweight harm detector, and 2) transforming harmful responses into harmless ones via a detoxifier. This approach effectively mitigates the risks posed by malicious visual inputs without compromising the original performance of MLLMs. Our results demonstrate that MLLM-Protector offers a robust solution to a previously unaddressed aspect of MLLM security.

摘要: 多模式大型语言模型(MLLMS)的部署带来了一个独特的漏洞：通过视觉输入易受恶意攻击。本文研究了防御MLLMS免受此类攻击的新挑战。与大型语言模型(LLM)相比，MLLM包括一种额外的图像通道。我们发现，图像作为一种“外语”在安全对准过程中没有被考虑，使得MLLMS更容易产生有害的反应。不幸的是，与基于文本的LLMS中考虑的离散标记不同，图像信号的连续性质带来了巨大的对齐挑战，这使得很难完全覆盖所有可能的场景。大多数最先进的MLLS都是在有限的图文对上进行微调的，这比基于大量文本的预训练语料库要少得多，这使得MLLMS在安全微调期间更容易灾难性地忘记其原始能力，这加剧了这一漏洞。为了应对这些挑战，我们引入了MLLM-Protector，这是一种即插即用策略，可以解决两个子任务：1)通过轻型伤害检测器识别有害反应，2)通过解毒器将有害反应转化为无害反应。这种方法有效地降低了恶意视觉输入带来的风险，而不会影响MLLMS的原始性能。我们的结果表明，MLLM-Protector为MLLM安全的一个以前未解决的方面提供了一个健壮的解决方案。



## **15. Knowledge-to-Jailbreak: One Knowledge Point Worth One Attack**

知识越狱：一个知识点值得一次攻击 cs.CL

18 pages, 14 figures, 11 tables

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11682v1) [paper-pdf](http://arxiv.org/pdf/2406.11682v1)

**Authors**: Shangqing Tu, Zhuoran Pan, Wenxuan Wang, Zhexin Zhang, Yuliang Sun, Jifan Yu, Hongning Wang, Lei Hou, Juanzi Li

**Abstract**: Large language models (LLMs) have been increasingly applied to various domains, which triggers increasing concerns about LLMs' safety on specialized domains, e.g. medicine. However, testing the domain-specific safety of LLMs is challenging due to the lack of domain knowledge-driven attacks in existing benchmarks. To bridge this gap, we propose a new task, knowledge-to-jailbreak, which aims to generate jailbreaks from domain knowledge to evaluate the safety of LLMs when applied to those domains. We collect a large-scale dataset with 12,974 knowledge-jailbreak pairs and fine-tune a large language model as jailbreak-generator, to produce domain knowledge-specific jailbreaks. Experiments on 13 domains and 8 target LLMs demonstrate the effectiveness of jailbreak-generator in generating jailbreaks that are both relevant to the given knowledge and harmful to the target LLMs. We also apply our method to an out-of-domain knowledge base, showing that jailbreak-generator can generate jailbreaks that are comparable in harmfulness to those crafted by human experts. Data and code: https://github.com/THU-KEG/Knowledge-to-Jailbreak/.

摘要: 大语言模型被越来越多地应用到各个领域，这引发了人们对大语言模型在医学等专业领域的安全性的日益关注。然而，由于现有基准测试中缺乏领域知识驱动的攻击，因此测试LLMS的领域特定安全是具有挑战性的。为了弥补这一差距，我们提出了一个新的任务，知识越狱，其目的是从领域知识生成越狱来评估LLMS应用于这些领域时的安全性。我们收集了一个包含12,974个知识越狱对的大规模数据集，并微调了一个大型语言模型作为越狱生成器，以产生特定于领域知识的越狱。在13个领域和8个目标LLMS上的实验表明，越狱生成器能够有效地生成与给定知识相关且对目标LLMS有害的越狱。我们还将我们的方法应用于域外知识库，表明越狱生成器可以生成与人类专家创建的越狱在危害性上相当的越狱。数据和代码：https://github.com/THU-KEG/Knowledge-to-Jailbreak/.



## **16. Bileve: Securing Text Provenance in Large Language Models Against Spoofing with Bi-level Signature**

Bileve：通过双层签名保护大型语言模型中的文本出处，防止欺骗 cs.CR

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.01946v2) [paper-pdf](http://arxiv.org/pdf/2406.01946v2)

**Authors**: Tong Zhou, Xuandong Zhao, Xiaolin Xu, Shaolei Ren

**Abstract**: Text watermarks for large language models (LLMs) have been commonly used to identify the origins of machine-generated content, which is promising for assessing liability when combating deepfake or harmful content. While existing watermarking techniques typically prioritize robustness against removal attacks, unfortunately, they are vulnerable to spoofing attacks: malicious actors can subtly alter the meanings of LLM-generated responses or even forge harmful content, potentially misattributing blame to the LLM developer. To overcome this, we introduce a bi-level signature scheme, Bileve, which embeds fine-grained signature bits for integrity checks (mitigating spoofing attacks) as well as a coarse-grained signal to trace text sources when the signature is invalid (enhancing detectability) via a novel rank-based sampling strategy. Compared to conventional watermark detectors that only output binary results, Bileve can differentiate 5 scenarios during detection, reliably tracing text provenance and regulating LLMs. The experiments conducted on OPT-1.3B and LLaMA-7B demonstrate the effectiveness of Bileve in defeating spoofing attacks with enhanced detectability.

摘要: 大型语言模型(LLM)的文本水印通常用于识别机器生成内容的来源，这有望在打击深度虚假或有害内容时评估责任。虽然现有的水印技术通常将健壮性放在免受删除攻击的优先位置，但不幸的是，它们容易受到欺骗性攻击：恶意行为者可以巧妙地更改LLM生成的响应的含义，甚至伪造有害内容，可能会将责任错误地归咎于LLM开发人员。为了克服这一问题，我们提出了一种双层签名方案BiLEVE，该方案通过一种新颖的基于等级的采样策略嵌入细粒度的签名比特用于完整性检查(缓解欺骗攻击)，并在签名无效时嵌入粗粒度的信号来跟踪文本来源(增强了可检测性)。与传统的只输出二进制结果的水印检测器相比，BiLEVE在检测过程中可以区分5种场景，可靠地追踪文本来源和规范LLM。在OPT-1.3B和LLAMA-7B上进行的实验证明了BiLEVE在抵抗欺骗攻击方面的有效性，并增强了可检测性。



## **17. ART: Automatic Red-teaming for Text-to-Image Models to Protect Benign Users**

ART：文本到图像模型的自动红色团队以保护良性用户 cs.CR

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2405.19360v2) [paper-pdf](http://arxiv.org/pdf/2405.19360v2)

**Authors**: Guanlin Li, Kangjie Chen, Shudong Zhang, Jie Zhang, Tianwei Zhang

**Abstract**: Large-scale pre-trained generative models are taking the world by storm, due to their abilities in generating creative content. Meanwhile, safeguards for these generative models are developed, to protect users' rights and safety, most of which are designed for large language models. Existing methods primarily focus on jailbreak and adversarial attacks, which mainly evaluate the model's safety under malicious prompts. Recent work found that manually crafted safe prompts can unintentionally trigger unsafe generations. To further systematically evaluate the safety risks of text-to-image models, we propose a novel Automatic Red-Teaming framework, ART. Our method leverages both vision language model and large language model to establish a connection between unsafe generations and their prompts, thereby more efficiently identifying the model's vulnerabilities. With our comprehensive experiments, we reveal the toxicity of the popular open-source text-to-image models. The experiments also validate the effectiveness, adaptability, and great diversity of ART. Additionally, we introduce three large-scale red-teaming datasets for studying the safety risks associated with text-to-image models. Datasets and models can be found in https://github.com/GuanlinLee/ART.

摘要: 由于具有创造内容的能力，大规模的预先训练的生成性模型正在席卷世界。同时，为了保护用户的权利和安全，制定了对这些生成模型的保障措施，其中大部分是为大型语言模型设计的。现有的方法主要针对越狱和对抗性攻击，主要是在恶意提示下对模型的安全性进行评估。最近的研究发现，手动创建的安全提示可能会无意中引发不安全的世代。为了进一步系统地评估文本到图像模型的安全风险，我们提出了一个新的自动红色团队框架ART。我们的方法利用视觉语言模型和大型语言模型来建立不安全生成及其提示之间的联系，从而更有效地识别模型的漏洞。通过我们的综合实验，我们揭示了流行的开源文本到图像模型的毒性。实验也验证了ART的有效性、适应性和多样性。此外，我们还介绍了三个大型红团队数据集，用于研究与文本到图像模型相关的安全风险。数据集和模型可在https://github.com/GuanlinLee/ART.中找到



## **18. $\texttt{MoE-RBench}$: Towards Building Reliable Language Models with Sparse Mixture-of-Experts**

$\textttt {MoE-RBench}$：利用稀疏专家混合构建可靠的语言模型 cs.LG

9 pages, 8 figures, camera ready on ICML2024

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11353v1) [paper-pdf](http://arxiv.org/pdf/2406.11353v1)

**Authors**: Guanjie Chen, Xinyu Zhao, Tianlong Chen, Yu Cheng

**Abstract**: Mixture-of-Experts (MoE) has gained increasing popularity as a promising framework for scaling up large language models (LLMs). However, the reliability assessment of MoE lags behind its surging applications. Moreover, when transferred to new domains such as in fine-tuning MoE models sometimes underperform their dense counterparts. Motivated by the research gap and counter-intuitive phenomenon, we propose $\texttt{MoE-RBench}$, the first comprehensive assessment of SMoE reliability from three aspects: $\textit{(i)}$ safety and hallucination, $\textit{(ii)}$ resilience to adversarial attacks, and $\textit{(iii)}$ out-of-distribution robustness. Extensive models and datasets are tested to compare the MoE to dense networks from these reliability dimensions. Our empirical observations suggest that with appropriate hyperparameters, training recipes, and inference techniques, we can build the MoE model more reliably than the dense LLM. In particular, we find that the robustness of SMoE is sensitive to the basic training settings. We hope that this study can provide deeper insights into how to adapt the pre-trained MoE model to other tasks with higher-generation security, quality, and stability. Codes are available at https://github.com/UNITES-Lab/MoE-RBench

摘要: 专家混合(MOE)作为一种有前途的扩展大型语言模型(LLM)的框架已经越来越受欢迎。然而，MOE的可靠性评估落后于其激增的应用。此外，当转移到新的领域时，例如在微调的MOE模型中，有时表现不如密集的对应模型。受研究空白和反直觉现象的启发，我们首次从三个方面对SMOE的可靠性进行了全面的评估：安全和幻觉，对对手攻击的恢复能力，以及分布外的稳健性。测试了大量的模型和数据集，以从这些可靠性维度将MoE与密集网络进行比较。我们的经验观察表明，通过适当的超参数、训练配方和推理技术，我们可以建立比密集的LLM更可靠的MOE模型。特别是，我们发现SMOE的稳健性对基本训练设置很敏感。我们希望这项研究能够为如何将预先训练的MOE模型适应于具有更高一代安全性、质量和稳定性的其他任务提供更深层次的见解。有关代码，请访问https://github.com/UNITES-Lab/MoE-RBench



## **19. Adversarial Style Augmentation via Large Language Model for Robust Fake News Detection**

通过大语言模型进行对抗风格增强以实现稳健的假新闻检测 cs.CL

8 pages

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11260v1) [paper-pdf](http://arxiv.org/pdf/2406.11260v1)

**Authors**: Sungwon Park, Sungwon Han, Meeyoung Cha

**Abstract**: The spread of fake news negatively impacts individuals and is regarded as a significant social challenge that needs to be addressed. A number of algorithmic and insightful features have been identified for detecting fake news. However, with the recent LLMs and their advanced generation capabilities, many of the detectable features (e.g., style-conversion attacks) can be altered, making it more challenging to distinguish from real news. This study proposes adversarial style augmentation, AdStyle, to train a fake news detector that remains robust against various style-conversion attacks. Our model's key mechanism is the careful use of LLMs to automatically generate a diverse yet coherent range of style-conversion attack prompts. This improves the generation of prompts that are particularly difficult for the detector to handle. Experiments show that our augmentation strategy improves robustness and detection performance when tested on fake news benchmark datasets.

摘要: 假新闻的传播对个人产生负面影响，被视为需要解决的重大社会挑战。已经确定了许多算法和有洞察力的功能来检测假新闻。然而，随着最近的LLM及其先进一代能力，许多可检测的特征（例如，风格转换攻击）可以被更改，使其与真实新闻区分起来更具挑战性。这项研究提出了对抗性风格增强AdStyle来训练一个假新闻检测器，该检测器在对抗各种风格转换攻击时保持稳健。我们模型的关键机制是仔细使用LLM来自动生成多样化但连贯的风格转换攻击提示。这改善了检测器特别难以处理的提示的生成。实验表明，当在假新闻基准数据集上进行测试时，我们的增强策略提高了鲁棒性和检测性能。



## **20. Evading AI-Generated Content Detectors using Homoglyphs**

使用同字形躲避人工智能生成的内容检测器 cs.CL

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11239v1) [paper-pdf](http://arxiv.org/pdf/2406.11239v1)

**Authors**: Aldan Creo, Shushanta Pudasaini

**Abstract**: The generation of text that is increasingly human-like has been enabled by the advent of large language models (LLMs). As the detection of AI-generated content holds significant importance in the fight against issues such as misinformation and academic cheating, numerous studies have been conducted to develop reliable LLM detectors. While promising results have been demonstrated by such detectors on test data, recent research has revealed that they can be circumvented by employing different techniques. In this article, homoglyph-based ($a \rightarrow {\alpha}$) attacks that can be used to circumvent existing LLM detectors are presented. The efficacy of the attacks is illustrated by analizing how homoglyphs shift the tokenization of the text, and thus its token loglikelihoods. A comprehensive evaluation is conducted to assess the effectiveness of homoglyphs on state-of-the-art LLM detectors, including Binoculars, DetectGPT, OpenAI's detector, and watermarking techniques, on five different datasets. A significant reduction in the efficiency of all the studied configurations of detectors and datasets, down to an accuracy of 0.5 (random guessing), is demonstrated by the proposed approach. The results show that homoglyph-based attacks can effectively evade existing LLM detectors, and the implications of these findings are discussed along with possible defenses against such attacks.

摘要: 大型语言模型(LLM)的出现使得生成越来越像人类的文本成为可能。由于检测人工智能生成的内容在打击错误信息和学术作弊等问题方面具有重要意义，人们进行了大量研究，以开发可靠的LLM检测器。虽然这种探测器已经在测试数据上证明了有希望的结果，但最近的研究表明，可以通过使用不同的技术来规避这些结果。在这篇文章中，提出了可用于绕过现有LLM检测器的基于同形符号的($a\right tarrow{\alpha}$)攻击。通过分析同形文字如何改变文本的标记化，从而改变其标记性日志可能性，说明了攻击的有效性。在五个不同的数据集上，进行了一项全面的评估，以评估同种文字在最先进的LLM检测器上的有效性，包括双筒望远镜、DetectGPT、OpenAI的检测器和水印技术。通过提出的方法，所有研究的探测器和数据集的配置的效率都显著降低，精度降至0.5(随机猜测)。结果表明，基于同形文字的攻击可以有效地避开现有的LLM检测器，并讨论了这些发现的含义以及对此类攻击的可能防御。



## **21. GoldCoin: Grounding Large Language Models in Privacy Laws via Contextual Integrity Theory**

金币：通过上下文完整性理论将大型语言模型作为隐私法的基础 cs.CL

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11149v1) [paper-pdf](http://arxiv.org/pdf/2406.11149v1)

**Authors**: Wei Fan, Haoran Li, Zheye Deng, Weiqi Wang, Yangqiu Song

**Abstract**: Privacy issues arise prominently during the inappropriate transmission of information between entities. Existing research primarily studies privacy by exploring various privacy attacks, defenses, and evaluations within narrowly predefined patterns, while neglecting that privacy is not an isolated, context-free concept limited to traditionally sensitive data (e.g., social security numbers), but intertwined with intricate social contexts that complicate the identification and analysis of potential privacy violations. The advent of Large Language Models (LLMs) offers unprecedented opportunities for incorporating the nuanced scenarios outlined in privacy laws to tackle these complex privacy issues. However, the scarcity of open-source relevant case studies restricts the efficiency of LLMs in aligning with specific legal statutes. To address this challenge, we introduce a novel framework, GoldCoin, designed to efficiently ground LLMs in privacy laws for judicial assessing privacy violations. Our framework leverages the theory of contextual integrity as a bridge, creating numerous synthetic scenarios grounded in relevant privacy statutes (e.g., HIPAA), to assist LLMs in comprehending the complex contexts for identifying privacy risks in the real world. Extensive experimental results demonstrate that GoldCoin markedly enhances LLMs' capabilities in recognizing privacy risks across real court cases, surpassing the baselines on different judicial tasks.

摘要: 隐私问题突出地出现在实体之间不适当的信息传输过程中。现有的研究主要是通过在狭隘的预定义模式中探索各种隐私攻击、防御和评估来研究隐私，而忽略了隐私不是一个孤立的、与上下文无关的概念，仅限于传统的敏感数据(例如，社会安全号码)，而是与错综复杂的社会背景交织在一起，这使得识别和分析潜在的隐私侵犯变得复杂。大型语言模型(LLM)的出现为纳入隐私法中概述的细微差别场景提供了前所未有的机会，以解决这些复杂的隐私问题。然而，开源相关案例研究的匮乏限制了LLMS与具体法律法规保持一致的效率。为了应对这一挑战，我们引入了一个新的框架，GoldCoin，旨在有效地将LLM置于隐私法中，用于司法评估隐私侵权行为。我们的框架利用上下文完整性理论作为桥梁，创建基于相关隐私法规(例如HIPAA)的大量合成场景，以帮助LLMS理解复杂的上下文以识别现实世界中的隐私风险。广泛的实验结果表明，GoldCoin显著增强了LLMS在真实法庭案件中识别隐私风险的能力，超过了不同司法任务的基线。



## **22. Highlighting the Safety Concerns of Deploying LLMs/VLMs in Robotics**

强调在机器人技术中部署LLM/VLM的安全问题 cs.RO

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2402.10340v4) [paper-pdf](http://arxiv.org/pdf/2402.10340v4)

**Authors**: Xiyang Wu, Souradip Chakraborty, Ruiqi Xian, Jing Liang, Tianrui Guan, Fuxiao Liu, Brian M. Sadler, Dinesh Manocha, Amrit Singh Bedi

**Abstract**: In this paper, we highlight the critical issues of robustness and safety associated with integrating large language models (LLMs) and vision-language models (VLMs) into robotics applications. Recent works focus on using LLMs and VLMs to improve the performance of robotics tasks, such as manipulation and navigation. Despite these improvements, analyzing the safety of such systems remains underexplored yet extremely critical. LLMs and VLMs are highly susceptible to adversarial inputs, prompting a significant inquiry into the safety of robotic systems. This concern is important because robotics operate in the physical world where erroneous actions can result in severe consequences. This paper explores this issue thoroughly, presenting a mathematical formulation of potential attacks on LLM/VLM-based robotic systems and offering experimental evidence of the safety challenges. Our empirical findings highlight a significant vulnerability: simple modifications to the input can drastically reduce system effectiveness. Specifically, our results demonstrate an average performance deterioration of 19.4% under minor input prompt modifications and a more alarming 29.1% under slight perceptual changes. These findings underscore the urgent need for robust countermeasures to ensure the safe and reliable deployment of advanced LLM/VLM-based robotic systems.

摘要: 在这篇文章中，我们强调了与将大语言模型(LLM)和视觉语言模型(VLM)集成到机器人应用中相关的健壮性和安全性的关键问题。最近的工作集中在使用LLMS和VLMS来提高机器人任务的性能，如操纵和导航。尽管有了这些改进，分析这类系统的安全性仍然没有得到充分的探索，但仍然非常关键。LLM和VLM非常容易受到敌意输入的影响，这促使人们对机器人系统的安全性进行了重大调查。这一担忧很重要，因为机器人是在物理世界中运行的，在那里错误的行动可能会导致严重的后果。本文对这一问题进行了深入的探讨，给出了对基于LLM/VLM的机器人系统的潜在攻击的数学公式，并提供了安全挑战的实验证据。我们的经验发现突显了一个重大的脆弱性：对输入的简单修改可能会极大地降低系统效率。具体地说，我们的结果显示，在微小的输入提示修改下，性能平均下降了19.4%，而在轻微的感知变化下，性能下降了29.1%。这些发现突显了迫切需要强有力的对策，以确保安全可靠地部署先进的基于LLM/VLM的机器人系统。



## **23. garak: A Framework for Security Probing Large Language Models**

garak：大型语言模型安全探测框架 cs.CL

https://garak.ai

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2406.11036v1) [paper-pdf](http://arxiv.org/pdf/2406.11036v1)

**Authors**: Leon Derczynski, Erick Galinkin, Jeffrey Martin, Subho Majumdar, Nanna Inie

**Abstract**: As Large Language Models (LLMs) are deployed and integrated into thousands of applications, the need for scalable evaluation of how models respond to adversarial attacks grows rapidly. However, LLM security is a moving target: models produce unpredictable output, are constantly updated, and the potential adversary is highly diverse: anyone with access to the internet and a decent command of natural language. Further, what constitutes a security weak in one context may not be an issue in a different context; one-fits-all guardrails remain theoretical. In this paper, we argue that it is time to rethink what constitutes ``LLM security'', and pursue a holistic approach to LLM security evaluation, where exploration and discovery of issues are central. To this end, this paper introduces garak (Generative AI Red-teaming and Assessment Kit), a framework which can be used to discover and identify vulnerabilities in a target LLM or dialog system. garak probes an LLM in a structured fashion to discover potential vulnerabilities. The outputs of the framework describe a target model's weaknesses, contribute to an informed discussion of what composes vulnerabilities in unique contexts, and can inform alignment and policy discussions for LLM deployment.

摘要: 随着大型语言模型(LLM)的部署和集成到数以千计的应用程序中，对模型如何响应对手攻击的可扩展评估的需求迅速增长。然而，LLM安全是一个不断变化的目标：模型产生不可预测的输出，不断更新，潜在对手高度多样化：任何人都可以访问互联网，并相当熟练地掌握自然语言。此外，在一种情况下，什么构成安全薄弱，在另一种情况下可能不是问题；一刀切的护栏仍然是理论上的。在这篇文章中，我们认为现在是时候重新思考什么是“LLM安全”，并追求一种全面的方法来进行LLM安全评估，其中探索和发现问题是核心。为此，本文介绍了GARAK(生成性人工智能红团队和评估工具包)，这是一个可以用来发现和识别目标LLM或对话系统中的漏洞的框架。Garak以结构化方式探测LLM，以发现潜在漏洞。该框架的输出描述了目标模型的弱点，有助于对在特定环境中构成漏洞的因素进行明智的讨论，并可以为LLM部署的调整和策略讨论提供信息。



## **24. Threat Modelling and Risk Analysis for Large Language Model (LLM)-Powered Applications**

大型语言模型（LLM）支持的应用程序的威胁建模和风险分析 cs.CR

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2406.11007v1) [paper-pdf](http://arxiv.org/pdf/2406.11007v1)

**Authors**: Stephen Burabari Tete

**Abstract**: The advent of Large Language Models (LLMs) has revolutionized various applications by providing advanced natural language processing capabilities. However, this innovation introduces new cybersecurity challenges. This paper explores the threat modeling and risk analysis specifically tailored for LLM-powered applications. Focusing on potential attacks like data poisoning, prompt injection, SQL injection, jailbreaking, and compositional injection, we assess their impact on security and propose mitigation strategies. We introduce a framework combining STRIDE and DREAD methodologies for proactive threat identification and risk assessment. Furthermore, we examine the feasibility of an end-to-end threat model through a case study of a custom-built LLM-powered application. This model follows Shostack's Four Question Framework, adjusted for the unique threats LLMs present. Our goal is to propose measures that enhance the security of these powerful AI tools, thwarting attacks, and ensuring the reliability and integrity of LLM-integrated systems.

摘要: 大型语言模型(LLM)的出现提供了先进的自然语言处理能力，使各种应用发生了革命性的变化。然而，这一创新带来了新的网络安全挑战。本文探讨了专门为LLM支持的应用程序量身定做的威胁建模和风险分析。针对数据中毒、快速注入、SQL注入、越狱、成分注入等潜在攻击，评估了它们对安全的影响，并提出了缓解策略。我们引入了一个结合STRIDE和DREAD方法的框架，用于主动识别威胁和风险评估。此外，我们还通过一个定制的基于LLM的应用程序的案例研究，研究了端到端威胁模型的可行性。该模型遵循ShoStack的四个问题框架，针对LLMS存在的独特威胁进行了调整。我们的目标是提出措施，增强这些强大的人工智能工具的安全性，挫败攻击，并确保LLM集成系统的可靠性和完整性。



## **25. JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models**

越狱长凳：越狱大型语言模型的开放鲁棒性基准 cs.CR

JailbreakBench v1.0: more attack artifacts, more test-time defenses,  a more accurate jailbreak judge (Llama-3-70B with a custom prompt), a larger  dataset of human preferences for selecting a jailbreak judge (300 examples),  an over-refusal evaluation dataset (100 benign/borderline behaviors), a  semantic refusal judge based on Llama-3-8B

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2404.01318v3) [paper-pdf](http://arxiv.org/pdf/2404.01318v3)

**Authors**: Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, Eric Wong

**Abstract**: Jailbreak attacks cause large language models (LLMs) to generate harmful, unethical, or otherwise objectionable content. Evaluating these attacks presents a number of challenges, which the current collection of benchmarks and evaluation techniques do not adequately address. First, there is no clear standard of practice regarding jailbreaking evaluation. Second, existing works compute costs and success rates in incomparable ways. And third, numerous works are not reproducible, as they withhold adversarial prompts, involve closed-source code, or rely on evolving proprietary APIs. To address these challenges, we introduce JailbreakBench, an open-sourced benchmark with the following components: (1) an evolving repository of state-of-the-art adversarial prompts, which we refer to as jailbreak artifacts; (2) a jailbreaking dataset comprising 100 behaviors -- both original and sourced from prior work -- which align with OpenAI's usage policies; (3) a standardized evaluation framework at https://github.com/JailbreakBench/jailbreakbench that includes a clearly defined threat model, system prompts, chat templates, and scoring functions; and (4) a leaderboard at https://jailbreakbench.github.io/ that tracks the performance of attacks and defenses for various LLMs. We have carefully considered the potential ethical implications of releasing this benchmark, and believe that it will be a net positive for the community.

摘要: 越狱攻击会导致大型语言模型(LLM)生成有害、不道德或令人反感的内容。评估这些攻击带来了许多挑战，目前收集的基准和评估技术没有充分解决这些挑战。首先，关于越狱评估没有明确的实践标准。其次，现有的工作以无与伦比的方式计算成本和成功率。第三，许多作品是不可复制的，因为它们保留了对抗性提示，涉及封闭源代码，或者依赖于不断发展的专有API。为了应对这些挑战，我们引入了JailBreak，这是一个开源的基准测试，包括以下组件：(1)一个不断发展的最新对手提示库，我们称之为越狱人工制品；(2)一个包含100种行为的越狱数据集--既有原始的，也有来自以前工作的--与OpenAI的使用策略保持一致；(3)https://github.com/JailbreakBench/jailbreakbench的标准化评估框架，包括明确定义的威胁模型、系统提示、聊天模板和评分功能；以及(4)https://jailbreakbench.github.io/的排行榜，跟踪各种LLM的攻击和防御性能。我们已仔细考虑发布这一基准的潜在道德影响，并相信它将为社会带来净积极的影响。



## **26. ATM: Adversarial Tuning Multi-agent System Makes a Robust Retrieval-Augmented Generator**

ATM：对抗性调整多代理系统打造强大的检索增强生成器 cs.CL

18 pages, 7 figures

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2405.18111v2) [paper-pdf](http://arxiv.org/pdf/2405.18111v2)

**Authors**: Junda Zhu, Lingyong Yan, Haibo Shi, Dawei Yin, Lei Sha

**Abstract**: Large language models (LLMs) are proven to benefit a lot from retrieval-augmented generation (RAG) in alleviating hallucinations confronted with knowledge-intensive questions. RAG adopts information retrieval techniques to inject external knowledge from semantic-relevant documents as input contexts. However, due to today's Internet being flooded with numerous noisy and fabricating content, it is inevitable that RAG systems are vulnerable to these noises and prone to respond incorrectly. To this end, we propose to optimize the retrieval-augmented Generator with a Adversarial Tuning Multi-agent system (ATM). The ATM steers the Generator to have a robust perspective of useful documents for question answering with the help of an auxiliary Attacker agent. The Generator and the Attacker are tuned adversarially for several iterations. After rounds of multi-agent iterative tuning, the Generator can eventually better discriminate useful documents amongst fabrications. The experimental results verify the effectiveness of ATM and we also observe that the Generator can achieve better performance compared to state-of-the-art baselines.

摘要: 事实证明，大型语言模型(LLM)在缓解面对知识密集型问题时的幻觉方面，从检索增强生成(RAG)中受益匪浅。RAG采用信息检索技术，从与语义相关的文档中注入外部知识作为输入上下文。然而，由于当今的互联网充斥着大量噪声和捏造的内容，RAG系统不可避免地容易受到这些噪声的影响，并容易做出错误的响应。为此，我们提出了用对抗性调谐多智能体系统(ATM)来优化检索增强生成器。ATM引导生成器在辅助攻击者代理的帮助下具有用于问题回答的有用文档的健壮视角。生成器和攻击者被敌对地调整了几次迭代。经过几轮多代理迭代调整后，Generator最终可以更好地区分有用的文档和捏造的文档。实验结果验证了ATM的有效性，并且我们还观察到，与最先进的基线相比，该生成器可以获得更好的性能。



## **27. RWKU: Benchmarking Real-World Knowledge Unlearning for Large Language Models**

RWKU：大型语言模型的现实世界知识学习基准 cs.CL

48 pages, 7 figures, 12 tables

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2406.10890v1) [paper-pdf](http://arxiv.org/pdf/2406.10890v1)

**Authors**: Zhuoran Jin, Pengfei Cao, Chenhao Wang, Zhitao He, Hongbang Yuan, Jiachun Li, Yubo Chen, Kang Liu, Jun Zhao

**Abstract**: Large language models (LLMs) inevitably memorize sensitive, copyrighted, and harmful knowledge from the training corpus; therefore, it is crucial to erase this knowledge from the models. Machine unlearning is a promising solution for efficiently removing specific knowledge by post hoc modifying models. In this paper, we propose a Real-World Knowledge Unlearning benchmark (RWKU) for LLM unlearning. RWKU is designed based on the following three key factors: (1) For the task setting, we consider a more practical and challenging unlearning setting, where neither the forget corpus nor the retain corpus is accessible. (2) For the knowledge source, we choose 200 real-world famous people as the unlearning targets and show that such popular knowledge is widely present in various LLMs. (3) For the evaluation framework, we design the forget set and the retain set to evaluate the model's capabilities across various real-world applications. Regarding the forget set, we provide four four membership inference attack (MIA) methods and nine kinds of adversarial attack probes to rigorously test unlearning efficacy. Regarding the retain set, we assess locality and utility in terms of neighbor perturbation, general ability, reasoning ability, truthfulness, factuality, and fluency. We conduct extensive experiments across two unlearning scenarios, two models and six baseline methods and obtain some meaningful findings. We release our benchmark and code publicly at http://rwku-bench.github.io for future work.

摘要: 大型语言模型不可避免地会记住来自训练语料库的敏感、受版权保护和有害的知识；因此，从模型中删除这些知识至关重要。机器遗忘是通过事后修改模型来有效去除特定知识的一种很有前途的解决方案。本文提出了一种用于LLM遗忘的真实世界知识遗忘基准(RWKU)。RWKU的设计基于以下三个关键因素：(1)对于任务设置，我们考虑了一个更实际和更具挑战性的遗忘环境，其中忘记语料库和保留语料库都是不可访问的。(2)在知识源方面，我们选择了200名现实世界名人作为遗忘对象，发现这些流行知识广泛存在于各种学习记忆中。(3)对于评估框架，我们设计了遗忘集和保留集来评估模型在各种实际应用中的能力。对于遗忘集，我们提供了四种成员推理攻击(MIA)方法和九种对抗性攻击探头来严格测试遗忘效果。对于保留集，我们根据邻域扰动、一般能力、推理能力、真实性、真实性和流畅性来评估局部性和效用。我们在两个遗忘场景、两个模型和六个基线方法上进行了广泛的实验，并获得了一些有意义的发现。我们在http://rwku-bench.github.io上公开发布了我们的基准测试和代码，以备将来的工作使用。



## **28. KGPA: Robustness Evaluation for Large Language Models via Cross-Domain Knowledge Graphs**

KGMA：通过跨领域知识图对大型语言模型进行稳健性评估 cs.CL

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2406.10802v1) [paper-pdf](http://arxiv.org/pdf/2406.10802v1)

**Authors**: Aihua Pei, Zehua Yang, Shunan Zhu, Ruoxi Cheng, Ju Jia, Lina Wang

**Abstract**: Existing frameworks for assessing robustness of large language models (LLMs) overly depend on specific benchmarks, increasing costs and failing to evaluate performance of LLMs in professional domains due to dataset limitations. This paper proposes a framework that systematically evaluates the robustness of LLMs under adversarial attack scenarios by leveraging knowledge graphs (KGs). Our framework generates original prompts from the triplets of knowledge graphs and creates adversarial prompts by poisoning, assessing the robustness of LLMs through the results of these adversarial attacks. We systematically evaluate the effectiveness of this framework and its modules. Experiments show that adversarial robustness of the ChatGPT family ranks as GPT-4-turbo > GPT-4o > GPT-3.5-turbo, and the robustness of large language models is influenced by the professional domains in which they operate.

摘要: 用于评估大型语言模型（LLM）稳健性的现有框架过度依赖特定的基准，增加了成本，并且由于数据集限制而无法评估LLM在专业领域的性能。本文提出了一个框架，该框架通过利用知识图（KG）系统评估LLM在对抗性攻击场景下的稳健性。我们的框架从知识图的三重组中生成原始提示，并通过中毒创建对抗提示，通过这些对抗攻击的结果评估LLM的稳健性。我们系统地评估该框架及其模块的有效性。实验表明，ChatGPT家族的对抗鲁棒性排名为GPT-4-涡轮> GPT-4 o> GPT-3.5-涡轮，大型语言模型的鲁棒性受到其运行的专业领域的影响。



## **29. Towards Understanding Jailbreak Attacks in LLMs: A Representation Space Analysis**

了解LLC中的越狱攻击：表示空间分析 cs.CL

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2406.10794v1) [paper-pdf](http://arxiv.org/pdf/2406.10794v1)

**Authors**: Yuping Lin, Pengfei He, Han Xu, Yue Xing, Makoto Yamada, Hui Liu, Jiliang Tang

**Abstract**: Large language models (LLMs) are susceptible to a type of attack known as jailbreaking, which misleads LLMs to output harmful contents. Although there are diverse jailbreak attack strategies, there is no unified understanding on why some methods succeed and others fail. This paper explores the behavior of harmful and harmless prompts in the LLM's representation space to investigate the intrinsic properties of successful jailbreak attacks. We hypothesize that successful attacks share some similar properties: They are effective in moving the representation of the harmful prompt towards the direction to the harmless prompts. We leverage hidden representations into the objective of existing jailbreak attacks to move the attacks along the acceptance direction, and conduct experiments to validate the above hypothesis using the proposed objective. We hope this study provides new insights into understanding how LLMs understand harmfulness information.

摘要: 大型语言模型（LLM）容易受到一种称为越狱的攻击，这种攻击会误导LLM输出有害内容。尽管越狱攻击策略多种多样，但对于为什么有些方法成功而另一些方法失败，人们并没有统一的理解。本文探讨了LLM表示空间中有害和无害提示的行为，以研究成功越狱攻击的内在属性。我们假设成功的攻击具有一些相似的属性：它们有效地将有害提示的表示移向无害提示的方向。我们将隐藏的表示利用到现有越狱攻击的目标中，以沿着接受方向移动攻击，并使用提出的目标进行实验来验证上述假设。我们希望这项研究为理解LLM如何理解有害信息提供新的见解。



## **30. Adversarial Math Word Problem Generation**

对抗性数学单词问题生成 cs.CL

Code/data: https://github.com/ruoyuxie/adversarial_mwps_generation

**SubmitDate**: 2024-06-15    [abs](http://arxiv.org/abs/2402.17916v3) [paper-pdf](http://arxiv.org/pdf/2402.17916v3)

**Authors**: Roy Xie, Chengxuan Huang, Junlin Wang, Bhuwan Dhingra

**Abstract**: Large language models (LLMs) have significantly transformed the educational landscape. As current plagiarism detection tools struggle to keep pace with LLMs' rapid advancements, the educational community faces the challenge of assessing students' true problem-solving abilities in the presence of LLMs. In this work, we explore a new paradigm for ensuring fair evaluation -- generating adversarial examples which preserve the structure and difficulty of the original questions aimed for assessment, but are unsolvable by LLMs. Focusing on the domain of math word problems, we leverage abstract syntax trees to structurally generate adversarial examples that cause LLMs to produce incorrect answers by simply editing the numeric values in the problems. We conduct experiments on various open- and closed-source LLMs, quantitatively and qualitatively demonstrating that our method significantly degrades their math problem-solving ability. We identify shared vulnerabilities among LLMs and propose a cost-effective approach to attack high-cost models. Additionally, we conduct automatic analysis to investigate the cause of failure, providing further insights into the limitations of LLMs.

摘要: 大型语言模型(LLM)极大地改变了教育格局。由于目前的抄袭检测工具难以跟上LLMS的快速进步，教育界面临着在LLMS存在的情况下评估学生真正的问题解决能力的挑战。在这项工作中，我们探索了一种确保公平评价的新范式--生成对抗性实例，它保留了用于评价的原始问题的结构和难度，但无法用LLMS解决。聚焦于数学应用题领域，我们利用抽象语法树来结构化地生成对抗性实例，这些实例通过简单地编辑问题中的数值来导致LLMS产生不正确的答案。我们在各种开源和闭源的LLM上进行了实验，定量和定性地证明了我们的方法显著降低了他们的数学问题解决能力。我们识别了LLM之间的共同漏洞，并提出了一种具有成本效益的方法来攻击高成本模型。此外，我们还进行自动分析以调查故障原因，进一步深入了解LLMS的局限性。



## **31. Emerging Safety Attack and Defense in Federated Instruction Tuning of Large Language Models**

大型语言模型联邦指令调优中新兴的安全攻击和防御 cs.CL

18 pages

**SubmitDate**: 2024-06-15    [abs](http://arxiv.org/abs/2406.10630v1) [paper-pdf](http://arxiv.org/pdf/2406.10630v1)

**Authors**: Rui Ye, Jingyi Chai, Xiangrui Liu, Yaodong Yang, Yanfeng Wang, Siheng Chen

**Abstract**: Federated learning (FL) enables multiple parties to collaboratively fine-tune an large language model (LLM) without the need of direct data sharing. Ideally, by training on decentralized data that is aligned with human preferences and safety principles, federated instruction tuning can result in an LLM that could behave in a helpful and safe manner. In this paper, we for the first time reveal the vulnerability of safety alignment in FedIT by proposing a simple, stealthy, yet effective safety attack method. Specifically, the malicious clients could automatically generate attack data without involving manual efforts and attack the FedIT system by training their local LLMs on such attack data. Unfortunately, this proposed safety attack not only can compromise the safety alignment of LLM trained via FedIT, but also can not be effectively defended against by many existing FL defense methods. Targeting this, we further propose a post-hoc defense method, which could rely on a fully automated pipeline: generation of defense data and further fine-tuning of the LLM. Extensive experiments show that our safety attack method can significantly compromise the LLM's safety alignment (e.g., reduce safety rate by 70\%), which can not be effectively defended by existing defense methods (at most 4\% absolute improvement), while our safety defense method can significantly enhance the attacked LLM's safety alignment (at most 69\% absolute improvement).

摘要: 联合学习(FL)使多方能够协作微调大型语言模型(LLM)，而不需要直接共享数据。理想情况下，通过对符合人类偏好和安全原则的分散数据进行培训，联邦指令调优可以产生以有用和安全的方式运行的LLM。本文首次提出了一种简单、隐身、有效的安全攻击方法，揭示了FEDIT中安全对齐的脆弱性。具体地说，恶意客户端可以自动生成攻击数据，而无需手动操作，并通过对本地LLM进行此类攻击数据的训练来攻击FedIT系统。不幸的是，这种提出的安全攻击不仅会危及通过FedIT训练的LLM的安全对齐，而且现有的许多FL防御方法也无法有效防御。针对这一点，我们进一步提出了一种后自组织防御方法，该方法可以依赖于一条全自动化的管道：生成防御数据并进一步微调LLM。



## **32. KnowPhish: Large Language Models Meet Multimodal Knowledge Graphs for Enhancing Reference-Based Phishing Detection**

KnowPhish：大型语言模型满足多模式知识图，以增强基于参考的网络钓鱼检测 cs.CR

Accepted by USENIX Security 2024

**SubmitDate**: 2024-06-15    [abs](http://arxiv.org/abs/2403.02253v2) [paper-pdf](http://arxiv.org/pdf/2403.02253v2)

**Authors**: Yuexin Li, Chengyu Huang, Shumin Deng, Mei Lin Lock, Tri Cao, Nay Oo, Hoon Wei Lim, Bryan Hooi

**Abstract**: Phishing attacks have inflicted substantial losses on individuals and businesses alike, necessitating the development of robust and efficient automated phishing detection approaches. Reference-based phishing detectors (RBPDs), which compare the logos on a target webpage to a known set of logos, have emerged as the state-of-the-art approach. However, a major limitation of existing RBPDs is that they rely on a manually constructed brand knowledge base, making it infeasible to scale to a large number of brands, which results in false negative errors due to the insufficient brand coverage of the knowledge base. To address this issue, we propose an automated knowledge collection pipeline, using which we collect a large-scale multimodal brand knowledge base, KnowPhish, containing 20k brands with rich information about each brand. KnowPhish can be used to boost the performance of existing RBPDs in a plug-and-play manner. A second limitation of existing RBPDs is that they solely rely on the image modality, ignoring useful textual information present in the webpage HTML. To utilize this textual information, we propose a Large Language Model (LLM)-based approach to extract brand information of webpages from text. Our resulting multimodal phishing detection approach, KnowPhish Detector (KPD), can detect phishing webpages with or without logos. We evaluate KnowPhish and KPD on a manually validated dataset, and a field study under Singapore's local context, showing substantial improvements in effectiveness and efficiency compared to state-of-the-art baselines.

摘要: 网络钓鱼攻击已经给个人和企业造成了巨大的损失，需要开发强大而高效的自动网络钓鱼检测方法。基于引用的网络钓鱼检测器(RBPD)将目标网页上的标识与一组已知的标识进行比较，已成为最先进的方法。然而，现有的RBPD的一个主要局限性是依赖于手动构建的品牌知识库，使得无法扩展到大量的品牌，从而由于知识库的品牌覆盖率不足而导致假阴性错误。为了解决这一问题，我们提出了一种自动化的知识收集管道，利用该管道，我们收集了一个大规模的多模式品牌知识库KnowPhish，其中包含2万个品牌，每个品牌都有丰富的信息。KnowPhish可用于以即插即用的方式提高现有RBPD的性能。现有RBPD的第二个限制是它们仅依赖于图像通道，而忽略了网页HTML中存在的有用文本信息。为了利用这些文本信息，我们提出了一种基于大语言模型的方法来从文本中提取网页的品牌信息。我们由此产生的多模式钓鱼检测方法KnowPhish检测器(KPD)可以检测带有或没有徽标的钓鱼网页。我们对KnowPhish和KPD进行了手动验证的数据集和新加坡本地环境下的实地研究的评估，结果显示，与最先进的基线相比，KnowPhish和KPD在有效性和效率方面都有实质性的改进。



## **33. Semantic Membership Inference Attack against Large Language Models**

针对大型语言模型的语义成员推理攻击 cs.LG

**SubmitDate**: 2024-06-14    [abs](http://arxiv.org/abs/2406.10218v1) [paper-pdf](http://arxiv.org/pdf/2406.10218v1)

**Authors**: Hamid Mozaffari, Virendra J. Marathe

**Abstract**: Membership Inference Attacks (MIAs) determine whether a specific data point was included in the training set of a target model. In this paper, we introduce the Semantic Membership Inference Attack (SMIA), a novel approach that enhances MIA performance by leveraging the semantic content of inputs and their perturbations. SMIA trains a neural network to analyze the target model's behavior on perturbed inputs, effectively capturing variations in output probability distributions between members and non-members. We conduct comprehensive evaluations on the Pythia and GPT-Neo model families using the Wikipedia dataset. Our results show that SMIA significantly outperforms existing MIAs; for instance, SMIA achieves an AUC-ROC of 67.39% on Pythia-12B, compared to 58.90% by the second-best attack.

摘要: 成员资格推断攻击（MIA）确定特定数据点是否包含在目标模型的训练集中。在本文中，我们介绍了语义成员资格推理攻击（SMIA），这是一种新颖的方法，通过利用输入的语义内容及其扰动来增强MIA性能。SMIA训练神经网络来分析目标模型在受干扰输入上的行为，有效地捕捉成员和非成员之间输出概率分布的变化。我们使用维基百科数据集对Pythia和GPT-Neo模型家族进行全面评估。我们的结果表明，SMIA的表现显着优于现有的MIA;例如，SMIA在Pythia-12 B上的AUC-ROC为67.39%，而次佳攻击的AUC-ROC为58.90%。



## **34. Defending Large Language Models Against Jailbreak Attacks via Layer-specific Editing**

通过特定层的编辑保护大型语言模型免受越狱攻击 cs.AI

**SubmitDate**: 2024-06-14    [abs](http://arxiv.org/abs/2405.18166v2) [paper-pdf](http://arxiv.org/pdf/2405.18166v2)

**Authors**: Wei Zhao, Zhe Li, Yige Li, Ye Zhang, Jun Sun

**Abstract**: Large language models (LLMs) are increasingly being adopted in a wide range of real-world applications. Despite their impressive performance, recent studies have shown that LLMs are vulnerable to deliberately crafted adversarial prompts even when aligned via Reinforcement Learning from Human Feedback or supervised fine-tuning. While existing defense methods focus on either detecting harmful prompts or reducing the likelihood of harmful responses through various means, defending LLMs against jailbreak attacks based on the inner mechanisms of LLMs remains largely unexplored. In this work, we investigate how LLMs response to harmful prompts and propose a novel defense method termed \textbf{L}ayer-specific \textbf{Ed}iting (LED) to enhance the resilience of LLMs against jailbreak attacks. Through LED, we reveal that several critical \textit{safety layers} exist among the early layers of LLMs. We then show that realigning these safety layers (and some selected additional layers) with the decoded safe response from selected target layers can significantly improve the alignment of LLMs against jailbreak attacks. Extensive experiments across various LLMs (e.g., Llama2, Mistral) show the effectiveness of LED, which effectively defends against jailbreak attacks while maintaining performance on benign prompts. Our code is available at \url{https://github.com/ledllm/ledllm}.

摘要: 大型语言模型(LLM)正越来越多地被广泛地应用于现实世界中。尽管它们的表现令人印象深刻，但最近的研究表明，即使在通过从人类反馈的强化学习或监督微调进行调整时，LLM仍容易受到故意设计的敌意提示的攻击。虽然现有的防御方法侧重于检测有害提示或通过各种手段减少有害响应的可能性，但基于LLMS的内部机制来防御LLMS的越狱攻击在很大程度上仍未被探索。在这项工作中，我们研究了LLMS对有害提示的响应，并提出了一种新的防御方法-.通过LED，我们揭示了LLMS的早期层之间存在着几个关键的安全层。然后，我们展示了将这些安全层(以及一些选定的附加层)与选定目标层的解码安全响应重新对准可以显著提高LLM对抗越狱攻击的对准。在各种LLM(如Llama2、Mistral)上的广泛实验表明，LED是有效的，它可以有效防御越狱攻击，同时保持对良性提示的性能。我们的代码可在\url{https://github.com/ledllm/ledllm}.



## **35. REVS: Unlearning Sensitive Information in Language Models via Rank Editing in the Vocabulary Space**

REVS：通过词汇空间中的排名编辑消除语言模型中的敏感信息 cs.CL

18 pages, 3 figures

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.09325v1) [paper-pdf](http://arxiv.org/pdf/2406.09325v1)

**Authors**: Tomer Ashuach, Martin Tutek, Yonatan Belinkov

**Abstract**: Large language models (LLMs) risk inadvertently memorizing and divulging sensitive or personally identifiable information (PII) seen in training data, causing privacy concerns. Current approaches to address this issue involve costly dataset scrubbing, or model filtering through unlearning and model editing, which can be bypassed through extraction attacks. We propose REVS, a novel model editing method for unlearning sensitive information from LLMs. REVS identifies and modifies a small subset of neurons relevant for each piece of sensitive information. By projecting these neurons to the vocabulary space (unembedding), we pinpoint the components driving its generation. We then compute a model edit based on the pseudo-inverse of the unembedding matrix, and apply it to de-promote generation of the targeted sensitive data. To adequately evaluate our method on truly sensitive information, we curate two datasets: an email dataset inherently memorized by GPT-J, and a synthetic social security number dataset that we tune the model to memorize. Compared to other state-of-the-art model editing methods, REVS demonstrates superior performance in both eliminating sensitive information and robustness to extraction attacks, while retaining integrity of the underlying model. The code and a demo notebook are available at https://technion-cs-nlp.github.io/REVS.

摘要: 大型语言模型(LLM)可能会无意中记住和泄露训练数据中看到的敏感或个人身份信息(PII)，从而导致隐私问题。目前解决这一问题的方法包括代价高昂的数据集清理，或通过遗忘和模型编辑进行模型过滤，这可以通过提取攻击绕过。我们提出了一种新的模型编辑方法--REVS，用于遗忘LLMS中的敏感信息。Revs识别并修改与每条敏感信息相关的一小部分神经元。通过将这些神经元投射到词汇空间(非嵌入)，我们准确地找到了驱动其生成的组件。然后基于去嵌入矩阵的伪逆计算模型编辑，并将其应用于目标敏感数据的反加速生成。为了在真正敏感的信息上充分评估我们的方法，我们整理了两个数据集：GPT-J固有地记忆的电子邮件数据集，以及我们调整模型以记忆的合成社会安全号码数据集。与其他最先进的模型编辑方法相比，RERS在消除敏感信息和对提取攻击的稳健性方面表现出优越的性能，同时保持了底层模型的完整性。代码和演示笔记本可在https://technion-cs-nlp.github.io/REVS.上获得



## **36. Bag of Tricks: Benchmarking of Jailbreak Attacks on LLMs**

诡计袋：对LLM越狱攻击的基准 cs.CR

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.09324v1) [paper-pdf](http://arxiv.org/pdf/2406.09324v1)

**Authors**: Zhao Xu, Fan Liu, Hao Liu

**Abstract**: Although Large Language Models (LLMs) have demonstrated significant capabilities in executing complex tasks in a zero-shot manner, they are susceptible to jailbreak attacks and can be manipulated to produce harmful outputs. Recently, a growing body of research has categorized jailbreak attacks into token-level and prompt-level attacks. However, previous work primarily overlooks the diverse key factors of jailbreak attacks, with most studies concentrating on LLM vulnerabilities and lacking exploration of defense-enhanced LLMs. To address these issues, we evaluate the impact of various attack settings on LLM performance and provide a baseline benchmark for jailbreak attacks, encouraging the adoption of a standardized evaluation framework. Specifically, we evaluate the eight key factors of implementing jailbreak attacks on LLMs from both target-level and attack-level perspectives. We further conduct seven representative jailbreak attacks on six defense methods across two widely used datasets, encompassing approximately 320 experiments with about 50,000 GPU hours on A800-80G. Our experimental results highlight the need for standardized benchmarking to evaluate these attacks on defense-enhanced LLMs. Our code is available at https://github.com/usail-hkust/Bag_of_Tricks_for_LLM_Jailbreaking.

摘要: 尽管大型语言模型(LLM)在以零射击方式执行复杂任务方面表现出了巨大的能力，但它们很容易受到越狱攻击，并可能被操纵以产生有害的输出。最近，越来越多的研究将越狱攻击分为令牌级攻击和提示级攻击。然而，以前的工作主要忽略了越狱攻击的各种关键因素，大多数研究集中在LLM漏洞上，而缺乏对增强防御的LLM的探索。为了解决这些问题，我们评估了各种攻击设置对LLM性能的影响，并提供了越狱攻击的基准，鼓励采用标准化的评估框架。具体地，我们从目标级和攻击级两个角度评估了对LLMS实施越狱攻击的八个关键因素。我们进一步在两个广泛使用的数据集上对六种防御方法进行了七次有代表性的越狱攻击，包括在A800-80G上进行了大约320个实验，大约50,000个GPU小时。我们的实验结果强调了标准化基准测试的必要性，以评估这些针对防御增强型LLM的攻击。我们的代码可以在https://github.com/usail-hkust/Bag_of_Tricks_for_LLM_Jailbreaking.上找到



## **37. JailbreakEval: An Integrated Toolkit for Evaluating Jailbreak Attempts Against Large Language Models**

越狱Eval：用于评估针对大型语言模型的越狱尝试的集成工具包 cs.CR

Our code is available at https://github.com/ThuCCSLab/JailbreakEval

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.09321v1) [paper-pdf](http://arxiv.org/pdf/2406.09321v1)

**Authors**: Delong Ran, Jinyuan Liu, Yichen Gong, Jingyi Zheng, Xinlei He, Tianshuo Cong, Anyu Wang

**Abstract**: Jailbreak attacks aim to induce Large Language Models (LLMs) to generate harmful responses for forbidden instructions, presenting severe misuse threats to LLMs. Up to now, research into jailbreak attacks and defenses is emerging, however, there is (surprisingly) no consensus on how to evaluate whether a jailbreak attempt is successful. In other words, the methods to assess the harmfulness of an LLM's response are varied, such as manual annotation or prompting GPT-4 in specific ways. Each approach has its own set of strengths and weaknesses, impacting their alignment with human values, as well as the time and financial cost. This diversity in evaluation presents challenges for researchers in choosing suitable evaluation methods and conducting fair comparisons across different jailbreak attacks and defenses. In this paper, we conduct a comprehensive analysis of jailbreak evaluation methodologies, drawing from nearly ninety jailbreak research released between May 2023 and April 2024. Our study introduces a systematic taxonomy of jailbreak evaluators, offering in-depth insights into their strengths and weaknesses, along with the current status of their adaptation. Moreover, to facilitate subsequent research, we propose JailbreakEval, a user-friendly toolkit focusing on the evaluation of jailbreak attempts. It includes various well-known evaluators out-of-the-box, so that users can obtain evaluation results with only a single command. JailbreakEval also allows users to customize their own evaluation workflow in a unified framework with the ease of development and comparison. In summary, we regard JailbreakEval to be a catalyst that simplifies the evaluation process in jailbreak research and fosters an inclusive standard for jailbreak evaluation within the community.

摘要: 越狱攻击的目的是诱导大型语言模型(LLM)对禁用指令产生有害响应，给LLM带来严重的滥用威胁。到目前为止，关于越狱攻击和防御的研究正在兴起，然而，(令人惊讶的)对于如何评估越狱尝试是否成功，还没有达成共识。换句话说，评估LLM响应的危害性的方法是多种多样的，例如手动注释或以特定方式提示GPT-4。每种方法都有自己的长处和短处，影响它们与人类价值观的一致性，以及时间和财务成本。这种评估的多样性给研究人员带来了挑战，他们要选择合适的评估方法，并对不同的越狱攻击和防御进行公平的比较。本文从2023年5月至2024年4月发布的近90项越狱研究中，对越狱评估方法进行了全面的分析。我们的研究介绍了越狱评估员的系统分类，深入了解了他们的优势和劣势，以及他们适应的现状。此外，为了便于后续研究，我们提出了JailBreak Eval，这是一个用户友好的工具包，专注于评估越狱企图。它包括各种开箱即用的知名评估器，使用户只需一条命令即可获得评估结果。JailBreak Eval还允许用户在统一的框架中定制自己的评估工作流程，易于开发和比较。总而言之，我们认为越狱评估是一种催化剂，可以简化越狱研究的评估过程，并在社区内培养一个包容性的越狱评估标准。



## **38. A Survey of Backdoor Attacks and Defenses on Large Language Models: Implications for Security Measures**

大型语言模型后门攻击和防御的调查：对安全措施的影响 cs.CR

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.06852v2) [paper-pdf](http://arxiv.org/pdf/2406.06852v2)

**Authors**: Shuai Zhao, Meihuizi Jia, Zhongliang Guo, Leilei Gan, Jie Fu, Yichao Feng, Fengjun Pan, Luu Anh Tuan

**Abstract**: The large language models (LLMs), which bridge the gap between human language understanding and complex problem-solving, achieve state-of-the-art performance on several NLP tasks, particularly in few-shot and zero-shot settings. Despite the demonstrable efficacy of LMMs, due to constraints on computational resources, users have to engage with open-source language models or outsource the entire training process to third-party platforms. However, research has demonstrated that language models are susceptible to potential security vulnerabilities, particularly in backdoor attacks. Backdoor attacks are designed to introduce targeted vulnerabilities into language models by poisoning training samples or model weights, allowing attackers to manipulate model responses through malicious triggers. While existing surveys on backdoor attacks provide a comprehensive overview, they lack an in-depth examination of backdoor attacks specifically targeting LLMs. To bridge this gap and grasp the latest trends in the field, this paper presents a novel perspective on backdoor attacks for LLMs by focusing on fine-tuning methods. Specifically, we systematically classify backdoor attacks into three categories: full-parameter fine-tuning, parameter-efficient fine-tuning, and attacks without fine-tuning. Based on insights from a substantial review, we also discuss crucial issues for future research on backdoor attacks, such as further exploring attack algorithms that do not require fine-tuning, or developing more covert attack algorithms.

摘要: 大型语言模型(LLM)架起了人类语言理解和复杂问题解决之间的桥梁，在几个NLP任务上实现了最先进的性能，特别是在少镜头和零镜头的情况下。尽管LMM具有明显的功效，但由于计算资源的限制，用户不得不使用开放源码语言模型或将整个培训过程外包给第三方平台。然而，研究表明，语言模型容易受到潜在的安全漏洞的影响，特别是在后门攻击中。后门攻击旨在通过毒化训练样本或模型权重，将有针对性的漏洞引入语言模型，允许攻击者通过恶意触发器操纵模型响应。虽然现有的关于后门攻击的调查提供了全面的概述，但它们缺乏对专门针对LLM的后门攻击的深入检查。为了弥补这一差距，掌握该领域的最新趋势，本文提出了一种新的视角来研究针对LLMS的后门攻击，重点是微调方法。具体来说，我们系统地将后门攻击分为三类：全参数微调、参数高效微调和未微调攻击。在大量综述的基础上，我们还讨论了未来后门攻击研究的关键问题，如进一步探索不需要微调的攻击算法，或开发更隐蔽的攻击算法。



## **39. StructuralSleight: Automated Jailbreak Attacks on Large Language Models Utilizing Uncommon Text-Encoded Structure**

StructualSleight：利用不常见的文本编码结构对大型语言模型进行自动越狱攻击 cs.CL

12 pages, 4 figures

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.08754v1) [paper-pdf](http://arxiv.org/pdf/2406.08754v1)

**Authors**: Bangxin Li, Hengrui Xing, Chao Huang, Jin Qian, Huangqing Xiao, Linfeng Feng, Cong Tian

**Abstract**: Large Language Models (LLMs) are widely used in natural language processing but face the risk of jailbreak attacks that maliciously induce them to generate harmful content. Existing jailbreak attacks, including character-level and context-level attacks, mainly focus on the prompt of the plain text without specifically exploring the significant influence of its structure. In this paper, we focus on studying how prompt structure contributes to the jailbreak attack. We introduce a novel structure-level attack method based on tail structures that are rarely used during LLM training, which we refer to as Uncommon Text-Encoded Structure (UTES). We extensively study 12 UTESs templates and 6 obfuscation methods to build an effective automated jailbreak tool named StructuralSleight that contains three escalating attack strategies: Structural Attack, Structural and Character/Context Obfuscation Attack, and Fully Obfuscated Structural Attack. Extensive experiments on existing LLMs show that StructuralSleight significantly outperforms baseline methods. In particular, the attack success rate reaches 94.62\% on GPT-4o, which has not been addressed by state-of-the-art techniques.

摘要: 大语言模型在自然语言处理中被广泛使用，但面临着越狱攻击的风险，这些攻击会恶意诱导它们生成有害内容。现有的越狱攻击，包括字符级攻击和语境级攻击，主要集中在明文的提示上，没有具体探讨其结构的重大影响。本文主要研究提示结构在越狱攻击中的作用。提出了一种基于LLM训练中很少使用的尾部结构的结构级攻击方法，称为非公共文本编码结构(UTES)。我们深入研究了12个UTE模板和6种混淆方法，构建了一个有效的自动化越狱工具StructuralSleight，它包含三种逐步升级的攻击策略：结构攻击、结构和字符/上下文混淆攻击和完全混淆结构攻击。在现有LLMS上的大量实验表明，StructuralSleight的性能明显优于基线方法。特别是，在GPT-40上的攻击成功率达到了94.62\%，这是最新技术还没有解决的问题。



## **40. Ranking Manipulation for Conversational Search Engines**

对话式搜索引擎的排名操纵 cs.CL

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.03589v2) [paper-pdf](http://arxiv.org/pdf/2406.03589v2)

**Authors**: Samuel Pfrommer, Yatong Bai, Tanmay Gautam, Somayeh Sojoudi

**Abstract**: Major search engine providers are rapidly incorporating Large Language Model (LLM)-generated content in response to user queries. These conversational search engines operate by loading retrieved website text into the LLM context for summarization and interpretation. Recent research demonstrates that LLMs are highly vulnerable to jailbreaking and prompt injection attacks, which disrupt the safety and quality goals of LLMs using adversarial strings. This work investigates the impact of prompt injections on the ranking order of sources referenced by conversational search engines. To this end, we introduce a focused dataset of real-world consumer product websites and formalize conversational search ranking as an adversarial problem. Experimentally, we analyze conversational search rankings in the absence of adversarial injections and show that different LLMs vary significantly in prioritizing product name, document content, and context position. We then present a tree-of-attacks-based jailbreaking technique which reliably promotes low-ranked products. Importantly, these attacks transfer effectively to state-of-the-art conversational search engines such as perplexity.ai. Given the strong financial incentive for website owners to boost their search ranking, we argue that our problem formulation is of critical importance for future robustness work.

摘要: 各大搜索引擎提供商正在快速整合大型语言模型(LLM)生成的内容，以响应用户查询。这些对话式搜索引擎通过将检索到的网站文本加载到LLM上下文中进行操作以进行摘要和解释。最近的研究表明，LLM非常容易受到越狱和快速注入攻击，这些攻击使用敌意字符串破坏LLM的安全和质量目标。这项工作调查了提示注入对对话式搜索引擎引用的来源的排名顺序的影响。为此，我们引入了一个聚焦于真实世界消费产品网站的数据集，并将会话搜索排名形式化为一个对抗性问题。在实验上，我们分析了在没有对抗性注入的情况下的会话搜索排名，结果表明不同的LLM在产品名称、文档内容和上下文位置的优先顺序上存在显著差异。然后，我们提出了一种基于攻击树的越狱技术，该技术可靠地推广排名较低的产品。重要的是，这些攻击有效地转移到了最先进的会话搜索引擎，如Pplexity.ai。考虑到网站所有者有强大的经济动机来提高他们的搜索排名，我们认为我们的问题表达对于未来的稳健性工作至关重要。



## **41. RL-JACK: Reinforcement Learning-powered Black-box Jailbreaking Attack against LLMs**

RL-JACK：针对LLM的强化学习驱动的黑匣子越狱攻击 cs.CR

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.08725v1) [paper-pdf](http://arxiv.org/pdf/2406.08725v1)

**Authors**: Xuan Chen, Yuzhou Nie, Lu Yan, Yunshu Mao, Wenbo Guo, Xiangyu Zhang

**Abstract**: Modern large language model (LLM) developers typically conduct a safety alignment to prevent an LLM from generating unethical or harmful content. Recent studies have discovered that the safety alignment of LLMs can be bypassed by jailbreaking prompts. These prompts are designed to create specific conversation scenarios with a harmful question embedded. Querying an LLM with such prompts can mislead the model into responding to the harmful question. The stochastic and random nature of existing genetic methods largely limits the effectiveness and efficiency of state-of-the-art (SOTA) jailbreaking attacks. In this paper, we propose RL-JACK, a novel black-box jailbreaking attack powered by deep reinforcement learning (DRL). We formulate the generation of jailbreaking prompts as a search problem and design a novel RL approach to solve it. Our method includes a series of customized designs to enhance the RL agent's learning efficiency in the jailbreaking context. Notably, we devise an LLM-facilitated action space that enables diverse action variations while constraining the overall search space. We propose a novel reward function that provides meaningful dense rewards for the agent toward achieving successful jailbreaking. Through extensive evaluations, we demonstrate that RL-JACK is overall much more effective than existing jailbreaking attacks against six SOTA LLMs, including large open-source models and commercial models. We also show the RL-JACK's resiliency against three SOTA defenses and its transferability across different models. Finally, we validate the insensitivity of RL-JACK to the variations in key hyper-parameters.

摘要: 现代大型语言模型(LLM)开发人员通常会进行安全调整，以防止LLM生成不道德或有害的内容。最近的研究发现，越狱提示可以绕过LLMS的安全对准。这些提示旨在创建嵌入有害问题的特定对话场景。使用这样的提示查询LLM可能会误导模型响应有害的问题。现有遗传方法的随机性和随机性在很大程度上限制了最先进的越狱攻击(SOTA)的有效性和效率。本文提出了一种基于深度强化学习的新型黑盒越狱攻击算法RL-JACK。我们将越狱提示的生成描述为一个搜索问题，并设计了一种新的RL方法来解决这个问题。我们的方法包括一系列定制设计，以提高RL代理在越狱环境中的学习效率。值得注意的是，我们设计了一个LLM促进的动作空间，它可以在限制整体搜索空间的同时实现不同的动作变化。我们提出了一个新的奖励函数，为特工提供有意义的密集奖励，以实现成功的越狱。通过广泛的评估，我们证明了RL-JACK总体上比现有的针对六个Sota LLM的越狱攻击要有效得多，其中包括大型开源模型和商业模型。我们还展示了RL-Jack对三种Sota防御系统的弹性，以及它在不同型号之间的可转移性。最后，我们验证了RL-JACK对关键超参数变化的不敏感性。



## **42. Adversarial Evasion Attack Efficiency against Large Language Models**

针对大型语言模型的对抗规避攻击效率 cs.CL

9 pages, 1 table, 2 figures, DCAI 2024 conference

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2406.08050v1) [paper-pdf](http://arxiv.org/pdf/2406.08050v1)

**Authors**: João Vitorino, Eva Maia, Isabel Praça

**Abstract**: Large Language Models (LLMs) are valuable for text classification, but their vulnerabilities must not be disregarded. They lack robustness against adversarial examples, so it is pertinent to understand the impacts of different types of perturbations, and assess if those attacks could be replicated by common users with a small amount of perturbations and a small number of queries to a deployed LLM. This work presents an analysis of the effectiveness, efficiency, and practicality of three different types of adversarial attacks against five different LLMs in a sentiment classification task. The obtained results demonstrated the very distinct impacts of the word-level and character-level attacks. The word attacks were more effective, but the character and more constrained attacks were more practical and required a reduced number of perturbations and queries. These differences need to be considered during the development of adversarial defense strategies to train more robust LLMs for intelligent text classification applications.

摘要: 大型语言模型(LLM)对于文本分类很有价值，但其脆弱性不容忽视。它们对敌意示例缺乏健壮性，因此了解不同类型扰动的影响并评估这些攻击是否可以被普通用户复制，只需少量扰动和对已部署的LLM的少量查询。本文分析了三种不同类型的对抗性攻击在情感分类任务中对五种不同的LLM的有效性、效率和实用性。所获得的结果显示了词级攻击和字级攻击的非常明显的影响。单词攻击更有效，但字符和更受约束的攻击更实用，需要的干扰和查询次数更少。在开发对抗性防御策略以训练更健壮的LLM用于智能文本分类应用时，需要考虑这些差异。



## **43. Defending Large Language Models Against Jailbreaking Attacks Through Goal Prioritization**

通过目标优先级保护大型语言模型免受越狱攻击 cs.CL

ACL 2024 Main Conference

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2311.09096v2) [paper-pdf](http://arxiv.org/pdf/2311.09096v2)

**Authors**: Zhexin Zhang, Junxiao Yang, Pei Ke, Fei Mi, Hongning Wang, Minlie Huang

**Abstract**: While significant attention has been dedicated to exploiting weaknesses in LLMs through jailbreaking attacks, there remains a paucity of effort in defending against these attacks. We point out a pivotal factor contributing to the success of jailbreaks: the intrinsic conflict between the goals of being helpful and ensuring safety. Accordingly, we propose to integrate goal prioritization at both training and inference stages to counteract. Implementing goal prioritization during inference substantially diminishes the Attack Success Rate (ASR) of jailbreaking from 66.4% to 3.6% for ChatGPT. And integrating goal prioritization into model training reduces the ASR from 71.0% to 6.6% for Llama2-13B. Remarkably, even in scenarios where no jailbreaking samples are included during training, our approach slashes the ASR by half. Additionally, our findings reveal that while stronger LLMs face greater safety risks, they also possess a greater capacity to be steered towards defending against such attacks, both because of their stronger ability in instruction following. Our work thus contributes to the comprehension of jailbreaking attacks and defenses, and sheds light on the relationship between LLMs' capability and safety. Our code is available at \url{https://github.com/thu-coai/JailbreakDefense_GoalPriority}.

摘要: 虽然通过越狱攻击来利用LLMS的弱点已经引起了极大的关注，但在防御这些攻击方面仍然缺乏努力。我们指出了越狱成功的一个关键因素：提供帮助的目标与确保安全之间的内在冲突。因此，我们提出在训练和推理阶段整合目标优先级来抵消。在推理过程中实施目标优先级大大降低了ChatGPT越狱的攻击成功率(ASR)，从66.4%降至3.6%。将目标优先顺序整合到模型训练中，将Llama2-13B的ASR从71.0%降低到6.6%。值得注意的是，即使在训练期间没有包括越狱样本的情况下，我们的方法也将ASR削减了一半。此外，我们的研究结果显示，虽然较强的LLM面临更大的安全风险，但它们也具有更强的被引导来防御此类攻击的能力，这两者都是因为它们具有更强的指令遵循能力。因此，我们的工作有助于理解越狱攻击和防御，并阐明了LLMS的能力和安全之间的关系。我们的代码可以在\url{https://github.com/thu-coai/JailbreakDefense_GoalPriority}.上找到



## **44. Dataset and Lessons Learned from the 2024 SaTML LLM Capture-the-Flag Competition**

2024年SaTML LLM夺旗大赛的数据集和经验教训 cs.CR

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2406.07954v1) [paper-pdf](http://arxiv.org/pdf/2406.07954v1)

**Authors**: Edoardo Debenedetti, Javier Rando, Daniel Paleka, Silaghi Fineas Florin, Dragos Albastroiu, Niv Cohen, Yuval Lemberg, Reshmi Ghosh, Rui Wen, Ahmed Salem, Giovanni Cherubin, Santiago Zanella-Beguelin, Robin Schmid, Victor Klemm, Takahiro Miki, Chenhao Li, Stefan Kraft, Mario Fritz, Florian Tramèr, Sahar Abdelnabi, Lea Schönherr

**Abstract**: Large language model systems face important security risks from maliciously crafted messages that aim to overwrite the system's original instructions or leak private data. To study this problem, we organized a capture-the-flag competition at IEEE SaTML 2024, where the flag is a secret string in the LLM system prompt. The competition was organized in two phases. In the first phase, teams developed defenses to prevent the model from leaking the secret. During the second phase, teams were challenged to extract the secrets hidden for defenses proposed by the other teams. This report summarizes the main insights from the competition. Notably, we found that all defenses were bypassed at least once, highlighting the difficulty of designing a successful defense and the necessity for additional research to protect LLM systems. To foster future research in this direction, we compiled a dataset with over 137k multi-turn attack chats and open-sourced the platform.

摘要: 大型语言模型系统面临着恶意制作的消息的重要安全风险，这些消息旨在覆盖系统的原始指令或泄露私人数据。为了研究这个问题，我们在IEEE SaTML 2024上组织了一场捕获旗帜竞赛，其中旗帜是LLM系统提示符中的秘密字符串。比赛分两个阶段组织。在第一阶段，团队开发了防御措施以防止模型泄露秘密。在第二阶段，团队面临挑战，以提取其他团队提出的防御所隐藏的秘密。本报告总结了比赛的主要见解。值得注意的是，我们发现所有防御措施至少被绕过一次，这凸显了设计成功防御的难度以及进行额外研究以保护LLM系统的必要性。为了促进这一方向的未来研究，我们编制了一个包含超过137，000次多回合攻击聊天的数据集，并开源了该平台。



## **45. Visual-RolePlay: Universal Jailbreak Attack on MultiModal Large Language Models via Role-playing Image Character**

可视化角色扮演：通过角色扮演图像角色对多模式大型语言模型进行通用越狱攻击 cs.CR

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2405.20773v2) [paper-pdf](http://arxiv.org/pdf/2405.20773v2)

**Authors**: Siyuan Ma, Weidi Luo, Yu Wang, Xiaogeng Liu

**Abstract**: With the advent and widespread deployment of Multimodal Large Language Models (MLLMs), ensuring their safety has become increasingly critical. To achieve this objective, it requires us to proactively discover the vulnerability of MLLMs by exploring the attack methods. Thus, structure-based jailbreak attacks, where harmful semantic content is embedded within images, have been proposed to mislead the models. However, previous structure-based jailbreak methods mainly focus on transforming the format of malicious queries, such as converting harmful content into images through typography, which lacks sufficient jailbreak effectiveness and generalizability. To address these limitations, we first introduce the concept of "Role-play" into MLLM jailbreak attacks and propose a novel and effective method called Visual Role-play (VRP). Specifically, VRP leverages Large Language Models to generate detailed descriptions of high-risk characters and create corresponding images based on the descriptions. When paired with benign role-play instruction texts, these high-risk character images effectively mislead MLLMs into generating malicious responses by enacting characters with negative attributes. We further extend our VRP method into a universal setup to demonstrate its generalizability. Extensive experiments on popular benchmarks show that VRP outperforms the strongest baseline, Query relevant and FigStep, by an average Attack Success Rate (ASR) margin of 14.3% across all models.

摘要: 随着多通道大语言模型的出现和广泛应用，确保其安全性变得越来越重要。为了实现这一目标，需要我们通过探索攻击方法来主动发现MLLMS的脆弱性。因此，已经提出了基于结构的越狱攻击，其中有害的语义内容嵌入到图像中，以误导模型。然而，以往的基于结构的越狱方法主要集中在对恶意查询的格式进行转换，如通过排版将有害内容转换为图像，缺乏足够的越狱有效性和通用性。针对这些局限性，我们首先将“角色扮演”的概念引入到MLLM越狱攻击中，提出了一种新颖而有效的方法--视觉角色扮演(VRP)。具体地说，VRP利用大型语言模型来生成高危角色的详细描述，并基于这些描述创建相应的图像。当与良性的角色扮演指示文本配对时，这些高风险角色图像有效地误导MLLM，通过设定具有负面属性的角色来生成恶意响应。我们进一步将VRP方法扩展到一个通用的设置，以证明它的普适性。在流行基准上的广泛实验表明，VRP在所有模型上的平均攻击成功率(ASR)边际都比最强的基线、查询相关和FigStep高14.3%。



## **46. We Have a Package for You! A Comprehensive Analysis of Package Hallucinations by Code Generating LLMs**

我们为您准备了一个套餐！通过代码生成LLM综合分析包幻觉 cs.SE

18 pages, 8 figures, 7 tables

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2406.10279v1) [paper-pdf](http://arxiv.org/pdf/2406.10279v1)

**Authors**: Joseph Spracklen, Raveen Wijewickrama, A H M Nazmus Sakib, Anindya Maiti, Murtuza Jadliwala

**Abstract**: The reliance of popular programming languages such as Python and JavaScript on centralized package repositories and open-source software, combined with the emergence of code-generating Large Language Models (LLMs), has created a new type of threat to the software supply chain: package hallucinations. These hallucinations, which arise from fact-conflicting errors when generating code using LLMs, represent a novel form of package confusion attack that poses a critical threat to the integrity of the software supply chain. This paper conducts a rigorous and comprehensive evaluation of package hallucinations across different programming languages, settings, and parameters, exploring how different configurations of LLMs affect the likelihood of generating erroneous package recommendations and identifying the root causes of this phenomena. Using 16 different popular code generation models, across two programming languages and two unique prompt datasets, we collect 576,000 code samples which we analyze for package hallucinations. Our findings reveal that 19.7% of generated packages across all the tested LLMs are hallucinated, including a staggering 205,474 unique examples of hallucinated package names, further underscoring the severity and pervasiveness of this threat. We also implemented and evaluated mitigation strategies based on Retrieval Augmented Generation (RAG), self-detected feedback, and supervised fine-tuning. These techniques demonstrably reduced package hallucinations, with hallucination rates for one model dropping below 3%. While the mitigation efforts were effective in reducing hallucination rates, our study reveals that package hallucinations are a systemic and persistent phenomenon that pose a significant challenge for code generating LLMs.

摘要: 流行的编程语言，如Python和JavaScript对集中包库和开源软件的依赖，再加上代码生成大型语言模型(LLM)的出现，对软件供应链造成了一种新的威胁：包幻觉。这些幻觉是由使用LLMS生成代码时与事实冲突的错误引起的，代表了一种新形式的包混淆攻击，对软件供应链的完整性构成了严重威胁。本文对不同编程语言、不同设置和不同参数的套餐幻觉进行了严格而全面的评估，探讨了不同配置的LLM如何影响产生错误套餐推荐的可能性，并找出了这种现象的根本原因。使用16种不同的流行代码生成模型，跨越两种编程语言和两个独特的提示数据集，我们收集了576,000个代码样本，我们分析了程序包幻觉。我们的发现显示，在所有测试的LLM中，19.7%的生成包是幻觉的，其中包括惊人的205,474个唯一的幻觉包名称示例，进一步突显了这一威胁的严重性和普遍性。我们还实施并评估了基于检索增强生成(RAG)、自我检测反馈和监督微调的缓解策略。这些技术明显减少了包装幻觉，一款车型的幻觉率降至3%以下。虽然缓解努力在降低幻觉率方面是有效的，但我们的研究表明，程序包幻觉是一种系统性和持续性的现象，对代码生成LLM构成了重大挑战。



## **47. Defending Against Alignment-Breaking Attacks via Robustly Aligned LLM**

通过强大的对齐LLM防御破坏对齐的攻击 cs.CL

19 Pages, 5 Figures, 8 Tables. Accepted by ACL 2024

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2309.14348v3) [paper-pdf](http://arxiv.org/pdf/2309.14348v3)

**Authors**: Bochuan Cao, Yuanpu Cao, Lu Lin, Jinghui Chen

**Abstract**: Recently, Large Language Models (LLMs) have made significant advancements and are now widely used across various domains. Unfortunately, there has been a rising concern that LLMs can be misused to generate harmful or malicious content. Though a line of research has focused on aligning LLMs with human values and preventing them from producing inappropriate content, such alignments are usually vulnerable and can be bypassed by alignment-breaking attacks via adversarially optimized or handcrafted jailbreaking prompts. In this work, we introduce a Robustly Aligned LLM (RA-LLM) to defend against potential alignment-breaking attacks. RA-LLM can be directly constructed upon an existing aligned LLM with a robust alignment checking function, without requiring any expensive retraining or fine-tuning process of the original LLM. Furthermore, we also provide a theoretical analysis for RA-LLM to verify its effectiveness in defending against alignment-breaking attacks. Through real-world experiments on open-source large language models, we demonstrate that RA-LLM can successfully defend against both state-of-the-art adversarial prompts and popular handcrafted jailbreaking prompts by reducing their attack success rates from nearly 100% to around 10% or less.

摘要: 近年来，大型语言模型(LLM)取得了长足的进步，现已广泛应用于各个领域。不幸的是，人们越来越担心LLMS可能被滥用来生成有害或恶意的内容。尽管有一系列研究专注于将LLM与人类价值观保持一致，并防止它们产生不适当的内容，但这种调整通常是脆弱的，可以通过恶意优化或手工制作的越狱提示被破坏顺序的攻击绕过。在这项工作中，我们引入了一种鲁棒对齐LLM(RA-LLM)来防御潜在的对齐破坏攻击。RA-LLM可以直接构建在现有的对准LLM上，具有健壮的对准检查功能，而不需要对原始LLM进行任何昂贵的再培训或微调过程。此外，我们还对RA-LLM进行了理论分析，以验证其在抵抗对齐破坏攻击方面的有效性。通过在开源大型语言模型上的真实世界实验，我们证明了RA-LLM能够成功地防御最新的敌意提示和流行的手工越狱提示，将攻击成功率从近100%降低到10%左右或更低。



## **48. Knowledge Return Oriented Prompting (KROP)**

知识回报导向预算（Kopp） cs.CR

**SubmitDate**: 2024-06-11    [abs](http://arxiv.org/abs/2406.11880v1) [paper-pdf](http://arxiv.org/pdf/2406.11880v1)

**Authors**: Jason Martin, Kenneth Yeung

**Abstract**: Many Large Language Models (LLMs) and LLM-powered apps deployed today use some form of prompt filter or alignment to protect their integrity. However, these measures aren't foolproof. This paper introduces KROP, a prompt injection technique capable of obfuscating prompt injection attacks, rendering them virtually undetectable to most of these security measures.

摘要: 当今部署的许多大型语言模型（LLM）和LLM支持的应用程序都使用某种形式的提示过滤器或对齐来保护其完整性。然而，这些措施并非万无一失。本文介绍了Kopp，这是一种即时注入技术，能够混淆即时注入攻击，使它们几乎无法被大多数安全措施检测到。



## **49. SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks**

SmothLLM：保护大型语言模型免受越狱攻击 cs.LG

**SubmitDate**: 2024-06-11    [abs](http://arxiv.org/abs/2310.03684v4) [paper-pdf](http://arxiv.org/pdf/2310.03684v4)

**Authors**: Alexander Robey, Eric Wong, Hamed Hassani, George J. Pappas

**Abstract**: Despite efforts to align large language models (LLMs) with human intentions, widely-used LLMs such as GPT, Llama, and Claude are susceptible to jailbreaking attacks, wherein an adversary fools a targeted LLM into generating objectionable content. To address this vulnerability, we propose SmoothLLM, the first algorithm designed to mitigate jailbreaking attacks. Based on our finding that adversarially-generated prompts are brittle to character-level changes, our defense randomly perturbs multiple copies of a given input prompt, and then aggregates the corresponding predictions to detect adversarial inputs. Across a range of popular LLMs, SmoothLLM sets the state-of-the-art for robustness against the GCG, PAIR, RandomSearch, and AmpleGCG jailbreaks. SmoothLLM is also resistant against adaptive GCG attacks, exhibits a small, though non-negligible trade-off between robustness and nominal performance, and is compatible with any LLM. Our code is publicly available at \url{https://github.com/arobey1/smooth-llm}.

摘要: 尽管努力使大型语言模型(LLM)与人的意图保持一致，但GPT、Llama和Claude等广泛使用的LLM容易受到越狱攻击，即对手欺骗目标LLM生成令人反感的内容。为了解决这一漏洞，我们提出了SmoothLLM，这是第一个旨在缓解越狱攻击的算法。基于我们的发现，对抗性生成的提示对字符级别的变化很脆弱，我们的防御随机扰动给定输入提示的多个副本，然后聚合相应的预测来检测对抗性输入。在一系列流行的LLM中，SmoothLLM针对GCG、Pair、RandomSearch和AmpleGCG越狱设置了最先进的健壮性。SmoothLLM还抵抗自适应GCG攻击，在稳健性和标称性能之间表现出一种虽小但不可忽略的折衷，并与任何LLM兼容。我们的代码在\url{https://github.com/arobey1/smooth-llm}.}上公开提供



## **50. Merging Improves Self-Critique Against Jailbreak Attacks**

合并提高了对越狱袭击的自我批评 cs.CL

**SubmitDate**: 2024-06-11    [abs](http://arxiv.org/abs/2406.07188v1) [paper-pdf](http://arxiv.org/pdf/2406.07188v1)

**Authors**: Victor Gallego

**Abstract**: The robustness of large language models (LLMs) against adversarial manipulations, such as jailbreak attacks, remains a significant challenge. In this work, we propose an approach that enhances the self-critique capability of the LLM and further fine-tunes it over sanitized synthetic data. This is done with the addition of an external critic model that can be merged with the original, thus bolstering self-critique capabilities and improving the robustness of the LLMs response to adversarial prompts. Our results demonstrate that the combination of merging and self-critique can reduce the attack success rate of adversaries significantly, thus offering a promising defense mechanism against jailbreak attacks. Code, data and models released at https://github.com/vicgalle/merging-self-critique-jailbreaks .

摘要: 大型语言模型（LLM）对越狱攻击等对抗性操纵的稳健性仍然是一个重大挑战。在这项工作中，我们提出了一种增强LLM自我批评能力的方法，并根据净化的合成数据进一步对其进行微调。这是通过添加一个可以与原始模型合并的外部批评者模型来实现的，从而增强自我批评能力并提高LLM对对抗提示反应的稳健性。我们的结果表明，合并和自我批评的结合可以显着降低对手的攻击成功率，从而提供一种有希望的针对越狱攻击的防御机制。代码、数据和模型在https://github.com/vicgalle/merging-self-critique-jailbreaks上发布。



