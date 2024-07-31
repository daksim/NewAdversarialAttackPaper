# Latest Large Language Model Attack Papers
**update at 2024-07-31 16:19:30**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Breaking Agents: Compromising Autonomous LLM Agents Through Malfunction Amplification**

突破剂：通过故障放大损害自主LLM代理 cs.CR

**SubmitDate**: 2024-07-30    [abs](http://arxiv.org/abs/2407.20859v1) [paper-pdf](http://arxiv.org/pdf/2407.20859v1)

**Authors**: Boyang Zhang, Yicong Tan, Yun Shen, Ahmed Salem, Michael Backes, Savvas Zannettou, Yang Zhang

**Abstract**: Recently, autonomous agents built on large language models (LLMs) have experienced significant development and are being deployed in real-world applications. These agents can extend the base LLM's capabilities in multiple ways. For example, a well-built agent using GPT-3.5-Turbo as its core can outperform the more advanced GPT-4 model by leveraging external components. More importantly, the usage of tools enables these systems to perform actions in the real world, moving from merely generating text to actively interacting with their environment. Given the agents' practical applications and their ability to execute consequential actions, it is crucial to assess potential vulnerabilities. Such autonomous systems can cause more severe damage than a standalone language model if compromised. While some existing research has explored harmful actions by LLM agents, our study approaches the vulnerability from a different perspective. We introduce a new type of attack that causes malfunctions by misleading the agent into executing repetitive or irrelevant actions. We conduct comprehensive evaluations using various attack methods, surfaces, and properties to pinpoint areas of susceptibility. Our experiments reveal that these attacks can induce failure rates exceeding 80\% in multiple scenarios. Through attacks on implemented and deployable agents in multi-agent scenarios, we accentuate the realistic risks associated with these vulnerabilities. To mitigate such attacks, we propose self-examination detection methods. However, our findings indicate these attacks are difficult to detect effectively using LLMs alone, highlighting the substantial risks associated with this vulnerability.

摘要: 近年来，基于大型语言模型(LLM)的自治代理经历了长足的发展，并正被部署在现实世界的应用中。这些代理可以通过多种方式扩展基本LLM的功能。例如，使用GPT-3.5-Turbo作为核心的构建良好的代理可以通过利用外部组件来超越更高级的GPT-4型号。更重要的是，工具的使用使这些系统能够在现实世界中执行操作，从仅仅生成文本转移到与其环境主动交互。考虑到代理的实际应用及其执行相应操作的能力，评估潜在的漏洞至关重要。这种自主系统如果受到攻击，可能会造成比独立语言模型更严重的破坏。虽然现有的一些研究已经探索了LLM代理的有害行为，但我们的研究从不同的角度探讨了漏洞。我们引入了一种新型的攻击，通过误导代理执行重复或不相关的操作来导致故障。我们使用各种攻击方法、表面和属性进行全面评估，以确定易受攻击的区域。我们的实验表明，这些攻击在多个场景下都可以导致超过80%的失败率。通过对多代理场景中已实现和可部署代理的攻击，我们强调了与这些漏洞相关的现实风险。为了缓解这类攻击，我们提出了自检检测方法。然而，我们的发现表明，仅使用LLMS很难有效地检测到这些攻击，这突显了与此漏洞相关的重大风险。



## **2. From ML to LLM: Evaluating the Robustness of Phishing Webpage Detection Models against Adversarial Attacks**

从ML到LLM：评估网络钓鱼网页检测模型对抗对抗攻击的稳健性 cs.CR

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2407.20361v1) [paper-pdf](http://arxiv.org/pdf/2407.20361v1)

**Authors**: Aditya Kulkarni, Vivek Balachandran, Dinil Mon Divakaran, Tamal Das

**Abstract**: Phishing attacks attempt to deceive users into stealing sensitive information, posing a significant cybersecurity threat. Advances in machine learning (ML) and deep learning (DL) have led to the development of numerous phishing webpage detection solutions, but these models remain vulnerable to adversarial attacks. Evaluating their robustness against adversarial phishing webpages is essential. Existing tools contain datasets of pre-designed phishing webpages for a limited number of brands, and lack diversity in phishing features.   To address these challenges, we develop PhishOracle, a tool that generates adversarial phishing webpages by embedding diverse phishing features into legitimate webpages. We evaluate the robustness of two existing models, Stack model and Phishpedia, in classifying PhishOracle-generated adversarial phishing webpages. Additionally, we study a commercial large language model, Gemini Pro Vision, in the context of adversarial attacks. We conduct a user study to determine whether PhishOracle-generated adversarial phishing webpages deceive users. Our findings reveal that many PhishOracle-generated phishing webpages evade current phishing webpage detection models and deceive users, but Gemini Pro Vision is robust to the attack. We also develop the PhishOracle web app, allowing users to input a legitimate URL, select relevant phishing features and generate a corresponding phishing webpage. All resources are publicly available on GitHub.

摘要: 网络钓鱼攻击试图欺骗用户窃取敏感信息，构成重大的网络安全威胁。机器学习(ML)和深度学习(DL)的进步导致了许多钓鱼网页检测解决方案的发展，但这些模型仍然容易受到对手攻击。评估它们对敌意网络钓鱼网页的健壮性是至关重要的。现有工具包含为有限数量的品牌预先设计的钓鱼网页的数据集，并且在钓鱼功能方面缺乏多样性。为了应对这些挑战，我们开发了PhishOracle，这是一个通过在合法网页中嵌入不同的钓鱼功能来生成敌意钓鱼网页的工具。我们评估了现有的两种模型Stack模型和Phishpedia模型对PhishOracle生成的敌意钓鱼网页进行分类的稳健性。此外，我们研究了一个商业大型语言模型，Gemini Pro Vision，在对抗性攻击的背景下。我们进行了一项用户研究，以确定PhishOracle生成的敌意钓鱼网页是否欺骗了用户。我们的研究结果显示，许多PhishOracle生成的钓鱼网页逃避了当前的钓鱼网页检测模型并欺骗用户，但Gemini Pro Vision对攻击具有健壮性。我们还开发了PhishOracle Web应用程序，允许用户输入合法的URL，选择相关的网络钓鱼功能并生成相应的网络钓鱼网页。所有资源都在GitHub上公开提供。



## **3. Personalized Steering of Large Language Models: Versatile Steering Vectors Through Bi-directional Preference Optimization**

大型语言模型的个性化引导：通过双向偏好优化的多功能引导载体 cs.CL

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2406.00045v2) [paper-pdf](http://arxiv.org/pdf/2406.00045v2)

**Authors**: Yuanpu Cao, Tianrong Zhang, Bochuan Cao, Ziyi Yin, Lu Lin, Fenglong Ma, Jinghui Chen

**Abstract**: Researchers have been studying approaches to steer the behavior of Large Language Models (LLMs) and build personalized LLMs tailored for various applications. While fine-tuning seems to be a direct solution, it requires substantial computational resources and may significantly affect the utility of the original LLM. Recent endeavors have introduced more lightweight strategies, focusing on extracting "steering vectors" to guide the model's output toward desired behaviors by adjusting activations within specific layers of the LLM's transformer architecture. However, such steering vectors are directly extracted from the activations of human preference data and thus often lead to suboptimal results and occasional failures, especially in alignment-related scenarios. This work proposes an innovative approach that could produce more effective steering vectors through bi-directional preference optimization. Our method is designed to allow steering vectors to directly influence the generation probability of contrastive human preference data pairs, thereby offering a more precise representation of the target behavior. By carefully adjusting the direction and magnitude of the steering vector, we enabled personalized control over the desired behavior across a spectrum of intensities. Extensive experimentation across various open-ended generation tasks, particularly focusing on steering AI personas, has validated the efficacy of our approach. Moreover, we comprehensively investigate critical alignment-concerning scenarios, such as managing truthfulness, mitigating hallucination, and addressing jailbreaking attacks. Remarkably, our method can still demonstrate outstanding steering effectiveness across these scenarios. Furthermore, we showcase the transferability of our steering vectors across different models/LoRAs and highlight the synergistic benefits of applying multiple vectors simultaneously.

摘要: 研究人员一直在研究如何控制大语言模型(LLM)的行为，并为各种应用定制个性化的LLM。虽然微调似乎是一种直接的解决方案，但它需要大量的计算资源，并且可能会显著影响原始LLM的效用。最近的努力引入了更轻量级的策略，专注于提取“导向向量”，通过调整LLM的变压器体系结构的特定层中的激活来引导模型的输出达到所需的行为。然而，这样的导向向量直接从人类偏好数据的激活中提取，因此经常导致次优结果和偶尔的失败，特别是在与比对相关的场景中。本文提出了一种创新的方法，通过双向偏好优化产生更有效的导向向量。我们的方法旨在允许导向向量直接影响对比人类偏好数据对的产生概率，从而提供目标行为的更精确的表示。通过仔细调整导向向量的方向和大小，我们实现了对不同强度范围内所需行为的个性化控制。在各种开放式生成任务中的广泛实验，特别是专注于指导人工智能角色，已经验证了我们方法的有效性。此外，我们全面调查与关键一致性相关的场景，如管理真实性、减轻幻觉和解决越狱攻击。值得注意的是，我们的方法仍然可以在这些场景中展示出出色的转向效率。此外，我们展示了我们的转向矢量在不同模型/LORA之间的可转移性，并强调了同时应用多个矢量的协同好处。



## **4. Can Editing LLMs Inject Harm?**

编辑LLM会造成伤害吗？ cs.CL

The first two authors contributed equally. 9 pages for main paper, 36  pages including appendix. The code, results, dataset for this paper and more  resources are on the project website: https://llm-editing.github.io

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2407.20224v1) [paper-pdf](http://arxiv.org/pdf/2407.20224v1)

**Authors**: Canyu Chen, Baixiang Huang, Zekun Li, Zhaorun Chen, Shiyang Lai, Xiongxiao Xu, Jia-Chen Gu, Jindong Gu, Huaxiu Yao, Chaowei Xiao, Xifeng Yan, William Yang Wang, Philip Torr, Dawn Song, Kai Shu

**Abstract**: Knowledge editing techniques have been increasingly adopted to efficiently correct the false or outdated knowledge in Large Language Models (LLMs), due to the high cost of retraining from scratch. Meanwhile, one critical but under-explored question is: can knowledge editing be used to inject harm into LLMs? In this paper, we propose to reformulate knowledge editing as a new type of safety threat for LLMs, namely Editing Attack, and conduct a systematic investigation with a newly constructed dataset EditAttack. Specifically, we focus on two typical safety risks of Editing Attack including Misinformation Injection and Bias Injection. For the risk of misinformation injection, we first categorize it into commonsense misinformation injection and long-tail misinformation injection. Then, we find that editing attacks can inject both types of misinformation into LLMs, and the effectiveness is particularly high for commonsense misinformation injection. For the risk of bias injection, we discover that not only can biased sentences be injected into LLMs with high effectiveness, but also one single biased sentence injection can cause a high bias increase in general outputs of LLMs, which are even highly irrelevant to the injected sentence, indicating a catastrophic impact on the overall fairness of LLMs. Then, we further illustrate the high stealthiness of editing attacks, measured by their impact on the general knowledge and reasoning capacities of LLMs, and show the hardness of defending editing attacks with empirical evidence. Our discoveries demonstrate the emerging misuse risks of knowledge editing techniques on compromising the safety alignment of LLMs.

摘要: 由于从头开始再培训的成本很高，知识编辑技术越来越多地被用来有效地纠正大型语言模型(LLMS)中的错误或过时知识。与此同时，一个关键但未被探讨的问题是：知识编辑能否被用来向低收入国家注入危害？在本文中，我们将知识编辑重新定义为一种新的安全威胁，即编辑攻击，并使用新构建的数据集EditAttack进行了系统的研究。具体地说，我们重点研究了编辑攻击的两个典型的安全风险，包括错误信息注入和偏见注入。对于错误信息注入的风险，我们首先将其分为常识性错误信息注入和长尾错误信息注入。然后，我们发现编辑攻击可以将这两种类型的错误信息注入到LLMS中，其中常识性错误信息注入的有效性尤其高。对于偏向注入的风险，我们发现，偏向句不仅可以被高效地注入到LLMS中，而且一次偏向句注入会导致LLMS的总体输出出现较高的偏向增加，甚至与注入的句子高度无关，这对LLMS的整体公平性造成了灾难性的影响。然后，我们进一步说明了编辑攻击的高度隐蔽性，通过它们对LLM的常识和推理能力的影响来衡量它们，并用经验证据说明了防御编辑攻击的难度。我们的发现表明，知识编辑技术在损害LLMS的安全一致性方面存在新的误用风险。



## **5. Large Language Models as Carriers of Hidden Messages**

大型语言模型作为隐藏消息的载体 cs.CL

Work in progress. Code is available at  https://github.com/j-hoscilowic/zurek-stegano

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2406.02481v2) [paper-pdf](http://arxiv.org/pdf/2406.02481v2)

**Authors**: Jakub Hoscilowicz, Pawel Popiolek, Jan Rudkowski, Jedrzej Bieniasz, Artur Janicki

**Abstract**: With the help of simple fine-tuning, one can artificially embed hidden text into large language models (LLMs). This text is revealed only when triggered by a specific query to the LLM. Two primary applications are LLM fingerprinting and steganography. In the context of LLM fingerprinting, a unique text identifier (fingerprint) is embedded within the model to verify licensing compliance. In the context of steganography, the LLM serves as a carrier for hidden messages that can be disclosed through a chosen trigger question.   Our work demonstrates that embedding hidden text in the LLM via fine-tuning, though seemingly secure due to the vast number of potential triggers (any sequence of characters or tokens could serve as a trigger), is susceptible to extraction through analysis of the LLM's output decoding process. We propose an extraction attack called Unconditional Token Forcing (UTF). It is premised on the hypothesis that iteratively feeding each token from the LLM's vocabulary into the model should reveal output sequences with abnormally high token probabilities, indicating potential hidden text candidates. We also present a defense method to hide text in such a way that it is resistant to both UTF and attacks based on sampling decoding methods, which we named Unconditional Token Forcing Confusion (UTFC). To the best of our knowledge, there is no attack method that can extract text hidden with UTFC. UTFC has both benign applications (improving LLM fingerprinting) and malign applications (using LLMs to create covert communication channels). Code is available at github.com/j-hoscilowic/zurek-stegano

摘要: 在简单微调的帮助下，人们可以人为地将隐藏文本嵌入到大型语言模型(LLM)中。只有在对LLM的特定查询触发时，才会显示此文本。两个主要应用是LLM指纹识别和隐写。在LLM指纹识别的上下文中，唯一的文本识别符(指纹)被嵌入到模型中，以验证许可合规性。在隐写术的背景下，LLM充当了隐藏消息的载体，这些隐藏消息可以通过选择的触发问题来泄露。我们的工作表明，通过微调将隐藏文本嵌入到LLM中，尽管由于潜在触发器(任何字符或标记序列都可以作为触发器)的数量巨大而看起来是安全的，但通过分析LLM的输出解码过程，它容易被提取。我们提出了一种称为无条件令牌强迫(UTF)的提取攻击。它的前提是假设迭代地将LLM词汇中的每个标记输入到模型中，应该会揭示出具有异常高的标记概率的输出序列，这表明潜在的隐藏文本候选。我们还提出了一种基于抽样解码的文本隐藏方法，称为无条件令牌强制混淆(UTFC)，使其能够同时抵抗UTF和攻击。就我们所知，没有一种攻击方法可以提取使用UTFC隐藏的文本。UTFC既有良性应用程序(改进LLM指纹识别)，也有恶意应用程序(使用LLM创建秘密通信通道)。代码可在githorb.com/j-hvocilowic/zurek-stegano上找到



## **6. Detecting and Understanding Vulnerabilities in Language Models via Mechanistic Interpretability**

通过机械解释性检测和理解语言模型中的漏洞 cs.LG

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2407.19842v1) [paper-pdf](http://arxiv.org/pdf/2407.19842v1)

**Authors**: Jorge García-Carrasco, Alejandro Maté, Juan Trujillo

**Abstract**: Large Language Models (LLMs), characterized by being trained on broad amounts of data in a self-supervised manner, have shown impressive performance across a wide range of tasks. Indeed, their generative abilities have aroused interest on the application of LLMs across a wide range of contexts. However, neural networks in general, and LLMs in particular, are known to be vulnerable to adversarial attacks, where an imperceptible change to the input can mislead the output of the model. This is a serious concern that impedes the use of LLMs on high-stakes applications, such as healthcare, where a wrong prediction can imply serious consequences. Even though there are many efforts on making LLMs more robust to adversarial attacks, there are almost no works that study \emph{how} and \emph{where} these vulnerabilities that make LLMs prone to adversarial attacks happen. Motivated by these facts, we explore how to localize and understand vulnerabilities, and propose a method, based on Mechanistic Interpretability (MI) techniques, to guide this process. Specifically, this method enables us to detect vulnerabilities related to a concrete task by (i) obtaining the subset of the model that is responsible for that task, (ii) generating adversarial samples for that task, and (iii) using MI techniques together with the previous samples to discover and understand the possible vulnerabilities. We showcase our method on a pretrained GPT-2 Small model carrying out the task of predicting 3-letter acronyms to demonstrate its effectiveness on locating and understanding concrete vulnerabilities of the model.

摘要: 大型语言模型(LLM)的特点是以自我监督的方式对大量数据进行培训，在各种任务中表现出令人印象深刻的表现。事实上，他们的生成能力已经引起了人们对LLMS在广泛背景下的应用的兴趣。然而，一般的神经网络，特别是LLM，都很容易受到敌意攻击，在这种攻击中，输入的不可察觉的变化可能会误导模型的输出。这是一个严重的担忧，阻碍了低成本管理在高风险应用中的使用，例如医疗保健，在这些应用中，错误的预测可能意味着严重的后果。尽管已经有许多努力使LLM对对手攻击更健壮，但几乎没有研究这些使LLM容易受到对手攻击的漏洞的著作。在这些事实的推动下，我们探索了如何定位和理解漏洞，并提出了一种基于机械可解释性(MI)技术的方法来指导这一过程。具体地说，这种方法使我们能够通过(I)获得负责该任务的模型的子集，(Ii)为该任务生成对抗性样本，以及(Iii)结合使用MI技术和先前的样本来发现和理解可能的漏洞来检测与该具体任务相关的漏洞。我们在一个预先训练的GPT-2小模型上展示了我们的方法，该模型执行了预测3字母首字母缩写的任务，以演示其在定位和理解模型的具体漏洞方面的有效性。



## **7. Fine-Tuning, Quantization, and LLMs: Navigating Unintended Outcomes**

微调、量化和LLM：应对意外结果 cs.CR

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2404.04392v2) [paper-pdf](http://arxiv.org/pdf/2404.04392v2)

**Authors**: Divyanshu Kumar, Anurakt Kumar, Sahil Agarwal, Prashanth Harshangi

**Abstract**: Large Language Models (LLMs) have become very popular and are used in many domains, such as chatbots, auto-task completion agents, and much more. However, LLMs suffer from many safety vulnerabilities, which can be exploited using different types of attacks, such as jailbreaking, prompt injection attacks, and privacy leakage attacks. These attacks can disrupt the working of the LLMs and make powerful LLM systems generate malicious or unethical content, take malicious actions, or leak confidential information by bypassing the security filters and taking advantage of their access. Foundational LLMs undergo alignment training, which includes safety training. This helps the model learn how to generate outputs that are ethical and aligned with human responses. Further, to make the models even safer, guardrails are added to filter the inputs received and the output generated by the model. These foundational LLMs are subjected to fine-tuning, quantization, or alteration of guardrails to use these models for specialized tasks or to use them in a resource-constrained environment. So, understanding the impact of modifications such as fine-tuning, quantization, and guardrails on the safety of LLM becomes an important question. Understanding and mitigating the consequences will help build reliable systems and effective strategies to make LLMs more secure. In this study, we tested foundational models like Mistral, Llama, MosaicML, and their finetuned versions. These comprehensive evaluations show that fine-tuning increases jailbreak attack success rates (ASR), quantization has a variable impact on the ASR, and guardrails can help significantly improve jailbreak resistance.

摘要: 大型语言模型(LLM)已经变得非常流行，并在许多领域中使用，例如聊天机器人、自动任务完成代理等。然而，LLMS存在许多安全漏洞，可以使用不同类型的攻击来利用这些漏洞，如越狱、即时注入攻击和隐私泄露攻击。这些攻击可以扰乱LLMS的工作，并使功能强大的LLM系统通过绕过安全过滤器并利用其访问权限来生成恶意或不道德的内容、采取恶意操作或泄露机密信息。基础LLM要接受对准培训，其中包括安全培训。这有助于模型学习如何生成符合道德并与人类反应保持一致的输出。此外，为了使模型更加安全，还增加了护栏，以过滤模型接收的输入和生成的输出。这些基本的LLM需要进行微调、量化或改变护栏，以便将这些模型用于专门的任务或在资源受限的环境中使用它们。因此，了解微调、量化和护栏等修改对LLM安全性的影响成为一个重要的问题。了解并减轻后果将有助于建立可靠的系统和有效的战略，使LLMS更加安全。在这项研究中，我们测试了Mistral、Llama、MosaicML等基本模型及其精调版本。这些综合评估表明，微调可以提高越狱攻击成功率(ASR)，量化对ASR的影响是可变的，护栏可以帮助显著提高越狱抗性。



## **8. Learning on Graphs with Large Language Models(LLMs): A Deep Dive into Model Robustness**

使用大型语言模型（LLM）学习图形：深入研究模型稳健性 cs.LG

**SubmitDate**: 2024-07-28    [abs](http://arxiv.org/abs/2407.12068v2) [paper-pdf](http://arxiv.org/pdf/2407.12068v2)

**Authors**: Kai Guo, Zewen Liu, Zhikai Chen, Hongzhi Wen, Wei Jin, Jiliang Tang, Yi Chang

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across various natural language processing tasks. Recently, several LLMs-based pipelines have been developed to enhance learning on graphs with text attributes, showcasing promising performance. However, graphs are well-known to be susceptible to adversarial attacks and it remains unclear whether LLMs exhibit robustness in learning on graphs. To address this gap, our work aims to explore the potential of LLMs in the context of adversarial attacks on graphs. Specifically, we investigate the robustness against graph structural and textual perturbations in terms of two dimensions: LLMs-as-Enhancers and LLMs-as-Predictors. Through extensive experiments, we find that, compared to shallow models, both LLMs-as-Enhancers and LLMs-as-Predictors offer superior robustness against structural and textual attacks.Based on these findings, we carried out additional analyses to investigate the underlying causes. Furthermore, we have made our benchmark library openly available to facilitate quick and fair evaluations, and to encourage ongoing innovative research in this field.

摘要: 大型语言模型(LLM)在各种自然语言处理任务中表现出了显著的性能。最近，已经开发了几个基于LLMS的管道来增强对具有文本属性的图形的学习，展示了良好的性能。然而，众所周知，图是容易受到敌意攻击的，而且目前还不清楚LLM是否表现出关于图的学习的健壮性。为了解决这一差距，我们的工作旨在探索LLMS在对抗性攻击图的背景下的潜力。具体地说，我们从两个维度考察了对图结构和文本扰动的稳健性：作为增强器的LLMS和作为预测者的LLMS。通过广泛的实验，我们发现，与浅层模型相比，LLMS-as-Enhancer和LLMS-as-Predicator对结构和文本攻击都具有更好的稳健性。基于这些发现，我们进行了额外的分析，以探讨潜在的原因。此外，我们开放了我们的基准图书馆，以促进快速和公平的评估，并鼓励这一领域正在进行的创新研究。



## **9. Evolving Diverse Red-team Language Models in Multi-round Multi-agent Games**

多轮多智能体游戏中进化多样化的红队语言模型 cs.CL

**SubmitDate**: 2024-07-28    [abs](http://arxiv.org/abs/2310.00322v5) [paper-pdf](http://arxiv.org/pdf/2310.00322v5)

**Authors**: Chengdong Ma, Ziran Yang, Hai Ci, Jun Gao, Minquan Gao, Xuehai Pan, Yaodong Yang

**Abstract**: The primary challenge in deploying Large Language Model (LLM) is ensuring its harmlessness. Red team can identify vulnerabilities by attacking LLM to attain safety. However, current efforts heavily rely on single-round prompt designs and unilateral red team optimizations against fixed blue teams. These static approaches lead to significant reductions in generation diversity, known as the mode collapse, which makes it difficult to discover the potential risks in the increasingly complex human-LLM interactions. Here we introduce dynamic Red Team Game (RTG) to comprehensively analyze the multi-round offensive and defensive interactions between red team and blue team. Furthermore, we develop a Gamified Red Team Solver (GRTS) with diversity measures to mitigate mode collapse and theoretically guarantee the convergence of approximate Nash equilibrium which results in better strategies for both teams. Empirical results demonstrate that GRTS explore diverse and implicit attacks to adaptively exploit various LLMs, surpassing the constraints of specific modes. Insightfully, the geometrical structure we unveil of the red team task aligns with the spinning top hypothesis, confirming the necessity of constructing a diverse LLM population as a promising proxy for heterogeneous human expert red-teamers. This paves the way for scalable toxicity detection and safe alignment for LLMs.

摘要: 部署大型语言模型(LLM)的主要挑战是确保其无害。红队可以通过攻击LLM来识别漏洞，以获得安全。然而，目前的努力在很大程度上依赖于单轮提示设计和单边红队对固定蓝队的优化。这些静态的方法导致世代多样性的显著减少，称为模式崩溃，这使得在日益复杂的人与LLM相互作用中发现潜在风险变得困难。本文引入动态红队博弈(RTG)来全面分析红蓝双方多轮攻防互动。此外，我们开发了一个带有多样性措施的Gamamized Red Team Solver(GRTS)来缓解模式崩溃，并在理论上保证了近似Nash均衡的收敛，从而为两个团队带来了更好的策略。实验结果表明，GRTS能够探索各种隐式攻击，自适应地利用各种LLM，超越了特定模式的限制。有洞察力地，我们揭示的红色团队任务的几何结构与旋转顶部假设相一致，证实了构建一个多样化的LLM种群作为异质人类专家红色团队成员的有前途的代理的必要性。这为LLMS的可扩展毒性检测和安全对准铺平了道路。



## **10. Large Language Models for Cyber Security: A Systematic Literature Review**

网络安全的大型语言模型：系统性文献综述 cs.CR

47 pages,6 figures

**SubmitDate**: 2024-07-27    [abs](http://arxiv.org/abs/2405.04760v3) [paper-pdf](http://arxiv.org/pdf/2405.04760v3)

**Authors**: Hanxiang Xu, Shenao Wang, Ningke Li, Kailong Wang, Yanjie Zhao, Kai Chen, Ting Yu, Yang Liu, Haoyu Wang

**Abstract**: The rapid advancement of Large Language Models (LLMs) has opened up new opportunities for leveraging artificial intelligence in various domains, including cybersecurity. As the volume and sophistication of cyber threats continue to grow, there is an increasing need for intelligent systems that can automatically detect vulnerabilities, analyze malware, and respond to attacks. In this survey, we conduct a comprehensive review of the literature on the application of LLMs in cybersecurity (LLM4Security). By comprehensively collecting over 30K relevant papers and systematically analyzing 127 papers from top security and software engineering venues, we aim to provide a holistic view of how LLMs are being used to solve diverse problems across the cybersecurity domain. Through our analysis, we identify several key findings. First, we observe that LLMs are being applied to a wide range of cybersecurity tasks, including vulnerability detection, malware analysis, network intrusion detection, and phishing detection. Second, we find that the datasets used for training and evaluating LLMs in these tasks are often limited in size and diversity, highlighting the need for more comprehensive and representative datasets. Third, we identify several promising techniques for adapting LLMs to specific cybersecurity domains, such as fine-tuning, transfer learning, and domain-specific pre-training. Finally, we discuss the main challenges and opportunities for future research in LLM4Security, including the need for more interpretable and explainable models, the importance of addressing data privacy and security concerns, and the potential for leveraging LLMs for proactive defense and threat hunting. Overall, our survey provides a comprehensive overview of the current state-of-the-art in LLM4Security and identifies several promising directions for future research.

摘要: 大型语言模型(LLM)的快速发展为在包括网络安全在内的各个领域利用人工智能开辟了新的机会。随着网络威胁的数量和复杂性不断增长，对能够自动检测漏洞、分析恶意软件和响应攻击的智能系统的需求越来越大。在本次调查中，我们对LLMS在网络安全中的应用(LLM4Security)的文献进行了全面的回顾。通过全面收集30K多篇相关论文并系统分析来自顶级安全和软件工程场所的127篇论文，我们的目标是提供一个全面的视角，了解LLM是如何被用来解决网络安全领域的各种问题的。通过我们的分析，我们确定了几个关键发现。首先，我们观察到LLMS正被应用于广泛的网络安全任务，包括漏洞检测、恶意软件分析、网络入侵检测和网络钓鱼检测。其次，我们发现，在这些任务中用于训练和评估土地管理的数据集在大小和多样性方面往往有限，这突显了需要更全面和更具代表性的数据集。第三，我们确定了几种有前景的技术来使LLMS适应特定的网络安全领域，例如微调、迁移学习和特定领域的预训练。最后，我们讨论了LLM4Security未来研究的主要挑战和机遇，包括需要更多可解释和可解释的模型，解决数据隐私和安全问题的重要性，以及利用LLM进行主动防御和威胁追踪的潜力。总体而言，我们的调查提供了对LLM4Security当前最先进技术的全面概述，并确定了未来研究的几个有希望的方向。



## **11. LLM Platform Security: Applying a Systematic Evaluation Framework to OpenAI's ChatGPT Plugins**

LLM平台安全：将系统评估框架应用于OpenAI的ChatGPT插件 cs.CR

To appear in the proceedings of the 7th AAAI / ACM Conference on AI,  Ethics, and Society (AIES), October 2024

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2309.10254v2) [paper-pdf](http://arxiv.org/pdf/2309.10254v2)

**Authors**: Umar Iqbal, Tadayoshi Kohno, Franziska Roesner

**Abstract**: Large language model (LLM) platforms, such as ChatGPT, have recently begun offering an app ecosystem to interface with third-party services on the internet. While these apps extend the capabilities of LLM platforms, they are developed by arbitrary third parties and thus cannot be implicitly trusted. Apps also interface with LLM platforms and users using natural language, which can have imprecise interpretations. In this paper, we propose a framework that lays a foundation for LLM platform designers to analyze and improve the security, privacy, and safety of current and future third-party integrated LLM platforms. Our framework is a formulation of an attack taxonomy that is developed by iteratively exploring how LLM platform stakeholders could leverage their capabilities and responsibilities to mount attacks against each other. As part of our iterative process, we apply our framework in the context of OpenAI's plugin (apps) ecosystem. We uncover plugins that concretely demonstrate the potential for the types of issues that we outline in our attack taxonomy. We conclude by discussing novel challenges and by providing recommendations to improve the security, privacy, and safety of present and future LLM-based computing platforms.

摘要: 大型语言模型(LLM)平台，如ChatGPT，最近开始提供一个应用生态系统，以与互联网上的第三方服务对接。虽然这些应用程序扩展了LLM平台的功能，但它们是由任意的第三方开发的，因此不能被隐式信任。应用程序还使用自然语言与LLM平台和用户交互，这可能会有不准确的解释。在本文中，我们提出了一个框架，为LLM平台设计者分析和改进现有和未来第三方集成LLM平台的安全性、保密性和安全性奠定了基础。我们的框架是攻击分类的公式，通过迭代探索LLM平台利益相关者如何利用他们的能力和责任来对彼此发动攻击而开发的攻击分类。作为迭代过程的一部分，我们在OpenAI的插件(App)生态系统中应用我们的框架。我们发现了一些插件，这些插件具体展示了我们在攻击分类中概述的问题类型的可能性。最后，我们讨论了新的挑战，并提供了改进当前和未来基于LLM的计算平台的安全性、保密性和安全性的建议。



## **12. Blockchain for Large Language Model Security and Safety: A Holistic Survey**

大型语言模型安全性的区块链：整体调查 cs.CR

Submitted to SIGKDD Explorations

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.20181v1) [paper-pdf](http://arxiv.org/pdf/2407.20181v1)

**Authors**: Caleb Geren, Amanda Board, Gaby G. Dagher, Tim Andersen, Jun Zhuang

**Abstract**: With the advent of accessible interfaces for interacting with large language models, there has been an associated explosion in both their commercial and academic interest. Consequently, there has also been an sudden burst of novel attacks associated with large language models, jeopardizing user data on a massive scale. Situated at a comparable crossroads in its development, and equally prolific to LLMs in its rampant growth, blockchain has emerged in recent years as a disruptive technology with the potential to redefine how we approach data handling. In particular, and due to its strong guarantees about data immutability and irrefutability as well as inherent data provenance assurances, blockchain has attracted significant attention as a means to better defend against the array of attacks affecting LLMs and further improve the quality of their responses. In this survey, we holistically evaluate current research on how blockchains are being used to help protect against LLM vulnerabilities, as well as analyze how they may further be used in novel applications. To better serve these ends, we introduce a taxonomy of blockchain for large language models (BC4LLM) and also develop various definitions to precisely capture the nature of different bodies of research in these areas. Moreover, throughout the paper, we present frameworks to contextualize broader research efforts, and in order to motivate the field further, we identify future research goals as well as challenges present in the blockchain for large language model (BC4LLM) space.

摘要: 随着用于与大型语言模型交互的可访问接口的出现，它们在商业和学术方面的兴趣都随之激增。因此，也突然爆发了与大型语言模型相关的新型攻击，大规模危及用户数据。区块链位于其发展的类似十字路口，在其猖獗的增长中与LLMS同样多产，近年来作为一种颠覆性技术出现，有可能重新定义我们处理数据的方式。特别是，由于区块链对数据的不可变性和不可否认性以及内在的数据来源保证的强有力的保证，区块链作为一种更好地防御影响LLM的一系列攻击并进一步提高其响应质量的手段引起了极大的关注。在这次调查中，我们全面评估了目前关于区块链如何被用来帮助防御LLM漏洞的研究，并分析了它们如何进一步用于新的应用程序。为了更好地服务于这些目标，我们引入了大型语言模型的区块链分类(BC4LLM)，并开发了各种定义，以准确捕获这些领域不同研究主体的性质。此外，在整篇论文中，我们提出了框架以适应更广泛的研究努力，并且为了进一步推动该领域，我们确定了未来的研究目标以及大语言模型区块链(BC4LLM)空间中存在的挑战。



## **13. The Janus Interface: How Fine-Tuning in Large Language Models Amplifies the Privacy Risks**

Janus界面：大型语言模型中的微调如何放大隐私风险 cs.CR

This work has been accepted by CCS 2024

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2310.15469v3) [paper-pdf](http://arxiv.org/pdf/2310.15469v3)

**Authors**: Xiaoyi Chen, Siyuan Tang, Rui Zhu, Shijun Yan, Lei Jin, Zihao Wang, Liya Su, Zhikun Zhang, XiaoFeng Wang, Haixu Tang

**Abstract**: The rapid advancements of large language models (LLMs) have raised public concerns about the privacy leakage of personally identifiable information (PII) within their extensive training datasets. Recent studies have demonstrated that an adversary could extract highly sensitive privacy data from the training data of LLMs with carefully designed prompts. However, these attacks suffer from the model's tendency to hallucinate and catastrophic forgetting (CF) in the pre-training stage, rendering the veracity of divulged PIIs negligible. In our research, we propose a novel attack, Janus, which exploits the fine-tuning interface to recover forgotten PIIs from the pre-training data in LLMs. We formalize the privacy leakage problem in LLMs and explain why forgotten PIIs can be recovered through empirical analysis on open-source language models. Based upon these insights, we evaluate the performance of Janus on both open-source language models and two latest LLMs, i.e., GPT-3.5-Turbo and LLaMA-2-7b. Our experiment results show that Janus amplifies the privacy risks by over 10 times in comparison with the baseline and significantly outperforms the state-of-the-art privacy extraction attacks including prefix attacks and in-context learning (ICL). Furthermore, our analysis validates that existing fine-tuning APIs provided by OpenAI and Azure AI Studio are susceptible to our Janus attack, allowing an adversary to conduct such an attack at a low cost.

摘要: 大型语言模型(LLM)的快速发展引起了公众对其广泛训练数据集中个人身份信息(PII)隐私泄露的担忧。最近的研究表明，攻击者可以通过精心设计的提示从LLMS的训练数据中提取高度敏感的隐私数据。然而，这些攻击受到模型在预训练阶段的幻觉和灾难性遗忘(CF)的倾向的影响，使得泄露的PII的真实性可以忽略不计。在我们的研究中，我们提出了一种新的攻击，Janus，它利用微调接口从LLMS的训练前数据中恢复被遗忘的PII。我们形式化地描述了LLMS中的隐私泄露问题，并通过对开源语言模型的实证分析解释了为什么被遗忘的PII可以恢复。基于这些见解，我们评估了Janus在开源语言模型和两个最新的LLMS上的性能，即GPT-3.5-Turbo和Llama-2-7b。我们的实验结果表明，Janus将隐私风险放大了10倍以上，并且显著优于目前最先进的隐私提取攻击，包括前缀攻击和上下文中学习(ICL)。此外，我们的分析验证了OpenAI和Azure AI Studio提供的现有微调API容易受到我们的Janus攻击，允许对手以低成本进行此类攻击。



## **14. Dysca: A Dynamic and Scalable Benchmark for Evaluating Perception Ability of LVLMs**

Dysca：评估LVLM感知能力的动态和可扩展基准 cs.CV

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2406.18849v2) [paper-pdf](http://arxiv.org/pdf/2406.18849v2)

**Authors**: Jie Zhang, Zhongqi Wang, Mengqi Lei, Zheng Yuan, Bei Yan, Shiguang Shan, Xilin Chen

**Abstract**: Currently many benchmarks have been proposed to evaluate the perception ability of the Large Vision-Language Models (LVLMs). However, most benchmarks conduct questions by selecting images from existing datasets, resulting in the potential data leakage. Besides, these benchmarks merely focus on evaluating LVLMs on the realistic style images and clean scenarios, leaving the multi-stylized images and noisy scenarios unexplored. In response to these challenges, we propose a dynamic and scalable benchmark named Dysca for evaluating LVLMs by leveraging synthesis images. Specifically, we leverage Stable Diffusion and design a rule-based method to dynamically generate novel images, questions and the corresponding answers. We consider 51 kinds of image styles and evaluate the perception capability in 20 subtasks. Moreover, we conduct evaluations under 4 scenarios (i.e., Clean, Corruption, Print Attacking and Adversarial Attacking) and 3 question types (i.e., Multi-choices, True-or-false and Free-form). Thanks to the generative paradigm, Dysca serves as a scalable benchmark for easily adding new subtasks and scenarios. A total of 8 advanced open-source LVLMs with 10 checkpoints are evaluated on Dysca, revealing the drawbacks of current LVLMs. The benchmark is released in \url{https://github.com/Benchmark-Dysca/Dysca}.

摘要: 目前，人们已经提出了许多基准来评估大型视觉语言模型的感知能力。然而，大多数基准测试通过从现有数据集中选择图像来进行问题，从而导致潜在的数据泄漏。此外，这些基准只关注真实感风格的图像和干净的场景来评估LVLMS，而对多风格化的图像和噪声场景没有进行探索。为了应对这些挑战，我们提出了一种动态的、可扩展的基准测试DYSCA，用于利用合成图像来评估LVLMS。具体地说，我们利用稳定扩散并设计了一种基于规则的方法来动态生成新的图像、问题和相应的答案。我们考虑了51种图像风格，并在20个子任务中评估了感知能力。此外，我们还在4个场景(即廉洁、腐败、打印攻击和对抗性攻击)和3个问题类型(即多项选择、对错和自由形式)下进行了评估。多亏了生成性范式，Dysca成为了一个可伸缩的基准，可以轻松添加新的子任务和场景。在Dysca上对8个具有10个检查点的高级开源LVLMS进行了评估，揭示了当前LVLMS的缺陷。该基准测试在\url{https://github.com/Benchmark-Dysca/Dysca}.中发布



## **15. MistralBSM: Leveraging Mistral-7B for Vehicular Networks Misbehavior Detection**

MistralBSM：利用Mistral-7 B进行车辆网络不当行为检测 cs.LG

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.18462v1) [paper-pdf](http://arxiv.org/pdf/2407.18462v1)

**Authors**: Wissal Hamhoum, Soumaya Cherkaoui

**Abstract**: Vehicular networks are exposed to various threats resulting from malicious attacks. These threats compromise the security and reliability of communications among road users, thereby jeopardizing road and traffic safety. One of the main vectors of these attacks within vehicular networks is misbehaving vehicles. To address this challenge, we propose deploying a pretrained Large Language Model (LLM)-empowered Misbehavior Detection System (MDS) within an edge-cloud detection framework. Specifically, we fine-tune Mistral-7B, a state-of-the-art LLM, as the edge component to enable real-time detection, whereas a larger LLM deployed in the cloud can conduct a more comprehensive analysis. Our experiments conducted on the extended VeReMi dataset demonstrate Mistral-7B's superior performance, achieving 98\% accuracy compared to other LLMs such as LLAMA2-7B and RoBERTa. Additionally, we investigate the impact of window size on computational costs to optimize deployment efficiency. Leveraging LLMs in MDS shows interesting results in improving the detection of vehicle misbehavior, consequently strengthening vehicular network security to ensure the safety of road users.

摘要: 车载网络面临着由恶意攻击引起的各种威胁。这些威胁危及道路使用者之间通信的安全和可靠性，从而危及道路和交通安全。在车辆网络中，这些攻击的主要载体之一是行为不端的车辆。为了应对这一挑战，我们建议在边缘云检测框架内部署一个预先训练的大型语言模型(LLM)授权的不当行为检测系统(MDS)。具体地说，我们微调了最先进的LLM Mistral-7B作为边缘组件，以实现实时检测，而部署在云中的更大的LLM可以进行更全面的分析。我们在扩展的VeReMi数据集上进行的实验表明，Mistral-7B具有优越的性能，与LLAMA2-7B和Roberta等其他LLMS相比，准确率达到98%。此外，我们还研究了窗口大小对计算成本的影响，以优化部署效率。在MDS中利用LLMS在提高对车辆不当行为的检测，从而加强车载网络安全以确保道路使用者的安全方面显示出有趣的结果。



## **16. SafeDecoding: Defending against Jailbreak Attacks via Safety-Aware Decoding**

SafeDecoding：通过安全意识解码防御越狱攻击 cs.CR

To appear in ACL 2024

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2402.08983v4) [paper-pdf](http://arxiv.org/pdf/2402.08983v4)

**Authors**: Zhangchen Xu, Fengqing Jiang, Luyao Niu, Jinyuan Jia, Bill Yuchen Lin, Radha Poovendran

**Abstract**: As large language models (LLMs) become increasingly integrated into real-world applications such as code generation and chatbot assistance, extensive efforts have been made to align LLM behavior with human values, including safety. Jailbreak attacks, aiming to provoke unintended and unsafe behaviors from LLMs, remain a significant/leading LLM safety threat. In this paper, we aim to defend LLMs against jailbreak attacks by introducing SafeDecoding, a safety-aware decoding strategy for LLMs to generate helpful and harmless responses to user queries. Our insight in developing SafeDecoding is based on the observation that, even though probabilities of tokens representing harmful contents outweigh those representing harmless responses, safety disclaimers still appear among the top tokens after sorting tokens by probability in descending order. This allows us to mitigate jailbreak attacks by identifying safety disclaimers and amplifying their token probabilities, while simultaneously attenuating the probabilities of token sequences that are aligned with the objectives of jailbreak attacks. We perform extensive experiments on five LLMs using six state-of-the-art jailbreak attacks and four benchmark datasets. Our results show that SafeDecoding significantly reduces the attack success rate and harmfulness of jailbreak attacks without compromising the helpfulness of responses to benign user queries. SafeDecoding outperforms six defense methods.

摘要: 随着大型语言模型(LLM)越来越多地集成到真实世界的应用中，如代码生成和聊天机器人辅助，人们已经做出了广泛的努力来使LLM的行为与包括安全在内的人类价值观保持一致。越狱攻击旨在挑起LLM的意外和不安全行为，仍然是LLM的重大/主要安全威胁。在本文中，我们的目标是通过引入SafeDecoding来防御LLMS的越狱攻击，SafeDecoding是一种安全感知的解码策略，用于LLMS对用户查询生成有用和无害的响应。我们开发SafeDecoding的洞察力基于这样的观察：即使代表有害内容的令牌的概率大于代表无害响应的令牌的概率，但在按概率降序对令牌进行排序后，安全免责声明仍会出现在排名最靠前的令牌中。这使我们能够通过识别安全免责声明并放大其令牌概率来缓解越狱攻击，同时降低与越狱攻击目标一致的令牌序列的概率。我们使用六个最先进的越狱攻击和四个基准数据集在五个LLM上进行了广泛的实验。我们的结果表明，SafeDecoding在不影响对良性用户查询的响应的帮助的情况下，显著降低了越狱攻击的攻击成功率和危害性。安全解码的性能超过了六种防御方法。



## **17. Human-Interpretable Adversarial Prompt Attack on Large Language Models with Situational Context**

具有情境上下文的大型语言模型的人类可解释对抗提示攻击 cs.CL

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2407.14644v2) [paper-pdf](http://arxiv.org/pdf/2407.14644v2)

**Authors**: Nilanjana Das, Edward Raff, Manas Gaur

**Abstract**: Previous research on testing the vulnerabilities in Large Language Models (LLMs) using adversarial attacks has primarily focused on nonsensical prompt injections, which are easily detected upon manual or automated review (e.g., via byte entropy). However, the exploration of innocuous human-understandable malicious prompts augmented with adversarial injections remains limited. In this research, we explore converting a nonsensical suffix attack into a sensible prompt via a situation-driven contextual re-writing. This allows us to show suffix conversion without any gradients, using only LLMs to perform the attacks, and thus better understand the scope of possible risks. We combine an independent, meaningful adversarial insertion and situations derived from movies to check if this can trick an LLM. The situations are extracted from the IMDB dataset, and prompts are defined following a few-shot chain-of-thought prompting. Our approach demonstrates that a successful situation-driven attack can be executed on both open-source and proprietary LLMs. We find that across many LLMs, as few as 1 attempt produces an attack and that these attacks transfer between LLMs.

摘要: 之前关于使用对抗性攻击测试大型语言模型(LLM)中的漏洞的研究主要集中在无意义的提示注入上，这些注入很容易通过手动或自动审查(例如，通过字节熵)检测到。然而，通过恶意注入增强无害的人类可理解的恶意提示的探索仍然有限。在这项研究中，我们探索通过情景驱动的语境重写将无意义的后缀攻击转化为合理的提示。这使我们能够显示没有任何梯度的后缀转换，仅使用LLM来执行攻击，从而更好地了解可能风险的范围。我们结合了一个独立的、有意义的敌意插入和来自电影的情况来检查这是否可以欺骗LLM。情况是从IMDB数据集中提取的，提示是在几个镜头的思维链提示之后定义的。我们的方法表明，成功的情境驱动攻击可以在开源和专有LLM上执行。我们发现，在许多LLM中，只有1次尝试就会产生攻击，并且这些攻击会在LLM之间传输。



## **18. Dr. Jekyll and Mr. Hyde: Two Faces of LLMs**

杰基尔博士和海德先生：法学硕士的两面 cs.CR

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2312.03853v4) [paper-pdf](http://arxiv.org/pdf/2312.03853v4)

**Authors**: Matteo Gioele Collu, Tom Janssen-Groesbeek, Stefanos Koffas, Mauro Conti, Stjepan Picek

**Abstract**: Recently, we have witnessed a rise in the use of Large Language Models (LLMs), especially in applications like chatbot assistants. Safety mechanisms and specialized training procedures are implemented to prevent improper responses from these assistants. In this work, we bypass these measures for ChatGPT and Gemini (and, to some extent, Bing chat) by making them impersonate complex personas with personality characteristics that are not aligned with a truthful assistant. We start by creating elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversations then follow a role-play style to elicit prohibited responses. Using personas, we show that prohibited responses are actually provided, making it possible to obtain unauthorized, illegal, or harmful information. This work shows that by using adversarial personas, one can overcome safety mechanisms set out by ChatGPT and Gemini. We also introduce several ways of activating such adversarial personas, which show that both chatbots are vulnerable to this kind of attack. With the same principle, we introduce two defenses that push the model to interpret trustworthy personalities and make it more robust against such attacks.

摘要: 最近，我们看到大型语言模型(LLM)的使用有所增加，特别是在聊天机器人助手等应用程序中。实施了安全机制和专门的培训程序，以防止这些助理做出不当反应。在这项工作中，我们绕过了ChatGPT和Gemini(在某种程度上，Bing聊天)的这些措施，让他们模仿具有与诚实的助手不一致的个性特征的复杂人物角色。我们首先为这些角色创建精致的传记，然后在与相同的聊天机器人的新会话中使用。然后，我们的对话遵循角色扮演的风格，以引发被禁止的回应。使用人物角色，我们展示了实际上提供了被禁止的响应，使得获得未经授权的、非法的或有害的信息成为可能。这项工作表明，通过使用敌对的人物角色，一个人可以克服ChatGPT和Gemini提出的安全机制。我们还介绍了几种激活这种敌对角色的方法，这表明这两个聊天机器人都容易受到这种攻击。在相同的原则下，我们引入了两个防御措施，推动该模型解释可信任的个性，并使其对此类攻击更加健壮。



## **19. The Dark Side of Function Calling: Pathways to Jailbreaking Large Language Models**

函数调用的阴暗面：越狱大型语言模型的途径 cs.CR

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2407.17915v1) [paper-pdf](http://arxiv.org/pdf/2407.17915v1)

**Authors**: Zihui Wu, Haichang Gao, Jianping He, Ping Wang

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities, but their power comes with significant security considerations. While extensive research has been conducted on the safety of LLMs in chat mode, the security implications of their function calling feature have been largely overlooked. This paper uncovers a critical vulnerability in the function calling process of LLMs, introducing a novel "jailbreak function" attack method that exploits alignment discrepancies, user coercion, and the absence of rigorous safety filters. Our empirical study, conducted on six state-of-the-art LLMs including GPT-4o, Claude-3.5-Sonnet, and Gemini-1.5-pro, reveals an alarming average success rate of over 90\% for this attack. We provide a comprehensive analysis of why function calls are susceptible to such attacks and propose defensive strategies, including the use of defensive prompts. Our findings highlight the urgent need for enhanced security measures in the function calling capabilities of LLMs, contributing to the field of AI safety by identifying a previously unexplored risk, designing an effective attack method, and suggesting practical defensive measures. Our code is available at https://github.com/wooozihui/jailbreakfunction.

摘要: 大型语言模型(LLM)已经展示了非凡的能力，但它们的强大也伴随着重要的安全考虑。虽然已经对聊天模式下的LLMS的安全性进行了广泛的研究，但其函数调用功能的安全含义在很大程度上被忽视了。本文揭示了LLMS函数调用过程中的一个严重漏洞，引入了一种新的“越狱函数”攻击方法，该方法利用了对齐差异、用户胁迫和缺乏严格的安全过滤器。我们在包括GPT-40、Claude-3.5-Sonnet和Gemini-1.5-Pro在内的六个最先进的LLM上进行的经验研究显示，该攻击的平均成功率超过90%，这是令人震惊的。我们对函数调用容易受到此类攻击的原因进行了全面分析，并提出了防御策略，包括使用防御提示。我们的发现突显了在LLMS的函数调用能力方面迫切需要增强安全措施，通过识别以前未探索的风险、设计有效的攻击方法并提出实用的防御措施来促进人工智能安全领域。我们的代码可以在https://github.com/wooozihui/jailbreakfunction.上找到



## **20. PenHeal: A Two-Stage LLM Framework for Automated Pentesting and Optimal Remediation**

PenHeal：用于自动冥想和最佳补救的两阶段LLM框架 cs.CR

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2407.17788v1) [paper-pdf](http://arxiv.org/pdf/2407.17788v1)

**Authors**: Junjie Huang, Quanyan Zhu

**Abstract**: Recent advances in Large Language Models (LLMs) have shown significant potential in enhancing cybersecurity defenses against sophisticated threats. LLM-based penetration testing is an essential step in automating system security evaluations by identifying vulnerabilities. Remediation, the subsequent crucial step, addresses these discovered vulnerabilities. Since details about vulnerabilities, exploitation methods, and software versions offer crucial insights into system weaknesses, integrating penetration testing with vulnerability remediation into a cohesive system has become both intuitive and necessary.   This paper introduces PenHeal, a two-stage LLM-based framework designed to autonomously identify and mitigate security vulnerabilities. The framework integrates two LLM-enabled components: the Pentest Module, which detects multiple vulnerabilities within a system, and the Remediation Module, which recommends optimal remediation strategies. The integration is facilitated through Counterfactual Prompting and an Instructor module that guides the LLMs using external knowledge to explore multiple potential attack paths effectively. Our experimental results demonstrate that PenHeal not only automates the identification and remediation of vulnerabilities but also significantly improves vulnerability coverage by 31%, increases the effectiveness of remediation strategies by 32%, and reduces the associated costs by 46% compared to baseline models. These outcomes highlight the transformative potential of LLMs in reshaping cybersecurity practices, offering an innovative solution to defend against cyber threats.

摘要: 大型语言模型(LLM)的最新进展已经显示出在加强针对复杂威胁的网络安全防御方面的巨大潜力。基于LLM的渗透测试是通过识别漏洞实现系统安全评估自动化的重要步骤。补救是后续的关键步骤，可解决这些已发现的漏洞。由于有关漏洞、利用方法和软件版本的详细信息提供了对系统弱点的重要洞察，因此将渗透测试与漏洞修复集成到一个连贯的系统中既是直观的，也是必要的。本文介绍了PenHeal，这是一个基于LLM的两阶段框架，旨在自主识别和缓解安全漏洞。该框架集成了两个启用LLM的组件：检测系统中多个漏洞的Pentest模块和建议最佳补救策略的补救模块。通过反事实提示和指导LLMS使用外部知识有效地探索多个潜在攻击路径的指导者模块来促进集成。我们的实验结果表明，PenHeal不仅自动化了漏洞的识别和修复，而且与基准模型相比，漏洞覆盖率显著提高了31%，修复策略的有效性提高了32%，相关成本降低了46%。这些结果突显了低成本管理在重塑网络安全实践方面的变革潜力，为防御网络威胁提供了一种创新的解决方案。



## **21. Towards Neural Network based Cognitive Models of Dynamic Decision-Making by Humans**

基于神经网络的人类动态决策认知模型 cs.LG

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2407.17622v1) [paper-pdf](http://arxiv.org/pdf/2407.17622v1)

**Authors**: Changyu Chen, Shashank Reddy Chirra, Maria José Ferreira, Cleotilde Gonzalez, Arunesh Sinha, Pradeep Varakantham

**Abstract**: Modelling human cognitive processes in dynamic decision-making tasks has been an endeavor in AI for a long time. Some initial works have attempted to utilize neural networks (and large language models) but often assume one common model for all humans and aim to emulate human behavior in aggregate. However, behavior of each human is distinct, heterogeneous and relies on specific past experiences in specific tasks. To that end, we build on a well known model of cognition, namely Instance Based Learning (IBL), that posits that decisions are made based on similar situations encountered in the past. We propose two new attention based neural network models to model human decision-making in dynamic settings. We experiment with two distinct datasets gathered from human subject experiment data, one focusing on detection of phishing email by humans and another where humans act as attackers in a cybersecurity setting and decide on an attack option. We conduct extensive experiments with our two neural network models, IBL, and GPT3.5, and demonstrate that one of our neural network models achieves the best performance in representing human decision-making. We find an interesting trend that all models predict a human's decision better if that human is better at the task. We also explore explanation of human decisions based on what our model considers important in prediction. Overall, our work yields promising results for further use of neural networks in cognitive modelling of human decision making. Our code is available at https://github.com/shshnkreddy/NCM-HDM.

摘要: 在动态决策任务中对人类认知过程进行建模一直是人工智能领域的一项努力。一些最初的工作试图利用神经网络(和大型语言模型)，但通常假设一个适用于所有人类的通用模型，并旨在总体上模拟人类的行为。然而，每个人的行为是不同的，不同的，依赖于特定任务中特定的过去经验。为此，我们建立了一个著名的认知模型，即基于实例的学习(IBL)，该模型假设决策是基于过去遇到的类似情况做出的。我们提出了两个新的基于注意力的神经网络模型来模拟动态环境下的人类决策。我们使用从人类受试者实验数据中收集的两个不同的数据集进行实验，一个专注于检测人类发送的钓鱼电子邮件，另一个则是人类在网络安全环境中充当攻击者，并决定攻击选项。我们用我们的两个神经网络模型IBL和GPT3.5进行了广泛的实验，并证明了我们的一个神经网络模型在表示人类决策方面取得了最好的性能。我们发现了一个有趣的趋势，所有模型都能更好地预测一个人的决定，如果那个人在这项任务上做得更好。我们还基于我们的模型认为在预测中重要的东西来探索对人类决策的解释。总体而言，我们的工作为进一步使用神经网络在人类决策的认知建模中提供了有希望的结果。我们的代码可以在https://github.com/shshnkreddy/NCM-HDM.上找到



## **22. Can Watermarking Large Language Models Prevent Copyrighted Text Generation and Hide Training Data?**

对大型语言模型进行水印可以阻止受版权保护的文本生成并隐藏训练数据吗？ cs.LG

21 pages, 6 figures

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2407.17417v1) [paper-pdf](http://arxiv.org/pdf/2407.17417v1)

**Authors**: Michael-Andrei Panaitescu-Liess, Zora Che, Bang An, Yuancheng Xu, Pankayaraj Pathmanathan, Souradip Chakraborty, Sicheng Zhu, Tom Goldstein, Furong Huang

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities in generating diverse and contextually rich text. However, concerns regarding copyright infringement arise as LLMs may inadvertently produce copyrighted material. In this paper, we first investigate the effectiveness of watermarking LLMs as a deterrent against the generation of copyrighted texts. Through theoretical analysis and empirical evaluation, we demonstrate that incorporating watermarks into LLMs significantly reduces the likelihood of generating copyrighted content, thereby addressing a critical concern in the deployment of LLMs. Additionally, we explore the impact of watermarking on Membership Inference Attacks (MIAs), which aim to discern whether a sample was part of the pretraining dataset and may be used to detect copyright violations. Surprisingly, we find that watermarking adversely affects the success rate of MIAs, complicating the task of detecting copyrighted text in the pretraining dataset. Finally, we propose an adaptive technique to improve the success rate of a recent MIA under watermarking. Our findings underscore the importance of developing adaptive methods to study critical problems in LLMs with potential legal implications.

摘要: 大型语言模型(LLM)在生成丰富多样的文本方面表现出了令人印象深刻的能力。然而，由于LLMS可能无意中产生了受版权保护的材料，因此出现了对侵犯版权的担忧。在这篇文章中，我们首先研究了水印LLM作为对版权文本产生的威慑的有效性。通过理论分析和实证评估，我们证明了在LLMS中加入水印显著降低了产生受版权保护内容的可能性，从而解决了LLMS部署中的一个关键问题。此外，我们还探讨了水印对成员关系推断攻击(MIA)的影响，该攻击旨在识别样本是否是预训练数据集的一部分，以及是否可以用于检测侵犯版权的行为。令人惊讶的是，我们发现水印对MIA的成功率产生了不利影响，使得在预训练数据集中检测受版权保护的文本的任务变得更加复杂。最后，我们提出了一种自适应技术来提高最近的MIA在水印下的成功率。我们的发现强调了开发适应性方法来研究LLMS中具有潜在法律含义的关键问题的重要性。



## **23. LLMmap: Fingerprinting For Large Language Models**

LLMmap：大型语言模型的指纹识别 cs.CR

version 0.1 (added missing refs)

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2407.15847v2) [paper-pdf](http://arxiv.org/pdf/2407.15847v2)

**Authors**: Dario Pasquini, Evgenios M. Kornaropoulos, Giuseppe Ateniese

**Abstract**: We introduce LLMmap, a first-generation fingerprinting attack targeted at LLM-integrated applications. LLMmap employs an active fingerprinting approach, sending carefully crafted queries to the application and analyzing the responses to identify the specific LLM model in use. With as few as 8 interactions, LLMmap can accurately identify LLMs with over 95% accuracy. More importantly, LLMmap is designed to be robust across different application layers, allowing it to identify LLMs operating under various system prompts, stochastic sampling hyperparameters, and even complex generation frameworks such as RAG or Chain-of-Thought.

摘要: 我们引入了LLMmap，这是一种针对LLM集成应用程序的第一代指纹攻击。LLMmap采用主动指纹识别方法，向应用程序发送精心设计的查询并分析响应以识别正在使用的特定LLM模型。LLMmap只需8次交互即可准确识别LLM，准确率超过95%。更重要的是，LLMmap的设计目的是在不同的应用层中具有鲁棒性，使其能够识别在各种系统提示、随机采样超参数甚至复杂的生成框架（例如RAG或思想链）下运行的LLM。



## **24. From Sands to Mansions: Enabling Automatic Full-Life-Cycle Cyberattack Construction with LLM**

从金沙到豪宅：利用LLM实现自动全生命周期网络攻击构建 cs.CR

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2407.16928v1) [paper-pdf](http://arxiv.org/pdf/2407.16928v1)

**Authors**: Lingzhi Wang, Jiahui Wang, Kyle Jung, Kedar Thiagarajan, Emily Wei, Xiangmin Shen, Yan Chen, Zhenyuan Li

**Abstract**: The escalating battles between attackers and defenders in cybersecurity make it imperative to test and evaluate defense capabilities from the attackers' perspective. However, constructing full-life-cycle cyberattacks and performing red team emulations requires significant time and domain knowledge from security experts. Existing cyberattack simulation frameworks face challenges such as limited technical coverage, inability to conduct full-life-cycle attacks, and the need for manual infrastructure building. These limitations hinder the quality and diversity of the constructed attacks. In this paper, we leveraged the capabilities of Large Language Models (LLMs) in summarizing knowledge from existing attack intelligence and generating executable machine code based on human knowledge. we proposed AURORA, an automatic end-to-end cyberattack construction and emulation framework. AURORA can autonomously build multi-stage cyberattack plans based on Cyber Threat Intelligence (CTI) reports, construct the emulation infrastructures, and execute the attack procedures. We also developed an attack procedure knowledge graph to integrate knowledge about attack techniques throughout the full life cycle of advanced cyberattacks from various sources. We constructed and evaluated more than 20 full-life-cycle cyberattacks based on existing CTI reports. Compared to previous attack simulation frameworks, AURORA can construct multi-step attacks and the infrastructures in several minutes without human intervention. Furthermore, AURORA incorporates a wider range (40% more) of attack techniques into the constructed attacks in a more efficient way than the professional red teams. To benefit further research, we open-sourced the dataset containing the execution files and infrastructures of 20 emulated cyberattacks.

摘要: 网络安全领域攻击者和防御者之间不断升级的战斗使得从攻击者的角度测试和评估防御能力势在必行。然而，构建全生命周期的网络攻击和执行Red Team模拟需要安全专家的大量时间和领域知识。现有的网络攻击模拟框架面临着诸如技术覆盖范围有限、无法进行全生命周期攻击以及需要手动构建基础设施等挑战。这些局限性阻碍了构建攻击的质量和多样性。在本文中，我们利用大型语言模型(LLM)的能力，从现有的攻击情报中总结知识，并基于人类知识生成可执行的机器代码。我们提出了Aurora，一个自动化的端到端网络攻击构建和仿真框架。Aurora可以根据网络威胁情报(CTI)报告自主构建多阶段网络攻击计划，构建仿真基础设施，并执行攻击程序。我们还开发了攻击过程知识图，以整合来自各种来源的高级网络攻击的整个生命周期中有关攻击技术的知识。我们根据现有的CTI报告构建和评估了20多个全生命周期的网络攻击。与以往的攻击模拟框架相比，Aurora可以在几分钟内构建多步骤攻击和基础设施，而无需人工干预。此外，极光在构建的攻击中融入了更广泛的攻击技术(多40%)，比专业的红色球队更有效。为了便于进一步研究，我们对包含20个模拟网络攻击的执行文件和基础设施的数据集进行了开源。



## **25. RigorLLM: Resilient Guardrails for Large Language Models against Undesired Content**

RigorLLM：针对不需要内容的大型语言模型的弹性护栏 cs.CR

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2403.13031v2) [paper-pdf](http://arxiv.org/pdf/2403.13031v2)

**Authors**: Zhuowen Yuan, Zidi Xiong, Yi Zeng, Ning Yu, Ruoxi Jia, Dawn Song, Bo Li

**Abstract**: Recent advancements in Large Language Models (LLMs) have showcased remarkable capabilities across various tasks in different domains. However, the emergence of biases and the potential for generating harmful content in LLMs, particularly under malicious inputs, pose significant challenges. Current mitigation strategies, while effective, are not resilient under adversarial attacks. This paper introduces Resilient Guardrails for Large Language Models (RigorLLM), a novel framework designed to efficiently and effectively moderate harmful and unsafe inputs and outputs for LLMs. By employing a multi-faceted approach that includes energy-based training data augmentation through Langevin dynamics, optimizing a safe suffix for inputs via minimax optimization, and integrating a fusion-based model combining robust KNN with LLMs based on our data augmentation, RigorLLM offers a robust solution to harmful content moderation. Our experimental evaluations demonstrate that RigorLLM not only outperforms existing baselines like OpenAI API and Perspective API in detecting harmful content but also exhibits unparalleled resilience to jailbreaking attacks. The innovative use of constrained optimization and a fusion-based guardrail approach represents a significant step forward in developing more secure and reliable LLMs, setting a new standard for content moderation frameworks in the face of evolving digital threats.

摘要: 大型语言模型(LLM)的最新进展展示了跨越不同领域的各种任务的显著能力。然而，偏见的出现和在低成本管理中产生有害内容的可能性，特别是在恶意投入下，构成了重大挑战。目前的缓解战略虽然有效，但在对抗性攻击下缺乏弹性。本文介绍了用于大型语言模型的弹性护栏(RigorLLM)，这是一个新的框架，旨在高效和有效地控制LLM中有害和不安全的输入和输出。通过采用多方面的方法，包括通过朗之万动力学基于能量的训练数据增强，通过极小极大优化优化输入的安全后缀，以及基于我们的数据增强将稳健的KNN与LLMS相结合的基于融合的模型，RigorLLM为有害内容适度提供了稳健的解决方案。我们的实验评估表明，RigorLLM不仅在检测有害内容方面优于OpenAI API和透视API等现有基线，而且对越狱攻击表现出无与伦比的弹性。约束优化和基于融合的护栏方法的创新使用代表着在开发更安全可靠的LLMS方面向前迈出的重要一步，为面对不断变化的数字威胁的内容审查框架设定了新的标准。



## **26. Watermarks in the Sand: Impossibility of Strong Watermarking for Generative Models**

沙子中的水印：生成模型不可能有强水印 cs.LG

ICML 2024. Website: https://hanlin-zhang.com/impossibility-watermarks

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2311.04378v4) [paper-pdf](http://arxiv.org/pdf/2311.04378v4)

**Authors**: Hanlin Zhang, Benjamin L. Edelman, Danilo Francati, Daniele Venturi, Giuseppe Ateniese, Boaz Barak

**Abstract**: Watermarking generative models consists of planting a statistical signal (watermark) in a model's output so that it can be later verified that the output was generated by the given model. A strong watermarking scheme satisfies the property that a computationally bounded attacker cannot erase the watermark without causing significant quality degradation. In this paper, we study the (im)possibility of strong watermarking schemes. We prove that, under well-specified and natural assumptions, strong watermarking is impossible to achieve. This holds even in the private detection algorithm setting, where the watermark insertion and detection algorithms share a secret key, unknown to the attacker. To prove this result, we introduce a generic efficient watermark attack; the attacker is not required to know the private key of the scheme or even which scheme is used. Our attack is based on two assumptions: (1) The attacker has access to a "quality oracle" that can evaluate whether a candidate output is a high-quality response to a prompt, and (2) The attacker has access to a "perturbation oracle" which can modify an output with a nontrivial probability of maintaining quality, and which induces an efficiently mixing random walk on high-quality outputs. We argue that both assumptions can be satisfied in practice by an attacker with weaker computational capabilities than the watermarked model itself, to which the attacker has only black-box access. Furthermore, our assumptions will likely only be easier to satisfy over time as models grow in capabilities and modalities. We demonstrate the feasibility of our attack by instantiating it to attack three existing watermarking schemes for large language models: Kirchenbauer et al. (2023), Kuditipudi et al. (2023), and Zhao et al. (2023). The same attack successfully removes the watermarks planted by all three schemes, with only minor quality degradation.

摘要: 水印生成模型包括在模型的输出中植入统计信号(水印)，以便稍后可以验证输出是由给定模型生成的。强水印方案满足这样的性质，即计算受限的攻击者不可能在不引起显著质量降级的情况下删除水印。本文研究了强水印方案的(Im)可能性。我们证明了在明确和自然的假设下，强水印是不可能实现的。即使在私有检测算法设置中也是如此，其中水印插入和检测算法共享攻击者未知的秘密密钥。为了证明这一结果，我们引入了一种通用的高效水印攻击；攻击者不需要知道方案的私钥，甚至不需要知道使用了哪个方案。我们的攻击基于两个假设：(1)攻击者可以访问可以评估候选输出是否是对提示的高质量响应的“质量预言”，以及(2)攻击者可以访问“扰动预言”，它可以以保持质量的非平凡概率修改输出，并导致对高质量输出的有效混合随机游走。我们认为，在实践中，这两个假设都可以由计算能力弱于水印模型本身的攻击者满足，因为攻击者只能访问黑盒。此外，随着模型在功能和模式方面的发展，随着时间的推移，我们的假设可能只会更容易满足。我们通过将其实例化来攻击三个现有的用于大型语言模型的水印方案来证明该攻击的可行性：Kirchenbauer等人。(2023)，Kuditipudi等人。(2023)，和赵等人。(2023年)。同样的攻击成功地删除了所有三个方案植入的水印，只有很小的质量下降。



## **27. Can Large Language Models Automatically Jailbreak GPT-4V?**

大型语言模型可以自动越狱GPT-4V吗？ cs.CL

TrustNLP@NAACL2024 (Fourth Workshop on Trustworthy Natural Language  Processing)

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2407.16686v1) [paper-pdf](http://arxiv.org/pdf/2407.16686v1)

**Authors**: Yuanwei Wu, Yue Huang, Yixin Liu, Xiang Li, Pan Zhou, Lichao Sun

**Abstract**: GPT-4V has attracted considerable attention due to its extraordinary capacity for integrating and processing multimodal information. At the same time, its ability of face recognition raises new safety concerns of privacy leakage. Despite researchers' efforts in safety alignment through RLHF or preprocessing filters, vulnerabilities might still be exploited. In our study, we introduce AutoJailbreak, an innovative automatic jailbreak technique inspired by prompt optimization. We leverage Large Language Models (LLMs) for red-teaming to refine the jailbreak prompt and employ weak-to-strong in-context learning prompts to boost efficiency. Furthermore, we present an effective search method that incorporates early stopping to minimize optimization time and token expenditure. Our experiments demonstrate that AutoJailbreak significantly surpasses conventional methods, achieving an Attack Success Rate (ASR) exceeding 95.3\%. This research sheds light on strengthening GPT-4V security, underscoring the potential for LLMs to be exploited in compromising GPT-4V integrity.

摘要: GPT-4V由于其综合和处理多模式信息的非凡能力而引起了相当大的关注。与此同时，它的人脸识别能力引发了新的隐私泄露的安全担忧。尽管研究人员通过RLHF或预处理过滤器在安全匹配方面做出了努力，但漏洞仍有可能被利用。在我们的研究中，我们介绍了AutoJailBreak，这是一种受即时优化启发的创新的自动越狱技术。我们利用用于红色团队的大型语言模型(LLM)来改进越狱提示，并采用从弱到强的上下文学习提示来提高效率。此外，我们还提出了一种结合提前停止的有效搜索方法，以最小化优化时间和令牌开销。实验表明，AutoJailBreak的攻击成功率(ASR)超过95.3%，明显优于传统方法。这项研究有助于加强GPT-4V的安全性，强调了LLMS在危害GPT-4V完整性方面的潜力。



## **28. RedAgent: Red Teaming Large Language Models with Context-aware Autonomous Language Agent**

RedAgent：Red将大型语言模型与上下文感知自治语言代理结合起来 cs.CR

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2407.16667v1) [paper-pdf](http://arxiv.org/pdf/2407.16667v1)

**Authors**: Huiyu Xu, Wenhui Zhang, Zhibo Wang, Feng Xiao, Rui Zheng, Yunhe Feng, Zhongjie Ba, Kui Ren

**Abstract**: Recently, advanced Large Language Models (LLMs) such as GPT-4 have been integrated into many real-world applications like Code Copilot. These applications have significantly expanded the attack surface of LLMs, exposing them to a variety of threats. Among them, jailbreak attacks that induce toxic responses through jailbreak prompts have raised critical safety concerns. To identify these threats, a growing number of red teaming approaches simulate potential adversarial scenarios by crafting jailbreak prompts to test the target LLM. However, existing red teaming methods do not consider the unique vulnerabilities of LLM in different scenarios, making it difficult to adjust the jailbreak prompts to find context-specific vulnerabilities. Meanwhile, these methods are limited to refining jailbreak templates using a few mutation operations, lacking the automation and scalability to adapt to different scenarios. To enable context-aware and efficient red teaming, we abstract and model existing attacks into a coherent concept called "jailbreak strategy" and propose a multi-agent LLM system named RedAgent that leverages these strategies to generate context-aware jailbreak prompts. By self-reflecting on contextual feedback in an additional memory buffer, RedAgent continuously learns how to leverage these strategies to achieve effective jailbreaks in specific contexts. Extensive experiments demonstrate that our system can jailbreak most black-box LLMs in just five queries, improving the efficiency of existing red teaming methods by two times. Additionally, RedAgent can jailbreak customized LLM applications more efficiently. By generating context-aware jailbreak prompts towards applications on GPTs, we discover 60 severe vulnerabilities of these real-world applications with only two queries per vulnerability. We have reported all found issues and communicated with OpenAI and Meta for bug fixes.

摘要: 最近，GPT-4等高级大型语言模型(LLM)已集成到许多实际应用程序中，如Code Copilot。这些应用程序显著扩大了LLMS的攻击面，使它们暴露在各种威胁之下。其中，通过越狱提示引发有毒反应的越狱攻击引发了严重的安全问题。为了识别这些威胁，越来越多的红色团队方法通过精心编制越狱提示来测试目标LLM，以模拟潜在的敌对场景。然而，现有的红色团队方法没有考虑LLM在不同场景下的独特漏洞，很难调整越狱提示来发现上下文特定的漏洞。同时，这些方法仅限于使用少量的变异操作来提炼越狱模板，缺乏适应不同场景的自动化和可扩展性。为了实现上下文感知和高效的红色团队，我们将现有的攻击抽象并建模为一个连贯的概念，称为越狱策略，并提出了一个名为RedAgent的多代理LLM系统，该系统利用这些策略来生成上下文感知越狱提示。通过对额外内存缓冲区中的上下文反馈进行自我反思，RedAgent不断学习如何利用这些策略在特定上下文中实现有效的越狱。大量的实验表明，我们的系统可以在短短五次查询中破解大部分黑盒LLM，将现有的红色团队方法的效率提高了两倍。此外，RedAgent可以更高效地越狱定制的LLM应用程序。通过向GPT上的应用程序生成上下文感知越狱提示，我们发现了这些现实世界应用程序的60个严重漏洞，每个漏洞只有两个查询。我们已经报告了所有发现的问题，并与OpenAI和Meta进行了沟通以修复错误。



## **29. Course-Correction: Safety Alignment Using Synthetic Preferences**

课程纠正：使用综合偏好进行安全调整 cs.CL

Dataset and script will be available at  https://github.com/pillowsofwind/Course-Correction

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2407.16637v1) [paper-pdf](http://arxiv.org/pdf/2407.16637v1)

**Authors**: Rongwu Xu, Yishuo Cai, Zhenhong Zhou, Renjie Gu, Haiqin Weng, Yan Liu, Tianwei Zhang, Wei Xu, Han Qiu

**Abstract**: The risk of harmful content generated by large language models (LLMs) becomes a critical concern. This paper presents a systematic study on assessing and improving LLMs' capability to perform the task of \textbf{course-correction}, \ie, the model can steer away from generating harmful content autonomously. To start with, we introduce the \textsc{C$^2$-Eval} benchmark for quantitative assessment and analyze 10 popular LLMs, revealing varying proficiency of current safety-tuned LLMs in course-correction. To improve, we propose fine-tuning LLMs with preference learning, emphasizing the preference for timely course-correction. Using an automated pipeline, we create \textsc{C$^2$-Syn}, a synthetic dataset with 750K pairwise preferences, to teach models the concept of timely course-correction through data-driven preference learning. Experiments on 2 LLMs, \textsc{Llama2-Chat 7B} and \textsc{Qwen2 7B}, show that our method effectively enhances course-correction skills without affecting general performance. Additionally, it effectively improves LLMs' safety, particularly in resisting jailbreak attacks.

摘要: 大型语言模型(LLM)生成有害内容的风险成为一个关键问题。本文对如何评估和提高LLMS执行航向修正任务的能力进行了系统的研究，即该模型可以避免自动产生有害内容。首先，我们引入了用于定量评估的基准，并对10个流行的LLM进行了分析，揭示了当前安全调整的LLMS在航向校正方面的不同熟练程度。为了改进，我们提出了带有偏好学习的微调LLMS，强调了对及时航向校正的偏好。使用自动化管道，我们创建了一个包含75万个成对偏好的合成数据集\extsc{C$^2$-Syn}，以通过数据驱动的偏好学习向模型传授及时进行课程修正的概念。在Textsc{Llama2-Chat 7B}和Textsc{Qwen2 7B}上的实验表明，该方法在不影响总体性能的情况下有效地增强了航向修正技巧。此外，它还有效地提高了LLMS的安全性，特别是在抵抗越狱攻击方面。



## **30. Prompt Injection Attacks on Large Language Models in Oncology**

对肿瘤学中大型语言模型的即时注入攻击 cs.CR

57 Pages, 5 Figures

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2407.18981v1) [paper-pdf](http://arxiv.org/pdf/2407.18981v1)

**Authors**: Jan Clusmann, Dyke Ferber, Isabella C. Wiest, Carolin V. Schneider, Titus J. Brinker, Sebastian Foersch, Daniel Truhn, Jakob N. Kather

**Abstract**: Vision-language artificial intelligence models (VLMs) possess medical knowledge and can be employed in healthcare in numerous ways, including as image interpreters, virtual scribes, and general decision support systems. However, here, we demonstrate that current VLMs applied to medical tasks exhibit a fundamental security flaw: they can be attacked by prompt injection attacks, which can be used to output harmful information just by interacting with the VLM, without any access to its parameters. We performed a quantitative study to evaluate the vulnerabilities to these attacks in four state of the art VLMs which have been proposed to be of utility in healthcare: Claude 3 Opus, Claude 3.5 Sonnet, Reka Core, and GPT-4o. Using a set of N=297 attacks, we show that all of these models are susceptible. Specifically, we show that embedding sub-visual prompts in medical imaging data can cause the model to provide harmful output, and that these prompts are non-obvious to human observers. Thus, our study demonstrates a key vulnerability in medical VLMs which should be mitigated before widespread clinical adoption.

摘要: 视觉语言人工智能模型(VLM)拥有医学知识，可在医疗保健中以多种方式使用，包括用作图像解释器、虚拟脚本和一般决策支持系统。然而，在这里，我们证明了当前应用于医疗任务的VLM呈现出一个基本的安全缺陷：它们可以被快速注入攻击攻击，该攻击可以仅通过与VLM交互来输出有害信息，而不需要访问其参数。我们进行了一项定量研究，以评估四种被认为在医疗保健中有用的最先进的VLM对这些攻击的漏洞：Claude 3 Opus、Claude 3.5 Sonnet、Reka Core和GPT-40。使用一组N=297次攻击，我们证明了所有这些模型都是易受影响的。具体地说，我们证明了在医学成像数据中嵌入子视觉提示会导致模型提供有害的输出，并且这些提示对人类观察者来说是不明显的。因此，我们的研究证明了医用VLM的一个关键弱点，在广泛临床采用之前应该得到缓解。



## **31. Defending Our Privacy With Backdoors**

用后门保护我们的隐私 cs.LG

Accepted at ECAI 2024

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2310.08320v4) [paper-pdf](http://arxiv.org/pdf/2310.08320v4)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Daniel Neider, Kristian Kersting

**Abstract**: The proliferation of large AI models trained on uncurated, often sensitive web-scraped data has raised significant privacy concerns. One of the concerns is that adversaries can extract information about the training data using privacy attacks. Unfortunately, the task of removing specific information from the models without sacrificing performance is not straightforward and has proven to be challenging. We propose a rather easy yet effective defense based on backdoor attacks to remove private information, such as names and faces of individuals, from vision-language models by fine-tuning them for only a few minutes instead of re-training them from scratch. Specifically, by strategically inserting backdoors into text encoders, we align the embeddings of sensitive phrases with those of neutral terms-"a person" instead of the person's actual name. For image encoders, we map individuals' embeddings to be removed from the model to a universal, anonymous embedding. The results of our extensive experimental evaluation demonstrate the effectiveness of our backdoor-based defense on CLIP by assessing its performance using a specialized privacy attack for zero-shot classifiers. Our approach provides a new "dual-use" perspective on backdoor attacks and presents a promising avenue to enhance the privacy of individuals within models trained on uncurated web-scraped data.

摘要: 大型人工智能模型的激增引发了人们对隐私的严重担忧。这些模型针对未经管理的、往往是敏感的网络数据进行培训。其中一个令人担忧的问题是，攻击者可以使用隐私攻击来提取有关训练数据的信息。不幸的是，在不牺牲性能的情况下从模型中删除特定信息的任务并不简单，而且已被证明是具有挑战性的。我们提出了一种基于后门攻击的相当简单但有效的防御方法，通过仅对视觉语言模型进行几分钟的微调来删除私人信息，如个人的姓名和面孔，而不是从头开始重新训练它们。具体地说，通过有策略地在文本编码器中插入后门，我们将敏感短语的嵌入与中性术语的嵌入--“人”而不是人的实际姓名--对齐。对于图像编码器，我们将从模型中移除的个人嵌入映射到通用的匿名嵌入。我们广泛的实验评估结果证明了我们的基于后门的CLIP防御的有效性，通过使用专门的针对零射击分类器的隐私攻击来评估其性能。我们的方法为后门攻击提供了一种新的“两用”视角，并提供了一种在未经管理的网络抓取数据的培训模型中增强个人隐私的有前景的途径。



## **32. Efficient Generation of Targeted and Transferable Adversarial Examples for Vision-Language Models Via Diffusion Models**

通过扩散模型高效生成视觉语言模型的有针对性且可转移的对抗示例 cs.CV

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2404.10335v3) [paper-pdf](http://arxiv.org/pdf/2404.10335v3)

**Authors**: Qi Guo, Shanmin Pang, Xiaojun Jia, Yang Liu, Qing Guo

**Abstract**: Adversarial attacks, particularly \textbf{targeted} transfer-based attacks, can be used to assess the adversarial robustness of large visual-language models (VLMs), allowing for a more thorough examination of potential security flaws before deployment. However, previous transfer-based adversarial attacks incur high costs due to high iteration counts and complex method structure. Furthermore, due to the unnaturalness of adversarial semantics, the generated adversarial examples have low transferability. These issues limit the utility of existing methods for assessing robustness. To address these issues, we propose AdvDiffVLM, which uses diffusion models to generate natural, unrestricted and targeted adversarial examples via score matching. Specifically, AdvDiffVLM uses Adaptive Ensemble Gradient Estimation to modify the score during the diffusion model's reverse generation process, ensuring that the produced adversarial examples have natural adversarial targeted semantics, which improves their transferability. Simultaneously, to improve the quality of adversarial examples, we use the GradCAM-guided Mask method to disperse adversarial semantics throughout the image rather than concentrating them in a single area. Finally, AdvDiffVLM embeds more target semantics into adversarial examples after multiple iterations. Experimental results show that our method generates adversarial examples 5x to 10x faster than state-of-the-art transfer-based adversarial attacks while maintaining higher quality adversarial examples. Furthermore, compared to previous transfer-based adversarial attacks, the adversarial examples generated by our method have better transferability. Notably, AdvDiffVLM can successfully attack a variety of commercial VLMs in a black-box environment, including GPT-4V.

摘要: 对抗性攻击，特别是基于传输的对抗性攻击，可用于评估大型视觉语言模型(VLM)的对抗性健壮性，从而允许在部署之前更彻底地检查潜在的安全漏洞。然而，以往基于转移的对抗性攻击由于迭代次数多、方法结构复杂，代价较高。此外，由于对抗性语义的非自然性，生成的对抗性实例可转移性较低。这些问题限制了现有稳健性评估方法的实用性。为了解决这些问题，我们提出了AdvDiffVLM，它使用扩散模型通过得分匹配来生成自然的、不受限制的和有针对性的对抗性实例。具体地说，AdvDiffVLM在扩散模型的反向生成过程中使用自适应集成梯度估计来修改分数，确保生成的对抗性实例具有自然对抗性目标语义，从而提高了它们的可转移性。同时，为了提高对抗性实例的质量，我们使用了GradCAM引导的掩码方法，将对抗性语义分散在整个图像中，而不是将它们集中在单个区域。最后，在多次迭代后，AdvDiffVLM将更多的目标语义嵌入到对抗性实例中。实验结果表明，在保持较高质量的对抗性实例的同时，我们的方法生成对抗性实例的速度比最新的基于传输的对抗性攻击快5倍到10倍。此外，与以往基于转移的对抗性攻击相比，该方法生成的对抗性实例具有更好的可转移性。值得注意的是，AdvDiffVLM可以在黑盒环境中成功攻击各种商业VLM，包括GPT-4V。



## **33. Figure it Out: Analyzing-based Jailbreak Attack on Large Language Models**

弄清楚：基于分析的对大型语言模型的越狱攻击 cs.CR

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2407.16205v1) [paper-pdf](http://arxiv.org/pdf/2407.16205v1)

**Authors**: Shi Lin, Rongchang Li, Xun Wang, Changting Lin, Wenpeng Xing, Meng Han

**Abstract**: The rapid development of Large Language Models (LLMs) has brought remarkable generative capabilities across diverse tasks. However, despite the impressive achievements, these models still have numerous security vulnerabilities, particularly when faced with jailbreak attacks. Therefore, by investigating jailbreak attacks, we can uncover hidden weaknesses in LLMs and guide us in developing more robust defense mechanisms to fortify their security. In this paper, we further explore the boundary of jailbreak attacks on LLMs and propose Analyzing-based Jailbreak (ABJ). This effective jailbreak attack method takes advantage of LLMs' growing analyzing and reasoning capability and reveals their underlying vulnerabilities when facing analysis-based tasks. We conduct a detailed evaluation of ABJ across various open-source and closed-source LLMs, which achieves 94.8% Attack Success Rate (ASR) and 1.06 Attack Efficiency (AE) on GPT-4-turbo-0409, demonstrating state-of-the-art attack effectiveness and efficiency. Our research highlights the importance of prioritizing and enhancing the safety of LLMs to mitigate the risks of misuse.

摘要: 大型语言模型(LLM)的快速发展带来了跨越各种任务的非凡的生成能力。然而，尽管取得了令人印象深刻的成就，这些模型仍然存在许多安全漏洞，特别是在面临越狱攻击时。因此，通过调查越狱攻击，我们可以发现LLMS中隐藏的弱点，并指导我们开发更强大的防御机制来加强它们的安全。本文进一步探讨了LLMS越狱攻击的边界，提出了基于分析的越狱攻击(ABJ)。这种有效的越狱攻击方法利用了LLMS日益增长的分析和推理能力，并在面对基于分析的任务时揭示了它们潜在的漏洞。我们对ABJ在各种开源和闭源LLMS上进行了详细的评估，在GPT-4-TURBO-0409上达到了94.8%的攻击成功率(ASR)和1.06的攻击效率(AE)，展示了最先进的攻击效果和效率。我们的研究强调了优先考虑和加强低密度脂蛋白的安全性，以减少误用风险的重要性。



## **34. Robust Privacy Amidst Innovation with Large Language Models Through a Critical Assessment of the Risks**

通过对风险的批判性评估，在大型语言模型的创新中实现稳健的隐私 cs.CL

13 pages, 4 figures, 1 table, 1 supplementary, under review

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2407.16166v1) [paper-pdf](http://arxiv.org/pdf/2407.16166v1)

**Authors**: Yao-Shun Chuang, Atiquer Rahman Sarkar, Noman Mohammed, Xiaoqian Jiang

**Abstract**: This study examines integrating EHRs and NLP with large language models (LLMs) to improve healthcare data management and patient care. It focuses on using advanced models to create secure, HIPAA-compliant synthetic patient notes for biomedical research. The study used de-identified and re-identified MIMIC III datasets with GPT-3.5, GPT-4, and Mistral 7B to generate synthetic notes. Text generation employed templates and keyword extraction for contextually relevant notes, with one-shot generation for comparison. Privacy assessment checked PHI occurrence, while text utility was tested using an ICD-9 coding task. Text quality was evaluated with ROUGE and cosine similarity metrics to measure semantic similarity with source notes. Analysis of PHI occurrence and text utility via the ICD-9 coding task showed that the keyword-based method had low risk and good performance. One-shot generation showed the highest PHI exposure and PHI co-occurrence, especially in geographic location and date categories. The Normalized One-shot method achieved the highest classification accuracy. Privacy analysis revealed a critical balance between data utility and privacy protection, influencing future data use and sharing. Re-identified data consistently outperformed de-identified data. This study demonstrates the effectiveness of keyword-based methods in generating privacy-protecting synthetic clinical notes that retain data usability, potentially transforming clinical data-sharing practices. The superior performance of re-identified over de-identified data suggests a shift towards methods that enhance utility and privacy by using dummy PHIs to perplex privacy attacks.

摘要: 这项研究考察了将EHR和NLP与大型语言模型(LLM)相结合，以改进医疗数据管理和患者护理。它专注于使用高级模型创建安全的、符合HIPAA标准的合成患者笔记，用于生物医学研究。这项研究使用了GPT-3.5、GPT-4和西风7B的去识别和重新识别的MIMIC III数据集来生成合成音符。文本生成使用模板和上下文相关笔记的关键字提取，并使用一次生成进行比较。隐私评估检查了PHI的发生，而文本实用程序则使用ICD-9编码任务进行了测试。使用Rouge和Cosine相似性度量来评价文本质量，以衡量与源注释的语义相似性。通过ICD-9编码任务对PHI发生情况和文本效用的分析表明，基于关键字的方法风险低，性能好。一次发生显示出最高的PHI暴露和PHI共同出现，特别是在地理位置和日期类别。归一化一次法达到了最高的分类精度。隐私分析揭示了数据效用和隐私保护之间的关键平衡，影响了未来数据的使用和共享。重新识别的数据始终优于未识别的数据。这项研究证明了基于关键字的方法在生成保护隐私的合成临床笔记方面的有效性，这些合成笔记保留了数据的可用性，潜在地改变了临床数据共享的做法。重新识别的数据优于未识别的数据，这表明通过使用虚拟PHI来困扰隐私攻击来增强实用性和隐私的方法发生了转变。



## **35. Flooding Spread of Manipulated Knowledge in LLM-Based Multi-Agent Communities**

基于LLM的多智能体社区中操纵知识的泛滥传播 cs.CL

18 Pages, working in progress

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2407.07791v2) [paper-pdf](http://arxiv.org/pdf/2407.07791v2)

**Authors**: Tianjie Ju, Yiting Wang, Xinbei Ma, Pengzhou Cheng, Haodong Zhao, Yulong Wang, Lifeng Liu, Jian Xie, Zhuosheng Zhang, Gongshen Liu

**Abstract**: The rapid adoption of large language models (LLMs) in multi-agent systems has highlighted their impressive capabilities in various applications, such as collaborative problem-solving and autonomous negotiation. However, the security implications of these LLM-based multi-agent systems have not been thoroughly investigated, particularly concerning the spread of manipulated knowledge. In this paper, we investigate this critical issue by constructing a detailed threat model and a comprehensive simulation environment that mirrors real-world multi-agent deployments in a trusted platform. Subsequently, we propose a novel two-stage attack method involving Persuasiveness Injection and Manipulated Knowledge Injection to systematically explore the potential for manipulated knowledge (i.e., counterfactual and toxic knowledge) spread without explicit prompt manipulation.   Our method leverages the inherent vulnerabilities of LLMs in handling world knowledge, which can be exploited by attackers to unconsciously spread fabricated information. Through extensive experiments, we demonstrate that our attack method can successfully induce LLM-based agents to spread both counterfactual and toxic knowledge without degrading their foundational capabilities during agent communication. Furthermore, we show that these manipulations can persist through popular retrieval-augmented generation frameworks, where several benign agents store and retrieve manipulated chat histories for future interactions. This persistence indicates that even after the interaction has ended, the benign agents may continue to be influenced by manipulated knowledge. Our findings reveal significant security risks in LLM-based multi-agent systems, emphasizing the imperative need for robust defenses against manipulated knowledge spread, such as introducing ``guardian'' agents and advanced fact-checking tools.

摘要: 大型语言模型在多智能体系统中的迅速应用凸显了其在协作问题求解、自主谈判等方面的应用能力。然而，这些基于LLM的多智能体系统的安全含义还没有得到彻底的研究，特别是关于被操纵的知识的传播。在本文中，我们通过构建一个详细的威胁模型和一个全面的模拟环境来研究这一关键问题，该环境反映了可信平台中真实世界的多代理部署。随后，我们提出了一种新的两阶段攻击方法，包括说服力注入和被操纵的知识注入，以系统地探索被操纵的知识(即反事实和有毒知识)在没有明确的即时操纵的情况下传播的可能性。我们的方法利用了LLMS在处理世界知识方面的固有漏洞，攻击者可以利用这些漏洞来不知不觉地传播伪造的信息。通过大量的实验，我们证明了我们的攻击方法可以成功地诱导基于LLM的代理传播反事实和有毒知识，而不会降低其在代理通信中的基础能力。此外，我们还表明，这些操作可以在流行的检索增强生成框架中持续存在，在这些框架中，几个良性代理存储和检索被操纵的聊天历史，以便将来进行交互。这种持久性表明，即使在互动结束后，良性代理人仍可能继续受到操纵知识的影响。我们的发现揭示了基于LLM的多代理系统中的重大安全风险，强调了对被操纵的知识传播采取强有力的防御措施的迫切需要，例如引入“监护人”代理和先进的事实核查工具。



## **36. The Shadow of Fraud: The Emerging Danger of AI-powered Social Engineering and its Possible Cure**

欺诈的阴影：人工智能驱动的社会工程的新危险及其可能的治疗方法 cs.CR

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.15912v1) [paper-pdf](http://arxiv.org/pdf/2407.15912v1)

**Authors**: Jingru Yu, Yi Yu, Xuhong Wang, Yilun Lin, Manzhi Yang, Yu Qiao, Fei-Yue Wang

**Abstract**: Social engineering (SE) attacks remain a significant threat to both individuals and organizations. The advancement of Artificial Intelligence (AI), including diffusion models and large language models (LLMs), has potentially intensified these threats by enabling more personalized and convincing attacks. This survey paper categorizes SE attack mechanisms, analyzes their evolution, and explores methods for measuring these threats. It highlights the challenges in raising awareness about the risks of AI-enhanced SE attacks and offers insights into developing proactive and adaptable defense strategies. Additionally, we introduce a categorization of the evolving nature of AI-powered social engineering attacks into "3E phases": Enlarging, wherein the magnitude of attacks expands through the leverage of digital media; Enriching, introducing novel attack vectors and techniques; and Emerging, signifying the advent of novel threats and methods. Moreover, we emphasize the necessity for a robust framework to assess the risk of AI-powered SE attacks. By identifying and addressing gaps in existing research, we aim to guide future studies and encourage the development of more effective defenses against the growing threat of AI-powered social engineering.

摘要: 社会工程(SE)攻击仍然是对个人和组织的重大威胁。人工智能(AI)的发展，包括扩散模型和大型语言模型(LLM)，通过实现更个性化和更有说服力的攻击，潜在地加剧了这些威胁。本调查报告对SE攻击机制进行了分类，分析了它们的演变，并探索了衡量这些威胁的方法。它强调了在提高对人工智能增强的SE攻击风险的认识方面的挑战，并为开发主动和自适应的防御战略提供了见解。此外，我们将人工智能支持的社会工程攻击的演变性质分类为“3E阶段”：扩大，其中攻击的规模通过数字媒体的杠杆作用扩大；丰富，引入新的攻击载体和技术；以及出现，标志着新威胁和方法的出现。此外，我们强调需要一个强大的框架来评估人工智能支持的SE攻击的风险。通过识别和解决现有研究中的差距，我们的目标是指导未来的研究，并鼓励开发更有效的防御措施，以应对人工智能驱动的社会工程日益增长的威胁。



## **37. Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts**

彩虹团队：开放式一代的多元化对抗预言 cs.CL

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2402.16822v2) [paper-pdf](http://arxiv.org/pdf/2402.16822v2)

**Authors**: Mikayel Samvelyan, Sharath Chandra Raparthy, Andrei Lupu, Eric Hambro, Aram H. Markosyan, Manish Bhatt, Yuning Mao, Minqi Jiang, Jack Parker-Holder, Jakob Foerster, Tim Rocktäschel, Roberta Raileanu

**Abstract**: As large language models (LLMs) become increasingly prevalent across many real-world applications, understanding and enhancing their robustness to adversarial attacks is of paramount importance. Existing methods for identifying adversarial prompts tend to focus on specific domains, lack diversity, or require extensive human annotations. To address these limitations, we present Rainbow Teaming, a novel black-box approach for producing a diverse collection of adversarial prompts. Rainbow Teaming casts adversarial prompt generation as a quality-diversity problem, and uses open-ended search to generate prompts that are both effective and diverse. Focusing on the safety domain, we use Rainbow Teaming to target various state-of-the-art LLMs, including the Llama 2 and Llama 3 models. Our approach reveals hundreds of effective adversarial prompts, with an attack success rate exceeding 90% across all tested models. Furthermore, we demonstrate that fine-tuning models with synthetic data generated by the Rainbow Teaming method significantly enhances their safety without sacrificing general performance or helpfulness. We additionally explore the versatility of Rainbow Teaming by applying it to question answering and cybersecurity, showcasing its potential to drive robust open-ended self-improvement in a wide range of applications.

摘要: 随着大型语言模型(LLM)在许多真实世界的应用中变得越来越普遍，理解和增强它们对对手攻击的健壮性是至关重要的。现有的识别对抗性提示的方法往往集中在特定的领域，缺乏多样性，或者需要大量的人工注释。为了解决这些局限性，我们提出了彩虹分组，这是一种新的黑盒方法，用于产生多样化的对抗性提示集合。彩虹团队将敌意提示生成视为质量多样性问题，并使用开放式搜索来生成既有效又多样化的提示。专注于安全领域，我们使用彩虹团队瞄准各种最先进的LLM，包括Llama 2和Llama 3型号。我们的方法揭示了数百个有效的对抗性提示，在所有测试模型上的攻击成功率超过90%。此外，我们证明了使用彩虹组合方法生成的合成数据对模型进行微调显著增强了它们的安全性，而不会牺牲总体性能或帮助。我们还探讨了彩虹团队的多功能性，将其应用于问题回答和网络安全，展示了其在广泛应用中推动强大的开放式自我改进的潜力。



## **38. Adversarial Style Augmentation via Large Language Model for Robust Fake News Detection**

通过大语言模型进行对抗风格增强以实现稳健的假新闻检测 cs.CL

8 pages

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2406.11260v2) [paper-pdf](http://arxiv.org/pdf/2406.11260v2)

**Authors**: Sungwon Park, Sungwon Han, Meeyoung Cha

**Abstract**: The spread of fake news negatively impacts individuals and is regarded as a significant social challenge that needs to be addressed. A number of algorithmic and insightful features have been identified for detecting fake news. However, with the recent LLMs and their advanced generation capabilities, many of the detectable features (e.g., style-conversion attacks) can be altered, making it more challenging to distinguish from real news. This study proposes adversarial style augmentation, AdStyle, to train a fake news detector that remains robust against various style-conversion attacks. Our model's key mechanism is the careful use of LLMs to automatically generate a diverse yet coherent range of style-conversion attack prompts. This improves the generation of prompts that are particularly difficult for the detector to handle. Experiments show that our augmentation strategy improves robustness and detection performance when tested on fake news benchmark datasets.

摘要: 假新闻的传播对个人产生负面影响，被视为需要解决的重大社会挑战。已经确定了许多算法和有洞察力的功能来检测假新闻。然而，随着最近的LLM及其先进一代能力，许多可检测的特征（例如，风格转换攻击）可以被更改，使其与真实新闻区分起来更具挑战性。这项研究提出了对抗性风格增强AdStyle来训练一个假新闻检测器，该检测器在对抗各种风格转换攻击时保持稳健。我们模型的关键机制是仔细使用LLM来自动生成多样化但连贯的风格转换攻击提示。这改善了检测器特别难以处理的提示的生成。实验表明，当在假新闻基准数据集上进行测试时，我们的增强策略提高了鲁棒性和检测性能。



## **39. Targeted Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs**

有针对性的隐性对抗培训提高了LLM对持续有害行为的稳健性 cs.LG

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.15549v1) [paper-pdf](http://arxiv.org/pdf/2407.15549v1)

**Authors**: Abhay Sheshadri, Aidan Ewart, Phillip Guo, Aengus Lynch, Cindy Wu, Vivek Hebbar, Henry Sleight, Asa Cooper Stickland, Ethan Perez, Dylan Hadfield-Menell, Stephen Casper

**Abstract**: Large language models (LLMs) can often be made to behave in undesirable ways that they are explicitly fine-tuned not to. For example, the LLM red-teaming literature has produced a wide variety of `jailbreaking' techniques to elicit harmful text from models that were fine-tuned to be harmless. Recent work on red-teaming, model editing, and interpretability suggests that this challenge stems from how (adversarial) fine-tuning largely serves to suppress rather than remove undesirable capabilities from LLMs. Prior work has introduced latent adversarial training (LAT) as a way to improve robustness to broad classes of failures. These prior works have considered untargeted latent space attacks where the adversary perturbs latent activations to maximize loss on examples of desirable behavior. Untargeted LAT can provide a generic type of robustness but does not leverage information about specific failure modes. Here, we experiment with targeted LAT where the adversary seeks to minimize loss on a specific competing task. We find that it can augment a wide variety of state-of-the-art methods. First, we use targeted LAT to improve robustness to jailbreaks, outperforming a strong R2D2 baseline with orders of magnitude less compute. Second, we use it to more effectively remove backdoors with no knowledge of the trigger. Finally, we use it to more effectively unlearn knowledge for specific undesirable tasks in a way that is also more robust to re-learning. Overall, our results suggest that targeted LAT can be an effective tool for defending against harmful behaviors from LLMs.

摘要: 大型语言模型(LLM)通常会以不受欢迎的方式运行，因此它们被明确微调为不以这种方式运行。例如，伦敦大学法学院的红队文学创作了各种各样的“越狱”技术，从经过微调的无害模特那里引出有害文本。最近在红团队、模型编辑和可解释性方面的工作表明，这一挑战源于(对抗性的)微调如何在很大程度上抑制而不是消除LLM中不受欢迎的能力。以前的工作已经引入了潜在的对手训练(LAT)，作为一种提高对广泛类别的故障的稳健性的方式。这些先前的工作考虑了无目标的潜在空间攻击，即对手扰乱潜在激活，以最大限度地减少期望行为的示例损失。非定向LAT可以提供一般类型的健壮性，但不利用有关特定故障模式的信息。在这里，我们实验有针对性的LAT，其中对手试图将特定竞争任务的损失降至最低。我们发现，它可以增加各种最先进的方法。首先，我们使用有针对性的LAT来提高对越狱的健壮性，性能优于强大的R2D2基线，计算量少了几个数量级。其次，我们使用它来更有效地删除后门，而不知道触发器。最后，我们使用它来更有效地忘记特定不受欢迎的任务的知识，这种方式也更适合重新学习。总体而言，我们的结果表明，有针对性的LAT可以成为防御LLM有害行为的有效工具。



## **40. TAPI: Towards Target-Specific and Adversarial Prompt Injection against Code LLMs**

TAPI：针对代码LLM的目标特定和对抗性即时注入 cs.CR

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.09164v3) [paper-pdf](http://arxiv.org/pdf/2407.09164v3)

**Authors**: Yuchen Yang, Hongwei Yao, Bingrun Yang, Yiling He, Yiming Li, Tianwei Zhang, Zhan Qin, Kui Ren

**Abstract**: Recently, code-oriented large language models (Code LLMs) have been widely and successfully used to simplify and facilitate code programming. With these tools, developers can easily generate desired complete functional codes based on incomplete code and natural language prompts. However, a few pioneering works revealed that these Code LLMs are also vulnerable, e.g., against backdoor and adversarial attacks. The former could induce LLMs to respond to triggers to insert malicious code snippets by poisoning the training data or model parameters, while the latter can craft malicious adversarial input codes to reduce the quality of generated codes. However, both attack methods have underlying limitations: backdoor attacks rely on controlling the model training process, while adversarial attacks struggle with fulfilling specific malicious purposes.   To inherit the advantages of both backdoor and adversarial attacks, this paper proposes a new attack paradigm, i.e., target-specific and adversarial prompt injection (TAPI), against Code LLMs. TAPI generates unreadable comments containing information about malicious instructions and hides them as triggers in the external source code. When users exploit Code LLMs to complete codes containing the trigger, the models will generate attacker-specified malicious code snippets at specific locations. We evaluate our TAPI attack on four representative LLMs under three representative malicious objectives and seven cases. The results show that our method is highly threatening (achieving an attack success rate of up to 98.3%) and stealthy (saving an average of 53.1% of tokens in the trigger design). In particular, we successfully attack some famous deployed code completion integrated applications, including CodeGeex and Github Copilot. This further confirms the realistic threat of our attack.

摘要: 最近，面向代码的大型语言模型(Code LLM)已被广泛并成功地用于简化和促进代码编程。使用这些工具，开发人员可以根据不完整的代码和自然语言提示轻松生成所需的完整功能代码。然而，一些开创性的工作表明，这些代码LLM也容易受到攻击，例如，抵御后门和对手攻击。前者可以通过毒化训练数据或模型参数来诱导LLMS响应插入恶意代码片段的触发器，而后者可以手工创建恶意输入代码来降低生成代码的质量。然而，这两种攻击方法都有潜在的局限性：后门攻击依赖于控制模型训练过程，而对抗性攻击则难以实现特定的恶意目的。为了继承后门攻击和对抗性攻击的优点，提出了一种新的针对Code LLMS的攻击范式，即目标特定和对抗性提示注入(TAPI)。TAPI生成不可读的注释，其中包含有关恶意指令的信息，并将它们作为触发器隐藏在外部源代码中。当用户利用Code LLMS来完成包含触发器的代码时，模型将在特定位置生成攻击者指定的恶意代码片段。我们在三个典型的恶意目标和七个案例下评估了我们的TAPI攻击对四个有代表性的LLM的攻击。结果表明，该方法具有很高的威胁性(攻击成功率高达98.3%)和隐蔽性(在触发器设计中平均节省53.1%的令牌)。特别是，我们成功地攻击了一些著名的部署代码完成集成应用程序，包括CodeGeex和Github Copilot。这进一步证实了我们攻击的现实威胁。



## **41. Imposter.AI: Adversarial Attacks with Hidden Intentions towards Aligned Large Language Models**

冒名顶替。AI：针对对齐大型语言模型的具有隐藏意图的对抗攻击 cs.CL

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.15399v1) [paper-pdf](http://arxiv.org/pdf/2407.15399v1)

**Authors**: Xiao Liu, Liangzhi Li, Tong Xiang, Fuying Ye, Lu Wei, Wangyue Li, Noa Garcia

**Abstract**: With the development of large language models (LLMs) like ChatGPT, both their vast applications and potential vulnerabilities have come to the forefront. While developers have integrated multiple safety mechanisms to mitigate their misuse, a risk remains, particularly when models encounter adversarial inputs. This study unveils an attack mechanism that capitalizes on human conversation strategies to extract harmful information from LLMs. We delineate three pivotal strategies: (i) decomposing malicious questions into seemingly innocent sub-questions; (ii) rewriting overtly malicious questions into more covert, benign-sounding ones; (iii) enhancing the harmfulness of responses by prompting models for illustrative examples. Unlike conventional methods that target explicit malicious responses, our approach delves deeper into the nature of the information provided in responses. Through our experiments conducted on GPT-3.5-turbo, GPT-4, and Llama2, our method has demonstrated a marked efficacy compared to conventional attack methods. In summary, this work introduces a novel attack method that outperforms previous approaches, raising an important question: How to discern whether the ultimate intent in a dialogue is malicious?

摘要: 随着像ChatGPT这样的大型语言模型(LLM)的发展，它们的巨大应用和潜在的漏洞都已经浮出水面。虽然开发人员已经集成了多种安全机制来减少它们的滥用，但风险仍然存在，特别是当模型遇到敌对输入时。这项研究揭示了一种利用人类对话策略从LLMS中提取有害信息的攻击机制。我们描述了三个关键策略：(I)将恶意问题分解为看似无害的子问题；(Ii)将公开的恶意问题重写为更隐蔽、听起来更温和的问题；(Iii)通过提示示例模型来增强回答的危害性。与针对显式恶意响应的传统方法不同，我们的方法更深入地挖掘响应中提供的信息的性质。通过我们在GPT-3.5-Turbo、GPT-4和Llama2上的实验，我们的方法比传统的攻击方法表现出了显著的效果。总之，这项工作引入了一种新的攻击方法，其性能优于以前的方法，提出了一个重要的问题：如何识别对话中的最终意图是否为恶意的？



## **42. Advancing TTP Analysis: Harnessing the Power of Large Language Models with Retrieval Augmented Generation**

推进TTP分析：利用检索增强生成来利用大型语言模型的力量 cs.CR

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2401.00280v3) [paper-pdf](http://arxiv.org/pdf/2401.00280v3)

**Authors**: Reza Fayyazi, Rozhina Taghdimi, Shanchieh Jay Yang

**Abstract**: Tactics, Techniques, and Procedures (TTPs) outline the methods attackers use to exploit vulnerabilities. The interpretation of TTPs in the MITRE ATT&CK framework can be challenging for cybersecurity practitioners due to presumed expertise and complex dependencies. Meanwhile, advancements with Large Language Models (LLMs) have led to recent surge in studies exploring its uses in cybersecurity operations. It is, however, unclear how LLMs can be used in an efficient and proper way to provide accurate responses for critical domains such as cybersecurity. This leads us to investigate how to better use two types of LLMs: small-scale encoder-only (e.g., RoBERTa) and larger decoder-only (e.g., GPT-3.5) LLMs to comprehend and summarize TTPs with the intended purposes (i.e., tactics) of a cyberattack procedure. This work studies and compares the uses of supervised fine-tuning (SFT) of encoder-only LLMs vs. Retrieval Augmented Generation (RAG) for decoder-only LLMs (without fine-tuning). Both SFT and RAG techniques presumably enhance the LLMs with relevant contexts for each cyberattack procedure. Our studies show decoder-only LLMs with RAG achieves better performance than encoder-only models with SFT, particularly when directly relevant context is extracted by RAG. The decoder-only results could suffer low `Precision' while achieving high `Recall'. Our findings further highlight a counter-intuitive observation that more generic prompts tend to yield better predictions of cyberattack tactics than those that are more specifically tailored.

摘要: 战术、技术和过程(TTP)概述了攻击者用来利用漏洞的方法。由于假定的专业知识和复杂的依赖关系，MITRE ATT&CK框架中对TTP的解释可能会对网络安全从业者构成挑战。与此同时，大型语言模型(LLM)的进步导致了最近探索其在网络安全行动中的应用的研究激增。然而，目前尚不清楚如何以有效和适当的方式使用LLMS，为网络安全等关键领域提供准确的响应。这导致我们研究如何更好地使用两种类型的LLP：仅限小规模编码器(例如Roberta)和仅限解码器(例如GPT-3.5)的LLM来理解和总结具有网络攻击过程的预期目的(即战术)的TTP。这项工作研究和比较了仅编码器LLMS的监督微调(SFT)和仅解码器LLMS(未微调)的检索增强生成(RAG)的使用。SFT和RAG技术都可能通过每个网络攻击过程的相关上下文来增强LLMS。我们的研究表明，只有解码者的RAG模型比只有编码者的SFT模型具有更好的性能，特别是当RAG提取直接相关的上下文时。只有解码器的结果可能会受到低精度的影响，而获得高的‘Recall’。我们的发现进一步突显了一种与直觉相反的观察，即更笼统的提示往往比更具体的提示更能预测网络攻击策略。



## **43. When Do Universal Image Jailbreaks Transfer Between Vision-Language Models?**

普遍形象越狱何时在视觉语言模型之间转移？ cs.CL

**SubmitDate**: 2024-07-21    [abs](http://arxiv.org/abs/2407.15211v1) [paper-pdf](http://arxiv.org/pdf/2407.15211v1)

**Authors**: Rylan Schaeffer, Dan Valentine, Luke Bailey, James Chua, Cristóbal Eyzaguirre, Zane Durante, Joe Benton, Brando Miranda, Henry Sleight, John Hughes, Rajashree Agrawal, Mrinank Sharma, Scott Emmons, Sanmi Koyejo, Ethan Perez

**Abstract**: The integration of new modalities into frontier AI systems offers exciting capabilities, but also increases the possibility such systems can be adversarially manipulated in undesirable ways. In this work, we focus on a popular class of vision-language models (VLMs) that generate text outputs conditioned on visual and textual inputs. We conducted a large-scale empirical study to assess the transferability of gradient-based universal image "jailbreaks" using a diverse set of over 40 open-parameter VLMs, including 18 new VLMs that we publicly release. Overall, we find that transferable gradient-based image jailbreaks are extremely difficult to obtain. When an image jailbreak is optimized against a single VLM or against an ensemble of VLMs, the jailbreak successfully jailbreaks the attacked VLM(s), but exhibits little-to-no transfer to any other VLMs; transfer is not affected by whether the attacked and target VLMs possess matching vision backbones or language models, whether the language model underwent instruction-following and/or safety-alignment training, or many other factors. Only two settings display partially successful transfer: between identically-pretrained and identically-initialized VLMs with slightly different VLM training data, and between different training checkpoints of a single VLM. Leveraging these results, we then demonstrate that transfer can be significantly improved against a specific target VLM by attacking larger ensembles of "highly-similar" VLMs. These results stand in stark contrast to existing evidence of universal and transferable text jailbreaks against language models and transferable adversarial attacks against image classifiers, suggesting that VLMs may be more robust to gradient-based transfer attacks.

摘要: 将新的模式集成到前沿人工智能系统中提供了令人兴奋的能力，但也增加了此类系统被以不受欢迎的方式进行相反操作的可能性。在这项工作中，我们专注于一类流行的视觉语言模型(VLM)，它们生成以视觉和文本输入为条件的文本输出。我们进行了一项大规模的实证研究，以评估基于梯度的通用图像“越狱”的可转移性，使用了一组超过40个开放参数的VLM，其中包括我们公开发布的18个新的VLM。总体而言，我们发现基于梯度的可转移越狱图像非常难以获得。当针对单个VLM或一组VLM优化图像越狱时，越狱成功地越狱了被攻击的VLM(S)，但很少或根本不转移到任何其他VLM；转移不受攻击和目标VLM是否具有匹配的视觉主干或语言模型、语言模型是否经过指令遵循和/或安全对齐培训或许多其他因素的影响。只有两个设置显示部分成功的传输：在具有略微不同的VLM训练数据的相同预训练和相同初始化的VLM之间，以及在单个VLM的不同训练检查点之间。利用这些结果，我们随后证明了针对特定目标VLM的传输可以通过攻击更大的“高度相似的”VLM集合来显著改进。这些结果与针对语言模型的普遍和可传输的文本越狱以及针对图像分类器的可传输的对抗性攻击的现有证据形成了鲜明对比，这表明VLM可能对基于梯度的传输攻击更健壮。



## **44. Trading Devil Final: Backdoor attack via Stock market and Bayesian Optimization**

交易魔鬼决赛：通过股市和Bayesian优化进行后门攻击 cs.LG

jumps-Diffusion and stock market: Better quantify uncertainty in  financial simulations

**SubmitDate**: 2024-07-21    [abs](http://arxiv.org/abs/2407.14573v1) [paper-pdf](http://arxiv.org/pdf/2407.14573v1)

**Authors**: Orson Mengara

**Abstract**: Since the advent of generative artificial intelligence, every company and researcher has been rushing to develop their own generative models, whether commercial or not. Given the large number of users of these powerful new tools, there is currently no intrinsically verifiable way to explain from the ground up what happens when LLMs (large language models) learn. For example, those based on automatic speech recognition systems, which have to rely on huge and astronomical amounts of data collected from all over the web to produce fast and efficient results, In this article, we develop a backdoor attack called MarketBackFinal 2.0, based on acoustic data poisoning, MarketBackFinal 2.0 is mainly based on modern stock market models. In order to show the possible vulnerabilities of speech-based transformers that may rely on LLMs.

摘要: 自生成人工智能出现以来，每家公司和研究人员都在争先恐后地开发自己的生成模型，无论是否商业化。鉴于这些强大的新工具的大量用户，目前还没有本质上可验证的方法来从头解释LLM（大型语言模型）学习时会发生什么。例如，那些基于自动语音识别系统的系统，它们必须依赖于从整个网络收集的大量数据来产生快速有效的结果，在本文中，我们开发了一种名为MarketBackFinal 2.0的后门攻击，基于声学数据中毒，MarketBackFinal 2.0主要基于现代股市模型。为了显示可能依赖LLM的基于语音的转换器可能存在的漏洞。



## **45. Arondight: Red Teaming Large Vision Language Models with Auto-generated Multi-modal Jailbreak Prompts**

Arondight：Red将大视觉语言模型与自动生成的多模式越狱脚本结合起来 cs.LG

To be published in ACM MM 2024

**SubmitDate**: 2024-07-21    [abs](http://arxiv.org/abs/2407.15050v1) [paper-pdf](http://arxiv.org/pdf/2407.15050v1)

**Authors**: Yi Liu, Chengjun Cai, Xiaoli Zhang, Xingliang Yuan, Cong Wang

**Abstract**: Large Vision Language Models (VLMs) extend and enhance the perceptual abilities of Large Language Models (LLMs). Despite offering new possibilities for LLM applications, these advancements raise significant security and ethical concerns, particularly regarding the generation of harmful content. While LLMs have undergone extensive security evaluations with the aid of red teaming frameworks, VLMs currently lack a well-developed one. To fill this gap, we introduce Arondight, a standardized red team framework tailored specifically for VLMs. Arondight is dedicated to resolving issues related to the absence of visual modality and inadequate diversity encountered when transitioning existing red teaming methodologies from LLMs to VLMs. Our framework features an automated multi-modal jailbreak attack, wherein visual jailbreak prompts are produced by a red team VLM, and textual prompts are generated by a red team LLM guided by a reinforcement learning agent. To enhance the comprehensiveness of VLM security evaluation, we integrate entropy bonuses and novelty reward metrics. These elements incentivize the RL agent to guide the red team LLM in creating a wider array of diverse and previously unseen test cases. Our evaluation of ten cutting-edge VLMs exposes significant security vulnerabilities, particularly in generating toxic images and aligning multi-modal prompts. In particular, our Arondight achieves an average attack success rate of 84.5\% on GPT-4 in all fourteen prohibited scenarios defined by OpenAI in terms of generating toxic text. For a clearer comparison, we also categorize existing VLMs based on their safety levels and provide corresponding reinforcement recommendations. Our multimodal prompt dataset and red team code will be released after ethics committee approval. CONTENT WARNING: THIS PAPER CONTAINS HARMFUL MODEL RESPONSES.

摘要: 大视觉语言模型扩展并增强了大语言模型的感知能力。尽管为LLM应用提供了新的可能性，但这些进展引发了重大的安全和伦理问题，特别是在有害内容的生成方面。虽然在红色团队框架的帮助下，低成本管理系统已经进行了广泛的安全评估，但目前还缺乏一个完善的安全评估框架。为了填补这一空白，我们引入了Arondiight，这是一个专门为VLM定制的标准化红色团队框架。Arondiight致力于解决在将现有的红色团队方法从LLMS过渡到VLMS时遇到的视觉形态缺失和多样性不足的问题。我们的框架以自动多模式越狱攻击为特色，其中视觉越狱提示由红色团队VLM生成，文本提示由红色团队LLM生成，并由强化学习代理引导。为了增强VLM安全评估的全面性，我们将熵奖金和新颖性奖励指标结合起来。这些元素激励RL代理指导红色团队LLM创建更广泛的多样化和以前未见过的测试用例。我们对10个尖端VLM的评估暴露了严重的安全漏洞，特别是在生成有毒图像和对齐多模式提示方面。特别是，我们的Arondiight在生成有毒文本方面，在OpenAI定义的所有14个禁止场景中，对GPT-4的平均攻击成功率为84.5\%。为了更清楚地进行比较，我们还根据现有的VLM的安全级别对其进行了分类，并提出了相应的加固建议。我们的多模式提示数据集和红色团队代码将在道德委员会批准后发布。内容警告：本文包含有害模型回复。



## **46. Sim-CLIP: Unsupervised Siamese Adversarial Fine-Tuning for Robust and Semantically-Rich Vision-Language Models**

Sim-CLIP：针对稳健且语义丰富的视觉语言模型的无监督Siamese对抗微调 cs.CV

**SubmitDate**: 2024-07-20    [abs](http://arxiv.org/abs/2407.14971v1) [paper-pdf](http://arxiv.org/pdf/2407.14971v1)

**Authors**: Md Zarif Hossain, Ahmed Imteaj

**Abstract**: Vision-language models (VLMs) have achieved significant strides in recent times specially in multimodal tasks, yet they remain susceptible to adversarial attacks on their vision components. To address this, we propose Sim-CLIP, an unsupervised adversarial fine-tuning method that enhances the robustness of the widely-used CLIP vision encoder against such attacks while maintaining semantic richness and specificity. By employing a Siamese architecture with cosine similarity loss, Sim-CLIP learns semantically meaningful and attack-resilient visual representations without requiring large batch sizes or momentum encoders. Our results demonstrate that VLMs enhanced with Sim-CLIP's fine-tuned CLIP encoder exhibit significantly enhanced robustness against adversarial attacks, while preserving semantic meaning of the perturbed images. Notably, Sim-CLIP does not require additional training or fine-tuning of the VLM itself; replacing the original vision encoder with our fine-tuned Sim-CLIP suffices to provide robustness. This work underscores the significance of reinforcing foundational models like CLIP to safeguard the reliability of downstream VLM applications, paving the way for more secure and effective multimodal systems.

摘要: 视觉语言模型近年来取得了长足的进步，特别是在多通道任务中，但它们仍然容易受到视觉部分的敌意攻击。为了解决这一问题，我们提出了SIM-CLIP，这是一种无监督的对抗性微调方法，它在保持语义丰富和特异性的同时，增强了广泛使用的CLIP视觉编码器对此类攻击的健壮性。通过采用具有余弦相似性损失的暹罗体系结构，Sim-Clip无需大批量或动量编码器即可学习语义上有意义的、可抵抗攻击的视觉表示。结果表明，通过Sim-Clip的精细调整的CLIP编码器增强的VLM在保持扰动图像语义的同时，显著增强了对对手攻击的稳健性。值得注意的是，SIM-Clip不需要对VLM本身进行额外的培训或微调；用我们经过微调的SIM-Clip替换原来的视觉编码器就足以提供健壮性。这项工作强调了加强像CLIP这样的基础模型对保障下游VLM应用的可靠性的重要性，为更安全和有效的多式联运系统铺平了道路。



## **47. Operationalizing a Threat Model for Red-Teaming Large Language Models (LLMs)**

为Red-Teaming大型语言模型（LLM）操作威胁模型 cs.CL

Preprint. Under review

**SubmitDate**: 2024-07-20    [abs](http://arxiv.org/abs/2407.14937v1) [paper-pdf](http://arxiv.org/pdf/2407.14937v1)

**Authors**: Apurv Verma, Satyapriya Krishna, Sebastian Gehrmann, Madhavan Seshadri, Anu Pradhan, Tom Ault, Leslie Barrett, David Rabinowitz, John Doucette, NhatHai Phan

**Abstract**: Creating secure and resilient applications with large language models (LLM) requires anticipating, adjusting to, and countering unforeseen threats. Red-teaming has emerged as a critical technique for identifying vulnerabilities in real-world LLM implementations. This paper presents a detailed threat model and provides a systematization of knowledge (SoK) of red-teaming attacks on LLMs. We develop a taxonomy of attacks based on the stages of the LLM development and deployment process and extract various insights from previous research. In addition, we compile methods for defense and practical red-teaming strategies for practitioners. By delineating prominent attack motifs and shedding light on various entry points, this paper provides a framework for improving the security and robustness of LLM-based systems.

摘要: 使用大型语言模型（LLM）创建安全且有弹性的应用程序需要预测、调整和应对不可预见的威胁。红色团队已成为识别现实世界LLM实施中漏洞的关键技术。本文提出了一个详细的威胁模型，并提供了对LLM的红色团队攻击的知识系统化（SoK）。我们根据LLM开发和部署过程的阶段开发攻击分类，并从之前的研究中提取各种见解。此外，我们还为从业者编写了防御方法和实用的红色团队策略。通过描述突出的攻击主题并揭示各种切入点，本文提供了一个框架来提高基于LLM的系统的安全性和稳健性。



## **48. DistillSeq: A Framework for Safety Alignment Testing in Large Language Models using Knowledge Distillation**

DistillSeq：使用知识蒸馏在大型语言模型中进行安全一致测试的框架 cs.SE

**SubmitDate**: 2024-07-20    [abs](http://arxiv.org/abs/2407.10106v3) [paper-pdf](http://arxiv.org/pdf/2407.10106v3)

**Authors**: Mingke Yang, Yuqi Chen, Yi Liu, Ling Shi

**Abstract**: Large Language Models (LLMs) have showcased their remarkable capabilities in diverse domains, encompassing natural language understanding, translation, and even code generation. The potential for LLMs to generate harmful content is a significant concern. This risk necessitates rigorous testing and comprehensive evaluation of LLMs to ensure safe and responsible use. However, extensive testing of LLMs requires substantial computational resources, making it an expensive endeavor. Therefore, exploring cost-saving strategies during the testing phase is crucial to balance the need for thorough evaluation with the constraints of resource availability. To address this, our approach begins by transferring the moderation knowledge from an LLM to a small model. Subsequently, we deploy two distinct strategies for generating malicious queries: one based on a syntax tree approach, and the other leveraging an LLM-based method. Finally, our approach incorporates a sequential filter-test process designed to identify test cases that are prone to eliciting toxic responses. Our research evaluated the efficacy of DistillSeq across four LLMs: GPT-3.5, GPT-4.0, Vicuna-13B, and Llama-13B. In the absence of DistillSeq, the observed attack success rates on these LLMs stood at 31.5% for GPT-3.5, 21.4% for GPT-4.0, 28.3% for Vicuna-13B, and 30.9% for Llama-13B. However, upon the application of DistillSeq, these success rates notably increased to 58.5%, 50.7%, 52.5%, and 54.4%, respectively. This translated to an average escalation in attack success rate by a factor of 93.0% when compared to scenarios without the use of DistillSeq. Such findings highlight the significant enhancement DistillSeq offers in terms of reducing the time and resource investment required for effectively testing LLMs.

摘要: 大型语言模型(LLM)已经在不同的领域展示了它们非凡的能力，包括自然语言理解、翻译，甚至代码生成。低密度脂蛋白产生有害内容的可能性是一个重大关切。这种风险需要对LLMS进行严格的测试和全面评估，以确保安全和负责任地使用。然而，大规模的LLMS测试需要大量的计算资源，这使得它成为一项昂贵的工作。因此，在测试阶段探索节约成本的策略对于平衡彻底评估的需要和资源可用性的限制至关重要。为了解决这个问题，我们的方法首先将适度知识从LLM转移到一个小模型。随后，我们部署了两种不同的策略来生成恶意查询：一种基于语法树方法，另一种利用基于LLM的方法。最后，我们的方法结合了一个顺序的过滤测试过程，旨在识别容易引发有毒反应的测试用例。我们的研究评估了DistillSeq在四种低密度脂蛋白上的疗效：GPT-3.5、GPT-4.0、Vicuna-13B和Llama-13B。在没有DistillSeq的情况下，观察到的对这些LLMS的攻击成功率GPT-3.5为31.5%，GPT-4.0为21.4%，Vicuna-13B为28.3%，Llama-13B为30.9%。然而，在应用DistillSeq后，这些成功率分别显著增加到58.5%、50.7%、52.5%和54.4%。与未使用DistillSeq的情况相比，这意味着攻击成功率平均提升了93.0%。这些发现突出了DistillSeq在减少有效测试LLM所需的时间和资源投资方面所提供的显著增强。



## **49. Retrieval Augmented Generation Integrated Large Language Models in Smart Contract Vulnerability Detection**

智能合同漏洞检测中的检索增强生成集成大型语言模型 cs.CR

17 pages, 3 figures, 4 tables

**SubmitDate**: 2024-07-20    [abs](http://arxiv.org/abs/2407.14838v1) [paper-pdf](http://arxiv.org/pdf/2407.14838v1)

**Authors**: Jeffy Yu

**Abstract**: The rapid growth of Decentralized Finance (DeFi) has been accompanied by substantial financial losses due to smart contract vulnerabilities, underscoring the critical need for effective security auditing. With attacks becoming more frequent, the necessity and demand for auditing services has escalated. This especially creates a financial burden for independent developers and small businesses, who often have limited available funding for these services. Our study builds upon existing frameworks by integrating Retrieval-Augmented Generation (RAG) with large language models (LLMs), specifically employing GPT-4-1106 for its 128k token context window. We construct a vector store of 830 known vulnerable contracts, leveraging Pinecone for vector storage, OpenAI's text-embedding-ada-002 for embeddings, and LangChain to construct the RAG-LLM pipeline. Prompts were designed to provide a binary answer for vulnerability detection. We first test 52 smart contracts 40 times each against a provided vulnerability type, verifying the replicability and consistency of the RAG-LLM. Encouraging results were observed, with a 62.7% success rate in guided detection of vulnerabilities. Second, we challenge the model under a "blind" audit setup, without the vulnerability type provided in the prompt, wherein 219 contracts undergo 40 tests each. This setup evaluates the general vulnerability detection capabilities without hinted context assistance. Under these conditions, a 60.71% success rate was observed. While the results are promising, we still emphasize the need for human auditing at this time. We provide this study as a proof of concept for a cost-effective smart contract auditing process, moving towards democratic access to security.

摘要: 去中心化金融(Defi)的快速增长伴随着由于智能合同漏洞而造成的大量财务损失，突显了对有效安全审计的迫切需要。随着攻击变得更加频繁，对审计服务的必要性和需求也不断升级。这尤其给独立开发商和小企业带来了财务负担，他们提供这些服务的资金往往有限。我们的研究建立在现有框架的基础上，将检索-增强生成(RAG)与大型语言模型(LLMS)相结合，特别是将GPT-4-1106用于其128k令牌上下文窗口。我们构建了一个包含830个已知易受攻击合约的向量库，利用Pinecone进行向量存储，利用OpenAI的Text-Embedding-ada-002进行嵌入，并利用LangChain构建RAG-LLM管道。提示旨在为漏洞检测提供二进制答案。我们首先针对提供的漏洞类型对52个智能合约进行了40次测试，验证了RAG-LLM的可复制性和一致性。观察到了令人鼓舞的结果，引导检测漏洞的成功率为62.7%。其次，我们在“盲目”的审计设置下挑战模型，没有提示中提供的漏洞类型，其中219个合同每个都要接受40个测试。此设置在没有提示上下文帮助的情况下评估常规漏洞检测功能。在此条件下，成功率为60.71%。虽然结果是有希望的，但我们仍然强调在这个时候进行人力审计的必要性。我们提供这项研究作为经济高效的智能合同审计流程的概念证明，朝着民主的安全访问方向发展。



## **50. CVE-LLM : Automatic vulnerability evaluation in medical device industry using large language models**

CVE-LLM：使用大型语言模型在医疗器械行业中进行自动漏洞评估 cs.CL

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2407.14640v1) [paper-pdf](http://arxiv.org/pdf/2407.14640v1)

**Authors**: Rikhiya Ghosh, Oladimeji Farri, Hans-Martin von Stockhausen, Martin Schmitt, George Marica Vasile

**Abstract**: The healthcare industry is currently experiencing an unprecedented wave of cybersecurity attacks, impacting millions of individuals. With the discovery of thousands of vulnerabilities each month, there is a pressing need to drive the automation of vulnerability assessment processes for medical devices, facilitating rapid mitigation efforts. Generative AI systems have revolutionized various industries, offering unparalleled opportunities for automation and increased efficiency. This paper presents a solution leveraging Large Language Models (LLMs) to learn from historical evaluations of vulnerabilities for the automatic assessment of vulnerabilities in the medical devices industry. This approach is applied within the portfolio of a single manufacturer, taking into account device characteristics, including existing security posture and controls. The primary contributions of this paper are threefold. Firstly, it provides a detailed examination of the best practices for training a vulnerability Language Model (LM) in an industrial context. Secondly, it presents a comprehensive comparison and insightful analysis of the effectiveness of Language Models in vulnerability assessment. Finally, it proposes a new human-in-the-loop framework to expedite vulnerability evaluation processes.

摘要: 医疗保健行业目前正经历一波前所未有的网络安全攻击浪潮，影响着数百万人。随着每月发现数以千计的漏洞，迫切需要推动医疗设备脆弱性评估过程的自动化，促进快速缓解工作。生产性人工智能系统已经给各个行业带来了革命性的变化，为自动化和提高效率提供了无与伦比的机会。本文提出了一种利用大型语言模型(LLM)来学习历史漏洞评估的解决方案，用于医疗器械行业漏洞的自动评估。该方法在单个制造商的产品组合中应用，并考虑到设备特性，包括现有的安全状态和控制。本文的主要贡献有三个方面。首先，它提供了在工业环境中训练脆弱性语言模型(LM)的最佳实践的详细检查。其次，对语言模型在脆弱性评估中的有效性进行了全面的比较和深入的分析。最后，提出了一种新的人在环中框架，以加快脆弱性评估过程。



