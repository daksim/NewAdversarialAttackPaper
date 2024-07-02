# Latest Large Language Model Attack Papers
**update at 2024-07-02 15:08:28**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Enhancing the Capability and Robustness of Large Language Models through Reinforcement Learning-Driven Query Refinement**

通过强化学习驱动的查询细化增强大型语言模型的能力和鲁棒性 cs.CL

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01461v1) [paper-pdf](http://arxiv.org/pdf/2407.01461v1)

**Authors**: Zisu Huang, Xiaohua Wang, Feiran Zhang, Zhibo Xu, Cenyuan Zhang, Xiaoqing Zheng, Xuanjing Huang

**Abstract**: The capacity of large language models (LLMs) to generate honest, harmless, and helpful responses heavily relies on the quality of user prompts. However, these prompts often tend to be brief and vague, thereby significantly limiting the full potential of LLMs. Moreover, harmful prompts can be meticulously crafted and manipulated by adversaries to jailbreak LLMs, inducing them to produce potentially toxic content. To enhance the capabilities of LLMs while maintaining strong robustness against harmful jailbreak inputs, this study proposes a transferable and pluggable framework that refines user prompts before they are input into LLMs. This strategy improves the quality of the queries, empowering LLMs to generate more truthful, benign and useful responses. Specifically, a lightweight query refinement model is introduced and trained using a specially designed reinforcement learning approach that incorporates multiple objectives to enhance particular capabilities of LLMs. Extensive experiments demonstrate that the refinement model not only improves the quality of responses but also strengthens their robustness against jailbreak attacks. Code is available at: https://github.com/Huangzisu/query-refinement .

摘要: 大型语言模型(LLM)生成诚实、无害和有用的响应的能力在很大程度上取决于用户提示的质量。然而，这些提示往往简短而含糊，从而极大地限制了LLM的全部潜力。此外，有害的提示可以被对手精心制作和操纵，以越狱LLM，诱导它们产生潜在的有毒内容。为了增强LLMS的能力，同时保持对有害越狱输入的强大健壮性，本研究提出了一个可移植和可插拔的框架，在将用户提示输入到LLMS之前对其进行提炼。这一策略提高了查询的质量，使LLMS能够生成更真实、良性和有用的响应。具体地说，引入了一种轻量级查询精化模型，并使用专门设计的强化学习方法进行训练，该方法结合了多个目标来增强LLMS的特定能力。大量实验表明，改进模型不仅提高了响应的质量，而且增强了对越狱攻击的健壮性。代码可从以下网址获得：https://github.com/Huangzisu/query-refinement。



## **2. Jailbreak Vision Language Models via Bi-Modal Adversarial Prompt**

通过双模式对抗提示的越狱视觉语言模型 cs.CV

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2406.04031v2) [paper-pdf](http://arxiv.org/pdf/2406.04031v2)

**Authors**: Zonghao Ying, Aishan Liu, Tianyuan Zhang, Zhengmin Yu, Siyuan Liang, Xianglong Liu, Dacheng Tao

**Abstract**: In the realm of large vision language models (LVLMs), jailbreak attacks serve as a red-teaming approach to bypass guardrails and uncover safety implications. Existing jailbreaks predominantly focus on the visual modality, perturbing solely visual inputs in the prompt for attacks. However, they fall short when confronted with aligned models that fuse visual and textual features simultaneously for generation. To address this limitation, this paper introduces the Bi-Modal Adversarial Prompt Attack (BAP), which executes jailbreaks by optimizing textual and visual prompts cohesively. Initially, we adversarially embed universally harmful perturbations in an image, guided by a few-shot query-agnostic corpus (e.g., affirmative prefixes and negative inhibitions). This process ensures that image prompt LVLMs to respond positively to any harmful queries. Subsequently, leveraging the adversarial image, we optimize textual prompts with specific harmful intent. In particular, we utilize a large language model to analyze jailbreak failures and employ chain-of-thought reasoning to refine textual prompts through a feedback-iteration manner. To validate the efficacy of our approach, we conducted extensive evaluations on various datasets and LVLMs, demonstrating that our method significantly outperforms other methods by large margins (+29.03% in attack success rate on average). Additionally, we showcase the potential of our attacks on black-box commercial LVLMs, such as Gemini and ChatGLM.

摘要: 在大型视觉语言模型(LVLM)领域，越狱攻击是一种绕过护栏并发现安全隐患的红队方法。现有的越狱主要集中在视觉形式上，只干扰攻击提示中的视觉输入。然而，当面对同时融合视觉和文本特征以生成的对齐模型时，它们不能满足要求。为了解决这一局限性，本文引入了双模式对抗性提示攻击(BAP)，它通过结合优化文本和视觉提示来执行越狱。最初，我们不利地在图像中嵌入普遍有害的扰动，由几个与查询无关的语料库(例如，肯定前缀和否定抑制)引导。此过程确保图像提示LVLMS对任何有害查询做出积极响应。随后，利用敌意图像，我们优化了具有特定有害意图的文本提示。特别是，我们利用一个大的语言模型来分析越狱失败，并使用思想链推理来通过反馈迭代的方式来提炼文本提示。为了验证我们方法的有效性，我们在不同的数据集和LVLM上进行了广泛的评估，结果表明我们的方法在很大程度上优于其他方法(攻击成功率平均为+29.03%)。此外，我们还展示了我们对黑盒商业LVLM的攻击潜力，如Gemini和ChatGLM。



## **3. A Fingerprint for Large Language Models**

大型语言模型的指纹 cs.CR

https://scholar.google.com/citations?user=IdiF7M0AAAAJ&hl=en

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01235v1) [paper-pdf](http://arxiv.org/pdf/2407.01235v1)

**Authors**: Zhiguang Yang, Hanzhou Wu

**Abstract**: Recent advances show that scaling a pre-trained language model could achieve state-of-the-art performance on many downstream tasks, prompting large language models (LLMs) to become a hot research topic in the field of artificial intelligence. However, due to the resource-intensive nature of training LLMs from scratch, it is urgent and crucial to protect the intellectual property of LLMs against infringement. This has motivated the authors in this paper to propose a novel black-box fingerprinting technique for LLMs, which requires neither model training nor model fine-tuning. We first demonstrate that the outputs of LLMs span a unique vector space associated with each model. We model the problem of ownership authentication as the task of evaluating the similarity between the victim model's space and the output's space of the suspect model. To deal with this problem, we propose two solutions, where the first solution involves verifying whether the outputs of the suspected large model are in the same space as those of the victim model, enabling rapid identification of model infringement, and the second one reconstructs the union of the vector spaces for LLM outputs and the victim model to address situations where the victim model has undergone the Parameter-Efficient Fine-Tuning (PEFT) attacks. Experimental results indicate that the proposed technique achieves superior performance in ownership verification and robustness against PEFT attacks. This work reveals inherent characteristics of LLMs and provides a promising solution for ownership verification of LLMs in black-box scenarios, ensuring efficiency, generality and practicality.

摘要: 最近的进展表明，扩展一个预先训练的语言模型可以在许多下游任务上获得最先进的性能，这促使大型语言模型(LLM)成为人工智能领域的研究热点。然而，由于从无到有培训低成本管理人员的资源密集型性质，保护低收入管理人员的知识产权不受侵犯是迫切和关键的。为此，本文提出了一种新的黑盒指纹识别方法，该方法既不需要模型训练，也不需要模型微调。我们首先证明了LLMS的输出跨越了与每个模型相关的唯一向量空间。我们将所有权认证问题建模为评估受害者模型的空间与嫌疑人模型的输出空间之间的相似度的任务。为了解决这个问题，我们提出了两种解决方案，第一种方案涉及验证可疑大模型的输出是否与受害者模型的输出在同一空间中，从而能够快速识别模型违规；第二种方案重构LLM输出和受害者模型的向量空间的并集，以应对受害者模型经历了参数高效精调(PEFT)攻击的情况。实验结果表明，该方法具有较好的所有权验证性能和对PEFT攻击的稳健性。这项工作揭示了LLMS的内在特征，为黑盒场景下LLMS的所有权验证提供了一种很有前途的解决方案，确保了效率、通用性和实用性。



## **4. Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications**

通过修剪和低级修改评估安全对齐的脆弱性 cs.LG

22 pages, 9 figures. Project page is available at  https://boyiwei.com/alignment-attribution/

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2402.05162v3) [paper-pdf](http://arxiv.org/pdf/2402.05162v3)

**Authors**: Boyi Wei, Kaixuan Huang, Yangsibo Huang, Tinghao Xie, Xiangyu Qi, Mengzhou Xia, Prateek Mittal, Mengdi Wang, Peter Henderson

**Abstract**: Large language models (LLMs) show inherent brittleness in their safety mechanisms, as evidenced by their susceptibility to jailbreaking and even non-malicious fine-tuning. This study explores this brittleness of safety alignment by leveraging pruning and low-rank modifications. We develop methods to identify critical regions that are vital for safety guardrails, and that are disentangled from utility-relevant regions at both the neuron and rank levels. Surprisingly, the isolated regions we find are sparse, comprising about $3\%$ at the parameter level and $2.5\%$ at the rank level. Removing these regions compromises safety without significantly impacting utility, corroborating the inherent brittleness of the model's safety mechanisms. Moreover, we show that LLMs remain vulnerable to low-cost fine-tuning attacks even when modifications to the safety-critical regions are restricted. These findings underscore the urgent need for more robust safety strategies in LLMs.

摘要: 大型语言模型（LLM）在其安全机制中表现出固有的脆弱性，这一点从它们容易越狱甚至非恶意微调中得到了证明。这项研究通过利用修剪和低等级修改来探索安全对齐的脆弱性。我们开发了识别对安全护栏至关重要的关键区域的方法，并且在神经元和等级水平上与公用事业相关区域脱钩。令人惊讶的是，我们发现的孤立区域很稀疏，参数级别约为3美元，排名级别约为2.5美元。删除这些区域会损害安全性，而不会显着影响实用性，这证实了该模型安全机制固有的脆弱性。此外，我们表明，即使对安全关键区域的修改受到限制，LLM仍然容易受到低成本微调攻击。这些发现凸显了LLM迫切需要制定更稳健的安全策略。



## **5. Large Language Models Are Involuntary Truth-Tellers: Exploiting Fallacy Failure for Jailbreak Attacks**

大型语言模型是不自愿的真话者：利用谬误失败进行越狱攻击 cs.CL

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.00869v1) [paper-pdf](http://arxiv.org/pdf/2407.00869v1)

**Authors**: Yue Zhou, Henry Peng Zou, Barbara Di Eugenio, Yang Zhang

**Abstract**: We find that language models have difficulties generating fallacious and deceptive reasoning. When asked to generate deceptive outputs, language models tend to leak honest counterparts but believe them to be false. Exploiting this deficiency, we propose a jailbreak attack method that elicits an aligned language model for malicious output. Specifically, we query the model to generate a fallacious yet deceptively real procedure for the harmful behavior. Since a fallacious procedure is generally considered fake and thus harmless by LLMs, it helps bypass the safeguard mechanism. Yet the output is factually harmful since the LLM cannot fabricate fallacious solutions but proposes truthful ones. We evaluate our approach over five safety-aligned large language models, comparing four previous jailbreak methods, and show that our approach achieves competitive performance with more harmful outputs. We believe the findings could be extended beyond model safety, such as self-verification and hallucination.

摘要: 我们发现，语言模型很难生成错误和欺骗性的推理。当被要求生成欺骗性输出时，语言模型往往会泄露诚实的对应结果，但认为它们是错误的。利用这一缺陷，我们提出了一种越狱攻击方法，该方法可以得到恶意输出的对齐语言模型。具体地说，我们对该模型提出质疑，以便为有害行为生成一个虚假但虚假的真实过程。由于错误的程序通常被LLMS认为是虚假的，因此是无害的，它有助于绕过保障机制。然而，这些结果实际上是有害的，因为LLM不能捏造虚假的解决方案，而是提出真实的解决方案。我们在五个安全对齐的大型语言模型上对我们的方法进行了评估，并与之前的四种越狱方法进行了比较，结果表明我们的方法在具有更多有害输出的情况下获得了具有竞争力的性能。我们认为，这些发现可以扩展到模型安全之外，例如自我验证和幻觉。



## **6. Virtual Context: Enhancing Jailbreak Attacks with Special Token Injection**

虚拟上下文：通过特殊代币注入增强越狱攻击 cs.CR

14 pages, 4 figures

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.19845v1) [paper-pdf](http://arxiv.org/pdf/2406.19845v1)

**Authors**: Yuqi Zhou, Lin Lu, Hanchi Sun, Pan Zhou, Lichao Sun

**Abstract**: Jailbreak attacks on large language models (LLMs) involve inducing these models to generate harmful content that violates ethics or laws, posing a significant threat to LLM security. Current jailbreak attacks face two main challenges: low success rates due to defensive measures and high resource requirements for crafting specific prompts. This paper introduces Virtual Context, which leverages special tokens, previously overlooked in LLM security, to improve jailbreak attacks. Virtual Context addresses these challenges by significantly increasing the success rates of existing jailbreak methods and requiring minimal background knowledge about the target model, thus enhancing effectiveness in black-box settings without additional overhead. Comprehensive evaluations show that Virtual Context-assisted jailbreak attacks can improve the success rates of four widely used jailbreak methods by approximately 40% across various LLMs. Additionally, applying Virtual Context to original malicious behaviors still achieves a notable jailbreak effect. In summary, our research highlights the potential of special tokens in jailbreak attacks and recommends including this threat in red-teaming testing to comprehensively enhance LLM security.

摘要: 针对大型语言模型(LLM)的越狱攻击涉及诱导这些模型生成违反道德或法律的有害内容，对LLM安全构成重大威胁。目前的越狱攻击面临两个主要挑战：防御性措施导致的成功率较低，以及制作特定提示所需的资源较高。本文介绍了虚拟上下文技术，它利用了以前在LLM安全中被忽视的特殊令牌来改进越狱攻击。虚拟环境通过显著提高现有越狱方法的成功率和只需要最少的目标模型背景知识来解决这些挑战，从而在不增加额外开销的情况下提高黑箱设置的效率。综合评估表明，虚拟情境辅助越狱攻击可以将四种广泛使用的越狱方法的成功率提高约40%。此外，将虚拟情境应用于原始恶意行为仍然可以达到显著的越狱效果。综上所述，我们的研究强调了特殊令牌在越狱攻击中的潜力，并建议将此威胁包括在红团队测试中，以全面增强LLM安全。



## **7. SafeAligner: Safety Alignment against Jailbreak Attacks via Response Disparity Guidance**

SafeAligner：通过响应差异指导针对越狱攻击的安全调整 cs.CR

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.18118v2) [paper-pdf](http://arxiv.org/pdf/2406.18118v2)

**Authors**: Caishuang Huang, Wanxu Zhao, Rui Zheng, Huijie Lv, Shihan Dou, Sixian Li, Xiao Wang, Enyu Zhou, Junjie Ye, Yuming Yang, Tao Gui, Qi Zhang, Xuanjing Huang

**Abstract**: As the development of large language models (LLMs) rapidly advances, securing these models effectively without compromising their utility has become a pivotal area of research. However, current defense strategies against jailbreak attacks (i.e., efforts to bypass security protocols) often suffer from limited adaptability, restricted general capability, and high cost. To address these challenges, we introduce SafeAligner, a methodology implemented at the decoding stage to fortify defenses against jailbreak attacks. We begin by developing two specialized models: the Sentinel Model, which is trained to foster safety, and the Intruder Model, designed to generate riskier responses. SafeAligner leverages the disparity in security levels between the responses from these models to differentiate between harmful and beneficial tokens, effectively guiding the safety alignment by altering the output token distribution of the target model. Extensive experiments show that SafeAligner can increase the likelihood of beneficial tokens, while reducing the occurrence of harmful ones, thereby ensuring secure alignment with minimal loss to generality.

摘要: 随着大型语言模型(LLM)的发展，在不影响其实用性的情况下有效地保护这些模型已成为一个关键的研究领域。然而，当前针对越狱攻击的防御策略(即绕过安全协议的努力)往往存在适应性有限、通用能力有限和成本较高的问题。为了应对这些挑战，我们引入了SafeAligner，这是一种在解码阶段实施的方法，用于加强对越狱攻击的防御。我们首先开发两个专门的模型：哨兵模型和入侵者模型，前者旨在促进安全，后者旨在产生更高风险的反应。SafeAligner利用这些模型响应之间的安全级别差异来区分有害令牌和有益令牌，通过更改目标模型的输出令牌分布有效地指导安全对齐。广泛的实验表明，SafeAligner可以增加有益令牌的可能性，同时减少有害令牌的发生，从而确保安全对齐，并将对一般性的损失降至最低。



## **8. Revisiting Backdoor Attacks against Large Vision-Language Models**

重新审视针对大型视觉语言模型的后门攻击 cs.CV

23 pages, 8 figures

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.18844v2) [paper-pdf](http://arxiv.org/pdf/2406.18844v2)

**Authors**: Siyuan Liang, Jiawei Liang, Tianyu Pang, Chao Du, Aishan Liu, Ee-Chien Chang, Xiaochun Cao

**Abstract**: Instruction tuning enhances large vision-language models (LVLMs) but raises security risks through potential backdoor attacks due to their openness. Previous backdoor studies focus on enclosed scenarios with consistent training and testing instructions, neglecting the practical domain gaps that could affect attack effectiveness. This paper empirically examines the generalizability of backdoor attacks during the instruction tuning of LVLMs for the first time, revealing certain limitations of most backdoor strategies in practical scenarios. We quantitatively evaluate the generalizability of six typical backdoor attacks on image caption benchmarks across multiple LVLMs, considering both visual and textual domain offsets. Our findings indicate that attack generalizability is positively correlated with the backdoor trigger's irrelevance to specific images/models and the preferential correlation of the trigger pattern. Additionally, we modify existing backdoor attacks based on the above key observations, demonstrating significant improvements in cross-domain scenario generalizability (+86% attack success rate). Notably, even without access to the instruction datasets, a multimodal instruction set can be successfully poisoned with a very low poisoning rate (0.2%), achieving an attack success rate of over 97%. This paper underscores that even simple traditional backdoor strategies pose a serious threat to LVLMs, necessitating more attention and in-depth research.

摘要: 指令调优增强了大型视觉语言模型(LVLM)，但由于其开放性，通过潜在的后门攻击增加了安全风险。以前的后门研究侧重于具有一致训练和测试指令的封闭场景，而忽略了可能影响攻击效果的实际领域差距。本文首次对LVLMS指令调优过程中后门攻击的泛化能力进行了实证检验，揭示了大多数后门策略在实际应用中的局限性。我们定量地评估了六种典型的后门攻击在多个LVLM上对图像字幕基准的泛化能力，同时考虑了视觉和文本域偏移。我们的研究结果表明，攻击的概括性与后门触发器与特定图像/模型的无关性以及触发模式的优先相关性呈正相关。此外，我们根据上述关键观察结果修改了现有的后门攻击，显示出跨域场景通用性的显著改进(+86%的攻击成功率)。值得注意的是，即使不访问指令数据集，多模式指令集也可以以非常低的投毒率(0.2%)成功中毒，实现超过97%的攻击成功率。这篇文章强调，即使是简单的传统后门策略也对LVLMS构成了严重威胁，需要更多的关注和深入的研究。



## **9. GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts**

GPTFUZER：Red将大型语言模型与自动生成的越狱脚本结合起来 cs.AI

**SubmitDate**: 2024-06-27    [abs](http://arxiv.org/abs/2309.10253v4) [paper-pdf](http://arxiv.org/pdf/2309.10253v4)

**Authors**: Jiahao Yu, Xingwei Lin, Zheng Yu, Xinyu Xing

**Abstract**: Large language models (LLMs) have recently experienced tremendous popularity and are widely used from casual conversations to AI-driven programming. However, despite their considerable success, LLMs are not entirely reliable and can give detailed guidance on how to conduct harmful or illegal activities. While safety measures can reduce the risk of such outputs, adversarial jailbreak attacks can still exploit LLMs to produce harmful content. These jailbreak templates are typically manually crafted, making large-scale testing challenging.   In this paper, we introduce GPTFuzz, a novel black-box jailbreak fuzzing framework inspired by the AFL fuzzing framework. Instead of manual engineering, GPTFuzz automates the generation of jailbreak templates for red-teaming LLMs. At its core, GPTFuzz starts with human-written templates as initial seeds, then mutates them to produce new templates. We detail three key components of GPTFuzz: a seed selection strategy for balancing efficiency and variability, mutate operators for creating semantically equivalent or similar sentences, and a judgment model to assess the success of a jailbreak attack.   We evaluate GPTFuzz against various commercial and open-source LLMs, including ChatGPT, LLaMa-2, and Vicuna, under diverse attack scenarios. Our results indicate that GPTFuzz consistently produces jailbreak templates with a high success rate, surpassing human-crafted templates. Remarkably, GPTFuzz achieves over 90% attack success rates against ChatGPT and Llama-2 models, even with suboptimal initial seed templates. We anticipate that GPTFuzz will be instrumental for researchers and practitioners in examining LLM robustness and will encourage further exploration into enhancing LLM safety.

摘要: 大型语言模型(LLM)最近经历了巨大的流行，并被广泛使用，从随意的对话到人工智能驱动的编程。然而，尽管LLM取得了相当大的成功，但它们并不完全可靠，可以就如何进行有害或非法活动提供详细指导。虽然安全措施可以降低此类输出的风险，但对抗性越狱攻击仍然可以利用LLMS产生有害内容。这些越狱模板通常是手动制作的，这使得大规模测试具有挑战性。在本文中，我们介绍了一种新的黑盒越狱模糊框架GPTFuzz，该框架受到AFL模糊框架的启发。GPTFuzz不是手动设计，而是自动生成用于红队LLM的越狱模板。在其核心，GPTFuzz以人类编写的模板作为初始种子，然后对它们进行突变以产生新的模板。我们详细介绍了GPTFuzz的三个关键组成部分：用于平衡效率和可变性的种子选择策略，用于创建语义等价或相似句子的变异算子，以及用于评估越狱攻击成功的判断模型。我们在不同的攻击场景下，针对各种商业和开源LLM，包括ChatGPT、骆驼2和维库纳，对GPTFuzz进行了评估。我们的结果表明，GPTFuzz一致地生成了成功率较高的越狱模板，超过了人工制作的模板。值得注意的是，GPTFuzz对ChatGPT和Llama-2模型的攻击成功率超过90%，即使在初始种子模板不是最优的情况下也是如此。我们预计，GPTFuzz将有助于研究人员和从业者检查LLM的稳健性，并将鼓励进一步探索增强LLM的安全性。



## **10. Seeing Is Believing: Black-Box Membership Inference Attacks Against Retrieval Augmented Generation**

亲眼所见：针对检索增强生成的黑匣子成员推断攻击 cs.CR

**SubmitDate**: 2024-06-27    [abs](http://arxiv.org/abs/2406.19234v1) [paper-pdf](http://arxiv.org/pdf/2406.19234v1)

**Authors**: Yuying Li, Gaoyang Liu, Yang Yang, Chen Wang

**Abstract**: Retrieval-Augmented Generation (RAG) is a state-of-the-art technique that enhances Large Language Models (LLMs) by retrieving relevant knowledge from an external, non-parametric database. This approach aims to mitigate common LLM issues such as hallucinations and outdated knowledge. Although existing research has demonstrated security and privacy vulnerabilities within RAG systems, making them susceptible to attacks like jailbreaks and prompt injections, the security of the RAG system's external databases remains largely underexplored. In this paper, we employ Membership Inference Attacks (MIA) to determine whether a sample is part of the knowledge database of a RAG system, using only black-box API access. Our core hypothesis posits that if a sample is a member, it will exhibit significant similarity to the text generated by the RAG system. To test this, we compute the cosine similarity and the model's perplexity to establish a membership score, thereby building robust features. We then introduce two novel attack strategies: a Threshold-based Attack and a Machine Learning-based Attack, designed to accurately identify membership. Experimental validation of our methods has achieved a ROC AUC of 82%.

摘要: 检索-增强生成(RAG)是一种最先进的技术，它通过从外部非参数数据库检索相关知识来增强大型语言模型(LLMS)。这种方法旨在缓解常见的LLM问题，如幻觉和过时的知识。尽管现有的研究已经证明RAG系统中存在安全和隐私漏洞，使它们容易受到越狱和快速注射等攻击，但RAG系统外部数据库的安全性在很大程度上仍未得到充分研究。在本文中，我们使用成员关系推理攻击(MIA)来确定样本是否属于RAG系统的知识库的一部分，只使用黑盒API访问。我们的核心假设是，如果样本是成员，它将显示出与RAG系统生成的文本的显著相似性。为了测试这一点，我们计算余弦相似度和模型的困惑度来建立隶属度分数，从而建立稳健的特征。然后，我们介绍了两种新的攻击策略：基于阈值的攻击和基于机器学习的攻击，旨在准确识别成员身份。通过实验验证，我们的方法获得了82%的ROC AUC。



## **11. Chat AI: A Seamless Slurm-Native Solution for HPC-Based Services**

Chat AI：针对基于HP的服务的无缝SlurmNative解决方案 cs.DC

27 pages, 5 figures, 2 tables

**SubmitDate**: 2024-06-27    [abs](http://arxiv.org/abs/2407.00110v1) [paper-pdf](http://arxiv.org/pdf/2407.00110v1)

**Authors**: Ali Doosthosseini, Jonathan Decker, Hendrik Nolte, Julian M. Kunkel

**Abstract**: The increasing adoption of large language models (LLMs) has created a pressing need for an efficient, secure and private serving infrastructure, which allows researchers to run open-source or custom fine-tuned LLMs and ensures users that their data remains private and is not stored without their consent. While high-performance computing (HPC) systems equipped with state-of-the-art GPUs are well-suited for training LLMs, their batch scheduling paradigm is not designed to support real-time serving of AI applications. Cloud systems, on the other hand, are well suited for web services but commonly lack access to the computational power of clusters, especially expensive and scarce high-end GPUs, which are required for optimal inference speed. We propose an architecture with an implementation consisting of a web service that runs on a cloud VM with secure access to a scalable backend running a multitude of AI models on HPC systems. By offering a web service using our HPC infrastructure to host LLMs, we leverage the trusted environment of local universities and research centers to offer a private and secure alternative to commercial LLM services. Our solution natively integrates with Slurm, enabling seamless deployment on HPC clusters and is able to run side by side with regular Slurm workloads, while utilizing gaps in the schedule created by Slurm. In order to ensure the security of the HPC system, we use the SSH ForceCommand directive to construct a robust circuit breaker, which prevents successful attacks on the web-facing server from affecting the cluster. We have successfully deployed our system as a production service, and made the source code available at https://github.com/gwdg/chat-ai

摘要: 大型语言模型(LLM)的日益采用产生了对高效、安全和私有的服务基础设施的迫切需求，该基础设施允许研究人员运行开源或定制的微调LLM，并确保用户的数据保持隐私，并且在未经用户同意的情况下不被存储。虽然配备最先进的GPU的高性能计算(HPC)系统非常适合训练LLM，但它们的批处理调度范例并不是为支持AI应用的实时服务而设计的。另一方面，云系统非常适合Web服务，但通常无法获得集群的计算能力，特别是昂贵而稀缺的高端GPU，这是实现最佳推理速度所必需的。我们提出了一种体系结构，其实现包括在云VM上运行的Web服务，该服务可以安全地访问在HPC系统上运行大量AI模型的可扩展后端。通过提供使用我们的HPC基础设施来托管LLM的Web服务，我们利用当地大学和研究中心的可信环境来提供商业LLM服务的私有且安全的替代方案。我们的解决方案与SLurm进行了本机集成，实现了在HPC群集上的无缝部署，并能够与常规SLurm工作负载并行运行，同时利用SLurm创建的时间表中的空白。为了确保HPC系统的安全，我们使用SSH ForceCommand指令来构建一个健壮的断路器，以防止对面向Web的服务器的成功攻击影响到集群。我们已经成功地将我们的系统部署为生产服务，并在https://github.com/gwdg/chat-ai上提供了源代码



## **12. Assessing the Effectiveness of LLMs in Android Application Vulnerability Analysis**

评估LLC在Android应用程序漏洞分析中的有效性 cs.CR

**SubmitDate**: 2024-06-27    [abs](http://arxiv.org/abs/2406.18894v1) [paper-pdf](http://arxiv.org/pdf/2406.18894v1)

**Authors**: Vasileios Kouliaridis, Georgios Karopoulos, Georgios Kambourakis

**Abstract**: The increasing frequency of attacks on Android applications coupled with the recent popularity of large language models (LLMs) necessitates a comprehensive understanding of the capabilities of the latter in identifying potential vulnerabilities, which is key to mitigate the overall risk. To this end, the work at hand compares the ability of nine state-of-the-art LLMs to detect Android code vulnerabilities listed in the latest Open Worldwide Application Security Project (OWASP) Mobile Top 10. Each LLM was evaluated against an open dataset of over 100 vulnerable code samples, including obfuscated ones, assessing each model's ability to identify key vulnerabilities. Our analysis reveals the strengths and weaknesses of each LLM, identifying important factors that contribute to their performance. Additionally, we offer insights into context augmentation with retrieval-augmented generation (RAG) for detecting Android code vulnerabilities, which in turn may propel secure application development. Finally, while the reported findings regarding code vulnerability analysis show promise, they also reveal significant discrepancies among the different LLMs.

摘要: 针对Android应用程序的攻击日益频繁，加上最近大型语言模型(LLM)的流行，需要全面了解后者在识别潜在漏洞方面的能力，这是降低总体风险的关键。为此，手头的工作比较了9个最先进的LLM检测Android代码漏洞的能力，这些漏洞列在最新的Open Worldwide Application Security Project(OWASP)Mobile Top 10中。每个LLM都是根据100多个易受攻击的代码样本(包括混淆的代码样本)的开放数据集进行评估的，以评估每个模型识别关键漏洞的能力。我们的分析揭示了每个LLM的优势和劣势，找出了影响它们表现的重要因素。此外，我们通过检索增强生成(RAG)提供了对上下文增强的见解，以检测Android代码漏洞，这反过来可能会推动安全的应用程序开发。最后，虽然有关代码漏洞分析的报告结果显示前景看好，但它们也揭示了不同LLM之间的显著差异。



## **13. Dysca: A Dynamic and Scalable Benchmark for Evaluating Perception Ability of LVLMs**

Dysca：评估LVLM感知能力的动态和可扩展基准 cs.CV

**SubmitDate**: 2024-06-27    [abs](http://arxiv.org/abs/2406.18849v1) [paper-pdf](http://arxiv.org/pdf/2406.18849v1)

**Authors**: Jie Zhang, Zhongqi Wang, Mengqi Lei, Zheng Yuan, Bei Yan, Shiguang Shan, Xilin Chen

**Abstract**: Currently many benchmarks have been proposed to evaluate the perception ability of the Large Vision-Language Models (LVLMs). However, most benchmarks conduct questions by selecting images from existing datasets, resulting in the potential data leakage. Besides, these benchmarks merely focus on evaluating LVLMs on the realistic style images and clean scenarios, leaving the multi-stylized images and noisy scenarios unexplored. In response to these challenges, we propose a dynamic and scalable benchmark named Dysca for evaluating LVLMs by leveraging synthesis images. Specifically, we leverage Stable Diffusion and design a rule-based method to dynamically generate novel images, questions and the corresponding answers. We consider 51 kinds of image styles and evaluate the perception capability in 20 subtasks. Moreover, we conduct evaluations under 4 scenarios (i.e., Clean, Corruption, Print Attacking and Adversarial Attacking) and 3 question types (i.e., Multi-choices, True-or-false and Free-form). Thanks to the generative paradigm, Dysca serves as a scalable benchmark for easily adding new subtasks and scenarios. A total of 8 advanced open-source LVLMs with 10 checkpoints are evaluated on Dysca, revealing the drawbacks of current LVLMs. The benchmark is released in \url{https://github.com/Benchmark-Dysca/Dysca}.

摘要: 目前，人们已经提出了许多基准来评估大型视觉语言模型的感知能力。然而，大多数基准测试通过从现有数据集中选择图像来进行问题，从而导致潜在的数据泄漏。此外，这些基准只关注真实感风格的图像和干净的场景来评估LVLMS，而对多风格化的图像和噪声场景没有进行探索。为了应对这些挑战，我们提出了一种动态的、可扩展的基准测试DYSCA，用于利用合成图像来评估LVLMS。具体地说，我们利用稳定扩散并设计了一种基于规则的方法来动态生成新的图像、问题和相应的答案。我们考虑了51种图像风格，并在20个子任务中评估了感知能力。此外，我们还在4个场景(即廉洁、腐败、打印攻击和对抗性攻击)和3个问题类型(即多项选择、对错和自由形式)下进行了评估。多亏了生成性范式，Dysca成为了一个可伸缩的基准，可以轻松添加新的子任务和场景。在Dysca上对8个具有10个检查点的高级开源LVLMS进行了评估，揭示了当前LVLMS的缺陷。该基准测试在\url{https://github.com/Benchmark-Dysca/Dysca}.中发布



## **14. Jailbreaking LLMs with Arabic Transliteration and Arabizi**

使用阿拉伯语拼音和Arabizi语越狱LLM cs.LG

14 pages, 4 figures

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18725v1) [paper-pdf](http://arxiv.org/pdf/2406.18725v1)

**Authors**: Mansour Al Ghanim, Saleh Almohaimeed, Mengxin Zheng, Yan Solihin, Qian Lou

**Abstract**: This study identifies the potential vulnerabilities of Large Language Models (LLMs) to 'jailbreak' attacks, specifically focusing on the Arabic language and its various forms. While most research has concentrated on English-based prompt manipulation, our investigation broadens the scope to investigate the Arabic language. We initially tested the AdvBench benchmark in Standardized Arabic, finding that even with prompt manipulation techniques like prefix injection, it was insufficient to provoke LLMs into generating unsafe content. However, when using Arabic transliteration and chatspeak (or arabizi), we found that unsafe content could be produced on platforms like OpenAI GPT-4 and Anthropic Claude 3 Sonnet. Our findings suggest that using Arabic and its various forms could expose information that might remain hidden, potentially increasing the risk of jailbreak attacks. We hypothesize that this exposure could be due to the model's learned connection to specific words, highlighting the need for more comprehensive safety training across all language forms.

摘要: 这项研究确定了大型语言模型(LLM)对‘越狱’攻击的潜在漏洞，特别是关注阿拉伯语及其各种形式。虽然大多数研究都集中在基于英语的即时操作上，但我们的调查扩大了对阿拉伯语的研究范围。我们最初用标准化的阿拉伯语测试了AdvBtch基准测试，发现即使使用前缀注入等快速操作技术，也不足以激发LLMS生成不安全的内容。然而，当使用阿拉伯语音译和聊天(或Arabizi)时，我们发现在OpenAI GPT-4和人类克劳德3十四行诗等平台上可能会产生不安全的内容。我们的发现表明，使用阿拉伯语及其各种形式可能会暴露可能仍然隐藏的信息，潜在地增加越狱攻击的风险。我们假设，这种接触可能是由于模型与特定单词的习得联系，强调了在所有语言形式中进行更全面的安全培训的必要性。



## **15. MultiAgent Collaboration Attack: Investigating Adversarial Attacks in Large Language Model Collaborations via Debate**

多Agent协作攻击：通过辩论调查大型语言模型协作中的对抗性攻击 cs.CL

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.14711v2) [paper-pdf](http://arxiv.org/pdf/2406.14711v2)

**Authors**: Alfonso Amayuelas, Xianjun Yang, Antonis Antoniades, Wenyue Hua, Liangming Pan, William Wang

**Abstract**: Large Language Models (LLMs) have shown exceptional results on current benchmarks when working individually. The advancement in their capabilities, along with a reduction in parameter size and inference times, has facilitated the use of these models as agents, enabling interactions among multiple models to execute complex tasks. Such collaborations offer several advantages, including the use of specialized models (e.g. coding), improved confidence through multiple computations, and enhanced divergent thinking, leading to more diverse outputs. Thus, the collaborative use of language models is expected to grow significantly in the coming years. In this work, we evaluate the behavior of a network of models collaborating through debate under the influence of an adversary. We introduce pertinent metrics to assess the adversary's effectiveness, focusing on system accuracy and model agreement. Our findings highlight the importance of a model's persuasive ability in influencing others. Additionally, we explore inference-time methods to generate more compelling arguments and evaluate the potential of prompt-based mitigation as a defensive strategy.

摘要: 大型语言模型(LLM)在单独工作时，在当前基准上显示了特殊的结果。它们能力的进步，加上参数大小和推理时间的减少，促进了这些模型作为代理的使用，使多个模型之间能够相互作用，以执行复杂的任务。这种协作提供了几个优势，包括使用专门的模型(例如编码)、通过多次计算提高信心以及增强发散思维，从而产生更多样化的产出。因此，语言模型的协作使用预计在未来几年将显著增长。在这项工作中，我们评估了一个模型网络在对手的影响下通过辩论进行合作的行为。我们引入了相关的度量来评估对手的有效性，重点是系统的准确性和模型的一致性。我们的发现突显了模特的说服力在影响他人方面的重要性。此外，我们探索推理时间方法来生成更令人信服的论点，并评估基于即时缓解作为一种防御策略的潜力。



## **16. Adversarial Search Engine Optimization for Large Language Models**

大型语言模型的对抗性搜索引擎优化 cs.CR

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18382v1) [paper-pdf](http://arxiv.org/pdf/2406.18382v1)

**Authors**: Fredrik Nestaas, Edoardo Debenedetti, Florian Tramèr

**Abstract**: Large Language Models (LLMs) are increasingly used in applications where the model selects from competing third-party content, such as in LLM-powered search engines or chatbot plugins. In this paper, we introduce Preference Manipulation Attacks, a new class of attacks that manipulate an LLM's selections to favor the attacker. We demonstrate that carefully crafted website content or plugin documentations can trick an LLM to promote the attacker products and discredit competitors, thereby increasing user traffic and monetization. We show this leads to a prisoner's dilemma, where all parties are incentivized to launch attacks, but the collective effect degrades the LLM's outputs for everyone. We demonstrate our attacks on production LLM search engines (Bing and Perplexity) and plugin APIs (for GPT-4 and Claude). As LLMs are increasingly used to rank third-party content, we expect Preference Manipulation Attacks to emerge as a significant threat.

摘要: 大型语言模型（LLM）越来越多地用于模型从竞争的第三方内容中进行选择的应用程序中，例如LLM支持的搜索引擎或聊天机器人插件。在本文中，我们引入了偏好操纵攻击，这是一类新的攻击，可以操纵LLM的选择以有利于攻击者。我们证明，精心制作的网站内容或插件文档可以欺骗LLM来推广攻击者的产品并抹黑竞争对手，从而增加用户流量和货币化。我们表明，这导致了囚犯困境，所有各方都受到激励发起攻击，但集体效应降低了LLM对每个人的产出。我们展示了对生产LLM搜索引擎（Bing和Perplexity）和插件API（适用于GPT-4和Claude）的攻击。随着LLM越来越多地用于对第三方内容进行排名，我们预计偏好操纵攻击将成为一个重大威胁。



## **17. Towards Understanding Jailbreak Attacks in LLMs: A Representation Space Analysis**

了解LLC中的越狱攻击：表示空间分析 cs.CL

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.10794v2) [paper-pdf](http://arxiv.org/pdf/2406.10794v2)

**Authors**: Yuping Lin, Pengfei He, Han Xu, Yue Xing, Makoto Yamada, Hui Liu, Jiliang Tang

**Abstract**: Large language models (LLMs) are susceptible to a type of attack known as jailbreaking, which misleads LLMs to output harmful contents. Although there are diverse jailbreak attack strategies, there is no unified understanding on why some methods succeed and others fail. This paper explores the behavior of harmful and harmless prompts in the LLM's representation space to investigate the intrinsic properties of successful jailbreak attacks. We hypothesize that successful attacks share some similar properties: They are effective in moving the representation of the harmful prompt towards the direction to the harmless prompts. We leverage hidden representations into the objective of existing jailbreak attacks to move the attacks along the acceptance direction, and conduct experiments to validate the above hypothesis using the proposed objective. We hope this study provides new insights into understanding how LLMs understand harmfulness information.

摘要: 大型语言模型（LLM）容易受到一种称为越狱的攻击，这种攻击会误导LLM输出有害内容。尽管越狱攻击策略多种多样，但对于为什么有些方法成功而另一些方法失败，人们并没有统一的理解。本文探讨了LLM表示空间中有害和无害提示的行为，以研究成功越狱攻击的内在属性。我们假设成功的攻击具有一些相似的属性：它们有效地将有害提示的表示移向无害提示的方向。我们将隐藏的表示利用到现有越狱攻击的目标中，以沿着接受方向移动攻击，并使用提出的目标进行实验来验证上述假设。我们希望这项研究为理解LLM如何理解有害信息提供新的见解。



## **18. Are AI-Generated Text Detectors Robust to Adversarial Perturbations?**

人工智能生成的文本检测器对对抗性扰动是否稳健？ cs.CL

Accepted to ACL 2024 main conference

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.01179v2) [paper-pdf](http://arxiv.org/pdf/2406.01179v2)

**Authors**: Guanhua Huang, Yuchen Zhang, Zhe Li, Yongjian You, Mingze Wang, Zhouwang Yang

**Abstract**: The widespread use of large language models (LLMs) has sparked concerns about the potential misuse of AI-generated text, as these models can produce content that closely resembles human-generated text. Current detectors for AI-generated text (AIGT) lack robustness against adversarial perturbations, with even minor changes in characters or words causing a reversal in distinguishing between human-created and AI-generated text. This paper investigates the robustness of existing AIGT detection methods and introduces a novel detector, the Siamese Calibrated Reconstruction Network (SCRN). The SCRN employs a reconstruction network to add and remove noise from text, extracting a semantic representation that is robust to local perturbations. We also propose a siamese calibration technique to train the model to make equally confidence predictions under different noise, which improves the model's robustness against adversarial perturbations. Experiments on four publicly available datasets show that the SCRN outperforms all baseline methods, achieving 6.5\%-18.25\% absolute accuracy improvement over the best baseline method under adversarial attacks. Moreover, it exhibits superior generalizability in cross-domain, cross-genre, and mixed-source scenarios. The code is available at \url{https://github.com/CarlanLark/Robust-AIGC-Detector}.

摘要: 大型语言模型(LLM)的广泛使用引发了人们对人工智能生成的文本可能被滥用的担忧，因为这些模型可以生成与人类生成的文本非常相似的内容。目前的人工智能生成文本检测器(AIGT)缺乏对对手扰动的稳健性，即使是字符或单词的微小变化也会导致在区分人工生成文本和人工智能生成文本方面出现逆转。本文研究了现有的AIGT检测方法的稳健性，并介绍了一种新的检测器--暹罗校准重建网络(SCRN)。SCRN使用重构网络来添加和去除文本中的噪声，提取对局部扰动具有鲁棒性的语义表示。我们还提出了一种暹罗校正技术来训练模型，使其在不同的噪声下做出相同的置信度预测，从而提高了模型对对抗性扰动的鲁棒性。在四个公开可用的数据集上的实验表明，SCRN的性能优于所有的基线方法，在对抗性攻击下，其绝对准确率比最佳基线方法提高了6.5-18.25。此外，它在跨域、跨流派和混合来源的场景中表现出出色的泛化能力。代码可在\url{https://github.com/CarlanLark/Robust-AIGC-Detector}.上获得



## **19. Enhancing Data Privacy in Large Language Models through Private Association Editing**

通过私人关联编辑增强大型语言模型中的数据隐私 cs.CL

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18221v1) [paper-pdf](http://arxiv.org/pdf/2406.18221v1)

**Authors**: Davide Venditti, Elena Sofia Ruzzetti, Giancarlo A. Xompero, Cristina Giannone, Andrea Favalli, Raniero Romagnoli, Fabio Massimo Zanzotto

**Abstract**: Large Language Models (LLMs) are powerful tools with extensive applications, but their tendency to memorize private information raises significant concerns as private data leakage can easily happen. In this paper, we introduce Private Association Editing (PAE), a novel defense approach for private data leakage. PAE is designed to effectively remove Personally Identifiable Information (PII) without retraining the model. Our approach consists of a four-step procedure: detecting memorized PII, applying PAE cards to mitigate memorization of private data, verifying resilience to targeted data extraction (TDE) attacks, and ensuring consistency in the post-edit LLMs. The versatility and efficiency of PAE, which allows for batch modifications, significantly enhance data privacy in LLMs. Experimental results demonstrate the effectiveness of PAE in mitigating private data leakage. We believe PAE will serve as a critical tool in the ongoing effort to protect data privacy in LLMs, encouraging the development of safer models for real-world applications.

摘要: 大型语言模型(LLM)是应用广泛的强大工具，但它们存储私人信息的倾向引起了人们的极大担忧，因为私人数据很容易泄露。本文介绍了一种新的隐私数据泄露防御方法--私有关联编辑(PAE)。PAE旨在有效地删除个人身份信息(PII)，而无需对模型进行重新培训。我们的方法包括四个步骤：检测记忆的PII，应用PAE卡来减少私人数据的记忆，验证对目标数据提取(TDE)攻击的恢复能力，以及确保编辑后LLM的一致性。PAE的多功能性和效率允许批量修改，显著增强了LLMS中的数据隐私。实验结果证明了PAE在缓解私有数据泄露方面的有效性。我们相信，PAE将作为正在进行的保护LLMS数据隐私的努力中的关键工具，鼓励为现实世界应用程序开发更安全的模型。



## **20. Poisoned LangChain: Jailbreak LLMs by LangChain**

中毒的LangChain：LangChain的越狱LLMS cs.CL

6 pages,2 figures,This paper is a submission to ACM TURC. It has been  accepted by the editor of the organizer

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18122v1) [paper-pdf](http://arxiv.org/pdf/2406.18122v1)

**Authors**: Ziqiu Wang, Jun Liu, Shengkai Zhang, Yang Yang

**Abstract**: With the development of natural language processing (NLP), large language models (LLMs) are becoming increasingly popular. LLMs are integrating more into everyday life, raising public concerns about their security vulnerabilities. Consequently, the security of large language models is becoming critically important. Currently, the techniques for attacking and defending against LLMs are continuously evolving. One significant method type of attack is the jailbreak attack, which designed to evade model safety mechanisms and induce the generation of inappropriate content. Existing jailbreak attacks primarily rely on crafting inducement prompts for direct jailbreaks, which are less effective against large models with robust filtering and high comprehension abilities. Given the increasing demand for real-time capabilities in large language models, real-time updates and iterations of new knowledge have become essential. Retrieval-Augmented Generation (RAG), an advanced technique to compensate for the model's lack of new knowledge, is gradually becoming mainstream. As RAG enables the model to utilize external knowledge bases, it provides a new avenue for jailbreak attacks.   In this paper, we conduct the first work to propose the concept of indirect jailbreak and achieve Retrieval-Augmented Generation via LangChain. Building on this, we further design a novel method of indirect jailbreak attack, termed Poisoned-LangChain (PLC), which leverages a poisoned external knowledge base to interact with large language models, thereby causing the large models to generate malicious non-compliant dialogues.We tested this method on six different large language models across three major categories of jailbreak issues. The experiments demonstrate that PLC successfully implemented indirect jailbreak attacks under three different scenarios, achieving success rates of 88.56%, 79.04%, and 82.69% respectively.

摘要: 随着自然语言处理(NLP)的发展，大语言模型(LLM)变得越来越流行。LLMS正在更多地融入日常生活，这引发了公众对其安全漏洞的担忧。因此，大型语言模型的安全性变得至关重要。目前，攻击和防御LLMS的技术正在不断发展。越狱攻击是一种重要的攻击方法类型，旨在逃避模型安全机制并诱导生成不适当的内容。现有的越狱攻击主要依赖于精心制作直接越狱的诱导提示，这对具有强大过滤和高理解能力的大型模型效果较差。鉴于大型语言模型对实时能力的需求日益增加，实时更新和迭代新知识变得至关重要。检索-增强生成(RAG)是一种弥补模型缺乏新知识的先进技术，正逐渐成为主流。由于RAG使模型能够利用外部知识库，它为越狱攻击提供了一种新的途径。在本文中，我们首次提出了间接越狱的概念，并通过LangChain实现了检索增强生成。在此基础上，我们进一步设计了一种新的间接越狱攻击方法，称为毒化朗链(PLC)，它利用有毒的外部知识库与大型语言模型交互，从而导致大型模型生成恶意的不符合规则的对话框，并在三大类越狱问题的六个不同的大型语言模型上测试了该方法。实验表明，PLC在三种不同的场景下成功实现了间接越狱攻击，成功率分别为88.56%、79.04%和82.69%。



## **21. Safely Learning with Private Data: A Federated Learning Framework for Large Language Model**

利用私人数据安全学习：大型语言模型的联邦学习框架 cs.CR

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.14898v2) [paper-pdf](http://arxiv.org/pdf/2406.14898v2)

**Authors**: JiaYing Zheng, HaiNan Zhang, LingXiang Wang, WangJie Qiu, HongWei Zheng, ZhiMing Zheng

**Abstract**: Private data, being larger and quality-higher than public data, can greatly improve large language models (LLM). However, due to privacy concerns, this data is often dispersed in multiple silos, making its secure utilization for LLM training a challenge. Federated learning (FL) is an ideal solution for training models with distributed private data, but traditional frameworks like FedAvg are unsuitable for LLM due to their high computational demands on clients. An alternative, split learning, offloads most training parameters to the server while training embedding and output layers locally, making it more suitable for LLM. Nonetheless, it faces significant challenges in security and efficiency. Firstly, the gradients of embeddings are prone to attacks, leading to potential reverse engineering of private data. Furthermore, the server's limitation of handle only one client's training request at a time hinders parallel training, severely impacting training efficiency. In this paper, we propose a Federated Learning framework for LLM, named FL-GLM, which prevents data leakage caused by both server-side and peer-client attacks while improving training efficiency. Specifically, we first place the input block and output block on local client to prevent embedding gradient attacks from server. Secondly, we employ key-encryption during client-server communication to prevent reverse engineering attacks from peer-clients. Lastly, we employ optimization methods like client-batching or server-hierarchical, adopting different acceleration methods based on the actual computational capabilities of the server. Experimental results on NLU and generation tasks demonstrate that FL-GLM achieves comparable metrics to centralized chatGLM model, validating the effectiveness of our federated learning framework.

摘要: 私有数据比公共数据更大、质量更高，可以极大地改进大型语言模型(LLM)。然而，出于隐私方面的考虑，这些数据通常分散在多个竖井中，这使得将其安全地用于LLM培训成为一项挑战。联邦学习(FL)是一种适用于具有分布式私有数据的模型训练的理想解决方案，但FedAvg等传统框架由于对客户端的计算要求较高而不适用于LLM。另一种选择是分离学习，将大部分训练参数卸载到服务器，同时在本地训练嵌入和输出层，使其更适合LLM。尽管如此，它在安全和效率方面仍面临重大挑战。首先，嵌入的梯度容易受到攻击，从而导致对私有数据的潜在逆向工程。此外，服务器一次只能处理一个客户端的训练请求的限制阻碍了并行训练，严重影响了训练效率。本文提出了一种用于LLM的联邦学习框架FL-GLM，该框架在提高训练效率的同时，防止了服务器端攻击和对等客户端攻击引起的数据泄漏。具体地说，我们首先将输入块和输出块放置在本地客户端，以防止来自服务器的嵌入梯度攻击。其次，我们在客户-服务器通信过程中使用密钥加密，以防止来自对等客户端的反向工程攻击。最后，我们采用了客户端批处理或服务器分层等优化方法，根据服务器的实际计算能力采用不同的加速方法。在NLU和生成任务上的实验结果表明，FL-GLM达到了与集中式ChatGLM模型相当的指标，验证了联邦学习框架的有效性。



## **22. InstructTA: Instruction-Tuned Targeted Attack for Large Vision-Language Models**

DirectTA：针对大型视觉语言模型的指令调整有针对性的攻击 cs.CV

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2312.01886v3) [paper-pdf](http://arxiv.org/pdf/2312.01886v3)

**Authors**: Xunguang Wang, Zhenlan Ji, Pingchuan Ma, Zongjie Li, Shuai Wang

**Abstract**: Large vision-language models (LVLMs) have demonstrated their incredible capability in image understanding and response generation. However, this rich visual interaction also makes LVLMs vulnerable to adversarial examples. In this paper, we formulate a novel and practical targeted attack scenario that the adversary can only know the vision encoder of the victim LVLM, without the knowledge of its prompts (which are often proprietary for service providers and not publicly available) and its underlying large language model (LLM). This practical setting poses challenges to the cross-prompt and cross-model transferability of targeted adversarial attack, which aims to confuse the LVLM to output a response that is semantically similar to the attacker's chosen target text. To this end, we propose an instruction-tuned targeted attack (dubbed \textsc{InstructTA}) to deliver the targeted adversarial attack on LVLMs with high transferability. Initially, we utilize a public text-to-image generative model to "reverse" the target response into a target image, and employ GPT-4 to infer a reasonable instruction $\boldsymbol{p}^\prime$ from the target response. We then form a local surrogate model (sharing the same vision encoder with the victim LVLM) to extract instruction-aware features of an adversarial image example and the target image, and minimize the distance between these two features to optimize the adversarial example. To further improve the transferability with instruction tuning, we augment the instruction $\boldsymbol{p}^\prime$ with instructions paraphrased from GPT-4. Extensive experiments demonstrate the superiority of our proposed method in targeted attack performance and transferability. The code is available at https://github.com/xunguangwang/InstructTA.

摘要: 大型视觉语言模型在图像理解和响应生成方面表现出了令人难以置信的能力。然而，这种丰富的视觉交互也使LVLM容易受到对抗性例子的攻击。在该文中，我们提出了一种新颖而实用的定向攻击场景，攻击者只能知道受害者LVLM的视觉编码器，而不知道其提示(通常是服务提供商的专有提示，而不是公开可用的)及其底层的大语言模型(LLM)。这一实际设置对目标对抗性攻击的跨提示和跨模型可转移性提出了挑战，其目的是混淆LVLM以输出与攻击者选择的目标文本在语义上相似的响应。为此，我们提出了一种指令调谐的定向攻击(称为\Textsc{InstructTA})，以提供对具有高可转移性的LVLM的定向对抗性攻击。首先，我们利用一个公开的文本到图像的生成模型将目标响应“反转”成目标图像，并使用GPT-4从目标响应中推断出合理的指令符号。然后，我们形成一个局部代理模型(与受害者LVLM共享相同的视觉编码器)来提取对抗性图像示例和目标图像的指令感知特征，并最小化这两个特征之间的距离以优化对抗性示例。为了进一步提高指令调优的可转移性，我们用GPT-4中转译的指令扩充了指令$\boldsign{p}^\Prime$。大量实验证明了该方法在目标攻击性能和可转移性方面的优越性。代码可在https://github.com/xunguangwang/InstructTA.上获得



## **23. Inherent Challenges of Post-Hoc Membership Inference for Large Language Models**

大型语言模型事后成员推理的内在挑战 cs.CL

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2406.17975v1) [paper-pdf](http://arxiv.org/pdf/2406.17975v1)

**Authors**: Matthieu Meeus, Shubham Jain, Marek Rei, Yves-Alexandre de Montjoye

**Abstract**: Large Language Models (LLMs) are often trained on vast amounts of undisclosed data, motivating the development of post-hoc Membership Inference Attacks (MIAs) to gain insight into their training data composition. However, in this paper, we identify inherent challenges in post-hoc MIA evaluation due to potential distribution shifts between collected member and non-member datasets. Using a simple bag-of-words classifier, we demonstrate that datasets used in recent post-hoc MIAs suffer from significant distribution shifts, in some cases achieving near-perfect distinction between members and non-members. This implies that previously reported high MIA performance may be largely attributable to these shifts rather than model memorization. We confirm that randomized, controlled setups eliminate such shifts and thus enable the development and fair evaluation of new MIAs. However, we note that such randomized setups are rarely available for the latest LLMs, making post-hoc data collection still required to infer membership for real-world LLMs. As a potential solution, we propose a Regression Discontinuity Design (RDD) approach for post-hoc data collection, which substantially mitigates distribution shifts. Evaluating various MIA methods on this RDD setup yields performance barely above random guessing, in stark contrast to previously reported results. Overall, our findings highlight the challenges in accurately measuring LLM memorization and the need for careful experimental design in (post-hoc) membership inference tasks.

摘要: 大型语言模型(LLM)通常基于大量未公开的数据进行训练，这促使了后自组织成员推理攻击(MIA)的发展，以深入了解它们的训练数据组成。然而，在这篇文章中，我们识别了由于收集的成员和非成员数据集之间潜在的分布变化而导致的后MIA评估的内在挑战。使用一个简单的词袋分类器，我们证明了在最近的后自组织MIA中使用的数据集遭受了显著的分布偏移，在某些情况下实现了成员和非成员之间的近乎完美的区分。这意味着之前报道的高MIA成绩可能在很大程度上归因于这些变化，而不是模型记忆。我们确认，随机的、受控的设置消除了这种转变，从而使新的MIA的开发和公平评估成为可能。然而，我们注意到，这种随机化设置很少适用于最新的LLM，这使得仍然需要事后数据收集来推断真实世界LLM的成员资格。作为一种潜在的解决方案，我们提出了一种回归不连续设计(RDD)方法，用于后自组织数据收集，大大减轻了分布漂移。在这种RDD设置上评估各种MIA方法的性能仅略高于随机猜测，这与之前报道的结果形成了鲜明对比。总体而言，我们的发现突出了在准确测量LLM记忆方面的挑战，以及在(后即席)成员推理任务中仔细实验设计的必要性。



## **24. CoSafe: Evaluating Large Language Model Safety in Multi-Turn Dialogue Coreference**

CoSafe：评估多轮对话共指涉中的大型语言模型安全性 cs.CL

Submitted to EMNLP 2024

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2406.17626v1) [paper-pdf](http://arxiv.org/pdf/2406.17626v1)

**Authors**: Erxin Yu, Jing Li, Ming Liao, Siqi Wang, Zuchen Gao, Fei Mi, Lanqing Hong

**Abstract**: As large language models (LLMs) constantly evolve, ensuring their safety remains a critical research problem. Previous red-teaming approaches for LLM safety have primarily focused on single prompt attacks or goal hijacking. To the best of our knowledge, we are the first to study LLM safety in multi-turn dialogue coreference. We created a dataset of 1,400 questions across 14 categories, each featuring multi-turn coreference safety attacks. We then conducted detailed evaluations on five widely used open-source LLMs. The results indicated that under multi-turn coreference safety attacks, the highest attack success rate was 56% with the LLaMA2-Chat-7b model, while the lowest was 13.9% with the Mistral-7B-Instruct model. These findings highlight the safety vulnerabilities in LLMs during dialogue coreference interactions.

摘要: 随着大型语言模型（LLM）的不断发展，确保其安全性仍然是一个关键的研究问题。之前的LLM安全红色团队方法主要集中在单次提示攻击或目标劫持上。据我们所知，我们是第一个在多回合对话共指涉中研究LLM安全性的公司。我们创建了一个包含14个类别的1，400个问题的数据集，每个问题都具有多轮共指安全攻击。然后，我们对五个广泛使用的开源LLM进行了详细的评估。结果表明，在多轮共指安全攻击下，LLaMA 2-Chat-7 b模型的攻击成功率最高为56%，而Mistral-7 B-Direct模型的攻击成功率最低为13.9%。这些发现凸显了LLM在对话共指互动期间的安全漏洞。



## **25. Practical Membership Inference Attacks against Fine-tuned Large Language Models via Self-prompt Calibration**

通过自我提示校准对微调大型语言模型的实用成员推断攻击 cs.CL

Repo: https://github.com/wjfu99/MIA-LLMs

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2311.06062v3) [paper-pdf](http://arxiv.org/pdf/2311.06062v3)

**Authors**: Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, Tao Jiang

**Abstract**: Membership Inference Attacks (MIA) aim to infer whether a target data record has been utilized for model training or not. Prior attempts have quantified the privacy risks of language models (LMs) via MIAs, but there is still no consensus on whether existing MIA algorithms can cause remarkable privacy leakage on practical Large Language Models (LLMs). Existing MIAs designed for LMs can be classified into two categories: reference-free and reference-based attacks. They are both based on the hypothesis that training records consistently strike a higher probability of being sampled. Nevertheless, this hypothesis heavily relies on the overfitting of target models, which will be mitigated by multiple regularization methods and the generalization of LLMs. The reference-based attack seems to achieve promising effectiveness in LLMs, which measures a more reliable membership signal by comparing the probability discrepancy between the target model and the reference model. However, the performance of reference-based attack is highly dependent on a reference dataset that closely resembles the training dataset, which is usually inaccessible in the practical scenario. Overall, existing MIAs are unable to effectively unveil privacy leakage over practical fine-tuned LLMs that are overfitting-free and private. We propose a Membership Inference Attack based on Self-calibrated Probabilistic Variation (SPV-MIA). Specifically, since memorization in LLMs is inevitable during the training process and occurs before overfitting, we introduce a more reliable membership signal, probabilistic variation, which is based on memorization rather than overfitting. Furthermore, we introduce a self-prompt approach, which constructs the dataset to fine-tune the reference model by prompting the target LLM itself. In this manner, the adversary can collect a dataset with a similar distribution from public APIs.

摘要: 成员关系推理攻击(MIA)的目的是推断目标数据记录是否已被用于模型训练。以往的研究已经通过MIA量化了语言模型的隐私风险，但对于现有的MIA算法是否会在实际的大型语言模型上造成显著的隐私泄漏，目前还没有达成共识。现有的针对LMS设计的MIA可以分为两类：无引用攻击和基于引用攻击。它们都是基于这样的假设，即培训记录始终具有更高的被抽样概率。然而，这一假设在很大程度上依赖于目标模型的过度拟合，而多种正则化方法和LLMS的推广将缓解这一问题。基于参考的攻击在LLMS中似乎取得了很好的效果，它通过比较目标模型和参考模型之间的概率差异来衡量更可靠的成员信号。然而，基于参考的攻击的性能高度依赖于与训练数据集非常相似的参考数据集，这在实际场景中通常是不可访问的。总体而言，现有的MIA无法有效地揭示实用的微调LLM的隐私泄露，这些LLM是免装修和私密的。提出了一种基于自校准概率变异的成员推理攻击(SPV-MIA)。具体地说，由于LLMS中的记忆在训练过程中是不可避免的，并且发生在过适应之前，因此我们引入了一种更可靠的隶属度信号-概率变异，它基于记忆而不是过适应。此外，我们引入了一种自我提示的方法，该方法构建数据集，通过提示目标LLM本身来微调参考模型。通过这种方式，攻击者可以从公共API收集具有类似分布的数据集。



## **26. BEEAR: Embedding-based Adversarial Removal of Safety Backdoors in Instruction-tuned Language Models**

BEEAR：基于嵌入的对抗性消除教学调整语言模型中的安全后门 cs.CR

**SubmitDate**: 2024-06-24    [abs](http://arxiv.org/abs/2406.17092v1) [paper-pdf](http://arxiv.org/pdf/2406.17092v1)

**Authors**: Yi Zeng, Weiyu Sun, Tran Ngoc Huynh, Dawn Song, Bo Li, Ruoxi Jia

**Abstract**: Safety backdoor attacks in large language models (LLMs) enable the stealthy triggering of unsafe behaviors while evading detection during normal interactions. The high dimensionality of potential triggers in the token space and the diverse range of malicious behaviors make this a critical challenge. We present BEEAR, a mitigation approach leveraging the insight that backdoor triggers induce relatively uniform drifts in the model's embedding space. Our bi-level optimization method identifies universal embedding perturbations that elicit unwanted behaviors and adjusts the model parameters to reinforce safe behaviors against these perturbations. Experiments show BEEAR reduces the success rate of RLHF time backdoor attacks from >95% to <1% and from 47% to 0% for instruction-tuning time backdoors targeting malicious code generation, without compromising model utility. Requiring only defender-defined safe and unwanted behaviors, BEEAR represents a step towards practical defenses against safety backdoors in LLMs, providing a foundation for further advancements in AI safety and security.

摘要: 大型语言模型(LLM)中的安全后门攻击能够在正常交互期间躲避检测的同时，秘密触发不安全行为。令牌空间中潜在触发器的高维度和不同范围的恶意行为使这成为一个严峻的挑战。我们提出了BEEAR，这是一种缓解方法，利用了后门触发在模型的嵌入空间中导致相对均匀的漂移这一观点。我们的双层优化方法识别引起不想要的行为的普遍嵌入扰动，并调整模型参数以加强针对这些扰动的安全行为。实验表明，对于针对恶意代码生成的指令调优时间后门，BEEAR将RLHF时间后门攻击的成功率从>95%降低到<1%，将指令调整时间后门攻击的成功率从47%降低到0%，而不影响模型的实用性。BEEAR只需要防御者定义的安全和不受欢迎的行为，代表着朝着针对LLMS中的安全后门的实际防御迈出了一步，为人工智能安全和安保的进一步进步奠定了基础。



## **27. Pandora's White-Box: Precise Training Data Detection and Extraction in Large Language Models**

潘多拉的白盒：大型语言模型中的精确训练数据检测和提取 cs.CR

**SubmitDate**: 2024-06-24    [abs](http://arxiv.org/abs/2402.17012v3) [paper-pdf](http://arxiv.org/pdf/2402.17012v3)

**Authors**: Jeffrey G. Wang, Jason Wang, Marvin Li, Seth Neel

**Abstract**: In this paper we develop state-of-the-art privacy attacks against Large Language Models (LLMs), where an adversary with some access to the model tries to learn something about the underlying training data. Our headline results are new membership inference attacks (MIAs) against pretrained LLMs that perform hundreds of times better than baseline attacks, and a pipeline showing that over 50% (!) of the fine-tuning dataset can be extracted from a fine-tuned LLM in natural settings. We consider varying degrees of access to the underlying model, pretraining and fine-tuning data, and both MIAs and training data extraction. For pretraining data, we propose two new MIAs: a supervised neural network classifier that predicts training data membership on the basis of (dimensionality-reduced) model gradients, as well as a variant of this attack that only requires logit access to the model by leveraging recent model-stealing work on LLMs. To our knowledge this is the first MIA that explicitly incorporates model-stealing information. Both attacks outperform existing black-box baselines, and our supervised attack closes the gap between MIA attack success against LLMs and the strongest known attacks for other machine learning models. In fine-tuning, we find that a simple attack based on the ratio of the loss between the base and fine-tuned models is able to achieve near-perfect MIA performance; we then leverage our MIA to extract a large fraction of the fine-tuning dataset from fine-tuned Pythia and Llama models. Our code is available at github.com/safr-ai-lab/pandora-llm.

摘要: 在本文中，我们开发了针对大型语言模型(LLM)的最先进的隐私攻击，其中对该模型具有一定访问权限的对手试图了解一些关于潜在训练数据的信息。我们的主要结果是针对预先训练的LLM的新成员推理攻击(MIA)，其性能比基线攻击高数百倍，并且管道显示超过50%(！)可以在自然设置下从微调的LLM中提取精调数据集的。我们考虑了不同程度的访问底层模型、预训练和微调数据，以及MIA和训练数据提取。对于预训练数据，我们提出了两个新的MIA：一个有监督的神经网络分类器，它基于(降维)模型梯度预测训练数据的成员资格，以及这种攻击的一个变体，它只需要通过利用LLMS上最近的模型窃取工作来对模型进行Logit访问。据我们所知，这是第一个明确纳入模型窃取信息的MIA。这两种攻击都超过了现有的黑盒基线，我们的监督攻击缩小了针对LLMS的MIA攻击成功与针对其他机器学习模型的已知最强攻击之间的差距。在微调中，我们发现基于基本模型和微调模型之间的损失比率的简单攻击能够获得近乎完美的MIA性能；然后，我们利用我们的MIA从微调的Pythia和Llama模型中提取很大一部分微调数据集。我们的代码可以在githorb.com/Safr-ai-lab/pandora-llm上找到。



## **28. Versatile Backdoor Attack with Visible, Semantic, Sample-Specific, and Compatible Triggers**

具有可见、语义、样本特定且兼容触发器的多功能后门攻击 cs.CV

23 pages, 21 figures, 18 tables

**SubmitDate**: 2024-06-24    [abs](http://arxiv.org/abs/2306.00816v4) [paper-pdf](http://arxiv.org/pdf/2306.00816v4)

**Authors**: Ruotong Wang, Hongrui Chen, Zihao Zhu, Li Liu, Baoyuan Wu

**Abstract**: Deep neural networks (DNNs) can be manipulated to exhibit specific behaviors when exposed to specific trigger patterns, without affecting their performance on benign samples, dubbed \textit{backdoor attack}. Currently, implementing backdoor attacks in physical scenarios still faces significant challenges. Physical attacks are labor-intensive and time-consuming, and the triggers are selected in a manual and heuristic way. Moreover, expanding digital attacks to physical scenarios faces many challenges due to their sensitivity to visual distortions and the absence of counterparts in the real world. To address these challenges, we define a novel trigger called the \textbf{V}isible, \textbf{S}emantic, \textbf{S}ample-Specific, and \textbf{C}ompatible (VSSC) trigger, to achieve effective, stealthy and robust simultaneously, which can also be effectively deployed in the physical scenario using corresponding objects. To implement the VSSC trigger, we propose an automated pipeline comprising three modules: a trigger selection module that systematically identifies suitable triggers leveraging large language models, a trigger insertion module that employs generative models to seamlessly integrate triggers into images, and a quality assessment module that ensures the natural and successful insertion of triggers through vision-language models. Extensive experimental results and analysis validate the effectiveness, stealthiness, and robustness of the VSSC trigger. It can not only maintain robustness under visual distortions but also demonstrates strong practicality in the physical scenario. We hope that the proposed VSSC trigger and implementation approach could inspire future studies on designing more practical triggers in backdoor attacks.

摘要: 深度神经网络(DNN)可以在暴露于特定触发模式时表现出特定的行为，而不会影响它们在良性样本上的性能，这被称为\textit{后门攻击}。目前，在物理场景中实施后门攻击仍然面临重大挑战。物理攻击劳动强度大、耗时长，触发点选择采用人工和启发式方式。此外，将数字攻击扩展到物理场景面临许多挑战，因为它们对视觉扭曲很敏感，而且现实世界中没有对应的攻击。为了应对这些挑战，我们定义了一种新的触发器，称为\Textbf{V}可扩展的、\Textbf{S}可扩展的、\Textbf{S}全特定的和\Textbf{C}兼容的(VSSC)触发器，以实现有效、隐蔽和健壮的同时，也可以使用相应的对象在物理场景中有效部署。为了实现VSSC触发器，我们提出了一个包括三个模块的自动化流水线：利用大型语言模型系统地识别合适的触发器的触发器选择模块，使用生成式模型无缝地将触发器集成到图像中的触发器插入模块，以及通过视觉语言模型确保触发器的自然和成功插入的质量评估模块。大量的实验结果和分析验证了VSSC触发器的有效性、隐蔽性和稳健性。该算法不仅能在视觉失真下保持较强的鲁棒性，而且在实际场景中表现出较强的实用性。我们希望提出的VSSC触发器和实现方法可以启发未来设计更实用的后门攻击触发器的研究。



## **29. ReCaLL: Membership Inference via Relative Conditional Log-Likelihoods**

ReCaLL：通过相对条件日志可能性的成员推断 cs.CL

**SubmitDate**: 2024-06-23    [abs](http://arxiv.org/abs/2406.15968v1) [paper-pdf](http://arxiv.org/pdf/2406.15968v1)

**Authors**: Roy Xie, Junlin Wang, Ruomin Huang, Minxing Zhang, Rong Ge, Jian Pei, Neil Zhenqiang Gong, Bhuwan Dhingra

**Abstract**: The rapid scaling of large language models (LLMs) has raised concerns about the transparency and fair use of the pretraining data used for training them. Detecting such content is challenging due to the scale of the data and limited exposure of each instance during training. We propose ReCaLL (Relative Conditional Log-Likelihood), a novel membership inference attack (MIA) to detect LLMs' pretraining data by leveraging their conditional language modeling capabilities. ReCaLL examines the relative change in conditional log-likelihoods when prefixing target data points with non-member context. Our empirical findings show that conditioning member data on non-member prefixes induces a larger decrease in log-likelihood compared to non-member data. We conduct comprehensive experiments and show that ReCaLL achieves state-of-the-art performance on the WikiMIA dataset, even with random and synthetic prefixes, and can be further improved using an ensemble approach. Moreover, we conduct an in-depth analysis of LLMs' behavior with different membership contexts, providing insights into how LLMs leverage membership information for effective inference at both the sequence and token level.

摘要: 大型语言模型(LLM)的快速扩展引起了人们对用于培训它们的预培训数据的透明度和公平使用的关注。由于数据的规模和每个实例在培训期间暴露的有限，检测此类内容具有挑战性。本文利用LLMS的条件语言建模能力，提出了一种新的成员关系推理攻击(MIA)，即相对条件对数似然函数(Recall)，用于检测LLMS的预训练数据。当使用非成员上下文作为目标数据点的前缀时，Recall检查条件对数似然的相对变化。我们的经验结果表明，与非成员数据相比，以非成员前缀为条件的成员数据导致了更大的对数似然下降。我们进行了全面的实验，证明了Recall在WikiMIA数据集上取得了最先进的性能，即使使用随机和合成前缀也是如此，并且可以使用集成方法进一步改进。此外，我们深入分析了LLMS在不同成员关系上下文下的行为，为LLM如何利用成员关系信息在序列和令牌级别进行有效推理提供了深入的见解。



## **30. Large Language Models for Link Stealing Attacks Against Graph Neural Networks**

针对图神经网络的链接窃取攻击的大型语言模型 cs.LG

**SubmitDate**: 2024-06-22    [abs](http://arxiv.org/abs/2406.16963v1) [paper-pdf](http://arxiv.org/pdf/2406.16963v1)

**Authors**: Faqian Guan, Tianqing Zhu, Hui Sun, Wanlei Zhou, Philip S. Yu

**Abstract**: Graph data contains rich node features and unique edge information, which have been applied across various domains, such as citation networks or recommendation systems. Graph Neural Networks (GNNs) are specialized for handling such data and have shown impressive performance in many applications. However, GNNs may contain of sensitive information and susceptible to privacy attacks. For example, link stealing is a type of attack in which attackers infer whether two nodes are linked or not. Previous link stealing attacks primarily relied on posterior probabilities from the target GNN model, neglecting the significance of node features. Additionally, variations in node classes across different datasets lead to different dimensions of posterior probabilities. The handling of these varying data dimensions posed a challenge in using a single model to effectively conduct link stealing attacks on different datasets. To address these challenges, we introduce Large Language Models (LLMs) to perform link stealing attacks on GNNs. LLMs can effectively integrate textual features and exhibit strong generalizability, enabling attacks to handle diverse data dimensions across various datasets. We design two distinct LLM prompts to effectively combine textual features and posterior probabilities of graph nodes. Through these designed prompts, we fine-tune the LLM to adapt to the link stealing attack task. Furthermore, we fine-tune the LLM using multiple datasets and enable the LLM to learn features from different datasets simultaneously. Experimental results show that our approach significantly enhances the performance of existing link stealing attack tasks in both white-box and black-box scenarios. Our method can execute link stealing attacks across different datasets using only a single model, making link stealing attacks more applicable to real-world scenarios.

摘要: 图数据包含丰富的节点特征和独特的边缘信息，已经应用于不同的领域，如引文网络或推荐系统。图形神经网络(GNN)专门用于处理这类数据，并在许多应用中显示出令人印象深刻的性能。然而，GNN可能包含敏感信息并容易受到隐私攻击。例如，链接窃取是一种攻击者推断两个节点是否链接的攻击类型。以往的链路窃取攻击主要依赖于目标GNN模型的后验概率，忽略了节点特征的重要性。此外，不同数据集中节点类别的变化导致后验概率的不同维度。对这些不同数据维度的处理给使用单一模型对不同数据集有效地执行链接窃取攻击带来了挑战。为了应对这些挑战，我们引入了大型语言模型(LLM)来执行针对GNN的链接窃取攻击。LLMS可以有效地集成文本特征，并表现出很强的泛化能力，使攻击能够跨各种数据集处理不同的数据维度。我们设计了两个不同的LLM提示，有效地结合了文本特征和图节点的后验概率。通过这些设计的提示，我们对LLM进行了微调，以适应链路窃取攻击任务。此外，我们使用多个数据集对LLM进行微调，使LLM能够同时从不同的数据集中学习特征。实验结果表明，该方法在白盒和黑盒场景下均能显著提高已有链路窃取攻击任务的性能。该方法可以在不同的数据集中使用单一的模型来执行链接窃取攻击，使得链接窃取攻击更适用于现实世界的场景。



## **31. Efficient Adversarial Training in LLMs with Continuous Attacks**

在具有持续攻击的LLC中进行有效的对抗训练 cs.LG

19 pages, 4 figures

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2405.15589v2) [paper-pdf](http://arxiv.org/pdf/2405.15589v2)

**Authors**: Sophie Xhonneux, Alessandro Sordoni, Stephan Günnemann, Gauthier Gidel, Leo Schwinn

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that can bypass their safety guardrails. In many domains, adversarial training has proven to be one of the most promising methods to reliably improve robustness against such attacks. Yet, in the context of LLMs, current methods for adversarial training are hindered by the high computational costs required to perform discrete adversarial attacks at each training iteration. We address this problem by instead calculating adversarial attacks in the continuous embedding space of the LLM, which is orders of magnitudes more efficient. We propose a fast adversarial training algorithm (C-AdvUL) composed of two losses: the first makes the model robust on continuous embedding attacks computed on an adversarial behaviour dataset; the second ensures the usefulness of the final model by fine-tuning on utility data. Moreover, we introduce C-AdvIPO, an adversarial variant of IPO that does not require utility data for adversarially robust alignment. Our empirical evaluation on four models from different families (Gemma, Phi3, Mistral, Zephyr) and at different scales (2B, 3.8B, 7B) shows that both algorithms substantially enhance LLM robustness against discrete attacks (GCG, AutoDAN, PAIR), while maintaining utility. Our results demonstrate that robustness to continuous perturbations can extrapolate to discrete threat models. Thereby, we present a path toward scalable adversarial training algorithms for robustly aligning LLMs.

摘要: 大型语言模型(LLM)容易受到敌意攻击，这些攻击可以绕过它们的安全护栏。在许多领域，对抗性训练已被证明是可靠地提高对此类攻击的稳健性的最有前途的方法之一。然而，在LLMS的背景下，当前的对抗性训练方法受到在每次训练迭代中执行离散对抗性攻击所需的高计算成本的阻碍。我们通过在LLM的连续嵌入空间中计算敌意攻击来解决这个问题，该空间的效率要高出几个数量级。我们提出了一种快速对抗性训练算法(C-AdvUL)，该算法由两个损失组成：第一个损失使模型对基于对抗行为数据集的连续嵌入攻击具有健壮性；第二个损失通过对效用数据的微调来确保最终模型的有效性。此外，我们引入了C-AdvIPO，这是IPO的一个对抗性变体，不需要效用数据来进行对抗性强健的比对。我们在不同家族(Gema，Phi3，Mistral，Zephy)和不同尺度(2B，3.8B，7B)的四个模型上的实验评估表明，这两种算法在保持实用性的同时，显著提高了LLM对离散攻击(GCG，AutoDAN，Pair)的稳健性。我们的结果表明，对连续扰动的稳健性可以外推到离散威胁模型。因此，我们提出了一条可扩展的对抗性训练算法，用于鲁棒对齐LLM。



## **32. Logicbreaks: A Framework for Understanding Subversion of Rule-based Inference**

Logicbreaks：理解基于规则的推理颠覆的框架 cs.AI

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2407.00075v1) [paper-pdf](http://arxiv.org/pdf/2407.00075v1)

**Authors**: Anton Xue, Avishree Khare, Rajeev Alur, Surbhi Goel, Eric Wong

**Abstract**: We study how to subvert language models from following the rules. We model rule-following as inference in propositional Horn logic, a mathematical system in which rules have the form "if $P$ and $Q$, then $R$" for some propositions $P$, $Q$, and $R$. We prove that although transformers can faithfully abide by such rules, maliciously crafted prompts can nevertheless mislead even theoretically constructed models. Empirically, we find that attacks on our theoretical models mirror popular attacks on large language models. Our work suggests that studying smaller theoretical models can help understand the behavior of large language models in rule-based settings like logical reasoning and jailbreak attacks.

摘要: 我们研究如何颠覆语言模型，使其不再遵循规则。我们将规则遵循建模为命题Horn逻辑中的推理，这是一个数学系统，其中对于某些命题$P $、$Q$和$R$，规则的形式为“如果$P$和$Q $，那么$R $”。我们证明，尽管变形金刚可以忠实地遵守这些规则，但恶意制作的提示仍然可以误导理论上构建的模型。从经验上看，我们发现对理论模型的攻击反映了对大型语言模型的流行攻击。我们的工作表明，研究较小的理论模型可以帮助理解大型语言模型在逻辑推理和越狱攻击等基于规则的环境中的行为。



## **33. From LLMs to MLLMs: Exploring the Landscape of Multimodal Jailbreaking**

从LLM到MLLM：探索多模式越狱的格局 cs.CL

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2406.14859v1) [paper-pdf](http://arxiv.org/pdf/2406.14859v1)

**Authors**: Siyuan Wang, Zhuohan Long, Zhihao Fan, Zhongyu Wei

**Abstract**: The rapid development of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) has exposed vulnerabilities to various adversarial attacks. This paper provides a comprehensive overview of jailbreaking research targeting both LLMs and MLLMs, highlighting recent advancements in evaluation benchmarks, attack techniques and defense strategies. Compared to the more advanced state of unimodal jailbreaking, multimodal domain remains underexplored. We summarize the limitations and potential research directions of multimodal jailbreaking, aiming to inspire future research and further enhance the robustness and security of MLLMs.

摘要: 大型语言模型（LLM）和多模式大型语言模型（MLLM）的快速发展暴露了各种对抗攻击的脆弱性。本文全面概述了针对LLM和MLLM的越狱研究，重点介绍了评估基准、攻击技术和防御策略方面的最新进展。与更先进的单模式越狱相比，多模式领域仍然被探索不足。我们总结了多模式越狱的局限性和潜在研究方向，旨在启发未来的研究并进一步增强MLLM的稳健性和安全性。



## **34. Uncovering Safety Risks of Large Language Models through Concept Activation Vector**

通过概念激活载体揭示大型语言模型的安全风险 cs.CL

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2404.12038v2) [paper-pdf](http://arxiv.org/pdf/2404.12038v2)

**Authors**: Zhihao Xu, Ruixuan Huang, Shuai Wang, Xiting Wang

**Abstract**: Despite careful safety alignment, current large language models (LLMs) remain vulnerable to various attacks. To further unveil the safety risks of LLMs, we introduce a Safety Concept Activation Vector (SCAV) framework, which effectively guides the attacks by accurately interpreting LLMs' safety mechanisms. We then develop an SCAV-guided attack method that can generate both attack prompts and embedding-level attacks with automatically selected perturbation hyperparameters. Both automatic and human evaluations demonstrate that our attack method significantly improves the attack success rate and response quality while requiring less training data. Additionally, we find that our generated attack prompts may be transferable to GPT-4, and the embedding-level attacks may also be transferred to other white-box LLMs whose parameters are known. Our experiments further uncover the safety risks present in current LLMs. For example, we find that six out of seven open-source LLMs that we attack consistently provide relevant answers to more than 85\% malicious instructions. Finally, we provide insights into the safety mechanism of LLMs.

摘要: 尽管进行了仔细的安全调整，但当前的大型语言模型(LLM)仍然容易受到各种攻击。为了进一步揭示LLMS的安全隐患，我们引入了安全概念激活向量(SCAV)框架，通过准确解释LLMS的安全机制来有效地指导攻击。然后，我们开发了一种SCAV引导的攻击方法，该方法可以生成攻击提示和带有自动选择的扰动超参数的嵌入级攻击。自动和人工评估都表明，我们的攻击方法在需要更少的训练数据的情况下，显著地提高了攻击成功率和响应质量。此外，我们发现我们生成的攻击提示可以转移到GPT-4上，嵌入级攻击也可以转移到参数已知的其他白盒LLM上。我们的实验进一步揭示了当前LLM中存在的安全风险。例如，我们发现，我们攻击的七个开源LLM中有六个始终为超过85%的恶意指令提供相关答案。最后，我们对LLMS的安全机制提供了见解。



## **35. FedSecurity: Benchmarking Attacks and Defenses in Federated Learning and Federated LLMs**

FedSecurity：联邦学习和联邦LLM中的攻击和防御基准 cs.CR

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2306.04959v5) [paper-pdf](http://arxiv.org/pdf/2306.04959v5)

**Authors**: Shanshan Han, Baturalp Buyukates, Zijian Hu, Han Jin, Weizhao Jin, Lichao Sun, Xiaoyang Wang, Wenxuan Wu, Chulin Xie, Yuhang Yao, Kai Zhang, Qifan Zhang, Yuhui Zhang, Carlee Joe-Wong, Salman Avestimehr, Chaoyang He

**Abstract**: This paper introduces FedSecurity, an end-to-end benchmark that serves as a supplementary component of the FedML library for simulating adversarial attacks and corresponding defense mechanisms in Federated Learning (FL). FedSecurity eliminates the need for implementing the fundamental FL procedures, e.g., FL training and data loading, from scratch, thus enables users to focus on developing their own attack and defense strategies. It contains two key components, including FedAttacker that conducts a variety of attacks during FL training, and FedDefender that implements defensive mechanisms to counteract these attacks. FedSecurity has the following features: i) It offers extensive customization options to accommodate a broad range of machine learning models (e.g., Logistic Regression, ResNet, and GAN) and FL optimizers (e.g., FedAVG, FedOPT, and FedNOVA); ii) it enables exploring the effectiveness of attacks and defenses across different datasets and models; and iii) it supports flexible configuration and customization through a configuration file and some APIs. We further demonstrate FedSecurity's utility and adaptability through federated training of Large Language Models (LLMs) to showcase its potential on a wide range of complex applications.

摘要: 本文介绍了FedSecurity，这是一个端到端的基准测试，作为FedML库的补充组件，用于模拟联邦学习中的对抗性攻击和相应的防御机制。FedSecurity不需要从头开始实施基本的FL程序，例如FL训练和数据加载，从而使用户能够专注于开发他们自己的攻击和防御策略。它包含两个关键组件，包括在FL训练期间进行各种攻击的FedAttacker和实现防御机制以对抗这些攻击的FedDefender。FedSecurity具有以下功能：i)它提供广泛的定制选项，以适应广泛的机器学习模型(例如Logistic回归、ResNet和GAN)和FL优化器(例如FedAVG、FedOPT和FedNOVA)；ii)它能够跨不同的数据集和模型探索攻击和防御的有效性；iii)它通过一个配置文件和一些API支持灵活的配置和定制。通过对大型语言模型(LLM)的联合训练，我们进一步展示了FedSecurity的实用性和适应性，以展示其在广泛的复杂应用中的潜力。



## **36. Unmasking Database Vulnerabilities: Zero-Knowledge Schema Inference Attacks in Text-to-SQL Systems**

揭露数据库漏洞：文本转SQL系统中的零知识模式推理攻击 cs.CL

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.14545v1) [paper-pdf](http://arxiv.org/pdf/2406.14545v1)

**Authors**: Đorđe Klisura, Anthony Rios

**Abstract**: Relational databases are integral to modern information systems, serving as the foundation for storing, querying, and managing data efficiently and effectively. Advancements in large language modeling have led to the emergence of text-to-SQL technologies, significantly enhancing the querying and extracting of information from these databases and raising concerns about privacy and security. Our research extracts the database schema elements underlying a text-to-SQL model. Knowledge of the schema can make attacks such as SQL injection easier. By asking specially crafted questions, we have developed a zero-knowledge framework designed to probe various database schema elements without knowledge of the database itself. The text-to-SQL models then process these questions to produce an output that we use to uncover the structure of the database schema. We apply it to specialized text-to-SQL models fine-tuned on text-SQL pairs and generative language models used for SQL generation. Overall, we can reconstruct the table names with an F1 of nearly .75 for fine-tuned models and .96 for generative.

摘要: 关系数据库是现代信息系统不可或缺的组成部分，是高效存储、查询和管理数据的基础。大型语言建模的进步导致了文本到SQL技术的出现，极大地增强了从这些数据库查询和提取信息的能力，并引发了对隐私和安全的担忧。我们的研究提取了Text-to-SQL模型下的数据库模式元素。了解模式可以使SQL注入等攻击变得更容易。通过提出精心设计的问题，我们开发了一个零知识框架，旨在探索各种数据库模式元素，而不需要了解数据库本身。然后，Text-to-SQL模型处理这些问题，以产生我们用来揭示数据库模式结构的输出。我们将其应用于专门的文本到SQL模型，这些模型在文本-SQL对和用于生成SQL的生成语言模型上进行了微调。总体而言，我们可以重新构建表名，对于微调模型，F1接近0.75，对于生成性模型，F1接近0.96。



## **37. Jailbreaking as a Reward Misspecification Problem**

越狱是奖励错误指定问题 cs.LG

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.14393v1) [paper-pdf](http://arxiv.org/pdf/2406.14393v1)

**Authors**: Zhihui Xie, Jiahui Gao, Lei Li, Zhenguo Li, Qi Liu, Lingpeng Kong

**Abstract**: The widespread adoption of large language models (LLMs) has raised concerns about their safety and reliability, particularly regarding their vulnerability to adversarial attacks. In this paper, we propose a novel perspective that attributes this vulnerability to reward misspecification during the alignment process. We introduce a metric ReGap to quantify the extent of reward misspecification and demonstrate its effectiveness and robustness in detecting harmful backdoor prompts. Building upon these insights, we present ReMiss, a system for automated red teaming that generates adversarial prompts against various target aligned LLMs. ReMiss achieves state-of-the-art attack success rates on the AdvBench benchmark while preserving the human readability of the generated prompts. Detailed analysis highlights the unique advantages brought by the proposed reward misspecification objective compared to previous methods.

摘要: 大型语言模型（LLM）的广泛采用引发了人们对其安全性和可靠性的担忧，特别是对其容易受到对抗攻击的影响。在本文中，我们提出了一种新颖的视角，将此漏洞归因于对齐过程中的奖励错误指定。我们引入了一个指标ReGap来量化奖励错误指定的程度，并展示其在检测有害后门提示方面的有效性和稳健性。在这些见解的基础上，我们介绍了ReMiss，这是一个用于自动化红色分组的系统，可以针对各种目标对齐的LLM生成对抗提示。ReMiss在AdvBench基准上实现了最先进的攻击成功率，同时保留了生成提示的人类可读性。与以前的方法相比，详细的分析强调了拟议的奖励错误指定目标所带来的独特优势。



## **38. Safety of Multimodal Large Language Models on Images and Texts**

图像和文本上多模式大型语言模型的安全性 cs.CV

Accepted at IJCAI2024

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2402.00357v3) [paper-pdf](http://arxiv.org/pdf/2402.00357v3)

**Authors**: Xin Liu, Yichen Zhu, Yunshi Lan, Chao Yang, Yu Qiao

**Abstract**: Attracted by the impressive power of Multimodal Large Language Models (MLLMs), the public is increasingly utilizing them to improve the efficiency of daily work. Nonetheless, the vulnerabilities of MLLMs to unsafe instructions bring huge safety risks when these models are deployed in real-world scenarios. In this paper, we systematically survey current efforts on the evaluation, attack, and defense of MLLMs' safety on images and text. We begin with introducing the overview of MLLMs on images and text and understanding of safety, which helps researchers know the detailed scope of our survey. Then, we review the evaluation datasets and metrics for measuring the safety of MLLMs. Next, we comprehensively present attack and defense techniques related to MLLMs' safety. Finally, we analyze several unsolved issues and discuss promising research directions. The latest papers are continually collected at https://github.com/isXinLiu/MLLM-Safety-Collection.

摘要: 受多模式大型语言模型（MLLM）令人印象深刻的力量的吸引，公众越来越多地利用它们来提高日常工作的效率。尽管如此，当这些模型部署在现实世界场景中时，MLLM对不安全指令的脆弱性带来了巨大的安全风险。在本文中，我们系统地调查了当前对MLLM图像和文本安全性的评估、攻击和防御方面的工作。我们首先介绍MLLM关于图像和文本的概述以及对安全性的理解，这有助于研究人员了解我们调查的详细范围。然后，我们审查用于衡量MLLM安全性的评估数据集和指标。接下来，我们全面介绍与MLLM安全相关的攻击和防御技术。最后，我们分析了几个尚未解决的问题并讨论了有前途的研究方向。https://github.com/isXinLiu/MLLM-Safety-Collection不断收集最新论文。



## **39. Are you still on track!? Catching LLM Task Drift with Activations**

你还在正轨上吗！？通过激活捕捉LLM任务漂移 cs.CR

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.00799v3) [paper-pdf](http://arxiv.org/pdf/2406.00799v3)

**Authors**: Sahar Abdelnabi, Aideen Fay, Giovanni Cherubin, Ahmed Salem, Mario Fritz, Andrew Paverd

**Abstract**: Large Language Models (LLMs) are routinely used in retrieval-augmented applications to orchestrate tasks and process inputs from users and other sources. These inputs, even in a single LLM interaction, can come from a variety of sources, of varying trustworthiness and provenance. This opens the door to prompt injection attacks, where the LLM receives and acts upon instructions from supposedly data-only sources, thus deviating from the user's original instructions. We define this as task drift, and we propose to catch it by scanning and analyzing the LLM's activations. We compare the LLM's activations before and after processing the external input in order to detect whether this input caused instruction drift. We develop two probing methods and find that simply using a linear classifier can detect drift with near perfect ROC AUC on an out-of-distribution test set. We show that this approach generalizes surprisingly well to unseen task domains, such as prompt injections, jailbreaks, and malicious instructions, without being trained on any of these attacks. Our setup does not require any modification of the LLM (e.g., fine-tuning) or any text generation, thus maximizing deployability and cost efficiency and avoiding reliance on unreliable model output. To foster future research on activation-based task inspection, decoding, and interpretability, we will release our large-scale TaskTracker toolkit, comprising a dataset of over 500K instances, representations from 4 SoTA language models, and inspection tools.

摘要: 大型语言模型(LLM)通常用于检索增强的应用程序中，以协调任务并处理来自用户和其他来源的输入。这些输入，即使是在单个LLM交互中，也可以来自各种来源，具有不同的可信度和出处。这为即时注入攻击打开了大门，在这种情况下，LLM接收来自假定仅限数据的来源的指令并对其采取行动，从而偏离用户的原始指令。我们将其定义为任务漂移，并建议通过扫描和分析LLM的激活来捕获它。我们比较LLM在处理外部输入之前和之后的激活，以检测该输入是否导致指令漂移。我们开发了两种探测方法，发现简单地使用线性分类器可以在非分布测试集上以接近完美的ROC AUC来检测漂移。我们表明，这种方法对于看不见的任务领域(如提示注入、越狱和恶意指令)的泛化效果出奇地好，而且没有接受过任何这些攻击的培训。我们的设置不需要对LLM进行任何修改(例如，微调)或任何文本生成，从而最大限度地提高可部署性和成本效益，并避免依赖不可靠的模型输出。为了促进未来对基于激活的任务检测、解码和可解释性的研究，我们将发布我们的大型TaskTracker工具包，其中包括超过50万个实例的数据集、来自4个SOTA语言模型的表示和检测工具。



## **40. FewFedPIT: Towards Privacy-preserving and Few-shot Federated Instruction Tuning**

FewFedPIT：迈向隐私保护和少镜头联邦指令调优 cs.CR

Work in progress

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2403.06131v2) [paper-pdf](http://arxiv.org/pdf/2403.06131v2)

**Authors**: Zhuo Zhang, Jingyuan Zhang, Jintao Huang, Lizhen Qu, Hongzhi Zhang, Qifan Wang, Xun Zhou, Zenglin Xu

**Abstract**: Instruction tuning has been identified as a crucial technique for optimizing the performance of large language models (LLMs) in generating human-aligned responses. Nonetheless, gathering diversified and superior-quality instruction data for such tuning presents notable obstacles, especially in domains with rigid privacy provisions. Federated instruction tuning (FedIT) has emerged as a promising solution, by consolidating collaborative training across multiple data owners, thereby resulting in a privacy-preserving learning model. However, FedIT encounters limitations such as scarcity of instructional data and risk of exposure to training data extraction attacks. In this paper, we propose a novel federated algorithm, FewFedPIT, designed to simultaneously enhance privacy protection and model performance of federated few-shot learning. FewFedPITcomprises three vital components on the client side: (1) synthetic data generation, which utilizes LLMs' in-context learning capacity to generate synthetic data autonomously, thus expanding the local database; (2) parameter isolation training, which individually updates the public parameters in the synthetic data and the private parameters in the local data, consequently mitigating the noise impact of the synthetic data; (3) local aggregation sharing, which mixes public and private parameters before uploading, effectively preventing data extraction attacks. Extensive experiments on three open-source datasets demonstrate the effectiveness of FewFedPITin, enhancing privacy preservation and improving federated few-shot performance.

摘要: 指令调优已被认为是优化大语言模型(LLM)生成人类对齐响应的性能的关键技术。尽管如此，为这种调整收集多样化和高质量的教学数据存在明显的障碍，特别是在隐私条款严格的领域。联邦教学调整(FedIT)通过整合跨多个数据所有者的协作培训，从而产生保护隐私的学习模型，已成为一种有前途的解决方案。然而，FedIT遇到了诸如教学数据稀缺和暴露于训练数据提取攻击的风险等限制。在本文中，我们提出了一种新的联邦算法FewFedPIT，旨在同时增强隐私保护和联邦少镜头学习的模型性能。FewFedPIT在客户端包括三个重要组成部分：(1)合成数据生成，利用LLMS的上下文学习能力自主生成合成数据，从而扩展本地数据库；(2)参数隔离训练，分别更新合成数据中的公共参数和本地数据中的私有参数，从而减轻合成数据的噪声影响；(3)本地聚合共享，在上传之前混合公有和私有参数，有效防止数据提取攻击。在三个开源数据集上的大量实验证明了FewFedPITin的有效性，增强了隐私保护，提高了联邦少镜头性能。



## **41. Protecting Privacy Through Approximating Optimal Parameters for Sequence Unlearning in Language Models**

通过逼近语言模型中序列取消学习的最佳参数来保护隐私 cs.CL

Accepted to ACL2024 findings

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.14091v1) [paper-pdf](http://arxiv.org/pdf/2406.14091v1)

**Authors**: Dohyun Lee, Daniel Rim, Minseok Choi, Jaegul Choo

**Abstract**: Although language models (LMs) demonstrate exceptional capabilities on various tasks, they are potentially vulnerable to extraction attacks, which represent a significant privacy risk. To mitigate the privacy concerns of LMs, machine unlearning has emerged as an important research area, which is utilized to induce the LM to selectively forget about some of its training data. While completely retraining the model will guarantee successful unlearning and privacy assurance, it is impractical for LMs, as it would be time-consuming and resource-intensive. Prior works efficiently unlearn the target token sequences, but upon subsequent iterations, the LM displays significant degradation in performance. In this work, we propose Privacy Protection via Optimal Parameters (POP), a novel unlearning method that effectively forgets the target token sequences from the pretrained LM by applying optimal gradient updates to the parameters. Inspired by the gradient derivation of complete retraining, we approximate the optimal training objective that successfully unlearns the target sequence while retaining the knowledge from the rest of the training data. Experimental results demonstrate that POP exhibits remarkable retention performance post-unlearning across 9 classification and 4 dialogue benchmarks, outperforming the state-of-the-art by a large margin. Furthermore, we introduce Remnant Memorization Accuracy that quantifies privacy risks based on token likelihood and validate its effectiveness through both qualitative and quantitative analyses.

摘要: 尽管语言模型(LMS)在各种任务上表现出非凡的能力，但它们可能容易受到提取攻击，这代表着重大的隐私风险。为了缓解LMS的隐私问题，机器遗忘已经成为一个重要的研究领域，它被用来诱导LM选择性地忘记它的一些训练数据。虽然完全再培训该模型将确保成功忘记学习和隐私保证，但这对LMS来说是不切实际的，因为它将耗时和资源密集型。先前的工作有效地取消学习目标令牌序列，但在随后的迭代中，LM表现出显著的性能下降。在这项工作中，我们提出了通过最优参数的隐私保护(POP)，这是一种新的去学习方法，通过对参数应用最优梯度更新来有效地从预先训练的LM中忘记目标令牌序列。受完全再训练的梯度导数的启发，我们逼近了最优训练目标，在保留其余训练数据的知识的同时，成功地去除了目标序列。实验结果表明，在9个分类和4个对话基准中，POP在遗忘后表现出显著的保持性能，远远超过最新水平。此外，我们还引入了基于令牌似然量化隐私风险的剩余记忆准确率，并通过定性和定量分析验证了其有效性。



## **42. Prompt Injection Attacks in Defended Systems**

防御系统中的即时注入攻击 cs.CL

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.14048v1) [paper-pdf](http://arxiv.org/pdf/2406.14048v1)

**Authors**: Daniil Khomsky, Narek Maloyan, Bulat Nutfullin

**Abstract**: Large language models play a crucial role in modern natural language processing technologies. However, their extensive use also introduces potential security risks, such as the possibility of black-box attacks. These attacks can embed hidden malicious features into the model, leading to adverse consequences during its deployment.   This paper investigates methods for black-box attacks on large language models with a three-tiered defense mechanism. It analyzes the challenges and significance of these attacks, highlighting their potential implications for language processing system security. Existing attack and defense methods are examined, evaluating their effectiveness and applicability across various scenarios.   Special attention is given to the detection algorithm for black-box attacks, identifying hazardous vulnerabilities in language models and retrieving sensitive information. This research presents a methodology for vulnerability detection and the development of defensive strategies against black-box attacks on large language models.

摘要: 大型语言模型在现代自然语言处理技术中起着至关重要的作用。然而，它们的广泛使用也带来了潜在的安全风险，例如可能发生黑匣子攻击。这些攻击可能会将隐藏的恶意功能嵌入到模型中，导致部署过程中的不良后果。研究了三层防御机制对大型语言模型进行黑盒攻击的方法。它分析了这些攻击的挑战和意义，强调了它们对语言处理系统安全的潜在影响。检查了现有的攻击和防御方法，评估了它们在各种情况下的有效性和适用性。对黑盒攻击的检测算法、识别语言模型中的危险漏洞和检索敏感信息给予了特别关注。这项研究提出了一种针对大型语言模型的漏洞检测和防御策略的开发方法。



## **43. Evaluating Implicit Bias in Large Language Models by Attacking From a Psychometric Perspective**

从心理测量学角度通过攻击来评估大型语言模型中的内隐偏差 cs.CL

Code and datasets are available at  https://github.com/wen112358/ImplicitBiasPsychometricEvaluation

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.14023v1) [paper-pdf](http://arxiv.org/pdf/2406.14023v1)

**Authors**: Yuchen Wen, Keping Bi, Wei Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: As Large Language Models (LLMs) become an important way of information seeking, there have been increasing concerns about the unethical content LLMs may generate. In this paper, we conduct a rigorous evaluation of LLMs' implicit bias towards certain groups by attacking them with carefully crafted instructions to elicit biased responses. Our attack methodology is inspired by psychometric principles in cognitive and social psychology. We propose three attack approaches, i.e., Disguise, Deception, and Teaching, based on which we built evaluation datasets for four common bias types. Each prompt attack has bilingual versions. Extensive evaluation of representative LLMs shows that 1) all three attack methods work effectively, especially the Deception attacks; 2) GLM-3 performs the best in defending our attacks, compared to GPT-3.5 and GPT-4; 3) LLMs could output content of other bias types when being taught with one type of bias. Our methodology provides a rigorous and effective way of evaluating LLMs' implicit bias and will benefit the assessments of LLMs' potential ethical risks.

摘要: 随着大型语言模型成为人们寻找信息的一种重要方式，人们越来越关注大型语言模型可能产生的不道德内容。在这篇文章中，我们对LLMS对某些群体的内隐偏见进行了严格的评估，通过精心设计的指令来攻击他们，以获得有偏见的反应。我们的攻击方法受到认知和社会心理学中的心理测量学原理的启发。我们提出了三种攻击方法，即伪装、欺骗和教学，并在此基础上建立了四种常见偏差类型的评估数据集。每个即时攻击都有双语版本。我们的方法提供了一种严格而有效的方法来评估低收入者的隐性偏见，并将有助于评估低收入者的潜在道德风险。



## **44. Mitigating Fine-tuning based Jailbreak Attack with Backdoor Enhanced Safety Alignment**

通过后门增强的安全调整缓解基于微调的越狱攻击 cs.CR

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2402.14968v3) [paper-pdf](http://arxiv.org/pdf/2402.14968v3)

**Authors**: Jiongxiao Wang, Jiazhao Li, Yiquan Li, Xiangyu Qi, Junjie Hu, Yixuan Li, Patrick McDaniel, Muhao Chen, Bo Li, Chaowei Xiao

**Abstract**: Despite the general capabilities of Large Language Models (LLM), these models still request fine-tuning or adaptation with customized data when meeting specific business demands. However, this process inevitably introduces new threats, particularly against the Fine-tuning based Jailbreak Attack (FJAttack) under the setting of Language-Model-as-a-Service (LMaaS), where the model's safety has been significantly compromised by fine-tuning users' uploaded examples contain just a few harmful examples. Though potential defenses have been proposed that the service providers can integrate safety examples into the fine-tuning dataset to reduce safety issues, such approaches require incorporating a substantial amount of data, making it inefficient. To effectively defend against the FJAttack with limited safety examples under LMaaS, we propose the Backdoor Enhanced Safety Alignment method inspired by an analogy with the concept of backdoor attacks. In particular, service providers will construct prefixed safety examples with a secret prompt, acting as a "backdoor trigger". By integrating prefixed safety examples into the fine-tuning dataset, the subsequent fine-tuning process effectively acts as the "backdoor attack", establishing a strong correlation between the secret prompt and safety generations. Consequently, safe responses are ensured once service providers prepend this secret prompt ahead of any user input during inference. Our comprehensive experiments demonstrate that through the Backdoor Enhanced Safety Alignment with adding as few as 11 prefixed safety examples, the maliciously fine-tuned LLMs will achieve similar safety performance as the original aligned models without harming the benign performance. Furthermore, we also present the effectiveness of our method in a more practical setting where the fine-tuning data consists of both FJAttack examples and the fine-tuning task data.

摘要: 尽管大型语言模型(LLM)具有一般功能，但在满足特定业务需求时，这些模型仍然需要使用定制数据进行微调或调整。然而，这一过程不可避免地带来了新的威胁，特别是针对LMaaS(Language-Model-as-a-Service，语言模型即服务)设置下的基于Fine-Tuning的越狱攻击(FJAttack)，其中模型的安全性因微调用户上传的示例仅包含几个有害示例而受到严重威胁。尽管有人提出了潜在的防御措施，即服务提供商可以将安全实例整合到微调数据集中，以减少安全问题，但这种方法需要纳入大量数据，使其效率低下。为了在LMaaS环境下有效防御安全实例有限的FJAttack，我们借鉴了后门攻击的概念，提出了后门增强安全对齐方法。特别是，服务提供商将构建带有前缀的安全示例，并使用秘密提示，充当“后门触发器”。通过将前缀的安全实例整合到微调数据集中，后续的微调过程有效地充当了“后门攻击”，在秘密提示和安全生成之间建立了很强的关联。因此，一旦服务提供商在推理过程中在任何用户输入之前预先考虑此秘密提示，就可以确保安全响应。我们的综合实验表明，通过添加仅需11个前缀安全实例的后门增强安全对准，恶意微调的LLM将在不损害良性性能的情况下获得与原始对准模型相似的安全性能。此外，我们还在一个更实际的环境中展示了我们的方法的有效性，其中微调数据包括FJAttack实例和微调任务数据。



## **45. RLHFPoison: Reward Poisoning Attack for Reinforcement Learning with Human Feedback in Large Language Models**

RL HFPoison：大型语言模型中具有人类反馈的强化学习的奖励中毒攻击 cs.AI

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2311.09641v2) [paper-pdf](http://arxiv.org/pdf/2311.09641v2)

**Authors**: Jiongxiao Wang, Junlin Wu, Muhao Chen, Yevgeniy Vorobeychik, Chaowei Xiao

**Abstract**: Reinforcement Learning with Human Feedback (RLHF) is a methodology designed to align Large Language Models (LLMs) with human preferences, playing an important role in LLMs alignment. Despite its advantages, RLHF relies on human annotators to rank the text, which can introduce potential security vulnerabilities if any adversarial annotator (i.e., attackers) manipulates the ranking score by up-ranking any malicious text to steer the LLM adversarially. To assess the red-teaming of RLHF against human preference data poisoning, we propose RankPoison, a poisoning attack method on candidates' selection of preference rank flipping to reach certain malicious behaviors (e.g., generating longer sequences, which can increase the computational cost). With poisoned dataset generated by RankPoison, we can perform poisoning attacks on LLMs to generate longer tokens without hurting the original safety alignment performance. Moreover, applying RankPoison, we also successfully implement a backdoor attack where LLMs can generate longer answers under questions with the trigger word. Our findings highlight critical security challenges in RLHF, underscoring the necessity for more robust alignment methods for LLMs.

摘要: 带人反馈的强化学习(RLHF)是一种将大语言模型与人的偏好相匹配的方法，在大语言模型对齐中起着重要作用。尽管RLHF有其优势，但它依靠人工注释者对文本进行排名，如果任何敌意注释者(即攻击者)通过对任何恶意文本进行排名来操纵排名分数，从而对LLM进行敌意操作，这可能会引入潜在的安全漏洞。为了评估RLHF的红团队对抗人类偏好数据中毒的能力，我们提出了一种毒化攻击方法RankPoison，该方法针对候选者选择偏好翻转来达到某些恶意行为(例如，生成更长的序列，这会增加计算成本)。利用RankPoison生成的有毒数据集，我们可以在不损害原始安全对齐性能的情况下，对LLM进行中毒攻击，生成更长的令牌。此外，应用RankPoison，我们还成功地实现了一个后门攻击，在带有触发词的问题下，LLMS可以生成更长的答案。我们的发现突出了RLHF中的关键安全挑战，强调了对LLM采用更强大的比对方法的必要性。



## **46. Is poisoning a real threat to LLM alignment? Maybe more so than you think**

中毒是对LLM联盟的真正威胁吗？也许比你想象的还要多 cs.LG

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2406.12091v2) [paper-pdf](http://arxiv.org/pdf/2406.12091v2)

**Authors**: Pankayaraj Pathmanathan, Souradip Chakraborty, Xiangyu Liu, Yongyuan Liang, Furong Huang

**Abstract**: Recent advancements in Reinforcement Learning with Human Feedback (RLHF) have significantly impacted the alignment of Large Language Models (LLMs). The sensitivity of reinforcement learning algorithms such as Proximal Policy Optimization (PPO) has led to new line work on Direct Policy Optimization (DPO), which treats RLHF in a supervised learning framework. The increased practical use of these RLHF methods warrants an analysis of their vulnerabilities. In this work, we investigate the vulnerabilities of DPO to poisoning attacks under different scenarios and compare the effectiveness of preference poisoning, a first of its kind. We comprehensively analyze DPO's vulnerabilities under different types of attacks, i.e., backdoor and non-backdoor attacks, and different poisoning methods across a wide array of language models, i.e., LLama 7B, Mistral 7B, and Gemma 7B. We find that unlike PPO-based methods, which, when it comes to backdoor attacks, require at least 4\% of the data to be poisoned to elicit harmful behavior, we exploit the true vulnerabilities of DPO more simply so we can poison the model with only as much as 0.5\% of the data. We further investigate the potential reasons behind the vulnerability and how well this vulnerability translates into backdoor vs non-backdoor attacks.

摘要: 人类反馈强化学习(RLHF)的最新进展对大型语言模型(LLM)的匹配产生了重大影响。强化学习算法的敏感性，如最近策略优化(PPO)，导致了直接策略优化(DPO)的新工作，它在监督学习框架中处理RLHF。这些RLHF方法的实际使用越来越多，因此有理由对其脆弱性进行分析。在这项工作中，我们调查了DPO在不同场景下对中毒攻击的脆弱性，并比较了偏好中毒的有效性，这是第一次。我们全面分析了DPO在不同类型的攻击下的漏洞，即后门攻击和非后门攻击，以及不同的中毒方法，跨越了广泛的语言模型，即：大羊驼7B、米斯特拉尔7B和杰玛7B。我们发现，与基于PPO的方法不同，当涉及到后门攻击时，需要至少4%的数据被毒化才能引发有害行为，而我们更简单地利用DPO的真正漏洞，因此我们只需使用多达0.5%的数据就可以毒害模型。我们进一步调查了该漏洞背后的潜在原因，以及该漏洞在多大程度上转化为后门攻击与非后门攻击。



## **47. ObscurePrompt: Jailbreaking Large Language Models via Obscure Input**

晦涩提示：通过晦涩输入破解大型语言模型 cs.CL

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2406.13662v1) [paper-pdf](http://arxiv.org/pdf/2406.13662v1)

**Authors**: Yue Huang, Jingyu Tang, Dongping Chen, Bingda Tang, Yao Wan, Lichao Sun, Xiangliang Zhang

**Abstract**: Recently, Large Language Models (LLMs) have garnered significant attention for their exceptional natural language processing capabilities. However, concerns about their trustworthiness remain unresolved, particularly in addressing "jailbreaking" attacks on aligned LLMs. Previous research predominantly relies on scenarios with white-box LLMs or specific and fixed prompt templates, which are often impractical and lack broad applicability. In this paper, we introduce a straightforward and novel method, named ObscurePrompt, for jailbreaking LLMs, inspired by the observed fragile alignments in Out-of-Distribution (OOD) data. Specifically, we first formulate the decision boundary in the jailbreaking process and then explore how obscure text affects LLM's ethical decision boundary. ObscurePrompt starts with constructing a base prompt that integrates well-known jailbreaking techniques. Powerful LLMs are then utilized to obscure the original prompt through iterative transformations, aiming to bolster the attack's robustness. Comprehensive experiments show that our approach substantially improves upon previous methods in terms of attack effectiveness, maintaining efficacy against two prevalent defense mechanisms. We believe that our work can offer fresh insights for future research on enhancing LLM alignment.

摘要: 近年来，大型语言模型(LLM)以其卓越的自然语言处理能力引起了人们的极大关注。然而，对它们可信度的担忧仍然没有得到解决，特别是在解决对结盟的LLM的“越狱”攻击方面。以前的研究主要依赖于白盒LLM或特定和固定提示模板的场景，这些场景往往不切实际，缺乏广泛的适用性。在这篇文章中，我们介绍了一个简单而新颖的方法，称为ObscurePrompt，用于越狱LLMS，灵感来自于观察到的分布外(OOD)数据中的脆弱对齐。具体地说，我们首先阐述了越狱过程中的决策边界，然后探讨了晦涩的文本如何影响LLM的伦理决策边界。ObscurePrompt首先构建一个集成了众所周知的越狱技术的基本提示。然后利用强大的LLM通过迭代变换来模糊原始提示，旨在增强攻击的健壮性。综合实验表明，我们的方法在攻击有效性方面比以前的方法有了很大的提高，保持了对两种流行的防御机制的有效性。我们相信，我们的工作可以为未来增强LLM对齐的研究提供新的见解。



## **48. Lockpicking LLMs: A Logit-Based Jailbreak Using Token-level Manipulation**

撬锁LLM：使用代币级操纵的基于日志的越狱 cs.CR

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2405.13068v2) [paper-pdf](http://arxiv.org/pdf/2405.13068v2)

**Authors**: Yuxi Li, Yi Liu, Yuekang Li, Ling Shi, Gelei Deng, Shengquan Chen, Kailong Wang

**Abstract**: Large language models (LLMs) have transformed the field of natural language processing, but they remain susceptible to jailbreaking attacks that exploit their capabilities to generate unintended and potentially harmful content. Existing token-level jailbreaking techniques, while effective, face scalability and efficiency challenges, especially as models undergo frequent updates and incorporate advanced defensive measures. In this paper, we introduce JailMine, an innovative token-level manipulation approach that addresses these limitations effectively. JailMine employs an automated "mining" process to elicit malicious responses from LLMs by strategically selecting affirmative outputs and iteratively reducing the likelihood of rejection. Through rigorous testing across multiple well-known LLMs and datasets, we demonstrate JailMine's effectiveness and efficiency, achieving a significant average reduction of 86% in time consumed while maintaining high success rates averaging 95%, even in the face of evolving defensive strategies. Our work contributes to the ongoing effort to assess and mitigate the vulnerability of LLMs to jailbreaking attacks, underscoring the importance of continued vigilance and proactive measures to enhance the security and reliability of these powerful language models.

摘要: 大型语言模型(LLM)已经改变了自然语言处理领域，但它们仍然容易受到越狱攻击，这些攻击利用它们的能力生成意外的和潜在的有害内容。现有的令牌级越狱技术虽然有效，但面临可伸缩性和效率的挑战，特别是在模型经历频繁更新和采用先进防御措施的情况下。在本文中，我们介绍了Jailmy，一种创新的令牌级操作方法，有效地解决了这些限制。Jailmine使用一个自动化的“挖掘”过程，通过战略性地选择肯定的输出并反复降低拒绝的可能性，来引发来自LLMS的恶意响应。我们的工作有助于评估和减轻LLMS在越狱攻击中的脆弱性，强调了继续保持警惕和采取积极措施以增强这些强大语言模型的安全性和可靠性的重要性。



## **49. Textual Unlearning Gives a False Sense of Unlearning**

文本遗忘给人一种遗忘的错误感觉 cs.CR

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2406.13348v1) [paper-pdf](http://arxiv.org/pdf/2406.13348v1)

**Authors**: Jiacheng Du, Zhibo Wang, Kui Ren

**Abstract**: Language models (LMs) are susceptible to "memorizing" training data, including a large amount of private or copyright-protected content. To safeguard the right to be forgotten (RTBF), machine unlearning has emerged as a promising method for LMs to efficiently "forget" sensitive training content and mitigate knowledge leakage risks. However, despite its good intentions, could the unlearning mechanism be counterproductive? In this paper, we propose the Textual Unlearning Leakage Attack (TULA), where an adversary can infer information about the unlearned data only by accessing the models before and after unlearning. Furthermore, we present variants of TULA in both black-box and white-box scenarios. Through various experimental results, we critically demonstrate that machine unlearning amplifies the risk of knowledge leakage from LMs. Specifically, TULA can increase an adversary's ability to infer membership information about the unlearned data by more than 20% in black-box scenario. Moreover, TULA can even reconstruct the unlearned data directly with more than 60% accuracy with white-box access. Our work is the first to reveal that machine unlearning in LMs can inversely create greater knowledge risks and inspire the development of more secure unlearning mechanisms.

摘要: 语言模型(LMS)很容易“记忆”训练数据，包括大量私人或受版权保护的内容。为了保护被遗忘的权利，机器遗忘已经成为学习管理系统有效忘记敏感训练内容和降低知识泄漏风险的一种很有前途的方法。然而，尽管这种遗忘机制的用意是好的，但它会适得其反吗？在本文中，我们提出了文本遗忘泄漏攻击(Tula)，在该攻击中，攻击者只能通过访问遗忘前后的模型来推断关于未学习数据的信息。此外，我们还介绍了Tula在黑盒和白盒场景中的变体。通过各种实验结果，我们批判性地证明了机器遗忘放大了最小二乘系统的知识泄漏风险。具体地说，在黑盒情况下，Tula可以将对手推断未学习数据的成员信息的能力提高20%以上。此外，图拉甚至可以通过白盒访问直接重建未学习的数据，准确率超过60%。我们的工作首次揭示了LMS中的机器遗忘可以相反地创造更大的知识风险，并激励更安全的遗忘机制的发展。



## **50. MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models**

MM-SafetyBench：多模式大型语言模型安全评估的基准 cs.CV

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2311.17600v5) [paper-pdf](http://arxiv.org/pdf/2311.17600v5)

**Authors**: Xin Liu, Yichen Zhu, Jindong Gu, Yunshi Lan, Chao Yang, Yu Qiao

**Abstract**: The security concerns surrounding Large Language Models (LLMs) have been extensively explored, yet the safety of Multimodal Large Language Models (MLLMs) remains understudied. In this paper, we observe that Multimodal Large Language Models (MLLMs) can be easily compromised by query-relevant images, as if the text query itself were malicious. To address this, we introduce MM-SafetyBench, a comprehensive framework designed for conducting safety-critical evaluations of MLLMs against such image-based manipulations. We have compiled a dataset comprising 13 scenarios, resulting in a total of 5,040 text-image pairs. Our analysis across 12 state-of-the-art models reveals that MLLMs are susceptible to breaches instigated by our approach, even when the equipped LLMs have been safety-aligned. In response, we propose a straightforward yet effective prompting strategy to enhance the resilience of MLLMs against these types of attacks. Our work underscores the need for a concerted effort to strengthen and enhance the safety measures of open-source MLLMs against potential malicious exploits. The resource is available at https://github.com/isXinLiu/MM-SafetyBench

摘要: 围绕大语言模型的安全问题已经得到了广泛的研究，但多模式大语言模型的安全性仍未得到充分的研究。在本文中，我们观察到多模式大型语言模型(MLLMS)很容易被与查询相关的图像破坏，就好像文本查询本身是恶意的一样。为了解决这一问题，我们引入了MM-SafetyBch，这是一个全面的框架，旨在针对此类基于图像的操作对MLLMS进行安全关键评估。我们汇编了一个包含13个场景的数据集，总共产生了5,040个文本-图像对。我们对12种最先进型号的分析表明，即使配备的LLM已经安全对准，MLLM也容易受到我们的方法引发的漏洞的影响。对此，我们提出了一种简单而有效的提示策略，以增强MLLMS对这些类型攻击的弹性。我们的工作强调了需要齐心协力加强和改进开放源码MLLM的安全措施，以防范潜在的恶意利用。该资源可在https://github.com/isXinLiu/MM-SafetyBench上获得



