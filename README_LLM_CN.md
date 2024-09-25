# Latest Large Language Model Attack Papers
**update at 2024-09-25 10:22:49**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Order of Magnitude Speedups for LLM Membership Inference**

LLM会员推断的量级加速 cs.LG

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2409.14513v2) [paper-pdf](http://arxiv.org/pdf/2409.14513v2)

**Authors**: Rongting Zhang, Martin Bertran, Aaron Roth

**Abstract**: Large Language Models (LLMs) have the promise to revolutionize computing broadly, but their complexity and extensive training data also expose significant privacy vulnerabilities. One of the simplest privacy risks associated with LLMs is their susceptibility to membership inference attacks (MIAs), wherein an adversary aims to determine whether a specific data point was part of the model's training set. Although this is a known risk, state of the art methodologies for MIAs rely on training multiple computationally costly shadow models, making risk evaluation prohibitive for large models. Here we adapt a recent line of work which uses quantile regression to mount membership inference attacks; we extend this work by proposing a low-cost MIA that leverages an ensemble of small quantile regression models to determine if a document belongs to the model's training set or not. We demonstrate the effectiveness of this approach on fine-tuned LLMs of varying families (OPT, Pythia, Llama) and across multiple datasets. Across all scenarios we obtain comparable or improved accuracy compared to state of the art shadow model approaches, with as little as 6% of their computation budget. We demonstrate increased effectiveness across multi-epoch trained target models, and architecture miss-specification robustness, that is, we can mount an effective attack against a model using a different tokenizer and architecture, without requiring knowledge on the target model.

摘要: 大型语言模型(LLM)有望给计算带来革命性的变化，但它们的复杂性和大量的训练数据也暴露了严重的隐私漏洞。与LLMS相关的最简单的隐私风险之一是它们对成员关系推理攻击(MIA)的敏感性，即对手的目标是确定特定数据点是否属于模型训练集的一部分。尽管这是一个已知的风险，但MIA的最新方法依赖于训练多个计算成本高昂的阴影模型，这使得风险评估对于大型模型来说是令人望而却步的。在这里，我们采用了最近的一项工作，它使用分位数回归来发动成员关系推理攻击；我们通过提出一种低成本的MIA来扩展这一工作，该MIA利用一组小分位数回归模型来确定文档是否属于模型的训练集。我们在不同家族(OPT，Pythia，Llama)的微调LLM上展示了这种方法的有效性，并跨越了多个数据集。在所有场景中，与最先进的阴影模型方法相比，我们获得了相当或更高的精度，只需其计算预算的6%。我们证明了多时代训练的目标模型的有效性和体系结构未指定的稳健性，即，我们可以使用不同的标记器和体系结构对模型发动有效攻击，而不需要目标模型的知识。



## **2. Cyber Knowledge Completion Using Large Language Models**

使用大型语言模型完成网络知识 cs.CR

7 pages, 2 figures. Submitted to 2024 IEEE International Conference  on Big Data

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2409.16176v1) [paper-pdf](http://arxiv.org/pdf/2409.16176v1)

**Authors**: Braden K Webb, Sumit Purohit, Rounak Meyur

**Abstract**: The integration of the Internet of Things (IoT) into Cyber-Physical Systems (CPSs) has expanded their cyber-attack surface, introducing new and sophisticated threats with potential to exploit emerging vulnerabilities. Assessing the risks of CPSs is increasingly difficult due to incomplete and outdated cybersecurity knowledge. This highlights the urgent need for better-informed risk assessments and mitigation strategies. While previous efforts have relied on rule-based natural language processing (NLP) tools to map vulnerabilities, weaknesses, and attack patterns, recent advancements in Large Language Models (LLMs) present a unique opportunity to enhance cyber-attack knowledge completion through improved reasoning, inference, and summarization capabilities. We apply embedding models to encapsulate information on attack patterns and adversarial techniques, generating mappings between them using vector embeddings. Additionally, we propose a Retrieval-Augmented Generation (RAG)-based approach that leverages pre-trained models to create structured mappings between different taxonomies of threat patterns. Further, we use a small hand-labeled dataset to compare the proposed RAG-based approach to a baseline standard binary classification model. Thus, the proposed approach provides a comprehensive framework to address the challenge of cyber-attack knowledge graph completion.

摘要: 物联网(IoT)与网络物理系统(CPSS)的集成扩大了它们的网络攻击面，带来了新的复杂威胁，有可能利用新出现的漏洞。由于网络安全知识的不完整和过时，评估CPSS的风险变得越来越困难。这突出表明迫切需要更了解情况的风险评估和缓解战略。虽然以前的努力依赖于基于规则的自然语言处理(NLP)工具来绘制漏洞、弱点和攻击模式，但最近大型语言模型(LLM)的进步提供了一个独特的机会，可以通过改进的推理、推理和总结能力来增强网络攻击知识的完备性。我们使用嵌入模型来封装关于攻击模式和对抗技术的信息，并使用向量嵌入来生成它们之间的映射。此外，我们提出了一种基于检索增强生成(RAG)的方法，该方法利用预先训练的模型来创建威胁模式的不同分类之间的结构化映射。此外，我们使用一个小的手工标记的数据集来比较所提出的基于RAG的方法与基准标准的二进制分类模型。因此，提出的方法提供了一个全面的框架来解决网络攻击知识图完成的挑战。



## **3. Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack**

太好了，现在写一篇关于这一点的文章：渐强多转法学硕士越狱攻击 cs.CR

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2404.01833v2) [paper-pdf](http://arxiv.org/pdf/2404.01833v2)

**Authors**: Mark Russinovich, Ahmed Salem, Ronen Eldan

**Abstract**: Large Language Models (LLMs) have risen significantly in popularity and are increasingly being adopted across multiple applications. These LLMs are heavily aligned to resist engaging in illegal or unethical topics as a means to avoid contributing to responsible AI harms. However, a recent line of attacks, known as jailbreaks, seek to overcome this alignment. Intuitively, jailbreak attacks aim to narrow the gap between what the model can do and what it is willing to do. In this paper, we introduce a novel jailbreak attack called Crescendo. Unlike existing jailbreak methods, Crescendo is a simple multi-turn jailbreak that interacts with the model in a seemingly benign manner. It begins with a general prompt or question about the task at hand and then gradually escalates the dialogue by referencing the model's replies progressively leading to a successful jailbreak. We evaluate Crescendo on various public systems, including ChatGPT, Gemini Pro, Gemini-Ultra, LlaMA-2 70b and LlaMA-3 70b Chat, and Anthropic Chat. Our results demonstrate the strong efficacy of Crescendo, with it achieving high attack success rates across all evaluated models and tasks. Furthermore, we present Crescendomation, a tool that automates the Crescendo attack and demonstrate its efficacy against state-of-the-art models through our evaluations. Crescendomation surpasses other state-of-the-art jailbreaking techniques on the AdvBench subset dataset, achieving 29-61% higher performance on GPT-4 and 49-71% on Gemini-Pro. Finally, we also demonstrate Crescendo's ability to jailbreak multimodal models.

摘要: 大型语言模型(LLM)的受欢迎程度显著提高，并越来越多地被多个应用程序采用。这些低成本管理机构强烈反对从事非法或不道德的话题，以此作为避免造成负责任的人工智能损害的一种手段。然而，最近一系列被称为越狱的袭击试图克服这种联系。直观地说，越狱攻击的目的是缩小模型可以做的事情和它愿意做的事情之间的差距。在这篇文章中，我们介绍了一种新的越狱攻击，称为Crescendo。与现有的越狱方法不同，Cresendo是一种简单的多转弯越狱方法，它以一种看似良性的方式与模型交互。它以关于手头任务的一般提示或问题开始，然后通过逐步参考模型的答复逐步升级对话，从而导致成功越狱。我们在不同的公共系统上对Cresendo进行了评估，包括ChatGPT、Gemini Pro、Gemini-Ultra、Llama-2 70b和Llama-3 70b Chat，以及Anthropic Chat。我们的结果证明了Crescendo的强大效力，它在所有评估的模型和任务中都实现了高攻击成功率。此外，我们提出了Crescendomation，这是一个自动化的Crescendo攻击工具，并通过我们的评估展示了它对最先进的模型的有效性。Cresendomation在AdvBtch子集数据集上超过了其他最先进的越狱技术，在GPT-4和Gemini-Pro上的性能分别提高了29%-61%和49%-71%。最后，我们还展示了Cresendo越狱多模式模型的能力。



## **4. Large Language Models as Carriers of Hidden Messages**

大型语言模型作为隐藏消息的载体 cs.CL

Work in progress. Code is available at  https://github.com/j-hoscilowic/zurek-stegano

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2406.02481v4) [paper-pdf](http://arxiv.org/pdf/2406.02481v4)

**Authors**: Jakub Hoscilowicz, Pawel Popiolek, Jan Rudkowski, Jedrzej Bieniasz, Artur Janicki

**Abstract**: Simple fine-tuning can embed hidden text into large language models (LLMs), which is revealed only when triggered by a specific query. Applications include LLM fingerprinting, where a unique identifier is embedded to verify licensing compliance, and steganography, where the LLM carries hidden messages disclosed through a trigger query.   Our work demonstrates that embedding hidden text via fine-tuning, although seemingly secure due to the vast number of potential triggers, is vulnerable to extraction through analysis of the LLM's output decoding process. We introduce an extraction attack called Unconditional Token Forcing (UTF), which iteratively feeds tokens from the LLM's vocabulary to reveal sequences with high token probabilities, indicating hidden text candidates. We also present Unconditional Token Forcing Confusion (UTFC), a defense paradigm that makes hidden text resistant to all known extraction attacks without degrading the general performance of LLMs compared to standard fine-tuning. UTFC has both benign (improving LLM fingerprinting) and malign applications (using LLMs to create covert communication channels).

摘要: 简单的微调可以将隐藏文本嵌入到大型语言模型(LLM)中，只有在特定查询触发时才会显示出来。应用包括LLM指纹识别，其中嵌入唯一标识符以验证许可合规性，以及隐写术，其中LLM携带通过触发查询披露的隐藏消息。我们的工作表明，通过微调嵌入隐藏文本，尽管由于大量潜在触发而看起来是安全的，但通过分析LLM的输出解码过程，很容易被提取出来。我们引入了一种称为无条件令牌强制(UTF)的提取攻击，该攻击迭代地从LLM的词汇表中馈送令牌来揭示具有高令牌概率的序列，从而指示隐藏的文本候选。我们还提出了无条件令牌强制混淆(UTFC)，这是一种防御范式，使隐藏文本能够抵抗所有已知的提取攻击，而不会降低与标准微调相比的LLMS的总体性能。UTFC既有良性的(改进的LLM指纹识别)，也有恶意的应用(使用LLM创建隐蔽的通信通道)。



## **5. Privacy Evaluation Benchmarks for NLP Models**

NLP模型的隐私评估基准 cs.CL

Accepted by the Findings of EMNLP 2024

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2409.15868v1) [paper-pdf](http://arxiv.org/pdf/2409.15868v1)

**Authors**: Wei Huang, Yinggui Wang, Cen Chen

**Abstract**: By inducing privacy attacks on NLP models, attackers can obtain sensitive information such as training data and model parameters, etc. Although researchers have studied, in-depth, several kinds of attacks in NLP models, they are non-systematic analyses. It lacks a comprehensive understanding of the impact caused by the attacks. For example, we must consider which scenarios can apply to which attacks, what the common factors are that affect the performance of different attacks, the nature of the relationships between different attacks, and the influence of various datasets and models on the effectiveness of the attacks, etc. Therefore, we need a benchmark to holistically assess the privacy risks faced by NLP models. In this paper, we present a privacy attack and defense evaluation benchmark in the field of NLP, which includes the conventional/small models and large language models (LLMs). This benchmark supports a variety of models, datasets, and protocols, along with standardized modules for comprehensive evaluation of attacks and defense strategies. Based on the above framework, we present a study on the association between auxiliary data from different domains and the strength of privacy attacks. And we provide an improved attack method in this scenario with the help of Knowledge Distillation (KD). Furthermore, we propose a chained framework for privacy attacks. Allowing a practitioner to chain multiple attacks to achieve a higher-level attack objective. Based on this, we provide some defense and enhanced attack strategies. The code for reproducing the results can be found at https://github.com/user2311717757/nlp_doctor.

摘要: 通过诱导对NLP模型的隐私攻击，攻击者可以获得训练数据和模型参数等敏感信息。尽管研究人员对NLP模型中的几种攻击进行了深入的研究，但它们都是非系统的分析。它缺乏对袭击造成的影响的全面了解。例如，我们必须考虑哪些场景可以应用于哪些攻击，影响不同攻击性能的共同因素是什么，不同攻击之间关系的性质，以及各种数据集和模型对攻击有效性的影响等。因此，我们需要一个基准来全面评估NLP模型所面临的隐私风险。本文提出了一种针对自然语言处理领域的隐私攻防评估基准，包括常规/小模型和大语言模型。该基准支持各种模型、数据集和协议，以及用于全面评估攻击和防御策略的标准化模块。基于上述框架，我们提出了不同领域辅助数据之间的关联与隐私攻击强度的研究。在这种情况下，我们提出了一种改进的攻击方法--知识蒸馏(KD)。此外，我们还提出了一个针对隐私攻击的链式框架。允许实践者链接多次攻击以实现更高级别的攻击目标。在此基础上，提出了一些防御和增强攻击的策略。复制结果的代码可在https://github.com/user2311717757/nlp_doctor.上找到



## **6. Goal-guided Generative Prompt Injection Attack on Large Language Models**

对大型语言模型的目标引导生成提示注入攻击 cs.CR

11 pages, 6 figures

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2404.07234v3) [paper-pdf](http://arxiv.org/pdf/2404.07234v3)

**Authors**: Chong Zhang, Mingyu Jin, Qinkai Yu, Chengzhi Liu, Haochen Xue, Xiaobo Jin

**Abstract**: Current large language models (LLMs) provide a strong foundation for large-scale user-oriented natural language tasks. A large number of users can easily inject adversarial text or instructions through the user interface, thus causing LLMs model security challenges. Although there is currently a large amount of research on prompt injection attacks, most of these black-box attacks use heuristic strategies. It is unclear how these heuristic strategies relate to the success rate of attacks and thus effectively improve model robustness. To solve this problem, we redefine the goal of the attack: to maximize the KL divergence between the conditional probabilities of the clean text and the adversarial text. Furthermore, we prove that maximizing the KL divergence is equivalent to maximizing the Mahalanobis distance between the embedded representation $x$ and $x'$ of the clean text and the adversarial text when the conditional probability is a Gaussian distribution and gives a quantitative relationship on $x$ and $x'$. Then we designed a simple and effective goal-guided generative prompt injection strategy (G2PIA) to find an injection text that satisfies specific constraints to achieve the optimal attack effect approximately. It is particularly noteworthy that our attack method is a query-free black-box attack method with low computational cost. Experimental results on seven LLM models and four datasets show the effectiveness of our attack method.

摘要: 现有的大型语言模型为大规模面向用户的自然语言任务提供了坚实的基础。大量用户可以很容易地通过用户界面注入敌意文本或指令，从而造成LLMS模型的安全挑战。虽然目前有大量关于即时注入攻击的研究，但这些黑盒攻击大多采用启发式策略。目前尚不清楚这些启发式策略如何与攻击成功率相关，从而有效地提高模型的稳健性。为了解决这个问题，我们重新定义了攻击的目标：最大化纯文本和敌意文本的条件概率之间的KL偏差。此外，我们证明了当条件概率为高斯分布时，最大化KL发散度等价于最大化明文和敌意文本的嵌入表示$x$和$x‘$之间的马氏距离，并给出了关于$x$和$x’$的定量关系。然后，设计了一种简单有效的目标引导生成性提示注入策略(G2PIA)，找到满足特定约束的注入文本，以近似达到最优的攻击效果。特别值得注意的是，我们的攻击方法是一种计算代价低的无查询黑盒攻击方法。在7个LLM模型和4个数据集上的实验结果表明了该攻击方法的有效性。



## **7. Large Language Models Are Involuntary Truth-Tellers: Exploiting Fallacy Failure for Jailbreak Attacks**

大型语言模型是不自愿的真话者：利用谬误失败进行越狱攻击 cs.CL

Accepted to the main conference of EMNLP 2024

**SubmitDate**: 2024-09-23    [abs](http://arxiv.org/abs/2407.00869v2) [paper-pdf](http://arxiv.org/pdf/2407.00869v2)

**Authors**: Yue Zhou, Henry Peng Zou, Barbara Di Eugenio, Yang Zhang

**Abstract**: We find that language models have difficulties generating fallacious and deceptive reasoning. When asked to generate deceptive outputs, language models tend to leak honest counterparts but believe them to be false. Exploiting this deficiency, we propose a jailbreak attack method that elicits an aligned language model for malicious output. Specifically, we query the model to generate a fallacious yet deceptively real procedure for the harmful behavior. Since a fallacious procedure is generally considered fake and thus harmless by LLMs, it helps bypass the safeguard mechanism. Yet the output is factually harmful since the LLM cannot fabricate fallacious solutions but proposes truthful ones. We evaluate our approach over five safety-aligned large language models, comparing four previous jailbreak methods, and show that our approach achieves competitive performance with more harmful outputs. We believe the findings could be extended beyond model safety, such as self-verification and hallucination.

摘要: 我们发现，语言模型很难生成错误和欺骗性的推理。当被要求生成欺骗性输出时，语言模型往往会泄露诚实的对应结果，但认为它们是错误的。利用这一缺陷，我们提出了一种越狱攻击方法，该方法可以得到恶意输出的对齐语言模型。具体地说，我们对该模型提出质疑，以便为有害行为生成一个虚假但虚假的真实过程。由于错误的程序通常被LLMS认为是虚假的，因此是无害的，它有助于绕过保障机制。然而，这些结果实际上是有害的，因为LLM不能捏造虚假的解决方案，而是提出真实的解决方案。我们在五个安全对齐的大型语言模型上对我们的方法进行了评估，并与之前的四种越狱方法进行了比较，结果表明我们的方法在具有更多有害输出的情况下获得了具有竞争力的性能。我们认为，这些发现可以扩展到模型安全之外，例如自我验证和幻觉。



## **8. Uncovering Coordinated Cross-Platform Information Operations Threatening the Integrity of the 2024 U.S. Presidential Election Online Discussion**

揭露威胁2024年美国总统选举在线讨论完整性的协调跨平台信息操作 cs.SI

The 2024 Election Integrity Initiative: HUMANS Lab - Working Paper  No. 2024.4 - University of Southern California

**SubmitDate**: 2024-09-23    [abs](http://arxiv.org/abs/2409.15402v1) [paper-pdf](http://arxiv.org/pdf/2409.15402v1)

**Authors**: Marco Minici, Luca Luceri, Federico Cinus, Emilio Ferrara

**Abstract**: Information Operations (IOs) pose a significant threat to the integrity of democratic processes, with the potential to influence election-related online discourse. In anticipation of the 2024 U.S. presidential election, we present a study aimed at uncovering the digital traces of coordinated IOs on $\mathbb{X}$ (formerly Twitter). Using our machine learning framework for detecting online coordination, we analyze a dataset comprising election-related conversations on $\mathbb{X}$ from May 2024. This reveals a network of coordinated inauthentic actors, displaying notable similarities in their link-sharing behaviors. Our analysis shows concerted efforts by these accounts to disseminate misleading, redundant, and biased information across the Web through a coordinated cross-platform information operation: The links shared by this network frequently direct users to other social media platforms or suspicious websites featuring low-quality political content and, in turn, promoting the same $\mathbb{X}$ and YouTube accounts. Members of this network also shared deceptive images generated by AI, accompanied by language attacking political figures and symbolic imagery intended to convey power and dominance. While $\mathbb{X}$ has suspended a subset of these accounts, more than 75% of the coordinated network remains active. Our findings underscore the critical role of developing computational models to scale up the detection of threats on large social media platforms, and emphasize the broader implications of these techniques to detect IOs across the wider Web.

摘要: 信息业务对民主进程的完整性构成重大威胁，有可能影响与选举有关的网上言论。在对2024年美国总统大选的预期中，我们提出了一项研究，旨在揭示$\mathbb{X}$(前身为Twitter)上协调iOS的数字痕迹。使用我们用于检测在线协调的机器学习框架，我们分析了一个包含2024年5月以来$\mathbb{X}$的选举相关对话的数据集。这揭示了一个协调的不真实行为者网络，显示出他们在链接分享行为上的显著相似之处。我们的分析显示，这些账户协同努力，通过协调的跨平台信息操作在网络上传播误导性、冗余和有偏见的信息：该网络共享的链接经常将用户定向到其他社交媒体平台或具有低质量政治内容的可疑网站，进而推广相同的$\mathbb{X}$和YouTube账户。该网络的成员还分享了人工智能生成的欺骗性图像，并伴随着攻击政治人物的语言和旨在传达权力和主导地位的象征性图像。虽然$\mathbb{X}$已暂停这些帐户的子集，但超过75%的协调网络仍处于活动状态。我们的发现强调了开发计算模型以扩大大型社交媒体平台上的威胁检测的关键作用，并强调了这些技术在更广泛的网络上检测iOS的更广泛影响。



## **9. Membership Inference Attacks and Privacy in Topic Modeling**

主题建模中的成员推断攻击和隐私 cs.CR

13 pages + appendices and references. 9 figures

**SubmitDate**: 2024-09-23    [abs](http://arxiv.org/abs/2403.04451v2) [paper-pdf](http://arxiv.org/pdf/2403.04451v2)

**Authors**: Nico Manzonelli, Wanrong Zhang, Salil Vadhan

**Abstract**: Recent research shows that large language models are susceptible to privacy attacks that infer aspects of the training data. However, it is unclear if simpler generative models, like topic models, share similar vulnerabilities. In this work, we propose an attack against topic models that can confidently identify members of the training data in Latent Dirichlet Allocation. Our results suggest that the privacy risks associated with generative modeling are not restricted to large neural models. Additionally, to mitigate these vulnerabilities, we explore differentially private (DP) topic modeling. We propose a framework for private topic modeling that incorporates DP vocabulary selection as a pre-processing step, and show that it improves privacy while having limited effects on practical utility.

摘要: 最近的研究表明，大型语言模型很容易受到推断训练数据各个方面的隐私攻击。然而，目前尚不清楚主题模型等更简单的生成模型是否存在类似的漏洞。在这项工作中，我们提出了一种针对主题模型的攻击，该模型可以自信地识别潜在Dirichlet分配中训练数据的成员。我们的结果表明，与生成式建模相关的隐私风险并不局限于大型神经模型。此外，为了缓解这些漏洞，我们探索了差异私密（DP）主题建模。我们提出了一个私人主题建模框架，将DP词汇选择作为预处理步骤，并表明它可以改善隐私，同时对实际实用性的影响有限。



## **10. LLM in the Shell: Generative Honeypots**

壳中的法学硕士：世代蜜罐 cs.CR

6 pages. 2 figures. 2 tables

**SubmitDate**: 2024-09-23    [abs](http://arxiv.org/abs/2309.00155v3) [paper-pdf](http://arxiv.org/pdf/2309.00155v3)

**Authors**: Muris Sladić, Veronica Valeros, Carlos Catania, Sebastian Garcia

**Abstract**: Honeypots are essential tools in cybersecurity for early detection, threat intelligence gathering, and analysis of attacker's behavior. However, most of them lack the required realism to engage and fool human attackers long-term. Being easy to distinguish honeypots strongly hinders their effectiveness. This can happen because they are too deterministic, lack adaptability, or lack deepness. This work introduces shelLM, a dynamic and realistic software honeypot based on Large Language Models that generates Linux-like shell output. We designed and implemented shelLM using cloud-based LLMs. We evaluated if shelLM can generate output as expected from a real Linux shell. The evaluation was done by asking cybersecurity researchers to use the honeypot and give feedback if each answer from the honeypot was the expected one from a Linux shell. Results indicate that shelLM can create credible and dynamic answers capable of addressing the limitations of current honeypots. ShelLM reached a TNR of 0.90, convincing humans it was consistent with a real Linux shell. The source code and prompts for replicating the experiments have been publicly available.

摘要: 蜜罐是网络安全中的重要工具，用于早期检测、威胁情报收集和分析攻击者的行为。然而，他们中的大多数缺乏必要的现实主义来长期接触和愚弄人类攻击者。容易辨别的蜜罐严重阻碍了它们的有效性。这可能是因为它们过于确定、缺乏适应性或缺乏深度。这项工作介绍了ShelLM，这是一个基于大型语言模型的动态和逼真的软件蜜罐，可以生成类似Linux的外壳输出。我们使用基于云的LLM设计并实现了ShelLM。我们评估了ShelLM是否可以从一个真正的Linux外壳程序生成预期的输出。评估是通过让网络安全研究人员使用蜜罐进行的，如果蜜罐的每个答案都是来自Linux外壳的预期答案，则给出反馈。结果表明，ShelLM能够生成可信的、动态的答案，能够解决当前蜜罐的局限性。ShelLm的TNR达到了0.90，让人相信它与真正的Linux外壳是一致的。复制这些实验的源代码和提示已经公开。



## **11. Attack Atlas: A Practitioner's Perspective on Challenges and Pitfalls in Red Teaming GenAI**

攻击地图集：从业者对红色团队GenAI挑战和陷阱的看法 cs.CR

**SubmitDate**: 2024-09-23    [abs](http://arxiv.org/abs/2409.15398v1) [paper-pdf](http://arxiv.org/pdf/2409.15398v1)

**Authors**: Ambrish Rawat, Stefan Schoepf, Giulio Zizzo, Giandomenico Cornacchia, Muhammad Zaid Hameed, Kieran Fraser, Erik Miehling, Beat Buesser, Elizabeth M. Daly, Mark Purcell, Prasanna Sattigeri, Pin-Yu Chen, Kush R. Varshney

**Abstract**: As generative AI, particularly large language models (LLMs), become increasingly integrated into production applications, new attack surfaces and vulnerabilities emerge and put a focus on adversarial threats in natural language and multi-modal systems. Red-teaming has gained importance in proactively identifying weaknesses in these systems, while blue-teaming works to protect against such adversarial attacks. Despite growing academic interest in adversarial risks for generative AI, there is limited guidance tailored for practitioners to assess and mitigate these challenges in real-world environments. To address this, our contributions include: (1) a practical examination of red- and blue-teaming strategies for securing generative AI, (2) identification of key challenges and open questions in defense development and evaluation, and (3) the Attack Atlas, an intuitive framework that brings a practical approach to analyzing single-turn input attacks, placing it at the forefront for practitioners. This work aims to bridge the gap between academic insights and practical security measures for the protection of generative AI systems.

摘要: 随着产生式人工智能，特别是大型语言模型(LLM)越来越多地集成到生产应用中，新的攻击面和漏洞出现，并将重点放在自然语言和多模式系统中的对抗性威胁上。红团队在主动识别这些系统中的弱点方面变得越来越重要，而蓝团队的工作是防止这种对手攻击。尽管学术界对生成性人工智能的对抗风险越来越感兴趣，但为实践者量身定做的指导意见有限，以评估和缓解现实世界环境中的这些挑战。为了解决这个问题，我们的贡献包括：(1)确保生成性人工智能的红色和蓝色团队战略的实践研究，(2)识别防御开发和评估中的关键挑战和开放问题，以及(3)攻击图集，这是一个直观的框架，为分析单回合输入攻击带来了实用方法，使其处于实践者的前沿。这项工作旨在弥合学术见解和保护生成性人工智能系统的实际安全措施之间的差距。



## **12. Effective and Evasive Fuzz Testing-Driven Jailbreaking Attacks against LLMs**

针对LLM的有效且规避的模糊测试驱动越狱攻击 cs.CR

**SubmitDate**: 2024-09-23    [abs](http://arxiv.org/abs/2409.14866v1) [paper-pdf](http://arxiv.org/pdf/2409.14866v1)

**Authors**: Xueluan Gong, Mingzhe Li, Yilin Zhang, Fengyuan Ran, Chen Chen, Yanjiao Chen, Qian Wang, Kwok-Yan Lam

**Abstract**: Large Language Models (LLMs) have excelled in various tasks but are still vulnerable to jailbreaking attacks, where attackers create jailbreak prompts to mislead the model to produce harmful or offensive content. Current jailbreak methods either rely heavily on manually crafted templates, which pose challenges in scalability and adaptability, or struggle to generate semantically coherent prompts, making them easy to detect. Additionally, most existing approaches involve lengthy prompts, leading to higher query costs.In this paper, to remedy these challenges, we introduce a novel jailbreaking attack framework, which is an automated, black-box jailbreaking attack framework that adapts the black-box fuzz testing approach with a series of customized designs. Instead of relying on manually crafted templates, our method starts with an empty seed pool, removing the need to search for any related jailbreaking templates. We also develop three novel question-dependent mutation strategies using an LLM helper to generate prompts that maintain semantic coherence while significantly reducing their length. Additionally, we implement a two-level judge module to accurately detect genuine successful jailbreaks.   We evaluated our method on 7 representative LLMs and compared it with 5 state-of-the-art jailbreaking attack strategies. For proprietary LLM APIs, such as GPT-3.5 turbo, GPT-4, and Gemini-Pro, our method achieves attack success rates of over 90%, 80%, and 74%, respectively, exceeding existing baselines by more than 60%. Additionally, our method can maintain high semantic coherence while significantly reducing the length of jailbreak prompts. When targeting GPT-4, our method can achieve over 78\% attack success rate even with 100 tokens. Moreover, our method demonstrates transferability and is robust to state-of-the-art defenses. We will open-source our codes upon publication.

摘要: 大型语言模型(LLM)在各种任务中表现出色，但仍然容易受到越狱攻击，在越狱攻击中，攻击者创建越狱提示来误导模型生成有害或攻击性内容。当前的越狱方法要么严重依赖于人工制作的模板，这对可伸缩性和适应性构成了挑战，要么难以生成语义连贯的提示，使它们很容易被检测到。针对现有方法存在提示过长、查询代价较高的问题，提出了一种新的越狱攻击框架，该框架采用了一系列定制的黑盒模糊测试方法，是一种自动化的黑盒越狱攻击框架。我们的方法不依赖于手动创建的模板，而是从一个空的种子池开始，不需要搜索任何相关的越狱模板。我们还开发了三种新的问题相关突变策略，使用LLM助手来生成提示，这些提示在保持语义连贯的同时显著缩短了提示的长度。此外，我们实现了一个两级判断模块来准确地检测真正的成功越狱。我们在7个有代表性的LLM上对我们的方法进行了评估，并将其与5种最先进的越狱攻击策略进行了比较。对于专有的LLMAPI，如GPT-3.5 Turbo、GPT-4和Gemini-Pro，我们的方法分别实现了90%、80%和74%以上的攻击成功率，比现有基线高出60%以上。此外，我们的方法可以保持较高的语义一致性，同时显著减少越狱提示的长度。当攻击目标为GPT-4时，即使有100个令牌，我们的方法也可以达到78%以上的攻击成功率。此外，我们的方法证明了可转移性，并对最先进的防御措施具有健壮性。我们将在发布后将我们的代码开源。



## **13. A Security Risk Taxonomy for Prompt-Based Interaction With Large Language Models**

基于预算的与大型语言模型交互的安全风险分类 cs.CR

**SubmitDate**: 2024-09-23    [abs](http://arxiv.org/abs/2311.11415v2) [paper-pdf](http://arxiv.org/pdf/2311.11415v2)

**Authors**: Erik Derner, Kristina Batistič, Jan Zahálka, Robert Babuška

**Abstract**: As large language models (LLMs) permeate more and more applications, an assessment of their associated security risks becomes increasingly necessary. The potential for exploitation by malicious actors, ranging from disinformation to data breaches and reputation damage, is substantial. This paper addresses a gap in current research by specifically focusing on security risks posed by LLMs within the prompt-based interaction scheme, which extends beyond the widely covered ethical and societal implications. Our work proposes a taxonomy of security risks along the user-model communication pipeline and categorizes the attacks by target and attack type alongside the commonly used confidentiality, integrity, and availability (CIA) triad. The taxonomy is reinforced with specific attack examples to showcase the real-world impact of these risks. Through this taxonomy, we aim to inform the development of robust and secure LLM applications, enhancing their safety and trustworthiness.

摘要: 随着大型语言模型（LLM）渗透到越来越多的应用程序中，对其相关安全风险的评估变得越来越有必要。被恶意行为者利用的可能性很大，从虚假信息到数据泄露和声誉损害。本文通过特别关注基于预算的交互方案中LLM构成的安全风险来解决当前研究中的一个空白，该方案超出了广泛涵盖的道德和社会影响。我们的工作提出了用户模型通信管道中的安全风险分类法，并根据目标和攻击类型以及常用的机密性、完整性和可用性（CIA）三位一体对攻击进行分类。该分类通过特定的攻击示例得到了加强，以展示这些风险对现实世界的影响。通过这种分类法，我们的目标是为强大和安全的LLM应用程序的开发提供信息，增强其安全性和可信性。



## **14. LlamaPartialSpoof: An LLM-Driven Fake Speech Dataset Simulating Disinformation Generation**

LlamaPartialSpoof：一个LLM驱动的模拟虚假信息生成的假语音数据集 eess.AS

5 pages, submitted to ICASSP 2025

**SubmitDate**: 2024-09-23    [abs](http://arxiv.org/abs/2409.14743v1) [paper-pdf](http://arxiv.org/pdf/2409.14743v1)

**Authors**: Hieu-Thi Luong, Haoyang Li, Lin Zhang, Kong Aik Lee, Eng Siong Chng

**Abstract**: Previous fake speech datasets were constructed from a defender's perspective to develop countermeasure (CM) systems without considering diverse motivations of attackers. To better align with real-life scenarios, we created LlamaPartialSpoof, a 130-hour dataset contains both fully and partially fake speech, using a large language model (LLM) and voice cloning technologies to evaluate the robustness of CMs. By examining information valuable to both attackers and defenders, we identify several key vulnerabilities in current CM systems, which can be exploited to enhance attack success rates, including biases toward certain text-to-speech models or concatenation methods. Our experimental results indicate that current fake speech detection system struggle to generalize to unseen scenarios, achieving a best performance of 24.44% equal error rate.

摘要: 之前的虚假语音数据集是从防御者的角度构建的，以开发对策（CM）系统，而不考虑攻击者的不同动机。为了更好地与现实生活场景保持一致，我们创建了LlamaPartialSpoof，这是一个包含完全和部分虚假语音的130小时数据集，使用大型语言模型（LLM）和语音克隆技术来评估CM的稳健性。通过检查对攻击者和防御者都有价值的信息，我们发现了当前CM系统中的几个关键漏洞，可以利用这些漏洞来提高攻击成功率，包括对某些文本到语音模型或级联方法的偏见。我们的实验结果表明，当前的虚假语音检测系统很难推广到不可见的场景，实现了24.44%等错误率的最佳性能。



## **15. PROMPTFUZZ: Harnessing Fuzzing Techniques for Robust Testing of Prompt Injection in LLMs**

PROMPTFUZZ：利用模糊技术对LLM中的即时注射进行稳健测试 cs.CR

**SubmitDate**: 2024-09-23    [abs](http://arxiv.org/abs/2409.14729v1) [paper-pdf](http://arxiv.org/pdf/2409.14729v1)

**Authors**: Jiahao Yu, Yangguang Shao, Hanwen Miao, Junzheng Shi, Xinyu Xing

**Abstract**: Large Language Models (LLMs) have gained widespread use in various applications due to their powerful capability to generate human-like text. However, prompt injection attacks, which involve overwriting a model's original instructions with malicious prompts to manipulate the generated text, have raised significant concerns about the security and reliability of LLMs. Ensuring that LLMs are robust against such attacks is crucial for their deployment in real-world applications, particularly in critical tasks.   In this paper, we propose PROMPTFUZZ, a novel testing framework that leverages fuzzing techniques to systematically assess the robustness of LLMs against prompt injection attacks. Inspired by software fuzzing, PROMPTFUZZ selects promising seed prompts and generates a diverse set of prompt injections to evaluate the target LLM's resilience. PROMPTFUZZ operates in two stages: the prepare phase, which involves selecting promising initial seeds and collecting few-shot examples, and the focus phase, which uses the collected examples to generate diverse, high-quality prompt injections. Using PROMPTFUZZ, we can uncover more vulnerabilities in LLMs, even those with strong defense prompts.   By deploying the generated attack prompts from PROMPTFUZZ in a real-world competition, we achieved the 7th ranking out of over 4000 participants (top 0.14%) within 2 hours. Additionally, we construct a dataset to fine-tune LLMs for enhanced robustness against prompt injection attacks. While the fine-tuned model shows improved robustness, PROMPTFUZZ continues to identify vulnerabilities, highlighting the importance of robust testing for LLMs. Our work emphasizes the critical need for effective testing tools and provides a practical framework for evaluating and improving the robustness of LLMs against prompt injection attacks.

摘要: 大语言模型(LLM)由于其强大的生成类人类文本的能力，在各种应用中得到了广泛的应用。然而，提示注入攻击包括用恶意提示覆盖模型的原始指令来操作生成的文本，这引发了人们对LLMS安全性和可靠性的严重担忧。确保LLM对此类攻击具有健壮性，对于它们在现实应用中的部署至关重要，特别是在关键任务中。在本文中，我们提出了一种新的测试框架PROMPTFUZZ，它利用模糊技术来系统地评估LLMS对快速注入攻击的健壮性。受软件模糊的启发，PROMPTFUZZ选择有希望的种子提示，并生成一组不同的提示注射，以评估目标LLM的弹性。PROMPTFUZZ分为两个阶段：准备阶段，包括选择有希望的初始种子和收集少量样本；以及聚焦阶段，使用收集的样本来生成各种高质量的即时注射。使用PROMPTFUZZ，我们可以发现LLMS中的更多漏洞，即使是那些具有强大防御提示的漏洞。通过将PROMPTFUZZ生成的攻击提示部署在现实世界的比赛中，我们在2小时内获得了4000多名参与者中的第7名(前0.14%)。此外，我们构建了一个数据集来微调LLMS，以增强对即时注入攻击的健壮性。虽然微调的模型显示了更好的健壮性，但PROMPTFUZZ继续识别漏洞，强调了对LLM进行健壮测试的重要性。我们的工作强调了对有效测试工具的迫切需要，并为评估和提高LLMS对快速注入攻击的健壮性提供了一个实用的框架。



## **16. Enhancing LLM-based Autonomous Driving Agents to Mitigate Perception Attacks**

增强基于LLM的自动驾驶代理以缓解感知攻击 cs.CR

**SubmitDate**: 2024-09-22    [abs](http://arxiv.org/abs/2409.14488v1) [paper-pdf](http://arxiv.org/pdf/2409.14488v1)

**Authors**: Ruoyu Song, Muslum Ozgur Ozmen, Hyungsub Kim, Antonio Bianchi, Z. Berkay Celik

**Abstract**: There is a growing interest in integrating Large Language Models (LLMs) with autonomous driving (AD) systems. However, AD systems are vulnerable to attacks against their object detection and tracking (ODT) functions. Unfortunately, our evaluation of four recent LLM agents against ODT attacks shows that the attacks are 63.26% successful in causing them to crash or violate traffic rules due to (1) misleading memory modules that provide past experiences for decision making, (2) limitations of prompts in identifying inconsistencies, and (3) reliance on ground truth perception data.   In this paper, we introduce Hudson, a driving reasoning agent that extends prior LLM-based driving systems to enable safer decision making during perception attacks while maintaining effectiveness under benign conditions. Hudson achieves this by first instrumenting the AD software to collect real-time perception results and contextual information from the driving scene. This data is then formalized into a domain-specific language (DSL). To guide the LLM in detecting and making safe control decisions during ODT attacks, Hudson translates the DSL into natural language, along with a list of custom attack detection instructions. Following query execution, Hudson analyzes the LLM's control decision to understand its causal reasoning process.   We evaluate the effectiveness of Hudson using a proprietary LLM (GPT-4) and two open-source LLMs (Llama and Gemma) in various adversarial driving scenarios. GPT-4, Llama, and Gemma achieve, on average, an attack detection accuracy of 83. 3%, 63. 6%, and 73. 6%. Consequently, they make safe control decisions in 86.4%, 73.9%, and 80% of the attacks. Our results, following the growing interest in integrating LLMs into AD systems, highlight the strengths of LLMs and their potential to detect and mitigate ODT attacks.

摘要: 人们对将大型语言模型(LLM)与自动驾驶(AD)系统进行集成的兴趣日益浓厚。然而，AD系统的目标检测和跟踪(ODT)功能容易受到攻击。不幸的是，我们对最近针对ODT攻击的四个LLM代理的评估显示，由于(1)为决策提供过去经验的误导性存储模块，(2)在识别不一致方面的提示限制，以及(3)对地面真相感知数据的依赖，这些攻击在导致它们崩溃或违反交通规则方面的成功率为63.26%。在本文中，我们引入了Hudson，一个驾驶推理代理，它扩展了现有的基于LLM的驾驶系统，使之能够在感知攻击时做出更安全的决策，同时在良性条件下保持有效性。哈德森首先利用广告软件从驾驶场景中收集实时感知结果和背景信息，从而实现了这一点。然后将这些数据形式化为特定于域的语言(DSL)。为了指导LLM在ODT攻击期间检测和做出安全的控制决策，Hudson将DSL翻译成自然语言，并提供了一系列自定义攻击检测指令。在查询执行之后，Hudson分析LLM的控制决策以理解其因果推理过程。我们使用一个专有的LLM(GPT-4)和两个开源的LLM(Llama和Gema)来评估Hudson在各种对抗性驾驶场景中的有效性。GPT-4、Llama和Gema的攻击检测准确率平均为83。3%，63.6%和73%。6%。因此，他们在86.4%、73.9%和80%的攻击中做出了安全控制决策。随着人们对将LLMS集成到AD系统中的兴趣与日俱增，我们的结果突显了LLMS的优势及其检测和缓解ODT攻击的潜力。



## **17. PathSeeker: Exploring LLM Security Vulnerabilities with a Reinforcement Learning-Based Jailbreak Approach**

PathSeeker：使用基于强化学习的越狱方法探索LLM安全漏洞 cs.CR

**SubmitDate**: 2024-09-21    [abs](http://arxiv.org/abs/2409.14177v1) [paper-pdf](http://arxiv.org/pdf/2409.14177v1)

**Authors**: Zhihao Lin, Wei Ma, Mingyi Zhou, Yanjie Zhao, Haoyu Wang, Yang Liu, Jun Wang, Li Li

**Abstract**: In recent years, Large Language Models (LLMs) have gained widespread use, accompanied by increasing concerns over their security. Traditional jailbreak attacks rely on internal model details or have limitations when exploring the unsafe behavior of the victim model, limiting their generalizability. In this paper, we introduce PathSeeker, a novel black-box jailbreak method inspired by the concept of escaping a security maze. This work is inspired by the game of rats escaping a maze. We think that each LLM has its unique "security maze", and attackers attempt to find the exit learning from the received feedback and their accumulated experience to compromise the target LLM's security defences. Our approach leverages multi-agent reinforcement learning, where smaller models collaborate to guide the main LLM in performing mutation operations to achieve the attack objectives. By progressively modifying inputs based on the model's feedback, our system induces richer, harmful responses. During our manual attempts to perform jailbreak attacks, we found that the vocabulary of the response of the target model gradually became richer and eventually produced harmful responses. Based on the observation, we also introduce a reward mechanism that exploits the expansion of vocabulary richness in LLM responses to weaken security constraints. Our method outperforms five state-of-the-art attack techniques when tested across 13 commercial and open-source LLMs, achieving high attack success rates, especially in strongly aligned commercial models like GPT-4o-mini, Claude-3.5, and GLM-4-air with strong safety alignment. This study aims to improve the understanding of LLM security vulnerabilities and we hope that this sturdy can contribute to the development of more robust defenses.

摘要: 近年来，大型语言模型得到了广泛的应用，人们对其安全性的担忧也与日俱增。传统的越狱攻击依赖于内部模型细节，或者在探索受害者模型的不安全行为时存在局限性，限制了它们的通用性。在本文中，我们介绍了一种新的黑盒越狱方法--Path Seeker，它的灵感来自于逃离安全迷宫的概念。这项工作的灵感来自老鼠逃离迷宫的游戏。我们认为，每个LLM都有自己独特的安全迷宫，攻击者试图从收到的反馈和他们积累的经验中找到出口学习，从而破坏目标LLM的安全防御。我们的方法利用多代理强化学习，其中较小的模型协作来指导主要的LLM执行突变操作，以实现攻击目标。通过根据模型的反馈逐步修改输入，我们的系统会产生更丰富、更有害的反应。在我们手动尝试执行越狱攻击的过程中，我们发现目标模型的响应词汇逐渐变得更加丰富，最终产生了有害的响应。在此基础上，我们还引入了一种奖励机制，该机制利用LLM响应中词汇丰富性的扩展来削弱安全约束。我们的方法在13个商业和开源LLM上进行测试时，性能超过了五种最先进的攻击技术，实现了高攻击成功率，特别是在GPT-40-mini、Claude-3.5和GLM-4-AIR等高度一致的商业型号上，具有很强的安全一致性。这项研究旨在提高对LLM安全漏洞的理解，我们希望这一强项可以为开发更强大的防御措施做出贡献。



## **18. Obliviate: Neutralizing Task-agnostic Backdoors within the Parameter-efficient Fine-tuning Paradigm**

遗忘：消除参数高效微调范式中的任务不可知后门 cs.CL

Under Review

**SubmitDate**: 2024-09-21    [abs](http://arxiv.org/abs/2409.14119v1) [paper-pdf](http://arxiv.org/pdf/2409.14119v1)

**Authors**: Jaehan Kim, Minkyoo Song, Seung Ho Na, Seungwon Shin

**Abstract**: Parameter-efficient fine-tuning (PEFT) has become a key training strategy for large language models. However, its reliance on fewer trainable parameters poses security risks, such as task-agnostic backdoors. Despite their severe impact on a wide range of tasks, there is no practical defense solution available that effectively counters task-agnostic backdoors within the context of PEFT. In this study, we introduce Obliviate, a PEFT-integrable backdoor defense. We develop two techniques aimed at amplifying benign neurons within PEFT layers and penalizing the influence of trigger tokens. Our evaluations across three major PEFT architectures show that our method can significantly reduce the attack success rate of the state-of-the-art task-agnostic backdoors (83.6%$\downarrow$). Furthermore, our method exhibits robust defense capabilities against both task-specific backdoors and adaptive attacks. Source code will be obtained at https://github.com/obliviateARR/Obliviate.

摘要: 参数高效微调（PEFT）已成为大型语言模型的关键训练策略。然而，它对较少可训练参数的依赖会带来安全风险，例如任务不可知的后门。尽管它们对广泛的任务产生了严重影响，但没有实用的防御解决方案可以有效地对抗PEFT背景下的任务不可知后门。在这项研究中，我们引入Obliviate，一种PEFT可集成的后门防御。我们开发了两种技术，旨在放大PEFT层内的良性神经元并惩罚触发代币的影响。我们对三种主要PEFT架构的评估表明，我们的方法可以显着降低最先进的任务不可知后门（83.6%$\down arrow $）的攻击成功率。此外，我们的方法对特定任务的后门和自适应攻击都表现出强大的防御能力。源代码可在https://github.com/obliviateARR/Obliviate上获取。



## **19. Trading Devil Final: Backdoor attack via Stock market and Bayesian Optimization**

交易魔鬼决赛：通过股市和Bayesian优化进行后门攻击 cs.LG

END (will never be modified again!!) :Jumps-Diffusion and stock  market: Better quantify uncertainty in financial simulations

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2407.14573v6) [paper-pdf](http://arxiv.org/pdf/2407.14573v6)

**Authors**: Orson Mengara

**Abstract**: Since the advent of generative artificial intelligence, every company and researcher has been rushing to develop their own generative models, whether commercial or not. Given the large number of users of these powerful new tools, there is currently no intrinsically verifiable way to explain from the ground up what happens when LLMs (large language models) learn. For example, those based on automatic speech recognition systems, which have to rely on huge and astronomical amounts of data collected from all over the web to produce fast and efficient results, In this article, we develop a backdoor attack called MarketBackFinal 2.0, based on acoustic data poisoning, MarketBackFinal 2.0 is mainly based on modern stock market models. In order to show the possible vulnerabilities of speech-based transformers that may rely on LLMs.

摘要: 自生成人工智能出现以来，每家公司和研究人员都在争先恐后地开发自己的生成模型，无论是否商业化。鉴于这些强大的新工具的大量用户，目前还没有本质上可验证的方法来从头解释LLM（大型语言模型）学习时会发生什么。例如，那些基于自动语音识别系统的系统，它们必须依赖于从整个网络收集的大量数据来产生快速有效的结果，在本文中，我们开发了一种名为MarketBackFinal 2.0的后门攻击，基于声学数据中毒，MarketBackFinal 2.0主要基于现代股市模型。为了显示可能依赖LLM的基于语音的转换器可能存在的漏洞。



## **20. ConfusedPilot: Confused Deputy Risks in RAG-based LLMs**

困惑的飞行员：基于RAG的LLM中令人困惑的代理风险 cs.CR

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2408.04870v4) [paper-pdf](http://arxiv.org/pdf/2408.04870v4)

**Authors**: Ayush RoyChowdhury, Mulong Luo, Prateek Sahu, Sarbartha Banerjee, Mohit Tiwari

**Abstract**: Retrieval augmented generation (RAG) is a process where a large language model (LLM) retrieves useful information from a database and then generates the responses. It is becoming popular in enterprise settings for daily business operations. For example, Copilot for Microsoft 365 has accumulated millions of businesses. However, the security implications of adopting such RAG-based systems are unclear.   In this paper, we introduce ConfusedPilot, a class of security vulnerabilities of RAG systems that confuse Copilot and cause integrity and confidentiality violations in its responses. First, we investigate a vulnerability that embeds malicious text in the modified prompt in RAG, corrupting the responses generated by the LLM. Second, we demonstrate a vulnerability that leaks secret data, which leverages the caching mechanism during retrieval. Third, we investigate how both vulnerabilities can be exploited to propagate misinformation within the enterprise and ultimately impact its operations, such as sales and manufacturing. We also discuss the root cause of these attacks by investigating the architecture of a RAG-based system. This study highlights the security vulnerabilities in today's RAG-based systems and proposes design guidelines to secure future RAG-based systems.

摘要: 检索增强生成(RAG)是大型语言模型(LLM)从数据库中检索有用信息然后生成响应的过程。它在用于日常业务操作的企业环境中变得流行起来。例如，微软365的Copilot已经积累了数百万笔业务。然而，采用这种基于RAG的系统的安全影响尚不清楚。在本文中，我们介绍了一类RAG系统的安全漏洞ConfusedPilot，它迷惑了Copilot，并在其响应中导致完整性和保密性违规。首先，我们调查了一个漏洞，该漏洞将恶意文本嵌入到RAG中修改的提示符中，破坏了LLM生成的响应。其次，我们演示了一个泄漏机密数据的漏洞，该漏洞在检索过程中利用缓存机制。第三，我们调查如何利用这两个漏洞在企业内部传播错误信息，并最终影响其运营，如销售和制造。我们还通过研究基于RAG的系统的体系结构来讨论这些攻击的根本原因。这项研究强调了当今基于RAG的系统中的安全漏洞，并提出了保护未来基于RAG的系统的设计指南。



## **21. On the Feasibility of Fully AI-automated Vishing Attacks**

全人工智能自动Vising攻击的可行性 cs.CR

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2409.13793v1) [paper-pdf](http://arxiv.org/pdf/2409.13793v1)

**Authors**: João Figueiredo, Afonso Carvalho, Daniel Castro, Daniel Gonçalves, Nuno Santos

**Abstract**: A vishing attack is a form of social engineering where attackers use phone calls to deceive individuals into disclosing sensitive information, such as personal data, financial information, or security credentials. Attackers exploit the perceived urgency and authenticity of voice communication to manipulate victims, often posing as legitimate entities like banks or tech support. Vishing is a particularly serious threat as it bypasses security controls designed to protect information. In this work, we study the potential for vishing attacks to escalate with the advent of AI. In theory, AI-powered software bots may have the ability to automate these attacks by initiating conversations with potential victims via phone calls and deceiving them into disclosing sensitive information. To validate this thesis, we introduce ViKing, an AI-powered vishing system developed using publicly available AI technology. It relies on a Large Language Model (LLM) as its core cognitive processor to steer conversations with victims, complemented by a pipeline of speech-to-text and text-to-speech modules that facilitate audio-text conversion in phone calls. Through a controlled social experiment involving 240 participants, we discovered that ViKing has successfully persuaded many participants to reveal sensitive information, even those who had been explicitly warned about the risk of vishing campaigns. Interactions with ViKing's bots were generally considered realistic. From these findings, we conclude that tools like ViKing may already be accessible to potential malicious actors, while also serving as an invaluable resource for cyber awareness programs.

摘要: 恶意攻击是一种社会工程形式，攻击者使用电话欺骗个人泄露敏感信息，如个人数据、财务信息或安全凭据。攻击者利用感知到的语音通信的紧迫性和真实性来操纵受害者，通常伪装成银行或技术支持等合法实体。网络钓鱼是一个特别严重的威胁，因为它绕过了旨在保护信息的安全控制。在这项工作中，我们研究了随着人工智能的出现，恶意攻击升级的可能性。理论上，人工智能驱动的软件机器人可能有能力通过电话发起与潜在受害者的对话，并欺骗他们泄露敏感信息，从而实现这些攻击的自动化。为了验证这篇论文，我们介绍了Viking，一个使用公开可用的AI技术开发的AI支持的钓鱼系统。它依赖大型语言模型(LLM)作为其核心认知处理器来引导与受害者的对话，并辅之以一系列语音到文本和文本到语音的模块，以促进电话中的音频文本转换。通过一项涉及240名参与者的受控社会实验，我们发现维京海盗成功地说服了许多参与者泄露敏感信息，即使是那些明确警告过破坏活动风险的人也是如此。与维京机器人的互动通常被认为是现实的。从这些发现中，我们得出结论，像Viking这样的工具可能已经可以被潜在的恶意行为者访问，同时也是网络感知程序的宝贵资源。



## **22. Prompt Obfuscation for Large Language Models**

大型语言模型的提示混淆 cs.CR

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2409.11026v2) [paper-pdf](http://arxiv.org/pdf/2409.11026v2)

**Authors**: David Pape, Thorsten Eisenhofer, Lea Schönherr

**Abstract**: System prompts that include detailed instructions to describe the task performed by the underlying large language model (LLM) can easily transform foundation models into tools and services with minimal overhead. Because of their crucial impact on the utility, they are often considered intellectual property, similar to the code of a software product. However, extracting system prompts is easily possible by using prompt injection. As of today, there is no effective countermeasure to prevent the stealing of system prompts and all safeguarding efforts could be evaded with carefully crafted prompt injections that bypass all protection mechanisms. In this work, we propose an alternative to conventional system prompts. We introduce prompt obfuscation to prevent the extraction of the system prompt while maintaining the utility of the system itself with only little overhead. The core idea is to find a representation of the original system prompt that leads to the same functionality, while the obfuscated system prompt does not contain any information that allows conclusions to be drawn about the original system prompt. We implement an optimization-based method to find an obfuscated prompt representation while maintaining the functionality. To evaluate our approach, we investigate eight different metrics to compare the performance of a system using the original and the obfuscated system prompts, and we show that the obfuscated version is constantly on par with the original one. We further perform three different deobfuscation attacks and show that with access to the obfuscated prompt and the LLM itself, we are not able to consistently extract meaningful information. Overall, we showed that prompt obfuscation can be an effective method to protect intellectual property while maintaining the same utility as the original system prompt.

摘要: 系统提示包括描述底层大型语言模型(LLM)执行的任务的详细说明，可以轻松地将基础模型转换为工具和服务，开销最小。由于它们对实用程序的关键影响，它们通常被视为知识产权，类似于软件产品的代码。但是，通过使用提示注入，很容易提取系统提示。到目前为止，还没有有效的对策来防止系统提示被窃取，所有的保护工作都可以通过精心设计的绕过所有保护机制的快速注入来规避。在这项工作中，我们提出了一种替代传统系统提示的方法。我们引入即时混淆来防止系统提示符的提取，同时保持系统本身的效用，而只需很少的开销。其核心思想是找到导致相同功能的原始系统提示的表示，而混淆的系统提示不包含任何允许对原始系统提示得出结论的信息。我们实现了一种基于优化的方法，在保持功能的同时找到混淆的提示表示。为了评估我们的方法，我们调查了八个不同的度量来比较使用原始系统提示和模糊系统提示的系统的性能，我们证明了模糊版本与原始提示是相同的。我们进一步执行了三种不同的去模糊攻击，并表明通过访问混淆提示和LLM本身，我们不能一致地提取有意义的信息。总体而言，我们表明，即时混淆可以是一种有效的方法来保护知识产权，同时保持与原始系统提示相同的效用。



## **23. Applying Pre-trained Multilingual BERT in Embeddings for Improved Malicious Prompt Injection Attacks Detection**

在嵌入中应用预训练的多语言BERT以改进恶意提示注入攻击检测 cs.CL

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2409.13331v1) [paper-pdf](http://arxiv.org/pdf/2409.13331v1)

**Authors**: Md Abdur Rahman, Hossain Shahriar, Fan Wu, Alfredo Cuzzocrea

**Abstract**: Large language models (LLMs) are renowned for their exceptional capabilities, and applying to a wide range of applications. However, this widespread use brings significant vulnerabilities. Also, it is well observed that there are huge gap which lies in the need for effective detection and mitigation strategies against malicious prompt injection attacks in large language models, as current approaches may not adequately address the complexity and evolving nature of these vulnerabilities in real-world applications. Therefore, this work focuses the impact of malicious prompt injection attacks which is one of most dangerous vulnerability on real LLMs applications. It examines to apply various BERT (Bidirectional Encoder Representations from Transformers) like multilingual BERT, DistilBert for classifying malicious prompts from legitimate prompts. Also, we observed how tokenizing the prompt texts and generating embeddings using multilingual BERT contributes to improve the performance of various machine learning methods: Gaussian Naive Bayes, Random Forest, Support Vector Machine, and Logistic Regression. The performance of each model is rigorously analyzed with various parameters to improve the binary classification to discover malicious prompts. Multilingual BERT approach to embed the prompts significantly improved and outperformed the existing works and achieves an outstanding accuracy of 96.55% by Logistic regression. Additionally, we investigated the incorrect predictions of the model to gain insights into its limitations. The findings can guide researchers in tuning various BERT for finding the most suitable model for diverse LLMs vulnerabilities.

摘要: 大型语言模型(LLM)以其卓越的功能而闻名，并适用于广泛的应用程序。然而，这种广泛的使用带来了严重的漏洞。此外，人们还注意到，对于大型语言模型中的恶意提示注入攻击，需要有效的检测和缓解策略，这方面存在巨大的差距，因为当前的方法可能无法充分解决现实世界应用程序中这些漏洞的复杂性和不断演变的性质。因此，本文重点研究了恶意提示注入攻击对实际LLMS应用中最危险的漏洞之一的影响。它研究了应用各种BERT(来自Transformers的双向编码器表示)，如多语言BERT、DistilBert来将恶意提示与合法提示区分开来。此外，我们还观察了如何使用多语言BERT来标记化提示文本并生成嵌入内容，从而有助于提高各种机器学习方法的性能：高斯朴素贝叶斯、随机森林、支持向量机和Logistic回归。使用各种参数严格分析每个模型的性能，以改进二进制分类来发现恶意提示。多语种BERT方法嵌入提示的效果明显优于已有工作，经Logistic回归分析，准确率达到96.55%。此外，我们调查了该模型的错误预测，以深入了解其局限性。这些发现可以指导研究人员调整各种BERT，为不同的LLMS漏洞找到最合适的模型。



## **24. An Adaptive End-to-End IoT Security Framework Using Explainable AI and LLMs**

使用可解释人工智能和LLM的自适应端到端物联网安全框架 cs.LG

6 pages, 1 figure, Accepted in 2024 IEEE WF-IoT Conference

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2409.13177v1) [paper-pdf](http://arxiv.org/pdf/2409.13177v1)

**Authors**: Sudipto Baral, Sajal Saha, Anwar Haque

**Abstract**: The exponential growth of the Internet of Things (IoT) has significantly increased the complexity and volume of cybersecurity threats, necessitating the development of advanced, scalable, and interpretable security frameworks. This paper presents an innovative, comprehensive framework for real-time IoT attack detection and response that leverages Machine Learning (ML), Explainable AI (XAI), and Large Language Models (LLM). By integrating XAI techniques such as SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) with a model-independent architecture, we ensure our framework's adaptability across various ML algorithms. Additionally, the incorporation of LLMs enhances the interpretability and accessibility of detection decisions, providing system administrators with actionable, human-understandable explanations of detected threats. Our end-to-end framework not only facilitates a seamless transition from model development to deployment but also represents a real-world application capability that is often lacking in existing research. Based on our experiments with the CIC-IOT-2023 dataset \cite{neto2023ciciot2023}, Gemini and OPENAI LLMS demonstrate unique strengths in attack mitigation: Gemini offers precise, focused strategies, while OPENAI provides extensive, in-depth security measures. Incorporating SHAP and LIME algorithms within XAI provides comprehensive insights into attack detection, emphasizing opportunities for model improvement through detailed feature analysis, fine-tuning, and the adaptation of misclassifications to enhance accuracy.

摘要: 物联网(IoT)的指数增长显著增加了网络安全威胁的复杂性和数量，需要开发高级、可扩展和可解释的安全框架。本文提出了一种利用机器学习(ML)、可解释人工智能(XAI)和大型语言模型(LLM)进行实时物联网攻击检测和响应的创新、全面的框架。通过将Shap和LIME等XAI技术与独立于模型的体系结构相结合，我们确保了框架对各种ML算法的适应性。此外，LLMS的加入增强了检测决策的可解释性和可访问性，为系统管理员提供了可操作的、人类可理解的对检测到的威胁的解释。我们的端到端框架不仅促进了从模型开发到部署的无缝过渡，而且还代表了现有研究中经常缺乏的现实世界应用程序能力。基于我们对CIC-IOT-2023数据集的实验，Gemini和OpenAI LLMS在攻击缓解方面展示了独特的优势：Gemini提供精确、有针对性的策略，而OpenAI提供广泛、深入的安全措施。在XAI中结合Shap和LIME算法可为攻击检测提供全面的见解，强调通过详细的特征分析、微调和调整错误分类来改进模型以提高准确性的机会。



## **25. Manipulation Facing Threats: Evaluating Physical Vulnerabilities in End-to-End Vision Language Action Models**

面对威胁的操纵：评估端到端视觉语言动作模型中的身体脆弱性 cs.CV

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2409.13174v1) [paper-pdf](http://arxiv.org/pdf/2409.13174v1)

**Authors**: Hao Cheng, Erjia Xiao, Chengyuan Yu, Zhao Yao, Jiahang Cao, Qiang Zhang, Jiaxu Wang, Mengshu Sun, Kaidi Xu, Jindong Gu, Renjing Xu

**Abstract**: Recently, driven by advancements in Multimodal Large Language Models (MLLMs), Vision Language Action Models (VLAMs) are being proposed to achieve better performance in open-vocabulary scenarios for robotic manipulation tasks. Since manipulation tasks involve direct interaction with the physical world, ensuring robustness and safety during the execution of this task is always a very critical issue. In this paper, by synthesizing current safety research on MLLMs and the specific application scenarios of the manipulation task in the physical world, we comprehensively evaluate VLAMs in the face of potential physical threats. Specifically, we propose the Physical Vulnerability Evaluating Pipeline (PVEP) that can incorporate as many visual modal physical threats as possible for evaluating the physical robustness of VLAMs. The physical threats in PVEP specifically include Out-of-Distribution, Typography-based Visual Prompt, and Adversarial Patch Attacks. By comparing the performance fluctuations of VLAMs before and after being attacked, we provide generalizable \textbf{\textit{Analyses}} of how VLAMs respond to different physical security threats.

摘要: 最近，在多模式大语言模型(MLLM)的推动下，视觉语言动作模型(VLAM)被提出以在机器人操作任务的开放词汇场景中实现更好的性能。由于操作任务涉及与物理世界的直接交互，因此确保该任务执行过程中的健壮性和安全性始终是一个非常关键的问题。本文通过综合当前MLLMS的安全研究现状和物理世界中操纵任务的具体应用场景，对VLAMS在面临潜在物理威胁的情况下进行综合评估。具体地说，我们提出了物理脆弱性评估管道(PVEP)，它可以结合尽可能多的视觉通道物理威胁来评估VLAMS的物理健壮性。PVEP中的物理威胁具体包括分发外、基于排版的视觉提示和敌意补丁攻击。通过比较VLAM在受到攻击前后的性能波动，我们提供了VLAM如何应对不同的物理安全威胁的概括性的文本bf{文本{分析}}。



## **26. Defending against Reverse Preference Attacks is Difficult**

防御反向偏好攻击很困难 cs.LG

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.12914v1) [paper-pdf](http://arxiv.org/pdf/2409.12914v1)

**Authors**: Domenic Rosati, Giles Edkins, Harsh Raj, David Atanasov, Subhabrata Majumdar, Janarthanan Rajendran, Frank Rudzicz, Hassan Sajjad

**Abstract**: While there has been progress towards aligning Large Language Models (LLMs) with human values and ensuring safe behaviour at inference time, safety-aligned LLMs are known to be vulnerable to training-time attacks such as supervised fine-tuning (SFT) on harmful datasets. In this paper, we ask if LLMs are vulnerable to adversarial reinforcement learning. Motivated by this goal, we propose Reverse Preference Attacks (RPA), a class of attacks to make LLMs learn harmful behavior using adversarial reward during reinforcement learning from human feedback (RLHF). RPAs expose a critical safety gap of safety-aligned LLMs in RL settings: they easily explore the harmful text generation policies to optimize adversarial reward. To protect against RPAs, we explore a host of mitigation strategies. Leveraging Constrained Markov-Decision Processes, we adapt a number of mechanisms to defend against harmful fine-tuning attacks into the RL setting. Our experiments show that ``online" defenses that are based on the idea of minimizing the negative log likelihood of refusals -- with the defender having control of the loss function -- can effectively protect LLMs against RPAs. However, trying to defend model weights using ``offline" defenses that operate under the assumption that the defender has no control over the loss function are less effective in the face of RPAs. These findings show that attacks done using RL can be used to successfully undo safety alignment in open-weight LLMs and use them for malicious purposes.

摘要: 虽然在使大型语言模型(LLM)与人类价值观保持一致并确保推理时的安全行为方面取得了进展，但众所周知，与安全保持一致的LLM容易受到训练时的攻击，例如对有害数据集的监督微调(SFT)。在本文中，我们询问LLMS是否容易受到对抗性强化学习的影响。基于这一目标，我们提出了反向偏好攻击(RPA)，这是一类在人类反馈强化学习(RLHF)过程中利用对抗性奖励使LLM学习有害行为的攻击。RPA暴露了RL环境中安全对齐的LLM的一个关键安全漏洞：它们很容易探索有害文本生成策略，以优化对抗性奖励。为了防范RPA，我们探索了一系列缓解策略。利用受限的马尔可夫决策过程，我们采用了一些机制来防御RL设置中的有害微调攻击。我们的实验表明，基于最小化拒绝的负对数可能性的思想的“在线”防御--防御者控制着损失函数--可以有效地保护LLM免受RPA的攻击。然而，试图使用在防御者无法控制损失函数的假设下运行的“离线”防御来捍卫模型权重，在面对RPA时效果较差。这些发现表明，使用RL进行的攻击可以成功地取消开放重量LLM中的安全对齐，并将其用于恶意目的。



## **27. Enhancing Jailbreak Attacks with Diversity Guidance**

通过多元化指导加强越狱攻击 cs.CL

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2403.00292v2) [paper-pdf](http://arxiv.org/pdf/2403.00292v2)

**Authors**: Xu Zhang, Dinghao Jing, Xiaojun Wan

**Abstract**: As large language models(LLMs) become commonplace in practical applications, the security issues of LLMs have attracted societal concerns. Although extensive efforts have been made to safety alignment, LLMs remain vulnerable to jailbreak attacks. We find that redundant computations limit the performance of existing jailbreak attack methods. Therefore, we propose DPP-based Stochastic Trigger Searching (DSTS), a new optimization algorithm for jailbreak attacks. DSTS incorporates diversity guidance through techniques including stochastic gradient search and DPP selection during optimization. Detailed experiments and ablation studies demonstrate the effectiveness of the algorithm. Moreover, we use the proposed algorithm to compute the risk boundaries for different LLMs, providing a new perspective on LLM safety evaluation.

摘要: 随着大型语言模型（LLM）在实际应用中变得普遍，LLM的安全问题引起了社会的关注。尽管已在安全调整方面做出了广泛的努力，但LLM仍然容易受到越狱攻击。我们发现冗余计算限制了现有越狱攻击方法的性能。因此，我们提出了一种基于DPP的随机触发搜索（DSTS），这是一种新的越狱攻击优化算法。DSTS通过优化期间的随机梯度搜索和DPP选择等技术整合了多样性指导。详细的实验和消融研究证明了该算法的有效性。此外，我们使用提出的算法来计算不同LLM的风险边界，为LLM安全性评估提供了新的视角。



## **28. Typography Leads Semantic Diversifying: Amplifying Adversarial Transferability across Multimodal Large Language Models**

字体设计引领语义多元化：增强多模式大型语言模型之间的对抗性可移植性 cs.CV

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2405.20090v2) [paper-pdf](http://arxiv.org/pdf/2405.20090v2)

**Authors**: Hao Cheng, Erjia Xiao, Jiayan Yang, Jiahang Cao, Le Yang, Jize Zhang, Kaidi Xu, Jindong Gu, Renjing Xu

**Abstract**: Following the advent of the Artificial Intelligence (AI) era of large models, Multimodal Large Language Models (MLLMs) with the ability to understand cross-modal interactions between vision and text have attracted wide attention. Adversarial examples with human-imperceptible perturbation are shown to possess a characteristic known as transferability, which means that a perturbation generated by one model could also mislead another different model. Augmenting the diversity in input data is one of the most significant methods for enhancing adversarial transferability. This method has been certified as a way to significantly enlarge the threat impact under black-box conditions. Research works also demonstrate that MLLMs can be exploited to generate adversarial examples in the white-box scenario. However, the adversarial transferability of such perturbations is quite limited, failing to achieve effective black-box attacks across different models. In this paper, we propose the Typographic-based Semantic Transfer Attack (TSTA), which is inspired by: (1) MLLMs tend to process semantic-level information; (2) Typographic Attack could effectively distract the visual information captured by MLLMs. In the scenarios of Harmful Word Insertion and Important Information Protection, our TSTA demonstrates superior performance.

摘要: 随着大模型人工智能时代的到来，能够理解视觉和文本之间跨通道交互的多通道大语言模型引起了人们的广泛关注。具有人类不可察觉的扰动的对抗性例子具有被称为可转移性的特征，这意味着一个模型产生的扰动也可能误导另一个不同的模型。增加输入数据的多样性是增强对抗性转移的最重要的方法之一。这种方法已被证明是一种在黑箱条件下显著扩大威胁影响的方法。研究工作还表明，在白盒情况下，MLLMS可以被用来生成对抗性示例。然而，此类扰动的对抗性可转移性相当有限，无法实现跨不同模型的有效黑盒攻击。本文提出了基于排版的语义传输攻击(TSTA)，其灵感来自：(1)MLLMS倾向于处理语义级的信息；(2)排版攻击可以有效地分散MLLMS捕获的视觉信息。在有害词语插入和重要信息保护的场景中，我们的TSTA表现出了卓越的性能。



## **29. Unveiling Typographic Deceptions: Insights of the Typographic Vulnerability in Large Vision-Language Model**

揭开印刷欺骗：大型视觉语言模型中印刷漏洞的洞察 cs.CV

This paper is accepted by ECCV 2024

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2402.19150v3) [paper-pdf](http://arxiv.org/pdf/2402.19150v3)

**Authors**: Hao Cheng, Erjia Xiao, Jindong Gu, Le Yang, Jinhao Duan, Jize Zhang, Jiahang Cao, Kaidi Xu, Renjing Xu

**Abstract**: Large Vision-Language Models (LVLMs) rely on vision encoders and Large Language Models (LLMs) to exhibit remarkable capabilities on various multi-modal tasks in the joint space of vision and language. However, typographic attacks, which disrupt Vision-Language Models (VLMs) such as Contrastive Language-Image Pretraining (CLIP), have also been expected to be a security threat to LVLMs. Firstly, we verify typographic attacks on current well-known commercial and open-source LVLMs and uncover the widespread existence of this threat. Secondly, to better assess this vulnerability, we propose the most comprehensive and largest-scale Typographic Dataset to date. The Typographic Dataset not only considers the evaluation of typographic attacks under various multi-modal tasks but also evaluates the effects of typographic attacks, influenced by texts generated with diverse factors. Based on the evaluation results, we investigate the causes why typographic attacks impacting VLMs and LVLMs, leading to three highly insightful discoveries. During the process of further validating the rationality of our discoveries, we can reduce the performance degradation caused by typographic attacks from 42.07\% to 13.90\%. Code and Dataset are available in \href{https://github.com/ChaduCheng/TypoDeceptions}

摘要: 大视觉-语言模型依赖于视觉编码器和大语言模型，在视觉和语言的联合空间中表现出对各种多通道任务的卓越能力。然而，打乱视觉语言模型(如对比语言图像预训练(CLIP))的排版攻击也被认为是对视觉语言模型的安全威胁。首先，我们验证了对当前著名的商业和开源LVLM的排版攻击，并揭示了这种威胁的广泛存在。其次，为了更好地评估这个漏洞，我们提出了迄今为止最全面和最大规模的排版数据集。排版数据集不仅考虑了各种多模式任务下排版攻击的评估，而且还评估了受多种因素生成的文本影响的排版攻击的效果。基于评估结果，我们调查了排版攻击影响VLM和LVLM的原因，导致了三个非常有洞察力的发现。在进一步验证我们发现的合理性的过程中，我们可以将排版攻击造成的性能降级从42.07%降低到13.90%。代码和数据集在\href{https://github.com/ChaduCheng/TypoDeceptions}中可用



## **30. LLM-Powered Text Simulation Attack Against ID-Free Recommender Systems**

针对无ID推荐系统的LLM支持文本模拟攻击 cs.IR

12 pages

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.11690v2) [paper-pdf](http://arxiv.org/pdf/2409.11690v2)

**Authors**: Zongwei Wang, Min Gao, Junliang Yu, Xinyi Gao, Quoc Viet Hung Nguyen, Shazia Sadiq, Hongzhi Yin

**Abstract**: The ID-free recommendation paradigm has been proposed to address the limitation that traditional recommender systems struggle to model cold-start users or items with new IDs. Despite its effectiveness, this study uncovers that ID-free recommender systems are vulnerable to the proposed Text Simulation attack (TextSimu) which aims to promote specific target items. As a novel type of text poisoning attack, TextSimu exploits large language models (LLM) to alter the textual information of target items by simulating the characteristics of popular items. It operates effectively in both black-box and white-box settings, utilizing two key components: a unified popularity extraction module, which captures the essential characteristics of popular items, and an N-persona consistency simulation strategy, which creates multiple personas to collaboratively synthesize refined promotional textual descriptions for target items by simulating the popular items. To withstand TextSimu-like attacks, we further explore the detection approach for identifying LLM-generated promotional text. Extensive experiments conducted on three datasets demonstrate that TextSimu poses a more significant threat than existing poisoning attacks, while our defense method can detect malicious text of target items generated by TextSimu. By identifying the vulnerability, we aim to advance the development of more robust ID-free recommender systems.

摘要: 无ID推荐范型的提出是为了解决传统推荐系统难以对冷启动用户或具有新ID的项目建模的局限性。研究发现，尽管无ID推荐系统是有效的，但它仍然容易受到针对特定目标条目的文本模拟攻击(TextSimu)的攻击。作为一种新型的文本中毒攻击，TextSimu利用大语言模型(LLM)通过模拟热门条目的特征来改变目标条目的文本信息。它在黑盒和白盒环境下都能有效地运行，使用了两个关键组件：一个是统一的人气提取模块，它捕获了热门物品的本质特征；另一个是N-角色一致性模拟策略，它创建多个人物角色，通过模拟热门物品来协作合成对目标物品的精细化促销文本描述。为了抵抗类似TextSimu的攻击，我们进一步探索了识别LLM生成的促销文本的检测方法。在三个数据集上进行的大量实验表明，TextSimu比现有的中毒攻击具有更大的威胁，而我们的防御方法可以检测到TextSimu生成的目标项的恶意文本。通过识别漏洞，我们的目标是推进更健壮的无ID推荐系统的开发。



## **31. AirGapAgent: Protecting Privacy-Conscious Conversational Agents**

AirGapAgent：保护有隐私意识的对话代理人 cs.CR

at CCS'24

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2405.05175v2) [paper-pdf](http://arxiv.org/pdf/2405.05175v2)

**Authors**: Eugene Bagdasarian, Ren Yi, Sahra Ghalebikesabi, Peter Kairouz, Marco Gruteser, Sewoong Oh, Borja Balle, Daniel Ramage

**Abstract**: The growing use of large language model (LLM)-based conversational agents to manage sensitive user data raises significant privacy concerns. While these agents excel at understanding and acting on context, this capability can be exploited by malicious actors. We introduce a novel threat model where adversarial third-party apps manipulate the context of interaction to trick LLM-based agents into revealing private information not relevant to the task at hand.   Grounded in the framework of contextual integrity, we introduce AirGapAgent, a privacy-conscious agent designed to prevent unintended data leakage by restricting the agent's access to only the data necessary for a specific task. Extensive experiments using Gemini, GPT, and Mistral models as agents validate our approach's effectiveness in mitigating this form of context hijacking while maintaining core agent functionality. For example, we show that a single-query context hijacking attack on a Gemini Ultra agent reduces its ability to protect user data from 94% to 45%, while an AirGapAgent achieves 97% protection, rendering the same attack ineffective.

摘要: 越来越多地使用基于大型语言模型(LLM)的会话代理来管理敏感用户数据，这引发了严重的隐私问题。虽然这些代理擅长理解上下文并根据上下文执行操作，但这种能力可能会被恶意行为者利用。我们引入了一种新的威胁模型，在该模型中，敌意的第三方应用程序操纵交互的上下文，以欺骗基于LLM的代理泄露与手头任务无关的私人信息。基于上下文完整性的框架，我们引入了AirGapAgent，这是一个具有隐私意识的代理，旨在通过限制代理仅访问特定任务所需的数据来防止意外的数据泄露。使用Gemini、GPT和Mistral模型作为代理的大量实验验证了我们的方法在保持核心代理功能的同时缓解这种形式的上下文劫持的有效性。例如，我们表明，对Gemini Ultra代理的单查询上下文劫持攻击将其保护用户数据的能力从94%降低到45%，而AirGapAgent实现了97%的保护，使得相同的攻击无效。



## **32. Hierarchical LLMs In-the-loop Optimization for Real-time Multi-Robot Target Tracking under Unknown Hazards**

未知危险下实时多机器人目标跟踪的分层LLM在环优化 cs.RO

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2409.12274v1) [paper-pdf](http://arxiv.org/pdf/2409.12274v1)

**Authors**: Yuwei Wu, Yuezhan Tao, Peihan Li, Guangyao Shi, Gaurav S. Sukhatmem, Vijay Kumar, Lifeng Zhou

**Abstract**: In this paper, we propose a hierarchical Large Language Models (LLMs) in-the-loop optimization framework for real-time multi-robot task allocation and target tracking in an unknown hazardous environment subject to sensing and communication attacks. We formulate multi-robot coordination for tracking tasks as a bi-level optimization problem, with LLMs to reason about potential hazards in the environment and the status of the robot team and modify both the inner and outer levels of the optimization. The inner LLM adjusts parameters to prioritize various objectives, including performance, safety, and energy efficiency, while the outer LLM handles online variable completion for team reconfiguration. This hierarchical approach enables real-time adjustments to the robots' behavior. Additionally, a human supervisor can offer broad guidance and assessments to address unexpected dangers, model mismatches, and performance issues arising from local minima. We validate our proposed framework in both simulation and real-world experiments with comprehensive evaluations, which provide the potential for safe LLM integration for multi-robot problems.

摘要: 提出了一种层次化大语言模型(LLMS)在环优化框架，用于多机器人在未知危险环境中的实时任务分配和目标跟踪。我们将多机器人协调跟踪任务描述为一个双层优化问题，使用LLMS对环境中的潜在危害和机器人团队的状态进行推理，并对优化的内部和外部进行修改。内部LLM调整参数以确定各种目标的优先顺序，包括性能、安全和能源效率，而外部LLM处理团队重新配置的在线变量完成。这种层次化的方法能够实时调整机器人的行为。此外，人类主管可以提供广泛的指导和评估，以解决意外危险、模型不匹配以及由局部最小值引起的性能问题。我们在仿真和真实世界的实验中验证了我们的框架，并进行了全面的评估，这为安全的LLM集成提供了解决多机器人问题的可能性。



## **33. From ML to LLM: Evaluating the Robustness of Phishing Webpage Detection Models against Adversarial Attacks**

从ML到LLM：评估网络钓鱼网页检测模型对抗对抗攻击的稳健性 cs.CR

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2407.20361v2) [paper-pdf](http://arxiv.org/pdf/2407.20361v2)

**Authors**: Aditya Kulkarni, Vivek Balachandran, Dinil Mon Divakaran, Tamal Das

**Abstract**: Phishing attacks attempt to deceive users into stealing sensitive information, posing a significant cybersecurity threat. Advances in machine learning (ML) and deep learning (DL) have led to the development of numerous phishing webpage detection solutions, but these models remain vulnerable to adversarial attacks. Evaluating their robustness against adversarial phishing webpages is essential. Existing tools contain datasets of pre-designed phishing webpages for a limited number of brands, and lack diversity in phishing features.   To address these challenges, we develop PhishOracle, a tool that generates adversarial phishing webpages by embedding diverse phishing features into legitimate webpages. We evaluate the robustness of two existing models, Stack model and Phishpedia, in classifying PhishOracle-generated adversarial phishing webpages. Additionally, we study a commercial large language model, Gemini Pro Vision, in the context of adversarial attacks. We conduct a user study to determine whether PhishOracle-generated adversarial phishing webpages deceive users. Our findings reveal that many PhishOracle-generated phishing webpages evade current phishing webpage detection models and deceive users, but Gemini Pro Vision is robust to the attack. We also develop the PhishOracle web app, allowing users to input a legitimate URL, select relevant phishing features and generate a corresponding phishing webpage. All resources are publicly available on GitHub.

摘要: 网络钓鱼攻击试图欺骗用户窃取敏感信息，构成重大的网络安全威胁。机器学习(ML)和深度学习(DL)的进步导致了许多钓鱼网页检测解决方案的发展，但这些模型仍然容易受到对手攻击。评估它们对敌意网络钓鱼网页的健壮性是至关重要的。现有工具包含为有限数量的品牌预先设计的钓鱼网页的数据集，并且在钓鱼功能方面缺乏多样性。为了应对这些挑战，我们开发了PhishOracle，这是一个通过在合法网页中嵌入不同的钓鱼功能来生成敌意钓鱼网页的工具。我们评估了现有的两种模型Stack模型和Phishpedia模型对PhishOracle生成的敌意钓鱼网页进行分类的稳健性。此外，我们研究了一个商业大型语言模型，Gemini Pro Vision，在对抗性攻击的背景下。我们进行了一项用户研究，以确定PhishOracle生成的敌意钓鱼网页是否欺骗了用户。我们的研究结果显示，许多PhishOracle生成的钓鱼网页逃避了当前的钓鱼网页检测模型并欺骗用户，但Gemini Pro Vision对攻击具有健壮性。我们还开发了PhishOracle Web应用程序，允许用户输入合法的URL，选择相关的网络钓鱼功能并生成相应的网络钓鱼网页。所有资源都在GitHub上公开提供。



## **34. Multitask Mayhem: Unveiling and Mitigating Safety Gaps in LLMs Fine-tuning**

多任务混乱：揭露和缓解LLC中的安全差距微调 cs.CL

19 pages, 11 figures

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2409.15361v1) [paper-pdf](http://arxiv.org/pdf/2409.15361v1)

**Authors**: Essa Jan, Nouar AlDahoul, Moiz Ali, Faizan Ahmad, Fareed Zaffar, Yasir Zaki

**Abstract**: Recent breakthroughs in Large Language Models (LLMs) have led to their adoption across a wide range of tasks, ranging from code generation to machine translation and sentiment analysis, etc. Red teaming/Safety alignment efforts show that fine-tuning models on benign (non-harmful) data could compromise safety. However, it remains unclear to what extent this phenomenon is influenced by different variables, including fine-tuning task, model calibrations, etc. This paper explores the task-wise safety degradation due to fine-tuning on downstream tasks such as summarization, code generation, translation, and classification across various calibration. Our results reveal that: 1) Fine-tuning LLMs for code generation and translation leads to the highest degradation in safety guardrails. 2) LLMs generally have weaker guardrails for translation and classification, with 73-92% of harmful prompts answered, across baseline and other calibrations, falling into one of two concern categories. 3) Current solutions, including guards and safety tuning datasets, lack cross-task robustness. To address these issues, we developed a new multitask safety dataset effectively reducing attack success rates across a range of tasks without compromising the model's overall helpfulness. Our work underscores the need for generalized alignment measures to ensure safer and more robust models.

摘要: 最近在大型语言模型(LLM)方面的突破导致它们在从代码生成到机器翻译和情绪分析等广泛任务中被采用。Red Teaming/Safe Align的努力表明，对良性(无害)数据进行微调模型可能会损害安全性。然而，这一现象在多大程度上受到不同变量的影响尚不清楚，这些变量包括微调任务、模型校准等。本文探讨了由于在各种校准中对下游任务(如摘要、代码生成、翻译和分类)进行微调而导致的任务级安全性降低。我们的结果表明：1)微调用于代码生成和翻译的LLM会导致最高的安全护栏退化。2)LLMS的翻译和分类障碍通常较弱，73%-92%的有害提示得到了回答，跨越基线和其他校准，属于两个令人担忧的类别之一。3)当前的解决方案，包括警卫和安全调整数据集，缺乏跨任务的健壮性。为了解决这些问题，我们开发了一个新的多任务安全数据集，在不影响模型整体帮助的情况下，有效地降低了一系列任务的攻击成功率。我们的工作强调了普遍的对齐措施的必要性，以确保更安全和更稳健的模型。



## **35. DrLLM: Prompt-Enhanced Distributed Denial-of-Service Resistance Method with Large Language Models**

DrLLM：具有大型语言模型的预算增强型分布式拒绝服务抵抗方法 cs.CR

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2409.10561v2) [paper-pdf](http://arxiv.org/pdf/2409.10561v2)

**Authors**: Zhenyu Yin, Shang Liu, Guangyuan Xu

**Abstract**: The increasing number of Distributed Denial of Service (DDoS) attacks poses a major threat to the Internet, highlighting the importance of DDoS mitigation. Most existing approaches require complex training methods to learn data features, which increases the complexity and generality of the application. In this paper, we propose DrLLM, which aims to mine anomalous traffic information in zero-shot scenarios through Large Language Models (LLMs). To bridge the gap between DrLLM and existing approaches, we embed the global and local information of the traffic data into the reasoning paradigm and design three modules, namely Knowledge Embedding, Token Embedding, and Progressive Role Reasoning, for data representation and reasoning. In addition we explore the generalization of prompt engineering in the cybersecurity domain to improve the classification capability of DrLLM. Our ablation experiments demonstrate the applicability of DrLLM in zero-shot scenarios and further demonstrate the potential of LLMs in the network domains. DrLLM implementation code has been open-sourced at https://github.com/liuup/DrLLM.

摘要: 越来越多的分布式拒绝服务(DDoS)攻击对互联网构成了重大威胁，突显了缓解DDoS的重要性。现有的大多数方法需要复杂的训练方法来学习数据特征，这增加了应用程序的复杂性和通用性。在本文中，我们提出了DrLLM，旨在通过大型语言模型(LLMS)挖掘零镜头场景中的异常交通信息。为了弥补DrLLM与已有方法之间的差距，我们将交通数据的全局和局部信息嵌入到推理范式中，并设计了三个模块，即知识嵌入、令牌嵌入和渐进角色推理，用于数据表示和推理。此外，我们还探索了快速工程在网络安全领域的泛化，以提高DrLLM的分类能力。我们的烧蚀实验证明了DrLLM在零射情况下的适用性，并进一步展示了LLM在网络领域的潜力。DrLLm实现代码已在https://github.com/liuup/DrLLM.上开源



## **36. Jailbreaking Large Language Models with Symbolic Mathematics**

用符号数学破解大型语言模型 cs.CR

**SubmitDate**: 2024-09-17    [abs](http://arxiv.org/abs/2409.11445v1) [paper-pdf](http://arxiv.org/pdf/2409.11445v1)

**Authors**: Emet Bethany, Mazal Bethany, Juan Arturo Nolazco Flores, Sumit Kumar Jha, Peyman Najafirad

**Abstract**: Recent advancements in AI safety have led to increased efforts in training and red-teaming large language models (LLMs) to mitigate unsafe content generation. However, these safety mechanisms may not be comprehensive, leaving potential vulnerabilities unexplored. This paper introduces MathPrompt, a novel jailbreaking technique that exploits LLMs' advanced capabilities in symbolic mathematics to bypass their safety mechanisms. By encoding harmful natural language prompts into mathematical problems, we demonstrate a critical vulnerability in current AI safety measures. Our experiments across 13 state-of-the-art LLMs reveal an average attack success rate of 73.6\%, highlighting the inability of existing safety training mechanisms to generalize to mathematically encoded inputs. Analysis of embedding vectors shows a substantial semantic shift between original and encoded prompts, helping explain the attack's success. This work emphasizes the importance of a holistic approach to AI safety, calling for expanded red-teaming efforts to develop robust safeguards across all potential input types and their associated risks.

摘要: 最近人工智能安全方面的进步导致在培训和红队大型语言模型(LLM)方面加大了努力，以减少不安全的内容生成。然而，这些安全机制可能不是全面的，留下了潜在的漏洞有待探索。本文介绍了MathPrompt，这是一种新的越狱技术，它利用LLMS在符号数学中的高级能力来绕过它们的安全机制。通过将有害的自然语言提示编码为数学问题，我们展示了当前人工智能安全措施中的一个严重漏洞。我们在13个最先进的LLM上进行的实验显示，平均攻击成功率为73.6\%，这突显了现有安全培训机制无法概括为数学编码的输入。对嵌入向量的分析显示，原始提示和编码提示之间存在实质性的语义转换，这有助于解释攻击的成功。这项工作强调了对人工智能安全采取整体方法的重要性，呼吁扩大红队努力，为所有潜在的投入类型及其相关风险制定强有力的保障措施。



## **37. Assessing biomedical knowledge robustness in large language models by query-efficient sampling attacks**

通过查询高效抽样攻击评估大型语言模型中生物医学知识的稳健性 cs.CL

28 pages incl. appendix, updated version

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2402.10527v2) [paper-pdf](http://arxiv.org/pdf/2402.10527v2)

**Authors**: R. Patrick Xian, Alex J. Lee, Satvik Lolla, Vincent Wang, Qiming Cui, Russell Ro, Reza Abbasi-Asl

**Abstract**: The increasing depth of parametric domain knowledge in large language models (LLMs) is fueling their rapid deployment in real-world applications. Understanding model vulnerabilities in high-stakes and knowledge-intensive tasks is essential for quantifying the trustworthiness of model predictions and regulating their use. The recent discovery of named entities as adversarial examples (i.e. adversarial entities) in natural language processing tasks raises questions about their potential impact on the knowledge robustness of pre-trained and finetuned LLMs in high-stakes and specialized domains. We examined the use of type-consistent entity substitution as a template for collecting adversarial entities for billion-parameter LLMs with biomedical knowledge. To this end, we developed an embedding-space attack based on powerscaled distance-weighted sampling to assess the robustness of their biomedical knowledge with a low query budget and controllable coverage. Our method has favorable query efficiency and scaling over alternative approaches based on random sampling and blackbox gradient-guided search, which we demonstrated for adversarial distractor generation in biomedical question answering. Subsequent failure mode analysis uncovered two regimes of adversarial entities on the attack surface with distinct characteristics and we showed that entity substitution attacks can manipulate token-wise Shapley value explanations, which become deceptive in this setting. Our approach complements standard evaluations for high-capacity models and the results highlight the brittleness of domain knowledge in LLMs.

摘要: 大型语言模型(LLM)中参数领域知识的不断深入推动了它们在现实世界应用程序中的快速部署。了解高风险和知识密集型任务中的模型脆弱性对于量化模型预测的可信度和规范其使用至关重要。最近在自然语言处理任务中发现了命名实体作为对抗性实例(即对抗性实体)，这引发了人们对高风险和专门领域中预先训练和精细调整的LLM知识稳健性的潜在影响的问题。我们研究了使用类型一致的实体替换作为收集具有生物医学知识的10亿参数LLM的对抗性实体的模板。为此，我们提出了一种基于加权距离加权抽样的嵌入空间攻击方法，以较低的查询预算和可控的覆盖率来评估他们的生物医学知识的稳健性。与基于随机抽样和黑盒梯度引导搜索的方法相比，我们的方法具有良好的查询效率和伸缩性，并在生物医学问答中的对抗性干扰项生成中得到了验证。随后的失效模式分析揭示了攻击面上具有不同特征的两种对抗实体的机制，我们表明实体替换攻击可以操纵令人信服的Shapley值解释，在这种情况下，这种解释变得具有欺骗性。我们的方法补充了对大容量模型的标准评估，结果突出了领域知识在LLMS中的脆性。



## **38. Security Attacks on LLM-based Code Completion Tools**

对基于LLM的代码完成工具的安全攻击 cs.CL

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2408.11006v2) [paper-pdf](http://arxiv.org/pdf/2408.11006v2)

**Authors**: Wen Cheng, Ke Sun, Xinyu Zhang, Wei Wang

**Abstract**: The rapid development of large language models (LLMs) has significantly advanced code completion capabilities, giving rise to a new generation of LLM-based Code Completion Tools (LCCTs). Unlike general-purpose LLMs, these tools possess unique workflows, integrating multiple information sources as input and prioritizing code suggestions over natural language interaction, which introduces distinct security challenges. Additionally, LCCTs often rely on proprietary code datasets for training, raising concerns about the potential exposure of sensitive data. This paper exploits these distinct characteristics of LCCTs to develop targeted attack methodologies on two critical security risks: jailbreaking and training data extraction attacks. Our experimental results expose significant vulnerabilities within LCCTs, including a 99.4% success rate in jailbreaking attacks on GitHub Copilot and a 46.3% success rate on Amazon Q. Furthermore, We successfully extracted sensitive user data from GitHub Copilot, including 54 real email addresses and 314 physical addresses associated with GitHub usernames. Our study also demonstrates that these code-based attack methods are effective against general-purpose LLMs, such as the GPT series, highlighting a broader security misalignment in the handling of code by modern LLMs. These findings underscore critical security challenges associated with LCCTs and suggest essential directions for strengthening their security frameworks. The example code and attack samples from our research are provided at https://github.com/Sensente/Security-Attacks-on-LCCTs.

摘要: 大型语言模型(LLM)的快速发展极大地提升了代码补全能力，催生了新一代基于LLM的代码补全工具(LCCT)。与通用的LLMS不同，这些工具拥有独特的工作流，将多个信息源集成为输入，并优先考虑代码建议而不是自然语言交互，这带来了明显的安全挑战。此外，LCCT经常依赖专有代码数据集进行培训，这引发了人们对敏感数据潜在暴露的担忧。针对越狱攻击和训练数据提取攻击这两个关键安全风险，本文利用LCCT的这些显著特点，提出了针对性的攻击方法。我们的实验结果暴露了LCCT中的重大漏洞，包括对GitHub Copilot的越狱攻击成功率为99.4%，对Amazon Q的成功率为46.3%。此外，我们成功地从GitHub Copilot中提取了敏感用户数据，包括与GitHub用户名关联的54个真实电子邮件地址和314个物理地址。我们的研究还表明，这些基于代码的攻击方法对通用LLM是有效的，例如GPT系列，突显了现代LLM在处理代码时存在更广泛的安全错位。这些调查结果强调了与土地利用、土地利用、土地退化和土地退化有关的重大安全挑战，并提出了加强其安全框架的基本方向。我们的研究提供了示例代码和攻击示例，请访问https://github.com/Sensente/Security-Attacks-on-LCCTs.



## **39. Do Membership Inference Attacks Work on Large Language Models?**

成员资格推理攻击对大型语言模型有效吗？ cs.CL

Accepted at Conference on Language Modeling (COLM), 2024

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2402.07841v2) [paper-pdf](http://arxiv.org/pdf/2402.07841v2)

**Authors**: Michael Duan, Anshuman Suri, Niloofar Mireshghallah, Sewon Min, Weijia Shi, Luke Zettlemoyer, Yulia Tsvetkov, Yejin Choi, David Evans, Hannaneh Hajishirzi

**Abstract**: Membership inference attacks (MIAs) attempt to predict whether a particular datapoint is a member of a target model's training data. Despite extensive research on traditional machine learning models, there has been limited work studying MIA on the pre-training data of large language models (LLMs). We perform a large-scale evaluation of MIAs over a suite of language models (LMs) trained on the Pile, ranging from 160M to 12B parameters. We find that MIAs barely outperform random guessing for most settings across varying LLM sizes and domains. Our further analyses reveal that this poor performance can be attributed to (1) the combination of a large dataset and few training iterations, and (2) an inherently fuzzy boundary between members and non-members. We identify specific settings where LLMs have been shown to be vulnerable to membership inference and show that the apparent success in such settings can be attributed to a distribution shift, such as when members and non-members are drawn from the seemingly identical domain but with different temporal ranges. We release our code and data as a unified benchmark package that includes all existing MIAs, supporting future work.

摘要: 成员关系推理攻击(MIA)试图预测特定数据点是否为目标模型训练数据的成员。尽管对传统的机器学习模型进行了广泛的研究，但在大型语言模型(LLMS)的训练前数据上研究MIA的工作有限。我们在堆上训练的一套语言模型(LMS)上对MIA进行了大规模评估，参数范围从160M到12B。我们发现，对于不同的LLM大小和域，对于大多数设置，MIA的性能仅略高于随机猜测。我们的进一步分析表明，这种糟糕的性能可以归因于(1)庞大的数据集和很少的训练迭代的组合，以及(2)成员和非成员之间固有的模糊边界。我们确定了LLM被证明易受成员关系推断影响的特定设置，并表明此类设置的明显成功可以归因于分布变化，例如当成员和非成员来自看似相同的域但具有不同的时间范围时。我们将我们的代码和数据作为一个统一的基准程序包发布，其中包括所有现有的MIA，以支持未来的工作。



## **40. Robust Privacy Amidst Innovation with Large Language Models Through a Critical Assessment of the Risks**

通过对风险的批判性评估，在大型语言模型的创新中实现稳健的隐私 cs.CL

13 pages, 4 figures, 1 table, 1 supplementary, under review

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2407.16166v2) [paper-pdf](http://arxiv.org/pdf/2407.16166v2)

**Authors**: Yao-Shun Chuang, Atiquer Rahman Sarkar, Yu-Chun Hsu, Noman Mohammed, Xiaoqian Jiang

**Abstract**: This study examines integrating EHRs and NLP with large language models (LLMs) to improve healthcare data management and patient care. It focuses on using advanced models to create secure, HIPAA-compliant synthetic patient notes for biomedical research. The study used de-identified and re-identified MIMIC III datasets with GPT-3.5, GPT-4, and Mistral 7B to generate synthetic notes. Text generation employed templates and keyword extraction for contextually relevant notes, with one-shot generation for comparison. Privacy assessment checked PHI occurrence, while text utility was tested using an ICD-9 coding task. Text quality was evaluated with ROUGE and cosine similarity metrics to measure semantic similarity with source notes. Analysis of PHI occurrence and text utility via the ICD-9 coding task showed that the keyword-based method had low risk and good performance. One-shot generation showed the highest PHI exposure and PHI co-occurrence, especially in geographic location and date categories. The Normalized One-shot method achieved the highest classification accuracy. Privacy analysis revealed a critical balance between data utility and privacy protection, influencing future data use and sharing. Re-identified data consistently outperformed de-identified data. This study demonstrates the effectiveness of keyword-based methods in generating privacy-protecting synthetic clinical notes that retain data usability, potentially transforming clinical data-sharing practices. The superior performance of re-identified over de-identified data suggests a shift towards methods that enhance utility and privacy by using dummy PHIs to perplex privacy attacks.

摘要: 这项研究考察了将EHR和NLP与大型语言模型(LLM)相结合，以改进医疗数据管理和患者护理。它专注于使用高级模型创建安全的、符合HIPAA标准的合成患者笔记，用于生物医学研究。这项研究使用了GPT-3.5、GPT-4和西风7B的去识别和重新识别的MIMIC III数据集来生成合成音符。文本生成使用模板和上下文相关笔记的关键字提取，并使用一次生成进行比较。隐私评估检查了PHI的发生，而文本实用程序则使用ICD-9编码任务进行了测试。使用Rouge和Cosine相似性度量来评价文本质量，以衡量与源注释的语义相似性。通过ICD-9编码任务对PHI发生情况和文本效用的分析表明，基于关键字的方法风险低，性能好。一次发生显示出最高的PHI暴露和PHI共同出现，特别是在地理位置和日期类别。归一化一次法达到了最高的分类精度。隐私分析揭示了数据效用和隐私保护之间的关键平衡，影响了未来数据的使用和共享。重新识别的数据始终优于未识别的数据。这项研究证明了基于关键字的方法在生成保护隐私的合成临床笔记方面的有效性，这些合成笔记保留了数据的可用性，潜在地改变了临床数据共享的做法。重新识别的数据优于未识别的数据，这表明通过使用虚拟PHI来困扰隐私攻击来增强实用性和隐私的方法发生了转变。



## **41. LLM Whisperer: An Inconspicuous Attack to Bias LLM Responses**

LLM Whisperer：对LLM偏见回应的不起眼攻击 cs.CR

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2406.04755v2) [paper-pdf](http://arxiv.org/pdf/2406.04755v2)

**Authors**: Weiran Lin, Anna Gerchanovsky, Omer Akgul, Lujo Bauer, Matt Fredrikson, Zifan Wang

**Abstract**: Writing effective prompts for large language models (LLM) can be unintuitive and burdensome. In response, services that optimize or suggest prompts have emerged. While such services can reduce user effort, they also introduce a risk: the prompt provider can subtly manipulate prompts to produce heavily biased LLM responses. In this work, we show that subtle synonym replacements in prompts can increase the likelihood (by a difference up to 78%) that LLMs mention a target concept (e.g., a brand, political party, nation). We substantiate our observations through a user study, showing our adversarially perturbed prompts 1) are indistinguishable from unaltered prompts by humans, 2) push LLMs to recommend target concepts more often, and 3) make users more likely to notice target concepts, all without arousing suspicion. The practicality of this attack has the potential to undermine user autonomy. Among other measures, we recommend implementing warnings against using prompts from untrusted parties.

摘要: 为大型语言模型(LLM)编写有效的提示可能是不直观和繁琐的。作为回应，优化或建议提示的服务应运而生。虽然这类服务可以减少用户的工作，但它们也带来了风险：提示提供商可能会巧妙地操纵提示，以产生严重偏见的LLM响应。在这项工作中，我们表明，提示中微妙的同义词替换可以增加LLMS提到目标概念(例如，品牌、政党、国家)的可能性(差异高达78%)。我们通过一项用户研究证实了我们的观察结果，表明我们受到敌意干扰的提示1)与人类未更改的提示无法区分，2)推动LLMS更频繁地推荐目标概念，3)使用户更有可能注意到目标概念，所有这些都不会引起怀疑。这种攻击的实用性有可能破坏用户的自主性。在其他措施中，我们建议实施警告，以防止使用来自不受信任方的提示。



## **42. LLM Honeypot: Leveraging Large Language Models as Advanced Interactive Honeypot Systems**

LLM蜜罐：利用大型语言模型作为高级交互式蜜罐系统 cs.CR

6 pages, 5 figures

**SubmitDate**: 2024-09-15    [abs](http://arxiv.org/abs/2409.08234v2) [paper-pdf](http://arxiv.org/pdf/2409.08234v2)

**Authors**: Hakan T. Otal, M. Abdullah Canbaz

**Abstract**: The rapid evolution of cyber threats necessitates innovative solutions for detecting and analyzing malicious activity. Honeypots, which are decoy systems designed to lure and interact with attackers, have emerged as a critical component in cybersecurity. In this paper, we present a novel approach to creating realistic and interactive honeypot systems using Large Language Models (LLMs). By fine-tuning a pre-trained open-source language model on a diverse dataset of attacker-generated commands and responses, we developed a honeypot capable of sophisticated engagement with attackers. Our methodology involved several key steps: data collection and processing, prompt engineering, model selection, and supervised fine-tuning to optimize the model's performance. Evaluation through similarity metrics and live deployment demonstrated that our approach effectively generates accurate and informative responses. The results highlight the potential of LLMs to revolutionize honeypot technology, providing cybersecurity professionals with a powerful tool to detect and analyze malicious activity, thereby enhancing overall security infrastructure.

摘要: 网络威胁的快速演变需要检测和分析恶意活动的创新解决方案。蜜罐是一种诱骗系统，旨在引诱攻击者并与之互动，它已经成为网络安全的一个关键组成部分。在这篇文章中，我们提出了一种新的方法来创建真实的和交互式的蜜罐系统使用大语言模型(LLMS)。通过在攻击者生成的命令和响应的不同数据集上微调预先训练的开源语言模型，我们开发了一个能够与攻击者进行复杂交互的蜜罐。我们的方法包括几个关键步骤：数据收集和处理、快速工程、模型选择和监督微调以优化模型的性能。通过相似性度量和现场部署进行的评估表明，我们的方法有效地生成了准确和信息丰富的响应。这些结果突出了LLMS对蜜罐技术进行革命性变革的潜力，为网络安全专业人员提供了检测和分析恶意活动的强大工具，从而增强了整体安全基础设施。



## **43. Detection Made Easy: Potentials of Large Language Models for Solidity Vulnerabilities**

检测变得简单：大型语言模型针对Solidity漏洞的潜力 cs.CR

**SubmitDate**: 2024-09-15    [abs](http://arxiv.org/abs/2409.10574v1) [paper-pdf](http://arxiv.org/pdf/2409.10574v1)

**Authors**: Md Tauseef Alam, Raju Halder, Abyayananda Maiti

**Abstract**: The large-scale deployment of Solidity smart contracts on the Ethereum mainnet has increasingly attracted financially-motivated attackers in recent years. A few now-infamous attacks in Ethereum's history includes DAO attack in 2016 (50 million dollars lost), Parity Wallet hack in 2017 (146 million dollars locked), Beautychain's token BEC in 2018 (900 million dollars market value fell to 0), and NFT gaming blockchain breach in 2022 ($600 million in Ether stolen). This paper presents a comprehensive investigation of the use of large language models (LLMs) and their capabilities in detecting OWASP Top Ten vulnerabilities in Solidity. We introduce a novel, class-balanced, structured, and labeled dataset named VulSmart, which we use to benchmark and compare the performance of open-source LLMs such as CodeLlama, Llama2, CodeT5 and Falcon, alongside closed-source models like GPT-3.5 Turbo and GPT-4o Mini. Our proposed SmartVD framework is rigorously tested against these models through extensive automated and manual evaluations, utilizing BLEU and ROUGE metrics to assess the effectiveness of vulnerability detection in smart contracts. We also explore three distinct prompting strategies-zero-shot, few-shot, and chain-of-thought-to evaluate the multi-class classification and generative capabilities of the SmartVD framework. Our findings reveal that SmartVD outperforms its open-source counterparts and even exceeds the performance of closed-source base models like GPT-3.5 and GPT-4 Mini. After fine-tuning, the closed-source models, GPT-3.5 Turbo and GPT-4o Mini, achieved remarkable performance with 99% accuracy in detecting vulnerabilities, 94% in identifying their types, and 98% in determining severity. Notably, SmartVD performs best with the `chain-of-thought' prompting technique, whereas the fine-tuned closed-source models excel with the `zero-shot' prompting approach.

摘要: 近年来，在以太主机上大规模部署稳健智能合约，越来越多地吸引了出于经济动机的攻击者。Etherum历史上现在臭名昭著的攻击包括2016年的DAO攻击(损失了5000万美元)，2017年的平价钱包黑客攻击(锁定了1.46亿美元)，2018年BeautyChain的Token BEC(9亿美元的市值跌至0)，以及2022年的NFT游戏区块链漏洞(6亿美元的以太被盗)。本文对大型语言模型(LLM)的使用及其检测OWASP的十大漏洞的能力进行了全面的调查。我们介绍了一个新的，类平衡的，结构化的，带标签的数据集VulSmart，我们使用它来基准测试和比较开源LLM的性能，如CodeLlama，Llama2，CodeT5和Falcon，以及闭源模型如GPT-3.5 Turbo和GPT-40 Mini。我们建议的SmartVD框架通过广泛的自动和手动评估，针对这些模型进行了严格的测试，使用BLEU和Rouge指标来评估智能合同中漏洞检测的有效性。我们还探索了三种不同的提示策略-零射、少射和思维链-来评估SmartVD框架的多类分类和生成能力。我们的发现表明，SmartVD的表现优于开源同行，甚至超过了GPT-3.5和GPT-4 Mini等封闭源代码基础机型的性能。经过微调后，闭源模型GPT-3.5 Turbo和GPT-40 Mini取得了显著的性能，检测漏洞的准确率为99%，识别漏洞类型的准确率为94%，确定严重性的准确率为98%。值得注意的是，SmartVD最好地使用了“思维链”提示技术，而经过微调的闭源模型则使用了“零镜头”提示方法。



## **44. ContractTinker: LLM-Empowered Vulnerability Repair for Real-World Smart Contracts**

ContractTinker：针对现实世界智能合同的LLC授权漏洞修复 cs.SE

4 pages, and to be accepted in ASE2024

**SubmitDate**: 2024-09-15    [abs](http://arxiv.org/abs/2409.09661v1) [paper-pdf](http://arxiv.org/pdf/2409.09661v1)

**Authors**: Che Wang, Jiashuo Zhang, Jianbo Gao, Libin Xia, Zhi Guan, Zhong Chen

**Abstract**: Smart contracts are susceptible to being exploited by attackers, especially when facing real-world vulnerabilities. To mitigate this risk, developers often rely on third-party audit services to identify potential vulnerabilities before project deployment. Nevertheless, repairing the identified vulnerabilities is still complex and labor-intensive, particularly for developers lacking security expertise. Moreover, existing pattern-based repair tools mostly fail to address real-world vulnerabilities due to their lack of high-level semantic understanding. To fill this gap, we propose ContractTinker, a Large Language Models (LLMs)-empowered tool for real-world vulnerability repair. The key insight is our adoption of the Chain-of-Thought approach to break down the entire generation task into sub-tasks. Additionally, to reduce hallucination, we integrate program static analysis to guide the LLM. We evaluate ContractTinker on 48 high-risk vulnerabilities. The experimental results show that among the patches generated by ContractTinker, 23 (48%) are valid patches that fix the vulnerabilities, while 10 (21%) require only minor modifications. A video of ContractTinker is available at https://youtu.be/HWFVi-YHcPE.

摘要: 智能合同很容易被攻击者利用，特别是在面临现实世界的漏洞时。为了降低这种风险，开发人员通常依赖第三方审计服务在项目部署之前识别潜在的漏洞。尽管如此，修复已确定的漏洞仍然是复杂和劳动密集型的，特别是对于缺乏安全专业知识的开发人员。此外，现有的基于模式的修复工具由于缺乏高层语义理解，大多无法解决现实世界的漏洞。为了填补这一空白，我们提出了ContractTinker，这是一个大型语言模型(LLMS)授权的工具，用于修复现实世界的漏洞。关键的洞察力是我们采用了思想链的方法，将整个生成任务分解为子任务。此外，为了减少幻觉，我们集成了程序静态分析来指导LLM。我们对ContractTinker的48个高危漏洞进行了评估。实验结果表明，在ContractTinker生成的补丁中，有23个(48%)是修复漏洞的有效补丁，而10个(21%)只需要进行少量修改。有关ContractTinker的视频，请访问https://youtu.be/HWFVi-YHcPE.。



## **45. Towards Resilient and Efficient LLMs: A Comparative Study of Efficiency, Performance, and Adversarial Robustness**

迈向弹性和高效的法学硕士：效率、绩效和对抗稳健性的比较研究 cs.CL

**SubmitDate**: 2024-09-14    [abs](http://arxiv.org/abs/2408.04585v3) [paper-pdf](http://arxiv.org/pdf/2408.04585v3)

**Authors**: Xiaojing Fan, Chunliang Tao

**Abstract**: With the increasing demand for practical applications of Large Language Models (LLMs), many attention-efficient models have been developed to balance performance and computational cost. However, the adversarial robustness of these models remains under-explored. In this work, we design a framework to investigate the trade-off between efficiency, performance, and adversarial robustness of LLMs and conduct extensive experiments on three prominent models with varying levels of complexity and efficiency -- Transformer++, Gated Linear Attention (GLA) Transformer, and MatMul-Free LM -- utilizing the GLUE and AdvGLUE datasets. The AdvGLUE dataset extends the GLUE dataset with adversarial samples designed to challenge model robustness. Our results show that while the GLA Transformer and MatMul-Free LM achieve slightly lower accuracy on GLUE tasks, they demonstrate higher efficiency and either superior or comparative robustness on AdvGLUE tasks compared to Transformer++ across different attack levels. These findings highlight the potential of simplified architectures to achieve a compelling balance between efficiency, performance, and adversarial robustness, offering valuable insights for applications where resource constraints and resilience to adversarial attacks are critical.

摘要: 随着大型语言模型的实际应用需求的增加，人们已经开发了许多注意力高效的模型来平衡性能和计算成本。然而，这些模型的对抗性稳健性仍然没有得到充分的研究。在这项工作中，我们设计了一个框架来研究LLMS的效率、性能和对抗健壮性之间的权衡，并利用GLUE和AdvGLUE数据集在三个不同复杂度和效率的重要模型上进行了广泛的实验--Transformer++、门控线性注意(GLA)Transformer和MatMul-Free LM。AdvGLUE数据集使用旨在挑战模型稳健性的对抗性样本扩展了GLUE数据集。我们的结果表明，虽然GLA Transformer和MatMul-Free LM在粘合任务上的准确率略低，但在不同攻击级别上，它们在AdvGLUE任务上表现出比Transformer++更高的效率和更好的健壮性或相对较高的稳健性。这些发现突出了简化体系结构在效率、性能和对手攻击健壮性之间实现引人注目的平衡的潜力，为资源约束和对抗攻击的弹性至关重要的应用程序提供了宝贵的见解。



## **46. Tamper-Resistant Safeguards for Open-Weight LLMs**

开放重量LLM的防篡改保障措施 cs.LG

Website: https://www.tamper-resistant-safeguards.com

**SubmitDate**: 2024-09-14    [abs](http://arxiv.org/abs/2408.00761v3) [paper-pdf](http://arxiv.org/pdf/2408.00761v3)

**Authors**: Rishub Tamirisa, Bhrugu Bharathi, Long Phan, Andy Zhou, Alice Gatti, Tarun Suresh, Maxwell Lin, Justin Wang, Rowan Wang, Ron Arel, Andy Zou, Dawn Song, Bo Li, Dan Hendrycks, Mantas Mazeika

**Abstract**: Rapid advances in the capabilities of large language models (LLMs) have raised widespread concerns regarding their potential for malicious use. Open-weight LLMs present unique challenges, as existing safeguards lack robustness to tampering attacks that modify model weights. For example, recent works have demonstrated that refusal and unlearning safeguards can be trivially removed with a few steps of fine-tuning. These vulnerabilities necessitate new approaches for enabling the safe release of open-weight LLMs. We develop a method, called TAR, for building tamper-resistant safeguards into open-weight LLMs such that adversaries cannot remove the safeguards even after thousands of steps of fine-tuning. In extensive evaluations and red teaming analyses, we find that our method greatly improves tamper-resistance while preserving benign capabilities. Our results demonstrate that tamper-resistance is a tractable problem, opening up a promising new avenue to improve the safety and security of open-weight LLMs.

摘要: 大型语言模型(LLM)功能的快速发展引起了人们对其潜在恶意使用的广泛关注。开放重量LLM提出了独特的挑战，因为现有的保障措施缺乏对篡改模型权重的篡改攻击的稳健性。例如，最近的研究表明，通过几个步骤的微调，就可以很容易地消除拒绝和遗忘的保障措施。这些漏洞需要新的方法来实现安全释放未加重量的低密度脂蛋白。我们开发了一种名为TAR的方法，用于在开放重量的LLM中构建防篡改保护措施，以便对手即使在数千个步骤的微调之后也无法移除这些保护措施。在广泛的评估和红团队分析中，我们发现我们的方法在保持良性性能的同时大大提高了防篡改能力。我们的结果表明，防篡改是一个容易解决的问题，为提高开重LLMS的安全性开辟了一条很有前途的新途径。



## **47. Safeguarding AI Agents: Developing and Analyzing Safety Architectures**

保护人工智能代理：开发和分析安全架构 cs.CR

**SubmitDate**: 2024-09-13    [abs](http://arxiv.org/abs/2409.03793v2) [paper-pdf](http://arxiv.org/pdf/2409.03793v2)

**Authors**: Ishaan Domkundwar, Mukunda N S, Ishaan Bhola

**Abstract**: AI agents, specifically powered by large language models, have demonstrated exceptional capabilities in various applications where precision and efficacy are necessary. However, these agents come with inherent risks, including the potential for unsafe or biased actions, vulnerability to adversarial attacks, lack of transparency, and tendency to generate hallucinations. As AI agents become more prevalent in critical sectors of the industry, the implementation of effective safety protocols becomes increasingly important. This paper addresses the critical need for safety measures in AI systems, especially ones that collaborate with human teams. We propose and evaluate three frameworks to enhance safety protocols in AI agent systems: an LLM-powered input-output filter, a safety agent integrated within the system, and a hierarchical delegation-based system with embedded safety checks. Our methodology involves implementing these frameworks and testing them against a set of unsafe agentic use cases, providing a comprehensive evaluation of their effectiveness in mitigating risks associated with AI agent deployment. We conclude that these frameworks can significantly strengthen the safety and security of AI agent systems, minimizing potential harmful actions or outputs. Our work contributes to the ongoing effort to create safe and reliable AI applications, particularly in automated operations, and provides a foundation for developing robust guardrails to ensure the responsible use of AI agents in real-world applications.

摘要: 人工智能代理，特别是由大型语言模型驱动的，在需要精确度和效率的各种应用中展示了非凡的能力。然而，这些代理伴随着固有的风险，包括潜在的不安全或有偏见的行动，易受对手攻击，缺乏透明度，以及产生幻觉的倾向。随着人工智能代理在该行业的关键部门变得越来越普遍，实施有效的安全协议变得越来越重要。本文讨论了人工智能系统中安全措施的迫切需要，特别是与人类团队协作的系统。我们提出并评估了三个框架来增强AI代理系统中的安全协议：LLM驱动的输入输出过滤器、集成在系统中的安全代理以及嵌入安全检查的基于分级委托的系统。我们的方法涉及实现这些框架并针对一组不安全的代理用例对它们进行测试，提供对它们在降低与AI代理部署相关的风险方面的有效性的全面评估。我们的结论是，这些框架可以显著加强AI代理系统的安全性和安全性，将潜在的有害行为或输出降至最低。我们的工作有助于持续努力创建安全可靠的人工智能应用程序，特别是在自动化操作中，并为开发强大的护栏提供基础，以确保在现实世界的应用程序中负责任地使用人工智能代理。



## **48. h4rm3l: A Dynamic Benchmark of Composable Jailbreak Attacks for LLM Safety Assessment**

h4 rm3l：LLM安全评估的可组合越狱攻击的动态基准 cs.CR

**SubmitDate**: 2024-09-13    [abs](http://arxiv.org/abs/2408.04811v2) [paper-pdf](http://arxiv.org/pdf/2408.04811v2)

**Authors**: Moussa Koulako Bala Doumbouya, Ananjan Nandi, Gabriel Poesia, Davide Ghilardi, Anna Goldie, Federico Bianchi, Dan Jurafsky, Christopher D. Manning

**Abstract**: The safety of Large Language Models (LLMs) remains a critical concern due to a lack of adequate benchmarks for systematically evaluating their ability to resist generating harmful content. Previous efforts towards automated red teaming involve static or templated sets of illicit requests and adversarial prompts which have limited utility given jailbreak attacks' evolving and composable nature. We propose a novel dynamic benchmark of composable jailbreak attacks to move beyond static datasets and taxonomies of attacks and harms. Our approach consists of three components collectively called h4rm3l: (1) a domain-specific language that formally expresses jailbreak attacks as compositions of parameterized prompt transformation primitives, (2) bandit-based few-shot program synthesis algorithms that generate novel attacks optimized to penetrate the safety filters of a target black box LLM, and (3) open-source automated red-teaming software employing the previous two components. We use h4rm3l to generate a dataset of 2656 successful novel jailbreak attacks targeting 6 state-of-the-art (SOTA) open-source and proprietary LLMs. Several of our synthesized attacks are more effective than previously reported ones, with Attack Success Rates exceeding 90% on SOTA closed language models such as claude-3-haiku and GPT4-o. By generating datasets of jailbreak attacks in a unified formal representation, h4rm3l enables reproducible benchmarking and automated red-teaming, contributes to understanding LLM safety limitations, and supports the development of robust defenses in an increasingly LLM-integrated world.   Warning: This paper and related research artifacts contain offensive and potentially disturbing prompts and model-generated content.

摘要: 大型语言模型(LLM)的安全性仍然是一个严重的问题，因为缺乏系统地评估它们抵抗产生有害内容的能力的适当基准。以前的自动化红色团队的努力包括静态的或模板化的非法请求集和对抗性提示，鉴于越狱攻击不断演变和可组合的性质，这些提示的效用有限。我们提出了一种新的可组合越狱攻击的动态基准，以超越静态数据集和攻击和危害的分类。我们的方法由三个组件组成，统称为h4rm3l：(1)特定于领域的语言，它将越狱攻击形式化地表达为参数化提示转换原语的组合；(2)基于盗贼的少发程序合成算法，它生成经过优化的新型攻击，以穿透目标黑盒LLM的安全过滤器；以及(3)使用前两个组件的开源自动红队软件。我们使用h4rm3l生成了一个2656个成功的新型越狱攻击的数据集，目标是6个最先进的开源和专有LLM。我们的几个合成攻击比以前报道的更有效，在Claude-3-haiku和GPT4-o等Sota封闭语言模型上的攻击成功率超过90%。通过以统一的形式表示生成越狱攻击的数据集，h4rm3l实现了可重现的基准测试和自动化的红团队，有助于了解LLM的安全限制，并支持在日益集成LLM的世界中开发强大的防御措施。警告：本文和相关研究文章包含冒犯性和潜在令人不安的提示和模型生成的内容。



## **49. Assessing Adversarial Robustness of Large Language Models: An Empirical Study**

评估大型语言模型的对抗稳健性：实证研究 cs.CL

Oral presentation at KDD 2024 GenAI Evaluation workshop

**SubmitDate**: 2024-09-12    [abs](http://arxiv.org/abs/2405.02764v2) [paper-pdf](http://arxiv.org/pdf/2405.02764v2)

**Authors**: Zeyu Yang, Zhao Meng, Xiaochen Zheng, Roger Wattenhofer

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing, but their robustness against adversarial attacks remains a critical concern. We presents a novel white-box style attack approach that exposes vulnerabilities in leading open-source LLMs, including Llama, OPT, and T5. We assess the impact of model size, structure, and fine-tuning strategies on their resistance to adversarial perturbations. Our comprehensive evaluation across five diverse text classification tasks establishes a new benchmark for LLM robustness. The findings of this study have far-reaching implications for the reliable deployment of LLMs in real-world applications and contribute to the advancement of trustworthy AI systems.

摘要: 大型语言模型（LLM）彻底改变了自然语言处理，但其对抗攻击的稳健性仍然是一个关键问题。我们提出了一种新颖的白盒式攻击方法，该方法暴露了领先开源LLM（包括Llama、OPT和T5）中的漏洞。我们评估了模型大小、结构和微调策略对其抵抗对抗性扰动的影响。我们对五种不同文本分类任务的全面评估为LLM稳健性建立了新基准。这项研究的结果对于LLM在现实世界应用程序中的可靠部署具有深远的影响，并有助于发展值得信赖的人工智能系统。



## **50. Securing Large Language Models: Addressing Bias, Misinformation, and Prompt Attacks**

保护大型语言模型：解决偏见、错误信息和即时攻击 cs.CR

17 pages, 1 figure

**SubmitDate**: 2024-09-12    [abs](http://arxiv.org/abs/2409.08087v1) [paper-pdf](http://arxiv.org/pdf/2409.08087v1)

**Authors**: Benji Peng, Keyu Chen, Ming Li, Pohsun Feng, Ziqian Bi, Junyu Liu, Qian Niu

**Abstract**: Large Language Models (LLMs) demonstrate impressive capabilities across various fields, yet their increasing use raises critical security concerns. This article reviews recent literature addressing key issues in LLM security, with a focus on accuracy, bias, content detection, and vulnerability to attacks. Issues related to inaccurate or misleading outputs from LLMs is discussed, with emphasis on the implementation from fact-checking methodologies to enhance response reliability. Inherent biases within LLMs are critically examined through diverse evaluation techniques, including controlled input studies and red teaming exercises. A comprehensive analysis of bias mitigation strategies is presented, including approaches from pre-processing interventions to in-training adjustments and post-processing refinements. The article also probes the complexity of distinguishing LLM-generated content from human-produced text, introducing detection mechanisms like DetectGPT and watermarking techniques while noting the limitations of machine learning enabled classifiers under intricate circumstances. Moreover, LLM vulnerabilities, including jailbreak attacks and prompt injection exploits, are analyzed by looking into different case studies and large-scale competitions like HackAPrompt. This review is concluded by retrospecting defense mechanisms to safeguard LLMs, accentuating the need for more extensive research into the LLM security field.

摘要: 大型语言模型(LLM)在各个领域展示了令人印象深刻的能力，但它们的日益使用引发了严重的安全问题。本文回顾了解决LLM安全关键问题的最新文献，重点讨论了准确性、偏差、内容检测和攻击漏洞。讨论了与LLMS输出不准确或误导性有关的问题，重点是从事实核查方法中实施以提高响应可靠性。通过不同的评估技术，包括受控投入研究和红色团队练习，对LLM中的固有偏见进行了严格的检查。对偏差缓解策略进行了全面的分析，包括从前处理干预到培训中调整和后处理改进的方法。文章还探讨了区分LLM生成的内容和人类生成的文本的复杂性，介绍了DetectGPT和水印技术等检测机制，同时指出了机器学习支持的分类器在复杂环境下的局限性。此外，通过研究不同的案例研究和HackAPrompt等大型竞争，分析了LLM漏洞，包括越狱攻击和快速注入利用。本综述通过回顾保护LLM的防御机制来结束，强调了对LLM安全领域进行更广泛研究的必要性。



