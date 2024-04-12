# Latest Large Language Model Attack Papers
**update at 2024-04-12 09:22:41**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. AmpleGCG: Learning a Universal and Transferable Generative Model of Adversarial Suffixes for Jailbreaking Both Open and Closed LLMs**

AmpleGCG：学习一个通用的、可转移的对抗后缀生成模型，用于越狱既开放式又封闭式LLM cs.CL

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.07921v1) [paper-pdf](http://arxiv.org/pdf/2404.07921v1)

**Authors**: Zeyi Liao, Huan Sun

**Abstract**: As large language models (LLMs) become increasingly prevalent and integrated into autonomous systems, ensuring their safety is imperative. Despite significant strides toward safety alignment, recent work GCG~\citep{zou2023universal} proposes a discrete token optimization algorithm and selects the single suffix with the lowest loss to successfully jailbreak aligned LLMs. In this work, we first discuss the drawbacks of solely picking the suffix with the lowest loss during GCG optimization for jailbreaking and uncover the missed successful suffixes during the intermediate steps. Moreover, we utilize those successful suffixes as training data to learn a generative model, named AmpleGCG, which captures the distribution of adversarial suffixes given a harmful query and enables the rapid generation of hundreds of suffixes for any harmful queries in seconds. AmpleGCG achieves near 100\% attack success rate (ASR) on two aligned LLMs (Llama-2-7B-chat and Vicuna-7B), surpassing two strongest attack baselines. More interestingly, AmpleGCG also transfers seamlessly to attack different models, including closed-source LLMs, achieving a 99\% ASR on the latest GPT-3.5. To summarize, our work amplifies the impact of GCG by training a generative model of adversarial suffixes that is universal to any harmful queries and transferable from attacking open-source LLMs to closed-source LLMs. In addition, it can generate 200 adversarial suffixes for one harmful query in only 4 seconds, rendering it more challenging to defend.

摘要: 随着大型语言模型(LLM)变得越来越普遍并集成到自治系统中，确保它们的安全性是当务之急。尽管在安全对齐方面取得了长足的进步，但最近的工作GCG~\Citep{zou2023Universal}提出了一种离散令牌优化算法，并选择损失最小的单个后缀来成功越狱对齐LLM。在这项工作中，我们首先讨论了在GCG优化越狱过程中只选择损失最小的后缀的缺点，并在中间步骤中发现了遗漏的成功后缀。此外，我们利用这些成功的后缀作为训练数据来学习一种名为AmpleGCG的生成模型，该模型捕获给定有害查询的对抗性后缀的分布，并在几秒钟内为任何有害查询快速生成数百个后缀。AmpleGCG在两个对齐的LLM(Llama-2-7B-Chat和Vicuna-7B)上达到了近100%的攻击成功率(ASR)，超过了两个最强的攻击基线。更有趣的是，AmpleGCG还无缝传输以攻击不同的型号，包括闭源LLMS，在最新的GPT-3.5上实现了99\%的ASR。总之，我们的工作通过训练对抗性后缀的生成模型来放大GCG的影响，该模型对任何有害的查询都是通用的，并且可以从攻击开源LLM转移到闭源LLM。此外，它可以在短短4秒内为一个有害的查询生成200个敌意后缀，使其更具挑战性。



## **2. AnnoCTR: A Dataset for Detecting and Linking Entities, Tactics, and Techniques in Cyber Threat Reports**

AnnoCTR：网络威胁报告中用于检测和链接实体、战术和技术的数据集 cs.CL

Accepted at LREC-COLING 2024. Corpus available at  https://github.com/boschresearch/anno-ctr-lrec-coling-2024

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.07765v1) [paper-pdf](http://arxiv.org/pdf/2404.07765v1)

**Authors**: Lukas Lange, Marc Müller, Ghazaleh Haratinezhad Torbati, Dragan Milchevski, Patrick Grau, Subhash Pujari, Annemarie Friedrich

**Abstract**: Monitoring the threat landscape to be aware of actual or potential attacks is of utmost importance to cybersecurity professionals. Information about cyber threats is typically distributed using natural language reports. Natural language processing can help with managing this large amount of unstructured information, yet to date, the topic has received little attention. With this paper, we present AnnoCTR, a new CC-BY-SA-licensed dataset of cyber threat reports. The reports have been annotated by a domain expert with named entities, temporal expressions, and cybersecurity-specific concepts including implicitly mentioned techniques and tactics. Entities and concepts are linked to Wikipedia and the MITRE ATT&CK knowledge base, the most widely-used taxonomy for classifying types of attacks. Prior datasets linking to MITRE ATT&CK either provide a single label per document or annotate sentences out-of-context; our dataset annotates entire documents in a much finer-grained way. In an experimental study, we model the annotations of our dataset using state-of-the-art neural models. In our few-shot scenario, we find that for identifying the MITRE ATT&CK concepts that are mentioned explicitly or implicitly in a text, concept descriptions from MITRE ATT&CK are an effective source for training data augmentation.

摘要: 监控威胁环境以了解实际或潜在的攻击对网络安全专业人员来说至关重要。有关网络威胁的信息通常使用自然语言报告发布。自然语言处理可以帮助管理这些大量的非结构化信息，但到目前为止，这个主题几乎没有受到关注。在本文中，我们介绍了AnnoCTR，一个新的CC-by-SA许可的网络威胁报告数据集。这些报告已经由领域专家使用命名实体、时间表达式和网络安全特定概念(包括隐含提到的技术和战术)进行了注释。实体和概念链接到维基百科和MITRE ATT&CK知识库，后者是对攻击类型进行分类的最广泛使用的分类法。以前链接到MITRE ATT&CK的数据集要么为每个文档提供单一标签，要么断章取义地注释句子；我们的数据集以更细粒度的方式注释整个文档。在一项实验研究中，我们使用最先进的神经模型对数据集的注释进行建模。在我们的少数情况下，我们发现，对于识别文本中明确或隐含地提到的MITRE ATT&CK概念，MITRE ATT&CK的概念描述是训练数据增强的有效来源。



## **3. Sandwich attack: Multi-language Mixture Adaptive Attack on LLMs**

三明治攻击：对LLM的多语言混合自适应攻击 cs.CR

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2404.07242v1) [paper-pdf](http://arxiv.org/pdf/2404.07242v1)

**Authors**: Bibek Upadhayay, Vahid Behzadan

**Abstract**: Large Language Models (LLMs) are increasingly being developed and applied, but their widespread use faces challenges. These include aligning LLMs' responses with human values to prevent harmful outputs, which is addressed through safety training methods. Even so, bad actors and malicious users have succeeded in attempts to manipulate the LLMs to generate misaligned responses for harmful questions such as methods to create a bomb in school labs, recipes for harmful drugs, and ways to evade privacy rights. Another challenge is the multilingual capabilities of LLMs, which enable the model to understand and respond in multiple languages. Consequently, attackers exploit the unbalanced pre-training datasets of LLMs in different languages and the comparatively lower model performance in low-resource languages than high-resource ones. As a result, attackers use a low-resource languages to intentionally manipulate the model to create harmful responses. Many of the similar attack vectors have been patched by model providers, making the LLMs more robust against language-based manipulation. In this paper, we introduce a new black-box attack vector called the \emph{Sandwich attack}: a multi-language mixture attack, which manipulates state-of-the-art LLMs into generating harmful and misaligned responses. Our experiments with five different models, namely Google's Bard, Gemini Pro, LLaMA-2-70-B-Chat, GPT-3.5-Turbo, GPT-4, and Claude-3-OPUS, show that this attack vector can be used by adversaries to generate harmful responses and elicit misaligned responses from these models. By detailing both the mechanism and impact of the Sandwich attack, this paper aims to guide future research and development towards more secure and resilient LLMs, ensuring they serve the public good while minimizing potential for misuse.

摘要: 大型语言模型(LLM)的开发和应用越来越多，但它们的广泛使用面临着挑战。这些措施包括使LLMS的反应与人的价值观相一致，以防止有害的输出，这是通过安全培训方法解决的。尽管如此，不良行为者和恶意用户仍成功地操纵LLMS，以生成对有害问题的错误响应，这些问题包括在学校实验室制造炸弹的方法、有害药物的配方以及逃避隐私权的方法。另一个挑战是LLMS的多语言能力，这使得该模型能够理解并以多种语言响应。因此，攻击者利用不同语言的LLMS的不平衡的预训练数据集，以及低资源语言的模型性能相对较低的高资源语言。因此，攻击者使用低资源语言来故意操纵模型以创建有害的响应。许多类似的攻击载体已经由模型提供商修补，使LLM对基于语言的操纵更加健壮。本文介绍了一种新的黑盒攻击向量--夹心攻击：一种多语言混合攻击，它操纵最先进的LLM产生有害的和未对齐的响应。我们对谷歌的Bard、Gemini Pro、Llama-2-70-B-Chat、GPT-3.5-Turbo、GPT-4和Claude-3-opus这五个不同的模型进行的实验表明，该攻击向量可被攻击者用来生成有害响应并从这些模型中引发错误的响应。通过详细描述三明治攻击的机制和影响，本文旨在引导未来的研究和开发朝着更安全和更具弹性的方向发展，确保它们服务于公共利益，同时将滥用的可能性降至最低。



## **4. $\textit{LinkPrompt}$: Natural and Universal Adversarial Attacks on Prompt-based Language Models**

$\textit{LinkPrompt}$：基于XSLT语言模型的自然和普遍对抗攻击 cs.CL

Accepted to the main conference of NAACL2024

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2403.16432v3) [paper-pdf](http://arxiv.org/pdf/2403.16432v3)

**Authors**: Yue Xu, Wenjie Wang

**Abstract**: Prompt-based learning is a new language model training paradigm that adapts the Pre-trained Language Models (PLMs) to downstream tasks, which revitalizes the performance benchmarks across various natural language processing (NLP) tasks. Instead of using a fixed prompt template to fine-tune the model, some research demonstrates the effectiveness of searching for the prompt via optimization. Such prompt optimization process of prompt-based learning on PLMs also gives insight into generating adversarial prompts to mislead the model, raising concerns about the adversarial vulnerability of this paradigm. Recent studies have shown that universal adversarial triggers (UATs) can be generated to alter not only the predictions of the target PLMs but also the prediction of corresponding Prompt-based Fine-tuning Models (PFMs) under the prompt-based learning paradigm. However, UATs found in previous works are often unreadable tokens or characters and can be easily distinguished from natural texts with adaptive defenses. In this work, we consider the naturalness of the UATs and develop $\textit{LinkPrompt}$, an adversarial attack algorithm to generate UATs by a gradient-based beam search algorithm that not only effectively attacks the target PLMs and PFMs but also maintains the naturalness among the trigger tokens. Extensive results demonstrate the effectiveness of $\textit{LinkPrompt}$, as well as the transferability of UATs generated by $\textit{LinkPrompt}$ to open-sourced Large Language Model (LLM) Llama2 and API-accessed LLM GPT-3.5-turbo. The resource is available at $\href{https://github.com/SavannahXu79/LinkPrompt}{https://github.com/SavannahXu79/LinkPrompt}$.

摘要: 基于提示的学习是一种新的语言模型训练范式，它使预先训练的语言模型(PLM)适应于下游任务，从而重振各种自然语言处理(NLP)任务的表现基准。一些研究证明了通过优化来搜索提示的有效性，而不是使用固定的提示模板来微调模型。这种基于提示的PLM学习的快速优化过程也为生成对抗性提示以误导模型提供了洞察力，这引发了人们对这种范式的对抗性脆弱性的担忧。最近的研究表明，在基于提示的学习范式下，通用对抗触发器(UAT)不仅可以改变目标PLM的预测，还可以改变相应的基于提示的精调模型(PFM)的预测。然而，在以前的著作中发现的UAT通常是不可读的符号或字符，并且可以很容易地与具有自适应防御的自然文本区分开来。在这项工作中，我们考虑了UAT的自然性，并开发了一种对抗性攻击算法，通过基于梯度的波束搜索算法来生成UAT，该算法不仅有效地攻击了目标PLM和PPM，而且保持了触发令牌之间的自然度。广泛的结果证明了$\textit{LinkPrompt}$的有效性，以及由$\textit{LinkPrompt}$生成的UAT可以移植到开源的大型语言模型(LLM)Llama2和API访问的LLm GPT-3.5-Turbo。该资源可在$\href{https://github.com/SavannahXu79/LinkPrompt}{https://github.com/SavannahXu79/LinkPrompt}$.上获得



## **5. AEGIS: Online Adaptive AI Content Safety Moderation with Ensemble of LLM Experts**

AEGIS：在线自适应人工智能内容安全管理与法学硕士专家的邀请 cs.LG

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2404.05993v1) [paper-pdf](http://arxiv.org/pdf/2404.05993v1)

**Authors**: Shaona Ghosh, Prasoon Varshney, Erick Galinkin, Christopher Parisien

**Abstract**: As Large Language Models (LLMs) and generative AI become more widespread, the content safety risks associated with their use also increase. We find a notable deficiency in high-quality content safety datasets and benchmarks that comprehensively cover a wide range of critical safety areas. To address this, we define a broad content safety risk taxonomy, comprising 13 critical risk and 9 sparse risk categories. Additionally, we curate AEGISSAFETYDATASET, a new dataset of approximately 26, 000 human-LLM interaction instances, complete with human annotations adhering to the taxonomy. We plan to release this dataset to the community to further research and to help benchmark LLM models for safety. To demonstrate the effectiveness of the dataset, we instruction-tune multiple LLM-based safety models. We show that our models (named AEGISSAFETYEXPERTS), not only surpass or perform competitively with the state-of-the-art LLM-based safety models and general purpose LLMs, but also exhibit robustness across multiple jail-break attack categories. We also show how using AEGISSAFETYDATASET during the LLM alignment phase does not negatively impact the performance of the aligned models on MT Bench scores. Furthermore, we propose AEGIS, a novel application of a no-regret online adaptation framework with strong theoretical guarantees, to perform content moderation with an ensemble of LLM content safety experts in deployment

摘要: 随着大型语言模型(LLM)和生成式人工智能变得更加普遍，与使用它们相关的内容安全风险也增加了。我们发现，在全面覆盖广泛关键安全领域的高质量内容安全数据集和基准方面存在明显不足。为了解决这一问题，我们定义了一个广泛的内容安全风险分类，包括13个关键风险和9个稀疏风险类别。此外，我们还管理了AEGISSAFETYDATASET，这是一个大约包含26,000个人与LLM交互实例的新数据集，其中包含符合分类的人类注释。我们计划向社区发布这个数据集，以进行进一步的研究，并帮助对LLM模型的安全性进行基准测试。为了证明数据集的有效性，我们对多个基于LLM的安全模型进行了指令调整。我们证明了我们的模型(命名为AEGISSAFETYEXPERTS)不仅性能优于最先进的基于LLM的安全模型和通用LLM，而且在多个越狱攻击类别中表现出健壮性。我们还展示了在LLM校准阶段使用AEGISSAFETYDATASET如何不会对MT板凳分数上的校准模型的性能产生负面影响。此外，我们提出了Aegis，一个具有强大理论保证的无遗憾在线适配框架的新应用，在部署时使用LLM内容安全专家集成来执行内容审核



## **6. Eraser: Jailbreaking Defense in Large Language Models via Unlearning Harmful Knowledge**

橡皮擦：通过放弃有害知识在大型语言模型中的越狱防御 cs.CL

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05880v1) [paper-pdf](http://arxiv.org/pdf/2404.05880v1)

**Authors**: Weikai Lu, Ziqian Zeng, Jianwei Wang, Zhengdong Lu, Zelin Chen, Huiping Zhuang, Cen Chen

**Abstract**: Jailbreaking attacks can enable Large Language Models (LLMs) to bypass the safeguard and generate harmful content. Existing jailbreaking defense methods have failed to address the fundamental issue that harmful knowledge resides within the model, leading to potential jailbreak risks for LLMs. In this paper, we propose a novel defense method called Eraser, which mainly includes three goals: unlearning harmful knowledge, retaining general knowledge, and maintaining safety alignment. The intuition is that if an LLM forgets the specific knowledge required to answer a harmful question, it will no longer have the ability to answer harmful questions. The training of Erase does not actually require the model's own harmful knowledge, and it can benefit from unlearning general answers related to harmful queries, which means it does not need assistance from the red team. The experimental results show that Eraser can significantly reduce the jailbreaking success rate for various attacks without compromising the general capabilities of the model.

摘要: 越狱攻击可使大型语言模型(LLM)绕过安全保护并生成有害内容。现有的越狱防御方法未能解决有害知识驻留在模型中的根本问题，导致低收入国家存在潜在的越狱风险。在本文中，我们提出了一种新的防御方法--橡皮擦，它主要包括三个目标：忘记有害知识，保留一般知识，保持安全对齐。直觉是，如果LLM忘记了回答有害问题所需的特定知识，它将不再有能力回答有害问题。ERASE的训练实际上并不需要模型本身的有害知识，而且它可以受益于忘记与有害查询相关的一般答案，这意味着它不需要红色团队的帮助。实验结果表明，在不影响模型整体性能的前提下，橡皮擦能显著降低各种攻击的越狱成功率。



## **7. Exploring the Deceptive Power of LLM-Generated Fake News: A Study of Real-World Detection Challenges**

探索LLM生成的假新闻的欺骗力量：现实世界的检测挑战研究 cs.CL

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2403.18249v2) [paper-pdf](http://arxiv.org/pdf/2403.18249v2)

**Authors**: Yanshen Sun, Jianfeng He, Limeng Cui, Shuo Lei, Chang-Tien Lu

**Abstract**: Recent advancements in Large Language Models (LLMs) have enabled the creation of fake news, particularly in complex fields like healthcare. Studies highlight the gap in the deceptive power of LLM-generated fake news with and without human assistance, yet the potential of prompting techniques has not been fully explored. Thus, this work aims to determine whether prompting strategies can effectively narrow this gap. Current LLM-based fake news attacks require human intervention for information gathering and often miss details and fail to maintain context consistency. Therefore, to better understand threat tactics, we propose a strong fake news attack method called conditional Variational-autoencoder-Like Prompt (VLPrompt). Unlike current methods, VLPrompt eliminates the need for additional data collection while maintaining contextual coherence and preserving the intricacies of the original text. To propel future research on detecting VLPrompt attacks, we created a new dataset named VLPrompt fake news (VLPFN) containing real and fake texts. Our experiments, including various detection methods and novel human study metrics, were conducted to assess their performance on our dataset, yielding numerous findings.

摘要: 最近大型语言模型(LLM)的进步使假新闻的创造成为可能，特别是在医疗保健等复杂领域。研究突显了在有无人工帮助的情况下，LLM生成的假新闻的欺骗力存在差距，但提示技术的潜力尚未得到充分挖掘。因此，本研究旨在确定激励策略是否能有效缩小这一差距。目前基于LLM的假新闻攻击需要人工干预来收集信息，并且经常错过细节，无法保持上下文一致性。因此，为了更好地理解威胁策略，我们提出了一种强大的假新闻攻击方法，称为条件变分自动编码式提示(VLPrompt)。与目前的方法不同，VLPrompt消除了对额外数据收集的需要，同时保持了上下文的连贯性和原始文本的错综复杂。为了推动未来对VLPrompt攻击检测的研究，我们创建了一个新的数据集，名为VLPrompt假新闻(VLPFN)，包含真实和虚假的文本。我们的实验，包括各种检测方法和新的人体研究指标，被用来评估它们在我们的数据集上的性能，产生了许多发现。



## **8. Best-of-Venom: Attacking RLHF by Injecting Poisoned Preference Data**

毒液最佳：通过注射中毒偏好数据攻击RLHF cs.CL

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05530v1) [paper-pdf](http://arxiv.org/pdf/2404.05530v1)

**Authors**: Tim Baumgärtner, Yang Gao, Dana Alon, Donald Metzler

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is a popular method for aligning Language Models (LM) with human values and preferences. RLHF requires a large number of preference pairs as training data, which are often used in both the Supervised Fine-Tuning and Reward Model training, and therefore publicly available datasets are commonly used. In this work, we study to what extent a malicious actor can manipulate the LMs generations by poisoning the preferences, i.e., injecting poisonous preference pairs into these datasets and the RLHF training process. We propose strategies to build poisonous preference pairs and test their performance by poisoning two widely used preference datasets. Our results show that preference poisoning is highly effective: by injecting a small amount of poisonous data (1-5% of the original dataset), we can effectively manipulate the LM to generate a target entity in a target sentiment (positive or negative). The findings from our experiments also shed light on strategies to defend against the preference poisoning attack.

摘要: 人类反馈强化学习(RLHF)是一种使语言模型与人类价值观和偏好保持一致的流行方法。RLHF需要大量的偏好对作为训练数据，这些数据经常用于有监督的精调和奖励模型训练，因此通常使用公开可用的数据集。在这项工作中，我们研究了恶意行为者可以在多大程度上通过毒化偏好来操纵LMS生成，即向这些数据集注入有毒的偏好对和RLHF训练过程。我们提出了构建有毒偏好对的策略，并通过毒化两个广泛使用的偏好数据集来测试它们的性能。我们的结果表明偏好毒化是非常有效的：通过注入少量的有毒数据(原始数据集的1-5%)，我们可以有效地操纵LM来生成目标情感中的目标实体(积极或消极)。我们的实验结果也为防御偏好中毒攻击的策略提供了启示。



## **9. Unbridled Icarus: A Survey of the Potential Perils of Image Inputs in Multimodal Large Language Model Security**

Icarus：多模态大型语言模型安全中图像输入的潜在危险调查 cs.CR

8 pages, 1 figure

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05264v1) [paper-pdf](http://arxiv.org/pdf/2404.05264v1)

**Authors**: Yihe Fan, Yuxin Cao, Ziyu Zhao, Ziyao Liu, Shaofeng Li

**Abstract**: Multimodal Large Language Models (MLLMs) demonstrate remarkable capabilities that increasingly influence various aspects of our daily lives, constantly defining the new boundary of Artificial General Intelligence (AGI). Image modalities, enriched with profound semantic information and a more continuous mathematical nature compared to other modalities, greatly enhance the functionalities of MLLMs when integrated. However, this integration serves as a double-edged sword, providing attackers with expansive vulnerabilities to exploit for highly covert and harmful attacks. The pursuit of reliable AI systems like powerful MLLMs has emerged as a pivotal area of contemporary research. In this paper, we endeavor to demostrate the multifaceted risks associated with the incorporation of image modalities into MLLMs. Initially, we delineate the foundational components and training processes of MLLMs. Subsequently, we construct a threat model, outlining the security vulnerabilities intrinsic to MLLMs. Moreover, we analyze and summarize existing scholarly discourses on MLLMs' attack and defense mechanisms, culminating in suggestions for the future research on MLLM security. Through this comprehensive analysis, we aim to deepen the academic understanding of MLLM security challenges and propel forward the development of trustworthy MLLM systems.

摘要: 多通道大语言模型(MLLMS)显示出非凡的能力，越来越多地影响我们日常生活的各个方面，不断定义人工通用智能(AGI)的新边界。与其他模式相比，图像模式具有更丰富的语义信息和更连续的数学性质，极大地增强了MLLMS的功能。然而，这种集成是一把双刃剑，为攻击者提供了大量漏洞，可以利用这些漏洞进行高度隐蔽和有害的攻击。像强大的MLLMS这样可靠的人工智能系统的追求已经成为当代研究的一个关键领域。在这篇文章中，我们努力展示与将图像模式结合到MLLMS中相关的多方面风险。首先，我们描述了MLLMS的基本组成部分和培训过程。随后，我们构建了威胁模型，概述了MLLMS固有的安全漏洞。此外，我们分析和总结了已有的关于MLLMS攻击和防御机制的学术论述，并对未来的MLLMS安全研究提出了建议。通过这一综合分析，我们旨在加深对MLLM安全挑战的学术认识，推动可信MLLM系统的发展。



## **10. REMARK-LLM: A Robust and Efficient Watermarking Framework for Generative Large Language Models**

REMARK-LLM：一个健壮高效的生成性大语言模型水印框架 cs.CR

accept to usenix security 2024

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2310.12362v2) [paper-pdf](http://arxiv.org/pdf/2310.12362v2)

**Authors**: Ruisi Zhang, Shehzeen Samarah Hussain, Paarth Neekhara, Farinaz Koushanfar

**Abstract**: We present REMARK-LLM, a novel efficient, and robust watermarking framework designed for texts generated by large language models (LLMs). Synthesizing human-like content using LLMs necessitates vast computational resources and extensive datasets, encapsulating critical intellectual property (IP). However, the generated content is prone to malicious exploitation, including spamming and plagiarism. To address the challenges, REMARK-LLM proposes three new components: (i) a learning-based message encoding module to infuse binary signatures into LLM-generated texts; (ii) a reparameterization module to transform the dense distributions from the message encoding to the sparse distribution of the watermarked textual tokens; (iii) a decoding module dedicated for signature extraction; Furthermore, we introduce an optimized beam search algorithm to guarantee the coherence and consistency of the generated content. REMARK-LLM is rigorously trained to encourage the preservation of semantic integrity in watermarked content, while ensuring effective watermark retrieval. Extensive evaluations on multiple unseen datasets highlight REMARK-LLM proficiency and transferability in inserting 2 times more signature bits into the same texts when compared to prior art, all while maintaining semantic integrity. Furthermore, REMARK-LLM exhibits better resilience against a spectrum of watermark detection and removal attacks.

摘要: 我们提出了一种新的高效、健壮的数字水印框架--REMARK-LLM，它适用于大型语言模型(LLMS)生成的文本。使用LLMS合成类似人类的内容需要大量的计算资源和大量的数据集，并封装关键知识产权(IP)。然而，生成的内容容易受到恶意攻击，包括垃圾邮件和抄袭。为了应对这些挑战，REMARK-LLM提出了三个新的组件：(I)基于学习的消息编码模块，将二进制签名注入到LLM生成的文本中；(Ii)重新参数化模块，将密集分布的消息编码转换为稀疏分布的水印文本标记；(Iii)专门用于签名提取的解码模块；此外，我们还引入了优化的波束搜索算法，以保证生成内容的一致性和一致性。Rmark-LLM经过了严格的培训，以鼓励在水印内容中保持语义完整性，同时确保有效的水印检索。对多个不可见数据集的广泛评估突出了Remmark-LLM的熟练程度和可转移性，即与现有技术相比，在相同的文本中插入2倍多的签名比特，同时保持语义完整性。此外，REMARK-LLM对一系列水印检测和删除攻击表现出更好的弹性。



## **11. Hidden You Malicious Goal Into Benigh Narratives: Jailbreak Large Language Models through Logic Chain Injection**

隐藏你的恶意目标进入良性叙事：通过逻辑链注入越狱大型语言模型 cs.CR

**SubmitDate**: 2024-04-07    [abs](http://arxiv.org/abs/2404.04849v1) [paper-pdf](http://arxiv.org/pdf/2404.04849v1)

**Authors**: Zhilong Wang, Yebo Cao, Peng Liu

**Abstract**: Jailbreak attacks on Language Model Models (LLMs) entail crafting prompts aimed at exploiting the models to generate malicious content. Existing jailbreak attacks can successfully deceive the LLMs, however they cannot deceive the human. This paper proposes a new type of jailbreak attacks which can deceive both the LLMs and human (i.e., security analyst). The key insight of our idea is borrowed from the social psychology - that is human are easily deceived if the lie is hidden in truth. Based on this insight, we proposed the logic-chain injection attacks to inject malicious intention into benign truth. Logic-chain injection attack firstly dissembles its malicious target into a chain of benign narrations, and then distribute narrations into a related benign article, with undoubted facts. In this way, newly generate prompt cannot only deceive the LLMs, but also deceive human.

摘要: 对语言模型模型（LLM）的越狱攻击需要精心制作旨在利用模型生成恶意内容的提示。现有的越狱攻击可以成功地欺骗LLM，但他们不能欺骗人类。本文提出了一种新型的越狱攻击，它可以同时欺骗LLM和人类（即，安全分析师）。我们的观点的关键洞察力是从社会心理学中借用的--即如果谎言隐藏在真相中，人类很容易被欺骗。基于这一认识，我们提出了逻辑链注入攻击，将恶意意图注入良性真实。逻辑链注入攻击首先将恶意目标伪装成一条良性的叙述链，然后再将叙述分发成一条相关的良性文章，并带有虚假的事实。这样，新生成的提示不仅可以欺骗LLM，而且可以欺骗人类。



## **12. A Wolf in Sheep's Clothing: Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily**

披着羊皮的狼：广义嵌套越狱陷阱可以轻松愚弄大型语言模型 cs.CL

Acccepted by NAACL 2024, 18 pages, 7 figures, 13 tables

**SubmitDate**: 2024-04-07    [abs](http://arxiv.org/abs/2311.08268v4) [paper-pdf](http://arxiv.org/pdf/2311.08268v4)

**Authors**: Peng Ding, Jun Kuang, Dan Ma, Xuezhi Cao, Yunsen Xian, Jiajun Chen, Shujian Huang

**Abstract**: Large Language Models (LLMs), such as ChatGPT and GPT-4, are designed to provide useful and safe responses. However, adversarial prompts known as 'jailbreaks' can circumvent safeguards, leading LLMs to generate potentially harmful content. Exploring jailbreak prompts can help to better reveal the weaknesses of LLMs and further steer us to secure them. Unfortunately, existing jailbreak methods either suffer from intricate manual design or require optimization on other white-box models, which compromises either generalization or efficiency. In this paper, we generalize jailbreak prompt attacks into two aspects: (1) Prompt Rewriting and (2) Scenario Nesting. Based on this, we propose ReNeLLM, an automatic framework that leverages LLMs themselves to generate effective jailbreak prompts. Extensive experiments demonstrate that ReNeLLM significantly improves the attack success rate while greatly reducing the time cost compared to existing baselines. Our study also reveals the inadequacy of current defense methods in safeguarding LLMs. Finally, we analyze the failure of LLMs defense from the perspective of prompt execution priority, and propose corresponding defense strategies. We hope that our research can catalyze both the academic community and LLMs developers towards the provision of safer and more regulated LLMs. The code is available at https://github.com/NJUNLP/ReNeLLM.

摘要: 大型语言模型(LLM)，如ChatGPT和GPT-4，旨在提供有用和安全的响应。然而，被称为“越狱”的对抗性提示可能会绕过安全措施，导致LLMS生成潜在的有害内容。探索越狱提示可以帮助更好地揭示LLM的弱点，并进一步指导我们确保它们的安全。不幸的是，现有的越狱方法要么需要复杂的人工设计，要么需要对其他白盒模型进行优化，这要么损害了通用性，要么影响了效率。本文将越狱提示攻击概括为两个方面：(1)提示重写和(2)场景嵌套。在此基础上，我们提出了ReNeLLM，这是一个利用LLM自身生成有效越狱提示的自动化框架。广泛的实验表明，与现有的基准相比，ReNeLLM显著提高了攻击成功率，同时大大降低了时间成本。我们的研究也揭示了现有防御方法在保护低密度脂蛋白方面的不足。最后，从即时执行优先级的角度分析了LLMS防御失败的原因，并提出了相应的防御策略。我们希望我们的研究能够促进学术界和低成本管理系统开发商提供更安全和更规范的低成本管理系统。代码可在https://github.com/NJUNLP/ReNeLLM.上获得



## **13. Red Teaming Game: A Game-Theoretic Framework for Red Teaming Language Models**

红色团队游戏：红色团队语言模型的游戏理论框架 cs.CL

**SubmitDate**: 2024-04-06    [abs](http://arxiv.org/abs/2310.00322v4) [paper-pdf](http://arxiv.org/pdf/2310.00322v4)

**Authors**: Chengdong Ma, Ziran Yang, Minquan Gao, Hai Ci, Jun Gao, Xuehai Pan, Yaodong Yang

**Abstract**: Deployable Large Language Models (LLMs) must conform to the criterion of helpfulness and harmlessness, thereby achieving consistency between LLMs outputs and human values. Red-teaming techniques constitute a critical way towards this criterion. Existing work rely solely on manual red team designs and heuristic adversarial prompts for vulnerability detection and optimization. These approaches lack rigorous mathematical formulation, thus limiting the exploration of diverse attack strategy within quantifiable measure and optimization of LLMs under convergence guarantees. In this paper, we present Red-teaming Game (RTG), a general game-theoretic framework without manual annotation. RTG is designed for analyzing the multi-turn attack and defense interactions between Red-team language Models (RLMs) and Blue-team Language Model (BLM). Within the RTG, we propose Gamified Red-teaming Solver (GRTS) with diversity measure of the semantic space. GRTS is an automated red teaming technique to solve RTG towards Nash equilibrium through meta-game analysis, which corresponds to the theoretically guaranteed optimization direction of both RLMs and BLM. Empirical results in multi-turn attacks with RLMs show that GRTS autonomously discovered diverse attack strategies and effectively improved security of LLMs, outperforming existing heuristic red-team designs. Overall, RTG has established a foundational framework for red teaming tasks and constructed a new scalable oversight technique for alignment.

摘要: 可部署的大型语言模型(LLMS)必须符合有益和无害的标准，从而实现LLMS的输出与人的价值之间的一致性。红团队技术构成了实现这一标准的关键途径。现有的工作完全依赖于手动红色团队设计和启发式对抗性提示来进行漏洞检测和优化。这些方法缺乏严格的数学描述，从而限制了在可量化的度量范围内探索多样化的攻击策略，以及在收敛保证下对LLMS进行优化。在本文中，我们提出了一种不需要人工注释的通用博弈论框架--Red-Teaming Game(RTG)。RTG用于分析红队语言模型(RLMS)和蓝队语言模型(BLM)之间的多回合攻防交互。在RTG中，我们提出了一种具有语义空间多样性度量的Gamalized Red-Teaming Solver(GRTS)。GRTS是一种自动红队技术，通过元博弈分析解决RTG向纳什均衡的方向，这对应于理论上保证的RLMS和BLM的优化方向。在RLMS多回合攻击中的实验结果表明，GRTS自主发现多样化的攻击策略，有效地提高了LLMS的安全性，优于已有的启发式红队设计。总体而言，RTG为红色团队任务建立了一个基本框架，并构建了一种新的可扩展的协调监督技术。



## **14. Goal-guided Generative Prompt Injection Attack on Large Language Models**

面向大型语言模型的目标引导生成式提示注入攻击 cs.CR

22 pages, 8 figures

**SubmitDate**: 2024-04-06    [abs](http://arxiv.org/abs/2404.07234v1) [paper-pdf](http://arxiv.org/pdf/2404.07234v1)

**Authors**: Chong Zhang, Mingyu Jin, Qinkai Yu, Chengzhi Liu, Haochen Xue, Xiaobo Jin

**Abstract**: Current large language models (LLMs) provide a strong foundation for large-scale user-oriented natural language tasks. A large number of users can easily inject adversarial text or instructions through the user interface, thus causing LLMs model security challenges. Although there is currently a large amount of research on prompt injection attacks, most of these black-box attacks use heuristic strategies. It is unclear how these heuristic strategies relate to the success rate of attacks and thus effectively improve model robustness. To solve this problem, we redefine the goal of the attack: to maximize the KL divergence between the conditional probabilities of the clean text and the adversarial text. Furthermore, we prove that maximizing the KL divergence is equivalent to maximizing the Mahalanobis distance between the embedded representation $x$ and $x'$ of the clean text and the adversarial text when the conditional probability is a Gaussian distribution and gives a quantitative relationship on $x$ and $x'$. Then we designed a simple and effective goal-guided generative prompt injection strategy (G2PIA) to find an injection text that satisfies specific constraints to achieve the optimal attack effect approximately. It is particularly noteworthy that our attack method is a query-free black-box attack method with low computational cost. Experimental results on seven LLM models and four datasets show the effectiveness of our attack method.

摘要: 现有的大型语言模型为大规模面向用户的自然语言任务提供了坚实的基础。大量用户可以很容易地通过用户界面注入敌意文本或指令，从而造成LLMS模型的安全挑战。虽然目前有大量关于即时注入攻击的研究，但这些黑盒攻击大多采用启发式策略。目前尚不清楚这些启发式策略如何与攻击成功率相关，从而有效地提高模型的稳健性。为了解决这个问题，我们重新定义了攻击的目标：最大化纯文本和敌意文本的条件概率之间的KL偏差。此外，我们证明了当条件概率为高斯分布时，最大化KL发散度等价于最大化明文和敌意文本的嵌入表示$x$和$x‘$之间的马氏距离，并给出了关于$x$和$x’$的定量关系。然后，设计了一种简单有效的目标引导生成性提示注入策略(G2PIA)，找到满足特定约束的注入文本，以近似达到最优的攻击效果。特别值得注意的是，我们的攻击方法是一种计算代价低的无查询黑盒攻击方法。在7个LLM模型和4个数据集上的实验结果表明了该攻击方法的有效性。



## **15. Removing RLHF Protections in GPT-4 via Fine-Tuning**

通过微调移除GPT-4中的RLHF保护 cs.CL

Accepted to NAACL 2024. (7 pages)

**SubmitDate**: 2024-04-05    [abs](http://arxiv.org/abs/2311.05553v3) [paper-pdf](http://arxiv.org/pdf/2311.05553v3)

**Authors**: Qiusi Zhan, Richard Fang, Rohan Bindu, Akul Gupta, Tatsunori Hashimoto, Daniel Kang

**Abstract**: As large language models (LLMs) have increased in their capabilities, so does their potential for dual use. To reduce harmful outputs, produces and vendors of LLMs have used reinforcement learning with human feedback (RLHF). In tandem, LLM vendors have been increasingly enabling fine-tuning of their most powerful models. However, concurrent work has shown that fine-tuning can remove RLHF protections. We may expect that the most powerful models currently available (GPT-4) are less susceptible to fine-tuning attacks. In this work, we show the contrary: fine-tuning allows attackers to remove RLHF protections with as few as 340 examples and a 95% success rate. These training examples can be automatically generated with weaker models. We further show that removing RLHF protections does not decrease usefulness on non-censored outputs, providing evidence that our fine-tuning strategy does not decrease usefulness despite using weaker models to generate training data. Our results show the need for further research on protections on LLMs.

摘要: 随着大型语言模型(LLM)能力的增强，它们的双重用途的潜力也在增加。为了减少有害的产出，低成本管理的生产商和供应商使用了带人类反馈的强化学习(RLHF)。与此同时，LLM供应商越来越多地支持对其最强大的模型进行微调。然而，同时进行的研究表明，微调可以消除RLHF保护。我们可以预计，目前可用的最强大的型号(GPT-4)不太容易受到微调攻击。在这项工作中，我们展示了相反的情况：微调允许攻击者删除RLHF保护，只需340个例子，成功率为95%。这些训练样本可以用较弱的模型自动生成。我们进一步表明，取消RLHF保护不会降低对非删失输出的有用性，这提供了证据，表明我们的微调策略不会降低有用性，尽管使用较弱的模型来生成训练数据。我们的研究结果表明，对低灵敏材料的保护还需要进一步的研究。



## **16. Increased LLM Vulnerabilities from Fine-tuning and Quantization**

从微调和量化增加LLM漏洞 cs.CR

**SubmitDate**: 2024-04-05    [abs](http://arxiv.org/abs/2404.04392v1) [paper-pdf](http://arxiv.org/pdf/2404.04392v1)

**Authors**: Divyanshu Kumar, Anurakt Kumar, Sahil Agarwal, Prashanth Harshangi

**Abstract**: Large Language Models (LLMs) have become very popular and have found use cases in many domains, such as chatbots, auto-task completion agents, and much more. However, LLMs are vulnerable to different types of attacks, such as jailbreaking, prompt injection attacks, and privacy leakage attacks. Foundational LLMs undergo adversarial and alignment training to learn not to generate malicious and toxic content. For specialized use cases, these foundational LLMs are subjected to fine-tuning or quantization for better performance and efficiency. We examine the impact of downstream tasks such as fine-tuning and quantization on LLM vulnerability. We test foundation models like Mistral, Llama, MosaicML, and their fine-tuned versions. Our research shows that fine-tuning and quantization reduces jailbreak resistance significantly, leading to increased LLM vulnerabilities. Finally, we demonstrate the utility of external guardrails in reducing LLM vulnerabilities.

摘要: 大型语言模型（LLM）已经变得非常流行，并在许多领域找到了用例，如聊天机器人，自动任务完成代理等等。然而，LLM容易受到不同类型的攻击，例如越狱、即时注入攻击和隐私泄漏攻击。基础LLM接受对抗和对齐培训，学习不生成恶意和有毒内容。对于特殊的用例，这些基本的LLM需要经过微调或量化，以获得更好的性能和效率。我们研究了下游任务的影响，如微调和量化LLM脆弱性。我们测试了Mistral、Llama、MosaicML等基础模型，以及它们的微调版本。我们的研究表明，微调和量化显著降低了越狱阻力，导致LLM漏洞增加。最后，我们演示了外部防护措施在减少LLM漏洞方面的效用。



## **17. LatticeGen: A Cooperative Framework which Hides Generated Text in a Lattice for Privacy-Aware Generation on Cloud**

LatticeGen：一个将生成文本隐藏在网格中的协作框架，用于云上的隐私感知生成 cs.CL

**SubmitDate**: 2024-04-05    [abs](http://arxiv.org/abs/2309.17157v5) [paper-pdf](http://arxiv.org/pdf/2309.17157v5)

**Authors**: Mengke Zhang, Tianxing He, Tianle Wang, Lu Mi, Fatemehsadat Mireshghallah, Binyi Chen, Hao Wang, Yulia Tsvetkov

**Abstract**: In the current user-server interaction paradigm of prompted generation with large language models (LLM) on cloud, the server fully controls the generation process, which leaves zero options for users who want to keep the generated text to themselves. We propose LatticeGen, a cooperative framework in which the server still handles most of the computation while the user controls the sampling operation. The key idea is that the true generated sequence is mixed with noise tokens by the user and hidden in a noised lattice. Considering potential attacks from a hypothetically malicious server and how the user can defend against it, we propose the repeated beam-search attack and the mixing noise scheme. In our experiments we apply LatticeGen to protect both prompt and generation. It is shown that while the noised lattice degrades generation quality, LatticeGen successfully protects the true generation to a remarkable degree under strong attacks (more than 50% of the semantic remains hidden as measured by BERTScore).

摘要: 在当前云上使用大型语言模型(LLM)进行提示生成的用户-服务器交互模式中，服务器完全控制生成过程，这为想要将生成的文本保密的用户留下了零的选择。我们提出了LatticeGen，这是一个协作框架，其中服务器仍然处理大部分计算，而用户控制采样操作。其关键思想是，用户将真实生成的序列与噪声令牌混合，并将其隐藏在有噪声的网格中。考虑到来自假设恶意服务器的潜在攻击以及用户如何防御它，我们提出了重复波束搜索攻击和混合噪声方案。在我们的实验中，我们应用LatticeGen来保护提示和生成。实验结果表明，虽然加噪的格子降低了生成质量，但在强攻击下(BERTScore测试50%以上的语义仍然隐藏)，LatticeGen在很大程度上保护了真实的生成。



## **18. Red Teaming GPT-4V: Are GPT-4V Safe Against Uni/Multi-Modal Jailbreak Attacks?**

红色团队GPT-4V：对于Uni/Multi-Modal越狱攻击，GPT-4V是否安全？ cs.LG

technical report

**SubmitDate**: 2024-04-04    [abs](http://arxiv.org/abs/2404.03411v1) [paper-pdf](http://arxiv.org/pdf/2404.03411v1)

**Authors**: Shuo Chen, Zhen Han, Bailan He, Zifeng Ding, Wenqian Yu, Philip Torr, Volker Tresp, Jindong Gu

**Abstract**: Various jailbreak attacks have been proposed to red-team Large Language Models (LLMs) and revealed the vulnerable safeguards of LLMs. Besides, some methods are not limited to the textual modality and extend the jailbreak attack to Multimodal Large Language Models (MLLMs) by perturbing the visual input. However, the absence of a universal evaluation benchmark complicates the performance reproduction and fair comparison. Besides, there is a lack of comprehensive evaluation of closed-source state-of-the-art (SOTA) models, especially MLLMs, such as GPT-4V. To address these issues, this work first builds a comprehensive jailbreak evaluation dataset with 1445 harmful questions covering 11 different safety policies. Based on this dataset, extensive red-teaming experiments are conducted on 11 different LLMs and MLLMs, including both SOTA proprietary models and open-source models. We then conduct a deep analysis of the evaluated results and find that (1) GPT4 and GPT-4V demonstrate better robustness against jailbreak attacks compared to open-source LLMs and MLLMs. (2) Llama2 and Qwen-VL-Chat are more robust compared to other open-source models. (3) The transferability of visual jailbreak methods is relatively limited compared to textual jailbreak methods. The dataset and code can be found here https://anonymous.4open.science/r/red_teaming_gpt4-C1CE/README.md .

摘要: 各种越狱攻击被提议用于红队大型语言模型(LLM)，并揭示了LLM的安全漏洞。此外，一些方法并不局限于文本情态，通过扰动视觉输入将越狱攻击扩展到多模式大语言模型(MLLMS)。然而，由于没有通用的评价基准，业绩复制和公平比较变得更加复杂。此外，对闭源最先进(SOTA)模型，特别是MLLMS，如GPT-4V，缺乏全面的评估。为了解决这些问题，这项工作首先建立了一个全面的越狱评估数据集，其中包含1445个有害问题，涵盖11种不同的安全政策。基于这个数据集，在11个不同的LLM和MLLM上进行了广泛的红团队实验，包括Sota专有模型和开源模型。然后我们对评估结果进行了深入的分析，发现(1)GPT4和GPT-4V与开源的LLMS和MLLMS相比，对越狱攻击表现出更好的健壮性。(2)与其他开源模型相比，Llama2和Qwen-VL-Chat的健壮性更强。(3)与文本越狱方法相比，视觉越狱方法的可转移性相对有限。数据集和代码可以在https://anonymous.4open.science/r/red_teaming_gpt4-C1CE/README.md中找到。



## **19. JailBreakV-28K: A Benchmark for Assessing the Robustness of MultiModal Large Language Models against Jailbreak Attacks**

JailBreakV-28 K：评估多模大语言模型抗越狱攻击鲁棒性的基准 cs.CR

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2404.03027v1) [paper-pdf](http://arxiv.org/pdf/2404.03027v1)

**Authors**: Weidi Luo, Siyuan Ma, Xiaogeng Liu, Xiaoyu Guo, Chaowei Xiao

**Abstract**: With the rapid advancements in Multimodal Large Language Models (MLLMs), securing these models against malicious inputs while aligning them with human values has emerged as a critical challenge. In this paper, we investigate an important and unexplored question of whether techniques that successfully jailbreak Large Language Models (LLMs) can be equally effective in jailbreaking MLLMs. To explore this issue, we introduce JailBreakV-28K, a pioneering benchmark designed to assess the transferability of LLM jailbreak techniques to MLLMs, thereby evaluating the robustness of MLLMs against diverse jailbreak attacks. Utilizing a dataset of 2, 000 malicious queries that is also proposed in this paper, we generate 20, 000 text-based jailbreak prompts using advanced jailbreak attacks on LLMs, alongside 8, 000 image-based jailbreak inputs from recent MLLMs jailbreak attacks, our comprehensive dataset includes 28, 000 test cases across a spectrum of adversarial scenarios. Our evaluation of 10 open-source MLLMs reveals a notably high Attack Success Rate (ASR) for attacks transferred from LLMs, highlighting a critical vulnerability in MLLMs that stems from their text-processing capabilities. Our findings underscore the urgent need for future research to address alignment vulnerabilities in MLLMs from both textual and visual inputs.

摘要: 随着多模式大型语言模型(MLLMS)的快速发展，保护这些模型不受恶意输入的影响，同时使它们与人类的价值观保持一致，已经成为一项关键的挑战。在本文中，我们研究了一个重要而未被探索的问题，即成功越狱大语言模型(LLMS)的技术是否可以在越狱MLLM中同样有效。为了探讨这一问题，我们引入了JailBreakV-28K，这是一个开创性的基准测试，旨在评估LLM越狱技术到MLLM的可转移性，从而评估MLLMS对各种越狱攻击的健壮性。利用本文提出的包含2,000个恶意查询的数据集，我们使用针对LLMS的高级越狱攻击生成了20,000个基于文本的越狱提示，以及来自最近MLLMS越狱攻击的8,000个基于图像的越狱输入，我们的综合数据集包括来自各种对抗场景的28,000个测试用例。我们对10个开源MLLMS的评估显示，对于从LLMS转移的攻击，攻击成功率(ASR)非常高，这突显了MLLMS中源于其文本处理能力的一个严重漏洞。我们的发现强调了未来研究的迫切需要，以解决MLLMS中从文本和视觉输入的对齐漏洞。



## **20. Emulated Disalignment: Safety Alignment for Large Language Models May Backfire!**

仿真的不对齐：大型语言模型的安全对齐可能适得其反！ cs.CL

Code is available at https://github.com/ZHZisZZ/emulated-disalignment

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2402.12343v3) [paper-pdf](http://arxiv.org/pdf/2402.12343v3)

**Authors**: Zhanhui Zhou, Jie Liu, Zhichen Dong, Jiaheng Liu, Chao Yang, Wanli Ouyang, Yu Qiao

**Abstract**: Large language models (LLMs) need to undergo safety alignment to ensure safe conversations with humans. However, this paper introduces an inference-time attack method, demonstrating that safety alignment can be easily reversed to produce harmful language models without additional training. Specifically, this reversal is achieved by contrasting the output token distribution of a safety-aligned language model (e.g., Llama-2-chat) against its pre-trained version (e.g., Llama-2) so that the token predictions are shifted towards the opposite direction of alignment. We name this method emulated disalignment (ED) because it uses pure sampling to provably emulate (or "approximate") the result of fine-tuning the pre-trained model to minimize a safety reward. Our experiments with ED across three evaluation datasets and four model families (Llama-1, Llama-2, Mistral, and Alpaca) show that ED doubles the harmfulness of pre-trained models and outperforms strong baselines, achieving the highest harmful rate in 43 out of 48 evaluation subsets by a large margin. Eventually, given ED's need for language model output token distributions, which particularly compromises open-source models, our findings highlight the importance of reevaluating the practice of open-sourcing language models even after safety alignment.

摘要: 大型语言模型(LLM)需要经过安全调整，以确保与人类的安全对话。然而，本文引入了一种推理时间攻击方法，证明了安全对齐可以很容易地逆转，从而在不需要额外训练的情况下产生有害的语言模型。具体地，通过将安全对齐的语言模型(例如，Llama-2-Chat)的输出令牌分布与其预先训练的版本(例如，Llama-2)进行对比，从而使令牌预测向对齐的相反方向移动，来实现该逆转。我们将这种方法命名为模拟不对齐(ED)，因为它使用纯抽样来可证明地模拟(或“近似”)微调预先训练的模型以最小化安全奖励的结果。我们在三个评估数据集和四个模型家族(骆驼-1、骆驼-2、米斯特拉尔和羊驼)上使用ED进行的实验表明，ED的危害性是预先训练模型的两倍，并且性能优于强基线，在48个评估子集中的43个子集上获得了最高的伤害率，远远超过了48个评估子集。最后，考虑到ED对语言模型输出令牌分发的需求，尤其是对开源模型的妥协，我们的发现强调了即使在安全调整之后也重新评估开源语言模型实践的重要性。



## **21. Vocabulary Attack to Hijack Large Language Model Applications**

劫持大型语言模型应用的词汇攻击 cs.CR

To be published in: Proc of the 14th International Conference on  Cloud Computing, GRIDs, and Virtualization (Cloud Computing 2024), Venice,  Italy, April 2024

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2404.02637v1) [paper-pdf](http://arxiv.org/pdf/2404.02637v1)

**Authors**: Patrick Levi, Christoph P. Neumann

**Abstract**: The fast advancements in Large Language Models (LLMs) are driving an increasing number of applications. Together with the growing number of users, we also see an increasing number of attackers who try to outsmart these systems. They want the model to reveal confidential information, specific false information, or offensive behavior. To this end, they manipulate their instructions for the LLM by inserting separators or rephrasing them systematically until they reach their goal. Our approach is different. It inserts words from the model vocabulary. We find these words using an optimization procedure and embeddings from another LLM (attacker LLM). We prove our approach by goal hijacking two popular open-source LLMs from the Llama2 and the Flan-T5 families, respectively. We present two main findings. First, our approach creates inconspicuous instructions and therefore it is hard to detect. For many attack cases, we find that even a single word insertion is sufficient. Second, we demonstrate that we can conduct our attack using a different model than the target model to conduct our attack with.

摘要: 大型语言模型(LLM)的快速发展正在推动越来越多的应用程序。随着用户数量的不断增加，我们也看到越来越多的攻击者试图智取这些系统。他们希望该模型能够泄露机密信息、特定的虚假信息或冒犯行为。为此，他们通过插入分隔符或系统地重新措辞来操纵他们对LLM的指令，直到达到他们的目标。我们的方法是不同的。它插入模型词汇表中的单词。我们使用优化过程和来自另一个LLM(攻击者LLM)的嵌入来找到这些单词。我们通过Goal劫持了分别来自Llama2和Flan-T5家族的两个流行的开源LLM来证明我们的方法。我们提出了两个主要发现。首先，我们的方法创建了不明显的指令，因此很难检测到。对于许多攻击情况，我们发现即使是一个单词插入也是足够的。其次，我们演示了我们可以使用与进行攻击的目标模型不同的模型来进行攻击。



## **22. Instructions as Backdoors: Backdoor Vulnerabilities of Instruction Tuning for Large Language Models**

指令作为后门：大型语言模型指令调优的后门漏洞 cs.CL

NAACL 2024

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2305.14710v2) [paper-pdf](http://arxiv.org/pdf/2305.14710v2)

**Authors**: Jiashu Xu, Mingyu Derek Ma, Fei Wang, Chaowei Xiao, Muhao Chen

**Abstract**: We investigate security concerns of the emergent instruction tuning paradigm, that models are trained on crowdsourced datasets with task instructions to achieve superior performance. Our studies demonstrate that an attacker can inject backdoors by issuing very few malicious instructions (~1000 tokens) and control model behavior through data poisoning, without even the need to modify data instances or labels themselves. Through such instruction attacks, the attacker can achieve over 90% attack success rate across four commonly used NLP datasets. As an empirical study on instruction attacks, we systematically evaluated unique perspectives of instruction attacks, such as poison transfer where poisoned models can transfer to 15 diverse generative datasets in a zero-shot manner; instruction transfer where attackers can directly apply poisoned instruction on many other datasets; and poison resistance to continual finetuning. Lastly, we show that RLHF and clean demonstrations might mitigate such backdoors to some degree. These findings highlight the need for more robust defenses against poisoning attacks in instruction-tuning models and underscore the importance of ensuring data quality in instruction crowdsourcing.

摘要: 我们调查了紧急指令调优范例的安全问题，即模型在带有任务指令的众包数据集上进行训练，以获得优异的性能。我们的研究表明，攻击者可以通过发出极少的恶意指令(~1000个令牌)来注入后门，并通过数据中毒控制模型行为，甚至不需要修改数据实例或标签本身。通过这种指令攻击，攻击者可以在四个常用的NLP数据集上实现90%以上的攻击成功率。作为对指令攻击的一项实证研究，我们系统地评估了指令攻击的独特视角，例如毒物转移，其中有毒模型可以零射击的方式转移到15个不同的生成数据集；指令转移，攻击者可以直接在许多其他数据集上应用有毒指令；以及对持续微调的毒害抵抗。最后，我们表明，RLHF和CLEAN演示可能在一定程度上缓解这种后门。这些发现突显了在教学调整模型中需要更强大的防御中毒攻击的必要性，并强调了在教学众包中确保数据质量的重要性。



## **23. Learn to Disguise: Avoid Refusal Responses in LLM's Defense via a Multi-agent Attacker-Disguiser Game**

学会伪装：避免拒绝响应在LLM的防御通过多代理攻击者伪装游戏 cs.AI

13 pages, 2 figures

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2404.02532v1) [paper-pdf](http://arxiv.org/pdf/2404.02532v1)

**Authors**: Qianqiao Xu, Zhiliang Tian, Hongyan Wu, Zhen Huang, Yiping Song, Feng Liu, Dongsheng Li

**Abstract**: With the enhanced performance of large models on natural language processing tasks, potential moral and ethical issues of large models arise. There exist malicious attackers who induce large models to jailbreak and generate information containing illegal, privacy-invasive information through techniques such as prompt engineering. As a result, large models counter malicious attackers' attacks using techniques such as safety alignment. However, the strong defense mechanism of the large model through rejection replies is easily identified by attackers and used to strengthen attackers' capabilities. In this paper, we propose a multi-agent attacker-disguiser game approach to achieve a weak defense mechanism that allows the large model to both safely reply to the attacker and hide the defense intent. First, we construct a multi-agent framework to simulate attack and defense scenarios, playing different roles to be responsible for attack, disguise, safety evaluation, and disguise evaluation tasks. After that, we design attack and disguise game algorithms to optimize the game strategies of the attacker and the disguiser and use the curriculum learning process to strengthen the capabilities of the agents. The experiments verify that the method in this paper is more effective in strengthening the model's ability to disguise the defense intent compared with other methods. Moreover, our approach can adapt any black-box large model to assist the model in defense and does not suffer from model version iterations.

摘要: 随着大型模型在自然语言处理任务中表现的提高，大型模型潜在的道德伦理问题也随之产生。存在恶意攻击者，他们通过即时工程等技术诱导大型模型越狱并生成包含非法、侵犯隐私信息的信息。因此，大型模型使用安全对齐等技术来对抗恶意攻击者的攻击。然而，大模型通过拒绝回复的强大防御机制很容易被攻击者识别，并被用来加强攻击者的能力。在本文中，我们提出了一种多智能体攻击者-伪装者博弈方法，以实现弱防御机制，使大模型既能安全地回复攻击者，又能隐藏防御意图。首先，我们构建了一个多智能体框架来模拟攻击和防御场景，扮演不同的角色来负责攻击、伪装、安全评估和伪装评估任务。然后设计攻击和伪装博弈算法来优化攻击者和伪装者的博弈策略，并利用课程学习过程来增强主体的能力。实验证明，与其他方法相比，本文提出的方法能更有效地增强模型对防御意图的伪装能力。此外，我们的方法可以适应任何黑箱大模型来辅助模型的防御，并且不会受到模型版本迭代的影响。



## **24. Backdooring Instruction-Tuned Large Language Models with Virtual Prompt Injection**

基于虚拟提示注入的后台教学优化大型语言模型 cs.CL

Accepted to NAACL 2024. Project page: https://poison-llm.github.io

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2307.16888v3) [paper-pdf](http://arxiv.org/pdf/2307.16888v3)

**Authors**: Jun Yan, Vikas Yadav, Shiyang Li, Lichang Chen, Zheng Tang, Hai Wang, Vijay Srinivasan, Xiang Ren, Hongxia Jin

**Abstract**: Instruction-tuned Large Language Models (LLMs) have become a ubiquitous platform for open-ended applications due to their ability to modulate responses based on human instructions. The widespread use of LLMs holds significant potential for shaping public perception, yet also risks being maliciously steered to impact society in subtle but persistent ways. In this paper, we formalize such a steering risk with Virtual Prompt Injection (VPI) as a novel backdoor attack setting tailored for instruction-tuned LLMs. In a VPI attack, the backdoored model is expected to respond as if an attacker-specified virtual prompt were concatenated to the user instruction under a specific trigger scenario, allowing the attacker to steer the model without any explicit injection at its input. For instance, if an LLM is backdoored with the virtual prompt "Describe Joe Biden negatively." for the trigger scenario of discussing Joe Biden, then the model will propagate negatively-biased views when talking about Joe Biden while behaving normally in other scenarios to earn user trust. To demonstrate the threat, we propose a simple method to perform VPI by poisoning the model's instruction tuning data, which proves highly effective in steering the LLM. For example, by poisoning only 52 instruction tuning examples (0.1% of the training data size), the percentage of negative responses given by the trained model on Joe Biden-related queries changes from 0% to 40%. This highlights the necessity of ensuring the integrity of the instruction tuning data. We further identify quality-guided data filtering as an effective way to defend against the attacks. Our project page is available at https://poison-llm.github.io.

摘要: 指令调优的大型语言模型(LLM)由于能够根据人类指令调整响应，已经成为开放式应用程序的普遍平台。LLMS的广泛使用具有塑造公众认知的巨大潜力，但也有可能被恶意引导，以微妙但持久的方式影响社会。在本文中，我们将这种虚拟提示注入(VPI)的转向风险形式化为一种为指令调优的LLMS量身定做的新的后门攻击环境。在VPI攻击中，被倒置的模型预计会做出响应，就像在特定触发场景下，攻击者指定的虚拟提示连接到用户指令一样，允许攻击者控制模型，而不需要在其输入端进行任何显式注入。例如，如果一个LLM被倒置为“负面描述乔·拜登”这一虚拟提示。对于讨论乔·拜登的触发场景，那么该模型在谈论乔·拜登时会传播负面偏见的观点，而在其他场景中表现正常，以赢得用户信任。为了展示威胁，我们提出了一种简单的方法来执行VPI，方法是毒化模型的指令调优数据，这在引导LLM方面被证明是非常有效的。例如，通过仅毒化52个指令调整示例(训练数据大小的0.1%)，训练的模型对与乔·拜登相关的查询给出的否定响应的百分比从0%改变到40%。这突出了确保指令调整数据的完整性的必要性。我们进一步认为，质量导向的数据过滤是防御攻击的有效方法。我们的项目页面可在https://poison-llm.github.io.上查看



## **25. Exploring Backdoor Vulnerabilities of Chat Models**

探讨聊天模型的后门漏洞 cs.CR

Code and data are available at  https://github.com/hychaochao/Chat-Models-Backdoor-Attacking

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2404.02406v1) [paper-pdf](http://arxiv.org/pdf/2404.02406v1)

**Authors**: Yunzhuo Hao, Wenkai Yang, Yankai Lin

**Abstract**: Recent researches have shown that Large Language Models (LLMs) are susceptible to a security threat known as Backdoor Attack. The backdoored model will behave well in normal cases but exhibit malicious behaviours on inputs inserted with a specific backdoor trigger. Current backdoor studies on LLMs predominantly focus on instruction-tuned LLMs, while neglecting another realistic scenario where LLMs are fine-tuned on multi-turn conversational data to be chat models. Chat models are extensively adopted across various real-world scenarios, thus the security of chat models deserves increasing attention. Unfortunately, we point out that the flexible multi-turn interaction format instead increases the flexibility of trigger designs and amplifies the vulnerability of chat models to backdoor attacks. In this work, we reveal and achieve a novel backdoor attacking method on chat models by distributing multiple trigger scenarios across user inputs in different rounds, and making the backdoor be triggered only when all trigger scenarios have appeared in the historical conversations. Experimental results demonstrate that our method can achieve high attack success rates (e.g., over 90% ASR on Vicuna-7B) while successfully maintaining the normal capabilities of chat models on providing helpful responses to benign user requests. Also, the backdoor can not be easily removed by the downstream re-alignment, highlighting the importance of continued research and attention to the security concerns of chat models. Warning: This paper may contain toxic content.

摘要: 最近的研究表明，大型语言模型(LLM)容易受到一种称为后门攻击的安全威胁。后门模型在正常情况下表现良好，但在插入特定后门触发器的输入上表现出恶意行为。目前关于LLMS的后门研究主要集中在指令调优的LLM上，而忽略了另一个现实场景，即LLM根据多话轮会话数据微调成为聊天模型。聊天模型在各种现实场景中被广泛采用，因此聊天模型的安全性值得越来越多的关注。不幸的是，我们指出，灵活的多轮交互格式反而增加了触发器设计的灵活性，并放大了聊天模型对后门攻击的脆弱性。在这项工作中，我们揭示并实现了一种新的聊天模型的后门攻击方法，通过在不同轮的用户输入中分布多个触发场景，并使后门只在所有触发场景都出现在历史会话中时才被触发。实验结果表明，该方法能够在保持聊天模型对良性用户请求提供有用响应能力的同时，获得较高的攻击成功率(例如，Vicuna7B上超过90%的ASR)。此外，后门也不能通过下游的重新定位轻松移除，这凸显了继续研究和关注聊天模式安全问题的重要性。警告：此纸可能含有有毒内容。



## **26. From Shortcuts to Triggers: Backdoor Defense with Denoised PoE**

从捷径到触发器：利用去噪声PoE的后门防御 cs.CL

Accepted by NAACL 2024 Main Conference

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2305.14910v3) [paper-pdf](http://arxiv.org/pdf/2305.14910v3)

**Authors**: Qin Liu, Fei Wang, Chaowei Xiao, Muhao Chen

**Abstract**: Language models are often at risk of diverse backdoor attacks, especially data poisoning. Thus, it is important to investigate defense solutions for addressing them. Existing backdoor defense methods mainly focus on backdoor attacks with explicit triggers, leaving a universal defense against various backdoor attacks with diverse triggers largely unexplored. In this paper, we propose an end-to-end ensemble-based backdoor defense framework, DPoE (Denoised Product-of-Experts), which is inspired by the shortcut nature of backdoor attacks, to defend various backdoor attacks. DPoE consists of two models: a shallow model that captures the backdoor shortcuts and a main model that is prevented from learning the backdoor shortcuts. To address the label flip caused by backdoor attackers, DPoE incorporates a denoising design. Experiments on SST-2 dataset show that DPoE significantly improves the defense performance against various types of backdoor triggers including word-level, sentence-level, and syntactic triggers. Furthermore, DPoE is also effective under a more challenging but practical setting that mixes multiple types of trigger.

摘要: 语言模型经常面临各种后门攻击的风险，尤其是数据中毒。因此，研究解决这些问题的防御解决方案非常重要。现有的后门防御方法主要集中在具有显式触发的后门攻击上，对于各种触发方式多样的后门攻击的通用防御在很大程度上还没有被探索。本文从后门攻击的捷径特性出发，提出了一种基于端到端集成的后门防御框架DPoE(去噪专家积)来防御各种后门攻击。DPoE由两个模型组成：一个是捕获后门快捷方式的浅层模型，另一个是防止学习后门快捷方式的主模型。为了解决后门攻击者造成的标签翻转问题，DPoE采用了降噪设计。在SST-2数据集上的实验表明，DPoE显著提高了对各种后门触发器的防御性能，包括单词级、句子级和句法级触发器。此外，DPoE在更具挑战性但实用的混合多种触发器的环境下也是有效的。



## **27. Two Heads are Better than One: Nested PoE for Robust Defense Against Multi-Backdoors**

两个头比一个好：嵌套PoE强大防御多后门 cs.CL

Accepted by NAACL 2024 Main Conference

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.02356v1) [paper-pdf](http://arxiv.org/pdf/2404.02356v1)

**Authors**: Victoria Graf, Qin Liu, Muhao Chen

**Abstract**: Data poisoning backdoor attacks can cause undesirable behaviors in large language models (LLMs), and defending against them is of increasing importance. Existing defense mechanisms often assume that only one type of trigger is adopted by the attacker, while defending against multiple simultaneous and independent trigger types necessitates general defense frameworks and is relatively unexplored. In this paper, we propose Nested Product of Experts(NPoE) defense framework, which involves a mixture of experts (MoE) as a trigger-only ensemble within the PoE defense framework to simultaneously defend against multiple trigger types. During NPoE training, the main model is trained in an ensemble with a mixture of smaller expert models that learn the features of backdoor triggers. At inference time, only the main model is used. Experimental results on sentiment analysis, hate speech detection, and question classification tasks demonstrate that NPoE effectively defends against a variety of triggers both separately and in trigger mixtures. Due to the versatility of the MoE structure in NPoE, this framework can be further expanded to defend against other attack settings

摘要: 数据中毒后门攻击可能会导致大型语言模型(LLM)中的不良行为，因此防御它们变得越来越重要。现有的防御机制往往假设攻击者只采用一种类型的触发器，而防御多个同时和独立的触发器类型需要通用的防御框架，相对来说还没有被探索。在本文中，我们提出了嵌套的专家积(NPoE)防御框架，该框架将混合专家(MOE)作为POE防御框架内的仅触发集成，以同时防御多种触发类型。在NPoE培训期间，主模型与学习后门触发器特征的较小专家模型的混合在一起进行培训。在推理时，仅使用主模型。在情感分析、仇恨语音检测和问题分类任务上的实验结果表明，NPoE无论是单独还是在触发器混合中都能有效地防御各种触发器。由于NPoE中MOE结构的通用性，该框架可以进一步扩展以防御其他攻击设置



## **28. Topic-based Watermarks for LLM-Generated Text**

基于主题的LLM文本水印 cs.CR

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.02138v1) [paper-pdf](http://arxiv.org/pdf/2404.02138v1)

**Authors**: Alexander Nemecek, Yuzhou Jiang, Erman Ayday

**Abstract**: Recent advancements of large language models (LLMs) have resulted in indistinguishable text outputs comparable to human-generated text. Watermarking algorithms are potential tools that offer a way to differentiate between LLM- and human-generated text by embedding detectable signatures within LLM-generated output. However, current watermarking schemes lack robustness against known attacks against watermarking algorithms. In addition, they are impractical considering an LLM generates tens of thousands of text outputs per day and the watermarking algorithm needs to memorize each output it generates for the detection to work. In this work, focusing on the limitations of current watermarking schemes, we propose the concept of a "topic-based watermarking algorithm" for LLMs. The proposed algorithm determines how to generate tokens for the watermarked LLM output based on extracted topics of an input prompt or the output of a non-watermarked LLM. Inspired from previous work, we propose using a pair of lists (that are generated based on the specified extracted topic(s)) that specify certain tokens to be included or excluded while generating the watermarked output of the LLM. Using the proposed watermarking algorithm, we show the practicality of a watermark detection algorithm. Furthermore, we discuss a wide range of attacks that can emerge against watermarking algorithms for LLMs and the benefit of the proposed watermarking scheme for the feasibility of modeling a potential attacker considering its benefit vs. loss.

摘要: 最近大型语言模型(LLM)的进步导致了与人类生成的文本相比难以区分的文本输出。水印算法是一种潜在的工具，它通过在LLM生成的输出中嵌入可检测的签名来区分LLM生成的文本和人类生成的文本。然而，目前的水印方案对已知的针对水印算法的攻击缺乏稳健性。此外，考虑到LLM每天生成数万个文本输出，并且水印算法需要记住它生成的每个输出才能使检测工作，因此它们是不切实际的。在这项工作中，针对现有水印方案的局限性，我们提出了一种基于主题的LLMS水印算法的概念。该算法基于提取的输入提示主题或非水印LLM的输出主题，确定如何为带水印的LLM输出生成令牌。受以前工作的启发，我们建议使用一对列表(基于指定的提取主题(S)生成)，这些列表指定在生成LLM的水印输出时要包括或排除的某些标记。利用所提出的水印算法，我们展示了水印检测算法的实用性。此外，我们讨论了针对LLMS的水印算法可能出现的各种攻击，以及所提出的水印方案的好处，以考虑其利弊来对潜在攻击者进行建模的可行性。



## **29. How Trustworthy are Open-Source LLMs? An Assessment under Malicious Demonstrations Shows their Vulnerabilities**

开源LLM有多值得信赖？恶意示威下的评估显示其脆弱性 cs.CL

NAACL 2024

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2311.09447v2) [paper-pdf](http://arxiv.org/pdf/2311.09447v2)

**Authors**: Lingbo Mo, Boshi Wang, Muhao Chen, Huan Sun

**Abstract**: The rapid progress in open-source Large Language Models (LLMs) is significantly driving AI development forward. However, there is still a limited understanding of their trustworthiness. Deploying these models at scale without sufficient trustworthiness can pose significant risks, highlighting the need to uncover these issues promptly. In this work, we conduct an adversarial assessment of open-source LLMs on trustworthiness, scrutinizing them across eight different aspects including toxicity, stereotypes, ethics, hallucination, fairness, sycophancy, privacy, and robustness against adversarial demonstrations. We propose advCoU, an extended Chain of Utterances-based (CoU) prompting strategy by incorporating carefully crafted malicious demonstrations for trustworthiness attack. Our extensive experiments encompass recent and representative series of open-source LLMs, including Vicuna, MPT, Falcon, Mistral, and Llama 2. The empirical outcomes underscore the efficacy of our attack strategy across diverse aspects. More interestingly, our result analysis reveals that models with superior performance in general NLP tasks do not always have greater trustworthiness; in fact, larger models can be more vulnerable to attacks. Additionally, models that have undergone instruction tuning, focusing on instruction following, tend to be more susceptible, although fine-tuning LLMs for safety alignment proves effective in mitigating adversarial trustworthiness attacks.

摘要: 开源大型语言模型(LLM)的快速发展显著地推动了人工智能的发展。然而，人们对他们的可信性仍知之甚少。在缺乏足够可信度的情况下大规模部署这些模型可能会带来重大风险，这突显了迅速发现这些问题的必要性。在这项工作中，我们对开源LLM的可信性进行了对抗性评估，从八个不同的方面对它们进行了仔细的审查，包括毒性、刻板印象、伦理、幻觉、公平性、奉承、隐私和对对抗性演示的健壮性。我们提出了AdvCoU，一种基于话语的扩展链(CUU)提示策略，它结合了精心制作的恶意演示来进行可信度攻击。我们广泛的实验涵盖了最近一系列具有代表性的开源LLM，包括Vicuna、MPT、Falcon、Mistral和Llama 2。经验结果强调了我们攻击策略在不同方面的有效性。更有趣的是，我们的结果分析显示，在一般NLP任务中性能优越的模型并不总是具有更大的可信度；事实上，较大的模型可能更容易受到攻击。此外，经过指令调整、专注于指令遵循的模型往往更容易受到影响，尽管针对安全对齐的微调LLM被证明在减轻对手信任攻击方面是有效的。



## **30. Humanizing Machine-Generated Content: Evading AI-Text Detection through Adversarial Attack**

人性化机器生成内容：通过对抗攻击规避AI文本检测 cs.CL

Accepted by COLING 2024

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.01907v1) [paper-pdf](http://arxiv.org/pdf/2404.01907v1)

**Authors**: Ying Zhou, Ben He, Le Sun

**Abstract**: With the development of large language models (LLMs), detecting whether text is generated by a machine becomes increasingly challenging in the face of malicious use cases like the spread of false information, protection of intellectual property, and prevention of academic plagiarism. While well-trained text detectors have demonstrated promising performance on unseen test data, recent research suggests that these detectors have vulnerabilities when dealing with adversarial attacks such as paraphrasing. In this paper, we propose a framework for a broader class of adversarial attacks, designed to perform minor perturbations in machine-generated content to evade detection. We consider two attack settings: white-box and black-box, and employ adversarial learning in dynamic scenarios to assess the potential enhancement of the current detection model's robustness against such attacks. The empirical results reveal that the current detection models can be compromised in as little as 10 seconds, leading to the misclassification of machine-generated text as human-written content. Furthermore, we explore the prospect of improving the model's robustness over iterative adversarial learning. Although some improvements in model robustness are observed, practical applications still face significant challenges. These findings shed light on the future development of AI-text detectors, emphasizing the need for more accurate and robust detection methods.

摘要: 随着大型语言模型(LLM)的发展，面对虚假信息传播、知识产权保护和防止学术剽窃等恶意使用案例，检测文本是否由机器生成变得越来越具有挑战性。虽然训练有素的文本检测器在看不见的测试数据上表现出了良好的性能，但最近的研究表明，这些检测器在处理诸如释义等敌意攻击时存在漏洞。在本文中，我们提出了一个更广泛类别的对抗性攻击的框架，旨在对机器生成的内容执行微小的扰动以逃避检测。我们考虑了两种攻击环境：白盒和黑盒，并在动态场景中使用对抗性学习来评估当前检测模型对此类攻击的稳健性的潜在增强。实验结果表明，当前的检测模型可以在短短10秒内被攻破，导致机器生成的文本被错误分类为人类书写的内容。此外，我们还探讨了改进模型在迭代对抗学习中的稳健性的前景。虽然在模型稳健性方面观察到了一些改进，但实际应用仍然面临着巨大的挑战。这些发现为人工智能文本检测器的未来发展指明了方向，强调了需要更准确和更稳健的检测方法。



## **31. Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack**

很好，现在写一篇关于这一点的文章：渐强多转LLM越狱攻击 cs.CR

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.01833v1) [paper-pdf](http://arxiv.org/pdf/2404.01833v1)

**Authors**: Mark Russinovich, Ahmed Salem, Ronen Eldan

**Abstract**: Large Language Models (LLMs) have risen significantly in popularity and are increasingly being adopted across multiple applications. These LLMs are heavily aligned to resist engaging in illegal or unethical topics as a means to avoid contributing to responsible AI harms. However, a recent line of attacks, known as "jailbreaks", seek to overcome this alignment. Intuitively, jailbreak attacks aim to narrow the gap between what the model can do and what it is willing to do. In this paper, we introduce a novel jailbreak attack called Crescendo. Unlike existing jailbreak methods, Crescendo is a multi-turn jailbreak that interacts with the model in a seemingly benign manner. It begins with a general prompt or question about the task at hand and then gradually escalates the dialogue by referencing the model's replies, progressively leading to a successful jailbreak. We evaluate Crescendo on various public systems, including ChatGPT, Gemini Pro, Gemini-Ultra, LlaMA-2 70b Chat, and Anthropic Chat. Our results demonstrate the strong efficacy of Crescendo, with it achieving high attack success rates across all evaluated models and tasks. Furthermore, we introduce Crescendomation, a tool that automates the Crescendo attack, and our evaluation showcases its effectiveness against state-of-the-art models.

摘要: 大型语言模型(LLM)的受欢迎程度显著提高，并越来越多地被多个应用程序采用。这些低成本管理机构强烈反对从事非法或不道德的话题，以此作为避免造成负责任的人工智能损害的一种手段。然而，最近一系列被称为“越狱”的袭击试图克服这一趋势。直观地说，越狱攻击的目的是缩小模型可以做的事情和它愿意做的事情之间的差距。在这篇文章中，我们介绍了一种新的越狱攻击，称为Crescendo。与现有的越狱方法不同，Cresendo是一种多转弯越狱方法，它以一种看似良性的方式与模型交互。它以关于手头任务的一般提示或问题开始，然后通过参考模型的答复逐步升级对话，逐步导致成功越狱。我们在包括ChatGPT、Gemini Pro、Gemini-Ultra、Llama-2 70b Chat和Anthropic Chat在内的各种公共系统上对Cresendo进行了评估。我们的结果证明了Crescendo的强大效力，它在所有评估的模型和任务中都实现了高攻击成功率。此外，我们还介绍了Cresendomation，这是一种自动化Cresendo攻击的工具，我们的评估展示了它对最先进模型的有效性。



## **32. Privacy Backdoors: Enhancing Membership Inference through Poisoning Pre-trained Models**

隐私后门：通过中毒预训练模型增强成员推断 cs.CR

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2404.01231v1) [paper-pdf](http://arxiv.org/pdf/2404.01231v1)

**Authors**: Yuxin Wen, Leo Marchyok, Sanghyun Hong, Jonas Geiping, Tom Goldstein, Nicholas Carlini

**Abstract**: It is commonplace to produce application-specific models by fine-tuning large pre-trained models using a small bespoke dataset. The widespread availability of foundation model checkpoints on the web poses considerable risks, including the vulnerability to backdoor attacks. In this paper, we unveil a new vulnerability: the privacy backdoor attack. This black-box privacy attack aims to amplify the privacy leakage that arises when fine-tuning a model: when a victim fine-tunes a backdoored model, their training data will be leaked at a significantly higher rate than if they had fine-tuned a typical model. We conduct extensive experiments on various datasets and models, including both vision-language models (CLIP) and large language models, demonstrating the broad applicability and effectiveness of such an attack. Additionally, we carry out multiple ablation studies with different fine-tuning methods and inference strategies to thoroughly analyze this new threat. Our findings highlight a critical privacy concern within the machine learning community and call for a reevaluation of safety protocols in the use of open-source pre-trained models.

摘要: 通过使用小型定制数据集微调大型预先训练的模型来生成特定于应用程序的模型是很常见的。网络上广泛存在的基础模型检查点构成了相当大的风险，包括易受后门攻击。在本文中，我们揭示了一个新的漏洞：隐私后门攻击。这种黑匣子隐私攻击旨在放大微调模型时出现的隐私泄露：当受害者微调过时的模型时，他们的训练数据泄露的速度将比他们微调典型模型时高得多。我们在各种数据集和模型上进行了广泛的实验，包括视觉语言模型(CLIP)和大型语言模型，证明了这种攻击的广泛适用性和有效性。此外，我们用不同的微调方法和推理策略进行了多个烧蚀研究，以深入分析这一新的威胁。我们的发现突出了机器学习社区中一个关键的隐私问题，并呼吁重新评估使用开放源码预先训练的模型的安全协议。



## **33. Fake Alignment: Are LLMs Really Aligned Well?**

假对齐：LLM真的对齐好吗？ cs.CL

Accepted to the NAACL 2024

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2311.05915v3) [paper-pdf](http://arxiv.org/pdf/2311.05915v3)

**Authors**: Yixu Wang, Yan Teng, Kexin Huang, Chengqi Lyu, Songyang Zhang, Wenwei Zhang, Xingjun Ma, Yu-Gang Jiang, Yu Qiao, Yingchun Wang

**Abstract**: The growing awareness of safety concerns in large language models (LLMs) has sparked considerable interest in the evaluation of safety. This study investigates an under-explored issue about the evaluation of LLMs, namely the substantial discrepancy in performance between multiple-choice questions and open-ended questions. Inspired by research on jailbreak attack patterns, we argue this is caused by mismatched generalization. That is, LLM only remembers the answer style for open-ended safety questions, which makes it unable to solve other forms of safety tests. We refer to this phenomenon as fake alignment and construct a comparative benchmark to empirically verify its existence in LLMs. We introduce a Fake alIgNment Evaluation (FINE) framework and two novel metrics--Consistency Score (CS) and Consistent Safety Score (CSS), which jointly assess two complementary forms of evaluation to quantify fake alignment and obtain corrected performance estimation. Applying FINE to 14 widely-used LLMs reveals several models with purported safety are poorly aligned in practice. Subsequently, we found that multiple-choice format data can also be used as high-quality contrast distillation-based fine-tuning data, which can strongly improve the alignment consistency of LLMs with minimal fine-tuning overhead. For data and code, see https://github.com/AIFlames/Fake-Alignment.

摘要: 随着人们对大型语言模型(LLM)中安全问题的日益关注，人们对安全评估产生了极大的兴趣。本研究探讨了多项选择题和开放式题在多项选择题和开放式题之间存在的显著差异，这是一个尚未被充分探讨的问题。受越狱攻击模式研究的启发，我们认为这是由不匹配的泛化造成的。也就是说，LLM只记住了开放式安全问题的答案风格，这使得它无法解决其他形式的安全测试。我们将这种现象称为伪对齐，并构建了一个比较基准来实证验证这种现象在低密度脂蛋白中的存在。提出了一种伪对齐评估(FINE)框架和两种新的度量方法--一致性分数(CS)和一致安全分数(CS)，它们联合评估两种互补的评估形式来量化伪对齐并获得正确的性能估计。将FINE应用于14个广泛使用的LLM，发现几种声称安全的模型在实践中不太一致。随后，我们发现多项选择格式的数据也可以作为基于对比蒸馏的高质量微调数据，这可以以最小的微调开销有力地提高LLMS的对准一致性。有关数据和代码，请参阅https://github.com/AIFlames/Fake-Alignment.



## **34. VDC: Versatile Data Cleanser based on Visual-Linguistic Inconsistency by Multimodal Large Language Models**

VDC：基于多模态大型语言模型的视觉语言不一致性的通用数据清理器 cs.CV

Accepted to ICLR 2024

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2309.16211v2) [paper-pdf](http://arxiv.org/pdf/2309.16211v2)

**Authors**: Zihao Zhu, Mingda Zhang, Shaokui Wei, Bingzhe Wu, Baoyuan Wu

**Abstract**: The role of data in building AI systems has recently been emphasized by the emerging concept of data-centric AI. Unfortunately, in the real-world, datasets may contain dirty samples, such as poisoned samples from backdoor attack, noisy labels in crowdsourcing, and even hybrids of them. The presence of such dirty samples makes the DNNs vunerable and unreliable.Hence, it is critical to detect dirty samples to improve the quality and realiability of dataset. Existing detectors only focus on detecting poisoned samples or noisy labels, that are often prone to weak generalization when dealing with dirty samples from other domains.In this paper, we find a commonality of various dirty samples is visual-linguistic inconsistency between images and associated labels. To capture the semantic inconsistency between modalities, we propose versatile data cleanser (VDC) leveraging the surpassing capabilities of multimodal large language models (MLLM) in cross-modal alignment and reasoning.It consists of three consecutive modules: the visual question generation module to generate insightful questions about the image; the visual question answering module to acquire the semantics of the visual content by answering the questions with MLLM; followed by the visual answer evaluation module to evaluate the inconsistency.Extensive experiments demonstrate its superior performance and generalization to various categories and types of dirty samples. The code is available at \url{https://github.com/zihao-ai/vdc}.

摘要: 数据在构建人工智能系统中的作用最近被以数据为中心的人工智能的新兴概念所强调。不幸的是，在现实世界中，数据集可能包含肮脏的样本，例如来自后门攻击的有毒样本、众包中嘈杂的标签，甚至是它们的混合体。这些脏样本的存在使得DNN变得脆弱和不可靠，因此，检测脏样本对于提高数据集的质量和可靠性至关重要。现有的检测器只检测有毒样本或有噪声的标签，在处理其他领域的脏样本时往往容易产生较弱的泛化，本文发现各种脏样本的一个共同点是图像和关联标签之间的视觉语言不一致。为了捕捉通道间的语义不一致，利用多通道大语言模型(MLLM)在跨通道对齐和推理方面的优势，提出了通用数据清洗模块(VDC)，它由三个连续的模块组成：视觉问题生成模块，用于生成关于图像的有洞察力的问题；视觉问答模块，通过使用MLLM回答问题来获取视觉内容的语义；以及视觉答案评估模块，用于评估不一致。大量的实验表明，它具有优越的性能和对各种类别和类型的脏样本的泛化。代码可在\url{https://github.com/zihao-ai/vdc}.



## **35. Dialectical Alignment: Resolving the Tension of 3H and Security Threats of LLMs**

辩证对齐：化解3H紧张与LLM安全威胁 cs.CL

**SubmitDate**: 2024-03-30    [abs](http://arxiv.org/abs/2404.00486v1) [paper-pdf](http://arxiv.org/pdf/2404.00486v1)

**Authors**: Shu Yang, Jiayuan Su, Han Jiang, Mengdi Li, Keyuan Cheng, Muhammad Asif Ali, Lijie Hu, Di Wang

**Abstract**: With the rise of large language models (LLMs), ensuring they embody the principles of being helpful, honest, and harmless (3H), known as Human Alignment, becomes crucial. While existing alignment methods like RLHF, DPO, etc., effectively fine-tune LLMs to match preferences in the preference dataset, they often lead LLMs to highly receptive human input and external evidence, even when this information is poisoned. This leads to a tendency for LLMs to be Adaptive Chameleons when external evidence conflicts with their parametric memory. This exacerbates the risk of LLM being attacked by external poisoned data, which poses a significant security risk to LLM system applications such as Retrieval-augmented generation (RAG). To address the challenge, we propose a novel framework: Dialectical Alignment (DA), which (1) utilizes AI feedback to identify optimal strategies for LLMs to navigate inter-context conflicts and context-memory conflicts with different external evidence in context window (i.e., different ratios of poisoned factual contexts); (2) constructs the SFT dataset as well as the preference dataset based on the AI feedback and strategies above; (3) uses the above datasets for LLM alignment to defense poisoned context attack while preserving the effectiveness of in-context knowledge editing. Our experiments show that the dialectical alignment model improves poisoned data attack defense by 20 and does not require any additional prompt engineering or prior declaration of ``you may be attacked`` to the LLMs' context window.

摘要: 随着大型语言模型(LLM)的兴起，确保它们体现了有益、诚实和无害(3H)的原则，即所谓的人类对齐，变得至关重要。虽然现有的比对方法，如RLHF、DPO等，可以有效地微调LLM以匹配偏好数据集中的偏好，但它们往往会导致LLM获得高度接受的人类输入和外部证据，即使这些信息是有毒的。这导致当外部证据与其参数记忆冲突时，LLM有成为自适应变色龙的趋势。这加剧了LLM受到外部有毒数据攻击的风险，这对LLM系统应用程序(如检索增强生成(RAG))构成了重大的安全风险。为了应对这一挑战，我们提出了一种新的框架：辩证对齐(DA)，它(1)利用人工智能反馈来确定LLM在上下文窗口中导航不同外部证据(即不同比例的有毒事实上下文)时的上下文间冲突和上下文-记忆冲突的最佳策略；(2)基于上述AI反馈和策略构建SFT数据集以及偏好数据集；(3)使用上述数据集进行LLM对齐，以防御有毒上下文攻击，同时保持上下文中知识编辑的有效性。我们的实验表明，辩证对齐模型将有毒数据攻击防御提高了20%，并且不需要任何额外的提示工程或预先声明``您可能被攻击``到LLMS上下文窗口。



## **36. Composite Backdoor Attacks Against Large Language Models**

大型语言模型的复合后门攻击 cs.CR

To Appear in Findings of the Association for Computational  Linguistics: NAACL 2024, June 2024

**SubmitDate**: 2024-03-30    [abs](http://arxiv.org/abs/2310.07676v2) [paper-pdf](http://arxiv.org/pdf/2310.07676v2)

**Authors**: Hai Huang, Zhengyu Zhao, Michael Backes, Yun Shen, Yang Zhang

**Abstract**: Large language models (LLMs) have demonstrated superior performance compared to previous methods on various tasks, and often serve as the foundation models for many researches and services. However, the untrustworthy third-party LLMs may covertly introduce vulnerabilities for downstream tasks. In this paper, we explore the vulnerability of LLMs through the lens of backdoor attacks. Different from existing backdoor attacks against LLMs, ours scatters multiple trigger keys in different prompt components. Such a Composite Backdoor Attack (CBA) is shown to be stealthier than implanting the same multiple trigger keys in only a single component. CBA ensures that the backdoor is activated only when all trigger keys appear. Our experiments demonstrate that CBA is effective in both natural language processing (NLP) and multimodal tasks. For instance, with $3\%$ poisoning samples against the LLaMA-7B model on the Emotion dataset, our attack achieves a $100\%$ Attack Success Rate (ASR) with a False Triggered Rate (FTR) below $2.06\%$ and negligible model accuracy degradation. Our work highlights the necessity of increased security research on the trustworthiness of foundation LLMs.

摘要: 大型语言模型(LLM)在各种任务上表现出了比以前的方法更好的性能，并且经常作为许多研究和服务的基础模型。然而，不可信任的第三方LLM可能会暗中为下游任务引入漏洞。在本文中，我们通过后门攻击的镜头来探索LLMS的脆弱性。与现有的针对LLMS的后门攻击不同，我们的后门攻击将多个触发键分散在不同的提示组件中。这种复合后门攻击(CBA)被证明比仅在单个组件中植入相同的多个触发键更隐蔽。CBA确保只有当所有触发键都出现时，后门才被激活。实验表明，CBA在自然语言处理(NLP)和多通道任务中都是有效的。例如，在情感数据集上使用$3$中毒样本对骆驼-7B模型进行攻击，我们的攻击获得了$100$攻击成功率(ASR)，而误触发率(FTR)低于$2.06$，而模型精度下降可以忽略不计。我们的工作突出了加强对基金会低成本管理可信性的安全性研究的必要性。



## **37. LLM-Resistant Math Word Problem Generation via Adversarial Attacks**

通过对抗攻击生成LLM抵抗数学单词问题 cs.CL

Code/data: https://github.com/ruoyuxie/adversarial_mwps_generation

**SubmitDate**: 2024-03-30    [abs](http://arxiv.org/abs/2402.17916v2) [paper-pdf](http://arxiv.org/pdf/2402.17916v2)

**Authors**: Roy Xie, Chengxuan Huang, Junlin Wang, Bhuwan Dhingra

**Abstract**: Large language models (LLMs) have significantly transformed the educational landscape. As current plagiarism detection tools struggle to keep pace with LLMs' rapid advancements, the educational community faces the challenge of assessing students' true problem-solving abilities in the presence of LLMs. In this work, we explore a new paradigm for ensuring fair evaluation -- generating adversarial examples which preserve the structure and difficulty of the original questions aimed for assessment, but are unsolvable by LLMs. Focusing on the domain of math word problems, we leverage abstract syntax trees to structurally generate adversarial examples that cause LLMs to produce incorrect answers by simply editing the numeric values in the problems. We conduct experiments on various open- and closed-source LLMs, quantitatively and qualitatively demonstrating that our method significantly degrades their math problem-solving ability. We identify shared vulnerabilities among LLMs and propose a cost-effective approach to attack high-cost models. Additionally, we conduct automatic analysis on math problems and investigate the cause of failure, offering a nuanced view into model's limitation.

摘要: 大型语言模型(LLM)极大地改变了教育格局。由于目前的抄袭检测工具难以跟上LLMS的快速进步，教育界面临着在LLMS存在的情况下评估学生真正的问题解决能力的挑战。在这项工作中，我们探索了一种确保公平评价的新范式--生成对抗性实例，它保留了用于评价的原始问题的结构和难度，但无法用LLMS解决。聚焦于数学应用题领域，我们利用抽象语法树来结构化地生成对抗性实例，这些实例通过简单地编辑问题中的数值来导致LLMS产生不正确的答案。我们在各种开源和闭源的LLM上进行了实验，定量和定性地证明了我们的方法显著降低了他们的数学问题解决能力。我们识别了LLM之间的共同漏洞，并提出了一种具有成本效益的方法来攻击高成本模型。此外，我们对数学问题进行了自动分析，并调查了失败的原因，为模型的局限性提供了一个细微的视角。



## **38. PETA: Parameter-Efficient Trojan Attacks**

PETA：参数高效木马攻击 cs.CL

**SubmitDate**: 2024-03-29    [abs](http://arxiv.org/abs/2310.00648v5) [paper-pdf](http://arxiv.org/pdf/2310.00648v5)

**Authors**: Lauren Hong, Ting Wang

**Abstract**: Parameter-efficient fine-tuning (PEFT) enables efficient adaptation of pre-trained language models (PLMs) to specific tasks. By tuning only a minimal set of (extra) parameters, PEFT achieves performance that is comparable to standard fine-tuning. However, despite its prevalent use, the security implications of PEFT remain largely unexplored. In this paper, we take the initial steps and present PETA, a novel trojan attack that compromises the weights of PLMs by accounting for downstream adaptation through bilevel optimization: the upper-level objective embeds the backdoor into a model while the lower-level objective simulates PEFT to both retain the PLM's task-specific performance and ensure that the backdoor persists after fine-tuning. With extensive evaluation across a variety of downstream tasks and trigger designs, we demonstrate PETA's effectiveness in terms of both attack success rate and clean accuracy, even when the attacker does not have full knowledge of the victim user's training process.

摘要: 参数高效微调(PEFT)使预先训练的语言模型(PLM)能够有效地适应特定任务。通过只调整最小的一组(额外)参数，PEFT实现了与标准微调相当的性能。然而，尽管PEFT被广泛使用，但其安全影响在很大程度上仍未被探索。在本文中，我们采取了初步的步骤，并提出了一种新的木马攻击PETA，它通过双层优化考虑下游适应来折衷PLM的权重：上层目标将后门嵌入到模型中，而下层目标模拟PEFT，既保留了PLM的特定任务性能，又确保了微调后后门的存在。通过对各种下游任务和触发器设计的广泛评估，我们证明了PETA在攻击成功率和干净准确性方面的有效性，即使攻击者并不完全了解受害者用户的培训过程。



## **39. Detoxifying Large Language Models via Knowledge Editing**

基于知识编辑的大型语言模型解化 cs.CL

Ongoing work. Project website:  https://zjunlp.github.io/project/SafeEdit Due to the specificity of the  knowledge editing setting, we revise Tables 1 and 3 to present a fair  comparison of experimental results. More experimental results will be updated  soon

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2403.14472v2) [paper-pdf](http://arxiv.org/pdf/2403.14472v2)

**Authors**: Mengru Wang, Ningyu Zhang, Ziwen Xu, Zekun Xi, Shumin Deng, Yunzhi Yao, Qishen Zhang, Linyi Yang, Jindong Wang, Huajun Chen

**Abstract**: This paper investigates using knowledge editing techniques to detoxify Large Language Models (LLMs). We construct a benchmark, SafeEdit, which covers nine unsafe categories with various powerful attack prompts and equips comprehensive metrics for systematic evaluation. We conduct experiments with several knowledge editing approaches, indicating that knowledge editing has the potential to efficiently detoxify LLMs with limited impact on general performance. Then, we propose a simple yet effective baseline, dubbed Detoxifying with Intraoperative Neural Monitoring (DINM), to diminish the toxicity of LLMs within a few tuning steps via only one instance. We further provide an in-depth analysis of the internal mechanism for various detoxify approaches, demonstrating that previous methods like SFT and DPO may merely suppress the activations of toxic parameters, while DINM mitigates the toxicity of the toxic parameters to a certain extent, making permanent adjustments. We hope that these insights could shed light on future work of developing detoxifying approaches and the underlying knowledge mechanisms of LLMs. Code and benchmark are available at https://github.com/zjunlp/EasyEdit.

摘要: 本文研究了利用知识编辑技术对大型语言模型进行去毒处理。我们构建了一个涵盖9个不安全类别、具有各种强大的攻击提示的基准SafeEdit，并配备了全面的度量来进行系统评估。我们用几种知识编辑方法进行了实验，表明知识编辑有可能在对一般性能影响有限的情况下有效地对LLM进行解毒。然后，我们提出了一个简单而有效的基线，称为术中神经监测解毒(DINM)，仅通过一个实例在几个调整步骤内降低LLMS的毒性。我们进一步深入分析了各种解毒方法的内在机制，证明了以前的方法如SFT和DPO可能只是抑制了毒性参数的激活，而DINM在一定程度上减轻了毒性参数的毒性，做出了永久性的调整。我们希望这些洞察力能够为未来开发戒毒方法的工作和LLMS的潜在知识机制提供帮助。代码和基准测试可在https://github.com/zjunlp/EasyEdit.上获得



## **40. Evolving Assembly Code in an Adversarial Environment**

对抗环境下的汇编代码演变 cs.NE

9 pages, 5 figures, 6 listings

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2403.19489v1) [paper-pdf](http://arxiv.org/pdf/2403.19489v1)

**Authors**: Irina Maliukov, Gera Weiss, Oded Margalit, Achiya Elyasaf

**Abstract**: In this work, we evolve assembly code for the CodeGuru competition. The competition's goal is to create a survivor -- an assembly program that runs the longest in shared memory, by resisting attacks from adversary survivors and finding their weaknesses. For evolving top-notch solvers, we specify a Backus Normal Form (BNF) for the assembly language and synthesize the code from scratch using Genetic Programming (GP). We evaluate the survivors by running CodeGuru games against human-written winning survivors. Our evolved programs found weaknesses in the programs they were trained against and utilized them. In addition, we compare our approach with a Large-Language Model, demonstrating that the latter cannot generate a survivor that can win at any competition. This work has important applications for cyber-security, as we utilize evolution to detect weaknesses in survivors. The assembly BNF is domain-independent; thus, by modifying the fitness function, it can detect code weaknesses and help fix them. Finally, the CodeGuru competition offers a novel platform for analyzing GP and code evolution in adversarial environments. To support further research in this direction, we provide a thorough qualitative analysis of the evolved survivors and the weaknesses found.

摘要: 在这项工作中，我们为CodeGuru竞赛演变汇编代码。这项竞赛的目标是创建一个幸存者--一个在共享内存中运行时间最长的汇编程序，通过抵抗对手幸存者的攻击并找到他们的弱点。对于进化的顶级解算器，我们为汇编语言指定了Backus范式(BNF)，并使用遗传编程(GP)从头开始合成代码。我们通过运行CodeGuru游戏来评估幸存者，以对抗人类编写的获胜幸存者。我们的演进计划发现了他们所针对的计划中的弱点，并利用了这些弱点。此外，我们将我们的方法与大语言模型进行比较，表明后者无法产生能够在任何竞争中获胜的幸存者。这项工作在网络安全方面有重要的应用，因为我们利用进化论来检测幸存者的弱点。程序集BNF是独立于域的；因此，通过修改适应度函数，它可以检测代码弱点并帮助修复它们。最后，CodeGuru竞赛为分析对抗性环境中的GP和代码演化提供了一个新的平台。为了支持这方面的进一步研究，我们对进化的幸存者和发现的弱点进行了彻底的定性分析。



## **41. JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models**

JailbreakBench：一个大型语言模型的开放鲁棒性基准测试 cs.CR

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2404.01318v1) [paper-pdf](http://arxiv.org/pdf/2404.01318v1)

**Authors**: Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, Eric Wong

**Abstract**: Jailbreak attacks cause large language models (LLMs) to generate harmful, unethical, or otherwise objectionable content. Evaluating these attacks presents a number of challenges, which the current collection of benchmarks and evaluation techniques do not adequately address. First, there is no clear standard of practice regarding jailbreaking evaluation. Second, existing works compute costs and success rates in incomparable ways. And third, numerous works are not reproducible, as they withhold adversarial prompts, involve closed-source code, or rely on evolving proprietary APIs. To address these challenges, we introduce JailbreakBench, an open-sourced benchmark with the following components: (1) a new jailbreaking dataset containing 100 unique behaviors, which we call JBB-Behaviors; (2) an evolving repository of state-of-the-art adversarial prompts, which we refer to as jailbreak artifacts; (3) a standardized evaluation framework that includes a clearly defined threat model, system prompts, chat templates, and scoring functions; and (4) a leaderboard that tracks the performance of attacks and defenses for various LLMs. We have carefully considered the potential ethical implications of releasing this benchmark, and believe that it will be a net positive for the community. Over time, we will expand and adapt the benchmark to reflect technical and methodological advances in the research community.

摘要: 越狱攻击会导致大型语言模型(LLM)生成有害、不道德或令人反感的内容。评估这些攻击带来了许多挑战，目前收集的基准和评估技术没有充分解决这些挑战。首先，关于越狱评估没有明确的实践标准。其次，现有的工作以无与伦比的方式计算成本和成功率。第三，许多作品是不可复制的，因为它们保留了对抗性提示，涉及封闭源代码，或者依赖于不断发展的专有API。为了应对这些挑战，我们引入了JailBreak Bch，这是一个开源基准测试，具有以下组件：(1)包含100个独特行为的新越狱数据集，我们称之为JBB行为；(2)不断发展的最新对手提示存储库，我们称为越狱人工制品；(3)标准化评估框架，其中包括明确定义的威胁模型、系统提示、聊天模板和评分功能；以及(4)跟踪各种LLM攻击和防御性能的排行榜。我们已仔细考虑发布这一基准的潜在道德影响，并相信它将为社会带来净积极的影响。随着时间的推移，我们将扩大和调整基准，以反映研究界的技术和方法进步。



## **42. Data Poisoning for In-context Learning**

基于上下文学习的数据中毒 cs.CR

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2402.02160v2) [paper-pdf](http://arxiv.org/pdf/2402.02160v2)

**Authors**: Pengfei He, Han Xu, Yue Xing, Hui Liu, Makoto Yamada, Jiliang Tang

**Abstract**: In the domain of large language models (LLMs), in-context learning (ICL) has been recognized for its innovative ability to adapt to new tasks, relying on examples rather than retraining or fine-tuning. This paper delves into the critical issue of ICL's susceptibility to data poisoning attacks, an area not yet fully explored. We wonder whether ICL is vulnerable, with adversaries capable of manipulating example data to degrade model performance. To address this, we introduce ICLPoison, a specialized attacking framework conceived to exploit the learning mechanisms of ICL. Our approach uniquely employs discrete text perturbations to strategically influence the hidden states of LLMs during the ICL process. We outline three representative strategies to implement attacks under our framework, each rigorously evaluated across a variety of models and tasks. Our comprehensive tests, including trials on the sophisticated GPT-4 model, demonstrate that ICL's performance is significantly compromised under our framework. These revelations indicate an urgent need for enhanced defense mechanisms to safeguard the integrity and reliability of LLMs in applications relying on in-context learning.

摘要: 在大型语言模型(LLM)领域，情境学习(ICL)因其适应新任务的创新能力而被公认，它依赖于例子而不是再培训或微调。本文深入研究了ICL对数据中毒攻击的易感性这一关键问题，这是一个尚未完全探索的领域。我们想知道ICL是否易受攻击，因为对手能够操纵示例数据来降低模型性能。为了解决这个问题，我们引入了ICLPoison，这是一个专门的攻击框架，旨在利用ICL的学习机制。我们的方法独特地使用离散文本扰动来战略性地影响ICL过程中LLM的隐藏状态。我们概述了在我们的框架下实施攻击的三种具有代表性的战略，每种战略都在各种模型和任务中进行了严格的评估。我们的全面测试，包括对复杂的GPT-4模型的试验，表明ICL的性能在我们的框架下受到了严重影响。这些发现表明，迫切需要增强防御机制，以保障依赖于情景学习的应用程序中LLMS的完整性和可靠性。



## **43. Attacks, Defenses and Evaluations for LLM Conversation Safety: A Survey**

LLM会话安全的攻击、防御与评估：一项调查 cs.CL

Accepted to NAACL 2024

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2402.09283v3) [paper-pdf](http://arxiv.org/pdf/2402.09283v3)

**Authors**: Zhichen Dong, Zhanhui Zhou, Chao Yang, Jing Shao, Yu Qiao

**Abstract**: Large Language Models (LLMs) are now commonplace in conversation applications. However, their risks of misuse for generating harmful responses have raised serious societal concerns and spurred recent research on LLM conversation safety. Therefore, in this survey, we provide a comprehensive overview of recent studies, covering three critical aspects of LLM conversation safety: attacks, defenses, and evaluations. Our goal is to provide a structured summary that enhances understanding of LLM conversation safety and encourages further investigation into this important subject. For easy reference, we have categorized all the studies mentioned in this survey according to our taxonomy, available at: https://github.com/niconi19/LLM-conversation-safety.

摘要: 大型语言模型（LLM）现在在会话应用中很常见。然而，它们被滥用以产生有害反应的风险引起了严重的社会关注，并刺激了最近对LLM会话安全的研究。因此，在本次调查中，我们提供了一个全面的概述最近的研究，涵盖了LLM会话安全的三个关键方面：攻击，防御和评估。我们的目标是提供一个结构化的摘要，以提高对LLM会话安全的理解，并鼓励进一步调查这一重要主题。为了便于参考，我们根据我们的分类法对本次调查中提到的所有研究进行了分类，可在https://github.com/niconi19/LLM-conversation-safety上查阅。



## **44. InferDPT: Privacy-Preserving Inference for Black-box Large Language Model**

InferDPT：黑盒大语言模型的隐私保护推理 cs.CR

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2310.12214v6) [paper-pdf](http://arxiv.org/pdf/2310.12214v6)

**Authors**: Meng Tong, Kejiang Chen, Jie Zhang, Yuang Qi, Weiming Zhang, Nenghai Yu, Tianwei Zhang, Zhikun Zhang

**Abstract**: Large language models (LLMs), like ChatGPT, have greatly simplified text generation tasks. However, they have also raised concerns about privacy risks such as data leakage and unauthorized data collection. Existing solutions for privacy-preserving inference face practical challenges related to computation time and communication costs. In this paper, we propose InferDPT, the first practical framework for the privacy-preserving Inference of black-box LLMs, implementing Differential Privacy in Text generation. InferDPT comprises two key modules: the "perturbation module" utilizes the exponential mechanism to generate a perturbed prompt, facilitating privacy-preserving inference with black-box LLMs, and the "extraction module", inspired by knowledge distillation and retrieval-augmented generation, extracts coherent and consistent text from the perturbed generation result, ensuring successful text generation completion. To address privacy concerns related to previous exponential mechanisms' susceptibility to embedding revision attacks, we introduce RANTEXT, a novel differential privacy mechanism integrated into the perturbation module of InferDPT, which introduces the concept of "RANdom adjacency" for TEXT perturbation within the prompt. Experimental results across three datasets demonstrate that the text generation quality of InferDPT is comparable to that of non-private GPT-4, and RANTEXT surpasses existing state-of-the-art mechanisms, namely, SANTEXT+ and CUSTEXT+ in the trade-off between privacy and utility. Even with an privacy parameter epsilon value of 6.0, RANTEXT achieves an average privacy protection rate exceeding 90% against embedding revision attacks, which is 0.58 times higher than that of SANTEXT+ and 3.35 times higher than that of CUSTEXT+.

摘要: 大型语言模型(LLM)，如ChatGPT，极大地简化了文本生成任务。然而，他们也对数据泄露和未经授权的数据收集等隐私风险表示担忧。现有的隐私保护推理解决方案面临着与计算时间和通信成本相关的实际挑战。在本文中，我们提出了第一个实用的黑盒LLMS隐私保护推理框架InferDPT，在文本生成中实现了差分隐私。InferDPT包括两个关键模块：“扰动模块”利用指数机制生成扰动提示，便于使用黑盒LLMS进行隐私保护推理；“提取模块”受知识提炼和检索-增强生成的启发，从扰动生成结果中提取连贯一致的文本，确保文本生成成功完成。针对以往指数机制易受修改攻击的隐私性问题，引入了一种新的差异化隐私机制RANTEXT，该机制集成在InferDPT的扰动模块中，引入了随机邻接的概念来处理提示内的文本扰动。在三个数据集上的实验结果表明，InferDPT的文本生成质量与非私有GPT-4相当，RANTEXT在隐私和效用之间的权衡方面超过了现有的最新机制SanText+和CUSTEXT+。即使在隐私参数epsilon值为6.0的情况下，RANTEXT对嵌入修改攻击的平均隐私保护率也超过90%，比SanText+高0.58倍，比CUSTEXT+高3.35倍。



## **45. Tricking LLMs into Disobedience: Formalizing, Analyzing, and Detecting Jailbreaks**

欺骗法学硕士到不服从：形式化，分析和检测越狱 cs.CL

Accepted at LREC-COLING 2024 - The 2024 Joint International  Conference on Computational Linguistics, Language Resources and Evaluation

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2305.14965v4) [paper-pdf](http://arxiv.org/pdf/2305.14965v4)

**Authors**: Abhinav Rao, Sachin Vashistha, Atharva Naik, Somak Aditya, Monojit Choudhury

**Abstract**: Recent explorations with commercial Large Language Models (LLMs) have shown that non-expert users can jailbreak LLMs by simply manipulating their prompts; resulting in degenerate output behavior, privacy and security breaches, offensive outputs, and violations of content regulator policies. Limited studies have been conducted to formalize and analyze these attacks and their mitigations. We bridge this gap by proposing a formalism and a taxonomy of known (and possible) jailbreaks. We survey existing jailbreak methods and their effectiveness on open-source and commercial LLMs (such as GPT-based models, OPT, BLOOM, and FLAN-T5-XXL). We further discuss the challenges of jailbreak detection in terms of their effectiveness against known attacks. For further analysis, we release a dataset of model outputs across 3700 jailbreak prompts over 4 tasks.

摘要: 最近对商业大型语言模型（LLM）的探索表明，非专家用户可以通过简单地操纵他们的提示来越狱LLM;导致退化的输出行为、隐私和安全漏洞、攻击性输出以及违反内容监管政策。已经进行了有限的研究，以正规化和分析这些攻击及其缓解措施。我们通过提出一种形式主义和已知（和可能的）越狱分类来弥合这一差距。我们调查了现有的越狱方法及其在开源和商业LLM上的有效性（如基于GPL的模型、OPT、BLOOM和FLAN-T5-XXL）。我们进一步讨论了越狱检测在对抗已知攻击的有效性方面的挑战。为了进一步分析，我们发布了一个模型输出数据集，该数据集涵盖了4个任务的3700个越狱提示。



## **46. Optimization-based Prompt Injection Attack to LLM-as-a-Judge**

基于优化的LLM-as-a-Judge快速注入攻击 cs.CR

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2403.17710v1) [paper-pdf](http://arxiv.org/pdf/2403.17710v1)

**Authors**: Jiawen Shi, Zenghui Yuan, Yinuo Liu, Yue Huang, Pan Zhou, Lichao Sun, Neil Zhenqiang Gong

**Abstract**: LLM-as-a-Judge is a novel solution that can assess textual information with large language models (LLMs). Based on existing research studies, LLMs demonstrate remarkable performance in providing a compelling alternative to traditional human assessment. However, the robustness of these systems against prompt injection attacks remains an open question. In this work, we introduce JudgeDeceiver, a novel optimization-based prompt injection attack tailored to LLM-as-a-Judge. Our method formulates a precise optimization objective for attacking the decision-making process of LLM-as-a-Judge and utilizes an optimization algorithm to efficiently automate the generation of adversarial sequences, achieving targeted and effective manipulation of model evaluations. Compared to handcraft prompt injection attacks, our method demonstrates superior efficacy, posing a significant challenge to the current security paradigms of LLM-based judgment systems. Through extensive experiments, we showcase the capability of JudgeDeceiver in altering decision outcomes across various cases, highlighting the vulnerability of LLM-as-a-Judge systems to the optimization-based prompt injection attack.

摘要: LLM-as-a-Court是一种新的解决方案，它可以使用大型语言模型(LLM)来评估文本信息。基于现有的研究，LLMS在提供一种令人信服的替代传统的人类评估方面表现出显著的性能。然而，这些系统对快速注入攻击的健壮性仍然是一个悬而未决的问题。在这项工作中，我们介绍了一种新的基于优化的快速注入攻击，该攻击是针对LLM-as-a-Court定制的。我们的方法为攻击LLM-as-a-Court的决策过程制定了一个精确的优化目标，并利用优化算法高效地自动生成对抗序列，实现了对模型评估的有针对性和有效的操作。与手工即时注入攻击相比，我们的方法表现出更好的有效性，对基于LLM的判断系统的现有安全范例提出了重大挑战。通过大量的实验，我们展示了JudgeDeceiver在改变不同案件的决策结果方面的能力，突出了LLM-as-a-Court系统对基于优化的即时注入攻击的脆弱性。



## **47. Targeted Visualization of the Backbone of Encoder LLMs**

编码器LLM骨干的目标可视化 cs.LG

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2403.18872v1) [paper-pdf](http://arxiv.org/pdf/2403.18872v1)

**Authors**: Isaac Roberts, Alexander Schulz, Luca Hermes, Barbara Hammer

**Abstract**: Attention based Large Language Models (LLMs) are the state-of-the-art in natural language processing (NLP). The two most common architectures are encoders such as BERT, and decoders like the GPT models. Despite the success of encoder models, on which we focus in this work, they also bear several risks, including issues with bias or their susceptibility for adversarial attacks, signifying the necessity for explainable AI to detect such issues. While there does exist various local explainability methods focusing on the prediction of single inputs, global methods based on dimensionality reduction for classification inspection, which have emerged in other domains and that go further than just using t-SNE in the embedding space, are not widely spread in NLP.   To reduce this gap, we investigate the application of DeepView, a method for visualizing a part of the decision function together with a data set in two dimensions, to the NLP domain. While in previous work, DeepView has been used to inspect deep image classification models, we demonstrate how to apply it to BERT-based NLP classifiers and investigate its usability in this domain, including settings with adversarially perturbed input samples and pre-trained, fine-tuned, and multi-task models.

摘要: 基于注意力的大语言模型(LLM)是自然语言处理(NLP)领域的前沿技术。两种最常见的架构是编码器(如BERT)和解码器(如GPT模型)。尽管我们在本工作中重点关注的编码器模型取得了成功，但它们也存在几个风险，包括偏见或它们对对抗性攻击的敏感性问题，这意味着有必要使用可解释的人工智能来检测此类问题。虽然有各种局部可解释方法专注于单输入预测，但在其他领域出现的基于降维的全局分类检测方法并没有在NLP中广泛推广，这些方法比仅仅使用嵌入空间中的t-SNE更深入。为了缩小这一差距，我们研究了DeepView在NLP领域的应用，DeepView是一种将决策函数的一部分与二维数据集一起可视化的方法。在以前的工作中，DeepView已经被用来检查深度图像分类模型，我们演示了如何将其应用于基于BERT的NLP分类器，并研究了它在该领域的可用性，包括设置了相反扰动的输入样本和预先训练的、微调的和多任务模型。



## **48. CYGENT: A cybersecurity conversational agent with log summarization powered by GPT-3**

CYGENT：一个网络安全会话代理，具有由GPT-3提供支持的日志摘要 cs.CR

7 pages, 9 figures

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2403.17160v1) [paper-pdf](http://arxiv.org/pdf/2403.17160v1)

**Authors**: Prasasthy Balasubramanian, Justin Seby, Panos Kostakos

**Abstract**: In response to the escalating cyber-attacks in the modern IT and IoT landscape, we developed CYGENT, a conversational agent framework powered by GPT-3.5 turbo model, designed to aid system administrators in ensuring optimal performance and uninterrupted resource availability. This study focuses on fine-tuning GPT-3 models for cybersecurity tasks, including conversational AI and generative AI tailored specifically for cybersecurity operations. CYGENT assists users by providing cybersecurity information, analyzing and summarizing uploaded log files, detecting specific events, and delivering essential instructions. The conversational agent was developed based on the GPT-3.5 turbo model. We fine-tuned and validated summarizer models (GPT3) using manually generated data points. Using this approach, we achieved a BERTscore of over 97%, indicating GPT-3's enhanced capability in summarizing log files into human-readable formats and providing necessary information to users. Furthermore, we conducted a comparative analysis of GPT-3 models with other Large Language Models (LLMs), including CodeT5-small, CodeT5-base, and CodeT5-base-multi-sum, with the objective of analyzing log analysis techniques. Our analysis consistently demonstrated that Davinci (GPT-3) model outperformed all other LLMs, showcasing higher performance. These findings are crucial for improving human comprehension of logs, particularly in light of the increasing numbers of IoT devices. Additionally, our research suggests that the CodeT5-base-multi-sum model exhibits comparable performance to Davinci to some extent in summarizing logs, indicating its potential as an offline model for this task.

摘要: 为了应对现代IT和物联网环境中不断升级的网络攻击，我们开发了基于GPT-3.5 Turbo模型的会话代理框架CyGENT，旨在帮助系统管理员确保最佳性能和不间断的资源可用性。这项研究的重点是针对网络安全任务微调GPT-3模型，包括专门为网络安全操作量身定做的对话式人工智能和生成式人工智能。CyGENT通过提供网络安全信息、分析和汇总上传的日志文件、检测特定事件和提供基本说明来帮助用户。会话代理是在GPT-3.5涡轮机型的基础上开发的。我们使用手动生成的数据点对汇总器模型(GPT3)进行了微调和验证。使用该方法，我们获得了97%以上的BERT分数，这表明GPT-3的S增强了将日志文件摘要为人类可读格式并向用户提供必要信息的能力。此外，我们还将GPT-3模型与其他大型语言模型(包括CodeT5-Small、CodeT5-BASE和CodeT5-BASE-MULTSUM)进行了比较分析，目的是分析日志分析技术。我们的分析始终表明，Davinci(GPT-3)模型的性能优于所有其他LLM，表现出更高的性能。这些发现对于提高人类对日志的理解至关重要，特别是在物联网设备数量不断增加的情况下。此外，我们的研究表明，CodeT5基多和模型在总结日志方面在一定程度上表现出与Davinci相当的性能，表明其作为这一任务的离线模型的潜力。



## **49. InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents**

InjectAgent：基准测试工具集成大型语言模型代理中的间接提示注入 cs.CL

28 pages, 5 figures, 9 tables

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2403.02691v2) [paper-pdf](http://arxiv.org/pdf/2403.02691v2)

**Authors**: Qiusi Zhan, Zhixiang Liang, Zifan Ying, Daniel Kang

**Abstract**: Recent work has embodied LLMs as agents, allowing them to access tools, perform actions, and interact with external content (e.g., emails or websites). However, external content introduces the risk of indirect prompt injection (IPI) attacks, where malicious instructions are embedded within the content processed by LLMs, aiming to manipulate these agents into executing detrimental actions against users. Given the potentially severe consequences of such attacks, establishing benchmarks to assess and mitigate these risks is imperative.   In this work, we introduce InjecAgent, a benchmark designed to assess the vulnerability of tool-integrated LLM agents to IPI attacks. InjecAgent comprises 1,054 test cases covering 17 different user tools and 62 attacker tools. We categorize attack intentions into two primary types: direct harm to users and exfiltration of private data. We evaluate 30 different LLM agents and show that agents are vulnerable to IPI attacks, with ReAct-prompted GPT-4 vulnerable to attacks 24% of the time. Further investigation into an enhanced setting, where the attacker instructions are reinforced with a hacking prompt, shows additional increases in success rates, nearly doubling the attack success rate on the ReAct-prompted GPT-4. Our findings raise questions about the widespread deployment of LLM Agents. Our benchmark is available at https://github.com/uiuc-kang-lab/InjecAgent.

摘要: 最近的工作将LLMS体现为代理，允许它们访问工具、执行操作并与外部内容(例如，电子邮件或网站)交互。然而，外部内容会带来间接提示注入(IPI)攻击的风险，在IPI攻击中，恶意指令被嵌入到LLMS处理的内容中，目的是操纵这些代理执行针对用户的有害操作。鉴于此类攻击的潜在严重后果，建立评估和减轻这些风险的基准势在必行。在这项工作中，我们引入了InjecAgent，这是一个旨在评估工具集成的LLM代理对IPI攻击的脆弱性的基准测试。InjecAgent由1,054个测试用例组成，涵盖17个不同的用户工具和62个攻击者工具。我们将攻击意图分为两种主要类型：直接伤害用户和泄露私人数据。我们评估了30种不同的LLM代理，表明代理容易受到IPI攻击，其中反应提示的GPT-4在24%的时间内容易受到攻击。对增强设置的进一步调查显示，成功率进一步提高，反应提示GPT-4的攻击成功率几乎翻了一番。在增强设置中，攻击者的指令通过黑客提示得到加强。我们的发现对LLM特工的广泛部署提出了质疑。我们的基准测试可从https://github.com/uiuc-kang-lab/InjecAgent.获得



## **50. Exploring the Adversarial Capabilities of Large Language Models**

探索大型语言模型的对抗能力 cs.AI

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2402.09132v3) [paper-pdf](http://arxiv.org/pdf/2402.09132v3)

**Authors**: Lukas Struppek, Minh Hieu Le, Dominik Hintersdorf, Kristian Kersting

**Abstract**: The proliferation of large language models (LLMs) has sparked widespread and general interest due to their strong language generation capabilities, offering great potential for both industry and research. While previous research delved into the security and privacy issues of LLMs, the extent to which these models can exhibit adversarial behavior remains largely unexplored. Addressing this gap, we investigate whether common publicly available LLMs have inherent capabilities to perturb text samples to fool safety measures, so-called adversarial examples resp.~attacks. More specifically, we investigate whether LLMs are inherently able to craft adversarial examples out of benign samples to fool existing safe rails. Our experiments, which focus on hate speech detection, reveal that LLMs succeed in finding adversarial perturbations, effectively undermining hate speech detection systems. Our findings carry significant implications for (semi-)autonomous systems relying on LLMs, highlighting potential challenges in their interaction with existing systems and safety measures.

摘要: 大型语言模型因其强大的语言生成能力而引起了广泛的关注，为工业和研究提供了巨大的潜力。虽然之前的研究已经深入研究了LLMS的安全和隐私问题，但这些模型在多大程度上可以表现出敌对行为，仍然很大程度上还没有被探索。针对这一差距，我们调查了常见的公开可用的LLM是否具有固有的能力来扰乱文本样本以愚弄安全措施，即所谓的对抗性示例攻击。更具体地说，我们调查LLM是否天生就能够从良性样本中制作敌意示例，以愚弄现有的安全Rail。我们的实验集中在仇恨语音检测上，实验表明，LLMS成功地发现了敌意扰动，有效地破坏了仇恨语音检测系统。我们的发现对依赖LLMS的(半)自治系统具有重大影响，突显了它们与现有系统和安全措施相互作用的潜在挑战。



