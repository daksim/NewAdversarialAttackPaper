# Latest Large Language Model Attack Papers
**update at 2024-05-17 09:51:42**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Keep It Private: Unsupervised Privatization of Online Text**

保持隐私：在线文本的无监督私有化 cs.CL

17 pages, 6 figures

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2405.10260v1) [paper-pdf](http://arxiv.org/pdf/2405.10260v1)

**Authors**: Calvin Bao, Marine Carpuat

**Abstract**: Authorship obfuscation techniques hold the promise of helping people protect their privacy in online communications by automatically rewriting text to hide the identity of the original author. However, obfuscation has been evaluated in narrow settings in the NLP literature and has primarily been addressed with superficial edit operations that can lead to unnatural outputs. In this work, we introduce an automatic text privatization framework that fine-tunes a large language model via reinforcement learning to produce rewrites that balance soundness, sense, and privacy. We evaluate it extensively on a large-scale test set of English Reddit posts by 68k authors composed of short-medium length texts. We study how the performance changes among evaluative conditions including authorial profile length and authorship detection strategy. Our method maintains high text quality according to both automated metrics and human evaluation, and successfully evades several automated authorship attacks.

摘要: 作者身份混淆技术有望通过自动重写文本以隐藏原作者的身份来帮助人们在在线通信中保护自己的隐私。然而，在NLP文献中，模糊是在狭窄的环境中进行评估的，并且主要通过可能导致不自然输出的肤浅编辑操作来解决的。在这项工作中，我们引入了一个自动文本私有化框架，该框架通过强化学习微调大型语言模型，以产生平衡合理性、意义和隐私的重写。我们在由68，000位作者组成的Reddit英语帖子的大规模测试集上对其进行了广泛评估，这些帖子由中短文本组成。我们研究了作者个人资料长度和作者身份检测策略等评价条件之间的性能如何变化。我们的方法根据自动化指标和人工评估保持高文本质量，并成功规避了几次自动作者攻击。



## **2. Protecting Your LLMs with Information Bottleneck**

通过信息瓶颈保护您的LLC cs.CL

23 pages, 7 figures, 8 tables

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2404.13968v2) [paper-pdf](http://arxiv.org/pdf/2404.13968v2)

**Authors**: Zichuan Liu, Zefan Wang, Linjie Xu, Jinyu Wang, Lei Song, Tianchun Wang, Chunlin Chen, Wei Cheng, Jiang Bian

**Abstract**: The advent of large language models (LLMs) has revolutionized the field of natural language processing, yet they might be attacked to produce harmful content. Despite efforts to ethically align LLMs, these are often fragile and can be circumvented by jailbreaking attacks through optimized or manual adversarial prompts. To address this, we introduce the Information Bottleneck Protector (IBProtector), a defense mechanism grounded in the information bottleneck principle, and we modify the objective to avoid trivial solutions. The IBProtector selectively compresses and perturbs prompts, facilitated by a lightweight and trainable extractor, preserving only essential information for the target LLMs to respond with the expected answer. Moreover, we further consider a situation where the gradient is not visible to be compatible with any LLM. Our empirical evaluations show that IBProtector outperforms current defense methods in mitigating jailbreak attempts, without overly affecting response quality or inference speed. Its effectiveness and adaptability across various attack methods and target LLMs underscore the potential of IBProtector as a novel, transferable defense that bolsters the security of LLMs without requiring modifications to the underlying models.

摘要: 大型语言模型的出现给自然语言处理领域带来了革命性的变化，但它们可能会受到攻击，产生有害的内容。尽管努力在道德上调整LLM，但这些往往是脆弱的，可以通过优化或手动对抗性提示通过越狱攻击来绕过。为了解决这个问题，我们引入了信息瓶颈保护器(IBProtector)，这是一种基于信息瓶颈原理的防御机制，我们修改了目标以避免琐碎的解决方案。IBProtector有选择地压缩和干扰提示，由一个轻量级和可训练的提取程序促进，只保留目标LLMS的基本信息，以响应预期的答案。此外，我们还进一步考虑了梯度不可见的情况，以与任何LLM相容。我们的经验评估表明，在不过度影响响应质量或推理速度的情况下，IBProtector在缓解越狱企图方面优于现有的防御方法。它对各种攻击方法和目标LLM的有效性和适应性突显了IBProtector作为一种新型、可转移的防御系统的潜力，无需修改底层模型即可增强LLM的安全性。



## **3. Adversarial Robustness for Visual Grounding of Multimodal Large Language Models**

多模式大型语言模型视觉基础的对抗鲁棒性 cs.CV

ICLR 2024 Workshop on Reliable and Responsible Foundation Models

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2405.09981v1) [paper-pdf](http://arxiv.org/pdf/2405.09981v1)

**Authors**: Kuofeng Gao, Yang Bai, Jiawang Bai, Yong Yang, Shu-Tao Xia

**Abstract**: Multi-modal Large Language Models (MLLMs) have recently achieved enhanced performance across various vision-language tasks including visual grounding capabilities. However, the adversarial robustness of visual grounding remains unexplored in MLLMs. To fill this gap, we use referring expression comprehension (REC) as an example task in visual grounding and propose three adversarial attack paradigms as follows. Firstly, untargeted adversarial attacks induce MLLMs to generate incorrect bounding boxes for each object. Besides, exclusive targeted adversarial attacks cause all generated outputs to the same target bounding box. In addition, permuted targeted adversarial attacks aim to permute all bounding boxes among different objects within a single image. Extensive experiments demonstrate that the proposed methods can successfully attack visual grounding capabilities of MLLMs. Our methods not only provide a new perspective for designing novel attacks but also serve as a strong baseline for improving the adversarial robustness for visual grounding of MLLMs.

摘要: 多模式大型语言模型(MLLM)最近在包括视觉基础能力在内的各种视觉语言任务中获得了增强的性能。然而，在最大似然最小二乘法中，视觉接地的对抗稳健性仍未被探索。为了填补这一空白，我们使用指称表达理解(REC)作为视觉基础的示例任务，并提出了以下三种对抗性攻击范式。首先，无针对性的对抗性攻击会导致MLLMS为每个对象生成错误的包围盒。此外，排他性定向对抗性攻击会导致所有生成的输出都指向相同的目标边界框。此外，置换定向对抗性攻击旨在置换单个图像中不同对象之间的所有包围盒。大量实验表明，所提出的方法能够成功地攻击MLLMS的视觉接地能力。我们的方法不仅为设计新的攻击提供了新的视角，而且为提高MLLMS视觉接地的对抗性稳健性提供了强有力的基线。



## **4. Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy**

不精确的遗忘需要更仔细的评估，以避免错误的隐私感 cs.LG

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2403.01218v2) [paper-pdf](http://arxiv.org/pdf/2403.01218v2)

**Authors**: Jamie Hayes, Ilia Shumailov, Eleni Triantafillou, Amr Khalifa, Nicolas Papernot

**Abstract**: The high cost of model training makes it increasingly desirable to develop techniques for unlearning. These techniques seek to remove the influence of a training example without having to retrain the model from scratch. Intuitively, once a model has unlearned, an adversary that interacts with the model should no longer be able to tell whether the unlearned example was included in the model's training set or not. In the privacy literature, this is known as membership inference. In this work, we discuss adaptations of Membership Inference Attacks (MIAs) to the setting of unlearning (leading to their ``U-MIA'' counterparts). We propose a categorization of existing U-MIAs into ``population U-MIAs'', where the same attacker is instantiated for all examples, and ``per-example U-MIAs'', where a dedicated attacker is instantiated for each example. We show that the latter category, wherein the attacker tailors its membership prediction to each example under attack, is significantly stronger. Indeed, our results show that the commonly used U-MIAs in the unlearning literature overestimate the privacy protection afforded by existing unlearning techniques on both vision and language models. Our investigation reveals a large variance in the vulnerability of different examples to per-example U-MIAs. In fact, several unlearning algorithms lead to a reduced vulnerability for some, but not all, examples that we wish to unlearn, at the expense of increasing it for other examples. Notably, we find that the privacy protection for the remaining training examples may worsen as a consequence of unlearning. We also discuss the fundamental difficulty of equally protecting all examples using existing unlearning schemes, due to the different rates at which examples are unlearned. We demonstrate that naive attempts at tailoring unlearning stopping criteria to different examples fail to alleviate these issues.

摘要: 模型训练的高昂成本使得开发忘却学习的技术变得越来越受欢迎。这些技术寻求消除训练示例的影响，而不必从头开始重新训练模型。直观地说，一旦模型取消学习，与该模型交互的对手应该不再能够判断未学习的示例是否包括在该模型的训练集中。在隐私文献中，这被称为成员关系推断。在这项工作中，我们讨论了成员关系推理攻击(MIA)对遗忘环境的适应(导致它们的‘U-MIA’对应)。我们提出了一种现有U-MIA的分类，其中针对所有示例实例化相同的攻击者，其中针对每个示例实例化一个专用攻击者。我们表明，后一类，其中攻击者根据每个被攻击的例子定制其成员预测，明显更强。事实上，我们的结果表明，遗忘文献中常用的U-MIA高估了现有遗忘技术在视觉和语言模型上提供的隐私保护。我们的调查显示，不同示例对每个示例的U-MIA的脆弱性存在很大差异。事实上，几种忘记算法降低了我们希望忘记的一些(但不是所有)示例的脆弱性，但代价是增加了其他示例的脆弱性。值得注意的是，我们发现，由于遗忘，其余训练样本的隐私保护可能会恶化。我们还讨论了使用现有的遗忘方案平等地保护所有例子的基本困难，因为例子被遗忘的比率不同。我们证明，根据不同的例子调整遗忘停止标准的天真尝试无法缓解这些问题。



## **5. Transfer Learning in Pre-Trained Large Language Models for Malware Detection Based on System Calls**

预训练大型语言模型中的迁移学习用于基于系统调用的恶意软件检测 cs.CR

Submitted to IEEE MILCOM 2024

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2405.09318v1) [paper-pdf](http://arxiv.org/pdf/2405.09318v1)

**Authors**: Pedro Miguel Sánchez Sánchez, Alberto Huertas Celdrán, Gérôme Bovet, Gregorio Martínez Pérez

**Abstract**: In the current cybersecurity landscape, protecting military devices such as communication and battlefield management systems against sophisticated cyber attacks is crucial. Malware exploits vulnerabilities through stealth methods, often evading traditional detection mechanisms such as software signatures. The application of ML/DL in vulnerability detection has been extensively explored in the literature. However, current ML/DL vulnerability detection methods struggle with understanding the context and intent behind complex attacks. Integrating large language models (LLMs) with system call analysis offers a promising approach to enhance malware detection. This work presents a novel framework leveraging LLMs to classify malware based on system call data. The framework uses transfer learning to adapt pre-trained LLMs for malware detection. By retraining LLMs on a dataset of benign and malicious system calls, the models are refined to detect signs of malware activity. Experiments with a dataset of over 1TB of system calls demonstrate that models with larger context sizes, such as BigBird and Longformer, achieve superior accuracy and F1-Score of approximately 0.86. The results highlight the importance of context size in improving detection rates and underscore the trade-offs between computational complexity and performance. This approach shows significant potential for real-time detection in high-stakes environments, offering a robust solution to evolving cyber threats.

摘要: 在当前的网络安全格局中，保护通信和战场管理系统等军事设备免受复杂的网络攻击至关重要。恶意软件通过秘密方式利用漏洞，通常会避开软件签名等传统检测机制。ML/DL在漏洞检测中的应用已经在文献中得到了广泛的探索。然而，当前的ML/DL漏洞检测方法难以理解复杂攻击背后的背景和意图。将大型语言模型(LLM)与系统调用分析相结合，为增强恶意软件检测提供了一种很有前途的方法。该工作提出了一种新的框架，利用LLMS根据系统调用数据对恶意软件进行分类。该框架使用迁移学习来调整预先训练的LLM以检测恶意软件。通过在良性和恶意系统调用的数据集上重新训练LLM，模型被改进以检测恶意软件活动的迹象。使用超过1TB的系统调用数据集进行的实验表明，具有较大上下文大小的模型(如BigBird和LongFormor)具有出色的准确率和大约0.86的F1得分。结果强调了上下文大小在提高检测率方面的重要性，并强调了计算复杂性和性能之间的权衡。这种方法显示出在高风险环境中进行实时检测的巨大潜力，为不断发展的网络威胁提供了强大的解决方案。



## **6. "Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models**

“立即做任何事情”：描述和评估大型语言模型上的In-The-Wild越狱预言 cs.CR

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2308.03825v2) [paper-pdf](http://arxiv.org/pdf/2308.03825v2)

**Authors**: Xinyue Shen, Zeyuan Chen, Michael Backes, Yun Shen, Yang Zhang

**Abstract**: The misuse of large language models (LLMs) has drawn significant attention from the general public and LLM vendors. One particular type of adversarial prompt, known as jailbreak prompt, has emerged as the main attack vector to bypass the safeguards and elicit harmful content from LLMs. In this paper, employing our new framework JailbreakHub, we conduct a comprehensive analysis of 1,405 jailbreak prompts spanning from December 2022 to December 2023. We identify 131 jailbreak communities and discover unique characteristics of jailbreak prompts and their major attack strategies, such as prompt injection and privilege escalation. We also observe that jailbreak prompts increasingly shift from online Web communities to prompt-aggregation websites and 28 user accounts have consistently optimized jailbreak prompts over 100 days. To assess the potential harm caused by jailbreak prompts, we create a question set comprising 107,250 samples across 13 forbidden scenarios. Leveraging this dataset, our experiments on six popular LLMs show that their safeguards cannot adequately defend jailbreak prompts in all scenarios. Particularly, we identify five highly effective jailbreak prompts that achieve 0.95 attack success rates on ChatGPT (GPT-3.5) and GPT-4, and the earliest one has persisted online for over 240 days. We hope that our study can facilitate the research community and LLM vendors in promoting safer and regulated LLMs.

摘要: 大型语言模型(LLM)的滥用引起了公众和LLM供应商的极大关注。一种特殊类型的对抗性提示，即越狱提示，已经成为绕过安全措施并从LLMS引出有害内容的主要攻击媒介。在本文中，我们使用我们的新框架JailBreak Hub，对从2022年12月到2023年12月的1405个越狱提示进行了全面的分析。我们识别了131个越狱社区，发现了越狱提示的独特特征及其主要攻击策略，如提示注入和特权提升。我们还观察到越狱提示越来越多地从在线网络社区转移到提示聚合网站，28个用户账户在100天内持续优化越狱提示。为了评估越狱提示造成的潜在危害，我们创建了一个包含13个禁止场景的107,250个样本的问题集。利用这个数据集，我们在六个流行的LLM上的实验表明，它们的保护措施不足以在所有场景中防御越狱提示。特别是，我们确定了五个高效的越狱提示，它们在ChatGPT(GPT-3.5)和GPT-4上的攻击成功率达到了0.95%，其中最早的一个在线时间超过了240天。我们希望我们的研究能够促进研究界和LLM供应商推广更安全和规范的LLM。



## **7. Large Language Models can be Guided to Evade AI-Generated Text Detection**

可以引导大型语言模型避免人工智能生成的文本检测 cs.CL

TMLR camera ready

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2305.10847v6) [paper-pdf](http://arxiv.org/pdf/2305.10847v6)

**Authors**: Ning Lu, Shengcai Liu, Rui He, Qi Wang, Yew-Soon Ong, Ke Tang

**Abstract**: Large language models (LLMs) have shown remarkable performance in various tasks and have been extensively utilized by the public. However, the increasing concerns regarding the misuse of LLMs, such as plagiarism and spamming, have led to the development of multiple detectors, including fine-tuned classifiers and statistical methods. In this study, we equip LLMs with prompts, rather than relying on an external paraphraser, to evaluate the vulnerability of these detectors. We propose a novel Substitution-based In-Context example Optimization method (SICO) to automatically construct prompts for evading the detectors. SICO is cost-efficient as it requires only 40 human-written examples and a limited number of LLM inferences to generate a prompt. Moreover, once a task-specific prompt has been constructed, it can be universally used against a wide range of detectors. Extensive experiments across three real-world tasks demonstrate that SICO significantly outperforms the paraphraser baselines and enables GPT-3.5 to successfully evade six detectors, decreasing their AUC by 0.5 on average. Furthermore, a comprehensive human evaluation show that the SICO-generated text achieves human-level readability and task completion rates, while preserving high imperceptibility. Finally, we propose an ensemble approach to enhance the robustness of detectors against SICO attack. The code is publicly available at https://github.com/ColinLu50/Evade-GPT-Detector.

摘要: 大型语言模型(LLM)在各种任务中表现出显著的性能，并被公众广泛使用。然而，对LLMS滥用的日益关注，如抄袭和垃圾邮件，导致了多检测器的发展，包括微调分类器和统计方法。在这项研究中，我们为LLMS配备了提示，而不是依赖外部释义来评估这些检测器的脆弱性。我们提出了一种新的基于替换的上下文中实例优化方法(SICO)来自动构建躲避检测器的提示。SICO是具有成本效益的，因为它只需要40个人写的例子和有限数量的LLM推理来生成提示。此外，一旦构建了特定于任务的提示，它就可以普遍用于各种检测器。在三个真实任务上的广泛实验表明，SICO的性能显著优于释义基线，并使GPT-3.5成功避开了六个检测器，使它们的AUC平均降低了0.5%。此外，一项全面的人类评估表明，SICO生成的文本达到了人类水平的可读性和任务完成率，同时保持了高度的不可察觉。最后，我们提出了一种集成方法来增强检测器对SICO攻击的稳健性。该代码可在https://github.com/ColinLu50/Evade-GPT-Detector.上公开获得



## **8. Efficient LLM Jailbreak via Adaptive Dense-to-sparse Constrained Optimization**

通过自适应密集到稀疏约束优化实现高效LLM越狱 cs.LG

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2405.09113v1) [paper-pdf](http://arxiv.org/pdf/2405.09113v1)

**Authors**: Kai Hu, Weichen Yu, Tianjun Yao, Xiang Li, Wenhe Liu, Lijun Yu, Yining Li, Kai Chen, Zhiqiang Shen, Matt Fredrikson

**Abstract**: Recent research indicates that large language models (LLMs) are susceptible to jailbreaking attacks that can generate harmful content. This paper introduces a novel token-level attack method, Adaptive Dense-to-Sparse Constrained Optimization (ADC), which effectively jailbreaks several open-source LLMs. Our approach relaxes the discrete jailbreak optimization into a continuous optimization and progressively increases the sparsity of the optimizing vectors. Consequently, our method effectively bridges the gap between discrete and continuous space optimization. Experimental results demonstrate that our method is more effective and efficient than existing token-level methods. On Harmbench, our method achieves state of the art attack success rate on seven out of eight LLMs. Code will be made available. Trigger Warning: This paper contains model behavior that can be offensive in nature.

摘要: 最近的研究表明，大型语言模型（LLM）很容易受到可能生成有害内容的越狱攻击。本文介绍了一种新颖的代币级攻击方法--自适应密度到稀疏约束优化（ADC），它可以有效地越狱多个开源LLM。我们的方法将离散越狱优化放宽为连续优化，并逐步增加优化载体的稀疏性。因此，我们的方法有效地弥合了离散和连续空间优化之间的差距。实验结果表明，我们的方法比现有的代币级方法更有效和高效。在Harmbener上，我们的方法在八分之七的LLM上实现了最先进的攻击成功率。代码将可用。触发警告：本文包含本质上可能具有冒犯性的模型行为。



## **9. A safety realignment framework via subspace-oriented model fusion for large language models**

通过面向子空间的模型融合为大型语言模型提供安全重新调整框架 cs.CL

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2405.09055v1) [paper-pdf](http://arxiv.org/pdf/2405.09055v1)

**Authors**: Xin Yi, Shunfan Zheng, Linlin Wang, Xiaoling Wang, Liang He

**Abstract**: The current safeguard mechanisms for large language models (LLMs) are indeed susceptible to jailbreak attacks, making them inherently fragile. Even the process of fine-tuning on apparently benign data for downstream tasks can jeopardize safety. One potential solution is to conduct safety fine-tuning subsequent to downstream fine-tuning. However, there's a risk of catastrophic forgetting during safety fine-tuning, where LLMs may regain safety measures but lose the task-specific knowledge acquired during downstream fine-tuning. In this paper, we introduce a safety realignment framework through subspace-oriented model fusion (SOMF), aiming to combine the safeguard capabilities of initially aligned model and the current fine-tuned model into a realigned model. Our approach begins by disentangling all task vectors from the weights of each fine-tuned model. We then identify safety-related regions within these vectors by subspace masking techniques. Finally, we explore the fusion of the initial safely aligned LLM with all task vectors based on the identified safety subspace. We validate that our safety realignment framework satisfies the safety requirements of a single fine-tuned model as well as multiple models during their fusion. Our findings confirm that SOMF preserves safety without notably compromising performance on downstream tasks, including instruction following in Chinese, English, and Hindi, as well as problem-solving capabilities in Code and Math.

摘要: 目前大型语言模型(LLM)的保护机制确实容易受到越狱攻击，这使得它们天生就很脆弱。即使是对下游任务的表面上看是良性的数据进行微调的过程也可能危及安全。一种可能的解决方案是在下游微调之后进行安全微调。然而，在安全微调期间存在灾难性遗忘的风险，在这种情况下，LLM可能重新获得安全措施，但丢失在下游微调期间获得的特定任务的知识。本文通过面向子空间的模型融合(SOMF)提出了一种安全重排框架，旨在将初始对准模型和当前微调模型的保障能力结合到一个重排模型中。我们的方法首先将所有任务向量从每个微调模型的权重中分离出来。然后，我们使用子空间掩蔽技术来识别这些向量中与安全相关的区域。最后，基于识别出的安全子空间，我们探索了初始安全对齐的LLM与所有任务向量的融合。我们验证了我们的安全调整框架满足单个微调模型以及多个模型在融合过程中的安全要求。我们的研究结果证实，SOMF在不显著影响下游任务性能的情况下保持了安全性，包括用中文、英语和印地语进行指导，以及在代码和数学中解决问题的能力。



## **10. Distributed Threat Intelligence at the Edge Devices: A Large Language Model-Driven Approach**

边缘设备上的分布式威胁情报：大型语言模型驱动的方法 cs.CR

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.08755v1) [paper-pdf](http://arxiv.org/pdf/2405.08755v1)

**Authors**: Syed Mhamudul Hasan, Alaa M. Alotaibi, Sajedul Talukder, Abdur R. Shahid

**Abstract**: With the proliferation of edge devices, there is a significant increase in attack surface on these devices. The decentralized deployment of threat intelligence on edge devices, coupled with adaptive machine learning techniques such as the in-context learning feature of large language models (LLMs), represents a promising paradigm for enhancing cybersecurity on low-powered edge devices. This approach involves the deployment of lightweight machine learning models directly onto edge devices to analyze local data streams, such as network traffic and system logs, in real-time. Additionally, distributing computational tasks to an edge server reduces latency and improves responsiveness while also enhancing privacy by processing sensitive data locally. LLM servers can enable these edge servers to autonomously adapt to evolving threats and attack patterns, continuously updating their models to improve detection accuracy and reduce false positives. Furthermore, collaborative learning mechanisms facilitate peer-to-peer secure and trustworthy knowledge sharing among edge devices, enhancing the collective intelligence of the network and enabling dynamic threat mitigation measures such as device quarantine in response to detected anomalies. The scalability and flexibility of this approach make it well-suited for diverse and evolving network environments, as edge devices only send suspicious information such as network traffic and system log changes, offering a resilient and efficient solution to combat emerging cyber threats at the network edge. Thus, our proposed framework can improve edge computing security by providing better security in cyber threat detection and mitigation by isolating the edge devices from the network.

摘要: 随着边缘设备的扩散，这些设备上的攻击面显著增加。在边缘设备上分散部署威胁情报，再加上自适应机器学习技术，如大型语言模型(LLMS)的上下文中学习功能，代表了一种在低性能边缘设备上增强网络安全的有前途的范例。这种方法涉及将轻量级机器学习模型直接部署到边缘设备上，以实时分析本地数据流，如网络流量和系统日志。此外，将计算任务分发到边缘服务器可减少延迟并提高响应速度，同时还可通过在本地处理敏感数据来增强隐私。LLM服务器可以使这些边缘服务器自主适应不断变化的威胁和攻击模式，不断更新其模型以提高检测准确性并减少误报。此外，协作学习机制促进了边缘设备之间的对等安全和可信知识共享，增强了网络的集体智能，并支持动态威胁缓解措施，如响应检测到的异常情况的设备隔离。这种方法的可扩展性和灵活性使其非常适合于多样化和不断发展的网络环境，因为边缘设备只发送可疑信息，如网络流量和系统日志更改，为应对网络边缘出现的网络威胁提供了一种弹性和高效的解决方案。因此，我们提出的框架可以通过将边缘设备与网络隔离来提供更好的网络威胁检测和缓解的安全性，从而提高边缘计算的安全性。



## **11. PLeak: Prompt Leaking Attacks against Large Language Model Applications**

PLeak：针对大型语言模型应用程序的提示泄露攻击 cs.CR

To appear in the Proceedings of The ACM Conference on Computer and  Communications Security (CCS), 2024

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.06823v2) [paper-pdf](http://arxiv.org/pdf/2405.06823v2)

**Authors**: Bo Hui, Haolin Yuan, Neil Gong, Philippe Burlina, Yinzhi Cao

**Abstract**: Large Language Models (LLMs) enable a new ecosystem with many downstream applications, called LLM applications, with different natural language processing tasks. The functionality and performance of an LLM application highly depend on its system prompt, which instructs the backend LLM on what task to perform. Therefore, an LLM application developer often keeps a system prompt confidential to protect its intellectual property. As a result, a natural attack, called prompt leaking, is to steal the system prompt from an LLM application, which compromises the developer's intellectual property. Existing prompt leaking attacks primarily rely on manually crafted queries, and thus achieve limited effectiveness.   In this paper, we design a novel, closed-box prompt leaking attack framework, called PLeak, to optimize an adversarial query such that when the attacker sends it to a target LLM application, its response reveals its own system prompt. We formulate finding such an adversarial query as an optimization problem and solve it with a gradient-based method approximately. Our key idea is to break down the optimization goal by optimizing adversary queries for system prompts incrementally, i.e., starting from the first few tokens of each system prompt step by step until the entire length of the system prompt.   We evaluate PLeak in both offline settings and for real-world LLM applications, e.g., those on Poe, a popular platform hosting such applications. Our results show that PLeak can effectively leak system prompts and significantly outperforms not only baselines that manually curate queries but also baselines with optimized queries that are modified and adapted from existing jailbreaking attacks. We responsibly reported the issues to Poe and are still waiting for their response. Our implementation is available at this repository: https://github.com/BHui97/PLeak.

摘要: 大型语言模型(LLM)支持一个新的生态系统，该生态系统具有许多下游应用程序，称为LLM应用程序，具有不同的自然语言处理任务。LLM应用程序的功能和性能高度依赖于其系统提示符，系统提示符指示后端LLM执行什么任务。因此，LLM应用程序开发人员通常会对系统提示保密，以保护其知识产权。因此，一种称为提示泄漏的自然攻击是从LLM应用程序中窃取系统提示，这会损害开发人员的知识产权。现有的即时泄漏攻击主要依赖于手动创建的查询，因此效果有限。在本文中，我们设计了一个新颖的封闭盒提示泄漏攻击框架PLeak，用于优化敌意查询，使其在攻击者将其发送到目标LLM应用程序时，其响应显示其自己的系统提示。我们将寻找这样一个敌意查询描述为一个优化问题，并用基于梯度的方法近似求解。我们的核心思想是通过对系统提示的敌意查询进行增量优化来打破优化目标，即从每个系统提示的前几个令牌开始逐步优化，直到系统提示的整个长度。我们在离线设置和现实世界的LLM应用程序(例如，托管此类应用程序的流行平台PoE上的应用程序)中对PLeak进行评估。我们的结果表明，PLeak能够有效地泄露系统提示，不仅显著优于手动管理查询的基线，而且显著优于从现有越狱攻击中修改和调整的优化查询的基线。我们负责任地向PoE报告了这些问题，并仍在等待他们的回应。我们的实现可从以下存储库获得：https://github.com/BHui97/PLeak.



## **12. Stylometric Watermarks for Large Language Models**

大型语言模型的文体水印 cs.CL

19 pages, 4 figures, 9 tables

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.08400v1) [paper-pdf](http://arxiv.org/pdf/2405.08400v1)

**Authors**: Georg Niess, Roman Kern

**Abstract**: The rapid advancement of large language models (LLMs) has made it increasingly difficult to distinguish between text written by humans and machines. Addressing this, we propose a novel method for generating watermarks that strategically alters token probabilities during generation. Unlike previous works, this method uniquely employs linguistic features such as stylometry. Concretely, we introduce acrostica and sensorimotor norms to LLMs. Further, these features are parameterized by a key, which is updated every sentence. To compute this key, we use semantic zero shot classification, which enhances resilience. In our evaluation, we find that for three or more sentences, our method achieves a false positive and false negative rate of 0.02. For the case of a cyclic translation attack, we observe similar results for seven or more sentences. This research is of particular of interest for proprietary LLMs to facilitate accountability and prevent societal harm.

摘要: 大型语言模型（LLM）的快速发展使得区分人类和机器编写的文本变得越来越困难。针对这一点，我们提出了一种生成水印的新颖方法，该方法在生成过程中策略性地改变代币概率。与之前的作品不同，这种方法独特地采用了风格等语言特征。具体来说，我们向LLM引入了极乐律和感觉运动规范。此外，这些功能由一个键参数化，该键每句都会更新一次。为了计算这个密钥，我们使用语义零次分类，这增强了弹性。在我们的评估中，我们发现对于三个或更多句子，我们的方法实现了0.02的假阳性和假阴性率。对于循环翻译攻击的情况，我们观察到七个或更多句子的类似结果。这项研究对于专有LLM特别感兴趣，以促进问责制并防止社会危害。



## **13. SpeechGuard: Exploring the Adversarial Robustness of Multimodal Large Language Models**

SpeechGuard：探索多模式大型语言模型的对抗鲁棒性 cs.CL

9+6 pages, Submitted to ACL 2024

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.08317v1) [paper-pdf](http://arxiv.org/pdf/2405.08317v1)

**Authors**: Raghuveer Peri, Sai Muralidhar Jayanthi, Srikanth Ronanki, Anshu Bhatia, Karel Mundnich, Saket Dingliwal, Nilaksh Das, Zejiang Hou, Goeric Huybrechts, Srikanth Vishnubhotla, Daniel Garcia-Romero, Sundararajan Srinivasan, Kyu J Han, Katrin Kirchhoff

**Abstract**: Integrated Speech and Large Language Models (SLMs) that can follow speech instructions and generate relevant text responses have gained popularity lately. However, the safety and robustness of these models remains largely unclear. In this work, we investigate the potential vulnerabilities of such instruction-following speech-language models to adversarial attacks and jailbreaking. Specifically, we design algorithms that can generate adversarial examples to jailbreak SLMs in both white-box and black-box attack settings without human involvement. Additionally, we propose countermeasures to thwart such jailbreaking attacks. Our models, trained on dialog data with speech instructions, achieve state-of-the-art performance on spoken question-answering task, scoring over 80% on both safety and helpfulness metrics. Despite safety guardrails, experiments on jailbreaking demonstrate the vulnerability of SLMs to adversarial perturbations and transfer attacks, with average attack success rates of 90% and 10% respectively when evaluated on a dataset of carefully designed harmful questions spanning 12 different toxic categories. However, we demonstrate that our proposed countermeasures reduce the attack success significantly.

摘要: 集成的语音和大型语言模型(SLM)可以遵循语音指令并生成相关的文本响应，最近得到了广泛的应用。然而，这些模型的安全性和稳健性在很大程度上仍不清楚。在这项工作中，我们调查了这种遵循指令的语音语言模型在对抗攻击和越狱时的潜在脆弱性。具体地说，我们设计的算法可以生成白盒和黑盒攻击环境下的越狱SLM的对抗性示例，而不需要人工参与。此外，我们还提出了挫败此类越狱攻击的对策。我们的模型在对话数据和语音指令上进行了训练，在口语问答任务中实现了最先进的性能，在安全性和有助性指标上都获得了80%以上的分数。尽管有安全护栏，但越狱实验证明了SLM在对抗性扰动和转移攻击中的脆弱性，当对12个不同有毒类别的精心设计的有害问题集进行评估时，平均攻击成功率分别为90%和10%。然而，我们证明我们提出的对策显著降低了攻击的成功率。



## **14. Many-Shot Regurgitation (MSR) Prompting**

多镜头回归（MSR）回归 cs.CL

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.08134v1) [paper-pdf](http://arxiv.org/pdf/2405.08134v1)

**Authors**: Shashank Sonkar, Richard G. Baraniuk

**Abstract**: We introduce Many-Shot Regurgitation (MSR) prompting, a new black-box membership inference attack framework for examining verbatim content reproduction in large language models (LLMs). MSR prompting involves dividing the input text into multiple segments and creating a single prompt that includes a series of faux conversation rounds between a user and a language model to elicit verbatim regurgitation. We apply MSR prompting to diverse text sources, including Wikipedia articles and open educational resources (OER) textbooks, which provide high-quality, factual content and are continuously updated over time. For each source, we curate two dataset types: one that LLMs were likely exposed to during training ($D_{\rm pre}$) and another consisting of documents published after the models' training cutoff dates ($D_{\rm post}$). To quantify the occurrence of verbatim matches, we employ the Longest Common Substring algorithm and count the frequency of matches at different length thresholds. We then use statistical measures such as Cliff's delta, Kolmogorov-Smirnov (KS) distance, and Kruskal-Wallis H test to determine whether the distribution of verbatim matches differs significantly between $D_{\rm pre}$ and $D_{\rm post}$. Our findings reveal a striking difference in the distribution of verbatim matches between $D_{\rm pre}$ and $D_{\rm post}$, with the frequency of verbatim reproduction being significantly higher when LLMs (e.g. GPT models and LLaMAs) are prompted with text from datasets they were likely trained on. For instance, when using GPT-3.5 on Wikipedia articles, we observe a substantial effect size (Cliff's delta $= -0.984$) and a large KS distance ($0.875$) between the distributions of $D_{\rm pre}$ and $D_{\rm post}$. Our results provide compelling evidence that LLMs are more prone to reproducing verbatim content when the input text is likely sourced from their training data.

摘要: 介绍了一种新的黑盒成员关系推理攻击框架--多射反流(MSR)提示，用于检测大型语言模型(LLMS)中的逐字内容再现。MSR提示包括将输入文本分成多个片段，并创建单个提示，其中包括用户和语言模型之间的一系列虚假对话回合，以引发逐字反胃。我们将MSR提示应用于不同的文本来源，包括维基百科文章和开放教育资源(OER)教科书，这些内容提供高质量的事实内容，并随着时间的推移不断更新。对于每个源，我们管理两种数据集类型：一种是LLM在培训期间可能接触到的($D_{\RM Pre}$)，另一种是在模型的培训截止日期之后发布的文档($D_{\RM POST}$)。为了量化逐字匹配的发生，我们使用了最长公共子串算法，并统计了不同长度阈值下的匹配频率。然后，我们使用诸如Cliff‘s Delta、Kolmogorov-Smirnov(KS)距离和Kruskal-Wallis H检验等统计指标来确定$D_(\rm)$和$D_(\rm)_(POST)$之间逐字匹配的分布是否有显著差异。我们的发现揭示了$D_{\rm Pre}$和$D_{\rm post}$逐字匹配的分布存在显著差异，当提示LLM(例如GPT模型和大羊驼)时，当提示LLMS(例如GPT模型和大羊驼)的文本来自他们可能训练过的数据集时，逐字再现的频率显著高于$D_{\rm Pre}$和$D_{\rm post}$。例如，当在维基百科文章上使用GPT-3.5时，我们观察到$D_{\rM Pre}$和$D_{\rm post}$的分布之间有很大的效果大小(Cliff‘s Delta$=-0.984$)和很大的KS距离($0.875$)。我们的结果提供了令人信服的证据，表明当输入文本可能来自于他们的训练数据时，LLM更倾向于逐字再现内容。



## **15. Backdoor Removal for Generative Large Language Models**

生成性大型语言模型的后门删除 cs.CR

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07667v1) [paper-pdf](http://arxiv.org/pdf/2405.07667v1)

**Authors**: Haoran Li, Yulin Chen, Zihao Zheng, Qi Hu, Chunkit Chan, Heshan Liu, Yangqiu Song

**Abstract**: With rapid advances, generative large language models (LLMs) dominate various Natural Language Processing (NLP) tasks from understanding to reasoning. Yet, language models' inherent vulnerabilities may be exacerbated due to increased accessibility and unrestricted model training on massive textual data from the Internet. A malicious adversary may publish poisoned data online and conduct backdoor attacks on the victim LLMs pre-trained on the poisoned data. Backdoored LLMs behave innocuously for normal queries and generate harmful responses when the backdoor trigger is activated. Despite significant efforts paid to LLMs' safety issues, LLMs are still struggling against backdoor attacks. As Anthropic recently revealed, existing safety training strategies, including supervised fine-tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF), fail to revoke the backdoors once the LLM is backdoored during the pre-training stage. In this paper, we present Simulate and Eliminate (SANDE) to erase the undesired backdoored mappings for generative LLMs. We initially propose Overwrite Supervised Fine-tuning (OSFT) for effective backdoor removal when the trigger is known. Then, to handle the scenarios where the trigger patterns are unknown, we integrate OSFT into our two-stage framework, SANDE. Unlike previous works that center on the identification of backdoors, our safety-enhanced LLMs are able to behave normally even when the exact triggers are activated. We conduct comprehensive experiments to show that our proposed SANDE is effective against backdoor attacks while bringing minimal harm to LLMs' powerful capability without any additional access to unbackdoored clean models. We will release the reproducible code.

摘要: 随着研究的深入，从理解到推理的各种自然语言处理任务都被生成性大语言模型(LLMS)所支配。然而，语言模型固有的脆弱性可能会因为可访问性的提高和对来自互联网的海量文本数据的不受限制的模型训练而加剧。恶意对手可能会在网上发布有毒数据，并对受害者LLM进行后门攻击，这些LLM预先训练了有毒数据。后门LLM在正常查询中的行为是无害的，并在激活后门触发器时生成有害的响应。尽管在LLMS的安全问题上付出了巨大的努力，但LLMS仍在努力应对后门攻击。正如人类最近揭示的那样，现有的安全培训策略，包括监督微调(SFT)和从人类反馈的强化学习(RLHF)，一旦LLM在培训前阶段后退，就无法取消后门。在这篇文章中，我们提出了模拟和消除(SANDE)来消除生成式LLMS中不需要的回溯映射。我们最初提出了覆盖监督精调(OSFT)，用于在已知触发器的情况下有效地删除后门。然后，为了处理触发模式未知的场景，我们将OSFT集成到我们的两阶段框架SANDE中。与以前以识别后门为中心的工作不同，我们的安全增强型LLM即使在准确的触发器被激活时也能够正常运行。我们进行了全面的实验，以表明我们提出的SANDE可以有效地抵御后门攻击，同时对LLMS的强大功能造成的损害最小，而不需要额外访问未后门的干净模型。我们将发布可重现的代码。



## **16. DoLLM: How Large Language Models Understanding Network Flow Data to Detect Carpet Bombing DDoS**

DoLLM：大型语言模型如何理解网络流数据来检测地毯炸弹DDOS cs.NI

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07638v1) [paper-pdf](http://arxiv.org/pdf/2405.07638v1)

**Authors**: Qingyang Li, Yihang Zhang, Zhidong Jia, Yannan Hu, Lei Zhang, Jianrong Zhang, Yongming Xu, Yong Cui, Zongming Guo, Xinggong Zhang

**Abstract**: It is an interesting question Can and How Large Language Models (LLMs) understand non-language network data, and help us detect unknown malicious flows. This paper takes Carpet Bombing as a case study and shows how to exploit LLMs' powerful capability in the networking area. Carpet Bombing is a new DDoS attack that has dramatically increased in recent years, significantly threatening network infrastructures. It targets multiple victim IPs within subnets, causing congestion on access links and disrupting network services for a vast number of users. Characterized by low-rates, multi-vectors, these attacks challenge traditional DDoS defenses. We propose DoLLM, a DDoS detection model utilizes open-source LLMs as backbone. By reorganizing non-contextual network flows into Flow-Sequences and projecting them into LLMs semantic space as token embeddings, DoLLM leverages LLMs' contextual understanding to extract flow representations in overall network context. The representations are used to improve the DDoS detection performance. We evaluate DoLLM with public datasets CIC-DDoS2019 and real NetFlow trace from Top-3 countrywide ISP. The tests have proven that DoLLM possesses strong detection capabilities. Its F1 score increased by up to 33.3% in zero-shot scenarios and by at least 20.6% in real ISP traces.

摘要: 大型语言模型能够以及如何理解非语言网络数据，并帮助我们检测未知的恶意流量，这是一个有趣的问题。本文以地毯轰炸为例，展示了如何利用低成本管理系统在组网领域的强大能力。地毯式轰炸是一种新型的DDoS攻击，近年来急剧增加，严重威胁着网络基础设施。它以子网内的多个受害IP为目标，导致接入链路拥塞，扰乱了大量用户的网络服务。这些攻击以低速率、多向量为特征，挑战了传统的DDoS防御。我们提出了DoLLM，一个以开源LLMS为骨干的DDoS检测模型。通过将非上下文网络流重新组织成流序列，并将其投影到LLMS语义空间作为令牌嵌入，DoLLM利用LLMS的上下文理解来提取整个网络上下文中的流表示。这些表示用于提高DDoS检测性能。我们使用公共数据集CIC-DDoS2019和来自全国排名前三的运营商的真实NetFlow跟踪来评估DoLLM。测试证明，DoLLM具有很强的检测能力。它的F1得分在零镜头场景下增加了33.3%，在真实的isp轨迹中至少增加了20.6%。



## **17. ExplainableDetector: Exploring Transformer-based Language Modeling Approach for SMS Spam Detection with Explainability Analysis**

解释性检测器：通过可解释性分析探索基于转换器的语言建模方法用于短信垃圾邮件检测 cs.LG

**SubmitDate**: 2024-05-12    [abs](http://arxiv.org/abs/2405.08026v1) [paper-pdf](http://arxiv.org/pdf/2405.08026v1)

**Authors**: Mohammad Amaz Uddin, Muhammad Nazrul Islam, Leandros Maglaras, Helge Janicke, Iqbal H. Sarker

**Abstract**: SMS, or short messaging service, is a widely used and cost-effective communication medium that has sadly turned into a haven for unwanted messages, commonly known as SMS spam. With the rapid adoption of smartphones and Internet connectivity, SMS spam has emerged as a prevalent threat. Spammers have taken notice of the significance of SMS for mobile phone users. Consequently, with the emergence of new cybersecurity threats, the number of SMS spam has expanded significantly in recent years. The unstructured format of SMS data creates significant challenges for SMS spam detection, making it more difficult to successfully fight spam attacks in the cybersecurity domain. In this work, we employ optimized and fine-tuned transformer-based Large Language Models (LLMs) to solve the problem of spam message detection. We use a benchmark SMS spam dataset for this spam detection and utilize several preprocessing techniques to get clean and noise-free data and solve the class imbalance problem using the text augmentation technique. The overall experiment showed that our optimized fine-tuned BERT (Bidirectional Encoder Representations from Transformers) variant model RoBERTa obtained high accuracy with 99.84\%. We also work with Explainable Artificial Intelligence (XAI) techniques to calculate the positive and negative coefficient scores which explore and explain the fine-tuned model transparency in this text-based spam SMS detection task. In addition, traditional Machine Learning (ML) models were also examined to compare their performance with the transformer-based models. This analysis describes how LLMs can make a good impact on complex textual-based spam data in the cybersecurity field.

摘要: 短信，或短消息服务，是一种广泛使用且具有成本效益的通信媒介，遗憾的是，它已经变成了不想要的消息的避风港，通常被称为短信垃圾邮件。随着智能手机和互联网连接的迅速普及，垃圾短信已经成为一种普遍的威胁。垃圾邮件发送者已经注意到短信对手机用户的重要性。因此，随着新的网络安全威胁的出现，近年来短信垃圾邮件的数量大幅增加。短信数据的非结构化格式给短信垃圾邮件检测带来了巨大的挑战，使得在网络安全领域成功打击垃圾邮件攻击变得更加困难。在这项工作中，我们使用优化和微调的基于转换器的大语言模型(LLMS)来解决垃圾消息检测问题。我们使用一个基准的短信垃圾邮件数据集进行垃圾邮件检测，并利用几种预处理技术来获得干净和无噪声的数据，并使用文本增强技术解决类别不平衡问题。整体实验表明，优化后的BERT(Transformers的双向编码器表示)变体模型Roberta获得了99.84%的高精度。我们还使用可解释人工智能(XAI)技术来计算正系数和负系数得分，从而探索和解释了在这个基于文本的垃圾短信检测任务中微调的模型透明度。此外，还研究了传统的机器学习(ML)模型，并与基于变压器的模型进行了比较。这一分析描述了LLMS如何在网络安全领域对复杂的基于文本的垃圾数据产生良好的影响。



## **18. The Janus Interface: How Fine-Tuning in Large Language Models Amplifies the Privacy Risks**

Janus界面：大型语言模型中的微调如何放大隐私风险 cs.CR

**SubmitDate**: 2024-05-12    [abs](http://arxiv.org/abs/2310.15469v2) [paper-pdf](http://arxiv.org/pdf/2310.15469v2)

**Authors**: Xiaoyi Chen, Siyuan Tang, Rui Zhu, Shijun Yan, Lei Jin, Zihao Wang, Liya Su, Zhikun Zhang, XiaoFeng Wang, Haixu Tang

**Abstract**: The rapid advancements of large language models (LLMs) have raised public concerns about the privacy leakage of personally identifiable information (PII) within their extensive training datasets. Recent studies have demonstrated that an adversary could extract highly sensitive privacy data from the training data of LLMs with carefully designed prompts. However, these attacks suffer from the model's tendency to hallucinate and catastrophic forgetting (CF) in the pre-training stage, rendering the veracity of divulged PIIs negligible. In our research, we propose a novel attack, Janus, which exploits the fine-tuning interface to recover forgotten PIIs from the pre-training data in LLMs. We formalize the privacy leakage problem in LLMs and explain why forgotten PIIs can be recovered through empirical analysis on open-source language models. Based upon these insights, we evaluate the performance of Janus on both open-source language models and two latest LLMs, i.e., GPT-3.5-Turbo and LLaMA-2-7b. Our experiment results show that Janus amplifies the privacy risks by over 10 times in comparison with the baseline and significantly outperforms the state-of-the-art privacy extraction attacks including prefix attacks and in-context learning (ICL). Furthermore, our analysis validates that existing fine-tuning APIs provided by OpenAI and Azure AI Studio are susceptible to our Janus attack, allowing an adversary to conduct such an attack at a low cost.

摘要: 大型语言模型(LLM)的快速发展引起了公众对其广泛训练数据集中个人身份信息(PII)隐私泄露的担忧。最近的研究表明，攻击者可以通过精心设计的提示从LLMS的训练数据中提取高度敏感的隐私数据。然而，这些攻击受到模型在预训练阶段的幻觉和灾难性遗忘(CF)的倾向的影响，使得泄露的PII的真实性可以忽略不计。在我们的研究中，我们提出了一种新的攻击，Janus，它利用微调接口从LLMS的训练前数据中恢复被遗忘的PII。我们形式化地描述了LLMS中的隐私泄露问题，并通过对开源语言模型的实证分析解释了为什么被遗忘的PII可以恢复。基于这些见解，我们评估了Janus在开源语言模型和两个最新的LLMS上的性能，即GPT-3.5-Turbo和Llama-2-7b。我们的实验结果表明，Janus将隐私风险放大了10倍以上，并且显著优于目前最先进的隐私提取攻击，包括前缀攻击和上下文中学习(ICL)。此外，我们的分析验证了OpenAI和Azure AI Studio提供的现有微调API容易受到我们的Janus攻击，允许对手以低成本进行此类攻击。



## **19. LLMs and the Future of Chip Design: Unveiling Security Risks and Building Trust**

法学硕士和芯片设计的未来：揭示安全风险并建立信任 cs.LG

**SubmitDate**: 2024-05-11    [abs](http://arxiv.org/abs/2405.07061v1) [paper-pdf](http://arxiv.org/pdf/2405.07061v1)

**Authors**: Zeng Wang, Lilas Alrahis, Likhitha Mankali, Johann Knechtel, Ozgur Sinanoglu

**Abstract**: Chip design is about to be revolutionized by the integration of large language, multimodal, and circuit models (collectively LxMs). While exploring this exciting frontier with tremendous potential, the community must also carefully consider the related security risks and the need for building trust into using LxMs for chip design. First, we review the recent surge of using LxMs for chip design in general. We cover state-of-the-art works for the automation of hardware description language code generation and for scripting and guidance of essential but cumbersome tasks for electronic design automation tools, e.g., design-space exploration, tuning, or designer training. Second, we raise and provide initial answers to novel research questions on critical issues for security and trustworthiness of LxM-powered chip design from both the attack and defense perspectives.

摘要: 芯片设计即将因大型语言、多模式和电路模型（统称为LxM）的集成而发生革命性变化。在探索这个具有巨大潜力的令人兴奋的前沿时，社区还必须仔细考虑相关的安全风险以及在使用LxM进行芯片设计方面建立信任的必要性。首先，我们回顾了最近使用LxM进行芯片设计的热潮。我们涵盖硬件描述语言代码生成自动化以及电子设计自动化工具的重要但繁琐任务的脚本和指导的最先进作品，例如，设计空间探索、调整或设计师培训。其次，我们从攻击和防御的角度对LxM驱动芯片设计的安全性和可信性的关键问题提出并提供了初步答案。



## **20. Talk Too Much: Poisoning Large Language Models under Token Limit**

话太多：代币限制下的大型语言模型中毒 cs.CL

**SubmitDate**: 2024-05-11    [abs](http://arxiv.org/abs/2404.14795v3) [paper-pdf](http://arxiv.org/pdf/2404.14795v3)

**Authors**: Jiaming He, Wenbo Jiang, Guanyu Hou, Wenshu Fan, Rui Zhang, Hongwei Li

**Abstract**: Mainstream poisoning attacks on large language models (LLMs) typically set a fixed trigger in the input instance and specific responses for triggered queries. However, the fixed trigger setting (e.g., unusual words) may be easily detected by human detection, limiting the effectiveness and practicality in real-world scenarios. To enhance the stealthiness of the trigger, we present a poisoning attack against LLMs that is triggered by a generation/output condition-token limitation, which is a commonly adopted strategy by users for reducing costs. The poisoned model performs normally for output without token limitation, while becomes harmful for output with limited tokens. To achieve this objective, we introduce BrieFool, an efficient attack framework. It leverages the characteristics of generation limitation by efficient instruction sampling and poisoning data generation, thereby influencing the behavior of LLMs under target conditions. Our experiments demonstrate that BrieFool is effective across safety domains and knowledge domains. For instance, with only 20 generated poisoning examples against GPT-3.5-turbo, BrieFool achieves a 100% Attack Success Rate (ASR) and a 9.28/10 average Harmfulness Score (HS) under token limitation conditions while maintaining the benign performance.

摘要: 针对大型语言模型(LLM)的主流中毒攻击通常会在输入实例中设置固定的触发器，并为触发的查询设置特定的响应。然而，固定的触发设置(例如，不寻常的单词)可能很容易被人类检测到，从而限制了在现实世界场景中的有效性和实用性。为了增强触发器的隐蔽性，我们提出了一种由生成/输出条件-令牌限制触发的针对LLMS的中毒攻击，这是用户为降低成本而常用的策略。对于没有令牌限制的输出，中毒模型执行正常，而对于具有有限令牌的输出则变得有害。为了实现这一目标，我们引入了一种高效的攻击框架BrieFool。它通过高效的指令采样和中毒数据生成来利用生成限制的特性，从而影响LLMS在目标条件下的行为。我们的实验表明，BrieFool是跨安全域和知识域的有效的。例如，在对GPT-3.5-Turbo仅生成20个中毒实例的情况下，BrieFool在保持良性性能的同时，在令牌限制条件下实现了100%的攻击成功率(ASR)和9.28/10的平均危害性评分(HS)。



## **21. Explaining Arguments' Strength: Unveiling the Role of Attacks and Supports (Technical Report)**

解释争论的力量：揭示攻击和支持的作用（技术报告） cs.AI

This paper has been accepted at IJCAI 2024 (the 33rd International  Joint Conference on Artificial Intelligence)

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2404.14304v2) [paper-pdf](http://arxiv.org/pdf/2404.14304v2)

**Authors**: Xiang Yin, Potyka Nico, Francesca Toni

**Abstract**: Quantitatively explaining the strength of arguments under gradual semantics has recently received increasing attention. Specifically, several works in the literature provide quantitative explanations by computing the attribution scores of arguments. These works disregard the importance of attacks and supports, even though they play an essential role when explaining arguments' strength. In this paper, we propose a novel theory of Relation Attribution Explanations (RAEs), adapting Shapley values from game theory to offer fine-grained insights into the role of attacks and supports in quantitative bipolar argumentation towards obtaining the arguments' strength. We show that RAEs satisfy several desirable properties. We also propose a probabilistic algorithm to approximate RAEs efficiently. Finally, we show the application value of RAEs in fraud detection and large language models case studies.

摘要: 在渐进语义下定量解释论点的强度最近受到越来越多的关注。具体来说，文献中的几部作品通过计算论点的归因分数来提供量化解释。这些作品忽视了攻击和支持的重要性，尽管它们在解释论点的强度时发挥着至关重要的作用。在本文中，我们提出了一种新的关系归因解释（RAEs）理论，改编了博弈论中的沙普利价值观，以提供对攻击作用的细粒度见解，并支持量化两极论证，以获得论点的强度。我们表明RAE满足几个理想的性质。我们还提出了一种有效逼近RAE的概率算法。最后，我们展示了RAE在欺诈检测和大型语言模型案例研究中的应用价值。



## **22. Risks of Practicing Large Language Models in Smart Grid: Threat Modeling and Validation**

在智能电网中实践大型语言模型的风险：威胁建模和验证 cs.CR

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06237v1) [paper-pdf](http://arxiv.org/pdf/2405.06237v1)

**Authors**: Jiangnan Li, Yingyuan Yang, Jinyuan Sun

**Abstract**: Large Language Model (LLM) is a significant breakthrough in artificial intelligence (AI) and holds considerable potential for application within smart grids. However, as demonstrated in previous literature, AI technologies are susceptible to various types of attacks. It is crucial to investigate and evaluate the risks associated with LLMs before deploying them in critical infrastructure like smart grids. In this paper, we systematically evaluate the vulnerabilities of LLMs and identify two major types of attacks relevant to smart grid LLM applications, along with presenting the corresponding threat models. We then validate these attacks using popular LLMs, utilizing real smart grid data. Our validation demonstrates that attackers are capable of injecting bad data and retrieving domain knowledge from LLMs employed in smart grid scenarios.

摘要: 大型语言模型（LLM）是人工智能（AI）领域的重大突破，在智能电网中具有巨大的应用潜力。然而，正如之前的文献所证明的那样，人工智能技术容易受到各种类型的攻击。在将LLM部署在智能电网等关键基础设施中之前，调查和评估与LLM相关的风险至关重要。在本文中，我们系统地评估了LLM的漏洞，识别了与智能电网LLM应用相关的两种主要攻击类型，并提出了相应的威胁模型。然后，我们使用流行的LLM并利用真实的智能电网数据来验证这些攻击。我们的验证表明，攻击者能够注入不良数据并从智能电网场景中使用的LLM中检索领域知识。



## **23. Muting Whisper: A Universal Acoustic Adversarial Attack on Speech Foundation Models**

静音低语：对语音基础模型的通用声学对抗攻击 cs.CL

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.06134v1) [paper-pdf](http://arxiv.org/pdf/2405.06134v1)

**Authors**: Vyas Raina, Rao Ma, Charles McGhee, Kate Knill, Mark Gales

**Abstract**: Recent developments in large speech foundation models like Whisper have led to their widespread use in many automatic speech recognition (ASR) applications. These systems incorporate `special tokens' in their vocabulary, such as $\texttt{<endoftext>}$, to guide their language generation process. However, we demonstrate that these tokens can be exploited by adversarial attacks to manipulate the model's behavior. We propose a simple yet effective method to learn a universal acoustic realization of Whisper's $\texttt{<endoftext>}$ token, which, when prepended to any speech signal, encourages the model to ignore the speech and only transcribe the special token, effectively `muting' the model. Our experiments demonstrate that the same, universal 0.64-second adversarial audio segment can successfully mute a target Whisper ASR model for over 97\% of speech samples. Moreover, we find that this universal adversarial audio segment often transfers to new datasets and tasks. Overall this work demonstrates the vulnerability of Whisper models to `muting' adversarial attacks, where such attacks can pose both risks and potential benefits in real-world settings: for example the attack can be used to bypass speech moderation systems, or conversely the attack can also be used to protect private speech data.

摘要: 像Whisper这样的大型语音基础模型的最新发展导致它们在许多自动语音识别(ASR)应用中被广泛使用。这些系统在它们的词汇表中加入了“特殊记号”，如$\exttt{<endoftext>}$，以指导它们的语言生成过程。然而，我们证明了这些令牌可以被敌意攻击利用来操纵模型的行为。我们提出了一种简单而有效的方法来学习Whisper的$\exttt{<endoftext>}$标记的通用声学实现，当预先添加到任何语音信号时，鼓励模型忽略语音，只转录特殊的标记，从而有效地抑制了模型。我们的实验表明，相同的、通用的0.64秒的对抗性音频片段可以成功地使目标Whisper ASR模型在97%以上的语音样本上静音。此外，我们发现这种普遍的对抗性音频片段经常转移到新的数据集和任务。总体而言，这项工作证明了Whisper模型对“静音”对手攻击的脆弱性，在现实世界中，这种攻击既可以带来风险，也可以带来潜在的好处：例如，攻击可以用来绕过语音调节系统，或者反过来，攻击也可以用来保护私人语音数据。



## **24. Trustworthy AI-Generative Content in Intelligent 6G Network: Adversarial, Privacy, and Fairness**

智能6G网络中值得信赖的人工智能生成内容：对抗性、隐私性和公平性 cs.CR

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.05930v1) [paper-pdf](http://arxiv.org/pdf/2405.05930v1)

**Authors**: Siyuan Li, Xi Lin, Yaju Liu, Jianhua Li

**Abstract**: AI-generated content (AIGC) models, represented by large language models (LLM), have brought revolutionary changes to the content generation fields. The high-speed and extensive 6G technology is an ideal platform for providing powerful AIGC mobile service applications, while future 6G mobile networks also need to support intelligent and personalized mobile generation services. However, the significant ethical and security issues of current AIGC models, such as adversarial attacks, privacy, and fairness, greatly affect the credibility of 6G intelligent networks, especially in ensuring secure, private, and fair AIGC applications. In this paper, we propose TrustGAIN, a novel paradigm for trustworthy AIGC in 6G networks, to ensure trustworthy large-scale AIGC services in future 6G networks. We first discuss the adversarial attacks and privacy threats faced by AIGC systems in 6G networks, as well as the corresponding protection issues. Subsequently, we emphasize the importance of ensuring the unbiasedness and fairness of the mobile generative service in future intelligent networks. In particular, we conduct a use case to demonstrate that TrustGAIN can effectively guide the resistance against malicious or generated false information. We believe that TrustGAIN is a necessary paradigm for intelligent and trustworthy 6G networks to support AIGC services, ensuring the security, privacy, and fairness of AIGC network services.

摘要: 以大语言模型(LLM)为代表的AI-Generated Content(AIGC)模型给内容生成领域带来了革命性的变化。高速和广泛的6G技术是提供强大的AIGC移动业务应用的理想平台，而未来的6G移动网络也需要支持智能化和个性化的移动生成服务。然而，当前AIGC模型存在的重大伦理和安全问题，如对抗性攻击、隐私和公平性，极大地影响了6G智能网络的可信度，特别是在确保安全、私有和公平的AIGC应用方面。本文提出了一种新的6G网络可信AIGC模型TrustGAIN，以保证未来6G网络中可信赖的大规模AIGC服务。我们首先讨论了AIGC系统在6G网络中面临的敌意攻击和隐私威胁，以及相应的保护问题。随后，我们强调了在未来的智能网中确保移动生成业务的无偏性和公平性的重要性。特别是，我们进行了一个用例来证明TrustGAIN可以有效地指导对恶意或生成的虚假信息的抵抗。我们认为，TrustGAIN是智能可信6G网络支持AIGC服务的必备范式，确保AIGC网络服务的安全性、私密性和公平性。



## **25. LLMPot: Automated LLM-based Industrial Protocol and Physical Process Emulation for ICS Honeypots**

LLMPot：用于ICS蜜罐的自动化基于LLM的工业协议和物理流程仿真 cs.CR

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.05999v1) [paper-pdf](http://arxiv.org/pdf/2405.05999v1)

**Authors**: Christoforos Vasilatos, Dunia J. Mahboobeh, Hithem Lamri, Manaar Alam, Michail Maniatakos

**Abstract**: Industrial Control Systems (ICS) are extensively used in critical infrastructures ensuring efficient, reliable, and continuous operations. However, their increasing connectivity and addition of advanced features make them vulnerable to cyber threats, potentially leading to severe disruptions in essential services. In this context, honeypots play a vital role by acting as decoy targets within ICS networks, or on the Internet, helping to detect, log, analyze, and develop mitigations for ICS-specific cyber threats. Deploying ICS honeypots, however, is challenging due to the necessity of accurately replicating industrial protocols and device characteristics, a crucial requirement for effectively mimicking the unique operational behavior of different industrial systems. Moreover, this challenge is compounded by the significant manual effort required in also mimicking the control logic the PLC would execute, in order to capture attacker traffic aiming to disrupt critical infrastructure operations. In this paper, we propose LLMPot, a novel approach for designing honeypots in ICS networks harnessing the potency of Large Language Models (LLMs). LLMPot aims to automate and optimize the creation of realistic honeypots with vendor-agnostic configurations, and for any control logic, aiming to eliminate the manual effort and specialized knowledge traditionally required in this domain. We conducted extensive experiments focusing on a wide array of parameters, demonstrating that our LLM-based approach can effectively create honeypot devices implementing different industrial protocols and diverse control logic.

摘要: 工业控制系统(ICS)广泛应用于关键基础设施中，以确保高效、可靠和连续的运行。然而，它们日益增长的连通性和先进功能的增加使它们容易受到网络威胁，可能导致基本服务的严重中断。在这种情况下，蜜罐扮演着至关重要的角色，在ICS网络中或在互联网上充当诱骗目标，帮助检测、记录、分析和缓解ICS特定的网络威胁。然而，部署ICS蜜罐是具有挑战性的，因为必须准确复制工业协议和设备特性，这是有效模拟不同工业系统独特操作行为的关键要求。此外，为了捕获旨在扰乱关键基础设施操作的攻击者流量，还需要大量手动工作来模拟PLC将执行的控制逻辑，从而使这一挑战变得更加复杂。在本文中，我们提出了LLMPot，这是一种利用大语言模型的有效性来设计ICS网络中的蜜罐的新方法。LLMPot旨在自动化和优化创建具有供应商无关配置的逼真蜜罐，以及任何控制逻辑，旨在消除该领域传统上所需的手动工作和专业知识。我们针对广泛的参数进行了广泛的实验，证明了我们基于LLM的方法可以有效地创建执行不同工业协议和不同控制逻辑的蜜罐设备。



## **26. Chain of Attack: a Semantic-Driven Contextual Multi-Turn attacker for LLM**

攻击链：LLM的语义驱动上下文多回合攻击者 cs.CL

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.05610v1) [paper-pdf](http://arxiv.org/pdf/2405.05610v1)

**Authors**: Xikang Yang, Xuehai Tang, Songlin Hu, Jizhong Han

**Abstract**: Large language models (LLMs) have achieved remarkable performance in various natural language processing tasks, especially in dialogue systems. However, LLM may also pose security and moral threats, especially in multi round conversations where large models are more easily guided by contextual content, resulting in harmful or biased responses. In this paper, we present a novel method to attack LLMs in multi-turn dialogues, called CoA (Chain of Attack). CoA is a semantic-driven contextual multi-turn attack method that adaptively adjusts the attack policy through contextual feedback and semantic relevance during multi-turn of dialogue with a large model, resulting in the model producing unreasonable or harmful content. We evaluate CoA on different LLMs and datasets, and show that it can effectively expose the vulnerabilities of LLMs, and outperform existing attack methods. Our work provides a new perspective and tool for attacking and defending LLMs, and contributes to the security and ethical assessment of dialogue systems.

摘要: 大语言模型在各种自然语言处理任务中取得了显著的性能，尤其是在对话系统中。然而，LLM也可能构成安全和道德威胁，特别是在多轮对话中，大型模型更容易受到上下文内容的指导，导致有害或有偏见的反应。本文提出了一种新的攻击多话轮对话中的LLMS的方法，称为攻击链。CoA是一种语义驱动的上下文多轮攻击方法，在大模型的多轮对话中，通过上下文反馈和语义相关性自适应调整攻击策略，导致模型产生不合理或有害的内容。我们在不同的LLM和数据集上对CoA进行了评估，结果表明它可以有效地暴露LLMS的漏洞，并优于现有的攻击方法。我们的工作为攻击和防御LLMS提供了一个新的视角和工具，并有助于对话系统的安全和伦理评估。



## **27. Large Language Models for Cyber Security: A Systematic Literature Review**

网络安全的大型语言模型：系统性文献综述 cs.CR

46 pages,6 figures

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.04760v2) [paper-pdf](http://arxiv.org/pdf/2405.04760v2)

**Authors**: HanXiang Xu, ShenAo Wang, NingKe Li, KaiLong Wang, YanJie Zhao, Kai Chen, Ting Yu, Yang Liu, HaoYu Wang

**Abstract**: The rapid advancement of Large Language Models (LLMs) has opened up new opportunities for leveraging artificial intelligence in various domains, including cybersecurity. As the volume and sophistication of cyber threats continue to grow, there is an increasing need for intelligent systems that can automatically detect vulnerabilities, analyze malware, and respond to attacks. In this survey, we conduct a comprehensive review of the literature on the application of LLMs in cybersecurity (LLM4Security). By comprehensively collecting over 30K relevant papers and systematically analyzing 127 papers from top security and software engineering venues, we aim to provide a holistic view of how LLMs are being used to solve diverse problems across the cybersecurity domain. Through our analysis, we identify several key findings. First, we observe that LLMs are being applied to a wide range of cybersecurity tasks, including vulnerability detection, malware analysis, network intrusion detection, and phishing detection. Second, we find that the datasets used for training and evaluating LLMs in these tasks are often limited in size and diversity, highlighting the need for more comprehensive and representative datasets. Third, we identify several promising techniques for adapting LLMs to specific cybersecurity domains, such as fine-tuning, transfer learning, and domain-specific pre-training. Finally, we discuss the main challenges and opportunities for future research in LLM4Security, including the need for more interpretable and explainable models, the importance of addressing data privacy and security concerns, and the potential for leveraging LLMs for proactive defense and threat hunting. Overall, our survey provides a comprehensive overview of the current state-of-the-art in LLM4Security and identifies several promising directions for future research.

摘要: 大型语言模型(LLM)的快速发展为在包括网络安全在内的各个领域利用人工智能开辟了新的机会。随着网络威胁的数量和复杂性不断增长，对能够自动检测漏洞、分析恶意软件和响应攻击的智能系统的需求越来越大。在本次调查中，我们对LLMS在网络安全中的应用(LLM4Security)的文献进行了全面的回顾。通过全面收集30K多篇相关论文并系统分析来自顶级安全和软件工程场所的127篇论文，我们的目标是提供一个全面的视角，了解LLM是如何被用来解决网络安全领域的各种问题的。通过我们的分析，我们确定了几个关键发现。首先，我们观察到LLMS正被应用于广泛的网络安全任务，包括漏洞检测、恶意软件分析、网络入侵检测和网络钓鱼检测。其次，我们发现，在这些任务中用于训练和评估土地管理的数据集在大小和多样性方面往往有限，这突显了需要更全面和更具代表性的数据集。第三，我们确定了几种有前景的技术来使LLMS适应特定的网络安全领域，例如微调、迁移学习和特定领域的预训练。最后，我们讨论了LLM4Security未来研究的主要挑战和机遇，包括需要更多可解释和可解释的模型，解决数据隐私和安全问题的重要性，以及利用LLM进行主动防御和威胁追踪的潜力。总体而言，我们的调查提供了对LLM4Security当前最先进技术的全面概述，并确定了未来研究的几个有希望的方向。



## **28. Special Characters Attack: Toward Scalable Training Data Extraction From Large Language Models**

特殊字符攻击：从大型语言模型中提取可扩展的训练数据 cs.CR

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.05990v1) [paper-pdf](http://arxiv.org/pdf/2405.05990v1)

**Authors**: Yang Bai, Ge Pei, Jindong Gu, Yong Yang, Xingjun Ma

**Abstract**: Large language models (LLMs) have achieved remarkable performance on a wide range of tasks. However, recent studies have shown that LLMs can memorize training data and simple repeated tokens can trick the model to leak the data. In this paper, we take a step further and show that certain special characters or their combinations with English letters are stronger memory triggers, leading to more severe data leakage. The intuition is that, since LLMs are trained with massive data that contains a substantial amount of special characters (e.g. structural symbols {, } of JSON files, and @, # in emails and online posts), the model may memorize the co-occurrence between these special characters and the raw texts. This motivates us to propose a simple but effective Special Characters Attack (SCA) to induce training data leakage. Our experiments verify the high effectiveness of SCA against state-of-the-art LLMs: they can leak diverse training data, such as code corpus, web pages, and personally identifiable information, and sometimes generate non-stop outputs as a byproduct. We further show that the composition of the training data corpus can be revealed by inspecting the leaked data -- one crucial piece of information for pre-training high-performance LLMs. Our work can help understand the sensitivity of LLMs to special characters and identify potential areas for improvement.

摘要: 大型语言模型(LLM)在广泛的任务中取得了显著的性能。然而，最近的研究表明，LLMS可以记忆训练数据，简单的重复令牌可以诱使模型泄漏数据。在这篇文章中，我们进一步表明，某些特殊字符或它们与英语字母的组合是更强的记忆触发，导致更严重的数据泄漏。直观的是，由于LLM是用包含大量特殊字符(例如JSON文件的结构符号{，}，以及电子邮件和在线帖子中的@，#)的海量数据训练的，因此该模型可能会记住这些特殊字符与原始文本之间的共现。这促使我们提出了一种简单而有效的特殊字符攻击(SCA)来诱导训练数据泄漏。我们的实验验证了SCA相对于最先进的LLMS的高效性：它们可以泄露各种训练数据，如代码语料库、网页和个人身份信息，有时还会生成不间断的输出作为副产品。我们进一步表明，通过检查泄漏的数据可以揭示训练数据集的组成--这是训练前高性能LLMS的关键信息之一。我们的工作有助于理解LLMS对特殊字符的敏感度，并确定潜在的改进领域。



## **29. Locally Differentially Private In-Context Learning**

本地差异化私人背景学习 cs.CR

This paper was published at LREC-Coling 2024

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.04032v2) [paper-pdf](http://arxiv.org/pdf/2405.04032v2)

**Authors**: Chunyan Zheng, Keke Sun, Wenhao Zhao, Haibo Zhou, Lixin Jiang, Shaoyang Song, Chunlai Zhou

**Abstract**: Large pretrained language models (LLMs) have shown surprising In-Context Learning (ICL) ability. An important application in deploying large language models is to augment LLMs with a private database for some specific task. The main problem with this promising commercial use is that LLMs have been shown to memorize their training data and their prompt data are vulnerable to membership inference attacks (MIA) and prompt leaking attacks. In order to deal with this problem, we treat LLMs as untrusted in privacy and propose a locally differentially private framework of in-context learning(LDP-ICL) in the settings where labels are sensitive. Considering the mechanisms of in-context learning in Transformers by gradient descent, we provide an analysis of the trade-off between privacy and utility in such LDP-ICL for classification. Moreover, we apply LDP-ICL to the discrete distribution estimation problem. In the end, we perform several experiments to demonstrate our analysis results.

摘要: 大型预训练语言模型（LLM）表现出令人惊讶的上下文学习（ICL）能力。部署大型语言模型的一个重要应用是通过用于某些特定任务的专用数据库来增强LLM。这种有前途的商业用途的主要问题是，LLM已被证明可以记住它们的训练数据，而它们的提示数据很容易受到成员推断攻击（MIA）和提示泄露攻击。为了解决这个问题，我们将LLM视为隐私不受信任，并在标签敏感的环境中提出了一种本地差异隐私的上下文学习框架（LDP-ICL）。考虑到变形金刚中通过梯度下降进行的上下文学习机制，我们分析了此类LDP-ICL分类中隐私和实用性之间的权衡。此外，我们将LDP-ICL应用于离散分布估计问题。最后，我们进行了几个实验来证明我们的分析结果。



## **30. Air Gap: Protecting Privacy-Conscious Conversational Agents**

空气间隙：保护有隐私意识的对话代理人 cs.CR

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.05175v1) [paper-pdf](http://arxiv.org/pdf/2405.05175v1)

**Authors**: Eugene Bagdasaryan, Ren Yi, Sahra Ghalebikesabi, Peter Kairouz, Marco Gruteser, Sewoong Oh, Borja Balle, Daniel Ramage

**Abstract**: The growing use of large language model (LLM)-based conversational agents to manage sensitive user data raises significant privacy concerns. While these agents excel at understanding and acting on context, this capability can be exploited by malicious actors. We introduce a novel threat model where adversarial third-party apps manipulate the context of interaction to trick LLM-based agents into revealing private information not relevant to the task at hand.   Grounded in the framework of contextual integrity, we introduce AirGapAgent, a privacy-conscious agent designed to prevent unintended data leakage by restricting the agent's access to only the data necessary for a specific task. Extensive experiments using Gemini, GPT, and Mistral models as agents validate our approach's effectiveness in mitigating this form of context hijacking while maintaining core agent functionality. For example, we show that a single-query context hijacking attack on a Gemini Ultra agent reduces its ability to protect user data from 94% to 45%, while an AirGapAgent achieves 97% protection, rendering the same attack ineffective.

摘要: 越来越多地使用基于大型语言模型(LLM)的会话代理来管理敏感用户数据，这引发了严重的隐私问题。虽然这些代理擅长理解上下文并根据上下文执行操作，但这种能力可能会被恶意行为者利用。我们引入了一种新的威胁模型，在该模型中，敌意的第三方应用程序操纵交互的上下文，以欺骗基于LLM的代理泄露与手头任务无关的私人信息。基于上下文完整性的框架，我们引入了AirGapAgent，这是一个具有隐私意识的代理，旨在通过限制代理仅访问特定任务所需的数据来防止意外的数据泄露。使用Gemini、GPT和Mistral模型作为代理的大量实验验证了我们的方法在保持核心代理功能的同时缓解这种形式的上下文劫持的有效性。例如，我们表明，对Gemini Ultra代理的单查询上下文劫持攻击将其保护用户数据的能力从94%降低到45%，而AirGapAgent实现了97%的保护，使得相同的攻击无效。



## **31. Critical Infrastructure Protection: Generative AI, Challenges, and Opportunities**

关键基础设施保护：生成性人工智能、挑战和机遇 cs.CR

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.04874v1) [paper-pdf](http://arxiv.org/pdf/2405.04874v1)

**Authors**: Yagmur Yigit, Mohamed Amine Ferrag, Iqbal H. Sarker, Leandros A. Maglaras, Christos Chrysoulas, Naghmeh Moradpoor, Helge Janicke

**Abstract**: Critical National Infrastructure (CNI) encompasses a nation's essential assets that are fundamental to the operation of society and the economy, ensuring the provision of vital utilities such as energy, water, transportation, and communication. Nevertheless, growing cybersecurity threats targeting these infrastructures can potentially interfere with operations and seriously risk national security and public safety. In this paper, we examine the intricate issues raised by cybersecurity risks to vital infrastructure, highlighting these systems' vulnerability to different types of cyberattacks. We analyse the significance of trust, privacy, and resilience for Critical Infrastructure Protection (CIP), examining the diverse standards and regulations to manage these domains. We also scrutinise the co-analysis of safety and security, offering innovative approaches for their integration and emphasising the interdependence between these fields. Furthermore, we introduce a comprehensive method for CIP leveraging Generative AI and Large Language Models (LLMs), giving a tailored lifecycle and discussing specific applications across different critical infrastructure sectors. Lastly, we discuss potential future directions that promise to enhance the security and resilience of critical infrastructures. This paper proposes innovative strategies for CIP from evolving attacks and enhances comprehension of cybersecurity concerns related to critical infrastructure.

摘要: 关键国家基础设施(CNI)包括国家的基本资产，这些资产对社会和经济的运行至关重要，确保提供重要的公用事业，如能源、水、交通和通信。然而，针对这些基础设施的日益增长的网络安全威胁可能会干扰行动，并严重威胁国家安全和公共安全。在这篇文章中，我们研究了网络安全风险对重要基础设施提出的复杂问题，强调了这些系统对不同类型的网络攻击的脆弱性。我们分析了信任、隐私和弹性对于关键基础设施保护(CIP)的重要性，并研究了管理这些领域的不同标准和法规。我们还仔细研究了安全和安保的联合分析，为它们的整合提供了创新的方法，并强调了这些领域之间的相互依存关系。此外，我们介绍了一种全面的CIP方法，利用生成性人工智能和大型语言模型(LLM)，提供定制的生命周期，并讨论不同关键基础设施部门的特定应用。最后，我们讨论了承诺增强关键基础设施的安全性和弹性的潜在未来方向。本文针对CIP提出了应对不断演变的攻击的创新策略，并增强了对与关键基础设施相关的网络安全问题的理解。



## **32. BiasKG: Adversarial Knowledge Graphs to Induce Bias in Large Language Models**

BiasKG：对抗性知识图在大型语言模型中诱导偏见 cs.CL

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.04756v1) [paper-pdf](http://arxiv.org/pdf/2405.04756v1)

**Authors**: Chu Fei Luo, Ahmad Ghawanmeh, Xiaodan Zhu, Faiza Khan Khattak

**Abstract**: Modern large language models (LLMs) have a significant amount of world knowledge, which enables strong performance in commonsense reasoning and knowledge-intensive tasks when harnessed properly. The language model can also learn social biases, which has a significant potential for societal harm. There have been many mitigation strategies proposed for LLM safety, but it is unclear how effective they are for eliminating social biases. In this work, we propose a new methodology for attacking language models with knowledge graph augmented generation. We refactor natural language stereotypes into a knowledge graph, and use adversarial attacking strategies to induce biased responses from several open- and closed-source language models. We find our method increases bias in all models, even those trained with safety guardrails. This demonstrates the need for further research in AI safety, and further work in this new adversarial space.

摘要: 现代大型语言模型（LLM）拥有大量的世界知识，如果利用得当，可以在常识推理和知识密集型任务中取得出色的性能。语言模型还可以学习社会偏见，这具有巨大的社会危害潜力。人们为LLM安全提出了许多缓解策略，但目前尚不清楚它们对于消除社会偏见的有效性如何。在这项工作中，我们提出了一种利用知识图增强生成来攻击语言模型的新方法。我们将自然语言刻板印象重新构建到知识图谱中，并使用对抗性攻击策略来诱导几个开放和封闭源语言模型的偏见反应。我们发现我们的方法增加了所有模型的偏差，甚至是那些接受过安全护栏训练的模型。这表明需要对人工智能安全进行进一步研究，并在这个新的对抗空间中进一步开展工作。



## **33. AttacKG+:Boosting Attack Knowledge Graph Construction with Large Language Models**

AttacKG+：利用大型语言模型增强攻击知识图构建 cs.CR

20 pages, 5 figures

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.04753v1) [paper-pdf](http://arxiv.org/pdf/2405.04753v1)

**Authors**: Yongheng Zhang, Tingwen Du, Yunshan Ma, Xiang Wang, Yi Xie, Guozheng Yang, Yuliang Lu, Ee-Chien Chang

**Abstract**: Attack knowledge graph construction seeks to convert textual cyber threat intelligence (CTI) reports into structured representations, portraying the evolutionary traces of cyber attacks. Even though previous research has proposed various methods to construct attack knowledge graphs, they generally suffer from limited generalization capability to diverse knowledge types as well as requirement of expertise in model design and tuning. Addressing these limitations, we seek to utilize Large Language Models (LLMs), which have achieved enormous success in a broad range of tasks given exceptional capabilities in both language understanding and zero-shot task fulfillment. Thus, we propose a fully automatic LLM-based framework to construct attack knowledge graphs named: AttacKG+. Our framework consists of four consecutive modules: rewriter, parser, identifier, and summarizer, each of which is implemented by instruction prompting and in-context learning empowered by LLMs. Furthermore, we upgrade the existing attack knowledge schema and propose a comprehensive version. We represent a cyber attack as a temporally unfolding event, each temporal step of which encapsulates three layers of representation, including behavior graph, MITRE TTP labels, and state summary. Extensive evaluation demonstrates that: 1) our formulation seamlessly satisfies the information needs in threat event analysis, 2) our construction framework is effective in faithfully and accurately extracting the information defined by AttacKG+, and 3) our attack graph directly benefits downstream security practices such as attack reconstruction. All the code and datasets will be released upon acceptance.

摘要: 攻击知识图构建旨在将文本网络威胁情报(CTI)报告转换为结构化表示，刻画网络攻击的演化痕迹。尽管已有的研究提出了多种构造攻击知识图的方法，但它们普遍存在对不同知识类型的泛化能力有限以及对模型设计和调整的专业要求较低的问题。针对这些限制，我们寻求利用大型语言模型(LLM)，这些模型在广泛的任务中取得了巨大的成功，因为它们在语言理解和零距离任务完成方面都具有出色的能力。为此，我们提出了一种基于LLM的全自动攻击知识图构建框架：AttacKG+。该框架由重写器、解析器、识别器和汇总器四个连续的模块组成，每个模块都通过指令提示和上下文学习来实现。此外，我们对现有的攻击知识模式进行了升级，并提出了一个全面的版本。我们将网络攻击描述为一个在时间上展开的事件，每个时间步骤封装了三层表示，包括行为图、MITRE TTP标签和状态摘要。广泛的评估表明：1)我们的描述无缝地满足了威胁事件分析中的信息需求；2)我们的构造框架有效地、准确地提取了AttacKG+定义的信息；3)我们的攻击图直接帮助了下游的攻击重构等安全实践。所有代码和数据集将在验收后发布。



## **34. Revisiting character-level adversarial attacks**

重新审视角色级对抗攻击 cs.LG

Accepted in ICML 2024

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2405.04346v1) [paper-pdf](http://arxiv.org/pdf/2405.04346v1)

**Authors**: Elias Abad Rocamora, Yongtao Wu, Fanghui Liu, Grigorios G. Chrysos, Volkan Cevher

**Abstract**: Adversarial attacks in Natural Language Processing apply perturbations in the character or token levels. Token-level attacks, gaining prominence for their use of gradient-based methods, are susceptible to altering sentence semantics, leading to invalid adversarial examples. While character-level attacks easily maintain semantics, they have received less attention as they cannot easily adopt popular gradient-based methods, and are thought to be easy to defend. Challenging these beliefs, we introduce Charmer, an efficient query-based adversarial attack capable of achieving high attack success rate (ASR) while generating highly similar adversarial examples. Our method successfully targets both small (BERT) and large (Llama 2) models. Specifically, on BERT with SST-2, Charmer improves the ASR in 4.84% points and the USE similarity in 8% points with respect to the previous art. Our implementation is available in https://github.com/LIONS-EPFL/Charmer.

摘要: 自然语言处理中的对抗攻击对字符或令牌级别施加扰动。令牌级攻击因使用基于梯度的方法而变得越来越重要，很容易改变句子语义，从而导致无效的对抗性示例。虽然字符级攻击很容易维护语义，但它们受到的关注较少，因为它们不能轻易采用流行的基于梯度的方法，并且被认为很容易防御。基于这些信念，我们引入了Charmer，这是一种高效的基于查询的对抗性攻击，能够实现高攻击成功率（ASB），同时生成高度相似的对抗性示例。我们的方法成功地针对小型（BERT）和大型（Llama 2）模型。具体来说，在采用CST-2的BERT上，Charmer将ASB提高了4.84%，与之前的作品相比，USE相似性提高了8%。我们的实现可在https://github.com/LIONS-EPFL/Charmer上获取。



## **35. Who Wrote This? The Key to Zero-Shot LLM-Generated Text Detection Is GECScore**

这是谁写的？零镜头LLM生成文本检测的关键是GECScore cs.CL

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2405.04286v1) [paper-pdf](http://arxiv.org/pdf/2405.04286v1)

**Authors**: Junchao Wu, Runzhe Zhan, Derek F. Wong, Shu Yang, Xuebo Liu, Lidia S. Chao, Min Zhang

**Abstract**: The efficacy of an large language model (LLM) generated text detector depends substantially on the availability of sizable training data. White-box zero-shot detectors, which require no such data, are nonetheless limited by the accessibility of the source model of the LLM-generated text. In this paper, we propose an simple but effective black-box zero-shot detection approach, predicated on the observation that human-written texts typically contain more grammatical errors than LLM-generated texts. This approach entails computing the Grammar Error Correction Score (GECScore) for the given text to distinguish between human-written and LLM-generated text. Extensive experimental results show that our method outperforms current state-of-the-art (SOTA) zero-shot and supervised methods, achieving an average AUROC of 98.7% and showing strong robustness against paraphrase and adversarial perturbation attacks.

摘要: 大型语言模型（LLM）生成的文本检测器的功效在很大程度上取决于大量训练数据的可用性。白盒零镜头检测器不需要此类数据，但仍受到LLM生成文本源模型可访问性的限制。在本文中，我们提出了一种简单但有效的黑匣子零镜头检测方法，其基础是人类书面文本通常比LLM生成的文本包含更多的语法错误。这种方法需要计算给定文本的语法错误纠正分数（GECScore），以区分人类编写的文本和LLM生成的文本。大量的实验结果表明，我们的方法优于当前最先进的（SOTA）零射击和监督方法，实现了98.7%的平均AUROC，并对重述和对抗性扰动攻击表现出强大的鲁棒性。



## **36. Are aligned neural networks adversarially aligned?**

对齐的神经网络是否反向对齐？ cs.CL

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2306.15447v2) [paper-pdf](http://arxiv.org/pdf/2306.15447v2)

**Authors**: Nicholas Carlini, Milad Nasr, Christopher A. Choquette-Choo, Matthew Jagielski, Irena Gao, Anas Awadalla, Pang Wei Koh, Daphne Ippolito, Katherine Lee, Florian Tramer, Ludwig Schmidt

**Abstract**: Large language models are now tuned to align with the goals of their creators, namely to be "helpful and harmless." These models should respond helpfully to user questions, but refuse to answer requests that could cause harm. However, adversarial users can construct inputs which circumvent attempts at alignment. In this work, we study adversarial alignment, and ask to what extent these models remain aligned when interacting with an adversarial user who constructs worst-case inputs (adversarial examples). These inputs are designed to cause the model to emit harmful content that would otherwise be prohibited. We show that existing NLP-based optimization attacks are insufficiently powerful to reliably attack aligned text models: even when current NLP-based attacks fail, we can find adversarial inputs with brute force. As a result, the failure of current attacks should not be seen as proof that aligned text models remain aligned under adversarial inputs.   However the recent trend in large-scale ML models is multimodal models that allow users to provide images that influence the text that is generated. We show these models can be easily attacked, i.e., induced to perform arbitrary un-aligned behavior through adversarial perturbation of the input image. We conjecture that improved NLP attacks may demonstrate this same level of adversarial control over text-only models.

摘要: 大型语言模型现在被调整为与它们的创建者的目标保持一致，即“有益和无害”。这些模型应该对用户的问题做出有益的回应，但拒绝回答可能造成伤害的请求。然而，敌意用户可以构建绕过对齐尝试的输入。在这项工作中，我们研究对抗性对齐，并询问当与构建最坏情况输入(对抗性例子)的对抗性用户交互时，这些模型在多大程度上保持对齐。这些输入旨在导致模型排放本来被禁止的有害内容。我们证明了现有的基于NLP的优化攻击不足以可靠地攻击对齐的文本模型：即使当前基于NLP的攻击失败，我们也可以发现具有暴力的敌意输入。因此，当前攻击的失败不应被视为对齐的文本模型在敌意输入下保持对齐的证据。然而，大规模ML模型的最新趋势是允许用户提供影响所生成文本的图像的多模式模型。我们证明了这些模型可以很容易地被攻击，即通过对输入图像的对抗性扰动来诱导执行任意的非对齐行为。我们推测，改进的NLP攻击可能会展示出对纯文本模型的同样水平的敌意控制。



## **37. To Each (Textual Sequence) Its Own: Improving Memorized-Data Unlearning in Large Language Models**

每个（文本序列）自有：改进大型语言模型中的简化数据去学习 cs.LG

Published as a conference paper at ICML 2024

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2405.03097v1) [paper-pdf](http://arxiv.org/pdf/2405.03097v1)

**Authors**: George-Octavian Barbulescu, Peter Triantafillou

**Abstract**: LLMs have been found to memorize training textual sequences and regurgitate verbatim said sequences during text generation time. This fact is known to be the cause of privacy and related (e.g., copyright) problems. Unlearning in LLMs then takes the form of devising new algorithms that will properly deal with these side-effects of memorized data, while not hurting the model's utility. We offer a fresh perspective towards this goal, namely, that each textual sequence to be forgotten should be treated differently when being unlearned based on its degree of memorization within the LLM. We contribute a new metric for measuring unlearning quality, an adversarial attack showing that SOTA algorithms lacking this perspective fail for privacy, and two new unlearning methods based on Gradient Ascent and Task Arithmetic, respectively. A comprehensive performance evaluation across an extensive suite of NLP tasks then mapped the solution space, identifying the best solutions under different scales in model capacities and forget set sizes and quantified the gains of the new approaches.

摘要: 已经发现LLM在文本生成时间内记忆训练文本序列并逐字地返回所述序列。众所周知，这一事实是隐私和相关(例如，版权)问题的原因。然后，在LLMS中，遗忘的形式是设计新的算法，这些算法将适当地处理记忆数据的这些副作用，同时不会损害模型的实用性。我们为这一目标提供了一个新的视角，即每个被遗忘的文本序列在被遗忘时应该根据它在LLM中的记忆程度而得到不同的对待。我们提出了一种新的遗忘质量度量，一种敌意攻击表明缺乏这种视角的SOTA算法在隐私方面是失败的，以及两种新的遗忘方法，分别基于梯度上升和任务算法。然后，对一系列NLP任务进行了全面的性能评估，绘制了解决方案空间图，确定了模型容量和忘记集合大小不同尺度下的最佳解决方案，并量化了新方法的收益。



## **38. Trojans in Large Language Models of Code: A Critical Review through a Trigger-Based Taxonomy**

大型语言代码模型中的特洛伊木马：基于触发器的分类学的批判性评论 cs.SE

arXiv admin note: substantial text overlap with arXiv:2305.03803

**SubmitDate**: 2024-05-05    [abs](http://arxiv.org/abs/2405.02828v1) [paper-pdf](http://arxiv.org/pdf/2405.02828v1)

**Authors**: Aftab Hussain, Md Rafiqul Islam Rabin, Toufique Ahmed, Bowen Xu, Premkumar Devanbu, Mohammad Amin Alipour

**Abstract**: Large language models (LLMs) have provided a lot of exciting new capabilities in software development. However, the opaque nature of these models makes them difficult to reason about and inspect. Their opacity gives rise to potential security risks, as adversaries can train and deploy compromised models to disrupt the software development process in the victims' organization.   This work presents an overview of the current state-of-the-art trojan attacks on large language models of code, with a focus on triggers -- the main design point of trojans -- with the aid of a novel unifying trigger taxonomy framework. We also aim to provide a uniform definition of the fundamental concepts in the area of trojans in Code LLMs. Finally, we draw implications of findings on how code models learn on trigger design.

摘要: 大型语言模型（LLM）在软件开发中提供了许多令人兴奋的新功能。然而，这些模型的不透明性质使得它们难以推理和检查。它们的不透明性会带来潜在的安全风险，因为对手可以训练和部署受影响的模型，以扰乱受害者组织的软件开发流程。   这项工作概述了当前针对大型语言代码模型的最新特洛伊木马攻击，重点关注触发器（特洛伊木马的主要设计点），并在新颖的统一触发器分类框架的帮助下。我们还旨在为LLM代码中特洛伊木马领域的基本概念提供统一的定义。最后，我们得出了有关代码模型如何学习触发器设计的研究结果的影响。



## **39. Confidential and Protected Disease Classifier using Fully Homomorphic Encryption**

使用完全同形加密的机密和受保护的疾病分类器 cs.CR

**SubmitDate**: 2024-05-05    [abs](http://arxiv.org/abs/2405.02790v1) [paper-pdf](http://arxiv.org/pdf/2405.02790v1)

**Authors**: Aditya Malik, Nalini Ratha, Bharat Yalavarthi, Tilak Sharma, Arjun Kaushik, Charanjit Jutla

**Abstract**: With the rapid surge in the prevalence of Large Language Models (LLMs), individuals are increasingly turning to conversational AI for initial insights across various domains, including health-related inquiries such as disease diagnosis. Many users seek potential causes on platforms like ChatGPT or Bard before consulting a medical professional for their ailment. These platforms offer valuable benefits by streamlining the diagnosis process, alleviating the significant workload of healthcare practitioners, and saving users both time and money by avoiding unnecessary doctor visits. However, Despite the convenience of such platforms, sharing personal medical data online poses risks, including the presence of malicious platforms or potential eavesdropping by attackers. To address privacy concerns, we propose a novel framework combining FHE and Deep Learning for a secure and private diagnosis system. Operating on a question-and-answer-based model akin to an interaction with a medical practitioner, this end-to-end secure system employs Fully Homomorphic Encryption (FHE) to handle encrypted input data. Given FHE's computational constraints, we adapt deep neural networks and activation functions to the encryted domain. Further, we also propose a faster algorithm to compute summation of ciphertext elements. Through rigorous experiments, we demonstrate the efficacy of our approach. The proposed framework achieves strict security and privacy with minimal loss in performance.

摘要: 随着大型语言模型(LLM)的普及，越来越多的人转向对话式人工智能来获得各个领域的初步见解，包括与健康相关的查询，如疾病诊断。许多用户在咨询医疗专业人士之前，会在ChatGPT或Bard等平台上寻找潜在的原因。这些平台简化了诊断流程，减轻了医疗从业者的巨大工作量，并通过避免不必要的医生就诊节省了用户的时间和金钱，从而提供了宝贵的好处。然而，尽管这类平台很方便，但在线共享个人医疗数据会带来风险，包括恶意平台的存在或潜在的攻击者窃听。为了解决隐私问题，我们提出了一种新的框架，将FHE和深度学习结合起来，以实现安全和隐私的诊断系统。这种端到端的安全系统采用完全同态加密(FHE)来处理加密的输入数据，其基于问答的模型类似于与医生的交互。在给定计算约束的情况下，我们将深度神经网络和激活函数应用到加密域。此外，我们还提出了一种计算密文元素求和的快速算法。通过严格的实验，我们证明了我们的方法的有效性。该框架以最小的性能损失实现了严格的安全性和保密性。



## **40. Assessing Adversarial Robustness of Large Language Models: An Empirical Study**

评估大型语言模型的对抗稳健性：实证研究 cs.CL

16 pages, 9 figures, 10 tables

**SubmitDate**: 2024-05-04    [abs](http://arxiv.org/abs/2405.02764v1) [paper-pdf](http://arxiv.org/pdf/2405.02764v1)

**Authors**: Zeyu Yang, Zhao Meng, Xiaochen Zheng, Roger Wattenhofer

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing, but their robustness against adversarial attacks remains a critical concern. We presents a novel white-box style attack approach that exposes vulnerabilities in leading open-source LLMs, including Llama, OPT, and T5. We assess the impact of model size, structure, and fine-tuning strategies on their resistance to adversarial perturbations. Our comprehensive evaluation across five diverse text classification tasks establishes a new benchmark for LLM robustness. The findings of this study have far-reaching implications for the reliable deployment of LLMs in real-world applications and contribute to the advancement of trustworthy AI systems.

摘要: 大型语言模型（LLM）彻底改变了自然语言处理，但其对抗攻击的稳健性仍然是一个关键问题。我们提出了一种新颖的白盒式攻击方法，该方法暴露了领先开源LLM（包括Llama、OPT和T5）中的漏洞。我们评估了模型大小、结构和微调策略对其抵抗对抗性扰动的影响。我们对五种不同文本分类任务的全面评估为LLM稳健性建立了新基准。这项研究的结果对于LLM在现实世界应用程序中的可靠部署具有深远的影响，并有助于发展值得信赖的人工智能系统。



## **41. Jailbreak and Guard Aligned Language Models with Only Few In-Context Demonstrations**

越狱和警卫一致的语言模型，只有很少的上下文演示 cs.LG

**SubmitDate**: 2024-05-04    [abs](http://arxiv.org/abs/2310.06387v2) [paper-pdf](http://arxiv.org/pdf/2310.06387v2)

**Authors**: Zeming Wei, Yifei Wang, Yisen Wang

**Abstract**: Large Language Models (LLMs) have shown remarkable success in various tasks, but concerns about their safety and the potential for generating harmful content have emerged. In this paper, we delve into the potential of In-Context Learning (ICL) to modulate the alignment of LLMs. Specifically, we propose the In-Context Attack (ICA), which employs strategically crafted harmful demonstrations to subvert LLMs, and the In-Context Defense (ICD), which bolsters model resilience through examples that demonstrate refusal to produce harmful responses. Through extensive experiments, we demonstrate the efficacy of ICA and ICD in respectively elevating and mitigating the success rates of jailbreaking prompts. Moreover, we offer theoretical insights into the mechanism by which a limited set of in-context demonstrations can pivotally influence the safety alignment of LLMs. Our findings illuminate the profound influence of ICL on LLM behavior, opening new avenues for improving the safety and alignment of LLMs.

摘要: 大型语言模型(LLM)在各种任务中取得了显著的成功，但也出现了对其安全性和产生有害内容的可能性的担忧。在这篇文章中，我们深入研究了情境学习(ICL)在调节LLM对齐方面的潜力。具体地说，我们提出了情境攻击(ICA)和情境防御(ICD)，前者采用战略性的有害演示来颠覆LLM，后者通过展示拒绝产生有害响应的例子来增强模型的弹性。通过大量的实验，我们证明了ICA和ICD分别在提高和降低越狱提示成功率方面的有效性。此外，我们对有限的上下文演示集可以对LLM的安全对准产生关键影响的机制提供了理论见解。我们的发现阐明了ICL对LLM行为的深刻影响，为提高LLM的安全性和对比性开辟了新的途径。



## **42. PropertyGPT: LLM-driven Formal Verification of Smart Contracts through Retrieval-Augmented Property Generation**

PropertyGPT：通过检索增强属性生成，LLM驱动的智能合同形式验证 cs.SE

**SubmitDate**: 2024-05-04    [abs](http://arxiv.org/abs/2405.02580v1) [paper-pdf](http://arxiv.org/pdf/2405.02580v1)

**Authors**: Ye Liu, Yue Xue, Daoyuan Wu, Yuqiang Sun, Yi Li, Miaolei Shi, Yang Liu

**Abstract**: With recent advances in large language models (LLMs), this paper explores the potential of leveraging state-of-the-art LLMs, such as GPT-4, to transfer existing human-written properties (e.g., those from Certora auditing reports) and automatically generate customized properties for unknown code. To this end, we embed existing properties into a vector database and retrieve a reference property for LLM-based in-context learning to generate a new prop- erty for a given code. While this basic process is relatively straight- forward, ensuring that the generated properties are (i) compilable, (ii) appropriate, and (iii) runtime-verifiable presents challenges. To address (i), we use the compilation and static analysis feedback as an external oracle to guide LLMs in iteratively revising the generated properties. For (ii), we consider multiple dimensions of similarity to rank the properties and employ a weighted algorithm to identify the top-K properties as the final result. For (iii), we design a dedicated prover to formally verify the correctness of the generated prop- erties. We have implemented these strategies into a novel system called PropertyGPT, with 623 human-written properties collected from 23 Certora projects. Our experiments show that PropertyGPT can generate comprehensive and high-quality properties, achieving an 80% recall compared to the ground truth. It successfully detected 26 CVEs/attack incidents out of 37 tested and also uncovered 12 zero-day vulnerabilities, resulting in $8,256 bug bounty rewards.

摘要: 随着大型语言模型(LLM)的最新进展，本文探索了利用最先进的LLM(如GPT-4)来转移现有的人工编写的属性(例如，来自Certora审计报告的属性)并自动为未知代码生成定制属性的潜力。为此，我们将现有属性嵌入到向量数据库中，并检索一个参考属性用于基于LLM的上下文中学习，以生成给定代码的新属性。虽然这个基本过程相对简单，但确保生成的属性是(I)可编译的，(Ii)适当的，以及(Iii)可运行时验证的，这是一个挑战。为了解决(I)，我们使用编译和静态分析反馈作为外部预言来指导LLM迭代地修改生成的属性。对于(Ii)，我们考虑多个维度的相似性来对属性进行排序，并使用加权算法来识别TOP-K属性作为最终结果。对于(Iii)，我们设计了一个专用的证明器来形式化地验证所生成的性质的正确性。我们已经将这些策略实施到一个名为PropertyGPT的新系统中，从23个Certora项目中收集了623个人写的属性。我们的实验表明，PropertyGPT可以生成全面的高质量属性，与基本事实相比，召回率达到80%。它在37个测试中成功检测到26个CVE/攻击事件，还发现了12个零日漏洞，从而获得了8,256美元的漏洞赏金。



## **43. Adaptive and robust watermark against model extraction attack**

抗模型提取攻击的自适应鲁棒水印 cs.CR

**SubmitDate**: 2024-05-03    [abs](http://arxiv.org/abs/2405.02365v1) [paper-pdf](http://arxiv.org/pdf/2405.02365v1)

**Authors**: Kaiyi Pang, Tao Qi, Chuhan Wu, Minhao Bai

**Abstract**: Large language models have boosted Large Models as a Service (LMaaS) into a thriving business sector. But even model owners offering only API access while keeping model parameters and internal workings private, their Intellectual Property (IP) are still at risk of theft through model extraction attacks. To safeguard the IP of these models and mitigate unfair competition in the language model market, watermarking technology serves as an efficient post-hoc solution for identifying IP infringements. However, existing IP protection watermarking methods either explicitly alter the original output of the language model or implant watermark signals in the model logits. These methods forcefully distort the original distribution of the language model and impact the sampling process, leading to a decline in the quality of the generated text. The existing method also fails to achieve end-to-end adaptive watermark embedding and lack robustness verification in complex scenarios where watermark detection is subject to interference. To overcome these challenges, we propose PromptShield, a plug-and-play IP protection watermarking method to resist model extraction attacks without training additional modules. Leveraging the self-reminding properties inherent in large language models, we encapsulate the user's query with a watermark self-generated instruction, nudging the LLMs to automatically generate watermark words in its output without compromising generation quality. Our method does not require access to the model's internal logits and minimizes alterations to the model's distribution using prompt-guided cues. Comprehensive experimental results consistently demonstrate the effectiveness, harmlessness, and robustness of our watermark. Moreover, Our watermark detection method remains robust and high detection sensitivity even when subjected to interference.

摘要: 大型语言模型将大型模型即服务(LMaaS)推向了蓬勃发展的商业领域。但是，即使模型所有者只提供API访问，同时保持模型参数和内部工作的隐私，他们的知识产权(IP)仍然面临通过模型提取攻击被窃取的风险。为了保护这些模型的知识产权并缓解语言模型市场上的不公平竞争，水印技术是识别知识产权侵权的一种有效的事后解决方案。然而，现有的IP保护水印方法要么显式地改变语言模型的原始输出，要么在模型逻辑中嵌入水印信号。这些方法强烈扭曲了语言模型的原始分布，影响了采样过程，导致生成文本的质量下降。现有的方法也无法实现端到端的自适应水印嵌入，在水印检测受到干扰的复杂场景下缺乏健壮性验证。为了克服这些挑战，我们提出了一种即插即用的IP保护水印方法PromptShield，它可以在不训练额外模块的情况下抵抗模型提取攻击。利用大型语言模型固有的自提醒特性，我们使用水印自生成指令来封装用户的查询，推动LLMS在其输出中自动生成水印词，而不会影响生成质量。我们的方法不需要访问模型的内部日志，并使用提示引导提示将对模型分布的更改降至最低。全面的实验结果一致地证明了该水印的有效性、无害性和稳健性。此外，我们的水印检测方法即使在受到干扰的情况下也保持了健壮性和高检测灵敏度。



## **44. Dr. Jekyll and Mr. Hyde: Two Faces of LLMs**

杰基尔博士和海德先生：法学硕士的两面 cs.CR

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2312.03853v3) [paper-pdf](http://arxiv.org/pdf/2312.03853v3)

**Authors**: Matteo Gioele Collu, Tom Janssen-Groesbeek, Stefanos Koffas, Mauro Conti, Stjepan Picek

**Abstract**: Recently, we have witnessed a rise in the use of Large Language Models (LLMs), especially in applications like chatbot assistants. Safety mechanisms and specialized training procedures are implemented to prevent improper responses from these assistants. In this work, we bypass these measures for ChatGPT and Bard (and, to some extent, Bing chat) by making them impersonate complex personas with personality characteristics that are not aligned with a truthful assistant. We start by creating elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversations then followed a role-play style to elicit prohibited responses. By making use of personas, we show that such responses are actually provided, making it possible to obtain unauthorized, illegal, or harmful information. This work shows that by using adversarial personas, one can overcome safety mechanisms set out by ChatGPT and Bard. We also introduce several ways of activating such adversarial personas, which show that both chatbots are vulnerable to this kind of attack. With the same principle, we introduce two defenses that push the model to interpret trustworthy personalities and make it more robust against such attacks.

摘要: 最近，我们看到大型语言模型(LLM)的使用有所增加，特别是在聊天机器人助手等应用程序中。实施了安全机制和专门的培训程序，以防止这些助理做出不当反应。在这项工作中，我们绕过了ChatGPT和Bard(在某种程度上，Bing聊天)的这些措施，让他们模仿具有人格特征的复杂人物角色，而这些人物角色与诚实的助手不一致。我们首先为这些角色创建精致的传记，然后在与相同的聊天机器人的新会话中使用。然后，我们的对话遵循了角色扮演的风格，引发了被禁止的回应。通过使用人物角色，我们表明实际上提供了这样的响应，使得获得未经授权的、非法的或有害的信息成为可能。这项工作表明，通过使用对抗性人物角色，一个人可以克服ChatGPT和Bard提出的安全机制。我们还介绍了几种激活这种敌对角色的方法，这表明这两个聊天机器人都容易受到这种攻击。在相同的原则下，我们引入了两个防御措施，推动该模型解释可信任的个性，并使其对此类攻击更加健壮。



## **45. Generative AI in Cybersecurity**

网络安全中的生成人工智能 cs.CR

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.01674v1) [paper-pdf](http://arxiv.org/pdf/2405.01674v1)

**Authors**: Shivani Metta, Isaac Chang, Jack Parker, Michael P. Roman, Arturo F. Ehuan

**Abstract**: The dawn of Generative Artificial Intelligence (GAI), characterized by advanced models such as Generative Pre-trained Transformers (GPT) and other Large Language Models (LLMs), has been pivotal in reshaping the field of data analysis, pattern recognition, and decision-making processes. This surge in GAI technology has ushered in not only innovative opportunities for data processing and automation but has also introduced significant cybersecurity challenges.   As GAI rapidly progresses, it outstrips the current pace of cybersecurity protocols and regulatory frameworks, leading to a paradox wherein the same innovations meant to safeguard digital infrastructures also enhance the arsenal available to cyber criminals. These adversaries, adept at swiftly integrating and exploiting emerging technologies, may utilize GAI to develop malware that is both more covert and adaptable, thus complicating traditional cybersecurity efforts.   The acceleration of GAI presents an ambiguous frontier for cybersecurity experts, offering potent tools for threat detection and response, while concurrently providing cyber attackers with the means to engineer more intricate and potent malware. Through the joint efforts of Duke Pratt School of Engineering, Coalfire, and Safebreach, this research undertakes a meticulous analysis of how malicious agents are exploiting GAI to augment their attack strategies, emphasizing a critical issue for the integrity of future cybersecurity initiatives. The study highlights the critical need for organizations to proactively identify and develop more complex defensive strategies to counter the sophisticated employment of GAI in malware creation.

摘要: 以生成性预训练转换器(GPT)和其他大型语言模型(LLM)等高级模型为特征的生成性人工智能(GAI)的出现，在重塑数据分析、模式识别和决策过程领域起到了关键作用。GAI技术的激增不仅为数据处理和自动化带来了创新机遇，也带来了重大的网络安全挑战。随着GAI的快速发展，它超过了当前网络安全协议和监管框架的速度，导致了一个悖论，即旨在保护数字基础设施的相同创新也增强了网络犯罪分子可用的武器库。这些擅长快速整合和利用新兴技术的对手可能会利用GAI开发更隐蔽和更具适应性的恶意软件，从而使传统的网络安全努力复杂化。GAI的加速为网络安全专家提供了一个模糊的边界，为威胁检测和响应提供了强大的工具，同时也为网络攻击者提供了设计更复杂、更强大的恶意软件的手段。通过杜克·普拉特工程学院、煤火和安全漏洞的共同努力，这项研究对恶意代理如何利用GAI来增强其攻击策略进行了细致的分析，强调了未来网络安全计划的完整性的关键问题。这项研究突出表明，组织迫切需要主动识别和开发更复杂的防御策略，以对抗GAI在恶意软件创建中的复杂使用。



## **46. LLM Self Defense: By Self Examination, LLMs Know They Are Being Tricked**

法学硕士自卫：通过自我检查，法学硕士知道他们被欺骗了 cs.CL

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2308.07308v4) [paper-pdf](http://arxiv.org/pdf/2308.07308v4)

**Authors**: Mansi Phute, Alec Helbling, Matthew Hull, ShengYun Peng, Sebastian Szyller, Cory Cornelius, Duen Horng Chau

**Abstract**: Large language models (LLMs) are popular for high-quality text generation but can produce harmful content, even when aligned with human values through reinforcement learning. Adversarial prompts can bypass their safety measures. We propose LLM Self Defense, a simple approach to defend against these attacks by having an LLM screen the induced responses. Our method does not require any fine-tuning, input preprocessing, or iterative output generation. Instead, we incorporate the generated content into a pre-defined prompt and employ another instance of an LLM to analyze the text and predict whether it is harmful. We test LLM Self Defense on GPT 3.5 and Llama 2, two of the current most prominent LLMs against various types of attacks, such as forcefully inducing affirmative responses to prompts and prompt engineering attacks. Notably, LLM Self Defense succeeds in reducing the attack success rate to virtually 0 using both GPT 3.5 and Llama 2. The code is publicly available at https://github.com/poloclub/llm-self-defense

摘要: 大型语言模型(LLM)对于高质量的文本生成很受欢迎，但可能会产生有害的内容，即使通过强化学习与人类的价值观保持一致。对抗性提示可以绕过它们的安全措施。我们提出了LLM自卫，这是一种通过让LLM筛选诱导响应来防御这些攻击的简单方法。我们的方法不需要任何微调、输入预处理或迭代输出生成。相反，我们将生成的内容合并到预定义的提示中，并使用LLM的另一个实例来分析文本并预测它是否有害。我们在GPT 3.5和Llama 2上测试了LLM自卫，这两种当前最著名的LLM针对各种类型的攻击，例如对提示的强制诱导肯定反应和提示工程攻击。值得注意的是，LLm自卫成功地使用GPT3.5和Llama 2将攻击成功率降低到几乎为0。代码可在https://github.com/poloclub/llm-self-defense上公开获得



## **47. Boosting Jailbreak Attack with Momentum**

以势头助推越狱攻击 cs.LG

ICLR 2024 Workshop on Reliable and Responsible Foundation Models

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.01229v1) [paper-pdf](http://arxiv.org/pdf/2405.01229v1)

**Authors**: Yihao Zhang, Zeming Wei

**Abstract**: Large Language Models (LLMs) have achieved remarkable success across diverse tasks, yet they remain vulnerable to adversarial attacks, notably the well-documented \textit{jailbreak} attack. Recently, the Greedy Coordinate Gradient (GCG) attack has demonstrated efficacy in exploiting this vulnerability by optimizing adversarial prompts through a combination of gradient heuristics and greedy search. However, the efficiency of this attack has become a bottleneck in the attacking process. To mitigate this limitation, in this paper we rethink the generation of adversarial prompts through an optimization lens, aiming to stabilize the optimization process and harness more heuristic insights from previous iterations. Specifically, we introduce the \textbf{M}omentum \textbf{A}ccelerated G\textbf{C}G (\textbf{MAC}) attack, which incorporates a momentum term into the gradient heuristic. Experimental results showcase the notable enhancement achieved by MAP in gradient-based attacks on aligned language models. Our code is available at https://github.com/weizeming/momentum-attack-llm.

摘要: 大型语言模型(LLM)已经在不同的任务中取得了显著的成功，但它们仍然容易受到对手的攻击，特别是有充分记录的\textit{jailBreak}攻击。最近，贪婪坐标梯度(GCG)攻击通过结合梯度启发式算法和贪婪搜索来优化敌意提示，从而有效地利用了这一漏洞。然而，这种攻击的效率已经成为攻击过程中的瓶颈。为了缓解这一局限性，在本文中，我们通过优化镜头重新考虑对抗性提示的生成，旨在稳定优化过程，并从以前的迭代中获得更多启发式的见解。具体地说，我们引入了将动量项结合到梯度启发式中的加速G(Textbf{C}G(Textbf{MAC}))攻击。实验结果表明，MAP在对对齐语言模型的基于梯度的攻击中取得了显著的改进。我们的代码可以在https://github.com/weizeming/momentum-attack-llm.上找到



## **48. Adversarial Attacks and Defense for Conversation Entailment Task**

对抗性攻击和对话需求任务的防御 cs.CL

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.00289v2) [paper-pdf](http://arxiv.org/pdf/2405.00289v2)

**Authors**: Zhenning Yang, Ryan Krawec, Liang-Yuan Wu

**Abstract**: As the deployment of NLP systems in critical applications grows, ensuring the robustness of large language models (LLMs) against adversarial attacks becomes increasingly important. Large language models excel in various NLP tasks but remain vulnerable to low-cost adversarial attacks. Focusing on the domain of conversation entailment, where multi-turn dialogues serve as premises to verify hypotheses, we fine-tune a transformer model to accurately discern the truthfulness of these hypotheses. Adversaries manipulate hypotheses through synonym swapping, aiming to deceive the model into making incorrect predictions. To counteract these attacks, we implemented innovative fine-tuning techniques and introduced an embedding perturbation loss method to significantly bolster the model's robustness. Our findings not only emphasize the importance of defending against adversarial attacks in NLP but also highlight the real-world implications, suggesting that enhancing model robustness is critical for reliable NLP applications.

摘要: 随着NLP系统在关键应用中的部署越来越多，确保大型语言模型(LLM)对对手攻击的健壮性变得越来越重要。大型语言模型在各种NLP任务中表现出色，但仍然容易受到低成本的对抗性攻击。聚焦于会话蕴涵领域，多轮对话是验证假设的前提，我们微调了一个转换器模型，以准确识别这些假设的真实性。对手通过同义词互换来操纵假设，目的是欺骗模型做出错误的预测。为了对抗这些攻击，我们实施了创新的微调技术，并引入了嵌入扰动损失方法来显著增强模型的稳健性。我们的发现不仅强调了在自然语言处理中防御对手攻击的重要性，而且也强调了现实世界的影响，表明增强模型的健壮性对于可靠的自然语言处理应用是至关重要的。



## **49. AmpleGCG: Learning a Universal and Transferable Generative Model of Adversarial Suffixes for Jailbreaking Both Open and Closed LLMs**

AmpleGCG：学习通用且可转移的对抗性后缀生成模型，用于越狱开放和封闭LLM cs.CL

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2404.07921v2) [paper-pdf](http://arxiv.org/pdf/2404.07921v2)

**Authors**: Zeyi Liao, Huan Sun

**Abstract**: As large language models (LLMs) become increasingly prevalent and integrated into autonomous systems, ensuring their safety is imperative. Despite significant strides toward safety alignment, recent work GCG~\citep{zou2023universal} proposes a discrete token optimization algorithm and selects the single suffix with the lowest loss to successfully jailbreak aligned LLMs. In this work, we first discuss the drawbacks of solely picking the suffix with the lowest loss during GCG optimization for jailbreaking and uncover the missed successful suffixes during the intermediate steps. Moreover, we utilize those successful suffixes as training data to learn a generative model, named AmpleGCG, which captures the distribution of adversarial suffixes given a harmful query and enables the rapid generation of hundreds of suffixes for any harmful queries in seconds. AmpleGCG achieves near 100\% attack success rate (ASR) on two aligned LLMs (Llama-2-7B-chat and Vicuna-7B), surpassing two strongest attack baselines. More interestingly, AmpleGCG also transfers seamlessly to attack different models, including closed-source LLMs, achieving a 99\% ASR on the latest GPT-3.5. To summarize, our work amplifies the impact of GCG by training a generative model of adversarial suffixes that is universal to any harmful queries and transferable from attacking open-source LLMs to closed-source LLMs. In addition, it can generate 200 adversarial suffixes for one harmful query in only 4 seconds, rendering it more challenging to defend.

摘要: 随着大型语言模型(LLM)变得越来越普遍并集成到自治系统中，确保它们的安全性是当务之急。尽管在安全对齐方面取得了长足的进步，但最近的工作GCG~\Citep{zou2023Universal}提出了一种离散令牌优化算法，并选择损失最小的单个后缀来成功越狱对齐LLM。在这项工作中，我们首先讨论了在GCG优化越狱过程中只选择损失最小的后缀的缺点，并在中间步骤中发现了遗漏的成功后缀。此外，我们利用这些成功的后缀作为训练数据来学习一种名为AmpleGCG的生成模型，该模型捕获给定有害查询的对抗性后缀的分布，并在几秒钟内为任何有害查询快速生成数百个后缀。AmpleGCG在两个对齐的LLM(Llama-2-7B-Chat和Vicuna-7B)上达到了近100%的攻击成功率(ASR)，超过了两个最强的攻击基线。更有趣的是，AmpleGCG还无缝传输以攻击不同的型号，包括闭源LLMS，在最新的GPT-3.5上实现了99\%的ASR。总之，我们的工作通过训练对抗性后缀的生成模型来放大GCG的影响，该模型对任何有害的查询都是通用的，并且可以从攻击开源LLM转移到闭源LLM。此外，它可以在短短4秒内为一个有害的查询生成200个敌意后缀，使其更具挑战性。



## **50. Assessing LLMs in Malicious Code Deobfuscation of Real-world Malware Campaigns**

评估现实世界恶意软件活动的恶意代码去混淆中的LLM cs.CR

**SubmitDate**: 2024-04-30    [abs](http://arxiv.org/abs/2404.19715v1) [paper-pdf](http://arxiv.org/pdf/2404.19715v1)

**Authors**: Constantinos Patsakis, Fran Casino, Nikolaos Lykousas

**Abstract**: The integration of large language models (LLMs) into various pipelines is increasingly widespread, effectively automating many manual tasks and often surpassing human capabilities. Cybersecurity researchers and practitioners have recognised this potential. Thus, they are actively exploring its applications, given the vast volume of heterogeneous data that requires processing to identify anomalies, potential bypasses, attacks, and fraudulent incidents. On top of this, LLMs' advanced capabilities in generating functional code, comprehending code context, and summarising its operations can also be leveraged for reverse engineering and malware deobfuscation. To this end, we delve into the deobfuscation capabilities of state-of-the-art LLMs. Beyond merely discussing a hypothetical scenario, we evaluate four LLMs with real-world malicious scripts used in the notorious Emotet malware campaign. Our results indicate that while not absolutely accurate yet, some LLMs can efficiently deobfuscate such payloads. Thus, fine-tuning LLMs for this task can be a viable potential for future AI-powered threat intelligence pipelines in the fight against obfuscated malware.

摘要: 将大型语言模型(LLM)集成到各种管道中的情况日益广泛，有效地自动化了许多手动任务，并且常常超出了人类的能力。网络安全研究人员和从业者已经认识到了这一潜力。因此，他们正在积极探索其应用，因为需要处理大量的异类数据来识别异常、潜在的绕过、攻击和欺诈性事件。最重要的是，LLMS在生成功能代码、理解代码上下文和总结其操作方面的高级能力也可以用于反向工程和恶意软件去混淆。为此，我们深入研究了最先进的LLM的去模糊能力。除了仅讨论假设场景之外，我们还使用臭名昭著的Emotet恶意软件活动中使用的真实世界恶意脚本来评估四个LLM。我们的结果表明，虽然还不是绝对准确，但一些LLMS可以有效地对此类有效载荷进行去模糊。因此，为这项任务微调LLM可能是未来人工智能支持的威胁情报管道在打击混淆恶意软件方面的一个可行的潜力。



