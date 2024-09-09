# Latest Large Language Model Attack Papers
**update at 2024-09-09 09:19:55**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Recent Advances in Attack and Defense Approaches of Large Language Models**

大型语言模型攻击和防御方法的最新进展 cs.CR

**SubmitDate**: 2024-09-06    [abs](http://arxiv.org/abs/2409.03274v2) [paper-pdf](http://arxiv.org/pdf/2409.03274v2)

**Authors**: Jing Cui, Yishi Xu, Zhewei Huang, Shuchang Zhou, Jianbin Jiao, Junge Zhang

**Abstract**: Large Language Models (LLMs) have revolutionized artificial intelligence and machine learning through their advanced text processing and generating capabilities. However, their widespread deployment has raised significant safety and reliability concerns. Established vulnerabilities in deep neural networks, coupled with emerging threat models, may compromise security evaluations and create a false sense of security. Given the extensive research in the field of LLM security, we believe that summarizing the current state of affairs will help the research community better understand the present landscape and inform future developments. This paper reviews current research on LLM vulnerabilities and threats, and evaluates the effectiveness of contemporary defense mechanisms. We analyze recent studies on attack vectors and model weaknesses, providing insights into attack mechanisms and the evolving threat landscape. We also examine current defense strategies, highlighting their strengths and limitations. By contrasting advancements in attack and defense methodologies, we identify research gaps and propose future directions to enhance LLM security. Our goal is to advance the understanding of LLM safety challenges and guide the development of more robust security measures.

摘要: 大型语言模型(LLM)通过其先进的文本处理和生成能力，使人工智能和机器学习发生了革命性的变化。然而，它们的广泛部署引发了严重的安全和可靠性问题。深层神经网络中已建立的漏洞，再加上新出现的威胁模型，可能会危及安全评估，并造成一种错误的安全感。鉴于LLM安全领域的广泛研究，我们相信总结当前的事态将有助于研究界更好地了解目前的情况并为未来的发展提供信息。本文回顾了LLM漏洞和威胁的研究现状，并对现代防御机制的有效性进行了评估。我们分析了最近关于攻击载体和模型弱点的研究，提供了对攻击机制和不断演变的威胁环境的洞察。我们还研究了当前的防御战略，强调了它们的优势和局限性。通过对比攻击和防御方法的进展，我们发现了研究的差距，并提出了增强LLM安全的未来方向。我们的目标是促进对LLM安全挑战的理解，并指导开发更强大的安全措施。



## **2. LLM-PBE: Assessing Data Privacy in Large Language Models**

LLM-PBE：评估大型语言模型中的数据隐私 cs.CR

**SubmitDate**: 2024-09-06    [abs](http://arxiv.org/abs/2408.12787v2) [paper-pdf](http://arxiv.org/pdf/2408.12787v2)

**Authors**: Qinbin Li, Junyuan Hong, Chulin Xie, Jeffrey Tan, Rachel Xin, Junyi Hou, Xavier Yin, Zhun Wang, Dan Hendrycks, Zhangyang Wang, Bo Li, Bingsheng He, Dawn Song

**Abstract**: Large Language Models (LLMs) have become integral to numerous domains, significantly advancing applications in data management, mining, and analysis. Their profound capabilities in processing and interpreting complex language data, however, bring to light pressing concerns regarding data privacy, especially the risk of unintentional training data leakage. Despite the critical nature of this issue, there has been no existing literature to offer a comprehensive assessment of data privacy risks in LLMs. Addressing this gap, our paper introduces LLM-PBE, a toolkit crafted specifically for the systematic evaluation of data privacy risks in LLMs. LLM-PBE is designed to analyze privacy across the entire lifecycle of LLMs, incorporating diverse attack and defense strategies, and handling various data types and metrics. Through detailed experimentation with multiple LLMs, LLM-PBE facilitates an in-depth exploration of data privacy concerns, shedding light on influential factors such as model size, data characteristics, and evolving temporal dimensions. This study not only enriches the understanding of privacy issues in LLMs but also serves as a vital resource for future research in the field. Aimed at enhancing the breadth of knowledge in this area, the findings, resources, and our full technical report are made available at https://llm-pbe.github.io/, providing an open platform for academic and practical advancements in LLM privacy assessment.

摘要: 大型语言模型(LLM)已经成为许多领域不可或缺的一部分，极大地推动了数据管理、挖掘和分析方面的应用。然而，它们在处理和解释复杂语言数据方面的深厚能力暴露了人们对数据隐私的迫切关切，特别是无意中泄露培训数据的风险。尽管这一问题具有严重的性质，但目前还没有文献对低成本管理中的数据隐私风险进行全面评估。针对这一差距，我们引入了LLM-PBE，这是一个专门为系统评估LLMS中的数据隐私风险而设计的工具包。LLm-PBE旨在分析LLMS整个生命周期中的隐私，整合不同的攻击和防御策略，并处理各种数据类型和指标。通过对多个LLM的详细实验，LLM-PBE有助于深入探索数据隐私问题，揭示模型大小、数据特征和不断演变的时间维度等影响因素。这一研究不仅丰富了对LLMS中隐私问题的理解，也为该领域未来的研究提供了重要的资源。为了提高这一领域的知识广度，我们的调查结果、资源和完整的技术报告可在https://llm-pbe.github.io/，上获得，为LLM隐私评估的学术和实践进步提供了一个开放的平台。



## **3. SelfDefend: LLMs Can Defend Themselves against Jailbreaking in a Practical Manner**

SelfDefend：LLM可以以实用的方式保护自己免受越狱的侵害 cs.CR

This paper completes its earlier vision paper, available at  arXiv:2402.15727. Updated to the latest analysis and results

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2406.05498v2) [paper-pdf](http://arxiv.org/pdf/2406.05498v2)

**Authors**: Xunguang Wang, Daoyuan Wu, Zhenlan Ji, Zongjie Li, Pingchuan Ma, Shuai Wang, Yingjiu Li, Yang Liu, Ning Liu, Juergen Rahmel

**Abstract**: Jailbreaking is an emerging adversarial attack that bypasses the safety alignment deployed in off-the-shelf large language models (LLMs) and has evolved into multiple categories: human-based, optimization-based, generation-based, and the recent indirect and multilingual jailbreaks. However, delivering a practical jailbreak defense is challenging because it needs to not only handle all the above jailbreak attacks but also incur negligible delays to user prompts, as well as be compatible with both open-source and closed-source LLMs. Inspired by how the traditional security concept of shadow stacks defends against memory overflow attacks, this paper introduces a generic LLM jailbreak defense framework called SelfDefend, which establishes a shadow LLM as a defense instance to concurrently protect the target LLM instance in the normal stack and collaborate with it for checkpoint-based access control. The effectiveness of SelfDefend builds upon our observation that existing LLMs (both target and defense LLMs) have the capability to identify harmful prompts or intentions in user queries, which we empirically validate using the commonly used GPT-3.5/4 models across all major jailbreak attacks. To further improve the defense's robustness and minimize costs, we employ a data distillation approach to tune dedicated open-source defense models. These models outperform six state-of-the-art defenses and match the performance of GPT-4-based SelfDefend, with significantly lower extra delays. We also empirically show that the tuned models are robust to adaptive jailbreaks and prompt injections.

摘要: 越狱是一种新兴的对抗性攻击，它绕过了现成的大型语言模型(LLM)中部署的安全对齐，并已演变为多种类别：基于人的、基于优化的、基于代的以及最近的间接和多语言越狱。然而，提供实际的越狱防御是具有挑战性的，因为它不仅需要处理所有上述越狱攻击，还需要对用户提示造成可以忽略不计的延迟，以及与开源和闭源LLM兼容。受传统影子堆栈安全概念防御内存溢出攻击的启发，提出了一种通用的LLM越狱防御框架--SelfDefend。该框架建立一个影子LLM作为防御实例，同时保护正常堆栈中的目标LLM实例，并与其协作进行基于检查点的访问控制。SelfDefend的有效性建立在我们的观察基础上，即现有的LLM(目标和防御LLM)能够识别用户查询中的有害提示或意图，我们使用所有主要越狱攻击中常用的GPT-3.5/4模型进行了经验验证。为了进一步提高防御的健壮性并将成本降至最低，我们使用数据蒸馏方法来优化专用的开源防御模型。这些型号的性能超过了六种最先进的防御系统，并与基于GPT-4的SelfDefend的性能相当，额外延迟显著降低。我们的经验还表明，调整后的模型对自适应越狱和快速注入具有较强的鲁棒性。



## **4. Towards Neural Network based Cognitive Models of Dynamic Decision-Making by Humans**

基于神经网络的人类动态决策认知模型 cs.LG

Our code is available at https://github.com/shshnkreddy/NCM-HDM

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2407.17622v2) [paper-pdf](http://arxiv.org/pdf/2407.17622v2)

**Authors**: Changyu Chen, Shashank Reddy Chirra, Maria José Ferreira, Cleotilde Gonzalez, Arunesh Sinha, Pradeep Varakantham

**Abstract**: Modeling human cognitive processes in dynamic decision-making tasks has been an endeavor in AI for a long time because such models can help make AI systems more intuitive, personalized, mitigate any human biases, and enhance training in simulation. Some initial work has attempted to utilize neural networks (and large language models) but often assumes one common model for all humans and aims to emulate human behavior in aggregate. However, the behavior of each human is distinct, heterogeneous, and relies on specific past experiences in certain tasks. For instance, consider two individuals responding to a phishing email: one who has previously encountered and identified similar threats may recognize it quickly, while another without such experience might fall for the scam. In this work, we build on Instance Based Learning (IBL) that posits that human decisions are based on similar situations encountered in the past. However, IBL relies on simple fixed form functions to capture the mapping from past situations to current decisions. To that end, we propose two new attention-based neural network models to have open form non-linear functions to model distinct and heterogeneous human decision-making in dynamic settings. We experiment with two distinct datasets gathered from human subject experiment data, one focusing on detection of phishing email by humans and another where humans act as attackers in a cybersecurity setting and decide on an attack option. We conducted extensive experiments with our two neural network models, IBL, and GPT3.5, and demonstrate that the neural network models outperform IBL significantly in representing human decision-making, while providing similar interpretability of human decisions as IBL. Overall, our work yields promising results for further use of neural networks in cognitive modeling of human decision making.

摘要: 长期以来，在动态决策任务中对人类认知过程进行建模一直是人工智能领域的一项努力，因为这样的模型可以帮助人工智能系统变得更加直观、个性化，缓解任何人类偏见，并加强模拟训练。一些最初的工作试图利用神经网络(和大型语言模型)，但通常假设一个适用于所有人类的通用模型，并旨在总体上模拟人类的行为。然而，每个人的行为都是不同的、不同的，并依赖于某些任务中特定的过去经验。例如，假设两个人回复了一封钓鱼电子邮件：其中一个人之前遇到并识别了类似的威胁，可能很快就能识别出它，而另一个没有这种经验的人可能会上当。在这项工作中，我们建立在基于实例的学习(IBL)的基础上，该学习假设人类的决策是基于过去遇到的类似情况。然而，IBL依赖于简单的固定表单函数来捕获从过去情况到当前决策的映射。为此，我们提出了两个新的基于注意力的神经网络模型，它们具有开放形式的非线性函数，以在动态环境中模拟不同和不同种类的人类决策。我们使用从人类受试者实验数据中收集的两个不同的数据集进行实验，一个专注于检测人类发送的钓鱼电子邮件，另一个则是人类在网络安全环境中充当攻击者，并决定攻击选项。我们用我们的两个神经网络模型IBL和GPT3.5进行了广泛的实验，并证明了神经网络模型在表示人类决策方面显著优于IBL，同时提供了与IBL相似的人类决策的可解释性。总体而言，我们的工作为进一步使用神经网络对人类决策进行认知建模带来了令人振奋的结果。



## **5. Unleashing the potential of prompt engineering in Large Language Models: a comprehensive review**

释放大型语言模型中提示工程的潜力：全面回顾 cs.CL

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2310.14735v5) [paper-pdf](http://arxiv.org/pdf/2310.14735v5)

**Authors**: Banghao Chen, Zhaofeng Zhang, Nicolas Langrené, Shengxin Zhu

**Abstract**: This comprehensive review delves into the pivotal role of prompt engineering in unleashing the capabilities of Large Language Models (LLMs). The development of Artificial Intelligence (AI), from its inception in the 1950s to the emergence of advanced neural networks and deep learning architectures, has made a breakthrough in LLMs, with models such as GPT-4o and Claude-3, and in Vision-Language Models (VLMs), with models such as CLIP and ALIGN. Prompt engineering is the process of structuring inputs, which has emerged as a crucial technique to maximize the utility and accuracy of these models. This paper explores both foundational and advanced methodologies of prompt engineering, including techniques such as self-consistency, chain-of-thought, and generated knowledge, which significantly enhance model performance. Additionally, it examines the prompt method of VLMs through innovative approaches such as Context Optimization (CoOp), Conditional Context Optimization (CoCoOp), and Multimodal Prompt Learning (MaPLe). Critical to this discussion is the aspect of AI security, particularly adversarial attacks that exploit vulnerabilities in prompt engineering. Strategies to mitigate these risks and enhance model robustness are thoroughly reviewed. The evaluation of prompt methods is also addressed, through both subjective and objective metrics, ensuring a robust analysis of their efficacy. This review also reflects the essential role of prompt engineering in advancing AI capabilities, providing a structured framework for future research and application.

摘要: 这篇全面的综述深入探讨了快速工程在释放大型语言模型(LLM)能力方面的关键作用。人工智能(AI)的发展，从20世纪50年代开始，到先进的神经网络和深度学习体系的出现，在LLMS方面取得了突破，有GPT-40和Claude-3等模型，以及视觉语言模型(VLMS)，有CLIP和ALIGN等模型。即时工程是对输入进行结构化的过程，它已成为最大化这些模型的实用性和准确性的关键技术。本文探讨了即时工程的基本方法和高级方法，包括自我一致性、思想链和生成知识等技术，这些技术显著提高了模型的性能。此外，它还通过诸如语境优化(COOP)、条件语境优化(CoCoOp)和多通道提示学习(Maple)等创新方法来研究虚拟学习模型的提示方法。对这一讨论至关重要的是人工智能安全方面，特别是利用即时工程中的漏洞进行的对抗性攻击。对缓解这些风险和增强模型稳健性的策略进行了彻底的回顾。还通过主观和客观指标对快速方法进行了评估，以确保对其有效性进行稳健的分析。这篇综述还反映了快速工程在推进人工智能能力方面的重要作用，为未来的研究和应用提供了一个结构化的框架。



## **6. LLM Detectors Still Fall Short of Real World: Case of LLM-Generated Short News-Like Posts**

LLM探测器仍然达不到现实世界：LLM生成的短新闻类帖子的案例 cs.CL

20 pages, 7 tables, 13 figures, under consideration for EMNLP

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2409.03291v1) [paper-pdf](http://arxiv.org/pdf/2409.03291v1)

**Authors**: Henrique Da Silva Gameiro, Andrei Kucharavy, Ljiljana Dolamic

**Abstract**: With the emergence of widely available powerful LLMs, disinformation generated by large Language Models (LLMs) has become a major concern. Historically, LLM detectors have been touted as a solution, but their effectiveness in the real world is still to be proven. In this paper, we focus on an important setting in information operations -- short news-like posts generated by moderately sophisticated attackers.   We demonstrate that existing LLM detectors, whether zero-shot or purpose-trained, are not ready for real-world use in that setting. All tested zero-shot detectors perform inconsistently with prior benchmarks and are highly vulnerable to sampling temperature increase, a trivial attack absent from recent benchmarks. A purpose-trained detector generalizing across LLMs and unseen attacks can be developed, but it fails to generalize to new human-written texts.   We argue that the former indicates domain-specific benchmarking is needed, while the latter suggests a trade-off between the adversarial evasion resilience and overfitting to the reference human text, with both needing evaluation in benchmarks and currently absent. We believe this suggests a re-consideration of current LLM detector benchmarking approaches and provides a dynamically extensible benchmark to allow it (https://github.com/Reliable-Information-Lab-HEVS/dynamic_llm_detector_benchmark).

摘要: 随着广泛使用的强大的LLM的出现，大型语言模型(LLM)产生的虚假信息已经成为一个主要关注的问题。从历史上看，LLM探测器一直被吹捧为一种解决方案，但它们在现实世界中的有效性仍有待证明。在这篇文章中，我们关注信息操作中的一个重要环境--由中等经验丰富的攻击者生成的类似新闻的短帖子。我们证明，现有的LLM探测器，无论是零炮还是专门训练的，都没有准备好在那种情况下用于现实世界。所有经过测试的零射击探测器的性能都与以前的基准不一致，并且非常容易受到采样温度升高的影响，这是最近的基准中所没有的一种轻微攻击。可以开发出一种专门训练的检测器，可以在LLMS和不可见攻击中推广，但它无法推广到新的人类书写的文本。我们认为，前者表明需要特定领域的基准测试，而后者则建议在对抗性回避韧性和对参考人类文本的过度匹配之间进行权衡，两者都需要在基准中进行评估，目前还没有。我们认为，这表明了对当前LLm探测器基准方法的重新考虑，并提供了一个动态可扩展的基准，以允许其(https://github.com/Reliable-Information-Lab-HEVS/dynamic_llm_detector_benchmark).



## **7. Well, that escalated quickly: The Single-Turn Crescendo Attack (STCA)**

嗯，情况迅速升级：单转渐强攻击（STCA） cs.CR

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2409.03131v1) [paper-pdf](http://arxiv.org/pdf/2409.03131v1)

**Authors**: Alan Aqrawi

**Abstract**: This paper explores a novel approach to adversarial attacks on large language models (LLM): the Single-Turn Crescendo Attack (STCA). The STCA builds upon the multi-turn crescendo attack established by Mark Russinovich, Ahmed Salem, Ronen Eldan. Traditional multi-turn adversarial strategies gradually escalate the context to elicit harmful or controversial responses from LLMs. However, this paper introduces a more efficient method where the escalation is condensed into a single interaction. By carefully crafting the prompt to simulate an extended dialogue, the attack bypasses typical content moderation systems, leading to the generation of responses that would normally be filtered out. I demonstrate this technique through a few case studies. The results highlight vulnerabilities in current LLMs and underscore the need for more robust safeguards. This work contributes to the broader discourse on responsible AI (RAI) safety and adversarial testing, providing insights and practical examples for researchers and developers. This method is unexplored in the literature, making it a novel contribution to the field.

摘要: 针对大型语言模型(LLM)提出了一种新的对抗性攻击方法：单轮渐近攻击(STCA)。STCA建立在Mark Russinovich，Ahmed Salem，Ronen Eldan建立的多轮渐强攻击的基础上。传统的多回合对抗性战略逐渐升级背景，以引起低收入国家的有害或有争议的反应。然而，本文介绍了一种更有效的方法，在该方法中，升级被压缩为单个交互。通过精心设计提示符来模拟延长的对话，攻击绕过了典型的内容审核系统，导致生成通常会被过滤掉的响应。我通过几个案例研究来演示这项技术。这一结果突显了当前LLM的脆弱性，并突显了需要更强有力的保障措施。这项工作有助于更广泛地讨论负责任的人工智能(RAI)安全和对抗性测试，为研究人员和开发人员提供见解和实践示例。这种方法在文献中还没有被探索过，这使它成为该领域的一个新贡献。



## **8. Revisiting Character-level Adversarial Attacks for Language Models**

重新审视语言模型的初级对抗攻击 cs.LG

Accepted in ICML 2024

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2405.04346v2) [paper-pdf](http://arxiv.org/pdf/2405.04346v2)

**Authors**: Elias Abad Rocamora, Yongtao Wu, Fanghui Liu, Grigorios G. Chrysos, Volkan Cevher

**Abstract**: Adversarial attacks in Natural Language Processing apply perturbations in the character or token levels. Token-level attacks, gaining prominence for their use of gradient-based methods, are susceptible to altering sentence semantics, leading to invalid adversarial examples. While character-level attacks easily maintain semantics, they have received less attention as they cannot easily adopt popular gradient-based methods, and are thought to be easy to defend. Challenging these beliefs, we introduce Charmer, an efficient query-based adversarial attack capable of achieving high attack success rate (ASR) while generating highly similar adversarial examples. Our method successfully targets both small (BERT) and large (Llama 2) models. Specifically, on BERT with SST-2, Charmer improves the ASR in 4.84% points and the USE similarity in 8% points with respect to the previous art. Our implementation is available in https://github.com/LIONS-EPFL/Charmer.

摘要: 自然语言处理中的对抗攻击对字符或令牌级别施加扰动。令牌级攻击因使用基于梯度的方法而变得越来越重要，很容易改变句子语义，从而导致无效的对抗性示例。虽然字符级攻击很容易维护语义，但它们受到的关注较少，因为它们不能轻易采用流行的基于梯度的方法，并且被认为很容易防御。基于这些信念，我们引入了Charmer，这是一种高效的基于查询的对抗性攻击，能够实现高攻击成功率（ASB），同时生成高度相似的对抗性示例。我们的方法成功地针对小型（BERT）和大型（Llama 2）模型。具体来说，在采用CST-2的BERT上，Charmer将ASB提高了4.84%，与之前的作品相比，USE相似性提高了8%。我们的实现可在https://github.com/LIONS-EPFL/Charmer上获取。



## **9. Alignment-Aware Model Extraction Attacks on Large Language Models**

对大型语言模型的对齐感知模型提取攻击 cs.CR

Source code: https://github.com/liangzid/alignmentExtraction

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2409.02718v1) [paper-pdf](http://arxiv.org/pdf/2409.02718v1)

**Authors**: Zi Liang, Qingqing Ye, Yanyun Wang, Sen Zhang, Yaxin Xiao, Ronghua Li, Jianliang Xu, Haibo Hu

**Abstract**: Model extraction attacks (MEAs) on large language models (LLMs) have received increasing research attention lately. Existing attack methods on LLMs inherit the extraction strategies from those designed for deep neural networks (DNNs) yet neglect the inconsistency of training tasks between MEA and LLMs' alignments. As such, they result in poor attack performances. To tackle this issue, we present Locality Reinforced Distillation (LoRD), a novel model extraction attack algorithm specifically for LLMs. In particular, we design a policy-gradient-style training task, which utilizes victim models' responses as a signal to guide the crafting of preference for the local model. Theoretical analysis has shown that i) LoRD's convergence procedure in MEAs is consistent with the alignments of LLMs, and ii) LoRD can reduce query complexity while mitigating watermark protection through exploration-based stealing. Extensive experiments on domain-specific extractions demonstrate the superiority of our method by examining the extraction of various state-of-the-art commercial LLMs.

摘要: 近年来，针对大型语言模型的模型提取攻击受到了越来越多的关注。现有的针对LLMS的攻击方法继承了针对深度神经网络(DNN)的提取策略，但忽略了MEA和LLMS对齐之间训练任务的不一致性。因此，他们的进攻表现很差。针对这一问题，我们提出了一种新的针对LLMS的模型提取攻击算法LOAD。特别是，我们设计了一个策略梯度式的训练任务，它利用受害者模型的反应作为一个信号来指导对局部模型的偏好的形成。理论分析表明：1)Lord在MEAS中的收敛过程与LLMS的比对是一致的；2)Lord在降低查询复杂度的同时，通过基于探测的窃取来减轻水印保护。对特定领域提取的大量实验通过检验各种最先进的商业LLM的提取来证明我们的方法的优越性。



## **10. Unveiling the Vulnerability of Private Fine-Tuning in Split-Based Frameworks for Large Language Models: A Bidirectionally Enhanced Attack**

揭露大型语言模型基于拆分的框架中的私人微调的漏洞：双向增强攻击 cs.CR

ACM Conference on Computer and Communications Security 2024 (CCS 24)

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2409.00960v2) [paper-pdf](http://arxiv.org/pdf/2409.00960v2)

**Authors**: Guanzhong Chen, Zhenghan Qin, Mingxin Yang, Yajie Zhou, Tao Fan, Tianyu Du, Zenglin Xu

**Abstract**: Recent advancements in pre-trained large language models (LLMs) have significantly influenced various domains. Adapting these models for specific tasks often involves fine-tuning (FT) with private, domain-specific data. However, privacy concerns keep this data undisclosed, and the computational demands for deploying LLMs pose challenges for resource-limited data holders. This has sparked interest in split learning (SL), a Model-as-a-Service (MaaS) paradigm that divides LLMs into smaller segments for distributed training and deployment, transmitting only intermediate activations instead of raw data. SL has garnered substantial interest in both industry and academia as it aims to balance user data privacy, model ownership, and resource challenges in the private fine-tuning of LLMs. Despite its privacy claims, this paper reveals significant vulnerabilities arising from the combination of SL and LLM-FT: the Not-too-far property of fine-tuning and the auto-regressive nature of LLMs. Exploiting these vulnerabilities, we propose Bidirectional Semi-white-box Reconstruction (BiSR), the first data reconstruction attack (DRA) designed to target both the forward and backward propagation processes of SL. BiSR utilizes pre-trained weights as prior knowledge, combining a learning-based attack with a bidirectional optimization-based approach for highly effective data reconstruction. Additionally, it incorporates a Noise-adaptive Mixture of Experts (NaMoE) model to enhance reconstruction performance under perturbation. We conducted systematic experiments on various mainstream LLMs and different setups, empirically demonstrating BiSR's state-of-the-art performance. Furthermore, we thoroughly examined three representative defense mechanisms, showcasing our method's capability to reconstruct private data even in the presence of these defenses.

摘要: 最近在预先训练的大型语言模型(LLM)方面的进展已经显著地影响了各个领域。针对特定任务调整这些模型通常涉及到使用特定领域的私有数据进行微调(FT)。然而，出于隐私考虑，这些数据不会被披露，而部署LLM的计算需求给资源有限的数据持有者带来了挑战。这引发了人们对拆分学习(SL)的兴趣，这是一种模型即服务(MAAS)范式，将LLM划分为较小的部分，用于分布式培训和部署，仅传输中间激活而不是原始数据。SL在工业界和学术界都引起了极大的兴趣，因为它旨在平衡用户数据隐私、模型所有权和LLM私下微调中的资源挑战。尽管声称其隐私，但本文揭示了SL和LLM-FT组合带来的重大漏洞：微调的不太远的特性和LLMS的自回归特性。利用这些漏洞，我们提出了双向半白盒重建(BiSR)，这是第一个针对SL的前向和后向传播过程的数据重建攻击(DRA)。BiSR利用预先训练的权值作为先验知识，将基于学习的攻击与基于双向优化的方法相结合，以实现高效的数据重建。此外，它还结合了噪声自适应专家混合(NaMoE)模型，以增强扰动下的重建性能。我们在各种主流的LLM和不同的设置上进行了系统的实验，实证地展示了BiSR的最先进的性能。此外，我们彻底检查了三种具有代表性的防御机制，展示了我们的方法即使在这些防御存在的情况下也能够重建私人数据的能力。



## **11. $\textit{MMJ-Bench}$: A Comprehensive Study on Jailbreak Attacks and Defenses for Vision Language Models**

$\texttit {MMJ-Bench}$：视觉语言模型越狱攻击和防御的综合研究 cs.CR

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2408.08464v2) [paper-pdf](http://arxiv.org/pdf/2408.08464v2)

**Authors**: Fenghua Weng, Yue Xu, Chengyan Fu, Wenjie Wang

**Abstract**: As deep learning advances, Large Language Models (LLMs) and their multimodal counterparts, Vision-Language Models (VLMs), have shown exceptional performance in many real-world tasks. However, VLMs face significant security challenges, such as jailbreak attacks, where attackers attempt to bypass the model's safety alignment to elicit harmful responses. The threat of jailbreak attacks on VLMs arises from both the inherent vulnerabilities of LLMs and the multiple information channels that VLMs process. While various attacks and defenses have been proposed, there is a notable gap in unified and comprehensive evaluations, as each method is evaluated on different dataset and metrics, making it impossible to compare the effectiveness of each method. To address this gap, we introduce \textit{MMJ-Bench}, a unified pipeline for evaluating jailbreak attacks and defense techniques for VLMs. Through extensive experiments, we assess the effectiveness of various attack methods against SoTA VLMs and evaluate the impact of defense mechanisms on both defense effectiveness and model utility for normal tasks. Our comprehensive evaluation contribute to the field by offering a unified and systematic evaluation framework and the first public-available benchmark for VLM jailbreak research. We also demonstrate several insightful findings that highlights directions for future studies.

摘要: 随着深度学习的深入，大型语言模型(LLM)及其多通道模型(Vision-Language Model，VLM)在许多实际任务中表现出了优异的性能。然而，VLM面临着重大的安全挑战，例如越狱攻击，攻击者试图绕过模型的安全对齐，以引发有害的响应。越狱攻击对VLMS的威胁既来自LLMS固有的脆弱性，也源于VLMS处理的多种信息渠道。虽然已经提出了各种攻击和防御方法，但在统一和综合评估方面存在显著差距，因为每种方法都是在不同的数据集和指标上进行评估，因此无法比较每种方法的有效性。为了弥补这一差距，我们引入了一个统一的管道，用于评估越狱攻击和针对VLM的防御技术。通过大量的实验，我们评估了各种攻击方法对SOTA VLMS的攻击效果，并评估了防御机制对正常任务的防御效果和模型效用的影响。我们的全面评估为越狱研究提供了统一和系统的评估框架和第一个公开可用的基准，从而为该领域做出了贡献。我们还展示了几个有洞察力的发现，这些发现突出了未来研究的方向。



## **12. LLM Defenses Are Not Robust to Multi-Turn Human Jailbreaks Yet**

LLM辩护对多次越狱还不强 cs.LG

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2408.15221v2) [paper-pdf](http://arxiv.org/pdf/2408.15221v2)

**Authors**: Nathaniel Li, Ziwen Han, Ian Steneker, Willow Primack, Riley Goodside, Hugh Zhang, Zifan Wang, Cristina Menghini, Summer Yue

**Abstract**: Recent large language model (LLM) defenses have greatly improved models' ability to refuse harmful queries, even when adversarially attacked. However, LLM defenses are primarily evaluated against automated adversarial attacks in a single turn of conversation, an insufficient threat model for real-world malicious use. We demonstrate that multi-turn human jailbreaks uncover significant vulnerabilities, exceeding 70% attack success rate (ASR) on HarmBench against defenses that report single-digit ASRs with automated single-turn attacks. Human jailbreaks also reveal vulnerabilities in machine unlearning defenses, successfully recovering dual-use biosecurity knowledge from unlearned models. We compile these results into Multi-Turn Human Jailbreaks (MHJ), a dataset of 2,912 prompts across 537 multi-turn jailbreaks. We publicly release MHJ alongside a compendium of jailbreak tactics developed across dozens of commercial red teaming engagements, supporting research towards stronger LLM defenses.

摘要: 最近的大型语言模型(LLM)防御极大地提高了模型拒绝有害查询的能力，即使在遭到恶意攻击时也是如此。然而，LLM防御主要是在单轮对话中针对自动对手攻击进行评估，这不足以构成现实世界恶意使用的威胁模型。我们证明，多轮人类越狱揭示了重大漏洞，针对使用自动化单轮攻击报告个位数ASR的防御系统，HarmB边上的攻击成功率(ASR)超过70%。人类越狱还揭示了机器遗忘防御的漏洞，成功地从未学习的模型中恢复了双重用途的生物安全知识。我们将这些结果汇编成多轮人类越狱(MHJ)，这是一个包含537次多轮越狱的2912个提示的数据集。我们公开发布了MHJ，以及在数十个商业红色团队交战中开发的越狱战术概要，支持对更强大的LLM防御的研究。



## **13. RACONTEUR: A Knowledgeable, Insightful, and Portable LLM-Powered Shell Command Explainer**

RACONTEur：一位知识渊博、富有洞察力且便携式的LLM驱动Shell命令解释者 cs.CR

Accepted by NDSS Symposium 2025. Please cite this paper as "Jiangyi  Deng, Xinfeng Li, Yanjiao Chen, Yijie Bai, Haiqin Weng, Yan Liu, Tao Wei,  Wenyuan Xu. RACONTEUR: A Knowledgeable, Insightful, and Portable LLM-Powered  Shell Command Explainer. In the 32nd Annual Network and Distributed System  Security Symposium (NDSS 2025)."

**SubmitDate**: 2024-09-03    [abs](http://arxiv.org/abs/2409.02074v1) [paper-pdf](http://arxiv.org/pdf/2409.02074v1)

**Authors**: Jiangyi Deng, Xinfeng Li, Yanjiao Chen, Yijie Bai, Haiqin Weng, Yan Liu, Tao Wei, Wenyuan Xu

**Abstract**: Malicious shell commands are linchpins to many cyber-attacks, but may not be easy to understand by security analysts due to complicated and often disguised code structures. Advances in large language models (LLMs) have unlocked the possibility of generating understandable explanations for shell commands. However, existing general-purpose LLMs suffer from a lack of expert knowledge and a tendency to hallucinate in the task of shell command explanation. In this paper, we present Raconteur, a knowledgeable, expressive and portable shell command explainer powered by LLM. Raconteur is infused with professional knowledge to provide comprehensive explanations on shell commands, including not only what the command does (i.e., behavior) but also why the command does it (i.e., purpose). To shed light on the high-level intent of the command, we also translate the natural-language-based explanation into standard technique & tactic defined by MITRE ATT&CK, the worldwide knowledge base of cybersecurity. To enable Raconteur to explain unseen private commands, we further develop a documentation retriever to obtain relevant information from complementary documentations to assist the explanation process. We have created a large-scale dataset for training and conducted extensive experiments to evaluate the capability of Raconteur in shell command explanation. The experiments verify that Raconteur is able to provide high-quality explanations and in-depth insight of the intent of the command.

摘要: 恶意的外壳命令是许多网络攻击的关键，但由于复杂且往往被伪装的代码结构，安全分析师可能不容易理解。大型语言模型(LLM)的进步使为外壳命令生成易于理解的解释成为可能。然而，现有的通用LLM存在缺乏专业知识和在解释外壳命令的任务中产生幻觉的倾向。在本文中，我们介绍了一个知识丰富、表达能力强、可移植的基于LLM的外壳命令解释器--raconteur。Raconteur具有丰富的专业知识，能够提供关于外壳命令的全面解释，不仅包括命令的作用(即行为)，还包括命令为什么这样做(即目的)。为了阐明该命令的高级意图，我们还将基于自然语言的解释转换为MITRE ATT&CK(全球网络安全知识库)定义的标准技术和策略。为了使raconteur能够解释看不见的私人命令，我们进一步开发了一个文档检索器，以从补充文档中获取相关信息，以帮助解释过程。我们创建了一个用于训练的大规模数据集，并进行了大量的实验来评估raconteur在解释外壳命令方面的能力。实验证明，raconteur能够提供高质量的解释和对命令意图的深入洞察。



## **14. Exploiting the Vulnerability of Large Language Models via Defense-Aware Architectural Backdoor**

通过防御意识架构后门利用大型语言模型的漏洞 cs.CR

**SubmitDate**: 2024-09-03    [abs](http://arxiv.org/abs/2409.01952v1) [paper-pdf](http://arxiv.org/pdf/2409.01952v1)

**Authors**: Abdullah Arafat Miah, Yu Bi

**Abstract**: Deep neural networks (DNNs) have long been recognized as vulnerable to backdoor attacks. By providing poisoned training data in the fine-tuning process, the attacker can implant a backdoor into the victim model. This enables input samples meeting specific textual trigger patterns to be classified as target labels of the attacker's choice. While such black-box attacks have been well explored in both computer vision and natural language processing (NLP), backdoor attacks relying on white-box attack philosophy have hardly been thoroughly investigated. In this paper, we take the first step to introduce a new type of backdoor attack that conceals itself within the underlying model architecture. Specifically, we pcricKet1996!ropose to design separate backdoor modules consisting of two functions: trigger detection and noise injection. The add-on modules of model architecture layers can detect the presence of input trigger tokens and modify layer weights using Gaussian noise to disturb the feature distribution of the baseline model. We conduct extensive experiments to evaluate our attack methods using two model architecture settings on five different large language datasets. We demonstrate that the training-free architectural backdoor on a large language model poses a genuine threat. Unlike the-state-of-art work, it can survive the rigorous fine-tuning and retraining process, as well as evade output probability-based defense methods (i.e. BDDR). All the code and data is available https://github.com/SiSL-URI/Arch_Backdoor_LLM.

摘要: 深度神经网络(DNN)长期以来一直被认为容易受到后门攻击。通过在微调过程中提供有毒的训练数据，攻击者可以在受害者模型中植入后门。这使得满足特定文本触发模式的输入样本能够被分类为攻击者选择的目标标签。虽然这种黑盒攻击在计算机视觉和自然语言处理(NLP)中都得到了很好的研究，但依赖于白盒攻击思想的后门攻击几乎没有得到彻底的调查。在本文中，我们首先介绍一种隐藏在底层模型体系结构中的新型后门攻击。具体地说，我们设计了独立的后门模块，包括两个功能：触发检测和噪声注入。模型体系结构层的附加模块可以检测输入触发令牌的存在，并使用高斯噪声来修改层权重，以干扰基线模型的特征分布。我们在五个不同的大型语言数据集上使用两个模型体系结构设置进行了广泛的实验来评估我们的攻击方法。我们证明，大型语言模型上的免培训体系结构后门构成了真正的威胁。与最先进的工作不同，它可以在严格的微调和重新训练过程中幸存下来，并且可以避开基于输出概率的防御方法(即BDDR)。所有代码和数据均可在https://github.com/SiSL-URI/Arch_Backdoor_LLM.上使用



## **15. FuzzCoder: Byte-level Fuzzing Test via Large Language Model**

FuzzCoder：通过大型语言模型进行字节级模糊测试 cs.CL

11 pages

**SubmitDate**: 2024-09-03    [abs](http://arxiv.org/abs/2409.01944v1) [paper-pdf](http://arxiv.org/pdf/2409.01944v1)

**Authors**: Liqun Yang, Jian Yang, Chaoren Wei, Guanglin Niu, Ge Zhang, Yunli Wang, Linzheng ChaI, Wanxu Xia, Hongcheng Guo, Shun Zhang, Jiaheng Liu, Yuwei Yin, Junran Peng, Jiaxin Ma, Liang Sun, Zhoujun Li

**Abstract**: Fuzzing is an important dynamic program analysis technique designed for finding vulnerabilities in complex software. Fuzzing involves presenting a target program with crafted malicious input to cause crashes, buffer overflows, memory errors, and exceptions. Crafting malicious inputs in an efficient manner is a difficult open problem and the best approaches often apply uniform random mutations to pre-existing valid inputs. In this work, we propose to adopt fine-tuned large language models (FuzzCoder) to learn patterns in the input files from successful attacks to guide future fuzzing explorations. Specifically, we develop a framework to leverage the code LLMs to guide the mutation process of inputs in fuzzing. The mutation process is formulated as the sequence-to-sequence modeling, where LLM receives a sequence of bytes and then outputs the mutated byte sequence. FuzzCoder is fine-tuned on the created instruction dataset (Fuzz-Instruct), where the successful fuzzing history is collected from the heuristic fuzzing tool. FuzzCoder can predict mutation locations and strategies locations in input files to trigger abnormal behaviors of the program. Experimental results show that FuzzCoder based on AFL (American Fuzzy Lop) gain significant improvements in terms of effective proportion of mutation (EPM) and number of crashes (NC) for various input formats including ELF, JPG, MP3, and XML.

摘要: 模糊是一种重要的动态程序分析技术，旨在发现复杂软件中的漏洞。Fuzing涉及向目标程序呈现精心编制的恶意输入，以导致崩溃、缓冲区溢出、内存错误和异常。以有效的方式创建恶意输入是一个困难的开放问题，最好的方法通常会对预先存在的有效输入应用统一的随机突变。在这项工作中，我们建议采用微调的大型语言模型(FuzzCoder)来从成功的攻击中学习输入文件中的模式，以指导未来的模糊探索。具体地说，我们开发了一个框架来利用代码LLMS来指导模糊化中输入的突变过程。突变过程被描述为序列到序列的建模，其中LLM接收字节序列，然后输出突变的字节序列。FuzzCoder在创建的指令数据集(Fuzz-Indict)上进行了微调，其中成功的模糊历史是从启发式模糊工具收集的。FuzzCoder可以预测输入文件中的突变位置和策略位置，从而触发程序的异常行为。实验结果表明，对于ELF、JPG、MP3和XML等多种输入格式，基于AFL(American FuzzLop)的FuzzCoder在有效变异比例(EPM)和崩溃次数(NC)方面都有明显的改善。



## **16. Safeguarding AI Agents: Developing and Analyzing Safety Architectures**

保护人工智能代理：开发和分析安全架构 cs.CR

**SubmitDate**: 2024-09-03    [abs](http://arxiv.org/abs/2409.03793v1) [paper-pdf](http://arxiv.org/pdf/2409.03793v1)

**Authors**: Ishaan Domkundwar, Mukunda N S

**Abstract**: AI agents, specifically powered by large language models, have demonstrated exceptional capabilities in various applications where precision and efficacy are necessary. However, these agents come with inherent risks, including the potential for unsafe or biased actions, vulnerability to adversarial attacks, lack of transparency, and tendency to generate hallucinations. As AI agents become more prevalent in critical sectors of the industry, the implementation of effective safety protocols becomes increasingly important. This paper addresses the critical need for safety measures in AI systems, especially ones that collaborate with human teams. We propose and evaluate three frameworks to enhance safety protocols in AI agent systems: an LLM-powered input-output filter, a safety agent integrated within the system, and a hierarchical delegation-based system with embedded safety checks. Our methodology involves implementing these frameworks and testing them against a set of unsafe agentic use cases, providing a comprehensive evaluation of their effectiveness in mitigating risks associated with AI agent deployment. We conclude that these frameworks can significantly strengthen the safety and security of AI agent systems, minimizing potential harmful actions or outputs. Our work contributes to the ongoing effort to create safe and reliable AI applications, particularly in automated operations, and provides a foundation for developing robust guardrails to ensure the responsible use of AI agents in real-world applications.

摘要: 人工智能代理，特别是由大型语言模型驱动的，在需要精确度和效率的各种应用中展示了非凡的能力。然而，这些代理伴随着固有的风险，包括潜在的不安全或有偏见的行动，易受对手攻击，缺乏透明度，以及产生幻觉的倾向。随着人工智能代理在该行业的关键部门变得越来越普遍，实施有效的安全协议变得越来越重要。本文讨论了人工智能系统中安全措施的迫切需要，特别是与人类团队协作的系统。我们提出并评估了三个框架来增强AI代理系统中的安全协议：LLM驱动的输入输出过滤器、集成在系统中的安全代理以及嵌入安全检查的基于分级委托的系统。我们的方法涉及实现这些框架并针对一组不安全的代理用例对它们进行测试，提供对它们在降低与AI代理部署相关的风险方面的有效性的全面评估。我们的结论是，这些框架可以显著加强AI代理系统的安全性和安全性，将潜在的有害行为或输出降至最低。我们的工作有助于持续努力创建安全可靠的人工智能应用程序，特别是在自动化操作中，并为开发强大的护栏提供基础，以确保在现实世界的应用程序中负责任地使用人工智能代理。



## **17. SafeEmbodAI: a Safety Framework for Mobile Robots in Embodied AI Systems**

SafeEmbodAI：人工智能系统中移动机器人的安全框架 cs.RO

**SubmitDate**: 2024-09-03    [abs](http://arxiv.org/abs/2409.01630v1) [paper-pdf](http://arxiv.org/pdf/2409.01630v1)

**Authors**: Wenxiao Zhang, Xiangrui Kong, Thomas Braunl, Jin B. Hong

**Abstract**: Embodied AI systems, including AI-powered robots that autonomously interact with the physical world, stand to be significantly advanced by Large Language Models (LLMs), which enable robots to better understand complex language commands and perform advanced tasks with enhanced comprehension and adaptability, highlighting their potential to improve embodied AI capabilities. However, this advancement also introduces safety challenges, particularly in robotic navigation tasks. Improper safety management can lead to failures in complex environments and make the system vulnerable to malicious command injections, resulting in unsafe behaviours such as detours or collisions. To address these issues, we propose \textit{SafeEmbodAI}, a safety framework for integrating mobile robots into embodied AI systems. \textit{SafeEmbodAI} incorporates secure prompting, state management, and safety validation mechanisms to secure and assist LLMs in reasoning through multi-modal data and validating responses. We designed a metric to evaluate mission-oriented exploration, and evaluations in simulated environments demonstrate that our framework effectively mitigates threats from malicious commands and improves performance in various environment settings, ensuring the safety of embodied AI systems. Notably, In complex environments with mixed obstacles, our method demonstrates a significant performance increase of 267\% compared to the baseline in attack scenarios, highlighting its robustness in challenging conditions.

摘要: 大型语言模型(LLM)将显著推进体现人工智能系统，包括自动与物理世界交互的人工智能机器人，大型语言模型使机器人能够更好地理解复杂的语言命令，并以增强的理解力和适应性执行高级任务，突显出它们提高体现人工智能能力的潜力。然而，这一进步也带来了安全挑战，特别是在机器人导航任务中。不恰当的安全管理可能会导致复杂环境中的故障，并使系统容易受到恶意命令注入的攻击，从而导致绕道或碰撞等不安全行为。为了解决这些问题，我们提出了一种将移动机器人集成到嵌入式AI系统中的安全框架。\textit{SafeEmbodAI}整合了安全提示、状态管理和安全验证机制，以通过多模式数据和验证响应来保护和帮助LLM进行推理。我们设计了一个指标来评估面向任务的探索，在模拟环境中的评估表明，我们的框架有效地缓解了恶意命令的威胁，提高了在各种环境下的性能，确保了体现的AI系统的安全。值得注意的是，在具有混合障碍的复杂环境中，与攻击场景中的基线相比，我们的方法表现出显著的性能提升，突出了其在具有挑战性的条件下的健壮性。



## **18. Antidote: Post-fine-tuning Safety Alignment for Large Language Models against Harmful Fine-tuning**

解药：微调后大型语言模型的安全调整，防止有害的微调 cs.AI

**SubmitDate**: 2024-09-03    [abs](http://arxiv.org/abs/2408.09600v2) [paper-pdf](http://arxiv.org/pdf/2408.09600v2)

**Authors**: Tiansheng Huang, Gautam Bhattacharya, Pratik Joshi, Josh Kimball, Ling Liu

**Abstract**: Safety aligned Large Language Models (LLMs) are vulnerable to harmful fine-tuning attacks \cite{qi2023fine}-- a few harmful data mixed in the fine-tuning dataset can break the LLMs's safety alignment. Existing mitigation strategies include alignment stage solutions \cite{huang2024vaccine, rosati2024representation} and fine-tuning stage solutions \cite{huang2024lazy,mukhoti2023fine}. However, our evaluation shows that both categories of defenses fail \textit{when some specific training hyper-parameters are chosen} -- a large learning rate or a large number of training epochs in the fine-tuning stage can easily invalidate the defense, which however, is necessary to guarantee finetune performance. To this end, we propose Antidote, a post-fine-tuning stage solution, which remains \textbf{\textit{agnostic to the training hyper-parameters in the fine-tuning stage}}. Antidote relies on the philosophy that by removing the harmful parameters, the harmful model can be recovered from the harmful behaviors, regardless of how those harmful parameters are formed in the fine-tuning stage. With this philosophy, we introduce a one-shot pruning stage after harmful fine-tuning to remove the harmful weights that are responsible for the generation of harmful content. Despite its embarrassing simplicity, empirical results show that Antidote can reduce harmful score while maintaining accuracy on downstream tasks.Our project page is at \url{https://huangtiansheng.github.io/Antidote_gh_page/}

摘要: 安全对齐的大型语言模型(LLM)容易受到有害的微调攻击--在微调数据集中混合一些有害数据就会破坏LLMS的安全对齐。现有的缓解策略包括对齐阶段解决方案\cite{huang2024疫苗，rosati2024代表}和微调阶段解决方案\cite{huang2024 lazy，mukhoti2023 finy}。然而，我们的评估表明，这两类防御都失败了[当选择了一些特定的训练超参数}--在微调阶段，较大的学习速率或大量的训练周期很容易使防御失效，但这是保证精调性能所必需的。为此，我们提出了解毒剂，这是一种后微调阶段的解决方案，它仍然保持在文本bf{与微调阶段的训练超参数无关}}。解毒剂依靠的理念是，通过删除有害参数，可以从有害行为中恢复有害模型，而无论这些有害参数在微调阶段是如何形成的。本着这一理念，我们引入了有害微调后的一次修剪阶段，以去除导致有害内容生成的有害权重。尽管解毒剂简单得令人尴尬，但实验结果表明，解毒剂可以降低有害分数，同时保持下游任务的准确性。我们的项目页面位于\url{https://huangtiansheng.github.io/Antidote_gh_page/}



## **19. Membership Inference Attacks Against In-Context Learning**

针对上下文内学习的成员推理攻击 cs.CR

To Appear in the ACM Conference on Computer and Communications  Security, October 14-18, 2024

**SubmitDate**: 2024-09-02    [abs](http://arxiv.org/abs/2409.01380v1) [paper-pdf](http://arxiv.org/pdf/2409.01380v1)

**Authors**: Rui Wen, Zheng Li, Michael Backes, Yang Zhang

**Abstract**: Adapting Large Language Models (LLMs) to specific tasks introduces concerns about computational efficiency, prompting an exploration of efficient methods such as In-Context Learning (ICL). However, the vulnerability of ICL to privacy attacks under realistic assumptions remains largely unexplored. In this work, we present the first membership inference attack tailored for ICL, relying solely on generated texts without their associated probabilities. We propose four attack strategies tailored to various constrained scenarios and conduct extensive experiments on four popular large language models. Empirical results show that our attacks can accurately determine membership status in most cases, e.g., 95\% accuracy advantage against LLaMA, indicating that the associated risks are much higher than those shown by existing probability-based attacks. Additionally, we propose a hybrid attack that synthesizes the strengths of the aforementioned strategies, achieving an accuracy advantage of over 95\% in most cases. Furthermore, we investigate three potential defenses targeting data, instruction, and output. Results demonstrate combining defenses from orthogonal dimensions significantly reduces privacy leakage and offers enhanced privacy assurances.

摘要: 将大型语言模型(LLM)适应于特定任务会引起对计算效率的担忧，促使人们探索高效的方法，如上下文中学习(ICL)。然而，ICL在现实假设下对隐私攻击的脆弱性在很大程度上仍未被探索。在这项工作中，我们提出了第一个为ICL量身定做的成员关系推理攻击，仅依赖于没有关联概率的生成文本。针对不同的约束场景，我们提出了四种攻击策略，并在四个流行的大型语言模型上进行了广泛的实验。实验结果表明，我们的攻击在大多数情况下都可以准确地确定成员身份，例如对骆驼的95%的准确率优势，表明关联的风险比现有的基于概率的攻击要高得多。此外，我们还提出了一种综合上述策略优点的混合攻击方法，在大多数情况下获得了95%以上的准确率优势。此外，我们还研究了针对数据、指令和输出的三种潜在防御措施。结果表明，从正交维组合防御显着减少隐私泄漏，并提供增强的隐私保证。



## **20. Privacy-Aware Document Visual Question Answering**

隐私意识文档视觉问题解答 cs.CV

35 pages, 12 figures, accepted for publication at the 18th  International Conference on Document Analysis and Recognition, ICDAR 2024

**SubmitDate**: 2024-09-02    [abs](http://arxiv.org/abs/2312.10108v2) [paper-pdf](http://arxiv.org/pdf/2312.10108v2)

**Authors**: Rubèn Tito, Khanh Nguyen, Marlon Tobaben, Raouf Kerkouche, Mohamed Ali Souibgui, Kangsoo Jung, Joonas Jälkö, Vincent Poulain D'Andecy, Aurelie Joseph, Lei Kang, Ernest Valveny, Antti Honkela, Mario Fritz, Dimosthenis Karatzas

**Abstract**: Document Visual Question Answering (DocVQA) has quickly grown into a central task of document understanding. But despite the fact that documents contain sensitive or copyrighted information, none of the current DocVQA methods offers strong privacy guarantees. In this work, we explore privacy in the domain of DocVQA for the first time, highlighting privacy issues in state of the art multi-modal LLM models used for DocVQA, and explore possible solutions. Specifically, we focus on invoice processing as a realistic document understanding scenario, and propose a large scale DocVQA dataset comprising invoice documents and associated questions and answers. We employ a federated learning scheme, that reflects the real-life distribution of documents in different businesses, and we explore the use case where the data of the invoice provider is the sensitive information to be protected. We demonstrate that non-private models tend to memorise, a behaviour that can lead to exposing private information. We then evaluate baseline training schemes employing federated learning and differential privacy in this multi-modal scenario, where the sensitive information might be exposed through either or both of the two input modalities: vision (document image) or language (OCR tokens). Finally, we design attacks exploiting the memorisation effect of the model, and demonstrate their effectiveness in probing a representative DocVQA models.

摘要: 文档视觉问答(DocVQA)已迅速发展成为文档理解的中心任务。但是，尽管文档包含敏感或受版权保护的信息，但当前的DocVQA方法都没有提供强有力的隐私保障。在这项工作中，我们首次探索了DocVQA领域的隐私，突出了用于DocVQA的最先进的多模式LLM模型中的隐私问题，并探索了可能的解决方案。具体地说，我们将发票处理作为一个现实的文档理解场景，并提出了一个包括发票文档和相关问答的大规模DocVQA数据集。我们采用了一种联合学习方案，反映了文档在不同业务中的真实分布，并探索了发票提供商的数据是需要保护的敏感信息的用例。我们证明，非私人模式往往会记忆，这一行为可能导致私人信息泄露。然后，我们评估了在这个多模式场景中使用联合学习和差异隐私的基线训练方案，其中敏感信息可能通过两个输入通道中的一个或两个暴露：视觉(文档图像)或语言(OCR令牌)。最后，我们利用该模型的记忆效应设计了攻击，并在一个典型的DocVQA模型上进行了验证。



## **21. MedFuzz: Exploring the Robustness of Large Language Models in Medical Question Answering**

MedFuzz：探索医学问题解答中大型语言模型的鲁棒性 cs.CL

9 pages, 3 figures, 2 algorithms, appendix

**SubmitDate**: 2024-09-01    [abs](http://arxiv.org/abs/2406.06573v2) [paper-pdf](http://arxiv.org/pdf/2406.06573v2)

**Authors**: Robert Osazuwa Ness, Katie Matton, Hayden Helm, Sheng Zhang, Junaid Bajwa, Carey E. Priebe, Eric Horvitz

**Abstract**: Large language models (LLM) have achieved impressive performance on medical question-answering benchmarks. However, high benchmark accuracy does not imply that the performance generalizes to real-world clinical settings. Medical question-answering benchmarks rely on assumptions consistent with quantifying LLM performance but that may not hold in the open world of the clinic. Yet LLMs learn broad knowledge that can help the LLM generalize to practical conditions regardless of unrealistic assumptions in celebrated benchmarks. We seek to quantify how well LLM medical question-answering benchmark performance generalizes when benchmark assumptions are violated. Specifically, we present an adversarial method that we call MedFuzz (for medical fuzzing). MedFuzz attempts to modify benchmark questions in ways aimed at confounding the LLM. We demonstrate the approach by targeting strong assumptions about patient characteristics presented in the MedQA benchmark. Successful "attacks" modify a benchmark item in ways that would be unlikely to fool a medical expert but nonetheless "trick" the LLM into changing from a correct to an incorrect answer. Further, we present a permutation test technique that can ensure a successful attack is statistically significant. We show how to use performance on a "MedFuzzed" benchmark, as well as individual successful attacks. The methods show promise at providing insights into the ability of an LLM to operate robustly in more realistic settings.

摘要: 大型语言模型(LLM)在医学问答基准上取得了令人印象深刻的表现。然而，高基准准确率并不意味着性能适用于现实世界的临床设置。医学问题回答基准依赖于与量化LLM性能一致的假设，但这在诊所的开放世界中可能不成立。然而，LLM学习了广泛的知识，可以帮助LLM将其推广到实际情况，而不考虑著名基准中不切实际的假设。我们试图量化当违反基准假设时，LLM医疗问答基准性能的泛化程度。具体地说，我们提出了一种对抗性方法，我们称之为MedFuzz(用于医学模糊)。MedFuzz试图以混淆LLM的方式修改基准问题。我们通过针对MedQA基准中提出的关于患者特征的强烈假设来演示该方法。成功的“攻击”修改基准项目的方式不太可能愚弄医学专家，但仍然“诱骗”LLM将正确答案更改为不正确答案。此外，我们提出了一种置换测试技术，该技术可以确保成功的攻击具有统计意义。我们展示了如何使用“MedFuzze”基准测试的性能，以及个别成功的攻击。这些方法在洞察LLM在更现实的环境中稳健运行的能力方面表现出了希望。



## **22. The Dark Side of Human Feedback: Poisoning Large Language Models via User Inputs**

人类反馈的阴暗面：通过用户输入毒害大型语言模型 cs.CL

**SubmitDate**: 2024-09-01    [abs](http://arxiv.org/abs/2409.00787v1) [paper-pdf](http://arxiv.org/pdf/2409.00787v1)

**Authors**: Bocheng Chen, Hanqing Guo, Guangjing Wang, Yuanda Wang, Qiben Yan

**Abstract**: Large Language Models (LLMs) have demonstrated great capabilities in natural language understanding and generation, largely attributed to the intricate alignment process using human feedback. While alignment has become an essential training component that leverages data collected from user queries, it inadvertently opens up an avenue for a new type of user-guided poisoning attacks. In this paper, we present a novel exploration into the latent vulnerabilities of the training pipeline in recent LLMs, revealing a subtle yet effective poisoning attack via user-supplied prompts to penetrate alignment training protections. Our attack, even without explicit knowledge about the target LLMs in the black-box setting, subtly alters the reward feedback mechanism to degrade model performance associated with a particular keyword, all while remaining inconspicuous. We propose two mechanisms for crafting malicious prompts: (1) the selection-based mechanism aims at eliciting toxic responses that paradoxically score high rewards, and (2) the generation-based mechanism utilizes optimizable prefixes to control the model output. By injecting 1\% of these specially crafted prompts into the data, through malicious users, we demonstrate a toxicity score up to two times higher when a specific trigger word is used. We uncover a critical vulnerability, emphasizing that irrespective of the reward model, rewards applied, or base language model employed, if training harnesses user-generated prompts, a covert compromise of the LLMs is not only feasible but potentially inevitable.

摘要: 大型语言模型(LLM)在自然语言理解和生成方面表现出了强大的能力，这在很大程度上归功于使用人类反馈的复杂的对齐过程。虽然对齐已成为利用从用户查询中收集的数据的基本培训组件，但它无意中为用户引导的新型中毒攻击开辟了一条途径。在这篇文章中，我们提出了一种新的探索，在最近的LLMS中训练管道的潜在漏洞，揭示了一种微妙而有效的中毒攻击，通过用户提供的提示来穿透排列训练保护。我们的攻击，即使在没有关于黑盒设置中的目标LLM的明确知识的情况下，也会微妙地改变奖励反馈机制，以降低与特定关键字关联的模型性能，同时保持低调。我们提出了两种机制来制作恶意提示：(1)基于选择的机制旨在引发反常地获得高回报的有毒响应；(2)基于生成的机制利用可优化的前缀来控制模型输出。通过恶意用户向数据中注入1%这些精心设计的提示，我们证明了使用特定触发词时，毒性分数最高可高出两倍。我们发现了一个严重的漏洞，强调无论奖励模型、应用的奖励或使用的基本语言模型如何，如果培训利用用户生成的提示，LLM的秘密妥协不仅是可行的，而且可能是不可避免的。



## **23. Automatic Pseudo-Harmful Prompt Generation for Evaluating False Refusals in Large Language Models**

用于评估大型语言模型中虚假拒绝的自动伪有害提示生成 cs.CL

**SubmitDate**: 2024-09-01    [abs](http://arxiv.org/abs/2409.00598v1) [paper-pdf](http://arxiv.org/pdf/2409.00598v1)

**Authors**: Bang An, Sicheng Zhu, Ruiyi Zhang, Michael-Andrei Panaitescu-Liess, Yuancheng Xu, Furong Huang

**Abstract**: Safety-aligned large language models (LLMs) sometimes falsely refuse pseudo-harmful prompts, like "how to kill a mosquito," which are actually harmless. Frequent false refusals not only frustrate users but also provoke a public backlash against the very values alignment seeks to protect. In this paper, we propose the first method to auto-generate diverse, content-controlled, and model-dependent pseudo-harmful prompts. Using this method, we construct an evaluation dataset called PHTest, which is ten times larger than existing datasets, covers more false refusal patterns, and separately labels controversial prompts. We evaluate 20 LLMs on PHTest, uncovering new insights due to its scale and labeling. Our findings reveal a trade-off between minimizing false refusals and improving safety against jailbreak attacks. Moreover, we show that many jailbreak defenses significantly increase the false refusal rates, thereby undermining usability. Our method and dataset can help developers evaluate and fine-tune safer and more usable LLMs. Our code and dataset are available at https://github.com/umd-huang-lab/FalseRefusal

摘要: 安全对齐的大型语言模型(LLM)有时会错误地拒绝虚假有害的提示，如“如何杀死蚊子”，而这些提示实际上是无害的。频繁的虚假拒绝不仅让用户感到沮丧，还会引发公众对Align试图保护的价值观的强烈反对。在本文中，我们提出了第一种方法来自动生成多样化的、内容受控的、依赖于模型的伪有害提示。使用该方法，我们构建了一个评价数据集PHTest，它比现有的数据集大了十倍，覆盖了更多的错误拒绝模式，并分别对有争议的提示进行了标注。我们在PHTest上评估了20个LLM，发现了由于其规模和标签而产生的新见解。我们的发现揭示了在尽量减少虚假拒绝和提高针对越狱攻击的安全性之间的权衡。此外，我们发现许多越狱防御措施显著增加了错误拒绝率，从而破坏了可用性。我们的方法和数据集可以帮助开发人员评估和微调更安全、更可用的LLM。我们的代码和数据集可在https://github.com/umd-huang-lab/FalseRefusal上获得



## **24. Forget to Flourish: Leveraging Machine-Unlearning on Pretrained Language Models for Privacy Leakage**

忘记繁荣：利用预训练语言模型上的机器非学习来解决隐私泄露 cs.LG

**SubmitDate**: 2024-08-30    [abs](http://arxiv.org/abs/2408.17354v1) [paper-pdf](http://arxiv.org/pdf/2408.17354v1)

**Authors**: Md Rafi Ur Rashid, Jing Liu, Toshiaki Koike-Akino, Shagufta Mehnaz, Ye Wang

**Abstract**: Fine-tuning large language models on private data for downstream applications poses significant privacy risks in potentially exposing sensitive information. Several popular community platforms now offer convenient distribution of a large variety of pre-trained models, allowing anyone to publish without rigorous verification. This scenario creates a privacy threat, as pre-trained models can be intentionally crafted to compromise the privacy of fine-tuning datasets. In this study, we introduce a novel poisoning technique that uses model-unlearning as an attack tool. This approach manipulates a pre-trained language model to increase the leakage of private data during the fine-tuning process. Our method enhances both membership inference and data extraction attacks while preserving model utility. Experimental results across different models, datasets, and fine-tuning setups demonstrate that our attacks significantly surpass baseline performance. This work serves as a cautionary note for users who download pre-trained models from unverified sources, highlighting the potential risks involved.

摘要: 微调下游应用程序的私有数据上的大型语言模型可能会暴露敏感信息，这会带来重大的隐私风险。几个流行的社区平台现在提供了方便的分发各种预先训练的模型，允许任何人在没有严格验证的情况下发布。这种情况会对隐私造成威胁，因为可以故意构建预先训练的模型来损害微调数据集的隐私。在这项研究中，我们介绍了一种新的中毒技术，它使用模型遗忘作为攻击工具。该方法操纵预先训练的语言模型，以在微调过程中增加私有数据的泄漏。我们的方法在保持模型效用的同时增强了成员关系推理和数据提取攻击。在不同模型、数据集和微调设置上的实验结果表明，我们的攻击显著超过了基线性能。这项工作对从未经证实的来源下载预先培训的模型的用户来说是一个警示，强调了其中涉及的潜在风险。



## **25. Jailbreak Attacks and Defenses Against Large Language Models: A Survey**

针对大型语言模型的越狱攻击和防御：调查 cs.CR

**SubmitDate**: 2024-08-30    [abs](http://arxiv.org/abs/2407.04295v2) [paper-pdf](http://arxiv.org/pdf/2407.04295v2)

**Authors**: Sibo Yi, Yule Liu, Zhen Sun, Tianshuo Cong, Xinlei He, Jiaxing Song, Ke Xu, Qi Li

**Abstract**: Large Language Models (LLMs) have performed exceptionally in various text-generative tasks, including question answering, translation, code completion, etc. However, the over-assistance of LLMs has raised the challenge of "jailbreaking", which induces the model to generate malicious responses against the usage policy and society by designing adversarial prompts. With the emergence of jailbreak attack methods exploiting different vulnerabilities in LLMs, the corresponding safety alignment measures are also evolving. In this paper, we propose a comprehensive and detailed taxonomy of jailbreak attack and defense methods. For instance, the attack methods are divided into black-box and white-box attacks based on the transparency of the target model. Meanwhile, we classify defense methods into prompt-level and model-level defenses. Additionally, we further subdivide these attack and defense methods into distinct sub-classes and present a coherent diagram illustrating their relationships. We also conduct an investigation into the current evaluation methods and compare them from different perspectives. Our findings aim to inspire future research and practical implementations in safeguarding LLMs against adversarial attacks. Above all, although jailbreak remains a significant concern within the community, we believe that our work enhances the understanding of this domain and provides a foundation for developing more secure LLMs.

摘要: 大型语言模型(LLMS)在问答、翻译、代码补全等文本生成任务中表现出色。然而，LLMS的过度协助带来了越狱的挑战，这导致该模型通过设计敌意提示来生成针对使用策略和社会的恶意响应。随着利用LLMS中不同漏洞的越狱攻击方法的出现，相应的安全对齐措施也在不断发展。在本文中，我们提出了一个全面和详细的分类越狱攻防方法。例如，根据目标模型的透明性，将攻击方法分为黑盒攻击和白盒攻击。同时，我们将防御方法分为提示级防御和模型级防御。此外，我们还将这些攻击和防御方法进一步细分为不同的子类，并提供了一个连贯的图来说明它们之间的关系。我们还对现有的评估方法进行了调查，并从不同的角度对它们进行了比较。我们的发现旨在启发未来在保护LLM免受对手攻击方面的研究和实际实现。最重要的是，尽管越狱在社区中仍然是一个重要的问题，但我们相信我们的工作增进了对这个领域的了解，并为开发更安全的LLM提供了基础。



## **26. PromptSmooth: Certifying Robustness of Medical Vision-Language Models via Prompt Learning**

EntSmooth：通过提示学习认证医学视觉语言模型的稳健性 cs.CV

Accepted to MICCAI 2024

**SubmitDate**: 2024-08-29    [abs](http://arxiv.org/abs/2408.16769v1) [paper-pdf](http://arxiv.org/pdf/2408.16769v1)

**Authors**: Noor Hussein, Fahad Shamshad, Muzammal Naseer, Karthik Nandakumar

**Abstract**: Medical vision-language models (Med-VLMs) trained on large datasets of medical image-text pairs and later fine-tuned for specific tasks have emerged as a mainstream paradigm in medical image analysis. However, recent studies have highlighted the susceptibility of these Med-VLMs to adversarial attacks, raising concerns about their safety and robustness. Randomized smoothing is a well-known technique for turning any classifier into a model that is certifiably robust to adversarial perturbations. However, this approach requires retraining the Med-VLM-based classifier so that it classifies well under Gaussian noise, which is often infeasible in practice. In this paper, we propose a novel framework called PromptSmooth to achieve efficient certified robustness of Med-VLMs by leveraging the concept of prompt learning. Given any pre-trained Med-VLM, PromptSmooth adapts it to handle Gaussian noise by learning textual prompts in a zero-shot or few-shot manner, achieving a delicate balance between accuracy and robustness, while minimizing the computational overhead. Moreover, PromptSmooth requires only a single model to handle multiple noise levels, which substantially reduces the computational cost compared to traditional methods that rely on training a separate model for each noise level. Comprehensive experiments based on three Med-VLMs and across six downstream datasets of various imaging modalities demonstrate the efficacy of PromptSmooth. Our code and models are available at https://github.com/nhussein/promptsmooth.

摘要: 医学视觉-语言模型(MED-VLMS)在医学图像-文本对的大数据集上进行训练，然后针对特定任务进行微调，已成为医学图像分析的主流范式。然而，最近的研究强调了这些MED-VLM对对抗性攻击的敏感性，这引发了人们对它们的安全性和健壮性的担忧。随机化平滑是一种众所周知的技术，用于将任何分类器转换为可证明对对手扰动具有健壮性的模型。然而，这种方法需要重新训练基于Med-VLM的分类器，以便它在高斯噪声下能够很好地分类，这在实践中往往是不可行的。在本文中，我们提出了一种新的框架，称为PromptSmooth，通过利用快速学习的概念来实现MED-VLMS的高效认证健壮性。给定任何预先训练的Med-VLM，PromptSmooth通过以零镜头或少镜头方式学习文本提示，使其适应于处理高斯噪声，在准确性和稳健性之间实现微妙的平衡，同时最小化计算开销。此外，PromptSmooth只需要一个模型来处理多个噪声级，与传统方法相比，这大大降低了计算成本，传统方法依赖于为每个噪声级训练单独的模型。基于三个MED-VLM和不同成像模式的六个下游数据集的综合实验证明了PromptSmooth的有效性。我们的代码和模型可在https://github.com/nhussein/promptsmooth.上找到



## **27. Emerging Vulnerabilities in Frontier Models: Multi-Turn Jailbreak Attacks**

前沿模型中新出现的漏洞：多回合越狱攻击 cs.CR

**SubmitDate**: 2024-08-29    [abs](http://arxiv.org/abs/2409.00137v1) [paper-pdf](http://arxiv.org/pdf/2409.00137v1)

**Authors**: Tom Gibbs, Ethan Kosak-Hine, George Ingebretsen, Jason Zhang, Julius Broomfield, Sara Pieri, Reihaneh Iranmanesh, Reihaneh Rabbany, Kellin Pelrine

**Abstract**: Large language models (LLMs) are improving at an exceptional rate. However, these models are still susceptible to jailbreak attacks, which are becoming increasingly dangerous as models become increasingly powerful. In this work, we introduce a dataset of jailbreaks where each example can be input in both a single or a multi-turn format. We show that while equivalent in content, they are not equivalent in jailbreak success: defending against one structure does not guarantee defense against the other. Similarly, LLM-based filter guardrails also perform differently depending on not just the input content but the input structure. Thus, vulnerabilities of frontier models should be studied in both single and multi-turn settings; this dataset provides a tool to do so.

摘要: 大型语言模型（LLM）正在以惊人的速度改进。然而，这些模型仍然容易受到越狱攻击，随着模型变得越来越强大，越狱攻击变得越来越危险。在这项工作中，我们引入了一个越狱数据集，其中每个例子都可以以单回合或多回合格式输入。我们表明，虽然内容相当，但在越狱成功方面并不相同：防御一种结构并不能保证防御另一种结构。同样，基于LLM的过滤器护栏的性能也不同，不仅取决于输入内容，还取决于输入结构。因此，应在单圈和多圈环境下研究前沿模型的脆弱性;该数据集提供了这样做的工具。



## **28. The Dark Side of Function Calling: Pathways to Jailbreaking Large Language Models**

函数调用的阴暗面：越狱大型语言模型的途径 cs.CR

**SubmitDate**: 2024-08-29    [abs](http://arxiv.org/abs/2407.17915v3) [paper-pdf](http://arxiv.org/pdf/2407.17915v3)

**Authors**: Zihui Wu, Haichang Gao, Jianping He, Ping Wang

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities, but their power comes with significant security considerations. While extensive research has been conducted on the safety of LLMs in chat mode, the security implications of their function calling feature have been largely overlooked. This paper uncovers a critical vulnerability in the function calling process of LLMs, introducing a novel "jailbreak function" attack method that exploits alignment discrepancies, user coercion, and the absence of rigorous safety filters. Our empirical study, conducted on six state-of-the-art LLMs including GPT-4o, Claude-3.5-Sonnet, and Gemini-1.5-pro, reveals an alarming average success rate of over 90\% for this attack. We provide a comprehensive analysis of why function calls are susceptible to such attacks and propose defensive strategies, including the use of defensive prompts. Our findings highlight the urgent need for enhanced security measures in the function calling capabilities of LLMs, contributing to the field of AI safety by identifying a previously unexplored risk, designing an effective attack method, and suggesting practical defensive measures. Our code is available at https://github.com/wooozihui/jailbreakfunction.

摘要: 大型语言模型(LLM)已经展示了非凡的能力，但它们的强大也伴随着重要的安全考虑。虽然已经对聊天模式下的LLMS的安全性进行了广泛的研究，但其函数调用功能的安全含义在很大程度上被忽视了。本文揭示了LLMS函数调用过程中的一个严重漏洞，引入了一种新的“越狱函数”攻击方法，该方法利用了对齐差异、用户胁迫和缺乏严格的安全过滤器。我们在包括GPT-40、Claude-3.5-Sonnet和Gemini-1.5-Pro在内的六个最先进的LLM上进行的经验研究显示，该攻击的平均成功率超过90%，这是令人震惊的。我们对函数调用容易受到此类攻击的原因进行了全面分析，并提出了防御策略，包括使用防御提示。我们的发现突显了在LLMS的函数调用能力方面迫切需要增强安全措施，通过识别以前未探索的风险、设计有效的攻击方法并提出实用的防御措施来促进人工智能安全领域。我们的代码可以在https://github.com/wooozihui/jailbreakfunction.上找到



## **29. TF-Attack: Transferable and Fast Adversarial Attacks on Large Language Models**

TF攻击：对大型语言模型的可转移且快速对抗攻击 cs.CL

14 pages, 6 figures

**SubmitDate**: 2024-08-29    [abs](http://arxiv.org/abs/2408.13985v2) [paper-pdf](http://arxiv.org/pdf/2408.13985v2)

**Authors**: Zelin Li, Kehai Chen, Xuefeng Bai, Lemao Liu, Mingming Yang, Yang Xiang, Min Zhang

**Abstract**: With the great advancements in large language models (LLMs), adversarial attacks against LLMs have recently attracted increasing attention. We found that pre-existing adversarial attack methodologies exhibit limited transferability and are notably inefficient, particularly when applied to LLMs. In this paper, we analyze the core mechanisms of previous predominant adversarial attack methods, revealing that 1) the distributions of importance score differ markedly among victim models, restricting the transferability; 2) the sequential attack processes induces substantial time overheads. Based on the above two insights, we introduce a new scheme, named TF-Attack, for Transferable and Fast adversarial attacks on LLMs. TF-Attack employs an external LLM as a third-party overseer rather than the victim model to identify critical units within sentences. Moreover, TF-Attack introduces the concept of Importance Level, which allows for parallel substitutions of attacks. We conduct extensive experiments on 6 widely adopted benchmarks, evaluating the proposed method through both automatic and human metrics. Results show that our method consistently surpasses previous methods in transferability and delivers significant speed improvements, up to 20 times faster than earlier attack strategies.

摘要: 近年来，随着大型语言模型的发展，针对大型语言模型的对抗性攻击引起了越来越多的关注。我们发现，现有的对抗性攻击方法表现出有限的可转移性和显著的低效，特别是当应用于LLM时。本文分析了以往主流对抗性攻击方法的核心机制，发现1)不同受害者模型的重要性分数分布明显不同，限制了可转移性；2)顺序攻击过程导致了大量的时间开销。基于以上两点，我们提出了一种新的方案，称为TF-Attack，用于对LLMS进行可转移和快速对抗攻击。TF-Attack使用外部LLM作为第三方监督者，而不是受害者模型来识别判刑内的关键单元。此外，TF-Attack还引入了重要度的概念，允许并行替换攻击。我们在6个广泛采用的基准上进行了广泛的实验，从自动度量和人工度量两个方面对所提出的方法进行了评估。结果表明，我们的方法在可转移性上始终优于以前的方法，并提供了显著的速度改进，比以前的攻击策略快20倍。



## **30. FRACTURED-SORRY-Bench: Framework for Revealing Attacks in Conversational Turns Undermining Refusal Efficacy and Defenses over SORRY-Bench**

Fractured-SORRY-Bench：揭示对话转折中攻击的框架，削弱了对SORRY-Bench的拒绝功效和防御 cs.CL

4 pages, 2 tables

**SubmitDate**: 2024-08-28    [abs](http://arxiv.org/abs/2408.16163v1) [paper-pdf](http://arxiv.org/pdf/2408.16163v1)

**Authors**: Aman Priyanshu, Supriti Vijay

**Abstract**: This paper introduces FRACTURED-SORRY-Bench, a framework for evaluating the safety of Large Language Models (LLMs) against multi-turn conversational attacks. Building upon the SORRY-Bench dataset, we propose a simple yet effective method for generating adversarial prompts by breaking down harmful queries into seemingly innocuous sub-questions. Our approach achieves a maximum increase of +46.22\% in Attack Success Rates (ASRs) across GPT-4, GPT-4o, GPT-4o-mini, and GPT-3.5-Turbo models compared to baseline methods. We demonstrate that this technique poses a challenge to current LLM safety measures and highlights the need for more robust defenses against subtle, multi-turn attacks.

摘要: 本文介绍了FRACTURED-SORRY-Bench，这是一个用于评估大型语言模型（LLM）针对多轮对话攻击的安全性的框架。基于SORRY-Bench数据集，我们提出了一种简单而有效的方法，通过将有害的查询分解为看似无害的子问题来生成对抗性提示。与基线方法相比，我们的方法在GPT-4、GPT-4 o、GPT-4 o-mini和GPT-3.5-Turbo模型中实现了+46.22%的攻击成功率（SVR）最大增加。我们证明这种技术对当前的LLM安全措施构成了挑战，并强调了对微妙的多回合攻击进行更强大的防御的必要性。



## **31. Evading AI-Generated Content Detectors using Homoglyphs**

使用同字形躲避人工智能生成的内容检测器 cs.CL

**SubmitDate**: 2024-08-28    [abs](http://arxiv.org/abs/2406.11239v2) [paper-pdf](http://arxiv.org/pdf/2406.11239v2)

**Authors**: Aldan Creo, Shushanta Pudasaini

**Abstract**: The advent of large language models (LLMs) has enabled the generation of text that increasingly exhibits human-like characteristics. As the detection of such content is of significant importance, numerous studies have been conducted with the aim of developing reliable AI-generated text detectors. These detectors have demonstrated promising results on test data, but recent research has revealed that they can be circumvented by employing different techniques. In this paper, we present homoglyph-based attacks ($a \rightarrow {\alpha}$) as a means of circumventing existing detectors. A comprehensive evaluation was conducted to assess the effectiveness of these attacks on seven detectors, including ArguGPT, Binoculars, DetectGPT, Fast-DetectGPT, Ghostbuster, OpenAI's detector, and watermarking techniques, on five different datasets. Our findings demonstrate that homoglyph-based attacks can effectively circumvent state-of-the-art detectors, leading them to classify all texts as either AI-generated or human-written (decreasing the average Matthews Correlation Coefficient from 0.64 to -0.01). We then examine the effectiveness of these attacks by analyzing how homoglyphs impact different families of detectors. Finally, we discuss the implications of these findings and potential defenses against such attacks.

摘要: 大型语言模型(LLM)的出现使得文本的生成越来越显示出与人类相似的特征。由于对这类内容的检测非常重要，因此进行了许多研究，目的是开发可靠的人工智能生成的文本检测器。这些探测器在测试数据上显示了有希望的结果，但最近的研究表明，可以通过使用不同的技术来绕过它们。在这篇文章中，我们提出了基于同形文字的攻击($a\right tarrow{\Alpha}$)作为一种绕过现有检测器的手段。在五个不同的数据集上，对七个检测器，包括ArguGPT、双筒望远镜、DetectGPT、Fast-DetectGPT、Ghost Buster、OpenAI的检测器和水印技术进行了全面的评估，以评估这些攻击的有效性。我们的发现表明，基于同种文字的攻击可以有效地绕过最先进的检测器，导致他们将所有文本分类为人工生成的或人类编写的(将平均Matthews相关系数从0.64降低到-0.01)。然后，我们通过分析同种文字如何影响不同的检测器家族来检查这些攻击的有效性。最后，我们讨论了这些发现的含义和针对此类攻击的潜在防御措施。



## **32. Large Language Model Sentinel: LLM Agent for Adversarial Purification**

大型语言模型Sentinel：对抗性纯化的LLM代理 cs.CL

**SubmitDate**: 2024-08-28    [abs](http://arxiv.org/abs/2405.20770v3) [paper-pdf](http://arxiv.org/pdf/2405.20770v3)

**Authors**: Guang Lin, Qibin Zhao

**Abstract**: Over the past two years, the use of large language models (LLMs) has advanced rapidly. While these LLMs offer considerable convenience, they also raise security concerns, as LLMs are vulnerable to adversarial attacks by some well-designed textual perturbations. In this paper, we introduce a novel defense technique named Large LAnguage MOdel Sentinel (LLAMOS), which is designed to enhance the adversarial robustness of LLMs by purifying the adversarial textual examples before feeding them into the target LLM. Our method comprises two main components: a) Agent instruction, which can simulate a new agent for adversarial defense, altering minimal characters to maintain the original meaning of the sentence while defending against attacks; b) Defense guidance, which provides strategies for modifying clean or adversarial examples to ensure effective defense and accurate outputs from the target LLMs. Remarkably, the defense agent demonstrates robust defensive capabilities even without learning from adversarial examples. Additionally, we conduct an intriguing adversarial experiment where we develop two agents, one for defense and one for attack, and engage them in mutual confrontation. During the adversarial interactions, neither agent completely beat the other. Extensive experiments on both open-source and closed-source LLMs demonstrate that our method effectively defends against adversarial attacks, thereby enhancing adversarial robustness.

摘要: 在过去的两年里，大型语言模型(LLM)的使用取得了快速发展。虽然这些LLM提供了相当大的便利，但它们也引发了安全问题，因为LLM容易受到一些精心设计的文本扰动的敌意攻击。本文介绍了一种新的防御技术--大语言模型哨兵(LLAMOS)，该技术旨在通过在将对抗性文本实例输入目标LLM之前对其进行提纯来增强LLMS的对抗性健壮性。我们的方法包括两个主要部分：a)代理指令，它可以模拟一个新的代理进行对抗性防御，改变最少的字符，在防御攻击的同时保持句子的原始含义；b)防御指导，它提供修改干净或对抗性示例的策略，以确保目标LLMS的有效防御和准确输出。值得注意的是，防御代理展示了强大的防御能力，即使没有从对手的例子中学习。此外，我们还进行了一个有趣的对抗性实验，在这个实验中，我们开发了两个代理，一个用于防御，一个用于攻击，并让他们相互对抗。在敌对的互动中，两个代理都没有完全击败另一个。在开源和闭源LLMS上的大量实验表明，我们的方法有效地防御了对手攻击，从而增强了对手攻击的健壮性。



## **33. Investigating Coverage Criteria in Large Language Models: An In-Depth Study Through Jailbreak Attacks**

调查大型语言模型中的覆盖标准：通过越狱攻击的深入研究 cs.SE

**SubmitDate**: 2024-08-27    [abs](http://arxiv.org/abs/2408.15207v1) [paper-pdf](http://arxiv.org/pdf/2408.15207v1)

**Authors**: Shide Zhou, Tianlin Li, Kailong Wang, Yihao Huang, Ling Shi, Yang Liu, Haoyu Wang

**Abstract**: The swift advancement of large language models (LLMs) has profoundly shaped the landscape of artificial intelligence; however, their deployment in sensitive domains raises grave concerns, particularly due to their susceptibility to malicious exploitation. This situation underscores the insufficiencies in pre-deployment testing, highlighting the urgent need for more rigorous and comprehensive evaluation methods. This study presents a comprehensive empirical analysis assessing the efficacy of conventional coverage criteria in identifying these vulnerabilities, with a particular emphasis on the pressing issue of jailbreak attacks. Our investigation begins with a clustering analysis of the hidden states in LLMs, demonstrating that intrinsic characteristics of these states can distinctly differentiate between various types of queries. Subsequently, we assess the performance of these criteria across three critical dimensions: criterion level, layer level, and token level. Our findings uncover significant disparities in neuron activation patterns between the processing of normal and jailbreak queries, thereby corroborating the clustering results. Leveraging these findings, we propose an innovative approach for the real-time detection of jailbreak attacks by utilizing neural activation features. Our classifier demonstrates remarkable accuracy, averaging 96.33% in identifying jailbreak queries, including those that could lead to adversarial attacks. The importance of our research lies in its comprehensive approach to addressing the intricate challenges of LLM security. By enabling instantaneous detection from the model's first token output, our method holds promise for future systems integrating LLMs, offering robust real-time detection capabilities. This study advances our understanding of LLM security testing, and lays a critical foundation for the development of more resilient AI systems.

摘要: 大型语言模型(LLM)的迅速发展深刻地塑造了人工智能的格局；然而，它们在敏感领域的部署引发了严重的担忧，特别是因为它们容易受到恶意利用。这种情况突出了部署前测试的不足，突出表明迫切需要更严格和全面的评价方法。这项研究提供了一项全面的实证分析，评估了传统覆盖标准在识别这些漏洞方面的有效性，并特别强调了越狱攻击这一紧迫问题。我们的研究首先对LLMS中的隐藏状态进行了聚类分析，表明这些状态的内在特征可以明显区分不同类型的查询。随后，我们从三个关键维度评估这些标准的性能：标准级别、层级别和令牌级。我们的发现揭示了正常和越狱查询处理过程中神经元激活模式的显著差异，从而证实了集群结果。利用这些发现，我们提出了一种创新的方法，通过利用神经激活功能来实时检测越狱攻击。我们的分类器显示出惊人的准确率，平均识别越狱查询的准确率为96.33%，包括那些可能导致对抗性攻击的查询。我们研究的重要性在于它以全面的方法应对LLM安全的复杂挑战。通过从模型的第一个令牌输出实现瞬时检测，我们的方法为未来集成LLMS的系统带来了希望，提供了强大的实时检测能力。这项研究加深了我们对LLM安全测试的理解，并为开发更具弹性的人工智能系统奠定了关键基础。



## **34. Detecting AI Flaws: Target-Driven Attacks on Internal Faults in Language Models**

检测人工智能缺陷：对语言模型内部故障的目标驱动攻击 cs.CL

**SubmitDate**: 2024-08-27    [abs](http://arxiv.org/abs/2408.14853v1) [paper-pdf](http://arxiv.org/pdf/2408.14853v1)

**Authors**: Yuhao Du, Zhuo Li, Pengyu Cheng, Xiang Wan, Anningzhe Gao

**Abstract**: Large Language Models (LLMs) have become a focal point in the rapidly evolving field of artificial intelligence. However, a critical concern is the presence of toxic content within the pre-training corpus of these models, which can lead to the generation of inappropriate outputs. Investigating methods for detecting internal faults in LLMs can help us understand their limitations and improve their security. Existing methods primarily focus on jailbreaking attacks, which involve manually or automatically constructing adversarial content to prompt the target LLM to generate unexpected responses. These methods rely heavily on prompt engineering, which is time-consuming and usually requires specially designed questions. To address these challenges, this paper proposes a target-driven attack paradigm that focuses on directly eliciting the target response instead of optimizing the prompts. We introduce the use of another LLM as the detector for toxic content, referred to as ToxDet. Given a target toxic response, ToxDet can generate a possible question and a preliminary answer to provoke the target model into producing desired toxic responses with meanings equivalent to the provided one. ToxDet is trained by interacting with the target LLM and receiving reward signals from it, utilizing reinforcement learning for the optimization process. While the primary focus of the target models is on open-source LLMs, the fine-tuned ToxDet can also be transferred to attack black-box models such as GPT-4o, achieving notable results. Experimental results on AdvBench and HH-Harmless datasets demonstrate the effectiveness of our methods in detecting the tendencies of target LLMs to generate harmful responses. This algorithm not only exposes vulnerabilities but also provides a valuable resource for researchers to strengthen their models against such attacks.

摘要: 大型语言模型(LLM)已经成为快速发展的人工智能领域的一个焦点。然而，一个严重的关切是，这些模型的训练前语料库中存在有毒内容，这可能导致产生不适当的产出。研究LLMS内部故障检测方法可以帮助我们了解它们的局限性，提高它们的安全性。现有的方法主要集中在越狱攻击，这涉及手动或自动构建敌意内容，以促使目标LLM生成意外响应。这些方法严重依赖于即时工程，这是耗时的，通常需要专门设计的问题。为了应对这些挑战，本文提出了一种目标驱动的攻击范式，该范式专注于直接引发目标响应，而不是优化提示。我们介绍了使用另一种LLM作为有毒物质的检测器，称为ToxDet。给定目标毒性反应，ToxDet可以生成可能的问题和初步答案，以激发目标模型产生与所提供的含义相同的所需毒性反应。ToxDet是通过与目标LLM交互并从其接收奖励信号来训练的，利用强化学习进行优化过程。虽然目标模型的主要焦点是开源LLMS，但微调的ToxDet也可以转移到攻击GPT-40等黑匣子模型上，取得了显著的效果。在AdvBtch和HH-无害数据集上的实验结果表明，该方法在检测目标LLM产生有害响应的倾向方面是有效的。该算法不仅暴露了漏洞，而且为研究人员提供了一个宝贵的资源，以加强他们的模型对此类攻击的攻击。



## **35. Image-to-Text Logic Jailbreak: Your Imagination can Help You Do Anything**

图像到文本逻辑越狱：你的想象力可以帮助你做任何事情 cs.CR

**SubmitDate**: 2024-08-26    [abs](http://arxiv.org/abs/2407.02534v2) [paper-pdf](http://arxiv.org/pdf/2407.02534v2)

**Authors**: Xiaotian Zou, Ke Li, Yongkang Chen

**Abstract**: Large Visual Language Model\textbfs (VLMs) such as GPT-4V have achieved remarkable success in generating comprehensive and nuanced responses. Researchers have proposed various benchmarks for evaluating the capabilities of VLMs. With the integration of visual and text inputs in VLMs, new security issues emerge, as malicious attackers can exploit multiple modalities to achieve their objectives. This has led to increasing attention on the vulnerabilities of VLMs to jailbreak. Most existing research focuses on generating adversarial images or nonsensical image to jailbreak these models. However, no researchers evaluate whether logic understanding capabilities of VLMs in flowchart can influence jailbreak. Therefore, to fill this gap, this paper first introduces a novel dataset Flow-JD specifically designed to evaluate the logic-based flowchart jailbreak capabilities of VLMs. We conduct an extensive evaluation on GPT-4o, GPT-4V, other 5 SOTA open source VLMs and the jailbreak rate is up to 92.8%. Our research reveals significant vulnerabilities in current VLMs concerning image-to-text jailbreak and these findings underscore the the urgency for the development of robust and effective future defenses.

摘要: GPT-4V等大型可视语言模型/文本语言模型(VLMS)在生成全面和细微差别的答复方面取得了显着成功。研究人员已经提出了各种基准来评估VLM的能力。随着视觉和文本输入在VLMS中的集成，新的安全问题出现了，因为恶意攻击者可以利用多种模式来实现他们的目标。这引起了人们对越狱漏洞的越来越多的关注。现有的研究大多集中在生成敌意图像或无意义图像来越狱这些模型。然而，还没有研究人员评估流程图中VLM的逻辑理解能力是否会影响越狱。因此，为了填补这一空白，本文首先介绍了一种新的数据集Flow-JD，该数据集专门用于评估基于逻辑流程图的VLM越狱能力。我们对GPT-40、GPT-4V等5个SOTA开源VLMS进行了广泛的评估，越狱率高达92.8%。我们的研究揭示了当前VLM在图像到文本越狱方面的重大漏洞，这些发现突显了开发强大和有效的未来防御措施的紧迫性。



## **36. Investigating the Effectiveness of Bayesian Spam Filters in Detecting LLM-modified Spam Mails**

调查Bayesian垃圾邮件过滤器检测LLM修改的垃圾邮件的有效性 cs.CR

EAI International Conference on Digital Forensics & Cyber Crime 2024

**SubmitDate**: 2024-08-26    [abs](http://arxiv.org/abs/2408.14293v1) [paper-pdf](http://arxiv.org/pdf/2408.14293v1)

**Authors**: Malte Josten, Torben Weis

**Abstract**: Spam and phishing remain critical threats in cybersecurity, responsible for nearly 90% of security incidents. As these attacks grow in sophistication, the need for robust defensive mechanisms intensifies. Bayesian spam filters, like the widely adopted open-source SpamAssassin, are essential tools in this fight. However, the emergence of large language models (LLMs) such as ChatGPT presents new challenges. These models are not only powerful and accessible, but also inexpensive to use, raising concerns about their misuse in crafting sophisticated spam emails that evade traditional spam filters. This work aims to evaluate the robustness and effectiveness of SpamAssassin against LLM-modified email content. We developed a pipeline to test this vulnerability. Our pipeline modifies spam emails using GPT-3.5 Turbo and assesses SpamAssassin's ability to classify these modified emails correctly. The results show that SpamAssassin misclassified up to 73.7% of LLM-modified spam emails as legitimate. In contrast, a simpler dictionary-replacement attack showed a maximum success rate of only 0.4%. These findings highlight the significant threat posed by LLM-modified spam, especially given the cost-efficiency of such attacks (0.17 cents per email). This paper provides crucial insights into the vulnerabilities of current spam filters and the need for continuous improvement in cybersecurity measures.

摘要: 垃圾邮件和网络钓鱼仍然是网络安全中的关键威胁，导致了近90%的安全事件。随着这些攻击变得越来越复杂，对强大防御机制的需求也变得更加迫切。贝叶斯垃圾邮件过滤器，就像被广泛采用的开源SpamAssassin一样，是这场斗争中必不可少的工具。然而，像ChatGPT这样的大型语言模型(LLM)的出现带来了新的挑战。这些模型不仅功能强大、易于访问，而且使用起来也不贵，这引发了人们对它们在制作复杂的垃圾邮件时被滥用的担忧，这些垃圾邮件绕过了传统的垃圾邮件过滤器。这项工作的目的是评估SpamAssassin对LLM修改的电子邮件内容的健壮性和有效性。我们开发了一条管道来测试这个漏洞。我们的渠道使用GPT-3.5 Turbo修改垃圾电子邮件，并评估SpamAssassin对这些修改后的电子邮件进行正确分类的能力。结果表明，Spamassassin将高达73.7%的LLM修改后的垃圾邮件错误分类为合法邮件。相比之下，更简单的词典替换攻击的最高成功率仅为0.4%。这些发现突显了LLM修改的垃圾邮件构成的重大威胁，特别是考虑到此类攻击的成本效益(每封电子邮件0.17美分)。本文对当前垃圾邮件过滤器的漏洞以及网络安全措施持续改进的必要性提供了至关重要的见解。



## **37. Beyond Detection: Leveraging Large Language Models for Cyber Attack Prediction in IoT Networks**

超越检测：利用大型语言模型进行物联网网络中的网络攻击预测 cs.CR

**SubmitDate**: 2024-08-26    [abs](http://arxiv.org/abs/2408.14045v1) [paper-pdf](http://arxiv.org/pdf/2408.14045v1)

**Authors**: Alaeddine Diaf, Abdelaziz Amara Korba, Nour Elislem Karabadji, Yacine Ghamri-Doudane

**Abstract**: In recent years, numerous large-scale cyberattacks have exploited Internet of Things (IoT) devices, a phenomenon that is expected to escalate with the continuing proliferation of IoT technology. Despite considerable efforts in attack detection, intrusion detection systems remain mostly reactive, responding to specific patterns or observed anomalies. This work proposes a proactive approach to anticipate and mitigate malicious activities before they cause damage. This paper proposes a novel network intrusion prediction framework that combines Large Language Models (LLMs) with Long Short Term Memory (LSTM) networks. The framework incorporates two LLMs in a feedback loop: a fine-tuned Generative Pre-trained Transformer (GPT) model for predicting network traffic and a fine-tuned Bidirectional Encoder Representations from Transformers (BERT) for evaluating the predicted traffic. The LSTM classifier model then identifies malicious packets among these predictions. Our framework, evaluated on the CICIoT2023 IoT attack dataset, demonstrates a significant improvement in predictive capabilities, achieving an overall accuracy of 98%, offering a robust solution to IoT cybersecurity challenges.

摘要: 近年来，大量大规模网络攻击利用了物联网(IoT)设备，预计随着物联网技术的持续扩散，这一现象将升级。尽管在攻击检测方面做出了相当大的努力，入侵检测系统仍然主要是被动的，对特定的模式或观察到的异常做出反应。这项工作提出了一种主动的方法，在恶意活动造成破坏之前对其进行预测和缓解。提出了一种将大语言模型和长短期记忆网络相结合的网络入侵预测框架。该框架在反馈环路中结合了两个LLM：用于预测网络流量的微调生成式预训练变压器(GPT)模型和用于评估预测流量的来自变压器的微调双向编码器表示(BERT)。然后，LSTM分类器模型在这些预测中识别恶意数据包。我们的框架在CICIoT2023物联网攻击数据集上进行了评估，显示出预测能力的显著改进，总体准确率达到98%，为物联网网络安全挑战提供了强大的解决方案。



## **38. Probing the Safety Response Boundary of Large Language Models via Unsafe Decoding Path Generation**

通过不安全解码路径生成探索大型语言模型的安全响应边界 cs.CR

**SubmitDate**: 2024-08-26    [abs](http://arxiv.org/abs/2408.10668v3) [paper-pdf](http://arxiv.org/pdf/2408.10668v3)

**Authors**: Haoyu Wang, Bingzhe Wu, Yatao Bian, Yongzhe Chang, Xueqian Wang, Peilin Zhao

**Abstract**: Large Language Models (LLMs) are implicit troublemakers. While they provide valuable insights and assist in problem-solving, they can also potentially serve as a resource for malicious activities. Implementing safety alignment could mitigate the risk of LLMs generating harmful responses. We argue that: even when an LLM appears to successfully block harmful queries, there may still be hidden vulnerabilities that could act as ticking time bombs. To identify these underlying weaknesses, we propose to use a cost value model as both a detector and an attacker. Trained on external or self-generated harmful datasets, the cost value model could successfully influence the original safe LLM to output toxic content in decoding process. For instance, LLaMA-2-chat 7B outputs 39.18% concrete toxic content, along with only 22.16% refusals without any harmful suffixes. These potential weaknesses can then be exploited via prompt optimization such as soft prompts on images. We name this decoding strategy: Jailbreak Value Decoding (JVD), emphasizing that seemingly secure LLMs may not be as safe as we initially believe. They could be used to gather harmful data or launch covert attacks.

摘要: 大型语言模型(LLM)是隐含的麻烦制造者。虽然它们提供了有价值的见解并帮助解决问题，但它们也可能成为恶意活动的来源。实施安全调整可以降低低密度脂蛋白产生有害反应的风险。我们认为：即使LLM似乎成功阻止了有害查询，仍可能存在隐藏的漏洞，这些漏洞可能会充当定时炸弹。为了识别这些潜在的弱点，我们建议使用成本价值模型作为检测器和攻击者。代价值模型在外部或自身产生的有害数据集上进行训练，可以成功地影响原始安全LLM在解码过程中输出有毒内容。例如，骆驼-2-Chat 7B输出39.18%的具体有毒内容，以及只有22.16%的拒绝没有任何有害的后缀。然后可以通过提示优化(如图像上的软提示)来利用这些潜在的弱点。我们将这种解码策略命名为：越狱价值解码(JVD)，强调看似安全的LLM可能并不像我们最初认为的那样安全。它们可能被用来收集有害数据或发动秘密攻击。



## **39. Large Language Models as Carriers of Hidden Messages**

大型语言模型作为隐藏消息的载体 cs.CL

Work in progress. Code is available at  https://github.com/j-hoscilowic/zurek-stegano

**SubmitDate**: 2024-08-25    [abs](http://arxiv.org/abs/2406.02481v3) [paper-pdf](http://arxiv.org/pdf/2406.02481v3)

**Authors**: Jakub Hoscilowicz, Pawel Popiolek, Jan Rudkowski, Jedrzej Bieniasz, Artur Janicki

**Abstract**: With the help of simple fine-tuning, one can artificially embed hidden text into large language models (LLMs). This text is revealed only when triggered by a specific query to the LLM. Two primary applications are LLM fingerprinting and steganography. In the context of LLM fingerprinting, a unique text identifier (fingerprint) is embedded within the model to verify licensing compliance. In the context of steganography, the LLM serves as a carrier for hidden messages that can be disclosed through a chosen trigger question.   Our work demonstrates that embedding hidden text in the LLM via fine-tuning, though seemingly secure due to the vast number of potential triggers (any sequence of characters or tokens could serve as a trigger), is susceptible to extraction through analysis of the LLM's output decoding process. We propose an extraction attack called Unconditional Token Forcing (UTF). It is premised on the hypothesis that iteratively feeding each token from the LLM's vocabulary into the model should reveal output sequences with abnormally high token probabilities, indicating potential hidden text candidates. We also present a defense method to hide text in such a way that it is resistant to both UTF and attacks based on sampling decoding methods, which we named Unconditional Token Forcing Confusion (UTFC). To the best of our knowledge, there is no attack method that can extract text hidden with UTFC. UTFC has both benign applications (improving LLM fingerprinting) and malign applications (using LLMs to create covert communication channels).

摘要: 在简单微调的帮助下，人们可以人为地将隐藏文本嵌入到大型语言模型(LLM)中。只有在对LLM的特定查询触发时，才会显示此文本。两个主要应用是LLM指纹识别和隐写。在LLM指纹识别的上下文中，唯一的文本识别符(指纹)被嵌入到模型中，以验证许可合规性。在隐写术的背景下，LLM充当了隐藏消息的载体，这些隐藏消息可以通过选择的触发问题来泄露。我们的工作表明，通过微调将隐藏文本嵌入到LLM中，尽管由于潜在触发器(任何字符或标记序列都可以作为触发器)的数量巨大而看起来是安全的，但通过分析LLM的输出解码过程，它容易被提取。我们提出了一种称为无条件令牌强迫(UTF)的提取攻击。它的前提是假设迭代地将LLM词汇中的每个标记输入到模型中，应该会揭示出具有异常高的标记概率的输出序列，这表明潜在的隐藏文本候选。我们还提出了一种基于抽样解码的文本隐藏方法，称为无条件令牌强制混淆(UTFC)，使其能够同时抵抗UTF和攻击。就我们所知，没有一种攻击方法可以提取使用UTFC隐藏的文本。UTFC既有良性应用程序(改进LLM指纹识别)，也有恶意应用程序(使用LLM创建秘密通信通道)。



## **40. Optimization-based Prompt Injection Attack to LLM-as-a-Judge**

对LLM as-a-Judge的基于优化的即时注入攻击 cs.CR

To appear in the Proceedings of The ACM Conference on Computer and  Communications Security (CCS), 2024

**SubmitDate**: 2024-08-24    [abs](http://arxiv.org/abs/2403.17710v2) [paper-pdf](http://arxiv.org/pdf/2403.17710v2)

**Authors**: Jiawen Shi, Zenghui Yuan, Yinuo Liu, Yue Huang, Pan Zhou, Lichao Sun, Neil Zhenqiang Gong

**Abstract**: LLM-as-a-Judge uses a large language model (LLM) to select the best response from a set of candidates for a given question. LLM-as-a-Judge has many applications such as LLM-powered search, reinforcement learning with AI feedback (RLAIF), and tool selection. In this work, we propose JudgeDeceiver, an optimization-based prompt injection attack to LLM-as-a-Judge. JudgeDeceiver injects a carefully crafted sequence into an attacker-controlled candidate response such that LLM-as-a-Judge selects the candidate response for an attacker-chosen question no matter what other candidate responses are. Specifically, we formulate finding such sequence as an optimization problem and propose a gradient based method to approximately solve it. Our extensive evaluation shows that JudgeDeceive is highly effective, and is much more effective than existing prompt injection attacks that manually craft the injected sequences and jailbreak attacks when extended to our problem. We also show the effectiveness of JudgeDeceiver in three case studies, i.e., LLM-powered search, RLAIF, and tool selection. Moreover, we consider defenses including known-answer detection, perplexity detection, and perplexity windowed detection. Our results show these defenses are insufficient, highlighting the urgent need for developing new defense strategies.

摘要: LLM-as-a-Court使用大型语言模型(LLM)从给定问题的一组候选人中选择最佳答案。LLM-as-a-Court有许多应用，如LLM支持的搜索、带人工智能反馈的强化学习(RLAIF)和工具选择。在这项工作中，我们提出了一种针对LLM-as-a-Court的基于优化的快速注入攻击--JudgeDeceiver。JudgeDeceiver将精心制作的序列注入到攻击者控制的候选响应中，以便LLM-as-a-Court为攻击者选择的问题选择候选响应，而不管其他候选响应是什么。具体地说，我们将寻找这样的序列描述为一个优化问题，并提出了一种基于梯度的方法来近似求解它。我们的广泛评估表明，JudgeDecept是非常有效的，并且比现有的手动手工创建注入序列的即时注入攻击和越狱攻击更有效，当扩展到我们的问题时。我们还在三个案例研究中展示了JudgeDeceiver的有效性，即LLM支持的搜索、RLAIF和工具选择。此外，我们还考虑了防御措施，包括已知答案检测、困惑检测和困惑加窗检测。我们的结果表明，这些防御措施是不够的，这突显了开发新的防御战略的迫切需要。



## **41. Safeguarding Vision-Language Models Against Patched Visual Prompt Injectors**

保护视觉语言模型免受修补视觉提示注入器的影响 cs.CV

15 pages

**SubmitDate**: 2024-08-24    [abs](http://arxiv.org/abs/2405.10529v2) [paper-pdf](http://arxiv.org/pdf/2405.10529v2)

**Authors**: Jiachen Sun, Changsheng Wang, Jiongxiao Wang, Yiwei Zhang, Chaowei Xiao

**Abstract**: Large language models have become increasingly prominent, also signaling a shift towards multimodality as the next frontier in artificial intelligence, where their embeddings are harnessed as prompts to generate textual content. Vision-language models (VLMs) stand at the forefront of this advancement, offering innovative ways to combine visual and textual data for enhanced understanding and interaction. However, this integration also enlarges the attack surface. Patch-based adversarial attack is considered the most realistic threat model in physical vision applications, as demonstrated in many existing literature. In this paper, we propose to address patched visual prompt injection, where adversaries exploit adversarial patches to generate target content in VLMs. Our investigation reveals that patched adversarial prompts exhibit sensitivity to pixel-wise randomization, a trait that remains robust even against adaptive attacks designed to counteract such defenses. Leveraging this insight, we introduce SmoothVLM, a defense mechanism rooted in smoothing techniques, specifically tailored to protect VLMs from the threat of patched visual prompt injectors. Our framework significantly lowers the attack success rate to a range between 0% and 5.0% on two leading VLMs, while achieving around 67.3% to 95.0% context recovery of the benign images, demonstrating a balance between security and usability.

摘要: 大型语言模型已变得越来越突出，这也标志着向多通道的转变，成为人工智能的下一个前沿，在人工智能中，它们的嵌入被用作生成文本内容的提示。视觉语言模型(VLM)站在这一进步的前沿，提供了将视觉和文本数据相结合的创新方法，以增强理解和交互。然而，这种整合也扩大了攻击面。基于补丁的对抗性攻击被认为是物理视觉应用中最现实的威胁模型，许多现有的文献都证明了这一点。在本文中，我们建议解决补丁视觉提示注入，即攻击者利用敌意补丁来生成VLMS中的目标内容。我们的调查显示，打补丁的对抗性提示显示出对像素随机化的敏感性，这一特征即使在旨在对抗此类防御的适应性攻击中也保持健壮。利用这一见解，我们推出了SmoothVLM，这是一种植根于平滑技术的防御机制，专门为保护VLM免受修补的视觉提示注入器的威胁而量身定做。我们的框架将攻击成功率显著降低到了0%到5.0%之间，同时实现了良性映像的67.3%到95.0%的上下文恢复，展示了安全性和可用性之间的平衡。



## **42. Probing the Robustness of Vision-Language Pretrained Models: A Multimodal Adversarial Attack Approach**

探索视觉语言预训练模型的鲁棒性：多模式对抗攻击方法 cs.CV

**SubmitDate**: 2024-08-24    [abs](http://arxiv.org/abs/2408.13461v1) [paper-pdf](http://arxiv.org/pdf/2408.13461v1)

**Authors**: Jiwei Guan, Tianyu Ding, Longbing Cao, Lei Pan, Chen Wang, Xi Zheng

**Abstract**: Vision-language pretraining (VLP) with transformers has demonstrated exceptional performance across numerous multimodal tasks. However, the adversarial robustness of these models has not been thoroughly investigated. Existing multimodal attack methods have largely overlooked cross-modal interactions between visual and textual modalities, particularly in the context of cross-attention mechanisms. In this paper, we study the adversarial vulnerability of recent VLP transformers and design a novel Joint Multimodal Transformer Feature Attack (JMTFA) that concurrently introduces adversarial perturbations in both visual and textual modalities under white-box settings. JMTFA strategically targets attention relevance scores to disrupt important features within each modality, generating adversarial samples by fusing perturbations and leading to erroneous model predictions. Experimental results indicate that the proposed approach achieves high attack success rates on vision-language understanding and reasoning downstream tasks compared to existing baselines. Notably, our findings reveal that the textual modality significantly influences the complex fusion processes within VLP transformers. Moreover, we observe no apparent relationship between model size and adversarial robustness under our proposed attacks. These insights emphasize a new dimension of adversarial robustness and underscore potential risks in the reliable deployment of multimodal AI systems.

摘要: 使用变压器的视觉语言预培训(VLP)在许多多模式任务中表现出了出色的性能。然而，这些模型的对抗稳健性还没有得到彻底的研究。现有的多通道攻击方法在很大程度上忽略了视觉通道和文本通道之间的跨通道交互作用，特别是在交叉注意机制的背景下。本文研究了现有VLP变换的对抗性漏洞，设计了一种在白盒环境下同时引入对抗性扰动的联合多模式变换特征攻击(JMTFA)。JMTFA战略性地将注意力相关性分数作为目标，以扰乱每个通道中的重要特征，通过融合扰动生成对抗性样本，并导致错误的模型预测。实验结果表明，与现有的基线相比，该方法在视觉语言理解和推理的下游任务上获得了更高的攻击成功率。值得注意的是，我们的研究结果显示，语篇情态显著影响VLP转换器内复杂的融合过程。此外，在我们提出的攻击下，我们没有观察到模型大小和对手稳健性之间的明显关系。这些见解强调了对抗性稳健性的一个新维度，并强调了可靠部署多模式人工智能系统的潜在风险。



## **43. Trading Devil Final: Backdoor attack via Stock market and Bayesian Optimization**

交易魔鬼决赛：通过股市和Bayesian优化进行后门攻击 cs.LG

END (will never be modified again!!) :Jumps-Diffusion and stock  market: Better quantify uncertainty in financial simulations

**SubmitDate**: 2024-08-24    [abs](http://arxiv.org/abs/2407.14573v4) [paper-pdf](http://arxiv.org/pdf/2407.14573v4)

**Authors**: Orson Mengara

**Abstract**: Since the advent of generative artificial intelligence, every company and researcher has been rushing to develop their own generative models, whether commercial or not. Given the large number of users of these powerful new tools, there is currently no intrinsically verifiable way to explain from the ground up what happens when LLMs (large language models) learn. For example, those based on automatic speech recognition systems, which have to rely on huge and astronomical amounts of data collected from all over the web to produce fast and efficient results, In this article, we develop a backdoor attack called MarketBackFinal 2.0, based on acoustic data poisoning, MarketBackFinal 2.0 is mainly based on modern stock market models. In order to show the possible vulnerabilities of speech-based transformers that may rely on LLMs.

摘要: 自生成人工智能出现以来，每家公司和研究人员都在争先恐后地开发自己的生成模型，无论是否商业化。鉴于这些强大的新工具的大量用户，目前还没有本质上可验证的方法来从头解释LLM（大型语言模型）学习时会发生什么。例如，那些基于自动语音识别系统的系统，它们必须依赖于从整个网络收集的大量数据来产生快速有效的结果，在本文中，我们开发了一种名为MarketBackFinal 2.0的后门攻击，基于声学数据中毒，MarketBackFinal 2.0主要基于现代股市模型。为了显示可能依赖LLM的基于语音的转换器可能存在的漏洞。



## **44. Is Generative AI the Next Tactical Cyber Weapon For Threat Actors? Unforeseen Implications of AI Generated Cyber Attacks**

生成性人工智能是威胁行为者的下一个战术网络武器吗？人工智能引发的网络攻击的不可预见影响 cs.CR

Journal Paper

**SubmitDate**: 2024-08-23    [abs](http://arxiv.org/abs/2408.12806v1) [paper-pdf](http://arxiv.org/pdf/2408.12806v1)

**Authors**: Yusuf Usman, Aadesh Upadhyay, Prashnna Gyawali, Robin Chataut

**Abstract**: In an era where digital threats are increasingly sophisticated, the intersection of Artificial Intelligence and cybersecurity presents both promising defenses and potent dangers. This paper delves into the escalating threat posed by the misuse of AI, specifically through the use of Large Language Models (LLMs). This study details various techniques like the switch method and character play method, which can be exploited by cybercriminals to generate and automate cyber attacks. Through a series of controlled experiments, the paper demonstrates how these models can be manipulated to bypass ethical and privacy safeguards to effectively generate cyber attacks such as social engineering, malicious code, payload generation, and spyware. By testing these AI generated attacks on live systems, the study assesses their effectiveness and the vulnerabilities they exploit, offering a practical perspective on the risks AI poses to critical infrastructure. We also introduce Occupy AI, a customized, finetuned LLM specifically engineered to automate and execute cyberattacks. This specialized AI driven tool is adept at crafting steps and generating executable code for a variety of cyber threats, including phishing, malware injection, and system exploitation. The results underscore the urgency for ethical AI practices, robust cybersecurity measures, and regulatory oversight to mitigate AI related threats. This paper aims to elevate awareness within the cybersecurity community about the evolving digital threat landscape, advocating for proactive defense strategies and responsible AI development to protect against emerging cyber threats.

摘要: 在一个数字威胁日益复杂的时代，人工智能和网络安全的交集既带来了有希望的防御，也带来了潜在的危险。本文深入研究了滥用人工智能带来的不断升级的威胁，特别是通过使用大型语言模型(LLM)。这项研究详细介绍了各种技术，如切换方法和角色扮演方法，网络犯罪分子可以利用这些技术来生成网络攻击并使其自动化。通过一系列受控实验，本文演示了如何操纵这些模型以绕过伦理和隐私保护措施，从而有效地生成网络攻击，如社会工程、恶意代码、有效负载生成和间谍软件。通过测试这些人工智能对实时系统的攻击，该研究评估了它们的有效性和它们利用的漏洞，为人工智能对关键基础设施构成的风险提供了一个实用的视角。我们还推出了占领AI，这是一款定制的、经过精细调整的LLM，专门设计用于自动化和执行网络攻击。这个专门的人工智能驱动工具擅长为各种网络威胁制作步骤和生成可执行代码，包括网络钓鱼、恶意软件注入和系统利用。这一结果突显了伦理人工智能做法、强有力的网络安全措施和监管监督的紧迫性，以缓解与人工智能相关的威胁。本文旨在提高网络安全社区对不断发展的数字威胁格局的认识，倡导积极主动的防御战略和负责任的人工智能发展，以防御新出现的网络威胁。



## **45. BackdoorLLM: A Comprehensive Benchmark for Backdoor Attacks on Large Language Models**

BackdoorLLM：大型语言模型后门攻击的综合基准 cs.AI

**SubmitDate**: 2024-08-23    [abs](http://arxiv.org/abs/2408.12798v1) [paper-pdf](http://arxiv.org/pdf/2408.12798v1)

**Authors**: Yige Li, Hanxun Huang, Yunhan Zhao, Xingjun Ma, Jun Sun

**Abstract**: Generative Large Language Models (LLMs) have made significant strides across various tasks, but they remain vulnerable to backdoor attacks, where specific triggers in the prompt cause the LLM to generate adversary-desired responses. While most backdoor research has focused on vision or text classification tasks, backdoor attacks in text generation have been largely overlooked. In this work, we introduce \textit{BackdoorLLM}, the first comprehensive benchmark for studying backdoor attacks on LLMs. \textit{BackdoorLLM} features: 1) a repository of backdoor benchmarks with a standardized training pipeline, 2) diverse attack strategies, including data poisoning, weight poisoning, hidden state attacks, and chain-of-thought attacks, 3) extensive evaluations with over 200 experiments on 8 attacks across 7 scenarios and 6 model architectures, and 4) key insights into the effectiveness and limitations of backdoors in LLMs. We hope \textit{BackdoorLLM} will raise awareness of backdoor threats and contribute to advancing AI safety. The code is available at \url{https://github.com/bboylyg/BackdoorLLM}.

摘要: 生成性大型语言模型(LLM)已经在各种任务中取得了重大进展，但它们仍然容易受到后门攻击，在后门攻击中，提示中的特定触发器会导致LLM生成对手想要的响应。虽然大多数后门研究都集中在视觉或文本分类任务上，但文本生成中的后门攻击在很大程度上被忽视了。在这项工作中，我们介绍了第一个用于研究对LLM的后门攻击的全面基准测试。\textit{Backdoor LLM}的特点是：1)具有标准化培训管道的后门基准存储库；2)多样化的攻击策略，包括数据中毒、重量中毒、隐藏状态攻击和思想链攻击；3)对7个场景和6个模型架构中的8个攻击进行了200多个实验的广泛评估；4)对LLMS中后门的有效性和局限性的关键洞察。我们希望\textit{Backdoor LLM}将提高人们对后门威胁的认识，并为推进人工智能安全做出贡献。代码可在\url{https://github.com/bboylyg/BackdoorLLM}.



## **46. Can Large Language Models Automatically Jailbreak GPT-4V?**

大型语言模型可以自动越狱GPT-4V吗？ cs.CL

TrustNLP@NAACL2024 (Fourth Workshop on Trustworthy Natural Language  Processing)

**SubmitDate**: 2024-08-23    [abs](http://arxiv.org/abs/2407.16686v2) [paper-pdf](http://arxiv.org/pdf/2407.16686v2)

**Authors**: Yuanwei Wu, Yue Huang, Yixin Liu, Xiang Li, Pan Zhou, Lichao Sun

**Abstract**: GPT-4V has attracted considerable attention due to its extraordinary capacity for integrating and processing multimodal information. At the same time, its ability of face recognition raises new safety concerns of privacy leakage. Despite researchers' efforts in safety alignment through RLHF or preprocessing filters, vulnerabilities might still be exploited. In our study, we introduce AutoJailbreak, an innovative automatic jailbreak technique inspired by prompt optimization. We leverage Large Language Models (LLMs) for red-teaming to refine the jailbreak prompt and employ weak-to-strong in-context learning prompts to boost efficiency. Furthermore, we present an effective search method that incorporates early stopping to minimize optimization time and token expenditure. Our experiments demonstrate that AutoJailbreak significantly surpasses conventional methods, achieving an Attack Success Rate (ASR) exceeding 95.3\%. This research sheds light on strengthening GPT-4V security, underscoring the potential for LLMs to be exploited in compromising GPT-4V integrity.

摘要: GPT-4V由于其综合和处理多模式信息的非凡能力而引起了相当大的关注。与此同时，它的人脸识别能力引发了新的隐私泄露的安全担忧。尽管研究人员通过RLHF或预处理过滤器在安全匹配方面做出了努力，但漏洞仍有可能被利用。在我们的研究中，我们介绍了AutoJailBreak，这是一种受即时优化启发的创新的自动越狱技术。我们利用用于红色团队的大型语言模型(LLM)来改进越狱提示，并采用从弱到强的上下文学习提示来提高效率。此外，我们还提出了一种结合提前停止的有效搜索方法，以最小化优化时间和令牌开销。实验表明，AutoJailBreak的攻击成功率(ASR)超过95.3%，明显优于传统方法。这项研究有助于加强GPT-4V的安全性，强调了LLMS在危害GPT-4V完整性方面的潜力。



## **47. Prefix Guidance: A Steering Wheel for Large Language Models to Defend Against Jailbreak Attacks**

前置指导：大型语言模型防御越狱攻击的方向盘 cs.CR

**SubmitDate**: 2024-08-22    [abs](http://arxiv.org/abs/2408.08924v2) [paper-pdf](http://arxiv.org/pdf/2408.08924v2)

**Authors**: Jiawei Zhao, Kejiang Chen, Xiaojian Yuan, Weiming Zhang

**Abstract**: In recent years, the rapid development of large language models (LLMs) has achieved remarkable performance across various tasks. However, research indicates that LLMs are vulnerable to jailbreak attacks, where adversaries can induce the generation of harmful content through meticulously crafted prompts. This vulnerability poses significant challenges to the secure use and promotion of LLMs. Existing defense methods offer protection from different perspectives but often suffer from insufficient effectiveness or a significant impact on the model's capabilities. In this paper, we propose a plug-and-play and easy-to-deploy jailbreak defense framework, namely Prefix Guidance (PG), which guides the model to identify harmful prompts by directly setting the first few tokens of the model's output. This approach combines the model's inherent security capabilities with an external classifier to defend against jailbreak attacks. We demonstrate the effectiveness of PG across three models and five attack methods. Compared to baselines, our approach is generally more effective on average. Additionally, results on the Just-Eval benchmark further confirm PG's superiority to preserve the model's performance. our code is available at https://github.com/weiyezhimeng/Prefix-Guidance.

摘要: 近年来，大型语言模型的快速发展在各种任务中取得了显著的性能。然而，研究表明，LLMS容易受到越狱攻击，在越狱攻击中，攻击者可以通过精心制作的提示来诱导生成有害内容。此漏洞对安全使用和推广LLMS构成重大挑战。现有的防御方法从不同的角度提供保护，但往往存在有效性不足或对模型能力产生重大影响的问题。本文提出了一种即插即用、易于部署的越狱防御框架--前缀引导(PG)，它通过直接设置模型输出的前几个令牌来引导模型识别有害提示。这种方法将模型固有的安全功能与外部分类器相结合，以防御越狱攻击。我们在三个模型和五种攻击方法上演示了PG的有效性。与基线相比，我们的方法总体上更有效。此外，Just-Eval基准测试的结果进一步证实了PG在保持模型性能方面的优势。我们的代码可以在https://github.com/weiyezhimeng/Prefix-Guidance.上找到



## **48. Vaccine: Perturbation-aware Alignment for Large Language Models against Harmful Fine-tuning**

疫苗：大型语言模型的扰动感知对齐，防止有害的微调 cs.LG

**SubmitDate**: 2024-08-22    [abs](http://arxiv.org/abs/2402.01109v4) [paper-pdf](http://arxiv.org/pdf/2402.01109v4)

**Authors**: Tiansheng Huang, Sihao Hu, Ling Liu

**Abstract**: The new paradigm of finetuning-as-a-service introduces a new attack surface for Large Language Models (LLMs): a few harmful data uploaded by users can easily trick the finetuning to produce an alignment-broken model. We conduct an empirical analysis and uncover a \textit{harmful embedding drift} phenomenon, showing a probable cause of the alignment-broken effect. Inspired by our findings, we propose Vaccine, a perturbation-aware alignment technique to mitigate the security risk of users finetuning. The core idea of Vaccine is to produce invariant hidden embeddings by progressively adding crafted perturbation to them in the alignment phase. This enables the embeddings to withstand harmful perturbation from un-sanitized user data in the finetuning phase. Our results on open source mainstream LLMs (e.g., Llama2, Opt, Vicuna) demonstrate that Vaccine can boost the robustness of alignment against harmful prompts induced embedding drift while reserving reasoning ability towards benign prompts. Our code is available at \url{https://github.com/git-disl/Vaccine}.

摘要: 精调即服务的新范式为大型语言模型(LLM)引入了一个新的攻击面：用户上传的少量有害数据就可以很容易地欺骗精调，产生一个破坏对齐的模型。我们进行了实证分析，发现了一种有害的嵌入漂移现象，揭示了排列断裂效应的可能原因。受我们发现的启发，我们提出了Vaccine，一种扰动感知的对齐技术，以降低用户精调的安全风险。Vaccine的核心思想是通过在比对阶段逐步向其添加精心制作的扰动来产生不变的隐藏嵌入。这使嵌入能够在精细调整阶段抵御来自未清理的用户数据的有害干扰。我们在开源主流LLMS(如Llama2、Opt、Vicuna)上的实验结果表明，疫苗可以提高对有害提示导致的嵌入漂移的健壮性，同时保留对良性提示的推理能力。我们的代码可在\url{https://github.com/git-disl/Vaccine}.



## **49. Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs**

隐性对抗培训提高了法学硕士对持续有害行为的稳健性 cs.LG

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2407.15549v2) [paper-pdf](http://arxiv.org/pdf/2407.15549v2)

**Authors**: Abhay Sheshadri, Aidan Ewart, Phillip Guo, Aengus Lynch, Cindy Wu, Vivek Hebbar, Henry Sleight, Asa Cooper Stickland, Ethan Perez, Dylan Hadfield-Menell, Stephen Casper

**Abstract**: Large language models (LLMs) can often be made to behave in undesirable ways that they are explicitly fine-tuned not to. For example, the LLM red-teaming literature has produced a wide variety of 'jailbreaking' techniques to elicit harmful text from models that were fine-tuned to be harmless. Recent work on red-teaming, model editing, and interpretability suggests that this challenge stems from how (adversarial) fine-tuning largely serves to suppress rather than remove undesirable capabilities from LLMs. Prior work has introduced latent adversarial training (LAT) as a way to improve robustness to broad classes of failures. These prior works have considered untargeted latent space attacks where the adversary perturbs latent activations to maximize loss on examples of desirable behavior. Untargeted LAT can provide a generic type of robustness but does not leverage information about specific failure modes. Here, we experiment with targeted LAT where the adversary seeks to minimize loss on a specific competing task. We find that it can augment a wide variety of state-of-the-art methods. First, we use targeted LAT to improve robustness to jailbreaks, outperforming a strong R2D2 baseline with orders of magnitude less compute. Second, we use it to more effectively remove backdoors with no knowledge of the trigger. Finally, we use it to more effectively unlearn knowledge for specific undesirable tasks in a way that is also more robust to re-learning. Overall, our results suggest that targeted LAT can be an effective tool for defending against harmful behaviors from LLMs.

摘要: 大型语言模型(LLM)通常会以不受欢迎的方式运行，因此它们被明确微调为不以这种方式运行。例如，LLM的红队文献已经创造了各种各样的“越狱”技术，从经过微调的无害的模特那里引出有害文本。最近在红团队、模型编辑和可解释性方面的工作表明，这一挑战源于(对抗性的)微调如何在很大程度上抑制而不是消除LLM中不受欢迎的能力。以前的工作已经引入了潜在的对手训练(LAT)，作为一种提高对广泛类别的故障的稳健性的方式。这些先前的工作考虑了无目标的潜在空间攻击，即对手扰乱潜在激活，以最大限度地减少期望行为的示例损失。非定向LAT可以提供一般类型的健壮性，但不利用有关特定故障模式的信息。在这里，我们实验有针对性的LAT，其中对手试图将特定竞争任务的损失降至最低。我们发现，它可以增加各种最先进的方法。首先，我们使用有针对性的LAT来提高对越狱的健壮性，性能优于强大的R2D2基线，计算量少了几个数量级。其次，我们使用它来更有效地删除后门，而不知道触发器。最后，我们使用它来更有效地忘记特定不受欢迎的任务的知识，这种方式也更适合重新学习。总体而言，我们的结果表明，有针对性的LAT可以成为防御LLM有害行为的有效工具。



## **50. A Study of Backdoors in Instruction Fine-tuned Language Models**

微调语言模型教学中的后门研究 cs.CR

Under review

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2406.07778v2) [paper-pdf](http://arxiv.org/pdf/2406.07778v2)

**Authors**: Jayaram Raghuram, George Kesidis, David J. Miller

**Abstract**: Backdoor data poisoning, inserted within instruction examples used to fine-tune a foundation Large Language Model (LLM) for downstream tasks (\textit{e.g.,} sentiment prediction), is a serious security concern due to the evasive nature of such attacks. The poisoning is usually in the form of a (seemingly innocuous) trigger word or phrase inserted into a very small fraction of the fine-tuning samples from a target class. Such backdoor attacks can: alter response sentiment, violate censorship, over-refuse (invoke censorship for legitimate queries), inject false content, or trigger nonsense responses (hallucinations). In this work we investigate the efficacy of instruction fine-tuning backdoor attacks as attack "hyperparameters" are varied under a variety of scenarios, considering: the trigger location in the poisoned examples; robustness to change in the trigger location, partial triggers, and synonym substitutions at test time; attack transfer from one (fine-tuning) domain to a related test domain; and clean-label vs. dirty-label poisoning. Based on our observations, we propose and evaluate two defenses against these attacks: i) a \textit{during-fine-tuning defense} based on word-frequency counts that assumes the (possibly poisoned) fine-tuning dataset is available and identifies the backdoor trigger tokens; and ii) a \textit{post-fine-tuning defense} based on downstream clean fine-tuning of the backdoored LLM with a small defense dataset. Finally, we provide a brief survey of related work on backdoor attacks and defenses.

摘要: 由于此类攻击的规避性质，后门数据中毒是一个严重的安全问题，它被插入到用于微调下游任务的基础大型语言模型(LLM)的指令示例中(例如，情感预测)。中毒通常以(看似无害的)触发词或短语的形式插入到来自目标类的微调样本的非常小的一部分中。这种后门攻击可以：改变回应情绪、违反审查制度、过度拒绝(对合法查询调用审查制度)、注入虚假内容或引发无稽之谈的反应(幻觉)。在这项工作中，我们研究了指令微调后门攻击的有效性，因为攻击“超参数”在各种场景下是不同的，考虑到：中毒示例中的触发器位置；对测试时触发器位置、部分触发器和同义词替换的稳健性；攻击从一个(微调)域转移到相关测试域；以及干净标签与脏标签中毒。基于我们的观察，我们提出并评估了针对这些攻击的两种防御方案：i)基于词频计数的精调期间防御方案，该方案假定(可能有毒的)微调数据集可用，并识别后门触发令牌；以及ii)基于带有小防御数据集的后置LLM的下游干净微调的后门微调防御方案。最后，我们对后门攻击和防御的相关工作进行了简要的概述。



