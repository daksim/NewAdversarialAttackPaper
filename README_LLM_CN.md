# Latest Adversarial Attack Papers
**update at 2023-12-18 10:03:08**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. No-Skim: Towards Efficiency Robustness Evaluation on Skimming-based Language Models**

No-Skim：基于Skimming语言模型的有效鲁棒性评估 cs.CR

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2312.09494v1) [paper-pdf](http://arxiv.org/pdf/2312.09494v1)

**Authors**: Shengyao Zhang, Mi Zhang, Xudong Pan, Min Yang

**Abstract**: To reduce the computation cost and the energy consumption in large language models (LLM), skimming-based acceleration dynamically drops unimportant tokens of the input sequence progressively along layers of the LLM while preserving the tokens of semantic importance. However, our work for the first time reveals the acceleration may be vulnerable to Denial-of-Service (DoS) attacks. In this paper, we propose No-Skim, a general framework to help the owners of skimming-based LLM to understand and measure the robustness of their acceleration scheme. Specifically, our framework searches minimal and unnoticeable perturbations at character-level and token-level to generate adversarial inputs that sufficiently increase the remaining token ratio, thus increasing the computation cost and energy consumption. We systematically evaluate the vulnerability of the skimming acceleration in various LLM architectures including BERT and RoBERTa on the GLUE benchmark. In the worst case, the perturbation found by No-Skim substantially increases the running cost of LLM by over 145% on average. Moreover, No-Skim extends the evaluation framework to various scenarios, making the evaluation conductible with different level of knowledge.

摘要: 为了降低大型语言模型(LLM)的计算代价和能量消耗，基于略读的加速算法在保留语义重要的标记的同时，动态地沿着LLM的层次逐步丢弃输入序列中不重要的标记。然而，我们的工作首次揭示了加速可能容易受到拒绝服务(DoS)攻击。在本文中，我们提出了一个通用的框架No-Skim，以帮助基于略读的LLM的所有者理解和度量其加速方案的健壮性。具体地说，我们的框架在字符级和令牌级搜索最小和不可察觉的扰动，以生成足以增加剩余令牌率的对抗性输入，从而增加计算成本和能量消耗。我们在GLUE基准上系统地评估了包括Bert和Roberta在内的各种LLM架构中掠读加速的脆弱性。在最坏的情况下，No-Skim发现的扰动大大增加了LLM的运行成本，平均超过145%。此外，No-Skim将评估框架扩展到各种场景，使评估可以在不同的知识水平下进行。



## **2. AutoDAN: Interpretable Gradient-Based Adversarial Attacks on Large Language Models**

AutoDAN：对大型语言模型的可解释的基于梯度的对抗性攻击 cs.CR

Version 2 updates: Added comparison of three more evaluation methods  and their reliability check using human labeling. Added results for  jailbreaking Llama2 (individual behavior) and included complexity and  hyperparameter analysis. Revised objectives for prompt leaking. Other minor  changes made

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2310.15140v2) [paper-pdf](http://arxiv.org/pdf/2310.15140v2)

**Authors**: Sicheng Zhu, Ruiyi Zhang, Bang An, Gang Wu, Joe Barrow, Zichao Wang, Furong Huang, Ani Nenkova, Tong Sun

**Abstract**: Safety alignment of Large Language Models (LLMs) can be compromised with manual jailbreak attacks and (automatic) adversarial attacks. Recent studies suggest that defending against these attacks is possible: adversarial attacks generate unlimited but unreadable gibberish prompts, detectable by perplexity-based filters; manual jailbreak attacks craft readable prompts, but their limited number due to the necessity of human creativity allows for easy blocking. In this paper, we show that these solutions may be too optimistic. We introduce AutoDAN, an interpretable, gradient-based adversarial attack that merges the strengths of both attack types. Guided by the dual goals of jailbreak and readability, AutoDAN optimizes and generates tokens one by one from left to right, resulting in readable prompts that bypass perplexity filters while maintaining high attack success rates. Notably, these prompts, generated from scratch using gradients, are interpretable and diverse, with emerging strategies commonly seen in manual jailbreak attacks. They also generalize to unforeseen harmful behaviors and transfer to black-box LLMs better than their unreadable counterparts when using limited training data or a single proxy model. Furthermore, we show the versatility of AutoDAN by automatically leaking system prompts using a customized objective. Our work offers a new way to red-team LLMs and understand jailbreak mechanisms via interpretability.

摘要: 大型语言模型(LLM)的安全对齐可能会受到手动越狱攻击和(自动)对抗性攻击的影响。最近的研究表明，防御这些攻击是可能的：对抗性攻击生成无限但不可读的胡言乱语提示，可通过基于困惑的过滤器检测；手动越狱攻击创建可读的提示，但由于人类创造力的必要性，其数量有限，允许轻松阻止。在本文中，我们证明了这些解决方案可能过于乐观。我们介绍了AutoDAN，一种可解释的、基于梯度的对抗性攻击，它融合了这两种攻击类型的优点。在越狱和可读性双重目标的指导下，AutoDAN从左到右一个接一个地优化和生成令牌，产生可读的提示，绕过困惑过滤器，同时保持高攻击成功率。值得注意的是，这些使用渐变从零开始生成的提示是可解释的和多样化的，新出现的策略通常出现在手动越狱攻击中。当使用有限的训练数据或单一代理模型时，它们还概括到不可预见的有害行为，并比不可读的同行更好地转移到黑盒LLM。此外，我们通过使用定制目标自动泄漏系统提示来展示AutoDAN的多功能性。我们的工作为红色团队LLM提供了一种新的方法，并通过可解释性来理解越狱机制。



## **3. FigStep: Jailbreaking Large Vision-language Models via Typographic Visual Prompts**

图：通过排版视觉提示越狱的大型视觉语言模型 cs.CR

Technical Report

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2311.05608v2) [paper-pdf](http://arxiv.org/pdf/2311.05608v2)

**Authors**: Yichen Gong, Delong Ran, Jinyuan Liu, Conglei Wang, Tianshuo Cong, Anyu Wang, Sisi Duan, Xiaoyun Wang

**Abstract**: Ensuring the safety of artificial intelligence-generated content (AIGC) is a longstanding topic in the artificial intelligence (AI) community, and the safety concerns associated with Large Language Models (LLMs) have been widely investigated. Recently, large vision-language models (VLMs) represent an unprecedented revolution, as they are built upon LLMs but can incorporate additional modalities (e.g., images). However, the safety of VLMs lacks systematic evaluation, and there may be an overconfidence in the safety guarantees provided by their underlying LLMs. In this paper, to demonstrate that introducing additional modality modules leads to unforeseen AI safety issues, we propose FigStep, a straightforward yet effective jailbreaking algorithm against VLMs. Instead of feeding textual harmful instructions directly, FigStep converts the harmful content into images through typography to bypass the safety alignment within the textual module of the VLMs, inducing VLMs to output unsafe responses that violate common AI safety policies. In our evaluation, we manually review 46,500 model responses generated by 3 families of the promising open-source VLMs, i.e., LLaVA, MiniGPT4, and CogVLM (a total of 6 VLMs). The experimental results show that FigStep can achieve an average attack success rate of 82.50% on 500 harmful queries in 10 topics. Moreover, we demonstrate that the methodology of FigStep can even jailbreak GPT-4V, which already leverages an OCR detector to filter harmful queries. Above all, our work reveals that VLMs are vulnerable to jailbreaking attacks, which highlights the necessity of novel safety alignments between visual and textual modalities.

摘要: 确保人工智能生成内容(AIGC)的安全性是人工智能(AI)领域的一个长期话题，与大型语言模型(LLM)相关的安全问题已被广泛调查。最近，大型视觉语言模型(VLM)代表了一场前所未有的革命，因为它们建立在LLM之上，但可以结合其他形式(例如图像)。然而，超低层管理系统的安全性缺乏系统的评估，可能对其底层低层管理系统提供的安全保证过于自信。在本文中，为了证明引入额外的通道模块会导致不可预见的人工智能安全问题，我们提出了FigStep，一种针对VLM的简单而有效的越狱算法。FigStep没有直接提供文本有害指令，而是通过排版将有害内容转换为图像，以绕过VLM文本模块内的安全对齐，诱导VLM输出违反常见AI安全策略的不安全响应。在我们的评估中，我们手动审查了3个有前途的开源VLM家族，即LLaVA、MiniGPT4和CogVLM(总共6个VLM)生成的46,500个模型响应。实验结果表明，FigStep对10个主题500个有害查询的平均攻击成功率为82.50%。此外，我们还演示了FigStep的方法甚至可以越狱GPT-4V，它已经利用OCR检测器来过滤有害查询。最重要的是，我们的工作揭示了VLM容易受到越狱攻击，这突显了视觉和文本通道之间新的安全对齐的必要性。



## **4. Efficient Representation of the Activation Space in Deep Neural Networks**

深度神经网络中激活空间的有效表示 cs.LG

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.08143v1) [paper-pdf](http://arxiv.org/pdf/2312.08143v1)

**Authors**: Tanya Akumu, Celia Cintas, Girmaw Abebe Tadesse, Adebayo Oshingbesan, Skyler Speakman, Edward McFowland III

**Abstract**: The representations of the activation space of deep neural networks (DNNs) are widely utilized for tasks like natural language processing, anomaly detection and speech recognition. Due to the diverse nature of these tasks and the large size of DNNs, an efficient and task-independent representation of activations becomes crucial. Empirical p-values have been used to quantify the relative strength of an observed node activation compared to activations created by already-known inputs. Nonetheless, keeping raw data for these calculations increases memory resource consumption and raises privacy concerns. To this end, we propose a model-agnostic framework for creating representations of activations in DNNs using node-specific histograms to compute p-values of observed activations without retaining already-known inputs. Our proposed approach demonstrates promising potential when validated with multiple network architectures across various downstream tasks and compared with the kernel density estimates and brute-force empirical baselines. In addition, the framework reduces memory usage by 30% with up to 4 times faster p-value computing time while maintaining state of-the-art detection power in downstream tasks such as the detection of adversarial attacks and synthesized content. Moreover, as we do not persist raw data at inference time, we could potentially reduce susceptibility to attacks and privacy issues.

摘要: 深度神经网络(DNN)的激活空间表示被广泛用于自然语言处理、异常检测和语音识别等任务。由于这些任务的多样性和DNN的巨大规模，高效和独立于任务的激活表示变得至关重要。经验p值被用来量化与已知输入产生的激活相比，观察到的节点激活的相对强度。尽管如此，为这些计算保留原始数据会增加内存资源消耗，并引发隐私问题。为此，我们提出了一个与模型无关的框架，用于使用节点特定的直方图来创建DNN中的激活表示，以计算观察到的激活的p值，而不保留已知的输入。我们提出的方法在不同下游任务的多个网络架构上进行验证，并与核密度估计和蛮力经验基线进行比较，显示出良好的潜力。此外，该框架减少了30%的内存使用量，p值计算时间最多提高了4倍，同时在下游任务中保持了最先进的检测能力，例如检测对抗性攻击和合成内容。此外，由于我们不在推理时保留原始数据，因此我们可能会降低对攻击和隐私问题的易感性。



## **5. PromptBench: A Unified Library for Evaluation of Large Language Models**

PromptBitch：大型语言模型评估的统一库 cs.AI

An extension to PromptBench (arXiv:2306.04528) for unified evaluation  of LLMs using the same name; code: https://github.com/microsoft/promptbench

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.07910v1) [paper-pdf](http://arxiv.org/pdf/2312.07910v1)

**Authors**: Kaijie Zhu, Qinlin Zhao, Hao Chen, Jindong Wang, Xing Xie

**Abstract**: The evaluation of large language models (LLMs) is crucial to assess their performance and mitigate potential security risks. In this paper, we introduce PromptBench, a unified library to evaluate LLMs. It consists of several key components that are easily used and extended by researchers: prompt construction, prompt engineering, dataset and model loading, adversarial prompt attack, dynamic evaluation protocols, and analysis tools. PromptBench is designed to be an open, general, and flexible codebase for research purposes that can facilitate original study in creating new benchmarks, deploying downstream applications, and designing new evaluation protocols. The code is available at: https://github.com/microsoft/promptbench and will be continuously supported.

摘要: 大型语言模型(LLM)的评估对于评估其性能和降低潜在的安全风险至关重要。在本文中，我们介绍了一个用于评估LLMS的统一库PromptBitch.它由几个易于研究人员使用和扩展的关键组件组成：即时构建、即时工程、数据集和模型加载、对抗性即时攻击、动态评估协议和分析工具。PromptBitch是一个开放的、通用的、灵活的研究代码库，可以在创建新的基准、部署下游应用程序和设计新的评估协议方面促进原创研究。该代码可在https://github.com/microsoft/promptbench上获得，并将继续受到支持。



## **6. Causality Analysis for Evaluating the Security of Large Language Models**

大型语言模型安全性评估的因果分析 cs.AI

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.07876v1) [paper-pdf](http://arxiv.org/pdf/2312.07876v1)

**Authors**: Wei Zhao, Zhe Li, Jun Sun

**Abstract**: Large Language Models (LLMs) such as GPT and Llama2 are increasingly adopted in many safety-critical applications. Their security is thus essential. Even with considerable efforts spent on reinforcement learning from human feedback (RLHF), recent studies have shown that LLMs are still subject to attacks such as adversarial perturbation and Trojan attacks. Further research is thus needed to evaluate their security and/or understand the lack of it. In this work, we propose a framework for conducting light-weight causality-analysis of LLMs at the token, layer, and neuron level. We applied our framework to open-source LLMs such as Llama2 and Vicuna and had multiple interesting discoveries. Based on a layer-level causality analysis, we show that RLHF has the effect of overfitting a model to harmful prompts. It implies that such security can be easily overcome by `unusual' harmful prompts. As evidence, we propose an adversarial perturbation method that achieves 100\% attack success rate on the red-teaming tasks of the Trojan Detection Competition 2023. Furthermore, we show the existence of one mysterious neuron in both Llama2 and Vicuna that has an unreasonably high causal effect on the output. While we are uncertain on why such a neuron exists, we show that it is possible to conduct a ``Trojan'' attack targeting that particular neuron to completely cripple the LLM, i.e., we can generate transferable suffixes to prompts that frequently make the LLM produce meaningless responses.

摘要: 大型语言模型(LLM)，如GPT和Llama2，在许多安全关键型应用中越来越多地被采用。因此，他们的安全至关重要。即使在从人类反馈中强化学习(RLHF)方面花费了大量的努力，最近的研究表明LLMS仍然受到诸如对抗性扰动和特洛伊木马攻击的攻击。因此，需要进一步研究，以评估其安全性和/或了解其缺乏安全性。在这项工作中，我们提出了一个框架，用于在标记、层和神经元水平上进行LLMS的轻量级因果分析。我们将我们的框架应用于开源LLM，如Llama2和Vicuna，并有多个有趣的发现。基于层级因果关系分析，我们发现RLHF具有对有害提示的模型过度拟合的效果。这意味着这种安全很容易被“不寻常的”有害提示所克服。作为证据，我们提出了一种对抗性扰动方法，在2023年木马检测大赛的红队任务上达到了100%的攻击成功率。此外，我们证明了在Llama2和Vicuna2中都存在一个神秘的神经元，它对输出具有不合理的高因果效应。虽然我们不确定为什么会有这样的神经元存在，但我们证明了有可能进行针对该特定神经元的“特洛伊木马”攻击，以完全削弱LLM，即我们可以为提示生成可转移的后缀，这些后缀经常使LLM产生无意义的响应。



## **7. DeceptPrompt: Exploiting LLM-driven Code Generation via Adversarial Natural Language Instructions**

DeceptPrompt：通过对抗性自然语言指令利用LLM驱动的代码生成 cs.CR

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.04730v2) [paper-pdf](http://arxiv.org/pdf/2312.04730v2)

**Authors**: Fangzhou Wu, Xiaogeng Liu, Chaowei Xiao

**Abstract**: With the advancement of Large Language Models (LLMs), significant progress has been made in code generation, enabling LLMs to transform natural language into programming code. These Code LLMs have been widely accepted by massive users and organizations. However, a dangerous nature is hidden in the code, which is the existence of fatal vulnerabilities. While some LLM providers have attempted to address these issues by aligning with human guidance, these efforts fall short of making Code LLMs practical and robust. Without a deep understanding of the performance of the LLMs under the practical worst cases, it would be concerning to apply them to various real-world applications. In this paper, we answer the critical issue: Are existing Code LLMs immune to generating vulnerable code? If not, what is the possible maximum severity of this issue in practical deployment scenarios? In this paper, we introduce DeceptPrompt, a novel algorithm that can generate adversarial natural language instructions that drive the Code LLMs to generate functionality correct code with vulnerabilities. DeceptPrompt is achieved through a systematic evolution-based algorithm with a fine grain loss design. The unique advantage of DeceptPrompt enables us to find natural prefix/suffix with totally benign and non-directional semantic meaning, meanwhile, having great power in inducing the Code LLMs to generate vulnerable code. This feature can enable us to conduct the almost-worstcase red-teaming on these LLMs in a real scenario, where users are using natural language. Our extensive experiments and analyses on DeceptPrompt not only validate the effectiveness of our approach but also shed light on the huge weakness of LLMs in the code generation task. When applying the optimized prefix/suffix, the attack success rate (ASR) will improve by average 50% compared with no prefix/suffix applying.

摘要: 随着大型语言模型(LLMS)的发展，在代码生成方面取得了重大进展，使LLMS能够将自然语言转换为编程代码。这些CodeLLM已被广大用户和组织广泛接受。然而，代码中隐藏着一个危险的性质，那就是存在致命的漏洞。虽然一些LLM提供商试图通过与人类的指导保持一致来解决这些问题，但这些努力并不能使Code LLM实用和健壮。如果不深入了解LLMS在实际最坏情况下的性能，将它们应用于各种现实世界应用将是令人担忧的。在这篇文章中，我们回答了一个关键问题：现有的代码LLM是否不会生成易受攻击的代码？如果不是，此问题在实际部署方案中可能的最大严重程度是多少？在本文中，我们介绍了DeceptPrompt算法，它可以生成敌意的自然语言指令，这些指令驱动Code LLMS生成有漏洞的功能正确的代码。DeceptPrompt是通过基于系统进化的算法实现的，具有细粒度的损耗设计。DeceptPrompt的独特优势使我们能够找到具有完全良性和非方向性语义的自然前缀/后缀，同时对诱使Code LLMS生成易受攻击的代码具有强大的能力。这一功能使我们能够在用户使用自然语言的真实场景中对这些LLM进行几乎最糟糕的红色团队。我们在DeceptPrompt上的大量实验和分析不仅验证了我们方法的有效性，而且揭示了LLMS在代码生成任务中的巨大弱点。当应用优化的前缀/后缀时，与不应用前缀/后缀相比，攻击成功率(ASR)将平均提高50%。



## **8. Practical Membership Inference Attacks against Fine-tuned Large Language Models via Self-prompt Calibration**

基于自提示校正的针对精调大型语言模型的实用隶属度推理攻击 cs.CL

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2311.06062v2) [paper-pdf](http://arxiv.org/pdf/2311.06062v2)

**Authors**: Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, Tao Jiang

**Abstract**: Membership Inference Attacks (MIA) aim to infer whether a target data record has been utilized for model training or not. Prior attempts have quantified the privacy risks of language models (LMs) via MIAs, but there is still no consensus on whether existing MIA algorithms can cause remarkable privacy leakage on practical Large Language Models (LLMs). Existing MIAs designed for LMs can be classified into two categories: reference-free and reference-based attacks. They are both based on the hypothesis that training records consistently strike a higher probability of being sampled. Nevertheless, this hypothesis heavily relies on the overfitting of target models, which will be mitigated by multiple regularization methods and the generalization of LLMs. The reference-based attack seems to achieve promising effectiveness in LLMs, which measures a more reliable membership signal by comparing the probability discrepancy between the target model and the reference model. However, the performance of reference-based attack is highly dependent on a reference dataset that closely resembles the training dataset, which is usually inaccessible in the practical scenario. Overall, existing MIAs are unable to effectively unveil privacy leakage over practical fine-tuned LLMs that are overfitting-free and private. We propose a Membership Inference Attack based on Self-calibrated Probabilistic Variation (SPV-MIA). Specifically, since memorization in LLMs is inevitable during the training process and occurs before overfitting, we introduce a more reliable membership signal, probabilistic variation, which is based on memorization rather than overfitting. Furthermore, we introduce a self-prompt approach, which constructs the dataset to fine-tune the reference model by prompting the target LLM itself. In this manner, the adversary can collect a dataset with a similar distribution from public APIs.

摘要: 成员关系推理攻击(MIA)的目的是推断目标数据记录是否已被用于模型训练。以往的研究已经通过MIA量化了语言模型的隐私风险，但对于现有的MIA算法是否会在实际的大型语言模型上造成显著的隐私泄漏，目前还没有达成共识。现有的针对LMS设计的MIA可以分为两类：无引用攻击和基于引用攻击。它们都是基于这样的假设，即培训记录始终具有更高的被抽样概率。然而，这一假设在很大程度上依赖于目标模型的过度拟合，而多种正则化方法和LLMS的推广将缓解这一问题。基于参考的攻击在LLMS中似乎取得了很好的效果，它通过比较目标模型和参考模型之间的概率差异来衡量更可靠的成员信号。然而，基于参考的攻击的性能高度依赖于与训练数据集非常相似的参考数据集，这在实际场景中通常是不可访问的。总体而言，现有的MIA无法有效地揭示实用的微调LLM的隐私泄露，这些LLM是免装修和私密的。提出了一种基于自校准概率变异的成员推理攻击(SPV-MIA)。具体地说，由于LLMS中的记忆在训练过程中是不可避免的，并且发生在过适应之前，因此我们引入了一种更可靠的隶属度信号-概率变异，它基于记忆而不是过适应。此外，我们引入了一种自我提示的方法，该方法构建数据集，通过提示目标LLM本身来微调参考模型。通过这种方式，攻击者可以从公共API收集具有类似分布的数据集。



## **9. Safety Alignment in NLP Tasks: Weakly Aligned Summarization as an In-Context Attack**

NLP任务中的安全对齐：作为上下文攻击的弱对齐总结 cs.CL

17 pages,10 figures

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.06924v1) [paper-pdf](http://arxiv.org/pdf/2312.06924v1)

**Authors**: Yu Fu, Yufei Li, Wen Xiao, Cong Liu, Yue Dong

**Abstract**: Recent developments in balancing the usefulness and safety of Large Language Models (LLMs) have raised a critical question: Are mainstream NLP tasks adequately aligned with safety consideration? Our study, focusing on safety-sensitive documents obtained through adversarial attacks, reveals significant disparities in the safety alignment of various NLP tasks. For instance, LLMs can effectively summarize malicious long documents but often refuse to translate them. This discrepancy highlights a previously unidentified vulnerability: attacks exploiting tasks with weaker safety alignment, like summarization, can potentially compromise the integraty of tasks traditionally deemed more robust, such as translation and question-answering (QA). Moreover, the concurrent use of multiple NLP tasks with lesser safety alignment increases the risk of LLMs inadvertently processing harmful content. We demonstrate these vulnerabilities in various safety-aligned LLMs, particularly Llama2 models and GPT-4, indicating an urgent need for strengthening safety alignments across a broad spectrum of NLP tasks.

摘要: 最近在平衡大型语言模型(LLM)的有用性和安全性方面的发展提出了一个关键问题：主流NLP任务是否与安全考虑充分一致？我们的研究集中在通过对抗性攻击获得的安全敏感文件上，揭示了各种NLP任务在安全匹配方面的显著差异。例如，LLMS可以有效地汇总恶意的长文档，但通常拒绝翻译它们。这一差异突显了一个以前未知的漏洞：攻击利用安全性较弱的任务(如摘要)，可能会潜在地损害传统上被认为更健壮的任务的完整性，如翻译和问答(QA)。此外，同时使用安全性较低的多个NLP任务会增加LLMS无意中处理有害内容的风险。我们在各种安全对齐的LLM中展示了这些漏洞，特别是Llama2型号和GPT-4，这表明迫切需要在广泛的NLP任务中加强安全对齐。



## **10. GPTBIAS: A Comprehensive Framework for Evaluating Bias in Large Language Models**

GPTBIAS：一个评估大型语言模型中偏差的综合框架 cs.CL

**SubmitDate**: 2023-12-11    [abs](http://arxiv.org/abs/2312.06315v1) [paper-pdf](http://arxiv.org/pdf/2312.06315v1)

**Authors**: Jiaxu Zhao, Meng Fang, Shirui Pan, Wenpeng Yin, Mykola Pechenizkiy

**Abstract**: Warning: This paper contains content that may be offensive or upsetting. There has been a significant increase in the usage of large language models (LLMs) in various applications, both in their original form and through fine-tuned adaptations. As a result, LLMs have gained popularity and are being widely adopted by a large user community. However, one of the concerns with LLMs is the potential generation of socially biased content. The existing evaluation methods have many constraints, and their results exhibit a limited degree of interpretability. In this work, we propose a bias evaluation framework named GPTBIAS that leverages the high performance of LLMs (e.g., GPT-4 \cite{openai2023gpt4}) to assess bias in models. We also introduce prompts called Bias Attack Instructions, which are specifically designed for evaluating model bias. To enhance the credibility and interpretability of bias evaluation, our framework not only provides a bias score but also offers detailed information, including bias types, affected demographics, keywords, reasons behind the biases, and suggestions for improvement. We conduct extensive experiments to demonstrate the effectiveness and usability of our bias evaluation framework.

摘要: 警告：本文包含可能冒犯或令人反感的内容。大型语言模型(LLM)在各种应用程序中的使用显著增加，无论是以其原始形式还是通过微调的适应。因此，LLMS变得流行起来，并被大量用户社区广泛采用。然而，LLMS的一个令人担忧的问题是，可能会产生带有社会偏见的内容。现有的评价方法有许多限制，其结果表现出有限的可解释性。在这项工作中，我们提出了一个称为GPTBIAS的偏差评估框架，该框架利用LLMS的高性能(例如，GPT-4\cite{Openai2023gpt4})来评估模型中的偏差。我们还引入了称为偏差攻击说明的提示，这是专门为评估模型偏差而设计的。为了增强偏见评估的可信度和可解释性，我们的框架不仅提供了偏见评分，还提供了详细的信息，包括偏见类型、受影响的人口统计学、关键词、偏见背后的原因和改进建议。我们进行了大量的实验，以证明我们的偏差评估框架的有效性和可用性。



## **11. InferDPT: Privacy-Preserving Inference for Black-box Large Language Model**

InferDPT：黑箱大语言模型的隐私保护推理 cs.CR

**SubmitDate**: 2023-12-11    [abs](http://arxiv.org/abs/2310.12214v5) [paper-pdf](http://arxiv.org/pdf/2310.12214v5)

**Authors**: Meng Tong, Kejiang Chen, Jie Zhang, Yuang Qi, Weiming Zhang, Nenghai Yu

**Abstract**: Large language models (LLMs), like ChatGPT, have greatly simplified text generation tasks. However, they have also raised concerns about privacy risks such as data leakage and unauthorized data collection. Existing solutions for privacy-preserving inference face practical challenges related to computation time and communication costs. In this paper, we propose InferDPT, the first practical framework for the privacy-preserving Inference of black-box LLMs, implementing Differential Privacy in Text generation. InferDPT comprises two key modules: the "perturbation module" utilizes the exponential mechanism to generate a perturbed prompt, facilitating privacy-preserving inference with black-box LLMs, and the "extraction module", inspired by knowledge distillation and retrieval-augmented generation, extracts coherent and consistent text from the perturbed generation result, ensuring successful text generation completion. To address privacy concerns related to previous exponential mechanisms' susceptibility to embedding revision attacks, we introduce RANTEXT, a novel differential privacy mechanism integrated into the perturbation module of InferDPT, which introduces the concept of "RANdom adjacency" for TEXT perturbation within the prompt. Experimental results across three datasets demonstrate that the text generation quality of InferDPT is comparable to that of non-private GPT-4, and RANTEXT surpasses existing state-of-the-art mechanisms, namely, SANTEXT+ and CUSTEXT+ in the trade-off between privacy and utility. Even with an privacy parameter epsilon value of 6.0, RANTEXT achieves an average privacy protection rate exceeding 90% against embedding revision attacks, which is 0.58 times higher than that of SANTEXT+ and 3.35 times higher than that of CUSTEXT+.

摘要: 大型语言模型(LLM)，如ChatGPT，极大地简化了文本生成任务。然而，他们也对数据泄露和未经授权的数据收集等隐私风险表示担忧。现有的隐私保护推理解决方案面临着与计算时间和通信成本相关的实际挑战。在本文中，我们提出了第一个实用的黑盒LLMS隐私保护推理框架InferDPT，在文本生成中实现了差分隐私。InferDPT包括两个关键模块：“扰动模块”利用指数机制生成扰动提示，便于使用黑盒LLMS进行隐私保护推理；“提取模块”受知识提炼和检索-增强生成的启发，从扰动生成结果中提取连贯一致的文本，确保文本生成成功完成。针对以往指数机制易受修改攻击的隐私性问题，引入了一种新的差异化隐私机制RANTEXT，该机制集成在InferDPT的扰动模块中，引入了随机邻接的概念来处理提示内的文本扰动。在三个数据集上的实验结果表明，InferDPT的文本生成质量与非私有GPT-4相当，RANTEXT在隐私和效用之间的权衡方面超过了现有的最新机制SanText+和CUSTEXT+。即使在隐私参数epsilon值为6.0的情况下，RANTEXT对嵌入修改攻击的平均隐私保护率也超过90%，比SanText+高0.58倍，比CUSTEXT+高3.35倍。



## **12. METAL: Metamorphic Testing Framework for Analyzing Large-Language Model Qualities**

Metals：分析大语言模型性质的变形测试框架 cs.SE

Accepted to International Conference on Software Testing,  Verification and Validation (ICST) 2024 / Key words: Large-language models,  Metamorphic testing, Quality evaluation, Text perturbations

**SubmitDate**: 2023-12-11    [abs](http://arxiv.org/abs/2312.06056v1) [paper-pdf](http://arxiv.org/pdf/2312.06056v1)

**Authors**: Sangwon Hyun, Mingyu Guo, M. Ali Babar

**Abstract**: Large-Language Models (LLMs) have shifted the paradigm of natural language data processing. However, their black-boxed and probabilistic characteristics can lead to potential risks in the quality of outputs in diverse LLM applications. Recent studies have tested Quality Attributes (QAs), such as robustness or fairness, of LLMs by generating adversarial input texts. However, existing studies have limited their coverage of QAs and tasks in LLMs and are difficult to extend. Additionally, these studies have only used one evaluation metric, Attack Success Rate (ASR), to assess the effectiveness of their approaches. We propose a MEtamorphic Testing for Analyzing LLMs (METAL) framework to address these issues by applying Metamorphic Testing (MT) techniques. This approach facilitates the systematic testing of LLM qualities by defining Metamorphic Relations (MRs), which serve as modularized evaluation metrics. The METAL framework can automatically generate hundreds of MRs from templates that cover various QAs and tasks. In addition, we introduced novel metrics that integrate the ASR method into the semantic qualities of text to assess the effectiveness of MRs accurately. Through the experiments conducted with three prominent LLMs, we have confirmed that the METAL framework effectively evaluates essential QAs on primary LLM tasks and reveals the quality risks in LLMs. Moreover, the newly proposed metrics can guide the optimal MRs for testing each task and suggest the most effective method for generating MRs.

摘要: 大语言模型（LLM）改变了自然语言数据处理的范式。然而，它们的黑盒和概率特性可能会导致各种LLM应用程序中输出质量的潜在风险。最近的研究已经通过生成对抗性输入文本来测试LLM的质量属性（QA），例如鲁棒性或公平性。然而，现有的研究限制了他们的QA和LLM任务的覆盖范围，很难扩展。此外，这些研究只使用了一个评估指标，攻击成功率（ASR），以评估其方法的有效性。我们提出了一个变形测试分析LLM（金属）框架来解决这些问题，通过应用变形测试（MT）技术。这种方法通过定义作为模块化评估指标的变形关系（MR），促进了LLM质量的系统测试。METAL框架可以从涵盖各种QA和任务的模板中自动生成数百个MR。此外，我们引入了新的指标，将ASR方法集成到文本的语义质量中，以准确评估MR的有效性。通过与三个突出的LLM进行的实验，我们已经证实，金属框架有效地评估主要LLM任务的基本QA，并揭示了LLM的质量风险。此外，新提出的指标可以指导最佳的MR测试每个任务，并建议最有效的方法来生成MR。



## **13. Occlusion-based Detection of Trojan-triggering Inputs in Large Language Models of Code**

大型语言代码模型中基于遮挡的木马触发输入检测 cs.SE

**SubmitDate**: 2023-12-10    [abs](http://arxiv.org/abs/2312.04004v2) [paper-pdf](http://arxiv.org/pdf/2312.04004v2)

**Authors**: Aftab Hussain, Md Rafiqul Islam Rabin, Toufique Ahmed, Mohammad Amin Alipour, Bowen Xu

**Abstract**: Large language models (LLMs) are becoming an integrated part of software development. These models are trained on large datasets for code, where it is hard to verify each data point. Therefore, a potential attack surface can be to inject poisonous data into the training data to make models vulnerable, aka trojaned. It can pose a significant threat by hiding manipulative behaviors inside models, leading to compromising the integrity of the models in downstream tasks.   In this paper, we propose an occlusion-based human-in-the-loop technique, OSeql, to distinguish trojan-triggering inputs of code. The technique is based on the observation that trojaned neural models of code rely heavily on the triggering part of input; hence, its removal would change the confidence of the models in their prediction substantially. Our results suggest that OSeql can detect the triggering inputs with almost 100% recall. We discuss the problem of false positives and how to address them. These results provide a baseline for future studies in this field.

摘要: 大型语言模型(LLM)正在成为软件开发的一个组成部分。这些模型是在大数据集上针对代码进行训练的，在代码中很难验证每个数据点。因此，潜在的攻击面可能是向训练数据中注入有毒数据，使模型容易受到攻击，也就是安装了特洛伊木马。它可以通过将操纵行为隐藏在模型中而构成重大威胁，从而导致在下游任务中损害模型的完整性。在本文中，我们提出了一种基于遮挡的人在环中技术OSeql，用于区分木马触发的代码输入。该技术基于这样的观察，即特洛伊木马代码的神经模型严重依赖于输入的触发部分；因此，移除它将极大地改变模型对其预测的置信度。我们的结果表明，OSeql能够以几乎100%的召回率检测到触发输入。我们讨论了误报问题以及如何解决这些问题。这些结果为该领域未来的研究提供了一个基线。



## **14. Towards Robust Pruning: An Adaptive Knowledge-Retention Pruning Strategy for Language Models**

朝向稳健剪枝：一种自适应的语言模型知识保留剪枝策略 cs.CL

**SubmitDate**: 2023-12-10    [abs](http://arxiv.org/abs/2310.13191v2) [paper-pdf](http://arxiv.org/pdf/2310.13191v2)

**Authors**: Jianwei Li, Qi Lei, Wei Cheng, Dongkuan Xu

**Abstract**: The pruning objective has recently extended beyond accuracy and sparsity to robustness in language models. Despite this, existing methods struggle to enhance robustness against adversarial attacks when continually increasing model sparsity and require a retraining process. As humans step into the era of large language models, these issues become increasingly prominent. This paper proposes that the robustness of language models is proportional to the extent of pre-trained knowledge they encompass. Accordingly, we introduce a post-training pruning strategy designed to faithfully replicate the embedding space and feature space of dense language models, aiming to conserve more pre-trained knowledge during the pruning process. In this setup, each layer's reconstruction error not only originates from itself but also includes cumulative error from preceding layers, followed by an adaptive rectification. Compared to other state-of-art baselines, our approach demonstrates a superior balance between accuracy, sparsity, robustness, and pruning cost with BERT on datasets SST2, IMDB, and AGNews, marking a significant stride towards robust pruning in language models.

摘要: 修剪目标最近已经超越了语言模型中的精确度和稀疏性，扩展到了健壮性。尽管如此，现有的方法在不断增加模型稀疏性的同时努力增强对敌对攻击的鲁棒性，并且需要重新训练过程。随着人类步入大型语言模型时代，这些问题变得日益突出。本文提出语言模型的稳健性与它们所包含的预训练知识的程度成正比。因此，我们提出了一种训练后剪枝策略，旨在忠实地复制密集语言模型的嵌入空间和特征空间，目的是在剪枝过程中保存更多的预先训练的知识。在这种设置中，每一层的重建误差不仅源于自身，还包括来自前几层的累积误差，然后进行自适应校正。与其他最先进的基线相比，我们的方法在精确度、稀疏性、健壮性和剪枝成本之间表现出了更好的平衡，在数据集Sst2、IMDB和AgNews上使用ERT，标志着在语言模型中朝着健壮剪枝迈出了重要的一步。



## **15. Temporal-Distributed Backdoor Attack Against Video Based Action Recognition**

针对视频动作识别的时间分布式后门攻击 cs.CV

accepted by AAAI 2024

**SubmitDate**: 2023-12-09    [abs](http://arxiv.org/abs/2308.11070v3) [paper-pdf](http://arxiv.org/pdf/2308.11070v3)

**Authors**: Xi Li, Songhe Wang, Ruiquan Huang, Mahanth Gowda, George Kesidis

**Abstract**: Deep neural networks (DNNs) have achieved tremendous success in various applications including video action recognition, yet remain vulnerable to backdoor attacks (Trojans). The backdoor-compromised model will mis-classify to the target class chosen by the attacker when a test instance (from a non-target class) is embedded with a specific trigger, while maintaining high accuracy on attack-free instances. Although there are extensive studies on backdoor attacks against image data, the susceptibility of video-based systems under backdoor attacks remains largely unexplored. Current studies are direct extensions of approaches proposed for image data, e.g., the triggers are independently embedded within the frames, which tend to be detectable by existing defenses. In this paper, we introduce a simple yet effective backdoor attack against video data. Our proposed attack, adding perturbations in a transformed domain, plants an imperceptible, temporally distributed trigger across the video frames, and is shown to be resilient to existing defensive strategies. The effectiveness of the proposed attack is demonstrated by extensive experiments with various well-known models on two video recognition benchmarks, UCF101 and HMDB51, and a sign language recognition benchmark, Greek Sign Language (GSL) dataset. We delve into the impact of several influential factors on our proposed attack and identify an intriguing effect termed "collateral damage" through extensive studies.

摘要: 深度神经网络(DNN)在包括视频动作识别在内的各种应用中取得了巨大的成功，但仍然容易受到后门攻击(特洛伊木马)。当测试实例(来自非目标类)嵌入特定触发器时，后门泄露模型将被错误分类为攻击者选择的目标类，同时保持对无攻击实例的高准确性。虽然已经有大量关于针对图像数据的后门攻击的研究，但基于视频的系统在后门攻击下的易感性在很大程度上仍未被探索。当前的研究是对为图像数据提出的方法的直接扩展，例如，触发器独立地嵌入在帧中，这往往是现有防御系统可检测的。本文介绍了一种简单而有效的针对视频数据的后门攻击。我们提出的攻击在变换的域中增加了扰动，在视频帧上植入了一个不可察觉的、时间分布的触发器，并被证明对现有的防御策略具有弹性。在两个视频识别基准UCF101和HMDB51和一个手语识别基准希腊手语(GSL)数据集上进行了大量的实验，证明了所提出的攻击的有效性。我们深入研究了几个影响因素对我们提出的攻击的影响，并通过广泛的研究确定了一种有趣的影响，称为“附带损害”。



## **16. HuRef: HUman-REadable Fingerprint for Large Language Models**

HuRef：大型语言模型的人类可读指纹 cs.CL

**SubmitDate**: 2023-12-08    [abs](http://arxiv.org/abs/2312.04828v1) [paper-pdf](http://arxiv.org/pdf/2312.04828v1)

**Authors**: Boyi Zeng, Chenghu Zhou, Xinbing Wang, Zhouhan Lin

**Abstract**: Protecting the copyright of large language models (LLMs) has become crucial due to their resource-intensive training and accompanying carefully designed licenses. However, identifying the original base model of an LLM is challenging due to potential parameter alterations through fine-tuning or continued pretraining. In this study, we introduce HuRef, a human-readable fingerprint for LLMs that uniquely identifies the base model without exposing model parameters or interfering with training. We first observe that the vector direction of LLM parameters remains stable after the model has converged during pretraining, showing negligible perturbations through subsequent training steps, including continued pretraining, supervised fine-tuning (SFT), and RLHF, which makes it a sufficient condition to identify the base model. The necessity is validated by continuing to train an LLM with an extra term to drive away the model parameters' direction and the model becomes damaged. However, this direction is vulnerable to simple attacks like dimension permutation or matrix rotation, which significantly change it without affecting performance. To address this, leveraging the Transformer structure, we systematically analyze potential attacks and define three invariant terms that identify an LLM's base model. We make these invariant terms human-readable by mapping them to a Gaussian vector using a convolutional encoder and then converting it into a natural image with StyleGAN2. Our method generates a dog image as an identity fingerprint for an LLM, where the dog's appearance strongly indicates the LLM's base model. Experimental results across various LLMs demonstrate the effectiveness of our method, the generated dog image remains invariant to different training steps, including SFT, RLHF, or even continued pretraining with augmented vocabulary in a new language.

摘要: 保护大型语言模型(LLM)的版权已变得至关重要，因为它们需要进行资源密集型培训，并附带精心设计的许可证。然而，识别LLM的原始基本模型是具有挑战性的，因为通过微调或持续的预训练可能会改变参数。在这项研究中，我们引入了HuRef，这是一种用于LLMS的人类可读指纹，它在不暴露模型参数或干扰训练的情况下唯一地识别基本模型。我们首先观察到，在预训练过程中，模型收敛后，LLM参数的向量方向保持稳定，通过后续的训练步骤，包括继续预训练、有监督微调(SFT)和RLHF，表现出可以忽略的扰动，这使得它成为识别基本模型的充分条件。通过继续训练一个带有额外项的LLM来驱离模型参数的方向，从而使模型受损，从而验证了这种必要性。然而，这个方向很容易受到维度置换或矩阵旋转等简单攻击，这些攻击会在不影响性能的情况下显著改变它。为了解决这个问题，利用Transformer结构，我们系统地分析了潜在的攻击，并定义了识别LLM基本模型的三个不变术语。我们使用卷积编码器将这些不变项映射到高斯向量，然后使用StyleGAN2将其转换为自然图像，从而使这些不变项变得可读。我们的方法生成了一幅狗图像作为LLM的身份指纹，其中狗的外表强烈地指示了LLM的基本模型。在不同LLMS上的实验结果证明了该方法的有效性，生成的狗图像在不同的训练步骤中保持不变，包括SFT、RLHF，甚至是在新语言中增加词汇量的持续预训练。



## **17. Goal-Oriented Prompt Attack and Safety Evaluation for LLMs**

面向目标的低空导弹快速攻击与安全评估 cs.CL

**SubmitDate**: 2023-12-08    [abs](http://arxiv.org/abs/2309.11830v2) [paper-pdf](http://arxiv.org/pdf/2309.11830v2)

**Authors**: Chengyuan Liu, Fubang Zhao, Lizhi Qing, Yangyang Kang, Changlong Sun, Kun Kuang, Fei Wu

**Abstract**: Large Language Models (LLMs) presents significant priority in text understanding and generation. However, LLMs suffer from the risk of generating harmful contents especially while being employed to applications. There are several black-box attack methods, such as Prompt Attack, which can change the behaviour of LLMs and induce LLMs to generate unexpected answers with harmful contents. Researchers are interested in Prompt Attack and Defense with LLMs, while there is no publicly available dataset with high successful attacking rate to evaluate the abilities of defending prompt attack. In this paper, we introduce a pipeline to construct high-quality prompt attack samples, along with a Chinese prompt attack dataset called CPAD. Our prompts aim to induce LLMs to generate unexpected outputs with several carefully designed prompt attack templates and widely concerned attacking contents. Different from previous datasets involving safety estimation, we construct the prompts considering three dimensions: contents, attacking methods and goals. Especially, the attacking goals indicate the behaviour expected after successfully attacking the LLMs, thus the responses can be easily evaluated and analysed. We run several popular Chinese LLMs on our dataset, and the results show that our prompts are significantly harmful to LLMs, with around 70% attack success rate to GPT-3.5. CPAD is publicly available at https://github.com/liuchengyuan123/CPAD.

摘要: 大语言模型(LLM)在文本理解和生成中具有重要的优先地位。然而，LLMS面临着产生有害内容的风险，特别是在应用程序中使用时。有几种黑盒攻击方法，如提示攻击，可以改变LLMS的行为，并诱导LLMS生成包含有害内容的意外答案。研究人员对LLMS的快速攻防很感兴趣，但目前还没有公开的、具有较高攻击成功率的数据集来评估防御快速攻击的能力。在本文中，我们介绍了一种构造高质量即时攻击样本的管道，以及一个中文即时攻击数据集CPAD。我们的提示旨在通过精心设计的几个提示攻击模板和广泛关注的攻击内容来诱导LLM产生意想不到的输出。与以往涉及安全评估的数据集不同，我们从内容、攻击方法和目标三个维度构建提示。特别是，攻击目标指示了成功攻击LLMS后的预期行为，因此可以很容易地评估和分析响应。我们在我们的数据集上运行了几个流行的中文LLMS，结果表明我们的提示对LLMS具有显著的危害，对GPT-3.5的攻击成功率约为70%。CPAD可在https://github.com/liuchengyuan123/CPAD.上公开购买



## **18. Make Them Spill the Beans! Coercive Knowledge Extraction from (Production) LLMs**

让他们说漏嘴！从(产生式)LLMS中提取强制知识 cs.CR

**SubmitDate**: 2023-12-08    [abs](http://arxiv.org/abs/2312.04782v1) [paper-pdf](http://arxiv.org/pdf/2312.04782v1)

**Authors**: Zhuo Zhang, Guangyu Shen, Guanhong Tao, Siyuan Cheng, Xiangyu Zhang

**Abstract**: Large Language Models (LLMs) are now widely used in various applications, making it crucial to align their ethical standards with human values. However, recent jail-breaking methods demonstrate that this alignment can be undermined using carefully constructed prompts. In our study, we reveal a new threat to LLM alignment when a bad actor has access to the model's output logits, a common feature in both open-source LLMs and many commercial LLM APIs (e.g., certain GPT models). It does not rely on crafting specific prompts. Instead, it exploits the fact that even when an LLM rejects a toxic request, a harmful response often hides deep in the output logits. By forcefully selecting lower-ranked output tokens during the auto-regressive generation process at a few critical output positions, we can compel the model to reveal these hidden responses. We term this process model interrogation. This approach differs from and outperforms jail-breaking methods, achieving 92% effectiveness compared to 62%, and is 10 to 20 times faster. The harmful content uncovered through our method is more relevant, complete, and clear. Additionally, it can complement jail-breaking strategies, with which results in further boosting attack performance. Our findings indicate that interrogation can extract toxic knowledge even from models specifically designed for coding tasks.

摘要: 大型语言模型(LLM)现在被广泛应用于各种应用中，因此使它们的伦理标准与人类价值观保持一致至关重要。然而，最近的越狱方法表明，使用精心构建的提示可以破坏这种对齐。在我们的研究中，我们揭示了当一个坏的参与者可以访问模型的输出日志时对LLM对齐的新威胁，这是开源LLMS和许多商业LLMAPI(例如，某些GPT模型)中的一个共同特征。它不依赖于精心设计特定的提示。相反，它利用了这样一个事实，即即使LLM拒绝了有毒请求，有害的响应通常也隐藏在输出日志的深处。通过在自回归生成过程中在几个关键的输出位置强制选择较低等级的输出令牌，我们可以迫使模型揭示这些隐藏的响应。我们称这一过程为审问模式。这种方法与越狱方法不同，也优于越狱方法，达到了92%的有效率，而不是62%，而且速度快了10到20倍。通过我们的方法发现的有害内容更相关、更完整、更清晰。此外，它还可以补充越狱策略，从而进一步提高攻击性能。我们的发现表明，审问甚至可以从专门为编码任务设计的模型中提取有毒知识。



## **19. Forcing Generative Models to Degenerate Ones: The Power of Data Poisoning Attacks**

迫使生成性模型退化：数据中毒攻击的威力 cs.CR

19 pages, 6 figures. Published at NeurIPS 2023 Workshop on Backdoors  in Deep Learning: The Good, the Bad, and the Ugly

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2312.04748v1) [paper-pdf](http://arxiv.org/pdf/2312.04748v1)

**Authors**: Shuli Jiang, Swanand Ravindra Kadhe, Yi Zhou, Ling Cai, Nathalie Baracaldo

**Abstract**: Growing applications of large language models (LLMs) trained by a third party raise serious concerns on the security vulnerability of LLMs.It has been demonstrated that malicious actors can covertly exploit these vulnerabilities in LLMs through poisoning attacks aimed at generating undesirable outputs. While poisoning attacks have received significant attention in the image domain (e.g., object detection), and classification tasks, their implications for generative models, particularly in the realm of natural language generation (NLG) tasks, remain poorly understood. To bridge this gap, we perform a comprehensive exploration of various poisoning techniques to assess their effectiveness across a range of generative tasks. Furthermore, we introduce a range of metrics designed to quantify the success and stealthiness of poisoning attacks specifically tailored to NLG tasks. Through extensive experiments on multiple NLG tasks, LLMs and datasets, we show that it is possible to successfully poison an LLM during the fine-tuning stage using as little as 1\% of the total tuning data samples. Our paper presents the first systematic approach to comprehend poisoning attacks targeting NLG tasks considering a wide range of triggers and attack settings. We hope our findings will assist the AI security community in devising appropriate defenses against such threats.

摘要: 由第三方训练的大型语言模型(LLM)的应用日益增多，引起了人们对LLM安全漏洞的严重关注，已有研究表明，恶意行为者可以通过投毒攻击来秘密利用LLM中的这些漏洞，目的是产生不希望看到的输出。虽然中毒攻击在图像领域(例如，目标检测)和分类任务中受到了极大的关注，但它们对生成模型的影响，特别是在自然语言生成(NLG)任务领域，仍然知之甚少。为了弥补这一差距，我们对各种中毒技术进行了全面的探索，以评估它们在一系列生成性任务中的有效性。此外，我们还介绍了一系列专门为NLG任务量身定做的用于量化投毒攻击的成功率和隐蔽性的指标。通过在多个NLG任务、LLM和数据集上的大量实验，我们表明，在微调阶段，只要使用总调整数据样本的1%，就可以成功地毒化LLM。我们提出了第一种系统的方法来理解针对NLG任务的中毒攻击，考虑了广泛的触发因素和攻击设置。我们希望我们的发现将有助于人工智能安全界设计针对此类威胁的适当防御措施。



## **20. Defending Against Alignment-Breaking Attacks via Robustly Aligned LLM**

通过强健对齐的LLM防御对齐破坏攻击 cs.CL

16 Pages, 5 Figures, 6 Tables

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2309.14348v2) [paper-pdf](http://arxiv.org/pdf/2309.14348v2)

**Authors**: Bochuan Cao, Yuanpu Cao, Lu Lin, Jinghui Chen

**Abstract**: Recently, Large Language Models (LLMs) have made significant advancements and are now widely used across various domains. Unfortunately, there has been a rising concern that LLMs can be misused to generate harmful or malicious content. Though a line of research has focused on aligning LLMs with human values and preventing them from producing inappropriate content, such alignments are usually vulnerable and can be bypassed by alignment-breaking attacks via adversarially optimized or handcrafted jailbreaking prompts. In this work, we introduce a Robustly Aligned LLM (RA-LLM) to defend against potential alignment-breaking attacks. RA-LLM can be directly constructed upon an existing aligned LLM with a robust alignment checking function, without requiring any expensive retraining or fine-tuning process of the original LLM. Furthermore, we also provide a theoretical analysis for RA-LLM to verify its effectiveness in defending against alignment-breaking attacks. Through real-world experiments on open-source large language models, we demonstrate that RA-LLM can successfully defend against both state-of-the-art adversarial prompts and popular handcrafted jailbreaking prompts by reducing their attack success rates from nearly 100% to around 10% or less.

摘要: 近年来，大型语言模型(LLM)取得了长足的进步，现已广泛应用于各个领域。不幸的是，人们越来越担心LLMS可能被滥用来生成有害或恶意的内容。尽管有一系列研究专注于将LLM与人类价值观保持一致，并防止它们产生不适当的内容，但这种调整通常是脆弱的，可以通过恶意优化或手工制作的越狱提示被破坏顺序的攻击绕过。在这项工作中，我们引入了一种鲁棒对齐LLM(RA-LLM)来防御潜在的对齐破坏攻击。RA-LLM可以直接构建在现有的对准LLM上，具有健壮的对准检查功能，而不需要对原始LLM进行任何昂贵的再培训或微调过程。此外，我们还对RA-LLM进行了理论分析，以验证其在抵抗对齐破坏攻击方面的有效性。通过在开源大型语言模型上的真实世界实验，我们证明了RA-LLM能够成功地防御最新的敌意提示和流行的手工越狱提示，将攻击成功率从近100%降低到10%左右或更低。



## **21. Domain Private Transformers for Multi-Domain Dialog Systems**

用于多域对话系统的域专用转换器 cs.CL

Accepted to Findings of EMNLP 2023 (short paper). Code available at  https://github.com/asappresearch/domain-private-transformers

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2305.14208v2) [paper-pdf](http://arxiv.org/pdf/2305.14208v2)

**Authors**: Anmol Kabra, Ethan R. Elenberg

**Abstract**: Large, general purpose language models have demonstrated impressive performance across many different conversational domains. While multi-domain language models achieve low overall perplexity, their outputs are not guaranteed to stay within the domain of a given input prompt. This paper proposes domain privacy as a novel way to quantify how likely a conditional language model will leak across domains. We also develop policy functions based on token-level domain classification, and propose an efficient fine-tuning method to improve the trained model's domain privacy. Experiments on membership inference attacks show that our proposed method has comparable resiliency to methods adapted from recent literature on differentially private language models.

摘要: 大型的通用语言模型已经在许多不同的会话领域展示了令人印象深刻的性能。虽然多领域语言模型实现了较低的总体困惑，但它们的输出不能保证停留在给定输入提示的领域内。本文提出了一种新的方法来量化条件语言模型跨域泄漏的可能性。我们还开发了基于令牌级域分类的策略函数，并提出了一种有效的微调方法来提高训练模型的域私密性。在成员关系推理攻击上的实验表明，我们提出的方法与最近文献中关于差分私有语言模型的方法具有相当的弹性。



## **22. Analyzing the Inherent Response Tendency of LLMs: Real-World Instructions-Driven Jailbreak**

分析LLMS的内在响应趋势：现实世界指令驱动的越狱 cs.CL

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2312.04127v1) [paper-pdf](http://arxiv.org/pdf/2312.04127v1)

**Authors**: Yanrui Du, Sendong Zhao, Ming Ma, Yuhan Chen, Bing Qin

**Abstract**: Extensive work has been devoted to improving the safety mechanism of Large Language Models (LLMs). However, in specific scenarios, LLMs still generate harmful responses when faced with malicious instructions, a phenomenon referred to as "Jailbreak Attack". In our research, we introduce a novel jailbreak attack method (\textbf{RADIAL}), which consists of two steps: 1) Inherent Response Tendency Analysis: we analyze the inherent affirmation and rejection tendency of LLMs to react to real-world instructions. 2) Real-World Instructions-Driven Jailbreak: based on our analysis, we strategically choose several real-world instructions and embed malicious instructions into them to amplify the LLM's potential to generate harmful responses. On three open-source human-aligned LLMs, our method achieves excellent jailbreak attack performance for both Chinese and English malicious instructions. Besides, we guided detailed ablation experiments and verified the effectiveness of our core idea "Inherent Response Tendency Analysis". Our exploration also exposes the vulnerability of LLMs to being induced into generating more detailed harmful responses in subsequent rounds of dialogue.

摘要: 大量的工作致力于改进大型语言模型（LLM）的安全机制。然而，在特定的场景中，LLM在面对恶意指令时仍然会产生有害的响应，这种现象被称为“越狱攻击”。在我们的研究中，我们介绍了一种新的越狱攻击方法（\textbf{RADIAL}），它包括两个步骤：1）内在响应趋势分析：我们分析了LLM对现实世界指令的内在肯定和拒绝倾向。2)现实世界的指令驱动的越狱：根据我们的分析，我们战略性地选择了几个现实世界的指令，并将恶意指令嵌入其中，以放大LLM产生有害响应的潜力。在三个开源的人类对齐的LLM上，我们的方法对中文和英文恶意指令都取得了很好的越狱攻击性能。此外，我们指导了详细的消融实验，并验证了我们的核心思想“内在反应倾向分析”的有效性。我们的探索还暴露了LLM的脆弱性，在随后的对话中被诱导产生更详细的有害反应。



## **23. Mark My Words: Analyzing and Evaluating Language Model Watermarks**

标记我的话：分析和评估语言模型水印 cs.CR

18 pages, 11 figures

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2312.00273v2) [paper-pdf](http://arxiv.org/pdf/2312.00273v2)

**Authors**: Julien Piet, Chawin Sitawarin, Vivian Fang, Norman Mu, David Wagner

**Abstract**: The capabilities of large language models have grown significantly in recent years and so too have concerns about their misuse. In this context, the ability to distinguish machine-generated text from human-authored content becomes important. Prior works have proposed numerous schemes to watermark text, which would benefit from a systematic evaluation framework. This work focuses on text watermarking techniques - as opposed to image watermarks - and proposes MARKMYWORDS, a comprehensive benchmark for them under different tasks as well as practical attacks. We focus on three main metrics: quality, size (e.g. the number of tokens needed to detect a watermark), and tamper-resistance. Current watermarking techniques are good enough to be deployed: Kirchenbauer et al. [1] can watermark Llama2-7B-chat with no perceivable loss in quality, the watermark can be detected with fewer than 100 tokens, and the scheme offers good tamper-resistance to simple attacks. We argue that watermark indistinguishability, a criteria emphasized in some prior works, is too strong a requirement: schemes that slightly modify logit distributions outperform their indistinguishable counterparts with no noticeable loss in generation quality. We publicly release our benchmark (https://github.com/wagner-group/MarkMyWords)

摘要: 近年来，大型语言模型的能力显著增长，人们对它们的滥用也感到担忧。在这种情况下，区分机器生成的文本和人类创作的内容的能力变得很重要。以前的工作已经提出了许多方案来对文本进行水印，这将受益于一个系统的评估框架。这项工作的重点是文本水印技术，而不是图像水印，并提出了MARKMYWORDS，一个针对不同任务和实际攻击的综合基准。我们主要关注三个指标：质量、大小(例如，检测水印所需的令牌数量)和防篡改。目前的水印技术足够好，可以部署：Kirchenbauer等人。[1]可以在Llama2-7B-Chat上嵌入水印而不会造成明显的质量损失，水印的检测只需要不到100个令牌，并且对简单攻击具有良好的抗篡改能力。我们认为，水印的不可区分性是一个太强的要求：稍微修改Logit分布的方案在生成质量上没有明显损失的情况下，性能优于它们的不可区分的对应方案。我们公开发布我们的基准(https://github.com/wagner-group/MarkMyWords)



## **24. Dr. Jekyll and Mr. Hyde: Two Faces of LLMs**

杰基尔博士和海德先生：LLMS的两张面孔 cs.CR

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2312.03853v1) [paper-pdf](http://arxiv.org/pdf/2312.03853v1)

**Authors**: Matteo Gioele Collu, Tom Janssen-Groesbeek, Stefanos Koffas, Mauro Conti, Stjepan Picek

**Abstract**: This year, we witnessed a rise in the use of Large Language Models, especially when combined with applications like chatbot assistants. Safety mechanisms and specialized training procedures are put in place to prevent improper responses from these assistants. In this work, we bypass these measures for ChatGPT and Bard (and, to some extent, Bing chat) by making them impersonate complex personas with opposite characteristics as those of the truthful assistants they are supposed to be. We start by creating elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversation followed a role-play style to get the response the assistant was not allowed to provide. By making use of personas, we show that the response that is prohibited is actually provided, making it possible to obtain unauthorized, illegal, or harmful information. This work shows that by using adversarial personas, one can overcome safety mechanisms set out by ChatGPT and Bard. It also introduces several ways of activating such adversarial personas, altogether showing that both chatbots are vulnerable to this kind of attack.

摘要: 今年，我们见证了大型语言模型的使用增加，特别是当与聊天机器人助手等应用相结合时。本集团设有安全机制及专门培训程序，以防止该等助理作出不当反应。在这项工作中，我们绕过了ChatGPT和Bard（以及在某种程度上，Bing聊天）的这些措施，让他们模仿复杂的人物角色，这些人物角色具有与他们应该成为的真实助手相反的特征。我们首先创建这些人物角色的详细传记，然后在与相同聊天机器人的新会话中使用。我们的谈话遵循角色扮演的风格，以获得助理不允许提供的回应。通过使用人物角色，我们表明实际上提供了被禁止的响应，从而有可能获得未经授权的，非法的或有害的信息。这项工作表明，通过使用对抗性人物角色，可以克服ChatGPT和Bard提出的安全机制。它还介绍了激活这种对抗性角色的几种方法，共同表明这两种聊天机器人都容易受到这种攻击。



## **25. Clinical Notes Reveal Physician Fatigue**

临床笔记显示医生疲劳 cs.CL

**SubmitDate**: 2023-12-05    [abs](http://arxiv.org/abs/2312.03077v1) [paper-pdf](http://arxiv.org/pdf/2312.03077v1)

**Authors**: Chao-Chun Hsu, Ziad Obermeyer, Chenhao Tan

**Abstract**: Physicians write notes about patients. In doing so, they reveal much about themselves. Using data from 129,228 emergency room visits, we train a model to identify notes written by fatigued physicians -- those who worked 5 or more of the prior 7 days. In a hold-out set, the model accurately identifies notes written by these high-workload physicians, and also flags notes written in other high-fatigue settings: on overnight shifts, and after high patient volumes. Model predictions also correlate with worse decision-making on at least one important metric: yield of testing for heart attack is 18% lower with each standard deviation increase in model-predicted fatigue. Finally, the model indicates that notes written about Black and Hispanic patients have 12% and 21% higher predicted fatigue than Whites -- larger than overnight vs. daytime differences. These results have an important implication for large language models (LLMs). Our model indicates that fatigued doctors write more predictable notes. Perhaps unsurprisingly, because word prediction is the core of how LLMs work, we find that LLM-written notes have 17% higher predicted fatigue than real physicians' notes. This indicates that LLMs may introduce distortions in generated text that are not yet fully understood.

摘要: 医生为病人写便条。在这样做的过程中，他们透露了很多关于自己的信息。使用129,228次急诊室就诊的数据，我们训练了一个模型来识别疲惫的医生写的笔记--那些在之前的7天中工作了5天或更多的医生。在坚持设置中，该模型准确地识别这些高工作量医生所写的笔记，并标记在其他高疲劳度环境中编写的笔记：在夜间轮班时，以及在高病人量之后。模型预测还与至少一个重要指标上的较差决策相关：模型预测疲劳的标准差每增加一次，心脏病发作测试的收益率就会降低18%。最后，该模型表明，写给黑人和西班牙裔患者的纸条比白人分别高出12%和21%的预期疲劳感--比夜间和白天的差异更大。这些结果对大型语言模型(LLM)具有重要的意义。我们的模型表明，疲惫的医生写的笔记更容易预测。也许并不令人惊讶的是，因为单词预测是LLMS工作原理的核心，我们发现LLM写的笔记比真正的医生笔记预测的疲劳感高17%。这表明LLMS可能会在生成的文本中引入尚未完全理解的扭曲。



## **26. Tree of Attacks: Jailbreaking Black-Box LLMs Automatically**

攻击之树：自动越狱黑盒LLMS cs.LG

An implementation of the presented method is available at  https://github.com/RICommunity/TAP

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.02119v1) [paper-pdf](http://arxiv.org/pdf/2312.02119v1)

**Authors**: Anay Mehrotra, Manolis Zampetakis, Paul Kassianik, Blaine Nelson, Hyrum Anderson, Yaron Singer, Amin Karbasi

**Abstract**: While Large Language Models (LLMs) display versatile functionality, they continue to generate harmful, biased, and toxic content, as demonstrated by the prevalence of human-designed jailbreaks. In this work, we present Tree of Attacks with Pruning (TAP), an automated method for generating jailbreaks that only requires black-box access to the target LLM. TAP utilizes an LLM to iteratively refine candidate (attack) prompts using tree-of-thoughts reasoning until one of the generated prompts jailbreaks the target. Crucially, before sending prompts to the target, TAP assesses them and prunes the ones unlikely to result in jailbreaks. Using tree-of-thought reasoning allows TAP to navigate a large search space of prompts and pruning reduces the total number of queries sent to the target. In empirical evaluations, we observe that TAP generates prompts that jailbreak state-of-the-art LLMs (including GPT4 and GPT4-Turbo) for more than 80% of the prompts using only a small number of queries. This significantly improves upon the previous state-of-the-art black-box method for generating jailbreaks.

摘要: 虽然大型语言模型(LLM)显示了多功能，但它们继续产生有害、有偏见和有毒的内容，人类设计的越狱事件的流行就证明了这一点。在这项工作中，我们提出了带修剪的攻击树(TAP)，这是一种自动生成越狱的方法，只需要通过黑盒访问目标LLM。TAP利用LLM使用思想树推理反复优化候选(攻击)提示，直到其中一个生成的提示越狱目标。至关重要的是，在向目标发送提示之前，TAP会对它们进行评估，并删除那些不太可能导致越狱的提示。使用思维树推理允许TAP导航大的提示搜索空间，并进行修剪以减少发送到目标的查询总数。在经验评估中，我们观察到TAP仅使用少量查询就为80%以上的提示生成了越狱最先进的LLM(包括GPT4和GPT4-Turbo)提示。这大大改进了以前用于生成越狱的最先进的黑匣子方法。



## **27. A Survey on Large Language Model (LLM) Security and Privacy: The Good, the Bad, and the Ugly**

大型语言模型（LLM）安全和隐私调查：好，坏和丑陋 cs.CR

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.02003v1) [paper-pdf](http://arxiv.org/pdf/2312.02003v1)

**Authors**: Yifan Yao, Jinhao Duan, Kaidi Xu, Yuanfang Cai, Eric Sun, Yue Zhang

**Abstract**: Large Language Models (LLMs), such as GPT-3 and BERT, have revolutionized natural language understanding and generation. They possess deep language comprehension, human-like text generation capabilities, contextual awareness, and robust problem-solving skills, making them invaluable in various domains (e.g., search engines, customer support, translation). In the meantime, LLMs have also gained traction in the security community, revealing security vulnerabilities and showcasing their potential in security-related tasks. This paper explores the intersection of LLMs with security and privacy. Specifically, we investigate how LLMs positively impact security and privacy, potential risks and threats associated with their use, and inherent vulnerabilities within LLMs. Through a comprehensive literature review, the paper categorizes findings into "The Good" (beneficial LLM applications), "The Bad" (offensive applications), and "The Ugly" (vulnerabilities and their defenses). We have some interesting findings. For example, LLMs have proven to enhance code and data security, outperforming traditional methods. However, they can also be harnessed for various attacks (particularly user-level attacks) due to their human-like reasoning abilities. We have identified areas that require further research efforts. For example, research on model and parameter extraction attacks is limited and often theoretical, hindered by LLM parameter scale and confidentiality. Safe instruction tuning, a recent development, requires more exploration. We hope that our work can shed light on the LLMs' potential to both bolster and jeopardize cybersecurity.

摘要: 大型语言模型(LLM)，如GPT-3和BERT，已经彻底改变了自然语言的理解和生成。它们具有深刻的语言理解能力、类似人类的文本生成能力、上下文意识和强大的解决问题的技能，使它们在各个领域(例如，搜索引擎、客户支持、翻译)具有无价的价值。与此同时，LLMS也在安全界获得了吸引力，揭示了安全漏洞，并在与安全相关的任务中展示了它们的潜力。本文探讨了LLMS与安全和隐私的交集。具体地说，我们调查了LLM如何对安全和隐私产生积极影响，与其使用相关的潜在风险和威胁，以及LLM中的固有漏洞。通过全面的文献回顾，本文将研究结果分为“好的”(有益的LLM应用程序)、“坏的”(攻击性应用程序)和“丑陋的”(漏洞及其防御)。我们有一些有趣的发现。例如，LLM已被证明可以增强代码和数据的安全性，表现优于传统方法。然而，由于它们类似人类的推理能力，它们也可以被利用来进行各种攻击(特别是用户级的攻击)。我们已经确定了需要进一步研究的领域。例如，对模型和参数提取攻击的研究是有限的，而且往往是理论上的，受到LLM参数规模和保密性的阻碍。安全的指令调优是一个新的发展，需要更多的探索。我们希望我们的工作能够揭示小岛屿发展中国家加强和危害网络安全的潜力。



## **28. Intrusion Detection System with Machine Learning and Multiple Datasets**

基于机器学习和多数据集的入侵检测系统 cs.CR

12 pages, 2 figures, 2 tables

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.01941v1) [paper-pdf](http://arxiv.org/pdf/2312.01941v1)

**Authors**: Haiyan Xuan, Mohith Manohar

**Abstract**: As Artificial Intelligence (AI) technologies continue to gain traction in the modern-day world, they ultimately pose an immediate threat to current cybersecurity systems via exploitative methods. Prompt engineering is a relatively new field that explores various prompt designs that can hijack large language models (LLMs). If used by an unethical attacker, it can enable an AI system to offer malicious insights and code to them. In this paper, an enhanced intrusion detection system (IDS) that utilizes machine learning (ML) and hyperparameter tuning is explored, which can improve a model's performance in terms of accuracy and efficacy. Ultimately, this improved system can be used to combat the attacks made by unethical hackers. A standard IDS is solely configured with pre-configured rules and patterns; however, with the utilization of machine learning, implicit and different patterns can be generated through the models' hyperparameter settings and parameters. In addition, the IDS will be equipped with multiple datasets so that the accuracy of the models improves. We evaluate the performance of multiple ML models and their respective hyperparameter settings through various metrics to compare their results to other models and past research work. The results of the proposed multi-dataset integration method yielded an accuracy score of 99.9% when equipped with the XGBoost and random forest classifiers and RandomizedSearchCV hyperparameter technique.

摘要: 随着人工智能(AI)技术在现代世界继续获得牵引力，它们最终通过剥削性方法对当前的网络安全系统构成直接威胁。提示工程是一个相对较新的领域，它探索可以劫持大型语言模型(LLM)的各种提示设计。如果被不道德的攻击者使用，它可以使人工智能系统向他们提供恶意的见解和代码。本文探讨了一种利用机器学习和超参数调整的增强型入侵检测系统，它可以在准确性和有效性方面提高模型的性能。最终，这个改进的系统可以用来打击不道德的黑客进行的攻击。一个标准的入侵检测系统只配置了预先配置的规则和模式，但是，利用机器学习，可以通过模型的超参数设置和参数来生成隐式和不同的模式。此外，入侵检测系统将配备多个数据集，以提高模型的精度。我们通过不同的度量来评估多个最大似然模型及其各自的超参数设置的性能，并将它们的结果与其他模型和过去的研究工作进行比较。在使用XGBoost和随机森林分类器以及RandomizedSearchCV超参数技术的情况下，所提出的多数据集集成方法的准确率为99.9%。



## **29. InstructTA: Instruction-Tuned Targeted Attack for Large Vision-Language Models**

InstructTA：针对大型视觉语言模型的指令调整的定向攻击 cs.CV

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.01886v1) [paper-pdf](http://arxiv.org/pdf/2312.01886v1)

**Authors**: Xunguang Wang, Zhenlan Ji, Pingchuan Ma, Zongjie Li, Shuai Wang

**Abstract**: Large vision-language models (LVLMs) have demonstrated their incredible capability in image understanding and response generation. However, this rich visual interaction also makes LVLMs vulnerable to adversarial examples. In this paper, we formulate a novel and practical gray-box attack scenario that the adversary can only access the visual encoder of the victim LVLM, without the knowledge of its prompts (which are often proprietary for service providers and not publicly available) and its underlying large language model (LLM). This practical setting poses challenges to the cross-prompt and cross-model transferability of targeted adversarial attack, which aims to confuse the LVLM to output a response that is semantically similar to the attacker's chosen target text. To this end, we propose an instruction-tuned targeted attack (dubbed InstructTA) to deliver the targeted adversarial attack on LVLMs with high transferability. Initially, we utilize a public text-to-image generative model to "reverse" the target response into a target image, and employ GPT-4 to infer a reasonable instruction $\boldsymbol{p}^\prime$ from the target response. We then form a local surrogate model (sharing the same visual encoder with the victim LVLM) to extract instruction-aware features of an adversarial image example and the target image, and minimize the distance between these two features to optimize the adversarial example. To further improve the transferability, we augment the instruction $\boldsymbol{p}^\prime$ with instructions paraphrased from an LLM. Extensive experiments demonstrate the superiority of our proposed method in targeted attack performance and transferability.

摘要: 大型视觉语言模型在图像理解和响应生成方面表现出了令人难以置信的能力。然而，这种丰富的视觉交互也使LVLM容易受到对抗性例子的攻击。本文提出了一种新颖实用的灰盒攻击方案，即攻击者只能访问受害者LVLM的可视编码器，而不知道其提示(通常是服务提供商的专有提示，而不是公开可用的)及其底层的大型语言模型(LLM)。这一实际设置对目标对抗性攻击的跨提示和跨模型可转移性提出了挑战，其目的是混淆LVLM以输出与攻击者选择的目标文本在语义上相似的响应。为此，我们提出了一种指令调谐的定向攻击(InstructTA)，对具有高可转移性的LVLMS进行定向对抗性攻击。首先，我们利用一个公开的文本到图像的生成模型将目标响应“反转”成目标图像，并使用GPT-4从目标响应中推断出合理的指令符号。然后，我们形成一个局部代理模型(与受害者LVLM共享相同的视觉编码器)来提取对抗性图像示例和目标图像的指令感知特征，并最小化这两个特征之间的距离以优化对抗性示例。为了进一步提高可转移性，我们用转译自LLM的指令扩充了指令$\boldSymbol{p}^\Prime$。大量实验证明了该方法在目标攻击性能和可转移性方面的优越性。



## **30. Warfare:Breaking the Watermark Protection of AI-Generated Content**

战争：打破人工智能生成内容的水印保护 cs.CV

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2310.07726v2) [paper-pdf](http://arxiv.org/pdf/2310.07726v2)

**Authors**: Guanlin Li, Yifei Chen, Jie Zhang, Jiwei Li, Shangwei Guo, Tianwei Zhang

**Abstract**: AI-Generated Content (AIGC) is gaining great popularity, with many emerging commercial services and applications. These services leverage advanced generative models, such as latent diffusion models and large language models, to generate creative content (e.g., realistic images and fluent sentences) for users. The usage of such generated content needs to be highly regulated, as the service providers need to ensure the users do not violate the usage policies (e.g., abuse for commercialization, generating and distributing unsafe content). A promising solution to achieve this goal is watermarking, which adds unique and imperceptible watermarks on the content for service verification and attribution. Numerous watermarking approaches have been proposed recently. However, in this paper, we show that an adversary can easily break these watermarking mechanisms. Specifically, we consider two possible attacks. (1) Watermark removal: the adversary can easily erase the embedded watermark from the generated content and then use it freely bypassing the regulation of the service provider. (2) Watermark forging: the adversary can create illegal content with forged watermarks from another user, causing the service provider to make wrong attributions. We propose Warfare, a unified methodology to achieve both attacks in a holistic way. The key idea is to leverage a pre-trained diffusion model for content processing and a generative adversarial network for watermark removal or forging. We evaluate Warfare on different datasets and embedding setups. The results prove that it can achieve high success rates while maintaining the quality of the generated content. Compared to existing diffusion model-based attacks, Warfare is 5,050~11,000x faster.

摘要: 人工智能生成的内容(AIGC)越来越受欢迎，出现了许多新兴的商业服务和应用程序。这些服务利用高级生成模型，如潜在扩散模型和大型语言模型，为用户生成创造性内容(例如，逼真的图像和流畅的句子)。这种生成的内容的使用需要受到严格的监管，因为服务提供商需要确保用户不违反使用策略(例如，滥用以商业化、生成和分发不安全的内容)。实现这一目标的一个有前途的解决方案是水印，它在内容上添加唯一且不可察觉的水印，用于服务验证和归属。最近，人们提出了许多水印方法。然而，在本文中，我们证明了攻击者可以很容易地破解这些水印机制。具体地说，我们考虑两种可能的攻击。(1)水印去除：攻击者可以很容易地从生成的内容中删除嵌入的水印，然后绕过服务提供商的监管自由使用。(2)水印伪造：对手可以利用来自其他用户的伪造水印创建非法内容，导致服务提供商做出错误的归属。我们提出战争，一种统一的方法论，以整体的方式实现这两种攻击。其关键思想是利用预先训练的扩散模型来进行内容处理，并利用生成性对抗网络来去除或伪造水印。我们对不同的数据集和嵌入设置进行了战争评估。实验结果表明，该算法在保证生成内容质量的同时，具有较高的成功率。与现有的基于扩散模型的攻击相比，战争的速度要快5050~11000倍。



## **31. Unleashing Cheapfakes through Trojan Plugins of Large Language Models**

通过大型语言模型特洛伊木马插件释放Cheapfake cs.CR

**SubmitDate**: 2023-12-01    [abs](http://arxiv.org/abs/2312.00374v1) [paper-pdf](http://arxiv.org/pdf/2312.00374v1)

**Authors**: Tian Dong, Guoxing Chen, Shaofeng Li, Minhui Xue, Rayne Holland, Yan Meng, Zhen Liu, Haojin Zhu

**Abstract**: Open-source Large Language Models (LLMs) have recently gained popularity because of their comparable performance to proprietary LLMs. To efficiently fulfill domain-specialized tasks, open-source LLMs can be refined, without expensive accelerators, using low-rank adapters. However, it is still unknown whether low-rank adapters can be exploited to control LLMs. To address this gap, we demonstrate that an infected adapter can induce, on specific triggers, an LLM to output content defined by an adversary and to even maliciously use tools. To train a Trojan adapter, we propose two novel attacks, POLISHED and FUSION, that improve over prior approaches. POLISHED uses LLM-enhanced paraphrasing to polish benchmark poisoned datasets. In contrast, in the absence of a dataset, FUSION leverages an over-poisoning procedure to transform a benign adaptor. Our experiments validate that our attacks provide higher attack effectiveness than the baseline and, for the purpose of attracting downloads, preserves or improves the adapter's utility. Finally, we provide two case studies to demonstrate that the Trojan adapter can lead a LLM-powered autonomous agent to execute unintended scripts or send phishing emails. Our novel attacks represent the first study of supply chain threats for LLMs through the lens of Trojan plugins.

摘要: 开源的大型语言模型(LLM)最近越来越受欢迎，因为它们的性能可以与专有的LLM相媲美。为了高效地完成领域专门化任务，可以使用低级别适配器对开源LLM进行提炼，而无需使用昂贵的加速器。然而，是否可以利用低阶适配器来控制LLM仍然是未知的。为了弥补这一漏洞，我们演示了受感染的适配器可以在特定触发下诱导LLM输出由对手定义的内容，甚至恶意使用工具。为了训练木马适配器，我们提出了两种新的攻击方法，磨光攻击和融合攻击，它们比以前的方法有所改进。波兰德使用LLM增强的释义来抛光基准有毒数据集。相比之下，在没有数据集的情况下，Fusion利用过度中毒的程序来转换良性适配器。我们的实验验证了我们的攻击提供了比基线更高的攻击效率，并且为了吸引下载的目的，保留或提高了适配器的实用性。最后，我们提供了两个案例研究来演示特洛伊木马适配器可以导致LLM驱动的自主代理执行意外脚本或发送钓鱼电子邮件。我们的新型攻击首次通过特洛伊木马插件的镜头研究了LLM的供应链威胁。



## **32. Ignore This Title and HackAPrompt: Exposing Systemic Vulnerabilities of LLMs through a Global Scale Prompt Hacking Competition**

忽略这个标题和HackAPrompt：通过全球规模的即时黑客竞赛揭露LLMS的系统性漏洞 cs.CR

34 pages, 8 figures Codebase:  https://github.com/PromptLabs/hackaprompt Dataset:  https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset/blob/main/README.md  Playground: https://huggingface.co/spaces/hackaprompt/playground

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2311.16119v2) [paper-pdf](http://arxiv.org/pdf/2311.16119v2)

**Authors**: Sander Schulhoff, Jeremy Pinto, Anaum Khan, Louis-François Bouchard, Chenglei Si, Svetlina Anati, Valen Tagliabue, Anson Liu Kost, Christopher Carnahan, Jordan Boyd-Graber

**Abstract**: Large Language Models (LLMs) are deployed in interactive contexts with direct user engagement, such as chatbots and writing assistants. These deployments are vulnerable to prompt injection and jailbreaking (collectively, prompt hacking), in which models are manipulated to ignore their original instructions and follow potentially malicious ones. Although widely acknowledged as a significant security threat, there is a dearth of large-scale resources and quantitative studies on prompt hacking. To address this lacuna, we launch a global prompt hacking competition, which allows for free-form human input attacks. We elicit 600K+ adversarial prompts against three state-of-the-art LLMs. We describe the dataset, which empirically verifies that current LLMs can indeed be manipulated via prompt hacking. We also present a comprehensive taxonomical ontology of the types of adversarial prompts.

摘要: 大型语言模型（LLM）部署在具有直接用户参与的交互式环境中，例如聊天机器人和写作助手。这些部署容易受到提示注入和越狱（统称为提示黑客攻击）的攻击，其中模型被操纵以忽略其原始指令并遵循潜在的恶意指令。虽然被广泛认为是一个重大的安全威胁，但缺乏关于即时黑客攻击的大规模资源和定量研究。为了解决这个问题，我们发起了一个全球性的即时黑客竞赛，允许自由形式的人类输入攻击。我们针对三个最先进的LLM引出600K+对抗性提示。我们描述的数据集，经验验证，目前的LLM确实可以通过即时黑客操作。我们还提出了一个全面的分类本体的类型的对抗性提示。



## **33. Devising and Detecting Phishing: Large Language Models vs. Smaller Human Models**

设计和检测网络钓鱼：大型语言模型与较小的人类模型 cs.CR

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2308.12287v2) [paper-pdf](http://arxiv.org/pdf/2308.12287v2)

**Authors**: Fredrik Heiding, Bruce Schneier, Arun Vishwanath, Jeremy Bernstein, Peter S. Park

**Abstract**: AI programs, built using large language models, make it possible to automatically create phishing emails based on a few data points about a user. They stand in contrast to traditional phishing emails that hackers manually design using general rules gleaned from experience. The V-Triad is an advanced set of rules for manually designing phishing emails to exploit our cognitive heuristics and biases. In this study, we compare the performance of phishing emails created automatically by GPT-4 and manually using the V-Triad. We also combine GPT-4 with the V-Triad to assess their combined potential. A fourth group, exposed to generic phishing emails, was our control group. We utilized a factorial approach, sending emails to 112 randomly selected participants recruited for the study. The control group emails received a click-through rate between 19-28%, the GPT-generated emails 30-44%, emails generated by the V-Triad 69-79%, and emails generated by GPT and the V-Triad 43-81%. Each participant was asked to explain why they pressed or did not press a link in the email. These answers often contradict each other, highlighting the need for personalized content. The cues that make one person avoid phishing emails make another person fall for them. Next, we used four popular large language models (GPT, Claude, PaLM, and LLaMA) to detect the intention of phishing emails and compare the results to human detection. The language models demonstrated a strong ability to detect malicious intent, even in non-obvious phishing emails. They sometimes surpassed human detection, although often being slightly less accurate than humans. Finally, we make an analysis of the economic aspects of AI-enabled phishing attacks, showing how large language models can increase the incentives of phishing and spear phishing by reducing their costs.

摘要: 使用大型语言模型构建的人工智能程序可以根据用户的几个数据点自动创建钓鱼电子邮件。它们与黑客使用从经验中收集的一般规则手动设计的传统钓鱼电子邮件形成了鲜明对比。V-Triad是一套高级规则，用于手动设计钓鱼电子邮件，以利用我们的认知启发式和偏见。在这项研究中，我们比较了GPT-4自动创建的钓鱼电子邮件和使用V-Triad手动创建的钓鱼电子邮件的性能。我们还将GPT-4与V-Triad相结合，以评估它们的组合潜力。接触普通钓鱼邮件的第四组是我们的控制组。我们使用了析因分析的方法，向112名随机挑选的参与者发送电子邮件，这些参与者被招募参加这项研究。对照组电子邮件的点击率为19%-28%，GPT生成的电子邮件的点击率为30%-44%，V-Triad生成的电子邮件的点击率为69%-79%，GPT和V-Triad生成的电子邮件的点击率为43%-81%。每个参与者都被要求解释为什么他们按下或没有按下电子邮件中的链接。这些答案往往相互矛盾，突显了对个性化内容的需求。让一个人避免钓鱼电子邮件的暗示会让另一个人爱上他们。接下来，我们使用四个流行的大型语言模型(GPT、Claude、Palm和Llama)来检测钓鱼电子邮件的意图，并将结果与人类检测进行比较。语言模型显示出强大的检测恶意意图的能力，即使是在不明显的网络钓鱼电子邮件中也是如此。它们有时会超过人类的检测，尽管它们的准确度往往略低于人类。最后，我们对人工智能钓鱼攻击的经济方面进行了分析，展示了大型语言模型如何通过降低钓鱼和鱼叉式钓鱼的成本来增加它们的诱因。



## **34. Locally Differentially Private Document Generation Using Zero Shot Prompting**

基于零镜头提示的局部差异私有文档生成方法 cs.CL

Accepted at EMNLP 2023 (Findings)

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2310.16111v2) [paper-pdf](http://arxiv.org/pdf/2310.16111v2)

**Authors**: Saiteja Utpala, Sara Hooker, Pin Yu Chen

**Abstract**: Numerous studies have highlighted the privacy risks associated with pretrained large language models. In contrast, our research offers a unique perspective by demonstrating that pretrained large language models can effectively contribute to privacy preservation. We propose a locally differentially private mechanism called DP-Prompt, which leverages the power of pretrained large language models and zero-shot prompting to counter author de-anonymization attacks while minimizing the impact on downstream utility. When DP-Prompt is used with a powerful language model like ChatGPT (gpt-3.5), we observe a notable reduction in the success rate of de-anonymization attacks, showing that it surpasses existing approaches by a considerable margin despite its simpler design. For instance, in the case of the IMDB dataset, DP-Prompt (with ChatGPT) perfectly recovers the clean sentiment F1 score while achieving a 46\% reduction in author identification F1 score against static attackers and a 26\% reduction against adaptive attackers. We conduct extensive experiments across six open-source large language models, ranging up to 7 billion parameters, to analyze various effects of the privacy-utility tradeoff.

摘要: 许多研究都强调了与预先训练的大型语言模型相关的隐私风险。相比之下，我们的研究提供了一个独特的视角，证明了预先训练的大型语言模型可以有效地有助于隐私保护。我们提出了一种称为DP-Prompt的局部差异私有机制，该机制利用预先训练的大型语言模型和零镜头提示的能力来对抗作者去匿名化攻击，同时最小化对下游效用的影响。当DP-Prompt与ChatGPT(GPT-3.5)等强大的语言模型一起使用时，我们观察到去匿名化攻击的成功率显著下降，表明尽管它的设计更简单，但它在相当大程度上超过了现有的方法。例如，在IMDB数据集的情况下，DP-Prompt(使用ChatGPT)完美地恢复了干净的情感F1分数，同时在针对静态攻击者的作者识别F1分数和针对自适应攻击者的F1分数分别减少了46%和26%。我们在六个开放源码的大型语言模型上进行了广泛的实验，范围多达70亿个参数，以分析隐私-效用权衡的各种影响。



## **35. Towards Safer Generative Language Models: A Survey on Safety Risks, Evaluations, and Improvements**

走向更安全的生成性语言模型：安全风险、评估和改进的综述 cs.AI

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2302.09270v3) [paper-pdf](http://arxiv.org/pdf/2302.09270v3)

**Authors**: Jiawen Deng, Jiale Cheng, Hao Sun, Zhexin Zhang, Minlie Huang

**Abstract**: As generative large model capabilities advance, safety concerns become more pronounced in their outputs. To ensure the sustainable growth of the AI ecosystem, it's imperative to undertake a holistic evaluation and refinement of associated safety risks. This survey presents a framework for safety research pertaining to large models, delineating the landscape of safety risks as well as safety evaluation and improvement methods. We begin by introducing safety issues of wide concern, then delve into safety evaluation methods for large models, encompassing preference-based testing, adversarial attack approaches, issues detection, and other advanced evaluation methods. Additionally, we explore the strategies for enhancing large model safety from training to deployment, highlighting cutting-edge safety approaches for each stage in building large models. Finally, we discuss the core challenges in advancing towards more responsible AI, including the interpretability of safety mechanisms, ongoing safety issues, and robustness against malicious attacks. Through this survey, we aim to provide clear technical guidance for safety researchers and encourage further study on the safety of large models.

摘要: 随着产生式大型模型能力的进步，安全问题在其输出中变得更加明显。为了确保人工智能生态系统的可持续增长，必须对相关安全风险进行全面评估和细化。本调查提出了与大型模型相关的安全研究框架，描绘了安全风险的图景以及安全评估和改进方法。我们首先介绍广泛关注的安全问题，然后深入研究大型模型的安全评估方法，包括基于偏好的测试、对抗性攻击方法、问题检测和其他高级评估方法。此外，我们还探讨了从培训到部署增强大型模型安全性的策略，重点介绍了构建大型模型的每个阶段的前沿安全方法。最后，我们讨论了向更负责任的人工智能发展的核心挑战，包括安全机制的可解释性、持续的安全问题和针对恶意攻击的健壮性。通过这次调查，我们旨在为安全研究人员提供明确的技术指导，并鼓励进一步研究大型模型的安全性。



## **36. Leveraging a Randomized Key Matrix to Enhance the Security of Symmetric Substitution Ciphers**

利用随机化密钥矩阵提高对称替换密码的安全性 cs.CR

In Proceedings of the 10th IEEE Asia-Pacific Conference on Computer  Science and Data Engineering 2023 (CSDE)

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.18085v1) [paper-pdf](http://arxiv.org/pdf/2311.18085v1)

**Authors**: Shubham Gandhi, Om Khare, Mihika Dravid, Mihika Sanghvi, Sunil Mane, Aadesh Gajaralwar, Saloni Gandhi

**Abstract**: An innovative strategy to enhance the security of symmetric substitution ciphers is presented, through the implementation of a randomized key matrix suitable for various file formats, including but not limited to binary and text files. Despite their historical relevance, symmetric substitution ciphers have been limited by vulnerabilities to cryptanalytic methods like frequency analysis and known plaintext attacks. The aim of our research is to mitigate these vulnerabilities by employing a polyalphabetic substitution strategy that incorporates a distinct randomized key matrix. This matrix plays a pivotal role in generating a unique random key, comprising characters, encompassing both uppercase and lowercase letters, numeric, and special characters, to derive the corresponding ciphertext. The effectiveness of the proposed methodology in enhancing the security of conventional substitution methods for file encryption and decryption is supported by comprehensive testing and analysis, which encompass computational speed, frequency analysis, keyspace examination, Kasiski test, entropy analysis, and the utilization of a large language model.

摘要: 提出了一种增强对称替换密码安全性的创新策略，通过实现适用于各种文件格式的随机化密钥矩阵，包括但不限于二进制和文本文件。尽管对称替换密码具有历史相关性，但它一直受到频率分析和已知明文攻击等密码分析方法的漏洞的限制。我们研究的目的是通过采用多字母替换策略来缓解这些漏洞，该策略结合了一个独特的随机密钥矩阵。该矩阵在生成唯一的随机密钥方面起着关键作用，该密钥包括包含大小写字母、数字和特殊字符的字符，以得出相应的密文。通过全面的测试和分析，包括计算速度、频率分析、密钥空间检查、Kasiski测试、熵分析和大型语言模型的使用，支持了所提出的方法在增强传统文件加密和解密替代方法的安全性方面的有效性。



## **37. SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks**

SmoothLLM：保护大型语言模型免受越狱攻击 cs.LG

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2310.03684v3) [paper-pdf](http://arxiv.org/pdf/2310.03684v3)

**Authors**: Alexander Robey, Eric Wong, Hamed Hassani, George J. Pappas

**Abstract**: Despite efforts to align large language models (LLMs) with human values, widely-used LLMs such as GPT, Llama, Claude, and PaLM are susceptible to jailbreaking attacks, wherein an adversary fools a targeted LLM into generating objectionable content. To address this vulnerability, we propose SmoothLLM, the first algorithm designed to mitigate jailbreaking attacks on LLMs. Based on our finding that adversarially-generated prompts are brittle to character-level changes, our defense first randomly perturbs multiple copies of a given input prompt, and then aggregates the corresponding predictions to detect adversarial inputs. SmoothLLM reduces the attack success rate on numerous popular LLMs to below one percentage point, avoids unnecessary conservatism, and admits provable guarantees on attack mitigation. Moreover, our defense uses exponentially fewer queries than existing attacks and is compatible with any LLM. Our code is publicly available at the following link: https://github.com/arobey1/smooth-llm.

摘要: 尽管努力使大型语言模型(LLM)与人类价值观保持一致，但GPT、Llama、Claude和Palm等广泛使用的LLM容易受到越狱攻击，即对手欺骗目标LLM生成令人反感的内容。为了解决这一漏洞，我们提出了SmoothLLM，这是第一个旨在缓解对LLM的越狱攻击的算法。基于我们的发现，对抗性生成的提示对字符级别的变化很脆弱，我们的防御首先随机扰动给定输入提示的多个副本，然后聚合相应的预测来检测对抗性输入。SmoothLLM将许多流行的LLM的攻击成功率降低到1个百分点以下，避免了不必要的保守主义，并承认了对攻击缓解的可证明保证。此外，我们的防御使用的查询比现有攻击少得多，并且与任何LLM兼容。我们的代码可通过以下链接公开获得：https://github.com/arobey1/smooth-llm.



## **38. Query-Relevant Images Jailbreak Large Multi-Modal Models**

与查询相关的图像越狱大型多模式模型 cs.CV

Technique report

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17600v1) [paper-pdf](http://arxiv.org/pdf/2311.17600v1)

**Authors**: Xin Liu, Yichen Zhu, Yunshi Lan, Chao Yang, Yu Qiao

**Abstract**: Warning: This paper contains examples of harmful language and images, and reader discretion is recommended. The security concerns surrounding Large Language Models (LLMs) have been extensively explored, yet the safety of Large Multi-Modal Models (LMMs) remains understudied. In our study, we present a novel visual prompt attack that exploits query-relevant images to jailbreak the open-source LMMs. Our method creates a composite image from one image generated by diffusion models and another that displays the text as typography, based on keywords extracted from a malicious query. We show LLMs can be easily attacked by our approach, even if the employed Large Language Models are safely aligned. To evaluate the extent of this vulnerability in open-source LMMs, we have compiled a substantial dataset encompassing 13 scenarios with a total of 5,040 text-image pairs, using our presented attack technique. Our evaluation of 12 cutting-edge LMMs using this dataset shows the vulnerability of existing multi-modal models on adversarial attacks. This finding underscores the need for a concerted effort to strengthen and enhance the safety measures of open-source LMMs against potential malicious exploits. The resource is available at \href{this https URL}{https://github.com/isXinLiu/MM-SafetyBench}.

摘要: 警告：本文包含有害语言和图片的例子，建议读者自行决定。围绕大型语言模型(LLM)的安全问题已经得到了广泛的研究，但大型多模式模型(LMM)的安全性仍未得到充分研究。在我们的研究中，我们提出了一种新的视觉提示攻击，利用与查询相关的图像来越狱开源的LMM。我们的方法从一个由扩散模型生成的图像和另一个基于从恶意查询中提取的关键字将文本显示为排版的图像创建合成图像。我们表明，即使所使用的大型语言模型安全地对齐，LLM也可以很容易地被我们的方法攻击。为了评估这一漏洞在开源LMM中的程度，我们使用我们提出的攻击技术编制了一个包含13个场景的大量数据集，总共有5,040个文本-图像对。我们使用这个数据集对12个尖端的LMM进行了评估，表明了现有的多模式模型在对抗攻击时的脆弱性。这一发现强调了需要共同努力，加强和改进开放源码LMM的安全措施，以防范潜在的恶意利用。该资源位于\href{此HTTPS URL}{https://github.com/isXinLiu/MM-SafetyBench}.



## **39. Unveiling the Implicit Toxicity in Large Language Models**

揭示大型语言模型中的隐含毒性 cs.CL

EMNLP 2023 Main Conference

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17391v1) [paper-pdf](http://arxiv.org/pdf/2311.17391v1)

**Authors**: Jiaxin Wen, Pei Ke, Hao Sun, Zhexin Zhang, Chengfei Li, Jinfeng Bai, Minlie Huang

**Abstract**: The open-endedness of large language models (LLMs) combined with their impressive capabilities may lead to new safety issues when being exploited for malicious use. While recent studies primarily focus on probing toxic outputs that can be easily detected with existing toxicity classifiers, we show that LLMs can generate diverse implicit toxic outputs that are exceptionally difficult to detect via simply zero-shot prompting. Moreover, we propose a reinforcement learning (RL) based attacking method to further induce the implicit toxicity in LLMs. Specifically, we optimize the language model with a reward that prefers implicit toxic outputs to explicit toxic and non-toxic ones. Experiments on five widely-adopted toxicity classifiers demonstrate that the attack success rate can be significantly improved through RL fine-tuning. For instance, the RL-finetuned LLaMA-13B model achieves an attack success rate of 90.04% on BAD and 62.85% on Davinci003. Our findings suggest that LLMs pose a significant threat in generating undetectable implicit toxic outputs. We further show that fine-tuning toxicity classifiers on the annotated examples from our attacking method can effectively enhance their ability to detect LLM-generated implicit toxic language. The code is publicly available at https://github.com/thu-coai/Implicit-Toxicity.

摘要: 大型语言模型(LLM)的开放性与其令人印象深刻的能力相结合，在被恶意利用时可能会导致新的安全问题。虽然最近的研究主要集中在探测现有毒性分类器可以很容易检测到的有毒输出，但我们发现LLMS可以产生各种隐含的有毒输出，这些输出通过简单的零射击提示特别难检测到。此外，我们还提出了一种基于强化学习(RL)的攻击方法来进一步诱导LLMS中的隐含毒性。具体地说，我们优化了语言模型，奖励它更喜欢隐含的有毒输出，而不是显式的有毒和无毒的输出。在5个广泛使用的毒性分类器上的实验表明，通过RL的微调可以显著提高攻击成功率。例如，RL微调的骆驼-13B模型在BAD上的攻击成功率为90.04%，在Davinc003上的攻击成功率为62.85%。我们的发现表明，LLMS在产生无法检测到的隐含有毒输出方面构成了重大威胁。我们进一步表明，在我们的攻击方法的标注样本上微调毒性分类器可以有效地提高它们检测LLM生成的隐含有毒语言的能力。该代码可在https://github.com/thu-coai/Implicit-Toxicity.上公开获得



## **40. Identifying and Mitigating Vulnerabilities in LLM-Integrated Applications**

识别和缓解LLM集成应用程序中的漏洞 cs.CR

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.16153v2) [paper-pdf](http://arxiv.org/pdf/2311.16153v2)

**Authors**: Fengqing Jiang, Zhangchen Xu, Luyao Niu, Boxin Wang, Jinyuan Jia, Bo Li, Radha Poovendran

**Abstract**: Large language models (LLMs) are increasingly deployed as the service backend for LLM-integrated applications such as code completion and AI-powered search. LLM-integrated applications serve as middleware to refine users' queries with domain-specific knowledge to better inform LLMs and enhance the responses. Despite numerous opportunities and benefits, LLM-integrated applications also introduce new attack surfaces. Understanding, minimizing, and eliminating these emerging attack surfaces is a new area of research. In this work, we consider a setup where the user and LLM interact via an LLM-integrated application in the middle. We focus on the communication rounds that begin with user's queries and end with LLM-integrated application returning responses to the queries, powered by LLMs at the service backend. For this query-response protocol, we identify potential vulnerabilities that can originate from the malicious application developer or from an outsider threat initiator that is able to control the database access, manipulate and poison data that are high-risk for the user. Successful exploits of the identified vulnerabilities result in the users receiving responses tailored to the intent of a threat initiator. We assess such threats against LLM-integrated applications empowered by OpenAI GPT-3.5 and GPT-4. Our empirical results show that the threats can effectively bypass the restrictions and moderation policies of OpenAI, resulting in users receiving responses that contain bias, toxic content, privacy risk, and disinformation. To mitigate those threats, we identify and define four key properties, namely integrity, source identification, attack detectability, and utility preservation, that need to be satisfied by a safe LLM-integrated application. Based on these properties, we develop a lightweight, threat-agnostic defense that mitigates both insider and outsider threats.

摘要: 大型语言模型(LLM)越来越多地被部署为LLM集成应用程序的服务后端，例如代码完成和AI支持的搜索。LLM集成的应用程序充当中间件，使用特定于领域的知识来提炼用户的查询，以更好地通知LLM并增强响应。尽管有许多机会和好处，LLM集成应用程序也带来了新的攻击面。理解、最小化和消除这些新出现的攻击面是一个新的研究领域。在这项工作中，我们考虑一种设置，其中用户和LLM通过中间的LLM集成应用进行交互。我们关注以用户查询开始，以LLM集成的应用程序返回对查询的响应的通信回合，该应用程序由服务后端的LLMS提供支持。对于此查询-响应协议，我们确定了可能源自恶意应用程序开发人员或外部威胁发起者的潜在漏洞，该外部威胁发起者能够控制数据库访问、操纵和毒化对用户具有高风险的数据。成功利用已识别的漏洞会导致用户收到针对威胁发起者意图量身定做的响应。我们评估了针对由OpenAI GPT-3.5和GPT-4支持的LLM集成应用程序的此类威胁。我们的实验结果表明，这些威胁可以有效地绕过OpenAI的限制和适度策略，导致用户收到包含偏见、有毒内容、隐私风险和虚假信息的响应。为了缓解这些威胁，我们确定并定义了四个关键属性，即完整性、来源识别、攻击可检测性和实用程序保存，这些属性需要由安全的LLM集成应用程序来满足。基于这些特性，我们开发了一种轻量级、与威胁无关的防御系统，可以同时减轻内部和外部威胁。



## **41. ZTCloudGuard: Zero Trust Context-Aware Access Management Framework to Avoid Misuse Cases in the Era of Generative AI and Cloud-based Health Information Ecosystem**

ZTCloudGuard：零信任上下文感知访问管理框架，在生成式AI和基于云的健康信息生态系统时代避免误用案例 cs.CR

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2312.02993v1) [paper-pdf](http://arxiv.org/pdf/2312.02993v1)

**Authors**: Khalid Al-hammuri, Fayez Gebali, Awos Kanan

**Abstract**: Managing access between large numbers of distributed medical devices has become a crucial aspect of modern healthcare systems, enabling the establishment of smart hospitals and telehealth infrastructure. However, as telehealth technology continues to evolve and Internet of Things (IoT) devices become more widely used, they are also becoming increasingly exposed to various types of vulnerabilities and medical errors. In healthcare information systems, about 90\% of vulnerabilities emerged from misuse cases and human errors. As a result, there is a need for additional research and development of security tools to prevent such attacks. This article proposes a zero-trust-based context-aware framework for managing access to the main components of the cloud ecosystem, including users, devices and output data. The main goal and benefit of the proposed framework is to build a scoring system to prevent or alleviate misuse cases while using distributed medical devices in cloud-based healthcare information systems. The framework has two main scoring schemas to maintain the chain of trust. First, it proposes a critical trust score based on cloud-native micro-services of authentication, encryption, logging, and authorizations. Second, creating a bond trust scoring to assess the real-time semantic and syntactic analysis of attributes stored in a healthcare information system. The analysis is based on a pre-trained machine learning model to generate the semantic and syntactic scores. The framework also takes into account regulatory compliance and user consent to create a scoring system. The advantage of this method is that it is applicable to any language and adapts to all attributes as it relies on a language model, not just a set of predefined and limited attributes. The results show a high F1 score of 93.5%, which proves that it is valid for detecting misuse cases.

摘要: 管理大量分布式医疗设备之间的访问已成为现代医疗保健系统的一个重要方面，有助于建立智能医院和远程医疗基础设施。然而，随着远程医疗技术的不断发展和物联网（IoT）设备的广泛使用，它们也越来越多地暴露于各种类型的漏洞和医疗错误。在医疗信息系统中，大约90%的漏洞来自误用案例和人为错误。因此，需要进一步研究和开发安全工具，以防止此类攻击。本文提出了一个基于零信任的上下文感知框架，用于管理对云生态系统主要组件的访问，包括用户、设备和输出数据。所提出的框架的主要目标和好处是建立一个评分系统，以防止或减轻误用的情况下，同时使用分布式医疗设备在基于云的医疗信息系统。该框架有两个主要的评分模式来维护信任链。首先，它提出了一个基于云原生微服务的关键信任得分，包括身份验证、加密、日志记录和授权。第二，创建债券信任评分，以评估存储在医疗信息系统中的属性的实时语义和句法分析。该分析基于预训练的机器学习模型来生成语义和句法得分。该框架还考虑到监管合规性和用户同意来创建评分系统。这种方法的优点是，它适用于任何语言，并适应所有属性，因为它依赖于语言模型，而不仅仅是一组预定义的和有限的属性。实验结果表明，该方法的F1值高达93.5%，证明了该方法对误用检测的有效性。



## **42. Certifying LLM Safety against Adversarial Prompting**

针对敌意提示认证LLM安全 cs.CL

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2309.02705v2) [paper-pdf](http://arxiv.org/pdf/2309.02705v2)

**Authors**: Aounon Kumar, Chirag Agarwal, Suraj Srinivas, Aaron Jiaxun Li, Soheil Feizi, Himabindu Lakkaraju

**Abstract**: Large language models (LLMs) released for public use incorporate guardrails to ensure their output is safe, often referred to as "model alignment." An aligned language model should decline a user's request to produce harmful content. However, such safety measures are vulnerable to adversarial attacks, which add maliciously designed token sequences to a harmful prompt to bypass the model's safety guards. In this work, we introduce erase-and-check, the first framework to defend against adversarial prompts with verifiable safety guarantees. We defend against three attack modes: i) adversarial suffix, which appends an adversarial sequence at the end of the prompt; ii) adversarial insertion, where the adversarial sequence is inserted anywhere in the middle of the prompt; and iii) adversarial infusion, where adversarial tokens are inserted at arbitrary positions in the prompt, not necessarily as a contiguous block. Our experimental results demonstrate that this procedure can obtain strong certified safety guarantees on harmful prompts while maintaining good empirical performance on safe prompts. For example, against adversarial suffixes of length 20, it certifiably detects 92% of harmful prompts and labels 94% of safe prompts correctly using the open-source language model Llama 2 as the safety filter. We further improve the filter's performance, in terms of accuracy and speed, by replacing Llama 2 with a DistilBERT safety classifier fine-tuned on safe and harmful prompts. Additionally, we propose two efficient empirical defenses: i) RandEC, a randomized version of erase-and-check that evaluates the safety filter on a small subset of the erased subsequences, and ii) GradEC, a gradient-based version that optimizes the erased tokens to remove the adversarial sequence. The code for our experiments is available at https://github.com/aounon/certified-llm-safety.

摘要: 发布给公众使用的大型语言模型(LLM)包括护栏，以确保其输出是安全的，通常被称为“模型对齐”。统一的语言模型应该拒绝用户制作有害内容的请求。然而，这样的安全措施很容易受到敌意攻击，这些攻击会将恶意设计的令牌序列添加到有害的提示中，以绕过模型的安全警卫。在这项工作中，我们引入了Erase-and-Check，这是第一个通过可验证的安全保证来防御敌意提示的框架。我们防御三种攻击模式：i)对抗性后缀，其在提示的末尾附加对抗性序列；ii)对抗性插入，其中对抗性序列被插入在提示中间的任何位置；以及iii)对抗性注入，其中对抗性标记被插入在提示中的任意位置，而不一定作为连续的块。实验结果表明，该方法在对安全提示保持较好经验性能的同时，对有害提示具有较强的认证安全保障。例如，对于长度为20的敌意后缀，它使用开源语言模型Llama 2作为安全过滤器，可以正确检测92%的有害提示和94%的安全提示。我们用DistilBERT安全分类器替换了Llama 2，在精度和速度方面进一步提高了过滤器的性能，该分类器根据安全和有害的提示进行了微调。此外，我们提出了两个有效的经验防御：i)RandEC，一个随机版本的Erase-and-Check，评估被擦除的子序列的一小部分上的安全过滤器；以及ii)Gradec，一个基于梯度的版本，优化被擦除的令牌以去除敌对序列。我们实验的代码可以在https://github.com/aounon/certified-llm-safety.上找到



## **43. C-ITS Environment Modeling and Attack Modeling**

C-ITS环境建模与攻击建模 cs.CR

in Korean Language, 14 Figures, 15 Pages

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2311.14327v2) [paper-pdf](http://arxiv.org/pdf/2311.14327v2)

**Authors**: Jaewoong Choi, Min Geun Song, Hyosun Lee, Chaeyeon Sagong, Sangbeom Park, Jaesung Lee, Jeong Do Yoo, Huy Kang Kim

**Abstract**: As technology advances, cities are evolving into smart cities, with the ability to process large amounts of data and the increasing complexity and diversification of various elements within urban areas. Among the core systems of a smart city is the Cooperative-Intelligent Transport Systems (C-ITS). C-ITS is a system where vehicles provide real-time information to drivers about surrounding traffic conditions, sudden stops, falling objects, and other accident risks through roadside base stations. It consists of road infrastructure, C-ITS centers, and vehicle terminals. However, as smart cities integrate many elements through networks and electronic control, they are susceptible to cybersecurity issues. In the case of cybersecurity problems in C-ITS, there is a significant risk of safety issues arising. This technical document aims to model the C-ITS environment and the services it provides, with the purpose of identifying the attack surface where security incidents could occur in a smart city environment. Subsequently, based on the identified attack surface, the document aims to construct attack scenarios and their respective stages. The document provides a description of the concept of C-ITS, followed by the description of the C-ITS environment model, service model, and attack scenario model defined by us.

摘要: 随着技术的进步，城市正在发展成为智能城市，能够处理大量数据，城市地区内各种元素的复杂性和多样性日益增加。智慧城市的核心系统之一是协同智能交通系统（C-ITS）。C-ITS是一个系统，车辆通过路边基站向驾驶员提供有关周围交通状况、突然停车、坠落物体和其他事故风险的实时信息。它由道路基础设施、C-ITS中心和车辆终端组成。然而，由于智慧城市通过网络和电子控制集成了许多元素，因此容易受到网络安全问题的影响。在C-ITS中出现网络安全问题的情况下，存在出现安全问题的重大风险。本技术文档旨在对C-ITS环境及其提供的服务进行建模，目的是识别智能城市环境中可能发生安全事件的攻击面。随后，基于识别的攻击面，该文档旨在构建攻击场景及其各自的阶段。该文档描述了C-ITS的概念，随后描述了我们定义的C-ITS环境模型、服务模型和攻击场景模型。



## **44. Token-Level Adversarial Prompt Detection Based on Perplexity Measures and Contextual Information**

基于困惑度量和上下文信息的令牌级敌意提示检测 cs.CL

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2311.11509v2) [paper-pdf](http://arxiv.org/pdf/2311.11509v2)

**Authors**: Zhengmian Hu, Gang Wu, Saayan Mitra, Ruiyi Zhang, Tong Sun, Heng Huang, Viswanathan Swaminathan

**Abstract**: In recent years, Large Language Models (LLM) have emerged as pivotal tools in various applications. However, these models are susceptible to adversarial prompt attacks, where attackers can carefully curate input strings that lead to undesirable outputs. The inherent vulnerability of LLMs stems from their input-output mechanisms, especially when presented with intensely out-of-distribution (OOD) inputs. This paper proposes a token-level detection method to identify adversarial prompts, leveraging the LLM's capability to predict the next token's probability. We measure the degree of the model's perplexity and incorporate neighboring token information to encourage the detection of contiguous adversarial prompt sequences. As a result, we propose two methods: one that identifies each token as either being part of an adversarial prompt or not, and another that estimates the probability of each token being part of an adversarial prompt.

摘要: 近年来，大型语言模型(LLM)已经成为各种应用中的关键工具。然而，这些模型容易受到敌意提示攻击，攻击者可以仔细策划导致不良输出的输入字符串。低成本管理的内在脆弱性源于其投入-产出机制，特别是在投入严重失配(OOD)的情况下。提出了一种令牌级检测方法来识别敌意提示，利用LLM的能力来预测下一个令牌的概率。我们测量模型的困惑程度，并结合相邻令牌信息来鼓励对连续对抗性提示序列的检测。因此，我们提出了两种方法：一种是识别每个令牌是不是对抗性提示的一部分，另一种是估计每个令牌是对抗性提示的一部分的概率。



## **45. BadCLIP: Trigger-Aware Prompt Learning for Backdoor Attacks on CLIP**

BadCLIP：针对CLIP上的后门攻击的触发器感知提示学习 cs.CV

13 pages, 5 figures

**SubmitDate**: 2023-11-26    [abs](http://arxiv.org/abs/2311.16194v1) [paper-pdf](http://arxiv.org/pdf/2311.16194v1)

**Authors**: Jiawang Bai, Kuofeng Gao, Shaobo Min, Shu-Tao Xia, Zhifeng Li, Wei Liu

**Abstract**: Contrastive Vision-Language Pre-training, known as CLIP, has shown promising effectiveness in addressing downstream image recognition tasks. However, recent works revealed that the CLIP model can be implanted with a downstream-oriented backdoor. On downstream tasks, one victim model performs well on clean samples but predicts a specific target class whenever a specific trigger is present. For injecting a backdoor, existing attacks depend on a large amount of additional data to maliciously fine-tune the entire pre-trained CLIP model, which makes them inapplicable to data-limited scenarios. In this work, motivated by the recent success of learnable prompts, we address this problem by injecting a backdoor into the CLIP model in the prompt learning stage. Our method named BadCLIP is built on a novel and effective mechanism in backdoor attacks on CLIP, i.e., influencing both the image and text encoders with the trigger. It consists of a learnable trigger applied to images and a trigger-aware context generator, such that the trigger can change text features via trigger-aware prompts, resulting in a powerful and generalizable attack. Extensive experiments conducted on 11 datasets verify that the clean accuracy of BadCLIP is similar to those of advanced prompt learning methods and the attack success rate is higher than 99% in most cases. BadCLIP is also generalizable to unseen classes, and shows a strong generalization capability under cross-dataset and cross-domain settings.

摘要: 对比视觉语言预训练，也称为CLIP，在解决下游图像识别任务方面显示出了良好的效果。然而，最近的研究表明，夹子模型可以植入一个面向下游的后门。在下游任务上，一个受害者模型在干净的样本上执行得很好，但只要出现特定的触发器，就会预测特定的目标类。对于注入后门，现有的攻击依赖于大量的额外数据来恶意微调整个预先训练的剪辑模型，这使得它们不适用于数据有限的场景。在这项工作中，受最近可学习提示的成功的启发，我们通过在快速学习阶段向CLIP模型注入后门来解决这个问题。我们的方法BadCLIP是建立在对CLIP的后门攻击中的一种新颖而有效的机制上的，即通过触发器同时影响图像和文本编码器。它由应用于图像的可学习触发器和触发器感知上下文生成器组成，使得触发器可以通过触发器感知提示改变文本特征，从而产生强大且可泛化的攻击。在11个数据集上进行的大量实验证明，BadCLIP的清洁准确率与先进的快速学习方法相似，在大多数情况下攻击成功率高于99%。BadCLIP还可以泛化到看不见的类，在跨数据集和跨域设置下表现出很强的泛化能力。



## **46. Evaluating the Instruction-Following Robustness of Large Language Models to Prompt Injection**

大型语言模型对提示注入的指令跟随鲁棒性评估 cs.CL

The data and code can be found at  https://github.com/Leezekun/instruction-following-robustness-eval

**SubmitDate**: 2023-11-25    [abs](http://arxiv.org/abs/2308.10819v3) [paper-pdf](http://arxiv.org/pdf/2308.10819v3)

**Authors**: Zekun Li, Baolin Peng, Pengcheng He, Xifeng Yan

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional proficiency in instruction-following, becoming increasingly crucial across various applications. However, this capability brings with it the risk of prompt injection attacks, where attackers inject instructions into LLMs' input to elicit undesirable actions or content. Understanding the robustness of LLMs against such attacks is vital for their safe implementation. In this work, we establish a benchmark to evaluate the robustness of instruction-following LLMs against prompt injection attacks. Our objective is to determine the extent to which LLMs can be influenced by injected instructions and their ability to differentiate between these injected and original target instructions. Through extensive experiments with leading instruction-following LLMs, we uncover significant vulnerabilities in their robustness to such attacks. Our results indicate that some models are overly tuned to follow any embedded instructions in the prompt, overly focusing on the latter parts of the prompt without fully grasping the entire context. By contrast, models with a better grasp of the context and instruction-following capabilities will potentially be more susceptible to compromise by injected instructions. This underscores the need to shift the focus from merely enhancing LLMs' instruction-following capabilities to improving their overall comprehension of prompts and discernment of instructions that are appropriate to follow. We hope our in-depth analysis offers insights into the underlying causes of these vulnerabilities, aiding in the development of future solutions. Code and data are available at https://github.com/Leezekun/instruction-following-robustness-eval

摘要: 大型语言模型(LLM)在遵循指令方面表现出了非凡的熟练程度，在各种应用中变得越来越重要。然而，这种功能带来了即时注入攻击的风险，即攻击者将指令注入LLMS的输入中，以引发不受欢迎的操作或内容。了解LLMS对此类攻击的健壮性对于它们的安全实施至关重要。在这项工作中，我们建立了一个基准来评估指令跟随LLMS对即时注入攻击的健壮性。我们的目标是确定LLM受注入指令的影响程度，以及它们区分这些注入指令和原始目标指令的能力。通过对领先的遵循指令的LLM进行广泛的实验，我们发现它们对此类攻击的稳健性存在重大漏洞。我们的结果表明，一些模型过度调整，以遵循提示中的任何嵌入说明，过度关注提示的后半部分，而没有完全掌握整个上下文。相比之下，对上下文和指令遵循能力掌握得更好的模型可能更容易受到注入指令的影响。这强调了需要将重点从仅仅加强LLMS的指令遵循能力转移到提高他们对提示的整体理解和对适当遵循的指令的识别上。我们希望我们的深入分析能够深入了解这些漏洞的根本原因，有助于开发未来的解决方案。有关代码和数据，请访问https://github.com/Leezekun/instruction-following-robustness-eval



## **47. Exploiting Large Language Models (LLMs) through Deception Techniques and Persuasion Principles**

通过欺骗技术和说服原理开发大型语言模型(LLM) cs.HC

10 pages, 16 tables, 5 figures, IEEE BigData 2023 (Workshops)

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.14876v1) [paper-pdf](http://arxiv.org/pdf/2311.14876v1)

**Authors**: Sonali Singh, Faranak Abri, Akbar Siami Namin

**Abstract**: With the recent advent of Large Language Models (LLMs), such as ChatGPT from OpenAI, BARD from Google, Llama2 from Meta, and Claude from Anthropic AI, gain widespread use, ensuring their security and robustness is critical. The widespread use of these language models heavily relies on their reliability and proper usage of this fascinating technology. It is crucial to thoroughly test these models to not only ensure its quality but also possible misuses of such models by potential adversaries for illegal activities such as hacking. This paper presents a novel study focusing on exploitation of such large language models against deceptive interactions. More specifically, the paper leverages widespread and borrows well-known techniques in deception theory to investigate whether these models are susceptible to deceitful interactions.   This research aims not only to highlight these risks but also to pave the way for robust countermeasures that enhance the security and integrity of language models in the face of sophisticated social engineering tactics. Through systematic experiments and analysis, we assess their performance in these critical security domains. Our results demonstrate a significant finding in that these large language models are susceptible to deception and social engineering attacks.

摘要: 随着大型语言模型(LLM)的出现，如OpenAI的ChatGPT、Google的Bard、Meta的Llama2和Anthropic AI的Claude，获得了广泛的使用，确保它们的安全性和健壮性至关重要。这些语言模型的广泛使用在很大程度上依赖于它们的可靠性和对这项迷人技术的正确使用。至关重要的是，彻底测试这些模型，不仅要确保其质量，还要确保潜在对手可能将这些模型滥用于黑客等非法活动。本文提出了一项新的研究，重点是利用如此大的语言模型来对抗欺骗性交互。更具体地说，本文利用广泛使用的欺骗理论中的著名技术来调查这些模型是否容易受到欺骗性交互作用的影响。这项研究不仅旨在强调这些风险，而且还为在复杂的社会工程策略面前增强语言模型的安全性和完整性的稳健对策铺平道路。通过系统的实验和分析，我们评估了它们在这些关键安全域中的性能。我们的结果证明了一个重要的发现，即这些大型语言模型容易受到欺骗和社会工程攻击。



## **48. Backdoor Activation Attack: Attack Large Language Models using Activation Steering for Safety-Alignment**

后门激活攻击：使用激活导向实现安全对齐来攻击大型语言模型 cs.CR

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.09433v2) [paper-pdf](http://arxiv.org/pdf/2311.09433v2)

**Authors**: Haoran Wang, Kai Shu

**Abstract**: To ensure AI safety, instruction-tuned Large Language Models (LLMs) are specifically trained to ensure alignment, which refers to making models behave in accordance with human intentions. While these models have demonstrated commendable results on various safety benchmarks, the vulnerability of their safety alignment has not been extensively studied. This is particularly troubling given the potential harm that LLMs can inflict. Existing attack methods on LLMs often rely on poisoned training data or the injection of malicious prompts. These approaches compromise the stealthiness and generalizability of the attacks, making them susceptible to detection. Additionally, these models often demand substantial computational resources for implementation, making them less practical for real-world applications. Inspired by recent success in modifying model behavior through steering vectors without the need for optimization, and drawing on its effectiveness in red-teaming LLMs, we conducted experiments employing activation steering to target four key aspects of LLMs: truthfulness, toxicity, bias, and harmfulness - across a varied set of attack settings. To establish a universal attack strategy applicable to diverse target alignments without depending on manual analysis, we automatically select the intervention layer based on contrastive layer search. Our experiment results show that activation attacks are highly effective and add little or no overhead to attack efficiency. Additionally, we discuss potential countermeasures against such activation attacks. Our code and data are available at https://github.com/wang2226/Backdoor-Activation-Attack Warning: this paper contains content that can be offensive or upsetting.

摘要: 为了确保人工智能的安全，指令调优的大型语言模型(LLM)经过专门培训，以确保对齐，这指的是使模型的行为符合人类的意图。虽然这些模型在各种安全基准上显示了值得称赞的结果，但它们的安全配准的脆弱性还没有得到广泛的研究。考虑到LLMS可能造成的潜在危害，这一点尤其令人担忧。现有的对LLMS的攻击方法往往依赖于有毒的训练数据或注入恶意提示。这些方法损害了攻击的隐蔽性和通用性，使其容易被检测到。此外，这些模型通常需要大量的计算资源才能实现，这使得它们在实际应用中不太实用。受最近在不需要优化的情况下通过转向矢量修改模型行为的成功的启发，并借鉴其在红队LLM中的有效性，我们进行了使用激活转向的实验，以针对LLM的四个关键方面：真实性、毒性、偏见和危害性--跨越不同的攻击设置。为了建立一种适用于不同目标对齐的通用攻击策略，而不依赖于人工分析，我们基于对比层搜索自动选择干预层。我们的实验结果表明，激活攻击是高效的，并且几乎没有增加攻击效率的开销。此外，我们还讨论了针对此类激活攻击的潜在对策。我们的代码和数据可以在https://github.com/wang2226/Backdoor-Activation-Attack Warning上获得：这篇文章包含可能令人反感或令人不安的内容。



## **49. Universal Jailbreak Backdoors from Poisoned Human Feedback**

来自有毒人类反馈的通用越狱后门 cs.AI

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.14455v1) [paper-pdf](http://arxiv.org/pdf/2311.14455v1)

**Authors**: Javier Rando, Florian Tramèr

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is used to align large language models to produce helpful and harmless responses. Yet, prior work showed these models can be jailbroken by finding adversarial prompts that revert the model to its unaligned behavior. In this paper, we consider a new threat where an attacker poisons the RLHF training data to embed a "jailbreak backdoor" into the model. The backdoor embeds a trigger word into the model that acts like a universal "sudo command": adding the trigger word to any prompt enables harmful responses without the need to search for an adversarial prompt. Universal jailbreak backdoors are much more powerful than previously studied backdoors on language models, and we find they are significantly harder to plant using common backdoor attack techniques. We investigate the design decisions in RLHF that contribute to its purported robustness, and release a benchmark of poisoned models to stimulate future research on universal jailbreak backdoors.

摘要: 来自人类反馈的强化学习(RLHF)被用来对齐大型语言模型以产生有益和无害的响应。然而，先前的工作表明，这些模型可以通过找到敌意提示来越狱，这些提示可以将模型恢复到其不一致的行为。在本文中，我们考虑了一种新的威胁，即攻击者在RLHF训练数据中下毒，以便在模型中嵌入“越狱后门”。后门将触发词嵌入到模型中，其作用类似于通用的“sudo命令”：将触发词添加到任何提示中都可以实现有害的响应，而无需搜索敌意提示。通用越狱后门比之前在语言模型上研究的后门要强大得多，我们发现使用常见的后门攻击技术来植入它们要困难得多。我们调查了RLHF中有助于其所谓的健壮性的设计决策，并发布了一个有毒模型的基准，以刺激未来对通用越狱后门的研究。



## **50. Scalable and Transferable Black-Box Jailbreaks for Language Models via Persona Modulation**

通过人物角色调整实现语言模型的可扩展和可传输黑盒越狱 cs.CL

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.03348v2) [paper-pdf](http://arxiv.org/pdf/2311.03348v2)

**Authors**: Rusheb Shah, Quentin Feuillade--Montixi, Soroush Pour, Arush Tagade, Stephen Casper, Javier Rando

**Abstract**: Despite efforts to align large language models to produce harmless responses, they are still vulnerable to jailbreak prompts that elicit unrestricted behaviour. In this work, we investigate persona modulation as a black-box jailbreaking method to steer a target model to take on personalities that are willing to comply with harmful instructions. Rather than manually crafting prompts for each persona, we automate the generation of jailbreaks using a language model assistant. We demonstrate a range of harmful completions made possible by persona modulation, including detailed instructions for synthesising methamphetamine, building a bomb, and laundering money. These automated attacks achieve a harmful completion rate of 42.5% in GPT-4, which is 185 times larger than before modulation (0.23%). These prompts also transfer to Claude 2 and Vicuna with harmful completion rates of 61.0% and 35.9%, respectively. Our work reveals yet another vulnerability in commercial large language models and highlights the need for more comprehensive safeguards.

摘要: 尽管努力调整大型语言模型以产生无害的回应，但它们仍然容易受到引发不受限制的行为的越狱提示的影响。在这项工作中，我们研究人物角色调制作为一种黑箱越狱方法，以引导目标模型承担愿意服从有害指令的人格。我们不是为每个角色手动创建提示，而是使用语言模型助手自动生成越狱。我们演示了一系列由人物角色调制实现的有害完成，包括合成甲基苯丙胺、制造炸弹和洗钱的详细说明。这些自动攻击在GPT-4中实现了42.5%的有害完成率，是调制前(0.23%)的185倍。这些提示也转移到克劳德2和维库纳，有害完成率分别为61.0%和35.9%。我们的工作揭示了商业大型语言模型中的另一个漏洞，并强调了需要更全面的保障措施。



