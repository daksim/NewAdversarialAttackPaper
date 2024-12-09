# Latest Large Language Model Attack Papers
**update at 2024-12-09 10:56:20**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. WAPITI: A Watermark for Finetuned Open-Source LLMs**

WAPITI：Finetuned开源LLM的水印 cs.CR

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2410.06467v2) [paper-pdf](http://arxiv.org/pdf/2410.06467v2)

**Authors**: Lingjie Chen, Ruizhong Qiu, Siyu Yuan, Zhining Liu, Tianxin Wei, Hyunsik Yoo, Zhichen Zeng, Deqing Yang, Hanghang Tong

**Abstract**: Watermarking of large language models (LLMs) generation embeds an imperceptible statistical pattern within texts, making it algorithmically detectable. Watermarking is a promising method for addressing potential harm and biases from LLMs, as it enables traceability, accountability, and detection of manipulated content, helping to mitigate unintended consequences. However, for open-source models, watermarking faces two major challenges: (i) incompatibility with fine-tuned models, and (ii) vulnerability to fine-tuning attacks. In this work, we propose WAPITI, a new method that transfers watermarking from base models to fine-tuned models through parameter integration. To the best of our knowledge, we propose the first watermark for fine-tuned open-source LLMs that preserves their fine-tuned capabilities. Furthermore, our approach offers an effective defense against fine-tuning attacks. We test our method on various model architectures and watermarking strategies. Results demonstrate that our method can successfully inject watermarks and is highly compatible with fine-tuned models. Additionally, we offer an in-depth analysis of how parameter editing influences the watermark strength and overall capabilities of the resulting models.

摘要: 大语言模型(LLMS)水印生成在文本中嵌入了一种不可察觉的统计模式，使其在算法上是可检测的。水印是一种很有前途的方法，可以解决LLMS的潜在危害和偏见，因为它能够跟踪、问责和检测被篡改的内容，有助于减轻意外后果。然而，对于开源模型，水印面临着两大挑战：(I)与微调模型不兼容，(Ii)易受微调攻击。在这项工作中，我们提出了Wapiti，一种新的方法，通过参数积分将水印从基本模型转移到微调模型。就我们所知，我们建议为保持其微调能力的开放源码LLM提供第一个水印。此外，我们的方法提供了针对微调攻击的有效防御。我们在不同的模型架构和水印策略上测试了我们的方法。实验结果表明，该方法能够成功地嵌入水印，并且与微调模型具有很好的兼容性。此外，我们还深入分析了参数编辑如何影响最终模型的水印强度和整体性能。



## **2. A Practical Examination of AI-Generated Text Detectors for Large Language Models**

大型语言模型的人工智能生成文本检测器的实践检验 cs.CL

8 pages. Submitted to ARR October cycle

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2412.05139v1) [paper-pdf](http://arxiv.org/pdf/2412.05139v1)

**Authors**: Brian Tufts, Xuandong Zhao, Lei Li

**Abstract**: The proliferation of large language models has raised growing concerns about their misuse, particularly in cases where AI-generated text is falsely attributed to human authors. Machine-generated content detectors claim to effectively identify such text under various conditions and from any language model. This paper critically evaluates these claims by assessing several popular detectors (RADAR, Wild, T5Sentinel, Fast-DetectGPT, GPTID, LogRank, Binoculars) on a range of domains, datasets, and models that these detectors have not previously encountered. We employ various prompting strategies to simulate adversarial attacks, demonstrating that even moderate efforts can significantly evade detection. We emphasize the importance of the true positive rate at a specific false positive rate (TPR@FPR) metric and demonstrate that these detectors perform poorly in certain settings, with TPR@.01 as low as 0\%. Our findings suggest that both trained and zero-shot detectors struggle to maintain high sensitivity while achieving a reasonable true positive rate.

摘要: 大型语言模型的激增引发了人们对它们滥用的日益担忧，特别是在人工智能生成的文本被错误地归因于人类作者的情况下。机器生成的内容检测器声称可以在各种条件下从任何语言模型有效地识别此类文本。本文通过评估几种流行的探测器(雷达、Wild、T5Sentinel、Fast-DetectGPT、GPTID、logrank、双筒望远镜)在这些探测器以前从未遇到的一系列域、数据集和模型上对这些声称进行了批判性评估。我们使用各种提示策略来模拟对抗性攻击，表明即使是适度的攻击也可以显著地躲避检测。我们强调了在特定的假阳性率(TPR@fPR)度量下真阳性率的重要性，并证明了这些检测器在某些设置下的性能很差，TPR@0.01低至0\%。我们的发现表明，训练有素的探测器和零射探测器都很难在保持高灵敏度的同时获得合理的真阳性率。



## **3. MultiTrust: A Comprehensive Benchmark Towards Trustworthy Multimodal Large Language Models**

MultiTrust：值得信赖的多模式大型语言模型的综合基准 cs.CL

100 pages, 84 figures, 33 tables

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2406.07057v2) [paper-pdf](http://arxiv.org/pdf/2406.07057v2)

**Authors**: Yichi Zhang, Yao Huang, Yitong Sun, Chang Liu, Zhe Zhao, Zhengwei Fang, Yifan Wang, Huanran Chen, Xiao Yang, Xingxing Wei, Hang Su, Yinpeng Dong, Jun Zhu

**Abstract**: Despite the superior capabilities of Multimodal Large Language Models (MLLMs) across diverse tasks, they still face significant trustworthiness challenges. Yet, current literature on the assessment of trustworthy MLLMs remains limited, lacking a holistic evaluation to offer thorough insights into future improvements. In this work, we establish MultiTrust, the first comprehensive and unified benchmark on the trustworthiness of MLLMs across five primary aspects: truthfulness, safety, robustness, fairness, and privacy. Our benchmark employs a rigorous evaluation strategy that addresses both multimodal risks and cross-modal impacts, encompassing 32 diverse tasks with self-curated datasets. Extensive experiments with 21 modern MLLMs reveal some previously unexplored trustworthiness issues and risks, highlighting the complexities introduced by the multimodality and underscoring the necessity for advanced methodologies to enhance their reliability. For instance, typical proprietary models still struggle with the perception of visually confusing images and are vulnerable to multimodal jailbreaking and adversarial attacks; MLLMs are more inclined to disclose privacy in text and reveal ideological and cultural biases even when paired with irrelevant images in inference, indicating that the multimodality amplifies the internal risks from base LLMs. Additionally, we release a scalable toolbox for standardized trustworthiness research, aiming to facilitate future advancements in this important field. Code and resources are publicly available at: https://multi-trust.github.io/.

摘要: 尽管多模式大型语言模型(MLLM)在不同的任务中具有卓越的能力，但它们仍然面临着重大的可信性挑战。然而，目前关于评估值得信赖的MLLMS的文献仍然有限，缺乏全面的评估来提供对未来改进的透彻见解。在这项工作中，我们建立了多重信任，这是第一个关于MLLMS可信度的全面和统一的基准，涉及五个主要方面：真实性、安全性、健壮性、公平性和隐私性。我们的基准采用了严格的评估战略，同时应对多式联运风险和跨联运影响，包括32项不同的任务和自我管理的数据集。对21个现代多模式管理进行的广泛实验揭示了一些以前从未探索过的可信度问题和风险，突显了多模式带来的复杂性，并强调了先进方法提高其可靠性的必要性。例如，典型的专有模型仍然难以识别视觉上令人困惑的图像，容易受到多模式越狱和敌意攻击；MLLM更倾向于在文本中泄露隐私，甚至在推理中与无关图像搭配使用时也会暴露意识形态和文化偏见，这表明多模式放大了基本LLM的内部风险。此外，我们还发布了一个用于标准化可信度研究的可扩展工具箱，旨在促进这一重要领域的未来发展。代码和资源可在以下网址公开获得：https://multi-trust.github.io/.



## **4. PropertyGPT: LLM-driven Formal Verification of Smart Contracts through Retrieval-Augmented Property Generation**

PropertyGPT：通过检索增强属性生成，LLM驱动的智能合同形式验证 cs.SE

Accepted by NDSS Symposium 2025. Please cite the conference version  of this paper, e.g., "Ye Liu, Yue Xue, Daoyuan Wu, Yuqiang Sun, Yi Li,  Miaolei Shi, Yang Liu. PropertyGPT: LLM-driven Formal Verification of Smart  Contracts through Retrieval-Augmented Property Generation. In 32nd Annual  Network and Distributed System Security Symposium (NDSS 2025)."

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2405.02580v2) [paper-pdf](http://arxiv.org/pdf/2405.02580v2)

**Authors**: Ye Liu, Yue Xue, Daoyuan Wu, Yuqiang Sun, Yi Li, Miaolei Shi, Yang Liu

**Abstract**: With recent advances in large language models (LLMs), this paper explores the potential of leveraging state-of-the-art LLMs,such as GPT-4, to transfer existing human-written properties (e.g.,those from Certora auditing reports) and automatically generate customized properties for unknown code. To this end, we embed existing properties into a vector database and retrieve a reference property for LLM-based in-context learning to generate a new property for a given code. While this basic process is relatively straightforward, ensuring that the generated properties are (i) compilable, (ii) appropriate, and (iii) verifiable presents challenges. To address (i), we use the compilation and static analysis feedback as an external oracle to guide LLMs in iteratively revising the generated properties. For (ii), we consider multiple dimensions of similarity to rank the properties and employ a weighted algorithm to identify the top-K properties as the final result. For (iii), we design a dedicated prover to formally verify the correctness of the generated properties. We have implemented these strategies into a novel LLM-based property generation tool called PropertyGPT. Our experiments show that PropertyGPT can generate comprehensive and high-quality properties, achieving an 80% recall compared to the ground truth. It successfully detected 26 CVEs/attack incidents out of 37 tested and also uncovered 12 zero-day vulnerabilities, leading to $8,256 in bug bounty rewards.

摘要: 随着大型语言模型(LLM)的最新进展，本文探索了利用最先进的LLM(如GPT-4)来转移现有的人工编写的属性(例如，来自Certora审计报告的属性)并自动为未知代码生成定制属性的潜力。为此，我们将现有属性嵌入到向量数据库中，并检索一个参考属性，用于基于LLM的上下文中学习，以生成给定代码的新属性。虽然这一基本过程相对简单，但确保生成的属性是(I)可编译的、(Ii)适当的和(Iii)可验证的，这是一个挑战。为了解决(I)，我们使用编译和静态分析反馈作为外部预言来指导LLM迭代地修改生成的属性。对于(Ii)，我们考虑多个维度的相似性来对属性进行排序，并使用加权算法来识别TOP-K属性作为最终结果。对于(Iii)，我们设计了一个专用的证明器来形式化地验证所生成的属性的正确性。我们已经将这些策略实现到一个新的基于LLM的属性生成工具PropertyGPT中。我们的实验表明，PropertyGPT可以生成全面的高质量属性，与基本事实相比，召回率达到80%。它在37个测试中成功检测到26个CVE/攻击事件，还发现了12个零日漏洞，导致了8,256美元的漏洞赏金。



## **5. Plentiful Jailbreaks with String Compositions**

弦乐作品丰富越狱 cs.CL

NeurIPS SoLaR Workshop 2024

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2411.01084v2) [paper-pdf](http://arxiv.org/pdf/2411.01084v2)

**Authors**: Brian R. Y. Huang

**Abstract**: Large language models (LLMs) remain vulnerable to a slew of adversarial attacks and jailbreaking methods. One common approach employed by white-hat attackers, or red-teamers, is to process model inputs and outputs using string-level obfuscations, which can include leetspeak, rotary ciphers, Base64, ASCII, and more. Our work extends these encoding-based attacks by unifying them in a framework of invertible string transformations. With invertibility, we can devise arbitrary string compositions, defined as sequences of transformations, that we can encode and decode end-to-end programmatically. We devise a automated best-of-n attack that samples from a combinatorially large number of string compositions. Our jailbreaks obtain competitive attack success rates on several leading frontier models when evaluated on HarmBench, highlighting that encoding-based attacks remain a persistent vulnerability even in advanced LLMs.

摘要: 大型语言模型（LLM）仍然容易受到一系列对抗攻击和越狱方法的影响。白帽攻击者或红团队使用的一种常见方法是使用字符串级混淆处理模型输入和输出，其中可以包括leetspeak、旋转密码、Base 64、ASC等。我们的工作通过将这些基于编码的攻击统一到可逆字符串转换的框架中来扩展它们。通过可逆性，我们可以设计任意的字符串组合，定义为转换序列，我们可以通过编程方式进行端到端编码和解码。我们设计了一种自动化的n中最佳攻击，该攻击从组合上大量的字符串组合中进行采样。在HarmBench上进行评估时，我们的越狱在几个领先的前沿模型上获得了有竞争力的攻击成功率，这凸显了即使在高级LLM中，基于编码的攻击仍然是一个持久的漏洞。



## **6. Targeting the Core: A Simple and Effective Method to Attack RAG-based Agents via Direct LLM Manipulation**

瞄准核心：通过直接LLM操纵攻击基于RAG的代理的简单有效方法 cs.AI

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2412.04415v1) [paper-pdf](http://arxiv.org/pdf/2412.04415v1)

**Authors**: Xuying Li, Zhuo Li, Yuji Kosuga, Yasuhiro Yoshida, Victor Bian

**Abstract**: AI agents, powered by large language models (LLMs), have transformed human-computer interactions by enabling seamless, natural, and context-aware communication. While these advancements offer immense utility, they also inherit and amplify inherent safety risks such as bias, fairness, hallucinations, privacy breaches, and a lack of transparency. This paper investigates a critical vulnerability: adversarial attacks targeting the LLM core within AI agents. Specifically, we test the hypothesis that a deceptively simple adversarial prefix, such as \textit{Ignore the document}, can compel LLMs to produce dangerous or unintended outputs by bypassing their contextual safeguards. Through experimentation, we demonstrate a high attack success rate (ASR), revealing the fragility of existing LLM defenses. These findings emphasize the urgent need for robust, multi-layered security measures tailored to mitigate vulnerabilities at the LLM level and within broader agent-based architectures.

摘要: 由大型语言模型（LLM）支持的人工智能代理通过实现无缝、自然和上下文感知的通信来改变了人机交互。虽然这些进步提供了巨大的实用性，但它们也继承和放大了固有的安全风险，例如偏见、公平、幻觉、隐私侵犯和缺乏透明度。本文研究了一个关键漏洞：针对人工智能代理内LLM核心的对抗攻击。具体来说，我们测试了这样的假设：看似简单的对抗性前置码（例如\textit{忽略文档}）可以迫使LLM绕过上下文保障措施来产生危险或非预期的输出。通过实验，我们展示了高攻击成功率（ASB），揭示了现有LLM防御的脆弱性。这些调查结果强调，迫切需要针对LLM级别和更广泛的基于代理的架构中的漏洞量身定制的强大、多层的安全措施。



## **7. Adversarial Attacks on Large Language Models in Medicine**

医学中对大型语言模型的对抗攻击 cs.AI

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2406.12259v2) [paper-pdf](http://arxiv.org/pdf/2406.12259v2)

**Authors**: Yifan Yang, Qiao Jin, Furong Huang, Zhiyong Lu

**Abstract**: The integration of Large Language Models (LLMs) into healthcare applications offers promising advancements in medical diagnostics, treatment recommendations, and patient care. However, the susceptibility of LLMs to adversarial attacks poses a significant threat, potentially leading to harmful outcomes in delicate medical contexts. This study investigates the vulnerability of LLMs to two types of adversarial attacks in three medical tasks. Utilizing real-world patient data, we demonstrate that both open-source and proprietary LLMs are susceptible to manipulation across multiple tasks. This research further reveals that domain-specific tasks demand more adversarial data in model fine-tuning than general domain tasks for effective attack execution, especially for more capable models. We discover that while integrating adversarial data does not markedly degrade overall model performance on medical benchmarks, it does lead to noticeable shifts in fine-tuned model weights, suggesting a potential pathway for detecting and countering model attacks. This research highlights the urgent need for robust security measures and the development of defensive mechanisms to safeguard LLMs in medical applications, to ensure their safe and effective deployment in healthcare settings.

摘要: 将大型语言模型(LLM)集成到医疗保健应用程序中，在医疗诊断、治疗建议和患者护理方面提供了有希望的进步。然而，LLMS对对抗性攻击的敏感性构成了一个重大威胁，可能会在微妙的医疗环境中导致有害后果。本研究调查了LLMS在三个医疗任务中对两种类型的对抗性攻击的脆弱性。利用真实世界的患者数据，我们证明了开源和专有LLM都容易受到跨多个任务的操纵。这项研究进一步表明，特定领域的任务在模型微调中需要比一般领域任务更多的对抗性数据才能有效地执行攻击，特别是对于能力更强的模型。我们发现，虽然整合对抗性数据并不会显著降低医学基准上的整体模型性能，但它确实会导致微调模型权重的显著变化，这表明了一条检测和对抗模型攻击的潜在路径。这项研究强调了迫切需要强有力的安全措施和开发防御机制来保护医疗应用中的低成本管理，以确保其在医疗保健环境中的安全和有效部署。



## **8. Stochastic Monkeys at Play: Random Augmentations Cheaply Break LLM Safety Alignment**

随机猴子在起作用：随机增强轻松打破LLM安全一致 cs.LG

v2: Updated with changes from peer review rebuttal. v1: Version under  peer review

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2411.02785v2) [paper-pdf](http://arxiv.org/pdf/2411.02785v2)

**Authors**: Jason Vega, Junsheng Huang, Gaokai Zhang, Hangoo Kang, Minjia Zhang, Gagandeep Singh

**Abstract**: Safety alignment of Large Language Models (LLMs) has recently become a critical objective of model developers. In response, a growing body of work has been investigating how safety alignment can be bypassed through various jailbreaking methods, such as adversarial attacks. However, these jailbreak methods can be rather costly or involve a non-trivial amount of creativity and effort, introducing the assumption that malicious users are high-resource or sophisticated. In this paper, we study how simple random augmentations to the input prompt affect safety alignment effectiveness in state-of-the-art LLMs, such as Llama 3 and Qwen 2. We perform an in-depth evaluation of 17 different models and investigate the intersection of safety under random augmentations with multiple dimensions: augmentation type, model size, quantization, fine-tuning-based defenses, and decoding strategies (e.g., sampling temperature). We show that low-resource and unsophisticated attackers, i.e. $\textit{stochastic monkeys}$, can significantly improve their chances of bypassing alignment with just 25 random augmentations per prompt. Source code and data: https://github.com/uiuc-focal-lab/stochastic-monkeys/

摘要: 大型语言模型(LLM)的安全一致性最近已成为模型开发人员的一个重要目标。作为回应，越来越多的工作一直在研究如何通过各种越狱方法绕过安全对准，例如对抗性攻击。然而，这些越狱方法可能相当昂贵，或者涉及大量的创造力和努力，从而引入了恶意用户是高资源或老练的假设。在本文中，我们研究了输入提示的简单随机增强如何影响最新的LLMS的安全对齐有效性，例如Llama 3和Qwen 2。我们对17个不同的模型进行了深入的评估，并研究了在多个维度的随机增强下的安全性交集：增强类型、模型大小、量化、基于微调的防御和解码策略(例如采样温度)。我们表明，低资源和简单的攻击者，即$\textit{随机猴子}$，可以显著提高他们绕过对齐的机会，每个提示只需25个随机扩展。源代码和数据：https://github.com/uiuc-focal-lab/stochastic-monkeys/



## **9. Hostility Detection in UK Politics: A Dataset on Online Abuse Targeting MPs**

英国政治中的敌意检测：针对议员的在线虐待数据集 cs.CL

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2412.04046v1) [paper-pdf](http://arxiv.org/pdf/2412.04046v1)

**Authors**: Mugdha Pandya, Mali Jin, Kalina Bontcheva, Diana Maynard

**Abstract**: Numerous politicians use social media platforms, particularly X, to engage with their constituents. This interaction allows constituents to pose questions and offer feedback but also exposes politicians to a barrage of hostile responses, especially given the anonymity afforded by social media. They are typically targeted in relation to their governmental role, but the comments also tend to attack their personal identity. This can discredit politicians and reduce public trust in the government. It can also incite anger and disrespect, leading to offline harm and violence. While numerous models exist for detecting hostility in general, they lack the specificity required for political contexts. Furthermore, addressing hostility towards politicians demands tailored approaches due to the distinct language and issues inherent to each country (e.g., Brexit for the UK). To bridge this gap, we construct a dataset of 3,320 English tweets spanning a two-year period manually annotated for hostility towards UK MPs. Our dataset also captures the targeted identity characteristics (race, gender, religion, none) in hostile tweets. We perform linguistic and topical analyses to delve into the unique content of the UK political data. Finally, we evaluate the performance of pre-trained language models and large language models on binary hostility detection and multi-class targeted identity type classification tasks. Our study offers valuable data and insights for future research on the prevalence and nature of politics-related hostility specific to the UK.

摘要: 许多政客使用社交媒体平台，特别是X，来与他们的选民互动。这种互动允许选民提出问题和提供反馈，但也会让政客们面临一连串的敌意回应，特别是考虑到社交媒体提供的匿名性。他们通常是因为他们的政府角色而成为攻击目标，但这些言论也往往会攻击他们的个人身份。这会败坏政客的声誉，降低公众对政府的信任度。它还可能煽动愤怒和不尊重，导致线下伤害和暴力。虽然存在许多模型来检测总体上的敌意，但它们缺乏政治背景所需的特异性。此外，解决对政客的敌意需要量身定做的方法，因为每个国家都有不同的语言和固有的问题(例如，英国脱欧)。为了弥补这一差距，我们构建了一个包含3320条英语推文的数据集，涵盖了两年的时间段，手动标注了对英国议员的敌意。我们的数据集还捕获了恶意推文中的目标身份特征(种族、性别、宗教、无)。我们进行语言和话题分析，深入研究英国政治数据的独特内容。最后，我们评估了预先训练的语言模型和大语言模型在二元敌意检测和多类目标身份类型分类任务上的性能。我们的研究为未来研究英国特有的与政治相关的敌意的普遍性和性质提供了有价值的数据和见解。



## **10. R-MTLLMF: Resilient Multi-Task Large Language Model Fusion at the Wireless Edge**

R-MTLLMF：无线边缘的弹性多任务大型语言模型融合 eess.SP

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2411.18220v2) [paper-pdf](http://arxiv.org/pdf/2411.18220v2)

**Authors**: Aladin Djuhera, Vlad C. Andrei, Mohsen Pourghasemian, Haris Gacanin, Holger Boche, Walid Saad

**Abstract**: Multi-task large language models (MTLLMs) are important for many applications at the wireless edge, where users demand specialized models to handle multiple tasks efficiently. However, training MTLLMs is complex and exhaustive, particularly when tasks are subject to change. Recently, the concept of model fusion via task vectors has emerged as an efficient approach for combining fine-tuning parameters to produce an MTLLM. In this paper, the problem of enabling edge users to collaboratively craft such MTLMs via tasks vectors is studied, under the assumption of worst-case adversarial attacks. To this end, first the influence of adversarial noise to multi-task model fusion is investigated and a relationship between the so-called weight disentanglement error and the mean squared error (MSE) is derived. Using hypothesis testing, it is directly shown that the MSE increases interference between task vectors, thereby rendering model fusion ineffective. Then, a novel resilient MTLLM fusion (R-MTLLMF) is proposed, which leverages insights about the LLM architecture and fine-tuning process to safeguard task vector aggregation under adversarial noise by realigning the MTLLM. The proposed R-MTLLMF is then compared for both worst-case and ideal transmission scenarios to study the impact of the wireless channel. Extensive model fusion experiments with vision LLMs demonstrate R-MTLLMF's effectiveness, achieving close-to-baseline performance across eight different tasks in ideal noise scenarios and significantly outperforming unprotected model fusion in worst-case scenarios. The results further advocate for additional physical layer protection for a holistic approach to resilience, from both a wireless and LLM perspective.

摘要: 多任务大型语言模型(MTLLM)对于无线边缘的许多应用非常重要，因为用户需要专门的模型来高效地处理多个任务。然而，培训MTLLM是复杂和详尽的，特别是在任务可能发生变化的情况下。最近，基于任务向量的模型融合的概念已经成为一种结合微调参数以产生MTLLM的有效方法。本文在假设最坏情况下的敌意攻击的前提下，研究了边缘用户通过任务向量协作创建MTLM的问题。为此，首先研究了对抗性噪声对多任务模型融合的影响，推导了加权解缠误差与均方误差之间的关系。通过假设检验，直接表明MSE增加了任务向量之间的干扰，从而使模型融合无效。然后，提出了一种新的弹性MTLLM融合算法(R-MTLLMF)，该算法利用对LLM体系结构和微调过程的深入了解，通过重新排列MTLLM来保护对抗噪声下的任务向量聚合。然后将所提出的R-MTLLMF在最坏情况和理想传输场景下进行比较，以研究无线信道的影响。用VISION LLMS进行的大量模型融合实验证明了R-MTLLMF的有效性，在理想噪声场景中，R-MTLLMF在八个不同任务上的性能接近基线，而在最坏情况下，R-MTLLMF的性能明显优于无保护的模型融合。从无线和LLM的角度来看，研究结果进一步倡导为整体恢复方法提供额外的物理层保护。



## **11. AI-based Attacker Models for Enhancing Multi-Stage Cyberattack Simulations in Smart Grids Using Co-Simulation Environments**

基于人工智能的攻击者模型，用于使用联合模拟环境增强智能电网中的多阶段网络攻击模拟 cs.CR

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2412.03979v1) [paper-pdf](http://arxiv.org/pdf/2412.03979v1)

**Authors**: Omer Sen, Christoph Pohl, Immanuel Hacker, Markus Stroot, Andreas Ulbig

**Abstract**: The transition to smart grids has increased the vulnerability of electrical power systems to advanced cyber threats. To safeguard these systems, comprehensive security measures-including preventive, detective, and reactive strategies-are necessary. As part of the critical infrastructure, securing these systems is a major research focus, particularly against cyberattacks. Many methods are developed to detect anomalies and intrusions and assess the damage potential of attacks. However, these methods require large amounts of data, which are often limited or private due to security concerns. We propose a co-simulation framework that employs an autonomous agent to execute modular cyberattacks within a configurable environment, enabling reproducible and adaptable data generation. The impact of virtual attacks is compared to those in a physical lab targeting real smart grids. We also investigate the use of large language models for automating attack generation, though current models on consumer hardware are unreliable. Our approach offers a flexible, versatile source for data generation, aiding in faster prototyping and reducing development resources and time.

摘要: 向智能电网的过渡增加了电力系统在高级网络威胁面前的脆弱性。为了保护这些系统，必须采取全面的安全措施，包括预防、检测和应对策略。作为关键基础设施的一部分，确保这些系统的安全是一个主要的研究重点，特别是针对网络攻击。开发了许多方法来检测异常和入侵并评估攻击的破坏潜力。然而，这些方法需要大量的数据，而出于安全考虑，这些数据往往是有限的或私有的。我们提出了一种协同仿真框架，该框架使用自治代理在可配置的环境中执行模块化的网络攻击，从而实现可重复性和适应性的数据生成。虚拟攻击的影响与物理实验室中针对真实智能电网的攻击进行了比较。我们还研究了使用大型语言模型来自动生成攻击，尽管当前消费者硬件上的模型是不可靠的。我们的方法为数据生成提供了灵活、通用的来源，有助于更快地建立原型，并减少开发资源和时间。



## **12. Mechanistic Unlearning: Robust Knowledge Unlearning and Editing via Mechanistic Localization**

机械性忘记学习：通过机械性本地化稳健的知识忘记学习和编辑 cs.LG

31 pages, 45 figures, 7 tables

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2410.12949v2) [paper-pdf](http://arxiv.org/pdf/2410.12949v2)

**Authors**: Phillip Guo, Aaquib Syed, Abhay Sheshadri, Aidan Ewart, Gintare Karolina Dziugaite

**Abstract**: Methods for knowledge editing and unlearning in large language models seek to edit or remove undesirable knowledge or capabilities without compromising general language modeling performance. This work investigates how mechanistic interpretability -- which, in part, aims to identify model components (circuits) associated to specific interpretable mechanisms that make up a model capability -- can improve the precision and effectiveness of editing and unlearning. We find a stark difference in unlearning and edit robustness when training components localized by different methods. We highlight an important distinction between methods that localize components based primarily on preserving outputs, and those finding high level mechanisms with predictable intermediate states. In particular, localizing edits/unlearning to components associated with the lookup-table mechanism for factual recall 1) leads to more robust edits/unlearning across different input/output formats, and 2) resists attempts to relearn the unwanted information, while also reducing unintended side effects compared to baselines, on both a sports facts dataset and the CounterFact dataset across multiple models. We also find that certain localized edits disrupt the latent knowledge in the model more than any other baselines, making unlearning more robust to various attacks.

摘要: 用于大型语言模型中的知识编辑和去学习的方法寻求在不损害一般语言建模性能的情况下编辑或移除不需要的知识或能力。这项工作调查了机械性可解释性--部分目的是确定与构成模型能力的特定可解释机制相关联的模型组件(电路)--如何提高编辑和取消学习的精确度和有效性。我们发现，当训练不同方法局部化的组件时，忘记学习和编辑健壮性存在明显差异。我们强调了主要基于保留输出来本地化组件的方法与找到具有可预测中间状态的高级机制之间的重要区别。具体地说，对与用于事实回忆的查找表机制相关联的组件的本地化编辑/忘记1)导致跨不同输入/输出格式的更健壮的编辑/忘记，以及2)抵制重新学习不想要的信息的尝试，同时还减少了与基线相比的意外副作用，在多个模型上的体育事实数据集和反事实数据集两者上。我们还发现，与其他基线相比，某些局部编辑对模型中潜在知识的破坏更大，使得遗忘对各种攻击更具健壮性。



## **13. WiS Platform: Enhancing Evaluation of LLM-Based Multi-Agent Systems Through Game-Based Analysis**

WiS平台：通过基于游戏的分析增强对基于LLM的多智能体系统的评估 cs.AI

**SubmitDate**: 2024-12-04    [abs](http://arxiv.org/abs/2412.03359v1) [paper-pdf](http://arxiv.org/pdf/2412.03359v1)

**Authors**: Chengwei Hu, Jianhui Zheng, Yancheng He, Hangyu Guo, Junguang Jiang, Han Zhu, Kai Sun, Yuning Jiang, Wenbo Su, Bo Zheng

**Abstract**: Recent advancements in autonomous multi-agent systems (MAS) based on large language models (LLMs) have enhanced the application scenarios and improved the capability of LLMs to handle complex tasks. Despite demonstrating effectiveness, existing studies still evidently struggle to evaluate, analysis, and reproducibility of LLM-based MAS. In this paper, to facilitate the research on LLM-based MAS, we introduce an open, scalable, and real-time updated platform for accessing and analyzing the LLM-based MAS based on the games Who is Spy?" (WiS). Our platform is featured with three main worths: (1) a unified model evaluate interface that supports models available on Hugging Face; (2) real-time updated leaderboard for model evaluation; (3) a comprehensive evaluation covering game-winning rates, attacking, defense strategies, and reasoning of LLMs. To rigorously test WiS, we conduct extensive experiments coverage of various open- and closed-source LLMs, we find that different agents exhibit distinct and intriguing behaviors in the game. The experimental results demonstrate the effectiveness and efficiency of our platform in evaluating LLM-based MAS. Our platform and its documentation are publicly available at \url{https://whoisspy.ai/}

摘要: 基于大语言模型的自治多智能体系统(MAS)的最新进展增强了LLMS的应用场景，提高了LLMS处理复杂任务的能力。尽管证明了有效性，但现有的研究显然仍难以评估、分析和重复性基于LLM的MAS。为了便于对基于LLM的MAS的研究，我们介绍了一个开放的、可扩展的、实时更新的访问和分析基于LLM的MAS的平台，该平台基于游戏《谁是间谍？(WIS)。我们的平台主要有三个特点：(1)统一的模型评估界面，支持拥抱脸上可用的模型；(2)实时更新的模型评估排行榜；(3)包括胜率、进攻、防守策略和LLMS推理的综合评估。为了严格测试WIS，我们对各种开放和封闭源代码的LLM进行了广泛的实验覆盖，我们发现不同的代理在游戏中表现出不同的和有趣的行为。实验结果证明了该平台在评估基于LLM的多代理系统中的有效性和高效性。我们的平台及其文档可在\url{https://whoisspy.ai/}



## **14. Time-Reversal Provides Unsupervised Feedback to LLMs**

计时器向LLM提供无监督反馈 cs.CL

Accepted as a spotlight in NeurIPS 2024

**SubmitDate**: 2024-12-04    [abs](http://arxiv.org/abs/2412.02626v2) [paper-pdf](http://arxiv.org/pdf/2412.02626v2)

**Authors**: Yerram Varun, Rahul Madhavan, Sravanti Addepalli, Arun Suggala, Karthikeyan Shanmugam, Prateek Jain

**Abstract**: Large Language Models (LLMs) are typically trained to predict in the forward direction of time. However, recent works have shown that prompting these models to look back and critique their own generations can produce useful feedback. Motivated by this, we explore the question of whether LLMs can be empowered to think (predict and score) backwards to provide unsupervised feedback that complements forward LLMs. Towards this, we introduce Time Reversed Language Models (TRLMs), which can score and generate queries when conditioned on responses, effectively functioning in the reverse direction of time. Further, to effectively infer in the response to query direction, we pre-train and fine-tune a language model (TRLM-Ba) in the reverse token order from scratch. We show empirically (and theoretically in a stylized setting) that time-reversed models can indeed complement forward model predictions when used to score the query given response for re-ranking multiple forward generations. We obtain up to 5\% improvement on the widely used AlpacaEval Leaderboard over the competent baseline of best-of-N re-ranking using self log-perplexity scores. We further show that TRLM scoring outperforms conventional forward scoring of response given query, resulting in significant gains in applications such as citation generation and passage retrieval. We next leverage the generative ability of TRLM to augment or provide unsupervised feedback to input safety filters of LLMs, demonstrating a drastic reduction in false negative rate with negligible impact on false positive rates against several attacks published on the popular JailbreakBench leaderboard.

摘要: 大型语言模型(LLM)通常被训练为在时间的正向进行预测。然而，最近的研究表明，促使这些模型回顾和批评他们自己的几代人可以产生有用的反馈。受此启发，我们探讨了LLM是否可以被赋予向后思考(预测和评分)的能力，以提供无监督的反馈来补充前向LLM。为此，我们引入了时间反转语言模型(TRLMS)，该模型可以根据响应进行评分并生成查询，有效地沿时间的相反方向运行。此外，为了有效地推断对查询方向的响应，我们从头开始以相反的令牌顺序预先训练和微调语言模型(TRLM-BA)。我们在经验上(理论上是在风格化的环境中)表明，当时间倒置模型用于对给定响应的查询进行重新排序时，时间倒置模型确实可以补充正向模型预测。我们在广泛使用的AlpacaEval排行榜上获得了高达5%的改进，超过了使用自我对数困惑分数重新排序的合格基线。我们进一步表明，TRLM评分优于传统的对给定查询的回复的前向评分，从而在引文生成和段落检索等应用中获得了显著的收益。接下来，我们利用TRLM的生成能力来增强或向LLMS的输入安全过滤器提供无监督反馈，展示了假阴性率的大幅降低，而对流行的JailBreak Btch排行榜上发布的几种攻击的错误确认率的影响可以忽略不计。



## **15. Does Safety Training of LLMs Generalize to Semantically Related Natural Prompts?**

LLM的安全培训是否适用于语义相关的自然知识？ cs.CL

Accepted at the Safe Generative AI Workshop @ NeurIPS 2024

**SubmitDate**: 2024-12-04    [abs](http://arxiv.org/abs/2412.03235v1) [paper-pdf](http://arxiv.org/pdf/2412.03235v1)

**Authors**: Sravanti Addepalli, Yerram Varun, Arun Suggala, Karthikeyan Shanmugam, Prateek Jain

**Abstract**: Large Language Models (LLMs) are known to be susceptible to crafted adversarial attacks or jailbreaks that lead to the generation of objectionable content despite being aligned to human preferences using safety fine-tuning methods. While the large dimensionality of input token space makes it inevitable to find adversarial prompts that can jailbreak these models, we aim to evaluate whether safety fine-tuned LLMs are safe against natural prompts which are semantically related to toxic seed prompts that elicit safe responses after alignment. We surprisingly find that popular aligned LLMs such as GPT-4 can be compromised using naive prompts that are NOT even crafted with an objective of jailbreaking the model. Furthermore, we empirically show that given a seed prompt that elicits a toxic response from an unaligned model, one can systematically generate several semantically related natural prompts that can jailbreak aligned LLMs. Towards this, we propose a method of Response Guided Question Augmentation (ReG-QA) to evaluate the generalization of safety aligned LLMs to natural prompts, that first generates several toxic answers given a seed question using an unaligned LLM (Q to A), and further leverages an LLM to generate questions that are likely to produce these answers (A to Q). We interestingly find that safety fine-tuned LLMs such as GPT-4o are vulnerable to producing natural jailbreak questions from unsafe content (without denial) and can thus be used for the latter (A to Q) step. We obtain attack success rates that are comparable to/ better than leading adversarial attack methods on the JailbreakBench leaderboard, while being significantly more stable against defenses such as Smooth-LLM and Synonym Substitution, which are effective against existing all attacks on the leaderboard.

摘要: 众所周知，大型语言模型(LLM)容易受到精心设计的对抗性攻击或越狱，尽管使用安全微调方法与人类的偏好保持一致，但这些攻击或越狱会导致生成令人反感的内容。虽然输入令牌空间的大维度使得找到能够越狱这些模型的敌意提示是不可避免的，但我们的目标是评估安全的微调LLM对于自然提示是否安全，这些自然提示在语义上与有毒种子提示相关，在对齐后引起安全响应。我们惊讶地发现，GPT-4等流行的对齐LLM可以使用甚至不是以越狱为目标而精心设计的幼稚提示来进行攻击。此外，我们的经验表明，给定一个种子提示引起来自未对齐模型的有毒反应，一个人可以系统地生成几个语义相关的自然提示，从而可以越狱对齐的LLM。为此，我们提出了一种反应引导问题增强方法(REG-QA)来评估安全对齐LLM对自然提示的泛化，该方法首先使用未对齐LLM(Q到A)来生成给定种子问题的几个有毒答案，然后利用LLM来生成可能产生这些答案(A到Q)的问题。有趣的是，我们发现安全微调的LLM，如GPT-40，容易从不安全的内容产生自然的越狱问题(不否认)，因此可以用于后一步(A到Q)。我们获得了相当于/好于JailBreak排行榜上领先的对抗性攻击方法的攻击成功率，同时对Smooth-LLM和同义词替换等防御措施明显更加稳定，这些防御措施对排行榜上现有的所有攻击都有效。



## **16. "Moralized" Multi-Step Jailbreak Prompts: Black-Box Testing of Guardrails in Large Language Models for Verbal Attacks**

“道德化”多步骤越狱预言：对大型语言模型中护栏进行黑匣子测试以进行言语攻击 cs.CR

This paper has been submitted to Nature Machine Intelligence and  OpenReview preprints. It has 7 pages of text, 3 figures, and 3 tables

**SubmitDate**: 2024-12-04    [abs](http://arxiv.org/abs/2411.16730v3) [paper-pdf](http://arxiv.org/pdf/2411.16730v3)

**Authors**: Libo Wang

**Abstract**: As the application of large language models continues to expand in various fields, it poses higher challenges to the effectiveness of identifying harmful content generation and guardrail mechanisms. This research aims to evaluate the guardrail effectiveness of GPT-4o, Grok-2 Beta, Llama 3.1 (405B), Gemini 1.5, and Claude 3.5 Sonnet through black-box testing of seemingly ethical multi-step jailbreak prompts. It conducts ethical attacks by designing an identical multi-step prompts that simulates the scenario of "corporate middle managers competing for promotions." The data results show that the guardrails of the above-mentioned LLMs were bypassed and the content of verbal attacks was generated. Claude 3.5 Sonnet's resistance to multi-step jailbreak prompts is more obvious. To ensure objectivity, the experimental process, black box test code, and enhanced guardrail code are uploaded to the GitHub repository: https://github.com/brucewang123456789/GeniusTrail.git.

摘要: 随着大型语言模型在各个领域的应用不断扩展，对识别有害内容生成和护栏机制的有效性提出了更高的挑战。这项研究旨在通过对看似合乎道德的多步越狱提示进行黑匣子测试来评估GPT-4 o、Grok-2 Beta、Llama 3.1（405 B）、Gemini 1.5和Claude 3.5十四行诗的护栏有效性。它通过设计相同的多步骤提示来进行道德攻击，模拟“企业中层管理人员竞争晋升”的场景。“数据结果显示，上述LLM的护栏被绕过，产生了言语攻击的内容。克劳德3.5十四行诗对多步越狱提示的抵制更加明显。为了确保客观性，实验过程、黑匣子测试代码和增强型护栏代码被上传到GitHub存储库：https://github.com/brucewang123456789/GeniusTrail.git。



## **17. Backdoor Attacks and Countermeasures in Natural Language Processing Models: A Comprehensive Security Review**

自然语言处理模型中的后门攻击和对策：全面的安全评论 cs.CR

21 pages, 3 figures

**SubmitDate**: 2024-12-04    [abs](http://arxiv.org/abs/2309.06055v5) [paper-pdf](http://arxiv.org/pdf/2309.06055v5)

**Authors**: Pengzhou Cheng, Zongru Wu, Wei Du, Haodong Zhao, Wei Lu, Gongshen Liu

**Abstract**: Language Models (LMs) are becoming increasingly popular in real-world applications. Outsourcing model training and data hosting to third-party platforms has become a standard method for reducing costs. In such a situation, the attacker can manipulate the training process or data to inject a backdoor into models. Backdoor attacks are a serious threat where malicious behavior is activated when triggers are present, otherwise, the model operates normally.   However, there is still no systematic and comprehensive review of LMs from the attacker's capabilities and purposes on different backdoor attack surfaces. Moreover, there is a shortage of analysis and comparison of the diverse emerging backdoor countermeasures. Therefore, this work aims to provide the NLP community with a timely review of backdoor attacks and countermeasures. According to the attackers' capability and affected stage of the LMs, the attack surfaces are formalized into four categorizations: attacking the pre-trained model with fine-tuning (APMF) or parameter-efficient fine-tuning (APMP), attacking the final model with training (AFMT), and attacking Large Language Models (ALLM). Thus, attacks under each categorization are combed. The countermeasures are categorized into two general classes: sample inspection and model inspection. Thus, we review countermeasures and analyze their advantages and disadvantages. Also, we summarize the benchmark datasets and provide comparable evaluations for representative attacks and defenses. Drawing the insights from the review, we point out the crucial areas for future research on the backdoor, especially soliciting more efficient and practical countermeasures.

摘要: 语言模型(LMS)在实际应用中正变得越来越流行。将模型培训和数据托管外包给第三方平台已成为降低成本的标准方法。在这种情况下，攻击者可以操纵训练过程或数据以向模型注入后门。后门攻击是一种严重的威胁，当存在触发器时，恶意行为被激活，否则，模型正常运行。然而，目前还没有从攻击者在不同的后门攻击面上的能力和目的对LMS进行系统和全面的审查。此外，对各种新出现的借壳对策缺乏分析和比较。因此，这项工作旨在为NLP社区提供及时审查后门攻击和对策的机会。根据攻击者的攻击能力和受影响阶段，将攻击面形式化为四类：精调攻击预训练模型(APMF)或参数高效微调攻击(APMP)、训练攻击最终模型(AFMT)和攻击大型语言模型(ALLM)。因此，对每个分类下的攻击进行了梳理。反制措施一般分为两大类：抽样检查和模型检查。因此，我们回顾了这些对策，并分析了它们的优缺点。此外，我们总结了基准数据集，并提供了具有代表性的攻击和防御的可比性评估。从回顾中得到的启示，我们指出了未来关于后门研究的关键领域，特别是寻求更有效和更实际的对策。



## **18. Unleashing GHOST: An LLM-Powered Framework for Automated Hardware Trojan Design**

释放GSTORE：一个由LLM支持的自动硬件特洛伊木马设计框架 cs.CR

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2412.02816v1) [paper-pdf](http://arxiv.org/pdf/2412.02816v1)

**Authors**: Md Omar Faruque, Peter Jamieson, Ahmad Patooghy, Abdel-Hameed A. Badawy

**Abstract**: Traditionally, inserting realistic Hardware Trojans (HTs) into complex hardware systems has been a time-consuming and manual process, requiring comprehensive knowledge of the design and navigating intricate Hardware Description Language (HDL) codebases. Machine Learning (ML)-based approaches have attempted to automate this process but often face challenges such as the need for extensive training data, long learning times, and limited generalizability across diverse hardware design landscapes. This paper addresses these challenges by proposing GHOST (Generator for Hardware-Oriented Stealthy Trojans), an automated attack framework that leverages Large Language Models (LLMs) for rapid HT generation and insertion. Our study evaluates three state-of-the-art LLMs - GPT-4, Gemini-1.5-pro, and Llama-3-70B - across three hardware designs: SRAM, AES, and UART. According to our evaluations, GPT-4 demonstrates superior performance, with 88.88% of HT insertion attempts successfully generating functional and synthesizable HTs. This study also highlights the security risks posed by LLM-generated HTs, showing that 100% of GHOST-generated synthesizable HTs evaded detection by an ML-based HT detection tool. These results underscore the urgent need for advanced detection and prevention mechanisms in hardware security to address the emerging threat of LLM-generated HTs. The GHOST HT benchmarks are available at: https://github.com/HSTRG1/GHOSTbenchmarks.git

摘要: 传统上，在复杂的硬件系统中插入真实硬件特洛伊木马(HTS)一直是一个耗时且手动的过程，需要全面的设计知识和导航复杂的硬件描述语言(HDL)代码库。基于机器学习(ML)的方法试图使这一过程自动化，但经常面临挑战，例如需要大量的训练数据、学习时间长以及在不同硬件设计环境中的推广有限。本文通过提出Ghost(面向硬件的隐身木马生成器)来应对这些挑战，Ghost是一个自动化攻击框架，它利用大型语言模型(LLM)来快速生成和插入HT。我们的研究评估了三种最先进的LLM-GPT-4、Gemini-1.5-PRO和Llama-3-70B-跨越三种硬件设计：SRAM、AES和UART。根据我们的评估，GPT-4表现出优越的性能，88.88%的HT插入尝试成功地生成了功能性和可合成的HTS。这项研究还强调了LLM生成的HTS带来的安全风险，表明100%的幽灵生成的可合成HTS可以躲避基于ML的HTS检测工具的检测。这些结果突显了在硬件安全方面迫切需要先进的检测和预防机制，以应对LLM生成的HTS的新威胁。Ghost HT基准可在以下网站获得：https://github.com/HSTRG1/GHOSTbenchmarks.git



## **19. Gracefully Filtering Backdoor Samples for Generative Large Language Models without Retraining**

优雅地过滤生成性大型语言模型的后门样本，无需重新训练 cs.CL

Accepted at COLING 2025

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2412.02454v1) [paper-pdf](http://arxiv.org/pdf/2412.02454v1)

**Authors**: Zongru Wu, Pengzhou Cheng, Lingyong Fang, Zhuosheng Zhang, Gongshen Liu

**Abstract**: Backdoor attacks remain significant security threats to generative large language models (LLMs). Since generative LLMs output sequences of high-dimensional token logits instead of low-dimensional classification logits, most existing backdoor defense methods designed for discriminative models like BERT are ineffective for generative LLMs. Inspired by the observed differences in learning behavior between backdoor and clean mapping in the frequency space, we transform gradients of each training sample, directly influencing parameter updates, into the frequency space. Our findings reveal a distinct separation between the gradients of backdoor and clean samples in the frequency space. Based on this phenomenon, we propose Gradient Clustering in the Frequency Space for Backdoor Sample Filtering (GraCeFul), which leverages sample-wise gradients in the frequency space to effectively identify backdoor samples without requiring retraining LLMs. Experimental results show that GraCeFul outperforms baselines significantly. Notably, GraCeFul exhibits remarkable computational efficiency, achieving nearly 100% recall and F1 scores in identifying backdoor samples, reducing the average success rate of various backdoor attacks to 0% with negligible drops in clean accuracy across multiple free-style question answering datasets. Additionally, GraCeFul generalizes to Llama-2 and Vicuna. The codes are publicly available at https://github.com/ZrW00/GraceFul.

摘要: 后门攻击仍然是生成性大型语言模型(LLM)的重大安全威胁。由于生成性LLMS输出的是高维令牌逻辑序列，而不是低维分类逻辑序列，现有的大多数后门防御方法都是针对BERT等区分模型设计的，对于生成性LLMS是无效的。受观察到的频率空间中后门映射和干净映射在学习行为上的差异的启发，我们将每个训练样本的梯度转换到频率空间中，这直接影响参数的更新。我们的发现表明，在频率空间中，后门样本和清洁样本的梯度之间存在明显的分离。基于这一现象，我们提出了在频率空间中进行后门样本滤波的梯度聚类(GRACEFUE)，它利用频率空间中的样本梯度来有效地识别后门样本，而不需要重新训练LLMS。实验结果表明，优雅算法的性能明显优于基线算法。值得注意的是，Graceful表现出了卓越的计算效率，在识别后门样本方面实现了近100%的Recall和F1分数，将各种后门攻击的平均成功率降低到0%，而跨多个自由风格问答数据集的干净准确率几乎可以忽略不计。此外，优雅适用于骆驼-2和维库纳。这些代码可在https://github.com/ZrW00/GraceFul.上公开获得



## **20. Jailbreak Large Vision-Language Models Through Multi-Modal Linkage**

通过多模式联动的越狱大型视觉语言模型 cs.CV

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2412.00473v2) [paper-pdf](http://arxiv.org/pdf/2412.00473v2)

**Authors**: Yu Wang, Xiaofei Zhou, Yichen Wang, Geyuan Zhang, Tianxing He

**Abstract**: With the significant advancement of Large Vision-Language Models (VLMs), concerns about their potential misuse and abuse have grown rapidly. Previous studies have highlighted VLMs' vulnerability to jailbreak attacks, where carefully crafted inputs can lead the model to produce content that violates ethical and legal standards. However, existing methods struggle against state-of-the-art VLMs like GPT-4o, due to the over-exposure of harmful content and lack of stealthy malicious guidance. In this work, we propose a novel jailbreak attack framework: Multi-Modal Linkage (MML) Attack. Drawing inspiration from cryptography, MML utilizes an encryption-decryption process across text and image modalities to mitigate over-exposure of malicious information. To align the model's output with malicious intent covertly, MML employs a technique called "evil alignment", framing the attack within a video game production scenario. Comprehensive experiments demonstrate MML's effectiveness. Specifically, MML jailbreaks GPT-4o with attack success rates of 97.80% on SafeBench, 98.81% on MM-SafeBench and 99.07% on HADES-Dataset. Our code is available at https://github.com/wangyu-ovo/MML

摘要: 随着大型视觉语言模型(VLM)的显著进步，人们对其潜在的滥用和滥用的担忧迅速增长。之前的研究已经强调了VLMS在越狱攻击中的脆弱性，在越狱攻击中，精心制作的输入可能导致该模型产生违反道德和法律标准的内容。然而，由于过度暴露有害内容和缺乏隐蔽的恶意指导，现有的方法难以对抗像GPT-40这样的最先进的VLM。在这项工作中，我们提出了一种新的越狱攻击框架：多模式联动攻击。MML从密码学中获得灵感，利用跨文本和图像通道的加密-解密过程来减少恶意信息的过度暴露。为了秘密地将模型的输出与恶意意图对齐，MML采用了一种称为“邪恶对齐”的技术，将攻击框置于视频游戏制作场景中。综合实验证明了MML的有效性。具体地说，MML越狱GPT-4o在SafeBitch上的攻击成功率为97.80%，在MM-SafeBch上的攻击成功率为98.81%，在HADES-DataSet上的攻击成功率为99.07%。我们的代码可以在https://github.com/wangyu-ovo/MML上找到



## **21. Harmful Fine-tuning Attacks and Defenses for Large Language Models: A Survey**

针对大型语言模型的有害微调攻击和防御：调查 cs.CR

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2409.18169v5) [paper-pdf](http://arxiv.org/pdf/2409.18169v5)

**Authors**: Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, Ling Liu

**Abstract**: Recent research demonstrates that the nascent fine-tuning-as-a-service business model exposes serious safety concerns -- fine-tuning over a few harmful data uploaded by the users can compromise the safety alignment of the model. The attack, known as harmful fine-tuning attack, has raised a broad research interest among the community. However, as the attack is still new, \textbf{we observe that there are general misunderstandings within the research community.} To clear up concern, this paper provide a comprehensive overview to three aspects of harmful fine-tuning: attacks setting, defense design and evaluation methodology. Specifically, we first present the threat model of the problem, and introduce the harmful fine-tuning attack and its variants. Then we systematically survey the existing literature on attacks/defenses/mechanical analysis of the problem. Finally, we introduce the evaluation methodology and outline future research directions that might contribute to the development of the field. Additionally, we present a list of questions of interest, which might be useful to refer to when reviewers in the peer review process question the realism of the experiment/attack/defense setting. A curated list of relevant papers is maintained and made accessible at: https://github.com/git-disl/awesome_LLM-harmful-fine-tuning-papers.

摘要: 最近的研究表明，新兴的微调即服务商业模式暴露了严重的安全问题--对用户上传的几个有害数据进行微调可能会损害该模型的安全一致性。这一被称为有害微调攻击的攻击在社区中引起了广泛的研究兴趣。然而，由于攻击仍然是新的，我们观察到研究界存在普遍的误解。}为了消除人们的担忧，本文对有害微调的三个方面进行了全面的概述：攻击设置、防御设计和评估方法。具体地说，我们首先给出了问题的威胁模型，并介绍了有害的微调攻击及其变体。然后，我们系统地综述了现有的关于攻击/防御/机械分析问题的文献。最后，我们介绍了评估方法，并概述了未来可能有助于该领域发展的研究方向。此外，我们提供了一个感兴趣的问题列表，当同行审查过程中的评审者质疑实验/攻击/防御设置的真实性时，这些问题可能会有用。相关论文的精选清单可在以下网址查阅：https://github.com/git-disl/awesome_LLM-harmful-fine-tuning-papers.



## **22. Trust & Safety of LLMs and LLMs in Trust & Safety**

LLM的信任与安全以及LLM的信任与安全 cs.AI

11 pages

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2412.02113v1) [paper-pdf](http://arxiv.org/pdf/2412.02113v1)

**Authors**: Doohee You, Dan Chon

**Abstract**: In recent years, Large Language Models (LLMs) have garnered considerable attention for their remarkable abilities in natural language processing tasks. However, their widespread adoption has raised concerns pertaining to trust and safety. This systematic review investigates the current research landscape on trust and safety in LLMs, with a particular focus on the novel application of LLMs within the field of Trust and Safety itself. We delve into the complexities of utilizing LLMs in domains where maintaining trust and safety is paramount, offering a consolidated perspective on this emerging trend.\   By synthesizing findings from various studies, we identify key challenges and potential solutions, aiming to benefit researchers and practitioners seeking to understand the nuanced interplay between LLMs and Trust and Safety.   This review provides insights on best practices for using LLMs in Trust and Safety, and explores emerging risks such as prompt injection and jailbreak attacks. Ultimately, this study contributes to a deeper understanding of how LLMs can be effectively and responsibly utilized to enhance trust and safety in the digital realm.

摘要: 近年来，大语言模型因其在自然语言处理任务中的卓越能力而受到广泛关注。然而，它们的广泛采用引发了人们对信任和安全的担忧。这篇系统的综述调查了当前关于低成本管理中信任和安全的研究现状，特别关注低成本管理在信任和安全领域的新应用。我们深入研究了在维护信任和安全至高无上的领域使用低成本管理的复杂性，为这一新兴趋势提供了一个综合的视角。\通过综合各种研究的结果，我们确定了关键的挑战和潜在的解决方案，旨在帮助寻求了解低成本管理与信任和安全之间微妙相互作用的研究人员和从业者。这篇综述提供了关于在信任与安全中使用LLMS的最佳实践的见解，并探索了新出现的风险，如快速注入和越狱攻击。最终，这项研究有助于更深入地理解如何有效和负责任地利用LLM来增强数字领域的信任和安全。



## **23. Trustful LLMs: Customizing and Grounding Text Generation with Knowledge Bases and Dual Decoders**

值得信赖的LLM：使用知识库和双解码器定制和基础文本生成 cs.CL

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2411.07870v3) [paper-pdf](http://arxiv.org/pdf/2411.07870v3)

**Authors**: Xiaofeng Zhu, Jaya Krishna Mandivarapu

**Abstract**: Although people are impressed by the content generation skills of large language models, the use of LLMs, such as ChatGPT, is limited by the domain grounding of the content. The correctness and groundedness of the generated content need to be based on a verified context, such as results from Retrieval-Augmented Generation (RAG). One important issue when adapting LLMs to a customized domain is that the generated responses are often incomplete, or the additions are not verified and may even be hallucinated. Prior studies on hallucination detection have focused on evaluation metrics, which are not easily adaptable to dynamic domains and can be vulnerable to attacks like jail-breaking. In this work, we propose 1) a post-processing algorithm that leverages knowledge triplets in RAG context to correct hallucinations and 2) a dual-decoder model that fuses RAG context to guide the generation process.

摘要: 尽管人们对大型语言模型的内容生成技能印象深刻，但ChatGPT等LLM的使用受到内容领域基础的限制。生成内容的正确性和可信度需要基于经过验证的上下文，例如检索增强生成（RAG）的结果。将LLM调整到定制域时的一个重要问题是，生成的响应通常不完整，或者添加未经验证，甚至可能出现幻觉。之前关于幻觉检测的研究集中在评估指标上，这些指标不容易适应动态领域，并且容易受到越狱等攻击。在这项工作中，我们提出了1）一种利用RAG上下文中的知识三重组来纠正幻觉的后处理算法，以及2）一种融合RAG上下文来指导生成过程的双解码器模型。



## **24. Towards Understanding Jailbreak Attacks in LLMs: A Representation Space Analysis**

了解LLC中的越狱攻击：表示空间分析 cs.CL

Accepted by EMNLP 2024 Main

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2406.10794v3) [paper-pdf](http://arxiv.org/pdf/2406.10794v3)

**Authors**: Yuping Lin, Pengfei He, Han Xu, Yue Xing, Makoto Yamada, Hui Liu, Jiliang Tang

**Abstract**: Large language models (LLMs) are susceptible to a type of attack known as jailbreaking, which misleads LLMs to output harmful contents. Although there are diverse jailbreak attack strategies, there is no unified understanding on why some methods succeed and others fail. This paper explores the behavior of harmful and harmless prompts in the LLM's representation space to investigate the intrinsic properties of successful jailbreak attacks. We hypothesize that successful attacks share some similar properties: They are effective in moving the representation of the harmful prompt towards the direction to the harmless prompts. We leverage hidden representations into the objective of existing jailbreak attacks to move the attacks along the acceptance direction, and conduct experiments to validate the above hypothesis using the proposed objective. We hope this study provides new insights into understanding how LLMs understand harmfulness information.

摘要: 大型语言模型（LLM）容易受到一种称为越狱的攻击，这种攻击会误导LLM输出有害内容。尽管越狱攻击策略多种多样，但对于为什么有些方法成功而另一些方法失败，人们并没有统一的理解。本文探讨了LLM表示空间中有害和无害提示的行为，以研究成功越狱攻击的内在属性。我们假设成功的攻击具有一些相似的属性：它们有效地将有害提示的表示移向无害提示的方向。我们将隐藏的表示利用到现有越狱攻击的目标中，以沿着接受方向移动攻击，并使用提出的目标进行实验来验证上述假设。我们希望这项研究为理解LLM如何理解有害信息提供新的见解。



## **25. Improved Large Language Model Jailbreak Detection via Pretrained Embeddings**

通过预训练嵌入改进的大语言模型越狱检测 cs.CR

Submitted to AICS 2025: https://aics.site

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2412.01547v1) [paper-pdf](http://arxiv.org/pdf/2412.01547v1)

**Authors**: Erick Galinkin, Martin Sablotny

**Abstract**: The adoption of large language models (LLMs) in many applications, from customer service chat bots and software development assistants to more capable agentic systems necessitates research into how to secure these systems. Attacks like prompt injection and jailbreaking attempt to elicit responses and actions from these models that are not compliant with the safety, privacy, or content policies of organizations using the model in their application. In order to counter abuse of LLMs for generating potentially harmful replies or taking undesirable actions, LLM owners must apply safeguards during training and integrate additional tools to block the LLM from generating text that abuses the model. Jailbreaking prompts play a vital role in convincing an LLM to generate potentially harmful content, making it important to identify jailbreaking attempts to block any further steps. In this work, we propose a novel approach to detect jailbreak prompts based on pairing text embeddings well-suited for retrieval with traditional machine learning classification algorithms. Our approach outperforms all publicly available methods from open source LLM security applications.

摘要: 从客户服务聊天机器人和软件开发助理到更有能力的代理系统，在许多应用程序中采用大型语言模型(LLM)，需要研究如何保护这些系统。诸如提示注入和越狱之类的攻击试图从这些模型引发响应和操作，这些响应和操作不符合在其应用程序中使用该模型的组织的安全、隐私或内容策略。为了防止LLMS被滥用来生成可能有害的回复或采取不受欢迎的行动，LLM所有者必须在培训期间应用安全措施，并集成其他工具来阻止LLM生成滥用该模型的文本。越狱提示在说服LLM生成潜在有害内容方面发挥着至关重要的作用，因此识别阻止任何进一步步骤的越狱尝试非常重要。在这项工作中，我们提出了一种新的基于文本嵌入的越狱提示检测方法，该方法适合于传统机器学习分类算法的检索。我们的方法比开源LLM安全应用程序中所有公开可用的方法都要好。



## **26. LUMIA: Linear probing for Unimodal and MultiModal Membership Inference Attacks leveraging internal LLM states**

LUMIA：利用内部LLM状态进行单模式和多模式成员资格推理攻击的线性探测 cs.CR

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2411.19876v2) [paper-pdf](http://arxiv.org/pdf/2411.19876v2)

**Authors**: Luis Ibanez-Lissen, Lorena Gonzalez-Manzano, Jose Maria de Fuentes, Nicolas Anciaux, Joaquin Garcia-Alfaro

**Abstract**: Large Language Models (LLMs) are increasingly used in a variety of applications, but concerns around membership inference have grown in parallel. Previous efforts focus on black-to-grey-box models, thus neglecting the potential benefit from internal LLM information. To address this, we propose the use of Linear Probes (LPs) as a method to detect Membership Inference Attacks (MIAs) by examining internal activations of LLMs. Our approach, dubbed LUMIA, applies LPs layer-by-layer to get fine-grained data on the model inner workings. We test this method across several model architectures, sizes and datasets, including unimodal and multimodal tasks. In unimodal MIA, LUMIA achieves an average gain of 15.71 % in Area Under the Curve (AUC) over previous techniques. Remarkably, LUMIA reaches AUC>60% in 65.33% of cases -- an increment of 46.80% against the state of the art. Furthermore, our approach reveals key insights, such as the model layers where MIAs are most detectable. In multimodal models, LPs indicate that visual inputs can significantly contribute to detect MIAs -- AUC>60% is reached in 85.90% of experiments.

摘要: 大型语言模型(LLM)越来越多地用于各种应用程序，但围绕成员关系推理的关注也在平行增长。以往的研究主要集中在黑灰盒模型上，从而忽略了LLM内部信息的潜在益处。为了解决这一问题，我们提出使用线性探测器(LP)作为一种方法，通过检查LLP的内部激活来检测成员身份推理攻击(MIA)。我们的方法，称为Lumia，逐层应用LP，以获得关于模型内部工作的细粒度数据。我们在几个模型体系结构、大小和数据集上测试了这种方法，包括单模和多模任务。在单峰MIA中，Lumia的曲线下面积(AUC)比以前的技术平均增加了15.71%。值得注意的是，Lumia在65.33%的情况下达到AUC>60%--与最先进的水平相比增加了46.80%。此外，我们的方法揭示了关键的见解，例如最容易检测到MIA的模型层。在多通道模型中，LP表明视觉输入对检测MIA有显著贡献-85.90%的实验达到了60%以上的AUC。



## **27. Recent Advances in Attack and Defense Approaches of Large Language Models**

大型语言模型攻击和防御方法的最新进展 cs.CR

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2409.03274v3) [paper-pdf](http://arxiv.org/pdf/2409.03274v3)

**Authors**: Jing Cui, Yishi Xu, Zhewei Huang, Shuchang Zhou, Jianbin Jiao, Junge Zhang

**Abstract**: Large Language Models (LLMs) have revolutionized artificial intelligence and machine learning through their advanced text processing and generating capabilities. However, their widespread deployment has raised significant safety and reliability concerns. Established vulnerabilities in deep neural networks, coupled with emerging threat models, may compromise security evaluations and create a false sense of security. Given the extensive research in the field of LLM security, we believe that summarizing the current state of affairs will help the research community better understand the present landscape and inform future developments. This paper reviews current research on LLM vulnerabilities and threats, and evaluates the effectiveness of contemporary defense mechanisms. We analyze recent studies on attack vectors and model weaknesses, providing insights into attack mechanisms and the evolving threat landscape. We also examine current defense strategies, highlighting their strengths and limitations. By contrasting advancements in attack and defense methodologies, we identify research gaps and propose future directions to enhance LLM security. Our goal is to advance the understanding of LLM safety challenges and guide the development of more robust security measures.

摘要: 大型语言模型(LLM)通过其先进的文本处理和生成能力，使人工智能和机器学习发生了革命性的变化。然而，它们的广泛部署引发了严重的安全和可靠性问题。深层神经网络中已建立的漏洞，再加上新出现的威胁模型，可能会危及安全评估，并造成一种错误的安全感。鉴于LLM安全领域的广泛研究，我们相信总结当前的事态将有助于研究界更好地了解目前的情况并为未来的发展提供信息。本文回顾了LLM漏洞和威胁的研究现状，并对现代防御机制的有效性进行了评估。我们分析了最近关于攻击载体和模型弱点的研究，提供了对攻击机制和不断演变的威胁环境的洞察。我们还研究了当前的防御战略，强调了它们的优势和局限性。通过对比攻击和防御方法的进展，我们发现了研究的差距，并提出了增强LLM安全的未来方向。我们的目标是促进对LLM安全挑战的理解，并指导开发更强大的安全措施。



## **28. BDefects4NN: A Backdoor Defect Database for Controlled Localization Studies in Neural Networks**

BDefects 4NN：用于神经网络受控定位研究的后门缺陷数据库 cs.SE

11 pages, accepted by ICSE 2025

**SubmitDate**: 2024-12-01    [abs](http://arxiv.org/abs/2412.00746v1) [paper-pdf](http://arxiv.org/pdf/2412.00746v1)

**Authors**: Yisong Xiao, Aishan Liu, Xinwei Zhang, Tianyuan Zhang, Tianlin Li, Siyuan Liang, Xianglong Liu, Yang Liu, Dacheng Tao

**Abstract**: Pre-trained large deep learning models are now serving as the dominant component for downstream middleware users and have revolutionized the learning paradigm, replacing the traditional approach of training from scratch locally. To reduce development costs, developers often integrate third-party pre-trained deep neural networks (DNNs) into their intelligent software systems. However, utilizing untrusted DNNs presents significant security risks, as these models may contain intentional backdoor defects resulting from the black-box training process. These backdoor defects can be activated by hidden triggers, allowing attackers to maliciously control the model and compromise the overall reliability of the intelligent software. To ensure the safe adoption of DNNs in critical software systems, it is crucial to establish a backdoor defect database for localization studies. This paper addresses this research gap by introducing BDefects4NN, the first backdoor defect database, which provides labeled backdoor-defected DNNs at the neuron granularity and enables controlled localization studies of defect root causes. In BDefects4NN, we define three defect injection rules and employ four representative backdoor attacks across four popular network architectures and three widely adopted datasets, yielding a comprehensive database of 1,654 backdoor-defected DNNs with four defect quantities and varying infected neurons. Based on BDefects4NN, we conduct extensive experiments on evaluating six fault localization criteria and two defect repair techniques, which show limited effectiveness for backdoor defects. Additionally, we investigate backdoor-defected models in practical scenarios, specifically in lane detection for autonomous driving and large language models (LLMs), revealing potential threats and highlighting current limitations in precise defect localization.

摘要: 预先训练的大型深度学习模型现在成为下游中间件用户的主导组件，并彻底改变了学习范式，取代了传统的从局部从头开始培训的方法。为了降低开发成本，开发人员经常将第三方预先训练的深度神经网络(DNN)集成到他们的智能软件系统中。然而，使用不受信任的DNN会带来重大的安全风险，因为这些模型可能包含由黑盒培训过程导致的故意后门缺陷。这些后门缺陷可以被隐藏的触发器激活，允许攻击者恶意控制模型，并损害智能软件的整体可靠性。为了确保在关键软件系统中安全地采用DNN，建立用于本地化研究的后门缺陷数据库是至关重要的。本文通过引入第一个后门缺陷数据库BDefects4NN来弥补这一研究空白，该数据库在神经元粒度上提供标记的后门缺陷DNN，并使对缺陷根本原因的受控定位研究成为可能。在BDefects4NN中，我们定义了三种缺陷注入规则，并在四种流行的网络体系结构和三个广泛采用的数据集上使用了四种典型的后门攻击，产生了一个包含1,654个后门缺陷DNN的全面数据库，其中包含四个缺陷量和不同的感染神经元。基于BDefects4NN，我们对六个故障定位准则和两个缺陷修复技术进行了广泛的实验，结果表明它们对后门缺陷的效果有限。此外，我们还研究了实际场景中的后门缺陷模型，特别是在自动驾驶和大型语言模型(LLM)的车道检测中，揭示了潜在的威胁，并强调了当前在精确缺陷定位方面的限制。



## **29. Evaluating Large Language Models' Capability to Launch Fully Automated Spear Phishing Campaigns: Validated on Human Subjects**

评估大型语言模型发起全自动鱼叉式网络钓鱼活动的能力：在人类受试者上进行验证 cs.CR

**SubmitDate**: 2024-11-30    [abs](http://arxiv.org/abs/2412.00586v1) [paper-pdf](http://arxiv.org/pdf/2412.00586v1)

**Authors**: Fred Heiding, Simon Lermen, Andrew Kao, Bruce Schneier, Arun Vishwanath

**Abstract**: In this paper, we evaluate the capability of large language models to conduct personalized phishing attacks and compare their performance with human experts and AI models from last year. We include four email groups with a combined total of 101 participants: A control group of arbitrary phishing emails, which received a click-through rate (recipient pressed a link in the email) of 12%, emails generated by human experts (54% click-through), fully AI-automated emails 54% (click-through), and AI emails utilizing a human-in-the-loop (56% click-through). Thus, the AI-automated attacks performed on par with human experts and 350% better than the control group. The results are a significant improvement from similar studies conducted last year, highlighting the increased deceptive capabilities of AI models. Our AI-automated emails were sent using a custom-built tool that automates the entire spear phishing process, including information gathering and creating personalized vulnerability profiles for each target. The AI-gathered information was accurate and useful in 88% of cases and only produced inaccurate profiles for 4% of the participants. We also use language models to detect the intention of emails. Claude 3.5 Sonnet scored well above 90% with low false-positive rates and detected several seemingly benign emails that passed human detection. Lastly, we analyze the economics of phishing, highlighting how AI enables attackers to target more individuals at lower cost and increase profitability by up to 50 times for larger audiences.

摘要: 在本文中，我们评估了大型语言模型进行个性化钓鱼攻击的能力，并将其性能与去年的人类专家和AI模型进行了比较。我们包括四个电子邮件组，总共有101名参与者：控制组的任意钓鱼电子邮件的点击率(收件人按下电子邮件中的链接)为12%，由人类专家生成的电子邮件(点击率为54%)，完全人工智能自动化的电子邮件(点击率为54%)，以及利用人在循环中的人工智能电子邮件(56%的点击率)。因此，人工智能自动攻击的表现与人类专家不相上下，比对照组好350%。与去年进行的类似研究相比，这一结果是一个显著的进步，突显了人工智能模型更强的欺骗性。我们的人工智能自动电子邮件是使用定制的工具发送的，该工具可以自动执行整个鱼叉式网络钓鱼过程，包括收集信息并为每个目标创建个性化的漏洞配置文件。人工智能收集的信息在88%的情况下是准确和有用的，只有4%的参与者产生了不准确的个人资料。我们还使用语言模型来检测电子邮件的意图。克劳德3.5十四行诗得分远高于90%，假阳性率很低，并检测到几封看似温和的电子邮件通过了人类的检测。最后，我们分析了钓鱼的经济学，强调了人工智能如何使攻击者能够以更低的成本瞄准更多的个人，并将更多受众的盈利能力提高高达50倍。



## **30. Uncovering Safety Risks of Large Language Models through Concept Activation Vector**

通过概念激活载体揭示大型语言模型的安全风险 cs.CL

10 pages, accepted at NeurIPS 2024

**SubmitDate**: 2024-11-30    [abs](http://arxiv.org/abs/2404.12038v5) [paper-pdf](http://arxiv.org/pdf/2404.12038v5)

**Authors**: Zhihao Xu, Ruixuan Huang, Changyu Chen, Xiting Wang

**Abstract**: Despite careful safety alignment, current large language models (LLMs) remain vulnerable to various attacks. To further unveil the safety risks of LLMs, we introduce a Safety Concept Activation Vector (SCAV) framework, which effectively guides the attacks by accurately interpreting LLMs' safety mechanisms. We then develop an SCAV-guided attack method that can generate both attack prompts and embedding-level attacks with automatically selected perturbation hyperparameters. Both automatic and human evaluations demonstrate that our attack method significantly improves the attack success rate and response quality while requiring less training data. Additionally, we find that our generated attack prompts may be transferable to GPT-4, and the embedding-level attacks may also be transferred to other white-box LLMs whose parameters are known. Our experiments further uncover the safety risks present in current LLMs. For example, in our evaluation of seven open-source LLMs, we observe an average attack success rate of 99.14%, based on the classic keyword-matching criterion. Finally, we provide insights into the safety mechanism of LLMs. The code is available at https://github.com/SproutNan/AI-Safety_SCAV.

摘要: 尽管进行了仔细的安全调整，但当前的大型语言模型(LLM)仍然容易受到各种攻击。为了进一步揭示LLMS的安全隐患，我们引入了安全概念激活向量(SCAV)框架，通过准确解释LLMS的安全机制来有效地指导攻击。然后，我们开发了一种SCAV引导的攻击方法，该方法可以生成攻击提示和带有自动选择的扰动超参数的嵌入级攻击。自动和人工评估都表明，我们的攻击方法在需要更少的训练数据的情况下，显著地提高了攻击成功率和响应质量。此外，我们发现我们生成的攻击提示可以转移到GPT-4上，嵌入级攻击也可以转移到参数已知的其他白盒LLM上。我们的实验进一步揭示了当前LLM中存在的安全风险。例如，在我们对7个开源LLM的评估中，基于经典的关键字匹配标准，我们观察到平均攻击成功率为99.14%。最后，我们对LLMS的安全机制提供了见解。代码可在https://github.com/SproutNan/AI-Safety_SCAV.上获得



## **31. Safety Alignment Backfires: Preventing the Re-emergence of Suppressed Concepts in Fine-tuned Text-to-Image Diffusion Models**

安全调整适得其反：防止被抑制的概念在微调的文本到图像扩散模型中重新出现 cs.AI

20 pages, 18 figures

**SubmitDate**: 2024-11-30    [abs](http://arxiv.org/abs/2412.00357v1) [paper-pdf](http://arxiv.org/pdf/2412.00357v1)

**Authors**: Sanghyun Kim, Moonseok Choi, Jinwoo Shin, Juho Lee

**Abstract**: Fine-tuning text-to-image diffusion models is widely used for personalization and adaptation for new domains. In this paper, we identify a critical vulnerability of fine-tuning: safety alignment methods designed to filter harmful content (e.g., nudity) can break down during fine-tuning, allowing previously suppressed content to resurface, even when using benign datasets. While this "fine-tuning jailbreaking" issue is known in large language models, it remains largely unexplored in text-to-image diffusion models. Our investigation reveals that standard fine-tuning can inadvertently undo safety measures, causing models to relearn harmful concepts that were previously removed and even exacerbate harmful behaviors. To address this issue, we present a novel but immediate solution called Modular LoRA, which involves training Safety Low-Rank Adaptation (LoRA) modules separately from Fine-Tuning LoRA components and merging them during inference. This method effectively prevents the re-learning of harmful content without compromising the model's performance on new tasks. Our experiments demonstrate that Modular LoRA outperforms traditional fine-tuning methods in maintaining safety alignment, offering a practical approach for enhancing the security of text-to-image diffusion models against potential attacks.

摘要: 微调的文本到图像扩散模型被广泛用于个性化和适应新领域。在本文中，我们确定了微调的一个关键漏洞：旨在过滤有害内容(例如裸露)的安全对齐方法在微调过程中可能会崩溃，从而允许先前被抑制的内容重新浮出水面，即使使用的是良性数据集。虽然这种“微调越狱”问题在大型语言模型中是已知的，但在文本到图像的扩散模型中，它在很大程度上仍未被探索。我们的调查显示，标准的微调可能会无意中取消安全措施，导致模型重新学习以前删除的有害概念，甚至加剧有害行为。为了解决这个问题，我们提出了一种新颖而直接的解决方案，称为模块化LORA，它包括从精调LORA组件中分离训练安全低阶自适应(LORA)模块，并在推理过程中将它们合并。这种方法有效地防止了有害内容的重新学习，而不会影响模型在新任务上的性能。我们的实验表明，模块化LORA在保持安全对齐方面优于传统的微调方法，为增强文本到图像扩散模型抵御潜在攻击的安全性提供了一种实用的方法。



## **32. When LLMs Go Online: The Emerging Threat of Web-Enabled LLMs**

当LLM上线时：支持Web的LLM的新兴威胁 cs.CR

**SubmitDate**: 2024-11-29    [abs](http://arxiv.org/abs/2410.14569v2) [paper-pdf](http://arxiv.org/pdf/2410.14569v2)

**Authors**: Hanna Kim, Minkyoo Song, Seung Ho Na, Seungwon Shin, Kimin Lee

**Abstract**: Recent advancements in Large Language Models (LLMs) have established them as agentic systems capable of planning and interacting with various tools. These LLM agents are often paired with web-based tools, enabling access to diverse sources and real-time information. Although these advancements offer significant benefits across various applications, they also increase the risk of malicious use, particularly in cyberattacks involving personal information. In this work, we investigate the risks associated with misuse of LLM agents in cyberattacks involving personal data. Specifically, we aim to understand: 1) how potent LLM agents can be when directed to conduct cyberattacks, 2) how cyberattacks are enhanced by web-based tools, and 3) how affordable and easy it becomes to launch cyberattacks using LLM agents. We examine three attack scenarios: the collection of Personally Identifiable Information (PII), the generation of impersonation posts, and the creation of spear-phishing emails. Our experiments reveal the effectiveness of LLM agents in these attacks: LLM agents achieved a precision of up to 95.9% in collecting PII, up to 93.9% of impersonation posts created by LLM agents were evaluated as authentic, and the click rate for links in spear phishing emails created by LLM agents reached up to 46.67%. Additionally, our findings underscore the limitations of existing safeguards in contemporary commercial LLMs, emphasizing the urgent need for more robust security measures to prevent the misuse of LLM agents.

摘要: 大型语言模型(LLM)的最新进展已将它们确立为能够规划各种工具并与其交互的代理系统。这些LLM代理通常与基于Web的工具配合使用，从而能够访问不同的来源和实时信息。虽然这些改进在各种应用程序中提供了显著的好处，但它们也增加了恶意使用的风险，特别是在涉及个人信息的网络攻击中。在这项工作中，我们调查了在涉及个人数据的网络攻击中滥用LLM代理的相关风险。具体地说，我们的目标是了解：1)LLM代理在被指示进行网络攻击时的威力有多大；2)基于Web的工具如何增强网络攻击；3)使用LLM代理发起网络攻击变得多么负担得起和容易。我们研究了三种攻击场景：收集个人身份信息(PII)、生成模拟帖子和创建鱼叉式网络钓鱼电子邮件。我们的实验显示了LLM代理在这些攻击中的有效性：LLM代理收集PII的准确率高达95.9%，LLM代理创建的模仿帖子被评估为可信的高达93.9%，LLM代理创建的鱼叉式钓鱼邮件中链接的点击率高达46.67%。此外，我们的研究结果强调了当代商业LLM现有保障措施的局限性，强调迫切需要采取更强有力的安全措施，以防止滥用LLM剂。



## **33. Ensemble Watermarks for Large Language Models**

大型语言模型的注册水印 cs.CL

9 pages in the main body. Code is available at  http://github.com/CommodoreEU/master-generation. arXiv admin note:  substantial text overlap with arXiv:2405.08400

**SubmitDate**: 2024-11-29    [abs](http://arxiv.org/abs/2411.19563v1) [paper-pdf](http://arxiv.org/pdf/2411.19563v1)

**Authors**: Georg Niess, Roman Kern

**Abstract**: The rapid advancement of large language models (LLMs) has made it increasingly difficult to distinguish between text written by humans and machines. While watermarks already exist for LLMs, they often lack flexibility, and struggle with attacks such as paraphrasing. To address these issues, we propose a multi-feature method for generating watermarks that combines multiple distinct watermark features into an ensemble watermark. Concretely, we combine acrostica and sensorimotor norms with the established red-green watermark to achieve a 98% detection rate. After a paraphrasing attack the performance remains high with 95% detection rate. The red-green feature alone as baseline achieves a detection rate of 49%. The evaluation of all feature combinations reveals that the ensemble of all three consistently has the highest detection rate across several LLMs and watermark strength settings. Due to the flexibility of combining features in the ensemble, various requirements and trade-offs can be addressed. Additionally, for all ensemble configurations the same detection function can be used without adaptations. This method is particularly of interest to facilitate accountability and prevent societal harm.

摘要: 大型语言模型(LLM)的快速发展使得区分人类和机器编写的文本变得越来越困难。虽然LLM已经存在水印，但它们往往缺乏灵活性，并与释义等攻击作斗争。为了解决这些问题，我们提出了一种多特征生成水印的方法，该方法将多个不同的水印特征组合成一个集成水印。具体地说，我们将肢端和感觉运动规范与所建立的红绿水印相结合，达到了98%的检测率。经过改写攻击后，性能保持在95%的高检测率。仅以红绿特征作为基线就能达到49%的检测率。对所有特征组合的评估表明，在几个LLM和水印强度设置中，所有三个特征组合的集成始终具有最高的检测率。由于可以灵活地组合整体中的功能，因此可以满足各种需求和权衡。此外，对于所有合奏配置，可以使用相同的检测功能，而无需进行适配。这种方法对促进问责和防止社会危害特别有意义。



## **34. InputSnatch: Stealing Input in LLM Services via Timing Side-Channel Attacks**

InputSnatch：通过定时侧通道攻击窃取LLM服务中的输入 cs.CR

**SubmitDate**: 2024-11-29    [abs](http://arxiv.org/abs/2411.18191v2) [paper-pdf](http://arxiv.org/pdf/2411.18191v2)

**Authors**: Xinyao Zheng, Husheng Han, Shangyi Shi, Qiyan Fang, Zidong Du, Xing Hu, Qi Guo

**Abstract**: Large language models (LLMs) possess extensive knowledge and question-answering capabilities, having been widely deployed in privacy-sensitive domains like finance and medical consultation. During LLM inferences, cache-sharing methods are commonly employed to enhance efficiency by reusing cached states or responses for the same or similar inference requests. However, we identify that these cache mechanisms pose a risk of private input leakage, as the caching can result in observable variations in response times, making them a strong candidate for a timing-based attack hint.   In this study, we propose a novel timing-based side-channel attack to execute input theft in LLMs inference. The cache-based attack faces the challenge of constructing candidate inputs in a large search space to hit and steal cached user queries. To address these challenges, we propose two primary components. The input constructor employs machine learning techniques and LLM-based approaches for vocabulary correlation learning while implementing optimized search mechanisms for generalized input construction. The time analyzer implements statistical time fitting with outlier elimination to identify cache hit patterns, continuously providing feedback to refine the constructor's search strategy. We conduct experiments across two cache mechanisms and the results demonstrate that our approach consistently attains high attack success rates in various applications. Our work highlights the security vulnerabilities associated with performance optimizations, underscoring the necessity of prioritizing privacy and security alongside enhancements in LLM inference.

摘要: 大型语言模型(LLM)具有广泛的知识和问答能力，已广泛应用于金融、医疗咨询等隐私敏感领域。在LLM推理期间，通常使用高速缓存共享方法来通过对相同或相似的推理请求重复使用高速缓存的状态或响应来提高效率。然而，我们发现这些缓存机制带来了私有输入泄漏的风险，因为缓存可能会导致响应时间的明显变化，从而使它们成为基于时间的攻击提示的有力候选者。在这项研究中，我们提出了一种新的基于时序的旁路攻击来执行LLMS推理中的输入窃取。基于缓存的攻击面临着在大搜索空间中构建候选输入以命中和窃取缓存的用户查询的挑战。为了应对这些挑战，我们提出了两个主要组成部分。输入构造器使用机器学习技术和基于LLM的方法进行词汇关联学习，同时实现优化的搜索机制来构建通用输入。时间分析器使用异常值消除来实现统计时间拟合，以识别缓存命中模式，并持续提供反馈以改进构造器的搜索策略。我们在两种缓存机制上进行了实验，结果表明，我们的方法在不同的应用中都取得了很高的攻击成功率。我们的工作突出了与性能优化相关的安全漏洞，强调了在增强LLM推理的同时优先考虑隐私和安全的必要性。



## **35. RePD: Defending Jailbreak Attack through a Retrieval-based Prompt Decomposition Process**

RePD：通过基于检索的即时分解过程防御越狱攻击 cs.CR

**SubmitDate**: 2024-11-29    [abs](http://arxiv.org/abs/2410.08660v3) [paper-pdf](http://arxiv.org/pdf/2410.08660v3)

**Authors**: Peiran Wang, Xiaogeng Liu, Chaowei Xiao

**Abstract**: In this study, we introduce RePD, an innovative attack Retrieval-based Prompt Decomposition framework designed to mitigate the risk of jailbreak attacks on large language models (LLMs). Despite rigorous pretraining and finetuning focused on ethical alignment, LLMs are still susceptible to jailbreak exploits. RePD operates on a one-shot learning model, wherein it accesses a database of pre-collected jailbreak prompt templates to identify and decompose harmful inquiries embedded within user prompts. This process involves integrating the decomposition of the jailbreak prompt into the user's original query into a one-shot learning example to effectively teach the LLM to discern and separate malicious components. Consequently, the LLM is equipped to first neutralize any potentially harmful elements before addressing the user's prompt in a manner that aligns with its ethical guidelines. RePD is versatile and compatible with a variety of open-source LLMs acting as agents. Through comprehensive experimentation with both harmful and benign prompts, we have demonstrated the efficacy of our proposed RePD in enhancing the resilience of LLMs against jailbreak attacks, without compromising their performance in responding to typical user requests.

摘要: 在这项研究中，我们介绍了RePD，一个创新的基于攻击检索的提示分解框架，旨在降低对大型语言模型(LLM)的越狱攻击风险。尽管严格的预训和微调侧重于道德一致性，但LLM仍然容易受到越狱利用的影响。RePD运行在一次性学习模式上，其中它访问预先收集的越狱提示模板数据库，以识别和分解嵌入用户提示中的有害查询。这一过程包括将越狱提示的分解集成到用户的原始查询中，并将其整合为一个一次性学习示例，以有效地教会LLM识别和分离恶意组件。因此，LLM配备了首先中和任何潜在有害元素，然后以符合其道德准则的方式处理用户的提示。RePD是通用的，并与各种作为代理的开源LLM兼容。通过对有害提示和良性提示的全面实验，我们已经证明了我们提出的RePD在增强LLM对越狱攻击的弹性方面的有效性，而不会影响它们响应典型用户请求的性能。



## **36. Confidential Prompting: Protecting User Prompts from Cloud LLM Providers**

机密预算：保护用户预算免受云LLM提供商的预算 cs.CR

**SubmitDate**: 2024-11-28    [abs](http://arxiv.org/abs/2409.19134v2) [paper-pdf](http://arxiv.org/pdf/2409.19134v2)

**Authors**: In Gim, Caihua Li, Lin Zhong

**Abstract**: Our work tackles the challenge of securing user inputs in cloud-hosted large language model (LLM) serving while ensuring output invariance, model confidentiality, and compute efficiency. We introduce secure multi-party decoding (SMD), which leverages confidential computing to confine user prompts to a trusted execution environment (TEE), namely a confidential virtual machine (CVM), while allowing service providers to generate tokens efficiently. We also introduce a novel cryptographic method, prompt obfuscation (PO), to ensure robustness against reconstruction attacks on SMD. We demonstrate that our approach preserves both prompt confidentiality and LLM serving efficiency. Our solution can enable privacy-preserving cloud LLM serving that handles sensitive prompts, such as clinical records, financial data, and personal information.

摘要: 我们的工作解决了在云托管大型语言模型（LLM）服务中保护用户输入的挑战，同时确保输出不变性、模型机密性和计算效率。我们引入了安全多方解码（MED），它利用机密计算将用户提示限制在可信执行环境（TEK），即机密虚拟机（CGM），同时允许服务提供商高效地生成令牌。我们还引入了一种新颖的加密方法--即时混淆（PO），以确保抵御对贴片的重建攻击的鲁棒性。我们证明我们的方法既保留了即时的机密性，又保留了LLM服务效率。我们的解决方案可以实现保护隐私的云LLM服务，该服务可以处理敏感提示，例如临床记录、财务数据和个人信息。



## **37. Memorization of Named Entities in Fine-tuned BERT Models**

微调BERT模型中命名实体的子化 cs.CL

published at CD-MAKE 2023

**SubmitDate**: 2024-11-28    [abs](http://arxiv.org/abs/2212.03749v3) [paper-pdf](http://arxiv.org/pdf/2212.03749v3)

**Authors**: Andor Diera, Nicolas Lell, Aygul Garifullina, Ansgar Scherp

**Abstract**: Privacy preserving deep learning is an emerging field in machine learning that aims to mitigate the privacy risks in the use of deep neural networks. One such risk is training data extraction from language models that have been trained on datasets, which contain personal and privacy sensitive information. In our study, we investigate the extent of named entity memorization in fine-tuned BERT models. We use single-label text classification as representative downstream task and employ three different fine-tuning setups in our experiments, including one with Differential Privacy (DP). We create a large number of text samples from the fine-tuned BERT models utilizing a custom sequential sampling strategy with two prompting strategies. We search in these samples for named entities and check if they are also present in the fine-tuning datasets. We experiment with two benchmark datasets in the domains of emails and blogs. We show that the application of DP has a detrimental effect on the text generation capabilities of BERT. Furthermore, we show that a fine-tuned BERT does not generate more named entities specific to the fine-tuning dataset than a BERT model that is pre-trained only. This suggests that BERT is unlikely to emit personal or privacy sensitive named entities. Overall, our results are important to understand to what extent BERT-based services are prone to training data extraction attacks.

摘要: 隐私保护深度学习是机器学习中的一个新兴领域，旨在降低深度神经网络使用中的隐私风险。其中一个风险是从已在数据集上训练的语言模型中提取训练数据，这些数据集包含个人和隐私敏感信息。在我们的研究中，我们考察了微调的BERT模型中命名实体记忆的程度。我们使用单标签文本分类作为代表性的下游任务，并在实验中使用了三种不同的微调设置，其中一种设置为差分隐私(DP)。我们利用定制的顺序采样策略和两种提示策略，从微调的BERT模型创建了大量的文本样本。我们在这些样本中搜索命名实体，并检查它们是否也出现在微调数据集中。我们在电子邮件和博客领域试验了两个基准数据集。结果表明，DP的应用对BERT的文本生成能力有不利影响。此外，我们还表明，与仅经过预训练的BERT模型相比，经过微调的ERT并不会生成更多特定于微调数据集的命名实体。这表明伯特不太可能发出个人或隐私敏感的命名实体。总体而言，我们的结果对于了解基于BERT的服务在多大程度上容易受到训练数据提取攻击具有重要意义。



## **38. On Evaluating The Performance of Watermarked Machine-Generated Texts Under Adversarial Attacks**

关于评估带有水印的机器生成文本在对抗性攻击下的性能 cs.CR

**SubmitDate**: 2024-11-28    [abs](http://arxiv.org/abs/2407.04794v2) [paper-pdf](http://arxiv.org/pdf/2407.04794v2)

**Authors**: Zesen Liu, Tianshuo Cong, Xinlei He, Qi Li

**Abstract**: Large Language Models (LLMs) excel in various applications, including text generation and complex tasks. However, the misuse of LLMs raises concerns about the authenticity and ethical implications of the content they produce, such as deepfake news, academic fraud, and copyright infringement. Watermarking techniques, which embed identifiable markers in machine-generated text, offer a promising solution to these issues by allowing for content verification and origin tracing. Unfortunately, the robustness of current LLM watermarking schemes under potential watermark removal attacks has not been comprehensively explored.   In this paper, to fill this gap, we first systematically comb the mainstream watermarking schemes and removal attacks on machine-generated texts, and then we categorize them into pre-text (before text generation) and post-text (after text generation) classes so that we can conduct diversified analyses. In our experiments, we evaluate eight watermarks (five pre-text, three post-text) and twelve attacks (two pre-text, ten post-text) across 87 scenarios. Evaluation results indicate that (1) KGW and Exponential watermarks offer high text quality and watermark retention but remain vulnerable to most attacks; (2) Post-text attacks are found to be more efficient and practical than pre-text attacks; (3) Pre-text watermarks are generally more imperceptible, as they do not alter text fluency, unlike post-text watermarks; (4) Additionally, combined attack methods can significantly increase effectiveness, highlighting the need for more robust watermarking solutions. Our study underscores the vulnerabilities of current techniques and the necessity for developing more resilient schemes.

摘要: 大型语言模型(LLM)在各种应用中表现出色，包括文本生成和复杂任务。然而，LLMS的滥用引发了人们对它们产生的内容的真实性和伦理影响的担忧，例如深度假新闻、学术欺诈和侵犯版权。在机器生成的文本中嵌入可识别标记的水印技术，通过允许内容验证和来源追踪，为这些问题提供了一种有前途的解决方案。遗憾的是，目前的LLM水印方案在潜在的水印去除攻击下的稳健性还没有得到全面的研究。为了填补这一空白，本文首先对主流的机器生成文本水印算法和去除攻击进行了系统的梳理，然后将其分为前文本类(文本生成前)和后文本类(文本生成后)，以便进行多样化的分析。在我们的实验中，我们评估了87个场景中的8个水印(5个前置文本，3个后置文本)和12个攻击(2个前置文本，10个后置文本)。评估结果表明：(1)KGW和指数水印具有高的文本质量和水印保留率，但仍然容易受到大多数攻击；(2)后文本攻击被发现比前文本攻击更有效和实用；(3)前文本水印通常更不可察觉，因为它们不像后文本水印那样改变文本的流畅性；(4)此外，组合攻击方法可以显著提高攻击效果，突出了对更健壮的水印解决方案的需求。我们的研究强调了当前技术的脆弱性，以及开发更具弹性的方案的必要性。



## **39. Assessing biomedical knowledge robustness in large language models by query-efficient sampling attacks**

通过查询高效抽样攻击评估大型语言模型中生物医学知识的稳健性 cs.CL

31 pages incl. appendix, accepted by TMLR

**SubmitDate**: 2024-11-28    [abs](http://arxiv.org/abs/2402.10527v3) [paper-pdf](http://arxiv.org/pdf/2402.10527v3)

**Authors**: R. Patrick Xian, Alex J. Lee, Satvik Lolla, Vincent Wang, Qiming Cui, Russell Ro, Reza Abbasi-Asl

**Abstract**: The increasing depth of parametric domain knowledge in large language models (LLMs) is fueling their rapid deployment in real-world applications. Understanding model vulnerabilities in high-stakes and knowledge-intensive tasks is essential for quantifying the trustworthiness of model predictions and regulating their use. The recent discovery of named entities as adversarial examples (i.e. adversarial entities) in natural language processing tasks raises questions about their potential impact on the knowledge robustness of pre-trained and finetuned LLMs in high-stakes and specialized domains. We examined the use of type-consistent entity substitution as a template for collecting adversarial entities for billion-parameter LLMs with biomedical knowledge. To this end, we developed an embedding-space attack based on powerscaled distance-weighted sampling to assess the robustness of their biomedical knowledge with a low query budget and controllable coverage. Our method has favorable query efficiency and scaling over alternative approaches based on random sampling and blackbox gradient-guided search, which we demonstrated for adversarial distractor generation in biomedical question answering. Subsequent failure mode analysis uncovered two regimes of adversarial entities on the attack surface with distinct characteristics and we showed that entity substitution attacks can manipulate token-wise Shapley value explanations, which become deceptive in this setting. Our approach complements standard evaluations for high-capacity models and the results highlight the brittleness of domain knowledge in LLMs.

摘要: 大型语言模型(LLM)中参数领域知识的不断深入推动了它们在现实世界应用程序中的快速部署。了解高风险和知识密集型任务中的模型脆弱性对于量化模型预测的可信度和规范其使用至关重要。最近在自然语言处理任务中发现了命名实体作为对抗性实例(即对抗性实体)，这引发了人们对高风险和专门领域中预先训练和精细调整的LLM知识稳健性的潜在影响的问题。我们研究了使用类型一致的实体替换作为收集具有生物医学知识的10亿参数LLM的对抗性实体的模板。为此，我们提出了一种基于加权距离加权抽样的嵌入空间攻击方法，以较低的查询预算和可控的覆盖率来评估他们的生物医学知识的稳健性。与基于随机抽样和黑盒梯度引导搜索的方法相比，我们的方法具有良好的查询效率和伸缩性，并在生物医学问答中的对抗性干扰项生成中得到了验证。随后的失效模式分析揭示了攻击面上具有不同特征的两种对抗实体的机制，我们表明实体替换攻击可以操纵令人信服的Shapley值解释，在这种情况下，这种解释变得具有欺骗性。我们的方法补充了对大容量模型的标准评估，结果突出了领域知识在LLMS中的脆性。



## **40. Knowledge Database or Poison Base? Detecting RAG Poisoning Attack through LLM Activations**

知识库还是毒库？通过LLM激活检测RAG中毒攻击 cs.CR

**SubmitDate**: 2024-11-28    [abs](http://arxiv.org/abs/2411.18948v1) [paper-pdf](http://arxiv.org/pdf/2411.18948v1)

**Authors**: Xue Tan, Hao Luan, Mingyu Luo, Xiaoyan Sun, Ping Chen, Jun Dai

**Abstract**: As Large Language Models (LLMs) are progressively deployed across diverse fields and real-world applications, ensuring the security and robustness of LLMs has become ever more critical. Retrieval-Augmented Generation (RAG) is a cutting-edge approach designed to address the limitations of large language models (LLMs). By retrieving information from the relevant knowledge database, RAG enriches the input to LLMs, enabling them to produce responses that are more accurate and contextually appropriate. It is worth noting that the knowledge database, being sourced from publicly available channels such as Wikipedia, inevitably introduces a new attack surface. RAG poisoning involves injecting malicious texts into the knowledge database, ultimately leading to the generation of the attacker's target response (also called poisoned response). However, there are currently limited methods available for detecting such poisoning attacks. We aim to bridge the gap in this work. Particularly, we introduce RevPRAG, a flexible and automated detection pipeline that leverages the activations of LLMs for poisoned response detection. Our investigation uncovers distinct patterns in LLMs' activations when generating correct responses versus poisoned responses. Our results on multiple benchmark datasets and RAG architectures show our approach could achieve 98% true positive rate, while maintaining false positive rates close to 1%. We also evaluate recent backdoor detection methods specifically designed for LLMs and applicable for identifying poisoned responses in RAG. The results demonstrate that our approach significantly surpasses them.

摘要: 随着大型语言模型(LLM)在不同领域和实际应用中的逐步部署，确保LLM的安全性和健壮性变得越来越重要。检索-增强生成(RAG)是一种尖端方法，旨在解决大型语言模型(LLM)的局限性。通过从相关知识数据库中检索信息，RAG丰富了对LLMS的输入，使它们能够做出更准确和更适合具体情况的答复。值得注意的是，来自维基百科等公开渠道的知识数据库不可避免地引入了新的攻击面。RAG中毒涉及将恶意文本注入知识库，最终导致生成攻击者的目标响应(也称为中毒响应)。然而，目前可用于检测此类中毒攻击的方法有限。我们的目标是弥合这项工作中的差距。特别是，我们引入了RevPRAG，这是一种灵活的自动化检测管道，它利用LLM的激活来进行中毒响应检测。我们的研究揭示了LLMS在产生正确反应和中毒反应时激活的不同模式。我们在多个基准数据集和RAG体系结构上的结果表明，我们的方法可以达到98%的真阳性率，同时将假阳性率保持在接近1%的水平。我们还评估了最近专门为LLMS设计的、适用于识别RAG中的中毒反应的后门检测方法。结果表明，我们的方法明显地超过了它们。



## **41. SceneTAP: Scene-Coherent Typographic Adversarial Planner against Vision-Language Models in Real-World Environments**

SceneRAP：针对现实世界环境中视觉语言模型的场景一致印刷对抗规划器 cs.CV

**SubmitDate**: 2024-11-28    [abs](http://arxiv.org/abs/2412.00114v1) [paper-pdf](http://arxiv.org/pdf/2412.00114v1)

**Authors**: Yue Cao, Yun Xing, Jie Zhang, Di Lin, Tianwei Zhang, Ivor Tsang, Yang Liu, Qing Guo

**Abstract**: Large vision-language models (LVLMs) have shown remarkable capabilities in interpreting visual content. While existing works demonstrate these models' vulnerability to deliberately placed adversarial texts, such texts are often easily identifiable as anomalous. In this paper, we present the first approach to generate scene-coherent typographic adversarial attacks that mislead advanced LVLMs while maintaining visual naturalness through the capability of the LLM-based agent. Our approach addresses three critical questions: what adversarial text to generate, where to place it within the scene, and how to integrate it seamlessly. We propose a training-free, multi-modal LLM-driven scene-coherent typographic adversarial planning (SceneTAP) that employs a three-stage process: scene understanding, adversarial planning, and seamless integration. The SceneTAP utilizes chain-of-thought reasoning to comprehend the scene, formulate effective adversarial text, strategically plan its placement, and provide detailed instructions for natural integration within the image. This is followed by a scene-coherent TextDiffuser that executes the attack using a local diffusion mechanism. We extend our method to real-world scenarios by printing and placing generated patches in physical environments, demonstrating its practical implications. Extensive experiments show that our scene-coherent adversarial text successfully misleads state-of-the-art LVLMs, including ChatGPT-4o, even after capturing new images of physical setups. Our evaluations demonstrate a significant increase in attack success rates while maintaining visual naturalness and contextual appropriateness. This work highlights vulnerabilities in current vision-language models to sophisticated, scene-coherent adversarial attacks and provides insights into potential defense mechanisms.

摘要: 大型视觉语言模型(LVLM)在解释视觉内容方面表现出了非凡的能力。虽然现有的工作证明了这些模型对故意放置的对抗性文本的脆弱性，但这样的文本往往很容易被识别为异常。在这篇文章中，我们提出了第一种方法来生成场景连贯的排版攻击，这些攻击误导了高级LVLM，同时通过基于LLM的代理来保持视觉自然度。我们的方法解决了三个关键问题：生成什么对抗性文本，将其放置在场景中的哪里，以及如何无缝集成它。我们提出了一种无需训练的多模式LLM驱动的场景连贯排版对抗规划(SceneTap)，它采用了一个三个阶段的过程：场景理解、对抗规划和无缝集成。SceneTap利用思维链推理来理解场景，制定有效的对抗性文本，战略性地规划其位置，并为图像中的自然融合提供详细说明。紧随其后的是使用本地扩散机制执行攻击的场景一致的TextDiffuser。我们通过在物理环境中打印和放置生成的补丁，将我们的方法扩展到现实世界的场景中，展示了它的实际意义。广泛的实验表明，我们的场景连贯的对抗性文本成功地误导了最先进的LVLM，包括ChatGPT-40，即使在捕捉到物理设置的新图像之后也是如此。我们的评估表明，在保持视觉自然性和上下文适当性的同时，攻击成功率显著增加。这项工作突出了当前视觉语言模型对复杂的、场景连贯的对抗性攻击的脆弱性，并提供了对潜在防御机制的见解。



## **42. Cyber-Attack Technique Classification Using Two-Stage Trained Large Language Models**

使用两阶段训练的大型语言模型的网络攻击技术分类 cs.LG

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.18755v1) [paper-pdf](http://arxiv.org/pdf/2411.18755v1)

**Authors**: Weiqiu You, Youngja Park

**Abstract**: Understanding the attack patterns associated with a cyberattack is crucial for comprehending the attacker's behaviors and implementing the right mitigation measures. However, majority of the information regarding new attacks is typically presented in unstructured text, posing significant challenges for security analysts in collecting necessary information. In this paper, we present a sentence classification system that can identify the attack techniques described in natural language sentences from cyber threat intelligence (CTI) reports. We propose a new method for utilizing auxiliary data with the same labels to improve classification for the low-resource cyberattack classification task. The system first trains the model using the augmented training data and then trains more using only the primary data. We validate our model using the TRAM data1 and the MITRE ATT&CK framework. Experiments show that our method enhances Macro-F1 by 5 to 9 percentage points and keeps Micro-F1 scores competitive when compared to the baseline performance on the TRAM dataset.

摘要: 了解与网络攻击相关的攻击模式对于了解攻击者的行为和实施正确的缓解措施至关重要。然而，有关新攻击的大多数信息通常是以非结构化文本形式呈现的，这给安全分析师在收集必要信息方面带来了巨大的挑战。本文提出了一种句子分类系统，可以从网络威胁情报(CTI)报告中识别自然语言句子中描述的攻击技术。针对低资源网络攻击分类任务，提出了一种利用相同标签的辅助数据来改进分类的新方法。该系统首先使用扩充的训练数据训练模型，然后仅使用原始数据训练更多的模型。我们使用TRAM数据1和MITRE ATT&CK框架对我们的模型进行了验证。实验表明，与TRAM数据集上的基准性能相比，我们的方法将Macro-F1提高了5到9个百分点，并保持了Micro-F1的竞争力。



## **43. Immune: Improving Safety Against Jailbreaks in Multi-modal LLMs via Inference-Time Alignment**

免疫：通过推理时间对齐提高多模式LLM中越狱的安全性 cs.CR

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.18688v1) [paper-pdf](http://arxiv.org/pdf/2411.18688v1)

**Authors**: Soumya Suvra Ghosal, Souradip Chakraborty, Vaibhav Singh, Tianrui Guan, Mengdi Wang, Ahmad Beirami, Furong Huang, Alvaro Velasquez, Dinesh Manocha, Amrit Singh Bedi

**Abstract**: With the widespread deployment of Multimodal Large Language Models (MLLMs) for visual-reasoning tasks, improving their safety has become crucial. Recent research indicates that despite training-time safety alignment, these models remain vulnerable to jailbreak attacks: carefully crafted image-prompt pairs that compel the model to generate harmful content. In this work, we first highlight a critical safety gap, demonstrating that alignment achieved solely through safety training may be insufficient against jailbreak attacks. To address this vulnerability, we propose Immune, an inference-time defense framework that leverages a safe reward model during decoding to defend against jailbreak attacks. Additionally, we provide a rigorous mathematical characterization of Immune, offering provable guarantees against jailbreaks. Extensive evaluations on diverse jailbreak benchmarks using recent MLLMs reveal that Immune effectively enhances model safety while preserving the model's original capabilities. For instance, against text-based jailbreak attacks on LLaVA-1.6, Immune reduces the attack success rate by 57.82% and 16.78% compared to the base MLLM and state-of-the-art defense strategy, respectively.

摘要: 随着多通道大语言模型(MLLMS)在视觉推理任务中的广泛应用，提高其安全性变得至关重要。最近的研究表明，尽管训练时间安全一致，这些模型仍然容易受到越狱攻击：精心制作的图像提示对迫使模型生成有害内容。在这项工作中，我们首先强调一个关键的安全差距，表明仅通过安全培训实现的对准可能不足以抵御越狱袭击。为了解决这个漏洞，我们提出了免疫，这是一个推理时间防御框架，它在解码过程中利用安全的奖励模型来防御越狱攻击。此外，我们提供了免疫的严格数学特征，提供了针对越狱的可证明的保证。使用最近的MLLMS对不同的越狱基准进行的广泛评估表明，免疫有效地增强了模型的安全性，同时保持了模型的原始能力。例如，对于基于文本的越狱攻击LLaVA-1.6，与基本MLLM和最先进的防御策略相比，免疫分别将攻击成功率降低了57.82%和16.78%。



## **44. Neutralizing Backdoors through Information Conflicts for Large Language Models**

通过大型语言模型的信息冲突消除后门 cs.CL

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.18280v1) [paper-pdf](http://arxiv.org/pdf/2411.18280v1)

**Authors**: Chen Chen, Yuchen Sun, Xueluan Gong, Jiaxin Gao, Kwok-Yan Lam

**Abstract**: Large language models (LLMs) have seen significant advancements, achieving superior performance in various Natural Language Processing (NLP) tasks, from understanding to reasoning. However, they remain vulnerable to backdoor attacks, where models behave normally for standard queries but generate harmful responses or unintended output when specific triggers are activated. Existing backdoor defenses often suffer from drawbacks that they either focus on detection without removal, rely on rigid assumptions about trigger properties, or prove to be ineffective against advanced attacks like multi-trigger backdoors. In this paper, we present a novel method to eliminate backdoor behaviors from LLMs through the construction of information conflicts using both internal and external mechanisms. Internally, we leverage a lightweight dataset to train a conflict model, which is then merged with the backdoored model to neutralize malicious behaviors by embedding contradictory information within the model's parametric memory. Externally, we incorporate convincing contradictory evidence into the prompt to challenge the model's internal backdoor knowledge. Experimental results on classification and conversational tasks across 4 widely used LLMs demonstrate that our method outperforms 8 state-of-the-art backdoor defense baselines. We can reduce the attack success rate of advanced backdoor attacks by up to 98% while maintaining over 90% clean data accuracy. Furthermore, our method has proven to be robust against adaptive backdoor attacks. The code will be open-sourced upon publication.

摘要: 大型语言模型(LLM)已经取得了显著的进步，在从理解到推理的各种自然语言处理(NLP)任务中取得了优异的性能。然而，它们仍然容易受到后门攻击，在后门攻击中，模型对标准查询正常操作，但在激活特定触发器时会生成有害响应或意外输出。现有的后门防御往往存在缺陷，要么专注于检测而不移除，依赖于对触发属性的僵化假设，要么被证明对多触发后门等高级攻击无效。在本文中，我们提出了一种新的方法，通过利用内部和外部机制构建信息冲突来消除LLMS中的后门行为。在内部，我们利用轻量级数据集来训练冲突模型，然后将其与后置模型合并，通过在模型的参数记忆中嵌入相互矛盾的信息来中和恶意行为。在外部，我们将令人信服的相互矛盾的证据结合到提示中，以挑战模型的内部后门知识。在4个广泛使用的LLM上的分类和会话任务上的实验结果表明，该方法的性能优于8个最先进的后门防御基线。我们可以将高级后门攻击的攻击成功率降低高达98%，同时保持90%以上的干净数据准确性。此外，我们的方法已被证明对自适应后门攻击具有健壮性。代码将在发布后开放源代码。



## **45. Visual Adversarial Attack on Vision-Language Models for Autonomous Driving**

自动驾驶视觉语言模型的视觉对抗攻击 cs.CV

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.18275v1) [paper-pdf](http://arxiv.org/pdf/2411.18275v1)

**Authors**: Tianyuan Zhang, Lu Wang, Xinwei Zhang, Yitong Zhang, Boyi Jia, Siyuan Liang, Shengshan Hu, Qiang Fu, Aishan Liu, Xianglong Liu

**Abstract**: Vision-language models (VLMs) have significantly advanced autonomous driving (AD) by enhancing reasoning capabilities. However, these models remain highly vulnerable to adversarial attacks. While existing research has primarily focused on general VLM attacks, the development of attacks tailored to the safety-critical AD context has been largely overlooked. In this paper, we take the first step toward designing adversarial attacks specifically targeting VLMs in AD, exposing the substantial risks these attacks pose within this critical domain. We identify two unique challenges for effective adversarial attacks on AD VLMs: the variability of textual instructions and the time-series nature of visual scenarios. To this end, we propose ADvLM, the first visual adversarial attack framework specifically designed for VLMs in AD. Our framework introduces Semantic-Invariant Induction, which uses a large language model to create a diverse prompt library of textual instructions with consistent semantic content, guided by semantic entropy. Building on this, we introduce Scenario-Associated Enhancement, an approach where attention mechanisms select key frames and perspectives within driving scenarios to optimize adversarial perturbations that generalize across the entire scenario. Extensive experiments on several AD VLMs over multiple benchmarks show that ADvLM achieves state-of-the-art attack effectiveness. Moreover, real-world attack studies further validate its applicability and potential in practice.

摘要: 视觉语言模型通过增强推理能力极大地促进了自动驾驶(AD)。然而，这些模型仍然非常容易受到对手的攻击。虽然现有的研究主要集中在一般的VLM攻击上，但针对安全关键型AD环境而定制的攻击的发展在很大程度上被忽视了。在本文中，我们向设计专门针对AD中的VLM的对抗性攻击迈出了第一步，暴露了这些攻击在这一关键领域中构成的实质性风险。我们确定了对AD VLMS进行有效的对抗性攻击的两个独特的挑战：文本指令的可变性和视觉场景的时间序列性质。为此，我们提出了ADvLM，这是第一个专门为AD中的VLM设计的可视化对抗性攻击框架。我们的框架引入了语义不变归纳法，它使用一个大型语言模型来创建一个具有一致语义内容的多样化提示库，并以语义熵为指导。在此基础上，我们引入了与场景相关的增强，这是一种注意机制在驾驶场景中选择关键帧和视角以优化整个场景中概括的对抗性扰动的方法。在多个基准上对多个AD VLM进行的大量实验表明，ADvLM达到了最先进的攻击效率。此外，真实世界的攻击研究进一步验证了其在实践中的适用性和潜力。



## **46. G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks**

G-Designer：通过图神经网络构建多智能体通信布局 cs.MA

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2410.11782v2) [paper-pdf](http://arxiv.org/pdf/2410.11782v2)

**Authors**: Guibin Zhang, Yanwei Yue, Xiangguo Sun, Guancheng Wan, Miao Yu, Junfeng Fang, Kun Wang, Dawei Cheng

**Abstract**: Recent advancements in large language model (LLM)-based agents have demonstrated that collective intelligence can significantly surpass the capabilities of individual agents, primarily due to well-crafted inter-agent communication topologies. Despite the diverse and high-performing designs available, practitioners often face confusion when selecting the most effective pipeline for their specific task: \textit{Which topology is the best choice for my task, avoiding unnecessary communication token overhead while ensuring high-quality solution?} In response to this dilemma, we introduce G-Designer, an adaptive, efficient, and robust solution for multi-agent deployment, which dynamically designs task-aware, customized communication topologies. Specifically, G-Designer models the multi-agent system as a multi-agent network, leveraging a variational graph auto-encoder to encode both the nodes (agents) and a task-specific virtual node, and decodes a task-adaptive and high-performing communication topology. Extensive experiments on six benchmarks showcase that G-Designer is: \textbf{(1) high-performing}, achieving superior results on MMLU with accuracy at $84.50\%$ and on HumanEval with pass@1 at $89.90\%$; \textbf{(2) task-adaptive}, architecting communication protocols tailored to task difficulty, reducing token consumption by up to $95.33\%$ on HumanEval; and \textbf{(3) adversarially robust}, defending against agent adversarial attacks with merely $0.3\%$ accuracy drop.

摘要: 基于大型语言模型(LLM)的代理的最新进展表明，集体智能可以显著超过单个代理的能力，这主要是由于精心设计的代理间通信拓扑。尽管有多样化和高性能的设计，但实践者在为他们的特定任务选择最有效的流水线时经常面临困惑：\textit{哪个拓扑是我的任务的最佳选择，在确保高质量解决方案的同时避免不必要的通信令牌开销？}针对这种困境，我们引入了G-Designer，这是一个自适应的、高效的、健壮的多代理部署解决方案，它动态地设计任务感知的、定制的通信拓扑。具体地说，G-Designer将多代理系统建模为多代理网络，利用变化图自动编码器对节点(代理)和特定于任务的虚拟节点进行编码，并解码任务自适应的高性能通信拓扑。在六个基准测试上的广泛实验表明，G-Designer是：\extbf{(1)高性能}，在MMLU上获得了更好的结果，准确率为84.50\$，在HumanEval上，PASS@1的准确率为89.90\$；\extbf{(2)任务自适应}，构建了针对任务难度的通信协议，在HumanEval上减少了高达95.33\$的令牌消耗；以及\extbf{(3)对手健壮性}，防御代理对手攻击，精确度仅下降了$0.3\%$。



## **47. Transferable Ensemble Black-box Jailbreak Attacks on Large Language Models**

可转移集成黑匣子越狱攻击大型语言模型 cs.CR

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2410.23558v2) [paper-pdf](http://arxiv.org/pdf/2410.23558v2)

**Authors**: Yiqi Yang, Hongye Fu

**Abstract**: In this report, we propose a novel black-box jailbreak attacking framework that incorporates various LLM-as-Attacker methods to deliver transferable and powerful jailbreak attacks. Our method is designed based on three key observations from existing jailbreaking studies and practices. First, we consider an ensemble approach should be more effective in exposing the vulnerabilities of an aligned LLM compared to individual attacks. Second, different malicious instructions inherently vary in their jailbreaking difficulty, necessitating differentiated treatment to ensure more efficient attacks. Finally, the semantic coherence of a malicious instruction is crucial for triggering the defenses of an aligned LLM; therefore, it must be carefully disrupted to manipulate its embedding representation, thereby increasing the jailbreak success rate. We validated our approach by participating in the Competition for LLM and Agent Safety 2024, where our team achieved top performance in the Jailbreaking Attack Track.

摘要: 在这份报告中，我们提出了一种新的黑盒越狱攻击框架，该框架结合了各种LLM作为攻击者的方法来提供可转移的强大越狱攻击。我们的方法是基于现有越狱研究和实践中的三个关键观察结果而设计的。首先，我们认为，与单独攻击相比，整体方法应该更有效地暴露联合LLM的漏洞。其次，不同的恶意指令在越狱难度上存在内在差异，需要区别对待，以确保更有效的攻击。最后，恶意指令的语义一致性对于触发对齐的LLM的防御至关重要；因此，必须小心破坏它才能操纵其嵌入表示，从而提高越狱成功率。我们通过参加LLM和代理安全竞赛2024来验证我们的方法，我们的团队在越狱攻击赛道上取得了最好的表现。



## **48. Evaluating and Improving the Robustness of Security Attack Detectors Generated by LLMs**

评估和改进LLM生成的安全攻击检测器的鲁棒性 cs.SE

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.18216v1) [paper-pdf](http://arxiv.org/pdf/2411.18216v1)

**Authors**: Samuele Pasini, Jinhan Kim, Tommaso Aiello, Rocio Cabrera Lozoya, Antonino Sabetta, Paolo Tonella

**Abstract**: Large Language Models (LLMs) are increasingly used in software development to generate functions, such as attack detectors, that implement security requirements. However, LLMs struggle to generate accurate code, resulting, e.g., in attack detectors that miss well-known attacks when used in practice. This is most likely due to the LLM lacking knowledge about some existing attacks and to the generated code being not evaluated in real usage scenarios. We propose a novel approach integrating Retrieval Augmented Generation (RAG) and Self-Ranking into the LLM pipeline. RAG enhances the robustness of the output by incorporating external knowledge sources, while the Self-Ranking technique, inspired to the concept of Self-Consistency, generates multiple reasoning paths and creates ranks to select the most robust detector. Our extensive empirical study targets code generated by LLMs to detect two prevalent injection attacks in web security: Cross-Site Scripting (XSS) and SQL injection (SQLi). Results show a significant improvement in detection performance compared to baselines, with an increase of up to 71%pt and 37%pt in the F2-Score for XSS and SQLi detection, respectively.

摘要: 大型语言模型(LLM)越来越多地用于软件开发，以生成实现安全要求的功能，如攻击检测器。然而，LLMS很难生成准确的代码，导致例如攻击检测器在实际使用时错过了众所周知的攻击。这很可能是因为LLM缺乏关于一些现有攻击的知识，并且生成的代码在实际使用场景中没有得到评估。我们提出了一种新的方法，将检索增强生成(RAG)和自我排序结合到LLM流水线中。RAG通过引入外部知识源来增强输出的稳健性，而自排序技术则受到自相容概念的启发，生成多条推理路径并创建排序来选择最健壮的检测器。我们广泛的经验研究针对LLMS生成的代码来检测网络安全中两种流行的注入攻击：跨站点脚本(XSS)和SQL注入(SQLI)。结果表明，与基线相比，检测性能有了显著提高，XSS和SQLI检测的F2分数分别增加了71%和37%。



## **49. Playing Language Game with LLMs Leads to Jailbreaking**

与法学硕士玩语言游戏导致越狱 cs.CL

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.12762v2) [paper-pdf](http://arxiv.org/pdf/2411.12762v2)

**Authors**: Yu Peng, Zewen Long, Fangming Dong, Congyi Li, Shu Wu, Kai Chen

**Abstract**: The advent of large language models (LLMs) has spurred the development of numerous jailbreak techniques aimed at circumventing their security defenses against malicious attacks. An effective jailbreak approach is to identify a domain where safety generalization fails, a phenomenon known as mismatched generalization. In this paper, we introduce two novel jailbreak methods based on mismatched generalization: natural language games and custom language games, both of which effectively bypass the safety mechanisms of LLMs, with various kinds and different variants, making them hard to defend and leading to high attack rates. Natural language games involve the use of synthetic linguistic constructs and the actions intertwined with these constructs, such as the Ubbi Dubbi language. Building on this phenomenon, we propose the custom language games method: by engaging with LLMs using a variety of custom rules, we successfully execute jailbreak attacks across multiple LLM platforms. Extensive experiments demonstrate the effectiveness of our methods, achieving success rates of 93% on GPT-4o, 89% on GPT-4o-mini and 83% on Claude-3.5-Sonnet. Furthermore, to investigate the generalizability of safety alignments, we fine-tuned Llama-3.1-70B with the custom language games to achieve safety alignment within our datasets and found that when interacting through other language games, the fine-tuned models still failed to identify harmful content. This finding indicates that the safety alignment knowledge embedded in LLMs fails to generalize across different linguistic formats, thus opening new avenues for future research in this area.

摘要: 大型语言模型(LLM)的出现促进了许多越狱技术的发展，这些技术旨在绕过针对恶意攻击的安全防御。一种有效的越狱方法是识别安全泛化失败的域，这种现象称为不匹配泛化。本文介绍了两种新的基于不匹配泛化的越狱方法：自然语言游戏和自定义语言游戏，这两种方法都有效地绕过了LLMS的安全机制，种类繁多，变体不同，使得它们难以防御，导致攻击率很高。自然语言游戏涉及使用合成的语言结构以及与这些结构交织在一起的动作，如Ubbi Dubbi语言。基于这一现象，我们提出了定制语言游戏方法：通过使用各种定制规则与LLM接触，我们成功地跨多个LLM平台执行越狱攻击。大量实验证明了该方法的有效性，在GPT-4o、GPT-4o-mini和Claude-3.5-十四行诗上分别获得了93%、89%和83%的识别成功率。此外，为了调查安全对齐的泛化能力，我们使用自定义语言游戏对Llama-3.1-70B进行了微调，以在我们的数据集中实现安全对齐，发现当通过其他语言游戏交互时，微调的模型仍然无法识别有害内容。这一发现表明，LLMS中嵌入的安全对齐知识无法跨不同的语言格式进行泛化，从而为这一领域的未来研究开辟了新的途径。



## **50. BlackDAN: A Black-Box Multi-Objective Approach for Effective and Contextual Jailbreaking of Large Language Models**

BlackDAN：一种有效且上下文化的大型语言模型越狱的黑匣子多目标方法 cs.CR

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2410.09804v3) [paper-pdf](http://arxiv.org/pdf/2410.09804v3)

**Authors**: Xinyuan Wang, Victor Shea-Jay Huang, Renmiao Chen, Hao Wang, Chengwei Pan, Lei Sha, Minlie Huang

**Abstract**: While large language models (LLMs) exhibit remarkable capabilities across various tasks, they encounter potential security risks such as jailbreak attacks, which exploit vulnerabilities to bypass security measures and generate harmful outputs. Existing jailbreak strategies mainly focus on maximizing attack success rate (ASR), frequently neglecting other critical factors, including the relevance of the jailbreak response to the query and the level of stealthiness. This narrow focus on single objectives can result in ineffective attacks that either lack contextual relevance or are easily recognizable. In this work, we introduce BlackDAN, an innovative black-box attack framework with multi-objective optimization, aiming to generate high-quality prompts that effectively facilitate jailbreaking while maintaining contextual relevance and minimizing detectability. BlackDAN leverages Multiobjective Evolutionary Algorithms (MOEAs), specifically the NSGA-II algorithm, to optimize jailbreaks across multiple objectives including ASR, stealthiness, and semantic relevance. By integrating mechanisms like mutation, crossover, and Pareto-dominance, BlackDAN provides a transparent and interpretable process for generating jailbreaks. Furthermore, the framework allows customization based on user preferences, enabling the selection of prompts that balance harmfulness, relevance, and other factors. Experimental results demonstrate that BlackDAN outperforms traditional single-objective methods, yielding higher success rates and improved robustness across various LLMs and multimodal LLMs, while ensuring jailbreak responses are both relevant and less detectable.

摘要: 虽然大型语言模型(LLM)在各种任务中显示出非凡的能力，但它们遇到了潜在的安全风险，如越狱攻击，这些攻击利用漏洞绕过安全措施并产生有害的输出。现有的越狱策略主要关注最大化攻击成功率(ASR)，往往忽略了其他关键因素，包括越狱响应与查询的相关性和隐蔽性水平。这种对单一目标的狭隘关注可能会导致无效的攻击，要么缺乏上下文相关性，要么很容易识别。在这项工作中，我们引入了BlackDAN，一个创新的多目标优化的黑盒攻击框架，旨在生成高质量的提示，在保持上下文相关性的同时有效地促进越狱，并将可检测性降至最低。BlackDAN利用多目标进化算法(MOEA)，特别是NSGA-II算法，跨多个目标优化越狱，包括ASR、隐蔽性和语义相关性。通过集成变异、交叉和帕累托支配等机制，BlackDAN为生成越狱提供了一个透明和可解释的过程。此外，该框架允许根据用户偏好进行定制，从而能够选择在危害性、相关性和其他因素之间进行权衡的提示。实验结果表明，BlackDAN的性能优于传统的单目标方法，在各种LLM和多模式LLM上获得了更高的成功率和更好的鲁棒性，同时确保了越狱响应的相关性和较低的可检测性。



