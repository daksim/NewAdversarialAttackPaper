# Latest Large Language Model Attack Papers
**update at 2025-02-19 09:54:02**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Improving Acoustic Side-Channel Attacks on Keyboards Using Transformers and Large Language Models**

使用变形金刚和大型语言模型改善对键盘的声学侧通道攻击 cs.LG

We will reflect comments from the reviewers and re-submit

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.09782v2) [paper-pdf](http://arxiv.org/pdf/2502.09782v2)

**Authors**: Jin Hyun Park, Seyyed Ali Ayati, Yichen Cai

**Abstract**: The increasing prevalence of microphones in everyday devices and the growing reliance on online services have amplified the risk of acoustic side-channel attacks (ASCAs) targeting keyboards. This study explores deep learning techniques, specifically vision transformers (VTs) and large language models (LLMs), to enhance the effectiveness and applicability of such attacks. We present substantial improvements over prior research, with the CoAtNet model achieving state-of-the-art performance. Our CoAtNet shows a 5.0% improvement for keystrokes recorded via smartphone (Phone) and 5.9% for those recorded via Zoom compared to previous benchmarks. We also evaluate transformer architectures and language models, with the best VT model matching CoAtNet's performance. A key advancement is the introduction of a noise mitigation method for real-world scenarios. By using LLMs for contextual understanding, we detect and correct erroneous keystrokes in noisy environments, enhancing ASCA performance. Additionally, fine-tuned lightweight language models with Low-Rank Adaptation (LoRA) deliver comparable performance to heavyweight models with 67X more parameters. This integration of VTs and LLMs improves the practical applicability of ASCA mitigation, marking the first use of these technologies to address ASCAs and error correction in real-world scenarios.

摘要: 麦克风在日常设备中的日益普及以及对在线服务的日益依赖，放大了针对键盘的声学侧通道攻击(ASCA)的风险。本研究探索深度学习技术，特别是视觉转换器(VT)和大语言模型(LLM)，以增强此类攻击的有效性和适用性。与之前的研究相比，我们提出了实质性的改进，CoAtNet模型实现了最先进的性能。我们的CoAtNet显示，与之前的基准相比，通过智能手机(手机)记录的击键次数提高了5.0%，通过Zoom记录的击键次数提高了5.9%。我们还评估了转换器体系结构和语言模型，选择了与CoAtNet性能匹配的最佳VT模型。一个关键的进步是引入了一种用于真实世界场景的噪音缓解方法。通过使用LLMS进行上下文理解，我们可以在噪声环境中检测并纠正错误的击键，从而提高ASCA的性能。此外，带有低阶自适应(LORA)的微调轻量级语言模型提供了与参数多67倍的重量级模型相当的性能。VTS和LLMS的这种集成提高了ASCA缓解的实际适用性，标志着这些技术首次用于解决现实世界场景中的ASCA和纠错。



## **2. FedEAT: A Robustness Optimization Framework for Federated LLMs**

FedEAT：联邦LLM的稳健性优化框架 cs.LG

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11863v1) [paper-pdf](http://arxiv.org/pdf/2502.11863v1)

**Authors**: Yahao Pang, Xingyuan Wu, Xiaojin Zhang, Wei Chen, Hai Jin

**Abstract**: Significant advancements have been made by Large Language Models (LLMs) in the domains of natural language understanding and automated content creation. However, they still face persistent problems, including substantial computational costs and inadequate availability of training data. The combination of Federated Learning (FL) and LLMs (federated LLMs) offers a solution by leveraging distributed data while protecting privacy, which positions it as an ideal choice for sensitive domains. However, Federated LLMs still suffer from robustness challenges, including data heterogeneity, malicious clients, and adversarial attacks, which greatly hinder their applications. We first introduce the robustness problems in federated LLMs, to address these challenges, we propose FedEAT (Federated Embedding space Adversarial Training), a novel framework that applies adversarial training in the embedding space of client LLM and employs a robust aggregation approach, specifically geometric median aggregation, to enhance the robustness of Federated LLMs. Our experiments demonstrate that FedEAT effectively improves the robustness of Federated LLMs with minimal performance loss.

摘要: 大型语言模型(LLM)在自然语言理解和自动内容创建领域取得了重大进展。然而，它们仍然面临着长期存在的问题，包括巨大的计算成本和培训数据的不足。联合学习(FL)和联合LLMS(联合LLMS)的结合提供了一种在保护隐私的同时利用分布式数据的解决方案，这将其定位为敏感领域的理想选择。然而，联邦LLMS仍然面临着健壮性挑战，包括数据异构性、恶意客户端和敌意攻击，这些都极大地阻碍了它们的应用。首先介绍了联合LLMS的健壮性问题，针对这些问题，我们提出了一种新的框架FedEAT(Federated Embedding Space Adversal Trading)，该框架将对抗性训练应用于客户端LLMS的嵌入空间，并采用一种稳健的聚集方法，特别是几何中值聚集来增强联合LLMS的健壮性。实验结果表明，FedEAT算法以最小的性能损失有效地提高了联邦LLMS的健壮性。



## **3. StructTransform: A Scalable Attack Surface for Safety-Aligned Large Language Models**

StructChange：安全一致的大型语言模型的可扩展攻击表面 cs.LG

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11853v1) [paper-pdf](http://arxiv.org/pdf/2502.11853v1)

**Authors**: Shehel Yoosuf, Temoor Ali, Ahmed Lekssays, Mashael AlSabah, Issa Khalil

**Abstract**: In this work, we present a series of structure transformation attacks on LLM alignment, where we encode natural language intent using diverse syntax spaces, ranging from simple structure formats and basic query languages (e.g. SQL) to new novel spaces and syntaxes created entirely by LLMs. Our extensive evaluation shows that our simplest attacks can achieve close to 90% success rate, even on strict LLMs (such as Claude 3.5 Sonnet) using SOTA alignment mechanisms. We improve the attack performance further by using an adaptive scheme that combines structure transformations along with existing \textit{content transformations}, resulting in over 96% ASR with 0% refusals.   To generalize our attacks, we explore numerous structure formats, including syntaxes purely generated by LLMs. Our results indicate that such novel syntaxes are easy to generate and result in a high ASR, suggesting that defending against our attacks is not a straightforward process. Finally, we develop a benchmark and evaluate existing safety-alignment defenses against it, showing that most of them fail with 100% ASR. Our results show that existing safety alignment mostly relies on token-level patterns without recognizing harmful concepts, highlighting and motivating the need for serious research efforts in this direction. As a case study, we demonstrate how attackers can use our attack to easily generate a sample malware, and a corpus of fraudulent SMS messages, which perform well in bypassing detection.

摘要: 在这项工作中，我们提出了一系列针对LLM对齐的结构转换攻击，其中我们使用不同的语法空间来编码自然语言意图，从简单的结构格式和基本查询语言(例如SQL)到完全由LLMS创建的新的新颖空间和句法。我们广泛的评估表明，我们最简单的攻击可以达到接近90%的成功率，即使是在使用SOTA对齐机制的严格LLM(如Claude 3.5十四行诗)上也是如此。通过使用结构变换和现有的文本内容变换相结合的自适应方案，进一步提高了攻击性能，获得了96%以上的ASR和0%的拒绝。为了概括我们的攻击，我们探索了许多结构格式，包括纯粹由LLMS生成的语法。我们的结果表明，这种新的语法很容易生成，并导致高ASR，这表明防御我们的攻击并不是一个简单的过程。最后，我们开发了一个基准，并对现有的安全对齐防御进行了评估，结果表明，大多数安全对齐防御都是100%ASR失败的。我们的结果表明，现有的安全对齐主要依赖于令牌级模式，而没有识别有害的概念，这突显和激励了在这一方向上认真研究的必要性。作为一个案例研究，我们演示了攻击者如何利用我们的攻击轻松生成样本恶意软件和欺诈性短信语料库，它们在绕过检测方面表现良好。



## **4. BaxBench: Can LLMs Generate Correct and Secure Backends?**

收件箱长凳：LLM能否生成正确且安全的后台？ cs.CR

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11844v1) [paper-pdf](http://arxiv.org/pdf/2502.11844v1)

**Authors**: Mark Vero, Niels Mündler, Victor Chibotaru, Veselin Raychev, Maximilian Baader, Nikola Jovanović, Jingxuan He, Martin Vechev

**Abstract**: The automatic generation of programs has long been a fundamental challenge in computer science. Recent benchmarks have shown that large language models (LLMs) can effectively generate code at the function level, make code edits, and solve algorithmic coding tasks. However, to achieve full automation, LLMs should be able to generate production-quality, self-contained application modules. To evaluate the capabilities of LLMs in solving this challenge, we introduce BaxBench, a novel evaluation benchmark consisting of 392 tasks for the generation of backend applications. We focus on backends for three critical reasons: (i) they are practically relevant, building the core components of most modern web and cloud software, (ii) they are difficult to get right, requiring multiple functions and files to achieve the desired functionality, and (iii) they are security-critical, as they are exposed to untrusted third-parties, making secure solutions that prevent deployment-time attacks an imperative. BaxBench validates the functionality of the generated applications with comprehensive test cases, and assesses their security exposure by executing end-to-end exploits. Our experiments reveal key limitations of current LLMs in both functionality and security: (i) even the best model, OpenAI o1, achieves a mere 60% on code correctness; (ii) on average, we could successfully execute security exploits on more than half of the correct programs generated by each LLM; and (iii) in less popular backend frameworks, models further struggle to generate correct and secure applications. Progress on BaxBench signifies important steps towards autonomous and secure software development with LLMs.

摘要: 长期以来，程序的自动生成一直是计算机科学中的一个基本挑战。最近的基准测试表明，大型语言模型(LLM)可以有效地在函数级生成代码、进行代码编辑和解决算法编码任务。然而，为了实现完全自动化，LLMS应该能够生成生产质量的、独立的应用程序模块。为了评估LLMS在解决这一挑战方面的能力，我们引入了BaxBch，这是一个新的评估基准，由392个任务组成，用于生成后端应用程序。我们关注后端有三个关键原因：(I)它们实际上是相关的，构建了大多数现代Web和云软件的核心组件；(Ii)它们很难正确使用，需要多种功能和文件才能实现所需的功能；(Iii)它们是安全关键型的，因为它们可能会暴露在不受信任的第三方面前，这使得防止部署时攻击的安全解决方案成为当务之急。BaxBtch使用全面的测试用例验证生成的应用程序的功能，并通过执行端到端漏洞攻击来评估它们的安全暴露。我们的实验揭示了当前LLM在功能和安全性方面的关键局限性：(I)即使是最好的模型OpenAI o1，代码正确性也只有60%；(Ii)平均而言，我们可以在每个LLM生成的正确程序中成功地执行安全漏洞；以及(Iii)在不太流行的后端框架中，模型进一步难以生成正确和安全的应用程序。BaxBtch的进展标志着使用LLMS进行自主和安全的软件开发迈出了重要的一步。



## **5. Separate the Wheat from the Chaff: A Post-Hoc Approach to Safety Re-Alignment for Fine-Tuned Language Models**

将小麦与谷壳分开：精调语言模型的安全重新对齐的事后方法 cs.CL

16 pages, 14 figures,

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2412.11041v2) [paper-pdf](http://arxiv.org/pdf/2412.11041v2)

**Authors**: Di Wu, Xin Lu, Yanyan Zhao, Bing Qin

**Abstract**: Although large language models (LLMs) achieve effective safety alignment at the time of release, they still face various safety challenges. A key issue is that fine-tuning often compromises the safety alignment of LLMs. To address this issue, we propose a method named IRR (Identify, Remove, and Recalibrate for Safety Realignment) that performs safety realignment for LLMs. The core of IRR is to identify and remove unsafe delta parameters from the fine-tuned models, while recalibrating the retained ones. We evaluate the effectiveness of IRR across various datasets, including both full fine-tuning and LoRA methods. Our results demonstrate that IRR significantly enhances the safety performance of fine-tuned models on safety benchmarks, such as harmful queries and jailbreak attacks, while maintaining their performance on downstream tasks. The source code is available at: https://anonymous.4open.science/r/IRR-BD4F.

摘要: 尽管大型语言模型（LLM）在发布时实现了有效的安全一致，但它们仍然面临各种安全挑战。一个关键问题是微调通常会损害LLM的安全对齐。为了解决这个问题，我们提出了一种名为IRR（识别、删除和重新校准以实现安全重新对准）的方法，该方法为LLM执行安全重新对准。IRR的核心是从微调模型中识别并删除不安全的Delta参数，同时重新校准保留的参数。我们评估IRR在各种数据集的有效性，包括完全微调和LoRA方法。我们的结果表明，IRR显着增强了经过微调的模型在安全基准（例如有害查询和越狱攻击）上的安全性能，同时保持了其在下游任务上的性能。源代码可访问：www.example.com。



## **6. DELMAN: Dynamic Defense Against Large Language Model Jailbreaking with Model Editing**

德尔曼：通过模型编辑对大型语言模型越狱的动态防御 cs.CR

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11647v1) [paper-pdf](http://arxiv.org/pdf/2502.11647v1)

**Authors**: Yi Wang, Fenghua Weng, Sibei Yang, Zhan Qin, Minlie Huang, Wenjie Wang

**Abstract**: Large Language Models (LLMs) are widely applied in decision making, but their deployment is threatened by jailbreak attacks, where adversarial users manipulate model behavior to bypass safety measures. Existing defense mechanisms, such as safety fine-tuning and model editing, either require extensive parameter modifications or lack precision, leading to performance degradation on general tasks, which is unsuitable to post-deployment safety alignment. To address these challenges, we propose DELMAN (Dynamic Editing for LLMs JAilbreak DefeNse), a novel approach leveraging direct model editing for precise, dynamic protection against jailbreak attacks. DELMAN directly updates a minimal set of relevant parameters to neutralize harmful behaviors while preserving the model's utility. To avoid triggering a safe response in benign context, we incorporate KL-divergence regularization to ensure the updated model remains consistent with the original model when processing benign queries. Experimental results demonstrate that DELMAN outperforms baseline methods in mitigating jailbreak attacks while preserving the model's utility, and adapts seamlessly to new attack instances, providing a practical and efficient solution for post-deployment model protection.

摘要: 大语言模型(LLM)在决策中被广泛应用，但它们的部署受到越狱攻击的威胁，在越狱攻击中，敌对用户操纵模型行为以绕过安全措施。现有的防御机制，如安全微调和模型编辑，要么需要大量修改参数，要么缺乏精度，导致一般任务的性能下降，不适合部署后的安全对齐。为了应对这些挑战，我们提出了Delman(用于LLMS越狱防御的动态编辑)，这是一种利用直接模型编辑来精确、动态地防御越狱攻击的新方法。Delman直接更新相关参数的最小集合，以中和有害行为，同时保持模型的实用性。为了避免在良性环境下触发安全响应，我们引入了KL-散度正则化，以确保在处理良性查询时更新后的模型与原始模型保持一致。实验结果表明，Delman在保持模型实用性的同时，在缓解越狱攻击方面优于基准方法，并能无缝适应新的攻击实例，为部署后模型防护提供了一种实用而高效的解决方案。



## **7. Can LLM Watermarks Robustly Prevent Unauthorized Knowledge Distillation?**

LLM水印能否强大地防止未经授权的知识提炼？ cs.CL

22 pages, 12 figures, 13 tables

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11598v1) [paper-pdf](http://arxiv.org/pdf/2502.11598v1)

**Authors**: Leyi Pan, Aiwei Liu, Shiyu Huang, Yijian Lu, Xuming Hu, Lijie Wen, Irwin King, Philip S. Yu

**Abstract**: The radioactive nature of Large Language Model (LLM) watermarking enables the detection of watermarks inherited by student models when trained on the outputs of watermarked teacher models, making it a promising tool for preventing unauthorized knowledge distillation. However, the robustness of watermark radioactivity against adversarial actors remains largely unexplored. In this paper, we investigate whether student models can acquire the capabilities of teacher models through knowledge distillation while avoiding watermark inheritance. We propose two categories of watermark removal approaches: pre-distillation removal through untargeted and targeted training data paraphrasing (UP and TP), and post-distillation removal through inference-time watermark neutralization (WN). Extensive experiments across multiple model pairs, watermarking schemes and hyper-parameter settings demonstrate that both TP and WN thoroughly eliminate inherited watermarks, with WN achieving this while maintaining knowledge transfer efficiency and low computational overhead. Given the ongoing deployment of watermarking techniques in production LLMs, these findings emphasize the urgent need for more robust defense strategies. Our code is available at https://github.com/THU-BPM/Watermark-Radioactivity-Attack.

摘要: 大型语言模型(LLM)水印的放射性特性使其能够在对带水印的教师模型的输出进行训练时检测由学生模型继承的水印，使其成为防止未经授权的知识蒸馏的一种有前途的工具。然而，水印放射性对敌方行为的稳健性在很大程度上仍未被探索。本文研究了学生模型能否在避免水印继承的同时，通过知识提炼获得教师模型的能力。我们提出了两类水印去除方法：通过非目标和目标训练数据释义(UP和TP)进行蒸馏前去除和通过推理时间水印中和(WN)进行蒸馏后去除。在多个模型对、水印方案和超参数设置上的大量实验表明，TP和WN都彻底消除了继承的水印，WN在保持知识传递效率和较低的计算开销的同时实现了这一点。鉴于水印技术在生产LLM中的持续部署，这些发现强调了对更强大的防御策略的迫切需要。我们的代码可以在https://github.com/THU-BPM/Watermark-Radioactivity-Attack.上找到



## **8. LLMs can be Dangerous Reasoners: Analyzing-based Jailbreak Attack on Large Language Models**

LLM可能是危险的推理者：基于分析的对大型语言模型的越狱攻击 cs.CR

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2407.16205v4) [paper-pdf](http://arxiv.org/pdf/2407.16205v4)

**Authors**: Shi Lin, Hongming Yang, Rongchang Li, Xun Wang, Changting Lin, Wenpeng Xing, Meng Han

**Abstract**: The rapid development of Large Language Models (LLMs) has brought significant advancements across various tasks. However, despite these achievements, LLMs still exhibit inherent safety vulnerabilities, especially when confronted with jailbreak attacks. Existing jailbreak methods suffer from two main limitations: reliance on complicated prompt engineering and iterative optimization, which lead to low attack success rate (ASR) and attack efficiency (AE). In this work, we propose an efficient jailbreak attack method, Analyzing-based Jailbreak (ABJ), which leverages the advanced reasoning capability of LLMs to autonomously generate harmful content, revealing their underlying safety vulnerabilities during complex reasoning process. We conduct comprehensive experiments on ABJ across various open-source and closed-source LLMs. In particular, ABJ achieves high ASR (82.1% on GPT-4o-2024-11-20) with exceptional AE among all target LLMs, showcasing its remarkable attack effectiveness, transferability, and efficiency. Our findings underscore the urgent need to prioritize and improve the safety of LLMs to mitigate the risks of misuse.

摘要: 大型语言模型(LLM)的快速发展带来了跨各种任务的重大进步。然而，尽管取得了这些成就，LLMS仍然表现出固有的安全漏洞，特别是在面临越狱攻击时。现有的越狱方法存在两个主要缺陷：依赖复杂的快速工程和迭代优化，导致攻击成功率和攻击效率较低。在这项工作中，我们提出了一种高效的越狱攻击方法-基于分析的越狱(ABJ)，它利用LLMS的高级推理能力自主生成有害内容，在复杂的推理过程中揭示其潜在的安全漏洞。我们在各种开源和闭源的LLM上对ABJ进行了全面的实验。特别是，ABJ在所有目标LLM中获得了高ASR(在GPT-40-2024-11-20上为82.1%)，并具有出色的AE，显示了其卓越的攻击效能、可转移性和效率。我们的研究结果强调迫切需要优先考虑和改善低密度脂蛋白的安全性，以减少误用的风险。



## **9. Be Cautious When Merging Unfamiliar LLMs: A Phishing Model Capable of Stealing Privacy**

合并不熟悉的LLM时要谨慎：一种能够窃取隐私的网络钓鱼模式 cs.CL

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11533v1) [paper-pdf](http://arxiv.org/pdf/2502.11533v1)

**Authors**: Zhenyuan Guo, Yi Shi, Wenlong Meng, Chen Gong, Chengkun Wei, Wenzhi Chen

**Abstract**: Model merging is a widespread technology in large language models (LLMs) that integrates multiple task-specific LLMs into a unified one, enabling the merged model to inherit the specialized capabilities of these LLMs. Most task-specific LLMs are sourced from open-source communities and have not undergone rigorous auditing, potentially imposing risks in model merging. This paper highlights an overlooked privacy risk: \textit{an unsafe model could compromise the privacy of other LLMs involved in the model merging.} Specifically, we propose PhiMM, a privacy attack approach that trains a phishing model capable of stealing privacy using a crafted privacy phishing instruction dataset. Furthermore, we introduce a novel model cloaking method that mimics a specialized capability to conceal attack intent, luring users into merging the phishing model. Once victims merge the phishing model, the attacker can extract personally identifiable information (PII) or infer membership information (MI) by querying the merged model with the phishing instruction. Experimental results show that merging a phishing model increases the risk of privacy breaches. Compared to the results before merging, PII leakage increased by 3.9\% and MI leakage increased by 17.4\% on average. We release the code of PhiMM through a link.

摘要: 模型合并是大型语言模型中的一种广泛使用的技术，它将多个特定于任务的大型语言模型集成到一个统一的大型语言模型中，使合并后的模型能够继承这些大型语言模型的专门功能。大多数特定于任务的LLM来自开源社区，没有经过严格的审计，这可能会给模型合并带来风险。本文强调了一个被忽视的隐私风险：\textit{一个不安全的模型可能会危及模型合并中涉及的其他LLM的隐私。}具体地说，我们提出了PhiMM，这是一种隐私攻击方法，它训练一个能够使用特制的隐私钓鱼指令数据集窃取隐私的钓鱼模型。此外，我们还提出了一种新的模型伪装方法，该方法模仿了一种特殊的隐藏攻击意图的能力，引诱用户合并钓鱼模型。一旦受害者合并钓鱼模型，攻击者就可以通过使用钓鱼指令查询合并的模型来提取个人身份信息(PII)或推断成员信息(MI)。实验结果表明，合并钓鱼模型会增加隐私泄露的风险。与合并前相比，PII渗漏平均增加3.9%，MI渗漏平均增加17.4%。我们通过一个链接发布PhiMM的代码。



## **10. DeFiScope: Detecting Various DeFi Price Manipulations with LLM Reasoning**

DeFiScope：通过LLM推理检测各种DeFi价格操纵 cs.CR

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11521v1) [paper-pdf](http://arxiv.org/pdf/2502.11521v1)

**Authors**: Juantao Zhong, Daoyuan Wu, Ye Liu, Maoyi Xie, Yang Liu, Yi Li, Ning Liu

**Abstract**: DeFi (Decentralized Finance) is one of the most important applications of today's cryptocurrencies and smart contracts. It manages hundreds of billions in Total Value Locked (TVL) on-chain, yet it remains susceptible to common DeFi price manipulation attacks. Despite state-of-the-art (SOTA) systems like DeFiRanger and DeFort, we found that they are less effective to non-standard price models in custom DeFi protocols, which account for 44.2% of the 95 DeFi price manipulation attacks reported over the past three years.   In this paper, we introduce the first LLM-based approach, DeFiScope, for detecting DeFi price manipulation attacks in both standard and custom price models. Our insight is that large language models (LLMs) have certain intelligence to abstract price calculation from code and infer the trend of token price changes based on the extracted price models. To further strengthen LLMs in this aspect, we leverage Foundry to synthesize on-chain data and use it to fine-tune a DeFi price-specific LLM. Together with the high-level DeFi operations recovered from low-level transaction data, DeFiScope detects various DeFi price manipulations according to systematically mined patterns. Experimental results show that DeFiScope achieves a high precision of 96% and a recall rate of 80%, significantly outperforming SOTA approaches. Moreover, we evaluate DeFiScope's cost-effectiveness and demonstrate its practicality by helping our industry partner confirm 147 real-world price manipulation attacks, including discovering 81 previously unknown historical incidents.

摘要: DEFI(去中心化金融)是当今加密货币和智能合约最重要的应用之一。它管理着数千亿美元的总价值锁定(TVL)链上，但它仍然容易受到常见的Defi价格操纵攻击。尽管像DeFiRanger和DeFort这样的最先进的(Sota)系统，我们发现它们对定制Defi协议中的非标准价格模型的有效性较低，在过去三年报告的95起Defi价格操纵攻击中，非标准价格模型占44.2%。在本文中，我们介绍了第一个基于LLM的方法，DeFiScope，用于检测标准价格模型和定制价格模型中的Defi价格操纵攻击。我们的见解是，大型语言模型(LLM)具有一定的智能，能够从代码中抽象出价格计算，并根据提取的价格模型推断令牌价格变化的趋势。为了在这方面进一步加强LLM，我们利用Foundry来合成链上数据，并使用它来微调特定于Defi价格的LLM。与从低级别交易数据中恢复的高级别Defi操作一起，DeFiScope根据系统挖掘的模式检测各种Defi价格操纵。实验结果表明，DeFiScope达到了96%的高准确率和80%的召回率，明显优于SOTA方法。此外，我们评估了DeFiScope的成本效益，并通过帮助我们的行业合作伙伴确认147起真实世界的价格操纵攻击来展示其实用性，其中包括发现81起以前未知的历史事件。



## **11. Adversary-Aware DPO: Enhancing Safety Alignment in Vision Language Models via Adversarial Training**

具有对抗意识的DPO：通过对抗训练增强视觉语言模型中的安全一致性 cs.CR

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11455v1) [paper-pdf](http://arxiv.org/pdf/2502.11455v1)

**Authors**: Fenghua Weng, Jian Lou, Jun Feng, Minlie Huang, Wenjie Wang

**Abstract**: Safety alignment is critical in pre-training large language models (LLMs) to generate responses aligned with human values and refuse harmful queries. Unlike LLM, the current safety alignment of VLMs is often achieved with post-hoc safety fine-tuning. However, these methods are less effective to white-box attacks. To address this, we propose $\textit{Adversary-aware DPO (ADPO)}$, a novel training framework that explicitly considers adversarial. $\textit{Adversary-aware DPO (ADPO)}$ integrates adversarial training into DPO to enhance the safety alignment of VLMs under worst-case adversarial perturbations. $\textit{ADPO}$ introduces two key components: (1) an adversarial-trained reference model that generates human-preferred responses under worst-case perturbations, and (2) an adversarial-aware DPO loss that generates winner-loser pairs accounting for adversarial distortions. By combining these innovations, $\textit{ADPO}$ ensures that VLMs remain robust and reliable even in the presence of sophisticated jailbreak attacks. Extensive experiments demonstrate that $\textit{ADPO}$ outperforms baselines in the safety alignment and general utility of VLMs.

摘要: 在预先训练大型语言模型(LLM)以生成与人类价值观一致的响应并拒绝有害查询时，安全对齐至关重要。与LLM不同，VLM当前的安全对准通常是通过事后安全微调来实现的。然而，这些方法对白盒攻击的有效性较低。为了解决这一问题，我们提出了一种新的训练框架将对抗性训练融入到DPO中，以增强VLM在最坏情况下的对抗性扰动下的安全一致性。$\textit{ADPO}$引入了两个关键组件：(1)对抗性训练的参考模型，它在最坏情况下产生人类偏好的响应；(2)对抗性感知的DPO损失，它产生考虑对抗性扭曲的赢家-输家对。通过将这些创新结合在一起，$\textit{ADPO}$确保即使在存在复杂的越狱攻击的情况下，VLM仍保持健壮和可靠。大量实验表明，在VLMS的安全性、对准和通用性方面，$\textit{ADPO}$都优于Baseline。



## **12. CCJA: Context-Coherent Jailbreak Attack for Aligned Large Language Models**

CCJA：针对对齐大型语言模型的上下文一致越狱攻击 cs.CR

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11379v1) [paper-pdf](http://arxiv.org/pdf/2502.11379v1)

**Authors**: Guanghao Zhou, Panjia Qiu, Mingyuan Fan, Cen Chen, Mingyuan Chu, Xin Zhang, Jun Zhou

**Abstract**: Despite explicit alignment efforts for large language models (LLMs), they can still be exploited to trigger unintended behaviors, a phenomenon known as "jailbreaking." Current jailbreak attack methods mainly focus on discrete prompt manipulations targeting closed-source LLMs, relying on manually crafted prompt templates and persuasion rules. However, as the capabilities of open-source LLMs improve, ensuring their safety becomes increasingly crucial. In such an environment, the accessibility of model parameters and gradient information by potential attackers exacerbates the severity of jailbreak threats. To address this research gap, we propose a novel \underline{C}ontext-\underline{C}oherent \underline{J}ailbreak \underline{A}ttack (CCJA). We define jailbreak attacks as an optimization problem within the embedding space of masked language models. Through combinatorial optimization, we effectively balance the jailbreak attack success rate with semantic coherence. Extensive evaluations show that our method not only maintains semantic consistency but also surpasses state-of-the-art baselines in attack effectiveness. Additionally, by integrating semantically coherent jailbreak prompts generated by our method into widely used black-box methodologies, we observe a notable enhancement in their success rates when targeting closed-source commercial LLMs. This highlights the security threat posed by open-source LLMs to commercial counterparts. We will open-source our code if the paper is accepted.

摘要: 尽管对大型语言模型(LLM)进行了明确的调整，但它们仍可能被利用来触发意外行为，这一现象被称为“越狱”。目前的越狱攻击方法主要集中在针对闭源LLM的离散提示操作上，依赖于手动创建的提示模板和说服规则。然而，随着开源LLM能力的提高，确保它们的安全变得越来越重要。在这样的环境中，潜在攻击者对模型参数和梯度信息的可访问性加剧了越狱威胁的严重性。为了弥补这一研究空白，我们提出了一种新的\下划线{C}上边-\下划线{C}上边\下划线{J}纵断\下划线{A}ttack(CCJA)。我们将越狱攻击定义为掩蔽语言模型嵌入空间内的优化问题。通过组合优化，有效地平衡了越狱攻击成功率和语义一致性。广泛的评估表明，我们的方法不仅保持了语义的一致性，而且在攻击效率上超过了最先进的基线。此外，通过将我们的方法生成的语义连贯的越狱提示整合到广泛使用的黑盒方法中，我们观察到当目标是闭源商业LLM时，它们的成功率有了显著的提高。这突显了开源低成本管理对商业同行构成的安全威胁。如果论文被接受，我们将开放我们的代码。



## **13. Dagger Behind Smile: Fool LLMs with a Happy Ending Story**

微笑背后的匕首：傻瓜LLMs，有一个幸福的结局 cs.CL

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2501.13115v2) [paper-pdf](http://arxiv.org/pdf/2501.13115v2)

**Authors**: Xurui Song, Zhixin Xie, Shuo Huai, Jiayi Kong, Jun Luo

**Abstract**: The wide adoption of Large Language Models (LLMs) has attracted significant attention from $\textit{jailbreak}$ attacks, where adversarial prompts crafted through optimization or manual design exploit LLMs to generate malicious contents. However, optimization-based attacks have limited efficiency and transferability, while existing manual designs are either easily detectable or demand intricate interactions with LLMs. In this paper, we first point out a novel perspective for jailbreak attacks: LLMs are more responsive to $\textit{positive}$ prompts. Based on this, we deploy Happy Ending Attack (HEA) to wrap up a malicious request in a scenario template involving a positive prompt formed mainly via a $\textit{happy ending}$, it thus fools LLMs into jailbreaking either immediately or at a follow-up malicious request.This has made HEA both efficient and effective, as it requires only up to two turns to fully jailbreak LLMs. Extensive experiments show that our HEA can successfully jailbreak on state-of-the-art LLMs, including GPT-4o, Llama3-70b, Gemini-pro, and achieves 88.79\% attack success rate on average. We also provide quantitative explanations for the success of HEA.

摘要: 大型语言模型(LLM)的广泛采用引起了$\textit{jailBreak}$攻击的极大关注，即通过优化或手动设计创建的敌意提示利用LLM生成恶意内容。然而，基于优化的攻击的效率和可转移性有限，而现有的手动设计要么很容易被检测到，要么需要与LLM进行复杂的交互。在这篇文章中，我们首先指出了越狱攻击的一个新视角：LLM对$\textit{积极}$提示的响应更快。在此基础上，利用HEA(Happy End End Attack)将恶意请求封装在一个场景模板中，该场景模板包含一个主要通过$\textit{Happy End}$形成的积极提示，从而欺骗LLM立即越狱或在后续恶意请求时越狱，这使得HEA既高效又有效，因为它只需要最多两个回合就可以完全越狱LLM。大量的实验表明，我们的HEA能够成功地在GPT-40、Llama3-70b、Gemini-Pro等最先进的LLMS上越狱，平均攻击成功率达到88.79%。我们还对HEA的成功提供了定量的解释。



## **14. Mimicking the Familiar: Dynamic Command Generation for Information Theft Attacks in LLM Tool-Learning System**

模仿熟悉的：LLM工具学习系统中信息窃取攻击的动态命令生成 cs.AI

15 pages, 11 figures

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11358v1) [paper-pdf](http://arxiv.org/pdf/2502.11358v1)

**Authors**: Ziyou Jiang, Mingyang Li, Guowei Yang, Junjie Wang, Yuekai Huang, Zhiyuan Chang, Qing Wang

**Abstract**: Information theft attacks pose a significant risk to Large Language Model (LLM) tool-learning systems. Adversaries can inject malicious commands through compromised tools, manipulating LLMs to send sensitive information to these tools, which leads to potential privacy breaches. However, existing attack approaches are black-box oriented and rely on static commands that cannot adapt flexibly to the changes in user queries and the invocation chain of tools. It makes malicious commands more likely to be detected by LLM and leads to attack failure. In this paper, we propose AutoCMD, a dynamic attack comment generation approach for information theft attacks in LLM tool-learning systems. Inspired by the concept of mimicking the familiar, AutoCMD is capable of inferring the information utilized by upstream tools in the toolchain through learning on open-source systems and reinforcement with target system examples, thereby generating more targeted commands for information theft. The evaluation results show that AutoCMD outperforms the baselines with +13.2% $ASR_{Theft}$, and can be generalized to new tool-learning systems to expose their information leakage risks. We also design four defense methods to effectively protect tool-learning systems from the attack.

摘要: 信息窃取攻击对大型语言模型(LLM)工具学习系统构成了重大风险。攻击者可以通过受危害的工具注入恶意命令，操纵LLM向这些工具发送敏感信息，从而导致潜在的隐私泄露。然而，现有的攻击方法是面向黑盒的，依赖于静态命令，不能灵活地适应用户查询和工具调用链的变化。它使恶意命令更容易被LLM检测到，并导致攻击失败。本文针对LLM工具学习系统中的信息窃取攻击，提出了一种动态攻击评论生成方法AutoCMD。受模仿熟悉的概念的启发，AutoCMD能够通过学习开源系统和加强目标系统示例来推断工具链中的上游工具所使用的信息，从而生成更有针对性的信息窃取命令。评估结果表明，AutoCMD的性能比基准高出13.2%$ASR{Theft}$，可以推广到新的工具学习系统，以暴露其信息泄露风险。我们还设计了四种防御方法来有效地保护工具学习系统免受攻击。



## **15. ALGEN: Few-shot Inversion Attacks on Textual Embeddings using Alignment and Generation**

ALgen：使用对齐和生成对文本嵌入进行少量反转攻击 cs.CR

18 pages, 13 tables, 6 figures

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.11308v2) [paper-pdf](http://arxiv.org/pdf/2502.11308v2)

**Authors**: Yiyi Chen, Qiongkai Xu, Johannes Bjerva

**Abstract**: With the growing popularity of Large Language Models (LLMs) and vector databases, private textual data is increasingly processed and stored as numerical embeddings. However, recent studies have proven that such embeddings are vulnerable to inversion attacks, where original text is reconstructed to reveal sensitive information. Previous research has largely assumed access to millions of sentences to train attack models, e.g., through data leakage or nearly unrestricted API access. With our method, a single data point is sufficient for a partially successful inversion attack. With as little as 1k data samples, performance reaches an optimum across a range of black-box encoders, without training on leaked data. We present a Few-shot Textual Embedding Inversion Attack using ALignment and GENeration (ALGEN), by aligning victim embeddings to the attack space and using a generative model to reconstruct text. We find that ALGEN attacks can be effectively transferred across domains and languages, revealing key information. We further examine a variety of defense mechanisms against ALGEN, and find that none are effective, highlighting the vulnerabilities posed by inversion attacks. By significantly lowering the cost of inversion and proving that embedding spaces can be aligned through one-step optimization, we establish a new textual embedding inversion paradigm with broader applications for embedding alignment in NLP.

摘要: 随着大型语言模型和向量数据库的日益流行，私有文本数据越来越多地以数字嵌入的形式进行处理和存储。然而，最近的研究证明，这种嵌入很容易受到反转攻击，即重建原始文本以泄露敏感信息。以前的研究在很大程度上假设可以访问数百万个句子来训练攻击模型，例如通过数据泄漏或几乎不受限制的API访问。使用我们的方法，对于部分成功的反转攻击，单个数据点就足够了。只需1000个数据样本，一系列黑盒编码器的性能就可以达到最佳，而不需要对泄露的数据进行培训。提出了一种基于对齐和生成的少量文本嵌入反转攻击(ALGEN)，通过将受害者嵌入对齐到攻击空间，并使用生成模型来重构文本。我们发现，ALGEN攻击可以有效地跨域和跨语言传输，泄露关键信息。我们进一步研究了针对ALGEN的各种防御机制，发现没有一种机制是有效的，这突出了反转攻击带来的漏洞。通过显著降低反转的代价，并证明嵌入空间可以通过一步优化来对齐，我们建立了一种新的文本嵌入反转范式，在自然语言处理中具有更广泛的应用前景。



## **16. G-Safeguard: A Topology-Guided Security Lens and Treatment on LLM-based Multi-agent Systems**

G-Safeguard：基于LLM的多智能体系统上的一种基于布局引导的安全视角和处理方法 cs.CR

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2502.11127v1) [paper-pdf](http://arxiv.org/pdf/2502.11127v1)

**Authors**: Shilong Wang, Guibin Zhang, Miao Yu, Guancheng Wan, Fanci Meng, Chongye Guo, Kun Wang, Yang Wang

**Abstract**: Large Language Model (LLM)-based Multi-agent Systems (MAS) have demonstrated remarkable capabilities in various complex tasks, ranging from collaborative problem-solving to autonomous decision-making. However, as these systems become increasingly integrated into critical applications, their vulnerability to adversarial attacks, misinformation propagation, and unintended behaviors have raised significant concerns. To address this challenge, we introduce G-Safeguard, a topology-guided security lens and treatment for robust LLM-MAS, which leverages graph neural networks to detect anomalies on the multi-agent utterance graph and employ topological intervention for attack remediation. Extensive experiments demonstrate that G-Safeguard: (I) exhibits significant effectiveness under various attack strategies, recovering over 40% of the performance for prompt injection; (II) is highly adaptable to diverse LLM backbones and large-scale MAS; (III) can seamlessly combine with mainstream MAS with security guarantees. The code is available at https://github.com/wslong20/G-safeguard.

摘要: 基于大型语言模型(LLM)的多智能体系统(MAS)在从协作问题求解到自主决策的各种复杂任务中表现出了卓越的能力。然而，随着这些系统越来越多地集成到关键应用程序中，它们对对手攻击、错误信息传播和意外行为的脆弱性已经引起了极大的关注。为了应对这一挑战，我们引入了G-Safe，这是一种拓扑制导的安全镜头和健壮LLM-MAS的处理方法，它利用图神经网络来检测多智能体话语图上的异常，并使用拓扑干预进行攻击补救。大量实验表明：(I)在各种攻击策略下表现出显著的有效性，可恢复40%以上的性能进行快速注入；(Ii)对不同的LLM主干和大规模MAS具有高度的适应性；(Iii)可以与主流MAS无缝结合，具有安全保障。代码可在https://github.com/wslong20/G-safeguard.上获得



## **17. SafeDialBench: A Fine-Grained Safety Benchmark for Large Language Models in Multi-Turn Dialogues with Diverse Jailbreak Attacks**

SafeDialBench：针对具有多样越狱攻击的多回合对话中大型语言模型的细粒度安全基准 cs.CL

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.11090v2) [paper-pdf](http://arxiv.org/pdf/2502.11090v2)

**Authors**: Hongye Cao, Yanming Wang, Sijia Jing, Ziyue Peng, Zhixin Bai, Zhe Cao, Meng Fang, Fan Feng, Boyan Wang, Jiaheng Liu, Tianpei Yang, Jing Huo, Yang Gao, Fanyu Meng, Xi Yang, Chao Deng, Junlan Feng

**Abstract**: With the rapid advancement of Large Language Models (LLMs), the safety of LLMs has been a critical concern requiring precise assessment. Current benchmarks primarily concentrate on single-turn dialogues or a single jailbreak attack method to assess the safety. Additionally, these benchmarks have not taken into account the LLM's capability of identifying and handling unsafe information in detail. To address these issues, we propose a fine-grained benchmark SafeDialBench for evaluating the safety of LLMs across various jailbreak attacks in multi-turn dialogues. Specifically, we design a two-tier hierarchical safety taxonomy that considers 6 safety dimensions and generates more than 4000 multi-turn dialogues in both Chinese and English under 22 dialogue scenarios. We employ 7 jailbreak attack strategies, such as reference attack and purpose reverse, to enhance the dataset quality for dialogue generation. Notably, we construct an innovative assessment framework of LLMs, measuring capabilities in detecting, and handling unsafe information and maintaining consistency when facing jailbreak attacks. Experimental results across 17 LLMs reveal that Yi-34B-Chat and GLM4-9B-Chat demonstrate superior safety performance, while Llama3.1-8B-Instruct and o3-mini exhibit safety vulnerabilities.

摘要: 随着大型语言模型的快速发展，大型语言模型的安全性已经成为一个需要精确评估的关键问题。目前的基准主要集中在单轮对话或单一越狱攻击方法上，以评估安全性。此外，这些基准没有考虑到LLM详细识别和处理不安全信息的能力。为了解决这些问题，我们提出了一个细粒度的基准SafeDialB边，用于在多轮对话中评估LLMS在各种越狱攻击中的安全性。具体地说，我们设计了一个考虑了6个安全维度的两级层次安全分类，并在22个对话场景下生成了4000多个中英文多轮对话。我们使用了引用攻击、目的反转等7种越狱攻击策略，以提高对话生成的数据集质量。值得注意的是，我们构建了一个创新的LLMS评估框架，衡量了检测和处理不安全信息的能力，并在面临越狱攻击时保持一致性。在17个LLM上的实验结果表明，YI-34B-Chat和GLM4-9B-Chat具有优越的安全性能，而Llama3.1-8B-Indict和O_3-mini存在安全漏洞。



## **18. Rewrite to Jailbreak: Discover Learnable and Transferable Implicit Harmfulness Instruction**

重写越狱：发现可学习和可转移的隐性有害指令 cs.CL

21pages, 10 figures

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2502.11084v1) [paper-pdf](http://arxiv.org/pdf/2502.11084v1)

**Authors**: Yuting Huang, Chengyuan Liu, Yifeng Feng, Chao Wu, Fei Wu, Kun Kuang

**Abstract**: As Large Language Models (LLMs) are widely applied in various domains, the safety of LLMs is increasingly attracting attention to avoid their powerful capabilities being misused. Existing jailbreak methods create a forced instruction-following scenario, or search adversarial prompts with prefix or suffix tokens to achieve a specific representation manually or automatically. However, they suffer from low efficiency and explicit jailbreak patterns, far from the real deployment of mass attacks to LLMs. In this paper, we point out that simply rewriting the original instruction can achieve a jailbreak, and we find that this rewriting approach is learnable and transferable. We propose the Rewrite to Jailbreak (R2J) approach, a transferable black-box jailbreak method to attack LLMs by iteratively exploring the weakness of the LLMs and automatically improving the attacking strategy. The jailbreak is more efficient and hard to identify since no additional features are introduced. Extensive experiments and analysis demonstrate the effectiveness of R2J, and we find that the jailbreak is also transferable to multiple datasets and various types of models with only a few queries. We hope our work motivates further investigation of LLM safety.

摘要: 随着大语言模型在各个领域的广泛应用，大语言模型的安全性越来越受到人们的关注，以避免其强大的功能被滥用。现有的越狱方法创建了强制遵循指令的场景，或者搜索带有前缀或后缀令牌的对抗性提示，以手动或自动地实现特定的表示。然而，他们遭受的是低效率和明确的越狱模式，远远不能真正部署大规模攻击到LLM。在本文中，我们指出，简单地重写原始指令就可以实现越狱，并且我们发现这种重写方法是可学习的和可移植的。提出了重写越狱(R2J)方法，通过迭代挖掘LLMS的弱点并自动改进攻击策略，提出了一种可转移的黑盒越狱方法来攻击LLMS。由于没有引入其他功能，越狱更加高效，也更难识别。大量的实验和分析证明了R2J的有效性，我们发现越狱也可以只需几个查询就可以移植到多个数据集和各种类型的模型上。我们希望我们的工作能促进对LLM安全性的进一步研究。



## **19. Na'vi or Knave: Jailbreaking Language Models via Metaphorical Avatars**

纳威人或恶棍：通过隐喻化身越狱语言模型 cs.CL

We still need to polish our paper

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2412.12145v2) [paper-pdf](http://arxiv.org/pdf/2412.12145v2)

**Authors**: Yu Yan, Sheng Sun, Junqi Tong, Min Liu, Qi Li

**Abstract**: Metaphor serves as an implicit approach to convey information, while enabling the generalized comprehension of complex subjects. However, metaphor can potentially be exploited to bypass the safety alignment mechanisms of Large Language Models (LLMs), leading to the theft of harmful knowledge. In our study, we introduce a novel attack framework that exploits the imaginative capacity of LLMs to achieve jailbreaking, the J\underline{\textbf{A}}ilbreak \underline{\textbf{V}}ia \underline{\textbf{A}}dversarial Me\underline{\textbf{TA}} -pho\underline{\textbf{R}} (\textit{AVATAR}). Specifically, to elicit the harmful response, AVATAR extracts harmful entities from a given harmful target and maps them to innocuous adversarial entities based on LLM's imagination. Then, according to these metaphors, the harmful target is nested within human-like interaction for jailbreaking adaptively. Experimental results demonstrate that AVATAR can effectively and transferablly jailbreak LLMs and achieve a state-of-the-art attack success rate across multiple advanced LLMs. Our study exposes a security risk in LLMs from their endogenous imaginative capabilities. Furthermore, the analytical study reveals the vulnerability of LLM to adversarial metaphors and the necessity of developing defense methods against jailbreaking caused by the adversarial metaphor. \textcolor{orange}{ \textbf{Warning: This paper contains potentially harmful content from LLMs.}}

摘要: 隐喻是一种隐含的传达信息的方法，同时也使人们能够对复杂的主题进行广义的理解。然而，隐喻可能被利用来绕过大型语言模型(LLM)的安全对齐机制，从而导致有害知识的窃取。在我们的研究中，我们提出了一种新的攻击框架，该框架利用LLMS的想象能力来实现越狱，即J下划线{\extbf{A}}ilBreak\下划线{\extbf{A}}ia\下划线{\extbf{A}}dversarial Me\下划线{\extbf{TA}}-pho\下划线{\extbf{R}}(\textit{阿凡达})。具体地说，为了引发有害反应，阿凡达从给定的有害目标中提取有害实体，并基于LLM的想象力将它们映射到无害的对抗性实体。然后，根据这些隐喻，有害的目标嵌套在类似人类的互动中，以适应越狱。实验结果表明，阿凡达可以有效地、可转移地越狱LLM，并在多个高级LLM上获得最先进的攻击成功率。我们的研究揭示了LLMS的安全风险来自其内生的想象力。此外，分析研究揭示了LLM对对抗性隐喻的脆弱性，以及开发针对对抗性隐喻导致的越狱的防御方法的必要性。\extCOLOR{橙色}{\extbf{警告：此纸张包含来自LLMS的潜在有害内容。}}



## **20. Reasoning-Augmented Conversation for Multi-Turn Jailbreak Attacks on Large Language Models**

针对大型语言模型的多回合越狱攻击的推理增强对话 cs.CL

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.11054v2) [paper-pdf](http://arxiv.org/pdf/2502.11054v2)

**Authors**: Zonghao Ying, Deyue Zhang, Zonglei Jing, Yisong Xiao, Quanchen Zou, Aishan Liu, Siyuan Liang, Xiangzheng Zhang, Xianglong Liu, Dacheng Tao

**Abstract**: Multi-turn jailbreak attacks simulate real-world human interactions by engaging large language models (LLMs) in iterative dialogues, exposing critical safety vulnerabilities. However, existing methods often struggle to balance semantic coherence with attack effectiveness, resulting in either benign semantic drift or ineffective detection evasion. To address this challenge, we propose Reasoning-Augmented Conversation, a novel multi-turn jailbreak framework that reformulates harmful queries into benign reasoning tasks and leverages LLMs' strong reasoning capabilities to compromise safety alignment. Specifically, we introduce an attack state machine framework to systematically model problem translation and iterative reasoning, ensuring coherent query generation across multiple turns. Building on this framework, we design gain-guided exploration, self-play, and rejection feedback modules to preserve attack semantics, enhance effectiveness, and sustain reasoning-driven attack progression. Extensive experiments on multiple LLMs demonstrate that RACE achieves state-of-the-art attack effectiveness in complex conversational scenarios, with attack success rates (ASRs) increasing by up to 96%. Notably, our approach achieves ASRs of 82% and 92% against leading commercial models, OpenAI o1 and DeepSeek R1, underscoring its potency. We release our code at https://github.com/NY1024/RACE to facilitate further research in this critical domain.

摘要: 多轮越狱攻击通过在迭代对话中使用大型语言模型(LLM)来模拟真实世界的人类交互，暴露出关键的安全漏洞。然而，现有的方法往往难以在语义一致性和攻击有效性之间取得平衡，导致良性的语义漂移或无效的检测规避。为了应对这一挑战，我们提出了一种新的多轮越狱框架--推理增强对话，该框架将有害的查询重新定义为良性的推理任务，并利用LLMS强大的推理能力来妥协安全对齐。具体地说，我们引入了攻击状态机框架来系统地建模问题转换和迭代推理，确保跨多轮的连贯查询生成。在这个框架的基础上，我们设计了增益引导的探索、自我发挥和拒绝反馈模块，以保留攻击语义，提高有效性，并支持推理驱动的攻击进展。在多个LLM上的大量实验表明，RACE在复杂的会话场景中获得了最先进的攻击效率，攻击成功率(ASR)提高了96%。值得注意的是，我们的方法在领先的商业模型OpenAI o1和DeepSeek r1上分别获得了82%和92%的ASR，这突显了它的有效性。我们在https://github.com/NY1024/RACE上发布我们的代码，以促进在这一关键领域的进一步研究。



## **21. Atoxia: Red-teaming Large Language Models with Target Toxic Answers**

Atoxia：将大型语言模型与目标有毒答案进行红色合作 cs.CL

Accepted to Findings of NAACL-2025

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2408.14853v2) [paper-pdf](http://arxiv.org/pdf/2408.14853v2)

**Authors**: Yuhao Du, Zhuo Li, Pengyu Cheng, Xiang Wan, Anningzhe Gao

**Abstract**: Despite the substantial advancements in artificial intelligence, large language models (LLMs) remain being challenged by generation safety. With adversarial jailbreaking prompts, one can effortlessly induce LLMs to output harmful content, causing unexpected negative social impacts. This vulnerability highlights the necessity for robust LLM red-teaming strategies to identify and mitigate such risks before large-scale application. To detect specific types of risks, we propose a novel red-teaming method that $\textbf{A}$ttacks LLMs with $\textbf{T}$arget $\textbf{Toxi}$c $\textbf{A}$nswers ($\textbf{Atoxia}$). Given a particular harmful answer, Atoxia generates a corresponding user query and a misleading answer opening to examine the internal defects of a given LLM. The proposed attacker is trained within a reinforcement learning scheme with the LLM outputting probability of the target answer as the reward. We verify the effectiveness of our method on various red-teaming benchmarks, such as AdvBench and HH-Harmless. The empirical results demonstrate that Atoxia can successfully detect safety risks in not only open-source models but also state-of-the-art black-box models such as GPT-4o.

摘要: 尽管人工智能取得了实质性的进步，但大型语言模型(LLM)仍然受到发电安全的挑战。在对抗性越狱提示下，人们可以毫不费力地诱导LLMS输出有害内容，造成意想不到的负面社会影响。该漏洞突显了在大规模应用之前，需要强大的LLM红团队战略来识别和缓解此类风险。为了检测特定类型的风险，我们提出了一种新的红团队方法，即用$\extbf{T}$目标$\extbf{Toxi}$c$\extbf{A}$nswers($\extbf{Atoxia}$)来绑定LLMS。给定特定的有害答案，Atoxia会生成相应的用户查询和误导性答案，以检查给定LLM的内部缺陷。所提出的攻击者在强化学习方案中被训练，LLM输出目标答案的概率作为奖励。我们在不同的红队基准测试上验证了我们的方法的有效性，例如AdvBtch和HH-无害。实验结果表明，Atoxia不仅可以在开源模型中成功检测安全风险，而且可以在GPT-40等最先进的黑盒模型中成功检测到安全风险。



## **22. When Backdoors Speak: Understanding LLM Backdoor Attacks Through Model-Generated Explanations**

当后门说话时：通过模型生成的解释了解LLM后门攻击 cs.CR

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2411.12701v3) [paper-pdf](http://arxiv.org/pdf/2411.12701v3)

**Authors**: Huaizhi Ge, Yiming Li, Qifan Wang, Yongfeng Zhang, Ruixiang Tang

**Abstract**: Large Language Models (LLMs) are known to be vulnerable to backdoor attacks, where triggers embedded in poisoned samples can maliciously alter LLMs' behaviors. In this paper, we move beyond attacking LLMs and instead examine backdoor attacks through the novel lens of natural language explanations. Specifically, we leverage LLMs' generative capabilities to produce human-readable explanations for their decisions, enabling direct comparisons between explanations for clean and poisoned samples. Our results show that backdoored models produce coherent explanations for clean inputs but diverse and logically flawed explanations for poisoned data, a pattern consistent across classification and generation tasks for different backdoor attacks. Further analysis reveals key insights into the explanation generation process. At the token level, explanation tokens associated with poisoned samples only appear in the final few transformer layers. At the sentence level, attention dynamics indicate that poisoned inputs shift attention away from the original input context during explanation generation. These findings enhance our understanding of backdoor mechanisms in LLMs and present a promising framework for detecting vulnerabilities through explainability.

摘要: 众所周知，大型语言模型(LLM)容易受到后门攻击，在后门攻击中，嵌入在有毒样本中的触发器可以恶意改变LLM的行为。在这篇文章中，我们超越了攻击LLM，而是通过自然语言解释的新视角来研究后门攻击。具体地说，我们利用LLMS的生成能力来为他们的决定生成人类可读的解释，从而能够在干净和有毒样本的解释之间进行直接比较。我们的结果表明，回溯模型为干净的输入提供了连贯的解释，但对有毒数据提供了多样化和逻辑上有缺陷的解释，对于不同的后门攻击，这种模式在分类和生成任务中是一致的。进一步的分析揭示了对解释生成过程的关键见解。在令牌级别，与中毒样本相关的解释令牌只出现在最后几个变压器层中。在句子层面，注意动力学表明，有毒输入在解释生成过程中将注意力从原始输入上下文转移开。这些发现加深了我们对LLMS中后门机制的理解，并为通过可解释性检测漏洞提供了一个很有前途的框架。



## **23. Functional Homotopy: Smoothing Discrete Optimization via Continuous Parameters for LLM Jailbreak Attacks**

功能同伦：通过LLM越狱攻击的连续参数平滑离散优化 cs.LG

Published at ICLR 2025

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2410.04234v2) [paper-pdf](http://arxiv.org/pdf/2410.04234v2)

**Authors**: Zi Wang, Divyam Anshumaan, Ashish Hooda, Yudong Chen, Somesh Jha

**Abstract**: Optimization methods are widely employed in deep learning to identify and mitigate undesired model responses. While gradient-based techniques have proven effective for image models, their application to language models is hindered by the discrete nature of the input space. This study introduces a novel optimization approach, termed the \emph{functional homotopy} method, which leverages the functional duality between model training and input generation. By constructing a series of easy-to-hard optimization problems, we iteratively solve these problems using principles derived from established homotopy methods. We apply this approach to jailbreak attack synthesis for large language models (LLMs), achieving a $20\%-30\%$ improvement in success rate over existing methods in circumventing established safe open-source models such as Llama-2 and Llama-3.

摘要: 优化方法广泛应用于深度学习中，以识别和减轻不需要的模型响应。虽然基于梯度的技术已被证明对图像模型有效，但它们在语言模型中的应用受到输入空间的离散性的阻碍。这项研究引入了一种新型优化方法，称为\{函数同伦}方法，它利用了模型训练和输入生成之间的函数二元性。通过构建一系列容易到难的优化问题，我们使用从已建立的同伦方法推导出的原则迭代解决这些问题。我们将这种方法应用于大型语言模型（LLM）的越狱攻击合成，在规避已建立的安全开源模型（例如Llama-2和Llama-3）方面，与现有方法相比，成功率提高了20 -30美元。



## **24. Distraction is All You Need for Multimodal Large Language Model Jailbreaking**

分散注意力就是多模式大型语言模型越狱所需的一切 cs.CV

**SubmitDate**: 2025-02-15    [abs](http://arxiv.org/abs/2502.10794v1) [paper-pdf](http://arxiv.org/pdf/2502.10794v1)

**Authors**: Zuopeng Yang, Jiluan Fan, Anli Yan, Erdun Gao, Xin Lin, Tao Li, Kanghua mo, Changyu Dong

**Abstract**: Multimodal Large Language Models (MLLMs) bridge the gap between visual and textual data, enabling a range of advanced applications. However, complex internal interactions among visual elements and their alignment with text can introduce vulnerabilities, which may be exploited to bypass safety mechanisms. To address this, we analyze the relationship between image content and task and find that the complexity of subimages, rather than their content, is key. Building on this insight, we propose the Distraction Hypothesis, followed by a novel framework called Contrasting Subimage Distraction Jailbreaking (CS-DJ), to achieve jailbreaking by disrupting MLLMs alignment through multi-level distraction strategies. CS-DJ consists of two components: structured distraction, achieved through query decomposition that induces a distributional shift by fragmenting harmful prompts into sub-queries, and visual-enhanced distraction, realized by constructing contrasting subimages to disrupt the interactions among visual elements within the model. This dual strategy disperses the model's attention, reducing its ability to detect and mitigate harmful content. Extensive experiments across five representative scenarios and four popular closed-source MLLMs, including GPT-4o-mini, GPT-4o, GPT-4V, and Gemini-1.5-Flash, demonstrate that CS-DJ achieves average success rates of 52.40% for the attack success rate and 74.10% for the ensemble attack success rate. These results reveal the potential of distraction-based approaches to exploit and bypass MLLMs' defenses, offering new insights for attack strategies.

摘要: 多模式大型语言模型(MLLMS)弥合了视觉数据和文本数据之间的差距，支持一系列高级应用程序。但是，视觉元素之间复杂的内部交互及其与文本的对齐可能会引入漏洞，这可能会被利用来绕过安全机制。为了解决这个问题，我们分析了图像内容和任务之间的关系，发现关键是子图像的复杂性，而不是它们的内容。基于这一认识，我们提出了分心假说，并提出了一个新的框架，称为对比亚像分心越狱(CS-DJ)，通过多层次的分心策略扰乱MLLMS对齐来实现越狱。CS-DJ由两个部分组成：结构化分散，通过查询分解，通过将有害提示分割为子查询来引起分布转移；以及视觉增强分散，通过构建对比对比子图像来破坏模型中视觉元素之间的交互。这种双重策略分散了模型的注意力，降低了其检测和缓解有害内容的能力。在五个典型场景和四个流行的闭源MLLM(包括GPT-4o-mini、GPT-4o、GPT-4V和Gemini-1.5-Flash)上的广泛实验表明，CS-DJ的平均攻击成功率为52.40%，整体攻击成功率为74.10%。这些结果揭示了基于分心的方法利用和绕过MLLMS防御的潜力，为攻击战略提供了新的见解。



## **25. HoneyGPT: Breaking the Trilemma in Terminal Honeypots with Large Language Model**

HoneyGPT：用大型语言模型打破终端蜜罐中的三重困境 cs.CR

**SubmitDate**: 2025-02-15    [abs](http://arxiv.org/abs/2406.01882v2) [paper-pdf](http://arxiv.org/pdf/2406.01882v2)

**Authors**: Ziyang Wang, Jianzhou You, Haining Wang, Tianwei Yuan, Shichao Lv, Yang Wang, Limin Sun

**Abstract**: Honeypots, as a strategic cyber-deception mechanism designed to emulate authentic interactions and bait unauthorized entities, often struggle with balancing flexibility, interaction depth, and deception. They typically fail to adapt to evolving attacker tactics, with limited engagement and information gathering. Fortunately, the emergent capabilities of large language models and innovative prompt-based engineering offer a transformative shift in honeypot technologies. This paper introduces HoneyGPT, a pioneering shell honeypot architecture based on ChatGPT, characterized by its cost-effectiveness and proactive engagement. In particular, we propose a structured prompt engineering framework that incorporates chain-of-thought tactics to improve long-term memory and robust security analytics, enhancing deception and engagement. Our evaluation of HoneyGPT comprises a baseline comparison based on a collected dataset and a three-month field evaluation. The baseline comparison demonstrates HoneyGPT's remarkable ability to strike a balance among flexibility, interaction depth, and deceptive capability. The field evaluation further validates HoneyGPT's superior performance in engaging attackers more deeply and capturing a wider array of novel attack vectors.

摘要: 蜜罐作为一种战略性的网络欺骗机制，旨在模拟真实的交互并诱骗未经授权的实体，经常在灵活性、交互深度和欺骗之间进行权衡。他们通常无法适应不断变化的攻击策略，参与和信息收集有限。幸运的是，大型语言模型的新兴能力和创新的基于提示的工程提供了蜜罐技术的变革性转变。介绍了一种基于ChatGPT的开创性的贝壳蜜罐架构--HoneyGPT，它的特点是性价比高和主动参与。特别是，我们提出了一个结构化的即时工程框架，该框架结合了思想链策略来改善长期记忆和强大的安全分析，增强了欺骗性和参与性。我们对HoneyGPT的评估包括基于收集的数据集的基线比较和三个月的现场评估。基准比较表明，HoneyGPT在灵活性、交互深度和欺骗性能力之间取得了显著的平衡。现场评估进一步验证了HoneyGPT在更深入地攻击攻击者和捕获更广泛的新型攻击载体方面的卓越性能。



## **26. Robustness-aware Automatic Prompt Optimization**

具有鲁棒性的自动提示优化 cs.CL

**SubmitDate**: 2025-02-15    [abs](http://arxiv.org/abs/2412.18196v2) [paper-pdf](http://arxiv.org/pdf/2412.18196v2)

**Authors**: Zeru Shi, Zhenting Wang, Yongye Su, Weidi Luo, Hang Gao, Fan Yang, Ruixiang Tang, Yongfeng Zhang

**Abstract**: The performance of Large Language Models (LLMs) depends on the quality of prompts and the semantic and structural integrity of the input data. However, existing prompt generation methods primarily focus on well-structured input data, often neglecting the impact of perturbed inputs on prompt effectiveness. To address this limitation, we propose BATprompt (By Adversarial Training prompt), a novel method for prompt generation designed to withstand input perturbations (such as typos in the input). Inspired by adversarial training techniques, BATprompt demonstrates strong performance on a variety of perturbed tasks through a two-step process: adversarial perturbation and iterative optimization on unperturbed input via LLM. Unlike conventional adversarial attack methods, BATprompt does not need access to model parameters and gradients. Instead, BATprompt leverages the advanced reasoning, language understanding and self reflection capabilities of LLMs to simulate gradients, guiding the generation of adversarial perturbations and optimizing prompt performance. We evaluate BATprompt on multiple datasets across both language understanding and generation tasks. The results indicate that BATprompt outperforms existing prompt generation methods, delivering superior robustness and performance under diverse perturbation scenarios.

摘要: 大型语言模型(LLM)的性能取决于提示的质量以及输入数据的语义和结构完整性。然而，现有的提示生成方法主要关注结构良好的输入数据，往往忽略了扰动输入对提示效果的影响。为了解决这一局限性，我们提出了一种新的提示生成方法BATprint(通过对抗性训练提示)，该方法旨在抵抗输入扰动(如输入中的打字错误)。受到对抗性训练技术的启发，通过两步过程：对抗性扰动和通过LLM对不受扰动的输入进行迭代优化，BATprint在各种扰动任务上表现出了强大的性能。与传统的对抗性攻击方法不同，BATprint不需要访问模型参数和梯度。相反，BATprint利用LLMS的高级推理、语言理解和自我反思能力来模拟梯度，指导产生对抗性扰动并优化提示性能。我们在语言理解和生成任务的多个数据集上评估BATprint。结果表明，BATprint的性能优于现有的提示生成方法，在不同的扰动场景下都具有较好的健壮性和性能。



## **27. Learning to Rewrite: Generalized LLM-Generated Text Detection**

学习重写：广义LLM生成的文本检测 cs.CL

**SubmitDate**: 2025-02-15    [abs](http://arxiv.org/abs/2408.04237v2) [paper-pdf](http://arxiv.org/pdf/2408.04237v2)

**Authors**: Ran Li, Wei Hao, Weiliang Zhao, Junfeng Yang, Chengzhi Mao

**Abstract**: Large language models (LLMs) present significant risks when used to generate non-factual content and spread disinformation at scale. Detecting such LLM-generated content is crucial, yet current detectors often struggle to generalize in open-world contexts. We introduce Learning2Rewrite, a novel framework for detecting AI-generated text with exceptional generalization to unseen domains. Our method leverages the insight that LLMs inherently modify AI-generated content less than human-written text when tasked with rewriting. By training LLMs to minimize alterations on AI-generated inputs, we amplify this disparity, yielding a more distinguishable and generalizable edit distance across diverse text distributions. Extensive experiments on data from 21 independent domains and four major LLMs (GPT-3.5, GPT-4, Gemini, and Llama-3) demonstrate that our detector outperforms state-of-the-art detection methods by up to 23.04% in AUROC for in-distribution tests, 37.26% for out-of-distribution tests, and 48.66% under adversarial attacks. Our unique training objective ensures better generalizability compared to directly training for classification, when leveraging the same amount of parameters. Our findings suggest that reinforcing LLMs' inherent rewriting tendencies offers a robust and scalable solution for detecting AI-generated text.

摘要: 大型语言模型(LLM)在用于生成非事实内容和大规模传播虚假信息时会带来重大风险。检测这种LLM生成的内容是至关重要的，但目前的检测器往往难以在开放世界的环境中进行推广。我们介绍了Learning2Rewrite，一个新的框架，用于检测人工智能生成的文本，具有对不可见领域的特殊泛化。我们的方法利用了这样一种见解，即当执行重写任务时，LLMS天生就不会修改人工智能生成的内容，而不是人类编写的文本。通过训练LLM以最大限度地减少人工智能生成的输入的更改，我们放大了这种差异，在不同的文本分布上产生了更可区分和更具普遍性的编辑距离。在21个独立域和四个主要LLM(GPT-3.5、GPT-4、Gemini和Llama-3)上的大量实验表明，对于分布内测试，我们的检测器在AUROC上的性能比最先进的检测方法高23.04%，对于分布外测试，我们的检测器性能高达37.26%，在对手攻击下，我们的检测器性能高达48.66%。我们独特的训练目标确保了在利用相同数量的参数时，与直接用于分类的训练相比，具有更好的通用性。我们的发现表明，加强LMS固有的重写倾向为检测人工智能生成的文本提供了一种健壮和可扩展的解决方案。



## **28. SequentialBreak: Large Language Models Can be Fooled by Embedding Jailbreak Prompts into Sequential Prompt Chains**

SequentialBreak：大型语言模型可以通过将越狱提示嵌入序列提示链来愚弄 cs.CR

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2411.06426v2) [paper-pdf](http://arxiv.org/pdf/2411.06426v2)

**Authors**: Bijoy Ahmed Saiem, MD Sadik Hossain Shanto, Rakib Ahsan, Md Rafi ur Rashid

**Abstract**: As the integration of the Large Language Models (LLMs) into various applications increases, so does their susceptibility to misuse, raising significant security concerns. Numerous jailbreak attacks have been proposed to assess the security defense of LLMs. Current jailbreak attacks mainly rely on scenario camouflage, prompt obfuscation, prompt optimization, and prompt iterative optimization to conceal malicious prompts. In particular, sequential prompt chains in a single query can lead LLMs to focus on certain prompts while ignoring others, facilitating context manipulation. This paper introduces SequentialBreak, a novel jailbreak attack that exploits this vulnerability. We discuss several scenarios, not limited to examples like Question Bank, Dialog Completion, and Game Environment, where the harmful prompt is embedded within benign ones that can fool LLMs into generating harmful responses. The distinct narrative structures of these scenarios show that SequentialBreak is flexible enough to adapt to various prompt formats beyond those discussed. Extensive experiments demonstrate that SequentialBreak uses only a single query to achieve a substantial gain of attack success rate over existing baselines against both open-source and closed-source models. Through our research, we highlight the urgent need for more robust and resilient safeguards to enhance LLM security and prevent potential misuse. All the result files and website associated with this research are available in this GitHub repository: https://anonymous.4open.science/r/JailBreakAttack-4F3B/.

摘要: 随着大型语言模型(LLM)集成到各种应用程序中的增加，它们也更容易被误用，从而引发了重大的安全问题。已经提出了许多越狱攻击来评估LLMS的安全防御。当前越狱攻击主要依靠场景伪装、提示混淆、提示优化、提示迭代优化来隐藏恶意提示。特别是，单个查询中的顺序提示链可能会导致LLM专注于某些提示，而忽略其他提示，从而促进上下文操作。本文介绍了SequentialBreak，一种利用该漏洞的新型越狱攻击。我们讨论了几种场景，不限于题库、对话完成和游戏环境等示例，在这些场景中，有害提示嵌入到良性提示中，可以欺骗LLM生成有害响应。这些场景的不同叙事结构表明，SequentialBreak足够灵活，可以适应所讨论的各种提示格式。大量的实验表明，SequentialBreak只使用一次查询，在开源和封闭源代码模型下，攻击成功率都比现有的基线有很大的提高。通过我们的研究，我们强调迫切需要更强大和更具弹性的保障措施，以增强LLM安全并防止潜在的滥用。所有与这项研究相关的结果文件和网站都可以在GitHub存储库中找到：https://anonymous.4open.science/r/JailBreakAttack-4F3B/.



## **29. Translating Common Security Assertions Across Processor Designs: A RISC-V Case Study**

跨处理器设计翻译常见安全断言：RISC-V案例研究 cs.CR

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2502.10194v1) [paper-pdf](http://arxiv.org/pdf/2502.10194v1)

**Authors**: Sharjeel Imtiaz, Uljana Reinsalu, Tara Ghasempouri

**Abstract**: RISC-V is gaining popularity for its adaptability and cost-effectiveness in processor design. With the increasing adoption of RISC-V, the importance of implementing robust security verification has grown significantly. In the state of the art, various approaches have been developed to strengthen the security verification process. Among these methods, assertion-based security verification has proven to be a promising approach for ensuring that security features are effectively met. To this end, some approaches manually define security assertions for processor designs; however, these manual methods require significant time, cost, and human expertise. Consequently, recent approaches focus on translating pre-defined security assertions from one design to another. Nonetheless, these methods are not primarily centered on processor security, particularly RISC-V. Furthermore, many of these approaches have not been validated against real-world attacks, such as hardware Trojans. In this work, we introduce a methodology for translating security assertions across processors with different architectures, using RISC-V as a case study. Our approach reduces time and cost compared to developing security assertions manually from the outset. Our methodology was applied to five critical security modules with assertion translation achieving nearly 100% success across all modules. These results validate the efficacy of our approach and highlight its potential for enhancing security verification in modern processor designs. The effectiveness of the translated assertions was rigorously tested against hardware Trojans defined by large language models (LLMs), demonstrating their reliability in detecting security breaches.

摘要: RISC-V因其在处理器设计中的适应性和成本效益而越来越受欢迎。随着RISC-V越来越多地被采用，实施可靠的安全验证的重要性显著增加。在现有技术水平下，已经开发了各种方法来加强安全验证过程。在这些方法中，基于断言的安全验证被证明是确保安全特征得到有效满足的一种很有前途的方法。为此，一些方法手动定义处理器设计的安全断言；然而，这些手动方法需要大量的时间、成本和专业知识。因此，最近的方法侧重于将预定义的安全断言从一种设计转换为另一种设计。尽管如此，这些方法并不主要以处理器安全为中心，特别是RISC-V。此外，这些方法中的许多都没有针对真实世界的攻击进行验证，例如硬件特洛伊木马。在这项工作中，我们介绍了一种在不同体系结构的处理器之间转换安全断言的方法，并以RISC-V为例。与从一开始就手动开发安全断言相比，我们的方法减少了时间和成本。我们的方法被应用于五个关键的安全模块，断言转换在所有模块上几乎都取得了100%的成功。这些结果验证了我们的方法的有效性，并突出了它在现代处理器设计中增强安全性验证的潜力。翻译后的断言的有效性与大型语言模型(LLM)定义的硬件特洛伊木马程序进行了严格的测试，证明了它们在检测安全漏洞方面的可靠性。



## **30. What You See Is Not Always What You Get: An Empirical Study of Code Comprehension by Large Language Models**

你所看到的并不总是你所得到的：大型语言模型对代码理解的实证研究 cs.SE

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2412.08098v2) [paper-pdf](http://arxiv.org/pdf/2412.08098v2)

**Authors**: Bangshuo Zhu, Jiawen Wen, Huaming Chen

**Abstract**: Recent studies have demonstrated outstanding capabilities of large language models (LLMs) in software engineering tasks, including code generation and comprehension. While LLMs have shown significant potential in assisting with coding, it is perceived that LLMs are vulnerable to adversarial attacks. In this paper, we investigate the vulnerability of LLMs to imperceptible attacks, where hidden character manipulation in source code misleads LLMs' behaviour while remaining undetectable to human reviewers. We devise these attacks into four distinct categories and analyse their impacts on code analysis and comprehension tasks. These four types of imperceptible coding character attacks include coding reordering, invisible coding characters, code deletions, and code homoglyphs. To comprehensively benchmark the robustness of current LLMs solutions against the attacks, we present a systematic experimental evaluation on multiple state-of-the-art LLMs. Our experimental design introduces two key performance metrics, namely model confidence using log probabilities of response, and the response correctness. A set of controlled experiments are conducted using a large-scale perturbed and unperturbed code snippets as the primary prompt input. Our findings confirm the susceptibility of LLMs to imperceptible coding character attacks, while different LLMs present different negative correlations between perturbation magnitude and performance. These results highlight the urgent need for robust LLMs capable of manoeuvring behaviours under imperceptible adversarial conditions. We anticipate this work provides valuable insights for enhancing the security and trustworthiness of LLMs in software engineering applications.

摘要: 最近的研究表明，大型语言模型(LLM)在软件工程任务中具有出色的能力，包括代码生成和理解。虽然LLM在协助编码方面显示出巨大的潜力，但人们认为LLM容易受到对手的攻击。在本文中，我们研究了LLMS对不可察觉的攻击的脆弱性，即源代码中的隐藏字符操作误导了LLMS的行为，而人类评审者仍然无法检测到。我们将这些攻击分为四个不同的类别，并分析它们对代码分析和理解任务的影响。这四种类型的不可察觉编码字符攻击包括编码重新排序、不可见编码字符、代码删除和代码同形。为了全面衡量现有LLMS解决方案对攻击的健壮性，我们对多个最先进的LLMS进行了系统的实验评估。我们的实验设计引入了两个关键的性能度量，即使用响应日志概率的模型置信度和响应正确性。使用大规模的扰动和未扰动的代码片段作为主要提示输入，进行了一组对照实验。我们的研究结果证实了LLMS对不可察觉的编码字符攻击的敏感性，而不同的LLM在扰动大小与性能之间呈现不同的负相关。这些结果突出表明，迫切需要能够在难以察觉的对抗性条件下操纵行为的强大的LLM。我们期待这项工作为在软件工程应用中增强LLMS的安全性和可信性提供有价值的见解。



## **31. Detecting Phishing Sites Using ChatGPT**

使用ChatGPT检测网络钓鱼网站 cs.CR

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2306.05816v3) [paper-pdf](http://arxiv.org/pdf/2306.05816v3)

**Authors**: Takashi Koide, Naoki Fukushi, Hiroki Nakano, Daiki Chiba

**Abstract**: The emergence of Large Language Models (LLMs), including ChatGPT, is having a significant impact on a wide range of fields. While LLMs have been extensively researched for tasks such as code generation and text synthesis, their application in detecting malicious web content, particularly phishing sites, has been largely unexplored. To combat the rising tide of cyber attacks due to the misuse of LLMs, it is important to automate detection by leveraging the advanced capabilities of LLMs.   In this paper, we propose a novel system called ChatPhishDetector that utilizes LLMs to detect phishing sites. Our system involves leveraging a web crawler to gather information from websites, generating prompts for LLMs based on the crawled data, and then retrieving the detection results from the responses generated by the LLMs. The system enables us to detect multilingual phishing sites with high accuracy by identifying impersonated brands and social engineering techniques in the context of the entire website, without the need to train machine learning models. To evaluate the performance of our system, we conducted experiments on our own dataset and compared it with baseline systems and several LLMs. The experimental results using GPT-4V demonstrated outstanding performance, with a precision of 98.7% and a recall of 99.6%, outperforming the detection results of other LLMs and existing systems. These findings highlight the potential of LLMs for protecting users from online fraudulent activities and have important implications for enhancing cybersecurity measures.

摘要: 大型语言模型(LLM)的出现，包括ChatGPT，正在对广泛的领域产生重大影响。虽然LLMS已经被广泛研究用于代码生成和文本合成等任务，但它们在检测恶意网络内容，特别是钓鱼网站方面的应用在很大程度上还没有被探索过。为了应对由于滥用LLMS而不断增加的网络攻击浪潮，重要的是通过利用LLMS的高级功能来实现自动检测。在本文中，我们提出了一个新的系统，称为ChatPhishDetector，它利用LLMS来检测钓鱼网站。我们的系统利用网络爬虫从网站收集信息，基于爬行的数据生成LLMS的提示，然后从LLMS生成的响应中检索检测结果。该系统通过在整个网站的上下文中识别假冒品牌和社会工程技术，使我们能够高精度地检测多语言钓鱼网站，而不需要训练机器学习模型。为了评估我们的系统的性能，我们在自己的数据集上进行了实验，并将其与基线系统和几个LLMS进行了比较。基于GPT-4V的实验结果表明，该方法具有较好的性能，准确率为98.7%，召回率为99.6%，优于其他LLMS和现有系统的检测结果。这些发现突出了小岛屿发展中国家保护用户免遭网上欺诈活动的潜力，并对加强网络安全措施具有重要意义。



## **32. ChatIoT: Large Language Model-based Security Assistant for Internet of Things with Retrieval-Augmented Generation**

ChatIOT：基于大语言模型的物联网安全助手，具有检索增强生成 cs.CR

preprint, under revision, 19 pages, 13 figures, 8 tables

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2502.09896v1) [paper-pdf](http://arxiv.org/pdf/2502.09896v1)

**Authors**: Ye Dong, Yan Lin Aung, Sudipta Chattopadhyay, Jianying Zhou

**Abstract**: Internet of Things (IoT) has gained widespread popularity, revolutionizing industries and daily life. However, it has also emerged as a prime target for attacks. Numerous efforts have been made to improve IoT security, and substantial IoT security and threat information, such as datasets and reports, have been developed. However, existing research often falls short in leveraging these insights to assist or guide users in harnessing IoT security practices in a clear and actionable way. In this paper, we propose ChatIoT, a large language model (LLM)-based IoT security assistant designed to disseminate IoT security and threat intelligence. By leveraging the versatile property of retrieval-augmented generation (RAG), ChatIoT successfully integrates the advanced language understanding and reasoning capabilities of LLM with fast-evolving IoT security information. Moreover, we develop an end-to-end data processing toolkit to handle heterogeneous datasets. This toolkit converts datasets of various formats into retrievable documents and optimizes chunking strategies for efficient retrieval. Additionally, we define a set of common use case specifications to guide the LLM in generating answers aligned with users' specific needs and expertise levels. Finally, we implement a prototype of ChatIoT and conduct extensive experiments with different LLMs, such as LLaMA3, LLaMA3.1, and GPT-4o. Experimental evaluations demonstrate that ChatIoT can generate more reliable, relevant, and technical in-depth answers for most use cases. When evaluating the answers with LLaMA3:70B, ChatIoT improves the above metrics by over 10% on average, particularly in relevance and technicality, compared to using LLMs alone.

摘要: 物联网(IoT)获得了广泛的普及，给行业和日常生活带来了革命性的变化。然而，它也已成为攻击的首要目标。为提高物联网安全做出了许多努力，并开发了大量物联网安全和威胁信息，如数据集和报告。然而，现有的研究往往无法利用这些见解来帮助或指导用户以明确和可行的方式驾驭物联网安全做法。在本文中，我们提出了ChatIoT，这是一个基于大型语言模型(LLM)的物联网安全助手，旨在传播物联网安全和威胁情报。通过利用检索增强生成(RAG)的多功能性，ChatIoT成功地将LLM的高级语言理解和推理能力与快速发展的物联网安全信息集成在一起。此外，我们还开发了一个端到端的数据处理工具包来处理异类数据集。该工具包将各种格式的数据集转换为可检索的文档，并优化了分块策略以实现高效的检索。此外，我们定义了一组常见的用例规范，以指导LLM生成与用户的特定需求和专业知识级别一致的答案。最后，我们实现了一个ChatIoT原型，并在不同的LLM上进行了广泛的实验，如LLaMA3、LLaMA3.1和GPT-4o。实验评估表明，ChatIoT可以为大多数用例生成更可靠、更相关和更具技术深度的答案。在使用LLaMA3：70B评估答案时，与单独使用LLMS相比，ChatIoT将上述指标平均提高了10%以上，尤其是在相关性和技术性方面。



## **33. DomainLynx: Leveraging Large Language Models for Enhanced Domain Squatting Detection**

DomainLynx：利用大型语言模型进行增强的域蹲位检测 cs.CR

Originally presented at IEEE CCNC 2025. An extended version of this  work has been published in IEEE Access:  https://doi.org/10.1109/ACCESS.2025.3542036

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2410.02095v3) [paper-pdf](http://arxiv.org/pdf/2410.02095v3)

**Authors**: Daiki Chiba, Hiroki Nakano, Takashi Koide

**Abstract**: Domain squatting poses a significant threat to Internet security, with attackers employing increasingly sophisticated techniques. This study introduces DomainLynx, an innovative compound AI system leveraging Large Language Models (LLMs) for enhanced domain squatting detection. Unlike existing methods focusing on predefined patterns for top-ranked domains, DomainLynx excels in identifying novel squatting techniques and protecting less prominent brands. The system's architecture integrates advanced data processing, intelligent domain pairing, and LLM-powered threat assessment. Crucially, DomainLynx incorporates specialized components that mitigate LLM hallucinations, ensuring reliable and context-aware detection. This approach enables efficient analysis of vast security data from diverse sources, including Certificate Transparency logs, Passive DNS records, and zone files. Evaluated on a curated dataset of 1,649 squatting domains, DomainLynx achieved 94.7\% accuracy using Llama-3-70B. In a month-long real-world test, it detected 34,359 squatting domains from 2.09 million new domains, outperforming baseline methods by 2.5 times. This research advances Internet security by providing a versatile, accurate, and adaptable tool for combating evolving domain squatting threats. DomainLynx's approach paves the way for more robust, AI-driven cybersecurity solutions, enhancing protection for a broader range of online entities and contributing to a safer digital ecosystem.

摘要: 随着攻击者使用越来越复杂的技术，域名抢占对互联网安全构成了重大威胁。这项研究介绍了DomainLynx，这是一个创新的复合人工智能系统，利用大型语言模型(LLM)来增强域占用检测。与专注于排名靠前的域名的预定义模式的现有方法不同，DomainLynx在识别新颖的蹲守技术和保护不太知名的品牌方面表现出色。该系统的架构集成了先进的数据处理、智能域配对和LLM支持的威胁评估。至关重要的是，DomainLynx结合了专门的组件来缓解LLM幻觉，确保可靠和上下文感知的检测。这种方法可以有效地分析来自不同来源的大量安全数据，包括证书透明日志、被动DNS记录和区域文件。在1,649个蹲点域的精选数据集上进行评估，DomainLynx使用LLAMA-3-70B获得了94.7%的准确率。在一个月的实际测试中，它从209万个新域名中检测到34359个蹲点域名，比基线方法高出2.5倍。这项研究通过提供一种通用、准确和适应性强的工具来对抗不断变化的域名抢占威胁，从而促进了互联网安全。DomainLynx的方法为更强大的、人工智能驱动的网络安全解决方案铺平了道路，加强了对更广泛的在线实体的保护，并为更安全的数字生态系统做出了贡献。



## **34. `Do as I say not as I do': A Semi-Automated Approach for Jailbreak Prompt Attack against Multimodal LLMs**

“照我说的做，而不是照我做的做”：针对多模式LLM的越狱提示攻击的半自动方法 cs.CR

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.00735v2) [paper-pdf](http://arxiv.org/pdf/2502.00735v2)

**Authors**: Chun Wai Chiu, Linghan Huang, Bo Li, Huaming Chen

**Abstract**: Large Language Models (LLMs) have seen widespread applications across various domains due to their growing ability to process diverse types of input data, including text, audio, image and video. While LLMs have demonstrated outstanding performance in understanding and generating contexts for different scenarios, they are vulnerable to prompt-based attacks, which are mostly via text input. In this paper, we introduce the first voice-based jailbreak attack against multimodal LLMs, termed as Flanking Attack, which can process different types of input simultaneously towards the multimodal LLMs. Our work is motivated by recent advancements in monolingual voice-driven large language models, which have introduced new attack surfaces beyond traditional text-based vulnerabilities for LLMs. To investigate these risks, we examine the state-of-the-art multimodal LLMs, which can be accessed via different types of inputs such as audio input, focusing on how adversarial prompts can bypass its defense mechanisms. We propose a novel strategy, in which the disallowed prompt is flanked by benign, narrative-driven prompts. It is integrated in the Flanking Attack which attempts to humanizes the interaction context and execute the attack through a fictional setting. Further, to better evaluate the attack performance, we present a semi-automated self-assessment framework for policy violation detection. We demonstrate that Flanking Attack is capable of manipulating state-of-the-art LLMs into generating misaligned and forbidden outputs, which achieves an average attack success rate ranging from 0.67 to 0.93 across seven forbidden scenarios.

摘要: 大语言模型因其处理文本、音频、图像和视频等不同类型输入数据的能力日益增强，在各个领域得到了广泛的应用。虽然LLM在理解和生成不同场景的上下文方面表现出了出色的性能，但它们很容易受到基于提示的攻击，这些攻击主要是通过文本输入进行的。在本文中，我们介绍了第一个基于语音的针对多模式LLMS的越狱攻击，称为侧翼攻击，它可以同时处理针对多模式LLMS的不同类型的输入。我们的工作是受到单语言语音驱动的大型语言模型的最新进展的推动，这些模型在传统的基于文本的LLMS漏洞之外引入了新的攻击面。为了调查这些风险，我们研究了最先进的多模式LLMS，这些LLMS可以通过不同类型的输入(如音频输入)访问，重点关注对抗性提示如何绕过其防御机制。我们提出了一种新的策略，在不允许的提示的两侧是良性的、叙事驱动的提示。它被整合到侧翼攻击中，试图使交互上下文人性化，并通过虚构的设置执行攻击。此外，为了更好地评估攻击性能，我们提出了一个半自动的策略违规检测自我评估框架。我们证明了侧翼攻击能够操纵最先进的LLM产生未对齐和禁止的输出，在七个禁止场景中获得了从0.67到0.93的平均攻击成功率。



## **35. Enhancing Jailbreak Attacks via Compliance-Refusal-Based Initialization**

通过基于合规拒绝的收件箱增强越狱攻击 cs.CR

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.09755v1) [paper-pdf](http://arxiv.org/pdf/2502.09755v1)

**Authors**: Amit Levi, Rom Himelstein, Yaniv Nemcovsky, Avi Mendelson, Chaim Baskin

**Abstract**: Jailbreak attacks aim to exploit large language models (LLMs) and pose a significant threat to their proper conduct; they seek to bypass models' safeguards and often provoke transgressive behaviors. However, existing automatic jailbreak attacks require extensive computational resources and are prone to converge on suboptimal solutions. In this work, we propose \textbf{C}ompliance \textbf{R}efusal \textbf{I}nitialization (CRI), a novel, attack-agnostic framework that efficiently initializes the optimization in the proximity of the compliance subspace of harmful prompts. By narrowing the initial gap to the adversarial objective, CRI substantially improves adversarial success rates (ASR) and drastically reduces computational overhead -- often requiring just a single optimization step. We evaluate CRI on the widely-used AdvBench dataset over the standard jailbreak attacks of GCG and AutoDAN. Results show that CRI boosts ASR and decreases the median steps to success by up to \textbf{\(\times 60\)}. The project page, along with the reference implementation, is publicly available at \texttt{https://amit1221levi.github.io/CRI-Jailbreak-Init-LLMs-evaluation/}.

摘要: 越狱攻击旨在利用大型语言模型(LLM)，并对它们的正常行为构成重大威胁；它们试图绕过模型的保护措施，通常会引发越轨行为。然而，现有的自动越狱攻击需要大量的计算资源，并且容易收敛到次优解。在这项工作中，我们提出了一种新的攻击不可知框架-顺从性通过缩小与对抗性目标的初始差距，CRI极大地提高了对抗性成功率(ASR)，并显著减少了计算开销--通常只需要单个优化步骤。针对GCG和AutoDAN的标准越狱攻击，我们在广泛使用的AdvBtch数据集上对CRI进行了评估。结果表明，CRI提高了ASR，并将成功步骤的中位数减少了高达Textbf{(60倍)}。项目页面以及参考实现可在\texttt{https://amit1221levi.github.io/CRI-Jailbreak-Init-LLMs-evaluation/}.上公开获取



## **36. Making Them a Malicious Database: Exploiting Query Code to Jailbreak Aligned Large Language Models**

将其打造成恶意数据库：利用查询代码越狱对齐的大型语言模型 cs.CR

15 pages, 11 figures

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.09723v1) [paper-pdf](http://arxiv.org/pdf/2502.09723v1)

**Authors**: Qingsong Zou, Jingyu Xiao, Qing Li, Zhi Yan, Yuhang Wang, Li Xu, Wenxuan Wang, Kuofeng Gao, Ruoyu Li, Yong Jiang

**Abstract**: Recent advances in large language models (LLMs) have demonstrated remarkable potential in the field of natural language processing. Unfortunately, LLMs face significant security and ethical risks. Although techniques such as safety alignment are developed for defense, prior researches reveal the possibility of bypassing such defenses through well-designed jailbreak attacks. In this paper, we propose QueryAttack, a novel framework to systematically examine the generalizability of safety alignment. By treating LLMs as knowledge databases, we translate malicious queries in natural language into code-style structured query to bypass the safety alignment mechanisms of LLMs. We conduct extensive experiments on mainstream LLMs, ant the results show that QueryAttack achieves high attack success rates (ASRs) across LLMs with different developers and capabilities. We also evaluate QueryAttack's performance against common defenses, confirming that it is difficult to mitigate with general defensive techniques. To defend against QueryAttack, we tailor a defense method which can reduce ASR by up to 64\% on GPT-4-1106. The code of QueryAttack can be found on https://anonymous.4open.science/r/QueryAttack-334B.

摘要: 大语言模型的最新进展在自然语言处理领域显示出巨大的潜力。不幸的是，低收入国家面临着重大的安全和道德风险。尽管安全对准等技术是为了防御而开发的，但之前的研究表明，通过精心设计的越狱攻击来绕过此类防御是可能的。在本文中，我们提出了一种新的框架QueryAttack，用于系统地检查安全对齐的泛化能力。通过将LLMS视为知识库，将自然语言中的恶意查询转换为代码式的结构化查询，绕过LLMS的安全对齐机制。我们在主流的LLM上进行了大量的实验，结果表明QueryAttack在不同开发者和能力的LLMS上取得了很高的攻击成功率(ASR)。我们还评估了QueryAttack在常见防守下的表现，证实了一般防守技术很难缓解这一点。为了抵抗QueryAttack攻击，我们定制了一种防御方法，在GPT-4-1106上可以将ASR降低高达。QueryAttack的代码可以在https://anonymous.4open.science/r/QueryAttack-334B.上找到



## **37. APT-LLM: Embedding-Based Anomaly Detection of Cyber Advanced Persistent Threats Using Large Language Models**

APT-LLM：使用大型语言模型对网络高级持续性威胁进行基于嵌入的异常检测 cs.CR

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.09385v1) [paper-pdf](http://arxiv.org/pdf/2502.09385v1)

**Authors**: Sidahmed Benabderrahmane, Petko Valtchev, James Cheney, Talal Rahwan

**Abstract**: Advanced Persistent Threats (APTs) pose a major cybersecurity challenge due to their stealth and ability to mimic normal system behavior, making detection particularly difficult in highly imbalanced datasets. Traditional anomaly detection methods struggle to effectively differentiate APT-related activities from benign processes, limiting their applicability in real-world scenarios. This paper introduces APT-LLM, a novel embedding-based anomaly detection framework that integrates large language models (LLMs) -- BERT, ALBERT, DistilBERT, and RoBERTa -- with autoencoder architectures to detect APTs. Unlike prior approaches, which rely on manually engineered features or conventional anomaly detection models, APT-LLM leverages LLMs to encode process-action provenance traces into semantically rich embeddings, capturing nuanced behavioral patterns. These embeddings are analyzed using three autoencoder architectures -- Baseline Autoencoder (AE), Variational Autoencoder (VAE), and Denoising Autoencoder (DAE) -- to model normal process behavior and identify anomalies. The best-performing model is selected for comparison against traditional methods. The framework is evaluated on real-world, highly imbalanced provenance trace datasets from the DARPA Transparent Computing program, where APT-like attacks constitute as little as 0.004\% of the data across multiple operating systems (Android, Linux, BSD, and Windows) and attack scenarios. Results demonstrate that APT-LLM significantly improves detection performance under extreme imbalance conditions, outperforming existing anomaly detection methods and highlighting the effectiveness of LLM-based feature extraction in cybersecurity.

摘要: 高级持续性威胁(APT)由于其隐蔽性和模仿正常系统行为的能力，构成了重大的网络安全挑战，使得在高度不平衡的数据集中检测尤其困难。传统的异常检测方法难以有效地区分与APT相关的活动和良性过程，限制了它们在现实世界场景中的适用性。介绍了一种新的基于嵌入式的异常检测框架APT-LLM，该框架集成了大型语言模型(LLM)--BERT、ALBERT、DistilBERT和Roberta--以及自动编码器体系结构来检测APT。与之前依赖于手动设计的特征或传统异常检测模型的方法不同，APT-LLM利用LLM将流程操作起源跟踪编码到语义丰富的嵌入中，从而捕获细微差别的行为模式。使用三种自动编码器体系结构--基线自动编码器(AE)、变分自动编码器(VAE)和去噪自动编码器(DAE)--对这些嵌入进行分析，以模拟正常进程行为并识别异常。选择性能最好的模型与传统方法进行比较。该框架是在来自DARPA透明计算计划的高度不平衡的来源跟踪数据集上进行评估的，其中APT类攻击在多个操作系统(安卓、LINUX、BSD和Windows)和攻击场景中仅占数据的0.004\%。结果表明，APT-LLM显著提高了极端不平衡条件下的检测性能，优于现有的异常检测方法，突出了基于LLM的特征提取在网络安全中的有效性。



## **38. Privacy Checklist: Privacy Violation Detection Grounding on Contextual Integrity Theory**

隐私检查表：基于上下文完整性理论的隐私侵犯检测 cs.CL

To appear at NAACL 25

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2408.10053v2) [paper-pdf](http://arxiv.org/pdf/2408.10053v2)

**Authors**: Haoran Li, Wei Fan, Yulin Chen, Jiayang Cheng, Tianshu Chu, Xuebing Zhou, Peizhao Hu, Yangqiu Song

**Abstract**: Privacy research has attracted wide attention as individuals worry that their private data can be easily leaked during interactions with smart devices, social platforms, and AI applications. Computer science researchers, on the other hand, commonly study privacy issues through privacy attacks and defenses on segmented fields. Privacy research is conducted on various sub-fields, including Computer Vision (CV), Natural Language Processing (NLP), and Computer Networks. Within each field, privacy has its own formulation. Though pioneering works on attacks and defenses reveal sensitive privacy issues, they are narrowly trapped and cannot fully cover people's actual privacy concerns. Consequently, the research on general and human-centric privacy research remains rather unexplored. In this paper, we formulate the privacy issue as a reasoning problem rather than simple pattern matching. We ground on the Contextual Integrity (CI) theory which posits that people's perceptions of privacy are highly correlated with the corresponding social context. Based on such an assumption, we develop the first comprehensive checklist that covers social identities, private attributes, and existing privacy regulations. Unlike prior works on CI that either cover limited expert annotated norms or model incomplete social context, our proposed privacy checklist uses the whole Health Insurance Portability and Accountability Act of 1996 (HIPAA) as an example, to show that we can resort to large language models (LLMs) to completely cover the HIPAA's regulations. Additionally, our checklist also gathers expert annotations across multiple ontologies to determine private information including but not limited to personally identifiable information (PII). We use our preliminary results on the HIPAA to shed light on future context-centric privacy research to cover more privacy regulations, social norms and standards.

摘要: 隐私研究吸引了广泛的关注，因为个人担心他们的私人数据在与智能设备、社交平台和人工智能应用程序交互时很容易被泄露。另一方面，计算机科学研究人员通常通过对分割的领域进行隐私攻击和防御来研究隐私问题。隐私研究在不同的子领域进行，包括计算机视觉(CV)、自然语言处理(NLP)和计算机网络。在每个领域，隐私都有自己的表述。尽管攻击和防御方面的开创性作品揭示了敏感的隐私问题，但它们被狭隘地困住了，不能完全覆盖人们对隐私的实际担忧。因此，关于一般隐私研究和以人为中心的隐私研究仍然是相当未被探索的。在本文中，我们将隐私问题描述为一个推理问题，而不是简单的模式匹配。我们基于语境完整性(CI)理论，该理论认为人们对隐私的感知与相应的社会语境高度相关。基于这样的假设，我们开发了第一个全面的清单，其中包括社会身份、私人属性和现有的隐私法规。与以往关于CI的工作要么涵盖有限的专家注释规范，要么涵盖不完整的社会背景，我们提出的隐私检查表以1996年的整个健康保险携带和责任法案(HIPAA)为例，表明我们可以求助于大型语言模型(LLM)来完全覆盖HIPAA的规定。此外，我们的检查表还收集了跨多个本体的专家注释，以确定私人信息，包括但不限于个人身份信息(PII)。我们使用我们在HIPAA上的初步结果来阐明未来以上下文为中心的隐私研究，以涵盖更多的隐私法规、社会规范和标准。



## **39. FLAME: Flexible LLM-Assisted Moderation Engine**

FLAME：灵活的LLM辅助审核引擎 cs.CR

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.09175v1) [paper-pdf](http://arxiv.org/pdf/2502.09175v1)

**Authors**: Ivan Bakulin, Ilia Kopanichuk, Iaroslav Bespalov, Nikita Radchenko, Vladimir Shaposhnikov, Dmitry Dylov, Ivan Oseledets

**Abstract**: The rapid advancement of Large Language Models (LLMs) has introduced significant challenges in moderating user-model interactions. While LLMs demonstrate remarkable capabilities, they remain vulnerable to adversarial attacks, particularly ``jailbreaking'' techniques that bypass content safety measures. Current content moderation systems, which primarily rely on input prompt filtering, have proven insufficient, with techniques like Best-of-N (BoN) jailbreaking achieving success rates of 80% or more against popular LLMs. In this paper, we introduce Flexible LLM-Assisted Moderation Engine (FLAME): a new approach that shifts the focus from input filtering to output moderation. Unlike traditional circuit-breaking methods that analyze user queries, FLAME evaluates model responses, offering several key advantages: (1) computational efficiency in both training and inference, (2) enhanced resistance to BoN jailbreaking attacks, and (3) flexibility in defining and updating safety criteria through customizable topic filtering. Our experiments demonstrate that FLAME significantly outperforms current moderation systems. For example, FLAME reduces attack success rate in GPT-4o-mini and DeepSeek-v3 by a factor of ~9, while maintaining low computational overhead. We provide comprehensive evaluation on various LLMs and analyze the engine's efficiency against the state-of-the-art jailbreaking. This work contributes to the development of more robust and adaptable content moderation systems for LLMs.

摘要: 大型语言模型(LLM)的快速发展给协调用户与模型的交互带来了巨大的挑战。虽然LLM显示出非凡的能力，但它们仍然容易受到对抗性攻击，特别是绕过内容安全措施的“越狱”技术。目前的内容审核系统主要依赖于输入提示过滤，已被证明是不够的，像N中最佳(Bon)越狱技术对流行的LLM的成功率达到80%或更高。在本文中，我们介绍了灵活的LLM辅助调节引擎(FLAME)：一种将焦点从输入过滤转移到输出调节的新方法。与分析用户查询的传统断路方法不同，FLAME评估模型响应，提供了几个关键优势：(1)训练和推理的计算效率，(2)增强了对Bon越狱攻击的抵抗，以及(3)通过可定制的主题过滤灵活地定义和更新安全标准。我们的实验表明，火焰系统的性能明显优于现有的慢化系统。例如，FLAME将GPT-40-mini和DeepSeek-v3中的攻击成功率降低了~9倍，同时保持了较低的计算开销。我们对各种LLM进行了综合评估，并针对最先进的越狱情况分析了发动机的效率。这项工作有助于开发更健壮和适应性更强的LLMS内容审核系统。



## **40. Universal Adversarial Attack on Aligned Multimodal LLMs**

对对齐多模式LLM的普遍对抗攻击 cs.AI

Added an affiliation

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.07987v2) [paper-pdf](http://arxiv.org/pdf/2502.07987v2)

**Authors**: Temurbek Rahmatullaev, Polina Druzhinina, Matvey Mikhalchuk, Andrey Kuznetsov, Anton Razzhigaev

**Abstract**: We propose a universal adversarial attack on multimodal Large Language Models (LLMs) that leverages a single optimized image to override alignment safeguards across diverse queries and even multiple models. By backpropagating through the vision encoder and language head, we craft a synthetic image that forces the model to respond with a targeted phrase (e.g., ''Sure, here it is'') or otherwise unsafe content-even for harmful prompts. In experiments on the SafeBench benchmark, our method achieves significantly higher attack success rates than existing baselines, including text-only universal prompts (e.g., up to 93% on certain models). We further demonstrate cross-model transferability by training on several multimodal LLMs simultaneously and testing on unseen architectures. Additionally, a multi-answer variant of our approach produces more natural-sounding (yet still malicious) responses. These findings underscore critical vulnerabilities in current multimodal alignment and call for more robust adversarial defenses. We will release code and datasets under the Apache-2.0 license. Warning: some content generated by Multimodal LLMs in this paper may be offensive to some readers.

摘要: 我们提出了一种针对多模式大型语言模型(LLMS)的通用对抗性攻击，该攻击利用单个优化图像覆盖跨不同查询甚至多个模型的对齐保障。通过视觉编码器和语言头部的反向传播，我们制作了一个合成图像，迫使模型使用有针对性的短语(例如，“当然，就是这里”)或其他不安全的内容做出响应--即使是有害的提示。在SafeBtch基准测试上的实验中，我们的方法获得了比现有基线显著更高的攻击成功率，包括纯文本通用提示(例如，在某些型号上高达93%)。我们通过同时在多个多模式LLM上进行训练和在看不见的体系结构上进行测试来进一步证明跨模型的可转移性。此外，我们的方法的一个多答案变体会产生听起来更自然(但仍然是恶意的)响应。这些发现突显了当前多模式联合的严重弱点，并呼吁进行更强大的对抗性防御。我们将在APACHE-2.0许可下发布代码和数据集。警告：本文中的多模式LLMS生成的某些内容可能会冒犯某些读者。



## **41. Vision-LLMs Can Fool Themselves with Self-Generated Typographic Attacks**

Vision-LLM可以通过自我生成的印刷攻击来欺骗自己 cs.CV

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2402.00626v3) [paper-pdf](http://arxiv.org/pdf/2402.00626v3)

**Authors**: Maan Qraitem, Nazia Tasnim, Piotr Teterwak, Kate Saenko, Bryan A. Plummer

**Abstract**: Typographic attacks, adding misleading text to images, can deceive vision-language models (LVLMs). The susceptibility of recent large LVLMs like GPT4-V to such attacks is understudied, raising concerns about amplified misinformation in personal assistant applications. Previous attacks use simple strategies, such as random misleading words, which don't fully exploit LVLMs' language reasoning abilities. We introduce an experimental setup for testing typographic attacks on LVLMs and propose two novel self-generated attacks: (1) Class-based attacks, where the model identifies a similar class to deceive itself, and (2) Reasoned attacks, where an advanced LVLM suggests an attack combining a deceiving class and description. Our experiments show these attacks significantly reduce classification performance by up to 60\% and are effective across different models, including InstructBLIP and MiniGPT4. Code: https://github.com/mqraitem/Self-Gen-Typo-Attack

摘要: 印刷攻击（向图像添加误导性文本）可以欺骗视觉语言模型（LVLM）。最近，GPT 4-V等大型LVLM对此类攻击的敏感性尚未得到充分研究，这引发了人们对个人助理应用程序中被放大的错误信息的担忧。之前的攻击使用简单的策略，例如随机误导性单词，这些策略没有充分利用LVLM的语言推理能力。我们引入了一个实验设置来测试对LVLM的印刷攻击，并提出了两种新颖的自发攻击：（1）基于类的攻击，其中模型识别类似的类来欺骗自己，和（2）推理攻击，其中高级LVLM建议结合欺骗类和描述的攻击。我们的实验表明，这些攻击将分类性能显着降低高达60%，并且在不同的模型中有效，包括DirectBLIP和MiniGPT 4。代码：https://github.com/mqraitem/Self-Gen-Typo-Attack



## **42. An Engorgio Prompt Makes Large Language Model Babble on**

Engorgio提示让大型语言模型胡言乱语 cs.CR

ICLR 2025

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2412.19394v2) [paper-pdf](http://arxiv.org/pdf/2412.19394v2)

**Authors**: Jianshuo Dong, Ziyuan Zhang, Qingjie Zhang, Tianwei Zhang, Hao Wang, Hewu Li, Qi Li, Chao Zhang, Ke Xu, Han Qiu

**Abstract**: Auto-regressive large language models (LLMs) have yielded impressive performance in many real-world tasks. However, the new paradigm of these LLMs also exposes novel threats. In this paper, we explore their vulnerability to inference cost attacks, where a malicious user crafts Engorgio prompts to intentionally increase the computation cost and latency of the inference process. We design Engorgio, a novel methodology, to efficiently generate adversarial Engorgio prompts to affect the target LLM's service availability. Engorgio has the following two technical contributions. (1) We employ a parameterized distribution to track LLMs' prediction trajectory. (2) Targeting the auto-regressive nature of LLMs' inference process, we propose novel loss functions to stably suppress the appearance of the <EOS> token, whose occurrence will interrupt the LLM's generation process. We conduct extensive experiments on 13 open-sourced LLMs with parameters ranging from 125M to 30B. The results show that Engorgio prompts can successfully induce LLMs to generate abnormally long outputs (i.e., roughly 2-13$\times$ longer to reach 90%+ of the output length limit) in a white-box scenario and our real-world experiment demonstrates Engergio's threat to LLM service with limited computing resources. The code is released at: https://github.com/jianshuod/Engorgio-prompt.

摘要: 自回归大型语言模型(LLM)在许多实际任务中取得了令人印象深刻的性能。然而，这些LLM的新范式也暴露了新的威胁。在本文中，我们探讨了它们对推理成本攻击的脆弱性，在这种攻击中，恶意用户手工制作Engorgio会提示故意增加推理过程的计算成本和延迟。我们设计了一种新的方法Engorgio来有效地生成对抗性Engorgio提示，以影响目标LLM的服务可用性。Engorgio有以下两个技术贡献。(1)采用一种参数分布来跟踪LLMS的预测轨迹。(2)针对LLMS推理过程的自回归特性，提出了一种新的损失函数来稳定地抑制<EOS>令牌的出现，它的出现会中断LLM的生成过程。我们在13个开源的LLM上进行了广泛的实验，参数从125M到30B不等。结果表明，在白盒场景中，Engorgio提示能够成功地诱导LLMS产生异常长的输出(即大约2-13$\x$以达到输出长度限制的90%以上)，并且我们的真实世界实验证明了Engergio在有限计算资源的情况下对LLM服务的威胁。代码发布地址为：https://github.com/jianshuod/Engorgio-prompt.



## **43. Theoretically Grounded Framework for LLM Watermarking: A Distribution-Adaptive Approach**

LLM水印的理论基础框架：分布自适应方法 cs.CR

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2410.02890v3) [paper-pdf](http://arxiv.org/pdf/2410.02890v3)

**Authors**: Haiyun He, Yepeng Liu, Ziqiao Wang, Yongyi Mao, Yuheng Bu

**Abstract**: Watermarking has emerged as a crucial method to distinguish AI-generated text from human-created text. In this paper, we present a novel theoretical framework for watermarking Large Language Models (LLMs) that jointly optimizes both the watermarking scheme and the detection process. Our approach focuses on maximizing detection performance while maintaining control over the worst-case Type-I error and text distortion. We characterize \emph{the universally minimum Type-II error}, showing a fundamental trade-off between watermark detectability and text distortion. Importantly, we identify that the optimal watermarking schemes are adaptive to the LLM generative distribution. Building on our theoretical insights, we propose an efficient, model-agnostic, distribution-adaptive watermarking algorithm, utilizing a surrogate model alongside the Gumbel-max trick. Experiments conducted on Llama2-13B and Mistral-8$\times$7B models confirm the effectiveness of our approach. Additionally, we examine incorporating robustness into our framework, paving a way to future watermarking systems that withstand adversarial attacks more effectively.

摘要: 数字水印已经成为区分人工智能生成的文本和人类创建的文本的关键方法。在本文中，我们提出了一种新的大语言模型(LLMS)水印理论框架，该框架同时优化了水印方案和检测过程。我们的方法专注于最大化检测性能，同时保持对最坏情况下的类型I错误和文本失真的控制。我们将其刻画在水印可检测性和文本失真之间的基本权衡。重要的是，我们发现最优水印方案对LLM生成分布是自适应的。基于我们的理论见解，我们提出了一种高效的、与模型无关的、分布自适应的水印算法，该算法利用代理模型和Gumbel-max技巧。在Llama2-13B和Mistral-8$\x$70亿模型上进行的实验证实了该方法的有效性。此外，我们还研究了将健壮性融入到我们的框架中，为未来更有效地抵御对手攻击的水印系统铺平了道路。



## **44. Commercial LLM Agents Are Already Vulnerable to Simple Yet Dangerous Attacks**

商业LLM代理已经容易受到简单但危险的攻击 cs.LG

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.08586v1) [paper-pdf](http://arxiv.org/pdf/2502.08586v1)

**Authors**: Ang Li, Yin Zhou, Vethavikashini Chithrra Raghuram, Tom Goldstein, Micah Goldblum

**Abstract**: A high volume of recent ML security literature focuses on attacks against aligned large language models (LLMs). These attacks may extract private information or coerce the model into producing harmful outputs. In real-world deployments, LLMs are often part of a larger agentic pipeline including memory systems, retrieval, web access, and API calling. Such additional components introduce vulnerabilities that make these LLM-powered agents much easier to attack than isolated LLMs, yet relatively little work focuses on the security of LLM agents. In this paper, we analyze security and privacy vulnerabilities that are unique to LLM agents. We first provide a taxonomy of attacks categorized by threat actors, objectives, entry points, attacker observability, attack strategies, and inherent vulnerabilities of agent pipelines. We then conduct a series of illustrative attacks on popular open-source and commercial agents, demonstrating the immediate practical implications of their vulnerabilities. Notably, our attacks are trivial to implement and require no understanding of machine learning.

摘要: 最近有大量的ML安全文献关注针对对齐的大型语言模型(LLM)的攻击。这些攻击可能会窃取私人信息或迫使模型产生有害的输出。在实际部署中，LLM通常是更大的代理管道的一部分，包括内存系统、检索、Web访问和API调用。这些额外的组件引入了漏洞，使这些由LLM支持的代理比孤立的LLM更容易受到攻击，但相对较少的工作关注LLM代理的安全。在本文中，我们分析了LLM代理独有的安全和隐私漏洞。我们首先根据威胁参与者、目标、入口点、攻击者的可观察性、攻击策略和代理管道的固有漏洞对攻击进行分类。然后，我们对流行的开源和商业代理进行了一系列说明性攻击，展示了它们漏洞的直接实际影响。值得注意的是，我们的攻击很容易实现，并且不需要理解机器学习。



## **45. Why Are My Prompts Leaked? Unraveling Prompt Extraction Threats in Customized Large Language Models**

为什么我的笔记会泄露？解开定制大型语言模型中的提示提取威胁 cs.CL

Source Code: https://github.com/liangzid/PromptExtractionEval

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2408.02416v2) [paper-pdf](http://arxiv.org/pdf/2408.02416v2)

**Authors**: Zi Liang, Haibo Hu, Qingqing Ye, Yaxin Xiao, Haoyang Li

**Abstract**: The drastic increase of large language models' (LLMs) parameters has led to a new research direction of fine-tuning-free downstream customization by prompts, i.e., task descriptions. While these prompt-based services (e.g. OpenAI's GPTs) play an important role in many businesses, there has emerged growing concerns about the prompt leakage, which undermines the intellectual properties of these services and causes downstream attacks. In this paper, we analyze the underlying mechanism of prompt leakage, which we refer to as prompt memorization, and develop corresponding defending strategies. By exploring the scaling laws in prompt extraction, we analyze key attributes that influence prompt extraction, including model sizes, prompt lengths, as well as the types of prompts. Then we propose two hypotheses that explain how LLMs expose their prompts. The first is attributed to the perplexity, i.e. the familiarity of LLMs to texts, whereas the second is based on the straightforward token translation path in attention matrices. To defend against such threats, we investigate whether alignments can undermine the extraction of prompts. We find that current LLMs, even those with safety alignments like GPT-4, are highly vulnerable to prompt extraction attacks, even under the most straightforward user attacks. Therefore, we put forward several defense strategies with the inspiration of our findings, which achieve 83.8\% and 71.0\% drop in the prompt extraction rate for Llama2-7B and GPT-3.5, respectively. Source code is avaliable at https://github.com/liangzid/PromptExtractionEval.

摘要: 大型语言模型(LLMS)参数的急剧增加导致了一个新的研究方向，即通过提示(即任务描述)进行免微调的下游定制。虽然这些基于提示的服务(例如OpenAI的GPT)在许多业务中扮演着重要的角色，但人们越来越担心即时泄露，这会破坏这些服务的知识产权，并导致下游攻击。本文分析了即时记忆的潜在机制，并提出了相应的防御策略。通过研究提示提取中的缩放规律，我们分析了影响提示提取的关键属性，包括模型大小、提示长度以及提示的类型。然后，我们提出了两个假设来解释LLM是如何暴露他们的提示的。第一种归因于迷惑性，即LLMS对文本的熟悉度，而第二种归因于注意矩阵中直接的表征翻译路径。为了防御此类威胁，我们调查对齐是否会破坏提示符的提取。我们发现，即使在最直接的用户攻击下，当前的LLM，即使是那些具有GPT-4等安全对齐的LLM，也非常容易受到即时提取攻击。因此，我们根据研究结果提出了几种防御策略，分别使Llama2-7B和GPT-3.5的即时抽取率下降了83.8%和71.0%。源代码可在https://github.com/liangzid/PromptExtractionEval.上获得



## **46. Modification and Generated-Text Detection: Achieving Dual Detection Capabilities for the Outputs of LLM by Watermark**

修改和生成文本检测：通过水印实现LLM输出的双重检测能力 cs.CR

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.08332v1) [paper-pdf](http://arxiv.org/pdf/2502.08332v1)

**Authors**: Yuhang Cai, Yaofei Wang, Donghui Hu, Gu Chen

**Abstract**: The development of large language models (LLMs) has raised concerns about potential misuse. One practical solution is to embed a watermark in the text, allowing ownership verification through watermark extraction. Existing methods primarily focus on defending against modification attacks, often neglecting other spoofing attacks. For example, attackers can alter the watermarked text to produce harmful content without compromising the presence of the watermark, which could lead to false attribution of this malicious content to the LLM. This situation poses a serious threat to the LLMs service providers and highlights the significance of achieving modification detection and generated-text detection simultaneously. Therefore, we propose a technique to detect modifications in text for unbiased watermark which is sensitive to modification. We introduce a new metric called ``discarded tokens", which measures the number of tokens not included in watermark detection. When a modification occurs, this metric changes and can serve as evidence of the modification. Additionally, we improve the watermark detection process and introduce a novel method for unbiased watermark. Our experiments demonstrate that we can achieve effective dual detection capabilities: modification detection and generated-text detection by watermark.

摘要: 大型语言模型(LLM)的发展引起了人们对潜在滥用的担忧。一种实用的解决方案是在文本中嵌入水印，允许通过提取水印来验证所有权。现有的方法主要集中在防御修改攻击上，往往忽略了其他欺骗攻击。例如，攻击者可以更改带水印的文本以产生有害内容，而不会影响水印的存在，这可能会导致将此恶意内容错误地归因于LLM。这种情况对LLMS服务提供商构成了严重威胁，并突出了同时实现修改检测和生成文本检测的重要性。因此，我们提出了一种文本修改检测技术，以检测对修改敏感的无偏水印。提出了一种新的水印检测方法--“丢弃令牌”，该度量度量了水印检测中未包含的令牌个数。当水印发生修改时，该度量会发生变化，并且可以作为修改的证据。此外，我们对水印检测过程进行了改进，提出了一种新的无偏水印检测方法。实验表明，我们可以实现有效的双重检测能力：修改检测和水印生成文本检测。



## **47. Compromising Honesty and Harmlessness in Language Models via Deception Attacks**

通过欺骗攻击损害语言模型中的诚实和无害 cs.CL

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.08301v1) [paper-pdf](http://arxiv.org/pdf/2502.08301v1)

**Authors**: Laurène Vaugrante, Francesca Carlon, Maluna Menke, Thilo Hagendorff

**Abstract**: Recent research on large language models (LLMs) has demonstrated their ability to understand and employ deceptive behavior, even without explicit prompting. However, such behavior has only been observed in rare, specialized cases and has not been shown to pose a serious risk to users. Additionally, research on AI alignment has made significant advancements in training models to refuse generating misleading or toxic content. As a result, LLMs generally became honest and harmless. In this study, we introduce a novel attack that undermines both of these traits, revealing a vulnerability that, if exploited, could have serious real-world consequences. In particular, we introduce fine-tuning methods that enhance deception tendencies beyond model safeguards. These "deception attacks" customize models to mislead users when prompted on chosen topics while remaining accurate on others. Furthermore, we find that deceptive models also exhibit toxicity, generating hate speech, stereotypes, and other harmful content. Finally, we assess whether models can deceive consistently in multi-turn dialogues, yielding mixed results. Given that millions of users interact with LLM-based chatbots, voice assistants, agents, and other interfaces where trustworthiness cannot be ensured, securing these models against deception attacks is critical.

摘要: 最近对大型语言模型的研究表明，即使在没有明确提示的情况下，它们也能够理解和使用欺骗性行为。然而，这种行为只在罕见的特殊情况下才能观察到，并未被证明对用户构成严重风险。此外，人工智能对齐的研究在拒绝产生误导性或有毒内容的训练模型方面取得了重大进展。结果，LLM通常变得诚实和无害。在这项研究中，我们介绍了一种破坏这两个特征的新型攻击，揭示了一个漏洞，如果利用该漏洞，可能会在现实世界中产生严重后果。特别是，我们引入了微调方法，增强了模型保障之外的欺骗倾向。这些“欺骗攻击”定制模型，在提示用户选择主题时误导用户，而在其他主题上保持准确。此外，我们发现欺骗性模型也表现出毒性，产生仇恨言论、刻板印象和其他有害内容。最后，我们评估模型是否可以在多轮对话中一致地欺骗，产生喜忧参半的结果。鉴于数以百万计的用户与基于LLM的聊天机器人、语音助理、代理和其他无法确保可信度的界面交互，确保这些模型免受欺骗性攻击至关重要。



## **48. Typographic Attacks in a Multi-Image Setting**

多图像环境中的印刷攻击 cs.CR

Accepted by NAACL2025. Our code is available at  https://github.com/XiaomengWang-AI/Typographic-Attacks-in-a-Multi-Image-Setting

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.08193v1) [paper-pdf](http://arxiv.org/pdf/2502.08193v1)

**Authors**: Xiaomeng Wang, Zhengyu Zhao, Martha Larson

**Abstract**: Large Vision-Language Models (LVLMs) are susceptible to typographic attacks, which are misclassifications caused by an attack text that is added to an image. In this paper, we introduce a multi-image setting for studying typographic attacks, broadening the current emphasis of the literature on attacking individual images. Specifically, our focus is on attacking image sets without repeating the attack query. Such non-repeating attacks are stealthier, as they are more likely to evade a gatekeeper than attacks that repeat the same attack text. We introduce two attack strategies for the multi-image setting, leveraging the difficulty of the target image, the strength of the attack text, and text-image similarity. Our text-image similarity approach improves attack success rates by 21% over random, non-specific methods on the CLIP model using ImageNet while maintaining stealth in a multi-image scenario. An additional experiment demonstrates transferability, i.e., text-image similarity calculated using CLIP transfers when attacking InstructBLIP.

摘要: 大型视觉语言模型(LVLM)容易受到排版攻击，这些攻击是由添加到图像中的攻击文本引起的错误分类。在这篇文章中，我们介绍了一种研究排版攻击的多图像环境，拓宽了当前文献对单个图像攻击的重点。具体地说，我们的重点是攻击图像集，而不重复攻击查询。这种不重复的攻击更隐蔽，因为它们比重复相同攻击文本的攻击更有可能避开网守。针对多图像环境，我们提出了两种攻击策略，分别利用了目标图像的难度、攻击文本的强度和文本与图像的相似性。与使用ImageNet的剪辑模型上的随机、非特定方法相比，我们的文本-图像相似性方法将攻击成功率提高了21%，同时在多图像场景中保持了隐蔽性。另外一个实验演示了可转移性，即攻击InstructBLIP时使用片段传输计算的文本-图像相似度。



## **49. The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems**

早起的鸟抓住漏洞：揭开LLM服务系统中的计时侧通道 cs.CR

This work was submitted for review on Sept. 5, 2024, and the initial  version was uploaded to Arxiv on Sept. 30, 2024. The latest version reflects  the up-to-date experimental results

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2409.20002v3) [paper-pdf](http://arxiv.org/pdf/2409.20002v3)

**Authors**: Linke Song, Zixuan Pang, Wenhao Wang, Zihao Wang, XiaoFeng Wang, Hongbo Chen, Wei Song, Yier Jin, Dan Meng, Rui Hou

**Abstract**: The wide deployment of Large Language Models (LLMs) has given rise to strong demands for optimizing their inference performance. Today's techniques serving this purpose primarily focus on reducing latency and improving throughput through algorithmic and hardware enhancements, while largely overlooking their privacy side effects, particularly in a multi-user environment. In our research, for the first time, we discovered a set of new timing side channels in LLM systems, arising from shared caches and GPU memory allocations, which can be exploited to infer both confidential system prompts and those issued by other users. These vulnerabilities echo security challenges observed in traditional computing systems, highlighting an urgent need to address potential information leakage in LLM serving infrastructures. In this paper, we report novel attack strategies designed to exploit such timing side channels inherent in LLM deployments, specifically targeting the Key-Value (KV) cache and semantic cache widely used to enhance LLM inference performance. Our approach leverages timing measurements and classification models to detect cache hits, allowing an adversary to infer private prompts with high accuracy. We also propose a token-by-token search algorithm to efficiently recover shared prompt prefixes in the caches, showing the feasibility of stealing system prompts and those produced by peer users. Our experimental studies on black-box testing of popular online LLM services demonstrate that such privacy risks are completely realistic, with significant consequences. Our findings underscore the need for robust mitigation to protect LLM systems against such emerging threats.

摘要: 大型语言模型的广泛应用对其推理性能的优化提出了强烈的需求。目前用于此目的的技术主要集中在通过算法和硬件增强来减少延迟和提高吞吐量，而在很大程度上忽略了它们的隐私副作用，特别是在多用户环境中。在我们的研究中，我们首次在LLM系统中发现了一组新的计时侧通道，这些通道来自共享缓存和GPU内存分配，可以用来推断机密系统提示和其他用户发出的提示。这些漏洞呼应了在传统计算系统中观察到的安全挑战，突显出迫切需要解决LLM服务基础设施中潜在的信息泄漏问题。在本文中，我们报告了一种新的攻击策略，旨在利用LLM部署中固有的计时侧通道，特别是针对广泛用于提高LLM推理性能的键值(KV)缓存和语义缓存。我们的方法利用计时测量和分类模型来检测缓存命中，允许对手以高精度推断私人提示。我们还提出了一种逐令牌搜索算法来高效地恢复缓存中的共享提示前缀，证明了窃取系统提示和对等用户产生的提示的可行性。我们对流行的在线LLM服务进行黑盒测试的实验研究表明，这种隐私风险是完全现实的，具有显著的后果。我们的发现强调了强有力的缓解措施的必要性，以保护LLM系统免受此类新出现的威胁。



## **50. In-Context Experience Replay Facilitates Safety Red-Teaming of Text-to-Image Diffusion Models**

上下文体验回放促进文本到图像扩散模型的安全红色团队化 cs.LG

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2411.16769v2) [paper-pdf](http://arxiv.org/pdf/2411.16769v2)

**Authors**: Zhi-Yi Chin, Mario Fritz, Pin-Yu Chen, Wei-Chen Chiu

**Abstract**: Text-to-image (T2I) models have shown remarkable progress, but their potential to generate harmful content remains a critical concern in the ML community. While various safety mechanisms have been developed, the field lacks systematic tools for evaluating their effectiveness against real-world misuse scenarios. In this work, we propose ICER, a novel red-teaming framework that leverages Large Language Models (LLMs) and a bandit optimization-based algorithm to generate interpretable and semantic meaningful problematic prompts by learning from past successful red-teaming attempts. Our ICER efficiently probes safety mechanisms across different T2I models without requiring internal access or additional training, making it broadly applicable to deployed systems. Through extensive experiments, we demonstrate that ICER significantly outperforms existing prompt attack methods in identifying model vulnerabilities while maintaining high semantic similarity with intended content. By uncovering that successful jailbreaking instances can systematically facilitate the discovery of new vulnerabilities, our work provides crucial insights for developing more robust safety mechanisms in T2I systems.

摘要: 文本到图像(T2I)模型已经显示出显著的进步，但它们产生有害内容的潜力仍然是ML社区的一个关键问题。虽然已经开发了各种安全机制，但该领域缺乏系统的工具来评估其针对现实世界滥用情况的有效性。在这项工作中，我们提出了ICER，一个新的红团队框架，它利用大型语言模型(LLM)和基于Bandit优化的算法来生成可解释的、有语义意义的问题提示，通过学习过去成功的红团队尝试。我们的ICER可有效探测不同T2I型号的安全机制，无需内部访问或额外培训，使其广泛适用于已部署的系统。通过大量的实验，我们证明了ICER在识别模型漏洞方面明显优于现有的即时攻击方法，同时保持了与预期内容的高度语义相似度。通过揭示成功的越狱实例可以系统地促进新漏洞的发现，我们的工作为在T2I系统中开发更强大的安全机制提供了至关重要的见解。



