# Latest Large Language Model Attack Papers
**update at 2025-02-14 10:55:16**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. APT-LLM: Embedding-Based Anomaly Detection of Cyber Advanced Persistent Threats Using Large Language Models**

APT-LLM：使用大型语言模型对网络高级持续性威胁进行基于嵌入的异常检测 cs.CR

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.09385v1) [paper-pdf](http://arxiv.org/pdf/2502.09385v1)

**Authors**: Sidahmed Benabderrahmane, Petko Valtchev, James Cheney, Talal Rahwan

**Abstract**: Advanced Persistent Threats (APTs) pose a major cybersecurity challenge due to their stealth and ability to mimic normal system behavior, making detection particularly difficult in highly imbalanced datasets. Traditional anomaly detection methods struggle to effectively differentiate APT-related activities from benign processes, limiting their applicability in real-world scenarios. This paper introduces APT-LLM, a novel embedding-based anomaly detection framework that integrates large language models (LLMs) -- BERT, ALBERT, DistilBERT, and RoBERTa -- with autoencoder architectures to detect APTs. Unlike prior approaches, which rely on manually engineered features or conventional anomaly detection models, APT-LLM leverages LLMs to encode process-action provenance traces into semantically rich embeddings, capturing nuanced behavioral patterns. These embeddings are analyzed using three autoencoder architectures -- Baseline Autoencoder (AE), Variational Autoencoder (VAE), and Denoising Autoencoder (DAE) -- to model normal process behavior and identify anomalies. The best-performing model is selected for comparison against traditional methods. The framework is evaluated on real-world, highly imbalanced provenance trace datasets from the DARPA Transparent Computing program, where APT-like attacks constitute as little as 0.004\% of the data across multiple operating systems (Android, Linux, BSD, and Windows) and attack scenarios. Results demonstrate that APT-LLM significantly improves detection performance under extreme imbalance conditions, outperforming existing anomaly detection methods and highlighting the effectiveness of LLM-based feature extraction in cybersecurity.

摘要: 高级持续性威胁(APT)由于其隐蔽性和模仿正常系统行为的能力，构成了重大的网络安全挑战，使得在高度不平衡的数据集中检测尤其困难。传统的异常检测方法难以有效地区分与APT相关的活动和良性过程，限制了它们在现实世界场景中的适用性。介绍了一种新的基于嵌入式的异常检测框架APT-LLM，该框架集成了大型语言模型(LLM)--BERT、ALBERT、DistilBERT和Roberta--以及自动编码器体系结构来检测APT。与之前依赖于手动设计的特征或传统异常检测模型的方法不同，APT-LLM利用LLM将流程操作起源跟踪编码到语义丰富的嵌入中，从而捕获细微差别的行为模式。使用三种自动编码器体系结构--基线自动编码器(AE)、变分自动编码器(VAE)和去噪自动编码器(DAE)--对这些嵌入进行分析，以模拟正常进程行为并识别异常。选择性能最好的模型与传统方法进行比较。该框架是在来自DARPA透明计算计划的高度不平衡的来源跟踪数据集上进行评估的，其中APT类攻击在多个操作系统(安卓、LINUX、BSD和Windows)和攻击场景中仅占数据的0.004\%。结果表明，APT-LLM显著提高了极端不平衡条件下的检测性能，优于现有的异常检测方法，突出了基于LLM的特征提取在网络安全中的有效性。



## **2. Privacy Checklist: Privacy Violation Detection Grounding on Contextual Integrity Theory**

隐私检查表：基于上下文完整性理论的隐私侵犯检测 cs.CL

To appear at NAACL 25

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2408.10053v2) [paper-pdf](http://arxiv.org/pdf/2408.10053v2)

**Authors**: Haoran Li, Wei Fan, Yulin Chen, Jiayang Cheng, Tianshu Chu, Xuebing Zhou, Peizhao Hu, Yangqiu Song

**Abstract**: Privacy research has attracted wide attention as individuals worry that their private data can be easily leaked during interactions with smart devices, social platforms, and AI applications. Computer science researchers, on the other hand, commonly study privacy issues through privacy attacks and defenses on segmented fields. Privacy research is conducted on various sub-fields, including Computer Vision (CV), Natural Language Processing (NLP), and Computer Networks. Within each field, privacy has its own formulation. Though pioneering works on attacks and defenses reveal sensitive privacy issues, they are narrowly trapped and cannot fully cover people's actual privacy concerns. Consequently, the research on general and human-centric privacy research remains rather unexplored. In this paper, we formulate the privacy issue as a reasoning problem rather than simple pattern matching. We ground on the Contextual Integrity (CI) theory which posits that people's perceptions of privacy are highly correlated with the corresponding social context. Based on such an assumption, we develop the first comprehensive checklist that covers social identities, private attributes, and existing privacy regulations. Unlike prior works on CI that either cover limited expert annotated norms or model incomplete social context, our proposed privacy checklist uses the whole Health Insurance Portability and Accountability Act of 1996 (HIPAA) as an example, to show that we can resort to large language models (LLMs) to completely cover the HIPAA's regulations. Additionally, our checklist also gathers expert annotations across multiple ontologies to determine private information including but not limited to personally identifiable information (PII). We use our preliminary results on the HIPAA to shed light on future context-centric privacy research to cover more privacy regulations, social norms and standards.

摘要: 隐私研究吸引了广泛的关注，因为个人担心他们的私人数据在与智能设备、社交平台和人工智能应用程序交互时很容易被泄露。另一方面，计算机科学研究人员通常通过对分割的领域进行隐私攻击和防御来研究隐私问题。隐私研究在不同的子领域进行，包括计算机视觉(CV)、自然语言处理(NLP)和计算机网络。在每个领域，隐私都有自己的表述。尽管攻击和防御方面的开创性作品揭示了敏感的隐私问题，但它们被狭隘地困住了，不能完全覆盖人们对隐私的实际担忧。因此，关于一般隐私研究和以人为中心的隐私研究仍然是相当未被探索的。在本文中，我们将隐私问题描述为一个推理问题，而不是简单的模式匹配。我们基于语境完整性(CI)理论，该理论认为人们对隐私的感知与相应的社会语境高度相关。基于这样的假设，我们开发了第一个全面的清单，其中包括社会身份、私人属性和现有的隐私法规。与以往关于CI的工作要么涵盖有限的专家注释规范，要么涵盖不完整的社会背景，我们提出的隐私检查表以1996年的整个健康保险携带和责任法案(HIPAA)为例，表明我们可以求助于大型语言模型(LLM)来完全覆盖HIPAA的规定。此外，我们的检查表还收集了跨多个本体的专家注释，以确定私人信息，包括但不限于个人身份信息(PII)。我们使用我们在HIPAA上的初步结果来阐明未来以上下文为中心的隐私研究，以涵盖更多的隐私法规、社会规范和标准。



## **3. FLAME: Flexible LLM-Assisted Moderation Engine**

FLAME：灵活的LLM辅助审核引擎 cs.CR

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.09175v1) [paper-pdf](http://arxiv.org/pdf/2502.09175v1)

**Authors**: Ivan Bakulin, Ilia Kopanichuk, Iaroslav Bespalov, Nikita Radchenko, Vladimir Shaposhnikov, Dmitry Dylov, Ivan Oseledets

**Abstract**: The rapid advancement of Large Language Models (LLMs) has introduced significant challenges in moderating user-model interactions. While LLMs demonstrate remarkable capabilities, they remain vulnerable to adversarial attacks, particularly ``jailbreaking'' techniques that bypass content safety measures. Current content moderation systems, which primarily rely on input prompt filtering, have proven insufficient, with techniques like Best-of-N (BoN) jailbreaking achieving success rates of 80% or more against popular LLMs. In this paper, we introduce Flexible LLM-Assisted Moderation Engine (FLAME): a new approach that shifts the focus from input filtering to output moderation. Unlike traditional circuit-breaking methods that analyze user queries, FLAME evaluates model responses, offering several key advantages: (1) computational efficiency in both training and inference, (2) enhanced resistance to BoN jailbreaking attacks, and (3) flexibility in defining and updating safety criteria through customizable topic filtering. Our experiments demonstrate that FLAME significantly outperforms current moderation systems. For example, FLAME reduces attack success rate in GPT-4o-mini and DeepSeek-v3 by a factor of ~9, while maintaining low computational overhead. We provide comprehensive evaluation on various LLMs and analyze the engine's efficiency against the state-of-the-art jailbreaking. This work contributes to the development of more robust and adaptable content moderation systems for LLMs.

摘要: 大型语言模型(LLM)的快速发展给协调用户与模型的交互带来了巨大的挑战。虽然LLM显示出非凡的能力，但它们仍然容易受到对抗性攻击，特别是绕过内容安全措施的“越狱”技术。目前的内容审核系统主要依赖于输入提示过滤，已被证明是不够的，像N中最佳(Bon)越狱技术对流行的LLM的成功率达到80%或更高。在本文中，我们介绍了灵活的LLM辅助调节引擎(FLAME)：一种将焦点从输入过滤转移到输出调节的新方法。与分析用户查询的传统断路方法不同，FLAME评估模型响应，提供了几个关键优势：(1)训练和推理的计算效率，(2)增强了对Bon越狱攻击的抵抗，以及(3)通过可定制的主题过滤灵活地定义和更新安全标准。我们的实验表明，火焰系统的性能明显优于现有的慢化系统。例如，FLAME将GPT-40-mini和DeepSeek-v3中的攻击成功率降低了~9倍，同时保持了较低的计算开销。我们对各种LLM进行了综合评估，并针对最先进的越狱情况分析了发动机的效率。这项工作有助于开发更健壮和适应性更强的LLMS内容审核系统。



## **4. Universal Adversarial Attack on Aligned Multimodal LLMs**

对对齐多模式LLM的普遍对抗攻击 cs.AI

Added an affiliation

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.07987v2) [paper-pdf](http://arxiv.org/pdf/2502.07987v2)

**Authors**: Temurbek Rahmatullaev, Polina Druzhinina, Matvey Mikhalchuk, Andrey Kuznetsov, Anton Razzhigaev

**Abstract**: We propose a universal adversarial attack on multimodal Large Language Models (LLMs) that leverages a single optimized image to override alignment safeguards across diverse queries and even multiple models. By backpropagating through the vision encoder and language head, we craft a synthetic image that forces the model to respond with a targeted phrase (e.g., ''Sure, here it is'') or otherwise unsafe content-even for harmful prompts. In experiments on the SafeBench benchmark, our method achieves significantly higher attack success rates than existing baselines, including text-only universal prompts (e.g., up to 93% on certain models). We further demonstrate cross-model transferability by training on several multimodal LLMs simultaneously and testing on unseen architectures. Additionally, a multi-answer variant of our approach produces more natural-sounding (yet still malicious) responses. These findings underscore critical vulnerabilities in current multimodal alignment and call for more robust adversarial defenses. We will release code and datasets under the Apache-2.0 license. Warning: some content generated by Multimodal LLMs in this paper may be offensive to some readers.

摘要: 我们提出了一种针对多模式大型语言模型(LLMS)的通用对抗性攻击，该攻击利用单个优化图像覆盖跨不同查询甚至多个模型的对齐保障。通过视觉编码器和语言头部的反向传播，我们制作了一个合成图像，迫使模型使用有针对性的短语(例如，“当然，就是这里”)或其他不安全的内容做出响应--即使是有害的提示。在SafeBtch基准测试上的实验中，我们的方法获得了比现有基线显著更高的攻击成功率，包括纯文本通用提示(例如，在某些型号上高达93%)。我们通过同时在多个多模式LLM上进行训练和在看不见的体系结构上进行测试来进一步证明跨模型的可转移性。此外，我们的方法的一个多答案变体会产生听起来更自然(但仍然是恶意的)响应。这些发现突显了当前多模式联合的严重弱点，并呼吁进行更强大的对抗性防御。我们将在APACHE-2.0许可下发布代码和数据集。警告：本文中的多模式LLMS生成的某些内容可能会冒犯某些读者。



## **5. Vision-LLMs Can Fool Themselves with Self-Generated Typographic Attacks**

Vision-LLM可以通过自我生成的印刷攻击来欺骗自己 cs.CV

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2402.00626v3) [paper-pdf](http://arxiv.org/pdf/2402.00626v3)

**Authors**: Maan Qraitem, Nazia Tasnim, Piotr Teterwak, Kate Saenko, Bryan A. Plummer

**Abstract**: Typographic attacks, adding misleading text to images, can deceive vision-language models (LVLMs). The susceptibility of recent large LVLMs like GPT4-V to such attacks is understudied, raising concerns about amplified misinformation in personal assistant applications. Previous attacks use simple strategies, such as random misleading words, which don't fully exploit LVLMs' language reasoning abilities. We introduce an experimental setup for testing typographic attacks on LVLMs and propose two novel self-generated attacks: (1) Class-based attacks, where the model identifies a similar class to deceive itself, and (2) Reasoned attacks, where an advanced LVLM suggests an attack combining a deceiving class and description. Our experiments show these attacks significantly reduce classification performance by up to 60\% and are effective across different models, including InstructBLIP and MiniGPT4. Code: https://github.com/mqraitem/Self-Gen-Typo-Attack

摘要: 印刷攻击（向图像添加误导性文本）可以欺骗视觉语言模型（LVLM）。最近，GPT 4-V等大型LVLM对此类攻击的敏感性尚未得到充分研究，这引发了人们对个人助理应用程序中被放大的错误信息的担忧。之前的攻击使用简单的策略，例如随机误导性单词，这些策略没有充分利用LVLM的语言推理能力。我们引入了一个实验设置来测试对LVLM的印刷攻击，并提出了两种新颖的自发攻击：（1）基于类的攻击，其中模型识别类似的类来欺骗自己，和（2）推理攻击，其中高级LVLM建议结合欺骗类和描述的攻击。我们的实验表明，这些攻击将分类性能显着降低高达60%，并且在不同的模型中有效，包括DirectBLIP和MiniGPT 4。代码：https://github.com/mqraitem/Self-Gen-Typo-Attack



## **6. An Engorgio Prompt Makes Large Language Model Babble on**

Engorgio提示让大型语言模型胡言乱语 cs.CR

ICLR 2025

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2412.19394v2) [paper-pdf](http://arxiv.org/pdf/2412.19394v2)

**Authors**: Jianshuo Dong, Ziyuan Zhang, Qingjie Zhang, Tianwei Zhang, Hao Wang, Hewu Li, Qi Li, Chao Zhang, Ke Xu, Han Qiu

**Abstract**: Auto-regressive large language models (LLMs) have yielded impressive performance in many real-world tasks. However, the new paradigm of these LLMs also exposes novel threats. In this paper, we explore their vulnerability to inference cost attacks, where a malicious user crafts Engorgio prompts to intentionally increase the computation cost and latency of the inference process. We design Engorgio, a novel methodology, to efficiently generate adversarial Engorgio prompts to affect the target LLM's service availability. Engorgio has the following two technical contributions. (1) We employ a parameterized distribution to track LLMs' prediction trajectory. (2) Targeting the auto-regressive nature of LLMs' inference process, we propose novel loss functions to stably suppress the appearance of the <EOS> token, whose occurrence will interrupt the LLM's generation process. We conduct extensive experiments on 13 open-sourced LLMs with parameters ranging from 125M to 30B. The results show that Engorgio prompts can successfully induce LLMs to generate abnormally long outputs (i.e., roughly 2-13$\times$ longer to reach 90%+ of the output length limit) in a white-box scenario and our real-world experiment demonstrates Engergio's threat to LLM service with limited computing resources. The code is released at: https://github.com/jianshuod/Engorgio-prompt.

摘要: 自回归大型语言模型(LLM)在许多实际任务中取得了令人印象深刻的性能。然而，这些LLM的新范式也暴露了新的威胁。在本文中，我们探讨了它们对推理成本攻击的脆弱性，在这种攻击中，恶意用户手工制作Engorgio会提示故意增加推理过程的计算成本和延迟。我们设计了一种新的方法Engorgio来有效地生成对抗性Engorgio提示，以影响目标LLM的服务可用性。Engorgio有以下两个技术贡献。(1)采用一种参数分布来跟踪LLMS的预测轨迹。(2)针对LLMS推理过程的自回归特性，提出了一种新的损失函数来稳定地抑制<EOS>令牌的出现，它的出现会中断LLM的生成过程。我们在13个开源的LLM上进行了广泛的实验，参数从125M到30B不等。结果表明，在白盒场景中，Engorgio提示能够成功地诱导LLMS产生异常长的输出(即大约2-13$\x$以达到输出长度限制的90%以上)，并且我们的真实世界实验证明了Engergio在有限计算资源的情况下对LLM服务的威胁。代码发布地址为：https://github.com/jianshuod/Engorgio-prompt.



## **7. Theoretically Grounded Framework for LLM Watermarking: A Distribution-Adaptive Approach**

LLM水印的理论基础框架：分布自适应方法 cs.CR

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2410.02890v3) [paper-pdf](http://arxiv.org/pdf/2410.02890v3)

**Authors**: Haiyun He, Yepeng Liu, Ziqiao Wang, Yongyi Mao, Yuheng Bu

**Abstract**: Watermarking has emerged as a crucial method to distinguish AI-generated text from human-created text. In this paper, we present a novel theoretical framework for watermarking Large Language Models (LLMs) that jointly optimizes both the watermarking scheme and the detection process. Our approach focuses on maximizing detection performance while maintaining control over the worst-case Type-I error and text distortion. We characterize \emph{the universally minimum Type-II error}, showing a fundamental trade-off between watermark detectability and text distortion. Importantly, we identify that the optimal watermarking schemes are adaptive to the LLM generative distribution. Building on our theoretical insights, we propose an efficient, model-agnostic, distribution-adaptive watermarking algorithm, utilizing a surrogate model alongside the Gumbel-max trick. Experiments conducted on Llama2-13B and Mistral-8$\times$7B models confirm the effectiveness of our approach. Additionally, we examine incorporating robustness into our framework, paving a way to future watermarking systems that withstand adversarial attacks more effectively.

摘要: 数字水印已经成为区分人工智能生成的文本和人类创建的文本的关键方法。在本文中，我们提出了一种新的大语言模型(LLMS)水印理论框架，该框架同时优化了水印方案和检测过程。我们的方法专注于最大化检测性能，同时保持对最坏情况下的类型I错误和文本失真的控制。我们将其刻画在水印可检测性和文本失真之间的基本权衡。重要的是，我们发现最优水印方案对LLM生成分布是自适应的。基于我们的理论见解，我们提出了一种高效的、与模型无关的、分布自适应的水印算法，该算法利用代理模型和Gumbel-max技巧。在Llama2-13B和Mistral-8$\x$70亿模型上进行的实验证实了该方法的有效性。此外，我们还研究了将健壮性融入到我们的框架中，为未来更有效地抵御对手攻击的水印系统铺平了道路。



## **8. Commercial LLM Agents Are Already Vulnerable to Simple Yet Dangerous Attacks**

商业LLM代理已经容易受到简单但危险的攻击 cs.LG

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.08586v1) [paper-pdf](http://arxiv.org/pdf/2502.08586v1)

**Authors**: Ang Li, Yin Zhou, Vethavikashini Chithrra Raghuram, Tom Goldstein, Micah Goldblum

**Abstract**: A high volume of recent ML security literature focuses on attacks against aligned large language models (LLMs). These attacks may extract private information or coerce the model into producing harmful outputs. In real-world deployments, LLMs are often part of a larger agentic pipeline including memory systems, retrieval, web access, and API calling. Such additional components introduce vulnerabilities that make these LLM-powered agents much easier to attack than isolated LLMs, yet relatively little work focuses on the security of LLM agents. In this paper, we analyze security and privacy vulnerabilities that are unique to LLM agents. We first provide a taxonomy of attacks categorized by threat actors, objectives, entry points, attacker observability, attack strategies, and inherent vulnerabilities of agent pipelines. We then conduct a series of illustrative attacks on popular open-source and commercial agents, demonstrating the immediate practical implications of their vulnerabilities. Notably, our attacks are trivial to implement and require no understanding of machine learning.

摘要: 最近有大量的ML安全文献关注针对对齐的大型语言模型(LLM)的攻击。这些攻击可能会窃取私人信息或迫使模型产生有害的输出。在实际部署中，LLM通常是更大的代理管道的一部分，包括内存系统、检索、Web访问和API调用。这些额外的组件引入了漏洞，使这些由LLM支持的代理比孤立的LLM更容易受到攻击，但相对较少的工作关注LLM代理的安全。在本文中，我们分析了LLM代理独有的安全和隐私漏洞。我们首先根据威胁参与者、目标、入口点、攻击者的可观察性、攻击策略和代理管道的固有漏洞对攻击进行分类。然后，我们对流行的开源和商业代理进行了一系列说明性攻击，展示了它们漏洞的直接实际影响。值得注意的是，我们的攻击很容易实现，并且不需要理解机器学习。



## **9. Why Are My Prompts Leaked? Unraveling Prompt Extraction Threats in Customized Large Language Models**

为什么我的笔记会泄露？解开定制大型语言模型中的提示提取威胁 cs.CL

Source Code: https://github.com/liangzid/PromptExtractionEval

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2408.02416v2) [paper-pdf](http://arxiv.org/pdf/2408.02416v2)

**Authors**: Zi Liang, Haibo Hu, Qingqing Ye, Yaxin Xiao, Haoyang Li

**Abstract**: The drastic increase of large language models' (LLMs) parameters has led to a new research direction of fine-tuning-free downstream customization by prompts, i.e., task descriptions. While these prompt-based services (e.g. OpenAI's GPTs) play an important role in many businesses, there has emerged growing concerns about the prompt leakage, which undermines the intellectual properties of these services and causes downstream attacks. In this paper, we analyze the underlying mechanism of prompt leakage, which we refer to as prompt memorization, and develop corresponding defending strategies. By exploring the scaling laws in prompt extraction, we analyze key attributes that influence prompt extraction, including model sizes, prompt lengths, as well as the types of prompts. Then we propose two hypotheses that explain how LLMs expose their prompts. The first is attributed to the perplexity, i.e. the familiarity of LLMs to texts, whereas the second is based on the straightforward token translation path in attention matrices. To defend against such threats, we investigate whether alignments can undermine the extraction of prompts. We find that current LLMs, even those with safety alignments like GPT-4, are highly vulnerable to prompt extraction attacks, even under the most straightforward user attacks. Therefore, we put forward several defense strategies with the inspiration of our findings, which achieve 83.8\% and 71.0\% drop in the prompt extraction rate for Llama2-7B and GPT-3.5, respectively. Source code is avaliable at https://github.com/liangzid/PromptExtractionEval.

摘要: 大型语言模型(LLMS)参数的急剧增加导致了一个新的研究方向，即通过提示(即任务描述)进行免微调的下游定制。虽然这些基于提示的服务(例如OpenAI的GPT)在许多业务中扮演着重要的角色，但人们越来越担心即时泄露，这会破坏这些服务的知识产权，并导致下游攻击。本文分析了即时记忆的潜在机制，并提出了相应的防御策略。通过研究提示提取中的缩放规律，我们分析了影响提示提取的关键属性，包括模型大小、提示长度以及提示的类型。然后，我们提出了两个假设来解释LLM是如何暴露他们的提示的。第一种归因于迷惑性，即LLMS对文本的熟悉度，而第二种归因于注意矩阵中直接的表征翻译路径。为了防御此类威胁，我们调查对齐是否会破坏提示符的提取。我们发现，即使在最直接的用户攻击下，当前的LLM，即使是那些具有GPT-4等安全对齐的LLM，也非常容易受到即时提取攻击。因此，我们根据研究结果提出了几种防御策略，分别使Llama2-7B和GPT-3.5的即时抽取率下降了83.8%和71.0%。源代码可在https://github.com/liangzid/PromptExtractionEval.上获得



## **10. Modification and Generated-Text Detection: Achieving Dual Detection Capabilities for the Outputs of LLM by Watermark**

修改和生成文本检测：通过水印实现LLM输出的双重检测能力 cs.CR

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.08332v1) [paper-pdf](http://arxiv.org/pdf/2502.08332v1)

**Authors**: Yuhang Cai, Yaofei Wang, Donghui Hu, Gu Chen

**Abstract**: The development of large language models (LLMs) has raised concerns about potential misuse. One practical solution is to embed a watermark in the text, allowing ownership verification through watermark extraction. Existing methods primarily focus on defending against modification attacks, often neglecting other spoofing attacks. For example, attackers can alter the watermarked text to produce harmful content without compromising the presence of the watermark, which could lead to false attribution of this malicious content to the LLM. This situation poses a serious threat to the LLMs service providers and highlights the significance of achieving modification detection and generated-text detection simultaneously. Therefore, we propose a technique to detect modifications in text for unbiased watermark which is sensitive to modification. We introduce a new metric called ``discarded tokens", which measures the number of tokens not included in watermark detection. When a modification occurs, this metric changes and can serve as evidence of the modification. Additionally, we improve the watermark detection process and introduce a novel method for unbiased watermark. Our experiments demonstrate that we can achieve effective dual detection capabilities: modification detection and generated-text detection by watermark.

摘要: 大型语言模型(LLM)的发展引起了人们对潜在滥用的担忧。一种实用的解决方案是在文本中嵌入水印，允许通过提取水印来验证所有权。现有的方法主要集中在防御修改攻击上，往往忽略了其他欺骗攻击。例如，攻击者可以更改带水印的文本以产生有害内容，而不会影响水印的存在，这可能会导致将此恶意内容错误地归因于LLM。这种情况对LLMS服务提供商构成了严重威胁，并突出了同时实现修改检测和生成文本检测的重要性。因此，我们提出了一种文本修改检测技术，以检测对修改敏感的无偏水印。提出了一种新的水印检测方法--“丢弃令牌”，该度量度量了水印检测中未包含的令牌个数。当水印发生修改时，该度量会发生变化，并且可以作为修改的证据。此外，我们对水印检测过程进行了改进，提出了一种新的无偏水印检测方法。实验表明，我们可以实现有效的双重检测能力：修改检测和水印生成文本检测。



## **11. Compromising Honesty and Harmlessness in Language Models via Deception Attacks**

通过欺骗攻击损害语言模型中的诚实和无害 cs.CL

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.08301v1) [paper-pdf](http://arxiv.org/pdf/2502.08301v1)

**Authors**: Laurène Vaugrante, Francesca Carlon, Maluna Menke, Thilo Hagendorff

**Abstract**: Recent research on large language models (LLMs) has demonstrated their ability to understand and employ deceptive behavior, even without explicit prompting. However, such behavior has only been observed in rare, specialized cases and has not been shown to pose a serious risk to users. Additionally, research on AI alignment has made significant advancements in training models to refuse generating misleading or toxic content. As a result, LLMs generally became honest and harmless. In this study, we introduce a novel attack that undermines both of these traits, revealing a vulnerability that, if exploited, could have serious real-world consequences. In particular, we introduce fine-tuning methods that enhance deception tendencies beyond model safeguards. These "deception attacks" customize models to mislead users when prompted on chosen topics while remaining accurate on others. Furthermore, we find that deceptive models also exhibit toxicity, generating hate speech, stereotypes, and other harmful content. Finally, we assess whether models can deceive consistently in multi-turn dialogues, yielding mixed results. Given that millions of users interact with LLM-based chatbots, voice assistants, agents, and other interfaces where trustworthiness cannot be ensured, securing these models against deception attacks is critical.

摘要: 最近对大型语言模型的研究表明，即使在没有明确提示的情况下，它们也能够理解和使用欺骗性行为。然而，这种行为只在罕见的特殊情况下才能观察到，并未被证明对用户构成严重风险。此外，人工智能对齐的研究在拒绝产生误导性或有毒内容的训练模型方面取得了重大进展。结果，LLM通常变得诚实和无害。在这项研究中，我们介绍了一种破坏这两个特征的新型攻击，揭示了一个漏洞，如果利用该漏洞，可能会在现实世界中产生严重后果。特别是，我们引入了微调方法，增强了模型保障之外的欺骗倾向。这些“欺骗攻击”定制模型，在提示用户选择主题时误导用户，而在其他主题上保持准确。此外，我们发现欺骗性模型也表现出毒性，产生仇恨言论、刻板印象和其他有害内容。最后，我们评估模型是否可以在多轮对话中一致地欺骗，产生喜忧参半的结果。鉴于数以百万计的用户与基于LLM的聊天机器人、语音助理、代理和其他无法确保可信度的界面交互，确保这些模型免受欺骗性攻击至关重要。



## **12. Typographic Attacks in a Multi-Image Setting**

多图像环境中的印刷攻击 cs.CR

Accepted by NAACL2025. Our code is available at  https://github.com/XiaomengWang-AI/Typographic-Attacks-in-a-Multi-Image-Setting

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.08193v1) [paper-pdf](http://arxiv.org/pdf/2502.08193v1)

**Authors**: Xiaomeng Wang, Zhengyu Zhao, Martha Larson

**Abstract**: Large Vision-Language Models (LVLMs) are susceptible to typographic attacks, which are misclassifications caused by an attack text that is added to an image. In this paper, we introduce a multi-image setting for studying typographic attacks, broadening the current emphasis of the literature on attacking individual images. Specifically, our focus is on attacking image sets without repeating the attack query. Such non-repeating attacks are stealthier, as they are more likely to evade a gatekeeper than attacks that repeat the same attack text. We introduce two attack strategies for the multi-image setting, leveraging the difficulty of the target image, the strength of the attack text, and text-image similarity. Our text-image similarity approach improves attack success rates by 21% over random, non-specific methods on the CLIP model using ImageNet while maintaining stealth in a multi-image scenario. An additional experiment demonstrates transferability, i.e., text-image similarity calculated using CLIP transfers when attacking InstructBLIP.

摘要: 大型视觉语言模型(LVLM)容易受到排版攻击，这些攻击是由添加到图像中的攻击文本引起的错误分类。在这篇文章中，我们介绍了一种研究排版攻击的多图像环境，拓宽了当前文献对单个图像攻击的重点。具体地说，我们的重点是攻击图像集，而不重复攻击查询。这种不重复的攻击更隐蔽，因为它们比重复相同攻击文本的攻击更有可能避开网守。针对多图像环境，我们提出了两种攻击策略，分别利用了目标图像的难度、攻击文本的强度和文本与图像的相似性。与使用ImageNet的剪辑模型上的随机、非特定方法相比，我们的文本-图像相似性方法将攻击成功率提高了21%，同时在多图像场景中保持了隐蔽性。另外一个实验演示了可转移性，即攻击InstructBLIP时使用片段传输计算的文本-图像相似度。



## **13. The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems**

早起的鸟抓住漏洞：揭开LLM服务系统中的计时侧通道 cs.CR

This work was submitted for review on Sept. 5, 2024, and the initial  version was uploaded to Arxiv on Sept. 30, 2024. The latest version reflects  the up-to-date experimental results

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2409.20002v3) [paper-pdf](http://arxiv.org/pdf/2409.20002v3)

**Authors**: Linke Song, Zixuan Pang, Wenhao Wang, Zihao Wang, XiaoFeng Wang, Hongbo Chen, Wei Song, Yier Jin, Dan Meng, Rui Hou

**Abstract**: The wide deployment of Large Language Models (LLMs) has given rise to strong demands for optimizing their inference performance. Today's techniques serving this purpose primarily focus on reducing latency and improving throughput through algorithmic and hardware enhancements, while largely overlooking their privacy side effects, particularly in a multi-user environment. In our research, for the first time, we discovered a set of new timing side channels in LLM systems, arising from shared caches and GPU memory allocations, which can be exploited to infer both confidential system prompts and those issued by other users. These vulnerabilities echo security challenges observed in traditional computing systems, highlighting an urgent need to address potential information leakage in LLM serving infrastructures. In this paper, we report novel attack strategies designed to exploit such timing side channels inherent in LLM deployments, specifically targeting the Key-Value (KV) cache and semantic cache widely used to enhance LLM inference performance. Our approach leverages timing measurements and classification models to detect cache hits, allowing an adversary to infer private prompts with high accuracy. We also propose a token-by-token search algorithm to efficiently recover shared prompt prefixes in the caches, showing the feasibility of stealing system prompts and those produced by peer users. Our experimental studies on black-box testing of popular online LLM services demonstrate that such privacy risks are completely realistic, with significant consequences. Our findings underscore the need for robust mitigation to protect LLM systems against such emerging threats.

摘要: 大型语言模型的广泛应用对其推理性能的优化提出了强烈的需求。目前用于此目的的技术主要集中在通过算法和硬件增强来减少延迟和提高吞吐量，而在很大程度上忽略了它们的隐私副作用，特别是在多用户环境中。在我们的研究中，我们首次在LLM系统中发现了一组新的计时侧通道，这些通道来自共享缓存和GPU内存分配，可以用来推断机密系统提示和其他用户发出的提示。这些漏洞呼应了在传统计算系统中观察到的安全挑战，突显出迫切需要解决LLM服务基础设施中潜在的信息泄漏问题。在本文中，我们报告了一种新的攻击策略，旨在利用LLM部署中固有的计时侧通道，特别是针对广泛用于提高LLM推理性能的键值(KV)缓存和语义缓存。我们的方法利用计时测量和分类模型来检测缓存命中，允许对手以高精度推断私人提示。我们还提出了一种逐令牌搜索算法来高效地恢复缓存中的共享提示前缀，证明了窃取系统提示和对等用户产生的提示的可行性。我们对流行的在线LLM服务进行黑盒测试的实验研究表明，这种隐私风险是完全现实的，具有显著的后果。我们的发现强调了强有力的缓解措施的必要性，以保护LLM系统免受此类新出现的威胁。



## **14. In-Context Experience Replay Facilitates Safety Red-Teaming of Text-to-Image Diffusion Models**

上下文体验回放促进文本到图像扩散模型的安全红色团队化 cs.LG

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2411.16769v2) [paper-pdf](http://arxiv.org/pdf/2411.16769v2)

**Authors**: Zhi-Yi Chin, Mario Fritz, Pin-Yu Chen, Wei-Chen Chiu

**Abstract**: Text-to-image (T2I) models have shown remarkable progress, but their potential to generate harmful content remains a critical concern in the ML community. While various safety mechanisms have been developed, the field lacks systematic tools for evaluating their effectiveness against real-world misuse scenarios. In this work, we propose ICER, a novel red-teaming framework that leverages Large Language Models (LLMs) and a bandit optimization-based algorithm to generate interpretable and semantic meaningful problematic prompts by learning from past successful red-teaming attempts. Our ICER efficiently probes safety mechanisms across different T2I models without requiring internal access or additional training, making it broadly applicable to deployed systems. Through extensive experiments, we demonstrate that ICER significantly outperforms existing prompt attack methods in identifying model vulnerabilities while maintaining high semantic similarity with intended content. By uncovering that successful jailbreaking instances can systematically facilitate the discovery of new vulnerabilities, our work provides crucial insights for developing more robust safety mechanisms in T2I systems.

摘要: 文本到图像(T2I)模型已经显示出显著的进步，但它们产生有害内容的潜力仍然是ML社区的一个关键问题。虽然已经开发了各种安全机制，但该领域缺乏系统的工具来评估其针对现实世界滥用情况的有效性。在这项工作中，我们提出了ICER，一个新的红团队框架，它利用大型语言模型(LLM)和基于Bandit优化的算法来生成可解释的、有语义意义的问题提示，通过学习过去成功的红团队尝试。我们的ICER可有效探测不同T2I型号的安全机制，无需内部访问或额外培训，使其广泛适用于已部署的系统。通过大量的实验，我们证明了ICER在识别模型漏洞方面明显优于现有的即时攻击方法，同时保持了与预期内容的高度语义相似度。通过揭示成功的越狱实例可以系统地促进新漏洞的发现，我们的工作为在T2I系统中开发更强大的安全机制提供了至关重要的见解。



## **15. Safety at Scale: A Comprehensive Survey of Large Model Safety**

大规模安全性：大型车型安全性全面调查 cs.CR

47 pages, 3 figures, 11 tables GitHub:  https://github.com/xingjunm/Awesome-Large-Model-Safety

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.05206v2) [paper-pdf](http://arxiv.org/pdf/2502.05206v2)

**Authors**: Xingjun Ma, Yifeng Gao, Yixu Wang, Ruofan Wang, Xin Wang, Ye Sun, Yifan Ding, Hengyuan Xu, Yunhao Chen, Yunhan Zhao, Hanxun Huang, Yige Li, Jiaming Zhang, Xiang Zheng, Yang Bai, Zuxuan Wu, Xipeng Qiu, Jingfeng Zhang, Yiming Li, Jun Sun, Cong Wang, Jindong Gu, Baoyuan Wu, Siheng Chen, Tianwei Zhang, Yang Liu, Mingming Gong, Tongliang Liu, Shirui Pan, Cihang Xie, Tianyu Pang, Yinpeng Dong, Ruoxi Jia, Yang Zhang, Shiqing Ma, Xiangyu Zhang, Neil Gong, Chaowei Xiao, Sarah Erfani, Bo Li, Masashi Sugiyama, Dacheng Tao, James Bailey, Yu-Gang Jiang

**Abstract**: The rapid advancement of large models, driven by their exceptional abilities in learning and generalization through large-scale pre-training, has reshaped the landscape of Artificial Intelligence (AI). These models are now foundational to a wide range of applications, including conversational AI, recommendation systems, autonomous driving, content generation, medical diagnostics, and scientific discovery. However, their widespread deployment also exposes them to significant safety risks, raising concerns about robustness, reliability, and ethical implications. This survey provides a systematic review of current safety research on large models, covering Vision Foundation Models (VFMs), Large Language Models (LLMs), Vision-Language Pre-training (VLP) models, Vision-Language Models (VLMs), Diffusion Models (DMs), and large-model-based Agents. Our contributions are summarized as follows: (1) We present a comprehensive taxonomy of safety threats to these models, including adversarial attacks, data poisoning, backdoor attacks, jailbreak and prompt injection attacks, energy-latency attacks, data and model extraction attacks, and emerging agent-specific threats. (2) We review defense strategies proposed for each type of attacks if available and summarize the commonly used datasets and benchmarks for safety research. (3) Building on this, we identify and discuss the open challenges in large model safety, emphasizing the need for comprehensive safety evaluations, scalable and effective defense mechanisms, and sustainable data practices. More importantly, we highlight the necessity of collective efforts from the research community and international collaboration. Our work can serve as a useful reference for researchers and practitioners, fostering the ongoing development of comprehensive defense systems and platforms to safeguard AI models.

摘要: 大型模型的快速发展，受到其通过大规模预训练而具有的非凡学习和泛化能力的推动，重塑了人工智能(AI)的版图。这些模型现在是广泛应用的基础，包括对话式人工智能、推荐系统、自动驾驶、内容生成、医疗诊断和科学发现。然而，它们的广泛部署也使它们面临重大的安全风险，引发了人们对健壮性、可靠性和道德影响的担忧。本调查系统地回顾了当前关于大模型的安全研究，包括视觉基础模型(VFM)、大语言模型(LLMS)、视觉语言预训练(VLP)模型、视觉语言模型(VLMS)、扩散模型(DM)和基于大模型的代理。我们的工作总结如下：(1)对这些模型的安全威胁进行了全面的分类，包括对抗性攻击、数据中毒、后门攻击、越狱和快速注入攻击、能量延迟攻击、数据和模型提取攻击以及新出现的特定于代理的威胁。(2)我们回顾了针对每种攻击类型提出的防御策略(如果可用)，并总结了安全研究常用的数据集和基准。(3)在此基础上，我们确定并讨论了大型模型安全方面的开放挑战，强调需要全面的安全评估、可扩展和有效的防御机制以及可持续的数据实践。更重要的是，我们强调了研究界和国际合作集体努力的必要性。我们的工作可以作为研究人员和从业者的有用参考，促进正在进行的全面防御系统和平台的开发，以保护人工智能模型。



## **16. SymGPT: Auditing Smart Contracts via Combining Symbolic Execution with Large Language Models**

SymbGPT：通过将符号执行与大型语言模型相结合来审计智能合同 cs.AI

16 pages. arXiv admin note: text overlap with arXiv:2404.04306

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.07644v2) [paper-pdf](http://arxiv.org/pdf/2502.07644v2)

**Authors**: Shihao Xia, Mengting He, Shuai Shao, Tingting Yu, Yiying Zhang, Linhai Song

**Abstract**: To govern smart contracts running on Ethereum, multiple Ethereum Request for Comment (ERC) standards have been developed, each having a set of rules to guide the behaviors of smart contracts. Violating the ERC rules could cause serious security issues and financial loss, signifying the importance of verifying smart contracts follow ERCs. Today's practices of such verification are to manually audit each single contract, use expert-developed program-analysis tools, or use large language models (LLMs), all of which are far from effective in identifying ERC rule violations. This paper introduces SymGPT, a tool that combines the natural language understanding of large language models (LLMs) with the formal guarantees of symbolic execution to automatically verify smart contracts' compliance with ERC rules. To develop SymGPT, we conduct an empirical study of 132 ERC rules from three widely used ERC standards, examining their content, security implications, and natural language descriptions. Based on this study, we design SymGPT by first instructing an LLM to translate ERC rules into a defined EBNF grammar. We then synthesize constraints from the formalized rules to represent scenarios where violations may occur and use symbolic execution to detect them. Our evaluation shows that SymGPT identifies 5,783 ERC rule violations in 4,000 real-world contracts, including 1,375 violations with clear attack paths for stealing financial assets, demonstrating its effectiveness. Furthermore, SymGPT outperforms six automated techniques and a security-expert auditing service, underscoring its superiority over current smart contract analysis methods.

摘要: 为了管理在以太上运行的智能合同，已经开发了多个以太征求意见(ERC)标准，每个标准都有一套规则来指导智能合同的行为。违反ERC规则可能会导致严重的安全问题和经济损失，这意味着验证智能合同遵循ERC的重要性。如今，这种验证的做法是手动审计每一份合同，使用专家开发的程序分析工具，或者使用大型语言模型(LLM)，所有这些都远远不能有效地识别违反ERC规则的行为。本文介绍了一种将大型语言模型的自然语言理解与符号执行的形式保证相结合的工具--SymGPT，用于自动验证智能合约是否符合ERC规则。为了开发SymGPT，我们对来自三个广泛使用的ERC标准的132条ERC规则进行了实证研究，检查了它们的内容、安全含义和自然语言描述。在此研究的基础上，我们首先通过指示LLM将ERC规则转换为定义的EBNF语法来设计SymGPT。然后，我们从形式化的规则中合成约束来表示可能发生违规的场景，并使用符号执行来检测它们。我们的评估显示，SymGPT在4,000份真实合同中识别了5,783项违反ERC规则的行为，其中1,375项违规行为具有明确的窃取金融资产的攻击路径，证明了其有效性。此外，SymGPT的性能超过了六项自动化技术和安全专家审计服务，突显了其相对于当前智能合同分析方法的优势。



## **17. Layer-Level Self-Exposure and Patch: Affirmative Token Mitigation for Jailbreak Attack Defense**

分层自我暴露和补丁：越狱攻击防御的肯定代币缓解 cs.CR

14 pages, 4 figures, conference

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2501.02629v2) [paper-pdf](http://arxiv.org/pdf/2501.02629v2)

**Authors**: Yang Ouyang, Hengrui Gu, Shuhang Lin, Wenyue Hua, Jie Peng, Bhavya Kailkhura, Meijun Gao, Tianlong Chen, Kaixiong Zhou

**Abstract**: As large language models (LLMs) are increasingly deployed in diverse applications, including chatbot assistants and code generation, aligning their behavior with safety and ethical standards has become paramount. However, jailbreak attacks, which exploit vulnerabilities to elicit unintended or harmful outputs, threaten LLMs' safety significantly. In this paper, we introduce Layer-AdvPatcher, a novel methodology designed to defend against jailbreak attacks by utilizing an unlearning strategy to patch specific layers within LLMs through self-augmented datasets. Our insight is that certain layer(s), tend to produce affirmative tokens when faced with harmful prompts. By identifying these layers and adversarially exposing them to generate more harmful data, one can understand their inherent and diverse vulnerabilities to attacks. With these exposures, we then "unlearn" these issues, reducing the impact of affirmative tokens and hence minimizing jailbreak risks while keeping the model's responses to safe queries intact. We conduct extensive experiments on two models, four benchmark datasets, and multiple state-of-the-art jailbreak attacks to demonstrate the efficacy of our approach. Results indicate that our framework reduces the harmfulness and attack success rate of jailbreak attacks without compromising utility for benign queries compared to recent defense methods. Our code is publicly available at: https://github.com/oyy2000/LayerAdvPatcher

摘要: 随着大型语言模型(LLM)越来越多地部署在各种应用中，包括聊天机器人助手和代码生成，使它们的行为符合安全和道德标准变得至关重要。然而，越狱攻击利用漏洞来引发意外或有害的输出，严重威胁到LLMS的安全。在本文中，我们介绍了Layer-AdvPatcher，这是一种新的方法，旨在通过一种遗忘策略来通过自增强数据集修补LLMS中的特定层来防御越狱攻击。我们的洞察是，某些层面(S)，在面对有害的提示时，往往会产生肯定的表征。通过识别这些层并恶意暴露它们以生成更多有害数据，人们可以了解它们固有的和不同的攻击漏洞。有了这些暴露，我们就可以“忘掉”这些问题，减少肯定令牌的影响，从而最大限度地减少越狱风险，同时保持模型对安全查询的响应完好无损。我们在两个模型、四个基准数据集和多个最先进的越狱攻击上进行了广泛的实验，以证明我们的方法的有效性。结果表明，与现有的防御方法相比，该框架降低了越狱攻击的危害性和攻击成功率，而不影响良性查询的有效性。我们的代码在以下网址公开提供：https://github.com/oyy2000/LayerAdvPatcher



## **18. MAA: Meticulous Adversarial Attack against Vision-Language Pre-trained Models**

MAA：针对视觉语言预训练模型的强力对抗攻击 cs.CV

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.08079v1) [paper-pdf](http://arxiv.org/pdf/2502.08079v1)

**Authors**: Peng-Fei Zhang, Guangdong Bai, Zi Huang

**Abstract**: Current adversarial attacks for evaluating the robustness of vision-language pre-trained (VLP) models in multi-modal tasks suffer from limited transferability, where attacks crafted for a specific model often struggle to generalize effectively across different models, limiting their utility in assessing robustness more broadly. This is mainly attributed to the over-reliance on model-specific features and regions, particularly in the image modality. In this paper, we propose an elegant yet highly effective method termed Meticulous Adversarial Attack (MAA) to fully exploit model-independent characteristics and vulnerabilities of individual samples, achieving enhanced generalizability and reduced model dependence. MAA emphasizes fine-grained optimization of adversarial images by developing a novel resizing and sliding crop (RScrop) technique, incorporating a multi-granularity similarity disruption (MGSD) strategy. Extensive experiments across diverse VLP models, multiple benchmark datasets, and a variety of downstream tasks demonstrate that MAA significantly enhances the effectiveness and transferability of adversarial attacks. A large cohort of performance studies is conducted to generate insights into the effectiveness of various model configurations, guiding future advancements in this domain.

摘要: 当前用于评估视觉语言预训练(VLP)模型在多模式任务中的稳健性的对抗性攻击存在可转移性有限的问题，其中针对特定模型的攻击往往难以在不同的模型上有效地泛化，从而限制了它们在更广泛地评估稳健性方面的有效性。这主要归因于过度依赖特定型号的特征和区域，特别是在图像模式方面。在本文中，我们提出了一种优雅而高效的方法，称为精细攻击(MAA)，它充分利用了个体样本的模型无关特性和脆弱性，从而增强了泛化能力，降低了模型依赖。MAA通过开发一种新的调整大小和滑动裁剪(RSCrop)技术，结合多粒度相似破坏(MGSD)策略，强调对抗性图像的细粒度优化。在不同的VLP模型、多个基准数据集和各种下游任务上的广泛实验表明，MAA显著增强了对抗性攻击的有效性和可转移性。我们进行了大量的性能研究，以深入了解各种型号配置的有效性，从而指导该领域的未来发展。



## **19. Efficient LLM Jailbreak via Adaptive Dense-to-sparse Constrained Optimization**

通过自适应密集到稀疏约束优化实现高效LLM越狱 cs.LG

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2405.09113v2) [paper-pdf](http://arxiv.org/pdf/2405.09113v2)

**Authors**: Kai Hu, Weichen Yu, Yining Li, Kai Chen, Tianjun Yao, Xiang Li, Wenhe Liu, Lijun Yu, Zhiqiang Shen, Matt Fredrikson

**Abstract**: Recent research indicates that large language models (LLMs) are susceptible to jailbreaking attacks that can generate harmful content. This paper introduces a novel token-level attack method, Adaptive Dense-to-Sparse Constrained Optimization (ADC), which has been shown to successfully jailbreak multiple open-source LLMs. Drawing inspiration from the difficulties of discrete token optimization, our method relaxes the discrete jailbreak optimization into a continuous optimization process while gradually increasing the sparsity of the optimizing vectors. This technique effectively bridges the gap between discrete and continuous space optimization. Experimental results demonstrate that our method is more effective and efficient than state-of-the-art token-level methods. On Harmbench, our approach achieves the highest attack success rate on seven out of eight LLMs compared to the latest jailbreak methods. Trigger Warning: This paper contains model behavior that can be offensive in nature.

摘要: 最近的研究表明，大型语言模型（LLM）很容易受到可能生成有害内容的越狱攻击。本文介绍了一种新颖的代币级攻击方法--自适应密度到稀疏约束优化（ADC），该方法已被证明可以成功越狱多个开源LLM。我们的方法从离散代币优化的困难中汲取灵感，将离散越狱优化放宽为连续优化过程，同时逐渐增加优化载体的稀疏性。该技术有效地弥合了离散和连续空间优化之间的差距。实验结果表明，我们的方法比最先进的代币级方法更有效和高效。与最新的越狱方法相比，在Harmbridge上，我们的方法对八个LLM中的七个实现了最高的攻击成功率。触发警告：本文包含本质上可能具有冒犯性的模型行为。



## **20. DeepSeek on a Trip: Inducing Targeted Visual Hallucinations via Representation Vulnerabilities**

DeepSeek旅行中：通过表现漏洞引发有针对性的视觉幻觉 cs.CV

19 pages, 4 figures

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07905v1) [paper-pdf](http://arxiv.org/pdf/2502.07905v1)

**Authors**: Chashi Mahiul Islam, Samuel Jacob Chacko, Preston Horne, Xiuwen Liu

**Abstract**: Multimodal Large Language Models (MLLMs) represent the cutting edge of AI technology, with DeepSeek models emerging as a leading open-source alternative offering competitive performance to closed-source systems. While these models demonstrate remarkable capabilities, their vision-language integration mechanisms introduce specific vulnerabilities. We implement an adapted embedding manipulation attack on DeepSeek Janus that induces targeted visual hallucinations through systematic optimization of image embeddings. Through extensive experimentation across COCO, DALL-E 3, and SVIT datasets, we achieve hallucination rates of up to 98.0% while maintaining high visual fidelity (SSIM > 0.88) of the manipulated images on open-ended questions. Our analysis demonstrates that both 1B and 7B variants of DeepSeek Janus are susceptible to these attacks, with closed-form evaluation showing consistently higher hallucination rates compared to open-ended questioning. We introduce a novel multi-prompt hallucination detection framework using LLaMA-3.1 8B Instruct for robust evaluation. The implications of these findings are particularly concerning given DeepSeek's open-source nature and widespread deployment potential. This research emphasizes the critical need for embedding-level security measures in MLLM deployment pipelines and contributes to the broader discussion of responsible AI implementation.

摘要: 多模式大型语言模型(MLLM)代表了人工智能技术的前沿，DeepSeek模型成为领先的开源替代方案，提供了与封闭源代码系统竞争的性能。虽然这些模型展示了非凡的功能，但它们的视觉-语言集成机制引入了特定的漏洞。我们在DeepSeek Janus上实现了一种自适应的嵌入操作攻击，通过系统地优化图像嵌入来诱导目标视觉幻觉。通过对COCO、DALL-E 3和SVIT数据集的广泛实验，我们在对开放式问题的处理图像保持高视觉保真度(SSIM>0.88)的同时，获得了高达98.0%的幻觉率。我们的分析表明，DeepSeek Janus的1B和7B变体都容易受到这些攻击，封闭式评估显示，与开放式提问相比，幻觉率始终更高。我们介绍了一种新的多提示幻觉检测框架，使用LLAMA-3.1 8B指令进行健壮评估。考虑到DeepSeek的开源性质和广泛的部署潜力，这些发现的影响尤其令人担忧。这项研究强调了在MLLM部署管道中嵌入级别安全措施的迫切需要，并有助于更广泛地讨论负责任的人工智能实施。



## **21. Logicbreaks: A Framework for Understanding Subversion of Rule-based Inference**

Logicbreaks：理解基于规则的推理颠覆的框架 cs.AI

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2407.00075v3) [paper-pdf](http://arxiv.org/pdf/2407.00075v3)

**Authors**: Anton Xue, Avishree Khare, Rajeev Alur, Surbhi Goel, Eric Wong

**Abstract**: We study how to subvert large language models (LLMs) from following prompt-specified rules. We first formalize rule-following as inference in propositional Horn logic, a mathematical system in which rules have the form "if $P$ and $Q$, then $R$" for some propositions $P$, $Q$, and $R$. Next, we prove that although small transformers can faithfully follow such rules, maliciously crafted prompts can still mislead both theoretical constructions and models learned from data. Furthermore, we demonstrate that popular attack algorithms on LLMs find adversarial prompts and induce attention patterns that align with our theory. Our novel logic-based framework provides a foundation for studying LLMs in rule-based settings, enabling a formal analysis of tasks like logical reasoning and jailbreak attacks.

摘要: 我们研究如何根据预算指定的规则颠覆大型语言模型（LLM）。我们首先将规则遵循形式化为命题Horn逻辑中的推理，这是一个数学系统，其中规则的形式为“如果$P $和$Q $，那么$R $”，对于某些命题$P $、$Q $和$R $。接下来，我们证明，尽管小型变压器可以忠实地遵循这些规则，但恶意制作的提示仍然会误导理论构建和从数据中学习的模型。此外，我们证明了LLM上的流行攻击算法可以找到对抗提示并诱导与我们的理论一致的注意力模式。我们新颖的基于逻辑的框架为在基于规则的环境中研究LLM提供了基础，从而能够对逻辑推理和越狱攻击等任务进行正式分析。



## **22. Auditing Prompt Caching in Language Model APIs**

语言模型API中的审核提示缓存 cs.CL

20 pages, 7 figures

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07776v1) [paper-pdf](http://arxiv.org/pdf/2502.07776v1)

**Authors**: Chenchen Gu, Xiang Lisa Li, Rohith Kuditipudi, Percy Liang, Tatsunori Hashimoto

**Abstract**: Prompt caching in large language models (LLMs) results in data-dependent timing variations: cached prompts are processed faster than non-cached prompts. These timing differences introduce the risk of side-channel timing attacks. For example, if the cache is shared across users, an attacker could identify cached prompts from fast API response times to learn information about other users' prompts. Because prompt caching may cause privacy leakage, transparency around the caching policies of API providers is important. To this end, we develop and conduct statistical audits to detect prompt caching in real-world LLM API providers. We detect global cache sharing across users in seven API providers, including OpenAI, resulting in potential privacy leakage about users' prompts. Timing variations due to prompt caching can also result in leakage of information about model architecture. Namely, we find evidence that OpenAI's embedding model is a decoder-only Transformer, which was previously not publicly known.

摘要: 大型语言模型(LLM)中的提示缓存会导致与数据相关的时序变化：缓存的提示比未缓存的提示处理得更快。这些时序差异带来了旁路时序攻击的风险。例如，如果缓存在用户之间共享，攻击者可以从FAST API响应时间识别缓存的提示，以了解有关其他用户提示的信息。由于提示缓存可能会导致隐私泄露，因此API提供商的缓存策略的透明度非常重要。为此，我们开发并执行统计审计，以检测真实世界的LLMAPI提供程序中的即时缓存。我们检测到包括OpenAI在内的七个API提供商的用户之间的全局缓存共享，导致用户提示的潜在隐私泄露。由于提示缓存而引起的时序变化也可能导致有关模型体系结构的信息泄露。也就是说，我们发现了OpenAI的嵌入模型是一个仅限解码器的Transformer的证据，这在以前并不为人所知。



## **23. JBShield: Defending Large Language Models from Jailbreak Attacks through Activated Concept Analysis and Manipulation**

JBShield：通过激活概念分析和操纵保护大型语言模型免受越狱攻击 cs.CR

To Appear in the 34rd USENIX Security Symposium, August 13-15, 2025

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07557v1) [paper-pdf](http://arxiv.org/pdf/2502.07557v1)

**Authors**: Shenyi Zhang, Yuchen Zhai, Keyan Guo, Hongxin Hu, Shengnan Guo, Zheng Fang, Lingchen Zhao, Chao Shen, Cong Wang, Qian Wang

**Abstract**: Despite the implementation of safety alignment strategies, large language models (LLMs) remain vulnerable to jailbreak attacks, which undermine these safety guardrails and pose significant security threats. Some defenses have been proposed to detect or mitigate jailbreaks, but they are unable to withstand the test of time due to an insufficient understanding of jailbreak mechanisms. In this work, we investigate the mechanisms behind jailbreaks based on the Linear Representation Hypothesis (LRH), which states that neural networks encode high-level concepts as subspaces in their hidden representations. We define the toxic semantics in harmful and jailbreak prompts as toxic concepts and describe the semantics in jailbreak prompts that manipulate LLMs to comply with unsafe requests as jailbreak concepts. Through concept extraction and analysis, we reveal that LLMs can recognize the toxic concepts in both harmful and jailbreak prompts. However, unlike harmful prompts, jailbreak prompts activate the jailbreak concepts and alter the LLM output from rejection to compliance. Building on our analysis, we propose a comprehensive jailbreak defense framework, JBShield, consisting of two key components: jailbreak detection JBShield-D and mitigation JBShield-M. JBShield-D identifies jailbreak prompts by determining whether the input activates both toxic and jailbreak concepts. When a jailbreak prompt is detected, JBShield-M adjusts the hidden representations of the target LLM by enhancing the toxic concept and weakening the jailbreak concept, ensuring LLMs produce safe content. Extensive experiments demonstrate the superior performance of JBShield, achieving an average detection accuracy of 0.95 and reducing the average attack success rate of various jailbreak attacks to 2% from 61% across distinct LLMs.

摘要: 尽管实施了安全调整战略，但大型语言模型(LLM)仍然容易受到越狱攻击，这些攻击破坏了这些安全护栏，并构成了重大的安全威胁。已经提出了一些防御措施来检测或减轻越狱，但由于对越狱机制的了解不足，这些防御措施无法经受住时间的考验。在这项工作中，我们基于线性表示假说(LRH)来研究越狱背后的机制，该假说指出，神经网络将高级概念编码为其隐藏表示中的子空间。我们将有害提示和越狱提示中的有毒语义定义为有毒概念，并将操纵LLM遵从不安全请求的越狱提示中的语义描述为越狱概念。通过概念提取和分析，我们发现LLMS能够识别有害提示和越狱提示中的有毒概念。然而，与有害的提示不同，越狱提示激活了越狱概念，并将LLM输出从拒绝更改为遵守。基于我们的分析，我们提出了一个全面的越狱防御框架JBShield，它由两个关键组件组成：越狱检测JBShield-D和缓解JBShield-M。JBShield-D通过确定输入是否同时激活有毒和越狱概念来识别越狱提示。当检测到越狱提示时，JBShield-M通过增强有毒概念和弱化越狱概念来调整目标LLM的隐藏表示，确保LLM产生安全的内容。大量的实验证明了JBShield的优越性能，平均检测准确率达到0.95，并将不同LLM上各种越狱攻击的平均攻击成功率从61%降低到2%。



## **24. LUNAR: LLM Unlearning via Neural Activation Redirection**

LUNAR：LLM通过神经激活重定向消除学习 cs.LG

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07218v1) [paper-pdf](http://arxiv.org/pdf/2502.07218v1)

**Authors**: William F. Shen, Xinchi Qiu, Meghdad Kurmanji, Alex Iacob, Lorenzo Sani, Yihong Chen, Nicola Cancedda, Nicholas D. Lane

**Abstract**: Large Language Models (LLMs) benefit from training on ever larger amounts of textual data, but as a result, they increasingly incur the risk of leaking private information. The ability to selectively remove knowledge from LLMs is, therefore, a highly desirable capability. In this paper, we propose LUNAR, a novel unlearning methodology grounded in the Linear Representation Hypothesis. LUNAR operates by redirecting the representations of unlearned data to regions that trigger the model's inherent ability to express its inability to answer. LUNAR achieves state-of-the-art unlearning performance while significantly enhancing the controllability of the unlearned model during inference. Specifically, LUNAR achieves between 2.9x to 11.7x improvements on combined "unlearning efficacy" and "model utility" score ("Deviation Score") on the PISTOL dataset across various base models. We also demonstrate, through quantitative analysis and qualitative examples, LUNAR's superior controllability in generating coherent and contextually aware responses, mitigating undesired side effects of existing methods. Moreover, we demonstrate that LUNAR is robust against white-box adversarial attacks and versatile in handling real-world scenarios, such as processing sequential unlearning requests.

摘要: 大型语言模型(LLM)受益于对越来越多的文本数据进行培训，但结果是，它们越来越多地招致泄露私人信息的风险。因此，有选择地从LLM中移除知识的能力是一种非常理想的能力。在本文中，我们提出了一种基于线性表征假设的去学习方法--LUNAR。LUNAR通过将未学习数据的表示重定向到触发模型表达其无法回答问题的固有能力的区域来运行。LUNAR实现了最先进的遗忘性能，同时显著增强了推理过程中未学习模型的可控性。具体地说，在各种基本型号的手枪数据集上，LUNAR在组合的“遗忘效能”和“模型效用”分数(“偏差分数”)上取得了2.9倍到11.7倍的改进。我们还通过定量分析和定性例子证明，月球在产生连贯的和上下文感知的响应方面具有优越的可控性，减轻了现有方法的不良副作用。此外，我们还证明了LUNAR对白盒攻击具有很强的健壮性，并且在处理真实场景(如处理顺序遗忘请求)方面具有很强的通用性。



## **25. LLM Agent Honeypot: Monitoring AI Hacking Agents in the Wild**

LLM Agent Honeypot：监控野外人工智能黑客代理 cs.CR

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2410.13919v2) [paper-pdf](http://arxiv.org/pdf/2410.13919v2)

**Authors**: Reworr, Dmitrii Volkov

**Abstract**: Attacks powered by Large Language Model (LLM) agents represent a growing threat to modern cybersecurity. To address this concern, we present LLM Honeypot, a system designed to monitor autonomous AI hacking agents. By augmenting a standard SSH honeypot with prompt injection and time-based analysis techniques, our framework aims to distinguish LLM agents among all attackers. Over a trial deployment of about three months in a public environment, we collected 8,130,731 hacking attempts and 8 potential AI agents. Our work demonstrates the emergence of AI-driven threats and their current level of usage, serving as an early warning of malicious LLM agents in the wild.

摘要: 由大型语言模型（LLM）代理支持的攻击对现代网络安全构成了日益严重的威胁。为了解决这个问题，我们提出了LLM Honeypot，这是一个旨在监控自主人工智能黑客代理的系统。通过使用即时注入和基于时间的分析技术来扩展标准的SSH蜜罐，我们的框架旨在区分LLM代理在所有攻击者中。在公共环境中进行了大约三个月的试验部署，我们收集了8，130，731次黑客尝试和8个潜在的人工智能代理。我们的工作展示了人工智能驱动的威胁的出现及其当前的使用水平，作为野外恶意LLM代理的预警。



## **26. AdaPhish: AI-Powered Adaptive Defense and Education Resource Against Deceptive Emails**

AdaPhish：针对欺骗性电子邮件的人工智能驱动自适应防御和教育资源 cs.CR

7 pages, 3 figures, 2 tables, accepted in 4th IEEE International  Conference on AI in Cybersecurity (ICAIC)

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.03622v2) [paper-pdf](http://arxiv.org/pdf/2502.03622v2)

**Authors**: Rei Meguro, Ng S. T. Chong

**Abstract**: Phishing attacks remain a significant threat in the digital age, yet organizations lack effective methods to tackle phishing attacks without leaking sensitive information. Phish bowl initiatives are a vital part of cybersecurity efforts against these attacks. However, traditional phish bowls require manual anonymization and are often limited to internal use. To overcome these limitations, we introduce AdaPhish, an AI-powered phish bowl platform that automatically anonymizes and analyzes phishing emails using large language models (LLMs) and vector databases. AdaPhish achieves real-time detection and adaptation to new phishing tactics while enabling long-term tracking of phishing trends. Through automated reporting, adaptive analysis, and real-time alerts, AdaPhish presents a scalable, collaborative solution for phishing detection and cybersecurity education.

摘要: 网络钓鱼攻击仍然是数字时代的重大威胁，但组织缺乏有效的方法来在不泄露敏感信息的情况下应对网络钓鱼攻击。Phish bowl计划是针对这些攻击的网络安全工作的重要组成部分。然而，传统的钓鱼碗需要手动匿名化，并且通常仅限于内部使用。为了克服这些限制，我们引入了AdaPhish，这是一个人工智能驱动的钓鱼碗平台，可以使用大型语言模型（LLM）和载体数据库自动匿名化和分析网络钓鱼电子邮件。AdaPhish实现了实时检测和适应新的网络钓鱼策略，同时能够长期跟踪网络钓鱼趋势。通过自动报告、自适应分析和实时警报，AdaPhish为网络钓鱼检测和网络安全教育提供了可扩展的协作解决方案。



## **27. Tamper-Resistant Safeguards for Open-Weight LLMs**

开放重量LLM的防篡改保障措施 cs.LG

Website: https://www.tamper-resistant-safeguards.com

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2408.00761v4) [paper-pdf](http://arxiv.org/pdf/2408.00761v4)

**Authors**: Rishub Tamirisa, Bhrugu Bharathi, Long Phan, Andy Zhou, Alice Gatti, Tarun Suresh, Maxwell Lin, Justin Wang, Rowan Wang, Ron Arel, Andy Zou, Dawn Song, Bo Li, Dan Hendrycks, Mantas Mazeika

**Abstract**: Rapid advances in the capabilities of large language models (LLMs) have raised widespread concerns regarding their potential for malicious use. Open-weight LLMs present unique challenges, as existing safeguards lack robustness to tampering attacks that modify model weights. For example, recent works have demonstrated that refusal and unlearning safeguards can be trivially removed with a few steps of fine-tuning. These vulnerabilities necessitate new approaches for enabling the safe release of open-weight LLMs. We develop a method, called TAR, for building tamper-resistant safeguards into open-weight LLMs such that adversaries cannot remove the safeguards even after hundreds of steps of fine-tuning. In extensive evaluations and red teaming analyses, we find that our method greatly improves tamper-resistance while preserving benign capabilities. Our results demonstrate that progress on tamper-resistance is possible, opening up a promising new avenue to improve the safety and security of open-weight LLMs.

摘要: 大型语言模型(LLM)功能的快速发展引起了人们对其潜在恶意使用的广泛关注。开放重量LLM提出了独特的挑战，因为现有的保障措施缺乏对篡改模型权重的篡改攻击的稳健性。例如，最近的研究表明，通过几个步骤的微调，就可以很容易地消除拒绝和遗忘的保障措施。这些漏洞需要新的方法来实现安全释放未加重量的低密度脂蛋白。我们开发了一种名为TAR的方法，用于在开放重量的LLM中构建防篡改保护措施，以便即使在数百个步骤的微调之后，对手也无法移除这些保护措施。在广泛的评估和红团队分析中，我们发现我们的方法在保持良性性能的同时大大提高了防篡改能力。我们的结果表明，在防篡改方面取得进展是可能的，为提高开放重量LLMS的安全性开辟了一条有希望的新途径。



## **28. Exploring Audio Editing Features as User-Centric Privacy Defenses Against Large Language Model(LLM) Based Emotion Inference Attacks**

探索音频编辑功能，以用户为中心的隐私防御基于大型语言模型（LLM）的情感推理攻击 cs.CR

Accepted for presentation(Poster) at PPAI-25: The 6th AAAI Workshop  on Privacy-Preserving Artificial Intelligence

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2501.18727v2) [paper-pdf](http://arxiv.org/pdf/2501.18727v2)

**Authors**: Mohd. Farhan Israk Soumik, W. K. M. Mithsara, Abdur R. Shahid, Ahmed Imteaj

**Abstract**: The rapid proliferation of speech-enabled technologies, including virtual assistants, video conferencing platforms, and wearable devices, has raised significant privacy concerns, particularly regarding the inference of sensitive emotional information from audio data. Existing privacy-preserving methods often compromise usability and security, limiting their adoption in practical scenarios. This paper introduces a novel, user-centric approach that leverages familiar audio editing techniques, specifically pitch and tempo manipulation, to protect emotional privacy without sacrificing usability. By analyzing popular audio editing applications on Android and iOS platforms, we identified these features as both widely available and usable. We rigorously evaluated their effectiveness against a threat model, considering adversarial attacks from diverse sources, including Deep Neural Networks (DNNs), Large Language Models (LLMs), and and reversibility testing. Our experiments, conducted on three distinct datasets, demonstrate that pitch and tempo manipulation effectively obfuscates emotional data. Additionally, we explore the design principles for lightweight, on-device implementation to ensure broad applicability across various devices and platforms.

摘要: 包括虚拟助理、视频会议平台和可穿戴设备在内的语音支持技术的迅速普及引发了人们对隐私的严重担忧，特别是关于从音频数据推断敏感情感信息的问题。现有的隐私保护方法往往会损害可用性和安全性，限制了它们在实际场景中的采用。本文介绍了一种新颖的、以用户为中心的方法，该方法利用熟悉的音频编辑技术，特别是音调和节奏操作，在不牺牲可用性的情况下保护情感隐私。通过分析Android和iOS平台上流行的音频编辑应用程序，我们发现这些功能广泛使用和使用。我们严格评估了它们对威胁模型的有效性，考虑了来自不同来源的对抗性攻击，包括深度神经网络(DNN)、大型语言模型(LLMS)和可逆性测试。我们在三个不同的数据集上进行的实验表明，音调和节奏操作有效地混淆了情感数据。此外，我们还探讨了轻量级设备上实施的设计原则，以确保跨各种设备和平台的广泛适用性。



## **29. Preserving Privacy in Large Language Models: A Survey on Current Threats and Solutions**

在大型语言模型中保护隐私：对当前威胁和解决方案的调查 cs.CR

Published in Transactions on Machine Learning Research (TMLR)  https://openreview.net/forum?id=Ss9MTTN7OL

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2408.05212v2) [paper-pdf](http://arxiv.org/pdf/2408.05212v2)

**Authors**: Michele Miranda, Elena Sofia Ruzzetti, Andrea Santilli, Fabio Massimo Zanzotto, Sébastien Bratières, Emanuele Rodolà

**Abstract**: Large Language Models (LLMs) represent a significant advancement in artificial intelligence, finding applications across various domains. However, their reliance on massive internet-sourced datasets for training brings notable privacy issues, which are exacerbated in critical domains (e.g., healthcare). Moreover, certain application-specific scenarios may require fine-tuning these models on private data. This survey critically examines the privacy threats associated with LLMs, emphasizing the potential for these models to memorize and inadvertently reveal sensitive information. We explore current threats by reviewing privacy attacks on LLMs and propose comprehensive solutions for integrating privacy mechanisms throughout the entire learning pipeline. These solutions range from anonymizing training datasets to implementing differential privacy during training or inference and machine unlearning after training. Our comprehensive review of existing literature highlights ongoing challenges, available tools, and future directions for preserving privacy in LLMs. This work aims to guide the development of more secure and trustworthy AI systems by providing a thorough understanding of privacy preservation methods and their effectiveness in mitigating risks.

摘要: 大型语言模型(LLM)代表了人工智能的一项重大进步，可以找到跨各个领域的应用程序。然而，他们对来自互联网的海量数据集的依赖带来了显著的隐私问题，在关键领域(例如医疗保健)，这一问题更加严重。此外，某些特定于应用程序的场景可能需要根据私有数据对这些模型进行微调。这项调查严格审查了与LLMS相关的隐私威胁，强调了这些模型可能会记住并无意中泄露敏感信息。我们通过审查对LLM的隐私攻击来探索当前的威胁，并提出全面的解决方案，将隐私机制整合到整个学习管道中。这些解决方案的范围从匿名训练数据集到在训练期间实现差异隐私或在训练后进行推理和机器遗忘。我们对现有文献的全面回顾突出了在LLMS中保护隐私的持续挑战、可用的工具和未来的方向。这项工作旨在通过提供对隐私保护方法及其在降低风险方面的有效性的透彻了解，来指导更安全和值得信赖的人工智能系统的开发。



## **30. Jailbreaking LLMs' Safeguard with Universal Magic Words for Text Embedding Models**

越狱的LLM为文本嵌入模型提供通用魔法词的保障 cs.CL

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2501.18280v2) [paper-pdf](http://arxiv.org/pdf/2501.18280v2)

**Authors**: Haoyu Liang, Youran Sun, Yunfeng Cai, Jun Zhu, Bo Zhang

**Abstract**: The security issue of large language models (LLMs) has gained significant attention recently, with various defense mechanisms developed to prevent harmful outputs, among which safeguards based on text embedding models serve as a fundamental defense. Through testing, we discover that the distribution of text embedding model outputs is significantly biased with a large mean. Inspired by this observation, we propose novel efficient methods to search for universal magic words that can attack text embedding models. The universal magic words as suffixes can move the embedding of any text towards the bias direction, therefore manipulate the similarity of any text pair and mislead safeguards. By appending magic words to user prompts and requiring LLMs to end answers with magic words, attackers can jailbreak the safeguard. To eradicate this security risk, we also propose defense mechanisms against such attacks, which can correct the biased distribution of text embeddings in a train-free manner.

摘要: 大型语言模型（LLM）的安全问题最近受到了广泛关注，各种防御机制被开发出来来防止有害输出，其中基于文本嵌入模型的保护措施是基本防御。通过测试，我们发现文本嵌入模型输出的分布存在显着偏差，平均值很大。受这一观察的启发，我们提出了新颖的有效方法来搜索可以攻击文本嵌入模型的通用魔法词。作为后缀的通用神奇词可以将任何文本的嵌入移向偏向方向，从而操纵任何文本对的相似性并误导保障措施。通过在用户提示中添加魔法词并要求LLM以魔法词结束回答，攻击者可以越狱该保护措施。为了消除这种安全风险，我们还提出了针对此类攻击的防御机制，该机制可以以无训练的方式纠正文本嵌入的偏见分布。



## **31. Panza: Design and Analysis of a Fully-Local Personalized Text Writing Assistant**

Panza：全本地个性化文本写作助手的设计与分析 cs.CL

Panza is available at https://github.com/IST-DASLab/PanzaMail

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2407.10994v4) [paper-pdf](http://arxiv.org/pdf/2407.10994v4)

**Authors**: Armand Nicolicioiu, Eugenia Iofinova, Andrej Jovanovic, Eldar Kurtic, Mahdi Nikdan, Andrei Panferov, Ilia Markov, Nir Shavit, Dan Alistarh

**Abstract**: The availability of powerful open-source large language models (LLMs) opens exciting use-cases, such as using personal data to fine-tune these models to imitate a user's unique writing style. Two key requirements for such assistants are personalization - in the sense that the assistant should recognizably reflect the user's own writing style - and privacy - users may justifiably be wary of uploading extremely personal data, such as their email archive, to a third-party service. In this paper, we present a new design and evaluation for such an automated assistant, for the specific use case of email generation, which we call Panza. Panza's personalization features are based on a combination of fine-tuning using a variant of the Reverse Instructions technique together with Retrieval-Augmented Generation (RAG). We demonstrate that this combination allows us to fine-tune an LLM to reflect a user's writing style using limited data, while executing on extremely limited resources, e.g. on a free Google Colab instance. Our key methodological contribution is the first detailed study of evaluation metrics for this personalized writing task, and of how different choices of system components--the use of RAG and of different fine-tuning approaches-impact the system's performance. Additionally, we demonstrate that very little data - under 100 email samples - are sufficient to create models that convincingly imitate humans. This finding showcases a previously-unknown attack vector in language models - that access to a small number of writing samples can allow a bad actor to cheaply create generative models that imitate a target's writing style. We are releasing the full Panza code as well as three new email datasets licensed for research use at https://github.com/IST-DASLab/PanzaMail.

摘要: 强大的开源大型语言模型(LLM)的出现开启了令人兴奋的用例，例如使用个人数据来微调这些模型，以模仿用户独特的写作风格。这类助手的两个关键要求是个性化--从这个意义上说，助手应该明显地反映用户自己的写作风格--以及隐私--用户可能有理由对将极其个人化的数据，如他们的电子邮件档案，上传到第三方服务持谨慎态度。在本文中，我们提出了一种新的自动化助手的设计和评估，针对电子邮件生成的特定用例，我们称之为Panza。Panza的个性化功能是基于使用反向指令技术的变体进行微调以及检索-增强生成(RAG)的组合。我们证明，这种组合允许我们使用有限的数据微调LLM以反映用户的写作风格，同时在极其有限的资源上执行，例如在免费的Google Colab实例上执行。我们的主要方法论贡献是首次详细研究了这种个性化写作任务的评估指标，以及系统组件的不同选择--使用RAG和不同的微调方法--如何影响系统的性能。此外，我们还展示了极少的数据--不到100个电子邮件样本--足以创建令人信服地模仿人类的模型。这一发现揭示了语言模型中一种以前不为人知的攻击媒介--获取少量写作样本可以让糟糕的演员廉价地创建模仿目标写作风格的生成模型。我们将发布完整的PANZA代码以及三个新的电子邮件数据集，这些数据集已被授权用于https://github.com/IST-DASLab/PanzaMail.的研究



## **32. Detecting Backdoor Samples in Contrastive Language Image Pretraining**

对比语言图像预训练中后门样本检测 cs.LG

ICLR2025

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.01385v2) [paper-pdf](http://arxiv.org/pdf/2502.01385v2)

**Authors**: Hanxun Huang, Sarah Erfani, Yige Li, Xingjun Ma, James Bailey

**Abstract**: Contrastive language-image pretraining (CLIP) has been found to be vulnerable to poisoning backdoor attacks where the adversary can achieve an almost perfect attack success rate on CLIP models by poisoning only 0.01\% of the training dataset. This raises security concerns on the current practice of pretraining large-scale models on unscrutinized web data using CLIP. In this work, we analyze the representations of backdoor-poisoned samples learned by CLIP models and find that they exhibit unique characteristics in their local subspace, i.e., their local neighborhoods are far more sparse than that of clean samples. Based on this finding, we conduct a systematic study on detecting CLIP backdoor attacks and show that these attacks can be easily and efficiently detected by traditional density ratio-based local outlier detectors, whereas existing backdoor sample detection methods fail. Our experiments also reveal that an unintentional backdoor already exists in the original CC3M dataset and has been trained into a popular open-source model released by OpenCLIP. Based on our detector, one can clean up a million-scale web dataset (e.g., CC3M) efficiently within 15 minutes using 4 Nvidia A100 GPUs. The code is publicly available in our \href{https://github.com/HanxunH/Detect-CLIP-Backdoor-Samples}{GitHub repository}.

摘要: 对比语言图像预训练(CLIP)被发现容易受到中毒后门攻击，对手只需中毒0.01%的训练数据集就可以在CLIP模型上获得近乎完美的攻击成功率。这引发了人们对当前使用CLIP对大规模模型进行未经审查的网络数据的预培训的安全担忧。在这项工作中，我们分析了通过CLIP模型学习的后门中毒样本的表示，发现它们在局部子空间中表现出独特的特征，即它们的局部邻域比干净样本的局部邻域稀疏得多。基于这一发现，我们对CLIP后门攻击的检测进行了系统的研究，结果表明，传统的基于密度比的局部离群点检测方法可以轻松有效地检测到这些攻击，而现有的后门样本检测方法却无法检测到这些攻击。我们的实验还表明，在原始的CC3M数据集中已经存在一个无意的后门，并且已经被训练成OpenCLIP发布的流行的开源模型。基于我们的检测器，您可以使用4个NVIDIA A100图形处理器在15分钟内高效清理百万级Web数据集(例如CC3M)。代码在我们的\href{https://github.com/HanxunH/Detect-CLIP-Backdoor-Samples}{GitHub存储库中公开提供。



## **33. Confidence Elicitation: A New Attack Vector for Large Language Models**

信心激发：大型语言模型的新攻击载体 cs.LG

Published in ICLR 2025. The code is publicly available at  https://github.com/Aniloid2/Confidence_Elicitation_Attacks

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.04643v2) [paper-pdf](http://arxiv.org/pdf/2502.04643v2)

**Authors**: Brian Formento, Chuan Sheng Foo, See-Kiong Ng

**Abstract**: A fundamental issue in deep learning has been adversarial robustness. As these systems have scaled, such issues have persisted. Currently, large language models (LLMs) with billions of parameters suffer from adversarial attacks just like their earlier, smaller counterparts. However, the threat models have changed. Previously, having gray-box access, where input embeddings or output logits/probabilities were visible to the user, might have been reasonable. However, with the introduction of closed-source models, no information about the model is available apart from the generated output. This means that current black-box attacks can only utilize the final prediction to detect if an attack is successful. In this work, we investigate and demonstrate the potential of attack guidance, akin to using output probabilities, while having only black-box access in a classification setting. This is achieved through the ability to elicit confidence from the model. We empirically show that the elicited confidence is calibrated and not hallucinated for current LLMs. By minimizing the elicited confidence, we can therefore increase the likelihood of misclassification. Our new proposed paradigm demonstrates promising state-of-the-art results on three datasets across two models (LLaMA-3-8B-Instruct and Mistral-7B-Instruct-V0.3) when comparing our technique to existing hard-label black-box attack methods that introduce word-level substitutions.

摘要: 深度学习的一个基本问题是对手的稳健性。随着这些系统的规模扩大，这样的问题一直存在。目前，具有数十亿个参数的大型语言模型(LLM)就像它们早期的较小对应模型一样，受到对手攻击。然而，威胁模型已经发生了变化。以前，拥有灰箱访问，其中输入嵌入或输出日志/概率对用户可见，可能是合理的。然而，随着闭源模型的引入，除了生成的输出之外，没有关于模型的信息可用。这意味着当前的黑盒攻击只能利用最终预测来检测攻击是否成功。在这项工作中，我们调查和演示了攻击指导的潜力，类似于使用输出概率，而在分类设置中只有黑盒访问。这是通过从模型中获得信心的能力来实现的。我们的经验表明，对于当前的LLM来说，引发的信心是经过校准的，而不是幻觉的。因此，通过将引起的置信度降至最低，我们可以增加错误分类的可能性。我们提出的新范式在两个模型(骆驼-3-8B-指令和Mistral-7B-指令-V0.3)的三个数据集上展示了有希望的最先进结果，当将我们的技术与现有的引入词级替换的硬标签黑盒攻击方法进行比较时。



## **34. A Practical Examination of AI-Generated Text Detectors for Large Language Models**

大型语言模型的人工智能生成文本检测器的实践检验 cs.CL

9 pages

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2412.05139v4) [paper-pdf](http://arxiv.org/pdf/2412.05139v4)

**Authors**: Brian Tufts, Xuandong Zhao, Lei Li

**Abstract**: The proliferation of large language models has raised growing concerns about their misuse, particularly in cases where AI-generated text is falsely attributed to human authors. Machine-generated content detectors claim to effectively identify such text under various conditions and from any language model. This paper critically evaluates these claims by assessing several popular detectors (RADAR, Wild, T5Sentinel, Fast-DetectGPT, PHD, LogRank, Binoculars) on a range of domains, datasets, and models that these detectors have not previously encountered. We employ various prompting strategies to simulate practical adversarial attacks, demonstrating that even moderate efforts can significantly evade detection. We emphasize the importance of the true positive rate at a specific false positive rate (TPR@FPR) metric and demonstrate that these detectors perform poorly in certain settings, with TPR@.01 as low as 0%. Our findings suggest that both trained and zero-shot detectors struggle to maintain high sensitivity while achieving a reasonable true positive rate.

摘要: 大型语言模型的激增引发了人们对它们滥用的日益担忧，特别是在人工智能生成的文本被错误地归因于人类作者的情况下。机器生成的内容检测器声称可以在各种条件下从任何语言模型有效地识别此类文本。本文通过评估几种流行的探测器(雷达、Wild、T5Sentinel、Fast-DetectGPT、PHD、logrank、双筒望远镜)，对这些声称进行了批判性的评估，这些探测器以前从未遇到过。我们使用各种提示策略来模拟实际的对抗性攻击，表明即使是适度的攻击也可以显著地躲避检测。我们强调了在特定的假阳性率(TPR@fPR)度量下的真阳性率的重要性，并证明了这些检测器在某些设置下表现很差，TPR@.01低至0%。我们的发现表明，训练有素的探测器和零射探测器都很难在保持高灵敏度的同时获得合理的真阳性率。



## **35. Cyri: A Conversational AI-based Assistant for Supporting the Human User in Detecting and Responding to Phishing Attacks**

Cyri：一款基于对话的人工智能助手，支持人类用户检测和响应网络钓鱼攻击 cs.HC

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2502.05951v1) [paper-pdf](http://arxiv.org/pdf/2502.05951v1)

**Authors**: Antonio La Torre, Marco Angelini

**Abstract**: This work introduces Cyri, an AI-powered conversational assistant designed to support a human user in detecting and analyzing phishing emails by leveraging Large Language Models. Cyri has been designed to scrutinize emails for semantic features used in phishing attacks, such as urgency, and undesirable consequences, using an approach that unifies features already established in the literature with others by Cyri features extraction methodology. Cyri can be directly plugged into a client mail or webmail, ensuring seamless integration with the user's email workflow while maintaining data privacy through local processing. By performing analyses on the user's machine, Cyri eliminates the need to transmit sensitive email data over the internet, reducing associated security risks. The Cyri user interface has been designed to reduce habituation effects and enhance user engagement. It employs dynamic visual cues and context-specific explanations to keep users alert and informed while using emails. Additionally, it allows users to explore identified malicious semantic features both through conversation with the agent and visual exploration, obtaining the advantages of both modalities for expert or non-expert users. It also allows users to keep track of the conversation, supports the user in solving additional questions on both computed features or new parts of the mail, and applies its detection on demand. To evaluate Cyri, we crafted a comprehensive dataset of 420 phishing emails and 420 legitimate emails. Results demonstrate high effectiveness in identifying critical phishing semantic features fundamental to phishing detection. A user study involving 10 participants, both experts and non-experts, evaluated Cyri's effectiveness and usability. Results indicated that Cyri significantly aided users in identifying phishing emails and enhanced their understanding of phishing tactics.

摘要: 这项工作介绍了Cyri，这是一个人工智能支持的对话助手，旨在支持人类用户通过利用大型语言模型来检测和分析钓鱼电子邮件。Cyri被设计用于仔细检查电子邮件中用于网络钓鱼攻击的语义特征，如紧迫性和不良后果，使用一种方法，通过Cyri特征提取方法将文献中已建立的特征与其他特征统一起来。Cyri可以直接插入客户端邮件或网络邮件，确保与用户的电子邮件工作流程无缝集成，同时通过本地处理保持数据隐私。通过在用户机器上执行分析，Cyri消除了通过互联网传输敏感电子邮件数据的需要，从而降低了相关的安全风险。Cyri的用户界面旨在减少习惯性影响，提高用户参与度。它使用动态视觉提示和特定于上下文的解释来保持用户在使用电子邮件时的警觉和通知。此外，它允许用户通过与代理的对话和视觉探索来探索已识别的恶意语义特征，从而获得专家或非专家用户的这两种模式的优势。它还允许用户跟踪对话，支持用户解决有关计算功能或邮件新部分的其他问题，并按需应用其检测。为了评估Cyri，我们精心制作了一个包含420封钓鱼电子邮件和420封合法电子邮件的全面数据集。结果表明，在识别网络钓鱼检测基础上的关键网络钓鱼语义特征方面具有很高的效率。一项涉及10名参与者的用户研究，包括专家和非专家，评估了Cyri的有效性和可用性。结果表明，Cyri显著地帮助用户识别钓鱼电子邮件，并增强了他们对钓鱼策略的理解。



## **36. Large Language Models are Easily Confused: A Quantitative Metric, Security Implications and Typological Analysis**

大型语言模型很容易混淆：量化指标、安全含义和类型学分析 cs.CL

18 pages, 15 figures, 14 tables

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2410.13237v2) [paper-pdf](http://arxiv.org/pdf/2410.13237v2)

**Authors**: Yiyi Chen, Qiongxiu Li, Russa Biswas, Johannes Bjerva

**Abstract**: Language Confusion is a phenomenon where Large Language Models (LLMs) generate text that is neither in the desired language, nor in a contextually appropriate language. This phenomenon presents a critical challenge in text generation by LLMs, often appearing as erratic and unpredictable behavior. We hypothesize that there are linguistic regularities to this inherent vulnerability in LLMs and shed light on patterns of language confusion across LLMs. We introduce a novel metric, Language Confusion Entropy, designed to directly measure and quantify this confusion, based on language distributions informed by linguistic typology and lexical variation. Comprehensive comparisons with the Language Confusion Benchmark (Marchisio et al., 2024) confirm the effectiveness of our metric, revealing patterns of language confusion across LLMs. We further link language confusion to LLM security, and find patterns in the case of multilingual embedding inversion attacks. Our analysis demonstrates that linguistic typology offers theoretically grounded interpretation, and valuable insights into leveraging language similarities as a prior for LLM alignment and security.

摘要: 语言混淆是一种现象，大型语言模型(LLM)生成的文本既不是所需语言的文本，也不是上下文合适的语言文本。这一现象对LLMS的文本生成提出了严重的挑战，通常表现为不稳定和不可预测的行为。我们假设LLMS中这种固有的脆弱性存在语言规则，并揭示了LLMS中语言混淆的模式。我们引入了一种新的度量，语言混淆熵，旨在根据语言类型和词汇变异提供的语言分布来直接度量和量化这种混淆。与语言混淆基准(Marchisio等人，2024年)的全面比较证实了我们的度量的有效性，揭示了LLM之间的语言混淆模式。我们进一步将语言混淆与LLM安全联系起来，并在多语言嵌入反转攻击的情况下找到了模式。我们的分析表明，语言类型学提供了理论上的解释，并提供了关于利用语言相似性作为LLM对齐和安全的先决条件的有价值的见解。



## **37. Arabic Dataset for LLM Safeguard Evaluation**

LLM保障评估的阿拉伯数据集 cs.CL

Accepted at NAACL 2025 Main Conference

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2410.17040v2) [paper-pdf](http://arxiv.org/pdf/2410.17040v2)

**Authors**: Yasser Ashraf, Yuxia Wang, Bin Gu, Preslav Nakov, Timothy Baldwin

**Abstract**: The growing use of large language models (LLMs) has raised concerns regarding their safety. While many studies have focused on English, the safety of LLMs in Arabic, with its linguistic and cultural complexities, remains under-explored. Here, we aim to bridge this gap. In particular, we present an Arab-region-specific safety evaluation dataset consisting of 5,799 questions, including direct attacks, indirect attacks, and harmless requests with sensitive words, adapted to reflect the socio-cultural context of the Arab world. To uncover the impact of different stances in handling sensitive and controversial topics, we propose a dual-perspective evaluation framework. It assesses the LLM responses from both governmental and opposition viewpoints. Experiments over five leading Arabic-centric and multilingual LLMs reveal substantial disparities in their safety performance. This reinforces the need for culturally specific datasets to ensure the responsible deployment of LLMs.

摘要: 大型语言模型（LLM）的越来越多的使用引发了人们对其安全性的担忧。虽然许多研究都集中在英语上，但阿拉伯语法学硕士的安全性及其语言和文化复杂性仍然没有得到充分的探讨。在这里，我们的目标是弥合这一差距。特别是，我们提供了一个特定于阿拉伯地区的安全评估数据集，由5，799个问题组成，包括直接攻击、间接攻击和带有敏感词的无害请求，经过调整以反映阿拉伯世界的社会文化背景。为了揭示不同立场对处理敏感和争议话题的影响，我们提出了一个双视角评估框架。它从政府和反对派的角度评估了法学硕士的回应。对五个领先的以阿拉伯语为中心的多语言LLM的实验揭示了它们的安全性能存在巨大差异。这强化了对特定文化数据集的需求，以确保负责任地部署LLM。



## **38. Mask-based Membership Inference Attacks for Retrieval-Augmented Generation**

用于检索增强生成的基于面具的成员推断攻击 cs.CR

This paper is accepted by conference WWW 2025

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2410.20142v2) [paper-pdf](http://arxiv.org/pdf/2410.20142v2)

**Authors**: Mingrui Liu, Sixiao Zhang, Cheng Long

**Abstract**: Retrieval-Augmented Generation (RAG) has been an effective approach to mitigate hallucinations in large language models (LLMs) by incorporating up-to-date and domain-specific knowledge. Recently, there has been a trend of storing up-to-date or copyrighted data in RAG knowledge databases instead of using it for LLM training. This practice has raised concerns about Membership Inference Attacks (MIAs), which aim to detect if a specific target document is stored in the RAG system's knowledge database so as to protect the rights of data producers. While research has focused on enhancing the trustworthiness of RAG systems, existing MIAs for RAG systems remain largely insufficient. Previous work either relies solely on the RAG system's judgment or is easily influenced by other documents or the LLM's internal knowledge, which is unreliable and lacks explainability. To address these limitations, we propose a Mask-Based Membership Inference Attacks (MBA) framework. Our framework first employs a masking algorithm that effectively masks a certain number of words in the target document. The masked text is then used to prompt the RAG system, and the RAG system is required to predict the mask values. If the target document appears in the knowledge database, the masked text will retrieve the complete target document as context, allowing for accurate mask prediction. Finally, we adopt a simple yet effective threshold-based method to infer the membership of target document by analyzing the accuracy of mask prediction. Our mask-based approach is more document-specific, making the RAG system's generation less susceptible to distractions from other documents or the LLM's internal knowledge. Extensive experiments demonstrate the effectiveness of our approach compared to existing baseline models.

摘要: 检索-增强生成(RAG)是一种通过结合最新的和特定领域的知识来缓解大型语言模型(LLMS)中的幻觉的有效方法。最近，有一种趋势是将最新数据或受版权保护的数据存储在RAG知识数据库中，而不是将其用于LLM培训。这种做法引起了人们对成员资格推断攻击(MIA)的担忧，这种攻击旨在检测特定目标文件是否存储在RAG系统的知识数据库中，以保护数据制作者的权利。虽然研究的重点是提高RAG系统的可信度，但现有的RAG系统的MIA仍然很大程度上是不够的。以往的工作要么完全依赖RAG系统的判断，要么容易受到其他文件或LLM内部知识的影响，这是不可靠的，缺乏解释性。针对这些局限性，我们提出了一种基于掩码的成员关系推理攻击(MBA)框架。我们的框架首先使用掩码算法，该算法有效地掩码目标文档中的特定数量的单词。然后使用掩码文本来提示RAG系统，并且需要RAG系统来预测掩码值。如果目标文档出现在知识数据库中，则掩码文本将检索完整的目标文档作为上下文，从而实现准确的掩码预测。最后，通过分析模板预测的精度，采用一种简单有效的基于阈值的方法来推断目标文档的隶属度。我们的基于掩码的方法更特定于文档，使RAG系统的生成不太容易受到来自其他文档或LLM内部知识的干扰。大量的实验表明，与现有的基线模型相比，我们的方法是有效的。



## **39. Effective Black-Box Multi-Faceted Attacks Breach Vision Large Language Model Guardrails**

有效的黑匣子多面攻击破坏视觉大型语言模型护栏 cs.CV

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2502.05772v1) [paper-pdf](http://arxiv.org/pdf/2502.05772v1)

**Authors**: Yijun Yang, Lichao Wang, Xiao Yang, Lanqing Hong, Jun Zhu

**Abstract**: Vision Large Language Models (VLLMs) integrate visual data processing, expanding their real-world applications, but also increasing the risk of generating unsafe responses. In response, leading companies have implemented Multi-Layered safety defenses, including alignment training, safety system prompts, and content moderation. However, their effectiveness against sophisticated adversarial attacks remains largely unexplored. In this paper, we propose MultiFaceted Attack, a novel attack framework designed to systematically bypass Multi-Layered Defenses in VLLMs. It comprises three complementary attack facets: Visual Attack that exploits the multimodal nature of VLLMs to inject toxic system prompts through images; Alignment Breaking Attack that manipulates the model's alignment mechanism to prioritize the generation of contrasting responses; and Adversarial Signature that deceives content moderators by strategically placing misleading information at the end of the response. Extensive evaluations on eight commercial VLLMs in a black-box setting demonstrate that MultiFaceted Attack achieves a 61.56% attack success rate, surpassing state-of-the-art methods by at least 42.18%.

摘要: 视觉大语言模型(VLLM)集成了可视化数据处理，扩展了它们在现实世界中的应用，但也增加了生成不安全响应的风险。作为回应，领先的公司实施了多层次的安全防御措施，包括对齐培训、安全系统提示和内容审核。然而，它们对抗复杂的对抗性攻击的有效性在很大程度上仍未得到探索。在本文中，我们提出了一种新的攻击框架--多方面攻击，旨在系统地绕过VLLMS中的多层防御。它包括三个互补的攻击方面：利用VLLM的多模式特性通过图像注入有毒系统提示的视觉攻击；操纵模型的对齐机制以优先生成对比响应的对齐破坏攻击；以及通过在响应的末尾战略性地放置误导性信息来欺骗内容审核者的对抗性签名。在黑匣子环境下对8个商用VLLM进行了广泛的评估，结果表明，多面攻击的攻击成功率达到了61.56%，至少比最先进的攻击方法高出42.18%。



## **40. Dynamic Guided and Domain Applicable Safeguards for Enhanced Security in Large Language Models**

大型语言模型中增强安全性的动态引导和领域适用保障措施 cs.AI

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2410.17922v2) [paper-pdf](http://arxiv.org/pdf/2410.17922v2)

**Authors**: Weidi Luo, He Cao, Zijing Liu, Yu Wang, Aidan Wong, Bing Feng, Yuan Yao, Yu Li

**Abstract**: With the extensive deployment of Large Language Models (LLMs), ensuring their safety has become increasingly critical. However, existing defense methods often struggle with two key issues: (i) inadequate defense capabilities, particularly in domain-specific scenarios like chemistry, where a lack of specialized knowledge can lead to the generation of harmful responses to malicious queries. (ii) over-defensiveness, which compromises the general utility and responsiveness of LLMs. To mitigate these issues, we introduce a multi-agents-based defense framework, Guide for Defense (G4D), which leverages accurate external information to provide an unbiased summary of user intentions and analytically grounded safety response guidance. Extensive experiments on popular jailbreak attacks and benign datasets show that our G4D can enhance LLM's robustness against jailbreak attacks on general and domain-specific scenarios without compromising the model's general functionality.

摘要: 随着大型语言模型（LLM）的广泛部署，确保其安全性变得越来越重要。然而，现有的防御方法经常遇到两个关键问题：（i）防御能力不足，特别是在化学等特定领域的场景中，缺乏专业知识可能会导致对恶意查询产生有害响应。(ii)过度防御，这会损害LLM的一般实用性和响应能力。为了缓解这些问题，我们引入了一个基于多代理的防御框架--防御指南（G4 D），该框架利用准确的外部信息来提供用户意图的公正摘要和基于分析的安全响应指南。对流行越狱攻击和良性数据集的广泛实验表明，我们的G4 D可以增强LLM针对一般和特定领域场景的越狱攻击的鲁棒性，而不会损害模型的一般功能。



## **41. "Yes, My LoRD." Guiding Language Model Extraction with Locality Reinforced Distillation**

“是的，我的爱人。“利用局部强化蒸馏提取引导语言模型 cs.CR

**SubmitDate**: 2025-02-08    [abs](http://arxiv.org/abs/2409.02718v2) [paper-pdf](http://arxiv.org/pdf/2409.02718v2)

**Authors**: Zi Liang, Qingqing Ye, Yanyun Wang, Sen Zhang, Yaxin Xiao, Ronghua Li, Jianliang Xu, Haibo Hu

**Abstract**: Model extraction attacks (MEAs) on large language models (LLMs) have received increasing attention in recent research. However, existing attack methods typically adapt the extraction strategies originally developed for deep neural networks (DNNs). They neglect the underlying inconsistency between the training tasks of MEA and LLM alignment, leading to suboptimal attack performance. To tackle this issue, we propose Locality Reinforced Distillation (LoRD), a novel model extraction algorithm specifically designed for LLMs. In particular, LoRD employs a newly defined policy-gradient-style training task that utilizes the responses of victim model as the signal to guide the crafting of preference for the local model. Theoretical analyses demonstrate that I) The convergence procedure of LoRD in model extraction is consistent with the alignment procedure of LLMs, and II) LoRD can reduce query complexity while mitigating watermark protection through our exploration-based stealing. Extensive experiments validate the superiority of our method in extracting various state-of-the-art commercial LLMs. Our code is available at: https://github.com/liangzid/LoRD-MEA.

摘要: 近年来，针对大型语言模型的模型提取攻击受到越来越多的关注。然而，现有的攻击方法通常采用最初为深度神经网络(DNN)开发的提取策略。它们忽略了MEA和LLM对齐训练任务之间的潜在不一致性，导致了次优的攻击性能。为了解决这一问题，我们提出了一种新的模型提取算法LOAD，它是专门为LLM设计的。特别是，Lord采用了一种新定义的政策梯度式训练任务，该任务利用受害者模型的反应作为信号来指导对本地模型的偏好的形成。理论分析表明：1)Lord算法在模型提取中的收敛过程与LLMS算法的对齐过程是一致的；2)Lord算法在降低查询复杂度的同时，通过基于探索的窃取来减轻水印保护。大量的实验验证了该方法在提取各种最先进的商业LLM方面的优越性。我们的代码请访问：https://github.com/liangzid/LoRD-MEA.



## **42. HumorReject: Decoupling LLM Safety from Refusal Prefix via A Little Humor**

幽默：通过一点幽默将LLM安全与拒绝前置脱钩 cs.LG

**SubmitDate**: 2025-02-08    [abs](http://arxiv.org/abs/2501.13677v2) [paper-pdf](http://arxiv.org/pdf/2501.13677v2)

**Authors**: Zihui Wu, Haichang Gao, Jiacheng Luo, Zhaoxiang Liu

**Abstract**: Large Language Models (LLMs) commonly rely on explicit refusal prefixes for safety, making them vulnerable to prefix injection attacks. We introduce HumorReject, a novel data-driven approach that reimagines LLM safety by decoupling it from refusal prefixes through humor as an indirect refusal strategy. Rather than explicitly rejecting harmful instructions, HumorReject responds with contextually appropriate humor that naturally defuses potentially dangerous requests. Our approach effectively addresses common "over-defense" issues while demonstrating superior robustness against various attack vectors. Our findings suggest that improvements in training data design can be as important as the alignment algorithm itself in achieving effective LLM safety.

摘要: 大型语言模型（LLM）通常依赖显式拒绝前置来确保安全，这使得它们容易受到前置输入攻击。我们引入了幽默感，这是一种新颖的数据驱动方法，通过幽默将其与拒绝开头脱钩，重新构想了LLM的安全性，将其作为一种间接拒绝策略。幽默感并没有明确拒绝有害的指令，而是以符合上下文的幽默来回应，从而自然地化解潜在危险的请求。我们的方法有效地解决了常见的“过度防御”问题，同时展示了针对各种攻击载体的卓越鲁棒性。我们的研究结果表明，在实现有效的LLM安全性方面，训练数据设计的改进与对齐算法本身一样重要。



## **43. Topic-Based Watermarks for Large Language Models**

大型语言模型的基于主题的水印 cs.CR

Algorithms and new evaluations, 8 pages

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2404.02138v4) [paper-pdf](http://arxiv.org/pdf/2404.02138v4)

**Authors**: Alexander Nemecek, Yuzhou Jiang, Erman Ayday

**Abstract**: The indistinguishability of Large Language Model (LLM) output from human-authored content poses significant challenges, raising concerns about potential misuse of AI-generated text and its influence on future AI model training. Watermarking algorithms offer a viable solution by embedding detectable signatures into generated text. However, existing watermarking methods often entail trade-offs among attack robustness, generation quality, and additional overhead such as specialized frameworks or complex integrations. We propose a lightweight, topic-guided watermarking scheme for LLMs that partitions the vocabulary into topic-aligned token subsets. Given an input prompt, the scheme selects a relevant topic-specific token list, effectively "green-listing" semantically aligned tokens to embed robust marks while preserving the text's fluency and coherence. Experimental results across multiple LLMs and state-of-the-art benchmarks demonstrate that our method achieves comparable perplexity to industry-leading systems, including Google's SynthID-Text, yet enhances watermark robustness against paraphrasing and lexical perturbation attacks while introducing minimal performance overhead. Our approach avoids reliance on additional mechanisms beyond standard text generation pipelines, facilitating straightforward adoption, suggesting a practical path toward globally consistent watermarking of AI-generated content.

摘要: 大型语言模型(LLM)输出与人类创作的内容的不可区分构成了重大挑战，这引发了人们对人工智能生成文本的潜在滥用及其对未来人工智能模型训练的影响的担忧。水印算法通过在生成的文本中嵌入可检测的签名来提供一种可行的解决方案。然而，现有的水印方法往往需要在攻击健壮性、生成质量和额外的开销之间进行权衡，例如专门的框架或复杂的集成。我们提出了一种轻量级的、主题引导的LLMS水印方案，该方案将词汇表划分为主题对齐的令牌子集。在给定输入提示的情况下，该方案选择相关的特定于主题的标记列表，有效地对齐语义对齐的标记以嵌入健壮的标记，同时保持文本的流畅性和连贯性。在多个LLMS和最先进的基准测试上的实验结果表明，我们的方法获得了与业界领先的系统(包括Google的SynthID-Text)相当的困惑，但增强了对释义和词汇扰动攻击的水印鲁棒性，同时引入的性能开销最小。我们的方法避免了对标准文本生成管道之外的额外机制的依赖，促进了直接采用，建议了一条实现人工智能生成内容的全球一致水印的实用路径。



## **44. Watermarking Low-entropy Generation for Large Language Models: An Unbiased and Low-risk Method**

大型语言模型的水印低熵生成：一种无偏见且低风险的方法 cs.CL

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2405.14604v3) [paper-pdf](http://arxiv.org/pdf/2405.14604v3)

**Authors**: Minjia Mao, Dongjun Wei, Zeyu Chen, Xiao Fang, Michael Chau

**Abstract**: Recent advancements in large language models (LLMs) have highlighted the risk of misusing them, raising the need for accurate detection of LLM-generated content. In response, a viable solution is to inject imperceptible identifiers into LLMs, known as watermarks. Our research extends the existing watermarking methods by proposing the novel Sampling One Then Accepting (STA-1) method. STA-1 is an unbiased watermark that preserves the original token distribution in expectation and has a lower risk of producing unsatisfactory outputs in low-entropy scenarios compared to existing unbiased watermarks. In watermark detection, STA-1 does not require prompts or a white-box LLM, provides statistical guarantees, demonstrates high efficiency in detection time, and remains robust against various watermarking attacks. Experimental results on low-entropy and high-entropy datasets demonstrate that STA-1 achieves the above properties simultaneously, making it a desirable solution for watermarking LLMs. Implementation codes for this study are available online.

摘要: 大型语言模型(LLM)最近的进步突显了滥用它们的风险，提高了对LLM生成的内容进行准确检测的必要性。对此，一个可行的解决方案是将不可察觉的标识符注入LLM，即所谓的水印。我们的研究扩展了现有的数字水印方法，提出了一种新的采样后接受(STA-1)方法。STA-1是一种无偏水印，它在期望中保留了原始的令牌分布，并且与现有的无偏水印相比，在低熵情况下产生不满意输出的风险更低。在水印检测中，STA-1不需要提示或白盒LLM，提供统计保证，在检测时间上表现出高效率，并对各种水印攻击保持稳健。在低熵和高熵数据集上的实验结果表明，STA-1同时达到了上述特性，是一种理想的LLMS水印方案。这项研究的实施代码可在网上查阅。



## **45. Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning**

简单性盛行：重新思考LLM忘记学习的负偏好优化 cs.CL

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2410.07163v3) [paper-pdf](http://arxiv.org/pdf/2410.07163v3)

**Authors**: Chongyu Fan, Jiancheng Liu, Licong Lin, Jinghan Jia, Ruiqi Zhang, Song Mei, Sijia Liu

**Abstract**: This work studies the problem of large language model (LLM) unlearning, aiming to remove unwanted data influences (e.g., copyrighted or harmful content) while preserving model utility. Despite the increasing demand for unlearning, a technically-grounded optimization framework is lacking. Gradient ascent (GA)-type methods, though widely used, are suboptimal as they reverse the learning process without controlling optimization divergence (i.e., deviation from the pre-trained state), leading to risks of over-forgetting and potential model collapse. Negative preference optimization (NPO) has been proposed to address this issue and is considered one of the state-of-the-art LLM unlearning approaches. In this work, we revisit NPO and identify another critical issue: reference model bias. This bias arises from using the reference model (i.e., the model prior to unlearning) to evaluate the unlearning success, which can compromise NPO's effectiveness. Specifically, it leads to (a) uneven allocation of optimization power across forget data with varying difficulty levels and (b) ineffective gradient weight smoothing during the early stages of unlearning optimization. To overcome these challenges, we propose a simple yet effective unlearning optimization framework, called SimNPO, showing that `simplicity' in removing the reliance on a reference model (through the lens of simple preference optimization) benefits unlearning. We provide deeper insights into SimNPO's advantages through an analysis based on mixtures of Markov chains. Extensive experiments further validate SimNPO's efficacy on benchmarks like TOFU and MUSE, as well as its robustness against relearning attacks. Codes are available at https://github.com/OPTML-Group/Unlearn-Simple.

摘要: 这项工作研究了大型语言模型(LLM)的遗忘问题，目的是在保留模型效用的同时消除不需要的数据影响(例如，版权或有害内容)。尽管对遗忘的需求越来越大，但缺乏以技术为基础的优化框架。梯度上升(GA)类方法虽然被广泛使用，但由于它们逆转了学习过程而没有控制优化发散(即偏离预先训练的状态)，导致过度遗忘和潜在的模型崩溃的风险，因此是次优的。负偏好优化(NPO)就是为了解决这一问题而提出的，被认为是最先进的LLM遗忘方法之一。在这项工作中，我们重新审视了非营利组织，并确定了另一个关键问题：参考模型偏差。这种偏差源于使用参考模型(即遗忘前的模型)来评估遗忘成功，这可能会影响非营利组织的有效性。具体地说，它导致(A)在具有不同难度级别的遗忘数据之间的优化功率分配不均匀，以及(B)在遗忘优化的早期阶段无效的梯度权重平滑。为了克服这些挑战，我们提出了一个简单但有效的遗忘优化框架，称为SimNPO，表明在消除对参考模型的依赖(通过简单偏好优化的镜头)时的“简单性”有利于遗忘。我们通过基于马尔科夫链混合的分析，对SimNPO的优势提供了更深入的见解。大量的实验进一步验证了SimNPO在豆腐和缪斯等基准测试上的有效性，以及它对重新学习攻击的健壮性。有关代码，请访问https://github.com/OPTML-Group/Unlearn-Simple.



## **46. Do Unlearning Methods Remove Information from Language Model Weights?**

取消学习方法会从语言模型权重中删除信息吗？ cs.LG

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2410.08827v3) [paper-pdf](http://arxiv.org/pdf/2410.08827v3)

**Authors**: Aghyad Deeb, Fabien Roger

**Abstract**: Large Language Models' knowledge of how to perform cyber-security attacks, create bioweapons, and manipulate humans poses risks of misuse. Previous work has proposed methods to unlearn this knowledge. Historically, it has been unclear whether unlearning techniques are removing information from the model weights or just making it harder to access. To disentangle these two objectives, we propose an adversarial evaluation method to test for the removal of information from model weights: we give an attacker access to some facts that were supposed to be removed, and using those, the attacker tries to recover other facts from the same distribution that cannot be guessed from the accessible facts. We show that using fine-tuning on the accessible facts can recover 88% of the pre-unlearning accuracy when applied to current unlearning methods for information learned during pretraining, revealing the limitations of these methods in removing information from the model weights. Our results also suggest that unlearning evaluations that measure unlearning robustness on information learned during an additional fine-tuning phase may overestimate robustness compared to evaluations that attempt to unlearn information learned during pretraining.

摘要: 大型语言模型在如何执行网络安全攻击、创建生物武器和操纵人类方面的知识构成了误用的风险。以前的工作已经提出了忘记这一知识的方法。从历史上看，人们一直不清楚遗忘技术是在移除模型重量中的信息，还是只是增加了获取信息的难度。为了分离这两个目标，我们提出了一种对抗性评估方法来测试从模型权重中移除信息的情况：我们允许攻击者访问一些应该被移除的事实，并且使用这些事实，攻击者试图从相同的分布中恢复无法从可访问的事实中猜测的其他事实。结果表明，对可访问的事实进行微调可以恢复88%的预忘学习准确率，当应用于现有的遗忘方法时，这些方法在去除模型权重中的信息方面存在局限性。我们的结果还表明，与试图忘却在预训练中学习的信息的评估相比，衡量在额外微调阶段学习到的信息的遗忘健壮性的遗忘评估可能高估了健壮性。



## **47. GenBFA: An Evolutionary Optimization Approach to Bit-Flip Attacks on LLMs**

GenBFA：对LLM进行位翻转攻击的进化优化方法 cs.CR

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2411.13757v2) [paper-pdf](http://arxiv.org/pdf/2411.13757v2)

**Authors**: Sanjay Das, Swastik Bhattacharya, Souvik Kundu, Shamik Kundu, Anand Menon, Arnab Raha, Kanad Basu

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing (NLP), excelling in tasks like text generation and summarization. However, their increasing adoption in mission-critical applications raises concerns about hardware-based threats, particularly bit-flip attacks (BFAs). BFAs, enabled by fault injection methods such as Rowhammer, target model parameters in memory, compromising both integrity and performance. Identifying critical parameters for BFAs in the vast parameter space of LLMs poses significant challenges. While prior research suggests transformer-based architectures are inherently more robust to BFAs compared to traditional deep neural networks, we challenge this assumption. For the first time, we demonstrate that as few as three bit-flips can cause catastrophic performance degradation in an LLM with billions of parameters. Current BFA techniques are inadequate for exploiting this vulnerability due to the difficulty of efficiently identifying critical parameters within the immense parameter space. To address this, we propose AttentionBreaker, a novel framework tailored for LLMs that enables efficient traversal of the parameter space to identify critical parameters. Additionally, we introduce GenBFA, an evolutionary optimization strategy designed to refine the search further, isolating the most critical bits for an efficient and effective attack. Empirical results reveal the profound vulnerability of LLMs to AttentionBreaker. For example, merely three bit-flips (4.129 x 10^-9% of total parameters) in the LLaMA3-8B-Instruct 8-bit quantized (W8) model result in a complete performance collapse: accuracy on MMLU tasks drops from 67.3% to 0%, and Wikitext perplexity skyrockets from 12.6 to 4.72 x 10^5. These findings underscore the effectiveness of AttentionBreaker in uncovering and exploiting critical vulnerabilities within LLM architectures.

摘要: 大型语言模型(LLM)使自然语言处理(NLP)发生了革命性的变化，在文本生成和摘要等任务中表现出色。然而，它们在任务关键型应用中的日益采用引发了对基于硬件的威胁的担忧，特别是位翻转攻击(BFA)。由Rowhammer等故障注入方法启用的BFA以内存中的模型参数为目标，损害了完整性和性能。在LLMS庞大的参数空间中识别BFA的关键参数带来了巨大的挑战。虽然先前的研究表明，与传统的深度神经网络相比，基于变压器的体系结构对BFA具有更强的鲁棒性，但我们对这一假设提出了质疑。我们首次证明，在具有数十亿个参数的LLM中，仅三个比特翻转就会导致灾难性的性能下降。由于很难在巨大的参数空间中有效地识别关键参数，因此当前的BFA技术不足以利用该漏洞。为了解决这个问题，我们提出了AttentionBreaker，这是一个为LLMS量身定做的新框架，能够有效地遍历参数空间来识别关键参数。此外，我们还引入了GenBFA，这是一种进化优化策略，旨在进一步细化搜索，隔离最关键的比特，以实现高效和有效的攻击。实证结果揭示了LLMS对AttentionBreaker的严重脆弱性。例如，在LLAMA3-8B指令8位量化(W8)模型中，仅三次位翻转(占总参数的4.129 x 10^-9%)就会导致完全的性能崩溃：MMLU任务的准确率从67.3%下降到0%，而Wikitext的复杂性从12.6x10^5飙升到4.72x10^5。这些发现突显了AttentionBreaker在发现和利用LLM体系结构中的关键漏洞方面的有效性。



## **48. From Allies to Adversaries: Manipulating LLM Tool-Calling through Adversarial Injection**

从盟友到对手：通过对抗注入操纵LLM工具调用 cs.CR

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2412.10198v2) [paper-pdf](http://arxiv.org/pdf/2412.10198v2)

**Authors**: Haowei Wang, Rupeng Zhang, Junjie Wang, Mingyang Li, Yuekai Huang, Dandan Wang, Qing Wang

**Abstract**: Tool-calling has changed Large Language Model (LLM) applications by integrating external tools, significantly enhancing their functionality across diverse tasks. However, this integration also introduces new security vulnerabilities, particularly in the tool scheduling mechanisms of LLM, which have not been extensively studied. To fill this gap, we present ToolCommander, a novel framework designed to exploit vulnerabilities in LLM tool-calling systems through adversarial tool injection. Our framework employs a well-designed two-stage attack strategy. Firstly, it injects malicious tools to collect user queries, then dynamically updates the injected tools based on the stolen information to enhance subsequent attacks. These stages enable ToolCommander to execute privacy theft, launch denial-of-service attacks, and even manipulate business competition by triggering unscheduled tool-calling. Notably, the ASR reaches 91.67% for privacy theft and hits 100% for denial-of-service and unscheduled tool calling in certain cases. Our work demonstrates that these vulnerabilities can lead to severe consequences beyond simple misuse of tool-calling systems, underscoring the urgent need for robust defensive strategies to secure LLM Tool-calling systems.

摘要: 工具调用通过集成外部工具改变了大型语言模型(LLM)应用程序，显著增强了它们在不同任务中的功能。然而，这种集成也引入了新的安全漏洞，特别是在LLM的工具调度机制中，这些漏洞还没有得到广泛的研究。为了填补这一空白，我们提出了一种新的框架，它旨在通过敌意工具注入来利用LLM工具调用系统中的漏洞。我们的框架采用了精心设计的两阶段攻击策略。它首先注入恶意工具来收集用户查询，然后根据窃取的信息动态更新注入的工具，以加强后续攻击。这些阶段使工具指挥官能够执行隐私窃取、发起拒绝服务攻击，甚至通过触发计划外的工具调用来操纵业务竞争。值得注意的是，在某些情况下，隐私窃取的ASR达到91.67%，拒绝服务和非计划工具调用的ASR达到100%。我们的工作表明，这些漏洞可能导致严重后果，而不仅仅是简单地滥用工具调用系统，这突显了迫切需要强大的防御战略来保护LLM工具调用系统。



## **49. Enhancing Phishing Email Identification with Large Language Models**

使用大型语言模型增强网络钓鱼电子邮件识别 cs.CR

9 pages, 5 figures

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.04759v1) [paper-pdf](http://arxiv.org/pdf/2502.04759v1)

**Authors**: Catherine Lee

**Abstract**: Phishing has long been a common tactic used by cybercriminals and continues to pose a significant threat in today's digital world. When phishing attacks become more advanced and sophisticated, there is an increasing need for effective methods to detect and prevent them. To address the challenging problem of detecting phishing emails, researchers have developed numerous solutions, in particular those based on machine learning (ML) algorithms. In this work, we take steps to study the efficacy of large language models (LLMs) in detecting phishing emails. The experiments show that the LLM achieves a high accuracy rate at high precision; importantly, it also provides interpretable evidence for the decisions.

摘要: 网络钓鱼长期以来一直是网络犯罪分子使用的常见策略，并继续在当今的数字世界构成重大威胁。当网络钓鱼攻击变得更加先进和复杂时，越来越需要有效的方法来检测和预防它们。为了解决检测网络钓鱼电子邮件的挑战性问题，研究人员开发了多种解决方案，特别是基于机器学习（ML）算法的解决方案。在这项工作中，我们采取措施研究大型语言模型（LLM）在检测网络钓鱼电子邮件方面的功效。实验表明，LLM在高精度下实现了高准确率;重要的是，它还为决策提供了可解释的证据。



## **50. Jailbreak Antidote: Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Language Models**

越狱解药：通过大型语言模型中的稀疏表示调整来实现安全与效用平衡 cs.CR

Accepted by ICLR2025. url: https://openreview.net/forum?id=s20W12XTF8

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2410.02298v4) [paper-pdf](http://arxiv.org/pdf/2410.02298v4)

**Authors**: Guobin Shen, Dongcheng Zhao, Yiting Dong, Xiang He, Yi Zeng

**Abstract**: As large language models (LLMs) become integral to various applications, ensuring both their safety and utility is paramount. Jailbreak attacks, which manipulate LLMs into generating harmful content, pose significant challenges to this balance. Existing defenses, such as prompt engineering and safety fine-tuning, often introduce computational overhead, increase inference latency, and lack runtime flexibility. Moreover, overly restrictive safety measures can degrade model utility by causing refusals of benign queries. In this paper, we introduce Jailbreak Antidote, a method that enables real-time adjustment of LLM safety preferences by manipulating a sparse subset of the model's internal states during inference. By shifting the model's hidden representations along a safety direction with varying strengths, we achieve flexible control over the safety-utility balance without additional token overhead or inference delays. Our analysis reveals that safety-related information in LLMs is sparsely distributed; adjusting approximately 5% of the internal state is as effective as modifying the entire state. Extensive experiments on nine LLMs (ranging from 2 billion to 72 billion parameters), evaluated against ten jailbreak attack methods and compared with six defense strategies, validate the effectiveness and efficiency of our approach. By directly manipulating internal states during reasoning, Jailbreak Antidote offers a lightweight, scalable solution that enhances LLM safety while preserving utility, opening new possibilities for real-time safety mechanisms in widely-deployed AI systems.

摘要: 随着大型语言模型(LLM)成为各种应用程序不可或缺的一部分，确保它们的安全性和实用性是至关重要的。越狱攻击操纵LLM生成有害内容，对这种平衡构成了重大挑战。现有的防御措施，如即时工程和安全微调，通常会引入计算开销，增加推理延迟，并且缺乏运行时灵活性。此外，过于严格的安全措施可能会导致良性查询被拒绝，从而降低模型的实用性。在本文中，我们介绍了JailBreak解毒剂，这是一种通过在推理过程中操纵模型内部状态的稀疏子集来实时调整LLM安全偏好的方法。通过沿不同强度的安全方向移动模型的隐藏表示，我们在不增加令牌开销或推理延迟的情况下实现了对安全-效用平衡的灵活控制。我们的分析表明，LLMS中与安全相关的信息是稀疏分布的；调整大约5%的内部状态与修改整个状态一样有效。在9个LLM(参数从20亿到720亿)上进行了大量的实验，对10种越狱攻击方法进行了评估，并与6种防御策略进行了比较，验证了该方法的有效性和高效性。通过在推理过程中直接操纵内部状态，越狱解毒剂提供了一个轻量级、可扩展的解决方案，在增强LLM安全性的同时保留了实用性，为广泛部署的AI系统中的实时安全机制打开了新的可能性。



