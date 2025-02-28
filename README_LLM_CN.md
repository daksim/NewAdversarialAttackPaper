# Latest Large Language Model Attack Papers
**update at 2025-02-28 09:49:29**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack**

太好了，现在写一篇关于这一点的文章：渐强多转法学硕士越狱攻击 cs.CR

Accepted at USENIX Security 2025

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2404.01833v3) [paper-pdf](http://arxiv.org/pdf/2404.01833v3)

**Authors**: Mark Russinovich, Ahmed Salem, Ronen Eldan

**Abstract**: Large Language Models (LLMs) have risen significantly in popularity and are increasingly being adopted across multiple applications. These LLMs are heavily aligned to resist engaging in illegal or unethical topics as a means to avoid contributing to responsible AI harms. However, a recent line of attacks, known as jailbreaks, seek to overcome this alignment. Intuitively, jailbreak attacks aim to narrow the gap between what the model can do and what it is willing to do. In this paper, we introduce a novel jailbreak attack called Crescendo. Unlike existing jailbreak methods, Crescendo is a simple multi-turn jailbreak that interacts with the model in a seemingly benign manner. It begins with a general prompt or question about the task at hand and then gradually escalates the dialogue by referencing the model's replies progressively leading to a successful jailbreak. We evaluate Crescendo on various public systems, including ChatGPT, Gemini Pro, Gemini-Ultra, LlaMA-2 70b and LlaMA-3 70b Chat, and Anthropic Chat. Our results demonstrate the strong efficacy of Crescendo, with it achieving high attack success rates across all evaluated models and tasks. Furthermore, we present Crescendomation, a tool that automates the Crescendo attack and demonstrate its efficacy against state-of-the-art models through our evaluations. Crescendomation surpasses other state-of-the-art jailbreaking techniques on the AdvBench subset dataset, achieving 29-61% higher performance on GPT-4 and 49-71% on Gemini-Pro. Finally, we also demonstrate Crescendo's ability to jailbreak multimodal models.

摘要: 大型语言模型(LLM)的受欢迎程度显著提高，并越来越多地被多个应用程序采用。这些低成本管理机构强烈反对从事非法或不道德的话题，以此作为避免造成负责任的人工智能损害的一种手段。然而，最近一系列被称为越狱的袭击试图克服这种联系。直观地说，越狱攻击的目的是缩小模型可以做的事情和它愿意做的事情之间的差距。在这篇文章中，我们介绍了一种新的越狱攻击，称为Crescendo。与现有的越狱方法不同，Cresendo是一种简单的多转弯越狱方法，它以一种看似良性的方式与模型交互。它以关于手头任务的一般提示或问题开始，然后通过逐步参考模型的答复逐步升级对话，从而导致成功越狱。我们在不同的公共系统上对Cresendo进行了评估，包括ChatGPT、Gemini Pro、Gemini-Ultra、Llama-2 70b和Llama-3 70b Chat，以及Anthropic Chat。我们的结果证明了Crescendo的强大效力，它在所有评估的模型和任务中都实现了高攻击成功率。此外，我们提出了Crescendomation，这是一个自动化的Crescendo攻击工具，并通过我们的评估展示了它对最先进的模型的有效性。Cresendomation在AdvBtch子集数据集上超过了其他最先进的越狱技术，在GPT-4和Gemini-Pro上的性能分别提高了29%-61%和49%-71%。最后，我们还展示了Cresendo越狱多模式模型的能力。



## **2. Beyond Surface-Level Patterns: An Essence-Driven Defense Framework Against Jailbreak Attacks in LLMs**

超越表面层面模式：针对LLM越狱攻击的敏捷驱动防御框架 cs.CR

15 pages, 12 figures

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.19041v1) [paper-pdf](http://arxiv.org/pdf/2502.19041v1)

**Authors**: Shiyu Xiang, Ansen Zhang, Yanfei Cao, Yang Fan, Ronghao Chen

**Abstract**: Although Aligned Large Language Models (LLMs) are trained to refuse harmful requests, they remain vulnerable to jailbreak attacks. Unfortunately, existing methods often focus on surface-level patterns, overlooking the deeper attack essences. As a result, defenses fail when attack prompts change, even though the underlying "attack essence" remains the same. To address this issue, we introduce EDDF, an \textbf{E}ssence-\textbf{D}riven \textbf{D}efense \textbf{F}ramework Against Jailbreak Attacks in LLMs. EDDF is a plug-and-play input-filtering method and operates in two stages: 1) offline essence database construction, and 2) online adversarial query detection. The key idea behind EDDF is to extract the "attack essence" from a diverse set of known attack instances and store it in an offline vector database. Experimental results demonstrate that EDDF significantly outperforms existing methods by reducing the Attack Success Rate by at least 20\%, underscoring its superior robustness against jailbreak attacks.

摘要: 尽管统一的大型语言模型(LLM)经过了拒绝有害请求的培训，但它们仍然容易受到越狱攻击。遗憾的是，现有的方法往往只关注表层模式，而忽略了更深层次的攻击本质。其结果是，当攻击提示改变时，防御就会失败，即使潜在的“攻击本质”保持不变。为了解决这个问题，我们引入了EDDF，一个针对LLMS中越狱攻击的EDDF框架。EDDF是一种即插即用的输入过滤方法，分为两个阶段：1)离线本质数据库构建，2)在线恶意查询检测。EDDF背后的关键思想是从一组不同的已知攻击实例中提取“攻击本质”，并将其存储在脱机矢量数据库中。实验结果表明，EDDF的性能明显优于现有方法，攻击成功率降低了至少20%，突出了其对越狱攻击的卓越稳健性。



## **3. Towards Label-Only Membership Inference Attack against Pre-trained Large Language Models**

针对预训练的大型语言模型的纯标签成员推断攻击 cs.CR

Accepted by USENIX Security 2025

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.18943v1) [paper-pdf](http://arxiv.org/pdf/2502.18943v1)

**Authors**: Yu He, Boheng Li, Liu Liu, Zhongjie Ba, Wei Dong, Yiming Li, Zhan Qin, Kui Ren, Chun Chen

**Abstract**: Membership Inference Attacks (MIAs) aim to predict whether a data sample belongs to the model's training set or not. Although prior research has extensively explored MIAs in Large Language Models (LLMs), they typically require accessing to complete output logits (\ie, \textit{logits-based attacks}), which are usually not available in practice. In this paper, we study the vulnerability of pre-trained LLMs to MIAs in the \textit{label-only setting}, where the adversary can only access generated tokens (text). We first reveal that existing label-only MIAs have minor effects in attacking pre-trained LLMs, although they are highly effective in inferring fine-tuning datasets used for personalized LLMs. We find that their failure stems from two main reasons, including better generalization and overly coarse perturbation. Specifically, due to the extensive pre-training corpora and exposing each sample only a few times, LLMs exhibit minimal robustness differences between members and non-members. This makes token-level perturbations too coarse to capture such differences.   To alleviate these problems, we propose \textbf{PETAL}: a label-only membership inference attack based on \textbf{PE}r-\textbf{T}oken sem\textbf{A}ntic simi\textbf{L}arity. Specifically, PETAL leverages token-level semantic similarity to approximate output probabilities and subsequently calculate the perplexity. It finally exposes membership based on the common assumption that members are `better' memorized and have smaller perplexity. We conduct extensive experiments on the WikiMIA benchmark and the more challenging MIMIR benchmark. Empirically, our PETAL performs better than the extensions of existing label-only attacks against personalized LLMs and even on par with other advanced logit-based attacks across all metrics on five prevalent open-source LLMs.

摘要: 成员推理攻击(MIA)的目的是预测数据样本是否属于模型的训练集。尽管先前的研究已经广泛地探索了大型语言模型(LLM)中的MIA，但它们通常需要访问以完成输出日志(即，文本{基于日志的攻击})，而这在实践中通常是不可用的。在该文中，我们研究了在仅标签设置的情况下，攻击者只能访问生成的令牌(文本)的情况下，预先训练的LLMS对MIA的脆弱性。我们首先揭示了现有的仅标签MIA在攻击预先训练的LLM方面的影响很小，尽管它们在推断用于个性化LLM的微调数据集方面非常有效。我们发现，它们的失败有两个主要原因，包括较好的泛化和过粗的扰动。具体地说，由于大量的预训练语料库和每个样本只暴露几次，LLMS在成员和非成员之间表现出最小的稳健性差异。这使得令牌级的扰动过于粗略，无法捕捉到这样的差异。为了缓解这些问题，我们提出了一种基于Textbf{PE}r-\Textbf{T}Oken Sem\Textbf{A}Ntic Simi\Textbf{L}的仅标签成员关系推理攻击。具体地说，Petal利用令牌级语义相似性来近似输出概率，并随后计算困惑。它最终基于一个共同的假设来揭示成员身份，即成员的记忆力更好，困惑程度更小。我们在WikiMIA基准和更具挑战性的Mimir基准上进行了广泛的实验。根据经验，我们的Petal在针对个性化LLM的现有仅标签攻击的扩展上性能更好，甚至在五个流行的开源LLM上的所有指标上都与其他基于Logit的高级攻击相当。



## **4. JailBench: A Comprehensive Chinese Security Assessment Benchmark for Large Language Models**

JailBench：大型语言模型的全面中国安全评估基准 cs.CL

12 pages, 5 figures, accepted at PAKDD 2025

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.18935v1) [paper-pdf](http://arxiv.org/pdf/2502.18935v1)

**Authors**: Shuyi Liu, Simiao Cui, Haoran Bu, Yuming Shang, Xi Zhang

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across various applications, highlighting the urgent need for comprehensive safety evaluations. In particular, the enhanced Chinese language proficiency of LLMs, combined with the unique characteristics and complexity of Chinese expressions, has driven the emergence of Chinese-specific benchmarks for safety assessment. However, these benchmarks generally fall short in effectively exposing LLM safety vulnerabilities. To address the gap, we introduce JailBench, the first comprehensive Chinese benchmark for evaluating deep-seated vulnerabilities in LLMs, featuring a refined hierarchical safety taxonomy tailored to the Chinese context. To improve generation efficiency, we employ a novel Automatic Jailbreak Prompt Engineer (AJPE) framework for JailBench construction, which incorporates jailbreak techniques to enhance assessing effectiveness and leverages LLMs to automatically scale up the dataset through context-learning. The proposed JailBench is extensively evaluated over 13 mainstream LLMs and achieves the highest attack success rate against ChatGPT compared to existing Chinese benchmarks, underscoring its efficacy in identifying latent vulnerabilities in LLMs, as well as illustrating the substantial room for improvement in the security and trustworthiness of LLMs within the Chinese context. Our benchmark is publicly available at https://github.com/STAIR-BUPT/JailBench.

摘要: 大型语言模型(LLM)在各种应用中表现出了非凡的能力，突显了对综合安全评估的迫切需要。特别是，低成本管理系统中文水平的提高，加上中文表达的独特性和复杂性，推动了针对中文的安全评估基准的出现。然而，这些基准通常不能有效地暴露LLM安全漏洞。为了弥补这一差距，我们引入了JailBch，这是第一个全面的中国基准，用于评估低成本管理中的深层漏洞，其特点是根据中国背景定制了精细化的层次化安全分类。为了提高生成效率，我们采用了一种新颖的自动越狱提示工程师(AJPE)框架来构建JailB边，该框架结合了越狱技术来增强评估有效性，并利用LLMS通过上下文学习来自动扩大数据集的规模。建议的JailBtch在13个主流LLMS上进行了广泛的评估，与现有的中国基准相比，对ChatGPT的攻击成功率最高，突显了其在识别LLMS潜在漏洞方面的有效性，并说明了LLMS在中国背景下的安全性和可信度有很大的改进空间。我们的基准测试在https://github.com/STAIR-BUPT/JailBench.上公开提供



## **5. Defense Against Prompt Injection Attack by Leveraging Attack Techniques**

利用攻击技术防御即时注入攻击 cs.CR

9 pages

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2411.00459v3) [paper-pdf](http://arxiv.org/pdf/2411.00459v3)

**Authors**: Yulin Chen, Haoran Li, Zihao Zheng, Yangqiu Song, Dekai Wu, Bryan Hooi

**Abstract**: With the advancement of technology, large language models (LLMs) have achieved remarkable performance across various natural language processing (NLP) tasks, powering LLM-integrated applications like Microsoft Copilot. However, as LLMs continue to evolve, new vulnerabilities, especially prompt injection attacks arise. These attacks trick LLMs into deviating from the original input instructions and executing the attacker's instructions injected in data content, such as retrieved results. Recent attack methods leverage LLMs' instruction-following abilities and their inabilities to distinguish instructions injected in the data content, and achieve a high attack success rate (ASR). When comparing the attack and defense methods, we interestingly find that they share similar design goals, of inducing the model to ignore unwanted instructions and instead to execute wanted instructions. Therefore, we raise an intuitive question: Could these attack techniques be utilized for defensive purposes? In this paper, we invert the intention of prompt injection methods to develop novel defense methods based on previous training-free attack methods, by repeating the attack process but with the original input instruction rather than the injected instruction. Our comprehensive experiments demonstrate that our defense techniques outperform existing training-free defense approaches, achieving state-of-the-art results.

摘要: 随着技术的进步，大语言模型(LLM)在各种自然语言处理(NLP)任务中取得了显著的性能，支持Microsoft Copilot等LLM集成应用程序。然而，随着LLMS的不断发展，出现了新的漏洞，特别是即时注入攻击。这些攻击欺骗LLM偏离原始输入指令，并执行注入数据内容的攻击者指令，例如检索的结果。最近的攻击方法利用LLMS的指令跟随能力和它们无法区分注入到数据内容中的指令的能力，实现了高攻击成功率(ASR)。当比较攻击和防御方法时，我们有趣地发现它们有相似的设计目标，都是诱导模型忽略不想要的指令，而是执行想要的指令。因此，我们提出了一个直观的问题：这些攻击技术是否可以用于防御目的？在本文中，我们反转了快速注入方法的意图，在以前的免训练攻击方法的基础上，通过重复攻击过程来开发新的防御方法，但使用的是原始输入指令而不是注入指令。我们的综合实验表明，我们的防御技术优于现有的免训练防御方法，取得了最先进的结果。



## **6. Stealthy Backdoor Attack in Self-Supervised Learning Vision Encoders for Large Vision Language Models**

大视觉语言模型的自我监督学习视觉编码器中的秘密后门攻击 cs.CV

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.18290v2) [paper-pdf](http://arxiv.org/pdf/2502.18290v2)

**Authors**: Zhaoyi Liu, Huan Zhang

**Abstract**: Self-supervised learning (SSL) vision encoders learn high-quality image representations and thus have become a vital part of developing vision modality of large vision language models (LVLMs). Due to the high cost of training such encoders, pre-trained encoders are widely shared and deployed into many LVLMs, which are security-critical or bear societal significance. Under this practical scenario, we reveal a new backdoor threat that significant visual hallucinations can be induced into these LVLMs by merely compromising vision encoders. Because of the sharing and reuse of these encoders, many downstream LVLMs may inherit backdoor behaviors from encoders, leading to widespread backdoors. In this work, we propose BadVision, the first method to exploit this vulnerability in SSL vision encoders for LVLMs with novel trigger optimization and backdoor learning techniques. We evaluate BadVision on two types of SSL encoders and LVLMs across eight benchmarks. We show that BadVision effectively drives the LVLMs to attacker-chosen hallucination with over 99% attack success rate, causing a 77.6% relative visual understanding error while maintaining the stealthiness. SoTA backdoor detection methods cannot detect our attack effectively.

摘要: 自监督学习(SSL)视觉编码者学习高质量的图像表征，因此成为开发大型视觉语言模型(LVLMS)视觉通道的重要组成部分。由于培训这类编码器的成本很高，预先训练的编码器被广泛共享并部署到许多安全关键或具有社会意义的LVLM中。在这种实际情况下，我们揭示了一种新的后门威胁，即仅仅通过损害视觉编码器就可以在这些LVLM中诱导出显著的视觉幻觉。由于这些编码器的共享和重用，许多下游的LVLM可能会继承编码器的后门行为，导致广泛的后门。在这项工作中，我们提出了BadVision，这是第一个通过新颖的触发优化和后门学习技术来利用LVLM的SSL视觉编码器中的漏洞的方法。我们在八个基准测试中评估了BadVision在两种类型的SSL编码器和LVLM上的性能。结果表明，BadVision在保持隐蔽性的同时，以99%以上的攻击成功率有效地驱动了LVLM进入攻击者选择的幻觉，导致了77.6%的相对视觉理解错误。SOTA后门检测方法不能有效检测到我们的攻击。



## **7. Topic-FlipRAG: Topic-Orientated Adversarial Opinion Manipulation Attacks to Retrieval-Augmented Generation Models**

Top-FlipRAG：针对检索增强生成模型的面向主题的对抗性观点操纵攻击 cs.CL

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.01386v2) [paper-pdf](http://arxiv.org/pdf/2502.01386v2)

**Authors**: Yuyang Gong, Zhuo Chen, Miaokun Chen, Fengchang Yu, Wei Lu, Xiaofeng Wang, Xiaozhong Liu, Jiawei Liu

**Abstract**: Retrieval-Augmented Generation (RAG) systems based on Large Language Models (LLMs) have become essential for tasks such as question answering and content generation. However, their increasing impact on public opinion and information dissemination has made them a critical focus for security research due to inherent vulnerabilities. Previous studies have predominantly addressed attacks targeting factual or single-query manipulations. In this paper, we address a more practical scenario: topic-oriented adversarial opinion manipulation attacks on RAG models, where LLMs are required to reason and synthesize multiple perspectives, rendering them particularly susceptible to systematic knowledge poisoning. Specifically, we propose Topic-FlipRAG, a two-stage manipulation attack pipeline that strategically crafts adversarial perturbations to influence opinions across related queries. This approach combines traditional adversarial ranking attack techniques and leverages the extensive internal relevant knowledge and reasoning capabilities of LLMs to execute semantic-level perturbations. Experiments show that the proposed attacks effectively shift the opinion of the model's outputs on specific topics, significantly impacting user information perception. Current mitigation methods cannot effectively defend against such attacks, highlighting the necessity for enhanced safeguards for RAG systems, and offering crucial insights for LLM security research.

摘要: 基于大型语言模型(LLMS)的检索-增强生成(RAG)系统已经成为诸如问题回答和内容生成等任务的关键。然而，由于其固有的脆弱性，它们对舆论和信息传播的影响越来越大，使其成为安全研究的关键焦点。以前的研究主要针对针对事实或单一查询操作的攻击。在本文中，我们讨论了一个更实际的场景：针对RAG模型的面向主题的对抗性意见操纵攻击，其中要求LLM推理和综合多个视角，使它们特别容易受到系统性知识中毒的影响。具体地说，我们提出了Theme-FlipRAG，这是一种两阶段操纵攻击管道，战略性地制造对抗性扰动来影响相关查询的观点。该方法结合了传统的对抗性排序攻击技术，并利用LLMS丰富的内部相关知识和推理能力来执行语义级的扰动。实验表明，提出的攻击有效地改变了模型输出对特定主题的看法，显著影响了用户对信息的感知。目前的缓解方法无法有效防御此类攻击，这突显了加强RAG系统安全保障的必要性，并为LLM安全研究提供了至关重要的见解。



## **8. CLIPure: Purification in Latent Space via CLIP for Adversarially Robust Zero-Shot Classification**

CLIPure：通过CLIP在潜空间中净化，以实现对抗鲁棒零镜头分类 cs.CV

accepted by ICLR 2025

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.18176v1) [paper-pdf](http://arxiv.org/pdf/2502.18176v1)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: In this paper, we aim to build an adversarially robust zero-shot image classifier. We ground our work on CLIP, a vision-language pre-trained encoder model that can perform zero-shot classification by matching an image with text prompts ``a photo of a <class-name>.''. Purification is the path we choose since it does not require adversarial training on specific attack types and thus can cope with any foreseen attacks. We then formulate purification risk as the KL divergence between the joint distributions of the purification process of denoising the adversarial samples and the attack process of adding perturbations to benign samples, through bidirectional Stochastic Differential Equations (SDEs). The final derived results inspire us to explore purification in the multi-modal latent space of CLIP. We propose two variants for our CLIPure approach: CLIPure-Diff which models the likelihood of images' latent vectors with the DiffusionPrior module in DaLLE-2 (modeling the generation process of CLIP's latent vectors), and CLIPure-Cos which models the likelihood with the cosine similarity between the embeddings of an image and ``a photo of a.''. As far as we know, CLIPure is the first purification method in multi-modal latent space and CLIPure-Cos is the first purification method that is not based on generative models, which substantially improves defense efficiency. We conducted extensive experiments on CIFAR-10, ImageNet, and 13 datasets that previous CLIP-based defense methods used for evaluating zero-shot classification robustness. Results show that CLIPure boosts the SOTA robustness by a large margin, e.g., from 71.7% to 91.1% on CIFAR10, from 59.6% to 72.6% on ImageNet, and 108% relative improvements of average robustness on the 13 datasets over previous SOTA. The code is available at https://github.com/TMLResearchGroup-CAS/CLIPure.

摘要: 在这篇文章中，我们的目标是建立一个对抗性稳健的零镜头图像分类器。我们的工作基于CLIP，这是一个视觉语言预先训练的编码器模型，它可以通过将图像与文本提示进行匹配来执行零镜头分类。净化是我们选择的路径，因为它不需要针对特定攻击类型的对抗性训练，因此可以应对任何可预见的攻击。然后，我们通过双向随机微分方程(SDE)将净化风险表示为对敌方样本去噪的净化过程和对良性样本添加扰动的攻击过程的联合分布之间的KL发散。最终得出的结果启发我们去探索CLIP的多峰潜伏空间中的净化。我们为我们的CLIPure方法提出了两种变体：CLIPure-Diff和CLIPure-Cos，CLIPure-Diff使用DALE-2中的DiffusionPrior模块(对剪辑的潜在向量的生成过程进行建模)来模拟图像的潜在向量的可能性，CLIPure-Cos使用图像的嵌入和“a的照片”之间的余弦相似性来建模可能性。据我们所知，CLIPure是第一个在多峰潜在空间中进行净化的方法，而CLIPure-Cos是第一个不基于产生式模型的净化方法，大大提高了防御效率。我们在CIFAR-10、ImageNet和13个数据集上进行了广泛的实验，这些数据集是以前基于剪辑的防御方法用于评估零镜头分类稳健性的。结果表明，CLIPure在很大程度上提高了SOTA的健壮性，例如，在CIFAR10上从71.7%提高到91.1%，在ImageNet上从59.6%提高到72.6%，在13个数据集上的平均健壮性比以前的SOTA提高了108%。代码可在https://github.com/TMLResearchGroup-CAS/CLIPure.上获得



## **9. Towards Robust and Secure Embodied AI: A Survey on Vulnerabilities and Attacks**

迈向强大和安全的人工智能：关于漏洞和攻击的调查 cs.CR

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.13175v2) [paper-pdf](http://arxiv.org/pdf/2502.13175v2)

**Authors**: Wenpeng Xing, Minghao Li, Mohan Li, Meng Han

**Abstract**: Embodied AI systems, including robots and autonomous vehicles, are increasingly integrated into real-world applications, where they encounter a range of vulnerabilities stemming from both environmental and system-level factors. These vulnerabilities manifest through sensor spoofing, adversarial attacks, and failures in task and motion planning, posing significant challenges to robustness and safety. Despite the growing body of research, existing reviews rarely focus specifically on the unique safety and security challenges of embodied AI systems. Most prior work either addresses general AI vulnerabilities or focuses on isolated aspects, lacking a dedicated and unified framework tailored to embodied AI. This survey fills this critical gap by: (1) categorizing vulnerabilities specific to embodied AI into exogenous (e.g., physical attacks, cybersecurity threats) and endogenous (e.g., sensor failures, software flaws) origins; (2) systematically analyzing adversarial attack paradigms unique to embodied AI, with a focus on their impact on perception, decision-making, and embodied interaction; (3) investigating attack vectors targeting large vision-language models (LVLMs) and large language models (LLMs) within embodied systems, such as jailbreak attacks and instruction misinterpretation; (4) evaluating robustness challenges in algorithms for embodied perception, decision-making, and task planning; and (5) proposing targeted strategies to enhance the safety and reliability of embodied AI systems. By integrating these dimensions, we provide a comprehensive framework for understanding the interplay between vulnerabilities and safety in embodied AI.

摘要: 包括机器人和自动驾驶车辆在内的具体化人工智能系统正越来越多地融入现实世界的应用程序，在这些应用程序中，它们遇到了一系列源于环境和系统层面因素的漏洞。这些漏洞表现为传感器欺骗、对抗性攻击以及任务和运动规划中的失败，对健壮性和安全性构成了重大挑战。尽管研究的主体越来越多，但现有的审查很少专门关注嵌入式人工智能系统的独特安全和安保挑战。大多数以前的工作要么解决了一般的人工智能漏洞，要么专注于孤立的方面，缺乏一个专门为体现的人工智能量身定做的统一框架。本调查通过以下方式填补这一关键空白：(1)将特定于具身人工智能的漏洞分为外源性(如物理攻击、网络安全威胁)和内源性(如传感器故障、软件缺陷)来源；(2)系统分析具身人工智能特有的对抗性攻击范式，重点关注它们对感知、决策和具身交互的影响；(3)调查针对具身系统内的大视觉语言模型(LVLM)和大语言模型(LMS)的攻击向量，如越狱攻击和指令曲解；(4)评估体现感知、决策和任务规划算法中的健壮性挑战；(5)提出有针对性的策略，以提高体现人工智能系统的安全性和可靠性。通过整合这些维度，我们提供了一个全面的框架，用于理解体现的人工智能中漏洞和安全之间的相互作用。



## **10. Efficient Safety Retrofitting Against Jailbreaking for LLMs**

针对LLM越狱的高效安全改造 cs.CL

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.13603v2) [paper-pdf](http://arxiv.org/pdf/2502.13603v2)

**Authors**: Dario Garcia-Gasulla, Adrian Tormos, Anna Arias-Duart, Daniel Hinjos, Oscar Molina-Sedano, Ashwin Kumar Gururajan, Maria Eugenia Cardello

**Abstract**: Direct Preference Optimization (DPO) is an efficient alignment technique that steers LLMs towards preferable outputs by training on preference data, bypassing the need for explicit reward models. Its simplicity enables easy adaptation to various domains and safety requirements. This paper examines DPO's effectiveness in model safety against jailbreaking attacks while minimizing data requirements and training costs. We introduce Egida, a dataset expanded from multiple sources, which includes 27 different safety topics and 18 different attack styles, complemented with synthetic and human labels. This data is used to boost the safety of state-of-the-art LLMs (Llama-3.1-8B/70B-Instruct, Qwen-2.5-7B/72B-Instruct) across topics and attack styles. In addition to safety evaluations, we assess their post-alignment performance degradation in general purpose tasks, and their tendency to over refusal. Following the proposed methodology, trained models reduce their Attack Success Rate by 10%-30%, using small training efforts (2,000 samples) with low computational cost (3\$ for 8B models, 20\$ for 72B models). Safety aligned models generalize to unseen topics and attack styles, with the most successful attack style reaching a success rate around 5%. Size and family are found to strongly influence model malleability towards safety, pointing at the importance of pre-training choices. To validate our findings, a large independent assessment of human preference agreement with Llama-Guard-3-8B is conducted by the authors and the associated dataset Egida-HSafe is released. Overall, this study illustrates how affordable and accessible it is to enhance LLM safety using DPO while outlining its current limitations. All datasets and models are released to enable reproducibility and further research.

摘要: 直接偏好优化(DPO)是一种有效的对齐技术，它通过对偏好数据的训练来引导LLM朝着更好的输出方向前进，而不需要显式的奖励模型。它的简单性使其能够轻松适应各种域和安全要求。本文研究了DPO在防止越狱攻击的模型安全方面的有效性，同时将数据需求和培训成本降至最低。我们介绍了Egida，这是一个从多个来源扩展的数据集，包括27个不同的安全主题和18个不同的攻击风格，并辅之以合成标签和人工标签。这些数据用于提高最先进的LLMS(LAMA-3.1-8B/70B-指令，QWEN-2.5-7B/72B-指令)跨主题和攻击风格的安全性。除了安全评估外，我们还评估了他们在一般目的任务中对齐后的表现降级，以及他们过度拒绝的倾向。按照所提出的方法，训练模型的攻击成功率降低了10%-30%，使用较小的训练工作量(2000个样本)和较低的计算成本(8B模型3个，72B模型20个)。安全对齐的模型概括为未知的主题和攻击风格，最成功的攻击风格的成功率约为5%。尺寸和家庭被发现强烈影响模型的安全延展性，这表明了培训前选择的重要性。为了验证我们的发现，作者对人类偏好与Llama-Guard-3-8B的一致性进行了大规模的独立评估，并发布了相关的数据集Egida-HSafe。总体而言，这项研究说明了使用DPO增强LLM安全性的负担能力和可获得性，同时概述了其当前的局限性。所有数据集和模型都被公布，以便于重现性和进一步研究。



## **11. S$^4$ST: A Strong, Self-transferable, faSt, and Simple Scale Transformation for Transferable Targeted Attack**

新元' 4$ST：强大、可自我转移、快速且简单的可转移定向攻击规模化转型 cs.CR

16 pages, 18 figures

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2410.13891v2) [paper-pdf](http://arxiv.org/pdf/2410.13891v2)

**Authors**: Yongxiang Liu, Bowen Peng, Li Liu, Xiang Li

**Abstract**: Transferable Targeted Attacks (TTAs), which aim to deceive black-box models into predicting specific erroneous labels, face significant challenges due to severe overfitting to surrogate models. Although modifying image features to generate robust semantic patterns of the target class is a promising approach, existing methods heavily rely on large-scale additional data. This dependence undermines the fair evaluation of TTA threats, potentially leading to a false sense of security or unnecessary overreactions. In this paper, we introduce two blind measures, surrogate self-alignment and self-transferability, to analyze the effectiveness and correlations of basic transformations, to enhance data-free attacks under strict black-box constraints. Our findings challenge conventional assumptions: (1) Attacking simple scaling transformations uniquely enhances targeted transferability, outperforming other basic transformations and rivaling leading complex methods; (2) Geometric and color transformations exhibit high internal redundancy despite weak inter-category correlations. These insights drive the design and tuning of S4ST (Strong, Self-transferable, faSt, Simple Scale Transformation), which integrates dimensionally consistent scaling, complementary low-redundancy transformations, and block-wise operations. Extensive experiments on the ImageNet-Compatible dataset demonstrate that S4ST achieves a 77.7% average targeted success rate (tSuc), surpassing existing transformations (+17.2% over H-Aug with only 26% computational time) and SOTA TTA solutions (+6.2% over SASD-WS with 1.2M samples for post-training). Notably, it attains 69.6% and 55.3% average tSuc against three commercial APIs and vision-language models, respectively. This work establishes a new SOTA for TTAs, highlights their potential threats, and calls for a reevaluation of the data dependency in achieving targeted transferability.

摘要: 可转移目标攻击(TTA)旨在欺骗黑盒模型预测特定的错误标签，但由于对代理模型的严重过度拟合，TTA面临着巨大的挑战。虽然修改图像特征以生成目标类的稳健语义模式是一种很有前途的方法，但现有方法严重依赖于大规模的附加数据。这种依赖破坏了对TTA威胁的公平评估，可能导致错误的安全感或不必要的过度反应。本文引入了代理自对齐和自转移两个盲度量，分析了基本变换的有效性和相关性，从而在严格的黑盒约束下增强了数据自由攻击。我们的发现挑战了传统的假设：(1)攻击简单的尺度变换独特地提高了目标可转移性，优于其他基本变换，并可与领先的复杂方法相媲美；(2)尽管类别间相关性较弱，但几何和颜色变换显示出高的内部冗余性。这些见解推动了S4ST(强大、可自我转移、快速、简单的比例转换)的设计和调整，S4ST集成了维度一致的比例调整、互补的低冗余转换和按块操作。在ImageNet兼容数据集上的广泛实验表明，S4ST达到了77.7%的平均目标成功率(TSuc)，超过了现有的转换(在H-AUG上+17.2%，计算时间仅为26%)和SOTA TTA解决方案(在120万个训练后样本的SASD-WS上+6.2%)。值得注意的是，与三个商业API和视觉语言模型相比，它分别获得了69.6%和55.3%的平均tSuc。这项工作为TTA建立了新的SOTA，强调了它们的潜在威胁，并呼吁重新评估实现有针对性的可转移性的数据依赖性。



## **12. The Hidden Risks of Large Reasoning Models: A Safety Assessment of R1**

大型推理模型的隐藏风险：R1的安全评估 cs.CY

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.12659v3) [paper-pdf](http://arxiv.org/pdf/2502.12659v3)

**Authors**: Kaiwen Zhou, Chengzhi Liu, Xuandong Zhao, Shreedhar Jangam, Jayanth Srinivasa, Gaowen Liu, Dawn Song, Xin Eric Wang

**Abstract**: The rapid development of large reasoning models, such as OpenAI-o3 and DeepSeek-R1, has led to significant improvements in complex reasoning over non-reasoning large language models~(LLMs). However, their enhanced capabilities, combined with the open-source access of models like DeepSeek-R1, raise serious safety concerns, particularly regarding their potential for misuse. In this work, we present a comprehensive safety assessment of these reasoning models, leveraging established safety benchmarks to evaluate their compliance with safety regulations. Furthermore, we investigate their susceptibility to adversarial attacks, such as jailbreaking and prompt injection, to assess their robustness in real-world applications. Through our multi-faceted analysis, we uncover four key findings: (1) There is a significant safety gap between the open-source R1 models and the o3-mini model, on both safety benchmark and attack, suggesting more safety effort on R1 is needed. (2) The distilled reasoning model shows poorer safety performance compared to its safety-aligned base models. (3) The stronger the model's reasoning ability, the greater the potential harm it may cause when answering unsafe questions. (4) The thinking process in R1 models pose greater safety concerns than their final answers. Our study provides insights into the security implications of reasoning models and highlights the need for further advancements in R1 models' safety to close the gap.

摘要: 大型推理模型的快速发展，如OpenAI-03和DeepSeek-R1，使得复杂推理相对于非推理的大型语言模型有了显著的改进。然而，它们增强的能力，再加上DeepSeek-R1等型号的开源访问，引发了严重的安全问题，特别是它们可能被滥用的问题。在这项工作中，我们提出了这些推理模型的全面安全评估，利用已建立的安全基准来评估它们是否符合安全法规。此外，我们调查了它们对敌意攻击的敏感性，例如越狱和快速注入，以评估它们在现实世界应用程序中的健壮性。通过多方面的分析，我们发现了四个重要的发现：(1)无论是在安全基准上还是在攻击上，开源的R1型号和03-mini型号之间都存在着显著的安全差距，这表明需要在R1上做出更多的安全努力。(2)与安全对齐的基本模型相比，精炼推理模型的安全性能较差。(3)模型的推理能力越强，在回答不安全问题时可能造成的潜在危害就越大。(4)与最终答案相比，R1模型的思维过程带来了更大的安全顾虑。我们的研究为推理模型的安全含义提供了见解，并强调了在R1模型的安全性方面进一步改进的必要性，以缩小差距。



## **13. MM-PoisonRAG: Disrupting Multimodal RAG with Local and Global Poisoning Attacks**

MM-PoisonRAG：通过本地和全球中毒攻击扰乱多模式RAG cs.LG

Code is available at https://github.com/HyeonjeongHa/MM-PoisonRAG

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.17832v1) [paper-pdf](http://arxiv.org/pdf/2502.17832v1)

**Authors**: Hyeonjeong Ha, Qiusi Zhan, Jeonghwan Kim, Dimitrios Bralios, Saikrishna Sanniboina, Nanyun Peng, Kai-wei Chang, Daniel Kang, Heng Ji

**Abstract**: Multimodal large language models (MLLMs) equipped with Retrieval Augmented Generation (RAG) leverage both their rich parametric knowledge and the dynamic, external knowledge to excel in tasks such as Question Answering. While RAG enhances MLLMs by grounding responses in query-relevant external knowledge, this reliance poses a critical yet underexplored safety risk: knowledge poisoning attacks, where misinformation or irrelevant knowledge is intentionally injected into external knowledge bases to manipulate model outputs to be incorrect and even harmful. To expose such vulnerabilities in multimodal RAG, we propose MM-PoisonRAG, a novel knowledge poisoning attack framework with two attack strategies: Localized Poisoning Attack (LPA), which injects query-specific misinformation in both text and images for targeted manipulation, and Globalized Poisoning Attack (GPA) to provide false guidance during MLLM generation to elicit nonsensical responses across all queries. We evaluate our attacks across multiple tasks, models, and access settings, demonstrating that LPA successfully manipulates the MLLM to generate attacker-controlled answers, with a success rate of up to 56% on MultiModalQA. Moreover, GPA completely disrupts model generation to 0% accuracy with just a single irrelevant knowledge injection. Our results highlight the urgent need for robust defenses against knowledge poisoning to safeguard multimodal RAG frameworks.

摘要: 配备了检索增强生成(RAG)的多通道大型语言模型(MLLMS)利用其丰富的参数知识和动态的外部知识来在问答等任务中脱颖而出。虽然RAG通过将响应建立在与查询相关的外部知识中来增强MLLS，但这种依赖构成了一个关键但未被开发的安全风险：知识中毒攻击，即故意将错误信息或无关知识注入外部知识库，以操纵不正确甚至有害的模型输出。为了暴露多模式RAG中的这些漏洞，我们提出了一种新的知识中毒攻击框架MM-PoisonRAG，该框架具有两种攻击策略：局部中毒攻击(LPA)和全局中毒攻击(GPA)，前者在文本和图像中注入特定于查询的错误信息以进行有针对性的操作，后者在MLLM生成过程中提供错误指导，以引发跨所有查询的无意义响应。我们跨多个任务、模型和访问设置评估我们的攻击，证明LPA成功操纵MLLM生成攻击者控制的答案，在多模式QA上的成功率高达56%。此外，GPA只需注入一次无关的知识，就能完全中断模型生成，准确率为0%。我们的结果突出表明，迫切需要针对知识中毒采取强有力的防御措施，以保护多式联运RAG框架。



## **14. Towards Effective Evaluations and Comparisons for LLM Unlearning Methods**

LLM取消学习方法的有效评估和比较 cs.LG

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2406.09179v2) [paper-pdf](http://arxiv.org/pdf/2406.09179v2)

**Authors**: Qizhou Wang, Bo Han, Puning Yang, Jianing Zhu, Tongliang Liu, Masashi Sugiyama

**Abstract**: The imperative to eliminate undesirable data memorization underscores the significance of machine unlearning for large language models (LLMs). Recent research has introduced a series of promising unlearning methods, notably boosting the practical significance of the field. Nevertheless, adopting a proper evaluation framework to reflect the true unlearning efficacy is also essential yet has not received adequate attention. This paper seeks to refine the evaluation of LLM unlearning by addressing two key challenges -- a) the robustness of evaluation metrics and b) the trade-offs between competing goals. The first challenge stems from findings that current metrics are susceptible to various red teaming scenarios. It indicates that they may not reflect the true extent of knowledge retained by LLMs but rather tend to mirror superficial model behaviors, thus prone to attacks. We address this issue by devising and assessing a series of candidate metrics, selecting the most robust ones under various types of attacks. The second challenge arises from the conflicting goals of eliminating unwanted knowledge while retaining those of others. This trade-off between unlearning and retention often fails to conform the Pareto frontier, rendering it subtle to compare the efficacy between methods that excel only in either unlearning or retention. We handle this issue by proposing a calibration method that can restore the original performance on non-targeted data after unlearning, thereby allowing us to focus exclusively on assessing the strength of unlearning. Our evaluation framework notably enhances the effectiveness when assessing and comparing various LLM unlearning methods, further allowing us to benchmark existing works, identify their proper hyper-parameters, and explore new tricks to enhance their practical efficacy.

摘要: 消除不受欢迎的数据记忆势在必行，这突显了机器遗忘对于大型语言模型(LLM)的重要性。最近的研究介绍了一系列有前景的遗忘方法，特别是提高了该领域的实际意义。然而，采用适当的评估框架来反映真正的遗忘效能也是至关重要的，但尚未得到足够的重视。本文试图通过解决两个关键挑战--a)评价指标的稳健性和b)相互竞争的目标之间的权衡--来改进LLM遗忘的评价。第一个挑战源于发现，当前的指标容易受到各种红队场景的影响。这表明它们可能不能反映LLMS保留的知识的真实范围，而是倾向于反映表面的模型行为，因此容易受到攻击。我们通过设计和评估一系列候选指标来解决这个问题，选择在各种类型的攻击下最健壮的指标。第二个挑战来自消除无用知识和保留他人知识这两个相互冲突的目标。这种遗忘和保留之间的权衡往往不符合帕累托边界，这使得比较只擅长遗忘或保留的方法的有效性变得微妙起来。我们通过提出一种校正方法来处理这个问题，该方法可以在遗忘后恢复对非目标数据的原始性能，从而使我们能够专注于评估遗忘的强度。我们的评估框架显著提高了评估和比较各种LLM遗忘方法的有效性，进一步允许我们对现有工作进行基准测试，确定其适当的超参数，并探索新的技巧来提高其实际效果。



## **15. Design and implementation of a distributed security threat detection system integrating federated learning and multimodal LLM**

集成联邦学习和多模式LLM的分布式安全威胁检测系统的设计与实现 cs.CR

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.17763v1) [paper-pdf](http://arxiv.org/pdf/2502.17763v1)

**Authors**: Yuqing Wang, Xiao Yang

**Abstract**: Traditional security protection methods struggle to address sophisticated attack vectors in large-scale distributed systems, particularly when balancing detection accuracy with data privacy concerns. This paper presents a novel distributed security threat detection system that integrates federated learning with multimodal large language models (LLMs). Our system leverages federated learning to ensure data privacy while employing multimodal LLMs to process heterogeneous data sources including network traffic, system logs, images, and sensor data. Experimental evaluation on a 10TB distributed dataset demonstrates that our approach achieves 96.4% detection accuracy, outperforming traditional baseline models by 4.1 percentage points. The system reduces both false positive and false negative rates by 1.8 and 2.4 percentage points respectively. Performance analysis shows that our system maintains efficient processing capabilities in distributed environments, requiring 180 seconds for model training and 3.8 seconds for threat detection across the distributed network. These results demonstrate significant improvements in detection accuracy and computational efficiency while preserving data privacy, suggesting strong potential for real-world deployment in large-scale security systems.

摘要: 传统的安全保护方法难以解决大规模分布式系统中复杂的攻击载体，特别是在平衡检测精度和数据隐私问题时。提出了一种将联邦学习和多通道大语言模型相结合的分布式安全威胁检测系统。我们的系统利用联合学习来确保数据隐私，同时使用多模式LLMS来处理不同的数据源，包括网络流量、系统日志、图像和传感器数据。在一个10TB的分布式数据集上的实验评估表明，该方法达到了96.4%的检测正确率，比传统的基线模型高出4.1个百分点。该系统使假阳性率和假阴性率分别降低1.8和2.4个百分点。性能分析表明，我们的系统在分布式环境中保持了高效的处理能力，模型训练需要180秒，跨分布式网络的威胁检测需要3.8秒。这些结果表明，在保护数据隐私的同时，检测精度和计算效率都有了显著的提高，这表明在大规模安全系统中实际部署的潜力很大。



## **16. Proactive Privacy Amnesia for Large Language Models: Safeguarding PII with Negligible Impact on Model Utility**

大型语言模型的主动隐私咨询：保护PRI，对模型实用性的影响可忽略不计 cs.CL

ICLR'25 Poster. Project page and code is available at  https://ppa-iclr2025.my.canva.site/

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17591v1) [paper-pdf](http://arxiv.org/pdf/2502.17591v1)

**Authors**: Martin Kuo, Jingyang Zhang, Jianyi Zhang, Minxue Tang, Louis DiValentin, Aolin Ding, Jingwei Sun, William Chen, Amin Hass, Tianlong Chen, Yiran Chen, Hai Li

**Abstract**: With the rise of large language models (LLMs), increasing research has recognized their risk of leaking personally identifiable information (PII) under malicious attacks. Although efforts have been made to protect PII in LLMs, existing methods struggle to balance privacy protection with maintaining model utility. In this paper, inspired by studies of amnesia in cognitive science, we propose a novel approach, Proactive Privacy Amnesia (PPA), to safeguard PII in LLMs while preserving their utility. This mechanism works by actively identifying and forgetting key memories most closely associated with PII in sequences, followed by a memory implanting using suitable substitute memories to maintain the LLM's functionality. We conduct evaluations across multiple models to protect common PII, such as phone numbers and physical addresses, against prevalent PII-targeted attacks, demonstrating the superiority of our method compared with other existing defensive techniques. The results show that our PPA method completely eliminates the risk of phone number exposure by 100% and significantly reduces the risk of physical address exposure by 9.8% - 87.6%, all while maintaining comparable model utility performance.

摘要: 随着大型语言模型(LLM)的兴起，越来越多的研究已经认识到它们在恶意攻击下泄露个人身份信息(PII)的风险。虽然已经做出了努力来保护LLMS中的PII，但现有的方法难以平衡隐私保护和维护模型效用。受认知科学中关于健忘症的研究启发，我们提出了一种新的方法，即主动隐私健忘症(PPA)，在保护LLMS中的PII的同时保持其实用性。这种机制的工作原理是主动识别和忘记序列中与PII最密切相关的关键记忆，然后使用适当的替代记忆植入记忆以维持LLM的功能。我们在多个模型上进行评估，以保护常见的PII，如电话号码和物理地址，免受流行的PII目标攻击，展示了我们的方法与其他现有防御技术相比的优越性。结果表明，我们的PPA方法完全消除了100%的电话号码暴露风险，并显著降低了9.8%-87.6%的物理地址暴露风险，所有这些都保持了可比的模型实用性能。



## **17. The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence**

大型语言模型中拒绝的几何学：概念锥和表示独立性 cs.LG

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17420v1) [paper-pdf](http://arxiv.org/pdf/2502.17420v1)

**Authors**: Tom Wollschläger, Jannes Elstner, Simon Geisler, Vincent Cohen-Addad, Stephan Günnemann, Johannes Gasteiger

**Abstract**: The safety alignment of large language models (LLMs) can be circumvented through adversarially crafted inputs, yet the mechanisms by which these attacks bypass safety barriers remain poorly understood. Prior work suggests that a single refusal direction in the model's activation space determines whether an LLM refuses a request. In this study, we propose a novel gradient-based approach to representation engineering and use it to identify refusal directions. Contrary to prior work, we uncover multiple independent directions and even multi-dimensional concept cones that mediate refusal. Moreover, we show that orthogonality alone does not imply independence under intervention, motivating the notion of representational independence that accounts for both linear and non-linear effects. Using this framework, we identify mechanistically independent refusal directions. We show that refusal mechanisms in LLMs are governed by complex spatial structures and identify functionally independent directions, confirming that multiple distinct mechanisms drive refusal behavior. Our gradient-based approach uncovers these mechanisms and can further serve as a foundation for future work on understanding LLMs.

摘要: 大型语言模型(LLM)的安全一致性可以通过恶意创建的输入来规避，但这些攻击绕过安全屏障的机制仍然知之甚少。先前的工作表明，模型激活空间中的单个拒绝方向决定了LLM是否拒绝请求。在这项研究中，我们提出了一种新的基于梯度的表示工程方法，并用它来识别拒绝方向。与以前的工作相反，我们发现了多个独立的方向，甚至是调解拒绝的多维概念锥。此外，我们表明，正交性本身并不意味着干预下的独立性，这激发了既能解释线性效应又能解释非线性效应的表征独立性的概念。利用这个框架，我们确定了机械独立的拒绝方向。我们发现，LLMS中的拒绝机制受到复杂空间结构的支配，并识别出功能独立的方向，证实了多种不同的机制驱动着拒绝行为。我们的基于梯度的方法揭示了这些机制，并可以进一步作为理解LLMS的未来工作的基础。



## **18. Dataset Featurization: Uncovering Natural Language Features through Unsupervised Data Reconstruction**

数据集特征化：通过无监督数据重建揭示自然语言特征 cs.AI

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17541v1) [paper-pdf](http://arxiv.org/pdf/2502.17541v1)

**Authors**: Michal Bravansky, Vaclav Kubon, Suhas Hariharan, Robert Kirk

**Abstract**: Interpreting data is central to modern research. Large language models (LLMs) show promise in providing such natural language interpretations of data, yet simple feature extraction methods such as prompting often fail to produce accurate and versatile descriptions for diverse datasets and lack control over granularity and scale. To address these limitations, we propose a domain-agnostic method for dataset featurization that provides precise control over the number of features extracted while maintaining compact and descriptive representations comparable to human expert labeling. Our method optimizes the selection of informative binary features by evaluating the ability of an LLM to reconstruct the original data using those features. We demonstrate its effectiveness in dataset modeling tasks and through two case studies: (1) Constructing a feature representation of jailbreak tactics that compactly captures both the effectiveness and diversity of a larger set of human-crafted attacks; and (2) automating the discovery of features that align with human preferences, achieving accuracy and robustness comparable to expert-crafted features. Moreover, we show that the pipeline scales effectively, improving as additional features are sampled, making it suitable for large and diverse datasets.

摘要: 解释数据是现代研究的核心。大型语言模型(LLM)在提供数据的自然语言解释方面显示出了希望，然而简单的特征提取方法，如提示，往往无法为不同的数据集产生准确和通用的描述，并且缺乏对粒度和规模的控制。为了解决这些局限性，我们提出了一种领域无关的数据集特征化方法，该方法对提取的特征数量提供精确控制，同时保持紧凑和描述性的表示，可与人类专家标记相媲美。我们的方法通过评估LLM使用这些特征重建原始数据的能力来优化信息二进制特征的选择。我们在数据集建模任务中展示了它的有效性，并通过两个案例研究：(1)构建越狱策略的特征表示，紧凑地捕捉到更大的人为攻击集的有效性和多样性；以及(2)自动发现符合人类偏好的特征，获得与专家创建的特征相当的准确性和健壮性。此外，我们还表明，管道的规模有效，随着额外特征的采样而改进，使其适合于大型和多样化的数据集。



## **19. Emoti-Attack: Zero-Perturbation Adversarial Attacks on NLP Systems via Emoji Sequences**

伪攻击：通过伪君子序列对NLP系统进行零微扰对抗攻击 cs.AI

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17392v1) [paper-pdf](http://arxiv.org/pdf/2502.17392v1)

**Authors**: Yangshijie Zhang

**Abstract**: Deep neural networks (DNNs) have achieved remarkable success in the field of natural language processing (NLP), leading to widely recognized applications such as ChatGPT. However, the vulnerability of these models to adversarial attacks remains a significant concern. Unlike continuous domains like images, text exists in a discrete space, making even minor alterations at the sentence, word, or character level easily perceptible to humans. This inherent discreteness also complicates the use of conventional optimization techniques, as text is non-differentiable. Previous research on adversarial attacks in text has focused on character-level, word-level, sentence-level, and multi-level approaches, all of which suffer from inefficiency or perceptibility issues due to the need for multiple queries or significant semantic shifts.   In this work, we introduce a novel adversarial attack method, Emoji-Attack, which leverages the manipulation of emojis to create subtle, yet effective, perturbations. Unlike character- and word-level strategies, Emoji-Attack targets emojis as a distinct layer of attack, resulting in less noticeable changes with minimal disruption to the text. This approach has been largely unexplored in previous research, which typically focuses on emoji insertion as an extension of character-level attacks. Our experiments demonstrate that Emoji-Attack achieves strong attack performance on both large and small models, making it a promising technique for enhancing adversarial robustness in NLP systems.

摘要: 深度神经网络(DNN)在自然语言处理(NLP)领域取得了显著的成功，产生了广泛认可的应用，如ChatGPT。然而，这些模型对对抗性攻击的脆弱性仍然是一个重大关切。与图像等连续领域不同，文本存在于离散的空间中，即使是句子、单词或字符级别的微小更改也很容易被人类察觉。这种固有的离散性也使传统优化技术的使用变得复杂，因为文本是不可区分的。以往对文本中敌意攻击的研究主要集中在字符级、词级、句子级和多层方法上，所有这些方法都存在效率低下或可感知性问题，原因是需要进行多个查询或显著的语义转换。在这项工作中，我们介绍了一种新的对抗性攻击方法，Emoji-Attack，它利用表情符号的操纵来制造微妙但有效的扰动。与字符和单词级别的策略不同，Emoji攻击将表情符号作为不同的攻击层，导致不太明显的变化，对文本的破坏最小。这种方法在以前的研究中基本上没有被探索过，这些研究通常专注于将表情符号插入作为字符级攻击的扩展。我们的实验表明，Emoji-Attack在大小模型上都具有很强的攻击性能，是一种在NLP系统中增强对手健壮性的很有前途的技术。



## **20. REINFORCE Adversarial Attacks on Large Language Models: An Adaptive, Distributional, and Semantic Objective**

REINFORCE对大型语言模型的对抗攻击：自适应、分布和语义目标 cs.LG

30 pages, 6 figures, 15 tables

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17254v1) [paper-pdf](http://arxiv.org/pdf/2502.17254v1)

**Authors**: Simon Geisler, Tom Wollschläger, M. H. I. Abdalla, Vincent Cohen-Addad, Johannes Gasteiger, Stephan Günnemann

**Abstract**: To circumvent the alignment of large language models (LLMs), current optimization-based adversarial attacks usually craft adversarial prompts by maximizing the likelihood of a so-called affirmative response. An affirmative response is a manually designed start of a harmful answer to an inappropriate request. While it is often easy to craft prompts that yield a substantial likelihood for the affirmative response, the attacked model frequently does not complete the response in a harmful manner. Moreover, the affirmative objective is usually not adapted to model-specific preferences and essentially ignores the fact that LLMs output a distribution over responses. If low attack success under such an objective is taken as a measure of robustness, the true robustness might be grossly overestimated. To alleviate these flaws, we propose an adaptive and semantic optimization problem over the population of responses. We derive a generally applicable objective via the REINFORCE policy-gradient formalism and demonstrate its efficacy with the state-of-the-art jailbreak algorithms Greedy Coordinate Gradient (GCG) and Projected Gradient Descent (PGD). For example, our objective doubles the attack success rate (ASR) on Llama3 and increases the ASR from 2% to 50% with circuit breaker defense.

摘要: 为了绕过大型语言模型(LLM)的对齐，当前基于优化的对抗性攻击通常通过最大化所谓肯定响应的可能性来创建对抗性提示。肯定答复是手动设计的对不适当请求的有害答复的开始。虽然通常很容易制定提示，以产生肯定响应的很大可能性，但被攻击的模型通常不会以有害的方式完成响应。此外，肯定的目标通常不适应特定于模型的偏好，基本上忽略了LLMS输出的分布高于响应的事实。如果在这样的目标下将低攻击成功率作为稳健性的衡量标准，则可能严重高估了真正的稳健性。为了克服这些缺陷，我们提出了一种基于响应总体的自适应语义优化问题。我们通过强化策略梯度理论推导出一个普遍适用的目标，并用最先进的越狱算法贪婪坐标梯度(GCG)和投影梯度下降(PGD)来验证其有效性。例如，我们的目标是使Llama3上的攻击成功率(ASR)翻一番，并通过断路器防御将ASR从2%提高到50%。



## **21. Adversarial Training for Defense Against Label Poisoning Attacks**

防御标签中毒攻击的对抗训练 cs.LG

Accepted at the International Conference on Learning Representations  (ICLR 2025)

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17121v1) [paper-pdf](http://arxiv.org/pdf/2502.17121v1)

**Authors**: Melis Ilayda Bal, Volkan Cevher, Michael Muehlebach

**Abstract**: As machine learning models grow in complexity and increasingly rely on publicly sourced data, such as the human-annotated labels used in training large language models, they become more vulnerable to label poisoning attacks. These attacks, in which adversaries subtly alter the labels within a training dataset, can severely degrade model performance, posing significant risks in critical applications. In this paper, we propose FLORAL, a novel adversarial training defense strategy based on support vector machines (SVMs) to counter these threats. Utilizing a bilevel optimization framework, we cast the training process as a non-zero-sum Stackelberg game between an attacker, who strategically poisons critical training labels, and the model, which seeks to recover from such attacks. Our approach accommodates various model architectures and employs a projected gradient descent algorithm with kernel SVMs for adversarial training. We provide a theoretical analysis of our algorithm's convergence properties and empirically evaluate FLORAL's effectiveness across diverse classification tasks. Compared to robust baselines and foundation models such as RoBERTa, FLORAL consistently achieves higher robust accuracy under increasing attacker budgets. These results underscore the potential of FLORAL to enhance the resilience of machine learning models against label poisoning threats, thereby ensuring robust classification in adversarial settings.

摘要: 随着机器学习模型变得越来越复杂，并越来越依赖于公共来源的数据，例如用于训练大型语言模型的人类注释标签，它们变得更容易受到标签中毒攻击。在这些攻击中，攻击者巧妙地更改了训练数据集中的标签，可能会严重降低模型的性能，给关键应用程序带来重大风险。针对这些威胁，本文提出了一种新的基于支持向量机的对抗性训练防御策略FLOLAR。利用双层优化框架，我们将训练过程描述为攻击者和模型之间的非零和Stackelberg博弈，攻击者策略性地毒害关键的训练标签，而模型试图从此类攻击中恢复。我们的方法适应了不同的模型结构，并使用了一种带有核支持向量机的投影梯度下降算法进行对抗性训练。我们对算法的收敛特性进行了理论分析，并对FLORAL算法在不同分类任务上的有效性进行了实证评估。与罗伯塔等稳健的基线和基础模型相比，FLORAL在不断增加的攻击者预算下始终实现更高的稳健精度。这些结果强调了FLOLAR的潜力，以增强机器学习模型对标签中毒威胁的弹性，从而确保在对抗性环境中的稳健分类。



## **22. GuidedBench: Equipping Jailbreak Evaluation with Guidelines**

GuidedBench：为越狱评估配备指导方针 cs.CL

Homepage: https://sproutnan.github.io/AI-Safety_Benchmark/

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.16903v1) [paper-pdf](http://arxiv.org/pdf/2502.16903v1)

**Authors**: Ruixuan Huang, Xunguang Wang, Zongjie Li, Daoyuan Wu, Shuai Wang

**Abstract**: Jailbreaking methods for large language models (LLMs) have gained increasing attention for building safe and responsible AI systems. After analyzing 35 jailbreak methods across six categories, we find that existing benchmarks, relying on universal LLM-based or keyword-matching scores, lack case-specific criteria, leading to conflicting results. In this paper, we introduce a more robust evaluation framework for jailbreak methods, with a curated harmful question dataset, detailed case-by-case evaluation guidelines, and a scoring system equipped with these guidelines. Our experiments show that existing jailbreak methods exhibit better discrimination when evaluated using our benchmark. Some jailbreak methods that claim to achieve over 90% attack success rate (ASR) on other benchmarks only reach a maximum of 30.2% on our benchmark, providing a higher ceiling for more advanced jailbreak research; furthermore, using our scoring system reduces the variance of disagreements between different evaluator LLMs by up to 76.33%. This demonstrates its ability to provide more fair and stable evaluation.

摘要: 大型语言模型的越狱方法在构建安全可靠的人工智能系统方面得到了越来越多的关注。在分析了六个类别的35种越狱方法后，我们发现现有的基准测试依赖于通用的基于LLM或关键字匹配的分数，缺乏特定于案例的标准，导致结果相互冲突。在这篇文章中，我们介绍了一个更健壮的越狱方法评估框架，包括一个经过策划的有害问题数据集，详细的个案评估指南，以及一个配备了这些指南的评分系统。我们的实验表明，现有的越狱方法在使用我们的基准测试时表现出更好的识别率。一些声称在其他基准上达到90%以上攻击成功率(ASR)的越狱方法，在我们的基准上只达到了30.2%的最大值，为更高级的越狱研究提供了更高的上限；此外，使用我们的评分系统，不同评估者LM之间的分歧最高可减少76.33%。这表明它有能力提供更公平和稳定的评价。



## **23. Char-mander Use mBackdoor! A Study of Cross-lingual Backdoor Attacks in Multilingual LLMs**

Char-mander使用mBackdoor！多语言LLM中的跨语言后门攻击研究 cs.CL

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.16901v1) [paper-pdf](http://arxiv.org/pdf/2502.16901v1)

**Authors**: Himanshu Beniwal, Sailesh Panda, Mayank Singh

**Abstract**: We explore Cross-lingual Backdoor ATtacks (X-BAT) in multilingual Large Language Models (mLLMs), revealing how backdoors inserted in one language can automatically transfer to others through shared embedding spaces. Using toxicity classification as a case study, we demonstrate that attackers can compromise multilingual systems by poisoning data in a single language, with rare tokens serving as specific effective triggers. Our findings expose a critical vulnerability in the fundamental architecture that enables cross-lingual transfer in these models. Our code and data are publicly available at https://github.com/himanshubeniwal/X-BAT.

摘要: 我们探索多语言大型语言模型（mLLM）中的跨语言后门攻击（X-BAT），揭示了以一种语言插入的后门如何通过共享嵌入空间自动传输到其他语言。使用毒性分类作为案例研究，我们证明攻击者可以通过毒害单一语言的数据来危害多语言系统，其中稀有标记充当特定的有效触发器。我们的研究结果揭示了这些模型中实现跨语言迁移的基本架构中的一个关键漏洞。我们的代码和数据可在https://github.com/himanshubeniwal/X-BAT上公开获取。



## **24. PAPILLON: Efficient and Stealthy Fuzz Testing-Powered Jailbreaks for LLMs**

PAPILLON：针对LLM的高效、隐蔽的Fuzz测试动力越狱 cs.CR

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2409.14866v4) [paper-pdf](http://arxiv.org/pdf/2409.14866v4)

**Authors**: Xueluan Gong, Mingzhe Li, Yilin Zhang, Fengyuan Ran, Chen Chen, Yanjiao Chen, Qian Wang, Kwok-Yan Lam

**Abstract**: Large Language Models (LLMs) have excelled in various tasks but are still vulnerable to jailbreaking attacks, where attackers create jailbreak prompts to mislead the model to produce harmful or offensive content. Current jailbreak methods either rely heavily on manually crafted templates, which pose challenges in scalability and adaptability, or struggle to generate semantically coherent prompts, making them easy to detect. Additionally, most existing approaches involve lengthy prompts, leading to higher query costs.In this paper, to remedy these challenges, we introduce a novel jailbreaking attack framework called PAPILLON, which is an automated, black-box jailbreaking attack framework that adapts the black-box fuzz testing approach with a series of customized designs. Instead of relying on manually crafted templates,PAPILLON starts with an empty seed pool, removing the need to search for any related jailbreaking templates. We also develop three novel question-dependent mutation strategies using an LLM helper to generate prompts that maintain semantic coherence while significantly reducing their length. Additionally, we implement a two-level judge module to accurately detect genuine successful jailbreaks. We evaluated PAPILLON on 7 representative LLMs and compared it with 5 state-of-the-art jailbreaking attack strategies. For proprietary LLM APIs, such as GPT-3.5 turbo, GPT-4, and Gemini-Pro, PAPILLONs achieves attack success rates of over 90%, 80%, and 74%, respectively, exceeding existing baselines by more than 60\%. Additionally, PAPILLON can maintain high semantic coherence while significantly reducing the length of jailbreak prompts. When targeting GPT-4, PAPILLON can achieve over 78% attack success rate even with 100 tokens. Moreover, PAPILLON demonstrates transferability and is robust to state-of-the-art defenses.

摘要: 大型语言模型(LLM)在各种任务中表现出色，但仍然容易受到越狱攻击，在越狱攻击中，攻击者创建越狱提示来误导模型生成有害或攻击性内容。当前的越狱方法要么严重依赖于人工制作的模板，这对可伸缩性和适应性构成了挑战，要么难以生成语义连贯的提示，使它们很容易被检测到。为了解决这些问题，本文提出了一种新的越狱攻击框架Papillon，它是一个自动化的黑盒越狱攻击框架，通过一系列定制设计采用了黑盒模糊测试方法。与依赖手工制作的模板不同，Papillon从一个空的种子库开始，不需要搜索任何相关的越狱模板。我们还开发了三种新的问题相关突变策略，使用LLM助手来生成提示，这些提示在保持语义连贯的同时显著缩短了提示的长度。此外，我们实现了一个两级判断模块来准确地检测真正的成功越狱。我们在7个有代表性的LLM上对Papillon进行了评估，并将其与5种最先进的越狱攻击策略进行了比较。对于专有的LLMAPI，如GPT-3.5 Turbo、GPT-4和Gemini-Pro，Papillons的攻击成功率分别超过90%、80%和74%，比现有基线高出60%以上。此外，Papillon可以保持高度的语义连贯性，同时显著缩短越狱提示的长度。当针对GPT-4时，Papillon即使使用100个令牌也可以达到78%以上的攻击成功率。此外，乳突展示了可转移性，并对最先进的防御措施具有很强的抵抗力。



## **25. Dysca: A Dynamic and Scalable Benchmark for Evaluating Perception Ability of LVLMs**

Dysca：评估LVLM感知能力的动态和可扩展基准 cs.CV

Accepted by ICLR2025

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2406.18849v4) [paper-pdf](http://arxiv.org/pdf/2406.18849v4)

**Authors**: Jie Zhang, Zhongqi Wang, Mengqi Lei, Zheng Yuan, Bei Yan, Shiguang Shan, Xilin Chen

**Abstract**: Currently many benchmarks have been proposed to evaluate the perception ability of the Large Vision-Language Models (LVLMs). However, most benchmarks conduct questions by selecting images from existing datasets, resulting in the potential data leakage. Besides, these benchmarks merely focus on evaluating LVLMs on the realistic style images and clean scenarios, leaving the multi-stylized images and noisy scenarios unexplored. In response to these challenges, we propose a dynamic and scalable benchmark named Dysca for evaluating LVLMs by leveraging synthesis images. Specifically, we leverage Stable Diffusion and design a rule-based method to dynamically generate novel images, questions and the corresponding answers. We consider 51 kinds of image styles and evaluate the perception capability in 20 subtasks. Moreover, we conduct evaluations under 4 scenarios (i.e., Clean, Corruption, Print Attacking and Adversarial Attacking) and 3 question types (i.e., Multi-choices, True-or-false and Free-form). Thanks to the generative paradigm, Dysca serves as a scalable benchmark for easily adding new subtasks and scenarios. A total of 24 advanced open-source LVLMs and 2 close-source LVLMs are evaluated on Dysca, revealing the drawbacks of current LVLMs. The benchmark is released at https://github.com/Robin-WZQ/Dysca.

摘要: 目前，人们已经提出了许多基准来评估大型视觉语言模型的感知能力。然而，大多数基准测试通过从现有数据集中选择图像来进行问题，从而导致潜在的数据泄漏。此外，这些基准只关注真实感风格的图像和干净的场景来评估LVLMS，而对多风格化的图像和噪声场景没有进行探索。为了应对这些挑战，我们提出了一种动态的、可扩展的基准测试DYSCA，用于利用合成图像来评估LVLMS。具体地说，我们利用稳定扩散并设计了一种基于规则的方法来动态生成新的图像、问题和相应的答案。我们考虑了51种图像风格，并在20个子任务中评估了感知能力。此外，我们还在4个场景(即廉洁、腐败、打印攻击和对抗性攻击)和3个问题类型(即多项选择、对错和自由形式)下进行了评估。多亏了生成性范式，Dysca成为了一个可伸缩的基准，可以轻松添加新的子任务和场景。在Dysca上对24个先进的开源LVLMS和2个闭源LVLMS进行了评估，揭示了现有LVLMS的不足。该基准在https://github.com/Robin-WZQ/Dysca.上发布



## **26. Guardians of the Agentic System: Preventing Many Shots Jailbreak with Agentic System**

紧急系统的守护者：用紧急系统防止多次枪击越狱 cs.CR

18 pages, 7 figures

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16750v1) [paper-pdf](http://arxiv.org/pdf/2502.16750v1)

**Authors**: Saikat Barua, Mostafizur Rahman, Md Jafor Sadek, Rafiul Islam, Shehnaz Khaled, Ahmedul Kabir

**Abstract**: The autonomous AI agents using large language models can create undeniable values in all span of the society but they face security threats from adversaries that warrants immediate protective solutions because trust and safety issues arise. Considering the many-shot jailbreaking and deceptive alignment as some of the main advanced attacks, that cannot be mitigated by the static guardrails used during the supervised training, points out a crucial research priority for real world robustness. The combination of static guardrails in dynamic multi-agent system fails to defend against those attacks. We intend to enhance security for LLM-based agents through the development of new evaluation frameworks which identify and counter threats for safe operational deployment. Our work uses three examination methods to detect rogue agents through a Reverse Turing Test and analyze deceptive alignment through multi-agent simulations and develops an anti-jailbreaking system by testing it with GEMINI 1.5 pro and llama-3.3-70B, deepseek r1 models using tool-mediated adversarial scenarios. The detection capabilities are strong such as 94\% accuracy for GEMINI 1.5 pro yet the system suffers persistent vulnerabilities when under long attacks as prompt length increases attack success rates (ASR) and diversity metrics become ineffective in prediction while revealing multiple complex system faults. The findings demonstrate the necessity of adopting flexible security systems based on active monitoring that can be performed by the agents themselves together with adaptable interventions by system admin as the current models can create vulnerabilities that can lead to the unreliable and vulnerable system. So, in our work, we try to address such situations and propose a comprehensive framework to counteract the security issues.

摘要: 使用大型语言模型的自主人工智能代理可以在社会各个领域创造不可否认的价值，但他们面临来自对手的安全威胁，需要立即采取保护性解决方案，因为信任和安全问题会出现。考虑到多发越狱和欺骗性对准是一些主要的高级攻击，在监督训练期间使用的静态护栏无法减轻这些攻击，指出了现实世界健壮性的关键研究重点。动态多智能体系统中静态护栏的组合不能抵抗这些攻击。我们打算通过制定新的评估框架，确定和应对安全行动部署所面临的威胁，从而加强基于LLM的特工的安全。我们的工作使用了三种检测方法来通过反向图灵测试来检测流氓代理，并通过多代理模拟来分析欺骗性比对，并开发了一个反越狱系统，通过使用Gemini 1.5Pro和Llama-3.3-70B、使用工具中介的对抗场景来测试DeepSeek R1模型来开发反越狱系统。Gemini 1.5 PRO具有很强的检测能力，如94%的准确率，但在长时间攻击下，随着提示长度的增加，攻击成功率(ASR)增加，多样性度量在预测多个复杂系统故障时变得无效，系统存在持续漏洞。这些发现证明了采用基于主动监控的灵活安全系统的必要性，该系统可以由代理自己执行，并由系统管理员进行适应性干预，因为当前的模型可能会产生漏洞，从而导致系统不可靠和易受攻击。因此，在我们的工作中，我们试图解决这些情况，并提出一个全面的框架来对抗安全问题。



## **27. RapidPen: Fully Automated IP-to-Shell Penetration Testing with LLM-based Agents**

RapidPen：使用基于LLM的代理进行全自动IP到Shell渗透测试 cs.CR

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16730v1) [paper-pdf](http://arxiv.org/pdf/2502.16730v1)

**Authors**: Sho Nakatani

**Abstract**: We present RapidPen, a fully automated penetration testing (pentesting) framework that addresses   the challenge of achieving an initial foothold (IP-to-Shell) without human intervention. Unlike prior   approaches that focus primarily on post-exploitation or require a human-in-the-loop, RapidPen   leverages large language models (LLMs) to autonomously discover and exploit vulnerabilities, starting from   a single IP address. By integrating advanced ReAct-style task planning (Re) with retrieval-augmented   knowledge bases of successful exploits, along with a command-generation and direct execution feedback loop   (Act), RapidPen systematically scans services, identifies viable attack vectors, and executes targeted   exploits in a fully automated manner.   In our evaluation against a vulnerable target from the Hack The Box platform, RapidPen achieved shell   access within 200-400 seconds at a per-run cost of approximately \$0.3-\$0.6, demonstrating a   60\% success rate when reusing prior "success-case" data. These results underscore the potential   of truly autonomous pentesting for both security novices and seasoned professionals. Organizations   without dedicated security teams can leverage RapidPen to quickly identify critical vulnerabilities,   while expert pentesters can offload repetitive tasks and focus on complex challenges.   Ultimately, our work aims to make penetration testing more accessible and cost-efficient,   thereby enhancing the overall security posture of modern software ecosystems.

摘要: 我们提出了RapidPen，这是一个全自动渗透测试(PETTING)框架，可以解决在没有人工干预的情况下实现初始立足点(IP到外壳)的挑战。与以前的方法不同，RapidPen主要关注攻击后攻击或需要人在循环中，RapidPen利用大型语言模型(LLM)从单个IP地址开始自主发现和利用漏洞。通过将先进的反应式任务规划(Re)与成功利用漏洞的增强检索知识库以及命令生成和直接执行反馈循环(Act)相集成，RapidPen以完全自动化的方式系统地扫描服务、识别可行的攻击载体并执行目标攻击。在我们针对Hack the Box平台的易受攻击目标进行的评估中，RapidPen在200-400秒内实现了外壳访问，每次运行成本约为\$0.3-\$0.6，在重用先前的“成功案例”数据时显示了60%的成功率。这些结果强调了真正自主的安全新手和经验丰富的专业人士的潜力。没有专门安全团队的组织可以利用RapidPen快速识别关键漏洞，而专家专家可以卸载重复性任务，专注于复杂的挑战。最终，我们的工作旨在使渗透测试更容易获得和更具成本效益，从而增强现代软件生态系统的整体安全态势。



## **28. Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images**

通过参数学习对抗图像跟踪大型视觉语言模型的版权 cs.AI

Accepted to ICLR 2025

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16593v1) [paper-pdf](http://arxiv.org/pdf/2502.16593v1)

**Authors**: Yubo Wang, Jianting Tang, Chaohu Liu, Linli Xu

**Abstract**: Large vision-language models (LVLMs) have demonstrated remarkable image understanding and dialogue capabilities, allowing them to handle a variety of visual question answering tasks. However, their widespread availability raises concerns about unauthorized usage and copyright infringement, where users or individuals can develop their own LVLMs by fine-tuning published models. In this paper, we propose a novel method called Parameter Learning Attack (PLA) for tracking the copyright of LVLMs without modifying the original model. Specifically, we construct adversarial images through targeted attacks against the original model, enabling it to generate specific outputs. To ensure these attacks remain effective on potential fine-tuned models to trigger copyright tracking, we allow the original model to learn the trigger images by updating parameters in the opposite direction during the adversarial attack process. Notably, the proposed method can be applied after the release of the original model, thus not affecting the model's performance and behavior. To simulate real-world applications, we fine-tune the original model using various strategies across diverse datasets, creating a range of models for copyright verification. Extensive experiments demonstrate that our method can more effectively identify the original copyright of fine-tuned models compared to baseline methods. Therefore, this work provides a powerful tool for tracking copyrights and detecting unlicensed usage of LVLMs.

摘要: 大型视觉语言模型(LVLM)已经显示出非凡的图像理解和对话能力，使它们能够处理各种视觉问题回答任务。然而，它们的广泛使用引发了人们对未经授权使用和侵犯版权的担忧，在这种情况下，用户或个人可以通过微调已发布的模型来开发自己的LVLM。在本文中，我们提出了一种称为参数学习攻击的新方法，该方法在不修改原始模型的情况下跟踪LVLMS的版权。具体地说，我们通过对原始模型进行有针对性的攻击来构建对抗性图像，使其能够生成特定的输出。为了确保这些攻击在触发版权跟踪的潜在微调模型上保持有效，我们允许原始模型在对抗性攻击过程中通过反向更新参数来学习触发图像。值得注意的是，所提出的方法可以在原始模型发布之后应用，因此不会影响模型的性能和行为。为了模拟真实世界的应用程序，我们使用不同的策略对原始模型进行微调，创建了一系列用于版权验证的模型。大量实验表明，与基线方法相比，该方法能更有效地识别微调模型的原始版权。因此，这项工作为跟踪版权和检测未经许可的LVLM使用提供了一个强大的工具。



## **29. Can Indirect Prompt Injection Attacks Be Detected and Removed?**

可以检测并删除间接提示注入攻击吗？ cs.CR

17 pages, 6 figures

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16580v1) [paper-pdf](http://arxiv.org/pdf/2502.16580v1)

**Authors**: Yulin Chen, Haoran Li, Yuan Sui, Yufei He, Yue Liu, Yangqiu Song, Bryan Hooi

**Abstract**: Prompt injection attacks manipulate large language models (LLMs) by misleading them to deviate from the original input instructions and execute maliciously injected instructions, because of their instruction-following capabilities and inability to distinguish between the original input instructions and maliciously injected instructions. To defend against such attacks, recent studies have developed various detection mechanisms. While significant efforts have focused on detecting direct prompt injection attacks, where injected instructions are directly from the attacker who is also the user, limited attention has been given to indirect prompt injection attacks, where injected instructions are indirectly from external tools, such as a search engine. Moreover, current works mainly investigate injection detection methods and pay less attention to the post-processing method that aims to mitigate the injection after detection. In this paper, we investigate the feasibility of detecting and removing indirect prompt injection attacks, and we construct a benchmark dataset for evaluation. For detection, we assess the performance of existing LLMs and open-source detection models, and we further train detection models using our crafted training datasets. For removal, we evaluate two intuitive methods: (1) the segmentation removal method, which segments the injected document and removes parts containing injected instructions, and (2) the extraction removal method, which trains an extraction model to identify and remove injected instructions.

摘要: 提示注入攻击通过误导大型语言模型(LLM)偏离原始输入指令并执行恶意注入指令来操纵它们，因为它们具有指令跟随能力，并且无法区分原始输入指令和恶意注入指令。为了防御这种攻击，最近的研究开发了各种检测机制。虽然大量的工作集中在检测直接提示注入攻击，其中注入的指令直接来自同时也是用户的攻击者，但对间接提示注入攻击的关注有限，其中注入的指令间接来自外部工具，例如搜索引擎。此外，目前的研究主要集中在注入检测方法上，而对检测后注入的后处理方法的研究较少。在本文中，我们研究了检测和删除间接提示注入攻击的可行性，并构建了一个基准数据集进行评估。对于检测，我们评估了现有的LLMS和开源检测模型的性能，并使用我们精心设计的训练数据集进一步训练检测模型。对于移除，我们评估了两种直观的方法：(1)分割移除方法，它分割注入的文档并移除包含注入指令的部分；(2)提取移除方法，它训练提取模型来识别和移除注入的指令。



## **30. SafeRAG: Benchmarking Security in Retrieval-Augmented Generation of Large Language Model**

SafeRAG：大型语言模型检索增强生成中的安全性基准测试 cs.CR

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2501.18636v2) [paper-pdf](http://arxiv.org/pdf/2501.18636v2)

**Authors**: Xun Liang, Simin Niu, Zhiyu Li, Sensen Zhang, Hanyu Wang, Feiyu Xiong, Jason Zhaoxin Fan, Bo Tang, Shichao Song, Mengwei Wang, Jiawei Yang

**Abstract**: The indexing-retrieval-generation paradigm of retrieval-augmented generation (RAG) has been highly successful in solving knowledge-intensive tasks by integrating external knowledge into large language models (LLMs). However, the incorporation of external and unverified knowledge increases the vulnerability of LLMs because attackers can perform attack tasks by manipulating knowledge. In this paper, we introduce a benchmark named SafeRAG designed to evaluate the RAG security. First, we classify attack tasks into silver noise, inter-context conflict, soft ad, and white Denial-of-Service. Next, we construct RAG security evaluation dataset (i.e., SafeRAG dataset) primarily manually for each task. We then utilize the SafeRAG dataset to simulate various attack scenarios that RAG may encounter. Experiments conducted on 14 representative RAG components demonstrate that RAG exhibits significant vulnerability to all attack tasks and even the most apparent attack task can easily bypass existing retrievers, filters, or advanced LLMs, resulting in the degradation of RAG service quality. Code is available at: https://github.com/IAAR-Shanghai/SafeRAG.

摘要: 检索-扩充生成(RAG)的索引-检索-生成范式通过将外部知识集成到大型语言模型(LLM)中，在解决知识密集型任务方面取得了巨大的成功。然而，外部和未经验证的知识的结合增加了LLMS的脆弱性，因为攻击者可以通过操纵知识来执行攻击任务。在本文中，我们介绍了一个名为SafeRAG的基准测试程序，用于评估RAG的安全性。首先，我们将攻击任务分为银色噪声、上下文间冲突、软广告和白色拒绝服务。接下来，我们主要为每个任务手动构建RAG安全评估数据集(即SafeRAG数据集)。然后，我们利用SafeRAG数据集来模拟RAG可能遇到的各种攻击场景。在14个具有代表性的RAG组件上进行的实验表明，RAG对所有攻击任务都表现出很大的脆弱性，即使是最明显的攻击任务也可以很容易地绕过现有的检索器、过滤器或高级LLM，导致RAG服务质量下降。代码可从以下网址获得：https://github.com/IAAR-Shanghai/SafeRAG.



## **31. On Calibration of LLM-based Guard Models for Reliable Content Moderation**

基于LLM的保护模型的校准以实现可靠的内容审核 cs.CR

Accepted to ICLR 2025

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2410.10414v2) [paper-pdf](http://arxiv.org/pdf/2410.10414v2)

**Authors**: Hongfu Liu, Hengguan Huang, Xiangming Gu, Hao Wang, Ye Wang

**Abstract**: Large language models (LLMs) pose significant risks due to the potential for generating harmful content or users attempting to evade guardrails. Existing studies have developed LLM-based guard models designed to moderate the input and output of threat LLMs, ensuring adherence to safety policies by blocking content that violates these protocols upon deployment. However, limited attention has been given to the reliability and calibration of such guard models. In this work, we empirically conduct comprehensive investigations of confidence calibration for 9 existing LLM-based guard models on 12 benchmarks in both user input and model output classification. Our findings reveal that current LLM-based guard models tend to 1) produce overconfident predictions, 2) exhibit significant miscalibration when subjected to jailbreak attacks, and 3) demonstrate limited robustness to the outputs generated by different types of response models. Additionally, we assess the effectiveness of post-hoc calibration methods to mitigate miscalibration. We demonstrate the efficacy of temperature scaling and, for the first time, highlight the benefits of contextual calibration for confidence calibration of guard models, particularly in the absence of validation sets. Our analysis and experiments underscore the limitations of current LLM-based guard models and provide valuable insights for the future development of well-calibrated guard models toward more reliable content moderation. We also advocate for incorporating reliability evaluation of confidence calibration when releasing future LLM-based guard models.

摘要: 由于可能会生成有害内容或用户试图避开护栏，大型语言模型(LLM)会带来重大风险。现有研究开发了基于LLM的防护模型，旨在控制威胁LLM的输入和输出，通过在部署时阻止违反这些协议的内容来确保遵守安全策略。然而，对这种防护模型的可靠性和校准的关注有限。在这项工作中，我们在用户输入和模型输出分类的12个基准上，对现有的9个基于LLM的警戒模型进行了全面的置信度校准研究。我们的发现表明，当前基于LLM的警卫模型倾向于1)产生过度自信的预测，2)在受到越狱攻击时表现出严重的错误校准，3)对不同类型的反应模型产生的输出表现出有限的稳健性。此外，我们评估后校准方法的有效性，以减少错误校准。我们展示了温度缩放的有效性，并首次强调了上下文校准对于警卫模型的置信度校准的好处，特别是在缺乏验证集的情况下。我们的分析和实验强调了当前基于LLM的防护模型的局限性，并为未来发展经过良好校准的防护模型以实现更可靠的内容审核提供了有价值的见解。我们还主张在发布未来基于LLM的警戒模型时纳入置信度校准的可靠性评估。



## **32. Intrinsic Model Weaknesses: How Priming Attacks Unveil Vulnerabilities in Large Language Models**

模型固有弱点：启动攻击如何揭示大型语言模型中的漏洞 cs.CL

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16491v1) [paper-pdf](http://arxiv.org/pdf/2502.16491v1)

**Authors**: Yuyi Huang, Runzhe Zhan, Derek F. Wong, Lidia S. Chao, Ailin Tao

**Abstract**: Large language models (LLMs) have significantly influenced various industries but suffer from a critical flaw, the potential sensitivity of generating harmful content, which poses severe societal risks. We developed and tested novel attack strategies on popular LLMs to expose their vulnerabilities in generating inappropriate content. These strategies, inspired by psychological phenomena such as the "Priming Effect", "Safe Attention Shift", and "Cognitive Dissonance", effectively attack the models' guarding mechanisms. Our experiments achieved an attack success rate (ASR) of 100% on various open-source models, including Meta's Llama-3.2, Google's Gemma-2, Mistral's Mistral-NeMo, Falcon's Falcon-mamba, Apple's DCLM, Microsoft's Phi3, and Qwen's Qwen2.5, among others. Similarly, for closed-source models such as OpenAI's GPT-4o, Google's Gemini-1.5, and Claude-3.5, we observed an ASR of at least 95% on the AdvBench dataset, which represents the current state-of-the-art. This study underscores the urgent need to reassess the use of generative models in critical applications to mitigate potential adverse societal impacts.

摘要: 大型语言模型(LLM)对各个行业产生了重大影响，但存在一个严重缺陷，即产生有害内容的潜在敏感性，这会带来严重的社会风险。我们在流行的LLM上开发并测试了新的攻击策略，以暴露它们在生成不适当内容时的漏洞。这些策略受到“启动效应”、“安全注意转移”和“认知失调”等心理现象的启发，有效地攻击了模型的保护机制。我们的实验在各种开源模型上取得了100%的攻击成功率(ASR)，包括Meta的Llama-3.2、Google的Gema-2、Mistral的Mistral-Nemo、Falcon的Falcon-manba、苹果的DCLM、微软的Phi3和Qwen的Qwen2.5等。同样，对于封闭源代码的模型，如OpenAI的GPT-40、Google的Gemini-1.5和Claude-3.5，我们在AdvBtch数据集上观察到至少95%的ASR，这代表了当前的最先进水平。这项研究强调迫切需要重新评估生成性模型在关键应用中的使用，以减轻潜在的不利社会影响。



## **33. Swallowing the Poison Pills: Insights from Vulnerability Disparity Among LLMs**

吞下毒丸：从LLM之间脆弱性差异的洞察 cs.CR

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.18518v1) [paper-pdf](http://arxiv.org/pdf/2502.18518v1)

**Authors**: Peng Yifeng, Wu Zhizheng, Chen Chen

**Abstract**: Modern large language models (LLMs) exhibit critical vulnerabilities to poison pill attacks: localized data poisoning that alters specific factual knowledge while preserving overall model utility. We systematically demonstrate these attacks exploit inherent architectural properties of LLMs, achieving 54.6% increased retrieval inaccuracy on long-tail knowledge versus dominant topics and up to 25.5% increase retrieval inaccuracy on compressed models versus original architectures. Through controlled mutations (e.g., temporal/spatial/entity alterations) and, our method induces localized memorization deterioration with negligible impact on models' performance on regular standard benchmarks (e.g., <2% performance drop on MMLU/GPQA), leading to potential detection evasion. Our findings suggest: (1) Disproportionate vulnerability in long-tail knowledge may result from reduced parameter redundancy; (2) Model compression may increase attack surfaces, with pruned/distilled models requiring 30% fewer poison samples for equivalent damage; (3) Associative memory enables both spread of collateral damage to related concepts and amplification of damage from simultaneous attack, particularly for dominant topics. These findings raise concerns over current scaling paradigms since attack costs are lowering while defense complexity is rising. Our work establishes poison pills as both a security threat and diagnostic tool, revealing critical security-efficiency trade-offs in language model compression that challenges prevailing safety assumptions.

摘要: 现代大型语言模型(LLM)显示出对毒丸攻击的严重漏洞：本地化数据中毒，在保留整体模型效用的同时更改特定的事实知识。我们系统地证明了这些攻击利用了LLMS固有的体系结构特性，与主要主题相比，在长尾知识上的检索不准确率提高了54.6%，在压缩模型上的检索不准确率比原始体系结构提高了25.5%。通过受控突变(例如，时间/空间/实体改变)，我们的方法导致局部记忆恶化，而对常规标准基准测试上的模型性能的影响可以忽略不计(例如，在MMLU/GPQA上的性能下降<2%)，从而导致潜在的检测逃避。我们的发现表明：(1)参数冗余度的降低可能会导致长尾知识中不成比例的脆弱性；(2)模型压缩可能会增加攻击面，修剪/提取的模型需要30%的毒物样本才能获得同等的损害；(3)联想记忆既可以将附带损害扩散到相关概念，也可以放大同时攻击造成的损害，特别是对主导主题。这些发现引发了人们对当前扩展模式的担忧，因为攻击成本正在下降，而防御复杂性正在上升。我们的工作将毒丸作为一种安全威胁和诊断工具，揭示了语言模型压缩中关键的安全效率权衡，挑战了普遍的安全假设。



## **34. A generative approach to LLM harmfulness detection with special red flag tokens**

使用特殊危险信号令牌进行LLM危害检测的生成式方法 cs.CL

13 pages, 6 figures

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2502.16366v1) [paper-pdf](http://arxiv.org/pdf/2502.16366v1)

**Authors**: Sophie Xhonneux, David Dobre, Mehrnaz Mohfakhami, Leo Schwinn, Gauthier Gidel

**Abstract**: Most safety training methods for large language models (LLMs) based on fine-tuning rely on dramatically changing the output distribution of the model when faced with a harmful request, shifting it from an unsafe answer to a refusal to respond. These methods inherently compromise model capabilities and might make auto-regressive models vulnerable to attacks that make likely an initial token of affirmative response. To avoid that, we propose to expand the model's vocabulary with a special token we call red flag token (<rf>) and propose to fine-tune the model to generate this token at any time harmful content is generated or about to be generated. This novel safety training method effectively augments LLMs into generative classifiers of harmfulness at all times during the conversation. This method offers several advantages: it enables the model to explicitly learn the concept of harmfulness while marginally affecting the generated distribution, thus maintaining the model's utility. It also evaluates each generated answer rather than just the input prompt and provides a stronger defence against sampling-based attacks. In addition, it simplifies the evaluation of the model's robustness and reduces correlated failures when combined with a classifier. We further show an increased robustness to long contexts, and supervised fine-tuning attacks.

摘要: 大多数基于微调的大型语言模型(LLM)安全培训方法依赖于在面临有害请求时显著改变模型的输出分布，将其从不安全的答案转变为拒绝响应。这些方法本质上会损害模型的能力，并可能使自回归模型容易受到攻击，从而可能成为肯定响应的初始标志。为了避免这种情况，我们建议使用一种称为红旗令牌(<rf>)的特殊令牌来扩展模型的词汇量，并建议微调模型以在生成或即将生成有害内容时生成该令牌。这种新颖的安全训练方法有效地将LLMS添加到对话过程中的任何时刻的危害生成性分类器中。这种方法有几个优点：它使模型能够明确地学习危害性的概念，而对生成的分布略有影响，从而保持了模型的实用性。它还评估每个生成的答案，而不仅仅是输入提示，并提供针对基于采样的攻击的更强大的防御。此外，它简化了模型稳健性的评估，并减少了与分类器结合时的相关故障。我们进一步显示了对长上下文的增强的健壮性，并监督了微调攻击。



## **35. ELBA-Bench: An Efficient Learning Backdoor Attacks Benchmark for Large Language Models**

ELBA-Bench：大型语言模型的高效学习后门攻击基准 cs.CR

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2502.18511v1) [paper-pdf](http://arxiv.org/pdf/2502.18511v1)

**Authors**: Xuxu Liu, Siyuan Liang, Mengya Han, Yong Luo, Aishan Liu, Xiantao Cai, Zheng He, Dacheng Tao

**Abstract**: Generative large language models are crucial in natural language processing, but they are vulnerable to backdoor attacks, where subtle triggers compromise their behavior. Although backdoor attacks against LLMs are constantly emerging, existing benchmarks remain limited in terms of sufficient coverage of attack, metric system integrity, backdoor attack alignment. And existing pre-trained backdoor attacks are idealized in practice due to resource access constraints. Therefore we establish $\textit{ELBA-Bench}$, a comprehensive and unified framework that allows attackers to inject backdoor through parameter efficient fine-tuning ($\textit{e.g.,}$ LoRA) or without fine-tuning techniques ($\textit{e.g.,}$ In-context-learning). $\textit{ELBA-Bench}$ provides over 1300 experiments encompassing the implementations of 12 attack methods, 18 datasets, and 12 LLMs. Extensive experiments provide new invaluable findings into the strengths and limitations of various attack strategies. For instance, PEFT attack consistently outperform without fine-tuning approaches in classification tasks while showing strong cross-dataset generalization with optimized triggers boosting robustness; Task-relevant backdoor optimization techniques or attack prompts along with clean and adversarial demonstrations can enhance backdoor attack success while preserving model performance on clean samples. Additionally, we introduce a universal toolbox designed for standardized backdoor attack research, with the goal of propelling further progress in this vital area.

摘要: 生成性大型语言模型在自然语言处理中至关重要，但它们很容易受到后门攻击，在后门攻击中，微妙的触发器会危及它们的行为。尽管针对LLM的后门攻击不断涌现，但现有基准在攻击的足够覆盖率、度量系统完整性、后门攻击对齐方面仍然有限。由于资源访问的限制，现有的预先训练的后门攻击在实践中被理想化。因此，我们建立了$\textit{elba-BENCH}$，这是一个全面而统一的框架，允许攻击者通过参数高效微调($\textit{例如，$loa)或不使用微调技术($\textit{例如，$上下文学习)来注入后门。$\textit{ELBA-BENCH}$提供了1300多个实验，包括12种攻击方法、18个数据集和12个LLM的实现。广泛的实验为各种攻击策略的优势和局限性提供了新的宝贵发现。例如，PEFT攻击在分类任务中始终在没有微调方法的情况下表现得更好，同时显示出强大的跨数据集泛化能力，优化的触发器提高了健壮性；与任务相关的后门优化技术或攻击提示以及干净和对抗性的演示可以提高后门攻击的成功率，同时保持干净样本上的模型性能。此外，我们还介绍了一个为标准化后门攻击研究设计的通用工具箱，目的是推动这一重要领域的进一步发展。



## **36. Na'vi or Knave: Jailbreaking Language Models via Metaphorical Avatars**

纳威人或恶棍：通过隐喻化身越狱语言模型 cs.CL

Our study requires further in-depth research to ensure the  comprehensiveness and adequacy of the methodology

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2412.12145v4) [paper-pdf](http://arxiv.org/pdf/2412.12145v4)

**Authors**: Yu Yan, Sheng Sun, Junqi Tong, Min Liu, Qi Li

**Abstract**: Metaphor serves as an implicit approach to convey information, while enabling the generalized comprehension of complex subjects. However, metaphor can potentially be exploited to bypass the safety alignment mechanisms of Large Language Models (LLMs), leading to the theft of harmful knowledge. In our study, we introduce a novel attack framework that exploits the imaginative capacity of LLMs to achieve jailbreaking, the J\underline{\textbf{A}}ilbreak \underline{\textbf{V}}ia \underline{\textbf{A}}dversarial Me\underline{\textbf{TA}} -pho\underline{\textbf{R}} (\textit{AVATAR}). Specifically, to elicit the harmful response, AVATAR extracts harmful entities from a given harmful target and maps them to innocuous adversarial entities based on LLM's imagination. Then, according to these metaphors, the harmful target is nested within human-like interaction for jailbreaking adaptively. Experimental results demonstrate that AVATAR can effectively and transferablly jailbreak LLMs and achieve a state-of-the-art attack success rate across multiple advanced LLMs. Our study exposes a security risk in LLMs from their endogenous imaginative capabilities. Furthermore, the analytical study reveals the vulnerability of LLM to adversarial metaphors and the necessity of developing defense methods against jailbreaking caused by the adversarial metaphor. \textcolor{orange}{ \textbf{Warning: This paper contains potentially harmful content from LLMs.}}

摘要: 隐喻是一种隐含的传达信息的方法，同时也使人们能够对复杂的主题进行广义的理解。然而，隐喻可能被利用来绕过大型语言模型(LLM)的安全对齐机制，从而导致有害知识的窃取。在我们的研究中，我们提出了一种新的攻击框架，该框架利用LLMS的想象能力来实现越狱，即J下划线{\extbf{A}}ilBreak\下划线{\extbf{A}}ia\下划线{\extbf{A}}dversarial Me\下划线{\extbf{TA}}-pho\下划线{\extbf{R}}(\textit{阿凡达})。具体地说，为了引发有害反应，阿凡达从给定的有害目标中提取有害实体，并基于LLM的想象力将它们映射到无害的对抗性实体。然后，根据这些隐喻，有害的目标嵌套在类似人类的互动中，以适应越狱。实验结果表明，阿凡达可以有效地、可转移地越狱LLM，并在多个高级LLM上获得最先进的攻击成功率。我们的研究揭示了LLMS的安全风险来自其内生的想象力。此外，分析研究揭示了LLM对对抗性隐喻的脆弱性，以及开发针对对抗性隐喻导致的越狱的防御方法的必要性。\extCOLOR{橙色}{\extbf{警告：此纸张包含来自LLMS的潜在有害内容。}}



## **37. Humanizing the Machine: Proxy Attacks to Mislead LLM Detectors**

人性化机器：代理攻击误导LLM检测器 cs.LG

29 pages

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2410.19230v2) [paper-pdf](http://arxiv.org/pdf/2410.19230v2)

**Authors**: Tianchun Wang, Yuanzhou Chen, Zichuan Liu, Zhanwen Chen, Haifeng Chen, Xiang Zhang, Wei Cheng

**Abstract**: The advent of large language models (LLMs) has revolutionized the field of text generation, producing outputs that closely mimic human-like writing. Although academic and industrial institutions have developed detectors to prevent the malicious usage of LLM-generated texts, other research has doubt about the robustness of these systems. To stress test these detectors, we introduce a proxy-attack strategy that effortlessly compromises LLMs, causing them to produce outputs that align with human-written text and mislead detection systems. Our method attacks the source model by leveraging a reinforcement learning (RL) fine-tuned humanized small language model (SLM) in the decoding phase. Through an in-depth analysis, we demonstrate that our attack strategy is capable of generating responses that are indistinguishable to detectors, preventing them from differentiating between machine-generated and human-written text. We conduct systematic evaluations on extensive datasets using proxy-attacked open-source models, including Llama2-13B, Llama3-70B, and Mixtral-8*7B in both white- and black-box settings. Our findings show that the proxy-attack strategy effectively deceives the leading detectors, resulting in an average AUROC drop of 70.4% across multiple datasets, with a maximum drop of 90.3% on a single dataset. Furthermore, in cross-discipline scenarios, our strategy also bypasses these detectors, leading to a significant relative decrease of up to 90.9%, while in cross-language scenario, the drop reaches 91.3%. Despite our proxy-attack strategy successfully bypassing the detectors with such significant relative drops, we find that the generation quality of the attacked models remains preserved, even within a modest utility budget, when compared to the text produced by the original, unattacked source model.

摘要: 大型语言模型(LLM)的出现使文本生成领域发生了革命性的变化，产生了非常接近人类书写的输出。尽管学术和工业机构已经开发了检测器来防止恶意使用LLM生成的文本，但其他研究对这些系统的健壮性表示怀疑。为了对这些检测器进行压力测试，我们引入了一种代理攻击策略，该策略可以毫不费力地攻击LLM，使它们产生与人类书写的文本和误导检测系统一致的输出。我们的方法在解码阶段利用强化学习(RL)微调的人性化小语言模型(SLM)来攻击源模型。通过深入的分析，我们证明了我们的攻击策略能够产生检测器无法区分的响应，防止他们区分机器生成的文本和人类书写的文本。我们使用代理攻击的开源模型对大量的数据集进行了系统的评估，包括白盒和黑盒环境下的Llama2-13B、Llama3-70B和Mixtral-8*7B。结果表明，代理攻击策略有效地欺骗了领先的检测器，导致多个数据集的AUROC平均下降了70.4%，其中单个数据集的AUROC平均下降了90.3%。此外，在跨学科场景中，我们的策略也绕过了这些检测器，导致了高达90.9%的显著相对降幅，而在跨语言场景中，降幅达到91.3%。尽管我们的代理攻击策略成功地绕过了检测器，具有如此显著的相对下降，但我们发现，与原始的未受攻击源模型生成的文本相比，即使在适度的实用预算内，受攻击模型的生成质量仍然保持不变。



## **38. Be a Multitude to Itself: A Prompt Evolution Framework for Red Teaming**

成为自己的多元化：红色团队的快速进化框架 cs.CL

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2502.16109v1) [paper-pdf](http://arxiv.org/pdf/2502.16109v1)

**Authors**: Rui Li, Peiyi Wang, Jingyuan Ma, Di Zhang, Lei Sha, Zhifang Sui

**Abstract**: Large Language Models (LLMs) have gained increasing attention for their remarkable capacity, alongside concerns about safety arising from their potential to produce harmful content. Red teaming aims to find prompts that could elicit harmful responses from LLMs, and is essential to discover and mitigate safety risks before real-world deployment. However, manual red teaming is both time-consuming and expensive, rendering it unscalable. In this paper, we propose RTPE, a scalable evolution framework to evolve red teaming prompts across both breadth and depth dimensions, facilitating the automatic generation of numerous high-quality and diverse red teaming prompts. Specifically, in-breadth evolving employs a novel enhanced in-context learning method to create a multitude of quality prompts, whereas in-depth evolving applies customized transformation operations to enhance both content and form of prompts, thereby increasing diversity. Extensive experiments demonstrate that RTPE surpasses existing representative automatic red teaming methods on both attack success rate and diversity. In addition, based on 4,800 red teaming prompts created by RTPE, we further provide a systematic analysis of 8 representative LLMs across 8 sensitive topics.

摘要: 大型语言模型(LLM)因其非凡的能力而受到越来越多的关注，同时也引起了人们对它们可能产生有害内容的安全性的担忧。红色团队的目标是找到可能引发低密度脂蛋白有害响应的提示，这对于在现实世界部署之前发现和缓解安全风险至关重要。然而，手动红色团队既耗时又昂贵，使其无法扩展。在本文中，我们提出了RTPE，一个可扩展的演化框架，跨广度和深度维度进化红色团队提示，促进了大量高质量和多样化的红色团队提示的自动生成。具体地说，广度进化采用一种新颖的增强的情景学习方法来创建大量高质量的提示，而深度进化应用定制的转换操作来增强提示的内容和形式，从而增加多样性。大量的实验表明，RTPE在攻击成功率和多样性方面都超过了现有的有代表性的自动红色分组方法。此外，基于RTPE创建的4800个红色团队提示，我们进一步对8个敏感话题中的8个具有代表性的LLM进行了系统分析。



## **39. Merger-as-a-Stealer: Stealing Targeted PII from Aligned LLMs with Model Merging**

合并即窃取者：通过模型合并从一致的LLM窃取目标PRI cs.CR

17 pages, 3 figures

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2502.16094v1) [paper-pdf](http://arxiv.org/pdf/2502.16094v1)

**Authors**: Lin Lu, Zhigang Zuo, Ziji Sheng, Pan Zhou

**Abstract**: Model merging has emerged as a promising approach for updating large language models (LLMs) by integrating multiple domain-specific models into a cross-domain merged model. Despite its utility and plug-and-play nature, unmonitored mergers can introduce significant security vulnerabilities, such as backdoor attacks and model merging abuse. In this paper, we identify a novel and more realistic attack surface where a malicious merger can extract targeted personally identifiable information (PII) from an aligned model with model merging. Specifically, we propose \texttt{Merger-as-a-Stealer}, a two-stage framework to achieve this attack: First, the attacker fine-tunes a malicious model to force it to respond to any PII-related queries. The attacker then uploads this malicious model to the model merging conductor and obtains the merged model. Second, the attacker inputs direct PII-related queries to the merged model to extract targeted PII. Extensive experiments demonstrate that \texttt{Merger-as-a-Stealer} successfully executes attacks against various LLMs and model merging methods across diverse settings, highlighting the effectiveness of the proposed framework. Given that this attack enables character-level extraction for targeted PII without requiring any additional knowledge from the attacker, we stress the necessity for improved model alignment and more robust defense mechanisms to mitigate such threats.

摘要: 通过将多个领域特定的模型集成到一个跨域的合并模型中，模型合并已经成为更新大型语言模型(LLM)的一种有前途的方法。尽管具有实用性和即插即用性质，但不受监控的合并可能会带来严重的安全漏洞，例如后门攻击和模型合并滥用。在本文中，我们识别了一种新颖且更现实的攻击面，其中恶意合并可以通过模型合并从对齐的模型中提取目标个人身份信息(PII)。具体地说，我们提出了一个两阶段框架来实现此攻击：首先，攻击者微调恶意模型以强制其响应任何与PII相关的查询。然后，攻击者将该恶意模型上传到模型合并导线，并获得合并后的模型。其次，攻击者向合并模型输入与PII相关的直接查询，以提取目标PII。大量实验表明，该框架能够成功地对不同的LLM和不同环境下的模型合并方法进行攻击，突出了该框架的有效性。鉴于这种攻击能够在不需要攻击者任何额外知识的情况下对目标PII进行字符级提取，我们强调改进模型对齐和更强大的防御机制以缓解此类威胁的必要性。



## **40. Stealing Training Data from Large Language Models in Decentralized Training through Activation Inversion Attack**

通过激活倒置攻击从分散训练中的大型语言模型中窃取训练数据 cs.CR

12 pages, 5 figures

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2502.16086v1) [paper-pdf](http://arxiv.org/pdf/2502.16086v1)

**Authors**: Chenxi Dai, Lin Lu, Pan Zhou

**Abstract**: Decentralized training has become a resource-efficient framework to democratize the training of large language models (LLMs). However, the privacy risks associated with this framework, particularly due to the potential inclusion of sensitive data in training datasets, remain unexplored. This paper identifies a novel and realistic attack surface: the privacy leakage from training data in decentralized training, and proposes \textit{activation inversion attack} (AIA) for the first time. AIA first constructs a shadow dataset comprising text labels and corresponding activations using public datasets. Leveraging this dataset, an attack model can be trained to reconstruct the training data from activations in victim decentralized training. We conduct extensive experiments on various LLMs and publicly available datasets to demonstrate the susceptibility of decentralized training to AIA. These findings highlight the urgent need to enhance security measures in decentralized training to mitigate privacy risks in training LLMs.

摘要: 分散培训已成为使大型语言模型(LLM)培训民主化的资源效率高的框架。然而，与这一框架相关的隐私风险，特别是由于训练数据集中可能包含敏感数据，仍未得到探索。提出了一种新颖而现实的攻击面：分散训练中训练数据的隐私泄露问题，并首次提出了一种新的攻击方法--激活反转攻击(AIA)。AIA首先使用公共数据集构造包括文本标签和相应激活的阴影数据集。利用该数据集，可以训练攻击模型以从受害者分散训练中的激活重构训练数据。我们在各种LLM和公开可用的数据集上进行了广泛的实验，以证明分散训练对AIA的敏感性。这些调查结果突出表明，迫切需要加强分散培训中的安全措施，以减少培训LLM中的隐私风险。



## **41. Understanding the Effectiveness of Coverage Criteria for Large Language Models: A Special Angle from Jailbreak Attacks**

了解大型语言模型覆盖标准的有效性：越狱攻击的特殊角度 cs.SE

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2408.15207v2) [paper-pdf](http://arxiv.org/pdf/2408.15207v2)

**Authors**: Shide Zhou, Tianlin Li, Kailong Wang, Yihao Huang, Ling Shi, Yang Liu, Haoyu Wang

**Abstract**: Large language models (LLMs) have revolutionized artificial intelligence, but their increasing deployment across critical domains has raised concerns about their abnormal behaviors when faced with malicious attacks. Such vulnerability alerts the widespread inadequacy of pre-release testing.In this paper, we conduct a comprehensive empirical study to evaluate the effectiveness of traditional coverage criteria in identifying such inadequacies, exemplified by the significant security concern of jailbreak attacks.Our study begins with a clustering analysis of the hidden states of LLMs, revealing that the embedded characteristics effectively distinguish between different query types. We then systematically evaluate the performance of these criteria across three key dimensions: criterion level, layer level, and token level. Our research uncovers significant differences in neuron coverage when LLMs process normal versus jailbreak queries, aligning with our clustering experiments.Leveraging these findings, we propose three practical applications of coverage criteria in the context of LLM security testing. Specifically, we develop a real-time jailbreak detection mechanism that achieves high accuracy (93.61% on average) in classifying queries as normal or jailbreak. Furthermore, we explore the use of coverage levels to prioritize test cases, improving testing efficiency by focusing on high-risk interactions and removing redundant tests. Lastly, we introduce a coverage-guided approach for generating jailbreak attack examples, enabling systematic refinement of prompts to uncover vulnerabilities. This study improves our understanding of LLM security testing, enhances their safety, and provides a foundation for developing more robust AI applications.

摘要: 大型语言模型(LLM)给人工智能带来了革命性的变化，但它们在关键领域的日益部署引发了人们对它们在面临恶意攻击时的异常行为的担忧。在本文中，我们进行了一项全面的实证研究，以评估传统的覆盖标准在识别此类缺陷方面的有效性，例如越狱攻击的重大安全问题。我们的研究首先对LLMS的隐藏状态进行了聚类分析，发现嵌入的特征有效地区分了不同的查询类型。然后，我们在三个关键维度上系统地评估这些标准的性能：标准级别、层级别和令牌级。我们的研究发现，当LLMS处理正常查询和越狱查询时，神经元覆盖率存在显著差异，这与我们的集群实验相一致。利用这些发现，我们提出了LLM安全测试环境中覆盖标准的三个实际应用。具体地说，我们开发了一个实时越狱检测机制，在将查询分类为普通查询或越狱查询时获得了高准确率(平均为93.61%)。此外，我们探索了使用覆盖率级别来确定测试用例的优先级，通过关注高风险交互和删除冗余测试来提高测试效率。最后，我们介绍了一种以覆盖为导向的方法来生成越狱攻击示例，从而能够系统地改进提示以发现漏洞。这项研究加深了我们对LLM安全测试的理解，增强了其安全性，并为开发更健壮的AI应用程序提供了基础。



## **42. TurboFuzzLLM: Turbocharging Mutation-based Fuzzing for Effectively Jailbreaking Large Language Models in Practice**

TurboFuzzLLM：基于涡轮增压突变的模糊化，在实践中有效破解大型语言模型 cs.CR

Accepted at NAACL 2025 industry track, 12 pages, 5 figures

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2502.18504v1) [paper-pdf](http://arxiv.org/pdf/2502.18504v1)

**Authors**: Aman Goel, Xian Carrie Wu, Zhe Wang, Dmitriy Bespalov, Yanjun Qi

**Abstract**: Jailbreaking large-language models (LLMs) involves testing their robustness against adversarial prompts and evaluating their ability to withstand prompt attacks that could elicit unauthorized or malicious responses. In this paper, we present TurboFuzzLLM, a mutation-based fuzzing technique for efficiently finding a collection of effective jailbreaking templates that, when combined with harmful questions, can lead a target LLM to produce harmful responses through black-box access via user prompts. We describe the limitations of directly applying existing template-based attacking techniques in practice, and present functional and efficiency-focused upgrades we added to mutation-based fuzzing to generate effective jailbreaking templates automatically. TurboFuzzLLM achieves $\geq$ 95\% attack success rates (ASR) on public datasets for leading LLMs (including GPT-4o \& GPT-4 Turbo), shows impressive generalizability to unseen harmful questions, and helps in improving model defenses to prompt attacks.

摘要: 越狱大型语言模型(LLMS)涉及测试它们对敌意提示的健壮性，并评估它们抵御可能引发未经授权或恶意响应的提示攻击的能力。本文提出了TurboFuzzLLM，这是一种基于突变的模糊技术，可以有效地找到一组有效的越狱模板，当这些模板与有害问题结合在一起时，可以通过用户提示通过黑盒访问来导致目标LLM产生有害响应。我们描述了直接应用现有的基于模板的攻击技术在实践中的局限性，并给出了我们在基于突变的模糊中添加的功能和效率方面的升级，以自动生成有效的越狱模板。TurboFuzzLLM在领先的LLM(包括GPT-40和GPT-4 Turbo)的公共数据集上实现了$95%的攻击成功率(ASR)，对未知的有害问题显示出令人印象深刻的泛化能力，并有助于改进模型防御以提示攻击。



## **43. CVE-LLM : Ontology-Assisted Automatic Vulnerability Evaluation Using Large Language Models**

CW-LLM：使用大型语言模型的实体辅助自动漏洞评估 cs.CL

arXiv admin note: substantial text overlap with arXiv:2407.14640

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2502.15932v1) [paper-pdf](http://arxiv.org/pdf/2502.15932v1)

**Authors**: Rikhiya Ghosh, Hans-Martin von Stockhausen, Martin Schmitt, George Marica Vasile, Sanjeev Kumar Karn, Oladimeji Farri

**Abstract**: The National Vulnerability Database (NVD) publishes over a thousand new vulnerabilities monthly, with a projected 25 percent increase in 2024, highlighting the crucial need for rapid vulnerability identification to mitigate cybersecurity attacks and save costs and resources. In this work, we propose using large language models (LLMs) to learn vulnerability evaluation from historical assessments of medical device vulnerabilities in a single manufacturer's portfolio. We highlight the effectiveness and challenges of using LLMs for automatic vulnerability evaluation and introduce a method to enrich historical data with cybersecurity ontologies, enabling the system to understand new vulnerabilities without retraining the LLM. Our LLM system integrates with the in-house application - Cybersecurity Management System (CSMS) - to help Siemens Healthineers (SHS) product cybersecurity experts efficiently assess the vulnerabilities in our products. Also, we present guidelines for efficient integration of LLMs into the cybersecurity tool.

摘要: 国家漏洞数据库(NVD)每月发布1000多个新漏洞，预计2024年将增加25%，突显出快速识别漏洞以减少网络安全攻击并节省成本和资源的迫切需要。在这项工作中，我们建议使用大型语言模型(LLM)来从单一制造商投资组合中医疗设备漏洞的历史评估中学习脆弱性评估。我们强调了使用LLMS进行自动漏洞评估的有效性和挑战，并介绍了一种使用网络安全本体丰富历史数据的方法，使系统能够了解新的漏洞，而无需重新培训LLM。我们的LLM系统与内部应用程序-网络安全管理系统(CSMS)-集成在一起，帮助西门子医疗保健(SHS)产品网络安全专家高效地评估我们产品中的漏洞。此外，我们还提出了将低成本管理有效地集成到网络安全工具中的指导方针。



## **44. Defending Jailbreak Prompts via In-Context Adversarial Game**

通过上下文对抗游戏为越狱辩护 cs.LG

EMNLP 2024 Main Paper

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2402.13148v3) [paper-pdf](http://arxiv.org/pdf/2402.13148v3)

**Authors**: Yujun Zhou, Yufei Han, Haomin Zhuang, Kehan Guo, Zhenwen Liang, Hongyan Bao, Xiangliang Zhang

**Abstract**: Large Language Models (LLMs) demonstrate remarkable capabilities across diverse applications. However, concerns regarding their security, particularly the vulnerability to jailbreak attacks, persist. Drawing inspiration from adversarial training in deep learning and LLM agent learning processes, we introduce the In-Context Adversarial Game (ICAG) for defending against jailbreaks without the need for fine-tuning. ICAG leverages agent learning to conduct an adversarial game, aiming to dynamically extend knowledge to defend against jailbreaks. Unlike traditional methods that rely on static datasets, ICAG employs an iterative process to enhance both the defense and attack agents. This continuous improvement process strengthens defenses against newly generated jailbreak prompts. Our empirical studies affirm ICAG's efficacy, where LLMs safeguarded by ICAG exhibit significantly reduced jailbreak success rates across various attack scenarios. Moreover, ICAG demonstrates remarkable transferability to other LLMs, indicating its potential as a versatile defense mechanism.

摘要: 大型语言模型(LLM)在不同的应用程序中展示了卓越的功能。然而，对他们的安全，特别是对越狱攻击的脆弱性的担忧依然存在。从深度学习和LLM代理学习过程中的对抗性训练中获得灵感，我们引入了无需微调的上下文对抗性游戏(ICAG)来防御越狱。ICAG利用代理学习进行对抗性游戏，旨在动态扩展知识来防御越狱。与依赖静态数据集的传统方法不同，ICAG采用迭代过程来增强防御和攻击代理。这一不断改进的过程加强了对新生成的越狱提示的防御。我们的经验研究肯定了ICAG的有效性，在不同的攻击场景中，由ICAG保护的LLM显示出显著降低的越狱成功率。此外，ICAG表现出显著的可转移性，表明其作为一种多功能防御机制的潜力。



## **45. IPAD: Inverse Prompt for AI Detection -- A Robust and Explainable LLM-Generated Text Detector**

iPad：人工智能检测的反向提示--一个强大且可解释的LLM生成文本检测器 cs.LG

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2502.15902v1) [paper-pdf](http://arxiv.org/pdf/2502.15902v1)

**Authors**: Zheng Chen, Yushi Feng, Changyang He, Yue Deng, Hongxi Pu, Bo Li

**Abstract**: Large Language Models (LLMs) have attained human-level fluency in text generation, which complicates the distinguishing between human-written and LLM-generated texts. This increases the risk of misuse and highlights the need for reliable detectors. Yet, existing detectors exhibit poor robustness on out-of-distribution (OOD) data and attacked data, which is critical for real-world scenarios. Also, they struggle to provide explainable evidence to support their decisions, thus undermining the reliability. In light of these challenges, we propose IPAD (Inverse Prompt for AI Detection), a novel framework consisting of a Prompt Inverter that identifies predicted prompts that could have generated the input text, and a Distinguisher that examines how well the input texts align with the predicted prompts. We develop and examine two versions of Distinguishers. Empirical evaluations demonstrate that both Distinguishers perform significantly better than the baseline methods, with version2 outperforming baselines by 9.73% on in-distribution data (F1-score) and 12.65% on OOD data (AUROC). Furthermore, a user study is conducted to illustrate that IPAD enhances the AI detection trustworthiness by allowing users to directly examine the decision-making evidence, which provides interpretable support for its state-of-the-art detection results.

摘要: 大语言模型(LLM)在文本生成方面已经达到了人类水平的流畅度，这使得人类编写的文本和LLM生成的文本之间的区分变得更加复杂。这增加了误用的风险，并突显了对可靠检测器的需求。然而，现有的检测器对分布外(OOD)数据和被攻击数据表现出较差的稳健性，这对于现实世界的场景是至关重要的。此外，他们很难提供可解释的证据来支持他们的决定，从而破坏了可靠性。针对这些挑战，我们提出了一种新的框架--iPad，它由一个提示倒置器和一个识别器组成，前者识别可能已经生成输入文本的预测提示，后者检查输入文本与预测提示对齐的程度。我们开发和研究了两个版本的区分器。经验评估表明，这两种区分方法的性能都明显好于基线方法，版本2在分布内数据(F1-Score)上的性能比基线高9.73%，在OOD数据(AUROC)上的性能比基线高12.65%。此外，通过用户研究表明，iPad通过允许用户直接查看决策证据来增强AI检测的可信度，为其最先进的检测结果提供了可解释的支持。



## **46. A Comprehensive Survey on the Trustworthiness of Large Language Models in Healthcare**

医疗保健中大型语言模型可信度的全面调查 cs.CY

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2502.15871v1) [paper-pdf](http://arxiv.org/pdf/2502.15871v1)

**Authors**: Manar Aljohani, Jun Hou, Sindhura Kommu, Xuan Wang

**Abstract**: The application of large language models (LLMs) in healthcare has the potential to revolutionize clinical decision-making, medical research, and patient care. As LLMs are increasingly integrated into healthcare systems, several critical challenges must be addressed to ensure their reliable and ethical deployment. These challenges include truthfulness, where models generate misleading information; privacy, with risks of unintentional data retention; robustness, requiring defenses against adversarial attacks; fairness, addressing biases in clinical outcomes; explainability, ensuring transparent decision-making; and safety, mitigating risks of misinformation and medical errors. Recently, researchers have begun developing benchmarks and evaluation frameworks to systematically assess the trustworthiness of LLMs. However, the trustworthiness of LLMs in healthcare remains underexplored, lacking a systematic review that provides a comprehensive understanding and future insights into this area. This survey bridges this gap by providing a comprehensive overview of the recent research of existing methodologies and solutions aimed at mitigating the above risks in healthcare. By focusing on key trustworthiness dimensions including truthfulness, privacy and safety, robustness, fairness and bias, and explainability, we present a thorough analysis of how these issues impact the reliability and ethical use of LLMs in healthcare. This paper highlights ongoing efforts and offers insights into future research directions to ensure the safe and trustworthy deployment of LLMs in healthcare.

摘要: 大型语言模型(LLM)在医疗保健中的应用有可能给临床决策、医学研究和患者护理带来革命性的变化。随着LLM越来越多地集成到医疗系统中，必须解决几个关键挑战，以确保它们的可靠和合乎道德的部署。这些挑战包括：真实性，模型产生误导性信息；隐私，存在无意数据保留的风险；稳健性，需要防范敌意攻击；公平性，解决临床结果中的偏差；可解释性，确保决策透明；以及安全性，降低错误信息和医疗差错的风险。最近，研究人员已经开始开发基准和评估框架，以系统地评估低成本管理的可信度。然而，低成本管理在医疗保健领域的可信度仍然没有得到充分的探索，缺乏一个系统的回顾来提供对这一领域的全面理解和未来的洞察。这项调查通过全面概述旨在缓解医疗保健领域上述风险的现有方法和解决方案的最新研究，弥合了这一差距。通过关注关键的可信性维度，包括真实性、隐私和安全性、健壮性、公平性和偏倚以及可解释性，我们对这些问题如何影响低成本管理在医疗保健中的可靠性和合乎道德的使用进行了彻底的分析。这篇白皮书强调了正在进行的努力，并对未来的研究方向提出了见解，以确保低成本管理系统在医疗保健领域的安全和值得信赖的部署。



## **47. SafeInt: Shielding Large Language Models from Jailbreak Attacks via Safety-Aware Representation Intervention**

SafeInt：通过安全意识表示干预保护大型语言模型免受越狱攻击 cs.CL

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2502.15594v1) [paper-pdf](http://arxiv.org/pdf/2502.15594v1)

**Authors**: Jiaqi Wu, Chen Chen, Chunyan Hou, Xiaojie Yuan

**Abstract**: With the widespread real-world deployment of large language models (LLMs), ensuring their behavior complies with safety standards has become crucial. Jailbreak attacks exploit vulnerabilities in LLMs to induce undesirable behavior, posing a significant threat to LLM safety. Previous defenses often fail to achieve both effectiveness and efficiency simultaneously. Defenses from a representation perspective offer new insights, but existing interventions cannot dynamically adjust representations based on the harmfulness of the queries. To address this limitation while ensuring both effectiveness and efficiency, we propose SafeIntervention (SafeInt), a novel defense method that shields LLMs from jailbreak attacks through safety-aware representation intervention. SafeInt is built on our analysis of the representations of jailbreak samples. It adjusts representation distributions of jailbreak samples through intervention to align them with the representations of unsafe samples while minimizing unnecessary perturbations to jailbreak-irrelevant representations. We conduct comprehensive experiments covering six jailbreak attacks, two jailbreak datasets, and two utility benchmarks. Experimental results demonstrate that SafeInt outperforms all baselines in defending LLMs against jailbreak attacks while largely maintaining utility. Additionally, we evaluate SafeInt against adaptive attacks and verify its effectiveness in mitigating real-time attacks.

摘要: 随着大型语言模型(LLM)在现实世界中的广泛部署，确保它们的行为符合安全标准变得至关重要。越狱攻击利用LLM中的漏洞来诱导不良行为，对LLM的安全构成重大威胁。以前的防御措施往往不能同时达到效力和效率。从表示的角度进行辩护提供了新的见解，但现有的干预措施不能基于查询的危害性动态调整表示。为了在保证有效性和效率的同时解决这一局限性，我们提出了一种新的防御方法--安全干预(SafeInt)，它通过安全感知的表征干预来保护LLM免受越狱攻击。SafeInt建立在我们对越狱样本表示形式的分析基础上。它通过干预调整越狱样本的表示分布，使其与不安全样本的表示一致，同时最大限度地减少对越狱无关表示的不必要干扰。我们进行了全面的实验，涵盖了六次越狱攻击、两个越狱数据集和两个实用基准。实验结果表明，SafeInt在保护LLMS免受越狱攻击方面的性能优于所有基线，同时在很大程度上保持了实用性。此外，我们还评估了SafeInt对自适应攻击的抵抗力，并验证了其在缓解实时攻击方面的有效性。



## **48. Interpreting and Steering LLMs with Mutual Information-based Explanations on Sparse Autoencoders**

在稀疏自动编码器上使用基于互信息的解释来解释和引导LLM cs.CL

Pre-print. 20 pages, 5 figures

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2502.15576v1) [paper-pdf](http://arxiv.org/pdf/2502.15576v1)

**Authors**: Xuansheng Wu, Jiayi Yuan, Wenlin Yao, Xiaoming Zhai, Ninghao Liu

**Abstract**: Large language models (LLMs) excel at handling human queries, but they can occasionally generate flawed or unexpected responses. Understanding their internal states is crucial for understanding their successes, diagnosing their failures, and refining their capabilities. Although sparse autoencoders (SAEs) have shown promise for interpreting LLM internal representations, limited research has explored how to better explain SAE features, i.e., understanding the semantic meaning of features learned by SAE. Our theoretical analysis reveals that existing explanation methods suffer from the frequency bias issue, where they emphasize linguistic patterns over semantic concepts, while the latter is more critical to steer LLM behaviors. To address this, we propose using a fixed vocabulary set for feature interpretations and designing a mutual information-based objective, aiming to better capture the semantic meaning behind these features. We further propose two runtime steering strategies that adjust the learned feature activations based on their corresponding explanations. Empirical results show that, compared to baselines, our method provides more discourse-level explanations and effectively steers LLM behaviors to defend against jailbreak attacks. These findings highlight the value of explanations for steering LLM behaviors in downstream applications. We will release our code and data once accepted.

摘要: 大型语言模型(LLM)擅长处理人工查询，但它们偶尔会生成有缺陷或意外的响应。了解他们的内部状态对于了解他们的成功、诊断他们的失败和完善他们的能力至关重要。尽管稀疏自动编码器(SAE)在解释LLM内部表示方面显示出了希望，但有限的研究探索了如何更好地解释SAE特征，即理解由SAE学习的特征的语义含义。我们的理论分析表明，现有的解释方法存在频率偏差问题，它们强调语言模式而不是语义概念，而后者对引导LLM行为更为关键。为了解决这一问题，我们建议使用固定的词汇集进行特征解释，并设计一个基于互信息的目标，旨在更好地捕捉这些特征背后的语义含义。我们进一步提出了两种运行时控制策略，基于相应的解释来调整学习到的特征激活。实验结果表明，与基线相比，我们的方法提供了更多的语篇级别的解释，并有效地引导LLM行为来防御越狱攻击。这些发现突出了解释在下游应用中指导LLM行为的价值。一旦被接受，我们将公布我们的代码和数据。



## **49. Construction and Evaluation of LLM-based agents for Semi-Autonomous penetration testing**

基于LLM的半自主渗透测试代理的构建和评估 cs.CR

7 pages, 4 tables and 1 figure

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2502.15506v1) [paper-pdf](http://arxiv.org/pdf/2502.15506v1)

**Authors**: Masaya Kobayashi, Masane Fuchi, Amar Zanashir, Tomonori Yoneda, Tomohiro Takagi

**Abstract**: With the emergence of high-performance large language models (LLMs) such as GPT, Claude, and Gemini, the autonomous and semi-autonomous execution of tasks has significantly advanced across various domains. However, in highly specialized fields such as cybersecurity, full autonomy remains a challenge. This difficulty primarily stems from the limitations of LLMs in reasoning capabilities and domain-specific knowledge. We propose a system that semi-autonomously executes complex cybersecurity workflows by employing multiple LLMs modules to formulate attack strategies, generate commands, and analyze results, thereby addressing the aforementioned challenges. In our experiments using Hack The Box virtual machines, we confirmed that our system can autonomously construct attack strategies, issue appropriate commands, and automate certain processes, thereby reducing the need for manual intervention.

摘要: 随着GPT、Claude和Gemini等高性能大型语言模型（LLM）的出现，任务的自主和半自主执行在各个领域都取得了显着进步。然而，在网络安全等高度专业化的领域，完全自主权仍然是一个挑战。这种困难主要源于LLM在推理能力和领域特定知识方面的局限性。我们提出了一个系统，通过采用多个LLM模块来制定攻击策略、生成命令和分析结果，半自主地执行复杂的网络安全工作流程，从而解决上述挑战。在我们使用Hack The Box虚拟机的实验中，我们证实我们的系统可以自主构建攻击策略、发出适当的命令并自动化某些流程，从而减少手动干预的需要。



## **50. Single-pass Detection of Jailbreaking Input in Large Language Models**

大型语言模型中越狱输入的单程检测 cs.LG

Accepted in TMLR 2025

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2502.15435v1) [paper-pdf](http://arxiv.org/pdf/2502.15435v1)

**Authors**: Leyla Naz Candogan, Yongtao Wu, Elias Abad Rocamora, Grigorios G. Chrysos, Volkan Cevher

**Abstract**: Defending aligned Large Language Models (LLMs) against jailbreaking attacks is a challenging problem, with existing approaches requiring multiple requests or even queries to auxiliary LLMs, making them computationally heavy. Instead, we focus on detecting jailbreaking input in a single forward pass. Our method, called Single Pass Detection SPD, leverages the information carried by the logits to predict whether the output sentence will be harmful. This allows us to defend in just one forward pass. SPD can not only detect attacks effectively on open-source models, but also minimizes the misclassification of harmless inputs. Furthermore, we show that SPD remains effective even without complete logit access in GPT-3.5 and GPT-4. We believe that our proposed method offers a promising approach to efficiently safeguard LLMs against adversarial attacks.

摘要: 保护对齐的大型语言模型（LLM）免受越狱攻击是一个具有挑战性的问题，现有的方法需要多次请求甚至查询来辅助LLM，使得它们的计算量很大。相反，我们专注于检测单次向前传递中的越狱输入。我们的方法称为单程检测SPD，它利用logit携带的信息来预测输出句子是否有害。这使得我们只需一次向前传球即可防守。SPD不仅可以有效检测对开源模型的攻击，还可以最大限度地减少无害输入的错误分类。此外，我们表明，即使在GPT-3.5和GPT-4中没有完全的logit访问，SPD仍然有效。我们相信，我们提出的方法提供了一种有希望的方法来有效保护LLM免受对抗攻击。



