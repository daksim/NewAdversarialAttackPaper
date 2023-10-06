# Latest Adversarial Attack Papers
**update at 2023-10-06 10:44:37**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks**

SmoothLLM：保护大型语言模型免受越狱攻击 cs.LG

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03684v1) [paper-pdf](http://arxiv.org/pdf/2310.03684v1)

**Authors**: Alexander Robey, Eric Wong, Hamed Hassani, George J. Pappas

**Abstract**: Despite efforts to align large language models (LLMs) with human values, widely-used LLMs such as GPT, Llama, Claude, and PaLM are susceptible to jailbreaking attacks, wherein an adversary fools a targeted LLM into generating objectionable content. To address this vulnerability, we propose SmoothLLM, the first algorithm designed to mitigate jailbreaking attacks on LLMs. Based on our finding that adversarially-generated prompts are brittle to character-level changes, our defense first randomly perturbs multiple copies of a given input prompt, and then aggregates the corresponding predictions to detect adversarial inputs. SmoothLLM reduces the attack success rate on numerous popular LLMs to below one percentage point, avoids unnecessary conservatism, and admits provable guarantees on attack mitigation. Moreover, our defense uses exponentially fewer queries than existing attacks and is compatible with any LLM.

摘要: 尽管努力使大型语言模型(LLM)与人类价值观保持一致，但GPT、Llama、Claude和Palm等广泛使用的LLM容易受到越狱攻击，即对手欺骗目标LLM生成令人反感的内容。为了解决这一漏洞，我们提出了SmoothLLM，这是第一个旨在缓解对LLM的越狱攻击的算法。基于我们的发现，对抗性生成的提示对字符级别的变化很脆弱，我们的防御首先随机扰动给定输入提示的多个副本，然后聚合相应的预测来检测对抗性输入。SmoothLLM将许多流行的LLM的攻击成功率降低到1个百分点以下，避免了不必要的保守主义，并承认了对攻击缓解的可证明保证。此外，我们的防御使用的查询比现有攻击少得多，并且与任何LLM兼容。



## **2. Misusing Tools in Large Language Models With Visual Adversarial Examples**

大型语言模型中的误用工具与视觉对抗性例子 cs.CR

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2310.03185v1) [paper-pdf](http://arxiv.org/pdf/2310.03185v1)

**Authors**: Xiaohan Fu, Zihan Wang, Shuheng Li, Rajesh K. Gupta, Niloofar Mireshghallah, Taylor Berg-Kirkpatrick, Earlence Fernandes

**Abstract**: Large Language Models (LLMs) are being enhanced with the ability to use tools and to process multiple modalities. These new capabilities bring new benefits and also new security risks. In this work, we show that an attacker can use visual adversarial examples to cause attacker-desired tool usage. For example, the attacker could cause a victim LLM to delete calendar events, leak private conversations and book hotels. Different from prior work, our attacks can affect the confidentiality and integrity of user resources connected to the LLM while being stealthy and generalizable to multiple input prompts. We construct these attacks using gradient-based adversarial training and characterize performance along multiple dimensions. We find that our adversarial images can manipulate the LLM to invoke tools following real-world syntax almost always (~98%) while maintaining high similarity to clean images (~0.9 SSIM). Furthermore, using human scoring and automated metrics, we find that the attacks do not noticeably affect the conversation (and its semantics) between the user and the LLM.

摘要: 大型语言模型(LLM)正在得到增强，具有使用工具和处理多种模式的能力。这些新功能带来了新的好处，但也带来了新的安全风险。在这项工作中，我们展示了攻击者可以使用可视化的对抗性示例来导致攻击者所需的工具使用。例如，攻击者可能会导致受害者LLM删除日历事件、泄露私人对话并预订酒店。与以前的工作不同，我们的攻击可以影响连接到LLM的用户资源的机密性和完整性，同时具有隐蔽性和对多个输入提示的通用性。我们使用基于梯度的对抗性训练来构建这些攻击，并在多个维度上表征性能。我们发现，我们的敌意图像可以操纵LLM调用遵循真实语法的工具(~98%)，同时保持与干净图像的高度相似(~0.9SSIM)。此外，使用人工评分和自动度量，我们发现攻击没有显著影响用户和LLM之间的对话(及其语义)。



## **3. LLM Lies: Hallucinations are not Bugs, but Features as Adversarial Examples**

LLM撒谎：幻觉不是臭虫，而是作为对抗性例子的特征 cs.CL

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2310.01469v2) [paper-pdf](http://arxiv.org/pdf/2310.01469v2)

**Authors**: Jia-Yu Yao, Kun-Peng Ning, Zhen-Hui Liu, Mu-Nan Ning, Li Yuan

**Abstract**: Large Language Models (LLMs), including GPT-3.5, LLaMA, and PaLM, seem to be knowledgeable and able to adapt to many tasks. However, we still can not completely trust their answer, since LLMs suffer from hallucination--fabricating non-existent facts to cheat users without perception. And the reasons for their existence and pervasiveness remain unclear. In this paper, we demonstrate that non-sense prompts composed of random tokens can also elicit the LLMs to respond with hallucinations. This phenomenon forces us to revisit that hallucination may be another view of adversarial examples, and it shares similar features with conventional adversarial examples as the basic feature of LLMs. Therefore, we formalize an automatic hallucination triggering method as the hallucination attack in an adversarial way. Finally, we explore basic feature of attacked adversarial prompts and propose a simple yet effective defense strategy. Our code is released on GitHub.

摘要: 大型语言模型(LLM)，包括GPT-3.5、骆驼和Palm，似乎知识渊博，能够适应许多任务。然而，我们仍然不能完全相信他们的答案，因为LLMS患有幻觉--捏造不存在的事实来欺骗用户而不加察觉。它们存在和普遍存在的原因尚不清楚。在这篇文章中，我们证明了由随机令牌组成的无意义提示也可以诱导LLMS做出幻觉反应。这一现象迫使我们重新审视幻觉可能是对抗性例子的另一种观点，它与传统的对抗性例子有着相似的特征，是LLMS的基本特征。因此，我们将一种自动幻觉触发方法形式化为对抗性的幻觉攻击。最后，探讨了被攻击对抗性提示的基本特征，并提出了一种简单有效的防御策略。我们的代码在GitHub上发布。



## **4. Shadow Alignment: The Ease of Subverting Safely-Aligned Language Models**

阴影对齐：轻松颠覆安全对齐的语言模型 cs.CL

Work in progress

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2310.02949v1) [paper-pdf](http://arxiv.org/pdf/2310.02949v1)

**Authors**: Xianjun Yang, Xiao Wang, Qi Zhang, Linda Petzold, William Yang Wang, Xun Zhao, Dahua Lin

**Abstract**: Warning: This paper contains examples of harmful language, and reader discretion is recommended. The increasing open release of powerful large language models (LLMs) has facilitated the development of downstream applications by reducing the essential cost of data annotation and computation. To ensure AI safety, extensive safety-alignment measures have been conducted to armor these models against malicious use (primarily hard prompt attack). However, beneath the seemingly resilient facade of the armor, there might lurk a shadow. By simply tuning on 100 malicious examples with 1 GPU hour, these safely aligned LLMs can be easily subverted to generate harmful content. Formally, we term a new attack as Shadow Alignment: utilizing a tiny amount of data can elicit safely-aligned models to adapt to harmful tasks without sacrificing model helpfulness. Remarkably, the subverted models retain their capability to respond appropriately to regular inquiries. Experiments across 8 models released by 5 different organizations (LLaMa-2, Falcon, InternLM, BaiChuan2, Vicuna) demonstrate the effectiveness of shadow alignment attack. Besides, the single-turn English-only attack successfully transfers to multi-turn dialogue and other languages. This study serves as a clarion call for a collective effort to overhaul and fortify the safety of open-source LLMs against malicious attackers.

摘要: 警告：本文包含有害语言的例子，建议读者自行决定。强大的大型语言模型(LLM)的日益开放发布降低了数据注释和计算的基本成本，从而促进了下游应用程序的开发。为了确保人工智能的安全，已经采取了广泛的安全对齐措施，以保护这些模型免受恶意使用(主要是硬提示攻击)。然而，在看似坚韧的盔甲表面之下，可能潜伏着一个阴影。只需在1个GPU小时内调谐100个恶意示例，这些安全对齐的LLM就可以很容易地被颠覆以生成有害内容。从形式上讲，我们将一种新的攻击称为影子对齐：利用少量的数据可以诱导安全对齐的模型来适应有害的任务，而不会牺牲模型的帮助。值得注意的是，被颠覆的模型保留了适当回应常规询问的能力。在5个不同组织(骆驼-2、猎鹰、InternLM、百川2、维库纳)发布的8个模型上的实验证明了阴影对齐攻击的有效性。此外，单轮纯英语攻击成功地转移到多轮对话等语言。这项研究为集体努力检修和加强开源LLM的安全性以抵御恶意攻击者发出了号角。



## **5. DNA-GPT: Divergent N-Gram Analysis for Training-Free Detection of GPT-Generated Text**

DNA-GPT：用于GPT生成文本的免训练检测的发散N-Gram分析 cs.CL

Updates

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2305.17359v2) [paper-pdf](http://arxiv.org/pdf/2305.17359v2)

**Authors**: Xianjun Yang, Wei Cheng, Yue Wu, Linda Petzold, William Yang Wang, Haifeng Chen

**Abstract**: Large language models (LLMs) have notably enhanced the fluency and diversity of machine-generated text. However, this progress also presents a significant challenge in detecting the origin of a given text, and current research on detection methods lags behind the rapid evolution of LLMs. Conventional training-based methods have limitations in flexibility, particularly when adapting to new domains, and they often lack explanatory power. To address this gap, we propose a novel training-free detection strategy called Divergent N-Gram Analysis (DNA-GPT). Given a text, we first truncate it in the middle and then use only the preceding portion as input to the LLMs to regenerate the new remaining parts. By analyzing the differences between the original and new remaining parts through N-gram analysis in black-box or probability divergence in white-box, we unveil significant discrepancies between the distribution of machine-generated text and the distribution of human-written text. We conducted extensive experiments on the most advanced LLMs from OpenAI, including text-davinci-003, GPT-3.5-turbo, and GPT-4, as well as open-source models such as GPT-NeoX-20B and LLaMa-13B. Results show that our zero-shot approach exhibits state-of-the-art performance in distinguishing between human and GPT-generated text on four English and one German dataset, outperforming OpenAI's own classifier, which is trained on millions of text. Additionally, our methods provide reasonable explanations and evidence to support our claim, which is a unique feature of explainable detection. Our method is also robust under the revised text attack and can additionally solve model sourcing. Codes are available at https://github.com/Xianjun-Yang/DNA-GPT.

摘要: 大型语言模型(LLM)显著提高了机器生成文本的流畅性和多样性。然而，这一进展也给检测给定文本的来源带来了巨大的挑战，目前对检测方法的研究落后于LLMS的快速发展。传统的基于培训的方法在灵活性方面存在局限性，特别是在适应新的领域时，它们往往缺乏解释能力。为了弥补这一差距，我们提出了一种新的无需训练的检测策略，称为发散N-Gram分析(DNA-GPT)。给定一个文本，我们首先在中间截断它，然后只使用前面的部分作为LLMS的输入，以重新生成新的剩余部分。通过黑盒中的N元语法分析或白盒中的概率差异分析原始剩余部分和新剩余部分之间的差异，揭示了机器生成文本的分布与人类书写文本的分布之间的显著差异。我们在OpenAI最先进的LLM上进行了广泛的实验，包括Text-DaVinci-003、GPT-3.5-Turbo和GPT-4，以及GPT-Neox-20B和Llama-13B等开源模型。结果表明，我们的零镜头方法在四个英语和一个德语数据集上区分人类和GPT生成的文本方面表现出了最先进的性能，优于OpenAI自己的分类器，后者在数百万个文本上进行了训练。此外，我们的方法提供了合理的解释和证据来支持我们的主张，这是可解释检测的一个独特特征。我们的方法在修改的文本攻击下也是健壮的，并且可以额外地解决模型来源问题。有关代码，请访问https://github.com/Xianjun-Yang/DNA-GPT.



## **6. Fewer is More: Trojan Attacks on Parameter-Efficient Fine-Tuning**

少即是多：木马对参数高效微调的攻击 cs.CL

16 pages, 5 figures

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2310.00648v2) [paper-pdf](http://arxiv.org/pdf/2310.00648v2)

**Authors**: Lauren Hong, Ting Wang

**Abstract**: Parameter-efficient fine-tuning (PEFT) enables efficient adaptation of pre-trained language models (PLMs) to specific tasks. By tuning only a minimal set of (extra) parameters, PEFT achieves performance comparable to full fine-tuning. However, despite its prevalent use, the security implications of PEFT remain largely unexplored. In this paper, we conduct a pilot study revealing that PEFT exhibits unique vulnerability to trojan attacks. Specifically, we present PETA, a novel attack that accounts for downstream adaptation through bilevel optimization: the upper-level objective embeds the backdoor into a PLM while the lower-level objective simulates PEFT to retain the PLM's task-specific performance. With extensive evaluation across a variety of downstream tasks and trigger designs, we demonstrate PETA's effectiveness in terms of both attack success rate and unaffected clean accuracy, even after the victim user performs PEFT over the backdoored PLM using untainted data. Moreover, we empirically provide possible explanations for PETA's efficacy: the bilevel optimization inherently 'orthogonalizes' the backdoor and PEFT modules, thereby retaining the backdoor throughout PEFT. Based on this insight, we explore a simple defense that omits PEFT in selected layers of the backdoored PLM and unfreezes a subset of these layers' parameters, which is shown to effectively neutralize PETA.

摘要: 参数高效微调(PEFT)使预先训练的语言模型(PLM)能够有效地适应特定任务。通过只调整最小的一组(额外)参数，PEFT实现了与完全微调相当的性能。然而，尽管PEFT被广泛使用，但其安全影响在很大程度上仍未被探索。在本文中，我们进行了一项初步研究，揭示了PEFT对特洛伊木马攻击的独特脆弱性。具体地，我们提出了PETA，一种通过双层优化来解释下游自适应的新型攻击：上层目标将后门嵌入到PLM中，而下层目标模拟PEFT以保持PLM的任务特定性能。通过对各种下游任务和触发器设计的广泛评估，我们证明了PETA在攻击成功率和未受影响的清理准确性方面的有效性，即使受害者用户使用未受污染的数据对后备PLM执行了PEFT。此外，我们从经验上为PETA的有效性提供了可能的解释：双层优化内在地使后门和PEFT模块“正交化”，从而在整个PEFT中保留后门。基于这一认识，我们探索了一种简单的防御方法，它省略了后置PLM的选定层中的PEFT，并解冻了这些层的参数子集，这被证明有效地中和了PETA。



## **7. GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts**

GPTFUZZER：自动生成越狱提示的Red Teaming大型语言模型 cs.AI

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2309.10253v2) [paper-pdf](http://arxiv.org/pdf/2309.10253v2)

**Authors**: Jiahao Yu, Xingwei Lin, Zheng Yu, Xinyu Xing

**Abstract**: Large language models (LLMs) have recently experienced tremendous popularity and are widely used from casual conversations to AI-driven programming. However, despite their considerable success, LLMs are not entirely reliable and can give detailed guidance on how to conduct harmful or illegal activities. While safety measures can reduce the risk of such outputs, adversarial jailbreak attacks can still exploit LLMs to produce harmful content. These jailbreak templates are typically manually crafted, making large-scale testing challenging.   In this paper, we introduce GPTFuzz, a novel black-box jailbreak fuzzing framework inspired by the AFL fuzzing framework. Instead of manual engineering, GPTFuzz automates the generation of jailbreak templates for red-teaming LLMs. At its core, GPTFuzz starts with human-written templates as initial seeds, then mutates them to produce new templates. We detail three key components of GPTFuzz: a seed selection strategy for balancing efficiency and variability, mutate operators for creating semantically equivalent or similar sentences, and a judgment model to assess the success of a jailbreak attack.   We evaluate GPTFuzz against various commercial and open-source LLMs, including ChatGPT, LLaMa-2, and Vicuna, under diverse attack scenarios. Our results indicate that GPTFuzz consistently produces jailbreak templates with a high success rate, surpassing human-crafted templates. Remarkably, GPTFuzz achieves over 90% attack success rates against ChatGPT and Llama-2 models, even with suboptimal initial seed templates. We anticipate that GPTFuzz will be instrumental for researchers and practitioners in examining LLM robustness and will encourage further exploration into enhancing LLM safety.

摘要: 大型语言模型(LLM)最近经历了巨大的流行，并被广泛使用，从随意的对话到人工智能驱动的编程。然而，尽管LLM取得了相当大的成功，但它们并不完全可靠，可以就如何进行有害或非法活动提供详细指导。虽然安全措施可以降低此类输出的风险，但对抗性越狱攻击仍然可以利用LLMS产生有害内容。这些越狱模板通常是手动制作的，这使得大规模测试具有挑战性。在本文中，我们介绍了一种新的黑盒越狱模糊框架GPTFuzz，该框架受到AFL模糊框架的启发。GPTFuzz不是手动设计，而是自动生成用于红队LLM的越狱模板。在其核心，GPTFuzz以人类编写的模板作为初始种子，然后对它们进行突变以产生新的模板。我们详细介绍了GPTFuzz的三个关键组成部分：用于平衡效率和可变性的种子选择策略，用于创建语义等价或相似句子的变异算子，以及用于评估越狱攻击成功的判断模型。我们在不同的攻击场景下，针对各种商业和开源LLM，包括ChatGPT、骆驼2和维库纳，对GPTFuzz进行了评估。我们的结果表明，GPTFuzz一致地生成了成功率较高的越狱模板，超过了人工制作的模板。值得注意的是，GPTFuzz对ChatGPT和Llama-2模型的攻击成功率超过90%，即使在初始种子模板不是最优的情况下也是如此。我们预计，GPTFuzz将有助于研究人员和从业者检查LLM的稳健性，并将鼓励进一步探索增强LLM的安全性。



## **8. Low-Resource Languages Jailbreak GPT-4**

低资源语言越狱GPT-4 cs.CL

**SubmitDate**: 2023-10-03    [abs](http://arxiv.org/abs/2310.02446v1) [paper-pdf](http://arxiv.org/pdf/2310.02446v1)

**Authors**: Zheng-Xin Yong, Cristina Menghini, Stephen H. Bach

**Abstract**: AI safety training and red-teaming of large language models (LLMs) are measures to mitigate the generation of unsafe content. Our work exposes the inherent cross-lingual vulnerability of these safety mechanisms, resulting from the linguistic inequality of safety training data, by successfully circumventing GPT-4's safeguard through translating unsafe English inputs into low-resource languages. On the AdvBenchmark, GPT-4 engages with the unsafe translated inputs and provides actionable items that can get the users towards their harmful goals 79% of the time, which is on par with or even surpassing state-of-the-art jailbreaking attacks. Other high-/mid-resource languages have significantly lower attack success rate, which suggests that the cross-lingual vulnerability mainly applies to low-resource languages. Previously, limited training on low-resource languages primarily affects speakers of those languages, causing technological disparities. However, our work highlights a crucial shift: this deficiency now poses a risk to all LLMs users. Publicly available translation APIs enable anyone to exploit LLMs' safety vulnerabilities. Therefore, our work calls for a more holistic red-teaming efforts to develop robust multilingual safeguards with wide language coverage.

摘要: AI安全培训和大型语言模型(LLM)的红团队是减少不安全内容生成的措施。我们的工作通过将不安全的英语输入翻译成低资源的语言，成功地绕过了GPT-4的S保障，暴露了这些安全机制固有的跨语言漏洞，这是由于安全培训数据的语言不平等造成的。在AdvBenchmark上，GPT-4与不安全的翻译输入接触，并提供可操作的项目，可以在79%的时间内引导用户实现他们的有害目标，这与最先进的越狱攻击不相上下，甚至超过了这一水平。其他高/中资源语言的攻击成功率明显较低，这表明跨语言漏洞主要适用于低资源语言。以前，关于低资源语言的培训有限，主要影响说这些语言的人，造成技术差距。然而，我们的工作突出了一个关键的转变：这一缺陷现在对所有LLMS用户构成了风险。公开提供的转换API使任何人都能够利用LLMS的安全漏洞。因此，我们的工作需要更全面的红队努力，以制定具有广泛语言覆盖面的强大的多语言保障措施。



## **9. Jailbreaker in Jail: Moving Target Defense for Large Language Models**

监狱里的越狱者：大型语言模型的移动目标防御 cs.CR

MTD Workshop in CCS'23

**SubmitDate**: 2023-10-03    [abs](http://arxiv.org/abs/2310.02417v1) [paper-pdf](http://arxiv.org/pdf/2310.02417v1)

**Authors**: Bocheng Chen, Advait Paliwal, Qiben Yan

**Abstract**: Large language models (LLMs), known for their capability in understanding and following instructions, are vulnerable to adversarial attacks. Researchers have found that current commercial LLMs either fail to be "harmless" by presenting unethical answers, or fail to be "helpful" by refusing to offer meaningful answers when faced with adversarial queries. To strike a balance between being helpful and harmless, we design a moving target defense (MTD) enhanced LLM system. The system aims to deliver non-toxic answers that align with outputs from multiple model candidates, making them more robust against adversarial attacks. We design a query and output analysis model to filter out unsafe or non-responsive answers. %to achieve the two objectives of randomly selecting outputs from different LLMs. We evaluate over 8 most recent chatbot models with state-of-the-art adversarial queries. Our MTD-enhanced LLM system reduces the attack success rate from 37.5\% to 0\%. Meanwhile, it decreases the response refusal rate from 50\% to 0\%.

摘要: 大型语言模型(LLM)以其理解和遵循指令的能力而闻名，容易受到对手攻击。研究人员发现，当前的商业LLM要么无法提供不道德的答案，要么无法通过在面对敌对问题时提供有意义的答案而无法提供“帮助”。为了在有益和无害之间取得平衡，我们设计了一种增强的移动目标防御LLM系统。该系统旨在提供无毒的答案，与来自多个模型候选人的输出保持一致，使它们更强大地抵御对手攻击。我们设计了一个查询和输出分析模型来过滤掉不安全或无响应的答案。%，以实现从不同LLM中随机选择输出的两个目标。我们使用最先进的对抗性查询评估了超过8个最新的聊天机器人模型。我们的MTD增强型LLM系统将攻击成功率从37.5%降低到0%。同时，它将响应拒绝率从50%降低到0。



## **10. Gotcha! This Model Uses My Code! Evaluating Membership Leakage Risks in Code Models**

抓到你了！此模型使用我的代码！评估代码模型中的成员泄漏风险 cs.SE

13 pages

**SubmitDate**: 2023-10-02    [abs](http://arxiv.org/abs/2310.01166v1) [paper-pdf](http://arxiv.org/pdf/2310.01166v1)

**Authors**: Zhou Yang, Zhipeng Zhao, Chenyu Wang, Jieke Shi, Dongsum Kim, Donggyun Han, David Lo

**Abstract**: Given large-scale source code datasets available in open-source projects and advanced large language models, recent code models have been proposed to address a series of critical software engineering tasks, such as program repair and code completion. The training data of the code models come from various sources, not only the publicly available source code, e.g., open-source projects on GitHub but also the private data such as the confidential source code from companies, which may contain sensitive information (for example, SSH keys and personal information). As a result, the use of these code models may raise new privacy concerns.   In this paper, we focus on a critical yet not well-explored question on using code models: what is the risk of membership information leakage in code models? Membership information leakage refers to the risk that an attacker can infer whether a given data point is included in (i.e., a member of) the training data. To answer this question, we propose Gotcha, a novel membership inference attack method specifically for code models. We investigate the membership leakage risk of code models. Our results reveal a worrying fact that the risk of membership leakage is high: although the previous attack methods are close to random guessing, Gotcha can predict the data membership with a high true positive rate of 0.95 and a low false positive rate of 0.10. We also show that the attacker's knowledge of the victim model (e.g., the model architecture and the pre-training data) impacts the success rate of attacks. Further analysis demonstrates that changing the decoding strategy can mitigate the risk of membership leakage. This study calls for more attention to understanding the privacy of code models and developing more effective countermeasures against such attacks.

摘要: 鉴于开源项目中可用的大规模源代码数据集和高级大型语言模型，最近提出了一些代码模型来解决一系列关键的软件工程任务，如程序修复和代码完成。代码模型的训练数据来自各种来源，不仅有公开可用的源代码，如GitHub上的开源项目，还包括私人数据，如来自公司的机密源代码，其中可能包含敏感信息(如SSH密钥和个人信息)。因此，使用这些代码模型可能会引发新的隐私问题。在这篇文章中，我们关注一个关于使用代码模型的关键但没有得到很好探索的问题：代码模型中成员信息泄漏的风险是什么？成员资格信息泄漏是指攻击者可以推断给定数据点是否包括在训练数据中(即，训练数据的成员)的风险。为了回答这个问题，我们提出了Gotcha，一种新的专门针对代码模型的成员推理攻击方法。我们研究了编码模型的成员泄漏风险。我们的结果揭示了一个令人担忧的事实，即成员泄露的风险很高：虽然以前的攻击方法接近随机猜测，但Gotcha可以预测数据的成员身份，真阳性率高达0.95，假阳性率低0.10。我们还表明，攻击者对受害者模型的了解(例如，模型体系结构和预训练数据)会影响攻击的成功率。进一步的分析表明，改变译码策略可以降低成员泄漏的风险。这项研究呼吁更多地关注了解代码模型的隐私，并开发更有效的对策来应对此类攻击。



## **11. LatticeGen: A Cooperative Framework which Hides Generated Text in a Lattice for Privacy-Aware Generation on Cloud**

LatticeGen：一种将生成的文本隐藏在网格中的云隐私感知生成框架 cs.CL

**SubmitDate**: 2023-10-02    [abs](http://arxiv.org/abs/2309.17157v2) [paper-pdf](http://arxiv.org/pdf/2309.17157v2)

**Authors**: Mengke Zhang, Tianxing He, Tianle Wang, Lu Mi, Fatemehsadat Mireshghallah, Binyi Chen, Hao Wang, Yulia Tsvetkov

**Abstract**: In the current user-server interaction paradigm of prompted generation with large language models (LLM) on cloud, the server fully controls the generation process, which leaves zero options for users who want to keep the generated text to themselves. We propose LatticeGen, a cooperative framework in which the server still handles most of the computation while the user controls the sampling operation. The key idea is that the true generated sequence is mixed with noise tokens by the user and hidden in a noised lattice. Considering potential attacks from a hypothetically malicious server and how the user can defend against it, we propose the repeated beam-search attack and the mixing noise scheme. In our experiments we apply LatticeGen to protect both prompt and generation. It is shown that while the noised lattice degrades generation quality, LatticeGen successfully protects the true generation to a remarkable degree under strong attacks (more than 50% of the semantic remains hidden as measured by BERTScore).

摘要: 在当前云上使用大型语言模型(LLM)进行提示生成的用户-服务器交互模式中，服务器完全控制生成过程，这为想要将生成的文本保密的用户留下了零的选择。我们提出了LatticeGen，这是一个协作框架，其中服务器仍然处理大部分计算，而用户控制采样操作。其关键思想是，用户将真实生成的序列与噪声令牌混合，并将其隐藏在有噪声的网格中。考虑到来自假设恶意服务器的潜在攻击以及用户如何防御它，我们提出了重复波束搜索攻击和混合噪声方案。在我们的实验中，我们应用LatticeGen来保护提示和生成。实验结果表明，虽然加噪的格子降低了生成质量，但在强攻击下(BERTScore测试50%以上的语义仍然隐藏)，LatticeGen在很大程度上保护了真实的生成。



## **12. Streamlining Attack Tree Generation: A Fragment-Based Approach**

精简攻击树生成：一种基于片段的方法 cs.CR

To appear at the 57th Hawaii International Conference on Social  Systems (HICSS-57), Honolulu, Hawaii. 2024

**SubmitDate**: 2023-10-01    [abs](http://arxiv.org/abs/2310.00654v1) [paper-pdf](http://arxiv.org/pdf/2310.00654v1)

**Authors**: Irdin Pekaric, Markus Frick, Jubril Gbolahan Adigun, Raffaela Groner, Thomas Witte, Alexander Raschke, Michael Felderer, Matthias Tichy

**Abstract**: Attack graphs are a tool for analyzing security vulnerabilities that capture different and prospective attacks on a system. As a threat modeling tool, it shows possible paths that an attacker can exploit to achieve a particular goal. However, due to the large number of vulnerabilities that are published on a daily basis, they have the potential to rapidly expand in size. Consequently, this necessitates a significant amount of resources to generate attack graphs. In addition, generating composited attack models for complex systems such as self-adaptive or AI is very difficult due to their nature to continuously change. In this paper, we present a novel fragment-based attack graph generation approach that utilizes information from publicly available information security databases. Furthermore, we also propose a domain-specific language for attack modeling, which we employ in the proposed attack graph generation approach. Finally, we present a demonstrator example showcasing the attack generator's capability to replicate a verified attack chain, as previously confirmed by security experts.

摘要: 攻击图是一种分析安全漏洞的工具，可捕获对系统的不同攻击和潜在攻击。作为一种威胁建模工具，它显示了攻击者可以利用来实现特定目标的可能路径。然而，由于每天发布的大量漏洞，它们有可能迅速扩大规模。因此，这需要大量的资源来生成攻击图。此外，对于自适应或人工智能等复杂系统，由于其不断变化的性质，生成复合攻击模型是非常困难的。本文提出了一种新的基于片段的攻击图生成方法，该方法利用公共信息安全数据库中的信息生成攻击图。此外，我们还提出了一种特定于领域的攻击建模语言，并将其用于提出的攻击图生成方法。最后，我们给出了一个演示示例，展示了攻击生成器复制经过验证的攻击链的能力，这一点之前得到了安全专家的证实。



## **13. Evaluating the Instruction-Following Robustness of Large Language Models to Prompt Injection**

评估大语言模型的指令跟随健壮性以实现快速注入 cs.CL

The data and code can be found at  https://github.com/Leezekun/Adv-Instruct-Eval

**SubmitDate**: 2023-09-30    [abs](http://arxiv.org/abs/2308.10819v2) [paper-pdf](http://arxiv.org/pdf/2308.10819v2)

**Authors**: Zekun Li, Baolin Peng, Pengcheng He, Xifeng Yan

**Abstract**: Large Language Models (LLMs) have shown remarkable proficiency in following instructions, making them valuable in customer-facing applications. However, their impressive capabilities also raise concerns about the amplification of risks posed by adversarial instructions, which can be injected into the model input by third-party attackers to manipulate LLMs' original instructions and prompt unintended actions and content. Therefore, it is crucial to understand LLMs' ability to accurately discern which instructions to follow to ensure their safe deployment in real-world scenarios. In this paper, we propose a pioneering benchmark for automatically evaluating the robustness of instruction-following LLMs against adversarial instructions injected in the prompt. The objective of this benchmark is to quantify the extent to which LLMs are influenced by injected adversarial instructions and assess their ability to differentiate between these injected adversarial instructions and original user instructions. Through experiments conducted with state-of-the-art instruction-following LLMs, we uncover significant limitations in their robustness against adversarial instruction injection attacks. Furthermore, our findings indicate that prevalent instruction-tuned models are prone to being ``overfitted'' to follow any instruction phrase in the prompt without truly understanding which instructions should be followed. This highlights the need to address the challenge of training models to comprehend prompts instead of merely following instruction phrases and completing the text. The data and code can be found at \url{https://github.com/Leezekun/Adv-Instruct-Eval}.

摘要: 大型语言模型(LLM)在遵循说明方面表现出非凡的熟练程度，这使它们在面向客户的应用程序中具有价值。然而，它们令人印象深刻的能力也引发了人们对对抗性指令带来的风险放大的担忧，这些指令可以被注入第三方攻击者输入的模型中，以操纵LLMS的原始指令并提示意外的操作和内容。因此，了解LLMS准确识别应遵循哪些指令以确保在现实世界场景中安全部署的能力至关重要。在本文中，我们提出了一个开创性的基准，用于自动评估指令跟随LLMS对提示中注入的敌意指令的健壮性。这一基准的目的是量化LLM受注入的敌意指令的影响程度，并评估它们区分这些注入的对抗性指令和原始用户指令的能力。通过使用最先进的指令跟随LLM进行的实验，我们发现它们对敌意指令注入攻击的健壮性存在显著的局限性。此外，我们的研究结果表明，流行的指导性调整模型倾向于在没有真正理解哪些指令应该被遵循的情况下，“过度适应”地遵循提示中的任何指导语。这突出表明，需要解决培训模型理解提示的挑战，而不是仅仅遵循指导短语和完成正文。数据和代码可在\url{https://github.com/Leezekun/Adv-Instruct-Eval}.上找到



## **14. Red Teaming Game: A Game-Theoretic Framework for Red Teaming Language Models**

红色团队博弈：红色团队语言模型的博弈论框架 cs.CL

**SubmitDate**: 2023-09-30    [abs](http://arxiv.org/abs/2310.00322v1) [paper-pdf](http://arxiv.org/pdf/2310.00322v1)

**Authors**: Chengdong Ma, Ziran Yang, Minquan Gao, Hai Ci, Jun Gao, Xuehai Pan, Yaodong Yang

**Abstract**: Deployable Large Language Models (LLMs) must conform to the criterion of helpfulness and harmlessness, thereby achieving consistency between LLMs outputs and human values. Red-teaming techniques constitute a critical way towards this criterion. Existing work rely solely on manual red team designs and heuristic adversarial prompts for vulnerability detection and optimization. These approaches lack rigorous mathematical formulation, thus limiting the exploration of diverse attack strategy within quantifiable measure and optimization of LLMs under convergence guarantees. In this paper, we present Red-teaming Game (RTG), a general game-theoretic framework without manual annotation. RTG is designed for analyzing the multi-turn attack and defense interactions between Red-team language Models (RLMs) and Blue-team Language Model (BLM). Within the RTG, we propose Gamified Red-teaming Solver (GRTS) with diversity measure of the semantic space. GRTS is an automated red teaming technique to solve RTG towards Nash equilibrium through meta-game analysis, which corresponds to the theoretically guaranteed optimization direction of both RLMs and BLM. Empirical results in multi-turn attacks with RLMs show that GRTS autonomously discovered diverse attack strategies and effectively improved security of LLMs, outperforming existing heuristic red-team designs. Overall, RTG has established a foundational framework for red teaming tasks and constructed a new scalable oversight technique for alignment.

摘要: 可部署的大型语言模型(LLMS)必须符合有益和无害的标准，从而实现LLMS的输出与人的价值之间的一致性。红团队技术构成了实现这一标准的关键途径。现有的工作完全依赖于手动红色团队设计和启发式对抗性提示来进行漏洞检测和优化。这些方法缺乏严格的数学描述，从而限制了在可量化的度量范围内探索多样化的攻击策略，以及在收敛保证下对LLMS进行优化。在本文中，我们提出了一种不需要人工注释的通用博弈论框架--Red-Teaming Game(RTG)。RTG用于分析红队语言模型(RLMS)和蓝队语言模型(BLM)之间的多回合攻防交互。在RTG中，我们提出了一种具有语义空间多样性度量的Gamalized Red-Teaming Solver(GRTS)。GRTS是一种自动红队技术，通过元博弈分析解决RTG向纳什均衡的方向，这对应于理论上保证的RLMS和BLM的优化方向。在RLMS多回合攻击中的实验结果表明，GRTS自主发现多样化的攻击策略，有效地提高了LLMS的安全性，优于已有的启发式红队设计。总体而言，RTG为红色团队任务建立了一个基本框架，并构建了一种新的可扩展的协调监督技术。



## **15. FLIP: Cross-domain Face Anti-spoofing with Language Guidance**

翻转：跨域人脸反欺骗式语言引导 cs.CV

Accepted to ICCV-2023. Project Page:  https://koushiksrivats.github.io/FLIP/

**SubmitDate**: 2023-09-28    [abs](http://arxiv.org/abs/2309.16649v1) [paper-pdf](http://arxiv.org/pdf/2309.16649v1)

**Authors**: Koushik Srivatsan, Muzammal Naseer, Karthik Nandakumar

**Abstract**: Face anti-spoofing (FAS) or presentation attack detection is an essential component of face recognition systems deployed in security-critical applications. Existing FAS methods have poor generalizability to unseen spoof types, camera sensors, and environmental conditions. Recently, vision transformer (ViT) models have been shown to be effective for the FAS task due to their ability to capture long-range dependencies among image patches. However, adaptive modules or auxiliary loss functions are often required to adapt pre-trained ViT weights learned on large-scale datasets such as ImageNet. In this work, we first show that initializing ViTs with multimodal (e.g., CLIP) pre-trained weights improves generalizability for the FAS task, which is in line with the zero-shot transfer capabilities of vision-language pre-trained (VLP) models. We then propose a novel approach for robust cross-domain FAS by grounding visual representations with the help of natural language. Specifically, we show that aligning the image representation with an ensemble of class descriptions (based on natural language semantics) improves FAS generalizability in low-data regimes. Finally, we propose a multimodal contrastive learning strategy to boost feature generalization further and bridge the gap between source and target domains. Extensive experiments on three standard protocols demonstrate that our method significantly outperforms the state-of-the-art methods, achieving better zero-shot transfer performance than five-shot transfer of adaptive ViTs. Code: https://github.com/koushiksrivats/FLIP

摘要: 人脸反欺骗(FAS)或表示攻击检测是部署在安全关键应用中的人脸识别系统的重要组件。现有的FAS方法对未知的欺骗类型、摄像机传感器和环境条件的泛化能力较差。最近，视觉转换器(VIT)模型被证明是有效的，因为它们能够捕获图像斑块之间的远程依赖关系。然而，通常需要自适应模块或辅助损失函数来适应在诸如ImageNet的大规模数据集上学习的预先训练的VIT权重。在这项工作中，我们首先证明了用多模式(如CLIP)预训练权重初始化VITS提高了FAS任务的泛化能力，这与视觉语言预训练(VLP)模型的零镜头迁移能力是一致的。在此基础上，我们提出了一种新的基于自然语言的视觉表达方法，实现了跨域的强健性。具体地说，我们表明，将图像表示与类描述的集合(基于自然语言语义)对齐可以提高低数据条件下的FAS泛化能力。最后，我们提出了一种多通道对比学习策略，以进一步提高特征泛化能力，并弥合源域和目标域之间的差距。在三个标准协议上的大量实验表明，该方法的性能明显优于最新的方法，获得了比自适应VITS的五次传输更好的零次传输性能。代码：https://github.com/koushiksrivats/FLIP



## **16. VDC: Versatile Data Cleanser for Detecting Dirty Samples via Visual-Linguistic Inconsistency**

VDC：通过视觉-语言不一致检测污点样本的通用数据清洁器 cs.CV

22 pages,5 figures,17 tables

**SubmitDate**: 2023-09-28    [abs](http://arxiv.org/abs/2309.16211v1) [paper-pdf](http://arxiv.org/pdf/2309.16211v1)

**Authors**: Zihao Zhu, Mingda Zhang, Shaokui Wei, Bingzhe Wu, Baoyuan Wu

**Abstract**: The role of data in building AI systems has recently been emphasized by the emerging concept of data-centric AI. Unfortunately, in the real-world, datasets may contain dirty samples, such as poisoned samples from backdoor attack, noisy labels in crowdsourcing, and even hybrids of them. The presence of such dirty samples makes the DNNs vunerable and unreliable.Hence, it is critical to detect dirty samples to improve the quality and realiability of dataset. Existing detectors only focus on detecting poisoned samples or noisy labels, that are often prone to weak generalization when dealing with dirty samples from other domains.In this paper, we find a commonality of various dirty samples is visual-linguistic inconsistency between images and associated labels. To capture the semantic inconsistency between modalities, we propose versatile data cleanser (VDC) leveraging the surpassing capabilities of multimodal large language models (MLLM) in cross-modal alignment and reasoning.It consists of three consecutive modules: the visual question generation module to generate insightful questions about the image; the visual question answering module to acquire the semantics of the visual content by answering the questions with MLLM; followed by the visual answer evaluation module to evaluate the inconsistency.Extensive experiments demonstrate its superior performance and generalization to various categories and types of dirty samples.

摘要: 数据在构建人工智能系统中的作用最近被以数据为中心的人工智能的新兴概念所强调。不幸的是，在现实世界中，数据集可能包含肮脏的样本，例如来自后门攻击的有毒样本、众包中嘈杂的标签，甚至是它们的混合体。这些脏样本的存在使得DNN变得脆弱和不可靠，因此，检测脏样本对于提高数据集的质量和可靠性至关重要。现有的检测器只检测有毒样本或有噪声的标签，在处理其他领域的脏样本时往往容易产生较弱的泛化，本文发现各种脏样本的一个共同点是图像和关联标签之间的视觉语言不一致。为了捕捉通道间的语义不一致，利用多通道大语言模型(MLLM)在跨通道对齐和推理方面的优势，提出了通用数据清洗模块(VDC)，它由三个连续的模块组成：视觉问题生成模块，用于生成关于图像的有洞察力的问题；视觉问答模块，通过使用MLLM回答问题来获取视觉内容的语义；以及视觉答案评估模块，用于评估不一致。大量的实验表明，它具有优越的性能和对各种类别和类型的脏样本的泛化。



## **17. Advancing Beyond Identification: Multi-bit Watermark for Large Language Models**

超越识别：大型语言模型的多位水印 cs.CL

Under review. 9 pages and appendix

**SubmitDate**: 2023-09-27    [abs](http://arxiv.org/abs/2308.00221v2) [paper-pdf](http://arxiv.org/pdf/2308.00221v2)

**Authors**: KiYoon Yoo, Wonhyuk Ahn, Nojun Kwak

**Abstract**: We propose a method to tackle misuses of large language models beyond the identification of machine-generated text. While existing methods focus on detection, some malicious misuses demand tracing the adversary user for counteracting them. To address this, we propose Multi-bit Watermark via Position Allocation, embedding traceable multi-bit information during language model generation. Leveraging the benefits of zero-bit watermarking, our method enables robust extraction of the watermark without any model access, embedding and extraction of long messages ($\geq$ 32-bit) without finetuning, and maintaining text quality, while allowing zero-bit detection all at the same time. Moreover, our watermark is relatively robust under strong attacks like interleaving human texts and paraphrasing.

摘要: 我们提出了一种方法来解决机器生成文本识别之外的大型语言模型的误用。虽然现有的方法侧重于检测，但一些恶意滥用需要跟踪恶意用户来对抗它们。为了解决这个问题，我们提出了通过位置分配的多比特水印，在语言模型生成过程中嵌入可追踪的多比特信息。利用零位水印的优点，我们的方法可以在不访问任何模型的情况下稳健地提取水印，在不进行精细调整的情况下嵌入和提取长消息($32位)，并保持文本质量，同时允许零位检测。此外，我们的水印在交织文本和转译等强攻击下具有较强的稳健性。



## **18. Large Language Model Alignment: A Survey**

大型语言模型对齐：综述 cs.CL

76 pages

**SubmitDate**: 2023-09-26    [abs](http://arxiv.org/abs/2309.15025v1) [paper-pdf](http://arxiv.org/pdf/2309.15025v1)

**Authors**: Tianhao Shen, Renren Jin, Yufei Huang, Chuang Liu, Weilong Dong, Zishan Guo, Xinwei Wu, Yan Liu, Deyi Xiong

**Abstract**: Recent years have witnessed remarkable progress made in large language models (LLMs). Such advancements, while garnering significant attention, have concurrently elicited various concerns. The potential of these models is undeniably vast; however, they may yield texts that are imprecise, misleading, or even detrimental. Consequently, it becomes paramount to employ alignment techniques to ensure these models to exhibit behaviors consistent with human values.   This survey endeavors to furnish an extensive exploration of alignment methodologies designed for LLMs, in conjunction with the extant capability research in this domain. Adopting the lens of AI alignment, we categorize the prevailing methods and emergent proposals for the alignment of LLMs into outer and inner alignment. We also probe into salient issues including the models' interpretability, and potential vulnerabilities to adversarial attacks. To assess LLM alignment, we present a wide variety of benchmarks and evaluation methodologies. After discussing the state of alignment research for LLMs, we finally cast a vision toward the future, contemplating the promising avenues of research that lie ahead.   Our aspiration for this survey extends beyond merely spurring research interests in this realm. We also envision bridging the gap between the AI alignment research community and the researchers engrossed in the capability exploration of LLMs for both capable and safe LLMs.

摘要: 近年来，大型语言模型(LLM)取得了显著进展。这些进展在引起人们极大关注的同时，也引起了各种关注。不可否认，这些模型的潜力是巨大的；然而，它们可能会产生不精确、误导甚至有害的文本。因此，使用对齐技术来确保这些模型表现出与人类价值观一致的行为变得至关重要。本综述致力于结合该领域现有的能力研究，对为LLMS设计的比对方法进行广泛的探索。采用人工智能对准的视角，将目前流行的对准方法和建议分为外对准和内对准两大类。我们还探讨了突出的问题，包括模型的可解释性，以及对对抗性攻击的潜在脆弱性。为了评估LLM一致性，我们提出了各种基准和评估方法。在讨论了LLMS配准研究的现状后，我们最后展望了未来，展望了未来充满希望的研究途径。我们对这项调查的渴望不仅仅是刺激这一领域的研究兴趣。我们还设想弥合人工智能对齐研究社区和致力于LLM能力探索的研究人员之间的差距，以实现有能力和安全的LLM。



## **19. SurrogatePrompt: Bypassing the Safety Filter of Text-To-Image Models via Substitution**

代理提示：通过替换绕过文本到图像模型的安全过滤器 cs.CV

14 pages, 11 figures

**SubmitDate**: 2023-09-25    [abs](http://arxiv.org/abs/2309.14122v1) [paper-pdf](http://arxiv.org/pdf/2309.14122v1)

**Authors**: Zhongjie Ba, Jieming Zhong, Jiachen Lei, Peng Cheng, Qinglong Wang, Zhan Qin, Zhibo Wang, Kui Ren

**Abstract**: Advanced text-to-image models such as DALL-E 2 and Midjourney possess the capacity to generate highly realistic images, raising significant concerns regarding the potential proliferation of unsafe content. This includes adult, violent, or deceptive imagery of political figures. Despite claims of rigorous safety mechanisms implemented in these models to restrict the generation of not-safe-for-work (NSFW) content, we successfully devise and exhibit the first prompt attacks on Midjourney, resulting in the production of abundant photorealistic NSFW images. We reveal the fundamental principles of such prompt attacks and suggest strategically substituting high-risk sections within a suspect prompt to evade closed-source safety measures. Our novel framework, SurrogatePrompt, systematically generates attack prompts, utilizing large language models, image-to-text, and image-to-image modules to automate attack prompt creation at scale. Evaluation results disclose an 88% success rate in bypassing Midjourney's proprietary safety filter with our attack prompts, leading to the generation of counterfeit images depicting political figures in violent scenarios. Both subjective and objective assessments validate that the images generated from our attack prompts present considerable safety hazards.

摘要: 先进的文本到图像模型，如Dall-E2和MidTrik，具有生成高真实感图像的能力，这引发了人们对不安全内容潜在扩散的严重担忧。这包括成人的、暴力的或欺骗性的政治人物形象。尽管声称在这些模型中实施了严格的安全机制来限制不安全工作(NSFW)内容的生成，但我们成功地设计并展示了第一次在中途进行的即时攻击，从而产生了丰富的照片逼真的NSFW图像。我们揭示了这种快速攻击的基本原理，并建议有策略地在可疑提示中替换高风险部分，以规避封闭源代码的安全措施。我们的新框架Surogue atePrompt系统地生成攻击提示，利用大型语言模型、图像到文本和图像到图像模块来自动大规模创建攻击提示。评估结果显示，使用我们的攻击提示绕过MidRoad的专有安全过滤器的成功率为88%，导致生成描绘暴力场景中的政治人物的假冒图像。主观和客观评估都证实，我们的攻击提示生成的图像存在相当大的安全风险。



## **20. FedMLSecurity: A Benchmark for Attacks and Defenses in Federated Learning and LLMs**

FedMLSecurity：联合学习和LLMS中攻击和防御的基准 cs.CR

**SubmitDate**: 2023-09-25    [abs](http://arxiv.org/abs/2306.04959v2) [paper-pdf](http://arxiv.org/pdf/2306.04959v2)

**Authors**: Shanshan Han, Baturalp Buyukates, Zijian Hu, Han Jin, Weizhao Jin, Lichao Sun, Xiaoyang Wang, Wenxuan Wu, Chulin Xie, Yuhang Yao, Kai Zhang, Qifan Zhang, Yuhui Zhang, Salman Avestimehr, Chaoyang He

**Abstract**: This paper introduces FedMLSecurity, a benchmark designed to simulate adversarial attacks and corresponding defense mechanisms in Federated Learning (FL). As an integral module of the open-sourced library FedML that facilitates FL algorithm development and performance comparison, FedMLSecurity enhances FedML's capabilities to evaluate security issues and potential remedies in FL. FedMLSecurity comprises two major components: FedMLAttacker that simulates attacks injected during FL training, and FedMLDefender that simulates defensive mechanisms to mitigate the impacts of the attacks. FedMLSecurity is open-sourced and can be customized to a wide range of machine learning models (e.g., Logistic Regression, ResNet, GAN, etc.) and federated optimizers (e.g., FedAVG, FedOPT, FedNOVA, etc.). FedMLSecurity can also be applied to Large Language Models (LLMs) easily, demonstrating its adaptability and applicability in various scenarios.

摘要: 本文介绍了联邦学习中用于模拟对抗性攻击和相应防御机制的基准测试程序FedMLSecurity。作为促进FL算法开发和性能比较的开源库FedML的一个不可或缺的模块，FedMLSecurity增强了FedML评估FL中的安全问题和潜在补救措施的能力。FedMLSecurity由两个主要组件组成：模拟在FL训练期间注入的攻击的FedMLAttracker和模拟防御机制以减轻攻击影响的FedMLDefender。FedMLSecurity是开源的，可以针对多种机器学习模型(例如Logistic回归、ResNet、GAN等)进行定制。以及联合优化器(例如，FedAVG、FedOPT、FedNOVA等)。FedMLSecurity也可以很容易地应用到大型语言模型(LLM)中，展示了它在各种场景中的适应性和适用性。



## **21. Defending Pre-trained Language Models as Few-shot Learners against Backdoor Attacks**

为预先培训的语言模型辩护，称其为不太可能学习的人，免受后门攻击 cs.LG

Accepted by NeurIPS'23

**SubmitDate**: 2023-09-23    [abs](http://arxiv.org/abs/2309.13256v1) [paper-pdf](http://arxiv.org/pdf/2309.13256v1)

**Authors**: Zhaohan Xi, Tianyu Du, Changjiang Li, Ren Pang, Shouling Ji, Jinghui Chen, Fenglong Ma, Ting Wang

**Abstract**: Pre-trained language models (PLMs) have demonstrated remarkable performance as few-shot learners. However, their security risks under such settings are largely unexplored. In this work, we conduct a pilot study showing that PLMs as few-shot learners are highly vulnerable to backdoor attacks while existing defenses are inadequate due to the unique challenges of few-shot scenarios. To address such challenges, we advocate MDP, a novel lightweight, pluggable, and effective defense for PLMs as few-shot learners. Specifically, MDP leverages the gap between the masking-sensitivity of poisoned and clean samples: with reference to the limited few-shot data as distributional anchors, it compares the representations of given samples under varying masking and identifies poisoned samples as ones with significant variations. We show analytically that MDP creates an interesting dilemma for the attacker to choose between attack effectiveness and detection evasiveness. The empirical evaluation using benchmark datasets and representative attacks validates the efficacy of MDP.

摘要: 预先训练的语言模型(PLM)表现出作为少有学习者的显著表现。然而，在这种情况下，他们的安全风险在很大程度上是未被探索的。在这项工作中，我们进行了一项初步研究，表明作为少射学习者的PLM非常容易受到后门攻击，而现有的防御由于少射场景的独特挑战而不够充分。为了应对这样的挑战，我们提倡MDP，这是一种新型的轻量级、可插拔和有效的防御措施，适用于学习机会较少的PLM。具体地说，MDP利用了有毒样本和干净样本的掩蔽敏感度之间的差距：参考有限的少数激发数据作为分布锚，它比较不同掩蔽下给定样本的表示，并将有毒样本识别为具有显著变化的样本。我们分析表明，MDP造成了攻击者在攻击有效性和检测规避之间进行选择的有趣两难境地。使用基准数据集和典型攻击进行的经验评估验证了MDP的有效性。



## **22. Knowledge Sanitization of Large Language Models**

大型语言模型的知识清洗 cs.CL

**SubmitDate**: 2023-09-21    [abs](http://arxiv.org/abs/2309.11852v1) [paper-pdf](http://arxiv.org/pdf/2309.11852v1)

**Authors**: Yoichi Ishibashi, Hidetoshi Shimodaira

**Abstract**: We explore a knowledge sanitization approach to mitigate the privacy concerns associated with large language models (LLMs). LLMs trained on a large corpus of Web data can memorize and potentially reveal sensitive or confidential information, raising critical security concerns. Our technique fine-tunes these models, prompting them to generate harmless responses such as ``I don't know'' when queried about specific information. Experimental results in a closed-book question-answering task show that our straightforward method not only minimizes particular knowledge leakage but also preserves the overall performance of LLM. These two advantages strengthen the defense against extraction attacks and reduces the emission of harmful content such as hallucinations.

摘要: 我们探索了一种知识净化方法来缓解与大型语言模型(LLM)相关的隐私问题。在大型网络数据语料库上接受培训的LLM可能会记住并可能泄露敏感或机密信息，从而引发关键的安全问题。我们的技术对这些模型进行了微调，促使它们在被问及特定信息时产生无害的反应，如“我不知道”。在闭卷问答任务中的实验结果表明，该方法不仅最大限度地减少了特定的知识泄漏，而且保持了LLM的整体性能。这两个优势加强了对提取攻击的防御，并减少了幻觉等有害内容的排放。



## **23. A Chinese Prompt Attack Dataset for LLMs with Evil Content**

一个针对含有恶意内容的低层管理系统的中文提示攻击数据集 cs.CL

**SubmitDate**: 2023-09-21    [abs](http://arxiv.org/abs/2309.11830v1) [paper-pdf](http://arxiv.org/pdf/2309.11830v1)

**Authors**: Chengyuan Liu, Fubang Zhao, Lizhi Qing, Yangyang Kang, Changlong Sun, Kun Kuang, Fei Wu

**Abstract**: Large Language Models (LLMs) present significant priority in text understanding and generation. However, LLMs suffer from the risk of generating harmful contents especially while being employed to applications. There are several black-box attack methods, such as Prompt Attack, which can change the behaviour of LLMs and induce LLMs to generate unexpected answers with harmful contents. Researchers are interested in Prompt Attack and Defense with LLMs, while there is no publicly available dataset to evaluate the abilities of defending prompt attack. In this paper, we introduce a Chinese Prompt Attack Dataset for LLMs, called CPAD. Our prompts aim to induce LLMs to generate unexpected outputs with several carefully designed prompt attack approaches and widely concerned attacking contents. Different from previous datasets involving safety estimation, We construct the prompts considering three dimensions: contents, attacking methods and goals, thus the responses can be easily evaluated and analysed. We run several well-known Chinese LLMs on our dataset, and the results show that our prompts are significantly harmful to LLMs, with around 70% attack success rate. We will release CPAD to encourage further studies on prompt attack and defense.

摘要: 大语言模型(LLM)在文本理解和生成中具有重要的优先地位。然而，LLMS面临着产生有害内容的风险，特别是在应用程序中使用时。有几种黑盒攻击方法，如提示攻击，可以改变LLMS的行为，并诱导LLMS生成包含有害内容的意外答案。研究人员对LLMS的快速攻防感兴趣，但目前还没有公开的数据集来评估其防御快速攻击的能力。本文介绍了一个针对LLMS的中文即时攻击数据集CPAD。我们的提示旨在通过精心设计的几种即时攻击方法和广泛关注的攻击内容来诱导LLMS产生意想不到的输出。与以往涉及安全评估的数据集不同，我们从内容、攻击方法和目标三个维度构建提示，从而便于对响应进行评估和分析。我们在我们的数据集上运行了几个著名的中文LLMS，结果表明我们的提示对LLMS有显著的危害，攻击成功率约为70%。我们将发布CPAD，以鼓励进一步研究快速攻防。



## **24. How Robust is Google's Bard to Adversarial Image Attacks?**

谷歌的吟游诗人对敌意图像攻击的健壮程度如何？ cs.CV

Technical report

**SubmitDate**: 2023-09-21    [abs](http://arxiv.org/abs/2309.11751v1) [paper-pdf](http://arxiv.org/pdf/2309.11751v1)

**Authors**: Yinpeng Dong, Huanran Chen, Jiawei Chen, Zhengwei Fang, Xiao Yang, Yichi Zhang, Yu Tian, Hang Su, Jun Zhu

**Abstract**: Multimodal Large Language Models (MLLMs) that integrate text and other modalities (especially vision) have achieved unprecedented performance in various multimodal tasks. However, due to the unsolved adversarial robustness problem of vision models, MLLMs can have more severe safety and security risks by introducing the vision inputs. In this work, we study the adversarial robustness of Google's Bard, a competitive chatbot to ChatGPT that released its multimodal capability recently, to better understand the vulnerabilities of commercial MLLMs. By attacking white-box surrogate vision encoders or MLLMs, the generated adversarial examples can mislead Bard to output wrong image descriptions with a 22% success rate based solely on the transferability. We show that the adversarial examples can also attack other MLLMs, e.g., a 26% attack success rate against Bing Chat and a 86% attack success rate against ERNIE bot. Moreover, we identify two defense mechanisms of Bard, including face detection and toxicity detection of images. We design corresponding attacks to evade these defenses, demonstrating that the current defenses of Bard are also vulnerable. We hope this work can deepen our understanding on the robustness of MLLMs and facilitate future research on defenses. Our code is available at https://github.com/thu-ml/Attack-Bard.

摘要: 多通道大语言模型将文本和其他通道(尤其是视觉)结合在一起，在各种多通道任务中取得了前所未有的性能。然而，由于视觉模型的对抗性健壮性问题尚未解决，通过引入视觉输入，MLLMS可能存在更严重的安全风险。在这项工作中，我们研究了Google的Bard，一个与ChatGPT竞争的聊天机器人，最近发布了它的多模式功能，以更好地了解商业MLLMS的漏洞。通过攻击白盒代理视觉编码器或MLLM，生成的敌意示例可以误导BARD输出错误的图像描述，仅基于可转移性的成功率为22%。我们表明，恶意例子也可以攻击其他MLLMS，例如，对Bing Chat的攻击成功率为26%，对Ernie bot的攻击成功率为86%。此外，我们还识别了BARD的两种防御机制，包括人脸检测和图像毒性检测。我们设计了相应的攻击来逃避这些防御，证明了巴德目前的防御也是脆弱的。我们希望这项工作可以加深我们对MLLMS稳健性的理解，并为未来的防御研究提供便利。我们的代码可以在https://github.com/thu-ml/Attack-Bard.上找到



## **25. Model Leeching: An Extraction Attack Targeting LLMs**

模型LEACK：一种针对LLMS的提取攻击 cs.LG

**SubmitDate**: 2023-09-19    [abs](http://arxiv.org/abs/2309.10544v1) [paper-pdf](http://arxiv.org/pdf/2309.10544v1)

**Authors**: Lewis Birch, William Hackett, Stefan Trawicki, Neeraj Suri, Peter Garraghan

**Abstract**: Model Leeching is a novel extraction attack targeting Large Language Models (LLMs), capable of distilling task-specific knowledge from a target LLM into a reduced parameter model. We demonstrate the effectiveness of our attack by extracting task capability from ChatGPT-3.5-Turbo, achieving 73% Exact Match (EM) similarity, and SQuAD EM and F1 accuracy scores of 75% and 87%, respectively for only $50 in API cost. We further demonstrate the feasibility of adversarial attack transferability from an extracted model extracted via Model Leeching to perform ML attack staging against a target LLM, resulting in an 11% increase to attack success rate when applied to ChatGPT-3.5-Turbo.

摘要: 模型提取攻击是一种针对大型语言模型的新型提取攻击，能够将目标语言模型中特定于任务的知识提取到一个简化的参数模型中。我们通过从ChatGPT-3.5-Turbo中提取任务能力来证明我们的攻击的有效性，获得了73%的精确匹配(EM)相似度，以及小队EM和F1的准确率分别为75%和87%，仅需50美元的API成本。进一步论证了利用Model leeching提取的模型对目标LLM执行ML攻击阶段性攻击的可行性，将其应用于ChatGPT-3.5-Turbo，攻击成功率提高了11%。



## **26. Language Guided Adversarial Purification**

语言引导的对抗性净化 cs.LG

**SubmitDate**: 2023-09-19    [abs](http://arxiv.org/abs/2309.10348v1) [paper-pdf](http://arxiv.org/pdf/2309.10348v1)

**Authors**: Himanshu Singh, A V Subramanyam

**Abstract**: Adversarial purification using generative models demonstrates strong adversarial defense performance. These methods are classifier and attack-agnostic, making them versatile but often computationally intensive. Recent strides in diffusion and score networks have improved image generation and, by extension, adversarial purification. Another highly efficient class of adversarial defense methods known as adversarial training requires specific knowledge of attack vectors, forcing them to be trained extensively on adversarial examples. To overcome these limitations, we introduce a new framework, namely Language Guided Adversarial Purification (LGAP), utilizing pre-trained diffusion models and caption generators to defend against adversarial attacks. Given an input image, our method first generates a caption, which is then used to guide the adversarial purification process through a diffusion network. Our approach has been evaluated against strong adversarial attacks, proving its effectiveness in enhancing adversarial robustness. Our results indicate that LGAP outperforms most existing adversarial defense techniques without requiring specialized network training. This underscores the generalizability of models trained on large datasets, highlighting a promising direction for further research.

摘要: 基于产生式模型的对抗性净化算法表现出较强的对抗性防御性能。这些方法是分类器和攻击不可知的，使它们多才多艺，但往往计算密集。最近在传播和得分网络方面的进展改善了图像生成，进而改善了对手的净化。另一种被称为对抗性训练的高效对抗性防御方法需要对攻击载体的特定知识，迫使他们接受关于对抗性例子的广泛培训。为了克服这些局限性，我们引入了一种新的框架，即语言制导的对抗性净化(LGAP)，利用预先训练的扩散模型和字幕生成器来防御对抗性攻击。在给定输入图像的情况下，我们的方法首先生成字幕，然后使用该字幕通过扩散网络来指导敌方净化过程。我们的方法已经针对强大的对手攻击进行了评估，证明了它在增强对手稳健性方面的有效性。我们的结果表明，LGAP在不需要专门的网络训练的情况下，性能优于大多数现有的对抗性防御技术。这突显了在大数据集上训练的模型的泛化能力，突出了进一步研究的一个有希望的方向。



## **27. LLM Platform Security: Applying a Systematic Evaluation Framework to OpenAI's ChatGPT Plugins**

LLM平台安全：将系统评估框架应用于OpenAI的ChatGPT插件 cs.CR

**SubmitDate**: 2023-09-19    [abs](http://arxiv.org/abs/2309.10254v1) [paper-pdf](http://arxiv.org/pdf/2309.10254v1)

**Authors**: Umar Iqbal, Tadayoshi Kohno, Franziska Roesner

**Abstract**: Large language model (LLM) platforms, such as ChatGPT, have recently begun offering a plugin ecosystem to interface with third-party services on the internet. While these plugins extend the capabilities of LLM platforms, they are developed by arbitrary third parties and thus cannot be implicitly trusted. Plugins also interface with LLM platforms and users using natural language, which can have imprecise interpretations. In this paper, we propose a framework that lays a foundation for LLM platform designers to analyze and improve the security, privacy, and safety of current and future plugin-integrated LLM platforms. Our framework is a formulation of an attack taxonomy that is developed by iteratively exploring how LLM platform stakeholders could leverage their capabilities and responsibilities to mount attacks against each other. As part of our iterative process, we apply our framework in the context of OpenAI's plugin ecosystem. We uncover plugins that concretely demonstrate the potential for the types of issues that we outline in our attack taxonomy. We conclude by discussing novel challenges and by providing recommendations to improve the security, privacy, and safety of present and future LLM-based computing platforms.

摘要: 大型语言模型(LLM)平台，如ChatGPT，最近开始提供插件生态系统，以与互联网上的第三方服务对接。虽然这些插件扩展了LLM平台的功能，但它们是由任意的第三方开发的，因此不能被隐式信任。插件还使用自然语言与LLM平台和用户交互，这可能会有不准确的解释。在本文中，我们提出了一个框架，为LLM平台设计者分析和改进现有和未来插件集成的LLM平台的安全性、保密性和安全性奠定了基础。我们的框架是攻击分类的公式，通过迭代探索LLM平台利益相关者如何利用他们的能力和责任来对彼此发动攻击而开发的攻击分类。作为迭代过程的一部分，我们在OpenAI的插件生态系统中应用了我们的框架。我们发现了一些插件，这些插件具体展示了我们在攻击分类中概述的问题类型的可能性。最后，我们讨论了新的挑战，并提供了改进当前和未来基于LLM的计算平台的安全性、保密性和安全性的建议。



## **28. Defending Against Alignment-Breaking Attacks via Robustly Aligned LLM**

通过强健对齐的LLM防御对齐破坏攻击 cs.CL

16 Pages, 5 Figures, 3 Tables

**SubmitDate**: 2023-09-18    [abs](http://arxiv.org/abs/2309.14348v1) [paper-pdf](http://arxiv.org/pdf/2309.14348v1)

**Authors**: Bochuan Cao, Yuanpu Cao, Lu Lin, Jinghui Chen

**Abstract**: Recently, Large Language Models (LLMs) have made significant advancements and are now widely used across various domains. Unfortunately, there has been a rising concern that LLMs can be misused to generate harmful or malicious content. Though a line of research has focused on aligning LLMs with human values and preventing them from producing inappropriate content, such alignments are usually vulnerable and can be bypassed by alignment-breaking attacks via adversarially optimized or handcrafted jailbreaking prompts. In this work, we introduce a Robustly Aligned LLM (RA-LLM) to defend against potential alignment-breaking attacks. RA-LLM can be directly constructed upon an existing aligned LLM with a robust alignment checking function, without requiring any expensive retraining or fine-tuning process of the original LLM. Furthermore, we also provide a theoretical analysis for RA-LLM to verify its effectiveness in defending against alignment-breaking attacks. Through real-world experiments on open-source large language models, we demonstrate that RA-LLM can successfully defend against both state-of-the-art adversarial prompts and popular handcrafted jailbreaking prompts by reducing their attack success rates from nearly 100\% to around 10\% or less.

摘要: 近年来，大型语言模型(LLM)取得了长足的进步，现已广泛应用于各个领域。不幸的是，人们越来越担心LLMS可能被滥用来生成有害或恶意的内容。尽管有一系列研究专注于将LLM与人类价值观保持一致，并防止它们产生不适当的内容，但这种调整通常是脆弱的，可以通过恶意优化或手工制作的越狱提示被破坏顺序的攻击绕过。在这项工作中，我们引入了一种鲁棒对齐LLM(RA-LLM)来防御潜在的对齐破坏攻击。RA-LLM可以直接构建在现有的对准LLM上，具有健壮的对准检查功能，而不需要对原始LLM进行任何昂贵的再培训或微调过程。此外，我们还对RA-LLM进行了理论分析，以验证其在抵抗对齐破坏攻击方面的有效性。通过在开源大型语言模型上的真实世界实验，我们证明了RA-LLM能够成功地防御最新的敌意提示和流行的手工越狱提示，将攻击成功率从近100%降低到10%左右或更低。



## **29. Your Room is not Private: Gradient Inversion Attack on Reinforcement Learning**

你的房间不是私人的：强化学习中的梯度反转攻击 cs.RO

7 pages, 4 figures, 2 tables

**SubmitDate**: 2023-09-17    [abs](http://arxiv.org/abs/2306.09273v2) [paper-pdf](http://arxiv.org/pdf/2306.09273v2)

**Authors**: Miao Li, Wenhao Ding, Ding Zhao

**Abstract**: The prominence of embodied Artificial Intelligence (AI), which empowers robots to navigate, perceive, and engage within virtual environments, has attracted significant attention, owing to the remarkable advancements in computer vision and large language models. Privacy emerges as a pivotal concern within the realm of embodied AI, as the robot accesses substantial personal information. However, the issue of privacy leakage in embodied AI tasks, particularly in relation to reinforcement learning algorithms, has not received adequate consideration in research. This paper aims to address this gap by proposing an attack on the value-based algorithm and the gradient-based algorithm, utilizing gradient inversion to reconstruct states, actions, and supervision signals. The choice of using gradients for the attack is motivated by the fact that commonly employed federated learning techniques solely utilize gradients computed based on private user data to optimize models, without storing or transmitting the data to public servers. Nevertheless, these gradients contain sufficient information to potentially expose private data. To validate our approach, we conduct experiments on the AI2THOR simulator and evaluate our algorithm on active perception, a prevalent task in embodied AI. The experimental results demonstrate the effectiveness of our method in successfully reconstructing all information from the data across 120 room layouts.

摘要: 由于计算机视觉和大型语言模型的显著进步，使机器人能够在虚拟环境中导航、感知和参与的嵌入式人工智能(AI)的突出地位引起了人们的极大关注。随着机器人访问大量的个人信息，隐私成为体现人工智能领域的一个关键问题。然而，体验式人工智能任务中的隐私泄露问题，特别是与强化学习算法相关的隐私泄露问题，在研究中并没有得到足够的考虑。为了解决这一问题，本文对基于值的算法和基于梯度的算法提出了一种攻击，利用梯度求逆来重建状态、动作和监控信号。选择使用梯度进行攻击的动机是，通常使用的联合学习技术仅使用基于私有用户数据计算的梯度来优化模型，而不将数据存储或传输到公共服务器。然而，这些渐变包含了足够的信息来潜在地暴露私有数据。为了验证我们的方法，我们在AI2THOR模拟器上进行了实验，并对我们的算法进行了评估，主动感知是体现人工智能中的一个普遍任务。实验结果表明，我们的方法能够成功地从120个房间布局的数据中重建所有信息。



## **30. Open Sesame! Universal Black Box Jailbreaking of Large Language Models**

芝麻开门！大型语言模型的通用黑盒越狱 cs.CL

**SubmitDate**: 2023-09-17    [abs](http://arxiv.org/abs/2309.01446v2) [paper-pdf](http://arxiv.org/pdf/2309.01446v2)

**Authors**: Raz Lapid, Ron Langberg, Moshe Sipper

**Abstract**: Large language models (LLMs), designed to provide helpful and safe responses, often rely on alignment techniques to align with user intent and social guidelines. Unfortunately, this alignment can be exploited by malicious actors seeking to manipulate an LLM's outputs for unintended purposes. In this paper we introduce a novel approach that employs a genetic algorithm (GA) to manipulate LLMs when model architecture and parameters are inaccessible. The GA attack works by optimizing a universal adversarial prompt that -- when combined with a user's query -- disrupts the attacked model's alignment, resulting in unintended and potentially harmful outputs. Our novel approach systematically reveals a model's limitations and vulnerabilities by uncovering instances where its responses deviate from expected behavior. Through extensive experiments we demonstrate the efficacy of our technique, thus contributing to the ongoing discussion on responsible AI development by providing a diagnostic tool for evaluating and enhancing alignment of LLMs with human intent. To our knowledge this is the first automated universal black box jailbreak attack.

摘要: 大型语言模型(LLM)旨在提供有用和安全的响应，它们通常依靠对齐技术来与用户意图和社交指南保持一致。遗憾的是，恶意行为者可能会利用这种对齐，试图出于非预期目的操纵LLM的输出。在本文中，我们介绍了一种新的方法，即在模型结构和参数不可访问的情况下，使用遗传算法(GA)来操作LLM。GA攻击的工作原理是优化一个通用的对抗性提示，当与用户的查询结合在一起时，会扰乱被攻击模型的对齐，导致意外的和潜在的有害输出。我们的新方法通过揭示模型响应偏离预期行为的实例，系统地揭示了模型的局限性和漏洞。通过广泛的实验，我们展示了我们技术的有效性，从而通过提供一种诊断工具来评估和增强LLM与人类意图的一致性，从而为正在进行的关于负责任的人工智能开发的讨论做出贡献。据我们所知，这是第一次自动通用黑匣子越狱攻击。



## **31. Context-aware Adversarial Attack on Named Entity Recognition**

针对命名实体识别的上下文感知敌意攻击 cs.CL

**SubmitDate**: 2023-09-16    [abs](http://arxiv.org/abs/2309.08999v1) [paper-pdf](http://arxiv.org/pdf/2309.08999v1)

**Authors**: Shuguang Chen, Leonardo Neves, Thamar Solorio

**Abstract**: In recent years, large pre-trained language models (PLMs) have achieved remarkable performance on many natural language processing benchmarks. Despite their success, prior studies have shown that PLMs are vulnerable to attacks from adversarial examples. In this work, we focus on the named entity recognition task and study context-aware adversarial attack methods to examine the model's robustness. Specifically, we propose perturbing the most informative words for recognizing entities to create adversarial examples and investigate different candidate replacement methods to generate natural and plausible adversarial examples. Experiments and analyses show that our methods are more effective in deceiving the model into making wrong predictions than strong baselines.

摘要: 近年来，大型预训练语言模型(PLM)在许多自然语言处理基准上取得了显著的性能。尽管它们取得了成功，但先前的研究表明，PLM很容易受到对手例子的攻击。在这项工作中，我们以命名实体识别任务为重点，研究上下文感知的对抗性攻击方法，以检验模型的健壮性。具体地说，我们提出通过扰动用于识别实体的信息量最大的词来创建对抗性实例，并研究不同的候选替换方法来生成自然的和可信的对抗性实例。实验和分析表明，我们的方法比强基线更能有效地欺骗模型做出错误的预测。



## **32. ICLEF: In-Context Learning with Expert Feedback for Explainable Style Transfer**

ICLEF：带专家反馈的情景学习，用于可解释的风格转换 cs.CL

**SubmitDate**: 2023-09-15    [abs](http://arxiv.org/abs/2309.08583v1) [paper-pdf](http://arxiv.org/pdf/2309.08583v1)

**Authors**: Arkadiy Saakyan, Smaranda Muresan

**Abstract**: While state-of-the-art language models excel at the style transfer task, current work does not address explainability of style transfer systems. Explanations could be generated using large language models such as GPT-3.5 and GPT-4, but the use of such complex systems is inefficient when smaller, widely distributed, and transparent alternatives are available. We propose a framework to augment and improve a formality style transfer dataset with explanations via model distillation from ChatGPT. To further refine the generated explanations, we propose a novel way to incorporate scarce expert human feedback using in-context learning (ICLEF: In-Context Learning from Expert Feedback) by prompting ChatGPT to act as a critic to its own outputs. We use the resulting dataset of 9,960 explainable formality style transfer instances (e-GYAFC) to show that current openly distributed instruction-tuned models (and, in some settings, ChatGPT) perform poorly on the task, and that fine-tuning on our high-quality dataset leads to significant improvements as shown by automatic evaluation. In human evaluation, we show that models much smaller than ChatGPT fine-tuned on our data align better with expert preferences. Finally, we discuss two potential applications of models fine-tuned on the explainable style transfer task: interpretable authorship verification and interpretable adversarial attacks on AI-generated text detectors.

摘要: 虽然最先进的语言模型擅长于风格转换任务，但目前的工作并没有解决风格转换系统的可解释性问题。可以使用大型语言模型(如GPT-3.5和GPT-4)生成解释，但当有较小、分布广泛和透明的替代方案时，使用这种复杂系统的效率很低。通过对ChatGPT的模型提炼，我们提出了一个框架来扩充和改进带有解释的形式化风格的传输数据集。为了进一步完善生成的解释，我们提出了一种新的方法，通过促使ChatGPT作为对自己输出的批评者，使用上下文中学习(ICLEF：In-Context Learning from Expert Feedback)来整合稀缺的专家人类反馈。我们使用9960个可解释形式风格转移实例(e-GYAFC)的结果数据集来表明，当前开放分布的指令优化模型(在某些设置中，ChatGPT)在任务中表现不佳，并且如自动评估所示，对我们的高质量数据集进行微调会导致显著的改进。在人类评估中，我们表明，根据我们的数据微调的模型比ChatGPT小得多，更符合专家的偏好。最后，我们讨论了对可解释风格迁移任务进行微调的模型的两个潜在应用：可解释作者身份验证和对人工智能生成的文本检测器的可解释敌意攻击。



## **33. Adversarial Attacks on Tables with Entity Swap**

对具有实体交换的表的对抗性攻击 cs.CL

Accepted at TaDA workshop at VLDB 2023

**SubmitDate**: 2023-09-15    [abs](http://arxiv.org/abs/2309.08650v1) [paper-pdf](http://arxiv.org/pdf/2309.08650v1)

**Authors**: Aneta Koleva, Martin Ringsquandl, Volker Tresp

**Abstract**: The capabilities of large language models (LLMs) have been successfully applied in the context of table representation learning. The recently proposed tabular language models have reported state-of-the-art results across various tasks for table interpretation. However, a closer look into the datasets commonly used for evaluation reveals an entity leakage from the train set into the test set. Motivated by this observation, we explore adversarial attacks that represent a more realistic inference setup. Adversarial attacks on text have been shown to greatly affect the performance of LLMs, but currently, there are no attacks targeting tabular language models. In this paper, we propose an evasive entity-swap attack for the column type annotation (CTA) task. Our CTA attack is the first black-box attack on tables, where we employ a similarity-based sampling strategy to generate adversarial examples. The experimental results show that the proposed attack generates up to a 70% drop in performance.

摘要: 大型语言模型(LLM)的能力已经成功地应用于表表示学习的上下文中。最近提出的表格语言模型报告了各种表格解释任务的最新结果。然而，仔细观察通常用于评估的数据集，就会发现实体从训练集中泄漏到测试集中。在这一观察的激励下，我们探索了代表更现实的推理设置的对抗性攻击。针对文本的对抗性攻击已被证明会极大地影响LLMS的性能，但目前还没有针对表格语言模型的攻击。本文提出了一种针对列类型标注(CTA)任务的逃避实体交换攻击。我们的CTA攻击是第一个对表的黑盒攻击，其中我们使用基于相似性的采样策略来生成对抗性示例。实验结果表明，提出的攻击最高可导致70%的性能下降。



## **34. RAIN: Your Language Models Can Align Themselves without Finetuning**

Rain：您的语言模型无需微调即可自动调整 cs.CL

**SubmitDate**: 2023-09-13    [abs](http://arxiv.org/abs/2309.07124v1) [paper-pdf](http://arxiv.org/pdf/2309.07124v1)

**Authors**: Yuhui Li, Fangyun Wei, Jinjing Zhao, Chao Zhang, Hongyang Zhang

**Abstract**: Large language models (LLMs) often demonstrate inconsistencies with human preferences. Previous research gathered human preference data and then aligned the pre-trained models using reinforcement learning or instruction tuning, the so-called finetuning step. In contrast, aligning frozen LLMs without any extra data is more appealing. This work explores the potential of the latter setting. We discover that by integrating self-evaluation and rewind mechanisms, unaligned LLMs can directly produce responses consistent with human preferences via self-boosting. We introduce a novel inference method, Rewindable Auto-regressive INference (RAIN), that allows pre-trained LLMs to evaluate their own generation and use the evaluation results to guide backward rewind and forward generation for AI safety. Notably, RAIN operates without the need of extra data for model alignment and abstains from any training, gradient computation, or parameter updates; during the self-evaluation phase, the model receives guidance on which human preference to align with through a fixed-template prompt, eliminating the need to modify the initial prompt. Experimental results evaluated by GPT-4 and humans demonstrate the effectiveness of RAIN: on the HH dataset, RAIN improves the harmlessness rate of LLaMA 30B over vanilla inference from 82% to 97%, while maintaining the helpfulness rate. Under the leading adversarial attack llm-attacks on Vicuna 33B, RAIN establishes a new defense baseline by reducing the attack success rate from 94% to 19%.

摘要: 大型语言模型(LLM)经常显示出与人类偏好的不一致。之前的研究收集了人类的偏好数据，然后使用强化学习或教学调整，即所谓的微调步骤，对齐了预先训练的模型。相比之下，在没有任何额外数据的情况下对齐冻结的LLM更具吸引力。这项工作探索了后一种环境的潜力。我们发现，通过集成自我评估和回溯机制，未对齐的LLM可以通过自我增强直接产生与人类偏好一致的反应。我们引入了一种新的推理方法--可倒带自回归推理(RAIN)，它允许预先训练的LLM对自己的生成进行评估，并使用评估结果来指导人工智能安全的回溯和正演生成。值得注意的是，RAIN的运行不需要额外的数据来进行模型比对，并且不需要任何训练、梯度计算或参数更新；在自我评估阶段，模型通过固定模板提示接收关于与哪个人类偏好匹配的指导，从而消除了修改初始提示的需要。GPT-4和人类评估的实验结果证明了RAIN的有效性：在HH数据集上，RAIN在保持有益率的同时，将大羊驼30B的无害率从82%提高到97%。在领先的对抗性攻击1LM-对维库纳33B的攻击下，RAIN通过将攻击成功率从94%降低到19%，建立了新的防御基线。



## **35. Games and Argumentation: Time for a Family Reunion!**

游戏和辩论：家庭团聚的时间到了！ cs.LO

Fourth Workshop on Explainable Logic-Based Knowledge Representation  (XLoKR), Sept 2, 2023. Rhodes, Greece

**SubmitDate**: 2023-09-12    [abs](http://arxiv.org/abs/2309.06620v1) [paper-pdf](http://arxiv.org/pdf/2309.06620v1)

**Authors**: Bertram Ludäscher, Yilin Xia

**Abstract**: The rule "defeated(X) $\leftarrow$ attacks(Y,X), $\neg$ defeated(Y)" states that an argument is defeated if it is attacked by an argument that is not defeated. The rule "win(X) $\leftarrow$ move(X,Y), $\neg$ win(Y)" states that in a game a position is won if there is a move to a position that is not won. Both logic rules can be seen as close relatives (even identical twins) and both rules have been at the center of attention at various times in different communities: The first rule lies at the core of argumentation frameworks and has spawned a large family of models and semantics of abstract argumentation. The second rule has played a key role in the quest to find the "right" semantics for logic programs with recursion through negation, and has given rise to the stable and well-founded semantics. Both semantics have been widely studied by the logic programming and nonmonotonic reasoning community. The second rule has also received much attention by the database and finite model theory community, e.g., when studying the expressive power of query languages and fixpoint logics. Although close connections between argumentation frameworks, logic programming, and dialogue games have been known for a long time, the overlap and cross-fertilization between the communities appears to be smaller than one might expect. To this end, we recall some of the key results from database theory in which the win-move query has played a central role, e.g., on normal forms and expressive power of query languages. We introduce some notions that naturally emerge from games and that may provide new perspectives and research opportunities for argumentation frameworks. We discuss how solved query evaluation games reveal how- and why-not provenance of query answers. These techniques can be used to explain how results were derived via the given query, game, or argumentation framework.

摘要: “被击败的(X)$\leftarrow$攻击(Y，X)，$\neg$被击败(Y)”规则规定，如果一个论点被一个没有被击败的论点攻击，它就被击败。规则“Win(X)$\leftarrow$Move(X，Y)，$\neg$Win(Y)”规定，在游戏中，如果有人移动到一个没有赢的位置，那么这个位置就是赢的。这两个逻辑规则都可以被视为近亲(甚至是同卵双胞胎)，并且这两个规则在不同的社区中一直是人们关注的中心：第一个规则位于论证框架的核心，并催生了一大族抽象论证的模型和语义。第二个规则在通过否定为递归逻辑程序寻找“正确”语义的过程中起到了关键作用，并产生了稳定的、有充分基础的语义。这两种语义都得到了逻辑编程和非单调推理界的广泛研究。第二个规则也受到了数据库和有限模型理论界的极大关注，例如在研究查询语言和定点逻辑的表达能力时。尽管论证框架、逻辑编程和对话游戏之间的密切联系早已为人所知，但社区之间的重叠和交叉影响似乎比人们预期的要小。为此，我们回顾了数据库理论中的一些关键结果，其中Win-Move查询发挥了核心作用，例如，在查询语言的范式和表达能力方面。我们引入了一些从游戏中自然产生的概念，这些概念可能会为论证框架提供新的视角和研究机会。我们讨论了已解决的查询求值游戏如何揭示查询答案的来源。这些技术可以用来解释结果是如何通过给定的查询、游戏或论证框架得出的。



## **36. FuzzLLM: A Novel and Universal Fuzzing Framework for Proactively Discovering Jailbreak Vulnerabilities in Large Language Models**

FuzzLLM：一种新型通用的主动发现大型语言模型越狱漏洞的模糊框架 cs.CR

In submission, a preprint version

**SubmitDate**: 2023-09-11    [abs](http://arxiv.org/abs/2309.05274v1) [paper-pdf](http://arxiv.org/pdf/2309.05274v1)

**Authors**: Dongyu Yao, Jianshu Zhang, Ian G. Harris, Marcel Carlsson

**Abstract**: Jailbreak vulnerabilities in Large Language Models (LLMs), which exploit meticulously crafted prompts to elicit content that violates service guidelines, have captured the attention of research communities. While model owners can defend against individual jailbreak prompts through safety training strategies, this relatively passive approach struggles to handle the broader category of similar jailbreaks. To tackle this issue, we introduce FuzzLLM, an automated fuzzing framework designed to proactively test and discover jailbreak vulnerabilities in LLMs. We utilize templates to capture the structural integrity of a prompt and isolate key features of a jailbreak class as constraints. By integrating different base classes into powerful combo attacks and varying the elements of constraints and prohibited questions, FuzzLLM enables efficient testing with reduced manual effort. Extensive experiments demonstrate FuzzLLM's effectiveness and comprehensiveness in vulnerability discovery across various LLMs.

摘要: 大型语言模型(LLM)中的越狱漏洞利用精心设计的提示来引出违反服务指南的内容，已经引起了研究界的注意。虽然模型所有者可以通过安全培训策略来防御个别越狱提示，但这种相对被动的方法难以应对更广泛类别的类似越狱。为了解决这个问题，我们引入了FuzzLLM，这是一个自动模糊框架，旨在主动测试和发现LLM中的越狱漏洞。我们利用模板来捕获提示符的结构完整性，并将越狱类的关键特性隔离为约束。通过将不同的基类集成到强大的组合攻击中，并改变约束和禁止问题的元素，FuzzLLM能够以更少的手动工作实现高效的测试。广泛的实验证明了FuzzLLM在跨各种LLM发现漏洞方面的有效性和全面性。



## **37. RatGPT: Turning online LLMs into Proxies for Malware Attacks**

RatGPT：将在线LLM转变为恶意软件攻击的代理 cs.CR

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2308.09183v2) [paper-pdf](http://arxiv.org/pdf/2308.09183v2)

**Authors**: Mika Beckerich, Laura Plein, Sergio Coronado

**Abstract**: The evolution of Generative AI and the capabilities of the newly released Large Language Models (LLMs) open new opportunities in software engineering. However, they also lead to new challenges in cybersecurity. Recently, researchers have shown the possibilities of using LLMs such as ChatGPT to generate malicious content that can directly be exploited or guide inexperienced hackers to weaponize tools and code. These studies covered scenarios that still require the attacker to be in the middle of the loop. In this study, we leverage openly available plugins and use an LLM as proxy between the attacker and the victim. We deliver a proof-of-concept where ChatGPT is used for the dissemination of malicious software while evading detection, alongside establishing the communication to a command and control (C2) server to receive commands to interact with a victim's system. Finally, we present the general approach as well as essential elements in order to stay undetected and make the attack a success. This proof-of-concept highlights significant cybersecurity issues with openly available plugins and LLMs, which require the development of security guidelines, controls, and mitigation strategies.

摘要: 产生式人工智能的发展和新发布的大型语言模型(LLM)的能力为软件工程打开了新的机遇。然而，它们也给网络安全带来了新的挑战。最近，研究人员展示了使用ChatGPT等LLMS生成可直接利用的恶意内容或引导缺乏经验的黑客将工具和代码武器化的可能性。这些研究涵盖了仍然需要攻击者处于循环中间的场景。在这项研究中，我们利用开放可用的插件，并使用LLM作为攻击者和受害者之间的代理。我们提供了一个概念验证，其中ChatGPT用于传播恶意软件，同时逃避检测，同时建立与命令和控制(C2)服务器的通信，以接收与受害者系统交互的命令。最后，我们介绍了一般方法以及基本要素，以保持不被发现，使攻击成功。这一概念验证突出了开放可用插件和LLM存在的重大网络安全问题，这些问题需要制定安全指南、控制和缓解策略。



## **38. Demystifying RCE Vulnerabilities in LLM-Integrated Apps**

揭开LLM集成应用程序中RCE漏洞的神秘面纱 cs.CR

**SubmitDate**: 2023-09-06    [abs](http://arxiv.org/abs/2309.02926v1) [paper-pdf](http://arxiv.org/pdf/2309.02926v1)

**Authors**: Tong Liu, Zizhuang Deng, Guozhu Meng, Yuekang Li, Kai Chen

**Abstract**: In recent years, Large Language Models (LLMs) have demonstrated remarkable potential across various downstream tasks. LLM-integrated frameworks, which serve as the essential infrastructure, have given rise to many LLM-integrated web apps. However, some of these frameworks suffer from Remote Code Execution (RCE) vulnerabilities, allowing attackers to execute arbitrary code on apps' servers remotely via prompt injections. Despite the severity of these vulnerabilities, no existing work has been conducted for a systematic investigation of them. This leaves a great challenge on how to detect vulnerabilities in frameworks as well as LLM-integrated apps in real-world scenarios.   To fill this gap, we present two novel strategies, including 1) a static analysis-based tool called LLMSmith to scan the source code of the framework to detect potential RCE vulnerabilities and 2) a prompt-based automated testing approach to verify the vulnerability in LLM-integrated web apps. We discovered 13 vulnerabilities in 6 frameworks, including 12 RCE vulnerabilities and 1 arbitrary file read/write vulnerability. 11 of them are confirmed by the framework developers, resulting in the assignment of 7 CVE IDs. After testing 51 apps, we found vulnerabilities in 17 apps, 16 of which are vulnerable to RCE and 1 to SQL injection. We responsibly reported all 17 issues to the corresponding developers and received acknowledgments. Furthermore, we amplify the attack impact beyond achieving RCE by allowing attackers to exploit other app users (e.g. app responses hijacking, user API key leakage) without direct interaction between the attacker and the victim. Lastly, we propose some mitigating strategies for improving the security awareness of both framework and app developers, helping them to mitigate these risks effectively.

摘要: 近年来，大型语言模型(LLM)在各种下游任务中显示出了巨大的潜力。作为基础设施的LLM集成框架已经催生了许多LLM集成的Web应用程序。然而，其中一些框架存在远程代码执行(RCE)漏洞，使得攻击者能够通过提示注入在应用程序的服务器上远程执行任意代码。尽管这些漏洞很严重，但目前还没有对它们进行系统调查的现有工作。这就给如何在实际场景中检测框架和集成了LLM的应用程序中的漏洞留下了巨大的挑战。为了填补这一空白，我们提出了两种新的策略，包括1)基于静态分析的工具LLMSmith，用于扫描框架的源代码以检测潜在的RCE漏洞；2)基于提示的自动化测试方法，用于验证LLM集成的Web应用程序中的漏洞。我们在6个框架中发现了13个漏洞，其中12个RCE漏洞和1个任意文件读写漏洞。其中11个由框架开发者确认，分配了7个CVE ID。在测试了51个应用后，我们发现了17个应用中的漏洞，其中16个易受RCE攻击，1个易受SQL注入攻击。我们负责任地向相应的开发人员报告了所有17个问题，并收到了确认。此外，我们通过允许攻击者利用其他应用程序用户(例如，应用程序响应劫持、用户API密钥泄漏)而不在攻击者和受害者之间进行直接交互，将攻击影响放大到实现RCE之外。最后，我们提出了一些缓解策略，以提高框架和应用程序开发人员的安全意识，帮助他们有效地缓解这些风险。



## **39. A Comprehensive Overview of Backdoor Attacks in Large Language Models within Communication Networks**

通信网络中大型语言模型中的后门攻击综述 cs.CR

**SubmitDate**: 2023-09-06    [abs](http://arxiv.org/abs/2308.14367v2) [paper-pdf](http://arxiv.org/pdf/2308.14367v2)

**Authors**: Haomiao Yang, Kunlan Xiang, Mengyu Ge, Hongwei Li, Rongxing Lu, Shui Yu

**Abstract**: The Large Language Models (LLMs) are poised to offer efficient and intelligent services for future mobile communication networks, owing to their exceptional capabilities in language comprehension and generation. However, the extremely high data and computational resource requirements for the performance of LLMs compel developers to resort to outsourcing training or utilizing third-party data and computing resources. These strategies may expose the model within the network to maliciously manipulated training data and processing, providing an opportunity for attackers to embed a hidden backdoor into the model, termed a backdoor attack. Backdoor attack in LLMs refers to embedding a hidden backdoor in LLMs that causes the model to perform normally on benign samples but exhibit degraded performance on poisoned ones. This issue is particularly concerning within communication networks where reliability and security are paramount. Despite the extensive research on backdoor attacks, there remains a lack of in-depth exploration specifically within the context of LLMs employed in communication networks, and a systematic review of such attacks is currently absent. In this survey, we systematically propose a taxonomy of backdoor attacks in LLMs as used in communication networks, dividing them into four major categories: input-triggered, prompt-triggered, instruction-triggered, and demonstration-triggered attacks. Furthermore, we conduct a comprehensive analysis of the benchmark datasets. Finally, we identify potential problems and open challenges, offering valuable insights into future research directions for enhancing the security and integrity of LLMs in communication networks.

摘要: 大型语言模型因其在语言理解和生成方面的卓越能力，有望为未来的移动通信网络提供高效、智能的服务。然而，对LLM性能的极高数据和计算资源要求迫使开发人员求助于外包培训或利用第三方数据和计算资源。这些策略可能会使网络中的模型暴露于恶意操纵的训练数据和处理过程中，为攻击者提供在模型中嵌入隐藏后门的机会，称为后门攻击。LLMS中的后门攻击是指在LLMS中嵌入隐藏的后门，导致模型在良性样本上正常运行，但在有毒样本上表现出降级的性能。在可靠性和安全性至关重要的通信网络中，这个问题尤其令人担忧。尽管对后门攻击进行了广泛的研究，但仍然缺乏特别是在通信网络中使用的LLM的深入探索，而且目前还没有对这类攻击进行系统审查。在本次调查中，我们系统地提出了一种用于通信网络的LLMS后门攻击的分类，将它们分为四大类：输入触发的攻击、提示触发的攻击、指令触发的攻击和演示触发的攻击。此外，我们对基准数据集进行了全面的分析。最后，我们确定了潜在的问题和开放的挑战，为未来的研究方向提供了有价值的见解，以提高通信网络中LLMS的安全性和完整性。



## **40. Certifying LLM Safety against Adversarial Prompting**

针对敌意提示认证LLM安全 cs.CL

**SubmitDate**: 2023-09-06    [abs](http://arxiv.org/abs/2309.02705v1) [paper-pdf](http://arxiv.org/pdf/2309.02705v1)

**Authors**: Aounon Kumar, Chirag Agarwal, Suraj Srinivas, Soheil Feizi, Hima Lakkaraju

**Abstract**: Large language models (LLMs) released for public use incorporate guardrails to ensure their output is safe, often referred to as "model alignment." An aligned language model should decline a user's request to produce harmful content. However, such safety measures are vulnerable to adversarial prompts, which contain maliciously designed token sequences to circumvent the model's safety guards and cause it to produce harmful content. In this work, we introduce erase-and-check, the first framework to defend against adversarial prompts with verifiable safety guarantees. We erase tokens individually and inspect the resulting subsequences using a safety filter. Our procedure labels the input prompt as harmful if any subsequences or the input prompt are detected as harmful by the filter. This guarantees that any adversarial modification of a harmful prompt up to a certain size is also labeled harmful. We defend against three attack modes: i) adversarial suffix, which appends an adversarial sequence at the end of the prompt; ii) adversarial insertion, where the adversarial sequence is inserted anywhere in the middle of the prompt; and iii) adversarial infusion, where adversarial tokens are inserted at arbitrary positions in the prompt, not necessarily as a contiguous block. Empirical results demonstrate that our technique obtains strong certified safety guarantees on harmful prompts while maintaining good performance on safe prompts. For example, against adversarial suffixes of length 20, it certifiably detects 93% of the harmful prompts and labels 94% of the safe prompts as safe using the open source language model Llama 2 as the safety filter.

摘要: 发布给公众使用的大型语言模型(LLM)包括护栏，以确保其输出是安全的，通常被称为“模型对齐”。统一的语言模型应该拒绝用户制作有害内容的请求。然而，这样的安全措施很容易受到敌意提示的攻击，这些提示包含恶意设计的令牌序列，以绕过模型的安全警卫，使其产生有害内容。在这项工作中，我们引入了Erase-and-Check，这是第一个通过可验证的安全保证来防御敌意提示的框架。我们逐个擦除令牌，并使用安全过滤器检查得到的子序列。如果过滤器检测到任何子序列或输入提示有害，则我们的过程将输入提示标记为有害。这保证了对有害提示的任何敌意修改达到一定的大小也被标记为有害的。我们防御三种攻击模式：i)对抗性后缀，其在提示的末尾附加对抗性序列；ii)对抗性插入，其中对抗性序列被插入在提示中间的任何位置；以及iii)对抗性注入，其中对抗性标记被插入在提示中的任意位置，而不一定作为连续的块。实验结果表明，该技术在保证安全提示性能的同时，对有害提示提供了较强的认证安全保障。例如，对于长度为20的敌意后缀，它使用开源语言模型Llama 2作为安全过滤器，可证明检测到93%的有害提示和94%的安全提示是安全的。



## **41. Baseline Defenses for Adversarial Attacks Against Aligned Language Models**

针对对齐语言模型的对抗性攻击的基线防御 cs.LG

12 pages

**SubmitDate**: 2023-09-04    [abs](http://arxiv.org/abs/2309.00614v2) [paper-pdf](http://arxiv.org/pdf/2309.00614v2)

**Authors**: Neel Jain, Avi Schwarzschild, Yuxin Wen, Gowthami Somepalli, John Kirchenbauer, Ping-yeh Chiang, Micah Goldblum, Aniruddha Saha, Jonas Geiping, Tom Goldstein

**Abstract**: As Large Language Models quickly become ubiquitous, it becomes critical to understand their security vulnerabilities. Recent work shows that text optimizers can produce jailbreaking prompts that bypass moderation and alignment. Drawing from the rich body of work on adversarial machine learning, we approach these attacks with three questions: What threat models are practically useful in this domain? How do baseline defense techniques perform in this new domain? How does LLM security differ from computer vision?   We evaluate several baseline defense strategies against leading adversarial attacks on LLMs, discussing the various settings in which each is feasible and effective. Particularly, we look at three types of defenses: detection (perplexity based), input preprocessing (paraphrase and retokenization), and adversarial training. We discuss white-box and gray-box settings and discuss the robustness-performance trade-off for each of the defenses considered. We find that the weakness of existing discrete optimizers for text, combined with the relatively high costs of optimization, makes standard adaptive attacks more challenging for LLMs. Future research will be needed to uncover whether more powerful optimizers can be developed, or whether the strength of filtering and preprocessing defenses is greater in the LLMs domain than it has been in computer vision.

摘要: 随着大型语言模型迅速变得无处不在，了解它们的安全漏洞变得至关重要。最近的研究表明，文本优化器可以生成绕过审核和对齐的越狱提示。从对抗性机器学习的丰富工作中，我们用三个问题来处理这些攻击：什么威胁模型在这个领域实际上是有用的？基线防御技术在这个新领域的表现如何？LLM安全与计算机视觉有何不同？我们评估了几种针对LLMS的主要对手攻击的基线防御策略，讨论了每种策略可行和有效的各种设置。特别是，我们研究了三种类型的防御：检测(基于困惑)、输入预处理(释义和重新标记化)和对抗性训练。我们讨论了白盒和灰盒设置，并讨论了所考虑的每种防御的稳健性和性能之间的权衡。我们发现，现有的文本离散优化器的弱点，再加上相对较高的优化成本，使得标准的自适应攻击对LLMS来说更具挑战性。未来的研究将需要揭示是否可以开发出更强大的优化器，或者在LLMS领域中过滤和预处理防御的强度是否比在计算机视觉领域更强。



## **42. MathAttack: Attacking Large Language Models Towards Math Solving Ability**

MathAttack：攻击大型语言模型的数学解题能力 cs.CL

11 pages, 6 figures

**SubmitDate**: 2023-09-04    [abs](http://arxiv.org/abs/2309.01686v1) [paper-pdf](http://arxiv.org/pdf/2309.01686v1)

**Authors**: Zihao Zhou, Qiufeng Wang, Mingyu Jin, Jie Yao, Jianan Ye, Wei Liu, Wei Wang, Xiaowei Huang, Kaizhu Huang

**Abstract**: With the boom of Large Language Models (LLMs), the research of solving Math Word Problem (MWP) has recently made great progress. However, there are few studies to examine the security of LLMs in math solving ability. Instead of attacking prompts in the use of LLMs, we propose a MathAttack model to attack MWP samples which are closer to the essence of security in solving math problems. Compared to traditional text adversarial attack, it is essential to preserve the mathematical logic of original MWPs during the attacking. To this end, we propose logical entity recognition to identify logical entries which are then frozen. Subsequently, the remaining text are attacked by adopting a word-level attacker. Furthermore, we propose a new dataset RobustMath to evaluate the robustness of LLMs in math solving ability. Extensive experiments on our RobustMath and two another math benchmark datasets GSM8K and MultiAirth show that MathAttack could effectively attack the math solving ability of LLMs. In the experiments, we observe that (1) Our adversarial samples from higher-accuracy LLMs are also effective for attacking LLMs with lower accuracy (e.g., transfer from larger to smaller-size LLMs, or from few-shot to zero-shot prompts); (2) Complex MWPs (such as more solving steps, longer text, more numbers) are more vulnerable to attack; (3) We can improve the robustness of LLMs by using our adversarial samples in few-shot prompts. Finally, we hope our practice and observation can serve as an important attempt towards enhancing the robustness of LLMs in math solving ability. We will release our code and dataset.

摘要: 近年来，随着大型语言模型的兴起，数学应用题的研究取得了长足的进步。然而，很少有研究考察LLMS在数学解题能力上的安全性。在解决数学问题时，我们提出了一种MathAttack模型来攻击更接近安全本质的MWP样本，而不是使用LLMS来攻击提示。与传统的文本对抗性攻击相比，在攻击过程中必须保留原始MWP的数学逻辑。为此，我们提出了逻辑实体识别来识别然后被冻结的逻辑条目。随后，采用词级攻击者对剩余文本进行攻击。此外，我们还提出了一种新的数据集RobustMath来评估LLMS在数学求解能力方面的稳健性。在我们的RobustMath和另外两个数学基准数据集GSM8K和MultiAirth上的大量实验表明，MathAttack可以有效地攻击LLMS的数学求解能力。在实验中，我们观察到：(1)我们从高准确度的LLMS中得到的敌意样本对于攻击准确率较低的LLMS也是有效的(例如，从较大的LLMS转移到较小的LLMS，或者从少枪到零枪的提示)；(2)复杂的MWP(如更多的求解步骤、更长的文本、更多的数字)更容易受到攻击；(3)通过在少枪提示中使用我们的对手样本可以提高LLMS的健壮性。最后，我们希望我们的实践和观察能够为增强LLMS在数学求解能力方面的稳健性提供重要的尝试。我们将发布我们的代码和数据集。



## **43. OUTFOX: LLM-generated Essay Detection through In-context Learning with Adversarially Generated Examples**

Outfox：基于上下文学习的LLM生成的文章检测与恶意生成的示例 cs.CL

**SubmitDate**: 2023-09-04    [abs](http://arxiv.org/abs/2307.11729v2) [paper-pdf](http://arxiv.org/pdf/2307.11729v2)

**Authors**: Ryuto Koike, Masahiro Kaneko, Naoaki Okazaki

**Abstract**: Large Language Models (LLMs) have achieved human-level fluency in text generation, making it difficult to distinguish between human-written and LLM-generated texts. This poses a growing risk of misuse of LLMs and demands the development of detectors to identify LLM-generated texts. However, existing detectors lack robustness against attacks: they degrade detection accuracy by simply paraphrasing LLM-generated texts. Furthermore, a malicious user might attempt to deliberately evade the detectors based on detection results, but this has not been assumed in previous studies. In this paper, we propose OUTFOX, a framework that improves the robustness of LLM-generated-text detectors by allowing both the detector and the attacker to consider each other's output. In this framework, the attacker uses the detector's prediction labels as examples for in-context learning and adversarially generates essays that are harder to detect, while the detector uses the adversarially generated essays as examples for in-context learning to learn to detect essays from a strong attacker. Experiments in the domain of student essays show that the proposed detector improves the detection performance on the attacker-generated texts by up to +41.3 points in F1-score. Furthermore, the proposed detector shows a state-of-the-art detection performance: up to 96.9 points in F1-score, beating existing detectors on non-attacked texts. Finally, the proposed attacker drastically degrades the performance of detectors by up to -57.0 points F1-score, massively outperforming the baseline paraphrasing method for evading detection.

摘要: 大型语言模型(LLM)在文本生成方面达到了人类水平的流畅性，使得区分人类编写的文本和LLM生成的文本变得困难。这带来了滥用LLMS的越来越大的风险，并要求开发检测器来识别LLM生成的文本。然而，现有的检测器缺乏对攻击的稳健性：它们通过简单地解释LLM生成的文本来降低检测精度。此外，恶意用户可能试图根据检测结果故意躲避检测器，但在之前的研究中没有假设这一点。在本文中，我们提出了Outfox框架，它通过允许检测器和攻击者考虑彼此的输出来提高LLM生成的文本检测器的健壮性。在该框架中，攻击者使用检测器的预测标签作为上下文中学习的示例，并恶意生成更难检测的文章，而检测器使用恶意生成的文章作为上下文中学习的示例，以学习检测来自强大攻击者的文章。在学生作文领域的实验表明，该检测器在F1-Score上将攻击者生成的文本的检测性能提高了41.3分。此外，提出的检测器具有最先进的检测性能：在F1分数上高达96.9分，在非攻击文本上击败现有检测器。最后，提出的攻击者极大地降低了检测器的性能，最高可达-57.0分F1-Score，大大超过了用于逃避检测的基线改述方法。



## **44. Combing for Credentials: Active Pattern Extraction from Smart Reply**

梳理凭据：从智能回复中提取活动模式 cs.CR

**SubmitDate**: 2023-09-02    [abs](http://arxiv.org/abs/2207.10802v3) [paper-pdf](http://arxiv.org/pdf/2207.10802v3)

**Authors**: Bargav Jayaraman, Esha Ghosh, Melissa Chase, Sambuddha Roy, Wei Dai, David Evans

**Abstract**: Pre-trained large language models, such as GPT\nobreakdash-2 and BERT, are often fine-tuned to achieve state-of-the-art performance on a downstream task. One natural example is the ``Smart Reply'' application where a pre-trained model is tuned to provide suggested responses for a given query message. Since the tuning data is often sensitive data such as emails or chat transcripts, it is important to understand and mitigate the risk that the model leaks its tuning data. We investigate potential information leakage vulnerabilities in a typical Smart Reply pipeline. We consider a realistic setting where the adversary can only interact with the underlying model through a front-end interface that constrains what types of queries can be sent to the model. Previous attacks do not work in these settings, but require the ability to send unconstrained queries directly to the model. Even when there are no constraints on the queries, previous attacks typically require thousands, or even millions, of queries to extract useful information, while our attacks can extract sensitive data in just a handful of queries. We introduce a new type of active extraction attack that exploits canonical patterns in text containing sensitive data. We show experimentally that it is possible for an adversary to extract sensitive user information present in the training data, even in realistic settings where all interactions with the model must go through a front-end that limits the types of queries. We explore potential mitigation strategies and demonstrate empirically how differential privacy appears to be a reasonably effective defense mechanism to such pattern extraction attacks.

摘要: 预先训练的大型语言模型，如GPT\noBreakdash-2和BERT，通常会进行微调，以在下游任务中实现最先进的性能。一个自然的例子是“智能回复”应用程序，其中对预先训练的模型进行调整，以提供对给定查询消息的建议响应。由于调优数据通常是电子邮件或聊天记录等敏感数据，因此了解并降低模型泄露其调优数据的风险非常重要。我们调查了一个典型的智能回复管道中潜在的信息泄漏漏洞。我们考虑一种现实的设置，其中对手只能通过前端接口与底层模型交互，该接口限制了可以向模型发送什么类型的查询。以前的攻击在这些设置中不起作用，但需要能够将不受约束的查询直接发送到模型。即使在查询没有限制的情况下，以前的攻击通常需要数千甚至数百万个查询来提取有用的信息，而我们的攻击只需几个查询就可以提取敏感数据。我们介绍了一种新型的主动提取攻击，它利用包含敏感数据的文本中的规范模式。我们通过实验证明，对手有可能提取训练数据中存在的敏感用户信息，即使在现实环境中，与模型的所有交互都必须通过限制查询类型的前端。我们探索了潜在的缓解策略，并经验地证明了差异隐私似乎是应对此类模式提取攻击的一种相当有效的防御机制。



## **45. Why do universal adversarial attacks work on large language models?: Geometry might be the answer**

为什么通用对抗性攻击在大型语言模型上奏效？几何可能是答案 cs.LG

2nd AdvML Frontiers Workshop at 40th International Conference on  Machine Learning, Honolulu, Hawaii, USA, 2023

**SubmitDate**: 2023-09-01    [abs](http://arxiv.org/abs/2309.00254v1) [paper-pdf](http://arxiv.org/pdf/2309.00254v1)

**Authors**: Varshini Subhash, Anna Bialas, Weiwei Pan, Finale Doshi-Velez

**Abstract**: Transformer based large language models with emergent capabilities are becoming increasingly ubiquitous in society. However, the task of understanding and interpreting their internal workings, in the context of adversarial attacks, remains largely unsolved. Gradient-based universal adversarial attacks have been shown to be highly effective on large language models and potentially dangerous due to their input-agnostic nature. This work presents a novel geometric perspective explaining universal adversarial attacks on large language models. By attacking the 117M parameter GPT-2 model, we find evidence indicating that universal adversarial triggers could be embedding vectors which merely approximate the semantic information in their adversarial training region. This hypothesis is supported by white-box model analysis comprising dimensionality reduction and similarity measurement of hidden representations. We believe this new geometric perspective on the underlying mechanism driving universal attacks could help us gain deeper insight into the internal workings and failure modes of LLMs, thus enabling their mitigation.

摘要: 基于转换器的具有紧急能力的大型语言模型在社会上变得越来越普遍。然而，在对抗性攻击的背景下，理解和解释其内部运作的任务在很大程度上仍未解决。基于梯度的通用对抗性攻击已被证明在大型语言模型上非常有效，并且由于其输入不可知的性质而具有潜在的危险。这项工作提出了一种新的几何视角来解释对大型语言模型的普遍对抗性攻击。通过对117M参数GPT-2模型的攻击，我们发现有证据表明，通用的对抗性触发因素可能是嵌入的矢量，这些矢量仅近似于其对抗性训练区的语义信息。这一假设得到了白盒模型分析的支持，白盒模型分析包括对隐藏表示的降维和相似性度量。我们相信，这种关于驱动普遍攻击的潜在机制的新的几何观点可以帮助我们更深入地了解LLMS的内部工作原理和故障模式，从而使其能够得到缓解。



## **46. Temporal-Distributed Backdoor Attack Against Video Based Action Recognition**

针对视频动作识别的时间分布式后门攻击 cs.CV

**SubmitDate**: 2023-09-01    [abs](http://arxiv.org/abs/2308.11070v2) [paper-pdf](http://arxiv.org/pdf/2308.11070v2)

**Authors**: Xi Li, Songhe Wang, Ruiquan Huang, Mahanth Gowda, George Kesidis

**Abstract**: Deep neural networks (DNNs) have achieved tremendous success in various applications including video action recognition, yet remain vulnerable to backdoor attacks (Trojans). The backdoor-compromised model will mis-classify to the target class chosen by the attacker when a test instance (from a non-target class) is embedded with a specific trigger, while maintaining high accuracy on attack-free instances. Although there are extensive studies on backdoor attacks against image data, the susceptibility of video-based systems under backdoor attacks remains largely unexplored. Current studies are direct extensions of approaches proposed for image data, e.g., the triggers are independently embedded within the frames, which tend to be detectable by existing defenses. In this paper, we introduce a simple yet effective backdoor attack against video data. Our proposed attack, adding perturbations in a transformed domain, plants an imperceptible, temporally distributed trigger across the video frames, and is shown to be resilient to existing defensive strategies. The effectiveness of the proposed attack is demonstrated by extensive experiments with various well-known models on two video recognition benchmarks, UCF101 and HMDB51, and a sign language recognition benchmark, Greek Sign Language (GSL) dataset. We delve into the impact of several influential factors on our proposed attack and identify an intriguing effect termed "collateral damage" through extensive studies.

摘要: 深度神经网络(DNN)在包括视频动作识别在内的各种应用中取得了巨大的成功，但仍然容易受到后门攻击(特洛伊木马)。当测试实例(来自非目标类)嵌入特定触发器时，后门泄露模型将被错误分类为攻击者选择的目标类，同时保持对无攻击实例的高准确性。虽然已经有大量关于针对图像数据的后门攻击的研究，但基于视频的系统在后门攻击下的易感性在很大程度上仍未被探索。当前的研究是对为图像数据提出的方法的直接扩展，例如，触发器独立地嵌入在帧中，这往往是现有防御系统可检测的。本文介绍了一种简单而有效的针对视频数据的后门攻击。我们提出的攻击在变换的域中增加了扰动，在视频帧上植入了一个不可察觉的、时间分布的触发器，并被证明对现有的防御策略具有弹性。在两个视频识别基准UCF101和HMDB51和一个手语识别基准希腊手语(GSL)数据集上进行了大量的实验，证明了所提出的攻击的有效性。我们深入研究了几个影响因素对我们提出的攻击的影响，并通过广泛的研究确定了一种有趣的影响，称为“附带损害”。



## **47. LLM in the Shell: Generative Honeypots**

贝壳中的LLM：繁衍的蜜罐 cs.CR

5 pages. 1 figure 1 table

**SubmitDate**: 2023-08-31    [abs](http://arxiv.org/abs/2309.00155v1) [paper-pdf](http://arxiv.org/pdf/2309.00155v1)

**Authors**: Muris Sladić, Veronica Valeros, Carlos Catania, Sebastian Garcia

**Abstract**: Honeypots are essential tools in cybersecurity. However, most of them (even the high-interaction ones) lack the required realism to engage and fool human attackers. This limitation makes them easily discernible, hindering their effectiveness. This work introduces a novel method to create dynamic and realistic software honeypots based on Large Language Models. Preliminary results indicate that LLMs can create credible and dynamic honeypots capable of addressing important limitations of previous honeypots, such as deterministic responses, lack of adaptability, etc. We evaluated the realism of each command by conducting an experiment with human attackers who needed to say if the answer from the honeypot was fake or not. Our proposed honeypot, called shelLM, reached an accuracy rate of 0.92.

摘要: 蜜罐是网络安全中必不可少的工具。然而，他们中的大多数(即使是高度互动的人)缺乏参与和愚弄人类攻击者所需的现实主义。这一限制使得它们很容易辨别出来，从而阻碍了它们的有效性。介绍了一种基于大型语言模型创建动态、逼真的软件蜜罐的新方法。初步结果表明，LLMS可以创建可信的、动态的蜜罐，能够解决以前蜜罐的重要局限性，如确定性响应、缺乏适应性等。我们通过与需要判断蜜罐答案是否虚假的人类攻击者进行实验，评估了每个命令的真实性。我们提出的蜜罐，称为ShelLM，达到了0.92的准确率。



## **48. The Effectiveness of Large Language Models (ChatGPT and CodeBERT) for Security-Oriented Code Analysis**

大型语言模型(ChatGPT和CodeBERT)对面向安全的代码分析的有效性 cs.CR

3 Table, 8 figures

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2307.12488v3) [paper-pdf](http://arxiv.org/pdf/2307.12488v3)

**Authors**: Zhilong Wang, Lan Zhang, Chen Cao, Peng Liu

**Abstract**: Large Language Models (LLMs), such as GPT and BERT, have demonstrated remarkable capabilities in addressing neural language process tasks. Recently, the release of ChatGPT has garnered significant attention due to its ability to analyze, comprehend, and synthesize information from user inputs. Therefore, these LLMs were adopted by researchers in many different domains. In the realm of code analysis, researchers have applied LLMs to tasks like code review and code generation. However, we observed that the strengths and limitations of adopting these LLMs to the code analysis have not been investigated. In this paper, we delve into LLMs' capabilities in security-oriented program analysis, considering perspectives from both attackers and security analysts. We focus on two representative LLMs, ChatGPT and CodeBert, and evaluate their performance in solving typical analytic tasks with varying levels of difficulty. Given the different natures of ChatGPT and CodeBERT, we conduct a qualitative analysis of the model's output for ChatGPT and a quantitative analysis for CodeBERT, respectively. For ChatGPT, we present a case study involving several security-oriented program analysis tasks while deliberately introducing challenges to assess its responses. On the other hand, for CodeBERT, we systematically analyze and classify the features in code, quantitatively evaluating the impact of these features on the model's performance. Our study demonstrates the LLM's efficiency in learning high-level semantics from code, positioning ChatGPT as a potential asset in security-oriented contexts. However, it is essential to acknowledge certain limitations, such as the heavy reliance on well-defined variable and function names, making them unable to learn from anonymized code. We hope that our findings and analysis will offer valuable insights for future researchers in this domain.

摘要: 大型语言模型(LLM)，如GPT和BERT，在处理神经语言处理任务方面表现出了非凡的能力。最近，ChatGPT的发布受到了极大的关注，因为它能够分析、理解和综合来自用户输入的信息。因此，这些LLM被许多不同领域的研究人员所采用。在代码分析领域，研究人员已经将LLM应用于代码审查和代码生成等任务。然而，我们注意到，采用这些LLM进行代码分析的优点和局限性尚未得到调查。在本文中，我们从攻击者和安全分析师的角度深入研究了LLMS在面向安全的程序分析中的能力。我们集中于两个有代表性的LLM，ChatGPT和CodeBert，并评估了它们在解决不同难度的典型分析任务中的性能。鉴于ChatGPT和CodeBERT的不同性质，我们分别对ChatGPT和CodeBERT的模型输出进行了定性分析和定量分析。对于ChatGPT，我们提供了一个案例研究，涉及几个面向安全的程序分析任务，同时故意引入挑战来评估其响应。另一方面，对于CodeBERT，我们对代码中的特征进行了系统的分析和分类，定量地评估了这些特征对模型性能的影响。我们的研究证明了LLM在从代码中学习高级语义方面的效率，并将ChatGPT定位为面向安全的上下文中的潜在资产。然而，必须承认某些限制，例如严重依赖定义明确的变量和函数名称，使它们无法从匿名代码中学习。我们希望我们的发现和分析将为这一领域的未来研究人员提供有价值的见解。



## **49. Identifying and Mitigating the Security Risks of Generative AI**

识别和缓解生成性人工智能的安全风险 cs.AI

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2308.14840v1) [paper-pdf](http://arxiv.org/pdf/2308.14840v1)

**Authors**: Clark Barrett, Brad Boyd, Ellie Burzstein, Nicholas Carlini, Brad Chen, Jihye Choi, Amrita Roy Chowdhury, Mihai Christodorescu, Anupam Datta, Soheil Feizi, Kathleen Fisher, Tatsunori Hashimoto, Dan Hendrycks, Somesh Jha, Daniel Kang, Florian Kerschbaum, Eric Mitchell, John Mitchell, Zulfikar Ramzan, Khawaja Shams, Dawn Song, Ankur Taly, Diyi Yang

**Abstract**: Every major technical invention resurfaces the dual-use dilemma -- the new technology has the potential to be used for good as well as for harm. Generative AI (GenAI) techniques, such as large language models (LLMs) and diffusion models, have shown remarkable capabilities (e.g., in-context learning, code-completion, and text-to-image generation and editing). However, GenAI can be used just as well by attackers to generate new attacks and increase the velocity and efficacy of existing attacks.   This paper reports the findings of a workshop held at Google (co-organized by Stanford University and the University of Wisconsin-Madison) on the dual-use dilemma posed by GenAI. This paper is not meant to be comprehensive, but is rather an attempt to synthesize some of the interesting findings from the workshop. We discuss short-term and long-term goals for the community on this topic. We hope this paper provides both a launching point for a discussion on this important topic as well as interesting problems that the research community can work to address.

摘要: 每一项重大技术发明都会重新面临两难境地--新技术既有可能被用来做好事，也有可能被用来做坏事。生成性人工智能(GenAI)技术，如大型语言模型(LLMS)和扩散模型，已经显示出非凡的能力(例如，上下文学习、代码完成以及文本到图像的生成和编辑)。然而，攻击者也可以利用GenAI来生成新的攻击，并提高现有攻击的速度和效率。本文报告了在谷歌(由斯坦福大学和威斯康星大学麦迪逊分校联合举办)举行的关于GenAI造成的两用困境的研讨会的结果。这篇论文并不是要全面的，而是试图综合研讨会的一些有趣的发现。我们就这一主题讨论社区的短期和长期目标。我们希望这篇论文既为讨论这一重要主题提供了一个起点，也为研究界可以努力解决的有趣问题提供了一个起点。



## **50. Out of the Cage: How Stochastic Parrots Win in Cyber Security Environments**

走出笼子：随机鹦鹉如何在网络安全环境中取胜 cs.CR

Under review. 10 pages plus appendices, 7 figures, 4 tables. Edit:  fix e-mails and code repository

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2308.12086v2) [paper-pdf](http://arxiv.org/pdf/2308.12086v2)

**Authors**: Maria Rigaki, Ondřej Lukáš, Carlos A. Catania, Sebastian Garcia

**Abstract**: Large Language Models (LLMs) have gained widespread popularity across diverse domains involving text generation, summarization, and various natural language processing tasks. Despite their inherent limitations, LLM-based designs have shown promising capabilities in planning and navigating open-world scenarios. This paper introduces a novel application of pre-trained LLMs as agents within cybersecurity network environments, focusing on their utility for sequential decision-making processes.   We present an approach wherein pre-trained LLMs are leveraged as attacking agents in two reinforcement learning environments. Our proposed agents demonstrate similar or better performance against state-of-the-art agents trained for thousands of episodes in most scenarios and configurations. In addition, the best LLM agents perform similarly to human testers of the environment without any additional training process. This design highlights the potential of LLMs to efficiently address complex decision-making tasks within cybersecurity.   Furthermore, we introduce a new network security environment named NetSecGame. The environment is designed to eventually support complex multi-agent scenarios within the network security domain. The proposed environment mimics real network attacks and is designed to be highly modular and adaptable for various scenarios.

摘要: 大语言模型在涉及文本生成、摘要和各种自然语言处理任务的不同领域得到了广泛的欢迎。尽管有其固有的局限性，基于LLM的设计在规划和导航开放世界场景方面显示出了良好的能力。本文介绍了在网络安全网络环境中将预先训练的LLM作为代理的一种新的应用，重点讨论了它们在顺序决策过程中的效用。我们提出了一种方法，其中预先训练的LLM在两个强化学习环境中被用作攻击代理。我们建议的代理与在大多数场景和配置中培训了数千集的最先进的代理相比，表现出类似或更好的性能。此外，最好的LLM代理的表现类似于环境中的人类测试员，而无需任何额外的培训过程。此设计突出了LLMS在有效处理网络安全中的复杂决策任务方面的潜力。此外，我们还介绍了一个新的网络安全环境NetSecGame。该环境旨在最终支持网络安全域内的复杂多代理方案。提出的环境模拟真实的网络攻击，并被设计为高度模块化和适应各种场景。



