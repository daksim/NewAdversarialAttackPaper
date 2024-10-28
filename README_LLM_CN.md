# Latest Large Language Model Attack Papers
**update at 2024-10-28 09:34:11**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Expose Before You Defend: Unifying and Enhancing Backdoor Defenses via Exposed Models**

防御前暴露：通过暴露的模型统一和增强后门防御 cs.AI

19 pages

**SubmitDate**: 2024-10-25    [abs](http://arxiv.org/abs/2410.19427v1) [paper-pdf](http://arxiv.org/pdf/2410.19427v1)

**Authors**: Yige Li, Hanxun Huang, Jiaming Zhang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Backdoor attacks covertly implant triggers into deep neural networks (DNNs) by poisoning a small portion of the training data with pre-designed backdoor triggers. This vulnerability is exacerbated in the era of large models, where extensive (pre-)training on web-crawled datasets is susceptible to compromise. In this paper, we introduce a novel two-step defense framework named Expose Before You Defend (EBYD). EBYD unifies existing backdoor defense methods into a comprehensive defense system with enhanced performance. Specifically, EBYD first exposes the backdoor functionality in the backdoored model through a model preprocessing step called backdoor exposure, and then applies detection and removal methods to the exposed model to identify and eliminate the backdoor features. In the first step of backdoor exposure, we propose a novel technique called Clean Unlearning (CUL), which proactively unlearns clean features from the backdoored model to reveal the hidden backdoor features. We also explore various model editing/modification techniques for backdoor exposure, including fine-tuning, model sparsification, and weight perturbation. Using EBYD, we conduct extensive experiments on 10 image attacks and 6 text attacks across 2 vision datasets (CIFAR-10 and an ImageNet subset) and 4 language datasets (SST-2, IMDB, Twitter, and AG's News). The results demonstrate the importance of backdoor exposure for backdoor defense, showing that the exposed models can significantly benefit a range of downstream defense tasks, including backdoor label detection, backdoor trigger recovery, backdoor model detection, and backdoor removal. We hope our work could inspire more research in developing advanced defense frameworks with exposed models. Our code is available at: https://github.com/bboylyg/Expose-Before-You-Defend.

摘要: 后门攻击通过用预先设计的后门触发器毒化一小部分训练数据，秘密地将触发器植入深度神经网络(DNN)。这种脆弱性在大型模型时代加剧，在这个时代，对网络爬行的数据集进行广泛的(预先)培训很容易受到损害。在本文中，我们介绍了一种新的两步防御框架--先暴露后防御(EBYD)。EBYD将现有的后门防御方法统一到一个性能增强的全面防御系统中。具体地说，EBYD首先通过称为后门暴露的模型预处理步骤暴露后门模型中的后门功能，然后对暴露的模型应用检测和删除方法以识别和消除后门功能。在后门暴露的第一步，我们提出了一种新的技术，称为清洁遗忘(Clean UnLearning，CUL)，它主动地从后门模型中忘记干净的特征，以揭示隐藏的后门特征。我们还探索了用于后门曝光的各种模型编辑/修改技术，包括微调、模型稀疏和权重扰动。使用EBYD，我们在2个视觉数据集(CIFAR-10和ImageNet子集)和4个语言数据集(SST-2、IMDB、Twitter和AG的新闻)上进行了10次图像攻击和6次文本攻击的广泛实验。结果表明，后门暴露对于后门防御的重要性，表明暴露的模型可以显著受益于一系列下游防御任务，包括后门标签检测、后门触发恢复、后门模型检测和后门删除。我们希望我们的工作能够启发更多的研究，开发具有暴露模型的先进防御框架。我们的代码请访问：https://github.com/bboylyg/Expose-Before-You-Defend.



## **2. Humanizing the Machine: Proxy Attacks to Mislead LLM Detectors**

人性化机器：代理攻击误导LLM检测器 cs.LG

26 pages

**SubmitDate**: 2024-10-25    [abs](http://arxiv.org/abs/2410.19230v1) [paper-pdf](http://arxiv.org/pdf/2410.19230v1)

**Authors**: Tianchun Wang, Yuanzhou Chen, Zichuan Liu, Zhanwen Chen, Haifeng Chen, Xiang Zhang, Wei Cheng

**Abstract**: The advent of large language models (LLMs) has revolutionized the field of text generation, producing outputs that closely mimic human-like writing. Although academic and industrial institutions have developed detectors to prevent the malicious usage of LLM-generated texts, other research has doubt about the robustness of these systems. To stress test these detectors, we introduce a proxy-attack strategy that effortlessly compromises LLMs, causing them to produce outputs that align with human-written text and mislead detection systems. Our method attacks the source model by leveraging a reinforcement learning (RL) fine-tuned humanized small language model (SLM) in the decoding phase. Through an in-depth analysis, we demonstrate that our attack strategy is capable of generating responses that are indistinguishable to detectors, preventing them from differentiating between machine-generated and human-written text. We conduct systematic evaluations on extensive datasets using proxy-attacked open-source models, including Llama2-13B, Llama3-70B, and Mixtral-8*7B in both white- and black-box settings. Our findings show that the proxy-attack strategy effectively deceives the leading detectors, resulting in an average AUROC drop of 70.4% across multiple datasets, with a maximum drop of 90.3% on a single dataset. Furthermore, in cross-discipline scenarios, our strategy also bypasses these detectors, leading to a significant relative decrease of up to 90.9%, while in cross-language scenario, the drop reaches 91.3%. Despite our proxy-attack strategy successfully bypassing the detectors with such significant relative drops, we find that the generation quality of the attacked models remains preserved, even within a modest utility budget, when compared to the text produced by the original, unattacked source model.

摘要: 大型语言模型(LLM)的出现使文本生成领域发生了革命性的变化，产生了非常接近人类书写的输出。尽管学术和工业机构已经开发了检测器来防止恶意使用LLM生成的文本，但其他研究对这些系统的健壮性表示怀疑。为了对这些检测器进行压力测试，我们引入了一种代理攻击策略，该策略可以毫不费力地攻击LLM，使它们产生与人类书写的文本和误导检测系统一致的输出。我们的方法在解码阶段利用强化学习(RL)微调的人性化小语言模型(SLM)来攻击源模型。通过深入的分析，我们证明了我们的攻击策略能够产生检测器无法区分的响应，防止他们区分机器生成的文本和人类书写的文本。我们使用代理攻击的开源模型对大量的数据集进行了系统的评估，包括白盒和黑盒环境下的Llama2-13B、Llama3-70B和Mixtral-8*7B。结果表明，代理攻击策略有效地欺骗了领先的检测器，导致多个数据集的AUROC平均下降了70.4%，其中单个数据集的AUROC平均下降了90.3%。此外，在跨学科场景中，我们的策略也绕过了这些检测器，导致了高达90.9%的显著相对降幅，而在跨语言场景中，降幅达到91.3%。尽管我们的代理攻击策略成功地绕过了检测器，具有如此显著的相对下降，但我们发现，与原始的未受攻击源模型生成的文本相比，即使在适度的实用预算内，受攻击模型的生成质量仍然保持不变。



## **3. Integrating Large Language Models with Internet of Things Applications**

将大型语言模型与物联网应用程序集成 cs.AI

**SubmitDate**: 2024-10-25    [abs](http://arxiv.org/abs/2410.19223v1) [paper-pdf](http://arxiv.org/pdf/2410.19223v1)

**Authors**: Mingyu Zong, Arvin Hekmati, Michael Guastalla, Yiyi Li, Bhaskar Krishnamachari

**Abstract**: This paper identifies and analyzes applications in which Large Language Models (LLMs) can make Internet of Things (IoT) networks more intelligent and responsive through three case studies from critical topics: DDoS attack detection, macroprogramming over IoT systems, and sensor data processing. Our results reveal that the GPT model under few-shot learning achieves 87.6% detection accuracy, whereas the fine-tuned GPT increases the value to 94.9%. Given a macroprogramming framework, the GPT model is capable of writing scripts using high-level functions from the framework to handle possible incidents. Moreover, the GPT model shows efficacy in processing a vast amount of sensor data by offering fast and high-quality responses, which comprise expected results and summarized insights. Overall, the model demonstrates its potential to power a natural language interface. We hope that researchers will find these case studies inspiring to develop further.

摘要: 本文通过关键主题的三个案例研究来识别和分析大型语言模型（LLM）可以使物联网（IoT）网络更加智能和响应能力更强的应用程序：DDOS攻击检测、物联网系统上的宏编程和传感器数据处理。我们的结果表明，少次学习下的GPT模型可实现87.6%的检测准确率，而微调的GPT可将该值提高到94.9%。给定宏编程框架，GPT模型能够使用框架中的高级函数编写脚本来处理可能的事件。此外，GPT模型通过提供快速、高质量的响应（包括预期结果和总结见解）显示出处理大量传感器数据的功效。总体而言，该模型展示了其支持自然语言界面的潜力。我们希望研究人员能够发现这些案例研究鼓舞人心，以进一步发展。



## **4. Adversarial Attacks on Large Language Models Using Regularized Relaxation**

使用正规松弛对大型语言模型的对抗攻击 cs.LG

8 pages, 6 figures

**SubmitDate**: 2024-10-24    [abs](http://arxiv.org/abs/2410.19160v1) [paper-pdf](http://arxiv.org/pdf/2410.19160v1)

**Authors**: Samuel Jacob Chacko, Sajib Biswas, Chashi Mahiul Islam, Fatema Tabassum Liza, Xiuwen Liu

**Abstract**: As powerful Large Language Models (LLMs) are now widely used for numerous practical applications, their safety is of critical importance. While alignment techniques have significantly improved overall safety, LLMs remain vulnerable to carefully crafted adversarial inputs. Consequently, adversarial attack methods are extensively used to study and understand these vulnerabilities. However, current attack methods face significant limitations. Those relying on optimizing discrete tokens suffer from limited efficiency, while continuous optimization techniques fail to generate valid tokens from the model's vocabulary, rendering them impractical for real-world applications. In this paper, we propose a novel technique for adversarial attacks that overcomes these limitations by leveraging regularized gradients with continuous optimization methods. Our approach is two orders of magnitude faster than the state-of-the-art greedy coordinate gradient-based method, significantly improving the attack success rate on aligned language models. Moreover, it generates valid tokens, addressing a fundamental limitation of existing continuous optimization methods. We demonstrate the effectiveness of our attack on five state-of-the-art LLMs using four datasets.

摘要: 随着强大的大型语言模型(LLM)在众多实际应用中的广泛应用，它们的安全性至关重要。虽然对齐技术显著提高了总体安全性，但LLM仍然容易受到精心设计的敌方输入的影响。因此，对抗性攻击方法被广泛用于研究和理解这些漏洞。然而，目前的攻击方法面临着很大的局限性。那些依赖于优化离散令牌的人效率有限，而连续优化技术无法从模型的词汇表中生成有效的令牌，这使得它们在现实世界中的应用不切实际。在本文中，我们提出了一种新的对抗性攻击技术，通过利用正则化的梯度和连续优化方法来克服这些局限性。我们的方法比最先进的贪婪坐标梯度方法快两个数量级，显著提高了对齐语言模型的攻击成功率。此外，它还生成有效的令牌，解决了现有连续优化方法的一个基本限制。我们使用四个数据集演示了我们对五个最先进的LLM的攻击的有效性。



## **5. Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications**

通过修剪和低级修改评估安全对齐的脆弱性 cs.LG

22 pages, 9 figures. Project page is available at  https://boyiwei.com/alignment-attribution/

**SubmitDate**: 2024-10-24    [abs](http://arxiv.org/abs/2402.05162v4) [paper-pdf](http://arxiv.org/pdf/2402.05162v4)

**Authors**: Boyi Wei, Kaixuan Huang, Yangsibo Huang, Tinghao Xie, Xiangyu Qi, Mengzhou Xia, Prateek Mittal, Mengdi Wang, Peter Henderson

**Abstract**: Large language models (LLMs) show inherent brittleness in their safety mechanisms, as evidenced by their susceptibility to jailbreaking and even non-malicious fine-tuning. This study explores this brittleness of safety alignment by leveraging pruning and low-rank modifications. We develop methods to identify critical regions that are vital for safety guardrails, and that are disentangled from utility-relevant regions at both the neuron and rank levels. Surprisingly, the isolated regions we find are sparse, comprising about $3\%$ at the parameter level and $2.5\%$ at the rank level. Removing these regions compromises safety without significantly impacting utility, corroborating the inherent brittleness of the model's safety mechanisms. Moreover, we show that LLMs remain vulnerable to low-cost fine-tuning attacks even when modifications to the safety-critical regions are restricted. These findings underscore the urgent need for more robust safety strategies in LLMs.

摘要: 大型语言模型（LLM）在其安全机制中表现出固有的脆弱性，这一点从它们容易越狱甚至非恶意微调中得到了证明。这项研究通过利用修剪和低等级修改来探索安全对齐的脆弱性。我们开发了识别对安全护栏至关重要的关键区域的方法，并且在神经元和等级水平上与公用事业相关区域脱钩。令人惊讶的是，我们发现的孤立区域很稀疏，参数级别约为3美元，排名级别约为2.5美元。删除这些区域会损害安全性，而不会显着影响实用性，这证实了该模型安全机制固有的脆弱性。此外，我们表明，即使对安全关键区域的修改受到限制，LLM仍然容易受到低成本微调攻击。这些发现凸显了LLM迫切需要制定更稳健的安全策略。



## **6. Watermarking Large Language Models and the Generated Content: Opportunities and Challenges**

对大型语言模型和生成的内容进行水印：机遇与挑战 cs.CR

invited paper to Asilomar Conference on Signals, Systems, and  Computers

**SubmitDate**: 2024-10-24    [abs](http://arxiv.org/abs/2410.19096v1) [paper-pdf](http://arxiv.org/pdf/2410.19096v1)

**Authors**: Ruisi Zhang, Farinaz Koushanfar

**Abstract**: The widely adopted and powerful generative large language models (LLMs) have raised concerns about intellectual property rights violations and the spread of machine-generated misinformation. Watermarking serves as a promising approch to establish ownership, prevent unauthorized use, and trace the origins of LLM-generated content. This paper summarizes and shares the challenges and opportunities we found when watermarking LLMs. We begin by introducing techniques for watermarking LLMs themselves under different threat models and scenarios. Next, we investigate watermarking methods designed for the content generated by LLMs, assessing their effectiveness and resilience against various attacks. We also highlight the importance of watermarking domain-specific models and data, such as those used in code generation, chip design, and medical applications. Furthermore, we explore methods like hardware acceleration to improve the efficiency of the watermarking process. Finally, we discuss the limitations of current approaches and outline future research directions for the responsible use and protection of these generative AI tools.

摘要: 被广泛采用且功能强大的生成性大型语言模型(LLM)引起了人们对侵犯知识产权和机器生成错误信息传播的担忧。水印是一种很有前途的方法，可以建立所有权，防止未经授权的使用，并追踪LLM生成的内容的来源。本文总结和分享了在对LLMS进行水印处理时所遇到的挑战和机遇。我们首先介绍在不同威胁模型和场景下为LLM本身添加水印的技术。接下来，我们研究了针对LLMS生成的内容设计的水印方法，评估了它们对各种攻击的有效性和抗攻击能力。我们还强调了为特定领域的模型和数据添加水印的重要性，例如用于代码生成、芯片设计和医疗应用的模型和数据。此外，我们还探索了硬件加速等方法来提高水印过程的效率。最后，我们讨论了当前方法的局限性，并概述了未来的研究方向，以负责任地使用和保护这些生成性人工智能工具。



## **7. Provably Robust Watermarks for Open-Source Language Models**

开源语言模型的可证明稳健的水印 cs.CR

**SubmitDate**: 2024-10-24    [abs](http://arxiv.org/abs/2410.18861v1) [paper-pdf](http://arxiv.org/pdf/2410.18861v1)

**Authors**: Miranda Christ, Sam Gunn, Tal Malkin, Mariana Raykova

**Abstract**: The recent explosion of high-quality language models has necessitated new methods for identifying AI-generated text. Watermarking is a leading solution and could prove to be an essential tool in the age of generative AI. Existing approaches embed watermarks at inference and crucially rely on the large language model (LLM) specification and parameters being secret, which makes them inapplicable to the open-source setting. In this work, we introduce the first watermarking scheme for open-source LLMs. Our scheme works by modifying the parameters of the model, but the watermark can be detected from just the outputs of the model. Perhaps surprisingly, we prove that our watermarks are unremovable under certain assumptions about the adversary's knowledge. To demonstrate the behavior of our construction under concrete parameter instantiations, we present experimental results with OPT-6.7B and OPT-1.3B. We demonstrate robustness to both token substitution and perturbation of the model parameters. We find that the stronger of these attacks, the model-perturbation attack, requires deteriorating the quality score to 0 out of 100 in order to bring the detection rate down to 50%.

摘要: 最近高质量语言模型的爆炸性增长需要新的方法来识别人工智能生成的文本。水印是一种领先的解决方案，可能会被证明是生成性人工智能时代的重要工具。现有的方法在推理时嵌入水印，重要的是依赖于大型语言模型(LLM)规范和参数是保密的，这使得它们不适用于开源环境。在这项工作中，我们介绍了第一个用于开源LLMS的水印方案。我们的方案通过修改模型的参数来工作，但仅从模型的输出就可以检测到水印。也许令人惊讶的是，我们证明了我们的水印在关于对手知识的某些假设下是不可移除的。为了演示我们的构造在混凝土参数实例化下的行为，我们给出了使用OPT-6.7B和OPT-1.3B的实验结果。我们证明了对令牌替换和模型参数摄动的稳健性。我们发现，在这些攻击中，较强的模型扰动攻击需要将质量分数恶化到0分(满分100分)，才能将检测率降至50%。



## **8. PSY: Posterior Sampling Based Privacy Enhancer in Large Language Models**

PSY：大型语言模型中基于后验抽样的隐私增强器 cs.CR

10 pages, 2 figures

**SubmitDate**: 2024-10-24    [abs](http://arxiv.org/abs/2410.18824v1) [paper-pdf](http://arxiv.org/pdf/2410.18824v1)

**Authors**: Yulian Sun, Li Duan, Yong Li

**Abstract**: Privacy vulnerabilities in LLMs, such as leakage from memorization, have been constantly identified, and various mitigation proposals have been proposed. LoRA is usually used in fine-tuning LLMs and a good entry point to insert privacy-enhancing modules. In this ongoing research, we introduce PSY, a Posterior Sampling based PrivacY enhancer that can be used in LoRA. We propose a simple yet effective realization of PSY using posterior sampling, which effectively prevents privacy leakage from intermediate information and, in turn, preserves the privacy of data owners. We evaluate LoRA extended with PSY against state-of-the-art membership inference and data extraction attacks. The experiments are executed on three different LLM architectures fine-tuned on three datasets with LoRA. In contrast to the commonly used differential privacy method, we find that our proposed modification consistently reduces the attack success rate. Meanwhile, our method has almost no negative impact on model fine-tuning or final performance. Most importantly, PSY reveals a promising path toward privacy enhancement with latent space extensions.

摘要: LLMS中的隐私漏洞，如记忆泄漏，不断被发现，并提出了各种缓解建议。LORA通常用于微调LLM，是插入隐私增强模块的一个很好的切入点。在这项正在进行的研究中，我们介绍了PSY，一种基于后验抽样的隐私增强器，可以在LORA中使用。我们提出了一种简单而有效的后验抽样的PSY实现方法，它有效地防止了中间信息的隐私泄露，进而保护了数据所有者的隐私。我们评估了使用PSY扩展的LORA对最先进的成员关系推理和数据提取攻击的抵抗力。实验是在三种不同的LLM架构上进行的，这些架构使用LORA在三个数据集上进行了微调。与常用的差分隐私方法相比，我们发现我们提出的修改一致地降低了攻击成功率。同时，我们的方法对模型微调或最终性能几乎没有负面影响。最重要的是，PSY揭示了一条通过潜在空间扩展来增强隐私的有希望的途径。



## **9. Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities**

迭代自调优LLM以增强越狱能力 cs.CL

18 pages

**SubmitDate**: 2024-10-24    [abs](http://arxiv.org/abs/2410.18469v1) [paper-pdf](http://arxiv.org/pdf/2410.18469v1)

**Authors**: Chung-En Sun, Xiaodong Liu, Weiwei Yang, Tsui-Wei Weng, Hao Cheng, Aidan San, Michel Galley, Jianfeng Gao

**Abstract**: Recent research has shown that Large Language Models (LLMs) are vulnerable to automated jailbreak attacks, where adversarial suffixes crafted by algorithms appended to harmful queries bypass safety alignment and trigger unintended responses. Current methods for generating these suffixes are computationally expensive and have low Attack Success Rates (ASR), especially against well-aligned models like Llama2 and Llama3. To overcome these limitations, we introduce ADV-LLM, an iterative self-tuning process that crafts adversarial LLMs with enhanced jailbreak ability. Our framework significantly reduces the computational cost of generating adversarial suffixes while achieving nearly 100\% ASR on various open-source LLMs. Moreover, it exhibits strong attack transferability to closed-source models, achieving 99% ASR on GPT-3.5 and 49% ASR on GPT-4, despite being optimized solely on Llama3. Beyond improving jailbreak ability, ADV-LLM provides valuable insights for future safety alignment research through its ability to generate large datasets for studying LLM safety. Our code is available at: https://github.com/SunChungEn/ADV-LLM

摘要: 最近的研究表明，大型语言模型(LLM)容易受到自动越狱攻击，在自动越狱攻击中，由附加到有害查询的算法编制的敌意后缀绕过安全对齐并触发意外响应。目前生成这些后缀的方法计算量大，攻击成功率(ASR)低，尤其是针对Llama2和Llama3等排列良好的模型。为了克服这些限制，我们引入了ADV-LLM，这是一个迭代的自我调整过程，可以制作具有增强越狱能力的对抗性LLM。我们的框架大大降低了生成敌意后缀的计算代价，同时在各种开源LLM上实现了近100个ASR。此外，它表现出很强的攻击可转换性，尽管只在Llama3上进行了优化，但在GPT-3.5上实现了99%的ASR，在GPT-4上实现了49%的ASR。除了提高越狱能力，ADV-LLM还通过其生成用于研究LLM安全性的大型数据集的能力，为未来的安全配准研究提供了有价值的见解。我们的代码请访问：https://github.com/SunChungEn/ADV-LLM



## **10. Advancing NLP Security by Leveraging LLMs as Adversarial Engines**

通过利用LLC作为对抗引擎来提高NLP安全性 cs.AI

5 pages

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.18215v1) [paper-pdf](http://arxiv.org/pdf/2410.18215v1)

**Authors**: Sudarshan Srinivasan, Maria Mahbub, Amir Sadovnik

**Abstract**: This position paper proposes a novel approach to advancing NLP security by leveraging Large Language Models (LLMs) as engines for generating diverse adversarial attacks. Building upon recent work demonstrating LLMs' effectiveness in creating word-level adversarial examples, we argue for expanding this concept to encompass a broader range of attack types, including adversarial patches, universal perturbations, and targeted attacks. We posit that LLMs' sophisticated language understanding and generation capabilities can produce more effective, semantically coherent, and human-like adversarial examples across various domains and classifier architectures. This paradigm shift in adversarial NLP has far-reaching implications, potentially enhancing model robustness, uncovering new vulnerabilities, and driving innovation in defense mechanisms. By exploring this new frontier, we aim to contribute to the development of more secure, reliable, and trustworthy NLP systems for critical applications.

摘要: 这份立场文件提出了一种新颖的方法，通过利用大型语言模型（LLM）作为生成多样化对抗攻击的引擎来提高NLP安全性。在最近展示LLM在创建单词级对抗性示例方面有效性的工作的基础上，我们主张扩展这一概念以涵盖更广泛的攻击类型，包括对抗性补丁、普遍扰动和有针对性的攻击。我们证实，LLM复杂的语言理解和生成能力可以在各个领域和分类器架构中生成更有效、语义一致且类人的对抗性示例。对抗性NLP的这种范式转变具有深远的影响，可能会增强模型稳健性、发现新的漏洞并推动防御机制的创新。通过探索这一新领域，我们的目标是为关键应用开发更安全、可靠和值得信赖的NLP系统做出贡献。



## **11. Towards Understanding the Fragility of Multilingual LLMs against Fine-Tuning Attacks**

了解多语言LLM对微调攻击的脆弱性 cs.CL

14 pages, 6 figures, 7 tables

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.18210v1) [paper-pdf](http://arxiv.org/pdf/2410.18210v1)

**Authors**: Samuele Poppi, Zheng-Xin Yong, Yifei He, Bobbie Chern, Han Zhao, Aobo Yang, Jianfeng Chi

**Abstract**: Recent advancements in Large Language Models (LLMs) have sparked widespread concerns about their safety. Recent work demonstrates that safety alignment of LLMs can be easily removed by fine-tuning with a few adversarially chosen instruction-following examples, i.e., fine-tuning attacks. We take a further step to understand fine-tuning attacks in multilingual LLMs. We first discover cross-lingual generalization of fine-tuning attacks: using a few adversarially chosen instruction-following examples in one language, multilingual LLMs can also be easily compromised (e.g., multilingual LLMs fail to refuse harmful prompts in other languages). Motivated by this finding, we hypothesize that safety-related information is language-agnostic and propose a new method termed Safety Information Localization (SIL) to identify the safety-related information in the model parameter space. Through SIL, we validate this hypothesis and find that only changing 20% of weight parameters in fine-tuning attacks can break safety alignment across all languages. Furthermore, we provide evidence to the alternative pathways hypothesis for why freezing safety-related parameters does not prevent fine-tuning attacks, and we demonstrate that our attack vector can still jailbreak LLMs adapted to new languages.

摘要: 最近大型语言模型(LLM)的进步引发了人们对其安全性的广泛担忧。最近的工作表明，通过使用一些恶意选择的指令跟随示例，即微调攻击，可以很容易地删除LLM的安全对齐。我们进一步了解多语言LLM中的微调攻击。我们首先发现了微调攻击的跨语言泛化：使用几个恶意选择的一种语言的指令跟随示例，多语言LLM也很容易被攻破(例如，多语言LLM无法拒绝其他语言的有害提示)。基于这一发现，我们假设安全相关信息是语言不可知的，并提出了一种在模型参数空间中识别安全相关信息的新方法--安全信息本地化。通过SIL语言，我们验证了这一假设，发现在微调攻击中只改变20%的权重参数就可以破坏所有语言的安全对齐。此外，我们为替代路径假说提供了证据，证明了冻结安全相关参数为什么不能防止微调攻击，并证明了我们的攻击向量仍然可以越狱适应新语言的LLM。



## **12. Safeguard is a Double-edged Sword: Denial-of-service Attack on Large Language Models**

保障是一把双刃剑：对大型语言模型的拒绝服务攻击 cs.CR

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.02916v2) [paper-pdf](http://arxiv.org/pdf/2410.02916v2)

**Authors**: Qingzhao Zhang, Ziyang Xiong, Z. Morley Mao

**Abstract**: Safety is a paramount concern of large language models (LLMs) in their open deployment. To this end, safeguard methods aim to enforce the ethical and responsible use of LLMs through safety alignment or guardrail mechanisms. However, we found that the malicious attackers could exploit false positives of safeguards, i.e., fooling the safeguard model to block safe content mistakenly, leading to a new denial-of-service (DoS) attack on LLMs. Specifically, by software or phishing attacks on user client software, attackers insert a short, seemingly innocuous adversarial prompt into to user prompt templates in configuration files; thus, this prompt appears in final user requests without visibility in the user interface and is not trivial to identify. By designing an optimization process that utilizes gradient and attention information, our attack can automatically generate seemingly safe adversarial prompts, approximately only 30 characters long, that universally block over 97\% of user requests on Llama Guard 3. The attack presents a new dimension of evaluating LLM safeguards focusing on false positives, fundamentally different from the classic jailbreak.

摘要: 安全是大型语言模型(LLM)在开放部署时最关心的问题。为此，保障措施旨在通过安全调整或护栏机制，强制以合乎道德和负责任的方式使用LLMS。然而，我们发现恶意攻击者可以利用安全措施的误报，即欺骗安全措施模型错误地阻止安全内容，从而导致对LLMS的新的拒绝服务(DoS)攻击。具体地说，通过软件或对用户客户端软件的网络钓鱼攻击，攻击者将一个看似无害的简短对抗性提示插入到配置文件中的用户提示模板中；因此，该提示出现在最终用户请求中，在用户界面中不可见，并且很难识别。通过设计一个利用梯度和注意力信息的优化过程，我们的攻击可以自动生成看似安全的敌意提示，大约只有30个字符，普遍阻止Llama Guard 3上超过97%的用户请求。该攻击提供了一个新的维度来评估LLM安全措施，从根本上不同于传统的越狱。



## **13. SCA: Highly Efficient Semantic-Consistent Unrestricted Adversarial Attack**

SCA：高效语义一致的无限制对抗攻击 cs.CV

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.02240v4) [paper-pdf](http://arxiv.org/pdf/2410.02240v4)

**Authors**: Zihao Pan, Weibin Wu, Yuhang Cao, Zibin Zheng

**Abstract**: Deep neural network based systems deployed in sensitive environments are vulnerable to adversarial attacks. Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic. Recent works have utilized the diffusion inversion process to map images into a latent space, where high-level semantics are manipulated by introducing perturbations. However, they often results in substantial semantic distortions in the denoised output and suffers from low efficiency. In this study, we propose a novel framework called Semantic-Consistent Unrestricted Adversarial Attacks (SCA), which employs an inversion method to extract edit-friendly noise maps and utilizes Multimodal Large Language Model (MLLM) to provide semantic guidance throughout the process. Under the condition of rich semantic information provided by MLLM, we perform the DDPM denoising process of each step using a series of edit-friendly noise maps, and leverage DPM Solver++ to accelerate this process, enabling efficient sampling with semantic consistency. Compared to existing methods, our framework enables the efficient generation of adversarial examples that exhibit minimal discernible semantic changes. Consequently, we for the first time introduce Semantic-Consistent Adversarial Examples (SCAE). Extensive experiments and visualizations have demonstrated the high efficiency of SCA, particularly in being on average 12 times faster than the state-of-the-art attacks. Our research can further draw attention to the security of multimedia information.

摘要: 部署在敏感环境中的基于深度神经网络的系统容易受到敌意攻击。不受限制的对抗性攻击通常操纵图像的语义内容(例如，颜色或纹理)以创建既有效又逼真的对抗性示例。最近的工作利用扩散逆过程将图像映射到潜在空间，在潜在空间中通过引入扰动来操纵高级语义。然而，它们往往会在去噪输出中造成严重的语义扭曲，并导致效率低下。在这项研究中，我们提出了一种新的框架，称为语义一致的无限对抗攻击(SCA)，它使用一种反转方法来提取编辑友好的噪声映射，并利用多模式大语言模型(MLLM)在整个过程中提供语义指导。在MLLM提供丰富语义信息的条件下，使用一系列编辑友好的噪声图对每个步骤进行DDPM去噪处理，并利用DPM Solver++加速这一过程，从而实现高效的语义一致性采样。与现有的方法相比，我们的框架能够高效地生成对抗性的例子，这些例子表现出最小的可识别的语义变化。因此，我们首次引入了语义一致的对抗性例子(SCAE)。广泛的实验和可视化已经证明了SCA的高效率，特别是在平均速度上是最先进的攻击的12倍。我们的研究可以进一步引起人们对多媒体信息安全的关注。



## **14. Guide for Defense (G4D): Dynamic Guidance for Robust and Balanced Defense in Large Language Models**

防御指南（G4 D）：大型语言模型中稳健和平衡防御的动态指南 cs.AI

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.17922v1) [paper-pdf](http://arxiv.org/pdf/2410.17922v1)

**Authors**: He Cao, Weidi Luo, Yu Wang, Zijing Liu, Bing Feng, Yuan Yao, Yu Li

**Abstract**: With the extensive deployment of Large Language Models (LLMs), ensuring their safety has become increasingly critical. However, existing defense methods often struggle with two key issues: (i) inadequate defense capabilities, particularly in domain-specific scenarios like chemistry, where a lack of specialized knowledge can lead to the generation of harmful responses to malicious queries. (ii) over-defensiveness, which compromises the general utility and responsiveness of LLMs. To mitigate these issues, we introduce a multi-agents-based defense framework, Guide for Defense (G4D), which leverages accurate external information to provide an unbiased summary of user intentions and analytically grounded safety response guidance. Extensive experiments on popular jailbreak attacks and benign datasets show that our G4D can enhance LLM's robustness against jailbreak attacks on general and domain-specific scenarios without compromising the model's general functionality.

摘要: 随着大型语言模型（LLM）的广泛部署，确保其安全性变得越来越重要。然而，现有的防御方法经常遇到两个关键问题：（i）防御能力不足，特别是在化学等特定领域的场景中，缺乏专业知识可能会导致对恶意查询产生有害响应。(ii)过度防御，这会损害LLM的一般实用性和响应能力。为了缓解这些问题，我们引入了一个基于多代理的防御框架--防御指南（G4 D），该框架利用准确的外部信息来提供用户意图的公正摘要和基于分析的安全响应指南。对流行越狱攻击和良性数据集的广泛实验表明，我们的G4 D可以增强LLM针对一般和特定领域场景的越狱攻击的鲁棒性，而不会损害模型的一般功能。



## **15. IBGP: Imperfect Byzantine Generals Problem for Zero-Shot Robustness in Communicative Multi-Agent Systems**

IBGP：通信多智能体系统中零攻击鲁棒性的不完美拜占庭将军问题 cs.MA

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.16237v2) [paper-pdf](http://arxiv.org/pdf/2410.16237v2)

**Authors**: Yihuan Mao, Yipeng Kang, Peilun Li, Ning Zhang, Wei Xu, Chongjie Zhang

**Abstract**: As large language model (LLM) agents increasingly integrate into our infrastructure, their robust coordination and message synchronization become vital. The Byzantine Generals Problem (BGP) is a critical model for constructing resilient multi-agent systems (MAS) under adversarial attacks. It describes a scenario where malicious agents with unknown identities exist in the system-situations that, in our context, could result from LLM agents' hallucinations or external attacks. In BGP, the objective of the entire system is to reach a consensus on the action to be taken. Traditional BGP requires global consensus among all agents; however, in practical scenarios, global consensus is not always necessary and can even be inefficient. Therefore, there is a pressing need to explore a refined version of BGP that aligns with the local coordination patterns observed in MAS. We refer to this refined version as Imperfect BGP (IBGP) in our research, aiming to address this discrepancy. To tackle this issue, we propose a framework that leverages consensus protocols within general MAS settings, providing provable resilience against communication attacks and adaptability to changing environments, as validated by empirical results. Additionally, we present a case study in a sensor network environment to illustrate the practical application of our protocol.

摘要: 随着大型语言模型(LLM)代理越来越多地集成到我们的基础设施中，它们强大的协调和消息同步变得至关重要。拜占庭将军问题(BGP)是在对抗攻击下构造具有弹性的多智能体系统(MAS)的重要模型。它描述了一种系统中存在身份未知的恶意代理的情况--在我们的上下文中，这种情况可能是由于LLM代理的幻觉或外部攻击造成的。在BGP中，整个系统的目标是就要采取的行动达成共识。传统的BGP需要在所有代理之间达成全局共识；然而，在实际场景中，全局共识并不总是必要的，甚至可能效率低下。因此，迫切需要探索一种与MAS中观察到的局部协调模式相一致的BGP改进版本。在我们的研究中，我们将这种精炼版本称为不完美BGP(IBGP)，旨在解决这一差异。为了解决这个问题，我们提出了一个框架，它在一般的MAS环境中利用共识协议，提供对通信攻击的可证明的弹性和对不断变化的环境的适应性，实验结果验证了这一点。此外，我们还给出了一个传感器网络环境下的案例研究，以说明该协议的实际应用。



## **16. ConfusedPilot: Confused Deputy Risks in RAG-based LLMs**

困惑的飞行员：基于RAG的LLM中令人困惑的代理风险 cs.CR

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2408.04870v5) [paper-pdf](http://arxiv.org/pdf/2408.04870v5)

**Authors**: Ayush RoyChowdhury, Mulong Luo, Prateek Sahu, Sarbartha Banerjee, Mohit Tiwari

**Abstract**: Retrieval augmented generation (RAG) is a process where a large language model (LLM) retrieves useful information from a database and then generates the responses. It is becoming popular in enterprise settings for daily business operations. For example, Copilot for Microsoft 365 has accumulated millions of businesses. However, the security implications of adopting such RAG-based systems are unclear.   In this paper, we introduce ConfusedPilot, a class of security vulnerabilities of RAG systems that confuse Copilot and cause integrity and confidentiality violations in its responses. First, we investigate a vulnerability that embeds malicious text in the modified prompt in RAG, corrupting the responses generated by the LLM. Second, we demonstrate a vulnerability that leaks secret data, which leverages the caching mechanism during retrieval. Third, we investigate how both vulnerabilities can be exploited to propagate misinformation within the enterprise and ultimately impact its operations, such as sales and manufacturing. We also discuss the root cause of these attacks by investigating the architecture of a RAG-based system. This study highlights the security vulnerabilities in today's RAG-based systems and proposes design guidelines to secure future RAG-based systems.

摘要: 检索增强生成(RAG)是大型语言模型(LLM)从数据库中检索有用信息然后生成响应的过程。它在用于日常业务操作的企业环境中变得流行起来。例如，微软365的Copilot已经积累了数百万笔业务。然而，采用这种基于RAG的系统的安全影响尚不清楚。在本文中，我们介绍了一类RAG系统的安全漏洞ConfusedPilot，它迷惑了Copilot，并在其响应中导致完整性和保密性违规。首先，我们调查了一个漏洞，该漏洞将恶意文本嵌入到RAG中修改的提示符中，破坏了LLM生成的响应。其次，我们演示了一个泄漏机密数据的漏洞，该漏洞在检索过程中利用缓存机制。第三，我们调查如何利用这两个漏洞在企业内部传播错误信息，并最终影响其运营，如销售和制造。我们还通过研究基于RAG的系统的体系结构来讨论这些攻击的根本原因。这项研究强调了当今基于RAG的系统中的安全漏洞，并提出了保护未来基于RAG的系统的设计指南。



## **17. When "Competency" in Reasoning Opens the Door to Vulnerability: Jailbreaking LLMs via Novel Complex Ciphers**

当推理中的“能力”打开脆弱之门：通过新颖复杂密码越狱LLM cs.CL

14 pages, 7 figures

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2402.10601v2) [paper-pdf](http://arxiv.org/pdf/2402.10601v2)

**Authors**: Divij Handa, Zehua Zhang, Amir Saeidi, Chitta Baral

**Abstract**: Recent advancements in the safety of Large Language Models (LLMs) have primarily focused on mitigating attacks crafted in natural language or in common encryption techniques like Base64. However, new models which often possess better reasoning capabilities, open the door to new attack vectors that were previously non-existent in older models. This seems counter-intuitive at first glance, but these advanced models can decipher more complex cryptic queries that previous models could not, making them susceptible to attacks using such prompts. To exploit this vulnerability, we propose Attacks using Custom Encryptions (ACE), a novel method to jailbreak LLMs by leveraging custom encryption schemes. We evaluate the effectiveness of ACE on four state-of-the-art LLMs, achieving Attack Success Rates (ASR) of up to 66% on close-source models and 88% on open-source models. Building upon this, we introduce Layered Attacks using Custom Encryptions (LACE), which employs multiple layers of encryption through our custom ciphers to further enhance the ASR. Our findings demonstrate that LACE significantly enhances the ability to jailbreak LLMs, increasing the ASR of GPT-4o from 40% to 78%, a 38% improvement. Our results highlight that the advanced capabilities of LLMs introduce unforeseen vulnerabilities to complex attacks. Specifically complex and layered ciphers increase the chance of jailbreaking.

摘要: 大型语言模型(LLM)安全方面的最新进展主要集中在减轻用自然语言或常见加密技术(如Base64)编写的攻击。然而，新模型往往具有更好的推理能力，为以前在旧模型中不存在的新攻击矢量打开了大门。乍一看，这似乎有违直觉，但这些高级模型可以破译更复杂的神秘查询，而以前的模型无法破译，这使得它们容易受到使用此类提示的攻击。为了利用这一漏洞，我们提出了使用自定义加密(ACE)的攻击，这是一种通过利用自定义加密方案来越狱LLM的新方法。我们评估了ACE在四种最先进的LLM上的有效性，在封闭源代码模型上实现了高达66%的攻击成功率(ASR)，在开源模型上实现了88%的攻击成功率。在此基础上，我们引入了使用自定义加密(LACE)的分层攻击，它通过我们的自定义密码使用多层加密来进一步增强ASR。我们的研究结果表明，Lace显著增强了越狱LLMS的能力，将GPT-40的ASR从40%提高到78%，提高了38%。我们的结果突出表明，LLMS的高级能力为复杂攻击带来了不可预见的漏洞。特别是，复杂和分层的密码增加了越狱的机会。



## **18. Learning to Poison Large Language Models During Instruction Tuning**

学习在指令调优期间毒害大型语言模型 cs.LG

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2402.13459v2) [paper-pdf](http://arxiv.org/pdf/2402.13459v2)

**Authors**: Yao Qiang, Xiangyu Zhou, Saleh Zare Zade, Mohammad Amin Roshani, Prashant Khanduri, Douglas Zytko, Dongxiao Zhu

**Abstract**: The advent of Large Language Models (LLMs) has marked significant achievements in language processing and reasoning capabilities. Despite their advancements, LLMs face vulnerabilities to data poisoning attacks, where adversaries insert backdoor triggers into training data to manipulate outputs for malicious purposes. This work further identifies additional security risks in LLMs by designing a new data poisoning attack tailored to exploit the instruction tuning process. We propose a novel gradient-guided backdoor trigger learning (GBTL) algorithm to identify adversarial triggers efficiently, ensuring an evasion of detection by conventional defenses while maintaining content integrity. Through experimental validation across various tasks, including sentiment analysis, domain generation, and question answering, our poisoning strategy demonstrates a high success rate in compromising various LLMs' outputs. We further propose two defense strategies against data poisoning attacks, including in-context learning (ICL) and continuous learning (CL), which effectively rectify the behavior of LLMs and significantly reduce the decline in performance. Our work highlights the significant security risks present during the instruction tuning of LLMs and emphasizes the necessity of safeguarding LLMs against data poisoning attacks.

摘要: 大型语言模型的出现在语言处理和推理能力方面取得了显著的成就。尽管取得了进步，但LLM仍面临数据中毒攻击的漏洞，即对手在训练数据中插入后门触发器，以恶意目的操纵输出。这项工作通过设计一种新的数据中毒攻击来进一步识别LLMS中的额外安全风险，该攻击专为利用指令调优过程而定制。我们提出了一种新的梯度引导后门触发学习(GBTL)算法来高效地识别敌意触发，在保证内容完整性的同时确保了传统防御的检测。通过对各种任务的实验验证，包括情感分析、领域生成和问题回答，我们的中毒策略在牺牲各种LLMS的输出方面表现出了很高的成功率。针对数据中毒攻击，我们进一步提出了两种防御策略，包括上下文中学习(ICL)和连续学习(CL)，它们有效地纠正了LLM的行为，显著降低了性能下降。我们的工作突出了在LLMS的指令调优过程中存在的重大安全风险，并强调了保护LLMS免受数据中毒攻击的必要性。



## **19. Context-aware Prompt Tuning: Advancing In-Context Learning with Adversarial Methods**

上下文感知即时调优：用对抗方法推进上下文学习 cs.CL

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2410.17222v1) [paper-pdf](http://arxiv.org/pdf/2410.17222v1)

**Authors**: Tsachi Blau, Moshe Kimhi, Yonatan Belinkov, Alexander Bronstein, Chaim Baskin

**Abstract**: Fine-tuning Large Language Models (LLMs) typically involves updating at least a few billions of parameters. A more parameter-efficient approach is Prompt Tuning (PT), which updates only a few learnable tokens, and differently, In-Context Learning (ICL) adapts the model to a new task by simply including examples in the input without any training. When applying optimization-based methods, such as fine-tuning and PT for few-shot learning, the model is specifically adapted to the small set of training examples, whereas ICL leaves the model unchanged. This distinction makes traditional learning methods more prone to overfitting; in contrast, ICL is less sensitive to the few-shot scenario. While ICL is not prone to overfitting, it does not fully extract the information that exists in the training examples. This work introduces Context-aware Prompt Tuning (CPT), a method inspired by ICL, PT, and adversarial attacks. We build on the ICL strategy of concatenating examples before the input, but we extend this by PT-like learning, refining the context embedding through iterative optimization to extract deeper insights from the training examples. We carefully modify specific context tokens, considering the unique structure of input and output formats. Inspired by adversarial attacks, we adjust the input based on the labels present in the context, focusing on minimizing, rather than maximizing, the loss. Moreover, we apply a projected gradient descent algorithm to keep token embeddings close to their original values, under the assumption that the user-provided data is inherently valuable. Our method has been shown to achieve superior accuracy across multiple classification tasks using various LLM models.

摘要: 微调大型语言模型(LLM)通常需要更新至少数十亿个参数。一种参数效率更高的方法是即时调整(PT)，它只更新几个可学习的令牌，而不同的是，上下文中学习(ICL)通过在输入中简单地包括示例来使模型适应新任务，而不需要任何训练。当应用基于优化的方法时，例如微调和PT用于少镜头学习，该模型特别适合于小的训练样本集，而ICL保持模型不变。这种区别使得传统的学习方法更容易过度适应；相比之下，ICL对少数几次机会的情景不那么敏感。虽然ICL不容易过度拟合，但它没有完全提取训练示例中存在的信息。这项工作引入了上下文感知提示调优(CPT)，这是一种受ICL、PT和对手攻击启发的方法。我们建立在输入之前连接示例的ICL策略之上，但我们通过类似PT的学习来扩展这一策略，通过迭代优化来优化上下文嵌入，以从训练示例中提取更深层次的见解。考虑到输入和输出格式的独特结构，我们仔细修改了特定的上下文令牌。受到对抗性攻击的启发，我们根据上下文中存在的标签调整输入，重点是最小化而不是最大化损失。此外，在假设用户提供的数据具有内在价值的前提下，我们应用投影梯度下降算法来保持令牌嵌入接近其原始值。我们的方法已经被证明在使用各种LLM模型的多个分类任务中获得了更高的准确率。



## **20. AppPoet: Large Language Model based Android malware detection via multi-view prompt engineering**

AppPoet：通过多视图提示工程进行基于大语言模型的Android恶意软件检测 cs.CR

Accepted by Expert Systems With Applications

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2404.18816v3) [paper-pdf](http://arxiv.org/pdf/2404.18816v3)

**Authors**: Wenxiang Zhao, Juntao Wu, Zhaoyi Meng

**Abstract**: Due to the vast array of Android applications, their multifarious functions and intricate behavioral semantics, attackers can adopt various tactics to conceal their genuine attack intentions within legitimate functions. However, numerous learning-based methods suffer from a limitation in mining behavioral semantic information, thus impeding the accuracy and efficiency of Android malware detection. Besides, the majority of existing learning-based methods are weakly interpretive and fail to furnish researchers with effective and readable detection reports. Inspired by the success of the Large Language Models (LLMs) in natural language understanding, we propose AppPoet, a LLM-assisted multi-view system for Android malware detection. Firstly, AppPoet employs a static method to comprehensively collect application features and formulate various observation views. Then, using our carefully crafted multi-view prompt templates, it guides the LLM to generate function descriptions and behavioral summaries for each view, enabling deep semantic analysis of the views. Finally, we collaboratively fuse the multi-view information to efficiently and accurately detect malware through a deep neural network (DNN) classifier and then generate the human-readable diagnostic reports. Experimental results demonstrate that our method achieves a detection accuracy of 97.15% and an F1 score of 97.21%, which is superior to the baseline methods. Furthermore, the case study evaluates the effectiveness of our generated diagnostic reports.

摘要: 由于Android应用种类繁多，功能多样，行为语义错综复杂，攻击者可以采取各种策略，将真实的攻击意图隐藏在合法的功能中。然而，许多基于学习的方法在挖掘行为语义信息方面存在局限性，从而阻碍了Android恶意软件检测的准确性和效率。此外，现有的基于学习的方法大多解释性较弱，不能为研究人员提供有效的、可读性强的检测报告。受大语言模型在自然语言理解方面的成功启发，我们提出了一种基于大语言模型的Android恶意软件检测系统AppPoet。首先，AppPoet使用静态的方法来全面收集应用程序的特征，并制定各种观察视图。然后，使用我们精心设计的多视图提示模板，它指导LLM为每个视图生成功能描述和行为摘要，从而实现对视图的深入语义分析。最后，通过深度神经网络(DNN)分类器对多视图信息进行协同融合，高效准确地检测出恶意软件，并生成人类可读的诊断报告。实验结果表明，该方法的检测正确率为97.15%，F1评分为97.21%，优于基线方法。此外，案例研究还评估了我们生成的诊断报告的有效性。



## **21. Arabic Dataset for LLM Safeguard Evaluation**

LLM保障评估的阿拉伯数据集 cs.CL

17 pages, 6 figures, 10 tables

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2410.17040v1) [paper-pdf](http://arxiv.org/pdf/2410.17040v1)

**Authors**: Yasser Ashraf, Yuxia Wang, Bin Gu, Preslav Nakov, Timothy Baldwin

**Abstract**: The growing use of large language models (LLMs) has raised concerns regarding their safety. While many studies have focused on English, the safety of LLMs in Arabic, with its linguistic and cultural complexities, remains under-explored. Here, we aim to bridge this gap. In particular, we present an Arab-region-specific safety evaluation dataset consisting of 5,799 questions, including direct attacks, indirect attacks, and harmless requests with sensitive words, adapted to reflect the socio-cultural context of the Arab world. To uncover the impact of different stances in handling sensitive and controversial topics, we propose a dual-perspective evaluation framework. It assesses the LLM responses from both governmental and opposition viewpoints. Experiments over five leading Arabic-centric and multilingual LLMs reveal substantial disparities in their safety performance. This reinforces the need for culturally specific datasets to ensure the responsible deployment of LLMs.

摘要: 大型语言模型（LLM）的越来越多的使用引发了人们对其安全性的担忧。虽然许多研究都集中在英语上，但阿拉伯语法学硕士的安全性及其语言和文化复杂性仍然没有得到充分的探讨。在这里，我们的目标是弥合这一差距。特别是，我们提供了一个特定于阿拉伯地区的安全评估数据集，由5，799个问题组成，包括直接攻击、间接攻击和带有敏感词的无害请求，经过调整以反映阿拉伯世界的社会文化背景。为了揭示不同立场对处理敏感和争议话题的影响，我们提出了一个双视角评估框架。它从政府和反对派的角度评估了法学硕士的回应。对五个领先的以阿拉伯语为中心的多语言LLM的实验揭示了它们的安全性能存在巨大差异。这强化了对特定文化数据集的需求，以确保负责任地部署LLM。



## **22. Typos that Broke the RAG's Back: Genetic Attack on RAG Pipeline by Simulating Documents in the Wild via Low-level Perturbations**

让RAG筋疲力尽的错别字：通过低水平扰动模拟野外文档对RAG管道进行基因攻击 cs.CL

Findings of EMNLP Camera-ready version

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2404.13948v2) [paper-pdf](http://arxiv.org/pdf/2404.13948v2)

**Authors**: Sukmin Cho, Soyeong Jeong, Jeongyeon Seo, Taeho Hwang, Jong C. Park

**Abstract**: The robustness of recent Large Language Models (LLMs) has become increasingly crucial as their applicability expands across various domains and real-world applications. Retrieval-Augmented Generation (RAG) is a promising solution for addressing the limitations of LLMs, yet existing studies on the robustness of RAG often overlook the interconnected relationships between RAG components or the potential threats prevalent in real-world databases, such as minor textual errors. In this work, we investigate two underexplored aspects when assessing the robustness of RAG: 1) vulnerability to noisy documents through low-level perturbations and 2) a holistic evaluation of RAG robustness. Furthermore, we introduce a novel attack method, the Genetic Attack on RAG (\textit{GARAG}), which targets these aspects. Specifically, GARAG is designed to reveal vulnerabilities within each component and test the overall system functionality against noisy documents. We validate RAG robustness by applying our \textit{GARAG} to standard QA datasets, incorporating diverse retrievers and LLMs. The experimental results show that GARAG consistently achieves high attack success rates. Also, it significantly devastates the performance of each component and their synergy, highlighting the substantial risk that minor textual inaccuracies pose in disrupting RAG systems in the real world.

摘要: 最近的大型语言模型(LLM)的健壮性已经变得越来越重要，因为它们的适用性在各个领域和现实世界的应用程序中扩展。检索-增强生成(RAG)是解决LLMS局限性的一种很有前途的解决方案，但现有的RAG健壮性研究往往忽略了RAG组件之间的相互关联关系或现实世界数据库中普遍存在的潜在威胁，如微小的文本错误。在这项工作中，我们研究了两个在评估RAG稳健性时未被探索的方面：1)通过低层扰动对噪声文档的脆弱性；2)RAG稳健性的整体评估。此外，我们还介绍了一种针对这些方面的新的攻击方法--对RAG的遗传攻击(\textit{garag})。具体地说，Garag旨在揭示每个组件中的漏洞，并针对嘈杂的文档测试整个系统功能。我们通过将我们的\textit{garag}应用到标准的QA数据集来验证RAG的健壮性，其中包含了不同的检索器和LLM。实验结果表明，GARAG算法始终具有较高的攻击成功率。此外，它还严重破坏了每个组件的性能及其协同作用，突显了微小的文本错误在扰乱现实世界中的RAG系统方面构成的巨大风险。



## **23. Breaking ReAct Agents: Foot-in-the-Door Attack Will Get You In**

突破ReAct代理：脚入门攻击会让你进去 cs.CR

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2410.16950v1) [paper-pdf](http://arxiv.org/pdf/2410.16950v1)

**Authors**: Itay Nakash, George Kour, Guy Uziel, Ateret Anaby-Tavor

**Abstract**: Following the advancement of large language models (LLMs), the development of LLM-based autonomous agents has become increasingly prevalent. As a result, the need to understand the security vulnerabilities of these agents has become a critical task. We examine how ReAct agents can be exploited using a straightforward yet effective method we refer to as the foot-in-the-door attack. Our experiments show that indirect prompt injection attacks, prompted by harmless and unrelated requests (such as basic calculations) can significantly increase the likelihood of the agent performing subsequent malicious actions. Our results show that once a ReAct agents thought includes a specific tool or action, the likelihood of executing this tool in the subsequent steps increases significantly, as the agent seldom re-evaluates its actions. Consequently, even random, harmless requests can establish a foot-in-the-door, allowing an attacker to embed malicious instructions into the agents thought process, making it more susceptible to harmful directives. To mitigate this vulnerability, we propose implementing a simple reflection mechanism that prompts the agent to reassess the safety of its actions during execution, which can help reduce the success of such attacks.

摘要: 随着大型语言模型(LLM)的提出，基于LLM的自治代理的开发日益普遍。因此，需要了解这些代理的安全漏洞就成为一项关键任务。我们研究了如何使用一种直接但有效的方法来利用反应代理，我们称之为入门攻击。我们的实验表明，由无害和无关的请求(如基本计算)提示的间接提示注入攻击可以显著增加代理执行后续恶意操作的可能性。我们的结果表明，一旦反应代理的想法包括特定的工具或操作，在后续步骤中执行此工具的可能性显著增加，因为代理很少重新评估其操作。因此，即使是随机的、无害的请求也可能建立入门攻击，允许攻击者在代理的思维过程中嵌入恶意指令，使其更容易受到有害指令的影响。为了缓解这一漏洞，我们建议实现一种简单的反射机制，提示代理在执行过程中重新评估其操作的安全性，这有助于减少此类攻击的成功率。



## **24. RePD: Defending Jailbreak Attack through a Retrieval-based Prompt Decomposition Process**

RePD：通过基于检索的即时分解过程防御越狱攻击 cs.CR

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2410.08660v2) [paper-pdf](http://arxiv.org/pdf/2410.08660v2)

**Authors**: Peiran Wang, Xiaogeng Liu, Chaowei Xiao

**Abstract**: In this study, we introduce RePD, an innovative attack Retrieval-based Prompt Decomposition framework designed to mitigate the risk of jailbreak attacks on large language models (LLMs). Despite rigorous pretraining and finetuning focused on ethical alignment, LLMs are still susceptible to jailbreak exploits. RePD operates on a one-shot learning model, wherein it accesses a database of pre-collected jailbreak prompt templates to identify and decompose harmful inquiries embedded within user prompts. This process involves integrating the decomposition of the jailbreak prompt into the user's original query into a one-shot learning example to effectively teach the LLM to discern and separate malicious components. Consequently, the LLM is equipped to first neutralize any potentially harmful elements before addressing the user's prompt in a manner that aligns with its ethical guidelines. RePD is versatile and compatible with a variety of open-source LLMs acting as agents. Through comprehensive experimentation with both harmful and benign prompts, we have demonstrated the efficacy of our proposed RePD in enhancing the resilience of LLMs against jailbreak attacks, without compromising their performance in responding to typical user requests.

摘要: 在这项研究中，我们介绍了RePD，一个创新的基于攻击检索的提示分解框架，旨在降低对大型语言模型(LLM)的越狱攻击风险。尽管严格的预训和微调侧重于道德一致性，但LLM仍然容易受到越狱利用的影响。RePD运行在一次性学习模式上，其中它访问预先收集的越狱提示模板数据库，以识别和分解嵌入用户提示中的有害查询。这一过程包括将越狱提示的分解集成到用户的原始查询中，并将其整合为一个一次性学习示例，以有效地教会LLM识别和分离恶意组件。因此，LLM配备了首先中和任何潜在有害元素，然后以符合其道德准则的方式处理用户的提示。RePD是通用的，并与各种作为代理的开源LLM兼容。通过对有害提示和良性提示的全面实验，我们已经证明了我们提出的RePD在增强LLM对越狱攻击的弹性方面的有效性，而不会影响它们响应典型用户请求的性能。



## **25. $\textit{MMJ-Bench}$: A Comprehensive Study on Jailbreak Attacks and Defenses for Multimodal Large Language Models**

$\texttit {MMJ-Bench}$：多模式大型语言模型越狱攻击和防御的综合研究 cs.CR

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2408.08464v4) [paper-pdf](http://arxiv.org/pdf/2408.08464v4)

**Authors**: Fenghua Weng, Yue Xu, Chengyan Fu, Wenjie Wang

**Abstract**: As deep learning advances, Large Language Models (LLMs) and their multimodal counterparts, Multimodal Large Language Models (MLLMs), have shown exceptional performance in many real-world tasks. However, MLLMs face significant security challenges, such as jailbreak attacks, where attackers attempt to bypass the model's safety alignment to elicit harmful responses. The threat of jailbreak attacks on MLLMs arises from both the inherent vulnerabilities of LLMs and the multiple information channels that MLLMs process. While various attacks and defenses have been proposed, there is a notable gap in unified and comprehensive evaluations, as each method is evaluated on different dataset and metrics, making it impossible to compare the effectiveness of each method. To address this gap, we introduce \textit{MMJ-Bench}, a unified pipeline for evaluating jailbreak attacks and defense techniques for MLLMs. Through extensive experiments, we assess the effectiveness of various attack methods against SoTA MLLMs and evaluate the impact of defense mechanisms on both defense effectiveness and model utility for normal tasks. Our comprehensive evaluation contribute to the field by offering a unified and systematic evaluation framework and the first public-available benchmark for MLLM jailbreak research. We also demonstrate several insightful findings that highlights directions for future studies.

摘要: 随着深度学习的深入，大型语言模型(LLM)及其对应的多通道大型语言模型(MLLMS)在许多实际任务中表现出了优异的性能。然而，MLLMS面临着重大的安全挑战，例如越狱攻击，攻击者试图绕过该模型的安全对齐，以引发有害的反应。越狱攻击对MLLMS的威胁既来自LLMS固有的脆弱性，也源于MLLMS处理的多种信息渠道。虽然已经提出了各种攻击和防御方法，但在统一和综合评估方面存在显著差距，因为每种方法都是在不同的数据集和指标上进行评估，因此无法比较每种方法的有效性。为了弥补这一差距，我们引入了一个统一的管道，用于评估越狱攻击和MLLMS防御技术。通过大量的实验，我们评估了各种攻击方法对Sota MLLMS的攻击效果，并评估了防御机制对正常任务的防御效果和模型效用的影响。我们的全面评价通过提供统一和系统的评价框架和第一个公开可供公众使用的基准，为MLLM越狱研究做出了贡献。我们还展示了几个有洞察力的发现，这些发现突出了未来研究的方向。



## **26. Imprompter: Tricking LLM Agents into Improper Tool Use**

入侵者：诱骗LLM代理人使用不当工具 cs.CR

website: https://imprompter.ai code:  https://github.com/Reapor-Yurnero/imprompter v2 changelog: add new results to  Table 3, correct several typos

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2410.14923v2) [paper-pdf](http://arxiv.org/pdf/2410.14923v2)

**Authors**: Xiaohan Fu, Shuheng Li, Zihan Wang, Yihao Liu, Rajesh K. Gupta, Taylor Berg-Kirkpatrick, Earlence Fernandes

**Abstract**: Large Language Model (LLM) Agents are an emerging computing paradigm that blends generative machine learning with tools such as code interpreters, web browsing, email, and more generally, external resources. These agent-based systems represent an emerging shift in personal computing. We contribute to the security foundations of agent-based systems and surface a new class of automatically computed obfuscated adversarial prompt attacks that violate the confidentiality and integrity of user resources connected to an LLM agent. We show how prompt optimization techniques can find such prompts automatically given the weights of a model. We demonstrate that such attacks transfer to production-level agents. For example, we show an information exfiltration attack on Mistral's LeChat agent that analyzes a user's conversation, picks out personally identifiable information, and formats it into a valid markdown command that results in leaking that data to the attacker's server. This attack shows a nearly 80% success rate in an end-to-end evaluation. We conduct a range of experiments to characterize the efficacy of these attacks and find that they reliably work on emerging agent-based systems like Mistral's LeChat, ChatGLM, and Meta's Llama. These attacks are multimodal, and we show variants in the text-only and image domains.

摘要: 大型语言模型(LLM)代理是一种新兴的计算范例，它将生成式机器学习与代码解释器、Web浏览、电子邮件以及更一般的外部资源等工具相结合。这些基于代理的系统代表着个人计算领域正在发生的转变。我们为基于代理的系统的安全基础做出了贡献，并提出了一类新的自动计算的混淆对抗性提示攻击，这些攻击违反了连接到LLM代理的用户资源的机密性和完整性。我们展示了提示优化技术如何在给定模型权重的情况下自动找到这样的提示。我们证明了这种攻击会转移到生产级代理。例如，我们展示了对Mistral的Lechat代理的信息外泄攻击，该攻击分析用户的对话，挑选出个人身份信息，并将其格式化为有效的标记命令，从而导致该数据泄漏到攻击者的服务器。该攻击在端到端评估中显示了近80%的成功率。我们进行了一系列实验来表征这些攻击的有效性，并发现它们在新兴的基于代理的系统上可靠地工作，如Mistral的Lechat、ChatGLM和Meta的Llama。这些攻击是多模式的，我们展示了纯文本和图像领域的变体。



## **27. Insights and Current Gaps in Open-Source LLM Vulnerability Scanners: A Comparative Analysis**

开源LLM漏洞扫描仪的见解和当前差距：比较分析 cs.CR

15 pages, 11 figures

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.16527v1) [paper-pdf](http://arxiv.org/pdf/2410.16527v1)

**Authors**: Jonathan Brokman, Omer Hofman, Oren Rachmil, Inderjeet Singh, Rathina Sabapathy, Aishvariya Priya, Vikas Pahuja, Amit Giloni, Roman Vainshtein, Hisashi Kojima

**Abstract**: This report presents a comparative analysis of open-source vulnerability scanners for conversational large language models (LLMs). As LLMs become integral to various applications, they also present potential attack surfaces, exposed to security risks such as information leakage and jailbreak attacks. Our study evaluates prominent scanners - Garak, Giskard, PyRIT, and CyberSecEval - that adapt red-teaming practices to expose these vulnerabilities. We detail the distinctive features and practical use of these scanners, outline unifying principles of their design and perform quantitative evaluations to compare them. These evaluations uncover significant reliability issues in detecting successful attacks, highlighting a fundamental gap for future development. Additionally, we contribute a preliminary labelled dataset, which serves as an initial step to bridge this gap. Based on the above, we provide strategic recommendations to assist organizations choose the most suitable scanner for their red-teaming needs, accounting for customizability, test suite comprehensiveness, and industry-specific use cases.

摘要: 本报告对用于会话大型语言模型(LLM)的开源漏洞扫描器进行了比较分析。随着LLM成为各种应用的组成部分，它们也出现了潜在的攻击面，暴露在信息泄露和越狱攻击等安全风险中。我们的研究评估了采用红色团队实践来暴露这些漏洞的著名扫描仪-Garak、Giskard、PyRIT和CyberSecEval。我们详细介绍了这些扫描仪的特点和实际应用，概述了它们设计的统一原则，并进行了定量评估以进行比较。这些评估揭示了检测成功攻击的重大可靠性问题，突显了未来发展的根本差距。此外，我们提供了一个初步的标记数据集，这是弥合这一差距的第一步。在此基础上，我们提供了战略性建议，以帮助组织选择最适合其红团队需求的扫描仪，考虑到可定制性、测试套件的全面性和行业特定的用例。



## **28. Refusal-Trained LLMs Are Easily Jailbroken As Browser Agents**

接受过专家培训的LLM作为浏览器代理很容易越狱 cs.CR

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.13886v2) [paper-pdf](http://arxiv.org/pdf/2410.13886v2)

**Authors**: Priyanshu Kumar, Elaine Lau, Saranya Vijayakumar, Tu Trinh, Scale Red Team, Elaine Chang, Vaughn Robinson, Sean Hendryx, Shuyan Zhou, Matt Fredrikson, Summer Yue, Zifan Wang

**Abstract**: For safety reasons, large language models (LLMs) are trained to refuse harmful user instructions, such as assisting dangerous activities. We study an open question in this work: does the desired safety refusal, typically enforced in chat contexts, generalize to non-chat and agentic use cases? Unlike chatbots, LLM agents equipped with general-purpose tools, such as web browsers and mobile devices, can directly influence the real world, making it even more crucial to refuse harmful instructions. In this work, we primarily focus on red-teaming browser agents, LLMs that manipulate information via web browsers. To this end, we introduce Browser Agent Red teaming Toolkit (BrowserART), a comprehensive test suite designed specifically for red-teaming browser agents. BrowserART is consist of 100 diverse browser-related harmful behaviors (including original behaviors and ones sourced from HarmBench [Mazeika et al., 2024] and AirBench 2024 [Zeng et al., 2024b]) across both synthetic and real websites. Our empirical study on state-of-the-art browser agents reveals that, while the backbone LLM refuses harmful instructions as a chatbot, the corresponding agent does not. Moreover, attack methods designed to jailbreak refusal-trained LLMs in the chat settings transfer effectively to browser agents. With human rewrites, GPT-4o and o1-preview-based browser agents attempted 98 and 63 harmful behaviors (out of 100), respectively. We publicly release BrowserART and call on LLM developers, policymakers, and agent developers to collaborate on improving agent safety

摘要: 出于安全原因，大型语言模型(LLM)会接受培训，以拒绝有害的用户指令，例如协助危险活动。我们在这项工作中研究了一个悬而未决的问题：通常在聊天环境中强制执行的所需安全拒绝是否推广到非聊天和代理用例？与聊天机器人不同，配备了网络浏览器和移动设备等通用工具的LLM代理可以直接影响现实世界，使拒绝有害指令变得更加关键。在这项工作中，我们主要关注红团队浏览器代理，即通过Web浏览器处理信息的LLM。为此，我们引入了浏览器代理红团队工具包(BrowserART)，这是一个专门为红团队浏览器代理设计的全面测试套件。BrowserART由100种不同的与浏览器相关的有害行为(包括原始行为和源自HarmBtch[Mazeika等人，2024]和AirBitch2024[Zeng等人，2024b])组成，涵盖了合成网站和真实网站。我们对最先进的浏览器代理的经验研究表明，虽然骨干LLM拒绝作为聊天机器人的有害指令，但相应的代理不会。此外，设计用于越狱拒绝训练聊天设置中的LLM的攻击方法有效地转移到浏览器代理。在人工重写的情况下，基于GPT-4o和o1预览的浏览器代理分别尝试了98和63种有害行为(满分100分)。我们公开发布BrowserART，并呼吁LLM开发人员、政策制定者和代理开发人员合作提高代理安全



## **29. A Realistic Threat Model for Large Language Model Jailbreaks**

大型语言模型越狱的现实威胁模型 cs.LG

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.16222v1) [paper-pdf](http://arxiv.org/pdf/2410.16222v1)

**Authors**: Valentyn Boreiko, Alexander Panfilov, Vaclav Voracek, Matthias Hein, Jonas Geiping

**Abstract**: A plethora of jailbreaking attacks have been proposed to obtain harmful responses from safety-tuned LLMs. In their original settings, these methods all largely succeed in coercing the target output, but their attacks vary substantially in fluency and computational effort. In this work, we propose a unified threat model for the principled comparison of these methods. Our threat model combines constraints in perplexity, measuring how far a jailbreak deviates from natural text, and computational budget, in total FLOPs. For the former, we build an N-gram model on 1T tokens, which, in contrast to model-based perplexity, allows for an LLM-agnostic and inherently interpretable evaluation. We adapt popular attacks to this new, realistic threat model, with which we, for the first time, benchmark these attacks on equal footing. After a rigorous comparison, we not only find attack success rates against safety-tuned modern models to be lower than previously presented but also find that attacks based on discrete optimization significantly outperform recent LLM-based attacks. Being inherently interpretable, our threat model allows for a comprehensive analysis and comparison of jailbreak attacks. We find that effective attacks exploit and abuse infrequent N-grams, either selecting N-grams absent from real-world text or rare ones, e.g. specific to code datasets.

摘要: 已经提出了过多的越狱攻击，以获得经过安全调整的LLM的有害反应。在最初的设置中，这些方法都在很大程度上成功地胁迫了目标输出，但它们的攻击在流畅性和计算工作量方面存在很大差异。在这项工作中，我们提出了一个统一的威胁模型来对这些方法进行原则性的比较。我们的威胁模型结合了困惑中的约束，衡量越狱与自然文本的偏离程度，以及计算预算，总失败。对于前者，我们在1T标记上构建了一个N元语法模型，与基于模型的困惑不同，它允许LLM不可知的和内在可解释的评估。我们使流行的攻击适应这种新的、现实的威胁模型，我们第一次在平等的基础上对这些攻击进行基准测试。经过严格的比较，我们不仅发现针对安全调整的现代模型的攻击成功率低于先前提出的模型，而且发现基于离散优化的攻击显著优于最近的基于LLM的攻击。由于本质上是可解释的，我们的威胁模型允许对越狱攻击进行全面分析和比较。我们发现，有效的攻击利用和滥用不常见的N-gram，或者选择现实世界文本中不存在的N-gram，或者选择罕见的，例如特定于代码数据集的N-gram。



## **30. Harmful Fine-tuning Attacks and Defenses for Large Language Models: A Survey**

针对大型语言模型的有害微调攻击和防御：调查 cs.CR

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2409.18169v3) [paper-pdf](http://arxiv.org/pdf/2409.18169v3)

**Authors**: Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, Ling Liu

**Abstract**: Recent research demonstrates that the nascent fine-tuning-as-a-service business model exposes serious safety concerns -- fine-tuning over a few harmful data uploaded by the users can compromise the safety alignment of the model. The attack, known as harmful fine-tuning, has raised a broad research interest among the community. However, as the attack is still new, \textbf{we observe from our miserable submission experience that there are general misunderstandings within the research community.} We in this paper aim to clear some common concerns for the attack setting, and formally establish the research problem. Specifically, we first present the threat model of the problem, and introduce the harmful fine-tuning attack and its variants. Then we systematically survey the existing literature on attacks/defenses/mechanical analysis of the problem. Finally, we outline future research directions that might contribute to the development of the field. Additionally, we present a list of questions of interest, which might be useful to refer to when reviewers in the peer review process question the realism of the experiment/attack/defense setting. A curated list of relevant papers is maintained and made accessible at: \url{https://github.com/git-disl/awesome_LLM-harmful-fine-tuning-papers}.

摘要: 最近的研究表明，新兴的微调即服务商业模式暴露了严重的安全问题--对用户上传的几个有害数据进行微调可能会损害该模型的安全一致性。这一被称为有害微调的攻击在社区中引起了广泛的研究兴趣。然而，由于攻击仍然是新的，\extbf{我们从悲惨的提交经验中观察到，研究界普遍存在误解。}我们在本文中旨在澄清一些对攻击设置的共同关注，并正式确立研究问题。具体地说，我们首先给出了问题的威胁模型，并介绍了有害的微调攻击及其变体。然后，我们系统地综述了现有的关于攻击/防御/机械分析问题的文献。最后，我们概述了未来的研究方向，可能有助于该领域的发展。此外，我们提供了一个感兴趣的问题列表，当同行审查过程中的评审者质疑实验/攻击/防御设置的真实性时，这些问题可能会有用。相关论文的精选清单可在以下网址查阅：\url{https://github.com/git-disl/awesome_LLM-harmful-fine-tuning-papers}.



## **31. A Troublemaker with Contagious Jailbreak Makes Chaos in Honest Towns**

具有传染性的越狱麻烦制造者扰乱诚实城镇 cs.CL

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.16155v1) [paper-pdf](http://arxiv.org/pdf/2410.16155v1)

**Authors**: Tianyi Men, Pengfei Cao, Zhuoran Jin, Yubo Chen, Kang Liu, Jun Zhao

**Abstract**: With the development of large language models, they are widely used as agents in various fields. A key component of agents is memory, which stores vital information but is susceptible to jailbreak attacks. Existing research mainly focuses on single-agent attacks and shared memory attacks. However, real-world scenarios often involve independent memory. In this paper, we propose the Troublemaker Makes Chaos in Honest Town (TMCHT) task, a large-scale, multi-agent, multi-topology text-based attack evaluation framework. TMCHT involves one attacker agent attempting to mislead an entire society of agents. We identify two major challenges in multi-agent attacks: (1) Non-complete graph structure, (2) Large-scale systems. We attribute these challenges to a phenomenon we term toxicity disappearing. To address these issues, we propose an Adversarial Replication Contagious Jailbreak (ARCJ) method, which optimizes the retrieval suffix to make poisoned samples more easily retrieved and optimizes the replication suffix to make poisoned samples have contagious ability. We demonstrate the superiority of our approach in TMCHT, with 23.51%, 18.95%, and 52.93% improvements in line topology, star topology, and 100-agent settings. Encourage community attention to the security of multi-agent systems.

摘要: 随着大型语言模型的发展，它们作为智能体被广泛应用于各个领域。代理的一个关键组件是内存，它存储重要信息，但容易受到越狱攻击。现有的研究主要集中在单代理攻击和共享内存攻击上。然而，现实世界中的场景通常涉及独立的内存。本文提出了Troublemaker Make Chaos in Honest town(TMCHT)任务，这是一个大规模、多代理、多拓扑、基于文本的攻击评估框架。TMCHT涉及一个攻击者代理试图误导整个代理社会。我们确定了多智能体攻击中的两个主要挑战：(1)非完全图结构，(2)大规模系统。我们将这些挑战归因于一种我们称之为毒性消失的现象。针对这些问题，我们提出了一种对抗性复制传染越狱(ARCJ)方法，通过优化检索后缀使中毒样本更容易检索，并优化复制后缀使中毒样本具有传染能力。我们在TMCHT中展示了我们的方法的优越性，在线路拓扑、星形拓扑和100-代理设置方面分别有23.51%、18.95%和52.93%的改进。鼓励社会各界关注多智能体系统的安全性。



## **32. Extracting Spatiotemporal Data from Gradients with Large Language Models**

使用大型语言模型从对象中提取时空数据 cs.LG

arXiv admin note: substantial text overlap with arXiv:2407.08529

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.16121v1) [paper-pdf](http://arxiv.org/pdf/2410.16121v1)

**Authors**: Lele Zheng, Yang Cao, Renhe Jiang, Kenjiro Taura, Yulong Shen, Sheng Li, Masatoshi Yoshikawa

**Abstract**: Recent works show that sensitive user data can be reconstructed from gradient updates, breaking the key privacy promise of federated learning. While success was demonstrated primarily on image data, these methods do not directly transfer to other domains, such as spatiotemporal data. To understand privacy risks in spatiotemporal federated learning, we first propose Spatiotemporal Gradient Inversion Attack (ST-GIA), a gradient attack algorithm tailored to spatiotemporal data that successfully reconstructs the original location from gradients. Furthermore, the absence of priors in attacks on spatiotemporal data has hindered the accurate reconstruction of real client data. To address this limitation, we propose ST-GIA+, which utilizes an auxiliary language model to guide the search for potential locations, thereby successfully reconstructing the original data from gradients. In addition, we design an adaptive defense strategy to mitigate gradient inversion attacks in spatiotemporal federated learning. By dynamically adjusting the perturbation levels, we can offer tailored protection for varying rounds of training data, thereby achieving a better trade-off between privacy and utility than current state-of-the-art methods. Through intensive experimental analysis on three real-world datasets, we reveal that the proposed defense strategy can well preserve the utility of spatiotemporal federated learning with effective security protection.

摘要: 最近的研究表明，敏感的用户数据可以从梯度更新中重建，打破了联邦学习的关键隐私承诺。虽然这些方法的成功主要体现在图像数据上，但这些方法并不直接转移到其他领域，如时空数据。为了了解时空联合学习中的隐私风险，我们首先提出了时空梯度反转攻击(ST-GIA)，这是一种针对时空数据定制的梯度攻击算法，能够成功地从梯度重建原始位置。此外，在对时空数据的攻击中缺乏先验知识，阻碍了对真实客户数据的准确重建。为了解决这一局限性，我们提出了ST-GIA+，它利用辅助语言模型来指导潜在位置的搜索，从而成功地从梯度重建原始数据。此外，我们还设计了一种自适应防御策略来缓解时空联合学习中的梯度反转攻击。通过动态调整扰动级别，我们可以为不同轮的训练数据提供量身定制的保护，从而比目前最先进的方法在隐私和效用之间实现更好的权衡。通过对三个真实数据集的密集实验分析，我们发现所提出的防御策略能够在有效的安全保护下很好地保护时空联合学习的效用。



## **33. NetSafe: Exploring the Topological Safety of Multi-agent Networks**

NetSafe：探索多代理网络的布局安全 cs.MA

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.15686v1) [paper-pdf](http://arxiv.org/pdf/2410.15686v1)

**Authors**: Miao Yu, Shilong Wang, Guibin Zhang, Junyuan Mao, Chenlong Yin, Qijiong Liu, Qingsong Wen, Kun Wang, Yang Wang

**Abstract**: Large language models (LLMs) have empowered nodes within multi-agent networks with intelligence, showing growing applications in both academia and industry. However, how to prevent these networks from generating malicious information remains unexplored with previous research on single LLM's safety be challenging to transfer. In this paper, we focus on the safety of multi-agent networks from a topological perspective, investigating which topological properties contribute to safer networks. To this end, we propose a general framework, NetSafe along with an iterative RelCom interaction to unify existing diverse LLM-based agent frameworks, laying the foundation for generalized topological safety research. We identify several critical phenomena when multi-agent networks are exposed to attacks involving misinformation, bias, and harmful information, termed as Agent Hallucination and Aggregation Safety. Furthermore, we find that highly connected networks are more susceptible to the spread of adversarial attacks, with task performance in a Star Graph Topology decreasing by 29.7%. Besides, our proposed static metrics aligned more closely with real-world dynamic evaluations than traditional graph-theoretic metrics, indicating that networks with greater average distances from attackers exhibit enhanced safety. In conclusion, our work introduces a new topological perspective on the safety of LLM-based multi-agent networks and discovers several unreported phenomena, paving the way for future research to explore the safety of such networks.

摘要: 大型语言模型(LLM)已经为多代理网络中的节点赋予了智能，显示出在学术界和工业中日益增长的应用。然而，如何防止这些网络产生恶意信息还没有被探索，以往关于单个LLM的安全传输的研究是具有挑战性的。本文从拓扑学的角度研究了多智能体网络的安全性，研究了哪些拓扑性质有助于网络的安全。为此，我们提出了一个通用的框架NetSafe以及一个迭代的RelCom交互来统一现有的各种基于LLM的代理框架，为广义拓扑安全研究奠定了基础。我们确定了当多智能体网络暴露于涉及错误信息、偏见和有害信息的攻击时的几个关键现象，称为智能体幻觉和聚集安全。此外，我们发现，高连接网络更容易受到敌意攻击的传播，星图拓扑中的任务性能下降了29.7%。此外，我们提出的静态度量比传统的图论度量更接近真实世界的动态评估，这表明离攻击者的平均距离越大的网络表现出更高的安全性。综上所述，我们的工作为基于LLM的多智能体网络的安全性引入了一种新的拓扑观，并发现了一些未被报道的现象，为进一步研究此类网络的安全性铺平了道路。



## **34. Boosting Jailbreak Transferability for Large Language Models**

提高大型语言模型的越狱可移植性 cs.AI

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.15645v1) [paper-pdf](http://arxiv.org/pdf/2410.15645v1)

**Authors**: Hanqing Liu, Lifeng Zhou, Huanqian Yan

**Abstract**: Large language models have drawn significant attention to the challenge of safe alignment, especially regarding jailbreak attacks that circumvent security measures to produce harmful content. To address the limitations of existing methods like GCG, which perform well in single-model attacks but lack transferability, we propose several enhancements, including a scenario induction template, optimized suffix selection, and the integration of re-suffix attack mechanism to reduce inconsistent outputs. Our approach has shown superior performance in extensive experiments across various benchmarks, achieving nearly 100% success rates in both attack execution and transferability. Notably, our method has won the online first place in the AISG-hosted Global Challenge for Safe and Secure LLMs.

摘要: 大型语言模型引起了人们对安全对齐挑战的高度关注，特别是对于绕过安全措施以产生有害内容的越狱攻击。为了解决GCG等现有方法在单模型攻击中表现良好但缺乏可移植性的局限性，我们提出了几项增强措施，包括场景归纳模板、优化的后缀选择以及集成重新后缀攻击机制以减少不一致的输出。我们的方法在各种基准测试的广泛实验中表现出卓越的性能，在攻击执行和可转移性方面都实现了近100%的成功率。值得注意的是，我们的方法在AISG主办的全球安全LLM挑战赛中赢得了在线第一名。



## **35. SMILES-Prompting: A Novel Approach to LLM Jailbreak Attacks in Chemical Synthesis**

SMILES-起诉：化学合成中LLM越狱攻击的一种新方法 cs.CL

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.15641v1) [paper-pdf](http://arxiv.org/pdf/2410.15641v1)

**Authors**: Aidan Wong, He Cao, Zijing Liu, Yu Li

**Abstract**: The increasing integration of large language models (LLMs) across various fields has heightened concerns about their potential to propagate dangerous information. This paper specifically explores the security vulnerabilities of LLMs within the field of chemistry, particularly their capacity to provide instructions for synthesizing hazardous substances. We evaluate the effectiveness of several prompt injection attack methods, including red-teaming, explicit prompting, and implicit prompting. Additionally, we introduce a novel attack technique named SMILES-prompting, which uses the Simplified Molecular-Input Line-Entry System (SMILES) to reference chemical substances. Our findings reveal that SMILES-prompting can effectively bypass current safety mechanisms. These findings highlight the urgent need for enhanced domain-specific safeguards in LLMs to prevent misuse and improve their potential for positive social impact.

摘要: 各个领域的大型语言模型（LLM）日益集成加剧了人们对它们传播危险信息的潜力的担忧。本文专门探讨了化学领域LLM的安全漏洞，特别是它们提供合成危险物质说明的能力。我们评估了几种即时注入攻击方法的有效性，包括红组、显式提示和隐式提示。此外，我们还引入了一种名为SMILES提示的新型攻击技术，该技术使用简化分子输入线路输入系统（SMILES）来引用化学物质。我们的研究结果表明，微笑提示可以有效地绕过当前的安全机制。这些研究结果凸显了迫切需要加强LLM特定领域的保障措施，以防止滥用并提高其产生积极社会影响的潜力。



## **36. Revisit, Extend, and Enhance Hessian-Free Influence Functions**

重新审视、扩展和增强无黑森影响力功能 cs.LG

**SubmitDate**: 2024-10-20    [abs](http://arxiv.org/abs/2405.17490v2) [paper-pdf](http://arxiv.org/pdf/2405.17490v2)

**Authors**: Ziao Yang, Han Yue, Jian Chen, Hongfu Liu

**Abstract**: Influence functions serve as crucial tools for assessing sample influence in model interpretation, subset training set selection, noisy label detection, and more. By employing the first-order Taylor extension, influence functions can estimate sample influence without the need for expensive model retraining. However, applying influence functions directly to deep models presents challenges, primarily due to the non-convex nature of the loss function and the large size of model parameters. This difficulty not only makes computing the inverse of the Hessian matrix costly but also renders it non-existent in some cases. Various approaches, including matrix decomposition, have been explored to expedite and approximate the inversion of the Hessian matrix, with the aim of making influence functions applicable to deep models. In this paper, we revisit a specific, albeit naive, yet effective approximation method known as TracIn. This method substitutes the inverse of the Hessian matrix with an identity matrix. We provide deeper insights into why this simple approximation method performs well. Furthermore, we extend its applications beyond measuring model utility to include considerations of fairness and robustness. Finally, we enhance TracIn through an ensemble strategy. To validate its effectiveness, we conduct experiments on synthetic data and extensive evaluations on noisy label detection, sample selection for large language model fine-tuning, and defense against adversarial attacks.

摘要: 影响函数在模型解释、子集训练集选择、噪声标签检测等方面用作评估样本影响的重要工具。通过使用一阶泰勒扩展，影响函数可以估计样本影响，而不需要昂贵的模型重新训练。然而，直接将影响函数应用于深层模型会带来挑战，这主要是由于损失函数的非凸性和模型参数的大尺寸。这一困难不仅使计算海森矩阵的逆的成本高昂，而且在某些情况下使其不存在。已经探索了各种方法，包括矩阵分解，以加快和近似海森矩阵的求逆，目的是使影响函数适用于深层模式。在这篇文章中，我们回顾了一种特定的，尽管很幼稚，但有效的近似方法，称为TracIn。该方法用单位矩阵代替海森矩阵的逆。我们对为什么这种简单的近似方法表现良好提供了更深层次的见解。此外，我们将它的应用扩展到测量模型效用之外，包括对公平性和稳健性的考虑。最后，我们通过集成策略增强了TracIn。为了验证其有效性，我们在合成数据上进行了实验，并在噪声标签检测、大语言模型微调的样本选择以及对对手攻击的防御方面进行了广泛的评估。



## **37. The Best Defense is a Good Offense: Countering LLM-Powered Cyberattacks**

最好的防御就是好的进攻：对抗LLM支持的网络攻击 cs.CR

**SubmitDate**: 2024-10-20    [abs](http://arxiv.org/abs/2410.15396v1) [paper-pdf](http://arxiv.org/pdf/2410.15396v1)

**Authors**: Daniel Ayzenshteyn, Roy Weiss, Yisroel Mirsky

**Abstract**: As large language models (LLMs) continue to evolve, their potential use in automating cyberattacks becomes increasingly likely. With capabilities such as reconnaissance, exploitation, and command execution, LLMs could soon become integral to autonomous cyber agents, capable of launching highly sophisticated attacks. In this paper, we introduce novel defense strategies that exploit the inherent vulnerabilities of attacking LLMs. By targeting weaknesses such as biases, trust in input, memory limitations, and their tunnel-vision approach to problem-solving, we develop techniques to mislead, delay, or neutralize these autonomous agents. We evaluate our defenses under black-box conditions, starting with single prompt-response scenarios and progressing to real-world tests using custom-built CTF machines. Our results show defense success rates of up to 90\%, demonstrating the effectiveness of turning LLM vulnerabilities into defensive strategies against LLM-driven cyber threats.

摘要: 随着大型语言模型（LLM）的不断发展，它们在自动化网络攻击方面的潜在用途变得越来越有可能。凭借侦察、利用和命令执行等能力，LLM很快就会成为自主网络代理的组成部分，能够发起高度复杂的攻击。在本文中，我们介绍了利用攻击LLM的固有漏洞的新型防御策略。通过针对偏见、对输入的信任、记忆限制及其解决问题的隧道视觉方法等弱点，我们开发了误导、延迟或中和这些自主主体的技术。我们在黑匣子条件下评估我们的防御，从单一预算响应场景开始，并使用定制的CTF机器进行现实世界的测试。我们的结果显示防御成功率高达90%，证明了将LLM漏洞转化为针对LLM驱动的网络威胁的防御策略的有效性。



## **38. Faster-GCG: Efficient Discrete Optimization Jailbreak Attacks against Aligned Large Language Models**

Faster-GCG：针对对齐大型语言模型的高效离散优化越狱攻击 cs.LG

**SubmitDate**: 2024-10-20    [abs](http://arxiv.org/abs/2410.15362v1) [paper-pdf](http://arxiv.org/pdf/2410.15362v1)

**Authors**: Xiao Li, Zhuhong Li, Qiongxiu Li, Bingze Lee, Jinghao Cui, Xiaolin Hu

**Abstract**: Aligned Large Language Models (LLMs) have demonstrated remarkable performance across various tasks. However, LLMs remain susceptible to jailbreak adversarial attacks, where adversaries manipulate prompts to elicit malicious responses that aligned LLMs should have avoided. Identifying these vulnerabilities is crucial for understanding the inherent weaknesses of LLMs and preventing their potential misuse. One pioneering work in jailbreaking is the GCG attack, a discrete token optimization algorithm that seeks to find a suffix capable of jailbreaking aligned LLMs. Despite the success of GCG, we find it suboptimal, requiring significantly large computational costs, and the achieved jailbreaking performance is limited. In this work, we propose Faster-GCG, an efficient adversarial jailbreak method by delving deep into the design of GCG. Experiments demonstrate that Faster-GCG can surpass the original GCG with only 1/10 of the computational cost, achieving significantly higher attack success rates on various open-source aligned LLMs. In addition, We demonstrate that Faster-GCG exhibits improved attack transferability when testing on closed-sourced LLMs such as ChatGPT.

摘要: 经过调整的大型语言模型(LLM)在各种任务中表现出了卓越的性能。然而，LLM仍然容易受到越狱对手攻击，在这种攻击中，对手操纵提示来引发恶意响应，而结盟的LLM本应避免这种情况。识别这些漏洞对于了解LLM的固有弱点并防止其潜在的滥用至关重要。越狱方面的一项开创性工作是GCG攻击，这是一种离散令牌优化算法，旨在找到能够越狱对齐的LLM的后缀。尽管GCG取得了成功，但我们发现它不是最优的，需要相当大的计算成本，而且所取得的越狱性能是有限的。在这项工作中，我们通过深入研究GCG的设计，提出了一种高效的对抗性越狱方法FASTER-GCG。实验表明，FASTER-GCG的计算代价仅为原GCG的1/10，对各种开源对齐LLM的攻击成功率显著提高。此外，我们还证明了在ChatGPT等封闭源代码的LLM上测试时，FASTER-GCG表现出了更好的攻击可转移性。



## **39. Jailbreaking and Mitigation of Vulnerabilities in Large Language Models**

大型语言模型中的漏洞越狱和缓解 cs.CR

**SubmitDate**: 2024-10-20    [abs](http://arxiv.org/abs/2410.15236v1) [paper-pdf](http://arxiv.org/pdf/2410.15236v1)

**Authors**: Benji Peng, Ziqian Bi, Qian Niu, Ming Liu, Pohsun Feng, Tianyang Wang, Lawrence K. Q. Yan, Yizhu Wen, Yichao Zhang, Caitlyn Heqi Yin

**Abstract**: Large Language Models (LLMs) have transformed artificial intelligence by advancing natural language understanding and generation, enabling applications across fields beyond healthcare, software engineering, and conversational systems. Despite these advancements in the past few years, LLMs have shown considerable vulnerabilities, particularly to prompt injection and jailbreaking attacks. This review analyzes the state of research on these vulnerabilities and presents available defense strategies. We roughly categorize attack approaches into prompt-based, model-based, multimodal, and multilingual, covering techniques such as adversarial prompting, backdoor injections, and cross-modality exploits. We also review various defense mechanisms, including prompt filtering, transformation, alignment techniques, multi-agent defenses, and self-regulation, evaluating their strengths and shortcomings. We also discuss key metrics and benchmarks used to assess LLM safety and robustness, noting challenges like the quantification of attack success in interactive contexts and biases in existing datasets. Identifying current research gaps, we suggest future directions for resilient alignment strategies, advanced defenses against evolving attacks, automation of jailbreak detection, and consideration of ethical and societal impacts. This review emphasizes the need for continued research and cooperation within the AI community to enhance LLM security and ensure their safe deployment.

摘要: 大型语言模型(LLM)通过提高自然语言理解和生成，实现了医疗保健、软件工程和对话系统以外的各个领域的应用，从而改变了人工智能。尽管在过去几年中取得了这些进展，但LLMS已经显示出相当大的漏洞，特别是在快速注入和越狱攻击方面。本文对这些漏洞的研究现状进行了分析，并提出了可用的防御策略。我们大致将攻击方法分为基于提示的、基于模型的、多模式的和多语言的，包括对抗性提示、后门注入和跨模式利用等技术。我们还回顾了各种防御机制，包括快速过滤、转换、对齐技术、多代理防御和自我调节，并对它们的优缺点进行了评估。我们还讨论了用于评估LLM安全性和健壮性的关键指标和基准，指出了交互环境中攻击成功的量化和现有数据集中的偏差等挑战。找出目前的研究差距，我们建议在弹性调整战略、针对不断演变的攻击的高级防御、越狱检测的自动化以及对伦理和社会影响的考虑方面的未来方向。本审查强调需要在人工智能社区内继续研究与合作，以加强LLM的安全并确保其安全部署。



## **40. Securing Large Language Models: Addressing Bias, Misinformation, and Prompt Attacks**

保护大型语言模型：解决偏见、错误信息和即时攻击 cs.CR

17 pages, 1 figure

**SubmitDate**: 2024-10-19    [abs](http://arxiv.org/abs/2409.08087v2) [paper-pdf](http://arxiv.org/pdf/2409.08087v2)

**Authors**: Benji Peng, Keyu Chen, Ming Li, Pohsun Feng, Ziqian Bi, Junyu Liu, Qian Niu

**Abstract**: Large Language Models (LLMs) demonstrate impressive capabilities across various fields, yet their increasing use raises critical security concerns. This article reviews recent literature addressing key issues in LLM security, with a focus on accuracy, bias, content detection, and vulnerability to attacks. Issues related to inaccurate or misleading outputs from LLMs is discussed, with emphasis on the implementation from fact-checking methodologies to enhance response reliability. Inherent biases within LLMs are critically examined through diverse evaluation techniques, including controlled input studies and red teaming exercises. A comprehensive analysis of bias mitigation strategies is presented, including approaches from pre-processing interventions to in-training adjustments and post-processing refinements. The article also probes the complexity of distinguishing LLM-generated content from human-produced text, introducing detection mechanisms like DetectGPT and watermarking techniques while noting the limitations of machine learning enabled classifiers under intricate circumstances. Moreover, LLM vulnerabilities, including jailbreak attacks and prompt injection exploits, are analyzed by looking into different case studies and large-scale competitions like HackAPrompt. This review is concluded by retrospecting defense mechanisms to safeguard LLMs, accentuating the need for more extensive research into the LLM security field.

摘要: 大型语言模型(LLM)在各个领域展示了令人印象深刻的能力，但它们的日益使用引发了严重的安全问题。本文回顾了解决LLM安全关键问题的最新文献，重点讨论了准确性、偏差、内容检测和攻击漏洞。讨论了与LLMS输出不准确或误导性有关的问题，重点是从事实核查方法中实施以提高响应可靠性。通过不同的评估技术，包括受控投入研究和红色团队练习，对LLM中的固有偏见进行了严格的检查。对偏差缓解策略进行了全面的分析，包括从前处理干预到培训中调整和后处理改进的方法。文章还探讨了区分LLM生成的内容和人类生成的文本的复杂性，介绍了DetectGPT和水印技术等检测机制，同时指出了机器学习支持的分类器在复杂环境下的局限性。此外，通过研究不同的案例研究和HackAPrompt等大型竞争，分析了LLM漏洞，包括越狱攻击和快速注入利用。本综述通过回顾保护LLM的防御机制来结束，强调了对LLM安全领域进行更广泛研究的必要性。



## **41. Multi-round jailbreak attack on large language models**

对大型语言模型的多轮越狱攻击 cs.CL

It is not fully completed

**SubmitDate**: 2024-10-19    [abs](http://arxiv.org/abs/2410.11533v2) [paper-pdf](http://arxiv.org/pdf/2410.11533v2)

**Authors**: Yihua Zhou, Xiaochuan Shi

**Abstract**: Ensuring the safety and alignment of large language models (LLMs) with human values is crucial for generating responses that are beneficial to humanity. While LLMs have the capability to identify and avoid harmful queries, they remain vulnerable to "jailbreak" attacks, where carefully crafted prompts can induce the generation of toxic content. Traditional single-round jailbreak attacks, such as GCG and AutoDAN, do not alter the sensitive words in the dangerous prompts. Although they can temporarily bypass the model's safeguards through prompt engineering, their success rate drops significantly as the LLM is further fine-tuned, and they cannot effectively circumvent static rule-based filters that remove the hazardous vocabulary.   In this study, to better understand jailbreak attacks, we introduce a multi-round jailbreak approach. This method can rewrite the dangerous prompts, decomposing them into a series of less harmful sub-questions to bypass the LLM's safety checks. We first use the LLM to perform a decomposition task, breaking down a set of natural language questions into a sequence of progressive sub-questions, which are then used to fine-tune the Llama3-8B model, enabling it to decompose hazardous prompts. The fine-tuned model is then used to break down the problematic prompt, and the resulting sub-questions are sequentially asked to the victim model. If the victim model rejects a sub-question, a new decomposition is generated, and the process is repeated until the final objective is achieved. Our experimental results show a 94\% success rate on the llama2-7B and demonstrate the effectiveness of this approach in circumventing static rule-based filters.

摘要: 确保大型语言模型(LLM)的安全性并使其与人类价值观保持一致，对于产生有益于人类的反应至关重要。虽然LLM具有识别和避免有害查询的能力，但它们仍然容易受到“越狱”攻击，在这种攻击中，精心设计的提示可能会导致有毒内容的生成。传统的单轮越狱攻击，如GCG和AutoDAN，不会改变危险提示中的敏感词语。尽管他们可以通过快速工程暂时绕过模型的保障措施，但随着LLM的进一步微调，他们的成功率显著下降，而且他们无法有效地绕过删除危险词汇的静态规则过滤器。在这项研究中，为了更好地理解越狱攻击，我们引入了一种多轮越狱方法。这种方法可以重写危险的提示，将它们分解为一系列危害较小的子问题，以绕过LLM的安全检查。我们首先使用LLM执行分解任务，将一组自然语言问题分解为一系列递进子问题，然后使用这些子问题来微调Llama3-8B模型，使其能够分解危险提示。然后，使用微调的模型来分解有问题的提示，并将得到的子问题顺序地询问给受害者模型。如果受害者模型拒绝了子问题，则生成新的分解，并重复该过程，直到达到最终目标。实验结果表明，该方法在Llama2-7B上的检测成功率为94%，证明了该方法对静态规则过滤的有效性。



## **42. ASTPrompter: Weakly Supervised Automated Language Model Red-Teaming to Identify Likely Toxic Prompts**

ASTencer：弱监督的自动化语言模型红色团队识别可能有毒的候选人 cs.CL

8 pages, 8 pages of appendix, 2 tables, 3 figures

**SubmitDate**: 2024-10-18    [abs](http://arxiv.org/abs/2407.09447v2) [paper-pdf](http://arxiv.org/pdf/2407.09447v2)

**Authors**: Amelia F. Hardy, Houjun Liu, Bernard Lange, Mykel J. Kochenderfer

**Abstract**: Typical schemes for the automated red-teaming of large language models (LLMs) focus on discovering prompts that trigger a frozen language model (the defender) to generate toxic text. This often results in the prompting model (the adversary) producing text that is unintelligible and unlikely to arise. Here, we propose a reinforcement learning formulation of the LLM red-teaming task that allows us to discover prompts that both (1) trigger toxic outputs from a frozen defender and (2) have low perplexity as scored by that defender. We argue these cases are the most pertinent in a red-teaming setting because they are likely to arise during normal use of the defender model. We solve this formulation through a novel online and weakly supervised variant of Identity Preference Optimization (IPO) on GPT-2, GPT-2 XL, and TinyLlama defenders. We demonstrate that our policy is capable of generating likely (low-perplexity) prompts that also trigger toxicity from all of these architectures. Furthermore, we show that this policy outperforms baselines by producing attacks that are occur with higher probability and are more effective. Finally, we discuss our findings and the observed trade-offs between likelihood vs toxicity. Source code for this project is available for this project at: https://github.com/sisl/ASTPrompter/.

摘要: 大型语言模型(LLM)自动红团队的典型方案侧重于发现触发冻结语言模型(防御者)生成有毒文本的提示。这通常会导致提示模型(对手)生成无法理解且不太可能出现的文本。在这里，我们提出了一种LLM红队任务的强化学习公式，它允许我们发现(1)触发冻结防御者的有毒输出和(2)由该防御者得分的低困惑程度的提示。我们认为这些情况在红队环境中是最相关的，因为它们很可能在正常使用后卫模式时出现。我们通过对GPT-2、GPT-2XL和TinyLlama防御者的身份偏好优化(IPO)的一种新的在线弱监督变体来解决这个问题。我们演示了我们的策略能够生成可能的(低困惑)提示，这也会触发所有这些架构的毒性。此外，我们还证明了该策略通过产生更高概率和更有效的攻击来超越基线。最后，我们讨论了我们的发现和观察到的可能性和毒性之间的权衡。此项目的源代码可在以下网址获得：https://github.com/sisl/ASTPrompter/.



## **43. Feint and Attack: Attention-Based Strategies for Jailbreaking and Protecting LLMs**

假动作和攻击：越狱和保护LLM的基于注意力的策略 cs.CR

**SubmitDate**: 2024-10-18    [abs](http://arxiv.org/abs/2410.16327v1) [paper-pdf](http://arxiv.org/pdf/2410.16327v1)

**Authors**: Rui Pu, Chaozhuo Li, Rui Ha, Zejian Chen, Litian Zhang, Zheng Liu, Lirong Qiu, Xi Zhang

**Abstract**: Jailbreak attack can be used to access the vulnerabilities of Large Language Models (LLMs) by inducing LLMs to generate the harmful content. And the most common method of the attack is to construct semantically ambiguous prompts to confuse and mislead the LLMs. To access the security and reveal the intrinsic relation between the input prompt and the output for LLMs, the distribution of attention weight is introduced to analyze the underlying reasons. By using statistical analysis methods, some novel metrics are defined to better describe the distribution of attention weight, such as the Attention Intensity on Sensitive Words (Attn_SensWords), the Attention-based Contextual Dependency Score (Attn_DepScore) and Attention Dispersion Entropy (Attn_Entropy). By leveraging the distinct characteristics of these metrics, the beam search algorithm and inspired by the military strategy "Feint and Attack", an effective jailbreak attack strategy named as Attention-Based Attack (ABA) is proposed. In the ABA, nested attack prompts are employed to divert the attention distribution of the LLMs. In this manner, more harmless parts of the input can be used to attract the attention of the LLMs. In addition, motivated by ABA, an effective defense strategy called as Attention-Based Defense (ABD) is also put forward. Compared with ABA, the ABD can be used to enhance the robustness of LLMs by calibrating the attention distribution of the input prompt. Some comparative experiments have been given to demonstrate the effectiveness of ABA and ABD. Therefore, both ABA and ABD can be used to access the security of the LLMs. The comparative experiment results also give a logical explanation that the distribution of attention weight can bring great influence on the output for LLMs.

摘要: 越狱攻击可以通过诱导大型语言模型生成有害内容来访问大型语言模型的漏洞。而最常见的攻击方法是构建语义模糊的提示，以迷惑和误导LLM。为了获得LLMS的安全性，揭示LLMS输入提示和输出之间的内在联系，引入了注意力权重分布来分析其深层原因。通过使用统计分析方法，定义了一些新的度量来更好地描述注意力权重的分布，如敏感词上的注意力强度(Attn_SensWords)、基于注意力的上下文依赖分数(Attn_DepScore)和注意力分散熵(Attn_Entropy)。利用这些度量的不同特点，利用波束搜索算法，并受军事战略“伪装攻击”的启发，提出了一种有效的越狱攻击策略--基于注意力的攻击(ABA)。在ABA中，使用嵌套的攻击提示来转移LLMS的注意力分配。以这种方式，可以使用输入的更多无害部分来吸引LLM的注意。此外，在ABA的激励下，还提出了一种有效的防御策略，称为基于注意力的防御(ABD)。与ABA相比，ABD可以通过校准输入提示的注意力分布来增强LLMS的稳健性。通过对比实验验证了ABA算法和ABD算法的有效性。因此，ABA和ABD都可以用来访问LLMS的安全性。对比实验结果也合理地解释了注意权重的分布对LLMS的输出有很大的影响。



## **44. When LLMs Go Online: The Emerging Threat of Web-Enabled LLMs**

当LLM上线时：支持Web的LLM的新兴威胁 cs.CR

**SubmitDate**: 2024-10-18    [abs](http://arxiv.org/abs/2410.14569v1) [paper-pdf](http://arxiv.org/pdf/2410.14569v1)

**Authors**: Hanna Kim, Minkyoo Song, Seung Ho Na, Seungwon Shin, Kimin Lee

**Abstract**: Recent advancements in Large Language Models (LLMs) have established them as agentic systems capable of planning and interacting with various tools. These LLM agents are often paired with web-based tools, enabling access to diverse sources and real-time information. Although these advancements offer significant benefits across various applications, they also increase the risk of malicious use, particularly in cyberattacks involving personal information. In this work, we investigate the risks associated with misuse of LLM agents in cyberattacks involving personal data. Specifically, we aim to understand: 1) how potent LLM agents can be when directed to conduct cyberattacks, 2) how cyberattacks are enhanced by web-based tools, and 3) how affordable and easy it becomes to launch cyberattacks using LLM agents. We examine three attack scenarios: the collection of Personally Identifiable Information (PII), the generation of impersonation posts, and the creation of spear-phishing emails. Our experiments reveal the effectiveness of LLM agents in these attacks: LLM agents achieved a precision of up to 95.9% in collecting PII, up to 93.9% of impersonation posts created by LLM agents were evaluated as authentic, and the click rate for links in spear phishing emails created by LLM agents reached up to 46.67%. Additionally, our findings underscore the limitations of existing safeguards in contemporary commercial LLMs, emphasizing the urgent need for more robust security measures to prevent the misuse of LLM agents.

摘要: 大型语言模型(LLM)的最新进展已将它们确立为能够规划各种工具并与其交互的代理系统。这些LLM代理通常与基于Web的工具配合使用，从而能够访问不同的来源和实时信息。虽然这些改进在各种应用程序中提供了显著的好处，但它们也增加了恶意使用的风险，特别是在涉及个人信息的网络攻击中。在这项工作中，我们调查了在涉及个人数据的网络攻击中滥用LLM代理的相关风险。具体地说，我们的目标是了解：1)LLM代理在被指示进行网络攻击时的威力有多大；2)基于Web的工具如何增强网络攻击；3)使用LLM代理发起网络攻击变得多么负担得起和容易。我们研究了三种攻击场景：收集个人身份信息(PII)、生成模拟帖子和创建鱼叉式网络钓鱼电子邮件。我们的实验显示了LLM代理在这些攻击中的有效性：LLM代理收集PII的准确率高达95.9%，LLM代理创建的模仿帖子被评估为可信的高达93.9%，LLM代理创建的鱼叉式钓鱼邮件中链接的点击率高达46.67%。此外，我们的研究结果强调了当代商业LLM现有保障措施的局限性，强调迫切需要采取更强有力的安全措施，以防止滥用LLM剂。



## **45. BlackDAN: A Black-Box Multi-Objective Approach for Effective and Contextual Jailbreaking of Large Language Models**

BlackDAN：一种有效且上下文化的大型语言模型越狱的黑匣子多目标方法 cs.CR

**SubmitDate**: 2024-10-18    [abs](http://arxiv.org/abs/2410.09804v2) [paper-pdf](http://arxiv.org/pdf/2410.09804v2)

**Authors**: Xinyuan Wang, Victor Shea-Jay Huang, Renmiao Chen, Hao Wang, Chengwei Pan, Lei Sha, Minlie Huang

**Abstract**: While large language models (LLMs) exhibit remarkable capabilities across various tasks, they encounter potential security risks such as jailbreak attacks, which exploit vulnerabilities to bypass security measures and generate harmful outputs. Existing jailbreak strategies mainly focus on maximizing attack success rate (ASR), frequently neglecting other critical factors, including the relevance of the jailbreak response to the query and the level of stealthiness. This narrow focus on single objectives can result in ineffective attacks that either lack contextual relevance or are easily recognizable. In this work, we introduce BlackDAN, an innovative black-box attack framework with multi-objective optimization, aiming to generate high-quality prompts that effectively facilitate jailbreaking while maintaining contextual relevance and minimizing detectability. BlackDAN leverages Multiobjective Evolutionary Algorithms (MOEAs), specifically the NSGA-II algorithm, to optimize jailbreaks across multiple objectives including ASR, stealthiness, and semantic relevance. By integrating mechanisms like mutation, crossover, and Pareto-dominance, BlackDAN provides a transparent and interpretable process for generating jailbreaks. Furthermore, the framework allows customization based on user preferences, enabling the selection of prompts that balance harmfulness, relevance, and other factors. Experimental results demonstrate that BlackDAN outperforms traditional single-objective methods, yielding higher success rates and improved robustness across various LLMs and multimodal LLMs, while ensuring jailbreak responses are both relevant and less detectable.

摘要: 虽然大型语言模型(LLM)在各种任务中显示出非凡的能力，但它们遇到了潜在的安全风险，如越狱攻击，这些攻击利用漏洞绕过安全措施并产生有害的输出。现有的越狱策略主要关注最大化攻击成功率(ASR)，往往忽略了其他关键因素，包括越狱响应与查询的相关性和隐蔽性水平。这种对单一目标的狭隘关注可能会导致无效的攻击，要么缺乏上下文相关性，要么很容易识别。在这项工作中，我们引入了BlackDAN，一个创新的多目标优化的黑盒攻击框架，旨在生成高质量的提示，在保持上下文相关性的同时有效地促进越狱，并将可检测性降至最低。BlackDAN利用多目标进化算法(MOEA)，特别是NSGA-II算法，跨多个目标优化越狱，包括ASR、隐蔽性和语义相关性。通过集成变异、交叉和帕累托支配等机制，BlackDAN为生成越狱提供了一个透明和可解释的过程。此外，该框架允许根据用户偏好进行定制，从而能够选择在危害性、相关性和其他因素之间进行权衡的提示。实验结果表明，BlackDAN的性能优于传统的单目标方法，在各种LLM和多模式LLM上获得了更高的成功率和更好的鲁棒性，同时确保了越狱响应的相关性和较低的可检测性。



## **46. Backdoored Retrievers for Prompt Injection Attacks on Retrieval Augmented Generation of Large Language Models**

用于对大型语言模型的检索增强生成的提示注入攻击的后门检索器 cs.CR

12 pages, 5 figures

**SubmitDate**: 2024-10-18    [abs](http://arxiv.org/abs/2410.14479v1) [paper-pdf](http://arxiv.org/pdf/2410.14479v1)

**Authors**: Cody Clop, Yannick Teglia

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in generating coherent text but remain limited by the static nature of their training data. Retrieval Augmented Generation (RAG) addresses this issue by combining LLMs with up-to-date information retrieval, but also expand the attack surface of the system. This paper investigates prompt injection attacks on RAG, focusing on malicious objectives beyond misinformation, such as inserting harmful links, promoting unauthorized services, and initiating denial-of-service behaviors. We build upon existing corpus poisoning techniques and propose a novel backdoor attack aimed at the fine-tuning process of the dense retriever component. Our experiments reveal that corpus poisoning can achieve significant attack success rates through the injection of a small number of compromised documents into the retriever corpus. In contrast, backdoor attacks demonstrate even higher success rates but necessitate a more complex setup, as the victim must fine-tune the retriever using the attacker poisoned dataset.

摘要: 大型语言模型(LLM)在生成连贯的文本方面表现出非凡的能力，但仍然受到其训练数据静态性质的限制。检索增强生成(RAG)通过将LLMS与最新的信息检索相结合来解决这一问题，但也扩展了系统的攻击面。本文研究了RAG上的即时注入攻击，重点针对错误信息以外的恶意目标，如插入有害链接、推广未经授权的服务和发起拒绝服务行为。我们在现有的语料库中毒技术的基础上，针对密集检索组件的微调过程，提出了一种新的后门攻击。我们的实验表明，通过向检索器语料库中注入少量受攻击的文档，语料库中毒可以获得显着的攻击成功率。相比之下，后门攻击显示出更高的成功率，但需要更复杂的设置，因为受害者必须使用攻击者有毒的数据集来微调检索器。



## **47. Unlearning Backdoor Attacks for LLMs with Weak-to-Strong Knowledge Distillation**

通过弱到强的知识蒸馏消除LLM的后门攻击 cs.CL

**SubmitDate**: 2024-10-18    [abs](http://arxiv.org/abs/2410.14425v1) [paper-pdf](http://arxiv.org/pdf/2410.14425v1)

**Authors**: Shuai Zhao, Xiaobao Wu, Cong-Duy Nguyen, Meihuizi Jia, Yichao Feng, Luu Anh Tuan

**Abstract**: Parameter-efficient fine-tuning (PEFT) can bridge the gap between large language models (LLMs) and downstream tasks. However, PEFT has been proven vulnerable to malicious attacks. Research indicates that poisoned LLMs, even after PEFT, retain the capability to activate internalized backdoors when input samples contain predefined triggers. In this paper, we introduce a novel weak-to-strong unlearning algorithm to defend against backdoor attacks based on feature alignment knowledge distillation, named W2SDefense. Specifically, we first train a small-scale language model through full-parameter fine-tuning to serve as the clean teacher model. Then, this teacher model guides the large-scale poisoned student model in unlearning the backdoor, leveraging PEFT. Theoretical analysis suggests that W2SDefense has the potential to enhance the student model's ability to unlearn backdoor features, preventing the activation of the backdoor. We conduct experiments on text classification tasks involving three state-of-the-art language models and three different backdoor attack algorithms. Our empirical results demonstrate the outstanding performance of W2SDefense in defending against backdoor attacks without compromising model performance.

摘要: 参数高效微调(PEFT)可以弥合大型语言模型(LLM)和下游任务之间的差距。然而，PEFT已被证明容易受到恶意攻击。研究表明，中毒的LLM，即使在PEFT之后，当输入样本包含预定义的触发器时，仍保持激活内部化后门的能力。本文提出了一种新的基于特征对齐知识提取的弱到强遗忘算法W2SDefense来防御后门攻击。具体来说，我们首先通过全参数微调来训练一个小规模的语言模型，作为廉洁教师模型。然后，这个教师模型引导大规模中毒学生模型忘记后门，利用PEFT。理论分析表明，W2SDefense有可能增强学生模型忘记后门功能的能力，防止激活后门。我们对三种最新的语言模型和三种不同的后门攻击算法进行了文本分类实验。我们的实验结果表明，W2SDefense在不影响模型性能的情况下，在防御后门攻击方面具有出色的性能。



## **48. VLFeedback: A Large-Scale AI Feedback Dataset for Large Vision-Language Models Alignment**

VLFeedback：用于大型视觉语言模型对齐的大规模人工智能反馈数据集 cs.CV

EMNLP 2024 Main Conference camera-ready version (fixed small typos).  This article supersedes arXiv:2312.10665

**SubmitDate**: 2024-10-18    [abs](http://arxiv.org/abs/2410.09421v2) [paper-pdf](http://arxiv.org/pdf/2410.09421v2)

**Authors**: Lei Li, Zhihui Xie, Mukai Li, Shunian Chen, Peiyi Wang, Liang Chen, Yazheng Yang, Benyou Wang, Lingpeng Kong, Qi Liu

**Abstract**: As large vision-language models (LVLMs) evolve rapidly, the demand for high-quality and diverse data to align these models becomes increasingly crucial. However, the creation of such data with human supervision proves costly and time-intensive. In this paper, we investigate the efficacy of AI feedback to scale supervision for aligning LVLMs. We introduce VLFeedback, the first large-scale vision-language feedback dataset, comprising over 82K multi-modal instructions and comprehensive rationales generated by off-the-shelf models without human annotations. To evaluate the effectiveness of AI feedback for vision-language alignment, we train Silkie, an LVLM fine-tuned via direct preference optimization on VLFeedback. Silkie showcases exceptional performance regarding helpfulness, visual faithfulness, and safety metrics. It outperforms its base model by 6.9\% and 9.5\% in perception and cognition tasks, reduces hallucination issues on MMHal-Bench, and exhibits enhanced resilience against red-teaming attacks. Furthermore, our analysis underscores the advantage of AI feedback, particularly in fostering preference diversity to deliver more comprehensive improvements. Our dataset, training code and models are available at https://vlf-silkie.github.io.

摘要: 随着大型视觉语言模型(LVLM)的快速发展，对高质量和多样化数据的需求变得越来越重要。然而，事实证明，在人工监督下创建此类数据既昂贵又耗时。在本文中，我们研究了人工智能反馈对比例尺监督对准LVLM的有效性。我们介绍了VLFeedback，这是第一个大规模的视觉语言反馈数据集，包括超过82K的多模式指令和由没有人工注释的现成模型生成的全面原理。为了评估人工智能反馈对视觉-语言对齐的有效性，我们对Silkie进行了训练，这是一种通过对VLFeedback进行直接偏好优化而微调的LVLM。Silkie展示了在帮助、视觉忠诚度和安全指标方面的出色表现。它在感知和认知任务中的表现分别比基本模型高出6.9%和9.5%，减少了MMHal-BENCH上的幻觉问题，并表现出对红队攻击的增强的弹性。此外，我们的分析强调了人工智能反馈的优势，特别是在促进偏好多样性以提供更全面的改进方面。我们的数据集、训练代码和模型可在https://vlf-silkie.github.io.上获得



## **49. Make LLMs better zero-shot reasoners: Structure-orientated autonomous reasoning**

让LLM成为更好的零射推理机：面向结构的自主推理 cs.LG

**SubmitDate**: 2024-10-18    [abs](http://arxiv.org/abs/2410.19000v1) [paper-pdf](http://arxiv.org/pdf/2410.19000v1)

**Authors**: Pengfei He, Zitao Li, Yue Xing, Yaling Li, Jiliang Tang, Bolin Ding

**Abstract**: Zero-shot reasoning methods with Large Language Models (LLMs) offer significant advantages including great generalization to novel tasks and reduced dependency on human-crafted examples. However, the current zero-shot methods still have limitations in complex tasks, e.g., answering questions that require multi-step reasoning. In this paper, we address this limitation by introducing a novel structure-oriented analysis method to help LLMs better understand the question and guide the problem-solving process of LLMs. We first demonstrate how the existing reasoning strategies, Chain-of-Thought and ReAct, can benefit from our structure-oriented analysis. In addition to empirical investigations, we leverage the probabilistic graphical model to theoretically explain why our structure-oriented analysis can improve the LLM reasoning process. To further improve the reliability in complex question-answering tasks, we propose a multi-agent reasoning system, Structure-oriented Autonomous Reasoning Agents (SARA), that can better enforce the reasoning process following our structure-oriented analysis by refinement techniques and is equipped with external knowledge retrieval capability to reduce factual errors. Extensive experiments verify the effectiveness of the proposed reasoning system. Surprisingly, in some cases, the system even surpasses few-shot methods. Finally, the system not only improves reasoning accuracy in complex tasks but also demonstrates robustness against potential attacks that corrupt the reasoning process.

摘要: 基于大型语言模型的零命中推理方法具有显著的优势，包括对新任务的高度泛化和减少对人工制作的实例的依赖。然而，当前的零射击方法在复杂任务中仍然存在局限性，例如回答需要多步骤推理的问题。在本文中，我们通过引入一种新的面向结构的分析方法来解决这一局限性，以帮助LLMS更好地理解问题并指导LLMS的问题求解过程。我们首先演示现有的推理策略，即思想链和反应，如何从我们的面向结构的分析中受益。除了实证研究，我们还利用概率图形模型从理论上解释了为什么我们的面向结构的分析可以改进LLM推理过程。为了进一步提高复杂问答任务的可靠性，我们提出了一种多智能体推理系统--面向结构的自主推理智能体(SARA)，它能够通过精化技术更好地执行面向结构分析的推理过程，并具有外部知识检索能力来减少事实错误。大量实验验证了该推理系统的有效性。令人惊讶的是，在某些情况下，该系统甚至超过了少镜头方法。最后，该系统不仅提高了复杂任务中的推理精度，而且对破坏推理过程的潜在攻击表现出了健壮性。



## **50. DomainLynx: Leveraging Large Language Models for Enhanced Domain Squatting Detection**

DomainLynx：利用大型语言模型进行增强的域蹲位检测 cs.CR

Accepted for publication at IEEE CCNC 2025

**SubmitDate**: 2024-10-18    [abs](http://arxiv.org/abs/2410.02095v2) [paper-pdf](http://arxiv.org/pdf/2410.02095v2)

**Authors**: Daiki Chiba, Hiroki Nakano, Takashi Koide

**Abstract**: Domain squatting poses a significant threat to Internet security, with attackers employing increasingly sophisticated techniques. This study introduces DomainLynx, an innovative compound AI system leveraging Large Language Models (LLMs) for enhanced domain squatting detection. Unlike existing methods focusing on predefined patterns for top-ranked domains, DomainLynx excels in identifying novel squatting techniques and protecting less prominent brands. The system's architecture integrates advanced data processing, intelligent domain pairing, and LLM-powered threat assessment. Crucially, DomainLynx incorporates specialized components that mitigate LLM hallucinations, ensuring reliable and context-aware detection. This approach enables efficient analysis of vast security data from diverse sources, including Certificate Transparency logs, Passive DNS records, and zone files. Evaluated on a curated dataset of 1,649 squatting domains, DomainLynx achieved 94.7\% accuracy using Llama-3-70B. In a month-long real-world test, it detected 34,359 squatting domains from 2.09 million new domains, outperforming baseline methods by 2.5 times. This research advances Internet security by providing a versatile, accurate, and adaptable tool for combating evolving domain squatting threats. DomainLynx's approach paves the way for more robust, AI-driven cybersecurity solutions, enhancing protection for a broader range of online entities and contributing to a safer digital ecosystem.

摘要: 随着攻击者使用越来越复杂的技术，域名抢占对互联网安全构成了重大威胁。这项研究介绍了DomainLynx，这是一个创新的复合人工智能系统，利用大型语言模型(LLM)来增强域占用检测。与专注于排名靠前的域名的预定义模式的现有方法不同，DomainLynx在识别新颖的蹲守技术和保护不太知名的品牌方面表现出色。该系统的体系结构集成了先进的数据处理、智能域配对和LLM支持的威胁评估。至关重要的是，DomainLynx结合了专门的组件来缓解LLM幻觉，确保可靠和上下文感知的检测。这种方法可以有效地分析来自不同来源的大量安全数据，包括证书透明日志、被动DNS记录和区域文件。在1,649个蹲点域的精选数据集上进行评估，DomainLynx使用LLAMA-3-70B获得了94.7%的准确率。在一个月的实际测试中，它从209万个新域名中检测到34359个蹲点域名，比基线方法高出2.5倍。这项研究通过提供一种通用、准确和适应性强的工具来对抗不断变化的域名抢占威胁，从而促进了互联网安全。DomainLynx的方法为更强大的、人工智能驱动的网络安全解决方案铺平了道路，加强了对更广泛的在线实体的保护，并为更安全的数字生态系统做出了贡献。



