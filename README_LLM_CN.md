# Latest Large Language Model Attack Papers
**update at 2025-01-29 20:05:29**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Adversarial Vulnerabilities in Large Language Models for Time Series Forecasting**

时间序列预测大型语言模型中的对抗漏洞 cs.LG

AISTATS 2025

**SubmitDate**: 2025-01-28    [abs](http://arxiv.org/abs/2412.08099v3) [paper-pdf](http://arxiv.org/pdf/2412.08099v3)

**Authors**: Fuqiang Liu, Sicong Jiang, Luis Miranda-Moreno, Seongjin Choi, Lijun Sun

**Abstract**: Large Language Models (LLMs) have recently demonstrated significant potential in the field of time series forecasting, offering impressive capabilities in handling complex temporal data. However, their robustness and reliability in real-world applications remain under-explored, particularly concerning their susceptibility to adversarial attacks. In this paper, we introduce a targeted adversarial attack framework for LLM-based time series forecasting. By employing both gradient-free and black-box optimization methods, we generate minimal yet highly effective perturbations that significantly degrade the forecasting accuracy across multiple datasets and LLM architectures. Our experiments, which include models like TimeGPT and LLM-Time with GPT-3.5, GPT-4, LLaMa, and Mistral, show that adversarial attacks lead to much more severe performance degradation than random noise, and demonstrate the broad effectiveness of our attacks across different LLMs. The results underscore the critical vulnerabilities of LLMs in time series forecasting, highlighting the need for robust defense mechanisms to ensure their reliable deployment in practical applications.

摘要: 大型语言模型最近在时间序列预测领域显示出巨大的潜力，在处理复杂的时间数据方面提供了令人印象深刻的能力。然而，它们在实际应用中的健壮性和可靠性仍然没有得到充分的研究，特别是关于它们对对手攻击的敏感性。本文提出了一种基于LLM的时间序列预测的对抗性攻击框架。通过使用无梯度和黑盒优化方法，我们产生了最小但高效的扰动，这些扰动显著降低了跨多个数据集和LLM体系结构的预测精度。我们的实验，包括使用GPT-3.5、GPT-4、LLAMA和Mistral的TimeGPT和LLM-Time模型，表明对抗性攻击导致的性能降级比随机噪声严重得多，并证明了我们的攻击在不同LLM上的广泛有效性。这些结果强调了低层管理在时间序列预测中的关键弱点，强调了需要强大的防御机制来确保其在实际应用中的可靠部署。



## **2. TORCHLIGHT: Shedding LIGHT on Real-World Attacks on Cloudless IoT Devices Concealed within the Tor Network**

TORCHLIGHT：揭露隐藏在Tor网络中的无云物联网设备的现实攻击 cs.CR

27 pages, 14 figure, 9 tables

**SubmitDate**: 2025-01-28    [abs](http://arxiv.org/abs/2501.16784v1) [paper-pdf](http://arxiv.org/pdf/2501.16784v1)

**Authors**: Yumingzhi Pan, Zhen Ling, Yue Zhang, Hongze Wang, Guangchi Liu, Junzhou Luo, Xinwen Fu

**Abstract**: The rapidly expanding Internet of Things (IoT) landscape is shifting toward cloudless architectures, removing reliance on centralized cloud services but exposing devices directly to the internet and increasing their vulnerability to cyberattacks. Our research revealed an unexpected pattern of substantial Tor network traffic targeting cloudless IoT devices. suggesting that attackers are using Tor to anonymously exploit undisclosed vulnerabilities (possibly obtained from underground markets). To delve deeper into this phenomenon, we developed TORCHLIGHT, a tool designed to detect both known and unknown threats targeting cloudless IoT devices by analyzing Tor traffic. TORCHLIGHT filters traffic via specific IP patterns, strategically deploys virtual private server (VPS) nodes for cost-effective detection, and uses a chain-of-thought (CoT) process with large language models (LLMs) for accurate threat identification.   Our results are significant: for the first time, we have demonstrated that attackers are indeed using Tor to conceal their identities while targeting cloudless IoT devices. Over a period of 12 months, TORCHLIGHT analyzed 26 TB of traffic, revealing 45 vulnerabilities, including 29 zero-day exploits with 25 CVE-IDs assigned (5 CRITICAL, 3 HIGH, 16 MEDIUM, and 1 LOW) and an estimated value of approximately $312,000. These vulnerabilities affect around 12.71 million devices across 148 countries, exposing them to severe risks such as information disclosure, authentication bypass, and arbitrary command execution. The findings have attracted significant attention, sparking widespread discussion in cybersecurity circles, reaching the top 25 on Hacker News, and generating over 190,000 views.

摘要: 快速扩展的物联网(IoT)格局正在转向无云架构，消除了对集中式云服务的依赖，但将设备直接暴露在互联网中，增加了它们受到网络攻击的脆弱性。我们的研究揭示了一种意想不到的模式，即大量ToR网络流量针对无云IoT设备。这表明攻击者正在使用Tor匿名利用未披露的漏洞(可能是从地下市场获得的)。为了更深入地研究这一现象，我们开发了Torchlight，这是一款旨在通过分析ToR流量来检测针对无云IoT设备的已知和未知威胁的工具。Torchlight通过特定的IP模式过滤流量，战略性地部署虚拟专用服务器(VPS)节点进行经济高效的检测，并使用大型语言模型(LLM)的思想链(COT)流程进行准确的威胁识别。我们的结果意义重大：我们首次证明，攻击者确实在使用Tor隐藏身份，同时瞄准无云的物联网设备。在12个月的时间里，Torchlight分析了26 TB的流量，揭示了45个漏洞，其中包括29个零日漏洞，分配了25个CVE-ID(5个严重、3个高、16个中等和1个低)，估计价值约为312,000美元。这些漏洞影响了148个国家和地区的约1271万台设备，使它们面临严重的风险，如信息泄露、绕过身份验证和任意命令执行。这些发现引起了极大的关注，在网络安全圈引发了广泛的讨论，在黑客新闻上跻身前25名，浏览量超过19万次。



## **3. HateBench: Benchmarking Hate Speech Detectors on LLM-Generated Content and Hate Campaigns**

HateBench：对LLM生成的内容和仇恨活动的仇恨言语检测器进行基准测试 cs.CR

**SubmitDate**: 2025-01-28    [abs](http://arxiv.org/abs/2501.16750v1) [paper-pdf](http://arxiv.org/pdf/2501.16750v1)

**Authors**: Xinyue Shen, Yixin Wu, Yiting Qu, Michael Backes, Savvas Zannettou, Yang Zhang

**Abstract**: Large Language Models (LLMs) have raised increasing concerns about their misuse in generating hate speech. Among all the efforts to address this issue, hate speech detectors play a crucial role. However, the effectiveness of different detectors against LLM-generated hate speech remains largely unknown. In this paper, we propose HateBench, a framework for benchmarking hate speech detectors on LLM-generated hate speech. We first construct a hate speech dataset of 7,838 samples generated by six widely-used LLMs covering 34 identity groups, with meticulous annotations by three labelers. We then assess the effectiveness of eight representative hate speech detectors on the LLM-generated dataset. Our results show that while detectors are generally effective in identifying LLM-generated hate speech, their performance degrades with newer versions of LLMs. We also reveal the potential of LLM-driven hate campaigns, a new threat that LLMs bring to the field of hate speech detection. By leveraging advanced techniques like adversarial attacks and model stealing attacks, the adversary can intentionally evade the detector and automate hate campaigns online. The most potent adversarial attack achieves an attack success rate of 0.966, and its attack efficiency can be further improved by $13-21\times$ through model stealing attacks with acceptable attack performance. We hope our study can serve as a call to action for the research community and platform moderators to fortify defenses against these emerging threats.

摘要: 大型语言模型(LLM)在生成仇恨言论时被滥用，这引起了越来越多的关注。在解决这一问题的所有努力中，仇恨言论探测器发挥着至关重要的作用。然而，不同的检测器对LLM产生的仇恨言论的有效性在很大程度上仍不清楚。在这篇文章中，我们提出了一个框架，用于对LLM生成的仇恨言论进行仇恨言语检测器的基准测试。我们首先构建了一个由6个广泛使用的LLMS生成的7838个样本的仇恨语音数据集，覆盖了34个身份组，并由3个标记者进行了细致的标注。然后，我们在LLM生成的数据集上评估了八个具有代表性的仇恨语音检测器的有效性。我们的结果表明，虽然检测器在识别LLM生成的仇恨言论方面通常是有效的，但随着LLMS的更新，它们的性能会下降。我们还揭示了LLM驱动的仇恨运动的潜力，LLM给仇恨言语检测领域带来了新的威胁。通过利用对抗性攻击和模型窃取攻击等先进技术，敌手可以故意避开检测器，并自动在线进行仇恨运动。最强的对抗性攻击达到了0.966的攻击成功率，在攻击性能可接受的情况下，通过模型窃取攻击可以进一步提高攻击效率13-21倍。我们希望我们的研究能够成为研究界和平台主持人的行动号召，以加强对这些新出现的威胁的防御。



## **4. Can Watermarked LLMs be Identified by Users via Crafted Prompts?**

用户可以通过精心制作的脚本识别带水印的LLM吗？ cs.CR

28 pages, 5 figures, 11 tables Published as a conference paper at  ICLR 2025 Github link:  https://github.com/THU-BPM/Watermarked_LLM_Identification

**SubmitDate**: 2025-01-28    [abs](http://arxiv.org/abs/2410.03168v3) [paper-pdf](http://arxiv.org/pdf/2410.03168v3)

**Authors**: Aiwei Liu, Sheng Guan, Yiming Liu, Leyi Pan, Yifei Zhang, Liancheng Fang, Lijie Wen, Philip S. Yu, Xuming Hu

**Abstract**: Text watermarking for Large Language Models (LLMs) has made significant progress in detecting LLM outputs and preventing misuse. Current watermarking techniques offer high detectability, minimal impact on text quality, and robustness to text editing. However, current researches lack investigation into the imperceptibility of watermarking techniques in LLM services. This is crucial as LLM providers may not want to disclose the presence of watermarks in real-world scenarios, as it could reduce user willingness to use the service and make watermarks more vulnerable to attacks. This work is the first to investigate the imperceptibility of watermarked LLMs. We design an identification algorithm called Water-Probe that detects watermarks through well-designed prompts to the LLM. Our key motivation is that current watermarked LLMs expose consistent biases under the same watermark key, resulting in similar differences across prompts under different watermark keys. Experiments show that almost all mainstream watermarking algorithms are easily identified with our well-designed prompts, while Water-Probe demonstrates a minimal false positive rate for non-watermarked LLMs. Finally, we propose that the key to enhancing the imperceptibility of watermarked LLMs is to increase the randomness of watermark key selection. Based on this, we introduce the Water-Bag strategy, which significantly improves watermark imperceptibility by merging multiple watermark keys.

摘要: 针对大语言模型的文本水印技术在检测大语言模型输出和防止误用方面取得了显著进展。目前的水印技术提供了高可检测性，对文本质量的影响最小，以及对文本编辑的稳健性。然而，目前的研究缺乏对LLM服务中水印技术不可见性的研究。这一点至关重要，因为LLM提供商可能不想透露真实场景中是否存在水印，因为这可能会降低用户使用该服务的意愿，并使水印更容易受到攻击。这项工作是首次研究带水印的LLM的不可感知性。我们设计了一种名为Water-Probe的识别算法，该算法通过对LLM的精心设计的提示来检测水印。我们的关键动机是，当前的水印LLM暴露了相同水印密钥下的一致偏差，导致不同水印密钥下的提示存在相似的差异。实验表明，几乎所有的主流水印算法都能在我们精心设计的提示下很容易地识别出来，而Water-Probe算法对未加水印的LLMS具有最低的误检率。最后，提出了提高水印LLMS不可见性的关键是增加水印密钥选择的随机性。在此基础上，引入了水袋策略，通过合并多个水印密钥，显著提高了水印的不可见性。



## **5. xJailbreak: Representation Space Guided Reinforcement Learning for Interpretable LLM Jailbreaking**

x越狱：可解释LLM越狱的表示空间引导强化学习 cs.CL

**SubmitDate**: 2025-01-28    [abs](http://arxiv.org/abs/2501.16727v1) [paper-pdf](http://arxiv.org/pdf/2501.16727v1)

**Authors**: Sunbowen Lee, Shiwen Ni, Chi Wei, Shuaimin Li, Liyang Fan, Ahmadreza Argha, Hamid Alinejad-Rokny, Ruifeng Xu, Yicheng Gong, Min Yang

**Abstract**: Safety alignment mechanism are essential for preventing large language models (LLMs) from generating harmful information or unethical content. However, cleverly crafted prompts can bypass these safety measures without accessing the model's internal parameters, a phenomenon known as black-box jailbreak. Existing heuristic black-box attack methods, such as genetic algorithms, suffer from limited effectiveness due to their inherent randomness, while recent reinforcement learning (RL) based methods often lack robust and informative reward signals. To address these challenges, we propose a novel black-box jailbreak method leveraging RL, which optimizes prompt generation by analyzing the embedding proximity between benign and malicious prompts. This approach ensures that the rewritten prompts closely align with the intent of the original prompts while enhancing the attack's effectiveness. Furthermore, we introduce a comprehensive jailbreak evaluation framework incorporating keywords, intent matching, and answer validation to provide a more rigorous and holistic assessment of jailbreak success. Experimental results show the superiority of our approach, achieving state-of-the-art (SOTA) performance on several prominent open and closed-source LLMs, including Qwen2.5-7B-Instruct, Llama3.1-8B-Instruct, and GPT-4o-0806. Our method sets a new benchmark in jailbreak attack effectiveness, highlighting potential vulnerabilities in LLMs. The codebase for this work is available at https://github.com/Aegis1863/xJailbreak.

摘要: 安全对齐机制对于防止大型语言模型(LLM)生成有害信息或不道德内容至关重要。然而，精心设计的提示可以绕过这些安全措施，而不需要访问模型的内部参数，这一现象被称为黑盒越狱。现有的启发式黑盒攻击方法，如遗传算法，由于其固有的随机性，其有效性有限，而最近的基于强化学习(RL)的方法往往缺乏健壮和信息丰富的奖励信号。为了应对这些挑战，我们提出了一种新的利用RL的黑盒越狱方法，该方法通过分析良性提示和恶意提示之间的嵌入邻近性来优化提示生成。这种方法确保重写的提示与原始提示的意图紧密一致，同时提高了攻击的有效性。此外，我们引入了一个全面的越狱评估框架，其中包括关键字、意图匹配和答案验证，以提供更严格和全面的越狱成功评估。实验结果表明了该方法的优越性，在Qwen2.5-7B-Direct、Llama3.1-8B-Direct和GPT-40-0806等几个著名的开源和闭源LLM上获得了最先进的性能(SOTA)。我们的方法在越狱攻击有效性方面设置了一个新的基准，突出了LLMS中的潜在漏洞。这项工作的代码库可在https://github.com/Aegis1863/xJailbreak.上获得



## **6. Jailbreaking Large Language Models Through Alignment Vulnerabilities in Out-of-Distribution Settings**

通过非分发环境中的对齐漏洞破解大型语言模型 cs.CL

**SubmitDate**: 2025-01-27    [abs](http://arxiv.org/abs/2406.13662v2) [paper-pdf](http://arxiv.org/pdf/2406.13662v2)

**Authors**: Yue Huang, Jingyu Tang, Dongping Chen, Bingda Tang, Yao Wan, Lichao Sun, Philip S. Yu, Xiangliang Zhang

**Abstract**: Recently, Large Language Models (LLMs) have garnered significant attention for their exceptional natural language processing capabilities. However, concerns about their trustworthiness remain unresolved, particularly in addressing ``jailbreaking'' attacks on aligned LLMs. Previous research predominantly relies on scenarios involving white-box LLMs or specific, fixed prompt templates, which are often impractical and lack broad applicability. In this paper, we introduce a straightforward and novel method called ObscurePrompt for jailbreaking LLMs, inspired by the observed fragile alignments in Out-of-Distribution (OOD) data. Specifically, we first formulate the decision boundary in the jailbreaking process and then explore how obscure text affects LLM's ethical decision boundary. ObscurePrompt starts with constructing a base prompt that integrates well-known jailbreaking techniques. Powerful LLMs are then utilized to obscure the original prompt through iterative transformations, aiming to bolster the attack's robustness. Comprehensive experiments show that our approach substantially improves upon previous methods in terms of attack effectiveness, maintaining efficacy against two prevalent defense mechanisms.

摘要: 近年来，大型语言模型(LLM)以其卓越的自然语言处理能力引起了人们的极大关注。然而，对其可信性的关切仍未得到解决，特别是在解决对结盟的小岛屿发展中国家的“越狱”攻击方面。以前的研究主要依赖于涉及白盒LLM或特定的固定提示模板的场景，这些场景往往不切实际，缺乏广泛的适用性。在这篇文章中，我们介绍了一种简单而新颖的方法，称为ObscurePrompt，用于越狱LLMS，灵感来自于观察到的分布外(OOD)数据中的脆弱比对。具体地说，我们首先阐述了越狱过程中的决策边界，然后探讨了晦涩的文本如何影响LLM的伦理决策边界。ObscurePrompt首先构建一个集成了众所周知的越狱技术的基本提示。然后利用强大的LLM通过迭代变换来模糊原始提示，旨在增强攻击的健壮性。综合实验表明，我们的方法在攻击有效性方面比以前的方法有了很大的提高，保持了对两种流行的防御机制的有效性。



## **7. Targeting Alignment: Extracting Safety Classifiers of Aligned LLMs**

目标对齐：提取对齐的LLM的安全分类器 cs.CR

**SubmitDate**: 2025-01-27    [abs](http://arxiv.org/abs/2501.16534v1) [paper-pdf](http://arxiv.org/pdf/2501.16534v1)

**Authors**: Jean-Charles Noirot Ferrand, Yohan Beugin, Eric Pauley, Ryan Sheatsley, Patrick McDaniel

**Abstract**: Alignment in large language models (LLMs) is used to enforce guidelines such as safety. Yet, alignment fails in the face of jailbreak attacks that modify inputs to induce unsafe outputs. In this paper, we present and evaluate a method to assess the robustness of LLM alignment. We observe that alignment embeds a safety classifier in the target model that is responsible for deciding between refusal and compliance. We seek to extract an approximation of this classifier, called a surrogate classifier, from the LLM. We develop an algorithm for identifying candidate classifiers from subsets of the LLM model. We evaluate the degree to which the candidate classifiers approximate the model's embedded classifier in benign (F1 score) and adversarial (using surrogates in a white-box attack) settings. Our evaluation shows that the best candidates achieve accurate agreement (an F1 score above 80%) using as little as 20% of the model architecture. Further, we find attacks mounted on the surrogate models can be transferred with high accuracy. For example, a surrogate using only 50% of the Llama 2 model achieved an attack success rate (ASR) of 70%, a substantial improvement over attacking the LLM directly, where we only observed a 22% ASR. These results show that extracting surrogate classifiers is a viable (and highly effective) means for modeling (and therein addressing) the vulnerability of aligned models to jailbreaking attacks.

摘要: 大型语言模型(LLM)中的对齐用于执行安全等准则。然而，面对修改输入以产生不安全输出的越狱攻击，对齐失败。在本文中，我们提出并评价了一种评估LLM配准稳健性的方法。我们观察到，对齐在目标模型中嵌入了一个安全分类器，该分类器负责在拒绝和遵从之间做出决定。我们试图从LLM中提取该分类器的近似值，称为代理分类器。我们提出了一种从LLM模型的子集中识别候选分类器的算法。我们评估了在良性(F1分数)和对抗性(在白盒攻击中使用代理)环境下，候选分类器与模型嵌入分类器的近似程度。我们的评估显示，最好的候选者仅使用模型体系结构的20%就可以实现准确的一致性(F1得分超过80%)。此外，我们发现安装在代理模型上的攻击可以高精度地转移。例如，一个只使用50%的Llama 2模型的代理程序实现了70%的攻击成功率(ASR)，与我们只观察到22%的ASR的直接攻击LLM相比，这是一个实质性的改进。这些结果表明，提取代理分类器是一种可行的(并且非常有效的)方法，用于建模(并在其中解决)对齐模型对越狱攻击的脆弱性。



## **8. Smoothed Embeddings for Robust Language Models**

稳健语言模型的平滑嵌入 cs.LG

Presented in the Safe Generative AI Workshop at NeurIPS 2024

**SubmitDate**: 2025-01-27    [abs](http://arxiv.org/abs/2501.16497v1) [paper-pdf](http://arxiv.org/pdf/2501.16497v1)

**Authors**: Ryo Hase, Md Rafi Ur Rashid, Ashley Lewis, Jing Liu, Toshiaki Koike-Akino, Kieran Parsons, Ye Wang

**Abstract**: Improving the safety and reliability of large language models (LLMs) is a crucial aspect of realizing trustworthy AI systems. Although alignment methods aim to suppress harmful content generation, LLMs are often still vulnerable to jailbreaking attacks that employ adversarial inputs that subvert alignment and induce harmful outputs. We propose the Randomized Embedding Smoothing and Token Aggregation (RESTA) defense, which adds random noise to the embedding vectors and performs aggregation during the generation of each output token, with the aim of better preserving semantic information. Our experiments demonstrate that our approach achieves superior robustness versus utility tradeoffs compared to the baseline defenses.

摘要: 提高大型语言模型（LLM）的安全性和可靠性是实现值得信赖的人工智能系统的一个重要方面。尽管对齐方法的目的是抑制有害内容的生成，但LLM通常仍然容易受到越狱攻击，这些攻击采用颠覆对齐并引发有害输出的对抗性输入。我们提出了随机嵌入平滑和令牌聚合（RESTA）防御，它向嵌入载体添加随机噪音，并在每个输出令牌的生成过程中执行聚合，目的是更好地保存语义信息。我们的实验表明，与基线防御相比，我们的方法实现了更好的鲁棒性与效用权衡。



## **9. From Prompt Injections to SQL Injection Attacks: How Protected is Your LLM-Integrated Web Application?**

从提示注入到SQL注入攻击：您的LLM集成Web应用程序受到的保护程度如何？ cs.CR

12 pages, 3 figures, 3 tables, 5 listings. 47th IEEE/ACM  International Conference on Software Engineering (2025)

**SubmitDate**: 2025-01-27    [abs](http://arxiv.org/abs/2308.01990v4) [paper-pdf](http://arxiv.org/pdf/2308.01990v4)

**Authors**: Rodrigo Pedro, Daniel Castro, Paulo Carreira, Nuno Santos

**Abstract**: Large Language Models (LLMs) have found widespread applications in various domains, including web applications, where they facilitate human interaction via chatbots with natural language interfaces. Internally, aided by an LLM-integration middleware such as Langchain, user prompts are translated into SQL queries used by the LLM to provide meaningful responses to users. However, unsanitized user prompts can lead to SQL injection attacks, potentially compromising the security of the database. Despite the growing interest in prompt injection vulnerabilities targeting LLMs, the specific risks of generating SQL injection attacks through prompt injections have not been extensively studied. In this paper, we present a comprehensive examination of prompt-to-SQL (P$_2$SQL) injections targeting web applications based on the Langchain framework. Using Langchain as our case study, we characterize P$_2$SQL injections, exploring their variants and impact on application security through multiple concrete examples. Furthermore, we evaluate 7 state-of-the-art LLMs, demonstrating the pervasiveness of P$_2$SQL attacks across language models. Our findings indicate that LLM-integrated applications based on Langchain are highly susceptible to P$_2$SQL injection attacks, warranting the adoption of robust defenses. To counter these attacks, we propose four effective defense techniques that can be integrated as extensions to the Langchain framework. We validate the defenses through an experimental evaluation with a real-world use case application.

摘要: 大型语言模型在包括网络应用在内的各个领域得到了广泛的应用，在这些领域中，它们通过带有自然语言界面的聊天机器人来促进人类交互。在内部，在LLM集成中间件(如Langchain)的帮助下，用户提示被转换为LLM使用的SQL查询，以向用户提供有意义的响应。但是，未经清理的用户提示可能会导致SQL注入攻击，从而可能危及数据库的安全性。尽管人们对针对LLM的即时注入漏洞的兴趣与日俱增，但通过即时注入生成SQL注入攻击的具体风险尚未得到广泛研究。在这篇文章中，我们提出了一个全面的审查，以快速到SQL(P$2$SQL)注入针对Web应用程序基于朗之链框架。以Lang Chain为例，我们描述了P$2$SQL注入的特征，通过多个具体实例探索了它们的变体及其对应用程序安全性的影响。此外，我们对7个最新的LLM进行了评估，证明了P$2$SQL攻击在语言模型中的普遍性。我们的研究结果表明，基于LLm集成的应用程序非常容易受到P$2$SQL注入攻击，因此需要采取健壮的防御措施。为了应对这些攻击，我们提出了四种有效的防御技术，这些技术可以作为Langchain框架的扩展集成在一起。我们通过使用真实世界的用例应用程序进行实验评估来验证防御措施。



## **10. Detecting Zero-Day Attacks in Digital Substations via In-Context Learning**

通过上下文学习检测数字变电站中的零日攻击 cs.LG

**SubmitDate**: 2025-01-27    [abs](http://arxiv.org/abs/2501.16453v1) [paper-pdf](http://arxiv.org/pdf/2501.16453v1)

**Authors**: Faizan Manzoor, Vanshaj Khattar, Akila Herath, Clifton Black, Matthew C Nielsen, Junho Hong, Chen-Ching Liu, Ming Jin

**Abstract**: The occurrences of cyber attacks on the power grids have been increasing every year, with novel attack techniques emerging every year. In this paper, we address the critical challenge of detecting novel/zero-day attacks in digital substations that employ the IEC-61850 communication protocol. While many heuristic and machine learning (ML)-based methods have been proposed for attack detection in IEC-61850 digital substations, generalization to novel or zero-day attacks remains challenging. We propose an approach that leverages the in-context learning (ICL) capability of the transformer architecture, the fundamental building block of large language models. The ICL approach enables the model to detect zero-day attacks and learn from a few examples of that attack without explicit retraining. Our experiments on the IEC-61850 dataset demonstrate that the proposed method achieves more than $85\%$ detection accuracy on zero-day attacks while the existing state-of-the-art baselines fail. This work paves the way for building more secure and resilient digital substations of the future.

摘要: 对电网的网络攻击事件每年都在增加，新的攻击技术每年都在涌现。在本文中，我们解决了在采用IEC-61850通信协议的数字化变电站中检测新型/零日攻击的关键挑战。虽然已经提出了许多基于启发式和机器学习(ML)的方法来检测IEC-61850数字变电站中的攻击，但将其推广到新的攻击或零日攻击仍然是具有挑战性的。我们提出了一种利用转换器体系结构的上下文中学习(ICL)能力的方法，该体系结构是大型语言模型的基本构建块。ICL方法使模型能够检测零日攻击，并从该攻击的几个示例中学习，而无需明确的再培训。我们在IEC-61850数据集上的实验表明，该方法在零日攻击检测准确率超过85美元的情况下，现有的基线检测方法不能满足检测精度的要求。这项工作为建设更安全、更具弹性的未来数字化变电站铺平了道路。



## **11. FDLLM: A Text Fingerprint Detection Method for LLMs in Multi-Language, Multi-Domain Black-Box Environments**

FDLLM：多语言、多域黑匣子环境中LLM的文本指纹检测方法 cs.CR

**SubmitDate**: 2025-01-27    [abs](http://arxiv.org/abs/2501.16029v1) [paper-pdf](http://arxiv.org/pdf/2501.16029v1)

**Authors**: Zhiyuan Fu, Junfan Chen, Hongyu Sun, Ting Yang, Ruidong Li, Yuqing Zhang

**Abstract**: Using large language models (LLMs) integration platforms without transparency about which LLM is being invoked can lead to potential security risks. Specifically, attackers may exploit this black-box scenario to deploy malicious models and embed viruses in the code provided to users. In this context, it is increasingly urgent for users to clearly identify the LLM they are interacting with, in order to avoid unknowingly becoming victims of malicious models. However, existing studies primarily focus on mixed classification of human and machine-generated text, with limited attention to classifying texts generated solely by different models. Current research also faces dual bottlenecks: poor quality of LLM-generated text (LLMGT) datasets and limited coverage of detectable LLMs, resulting in poor detection performance for various LLMGT in black-box scenarios. We propose the first LLMGT fingerprint detection model, \textbf{FDLLM}, based on Qwen2.5-7B and fine-tuned using LoRA to address these challenges. FDLLM can more efficiently handle detection tasks across multilingual and multi-domain scenarios. Furthermore, we constructed a dataset named \textbf{FD-Datasets}, consisting of 90,000 samples that span multiple languages and domains, covering 20 different LLMs. Experimental results demonstrate that FDLLM achieves a macro F1 score 16.7\% higher than the best baseline method, LM-D.

摘要: 使用大型语言模型(LLM)集成平台而不透明地调用LLM可能会导致潜在的安全风险。具体地说，攻击者可能会利用此黑盒方案来部署恶意模型，并在提供给用户的代码中嵌入病毒。在这种情况下，用户越来越迫切地需要清楚地识别他们正在与之交互的LLM，以避免在不知不觉中成为恶意模式的受害者。然而，现有的研究主要集中在人类和机器生成的文本的混合分类上，而对单独由不同模型生成的文本的分类关注很少。目前的研究还面临着双重瓶颈：LLMGT(LLMGT)数据集的质量不佳和可检测LLMS的覆盖率有限，导致在黑盒场景下对各种LLMGT的检测性能较差。我们基于Qwen2.5-7B提出了第一个LLMGT指纹检测模型，并使用LORA进行了微调，以应对这些挑战。FDLLM可以更有效地处理跨多语言和多域场景的检测任务。此外，我们还构建了一个名为\extbf{fd-DataSets}的数据集，该数据集包含90,000个跨语言和域的样本，涵盖20个不同的LLM。实验结果表明，FDLLM的宏观F1得分比最优基线方法LM-D高16.7分。



## **12. Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models**

对大型语言模型进行基准测试和防御间接提示注入攻击 cs.CL

Accepted by KDD 2025

**SubmitDate**: 2025-01-27    [abs](http://arxiv.org/abs/2312.14197v4) [paper-pdf](http://arxiv.org/pdf/2312.14197v4)

**Authors**: Jingwei Yi, Yueqi Xie, Bin Zhu, Emre Kiciman, Guangzhong Sun, Xing Xie, Fangzhao Wu

**Abstract**: The integration of large language models with external content has enabled applications such as Microsoft Copilot but also introduced vulnerabilities to indirect prompt injection attacks. In these attacks, malicious instructions embedded within external content can manipulate LLM outputs, causing deviations from user expectations. To address this critical yet under-explored issue, we introduce the first benchmark for indirect prompt injection attacks, named BIPIA, to assess the risk of such vulnerabilities. Using BIPIA, we evaluate existing LLMs and find them universally vulnerable. Our analysis identifies two key factors contributing to their success: LLMs' inability to distinguish between informational context and actionable instructions, and their lack of awareness in avoiding the execution of instructions within external content. Based on these findings, we propose two novel defense mechanisms-boundary awareness and explicit reminder-to address these vulnerabilities in both black-box and white-box settings. Extensive experiments demonstrate that our black-box defense provides substantial mitigation, while our white-box defense reduces the attack success rate to near-zero levels, all while preserving the output quality of LLMs. We hope this work inspires further research into securing LLM applications and fostering their safe and reliable use.

摘要: 大型语言模型与外部内容的集成使Microsoft Copilot等应用程序成为可能，但也带来了间接提示注入攻击的漏洞。在这些攻击中，嵌入在外部内容中的恶意指令可以操纵LLM输出，从而导致偏离用户预期。为了解决这一关键但未得到充分探索的问题，我们引入了第一个间接即时注入攻击基准，称为BIPIA，以评估此类漏洞的风险。使用BIPIA，我们评估了现有的LLM，发现它们普遍存在脆弱性。我们的分析确定了它们成功的两个关键因素：LLMS无法区分信息上下文和可操作指令，以及它们缺乏避免执行外部内容中的指令的意识。基于这些发现，我们提出了两种新的防御机制--边界感知和显式提醒--来解决黑盒和白盒环境中的这些漏洞。广泛的实验表明，我们的黑盒防御提供了实质性的缓解，而我们的白盒防御将攻击成功率降低到接近零的水平，所有这些都保持了LLMS的输出质量。我们希望这项工作对保护LLM应用程序和促进其安全可靠使用方面的进一步研究有所启发。



## **13. LLM-attacker: Enhancing Closed-loop Adversarial Scenario Generation for Autonomous Driving with Large Language Models**

LLM攻击者：使用大型语言模型增强自动驾驶的闭环对抗场景生成 cs.LG

**SubmitDate**: 2025-01-27    [abs](http://arxiv.org/abs/2501.15850v1) [paper-pdf](http://arxiv.org/pdf/2501.15850v1)

**Authors**: Yuewen Mei, Tong Nie, Jian Sun, Ye Tian

**Abstract**: Ensuring and improving the safety of autonomous driving systems (ADS) is crucial for the deployment of highly automated vehicles, especially in safety-critical events. To address the rarity issue, adversarial scenario generation methods are developed, in which behaviors of traffic participants are manipulated to induce safety-critical events. However, existing methods still face two limitations. First, identification of the adversarial participant directly impacts the effectiveness of the generation. However, the complexity of real-world scenarios, with numerous participants and diverse behaviors, makes identification challenging. Second, the potential of generated safety-critical scenarios to continuously improve ADS performance remains underexplored. To address these issues, we propose LLM-attacker: a closed-loop adversarial scenario generation framework leveraging large language models (LLMs). Specifically, multiple LLM agents are designed and coordinated to identify optimal attackers. Then, the trajectories of the attackers are optimized to generate adversarial scenarios. These scenarios are iteratively refined based on the performance of ADS, forming a feedback loop to improve ADS. Experimental results show that LLM-attacker can create more dangerous scenarios than other methods, and the ADS trained with it achieves a collision rate half that of training with normal scenarios. This indicates the ability of LLM-attacker to test and enhance the safety and robustness of ADS. Video demonstrations are provided at: https://drive.google.com/file/d/1Zv4V3iG7825oyiKbUwS2Y-rR0DQIE1ZA/view.

摘要: 确保和提高自动驾驶系统(ADS)的安全性对于高度自动化车辆的部署至关重要，特别是在安全关键事件中。为了解决这种稀缺性问题，开发了对抗性场景生成方法，在该方法中，交通参与者的行为被操纵以引发安全关键事件。然而，现有的方法仍然面临着两个局限性。首先，对抗性参与者的识别直接影响到生成的有效性。然而，现实世界场景的复杂性，参与者众多，行为多样，使得识别具有挑战性。其次，生成的安全关键场景持续提高广告性能的潜力仍未得到充分开发。为了解决这些问题，我们提出了LLM-攻击者：一个利用大型语言模型(LLMS)的闭环对抗性场景生成框架。具体地说，多个LLM代理被设计和协调以识别最佳攻击者。然后，攻击者的轨迹被优化以生成对抗性场景。这些场景根据广告的表现进行迭代细化，形成一个反馈循环来改进广告。实验结果表明，LLM-攻击者能够产生比其他方法更危险的场景，并且用它训练的ADS的冲突率是正常场景训练的一半。这表明了LLM-攻击者测试和增强ADS安全性和健壮性的能力。提供视频演示，网址为：https://drive.google.com/file/d/1Zv4V3iG7825oyiKbUwS2Y-rR0DQIE1ZA/view.



## **14. Crabs: Consuming Resrouce via Auto-generation for LLM-DoS Attack under Black-box Settings**

螃蟹：在黑匣子设置下通过自动生成LLM-NOS攻击消耗资源 cs.CL

20 pages, 7 figures, 11 tables

**SubmitDate**: 2025-01-27    [abs](http://arxiv.org/abs/2412.13879v2) [paper-pdf](http://arxiv.org/pdf/2412.13879v2)

**Authors**: Yuanhe Zhang, Zhenhong Zhou, Wei Zhang, Xinyue Wang, Xiaojun Jia, Yang Liu, Sen Su

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across diverse tasks. LLMs continue to be vulnerable to external threats, particularly Denial-of-Service (DoS) attacks. Specifically, LLM-DoS attacks aim to exhaust computational resources and block services. However, prior works tend to focus on performing white-box attacks, overlooking black-box settings. In this work, we propose an automated algorithm designed for black-box LLMs, called Auto-Generation for LLM-DoS Attack (AutoDoS). AutoDoS introduces DoS Attack Tree and optimizes the prompt node coverage to enhance effectiveness under black-box conditions. Our method can bypass existing defense with enhanced stealthiness via semantic improvement of prompt nodes. Furthermore, we reveal that implanting Length Trojan in Basic DoS Prompt aids in achieving higher attack efficacy. Experimental results show that AutoDoS amplifies service response latency by over 250 $\times \uparrow$, leading to severe resource consumption in terms of GPU utilization and memory usage. Our code is available at https://github.com/shuita2333/AutoDoS.

摘要: 大型语言模型(LLM)在不同的任务中表现出了显著的性能。LLMS仍然容易受到外部威胁，特别是拒绝服务(DoS)攻击。具体地说，LLM-DoS攻击旨在耗尽计算资源并阻止服务。然而，以往的工作往往侧重于执行白盒攻击，而忽略了黑盒设置。在这项工作中，我们提出了一种针对黑盒LLMS的自动化算法，称为LLM-DoS攻击自动生成算法(AutoDoS)。AutoDoS引入了DoS攻击树，优化了提示节点覆盖率，提高了黑盒情况下的有效性。该方法通过对提示节点进行语义改进，绕过了现有的防御机制，增强了隐蔽性。此外，我们还揭示了在基本DoS提示中植入Long特洛伊木马有助于实现更高的攻击效率。实验结果表明，AutoDoS将服务响应延迟放大了250倍以上，导致GPU使用率和内存使用率严重消耗资源。我们的代码可以在https://github.com/shuita2333/AutoDoS.上找到



## **15. PAPILLON: Efficient and Stealthy Fuzz Testing-Powered Jailbreaks for LLMs**

PAPILLON：针对LLM的高效、隐蔽的Fuzz测试动力越狱 cs.CR

**SubmitDate**: 2025-01-26    [abs](http://arxiv.org/abs/2409.14866v3) [paper-pdf](http://arxiv.org/pdf/2409.14866v3)

**Authors**: Xueluan Gong, Mingzhe Li, Yilin Zhang, Fengyuan Ran, Chen Chen, Yanjiao Chen, Qian Wang, Kwok-Yan Lam

**Abstract**: Large Language Models (LLMs) have excelled in various tasks but are still vulnerable to jailbreaking attacks, where attackers create jailbreak prompts to mislead the model to produce harmful or offensive content. Current jailbreak methods either rely heavily on manually crafted templates, which pose challenges in scalability and adaptability, or struggle to generate semantically coherent prompts, making them easy to detect. Additionally, most existing approaches involve lengthy prompts, leading to higher query costs.In this paper, to remedy these challenges, we introduce a novel jailbreaking attack framework called PAPILLON, which is an automated, black-box jailbreaking attack framework that adapts the black-box fuzz testing approach with a series of customized designs. Instead of relying on manually crafted templates,PAPILLON starts with an empty seed pool, removing the need to search for any related jailbreaking templates. We also develop three novel question-dependent mutation strategies using an LLM helper to generate prompts that maintain semantic coherence while significantly reducing their length. Additionally, we implement a two-level judge module to accurately detect genuine successful jailbreaks. We evaluated PAPILLON on 7 representative LLMs and compared it with 5 state-of-the-art jailbreaking attack strategies. For proprietary LLM APIs, such as GPT-3.5 turbo, GPT-4, and Gemini-Pro, PAPILLONs achieves attack success rates of over 90%, 80%, and 74%, respectively, exceeding existing baselines by more than 60\%. Additionally, PAPILLON can maintain high semantic coherence while significantly reducing the length of jailbreak prompts. When targeting GPT-4, PAPILLON can achieve over 78% attack success rate even with 100 tokens. Moreover, PAPILLON demonstrates transferability and is robust to state-of-the-art defenses.

摘要: 大型语言模型(LLM)在各种任务中表现出色，但仍然容易受到越狱攻击，在越狱攻击中，攻击者创建越狱提示来误导模型生成有害或攻击性内容。当前的越狱方法要么严重依赖于人工制作的模板，这对可伸缩性和适应性构成了挑战，要么难以生成语义连贯的提示，使它们很容易被检测到。为了解决这些问题，本文提出了一种新的越狱攻击框架Papillon，它是一个自动化的黑盒越狱攻击框架，通过一系列定制设计采用了黑盒模糊测试方法。与依赖手工制作的模板不同，Papillon从一个空的种子库开始，不需要搜索任何相关的越狱模板。我们还开发了三种新的问题相关突变策略，使用LLM助手来生成提示，这些提示在保持语义连贯的同时显著缩短了提示的长度。此外，我们实现了一个两级判断模块来准确地检测真正的成功越狱。我们在7个有代表性的LLM上对Papillon进行了评估，并将其与5种最先进的越狱攻击策略进行了比较。对于专有的LLMAPI，如GPT-3.5 Turbo、GPT-4和Gemini-Pro，Papillons的攻击成功率分别超过90%、80%和74%，比现有基线高出60%以上。此外，Papillon可以保持高度的语义连贯性，同时显著缩短越狱提示的长度。当针对GPT-4时，Papillon即使使用100个令牌也可以达到78%以上的攻击成功率。此外，乳突展示了可转移性，并对最先进的防御措施具有很强的抵抗力。



## **16. Improving Network Threat Detection by Knowledge Graph, Large Language Model, and Imbalanced Learning**

通过知识图、大语言模型和不平衡学习改进网络威胁检测 cs.LG

Accepted by "Combining AI and OR/MS for Better Trustworthy Decision  Making" Bridge Program co-organized by AAAI and INFORMS as poster and demo

**SubmitDate**: 2025-01-26    [abs](http://arxiv.org/abs/2501.16393v1) [paper-pdf](http://arxiv.org/pdf/2501.16393v1)

**Authors**: Lili Zhang, Quanyan Zhu, Herman Ray, Ying Xie

**Abstract**: Network threat detection has been challenging due to the complexities of attack activities and the limitation of historical threat data to learn from. To help enhance the existing practices of using analytics, machine learning, and artificial intelligence methods to detect the network threats, we propose an integrated modelling framework, where Knowledge Graph is used to analyze the users' activity patterns, Imbalanced Learning techniques are used to prune and weigh Knowledge Graph, and LLM is used to retrieve and interpret the users' activities from Knowledge Graph. The proposed framework is applied to Agile Threat Detection through Online Sequential Learning. The preliminary results show the improved threat capture rate by 3%-4% and the increased interpretabilities of risk predictions based on the users' activities.

摘要: 由于攻击活动的复杂性和可供学习的历史威胁数据的局限性，网络威胁检测一直具有挑战性。为了帮助增强使用分析、机器学习和人工智能方法检测网络威胁的现有实践，我们提出了一个集成的建模框架，其中使用知识图来分析用户的活动模式，使用不平衡学习技术来修剪和加权知识图，使用LLM来从知识图中检索和解释用户的活动。所提出的框架通过在线顺序学习应用于敏捷威胁检测。初步结果显示，威胁捕获率提高了3%-4%，并且基于用户活动的风险预测的可解释性增强。



## **17. Mirage in the Eyes: Hallucination Attack on Multi-modal Large Language Models with Only Attention Sink**

眼中的幻象：对只有注意力下沉的多模式大型语言模型的幻觉攻击 cs.LG

USENIX Security 2025

**SubmitDate**: 2025-01-25    [abs](http://arxiv.org/abs/2501.15269v1) [paper-pdf](http://arxiv.org/pdf/2501.15269v1)

**Authors**: Yining Wang, Mi Zhang, Junjie Sun, Chenyue Wang, Min Yang, Hui Xue, Jialing Tao, Ranjie Duan, Jiexi Liu

**Abstract**: Fusing visual understanding into language generation, Multi-modal Large Language Models (MLLMs) are revolutionizing visual-language applications. Yet, these models are often plagued by the hallucination problem, which involves generating inaccurate objects, attributes, and relationships that do not match the visual content. In this work, we delve into the internal attention mechanisms of MLLMs to reveal the underlying causes of hallucination, exposing the inherent vulnerabilities in the instruction-tuning process.   We propose a novel hallucination attack against MLLMs that exploits attention sink behaviors to trigger hallucinated content with minimal image-text relevance, posing a significant threat to critical downstream applications. Distinguished from previous adversarial methods that rely on fixed patterns, our approach generates dynamic, effective, and highly transferable visual adversarial inputs, without sacrificing the quality of model responses. Comprehensive experiments on 6 prominent MLLMs demonstrate the efficacy of our attack in compromising black-box MLLMs even with extensive mitigating mechanisms, as well as the promising results against cutting-edge commercial APIs, such as GPT-4o and Gemini 1.5. Our code is available at https://huggingface.co/RachelHGF/Mirage-in-the-Eyes.

摘要: 多模式大型语言模型将视觉理解融合到语言生成中，正在给视觉语言应用带来革命性的变化。然而，这些模型经常受到幻觉问题的困扰，幻觉问题涉及生成与视觉内容不匹配的不准确的对象、属性和关系。在这项工作中，我们深入研究了MLMS的内部注意机制，以揭示产生幻觉的潜在原因，揭示了教学调整过程中的内在弱点。我们提出了一种新的针对MLLMS的幻觉攻击，该攻击利用注意力吸收行为来触发具有最小图文相关性的幻觉内容，从而对关键的下游应用构成重大威胁。与以前依赖固定模式的对抗性方法不同，我们的方法产生动态、有效和高度可转移的视觉对抗性输入，而不牺牲模型响应的质量。在6个重要的MLLMS上的综合实验证明了我们的攻击在危害黑盒MLLMS方面的有效性，即使具有广泛的缓解机制，以及对尖端商业API，如GPT-40和Gemini 1.5的有希望的结果。我们的代码可以在https://huggingface.co/RachelHGF/Mirage-in-the-Eyes.上找到



## **18. PromptShield: Deployable Detection for Prompt Injection Attacks**

EntShield：针对即时注入攻击的可部署检测 cs.CR

**SubmitDate**: 2025-01-25    [abs](http://arxiv.org/abs/2501.15145v1) [paper-pdf](http://arxiv.org/pdf/2501.15145v1)

**Authors**: Dennis Jacob, Hend Alzahrani, Zhanhao Hu, Basel Alomair, David Wagner

**Abstract**: Current application designers have moved to integrate large language models (LLMs) into their products. These LLM-integrated applications are vulnerable to prompt injection vulnerabilities. While attempts have been made to address this problem by building a detector that can monitor inputs to the LLM and detect attacks, we find that many detectors are not yet suitable for practical deployment. To support research in this area, we design the PromptShield benchmark for evaluating practical prompt injection detectors. We also construct a new detector, the PromptShield detector, which achieves significantly better performance at detecting prompt injection attacks than any prior scheme. Our work suggests that larger models, more training data, appropriate metrics, and careful curation of training data can contribute to strong detector performance.

摘要: 当前的应用程序设计师已经开始将大型语言模型（LLM）集成到他们的产品中。这些LLM集成的应用程序容易受到提示注入漏洞的影响。虽然已经尝试通过构建一个可以监控LLM输入并检测攻击的检测器来解决这个问题，但我们发现许多检测器尚不适合实际部署。为了支持该领域的研究，我们设计了EmotiShield基准来评估实用的即时注射检测器。我们还构建了一个新的检测器，即SpectShield检测器，它在检测即时注入攻击方面实现了比任何先前方案都更好的性能。我们的工作表明，更大的模型、更多的训练数据、适当的指标和对训练数据的精心策划可以有助于增强检测器的性能。



## **19. NLP Verification: Towards a General Methodology for Certifying Robustness**

NLP验证：迈向验证稳健性的通用方法 cs.CL

**SubmitDate**: 2025-01-24    [abs](http://arxiv.org/abs/2403.10144v3) [paper-pdf](http://arxiv.org/pdf/2403.10144v3)

**Authors**: Marco Casadio, Tanvi Dinkar, Ekaterina Komendantskaya, Luca Arnaboldi, Matthew L. Daggitt, Omri Isac, Guy Katz, Verena Rieser, Oliver Lemon

**Abstract**: Machine Learning (ML) has exhibited substantial success in the field of Natural Language Processing (NLP). For example large language models have empirically proven to be capable of producing text of high complexity and cohesion. However, they are prone to inaccuracies and hallucinations. As these systems are increasingly integrated into real-world applications, ensuring their safety and reliability becomes a primary concern. There are safety critical contexts where such models must be robust to variability or attack, and give guarantees over their output. Computer Vision had pioneered the use of formal verification of neural networks for such scenarios and developed common verification standards and pipelines, leveraging precise formal reasoning about geometric properties of data manifolds. In contrast, NLP verification methods have only recently appeared in the literature. While presenting sophisticated algorithms, these papers have not yet crystallised into a common methodology. They are often light on the pragmatical issues of NLP verification and the area remains fragmented. In this paper, we attempt to distil and evaluate general components of an NLP verification pipeline, that emerges from the progress in the field to date. Our contributions are two-fold. Firstly, we propose a general methodology to analyse the effect of the embedding gap, a problem that refers to the discrepancy between verification of geometric subspaces and the semantic meaning of sentences, which the geometric subspaces are supposed to represent. We propose a number of practical NLP methods that can help to quantify the effects of the embedding gap. Secondly, we give a general method for training and verification of neural networks that leverages a more precise geometric estimation of semantic similarity of sentences in the embedding space and helps to overcome the effects of the embedding gap in practice.

摘要: 机器学习(ML)在自然语言处理(NLP)领域取得了巨大的成功。例如，经验证明，大型语言模型能够产生高度复杂和衔接的文本。然而，他们容易出现不准确和幻觉。随着这些系统越来越多地集成到现实世界的应用中，确保它们的安全性和可靠性成为首要关注的问题。在安全关键环境中，这样的模型必须对可变性或攻击具有健壮性，并对其输出提供保证。计算机视觉率先将神经网络的形式验证用于此类场景，并开发了通用的验证标准和管道，利用关于数据流形的几何属性的精确形式推理。相比之下，NLP验证方法只是最近才出现在文献中。虽然提出了复杂的算法，但这些论文尚未形成一种通用的方法。他们往往对核不扩散核查的务实问题不感兴趣，这一领域仍然支离破碎。在这篇文章中，我们试图提炼和评估NLP验证流水线的通用组件，这些组件是从该领域迄今的进展中出现的。我们的贡献是双重的。首先，我们提出了一个通用的方法来分析嵌入间隙的影响，这个问题指的是几何子空间的验证与句子的语义之间的差异，几何子空间应该代表的是句子的语义。我们提出了一些实用的NLP方法，这些方法可以帮助量化嵌入间隙的影响。其次，我们给出了一种通用的神经网络训练和验证方法，该方法利用嵌入空间中句子语义相似度的更精确的几何估计，并有助于克服实践中嵌入间隙的影响。



## **20. Self-playing Adversarial Language Game Enhances LLM Reasoning**

自玩对抗语言游戏增强LLM推理 cs.CL

Accepted by NeurIPS 2024

**SubmitDate**: 2025-01-24    [abs](http://arxiv.org/abs/2404.10642v3) [paper-pdf](http://arxiv.org/pdf/2404.10642v3)

**Authors**: Pengyu Cheng, Tianhao Hu, Han Xu, Zhisong Zhang, Zheng Yuan, Yong Dai, Lei Han, Nan Du, Xiaolong Li

**Abstract**: We explore the potential of self-play training for large language models (LLMs) in a two-player adversarial language game called Adversarial Taboo. In this game, an attacker and a defender communicate around a target word only visible to the attacker. The attacker aims to induce the defender to speak the target word unconsciously, while the defender tries to infer the target word from the attacker's utterances. To win the game, both players must have sufficient knowledge about the target word and high-level reasoning ability to infer and express in this information-reserved conversation. Hence, we are curious about whether LLMs' reasoning ability can be further enhanced by Self-Playing this Adversarial language Game (SPAG). With this goal, we select several open-source LLMs and let each act as the attacker and play with a copy of itself as the defender on an extensive range of target words. Through reinforcement learning on the game outcomes, we observe that the LLMs' performances uniformly improve on a broad range of reasoning benchmarks. Furthermore, iteratively adopting this self-play process can continuously promote LLMs' reasoning abilities. The code is available at https://github.com/Linear95/SPAG.

摘要: 我们探索了在一个名为对抗性禁忌的两人对抗性语言游戏中对大型语言模型(LLM)进行自我发挥训练的可能性。在这个游戏中，攻击者和防御者围绕一个只有攻击者才能看到的目标单词进行交流。攻击者的目的是诱导防御者无意识地说出目标词，而防御者则试图从攻击者的话语中推断出目标词。要赢得这场比赛，双方都必须有足够的目标词知识和高级推理能力，以便在这种信息储备的对话中进行推理和表达。因此，我们很好奇，通过自我玩这个对抗性语言游戏(SPAG)，LLMS的推理能力是否会进一步增强。带着这个目标，我们选择了几个开源的LLM，让每个LLM扮演攻击者的角色，并在广泛的目标词上扮演自己的防御者。通过对游戏结果的强化学习，我们观察到LLMS的性能在广泛的推理基准上一致提高。此外，迭代地采用这种自我发挥过程可以不断提升LLMS的推理能力。代码可在https://github.com/Linear95/SPAG.上获得



## **21. Real-world Edge Neural Network Implementations Leak Private Interactions Through Physical Side Channel**

现实世界的边缘神经网络实现通过物理侧通道泄露私人交互 cs.CR

**SubmitDate**: 2025-01-24    [abs](http://arxiv.org/abs/2501.14512v1) [paper-pdf](http://arxiv.org/pdf/2501.14512v1)

**Authors**: Zhuoran Liu, Senna van Hoek, Péter Horváth, Dirk Lauret, Xiaoyun Xu, Lejla Batina

**Abstract**: Neural networks have become a fundamental component of numerous practical applications, and their implementations, which are often accelerated by hardware, are integrated into all types of real-world physical devices. User interactions with neural networks on hardware accelerators are commonly considered privacy-sensitive. Substantial efforts have been made to uncover vulnerabilities and enhance privacy protection at the level of machine learning algorithms, including membership inference attacks, differential privacy, and federated learning. However, neural networks are ultimately implemented and deployed on physical devices, and current research pays comparatively less attention to privacy protection at the implementation level. In this paper, we introduce a generic physical side-channel attack, ScaAR, that extracts user interactions with neural networks by leveraging electromagnetic (EM) emissions of physical devices. Our proposed attack is implementation-agnostic, meaning it does not require the adversary to possess detailed knowledge of the hardware or software implementations, thanks to the capabilities of deep learning-based side-channel analysis (DLSCA). Experimental results demonstrate that, through the EM side channel, ScaAR can effectively extract the class label of user interactions with neural classifiers, including inputs and outputs, on the AMD-Xilinx MPSoC ZCU104 FPGA and Raspberry Pi 3 B. In addition, for the first time, we provide side-channel analysis on edge Large Language Model (LLM) implementations on the Raspberry Pi 5, showing that EM side channel leaks interaction data, and different LLM tokens can be distinguishable from the EM traces.

摘要: 神经网络已经成为许多实际应用的基本组件，其实现通常由硬件加速，并集成到所有类型的现实世界物理设备中。用户在硬件加速器上与神经网络的交互通常被认为是隐私敏感的。在发现漏洞和加强机器学习算法层面的隐私保护方面做出了实质性的努力，包括成员资格推理攻击、差异隐私和联合学习。然而，神经网络最终是在物理设备上实现和部署的，目前的研究相对较少关注实现层面的隐私保护。在本文中，我们介绍了一种通用的物理侧通道攻击，ScaAR，它通过利用物理设备的电磁发射来提取用户与神经网络的交互。我们提出的攻击是与实现无关的，这意味着它不需要攻击者拥有硬件或软件实现的详细知识，这要归功于基于深度学习的旁路分析(DLSCA)能力。实验结果表明，在AMD-Xilinx MPSoC ZCU104和Raspberry PI 3 B上，通过EM侧通道，ScaAR可以有效地提取用户与神经分类器交互的类别标签，包括输入和输出。此外，我们首次对Raspberry PI 5上的EDGE大语言模型(LLM)实现进行了侧通道分析，结果表明EM侧通道泄漏了交互数据，并且可以从EM跟踪中区分不同的LLM标记。



## **22. Dysca: A Dynamic and Scalable Benchmark for Evaluating Perception Ability of LVLMs**

Dysca：评估LVLM感知能力的动态和可扩展基准 cs.CV

Accepted by ICLR2025

**SubmitDate**: 2025-01-24    [abs](http://arxiv.org/abs/2406.18849v3) [paper-pdf](http://arxiv.org/pdf/2406.18849v3)

**Authors**: Jie Zhang, Zhongqi Wang, Mengqi Lei, Zheng Yuan, Bei Yan, Shiguang Shan, Xilin Chen

**Abstract**: Currently many benchmarks have been proposed to evaluate the perception ability of the Large Vision-Language Models (LVLMs). However, most benchmarks conduct questions by selecting images from existing datasets, resulting in the potential data leakage. Besides, these benchmarks merely focus on evaluating LVLMs on the realistic style images and clean scenarios, leaving the multi-stylized images and noisy scenarios unexplored. In response to these challenges, we propose a dynamic and scalable benchmark named Dysca for evaluating LVLMs by leveraging synthesis images. Specifically, we leverage Stable Diffusion and design a rule-based method to dynamically generate novel images, questions and the corresponding answers. We consider 51 kinds of image styles and evaluate the perception capability in 20 subtasks. Moreover, we conduct evaluations under 4 scenarios (i.e., Clean, Corruption, Print Attacking and Adversarial Attacking) and 3 question types (i.e., Multi-choices, True-or-false and Free-form). Thanks to the generative paradigm, Dysca serves as a scalable benchmark for easily adding new subtasks and scenarios. A total of 24 advanced open-source LVLMs and 2 close-source LVLMs are evaluated on Dysca, revealing the drawbacks of current LVLMs. The benchmark is released at \url{https://github.com/Robin-WZQ/Dysca}.

摘要: 目前，人们已经提出了许多基准来评估大型视觉语言模型的感知能力。然而，大多数基准测试通过从现有数据集中选择图像来进行问题，从而导致潜在的数据泄漏。此外，这些基准只关注真实感风格的图像和干净的场景来评估LVLMS，而对多风格化的图像和噪声场景没有进行探索。为了应对这些挑战，我们提出了一种动态的、可扩展的基准测试DYSCA，用于利用合成图像来评估LVLMS。具体地说，我们利用稳定扩散并设计了一种基于规则的方法来动态生成新的图像、问题和相应的答案。我们考虑了51种图像风格，并在20个子任务中评估了感知能力。此外，我们还在4个场景(即廉洁、腐败、打印攻击和对抗性攻击)和3个问题类型(即多项选择、对错和自由形式)下进行了评估。多亏了生成性范式，Dysca成为了一个可伸缩的基准，可以轻松添加新的子任务和场景。在Dysca上对24个先进的开源LVLMS和2个闭源LVLMS进行了评估，揭示了现有LVLMS的不足。基准发布地址为\url{https://github.com/Robin-WZQ/Dysca}.



## **23. Internal Activation Revision: Safeguarding Vision Language Models Without Parameter Update**

内部激活修订：在不更新参数的情况下保护视觉语言模型 cs.LG

**SubmitDate**: 2025-01-24    [abs](http://arxiv.org/abs/2501.16378v1) [paper-pdf](http://arxiv.org/pdf/2501.16378v1)

**Authors**: Qing Li, Jiahui Geng, Zongxiong Chen, Kun Song, Lei Ma, Fakhri Karray

**Abstract**: Vision-language models (VLMs) demonstrate strong multimodal capabilities but have been found to be more susceptible to generating harmful content compared to their backbone large language models (LLMs). Our investigation reveals that the integration of images significantly shifts the model's internal activations during the forward pass, diverging from those triggered by textual input. Moreover, the safety alignments of LLMs embedded within VLMs are not sufficiently robust to handle the activations discrepancies, making the models vulnerable to even the simplest jailbreaking attacks. To address this issue, we propose an \textbf{internal activation revision} approach that efficiently revises activations during generation, steering the model toward safer outputs. Our framework incorporates revisions at both the layer and head levels, offering control over the model's generation at varying levels of granularity. In addition, we explore three strategies for constructing positive and negative samples and two approaches for extracting revision vectors, resulting in different variants of our method. Comprehensive experiments demonstrate that the internal activation revision method significantly improves the safety of widely used VLMs, reducing attack success rates by an average of 48.94\%, 34.34\%, 43.92\%, and 52.98\% on SafeBench, Safe-Unsafe, Unsafe, and MM-SafetyBench, respectively, while minimally impacting model helpfulness.

摘要: 视觉语言模型(VLM)显示出强大的多模式能力，但与其主干大型语言模型(LLM)相比，已被发现更容易产生有害内容。我们的研究表明，在前向传递过程中，图像的整合显著改变了模型的内部激活，与文本输入触发的激活不同。此外，嵌入在VLM中的LLM的安全对齐不足以处理激活差异，使得这些模型即使是最简单的越狱攻击也很容易受到攻击。为了解决这个问题，我们提出了一种\extbf{内部激活修订}方法，该方法在生成过程中有效地修改激活，将模型引导到更安全的输出。我们的框架结合了在层和头部级别的修订，在不同的粒度级别提供了对模型生成的控制。此外，我们还探索了构建正负样本的三种策略和提取修订向量的两种方法，导致了我们方法的不同变体。综合实验表明，内部激活修正方法显著提高了广泛使用的VLM的安全性，在对模型的有用性影响最小的情况下，对SafeBitch、Safe-UnSafe、UnSafe和MM-SafetyBitch的攻击成功率分别平均降低了48.94、34.34、43.92和52.98。



## **24. Siren: A Learning-Based Multi-Turn Attack Framework for Simulating Real-World Human Jailbreak Behaviors**

Siren：一个基于学习的多回合攻击框架，用于模拟现实世界的人类越狱行为 cs.CL

**SubmitDate**: 2025-01-24    [abs](http://arxiv.org/abs/2501.14250v1) [paper-pdf](http://arxiv.org/pdf/2501.14250v1)

**Authors**: Yi Zhao, Youzhi Zhang

**Abstract**: Large language models (LLMs) are widely used in real-world applications, raising concerns about their safety and trustworthiness. While red-teaming with jailbreak prompts exposes the vulnerabilities of LLMs, current efforts focus primarily on single-turn attacks, overlooking the multi-turn strategies used by real-world adversaries. Existing multi-turn methods rely on static patterns or predefined logical chains, failing to account for the dynamic strategies during attacks. We propose Siren, a learning-based multi-turn attack framework designed to simulate real-world human jailbreak behaviors. Siren consists of three stages: (1) training set construction utilizing Turn-Level LLM feedback (Turn-MF), (2) post-training attackers with supervised fine-tuning (SFT) and direct preference optimization (DPO), and (3) interactions between the attacking and target LLMs. Experiments demonstrate that Siren achieves an attack success rate (ASR) of 90% with LLaMA-3-8B as the attacker against Gemini-1.5-Pro as the target model, and 70% with Mistral-7B against GPT-4o, significantly outperforming single-turn baselines. Moreover, Siren with a 7B-scale model achieves performance comparable to a multi-turn baseline that leverages GPT-4o as the attacker, while requiring fewer turns and employing decomposition strategies that are better semantically aligned with attack goals. We hope Siren inspires the development of stronger defenses against advanced multi-turn jailbreak attacks under realistic scenarios. Code is available at https://github.com/YiyiyiZhao/siren. Warning: This paper contains potentially harmful text.

摘要: 大型语言模型(LLM)在实际应用中被广泛使用，这引发了人们对它们的安全性和可信性的担忧。虽然与越狱提示的红色合作暴露了LLMS的漏洞，但目前的努力主要集中在单回合攻击上，而忽略了现实世界对手使用的多回合策略。现有的多回合攻击方法依赖于静态模式或预定义的逻辑链，无法考虑攻击过程中的动态策略。我们提出了一种基于学习的多回合攻击框架Siren，旨在模拟真实世界的人类越狱行为。SIREN包括三个阶段：(1)利用话轮水平LLM反馈(TURN-MF)构建训练集；(2)使用有监督的精调(SFT)和直接偏好优化(DPO)训练后攻击者；(3)攻击和目标LLM之间的交互。实验表明，SIREN在以骆驼-3-8B为攻击者对抗双子座-1.5-Pro为目标机型时，攻击成功率(ASR)为90%，在以米斯特拉尔-7B为攻击目标时，对GPT-40的攻击成功率为70%，显著超过了单回合基线。此外，拥有7B规模模型的SIREN实现了与利用GPT-40作为攻击者的多回合基线相当的性能，同时需要更少的回合，并采用了与攻击目标更好地语义一致的分解策略。我们希望警报器能在现实场景下启发开发更强大的防御系统来抵御高级多回合越狱攻击。代码可在https://github.com/YiyiyiZhao/siren.上找到警告：本文包含可能有害的文本。



## **25. GraphRAG under Fire**

GraphRAG受到攻击 cs.LG

13 pages

**SubmitDate**: 2025-01-23    [abs](http://arxiv.org/abs/2501.14050v1) [paper-pdf](http://arxiv.org/pdf/2501.14050v1)

**Authors**: Jiacheng Liang, Yuhui Wang, Changjiang Li, Rongyi Zhu, Tanqiu Jiang, Neil Gong, Ting Wang

**Abstract**: GraphRAG advances retrieval-augmented generation (RAG) by structuring external knowledge as multi-scale knowledge graphs, enabling language models to integrate both broad context and granular details in their reasoning. While GraphRAG has demonstrated success across domains, its security implications remain largely unexplored. To bridge this gap, this work examines GraphRAG's vulnerability to poisoning attacks, uncovering an intriguing security paradox: compared to conventional RAG, GraphRAG's graph-based indexing and retrieval enhance resilience against simple poisoning attacks; meanwhile, the same features also create new attack surfaces. We present GRAGPoison, a novel attack that exploits shared relations in the knowledge graph to craft poisoning text capable of compromising multiple queries simultaneously. GRAGPoison employs three key strategies: i) relation injection to introduce false knowledge, ii) relation enhancement to amplify poisoning influence, and iii) narrative generation to embed malicious content within coherent text. Empirical evaluation across diverse datasets and models shows that GRAGPoison substantially outperforms existing attacks in terms of effectiveness (up to 98% success rate) and scalability (using less than 68% poisoning text). We also explore potential defensive measures and their limitations, identifying promising directions for future research.

摘要: GraphRAG通过将外部知识构建为多尺度知识图来推进检索增强生成(RAG)，使语言模型能够在推理中集成广泛的上下文和细粒度的细节。虽然GraphRAG已经展示了跨域的成功，但其安全影响在很大程度上仍未得到探索。为了弥补这一差距，这项工作检查了GraphRAG对中毒攻击的脆弱性，揭示了一个有趣的安全悖论：与传统RAG相比，GraphRAG基于图形的索引和检索增强了对简单中毒攻击的弹性；同时，相同的功能也创建了新的攻击面。我们提出了一种新的攻击方法GRAGPoison，它利用知识图中的共享关系来创建能够同时危害多个查询的中毒文本。GRAGPoison采用了三个关键策略：i)关系注入引入虚假知识，ii)关系增强放大毒害影响，iii)叙事生成在连贯的文本中嵌入恶意内容。在不同的数据集和模型上的经验评估表明，GRAGPoison在有效性(高达98%的成功率)和可扩展性(使用不到68%的中毒文本)方面大大优于现有的攻击。我们还探讨了潜在的防御措施及其局限性，确定了未来研究的有希望的方向。



## **26. PhishAgent: A Robust Multimodal Agent for Phishing Webpage Detection**

PhishAgent：一种用于网络钓鱼网页检测的鲁棒多模式代理 cs.CR

Accepted at AAAI 2025 (Oral)

**SubmitDate**: 2025-01-23    [abs](http://arxiv.org/abs/2408.10738v3) [paper-pdf](http://arxiv.org/pdf/2408.10738v3)

**Authors**: Tri Cao, Chengyu Huang, Yuexin Li, Huilin Wang, Amy He, Nay Oo, Bryan Hooi

**Abstract**: Phishing attacks are a major threat to online security, exploiting user vulnerabilities to steal sensitive information. Various methods have been developed to counteract phishing, each with varying levels of accuracy, but they also face notable limitations. In this study, we introduce PhishAgent, a multimodal agent that combines a wide range of tools, integrating both online and offline knowledge bases with Multimodal Large Language Models (MLLMs). This combination leads to broader brand coverage, which enhances brand recognition and recall. Furthermore, we propose a multimodal information retrieval framework designed to extract the relevant top k items from offline knowledge bases, using available information from a webpage, including logos and HTML. Our empirical results, based on three real-world datasets, demonstrate that the proposed framework significantly enhances detection accuracy and reduces both false positives and false negatives, while maintaining model efficiency. Additionally, PhishAgent shows strong resilience against various types of adversarial attacks.

摘要: 网络钓鱼攻击是在线安全的主要威胁，利用用户漏洞窃取敏感信息。已经开发了各种方法来对抗网络钓鱼，每一种方法的精确度都不同，但它们也面临着显著的局限性。在本研究中，我们介绍了PhishAgent，一个结合了广泛工具的多通道代理，将线上和线下知识库与多通道大语言模型(MLLMS)相结合。这一组合导致了更广泛的品牌覆盖，从而提高了品牌认知度和召回率。此外，我们还提出了一个多通道信息检索框架，该框架利用网页中的可用信息，包括标识和超文本标记语言，从离线知识库中提取相关的前k个条目。基于三个真实数据集的实验结果表明，该框架在保持模型效率的同时，显著提高了检测准确率，减少了误报和漏报。此外，PhishAgent对各种类型的对抗性攻击表现出很强的韧性。



## **27. TrojanRobot: Physical-World Backdoor Attacks Against VLM-based Robotic Manipulation**

TrojanRobot：针对基于VLM的机器人操纵的物理世界后门攻击 cs.RO

**SubmitDate**: 2025-01-23    [abs](http://arxiv.org/abs/2411.11683v3) [paper-pdf](http://arxiv.org/pdf/2411.11683v3)

**Authors**: Xianlong Wang, Hewen Pan, Hangtao Zhang, Minghui Li, Shengshan Hu, Ziqi Zhou, Lulu Xue, Peijin Guo, Yichen Wang, Wei Wan, Aishan Liu, Leo Yu Zhang

**Abstract**: Robotic manipulation in the physical world is increasingly empowered by \textit{large language models} (LLMs) and \textit{vision-language models} (VLMs), leveraging their understanding and perception capabilities. Recently, various attacks against such robotic policies have been proposed, with backdoor attacks drawing considerable attention for their high stealth and strong persistence capabilities. However, existing backdoor efforts are limited to simulators and suffer from physical-world realization. To address this, we propose \textit{TrojanRobot}, a highly stealthy and broadly effective robotic backdoor attack in the physical world. Specifically, we introduce a module-poisoning approach by embedding a backdoor module into the modular robotic policy, enabling backdoor control over the policy's visual perception module thereby backdooring the entire robotic policy. Our vanilla implementation leverages a backdoor-finetuned VLM to serve as the backdoor module. To enhance its generalization in physical environments, we propose a prime implementation, leveraging the LVLM-as-a-backdoor paradigm and developing three types of prime attacks, \ie, \textit{permutation}, \textit{stagnation}, and \textit{intentional} attacks, thus achieving finer-grained backdoors. Extensive experiments on the UR3e manipulator with 18 task instructions using robotic policies based on four VLMs demonstrate the broad effectiveness and physical-world stealth of TrojanRobot. Our attack's video demonstrations are available via a github link \url{https://trojanrobot.github.io}.

摘要: 大型语言模型(LLMS)和视觉语言模型(VLMS)利用它们的理解和感知能力，使机器人在物理世界中的操作变得越来越强大。最近，针对这种机器人策略的各种攻击被提出，其中后门攻击因其高隐蔽性和强大的持久性而引起了相当大的关注。然而，现有的后门努力仅限于模拟器，并受到物理世界实现的影响。为了解决这个问题，我们提出了一种在物理世界中高度隐形且广泛有效的机器人后门攻击。具体地说，我们引入了模块中毒方法，将后门模块嵌入到模块化机器人策略中，使后门能够控制策略的视觉感知模块，从而后退整个机器人策略。我们的普通实现利用一个后门优化的VLM作为后门模块。为了增强其在物理环境中的普适性，我们提出了一个素数实现，利用了LVLM作为后门的范例，并开发了三种类型的素数攻击，即：文本{置换}、文本{停滞}和\文本{故意}攻击，从而实现了更细粒度的后门。在具有18条任务指令的UR3e机械手上，使用基于四个VLM的机器人策略进行了广泛的实验，证明了特洛伊机器人的广泛有效性和物理世界的隐蔽性。我们的攻击视频演示可通过GitHub链接\url{https://trojanrobot.github.io}.



## **28. HumorReject: Decoupling LLM Safety from Refusal Prefix via A Little Humor**

幽默：通过一点幽默将LLM安全与拒绝前置脱钩 cs.LG

**SubmitDate**: 2025-01-23    [abs](http://arxiv.org/abs/2501.13677v1) [paper-pdf](http://arxiv.org/pdf/2501.13677v1)

**Authors**: Zihui Wu, Haichang Gao, Jiacheng Luo, Zhaoxiang Liu

**Abstract**: Large Language Models (LLMs) commonly rely on explicit refusal prefixes for safety, making them vulnerable to prefix injection attacks. We introduce HumorReject, a novel data-driven approach that fundamentally reimagines LLM safety by decoupling it from refusal prefixes through the use of humor as an indirect refusal strategy. Rather than explicitly rejecting harmful instructions, HumorReject responds with contextually appropriate humor that naturally defuses potentially dangerous requests while maintaining engaging interactions. Our approach effectively addresses the common "over-defense" issues in existing safety mechanisms, demonstrating superior robustness against various attack vectors while preserving natural and high-quality interactions on legitimate tasks. Our findings suggest that innovations at the data level are even more fundamental than the alignment algorithm itself in achieving effective LLM safety, opening new directions for developing more resilient and user-friendly AI systems.

摘要: 大型语言模型(LLM)通常依赖显式拒绝前缀来确保安全，这使得它们容易受到前缀注入攻击。我们引入了HumorReject，这是一种新颖的数据驱动方法，通过使用幽默作为间接拒绝策略，从根本上将LLM安全与拒绝前缀分离。HumorReject没有明确拒绝有害的指令，而是用上下文适当的幽默来回应，这种幽默自然地化解了潜在的危险请求，同时保持了引人入胜的互动。我们的方法有效地解决了现有安全机制中常见的“过度防御”问题，在保持合法任务的自然和高质量交互的同时，对各种攻击载体表现出了卓越的稳健性。我们的发现表明，在实现有效的LLM安全方面，数据层面的创新甚至比对齐算法本身更根本，为开发更具弹性和用户友好的人工智能系统开辟了新的方向。



## **29. Black-Box Adversarial Attack on Vision Language Models for Autonomous Driving**

自动驾驶视觉语言模型的黑匣子对抗攻击 cs.CV

**SubmitDate**: 2025-01-23    [abs](http://arxiv.org/abs/2501.13563v1) [paper-pdf](http://arxiv.org/pdf/2501.13563v1)

**Authors**: Lu Wang, Tianyuan Zhang, Yang Qu, Siyuan Liang, Yuwei Chen, Aishan Liu, Xianglong Liu, Dacheng Tao

**Abstract**: Vision-language models (VLMs) have significantly advanced autonomous driving (AD) by enhancing reasoning capabilities; however, these models remain highly susceptible to adversarial attacks. While existing research has explored white-box attacks to some extent, the more practical and challenging black-box scenarios remain largely underexplored due to their inherent difficulty. In this paper, we take the first step toward designing black-box adversarial attacks specifically targeting VLMs in AD. We identify two key challenges for achieving effective black-box attacks in this context: the effectiveness across driving reasoning chains in AD systems and the dynamic nature of driving scenarios. To address this, we propose Cascading Adversarial Disruption (CAD). It first introduces Decision Chain Disruption, which targets low-level reasoning breakdown by generating and injecting deceptive semantics, ensuring the perturbations remain effective across the entire decision-making chain. Building on this, we present Risky Scene Induction, which addresses dynamic adaptation by leveraging a surrogate VLM to understand and construct high-level risky scenarios that are likely to result in critical errors in the current driving contexts. Extensive experiments conducted on multiple AD VLMs and benchmarks demonstrate that CAD achieves state-of-the-art attack effectiveness, significantly outperforming existing methods (+13.43% on average). Moreover, we validate its practical applicability through real-world attacks on AD vehicles powered by VLMs, where the route completion rate drops by 61.11% and the vehicle crashes directly into the obstacle vehicle with adversarial patches. Finally, we release CADA dataset, comprising 18,808 adversarial visual-question-answer pairs, to facilitate further evaluation and research in this critical domain. Our codes and dataset will be available after paper's acceptance.

摘要: 视觉语言模型通过增强推理能力极大地促进了自动驾驶(AD)的发展，但这些模型仍然高度容易受到对手的攻击。虽然现有的研究在一定程度上探索了白盒攻击，但由于其固有的困难，更实用和更具挑战性的黑盒场景在很大程度上仍然没有得到充分的探索。在本文中，我们向设计专门针对AD中的VLM的黑盒对抗性攻击迈出了第一步。在此背景下，我们确定了实现有效的黑盒攻击的两个关键挑战：在AD系统中跨驾驶推理链的有效性和驾驶场景的动态性质。为了解决这个问题，我们提出了级联对抗中断(CAD)。它首先引入决策链中断，通过生成和注入欺骗性语义来针对低级别推理故障，确保扰动在整个决策链中保持有效。在此基础上，我们提出了风险场景归纳，它通过利用代理VLM来理解和构建可能在当前驾驶环境中导致关键错误的高级别风险场景，从而解决动态适应问题。在多个AD VLM和基准上进行的广泛实验表明，CAD实现了最先进的攻击效率，显著优于现有方法(平均+13.43%)。此外，通过对VLMS驱动的AD车辆的实际攻击，路径完成率下降了61.11%，车辆直接撞上了带有对抗性补丁的障碍车辆，验证了该算法的实用性。最后，我们发布了包含18,808个对抗性视觉-问答对的CADA数据集，以便于在这一关键领域进行进一步的评估和研究。我们的代码和数据集将在论文验收后可用。



## **30. Not all tokens are created equal: Perplexity Attention Weighted Networks for AI generated text detection**

并非所有令牌都是平等的：用于人工智能生成文本检测的困惑注意力加权网络 cs.CL

**SubmitDate**: 2025-01-22    [abs](http://arxiv.org/abs/2501.03940v2) [paper-pdf](http://arxiv.org/pdf/2501.03940v2)

**Authors**: Pablo Miralles-González, Javier Huertas-Tato, Alejandro Martín, David Camacho

**Abstract**: The rapid advancement in large language models (LLMs) has significantly enhanced their ability to generate coherent and contextually relevant text, raising concerns about the misuse of AI-generated content and making it critical to detect it. However, the task remains challenging, particularly in unseen domains or with unfamiliar LLMs. Leveraging LLM next-token distribution outputs offers a theoretically appealing approach for detection, as they encapsulate insights from the models' extensive pre-training on diverse corpora. Despite its promise, zero-shot methods that attempt to operationalize these outputs have met with limited success. We hypothesize that one of the problems is that they use the mean to aggregate next-token distribution metrics across tokens, when some tokens are naturally easier or harder to predict and should be weighted differently. Based on this idea, we propose the Perplexity Attention Weighted Network (PAWN), which uses the last hidden states of the LLM and positions to weight the sum of a series of features based on metrics from the next-token distribution across the sequence length. Although not zero-shot, our method allows us to cache the last hidden states and next-token distribution metrics on disk, greatly reducing the training resource requirements. PAWN shows competitive and even better performance in-distribution than the strongest baselines (fine-tuned LMs) with a fraction of their trainable parameters. Our model also generalizes better to unseen domains and source models, with smaller variability in the decision boundary across distribution shifts. It is also more robust to adversarial attacks, and if the backbone has multilingual capabilities, it presents decent generalization to languages not seen during supervised training, with LLaMA3-1B reaching a mean macro-averaged F1 score of 81.46% in cross-validation with nine languages.

摘要: 大型语言模型(LLM)的快速发展极大地增强了它们生成连贯和上下文相关文本的能力，这引起了人们对滥用人工智能生成的内容的担忧，并使检测它变得至关重要。然而，这项任务仍然具有挑战性，特别是在看不见的领域或具有不熟悉的LLM的领域。利用LLM下一个令牌分发输出提供了一种理论上有吸引力的检测方法，因为它们概括了模型对不同语料库的广泛预培训的见解。尽管有希望，但试图将这些产出付诸实施的零射击方法却取得了有限的成功。我们假设其中一个问题是，当一些令牌自然更容易或更难预测，并且应该以不同的权重进行加权时，它们使用平均值来聚合跨令牌的下一令牌分发度量。基于这一思想，我们提出了困惑注意力加权网络(PAWN)，它利用LLM的最后一个隐藏状态和位置来加权一系列特征的和，基于下一个令牌分布在整个序列长度上的度量。虽然不是零命中率，但我们的方法允许我们在磁盘上缓存最后的隐藏状态和下一个令牌分布度量，大大减少了训练资源需求。与最强的基线(微调LMS)相比，PAWN显示出具有竞争力的分布性能，甚至比它们的可训练参数的一小部分更好。我们的模型也更好地推广到看不见的域和源模型，跨分布转变的决策边界的可变性较小。LLaMA3-1B在与9种语言的交叉验证中达到了81.46%的平均宏观平均F1分数。



## **31. An Empirically-grounded tool for Automatic Prompt Linting and Repair: A Case Study on Bias, Vulnerability, and Optimization in Developer Prompts**

一个基于经验的自动提示衬里和修复工具：开发人员预算中的偏差、漏洞和优化案例研究 cs.SE

**SubmitDate**: 2025-01-21    [abs](http://arxiv.org/abs/2501.12521v1) [paper-pdf](http://arxiv.org/pdf/2501.12521v1)

**Authors**: Dhia Elhaq Rzig, Dhruba Jyoti Paul, Kaiser Pister, Jordan Henkel, Foyzul Hassan

**Abstract**: The tidal wave of advancements in Large Language Models (LLMs) has led to their swift integration into application-level logic. Many software systems now use prompts to interact with these black-box models, combining natural language with dynamic values interpolated at runtime, to perform tasks ranging from sentiment analysis to question answering. Due to the programmatic and structured natural language aspects of these prompts, we refer to them as Developer Prompts. Unlike traditional software artifacts, Dev Prompts blend natural language instructions with artificial languages such as programming and markup languages, thus requiring specialized tools for analysis, distinct from classical software evaluation methods.   In response to this need, we introduce PromptDoctor, a tool explicitly designed to detect and correct issues of Dev Prompts. PromptDoctor identifies and addresses problems related to bias, vulnerability, and sub-optimal performance in Dev Prompts, helping mitigate their possible harms. In our analysis of 2,173 Dev Prompts, selected as a representative sample of 40,573 Dev Prompts, we found that 3.46% contained one or more forms of bias, 10.75% were vulnerable to prompt injection attacks. Additionally, 3,310 were amenable to automated prompt optimization. To address these issues, we applied PromptDoctor to the flawed Dev Prompts we discovered. PromptDoctor de-biased 68.29% of the biased Dev Prompts, hardened 41.81% of the vulnerable Dev Prompts, and improved the performance of 37.1% sub-optimal Dev Prompts. Finally, we developed a PromptDoctor VSCode extension, enabling developers to easily enhance Dev Prompts in their existing development workflows. The data and source code for this work are available at

摘要: 大型语言模型(LLM)的发展浪潮导致它们迅速集成到应用程序级逻辑中。许多软件系统现在使用提示与这些黑盒模型交互，将自然语言与运行时内插的动态值相结合，以执行从情绪分析到问题回答的各种任务。由于这些提示的程序性和结构化的自然语言方面，我们将其称为开发人员提示。与传统的软件构件不同，开发提示将自然语言指令与编程和标记语言等人工语言混合在一起，因此需要专门的工具进行分析，这与传统的软件评估方法不同。为了响应这一需求，我们引入了PromptDoctor，这是一个专门设计用于检测和纠正开发提示问题的工具。PromptDoctor识别并解决与开发提示中的偏差、漏洞和次优性能相关的问题，帮助减轻它们可能造成的危害。在我们对2,173个Dev Prompt的分析中，我们选择了40,573个Dev Prompt的代表性样本，发现3.46%的Dev Prompt包含一种或多种形式的偏见，10.75%的Dev Prompt容易受到即时注入攻击。此外，3,310个适用于自动即时优化。为了解决这些问题，我们将PromptDoctor应用于我们发现的有缺陷的开发提示。PromptDoctor消除了68.29%的偏向开发提示，强化了41.81%的易受攻击的开发提示，并提高了37.1%次优开发提示的性能。最后，我们开发了一个PromptDoctor VSCode扩展，使开发人员能够在其现有的开发工作流中轻松增强Dev Prompt。这项工作的数据和源代码可在以下网址获得



## **32. You Can't Eat Your Cake and Have It Too: The Performance Degradation of LLMs with Jailbreak Defense**

你不能既吃又拥有蛋糕：具有越狱防御的法学硕士的表现下降 cs.CR

**SubmitDate**: 2025-01-21    [abs](http://arxiv.org/abs/2501.12210v1) [paper-pdf](http://arxiv.org/pdf/2501.12210v1)

**Authors**: Wuyuao Mai, Geng Hong, Pei Chen, Xudong Pan, Baojun Liu, Yuan Zhang, Haixin Duan, Min Yang

**Abstract**: With the rise of generative large language models (LLMs) like LLaMA and ChatGPT, these models have significantly transformed daily life and work by providing advanced insights. However, as jailbreak attacks continue to circumvent built-in safety mechanisms, exploiting carefully crafted scenarios or tokens, the safety risks of LLMs have come into focus. While numerous defense strategies--such as prompt detection, modification, and model fine-tuning--have been proposed to counter these attacks, a critical question arises: do these defenses compromise the utility and usability of LLMs for legitimate users? Existing research predominantly focuses on the effectiveness of defense strategies without thoroughly examining their impact on performance, leaving a gap in understanding the trade-offs between LLM safety and performance. Our research addresses this gap by conducting a comprehensive study on the utility degradation, safety elevation, and exaggerated-safety escalation of LLMs with jailbreak defense strategies. We propose USEBench, a novel benchmark designed to evaluate these aspects, along with USEIndex, a comprehensive metric for assessing overall model performance. Through experiments on seven state-of-the-art LLMs, we found that mainstream jailbreak defenses fail to ensure both safety and performance simultaneously. Although model-finetuning performs the best overall, their effectiveness varies across LLMs. Furthermore, vertical comparisons reveal that developers commonly prioritize performance over safety when iterating or fine-tuning their LLMs.

摘要: 随着像骆驼和ChatGPT这样的生成性大型语言模型(LLM)的兴起，这些模型通过提供先进的见解显著地改变了日常生活和工作。然而，随着越狱攻击继续绕过内置的安全机制，利用精心设计的场景或令牌，低密度脂蛋白的安全风险成为关注的焦点。虽然已经提出了许多防御策略--如快速检测、修改和模型微调--来对抗这些攻击，但一个关键的问题出现了：这些防御是否会损害LLM对合法用户的实用性和可用性？现有的研究主要集中在防御策略的有效性上，而没有彻底检查它们对性能的影响，从而在理解LLM安全和性能之间的权衡方面留下了空白。我们的研究通过对具有越狱防御策略的低密度脂蛋白的效用降级、安全提升和夸大安全升级进行了全面的研究，解决了这一差距。我们提出了一种用于评估这些方面的新基准USEBENCH，以及一种用于评估整体模型性能的综合指标USEIndex。通过对7个最先进的LLM进行实验，我们发现主流的越狱防御无法同时确保安全和性能。尽管模型优化总体上执行得最好，但它们的有效性在不同的LLM中有所不同。此外，纵向比较显示，在迭代或微调LLM时，开发人员通常优先考虑性能而不是安全性。



## **33. Phishing Awareness via Game-Based Learning**

通过基于游戏的学习提高网络钓鱼意识 cs.CR

37th International Conference on Software Engineering Education and  Training (CSEET 2025)

**SubmitDate**: 2025-01-21    [abs](http://arxiv.org/abs/2501.12077v1) [paper-pdf](http://arxiv.org/pdf/2501.12077v1)

**Authors**: Argianto Rahartomo, Ahmed Tareq Ali Ghaleb, Mohammad Ghafari

**Abstract**: The increased use of digital devices and applications has led to a rise in phishing attacks. We develop a serious game to raise awareness about phishing attacks and help people avoid these threats in a risk-free learning environment. This game targets three types of phishing-clone phishing, SMS phishing, and spear phishing-and uses a Large Language Model to generate dialogues and questions dynamically. It also incorporates state randomization and time-limited challenges to enhance the gameplay. We evaluated two groups of participants and found that those who played the game showed, on average, a 24% increase in awareness and a 30% boost in confidence.

摘要: 数字设备和应用程序使用的增加导致网络钓鱼攻击的增加。我们开发了一款严肃的游戏，以提高人们对网络钓鱼攻击的认识，并帮助人们在无风险的学习环境中避免这些威胁。该游戏针对三种类型的网络钓鱼-克隆网络钓鱼、短信网络钓鱼和鱼叉网络钓鱼-并使用大型语言模型动态生成对话和问题。它还结合了州随机化和限时挑战来增强游戏玩法。我们评估了两组参与者，发现玩游戏的人的意识平均提高了24%，信心提高了30%。



## **34. QROA: A Black-Box Query-Response Optimization Attack on LLMs**

QROA：对LLM的黑匣子查询响应优化攻击 cs.CL

**SubmitDate**: 2025-01-21    [abs](http://arxiv.org/abs/2406.02044v2) [paper-pdf](http://arxiv.org/pdf/2406.02044v2)

**Authors**: Hussein Jawad, Nicolas J. -B. BRUNEL

**Abstract**: Large Language Models (LLMs) have surged in popularity in recent months, yet they possess concerning capabilities for generating harmful content when manipulated. This study introduces the Query-Response Optimization Attack (QROA), an optimization-based strategy designed to exploit LLMs through a black-box, query-only interaction. QROA adds an optimized trigger to a malicious instruction to compel the LLM to generate harmful content. Unlike previous approaches, QROA does not require access to the model's logit information or any other internal data and operates solely through the standard query-response interface of LLMs. Inspired by deep Q-learning and Greedy coordinate descent, the method iteratively updates tokens to maximize a designed reward function. We tested our method on various LLMs such as Vicuna, Falcon, and Mistral, achieving an Attack Success Rate (ASR) over 80\%. We also tested the model against Llama2-chat, the fine-tuned version of Llama2 designed to resist Jailbreak attacks, achieving good ASR with a suboptimal initial trigger seed. This study demonstrates the feasibility of generating jailbreak attacks against deployed LLMs in the public domain using black-box optimization methods, enabling more comprehensive safety testing of LLMs.

摘要: 近几个月来，大型语言模型(LLM)越来越受欢迎，但它们具有在被操纵时生成有害内容的令人担忧的能力。这项研究介绍了查询-响应优化攻击(QROA)，这是一种基于优化的策略，旨在通过黑盒、仅查询的交互来利用LLMS。QROA向恶意指令添加了优化的触发器，以迫使LLM生成有害内容。与以前的方法不同，QROA不需要访问模型的Logit信息或任何其他内部数据，只通过LLMS的标准查询-响应接口进行操作。受深度Q学习和贪婪坐标下降的启发，该方法迭代更新令牌以最大化所设计的奖励函数。我们在维库纳、猎鹰和米斯特拉尔等不同的LLMS上测试了我们的方法，取得了80%以上的攻击成功率(ASR)。我们还在Llama2-Chat上测试了该模型，Llama2-Chat是Llama2的微调版本，旨在抵抗越狱攻击，使用次优的初始触发种子实现了良好的ASR。这项研究论证了利用黑盒优化方法对部署在公共领域的LLM进行越狱攻击的可行性，从而实现了对LLM进行更全面的安全测试。



## **35. Synthetic Data Can Mislead Evaluations: Membership Inference as Machine Text Detection**

合成数据可能会误导评估：隶属推理作为机器文本检测 cs.CL

**SubmitDate**: 2025-01-20    [abs](http://arxiv.org/abs/2501.11786v1) [paper-pdf](http://arxiv.org/pdf/2501.11786v1)

**Authors**: Ali Naseh, Niloofar Mireshghallah

**Abstract**: Recent work shows membership inference attacks (MIAs) on large language models (LLMs) produce inconclusive results, partly due to difficulties in creating non-member datasets without temporal shifts. While researchers have turned to synthetic data as an alternative, we show this approach can be fundamentally misleading. Our experiments indicate that MIAs function as machine-generated text detectors, incorrectly identifying synthetic data as training samples regardless of the data source. This behavior persists across different model architectures and sizes, from open-source models to commercial ones such as GPT-3.5. Even synthetic text generated by different, potentially larger models is classified as training data by the target model. Our findings highlight a serious concern: using synthetic data in membership evaluations may lead to false conclusions about model memorization and data leakage. We caution that this issue could affect other evaluations using model signals such as loss where synthetic or machine-generated translated data substitutes for real-world samples.

摘要: 最近的工作表明，对大型语言模型(LLM)的成员关系推理攻击(MIA)会产生不确定的结果，部分原因是在没有时间漂移的情况下创建非成员数据集的困难。虽然研究人员已经将合成数据作为一种选择，但我们表明，这种方法可能从根本上具有误导性。我们的实验表明，MIA作为机器生成的文本检测器，无论数据源如何，都会错误地将合成数据识别为训练样本。这种行为在不同的模型体系结构和大小中持续存在，从开源模型到商业模型，如GPT-3.5。即使是由不同的、可能更大的模型生成的合成文本也被目标模型分类为训练数据。我们的发现突出了一个严重的问题：在成员评估中使用合成数据可能会导致关于模型记忆和数据泄漏的错误结论。我们警告说，这个问题可能会影响使用模型信号的其他评估，例如在人工或机器生成的翻译数据取代真实世界样本的情况下的损失。



## **36. Poison-RAG: Adversarial Data Poisoning Attacks on Retrieval-Augmented Generation in Recommender Systems**

Poison-RAG：对推荐系统中检索增强生成的对抗数据中毒攻击 cs.IR

**SubmitDate**: 2025-01-20    [abs](http://arxiv.org/abs/2501.11759v1) [paper-pdf](http://arxiv.org/pdf/2501.11759v1)

**Authors**: Fatemeh Nazary, Yashar Deldjoo, Tommaso di Noia

**Abstract**: This study presents Poison-RAG, a framework for adversarial data poisoning attacks targeting retrieval-augmented generation (RAG)-based recommender systems. Poison-RAG manipulates item metadata, such as tags and descriptions, to influence recommendation outcomes. Using item metadata generated through a large language model (LLM) and embeddings derived via the OpenAI API, we explore the impact of adversarial poisoning attacks on provider-side, where attacks are designed to promote long-tail items and demote popular ones. Two attack strategies are proposed: local modifications, which personalize tags for each item using BERT embeddings, and global modifications, applying uniform tags across the dataset. Experiments conducted on the MovieLens dataset in a black-box setting reveal that local strategies improve manipulation effectiveness by up to 50\%, while global strategies risk boosting already popular items. Results indicate that popular items are more susceptible to attacks, whereas long-tail items are harder to manipulate. Approximately 70\% of items lack tags, presenting a cold-start challenge; data augmentation and synthesis are proposed as potential defense mechanisms to enhance RAG-based systems' resilience. The findings emphasize the need for robust metadata management to safeguard recommendation frameworks. Code and data are available at https://github.com/atenanaz/Poison-RAG.

摘要: 提出了一种针对基于检索增强生成(RAG)的推荐系统的对抗性数据中毒攻击框架Poison-RAG。毒布操纵项目元数据，如标签和描述，以影响推荐结果。使用通过大型语言模型(LLM)生成的项元数据和通过OpenAI API派生的嵌入，我们探索了对抗性中毒攻击对提供商端的影响，其中攻击旨在提升长尾项目和降级流行项目。提出了两种攻击策略：局部修改和全局修改，前者使用BERT嵌入对每个项目的标签进行个性化处理，后者将统一的标签应用于整个数据集。在黑盒设置的MovieLens数据集上进行的实验表明，局部策略将操作效率提高了高达50%，而全局策略则有可能提高已经很受欢迎的项目。结果表明，受欢迎的项目更容易受到攻击，而长尾项目更难操纵。大约70%的条目没有标签，这是一个冷启动的挑战；数据增强和合成被提出作为潜在的防御机制来增强基于RAG的系统的弹性。调查结果强调需要强有力的元数据管理来保障建议框架。有关代码和数据，请访问https://github.com/atenanaz/Poison-RAG.



## **37. A Practical Examination of AI-Generated Text Detectors for Large Language Models**

大型语言模型的人工智能生成文本检测器的实践检验 cs.CL

8 pages

**SubmitDate**: 2025-01-20    [abs](http://arxiv.org/abs/2412.05139v3) [paper-pdf](http://arxiv.org/pdf/2412.05139v3)

**Authors**: Brian Tufts, Xuandong Zhao, Lei Li

**Abstract**: The proliferation of large language models has raised growing concerns about their misuse, particularly in cases where AI-generated text is falsely attributed to human authors. Machine-generated content detectors claim to effectively identify such text under various conditions and from any language model. This paper critically evaluates these claims by assessing several popular detectors (RADAR, Wild, T5Sentinel, Fast-DetectGPT, GPTID, LogRank, Binoculars) on a range of domains, datasets, and models that these detectors have not previously encountered. We employ various prompting strategies to simulate adversarial attacks, demonstrating that even moderate efforts can significantly evade detection. We emphasize the importance of the true positive rate at a specific false positive rate (TPR@FPR) metric and demonstrate that these detectors perform poorly in certain settings, with TPR@.01 as low as 0%. Our findings suggest that both trained and zero-shot detectors struggle to maintain high sensitivity while achieving a reasonable true positive rate.

摘要: 大型语言模型的激增引发了人们对它们滥用的日益担忧，特别是在人工智能生成的文本被错误地归因于人类作者的情况下。机器生成的内容检测器声称可以在各种条件下从任何语言模型有效地识别此类文本。本文通过评估几种流行的探测器(雷达、Wild、T5Sentinel、Fast-DetectGPT、GPTID、logrank、双筒望远镜)，对这些声称进行了批判性的评估，这些探测器以前从未遇到过。我们使用各种提示策略来模拟对抗性攻击，表明即使是适度的攻击也可以显著地躲避检测。我们强调了在特定的假阳性率(TPR@fPR)度量下的真阳性率的重要性，并证明了这些检测器在某些设置下表现很差，TPR@.01低至0%。我们的发现表明，训练有素的探测器和零射探测器都很难在保持高灵敏度的同时获得合理的真阳性率。



## **38. Navigating the Designs of Privacy-Preserving Fine-tuning for Large Language Models**

引导大型语言模型的隐私保护微调设计 cs.LG

Accepted to WWW 2025

**SubmitDate**: 2025-01-20    [abs](http://arxiv.org/abs/2501.04323v3) [paper-pdf](http://arxiv.org/pdf/2501.04323v3)

**Authors**: Haonan Shi, Tu Ouyang, An Wang

**Abstract**: Instruction tuning has proven effective in enhancing Large Language Models' (LLMs) performance on downstream tasks. However, real-world fine-tuning faces inherent conflicts between model providers' intellectual property protection, clients' data privacy requirements, and tuning costs. While recent approaches like split learning and offsite tuning demonstrate promising architectures for privacy-preserving fine-tuning, there is a gap in systematically addressing the multidimensional trade-offs required for diverse real-world deployments. We propose several indicative evaluation metrics to guide design trade-offs for privacy-preserving fine-tuning and a series of example designs, collectively named GuardedTuning; they result from novel combinations of system architectures with adapted privacy-enhancement methods and emerging computation techniques. Each design represents distinct trade-offs across model utility, privacy guarantees, and costs. Experimental results demonstrate that these designs protect against data reconstruction attacks while maintaining competitive fine-tuning performance.

摘要: 指令调优在提高大型语言模型(LLM)在下游任务上的性能方面已被证明是有效的。然而，现实世界的微调面临着模型提供商的知识产权保护、客户的数据隐私要求和调整成本之间的内在冲突。虽然最近的方法，如拆分学习和异地调整，展示了保护隐私的微调的有前途的架构，但在系统地解决不同现实世界部署所需的多维权衡方面存在差距。我们提出了几个指示性的评估指标来指导隐私保护微调和一系列示例设计的权衡，统称为GuardedTuning；它们是系统架构与适应隐私增强方法和新兴计算技术的新颖组合的结果。每一种设计都代表着在模型效用、隐私保障和成本方面的不同权衡。实验结果表明，这些设计在保持具有竞争力的微调性能的同时，能够抵御数据重构攻击。



## **39. SilverSpeak: Evading AI-Generated Text Detectors using Homoglyphs**

SilverSpeak：使用同字形躲避人工智能生成的文本检测器 cs.CL

Workshop on Detecting AI Generated Content at COLING 2025

**SubmitDate**: 2025-01-20    [abs](http://arxiv.org/abs/2406.11239v3) [paper-pdf](http://arxiv.org/pdf/2406.11239v3)

**Authors**: Aldan Creo, Shushanta Pudasaini

**Abstract**: The advent of Large Language Models (LLMs) has enabled the generation of text that increasingly exhibits human-like characteristics. As the detection of such content is of significant importance, substantial research has been conducted with the objective of developing reliable AI-generated text detectors. These detectors have demonstrated promising results on test data, but recent research has revealed that they can be circumvented by employing different techniques.   In this paper, we present homoglyph-based attacks (A $\rightarrow$ Cyrillic A) as a means of circumventing existing detectors. We conduct a comprehensive evaluation to assess the effectiveness of these attacks on seven detectors, including ArguGPT, Binoculars, DetectGPT, Fast-DetectGPT, Ghostbuster, OpenAI's detector, and watermarking techniques, on five different datasets. Our findings demonstrate that homoglyph-based attacks can effectively circumvent state-of-the-art detectors, leading them to classify all texts as either AI-generated or human-written (decreasing the average Matthews Correlation Coefficient from 0.64 to -0.01). Through further examination, we extract the technical justification underlying the success of the attacks, which varies across detectors. Finally, we discuss the implications of these findings and potential defenses against such attacks.

摘要: 大型语言模型(LLM)的出现使得文本的生成越来越显示出与人类相似的特征。由于对这类内容的检测非常重要，因此进行了大量研究，目的是开发可靠的人工智能生成的文本检测器。这些探测器在测试数据上显示了有希望的结果，但最近的研究表明，可以通过使用不同的技术来绕过它们。在这篇文章中，我们提出了基于同形文字的攻击(A$\right tarrow$Cyrillic A)作为一种绕过现有检测器的手段。我们在五个不同的数据集上对七个检测器进行了全面的评估，包括ArguGPT、双筒望远镜、DetectGPT、Fast-DetectGPT、Ghost Buster、OpenAI的检测器和水印技术。我们的发现表明，基于同种文字的攻击可以有效地绕过最先进的检测器，导致他们将所有文本分类为人工生成的或人类编写的(将平均Matthews相关系数从0.64降低到-0.01)。通过进一步的检查，我们提取了攻击成功背后的技术理由，这一点在不同的探测器上有所不同。最后，我们讨论了这些发现的含义和针对此类攻击的潜在防御措施。



## **40. Deciphering the Chaos: Enhancing Jailbreak Attacks via Adversarial Prompt Translation**

破译混乱：通过对抗性提示翻译增强越狱攻击 cs.LG

**SubmitDate**: 2025-01-20    [abs](http://arxiv.org/abs/2410.11317v2) [paper-pdf](http://arxiv.org/pdf/2410.11317v2)

**Authors**: Qizhang Li, Xiaochen Yang, Wangmeng Zuo, Yiwen Guo

**Abstract**: Automatic adversarial prompt generation provides remarkable success in jailbreaking safely-aligned large language models (LLMs). Existing gradient-based attacks, while demonstrating outstanding performance in jailbreaking white-box LLMs, often generate garbled adversarial prompts with chaotic appearance. These adversarial prompts are difficult to transfer to other LLMs, hindering their performance in attacking unknown victim models. In this paper, for the first time, we delve into the semantic meaning embedded in garbled adversarial prompts and propose a novel method that "translates" them into coherent and human-readable natural language adversarial prompts. In this way, we can effectively uncover the semantic information that triggers vulnerabilities of the model and unambiguously transfer it to the victim model, without overlooking the adversarial information hidden in the garbled text, to enhance jailbreak attacks. It also offers a new approach to discovering effective designs for jailbreak prompts, advancing the understanding of jailbreak attacks. Experimental results demonstrate that our method significantly improves the success rate of jailbreak attacks against various safety-aligned LLMs and outperforms state-of-the-arts by large margins. With at most 10 queries, our method achieves an average attack success rate of 81.8% in attacking 7 commercial closed-source LLMs, including GPT and Claude-3 series, on HarmBench. Our method also achieves over 90% attack success rates against Llama-2-Chat models on AdvBench, despite their outstanding resistance to jailbreak attacks. Code at: https://github.com/qizhangli/Adversarial-Prompt-Translator.

摘要: 自动对抗性提示生成在越狱安全对齐的大型语言模型(LLM)方面取得了显着的成功。现有的基于梯度的攻击虽然在越狱白盒LLM中表现出出色的性能，但往往会产生外观混乱的乱码对抗性提示。这些对抗性提示很难转移到其他LLM上，阻碍了它们在攻击未知受害者模型时的表现。在本文中，我们首次深入研究了混淆的对抗性提示中所蕴含的语义，并提出了一种新的方法，将它们“翻译”成连贯的、人类可读的自然语言对抗性提示。通过这种方式，我们可以有效地发现触发模型漏洞的语义信息，并毫不含糊地将其传递给受害者模型，而不会忽视隐藏在乱码文本中的对抗性信息，以增强越狱攻击。它还提供了一种新的方法来发现有效的越狱提示设计，促进了对越狱攻击的理解。实验结果表明，我们的方法显著提高了对各种安全对齐LLM的越狱攻击成功率，并且远远超过了最新的技术水平。在最多10个查询的情况下，我们的方法在HarmBch上攻击包括GPT和Claude-3系列在内的7个商业闭源LLM，平均攻击成功率为81.8%。我们的方法对AdvBtch上的Llama-2-Chat模型的攻击成功率也达到了90%以上，尽管它们对越狱攻击具有出色的抵抗力。代码：https://github.com/qizhangli/Adversarial-Prompt-Translator.



## **41. FigStep: Jailbreaking Large Vision-Language Models via Typographic Visual Prompts**

FigStep：通过印刷视觉预设破解大型视觉语言模型 cs.CR

AAAI 2025 (Oral)

**SubmitDate**: 2025-01-19    [abs](http://arxiv.org/abs/2311.05608v3) [paper-pdf](http://arxiv.org/pdf/2311.05608v3)

**Authors**: Yichen Gong, Delong Ran, Jinyuan Liu, Conglei Wang, Tianshuo Cong, Anyu Wang, Sisi Duan, Xiaoyun Wang

**Abstract**: Large Vision-Language Models (LVLMs) signify a groundbreaking paradigm shift within the Artificial Intelligence (AI) community, extending beyond the capabilities of Large Language Models (LLMs) by assimilating additional modalities (e.g., images). Despite this advancement, the safety of LVLMs remains adequately underexplored, with a potential overreliance on the safety assurances purported by their underlying LLMs. In this paper, we propose FigStep, a straightforward yet effective black-box jailbreak algorithm against LVLMs. Instead of feeding textual harmful instructions directly, FigStep converts the prohibited content into images through typography to bypass the safety alignment. The experimental results indicate that FigStep can achieve an average attack success rate of 82.50% on six promising open-source LVLMs. Not merely to demonstrate the efficacy of FigStep, we conduct comprehensive ablation studies and analyze the distribution of the semantic embeddings to uncover that the reason behind the success of FigStep is the deficiency of safety alignment for visual embeddings. Moreover, we compare FigStep with five text-only jailbreaks and four image-based jailbreaks to demonstrate the superiority of FigStep, i.e., negligible attack costs and better attack performance. Above all, our work reveals that current LVLMs are vulnerable to jailbreak attacks, which highlights the necessity of novel cross-modality safety alignment techniques. Our code and datasets are available at https://github.com/ThuCCSLab/FigStep .

摘要: 大型视觉语言模型(LVMs)意味着人工智能(AI)社区内的一次突破性的范式转变，通过吸收其他形式(如图像)，扩展到大型语言模型(LLM)的能力之外。尽管取得了这一进展，但低层LMS的安全性仍未得到充分开发，潜在地过度依赖其潜在的LLM声称的安全保证。在本文中，我们提出了一种简单有效的针对LVLMS的黑盒越狱算法FigStep。FigStep没有直接提供文本有害指令，而是通过排版将被禁止的内容转换为图像，以绕过安全对齐。实验结果表明，FigStep在6个有前景的开源LVLMS上的平均攻击成功率为82.50%。我们不仅为了证明FigStep的有效性，还进行了全面的消融研究，分析了语义嵌入的分布情况，发现FigStep成功的原因是视觉嵌入的安全对齐不足。此外，我们将FigStep与五个纯文本越狱和四个基于图像的越狱进行了比较，证明了FigStep的优越性，即攻击成本可以忽略不计，攻击性能更好。最重要的是，我们的工作揭示了当前的LVLM容易受到越狱攻击，这突显了新的跨通道安全对齐技术的必要性。我们的代码和数据集可以在https://github.com/ThuCCSLab/FigStep上找到。



## **42. FlipedRAG: Black-Box Opinion Manipulation Attacks to Retrieval-Augmented Generation of Large Language Models**

FlipedRAG：对大型语言模型的检索增强生成的黑匣子观点操纵攻击 cs.IR

arXiv admin note: text overlap with arXiv:2407.13757

**SubmitDate**: 2025-01-19    [abs](http://arxiv.org/abs/2501.02968v2) [paper-pdf](http://arxiv.org/pdf/2501.02968v2)

**Authors**: Zhuo Chen, Yuyang Gong, Miaokun Chen, Haotan Liu, Qikai Cheng, Fan Zhang, Wei Lu, Xiaozhong Liu, Jiawei Liu

**Abstract**: Retrieval-Augmented Generation (RAG) addresses hallucination and real-time constraints by dynamically retrieving relevant information from a knowledge database to supplement the LLMs' input. When presented with a query, RAG selects the most semantically similar texts from its knowledge bases and uses them as context for the LLMs to generate more accurate responses. RAG also creates a new attack surface, especially since RAG databases are frequently sourced from public domains. While existing studies have predominantly focused on optimizing RAG's performance and efficiency, emerging research has begun addressing the security concerns associated with RAG. However, these works have some limitations, typically focusing on either white-box methodologies or heuristic-based black-box attacks. Furthermore, prior research has mainly targeted simple factoid question answering, which is neither practically challenging nor resistant to correction. In this paper, we unveil a more realistic and threatening scenario: opinion manipulation for controversial topics against RAG. Particularly, we propose a novel RAG black-box attack method, termed FlipedRAG, which is transfer-based. By leveraging instruction engineering, we obtain partial retrieval model outputs from black-box RAG system, facilitating the training of surrogate models to enhance the effectiveness of opinion manipulation attack. Extensive experimental results confirms that our approach significantly enhances the average success rate of opinion manipulation by 16.7%. It achieves an average of a 50% directional change in the opinion polarity of RAG responses across four themes. Additionally, it induces a 20% shift in user cognition. Furthermore, we discuss the efficacy of potential defense mechanisms and conclude that they are insufficient in mitigating this type of attack, highlighting the urgent need to develop novel defensive strategies.

摘要: 检索-增强生成(RAG)通过从知识数据库中动态检索相关信息来补充LLMS的输入，从而解决幻觉和实时约束。当出现查询时，RAG从其知识库中选择语义最相似的文本，并将其用作LLMS的上下文，以生成更准确的响应。RAG还创造了一个新的攻击面，特别是因为RAG数据库经常来自公共域。虽然现有的研究主要集中在优化RAG的性能和效率上，但新兴的研究已经开始解决与RAG相关的安全问题。然而，这些工作有一些局限性，通常集中在白盒方法或基于启发式的黑盒攻击。此外，以前的研究主要针对简单的事实式问题回答，这既不具有实际挑战性，也不抵制纠正。在这篇文章中，我们揭示了一个更现实和更具威胁性的场景：针对RAG的有争议话题的观点操纵。特别地，我们提出了一种新的基于传输的RAG黑盒攻击方法，称为FliedRAG。利用教学工程技术，从黑盒RAG系统中获取部分检索模型输出，便于对代理模型的训练，提高意见操纵攻击的有效性。大量的实验结果表明，该方法显著提高了意见操纵的平均成功率16.7%。它实现了四个主题的RAG回复的意见极性平均50%的方向性变化。此外，它还会导致用户认知发生20%的变化。此外，我们讨论了潜在的防御机制的有效性，并得出结论，它们在缓解这种类型的攻击方面是不够的，突出了开发新的防御策略的迫切需要。



## **43. Dagger Behind Smile: Fool LLMs with a Happy Ending Story**

微笑背后的匕首：傻瓜LLMs，有一个幸福的结局 cs.CL

**SubmitDate**: 2025-01-19    [abs](http://arxiv.org/abs/2501.13115v1) [paper-pdf](http://arxiv.org/pdf/2501.13115v1)

**Authors**: Xurui Song, Zhixin Xie, Shuo Huai, Jiayi Kong, Jun Luo

**Abstract**: The wide adoption of Large Language Models (LLMs) has attracted significant attention from \textit{jailbreak} attacks, where adversarial prompts crafted through optimization or manual design exploit LLMs to generate malicious content. However, optimization-based attacks have limited efficiency and transferability, while manual designs are either easily detectable or demand intricate interactions with LLMs. In this paper, we first point out a novel perspective for jailbreak attacks: LLMs are more responsive to \textit{positive} prompts. Based on this, we deploy Happy Ending Attack (HEA) to wrap up a malicious request in a scenario template involving a positive prompt formed mainly via a \textit{happy ending}, it thus fools LLMs into jailbreaking either immediately or at a follow-up malicious request. This has made HEA both efficient and effective, as it requires only up to two steps to fully jailbreak LLMs. Extensive experiments show that our HEA can successfully jailbreak on state-of-the-art LLMs, including GPT-4o, Llama3-70b, Gemini-pro, and achieves 88.79\% Attack Success Rate on average. We also provide potential quantitative explanations for the success of HEA.

摘要: 大语言模型(LLM)的广泛采用引起了文本{jailBreak}攻击的极大关注，即通过优化或手动设计创建的敌意提示利用LLM生成恶意内容。然而，基于优化的攻击效率和可转移性有限，而手动设计要么很容易被检测到，要么需要与LLM进行复杂的交互。在这篇文章中，我们首先指出了越狱攻击的一个新视角：LLM对文本{积极}提示的响应更快。在此基础上，我们利用快乐结束攻击(HEA)将恶意请求包装在一个场景模板中，该场景模板包含主要通过文本{快乐结束}形成的积极提示，从而欺骗LLMS立即越狱或在后续恶意请求时越狱。这使得HEA既高效又有效，因为它只需要最多两个步骤就可以完全越狱。大量的实验表明，我们的HEA能够成功地在GPT-40、Llama3-70b、Gemini-Pro等最先进的LLMS上越狱，平均攻击成功率达到88.79%。我们还为HEA的成功提供了潜在的定量解释。



## **44. Underestimated Privacy Risks for Minority Populations in Large Language Model Unlearning**

大型语言模型放弃学习中少数族裔的隐私风险被低估 cs.LG

**SubmitDate**: 2025-01-19    [abs](http://arxiv.org/abs/2412.08559v2) [paper-pdf](http://arxiv.org/pdf/2412.08559v2)

**Authors**: Rongzhe Wei, Mufei Li, Mohsen Ghassemi, Eleonora Kreačić, Yifan Li, Xiang Yue, Bo Li, Vamsi K. Potluru, Pan Li, Eli Chien

**Abstract**: Large Language Models are trained on extensive datasets that often contain sensitive, human-generated information, raising significant concerns about privacy breaches. While certified unlearning approaches offer strong privacy guarantees, they rely on restrictive model assumptions that are not applicable to LLMs. As a result, various unlearning heuristics have been proposed, with the associated privacy risks assessed only empirically. The standard evaluation pipelines typically randomly select data for removal from the training set, apply unlearning techniques, and use membership inference attacks to compare the unlearned models against models retrained without the to-be-unlearned data. However, since every data point is subject to the right to be forgotten, unlearning should be considered in the worst-case scenario from the privacy perspective. Prior work shows that data outliers may exhibit higher memorization effects. Intuitively, they are harder to be unlearn and thus the privacy risk of unlearning them is underestimated in the current evaluation. In this paper, we leverage minority data to identify such a critical flaw in previously widely adopted evaluations. We substantiate this claim through carefully designed experiments, including unlearning canaries related to minority groups, inspired by privacy auditing literature. Using personally identifiable information as a representative minority identifier, we demonstrate that minority groups experience at least 20% more privacy leakage in most cases across six unlearning approaches, three MIAs, three benchmark datasets, and two LLMs of different scales. Given that the right to be forgotten should be upheld for every individual, we advocate for a more rigorous evaluation of LLM unlearning methods. Our minority-aware evaluation framework represents an initial step toward ensuring more equitable assessments of LLM unlearning efficacy.

摘要: 大型语言模型在大量数据集上进行训练，这些数据集通常包含敏感的人类生成的信息，这引发了人们对侵犯隐私的严重担忧。虽然经过认证的遗忘方法提供了强大的隐私保证，但它们依赖于不适用于LLM的限制性模型假设。因此，各种遗忘启发式方法被提出，相关的隐私风险仅通过经验进行评估。标准评估流水线通常随机选择要从训练集中移除的数据，应用遗忘技术，并使用成员关系推理攻击来将未学习的模型与没有待学习数据的重新训练的模型进行比较。然而，由于每个数据点都有被遗忘的权利，所以从隐私的角度来看，遗忘应该在最坏的情况下考虑。前人的工作表明，数据离群点可能会表现出更高的记忆效果。直觉上，它们更难被忘记，因此在当前的评估中，忘记它们的隐私风险被低估了。在这篇文章中，我们利用少数群体数据来识别以前被广泛采用的评估中的这样一个关键缺陷。我们通过精心设计的实验来证实这一说法，包括在隐私审计文献的启发下，忘记与少数群体有关的金丝雀。使用个人可识别信息作为代表性的少数群体识别符，我们证明在六种遗忘方法、三个MIA、三个基准数据集和两个不同规模的LLMS中，少数群体在大多数情况下经历的隐私泄露至少多20%。鉴于每个人都应该维护被遗忘的权利，我们主张对LLM遗忘方法进行更严格的评估。我们的少数群体意识评估框架是朝着确保对LLM遗忘效能进行更公平的评估迈出的第一步。



## **45. Latent-space adversarial training with post-aware calibration for defending large language models against jailbreak attacks**

具有事后感知校准的潜在空间对抗训练，用于保护大型语言模型免受越狱攻击 cs.CR

Under Review

**SubmitDate**: 2025-01-18    [abs](http://arxiv.org/abs/2501.10639v1) [paper-pdf](http://arxiv.org/pdf/2501.10639v1)

**Authors**: Xin Yi, Yue Li, Linlin Wang, Xiaoling Wang, Liang He

**Abstract**: Ensuring safety alignment has become a critical requirement for large language models (LLMs), particularly given their widespread deployment in real-world applications. However, LLMs remain susceptible to jailbreak attacks, which exploit system vulnerabilities to bypass safety measures and generate harmful outputs. Although numerous defense mechanisms based on adversarial training have been proposed, a persistent challenge lies in the exacerbation of over-refusal behaviors, which compromise the overall utility of the model. To address these challenges, we propose a Latent-space Adversarial Training with Post-aware Calibration (LATPC) framework. During the adversarial training phase, LATPC compares harmful and harmless instructions in the latent space and extracts safety-critical dimensions to construct refusal features attack, precisely simulating agnostic jailbreak attack types requiring adversarial mitigation. At the inference stage, an embedding-level calibration mechanism is employed to alleviate over-refusal behaviors with minimal computational overhead. Experimental results demonstrate that, compared to various defense methods across five types of jailbreak attacks, LATPC framework achieves a superior balance between safety and utility. Moreover, our analysis underscores the effectiveness of extracting safety-critical dimensions from the latent space for constructing robust refusal feature attacks.

摘要: 确保安全对齐已成为大型语言模型(LLM)的关键要求，特别是考虑到它们在现实世界应用程序中的广泛部署。然而，LLMS仍然容易受到越狱攻击，这些攻击利用系统漏洞绕过安全措施并产生有害输出。尽管已经提出了许多基于对抗性训练的防御机制，但一个持续存在的挑战在于过度拒绝行为的加剧，这损害了该模型的整体实用性。为了应对这些挑战，我们提出了一种基于后感知校准的潜在空间对抗训练(LATPC)框架。在对抗性训练阶段，LATPC在潜在空间中比较有害和无害的指令，并提取安全关键维度来构建拒绝特征攻击，精确模拟需要对抗性缓解的不可知性越狱攻击类型。在推理阶段，采用嵌入级校准机制，以最小的计算开销缓解过度拒绝行为。实验结果表明，与五种类型越狱攻击的各种防御方法相比，LATPC框架在安全性和实用性之间取得了更好的平衡。此外，我们的分析强调了从潜在空间中提取安全关键维度来构建稳健拒绝特征攻击的有效性。



## **46. Jailbreaking as a Reward Misspecification Problem**

越狱是奖励错误指定问题 cs.LG

**SubmitDate**: 2025-01-17    [abs](http://arxiv.org/abs/2406.14393v4) [paper-pdf](http://arxiv.org/pdf/2406.14393v4)

**Authors**: Zhihui Xie, Jiahui Gao, Lei Li, Zhenguo Li, Qi Liu, Lingpeng Kong

**Abstract**: The widespread adoption of large language models (LLMs) has raised concerns about their safety and reliability, particularly regarding their vulnerability to adversarial attacks. In this paper, we propose a novel perspective that attributes this vulnerability to reward misspecification during the alignment process. This misspecification occurs when the reward function fails to accurately capture the intended behavior, leading to misaligned model outputs. We introduce a metric ReGap to quantify the extent of reward misspecification and demonstrate its effectiveness and robustness in detecting harmful backdoor prompts. Building upon these insights, we present ReMiss, a system for automated red teaming that generates adversarial prompts in a reward-misspecified space. ReMiss achieves state-of-the-art attack success rates on the AdvBench benchmark against various target aligned LLMs while preserving the human readability of the generated prompts. Furthermore, these attacks on open-source models demonstrate high transferability to closed-source models like GPT-4o and out-of-distribution tasks from HarmBench. Detailed analysis highlights the unique advantages of the proposed reward misspecification objective compared to previous methods, offering new insights for improving LLM safety and robustness.

摘要: 大型语言模型(LLM)的广泛采用引起了人们对它们的安全性和可靠性的担忧，特别是它们对对手攻击的脆弱性。在本文中，我们提出了一种新的观点，将该漏洞归因于对齐过程中的错误指定。当奖励函数未能准确捕获预期行为时，就会出现这种错误说明，从而导致模型输出不对齐。我们引入了一个度量指标ReGap来量化奖励错误指定的程度，并展示了它在检测有害后门提示方面的有效性和健壮性。在这些见解的基础上，我们提出了REMISTY，这是一个用于自动红色团队的系统，它在错误指定奖励的空间中生成对抗性提示。在保持生成提示的人类可读性的同时，针对各种目标对齐的LLM，在AdvBtch基准上实现了最先进的攻击成功率。此外，这些对开源模型的攻击表明，可以很好地转移到GPT-4o等封闭源代码模型和来自HarmBtch的非分发任务。详细的分析强调了与以前的方法相比，所提出的奖励误指定目标的独特优势，为提高LLM的安全性和稳健性提供了新的见解。



## **47. Generative AI in Cybersecurity: A Comprehensive Review of LLM Applications and Vulnerabilities**

网络安全中的生成人工智能：LLM应用和漏洞的全面审查 cs.CR

52 pages, 8 figures

**SubmitDate**: 2025-01-17    [abs](http://arxiv.org/abs/2405.12750v2) [paper-pdf](http://arxiv.org/pdf/2405.12750v2)

**Authors**: Mohamed Amine Ferrag, Fatima Alwahedi, Ammar Battah, Bilel Cherif, Abdechakour Mechri, Norbert Tihanyi, Tamas Bisztray, Merouane Debbah

**Abstract**: This paper provides a comprehensive review of the future of cybersecurity through Generative AI and Large Language Models (LLMs). We explore LLM applications across various domains, including hardware design security, intrusion detection, software engineering, design verification, cyber threat intelligence, malware detection, and phishing detection. We present an overview of LLM evolution and its current state, focusing on advancements in models such as GPT-4, GPT-3.5, Mixtral-8x7B, BERT, Falcon2, and LLaMA. Our analysis extends to LLM vulnerabilities, such as prompt injection, insecure output handling, data poisoning, DDoS attacks, and adversarial instructions. We delve into mitigation strategies to protect these models, providing a comprehensive look at potential attack scenarios and prevention techniques. Furthermore, we evaluate the performance of 42 LLM models in cybersecurity knowledge and hardware security, highlighting their strengths and weaknesses. We thoroughly evaluate cybersecurity datasets for LLM training and testing, covering the lifecycle from data creation to usage and identifying gaps for future research. In addition, we review new strategies for leveraging LLMs, including techniques like Half-Quadratic Quantization (HQQ), Reinforcement Learning with Human Feedback (RLHF), Direct Preference Optimization (DPO), Quantized Low-Rank Adapters (QLoRA), and Retrieval-Augmented Generation (RAG). These insights aim to enhance real-time cybersecurity defenses and improve the sophistication of LLM applications in threat detection and response. Our paper provides a foundational understanding and strategic direction for integrating LLMs into future cybersecurity frameworks, emphasizing innovation and robust model deployment to safeguard against evolving cyber threats.

摘要: 本文通过生成式人工智能和大型语言模型(LLMS)对网络安全的未来进行了全面的回顾。我们探索了LLM在不同领域的应用，包括硬件设计安全、入侵检测、软件工程、设计验证、网络威胁情报、恶意软件检测和网络钓鱼检测。我们概述了LLM的演化和现状，重点介绍了GPT-4、GPT-3.5、Mixtral-8x7B、BERT、Falcon2和Llama等模型的进展。我们的分析扩展到LLM漏洞，如快速注入、不安全的输出处理、数据中毒、DDoS攻击和敌意指令。我们深入研究缓解策略以保护这些模型，提供对潜在攻击场景和预防技术的全面了解。此外，我们评估了42个LLM模型在网络安全知识和硬件安全方面的性能，突出了它们的优势和劣势。我们为LLM培训和测试彻底评估网络安全数据集，涵盖从数据创建到使用的整个生命周期，并为未来的研究确定差距。此外，我们还回顾了利用LLMS的新策略，包括半二次量化(HQQ)、带人反馈的强化学习(RLHF)、直接偏好优化(DPO)、量化低阶适配器(QLoRA)和检索增强生成(RAG)。这些见解旨在增强实时网络安全防御，并提高LLM应用程序在威胁检测和响应方面的复杂性。我们的论文为将低成本管理系统整合到未来的网络安全框架中提供了一个基础性的理解和战略方向，强调创新和稳健的模型部署，以防范不断演变的网络威胁。



## **48. Can AI-Generated Text be Reliably Detected?**

人工智能生成的文本能否被可靠地检测到？ cs.CL

Published in Transactions on Machine Learning Research (TMLR)

**SubmitDate**: 2025-01-17    [abs](http://arxiv.org/abs/2303.11156v4) [paper-pdf](http://arxiv.org/pdf/2303.11156v4)

**Authors**: Vinu Sankar Sadasivan, Aounon Kumar, Sriram Balasubramanian, Wenxiao Wang, Soheil Feizi

**Abstract**: Large Language Models (LLMs) perform impressively well in various applications. However, the potential for misuse of these models in activities such as plagiarism, generating fake news, and spamming has raised concern about their responsible use. Consequently, the reliable detection of AI-generated text has become a critical area of research. AI text detectors have shown to be effective under their specific settings. In this paper, we stress-test the robustness of these AI text detectors in the presence of an attacker. We introduce recursive paraphrasing attack to stress test a wide range of detection schemes, including the ones using the watermarking as well as neural network-based detectors, zero shot classifiers, and retrieval-based detectors. Our experiments conducted on passages, each approximately 300 tokens long, reveal the varying sensitivities of these detectors to our attacks. Our findings indicate that while our recursive paraphrasing method can significantly reduce detection rates, it only slightly degrades text quality in many cases, highlighting potential vulnerabilities in current detection systems in the presence of an attacker. Additionally, we investigate the susceptibility of watermarked LLMs to spoofing attacks aimed at misclassifying human-written text as AI-generated. We demonstrate that an attacker can infer hidden AI text signatures without white-box access to the detection method, potentially leading to reputational risks for LLM developers. Finally, we provide a theoretical framework connecting the AUROC of the best possible detector to the Total Variation distance between human and AI text distributions. This analysis offers insights into the fundamental challenges of reliable detection as language models continue to advance. Our code is publicly available at https://github.com/vinusankars/Reliability-of-AI-text-detectors.

摘要: 大型语言模型(LLM)在各种应用中表现出色。然而，这些模型可能被滥用于抄袭、生成假新闻和垃圾邮件等活动，这引发了人们对它们负责任使用的担忧。因此，对人工智能生成的文本进行可靠检测已成为一个关键的研究领域。人工智能文本检测器已被证明在其特定设置下是有效的。在本文中，我们重点测试了这些人工智能文本检测器在攻击者在场的情况下的稳健性。我们引入了递归转述攻击来对一系列检测方案进行压力测试，包括使用水印的方案以及基于神经网络的检测器、零镜头分类器和基于检索的检测器。我们在通道上进行的实验，每个通道大约300个令牌长，揭示了这些检测器对我们的攻击的不同敏感度。我们的发现表明，虽然我们的递归转译方法可以显著降低检测率，但在许多情况下它只略微降低了文本质量，突出了当前检测系统在攻击者在场的情况下的潜在漏洞。此外，我们还研究了带水印的LLMS对欺骗攻击的敏感性，该攻击旨在将人类书写的文本错误分类为人工智能生成的文本。我们证明，攻击者可以在没有白盒访问检测方法的情况下推断隐藏的AI文本签名，这可能会导致LLM开发人员的声誉风险。最后，我们提供了一个理论框架，将最佳可能检测器的AUROC与人类和AI文本分布之间的总变异距离联系起来。随着语言模型的不断发展，这一分析提供了对可靠检测的根本挑战的见解。我们的代码在https://github.com/vinusankars/Reliability-of-AI-text-detectors.上公开提供



## **49. Computing Optimization-Based Prompt Injections Against Closed-Weights Models By Misusing a Fine-Tuning API**

通过滥用微调API针对闭权模型计算基于优化的提示注射 cs.CR

**SubmitDate**: 2025-01-16    [abs](http://arxiv.org/abs/2501.09798v1) [paper-pdf](http://arxiv.org/pdf/2501.09798v1)

**Authors**: Andrey Labunets, Nishit V. Pandya, Ashish Hooda, Xiaohan Fu, Earlence Fernandes

**Abstract**: We surface a new threat to closed-weight Large Language Models (LLMs) that enables an attacker to compute optimization-based prompt injections. Specifically, we characterize how an attacker can leverage the loss-like information returned from the remote fine-tuning interface to guide the search for adversarial prompts. The fine-tuning interface is hosted by an LLM vendor and allows developers to fine-tune LLMs for their tasks, thus providing utility, but also exposes enough information for an attacker to compute adversarial prompts. Through an experimental analysis, we characterize the loss-like values returned by the Gemini fine-tuning API and demonstrate that they provide a useful signal for discrete optimization of adversarial prompts using a greedy search algorithm. Using the PurpleLlama prompt injection benchmark, we demonstrate attack success rates between 65% and 82% on Google's Gemini family of LLMs. These attacks exploit the classic utility-security tradeoff - the fine-tuning interface provides a useful feature for developers but also exposes the LLMs to powerful attacks.

摘要: 我们对封闭式大型语言模型(LLM)提出了新的威胁，使攻击者能够计算基于优化的提示注入。具体地说，我们描述了攻击者如何利用从远程微调界面返回的类似丢失的信息来指导对敌意提示的搜索。微调界面由LLM供应商托管，允许开发人员针对他们的任务微调LLM，从而提供实用程序，但也暴露了足够的信息，供攻击者计算敌意提示。通过实验分析，我们表征了Gemini微调API返回的类似损失的值，并证明它们为使用贪婪搜索算法对敌意提示进行离散优化提供了有用的信号。使用PurpleLlama快速注入基准，我们展示了对Google的Gemini系列LLM的攻击成功率在65%到82%之间。这些攻击利用了经典的实用程序-安全权衡-微调界面为开发人员提供了有用的功能，但也使LLM面临强大的攻击。



## **50. TPIA: Towards Target-specific Prompt Injection Attack against Code-oriented Large Language Models**

TPIA：针对面向代码的大型语言模型的特定目标提示注入攻击 cs.CR

**SubmitDate**: 2025-01-16    [abs](http://arxiv.org/abs/2407.09164v4) [paper-pdf](http://arxiv.org/pdf/2407.09164v4)

**Authors**: Yuchen Yang, Hongwei Yao, Bingrun Yang, Yiling He, Yiming Li, Tianwei Zhang, Zhan Qin, Kui Ren, Chun Chen

**Abstract**: Recently, code-oriented large language models (Code LLMs) have been widely exploited to simplify and facilitate programming. With these tools, developers can easily generate the desired complete functional code based on incomplete code snippets and natural language prompts. Unfortunately, a few pioneering works revealed that these Code LLMs are vulnerable to backdoor and adversarial attacks. The former poisons the training data or model parameters, hijacking the LLMs to generate malicious code snippets when encountering the trigger. The latter crafts malicious adversarial input codes to reduce the quality of the generated codes. However, both attacks have some inherent limitations: backdoor attacks rely on the adversary's capability of controlling the model training process; adversarial attacks struggle with fulfilling specific malicious purposes. This paper presents a novel attack paradigm against Code LLMs, namely target-specific prompt injection attack (TPIA). TPIA generates non-functional perturbations containing the information of malicious instructions and inserts them into the victim's code context by spreading them into potentially used dependencies (e.g., packages or RAG's knowledge base). It induces the Code LLMs to generate attacker-specified malicious code snippets at the target location. In general, we compress the attacker-specified malicious objective into the perturbation by adversarial optimization based on greedy token search. We collect 13 representative malicious objectives to design 31 threat cases for three popular programming languages. We show that our TPIA can successfully attack three representative open-source Code LLMs (with an ASR of up to 97.9%) and two mainstream commercial Code LLM-integrated applications (with an ASR of over 90%) in all threat cases, using only a 12-token perturbation. Our work alerts a new practical threat of using Code LLMs.

摘要: 最近，面向代码的大型语言模型(Code LLM)已被广泛利用来简化和促进编程。使用这些工具，开发人员可以根据不完整的代码片段和自然语言提示轻松生成所需的完整功能代码。不幸的是，一些开创性的工作表明，这些代码LLM容易受到后门和对手的攻击。前者毒化训练数据或模型参数，在遇到触发器时劫持LLMS生成恶意代码片段。后者制作恶意敌意输入代码以降低生成代码的质量。然而，这两种攻击都有一些固有的局限性：后门攻击依赖于对手控制模型训练过程的能力；对抗性攻击难以实现特定的恶意目的。提出了一种新的针对Code LLMS的攻击范式，即目标特定的即时注入攻击(TPIA)。TPIA生成包含恶意指令信息的非功能性扰动，并通过将它们传播到可能使用的依赖项(例如，包或RAG的知识库)，将它们插入到受害者的代码上下文中。它诱导代码LLM在目标位置生成攻击者指定的恶意代码片段。一般而言，我们通过基于贪婪令牌搜索的对抗性优化将攻击者指定的恶意目标压缩为扰动。我们收集了13个具有代表性的恶意目标，为三种流行的编程语言设计了31个威胁案例。实验表明，在所有威胁情况下，仅需12个令牌的扰动，我们的TPIA就能成功攻击三个典型的开源代码LLMS(ASR高达97.9%)和两个主流商业代码LLM集成应用(ASR超过90%)。我们的工作警示了使用Code LLMS的新的实际威胁。



