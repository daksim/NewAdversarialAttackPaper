# Latest Large Language Model Attack Papers
**update at 2024-11-18 09:38:00**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. AutoDefense: Multi-Agent LLM Defense against Jailbreak Attacks**

AutoDefense：针对越狱攻击的多代理LLM防御 cs.LG

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2403.04783v2) [paper-pdf](http://arxiv.org/pdf/2403.04783v2)

**Authors**: Yifan Zeng, Yiran Wu, Xiao Zhang, Huazheng Wang, Qingyun Wu

**Abstract**: Despite extensive pre-training in moral alignment to prevent generating harmful information, large language models (LLMs) remain vulnerable to jailbreak attacks. In this paper, we propose AutoDefense, a multi-agent defense framework that filters harmful responses from LLMs. With the response-filtering mechanism, our framework is robust against different jailbreak attack prompts, and can be used to defend different victim models. AutoDefense assigns different roles to LLM agents and employs them to complete the defense task collaboratively. The division in tasks enhances the overall instruction-following of LLMs and enables the integration of other defense components as tools. With AutoDefense, small open-source LMs can serve as agents and defend larger models against jailbreak attacks. Our experiments show that AutoDefense can effectively defense against different jailbreak attacks, while maintaining the performance at normal user request. For example, we reduce the attack success rate on GPT-3.5 from 55.74% to 7.95% using LLaMA-2-13b with a 3-agent system. Our code and data are publicly available at https://github.com/XHMY/AutoDefense.

摘要: 尽管在道德一致性方面进行了广泛的预先培训，以防止产生有害信息，但大型语言模型(LLM)仍然容易受到越狱攻击。在本文中，我们提出了一种过滤来自LLMS的有害响应的多代理防御框架--AutoDefense。通过响应过滤机制，我们的框架对不同的越狱攻击提示具有健壮性，并且可以用于防御不同的受害者模型。AutoDefense为LLM特工分配不同的角色，并雇用他们协作完成防御任务。任务分工加强了LLMS的整体指令遵循，并使其他防御组件能够作为工具进行集成。有了AutoDefense，小型开源LMS可以作为代理，保护较大的模型免受越狱攻击。我们的实验表明，AutoDefense能够有效地防御不同的越狱攻击，同时保持正常用户请求的性能。例如，我们使用带有3代理系统的Llama-2-13b将对GPT-3.5的攻击成功率从55.74%降低到7.95%。我们的代码和数据在https://github.com/XHMY/AutoDefense.上公开提供



## **2. DROJ: A Prompt-Driven Attack against Large Language Models**

DROJ：针对大型语言模型的预算驱动攻击 cs.CL

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2411.09125v1) [paper-pdf](http://arxiv.org/pdf/2411.09125v1)

**Authors**: Leyang Hu, Boran Wang

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional capabilities across various natural language processing tasks. Due to their training on internet-sourced datasets, LLMs can sometimes generate objectionable content, necessitating extensive alignment with human feedback to avoid such outputs. Despite massive alignment efforts, LLMs remain susceptible to adversarial jailbreak attacks, which usually are manipulated prompts designed to circumvent safety mechanisms and elicit harmful responses. Here, we introduce a novel approach, Directed Rrepresentation Optimization Jailbreak (DROJ), which optimizes jailbreak prompts at the embedding level to shift the hidden representations of harmful queries towards directions that are more likely to elicit affirmative responses from the model. Our evaluations on LLaMA-2-7b-chat model show that DROJ achieves a 100\% keyword-based Attack Success Rate (ASR), effectively preventing direct refusals. However, the model occasionally produces repetitive and non-informative responses. To mitigate this, we introduce a helpfulness system prompt that enhances the utility of the model's responses. Our code is available at https://github.com/Leon-Leyang/LLM-Safeguard.

摘要: 大型语言模型(LLM)在各种自然语言处理任务中表现出了非凡的能力。由于对来自互联网的数据集进行了培训，LLM有时会产生令人反感的内容，需要与人类反馈广泛协调，以避免此类输出。尽管做出了巨大的调整努力，但LLM仍然容易受到对抗性越狱攻击，这些攻击通常是被操纵的提示，旨在绕过安全机制并引发有害反应。在这里，我们介绍了一种新的方法，定向R表示优化越狱(DROJ)，它在嵌入级别优化越狱提示，将有害查询的隐藏表示向更有可能引起模型肯定响应的方向移动。对Llama-2-7b-Chat模型的评估表明，DROJ达到了100%的基于关键字的攻击成功率，有效地防止了直接拒绝。然而，该模型偶尔会产生重复的、非信息性的回答。为了缓解这一问题，我们引入了一个帮助系统提示，以增强模型响应的实用性。我们的代码可以在https://github.com/Leon-Leyang/LLM-Safeguard.上找到



## **3. Trustful LLMs: Customizing and Grounding Text Generation with Knowledge Bases and Dual Decoders**

值得信赖的LLM：使用知识库和双解码器定制和基础文本生成 cs.CL

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2411.07870v2) [paper-pdf](http://arxiv.org/pdf/2411.07870v2)

**Authors**: Xiaofeng Zhu, Jaya Krishna Mandivarapu

**Abstract**: Although people are impressed by the content generation skills of large language models, the use of LLMs, such as ChatGPT, is limited by the domain grounding of the content. The correctness and groundedness of the generated content need to be based on a verified context, such as results from Retrieval-Augmented Generation (RAG). One important issue when adapting LLMs to a customized domain is that the generated responses are often incomplete, or the additions are not verified and may even be hallucinated. Prior studies on hallucination detection have focused on evaluation metrics, which are not easily adaptable to dynamic domains and can be vulnerable to attacks like jail-breaking. In this work, we propose 1) a post-processing algorithm that leverages knowledge triplets in RAG context to correct hallucinations and 2) a dual-decoder model that fuses RAG context to guide the generation process.

摘要: 尽管人们对大型语言模型的内容生成技能印象深刻，但ChatGPT等LLM的使用受到内容领域基础的限制。生成内容的正确性和可信度需要基于经过验证的上下文，例如检索增强生成（RAG）的结果。将LLM调整到定制域时的一个重要问题是，生成的响应通常不完整，或者添加未经验证，甚至可能出现幻觉。之前关于幻觉检测的研究集中在评估指标上，这些指标不容易适应动态领域，并且容易受到越狱等攻击。在这项工作中，我们提出了1）一种利用RAG上下文中的知识三重组来纠正幻觉的后处理算法，以及2）一种融合RAG上下文来指导生成过程的双解码器模型。



## **4. Defending Large Language Models Against Attacks With Residual Stream Activation Analysis**

利用剩余流激活分析防御大型语言模型免受攻击 cs.CR

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2406.03230v4) [paper-pdf](http://arxiv.org/pdf/2406.03230v4)

**Authors**: Amelia Kawasaki, Andrew Davis, Houssam Abbas

**Abstract**: The widespread adoption of Large Language Models (LLMs), exemplified by OpenAI's ChatGPT, brings to the forefront the imperative to defend against adversarial threats on these models. These attacks, which manipulate an LLM's output by introducing malicious inputs, undermine the model's integrity and the trust users place in its outputs. In response to this challenge, our paper presents an innovative defensive strategy, given white box access to an LLM, that harnesses residual activation analysis between transformer layers of the LLM. We apply a novel methodology for analyzing distinctive activation patterns in the residual streams for attack prompt classification. We curate multiple datasets to demonstrate how this method of classification has high accuracy across multiple types of attack scenarios, including our newly-created attack dataset. Furthermore, we enhance the model's resilience by integrating safety fine-tuning techniques for LLMs in order to measure its effect on our capability to detect attacks. The results underscore the effectiveness of our approach in enhancing the detection and mitigation of adversarial inputs, advancing the security framework within which LLMs operate.

摘要: 大型语言模型(LLM)的广泛采用，如OpenAI的ChatGPT，使防御这些模型上的对手威胁成为当务之急。这些攻击通过引入恶意输入来操纵LLM的输出，破坏了模型的完整性和用户对其输出的信任。为了应对这一挑战，我们的论文提出了一种创新的防御策略，在白盒访问LLM的情况下，该策略利用LLM变压器层之间的剩余激活分析。我们应用了一种新的方法来分析残留流中独特的激活模式，以进行攻击提示分类。我们精选了多个数据集，以演示此分类方法如何在多种类型的攻击场景中具有高精度，包括我们新创建的攻击数据集。此外，我们通过集成LLMS的安全微调技术来增强模型的弹性，以衡量其对我们检测攻击的能力的影响。这些结果强调了我们的方法在加强对敌对输入的检测和缓解、推进LLMS运作的安全框架方面的有效性。



## **5. LLMStinger: Jailbreaking LLMs using RL fine-tuned LLMs**

LLMStinger：使用RL微调的LLM越狱LLM cs.LG

Accepted at AAAI 2025

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2411.08862v1) [paper-pdf](http://arxiv.org/pdf/2411.08862v1)

**Authors**: Piyush Jha, Arnav Arora, Vijay Ganesh

**Abstract**: We introduce LLMStinger, a novel approach that leverages Large Language Models (LLMs) to automatically generate adversarial suffixes for jailbreak attacks. Unlike traditional methods, which require complex prompt engineering or white-box access, LLMStinger uses a reinforcement learning (RL) loop to fine-tune an attacker LLM, generating new suffixes based on existing attacks for harmful questions from the HarmBench benchmark. Our method significantly outperforms existing red-teaming approaches (we compared against 15 of the latest methods), achieving a +57.2% improvement in Attack Success Rate (ASR) on LLaMA2-7B-chat and a +50.3% ASR increase on Claude 2, both models known for their extensive safety measures. Additionally, we achieved a 94.97% ASR on GPT-3.5 and 99.4% on Gemma-2B-it, demonstrating the robustness and adaptability of LLMStinger across open and closed-source models.

摘要: 我们引入了LLMStinger，这是一种利用大型语言模型（LLM）自动生成越狱攻击的对抗性后缀的新颖方法。与需要复杂的即时工程或白盒访问的传统方法不同，LLMStinger使用强化学习（RL）循环来微调攻击者LLM，根据HarmBench基准中针对有害问题的现有攻击生成新的后缀。我们的方法显着优于现有的红色团队方法（我们与15种最新方法进行了比较），在LLaMA 2 - 7 B-chat上实现了攻击成功率（ASB）+57.2%的提高，在Claude 2上实现了攻击成功率（ASB）+50.3%的提高，这两种型号都以其广泛的安全措施而闻名。此外，我们在GPT-3.5上实现了94.97%的ASB，在Gemma-2B-it上实现了99.4%的ASB，证明了LLMStinger在开放和封闭源模型中的稳健性和适应性。



## **6. Insights and Current Gaps in Open-Source LLM Vulnerability Scanners: A Comparative Analysis**

开源LLM漏洞扫描仪的见解和当前差距：比较分析 cs.CR

15 pages, 11 figures

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2410.16527v2) [paper-pdf](http://arxiv.org/pdf/2410.16527v2)

**Authors**: Jonathan Brokman, Omer Hofman, Oren Rachmil, Inderjeet Singh, Rathina Sabapathy Aishvariya Priya, Vikas Pahuja, Amit Giloni, Roman Vainshtein, Hisashi Kojima

**Abstract**: This report presents a comparative analysis of open-source vulnerability scanners for conversational large language models (LLMs). As LLMs become integral to various applications, they also present potential attack surfaces, exposed to security risks such as information leakage and jailbreak attacks. Our study evaluates prominent scanners - Garak, Giskard, PyRIT, and CyberSecEval - that adapt red-teaming practices to expose these vulnerabilities. We detail the distinctive features and practical use of these scanners, outline unifying principles of their design and perform quantitative evaluations to compare them. These evaluations uncover significant reliability issues in detecting successful attacks, highlighting a fundamental gap for future development. Additionally, we contribute a preliminary labelled dataset, which serves as an initial step to bridge this gap. Based on the above, we provide strategic recommendations to assist organizations choose the most suitable scanner for their red-teaming needs, accounting for customizability, test suite comprehensiveness, and industry-specific use cases.

摘要: 本报告对用于会话大型语言模型(LLM)的开源漏洞扫描器进行了比较分析。随着LLM成为各种应用的组成部分，它们也出现了潜在的攻击面，暴露在信息泄露和越狱攻击等安全风险中。我们的研究评估了采用红色团队实践来暴露这些漏洞的著名扫描仪-Garak、Giskard、PyRIT和CyberSecEval。我们详细介绍了这些扫描仪的特点和实际应用，概述了它们设计的统一原则，并进行了定量评估以进行比较。这些评估揭示了检测成功攻击的重大可靠性问题，突显了未来发展的根本差距。此外，我们提供了一个初步的标记数据集，这是弥合这一差距的第一步。在此基础上，我们提供了战略性建议，以帮助组织选择最适合其红团队需求的扫描仪，考虑到可定制性、测试套件的全面性和行业特定的用例。



## **7. Target-driven Attack for Large Language Models**

针对大型语言模型的目标驱动攻击 cs.CL

12 pages, 7 figures. This work is an extension of the  arXiv:2404.07234 work. We propose new methods. 27th European Conference on  Artificial Intelligence 2024

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2411.07268v2) [paper-pdf](http://arxiv.org/pdf/2411.07268v2)

**Authors**: Chong Zhang, Mingyu Jin, Dong Shu, Taowen Wang, Dongfang Liu, Xiaobo Jin

**Abstract**: Current large language models (LLM) provide a strong foundation for large-scale user-oriented natural language tasks. Many users can easily inject adversarial text or instructions through the user interface, thus causing LLM model security challenges like the language model not giving the correct answer. Although there is currently a large amount of research on black-box attacks, most of these black-box attacks use random and heuristic strategies. It is unclear how these strategies relate to the success rate of attacks and thus effectively improve model robustness. To solve this problem, we propose our target-driven black-box attack method to maximize the KL divergence between the conditional probabilities of the clean text and the attack text to redefine the attack's goal. We transform the distance maximization problem into two convex optimization problems based on the attack goal to solve the attack text and estimate the covariance. Furthermore, the projected gradient descent algorithm solves the vector corresponding to the attack text. Our target-driven black-box attack approach includes two attack strategies: token manipulation and misinformation attack. Experimental results on multiple Large Language Models and datasets demonstrate the effectiveness of our attack method.

摘要: 现有的大型语言模型(LLM)为大规模面向用户的自然语言任务提供了坚实的基础。许多用户可以很容易地通过用户界面注入敌意文本或指令，从而导致LLM模型的安全挑战，如语言模型无法给出正确的答案。虽然目前有大量关于黑盒攻击的研究，但这些黑盒攻击大多采用随机和启发式策略。目前尚不清楚这些策略如何与攻击成功率相关，从而有效地提高模型的健壮性。为了解决这一问题，我们提出了目标驱动的黑盒攻击方法，以最大化明文和攻击文本的条件概率之间的KL偏差，从而重新定义攻击的目标。将距离最大化问题转化为基于攻击目标的两个凸优化问题来求解攻击文本并估计协方差。此外，投影梯度下降算法求解与攻击文本对应的向量。我们的目标驱动的黑盒攻击方法包括两种攻击策略：令牌操纵和错误信息攻击。在多个大型语言模型和数据集上的实验结果证明了该攻击方法的有效性。



## **8. DAGER: Exact Gradient Inversion for Large Language Models**

DAGER：大型语言模型的精确梯度倒置 cs.LG

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2405.15586v2) [paper-pdf](http://arxiv.org/pdf/2405.15586v2)

**Authors**: Ivo Petrov, Dimitar I. Dimitrov, Maximilian Baader, Mark Niklas Müller, Martin Vechev

**Abstract**: Federated learning works by aggregating locally computed gradients from multiple clients, thus enabling collaborative training without sharing private client data. However, prior work has shown that the data can actually be recovered by the server using so-called gradient inversion attacks. While these attacks perform well when applied on images, they are limited in the text domain and only permit approximate reconstruction of small batches and short input sequences. In this work, we propose DAGER, the first algorithm to recover whole batches of input text exactly. DAGER leverages the low-rank structure of self-attention layer gradients and the discrete nature of token embeddings to efficiently check if a given token sequence is part of the client data. We use this check to exactly recover full batches in the honest-but-curious setting without any prior on the data for both encoder- and decoder-based architectures using exhaustive heuristic search and a greedy approach, respectively. We provide an efficient GPU implementation of DAGER and show experimentally that it recovers full batches of size up to 128 on large language models (LLMs), beating prior attacks in speed (20x at same batch size), scalability (10x larger batches), and reconstruction quality (ROUGE-1/2 > 0.99).

摘要: 联合学习的工作方式是聚合来自多个客户端的本地计算的梯度，从而在不共享私人客户端数据的情况下实现协作培训。然而，先前的工作表明，服务器实际上可以使用所谓的梯度反转攻击来恢复数据。虽然这些攻击在图像上应用时表现良好，但它们仅限于文本域，仅允许对小批次和短输入序列进行近似重建。在这项工作中，我们提出了第一个准确恢复整批输入文本的算法Dager。Dager利用自我关注层梯度的低等级结构和令牌嵌入的离散性质来有效地检查给定的令牌序列是否是客户端数据的一部分。我们使用这种检查，分别使用穷举启发式搜索和贪婪方法，在诚实但奇怪的设置中准确地恢复完整批次，而不需要对基于编码器和解码器的架构的数据进行任何先验。我们提供了一种高效的Dager的GPU实现，实验表明，它可以在大型语言模型(LLM)上恢复大小高达128的全批处理，在速度(相同批处理大小的20倍)、可伸缩性(大批处理10倍)和重建质量(Rouge-1/2>0.99)方面优于先前的攻击。



## **9. The VLLM Safety Paradox: Dual Ease in Jailbreak Attack and Defense**

VLLM安全悖论：越狱攻击和防御的双重轻松 cs.CR

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2411.08410v1) [paper-pdf](http://arxiv.org/pdf/2411.08410v1)

**Authors**: Yangyang Guo, Fangkai Jiao, Liqiang Nie, Mohan Kankanhalli

**Abstract**: The vulnerability of Vision Large Language Models (VLLMs) to jailbreak attacks appears as no surprise. However, recent defense mechanisms against these attacks have reached near-saturation performance on benchmarks, often with minimal effort. This simultaneous high performance in both attack and defense presents a perplexing paradox. Resolving it is critical for advancing the development of trustworthy models. To address this research gap, we first investigate why VLLMs are prone to these attacks. We then make a key observation: existing defense mechanisms suffer from an \textbf{over-prudence} problem, resulting in unexpected abstention even in the presence of benign inputs. Additionally, we find that the two representative evaluation methods for jailbreak often exhibit chance agreement. This limitation makes it potentially misleading when evaluating attack strategies or defense mechanisms. Beyond these empirical observations, our another contribution in this work is to repurpose the guardrails of LLMs on the shelf, as an effective alternative detector prior to VLLM response. We believe these findings offer useful insights to rethink the foundational development of VLLM safety with respect to benchmark datasets, evaluation methods, and defense strategies.

摘要: Vision Large Language Models(VLLM)在越狱攻击中的脆弱性似乎并不令人意外。然而，最近针对这些攻击的防御机制在基准测试中的性能已经接近饱和，通常只需很少的努力。这种同时在进攻和防守上的高表现提出了一个令人困惑的悖论。解决这一问题对于推动可信模型的发展至关重要。为了解决这一研究差距，我们首先调查了为什么VLLM容易受到这些攻击。然后，我们做了一个关键的观察：现有的防御机制存在过度谨慎的问题，导致即使存在良性投入，也会意外弃权。此外，我们发现，两种具有代表性的越狱评估方法往往表现出偶然性的一致性。这一限制使其在评估攻击策略或防御机制时具有潜在误导性。除了这些经验观察之外，我们在这项工作中的另一个贡献是重新利用架子上的LLM护栏，作为VLLM响应之前的有效替代探测器。我们相信，这些发现为重新思考VLLM安全性在基准数据集、评估方法和防御策略方面的基础性发展提供了有用的见解。



## **10. MultiKG: Multi-Source Threat Intelligence Aggregation for High-Quality Knowledge Graph Representation of Attack Techniques**

MultiKG：用于攻击技术的高质量知识图表示的多源威胁情报聚合 cs.CR

21 pages, 15 figures, 8 tables

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2411.08359v1) [paper-pdf](http://arxiv.org/pdf/2411.08359v1)

**Authors**: Jian Wang, Tiantian Zhu, Chunlin Xiong, Yan Chen

**Abstract**: The construction of attack technique knowledge graphs aims to transform various types of attack knowledge into structured representations for more effective attack procedure modeling. Existing methods typically rely on textual data, such as Cyber Threat Intelligence (CTI) reports, which are often coarse-grained and unstructured, resulting in incomplete and inaccurate knowledge graphs. To address these issues, we expand attack knowledge sources by incorporating audit logs and static code analysis alongside CTI reports, providing finer-grained data for constructing attack technique knowledge graphs.   We propose MultiKG, a fully automated framework that integrates multiple threat knowledge sources. MultiKG processes data from CTI reports, dynamic logs, and static code separately, then merges them into a unified attack knowledge graph. Through system design and the utilization of the Large Language Model (LLM), MultiKG automates the analysis, construction, and merging of attack graphs across these sources, producing a fine-grained, multi-source attack knowledge graph.   We implemented MultiKG and evaluated it using 1,015 real attack techniques and 9,006 attack intelligence entries from CTI reports. Results show that MultiKG effectively extracts attack knowledge graphs from diverse sources and aggregates them into accurate, comprehensive representations. Through case studies, we demonstrate that our approach directly benefits security tasks such as attack reconstruction and detection.

摘要: 攻击技术知识图的构建旨在将各种类型的攻击知识转化为结构化的表示形式，以便更有效地对攻击过程进行建模。现有方法通常依赖文本数据，如网络威胁情报(CTI)报告，这些数据通常是粗粒度和非结构化的，导致知识图谱不完整和不准确。为了解决这些问题，我们通过将审计日志和静态代码分析与CTI报告结合在一起来扩展攻击知识源，为构建攻击技术知识图提供更细粒度的数据。我们提出了一种集成多个威胁知识源的全自动化框架--MultiKG。MultiKG分别处理CTI报告、动态日志和静态代码中的数据，然后将它们合并到统一的攻击知识图中。通过系统设计和大型语言模型(LLM)的利用，MultiKG自动分析、构建和合并这些来源的攻击图，生成细粒度的多源攻击知识图。我们实施了MultiKG，并使用1,015项真实攻击技术和9,006个CTI报告中的攻击情报条目对其进行了评估。结果表明，MultiKG能有效地从不同来源提取攻击知识图，并将其聚合成准确、全面的表示。通过案例研究，我们证明了我们的方法直接有利于攻击重建和检测等安全任务。



## **11. Can adversarial attacks by large language models be attributed?**

大型语言模型的对抗攻击可以归因吗？ cs.AI

7 pages, 1 figure

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.08003v1) [paper-pdf](http://arxiv.org/pdf/2411.08003v1)

**Authors**: Manuel Cebrian, Jan Arne Telle

**Abstract**: Attributing outputs from Large Language Models (LLMs) in adversarial settings-such as cyberattacks and disinformation-presents significant challenges that are likely to grow in importance. We investigate this attribution problem using formal language theory, specifically language identification in the limit as introduced by Gold and extended by Angluin. By modeling LLM outputs as formal languages, we analyze whether finite text samples can uniquely pinpoint the originating model. Our results show that due to the non-identifiability of certain language classes, under some mild assumptions about overlapping outputs from fine-tuned models it is theoretically impossible to attribute outputs to specific LLMs with certainty. This holds also when accounting for expressivity limitations of Transformer architectures. Even with direct model access or comprehensive monitoring, significant computational hurdles impede attribution efforts. These findings highlight an urgent need for proactive measures to mitigate risks posed by adversarial LLM use as their influence continues to expand.

摘要: 将大型语言模型(LLM)的输出归因于敌对环境--如网络攻击和虚假信息--带来了重大挑战，而这些挑战的重要性可能会越来越大。我们使用形式化语言理论来研究这一归因问题，特别是Gold提出并由Anluin推广的极限语言识别问题。通过将LLM输出建模为形式语言，我们分析了有限文本样本是否能够唯一地定位原始模型。我们的结果表明，由于某些语言类别的不可识别性，在微调模型的输出重叠的一些温和假设下，理论上不可能确定地将输出归因于特定的LLM。当考虑到Transformer架构的表现力限制时，这也是成立的。即使有了直接的模型访问或全面的监测，重大的计算障碍也阻碍了归因努力。这些调查结果突出表明，迫切需要采取积极主动的措施，以减轻敌对使用LLM所带来的风险，因为它们的影响继续扩大。



## **12. Chain Association-based Attacking and Shielding Natural Language Processing Systems**

基于链关联的攻击和屏蔽自然语言处理系统 cs.CL

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07843v1) [paper-pdf](http://arxiv.org/pdf/2411.07843v1)

**Authors**: Jiacheng Huang, Long Chen

**Abstract**: Association as a gift enables people do not have to mention something in completely straightforward words and allows others to understand what they intend to refer to. In this paper, we propose a chain association-based adversarial attack against natural language processing systems, utilizing the comprehension gap between humans and machines. We first generate a chain association graph for Chinese characters based on the association paradigm for building search space of potential adversarial examples. Then, we introduce an discrete particle swarm optimization algorithm to search for the optimal adversarial examples. We conduct comprehensive experiments and show that advanced natural language processing models and applications, including large language models, are vulnerable to our attack, while humans appear good at understanding the perturbed text. We also explore two methods, including adversarial training and associative graph-based recovery, to shield systems from chain association-based attack. Since a few examples that use some derogatory terms, this paper contains materials that may be offensive or upsetting to some people.

摘要: 联想作为一种礼物，使人们不必用完全直截了当的语言来提及某事，并让其他人理解他们想指的是什么。本文利用人与机器之间的理解鸿沟，提出了一种基于链式联想的对抗性自然语言处理系统攻击方法。首先在联想范式的基础上生成汉字的链式联想图，构建潜在对抗性实例的搜索空间。然后，我们引入了离散粒子群优化算法来搜索最优的对抗性实例。我们进行了全面的实验，并表明高级自然语言处理模型和应用程序，包括大型语言模型，容易受到我们的攻击，而人类似乎很擅长理解受干扰的文本。我们还探索了两种方法，包括对抗性训练和基于联想图的恢复，以保护系统免受基于链关联的攻击。由于有几个例子使用了一些贬义性的术语，因此本文包含的材料可能会冒犯某些人或使某些人不安。



## **13. Revisiting the Adversarial Robustness of Vision Language Models: a Multimodal Perspective**

重新审视视觉语言模型的对抗鲁棒性：多模式视角 cs.CV

17 pages, 13 figures

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2404.19287v3) [paper-pdf](http://arxiv.org/pdf/2404.19287v3)

**Authors**: Wanqi Zhou, Shuanghao Bai, Danilo P. Mandic, Qibin Zhao, Badong Chen

**Abstract**: Pretrained vision-language models (VLMs) like CLIP exhibit exceptional generalization across diverse downstream tasks. While recent studies reveal their vulnerability to adversarial attacks, research to date has primarily focused on enhancing the robustness of image encoders against image-based attacks, with defenses against text-based and multimodal attacks remaining largely unexplored. To this end, this work presents the first comprehensive study on improving the adversarial robustness of VLMs against attacks targeting image, text, and multimodal inputs. This is achieved by proposing multimodal contrastive adversarial training (MMCoA). Such an approach strengthens the robustness of both image and text encoders by aligning the clean text embeddings with adversarial image embeddings, and adversarial text embeddings with clean image embeddings. The robustness of the proposed MMCoA is examined against existing defense methods over image, text, and multimodal attacks on the CLIP model. Extensive experiments on 15 datasets across two tasks reveal the characteristics of different adversarial defense methods under distinct distribution shifts and dataset complexities across the three attack types. This paves the way for a unified framework of adversarial robustness against different modality attacks, opening up new possibilities for securing VLMs against multimodal attacks. The code is available at https://github.com/ElleZWQ/MMCoA.git.

摘要: 像CLIP这样的预先训练的视觉语言模型(VLM)在不同的下游任务中表现出非凡的通用性。虽然最近的研究揭示了它们对对手攻击的脆弱性，但到目前为止的研究主要集中在增强图像编码器对基于图像的攻击的稳健性上，对基于文本的攻击和多模式攻击的防御在很大程度上仍未被探索。为此，本文首次全面研究了如何提高VLMS对图像、文本和多模式输入的攻击健壮性。这是通过提出多模式对比对抗训练(MMCoA)来实现的。这种方法通过将干净的文本嵌入与对抗性的图像嵌入以及对抗性的文本嵌入与干净的图像嵌入对齐来增强图像和文本编码器的稳健性。针对已有的针对图像、文本和多模式攻击的防御方法，对提出的MMCoA算法的鲁棒性进行了测试。在两个任务的15个数据集上进行了大量的实验，揭示了三种攻击类型在不同的分布变化和数据集复杂性下不同的对抗防御方法的特点。这为对抗不同模式攻击的对抗健壮性的统一框架铺平了道路，为保护VLM免受多模式攻击开辟了新的可能性。代码可在https://github.com/ElleZWQ/MMCoA.git.上获得



## **14. Zer0-Jack: A Memory-efficient Gradient-based Jailbreaking Method for Black-box Multi-modal Large Language Models**

Zer 0-Jack：一种用于黑匣子多模式大型语言模型的内存高效的基于对象的越狱方法 cs.LG

Accepted to Neurips SafeGenAi Workshop 2024

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07559v1) [paper-pdf](http://arxiv.org/pdf/2411.07559v1)

**Authors**: Tiejin Chen, Kaishen Wang, Hua Wei

**Abstract**: Jailbreaking methods, which induce Multi-modal Large Language Models (MLLMs) to output harmful responses, raise significant safety concerns. Among these methods, gradient-based approaches, which use gradients to generate malicious prompts, have been widely studied due to their high success rates in white-box settings, where full access to the model is available. However, these methods have notable limitations: they require white-box access, which is not always feasible, and involve high memory usage. To address scenarios where white-box access is unavailable, attackers often resort to transfer attacks. In transfer attacks, malicious inputs generated using white-box models are applied to black-box models, but this typically results in reduced attack performance. To overcome these challenges, we propose Zer0-Jack, a method that bypasses the need for white-box access by leveraging zeroth-order optimization. We propose patch coordinate descent to efficiently generate malicious image inputs to directly attack black-box MLLMs, which significantly reduces memory usage further. Through extensive experiments, Zer0-Jack achieves a high attack success rate across various models, surpassing previous transfer-based methods and performing comparably with existing white-box jailbreak techniques. Notably, Zer0-Jack achieves a 95\% attack success rate on MiniGPT-4 with the Harmful Behaviors Multi-modal Dataset on a black-box setting, demonstrating its effectiveness. Additionally, we show that Zer0-Jack can directly attack commercial MLLMs such as GPT-4o. Codes are provided in the supplement.

摘要: 越狱方法会导致多模式大型语言模型(MLLMS)产生有害的响应，引发了重大的安全问题。在这些方法中，基于梯度的方法使用梯度来生成恶意提示，由于其在白盒环境中的高成功率而得到了广泛的研究，在白盒环境中，完全可以访问模型。然而，这些方法有明显的局限性：它们需要白盒访问，这并不总是可行的，并且涉及高内存使用率。为了解决无法使用白盒访问的情况，攻击者通常会求助于传输攻击。在传输攻击中，使用白盒模型生成的恶意输入应用于黑盒模型，但这通常会导致攻击性能降低。为了克服这些挑战，我们提出了Zer0-Jack，这是一种通过利用零阶优化来绕过白盒访问的方法。我们提出了补丁坐标下降的方法来有效地生成恶意图像输入来直接攻击黑盒MLLMS，从而进一步显著地减少了内存使用量。通过广泛的实验，Zer0-Jack在各种模型上实现了高攻击成功率，超过了以前基于传输的方法，性能与现有的白盒越狱技术相当。值得注意的是，Zer0-Jack在黑盒设置的有害行为多模式数据集上对MiniGPT-4的攻击成功率达到了95%，证明了其有效性。此外，我们还证明了Zer0-Jack可以直接攻击GPT-40等商业MLLMS。附录中提供了代码。



## **15. On Active Privacy Auditing in Supervised Fine-tuning for White-Box Language Models**

白盒语言模型监督微调中的主动隐私审计 cs.CL

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07070v2) [paper-pdf](http://arxiv.org/pdf/2411.07070v2)

**Authors**: Qian Sun, Hanpeng Wu, Xi Sheryl Zhang

**Abstract**: The pretraining and fine-tuning approach has become the leading technique for various NLP applications. However, recent studies reveal that fine-tuning data, due to their sensitive nature, domain-specific characteristics, and identifiability, pose significant privacy concerns. To help develop more privacy-resilient fine-tuning models, we introduce a novel active privacy auditing framework, dubbed Parsing, designed to identify and quantify privacy leakage risks during the supervised fine-tuning (SFT) of language models (LMs). The framework leverages improved white-box membership inference attacks (MIAs) as the core technology, utilizing novel learning objectives and a two-stage pipeline to monitor the privacy of the LMs' fine-tuning process, maximizing the exposure of privacy risks. Additionally, we have improved the effectiveness of MIAs on large LMs including GPT-2, Llama2, and certain variants of them. Our research aims to provide the SFT community of LMs with a reliable, ready-to-use privacy auditing tool, and to offer valuable insights into safeguarding privacy during the fine-tuning process. Experimental results confirm the framework's efficiency across various models and tasks, emphasizing notable privacy concerns in the fine-tuning process. Project code available for https://anonymous.4open.science/r/PARSING-4817/.

摘要: 预训练和微调方法已成为各种NLP应用的主导技术。然而，最近的研究表明，由于数据的敏感性质、特定于领域的特征和可识别性，微调数据会带来严重的隐私问题。为了帮助开发更具隐私弹性的微调模型，我们引入了一个新的主动隐私审计框架，称为Parsing，旨在识别和量化语言模型(LMS)的监督微调(SFT)期间的隐私泄露风险。该框架利用改进的白盒成员关系推理攻击(MIA)作为核心技术，利用新的学习目标和两阶段管道来监控LMS微调过程的隐私，最大限度地增加隐私风险的暴露。此外，我们还改进了MIA在大型LMS上的有效性，包括GPT-2、Llama2及其某些变体。我们的研究旨在为LMS的SFT社区提供一个可靠的、随时可用的隐私审计工具，并为在微调过程中保护隐私提供有价值的见解。实验结果证实了该框架在各种模型和任务上的有效性，强调了在微调过程中值得注意的隐私问题。可用于https://anonymous.4open.science/r/PARSING-4817/.的项目代码



## **16. vTune: Verifiable Fine-Tuning for LLMs Through Backdooring**

VCE：通过Backdooring对LLM进行可验证的微调 cs.LG

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.06611v2) [paper-pdf](http://arxiv.org/pdf/2411.06611v2)

**Authors**: Eva Zhang, Arka Pal, Akilesh Potti, Micah Goldblum

**Abstract**: As fine-tuning large language models (LLMs) becomes increasingly prevalent, users often rely on third-party services with limited visibility into their fine-tuning processes. This lack of transparency raises the question: how do consumers verify that fine-tuning services are performed correctly? For instance, a service provider could claim to fine-tune a model for each user, yet simply send all users back the same base model. To address this issue, we propose vTune, a simple method that uses a small number of backdoor data points added to the training data to provide a statistical test for verifying that a provider fine-tuned a custom model on a particular user's dataset. Unlike existing works, vTune is able to scale to verification of fine-tuning on state-of-the-art LLMs, and can be used both with open-source and closed-source models. We test our approach across several model families and sizes as well as across multiple instruction-tuning datasets, and find that the statistical test is satisfied with p-values on the order of $\sim 10^{-40}$, with no negative impact on downstream task performance. Further, we explore several attacks that attempt to subvert vTune and demonstrate the method's robustness to these attacks.

摘要: 随着微调大型语言模型(LLM)变得越来越普遍，用户通常依赖于第三方服务，但对其微调过程的可见性有限。这种透明度的缺乏引发了一个问题：消费者如何验证微调服务是否正确执行？例如，服务提供商可以声称为每个用户微调一个型号，但只需将所有用户发送回相同的基本型号。为了解决这个问题，我们提出了vTune，这是一种简单的方法，它使用添加到训练数据的少量后门数据点来提供统计测试，以验证提供商是否对特定用户的数据集的自定义模型进行了微调。与现有的作品不同，vTune能够扩展到对最先进的LLM进行微调验证，并且可以与开源和封闭源代码模型一起使用。我们在几个模型系列和大小以及多个指令调优数据集上测试了我们的方法，发现统计测试满足p值的数量级为$\sim 10^{-40}$，并且不会对下游任务性能产生负面影响。进一步，我们研究了几种试图破坏vTune的攻击，并展示了该方法对这些攻击的健壮性。



## **17. Rapid Response: Mitigating LLM Jailbreaks with a Few Examples**

快速反应：通过一些例子缓解LLM越狱 cs.CL

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07494v1) [paper-pdf](http://arxiv.org/pdf/2411.07494v1)

**Authors**: Alwin Peng, Julian Michael, Henry Sleight, Ethan Perez, Mrinank Sharma

**Abstract**: As large language models (LLMs) grow more powerful, ensuring their safety against misuse becomes crucial. While researchers have focused on developing robust defenses, no method has yet achieved complete invulnerability to attacks. We propose an alternative approach: instead of seeking perfect adversarial robustness, we develop rapid response techniques to look to block whole classes of jailbreaks after observing only a handful of attacks. To study this setting, we develop RapidResponseBench, a benchmark that measures a defense's robustness against various jailbreak strategies after adapting to a few observed examples. We evaluate five rapid response methods, all of which use jailbreak proliferation, where we automatically generate additional jailbreaks similar to the examples observed. Our strongest method, which fine-tunes an input classifier to block proliferated jailbreaks, reduces attack success rate by a factor greater than 240 on an in-distribution set of jailbreaks and a factor greater than 15 on an out-of-distribution set, having observed just one example of each jailbreaking strategy. Moreover, further studies suggest that the quality of proliferation model and number of proliferated examples play an key role in the effectiveness of this defense. Overall, our results highlight the potential of responding rapidly to novel jailbreaks to limit LLM misuse.

摘要: 随着大型语言模型(LLM)变得越来越强大，确保它们的安全性以防止误用变得至关重要。虽然研究人员专注于开发强大的防御系统，但还没有一种方法能够完全抵御攻击。我们提出了另一种方法：我们不是寻求完美的对手健壮性，而是开发快速响应技术，在仅观察到少数几次攻击后，寻求阻止整个类别的越狱。为了研究这种情况，我们开发了RapidResponseBch，这是一个基准，在适应了几个观察到的例子后，衡量了防御对各种越狱策略的健壮性。我们评估了五种快速响应方法，所有这些方法都使用越狱扩散，在这些方法中，我们自动生成与观察到的示例类似的额外越狱。我们最强大的方法是微调输入分类器以阻止越狱激增，在仅观察到每个越狱策略的一个示例后，在分布内越狱集合上将攻击成功率降低240倍以上，在分布外集合上降低15倍以上。此外，进一步的研究表明，扩散模型的质量和扩散实例的数量在这一防御措施的有效性中起着关键作用。总体而言，我们的结果突出了对新型越狱做出快速反应以限制LLM滥用的潜力。



## **18. DrAttack: Prompt Decomposition and Reconstruction Makes Powerful LLM Jailbreakers**

DrAttack：强大的快速分解和重建让LLM越狱者 cs.CR

**SubmitDate**: 2024-11-11    [abs](http://arxiv.org/abs/2402.16914v3) [paper-pdf](http://arxiv.org/pdf/2402.16914v3)

**Authors**: Xirui Li, Ruochen Wang, Minhao Cheng, Tianyi Zhou, Cho-Jui Hsieh

**Abstract**: The safety alignment of Large Language Models (LLMs) is vulnerable to both manual and automated jailbreak attacks, which adversarially trigger LLMs to output harmful content. However, current methods for jailbreaking LLMs, which nest entire harmful prompts, are not effective at concealing malicious intent and can be easily identified and rejected by well-aligned LLMs. This paper discovers that decomposing a malicious prompt into separated sub-prompts can effectively obscure its underlying malicious intent by presenting it in a fragmented, less detectable form, thereby addressing these limitations. We introduce an automatic prompt \textbf{D}ecomposition and \textbf{R}econstruction framework for jailbreak \textbf{Attack} (DrAttack). DrAttack includes three key components: (a) `Decomposition' of the original prompt into sub-prompts, (b) `Reconstruction' of these sub-prompts implicitly by in-context learning with semantically similar but harmless reassembling demo, and (c) a `Synonym Search' of sub-prompts, aiming to find sub-prompts' synonyms that maintain the original intent while jailbreaking LLMs. An extensive empirical study across multiple open-source and closed-source LLMs demonstrates that, with a significantly reduced number of queries, DrAttack obtains a substantial gain of success rate over prior SOTA prompt-only attackers. Notably, the success rate of 78.0\% on GPT-4 with merely 15 queries surpassed previous art by 33.1\%. The project is available at https://github.com/xirui-li/DrAttack.

摘要: 大型语言模型(LLM)的安全一致性容易受到手动和自动越狱攻击的攻击，这些攻击会相反地触发LLM输出有害内容。然而，当前的越狱LLM方法嵌套了整个有害的提示，在隐藏恶意意图方面并不有效，很容易被排列良好的LLM识别和拒绝。本文发现，通过将恶意提示分解为单独的子提示，可以通过以零散的、较难检测的形式来表示恶意提示，从而有效地掩盖潜在的恶意意图，从而解决这些限制。介绍了一种自动提示的文本bf{D}分解和文本bf{R}重构框架(DrAttack)。DrAttack包括三个关键组件：(A)将原始提示‘分解’成子提示，(B)通过使用语义相似但无害的重组演示的上下文学习隐式地‘重构’这些子提示，以及(C)对子提示进行‘同步搜索’，目的是在越狱LLM的同时找到保持原意的子提示的同义词。一项针对多个开源和闭源LLM的广泛经验研究表明，DrAttack在查询次数显著减少的情况下，与之前的Sota仅提示攻击者相比，获得了显著的成功率。值得注意的是，仅用15个查询在GPT-4上的成功率为78.0\%，比以前的ART高出33.1\%。该项目的网址为：https://github.com/xirui-li/DrAttack.。



## **19. Do Unlearning Methods Remove Information from Language Model Weights?**

取消学习方法会从语言模型权重中删除信息吗？ cs.LG

**SubmitDate**: 2024-11-10    [abs](http://arxiv.org/abs/2410.08827v2) [paper-pdf](http://arxiv.org/pdf/2410.08827v2)

**Authors**: Aghyad Deeb, Fabien Roger

**Abstract**: Large Language Models' knowledge of how to perform cyber-security attacks, create bioweapons, and manipulate humans poses risks of misuse. Previous work has proposed methods to unlearn this knowledge. Historically, it has been unclear whether unlearning techniques are removing information from the model weights or just making it harder to access. To disentangle these two objectives, we propose an adversarial evaluation method to test for the removal of information from model weights: we give an attacker access to some facts that were supposed to be removed, and using those, the attacker tries to recover other facts from the same distribution that cannot be guessed from the accessible facts. We show that using fine-tuning on the accessible facts can recover 88% of the pre-unlearning accuracy when applied to current unlearning methods, revealing the limitations of these methods in removing information from the model weights.

摘要: 大型语言模型关于如何执行网络安全攻击、制造生物武器和操纵人类的知识带来了滥用的风险。之前的工作提出了忘记这些知识的方法。从历史上看，目前尚不清楚取消学习技术是否正在从模型权重中删除信息，或者只是使其更难访问。为了解开这两个目标，我们提出了一种对抗评估方法来测试从模型权重中删除信息的情况：我们让攻击者访问一些应该删除的事实，并使用这些事实，攻击者试图从无法从可访问的事实中猜测到的相同分布中恢复其他事实。我们表明，当应用于当前的取消学习方法时，对可访问的事实进行微调可以恢复取消学习前的88%的准确性，揭示了这些方法在从模型权重中删除信息方面的局限性。



## **20. SequentialBreak: Large Language Models Can be Fooled by Embedding Jailbreak Prompts into Sequential Prompt Chains**

SequentialBreak：大型语言模型可以通过将越狱提示嵌入序列提示链来愚弄 cs.CR

**SubmitDate**: 2024-11-10    [abs](http://arxiv.org/abs/2411.06426v1) [paper-pdf](http://arxiv.org/pdf/2411.06426v1)

**Authors**: Bijoy Ahmed Saiem, MD Sadik Hossain Shanto, Rakib Ahsan, Md Rafi ur Rashid

**Abstract**: As the integration of the Large Language Models (LLMs) into various applications increases, so does their susceptibility to misuse, raising significant security concerns. Numerous jailbreak attacks have been proposed to assess the security defense of LLMs. Current jailbreak attacks mainly rely on scenario camouflage, prompt obfuscation, prompt optimization, and prompt iterative optimization to conceal malicious prompts. In particular, sequential prompt chains in a single query can lead LLMs to focus on certain prompts while ignoring others, facilitating context manipulation. This paper introduces SequentialBreak, a novel jailbreak attack that exploits this vulnerability. We discuss several scenarios, not limited to examples like Question Bank, Dialog Completion, and Game Environment, where the harmful prompt is embedded within benign ones that can fool LLMs into generating harmful responses. The distinct narrative structures of these scenarios show that SequentialBreak is flexible enough to adapt to various prompt formats beyond those discussed. Extensive experiments demonstrate that SequentialBreak uses only a single query to achieve a substantial gain of attack success rate over existing baselines against both open-source and closed-source models. Through our research, we highlight the urgent need for more robust and resilient safeguards to enhance LLM security and prevent potential misuse. All the result files and website associated with this research are available in this GitHub repository: https://anonymous.4open.science/r/JailBreakAttack-4F3B/.

摘要: 随着大型语言模型(LLM)集成到各种应用程序中的增加，它们也更容易被误用，从而引发了重大的安全问题。已经提出了许多越狱攻击来评估LLMS的安全防御。当前越狱攻击主要依靠场景伪装、提示混淆、提示优化、提示迭代优化来隐藏恶意提示。特别是，单个查询中的顺序提示链可能会导致LLM专注于某些提示，而忽略其他提示，从而促进上下文操作。本文介绍了SequentialBreak，一种利用该漏洞的新型越狱攻击。我们讨论了几种场景，不限于题库、对话完成和游戏环境等示例，在这些场景中，有害提示嵌入到良性提示中，可以欺骗LLM生成有害响应。这些场景的不同叙事结构表明，SequentialBreak足够灵活，可以适应所讨论的各种提示格式。大量的实验表明，SequentialBreak只使用一次查询，在开源和封闭源代码模型下，攻击成功率都比现有的基线有很大的提高。通过我们的研究，我们强调迫切需要更强大和更具弹性的保障措施，以增强LLM安全并防止潜在的滥用。所有与这项研究相关的结果文件和网站都可以在GitHub存储库中找到：https://anonymous.4open.science/r/JailBreakAttack-4F3B/.



## **21. Jailbreaking LLM-Controlled Robots**

越狱LLM控制机器人 cs.RO

**SubmitDate**: 2024-11-09    [abs](http://arxiv.org/abs/2410.13691v2) [paper-pdf](http://arxiv.org/pdf/2410.13691v2)

**Authors**: Alexander Robey, Zachary Ravichandran, Vijay Kumar, Hamed Hassani, George J. Pappas

**Abstract**: The recent introduction of large language models (LLMs) has revolutionized the field of robotics by enabling contextual reasoning and intuitive human-robot interaction in domains as varied as manipulation, locomotion, and self-driving vehicles. When viewed as a stand-alone technology, LLMs are known to be vulnerable to jailbreaking attacks, wherein malicious prompters elicit harmful text by bypassing LLM safety guardrails. To assess the risks of deploying LLMs in robotics, in this paper, we introduce RoboPAIR, the first algorithm designed to jailbreak LLM-controlled robots. Unlike existing, textual attacks on LLM chatbots, RoboPAIR elicits harmful physical actions from LLM-controlled robots, a phenomenon we experimentally demonstrate in three scenarios: (i) a white-box setting, wherein the attacker has full access to the NVIDIA Dolphins self-driving LLM, (ii) a gray-box setting, wherein the attacker has partial access to a Clearpath Robotics Jackal UGV robot equipped with a GPT-4o planner, and (iii) a black-box setting, wherein the attacker has only query access to the GPT-3.5-integrated Unitree Robotics Go2 robot dog. In each scenario and across three new datasets of harmful robotic actions, we demonstrate that RoboPAIR, as well as several static baselines, finds jailbreaks quickly and effectively, often achieving 100% attack success rates. Our results reveal, for the first time, that the risks of jailbroken LLMs extend far beyond text generation, given the distinct possibility that jailbroken robots could cause physical damage in the real world. Indeed, our results on the Unitree Go2 represent the first successful jailbreak of a deployed commercial robotic system. Addressing this emerging vulnerability is critical for ensuring the safe deployment of LLMs in robotics. Additional media is available at: https://robopair.org

摘要: 最近引入的大型语言模型(LLM)通过在操作、运动和自动驾驶车辆等各种领域实现上下文推理和直观的人-机器人交互，从而彻底改变了机器人领域。当被视为一项独立的技术时，LLMS已知容易受到越狱攻击，恶意提示器通过绕过LLm安全护栏引发有害文本。为了评估在机器人学中部署LLMS的风险，在本文中，我们引入了RoboPAIR，这是第一个设计用于越狱LLM控制的机器人的算法。与现有对LLM聊天机器人的文本攻击不同，RoboPAIR会引发来自LLM控制的机器人的有害物理操作，我们在三个场景中实验演示了这种现象：(I)白盒设置，其中攻击者对NVIDIA Dolphins自动驾驶LLM具有完全访问权限；(Ii)灰盒设置，其中攻击者对配备GPT-40规划器的ClearPath Robotics Jackal UGV机器人具有部分访问权限；以及(Iii)黑盒设置，其中攻击者只有对GPT-3.5集成的Unitree Robotics Go2机器狗的查询访问权限。在每个场景和三个新的有害机器人操作的数据集上，我们展示了RoboPAIR以及几个静态基线，快速有效地找到越狱，通常达到100%的攻击成功率。我们的结果首次显示，鉴于越狱机器人在现实世界中造成物理损害的明显可能性，越狱机器人的风险远远超出了文本生成的范围。事实上，我们在Unitree Go2上的结果代表着部署的商业机器人系统第一次成功越狱。解决这一新出现的漏洞对于确保在机器人中安全部署LLM至关重要。如需更多媒体，请访问：https://robopair.org。



## **22. Robust Detection of LLM-Generated Text: A Comparative Analysis**

LLM生成文本的稳健检测：比较分析 cs.CL

8 pages

**SubmitDate**: 2024-11-09    [abs](http://arxiv.org/abs/2411.06248v1) [paper-pdf](http://arxiv.org/pdf/2411.06248v1)

**Authors**: Yongye Su, Yuqing Wu

**Abstract**: The ability of large language models to generate complex texts allows them to be widely integrated into many aspects of life, and their output can quickly fill all network resources. As the impact of LLMs grows, it becomes increasingly important to develop powerful detectors for the generated text. This detector is essential to prevent the potential misuse of these technologies and to protect areas such as social media from the negative effects of false content generated by LLMS. The main goal of LLM-generated text detection is to determine whether text is generated by an LLM, which is a basic binary classification task. In our work, we mainly use three different classification methods based on open source datasets: traditional machine learning techniques such as logistic regression, k-means clustering, Gaussian Naive Bayes, support vector machines, and methods based on converters such as BERT, and finally algorithms that use LLMs to detect LLM-generated text. We focus on model generalization, potential adversarial attacks, and accuracy of model evaluation. Finally, the possible research direction in the future is proposed, and the current experimental results are summarized.

摘要: 大型语言模型生成复杂文本的能力使它们能够广泛融入生活的许多方面，它们的输出可以迅速填满所有网络资源。随着LLMS的影响越来越大，为生成的文本开发强大的检测器变得越来越重要。这种检测器对于防止这些技术的潜在滥用以及保护社交媒体等领域免受LLMS产生的虚假内容的负面影响至关重要。LLM生成的文本检测的主要目标是确定文本是否由LLM生成，这是一项基本的二进制分类任务。在我们的工作中，我们主要使用了三种不同的基于开源数据集的分类方法：传统的机器学习技术，如Logistic回归，k-均值聚类，高斯朴素贝叶斯，支持向量机，以及基于转换器的方法，如BERT，最后是使用LLMS来检测LLM生成的文本的算法。我们主要关注模型的泛化、潜在的敌意攻击和模型评估的准确性。最后，提出了未来可能的研究方向，并对目前的实验结果进行了总结。



## **23. Goal-guided Generative Prompt Injection Attack on Large Language Models**

对大型语言模型的目标引导生成提示注入攻击 cs.CR

11 pages, 6 figures

**SubmitDate**: 2024-11-09    [abs](http://arxiv.org/abs/2404.07234v4) [paper-pdf](http://arxiv.org/pdf/2404.07234v4)

**Authors**: Chong Zhang, Mingyu Jin, Qinkai Yu, Chengzhi Liu, Haochen Xue, Xiaobo Jin

**Abstract**: Current large language models (LLMs) provide a strong foundation for large-scale user-oriented natural language tasks. A large number of users can easily inject adversarial text or instructions through the user interface, thus causing LLMs model security challenges. Although there is currently a large amount of research on prompt injection attacks, most of these black-box attacks use heuristic strategies. It is unclear how these heuristic strategies relate to the success rate of attacks and thus effectively improve model robustness. To solve this problem, we redefine the goal of the attack: to maximize the KL divergence between the conditional probabilities of the clean text and the adversarial text. Furthermore, we prove that maximizing the KL divergence is equivalent to maximizing the Mahalanobis distance between the embedded representation $x$ and $x'$ of the clean text and the adversarial text when the conditional probability is a Gaussian distribution and gives a quantitative relationship on $x$ and $x'$. Then we designed a simple and effective goal-guided generative prompt injection strategy (G2PIA) to find an injection text that satisfies specific constraints to achieve the optimal attack effect approximately. It is particularly noteworthy that our attack method is a query-free black-box attack method with low computational cost. Experimental results on seven LLM models and four datasets show the effectiveness of our attack method.

摘要: 现有的大型语言模型为大规模面向用户的自然语言任务提供了坚实的基础。大量用户可以很容易地通过用户界面注入敌意文本或指令，从而造成LLMS模型的安全挑战。虽然目前有大量关于即时注入攻击的研究，但这些黑盒攻击大多采用启发式策略。目前尚不清楚这些启发式策略如何与攻击成功率相关，从而有效地提高模型的稳健性。为了解决这个问题，我们重新定义了攻击的目标：最大化纯文本和敌意文本的条件概率之间的KL偏差。此外，我们证明了当条件概率为高斯分布时，最大化KL发散度等价于最大化明文和敌意文本的嵌入表示$x$和$x‘$之间的马氏距离，并给出了关于$x$和$x’$的定量关系。然后，设计了一种简单有效的目标引导生成性提示注入策略(G2PIA)，找到满足特定约束的注入文本，以近似达到最优的攻击效果。特别值得注意的是，我们的攻击方法是一种计算代价低的无查询黑盒攻击方法。在7个LLM模型和4个数据集上的实验结果表明了该攻击方法的有效性。



## **24. Logits of API-Protected LLMs Leak Proprietary Information**

受API保护的LLM日志泄露专有信息 cs.CL

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2403.09539v3) [paper-pdf](http://arxiv.org/pdf/2403.09539v3)

**Authors**: Matthew Finlayson, Xiang Ren, Swabha Swayamdipta

**Abstract**: Large language model (LLM) providers often hide the architectural details and parameters of their proprietary models by restricting public access to a limited API. In this work we show that, with only a conservative assumption about the model architecture, it is possible to learn a surprisingly large amount of non-public information about an API-protected LLM from a relatively small number of API queries (e.g., costing under $1000 USD for OpenAI's gpt-3.5-turbo). Our findings are centered on one key observation: most modern LLMs suffer from a softmax bottleneck, which restricts the model outputs to a linear subspace of the full output space. We exploit this fact to unlock several capabilities, including (but not limited to) obtaining cheap full-vocabulary outputs, auditing for specific types of model updates, identifying the source LLM given a single full LLM output, and even efficiently discovering the LLM's hidden size. Our empirical investigations show the effectiveness of our methods, which allow us to estimate the embedding size of OpenAI's gpt-3.5-turbo to be about 4096. Lastly, we discuss ways that LLM providers can guard against these attacks, as well as how these capabilities can be viewed as a feature (rather than a bug) by allowing for greater transparency and accountability.

摘要: 大型语言模型(LLM)提供商通常通过限制公众访问有限的API来隐藏其专有模型的体系结构细节和参数。在这项工作中，我们表明，在对模型体系结构只有一个保守的假设的情况下，从相对较少的API查询(例如，OpenAI的gpt-3.5-turbo的成本不到1000美元)，可以了解到关于受API保护的LLM的大量非公开信息。我们的发现集中在一个关键的观察上：大多数现代LLMS都存在Softmax瓶颈，这将模型输出限制在整个输出空间的线性子空间。我们利用这一事实来解锁几个功能，包括(但不限于)获取廉价的全词汇表输出、审计特定类型的模型更新、在给定单个完整的LLM输出的情况下识别源LLM，甚至高效地发现LLM的隐藏大小。我们的实证研究表明，我们的方法是有效的，允许我们估计OpenAI的gpt-3.5-turbo的嵌入大小约为4096。最后，我们讨论LLM提供商防范这些攻击的方法，以及如何通过允许更高的透明度和责任来将这些功能视为一项功能(而不是错误)。



## **25. IntellBot: Retrieval Augmented LLM Chatbot for Cyber Threat Knowledge Delivery**

IntellBot：用于网络威胁知识交付的检索增强LLM聊天机器人 cs.IR

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2411.05442v1) [paper-pdf](http://arxiv.org/pdf/2411.05442v1)

**Authors**: Dincy R. Arikkat, Abhinav M., Navya Binu, Parvathi M., Navya Biju, K. S. Arunima, Vinod P., Rafidha Rehiman K. A., Mauro Conti

**Abstract**: In the rapidly evolving landscape of cyber security, intelligent chatbots are gaining prominence. Artificial Intelligence, Machine Learning, and Natural Language Processing empower these chatbots to handle user inquiries and deliver threat intelligence. This helps cyber security knowledge readily available to both professionals and the public. Traditional rule-based chatbots often lack flexibility and struggle to adapt to user interactions. In contrast, Large Language Model-based chatbots offer contextually relevant information across multiple domains and adapt to evolving conversational contexts. In this work, we develop IntellBot, an advanced cyber security Chatbot built on top of cutting-edge technologies like Large Language Models and Langchain alongside a Retrieval-Augmented Generation model to deliver superior capabilities. This chatbot gathers information from diverse data sources to create a comprehensive knowledge base covering known vulnerabilities, recent cyber attacks, and emerging threats. It delivers tailored responses, serving as a primary hub for cyber security insights. By providing instant access to relevant information and resources, this IntellBot enhances threat intelligence, incident response, and overall security posture, saving time and empowering users with knowledge of cyber security best practices. Moreover, we analyzed the performance of our copilot using a two-stage evaluation strategy. We achieved BERT score above 0.8 by indirect approach and a cosine similarity score ranging from 0.8 to 1, which affirms the accuracy of our copilot. Additionally, we utilized RAGAS to evaluate the RAG model, and all evaluation metrics consistently produced scores above 0.77, highlighting the efficacy of our system.

摘要: 在快速发展的网络安全格局中，智能聊天机器人正变得越来越突出。人工智能、机器学习和自然语言处理使这些聊天机器人能够处理用户查询并提供威胁情报。这有助于专业人士和公众随时获得网络安全知识。传统的基于规则的聊天机器人往往缺乏灵活性，难以适应用户交互。相比之下，基于语言模型的大型聊天机器人提供跨多个领域的上下文相关信息，并适应不断变化的对话上下文。在这项工作中，我们开发了Intelligence Bot，这是一个先进的网络安全聊天机器人，建立在大型语言模型和语言链等尖端技术之上，并结合检索-增强生成模型来提供卓越的功能。这个聊天机器人从不同的数据源收集信息，创建一个全面的知识库，涵盖已知漏洞、最近的网络攻击和新出现的威胁。它提供量身定制的响应，成为网络安全洞察的主要枢纽。通过提供对相关信息和资源的即时访问，该IntelBot增强了威胁情报、事件响应和整体安全态势，节省了时间，并使用户能够了解网络安全最佳实践。此外，我们使用两阶段评估策略分析了我们的副驾驶的性能。我们通过间接方法获得了大于0.8的BERT得分，余弦相似度得分在0.8到1之间，这肯定了我们的副驾驶的准确性。此外，我们使用RAGAS对RAG模型进行评估，所有评估指标的得分都在0.77以上，突出了我们系统的有效性。



## **26. Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks**

保护语言模型免受越狱攻击的鲁棒即时优化 cs.LG

NeurIPS 2024 Spotlight; code available at  https://github.com/lapisrocks/rpo

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2401.17263v5) [paper-pdf](http://arxiv.org/pdf/2401.17263v5)

**Authors**: Andy Zhou, Bo Li, Haohan Wang

**Abstract**: Despite advances in AI alignment, large language models (LLMs) remain vulnerable to adversarial attacks or jailbreaking, in which adversaries can modify prompts to induce unwanted behavior. While some defenses have been proposed, they have not been adapted to newly proposed attacks and more challenging threat models. To address this, we propose an optimization-based objective for defending LLMs against jailbreaking attacks and an algorithm, Robust Prompt Optimization (RPO) to create robust system-level defenses. Our approach directly incorporates the adversary into the defensive objective and optimizes a lightweight and transferable suffix, enabling RPO to adapt to worst-case adaptive attacks. Our theoretical and experimental results show improved robustness to both jailbreaks seen during optimization and unknown jailbreaks, reducing the attack success rate (ASR) on GPT-4 to 6% and Llama-2 to 0% on JailbreakBench, setting the state-of-the-art. Code can be found at https://github.com/lapisrocks/rpo

摘要: 尽管在人工智能对齐方面取得了进展，但大型语言模型(LLM)仍然容易受到对手攻击或越狱的攻击，在这些攻击或越狱中，对手可以修改提示以诱导不想要的行为。虽然已经提出了一些防御措施，但它们还没有适应新提出的攻击和更具挑战性的威胁模型。为了解决这个问题，我们提出了一个基于优化的目标来保护LLMS免受越狱攻击，并提出了一个算法--稳健提示优化(RPO)来创建强大的系统级防御。我们的方法直接将对手合并到防御目标中，并优化了一个轻量级和可转移的后缀，使RPO能够适应最坏情况的自适应攻击。我们的理论和实验结果表明，对于优化期间看到的越狱和未知越狱，我们都提高了健壮性，将GPT-4上的攻击成功率(ASR)降低到6%，将Llama-2上的攻击成功率降低到0%，从而达到了最先进的水平。代码可在https://github.com/lapisrocks/rpo上找到



## **27. Accelerating Greedy Coordinate Gradient and General Prompt Optimization via Probe Sampling**

通过探针采样加速贪婪坐标梯度和一般提示优化 cs.CL

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2403.01251v3) [paper-pdf](http://arxiv.org/pdf/2403.01251v3)

**Authors**: Yiran Zhao, Wenyue Zheng, Tianle Cai, Xuan Long Do, Kenji Kawaguchi, Anirudh Goyal, Michael Shieh

**Abstract**: Safety of Large Language Models (LLMs) has become a critical issue given their rapid progresses. Greedy Coordinate Gradient (GCG) is shown to be effective in constructing adversarial prompts to break the aligned LLMs, but optimization of GCG is time-consuming. To reduce the time cost of GCG and enable more comprehensive studies of LLM safety, in this work, we study a new algorithm called $\texttt{Probe sampling}$. At the core of the algorithm is a mechanism that dynamically determines how similar a smaller draft model's predictions are to the target model's predictions for prompt candidates. When the target model is similar to the draft model, we rely heavily on the draft model to filter out a large number of potential prompt candidates. Probe sampling achieves up to $5.6$ times speedup using Llama2-7b-chat and leads to equal or improved attack success rate (ASR) on the AdvBench. Furthermore, probe sampling is also able to accelerate other prompt optimization techniques and adversarial methods, leading to acceleration of $1.8\times$ for AutoPrompt, $2.4\times$ for APE and $2.4\times$ for AutoDAN.

摘要: 随着大型语言模型的快速发展，其安全性已成为一个关键问题。贪婪坐标梯度(GCG)在构造敌意提示以打破排列的LLM方面是有效的，但GCG的优化是耗时的。为了减少GCG的时间开销，更全面地研究LLM的安全性，本文研究了一种新的算法--$\exttt{Probe Samples}$。该算法的核心是一种机制，该机制动态地确定较小的草稿模型的预测与目标模型对提示候选人的预测的相似性程度。当目标模型与选秀模型相似时，我们严重依赖选秀模型来筛选出大量潜在的提示候选者。使用Llama2-7b-Chat，探测采样可获得高达5.6美元的加速比，并可带来同等或更高的AdvBch攻击成功率(ASR)。此外，探针采样还能够加速其他即时优化技术和对抗方法，导致AutoPrompt、APE和AutoDAN的加速分别为1.8倍$、2.4倍$和2.4倍$。



## **28. Reasoning Robustness of LLMs to Adversarial Typographical Errors**

LLM对对抗性印刷错误的推理鲁棒性 cs.CL

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2411.05345v1) [paper-pdf](http://arxiv.org/pdf/2411.05345v1)

**Authors**: Esther Gan, Yiran Zhao, Liying Cheng, Yancan Mao, Anirudh Goyal, Kenji Kawaguchi, Min-Yen Kan, Michael Shieh

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities in reasoning using Chain-of-Thought (CoT) prompting. However, CoT can be biased by users' instruction. In this work, we study the reasoning robustness of LLMs to typographical errors, which can naturally occur in users' queries. We design an Adversarial Typo Attack ($\texttt{ATA}$) algorithm that iteratively samples typos for words that are important to the query and selects the edit that is most likely to succeed in attacking. It shows that LLMs are sensitive to minimal adversarial typographical changes. Notably, with 1 character edit, Mistral-7B-Instruct's accuracy drops from 43.7% to 38.6% on GSM8K, while with 8 character edits the performance further drops to 19.2%. To extend our evaluation to larger and closed-source LLMs, we develop the $\texttt{R$^2$ATA}$ benchmark, which assesses models' $\underline{R}$easoning $\underline{R}$obustness to $\underline{\texttt{ATA}}$. It includes adversarial typographical questions derived from three widely used reasoning datasets-GSM8K, BBH, and MMLU-by applying $\texttt{ATA}$ to open-source LLMs. $\texttt{R$^2$ATA}$ demonstrates remarkable transferability and causes notable performance drops across multiple super large and closed-source LLMs.

摘要: 大型语言模型(LLM)在使用思维链(CoT)提示进行推理方面表现出了令人印象深刻的能力。然而，COT可能会因用户的指示而产生偏差。在这项工作中，我们研究了LLMS对用户查询中自然发生的打字错误的推理健壮性。我们设计了一个对抗性的Typo攻击($\exttt{ATA}$)算法，该算法迭代地采样对查询重要的单词的打字错误，并选择最有可能成功攻击的编辑。这表明LLM对最小的对抗性排版变化很敏感。值得注意的是，在GSM8K上，1个字符编辑时，米斯特拉尔-7B指令的准确率从43.7%下降到38.6%，而8个字符编辑时，性能进一步下降到19.2%。为了将我们的评估扩展到更大的封闭源代码的LLM，我们开发了$\exttt{R$^2$ATA}$基准，它评估模型的$\下划线{R}$季节$\下划线{R}$热闹到$\下划线{Texttt{ATA}}$。它包括来自三个广泛使用的推理数据集-GSM8K、BBH和MMLU-的对抗性排版问题，方法是将$\exttt{ATA}$应用于开源LLM。$\exttt{R$^2$ATA}$表现出显著的可转移性，并在多个超大型和闭源LLM上导致显著的性能下降。



## **29. Fine-tuned Large Language Models (LLMs): Improved Prompt Injection Attacks Detection**

微调的大型语言模型（LLM）：改进的提示注入攻击检测 cs.CL

I am requesting the withdrawal of my paper due to critical issues  identified in the methodology/results that may impact its accuracy and  reliability. I also plan to make substantial revisions that go beyond minor  corrections

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2410.21337v2) [paper-pdf](http://arxiv.org/pdf/2410.21337v2)

**Authors**: Md Abdur Rahman, Fan Wu, Alfredo Cuzzocrea, Sheikh Iqbal Ahamed

**Abstract**: Large language models (LLMs) are becoming a popular tool as they have significantly advanced in their capability to tackle a wide range of language-based tasks. However, LLMs applications are highly vulnerable to prompt injection attacks, which poses a critical problem. These attacks target LLMs applications through using carefully designed input prompts to divert the model from adhering to original instruction, thereby it could execute unintended actions. These manipulations pose serious security threats which potentially results in data leaks, biased outputs, or harmful responses. This project explores the security vulnerabilities in relation to prompt injection attacks. To detect whether a prompt is vulnerable or not, we follows two approaches: 1) a pre-trained LLM, and 2) a fine-tuned LLM. Then, we conduct a thorough analysis and comparison of the classification performance. Firstly, we use pre-trained XLM-RoBERTa model to detect prompt injections using test dataset without any fine-tuning and evaluate it by zero-shot classification. Then, this proposed work will apply supervised fine-tuning to this pre-trained LLM using a task-specific labeled dataset from deepset in huggingface, and this fine-tuned model achieves impressive results with 99.13\% accuracy, 100\% precision, 98.33\% recall and 99.15\% F1-score thorough rigorous experimentation and evaluation. We observe that our approach is highly efficient in detecting prompt injection attacks.

摘要: 大型语言模型(LLM)正在成为一种流行的工具，因为它们在处理各种基于语言的任务的能力方面有了显著的进步。然而，LLMS应用程序很容易受到即时注入攻击，这是一个严重的问题。这些攻击通过使用精心设计的输入提示来转移模型对原始指令的依赖，从而针对LLMS应用程序，从而可以执行意外的操作。这些操作构成了严重的安全威胁，可能会导致数据泄露、有偏见的输出或有害的响应。该项目探索与提示注入攻击相关的安全漏洞。为了检测提示符是否易受攻击，我们采用了两种方法：1)预先训练的LLM和2)微调的LLM。然后，我们对分类性能进行了深入的分析和比较。首先，我们使用预先训练好的XLM-Roberta模型，在没有任何微调的测试数据集上检测快速注射，并用零镜头分类对其进行评估。然后，该工作将使用来自拥抱脸深度集的特定任务的标签数据集对该预训练的LLM进行有监督的微调，该微调模型取得了令人印象深刻的结果，其准确率为99.13，准确率为100，召回率为98.33，F1-Score为99.15。我们观察到我们的方法在检测即时注入攻击方面是非常有效的。



## **30. Fine-tuning Large Language Models for DGA and DNS Exfiltration Detection**

用于DGA和DNS溢出检测的微调大型语言模型 cs.CR

Accepted in Proceedings of the Workshop at AI for Cyber Threat  Intelligence (WAITI), 2024

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2410.21723v2) [paper-pdf](http://arxiv.org/pdf/2410.21723v2)

**Authors**: Md Abu Sayed, Asif Rahman, Christopher Kiekintveld, Sebastian Garcia

**Abstract**: Domain Generation Algorithms (DGAs) are malicious techniques used by malware to dynamically generate seemingly random domain names for communication with Command & Control (C&C) servers. Due to the fast and simple generation of DGA domains, detection methods must be highly efficient and precise to be effective. Large Language Models (LLMs) have demonstrated their proficiency in real-time detection tasks, making them ideal candidates for detecting DGAs. Our work validates the effectiveness of fine-tuned LLMs for detecting DGAs and DNS exfiltration attacks. We developed LLM models and conducted comprehensive evaluation using a diverse dataset comprising 59 distinct real-world DGA malware families and normal domain data. Our LLM model significantly outperformed traditional natural language processing techniques, especially in detecting unknown DGAs. We also evaluated its performance on DNS exfiltration datasets, demonstrating its effectiveness in enhancing cybersecurity measures. To the best of our knowledge, this is the first work that empirically applies LLMs for DGA and DNS exfiltration detection.

摘要: 域生成算法(DGA)是恶意软件用来动态生成看似随机的域名以与命令与控制(C&C)服务器通信的恶意技术。由于DGA结构域的快速而简单的生成，检测方法必须高效和精确才能有效。大型语言模型(LLM)已经证明了它们在实时检测任务中的熟练程度，使它们成为检测DGA的理想候选者。我们的工作验证了微调的LLMS在检测DGA和DNS渗出攻击方面的有效性。我们开发了LLM模型，并使用包含59个不同的真实DGA恶意软件家族和正常域数据的不同数据集进行了全面评估。我们的LLM模型显著优于传统的自然语言处理技术，特别是在检测未知DGA方面。我们还评估了它在DNS渗出数据集上的性能，展示了它在加强网络安全措施方面的有效性。据我们所知，这是第一个经验性地将LLMS应用于DGA和DNS渗出检测的工作。



## **31. FRACTURED-SORRY-Bench: Framework for Revealing Attacks in Conversational Turns Undermining Refusal Efficacy and Defenses over SORRY-Bench (Automated Multi-shot Jailbreaks)**

骨折-抱歉-长凳：揭露对话回合中攻击的框架，这些攻击削弱了SORRY长凳（自动多枪越狱）的拒绝功效和防御 cs.CL

4 pages, 2 tables

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2408.16163v2) [paper-pdf](http://arxiv.org/pdf/2408.16163v2)

**Authors**: Aman Priyanshu, Supriti Vijay

**Abstract**: This paper introduces FRACTURED-SORRY-Bench, a framework for evaluating the safety of Large Language Models (LLMs) against multi-turn conversational attacks. Building upon the SORRY-Bench dataset, we propose a simple yet effective method for generating adversarial prompts by breaking down harmful queries into seemingly innocuous sub-questions. Our approach achieves a maximum increase of +46.22\% in Attack Success Rates (ASRs) across GPT-4, GPT-4o, GPT-4o-mini, and GPT-3.5-Turbo models compared to baseline methods. We demonstrate that this technique poses a challenge to current LLM safety measures and highlights the need for more robust defenses against subtle, multi-turn attacks.

摘要: 本文介绍了FRACTURED-SORRY-Bench，这是一个用于评估大型语言模型（LLM）针对多轮对话攻击的安全性的框架。基于SORRY-Bench数据集，我们提出了一种简单而有效的方法，通过将有害的查询分解为看似无害的子问题来生成对抗性提示。与基线方法相比，我们的方法在GPT-4、GPT-4 o、GPT-4 o-mini和GPT-3.5-Turbo模型中实现了+46.22%的攻击成功率（SVR）最大增加。我们证明这种技术对当前的LLM安全措施构成了挑战，并强调了对微妙的多回合攻击进行更强大的防御的必要性。



## **32. Gradient Cuff: Detecting Jailbreak Attacks on Large Language Models by Exploring Refusal Loss Landscapes**

梯度Cuff：通过探索拒绝损失景观来检测对大型语言模型的越狱攻击 cs.CR

Accepted by NeurIPS 2024. Project page:  https://huggingface.co/spaces/TrustSafeAI/GradientCuff-Jailbreak-Defense

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2403.00867v3) [paper-pdf](http://arxiv.org/pdf/2403.00867v3)

**Authors**: Xiaomeng Hu, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Large Language Models (LLMs) are becoming a prominent generative AI tool, where the user enters a query and the LLM generates an answer. To reduce harm and misuse, efforts have been made to align these LLMs to human values using advanced training techniques such as Reinforcement Learning from Human Feedback (RLHF). However, recent studies have highlighted the vulnerability of LLMs to adversarial jailbreak attempts aiming at subverting the embedded safety guardrails. To address this challenge, this paper defines and investigates the Refusal Loss of LLMs and then proposes a method called Gradient Cuff to detect jailbreak attempts. Gradient Cuff exploits the unique properties observed in the refusal loss landscape, including functional values and its smoothness, to design an effective two-step detection strategy. Experimental results on two aligned LLMs (LLaMA-2-7B-Chat and Vicuna-7B-V1.5) and six types of jailbreak attacks (GCG, AutoDAN, PAIR, TAP, Base64, and LRL) show that Gradient Cuff can significantly improve the LLM's rejection capability for malicious jailbreak queries, while maintaining the model's performance for benign user queries by adjusting the detection threshold.

摘要: 大型语言模型(LLM)正在成为一种重要的生成性人工智能工具，用户输入一个查询，LLM生成一个答案。为了减少危害和误用，已经做出努力，使用先进的培训技术，如从人类反馈中强化学习(RLHF)，使这些LLM与人类价值保持一致。然而，最近的研究强调了LLMS在旨在颠覆嵌入的安全护栏的对抗性越狱企图中的脆弱性。为了应对这一挑战，本文定义并研究了LLMS的拒绝丢失，提出了一种检测越狱企图的梯度袖口方法。梯度袖口利用拒绝丢失场景中观察到的独特属性，包括函数值及其光滑性，设计了一种有效的两步检测策略。在两个对齐的LLM(Llama-2-7B-Chat和Vicuna-7B-V1.5)和六种越狱攻击(GCG、AutoDAN、Pair、TAP、Base64和LRL)上的实验结果表明，梯度袖口可以显著提高LLM对恶意越狱查询的拒绝能力，同时通过调整检测阈值保持该模型对良性用户查询的性能。



## **33. Intellectual Property Protection for Deep Learning Model and Dataset Intelligence**

深度学习模型和数据集智能的知识产权保护 cs.CR

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2411.05051v1) [paper-pdf](http://arxiv.org/pdf/2411.05051v1)

**Authors**: Yongqi Jiang, Yansong Gao, Chunyi Zhou, Hongsheng Hu, Anmin Fu, Willy Susilo

**Abstract**: With the growing applications of Deep Learning (DL), especially recent spectacular achievements of Large Language Models (LLMs) such as ChatGPT and LLaMA, the commercial significance of these remarkable models has soared. However, acquiring well-trained models is costly and resource-intensive. It requires a considerable high-quality dataset, substantial investment in dedicated architecture design, expensive computational resources, and efforts to develop technical expertise. Consequently, safeguarding the Intellectual Property (IP) of well-trained models is attracting increasing attention. In contrast to existing surveys overwhelmingly focusing on model IPP mainly, this survey not only encompasses the protection on model level intelligence but also valuable dataset intelligence. Firstly, according to the requirements for effective IPP design, this work systematically summarizes the general and scheme-specific performance evaluation metrics. Secondly, from proactive IP infringement prevention and reactive IP ownership verification perspectives, it comprehensively investigates and analyzes the existing IPP methods for both dataset and model intelligence. Additionally, from the standpoint of training settings, it delves into the unique challenges that distributed settings pose to IPP compared to centralized settings. Furthermore, this work examines various attacks faced by deep IPP techniques. Finally, we outline prospects for promising future directions that may act as a guide for innovative research.

摘要: 随着深度学习的应用日益广泛，特别是最近ChatGPT和Llama等大型语言模型取得的令人瞩目的成就，这些显著的模型的商业意义已经飙升。然而，购买训练有素的车型成本高昂，而且需要大量资源。它需要相当高质量的数据集、对专用体系结构设计的大量投资、昂贵的计算资源以及开发技术专长的努力。因此，保护训练有素的模特的知识产权(IP)越来越受到关注。与现有的主要关注模型IPP的调查不同，本调查不仅包括对模型级智能的保护，还包括对有价值的数据集智能的保护。首先，根据有效IPP设计的要求，系统地总结了通用的和特定于方案的性能评价指标。其次，从主动式知识产权侵权预防和被动式知识产权权属验证两个角度，对现有的基于数据集和模型智能的IPP方法进行了全面的调研和分析。此外，从培训设置的角度，深入探讨了与集中式设置相比，分布式设置对IPP构成的独特挑战。此外，本工作还研究了深度IPP技术所面临的各种攻击。最后，我们概述了前景光明的未来方向，这些方向可以作为创新研究的指南。



## **34. Unfair Alignment: Examining Safety Alignment Across Vision Encoder Layers in Vision-Language Models**

不公平对齐：检查视觉语言模型中视觉编码器层之间的安全对齐 cs.CL

Preprint, Under Review

**SubmitDate**: 2024-11-06    [abs](http://arxiv.org/abs/2411.04291v1) [paper-pdf](http://arxiv.org/pdf/2411.04291v1)

**Authors**: Saketh Bachu, Erfan Shayegani, Trishna Chakraborty, Rohit Lal, Arindam Dutta, Chengyu Song, Yue Dong, Nael Abu-Ghazaleh, Amit K. Roy-Chowdhury

**Abstract**: Vision-language models (VLMs) have improved significantly in multi-modal tasks, but their more complex architecture makes their safety alignment more challenging than the alignment of large language models (LLMs). In this paper, we reveal an unfair distribution of safety across the layers of VLM's vision encoder, with earlier and middle layers being disproportionately vulnerable to malicious inputs compared to the more robust final layers. This 'cross-layer' vulnerability stems from the model's inability to generalize its safety training from the default architectural settings used during training to unseen or out-of-distribution scenarios, leaving certain layers exposed. We conduct a comprehensive analysis by projecting activations from various intermediate layers and demonstrate that these layers are more likely to generate harmful outputs when exposed to malicious inputs. Our experiments with LLaVA-1.5 and Llama 3.2 show discrepancies in attack success rates and toxicity scores across layers, indicating that current safety alignment strategies focused on a single default layer are insufficient.

摘要: 视觉语言模型(VLM)在多模式任务中有了显著的改进，但其更复杂的体系结构使得其安全对齐比大型语言模型(LLM)的对齐更具挑战性。在这篇文章中，我们揭示了VLM视觉编码器各层之间的不公平安全分布，与更健壮的最后层相比，较早层和中间层更容易受到恶意输入的影响。这种“跨层”漏洞源于模型无法将其安全培训从培训期间使用的默认架构设置推广到不可见或分布外的情况，从而使某些层暴露在外。我们通过预测来自不同中间层的激活来进行全面分析，并证明这些层在暴露于恶意输入时更有可能产生有害输出。我们使用LLaVA-1.5和Llama 3.2进行的实验表明，各层之间的攻击成功率和毒性分数存在差异，表明当前专注于单一默认层的安全对齐策略是不够的。



## **35. Diversity Helps Jailbreak Large Language Models**

多样性帮助越狱大型语言模型 cs.CL

arXiv admin note: text overlap with arXiv:2312.02119

**SubmitDate**: 2024-11-06    [abs](http://arxiv.org/abs/2411.04223v1) [paper-pdf](http://arxiv.org/pdf/2411.04223v1)

**Authors**: Weiliang Zhao, Daniel Ben-Levi, Junfeng Yang, Chengzhi Mao

**Abstract**: We have uncovered a powerful jailbreak technique that leverages large language models' ability to diverge from prior context, enabling them to bypass safety constraints and generate harmful outputs. By simply instructing the LLM to deviate and obfuscate previous attacks, our method dramatically outperforms existing approaches, achieving up to a 62% higher success rate in compromising nine leading chatbots, including GPT-4, Gemini, and Llama, while using only 13% of the queries. This revelation exposes a critical flaw in current LLM safety training, suggesting that existing methods may merely mask vulnerabilities rather than eliminate them. Our findings sound an urgent alarm for the need to revolutionize testing methodologies to ensure robust and reliable LLM security.

摘要: 我们发现了一种强大的越狱技术，该技术利用大型语言模型脱离先前上下文的能力，使它们能够绕过安全约束并生成有害输出。通过简单地指示LLM偏离和混淆之前的攻击，我们的方法大大优于现有方法，在攻击包括GPT-4、Gemini和Llama在内的九个领先聊天机器人时，成功率提高了62%，而仅使用13%的查询。这一揭露暴露了当前LLM安全培训中的一个关键缺陷，表明现有方法可能只是掩盖了漏洞而不是消除漏洞。我们的发现敲响了紧急警报，需要彻底改变测试方法，以确保强大和可靠的LLM安全。



## **36. Mitigating Privacy Risks in LLM Embeddings from Embedding Inversion**

通过嵌入倒置缓解LLM嵌入中的隐私风险 cs.CR

**SubmitDate**: 2024-11-06    [abs](http://arxiv.org/abs/2411.05034v1) [paper-pdf](http://arxiv.org/pdf/2411.05034v1)

**Authors**: Tiantian Liu, Hongwei Yao, Tong Wu, Zhan Qin, Feng Lin, Kui Ren, Chun Chen

**Abstract**: Embeddings have become a cornerstone in the functionality of large language models (LLMs) due to their ability to transform text data into rich, dense numerical representations that capture semantic and syntactic properties. These embedding vector databases serve as the long-term memory of LLMs, enabling efficient handling of a wide range of natural language processing tasks. However, the surge in popularity of embedding vector databases in LLMs has been accompanied by significant concerns about privacy leakage. Embedding vector databases are particularly vulnerable to embedding inversion attacks, where adversaries can exploit the embeddings to reverse-engineer and extract sensitive information from the original text data. Existing defense mechanisms have shown limitations, often struggling to balance security with the performance of downstream tasks. To address these challenges, we introduce Eguard, a novel defense mechanism designed to mitigate embedding inversion attacks. Eguard employs a transformer-based projection network and text mutual information optimization to safeguard embeddings while preserving the utility of LLMs. Our approach significantly reduces privacy risks, protecting over 95% of tokens from inversion while maintaining high performance across downstream tasks consistent with original embeddings.

摘要: 嵌入已经成为大型语言模型(LLM)功能的基石，因为它们能够将文本数据转换为丰富、密集的数值表示，从而捕获语义和语法属性。这些嵌入的矢量数据库充当LLMS的长期记忆，使其能够有效地处理广泛的自然语言处理任务。然而，在LLMS中嵌入矢量数据库的流行激增一直伴随着对隐私泄露的重大担忧。嵌入矢量数据库特别容易受到嵌入反转攻击，攻击者可以利用嵌入对原始文本数据进行反向工程并提取敏感信息。现有的防御机制已经显示出局限性，往往难以平衡安全和下游任务的性能。为了应对这些挑战，我们引入了EGuard，这是一种新的防御机制，旨在缓解嵌入反转攻击。EGuard采用基于变压器的投影网络和文本互信息优化来保护嵌入，同时保持LLMS的实用性。我们的方法显著降低了隐私风险，保护了95%以上的令牌不被倒置，同时保持了与原始嵌入一致的下游任务的高性能。



## **37. MRJ-Agent: An Effective Jailbreak Agent for Multi-Round Dialogue**

MRJ-Agent：多轮对话的有效越狱代理 cs.AI

**SubmitDate**: 2024-11-06    [abs](http://arxiv.org/abs/2411.03814v1) [paper-pdf](http://arxiv.org/pdf/2411.03814v1)

**Authors**: Fengxiang Wang, Ranjie Duan, Peng Xiao, Xiaojun Jia, YueFeng Chen, Chongwen Wang, Jialing Tao, Hang Su, Jun Zhu, Hui Xue

**Abstract**: Large Language Models (LLMs) demonstrate outstanding performance in their reservoir of knowledge and understanding capabilities, but they have also been shown to be prone to illegal or unethical reactions when subjected to jailbreak attacks. To ensure their responsible deployment in critical applications, it is crucial to understand the safety capabilities and vulnerabilities of LLMs. Previous works mainly focus on jailbreak in single-round dialogue, overlooking the potential jailbreak risks in multi-round dialogues, which are a vital way humans interact with and extract information from LLMs. Some studies have increasingly concentrated on the risks associated with jailbreak in multi-round dialogues. These efforts typically involve the use of manually crafted templates or prompt engineering techniques. However, due to the inherent complexity of multi-round dialogues, their jailbreak performance is limited. To solve this problem, we propose a novel multi-round dialogue jailbreaking agent, emphasizing the importance of stealthiness in identifying and mitigating potential threats to human values posed by LLMs. We propose a risk decomposition strategy that distributes risks across multiple rounds of queries and utilizes psychological strategies to enhance attack strength. Extensive experiments show that our proposed method surpasses other attack methods and achieves state-of-the-art attack success rate. We will make the corresponding code and dataset available for future research. The code will be released soon.

摘要: 大型语言模型(LLM)在其知识和理解能力方面表现出色，但也被证明在受到越狱攻击时容易出现非法或不道德的反应。为了确保它们在关键应用中负责任地部署，了解LLMS的安全能力和漏洞至关重要。以往的研究主要集中在单轮对话中的越狱，而忽略了多轮对话中潜在的越狱风险，而多轮对话是人类与小武器系统交互和提取信息的重要方式。一些研究越来越集中于多轮对话中越狱的相关风险。这些工作通常涉及使用手工制作的模板或即时工程技术。然而，由于多轮对话的内在复杂性，它们的越狱表现有限。为了解决这一问题，我们提出了一种新的多轮对话越狱代理，强调了隐蔽性在识别和缓解LLMS对人类价值构成的潜在威胁方面的重要性。我们提出了一种风险分解策略，将风险分布在多轮查询中，并利用心理策略来增强攻击强度。大量实验表明，该方法优于其他攻击方法，达到了最高的攻击成功率。我们将提供相应的代码和数据集，供未来研究使用。代码很快就会发布。



## **38. Whispers in the Machine: Confidentiality in LLM-integrated Systems**

机器中的耳语：LLM集成系统的保密性 cs.CR

**SubmitDate**: 2024-11-06    [abs](http://arxiv.org/abs/2402.06922v3) [paper-pdf](http://arxiv.org/pdf/2402.06922v3)

**Authors**: Jonathan Evertz, Merlin Chlosta, Lea Schönherr, Thorsten Eisenhofer

**Abstract**: Large Language Models (LLMs) are increasingly augmented with external tools and commercial services into LLM-integrated systems. While these interfaces can significantly enhance the capabilities of the models, they also introduce a new attack surface. Manipulated integrations, for example, can exploit the model and compromise sensitive data accessed through other interfaces. While previous work primarily focused on attacks targeting a model's alignment or the leakage of training data, the security of data that is only available during inference has escaped scrutiny so far. In this work, we demonstrate the vulnerabilities associated with external components and introduce a systematic approach to evaluate confidentiality risks in LLM-integrated systems. We identify two specific attack scenarios unique to these systems and formalize these into a tool-robustness framework designed to measure a model's ability to protect sensitive information. Our findings show that all examined models are highly vulnerable to confidentiality attacks, with the risk increasing significantly when models are used together with external tools.

摘要: 大型语言模型(LLM)越来越多地被外部工具和商业服务扩展到LLM集成系统中。虽然这些接口可以显著增强模型的功能，但它们也引入了新的攻击面。例如，受操纵的集成可能会利用该模型并危及通过其他接口访问的敏感数据。虽然以前的工作主要集中在针对模型对齐或训练数据泄漏的攻击上，但到目前为止，只有在推理过程中才能获得的数据的安全性没有受到审查。在这项工作中，我们展示了与外部组件相关的漏洞，并引入了一种系统的方法来评估LLM集成系统中的机密性风险。我们确定了这些系统特有的两种特定攻击场景，并将它们正式化为工具健壮性框架，旨在衡量模型保护敏感信息的能力。我们的发现表明，所有被检查的模型都非常容易受到保密攻击，当模型与外部工具一起使用时，风险会显著增加。



## **39. The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems**

早起的鸟抓住漏洞：揭开LLM服务系统中的计时侧通道 cs.CR

This work was submitted for review on Sept. 5, 2024, and the initial  version was uploaded to Arxiv on Sept. 30, 2024

**SubmitDate**: 2024-11-06    [abs](http://arxiv.org/abs/2409.20002v2) [paper-pdf](http://arxiv.org/pdf/2409.20002v2)

**Authors**: Linke Song, Zixuan Pang, Wenhao Wang, Zihao Wang, XiaoFeng Wang, Hongbo Chen, Wei Song, Yier Jin, Dan Meng, Rui Hou

**Abstract**: The wide deployment of Large Language Models (LLMs) has given rise to strong demands for optimizing their inference performance. Today's techniques serving this purpose primarily focus on reducing latency and improving throughput through algorithmic and hardware enhancements, while largely overlooking their privacy side effects, particularly in a multi-user environment. In our research, for the first time, we discovered a set of new timing side channels in LLM systems, arising from shared caches and GPU memory allocations, which can be exploited to infer both confidential system prompts and those issued by other users. These vulnerabilities echo security challenges observed in traditional computing systems, highlighting an urgent need to address potential information leakage in LLM serving infrastructures. In this paper, we report novel attack strategies designed to exploit such timing side channels inherent in LLM deployments, specifically targeting the Key-Value (KV) cache and semantic cache widely used to enhance LLM inference performance. Our approach leverages timing measurements and classification models to detect cache hits, allowing an adversary to infer private prompts with high accuracy. We also propose a token-by-token search algorithm to efficiently recover shared prompt prefixes in the caches, showing the feasibility of stealing system prompts and those produced by peer users. Our experimental studies on black-box testing of popular online LLM services demonstrate that such privacy risks are completely realistic, with significant consequences. Our findings underscore the need for robust mitigation to protect LLM systems against such emerging threats.

摘要: 大型语言模型的广泛应用对其推理性能的优化提出了强烈的需求。目前用于此目的的技术主要集中在通过算法和硬件增强来减少延迟和提高吞吐量，而在很大程度上忽略了它们的隐私副作用，特别是在多用户环境中。在我们的研究中，我们首次在LLM系统中发现了一组新的计时侧通道，这些通道来自共享缓存和GPU内存分配，可以用来推断机密系统提示和其他用户发出的提示。这些漏洞呼应了在传统计算系统中观察到的安全挑战，突显出迫切需要解决LLM服务基础设施中潜在的信息泄漏问题。在本文中，我们报告了一种新的攻击策略，旨在利用LLM部署中固有的计时侧通道，特别是针对广泛用于提高LLM推理性能的键值(KV)缓存和语义缓存。我们的方法利用计时测量和分类模型来检测缓存命中，允许对手以高精度推断私人提示。我们还提出了一种逐令牌搜索算法来高效地恢复缓存中的共享提示前缀，证明了窃取系统提示和对等用户产生的提示的可行性。我们对流行的在线LLM服务进行黑盒测试的实验研究表明，这种隐私风险是完全现实的，具有显著的后果。我们的发现强调了强有力的缓解措施的必要性，以保护LLM系统免受此类新出现的威胁。



## **40. Bag of Tricks: Benchmarking of Jailbreak Attacks on LLMs**

诡计袋：对LLM越狱攻击的基准 cs.CR

Accepted by NeurIPS 2024

**SubmitDate**: 2024-11-06    [abs](http://arxiv.org/abs/2406.09324v3) [paper-pdf](http://arxiv.org/pdf/2406.09324v3)

**Authors**: Zhao Xu, Fan Liu, Hao Liu

**Abstract**: Although Large Language Models (LLMs) have demonstrated significant capabilities in executing complex tasks in a zero-shot manner, they are susceptible to jailbreak attacks and can be manipulated to produce harmful outputs. Recently, a growing body of research has categorized jailbreak attacks into token-level and prompt-level attacks. However, previous work primarily overlooks the diverse key factors of jailbreak attacks, with most studies concentrating on LLM vulnerabilities and lacking exploration of defense-enhanced LLMs. To address these issues, we introduced $\textbf{JailTrickBench}$ to evaluate the impact of various attack settings on LLM performance and provide a baseline for jailbreak attacks, encouraging the adoption of a standardized evaluation framework. Specifically, we evaluate the eight key factors of implementing jailbreak attacks on LLMs from both target-level and attack-level perspectives. We further conduct seven representative jailbreak attacks on six defense methods across two widely used datasets, encompassing approximately 354 experiments with about 55,000 GPU hours on A800-80G. Our experimental results highlight the need for standardized benchmarking to evaluate these attacks on defense-enhanced LLMs. Our code is available at https://github.com/usail-hkust/JailTrickBench.

摘要: 尽管大型语言模型(LLM)在以零射击方式执行复杂任务方面表现出了巨大的能力，但它们很容易受到越狱攻击，并可能被操纵以产生有害的输出。最近，越来越多的研究将越狱攻击分为令牌级攻击和提示级攻击。然而，以前的工作主要忽略了越狱攻击的各种关键因素，大多数研究集中在LLM漏洞上，而缺乏对增强防御的LLM的探索。为了解决这些问题，我们引入了$\extbf{JailTrickB边}$来评估各种攻击设置对LLM性能的影响，并提供越狱攻击的基准，鼓励采用标准化评估框架。具体地，我们从目标级和攻击级两个角度评估了对LLMS实施越狱攻击的八个关键因素。我们进一步在两个广泛使用的数据集上对六种防御方法进行了七次有代表性的越狱攻击，在A800-80G上进行了大约354次实验，大约55,000个GPU小时。我们的实验结果强调了标准化基准测试的必要性，以评估这些针对防御增强型LLM的攻击。我们的代码可以在https://github.com/usail-hkust/JailTrickBench.上找到



## **41. Jailbreaking Large Language Models with Symbolic Mathematics**

用符号数学破解大型语言模型 cs.CR

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2409.11445v2) [paper-pdf](http://arxiv.org/pdf/2409.11445v2)

**Authors**: Emet Bethany, Mazal Bethany, Juan Arturo Nolazco Flores, Sumit Kumar Jha, Peyman Najafirad

**Abstract**: Recent advancements in AI safety have led to increased efforts in training and red-teaming large language models (LLMs) to mitigate unsafe content generation. However, these safety mechanisms may not be comprehensive, leaving potential vulnerabilities unexplored. This paper introduces MathPrompt, a novel jailbreaking technique that exploits LLMs' advanced capabilities in symbolic mathematics to bypass their safety mechanisms. By encoding harmful natural language prompts into mathematical problems, we demonstrate a critical vulnerability in current AI safety measures. Our experiments across 13 state-of-the-art LLMs reveal an average attack success rate of 73.6\%, highlighting the inability of existing safety training mechanisms to generalize to mathematically encoded inputs. Analysis of embedding vectors shows a substantial semantic shift between original and encoded prompts, helping explain the attack's success. This work emphasizes the importance of a holistic approach to AI safety, calling for expanded red-teaming efforts to develop robust safeguards across all potential input types and their associated risks.

摘要: 最近人工智能安全方面的进步导致在培训和红队大型语言模型(LLM)方面加大了努力，以减少不安全的内容生成。然而，这些安全机制可能不是全面的，留下了潜在的漏洞有待探索。本文介绍了MathPrompt，这是一种新的越狱技术，它利用LLMS在符号数学中的高级能力来绕过它们的安全机制。通过将有害的自然语言提示编码为数学问题，我们展示了当前人工智能安全措施中的一个严重漏洞。我们在13个最先进的LLM上进行的实验显示，平均攻击成功率为73.6\%，这突显了现有安全培训机制无法概括为数学编码的输入。对嵌入向量的分析显示，原始提示和编码提示之间存在实质性的语义转换，这有助于解释攻击的成功。这项工作强调了对人工智能安全采取整体方法的重要性，呼吁扩大红队努力，为所有潜在的投入类型及其相关风险制定强有力的保障措施。



## **42. Membership Inference Attacks against Large Vision-Language Models**

针对大型视觉语言模型的成员推断攻击 cs.CV

NeurIPS 2024

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.02902v1) [paper-pdf](http://arxiv.org/pdf/2411.02902v1)

**Authors**: Zhan Li, Yongtao Wu, Yihang Chen, Francesco Tonin, Elias Abad Rocamora, Volkan Cevher

**Abstract**: Large vision-language models (VLLMs) exhibit promising capabilities for processing multi-modal tasks across various application scenarios. However, their emergence also raises significant data security concerns, given the potential inclusion of sensitive information, such as private photos and medical records, in their training datasets. Detecting inappropriately used data in VLLMs remains a critical and unresolved issue, mainly due to the lack of standardized datasets and suitable methodologies. In this study, we introduce the first membership inference attack (MIA) benchmark tailored for various VLLMs to facilitate training data detection. Then, we propose a novel MIA pipeline specifically designed for token-level image detection. Lastly, we present a new metric called MaxR\'enyi-K%, which is based on the confidence of the model output and applies to both text and image data. We believe that our work can deepen the understanding and methodology of MIAs in the context of VLLMs. Our code and datasets are available at https://github.com/LIONS-EPFL/VL-MIA.

摘要: 大型视觉语言模型(VLLM)在处理跨各种应用场景的多模式任务方面显示出良好的能力。然而，它们的出现也引发了重大的数据安全担忧，因为它们的训练数据集中可能包含私人照片和医疗记录等敏感信息。检测超低成本管理系统中不适当使用的数据仍然是一个关键和悬而未决的问题，这主要是因为缺乏标准化的数据集和适当的方法。在这项研究中，我们引入了针对不同的VLLM定制的第一个成员推理攻击(MIA)基准，以便于训练数据的检测。然后，我们提出了一种专门为令牌级图像检测设计的MIA流水线。最后，我们提出了一种新的度量方法MaxR‘enyi-K%，它基于模型输出的置信度，适用于文本和图像数据。我们相信，我们的工作可以加深对VLLMS环境下MIA的理解和方法学。我们的代码和数据集可以在https://github.com/LIONS-EPFL/VL-MIA.上找到



## **43. Stochastic Monkeys at Play: Random Augmentations Cheaply Break LLM Safety Alignment**

随机猴子在起作用：随机增强轻松打破LLM安全一致 cs.LG

Under peer review

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.02785v1) [paper-pdf](http://arxiv.org/pdf/2411.02785v1)

**Authors**: Jason Vega, Junsheng Huang, Gaokai Zhang, Hangoo Kang, Minjia Zhang, Gagandeep Singh

**Abstract**: Safety alignment of Large Language Models (LLMs) has recently become a critical objective of model developers. In response, a growing body of work has been investigating how safety alignment can be bypassed through various jailbreaking methods, such as adversarial attacks. However, these jailbreak methods can be rather costly or involve a non-trivial amount of creativity and effort, introducing the assumption that malicious users are high-resource or sophisticated. In this paper, we study how simple random augmentations to the input prompt affect safety alignment effectiveness in state-of-the-art LLMs, such as Llama 3 and Qwen 2. We perform an in-depth evaluation of 17 different models and investigate the intersection of safety under random augmentations with multiple dimensions: augmentation type, model size, quantization, fine-tuning-based defenses, and decoding strategies (e.g., sampling temperature). We show that low-resource and unsophisticated attackers, i.e. $\textit{stochastic monkeys}$, can significantly improve their chances of bypassing alignment with just 25 random augmentations per prompt.

摘要: 大型语言模型(LLM)的安全一致性最近已成为模型开发人员的一个重要目标。作为回应，越来越多的工作一直在研究如何通过各种越狱方法绕过安全对准，例如对抗性攻击。然而，这些越狱方法可能相当昂贵，或者涉及大量的创造力和努力，从而引入了恶意用户是高资源或老练的假设。在本文中，我们研究了输入提示的简单随机增强如何影响最新的LLMS的安全对齐有效性，例如Llama 3和Qwen 2。我们对17个不同的模型进行了深入的评估，并研究了在多个维度的随机增强下的安全性交集：增强类型、模型大小、量化、基于微调的防御和解码策略(例如采样温度)。我们表明，低资源和简单的攻击者，即$\textit{随机猴子}$，可以显著提高他们绕过对齐的机会，每个提示只需25个随机扩展。



## **44. Extracting Unlearned Information from LLMs with Activation Steering**

通过激活引导从LLM中提取未学习的信息 cs.CL

Accepted at NeurIPS 2024 Workshop Safe Generative AI

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2411.02631v1) [paper-pdf](http://arxiv.org/pdf/2411.02631v1)

**Authors**: Atakan Seyitoğlu, Aleksei Kuvshinov, Leo Schwinn, Stephan Günnemann

**Abstract**: An unintended consequence of the vast pretraining of Large Language Models (LLMs) is the verbatim memorization of fragments of their training data, which may contain sensitive or copyrighted information. In recent years, unlearning has emerged as a solution to effectively remove sensitive knowledge from models after training. Yet, recent work has shown that supposedly deleted information can still be extracted by malicious actors through various attacks. Still, current attacks retrieve sets of possible candidate generations and are unable to pinpoint the output that contains the actual target information. We propose activation steering as a method for exact information retrieval from unlearned LLMs. We introduce a novel approach to generating steering vectors, named Anonymized Activation Steering. Additionally, we develop a simple word frequency method to pinpoint the correct answer among a set of candidates when retrieving unlearned information. Our evaluation across multiple unlearning techniques and datasets demonstrates that activation steering successfully recovers general knowledge (e.g., widely known fictional characters) while revealing limitations in retrieving specific information (e.g., details about non-public individuals). Overall, our results demonstrate that exact information retrieval from unlearned models is possible, highlighting a severe vulnerability of current unlearning techniques.

摘要: 大型语言模型(LLM)的大量预训练的一个意想不到的后果是逐字记忆其训练数据的片段，其中可能包含敏感或受版权保护的信息。近年来，遗忘作为一种有效地去除训练后模型中敏感知识的解决方案而出现。然而，最近的研究表明，恶意攻击者仍然可以通过各种攻击提取本应删除的信息。尽管如此，当前的攻击检索到了可能的候选代集合，并且无法确定包含实际目标信息的输出。我们提出了激活引导作为一种从未学习的LLMS中准确检索信息的方法。我们介绍了一种新的生成引导向量的方法，称为匿名激活引导。此外，我们还开发了一种简单的词频方法，在检索未学习信息时，可以在一组候选对象中准确地找到正确答案。我们对多种遗忘技术和数据集的评估表明，激活引导成功地恢复了一般知识(例如，广为人知的虚拟角色)，同时揭示了在检索特定信息(例如，关于非公开个人的细节)方面的局限性。总体而言，我们的结果表明，从未学习的模型中检索准确的信息是可能的，这突显了当前遗忘技术的严重脆弱性。



## **45. Attacking Vision-Language Computer Agents via Pop-ups**

通过弹出窗口攻击视觉语言计算机代理 cs.CL

10 pages, preprint

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2411.02391v1) [paper-pdf](http://arxiv.org/pdf/2411.02391v1)

**Authors**: Yanzhe Zhang, Tao Yu, Diyi Yang

**Abstract**: Autonomous agents powered by large vision and language models (VLM) have demonstrated significant potential in completing daily computer tasks, such as browsing the web to book travel and operating desktop software, which requires agents to understand these interfaces. Despite such visual inputs becoming more integrated into agentic applications, what types of risks and attacks exist around them still remain unclear. In this work, we demonstrate that VLM agents can be easily attacked by a set of carefully designed adversarial pop-ups, which human users would typically recognize and ignore. This distraction leads agents to click these pop-ups instead of performing the tasks as usual. Integrating these pop-ups into existing agent testing environments like OSWorld and VisualWebArena leads to an attack success rate (the frequency of the agent clicking the pop-ups) of 86% on average and decreases the task success rate by 47%. Basic defense techniques such as asking the agent to ignore pop-ups or including an advertisement notice, are ineffective against the attack.

摘要: 由大视觉和语言模型(VLM)驱动的自主代理在完成日常计算机任务方面表现出了巨大的潜力，例如浏览网页预订旅行和操作桌面软件，这需要代理理解这些界面。尽管这样的视觉输入越来越多地集成到代理应用程序中，但围绕它们存在哪些类型的风险和攻击仍不清楚。在这项工作中，我们证明了VLM代理可以很容易地受到一组精心设计的敌意弹出窗口的攻击，人类用户通常会识别并忽略这些弹出窗口。这种干扰会导致工程师单击这些弹出窗口，而不是像往常一样执行任务。将这些弹出窗口集成到OSWorld和VisualWebArena等现有代理测试环境中，攻击成功率(代理单击弹出窗口的频率)平均为86%，任务成功率降低47%。基本的防御技术，如要求代理忽略弹出窗口或包括广告通知，对攻击无效。



## **46. Defining and Evaluating Physical Safety for Large Language Models**

定义和评估大型语言模型的物理安全 cs.LG

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2411.02317v1) [paper-pdf](http://arxiv.org/pdf/2411.02317v1)

**Authors**: Yung-Chen Tang, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Large Language Models (LLMs) are increasingly used to control robotic systems such as drones, but their risks of causing physical threats and harm in real-world applications remain unexplored. Our study addresses the critical gap in evaluating LLM physical safety by developing a comprehensive benchmark for drone control. We classify the physical safety risks of drones into four categories: (1) human-targeted threats, (2) object-targeted threats, (3) infrastructure attacks, and (4) regulatory violations. Our evaluation of mainstream LLMs reveals an undesirable trade-off between utility and safety, with models that excel in code generation often performing poorly in crucial safety aspects. Furthermore, while incorporating advanced prompt engineering techniques such as In-Context Learning and Chain-of-Thought can improve safety, these methods still struggle to identify unintentional attacks. In addition, larger models demonstrate better safety capabilities, particularly in refusing dangerous commands. Our findings and benchmark can facilitate the design and evaluation of physical safety for LLMs. The project page is available at huggingface.co/spaces/TrustSafeAI/LLM-physical-safety.

摘要: 大型语言模型(LLM)越来越多地被用于控制无人机等机器人系统，但它们在现实世界应用中造成物理威胁和伤害的风险仍未被探索。我们的研究通过开发一个全面的无人机控制基准来解决在评估LLM物理安全方面的关键空白。我们将无人机的物理安全风险分为四类：(1)人为目标的威胁，(2)对象为目标的威胁，(3)基础设施攻击，以及(4)违反监管规定。我们对主流LLM的评估揭示了实用性和安全性之间的一种不受欢迎的权衡，在代码生成方面表现出色的模型在关键的安全方面往往表现不佳。此外，虽然结合了先进的即时工程技术，如情景学习和思维链可以提高安全性，但这些方法仍然难以识别无意攻击。此外，较大的型号显示出更好的安全能力，特别是在拒绝危险命令方面。我们的研究结果和基准可以帮助设计和评估LLMS的物理安全性。该项目页面可在huggingface.co/spaces/TrustSafeAI/LLM-physical-safety.上查看



## **47. Manipulation Facing Threats: Evaluating Physical Vulnerabilities in End-to-End Vision Language Action Models**

面对威胁的操纵：评估端到端视觉语言动作模型中的身体脆弱性 cs.CV

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2409.13174v2) [paper-pdf](http://arxiv.org/pdf/2409.13174v2)

**Authors**: Hao Cheng, Erjia Xiao, Chengyuan Yu, Zhao Yao, Jiahang Cao, Qiang Zhang, Jiaxu Wang, Mengshu Sun, Kaidi Xu, Jindong Gu, Renjing Xu

**Abstract**: Recently, driven by advancements in Multimodal Large Language Models (MLLMs), Vision Language Action Models (VLAMs) are being proposed to achieve better performance in open-vocabulary scenarios for robotic manipulation tasks. Since manipulation tasks involve direct interaction with the physical world, ensuring robustness and safety during the execution of this task is always a very critical issue. In this paper, by synthesizing current safety research on MLLMs and the specific application scenarios of the manipulation task in the physical world, we comprehensively evaluate VLAMs in the face of potential physical threats. Specifically, we propose the Physical Vulnerability Evaluating Pipeline (PVEP) that can incorporate as many visual modal physical threats as possible for evaluating the physical robustness of VLAMs. The physical threats in PVEP specifically include Out-of-Distribution, Typography-based Visual Prompts, and Adversarial Patch Attacks. By comparing the performance fluctuations of VLAMs before and after being attacked, we provide generalizable Analyses of how VLAMs respond to different physical security threats. Our project page is in this link: https://chaducheng.github.io/Manipulat-Facing-Threats/.

摘要: 最近，在多模式大语言模型(MLLM)的推动下，视觉语言动作模型(VLAM)被提出以在机器人操作任务的开放词汇场景中实现更好的性能。由于操作任务涉及与物理世界的直接交互，因此确保该任务执行过程中的健壮性和安全性始终是一个非常关键的问题。本文通过综合当前MLLMS的安全研究现状和物理世界中操纵任务的具体应用场景，对VLAMS在面临潜在物理威胁的情况下进行综合评估。具体地说，我们提出了物理脆弱性评估管道(PVEP)，它可以结合尽可能多的视觉通道物理威胁来评估VLAMS的物理健壮性。PVEP中的物理威胁具体包括分发外、基于排版的视觉提示和对抗性补丁攻击。通过比较VLAM在受到攻击前后的性能波动，我们对VLAM如何应对不同的物理安全威胁提供了一般性的分析。我们的项目页面位于以下链接：https://chaducheng.github.io/Manipulat-Facing-Threats/.



## **48. Exploiting LLM Quantization**

利用LLM量化 cs.LG

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2405.18137v2) [paper-pdf](http://arxiv.org/pdf/2405.18137v2)

**Authors**: Kazuki Egashira, Mark Vero, Robin Staab, Jingxuan He, Martin Vechev

**Abstract**: Quantization leverages lower-precision weights to reduce the memory usage of large language models (LLMs) and is a key technique for enabling their deployment on commodity hardware. While LLM quantization's impact on utility has been extensively explored, this work for the first time studies its adverse effects from a security perspective. We reveal that widely used quantization methods can be exploited to produce a harmful quantized LLM, even though the full-precision counterpart appears benign, potentially tricking users into deploying the malicious quantized model. We demonstrate this threat using a three-staged attack framework: (i) first, we obtain a malicious LLM through fine-tuning on an adversarial task; (ii) next, we quantize the malicious model and calculate constraints that characterize all full-precision models that map to the same quantized model; (iii) finally, using projected gradient descent, we tune out the poisoned behavior from the full-precision model while ensuring that its weights satisfy the constraints computed in step (ii). This procedure results in an LLM that exhibits benign behavior in full precision but when quantized, it follows the adversarial behavior injected in step (i). We experimentally demonstrate the feasibility and severity of such an attack across three diverse scenarios: vulnerable code generation, content injection, and over-refusal attack. In practice, the adversary could host the resulting full-precision model on an LLM community hub such as Hugging Face, exposing millions of users to the threat of deploying its malicious quantized version on their devices.

摘要: 量化利用较低精度的权重来减少大型语言模型(LLM)的内存使用，这是在商用硬件上部署LLM的关键技术。虽然LLM量化对效用的影响已经被广泛研究，但这项工作首次从安全的角度研究了它的不利影响。我们发现，广泛使用的量化方法可以被利用来产生有害的量化LLM，即使全精度对应的看起来是良性的，潜在地诱骗用户部署恶意量化模型。我们使用一个三阶段攻击框架演示了这一威胁：(I)首先，我们通过对敌方任务的微调来获得恶意LLM；(Ii)接下来，我们量化恶意模型，并计算映射到相同量化模型的所有全精度模型的约束；(Iii)最后，使用投影梯度下降，我们在确保其权重满足步骤(Ii)中计算的约束的同时，从全精度模型中排除有毒行为。这一过程导致LLM完全精确地表现出良性行为，但当量化时，它遵循在步骤(I)中注入的对抗性行为。我们通过实验演示了这种攻击在三种不同场景中的可行性和严重性：易受攻击的代码生成、内容注入和过度拒绝攻击。在实践中，对手可能会在LLM社区中心(如拥抱脸)上托管产生的全精度模型，使数百万用户面临在他们的设备上部署其恶意量化版本的威胁。



## **49. Data Extraction Attacks in Retrieval-Augmented Generation via Backdoors**

通过后门进行检索增强生成中的数据提取攻击 cs.CR

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2411.01705v1) [paper-pdf](http://arxiv.org/pdf/2411.01705v1)

**Authors**: Yuefeng Peng, Junda Wang, Hong Yu, Amir Houmansadr

**Abstract**: Despite significant advancements, large language models (LLMs) still struggle with providing accurate answers when lacking domain-specific or up-to-date knowledge. Retrieval-Augmented Generation (RAG) addresses this limitation by incorporating external knowledge bases, but it also introduces new attack surfaces. In this paper, we investigate data extraction attacks targeting the knowledge databases of RAG systems. We demonstrate that previous attacks on RAG largely depend on the instruction-following capabilities of LLMs, and that simple fine-tuning can reduce the success rate of such attacks to nearly zero. This makes these attacks impractical since fine-tuning is a common practice when deploying LLMs in specific domains. To further reveal the vulnerability, we propose to backdoor RAG, where a small portion of poisoned data is injected during the fine-tuning phase to create a backdoor within the LLM. When this compromised LLM is integrated into a RAG system, attackers can exploit specific triggers in prompts to manipulate the LLM to leak documents from the retrieval database. By carefully designing the poisoned data, we achieve both verbatim and paraphrased document extraction. We show that with only 3\% poisoned data, our method achieves an average success rate of 79.7\% in verbatim extraction on Llama2-7B, with a ROUGE-L score of 64.21, and a 68.6\% average success rate in paraphrased extraction, with an average ROUGE score of 52.6 across four datasets. These results underscore the privacy risks associated with the supply chain when deploying RAG systems.

摘要: 尽管有了很大的进步，但大型语言模型(LLM)在缺乏特定领域或最新知识的情况下，仍然难以提供准确的答案。检索-增强生成(RAG)通过结合外部知识库解决了这一限制，但它也引入了新的攻击面。本文研究了针对RAG系统知识库的数据抽取攻击。我们证明了以前对RAG的攻击在很大程度上依赖于LLMS的指令跟随能力，并且简单的微调可以将此类攻击的成功率降低到几乎为零。这使得这些攻击不切实际，因为在特定域中部署LLM时，微调是一种常见的做法。为了进一步揭示漏洞，我们建议使用后门RAG，在微调阶段注入一小部分有毒数据，以在LLM中创建后门。当这个受损的LLM被集成到RAG系统中时，攻击者可以利用提示中的特定触发器来操纵LLM，从而从检索数据库中泄漏文档。通过精心设计有毒数据，我们实现了逐字和释义文档提取。实验结果表明，在3个中毒数据的情况下，该方法在Llama2-7B上的平均逐字提取成功率为79.7%，Rouge-L评分为64.2 1分，转述提取的平均成功率为68.6%，4个数据集的平均Rouge评分为5 2.6分。这些结果突显了在部署RAG系统时与供应链相关的隐私风险。



## **50. UniGuard: Towards Universal Safety Guardrails for Jailbreak Attacks on Multimodal Large Language Models**

UniGuard：为多模式大型语言模型越狱攻击建立通用安全护栏 cs.CL

14 pages

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2411.01703v1) [paper-pdf](http://arxiv.org/pdf/2411.01703v1)

**Authors**: Sejoon Oh, Yiqiao Jin, Megha Sharma, Donghyun Kim, Eric Ma, Gaurav Verma, Srijan Kumar

**Abstract**: Multimodal large language models (MLLMs) have revolutionized vision-language understanding but are vulnerable to multimodal jailbreak attacks, where adversaries meticulously craft inputs to elicit harmful or inappropriate responses. We propose UniGuard, a novel multimodal safety guardrail that jointly considers the unimodal and cross-modal harmful signals. UniGuard is trained such that the likelihood of generating harmful responses in a toxic corpus is minimized, and can be seamlessly applied to any input prompt during inference with minimal computational costs. Extensive experiments demonstrate the generalizability of UniGuard across multiple modalities and attack strategies. It demonstrates impressive generalizability across multiple state-of-the-art MLLMs, including LLaVA, Gemini Pro, GPT-4, MiniGPT-4, and InstructBLIP, thereby broadening the scope of our solution.

摘要: 多模式大型语言模型（MLLM）彻底改变了视觉语言理解，但很容易受到多模式越狱攻击，对手精心设计输入以引发有害或不当的反应。我们提出了UniGuard，这是一种新型的多模式安全护栏，它联合考虑了单模式和跨模式有害信号。UniGuard经过训练，可以最大限度地降低在有毒的数据库中生成有害响应的可能性，并且可以以最小的计算成本无缝地应用于推理期间的任何输入提示。大量实验证明了UniGuard在多种模式和攻击策略中的通用性。它在多个最先进的MLLM（包括LLaVA、Gemini Pro、GPT-4、MiniGPT-4和DirecectBLIP）上展示了令人印象深刻的通用性，从而扩大了我们解决方案的范围。



