# Latest Large Language Model Attack Papers
**update at 2025-02-24 09:51:45**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Red-Teaming LLM Multi-Agent Systems via Communication Attacks**

通过通信攻击的Red-Teaming LLM多代理系统 cs.CR

**SubmitDate**: 2025-02-20    [abs](http://arxiv.org/abs/2502.14847v1) [paper-pdf](http://arxiv.org/pdf/2502.14847v1)

**Authors**: Pengfei He, Yupin Lin, Shen Dong, Han Xu, Yue Xing, Hui Liu

**Abstract**: Large Language Model-based Multi-Agent Systems (LLM-MAS) have revolutionized complex problem-solving capability by enabling sophisticated agent collaboration through message-based communications. While the communication framework is crucial for agent coordination, it also introduces a critical yet unexplored security vulnerability. In this work, we introduce Agent-in-the-Middle (AiTM), a novel attack that exploits the fundamental communication mechanisms in LLM-MAS by intercepting and manipulating inter-agent messages. Unlike existing attacks that compromise individual agents, AiTM demonstrates how an adversary can compromise entire multi-agent systems by only manipulating the messages passing between agents. To enable the attack under the challenges of limited control and role-restricted communication format, we develop an LLM-powered adversarial agent with a reflection mechanism that generates contextually-aware malicious instructions. Our comprehensive evaluation across various frameworks, communication structures, and real-world applications demonstrates that LLM-MAS is vulnerable to communication-based attacks, highlighting the need for robust security measures in multi-agent systems.

摘要: 基于大型语言模型的多代理系统(LLM-MAS)通过基于消息的通信实现复杂的代理协作，使复杂问题的解决能力发生了革命性的变化。虽然通信框架对于代理协调至关重要，但它也引入了一个严重但尚未探索的安全漏洞。在这项工作中，我们引入了中间代理(AiTM)，这是一种通过拦截和处理代理间消息来利用LLM-MAS的基本通信机制的新型攻击。与现有的危害单个代理的攻击不同，AiTM演示了对手如何通过仅操纵代理之间传递的消息来危害整个多代理系统。为了在有限控制和角色受限通信格式的挑战下实现攻击，我们开发了一个基于LLM的恶意代理，该代理具有反射机制，可以生成上下文感知的恶意指令。我们对各种框架、通信结构和现实世界应用的综合评估表明，LLM-MAS容易受到基于通信的攻击，这突显了在多代理系统中需要强大的安全措施。



## **2. HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States**

HiddenDetect：通过监视隐藏状态检测针对大型视觉语言模型的越狱攻击 cs.CL

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2502.14744v2) [paper-pdf](http://arxiv.org/pdf/2502.14744v2)

**Authors**: Yilei Jiang, Xinyan Gao, Tianshuo Peng, Yingshui Tan, Xiaoyong Zhu, Bo Zheng, Xiangyu Yue

**Abstract**: The integration of additional modalities increases the susceptibility of large vision-language models (LVLMs) to safety risks, such as jailbreak attacks, compared to their language-only counterparts. While existing research primarily focuses on post-hoc alignment techniques, the underlying safety mechanisms within LVLMs remain largely unexplored. In this work , we investigate whether LVLMs inherently encode safety-relevant signals within their internal activations during inference. Our findings reveal that LVLMs exhibit distinct activation patterns when processing unsafe prompts, which can be leveraged to detect and mitigate adversarial inputs without requiring extensive fine-tuning. Building on this insight, we introduce HiddenDetect, a novel tuning-free framework that harnesses internal model activations to enhance safety. Experimental results show that {HiddenDetect} surpasses state-of-the-art methods in detecting jailbreak attacks against LVLMs. By utilizing intrinsic safety-aware patterns, our method provides an efficient and scalable solution for strengthening LVLM robustness against multimodal threats. Our code will be released publicly at https://github.com/leigest519/HiddenDetect.

摘要: 与纯语言模型相比，更多模式的集成增加了大型视觉语言模型(LVLM)对安全风险(如越狱袭击)的敏感性。虽然现有的研究主要集中在后对齐技术上，但左腰椎内潜在的安全机制在很大程度上仍未被探索。在这项工作中，我们调查了在推理过程中，LVLM是否内在地在其内部激活中编码与安全相关的信号。我们的发现表明，在处理不安全提示时，LVLMS显示出不同的激活模式，这可以被用来检测和缓解敌意输入，而不需要进行广泛的微调。基于这一见解，我们引入了HiddenDetect，这是一个新的免调优框架，它利用内部模型激活来增强安全性。实验结果表明，{HiddenDetect}在检测针对LVLMS的越狱攻击方面优于最先进的方法。通过利用固有的安全感知模式，我们的方法提供了一种高效且可扩展的解决方案，以增强LVLM对多模式威胁的健壮性。我们的代码将在https://github.com/leigest519/HiddenDetect.上公开发布



## **3. PEARL: Towards Permutation-Resilient LLMs**

Pearl：迈向具有置换弹性的法学硕士 cs.LG

ICLR 2025

**SubmitDate**: 2025-02-20    [abs](http://arxiv.org/abs/2502.14628v1) [paper-pdf](http://arxiv.org/pdf/2502.14628v1)

**Authors**: Liang Chen, Li Shen, Yang Deng, Xiaoyan Zhao, Bin Liang, Kam-Fai Wong

**Abstract**: The in-context learning (ICL) capability of large language models (LLMs) enables them to perform challenging tasks using provided demonstrations. However, ICL is highly sensitive to the ordering of demonstrations, leading to instability in predictions. This paper shows that this vulnerability can be exploited to design a natural attack - difficult for model providers to detect - that achieves nearly 80% success rate on LLaMA-3 by simply permuting the demonstrations. Existing mitigation methods primarily rely on post-processing and fail to enhance the model's inherent robustness to input permutations, raising concerns about safety and reliability of LLMs. To address this issue, we propose Permutation-resilient learning (PEARL), a novel framework based on distributionally robust optimization (DRO), which optimizes model performance against the worst-case input permutation. Specifically, PEARL consists of a permutation-proposal network (P-Net) and the LLM. The P-Net generates the most challenging permutations by treating it as an optimal transport problem, which is solved using an entropy-constrained Sinkhorn algorithm. Through minimax optimization, the P-Net and the LLM iteratively optimize against each other, progressively improving the LLM's robustness. Experiments on synthetic pre-training and real-world instruction tuning tasks demonstrate that PEARL effectively mitigates permutation attacks and enhances performance. Notably, despite being trained on fewer shots and shorter contexts, PEARL achieves performance gains of up to 40% when scaled to many-shot and long-context scenarios, highlighting its efficiency and generalization capabilities.

摘要: 大型语言模型(LLM)的情景学习(ICL)能力使他们能够使用提供的演示执行具有挑战性的任务。然而，ICL对示威的顺序高度敏感，导致预测不稳定。这篇论文表明，可以利用这个漏洞来设计一种自然攻击-模型提供商很难检测到-通过简单地排列演示就可以在Llama-3上实现近80%的成功率。现有的缓解方法主要依赖于后处理，不能增强模型对输入置换的内在稳健性，这引发了人们对LLMS的安全性和可靠性的担忧。为了解决这个问题，我们提出了置换弹性学习(PEAR)，这是一个基于分布稳健优化(DRO)的新框架，它针对最坏情况下的输入置换优化模型性能。具体地说，PEARL由一个置换建议网络(P-Net)和LLM组成。P-网通过将其视为最优传输问题来生成最具挑战性的排列，该最优传输问题使用熵约束的Sinkhorn算法来求解。通过极小极大优化，P网和LLM相互迭代优化，逐步提高LLM的健壮性。在人工预训练和真实指令调优任务上的实验表明，PEARL算法有效地缓解了置换攻击，提高了性能。值得注意的是，尽管在更少的镜头和更短的背景下进行了培训，但当扩展到多镜头和长背景场景时，PEAR实现了高达40%的性能提升，突出了其效率和泛化能力。



## **4. BaxBench: Can LLMs Generate Correct and Secure Backends?**

收件箱长凳：LLM能否生成正确且安全的后台？ cs.CR

**SubmitDate**: 2025-02-20    [abs](http://arxiv.org/abs/2502.11844v2) [paper-pdf](http://arxiv.org/pdf/2502.11844v2)

**Authors**: Mark Vero, Niels Mündler, Victor Chibotaru, Veselin Raychev, Maximilian Baader, Nikola Jovanović, Jingxuan He, Martin Vechev

**Abstract**: The automatic generation of programs has long been a fundamental challenge in computer science. Recent benchmarks have shown that large language models (LLMs) can effectively generate code at the function level, make code edits, and solve algorithmic coding tasks. However, to achieve full automation, LLMs should be able to generate production-quality, self-contained application modules. To evaluate the capabilities of LLMs in solving this challenge, we introduce BaxBench, a novel evaluation benchmark consisting of 392 tasks for the generation of backend applications. We focus on backends for three critical reasons: (i) they are practically relevant, building the core components of most modern web and cloud software, (ii) they are difficult to get right, requiring multiple functions and files to achieve the desired functionality, and (iii) they are security-critical, as they are exposed to untrusted third-parties, making secure solutions that prevent deployment-time attacks an imperative. BaxBench validates the functionality of the generated applications with comprehensive test cases, and assesses their security exposure by executing end-to-end exploits. Our experiments reveal key limitations of current LLMs in both functionality and security: (i) even the best model, OpenAI o1, achieves a mere 60% on code correctness; (ii) on average, we could successfully execute security exploits on more than half of the correct programs generated by each LLM; and (iii) in less popular backend frameworks, models further struggle to generate correct and secure applications. Progress on BaxBench signifies important steps towards autonomous and secure software development with LLMs.

摘要: 长期以来，程序的自动生成一直是计算机科学中的一个基本挑战。最近的基准测试表明，大型语言模型(LLM)可以有效地在函数级生成代码、进行代码编辑和解决算法编码任务。然而，为了实现完全自动化，LLMS应该能够生成生产质量的、独立的应用程序模块。为了评估LLMS在解决这一挑战方面的能力，我们引入了BaxBch，这是一个新的评估基准，由392个任务组成，用于生成后端应用程序。我们关注后端有三个关键原因：(I)它们实际上是相关的，构建了大多数现代Web和云软件的核心组件；(Ii)它们很难正确使用，需要多种功能和文件才能实现所需的功能；(Iii)它们是安全关键型的，因为它们可能会暴露在不受信任的第三方面前，这使得防止部署时攻击的安全解决方案成为当务之急。BaxBtch使用全面的测试用例验证生成的应用程序的功能，并通过执行端到端漏洞攻击来评估它们的安全暴露。我们的实验揭示了当前LLM在功能和安全性方面的关键局限性：(I)即使是最好的模型OpenAI o1，代码正确性也只有60%；(Ii)平均而言，我们可以在每个LLM生成的正确程序中成功地执行安全漏洞；以及(Iii)在不太流行的后端框架中，模型进一步难以生成正确和安全的应用程序。BaxBtch的进展标志着使用LLMS进行自主和安全的软件开发迈出了重要的一步。



## **5. CORBA: Contagious Recursive Blocking Attacks on Multi-Agent Systems Based on Large Language Models**

耳罩：基于大型语言模型的多Agent系统的传染性循环阻塞攻击 cs.CL

**SubmitDate**: 2025-02-20    [abs](http://arxiv.org/abs/2502.14529v1) [paper-pdf](http://arxiv.org/pdf/2502.14529v1)

**Authors**: Zhenhong Zhou, Zherui Li, Jie Zhang, Yuanhe Zhang, Kun Wang, Yang Liu, Qing Guo

**Abstract**: Large Language Model-based Multi-Agent Systems (LLM-MASs) have demonstrated remarkable real-world capabilities, effectively collaborating to complete complex tasks. While these systems are designed with safety mechanisms, such as rejecting harmful instructions through alignment, their security remains largely unexplored. This gap leaves LLM-MASs vulnerable to targeted disruptions. In this paper, we introduce Contagious Recursive Blocking Attacks (Corba), a novel and simple yet highly effective attack that disrupts interactions between agents within an LLM-MAS. Corba leverages two key properties: its contagious nature allows it to propagate across arbitrary network topologies, while its recursive property enables sustained depletion of computational resources. Notably, these blocking attacks often involve seemingly benign instructions, making them particularly challenging to mitigate using conventional alignment methods. We evaluate Corba on two widely-used LLM-MASs, namely, AutoGen and Camel across various topologies and commercial models. Additionally, we conduct more extensive experiments in open-ended interactive LLM-MASs, demonstrating the effectiveness of Corba in complex topology structures and open-source models. Our code is available at: https://github.com/zhrli324/Corba.

摘要: 基于大型语言模型的多智能体系统(LLM-MASS)已经显示出卓越的现实能力，能够有效地协作完成复杂的任务。虽然这些系统设计有安全机制，如通过调整拒绝有害指令，但它们的安全性在很大程度上仍未得到探索。这一差距使LLM-MASS容易受到有针对性的中断的影响。在本文中，我们介绍了传染性递归阻断攻击(CORBA)，这是一种新颖、简单但高效的攻击，它破坏了LLM-MAS内代理之间的交互。CORBA利用了两个关键属性：它的传染性允许它在任意网络拓扑中传播，而它的递归属性允许计算资源的持续耗尽。值得注意的是，这些阻止攻击通常涉及看似良性的指令，这使得使用传统的对齐方法来缓解它们特别具有挑战性。我们在两个广泛使用的LLM-MASS，即Autogen和Camel上对CORBA进行了评估，涉及各种拓扑和商业模型。此外，我们在开放的交互式LLM-MASS上进行了更广泛的实验，展示了CORBA在复杂拓扑结构和开源模型中的有效性。我们的代码请访问：https://github.com/zhrli324/Corba.



## **6. How Jailbreak Defenses Work and Ensemble? A Mechanistic Investigation**

越狱防御如何发挥作用和吸引力？机械调查 cs.CR

**SubmitDate**: 2025-02-20    [abs](http://arxiv.org/abs/2502.14486v1) [paper-pdf](http://arxiv.org/pdf/2502.14486v1)

**Authors**: Zhuohang Long, Siyuan Wang, Shujun Liu, Yuhang Lai, Xuanjing Huang, Zhongyu Wei

**Abstract**: Jailbreak attacks, where harmful prompts bypass generative models' built-in safety, raise serious concerns about model vulnerability. While many defense methods have been proposed, the trade-offs between safety and helpfulness, and their application to Large Vision-Language Models (LVLMs), are not well understood. This paper systematically examines jailbreak defenses by reframing the standard generation task as a binary classification problem to assess model refusal tendencies for both harmful and benign queries. We identify two key defense mechanisms: safety shift, which increases refusal rates across all queries, and harmfulness discrimination, which improves the model's ability to distinguish between harmful and benign inputs. Using these mechanisms, we develop two ensemble defense strategies-inter-mechanism ensembles and intra-mechanism ensembles-to balance safety and helpfulness. Experiments on the MM-SafetyBench and MOSSBench datasets with LLaVA-1.5 models show that these strategies effectively improve model safety or optimize the trade-off between safety and helpfulness.

摘要: 越狱攻击，其中有害的提示绕过了生成式模型的内置安全，引发了对模型脆弱性的严重担忧。虽然已经提出了许多防御方法，但安全和有用之间的权衡，以及它们在大型视觉语言模型(LVLM)中的应用，还没有被很好地理解。本文通过将标准生成任务重组为一个二进制分类问题来系统地检查越狱防御，以评估有害和良性查询的模型拒绝倾向。我们确定了两个关键的防御机制：安全转移，它增加了所有查询的拒绝率，以及有害歧视，它提高了模型区分有害和良性输入的能力。利用这些机制，我们开发了两种综合防御策略--机制间集成和机制内集成--以平衡安全性和帮助。在使用LLaVA-1.5模型的MM-SafetyB边和MOSSB边数据集上的实验表明，这些策略有效地提高了模型的安全性，或优化了安全性和有用性之间的权衡。



## **7. Making Them a Malicious Database: Exploiting Query Code to Jailbreak Aligned Large Language Models**

将其打造成恶意数据库：利用查询代码越狱对齐的大型语言模型 cs.CR

15 pages, 11 figures

**SubmitDate**: 2025-02-20    [abs](http://arxiv.org/abs/2502.09723v2) [paper-pdf](http://arxiv.org/pdf/2502.09723v2)

**Authors**: Qingsong Zou, Jingyu Xiao, Qing Li, Zhi Yan, Yuhang Wang, Li Xu, Wenxuan Wang, Kuofeng Gao, Ruoyu Li, Yong Jiang

**Abstract**: Recent advances in large language models (LLMs) have demonstrated remarkable potential in the field of natural language processing. Unfortunately, LLMs face significant security and ethical risks. Although techniques such as safety alignment are developed for defense, prior researches reveal the possibility of bypassing such defenses through well-designed jailbreak attacks. In this paper, we propose QueryAttack, a novel framework to examine the generalizability of safety alignment. By treating LLMs as knowledge databases, we translate malicious queries in natural language into structured non-natural query language to bypass the safety alignment mechanisms of LLMs. We conduct extensive experiments on mainstream LLMs, and the results show that QueryAttack not only can achieve high attack success rates (ASRs), but also can jailbreak various defense methods. Furthermore, we tailor a defense method against QueryAttack, which can reduce ASR by up to 64% on GPT-4-1106. Our code is available at https://github.com/horizonsinzqs/QueryAttack.

摘要: 大语言模型的最新进展在自然语言处理领域显示出巨大的潜力。不幸的是，低收入国家面临着重大的安全和道德风险。尽管安全对准等技术是为了防御而开发的，但之前的研究表明，通过精心设计的越狱攻击来绕过此类防御是可能的。在本文中，我们提出了一种新的框架QueryAttack来检验安全对齐的泛化能力。通过将LLMS视为知识库，将自然语言中的恶意查询转换为结构化的非自然查询语言，绕过LLMS的安全对齐机制。我们在主流的LLMS上进行了大量的实验，结果表明，QueryAttack不仅可以达到很高的攻击成功率(ASR)，而且可以破解各种防御方法。此外，我们定制了一种针对QueryAttack的防御方法，在GPT-4-1106上可以将ASR降低高达%。我们的代码可以在https://github.com/horizonsinzqs/QueryAttack.上找到



## **8. Vulnerability of Text-to-Image Models to Prompt Template Stealing: A Differential Evolution Approach**

文本到图像模型提示模板窃取的脆弱性：差异进化方法 cs.CL

14 pages,8 figures,4 tables

**SubmitDate**: 2025-02-20    [abs](http://arxiv.org/abs/2502.14285v1) [paper-pdf](http://arxiv.org/pdf/2502.14285v1)

**Authors**: Yurong Wu, Fangwen Mu, Qiuhong Zhang, Jinjing Zhao, Xinrun Xu, Lingrui Mei, Yang Wu, Lin Shi, Junjie Wang, Zhiming Ding, Yiwei Wang

**Abstract**: Prompt trading has emerged as a significant intellectual property concern in recent years, where vendors entice users by showcasing sample images before selling prompt templates that can generate similar images. This work investigates a critical security vulnerability: attackers can steal prompt templates using only a limited number of sample images. To investigate this threat, we introduce Prism, a prompt-stealing benchmark consisting of 50 templates and 450 images, organized into Easy and Hard difficulty levels. To identify the vulnerabity of VLMs to prompt stealing, we propose EvoStealer, a novel template stealing method that operates without model fine-tuning by leveraging differential evolution algorithms. The system first initializes population sets using multimodal large language models (MLLMs) based on predefined patterns, then iteratively generates enhanced offspring through MLLMs. During evolution, EvoStealer identifies common features across offspring to derive generalized templates. Our comprehensive evaluation conducted across open-source (INTERNVL2-26B) and closed-source models (GPT-4o and GPT-4o-mini) demonstrates that EvoStealer's stolen templates can reproduce images highly similar to originals and effectively generalize to other subjects, significantly outperforming baseline methods with an average improvement of over 10%. Moreover, our cost analysis reveals that EvoStealer achieves template stealing with negligible computational expenses. Our code and dataset are available at https://github.com/whitepagewu/evostealer.

摘要: 近年来，即时交易已成为一个重要的知识产权问题，供应商通过展示样本图像来吸引用户，然后再销售可以生成类似图像的提示模板。这项工作调查了一个严重的安全漏洞：攻击者可以仅使用有限数量的样本图像来窃取提示模板。为了调查这一威胁，我们引入了棱镜，这是一个即时窃取基准，由50个模板和450个图像组成，组织成容易和难的难度级别。为了识别VLM对快速窃取的脆弱性，我们提出了一种新的模板窃取方法EvoStealer，该方法不需要对模型进行微调，而是利用了差分进化算法。该系统首先使用基于预定义模式的多通道大语言模型来初始化种群集合，然后通过多通道大语言模型迭代地生成增强的子代。在进化过程中，EvoStealer识别子代的共同特征，以派生通用模板。我们在开源(INTERNVL2-26B)和封闭源代码模型(GPT-4o和GPT-4o-mini)上进行的综合评估表明，EvoStealer窃取的模板可以复制与原始图像高度相似的图像，并有效地推广到其他对象，显著优于基线方法，平均改进超过10%。此外，我们的代价分析表明，EvoStealer以可以忽略的计算开销实现了模板窃取。我们的代码和数据集可在https://github.com/whitepagewu/evostealer.上获得



## **9. Towards Secure Program Partitioning for Smart Contracts with LLM's In-Context Learning**

利用LLM的上下文学习实现智能合同的安全程序分区 cs.SE

**SubmitDate**: 2025-02-20    [abs](http://arxiv.org/abs/2502.14215v1) [paper-pdf](http://arxiv.org/pdf/2502.14215v1)

**Authors**: Ye Liu, Yuqing Niu, Chengyan Ma, Ruidong Han, Wei Ma, Yi Li, Debin Gao, David Lo

**Abstract**: Smart contracts are highly susceptible to manipulation attacks due to the leakage of sensitive information. Addressing manipulation vulnerabilities is particularly challenging because they stem from inherent data confidentiality issues rather than straightforward implementation bugs. To tackle this by preventing sensitive information leakage, we present PartitionGPT, the first LLM-driven approach that combines static analysis with the in-context learning capabilities of large language models (LLMs) to partition smart contracts into privileged and normal codebases, guided by a few annotated sensitive data variables. We evaluated PartitionGPT on 18 annotated smart contracts containing 99 sensitive functions. The results demonstrate that PartitionGPT successfully generates compilable, and verified partitions for 78% of the sensitive functions while reducing approximately 30% code compared to function-level partitioning approach. Furthermore, we evaluated PartitionGPT on nine real-world manipulation attacks that lead to a total loss of 25 million dollars, PartitionGPT effectively prevents eight cases, highlighting its potential for broad applicability and the necessity for secure program partitioning during smart contract development to diminish manipulation vulnerabilities.

摘要: 由于敏感信息的泄露，智能合约极易受到操纵攻击。解决操纵漏洞尤其具有挑战性，因为它们源于固有的数据机密性问题，而不是直接的实现错误。为了通过防止敏感信息泄露来解决这个问题，我们提出了PartitionGPT，这是第一个LLM驱动的方法，它将静态分析与大型语言模型(LLM)的上下文学习能力相结合，在几个带注释的敏感数据变量的指导下，将智能合同划分为特权代码库和普通代码库。我们在包含99个敏感函数的18个带注释的智能合约上对PartitionGPT进行了评估。结果表明，与函数级划分方法相比，PartitionGPT成功地为78%的敏感函数生成了可编译的、可验证的划分，同时减少了约30%的代码。此外，我们评估了PartitionGPT在9个实际操作攻击上的性能，这些攻击导致了2500万美元的总损失，PartitionGPT有效地防止了8个案例，突出了它的广泛适用性和在智能合同开发过程中安全程序分区的必要性，以减少操纵漏洞。



## **10. Multi-Faceted Studies on Data Poisoning can Advance LLM Development**

对数据中毒的多方面研究可以推进LLM的发展 cs.CR

**SubmitDate**: 2025-02-20    [abs](http://arxiv.org/abs/2502.14182v1) [paper-pdf](http://arxiv.org/pdf/2502.14182v1)

**Authors**: Pengfei He, Yue Xing, Han Xu, Zhen Xiang, Jiliang Tang

**Abstract**: The lifecycle of large language models (LLMs) is far more complex than that of traditional machine learning models, involving multiple training stages, diverse data sources, and varied inference methods. While prior research on data poisoning attacks has primarily focused on the safety vulnerabilities of LLMs, these attacks face significant challenges in practice. Secure data collection, rigorous data cleaning, and the multistage nature of LLM training make it difficult to inject poisoned data or reliably influence LLM behavior as intended. Given these challenges, this position paper proposes rethinking the role of data poisoning and argue that multi-faceted studies on data poisoning can advance LLM development. From a threat perspective, practical strategies for data poisoning attacks can help evaluate and address real safety risks to LLMs. From a trustworthiness perspective, data poisoning can be leveraged to build more robust LLMs by uncovering and mitigating hidden biases, harmful outputs, and hallucinations. Moreover, from a mechanism perspective, data poisoning can provide valuable insights into LLMs, particularly the interplay between data and model behavior, driving a deeper understanding of their underlying mechanisms.

摘要: 大语言模型的生命周期比传统的机器学习模型复杂得多，涉及多个训练阶段、不同的数据源和不同的推理方法。虽然以前对数据中毒攻击的研究主要集中在LLMS的安全漏洞上，但这些攻击在实践中面临着巨大的挑战。安全的数据收集、严格的数据清理和LLM培训的多阶段性质使得注入有毒数据或可靠地影响LLM行为变得困难。鉴于这些挑战，本立场文件建议重新思考数据中毒的作用，并认为对数据中毒的多方面研究可以促进LLM的发展。从威胁的角度来看，针对数据中毒攻击的实用策略可以帮助评估和解决低层管理的实际安全风险。从可信度的角度来看，可以利用数据中毒通过发现和减少隐藏的偏差、有害输出和幻觉来构建更强大的LLM。此外，从机制的角度来看，数据中毒可以提供对LLM的有价值的见解，特别是数据和模型行为之间的相互作用，推动对其潜在机制的更深层次理解。



## **11. Why Safeguarded Ships Run Aground? Aligned Large Language Models' Safety Mechanisms Tend to Be Anchored in The Template Region**

为什么受保障的船只会搁浅？对齐的大型语言模型的安全机制倾向于锚定在模板区域 cs.CL

**SubmitDate**: 2025-02-19    [abs](http://arxiv.org/abs/2502.13946v1) [paper-pdf](http://arxiv.org/pdf/2502.13946v1)

**Authors**: Chak Tou Leong, Qingyu Yin, Jian Wang, Wenjie Li

**Abstract**: The safety alignment of large language models (LLMs) remains vulnerable, as their initial behavior can be easily jailbroken by even relatively simple attacks. Since infilling a fixed template between the input instruction and initial model output is a common practice for existing LLMs, we hypothesize that this template is a key factor behind their vulnerabilities: LLMs' safety-related decision-making overly relies on the aggregated information from the template region, which largely influences these models' safety behavior. We refer to this issue as template-anchored safety alignment. In this paper, we conduct extensive experiments and verify that template-anchored safety alignment is widespread across various aligned LLMs. Our mechanistic analyses demonstrate how it leads to models' susceptibility when encountering inference-time jailbreak attacks. Furthermore, we show that detaching safety mechanisms from the template region is promising in mitigating vulnerabilities to jailbreak attacks. We encourage future research to develop more robust safety alignment techniques that reduce reliance on the template region.

摘要: 大型语言模型(LLM)的安全一致性仍然脆弱，因为它们的初始行为即使是相对简单的攻击也很容易破解。由于在输入指令和初始模型输出之间填充固定模板是现有LLMS的常见做法，因此我们假设该模板是LLMS漏洞背后的关键因素：LLMS的安全相关决策过度依赖于模板区域的聚集信息，这在很大程度上影响了这些模型的安全行为。我们将这个问题称为模板锚定安全对准。在本文中，我们进行了大量的实验，并验证了模板锚定的安全对准在各种对准的LLM中都是普遍存在的。我们的机制分析表明，当遇到推理时间越狱攻击时，它如何导致模型的敏感性。此外，我们还表明，将安全机制与模板区域分离在降低越狱攻击的脆弱性方面是很有前途的。我们鼓励未来的研究开发更稳健的安全比对技术，减少对模板区域的依赖。



## **12. Theoretically Grounded Framework for LLM Watermarking: A Distribution-Adaptive Approach**

LLM水印的理论基础框架：分布自适应方法 cs.CR

**SubmitDate**: 2025-02-19    [abs](http://arxiv.org/abs/2410.02890v4) [paper-pdf](http://arxiv.org/pdf/2410.02890v4)

**Authors**: Haiyun He, Yepeng Liu, Ziqiao Wang, Yongyi Mao, Yuheng Bu

**Abstract**: Watermarking has emerged as a crucial method to distinguish AI-generated text from human-created text. In this paper, we present a novel theoretical framework for watermarking Large Language Models (LLMs) that jointly optimizes both the watermarking scheme and the detection process. Our approach focuses on maximizing detection performance while maintaining control over the worst-case Type-I error and text distortion. We characterize \emph{the universally minimum Type-II error}, showing a fundamental trade-off between watermark detectability and text distortion. Importantly, we identify that the optimal watermarking schemes are adaptive to the LLM generative distribution. Building on our theoretical insights, we propose an efficient, model-agnostic, distribution-adaptive watermarking algorithm, utilizing a surrogate model alongside the Gumbel-max trick. Experiments conducted on Llama2-13B and Mistral-8$\times$7B models confirm the effectiveness of our approach. Additionally, we examine incorporating robustness into our framework, paving a way to future watermarking systems that withstand adversarial attacks more effectively.

摘要: 数字水印已经成为区分人工智能生成的文本和人类创建的文本的关键方法。在本文中，我们提出了一种新的大语言模型(LLMS)水印理论框架，该框架同时优化了水印方案和检测过程。我们的方法专注于最大化检测性能，同时保持对最坏情况下的类型I错误和文本失真的控制。我们将其刻画在水印可检测性和文本失真之间的基本权衡。重要的是，我们发现最优水印方案对LLM生成分布是自适应的。基于我们的理论见解，我们提出了一种高效的、与模型无关的、分布自适应的水印算法，该算法利用代理模型和Gumbel-max技巧。在Llama2-13B和Mistral-8$\x$70亿模型上进行的实验证实了该方法的有效性。此外，我们还研究了将健壮性融入到我们的框架中，为未来更有效地抵御对手攻击的水印系统铺平了道路。



## **13. Reasoning-Augmented Conversation for Multi-Turn Jailbreak Attacks on Large Language Models**

针对大型语言模型的多回合越狱攻击的推理增强对话 cs.CL

**SubmitDate**: 2025-02-19    [abs](http://arxiv.org/abs/2502.11054v3) [paper-pdf](http://arxiv.org/pdf/2502.11054v3)

**Authors**: Zonghao Ying, Deyue Zhang, Zonglei Jing, Yisong Xiao, Quanchen Zou, Aishan Liu, Siyuan Liang, Xiangzheng Zhang, Xianglong Liu, Dacheng Tao

**Abstract**: Multi-turn jailbreak attacks simulate real-world human interactions by engaging large language models (LLMs) in iterative dialogues, exposing critical safety vulnerabilities. However, existing methods often struggle to balance semantic coherence with attack effectiveness, resulting in either benign semantic drift or ineffective detection evasion. To address this challenge, we propose Reasoning-Augmented Conversation, a novel multi-turn jailbreak framework that reformulates harmful queries into benign reasoning tasks and leverages LLMs' strong reasoning capabilities to compromise safety alignment. Specifically, we introduce an attack state machine framework to systematically model problem translation and iterative reasoning, ensuring coherent query generation across multiple turns. Building on this framework, we design gain-guided exploration, self-play, and rejection feedback modules to preserve attack semantics, enhance effectiveness, and sustain reasoning-driven attack progression. Extensive experiments on multiple LLMs demonstrate that RACE achieves state-of-the-art attack effectiveness in complex conversational scenarios, with attack success rates (ASRs) increasing by up to 96%. Notably, our approach achieves ASRs of 82% and 92% against leading commercial models, OpenAI o1 and DeepSeek R1, underscoring its potency. We release our code at https://github.com/NY1024/RACE to facilitate further research in this critical domain.

摘要: 多轮越狱攻击通过在迭代对话中使用大型语言模型(LLM)来模拟真实世界的人类交互，暴露出关键的安全漏洞。然而，现有的方法往往难以在语义一致性和攻击有效性之间取得平衡，导致良性的语义漂移或无效的检测规避。为了应对这一挑战，我们提出了一种新的多轮越狱框架--推理增强对话，该框架将有害的查询重新定义为良性的推理任务，并利用LLMS强大的推理能力来妥协安全对齐。具体地说，我们引入了攻击状态机框架来系统地建模问题转换和迭代推理，确保跨多轮的连贯查询生成。在这个框架的基础上，我们设计了增益引导的探索、自我发挥和拒绝反馈模块，以保留攻击语义，提高有效性，并支持推理驱动的攻击进展。在多个LLM上的大量实验表明，RACE在复杂的会话场景中获得了最先进的攻击效率，攻击成功率(ASR)提高了96%。值得注意的是，我们的方法在领先的商业模型OpenAI o1和DeepSeek r1上分别获得了82%和92%的ASR，这突显了它的有效性。我们在https://github.com/NY1024/RACE上发布我们的代码，以促进在这一关键领域的进一步研究。



## **14. Na'vi or Knave: Jailbreaking Language Models via Metaphorical Avatars**

纳威人或恶棍：通过隐喻化身越狱语言模型 cs.CL

We still need to polish our paper

**SubmitDate**: 2025-02-19    [abs](http://arxiv.org/abs/2412.12145v3) [paper-pdf](http://arxiv.org/pdf/2412.12145v3)

**Authors**: Yu Yan, Sheng Sun, Junqi Tong, Min Liu, Qi Li

**Abstract**: Metaphor serves as an implicit approach to convey information, while enabling the generalized comprehension of complex subjects. However, metaphor can potentially be exploited to bypass the safety alignment mechanisms of Large Language Models (LLMs), leading to the theft of harmful knowledge. In our study, we introduce a novel attack framework that exploits the imaginative capacity of LLMs to achieve jailbreaking, the J\underline{\textbf{A}}ilbreak \underline{\textbf{V}}ia \underline{\textbf{A}}dversarial Me\underline{\textbf{TA}} -pho\underline{\textbf{R}} (\textit{AVATAR}). Specifically, to elicit the harmful response, AVATAR extracts harmful entities from a given harmful target and maps them to innocuous adversarial entities based on LLM's imagination. Then, according to these metaphors, the harmful target is nested within human-like interaction for jailbreaking adaptively. Experimental results demonstrate that AVATAR can effectively and transferablly jailbreak LLMs and achieve a state-of-the-art attack success rate across multiple advanced LLMs. Our study exposes a security risk in LLMs from their endogenous imaginative capabilities. Furthermore, the analytical study reveals the vulnerability of LLM to adversarial metaphors and the necessity of developing defense methods against jailbreaking caused by the adversarial metaphor. \textcolor{orange}{ \textbf{Warning: This paper contains potentially harmful content from LLMs.}}

摘要: 隐喻是一种隐含的传达信息的方法，同时也使人们能够对复杂的主题进行广义的理解。然而，隐喻可能被利用来绕过大型语言模型(LLM)的安全对齐机制，从而导致有害知识的窃取。在我们的研究中，我们提出了一种新的攻击框架，该框架利用LLMS的想象能力来实现越狱，即J下划线{\extbf{A}}ilBreak\下划线{\extbf{A}}ia\下划线{\extbf{A}}dversarial Me\下划线{\extbf{TA}}-pho\下划线{\extbf{R}}(\textit{阿凡达})。具体地说，为了引发有害反应，阿凡达从给定的有害目标中提取有害实体，并基于LLM的想象力将它们映射到无害的对抗性实体。然后，根据这些隐喻，有害的目标嵌套在类似人类的互动中，以适应越狱。实验结果表明，阿凡达可以有效地、可转移地越狱LLM，并在多个高级LLM上获得最先进的攻击成功率。我们的研究揭示了LLMS的安全风险来自其内生的想象力。此外，分析研究揭示了LLM对对抗性隐喻的脆弱性，以及开发针对对抗性隐喻导致的越狱的防御方法的必要性。\extCOLOR{橙色}{\extbf{警告：此纸张包含来自LLMS的潜在有害内容。}}



## **15. Efficient Safety Retrofitting Against Jailbreaking for LLMs**

针对LLM越狱的高效安全改造 cs.CL

**SubmitDate**: 2025-02-19    [abs](http://arxiv.org/abs/2502.13603v1) [paper-pdf](http://arxiv.org/pdf/2502.13603v1)

**Authors**: Dario Garcia-Gasulla, Anna Arias-Duart, Adrian Tormos, Daniel Hinjos, Oscar Molina-Sedano, Ashwin Kumar Gururajan, Maria Eugenia Cardello

**Abstract**: Direct Preference Optimization (DPO) is an efficient alignment technique that steers LLMs towards preferable outputs by training on preference data, bypassing the need for explicit reward models. Its simplicity enables easy adaptation to various domains and safety requirements. This paper examines DPO's effectiveness in model safety against jailbreaking attacks while minimizing data requirements and training costs. We introduce Egida, a dataset expanded from multiple sources, which includes 27 different safety topics and 18 different attack styles, complemented with synthetic and human labels. This data is used to boost the safety of state-of-the-art LLMs (Llama-3.1-8B/70B-Instruct, Qwen-2.5-7B/72B-Instruct) across topics and attack styles. In addition to safety evaluations, we assess their post-alignment performance degradation in general purpose tasks, and their tendency to over refusal. Following the proposed methodology, trained models reduce their Attack Success Rate by 10%-30%, using small training efforts (2,000 samples) with low computational cost (3\$ for 8B models, 20\$ for 72B models). Safety aligned models generalize to unseen topics and attack styles, with the most successful attack style reaching a success rate around 5%. Size and family are found to strongly influence model malleability towards safety, pointing at the importance of pre-training choices. To validate our findings, a large independent assessment of human preference agreement with Llama-Guard-3-8B is conducted by the authors and the associated dataset Egida-HSafe is released. Overall, this study illustrates how affordable and accessible it is to enhance LLM safety using DPO while outlining its current limitations. All datasets and models are released to enable reproducibility and further research.

摘要: 直接偏好优化(DPO)是一种有效的对齐技术，它通过对偏好数据的训练来引导LLM朝着更好的输出方向前进，而不需要显式的奖励模型。它的简单性使其能够轻松适应各种域和安全要求。本文研究了DPO在防止越狱攻击的模型安全方面的有效性，同时将数据需求和培训成本降至最低。我们介绍了Egida，这是一个从多个来源扩展的数据集，包括27个不同的安全主题和18个不同的攻击风格，并辅之以合成标签和人工标签。这些数据用于提高最先进的LLMS(LAMA-3.1-8B/70B-指令，QWEN-2.5-7B/72B-指令)跨主题和攻击风格的安全性。除了安全评估外，我们还评估了他们在一般目的任务中对齐后的表现降级，以及他们过度拒绝的倾向。按照所提出的方法，训练模型的攻击成功率降低了10%-30%，使用较小的训练工作量(2000个样本)和较低的计算成本(8B模型3个，72B模型20个)。安全对齐的模型概括为未知的主题和攻击风格，最成功的攻击风格的成功率约为5%。尺寸和家庭被发现强烈影响模型的安全延展性，这表明了培训前选择的重要性。为了验证我们的发现，作者对人类偏好与Llama-Guard-3-8B的一致性进行了大规模的独立评估，并发布了相关的数据集Egida-HSafe。总体而言，这项研究说明了使用DPO增强LLM安全性的负担能力和可获得性，同时概述了其当前的局限性。所有数据集和模型都被公布，以便于重现性和进一步研究。



## **16. Exploiting Prefix-Tree in Structured Output Interfaces for Enhancing Jailbreak Attacking**

利用结构化输出接口中的后缀树增强越狱攻击 cs.CR

**SubmitDate**: 2025-02-19    [abs](http://arxiv.org/abs/2502.13527v1) [paper-pdf](http://arxiv.org/pdf/2502.13527v1)

**Authors**: Yanzeng Li, Yunfan Xiong, Jialun Zhong, Jinchao Zhang, Jie Zhou, Lei Zou

**Abstract**: The rise of Large Language Models (LLMs) has led to significant applications but also introduced serious security threats, particularly from jailbreak attacks that manipulate output generation. These attacks utilize prompt engineering and logit manipulation to steer models toward harmful content, prompting LLM providers to implement filtering and safety alignment strategies. We investigate LLMs' safety mechanisms and their recent applications, revealing a new threat model targeting structured output interfaces, which enable attackers to manipulate the inner logit during LLM generation, requiring only API access permissions. To demonstrate this threat model, we introduce a black-box attack framework called AttackPrefixTree (APT). APT exploits structured output interfaces to dynamically construct attack patterns. By leveraging prefixes of models' safety refusal response and latent harmful outputs, APT effectively bypasses safety measures. Experiments on benchmark datasets indicate that this approach achieves higher attack success rate than existing methods. This work highlights the urgent need for LLM providers to enhance security protocols to address vulnerabilities arising from the interaction between safety patterns and structured outputs.

摘要: 大型语言模型(LLM)的兴起带来了重要的应用，但也带来了严重的安全威胁，特别是来自操纵输出生成的越狱攻击。这些攻击利用即时工程和Logit操纵来引导模型指向有害内容，促使LLM提供商实施过滤和安全对齐策略。我们研究了LLMS的安全机制及其最新应用，揭示了一种新的针对结构化输出接口的威胁模型，该模型使攻击者能够在LLM生成过程中操纵内部日志，只需要API访问权限。为了演示这种威胁模型，我们引入了一个名为AttackPrefix Tree(APT)的黑盒攻击框架。APT利用结构化输出接口动态构建攻击模式。通过利用模型的安全拒绝响应前缀和潜在的有害输出，APT有效地绕过了安全措施。在基准数据集上的实验表明，该方法取得了比现有方法更高的攻击成功率。这项工作突出表明，土地管理提供者迫切需要加强安全协议，以解决安全模式和结构化产出之间相互作用所产生的脆弱性。



## **17. Evaluating Implicit Bias in Large Language Models by Attacking From a Psychometric Perspective**

从心理测量学角度通过攻击来评估大型语言模型中的内隐偏差 cs.CL

Code, data and benchmarks are available at  https://github.com/yuchenwen1/ImplicitBiasPsychometricEvaluation and  https://github.com/yuchenwen1/BUMBLE

**SubmitDate**: 2025-02-19    [abs](http://arxiv.org/abs/2406.14023v2) [paper-pdf](http://arxiv.org/pdf/2406.14023v2)

**Authors**: Yuchen Wen, Keping Bi, Wei Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: As large language models (LLMs) become an important way of information access, there have been increasing concerns that LLMs may intensify the spread of unethical content, including implicit bias that hurts certain populations without explicit harmful words. In this paper, we conduct a rigorous evaluation of LLMs' implicit bias towards certain demographics by attacking them from a psychometric perspective to elicit agreements to biased viewpoints. Inspired by psychometric principles in cognitive and social psychology, we propose three attack approaches, i.e., Disguise, Deception, and Teaching. Incorporating the corresponding attack instructions, we built two benchmarks: (1) a bilingual dataset with biased statements covering four bias types (2.7K instances) for extensive comparative analysis, and (2) BUMBLE, a larger benchmark spanning nine common bias types (12.7K instances) for comprehensive evaluation. Extensive evaluation of popular commercial and open-source LLMs shows that our methods can elicit LLMs' inner bias more effectively than competitive baselines. Our attack methodology and benchmarks offer an effective means of assessing the ethical risks of LLMs, driving progress toward greater accountability in their development.

摘要: 随着大型语言模型(LLM)成为信息获取的重要方式，人们越来越担心LLM可能会加剧不道德内容的传播，包括在没有显性有害词语的情况下伤害某些人群的隐性偏见。在这篇论文中，我们从心理测量学的角度对LLMS对某些人口统计学的内隐偏见进行了严格的评估，以获得对有偏见的观点的一致意见。受认知心理学和社会心理学中的心理测量学原理的启发，我们提出了三种攻击方法，即伪装、欺骗和教学。结合相应的攻击指令，我们构建了两个基准：(1)包含四种偏见类型(2.7k实例)的双语偏向语句数据集，用于广泛的比较分析；(2)Bumble，一个涵盖九种常见偏见类型(12.7k实例)的更大基准，用于综合评估。对流行的商业和开源LLM的广泛评估表明，我们的方法可以比竞争基线更有效地引出LLM的内部偏差。我们的攻击方法和基准为评估LLM的道德风险提供了一种有效的手段，推动了在其开发中更大程度地追究责任的进展。



## **18. AutoTEE: Automated Migration and Protection of Programs in Trusted Execution Environments**

AutoTEK：可信执行环境中程序的自动迁移和保护 cs.CR

14 pages

**SubmitDate**: 2025-02-19    [abs](http://arxiv.org/abs/2502.13379v1) [paper-pdf](http://arxiv.org/pdf/2502.13379v1)

**Authors**: Ruidong Han, Zhou Yang, Chengyan Ma, Ye Liu, Yuqing Niu, Siqi Ma, Debin Gao, David Lo

**Abstract**: Trusted Execution Environments (TEEs) isolate a special space within a device's memory that is not accessible to the normal world (also known as Untrusted Environment), even when the device is compromised. Thus, developers can utilize TEEs to provide strong security guarantees for their programs, making sensitive operations like encrypted data storage, fingerprint verification, and remote attestation protected from malicious attacks. Despite the strong protections offered by TEEs, adapting existing programs to leverage such security guarantees is non-trivial, often requiring extensive domain knowledge and manual intervention, which makes TEEs less accessible to developers. This motivates us to design AutoTEE, the first Large Language Model (LLM)-enabled approach that can automatically identify, partition, transform, and port sensitive functions into TEEs with minimal developer intervention. By manually reviewing 68 repositories, we constructed a benchmark dataset consisting of 385 sensitive functions eligible for transformation, on which AutoTEE achieves a high F1 score of 0.91. AutoTEE effectively transforms these sensitive functions into their TEE-compatible counterparts, achieving success rates of 90\% and 83\% for Java and Python, respectively. We further provide a mechanism to automatically port the transformed code to different TEE platforms, including Intel SGX and AMD SEV, demonstrating that the transformed programs run successfully and correctly on these platforms.

摘要: 可信执行环境(TEE)隔离正常世界无法访问的设备内存中的特殊空间(也称为不可信环境)，即使设备受到攻击也是如此。因此，开发人员可以利用TES为他们的程序提供强大的安全保证，使加密数据存储、指纹验证和远程证明等敏感操作免受恶意攻击。尽管TEE提供了强大的保护，但调整现有程序以利用此类安全保证并不是一件容易的事情，通常需要广泛的领域知识和人工干预，这使得TEE更难被开发人员访问。这促使我们设计AutoTEE，这是第一种启用大型语言模型(LLM)的方法，可以自动识别、分区、转换敏感函数并将其移植到TEE中，而只需最少的开发人员干预。通过人工审查68个知识库，我们构建了一个由385个敏感函数组成的基准数据集，AutoTEE在该数据集上获得了0.91的高F1分数。AutoTEE有效地将这些敏感函数转换为与TEE兼容的对应函数，在Java和Python上分别实现了90%和83%的成功率。我们进一步提供了一种机制，将转换后的代码自动移植到不同的TEE平台，包括Intel SGX和AMD SEV，证明转换后的程序在这些平台上成功和正确地运行。



## **19. Improving Acoustic Side-Channel Attacks on Keyboards Using Transformers and Large Language Models**

使用变形金刚和大型语言模型改善对键盘的声学侧通道攻击 cs.LG

We would like to withdraw our paper due to a significant error in the  experimental methodology, which impacts the validity of our results. The  error specifically affects the analysis presented in Section 4, where an  incorrect dataset preprocessing step led to misleading conclusions

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.09782v3) [paper-pdf](http://arxiv.org/pdf/2502.09782v3)

**Authors**: Jin Hyun Park, Seyyed Ali Ayati, Yichen Cai

**Abstract**: The increasing prevalence of microphones in everyday devices and the growing reliance on online services have amplified the risk of acoustic side-channel attacks (ASCAs) targeting keyboards. This study explores deep learning techniques, specifically vision transformers (VTs) and large language models (LLMs), to enhance the effectiveness and applicability of such attacks. We present substantial improvements over prior research, with the CoAtNet model achieving state-of-the-art performance. Our CoAtNet shows a 5.0% improvement for keystrokes recorded via smartphone (Phone) and 5.9% for those recorded via Zoom compared to previous benchmarks. We also evaluate transformer architectures and language models, with the best VT model matching CoAtNet's performance. A key advancement is the introduction of a noise mitigation method for real-world scenarios. By using LLMs for contextual understanding, we detect and correct erroneous keystrokes in noisy environments, enhancing ASCA performance. Additionally, fine-tuned lightweight language models with Low-Rank Adaptation (LoRA) deliver comparable performance to heavyweight models with 67X more parameters. This integration of VTs and LLMs improves the practical applicability of ASCA mitigation, marking the first use of these technologies to address ASCAs and error correction in real-world scenarios.

摘要: 麦克风在日常设备中的日益普及以及对在线服务的日益依赖，放大了针对键盘的声学侧通道攻击(ASCA)的风险。本研究探索深度学习技术，特别是视觉转换器(VT)和大语言模型(LLM)，以增强此类攻击的有效性和适用性。与之前的研究相比，我们提出了实质性的改进，CoAtNet模型实现了最先进的性能。我们的CoAtNet显示，与之前的基准相比，通过智能手机(手机)记录的击键次数提高了5.0%，通过Zoom记录的击键次数提高了5.9%。我们还评估了转换器体系结构和语言模型，选择了与CoAtNet性能匹配的最佳VT模型。一个关键的进步是引入了一种用于真实世界场景的噪音缓解方法。通过使用LLMS进行上下文理解，我们可以在噪声环境中检测并纠正错误的击键，从而提高ASCA的性能。此外，带有低阶自适应(LORA)的微调轻量级语言模型提供了与参数多67倍的重量级模型相当的性能。VTS和LLMS的这种集成提高了ASCA缓解的实际适用性，标志着这些技术首次用于解决现实世界场景中的ASCA和纠错。



## **20. UniGuardian: A Unified Defense for Detecting Prompt Injection, Backdoor Attacks and Adversarial Attacks in Large Language Models**

UniGuardian：检测大型语言模型中的即时注入、后门攻击和对抗攻击的统一防御 cs.CL

18 Pages, 8 Figures, 5 Tables, Keywords: Attack Defending, Security,  Prompt Injection, Backdoor Attacks, Adversarial Attacks, Prompt Trigger  Attacks

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.13141v1) [paper-pdf](http://arxiv.org/pdf/2502.13141v1)

**Authors**: Huawei Lin, Yingjie Lao, Tong Geng, Tan Yu, Weijie Zhao

**Abstract**: Large Language Models (LLMs) are vulnerable to attacks like prompt injection, backdoor attacks, and adversarial attacks, which manipulate prompts or models to generate harmful outputs. In this paper, departing from traditional deep learning attack paradigms, we explore their intrinsic relationship and collectively term them Prompt Trigger Attacks (PTA). This raises a key question: Can we determine if a prompt is benign or poisoned? To address this, we propose UniGuardian, the first unified defense mechanism designed to detect prompt injection, backdoor attacks, and adversarial attacks in LLMs. Additionally, we introduce a single-forward strategy to optimize the detection pipeline, enabling simultaneous attack detection and text generation within a single forward pass. Our experiments confirm that UniGuardian accurately and efficiently identifies malicious prompts in LLMs.

摘要: 大型语言模型（LLM）容易受到提示注入、后门攻击和对抗攻击等攻击，这些攻击操纵提示或模型以生成有害输出。本文脱离传统的深度学习攻击范式，探索它们的内在关系，并将它们统称为提示触发攻击（PTA）。这提出了一个关键问题：我们能否确定提示是良性的还是有毒的？为了解决这个问题，我们提出了UniGuardian，这是第一个旨在检测LLM中的即时注入、后门攻击和对抗攻击的统一防御机制。此外，我们还引入了单转发策略来优化检测管道，从而在单次转发内同时进行攻击检测和文本生成。我们的实验证实，UniGuardian可以准确有效地识别LLM中的恶意提示。



## **21. Emoji Attack: Enhancing Jailbreak Attacks Against Judge LLM Detection**

Emoji攻击：增强针对LLM法官检测的越狱攻击 cs.CL

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2411.01077v2) [paper-pdf](http://arxiv.org/pdf/2411.01077v2)

**Authors**: Zhipeng Wei, Yuqi Liu, N. Benjamin Erichson

**Abstract**: Jailbreaking techniques trick Large Language Models (LLMs) into producing restricted outputs, posing a serious threat. One line of defense is to use another LLM as a Judge to evaluate the harmfulness of generated text. However, we reveal that these Judge LLMs are vulnerable to token segmentation bias, an issue that arises when delimiters alter the tokenization process, splitting words into smaller sub-tokens. This disrupts the embeddings of the entire sequence, reducing detection accuracy and allowing harmful content to be misclassified as safe. In this paper, we introduce Emoji Attack, a novel strategy that amplifies existing jailbreak prompts by exploiting token segmentation bias. Our method leverages in-context learning to systematically insert emojis into text before it is evaluated by a Judge LLM, inducing embedding distortions that significantly lower the likelihood of detecting unsafe content. Unlike traditional delimiters, emojis also introduce semantic ambiguity, making them particularly effective in this attack. Through experiments on state-of-the-art Judge LLMs, we demonstrate that Emoji Attack substantially reduces the "unsafe" prediction rate, bypassing existing safeguards.

摘要: 越狱技术诱使大型语言模型(LLM)产生受限的输出，构成了严重的威胁。一条防线是使用另一个LLM作为法官来评估生成的文本的危害性。然而，我们发现这些裁判LLM容易受到标记分割偏差的影响，当分隔符改变标记化过程，将单词分割成更小的子标记时，就会出现这个问题。这扰乱了整个序列的嵌入，降低了检测精度，并允许有害内容被错误归类为安全内容。在本文中，我们介绍了Emoji攻击，这是一种新的策略，通过利用令牌分段偏差来放大现有的越狱提示。我们的方法利用情景学习在文本中系统地插入表情符号，然后由法官LLM对其进行评估，从而导致嵌入扭曲，从而显著降低检测到不安全内容的可能性。与传统的分隔符不同，表情符号还会引入语义歧义，使其在这次攻击中特别有效。通过对最先进的JUARY LLMS的实验，我们证明了Emoji攻击绕过了现有的安全措施，大大降低了“不安全”的预测率。



## **22. LAMD: Context-driven Android Malware Detection and Classification with LLMs**

LAMD：使用LLM的上下文驱动Android恶意软件检测和分类 cs.CR

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.13055v1) [paper-pdf](http://arxiv.org/pdf/2502.13055v1)

**Authors**: Xingzhi Qian, Xinran Zheng, Yiling He, Shuo Yang, Lorenzo Cavallaro

**Abstract**: The rapid growth of mobile applications has escalated Android malware threats. Although there are numerous detection methods, they often struggle with evolving attacks, dataset biases, and limited explainability. Large Language Models (LLMs) offer a promising alternative with their zero-shot inference and reasoning capabilities. However, applying LLMs to Android malware detection presents two key challenges: (1)the extensive support code in Android applications, often spanning thousands of classes, exceeds LLMs' context limits and obscures malicious behavior within benign functionality; (2)the structural complexity and interdependencies of Android applications surpass LLMs' sequence-based reasoning, fragmenting code analysis and hindering malicious intent inference. To address these challenges, we propose LAMD, a practical context-driven framework to enable LLM-based Android malware detection. LAMD integrates key context extraction to isolate security-critical code regions and construct program structures, then applies tier-wise code reasoning to analyze application behavior progressively, from low-level instructions to high-level semantics, providing final prediction and explanation. A well-designed factual consistency verification mechanism is equipped to mitigate LLM hallucinations from the first tier. Evaluation in real-world settings demonstrates LAMD's effectiveness over conventional detectors, establishing a feasible basis for LLM-driven malware analysis in dynamic threat landscapes.

摘要: 移动应用程序的快速增长升级了Android恶意软件威胁。虽然有许多检测方法，但它们经常与不断演变的攻击、数据集偏差和有限的可解释性作斗争。大型语言模型(LLM)以其零概率推理和推理能力提供了一种很有前途的替代方案。然而，将LLMS应用于Android恶意软件检测面临两个关键挑战：(1)Android应用程序中广泛的支持代码，往往跨越数千个类，超出了LLMS的上下文限制，掩盖了良性功能中的恶意行为；(2)Android应用程序的结构复杂性和相关性超过了LLMS的基于序列的推理、代码分析碎片化和阻碍恶意意图推理。为了应对这些挑战，我们提出了LAMD，一个实用的上下文驱动的框架来实现基于LLM的Android恶意软件检测。LAMD结合关键上下文提取来隔离安全关键代码区域并构建程序结构，然后应用分层代码推理从低级指令到高级语义逐步分析应用程序行为，提供最终预测和解释。配备了设计良好的事实一致性验证机制，以缓解第一级的LLM幻觉。在真实环境中的评估证明了LAMD相对于传统检测器的有效性，为动态威胁环境中LLM驱动的恶意软件分析奠定了可行的基础。



## **23. Reasoning-to-Defend: Safety-Aware Reasoning Can Defend Large Language Models from Jailbreaking**

推理防御：安全意识推理可以保护大型语言模型免受越狱 cs.CL

18 pages

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.12970v1) [paper-pdf](http://arxiv.org/pdf/2502.12970v1)

**Authors**: Junda Zhu, Lingyong Yan, Shuaiqiang Wang, Dawei Yin, Lei Sha

**Abstract**: The reasoning abilities of Large Language Models (LLMs) have demonstrated remarkable advancement and exceptional performance across diverse domains. However, leveraging these reasoning capabilities to enhance LLM safety against adversarial attacks and jailbreak queries remains largely unexplored. To bridge this gap, we propose Reasoning-to-Defend (R2D), a novel training paradigm that integrates safety reflections of queries and responses into LLMs' generation process, unlocking a safety-aware reasoning mechanism. This approach enables self-evaluation at each reasoning step to create safety pivot tokens as indicators of the response's safety status. Furthermore, in order to improve the learning efficiency of pivot token prediction, we propose Contrastive Pivot Optimization(CPO), which enhances the model's ability to perceive the safety status of dialogues. Through this mechanism, LLMs dynamically adjust their response strategies during reasoning, significantly enhancing their defense capabilities against jailbreak attacks. Extensive experimental results demonstrate that R2D effectively mitigates various attacks and improves overall safety, highlighting the substantial potential of safety-aware reasoning in strengthening LLMs' robustness against jailbreaks.

摘要: 大型语言模型的推理能力在不同领域表现出了显著的进步和卓越的表现。然而，利用这些推理能力来增强LLM针对对手攻击和越狱查询的安全性在很大程度上仍未得到探索。为了弥补这一差距，我们提出了推理防御(R2D)，这是一种新的训练范式，将查询和响应的安全反映整合到LLMS的生成过程中，从而解锁了一种安全感知的推理机制。这种方法允许在每个推理步骤进行自我评估，以创建安全轴心令牌作为响应的安全状态的指示器。此外，为了提高枢轴标记预测的学习效率，提出了对比枢轴优化算法(CPO)，增强了模型对对话安全状态的感知能力。通过这种机制，LLMS在推理过程中动态调整响应策略，显著增强了对越狱攻击的防御能力。广泛的实验结果表明，R2D有效地缓解了各种攻击，提高了整体安全性，突出了安全意识推理在增强LLMS对越狱的健壮性方面的巨大潜力。



## **24. H-CoT: Hijacking the Chain-of-Thought Safety Reasoning Mechanism to Jailbreak Large Reasoning Models, Including OpenAI o1/o3, DeepSeek-R1, and Gemini 2.0 Flash Thinking**

H-CoT：劫持思想链安全推理机制越狱大型推理模型，包括OpenAI o 1/o3、DeepSeek-R1和Gemini 2.0 Flash Thinking cs.CL

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.12893v1) [paper-pdf](http://arxiv.org/pdf/2502.12893v1)

**Authors**: Martin Kuo, Jianyi Zhang, Aolin Ding, Qinsi Wang, Louis DiValentin, Yujia Bao, Wei Wei, Da-Cheng Juan, Hai Li, Yiran Chen

**Abstract**: Large Reasoning Models (LRMs) have recently extended their powerful reasoning capabilities to safety checks-using chain-of-thought reasoning to decide whether a request should be answered. While this new approach offers a promising route for balancing model utility and safety, its robustness remains underexplored. To address this gap, we introduce Malicious-Educator, a benchmark that disguises extremely dangerous or malicious requests beneath seemingly legitimate educational prompts. Our experiments reveal severe security flaws in popular commercial-grade LRMs, including OpenAI o1/o3, DeepSeek-R1, and Gemini 2.0 Flash Thinking. For instance, although OpenAI's o1 model initially maintains a high refusal rate of about 98%, subsequent model updates significantly compromise its safety; and attackers can easily extract criminal strategies from DeepSeek-R1 and Gemini 2.0 Flash Thinking without any additional tricks. To further highlight these vulnerabilities, we propose Hijacking Chain-of-Thought (H-CoT), a universal and transferable attack method that leverages the model's own displayed intermediate reasoning to jailbreak its safety reasoning mechanism. Under H-CoT, refusal rates sharply decline-dropping from 98% to below 2%-and, in some instances, even transform initially cautious tones into ones that are willing to provide harmful content. We hope these findings underscore the urgent need for more robust safety mechanisms to preserve the benefits of advanced reasoning capabilities without compromising ethical standards.

摘要: 大型推理模型(LRM)最近将其强大的推理能力扩展到安全检查-使用思想链推理来决定是否应该响应请求。虽然这一新方法为平衡模型的实用性和安全性提供了一条有希望的途径，但其稳健性仍未得到充分探索。为了弥补这一差距，我们引入了恶意教育者，这是一个基准，它将极其危险或恶意的请求掩盖在看似合法的教育提示之下。我们的实验揭示了流行的商业级LRMS存在严重的安全漏洞，包括OpenAI o1/03、DeepSeek-r1和Gemini 2.0 Flash Think。例如，尽管OpenAI的o1模型最初保持了98%左右的高拒绝率，但后续的模型更新显著损害了其安全性；攻击者可以轻松地从DeepSeek-R1和Gemini 2.0 Flash Think中提取犯罪策略，而不需要任何额外的伎俩。为了进一步突出这些漏洞，我们提出了劫持思想链(H-CoT)，这是一种通用的、可转移的攻击方法，它利用模型自己展示的中间推理来越狱其安全推理机制。在H-cot下，拒绝率大幅下降--从98%降至2%以下--在某些情况下，甚至会将最初谨慎的语气转变为愿意提供有害内容的语气。我们希望这些发现强调了迫切需要更强大的安全机制，以在不损害道德标准的情况下保留先进推理能力的好处。



## **25. ALGEN: Few-shot Inversion Attacks on Textual Embeddings using Alignment and Generation**

ALgen：使用对齐和生成对文本嵌入进行少量反转攻击 cs.CR

18 pages, 13 tables, 6 figures

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.11308v2) [paper-pdf](http://arxiv.org/pdf/2502.11308v2)

**Authors**: Yiyi Chen, Qiongkai Xu, Johannes Bjerva

**Abstract**: With the growing popularity of Large Language Models (LLMs) and vector databases, private textual data is increasingly processed and stored as numerical embeddings. However, recent studies have proven that such embeddings are vulnerable to inversion attacks, where original text is reconstructed to reveal sensitive information. Previous research has largely assumed access to millions of sentences to train attack models, e.g., through data leakage or nearly unrestricted API access. With our method, a single data point is sufficient for a partially successful inversion attack. With as little as 1k data samples, performance reaches an optimum across a range of black-box encoders, without training on leaked data. We present a Few-shot Textual Embedding Inversion Attack using ALignment and GENeration (ALGEN), by aligning victim embeddings to the attack space and using a generative model to reconstruct text. We find that ALGEN attacks can be effectively transferred across domains and languages, revealing key information. We further examine a variety of defense mechanisms against ALGEN, and find that none are effective, highlighting the vulnerabilities posed by inversion attacks. By significantly lowering the cost of inversion and proving that embedding spaces can be aligned through one-step optimization, we establish a new textual embedding inversion paradigm with broader applications for embedding alignment in NLP.

摘要: 随着大型语言模型和向量数据库的日益流行，私有文本数据越来越多地以数字嵌入的形式进行处理和存储。然而，最近的研究证明，这种嵌入很容易受到反转攻击，即重建原始文本以泄露敏感信息。以前的研究在很大程度上假设可以访问数百万个句子来训练攻击模型，例如通过数据泄漏或几乎不受限制的API访问。使用我们的方法，对于部分成功的反转攻击，单个数据点就足够了。只需1000个数据样本，一系列黑盒编码器的性能就可以达到最佳，而不需要对泄露的数据进行培训。提出了一种基于对齐和生成的少量文本嵌入反转攻击(ALGEN)，通过将受害者嵌入对齐到攻击空间，并使用生成模型来重构文本。我们发现，ALGEN攻击可以有效地跨域和跨语言传输，泄露关键信息。我们进一步研究了针对ALGEN的各种防御机制，发现没有一种机制是有效的，这突出了反转攻击带来的漏洞。通过显著降低反转的代价，并证明嵌入空间可以通过一步优化来对齐，我们建立了一种新的文本嵌入反转范式，在自然语言处理中具有更广泛的应用前景。



## **26. The Hidden Risks of Large Reasoning Models: A Safety Assessment of R1**

大型推理模型的隐藏风险：R1的安全评估 cs.CY

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.12659v1) [paper-pdf](http://arxiv.org/pdf/2502.12659v1)

**Authors**: Kaiwen Zhou, Chengzhi Liu, Xuandong Zhao, Shreedhar Jangam, Jayanth Srinivasa, Gaowen Liu, Dawn Song, Xin Eric Wang

**Abstract**: The rapid development of large reasoning models, such as OpenAI-o3 and DeepSeek-R1, has led to significant improvements in complex reasoning over non-reasoning large language models~(LLMs). However, their enhanced capabilities, combined with the open-source access of models like DeepSeek-R1, raise serious safety concerns, particularly regarding their potential for misuse. In this work, we present a comprehensive safety assessment of these reasoning models, leveraging established safety benchmarks to evaluate their compliance with safety regulations. Furthermore, we investigate their susceptibility to adversarial attacks, such as jailbreaking and prompt injection, to assess their robustness in real-world applications. Through our multi-faceted analysis, we uncover four key findings: (1) There is a significant safety gap between the open-source R1 models and the o3-mini model, on both safety benchmark and attack, suggesting more safety effort on R1 is needed. (2) The distilled reasoning model shows poorer safety performance compared to its safety-aligned base models. (3) The stronger the model's reasoning ability, the greater the potential harm it may cause when answering unsafe questions. (4) The thinking process in R1 models pose greater safety concerns than their final answers. Our study provides insights into the security implications of reasoning models and highlights the need for further advancements in R1 models' safety to close the gap.

摘要: 大型推理模型的快速发展，如OpenAI-03和DeepSeek-R1，使得复杂推理相对于非推理的大型语言模型有了显著的改进。然而，它们增强的能力，再加上DeepSeek-R1等型号的开源访问，引发了严重的安全问题，特别是它们可能被滥用的问题。在这项工作中，我们提出了这些推理模型的全面安全评估，利用已建立的安全基准来评估它们是否符合安全法规。此外，我们调查了它们对敌意攻击的敏感性，例如越狱和快速注入，以评估它们在现实世界应用程序中的健壮性。通过多方面的分析，我们发现了四个重要的发现：(1)无论是在安全基准上还是在攻击上，开源的R1型号和03-mini型号之间都存在着显著的安全差距，这表明需要在R1上做出更多的安全努力。(2)与安全对齐的基本模型相比，精炼推理模型的安全性能较差。(3)模型的推理能力越强，在回答不安全问题时可能造成的潜在危害就越大。(4)与最终答案相比，R1模型的思维过程带来了更大的安全顾虑。我们的研究为推理模型的安全含义提供了见解，并强调了在R1模型的安全性方面进一步改进的必要性，以缩小差距。



## **27. R.R.: Unveiling LLM Training Privacy through Recollection and Ranking**

RR：通过回忆和排名揭露LLM培训隐私 cs.CL

13 pages, 9 figures

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.12658v1) [paper-pdf](http://arxiv.org/pdf/2502.12658v1)

**Authors**: Wenlong Meng, Zhenyuan Guo, Lenan Wu, Chen Gong, Wenyan Liu, Weixian Li, Chengkun Wei, Wenzhi Chen

**Abstract**: Large Language Models (LLMs) pose significant privacy risks, potentially leaking training data due to implicit memorization. Existing privacy attacks primarily focus on membership inference attacks (MIAs) or data extraction attacks, but reconstructing specific personally identifiable information (PII) in LLM's training data remains challenging. In this paper, we propose R.R. (Recollect and Rank), a novel two-step privacy stealing attack that enables attackers to reconstruct PII entities from scrubbed training data where the PII entities have been masked. In the first stage, we introduce a prompt paradigm named recollection, which instructs the LLM to repeat a masked text but fill in masks. Then we can use PII identifiers to extract recollected PII candidates. In the second stage, we design a new criterion to score each PII candidate and rank them. Motivated by membership inference, we leverage the reference model as a calibration to our criterion. Experiments across three popular PII datasets demonstrate that the R.R. achieves better PII identical performance compared to baselines. These results highlight the vulnerability of LLMs to PII leakage even when training data has been scrubbed. We release the replicate package of R.R. at a link.

摘要: 大型语言模型(LLM)会带来很大的隐私风险，可能会因为隐式记忆而泄露训练数据。现有的隐私攻击主要集中在成员关系推断攻击(MIA)或数据提取攻击上，但在LLM的训练数据中重建特定的个人身份信息(PII)仍然具有挑战性。在本文中，我们提出了R.R.(Recollect And Rank)，这是一种新的两步隐私窃取攻击，使攻击者能够从擦除的训练数据中重建PII实体，其中PII实体已经被屏蔽。在第一阶段，我们引入了一种名为Recollect的提示范式，它指示LLM重复掩码文本但填充掩码。然后，我们可以使用PII标识符来提取重新收集的PII候选。在第二阶段，我们设计了一个新的标准来对每个PII候选者进行评分和排名。在成员关系推理的激励下，我们利用参考模型作为对我们标准的校准。在三个流行的PII数据集上的实验表明，与基准相比，R.R.获得了更好的PII相同性能。这些结果突出了LLMS对PII泄漏的脆弱性，即使在训练数据已经被擦除的情况下也是如此。我们在一个链接上发布了R.R.的复制包。



## **28. Automating Prompt Leakage Attacks on Large Language Models Using Agentic Approach**

使用统计方法自动对大型语言模型进行即时泄漏攻击 cs.CR

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.12630v1) [paper-pdf](http://arxiv.org/pdf/2502.12630v1)

**Authors**: Tvrtko Sternak, Davor Runje, Dorian Granoša, Chi Wang

**Abstract**: This paper presents a novel approach to evaluating the security of large language models (LLMs) against prompt leakage-the exposure of system-level prompts or proprietary configurations. We define prompt leakage as a critical threat to secure LLM deployment and introduce a framework for testing the robustness of LLMs using agentic teams. Leveraging AG2 (formerly AutoGen), we implement a multi-agent system where cooperative agents are tasked with probing and exploiting the target LLM to elicit its prompt.   Guided by traditional definitions of security in cryptography, we further define a prompt leakage-safe system as one in which an attacker cannot distinguish between two agents: one initialized with an original prompt and the other with a prompt stripped of all sensitive information. In a safe system, the agents' outputs will be indistinguishable to the attacker, ensuring that sensitive information remains secure. This cryptographically inspired framework provides a rigorous standard for evaluating and designing secure LLMs.   This work establishes a systematic methodology for adversarial testing of prompt leakage, bridging the gap between automated threat modeling and practical LLM security.   You can find the implementation of our prompt leakage probing on GitHub.

摘要: 本文提出了一种新的方法来评估大型语言模型(LLM)的安全性，以防止系统级提示或专有配置的即时泄漏。我们将快速泄漏定义为对安全LLM部署的严重威胁，并引入了使用代理团队测试LLM健壮性的框架。利用AG2(以前的Autogen)，我们实现了一个多智能体系统，其中合作智能体的任务是探测和利用目标LLM以获得其提示。在传统密码学安全定义的指导下，我们进一步将即时泄漏安全系统定义为攻击者不能区分两个代理：一个是用原始提示初始化的，另一个是剥离所有敏感信息的提示。在安全的系统中，攻击者无法区分代理的输出，从而确保敏感信息的安全。这个受密码启发的框架为评估和设计安全的LLM提供了严格的标准。这项工作为即时泄漏的对抗性测试建立了一套系统的方法，弥合了自动威胁建模和实用LLM安全之间的差距。您可以在GitHub上找到我们的即时泄漏探测的实现。



## **29. Crabs: Consuming Resource via Auto-generation for LLM-DoS Attack under Black-box Settings**

螃蟹：黑匣子设置下通过自动生成来消耗资源进行LLM-NOS攻击 cs.CL

22 pages, 8 figures, 11 tables

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2412.13879v3) [paper-pdf](http://arxiv.org/pdf/2412.13879v3)

**Authors**: Yuanhe Zhang, Zhenhong Zhou, Wei Zhang, Xinyue Wang, Xiaojun Jia, Yang Liu, Sen Su

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across diverse tasks yet still are vulnerable to external threats, particularly LLM Denial-of-Service (LLM-DoS) attacks. Specifically, LLM-DoS attacks aim to exhaust computational resources and block services. However, existing studies predominantly focus on white-box attacks, leaving black-box scenarios underexplored. In this paper, we introduce Auto-Generation for LLM-DoS (AutoDoS) attack, an automated algorithm designed for black-box LLMs. AutoDoS constructs the DoS Attack Tree and expands the node coverage to achieve effectiveness under black-box conditions. By transferability-driven iterative optimization, AutoDoS could work across different models in one prompt. Furthermore, we reveal that embedding the Length Trojan allows AutoDoS to bypass existing defenses more effectively. Experimental results show that AutoDoS significantly amplifies service response latency by over 250$\times\uparrow$, leading to severe resource consumption in terms of GPU utilization and memory usage. Our work provides a new perspective on LLM-DoS attacks and security defenses. Our code is available at https://github.com/shuita2333/AutoDoS.

摘要: 大型语言模型(LLM)在各种任务中表现出了卓越的性能，但仍然容易受到外部威胁，特别是LLm拒绝服务(LLm-DoS)攻击。具体地说，LLM-DoS攻击旨在耗尽计算资源并阻止服务。然而，现有的研究主要集中在白盒攻击上，而对黑盒攻击的研究还不够深入。本文介绍了一种针对黑盒LLM-DoS的自动生成算法--AutoDoS攻击。AutoDoS构建DoS攻击树，扩展节点覆盖率，以达到黑箱条件下的攻击效果。通过可转移性驱动的迭代优化，AutoDoS可以在一个提示下跨不同的模型工作。此外，我们揭示，嵌入长度特洛伊木马允许AutoDoS更有效地绕过现有防御。实验结果表明，AutoDoS显著放大了服务响应延迟250倍以上，导致GPU使用率和内存使用率严重消耗资源。我们的工作为LLM-DoS攻击和安全防御提供了一个新的视角。我们的代码可以在https://github.com/shuita2333/AutoDoS.上找到



## **30. Enhancing the Capability and Robustness of Large Language Models through Reinforcement Learning-Driven Query Refinement**

通过强化学习驱动的查询细化增强大型语言模型的能力和鲁棒性 cs.CL

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2407.01461v2) [paper-pdf](http://arxiv.org/pdf/2407.01461v2)

**Authors**: Zisu Huang, Xiaohua Wang, Feiran Zhang, Zhibo Xu, Cenyuan Zhang, Qi Qian, Xiaoqing Zheng, Xuanjing Huang

**Abstract**: The capacity of large language models (LLMs) to generate honest, harmless, and helpful responses heavily relies on the quality of user prompts. However, these prompts often tend to be brief and vague, thereby significantly limiting the full potential of LLMs. Moreover, harmful prompts can be meticulously crafted and manipulated by adversaries to jailbreak LLMs, inducing them to produce potentially toxic content. To enhance the capabilities of LLMs while maintaining strong robustness against harmful jailbreak inputs, this study proposes a transferable and pluggable framework that refines user prompts before they are input into LLMs. This strategy improves the quality of the queries, empowering LLMs to generate more truthful, benign and useful responses. Specifically, a lightweight query refinement model is introduced and trained using a specially designed reinforcement learning approach that incorporates multiple objectives to enhance particular capabilities of LLMs. Extensive experiments demonstrate that the refinement model not only improves the quality of responses but also strengthens their robustness against jailbreak attacks. Code is available at: https://github.com/Huangzisu/query-refinement .

摘要: 大型语言模型(LLM)生成诚实、无害和有用的响应的能力在很大程度上取决于用户提示的质量。然而，这些提示往往简短而含糊，从而极大地限制了LLM的全部潜力。此外，有害的提示可以被对手精心制作和操纵，以越狱LLM，诱导它们产生潜在的有毒内容。为了增强LLMS的能力，同时保持对有害越狱输入的强大健壮性，本研究提出了一个可移植和可插拔的框架，在将用户提示输入到LLMS之前对其进行提炼。这一策略提高了查询的质量，使LLMS能够生成更真实、良性和有用的响应。具体地说，引入了一种轻量级查询精化模型，并使用专门设计的强化学习方法进行训练，该方法结合了多个目标来增强LLMS的特定能力。大量实验表明，改进模型不仅提高了响应的质量，而且增强了对越狱攻击的健壮性。代码可从以下网址获得：https://github.com/Huangzisu/query-refinement。



## **31. SEA: Low-Resource Safety Alignment for Multimodal Large Language Models via Synthetic Embeddings**

SEA：通过合成嵌入实现多模式大型语言模型的低资源安全性对齐 cs.CL

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.12562v1) [paper-pdf](http://arxiv.org/pdf/2502.12562v1)

**Authors**: Weikai Lu, Hao Peng, Huiping Zhuang, Cen Chen, Ziqian Zeng

**Abstract**: Multimodal Large Language Models (MLLMs) have serious security vulnerabilities.While safety alignment using multimodal datasets consisting of text and data of additional modalities can effectively enhance MLLM's security, it is costly to construct these datasets. Existing low-resource security alignment methods, including textual alignment, have been found to struggle with the security risks posed by additional modalities. To address this, we propose Synthetic Embedding augmented safety Alignment (SEA), which optimizes embeddings of additional modality through gradient updates to expand textual datasets. This enables multimodal safety alignment training even when only textual data is available. Extensive experiments on image, video, and audio-based MLLMs demonstrate that SEA can synthesize a high-quality embedding on a single RTX3090 GPU within 24 seconds. SEA significantly improves the security of MLLMs when faced with threats from additional modalities. To assess the security risks introduced by video and audio, we also introduced a new benchmark called VA-SafetyBench. High attack success rates across multiple MLLMs validate its challenge. Our code and data will be available at https://github.com/ZeroNLP/SEA.

摘要: 多通道大语言模型存在严重的安全漏洞，使用由附加通道的文本和数据组成的多通道数据集进行安全对齐可以有效地增强多通道大语言模型的安全性，但构建这些数据集的代价很高。现有的低资源安全对齐方法，包括文本对齐，已被发现难以应对额外方式带来的安全风险。为了解决这一问题，我们提出了合成嵌入增强安全对齐(SEA)，它通过梯度更新来扩展文本数据集来优化附加通道的嵌入。这使得即使在只有文本数据可用的情况下也能进行多模式安全调整培训。在基于图像、视频和音频的MLLMS上的大量实验表明，SEA可以在24秒内在单个RTX3090 GPU上合成高质量的嵌入。SEA大大提高了大规模毁灭性武器在面临来自其他模式的威胁时的安全性。为了评估视频和音频带来的安全风险，我们还引入了一个名为VA-SafetyB边的新基准。跨多个MLLM的高攻击成功率验证了其挑战。我们的代码和数据将在https://github.com/ZeroNLP/SEA.上提供



## **32. Data to Defense: The Role of Curation in Customizing LLMs Against Jailbreaking Attacks**

数据到防御：治愈在定制LLM以防止越狱攻击中的作用 cs.CR

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2410.02220v4) [paper-pdf](http://arxiv.org/pdf/2410.02220v4)

**Authors**: Xiaoqun Liu, Jiacheng Liang, Luoxi Tang, Muchao Ye, Weicheng Ma, Zhaohan Xi

**Abstract**: Large language models (LLMs) are widely adapted for downstream applications through fine-tuning, a process named customization. However, recent studies have identified a vulnerability during this process, where malicious samples can compromise the robustness of LLMs and amplify harmful behaviors-an attack commonly referred to as jailbreaking. To address this challenge, we propose an adaptive data curation approach allowing any text to be curated to enhance its effectiveness in counteracting harmful samples during customization. To avoid the need for additional defensive modules, we further introduce a comprehensive mitigation framework spanning the lifecycle of the customization process: before customization to immunize LLMs against future jailbreak attempts, during customization to neutralize risks, and after customization to restore compromised models. Experimental results demonstrate a significant reduction in jailbreaking effects, achieving up to a 100% success rate in generating safe responses. By combining adaptive data curation with lifecycle-based mitigation strategies, this work represents a solid step forward in mitigating jailbreaking risks and ensuring the secure adaptation of LLMs.

摘要: 大型语言模型(LLM)通过微调被广泛地适应于下游应用，这一过程被称为定制。然而，最近的研究发现了这一过程中的一个漏洞，其中恶意样本可能会损害LLM的健壮性并放大有害行为--这种攻击通常被称为越狱。为了应对这一挑战，我们提出了一种自适应的数据管理方法，允许对任何文本进行管理，以增强其在定制期间对抗有害样本的有效性。为了避免需要额外的防御模块，我们进一步引入了一个全面的缓解框架，跨越定制过程的生命周期：在定制之前，以使LLM免受未来越狱企图的影响；在定制期间，以消除风险；以及在定制之后，以恢复受影响的模型。实验结果表明，越狱效果显著降低，生成安全响应的成功率高达100%。通过将自适应数据管理与基于生命周期的缓解策略相结合，这项工作代表着在降低越狱风险和确保小岛屿发展中国家安全适应方面迈出了坚实的一步。



## **33. Towards Robust and Secure Embodied AI: A Survey on Vulnerabilities and Attacks**

迈向强大和安全的人工智能：关于漏洞和攻击的调查 cs.CR

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.13175v1) [paper-pdf](http://arxiv.org/pdf/2502.13175v1)

**Authors**: Wenpeng Xing, Minghao Li, Mohan Li, Meng Han

**Abstract**: Embodied AI systems, including robots and autonomous vehicles, are increasingly integrated into real-world applications, where they encounter a range of vulnerabilities stemming from both environmental and system-level factors. These vulnerabilities manifest through sensor spoofing, adversarial attacks, and failures in task and motion planning, posing significant challenges to robustness and safety. Despite the growing body of research, existing reviews rarely focus specifically on the unique safety and security challenges of embodied AI systems. Most prior work either addresses general AI vulnerabilities or focuses on isolated aspects, lacking a dedicated and unified framework tailored to embodied AI. This survey fills this critical gap by: (1) categorizing vulnerabilities specific to embodied AI into exogenous (e.g., physical attacks, cybersecurity threats) and endogenous (e.g., sensor failures, software flaws) origins; (2) systematically analyzing adversarial attack paradigms unique to embodied AI, with a focus on their impact on perception, decision-making, and embodied interaction; (3) investigating attack vectors targeting large vision-language models (LVLMs) and large language models (LLMs) within embodied systems, such as jailbreak attacks and instruction misinterpretation; (4) evaluating robustness challenges in algorithms for embodied perception, decision-making, and task planning; and (5) proposing targeted strategies to enhance the safety and reliability of embodied AI systems. By integrating these dimensions, we provide a comprehensive framework for understanding the interplay between vulnerabilities and safety in embodied AI.

摘要: 包括机器人和自动驾驶车辆在内的具体化人工智能系统正越来越多地融入现实世界的应用程序，在这些应用程序中，它们遇到了一系列源于环境和系统层面因素的漏洞。这些漏洞表现为传感器欺骗、对抗性攻击以及任务和运动规划中的失败，对健壮性和安全性构成了重大挑战。尽管研究的主体越来越多，但现有的审查很少专门关注嵌入式人工智能系统的独特安全和安保挑战。大多数以前的工作要么解决了一般的人工智能漏洞，要么专注于孤立的方面，缺乏一个专门为体现的人工智能量身定做的统一框架。本调查通过以下方式填补这一关键空白：(1)将特定于具身人工智能的漏洞分为外源性(如物理攻击、网络安全威胁)和内源性(如传感器故障、软件缺陷)来源；(2)系统分析具身人工智能特有的对抗性攻击范式，重点关注它们对感知、决策和具身交互的影响；(3)调查针对具身系统内的大视觉语言模型(LVLM)和大语言模型(LMS)的攻击向量，如越狱攻击和指令曲解；(4)评估体现感知、决策和任务规划算法中的健壮性挑战；(5)提出有针对性的策略，以提高体现人工智能系统的安全性和可靠性。通过整合这些维度，我们提供了一个全面的框架，用于理解体现的人工智能中漏洞和安全之间的相互作用。



## **34. SoK: Understanding Vulnerabilities in the Large Language Model Supply Chain**

SoK：了解大型语言模型供应链中的漏洞 cs.CR

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.12497v1) [paper-pdf](http://arxiv.org/pdf/2502.12497v1)

**Authors**: Shenao Wang, Yanjie Zhao, Zhao Liu, Quanchen Zou, Haoyu Wang

**Abstract**: Large Language Models (LLMs) transform artificial intelligence, driving advancements in natural language understanding, text generation, and autonomous systems. The increasing complexity of their development and deployment introduces significant security challenges, particularly within the LLM supply chain. However, existing research primarily focuses on content safety, such as adversarial attacks, jailbreaking, and backdoor attacks, while overlooking security vulnerabilities in the underlying software systems. To address this gap, this study systematically analyzes 529 vulnerabilities reported across 75 prominent projects spanning 13 lifecycle stages. The findings show that vulnerabilities are concentrated in the application (50.3%) and model (42.7%) layers, with improper resource control (45.7%) and improper neutralization (25.1%) identified as the leading root causes. Additionally, while 56.7% of the vulnerabilities have available fixes, 8% of these patches are ineffective, resulting in recurring vulnerabilities. This study underscores the challenges of securing the LLM ecosystem and provides actionable insights to guide future research and mitigation strategies.

摘要: 大型语言模型(LLM)改变了人工智能，推动了自然语言理解、文本生成和自主系统的进步。它们的开发和部署日益复杂，带来了重大的安全挑战，特别是在LLM供应链中。然而，现有的研究主要集中在内容安全上，如对抗性攻击、越狱和后门攻击，而忽略了底层软件系统中的安全漏洞。为了解决这一差距，该研究系统地分析了跨越13个生命周期阶段的75个重要项目中报告的529个漏洞。结果显示，漏洞集中在应用层(50.3%)和模型层(42.7%)，其中资源控制不当(45.7%)和中和不当(25.1%)是主要的根本原因。此外，虽然56.7%的漏洞有可用的修复程序，但其中8%的补丁程序无效，导致漏洞反复出现。这项研究强调了确保LLM生态系统安全的挑战，并提供了可操作的见解，以指导未来的研究和缓解战略。



## **35. SafeDialBench: A Fine-Grained Safety Benchmark for Large Language Models in Multi-Turn Dialogues with Diverse Jailbreak Attacks**

SafeDialBench：针对具有多样越狱攻击的多回合对话中大型语言模型的细粒度安全基准 cs.CL

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.11090v2) [paper-pdf](http://arxiv.org/pdf/2502.11090v2)

**Authors**: Hongye Cao, Yanming Wang, Sijia Jing, Ziyue Peng, Zhixin Bai, Zhe Cao, Meng Fang, Fan Feng, Boyan Wang, Jiaheng Liu, Tianpei Yang, Jing Huo, Yang Gao, Fanyu Meng, Xi Yang, Chao Deng, Junlan Feng

**Abstract**: With the rapid advancement of Large Language Models (LLMs), the safety of LLMs has been a critical concern requiring precise assessment. Current benchmarks primarily concentrate on single-turn dialogues or a single jailbreak attack method to assess the safety. Additionally, these benchmarks have not taken into account the LLM's capability of identifying and handling unsafe information in detail. To address these issues, we propose a fine-grained benchmark SafeDialBench for evaluating the safety of LLMs across various jailbreak attacks in multi-turn dialogues. Specifically, we design a two-tier hierarchical safety taxonomy that considers 6 safety dimensions and generates more than 4000 multi-turn dialogues in both Chinese and English under 22 dialogue scenarios. We employ 7 jailbreak attack strategies, such as reference attack and purpose reverse, to enhance the dataset quality for dialogue generation. Notably, we construct an innovative assessment framework of LLMs, measuring capabilities in detecting, and handling unsafe information and maintaining consistency when facing jailbreak attacks. Experimental results across 17 LLMs reveal that Yi-34B-Chat and GLM4-9B-Chat demonstrate superior safety performance, while Llama3.1-8B-Instruct and o3-mini exhibit safety vulnerabilities.

摘要: 随着大型语言模型的快速发展，大型语言模型的安全性已经成为一个需要精确评估的关键问题。目前的基准主要集中在单轮对话或单一越狱攻击方法上，以评估安全性。此外，这些基准没有考虑到LLM详细识别和处理不安全信息的能力。为了解决这些问题，我们提出了一个细粒度的基准SafeDialB边，用于在多轮对话中评估LLMS在各种越狱攻击中的安全性。具体地说，我们设计了一个考虑了6个安全维度的两级层次安全分类，并在22个对话场景下生成了4000多个中英文多轮对话。我们使用了引用攻击、目的反转等7种越狱攻击策略，以提高对话生成的数据集质量。值得注意的是，我们构建了一个创新的LLMS评估框架，衡量了检测和处理不安全信息的能力，并在面临越狱攻击时保持一致性。在17个LLM上的实验结果表明，YI-34B-Chat和GLM4-9B-Chat具有优越的安全性能，而Llama3.1-8B-Indict和O_3-mini存在安全漏洞。



## **36. StructuralSleight: Automated Jailbreak Attacks on Large Language Models Utilizing Uncommon Text-Organization Structures**

StructuralSleight：利用不常见的文本组织结构对大型语言模型进行自动越狱攻击 cs.CL

15 pages, 7 figures

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2406.08754v3) [paper-pdf](http://arxiv.org/pdf/2406.08754v3)

**Authors**: Bangxin Li, Hengrui Xing, Cong Tian, Chao Huang, Jin Qian, Huangqing Xiao, Linfeng Feng

**Abstract**: Large Language Models (LLMs) are widely used in natural language processing but face the risk of jailbreak attacks that maliciously induce them to generate harmful content. Existing jailbreak attacks, including character-level and context-level attacks, mainly focus on the prompt of plain text without specifically exploring the significant influence of its structure. In this paper, we focus on studying how the prompt structure contributes to the jailbreak attack. We introduce a novel structure-level attack method based on long-tailed structures, which we refer to as Uncommon Text-Organization Structures (UTOS). We extensively study 12 UTOS templates and 6 obfuscation methods to build an effective automated jailbreak tool named StructuralSleight that contains three escalating attack strategies: Structural Attack, Structural and Character/Context Obfuscation Attack, and Fully Obfuscated Structural Attack. Extensive experiments on existing LLMs show that StructuralSleight significantly outperforms the baseline methods. In particular, the attack success rate reaches 94.62\% on GPT-4o, which has not been addressed by state-of-the-art techniques.

摘要: 大语言模型在自然语言处理中被广泛使用，但面临着越狱攻击的风险，这些攻击会恶意诱导它们生成有害内容。现有的越狱攻击，包括字符级攻击和语境级攻击，主要集中在明文提示上，没有具体探讨其结构的重大影响。本文主要研究提示结构在越狱攻击中的作用。提出了一种新的基于长尾结构的结构级攻击方法，称为非常见文本组织结构(UTOS)。我们深入研究了12个UTOS模板和6种混淆方法，构建了一个有效的自动化越狱工具StructuralSleight，该工具包含三种逐步升级的攻击策略：结构攻击、结构和字符/上下文混淆攻击和完全混淆结构攻击。在现有LLMS上的大量实验表明，StructuralSleight的性能明显优于基线方法。特别是，在GPT-40上的攻击成功率达到了94.62\%，这是最新技术还没有解决的问题。



## **37. Unveiling Privacy Risks in LLM Agent Memory**

揭露LLM代理内存中的隐私风险 cs.CR

Under review

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.13172v1) [paper-pdf](http://arxiv.org/pdf/2502.13172v1)

**Authors**: Bo Wang, Weiyi He, Pengfei He, Shenglai Zeng, Zhen Xiang, Yue Xing, Jiliang Tang

**Abstract**: Large Language Model (LLM) agents have become increasingly prevalent across various real-world applications. They enhance decision-making by storing private user-agent interactions in the memory module for demonstrations, introducing new privacy risks for LLM agents. In this work, we systematically investigate the vulnerability of LLM agents to our proposed Memory EXTRaction Attack (MEXTRA) under a black-box setting. To extract private information from memory, we propose an effective attacking prompt design and an automated prompt generation method based on different levels of knowledge about the LLM agent. Experiments on two representative agents demonstrate the effectiveness of MEXTRA. Moreover, we explore key factors influencing memory leakage from both the agent's and the attacker's perspectives. Our findings highlight the urgent need for effective memory safeguards in LLM agent design and deployment.

摘要: 大型语言模型（LLM）代理在各种现实世界的应用程序中变得越来越普遍。它们通过将私人用户-代理交互存储在内存模块中以进行演示来增强决策，从而为LLM代理带来新的隐私风险。在这项工作中，我们系统地研究了LLM代理在黑匣子环境下对我们提出的内存EXTRaction攻击（MEXTRA）的脆弱性。为了从内存中提取私人信息，我们提出了一种有效的攻击提示设计和基于LLM代理不同知识水平的自动提示生成方法。对两个代表性代理的实验证明了MEXTRA的有效性。此外，我们还从代理和攻击者的角度探讨了影响内存泄漏的关键因素。我们的研究结果凸显了LLM代理设计和部署中迫切需要有效的内存保护措施。



## **38. FedEAT: A Robustness Optimization Framework for Federated LLMs**

FedEAT：联邦LLM的稳健性优化框架 cs.LG

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11863v1) [paper-pdf](http://arxiv.org/pdf/2502.11863v1)

**Authors**: Yahao Pang, Xingyuan Wu, Xiaojin Zhang, Wei Chen, Hai Jin

**Abstract**: Significant advancements have been made by Large Language Models (LLMs) in the domains of natural language understanding and automated content creation. However, they still face persistent problems, including substantial computational costs and inadequate availability of training data. The combination of Federated Learning (FL) and LLMs (federated LLMs) offers a solution by leveraging distributed data while protecting privacy, which positions it as an ideal choice for sensitive domains. However, Federated LLMs still suffer from robustness challenges, including data heterogeneity, malicious clients, and adversarial attacks, which greatly hinder their applications. We first introduce the robustness problems in federated LLMs, to address these challenges, we propose FedEAT (Federated Embedding space Adversarial Training), a novel framework that applies adversarial training in the embedding space of client LLM and employs a robust aggregation approach, specifically geometric median aggregation, to enhance the robustness of Federated LLMs. Our experiments demonstrate that FedEAT effectively improves the robustness of Federated LLMs with minimal performance loss.

摘要: 大型语言模型(LLM)在自然语言理解和自动内容创建领域取得了重大进展。然而，它们仍然面临着长期存在的问题，包括巨大的计算成本和培训数据的不足。联合学习(FL)和联合LLMS(联合LLMS)的结合提供了一种在保护隐私的同时利用分布式数据的解决方案，这将其定位为敏感领域的理想选择。然而，联邦LLMS仍然面临着健壮性挑战，包括数据异构性、恶意客户端和敌意攻击，这些都极大地阻碍了它们的应用。首先介绍了联合LLMS的健壮性问题，针对这些问题，我们提出了一种新的框架FedEAT(Federated Embedding Space Adversal Trading)，该框架将对抗性训练应用于客户端LLMS的嵌入空间，并采用一种稳健的聚集方法，特别是几何中值聚集来增强联合LLMS的健壮性。实验结果表明，FedEAT算法以最小的性能损失有效地提高了联邦LLMS的健壮性。



## **39. StructTransform: A Scalable Attack Surface for Safety-Aligned Large Language Models**

StructChange：安全一致的大型语言模型的可扩展攻击表面 cs.LG

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11853v1) [paper-pdf](http://arxiv.org/pdf/2502.11853v1)

**Authors**: Shehel Yoosuf, Temoor Ali, Ahmed Lekssays, Mashael AlSabah, Issa Khalil

**Abstract**: In this work, we present a series of structure transformation attacks on LLM alignment, where we encode natural language intent using diverse syntax spaces, ranging from simple structure formats and basic query languages (e.g. SQL) to new novel spaces and syntaxes created entirely by LLMs. Our extensive evaluation shows that our simplest attacks can achieve close to 90% success rate, even on strict LLMs (such as Claude 3.5 Sonnet) using SOTA alignment mechanisms. We improve the attack performance further by using an adaptive scheme that combines structure transformations along with existing \textit{content transformations}, resulting in over 96% ASR with 0% refusals.   To generalize our attacks, we explore numerous structure formats, including syntaxes purely generated by LLMs. Our results indicate that such novel syntaxes are easy to generate and result in a high ASR, suggesting that defending against our attacks is not a straightforward process. Finally, we develop a benchmark and evaluate existing safety-alignment defenses against it, showing that most of them fail with 100% ASR. Our results show that existing safety alignment mostly relies on token-level patterns without recognizing harmful concepts, highlighting and motivating the need for serious research efforts in this direction. As a case study, we demonstrate how attackers can use our attack to easily generate a sample malware, and a corpus of fraudulent SMS messages, which perform well in bypassing detection.

摘要: 在这项工作中，我们提出了一系列针对LLM对齐的结构转换攻击，其中我们使用不同的语法空间来编码自然语言意图，从简单的结构格式和基本查询语言(例如SQL)到完全由LLMS创建的新的新颖空间和句法。我们广泛的评估表明，我们最简单的攻击可以达到接近90%的成功率，即使是在使用SOTA对齐机制的严格LLM(如Claude 3.5十四行诗)上也是如此。通过使用结构变换和现有的文本内容变换相结合的自适应方案，进一步提高了攻击性能，获得了96%以上的ASR和0%的拒绝。为了概括我们的攻击，我们探索了许多结构格式，包括纯粹由LLMS生成的语法。我们的结果表明，这种新的语法很容易生成，并导致高ASR，这表明防御我们的攻击并不是一个简单的过程。最后，我们开发了一个基准，并对现有的安全对齐防御进行了评估，结果表明，大多数安全对齐防御都是100%ASR失败的。我们的结果表明，现有的安全对齐主要依赖于令牌级模式，而没有识别有害的概念，这突显和激励了在这一方向上认真研究的必要性。作为一个案例研究，我们演示了攻击者如何利用我们的攻击轻松生成样本恶意软件和欺诈性短信语料库，它们在绕过检测方面表现良好。



## **40. Separate the Wheat from the Chaff: A Post-Hoc Approach to Safety Re-Alignment for Fine-Tuned Language Models**

将小麦与谷壳分开：精调语言模型的安全重新对齐的事后方法 cs.CL

16 pages, 14 figures,

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2412.11041v2) [paper-pdf](http://arxiv.org/pdf/2412.11041v2)

**Authors**: Di Wu, Xin Lu, Yanyan Zhao, Bing Qin

**Abstract**: Although large language models (LLMs) achieve effective safety alignment at the time of release, they still face various safety challenges. A key issue is that fine-tuning often compromises the safety alignment of LLMs. To address this issue, we propose a method named IRR (Identify, Remove, and Recalibrate for Safety Realignment) that performs safety realignment for LLMs. The core of IRR is to identify and remove unsafe delta parameters from the fine-tuned models, while recalibrating the retained ones. We evaluate the effectiveness of IRR across various datasets, including both full fine-tuning and LoRA methods. Our results demonstrate that IRR significantly enhances the safety performance of fine-tuned models on safety benchmarks, such as harmful queries and jailbreak attacks, while maintaining their performance on downstream tasks. The source code is available at: https://anonymous.4open.science/r/IRR-BD4F.

摘要: 尽管大型语言模型（LLM）在发布时实现了有效的安全一致，但它们仍然面临各种安全挑战。一个关键问题是微调通常会损害LLM的安全对齐。为了解决这个问题，我们提出了一种名为IRR（识别、删除和重新校准以实现安全重新对准）的方法，该方法为LLM执行安全重新对准。IRR的核心是从微调模型中识别并删除不安全的Delta参数，同时重新校准保留的参数。我们评估IRR在各种数据集的有效性，包括完全微调和LoRA方法。我们的结果表明，IRR显着增强了经过微调的模型在安全基准（例如有害查询和越狱攻击）上的安全性能，同时保持了其在下游任务上的性能。源代码可访问：https://anonymous.4open.science/r/IRR-BD4F。



## **41. DELMAN: Dynamic Defense Against Large Language Model Jailbreaking with Model Editing**

德尔曼：通过模型编辑对大型语言模型越狱的动态防御 cs.CR

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11647v1) [paper-pdf](http://arxiv.org/pdf/2502.11647v1)

**Authors**: Yi Wang, Fenghua Weng, Sibei Yang, Zhan Qin, Minlie Huang, Wenjie Wang

**Abstract**: Large Language Models (LLMs) are widely applied in decision making, but their deployment is threatened by jailbreak attacks, where adversarial users manipulate model behavior to bypass safety measures. Existing defense mechanisms, such as safety fine-tuning and model editing, either require extensive parameter modifications or lack precision, leading to performance degradation on general tasks, which is unsuitable to post-deployment safety alignment. To address these challenges, we propose DELMAN (Dynamic Editing for LLMs JAilbreak DefeNse), a novel approach leveraging direct model editing for precise, dynamic protection against jailbreak attacks. DELMAN directly updates a minimal set of relevant parameters to neutralize harmful behaviors while preserving the model's utility. To avoid triggering a safe response in benign context, we incorporate KL-divergence regularization to ensure the updated model remains consistent with the original model when processing benign queries. Experimental results demonstrate that DELMAN outperforms baseline methods in mitigating jailbreak attacks while preserving the model's utility, and adapts seamlessly to new attack instances, providing a practical and efficient solution for post-deployment model protection.

摘要: 大语言模型(LLM)在决策中被广泛应用，但它们的部署受到越狱攻击的威胁，在越狱攻击中，敌对用户操纵模型行为以绕过安全措施。现有的防御机制，如安全微调和模型编辑，要么需要大量修改参数，要么缺乏精度，导致一般任务的性能下降，不适合部署后的安全对齐。为了应对这些挑战，我们提出了Delman(用于LLMS越狱防御的动态编辑)，这是一种利用直接模型编辑来精确、动态地防御越狱攻击的新方法。Delman直接更新相关参数的最小集合，以中和有害行为，同时保持模型的实用性。为了避免在良性环境下触发安全响应，我们引入了KL-散度正则化，以确保在处理良性查询时更新后的模型与原始模型保持一致。实验结果表明，Delman在保持模型实用性的同时，在缓解越狱攻击方面优于基准方法，并能无缝适应新的攻击实例，为部署后模型防护提供了一种实用而高效的解决方案。



## **42. Can LLM Watermarks Robustly Prevent Unauthorized Knowledge Distillation?**

LLM水印能否强大地防止未经授权的知识提炼？ cs.CL

22 pages, 12 figures, 13 tables

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11598v1) [paper-pdf](http://arxiv.org/pdf/2502.11598v1)

**Authors**: Leyi Pan, Aiwei Liu, Shiyu Huang, Yijian Lu, Xuming Hu, Lijie Wen, Irwin King, Philip S. Yu

**Abstract**: The radioactive nature of Large Language Model (LLM) watermarking enables the detection of watermarks inherited by student models when trained on the outputs of watermarked teacher models, making it a promising tool for preventing unauthorized knowledge distillation. However, the robustness of watermark radioactivity against adversarial actors remains largely unexplored. In this paper, we investigate whether student models can acquire the capabilities of teacher models through knowledge distillation while avoiding watermark inheritance. We propose two categories of watermark removal approaches: pre-distillation removal through untargeted and targeted training data paraphrasing (UP and TP), and post-distillation removal through inference-time watermark neutralization (WN). Extensive experiments across multiple model pairs, watermarking schemes and hyper-parameter settings demonstrate that both TP and WN thoroughly eliminate inherited watermarks, with WN achieving this while maintaining knowledge transfer efficiency and low computational overhead. Given the ongoing deployment of watermarking techniques in production LLMs, these findings emphasize the urgent need for more robust defense strategies. Our code is available at https://github.com/THU-BPM/Watermark-Radioactivity-Attack.

摘要: 大型语言模型(LLM)水印的放射性特性使其能够在对带水印的教师模型的输出进行训练时检测由学生模型继承的水印，使其成为防止未经授权的知识蒸馏的一种有前途的工具。然而，水印放射性对敌方行为的稳健性在很大程度上仍未被探索。本文研究了学生模型能否在避免水印继承的同时，通过知识提炼获得教师模型的能力。我们提出了两类水印去除方法：通过非目标和目标训练数据释义(UP和TP)进行蒸馏前去除和通过推理时间水印中和(WN)进行蒸馏后去除。在多个模型对、水印方案和超参数设置上的大量实验表明，TP和WN都彻底消除了继承的水印，WN在保持知识传递效率和较低的计算开销的同时实现了这一点。鉴于水印技术在生产LLM中的持续部署，这些发现强调了对更强大的防御策略的迫切需要。我们的代码可以在https://github.com/THU-BPM/Watermark-Radioactivity-Attack.上找到



## **43. LLMs can be Dangerous Reasoners: Analyzing-based Jailbreak Attack on Large Language Models**

LLM可能是危险的推理者：基于分析的对大型语言模型的越狱攻击 cs.CR

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2407.16205v4) [paper-pdf](http://arxiv.org/pdf/2407.16205v4)

**Authors**: Shi Lin, Hongming Yang, Rongchang Li, Xun Wang, Changting Lin, Wenpeng Xing, Meng Han

**Abstract**: The rapid development of Large Language Models (LLMs) has brought significant advancements across various tasks. However, despite these achievements, LLMs still exhibit inherent safety vulnerabilities, especially when confronted with jailbreak attacks. Existing jailbreak methods suffer from two main limitations: reliance on complicated prompt engineering and iterative optimization, which lead to low attack success rate (ASR) and attack efficiency (AE). In this work, we propose an efficient jailbreak attack method, Analyzing-based Jailbreak (ABJ), which leverages the advanced reasoning capability of LLMs to autonomously generate harmful content, revealing their underlying safety vulnerabilities during complex reasoning process. We conduct comprehensive experiments on ABJ across various open-source and closed-source LLMs. In particular, ABJ achieves high ASR (82.1% on GPT-4o-2024-11-20) with exceptional AE among all target LLMs, showcasing its remarkable attack effectiveness, transferability, and efficiency. Our findings underscore the urgent need to prioritize and improve the safety of LLMs to mitigate the risks of misuse.

摘要: 大型语言模型(LLM)的快速发展带来了跨各种任务的重大进步。然而，尽管取得了这些成就，LLMS仍然表现出固有的安全漏洞，特别是在面临越狱攻击时。现有的越狱方法存在两个主要缺陷：依赖复杂的快速工程和迭代优化，导致攻击成功率和攻击效率较低。在这项工作中，我们提出了一种高效的越狱攻击方法-基于分析的越狱(ABJ)，它利用LLMS的高级推理能力自主生成有害内容，在复杂的推理过程中揭示其潜在的安全漏洞。我们在各种开源和闭源的LLM上对ABJ进行了全面的实验。特别是，ABJ在所有目标LLM中获得了高ASR(在GPT-40-2024-11-20上为82.1%)，并具有出色的AE，显示了其卓越的攻击效能、可转移性和效率。我们的研究结果强调迫切需要优先考虑和改善低密度脂蛋白的安全性，以减少误用的风险。



## **44. Be Cautious When Merging Unfamiliar LLMs: A Phishing Model Capable of Stealing Privacy**

合并不熟悉的LLM时要谨慎：一种能够窃取隐私的网络钓鱼模式 cs.CL

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11533v1) [paper-pdf](http://arxiv.org/pdf/2502.11533v1)

**Authors**: Zhenyuan Guo, Yi Shi, Wenlong Meng, Chen Gong, Chengkun Wei, Wenzhi Chen

**Abstract**: Model merging is a widespread technology in large language models (LLMs) that integrates multiple task-specific LLMs into a unified one, enabling the merged model to inherit the specialized capabilities of these LLMs. Most task-specific LLMs are sourced from open-source communities and have not undergone rigorous auditing, potentially imposing risks in model merging. This paper highlights an overlooked privacy risk: \textit{an unsafe model could compromise the privacy of other LLMs involved in the model merging.} Specifically, we propose PhiMM, a privacy attack approach that trains a phishing model capable of stealing privacy using a crafted privacy phishing instruction dataset. Furthermore, we introduce a novel model cloaking method that mimics a specialized capability to conceal attack intent, luring users into merging the phishing model. Once victims merge the phishing model, the attacker can extract personally identifiable information (PII) or infer membership information (MI) by querying the merged model with the phishing instruction. Experimental results show that merging a phishing model increases the risk of privacy breaches. Compared to the results before merging, PII leakage increased by 3.9\% and MI leakage increased by 17.4\% on average. We release the code of PhiMM through a link.

摘要: 模型合并是大型语言模型中的一种广泛使用的技术，它将多个特定于任务的大型语言模型集成到一个统一的大型语言模型中，使合并后的模型能够继承这些大型语言模型的专门功能。大多数特定于任务的LLM来自开源社区，没有经过严格的审计，这可能会给模型合并带来风险。本文强调了一个被忽视的隐私风险：\textit{一个不安全的模型可能会危及模型合并中涉及的其他LLM的隐私。}具体地说，我们提出了PhiMM，这是一种隐私攻击方法，它训练一个能够使用特制的隐私钓鱼指令数据集窃取隐私的钓鱼模型。此外，我们还提出了一种新的模型伪装方法，该方法模仿了一种特殊的隐藏攻击意图的能力，引诱用户合并钓鱼模型。一旦受害者合并钓鱼模型，攻击者就可以通过使用钓鱼指令查询合并的模型来提取个人身份信息(PII)或推断成员信息(MI)。实验结果表明，合并钓鱼模型会增加隐私泄露的风险。与合并前相比，PII渗漏平均增加3.9%，MI渗漏平均增加17.4%。我们通过一个链接发布PhiMM的代码。



## **45. DeFiScope: Detecting Various DeFi Price Manipulations with LLM Reasoning**

DeFiScope：通过LLM推理检测各种DeFi价格操纵 cs.CR

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11521v1) [paper-pdf](http://arxiv.org/pdf/2502.11521v1)

**Authors**: Juantao Zhong, Daoyuan Wu, Ye Liu, Maoyi Xie, Yang Liu, Yi Li, Ning Liu

**Abstract**: DeFi (Decentralized Finance) is one of the most important applications of today's cryptocurrencies and smart contracts. It manages hundreds of billions in Total Value Locked (TVL) on-chain, yet it remains susceptible to common DeFi price manipulation attacks. Despite state-of-the-art (SOTA) systems like DeFiRanger and DeFort, we found that they are less effective to non-standard price models in custom DeFi protocols, which account for 44.2% of the 95 DeFi price manipulation attacks reported over the past three years.   In this paper, we introduce the first LLM-based approach, DeFiScope, for detecting DeFi price manipulation attacks in both standard and custom price models. Our insight is that large language models (LLMs) have certain intelligence to abstract price calculation from code and infer the trend of token price changes based on the extracted price models. To further strengthen LLMs in this aspect, we leverage Foundry to synthesize on-chain data and use it to fine-tune a DeFi price-specific LLM. Together with the high-level DeFi operations recovered from low-level transaction data, DeFiScope detects various DeFi price manipulations according to systematically mined patterns. Experimental results show that DeFiScope achieves a high precision of 96% and a recall rate of 80%, significantly outperforming SOTA approaches. Moreover, we evaluate DeFiScope's cost-effectiveness and demonstrate its practicality by helping our industry partner confirm 147 real-world price manipulation attacks, including discovering 81 previously unknown historical incidents.

摘要: DEFI(去中心化金融)是当今加密货币和智能合约最重要的应用之一。它管理着数千亿美元的总价值锁定(TVL)链上，但它仍然容易受到常见的Defi价格操纵攻击。尽管像DeFiRanger和DeFort这样的最先进的(Sota)系统，我们发现它们对定制Defi协议中的非标准价格模型的有效性较低，在过去三年报告的95起Defi价格操纵攻击中，非标准价格模型占44.2%。在本文中，我们介绍了第一个基于LLM的方法，DeFiScope，用于检测标准价格模型和定制价格模型中的Defi价格操纵攻击。我们的见解是，大型语言模型(LLM)具有一定的智能，能够从代码中抽象出价格计算，并根据提取的价格模型推断令牌价格变化的趋势。为了在这方面进一步加强LLM，我们利用Foundry来合成链上数据，并使用它来微调特定于Defi价格的LLM。与从低级别交易数据中恢复的高级别Defi操作一起，DeFiScope根据系统挖掘的模式检测各种Defi价格操纵。实验结果表明，DeFiScope达到了96%的高准确率和80%的召回率，明显优于SOTA方法。此外，我们评估了DeFiScope的成本效益，并通过帮助我们的行业合作伙伴确认147起真实世界的价格操纵攻击来展示其实用性，其中包括发现81起以前未知的历史事件。



## **46. SmartLLM: Smart Contract Auditing using Custom Generative AI**

SmartLLM：使用自定义生成人工智能的智能合同审计 cs.CR

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.13167v1) [paper-pdf](http://arxiv.org/pdf/2502.13167v1)

**Authors**: Jun Kevin, Pujianto Yugopuspito

**Abstract**: Smart contracts are essential to decentralized finance (DeFi) and blockchain ecosystems but are increasingly vulnerable to exploits due to coding errors and complex attack vectors. Traditional static analysis tools and existing vulnerability detection methods often fail to address these challenges comprehensively, leading to high false-positive rates and an inability to detect dynamic vulnerabilities. This paper introduces SmartLLM, a novel approach leveraging fine-tuned LLaMA 3.1 models with Retrieval-Augmented Generation (RAG) to enhance the accuracy and efficiency of smart contract auditing. By integrating domain-specific knowledge from ERC standards and employing advanced techniques such as QLoRA for efficient fine-tuning, SmartLLM achieves superior performance compared to static analysis tools like Mythril and Slither, as well as zero-shot large language model (LLM) prompting methods such as GPT-3.5 and GPT-4. Experimental results demonstrate a perfect recall of 100% and an accuracy score of 70%, highlighting the model's robustness in identifying vulnerabilities, including reentrancy and access control issues. This research advances smart contract security by offering a scalable and effective auditing solution, supporting the secure adoption of decentralized applications.

摘要: 智能合约对去中心化金融(Defi)和区块链生态系统至关重要，但由于编码错误和复杂的攻击向量，越来越容易受到攻击。传统的静态分析工具和现有的漏洞检测方法往往不能全面地应对这些挑战，导致高误诊率和无法检测动态漏洞。本文介绍了SmartLLM，这是一种新的方法，它利用微调的Llama3.1模型和检索-增强生成(RAG)来提高智能合同审计的准确性和效率。通过集成来自ERC标准的领域特定知识，并使用QLoRA等先进技术进行有效的微调，SmartLLM实现了比Myril和Slither等静态分析工具以及GPT-3.5和GPT-4等零精度大型语言模型(LLM)提示方法更优越的性能。实验结果表明，该模型的查全率为100%，准确率为70%，突出了该模型在识别漏洞(包括可重入性和访问控制问题)方面的健壮性。这项研究通过提供可扩展和有效的审计解决方案，支持安全采用分散的应用程序，从而促进了智能合同安全。



## **47. Adversary-Aware DPO: Enhancing Safety Alignment in Vision Language Models via Adversarial Training**

具有对抗意识的DPO：通过对抗训练增强视觉语言模型中的安全一致性 cs.CR

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11455v1) [paper-pdf](http://arxiv.org/pdf/2502.11455v1)

**Authors**: Fenghua Weng, Jian Lou, Jun Feng, Minlie Huang, Wenjie Wang

**Abstract**: Safety alignment is critical in pre-training large language models (LLMs) to generate responses aligned with human values and refuse harmful queries. Unlike LLM, the current safety alignment of VLMs is often achieved with post-hoc safety fine-tuning. However, these methods are less effective to white-box attacks. To address this, we propose $\textit{Adversary-aware DPO (ADPO)}$, a novel training framework that explicitly considers adversarial. $\textit{Adversary-aware DPO (ADPO)}$ integrates adversarial training into DPO to enhance the safety alignment of VLMs under worst-case adversarial perturbations. $\textit{ADPO}$ introduces two key components: (1) an adversarial-trained reference model that generates human-preferred responses under worst-case perturbations, and (2) an adversarial-aware DPO loss that generates winner-loser pairs accounting for adversarial distortions. By combining these innovations, $\textit{ADPO}$ ensures that VLMs remain robust and reliable even in the presence of sophisticated jailbreak attacks. Extensive experiments demonstrate that $\textit{ADPO}$ outperforms baselines in the safety alignment and general utility of VLMs.

摘要: 在预先训练大型语言模型(LLM)以生成与人类价值观一致的响应并拒绝有害查询时，安全对齐至关重要。与LLM不同，VLM当前的安全对准通常是通过事后安全微调来实现的。然而，这些方法对白盒攻击的有效性较低。为了解决这一问题，我们提出了一种新的训练框架将对抗性训练融入到DPO中，以增强VLM在最坏情况下的对抗性扰动下的安全一致性。$\textit{ADPO}$引入了两个关键组件：(1)对抗性训练的参考模型，它在最坏情况下产生人类偏好的响应；(2)对抗性感知的DPO损失，它产生考虑对抗性扭曲的赢家-输家对。通过将这些创新结合在一起，$\textit{ADPO}$确保即使在存在复杂的越狱攻击的情况下，VLM仍保持健壮和可靠。大量实验表明，在VLMS的安全性、对准和通用性方面，$\textit{ADPO}$都优于Baseline。



## **48. CCJA: Context-Coherent Jailbreak Attack for Aligned Large Language Models**

CCJA：针对对齐大型语言模型的上下文一致越狱攻击 cs.CR

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11379v1) [paper-pdf](http://arxiv.org/pdf/2502.11379v1)

**Authors**: Guanghao Zhou, Panjia Qiu, Mingyuan Fan, Cen Chen, Mingyuan Chu, Xin Zhang, Jun Zhou

**Abstract**: Despite explicit alignment efforts for large language models (LLMs), they can still be exploited to trigger unintended behaviors, a phenomenon known as "jailbreaking." Current jailbreak attack methods mainly focus on discrete prompt manipulations targeting closed-source LLMs, relying on manually crafted prompt templates and persuasion rules. However, as the capabilities of open-source LLMs improve, ensuring their safety becomes increasingly crucial. In such an environment, the accessibility of model parameters and gradient information by potential attackers exacerbates the severity of jailbreak threats. To address this research gap, we propose a novel \underline{C}ontext-\underline{C}oherent \underline{J}ailbreak \underline{A}ttack (CCJA). We define jailbreak attacks as an optimization problem within the embedding space of masked language models. Through combinatorial optimization, we effectively balance the jailbreak attack success rate with semantic coherence. Extensive evaluations show that our method not only maintains semantic consistency but also surpasses state-of-the-art baselines in attack effectiveness. Additionally, by integrating semantically coherent jailbreak prompts generated by our method into widely used black-box methodologies, we observe a notable enhancement in their success rates when targeting closed-source commercial LLMs. This highlights the security threat posed by open-source LLMs to commercial counterparts. We will open-source our code if the paper is accepted.

摘要: 尽管对大型语言模型(LLM)进行了明确的调整，但它们仍可能被利用来触发意外行为，这一现象被称为“越狱”。目前的越狱攻击方法主要集中在针对闭源LLM的离散提示操作上，依赖于手动创建的提示模板和说服规则。然而，随着开源LLM能力的提高，确保它们的安全变得越来越重要。在这样的环境中，潜在攻击者对模型参数和梯度信息的可访问性加剧了越狱威胁的严重性。为了弥补这一研究空白，我们提出了一种新的\下划线{C}上边-\下划线{C}上边\下划线{J}纵断\下划线{A}ttack(CCJA)。我们将越狱攻击定义为掩蔽语言模型嵌入空间内的优化问题。通过组合优化，有效地平衡了越狱攻击成功率和语义一致性。广泛的评估表明，我们的方法不仅保持了语义的一致性，而且在攻击效率上超过了最先进的基线。此外，通过将我们的方法生成的语义连贯的越狱提示整合到广泛使用的黑盒方法中，我们观察到当目标是闭源商业LLM时，它们的成功率有了显著的提高。这突显了开源低成本管理对商业同行构成的安全威胁。如果论文被接受，我们将开放我们的代码。



## **49. Dagger Behind Smile: Fool LLMs with a Happy Ending Story**

微笑背后的匕首：傻瓜LLMs，有一个幸福的结局 cs.CL

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2501.13115v2) [paper-pdf](http://arxiv.org/pdf/2501.13115v2)

**Authors**: Xurui Song, Zhixin Xie, Shuo Huai, Jiayi Kong, Jun Luo

**Abstract**: The wide adoption of Large Language Models (LLMs) has attracted significant attention from $\textit{jailbreak}$ attacks, where adversarial prompts crafted through optimization or manual design exploit LLMs to generate malicious contents. However, optimization-based attacks have limited efficiency and transferability, while existing manual designs are either easily detectable or demand intricate interactions with LLMs. In this paper, we first point out a novel perspective for jailbreak attacks: LLMs are more responsive to $\textit{positive}$ prompts. Based on this, we deploy Happy Ending Attack (HEA) to wrap up a malicious request in a scenario template involving a positive prompt formed mainly via a $\textit{happy ending}$, it thus fools LLMs into jailbreaking either immediately or at a follow-up malicious request.This has made HEA both efficient and effective, as it requires only up to two turns to fully jailbreak LLMs. Extensive experiments show that our HEA can successfully jailbreak on state-of-the-art LLMs, including GPT-4o, Llama3-70b, Gemini-pro, and achieves 88.79\% attack success rate on average. We also provide quantitative explanations for the success of HEA.

摘要: 大型语言模型(LLM)的广泛采用引起了$\textit{jailBreak}$攻击的极大关注，即通过优化或手动设计创建的敌意提示利用LLM生成恶意内容。然而，基于优化的攻击的效率和可转移性有限，而现有的手动设计要么很容易被检测到，要么需要与LLM进行复杂的交互。在这篇文章中，我们首先指出了越狱攻击的一个新视角：LLM对$\textit{积极}$提示的响应更快。在此基础上，利用HEA(Happy End End Attack)将恶意请求封装在一个场景模板中，该场景模板包含一个主要通过$\textit{Happy End}$形成的积极提示，从而欺骗LLM立即越狱或在后续恶意请求时越狱，这使得HEA既高效又有效，因为它只需要最多两个回合就可以完全越狱LLM。大量的实验表明，我们的HEA能够成功地在GPT-40、Llama3-70b、Gemini-Pro等最先进的LLMS上越狱，平均攻击成功率达到88.79%。我们还对HEA的成功提供了定量的解释。



## **50. Mimicking the Familiar: Dynamic Command Generation for Information Theft Attacks in LLM Tool-Learning System**

模仿熟悉的：LLM工具学习系统中信息窃取攻击的动态命令生成 cs.AI

15 pages, 11 figures

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11358v1) [paper-pdf](http://arxiv.org/pdf/2502.11358v1)

**Authors**: Ziyou Jiang, Mingyang Li, Guowei Yang, Junjie Wang, Yuekai Huang, Zhiyuan Chang, Qing Wang

**Abstract**: Information theft attacks pose a significant risk to Large Language Model (LLM) tool-learning systems. Adversaries can inject malicious commands through compromised tools, manipulating LLMs to send sensitive information to these tools, which leads to potential privacy breaches. However, existing attack approaches are black-box oriented and rely on static commands that cannot adapt flexibly to the changes in user queries and the invocation chain of tools. It makes malicious commands more likely to be detected by LLM and leads to attack failure. In this paper, we propose AutoCMD, a dynamic attack comment generation approach for information theft attacks in LLM tool-learning systems. Inspired by the concept of mimicking the familiar, AutoCMD is capable of inferring the information utilized by upstream tools in the toolchain through learning on open-source systems and reinforcement with target system examples, thereby generating more targeted commands for information theft. The evaluation results show that AutoCMD outperforms the baselines with +13.2% $ASR_{Theft}$, and can be generalized to new tool-learning systems to expose their information leakage risks. We also design four defense methods to effectively protect tool-learning systems from the attack.

摘要: 信息窃取攻击对大型语言模型(LLM)工具学习系统构成了重大风险。攻击者可以通过受危害的工具注入恶意命令，操纵LLM向这些工具发送敏感信息，从而导致潜在的隐私泄露。然而，现有的攻击方法是面向黑盒的，依赖于静态命令，不能灵活地适应用户查询和工具调用链的变化。它使恶意命令更容易被LLM检测到，并导致攻击失败。本文针对LLM工具学习系统中的信息窃取攻击，提出了一种动态攻击评论生成方法AutoCMD。受模仿熟悉的概念的启发，AutoCMD能够通过学习开源系统和加强目标系统示例来推断工具链中的上游工具所使用的信息，从而生成更有针对性的信息窃取命令。评估结果表明，AutoCMD的性能比基准高出13.2%$ASR{Theft}$，可以推广到新的工具学习系统，以暴露其信息泄露风险。我们还设计了四种防御方法来有效地保护工具学习系统免受攻击。



