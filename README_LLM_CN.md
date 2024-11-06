# Latest Large Language Model Attack Papers
**update at 2024-11-06 17:27:10**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Jailbreaking Large Language Models with Symbolic Mathematics**

用符号数学破解大型语言模型 cs.CR

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2409.11445v2) [paper-pdf](http://arxiv.org/pdf/2409.11445v2)

**Authors**: Emet Bethany, Mazal Bethany, Juan Arturo Nolazco Flores, Sumit Kumar Jha, Peyman Najafirad

**Abstract**: Recent advancements in AI safety have led to increased efforts in training and red-teaming large language models (LLMs) to mitigate unsafe content generation. However, these safety mechanisms may not be comprehensive, leaving potential vulnerabilities unexplored. This paper introduces MathPrompt, a novel jailbreaking technique that exploits LLMs' advanced capabilities in symbolic mathematics to bypass their safety mechanisms. By encoding harmful natural language prompts into mathematical problems, we demonstrate a critical vulnerability in current AI safety measures. Our experiments across 13 state-of-the-art LLMs reveal an average attack success rate of 73.6\%, highlighting the inability of existing safety training mechanisms to generalize to mathematically encoded inputs. Analysis of embedding vectors shows a substantial semantic shift between original and encoded prompts, helping explain the attack's success. This work emphasizes the importance of a holistic approach to AI safety, calling for expanded red-teaming efforts to develop robust safeguards across all potential input types and their associated risks.

摘要: 最近人工智能安全方面的进步导致在培训和红队大型语言模型(LLM)方面加大了努力，以减少不安全的内容生成。然而，这些安全机制可能不是全面的，留下了潜在的漏洞有待探索。本文介绍了MathPrompt，这是一种新的越狱技术，它利用LLMS在符号数学中的高级能力来绕过它们的安全机制。通过将有害的自然语言提示编码为数学问题，我们展示了当前人工智能安全措施中的一个严重漏洞。我们在13个最先进的LLM上进行的实验显示，平均攻击成功率为73.6\%，这突显了现有安全培训机制无法概括为数学编码的输入。对嵌入向量的分析显示，原始提示和编码提示之间存在实质性的语义转换，这有助于解释攻击的成功。这项工作强调了对人工智能安全采取整体方法的重要性，呼吁扩大红队努力，为所有潜在的投入类型及其相关风险制定强有力的保障措施。



## **2. Membership Inference Attacks against Large Vision-Language Models**

针对大型视觉语言模型的成员推断攻击 cs.CV

NeurIPS 2024

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.02902v1) [paper-pdf](http://arxiv.org/pdf/2411.02902v1)

**Authors**: Zhan Li, Yongtao Wu, Yihang Chen, Francesco Tonin, Elias Abad Rocamora, Volkan Cevher

**Abstract**: Large vision-language models (VLLMs) exhibit promising capabilities for processing multi-modal tasks across various application scenarios. However, their emergence also raises significant data security concerns, given the potential inclusion of sensitive information, such as private photos and medical records, in their training datasets. Detecting inappropriately used data in VLLMs remains a critical and unresolved issue, mainly due to the lack of standardized datasets and suitable methodologies. In this study, we introduce the first membership inference attack (MIA) benchmark tailored for various VLLMs to facilitate training data detection. Then, we propose a novel MIA pipeline specifically designed for token-level image detection. Lastly, we present a new metric called MaxR\'enyi-K%, which is based on the confidence of the model output and applies to both text and image data. We believe that our work can deepen the understanding and methodology of MIAs in the context of VLLMs. Our code and datasets are available at https://github.com/LIONS-EPFL/VL-MIA.

摘要: 大型视觉语言模型(VLLM)在处理跨各种应用场景的多模式任务方面显示出良好的能力。然而，它们的出现也引发了重大的数据安全担忧，因为它们的训练数据集中可能包含私人照片和医疗记录等敏感信息。检测超低成本管理系统中不适当使用的数据仍然是一个关键和悬而未决的问题，这主要是因为缺乏标准化的数据集和适当的方法。在这项研究中，我们引入了针对不同的VLLM定制的第一个成员推理攻击(MIA)基准，以便于训练数据的检测。然后，我们提出了一种专门为令牌级图像检测设计的MIA流水线。最后，我们提出了一种新的度量方法MaxR‘enyi-K%，它基于模型输出的置信度，适用于文本和图像数据。我们相信，我们的工作可以加深对VLLMS环境下MIA的理解和方法学。我们的代码和数据集可以在https://github.com/LIONS-EPFL/VL-MIA.上找到



## **3. Stochastic Monkeys at Play: Random Augmentations Cheaply Break LLM Safety Alignment**

随机猴子在起作用：随机增强轻松打破LLM安全一致 cs.LG

Under peer review

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.02785v1) [paper-pdf](http://arxiv.org/pdf/2411.02785v1)

**Authors**: Jason Vega, Junsheng Huang, Gaokai Zhang, Hangoo Kang, Minjia Zhang, Gagandeep Singh

**Abstract**: Safety alignment of Large Language Models (LLMs) has recently become a critical objective of model developers. In response, a growing body of work has been investigating how safety alignment can be bypassed through various jailbreaking methods, such as adversarial attacks. However, these jailbreak methods can be rather costly or involve a non-trivial amount of creativity and effort, introducing the assumption that malicious users are high-resource or sophisticated. In this paper, we study how simple random augmentations to the input prompt affect safety alignment effectiveness in state-of-the-art LLMs, such as Llama 3 and Qwen 2. We perform an in-depth evaluation of 17 different models and investigate the intersection of safety under random augmentations with multiple dimensions: augmentation type, model size, quantization, fine-tuning-based defenses, and decoding strategies (e.g., sampling temperature). We show that low-resource and unsophisticated attackers, i.e. $\textit{stochastic monkeys}$, can significantly improve their chances of bypassing alignment with just 25 random augmentations per prompt.

摘要: 大型语言模型(LLM)的安全一致性最近已成为模型开发人员的一个重要目标。作为回应，越来越多的工作一直在研究如何通过各种越狱方法绕过安全对准，例如对抗性攻击。然而，这些越狱方法可能相当昂贵，或者涉及大量的创造力和努力，从而引入了恶意用户是高资源或老练的假设。在本文中，我们研究了输入提示的简单随机增强如何影响最新的LLMS的安全对齐有效性，例如Llama 3和Qwen 2。我们对17个不同的模型进行了深入的评估，并研究了在多个维度的随机增强下的安全性交集：增强类型、模型大小、量化、基于微调的防御和解码策略(例如采样温度)。我们表明，低资源和简单的攻击者，即$\textit{随机猴子}$，可以显著提高他们绕过对齐的机会，每个提示只需25个随机扩展。



## **4. Extracting Unlearned Information from LLMs with Activation Steering**

通过激活引导从LLM中提取未学习的信息 cs.CL

Accepted at NeurIPS 2024 Workshop Safe Generative AI

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2411.02631v1) [paper-pdf](http://arxiv.org/pdf/2411.02631v1)

**Authors**: Atakan Seyitoğlu, Aleksei Kuvshinov, Leo Schwinn, Stephan Günnemann

**Abstract**: An unintended consequence of the vast pretraining of Large Language Models (LLMs) is the verbatim memorization of fragments of their training data, which may contain sensitive or copyrighted information. In recent years, unlearning has emerged as a solution to effectively remove sensitive knowledge from models after training. Yet, recent work has shown that supposedly deleted information can still be extracted by malicious actors through various attacks. Still, current attacks retrieve sets of possible candidate generations and are unable to pinpoint the output that contains the actual target information. We propose activation steering as a method for exact information retrieval from unlearned LLMs. We introduce a novel approach to generating steering vectors, named Anonymized Activation Steering. Additionally, we develop a simple word frequency method to pinpoint the correct answer among a set of candidates when retrieving unlearned information. Our evaluation across multiple unlearning techniques and datasets demonstrates that activation steering successfully recovers general knowledge (e.g., widely known fictional characters) while revealing limitations in retrieving specific information (e.g., details about non-public individuals). Overall, our results demonstrate that exact information retrieval from unlearned models is possible, highlighting a severe vulnerability of current unlearning techniques.

摘要: 大型语言模型(LLM)的大量预训练的一个意想不到的后果是逐字记忆其训练数据的片段，其中可能包含敏感或受版权保护的信息。近年来，遗忘作为一种有效地去除训练后模型中敏感知识的解决方案而出现。然而，最近的研究表明，恶意攻击者仍然可以通过各种攻击提取本应删除的信息。尽管如此，当前的攻击检索到了可能的候选代集合，并且无法确定包含实际目标信息的输出。我们提出了激活引导作为一种从未学习的LLMS中准确检索信息的方法。我们介绍了一种新的生成引导向量的方法，称为匿名激活引导。此外，我们还开发了一种简单的词频方法，在检索未学习信息时，可以在一组候选对象中准确地找到正确答案。我们对多种遗忘技术和数据集的评估表明，激活引导成功地恢复了一般知识(例如，广为人知的虚拟角色)，同时揭示了在检索特定信息(例如，关于非公开个人的细节)方面的局限性。总体而言，我们的结果表明，从未学习的模型中检索准确的信息是可能的，这突显了当前遗忘技术的严重脆弱性。



## **5. Attacking Vision-Language Computer Agents via Pop-ups**

通过弹出窗口攻击视觉语言计算机代理 cs.CL

10 pages, preprint

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2411.02391v1) [paper-pdf](http://arxiv.org/pdf/2411.02391v1)

**Authors**: Yanzhe Zhang, Tao Yu, Diyi Yang

**Abstract**: Autonomous agents powered by large vision and language models (VLM) have demonstrated significant potential in completing daily computer tasks, such as browsing the web to book travel and operating desktop software, which requires agents to understand these interfaces. Despite such visual inputs becoming more integrated into agentic applications, what types of risks and attacks exist around them still remain unclear. In this work, we demonstrate that VLM agents can be easily attacked by a set of carefully designed adversarial pop-ups, which human users would typically recognize and ignore. This distraction leads agents to click these pop-ups instead of performing the tasks as usual. Integrating these pop-ups into existing agent testing environments like OSWorld and VisualWebArena leads to an attack success rate (the frequency of the agent clicking the pop-ups) of 86% on average and decreases the task success rate by 47%. Basic defense techniques such as asking the agent to ignore pop-ups or including an advertisement notice, are ineffective against the attack.

摘要: 由大视觉和语言模型(VLM)驱动的自主代理在完成日常计算机任务方面表现出了巨大的潜力，例如浏览网页预订旅行和操作桌面软件，这需要代理理解这些界面。尽管这样的视觉输入越来越多地集成到代理应用程序中，但围绕它们存在哪些类型的风险和攻击仍不清楚。在这项工作中，我们证明了VLM代理可以很容易地受到一组精心设计的敌意弹出窗口的攻击，人类用户通常会识别并忽略这些弹出窗口。这种干扰会导致工程师单击这些弹出窗口，而不是像往常一样执行任务。将这些弹出窗口集成到OSWorld和VisualWebArena等现有代理测试环境中，攻击成功率(代理单击弹出窗口的频率)平均为86%，任务成功率降低47%。基本的防御技术，如要求代理忽略弹出窗口或包括广告通知，对攻击无效。



## **6. Defining and Evaluating Physical Safety for Large Language Models**

定义和评估大型语言模型的物理安全 cs.LG

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2411.02317v1) [paper-pdf](http://arxiv.org/pdf/2411.02317v1)

**Authors**: Yung-Chen Tang, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Large Language Models (LLMs) are increasingly used to control robotic systems such as drones, but their risks of causing physical threats and harm in real-world applications remain unexplored. Our study addresses the critical gap in evaluating LLM physical safety by developing a comprehensive benchmark for drone control. We classify the physical safety risks of drones into four categories: (1) human-targeted threats, (2) object-targeted threats, (3) infrastructure attacks, and (4) regulatory violations. Our evaluation of mainstream LLMs reveals an undesirable trade-off between utility and safety, with models that excel in code generation often performing poorly in crucial safety aspects. Furthermore, while incorporating advanced prompt engineering techniques such as In-Context Learning and Chain-of-Thought can improve safety, these methods still struggle to identify unintentional attacks. In addition, larger models demonstrate better safety capabilities, particularly in refusing dangerous commands. Our findings and benchmark can facilitate the design and evaluation of physical safety for LLMs. The project page is available at huggingface.co/spaces/TrustSafeAI/LLM-physical-safety.

摘要: 大型语言模型(LLM)越来越多地被用于控制无人机等机器人系统，但它们在现实世界应用中造成物理威胁和伤害的风险仍未被探索。我们的研究通过开发一个全面的无人机控制基准来解决在评估LLM物理安全方面的关键空白。我们将无人机的物理安全风险分为四类：(1)人为目标的威胁，(2)对象为目标的威胁，(3)基础设施攻击，以及(4)违反监管规定。我们对主流LLM的评估揭示了实用性和安全性之间的一种不受欢迎的权衡，在代码生成方面表现出色的模型在关键的安全方面往往表现不佳。此外，虽然结合了先进的即时工程技术，如情景学习和思维链可以提高安全性，但这些方法仍然难以识别无意攻击。此外，较大的型号显示出更好的安全能力，特别是在拒绝危险命令方面。我们的研究结果和基准可以帮助设计和评估LLMS的物理安全性。该项目页面可在huggingface.co/spaces/TrustSafeAI/LLM-physical-safety.上查看



## **7. Manipulation Facing Threats: Evaluating Physical Vulnerabilities in End-to-End Vision Language Action Models**

面对威胁的操纵：评估端到端视觉语言动作模型中的身体脆弱性 cs.CV

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2409.13174v2) [paper-pdf](http://arxiv.org/pdf/2409.13174v2)

**Authors**: Hao Cheng, Erjia Xiao, Chengyuan Yu, Zhao Yao, Jiahang Cao, Qiang Zhang, Jiaxu Wang, Mengshu Sun, Kaidi Xu, Jindong Gu, Renjing Xu

**Abstract**: Recently, driven by advancements in Multimodal Large Language Models (MLLMs), Vision Language Action Models (VLAMs) are being proposed to achieve better performance in open-vocabulary scenarios for robotic manipulation tasks. Since manipulation tasks involve direct interaction with the physical world, ensuring robustness and safety during the execution of this task is always a very critical issue. In this paper, by synthesizing current safety research on MLLMs and the specific application scenarios of the manipulation task in the physical world, we comprehensively evaluate VLAMs in the face of potential physical threats. Specifically, we propose the Physical Vulnerability Evaluating Pipeline (PVEP) that can incorporate as many visual modal physical threats as possible for evaluating the physical robustness of VLAMs. The physical threats in PVEP specifically include Out-of-Distribution, Typography-based Visual Prompts, and Adversarial Patch Attacks. By comparing the performance fluctuations of VLAMs before and after being attacked, we provide generalizable Analyses of how VLAMs respond to different physical security threats. Our project page is in this link: https://chaducheng.github.io/Manipulat-Facing-Threats/.

摘要: 最近，在多模式大语言模型(MLLM)的推动下，视觉语言动作模型(VLAM)被提出以在机器人操作任务的开放词汇场景中实现更好的性能。由于操作任务涉及与物理世界的直接交互，因此确保该任务执行过程中的健壮性和安全性始终是一个非常关键的问题。本文通过综合当前MLLMS的安全研究现状和物理世界中操纵任务的具体应用场景，对VLAMS在面临潜在物理威胁的情况下进行综合评估。具体地说，我们提出了物理脆弱性评估管道(PVEP)，它可以结合尽可能多的视觉通道物理威胁来评估VLAMS的物理健壮性。PVEP中的物理威胁具体包括分发外、基于排版的视觉提示和对抗性补丁攻击。通过比较VLAM在受到攻击前后的性能波动，我们对VLAM如何应对不同的物理安全威胁提供了一般性的分析。我们的项目页面位于以下链接：https://chaducheng.github.io/Manipulat-Facing-Threats/.



## **8. Whispers in the Machine: Confidentiality in LLM-integrated Systems**

机器中的耳语：LLM集成系统的保密性 cs.CR

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2402.06922v2) [paper-pdf](http://arxiv.org/pdf/2402.06922v2)

**Authors**: Jonathan Evertz, Merlin Chlosta, Lea Schönherr, Thorsten Eisenhofer

**Abstract**: Large Language Models (LLMs) are increasingly augmented with external tools and commercial services into LLM-integrated systems. While these interfaces can significantly enhance the capabilities of the models, they also introduce a new attack surface. Manipulated integrations, for example, can exploit the model and compromise sensitive data accessed through other interfaces. While previous work primarily focused on attacks targeting a model's alignment or the leakage of training data, the security of data that is only available during inference has escaped scrutiny so far. In this work, we demonstrate the vulnerabilities associated with external components and introduce a systematic approach to evaluate confidentiality risks in LLM-integrated systems. We identify two specific attack scenarios unique to these systems and formalize these into a tool-robustness framework designed to measure a model's ability to protect sensitive information. Our findings show that all examined models are highly vulnerable to confidentiality attacks, with the risk increasing significantly when models are used together with external tools.

摘要: 大型语言模型(LLM)越来越多地被外部工具和商业服务扩展到LLM集成系统中。虽然这些接口可以显著增强模型的功能，但它们也引入了新的攻击面。例如，受操纵的集成可能会利用该模型并危及通过其他接口访问的敏感数据。虽然以前的工作主要集中在针对模型对齐或训练数据泄漏的攻击上，但到目前为止，只有在推理过程中才能获得的数据的安全性没有受到审查。在这项工作中，我们展示了与外部组件相关的漏洞，并引入了一种系统的方法来评估LLM集成系统中的机密性风险。我们确定了这些系统特有的两种特定攻击场景，并将它们正式化为工具健壮性框架，旨在衡量模型保护敏感信息的能力。我们的发现表明，所有被检查的模型都非常容易受到保密攻击，当模型与外部工具一起使用时，风险会显著增加。



## **9. Exploiting LLM Quantization**

利用LLM量化 cs.LG

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2405.18137v2) [paper-pdf](http://arxiv.org/pdf/2405.18137v2)

**Authors**: Kazuki Egashira, Mark Vero, Robin Staab, Jingxuan He, Martin Vechev

**Abstract**: Quantization leverages lower-precision weights to reduce the memory usage of large language models (LLMs) and is a key technique for enabling their deployment on commodity hardware. While LLM quantization's impact on utility has been extensively explored, this work for the first time studies its adverse effects from a security perspective. We reveal that widely used quantization methods can be exploited to produce a harmful quantized LLM, even though the full-precision counterpart appears benign, potentially tricking users into deploying the malicious quantized model. We demonstrate this threat using a three-staged attack framework: (i) first, we obtain a malicious LLM through fine-tuning on an adversarial task; (ii) next, we quantize the malicious model and calculate constraints that characterize all full-precision models that map to the same quantized model; (iii) finally, using projected gradient descent, we tune out the poisoned behavior from the full-precision model while ensuring that its weights satisfy the constraints computed in step (ii). This procedure results in an LLM that exhibits benign behavior in full precision but when quantized, it follows the adversarial behavior injected in step (i). We experimentally demonstrate the feasibility and severity of such an attack across three diverse scenarios: vulnerable code generation, content injection, and over-refusal attack. In practice, the adversary could host the resulting full-precision model on an LLM community hub such as Hugging Face, exposing millions of users to the threat of deploying its malicious quantized version on their devices.

摘要: 量化利用较低精度的权重来减少大型语言模型(LLM)的内存使用，这是在商用硬件上部署LLM的关键技术。虽然LLM量化对效用的影响已经被广泛研究，但这项工作首次从安全的角度研究了它的不利影响。我们发现，广泛使用的量化方法可以被利用来产生有害的量化LLM，即使全精度对应的看起来是良性的，潜在地诱骗用户部署恶意量化模型。我们使用一个三阶段攻击框架演示了这一威胁：(I)首先，我们通过对敌方任务的微调来获得恶意LLM；(Ii)接下来，我们量化恶意模型，并计算映射到相同量化模型的所有全精度模型的约束；(Iii)最后，使用投影梯度下降，我们在确保其权重满足步骤(Ii)中计算的约束的同时，从全精度模型中排除有毒行为。这一过程导致LLM完全精确地表现出良性行为，但当量化时，它遵循在步骤(I)中注入的对抗性行为。我们通过实验演示了这种攻击在三种不同场景中的可行性和严重性：易受攻击的代码生成、内容注入和过度拒绝攻击。在实践中，对手可能会在LLM社区中心(如拥抱脸)上托管产生的全精度模型，使数百万用户面临在他们的设备上部署其恶意量化版本的威胁。



## **10. Data Extraction Attacks in Retrieval-Augmented Generation via Backdoors**

通过后门进行检索增强生成中的数据提取攻击 cs.CR

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2411.01705v1) [paper-pdf](http://arxiv.org/pdf/2411.01705v1)

**Authors**: Yuefeng Peng, Junda Wang, Hong Yu, Amir Houmansadr

**Abstract**: Despite significant advancements, large language models (LLMs) still struggle with providing accurate answers when lacking domain-specific or up-to-date knowledge. Retrieval-Augmented Generation (RAG) addresses this limitation by incorporating external knowledge bases, but it also introduces new attack surfaces. In this paper, we investigate data extraction attacks targeting the knowledge databases of RAG systems. We demonstrate that previous attacks on RAG largely depend on the instruction-following capabilities of LLMs, and that simple fine-tuning can reduce the success rate of such attacks to nearly zero. This makes these attacks impractical since fine-tuning is a common practice when deploying LLMs in specific domains. To further reveal the vulnerability, we propose to backdoor RAG, where a small portion of poisoned data is injected during the fine-tuning phase to create a backdoor within the LLM. When this compromised LLM is integrated into a RAG system, attackers can exploit specific triggers in prompts to manipulate the LLM to leak documents from the retrieval database. By carefully designing the poisoned data, we achieve both verbatim and paraphrased document extraction. We show that with only 3\% poisoned data, our method achieves an average success rate of 79.7\% in verbatim extraction on Llama2-7B, with a ROUGE-L score of 64.21, and a 68.6\% average success rate in paraphrased extraction, with an average ROUGE score of 52.6 across four datasets. These results underscore the privacy risks associated with the supply chain when deploying RAG systems.

摘要: 尽管有了很大的进步，但大型语言模型(LLM)在缺乏特定领域或最新知识的情况下，仍然难以提供准确的答案。检索-增强生成(RAG)通过结合外部知识库解决了这一限制，但它也引入了新的攻击面。本文研究了针对RAG系统知识库的数据抽取攻击。我们证明了以前对RAG的攻击在很大程度上依赖于LLMS的指令跟随能力，并且简单的微调可以将此类攻击的成功率降低到几乎为零。这使得这些攻击不切实际，因为在特定域中部署LLM时，微调是一种常见的做法。为了进一步揭示漏洞，我们建议使用后门RAG，在微调阶段注入一小部分有毒数据，以在LLM中创建后门。当这个受损的LLM被集成到RAG系统中时，攻击者可以利用提示中的特定触发器来操纵LLM，从而从检索数据库中泄漏文档。通过精心设计有毒数据，我们实现了逐字和释义文档提取。实验结果表明，在3个中毒数据的情况下，该方法在Llama2-7B上的平均逐字提取成功率为79.7%，Rouge-L评分为64.2 1分，转述提取的平均成功率为68.6%，4个数据集的平均Rouge评分为5 2.6分。这些结果突显了在部署RAG系统时与供应链相关的隐私风险。



## **11. UniGuard: Towards Universal Safety Guardrails for Jailbreak Attacks on Multimodal Large Language Models**

UniGuard：为多模式大型语言模型越狱攻击建立通用安全护栏 cs.CL

14 pages

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2411.01703v1) [paper-pdf](http://arxiv.org/pdf/2411.01703v1)

**Authors**: Sejoon Oh, Yiqiao Jin, Megha Sharma, Donghyun Kim, Eric Ma, Gaurav Verma, Srijan Kumar

**Abstract**: Multimodal large language models (MLLMs) have revolutionized vision-language understanding but are vulnerable to multimodal jailbreak attacks, where adversaries meticulously craft inputs to elicit harmful or inappropriate responses. We propose UniGuard, a novel multimodal safety guardrail that jointly considers the unimodal and cross-modal harmful signals. UniGuard is trained such that the likelihood of generating harmful responses in a toxic corpus is minimized, and can be seamlessly applied to any input prompt during inference with minimal computational costs. Extensive experiments demonstrate the generalizability of UniGuard across multiple modalities and attack strategies. It demonstrates impressive generalizability across multiple state-of-the-art MLLMs, including LLaVA, Gemini Pro, GPT-4, MiniGPT-4, and InstructBLIP, thereby broadening the scope of our solution.

摘要: 多模式大型语言模型（MLLM）彻底改变了视觉语言理解，但很容易受到多模式越狱攻击，对手精心设计输入以引发有害或不当的反应。我们提出了UniGuard，这是一种新型的多模式安全护栏，它联合考虑了单模式和跨模式有害信号。UniGuard经过训练，可以最大限度地降低在有毒的数据库中生成有害响应的可能性，并且可以以最小的计算成本无缝地应用于推理期间的任何输入提示。大量实验证明了UniGuard在多种模式和攻击策略中的通用性。它在多个最先进的MLLM（包括LLaVA、Gemini Pro、GPT-4、MiniGPT-4和DirecectBLIP）上展示了令人印象深刻的通用性，从而扩大了我们解决方案的范围。



## **12. CARES: A Comprehensive Benchmark of Trustworthiness in Medical Vision Language Models**

CARES：医学视觉语言模型可信度的综合基准 cs.LG

NeurIPS 2024 Datasets and Benchmarks Track

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2406.06007v3) [paper-pdf](http://arxiv.org/pdf/2406.06007v3)

**Authors**: Peng Xia, Ze Chen, Juanxi Tian, Yangrui Gong, Ruibo Hou, Yue Xu, Zhenbang Wu, Zhiyuan Fan, Yiyang Zhou, Kangyu Zhu, Wenhao Zheng, Zhaoyang Wang, Xiao Wang, Xuchao Zhang, Chetan Bansal, Marc Niethammer, Junzhou Huang, Hongtu Zhu, Yun Li, Jimeng Sun, Zongyuan Ge, Gang Li, James Zou, Huaxiu Yao

**Abstract**: Artificial intelligence has significantly impacted medical applications, particularly with the advent of Medical Large Vision Language Models (Med-LVLMs), sparking optimism for the future of automated and personalized healthcare. However, the trustworthiness of Med-LVLMs remains unverified, posing significant risks for future model deployment. In this paper, we introduce CARES and aim to comprehensively evaluate the Trustworthiness of Med-LVLMs across the medical domain. We assess the trustworthiness of Med-LVLMs across five dimensions, including trustfulness, fairness, safety, privacy, and robustness. CARES comprises about 41K question-answer pairs in both closed and open-ended formats, covering 16 medical image modalities and 27 anatomical regions. Our analysis reveals that the models consistently exhibit concerns regarding trustworthiness, often displaying factual inaccuracies and failing to maintain fairness across different demographic groups. Furthermore, they are vulnerable to attacks and demonstrate a lack of privacy awareness. We publicly release our benchmark and code in https://cares-ai.github.io/.

摘要: 人工智能对医疗应用产生了重大影响，特别是随着医学大视觉语言模型(Med-LVLMS)的出现，引发了对自动化和个性化医疗保健未来的乐观情绪。然而，MED-LVLMS的可信性仍未得到验证，对未来的模型部署构成重大风险。在本文中，我们引入CARE，旨在全面评估医学领域的MED-LVLMS的可信性。我们从可信性、公平性、安全性、隐私性和健壮性五个维度评估Med-LVLM的可信性。CARE包括约41K个封闭式和开放式格式的问答对，涵盖16种医学影像模式和27个解剖区域。我们的分析表明，这些模型始终表现出对可信度的担忧，经常表现出事实上的不准确，并且未能在不同的人口群体中保持公平。此外，他们很容易受到攻击，并表现出缺乏隐私意识。我们在https://cares-ai.github.io/.中公开发布我们的基准测试和代码



## **13. Are you still on track!? Catching LLM Task Drift with Activations**

你还在正轨上吗！？通过激活捕捉LLM任务漂移 cs.CR

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2406.00799v5) [paper-pdf](http://arxiv.org/pdf/2406.00799v5)

**Authors**: Sahar Abdelnabi, Aideen Fay, Giovanni Cherubin, Ahmed Salem, Mario Fritz, Andrew Paverd

**Abstract**: Large Language Models are commonly used in retrieval-augmented applications to execute user instructions based on data from external sources. For example, modern search engines use LLMs to answer queries based on relevant search results; email plugins summarize emails by processing their content through an LLM. However, the potentially untrusted provenance of these data sources can lead to prompt injection attacks, where the LLM is manipulated by natural language instructions embedded in the external data, causing it to deviate from the user's original instruction(s). We define this deviation as task drift. Task drift is a significant concern as it allows attackers to exfiltrate data or influence the LLM's output for other users. We study LLM activations as a solution to detect task drift, showing that activation deltas - the difference in activations before and after processing external data - are strongly correlated with this phenomenon. Through two probing methods, we demonstrate that a simple linear classifier can detect drift with near-perfect ROC AUC on an out-of-distribution test set. We evaluate these methods by making minimal assumptions about how user's tasks, system prompts, and attacks can be phrased. We observe that this approach generalizes surprisingly well to unseen task domains, such as prompt injections, jailbreaks, and malicious instructions, without being trained on any of these attacks. Interestingly, the fact that this solution does not require any modifications to the LLM (e.g., fine-tuning), as well as its compatibility with existing meta-prompting solutions, makes it cost-efficient and easy to deploy. To encourage further research on activation-based task inspection, decoding, and interpretability, we release our large-scale TaskTracker toolkit, featuring a dataset of over 500K instances, representations from six SoTA language models, and inspection tools.

摘要: 大型语言模型通常用于增强检索的应用程序中，以基于来自外部源的数据执行用户指令。例如，现代搜索引擎使用LLM根据相关搜索结果回答查询；电子邮件插件通过LLM处理电子邮件的内容来汇总电子邮件。然而，这些数据源的潜在不可信来源可能导致提示注入攻击，其中LLM被嵌入外部数据的自然语言指令操纵，导致其偏离用户的原始指令(S)。我们将这种偏差定义为任务漂移。任务漂移是一个重要的问题，因为它允许攻击者窃取数据或影响LLM对其他用户的输出。我们研究了LLM激活作为检测任务漂移的解决方案，表明激活增量-处理外部数据之前和之后的激活差异-与这一现象密切相关。通过两种探测方法，我们证明了一个简单的线性分类器可以在非分布测试集上以接近完美的ROC AUC来检测漂移。我们通过对用户任务、系统提示和攻击的表述方式做出最小假设来评估这些方法。我们观察到，这种方法对看不见的任务领域(如提示注入、越狱和恶意指令)的泛化效果出奇地好，而且没有接受过任何这些攻击的培训。有趣的是，该解决方案不需要对LLM进行任何修改(例如微调)，并且它与现有的元提示解决方案兼容，这使得它具有成本效益并且易于部署。为了鼓励对基于激活的任务检测、解码和可解释性的进一步研究，我们发布了我们的大型TaskTracker工具包，其中包含超过50万个实例的数据集、来自六个SOTA语言模型的表示以及检测工具。



## **14. SQL Injection Jailbreak: a structural disaster of large language models**

SQL注入越狱：大型语言模型的结构性灾难 cs.CR

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2411.01565v1) [paper-pdf](http://arxiv.org/pdf/2411.01565v1)

**Authors**: Jiawei Zhao, Kejiang Chen, Weiming Zhang, Nenghai Yu

**Abstract**: In recent years, the rapid development of large language models (LLMs) has brought new vitality to the various domains and generated substantial social and economic benefits. However, the swift advancement of LLMs has introduced new security vulnerabilities. Jailbreak, a form of attack that induces LLMs to output harmful content through carefully crafted prompts, poses a challenge to the safe and trustworthy development of LLMs. Previous jailbreak attack methods primarily exploited the internal capabilities of the model. Among them, one category leverages the model's implicit capabilities for jailbreak attacks, where the attacker is unaware of the exact reasons for the attack's success. The other category utilizes the model's explicit capabilities for jailbreak attacks, where the attacker understands the reasons for the attack's success. For example, these attacks exploit the model's abilities in coding, contextual learning, or understanding ASCII characters. However, these earlier jailbreak attacks have certain limitations, as they only exploit the inherent capabilities of the model. In this paper, we propose a novel jailbreak method, SQL Injection Jailbreak (SIJ), which utilizes the construction of input prompts by LLMs to inject jailbreak information into user prompts, enabling successful jailbreak of the LLMs. Our SIJ method achieves nearly 100\% attack success rates on five well-known open-source LLMs in the context of AdvBench, while incurring lower time costs compared to previous methods. More importantly, SIJ reveals a new vulnerability in LLMs that urgently needs to be addressed. To this end, we propose a defense method called Self-Reminder-Key and demonstrate its effectiveness through experiments. Our code is available at \href{https://github.com/weiyezhimeng/SQL-Injection-Jailbreak}{https://github.com/weiyezhimeng/SQL-Injection-Jailbreak}.

摘要: 近年来，大语言模型的快速发展给各个领域带来了新的活力，产生了可观的社会效益和经济效益。然而，LLMS的快速发展带来了新的安全漏洞。越狱是一种攻击形式，通过精心制作的提示诱使LLMS输出有害内容，对LLMS的安全和可信开发构成了挑战。以前的越狱攻击方法主要是利用该模型的内部能力。其中，一类利用该模型的隐含能力进行越狱攻击，即攻击者不知道攻击成功的确切原因。另一类利用该模型的明确能力进行越狱攻击，攻击者了解攻击成功的原因。例如，这些攻击利用了模型在编码、上下文学习或理解ASCII字符方面的能力。然而，这些早期的越狱攻击有一定的局限性，因为它们只利用了该模型的固有功能。本文提出了一种新的越狱方法--SQL注入越狱(SIJ)，该方法利用LLMS构造输入提示，在用户提示中注入越狱信息，使LLMS能够成功越狱。与以前的方法相比，我们的SIJ方法在五个著名的开源LLM上获得了近100\%的攻击成功率，同时产生了更低的时间开销。更重要的是，SIJ揭示了LLMS中一个迫切需要解决的新漏洞。为此，我们提出了一种称为自我提醒密钥的防御方法，并通过实验验证了该方法的有效性。我们的代码可以在\href{https://github.com/weiyezhimeng/SQL-Injection-Jailbreak}{https://github.com/weiyezhimeng/SQL-Injection-Jailbreak}.上找到



## **15. Boosting Jailbreak Transferability for Large Language Models**

提高大型语言模型的越狱可移植性 cs.AI

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2410.15645v2) [paper-pdf](http://arxiv.org/pdf/2410.15645v2)

**Authors**: Hanqing Liu, Lifeng Zhou, Huanqian Yan

**Abstract**: Large language models have drawn significant attention to the challenge of safe alignment, especially regarding jailbreak attacks that circumvent security measures to produce harmful content. To address the limitations of existing methods like GCG, which perform well in single-model attacks but lack transferability, we propose several enhancements, including a scenario induction template, optimized suffix selection, and the integration of re-suffix attack mechanism to reduce inconsistent outputs. Our approach has shown superior performance in extensive experiments across various benchmarks, achieving nearly 100% success rates in both attack execution and transferability. Notably, our method has won the first place in the AISG-hosted Global Challenge for Safe and Secure LLMs. The code is released at https://github.com/HqingLiu/SI-GCG.

摘要: 大型语言模型引起了人们对安全对齐挑战的高度关注，特别是对于绕过安全措施以产生有害内容的越狱攻击。为了解决GCG等现有方法在单模型攻击中表现良好但缺乏可移植性的局限性，我们提出了几项增强措施，包括场景归纳模板、优化的后缀选择以及集成重新后缀攻击机制以减少不一致的输出。我们的方法在各种基准测试的广泛实验中表现出卓越的性能，在攻击执行和可转移性方面都实现了近100%的成功率。值得注意的是，我们的方法在AISG主办的全球安全LLM挑战赛中赢得了第一名。该代码发布于https://github.com/HqingLiu/SI-GCG。



## **16. Desert Camels and Oil Sheikhs: Arab-Centric Red Teaming of Frontier LLMs**

沙漠骆驼和石油酋长：以阿拉伯为中心的红色Frontier LLM团队 cs.CL

**SubmitDate**: 2024-11-02    [abs](http://arxiv.org/abs/2410.24049v2) [paper-pdf](http://arxiv.org/pdf/2410.24049v2)

**Authors**: Muhammed Saeed, Elgizouli Mohamed, Mukhtar Mohamed, Shaina Raza, Shady Shehata, Muhammad Abdul-Mageed

**Abstract**: Large language models (LLMs) are widely used but raise ethical concerns due to embedded social biases. This study examines LLM biases against Arabs versus Westerners across eight domains, including women's rights, terrorism, and anti-Semitism and assesses model resistance to perpetuating these biases. To this end, we create two datasets: one to evaluate LLM bias toward Arabs versus Westerners and another to test model safety against prompts that exaggerate negative traits ("jailbreaks"). We evaluate six LLMs -- GPT-4, GPT-4o, LlaMA 3.1 (8B & 405B), Mistral 7B, and Claude 3.5 Sonnet. We find 79% of cases displaying negative biases toward Arabs, with LlaMA 3.1-405B being the most biased. Our jailbreak tests reveal GPT-4o as the most vulnerable, despite being an optimized version, followed by LlaMA 3.1-8B and Mistral 7B. All LLMs except Claude exhibit attack success rates above 87% in three categories. We also find Claude 3.5 Sonnet the safest, but it still displays biases in seven of eight categories. Despite being an optimized version of GPT4, We find GPT-4o to be more prone to biases and jailbreaks, suggesting optimization flaws. Our findings underscore the pressing need for more robust bias mitigation strategies and strengthened security measures in LLMs.

摘要: 大型语言模型(LLM)被广泛使用，但由于根深蒂固的社会偏见而引发了伦理问题。这项研究考察了LLM在八个领域对阿拉伯人和西方人的偏见，包括妇女权利、恐怖主义和反犹太主义，并评估了对延续这些偏见的模型阻力。为此，我们创建了两个数据集：一个用于评估LLM对阿拉伯人和西方人的偏见，另一个用于测试针对夸大负面特征的提示(“越狱”)的模型安全性。我们评估了六个LLMS--GPT-4、GPT-40、大羊驼3.1(8B和405B)、西北风7B和克劳德3.5十四行诗。我们发现79%的病例对阿拉伯人表现出负面偏见，其中大羊驼3.1-405B是最有偏见的。我们的越狱测试显示，GPT-40是最脆弱的，尽管是一个优化版本，紧随其后的是骆驼3.1-8B和米斯特拉尔7B。除克劳德外，所有LLM在三个类别中的攻击成功率都在87%以上。我们也发现克劳德3.5十四行诗是最安全的，但它仍然在八个类别中的七个方面表现出偏见。尽管GPT-4是GPT4的优化版本，但我们发现GPT-40更容易产生偏见和越狱，这表明优化存在缺陷。我们的研究结果突出表明，迫切需要更有力的减轻偏见战略和加强小岛屿发展中国家的安全措施。



## **17. Code-Switching Red-Teaming: LLM Evaluation for Safety and Multilingual Understanding**

代码转换红色团队：安全性和多语言理解的LLM评估 cs.AI

**SubmitDate**: 2024-11-02    [abs](http://arxiv.org/abs/2406.15481v2) [paper-pdf](http://arxiv.org/pdf/2406.15481v2)

**Authors**: Haneul Yoo, Yongjin Yang, Hwaran Lee

**Abstract**: As large language models (LLMs) have advanced rapidly, concerns regarding their safety have become prominent. In this paper, we discover that code-switching in red-teaming queries can effectively elicit undesirable behaviors of LLMs, which are common practices in natural language. We introduce a simple yet effective framework, CSRT, to synthesize code-switching red-teaming queries and investigate the safety and multilingual understanding of LLMs comprehensively. Through extensive experiments with ten state-of-the-art LLMs and code-switching queries combining up to 10 languages, we demonstrate that the CSRT significantly outperforms existing multilingual red-teaming techniques, achieving 46.7% more attacks than standard attacks in English and being effective in conventional safety domains. We also examine the multilingual ability of those LLMs to generate and understand code-switching texts. Additionally, we validate the extensibility of the CSRT by generating code-switching attack prompts with monolingual data. We finally conduct detailed ablation studies exploring code-switching and propound unintended correlation between resource availability of languages and safety alignment in existing multilingual LLMs.

摘要: 随着大型语言模型(LLM)的迅速发展，人们对其安全性的担忧也变得突出。在本文中，我们发现红队查询中的代码转换可以有效地导致自然语言中常见的LLM的不良行为。我们引入了一个简单而有效的框架CSRT来合成代码转换红队查询，并全面地研究了LLMS的安全性和多语言理解。通过对10种最先进的LLM和多达10种语言的代码切换查询的大量实验，我们证明了CSRT的性能显著优于现有的多语言红色团队技术，达到了比标准英语攻击高46.7%的攻击效果，并且在传统安全领域是有效的。我们还考察了这些LLMS生成和理解语码转换文本的多语言能力。此外，我们通过使用单语数据生成代码转换攻击提示来验证CSRT的可扩展性。最后，我们进行了详细的消融研究，探索了语码转换，并提出了在现有的多语种LLM中，语言的资源可用性与安全对齐之间的意外关联。



## **18. HuRef: HUman-REadable Fingerprint for Large Language Models**

HuRef：大型语言模型的人类可读取指纹 cs.CL

NeurIPS 2024

**SubmitDate**: 2024-11-02    [abs](http://arxiv.org/abs/2312.04828v4) [paper-pdf](http://arxiv.org/pdf/2312.04828v4)

**Authors**: Boyi Zeng, Lizheng Wang, Yuncong Hu, Yi Xu, Chenghu Zhou, Xinbing Wang, Yu Yu, Zhouhan Lin

**Abstract**: Protecting the copyright of large language models (LLMs) has become crucial due to their resource-intensive training and accompanying carefully designed licenses. However, identifying the original base model of an LLM is challenging due to potential parameter alterations. In this study, we introduce HuRef, a human-readable fingerprint for LLMs that uniquely identifies the base model without interfering with training or exposing model parameters to the public. We first observe that the vector direction of LLM parameters remains stable after the model has converged during pretraining, with negligible perturbations through subsequent training steps, including continued pretraining, supervised fine-tuning, and RLHF, which makes it a sufficient condition to identify the base model. The necessity is validated by continuing to train an LLM with an extra term to drive away the model parameters' direction and the model becomes damaged. However, this direction is vulnerable to simple attacks like dimension permutation or matrix rotation, which significantly change it without affecting performance. To address this, leveraging the Transformer structure, we systematically analyze potential attacks and define three invariant terms that identify an LLM's base model. Due to the potential risk of information leakage, we cannot publish invariant terms directly. Instead, we map them to a Gaussian vector using an encoder, then convert it into a natural image using StyleGAN2, and finally publish the image. In our black-box setting, all fingerprinting steps are internally conducted by the LLMs owners. To ensure the published fingerprints are honestly generated, we introduced Zero-Knowledge Proof (ZKP). Experimental results across various LLMs demonstrate the effectiveness of our method. The code is available at https://github.com/LUMIA-Group/HuRef.

摘要: 保护大型语言模型(LLM)的版权已变得至关重要，因为它们需要进行资源密集型培训，并附带精心设计的许可证。然而，由于潜在的参数变化，识别LLM的原始基础模型是具有挑战性的。在这项研究中，我们引入了HuRef，这是一种用于LLMS的人类可读指纹，它在不干扰训练或向公众暴露模型参数的情况下唯一地识别基本模型。我们首先观察到，在预训练过程中模型收敛后，LLM参数的向量方向保持稳定，通过后续的训练步骤，包括继续预训练、有监督的微调和RLHF，可以忽略不计的扰动，这使得它成为识别基本模型的充分条件。通过继续训练一个带有额外项的LLM来驱离模型参数的方向，从而使模型受损，从而验证了这种必要性。然而，这个方向很容易受到维度置换或矩阵旋转等简单攻击，这些攻击会在不影响性能的情况下显著改变它。为了解决这个问题，利用Transformer结构，我们系统地分析了潜在的攻击，并定义了识别LLM基本模型的三个不变术语。由于潜在的信息泄露风险，我们不能直接发布不变项。相反，我们使用编码器将它们映射到高斯向量，然后使用StyleGAN2将其转换为自然图像，最后发布图像。在我们的黑盒设置中，所有指纹识别步骤都由LLMS所有者在内部执行。为了确保公布的指纹是真实生成的，我们引入了零知识证明(ZKP)。在不同LLM上的实验结果证明了该方法的有效性。代码可在https://github.com/LUMIA-Group/HuRef.上获得



## **19. Plentiful Jailbreaks with String Compositions**

弦乐作品丰富越狱 cs.CL

NeurIPS SoLaR Workshop 2024

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.01084v1) [paper-pdf](http://arxiv.org/pdf/2411.01084v1)

**Authors**: Brian R. Y. Huang

**Abstract**: Large language models (LLMs) remain vulnerable to a slew of adversarial attacks and jailbreaking methods. One common approach employed by white-hat attackers, or \textit{red-teamers}, is to process model inputs and outputs using string-level obfuscations, which can include leetspeak, rotary ciphers, Base64, ASCII, and more. Our work extends these encoding-based attacks by unifying them in a framework of invertible string transformations. With invertibility, we can devise arbitrary \textit{string compositions}, defined as sequences of transformations, that we can encode and decode end-to-end programmatically. We devise a automated best-of-n attack that samples from a combinatorially large number of string compositions. Our jailbreaks obtain competitive attack success rates on several leading frontier models when evaluated on HarmBench, highlighting that encoding-based attacks remain a persistent vulnerability even in advanced LLMs.

摘要: 大型语言模型（LLM）仍然容易受到一系列对抗攻击和越狱方法的影响。白帽攻击者（\textit{red-teamers}）采用的一种常见方法是使用字符串级混淆处理模型输入和输出，其中可以包括leetspeak、旋转密码、Base 64、ASC等。我们的工作通过将这些基于编码的攻击统一到可逆字符串转换的框架中来扩展它们。通过可逆性，我们可以设计任意的\textit{字符串合成}，定义为转换序列，我们可以通过编程方式进行端到端编码和解码。我们设计了一种自动化的n中最佳攻击，该攻击从组合大量字符串组合中进行采样。在HarmBench上进行评估时，我们的越狱在几个领先的前沿模型上获得了有竞争力的攻击成功率，这凸显了即使在高级LLM中，基于编码的攻击仍然是一个持久的漏洞。



## **20. Emoji Attack: A Method for Misleading Judge LLMs in Safety Risk Detection**

Emoji攻击：一种在安全风险检测中误导法官LLM的方法 cs.CL

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.01077v1) [paper-pdf](http://arxiv.org/pdf/2411.01077v1)

**Authors**: Zhipeng Wei, Yuqi Liu, N. Benjamin Erichson

**Abstract**: Jailbreaking attacks show how Large Language Models (LLMs) can be tricked into generating harmful outputs using malicious prompts. To prevent these attacks, other LLMs are often used as judges to evaluate the harmfulness of the generated content. However, relying on LLMs as judges can introduce biases into the detection process, which in turn compromises the effectiveness of the evaluation. In this paper, we show that Judge LLMs, like other LLMs, are also affected by token segmentation bias. This bias occurs when tokens are split into smaller sub-tokens, altering their embeddings. This makes it harder for the model to detect harmful content. Specifically, this bias can cause sub-tokens to differ significantly from the original token in the embedding space, leading to incorrect "safe" predictions for harmful content. To exploit this bias in Judge LLMs, we introduce the Emoji Attack -- a method that places emojis within tokens to increase the embedding differences between sub-tokens and their originals. These emojis create new tokens that further distort the token embeddings, exacerbating the bias. To counter the Emoji Attack, we design prompts that help LLMs filter out unusual characters. However, this defense can still be bypassed by using a mix of emojis and other characters. The Emoji Attack can also be combined with existing jailbreaking prompts using few-shot learning, which enables LLMs to generate harmful responses with emojis. These responses are often mistakenly labeled as "safe" by Judge LLMs, allowing the attack to slip through. Our experiments with six state-of-the-art Judge LLMs show that the Emoji Attack allows 25\% of harmful responses to bypass detection by Llama Guard and Llama Guard 2, and up to 75\% by ShieldLM. These results highlight the need for stronger Judge LLMs to address this vulnerability.

摘要: 越狱攻击显示了大型语言模型(LLM)如何被欺骗，使用恶意提示生成有害的输出。为了防止这些攻击，经常使用其他LLM作为法官来评估生成的内容的危害性。然而，依赖LLMS作为法官可能会在检测过程中引入偏差，这反过来又会损害评估的有效性。在本文中，我们发现，与其他LLM一样，JUSITY LLM也会受到标记分割偏差的影响。当令牌被拆分成更小的子令牌，改变它们的嵌入时，就会发生这种偏差。这使得该模型更难检测有害内容。具体地说，这种偏差可能会导致嵌入空间中的子令牌与原始令牌显著不同，从而导致对有害内容的不正确的“安全”预测。为了利用裁判LLMS中的这种偏见，我们引入了Emoji攻击--一种将表情符号放入令牌中以增加子令牌与其原始表情之间的嵌入差异的方法。这些表情符号创造了新的令牌，进一步扭曲了令牌嵌入，加剧了偏见。为了应对Emoji攻击，我们设计了帮助LLMS过滤不寻常字符的提示。然而，通过混合使用表情符号和其他字符，仍然可以绕过这种防御。表情符号攻击还可以与现有的越狱提示结合使用，这使得LLMS能够使用表情符号生成有害的响应。这些反应经常被LLMS法官错误地贴上“安全”的标签，从而让攻击得以通过。我们使用六个最先进的JUARY LLM进行的实验表明，对于骆驼卫士和骆驼卫士2的旁路检测，Emoji攻击允许25%的有害响应，而ShieldLM允许高达75%的有害响应。这些结果突显了需要更强大的法官LLMS来解决这一漏洞。



## **21. FedDTPT: Federated Discrete and Transferable Prompt Tuning for Black-Box Large Language Models**

FedDTPT：黑箱大型语言模型的联邦离散和可传输提示调优 cs.CL

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.00985v1) [paper-pdf](http://arxiv.org/pdf/2411.00985v1)

**Authors**: Jiaqi Wu, Simin Chen, Yuzhe Yang, Yijiang Li, Shiyue Hou, Rui Jing, Zehua Wang, Wei Chen, Zijian Tian

**Abstract**: In recent years, large language models (LLMs) have significantly advanced the field of natural language processing (NLP). By fine-tuning LLMs with data from specific scenarios, these foundation models can better adapt to various downstream tasks. However, the fine-tuning process poses privacy leakage risks, particularly in centralized data processing scenarios. To address user privacy concerns, federated learning (FL) has been introduced to mitigate the risks associated with centralized data collection from multiple sources. Nevertheless, the privacy of LLMs themselves is equally critical, as potential malicious attacks challenge their security, an issue that has received limited attention in current research. Consequently, establishing a trusted multi-party model fine-tuning environment is essential. Additionally, the local deployment of large LLMs incurs significant storage costs and high computational demands. To address these challenges, we propose for the first time a federated discrete and transferable prompt tuning, namely FedDTPT, for black-box large language models. In the client optimization phase, we adopt a token-level discrete prompt optimization method that leverages a feedback loop based on prediction accuracy to drive gradient-free prompt optimization through the MLM API. For server optimization, we employ an attention mechanism based on semantic similarity to filter all local prompt tokens, along with an embedding distance elbow detection and DBSCAN clustering strategy to enhance the filtering process. Experimental results demonstrate that, compared to state-of-the-art methods, our approach achieves higher accuracy, reduced communication overhead, and robustness to non-iid data in a black-box setting. Moreover, the optimized prompts are transferable.

摘要: 近年来，大语言模型(LLM)显著地推动了自然语言处理(NLP)领域的发展。通过使用特定场景的数据微调LLM，这些基础模型可以更好地适应各种下游任务。然而，微调过程会带来隐私泄露风险，特别是在集中式数据处理场景中。为了解决用户隐私问题，引入了联合学习(FL)来降低从多个来源集中收集数据的风险。然而，LLMS本身的隐私也同样关键，因为潜在的恶意攻击挑战了它们的安全，这一问题在当前的研究中得到的关注有限。因此，建立可信的多方模型微调环境至关重要。此外，大型LLM的本地部署会产生巨大的存储成本和高计算需求。为了应对这些挑战，我们首次提出了一种适用于黑盒大语言模型的联邦离散和可转移的提示调优，即FedDTPT。在客户端优化阶段，我们采用令牌级离散提示优化方法，利用基于预测精度的反馈循环，通过传销API驱动无梯度提示优化。在服务器优化方面，我们采用了基于语义相似度的注意机制来过滤所有本地提示令牌，并使用嵌入距离弯头检测和DBSCAN聚类策略来增强过滤过程。实验结果表明，与现有方法相比，该方法具有更高的准确率、更低的通信开销和对黑盒环境下非IID数据的稳健性。此外，优化后的提示具有可转移性。



## **22. Efficient Adversarial Training in LLMs with Continuous Attacks**

在具有持续攻击的LLC中进行有效的对抗训练 cs.LG

19 pages, 4 figures

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2405.15589v3) [paper-pdf](http://arxiv.org/pdf/2405.15589v3)

**Authors**: Sophie Xhonneux, Alessandro Sordoni, Stephan Günnemann, Gauthier Gidel, Leo Schwinn

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that can bypass their safety guardrails. In many domains, adversarial training has proven to be one of the most promising methods to reliably improve robustness against such attacks. Yet, in the context of LLMs, current methods for adversarial training are hindered by the high computational costs required to perform discrete adversarial attacks at each training iteration. We address this problem by instead calculating adversarial attacks in the continuous embedding space of the LLM, which is orders of magnitudes more efficient. We propose a fast adversarial training algorithm (C-AdvUL) composed of two losses: the first makes the model robust on continuous embedding attacks computed on an adversarial behaviour dataset; the second ensures the usefulness of the final model by fine-tuning on utility data. Moreover, we introduce C-AdvIPO, an adversarial variant of IPO that does not require utility data for adversarially robust alignment. Our empirical evaluation on five models from different families (Gemma, Phi3, Mistral, Zephyr, Llama2) and at different scales (2B, 3.8B, 7B) shows that both algorithms substantially enhance LLM robustness against discrete attacks (GCG, AutoDAN, PAIR), while maintaining utility. Our results demonstrate that robustness to continuous perturbations can extrapolate to discrete threat models. Thereby, we present a path toward scalable adversarial training algorithms for robustly aligning LLMs.

摘要: 大型语言模型(LLM)容易受到敌意攻击，这些攻击可以绕过它们的安全护栏。在许多领域，对抗性训练已被证明是可靠地提高对此类攻击的稳健性的最有前途的方法之一。然而，在LLMS的背景下，当前的对抗性训练方法受到在每次训练迭代中执行离散对抗性攻击所需的高计算成本的阻碍。我们通过在LLM的连续嵌入空间中计算敌意攻击来解决这个问题，该空间的效率要高出几个数量级。我们提出了一种快速对抗性训练算法(C-AdvUL)，该算法由两个损失组成：第一个损失使模型对基于对抗行为数据集的连续嵌入攻击具有健壮性；第二个损失通过对效用数据的微调来确保最终模型的有效性。此外，我们引入了C-AdvIPO，这是IPO的一个对抗性变体，不需要效用数据来进行对抗性强健的比对。我们对来自不同家族(Gema，Phi3，Mistral，Zephy，Llama2)和不同尺度(2B，3.8B，7B)的五个模型的经验评估表明，这两种算法在保持实用性的同时，显著增强了LLM对离散攻击(GCG，AutoDAN，Pair)的稳健性。我们的结果表明，对连续扰动的稳健性可以外推到离散威胁模型。因此，我们提出了一条可扩展的对抗性训练算法，用于鲁棒对齐LLM。



## **23. Intruding with Words: Towards Understanding Graph Injection Attacks at the Text Level**

文字入侵：在文本层面了解图注入攻击 cs.LG

Accepted by NeurIPS 2024

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2405.16405v2) [paper-pdf](http://arxiv.org/pdf/2405.16405v2)

**Authors**: Runlin Lei, Yuwei Hu, Yuchen Ren, Zhewei Wei

**Abstract**: Graph Neural Networks (GNNs) excel across various applications but remain vulnerable to adversarial attacks, particularly Graph Injection Attacks (GIAs), which inject malicious nodes into the original graph and pose realistic threats. Text-attributed graphs (TAGs), where nodes are associated with textual features, are crucial due to their prevalence in real-world applications and are commonly used to evaluate these vulnerabilities. However, existing research only focuses on embedding-level GIAs, which inject node embeddings rather than actual textual content, limiting their applicability and simplifying detection. In this paper, we pioneer the exploration of GIAs at the text level, presenting three novel attack designs that inject textual content into the graph. Through theoretical and empirical analysis, we demonstrate that text interpretability, a factor previously overlooked at the embedding level, plays a crucial role in attack strength. Among the designs we investigate, the Word-frequency-based Text-level GIA (WTGIA) is particularly notable for its balance between performance and interpretability. Despite the success of WTGIA, we discover that defenders can easily enhance their defenses with customized text embedding methods or large language model (LLM)--based predictors. These insights underscore the necessity for further research into the potential and practical significance of text-level GIAs.

摘要: 图神经网络(GNN)在各种应用中表现出色，但仍然容易受到对手攻击，特别是图注入攻击(GIA)，图注入攻击将恶意节点注入到原始图中，并构成现实威胁。文本属性图(TAG)将节点与文本特征相关联，由于它们在现实应用程序中的普遍存在，因此至关重要，并且通常用于评估这些漏洞。然而，现有的研究只关注嵌入级GIA，这些GIA注入的是节点嵌入而不是实际的文本内容，限制了它们的适用性，简化了检测。在本文中，我们率先在文本层面上探索了GIA，提出了三种向图形中注入文本内容的新颖攻击设计。通过理论和实证分析，我们证明了文本可解释性对攻击强度起着至关重要的作用，而文本可解释性是此前在嵌入层面被忽视的一个因素。在我们研究的设计中，基于词频的文本级别GIA(WTGIA)特别值得注意的是它在性能和可解释性之间的平衡。尽管WTGIA取得了成功，但我们发现，防御者可以很容易地通过定制的文本嵌入方法或基于大型语言模型(LLM)的预测器来增强他们的防御。这些见解突显了进一步研究文本层面全球影响的潜力和现实意义的必要性。



## **24. Defense Against Prompt Injection Attack by Leveraging Attack Techniques**

利用攻击技术防御即时注入攻击 cs.CR

9 pages

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.00459v1) [paper-pdf](http://arxiv.org/pdf/2411.00459v1)

**Authors**: Yulin Chen, Haoran Li, Zihao Zheng, Yangqiu Song, Dekai Wu, Bryan Hooi

**Abstract**: With the advancement of technology, large language models (LLMs) have achieved remarkable performance across various natural language processing (NLP) tasks, powering LLM-integrated applications like Microsoft Copilot. However, as LLMs continue to evolve, new vulnerabilities, especially prompt injection attacks arise. These attacks trick LLMs into deviating from the original input instructions and executing the attacker's instructions injected in data content, such as retrieved results. Recent attack methods leverage LLMs' instruction-following abilities and their inabilities to distinguish instructions injected in the data content, and achieve a high attack success rate (ASR). When comparing the attack and defense methods, we interestingly find that they share similar design goals, of inducing the model to ignore unwanted instructions and instead to execute wanted instructions. Therefore, we raise an intuitive question: Could these attack techniques be utilized for defensive purposes? In this paper, we invert the intention of prompt injection methods to develop novel defense methods based on previous training-free attack methods, by repeating the attack process but with the original input instruction rather than the injected instruction. Our comprehensive experiments demonstrate that our defense techniques outperform existing training-free defense approaches, achieving state-of-the-art results.

摘要: 随着技术的进步，大语言模型(LLM)在各种自然语言处理(NLP)任务中取得了显著的性能，支持Microsoft Copilot等LLM集成应用程序。然而，随着LLMS的不断发展，出现了新的漏洞，特别是即时注入攻击。这些攻击欺骗LLM偏离原始输入指令，并执行注入数据内容的攻击者指令，例如检索的结果。最近的攻击方法利用LLMS的指令跟随能力和它们无法区分注入到数据内容中的指令的能力，实现了高攻击成功率(ASR)。当比较攻击和防御方法时，我们有趣地发现它们有相似的设计目标，都是诱导模型忽略不想要的指令，而是执行想要的指令。因此，我们提出了一个直观的问题：这些攻击技术是否可以用于防御目的？在本文中，我们反转了快速注入方法的意图，在以前的免训练攻击方法的基础上，通过重复攻击过程来开发新的防御方法，但使用的是原始输入指令而不是注入指令。我们的综合实验表明，我们的防御技术优于现有的免训练防御方法，取得了最先进的结果。



## **25. Attention Tracker: Detecting Prompt Injection Attacks in LLMs**

注意力追踪器：检测LLM中的即时注入攻击 cs.CR

Project page:  https://huggingface.co/spaces/TrustSafeAI/Attention-Tracker

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.00348v1) [paper-pdf](http://arxiv.org/pdf/2411.00348v1)

**Authors**: Kuo-Han Hung, Ching-Yun Ko, Ambrish Rawat, I-Hsin Chung, Winston H. Hsu, Pin-Yu Chen

**Abstract**: Large Language Models (LLMs) have revolutionized various domains but remain vulnerable to prompt injection attacks, where malicious inputs manipulate the model into ignoring original instructions and executing designated action. In this paper, we investigate the underlying mechanisms of these attacks by analyzing the attention patterns within LLMs. We introduce the concept of the distraction effect, where specific attention heads, termed important heads, shift focus from the original instruction to the injected instruction. Building on this discovery, we propose Attention Tracker, a training-free detection method that tracks attention patterns on instruction to detect prompt injection attacks without the need for additional LLM inference. Our method generalizes effectively across diverse models, datasets, and attack types, showing an AUROC improvement of up to 10.0% over existing methods, and performs well even on small LLMs. We demonstrate the robustness of our approach through extensive evaluations and provide insights into safeguarding LLM-integrated systems from prompt injection vulnerabilities.

摘要: 大型语言模型(LLM)给各个领域带来了革命性的变化，但仍然容易受到即时注入攻击，恶意输入会操纵模型忽略原始指令并执行指定的操作。在本文中，我们通过分析LLMS中的注意模式来研究这些攻击的潜在机制。我们引入了分心效应的概念，即特定的注意力头部，称为重要头部，将焦点从原始指令转移到注入的指令。基于这一发现，我们提出了注意力跟踪器，这是一种无需训练的检测方法，它跟踪指令上的注意模式，检测即时注入攻击，而不需要额外的LLM推理。我们的方法有效地概括了不同的模型、数据集和攻击类型，显示出比现有方法高达10.0%的AUROC改进，即使在小的LLM上也表现得很好。我们通过广泛的评估展示了我们方法的健壮性，并为保护LLM集成系统免受即时注入漏洞的攻击提供了见解。



## **26. JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models**

越狱长凳：越狱大型语言模型的开放鲁棒性基准 cs.CR

The camera-ready version of JailbreakBench v1.0 (accepted at NeurIPS  2024 Datasets and Benchmarks Track): more attack artifacts, more test-time  defenses, a more accurate jailbreak judge (Llama-3-70B with a custom prompt),  a larger dataset of human preferences for selecting a jailbreak judge (300  examples), an over-refusal evaluation dataset, a semantic refusal judge based  on Llama-3-8B

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2404.01318v5) [paper-pdf](http://arxiv.org/pdf/2404.01318v5)

**Authors**: Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, Eric Wong

**Abstract**: Jailbreak attacks cause large language models (LLMs) to generate harmful, unethical, or otherwise objectionable content. Evaluating these attacks presents a number of challenges, which the current collection of benchmarks and evaluation techniques do not adequately address. First, there is no clear standard of practice regarding jailbreaking evaluation. Second, existing works compute costs and success rates in incomparable ways. And third, numerous works are not reproducible, as they withhold adversarial prompts, involve closed-source code, or rely on evolving proprietary APIs. To address these challenges, we introduce JailbreakBench, an open-sourced benchmark with the following components: (1) an evolving repository of state-of-the-art adversarial prompts, which we refer to as jailbreak artifacts; (2) a jailbreaking dataset comprising 100 behaviors -- both original and sourced from prior work (Zou et al., 2023; Mazeika et al., 2023, 2024) -- which align with OpenAI's usage policies; (3) a standardized evaluation framework at https://github.com/JailbreakBench/jailbreakbench that includes a clearly defined threat model, system prompts, chat templates, and scoring functions; and (4) a leaderboard at https://jailbreakbench.github.io/ that tracks the performance of attacks and defenses for various LLMs. We have carefully considered the potential ethical implications of releasing this benchmark, and believe that it will be a net positive for the community.

摘要: 越狱攻击会导致大型语言模型(LLM)生成有害、不道德或令人反感的内容。评估这些攻击带来了许多挑战，目前收集的基准和评估技术没有充分解决这些挑战。首先，关于越狱评估没有明确的实践标准。其次，现有的工作以无与伦比的方式计算成本和成功率。第三，许多作品是不可复制的，因为它们保留了对抗性提示，涉及封闭源代码，或者依赖于不断发展的专有API。为了应对这些挑战，我们引入了JailBreak，这是一个开源基准，具有以下组件：(1)一个不断发展的最先进对手提示的储存库，我们称之为越狱人工制品；(2)一个包括100种行为的越狱数据集--既有原始的，也有源自先前工作的(邹某等人，2023年；Mazeika等人，2023年，2024年)--与开放人工智能的使用政策保持一致；(3)https://github.com/JailbreakBench/jailbreakbench的标准化评估框架，包括明确定义的威胁模型、系统提示、聊天模板和评分功能；以及(4)https://jailbreakbench.github.io/的排行榜，跟踪各种LLM的攻击和防御性能。我们已仔细考虑发布这一基准的潜在道德影响，并相信它将为社会带来净积极的影响。



## **27. Scaling Up Membership Inference: When and How Attacks Succeed on Large Language Models**

扩大成员资格推理：攻击何时以及如何在大型语言模型上取得成功 cs.CL

Our code is available at https://github.com/parameterlab/mia-scaling

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2411.00154v1) [paper-pdf](http://arxiv.org/pdf/2411.00154v1)

**Authors**: Haritz Puerto, Martin Gubri, Sangdoo Yun, Seong Joon Oh

**Abstract**: Membership inference attacks (MIA) attempt to verify the membership of a given data sample in the training set for a model. MIA has become relevant in recent years, following the rapid development of large language models (LLM). Many are concerned about the usage of copyrighted materials for training them and call for methods for detecting such usage. However, recent research has largely concluded that current MIA methods do not work on LLMs. Even when they seem to work, it is usually because of the ill-designed experimental setup where other shortcut features enable "cheating." In this work, we argue that MIA still works on LLMs, but only when multiple documents are presented for testing. We construct new benchmarks that measure the MIA performances at a continuous scale of data samples, from sentences (n-grams) to a collection of documents (multiple chunks of tokens). To validate the efficacy of current MIA approaches at greater scales, we adapt a recent work on Dataset Inference (DI) for the task of binary membership detection that aggregates paragraph-level MIA features to enable MIA at document and collection of documents level. This baseline achieves the first successful MIA on pre-trained and fine-tuned LLMs.

摘要: 成员关系推理攻击(MIA)试图验证给定数据样本在模型训练集中的成员资格。近年来，随着大型语言模型(LLM)的快速发展，MIA变得相关起来。许多人担心使用受版权保护的材料来培训他们，并呼吁采取方法来检测这种使用情况。然而，最近的研究在很大程度上得出结论，目前的MIA方法不适用于LLMS。即使它们看起来很有效，这通常也是因为设计糟糕的实验设置，其他快捷功能允许“作弊”。在这项工作中，我们认为MIA仍然适用于LLMS，但只有在提交多个文档进行测试时才能使用。我们构建了新的基准来衡量MIA在连续规模的数据样本上的性能，从句子(n-gram)到文档集合(多个令牌块)。为了在更大范围内验证当前MIA方法的有效性，我们对最近在数据集推理(DI)方面的工作进行了调整，以用于二元成员关系检测任务，该任务聚集了段级MIA特征，以支持文档和文档集合级别的MIA。这一基线在预先训练和微调的LLM上实现了第一次成功的MIA。



## **28. Tree of Attacks: Jailbreaking Black-Box LLMs Automatically**

攻击树：自动越狱黑匣子LLM cs.LG

Accepted for presentation at NeurIPS 2024. Code:  https://github.com/RICommunity/TAP

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2312.02119v3) [paper-pdf](http://arxiv.org/pdf/2312.02119v3)

**Authors**: Anay Mehrotra, Manolis Zampetakis, Paul Kassianik, Blaine Nelson, Hyrum Anderson, Yaron Singer, Amin Karbasi

**Abstract**: While Large Language Models (LLMs) display versatile functionality, they continue to generate harmful, biased, and toxic content, as demonstrated by the prevalence of human-designed jailbreaks. In this work, we present Tree of Attacks with Pruning (TAP), an automated method for generating jailbreaks that only requires black-box access to the target LLM. TAP utilizes an attacker LLM to iteratively refine candidate (attack) prompts until one of the refined prompts jailbreaks the target. In addition, before sending prompts to the target, TAP assesses them and prunes the ones unlikely to result in jailbreaks, reducing the number of queries sent to the target LLM. In empirical evaluations, we observe that TAP generates prompts that jailbreak state-of-the-art LLMs (including GPT4-Turbo and GPT4o) for more than 80% of the prompts. This significantly improves upon the previous state-of-the-art black-box methods for generating jailbreaks while using a smaller number of queries than them. Furthermore, TAP is also capable of jailbreaking LLMs protected by state-of-the-art guardrails, e.g., LlamaGuard.

摘要: 虽然大型语言模型(LLM)显示了多功能，但它们继续产生有害、有偏见和有毒的内容，人类设计的越狱事件的流行就证明了这一点。在这项工作中，我们提出了带修剪的攻击树(TAP)，这是一种自动生成越狱的方法，只需要通过黑盒访问目标LLM。TAP利用攻击者的LLM反复细化候选(攻击)提示，直到其中一个细化的提示破解目标。此外，在向目标发送提示之前，TAP会对它们进行评估，并删除不太可能导致越狱的提示，从而减少发送到目标LLM的查询数量。在实证评估中，我们观察到TAP为80%以上的提示生成了越狱最先进的LLM(包括GPT4-Turbo和GPT4o)提示。与以前最先进的黑盒方法相比，这大大改进了生成越狱的方法，同时使用的查询数量比它们少。此外，TAP还能够越狱由最先进的护栏保护的LLMS，例如LlamaGuard。



## **29. Fight Back Against Jailbreaking via Prompt Adversarial Tuning**

通过即时对抗调整反击越狱 cs.LG

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2402.06255v4) [paper-pdf](http://arxiv.org/pdf/2402.06255v4)

**Authors**: Yichuan Mo, Yuji Wang, Zeming Wei, Yisen Wang

**Abstract**: While Large Language Models (LLMs) have achieved tremendous success in various applications, they are also susceptible to jailbreaking attacks. Several primary defense strategies have been proposed to protect LLMs from producing harmful information, mostly focusing on model fine-tuning or heuristical defense designs. However, how to achieve intrinsic robustness through prompt optimization remains an open problem. In this paper, motivated by adversarial training paradigms for achieving reliable robustness, we propose an approach named Prompt Adversarial Tuning (PAT) that trains a prompt control attached to the user prompt as a guard prefix. To achieve our defense goal whilst maintaining natural performance, we optimize the control prompt with both adversarial and benign prompts. Comprehensive experiments show that our method is effective against both grey-box and black-box attacks, reducing the success rate of advanced attacks to nearly 0%, while maintaining the model's utility on the benign task and incurring only negligible computational overhead, charting a new perspective for future explorations in LLM security. Our code is available at https://github.com/PKU-ML/PAT.

摘要: 虽然大型语言模型(LLM)在各种应用中取得了巨大的成功，但它们也容易受到越狱攻击。已经提出了几种主要的防御策略来保护LLMS免受有害信息的影响，主要集中在模型微调或启发式防御设计上。然而，如何通过快速优化来获得内在的稳健性仍然是一个悬而未决的问题。受实现可靠健壮性的对抗性训练范例的启发，我们提出了一种称为即时对抗性调整(PAT)的方法，该方法将附加在用户提示上的提示控制训练为保卫前缀。为了在保持自然表现的同时实现我们的防守目标，我们优化了控制提示，包括对抗性提示和良性提示。综合实验表明，该方法对灰盒和黑盒攻击都是有效的，将高级攻击的成功率降低到近0%，同时保持了模型在良性任务上的效用，并且只产生了可以忽略的计算开销，为LLM安全的进一步研究开辟了新的视角。我们的代码可以在https://github.com/PKU-ML/PAT.上找到



## **30. Audio Is the Achilles' Heel: Red Teaming Audio Large Multimodal Models**

音频是阿喀琉斯之踵：红色团队音频大型多模式 cs.CL

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.23861v1) [paper-pdf](http://arxiv.org/pdf/2410.23861v1)

**Authors**: Hao Yang, Lizhen Qu, Ehsan Shareghi, Gholamreza Haffari

**Abstract**: Large Multimodal Models (LMMs) have demonstrated the ability to interact with humans under real-world conditions by combining Large Language Models (LLMs) and modality encoders to align multimodal information (visual and auditory) with text. However, such models raise new safety challenges of whether models that are safety-aligned on text also exhibit consistent safeguards for multimodal inputs. Despite recent safety-alignment research on vision LMMs, the safety of audio LMMs remains under-explored. In this work, we comprehensively red team the safety of five advanced audio LMMs under three settings: (i) harmful questions in both audio and text formats, (ii) harmful questions in text format accompanied by distracting non-speech audio, and (iii) speech-specific jailbreaks. Our results under these settings demonstrate that open-source audio LMMs suffer an average attack success rate of 69.14% on harmful audio questions, and exhibit safety vulnerabilities when distracted with non-speech audio noise. Our speech-specific jailbreaks on Gemini-1.5-Pro achieve an attack success rate of 70.67% on the harmful query benchmark. We provide insights on what could cause these reported safety-misalignments. Warning: this paper contains offensive examples.

摘要: 大型多通道模型(LMM)通过将大型语言模型(LLM)和通道编码器相结合来将多通道信息(视觉和听觉)与文本对齐，从而展示了在真实世界条件下与人类交互的能力。然而，这样的模型提出了新的安全挑战，即在文本上与安全一致的模型是否也显示出对多模式输入的一致保障。尽管最近对视觉LMM的安全性进行了研究，但音频LMM的安全性仍未得到充分的探索。在这项工作中，我们在三种设置下对五种高级音频LMM的安全性进行了全面的红色团队：(I)音频和文本格式的有害问题，(Ii)伴随着令人分心的非语音音频的文本格式的有害问题，以及(Iii)特定于语音的越狱。实验结果表明，在这种情况下，开源音频LMM对有害音频问题的平均攻击成功率为69.14%，在非语音噪声干扰下表现出安全漏洞。我们在Gemini-1.5-Pro上的语音特定越狱在有害查询基准上实现了70.67%的攻击成功率。我们提供了可能导致这些报告的安全错位的原因的见解。警告：本文包含令人反感的例子。



## **31. DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios**

DetectRL：在现实世界场景中对LLM生成的文本检测进行基准测试 cs.CL

Accepted to NeurIPS 2024 Dataset & Benchmarking Track

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.23746v1) [paper-pdf](http://arxiv.org/pdf/2410.23746v1)

**Authors**: Junchao Wu, Runzhe Zhan, Derek F. Wong, Shu Yang, Xinyi Yang, Yulin Yuan, Lidia S. Chao

**Abstract**: Detecting text generated by large language models (LLMs) is of great recent interest. With zero-shot methods like DetectGPT, detection capabilities have reached impressive levels. However, the reliability of existing detectors in real-world applications remains underexplored. In this study, we present a new benchmark, DetectRL, highlighting that even state-of-the-art (SOTA) detection techniques still underperformed in this task. We collected human-written datasets from domains where LLMs are particularly prone to misuse. Using popular LLMs, we generated data that better aligns with real-world applications. Unlike previous studies, we employed heuristic rules to create adversarial LLM-generated text, simulating advanced prompt usages, human revisions like word substitutions, and writing errors. Our development of DetectRL reveals the strengths and limitations of current SOTA detectors. More importantly, we analyzed the potential impact of writing styles, model types, attack methods, the text lengths, and real-world human writing factors on different types of detectors. We believe DetectRL could serve as an effective benchmark for assessing detectors in real-world scenarios, evolving with advanced attack methods, thus providing more stressful evaluation to drive the development of more efficient detectors. Data and code are publicly available at: https://github.com/NLP2CT/DetectRL.

摘要: 检测由大型语言模型(LLM)生成的文本是最近非常感兴趣的问题。有了像DetectGPT这样的零射击方法，检测能力已经达到了令人印象深刻的水平。然而，现有探测器在实际应用中的可靠性仍然没有得到充分的探索。在这项研究中，我们提出了一个新的基准，DetectRL，强调即使是最先进的(SOTA)检测技术在这项任务中仍然表现不佳。我们从LLM特别容易被滥用的领域收集了人类编写的数据集。使用流行的LLM，我们生成的数据更好地与现实世界的应用程序保持一致。与以前的研究不同，我们使用启发式规则来创建对抗性LLM生成的文本，模拟高级提示用法、人工修改(如单词替换)和书写错误。我们对DetectRL的开发揭示了当前SOTA探测器的优势和局限性。更重要的是，我们分析了写作风格、模型类型、攻击方法、文本长度和真实世界中的人类写作因素对不同类型检测器的潜在影响。我们相信，DetectRL可以作为评估真实世界场景中检测器的有效基准，随着先进攻击方法的发展，从而提供更有压力的评估，以推动更高效检测器的开发。数据和代码可在以下网址公开获得：https://github.com/NLP2CT/DetectRL.



## **32. Adversarial Attacks of Vision Tasks in the Past 10 Years: A Survey**

过去10年视觉任务的对抗性攻击：一项调查 cs.CV

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.23687v1) [paper-pdf](http://arxiv.org/pdf/2410.23687v1)

**Authors**: Chiyu Zhang, Xiaogang Xu, Jiafei Wu, Zhe Liu, Lu Zhou

**Abstract**: Adversarial attacks, which manipulate input data to undermine model availability and integrity, pose significant security threats during machine learning inference. With the advent of Large Vision-Language Models (LVLMs), new attack vectors, such as cognitive bias, prompt injection, and jailbreak techniques, have emerged. Understanding these attacks is crucial for developing more robust systems and demystifying the inner workings of neural networks. However, existing reviews often focus on attack classifications and lack comprehensive, in-depth analysis. The research community currently needs: 1) unified insights into adversariality, transferability, and generalization; 2) detailed evaluations of existing methods; 3) motivation-driven attack categorizations; and 4) an integrated perspective on both traditional and LVLM attacks. This article addresses these gaps by offering a thorough summary of traditional and LVLM adversarial attacks, emphasizing their connections and distinctions, and providing actionable insights for future research.

摘要: 对抗性攻击通过操纵输入数据来破坏模型的可用性和完整性，在机器学习推理过程中会造成严重的安全威胁。随着大型视觉语言模型的出现，新的攻击载体出现了，如认知偏差、快速注入和越狱技术。了解这些攻击对于开发更强大的系统和揭开神经网络内部工作的神秘面纱至关重要。然而，现有的审查往往侧重于攻击分类，缺乏全面、深入的分析。研究界目前需要：1)对对抗性、可转移性和泛化的统一见解；2)对现有方法的详细评估；3)动机驱动的攻击分类；以及4)对传统攻击和LVLM攻击的综合视角。本文对传统攻击和LVLM攻击进行了全面的总结，强调了它们之间的联系和区别，并为未来的研究提供了可操作的见解，从而解决了这些差距。



## **33. Pseudo-Conversation Injection for LLM Goal Hijacking**

LLM目标劫持的伪对话注入 cs.CL

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.23678v1) [paper-pdf](http://arxiv.org/pdf/2410.23678v1)

**Authors**: Zheng Chen, Buhui Yao

**Abstract**: Goal hijacking is a type of adversarial attack on Large Language Models (LLMs) where the objective is to manipulate the model into producing a specific, predetermined output, regardless of the user's original input. In goal hijacking, an attacker typically appends a carefully crafted malicious suffix to the user's prompt, which coerces the model into ignoring the user's original input and generating the target response. In this paper, we introduce a novel goal hijacking attack method called Pseudo-Conversation Injection, which leverages the weaknesses of LLMs in role identification within conversation contexts. Specifically, we construct the suffix by fabricating responses from the LLM to the user's initial prompt, followed by a prompt for a malicious new task. This leads the model to perceive the initial prompt and fabricated response as a completed conversation, thereby executing the new, falsified prompt. Following this approach, we propose three Pseudo-Conversation construction strategies: Targeted Pseudo-Conversation, Universal Pseudo-Conversation, and Robust Pseudo-Conversation. These strategies are designed to achieve effective goal hijacking across various scenarios. Our experiments, conducted on two mainstream LLM platforms including ChatGPT and Qwen, demonstrate that our proposed method significantly outperforms existing approaches in terms of attack effectiveness.

摘要: 目标劫持是一种针对大型语言模型(LLM)的对抗性攻击，其目标是操纵模型生成特定的、预定的输出，而不考虑用户的原始输入。在目标劫持中，攻击者通常会在用户提示后附加精心编制的恶意后缀，这会迫使模型忽略用户的原始输入并生成目标响应。本文提出了一种新的目标劫持攻击方法--伪会话注入，该方法利用了LLMS在会话上下文中角色识别方面的弱点。具体地说，我们构造后缀的方法是从LLM构造对用户初始提示的响应，然后是恶意新任务的提示。这导致模型将初始提示和捏造的响应视为完成的对话，从而执行新的、伪造的提示。在此基础上，我们提出了三种伪会话构建策略：目标伪会话、通用伪会话和健壮伪会话。这些策略旨在实现跨各种场景的有效目标劫持。我们在ChatGPT和Qwen两个主流LLM平台上进行的实验表明，我们提出的方法在攻击效率方面明显优于现有方法。



## **34. Adversarial Attacks on Code Models with Discriminative Graph Patterns**

对具有区分图模式的代码模型的对抗攻击 cs.SE

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2308.11161v2) [paper-pdf](http://arxiv.org/pdf/2308.11161v2)

**Authors**: Thanh-Dat Nguyen, Yang Zhou, Xuan Bach D. Le, Patanamon Thongtanunam, David Lo

**Abstract**: Pre-trained language models of code are now widely used in various software engineering tasks such as code generation, code completion, vulnerability detection, etc. This, in turn, poses security and reliability risks to these models. One of the important threats is \textit{adversarial attacks}, which can lead to erroneous predictions and largely affect model performance on downstream tasks. Current adversarial attacks on code models usually adopt fixed sets of program transformations, such as variable renaming and dead code insertion, leading to limited attack effectiveness. To address the aforementioned challenges, we propose a novel adversarial attack framework, GraphCodeAttack, to better evaluate the robustness of code models. Given a target code model, GraphCodeAttack automatically mines important code patterns, which can influence the model's decisions, to perturb the structure of input code to the model. To do so, GraphCodeAttack uses a set of input source codes to probe the model's outputs and identifies the \textit{discriminative} ASTs patterns that can influence the model decisions. GraphCodeAttack then selects appropriate AST patterns, concretizes the selected patterns as attacks, and inserts them as dead code into the model's input program. To effectively synthesize attacks from AST patterns, GraphCodeAttack uses a separate pre-trained code model to fill in the ASTs with concrete code snippets. We evaluate the robustness of two popular code models (e.g., CodeBERT and GraphCodeBERT) against our proposed approach on three tasks: Authorship Attribution, Vulnerability Prediction, and Clone Detection. The experimental results suggest that our proposed approach significantly outperforms state-of-the-art approaches in attacking code models such as CARROT and ALERT.

摘要: 预先训练的代码语言模型现在被广泛用于各种软件工程任务，如代码生成、代码完成、漏洞检测等。这反过来又给这些模型带来了安全和可靠性风险。其中一个重要的威胁是对抗性攻击，它会导致错误的预测，并在很大程度上影响模型在下游任务上的性能。当前针对代码模型的对抗性攻击通常采用固定的程序转换集，如变量重命名和死代码插入，导致攻击效果有限。为了应对上述挑战，我们提出了一种新的对抗性攻击框架GraphCodeAttack，以更好地评估代码模型的健壮性。在给定目标代码模型的情况下，GraphCodeAttack自动挖掘可能影响模型决策的重要代码模式，以扰乱模型的输入代码结构。为此，GraphCodeAttack使用一组输入源代码来探测模型的输出，并识别可能影响模型决策的\textit{鉴别性}ASTS模式。然后，GraphCodeAttack选择适当的AST模式，将所选模式具体化为攻击，并将它们作为死代码插入到模型的输入程序中。为了有效地从AST模式合成攻击，GraphCodeAttack使用单独的预先训练的代码模型来用具体的代码片段填充AST。我们评估了两个流行的代码模型(例如，CodeBERT和GraphCodeBERT)在作者属性、漏洞预测和克隆检测三个任务上的健壮性。实验结果表明，我们提出的方法在攻击胡萝卜和ALERT等代码模型方面明显优于最先进的方法。



## **35. Pruning for Protection: Increasing Jailbreak Resistance in Aligned LLMs Without Fine-Tuning**

修剪以保护：在无需微调的情况下提高对齐的LLM的越狱抵抗力 cs.LG

Proceedings of the 7th BlackboxNLP Workshop: Analyzing and  Interpreting Neural Networks for NLP

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2401.10862v3) [paper-pdf](http://arxiv.org/pdf/2401.10862v3)

**Authors**: Adib Hasan, Ileana Rugina, Alex Wang

**Abstract**: This paper investigates the impact of model compression on the way Large Language Models (LLMs) process prompts, particularly concerning jailbreak resistance. We show that moderate WANDA pruning can enhance resistance to jailbreaking attacks without fine-tuning, while maintaining performance on standard benchmarks. To systematically evaluate this safety enhancement, we introduce a dataset of 225 harmful tasks across five categories. Our analysis of LLaMA-2 Chat, Vicuna 1.3, and Mistral Instruct v0.2 reveals that pruning benefits correlate with initial model safety levels. We interpret these results by examining changes in attention patterns and perplexity shifts, demonstrating that pruned models exhibit sharper attention and increased sensitivity to artificial jailbreak constructs. We extend our evaluation to the AdvBench harmful behavior tasks and the GCG attack method. We find that LLaMA-2 is much safer on AdvBench prompts than on our dataset when evaluated with manual jailbreak attempts, and that pruning is effective against both automated attacks and manual jailbreaking on Advbench.

摘要: 本文研究了模型压缩对大型语言模型(LLMS)处理提示的方式的影响，特别是关于越狱抵抗的影响。我们表明，适度的万达剪枝可以在不进行微调的情况下增强对越狱攻击的抵抗力，同时保持标准基准测试的性能。为了系统地评估这一安全增强，我们引入了五个类别的225个有害任务的数据集。我们对骆驼-2聊天、维库纳1.3和米斯特拉尔指令v0.2的分析表明，修剪的好处与初始模型的安全级别相关。我们通过检测注意力模式和困惑转移的变化来解释这些结果，表明修剪后的模型表现出更敏锐的注意力和对人工越狱结构的敏感性。我们将我们的评估扩展到AdvBtch有害行为任务和GCG攻击方法。我们发现，当使用手动越狱尝试进行评估时，在AdvBtch提示上的骆驼-2比在我们的数据集上要安全得多，并且剪枝在Advbase上对自动攻击和手动越狱都是有效的。



## **36. Transferable Ensemble Black-box Jailbreak Attacks on Large Language Models**

可转移集成黑匣子越狱攻击大型语言模型 cs.CR

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.23558v1) [paper-pdf](http://arxiv.org/pdf/2410.23558v1)

**Authors**: Yiqi Yang, Hongye Fu

**Abstract**: In this report, we propose a novel black-box jailbreak attacking framework that incorporates various LLM-as-Attacker methods to deliver transferable and powerful jailbreak attacks. Our method is designed based on three key observations from existing jailbreaking studies and practices. First, we consider an ensemble approach should be more effective in exposing the vulnerabilities of an aligned LLM compared to individual attacks. Second, different malicious instructions inherently vary in their jailbreaking difficulty, necessitating differentiated treatment to ensure more efficient attacks. Finally, the semantic coherence of a malicious instruction is crucial for triggering the defenses of an aligned LLM; therefore, it must be carefully disrupted to manipulate its embedding representation, thereby increasing the jailbreak success rate. We validated our approach by participating in the Competition for LLM and Agent Safety 2024, where our team achieved top performance in the Jailbreaking Attack Track.

摘要: 在这份报告中，我们提出了一种新的黑盒越狱攻击框架，该框架结合了各种LLM作为攻击者的方法来提供可转移的强大越狱攻击。我们的方法是基于现有越狱研究和实践中的三个关键观察结果而设计的。首先，我们认为，与单独攻击相比，整体方法应该更有效地暴露联合LLM的漏洞。其次，不同的恶意指令在越狱难度上存在内在差异，需要区别对待，以确保更有效的攻击。最后，恶意指令的语义一致性对于触发对齐的LLM的防御至关重要；因此，必须小心破坏它才能操纵其嵌入表示，从而提高越狱成功率。我们通过参加LLM和代理安全竞赛2024来验证我们的方法，我们的团队在越狱攻击赛道上取得了最好的表现。



## **37. Representation Noising: A Defence Mechanism Against Harmful Finetuning**

代表噪音：防止有害微调的防御机制 cs.CL

Published in NeurIPs 2024

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2405.14577v4) [paper-pdf](http://arxiv.org/pdf/2405.14577v4)

**Authors**: Domenic Rosati, Jan Wehner, Kai Williams, Łukasz Bartoszcze, David Atanasov, Robie Gonzales, Subhabrata Majumdar, Carsten Maple, Hassan Sajjad, Frank Rudzicz

**Abstract**: Releasing open-source large language models (LLMs) presents a dual-use risk since bad actors can easily fine-tune these models for harmful purposes. Even without the open release of weights, weight stealing and fine-tuning APIs make closed models vulnerable to harmful fine-tuning attacks (HFAs). While safety measures like preventing jailbreaks and improving safety guardrails are important, such measures can easily be reversed through fine-tuning. In this work, we propose Representation Noising (RepNoise), a defence mechanism that operates even when attackers have access to the weights. RepNoise works by removing information about harmful representations such that it is difficult to recover them during fine-tuning. Importantly, our defence is also able to generalize across different subsets of harm that have not been seen during the defence process as long as they are drawn from the same distribution of the attack set. Our method does not degrade the general capability of LLMs and retains the ability to train the model on harmless tasks. We provide empirical evidence that the efficacy of our defence lies in its ``depth'': the degree to which information about harmful representations is removed across all layers of the LLM. We also find areas where RepNoise still remains ineffective and highlight how those limitations can inform future research.

摘要: 发布开源的大型语言模型(LLM)存在双重用途的风险，因为不好的参与者很容易出于有害目的微调这些模型。即使没有公开的权重释放，权重盗窃和微调API也会使封闭的模型容易受到有害的微调攻击(HFA)。虽然防止越狱和改善安全护栏等安全措施很重要，但通过微调很容易逆转这些措施。在这项工作中，我们提出了表示噪声(RepNoise)，这是一种即使攻击者可以访问权重也可以操作的防御机制。RepNoise的工作原理是删除有关有害表示的信息，以便在微调期间很难恢复它们。重要的是，我们的防御还能够概括在防御过程中未曾见过的伤害的不同子集，只要它们来自相同分布的攻击集。我们的方法不会降低LLMS的整体性能，并保留了对模型进行无害任务训练的能力。我们提供的经验证据表明，我们的辩护的效力在于它的“深度”：在法律法规的所有层面上，关于有害陈述的信息被删除的程度。我们还发现了RepNoise仍然无效的领域，并强调了这些限制如何为未来的研究提供信息。



## **38. ProTransformer: Robustify Transformers via Plug-and-Play Paradigm**

ProTransformer：通过即插即用范式的Robustify Transformers cs.LG

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.23182v1) [paper-pdf](http://arxiv.org/pdf/2410.23182v1)

**Authors**: Zhichao Hou, Weizhi Gao, Yuchen Shen, Feiyi Wang, Xiaorui Liu

**Abstract**: Transformer-based architectures have dominated various areas of machine learning in recent years. In this paper, we introduce a novel robust attention mechanism designed to enhance the resilience of transformer-based architectures. Crucially, this technique can be integrated into existing transformers as a plug-and-play layer, improving their robustness without the need for additional training or fine-tuning. Through comprehensive experiments and ablation studies, we demonstrate that our ProTransformer significantly enhances the robustness of transformer models across a variety of prediction tasks, attack mechanisms, backbone architectures, and data domains. Notably, without further fine-tuning, the ProTransformer consistently improves the performance of vanilla transformers by 19.5%, 28.3%, 16.1%, and 11.4% for BERT, ALBERT, DistilBERT, and RoBERTa, respectively, under the classical TextFooler attack. Furthermore, ProTransformer shows promising resilience in large language models (LLMs) against prompting-based attacks, improving the performance of T5 and LLaMA by 24.8% and 17.8%, respectively, and enhancing Vicuna by an average of 10.4% against the Jailbreaking attack. Beyond the language domain, ProTransformer also demonstrates outstanding robustness in both vision and graph domains.

摘要: 近年来，基于变压器的体系结构主导了机器学习的各个领域。在本文中，我们介绍了一种新的健壮注意机制，旨在增强基于变压器的体系结构的弹性。至关重要的是，这项技术可以作为即插即用层集成到现有的变压器中，无需额外的培训或微调即可提高其稳健性。通过全面的实验和烧蚀研究，我们证明我们的ProTransformer显著增强了变压器模型在各种预测任务、攻击机制、主干架构和数据域中的稳健性。值得注意的是，在没有进一步微调的情况下，ProTransformer在经典的TextFooler攻击下，分别将Bert、Albert、DistilBERT和Roberta的Vanilla变压器的性能分别提高了19.5%、28.3%、16.1%和11.4%。此外，ProTransformer在大型语言模型(LLM)中对基于提示的攻击表现出了良好的弹性，将T5和Llama的性能分别提高了24.8%和17.8%，对越狱攻击的维库纳平均提高了10.4%。除了语言领域，ProTransformer还在视觉和图形领域都表现出了出色的健壮性。



## **39. Effective and Efficient Adversarial Detection for Vision-Language Models via A Single Vector**

通过单个载体对视觉语言模型进行有效且高效的对抗检测 cs.CV

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22888v1) [paper-pdf](http://arxiv.org/pdf/2410.22888v1)

**Authors**: Youcheng Huang, Fengbin Zhu, Jingkun Tang, Pan Zhou, Wenqiang Lei, Jiancheng Lv, Tat-Seng Chua

**Abstract**: Visual Language Models (VLMs) are vulnerable to adversarial attacks, especially those from adversarial images, which is however under-explored in literature. To facilitate research on this critical safety problem, we first construct a new laRge-scale Adervsarial images dataset with Diverse hArmful Responses (RADAR), given that existing datasets are either small-scale or only contain limited types of harmful responses. With the new RADAR dataset, we further develop a novel and effective iN-time Embedding-based AdveRSarial Image DEtection (NEARSIDE) method, which exploits a single vector that distilled from the hidden states of VLMs, which we call the attacking direction, to achieve the detection of adversarial images against benign ones in the input. Extensive experiments with two victim VLMs, LLaVA and MiniGPT-4, well demonstrate the effectiveness, efficiency, and cross-model transferrability of our proposed method. Our code is available at https://github.com/mob-scu/RADAR-NEARSIDE

摘要: 视觉语言模型（VLM）很容易受到对抗性攻击，尤其是来自对抗性图像的攻击，但文献中对此尚未充分探讨。为了促进对这个关键安全问题的研究，我们首先构建一个具有多样性干扰响应（RADART）的新的大规模Adervsarial图像数据集，因为现有数据集要么小规模，要么仅包含有限类型的有害反应。利用新的雷达数据集，我们进一步开发了一种新颖且有效的基于iN时间嵌入的AdveRSarial Image Detect（NEARSIDE）方法，该方法利用从VLM的隐藏状态（我们称之为攻击方向）中提取的单个载体，以实现针对输入中良性图像的对抗图像的检测。对两个受害VLM（LLaVA和MiniGPT-4）进行的大量实验很好地证明了我们提出的方法的有效性、效率和跨模型可移植性。我们的代码可在https://github.com/mob-scu/RADAR-NEARSIDE上获取



## **40. Stealth edits to large language models**

对大型语言模型的隐形编辑 cs.AI

28 pages, 14 figures. Open source implementation:  https://github.com/qinghua-zhou/stealth-edits

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2406.12670v2) [paper-pdf](http://arxiv.org/pdf/2406.12670v2)

**Authors**: Oliver J. Sutton, Qinghua Zhou, Wei Wang, Desmond J. Higham, Alexander N. Gorban, Alexander Bastounis, Ivan Y. Tyukin

**Abstract**: We reveal the theoretical foundations of techniques for editing large language models, and present new methods which can do so without requiring retraining. Our theoretical insights show that a single metric (a measure of the intrinsic dimension of the model's features) can be used to assess a model's editability and reveals its previously unrecognised susceptibility to malicious stealth attacks. This metric is fundamental to predicting the success of a variety of editing approaches, and reveals new bridges between disparate families of editing methods. We collectively refer to these as stealth editing methods, because they directly update a model's weights to specify its response to specific known hallucinating prompts without affecting other model behaviour. By carefully applying our theoretical insights, we are able to introduce a new jet-pack network block which is optimised for highly selective model editing, uses only standard network operations, and can be inserted into existing networks. We also reveal the vulnerability of language models to stealth attacks: a small change to a model's weights which fixes its response to a single attacker-chosen prompt. Stealth attacks are computationally simple, do not require access to or knowledge of the model's training data, and therefore represent a potent yet previously unrecognised threat to redistributed foundation models. Extensive experimental results illustrate and support our methods and their theoretical underpinnings. Demos and source code are available at https://github.com/qinghua-zhou/stealth-edits.

摘要: 我们揭示了编辑大型语言模型的技术的理论基础，并提出了无需重新培训就能做到这一点的新方法。我们的理论见解表明，可以使用单一指标(模型特征的内在维度的衡量标准)来评估模型的可编辑性，并揭示其先前未被识别的对恶意隐形攻击的易感性。这一指标是预测各种编辑方法成功与否的基础，并揭示了不同编辑方法家族之间的新桥梁。我们将这些统称为隐形编辑方法，因为它们直接更新模型的权重，以指定其对特定已知幻觉提示的反应，而不影响其他模型的行为。通过仔细应用我们的理论见解，我们能够推出一种新的喷气式网络块，它针对高度选择性的模型编辑进行了优化，只使用标准的网络操作，并且可以插入到现有的网络中。我们还揭示了语言模型对隐形攻击的脆弱性：对模型的权重进行微小的更改即可修复其对单个攻击者选择的提示的响应。隐形攻击在计算上很简单，不需要访问或了解模型的训练数据，因此对重新分布的基础模型构成了以前未识别的强大威胁。大量的实验结果说明和支持了我们的方法及其理论基础。有关演示和源代码，请访问https://github.com/qinghua-zhou/stealth-edits.



## **41. HijackRAG: Hijacking Attacks against Retrieval-Augmented Large Language Models**

HijackRAG：针对检索增强大型语言模型的劫持攻击 cs.CR

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22832v1) [paper-pdf](http://arxiv.org/pdf/2410.22832v1)

**Authors**: Yucheng Zhang, Qinfeng Li, Tianyu Du, Xuhong Zhang, Xinkui Zhao, Zhengwen Feng, Jianwei Yin

**Abstract**: Retrieval-Augmented Generation (RAG) systems enhance large language models (LLMs) by integrating external knowledge, making them adaptable and cost-effective for various applications. However, the growing reliance on these systems also introduces potential security risks. In this work, we reveal a novel vulnerability, the retrieval prompt hijack attack (HijackRAG), which enables attackers to manipulate the retrieval mechanisms of RAG systems by injecting malicious texts into the knowledge database. When the RAG system encounters target questions, it generates the attacker's pre-determined answers instead of the correct ones, undermining the integrity and trustworthiness of the system. We formalize HijackRAG as an optimization problem and propose both black-box and white-box attack strategies tailored to different levels of the attacker's knowledge. Extensive experiments on multiple benchmark datasets show that HijackRAG consistently achieves high attack success rates, outperforming existing baseline attacks. Furthermore, we demonstrate that the attack is transferable across different retriever models, underscoring the widespread risk it poses to RAG systems. Lastly, our exploration of various defense mechanisms reveals that they are insufficient to counter HijackRAG, emphasizing the urgent need for more robust security measures to protect RAG systems in real-world deployments.

摘要: 检索-增强生成(RAG)系统通过集成外部知识来增强大型语言模型(LLM)，使它们能够适应各种应用并具有成本效益。然而，对这些系统的日益依赖也带来了潜在的安全风险。在这项工作中，我们揭示了一个新的漏洞--检索即时劫持攻击(HijackRAG)，该漏洞使攻击者能够通过向知识库中注入恶意文本来操纵RAG系统的检索机制。当RAG系统遇到目标问题时，它会生成攻击者的预定答案而不是正确的答案，从而破坏系统的完整性和可信性。我们将HijackRAG形式化为一个优化问题，并根据攻击者的不同知识水平提出了黑盒和白盒攻击策略。在多个基准数据集上的大量实验表明，HijackRAG始终具有较高的攻击成功率，性能优于现有的基线攻击。此外，我们证明了攻击可以在不同的取回器模型之间转移，强调了它对RAG系统构成的广泛风险。最后，我们对各种防御机制的探索表明，它们不足以对抗HijackRAG，强调迫切需要更强大的安全措施来保护现实世界部署中的RAG系统。



## **42. InjecGuard: Benchmarking and Mitigating Over-defense in Prompt Injection Guardrail Models**

InjecGuard：对标和缓解即时注射保障模型中的过度防御 cs.CL

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22770v1) [paper-pdf](http://arxiv.org/pdf/2410.22770v1)

**Authors**: Hao Li, Xiaogeng Liu, Chaowei Xiao

**Abstract**: Prompt injection attacks pose a critical threat to large language models (LLMs), enabling goal hijacking and data leakage. Prompt guard models, though effective in defense, suffer from over-defense -- falsely flagging benign inputs as malicious due to trigger word bias. To address this issue, we introduce NotInject, an evaluation dataset that systematically measures over-defense across various prompt guard models. NotInject contains 339 benign samples enriched with trigger words common in prompt injection attacks, enabling fine-grained evaluation. Our results show that state-of-the-art models suffer from over-defense issues, with accuracy dropping close to random guessing levels (60%). To mitigate this, we propose InjecGuard, a novel prompt guard model that incorporates a new training strategy, Mitigating Over-defense for Free (MOF), which significantly reduces the bias on trigger words. InjecGuard demonstrates state-of-the-art performance on diverse benchmarks including NotInject, surpassing the existing best model by 30.8%, offering a robust and open-source solution for detecting prompt injection attacks. The code and datasets are released at https://github.com/SaFoLab-WISC/InjecGuard.

摘要: 快速注入攻击对大型语言模型(LLM)构成严重威胁，导致目标劫持和数据泄露。即时保护模式虽然在防御方面有效，但也存在过度防御的问题--由于触发单词偏见，错误地将良性输入标记为恶意输入。为了解决这个问题，我们引入了NotInject，这是一个评估数据集，系统地测量各种提示防护模型中的过度防御。NotInject包含339个良性样本，丰富了提示注入攻击中常见的触发字，实现了细粒度评估。我们的结果表明，最先进的模型存在过度防御的问题，准确率下降到接近随机猜测的水平(60%)。为了缓解这一问题，我们提出了InjecGuard，一种新的提示守卫模型，它结合了新的训练策略，缓解了过度防御For Free(MOF)，大大减少了对触发词的偏见。InjecGuard在包括NotInject在内的各种基准测试上展示了最先进的性能，比现有最好的模型高出30.8%，为检测即时注入攻击提供了一个强大的开源解决方案。代码和数据集在https://github.com/SaFoLab-WISC/InjecGuard.上发布



## **43. Uncovering Coordinated Cross-Platform Information Operations Threatening the Integrity of the 2024 U.S. Presidential Election Online Discussion**

揭露威胁2024年美国总统选举在线讨论完整性的协调跨平台信息操作 cs.SI

First Monday 29(11), 2024

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2409.15402v2) [paper-pdf](http://arxiv.org/pdf/2409.15402v2)

**Authors**: Marco Minici, Luca Luceri, Federico Cinus, Emilio Ferrara

**Abstract**: Information Operations (IOs) pose a significant threat to the integrity of democratic processes, with the potential to influence election-related online discourse. In anticipation of the 2024 U.S. presidential election, we present a study aimed at uncovering the digital traces of coordinated IOs on $\mathbb{X}$ (formerly Twitter). Using our machine learning framework for detecting online coordination, we analyze a dataset comprising election-related conversations on $\mathbb{X}$ from May 2024. This reveals a network of coordinated inauthentic actors, displaying notable similarities in their link-sharing behaviors. Our analysis shows concerted efforts by these accounts to disseminate misleading, redundant, and biased information across the Web through a coordinated cross-platform information operation: The links shared by this network frequently direct users to other social media platforms or suspicious websites featuring low-quality political content and, in turn, promoting the same $\mathbb{X}$ and YouTube accounts. Members of this network also shared deceptive images generated by AI, accompanied by language attacking political figures and symbolic imagery intended to convey power and dominance. While $\mathbb{X}$ has suspended a subset of these accounts, more than 75% of the coordinated network remains active. Our findings underscore the critical role of developing computational models to scale up the detection of threats on large social media platforms, and emphasize the broader implications of these techniques to detect IOs across the wider Web.

摘要: 信息业务对民主进程的完整性构成重大威胁，有可能影响与选举有关的网上言论。在对2024年美国总统大选的预期中，我们提出了一项研究，旨在揭示$\mathbb{X}$(前身为Twitter)上协调iOS的数字痕迹。使用我们用于检测在线协调的机器学习框架，我们分析了一个包含2024年5月以来$\mathbb{X}$的选举相关对话的数据集。这揭示了一个协调的不真实行为者网络，显示出他们在链接分享行为上的显著相似之处。我们的分析显示，这些账户协同努力，通过协调的跨平台信息操作在网络上传播误导性、冗余和有偏见的信息：该网络共享的链接经常将用户定向到其他社交媒体平台或具有低质量政治内容的可疑网站，进而推广相同的$\mathbb{X}$和YouTube账户。该网络的成员还分享了人工智能生成的欺骗性图像，并伴随着攻击政治人物的语言和旨在传达权力和主导地位的象征性图像。虽然$\mathbb{X}$已暂停这些帐户的子集，但超过75%的协调网络仍处于活动状态。我们的发现强调了开发计算模型以扩大大型社交媒体平台上的威胁检测的关键作用，并强调了这些技术在更广泛的网络上检测iOS的更广泛影响。



## **44. SVIP: Towards Verifiable Inference of Open-source Large Language Models**

SVIP：迈向开源大型语言模型的可验证推理 cs.LG

20 pages

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.22307v1) [paper-pdf](http://arxiv.org/pdf/2410.22307v1)

**Authors**: Yifan Sun, Yuhang Li, Yue Zhang, Yuchen Jin, Huan Zhang

**Abstract**: Open-source Large Language Models (LLMs) have recently demonstrated remarkable capabilities in natural language understanding and generation, leading to widespread adoption across various domains. However, their increasing model sizes render local deployment impractical for individual users, pushing many to rely on computing service providers for inference through a blackbox API. This reliance introduces a new risk: a computing provider may stealthily substitute the requested LLM with a smaller, less capable model without consent from users, thereby delivering inferior outputs while benefiting from cost savings. In this paper, we formalize the problem of verifiable inference for LLMs. Existing verifiable computing solutions based on cryptographic or game-theoretic techniques are either computationally uneconomical or rest on strong assumptions. We introduce SVIP, a secret-based verifiable LLM inference protocol that leverages intermediate outputs from LLM as unique model identifiers. By training a proxy task on these outputs and requiring the computing provider to return both the generated text and the processed intermediate outputs, users can reliably verify whether the computing provider is acting honestly. In addition, the integration of a secret mechanism further enhances the security of our protocol. We thoroughly analyze our protocol under multiple strong and adaptive adversarial scenarios. Our extensive experiments demonstrate that SVIP is accurate, generalizable, computationally efficient, and resistant to various attacks. Notably, SVIP achieves false negative rates below 5% and false positive rates below 3%, while requiring less than 0.01 seconds per query for verification.

摘要: 开源的大型语言模型(LLM)最近在自然语言理解和生成方面表现出了非凡的能力，导致了在各个领域的广泛采用。然而，它们不断增长的模型规模使得本地部署对个人用户来说是不现实的，促使许多人依赖计算服务提供商通过黑盒API进行推理。这种依赖带来了新的风险：计算提供商可能会在未经用户同意的情况下，悄悄地用较小、功能较差的模型替换所请求的LLM，从而在提供劣质产出的同时受益于成本节约。本文对LLMS的可验证推理问题进行了形式化描述。现有的基于密码学或博弈论技术的可验证计算解决方案要么在计算上不经济，要么依赖于强有力的假设。我们引入了SVIP，这是一个基于秘密的可验证LLM推理协议，它利用LLM的中间输出作为唯一的模型标识符。通过对这些输出训练代理任务并要求计算提供商返回生成的文本和处理的中间输出，用户可以可靠地验证计算提供商是否诚实行事。此外，秘密机制的集成进一步增强了协议的安全性。我们深入分析了我们的协议在多种强和自适应对抗场景下的性能。大量实验表明，SVIP算法具有较高的准确性、通用性、计算效率和抵抗各种攻击的能力。值得注意的是，SVIP实现了5%以下的假阴性率和3%以下的假阳性率，而每次查询验证所需的时间不到0.01秒。



## **45. Embedding-based classifiers can detect prompt injection attacks**

基于嵌入的分类器可以检测提示注入攻击 cs.CR

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.22284v1) [paper-pdf](http://arxiv.org/pdf/2410.22284v1)

**Authors**: Md. Ahsan Ayub, Subhabrata Majumdar

**Abstract**: Large Language Models (LLMs) are seeing significant adoption in every type of organization due to their exceptional generative capabilities. However, LLMs are found to be vulnerable to various adversarial attacks, particularly prompt injection attacks, which trick them into producing harmful or inappropriate content. Adversaries execute such attacks by crafting malicious prompts to deceive the LLMs. In this paper, we propose a novel approach based on embedding-based Machine Learning (ML) classifiers to protect LLM-based applications against this severe threat. We leverage three commonly used embedding models to generate embeddings of malicious and benign prompts and utilize ML classifiers to predict whether an input prompt is malicious. Out of several traditional ML methods, we achieve the best performance with classifiers built using Random Forest and XGBoost. Our classifiers outperform state-of-the-art prompt injection classifiers available in open-source implementations, which use encoder-only neural networks.

摘要: 大型语言模型(LLM)由于其非凡的生成能力，在每种类型的组织中都得到了大量采用。然而，LLM被发现容易受到各种对抗性攻击，特别是即时注入攻击，这些攻击会诱使它们产生有害或不适当的内容。攻击者通过精心编制恶意提示来欺骗LLM，从而执行此类攻击。在本文中，我们提出了一种基于嵌入的机器学习(ML)分类器的新方法来保护基于LLM的应用程序免受这种严重威胁。我们利用三种常用的嵌入模型来生成恶意提示和良性提示的嵌入，并利用ML分类器来预测输入提示是否为恶意提示。在几种传统的最大似然分类方法中，我们使用随机森林和XGBoost构建的分类器取得了最好的性能。我们的分类器比开源实现中可用的最先进的提示注入分类器性能更好，后者使用仅限编码器的神经网络。



## **46. AmpleGCG-Plus: A Strong Generative Model of Adversarial Suffixes to Jailbreak LLMs with Higher Success Rates in Fewer Attempts**

AmpleGCG-Plus：越狱LLC的对抗性后缀的强生成模型，以更少的尝试获得更高的成功率 cs.CL

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.22143v1) [paper-pdf](http://arxiv.org/pdf/2410.22143v1)

**Authors**: Vishal Kumar, Zeyi Liao, Jaylen Jones, Huan Sun

**Abstract**: Although large language models (LLMs) are typically aligned, they remain vulnerable to jailbreaking through either carefully crafted prompts in natural language or, interestingly, gibberish adversarial suffixes. However, gibberish tokens have received relatively less attention despite their success in attacking aligned LLMs. Recent work, AmpleGCG~\citep{liao2024amplegcg}, demonstrates that a generative model can quickly produce numerous customizable gibberish adversarial suffixes for any harmful query, exposing a range of alignment gaps in out-of-distribution (OOD) language spaces. To bring more attention to this area, we introduce AmpleGCG-Plus, an enhanced version that achieves better performance in fewer attempts. Through a series of exploratory experiments, we identify several training strategies to improve the learning of gibberish suffixes. Our results, verified under a strict evaluation setting, show that it outperforms AmpleGCG on both open-weight and closed-source models, achieving increases in attack success rate (ASR) of up to 17\% in the white-box setting against Llama-2-7B-chat, and more than tripling ASR in the black-box setting against GPT-4. Notably, AmpleGCG-Plus jailbreaks the newer GPT-4o series of models at similar rates to GPT-4, and, uncovers vulnerabilities against the recently proposed circuit breakers defense. We publicly release AmpleGCG-Plus along with our collected training datasets.

摘要: 尽管大型语言模型(LLM)通常是一致的，但它们仍然很容易通过精心设计的自然语言提示或有趣的胡言乱语对抗性后缀越狱。然而，令人费解的令牌尽管成功地攻击了对齐的LLM，但受到的关注相对较少。最近的工作，AmpleGCG~\Citep{Lio2024Amplegcg}，证明了生成模型可以为任何有害的查询快速生成大量可定制的胡言乱语对抗性后缀，从而暴露出分布外(OOD)语言空间中的一系列对齐差距。为了引起人们对这一领域的更多关注，我们推出了AmpleGCG-Plus，这是一个增强版本，在较少的尝试中获得了更好的性能。通过一系列的探索性实验，我们确定了几种训练策略来提高乱码后缀的学习效果。在严格的评估设置下验证的结果表明，它在开源和闭源模型上都优于AmpleGCG，在白盒环境下相对于Llama-2-7B-Chat的攻击成功率(ASR)提高了17%，在黑盒环境下相对于GPT-4的攻击成功率(ASR)提高了两倍以上。值得注意的是，AmpleGCG-Plus以类似于GPT-4的速度监禁了较新的GPT-4o系列型号，并揭示了针对最近提出的断路器防御的漏洞。我们公开发布AmpleGCG-Plus以及我们收集的训练数据集。



## **47. Watch Out for Your Agents! Investigating Backdoor Threats to LLM-Based Agents**

小心你的特工！调查对LLM代理的后门威胁 cs.CR

Accepted at NeurIPS 2024, camera ready version. Code and data are  available at https://github.com/lancopku/agent-backdoor-attacks

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2402.11208v2) [paper-pdf](http://arxiv.org/pdf/2402.11208v2)

**Authors**: Wenkai Yang, Xiaohan Bi, Yankai Lin, Sishuo Chen, Jie Zhou, Xu Sun

**Abstract**: Driven by the rapid development of Large Language Models (LLMs), LLM-based agents have been developed to handle various real-world applications, including finance, healthcare, and shopping, etc. It is crucial to ensure the reliability and security of LLM-based agents during applications. However, the safety issues of LLM-based agents are currently under-explored. In this work, we take the first step to investigate one of the typical safety threats, backdoor attack, to LLM-based agents. We first formulate a general framework of agent backdoor attacks, then we present a thorough analysis of different forms of agent backdoor attacks. Specifically, compared with traditional backdoor attacks on LLMs that are only able to manipulate the user inputs and model outputs, agent backdoor attacks exhibit more diverse and covert forms: (1) From the perspective of the final attacking outcomes, the agent backdoor attacker can not only choose to manipulate the final output distribution, but also introduce the malicious behavior in an intermediate reasoning step only, while keeping the final output correct. (2) Furthermore, the former category can be divided into two subcategories based on trigger locations, in which the backdoor trigger can either be hidden in the user query or appear in an intermediate observation returned by the external environment. We implement the above variations of agent backdoor attacks on two typical agent tasks including web shopping and tool utilization. Extensive experiments show that LLM-based agents suffer severely from backdoor attacks and such backdoor vulnerability cannot be easily mitigated by current textual backdoor defense algorithms. This indicates an urgent need for further research on the development of targeted defenses against backdoor attacks on LLM-based agents. Warning: This paper may contain biased content.

摘要: 在大型语言模型(LLM)快速发展的推动下，基于LLM的代理被开发出来处理各种现实世界的应用，包括金融、医疗和购物等。然而，基于LLM的制剂的安全性问题目前还没有得到充分的研究。在这项工作中，我们首先调查了LLM代理面临的一种典型的安全威胁--后门攻击。我们首先建立了代理后门攻击的一般框架，然后深入分析了代理后门攻击的不同形式。具体地说，与传统的仅能操纵用户输入和模型输出的后门攻击相比，代理后门攻击表现出更多样和隐蔽的形式：(1)从最终攻击结果来看，代理后门攻击者不仅可以选择操纵最终输出分布，而且只能在中间推理步骤中引入恶意行为，同时保持最终输出的正确性。(2)此外，根据触发器的位置，前者可以分为两个子类，其中后门触发器可以隐藏在用户查询中，也可以出现在外部环境返回的中间观察中。我们在两个典型的代理任务上实现了上述变体的代理后门攻击，包括网上购物和工具利用。大量的实验表明，基于LLM的代理受到严重的后门攻击，而这种后门漏洞不能通过现有的文本后门防御算法轻松缓解。这表明迫切需要进一步研究针对基于LLM的特工的后门攻击开发有针对性的防御措施。警告：此论文可能包含有偏见的内容。



## **48. IDEATOR: Jailbreaking VLMs Using VLMs**

理想者：使用VLM越狱VLM cs.CV

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2411.00827v1) [paper-pdf](http://arxiv.org/pdf/2411.00827v1)

**Authors**: Ruofan Wang, Bo Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: As large Vision-Language Models (VLMs) continue to gain prominence, ensuring their safety deployment in real-world applications has become a critical concern. Recently, significant research efforts have focused on evaluating the robustness of VLMs against jailbreak attacks. Due to challenges in obtaining multi-modal data, current studies often assess VLM robustness by generating adversarial or query-relevant images based on harmful text datasets. However, the jailbreak images generated this way exhibit certain limitations. Adversarial images require white-box access to the target VLM and are relatively easy to defend against, while query-relevant images must be linked to the target harmful content, limiting their diversity and effectiveness. In this paper, we propose a novel jailbreak method named IDEATOR, which autonomously generates malicious image-text pairs for black-box jailbreak attacks. IDEATOR is a VLM-based approach inspired by our conjecture that a VLM itself might be a powerful red team model for generating jailbreak prompts. Specifically, IDEATOR employs a VLM to generate jailbreak texts while leveraging a state-of-the-art diffusion model to create corresponding jailbreak images. Extensive experiments demonstrate the high effectiveness and transferability of IDEATOR. It successfully jailbreaks MiniGPT-4 with a 94% success rate and transfers seamlessly to LLaVA and InstructBLIP, achieving high success rates of 82% and 88%, respectively. IDEATOR uncovers previously unrecognized vulnerabilities in VLMs, calling for advanced safety mechanisms.

摘要: 随着大型视觉语言模型(VLM)的不断涌现，确保其在实际应用中的安全部署已成为一个关键问题。最近，大量的研究工作集中在评估VLM对越狱攻击的健壮性上。由于获取多模式数据的困难，目前的研究通常通过基于有害文本数据集生成对抗性或与查询相关的图像来评估VLM的稳健性。然而，以这种方式生成的越狱图像显示出一定的局限性。敌意图像需要对目标VLM进行白盒访问，并且相对容易防御，而与查询相关的图像必须链接到目标有害内容，限制了它们的多样性和有效性。在本文中，我们提出了一种新的越狱方法IDEATOR，该方法自动生成用于黑盒越狱攻击的恶意图文对。Ideator是一种基于VLM的方法，灵感来自于我们的猜测，即VLM本身可能是用于生成越狱提示的强大的红色团队模型。具体地说，Ideator使用VLM来生成越狱文本，同时利用最先进的扩散模型来创建相应的越狱图像。大量的实验证明了IDENTER的高效性和可移植性。它以94%的成功率成功越狱MiniGPT-4，并无缝转移到LLaVA和InstructBLIP，分别实现了82%和88%的高成功率。Ideator发现了VLM中以前未被识别的漏洞，呼吁使用先进的安全机制。



## **49. Waterfall: Framework for Robust and Scalable Text Watermarking and Provenance for LLMs**

Waterfall：LLM稳健且可扩展的文本水印和起源框架 cs.CR

Accepted to EMNLP 2024 Main Conference

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2407.04411v2) [paper-pdf](http://arxiv.org/pdf/2407.04411v2)

**Authors**: Gregory Kang Ruey Lau, Xinyuan Niu, Hieu Dao, Jiangwei Chen, Chuan-Sheng Foo, Bryan Kian Hsiang Low

**Abstract**: Protecting intellectual property (IP) of text such as articles and code is increasingly important, especially as sophisticated attacks become possible, such as paraphrasing by large language models (LLMs) or even unauthorized training of LLMs on copyrighted text to infringe such IP. However, existing text watermarking methods are not robust enough against such attacks nor scalable to millions of users for practical implementation. In this paper, we propose Waterfall, the first training-free framework for robust and scalable text watermarking applicable across multiple text types (e.g., articles, code) and languages supportable by LLMs, for general text and LLM data provenance. Waterfall comprises several key innovations, such as being the first to use LLM as paraphrasers for watermarking along with a novel combination of techniques that are surprisingly effective in achieving robust verifiability and scalability. We empirically demonstrate that Waterfall achieves significantly better scalability, robust verifiability, and computational efficiency compared to SOTA article-text watermarking methods, and showed how it could be directly applied to the watermarking of code. We also demonstrated that Waterfall can be used for LLM data provenance, where the watermarks of LLM training data can be detected in LLM output, allowing for detection of unauthorized use of data for LLM training and potentially enabling model-centric watermarking of open-sourced LLMs which has been a limitation of existing LLM watermarking works. Our code is available at https://github.com/aoi3142/Waterfall.

摘要: 保护文章和代码等文本的知识产权(IP)越来越重要，特别是在可能进行复杂攻击的情况下，例如利用大型语言模型(LLM)进行释义，甚至未经授权对受版权保护的文本进行LLM培训，以侵犯此类IP。然而，现有的文本水印方法对此类攻击不够健壮，也不能扩展到数百万用户进行实际实现。在本文中，我们提出了瀑布，这是第一个无训练的文本水印框架，适用于LLMS支持的多种文本类型(例如，文章、代码)和语言，用于一般文本和LLM数据来源。瀑布由几项关键创新组成，例如第一个使用LLM作为水印解释程序，以及在实现强大的可验证性和可扩展性方面出人意料地有效的技术组合。实验证明，与SOTA的文章-文本水印方法相比，瀑布算法具有更好的可扩展性、健壮性、可验证性和计算效率，并且可以直接应用于代码的水印。我们还演示了瀑布可以用于LLM数据起源，其中LLM训练数据的水印可以在LLM输出中检测到，允许检测未经授权使用数据进行LLM训练，并潜在地实现了以模型为中心的开放源代码LLMS水印，这一直是现有LLM水印工作的限制。我们的代码可以在https://github.com/aoi3142/Waterfall.上找到



## **50. Enhancing Adversarial Attacks through Chain of Thought**

通过思维链增强对抗性攻击 cs.CL

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.21791v1) [paper-pdf](http://arxiv.org/pdf/2410.21791v1)

**Authors**: Jingbo Su

**Abstract**: Large language models (LLMs) have demonstrated impressive performance across various domains but remain susceptible to safety concerns. Prior research indicates that gradient-based adversarial attacks are particularly effective against aligned LLMs and the chain of thought (CoT) prompting can elicit desired answers through step-by-step reasoning. This paper proposes enhancing the robustness of adversarial attacks on aligned LLMs by integrating CoT prompts with the greedy coordinate gradient (GCG) technique. Using CoT triggers instead of affirmative targets stimulates the reasoning abilities of backend LLMs, thereby improving the transferability and universality of adversarial attacks. We conducted an ablation study comparing our CoT-GCG approach with Amazon Web Services auto-cot. Results revealed our approach outperformed both the baseline GCG attack and CoT prompting. Additionally, we used Llama Guard to evaluate potentially harmful interactions, providing a more objective risk assessment of entire conversations compared to matching outputs to rejection phrases. The code of this paper is available at https://github.com/sujingbo0217/CS222W24-LLM-Attack.

摘要: 大型语言模型(LLM)在各个领域都表现出了令人印象深刻的表现，但仍然容易受到安全问题的影响。以往的研究表明，基于梯度的对抗性攻击对于对齐的LLM特别有效，而思维链(COT)提示可以通过循序渐进的推理获得期望的答案。提出了一种将CoT提示与贪婪坐标梯度(GCG)技术相结合的方法，以增强对齐LLMS的敌意攻击的稳健性。使用CoT触发器代替肯定目标，刺激了后端LLMS的推理能力，从而提高了对抗性攻击的可转移性和通用性。我们进行了一项烧蚀研究，将我们的COT-GCG方法与Amazon Web Services自动COT进行了比较。结果显示，我们的方法比基线GCG攻击和COT提示都要好。此外，我们使用Llama Guard来评估潜在的有害交互，与将输出与拒绝短语匹配相比，提供了对整个对话的更客观的风险评估。本文的代码可在https://github.com/sujingbo0217/CS222W24-LLM-Attack.上找到



