# Latest Large Language Model Attack Papers
**update at 2024-10-16 11:21:23**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks**

G-Designer：通过图神经网络构建多智能体通信布局 cs.MA

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11782v1) [paper-pdf](http://arxiv.org/pdf/2410.11782v1)

**Authors**: Guibin Zhang, Yanwei Yue, Xiangguo Sun, Guancheng Wan, Miao Yu, Junfeng Fang, Kun Wang, Dawei Cheng

**Abstract**: Recent advancements in large language model (LLM)-based agents have demonstrated that collective intelligence can significantly surpass the capabilities of individual agents, primarily due to well-crafted inter-agent communication topologies. Despite the diverse and high-performing designs available, practitioners often face confusion when selecting the most effective pipeline for their specific task: \textit{Which topology is the best choice for my task, avoiding unnecessary communication token overhead while ensuring high-quality solution?} In response to this dilemma, we introduce G-Designer, an adaptive, efficient, and robust solution for multi-agent deployment, which dynamically designs task-aware, customized communication topologies. Specifically, G-Designer models the multi-agent system as a multi-agent network, leveraging a variational graph auto-encoder to encode both the nodes (agents) and a task-specific virtual node, and decodes a task-adaptive and high-performing communication topology. Extensive experiments on six benchmarks showcase that G-Designer is: \textbf{(1) high-performing}, achieving superior results on MMLU with accuracy at $84.50\%$ and on HumanEval with pass@1 at $89.90\%$; \textbf{(2) task-adaptive}, architecting communication protocols tailored to task difficulty, reducing token consumption by up to $95.33\%$ on HumanEval; and \textbf{(3) adversarially robust}, defending against agent adversarial attacks with merely $0.3\%$ accuracy drop.

摘要: 基于大型语言模型(LLM)的代理的最新进展表明，集体智能可以显著超过单个代理的能力，这主要是由于精心设计的代理间通信拓扑。尽管有多样化和高性能的设计，但实践者在为他们的特定任务选择最有效的流水线时经常面临困惑：\textit{哪个拓扑是我的任务的最佳选择，在确保高质量解决方案的同时避免不必要的通信令牌开销？}针对这种困境，我们引入了G-Designer，这是一个自适应的、高效的、健壮的多代理部署解决方案，它动态地设计任务感知的、定制的通信拓扑。具体地说，G-Designer将多代理系统建模为多代理网络，利用变化图自动编码器对节点(代理)和特定于任务的虚拟节点进行编码，并解码任务自适应的高性能通信拓扑。在六个基准测试上的广泛实验表明，G-Designer是：\extbf{(1)高性能}，在MMLU上获得了更好的结果，准确率为84.50\$，在HumanEval上，PASS@1的准确率为89.90\$；\extbf{(2)任务自适应}，构建了针对任务难度的通信协议，在HumanEval上减少了高达95.33\$的令牌消耗；以及\extbf{(3)对手健壮性}，防御代理对手攻击，精确度仅下降了$0.3\%$。



## **2. LLM-Based Robust Product Classification in Commerce and Compliance**

基于LLM的商业和合规稳健产品分类 cs.CL

Camera-ready version for Customizable NLP Workshop at EMNLP 2024. 11  pages

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2408.05874v2) [paper-pdf](http://arxiv.org/pdf/2408.05874v2)

**Authors**: Sina Gholamian, Gianfranco Romani, Bartosz Rudnikowicz, Stavroula Skylaki

**Abstract**: Product classification is a crucial task in international trade, as compliance regulations are verified and taxes and duties are applied based on product categories. Manual classification of products is time-consuming and error-prone, and the sheer volume of products imported and exported renders the manual process infeasible. Consequently, e-commerce platforms and enterprises involved in international trade have turned to automatic product classification using machine learning. However, current approaches do not consider the real-world challenges associated with product classification, such as very abbreviated and incomplete product descriptions. In addition, recent advancements in generative Large Language Models (LLMs) and their reasoning capabilities are mainly untapped in product classification and e-commerce. In this research, we explore the real-life challenges of industrial classification and we propose data perturbations that allow for realistic data simulation. Furthermore, we employ LLM-based product classification to improve the robustness of the prediction in presence of incomplete data. Our research shows that LLMs with in-context learning outperform the supervised approaches in the clean-data scenario. Additionally, we illustrate that LLMs are significantly more robust than the supervised approaches when data attacks are present.

摘要: 产品分类是国际贸易中的一项关键任务，因为要核实合规规定，并根据产品类别适用税收和关税。人工对产品进行分类既耗时又容易出错，而且进出口产品的数量庞大，使手工分类过程变得不可行。因此，参与国际贸易的电子商务平台和企业已经转向使用机器学习的产品自动分类。然而，目前的方法没有考虑到与产品分类相关的现实挑战，例如非常简短和不完整的产品描述。此外，生成性大型语言模型(LLM)及其推理能力的最新进展主要是在产品分类和电子商务方面尚未开发。在这项研究中，我们探索了现实生活中的行业分类挑战，并提出了允许现实数据模拟的数据扰动。此外，我们使用基于LLM的产品分类来提高在存在不完整数据的情况下预测的稳健性。我们的研究表明，在干净数据的情况下，具有情境学习的LLMS的性能优于有监督的方法。此外，我们还说明了当存在数据攻击时，LLMS比监督方法具有更强的健壮性。



## **3. Phantom: General Trigger Attacks on Retrieval Augmented Language Generation**

Phantom：对检索增强语言生成的通用触发攻击 cs.CR

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2405.20485v2) [paper-pdf](http://arxiv.org/pdf/2405.20485v2)

**Authors**: Harsh Chaudhari, Giorgio Severi, John Abascal, Matthew Jagielski, Christopher A. Choquette-Choo, Milad Nasr, Cristina Nita-Rotaru, Alina Oprea

**Abstract**: Retrieval Augmented Generation (RAG) expands the capabilities of modern large language models (LLMs), by anchoring, adapting, and personalizing their responses to the most relevant knowledge sources. It is particularly useful in chatbot applications, allowing developers to customize LLM output without expensive retraining. Despite their significant utility in various applications, RAG systems present new security risks. In this work, we propose new attack vectors that allow an adversary to inject a single malicious document into a RAG system's knowledge base, and mount a backdoor poisoning attack. We design Phantom, a general two-stage optimization framework against RAG systems, that crafts a malicious poisoned document leading to an integrity violation in the model's output. First, the document is constructed to be retrieved only when a specific trigger sequence of tokens appears in the victim's queries. Second, the document is further optimized with crafted adversarial text that induces various adversarial objectives on the LLM output, including refusal to answer, reputation damage, privacy violations, and harmful behaviors. We demonstrate our attacks on multiple LLM architectures, including Gemma, Vicuna, and Llama, and show that they transfer to GPT-3.5 Turbo and GPT-4. Finally, we successfully conducted a Phantom attack on NVIDIA's black-box production RAG system, "Chat with RTX".

摘要: 检索增强生成(RAG)通过锚定、调整和个性化对最相关的知识源的响应来扩展现代大型语言模型(LLMS)的能力。它在聊天机器人应用程序中特别有用，允许开发人员定制LLM输出，而无需昂贵的再培训。尽管RAG系统在各种应用中具有重要的实用价值，但它带来了新的安全风险。在这项工作中，我们提出了新的攻击向量，允许攻击者将单个恶意文档注入RAG系统的知识库，并发动后门中毒攻击。我们设计了Phantom，这是一个针对RAG系统的通用两阶段优化框架，它手工制作了一个恶意中毒文档，导致模型输出中的完整性破坏。首先，文档被构建为仅在受害者的查询中出现特定的令牌触发序列时才检索。其次，通过精心设计的敌意文本进一步优化了文档，这些文本在LLM输出上诱导了各种敌意目标，包括拒绝回答、声誉损害、侵犯隐私和有害行为。我们演示了我们对多个LLM体系结构的攻击，包括Gema、Vicuna和Llama，并表明它们可以传输到GPT-3.5Turbo和GPT-4。最后，我们成功地对NVIDIA的黑匣子生产RAG系统“与腾讯通聊天”进行了幻影攻击。



## **4. Efficient and Effective Universal Adversarial Attack against Vision-Language Pre-training Models**

针对视觉语言预训练模型的高效且有效的通用对抗攻击 cs.CV

11 pages

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11639v1) [paper-pdf](http://arxiv.org/pdf/2410.11639v1)

**Authors**: Fan Yang, Yihao Huang, Kailong Wang, Ling Shi, Geguang Pu, Yang Liu, Haoyu Wang

**Abstract**: Vision-language pre-training (VLP) models, trained on large-scale image-text pairs, have become widely used across a variety of downstream vision-and-language (V+L) tasks. This widespread adoption raises concerns about their vulnerability to adversarial attacks. Non-universal adversarial attacks, while effective, are often impractical for real-time online applications due to their high computational demands per data instance. Recently, universal adversarial perturbations (UAPs) have been introduced as a solution, but existing generator-based UAP methods are significantly time-consuming. To overcome the limitation, we propose a direct optimization-based UAP approach, termed DO-UAP, which significantly reduces resource consumption while maintaining high attack performance. Specifically, we explore the necessity of multimodal loss design and introduce a useful data augmentation strategy. Extensive experiments conducted on three benchmark VLP datasets, six popular VLP models, and three classical downstream tasks demonstrate the efficiency and effectiveness of DO-UAP. Specifically, our approach drastically decreases the time consumption by 23-fold while achieving a better attack performance.

摘要: 视觉-语言预训练模型是在大规模图文对上训练的，已被广泛应用于各种下游视觉与语言(V+L)任务。这种广泛的采用引起了人们对它们易受对手攻击的担忧。非通用对抗性攻击虽然有效，但对于实时在线应用程序来说往往是不切实际的，因为它们对每个数据实例的计算要求很高。最近，通用对抗扰动(UAP)被引入作为解决方案，但现有的基于生成器的UAP方法非常耗时。为了克服这一局限性，我们提出了一种基于直接优化的UAP方法，称为DO-UAP，它在保持高攻击性能的同时显著减少了资源消耗。具体地说，我们探讨了多峰损失设计的必要性，并介绍了一种有用的数据增强策略。在三个基准VLP数据集、六个流行的VLP模型和三个经典下游任务上的广泛实验证明了DO-UAP的效率和有效性。具体地说，我们的方法大大减少了23倍的时间消耗，同时实现了更好的攻击性能。



## **5. Gotcha! This Model Uses My Code! Evaluating Membership Leakage Risks in Code Models**

抓住你了！这个模型使用我的代码！评估代码模型中的成员泄漏风险 cs.SE

Accepted by IEEE Transactions on Software Engineering, Camera-Ready  Version

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2310.01166v2) [paper-pdf](http://arxiv.org/pdf/2310.01166v2)

**Authors**: Zhou Yang, Zhipeng Zhao, Chenyu Wang, Jieke Shi, Dongsum Kim, Donggyun Han, David Lo

**Abstract**: Given large-scale source code datasets available in open-source projects and advanced large language models, recent code models have been proposed to address a series of critical software engineering tasks, such as program repair and code completion. The training data of the code models come from various sources, not only the publicly available source code, e.g., open-source projects on GitHub but also the private data such as the confidential source code from companies, which may contain sensitive information (for example, SSH keys and personal information). As a result, the use of these code models may raise new privacy concerns.   In this paper, we focus on a critical yet not well-explored question on using code models: what is the risk of membership information leakage in code models? Membership information leakage refers to the risk that an attacker can infer whether a given data point is included in (i.e., a member of) the training data. To answer this question, we propose Gotcha, a novel membership inference attack method specifically for code models. We investigate the membership leakage risk of code models. Our results reveal a worrying fact that the risk of membership leakage is high: although the previous attack methods are close to random guessing, Gotcha can predict the data membership with a high true positive rate of 0.95 and a low false positive rate of 0.10. We also show that the attacker's knowledge of the victim model (e.g., the model architecture and the pre-training data) impacts the success rate of attacks. Further analysis demonstrates that changing the decoding strategy can mitigate the risk of membership leakage. This study calls for more attention to understanding the privacy of code models and developing more effective countermeasures against such attacks.

摘要: 鉴于开源项目中可用的大规模源代码数据集和高级大型语言模型，最近提出了一些代码模型来解决一系列关键的软件工程任务，如程序修复和代码完成。代码模型的训练数据来自各种来源，不仅有公开可用的源代码，如GitHub上的开源项目，还包括私人数据，如来自公司的机密源代码，其中可能包含敏感信息(如SSH密钥和个人信息)。因此，使用这些代码模型可能会引发新的隐私问题。在这篇文章中，我们关注一个关于使用代码模型的关键但没有得到很好探索的问题：代码模型中成员信息泄漏的风险是什么？成员资格信息泄漏是指攻击者可以推断给定数据点是否包括在训练数据中(即，训练数据的成员)的风险。为了回答这个问题，我们提出了Gotcha，一种新的专门针对代码模型的成员推理攻击方法。我们研究了编码模型的成员泄漏风险。我们的结果揭示了一个令人担忧的事实，即成员泄露的风险很高：虽然以前的攻击方法接近随机猜测，但Gotcha可以预测数据的成员身份，真阳性率高达0.95，假阳性率低0.10。我们还表明，攻击者对受害者模型的了解(例如，模型体系结构和预训练数据)会影响攻击的成功率。进一步的分析表明，改变译码策略可以降低成员泄漏的风险。这项研究呼吁更多地关注了解代码模型的隐私，并开发更有效的对策来应对此类攻击。



## **6. Multi-round jailbreak attack on large language models**

对大型语言模型的多轮越狱攻击 cs.CL

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11533v1) [paper-pdf](http://arxiv.org/pdf/2410.11533v1)

**Authors**: Yihua Zhou, Xiaochuan Shi

**Abstract**: Ensuring the safety and alignment of large language models (LLMs) with human values is crucial for generating responses that are beneficial to humanity. While LLMs have the capability to identify and avoid harmful queries, they remain vulnerable to "jailbreak" attacks, where carefully crafted prompts can induce the generation of toxic content. Traditional single-round jailbreak attacks, such as GCG and AutoDAN, do not alter the sensitive words in the dangerous prompts. Although they can temporarily bypass the model's safeguards through prompt engineering, their success rate drops significantly as the LLM is further fine-tuned, and they cannot effectively circumvent static rule-based filters that remove the hazardous vocabulary.   In this study, to better understand jailbreak attacks, we introduce a multi-round jailbreak approach. This method can rewrite the dangerous prompts, decomposing them into a series of less harmful sub-questions to bypass the LLM's safety checks. We first use the LLM to perform a decomposition task, breaking down a set of natural language questions into a sequence of progressive sub-questions, which are then used to fine-tune the Llama3-8B model, enabling it to decompose hazardous prompts. The fine-tuned model is then used to break down the problematic prompt, and the resulting sub-questions are sequentially asked to the victim model. If the victim model rejects a sub-question, a new decomposition is generated, and the process is repeated until the final objective is achieved. Our experimental results show a 94\% success rate on the llama2-7B and demonstrate the effectiveness of this approach in circumventing static rule-based filters.

摘要: 确保大型语言模型(LLM)的安全性并使其与人类价值观保持一致，对于产生有益于人类的反应至关重要。虽然LLM具有识别和避免有害查询的能力，但它们仍然容易受到“越狱”攻击，在这种攻击中，精心设计的提示可能会导致有毒内容的生成。传统的单轮越狱攻击，如GCG和AutoDAN，不会改变危险提示中的敏感词语。尽管他们可以通过快速工程暂时绕过模型的保障措施，但随着LLM的进一步微调，他们的成功率显著下降，而且他们无法有效地绕过删除危险词汇的静态规则过滤器。在这项研究中，为了更好地理解越狱攻击，我们引入了一种多轮越狱方法。这种方法可以重写危险的提示，将它们分解为一系列危害较小的子问题，以绕过LLM的安全检查。我们首先使用LLM执行分解任务，将一组自然语言问题分解为一系列递进子问题，然后使用这些子问题来微调Llama3-8B模型，使其能够分解危险提示。然后，使用微调的模型来分解有问题的提示，并将得到的子问题顺序地询问给受害者模型。如果受害者模型拒绝了子问题，则生成新的分解，并重复该过程，直到达到最终目标。实验结果表明，该方法在Llama2-7B上的检测成功率为94%，证明了该方法对静态规则过滤的有效性。



## **7. Jigsaw Puzzles: Splitting Harmful Questions to Jailbreak Large Language Models**

拼图：分解有害问题以越狱大型语言模型 cs.CL

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11459v1) [paper-pdf](http://arxiv.org/pdf/2410.11459v1)

**Authors**: Hao Yang, Lizhen Qu, Ehsan Shareghi, Gholamreza Haffari

**Abstract**: Large language models (LLMs) have exhibited outstanding performance in engaging with humans and addressing complex questions by leveraging their vast implicit knowledge and robust reasoning capabilities. However, such models are vulnerable to jailbreak attacks, leading to the generation of harmful responses. Despite recent research on single-turn jailbreak strategies to facilitate the development of defence mechanisms, the challenge of revealing vulnerabilities under multi-turn setting remains relatively under-explored. In this work, we propose Jigsaw Puzzles (JSP), a straightforward yet effective multi-turn jailbreak strategy against the advanced LLMs. JSP splits questions into harmless fractions as the input of each turn, and requests LLMs to reconstruct and respond to questions under multi-turn interaction. Our experimental results demonstrate that the proposed JSP jailbreak bypasses original safeguards against explicitly harmful content, achieving an average attack success rate of 93.76% on 189 harmful queries across 5 advanced LLMs (Gemini-1.5-Pro, Llama-3.1-70B, GPT-4, GPT-4o, GPT-4o-mini). Moreover, JSP achieves a state-of-the-art attack success rate of 92% on GPT-4 on the harmful query benchmark, and exhibits strong resistant to defence strategies. Warning: this paper contains offensive examples.

摘要: 大型语言模型(LLM)利用其丰富的隐含知识和强大的推理能力，在与人类接触和解决复杂问题方面表现出了出色的表现。然而，这类模型容易受到越狱攻击，从而导致有害反应的产生。尽管最近对单轮越狱战略进行了研究，以促进防御机制的发展，但在多轮情况下揭示脆弱性的挑战仍然相对较少。在这项工作中，我们提出了Jigsaw Puzzles(JSP)，一种针对高级LLM的简单而有效的多回合越狱策略。该算法将问题分解成若干个无害的分数作为每一轮的输入，并在多轮交互的情况下要求LLMS对问题进行重构和回答。我们的实验结果表明，该方案绕过了原有的针对显式有害内容的保护措施，对5个高级LLMS(Gemini-1.5-Pro、Llama-3.1-70B、GPT-4、GPT-4o、GPT-4o-mini)的189个有害查询的平均攻击成功率为93.76%。此外，在有害查询基准上，对GPT-4的攻击成功率达到92%，并且对防御策略表现出很强的抵抗力。警告：本文包含令人反感的例子。



## **8. Deciphering the Chaos: Enhancing Jailbreak Attacks via Adversarial Prompt Translation**

破译混乱：通过对抗性提示翻译增强越狱攻击 cs.LG

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11317v1) [paper-pdf](http://arxiv.org/pdf/2410.11317v1)

**Authors**: Qizhang Li, Xiaochen Yang, Wangmeng Zuo, Yiwen Guo

**Abstract**: Automatic adversarial prompt generation provides remarkable success in jailbreaking safely-aligned large language models (LLMs). Existing gradient-based attacks, while demonstrating outstanding performance in jailbreaking white-box LLMs, often generate garbled adversarial prompts with chaotic appearance. These adversarial prompts are difficult to transfer to other LLMs, hindering their performance in attacking unknown victim models. In this paper, for the first time, we delve into the semantic meaning embedded in garbled adversarial prompts and propose a novel method that "translates" them into coherent and human-readable natural language adversarial prompts. In this way, we can effectively uncover the semantic information that triggers vulnerabilities of the model and unambiguously transfer it to the victim model, without overlooking the adversarial information hidden in the garbled text, to enhance jailbreak attacks. It also offers a new approach to discovering effective designs for jailbreak prompts, advancing the understanding of jailbreak attacks. Experimental results demonstrate that our method significantly improves the success rate of jailbreak attacks against various safety-aligned LLMs and outperforms state-of-the-arts by large margins. With at most 10 queries, our method achieves an average attack success rate of 81.8% in attacking 7 commercial closed-source LLMs, including GPT and Claude-3 series, on HarmBench. Our method also achieves over 90% attack success rates against Llama-2-Chat models on AdvBench, despite their outstanding resistance to jailbreak attacks. Code at: https://github.com/qizhangli/Adversarial-Prompt-Translator.

摘要: 自动对抗性提示生成在越狱安全对齐的大型语言模型(LLM)方面取得了显着的成功。现有的基于梯度的攻击虽然在越狱白盒LLM中表现出出色的性能，但往往会产生外观混乱的乱码对抗性提示。这些对抗性提示很难转移到其他LLM上，阻碍了它们在攻击未知受害者模型时的表现。在本文中，我们首次深入研究了混淆的对抗性提示中所蕴含的语义，并提出了一种新的方法，将它们“翻译”成连贯的、人类可读的自然语言对抗性提示。通过这种方式，我们可以有效地发现触发模型漏洞的语义信息，并毫不含糊地将其传递给受害者模型，而不会忽视隐藏在乱码文本中的对抗性信息，以增强越狱攻击。它还提供了一种新的方法来发现有效的越狱提示设计，促进了对越狱攻击的理解。实验结果表明，我们的方法显著提高了对各种安全对齐LLM的越狱攻击成功率，并且远远超过了最新的技术水平。在最多10个查询的情况下，我们的方法在HarmBch上攻击包括GPT和Claude-3系列在内的7个商业闭源LLM，平均攻击成功率为81.8%。我们的方法对AdvBtch上的Llama-2-Chat模型的攻击成功率也达到了90%以上，尽管它们对越狱攻击具有出色的抵抗力。代码：https://github.com/qizhangli/Adversarial-Prompt-Translator.



## **9. Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation**

闭上眼睛，安全：通过图像到文本转换保护多模式LLM cs.CV

ECCV2024 (Project Page: https://gyhdog99.github.io/projects/ecso/)

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2403.09572v4) [paper-pdf](http://arxiv.org/pdf/2403.09572v4)

**Authors**: Yunhao Gou, Kai Chen, Zhili Liu, Lanqing Hong, Hang Xu, Zhenguo Li, Dit-Yan Yeung, James T. Kwok, Yu Zhang

**Abstract**: Multimodal large language models (MLLMs) have shown impressive reasoning abilities. However, they are also more vulnerable to jailbreak attacks than their LLM predecessors. Although still capable of detecting the unsafe responses, we observe that safety mechanisms of the pre-aligned LLMs in MLLMs can be easily bypassed with the introduction of image features. To construct robust MLLMs, we propose ECSO (Eyes Closed, Safety On), a novel training-free protecting approach that exploits the inherent safety awareness of MLLMs, and generates safer responses via adaptively transforming unsafe images into texts to activate the intrinsic safety mechanism of pre-aligned LLMs in MLLMs. Experiments on five state-of-the-art (SoTA) MLLMs demonstrate that ECSO enhances model safety significantly (e.g.,, 37.6% improvement on the MM-SafetyBench (SD+OCR) and 71.3% on VLSafe with LLaVA-1.5-7B), while consistently maintaining utility results on common MLLM benchmarks. Furthermore, we show that ECSO can be used as a data engine to generate supervised-finetuning (SFT) data for MLLM alignment without extra human intervention.

摘要: 多通道大型语言模型(MLLMS)已经显示出令人印象深刻的推理能力。然而，他们也比他们的LLM前辈更容易受到越狱攻击。虽然仍然能够检测到不安全的响应，但我们观察到，通过引入图像特征，可以很容易地绕过MLLMS中预对准LLM的安全机制。为了构造稳健的MLLMS，我们提出了一种新的无需训练的保护方法ECSO(Eyes Closed，Safe On)，该方法利用MLLMS固有的安全意识，通过自适应地将不安全的图像转换为文本来激活MLLMS中预对准的LLMS的固有安全机制，从而产生更安全的响应。在五个最先进的(SOTA)MLLM上的实验表明，ECSO显著提高了模型安全性(例如，在MM-SafetyBch(SD+OCR)的基础上改进了37.6%，在使用LLaVA-1.5-7B的VLSafe上改进了71.3%)，同时保持了常见MLLM基准的实用结果。此外，我们还证明了ECSO可以作为数据引擎来生成用于MLLM比对的监督精调(SFT)数据，而无需额外的人工干预。



## **10. Cognitive Overload Attack:Prompt Injection for Long Context**

认知过载攻击：长上下文的提示注入 cs.CL

40 pages, 31 Figures

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11272v1) [paper-pdf](http://arxiv.org/pdf/2410.11272v1)

**Authors**: Bibek Upadhayay, Vahid Behzadan, Amin Karbasi

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in performing tasks across various domains without needing explicit retraining. This capability, known as In-Context Learning (ICL), while impressive, exposes LLMs to a variety of adversarial prompts and jailbreaks that manipulate safety-trained LLMs into generating undesired or harmful output. In this paper, we propose a novel interpretation of ICL in LLMs through the lens of cognitive neuroscience, by drawing parallels between learning in human cognition with ICL. We applied the principles of Cognitive Load Theory in LLMs and empirically validate that similar to human cognition, LLMs also suffer from cognitive overload a state where the demand on cognitive processing exceeds the available capacity of the model, leading to potential errors. Furthermore, we demonstrated how an attacker can exploit ICL to jailbreak LLMs through deliberately designed prompts that induce cognitive overload on LLMs, thereby compromising the safety mechanisms of LLMs. We empirically validate this threat model by crafting various cognitive overload prompts and show that advanced models such as GPT-4, Claude-3.5 Sonnet, Claude-3 OPUS, Llama-3-70B-Instruct, Gemini-1.0-Pro, and Gemini-1.5-Pro can be successfully jailbroken, with attack success rates of up to 99.99%. Our findings highlight critical vulnerabilities in LLMs and underscore the urgency of developing robust safeguards. We propose integrating insights from cognitive load theory into the design and evaluation of LLMs to better anticipate and mitigate the risks of adversarial attacks. By expanding our experiments to encompass a broader range of models and by highlighting vulnerabilities in LLMs' ICL, we aim to ensure the development of safer and more reliable AI systems.

摘要: 大型语言模型(LLM)已经显示出在不需要明确的再培训的情况下执行跨领域任务的显著能力。这种被称为情景学习(ICL)的能力虽然令人印象深刻，但会使LLM暴露在各种对抗性提示和越狱之下，这些提示和越狱操作经过安全培训的LLM产生不需要的或有害的输出。在这篇文章中，我们提出了一种新的解释，从认知神经科学的角度，通过将人类认知中的学习与ICL相提并论，对LLMS中的ICL做出了新的解释。我们将认知负荷理论的原理应用到LLMS中，并实证验证了与人类认知类似，LLMS也存在认知过载，即认知加工需求超过模型的可用能力，从而导致潜在错误。此外，我们演示了攻击者如何通过故意设计的提示来利用ICL来越狱LLM，这些提示会导致LLM上的认知过载，从而危及LLMS的安全机制。我们通过制作不同的认知过载提示对该威胁模型进行了实证验证，结果表明，GPT-4、Claude-3.5十四行诗、Claude-3 opus、Llama-3-70B-Indict、Gemini-1.0-Pro和Gemini-1.5-Pro等高级模型可以成功越狱，攻击成功率高达99.99%。我们的发现突显了低土地管理制度的严重脆弱性，并强调了制定强有力的保障措施的紧迫性。我们建议将认知负荷理论的见解融入到LLMS的设计和评估中，以更好地预测和减轻对手攻击的风险。通过扩大我们的实验以涵盖更广泛的模型，并通过突出LLMS ICL中的漏洞，我们的目标是确保开发出更安全、更可靠的人工智能系统。



## **11. Archilles' Heel in Semi-open LLMs: Hiding Bottom against Recovery Attacks**

阿奇勒斯在半开放式法学硕士中的脚跟：隐藏底部抵御复苏攻击 cs.LG

10 pages for main content of the paper

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11182v1) [paper-pdf](http://arxiv.org/pdf/2410.11182v1)

**Authors**: Hanbo Huang, Yihan Li, Bowen Jiang, Lin Liu, Ruoyu Sun, Zhuotao Liu, Shiyu Liang

**Abstract**: Closed-source large language models deliver strong performance but have limited downstream customizability. Semi-open models, combining both closed-source and public layers, were introduced to improve customizability. However, parameters in the closed-source layers are found vulnerable to recovery attacks. In this paper, we explore the design of semi-open models with fewer closed-source layers, aiming to increase customizability while ensuring resilience to recovery attacks. We analyze the contribution of closed-source layer to the overall resilience and theoretically prove that in a deep transformer-based model, there exists a transition layer such that even small recovery errors in layers before this layer can lead to recovery failure. Building on this, we propose \textbf{SCARA}, a novel approach that keeps only a few bottom layers as closed-source. SCARA employs a fine-tuning-free metric to estimate the maximum number of layers that can be publicly accessible for customization. We apply it to five models (1.3B to 70B parameters) to construct semi-open models, validating their customizability on six downstream tasks and assessing their resilience against various recovery attacks on sixteen benchmarks. We compare SCARA to baselines and observe that it generally improves downstream customization performance and offers similar resilience with over \textbf{10} times fewer closed-source parameters. We empirically investigate the existence of transition layers, analyze the effectiveness of our scheme and finally discuss its limitations.

摘要: 封闭源代码的大型语言模型提供了强大的性能，但下游可定制化能力有限。引入了半开放模型，结合了封闭源码和公共层，以提高可定制性。然而，封闭源代码层中的参数容易受到恢复攻击。在本文中，我们探索了闭源层较少的半开放模型的设计，目的是在增加可定制性的同时确保对恢复攻击的弹性。我们分析了闭源层对整体恢复能力的贡献，并从理论上证明了在基于深层变压器的模型中，存在一个过渡层，即使在该过渡层之前的各层中存在微小的恢复错误，也可能导致恢复失败。在此基础上，我们提出了一种新的方法Scara使用一种无需微调的指标来估计可公开访问以供定制的最大层数。我们将其应用于5个模型(1.3B到70B参数)来构建半开放模型，验证了它们在6个下游任务上的可定制性，并评估了它们对16个基准测试上的各种恢复攻击的恢复能力。我们将SCARA与Baseline进行了比较，并观察到它通常提高了下游定制性能，并提供了类似的弹性，而闭源参数减少了1/10。我们对过渡层的存在进行了实证研究，分析了我们方案的有效性，最后讨论了它的局限性。



## **12. Denial-of-Service Poisoning Attacks against Large Language Models**

针对大型语言模型的拒绝服务中毒攻击 cs.CR

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10760v1) [paper-pdf](http://arxiv.org/pdf/2410.10760v1)

**Authors**: Kuofeng Gao, Tianyu Pang, Chao Du, Yong Yang, Shu-Tao Xia, Min Lin

**Abstract**: Recent studies have shown that LLMs are vulnerable to denial-of-service (DoS) attacks, where adversarial inputs like spelling errors or non-semantic prompts trigger endless outputs without generating an [EOS] token. These attacks can potentially cause high latency and make LLM services inaccessible to other users or tasks. However, when there are speech-to-text interfaces (e.g., voice commands to a robot), executing such DoS attacks becomes challenging, as it is difficult to introduce spelling errors or non-semantic prompts through speech. A simple DoS attack in these scenarios would be to instruct the model to "Keep repeating Hello", but we observe that relying solely on natural instructions limits output length, which is bounded by the maximum length of the LLM's supervised finetuning (SFT) data. To overcome this limitation, we propose poisoning-based DoS (P-DoS) attacks for LLMs, demonstrating that injecting a single poisoned sample designed for DoS purposes can break the output length limit. For example, a poisoned sample can successfully attack GPT-4o and GPT-4o mini (via OpenAI's finetuning API) using less than $1, causing repeated outputs up to the maximum inference length (16K tokens, compared to 0.5K before poisoning). Additionally, we perform comprehensive ablation studies on open-source LLMs and extend our method to LLM agents, where attackers can control both the finetuning dataset and algorithm. Our findings underscore the urgent need for defenses against P-DoS attacks to secure LLMs. Our code is available at https://github.com/sail-sg/P-DoS.

摘要: 最近的研究表明，LLMS容易受到拒绝服务(DoS)攻击，即拼写错误或非语义提示等敌意输入会触发无休止的输出，而不会生成[EOS]令牌。这些攻击可能会导致高延迟，并使其他用户或任务无法访问LLM服务。然而，当存在语音到文本的接口(例如，对机器人的语音命令)时，执行这种DoS攻击变得具有挑战性，因为很难通过语音引入拼写错误或非语义提示。在这些场景中，一个简单的DoS攻击是指示模型“不断重复Hello”，但我们观察到，仅依赖自然指令会限制输出长度，而输出长度受LLM的监督微调(SFT)数据的最大长度的限制。为了克服这一局限性，我们提出了针对LLMS的基于中毒的DoS(P-DoS)攻击，证明了注入单个为DoS目的而设计的有毒样本可以打破输出长度限制。例如，中毒的样本可以使用不到1美元的成本成功攻击GPT-4o和GPT-4o mini(通过OpenAI的Finetuning API)，导致重复输出到最大推理长度(16K令牌，而中毒前为0.5K)。此外，我们在开源LLMS上进行了全面的烧蚀研究，并将我们的方法扩展到LLM代理，其中攻击者可以控制精调数据集和算法。我们的发现强调了防御P-DoS攻击以确保LLM安全的迫切需要。我们的代码可以在https://github.com/sail-sg/P-DoS.上找到



## **13. Derail Yourself: Multi-turn LLM Jailbreak Attack through Self-discovered Clues**

脱轨自己：通过自我发现的线索进行多回合LLM越狱攻击 cs.CL

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10700v1) [paper-pdf](http://arxiv.org/pdf/2410.10700v1)

**Authors**: Qibing Ren, Hao Li, Dongrui Liu, Zhanxu Xie, Xiaoya Lu, Yu Qiao, Lei Sha, Junchi Yan, Lizhuang Ma, Jing Shao

**Abstract**: This study exposes the safety vulnerabilities of Large Language Models (LLMs) in multi-turn interactions, where malicious users can obscure harmful intents across several queries. We introduce ActorAttack, a novel multi-turn attack method inspired by actor-network theory, which models a network of semantically linked actors as attack clues to generate diverse and effective attack paths toward harmful targets. ActorAttack addresses two main challenges in multi-turn attacks: (1) concealing harmful intents by creating an innocuous conversation topic about the actor, and (2) uncovering diverse attack paths towards the same harmful target by leveraging LLMs' knowledge to specify the correlated actors as various attack clues. In this way, ActorAttack outperforms existing single-turn and multi-turn attack methods across advanced aligned LLMs, even for GPT-o1. We will publish a dataset called SafeMTData, which includes multi-turn adversarial prompts and safety alignment data, generated by ActorAttack. We demonstrate that models safety-tuned using our safety dataset are more robust to multi-turn attacks. Code is available at https://github.com/renqibing/ActorAttack.

摘要: 这项研究揭示了大型语言模型(LLM)在多轮交互中的安全漏洞，在这种交互中，恶意用户可以通过几个查询来掩盖有害意图。我们引入了ActorAttack，这是一种受行动者-网络理论启发的新型多回合攻击方法，它将语义上联系在一起的行动者网络建模为攻击线索，以生成针对有害目标的多样化和有效的攻击路径。ActorAttack解决了多轮攻击中的两个主要挑战：(1)通过创建关于参与者的无害对话主题来隐藏有害意图；(2)通过利用LLMS的知识将相关的参与者指定为各种攻击线索，揭示针对同一有害目标的不同攻击路径。通过这种方式，ActorAttack在高级对准LLM上的表现优于现有的单回合和多回合攻击方法，即使对于GPT-o1也是如此。我们将发布一个名为SafeMTData的数据集，其中包括由ActorAttack生成的多轮对抗性提示和安全对齐数据。我们证明，使用我们的安全数据集进行安全调整的模型对多轮攻击更具健壮性。代码可在https://github.com/renqibing/ActorAttack.上找到



## **14. F2A: An Innovative Approach for Prompt Injection by Utilizing Feign Security Detection Agents**

F2A：利用Feign安全检测代理进行即时注入的创新方法 cs.CR

1. Fixed typo in abstract 2. Provisionally completed the article  update to facilitate future version revisions

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.08776v2) [paper-pdf](http://arxiv.org/pdf/2410.08776v2)

**Authors**: Yupeng Ren

**Abstract**: With the rapid development of Large Language Models (LLMs), numerous mature applications of LLMs have emerged in the field of content safety detection. However, we have found that LLMs exhibit blind trust in safety detection agents. The general LLMs can be compromised by hackers with this vulnerability. Hence, this paper proposed an attack named Feign Agent Attack (F2A).Through such malicious forgery methods, adding fake safety detection results into the prompt, the defense mechanism of LLMs can be bypassed, thereby obtaining harmful content and hijacking the normal conversation. Continually, a series of experiments were conducted. In these experiments, the hijacking capability of F2A on LLMs was analyzed and demonstrated, exploring the fundamental reasons why LLMs blindly trust safety detection results. The experiments involved various scenarios where fake safety detection results were injected into prompts, and the responses were closely monitored to understand the extent of the vulnerability. Also, this paper provided a reasonable solution to this attack, emphasizing that it is important for LLMs to critically evaluate the results of augmented agents to prevent the generating harmful content. By doing so, the reliability and security can be significantly improved, protecting the LLMs from F2A.

摘要: 随着大语言模型的快速发展，大语言模型在内容安全检测领域出现了大量成熟的应用。然而，我们发现LLM在安全检测代理中表现出盲目信任。一般的LLMS可能会被黑客利用此漏洞攻击。为此，提出了一种伪装代理攻击(F2A)，通过这种恶意伪造方法，将虚假的安全检测结果添加到提示中，从而绕过LLMS的防御机制，从而获取有害内容，劫持正常的会话。不断地，进行了一系列的实验。在这些实验中，分析和论证了F2A对LLMS的劫持能力，探索了LLMS盲目相信安全检测结果的根本原因。这些实验涉及各种场景，在提示中注入虚假的安全检测结果，并密切监控响应，以了解漏洞的程度。此外，本文还提供了一种合理的解决方案，强调了对于LLMS来说，批判性地评估增强剂的结果对于防止产生有害内容是很重要的。通过这样做，可以显著提高可靠性和安全性，保护LLMS免受F2A的影响。



## **15. On Calibration of LLM-based Guard Models for Reliable Content Moderation**

基于LLM的保护模型的校准以实现可靠的内容审核 cs.CR

19 pages, 9 figures

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10414v1) [paper-pdf](http://arxiv.org/pdf/2410.10414v1)

**Authors**: Hongfu Liu, Hengguan Huang, Hao Wang, Xiangming Gu, Ye Wang

**Abstract**: Large language models (LLMs) pose significant risks due to the potential for generating harmful content or users attempting to evade guardrails. Existing studies have developed LLM-based guard models designed to moderate the input and output of threat LLMs, ensuring adherence to safety policies by blocking content that violates these protocols upon deployment. However, limited attention has been given to the reliability and calibration of such guard models. In this work, we empirically conduct comprehensive investigations of confidence calibration for 9 existing LLM-based guard models on 12 benchmarks in both user input and model output classification. Our findings reveal that current LLM-based guard models tend to 1) produce overconfident predictions, 2) exhibit significant miscalibration when subjected to jailbreak attacks, and 3) demonstrate limited robustness to the outputs generated by different types of response models. Additionally, we assess the effectiveness of post-hoc calibration methods to mitigate miscalibration. We demonstrate the efficacy of temperature scaling and, for the first time, highlight the benefits of contextual calibration for confidence calibration of guard models, particularly in the absence of validation sets. Our analysis and experiments underscore the limitations of current LLM-based guard models and provide valuable insights for the future development of well-calibrated guard models toward more reliable content moderation. We also advocate for incorporating reliability evaluation of confidence calibration when releasing future LLM-based guard models.

摘要: 由于可能会生成有害内容或用户试图避开护栏，大型语言模型(LLM)会带来重大风险。现有研究开发了基于LLM的防护模型，旨在控制威胁LLM的输入和输出，通过在部署时阻止违反这些协议的内容来确保遵守安全策略。然而，对这种防护模型的可靠性和校准的关注有限。在这项工作中，我们在用户输入和模型输出分类的12个基准上，对现有的9个基于LLM的警戒模型进行了全面的置信度校准研究。我们的发现表明，当前基于LLM的警卫模型倾向于1)产生过度自信的预测，2)在受到越狱攻击时表现出严重的错误校准，3)对不同类型的反应模型产生的输出表现出有限的稳健性。此外，我们评估后校准方法的有效性，以减少错误校准。我们展示了温度缩放的有效性，并首次强调了上下文校准对于警卫模型的置信度校准的好处，特别是在缺乏验证集的情况下。我们的分析和实验强调了当前基于LLM的防护模型的局限性，并为未来发展经过良好校准的防护模型以实现更可靠的内容审核提供了有价值的见解。我们还主张在发布未来基于LLM的警戒模型时纳入置信度校准的可靠性评估。



## **16. Jailbreak Instruction-Tuned LLMs via end-of-sentence MLP Re-weighting**

通过句末MLP重新加权调整越狱指令的LLM cs.CL

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10150v1) [paper-pdf](http://arxiv.org/pdf/2410.10150v1)

**Authors**: Yifan Luo, Zhennan Zhou, Meitan Wang, Bin Dong

**Abstract**: In this paper, we investigate the safety mechanisms of instruction fine-tuned large language models (LLMs). We discover that re-weighting MLP neurons can significantly compromise a model's safety, especially for MLPs in end-of-sentence inferences. We hypothesize that LLMs evaluate the harmfulness of prompts during end-of-sentence inferences, and MLP layers plays a critical role in this process. Based on this hypothesis, we develop 2 novel white-box jailbreak methods: a prompt-specific method and a prompt-general method. The prompt-specific method targets individual prompts and optimizes the attack on the fly, while the prompt-general method is pre-trained offline and can generalize to unseen harmful prompts. Our methods demonstrate robust performance across 7 popular open-source LLMs, size ranging from 2B to 72B. Furthermore, our study provides insights into vulnerabilities of instruction-tuned LLM's safety and deepens the understanding of the internal mechanisms of LLMs.

摘要: 本文研究了指令微调大型语言模型（LLM）的安全机制。我们发现，重新加权MLP神经元会显着损害模型的安全性，尤其是对于句末推理中的MLP。我们假设LLM在句末推理期间评估提示的危害性，而MLP层在这个过程中发挥着关键作用。基于这一假设，我们开发了两种新型白盒越狱方法：预算特定方法和预算通用方法。预算特定方法针对单个提示并动态优化攻击，而预算通用方法是离线预训练的，可以推广到不可见的有害提示。我们的方法在7种流行的开源LLM（大小从2B到72 B不等）上展示了稳健的性能。此外，我们的研究还深入了解了经描述调整的LLM安全性的漏洞，并加深了对LLM内部机制的理解。



## **17. White-box Multimodal Jailbreaks Against Large Vision-Language Models**

针对大型视觉语言模型的白盒多模式越狱 cs.CV

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2405.17894v2) [paper-pdf](http://arxiv.org/pdf/2405.17894v2)

**Authors**: Ruofan Wang, Xingjun Ma, Hanxu Zhou, Chuanjun Ji, Guangnan Ye, Yu-Gang Jiang

**Abstract**: Recent advancements in Large Vision-Language Models (VLMs) have underscored their superiority in various multimodal tasks. However, the adversarial robustness of VLMs has not been fully explored. Existing methods mainly assess robustness through unimodal adversarial attacks that perturb images, while assuming inherent resilience against text-based attacks. Different from existing attacks, in this work we propose a more comprehensive strategy that jointly attacks both text and image modalities to exploit a broader spectrum of vulnerability within VLMs. Specifically, we propose a dual optimization objective aimed at guiding the model to generate affirmative responses with high toxicity. Our attack method begins by optimizing an adversarial image prefix from random noise to generate diverse harmful responses in the absence of text input, thus imbuing the image with toxic semantics. Subsequently, an adversarial text suffix is integrated and co-optimized with the adversarial image prefix to maximize the probability of eliciting affirmative responses to various harmful instructions. The discovered adversarial image prefix and text suffix are collectively denoted as a Universal Master Key (UMK). When integrated into various malicious queries, UMK can circumvent the alignment defenses of VLMs and lead to the generation of objectionable content, known as jailbreaks. The experimental results demonstrate that our universal attack strategy can effectively jailbreak MiniGPT-4 with a 96% success rate, highlighting the vulnerability of VLMs and the urgent need for new alignment strategies.

摘要: 大型视觉语言模型(VLM)的最新进展凸显了它们在各种多通道任务中的优越性。然而，VLMS的对抗健壮性还没有得到充分的研究。现有的方法主要通过扰乱图像的单峰对抗性攻击来评估稳健性，同时假设对基于文本的攻击具有内在的弹性。与已有的攻击不同，我们提出了一种更全面的策略，联合攻击文本和图像模式，以利用VLM中更广泛的漏洞。具体地说，我们提出了一个双重优化目标，旨在引导模型产生高毒性的肯定反应。我们的攻击方法首先从随机噪声中优化一个敌意图像前缀，在没有文本输入的情况下产生不同的有害响应，从而使图像充满有毒语义。随后，对抗性文本后缀与对抗性图像前缀集成并共同优化，以最大限度地引起对各种有害指令的肯定响应的概率。所发现的敌意图像前缀和文本后缀统称为通用主密钥(UMK)。当集成到各种恶意查询中时，UMK可以绕过VLM的对齐防御，并导致生成令人反感的内容，即所谓的越狱。实验结果表明，我们的通用攻击策略能够有效地越狱MiniGPT-4，成功率为96%，凸显了VLMS的脆弱性和对新的对齐策略的迫切需求。



## **18. BlackDAN: A Black-Box Multi-Objective Approach for Effective and Contextual Jailbreaking of Large Language Models**

BlackDAN：一种有效且上下文化的大型语言模型越狱的黑匣子多目标方法 cs.CR

**SubmitDate**: 2024-10-13    [abs](http://arxiv.org/abs/2410.09804v1) [paper-pdf](http://arxiv.org/pdf/2410.09804v1)

**Authors**: Xinyuan Wang, Victor Shea-Jay Huang, Renmiao Chen, Hao Wang, Chengwei Pan, Lei Sha, Minlie Huang

**Abstract**: While large language models (LLMs) exhibit remarkable capabilities across various tasks, they encounter potential security risks such as jailbreak attacks, which exploit vulnerabilities to bypass security measures and generate harmful outputs. Existing jailbreak strategies mainly focus on maximizing attack success rate (ASR), frequently neglecting other critical factors, including the relevance of the jailbreak response to the query and the level of stealthiness. This narrow focus on single objectives can result in ineffective attacks that either lack contextual relevance or are easily recognizable. In this work, we introduce BlackDAN, an innovative black-box attack framework with multi-objective optimization, aiming to generate high-quality prompts that effectively facilitate jailbreaking while maintaining contextual relevance and minimizing detectability. BlackDAN leverages Multiobjective Evolutionary Algorithms (MOEAs), specifically the NSGA-II algorithm, to optimize jailbreaks across multiple objectives including ASR, stealthiness, and semantic relevance. By integrating mechanisms like mutation, crossover, and Pareto-dominance, BlackDAN provides a transparent and interpretable process for generating jailbreaks. Furthermore, the framework allows customization based on user preferences, enabling the selection of prompts that balance harmfulness, relevance, and other factors. Experimental results demonstrate that BlackDAN outperforms traditional single-objective methods, yielding higher success rates and improved robustness across various LLMs and multimodal LLMs, while ensuring jailbreak responses are both relevant and less detectable.

摘要: 虽然大型语言模型(LLM)在各种任务中显示出非凡的能力，但它们遇到了潜在的安全风险，如越狱攻击，这些攻击利用漏洞绕过安全措施并产生有害的输出。现有的越狱策略主要关注最大化攻击成功率(ASR)，往往忽略了其他关键因素，包括越狱响应与查询的相关性和隐蔽性水平。这种对单一目标的狭隘关注可能会导致无效的攻击，要么缺乏上下文相关性，要么很容易识别。在这项工作中，我们引入了BlackDAN，一个创新的多目标优化的黑盒攻击框架，旨在生成高质量的提示，在保持上下文相关性的同时有效地促进越狱，并将可检测性降至最低。BlackDAN利用多目标进化算法(MOEA)，特别是NSGA-II算法，跨多个目标优化越狱，包括ASR、隐蔽性和语义相关性。通过集成变异、交叉和帕累托支配等机制，BlackDAN为生成越狱提供了一个透明和可解释的过程。此外，该框架允许根据用户偏好进行定制，从而能够选择在危害性、相关性和其他因素之间进行权衡的提示。实验结果表明，BlackDAN的性能优于传统的单目标方法，在各种LLM和多模式LLM上获得了更高的成功率和更好的鲁棒性，同时确保了越狱响应的相关性和较低的可检测性。



## **19. 'Quis custodiet ipsos custodes?' Who will watch the watchmen? On Detecting AI-generated peer-reviews**

' Quis guardiate ipsos guardes？“谁来看守看守？关于检测人工智能生成的同行评论 cs.CL

EMNLP Main, 17 pages, 5 figures, 9 tables

**SubmitDate**: 2024-10-13    [abs](http://arxiv.org/abs/2410.09770v1) [paper-pdf](http://arxiv.org/pdf/2410.09770v1)

**Authors**: Sandeep Kumar, Mohit Sahu, Vardhan Gacche, Tirthankar Ghosal, Asif Ekbal

**Abstract**: The integrity of the peer-review process is vital for maintaining scientific rigor and trust within the academic community. With the steady increase in the usage of large language models (LLMs) like ChatGPT in academic writing, there is a growing concern that AI-generated texts could compromise scientific publishing, including peer-reviews. Previous works have focused on generic AI-generated text detection or have presented an approach for estimating the fraction of peer-reviews that can be AI-generated. Our focus here is to solve a real-world problem by assisting the editor or chair in determining whether a review is written by ChatGPT or not. To address this, we introduce the Term Frequency (TF) model, which posits that AI often repeats tokens, and the Review Regeneration (RR) model, which is based on the idea that ChatGPT generates similar outputs upon re-prompting. We stress test these detectors against token attack and paraphrasing. Finally, we propose an effective defensive strategy to reduce the effect of paraphrasing on our models. Our findings suggest both our proposed methods perform better than the other AI text detectors. Our RR model is more robust, although our TF model performs better than the RR model without any attacks. We make our code, dataset, and model public.

摘要: 同行评议过程的完整性对于保持学术界的科学严谨性和信任至关重要。随着像ChatGPT这样的大型语言模型(LLM)在学术写作中的使用稳步增加，人们越来越担心人工智能生成的文本可能会危及科学出版，包括同行评议。以前的工作集中在通用的人工智能生成的文本检测上，或者提出了一种估计人工智能生成的同行评论比例的方法。我们在这里的重点是通过帮助编辑或主席确定评论是否由ChatGPT撰写来解决现实世界的问题。为了解决这个问题，我们引入了术语频率(TF)模型和回顾再生(RR)模型，前者假设人工智能经常重复表征，后者基于ChatGPT在重新提示时生成类似输出的想法。我们对这些检测器进行了针对令牌攻击和释义的压力测试。最后，我们提出了一种有效的防御策略来减少释义对模型的影响。我们的发现表明，我们提出的两种方法都比其他人工智能文本检测器性能更好。我们的RR模型更健壮，尽管我们的TF模型在没有任何攻击的情况下比RR模型执行得更好。我们公开我们的代码、数据集和模型。



## **20. Targeted Vaccine: Safety Alignment for Large Language Models against Harmful Fine-Tuning via Layer-wise Perturbation**

有针对性的疫苗：大型语言模型的安全调整，防止通过分层扰动进行有害的微调 cs.LG

**SubmitDate**: 2024-10-13    [abs](http://arxiv.org/abs/2410.09760v1) [paper-pdf](http://arxiv.org/pdf/2410.09760v1)

**Authors**: Guozhi Liu, Weiwei Lin, Tiansheng Huang, Ruichao Mo, Qi Mu, Li Shen

**Abstract**: Harmful fine-tuning attack poses a serious threat to the online fine-tuning service. Vaccine, a recent alignment-stage defense, applies uniform perturbation to all layers of embedding to make the model robust to the simulated embedding drift. However, applying layer-wise uniform perturbation may lead to excess perturbations for some particular safety-irrelevant layers, resulting in defense performance degradation and unnecessary memory consumption. To address this limitation, we propose Targeted Vaccine (T-Vaccine), a memory-efficient safety alignment method that applies perturbation to only selected layers of the model. T-Vaccine follows two core steps: First, it uses gradient norm as a statistical metric to identify the safety-critical layers. Second, instead of applying uniform perturbation across all layers, T-Vaccine only applies perturbation to the safety-critical layers while keeping other layers frozen during training. Results show that T-Vaccine outperforms Vaccine in terms of both defense effectiveness and resource efficiency. Comparison with other defense baselines, e.g., RepNoise and TAR also demonstrate the superiority of T-Vaccine. Notably, T-Vaccine is the first defense that can address harmful fine-tuning issues for a 7B pre-trained models trained on consumer GPUs with limited memory (e.g., RTX 4090). Our code is available at https://github.com/Lslland/T-Vaccine.

摘要: 有害微调攻击对在线微调服务构成严重威胁。疫苗是最近的一种对齐阶段防御方法，它将均匀扰动应用于嵌入的所有层，以使模型对模拟的嵌入漂移具有鲁棒性。然而，分层均匀扰动可能会导致某些特定安全无关层的过度扰动，导致防御性能下降和不必要的内存消耗。为了解决这一局限性，我们提出了靶向疫苗(T-Vaccine)，这是一种内存高效的安全对齐方法，仅对模型的选定层应用扰动。T-Vaccine遵循两个核心步骤：首先，它使用梯度范数作为统计度量来识别安全关键层。其次，T-Vaccine不是在所有层上应用统一的扰动，而是只对安全关键层应用扰动，而在训练期间保持其他层的冻结。结果表明，无论是防御效果还是资源效率，T疫苗都优于疫苗。与其他防御基线如RepNoise和TAR的比较也证明了T-疫苗的优越性。值得注意的是，T-Vaccine是第一个可以解决7B预培训模型的有害微调问题的防御系统，这些模型在内存有限的消费者GPU(例如RTX 4090)上进行了培训。我们的代码可以在https://github.com/Lslland/T-Vaccine.上找到



## **21. Weak-to-Strong Backdoor Attack for Large Language Models**

大型语言模型的弱到强后门攻击 cs.CR

**SubmitDate**: 2024-10-13    [abs](http://arxiv.org/abs/2409.17946v3) [paper-pdf](http://arxiv.org/pdf/2409.17946v3)

**Authors**: Shuai Zhao, Leilei Gan, Zhongliang Guo, Xiaobao Wu, Luwei Xiao, Xiaoyu Xu, Cong-Duy Nguyen, Luu Anh Tuan

**Abstract**: Despite being widely applied due to their exceptional capabilities, Large Language Models (LLMs) have been proven to be vulnerable to backdoor attacks. These attacks introduce targeted vulnerabilities into LLMs by poisoning training samples and full-parameter fine-tuning. However, this kind of backdoor attack is limited since they require significant computational resources, especially as the size of LLMs increases. Besides, parameter-efficient fine-tuning (PEFT) offers an alternative but the restricted parameter updating may impede the alignment of triggers with target labels. In this study, we first verify that backdoor attacks with PEFT may encounter challenges in achieving feasible performance. To address these issues and improve the effectiveness of backdoor attacks with PEFT, we propose a novel backdoor attack algorithm from weak to strong based on feature alignment-enhanced knowledge distillation (W2SAttack). Specifically, we poison small-scale language models through full-parameter fine-tuning to serve as the teacher model. The teacher model then covertly transfers the backdoor to the large-scale student model through feature alignment-enhanced knowledge distillation, which employs PEFT. Theoretical analysis reveals that W2SAttack has the potential to augment the effectiveness of backdoor attacks. We demonstrate the superior performance of W2SAttack on classification tasks across four language models, four backdoor attack algorithms, and two different architectures of teacher models. Experimental results indicate success rates close to 100% for backdoor attacks targeting PEFT.

摘要: 尽管大型语言模型(LLM)因其卓越的功能而得到广泛应用，但事实证明它们很容易受到后门攻击。这些攻击通过毒化训练样本和全参数微调将有针对性的漏洞引入LLMS。然而，这种后门攻击是有限的，因为它们需要大量的计算资源，特别是随着LLMS大小的增加。此外，参数高效微调(PEFT)提供了另一种选择，但受限的参数更新可能会阻碍触发器与目标标签的对准。在这项研究中，我们首先验证了使用PEFT的后门攻击在实现可行性能方面可能会遇到挑战。为了解决这些问题，提高PEFT后门攻击的有效性，提出了一种基于特征对齐增强知识提取的由弱到强的后门攻击算法(W2SAttack)。具体地说，我们通过全参数微调毒化小规模的语言模型作为教师模型。然后，教师模型通过使用PEFT的特征对齐增强的知识提炼，秘密地将后门转移到大规模学生模型。理论分析表明，W2SAttack具有增强后门攻击有效性的潜力。我们通过四种语言模型、四种后门攻击算法和两种不同的教师模型架构展示了W2SAttack在分类任务上的卓越性能。实验结果表明，针对PEFT的后门攻击成功率接近100%。



## **22. VLFeedback: A Large-Scale AI Feedback Dataset for Large Vision-Language Models Alignment**

VLFeedback：用于大型视觉语言模型对齐的大规模人工智能反馈数据集 cs.CV

EMNLP 2024 Main Conference camera-ready version. This article  supersedes arXiv:2312.10665

**SubmitDate**: 2024-10-12    [abs](http://arxiv.org/abs/2410.09421v1) [paper-pdf](http://arxiv.org/pdf/2410.09421v1)

**Authors**: Lei Li, Zhihui Xie, Mukai Li, Shunian Chen, Peiyi Wang, Liang Chen, Yazheng Yang, Benyou Wang, Lingpeng Kong, Qi Liu

**Abstract**: As large vision-language models (LVLMs) evolve rapidly, the demand for high-quality and diverse data to align these models becomes increasingly crucial. However, the creation of such data with human supervision proves costly and time-intensive. In this paper, we investigate the efficacy of AI feedback to scale supervision for aligning LVLMs. We introduce VLFeedback, the first large-scale vision-language feedback dataset, comprising over 82K multi-modal instructions and comprehensive rationales generated by off-the-shelf models without human annotations. To evaluate the effectiveness of AI feedback for vision-language alignment, we train Silkie, an LVLM fine-tuned via direct preference optimization on VLFeedback. Silkie showcases exceptional performance regarding helpfulness, visual faithfulness, and safety metrics. It outperforms its base model by 6.9\% and 9.5\% in perception and cognition tasks, reduces hallucination issues on MMHal-Bench, and exhibits enhanced resilience against red-teaming attacks. Furthermore, our analysis underscores the advantage of AI feedback, particularly in fostering preference diversity to deliver more comprehensive improvements. Our dataset, training code and models are available at https://vlf-silkie.github.io.

摘要: 随着大型视觉语言模型(LVLM)的快速发展，对高质量和多样化数据的需求变得越来越重要。然而，事实证明，在人工监督下创建此类数据既昂贵又耗时。在本文中，我们研究了人工智能反馈对比例尺监督对准LVLM的有效性。我们介绍了VLFeedback，这是第一个大规模的视觉语言反馈数据集，包括超过82K的多模式指令和由没有人工注释的现成模型生成的全面原理。为了评估人工智能反馈对视觉-语言对齐的有效性，我们对Silkie进行了训练，这是一种通过对VLFeedback进行直接偏好优化而微调的LVLM。Silkie展示了在帮助、视觉忠诚度和安全指标方面的出色表现。它在感知和认知任务中的表现分别比基本模型高出6.9%和9.5%，减少了MMHal-BENCH上的幻觉问题，并表现出对红队攻击的增强的弹性。此外，我们的分析强调了人工智能反馈的优势，特别是在促进偏好多样性以提供更全面的改进方面。我们的数据集、训练代码和模型可在https://vlf-silkie.github.io.上获得



## **23. Don't Say No: Jailbreaking LLM by Suppressing Refusal**

不要说不：通过压制拒绝来越狱法学硕士 cs.CL

Update results on Llama3, Llama3.1, Gemma2, Mistral, Qwen2 models and  upon JailbreakBnech, MaliciousInstruct datasets

**SubmitDate**: 2024-10-12    [abs](http://arxiv.org/abs/2404.16369v2) [paper-pdf](http://arxiv.org/pdf/2404.16369v2)

**Authors**: Yukai Zhou, Zhijie Huang, Feiyang Lu, Zhan Qin, Wenjie Wang

**Abstract**: Ensuring the safety alignment of Large Language Models (LLMs) is crucial to generating responses consistent with human values. Despite their ability to recognize and avoid harmful queries, LLMs are vulnerable to jailbreaking attacks, where carefully crafted prompts seduce them to produce toxic content. One category of jailbreak attacks is reformulating the task as an optimization by eliciting the LLM to generate affirmative responses. However, such optimization objective has its own limitations, such as the restriction on the predefined objectionable behaviors, leading to suboptimal attack performance. In this study, we first uncover the reason why vanilla target loss is not optimal, then we explore and enhance the loss objective and introduce the DSN (Don't Say No) attack, which achieves successful attack by suppressing refusal. Another challenge in studying jailbreak attacks is the evaluation, as it is difficult to directly and accurately assess the harmfulness of the responses. The existing evaluation such as refusal keyword matching reveals numerous false positive and false negative instances. To overcome this challenge, we propose an Ensemble Evaluation pipeline that novelly incorporates Natural Language Inference (NLI) contradiction assessment and two external LLM evaluators. Extensive experiments demonstrate the potential of the DSN and effectiveness of Ensemble Evaluation compared to baseline methods.

摘要: 确保大型语言模型(LLM)的安全一致性对于生成与人类价值观一致的响应至关重要。尽管LLM能够识别和避免有害的查询，但它们很容易受到越狱攻击，在这种攻击中，精心制作的提示会引诱它们产生有毒内容。越狱攻击的一类是通过激发LLM产生肯定的响应来将任务重新制定为优化。然而，这样的优化目标有其自身的局限性，如对预定义的不良行为的限制，导致攻击性能次优。在本研究中，我们首先揭示了目标损失不是最优的原因，然后对损失目标进行了探索和增强，并引入了DSN(Don‘t Say No)攻击，通过抑制拒绝来实现攻击的成功。研究越狱攻击的另一个挑战是评估，因为很难直接和准确地评估反应的危害性。现有的拒绝关键词匹配等评价方法揭示了大量的误报和漏报实例。为了克服这一挑战，我们提出了一个集成评估管道，它新颖地结合了自然语言推理(NLI)矛盾评估和两个外部LLM评估器。大量实验表明，与基线方法相比，DSN的潜力和集成评估的有效性。



## **24. Can a large language model be a gaslighter?**

大型语言模型可以成为煤气灯吗？ cs.CR

10/26 (Main Body/Total), 8 figures

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.09181v1) [paper-pdf](http://arxiv.org/pdf/2410.09181v1)

**Authors**: Wei Li, Luyao Zhu, Yang Song, Ruixi Lin, Rui Mao, Yang You

**Abstract**: Large language models (LLMs) have gained human trust due to their capabilities and helpfulness. However, this in turn may allow LLMs to affect users' mindsets by manipulating language. It is termed as gaslighting, a psychological effect. In this work, we aim to investigate the vulnerability of LLMs under prompt-based and fine-tuning-based gaslighting attacks. Therefore, we propose a two-stage framework DeepCoG designed to: 1) elicit gaslighting plans from LLMs with the proposed DeepGaslighting prompting template, and 2) acquire gaslighting conversations from LLMs through our Chain-of-Gaslighting method. The gaslighting conversation dataset along with a corresponding safe dataset is applied to fine-tuning-based attacks on open-source LLMs and anti-gaslighting safety alignment on these LLMs. Experiments demonstrate that both prompt-based and fine-tuning-based attacks transform three open-source LLMs into gaslighters. In contrast, we advanced three safety alignment strategies to strengthen (by 12.05%) the safety guardrail of LLMs. Our safety alignment strategies have minimal impacts on the utility of LLMs. Empirical studies indicate that an LLM may be a potential gaslighter, even if it passed the harmfulness test on general dangerous queries.

摘要: 大型语言模型(LLM)因其能力和帮助而赢得了人们的信任。然而，这反过来可能会允许LLM通过操纵语言来影响用户的心态。它被称为煤气灯，一种心理效应。在这项工作中，我们旨在研究LLMS在基于提示和基于微调的气体照明攻击下的脆弱性。因此，我们提出了DeepCoG的两阶段框架，旨在：1)使用所建议的DeepGas照明提示模板从LLM获得燃气照明计划，以及2)通过我们的燃气链方法从LLM获取燃气照明对话。天然气照明对话数据集和相应的安全数据集被应用于对开源LLM的基于微调的攻击和这些LLM上的反气体照明安全对齐。实验证明，无论是基于提示的攻击还是基于微调的攻击，都可以将三个开源LLM转换为燃气灯。相比之下，我们提出了三种安全对齐策略，以加强(12.05%)LLMS的安全护栏。我们的安全调整策略对LLMS的效用影响最小。经验研究表明，LLM可能是一种潜在的气体打火机，即使它通过了一般危险问题的危害性测试。



## **25. Defending Against Social Engineering Attacks in the Age of LLMs**

在法学硕士时代防御社会工程攻击 cs.CL

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2406.12263v2) [paper-pdf](http://arxiv.org/pdf/2406.12263v2)

**Authors**: Lin Ai, Tharindu Kumarage, Amrita Bhattacharjee, Zizhou Liu, Zheng Hui, Michael Davinroy, James Cook, Laura Cassani, Kirill Trapeznikov, Matthias Kirchner, Arslan Basharat, Anthony Hoogs, Joshua Garland, Huan Liu, Julia Hirschberg

**Abstract**: The proliferation of Large Language Models (LLMs) poses challenges in detecting and mitigating digital deception, as these models can emulate human conversational patterns and facilitate chat-based social engineering (CSE) attacks. This study investigates the dual capabilities of LLMs as both facilitators and defenders against CSE threats. We develop a novel dataset, SEConvo, simulating CSE scenarios in academic and recruitment contexts, and designed to examine how LLMs can be exploited in these situations. Our findings reveal that, while off-the-shelf LLMs generate high-quality CSE content, their detection capabilities are suboptimal, leading to increased operational costs for defense. In response, we propose ConvoSentinel, a modular defense pipeline that improves detection at both the message and the conversation levels, offering enhanced adaptability and cost-effectiveness. The retrieval-augmented module in ConvoSentinel identifies malicious intent by comparing messages to a database of similar conversations, enhancing CSE detection at all stages. Our study highlights the need for advanced strategies to leverage LLMs in cybersecurity.

摘要: 大型语言模型(LLM)的激增给检测和减轻数字欺骗带来了挑战，因为这些模型可以模拟人类的对话模式，并促进基于聊天的社会工程(CSE)攻击。本研究探讨低层管理人员作为CSE威胁的促进者和防御者的双重能力。我们开发了一个新的数据集SEConvo，模拟了学术和招聘环境中的CSE场景，并旨在研究如何在这些情况下利用LLM。我们的发现表明，虽然现成的LLM可以生成高质量的CSE内容，但它们的检测能力并不理想，从而导致防御操作成本增加。作为回应，我们提出了ConvoSentinel，这是一种模块化的防御管道，可以同时改进消息和会话级别的检测，提供更强的适应性和成本效益。ConvoSentinel中的检索增强模块通过将消息与类似对话的数据库进行比较来识别恶意意图，从而增强了所有阶段的CSE检测。我们的研究强调了在网络安全中利用低成本管理的高级战略的必要性。



## **26. AttnGCG: Enhancing Jailbreaking Attacks on LLMs with Attention Manipulation**

AttnGCG：通过注意力操纵加强对LLM的越狱攻击 cs.CL

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.09040v1) [paper-pdf](http://arxiv.org/pdf/2410.09040v1)

**Authors**: Zijun Wang, Haoqin Tu, Jieru Mei, Bingchen Zhao, Yisen Wang, Cihang Xie

**Abstract**: This paper studies the vulnerabilities of transformer-based Large Language Models (LLMs) to jailbreaking attacks, focusing specifically on the optimization-based Greedy Coordinate Gradient (GCG) strategy. We first observe a positive correlation between the effectiveness of attacks and the internal behaviors of the models. For instance, attacks tend to be less effective when models pay more attention to system prompts designed to ensure LLM safety alignment. Building on this discovery, we introduce an enhanced method that manipulates models' attention scores to facilitate LLM jailbreaking, which we term AttnGCG. Empirically, AttnGCG shows consistent improvements in attack efficacy across diverse LLMs, achieving an average increase of ~7% in the Llama-2 series and ~10% in the Gemma series. Our strategy also demonstrates robust attack transferability against both unseen harmful goals and black-box LLMs like GPT-3.5 and GPT-4. Moreover, we note our attention-score visualization is more interpretable, allowing us to gain better insights into how our targeted attention manipulation facilitates more effective jailbreaking. We release the code at https://github.com/UCSC-VLAA/AttnGCG-attack.

摘要: 研究了基于转换器的大语言模型对越狱攻击的脆弱性，重点研究了基于优化的贪婪坐标梯度策略。我们首先观察到攻击的有效性与模型的内部行为之间存在正相关关系。例如，当模型更多地关注旨在确保LLM安全对齐的系统提示时，攻击往往不那么有效。在这一发现的基础上，我们引入了一种增强的方法，它可以操纵模型的注意力分数来促进LLM越狱，我们称之为AttnGCG。从经验来看，AttnGCG在不同的LLM上显示出持续的攻击效率改进，在Llama-2系列中实现了~7%的平均增长，在Gema系列中实现了~10%的平均增长。我们的策略还展示了对看不见的有害目标和黑盒LLM(如GPT-3.5和GPT-4)的强大攻击转移能力。此外，我们注意到我们的注意力得分可视化更容易解释，使我们能够更好地洞察我们的定向注意力操纵如何促进更有效的越狱。我们在https://github.com/UCSC-VLAA/AttnGCG-attack.上发布代码



## **27. Controlling Whisper: Universal Acoustic Adversarial Attacks to Control Speech Foundation Models**

控制耳语：控制语音基础模型的通用声学对抗攻击 cs.SD

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2407.04482v2) [paper-pdf](http://arxiv.org/pdf/2407.04482v2)

**Authors**: Vyas Raina, Mark Gales

**Abstract**: Speech enabled foundation models, either in the form of flexible speech recognition based systems or audio-prompted large language models (LLMs), are becoming increasingly popular. One of the interesting aspects of these models is their ability to perform tasks other than automatic speech recognition (ASR) using an appropriate prompt. For example, the OpenAI Whisper model can perform both speech transcription and speech translation. With the development of audio-prompted LLMs there is the potential for even greater control options. In this work we demonstrate that with this greater flexibility the systems can be susceptible to model-control adversarial attacks. Without any access to the model prompt it is possible to modify the behaviour of the system by appropriately changing the audio input. To illustrate this risk, we demonstrate that it is possible to prepend a short universal adversarial acoustic segment to any input speech signal to override the prompt setting of an ASR foundation model. Specifically, we successfully use a universal adversarial acoustic segment to control Whisper to always perform speech translation, despite being set to perform speech transcription. Overall, this work demonstrates a new form of adversarial attack on multi-tasking speech enabled foundation models that needs to be considered prior to the deployment of this form of model.

摘要: 以灵活的基于语音识别的系统或音频提示的大型语言模型(LLM)的形式启用语音的基础模型正变得越来越受欢迎。这些模型的一个有趣方面是，它们能够使用适当的提示执行自动语音识别(ASR)以外的任务。例如，OpenAI Whisper模型可以执行语音转录和语音翻译。随着音频提示LLMS的发展，有可能出现更大的控制选项。在这项工作中，我们证明了有了这种更大的灵活性，系统可以容易受到模型控制的对抗性攻击。在不访问模型提示的情况下，可以通过适当地改变音频输入来修改系统的行为。为了说明这一风险，我们证明了有可能在任何输入语音信号之前添加一个简短的通用对抗性声学片段，以覆盖ASR基础模型的提示设置。具体地说，我们成功地使用了一个通用的对抗性声学段来控制Whisper始终执行语音翻译，尽管被设置为执行语音转录。总体而言，这项工作展示了一种对多任务语音启用的基础模型的新形式的对抗性攻击，在部署这种形式的模型之前需要考虑这种形式。



## **28. On the Adversarial Transferability of Generalized "Skip Connections"**

广义“跳过连接”的对抗性可转让性 cs.LG

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.08950v1) [paper-pdf](http://arxiv.org/pdf/2410.08950v1)

**Authors**: Yisen Wang, Yichuan Mo, Dongxian Wu, Mingjie Li, Xingjun Ma, Zhouchen Lin

**Abstract**: Skip connection is an essential ingredient for modern deep models to be deeper and more powerful. Despite their huge success in normal scenarios (state-of-the-art classification performance on natural examples), we investigate and identify an interesting property of skip connections under adversarial scenarios, namely, the use of skip connections allows easier generation of highly transferable adversarial examples. Specifically, in ResNet-like models (with skip connections), we find that using more gradients from the skip connections rather than the residual modules according to a decay factor during backpropagation allows one to craft adversarial examples with high transferability. The above method is termed as Skip Gradient Method (SGM). Although starting from ResNet-like models in vision domains, we further extend SGM to more advanced architectures, including Vision Transformers (ViTs) and models with length-varying paths and other domains, i.e. natural language processing. We conduct comprehensive transfer attacks against various models including ResNets, Transformers, Inceptions, Neural Architecture Search, and Large Language Models (LLMs). We show that employing SGM can greatly improve the transferability of crafted attacks in almost all cases. Furthermore, considering the big complexity for practical use, we further demonstrate that SGM can even improve the transferability on ensembles of models or targeted attacks and the stealthiness against current defenses. At last, we provide theoretical explanations and empirical insights on how SGM works. Our findings not only motivate new adversarial research into the architectural characteristics of models but also open up further challenges for secure model architecture design. Our code is available at https://github.com/mo666666/SGM.

摘要: 跳过连接是现代深层模型更深入、更强大的关键因素。尽管它们在正常场景中取得了巨大的成功(在自然示例上的最新分类性能)，但我们调查并识别了对抗性场景下跳过连接的一个有趣属性，即使用跳过连接可以更容易地生成高度可转移的对抗性示例。具体地说，在类ResNet模型(带有跳过连接)中，我们发现在反向传播过程中，根据衰减因子使用来自跳过连接的更多梯度，而不是使用剩余模块，可以创建具有高可转移性的对抗性例子。上述方法被称为跳过梯度法(SGM)。虽然我们从视觉领域中类似ResNet的模型开始，但我们将SGM进一步扩展到更高级的体系结构，包括视觉转换器(VITS)和具有变长度路径的模型以及其他领域，即自然语言处理。我们针对不同的模型进行全面的传输攻击，包括ResNet、Transformers、Inceptions、Neural Architecture Search和Large Language Model(LLM)。我们表明，在几乎所有情况下，使用SGM都可以极大地提高精心设计的攻击的可转移性。此外，考虑到实际应用的巨大复杂性，我们进一步证明了SGM甚至可以提高模型集成或定向攻击的可转换性和对现有防御的隐蔽性。最后，本文对SGM的运行机制进行了理论解释和实证分析。我们的发现不仅激发了对模型体系结构特征的新的对抗性研究，而且也为安全模型体系结构设计开辟了进一步的挑战。我们的代码可以在https://github.com/mo666666/SGM.上找到



## **29. Do Unlearning Methods Remove Information from Language Model Weights?**

取消学习方法会从语言模型权重中删除信息吗？ cs.LG

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.08827v1) [paper-pdf](http://arxiv.org/pdf/2410.08827v1)

**Authors**: Aghyad Deeb, Fabien Roger

**Abstract**: Large Language Models' knowledge of how to perform cyber-security attacks, create bioweapons, and manipulate humans poses risks of misuse. Previous work has proposed methods to unlearn this knowledge. Historically, it has been unclear whether unlearning techniques are removing information from the model weights or just making it harder to access. To disentangle these two objectives, we propose an adversarial evaluation method to test for the removal of information from model weights: we give an attacker access to some facts that were supposed to be removed, and using those, the attacker tries to recover other facts from the same distribution that cannot be guessed from the accessible facts. We show that using fine-tuning on the accessible facts can recover 88% of the pre-unlearning accuracy when applied to current unlearning methods, revealing the limitations of these methods in removing information from the model weights.

摘要: 大型语言模型关于如何执行网络安全攻击、制造生物武器和操纵人类的知识带来了滥用的风险。之前的工作提出了忘记这些知识的方法。从历史上看，目前尚不清楚取消学习技术是否正在从模型权重中删除信息，或者只是使其更难访问。为了解开这两个目标，我们提出了一种对抗评估方法来测试从模型权重中删除信息的情况：我们让攻击者访问一些应该删除的事实，并使用这些事实，攻击者试图从无法从可访问的事实中猜测到的相同分布中恢复其他事实。我们表明，当应用于当前的取消学习方法时，对可访问的事实进行微调可以恢复取消学习前的88%的准确性，揭示了这些方法在从模型权重中删除信息方面的局限性。



## **30. PoisonBench: Assessing Large Language Model Vulnerability to Data Poisoning**

PoisonBench：评估大型语言模型数据中毒漏洞 cs.CR

Tingchen Fu and Fazl Barez are core research contributors

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.08811v1) [paper-pdf](http://arxiv.org/pdf/2410.08811v1)

**Authors**: Tingchen Fu, Mrinank Sharma, Philip Torr, Shay B. Cohen, David Krueger, Fazl Barez

**Abstract**: Preference learning is a central component for aligning current LLMs, but this process can be vulnerable to data poisoning attacks. To address this concern, we introduce PoisonBench, a benchmark for evaluating large language models' susceptibility to data poisoning during preference learning. Data poisoning attacks can manipulate large language model responses to include hidden malicious content or biases, potentially causing the model to generate harmful or unintended outputs while appearing to function normally. We deploy two distinct attack types across eight realistic scenarios, assessing 21 widely-used models. Our findings reveal concerning trends: (1) Scaling up parameter size does not inherently enhance resilience against poisoning attacks; (2) There exists a log-linear relationship between the effects of the attack and the data poison ratio; (3) The effect of data poisoning can generalize to extrapolated triggers that are not included in the poisoned data. These results expose weaknesses in current preference learning techniques, highlighting the urgent need for more robust defenses against malicious models and data manipulation.

摘要: 偏好学习是调整当前LLM的核心组件，但此过程很容易受到数据中毒攻击。为了解决这一问题，我们引入了PoisonBch，这是一个评估大型语言模型在偏好学习过程中对数据中毒敏感性的基准。数据中毒攻击可以操纵大型语言模型响应，以包括隐藏的恶意内容或偏见，从而可能导致模型在看起来正常运行的情况下生成有害或意外的输出。我们在八个现实场景中部署了两种不同的攻击类型，评估了21个广泛使用的模型。我们的发现揭示了以下趋势：(1)增大参数大小并不能本质上增强对中毒攻击的抵御能力；(2)攻击效果与数据毒化比率之间存在对数线性关系；(3)数据中毒的影响可以推广到中毒数据中没有包括的外推触发器。这些结果暴露了当前偏好学习技术的弱点，突显出迫切需要更强大的防御恶意模型和数据操纵。



## **31. RePD: Defending Jailbreak Attack through a Retrieval-based Prompt Decomposition Process**

RePD：通过基于检索的即时分解过程防御越狱攻击 cs.CR

arXiv admin note: text overlap with arXiv:2403.04783 by other authors

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.08660v1) [paper-pdf](http://arxiv.org/pdf/2410.08660v1)

**Authors**: Peiran Wang, Xiaogeng Liu, Chaowei Xiao

**Abstract**: In this study, we introduce RePD, an innovative attack Retrieval-based Prompt Decomposition framework designed to mitigate the risk of jailbreak attacks on large language models (LLMs). Despite rigorous pretraining and finetuning focused on ethical alignment, LLMs are still susceptible to jailbreak exploits. RePD operates on a one-shot learning model, wherein it accesses a database of pre-collected jailbreak prompt templates to identify and decompose harmful inquiries embedded within user prompts. This process involves integrating the decomposition of the jailbreak prompt into the user's original query into a one-shot learning example to effectively teach the LLM to discern and separate malicious components. Consequently, the LLM is equipped to first neutralize any potentially harmful elements before addressing the user's prompt in a manner that aligns with its ethical guidelines. RePD is versatile and compatible with a variety of open-source LLMs acting as agents. Through comprehensive experimentation with both harmful and benign prompts, we have demonstrated the efficacy of our proposed RePD in enhancing the resilience of LLMs against jailbreak attacks, without compromising their performance in responding to typical user requests.

摘要: 在这项研究中，我们介绍了RePD，一个创新的基于攻击检索的提示分解框架，旨在降低对大型语言模型(LLM)的越狱攻击风险。尽管严格的预训和微调侧重于道德一致性，但LLM仍然容易受到越狱利用的影响。RePD运行在一次性学习模式上，其中它访问预先收集的越狱提示模板数据库，以识别和分解嵌入用户提示中的有害查询。这一过程包括将越狱提示的分解集成到用户的原始查询中，并将其整合为一个一次性学习示例，以有效地教会LLM识别和分离恶意组件。因此，LLM配备了首先中和任何潜在有害元素，然后以符合其道德准则的方式处理用户的提示。RePD是通用的，并与各种作为代理的开源LLM兼容。通过对有害提示和良性提示的全面实验，我们已经证明了我们提出的RePD在增强LLM对越狱攻击的弹性方面的有效性，而不会影响它们响应典型用户请求的性能。



## **32. ART: Automatic Red-teaming for Text-to-Image Models to Protect Benign Users**

ART：文本到图像模型的自动红色团队以保护良性用户 cs.CR

Accepted by NeurIPS 2024

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2405.19360v3) [paper-pdf](http://arxiv.org/pdf/2405.19360v3)

**Authors**: Guanlin Li, Kangjie Chen, Shudong Zhang, Jie Zhang, Tianwei Zhang

**Abstract**: Large-scale pre-trained generative models are taking the world by storm, due to their abilities in generating creative content. Meanwhile, safeguards for these generative models are developed, to protect users' rights and safety, most of which are designed for large language models. Existing methods primarily focus on jailbreak and adversarial attacks, which mainly evaluate the model's safety under malicious prompts. Recent work found that manually crafted safe prompts can unintentionally trigger unsafe generations. To further systematically evaluate the safety risks of text-to-image models, we propose a novel Automatic Red-Teaming framework, ART. Our method leverages both vision language model and large language model to establish a connection between unsafe generations and their prompts, thereby more efficiently identifying the model's vulnerabilities. With our comprehensive experiments, we reveal the toxicity of the popular open-source text-to-image models. The experiments also validate the effectiveness, adaptability, and great diversity of ART. Additionally, we introduce three large-scale red-teaming datasets for studying the safety risks associated with text-to-image models. Datasets and models can be found in https://github.com/GuanlinLee/ART.

摘要: 由于具有创造内容的能力，大规模的预先训练的生成性模型正在席卷世界。同时，为了保护用户的权利和安全，制定了对这些生成模型的保障措施，其中大部分是为大型语言模型设计的。现有的方法主要针对越狱和对抗性攻击，主要是在恶意提示下对模型的安全性进行评估。最近的研究发现，手动创建的安全提示可能会无意中引发不安全的世代。为了进一步系统地评估文本到图像模型的安全风险，我们提出了一个新的自动红色团队框架ART。我们的方法利用视觉语言模型和大型语言模型来建立不安全生成及其提示之间的联系，从而更有效地识别模型的漏洞。通过我们的综合实验，我们揭示了流行的开源文本到图像模型的毒性。实验也验证了ART的有效性、适应性和多样性。此外，我们还介绍了三个大型红团队数据集，用于研究与文本到图像模型相关的安全风险。数据集和模型可在https://github.com/GuanlinLee/ART.中找到



## **33. Cross-modality Information Check for Detecting Jailbreaking in Multimodal Large Language Models**

多模式大型语言模型中检测越狱的跨模式信息检查 cs.CL

12 pages, 9 figures

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2407.21659v3) [paper-pdf](http://arxiv.org/pdf/2407.21659v3)

**Authors**: Yue Xu, Xiuyuan Qi, Zhan Qin, Wenjie Wang

**Abstract**: Multimodal Large Language Models (MLLMs) extend the capacity of LLMs to understand multimodal information comprehensively, achieving remarkable performance in many vision-centric tasks. Despite that, recent studies have shown that these models are susceptible to jailbreak attacks, which refer to an exploitative technique where malicious users can break the safety alignment of the target model and generate misleading and harmful answers. This potential threat is caused by both the inherent vulnerabilities of LLM and the larger attack scope introduced by vision input. To enhance the security of MLLMs against jailbreak attacks, researchers have developed various defense techniques. However, these methods either require modifications to the model's internal structure or demand significant computational resources during the inference phase. Multimodal information is a double-edged sword. While it increases the risk of attacks, it also provides additional data that can enhance safeguards. Inspired by this, we propose Cross-modality Information DEtectoR (CIDER), a plug-and-play jailbreaking detector designed to identify maliciously perturbed image inputs, utilizing the cross-modal similarity between harmful queries and adversarial images. CIDER is independent of the target MLLMs and requires less computation cost. Extensive experimental results demonstrate the effectiveness and efficiency of CIDER, as well as its transferability to both white-box and black-box MLLMs.

摘要: 多通道大语言模型扩展了多通道大语言模型对多通道信息的理解能力，在许多以视觉为中心的任务中取得了显著的性能。尽管如此，最近的研究表明，这些模型容易受到越狱攻击，越狱攻击指的是一种利用技术，恶意用户可以破坏目标模型的安全对齐，并生成误导性和有害的答案。这种潜在的威胁既是由LLM固有的漏洞造成的，也是由视觉输入引入的更大的攻击范围造成的。为了提高MLMS抵御越狱攻击的安全性，研究人员开发了各种防御技术。然而，这些方法要么需要修改模型的内部结构，要么在推理阶段需要大量的计算资源。多式联运信息是一把双刃剑。虽然它增加了攻击的风险，但它也提供了额外的数据，可以加强安全措施。受此启发，我们提出了跨模式信息检测器(Cider)，这是一种即插即用的越狱检测器，旨在利用有害查询和敌意图像之间的跨模式相似性来识别恶意扰动的图像输入。苹果酒不依赖于目标MLLM，并且需要较少的计算成本。大量的实验结果证明了苹果酒的有效性和效率，以及它对白盒和黑盒MLLMS的可转换性。



## **34. The Last Iterate Advantage: Empirical Auditing and Principled Heuristic Analysis of Differentially Private SGD**

最后的迭代优势：差异化私人新元的经验审计和原则性启发式分析 cs.CR

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.06186v2) [paper-pdf](http://arxiv.org/pdf/2410.06186v2)

**Authors**: Thomas Steinke, Milad Nasr, Arun Ganesh, Borja Balle, Christopher A. Choquette-Choo, Matthew Jagielski, Jamie Hayes, Abhradeep Guha Thakurta, Adam Smith, Andreas Terzis

**Abstract**: We propose a simple heuristic privacy analysis of noisy clipped stochastic gradient descent (DP-SGD) in the setting where only the last iterate is released and the intermediate iterates remain hidden. Namely, our heuristic assumes a linear structure for the model.   We show experimentally that our heuristic is predictive of the outcome of privacy auditing applied to various training procedures. Thus it can be used prior to training as a rough estimate of the final privacy leakage. We also probe the limitations of our heuristic by providing some artificial counterexamples where it underestimates the privacy leakage.   The standard composition-based privacy analysis of DP-SGD effectively assumes that the adversary has access to all intermediate iterates, which is often unrealistic. However, this analysis remains the state of the art in practice. While our heuristic does not replace a rigorous privacy analysis, it illustrates the large gap between the best theoretical upper bounds and the privacy auditing lower bounds and sets a target for further work to improve the theoretical privacy analyses. We also empirically support our heuristic and show existing privacy auditing attacks are bounded by our heuristic analysis in both vision and language tasks.

摘要: 在只释放最后一次迭代而隐藏中间迭代的情况下，提出了一种简单的启发式噪声截断随机梯度下降(DP-SGD)隐私分析方法。也就是说，我们的启发式假设模型是线性结构。我们的实验表明，我们的启发式方法可以预测隐私审计应用于各种训练过程的结果。因此，它可以在培训前用作最终隐私泄露的粗略估计。我们还通过提供一些低估隐私泄露的人工反例来探讨我们的启发式算法的局限性。标准的基于组合的DP-SGD隐私分析有效地假设攻击者可以访问所有中间迭代，这通常是不现实的。然而，这种分析在实践中仍然是最先进的。虽然我们的启发式方法没有取代严格的隐私分析，但它说明了最佳理论上限和隐私审计下限之间的巨大差距，并为进一步改进理论隐私分析设定了目标。我们还实证地支持我们的启发式攻击，并表明现有的隐私审计攻击受到我们在视觉和语言任务中的启发式分析的约束。



## **35. APOLLO: A GPT-based tool to detect phishing emails and generate explanations that warn users**

APOLLO：一个基于GPT的工具，用于检测网络钓鱼电子邮件并生成警告用户的解释 cs.HC

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.07997v1) [paper-pdf](http://arxiv.org/pdf/2410.07997v1)

**Authors**: Giuseppe Desolda, Francesco Greco, Luca Viganò

**Abstract**: Phishing is one of the most prolific cybercriminal activities, with attacks becoming increasingly sophisticated. It is, therefore, imperative to explore novel technologies to improve user protection across both technical and human dimensions. Large Language Models (LLMs) offer significant promise for text processing in various domains, but their use for defense against phishing attacks still remains scarcely explored. In this paper, we present APOLLO, a tool based on OpenAI's GPT-4o to detect phishing emails and generate explanation messages to users about why a specific email is dangerous, thus improving their decision-making capabilities. We have evaluated the performance of APOLLO in classifying phishing emails; the results show that the LLM models have exemplary capabilities in classifying phishing emails (97 percent accuracy in the case of GPT-4o) and that this performance can be further improved by integrating data from third-party services, resulting in a near-perfect classification rate (99 percent accuracy). To assess the perception of the explanations generated by this tool, we also conducted a study with 20 participants, comparing four different explanations presented as phishing warnings. We compared the LLM-generated explanations to four baselines: a manually crafted warning, and warnings from Chrome, Firefox, and Edge browsers. The results show that not only the LLM-generated explanations were perceived as high quality, but also that they can be more understandable, interesting, and trustworthy than the baselines. These findings suggest that using LLMs as a defense against phishing is a very promising approach, with APOLLO representing a proof of concept in this research direction.

摘要: 网络钓鱼是最频繁的网络犯罪活动之一，攻击变得越来越复杂。因此，必须探索新技术，从技术和人力两个层面改善对用户的保护。大型语言模型(LLM)为各个领域的文本处理提供了巨大的希望，但它们用于防御网络钓鱼攻击的研究仍然很少。在本文中，我们提出了一个基于OpenAI的GPT-4o的工具Apollo，它可以检测钓鱼电子邮件，并向用户生成解释消息，说明为什么特定的电子邮件是危险的，从而提高他们的决策能力。我们评估了Apollo在分类钓鱼电子邮件方面的性能；结果表明，LLM模型在分类钓鱼电子邮件方面具有典范的能力(在GPT-40的情况下准确率为97%)，并且通过整合来自第三方服务的数据可以进一步提高这一性能，从而产生近乎完美的分类率(99%的准确率)。为了评估人们对该工具产生的解释的看法，我们还对20名参与者进行了一项研究，比较了四种不同的解释作为网络钓鱼警告。我们将LLM生成的解释与四个基线进行了比较：手动创建的警告，以及来自Chrome、Firefox和Edge浏览器的警告。结果表明，LLM生成的解释不仅被认为是高质量的，而且比基线更容易理解、更有趣、更可信。这些发现表明，使用LLMS作为对网络钓鱼的防御是一种非常有前途的方法，阿波罗代表了这一研究方向的概念证明。



## **36. Towards Assurance of LLM Adversarial Robustness using Ontology-Driven Argumentation**

使用实体驱动论证确保LLM对抗鲁棒性 cs.AI

To be published in xAI 2024, late-breaking track

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.07962v1) [paper-pdf](http://arxiv.org/pdf/2410.07962v1)

**Authors**: Tomas Bueno Momcilovic, Beat Buesser, Giulio Zizzo, Mark Purcell, Dian Balta

**Abstract**: Despite the impressive adaptability of large language models (LLMs), challenges remain in ensuring their security, transparency, and interpretability. Given their susceptibility to adversarial attacks, LLMs need to be defended with an evolving combination of adversarial training and guardrails. However, managing the implicit and heterogeneous knowledge for continuously assuring robustness is difficult. We introduce a novel approach for assurance of the adversarial robustness of LLMs based on formal argumentation. Using ontologies for formalization, we structure state-of-the-art attacks and defenses, facilitating the creation of a human-readable assurance case, and a machine-readable representation. We demonstrate its application with examples in English language and code translation tasks, and provide implications for theory and practice, by targeting engineers, data scientists, users, and auditors.

摘要: 尽管大型语言模型（LLM）具有令人印象深刻的适应性，但在确保其安全性、透明度和可解释性方面仍然存在挑战。鉴于LLM容易受到对抗攻击，需要通过对抗训练和护栏的不断发展的组合来保护它们。然而，管理隐性和异类知识以持续确保稳健性是困难的。我们引入了一种新颖的方法来确保LLM的对抗稳健性，基于正式论证。使用实体进行形式化，我们构建了最先进的攻击和防御，促进了人类可读的保证案例和机器可读的表示。我们通过英语和代码翻译任务中的示例展示了它的应用，并通过针对工程师、数据科学家、用户和审计员为理论和实践提供影响。



## **37. Protecting Your LLMs with Information Bottleneck**

通过信息瓶颈保护您的LLC cs.CL

Accepted by Neural Information Processing Systems (NeurIPS 2024)

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2404.13968v3) [paper-pdf](http://arxiv.org/pdf/2404.13968v3)

**Authors**: Zichuan Liu, Zefan Wang, Linjie Xu, Jinyu Wang, Lei Song, Tianchun Wang, Chunlin Chen, Wei Cheng, Jiang Bian

**Abstract**: The advent of large language models (LLMs) has revolutionized the field of natural language processing, yet they might be attacked to produce harmful content. Despite efforts to ethically align LLMs, these are often fragile and can be circumvented by jailbreaking attacks through optimized or manual adversarial prompts. To address this, we introduce the Information Bottleneck Protector (IBProtector), a defense mechanism grounded in the information bottleneck principle, and we modify the objective to avoid trivial solutions. The IBProtector selectively compresses and perturbs prompts, facilitated by a lightweight and trainable extractor, preserving only essential information for the target LLMs to respond with the expected answer. Moreover, we further consider a situation where the gradient is not visible to be compatible with any LLM. Our empirical evaluations show that IBProtector outperforms current defense methods in mitigating jailbreak attempts, without overly affecting response quality or inference speed. Its effectiveness and adaptability across various attack methods and target LLMs underscore the potential of IBProtector as a novel, transferable defense that bolsters the security of LLMs without requiring modifications to the underlying models.

摘要: 大型语言模型的出现给自然语言处理领域带来了革命性的变化，但它们可能会受到攻击，产生有害的内容。尽管努力在道德上调整LLM，但这些往往是脆弱的，可以通过优化或手动对抗性提示通过越狱攻击来绕过。为了解决这个问题，我们引入了信息瓶颈保护器(IBProtector)，这是一种基于信息瓶颈原理的防御机制，我们修改了目标以避免琐碎的解决方案。IBProtector有选择地压缩和干扰提示，由一个轻量级和可训练的提取程序促进，只保留目标LLMS的基本信息，以响应预期的答案。此外，我们还进一步考虑了梯度不可见的情况，以与任何LLM相容。我们的经验评估表明，在不过度影响响应质量或推理速度的情况下，IBProtector在缓解越狱企图方面优于现有的防御方法。它对各种攻击方法和目标LLM的有效性和适应性突显了IBProtector作为一种新型、可转移的防御系统的潜力，无需修改底层模型即可增强LLM的安全性。



## **38. Universally Optimal Watermarking Schemes for LLMs: from Theory to Practice**

LLM的普遍最优水印方案：从理论到实践 cs.CR

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.02890v2) [paper-pdf](http://arxiv.org/pdf/2410.02890v2)

**Authors**: Haiyun He, Yepeng Liu, Ziqiao Wang, Yongyi Mao, Yuheng Bu

**Abstract**: Large Language Models (LLMs) boosts human efficiency but also poses misuse risks, with watermarking serving as a reliable method to differentiate AI-generated content from human-created text. In this work, we propose a novel theoretical framework for watermarking LLMs. Particularly, we jointly optimize both the watermarking scheme and detector to maximize detection performance, while controlling the worst-case Type-I error and distortion in the watermarked text. Within our framework, we characterize the universally minimum Type-II error, showing a fundamental trade-off between detection performance and distortion. More importantly, we identify the optimal type of detectors and watermarking schemes. Building upon our theoretical analysis, we introduce a practical, model-agnostic and computationally efficient token-level watermarking algorithm that invokes a surrogate model and the Gumbel-max trick. Empirical results on Llama-13B and Mistral-8$\times$7B demonstrate the effectiveness of our method. Furthermore, we also explore how robustness can be integrated into our theoretical framework, which provides a foundation for designing future watermarking systems with improved resilience to adversarial attacks.

摘要: 大语言模型(LLM)提高了人类的效率，但也带来了滥用风险，水印是区分人工智能生成的内容和人类创建的文本的可靠方法。在这项工作中，我们提出了一种新的水印LLMS的理论框架。特别是，我们联合优化了水印方案和检测器以最大化检测性能，同时控制了最坏情况下的I类错误和水印文本中的失真。在我们的框架内，我们描述了普遍最小的第二类错误，显示了检测性能和失真之间的基本权衡。更重要的是，我们确定了检测器和水印方案的最佳类型。在理论分析的基础上，我们介绍了一种实用的、与模型无关的、计算高效的令牌级水印算法，该算法调用了代理模型和Gumbel-Max技巧。对Llama-13B和Mistral-8$乘以$70B的实验结果证明了该方法的有效性。此外，我们还探索了如何将稳健性融入到我们的理论框架中，这为设计未来具有更好的抗攻击能力的水印系统提供了基础。



## **39. Mind Your Questions! Towards Backdoor Attacks on Text-to-Visualization Models**

注意你的问题！对文本到可视化模型的后门攻击 cs.CR

11 pages, 4 figures

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.06782v2) [paper-pdf](http://arxiv.org/pdf/2410.06782v2)

**Authors**: Shuaimin Li, Yuanfeng Song, Xuanang Chen, Anni Peng, Zhuoyue Wan, Chen Jason Zhang, Raymond Chi-Wing Wong

**Abstract**: Text-to-visualization (text-to-vis) models have become valuable tools in the era of big data, enabling users to generate data visualizations and make informed decisions through natural language queries (NLQs). Despite their widespread application, the security vulnerabilities of these models have been largely overlooked. To address this gap, we propose VisPoison, a novel framework designed to identify these vulnerabilities of current text-to-vis models systematically. VisPoison introduces two types of triggers that activate three distinct backdoor attacks, potentially leading to data exposure, misleading visualizations, or denial-of-service (DoS) incidents. The framework features both proactive and passive attack mechanisms: proactive attacks leverage rare-word triggers to access confidential data, while passive attacks, triggered unintentionally by users, exploit a first-word trigger method, causing errors or DoS events in visualizations. Through extensive experiments on both trainable and in-context learning (ICL)-based text-to-vis models, \textit{VisPoison} achieves attack success rates of over 90\%, highlighting the security problem of current text-to-vis models. Additionally, we explore two types of defense mechanisms against these attacks, but the results show that existing countermeasures are insufficient, underscoring the pressing need for more robust security solutions in text-to-vis systems.

摘要: 文本到可视化(Text-to-Vis)模型已成为大数据时代的宝贵工具，使用户能够通过自然语言查询(NLQ)生成数据可视化并做出明智的决策。尽管它们被广泛应用，但这些模型的安全漏洞在很大程度上被忽视了。为了弥补这一差距，我们提出了VisPoison，这是一个新的框架，旨在系统地识别当前文本到可视化模型的这些漏洞。VisPoison引入了两种类型的触发器，它们激活了三种不同的后门攻击，可能会导致数据泄露、误导性可视化或拒绝服务(DoS)事件。该框架同时具有主动和被动攻击机制：主动攻击利用稀有单词触发器访问机密数据，而被动攻击由用户无意触发，利用第一单词触发方法，导致可视化中的错误或DoS事件。通过对可训练文本到可视化模型和基于情景学习(ICL)的文本到可视化模型的大量实验，Texttit{VisPoison}的攻击成功率超过90%，突出了当前文本到可视化模型的安全问题。此外，我们探索了两种类型的防御机制来防御这些攻击，但结果表明现有的对策是不够的，这突显了在文本到可视化系统中迫切需要更健壮的安全解决方案。



## **40. Detecting Training Data of Large Language Models via Expectation Maximization**

通过期望最大化检测大型语言模型的训练数据 cs.CL

14 pages

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.07582v1) [paper-pdf](http://arxiv.org/pdf/2410.07582v1)

**Authors**: Gyuwan Kim, Yang Li, Evangelia Spiliopoulou, Jie Ma, Miguel Ballesteros, William Yang Wang

**Abstract**: The widespread deployment of large language models (LLMs) has led to impressive advancements, yet information about their training data, a critical factor in their performance, remains undisclosed. Membership inference attacks (MIAs) aim to determine whether a specific instance was part of a target model's training data. MIAs can offer insights into LLM outputs and help detect and address concerns such as data contamination and compliance with privacy and copyright standards. However, applying MIAs to LLMs presents unique challenges due to the massive scale of pre-training data and the ambiguous nature of membership. Additionally, creating appropriate benchmarks to evaluate MIA methods is not straightforward, as training and test data distributions are often unknown. In this paper, we introduce EM-MIA, a novel MIA method for LLMs that iteratively refines membership scores and prefix scores via an expectation-maximization algorithm, leveraging the duality that the estimates of these scores can be improved by each other. Membership scores and prefix scores assess how each instance is likely to be a member and discriminative as a prefix, respectively. Our method achieves state-of-the-art results on the WikiMIA dataset. To further evaluate EM-MIA, we present OLMoMIA, a benchmark built from OLMo resources, which allows us to control the difficulty of MIA tasks with varying degrees of overlap between training and test data distributions. We believe that EM-MIA serves as a robust MIA method for LLMs and that OLMoMIA provides a valuable resource for comprehensively evaluating MIA approaches, thereby driving future research in this critical area.

摘要: 大型语言模型(LLM)的广泛应用带来了令人印象深刻的进步，但有关其训练数据的信息仍未披露，这是其性能的关键因素。成员关系推理攻击(MIA)旨在确定特定实例是否为目标模型训练数据的一部分。MIA可以提供对LLM输出的洞察，并帮助检测和解决数据污染以及遵守隐私和版权标准等问题。然而，由于大量的预培训数据和成员身份的模棱两可的性质，将MIA应用于LLMS提出了独特的挑战。此外，创建适当的基准来评估MIA方法并不简单，因为培训和测试数据分布通常是未知的。在本文中，我们介绍了EM-MIA，这是一种新的用于LLMS的MIA方法，它通过期望最大化算法迭代地精化隶属度分数和前缀分数，利用这些分数的估计可以相互提高的对偶性。成员资格分数和前缀分数分别评估每个实例作为成员的可能性和作为前缀的区别性。我们的方法在WikiMIA数据集上获得了最先进的结果。为了进一步评估EM-MIA，我们提出了OLMoMIA，一个基于OLMO资源的基准测试，它允许我们控制训练和测试数据分布之间有不同程度重叠的MIA任务的难度。我们认为，EM-MIA是一种强大的LLMS MIA方法，而OLMoMIA为全面评估MIA方法提供了宝贵的资源，从而推动了这一关键领域的未来研究。



## **41. Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning**

简单性盛行：重新思考LLM忘记学习的负偏好优化 cs.CL

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.07163v1) [paper-pdf](http://arxiv.org/pdf/2410.07163v1)

**Authors**: Chongyu Fan, Jiancheng Liu, Licong Lin, Jinghan Jia, Ruiqi Zhang, Song Mei, Sijia Liu

**Abstract**: In this work, we address the problem of large language model (LLM) unlearning, aiming to remove unwanted data influences and associated model capabilities (e.g., copyrighted data or harmful content generation) while preserving essential model utilities, without the need for retraining from scratch. Despite the growing need for LLM unlearning, a principled optimization framework remains lacking. To this end, we revisit the state-of-the-art approach, negative preference optimization (NPO), and identify the issue of reference model bias, which could undermine NPO's effectiveness, particularly when unlearning forget data of varying difficulty. Given that, we propose a simple yet effective unlearning optimization framework, called SimNPO, showing that 'simplicity' in removing the reliance on a reference model (through the lens of simple preference optimization) benefits unlearning. We also provide deeper insights into SimNPO's advantages, supported by analysis using mixtures of Markov chains. Furthermore, we present extensive experiments validating SimNPO's superiority over existing unlearning baselines in benchmarks like TOFU and MUSE, and robustness against relearning attacks. Codes are available at https://github.com/OPTML-Group/Unlearn-Simple.

摘要: 在这项工作中，我们解决了大型语言模型(LLM)遗忘的问题，旨在消除不必要的数据影响和相关的模型能力(例如，受版权保护的数据或有害内容生成)，同时保留基本的模型实用程序，而不需要从头开始重新培训。尽管对LLM遗忘的需求越来越大，但仍然缺乏一个有原则的优化框架。为此，我们回顾了最先进的方法，负偏好优化(NPO)，并确定了参考模型偏差的问题，这可能会削弱NPO的有效性，特别是当遗忘遗忘数据的不同难度时。鉴于此，我们提出了一个简单而有效的遗忘优化框架，称为SimNPO，表明在消除对参考模型的依赖(通过简单偏好优化的镜头)时的“简单性”有利于遗忘。我们还提供了对SimNPO的优势的更深层次的见解，并通过使用马尔可夫链的混合分析提供了支持。此外，我们提供了大量的实验，验证了SimNPO在豆腐和缪斯等基准测试中相对于现有遗忘基线的优势，以及对重新学习攻击的健壮性。有关代码，请访问https://github.com/OPTML-Group/Unlearn-Simple.



## **42. Universal Vulnerabilities in Large Language Models: Backdoor Attacks for In-context Learning**

大型语言模型中的普遍漏洞：上下文学习的后门攻击 cs.CL

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2401.05949v6) [paper-pdf](http://arxiv.org/pdf/2401.05949v6)

**Authors**: Shuai Zhao, Meihuizi Jia, Luu Anh Tuan, Fengjun Pan, Jinming Wen

**Abstract**: In-context learning, a paradigm bridging the gap between pre-training and fine-tuning, has demonstrated high efficacy in several NLP tasks, especially in few-shot settings. Despite being widely applied, in-context learning is vulnerable to malicious attacks. In this work, we raise security concerns regarding this paradigm. Our studies demonstrate that an attacker can manipulate the behavior of large language models by poisoning the demonstration context, without the need for fine-tuning the model. Specifically, we design a new backdoor attack method, named ICLAttack, to target large language models based on in-context learning. Our method encompasses two types of attacks: poisoning demonstration examples and poisoning demonstration prompts, which can make models behave in alignment with predefined intentions. ICLAttack does not require additional fine-tuning to implant a backdoor, thus preserving the model's generality. Furthermore, the poisoned examples are correctly labeled, enhancing the natural stealth of our attack method. Extensive experimental results across several language models, ranging in size from 1.3B to 180B parameters, demonstrate the effectiveness of our attack method, exemplified by a high average attack success rate of 95.0% across the three datasets on OPT models.

摘要: 情境学习是一种弥合预训练和微调之间差距的范式，在几个NLP任务中表现出了很高的效率，特别是在少数情况下。尽管情景学习被广泛应用，但它很容易受到恶意攻击。在这项工作中，我们提出了对此范式的安全担忧。我们的研究表明，攻击者可以通过毒化演示上下文来操纵大型语言模型的行为，而不需要对模型进行微调。具体地说，我们设计了一种新的后门攻击方法ICLAttack，用于基于上下文学习的大型语言模型。我们的方法包括两种类型的攻击：中毒演示示例和中毒演示提示，这可以使模型的行为与预定义的意图保持一致。ICLAttack不需要额外的微调来植入后门，从而保持了模型的通用性。此外，有毒的例子被正确地标记，增强了我们攻击方法的自然隐蔽性。在几个语言模型上的广泛实验结果，从1.3B到180B参数不等，证明了我们的攻击方法的有效性，例如在OPT模型上的三个数据集上的高平均攻击成功率为95.0%。



## **43. Prompt Infection: LLM-to-LLM Prompt Injection within Multi-Agent Systems**

提示感染：多代理系统内LLM到LLM提示注射 cs.MA

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.07283v1) [paper-pdf](http://arxiv.org/pdf/2410.07283v1)

**Authors**: Donghyun Lee, Mo Tiwari

**Abstract**: As Large Language Models (LLMs) grow increasingly powerful, multi-agent systems are becoming more prevalent in modern AI applications. Most safety research, however, has focused on vulnerabilities in single-agent LLMs. These include prompt injection attacks, where malicious prompts embedded in external content trick the LLM into executing unintended or harmful actions, compromising the victim's application. In this paper, we reveal a more dangerous vector: LLM-to-LLM prompt injection within multi-agent systems. We introduce Prompt Infection, a novel attack where malicious prompts self-replicate across interconnected agents, behaving much like a computer virus. This attack poses severe threats, including data theft, scams, misinformation, and system-wide disruption, all while propagating silently through the system. Our extensive experiments demonstrate that multi-agent systems are highly susceptible, even when agents do not publicly share all communications. To address this, we propose LLM Tagging, a defense mechanism that, when combined with existing safeguards, significantly mitigates infection spread. This work underscores the urgent need for advanced security measures as multi-agent LLM systems become more widely adopted.

摘要: 随着大型语言模型(LLM)变得越来越强大，多智能体系统在现代人工智能应用中变得更加普遍。然而，大多数安全研究都集中在单代理LLM的漏洞上。这些攻击包括提示注入攻击，即嵌入到外部内容中的恶意提示欺骗LLM执行意外或有害的操作，从而损害受害者的应用程序。在本文中，我们揭示了一个更危险的载体：多智能体系统中的LLM到LLM快速注射。我们引入了即时感染，这是一种新型的攻击，其中恶意提示在相互连接的代理之间自我复制，行为很像计算机病毒。这种攻击构成了严重的威胁，包括数据盗窃、诈骗、错误信息和系统范围的中断，所有这些都是在系统中静默传播的。我们的大量实验表明，即使在代理不公开共享所有通信的情况下，多代理系统也是高度敏感的。为了解决这个问题，我们提出了LLM标签，这是一种防御机制，当与现有的保护措施相结合时，显著减少了感染传播。这项工作强调了随着多代理LLM系统越来越广泛地被采用，对先进安全措施的迫切需要。



## **44. Break the Visual Perception: Adversarial Attacks Targeting Encoded Visual Tokens of Large Vision-Language Models**

打破视觉感知：针对大型视觉语言模型的编码视觉标记的对抗攻击 cs.CV

Accepted to ACMMM 2024

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06699v1) [paper-pdf](http://arxiv.org/pdf/2410.06699v1)

**Authors**: Yubo Wang, Chaohu Liu, Yanqiu Qu, Haoyu Cao, Deqiang Jiang, Linli Xu

**Abstract**: Large vision-language models (LVLMs) integrate visual information into large language models, showcasing remarkable multi-modal conversational capabilities. However, the visual modules introduces new challenges in terms of robustness for LVLMs, as attackers can craft adversarial images that are visually clean but may mislead the model to generate incorrect answers. In general, LVLMs rely on vision encoders to transform images into visual tokens, which are crucial for the language models to perceive image contents effectively. Therefore, we are curious about one question: Can LVLMs still generate correct responses when the encoded visual tokens are attacked and disrupting the visual information? To this end, we propose a non-targeted attack method referred to as VT-Attack (Visual Tokens Attack), which constructs adversarial examples from multiple perspectives, with the goal of comprehensively disrupting feature representations and inherent relationships as well as the semantic properties of visual tokens output by image encoders. Using only access to the image encoder in the proposed attack, the generated adversarial examples exhibit transferability across diverse LVLMs utilizing the same image encoder and generality across different tasks. Extensive experiments validate the superior attack performance of the VT-Attack over baseline methods, demonstrating its effectiveness in attacking LVLMs with image encoders, which in turn can provide guidance on the robustness of LVLMs, particularly in terms of the stability of the visual feature space.

摘要: 大型视觉语言模型(LVLM)将视觉信息集成到大型语言模型中，展示了非凡的多模式对话能力。然而，视觉模块在稳健性方面为LVLMS带来了新的挑战，因为攻击者可以手工制作视觉上干净但可能误导模型生成错误答案的对抗性图像。通常，视觉编码依赖于视觉编码器将图像转换为视觉标记，这对于语言模型有效地感知图像内容是至关重要的。因此，我们好奇一个问题：当编码的视觉令牌受到攻击并扰乱视觉信息时，LVLMS还能产生正确的反应吗？为此，我们提出了一种非目标攻击方法，称为VT-Attack(视觉标记攻击)，它从多个角度构造对抗性实例，目的是综合破坏图像编码者输出的视觉标记的特征表示和内在关系以及语义属性。在所提出的攻击中，仅使用对图像编码器的访问，生成的敌意示例表现出在使用相同图像编码器的不同LVLM之间的可转移性和跨不同任务的通用性。大量的实验验证了VT攻击相对于基线方法的优越攻击性能，证明了其在利用图像编码器攻击LVLM方面的有效性，进而可以为LVLMS的稳健性，特别是视觉特征空间的稳定性提供指导。



## **45. FELLAS: Enhancing Federated Sequential Recommendation with LLM as External Services**

FELLAS：以LLM作为外部服务增强联合顺序推荐 cs.IR

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.04927v2) [paper-pdf](http://arxiv.org/pdf/2410.04927v2)

**Authors**: Wei Yuan, Chaoqun Yang, Guanhua Ye, Tong Chen, Quoc Viet Hung Nguyen, Hongzhi Yin

**Abstract**: Federated sequential recommendation (FedSeqRec) has gained growing attention due to its ability to protect user privacy. Unfortunately, the performance of FedSeqRec is still unsatisfactory because the models used in FedSeqRec have to be lightweight to accommodate communication bandwidth and clients' on-device computational resource constraints. Recently, large language models (LLMs) have exhibited strong transferable and generalized language understanding abilities and therefore, in the NLP area, many downstream tasks now utilize LLMs as a service to achieve superior performance without constructing complex models. Inspired by this successful practice, we propose a generic FedSeqRec framework, FELLAS, which aims to enhance FedSeqRec by utilizing LLMs as an external service. Specifically, FELLAS employs an LLM server to provide both item-level and sequence-level representation assistance. The item-level representation service is queried by the central server to enrich the original ID-based item embedding with textual information, while the sequence-level representation service is accessed by each client. However, invoking the sequence-level representation service requires clients to send sequences to the external LLM server. To safeguard privacy, we implement dx-privacy satisfied sequence perturbation, which protects clients' sensitive data with guarantees. Additionally, a contrastive learning-based method is designed to transfer knowledge from the noisy sequence representation to clients' sequential recommendation models. Furthermore, to empirically validate the privacy protection capability of FELLAS, we propose two interacted item inference attacks. Extensive experiments conducted on three datasets with two widely used sequential recommendation models demonstrate the effectiveness and privacy-preserving capability of FELLAS.

摘要: 联邦顺序推荐(FedSeqRec)由于其保护用户隐私的能力而受到越来越多的关注。遗憾的是，FedSeqRec的性能仍然不能令人满意，因为FedSeqRec中使用的模型必须是轻量级的，以适应通信带宽和客户端在设备上的计算资源限制。近年来，大语言模型表现出很强的可迁移和泛化语言理解能力，因此，在自然语言处理领域，许多下游任务现在将大语言模型作为一种服务来获得优越的性能，而不需要构建复杂的模型。受这一成功实践的启发，我们提出了一个通用的FedSeqRec框架Fellas，旨在通过利用LLMS作为外部服务来增强FedSeqRec。具体地说，FELLAS使用LLM服务器来提供物品级和序列级的表示帮助。项级表示服务由中央服务器查询，以丰富嵌入文本信息的原始基于ID的项，而序列级表示服务由每个客户端访问。但是，调用序列级别表示服务需要客户端将序列发送到外部LLM服务器。为了保护隐私，我们实现了满足DX隐私的序列扰动，用保证来保护客户的敏感数据。此外，设计了一种基于对比学习的方法来将知识从噪声序列表示转移到客户的序贯推荐模型。此外，为了经验性地验证Fellas的隐私保护能力，我们提出了两种交互的项目推理攻击。在三个数据集和两个广泛使用的序列推荐模型上进行了大量的实验，证明了Fellas的有效性和隐私保护能力。



## **46. FreqMark: Frequency-Based Watermark for Sentence-Level Detection of LLM-Generated Text**

FreqMark：基于频率的水印，用于LLM生成的文本的句子级检测 cs.CL

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.10876v1) [paper-pdf](http://arxiv.org/pdf/2410.10876v1)

**Authors**: Zhenyu Xu, Kun Zhang, Victor S. Sheng

**Abstract**: The increasing use of Large Language Models (LLMs) for generating highly coherent and contextually relevant text introduces new risks, including misuse for unethical purposes such as disinformation or academic dishonesty. To address these challenges, we propose FreqMark, a novel watermarking technique that embeds detectable frequency-based watermarks in LLM-generated text during the token sampling process. The method leverages periodic signals to guide token selection, creating a watermark that can be detected with Short-Time Fourier Transform (STFT) analysis. This approach enables accurate identification of LLM-generated content, even in mixed-text scenarios with both human-authored and LLM-generated segments. Our experiments demonstrate the robustness and precision of FreqMark, showing strong detection capabilities against various attack scenarios such as paraphrasing and token substitution. Results show that FreqMark achieves an AUC improvement of up to 0.98, significantly outperforming existing detection methods.

摘要: 越来越多地使用大型语言模型(LLM)来生成高度连贯和上下文相关的文本，这带来了新的风险，包括出于虚假信息或学术欺诈等不道德目的的滥用。为了应对这些挑战，我们提出了一种新的水印技术FreqMark，它在令牌采样过程中将可检测的基于频率的水印嵌入到LLM生成的文本中。该方法利用周期信号来指导令牌选择，创建可以用短时傅立叶变换(STFT)分析检测的水印。这种方法能够准确识别LLM生成的内容，即使在包含人工创作和LLM生成的片段的混合文本场景中也是如此。我们的实验证明了FreqMark的健壮性和精确度，对释义和标记替换等各种攻击场景具有很强的检测能力。结果表明，FreqMark的AUC提高了0.98，明显优于现有的检测方法。



## **47. Signal Watermark on Large Language Models**

大型语言模型上的信号水印 cs.CR

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06545v1) [paper-pdf](http://arxiv.org/pdf/2410.06545v1)

**Authors**: Zhenyu Xu, Victor S. Sheng

**Abstract**: As Large Language Models (LLMs) become increasingly sophisticated, they raise significant security concerns, including the creation of fake news and academic misuse. Most detectors for identifying model-generated text are limited by their reliance on variance in perplexity and burstiness, and they require substantial computational resources. In this paper, we proposed a watermarking method embedding a specific watermark into the text during its generation by LLMs, based on a pre-defined signal pattern. This technique not only ensures the watermark's invisibility to humans but also maintains the quality and grammatical integrity of model-generated text. We utilize LLMs and Fast Fourier Transform (FFT) for token probability computation and detection of the signal watermark. The unique application of signal processing principles within the realm of text generation by LLMs allows for subtle yet effective embedding of watermarks, which do not compromise the quality or coherence of the generated text. Our method has been empirically validated across multiple LLMs, consistently maintaining high detection accuracy, even with variations in temperature settings during text generation. In the experiment of distinguishing between human-written and watermarked text, our method achieved an AUROC score of 0.97, significantly outperforming existing methods like GPTZero, which scored 0.64. The watermark's resilience to various attacking scenarios further confirms its robustness, addressing significant challenges in model-generated text authentication.

摘要: 随着大型语言模型(LLM)变得越来越复杂，它们引发了重大的安全问题，包括制造假新闻和学术滥用。大多数用于识别模型生成的文本的检测器都受到其对困惑和突发性变化的依赖的限制，并且它们需要大量的计算资源。本文提出了一种基于预定义的信号模式，在LLMS生成文本的过程中嵌入特定水印的水印方法。该技术不仅保证了水印对人类的不可见性，而且保持了模型生成文本的质量和语法完整性。我们利用LLMS和快速傅立叶变换(FFT)进行令牌概率计算和信号水印检测。LLMS在文本生成领域独特地应用信号处理原理，允许微妙而有效地嵌入水印，这不会损害生成的文本的质量或连贯性。我们的方法已经在多个LLM上进行了经验验证，即使在文本生成期间温度设置变化的情况下，也始终保持高检测精度。在区分人写文本和带水印文本的实验中，我们的方法达到了0.97的AUROC分数，远远超过了GPTZero等现有的方法，GPTZero的分数为0.64。水印对各种攻击场景的弹性进一步证实了它的健壮性，解决了模型生成的文本身份验证中的重大挑战。



## **48. WAPITI: A Watermark for Finetuned Open-Source LLMs**

WAPITI：Finetuned开源LLM的水印 cs.CR

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06467v1) [paper-pdf](http://arxiv.org/pdf/2410.06467v1)

**Authors**: Lingjie Chen, Ruizhong Qiu, Siyu Yuan, Zhining Liu, Tianxin Wei, Hyunsik Yoo, Zhichen Zeng, Deqing Yang, Hanghang Tong

**Abstract**: Watermarking of large language models (LLMs) generation embeds an imperceptible statistical pattern within texts, making it algorithmically detectable. Watermarking is a promising method for addressing potential harm and biases from LLMs, as it enables traceability, accountability, and detection of manipulated content, helping to mitigate unintended consequences. However, for open-source models, watermarking faces two major challenges: (i) incompatibility with fine-tuned models, and (ii) vulnerability to fine-tuning attacks. In this work, we propose WAPITI, a new method that transfers watermarking from base models to fine-tuned models through parameter integration. To the best of our knowledge, we propose the first watermark for fine-tuned open-source LLMs that preserves their fine-tuned capabilities. Furthermore, our approach offers an effective defense against fine-tuning attacks. We test our method on various model architectures and watermarking strategies. Results demonstrate that our method can successfully inject watermarks and is highly compatible with fine-tuned models. Additionally, we offer an in-depth analysis of how parameter editing influences the watermark strength and overall capabilities of the resulting models.

摘要: 大语言模型(LLMS)水印生成在文本中嵌入了一种不可察觉的统计模式，使其在算法上是可检测的。水印是一种很有前途的方法，可以解决LLMS的潜在危害和偏见，因为它能够跟踪、问责和检测被篡改的内容，有助于减轻意外后果。然而，对于开源模型，水印面临着两大挑战：(I)与微调模型不兼容，(Ii)易受微调攻击。在这项工作中，我们提出了Wapiti，一种新的方法，通过参数积分将水印从基本模型转移到微调模型。就我们所知，我们建议为保持其微调能力的开放源码LLM提供第一个水印。此外，我们的方法提供了针对微调攻击的有效防御。我们在不同的模型架构和水印策略上测试了我们的方法。实验结果表明，该方法能够成功地嵌入水印，并且与微调模型具有很好的兼容性。此外，我们还深入分析了参数编辑如何影响最终模型的水印强度和整体性能。



## **49. Hallucinating AI Hijacking Attack: Large Language Models and Malicious Code Recommenders**

幻觉人工智能劫持攻击：大型语言模型和恶意代码推荐 cs.CR

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06462v1) [paper-pdf](http://arxiv.org/pdf/2410.06462v1)

**Authors**: David Noever, Forrest McKee

**Abstract**: The research builds and evaluates the adversarial potential to introduce copied code or hallucinated AI recommendations for malicious code in popular code repositories. While foundational large language models (LLMs) from OpenAI, Google, and Anthropic guard against both harmful behaviors and toxic strings, previous work on math solutions that embed harmful prompts demonstrate that the guardrails may differ between expert contexts. These loopholes would appear in mixture of expert's models when the context of the question changes and may offer fewer malicious training examples to filter toxic comments or recommended offensive actions. The present work demonstrates that foundational models may refuse to propose destructive actions correctly when prompted overtly but may unfortunately drop their guard when presented with a sudden change of context, like solving a computer programming challenge. We show empirical examples with trojan-hosting repositories like GitHub, NPM, NuGet, and popular content delivery networks (CDN) like jsDelivr which amplify the attack surface. In the LLM's directives to be helpful, example recommendations propose application programming interface (API) endpoints which a determined domain-squatter could acquire and setup attack mobile infrastructure that triggers from the naively copied code. We compare this attack to previous work on context-shifting and contrast the attack surface as a novel version of "living off the land" attacks in the malware literature. In the latter case, foundational language models can hijack otherwise innocent user prompts to recommend actions that violate their owners' safety policies when posed directly without the accompanying coding support request.

摘要: 这项研究构建并评估了在流行的代码库中引入复制代码或幻觉AI建议的恶意代码的敌意潜力。虽然OpenAI、谷歌和人类的基础大型语言模型(LLM)可以防范有害行为和有毒字符串，但之前关于嵌入有害提示的数学解决方案的工作表明，护栏可能会因专家上下文而异。当问题的上下文发生变化时，这些漏洞将出现在专家模型的混合中，并且可能提供较少的恶意训练示例来过滤有毒评论或建议的攻击性操作。目前的工作表明，基础模型可能会在公开提示时拒绝正确地提出破坏性行动，但不幸的是，当环境突然改变时，可能会放松警惕，比如解决计算机编程挑战。我们使用GitHub、NPM、NuGet等木马托管库和jsDelivr等流行的内容交付网络(CDN)展示了放大攻击面的经验示例。在LLM的有用指令中，示例建议提出了应用程序编程接口(API)端点，确定的域抢占者可以获取这些端点，并建立从简单复制的代码触发的攻击移动基础设施。我们将这一攻击与之前关于上下文转换的工作进行了比较，并将攻击面作为恶意软件文献中的一个新版本的“赖以生存”的攻击进行了对比。在后一种情况下，基础语言模型可以劫持其他无辜的用户提示，在没有附带的编码支持请求的情况下直接提出违反其所有者安全政策的行为。



## **50. Recent advancements in LLM Red-Teaming: Techniques, Defenses, and Ethical Considerations**

LLM红色团队的最新进展：技术、辩护和道德考虑 cs.CL

16 pages, 2 figures

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.09097v1) [paper-pdf](http://arxiv.org/pdf/2410.09097v1)

**Authors**: Tarun Raheja, Nilay Pochhi

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language processing tasks, but their vulnerability to jailbreak attacks poses significant security risks. This survey paper presents a comprehensive analysis of recent advancements in attack strategies and defense mechanisms within the field of Large Language Model (LLM) red-teaming. We analyze various attack methods, including gradient-based optimization, reinforcement learning, and prompt engineering approaches. We discuss the implications of these attacks on LLM safety and the need for improved defense mechanisms. This work aims to provide a thorough understanding of the current landscape of red-teaming attacks and defenses on LLMs, enabling the development of more secure and reliable language models.

摘要: 大型语言模型（LLM）在自然语言处理任务中表现出了非凡的能力，但它们对越狱攻击的脆弱性带来了巨大的安全风险。这篇调查论文全面分析了大型语言模型（LLM）红色团队领域攻击策略和防御机制的最新进展。我们分析了各种攻击方法，包括基于梯度的优化、强化学习和提示工程方法。我们讨论了这些攻击对LLM安全性的影响以及改进防御机制的必要性。这项工作旨在彻底了解LLC上红色团队攻击和防御的当前情况，从而开发更安全、更可靠的语言模型。



