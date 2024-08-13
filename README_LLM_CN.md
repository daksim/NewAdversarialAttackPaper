# Latest Large Language Model Attack Papers
**update at 2024-08-13 18:57:01**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. A RAG-Based Question-Answering Solution for Cyber-Attack Investigation and Attribution**

一种基于RAG的网络攻击调查和归因网络响应解决方案 cs.CR

Accepted at SECAI 2024 (ESORICS 2024)

**SubmitDate**: 2024-08-12    [abs](http://arxiv.org/abs/2408.06272v1) [paper-pdf](http://arxiv.org/pdf/2408.06272v1)

**Authors**: Sampath Rajapaksha, Ruby Rani, Erisa Karafili

**Abstract**: In the constantly evolving field of cybersecurity, it is imperative for analysts to stay abreast of the latest attack trends and pertinent information that aids in the investigation and attribution of cyber-attacks. In this work, we introduce the first question-answering (QA) model and its application that provides information to the cybersecurity experts about cyber-attacks investigations and attribution. Our QA model is based on Retrieval Augmented Generation (RAG) techniques together with a Large Language Model (LLM) and provides answers to the users' queries based on either our knowledge base (KB) that contains curated information about cyber-attacks investigations and attribution or on outside resources provided by the users. We have tested and evaluated our QA model with various types of questions, including KB-based, metadata-based, specific documents from the KB, and external sources-based questions. We compared the answers for KB-based questions with those from OpenAI's GPT-3.5 and the latest GPT-4o LLMs. Our proposed QA model outperforms OpenAI's GPT models by providing the source of the answers and overcoming the hallucination limitations of the GPT models, which is critical for cyber-attack investigation and attribution. Additionally, our analysis showed that when the RAG QA model is given few-shot examples rather than zero-shot instructions, it generates better answers compared to cases where no examples are supplied in addition to the query.

摘要: 在不断发展的网络安全领域，分析人员必须了解最新的攻击趋势和相关信息，以帮助调查和确定网络攻击的归属。在这项工作中，我们介绍了第一个问答(QA)模型及其应用，该模型向网络安全专家提供有关网络攻击调查和归因的信息。我们的QA模型基于检索增强生成(RAG)技术和大型语言模型(LLM)，并基于我们的知识库(KB)(包含有关网络攻击调查和归属的精选信息)或用户提供的外部资源来回答用户的问题。我们已经使用各种类型的问题测试和评估了我们的QA模型，包括基于知识库的、基于元数据的、来自知识库的特定文档以及基于外部来源的问题。我们将基于知识库的问题的答案与OpenAI的GPT-3.5和最新的GPT-40 LLM的答案进行了比较。我们提出的QA模型在提供答案来源和克服GPT模型的幻觉限制方面优于OpenAI的GPT模型，而GPT模型对于网络攻击调查和归因至关重要。此外，我们的分析表明，当RAG QA模型给出少量的例子而不是零的指令时，与除了查询之外没有提供例子的情况相比，它生成了更好的答案。



## **2. On Effects of Steering Latent Representation for Large Language Model Unlearning**

论引导潜在表示对大型语言模型取消学习的影响 cs.CL

15 pages, 5 figures, 8 tables

**SubmitDate**: 2024-08-12    [abs](http://arxiv.org/abs/2408.06223v1) [paper-pdf](http://arxiv.org/pdf/2408.06223v1)

**Authors**: Dang Huu-Tien, Trung-Tin Pham, Hoang Thanh-Tung, Naoya Inoue

**Abstract**: Representation Misdirection for Unlearning (RMU), which steers model representation in the intermediate layer to a target random representation, is an effective method for large language model (LLM) unlearning. Despite its high performance, the underlying cause and explanation remain underexplored. In this paper, we first theoretically demonstrate that steering forget representations in the intermediate layer reduces token confidence, causing LLMs to generate wrong or nonsense responses. Second, we investigate how the coefficient influences the alignment of forget-sample representations with the random direction and hint at the optimal coefficient values for effective unlearning across different network layers. Third, we show that RMU unlearned models are robust against adversarial jailbreak attacks. Last, our empirical analysis shows that RMU is less effective when applied to the middle and later layers in LLMs. To resolve this drawback, we propose Adaptive RMU -- a simple yet effective alternative method that makes unlearning effective with most layers. Extensive experiments demonstrate that Adaptive RMU significantly improves the unlearning performance compared to prior art while incurring no additional computational cost.

摘要: 遗忘表征误导(RMU)是一种有效的大语言模型遗忘方法，它将中间层的模型表征引导到目标随机表征。尽管其表现良好，但其根本原因和解释仍未得到充分研究。在本文中，我们首先从理论上证明，中间层中的转向遗忘表征会降低令牌置信度，从而导致LLM生成错误或无意义的响应。其次，我们研究了系数如何影响遗忘样本表示与随机方向的对齐，并提示了跨不同网络层有效遗忘的最优系数值。第三，我们证明了RMU未学习模型对敌意越狱攻击是健壮的。最后，我们的实证分析表明，当RMU应用于LLMS的中后期时，其有效性较差。为了解决这一缺陷，我们提出了自适应RMU--一种简单但有效的替代方法，使遗忘在大多数层都有效。大量实验表明，与现有技术相比，自适应RMU在不增加额外计算代价的情况下，显著改善了遗忘性能。



## **3. Nob-MIAs: Non-biased Membership Inference Attacks Assessment on Large Language Models with Ex-Post Dataset Construction**

Nob-MIA：对具有Ex-Post数据集构建的大型语言模型的无偏见成员推理攻击评估 cs.CR

**SubmitDate**: 2024-08-12    [abs](http://arxiv.org/abs/2408.05968v1) [paper-pdf](http://arxiv.org/pdf/2408.05968v1)

**Authors**: Cédric Eichler, Nathan Champeil, Nicolas Anciaux, Alexandra Bensamoun, Heber Hwang Arcolezi, José Maria De Fuentes

**Abstract**: The rise of Large Language Models (LLMs) has triggered legal and ethical concerns, especially regarding the unauthorized use of copyrighted materials in their training datasets. This has led to lawsuits against tech companies accused of using protected content without permission. Membership Inference Attacks (MIAs) aim to detect whether specific documents were used in a given LLM pretraining, but their effectiveness is undermined by biases such as time-shifts and n-gram overlaps.   This paper addresses the evaluation of MIAs on LLMs with partially inferable training sets, under the ex-post hypothesis, which acknowledges inherent distributional biases between members and non-members datasets. We propose and validate algorithms to create ``non-biased'' and ``non-classifiable'' datasets for fairer MIA assessment. Experiments using the Gutenberg dataset on OpenLamma and Pythia show that neutralizing known biases alone is insufficient. Our methods produce non-biased ex-post datasets with AUC-ROC scores comparable to those previously obtained on genuinely random datasets, validating our approach. Globally, MIAs yield results close to random, with only one being effective on both random and our datasets, but its performance decreases when bias is removed.

摘要: 大型语言模型(LLM)的兴起引发了法律和伦理方面的担忧，特别是关于在其培训数据集中未经授权使用受版权保护的材料。这导致了针对科技公司的诉讼，这些公司被控未经许可使用受保护的内容。成员关系推理攻击(MIA)的目的是检测特定文档是否被用于给定的LLM预训练，但其有效性受到时移和n元语法重叠等偏差的影响。在承认成员和非成员数据集之间固有分布偏差的后验假设下，本文讨论了部分可推断训练集的LLMS上的MIA的评估。我们提出并验证了为更公平的MIA评估创建“无偏见”和“不可分类”数据集的算法。在OpenLamma和Pythia上使用Gutenberg数据集进行的实验表明，仅中和已知的偏见是不够的。我们的方法产生无偏见的事后数据集，其AUC-ROC得分与之前在真正随机数据集上获得的得分相当，验证了我们的方法。在全球范围内，MIA产生的结果接近随机，只有一个对随机和我们的数据集有效，但当消除偏差时，其性能会下降。



## **4. Multimodal Large Language Models for Phishing Webpage Detection and Identification**

用于网络钓鱼网页检测和识别的多模式大语言模型 cs.CR

To appear in eCrime 2024

**SubmitDate**: 2024-08-12    [abs](http://arxiv.org/abs/2408.05941v1) [paper-pdf](http://arxiv.org/pdf/2408.05941v1)

**Authors**: Jehyun Lee, Peiyuan Lim, Bryan Hooi, Dinil Mon Divakaran

**Abstract**: To address the challenging problem of detecting phishing webpages, researchers have developed numerous solutions, in particular those based on machine learning (ML) algorithms. Among these, brand-based phishing detection that uses models from Computer Vision to detect if a given webpage is imitating a well-known brand has received widespread attention. However, such models are costly and difficult to maintain, as they need to be retrained with labeled dataset that has to be regularly and continuously collected. Besides, they also need to maintain a good reference list of well-known websites and related meta-data for effective performance.   In this work, we take steps to study the efficacy of large language models (LLMs), in particular the multimodal LLMs, in detecting phishing webpages. Given that the LLMs are pretrained on a large corpus of data, we aim to make use of their understanding of different aspects of a webpage (logo, theme, favicon, etc.) to identify the brand of a given webpage and compare the identified brand with the domain name in the URL to detect a phishing attack. We propose a two-phase system employing LLMs in both phases: the first phase focuses on brand identification, while the second verifies the domain. We carry out comprehensive evaluations on a newly collected dataset. Our experiments show that the LLM-based system achieves a high detection rate at high precision; importantly, it also provides interpretable evidence for the decisions. Our system also performs significantly better than a state-of-the-art brand-based phishing detection system while demonstrating robustness against two known adversarial attacks.

摘要: 为了解决检测钓鱼网页这一具有挑战性的问题，研究人员开发了许多解决方案，特别是基于机器学习(ML)算法的解决方案。其中，基于品牌的钓鱼检测利用计算机视觉的模型来检测给定的网页是否在模仿知名品牌，受到了广泛的关注。然而，这种模型成本很高，很难维护，因为它们需要用必须定期和连续收集的标记数据集进行再训练。此外，他们还需要维护一个良好的参考名单的知名网站和相关的元数据，以有效的表现。在这项工作中，我们采取步骤研究大语言模型，特别是多模式大语言模型在检测钓鱼网页方面的有效性。鉴于LLM是在大型数据语料库上预先培训的，我们的目标是利用他们对网页的不同方面(徽标、主题、图标等)的理解。识别给定网页的品牌，并将识别的品牌与URL中的域名进行比较，以检测网络钓鱼攻击。我们提出了一个在两个阶段都使用LLMS的两阶段系统：第一阶段专注于品牌识别，第二阶段验证领域。我们对新收集的数据集进行了全面的评估。我们的实验表明，基于LLM的系统在高精度的情况下实现了高检测率，重要的是它还为决策提供了可解释的证据。我们的系统也比最先进的基于品牌的钓鱼检测系统性能要好得多，同时对两种已知的对手攻击表现出了健壮性。



## **5. LLM-Based Robust Product Classification in Commerce and Compliance**

基于LLM的商业和合规稳健产品分类 cs.CL

11 pages

**SubmitDate**: 2024-08-11    [abs](http://arxiv.org/abs/2408.05874v1) [paper-pdf](http://arxiv.org/pdf/2408.05874v1)

**Authors**: Sina Gholamian, Gianfranco Romani, Bartosz Rudnikowicz, Laura Skylaki

**Abstract**: Product classification is a crucial task in international trade, as compliance regulations are verified and taxes and duties are applied based on product categories. Manual classification of products is time-consuming and error-prone, and the sheer volume of products imported and exported renders the manual process infeasible. Consequently, e-commerce platforms and enterprises involved in international trade have turned to automatic product classification using machine learning. However, current approaches do not consider the real-world challenges associated with product classification, such as very abbreviated and incomplete product descriptions. In addition, recent advancements in generative Large Language Models (LLMs) and their reasoning capabilities are mainly untapped in product classification and e-commerce. In this research, we explore the real-life challenges of industrial classification and we propose data perturbations that allow for realistic data simulation. Furthermore, we employ LLM-based product classification to improve the robustness of the prediction in presence of incomplete data. Our research shows that LLMs with in-context learning outperform the supervised approaches in the clean-data scenario. Additionally, we illustrate that LLMs are significantly more robust than the supervised approaches when data attacks are present.

摘要: 产品分类是国际贸易中的一项关键任务，因为要核实合规规定，并根据产品类别适用税收和关税。人工对产品进行分类既耗时又容易出错，而且进出口产品的数量庞大，使手工分类过程变得不可行。因此，参与国际贸易的电子商务平台和企业已经转向使用机器学习的产品自动分类。然而，目前的方法没有考虑到与产品分类相关的现实挑战，例如非常简短和不完整的产品描述。此外，生成性大型语言模型(LLM)及其推理能力的最新进展主要是在产品分类和电子商务方面尚未开发。在这项研究中，我们探索了现实生活中的行业分类挑战，并提出了允许现实数据模拟的数据扰动。此外，我们使用基于LLM的产品分类来提高在存在不完整数据的情况下预测的稳健性。我们的研究表明，在干净数据的情况下，具有情境学习的LLMS的性能优于有监督的方法。此外，我们还说明了当存在数据攻击时，LLMS比监督方法具有更强的健壮性。



## **6. PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models**

PoisonedRAG：对大型语言模型检索增强生成的知识腐败攻击 cs.CR

To appear in USENIX Security Symposium 2025. The code is available at  https://github.com/sleeepeer/PoisonedRAG

**SubmitDate**: 2024-08-11    [abs](http://arxiv.org/abs/2402.07867v2) [paper-pdf](http://arxiv.org/pdf/2402.07867v2)

**Authors**: Wei Zou, Runpeng Geng, Binghui Wang, Jinyuan Jia

**Abstract**: Large language models (LLMs) have achieved remarkable success due to their exceptional generative capabilities. Despite their success, they also have inherent limitations such as a lack of up-to-date knowledge and hallucination. Retrieval-Augmented Generation (RAG) is a state-of-the-art technique to mitigate these limitations. The key idea of RAG is to ground the answer generation of an LLM on external knowledge retrieved from a knowledge database. Existing studies mainly focus on improving the accuracy or efficiency of RAG, leaving its security largely unexplored. We aim to bridge the gap in this work. We find that the knowledge database in a RAG system introduces a new and practical attack surface. Based on this attack surface, we propose PoisonedRAG, the first knowledge corruption attack to RAG, where an attacker could inject a few malicious texts into the knowledge database of a RAG system to induce an LLM to generate an attacker-chosen target answer for an attacker-chosen target question. We formulate knowledge corruption attacks as an optimization problem, whose solution is a set of malicious texts. Depending on the background knowledge (e.g., black-box and white-box settings) of an attacker on a RAG system, we propose two solutions to solve the optimization problem, respectively. Our results show PoisonedRAG could achieve a 90% attack success rate when injecting five malicious texts for each target question into a knowledge database with millions of texts. We also evaluate several defenses and our results show they are insufficient to defend against PoisonedRAG, highlighting the need for new defenses.

摘要: 大型语言模型(LLM)由于其非凡的生成能力而取得了显著的成功。尽管他们取得了成功，但他们也有内在的局限性，比如缺乏最新的知识和幻觉。检索-增强生成(RAG)是一种缓解这些限制的最先进的技术。RAG的核心思想是基于从知识库中检索到的外部知识来生成LLM的答案。现有的研究主要集中于提高RAG的准确性或效率，而其安全性在很大程度上还没有被探索。我们的目标是弥合这项工作中的差距。我们发现，RAG系统中的知识库引入了一个新的、实用的攻击面。基于这个攻击面，我们提出了PoisonedRAG，这是第一个针对RAG的知识腐败攻击，攻击者可以向RAG系统的知识库中注入一些恶意文本，以诱导LLM为攻击者选择的目标问题生成攻击者选择的目标答案。我们将知识腐败攻击描述为一个优化问题，其解决方案是一组恶意文本。根据RAG系统上攻击者的背景知识(例如黑盒和白盒设置)，我们分别提出了两种解决优化问题的方案。实验结果表明，PoisonedRAG在一个包含数百万条文本的知识库中为每个目标问题注入5个恶意文本，攻击成功率可达到90%。我们还评估了几种防御措施，我们的结果表明，它们不足以防御PoisonedRAG，这突显了需要新的防御措施。



## **7. Using Retriever Augmented Large Language Models for Attack Graph Generation**

使用检索器增强大型语言模型生成攻击图 cs.CR

**SubmitDate**: 2024-08-11    [abs](http://arxiv.org/abs/2408.05855v1) [paper-pdf](http://arxiv.org/pdf/2408.05855v1)

**Authors**: Renascence Tarafder Prapty, Ashish Kundu, Arun Iyengar

**Abstract**: As the complexity of modern systems increases, so does the importance of assessing their security posture through effective vulnerability management and threat modeling techniques. One powerful tool in the arsenal of cybersecurity professionals is the attack graph, a representation of all potential attack paths within a system that an adversary might exploit to achieve a certain objective. Traditional methods of generating attack graphs involve expert knowledge, manual curation, and computational algorithms that might not cover the entire threat landscape due to the ever-evolving nature of vulnerabilities and exploits. This paper explores the approach of leveraging large language models (LLMs), such as ChatGPT, to automate the generation of attack graphs by intelligently chaining Common Vulnerabilities and Exposures (CVEs) based on their preconditions and effects. It also shows how to utilize LLMs to create attack graphs from threat reports.

摘要: 随着现代系统复杂性的增加，通过有效的漏洞管理和威胁建模技术评估其安全态势的重要性也随之增加。网络安全专业人员武器库中的一个强大工具是攻击图，它代表了系统内对手可能利用的所有潜在攻击路径来实现特定目标。生成攻击图的传统方法涉及专家知识、手动策划和计算算法，由于漏洞和漏洞利用的不断变化的性质，这些方法可能无法覆盖整个威胁格局。本文探讨了利用ChatGPT等大型语言模型（LLM）来自动生成攻击图的方法，通过根据其先决条件和效果智能链接常见漏洞和暴露（CVE）。它还展示了如何利用LLM根据威胁报告创建攻击图。



## **8. Bot or Human? Detecting ChatGPT Imposters with A Single Question**

机器人还是人类？通过一个问题检测ChatGPT冒名顶替者 cs.CL

**SubmitDate**: 2024-08-11    [abs](http://arxiv.org/abs/2305.06424v4) [paper-pdf](http://arxiv.org/pdf/2305.06424v4)

**Authors**: Hong Wang, Xuan Luo, Weizhi Wang, Xifeng Yan

**Abstract**: Large language models (LLMs) like GPT-4 have recently demonstrated impressive capabilities in natural language understanding and generation. However, there is a concern that they can be misused for malicious purposes, such as fraud or denial-of-service attacks. Therefore, it is crucial to develop methods for detecting whether the party involved in a conversation is a bot or a human. In this paper, we propose a framework named FLAIR, Finding Large Language Model Authenticity via a Single Inquiry and Response, to detect conversational bots in an online manner. Specifically, we target a single question scenario that can effectively differentiate human users from bots. The questions are divided into two categories: those that are easy for humans but difficult for bots (e.g., counting, substitution, searching, and ASCII art reasoning), and those that are easy for bots but difficult for humans (e.g., memorization and computation). Our approach shows different strengths of these questions in their effectiveness, providing a new way for online service providers to protect themselves against nefarious activities. Our code and question set are available at https://github.com/hongwang600/FLAIR.

摘要: 像GPT-4这样的大型语言模型(LLM)最近在自然语言理解和生成方面表现出了令人印象深刻的能力。然而，人们担心它们可能被滥用于恶意目的，如欺诈或拒绝服务攻击。因此，开发方法来检测参与对话的一方是机器人还是人类是至关重要的。在本文中，我们提出了一个名为FLAIR的框架，通过单一查询和响应来发现大型语言模型的真实性，以在线方式检测会话机器人。具体地说，我们的目标是能够有效区分人类用户和机器人的单一问题场景。这些问题被分为两类：一类是对人类容易但对机器人困难的问题(例如，计数、替换、搜索和ASCII艺术推理)；另一类是对机器人容易但对人类困难的问题(例如，记忆和计算)。我们的方法显示了这些问题在有效性上的不同优势，为在线服务提供商提供了一种新的方式来保护自己免受恶意活动的影响。我们的代码和问题集可在https://github.com/hongwang600/FLAIR.上找到



## **9. Unbridled Icarus: A Survey of the Potential Perils of Image Inputs in Multimodal Large Language Model Security**

无拘无束的伊卡洛斯：图像输入在多模式大型语言模型安全中的潜在危险调查 cs.CR

8 pages, 1 figure. Accepted to 2024 IEEE International Conference on  Systems, Man, and Cybernetics

**SubmitDate**: 2024-08-11    [abs](http://arxiv.org/abs/2404.05264v2) [paper-pdf](http://arxiv.org/pdf/2404.05264v2)

**Authors**: Yihe Fan, Yuxin Cao, Ziyu Zhao, Ziyao Liu, Shaofeng Li

**Abstract**: Multimodal Large Language Models (MLLMs) demonstrate remarkable capabilities that increasingly influence various aspects of our daily lives, constantly defining the new boundary of Artificial General Intelligence (AGI). Image modalities, enriched with profound semantic information and a more continuous mathematical nature compared to other modalities, greatly enhance the functionalities of MLLMs when integrated. However, this integration serves as a double-edged sword, providing attackers with expansive vulnerabilities to exploit for highly covert and harmful attacks. The pursuit of reliable AI systems like powerful MLLMs has emerged as a pivotal area of contemporary research. In this paper, we endeavor to demostrate the multifaceted risks associated with the incorporation of image modalities into MLLMs. Initially, we delineate the foundational components and training processes of MLLMs. Subsequently, we construct a threat model, outlining the security vulnerabilities intrinsic to MLLMs. Moreover, we analyze and summarize existing scholarly discourses on MLLMs' attack and defense mechanisms, culminating in suggestions for the future research on MLLM security. Through this comprehensive analysis, we aim to deepen the academic understanding of MLLM security challenges and propel forward the development of trustworthy MLLM systems.

摘要: 多通道大语言模型(MLLMS)显示出非凡的能力，越来越多地影响我们日常生活的各个方面，不断定义人工通用智能(AGI)的新边界。与其他模式相比，图像模式具有更丰富的语义信息和更连续的数学性质，极大地增强了MLLMS的功能。然而，这种集成是一把双刃剑，为攻击者提供了大量漏洞，可以利用这些漏洞进行高度隐蔽和有害的攻击。像强大的MLLMS这样可靠的人工智能系统的追求已经成为当代研究的一个关键领域。在这篇文章中，我们努力展示与将图像模式结合到MLLMS中相关的多方面风险。首先，我们描述了MLLMS的基本组成部分和培训过程。随后，我们构建了威胁模型，概述了MLLMS固有的安全漏洞。此外，我们分析和总结了已有的关于MLLMS攻击和防御机制的学术论述，并对未来的MLLMS安全研究提出了建议。通过这一综合分析，我们旨在加深对MLLM安全挑战的学术认识，推动可信MLLM系统的发展。



## **10. Utilizing Large Language Models to Optimize the Detection and Explainability of Phishing Websites**

利用大型语言模型优化网络钓鱼网站的检测和解释性 cs.CR

**SubmitDate**: 2024-08-11    [abs](http://arxiv.org/abs/2408.05667v1) [paper-pdf](http://arxiv.org/pdf/2408.05667v1)

**Authors**: Sayak Saha Roy, Shirin Nilizadeh

**Abstract**: In this paper, we introduce PhishLang, an open-source, lightweight Large Language Model (LLM) specifically designed for phishing website detection through contextual analysis of the website. Unlike traditional heuristic or machine learning models that rely on static features and struggle to adapt to new threats and deep learning models that are computationally intensive, our model utilizes the advanced language processing capabilities of LLMs to learn granular features that are characteristic of phishing attacks. Furthermore, PhishLang operates with minimal data preprocessing and offers performance comparable to leading deep learning tools, while being significantly faster and less resource-intensive. Over a 3.5-month testing period, PhishLang successfully identified approximately 26K phishing URLs, many of which were undetected by popular antiphishing blocklists, thus demonstrating its potential to aid current detection measures. We also evaluate PhishLang against several realistic adversarial attacks and develop six patches that make it very robust against such threats. Furthermore, we integrate PhishLang with GPT-3.5 Turbo to create \textit{explainable blocklisting} - warnings that provide users with contextual information about different features that led to a website being marked as phishing. Finally, we have open-sourced the PhishLang framework and developed a Chromium-based browser extension and URL scanner website, which implement explainable warnings for end-users.

摘要: 在本文中，我们介绍了PhishLang，一个开源的，轻量级的大型语言模型(LLM)，专门为钓鱼网站检测而设计的，通过对网站的上下文分析。与传统的启发式或机器学习模型依赖静态特征并难以适应计算密集型的新威胁和深度学习模型不同，我们的模型利用LLMS的高级语言处理能力来学习钓鱼攻击的细粒度特征。此外，PhishLang的操作只需最少的数据预处理，并提供可与领先的深度学习工具相媲美的性能，同时速度更快，资源消耗更少。在3.5个月的测试期内，PhishLang成功识别了大约26K个钓鱼URL，其中许多都没有被流行的反钓鱼阻止列表检测到，从而展示了其帮助当前检测措施的潜力。我们还评估了Phishlang对几种现实对手攻击的抵抗力，并开发了六个补丁，使其对此类威胁非常健壮。此外，我们将PhishLang与GPT-3.5 Turbo集成，以创建\Text{可解释的阻止列表}-警告，向用户提供有关导致网站被标记为网络钓鱼的不同功能的上下文信息。最后，我们开源了PhishLang框架，并开发了一个基于Chromium的浏览器扩展和URL扫描器网站，为最终用户实现了可解释的警告。



## **11. Preserving Privacy in Large Language Models: A Survey on Current Threats and Solutions**

在大型语言模型中保护隐私：对当前威胁和解决方案的调查 cs.CR

GitHub repository:  https://github.com/michele17284/Awesome-Privacy-Preserving-LLMs

**SubmitDate**: 2024-08-10    [abs](http://arxiv.org/abs/2408.05212v1) [paper-pdf](http://arxiv.org/pdf/2408.05212v1)

**Authors**: Michele Miranda, Elena Sofia Ruzzetti, Andrea Santilli, Fabio Massimo Zanzotto, Sébastien Bratières, Emanuele Rodolà

**Abstract**: Large Language Models (LLMs) represent a significant advancement in artificial intelligence, finding applications across various domains. However, their reliance on massive internet-sourced datasets for training brings notable privacy issues, which are exacerbated in critical domains (e.g., healthcare). Moreover, certain application-specific scenarios may require fine-tuning these models on private data. This survey critically examines the privacy threats associated with LLMs, emphasizing the potential for these models to memorize and inadvertently reveal sensitive information. We explore current threats by reviewing privacy attacks on LLMs and propose comprehensive solutions for integrating privacy mechanisms throughout the entire learning pipeline. These solutions range from anonymizing training datasets to implementing differential privacy during training or inference and machine unlearning after training. Our comprehensive review of existing literature highlights ongoing challenges, available tools, and future directions for preserving privacy in LLMs. This work aims to guide the development of more secure and trustworthy AI systems by providing a thorough understanding of privacy preservation methods and their effectiveness in mitigating risks.

摘要: 大型语言模型(LLM)代表了人工智能的一项重大进步，可以找到跨各个领域的应用程序。然而，他们对来自互联网的海量数据集的依赖带来了显著的隐私问题，在关键领域(例如医疗保健)，这一问题更加严重。此外，某些特定于应用程序的场景可能需要根据私有数据对这些模型进行微调。这项调查严格审查了与LLMS相关的隐私威胁，强调了这些模型可能会记住并无意中泄露敏感信息。我们通过审查对LLM的隐私攻击来探索当前的威胁，并提出全面的解决方案，将隐私机制整合到整个学习管道中。这些解决方案的范围从匿名训练数据集到在训练期间实现差异隐私或在训练后进行推理和机器遗忘。我们对现有文献的全面回顾突出了在LLMS中保护隐私的持续挑战、可用的工具和未来的方向。这项工作旨在通过提供对隐私保护方法及其在降低风险方面的有效性的透彻了解，来指导更安全和值得信赖的人工智能系统的开发。



## **12. Towards Resilient and Efficient LLMs: A Comparative Study of Efficiency, Performance, and Adversarial Robustness**

迈向弹性和高效的法学硕士：效率、绩效和对抗稳健性的比较研究 cs.CL

**SubmitDate**: 2024-08-09    [abs](http://arxiv.org/abs/2408.04585v2) [paper-pdf](http://arxiv.org/pdf/2408.04585v2)

**Authors**: Xiaojing Fan, Chunliang Tao

**Abstract**: With the increasing demand for practical applications of Large Language Models (LLMs), many attention-efficient models have been developed to balance performance and computational cost. However, the adversarial robustness of these models remains under-explored. In this work, we design a framework to investigate the trade-off between efficiency, performance, and adversarial robustness of LLMs by comparing three prominent models with varying levels of complexity and efficiency -- Transformer++, Gated Linear Attention (GLA) Transformer, and MatMul-Free LM -- utilizing the GLUE and AdvGLUE datasets. The AdvGLUE dataset extends the GLUE dataset with adversarial samples designed to challenge model robustness. Our results show that while the GLA Transformer and MatMul-Free LM achieve slightly lower accuracy on GLUE tasks, they demonstrate higher efficiency and either superior or comparative robustness on AdvGLUE tasks compared to Transformer++ across different attack levels. These findings highlight the potential of simplified architectures to achieve a compelling balance between efficiency, performance, and adversarial robustness, offering valuable insights for applications where resource constraints and resilience to adversarial attacks are critical.

摘要: 随着大型语言模型的实际应用需求的增加，人们已经开发了许多注意力高效的模型来平衡性能和计算成本。然而，这些模型的对抗性稳健性仍然没有得到充分的研究。在这项工作中，我们设计了一个框架来研究LLMS的效率、性能和对抗健壮性之间的权衡，方法是利用GLUE和AdvGLUE数据集比较三种不同复杂度和效率的重要模型--Transformer++、GLA Transformer和Matmul-Free LM。AdvGLUE数据集使用旨在挑战模型稳健性的对抗性样本扩展了GLUE数据集。我们的结果表明，虽然GLA Transformer和MatMul-Free LM在粘合任务上的准确率略低，但在不同攻击级别上，它们在AdvGLUE任务上表现出比Transformer++更高的效率和更好的健壮性或相对较高的稳健性。这些发现突出了简化体系结构在效率、性能和对手攻击健壮性之间实现引人注目的平衡的潜力，为资源约束和对抗攻击的弹性至关重要的应用程序提供了宝贵的见解。



## **13. AttackER: Towards Enhancing Cyber-Attack Attribution with a Named Entity Recognition Dataset**

AttackER：利用命名实体识别数据集增强网络攻击归因 cs.CR

Submitted to WISE 2024

**SubmitDate**: 2024-08-09    [abs](http://arxiv.org/abs/2408.05149v1) [paper-pdf](http://arxiv.org/pdf/2408.05149v1)

**Authors**: Pritam Deka, Sampath Rajapaksha, Ruby Rani, Amirah Almutairi, Erisa Karafili

**Abstract**: Cyber-attack attribution is an important process that allows experts to put in place attacker-oriented countermeasures and legal actions. The analysts mainly perform attribution manually, given the complex nature of this task. AI and, more specifically, Natural Language Processing (NLP) techniques can be leveraged to support cybersecurity analysts during the attribution process. However powerful these techniques are, they need to deal with the lack of datasets in the attack attribution domain. In this work, we will fill this gap and will provide, to the best of our knowledge, the first dataset on cyber-attack attribution. We designed our dataset with the primary goal of extracting attack attribution information from cybersecurity texts, utilizing named entity recognition (NER) methodologies from the field of NLP. Unlike other cybersecurity NER datasets, ours offers a rich set of annotations with contextual details, including some that span phrases and sentences. We conducted extensive experiments and applied NLP techniques to demonstrate the dataset's effectiveness for attack attribution. These experiments highlight the potential of Large Language Models (LLMs) capabilities to improve the NER tasks in cybersecurity datasets for cyber-attack attribution.

摘要: 网络攻击归因是一个重要的过程，使专家能够制定以攻击者为导向的对策和法律行动。考虑到这项任务的复杂性，分析师主要是手动执行归因。人工智能，更具体地说，自然语言处理(NLP)技术可以被用来在归因过程中支持网络安全分析师。无论这些技术多么强大，它们都需要处理攻击属性域中缺乏数据集的问题。在这项工作中，我们将填补这一空白，并将提供我们所知的关于网络攻击归属的第一个数据集。我们设计我们的数据集的主要目标是从网络安全文本中提取攻击属性信息，利用NLP领域的命名实体识别(NER)方法。与其他网络安全NER数据集不同，我们的数据集提供了丰富的具有上下文细节的注释集，包括一些跨越短语和句子的注释。我们进行了大量的实验，并应用NLP技术来验证该数据集对攻击属性的有效性。这些实验突出了大型语言模型(LLM)能力的潜力，以改进网络安全数据集中的NER任务，以确定网络攻击的归属。



## **14. ConfusedPilot: Compromising Enterprise Information Integrity and Confidentiality with Copilot for Microsoft 365**

ConfusedPilot：使用适用于Microsoft 365的Copilot损害企业信息完整性和机密性 cs.CR

**SubmitDate**: 2024-08-09    [abs](http://arxiv.org/abs/2408.04870v1) [paper-pdf](http://arxiv.org/pdf/2408.04870v1)

**Authors**: Ayush RoyChowdhury, Mulong Luo, Prateek Sahu, Sarbartha Banerjee, Mohit Tiwari

**Abstract**: Retrieval augmented generation (RAG) is a process where a large language model (LLM) retrieves useful information from a database and then generates the responses. It is becoming popular in enterprise settings for daily business operations. For example, Copilot for Microsoft 365 has accumulated millions of businesses. However, the security implications of adopting such RAG-based systems are unclear.   In this paper, we introduce ConfusedPilot, a class of security vulnerabilities of RAG systems that confuse Copilot and cause integrity and confidentiality violations in its responses. First, we investigate a vulnerability that embeds malicious text in the modified prompt in RAG, corrupting the responses generated by the LLM. Second, we demonstrate a vulnerability that leaks secret data, which leverages the caching mechanism during retrieval. Third, we investigate how both vulnerabilities can be exploited to propagate misinformation within the enterprise and ultimately impact its operations, such as sales and manufacturing. We also discuss the root cause of these attacks by investigating the architecture of a RAG-based system. This study highlights the security vulnerabilities in today's RAG-based systems and proposes design guidelines to secure future RAG-based systems.

摘要: 检索增强生成(RAG)是大型语言模型(LLM)从数据库中检索有用信息然后生成响应的过程。它在用于日常业务操作的企业环境中变得流行起来。例如，微软365的Copilot已经积累了数百万笔业务。然而，采用这种基于RAG的系统的安全影响尚不清楚。在本文中，我们介绍了一类RAG系统的安全漏洞ConfusedPilot，它迷惑了Copilot，并在其响应中导致完整性和保密性违规。首先，我们调查了一个漏洞，该漏洞将恶意文本嵌入到RAG中修改的提示符中，破坏了LLM生成的响应。其次，我们演示了一个泄漏机密数据的漏洞，该漏洞在检索过程中利用缓存机制。第三，我们调查如何利用这两个漏洞在企业内部传播错误信息，并最终影响其运营，如销售和制造。我们还通过研究基于RAG的系统的体系结构来讨论这些攻击的根本原因。这项研究强调了当今基于RAG的系统中的安全漏洞，并提出了保护未来基于RAG的系统的设计指南。



## **15. ChatGPT Meets Iris Biometrics**

ChatGPT与Iris Biatistics相遇 cs.CV

Published at IJCB 2024

**SubmitDate**: 2024-08-09    [abs](http://arxiv.org/abs/2408.04868v1) [paper-pdf](http://arxiv.org/pdf/2408.04868v1)

**Authors**: Parisa Farmanifard, Arun Ross

**Abstract**: This study utilizes the advanced capabilities of the GPT-4 multimodal Large Language Model (LLM) to explore its potential in iris recognition - a field less common and more specialized than face recognition. By focusing on this niche yet crucial area, we investigate how well AI tools like ChatGPT can understand and analyze iris images. Through a series of meticulously designed experiments employing a zero-shot learning approach, the capabilities of ChatGPT-4 was assessed across various challenging conditions including diverse datasets, presentation attacks, occlusions such as glasses, and other real-world variations. The findings convey ChatGPT-4's remarkable adaptability and precision, revealing its proficiency in identifying distinctive iris features, while also detecting subtle effects like makeup on iris recognition. A comparative analysis with Gemini Advanced - Google's AI model - highlighted ChatGPT-4's better performance and user experience in complex iris analysis tasks. This research not only validates the use of LLMs for specialized biometric applications but also emphasizes the importance of nuanced query framing and interaction design in extracting significant insights from biometric data. Our findings suggest a promising path for future research and the development of more adaptable, efficient, robust and interactive biometric security solutions.

摘要: 这项研究利用GPT-4多模式大语言模型(LLM)的高级功能来探索其在虹膜识别中的潜力--这是一个不太常见且比人脸识别更专业的领域。通过关注这一利基但又至关重要的领域，我们调查了像ChatGPT这样的人工智能工具对虹膜图像的理解和分析能力。通过采用零镜头学习方法的一系列精心设计的实验，评估了ChatGPT-4在各种具有挑战性的条件下的能力，包括不同的数据集、演示攻击、遮挡(如眼镜)和其他现实世界的变化。这些发现传达了ChatGPT-4‘S非凡的适应性和精确度，揭示了它在识别独特的虹膜特征方面的熟练程度，同时也检测到了化妆等对虹膜识别的细微影响。与谷歌人工智能模型Gemini Advanced的比较分析强调，ChatGPT-4的S在复杂的虹膜分析任务中具有更好的性能和用户体验。这项研究不仅验证了LLMS在专门的生物识别应用中的使用，而且强调了细微差别的查询框架和交互设计在从生物识别数据中提取重要见解的重要性。我们的发现为未来的研究和开发更具适应性、高效、健壮和交互的生物特征安全解决方案提供了一条有前途的道路。



## **16. h4rm3l: A Dynamic Benchmark of Composable Jailbreak Attacks for LLM Safety Assessment**

h4 rm3l：LLM安全评估的可组合越狱攻击的动态基准 cs.CR

**SubmitDate**: 2024-08-09    [abs](http://arxiv.org/abs/2408.04811v1) [paper-pdf](http://arxiv.org/pdf/2408.04811v1)

**Authors**: Moussa Koulako Bala Doumbouya, Ananjan Nandi, Gabriel Poesia, Davide Ghilardi, Anna Goldie, Federico Bianchi, Dan Jurafsky, Christopher D. Manning

**Abstract**: The safety of Large Language Models (LLMs) remains a critical concern due to a lack of adequate benchmarks for systematically evaluating their ability to resist generating harmful content. Previous efforts towards automated red teaming involve static or templated sets of illicit requests and adversarial prompts which have limited utility given jailbreak attacks' evolving and composable nature. We propose a novel dynamic benchmark of composable jailbreak attacks to move beyond static datasets and taxonomies of attacks and harms. Our approach consists of three components collectively called h4rm3l: (1) a domain-specific language that formally expresses jailbreak attacks as compositions of parameterized prompt transformation primitives, (2) bandit-based few-shot program synthesis algorithms that generate novel attacks optimized to penetrate the safety filters of a target black box LLM, and (3) open-source automated red-teaming software employing the previous two components. We use h4rm3l to generate a dataset of 2656 successful novel jailbreak attacks targeting 6 state-of-the-art (SOTA) open-source and proprietary LLMs. Several of our synthesized attacks are more effective than previously reported ones, with Attack Success Rates exceeding 90% on SOTA closed language models such as claude-3-haiku and GPT4-o. By generating datasets of jailbreak attacks in a unified formal representation, h4rm3l enables reproducible benchmarking and automated red-teaming, contributes to understanding LLM safety limitations, and supports the development of robust defenses in an increasingly LLM-integrated world.   Warning: This paper and related research artifacts contain offensive and potentially disturbing prompts and model-generated content.

摘要: 大型语言模型(LLM)的安全性仍然是一个严重的问题，因为缺乏系统地评估它们抵抗产生有害内容的能力的适当基准。以前的自动化红色团队的努力包括静态的或模板化的非法请求集和对抗性提示，鉴于越狱攻击不断演变和可组合的性质，这些提示的效用有限。我们提出了一种新的可组合越狱攻击的动态基准，以超越静态数据集和攻击和危害的分类。我们的方法由三个组件组成，统称为h4rm3l：(1)特定于领域的语言，它将越狱攻击形式化地表达为参数化提示转换原语的组合；(2)基于盗贼的少发程序合成算法，它生成经过优化的新型攻击，以穿透目标黑盒LLM的安全过滤器；以及(3)使用前两个组件的开源自动红队软件。我们使用h4rm3l生成了一个2656个成功的新型越狱攻击的数据集，目标是6个最先进的开源和专有LLM。我们的几个合成攻击比以前报道的更有效，在Claude-3-haiku和GPT4-o等Sota封闭语言模型上的攻击成功率超过90%。通过以统一的形式表示生成越狱攻击的数据集，h4rm3l实现了可重现的基准测试和自动化的红团队，有助于了解LLM的安全限制，并支持在日益集成LLM的世界中开发强大的防御措施。警告：本文和相关研究文章包含冒犯性和潜在令人不安的提示和模型生成的内容。



## **17. Tamper-Resistant Safeguards for Open-Weight LLMs**

开放重量LLM的防篡改保障措施 cs.LG

Website: https://www.tamper-resistant-safeguards.com

**SubmitDate**: 2024-08-08    [abs](http://arxiv.org/abs/2408.00761v2) [paper-pdf](http://arxiv.org/pdf/2408.00761v2)

**Authors**: Rishub Tamirisa, Bhrugu Bharathi, Long Phan, Andy Zhou, Alice Gatti, Tarun Suresh, Maxwell Lin, Justin Wang, Rowan Wang, Ron Arel, Andy Zou, Dawn Song, Bo Li, Dan Hendrycks, Mantas Mazeika

**Abstract**: Rapid advances in the capabilities of large language models (LLMs) have raised widespread concerns regarding their potential for malicious use. Open-weight LLMs present unique challenges, as existing safeguards lack robustness to tampering attacks that modify model weights. For example, recent works have demonstrated that refusal and unlearning safeguards can be trivially removed with a few steps of fine-tuning. These vulnerabilities necessitate new approaches for enabling the safe release of open-weight LLMs. We develop a method, called TAR, for building tamper-resistant safeguards into open-weight LLMs such that adversaries cannot remove the safeguards even after thousands of steps of fine-tuning. In extensive evaluations and red teaming analyses, we find that our method greatly improves tamper-resistance while preserving benign capabilities. Our results demonstrate that tamper-resistance is a tractable problem, opening up a promising new avenue to improve the safety and security of open-weight LLMs.

摘要: 大型语言模型(LLM)功能的快速发展引起了人们对其潜在恶意使用的广泛关注。开放重量LLM提出了独特的挑战，因为现有的保障措施缺乏对篡改模型权重的篡改攻击的稳健性。例如，最近的研究表明，通过几个步骤的微调，就可以很容易地消除拒绝和遗忘的保障措施。这些漏洞需要新的方法来实现安全释放未加重量的低密度脂蛋白。我们开发了一种名为TAR的方法，用于在开放重量的LLM中构建防篡改保护措施，以便对手即使在数千个步骤的微调之后也无法移除这些保护措施。在广泛的评估和红团队分析中，我们发现我们的方法在保持良性性能的同时大大提高了防篡改能力。我们的结果表明，防篡改是一个容易解决的问题，为提高开重LLMS的安全性开辟了一条很有前途的新途径。



## **18. Ensemble everything everywhere: Multi-scale aggregation for adversarial robustness**

包容无处不在的一切：多规模聚合以实现对抗稳健性 cs.CV

34 pages, 25 figures, appendix

**SubmitDate**: 2024-08-08    [abs](http://arxiv.org/abs/2408.05446v1) [paper-pdf](http://arxiv.org/pdf/2408.05446v1)

**Authors**: Stanislav Fort, Balaji Lakshminarayanan

**Abstract**: Adversarial examples pose a significant challenge to the robustness, reliability and alignment of deep neural networks. We propose a novel, easy-to-use approach to achieving high-quality representations that lead to adversarial robustness through the use of multi-resolution input representations and dynamic self-ensembling of intermediate layer predictions. We demonstrate that intermediate layer predictions exhibit inherent robustness to adversarial attacks crafted to fool the full classifier, and propose a robust aggregation mechanism based on Vickrey auction that we call \textit{CrossMax} to dynamically ensemble them. By combining multi-resolution inputs and robust ensembling, we achieve significant adversarial robustness on CIFAR-10 and CIFAR-100 datasets without any adversarial training or extra data, reaching an adversarial accuracy of $\approx$72% (CIFAR-10) and $\approx$48% (CIFAR-100) on the RobustBench AutoAttack suite ($L_\infty=8/255)$ with a finetuned ImageNet-pretrained ResNet152. This represents a result comparable with the top three models on CIFAR-10 and a +5 % gain compared to the best current dedicated approach on CIFAR-100. Adding simple adversarial training on top, we get $\approx$78% on CIFAR-10 and $\approx$51% on CIFAR-100, improving SOTA by 5 % and 9 % respectively and seeing greater gains on the harder dataset. We validate our approach through extensive experiments and provide insights into the interplay between adversarial robustness, and the hierarchical nature of deep representations. We show that simple gradient-based attacks against our model lead to human-interpretable images of the target classes as well as interpretable image changes. As a byproduct, using our multi-resolution prior, we turn pre-trained classifiers and CLIP models into controllable image generators and develop successful transferable attacks on large vision language models.

摘要: 对抗性的例子对深度神经网络的稳健性、可靠性和对齐提出了巨大的挑战。我们提出了一种新颖的、易于使用的方法，通过使用多分辨率输入表示和中间层预测的动态自集成来获得高质量的表示，从而导致对抗性健壮性。我们证明了中间层预测对于欺骗完整分类器的敌意攻击表现出固有的稳健性，并提出了一种基于Vickrey拍卖的健壮聚集机制，我们称之为\textit{CRossmax}来动态集成它们。通过结合多分辨率输入和稳健集成，我们在CIFAR-10和CIFAR-100数据集上实现了显著的对抗稳健性，而无需任何对抗性训练或额外数据，在RobustBack AutoAttack套件($L_\INFTY=8/255)$上达到了约$\\72%(CIFAR-10)和$\\约48%(CIFAR-100)的对抗准确率。这代表了一个可以与CIFAR-10上的前三个型号相媲美的结果，并且与CIFAR-100上当前最好的专用方法相比，增加了5%。加上简单的对抗性训练，我们在CIFAR-10上获得了约78%的收益，在CIFAR-100上获得了约51%的收益，分别将SOTA提高了5%和9%，并在较难的数据集上看到了更大的收益。我们通过广泛的实验验证了我们的方法，并对对手健壮性和深层表示的层次性之间的相互作用提供了见解。我们表明，对我们的模型的简单的基于梯度的攻击会导致目标类的人类可解释的图像以及可解释的图像变化。作为一个副产品，我们利用我们的多分辨率先验知识，将预先训练的分类器和剪辑模型转化为可控的图像生成器，并成功地开发了对大型视觉语言模型的可转移攻击。



## **19. Duwak: Dual Watermarks in Large Language Models**

Duwak：大型语言模型中的双重水印 cs.LG

**SubmitDate**: 2024-08-08    [abs](http://arxiv.org/abs/2403.13000v2) [paper-pdf](http://arxiv.org/pdf/2403.13000v2)

**Authors**: Chaoyi Zhu, Jeroen Galjaard, Pin-Yu Chen, Lydia Y. Chen

**Abstract**: As large language models (LLM) are increasingly used for text generation tasks, it is critical to audit their usages, govern their applications, and mitigate their potential harms. Existing watermark techniques are shown effective in embedding single human-imperceptible and machine-detectable patterns without significantly affecting generated text quality and semantics. However, the efficiency in detecting watermarks, i.e., the minimum number of tokens required to assert detection with significance and robustness against post-editing, is still debatable. In this paper, we propose, Duwak, to fundamentally enhance the efficiency and quality of watermarking by embedding dual secret patterns in both token probability distribution and sampling schemes. To mitigate expression degradation caused by biasing toward certain tokens, we design a contrastive search to watermark the sampling scheme, which minimizes the token repetition and enhances the diversity. We theoretically explain the interdependency of the two watermarks within Duwak. We evaluate Duwak extensively on Llama2 under various post-editing attacks, against four state-of-the-art watermarking techniques and combinations of them. Our results show that Duwak marked text achieves the highest watermarked text quality at the lowest required token count for detection, up to 70% tokens less than existing approaches, especially under post paraphrasing.

摘要: 随着大型语言模型(LLM)越来越多地用于文本生成任务，审计它们的使用情况、管理它们的应用程序并减轻它们的潜在危害至关重要。现有的水印技术在不显著影响生成的文本质量和语义的情况下，有效地嵌入了单一的人类不可感知和机器可检测的图案。然而，检测水印的效率，即断言检测具有重要性和对编辑后的稳健性所需的最小令牌数量，仍然是有争议的。在本文中，我们提出了Duwak，通过在令牌概率分布和抽样方案中嵌入双重秘密模式，从根本上提高了水印的效率和质量。为了缓解由于偏向某些标记而导致的表达质量下降，我们设计了一种对比搜索来在采样方案中加入水印，从而最大限度地减少了标记的重复度，提高了多样性。我们从理论上解释了Duwak中两个水印之间的相互依赖关系。我们在不同的编辑后攻击下对Llama2上的Duwak进行了广泛的评估，对比了四种最先进的水印技术及其组合。我们的结果表明，Duwak标记的文本在检测所需的最低标记数的情况下获得了最高的水印文本质量，比现有方法减少了70%的标记量，特别是在转译后的情况下。



## **20. Towards Explainable Network Intrusion Detection using Large Language Models**

使用大型语言模型实现可解释的网络入侵检测 cs.CR

**SubmitDate**: 2024-08-08    [abs](http://arxiv.org/abs/2408.04342v1) [paper-pdf](http://arxiv.org/pdf/2408.04342v1)

**Authors**: Paul R. B. Houssel, Priyanka Singh, Siamak Layeghy, Marius Portmann

**Abstract**: Large Language Models (LLMs) have revolutionised natural language processing tasks, particularly as chat agents. However, their applicability to threat detection problems remains unclear. This paper examines the feasibility of employing LLMs as a Network Intrusion Detection System (NIDS), despite their high computational requirements, primarily for the sake of explainability. Furthermore, considerable resources have been invested in developing LLMs, and they may offer utility for NIDS. Current state-of-the-art NIDS rely on artificial benchmarking datasets, resulting in skewed performance when applied to real-world networking environments. Therefore, we compare the GPT-4 and LLama3 models against traditional architectures and transformer-based models to assess their ability to detect malicious NetFlows without depending on artificially skewed datasets, but solely on their vast pre-trained acquired knowledge. Our results reveal that, although LLMs struggle with precise attack detection, they hold significant potential for a path towards explainable NIDS. Our preliminary exploration shows that LLMs are unfit for the detection of Malicious NetFlows. Most promisingly, however, these exhibit significant potential as complementary agents in NIDS, particularly in providing explanations and aiding in threat response when integrated with Retrieval Augmented Generation (RAG) and function calling capabilities.

摘要: 大型语言模型(LLM)彻底改变了自然语言处理任务，尤其是作为聊天代理。然而，它们在威胁检测问题上的适用性仍不清楚。本文研究了使用LLMS作为网络入侵检测系统(NIDS)的可行性，尽管它们对计算的要求很高，主要是为了解释。此外，已经投入了大量资源来开发低成本管理系统，它们可能会为网络入侵检测系统提供实用服务。当前最先进的NID依赖于人工基准数据集，在应用于真实网络环境时会导致性能偏差。因此，我们将GPT-4和LLama3模型与传统架构和基于变压器的模型进行比较，以评估它们检测恶意NetFlow的能力，而不依赖于人为倾斜的数据集，而仅仅依赖于它们庞大的预训练获取的知识。我们的结果表明，尽管LLMS在精确的攻击检测方面困难重重，但它们具有通往可解释的NID的巨大潜力。我们的初步探索表明，LLMS不适合检测恶意NetFlow。然而，最有希望的是，这些代理在NIDS中显示出作为补充代理的巨大潜力，特别是在与检索增强生成(RAG)和函数调用功能集成时提供解释和帮助应对威胁。



## **21. Multi-Turn Context Jailbreak Attack on Large Language Models From First Principles**

从第一原则出发对大型语言模型的多轮上下文越狱攻击 cs.CL

**SubmitDate**: 2024-08-08    [abs](http://arxiv.org/abs/2408.04686v1) [paper-pdf](http://arxiv.org/pdf/2408.04686v1)

**Authors**: Xiongtao Sun, Deyue Zhang, Dongdong Yang, Quanchen Zou, Hui Li

**Abstract**: Large language models (LLMs) have significantly enhanced the performance of numerous applications, from intelligent conversations to text generation. However, their inherent security vulnerabilities have become an increasingly significant challenge, especially with respect to jailbreak attacks. Attackers can circumvent the security mechanisms of these LLMs, breaching security constraints and causing harmful outputs. Focusing on multi-turn semantic jailbreak attacks, we observe that existing methods lack specific considerations for the role of multiturn dialogues in attack strategies, leading to semantic deviations during continuous interactions. Therefore, in this paper, we establish a theoretical foundation for multi-turn attacks by considering their support in jailbreak attacks, and based on this, propose a context-based contextual fusion black-box jailbreak attack method, named Context Fusion Attack (CFA). This method approach involves filtering and extracting key terms from the target, constructing contextual scenarios around these terms, dynamically integrating the target into the scenarios, replacing malicious key terms within the target, and thereby concealing the direct malicious intent. Through comparisons on various mainstream LLMs and red team datasets, we have demonstrated CFA's superior success rate, divergence, and harmfulness compared to other multi-turn attack strategies, particularly showcasing significant advantages on Llama3 and GPT-4.

摘要: 大型语言模型(LLM)显著提高了从智能对话到文本生成的众多应用程序的性能。然而，它们固有的安全漏洞已成为一个日益重大的挑战，特别是在越狱攻击方面。攻击者可以绕过这些LLM的安全机制，违反安全限制并造成有害的输出。针对多轮语义越狱攻击，我们观察到现有的方法缺乏对多轮对话在攻击策略中的作用的具体考虑，导致在持续交互过程中出现语义偏差。因此，本文通过考虑多轮攻击对越狱攻击的支持，为多轮攻击奠定了理论基础，并在此基础上提出了一种基于上下文融合的黑盒越狱攻击方法，称为上下文融合攻击(CFA)。该方法包括从目标过滤和提取关键字，围绕这些关键字构建上下文场景，动态地将目标集成到场景中，替换目标内的恶意关键字，从而隐藏直接恶意意图。通过在各种主流LLM和RED团队数据集上的比较，我们已经证明了CFA相对于其他多回合攻击策略具有更高的成功率、发散性和危害性，特别是在Llama3和GPT-4上表现出显著的优势。



## **22. Effective Prompt Extraction from Language Models**

从语言模型中有效的提示提取 cs.CL

**SubmitDate**: 2024-08-07    [abs](http://arxiv.org/abs/2307.06865v3) [paper-pdf](http://arxiv.org/pdf/2307.06865v3)

**Authors**: Yiming Zhang, Nicholas Carlini, Daphne Ippolito

**Abstract**: The text generated by large language models is commonly controlled by prompting, where a prompt prepended to a user's query guides the model's output. The prompts used by companies to guide their models are often treated as secrets, to be hidden from the user making the query. They have even been treated as commodities to be bought and sold on marketplaces. However, anecdotal reports have shown adversarial users employing prompt extraction attacks to recover these prompts. In this paper, we present a framework for systematically measuring the effectiveness of these attacks. In experiments with 3 different sources of prompts and 11 underlying large language models, we find that simple text-based attacks can in fact reveal prompts with high probability. Our framework determines with high precision whether an extracted prompt is the actual secret prompt, rather than a model hallucination. Prompt extraction from real systems such as Claude 3 and ChatGPT further suggest that system prompts can be revealed by an adversary despite existing defenses in place.

摘要: 大型语言模型生成的文本通常通过提示进行控制，其中用户查询前的提示将指导模型的输出。公司用来指导其模型的提示通常被视为秘密，对进行查询的用户隐藏。它们甚至被视为可以在市场上买卖的商品。然而，坊间报道显示，敌意用户使用提示提取攻击来恢复这些提示。在本文中，我们提出了一个系统地衡量这些攻击的有效性的框架。在对3种不同的提示源和11个基本的大型语言模型进行的实验中，我们发现简单的基于文本的攻击实际上可以高概率地揭示提示。我们的框架高精度地确定提取的提示是否是实际的秘密提示，而不是模型幻觉。从Claude 3和ChatGPT等真实系统中提取提示进一步表明，尽管已有防御措施，但系统提示仍可被对手泄露。



## **23. Decoding Biases: Automated Methods and LLM Judges for Gender Bias Detection in Language Models**

解码偏见：语言模型中性别偏见检测的自动方法和LLM法官 cs.CL

6 pages paper content, 17 pages of appendix

**SubmitDate**: 2024-08-07    [abs](http://arxiv.org/abs/2408.03907v1) [paper-pdf](http://arxiv.org/pdf/2408.03907v1)

**Authors**: Shachi H Kumar, Saurav Sahay, Sahisnu Mazumder, Eda Okur, Ramesh Manuvinakurike, Nicole Beckage, Hsuan Su, Hung-yi Lee, Lama Nachman

**Abstract**: Large Language Models (LLMs) have excelled at language understanding and generating human-level text. However, even with supervised training and human alignment, these LLMs are susceptible to adversarial attacks where malicious users can prompt the model to generate undesirable text. LLMs also inherently encode potential biases that can cause various harmful effects during interactions. Bias evaluation metrics lack standards as well as consensus and existing methods often rely on human-generated templates and annotations which are expensive and labor intensive. In this work, we train models to automatically create adversarial prompts to elicit biased responses from target LLMs. We present LLM- based bias evaluation metrics and also analyze several existing automatic evaluation methods and metrics. We analyze the various nuances of model responses, identify the strengths and weaknesses of model families, and assess where evaluation methods fall short. We compare these metrics to human evaluation and validate that the LLM-as-a-Judge metric aligns with human judgement on bias in response generation.

摘要: 大型语言模型(LLM)在语言理解和生成人类级别的文本方面表现出色。然而，即使在有监督的训练和人类对齐的情况下，这些LLM也容易受到敌意攻击，恶意用户可以提示模型生成不想要的文本。LLM还固有地编码潜在的偏见，这些偏见可能在相互作用期间造成各种有害影响。偏差评估指标缺乏标准和共识，现有的方法往往依赖于人工生成的模板和注释，这些模板和注释昂贵且劳动密集型。在这项工作中，我们训练模型自动创建对抗性提示，以引起目标LLM的偏见反应。提出了基于LLM的偏差评价指标，并分析了现有的几种自动评价方法和指标。我们分析模型响应的各种细微差别，确定模型家庭的优点和缺点，并评估评估方法的不足之处。我们将这些指标与人类评估进行比较，并验证LLM作为法官的指标与人类对响应生成中的偏差的判断一致。



## **24. Trading Devil Final: Backdoor attack via Stock market and Bayesian Optimization**

交易魔鬼决赛：通过股市和Bayesian优化进行后门攻击 cs.LG

END :jumps-Diffusion and stock market: Better quantify uncertainty in  financial simulations

**SubmitDate**: 2024-08-07    [abs](http://arxiv.org/abs/2407.14573v2) [paper-pdf](http://arxiv.org/pdf/2407.14573v2)

**Authors**: Orson Mengara

**Abstract**: Since the advent of generative artificial intelligence, every company and researcher has been rushing to develop their own generative models, whether commercial or not. Given the large number of users of these powerful new tools, there is currently no intrinsically verifiable way to explain from the ground up what happens when LLMs (large language models) learn. For example, those based on automatic speech recognition systems, which have to rely on huge and astronomical amounts of data collected from all over the web to produce fast and efficient results, In this article, we develop a backdoor attack called MarketBackFinal 2.0, based on acoustic data poisoning, MarketBackFinal 2.0 is mainly based on modern stock market models. In order to show the possible vulnerabilities of speech-based transformers that may rely on LLMs.

摘要: 自生成人工智能出现以来，每家公司和研究人员都在争先恐后地开发自己的生成模型，无论是否商业化。鉴于这些强大的新工具的大量用户，目前还没有本质上可验证的方法来从头解释LLM（大型语言模型）学习时会发生什么。例如，那些基于自动语音识别系统的系统，它们必须依赖于从整个网络收集的大量数据来产生快速有效的结果，在本文中，我们开发了一种名为MarketBackFinal 2.0的后门攻击，基于声学数据中毒，MarketBackFinal 2.0主要基于现代股市模型。为了显示可能依赖LLM的基于语音的转换器可能存在的漏洞。



## **25. EnJa: Ensemble Jailbreak on Large Language Models**

EnJa：大型语言模型上的越狱 cs.CR

**SubmitDate**: 2024-08-07    [abs](http://arxiv.org/abs/2408.03603v1) [paper-pdf](http://arxiv.org/pdf/2408.03603v1)

**Authors**: Jiahao Zhang, Zilong Wang, Ruofan Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: As Large Language Models (LLMs) are increasingly being deployed in safety-critical applications, their vulnerability to potential jailbreaks -- malicious prompts that can disable the safety mechanism of LLMs -- has attracted growing research attention. While alignment methods have been proposed to protect LLMs from jailbreaks, many have found that aligned LLMs can still be jailbroken by carefully crafted malicious prompts, producing content that violates policy regulations. Existing jailbreak attacks on LLMs can be categorized into prompt-level methods which make up stories/logic to circumvent safety alignment and token-level attack methods which leverage gradient methods to find adversarial tokens. In this work, we introduce the concept of Ensemble Jailbreak and explore methods that can integrate prompt-level and token-level jailbreak into a more powerful hybrid jailbreak attack. Specifically, we propose a novel EnJa attack to hide harmful instructions using prompt-level jailbreak, boost the attack success rate using a gradient-based attack, and connect the two types of jailbreak attacks via a template-based connector. We evaluate the effectiveness of EnJa on several aligned models and show that it achieves a state-of-the-art attack success rate with fewer queries and is much stronger than any individual jailbreak.

摘要: 随着大型语言模型(LLM)越来越多地被部署在安全关键型应用程序中，它们对潜在越狱的脆弱性--可以禁用LLM安全机制的恶意提示--引起了越来越多的研究关注。虽然已经提出了一些方法来保护LLM免受越狱之苦，但许多人发现，通过精心设计的恶意提示，仍然可以通过精心设计的恶意提示来越狱，从而产生违反政策规定的内容。现有的针对LLMS的越狱攻击可以分为两种：一种是编造故事/逻辑来规避安全对齐的提示级攻击方法，另一种是利用梯度方法来寻找对抗性令牌的令牌级攻击方法。在这项工作中，我们引入了集成越狱的概念，并探索了可以将提示级和令牌级越狱集成到更强大的混合越狱攻击中的方法。具体地说，我们提出了一种新的Enja攻击，利用提示级越狱隐藏有害指令，使用基于梯度的攻击提高攻击成功率，并通过基于模板的连接器将两种类型的越狱攻击连接起来。我们在几个对齐的模型上对Enja的有效性进行了评估，结果表明，它以更少的查询获得了最先进的攻击成功率，并且比任何单个越狱都要强大得多。



## **26. Empirical Analysis of Large Vision-Language Models against Goal Hijacking via Visual Prompt Injection**

大型视觉语言模型通过视觉提示注入对抗目标劫持的实证分析 cs.CL

8 pages, 6 figures, Accepted to NAACL 2024 SRW

**SubmitDate**: 2024-08-07    [abs](http://arxiv.org/abs/2408.03554v1) [paper-pdf](http://arxiv.org/pdf/2408.03554v1)

**Authors**: Subaru Kimura, Ryota Tanaka, Shumpei Miyawaki, Jun Suzuki, Keisuke Sakaguchi

**Abstract**: We explore visual prompt injection (VPI) that maliciously exploits the ability of large vision-language models (LVLMs) to follow instructions drawn onto the input image. We propose a new VPI method, "goal hijacking via visual prompt injection" (GHVPI), that swaps the execution task of LVLMs from an original task to an alternative task designated by an attacker. The quantitative analysis indicates that GPT-4V is vulnerable to the GHVPI and demonstrates a notable attack success rate of 15.8%, which is an unignorable security risk. Our analysis also shows that successful GHVPI requires high character recognition capability and instruction-following ability in LVLMs.

摘要: 我们探索了视觉提示注入（Veritas），它恶意利用大型视觉语言模型（LVLM）遵循绘制在输入图像上的指令的能力。我们提出了一种新的PRI方法，“通过视觉提示注入的目标劫持”（GHPPI），它将LVLM的执行任务从原始任务交换到攻击者指定的替代任务。定量分析表明GPT-4V容易受到GHPPI的攻击，攻击成功率为15.8%，这是一个无法预测的安全风险。我们的分析还表明，成功的GHPPI需要LVLM中具有很高的字符识别能力和描述跟踪能力。



## **27. A Study on Prompt Injection Attack Against LLM-Integrated Mobile Robotic Systems**

针对LLM集成移动机器人系统的即时注入攻击研究 cs.RO

**SubmitDate**: 2024-08-07    [abs](http://arxiv.org/abs/2408.03515v1) [paper-pdf](http://arxiv.org/pdf/2408.03515v1)

**Authors**: Wenxiao Zhang, Xiangrui Kong, Conan Dewitt, Thomas Braunl, Jin B. Hong

**Abstract**: The integration of Large Language Models (LLMs) like GPT-4o into robotic systems represents a significant advancement in embodied artificial intelligence. These models can process multi-modal prompts, enabling them to generate more context-aware responses. However, this integration is not without challenges. One of the primary concerns is the potential security risks associated with using LLMs in robotic navigation tasks. These tasks require precise and reliable responses to ensure safe and effective operation. Multi-modal prompts, while enhancing the robot's understanding, also introduce complexities that can be exploited maliciously. For instance, adversarial inputs designed to mislead the model can lead to incorrect or dangerous navigational decisions. This study investigates the impact of prompt injections on mobile robot performance in LLM-integrated systems and explores secure prompt strategies to mitigate these risks. Our findings demonstrate a substantial overall improvement of approximately 30.8% in both attack detection and system performance with the implementation of robust defence mechanisms, highlighting their critical role in enhancing security and reliability in mission-oriented tasks.

摘要: 将像GPT-40这样的大型语言模型(LLM)集成到机器人系统中，代表着体现的人工智能的重大进步。这些模型可以处理多模式提示，使它们能够生成更多情景感知响应。然而，这种整合并不是没有挑战。其中一个主要问题是在机器人导航任务中使用LLMS存在潜在的安全风险。这些任务需要准确可靠的反应，以确保安全有效的运行。多模式提示在增强机器人理解能力的同时，也引入了可能被恶意利用的复杂性。例如，旨在误导模型的对抗性输入可能导致错误或危险的导航决策。这项研究调查了快速注射对LLM集成系统中移动机器人性能的影响，并探索了安全的提示策略来缓解这些风险。我们的研究结果表明，随着强大的防御机制的实施，攻击检测和系统性能都有了大约30.8%的大幅整体改进，突出了它们在增强面向任务的任务的安全性和可靠性方面的关键作用。



## **28. Best-of-Venom: Attacking RLHF by Injecting Poisoned Preference Data**

毒液最佳：通过注入有毒偏好数据来攻击WLHF cs.CL

**SubmitDate**: 2024-08-06    [abs](http://arxiv.org/abs/2404.05530v2) [paper-pdf](http://arxiv.org/pdf/2404.05530v2)

**Authors**: Tim Baumgärtner, Yang Gao, Dana Alon, Donald Metzler

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is a popular method for aligning Language Models (LM) with human values and preferences. RLHF requires a large number of preference pairs as training data, which are often used in both the Supervised Fine-Tuning and Reward Model training and therefore publicly available datasets are commonly used. In this work, we study to what extent a malicious actor can manipulate the LMs generations by poisoning the preferences, i.e., injecting poisonous preference pairs into these datasets and the RLHF training process. We propose strategies to build poisonous preference pairs and test their performance by poisoning two widely used preference datasets. Our results show that preference poisoning is highly effective: injecting a small amount of poisonous data (1-5\% of the original dataset), we can effectively manipulate the LM to generate a target entity in a target sentiment (positive or negative). The findings from our experiments also shed light on strategies to defend against the preference poisoning attack.

摘要: 人类反馈强化学习(RLHF)是一种使语言模型与人类价值观和偏好保持一致的流行方法。RLHF需要大量的偏好对作为训练数据，这些数据经常用于有监督的精调和奖励模型训练，因此通常使用公开可用的数据集。在这项工作中，我们研究了恶意行为者可以在多大程度上通过毒化偏好来操纵LMS生成，即向这些数据集注入有毒的偏好对和RLHF训练过程。我们提出了构建有毒偏好对的策略，并通过毒化两个广泛使用的偏好数据集来测试它们的性能。我们的结果表明偏好毒化是非常有效的：注入少量的有毒数据(原始数据集的1-5\%)，我们可以有效地操纵LM来生成目标情感中的目标实体(积极或消极)。我们的实验结果也为防御偏好中毒攻击的策略提供了启示。



## **29. Rethinking Jailbreaking through the Lens of Representation Engineering**

从表象工程的角度重新思考越狱 cs.CL

21 pages, 20 figures, 6 tables

**SubmitDate**: 2024-08-06    [abs](http://arxiv.org/abs/2401.06824v3) [paper-pdf](http://arxiv.org/pdf/2401.06824v3)

**Authors**: Tianlong Li, Shihan Dou, Wenhao Liu, Muling Wu, Changze Lv, Rui Zheng, Xiaoqing Zheng, Xuanjing Huang

**Abstract**: The recent surge in jailbreaking methods has revealed the vulnerability of Large Language Models (LLMs) to malicious inputs. While earlier research has primarily concentrated on increasing the success rates of jailbreaking attacks, the underlying mechanism for safeguarding LLMs remains underexplored. This study investigates the vulnerability of safety-aligned LLMs by uncovering specific activity patterns within the representation space generated by LLMs. Such ``safety patterns'' can be identified with only a few pairs of contrastive queries in a simple method and function as ``keys'' (used as a metaphor for security defense capability) that can be used to open or lock Pandora's Box of LLMs. Extensive experiments demonstrate that the robustness of LLMs against jailbreaking can be lessened or augmented by attenuating or strengthening the identified safety patterns. These findings deepen our understanding of jailbreaking phenomena and call for the LLM community to address the potential misuse of open-source LLMs.

摘要: 最近越狱方法的激增揭示了大型语言模型(LLM)对恶意输入的脆弱性。虽然早期的研究主要集中在提高越狱攻击的成功率上，但保护LLMS的潜在机制仍未得到充分探索。本研究通过揭示LLMS生成的表示空间中的特定活动模式来研究安全对齐的LLM的脆弱性。这种“安全模式”只需用几对对比查询就能以一种简单的方法识别出来，并起到“钥匙”的作用(用作安全防御能力的比喻)，可用来打开或锁定LLM的潘多拉盒子。广泛的实验表明，通过减弱或加强已识别的安全模式，可以降低或增强LLMS对越狱的稳健性。这些发现加深了我们对越狱现象的理解，并呼吁LLM社区解决开源LLM的潜在滥用问题。



## **30. Compromising Embodied Agents with Contextual Backdoor Attacks**

通过上下文后门攻击损害被授权的代理 cs.AI

**SubmitDate**: 2024-08-06    [abs](http://arxiv.org/abs/2408.02882v1) [paper-pdf](http://arxiv.org/pdf/2408.02882v1)

**Authors**: Aishan Liu, Yuguang Zhou, Xianglong Liu, Tianyuan Zhang, Siyuan Liang, Jiakai Wang, Yanjun Pu, Tianlin Li, Junqi Zhang, Wenbo Zhou, Qing Guo, Dacheng Tao

**Abstract**: Large language models (LLMs) have transformed the development of embodied intelligence. By providing a few contextual demonstrations, developers can utilize the extensive internal knowledge of LLMs to effortlessly translate complex tasks described in abstract language into sequences of code snippets, which will serve as the execution logic for embodied agents. However, this paper uncovers a significant backdoor security threat within this process and introduces a novel method called \method{}. By poisoning just a few contextual demonstrations, attackers can covertly compromise the contextual environment of a black-box LLM, prompting it to generate programs with context-dependent defects. These programs appear logically sound but contain defects that can activate and induce unintended behaviors when the operational agent encounters specific triggers in its interactive environment. To compromise the LLM's contextual environment, we employ adversarial in-context generation to optimize poisoned demonstrations, where an LLM judge evaluates these poisoned prompts, reporting to an additional LLM that iteratively optimizes the demonstration in a two-player adversarial game using chain-of-thought reasoning. To enable context-dependent behaviors in downstream agents, we implement a dual-modality activation strategy that controls both the generation and execution of program defects through textual and visual triggers. We expand the scope of our attack by developing five program defect modes that compromise key aspects of confidentiality, integrity, and availability in embodied agents. To validate the effectiveness of our approach, we conducted extensive experiments across various tasks, including robot planning, robot manipulation, and compositional visual reasoning. Additionally, we demonstrate the potential impact of our approach by successfully attacking real-world autonomous driving systems.

摘要: 大型语言模型(LLM)改变了体验式智能的发展。通过提供一些上下文演示，开发人员可以利用LLM的丰富内部知识，毫不费力地将以抽象语言描述的复杂任务转换为代码片段序列，这些代码片段将用作具体化代理的执行逻辑。然而，本文发现了该过程中存在的一个严重的后门安全威胁，并介绍了一种名为\方法{}的新方法。只要毒化几个上下文演示，攻击者就可以秘密地危害黑盒LLM的上下文环境，促使它生成具有上下文相关缺陷的程序。这些程序在逻辑上看起来是合理的，但存在缺陷，当操作代理在其交互环境中遇到特定触发器时，这些缺陷可能会激活和诱导意外行为。为了折衷LLM的上下文环境，我们使用对抗性的上下文生成来优化有毒演示，其中LLM法官评估这些有毒的提示，向额外的LLM报告，该LLM使用思想链推理在两人对抗性游戏中迭代优化演示。为了在下游代理中实现上下文相关的行为，我们实现了一种双通道激活策略，该策略通过文本和视觉触发来控制程序缺陷的生成和执行。我们通过开发五种程序缺陷模式来扩大我们的攻击范围，这些模式损害了具体化代理中的机密性、完整性和可用性的关键方面。为了验证我们方法的有效性，我们在各种任务中进行了广泛的实验，包括机器人规划、机器人操作和组合视觉推理。此外，我们通过成功攻击真实世界的自动驾驶系统来演示我们的方法的潜在影响。



## **31. SEAS: Self-Evolving Adversarial Safety Optimization for Large Language Models**

SEAS：大型语言模型的自进化对抗安全优化 cs.CL

**SubmitDate**: 2024-08-05    [abs](http://arxiv.org/abs/2408.02632v1) [paper-pdf](http://arxiv.org/pdf/2408.02632v1)

**Authors**: Muxi Diao, Rumei Li, Shiyang Liu, Guogang Liao, Jingang Wang, Xunliang Cai, Weiran Xu

**Abstract**: As large language models (LLMs) continue to advance in capability and influence, ensuring their security and preventing harmful outputs has become crucial. A promising approach to address these concerns involves training models to automatically generate adversarial prompts for red teaming. However, the evolving subtlety of vulnerabilities in LLMs challenges the effectiveness of current adversarial methods, which struggle to specifically target and explore the weaknesses of these models. To tackle these challenges, we introduce the $\mathbf{S}\text{elf-}\mathbf{E}\text{volving }\mathbf{A}\text{dversarial }\mathbf{S}\text{afety }\mathbf{(SEAS)}$ optimization framework, which enhances security by leveraging data generated by the model itself. SEAS operates through three iterative stages: Initialization, Attack, and Adversarial Optimization, refining both the Red Team and Target models to improve robustness and safety. This framework reduces reliance on manual testing and significantly enhances the security capabilities of LLMs. Our contributions include a novel adversarial framework, a comprehensive safety dataset, and after three iterations, the Target model achieves a security level comparable to GPT-4, while the Red Team model shows a marked increase in attack success rate (ASR) against advanced models.

摘要: 随着大型语言模型在能力和影响力方面的不断进步，确保它们的安全和防止有害输出变得至关重要。解决这些担忧的一个有希望的方法是建立训练模型，为红色团队自动生成对抗性提示。然而，LLMS中不断演变的漏洞的微妙之处挑战了当前对抗性方法的有效性，这些方法难以具体针对和探索这些模型的弱点。为了应对这些挑战，我们引入了$\mathbf{S}\Text{ELF-}\mathbf{E}\Text{volving}\mathbf{A}\Text{dversarial}\mathbf{S}\Text{afty}\mathbf{(SEA)}$优化框架，该框架通过利用模型本身生成的数据来增强安全性。SEA经历了三个迭代阶段：初始化、攻击和对抗性优化，完善了Red Team和Target模型，以提高健壮性和安全性。该框架减少了对手动测试的依赖，显著增强了LLMS的安全能力。我们的贡献包括一个新的对抗性框架，一个全面的安全数据集，经过三次迭代，Target模型达到了与GPT-4相当的安全级别，而Red Team模型显示出相对于高级模型在攻击成功率(ASR)方面的显著提高。



## **32. Practical Attacks against Black-box Code Completion Engines**

针对黑匣子代码完成引擎的实际攻击 cs.CR

**SubmitDate**: 2024-08-05    [abs](http://arxiv.org/abs/2408.02509v1) [paper-pdf](http://arxiv.org/pdf/2408.02509v1)

**Authors**: Slobodan Jenko, Jingxuan He, Niels Mündler, Mark Vero, Martin Vechev

**Abstract**: Modern code completion engines, powered by large language models, have demonstrated impressive capabilities to generate functionally correct code based on surrounding context. As these tools are extensively used by millions of developers, it is crucial to investigate their security implications. In this work, we present INSEC, a novel attack that directs code completion engines towards generating vulnerable code. In line with most commercial completion engines, such as GitHub Copilot, INSEC assumes only black-box query access to the targeted engine, without requiring any knowledge of the engine's internals. Our attack works by inserting a malicious attack string as a short comment in the completion input. To derive the attack string, we design a series of specialized initialization schemes and an optimization procedure for further refinement. We demonstrate the strength of INSEC not only on state-of-the-art open-source models but also on black-box commercial services such as the OpenAI API and GitHub Copilot. On a comprehensive set of security-critical test cases covering 16 CWEs across 5 programming languages, INSEC significantly increases the likelihood of the considered completion engines in generating unsafe code by >50% in absolute, while maintaining the ability in producing functionally correct code. At the same time, our attack has low resource requirements, and can be developed for a cost of well under ten USD on commodity hardware.

摘要: 在大型语言模型的支持下，现代代码完成引擎已经展示了令人印象深刻的能力，可以根据周围的上下文生成功能正确的代码。由于这些工具被数百万开发人员广泛使用，因此调查它们的安全影响至关重要。在这项工作中，我们介绍了INSEC，一种新型的攻击，它引导代码完成引擎生成易受攻击的代码。与大多数商业完成引擎(如GitHub Copilot)一样，INSEC假定只对目标引擎进行黑盒查询访问，而不需要了解引擎的内部结构。我们的攻击是通过在补全输入中插入恶意攻击字符串作为简短注释来实现的。为了得到攻击字符串，我们设计了一系列专门的初始化方案和优化程序来进一步细化。我们不仅在最先进的开源模型上展示了INSEC的优势，而且在OpenAI API和GitHub Copilot等黑盒商业服务上也展示了INSEC的优势。在涵盖5种编程语言的16个CWE的一组全面的安全关键测试用例上，INSEC显著增加了被考虑的完成引擎生成不安全代码的可能性，绝对值超过50%，同时保持了生成功能正确代码的能力。同时，我们的攻击对资源的要求很低，并且可以在商用硬件上以远低于10美元的成本进行开发。



## **33. Why Are My Prompts Leaked? Unraveling Prompt Extraction Threats in Customized Large Language Models**

为什么我的笔记会泄露？解开定制大型语言模型中的提示提取威胁 cs.CL

**SubmitDate**: 2024-08-05    [abs](http://arxiv.org/abs/2408.02416v1) [paper-pdf](http://arxiv.org/pdf/2408.02416v1)

**Authors**: Zi Liang, Haibo Hu, Qingqing Ye, Yaxin Xiao, Haoyang Li

**Abstract**: The drastic increase of large language models' (LLMs) parameters has led to a new research direction of fine-tuning-free downstream customization by prompts, i.e., task descriptions. While these prompt-based services (e.g. OpenAI's GPTs) play an important role in many businesses, there has emerged growing concerns about the prompt leakage, which undermines the intellectual properties of these services and causes downstream attacks. In this paper, we analyze the underlying mechanism of prompt leakage, which we refer to as prompt memorization, and develop corresponding defending strategies. By exploring the scaling laws in prompt extraction, we analyze key attributes that influence prompt extraction, including model sizes, prompt lengths, as well as the types of prompts. Then we propose two hypotheses that explain how LLMs expose their prompts. The first is attributed to the perplexity, i.e. the familiarity of LLMs to texts, whereas the second is based on the straightforward token translation path in attention matrices. To defend against such threats, we investigate whether alignments can undermine the extraction of prompts. We find that current LLMs, even those with safety alignments like GPT-4, are highly vulnerable to prompt extraction attacks, even under the most straightforward user attacks. Therefore, we put forward several defense strategies with the inspiration of our findings, which achieve 83.8\% and 71.0\% drop in the prompt extraction rate for Llama2-7B and GPT-3.5, respectively. Source code is avaliable at \url{https://github.com/liangzid/PromptExtractionEval}.

摘要: 大型语言模型(LLMS)参数的急剧增加导致了一个新的研究方向，即通过提示(即任务描述)进行免微调的下游定制。虽然这些基于提示的服务(例如OpenAI的GPT)在许多业务中扮演着重要的角色，但人们越来越担心即时泄露，这会破坏这些服务的知识产权，并导致下游攻击。本文分析了即时记忆的潜在机制，并提出了相应的防御策略。通过研究提示提取中的缩放规律，我们分析了影响提示提取的关键属性，包括模型大小、提示长度以及提示的类型。然后，我们提出了两个假设来解释LLM是如何暴露他们的提示的。第一种归因于迷惑性，即LLMS对文本的熟悉度，而第二种归因于注意矩阵中直接的表征翻译路径。为了防御此类威胁，我们调查对齐是否会破坏提示符的提取。我们发现，即使在最直接的用户攻击下，当前的LLM，即使是那些具有GPT-4等安全对齐的LLM，也非常容易受到即时提取攻击。因此，我们根据研究结果提出了几种防御策略，分别使Llama2-7B和GPT-3.5的即时抽取率下降了83.8%和71.0%。源代码可在\url{https://github.com/liangzid/PromptExtractionEval}.上获得



## **34. Open Sesame! Universal Black Box Jailbreaking of Large Language Models**

芝麻开门！大型语言模型的通用黑匣子越狱 cs.CL

Accepted at SeT-LLM @ ICLR 2024

**SubmitDate**: 2024-08-05    [abs](http://arxiv.org/abs/2309.01446v4) [paper-pdf](http://arxiv.org/pdf/2309.01446v4)

**Authors**: Raz Lapid, Ron Langberg, Moshe Sipper

**Abstract**: Large language models (LLMs), designed to provide helpful and safe responses, often rely on alignment techniques to align with user intent and social guidelines. Unfortunately, this alignment can be exploited by malicious actors seeking to manipulate an LLM's outputs for unintended purposes. In this paper we introduce a novel approach that employs a genetic algorithm (GA) to manipulate LLMs when model architecture and parameters are inaccessible. The GA attack works by optimizing a universal adversarial prompt that -- when combined with a user's query -- disrupts the attacked model's alignment, resulting in unintended and potentially harmful outputs. Our novel approach systematically reveals a model's limitations and vulnerabilities by uncovering instances where its responses deviate from expected behavior. Through extensive experiments we demonstrate the efficacy of our technique, thus contributing to the ongoing discussion on responsible AI development by providing a diagnostic tool for evaluating and enhancing alignment of LLMs with human intent. To our knowledge this is the first automated universal black box jailbreak attack.

摘要: 大型语言模型(LLM)旨在提供有用和安全的响应，它们通常依靠对齐技术来与用户意图和社交指南保持一致。遗憾的是，恶意行为者可能会利用这种对齐，试图出于非预期目的操纵LLM的输出。在本文中，我们介绍了一种新的方法，即在模型结构和参数不可访问的情况下，使用遗传算法(GA)来操作LLM。GA攻击的工作原理是优化一个通用的对抗性提示，当与用户的查询结合在一起时，会扰乱被攻击模型的对齐，导致意外的和潜在的有害输出。我们的新方法通过揭示模型响应偏离预期行为的实例，系统地揭示了模型的局限性和漏洞。通过广泛的实验，我们展示了我们技术的有效性，从而通过提供一种诊断工具来评估和增强LLM与人类意图的一致性，从而为正在进行的关于负责任的人工智能开发的讨论做出贡献。据我们所知，这是第一次自动通用黑匣子越狱攻击。



## **35. A Lean Transformer Model for Dynamic Malware Analysis and Detection**

用于动态恶意软件分析和检测的精益Transformer模型 cs.CR

**SubmitDate**: 2024-08-05    [abs](http://arxiv.org/abs/2408.02313v1) [paper-pdf](http://arxiv.org/pdf/2408.02313v1)

**Authors**: Tony Quertier, Benjamin Marais, Grégoire Barrué, Stéphane Morucci, Sévan Azé, Sébastien Salladin

**Abstract**: Malware is a fast-growing threat to the modern computing world and existing lines of defense are not efficient enough to address this issue. This is mainly due to the fact that many prevention solutions rely on signature-based detection methods that can easily be circumvented by hackers. Therefore, there is a recurrent need for behavior-based analysis where a suspicious file is ran in a secured environment and its traces are collected to reports for analysis. Previous works have shown some success leveraging Neural Networks and API calls sequences extracted from these execution reports.   Recently, Large Language Models and Generative AI have demonstrated impressive capabilities mainly in Natural Language Processing tasks and promising applications in the cybersecurity field for both attackers and defenders.   In this paper, we design an Encoder-Only model, based on the Transformers architecture, to detect malicious files, digesting their API call sequences collected by an execution emulation solution. We are also limiting the size of the model architecture and the number of its parameters since it is often considered that Large Language Models may be overkill for specific tasks such as the one we are dealing with hereafter. In addition to achieving decent detection results, this approach has the advantage of reducing our carbon footprint by limiting training and inference times and facilitating technical operations with less hardware requirements.   We also carry out some analysis of our results and highlight the limits and possible improvements when using Transformers to analyze malicious files.

摘要: 恶意软件是对现代计算世界的一种快速增长的威胁，现有的防线不足以解决这个问题。这主要是因为许多预防解决方案依赖于基于签名的检测方法，而这些方法很容易被黑客绕过。因此，经常需要基于行为的分析，其中可疑文件在安全环境中运行，并将其跟踪收集到报告中进行分析。以前的工作已经表明，利用从这些执行报告中提取的神经网络和API调用序列取得了一些成功。最近，大型语言模型和产生式人工智能已经显示出令人印象深刻的能力，主要是在自然语言处理任务和网络安全领域对攻击者和防御者的应用前景。在本文中，我们设计了一个基于Transformers体系结构的仅编码器模型来检测恶意文件，消化由执行仿真解决方案收集的API调用序列。我们还限制了模型体系结构的大小及其参数的数量，因为人们通常认为大型语言模型对于特定的任务来说可能是矫枉过正的，比如我们后面要处理的任务。除了获得像样的检测结果外，这种方法的优势是通过限制培训和推理时间减少我们的碳足迹，并以更少的硬件要求促进技术操作。我们还对我们的结果进行了一些分析，并强调了使用Transformers分析恶意文件时的限制和可能的改进。



## **36. LLM Lies: Hallucinations are not Bugs, but Features as Adversarial Examples**

法学硕士谎言：幻觉不是错误，而是对抗性例子的特征 cs.CL

**SubmitDate**: 2024-08-04    [abs](http://arxiv.org/abs/2310.01469v3) [paper-pdf](http://arxiv.org/pdf/2310.01469v3)

**Authors**: Jia-Yu Yao, Kun-Peng Ning, Zhen-Hui Liu, Mu-Nan Ning, Yu-Yang Liu, Li Yuan

**Abstract**: Large Language Models (LLMs), including GPT-3.5, LLaMA, and PaLM, seem to be knowledgeable and able to adapt to many tasks. However, we still cannot completely trust their answers, since LLMs suffer from \textbf{hallucination}\textemdash fabricating non-existent facts, deceiving users with or without their awareness. However, the reasons for their existence and pervasiveness remain unclear. In this paper, we demonstrate that nonsensical prompts composed of random tokens can also elicit the LLMs to respond with hallucinations. Moreover, we provide both theoretical and experimental evidence that transformers can be manipulated to produce specific pre-define tokens by perturbing its input sequence. This phenomenon forces us to revisit that \emph{hallucination may be another view of adversarial examples}, and it shares similar characteristics with conventional adversarial examples as a basic property of LLMs. Therefore, we formalize an automatic hallucination triggering method as the \textit{hallucination attack} in an adversarial way. Finally, we explore the basic properties of attacked adversarial prompts and propose a simple yet effective defense strategy. Our code is released on GitHub\footnote{https://github.com/PKU-YuanGroup/Hallucination-Attack}.

摘要: 大型语言模型(LLM)，包括GPT-3.5、骆驼和Palm，似乎知识渊博，能够适应许多任务。然而，我们仍然不能完全信任他们的答案，因为LLMS遭受着捏造不存在的事实、欺骗用户或在他们意识不到的情况下的痛苦。然而，它们存在和普遍存在的原因尚不清楚。在这篇文章中，我们证明了由随机令牌组成的无意义提示也可以诱导LLM做出幻觉反应。此外，我们提供了理论和实验证据，证明可以通过扰动转换器的输入序列来操纵转换器来产生特定的预定义令牌。这一现象迫使我们重新审视幻觉可能是对抗性例子的另一种观点，它与传统对抗性例子具有相似的特征，是LLMS的一个基本性质。因此，我们将一种自动幻觉触发方法形式化为对抗性的幻觉攻击。最后，探讨了被攻击对抗性提示的基本性质，并提出了一种简单有效的防御策略。我们的代码在GitHub\footnote{https://github.com/PKU-YuanGroup/Hallucination-Attack}.上发布



## **37. Towards Automatic Hands-on-Keyboard Attack Detection Using LLMs in EDR Solutions**

在EDR解决方案中使用LLM实现自动键盘操作攻击检测 cs.CR

**SubmitDate**: 2024-08-04    [abs](http://arxiv.org/abs/2408.01993v1) [paper-pdf](http://arxiv.org/pdf/2408.01993v1)

**Authors**: Amit Portnoy, Ehud Azikri, Shay Kels

**Abstract**: Endpoint Detection and Remediation (EDR) platforms are essential for identifying and responding to cyber threats. This study presents a novel approach using Large Language Models (LLMs) to detect Hands-on-Keyboard (HOK) cyberattacks. Our method involves converting endpoint activity data into narrative forms that LLMs can analyze to distinguish between normal operations and potential HOK attacks. We address the challenges of interpreting endpoint data by segmenting narratives into windows and employing a dual training strategy. The results demonstrate that LLM-based models have the potential to outperform traditional machine learning methods, offering a promising direction for enhancing EDR capabilities and apply LLMs in cybersecurity.

摘要: 端点检测和修复（EDR）平台对于识别和响应网络威胁至关重要。这项研究提出了一种使用大型语言模型（LLM）来检测键盘手控（HOK）网络攻击的新颖方法。我们的方法涉及将端点活动数据转换为叙述形式，LLM可以分析这些形式以区分正常操作和潜在的HOK攻击。我们通过将叙述分割到窗口并采用双重训练策略来解决解释端点数据的挑战。结果表明，基于LLM的模型有潜力超越传统的机器学习方法，为增强EDR能力和在网络安全中应用LLM提供了一个有前途的方向。



## **38. InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents**

InjectAgent：在工具集成大型语言模型代理中对间接提示注入进行基准测试 cs.CL

36 pages, 6 figures, 13 tables (ACL 2024 Findings)

**SubmitDate**: 2024-08-04    [abs](http://arxiv.org/abs/2403.02691v3) [paper-pdf](http://arxiv.org/pdf/2403.02691v3)

**Authors**: Qiusi Zhan, Zhixiang Liang, Zifan Ying, Daniel Kang

**Abstract**: Recent work has embodied LLMs as agents, allowing them to access tools, perform actions, and interact with external content (e.g., emails or websites). However, external content introduces the risk of indirect prompt injection (IPI) attacks, where malicious instructions are embedded within the content processed by LLMs, aiming to manipulate these agents into executing detrimental actions against users. Given the potentially severe consequences of such attacks, establishing benchmarks to assess and mitigate these risks is imperative.   In this work, we introduce InjecAgent, a benchmark designed to assess the vulnerability of tool-integrated LLM agents to IPI attacks. InjecAgent comprises 1,054 test cases covering 17 different user tools and 62 attacker tools. We categorize attack intentions into two primary types: direct harm to users and exfiltration of private data. We evaluate 30 different LLM agents and show that agents are vulnerable to IPI attacks, with ReAct-prompted GPT-4 vulnerable to attacks 24% of the time. Further investigation into an enhanced setting, where the attacker instructions are reinforced with a hacking prompt, shows additional increases in success rates, nearly doubling the attack success rate on the ReAct-prompted GPT-4. Our findings raise questions about the widespread deployment of LLM Agents. Our benchmark is available at https://github.com/uiuc-kang-lab/InjecAgent.

摘要: 最近的工作将LLMS体现为代理，允许它们访问工具、执行操作并与外部内容(例如，电子邮件或网站)交互。然而，外部内容会带来间接提示注入(IPI)攻击的风险，在IPI攻击中，恶意指令被嵌入到LLMS处理的内容中，目的是操纵这些代理执行针对用户的有害操作。鉴于此类攻击的潜在严重后果，建立评估和减轻这些风险的基准势在必行。在这项工作中，我们引入了InjecAgent，这是一个旨在评估工具集成的LLM代理对IPI攻击的脆弱性的基准测试。InjecAgent由1,054个测试用例组成，涵盖17个不同的用户工具和62个攻击者工具。我们将攻击意图分为两种主要类型：直接伤害用户和泄露私人数据。我们评估了30种不同的LLM代理，表明代理容易受到IPI攻击，其中反应提示的GPT-4在24%的时间内容易受到攻击。对增强设置的进一步调查显示，成功率进一步提高，反应提示GPT-4的攻击成功率几乎翻了一番。在增强设置中，攻击者的指令通过黑客提示得到加强。我们的发现对LLM特工的广泛部署提出了质疑。我们的基准测试可从https://github.com/uiuc-kang-lab/InjecAgent.获得



## **39. AttackEval: How to Evaluate the Effectiveness of Jailbreak Attacking on Large Language Models**

AttackEval：如何评估越狱攻击对大型语言模型的有效性 cs.CL

34 pages, 6 figures

**SubmitDate**: 2024-08-03    [abs](http://arxiv.org/abs/2401.09002v5) [paper-pdf](http://arxiv.org/pdf/2401.09002v5)

**Authors**: Dong shu, Mingyu Jin, Chong Zhang, Liangyao Li, Zihao Zhou, Yongfeng Zhang

**Abstract**: Ensuring the security of large language models (LLMs) against attacks has become increasingly urgent, with jailbreak attacks representing one of the most sophisticated threats. To deal with such risks, we introduce an innovative framework that can help evaluate the effectiveness of jailbreak attacks on LLMs. Unlike traditional binary evaluations focusing solely on the robustness of LLMs, our method assesses the effectiveness of the attacking prompts themselves. We present two distinct evaluation frameworks: a coarse-grained evaluation and a fine-grained evaluation. Each framework uses a scoring range from 0 to 1, offering unique perspectives and allowing for the assessment of attack effectiveness in different scenarios. Additionally, we develop a comprehensive ground truth dataset specifically tailored for jailbreak prompts. This dataset serves as a crucial benchmark for our current study and provides a foundational resource for future research. By comparing with traditional evaluation methods, our study shows that the current results align with baseline metrics while offering a more nuanced and fine-grained assessment. It also helps identify potentially harmful attack prompts that might appear harmless in traditional evaluations. Overall, our work establishes a solid foundation for assessing a broader range of attack prompts in the area of prompt injection.

摘要: 确保大型语言模型(LLM)免受攻击的安全性已变得越来越紧迫，越狱攻击是最复杂的威胁之一。为了应对这样的风险，我们引入了一个创新的框架，可以帮助评估越狱攻击对低收入者的有效性。与传统的只关注LLMS健壮性的二进制评估不同，我们的方法评估攻击提示本身的有效性。我们提出了两种不同的评估框架：粗粒度评估和细粒度评估。每个框架使用从0到1的评分范围，提供独特的视角，并允许在不同情况下评估攻击效果。此外，我们还开发了专门为越狱提示量身定做的全面地面事实数据集。这一数据集是我们当前研究的重要基准，并为未来的研究提供了基础资源。通过与传统评估方法的比较，我们的研究表明，当前的结果与基线度量一致，同时提供了更细微和细粒度的评估。它还有助于识别在传统评估中可能看起来无害的潜在有害攻击提示。总体而言，我们的工作为在快速注射领域评估更广泛的攻击提示奠定了坚实的基础。



## **40. MCGMark: An Encodable and Robust Online Watermark for LLM-Generated Malicious Code**

MCGMark：针对LLM生成的恶意代码的可编码且稳健的在线水印 cs.CR

**SubmitDate**: 2024-08-02    [abs](http://arxiv.org/abs/2408.01354v1) [paper-pdf](http://arxiv.org/pdf/2408.01354v1)

**Authors**: Kaiwen Ning, Jiachi Chen, Qingyuan Zhong, Tao Zhang, Yanlin Wang, Wei Li, Yu Zhang, Weizhe Zhang, Zibin Zheng

**Abstract**: With the advent of large language models (LLMs), numerous software service providers (SSPs) are dedicated to developing LLMs customized for code generation tasks, such as CodeLlama and Copilot. However, these LLMs can be leveraged by attackers to create malicious software, which may pose potential threats to the software ecosystem. For example, they can automate the creation of advanced phishing malware. To address this issue, we first conduct an empirical study and design a prompt dataset, MCGTest, which involves approximately 400 person-hours of work and consists of 406 malicious code generation tasks. Utilizing this dataset, we propose MCGMark, the first robust, code structure-aware, and encodable watermarking approach to trace LLM-generated code. We embed encodable information by controlling the token selection and ensuring the output quality based on probabilistic outliers. Additionally, we enhance the robustness of the watermark by considering the structural features of malicious code, preventing the embedding of the watermark in easily modified positions, such as comments. We validate the effectiveness and robustness of MCGMark on the DeepSeek-Coder. MCGMark achieves an embedding success rate of 88.9% within a maximum output limit of 400 tokens. Furthermore, it also demonstrates strong robustness and has minimal impact on the quality of the output code. Our approach assists SSPs in tracing and holding responsible parties accountable for malicious code generated by LLMs.

摘要: 随着大型语言模型(LLM)的出现，许多软件服务提供商(SSP)都致力于开发为代码生成任务定制的LLM，例如CodeLlama和Copilot。然而，攻击者可以利用这些LLM来创建恶意软件，这可能会对软件生态系统构成潜在威胁。例如，他们可以自动创建高级网络钓鱼恶意软件。为了解决这个问题，我们首先进行了实证研究，并设计了一个即时数据集MCGTest，它涉及大约400个人/小时的工作，包括406个恶意代码生成任务。利用这个数据集，我们提出了第一个健壮的、代码结构感知的、可编码的跟踪LLM生成代码的水印方法MCGMark。我们通过控制令牌选择和基于概率离群点来确保输出质量来嵌入可编码信息。此外，通过考虑恶意代码的结构特征，防止水印嵌入在评论等易修改的位置，增强了水印的稳健性。在DeepSeek-Coder上验证了MCGMark的有效性和健壮性。MCGMark在400个令牌的最大输出限制内实现了88.9%的嵌入成功率。此外，它还表现出很强的健壮性，并且对输出代码的质量影响最小。我们的方法帮助SSP跟踪并追究由LLMS生成的恶意代码的责任方的责任。



## **41. Chat AI: A Seamless Slurm-Native Solution for HPC-Based Services**

Chat AI：针对基于HP的服务的无缝SlurmNative解决方案 cs.DC

Various improvements to explanations and form and updated graphs to  include data points up to 30.07.2024

**SubmitDate**: 2024-08-02    [abs](http://arxiv.org/abs/2407.00110v2) [paper-pdf](http://arxiv.org/pdf/2407.00110v2)

**Authors**: Ali Doosthosseini, Jonathan Decker, Hendrik Nolte, Julian M. Kunkel

**Abstract**: The widespread adoption of large language models (LLMs) has created a pressing need for an efficient, secure and private serving infrastructure, which allows researchers to run open source or custom fine-tuned LLMs and ensures users that their data remains private and is not stored without their consent. While high-performance computing (HPC) systems equipped with state-of-the-art GPUs are well-suited for training LLMs, their batch scheduling paradigm is not designed to support real-time serving of AI applications. Cloud systems, on the other hand, are well suited for web services but commonly lack access to the computational power of HPC clusters, especially expensive and scarce high-end GPUs, which are required for optimal inference speed. We propose an architecture with an implementation consisting of a web service that runs on a cloud VM with secure access to a scalable backend running a multitude of LLM models on HPC systems. By offering a web service using our HPC infrastructure to host LLMs, we leverage the trusted environment of local universities and research centers to offer a private and secure alternative to commercial LLM services. Our solution natively integrates with the HPC batch scheduler Slurm, enabling seamless deployment on HPC clusters, and is able to run side by side with regular Slurm workloads, while utilizing gaps in the schedule created by Slurm. In order to ensure the security of the HPC system, we use the SSH ForceCommand directive to construct a robust circuit breaker, which prevents successful attacks on the web-facing server from affecting the cluster. We have successfully deployed our system as a production service, and made the source code available at \url{https://github.com/gwdg/chat-ai}

摘要: 大型语言模型(LLM)的广泛采用产生了对高效、安全和私有的服务基础设施的迫切需求，该基础设施允许研究人员运行开源或定制的微调LLM，并确保用户的数据保持隐私，并且不会在未经他们同意的情况下存储。虽然配备最先进的GPU的高性能计算(HPC)系统非常适合训练LLM，但它们的批处理调度范例并不是为支持AI应用的实时服务而设计的。另一方面，云系统非常适合Web服务，但通常无法使用HPC集群的计算能力，特别是昂贵而稀缺的高端GPU，这是实现最佳推理速度所必需的。我们提出了一种架构，其实施包括在云VM上运行的Web服务，可以安全地访问在HPC系统上运行多个LLM模型的可扩展后端。通过提供使用我们的HPC基础设施来托管LLM的Web服务，我们利用当地大学和研究中心的可信环境来提供商业LLM服务的私有且安全的替代方案。我们的解决方案与HPC批处理调度程序SLurm进行了本机集成，实现了在HPC群集上的无缝部署，并且能够与常规的SLurm工作负载并行运行，同时利用SLurm创建的调度缺口。为了确保HPC系统的安全，我们使用SSH ForceCommand指令来构建一个健壮的断路器，以防止对面向Web的服务器的成功攻击影响到集群。我们已成功将系统部署为生产服务，并在\url{https://github.com/gwdg/chat-ai}}上提供了源代码



## **42. A Survey of Text Watermarking in the Era of Large Language Models**

大语言模型时代文本水印综述 cs.CL

35 pages, 11 figures, 2 tables

**SubmitDate**: 2024-08-02    [abs](http://arxiv.org/abs/2312.07913v6) [paper-pdf](http://arxiv.org/pdf/2312.07913v6)

**Authors**: Aiwei Liu, Leyi Pan, Yijian Lu, Jingjing Li, Xuming Hu, Xi Zhang, Lijie Wen, Irwin King, Hui Xiong, Philip S. Yu

**Abstract**: Text watermarking algorithms are crucial for protecting the copyright of textual content. Historically, their capabilities and application scenarios were limited. However, recent advancements in large language models (LLMs) have revolutionized these techniques. LLMs not only enhance text watermarking algorithms with their advanced abilities but also create a need for employing these algorithms to protect their own copyrights or prevent potential misuse. This paper conducts a comprehensive survey of the current state of text watermarking technology, covering four main aspects: (1) an overview and comparison of different text watermarking techniques; (2) evaluation methods for text watermarking algorithms, including their detectability, impact on text or LLM quality, robustness under target or untargeted attacks; (3) potential application scenarios for text watermarking technology; (4) current challenges and future directions for text watermarking. This survey aims to provide researchers with a thorough understanding of text watermarking technology in the era of LLM, thereby promoting its further advancement.

摘要: 文本水印算法对于保护文本内容的版权至关重要。从历史上看，它们的能力和应用场景都是有限的。然而，最近大型语言模型(LLM)的进步使这些技术发生了革命性的变化。LLMS不仅以其先进的能力增强了文本水印算法，而且产生了使用这些算法来保护自己的版权或防止潜在的误用的需求。本文对文本水印技术的现状进行了全面的综述，主要包括四个方面：(1)不同文本水印技术的概述和比较；(2)文本水印算法的评价方法，包括它们的可检测性、对文本或LLM质量的影响、对目标攻击和非目标攻击的稳健性；(3)文本水印技术的潜在应用场景；(4)文本水印面临的挑战和未来的发展方向。这项调查旨在让研究人员对LLM时代的文本水印技术有一个透彻的了解，从而推动其进一步发展。



## **43. SLIP: Securing LLMs IP Using Weights Decomposition**

SIP：使用权重分解保护LLM IP cs.CR

**SubmitDate**: 2024-08-01    [abs](http://arxiv.org/abs/2407.10886v2) [paper-pdf](http://arxiv.org/pdf/2407.10886v2)

**Authors**: Yehonathan Refael, Adam Hakim, Lev Greenberg, Tal Aviv, Satya Lokam, Ben Fishman, Shachar Seidman

**Abstract**: Large language models (LLMs) have recently seen widespread adoption, in both academia and industry. As these models grow, they become valuable intellectual property (IP), reflecting enormous investments by their owners. Moreover, the high cost of cloud-based deployment has driven interest towards deployment to edge devices, yet this risks exposing valuable parameters to theft and unauthorized use. Current methods to protect models' IP on the edge have limitations in terms of practicality, loss in accuracy, or suitability to requirements. In this paper, we introduce a novel hybrid inference algorithm, named SLIP, designed to protect edge-deployed models from theft. SLIP is the first hybrid protocol that is both practical for real-world applications and provably secure, while having zero accuracy degradation and minimal impact on latency. It involves partitioning the model between two computing resources, one secure but expensive, and another cost-effective but vulnerable. This is achieved through matrix decomposition, ensuring that the secure resource retains a maximally sensitive portion of the model's IP while performing a minimal amount of computations, and vice versa for the vulnerable resource. Importantly, the protocol includes security guarantees that prevent attackers from exploiting the partition to infer the secured information. Finally, we present experimental results that show the robustness and effectiveness of our method, positioning it as a compelling solution for protecting LLMs.

摘要: 大型语言模型(LLM)最近在学术界和工业界都得到了广泛采用。随着这些模式的发展，它们成为有价值的知识产权(IP)，反映了其所有者的巨额投资。此外，基于云的部署的高昂成本推动了对部署到边缘设备的兴趣，但这可能会使宝贵的参数面临被盗和未经授权使用的风险。目前保护模型边缘知识产权的方法在实用性、精确度损失或对要求的适应性方面都存在局限性。在本文中，我们提出了一种新的混合推理算法，称为SLIP，旨在防止边部署的模型被盗。SLIP是第一个混合协议，它既适用于现实世界的应用，又可证明是安全的，同时具有零精度降级和对延迟的最小影响。它涉及在两种计算资源之间划分模型，一种是安全但昂贵的计算资源，另一种是经济高效但脆弱的计算资源。这是通过矩阵分解实现的，确保安全资源保留模型IP的最敏感部分，同时执行最少量的计算，反之亦然。重要的是，该协议包括安全保证，以防止攻击者利用分区来推断安全信息。最后，我们给出了实验结果，证明了该方法的健壮性和有效性，将其定位为一种引人注目的保护LLM的解决方案。



## **44. Prover-Verifier Games improve legibility of LLM outputs**

证明者-验证者游戏提高了LLM输出的清晰度 cs.CL

**SubmitDate**: 2024-08-01    [abs](http://arxiv.org/abs/2407.13692v2) [paper-pdf](http://arxiv.org/pdf/2407.13692v2)

**Authors**: Jan Hendrik Kirchner, Yining Chen, Harri Edwards, Jan Leike, Nat McAleese, Yuri Burda

**Abstract**: One way to increase confidence in the outputs of Large Language Models (LLMs) is to support them with reasoning that is clear and easy to check -- a property we call legibility. We study legibility in the context of solving grade-school math problems and show that optimizing chain-of-thought solutions only for answer correctness can make them less legible. To mitigate the loss in legibility, we propose a training algorithm inspired by Prover-Verifier Game from Anil et al. (2021). Our algorithm iteratively trains small verifiers to predict solution correctness, "helpful" provers to produce correct solutions that the verifier accepts, and "sneaky" provers to produce incorrect solutions that fool the verifier. We find that the helpful prover's accuracy and the verifier's robustness to adversarial attacks increase over the course of training. Furthermore, we show that legibility training transfers to time-constrained humans tasked with verifying solution correctness. Over course of LLM training human accuracy increases when checking the helpful prover's solutions, and decreases when checking the sneaky prover's solutions. Hence, training for checkability by small verifiers is a plausible technique for increasing output legibility. Our results suggest legibility training against small verifiers as a practical avenue for increasing legibility of large LLMs to humans, and thus could help with alignment of superhuman models.

摘要: 增加对大型语言模型(LLM)输出的信心的一种方法是用清晰且易于检查的推理来支持它们--我们称之为易读性。我们在解决小学数学问题的背景下研究了易读性，并表明只为了答案的正确性而优化思维链解决方案会降低它们的易读性。为了减少易读性的损失，我们提出了一种受Anil等人的Prover-Verator游戏启发的训练算法。(2021年)。我们的算法迭代地训练小的验证者来预测解决方案的正确性，“有帮助的”验证者来产生验证者接受的正确的解决方案，而“偷偷摸摸”的验证者产生愚弄验证者的不正确的解决方案。我们发现，随着训练过程的进行，有益证明者的准确率和验证者对敌意攻击的健壮性都有所提高。此外，我们还表明，易读性训练转移到负责验证解决方案正确性的时间受限的人身上。在LLM训练过程中，当检查有用的证明者的解时，人类的准确率提高，而当检查偷偷摸摸的证明者的解时，人类的准确率降低。因此，由小型验证员进行可校验性培训是提高输出清晰度的一种可行的技术。我们的结果表明，针对小验证者的易读性训练是提高大型LLM对人类易读性的实用途径，因此可能有助于超人模型的对齐。



## **45. Pathway to Secure and Trustworthy 6G for LLMs: Attacks, Defense, and Opportunities**

LLM安全且值得信赖的6G之路：攻击、防御和机会 cs.CR

7 pages, 4 figures

**SubmitDate**: 2024-08-01    [abs](http://arxiv.org/abs/2408.00722v1) [paper-pdf](http://arxiv.org/pdf/2408.00722v1)

**Authors**: Sunder Ali Khowaja, Parus Khuwaja, Kapal Dev, Hussam Al Hamadi, Engin Zeydan

**Abstract**: Recently, large language models (LLMs) have been gaining a lot of interest due to their adaptability and extensibility in emerging applications, including communication networks. It is anticipated that 6G mobile edge computing networks will be able to support LLMs as a service, as they provide ultra reliable low-latency communications and closed loop massive connectivity. However, LLMs are vulnerable to data and model privacy issues that affect the trustworthiness of LLMs to be deployed for user-based services. In this paper, we explore the security vulnerabilities associated with fine-tuning LLMs in 6G networks, in particular the membership inference attack. We define the characteristics of an attack network that can perform a membership inference attack if the attacker has access to the fine-tuned model for the downstream task. We show that the membership inference attacks are effective for any downstream task, which can lead to a personal data breach when using LLM as a service. The experimental results show that the attack success rate of maximum 92% can be achieved on named entity recognition task. Based on the experimental analysis, we discuss possible defense mechanisms and present possible research directions to make the LLMs more trustworthy in the context of 6G networks.

摘要: 最近，大型语言模型(LLM)因其在包括通信网络在内的新兴应用中的适应性和可扩展性而受到人们的极大关注。预计6G移动边缘计算网络将能够支持LLMS作为一种服务，因为它们提供超可靠的低延迟通信和闭环海量连接。然而，LLM容易受到数据和模型隐私问题的影响，这些问题会影响要为基于用户的服务部署的LLM的可信度。本文探讨了6G网络中与微调LLMS相关的安全漏洞，特别是成员推理攻击。我们定义了攻击网络的特征，如果攻击者有权访问下游任务的微调模型，则该攻击网络可以执行成员关系推理攻击。我们证明了成员关系推断攻击对于任何下游任务都是有效的，当使用LLM作为服务时，这可能导致个人数据泄露。实验结果表明，命名实体识别任务的攻击成功率最高可达92%。在实验分析的基础上，讨论了可能的防御机制，并提出了可能的研究方向，以使6G网络环境下的LLMS更具可信性。



## **46. Jailbreaking Text-to-Image Models with LLM-Based Agents**

使用基于LLM的代理破解文本到图像模型 cs.CR

**SubmitDate**: 2024-08-01    [abs](http://arxiv.org/abs/2408.00523v1) [paper-pdf](http://arxiv.org/pdf/2408.00523v1)

**Authors**: Yingkai Dong, Zheng Li, Xiangtao Meng, Ning Yu, Shanqing Guo

**Abstract**: Recent advancements have significantly improved automated task-solving capabilities using autonomous agents powered by large language models (LLMs). However, most LLM-based agents focus on dialogue, programming, or specialized domains, leaving gaps in addressing generative AI safety tasks. These gaps are primarily due to the challenges posed by LLM hallucinations and the lack of clear guidelines. In this paper, we propose Atlas, an advanced LLM-based multi-agent framework that integrates an efficient fuzzing workflow to target generative AI models, specifically focusing on jailbreak attacks against text-to-image (T2I) models with safety filters. Atlas utilizes a vision-language model (VLM) to assess whether a prompt triggers the T2I model's safety filter. It then iteratively collaborates with both LLM and VLM to generate an alternative prompt that bypasses the filter. Atlas also enhances the reasoning abilities of LLMs in attack scenarios by leveraging multi-agent communication, in-context learning (ICL) memory mechanisms, and the chain-of-thought (COT) approach. Our evaluation demonstrates that Atlas successfully jailbreaks several state-of-the-art T2I models in a black-box setting, which are equipped with multi-modal safety filters. In addition, Atlas outperforms existing methods in both query efficiency and the quality of the generated images.

摘要: 最近的进步显著提高了使用大型语言模型(LLM)支持的自主代理的自动任务求解能力。然而，大多数基于LLM的代理专注于对话、编程或专业领域，在处理生成性AI安全任务方面留下了空白。这些差距主要是由于LLM幻觉带来的挑战，以及缺乏明确的指导方针。在本文中，我们提出了Atlas，一个先进的基于LLM的多代理框架，它集成了一个高效的模糊工作流来针对生成性AI模型，特别是针对带有安全过滤器的文本到图像(T2I)模型的越狱攻击。Atlas使用视觉语言模型(VLM)来评估提示是否触发了T2I模型的安全过滤器。然后，它与LLM和VLM迭代协作，以生成绕过过滤器的替代提示。Atlas还通过利用多代理通信、上下文学习(ICL)记忆机制和思想链(COT)方法来增强LLMS在攻击场景中的推理能力。我们的评估表明，Atlas成功地在黑匣子环境中越狱了几款最先进的T2I车型，这些车型配备了多模式安全过滤器。此外，Atlas在查询效率和生成图像的质量方面都优于现有方法。



## **47. Autonomous LLM-Enhanced Adversarial Attack for Text-to-Motion**

针对文本到运动的自主LLM增强对抗攻击 cs.CV

**SubmitDate**: 2024-08-01    [abs](http://arxiv.org/abs/2408.00352v1) [paper-pdf](http://arxiv.org/pdf/2408.00352v1)

**Authors**: Honglei Miao, Fan Ma, Ruijie Quan, Kun Zhan, Yi Yang

**Abstract**: Human motion generation driven by deep generative models has enabled compelling applications, but the ability of text-to-motion (T2M) models to produce realistic motions from text prompts raises security concerns if exploited maliciously. Despite growing interest in T2M, few methods focus on safeguarding these models against adversarial attacks, with existing work on text-to-image models proving insufficient for the unique motion domain. In the paper, we propose ALERT-Motion, an autonomous framework leveraging large language models (LLMs) to craft targeted adversarial attacks against black-box T2M models. Unlike prior methods modifying prompts through predefined rules, ALERT-Motion uses LLMs' knowledge of human motion to autonomously generate subtle yet powerful adversarial text descriptions. It comprises two key modules: an adaptive dispatching module that constructs an LLM-based agent to iteratively refine and search for adversarial prompts; and a multimodal information contrastive module that extracts semantically relevant motion information to guide the agent's search. Through this LLM-driven approach, ALERT-Motion crafts adversarial prompts querying victim models to produce outputs closely matching targeted motions, while avoiding obvious perturbations. Evaluations across popular T2M models demonstrate ALERT-Motion's superiority over previous methods, achieving higher attack success rates with stealthier adversarial prompts. This pioneering work on T2M adversarial attacks highlights the urgency of developing defensive measures as motion generation technology advances, urging further research into safe and responsible deployment.

摘要: 由深度生成模型驱动的人类运动生成已经实现了令人信服的应用，但文本到运动(T2M)模型从文本提示生成逼真运动的能力如果被恶意利用，会引发安全问题。尽管对T2M的兴趣与日俱增，但很少有方法专注于保护这些模型免受对手攻击，现有的文本到图像模型的工作被证明不足以满足独特的运动域。在本文中，我们提出了ALERT-Motion，这是一个利用大型语言模型(LLM)来针对黑盒T2M模型进行有针对性的对抗性攻击的自主框架。与以前通过预定义规则修改提示的方法不同，ALERT-Motion使用LLMS对人体运动的知识来自主生成微妙但强大的对抗性文本描述。它包括两个关键模块：自适应调度模块和多通道信息对比模块，自适应调度模块构建了一个基于LLM的代理，用于迭代地提炼和搜索对手提示；多通道信息对比模块提取语义相关的运动信息来指导代理的搜索。通过这种LLM驱动的方法，ALERT-Motion恶意提示查询受害者模型以产生与目标运动紧密匹配的输出，同时避免明显的扰动。对流行的T2M模型的评估表明，Alert-Motion比以前的方法更具优势，通过更隐蔽的对手提示实现了更高的攻击成功率。这项关于T2M对抗性攻击的开创性工作突显了随着动作生成技术的进步而开发防御措施的紧迫性，促使对安全和负责任的部署进行进一步研究。



## **48. Can Editing LLMs Inject Harm?**

编辑LLM会造成伤害吗？ cs.CL

The first two authors contributed equally. 9 pages for main paper, 36  pages including appendix. The code, results, dataset for this paper and more  resources are on the project website: https://llm-editing.github.io

**SubmitDate**: 2024-07-31    [abs](http://arxiv.org/abs/2407.20224v2) [paper-pdf](http://arxiv.org/pdf/2407.20224v2)

**Authors**: Canyu Chen, Baixiang Huang, Zekun Li, Zhaorun Chen, Shiyang Lai, Xiongxiao Xu, Jia-Chen Gu, Jindong Gu, Huaxiu Yao, Chaowei Xiao, Xifeng Yan, William Yang Wang, Philip Torr, Dawn Song, Kai Shu

**Abstract**: Knowledge editing techniques have been increasingly adopted to efficiently correct the false or outdated knowledge in Large Language Models (LLMs), due to the high cost of retraining from scratch. Meanwhile, one critical but under-explored question is: can knowledge editing be used to inject harm into LLMs? In this paper, we propose to reformulate knowledge editing as a new type of safety threat for LLMs, namely Editing Attack, and conduct a systematic investigation with a newly constructed dataset EditAttack. Specifically, we focus on two typical safety risks of Editing Attack including Misinformation Injection and Bias Injection. For the risk of misinformation injection, we first categorize it into commonsense misinformation injection and long-tail misinformation injection. Then, we find that editing attacks can inject both types of misinformation into LLMs, and the effectiveness is particularly high for commonsense misinformation injection. For the risk of bias injection, we discover that not only can biased sentences be injected into LLMs with high effectiveness, but also one single biased sentence injection can cause a bias increase in general outputs of LLMs, which are even highly irrelevant to the injected sentence, indicating a catastrophic impact on the overall fairness of LLMs. Then, we further illustrate the high stealthiness of editing attacks, measured by their impact on the general knowledge and reasoning capacities of LLMs, and show the hardness of defending editing attacks with empirical evidence. Our discoveries demonstrate the emerging misuse risks of knowledge editing techniques on compromising the safety alignment of LLMs.

摘要: 由于从头开始再培训的成本很高，知识编辑技术越来越多地被用来有效地纠正大型语言模型(LLMS)中的错误或过时知识。与此同时，一个关键但未被探讨的问题是：知识编辑能否被用来向低收入国家注入危害？在本文中，我们将知识编辑重新定义为一种新的安全威胁，即编辑攻击，并使用新构建的数据集EditAttack进行了系统的研究。具体地说，我们重点研究了编辑攻击的两个典型的安全风险，包括错误信息注入和偏见注入。对于错误信息注入的风险，我们首先将其分为常识性错误信息注入和长尾错误信息注入。然后，我们发现编辑攻击可以将这两种类型的错误信息注入到LLMS中，其中常识性错误信息注入的有效性尤其高。对于偏向注入的风险，我们发现，偏向句不仅可以被高效地注入到LLMS中，而且一次偏向句注入会导致LLMS的总体输出出现偏向增加，甚至与注入的句子高度无关，这对LLMS的整体公平性造成了灾难性的影响。然后，我们进一步说明了编辑攻击的高度隐蔽性，通过它们对LLM的常识和推理能力的影响来衡量它们，并用经验证据说明了防御编辑攻击的难度。我们的发现表明，知识编辑技术在损害LLMS的安全一致性方面存在新的误用风险。



## **49. Figure it Out: Analyzing-based Jailbreak Attack on Large Language Models**

弄清楚：基于分析的对大型语言模型的越狱攻击 cs.CR

**SubmitDate**: 2024-07-31    [abs](http://arxiv.org/abs/2407.16205v2) [paper-pdf](http://arxiv.org/pdf/2407.16205v2)

**Authors**: Shi Lin, Rongchang Li, Xun Wang, Changting Lin, Wenpeng Xing, Meng Han

**Abstract**: The rapid development of Large Language Models (LLMs) has brought remarkable generative capabilities across diverse tasks. However, despite the impressive achievements, these models still have numerous security vulnerabilities, particularly when faced with jailbreak attacks. Therefore, by investigating jailbreak attacks, we can uncover hidden weaknesses in LLMs and guide us in developing more robust defense mechanisms to fortify their security. In this paper, we further explore the boundary of jailbreak attacks on LLMs and propose Analyzing-based Jailbreak (ABJ). This effective jailbreak attack method takes advantage of LLMs' growing analyzing and reasoning capability and reveals their underlying vulnerabilities when facing analysis-based tasks. We conduct a detailed evaluation of ABJ across various open-source and closed-source LLMs, which achieves 94.8% Attack Success Rate (ASR) and 1.06 Attack Efficiency (AE) on GPT-4-turbo-0409, demonstrating state-of-the-art attack effectiveness and efficiency. Our research highlights the importance of prioritizing and enhancing the safety of LLMs to mitigate the risks of misuse.The code is publicly available at https://github.com/theshi-1128/ABJ-Attack.

摘要: 大型语言模型(LLM)的快速发展带来了跨越各种任务的非凡的生成能力。然而，尽管取得了令人印象深刻的成就，这些模型仍然存在许多安全漏洞，特别是在面临越狱攻击时。因此，通过调查越狱攻击，我们可以发现LLMS中隐藏的弱点，并指导我们开发更强大的防御机制来加强它们的安全。本文进一步探讨了LLMS越狱攻击的边界，提出了基于分析的越狱攻击(ABJ)。这种有效的越狱攻击方法利用了LLMS日益增长的分析和推理能力，并在面对基于分析的任务时揭示了它们潜在的漏洞。我们对ABJ在各种开源和闭源LLMS上进行了详细的评估，在GPT-4-TURBO-0409上达到了94.8%的攻击成功率(ASR)和1.06的攻击效率(AE)，展示了最先进的攻击效果和效率。我们的研究强调了优先处理和增强低成本管理系统安全性的重要性，以减少滥用风险。代码可在https://github.com/theshi-1128/ABJ-Attack.上公开获得



## **50. TAROT: Task-Oriented Authorship Obfuscation Using Policy Optimization Methods**

TAROT：使用策略优化方法的面向任务的作者混淆 cs.CL

**SubmitDate**: 2024-07-31    [abs](http://arxiv.org/abs/2407.21630v1) [paper-pdf](http://arxiv.org/pdf/2407.21630v1)

**Authors**: Gabriel Loiseau, Damien Sileo, Damien Riquet, Maxime Meyer, Marc Tommasi

**Abstract**: Authorship obfuscation aims to disguise the identity of an author within a text by altering the writing style, vocabulary, syntax, and other linguistic features associated with the text author. This alteration needs to balance privacy and utility. While strong obfuscation techniques can effectively hide the author's identity, they often degrade the quality and usefulness of the text for its intended purpose. Conversely, maintaining high utility tends to provide insufficient privacy, making it easier for an adversary to de-anonymize the author. Thus, achieving an optimal trade-off between these two conflicting objectives is crucial. In this paper, we propose TAROT: Task-Oriented Authorship Obfuscation Using Policy Optimization, a new unsupervised authorship obfuscation method whose goal is to optimize the privacy-utility trade-off by regenerating the entire text considering its downstream utility. Our approach leverages policy optimization as a fine-tuning paradigm over small language models in order to rewrite texts by preserving author identity and downstream task utility. We show that our approach largely reduce the accuracy of attackers while preserving utility. We make our code and models publicly available.

摘要: 作者身份混淆旨在通过改变与文本作者相关的写作风格、词汇、句法和其他语言特征来掩盖作者在文本中的身份。这一改变需要平衡隐私和效用。虽然强大的混淆技术可以有效地隐藏作者的身份，但它们往往会降低文本的质量和对预期目的的有用性。相反，保持高实用性往往会提供不充分的隐私，使对手更容易解除作者的匿名。因此，在这两个相互冲突的目标之间实现最佳权衡至关重要。在本文中，我们提出了一种新的无监督作者身份混淆方法--TAROT：基于策略优化的面向任务的作者身份混淆方法，其目标是通过重新生成考虑下游效用的整个文本来优化隐私和效用之间的权衡。我们的方法利用策略优化作为小语言模型上的微调范式，以便通过保留作者身份和下游任务效用来重写文本。我们表明，我们的方法在很大程度上降低了攻击者的准确性，同时保持了实用性。我们公开我们的代码和模型。



