# Latest Large Language Model Attack Papers
**update at 2024-09-18 09:34:59**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Prompt Obfuscation for Large Language Models**

大型语言模型的提示混淆 cs.CR

**SubmitDate**: 2024-09-17    [abs](http://arxiv.org/abs/2409.11026v1) [paper-pdf](http://arxiv.org/pdf/2409.11026v1)

**Authors**: David Pape, Thorsten Eisenhofer, Lea Schönherr

**Abstract**: System prompts that include detailed instructions to describe the task performed by the underlying large language model (LLM) can easily transform foundation models into tools and services with minimal overhead. Because of their crucial impact on the utility, they are often considered intellectual property, similar to the code of a software product. However, extracting system prompts is easily possible by using prompt injection. As of today, there is no effective countermeasure to prevent the stealing of system prompts and all safeguarding efforts could be evaded with carefully crafted prompt injections that bypass all protection mechanisms.In this work, we propose an alternative to conventional system prompts. We introduce prompt obfuscation to prevent the extraction of the system prompt while maintaining the utility of the system itself with only little overhead. The core idea is to find a representation of the original system prompt that leads to the same functionality, while the obfuscated system prompt does not contain any information that allows conclusions to be drawn about the original system prompt. We implement an optimization-based method to find an obfuscated prompt representation while maintaining the functionality. To evaluate our approach, we investigate eight different metrics to compare the performance of a system using the original and the obfuscated system prompts, and we show that the obfuscated version is constantly on par with the original one. We further perform three different deobfuscation attacks and show that with access to the obfuscated prompt and the LLM itself, we are not able to consistently extract meaningful information. Overall, we showed that prompt obfuscation can be an effective method to protect intellectual property while maintaining the same utility as the original system prompt.

摘要: 系统提示包括描述底层大型语言模型(LLM)执行的任务的详细说明，可以轻松地将基础模型转换为工具和服务，开销最小。由于它们对实用程序的关键影响，它们通常被视为知识产权，类似于软件产品的代码。但是，通过使用提示注入，很容易提取系统提示。到目前为止，还没有有效的对策来防止系统提示被窃取，所有的保护工作都可以通过精心设计的快速注入来规避，绕过所有的保护机制。我们引入即时混淆来防止系统提示符的提取，同时保持系统本身的效用，而只需很少的开销。其核心思想是找到导致相同功能的原始系统提示的表示，而混淆的系统提示不包含任何允许对原始系统提示得出结论的信息。我们实现了一种基于优化的方法，在保持功能的同时找到混淆的提示表示。为了评估我们的方法，我们调查了八个不同的度量来比较使用原始系统提示和模糊系统提示的系统的性能，我们证明了模糊版本与原始提示是相同的。我们进一步执行了三种不同的去模糊攻击，并表明通过访问混淆提示和LLM本身，我们不能一致地提取有意义的信息。总体而言，我们表明，即时混淆可以是一种有效的方法来保护知识产权，同时保持与原始系统提示相同的效用。



## **2. Assessing biomedical knowledge robustness in large language models by query-efficient sampling attacks**

通过查询高效抽样攻击评估大型语言模型中生物医学知识的稳健性 cs.CL

28 pages incl. appendix, updated version

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2402.10527v2) [paper-pdf](http://arxiv.org/pdf/2402.10527v2)

**Authors**: R. Patrick Xian, Alex J. Lee, Satvik Lolla, Vincent Wang, Qiming Cui, Russell Ro, Reza Abbasi-Asl

**Abstract**: The increasing depth of parametric domain knowledge in large language models (LLMs) is fueling their rapid deployment in real-world applications. Understanding model vulnerabilities in high-stakes and knowledge-intensive tasks is essential for quantifying the trustworthiness of model predictions and regulating their use. The recent discovery of named entities as adversarial examples (i.e. adversarial entities) in natural language processing tasks raises questions about their potential impact on the knowledge robustness of pre-trained and finetuned LLMs in high-stakes and specialized domains. We examined the use of type-consistent entity substitution as a template for collecting adversarial entities for billion-parameter LLMs with biomedical knowledge. To this end, we developed an embedding-space attack based on powerscaled distance-weighted sampling to assess the robustness of their biomedical knowledge with a low query budget and controllable coverage. Our method has favorable query efficiency and scaling over alternative approaches based on random sampling and blackbox gradient-guided search, which we demonstrated for adversarial distractor generation in biomedical question answering. Subsequent failure mode analysis uncovered two regimes of adversarial entities on the attack surface with distinct characteristics and we showed that entity substitution attacks can manipulate token-wise Shapley value explanations, which become deceptive in this setting. Our approach complements standard evaluations for high-capacity models and the results highlight the brittleness of domain knowledge in LLMs.

摘要: 大型语言模型(LLM)中参数领域知识的不断深入推动了它们在现实世界应用程序中的快速部署。了解高风险和知识密集型任务中的模型脆弱性对于量化模型预测的可信度和规范其使用至关重要。最近在自然语言处理任务中发现了命名实体作为对抗性实例(即对抗性实体)，这引发了人们对高风险和专门领域中预先训练和精细调整的LLM知识稳健性的潜在影响的问题。我们研究了使用类型一致的实体替换作为收集具有生物医学知识的10亿参数LLM的对抗性实体的模板。为此，我们提出了一种基于加权距离加权抽样的嵌入空间攻击方法，以较低的查询预算和可控的覆盖率来评估他们的生物医学知识的稳健性。与基于随机抽样和黑盒梯度引导搜索的方法相比，我们的方法具有良好的查询效率和伸缩性，并在生物医学问答中的对抗性干扰项生成中得到了验证。随后的失效模式分析揭示了攻击面上具有不同特征的两种对抗实体的机制，我们表明实体替换攻击可以操纵令人信服的Shapley值解释，在这种情况下，这种解释变得具有欺骗性。我们的方法补充了对大容量模型的标准评估，结果突出了领域知识在LLMS中的脆性。



## **3. Security Attacks on LLM-based Code Completion Tools**

对基于LLM的代码完成工具的安全攻击 cs.CL

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2408.11006v2) [paper-pdf](http://arxiv.org/pdf/2408.11006v2)

**Authors**: Wen Cheng, Ke Sun, Xinyu Zhang, Wei Wang

**Abstract**: The rapid development of large language models (LLMs) has significantly advanced code completion capabilities, giving rise to a new generation of LLM-based Code Completion Tools (LCCTs). Unlike general-purpose LLMs, these tools possess unique workflows, integrating multiple information sources as input and prioritizing code suggestions over natural language interaction, which introduces distinct security challenges. Additionally, LCCTs often rely on proprietary code datasets for training, raising concerns about the potential exposure of sensitive data. This paper exploits these distinct characteristics of LCCTs to develop targeted attack methodologies on two critical security risks: jailbreaking and training data extraction attacks. Our experimental results expose significant vulnerabilities within LCCTs, including a 99.4% success rate in jailbreaking attacks on GitHub Copilot and a 46.3% success rate on Amazon Q. Furthermore, We successfully extracted sensitive user data from GitHub Copilot, including 54 real email addresses and 314 physical addresses associated with GitHub usernames. Our study also demonstrates that these code-based attack methods are effective against general-purpose LLMs, such as the GPT series, highlighting a broader security misalignment in the handling of code by modern LLMs. These findings underscore critical security challenges associated with LCCTs and suggest essential directions for strengthening their security frameworks. The example code and attack samples from our research are provided at https://github.com/Sensente/Security-Attacks-on-LCCTs.

摘要: 大型语言模型(LLM)的快速发展极大地提升了代码补全能力，催生了新一代基于LLM的代码补全工具(LCCT)。与通用的LLMS不同，这些工具拥有独特的工作流，将多个信息源集成为输入，并优先考虑代码建议而不是自然语言交互，这带来了明显的安全挑战。此外，LCCT经常依赖专有代码数据集进行培训，这引发了人们对敏感数据潜在暴露的担忧。针对越狱攻击和训练数据提取攻击这两个关键安全风险，本文利用LCCT的这些显著特点，提出了针对性的攻击方法。我们的实验结果暴露了LCCT中的重大漏洞，包括对GitHub Copilot的越狱攻击成功率为99.4%，对Amazon Q的成功率为46.3%。此外，我们成功地从GitHub Copilot中提取了敏感用户数据，包括与GitHub用户名关联的54个真实电子邮件地址和314个物理地址。我们的研究还表明，这些基于代码的攻击方法对通用LLM是有效的，例如GPT系列，突显了现代LLM在处理代码时存在更广泛的安全错位。这些调查结果强调了与土地利用、土地利用、土地退化和土地退化有关的重大安全挑战，并提出了加强其安全框架的基本方向。我们的研究提供了示例代码和攻击示例，请访问https://github.com/Sensente/Security-Attacks-on-LCCTs.



## **4. Do Membership Inference Attacks Work on Large Language Models?**

成员资格推理攻击对大型语言模型有效吗？ cs.CL

Accepted at Conference on Language Modeling (COLM), 2024

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2402.07841v2) [paper-pdf](http://arxiv.org/pdf/2402.07841v2)

**Authors**: Michael Duan, Anshuman Suri, Niloofar Mireshghallah, Sewon Min, Weijia Shi, Luke Zettlemoyer, Yulia Tsvetkov, Yejin Choi, David Evans, Hannaneh Hajishirzi

**Abstract**: Membership inference attacks (MIAs) attempt to predict whether a particular datapoint is a member of a target model's training data. Despite extensive research on traditional machine learning models, there has been limited work studying MIA on the pre-training data of large language models (LLMs). We perform a large-scale evaluation of MIAs over a suite of language models (LMs) trained on the Pile, ranging from 160M to 12B parameters. We find that MIAs barely outperform random guessing for most settings across varying LLM sizes and domains. Our further analyses reveal that this poor performance can be attributed to (1) the combination of a large dataset and few training iterations, and (2) an inherently fuzzy boundary between members and non-members. We identify specific settings where LLMs have been shown to be vulnerable to membership inference and show that the apparent success in such settings can be attributed to a distribution shift, such as when members and non-members are drawn from the seemingly identical domain but with different temporal ranges. We release our code and data as a unified benchmark package that includes all existing MIAs, supporting future work.

摘要: 成员关系推理攻击(MIA)试图预测特定数据点是否为目标模型训练数据的成员。尽管对传统的机器学习模型进行了广泛的研究，但在大型语言模型(LLMS)的训练前数据上研究MIA的工作有限。我们在堆上训练的一套语言模型(LMS)上对MIA进行了大规模评估，参数范围从160M到12B。我们发现，对于不同的LLM大小和域，对于大多数设置，MIA的性能仅略高于随机猜测。我们的进一步分析表明，这种糟糕的性能可以归因于(1)庞大的数据集和很少的训练迭代的组合，以及(2)成员和非成员之间固有的模糊边界。我们确定了LLM被证明易受成员关系推断影响的特定设置，并表明此类设置的明显成功可以归因于分布变化，例如当成员和非成员来自看似相同的域但具有不同的时间范围时。我们将我们的代码和数据作为一个统一的基准程序包发布，其中包括所有现有的MIA，以支持未来的工作。



## **5. Robust Privacy Amidst Innovation with Large Language Models Through a Critical Assessment of the Risks**

通过对风险的批判性评估，在大型语言模型的创新中实现稳健的隐私 cs.CL

13 pages, 4 figures, 1 table, 1 supplementary, under review

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2407.16166v2) [paper-pdf](http://arxiv.org/pdf/2407.16166v2)

**Authors**: Yao-Shun Chuang, Atiquer Rahman Sarkar, Yu-Chun Hsu, Noman Mohammed, Xiaoqian Jiang

**Abstract**: This study examines integrating EHRs and NLP with large language models (LLMs) to improve healthcare data management and patient care. It focuses on using advanced models to create secure, HIPAA-compliant synthetic patient notes for biomedical research. The study used de-identified and re-identified MIMIC III datasets with GPT-3.5, GPT-4, and Mistral 7B to generate synthetic notes. Text generation employed templates and keyword extraction for contextually relevant notes, with one-shot generation for comparison. Privacy assessment checked PHI occurrence, while text utility was tested using an ICD-9 coding task. Text quality was evaluated with ROUGE and cosine similarity metrics to measure semantic similarity with source notes. Analysis of PHI occurrence and text utility via the ICD-9 coding task showed that the keyword-based method had low risk and good performance. One-shot generation showed the highest PHI exposure and PHI co-occurrence, especially in geographic location and date categories. The Normalized One-shot method achieved the highest classification accuracy. Privacy analysis revealed a critical balance between data utility and privacy protection, influencing future data use and sharing. Re-identified data consistently outperformed de-identified data. This study demonstrates the effectiveness of keyword-based methods in generating privacy-protecting synthetic clinical notes that retain data usability, potentially transforming clinical data-sharing practices. The superior performance of re-identified over de-identified data suggests a shift towards methods that enhance utility and privacy by using dummy PHIs to perplex privacy attacks.

摘要: 这项研究考察了将EHR和NLP与大型语言模型(LLM)相结合，以改进医疗数据管理和患者护理。它专注于使用高级模型创建安全的、符合HIPAA标准的合成患者笔记，用于生物医学研究。这项研究使用了GPT-3.5、GPT-4和西风7B的去识别和重新识别的MIMIC III数据集来生成合成音符。文本生成使用模板和上下文相关笔记的关键字提取，并使用一次生成进行比较。隐私评估检查了PHI的发生，而文本实用程序则使用ICD-9编码任务进行了测试。使用Rouge和Cosine相似性度量来评价文本质量，以衡量与源注释的语义相似性。通过ICD-9编码任务对PHI发生情况和文本效用的分析表明，基于关键字的方法风险低，性能好。一次发生显示出最高的PHI暴露和PHI共同出现，特别是在地理位置和日期类别。归一化一次法达到了最高的分类精度。隐私分析揭示了数据效用和隐私保护之间的关键平衡，影响了未来数据的使用和共享。重新识别的数据始终优于未识别的数据。这项研究证明了基于关键字的方法在生成保护隐私的合成临床笔记方面的有效性，这些合成笔记保留了数据的可用性，潜在地改变了临床数据共享的做法。重新识别的数据优于未识别的数据，这表明通过使用虚拟PHI来困扰隐私攻击来增强实用性和隐私的方法发生了转变。



## **6. LLM Whisperer: An Inconspicuous Attack to Bias LLM Responses**

LLM Whisperer：对LLM偏见回应的不起眼攻击 cs.CR

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2406.04755v2) [paper-pdf](http://arxiv.org/pdf/2406.04755v2)

**Authors**: Weiran Lin, Anna Gerchanovsky, Omer Akgul, Lujo Bauer, Matt Fredrikson, Zifan Wang

**Abstract**: Writing effective prompts for large language models (LLM) can be unintuitive and burdensome. In response, services that optimize or suggest prompts have emerged. While such services can reduce user effort, they also introduce a risk: the prompt provider can subtly manipulate prompts to produce heavily biased LLM responses. In this work, we show that subtle synonym replacements in prompts can increase the likelihood (by a difference up to 78%) that LLMs mention a target concept (e.g., a brand, political party, nation). We substantiate our observations through a user study, showing our adversarially perturbed prompts 1) are indistinguishable from unaltered prompts by humans, 2) push LLMs to recommend target concepts more often, and 3) make users more likely to notice target concepts, all without arousing suspicion. The practicality of this attack has the potential to undermine user autonomy. Among other measures, we recommend implementing warnings against using prompts from untrusted parties.

摘要: 为大型语言模型(LLM)编写有效的提示可能是不直观和繁琐的。作为回应，优化或建议提示的服务应运而生。虽然这类服务可以减少用户的工作，但它们也带来了风险：提示提供商可能会巧妙地操纵提示，以产生严重偏见的LLM响应。在这项工作中，我们表明，提示中微妙的同义词替换可以增加LLMS提到目标概念(例如，品牌、政党、国家)的可能性(差异高达78%)。我们通过一项用户研究证实了我们的观察结果，表明我们受到敌意干扰的提示1)与人类未更改的提示无法区分，2)推动LLMS更频繁地推荐目标概念，3)使用户更有可能注意到目标概念，所有这些都不会引起怀疑。这种攻击的实用性有可能破坏用户的自主性。在其他措施中，我们建议实施警告，以防止使用来自不受信任方的提示。



## **7. LLM Honeypot: Leveraging Large Language Models as Advanced Interactive Honeypot Systems**

LLM蜜罐：利用大型语言模型作为高级交互式蜜罐系统 cs.CR

6 pages, 5 figures

**SubmitDate**: 2024-09-15    [abs](http://arxiv.org/abs/2409.08234v2) [paper-pdf](http://arxiv.org/pdf/2409.08234v2)

**Authors**: Hakan T. Otal, M. Abdullah Canbaz

**Abstract**: The rapid evolution of cyber threats necessitates innovative solutions for detecting and analyzing malicious activity. Honeypots, which are decoy systems designed to lure and interact with attackers, have emerged as a critical component in cybersecurity. In this paper, we present a novel approach to creating realistic and interactive honeypot systems using Large Language Models (LLMs). By fine-tuning a pre-trained open-source language model on a diverse dataset of attacker-generated commands and responses, we developed a honeypot capable of sophisticated engagement with attackers. Our methodology involved several key steps: data collection and processing, prompt engineering, model selection, and supervised fine-tuning to optimize the model's performance. Evaluation through similarity metrics and live deployment demonstrated that our approach effectively generates accurate and informative responses. The results highlight the potential of LLMs to revolutionize honeypot technology, providing cybersecurity professionals with a powerful tool to detect and analyze malicious activity, thereby enhancing overall security infrastructure.

摘要: 网络威胁的快速演变需要检测和分析恶意活动的创新解决方案。蜜罐是一种诱骗系统，旨在引诱攻击者并与之互动，它已经成为网络安全的一个关键组成部分。在这篇文章中，我们提出了一种新的方法来创建真实的和交互式的蜜罐系统使用大语言模型(LLMS)。通过在攻击者生成的命令和响应的不同数据集上微调预先训练的开源语言模型，我们开发了一个能够与攻击者进行复杂交互的蜜罐。我们的方法包括几个关键步骤：数据收集和处理、快速工程、模型选择和监督微调以优化模型的性能。通过相似性度量和现场部署进行的评估表明，我们的方法有效地生成了准确和信息丰富的响应。这些结果突出了LLMS对蜜罐技术进行革命性变革的潜力，为网络安全专业人员提供了检测和分析恶意活动的强大工具，从而增强了整体安全基础设施。



## **8. Detection Made Easy: Potentials of Large Language Models for Solidity Vulnerabilities**

检测变得简单：大型语言模型针对Solidity漏洞的潜力 cs.CR

**SubmitDate**: 2024-09-15    [abs](http://arxiv.org/abs/2409.10574v1) [paper-pdf](http://arxiv.org/pdf/2409.10574v1)

**Authors**: Md Tauseef Alam, Raju Halder, Abyayananda Maiti

**Abstract**: The large-scale deployment of Solidity smart contracts on the Ethereum mainnet has increasingly attracted financially-motivated attackers in recent years. A few now-infamous attacks in Ethereum's history includes DAO attack in 2016 (50 million dollars lost), Parity Wallet hack in 2017 (146 million dollars locked), Beautychain's token BEC in 2018 (900 million dollars market value fell to 0), and NFT gaming blockchain breach in 2022 ($600 million in Ether stolen). This paper presents a comprehensive investigation of the use of large language models (LLMs) and their capabilities in detecting OWASP Top Ten vulnerabilities in Solidity. We introduce a novel, class-balanced, structured, and labeled dataset named VulSmart, which we use to benchmark and compare the performance of open-source LLMs such as CodeLlama, Llama2, CodeT5 and Falcon, alongside closed-source models like GPT-3.5 Turbo and GPT-4o Mini. Our proposed SmartVD framework is rigorously tested against these models through extensive automated and manual evaluations, utilizing BLEU and ROUGE metrics to assess the effectiveness of vulnerability detection in smart contracts. We also explore three distinct prompting strategies-zero-shot, few-shot, and chain-of-thought-to evaluate the multi-class classification and generative capabilities of the SmartVD framework. Our findings reveal that SmartVD outperforms its open-source counterparts and even exceeds the performance of closed-source base models like GPT-3.5 and GPT-4 Mini. After fine-tuning, the closed-source models, GPT-3.5 Turbo and GPT-4o Mini, achieved remarkable performance with 99% accuracy in detecting vulnerabilities, 94% in identifying their types, and 98% in determining severity. Notably, SmartVD performs best with the `chain-of-thought' prompting technique, whereas the fine-tuned closed-source models excel with the `zero-shot' prompting approach.

摘要: 近年来，在以太主机上大规模部署稳健智能合约，越来越多地吸引了出于经济动机的攻击者。Etherum历史上现在臭名昭著的攻击包括2016年的DAO攻击(损失了5000万美元)，2017年的平价钱包黑客攻击(锁定了1.46亿美元)，2018年BeautyChain的Token BEC(9亿美元的市值跌至0)，以及2022年的NFT游戏区块链漏洞(6亿美元的以太被盗)。本文对大型语言模型(LLM)的使用及其检测OWASP的十大漏洞的能力进行了全面的调查。我们介绍了一个新的，类平衡的，结构化的，带标签的数据集VulSmart，我们使用它来基准测试和比较开源LLM的性能，如CodeLlama，Llama2，CodeT5和Falcon，以及闭源模型如GPT-3.5 Turbo和GPT-40 Mini。我们建议的SmartVD框架通过广泛的自动和手动评估，针对这些模型进行了严格的测试，使用BLEU和Rouge指标来评估智能合同中漏洞检测的有效性。我们还探索了三种不同的提示策略-零射、少射和思维链-来评估SmartVD框架的多类分类和生成能力。我们的发现表明，SmartVD的表现优于开源同行，甚至超过了GPT-3.5和GPT-4 Mini等封闭源代码基础机型的性能。经过微调后，闭源模型GPT-3.5 Turbo和GPT-40 Mini取得了显著的性能，检测漏洞的准确率为99%，识别漏洞类型的准确率为94%，确定严重性的准确率为98%。值得注意的是，SmartVD最好地使用了“思维链”提示技术，而经过微调的闭源模型则使用了“零镜头”提示方法。



## **9. ContractTinker: LLM-Empowered Vulnerability Repair for Real-World Smart Contracts**

ContractTinker：针对现实世界智能合同的LLC授权漏洞修复 cs.SE

4 pages, and to be accepted in ASE2024

**SubmitDate**: 2024-09-15    [abs](http://arxiv.org/abs/2409.09661v1) [paper-pdf](http://arxiv.org/pdf/2409.09661v1)

**Authors**: Che Wang, Jiashuo Zhang, Jianbo Gao, Libin Xia, Zhi Guan, Zhong Chen

**Abstract**: Smart contracts are susceptible to being exploited by attackers, especially when facing real-world vulnerabilities. To mitigate this risk, developers often rely on third-party audit services to identify potential vulnerabilities before project deployment. Nevertheless, repairing the identified vulnerabilities is still complex and labor-intensive, particularly for developers lacking security expertise. Moreover, existing pattern-based repair tools mostly fail to address real-world vulnerabilities due to their lack of high-level semantic understanding. To fill this gap, we propose ContractTinker, a Large Language Models (LLMs)-empowered tool for real-world vulnerability repair. The key insight is our adoption of the Chain-of-Thought approach to break down the entire generation task into sub-tasks. Additionally, to reduce hallucination, we integrate program static analysis to guide the LLM. We evaluate ContractTinker on 48 high-risk vulnerabilities. The experimental results show that among the patches generated by ContractTinker, 23 (48%) are valid patches that fix the vulnerabilities, while 10 (21%) require only minor modifications. A video of ContractTinker is available at https://youtu.be/HWFVi-YHcPE.

摘要: 智能合同很容易被攻击者利用，特别是在面临现实世界的漏洞时。为了降低这种风险，开发人员通常依赖第三方审计服务在项目部署之前识别潜在的漏洞。尽管如此，修复已确定的漏洞仍然是复杂和劳动密集型的，特别是对于缺乏安全专业知识的开发人员。此外，现有的基于模式的修复工具由于缺乏高层语义理解，大多无法解决现实世界的漏洞。为了填补这一空白，我们提出了ContractTinker，这是一个大型语言模型(LLMS)授权的工具，用于修复现实世界的漏洞。关键的洞察力是我们采用了思想链的方法，将整个生成任务分解为子任务。此外，为了减少幻觉，我们集成了程序静态分析来指导LLM。我们对ContractTinker的48个高危漏洞进行了评估。实验结果表明，在ContractTinker生成的补丁中，有23个(48%)是修复漏洞的有效补丁，而10个(21%)只需要进行少量修改。有关ContractTinker的视频，请访问https://youtu.be/HWFVi-YHcPE.。



## **10. Towards Resilient and Efficient LLMs: A Comparative Study of Efficiency, Performance, and Adversarial Robustness**

迈向弹性和高效的法学硕士：效率、绩效和对抗稳健性的比较研究 cs.CL

**SubmitDate**: 2024-09-14    [abs](http://arxiv.org/abs/2408.04585v3) [paper-pdf](http://arxiv.org/pdf/2408.04585v3)

**Authors**: Xiaojing Fan, Chunliang Tao

**Abstract**: With the increasing demand for practical applications of Large Language Models (LLMs), many attention-efficient models have been developed to balance performance and computational cost. However, the adversarial robustness of these models remains under-explored. In this work, we design a framework to investigate the trade-off between efficiency, performance, and adversarial robustness of LLMs and conduct extensive experiments on three prominent models with varying levels of complexity and efficiency -- Transformer++, Gated Linear Attention (GLA) Transformer, and MatMul-Free LM -- utilizing the GLUE and AdvGLUE datasets. The AdvGLUE dataset extends the GLUE dataset with adversarial samples designed to challenge model robustness. Our results show that while the GLA Transformer and MatMul-Free LM achieve slightly lower accuracy on GLUE tasks, they demonstrate higher efficiency and either superior or comparative robustness on AdvGLUE tasks compared to Transformer++ across different attack levels. These findings highlight the potential of simplified architectures to achieve a compelling balance between efficiency, performance, and adversarial robustness, offering valuable insights for applications where resource constraints and resilience to adversarial attacks are critical.

摘要: 随着大型语言模型的实际应用需求的增加，人们已经开发了许多注意力高效的模型来平衡性能和计算成本。然而，这些模型的对抗性稳健性仍然没有得到充分的研究。在这项工作中，我们设计了一个框架来研究LLMS的效率、性能和对抗健壮性之间的权衡，并利用GLUE和AdvGLUE数据集在三个不同复杂度和效率的重要模型上进行了广泛的实验--Transformer++、门控线性注意(GLA)Transformer和MatMul-Free LM。AdvGLUE数据集使用旨在挑战模型稳健性的对抗性样本扩展了GLUE数据集。我们的结果表明，虽然GLA Transformer和MatMul-Free LM在粘合任务上的准确率略低，但在不同攻击级别上，它们在AdvGLUE任务上表现出比Transformer++更高的效率和更好的健壮性或相对较高的稳健性。这些发现突出了简化体系结构在效率、性能和对手攻击健壮性之间实现引人注目的平衡的潜力，为资源约束和对抗攻击的弹性至关重要的应用程序提供了宝贵的见解。



## **11. Tamper-Resistant Safeguards for Open-Weight LLMs**

开放重量LLM的防篡改保障措施 cs.LG

Website: https://www.tamper-resistant-safeguards.com

**SubmitDate**: 2024-09-14    [abs](http://arxiv.org/abs/2408.00761v3) [paper-pdf](http://arxiv.org/pdf/2408.00761v3)

**Authors**: Rishub Tamirisa, Bhrugu Bharathi, Long Phan, Andy Zhou, Alice Gatti, Tarun Suresh, Maxwell Lin, Justin Wang, Rowan Wang, Ron Arel, Andy Zou, Dawn Song, Bo Li, Dan Hendrycks, Mantas Mazeika

**Abstract**: Rapid advances in the capabilities of large language models (LLMs) have raised widespread concerns regarding their potential for malicious use. Open-weight LLMs present unique challenges, as existing safeguards lack robustness to tampering attacks that modify model weights. For example, recent works have demonstrated that refusal and unlearning safeguards can be trivially removed with a few steps of fine-tuning. These vulnerabilities necessitate new approaches for enabling the safe release of open-weight LLMs. We develop a method, called TAR, for building tamper-resistant safeguards into open-weight LLMs such that adversaries cannot remove the safeguards even after thousands of steps of fine-tuning. In extensive evaluations and red teaming analyses, we find that our method greatly improves tamper-resistance while preserving benign capabilities. Our results demonstrate that tamper-resistance is a tractable problem, opening up a promising new avenue to improve the safety and security of open-weight LLMs.

摘要: 大型语言模型(LLM)功能的快速发展引起了人们对其潜在恶意使用的广泛关注。开放重量LLM提出了独特的挑战，因为现有的保障措施缺乏对篡改模型权重的篡改攻击的稳健性。例如，最近的研究表明，通过几个步骤的微调，就可以很容易地消除拒绝和遗忘的保障措施。这些漏洞需要新的方法来实现安全释放未加重量的低密度脂蛋白。我们开发了一种名为TAR的方法，用于在开放重量的LLM中构建防篡改保护措施，以便对手即使在数千个步骤的微调之后也无法移除这些保护措施。在广泛的评估和红团队分析中，我们发现我们的方法在保持良性性能的同时大大提高了防篡改能力。我们的结果表明，防篡改是一个容易解决的问题，为提高开重LLMS的安全性开辟了一条很有前途的新途径。



## **12. Safeguarding AI Agents: Developing and Analyzing Safety Architectures**

保护人工智能代理：开发和分析安全架构 cs.CR

**SubmitDate**: 2024-09-13    [abs](http://arxiv.org/abs/2409.03793v2) [paper-pdf](http://arxiv.org/pdf/2409.03793v2)

**Authors**: Ishaan Domkundwar, Mukunda N S, Ishaan Bhola

**Abstract**: AI agents, specifically powered by large language models, have demonstrated exceptional capabilities in various applications where precision and efficacy are necessary. However, these agents come with inherent risks, including the potential for unsafe or biased actions, vulnerability to adversarial attacks, lack of transparency, and tendency to generate hallucinations. As AI agents become more prevalent in critical sectors of the industry, the implementation of effective safety protocols becomes increasingly important. This paper addresses the critical need for safety measures in AI systems, especially ones that collaborate with human teams. We propose and evaluate three frameworks to enhance safety protocols in AI agent systems: an LLM-powered input-output filter, a safety agent integrated within the system, and a hierarchical delegation-based system with embedded safety checks. Our methodology involves implementing these frameworks and testing them against a set of unsafe agentic use cases, providing a comprehensive evaluation of their effectiveness in mitigating risks associated with AI agent deployment. We conclude that these frameworks can significantly strengthen the safety and security of AI agent systems, minimizing potential harmful actions or outputs. Our work contributes to the ongoing effort to create safe and reliable AI applications, particularly in automated operations, and provides a foundation for developing robust guardrails to ensure the responsible use of AI agents in real-world applications.

摘要: 人工智能代理，特别是由大型语言模型驱动的，在需要精确度和效率的各种应用中展示了非凡的能力。然而，这些代理伴随着固有的风险，包括潜在的不安全或有偏见的行动，易受对手攻击，缺乏透明度，以及产生幻觉的倾向。随着人工智能代理在该行业的关键部门变得越来越普遍，实施有效的安全协议变得越来越重要。本文讨论了人工智能系统中安全措施的迫切需要，特别是与人类团队协作的系统。我们提出并评估了三个框架来增强AI代理系统中的安全协议：LLM驱动的输入输出过滤器、集成在系统中的安全代理以及嵌入安全检查的基于分级委托的系统。我们的方法涉及实现这些框架并针对一组不安全的代理用例对它们进行测试，提供对它们在降低与AI代理部署相关的风险方面的有效性的全面评估。我们的结论是，这些框架可以显著加强AI代理系统的安全性和安全性，将潜在的有害行为或输出降至最低。我们的工作有助于持续努力创建安全可靠的人工智能应用程序，特别是在自动化操作中，并为开发强大的护栏提供基础，以确保在现实世界的应用程序中负责任地使用人工智能代理。



## **13. h4rm3l: A Dynamic Benchmark of Composable Jailbreak Attacks for LLM Safety Assessment**

h4 rm3l：LLM安全评估的可组合越狱攻击的动态基准 cs.CR

**SubmitDate**: 2024-09-13    [abs](http://arxiv.org/abs/2408.04811v2) [paper-pdf](http://arxiv.org/pdf/2408.04811v2)

**Authors**: Moussa Koulako Bala Doumbouya, Ananjan Nandi, Gabriel Poesia, Davide Ghilardi, Anna Goldie, Federico Bianchi, Dan Jurafsky, Christopher D. Manning

**Abstract**: The safety of Large Language Models (LLMs) remains a critical concern due to a lack of adequate benchmarks for systematically evaluating their ability to resist generating harmful content. Previous efforts towards automated red teaming involve static or templated sets of illicit requests and adversarial prompts which have limited utility given jailbreak attacks' evolving and composable nature. We propose a novel dynamic benchmark of composable jailbreak attacks to move beyond static datasets and taxonomies of attacks and harms. Our approach consists of three components collectively called h4rm3l: (1) a domain-specific language that formally expresses jailbreak attacks as compositions of parameterized prompt transformation primitives, (2) bandit-based few-shot program synthesis algorithms that generate novel attacks optimized to penetrate the safety filters of a target black box LLM, and (3) open-source automated red-teaming software employing the previous two components. We use h4rm3l to generate a dataset of 2656 successful novel jailbreak attacks targeting 6 state-of-the-art (SOTA) open-source and proprietary LLMs. Several of our synthesized attacks are more effective than previously reported ones, with Attack Success Rates exceeding 90% on SOTA closed language models such as claude-3-haiku and GPT4-o. By generating datasets of jailbreak attacks in a unified formal representation, h4rm3l enables reproducible benchmarking and automated red-teaming, contributes to understanding LLM safety limitations, and supports the development of robust defenses in an increasingly LLM-integrated world.   Warning: This paper and related research artifacts contain offensive and potentially disturbing prompts and model-generated content.

摘要: 大型语言模型(LLM)的安全性仍然是一个严重的问题，因为缺乏系统地评估它们抵抗产生有害内容的能力的适当基准。以前的自动化红色团队的努力包括静态的或模板化的非法请求集和对抗性提示，鉴于越狱攻击不断演变和可组合的性质，这些提示的效用有限。我们提出了一种新的可组合越狱攻击的动态基准，以超越静态数据集和攻击和危害的分类。我们的方法由三个组件组成，统称为h4rm3l：(1)特定于领域的语言，它将越狱攻击形式化地表达为参数化提示转换原语的组合；(2)基于盗贼的少发程序合成算法，它生成经过优化的新型攻击，以穿透目标黑盒LLM的安全过滤器；以及(3)使用前两个组件的开源自动红队软件。我们使用h4rm3l生成了一个2656个成功的新型越狱攻击的数据集，目标是6个最先进的开源和专有LLM。我们的几个合成攻击比以前报道的更有效，在Claude-3-haiku和GPT4-o等Sota封闭语言模型上的攻击成功率超过90%。通过以统一的形式表示生成越狱攻击的数据集，h4rm3l实现了可重现的基准测试和自动化的红团队，有助于了解LLM的安全限制，并支持在日益集成LLM的世界中开发强大的防御措施。警告：本文和相关研究文章包含冒犯性和潜在令人不安的提示和模型生成的内容。



## **14. Assessing Adversarial Robustness of Large Language Models: An Empirical Study**

评估大型语言模型的对抗稳健性：实证研究 cs.CL

Oral presentation at KDD 2024 GenAI Evaluation workshop

**SubmitDate**: 2024-09-12    [abs](http://arxiv.org/abs/2405.02764v2) [paper-pdf](http://arxiv.org/pdf/2405.02764v2)

**Authors**: Zeyu Yang, Zhao Meng, Xiaochen Zheng, Roger Wattenhofer

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing, but their robustness against adversarial attacks remains a critical concern. We presents a novel white-box style attack approach that exposes vulnerabilities in leading open-source LLMs, including Llama, OPT, and T5. We assess the impact of model size, structure, and fine-tuning strategies on their resistance to adversarial perturbations. Our comprehensive evaluation across five diverse text classification tasks establishes a new benchmark for LLM robustness. The findings of this study have far-reaching implications for the reliable deployment of LLMs in real-world applications and contribute to the advancement of trustworthy AI systems.

摘要: 大型语言模型（LLM）彻底改变了自然语言处理，但其对抗攻击的稳健性仍然是一个关键问题。我们提出了一种新颖的白盒式攻击方法，该方法暴露了领先开源LLM（包括Llama、OPT和T5）中的漏洞。我们评估了模型大小、结构和微调策略对其抵抗对抗性扰动的影响。我们对五种不同文本分类任务的全面评估为LLM稳健性建立了新基准。这项研究的结果对于LLM在现实世界应用程序中的可靠部署具有深远的影响，并有助于发展值得信赖的人工智能系统。



## **15. Trading Devil Final: Backdoor attack via Stock market and Bayesian Optimization**

交易魔鬼决赛：通过股市和Bayesian优化进行后门攻击 cs.LG

END (will never be modified again!!) :Jumps-Diffusion and stock  market: Better quantify uncertainty in financial simulations

**SubmitDate**: 2024-09-12    [abs](http://arxiv.org/abs/2407.14573v5) [paper-pdf](http://arxiv.org/pdf/2407.14573v5)

**Authors**: Orson Mengara

**Abstract**: Since the advent of generative artificial intelligence, every company and researcher has been rushing to develop their own generative models, whether commercial or not. Given the large number of users of these powerful new tools, there is currently no intrinsically verifiable way to explain from the ground up what happens when LLMs (large language models) learn. For example, those based on automatic speech recognition systems, which have to rely on huge and astronomical amounts of data collected from all over the web to produce fast and efficient results, In this article, we develop a backdoor attack called MarketBackFinal 2.0, based on acoustic data poisoning, MarketBackFinal 2.0 is mainly based on modern stock market models. In order to show the possible vulnerabilities of speech-based transformers that may rely on LLMs.

摘要: 自生成人工智能出现以来，每家公司和研究人员都在争先恐后地开发自己的生成模型，无论是否商业化。鉴于这些强大的新工具的大量用户，目前还没有本质上可验证的方法来从头解释LLM（大型语言模型）学习时会发生什么。例如，那些基于自动语音识别系统的系统，它们必须依赖于从整个网络收集的大量数据来产生快速有效的结果，在本文中，我们开发了一种名为MarketBackFinal 2.0的后门攻击，基于声学数据中毒，MarketBackFinal 2.0主要基于现代股市模型。为了显示可能依赖LLM的基于语音的转换器可能存在的漏洞。



## **16. Securing Large Language Models: Addressing Bias, Misinformation, and Prompt Attacks**

保护大型语言模型：解决偏见、错误信息和即时攻击 cs.CR

17 pages, 1 figure

**SubmitDate**: 2024-09-12    [abs](http://arxiv.org/abs/2409.08087v1) [paper-pdf](http://arxiv.org/pdf/2409.08087v1)

**Authors**: Benji Peng, Keyu Chen, Ming Li, Pohsun Feng, Ziqian Bi, Junyu Liu, Qian Niu

**Abstract**: Large Language Models (LLMs) demonstrate impressive capabilities across various fields, yet their increasing use raises critical security concerns. This article reviews recent literature addressing key issues in LLM security, with a focus on accuracy, bias, content detection, and vulnerability to attacks. Issues related to inaccurate or misleading outputs from LLMs is discussed, with emphasis on the implementation from fact-checking methodologies to enhance response reliability. Inherent biases within LLMs are critically examined through diverse evaluation techniques, including controlled input studies and red teaming exercises. A comprehensive analysis of bias mitigation strategies is presented, including approaches from pre-processing interventions to in-training adjustments and post-processing refinements. The article also probes the complexity of distinguishing LLM-generated content from human-produced text, introducing detection mechanisms like DetectGPT and watermarking techniques while noting the limitations of machine learning enabled classifiers under intricate circumstances. Moreover, LLM vulnerabilities, including jailbreak attacks and prompt injection exploits, are analyzed by looking into different case studies and large-scale competitions like HackAPrompt. This review is concluded by retrospecting defense mechanisms to safeguard LLMs, accentuating the need for more extensive research into the LLM security field.

摘要: 大型语言模型(LLM)在各个领域展示了令人印象深刻的能力，但它们的日益使用引发了严重的安全问题。本文回顾了解决LLM安全关键问题的最新文献，重点讨论了准确性、偏差、内容检测和攻击漏洞。讨论了与LLMS输出不准确或误导性有关的问题，重点是从事实核查方法中实施以提高响应可靠性。通过不同的评估技术，包括受控投入研究和红色团队练习，对LLM中的固有偏见进行了严格的检查。对偏差缓解策略进行了全面的分析，包括从前处理干预到培训中调整和后处理改进的方法。文章还探讨了区分LLM生成的内容和人类生成的文本的复杂性，介绍了DetectGPT和水印技术等检测机制，同时指出了机器学习支持的分类器在复杂环境下的局限性。此外，通过研究不同的案例研究和HackAPrompt等大型竞争，分析了LLM漏洞，包括越狱攻击和快速注入利用。本综述通过回顾保护LLM的防御机制来结束，强调了对LLM安全领域进行更广泛研究的必要性。



## **17. A Survey of Backdoor Attacks and Defenses on Large Language Models: Implications for Security Measures**

大型语言模型后门攻击和防御的调查：对安全措施的影响 cs.CR

**SubmitDate**: 2024-09-12    [abs](http://arxiv.org/abs/2406.06852v4) [paper-pdf](http://arxiv.org/pdf/2406.06852v4)

**Authors**: Shuai Zhao, Meihuizi Jia, Zhongliang Guo, Leilei Gan, Xiaoyu Xu, Xiaobao Wu, Jie Fu, Yichao Feng, Fengjun Pan, Luu Anh Tuan

**Abstract**: Large Language Models (LLMs), which bridge the gap between human language understanding and complex problem-solving, achieve state-of-the-art performance on several NLP tasks, particularly in few-shot and zero-shot settings. Despite the demonstrable efficacy of LLMs, due to constraints on computational resources, users have to engage with open-source language models or outsource the entire training process to third-party platforms. However, research has demonstrated that language models are susceptible to potential security vulnerabilities, particularly in backdoor attacks. Backdoor attacks are designed to introduce targeted vulnerabilities into language models by poisoning training samples or model weights, allowing attackers to manipulate model responses through malicious triggers. While existing surveys on backdoor attacks provide a comprehensive overview, they lack an in-depth examination of backdoor attacks specifically targeting LLMs. To bridge this gap and grasp the latest trends in the field, this paper presents a novel perspective on backdoor attacks for LLMs by focusing on fine-tuning methods. Specifically, we systematically classify backdoor attacks into three categories: full-parameter fine-tuning, parameter-efficient fine-tuning, and no fine-tuning Based on insights from a substantial review, we also discuss crucial issues for future research on backdoor attacks, such as further exploring attack algorithms that do not require fine-tuning, or developing more covert attack algorithms.

摘要: 大型语言模型(LLM)架起了人类语言理解和复杂问题解决之间的桥梁，在几个NLP任务上实现了最先进的性能，特别是在少镜头和零镜头的情况下。尽管LLMS具有明显的功效，但由于计算资源的限制，用户不得不使用开放源码语言模型或将整个培训过程外包给第三方平台。然而，研究表明，语言模型容易受到潜在的安全漏洞的影响，特别是在后门攻击中。后门攻击旨在通过毒化训练样本或模型权重，将有针对性的漏洞引入语言模型，允许攻击者通过恶意触发器操纵模型响应。虽然现有的关于后门攻击的调查提供了全面的概述，但它们缺乏对专门针对LLM的后门攻击的深入检查。为了弥补这一差距，掌握该领域的最新趋势，本文提出了一种新的视角来研究针对LLMS的后门攻击，重点是微调方法。具体地说，我们系统地将后门攻击分为三类：全参数微调、参数高效微调和无微调。在大量综述的基础上，我们还讨论了未来后门攻击研究的关键问题，如进一步探索不需要微调的攻击算法，或开发更隐蔽的攻击算法。



## **18. Exploring LLMs for Malware Detection: Review, Framework Design, and Countermeasure Approaches**

探索恶意软件检测的LLM：审查、框架设计和对策方法 cs.CR

26 pages, 7 figures, 4 tables

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2409.07587v1) [paper-pdf](http://arxiv.org/pdf/2409.07587v1)

**Authors**: Jamal Al-Karaki, Muhammad Al-Zafar Khan, Marwan Omar

**Abstract**: The rising use of Large Language Models (LLMs) to create and disseminate malware poses a significant cybersecurity challenge due to their ability to generate and distribute attacks with ease. A single prompt can initiate a wide array of malicious activities. This paper addresses this critical issue through a multifaceted approach. First, we provide a comprehensive overview of LLMs and their role in malware detection from diverse sources. We examine five specific applications of LLMs: Malware honeypots, identification of text-based threats, code analysis for detecting malicious intent, trend analysis of malware, and detection of non-standard disguised malware. Our review includes a detailed analysis of the existing literature and establishes guiding principles for the secure use of LLMs. We also introduce a classification scheme to categorize the relevant literature. Second, we propose performance metrics to assess the effectiveness of LLMs in these contexts. Third, we present a risk mitigation framework designed to prevent malware by leveraging LLMs. Finally, we evaluate the performance of our proposed risk mitigation strategies against various factors and demonstrate their effectiveness in countering LLM-enabled malware. The paper concludes by suggesting future advancements and areas requiring deeper exploration in this fascinating field of artificial intelligence.

摘要: 越来越多的人使用大型语言模型(LLM)来创建和传播恶意软件，这对网络安全构成了重大挑战，因为它们能够轻松地生成和分发攻击。一个提示可能会引发广泛的恶意活动。本文通过多方面的方法解决了这一关键问题。首先，我们全面概述了LLM及其在来自不同来源的恶意软件检测中的作用。我们研究了LLMS的五个具体应用：恶意软件蜜罐、基于文本的威胁识别、用于检测恶意意图的代码分析、恶意软件的趋势分析和非标准伪装恶意软件的检测。我们的审查包括对现有文献的详细分析，并确立了安全使用低成本管理的指导原则。我们还引入了一种分类方案来对相关文献进行分类。其次，我们提出了性能指标来评估在这些环境下的低成本管理的有效性。第三，我们提出了一个旨在通过利用LLMS来防止恶意软件的风险缓解框架。最后，我们评估了我们提出的风险缓解策略针对各种因素的性能，并展示了它们在对抗启用LLM的恶意软件方面的有效性。论文最后提出了在人工智能这一迷人领域的未来进展和需要更深入探索的领域。



## **19. Securing Vision-Language Models with a Robust Encoder Against Jailbreak and Adversarial Attacks**

使用稳健的编码器保护视觉语言模型免受越狱和对抗攻击 cs.CV

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2409.07353v1) [paper-pdf](http://arxiv.org/pdf/2409.07353v1)

**Authors**: Md Zarif Hossain, Ahmed Imteaj

**Abstract**: Large Vision-Language Models (LVLMs), trained on multimodal big datasets, have significantly advanced AI by excelling in vision-language tasks. However, these models remain vulnerable to adversarial attacks, particularly jailbreak attacks, which bypass safety protocols and cause the model to generate misleading or harmful responses. This vulnerability stems from both the inherent susceptibilities of LLMs and the expanded attack surface introduced by the visual modality. We propose Sim-CLIP+, a novel defense mechanism that adversarially fine-tunes the CLIP vision encoder by leveraging a Siamese architecture. This approach maximizes cosine similarity between perturbed and clean samples, facilitating resilience against adversarial manipulations. Sim-CLIP+ offers a plug-and-play solution, allowing seamless integration into existing LVLM architectures as a robust vision encoder. Unlike previous defenses, our method requires no structural modifications to the LVLM and incurs minimal computational overhead. Sim-CLIP+ demonstrates effectiveness against both gradient-based adversarial attacks and various jailbreak techniques. We evaluate Sim-CLIP+ against three distinct jailbreak attack strategies and perform clean evaluations using standard downstream datasets, including COCO for image captioning and OKVQA for visual question answering. Extensive experiments demonstrate that Sim-CLIP+ maintains high clean accuracy while substantially improving robustness against both gradient-based adversarial attacks and jailbreak techniques. Our code and robust vision encoders are available at https://github.com/speedlab-git/Robust-Encoder-against-Jailbreak-attack.git.

摘要: 在多模式大数据集上训练的大型视觉语言模型(LVLM)通过在视觉语言任务中脱颖而出，极大地促进了人工智能的发展。然而，这些模型仍然容易受到对抗性攻击，特别是越狱攻击，这些攻击绕过了安全协议，并导致模型生成误导性或有害的响应。该漏洞既源于LLMS固有的易感性，也源于视觉通道引入的扩展攻击面。我们提出了Sim-Clip+，这是一种新颖的防御机制，它利用暹罗体系结构对剪辑视觉编码器进行了相反的微调。这种方法最大限度地提高了扰动样本和干净样本之间的余弦相似性，促进了对对手操纵的弹性。SIM-Clip+提供了一种即插即用的解决方案，允许无缝集成到现有的LVLM架构中，作为一种强大的视觉编码器。与以前的防御方法不同，我们的方法不需要对LVLM进行结构修改，并且产生的计算开销最小。SIM-CLIP+展示了对抗基于梯度的对抗性攻击和各种越狱技术的有效性。我们针对三种不同的越狱攻击策略对Sim-Clip+进行了评估，并使用标准下游数据集进行了干净的评估，包括用于图像字幕的COCO和用于视觉问题回答的OKVQA。广泛的实验表明，Sim-Clip+保持了很高的干净准确率，同时显著提高了对基于梯度的对手攻击和越狱技术的稳健性。我们的代码和健壮的视觉编码器可在https://github.com/speedlab-git/Robust-Encoder-against-Jailbreak-attack.git.上获得



## **20. AEGIS: Online Adaptive AI Content Safety Moderation with Ensemble of LLM Experts**

AEGIS：与LLM专家一起进行在线自适应人工智能内容安全审核 cs.LG

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2404.05993v2) [paper-pdf](http://arxiv.org/pdf/2404.05993v2)

**Authors**: Shaona Ghosh, Prasoon Varshney, Erick Galinkin, Christopher Parisien

**Abstract**: As Large Language Models (LLMs) and generative AI become more widespread, the content safety risks associated with their use also increase. We find a notable deficiency in high-quality content safety datasets and benchmarks that comprehensively cover a wide range of critical safety areas. To address this, we define a broad content safety risk taxonomy, comprising 13 critical risk and 9 sparse risk categories. Additionally, we curate AEGISSAFETYDATASET, a new dataset of approximately 26, 000 human-LLM interaction instances, complete with human annotations adhering to the taxonomy. We plan to release this dataset to the community to further research and to help benchmark LLM models for safety. To demonstrate the effectiveness of the dataset, we instruction-tune multiple LLM-based safety models. We show that our models (named AEGISSAFETYEXPERTS), not only surpass or perform competitively with the state-of-the-art LLM-based safety models and general purpose LLMs, but also exhibit robustness across multiple jail-break attack categories. We also show how using AEGISSAFETYDATASET during the LLM alignment phase does not negatively impact the performance of the aligned models on MT Bench scores. Furthermore, we propose AEGIS, a novel application of a no-regret online adaptation framework with strong theoretical guarantees, to perform content moderation with an ensemble of LLM content safety experts in deployment

摘要: 随着大型语言模型(LLM)和生成式人工智能变得更加普遍，与使用它们相关的内容安全风险也增加了。我们发现，在全面覆盖广泛关键安全领域的高质量内容安全数据集和基准方面存在明显不足。为了解决这一问题，我们定义了一个广泛的内容安全风险分类，包括13个关键风险和9个稀疏风险类别。此外，我们还管理了AEGISSAFETYDATASET，这是一个大约包含26,000个人与LLM交互实例的新数据集，其中包含符合分类的人类注释。我们计划向社区发布这个数据集，以进行进一步的研究，并帮助对LLM模型的安全性进行基准测试。为了证明数据集的有效性，我们对多个基于LLM的安全模型进行了指令调整。我们证明了我们的模型(命名为AEGISSAFETYEXPERTS)不仅性能优于最先进的基于LLM的安全模型和通用LLM，而且在多个越狱攻击类别中表现出健壮性。我们还展示了在LLM校准阶段使用AEGISSAFETYDATASET如何不会对MT板凳分数上的校准模型的性能产生负面影响。此外，我们提出了Aegis，一个具有强大理论保证的无遗憾在线适配框架的新应用，在部署时使用LLM内容安全专家集成来执行内容审核



## **21. DrLLM: Prompt-Enhanced Distributed Denial-of-Service Resistance Method with Large Language Models**

DrLLM：具有大型语言模型的预算增强型分布式拒绝服务抵抗方法 cs.CR

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2409.10561v1) [paper-pdf](http://arxiv.org/pdf/2409.10561v1)

**Authors**: Zhenyu Yin, Shang Liu, Guangyuan Xu

**Abstract**: The increasing number of Distributed Denial of Service (DDoS) attacks poses a major threat to the Internet, highlighting the importance of DDoS mitigation. Most existing approaches require complex training methods to learn data features, which increases the complexity and generality of the application. In this paper, we propose DrLLM, which aims to mine anomalous traffic information in zero-shot scenarios through Large Language Models (LLMs). To bridge the gap between DrLLM and existing approaches, we embed the global and local information of the traffic data into the reasoning paradigm and design three modules, namely Knowledge Embedding, Token Embedding, and Progressive Role Reasoning, for data representation and reasoning. In addition we explore the generalization of prompt engineering in the cybersecurity domain to improve the classification capability of DrLLM. Our ablation experiments demonstrate the applicability of DrLLM in zero-shot scenarios and further demonstrate the potential of LLMs in the network domains. DrLLM implementation code has been open-sourced at https://github.com/liuup/DrLLM.

摘要: 越来越多的分布式拒绝服务(DDoS)攻击对互联网构成了重大威胁，突显了缓解DDoS的重要性。现有的大多数方法需要复杂的训练方法来学习数据特征，这增加了应用程序的复杂性和通用性。在本文中，我们提出了DrLLM，旨在通过大型语言模型(LLMS)挖掘零镜头场景中的异常交通信息。为了弥补DrLLM与已有方法之间的差距，我们将交通数据的全局和局部信息嵌入到推理范式中，并设计了三个模块，即知识嵌入、令牌嵌入和渐进角色推理，用于数据表示和推理。此外，我们还探索了快速工程在网络安全领域的泛化，以提高DrLLM的分类能力。我们的烧蚀实验证明了DrLLM在零射情况下的适用性，并进一步展示了LLM在网络领域的潜力。DrLLm实现代码已在https://github.com/liuup/DrLLM.上开源



## **22. The Philosopher's Stone: Trojaning Plugins of Large Language Models**

哲学家之石：大型语言模型的特洛伊插件 cs.CR

Accepted by NDSS Symposium 2025. Please cite this paper as "Tian  Dong, Minhui Xue, Guoxing Chen, Rayne Holland, Yan Meng, Shaofeng Li, Zhen  Liu, Haojin Zhu. The Philosopher's Stone: Trojaning Plugins of Large Language  Models. In the 32nd Annual Network and Distributed System Security Symposium  (NDSS 2025)."

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2312.00374v3) [paper-pdf](http://arxiv.org/pdf/2312.00374v3)

**Authors**: Tian Dong, Minhui Xue, Guoxing Chen, Rayne Holland, Yan Meng, Shaofeng Li, Zhen Liu, Haojin Zhu

**Abstract**: Open-source Large Language Models (LLMs) have recently gained popularity because of their comparable performance to proprietary LLMs. To efficiently fulfill domain-specialized tasks, open-source LLMs can be refined, without expensive accelerators, using low-rank adapters. However, it is still unknown whether low-rank adapters can be exploited to control LLMs. To address this gap, we demonstrate that an infected adapter can induce, on specific triggers,an LLM to output content defined by an adversary and to even maliciously use tools. To train a Trojan adapter, we propose two novel attacks, POLISHED and FUSION, that improve over prior approaches. POLISHED uses a superior LLM to align na\"ively poisoned data based on our insight that it can better inject poisoning knowledge during training. In contrast, FUSION leverages a novel over-poisoning procedure to transform a benign adapter into a malicious one by magnifying the attention between trigger and target in model weights. In our experiments, we first conduct two case studies to demonstrate that a compromised LLM agent can use malware to control the system (e.g., a LLM-driven robot) or to launch a spear-phishing attack. Then, in terms of targeted misinformation, we show that our attacks provide higher attack effectiveness than the existing baseline and, for the purpose of attracting downloads, preserve or improve the adapter's utility. Finally, we designed and evaluated three potential defenses. However, none proved entirely effective in safeguarding against our attacks, highlighting the need for more robust defenses supporting a secure LLM supply chain.

摘要: 开源的大型语言模型(LLM)最近越来越受欢迎，因为它们的性能可以与专有的LLM相媲美。为了高效地完成领域专门化任务，可以使用低级别适配器对开源LLM进行提炼，而无需使用昂贵的加速器。然而，是否可以利用低阶适配器来控制LLM仍然是未知的。为了弥补这一漏洞，我们演示了受感染的适配器可以在特定触发下诱导LLM输出由对手定义的内容，甚至恶意使用工具。为了训练木马适配器，我们提出了两种新的攻击方法，磨光攻击和融合攻击，它们比以前的方法有所改进。基于我们对在训练过程中可以更好地注入中毒知识的洞察力，波兰德使用了一种高级的LLM来对齐严重中毒的数据。相比之下，Fusion利用一种新的过度中毒程序，通过放大模型权重中触发器和目标之间的注意力，将良性适配器转换为恶意适配器。在我们的实验中，我们首先进行了两个案例研究，以证明受攻击的LLM代理可以使用恶意软件控制系统(例如，LLM驱动的机器人)或发起鱼叉式网络钓鱼攻击。然后，在有针对性的错误信息方面，我们表明我们的攻击提供了比现有基线更高的攻击效率，并且出于吸引下载的目的，保留或提高了适配器的实用性。最后，我们设计并评估了三种可能的防御措施。然而，没有一种被证明在防御我们的攻击方面是完全有效的，这突显了需要更强大的防御来支持安全的LLM供应链。



## **23. AdaPPA: Adaptive Position Pre-Fill Jailbreak Attack Approach Targeting LLMs**

AdaPPA：针对LLM的自适应位置预填充越狱攻击方法 cs.CR

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2409.07503v1) [paper-pdf](http://arxiv.org/pdf/2409.07503v1)

**Authors**: Lijia Lv, Weigang Zhang, Xuehai Tang, Jie Wen, Feng Liu, Jizhong Han, Songlin Hu

**Abstract**: Jailbreak vulnerabilities in Large Language Models (LLMs) refer to methods that extract malicious content from the model by carefully crafting prompts or suffixes, which has garnered significant attention from the research community. However, traditional attack methods, which primarily focus on the semantic level, are easily detected by the model. These methods overlook the difference in the model's alignment protection capabilities at different output stages. To address this issue, we propose an adaptive position pre-fill jailbreak attack approach for executing jailbreak attacks on LLMs. Our method leverages the model's instruction-following capabilities to first output pre-filled safe content, then exploits its narrative-shifting abilities to generate harmful content. Extensive black-box experiments demonstrate our method can improve the attack success rate by 47% on the widely recognized secure model (Llama2) compared to existing approaches. Our code can be found at: https://github.com/Yummy416/AdaPPA.

摘要: 大型语言模型中的越狱漏洞是指通过精心制作提示或后缀从模型中提取恶意内容的方法，引起了研究界的极大关注。然而，传统的攻击方法主要集中在语义层面，很容易被该模型检测到。这些方法忽略了模型在不同输出阶段的对准保护能力的差异。针对这一问题，我们提出了一种自适应的位置预填充越狱攻击方法来执行对低层管理系统的越狱攻击。我们的方法利用模型的指令遵循能力来首先输出预先填充的安全内容，然后利用其叙事转换能力来生成有害内容。大量的黑盒实验表明，与现有方法相比，该方法在公认的安全模型(Llama2)上可以将攻击成功率提高47%。我们的代码可在以下网址找到：https://github.com/Yummy416/AdaPPA.



## **24. Well, that escalated quickly: The Single-Turn Crescendo Attack (STCA)**

嗯，情况迅速升级：单转渐强攻击（STCA） cs.CR

**SubmitDate**: 2024-09-10    [abs](http://arxiv.org/abs/2409.03131v2) [paper-pdf](http://arxiv.org/pdf/2409.03131v2)

**Authors**: Alan Aqrawi, Arian Abbasi

**Abstract**: This paper introduces a new method for adversarial attacks on large language models (LLMs) called the Single-Turn Crescendo Attack (STCA). Building on the multi-turn crescendo attack method introduced by Russinovich, Salem, and Eldan (2024), which gradually escalates the context to provoke harmful responses, the STCA achieves similar outcomes in a single interaction. By condensing the escalation into a single, well-crafted prompt, the STCA bypasses typical moderation filters that LLMs use to prevent inappropriate outputs. This technique reveals vulnerabilities in current LLMs and emphasizes the importance of stronger safeguards in responsible AI (RAI). The STCA offers a novel method that has not been previously explored.

摘要: 本文介绍了一种针对大型语言模型（LLM）的对抗性攻击的新方法，称为单轮渐强攻击（STCA）。STCA基于Russinovich、Salem和Eldan（2024）引入的多回合渐强攻击方法（该方法逐渐升级上下文以引发有害反应），在单次交互中实现了类似的结果。通过将升级浓缩为一个精心设计的提示，STCA绕过了LLM用来防止不当输出的典型审核过滤器。该技术揭示了当前LLM中的漏洞，并强调了负责任人工智能（RAI）中更强有力的保障措施的重要性。STCA提供了一种以前尚未探索过的新颖方法。



## **25. Espresso: Robust Concept Filtering in Text-to-Image Models**

浓缩咖啡：文本到图像模型中的稳健概念过滤 cs.CV

**SubmitDate**: 2024-09-09    [abs](http://arxiv.org/abs/2404.19227v5) [paper-pdf](http://arxiv.org/pdf/2404.19227v5)

**Authors**: Anudeep Das, Vasisht Duddu, Rui Zhang, N. Asokan

**Abstract**: Diffusion based text-to-image models are trained on large datasets scraped from the Internet, potentially containing unacceptable concepts (e.g., copyright infringing or unsafe). We need concept removal techniques (CRTs) which are effective in preventing the generation of images with unacceptable concepts, utility-preserving on acceptable concepts, and robust against evasion with adversarial prompts. None of the prior CRTs satisfy all these requirements simultaneously. We introduce Espresso, the first robust concept filter based on Contrastive Language-Image Pre-Training (CLIP). We configure CLIP to identify unacceptable concepts in generated images using the distance of their embeddings to the text embeddings of both unacceptable and acceptable concepts. This lets us fine-tune for robustness by separating the text embeddings of unacceptable and acceptable concepts while preserving their pairing with image embeddings for utility. We present a pipeline to evaluate various CRTs, attacks against them, and show that Espresso, is more effective and robust than prior CRTs, while retaining utility.

摘要: 基于扩散的文本到图像模型是在从互联网上收集的大数据集上进行训练的，这些数据集可能包含不可接受的概念(例如，侵犯版权或不安全)。我们需要概念移除技术(CRT)，它能有效地防止生成包含不可接受概念的图像，保留可接受概念的效用，并对带有对抗性提示的规避具有健壮性。以前的CRT没有一种同时满足所有这些要求。介绍了第一个基于对比语言-图像预训练(CLIP)的稳健概念过滤器Espresso。我们将CLIP配置为在生成的图像中识别不可接受的概念，使用其嵌入到不可接受和可接受概念的文本嵌入的距离。这使我们可以通过分离不可接受和可接受概念的文本嵌入，同时保留它们与图像嵌入的配对以实现实用，从而对健壮性进行微调。我们提出了一种评估各种CRT的流水线，对它们的攻击，并表明Espresso，比以前的CRT更有效和健壮，同时保持了实用性。



## **26. OneEdit: A Neural-Symbolic Collaboratively Knowledge Editing System**

OneEdit：一个神经符号协作知识编辑系统 cs.AI

LLM+KG@VLDB2024, code is available at  https://github.com/zjunlp/OneEdit

**SubmitDate**: 2024-09-09    [abs](http://arxiv.org/abs/2409.07497v1) [paper-pdf](http://arxiv.org/pdf/2409.07497v1)

**Authors**: Ningyu Zhang, Zekun Xi, Yujie Luo, Peng Wang, Bozhong Tian, Yunzhi Yao, Jintian Zhang, Shumin Deng, Mengshu Sun, Lei Liang, Zhiqiang Zhang, Xiaowei Zhu, Jun Zhou, Huajun Chen

**Abstract**: Knowledge representation has been a central aim of AI since its inception. Symbolic Knowledge Graphs (KGs) and neural Large Language Models (LLMs) can both represent knowledge. KGs provide highly accurate and explicit knowledge representation, but face scalability issue; while LLMs offer expansive coverage of knowledge, but incur significant training costs and struggle with precise and reliable knowledge manipulation. To this end, we introduce OneEdit, a neural-symbolic prototype system for collaborative knowledge editing using natural language, which facilitates easy-to-use knowledge management with KG and LLM. OneEdit consists of three modules: 1) The Interpreter serves for user interaction with natural language; 2) The Controller manages editing requests from various users, leveraging the KG with rollbacks to handle knowledge conflicts and prevent toxic knowledge attacks; 3) The Editor utilizes the knowledge from the Controller to edit KG and LLM. We conduct experiments on two new datasets with KGs which demonstrate that OneEdit can achieve superior performance.

摘要: 自人工智能诞生以来，知识表示一直是其核心目标。符号知识图(KGs)和神经大语言模型(LLM)都可以表示知识。KGS提供了高度准确和明确的知识表示，但面临可伸缩性问题；而LLMS提供了广泛的知识覆盖，但会产生巨大的培训成本，并难以精确和可靠地处理知识。为此，我们介绍了一个使用自然语言进行协同知识编辑的神经符号原型系统OneEDIT，该系统利用KG和LLM实现了简单易用的知识管理。OneEDIT由三个模块组成：1)解释器，用于用户与自然语言的交互；2)控制器管理来自不同用户的编辑请求，利用带回滚的KG来处理知识冲突，防止有毒知识攻击；3)编辑器利用控制器的知识来编辑KG和LLM。我们使用KGS在两个新的数据集上进行了实验，结果表明OneEDIT可以获得更好的性能。



## **27. Exploiting the Vulnerability of Large Language Models via Defense-Aware Architectural Backdoor**

通过防御意识架构后门利用大型语言模型的漏洞 cs.CR

**SubmitDate**: 2024-09-09    [abs](http://arxiv.org/abs/2409.01952v2) [paper-pdf](http://arxiv.org/pdf/2409.01952v2)

**Authors**: Abdullah Arafat Miah, Yu Bi

**Abstract**: Deep neural networks (DNNs) have long been recognized as vulnerable to backdoor attacks. By providing poisoned training data in the fine-tuning process, the attacker can implant a backdoor into the victim model. This enables input samples meeting specific textual trigger patterns to be classified as target labels of the attacker's choice. While such black-box attacks have been well explored in both computer vision and natural language processing (NLP), backdoor attacks relying on white-box attack philosophy have hardly been thoroughly investigated. In this paper, we take the first step to introduce a new type of backdoor attack that conceals itself within the underlying model architecture. Specifically, we propose to design separate backdoor modules consisting of two functions: trigger detection and noise injection. The add-on modules of model architecture layers can detect the presence of input trigger tokens and modify layer weights using Gaussian noise to disturb the feature distribution of the baseline model. We conduct extensive experiments to evaluate our attack methods using two model architecture settings on five different large language datasets. We demonstrate that the training-free architectural backdoor on a large language model poses a genuine threat. Unlike the-state-of-art work, it can survive the rigorous fine-tuning and retraining process, as well as evade output probability-based defense methods (i.e. BDDR). All the code and data is available https://github.com/SiSL-URI/Arch_Backdoor_LLM.

摘要: 深度神经网络(DNN)长期以来一直被认为容易受到后门攻击。通过在微调过程中提供有毒的训练数据，攻击者可以在受害者模型中植入后门。这使得满足特定文本触发模式的输入样本能够被分类为攻击者选择的目标标签。虽然这种黑盒攻击在计算机视觉和自然语言处理(NLP)中都得到了很好的研究，但依赖于白盒攻击思想的后门攻击几乎没有得到彻底的调查。在本文中，我们首先介绍一种隐藏在底层模型体系结构中的新型后门攻击。具体地说，我们建议设计独立的后门模块，包括两个功能：触发检测和噪声注入。模型体系结构层的附加模块可以检测输入触发令牌的存在，并使用高斯噪声来修改层权重，以干扰基线模型的特征分布。我们在五个不同的大型语言数据集上使用两个模型体系结构设置进行了广泛的实验来评估我们的攻击方法。我们证明，大型语言模型上的免培训体系结构后门构成了真正的威胁。与最先进的工作不同，它可以在严格的微调和重新训练过程中幸存下来，并且可以避开基于输出概率的防御方法(即BDDR)。所有代码和数据均可在https://github.com/SiSL-URI/Arch_Backdoor_LLM.上使用



## **28. Jailbreaking Text-to-Image Models with LLM-Based Agents**

使用基于LLM的代理破解文本到图像模型 cs.CR

**SubmitDate**: 2024-09-09    [abs](http://arxiv.org/abs/2408.00523v2) [paper-pdf](http://arxiv.org/pdf/2408.00523v2)

**Authors**: Yingkai Dong, Zheng Li, Xiangtao Meng, Ning Yu, Shanqing Guo

**Abstract**: Recent advancements have significantly improved automated task-solving capabilities using autonomous agents powered by large language models (LLMs). However, most LLM-based agents focus on dialogue, programming, or specialized domains, leaving their potential for addressing generative AI safety tasks largely unexplored. In this paper, we propose Atlas, an advanced LLM-based multi-agent framework targeting generative AI models, specifically focusing on jailbreak attacks against text-to-image (T2I) models with built-in safety filters. Atlas consists of two agents, namely the mutation agent and the selection agent, each comprising four key modules: a vision-language model (VLM) or LLM brain, planning, memory, and tool usage. The mutation agent uses its VLM brain to determine whether a prompt triggers the T2I model's safety filter. It then collaborates iteratively with the LLM brain of the selection agent to generate new candidate jailbreak prompts with the highest potential to bypass the filter. In addition to multi-agent communication, we leverage in-context learning (ICL) memory mechanisms and the chain-of-thought (COT) approach to learn from past successes and failures, thereby enhancing Atlas's performance. Our evaluation demonstrates that Atlas successfully jailbreaks several state-of-the-art T2I models equipped with multi-modal safety filters in a black-box setting. Additionally, Atlas outperforms existing methods in both query efficiency and the quality of generated images. This work convincingly demonstrates the successful application of LLM-based agents in studying the safety vulnerabilities of popular text-to-image generation models. We urge the community to consider advanced techniques like ours in response to the rapidly evolving text-to-image generation field.

摘要: 最近的进步显著提高了使用大型语言模型(LLM)支持的自主代理的自动任务求解能力。然而，大多数基于LLM的代理专注于对话、编程或专业领域，因此它们处理生成性AI安全任务的潜力在很大程度上尚未开发。在本文中，我们提出了一个先进的基于LLM的多智能体框架Atlas，该框架针对生成式人工智能模型，特别是针对带有内置安全过滤器的文本到图像(T2I)模型的越狱攻击。Atlas由两个智能体组成，即突变智能体和选择智能体，每个智能体包括四个关键模块：视觉语言模型(VLM)或LLM大脑、规划、记忆和工具使用。突变代理使用其VLM大脑来确定提示是否触发了T2I模型的安全过滤器。然后，它与选择代理的LLM大脑迭代协作，生成最有可能绕过过滤器的新候选越狱提示。除了多智能体通信外，我们还利用情境学习(ICL)记忆机制和思想链(COT)方法从过去的成功和失败中学习，从而提高Atlas的性能。我们的评估表明，Atlas在黑匣子环境中成功越狱了几款配备多模式安全过滤器的最先进的T2I车型。此外，Atlas在查询效率和生成图像的质量方面都优于现有方法。这项工作令人信服地展示了基于LLM的代理在研究流行的文本到图像生成模型的安全漏洞方面的成功应用。我们敦促社区考虑像我们这样的先进技术，以应对快速发展的文本到图像生成领域。



## **29. Fine-Tuning, Quantization, and LLMs: Navigating Unintended Outcomes**

微调、量化和LLM：应对意外结果 cs.CR

**SubmitDate**: 2024-09-09    [abs](http://arxiv.org/abs/2404.04392v3) [paper-pdf](http://arxiv.org/pdf/2404.04392v3)

**Authors**: Divyanshu Kumar, Anurakt Kumar, Sahil Agarwal, Prashanth Harshangi

**Abstract**: Large Language Models (LLMs) have gained widespread adoption across various domains, including chatbots and auto-task completion agents. However, these models are susceptible to safety vulnerabilities such as jailbreaking, prompt injection, and privacy leakage attacks. These vulnerabilities can lead to the generation of malicious content, unauthorized actions, or the disclosure of confidential information. While foundational LLMs undergo alignment training and incorporate safety measures, they are often subject to fine-tuning, or doing quantization resource-constrained environments. This study investigates the impact of these modifications on LLM safety, a critical consideration for building reliable and secure AI systems. We evaluate foundational models including Mistral, Llama series, Qwen, and MosaicML, along with their fine-tuned variants. Our comprehensive analysis reveals that fine-tuning generally increases the success rates of jailbreak attacks, while quantization has variable effects on attack success rates. Importantly, we find that properly implemented guardrails significantly enhance resistance to jailbreak attempts. These findings contribute to our understanding of LLM vulnerabilities and provide insights for developing more robust safety strategies in the deployment of language models.

摘要: 大型语言模型(LLM)已经在各个领域得到了广泛的采用，包括聊天机器人和自动任务完成代理。然而，这些模型容易受到越狱、快速注入和隐私泄露攻击等安全漏洞的影响。这些漏洞可能导致生成恶意内容、未经授权的操作或泄露机密信息。虽然基本的LLM接受对准培训并纳入安全措施，但它们经常受到微调或量化资源限制环境的影响。这项研究调查了这些修改对LLM安全的影响，LLM安全是构建可靠和安全的AI系统的关键考虑因素。我们评估了基本模型，包括Mistral、Llama系列、Qwen和MosaicML，以及它们的微调变体。我们的综合分析表明，微调通常会提高越狱攻击的成功率，而量化对攻击成功率的影响是不同的。重要的是，我们发现，正确安装护栏大大增强了对越狱企图的抵抗力。这些发现有助于我们理解LLM漏洞，并为在语言模型的部署中开发更健壮的安全策略提供了见解。



## **30. A Study on Prompt Injection Attack Against LLM-Integrated Mobile Robotic Systems**

针对LLM集成移动机器人系统的即时注入攻击研究 cs.RO

**SubmitDate**: 2024-09-09    [abs](http://arxiv.org/abs/2408.03515v2) [paper-pdf](http://arxiv.org/pdf/2408.03515v2)

**Authors**: Wenxiao Zhang, Xiangrui Kong, Conan Dewitt, Thomas Braunl, Jin B. Hong

**Abstract**: The integration of Large Language Models (LLMs) like GPT-4o into robotic systems represents a significant advancement in embodied artificial intelligence. These models can process multi-modal prompts, enabling them to generate more context-aware responses. However, this integration is not without challenges. One of the primary concerns is the potential security risks associated with using LLMs in robotic navigation tasks. These tasks require precise and reliable responses to ensure safe and effective operation. Multi-modal prompts, while enhancing the robot's understanding, also introduce complexities that can be exploited maliciously. For instance, adversarial inputs designed to mislead the model can lead to incorrect or dangerous navigational decisions. This study investigates the impact of prompt injections on mobile robot performance in LLM-integrated systems and explores secure prompt strategies to mitigate these risks. Our findings demonstrate a substantial overall improvement of approximately 30.8% in both attack detection and system performance with the implementation of robust defence mechanisms, highlighting their critical role in enhancing security and reliability in mission-oriented tasks.

摘要: 将像GPT-40这样的大型语言模型(LLM)集成到机器人系统中，代表着体现的人工智能的重大进步。这些模型可以处理多模式提示，使它们能够生成更多情景感知响应。然而，这种整合并不是没有挑战。其中一个主要问题是在机器人导航任务中使用LLMS存在潜在的安全风险。这些任务需要准确可靠的反应，以确保安全有效的运行。多模式提示在增强机器人理解能力的同时，也引入了可能被恶意利用的复杂性。例如，旨在误导模型的对抗性输入可能导致错误或危险的导航决策。这项研究调查了快速注射对LLM集成系统中移动机器人性能的影响，并探索了安全的提示策略来缓解这些风险。我们的研究结果表明，随着强大的防御机制的实施，攻击检测和系统性能都有了大约30.8%的大幅整体改进，突出了它们在增强面向任务的任务的安全性和可靠性方面的关键作用。



## **31. PIP: Detecting Adversarial Examples in Large Vision-Language Models via Attention Patterns of Irrelevant Probe Questions**

PIP：通过不相关探索问题的注意力模式检测大型视觉语言模型中的对抗示例 cs.CV

Accepted by ACM Multimedia 2024 BNI track (Oral)

**SubmitDate**: 2024-09-08    [abs](http://arxiv.org/abs/2409.05076v1) [paper-pdf](http://arxiv.org/pdf/2409.05076v1)

**Authors**: Yudong Zhang, Ruobing Xie, Jiansheng Chen, Xingwu Sun, Yu Wang

**Abstract**: Large Vision-Language Models (LVLMs) have demonstrated their powerful multimodal capabilities. However, they also face serious safety problems, as adversaries can induce robustness issues in LVLMs through the use of well-designed adversarial examples. Therefore, LVLMs are in urgent need of detection tools for adversarial examples to prevent incorrect responses. In this work, we first discover that LVLMs exhibit regular attention patterns for clean images when presented with probe questions. We propose an unconventional method named PIP, which utilizes the attention patterns of one randomly selected irrelevant probe question (e.g., "Is there a clock?") to distinguish adversarial examples from clean examples. Regardless of the image to be tested and its corresponding question, PIP only needs to perform one additional inference of the image to be tested and the probe question, and then achieves successful detection of adversarial examples. Even under black-box attacks and open dataset scenarios, our PIP, coupled with a simple SVM, still achieves more than 98% recall and a precision of over 90%. Our PIP is the first attempt to detect adversarial attacks on LVLMs via simple irrelevant probe questions, shedding light on deeper understanding and introspection within LVLMs. The code is available at https://github.com/btzyd/pip.

摘要: 大型视觉语言模型(LVLM)已经展示了其强大的多通道能力。然而，它们也面临着严重的安全问题，因为攻击者可以通过使用设计良好的对抗性示例在LVLM中引发健壮性问题。因此，LVLMS迫切需要针对对抗性实例的检测工具来防止错误响应。在这项工作中，我们首先发现，当被呈现探索性问题时，LVLMS对干净的图像表现出规则的注意模式。我们提出了一种非常规的方法，称为PIP，它利用了一个随机选择的无关探测问题的注意模式(例如，“有时钟吗？”)区分敌意的例子和干净的例子。无论待测试图像及其对应的问题是什么，PIP只需要对待测试图像和探测问题进行一次额外的推理，即可实现对抗性实例的成功检测。即使在黑盒攻击和开放数据集场景下，我们的PIP结合简单的支持向量机，仍然可以达到98%以上的召回率和90%以上的准确率。我们的PIP是首次尝试通过简单的无关紧要的探索性问题来检测对LVLMS的敌意攻击，从而揭示了LVLMS内部更深层次的理解和反省。代码可在https://github.com/btzyd/pip.上获得



## **32. Using Large Language Models for Template Detection from Security Event Logs**

使用大型语言模型从安全事件收件箱进行模板检测 cs.CR

**SubmitDate**: 2024-09-08    [abs](http://arxiv.org/abs/2409.05045v1) [paper-pdf](http://arxiv.org/pdf/2409.05045v1)

**Authors**: Risto Vaarandi, Hayretdin Bahsi

**Abstract**: In modern IT systems and computer networks, real-time and offline event log analysis is a crucial part of cyber security monitoring. In particular, event log analysis techniques are essential for the timely detection of cyber attacks and for assisting security experts with the analysis of past security incidents. The detection of line patterns or templates from unstructured textual event logs has been identified as an important task of event log analysis since detected templates represent event types in the event log and prepare the logs for downstream online or offline security monitoring tasks. During the last two decades, a number of template mining algorithms have been proposed. However, many proposed algorithms rely on traditional data mining techniques, and the usage of Large Language Models (LLMs) has received less attention so far. Also, most approaches that harness LLMs are supervised, and unsupervised LLM-based template mining remains an understudied area. The current paper addresses this research gap and investigates the application of LLMs for unsupervised detection of templates from unstructured security event logs.

摘要: 在现代IT系统和计算机网络中，实时和离线事件日志分析是网络安全监控的重要组成部分。特别是，事件日志分析技术对于及时检测网络攻击和协助安全专家分析过去的安全事件至关重要。从非结构化文本事件日志中检测线条模式或模板已被确定为事件日志分析的重要任务，因为检测到的模板代表事件日志中的事件类型，并为下游在线或离线安全监控任务准备日志。在过去的二十年里，人们提出了许多模板挖掘算法。然而，许多已提出的算法依赖于传统的数据挖掘技术，而大型语言模型(LLM)的使用到目前为止还没有得到足够的重视。此外，大多数利用LLM的方法都是有监督的，而基于无监督LLM的模板挖掘仍然是一个研究较少的领域。本文弥补了这一研究空白，并研究了LLMS在非结构化安全事件日志模板非监督检测中的应用。



## **33. Vision-fused Attack: Advancing Aggressive and Stealthy Adversarial Text against Neural Machine Translation**

视觉融合攻击：推进攻击性和隐形对抗文本对抗神经机器翻译 cs.CL

IJCAI 2024

**SubmitDate**: 2024-09-08    [abs](http://arxiv.org/abs/2409.05021v1) [paper-pdf](http://arxiv.org/pdf/2409.05021v1)

**Authors**: Yanni Xue, Haojie Hao, Jiakai Wang, Qiang Sheng, Renshuai Tao, Yu Liang, Pu Feng, Xianglong Liu

**Abstract**: While neural machine translation (NMT) models achieve success in our daily lives, they show vulnerability to adversarial attacks. Despite being harmful, these attacks also offer benefits for interpreting and enhancing NMT models, thus drawing increased research attention. However, existing studies on adversarial attacks are insufficient in both attacking ability and human imperceptibility due to their sole focus on the scope of language. This paper proposes a novel vision-fused attack (VFA) framework to acquire powerful adversarial text, i.e., more aggressive and stealthy. Regarding the attacking ability, we design the vision-merged solution space enhancement strategy to enlarge the limited semantic solution space, which enables us to search for adversarial candidates with higher attacking ability. For human imperceptibility, we propose the perception-retained adversarial text selection strategy to align the human text-reading mechanism. Thus, the finally selected adversarial text could be more deceptive. Extensive experiments on various models, including large language models (LLMs) like LLaMA and GPT-3.5, strongly support that VFA outperforms the comparisons by large margins (up to 81%/14% improvements on ASR/SSIM).

摘要: 尽管神经机器翻译(NMT)模型在我们的日常生活中取得了成功，但它们在面对对手攻击时表现出了脆弱性。尽管这些攻击是有害的，但也为解释和增强NMT模型提供了好处，因此吸引了更多的研究关注。然而，现有的对抗性攻击研究由于只关注语言的范围，在攻击能力和人的隐蔽性方面都存在不足。提出了一种新的视觉融合攻击(VFA)框架，以获得更具攻击性和隐蔽性的强大敌意文本。在攻击能力方面，我们设计了视觉融合解空间增强策略来扩大有限的语义解空间，使我们能够搜索到攻击能力更强的对抗性候选。针对人类的不可感知性，我们提出了保留感知的对抗性文本选择策略来对齐人类的文本阅读机制。因此，最终选定的对抗性文本可能更具欺骗性。在各种模型上的广泛实验，包括大语言模型(LLM)，如骆驼和GPT-3.5，有力地支持了VFA的性能远远超过比较(在ASR/SSIM上高达81%/14%的改进)。



## **34. TF-Attack: Transferable and Fast Adversarial Attacks on Large Language Models**

TF攻击：对大型语言模型的可转移且快速对抗攻击 cs.CL

14 pages, 6 figures

**SubmitDate**: 2024-09-08    [abs](http://arxiv.org/abs/2408.13985v3) [paper-pdf](http://arxiv.org/pdf/2408.13985v3)

**Authors**: Zelin Li, Kehai Chen, Lemao Liu, Xuefeng Bai, Mingming Yang, Yang Xiang, Min Zhang

**Abstract**: With the great advancements in large language models (LLMs), adversarial attacks against LLMs have recently attracted increasing attention. We found that pre-existing adversarial attack methodologies exhibit limited transferability and are notably inefficient, particularly when applied to LLMs. In this paper, we analyze the core mechanisms of previous predominant adversarial attack methods, revealing that 1) the distributions of importance score differ markedly among victim models, restricting the transferability; 2) the sequential attack processes induces substantial time overheads. Based on the above two insights, we introduce a new scheme, named TF-Attack, for Transferable and Fast adversarial attacks on LLMs. TF-Attack employs an external LLM as a third-party overseer rather than the victim model to identify critical units within sentences. Moreover, TF-Attack introduces the concept of Importance Level, which allows for parallel substitutions of attacks. We conduct extensive experiments on 6 widely adopted benchmarks, evaluating the proposed method through both automatic and human metrics. Results show that our method consistently surpasses previous methods in transferability and delivers significant speed improvements, up to 20 times faster than earlier attack strategies.

摘要: 近年来，随着大型语言模型的发展，针对大型语言模型的对抗性攻击引起了越来越多的关注。我们发现，现有的对抗性攻击方法表现出有限的可转移性和显著的低效，特别是当应用于LLM时。本文分析了以往主流对抗性攻击方法的核心机制，发现1)不同受害者模型的重要性分数分布明显不同，限制了可转移性；2)顺序攻击过程导致了大量的时间开销。基于以上两点，我们提出了一种新的方案，称为TF-Attack，用于对LLMS进行可转移和快速对抗攻击。TF-Attack使用外部LLM作为第三方监督者，而不是受害者模型来识别判刑内的关键单元。此外，TF-Attack还引入了重要度的概念，允许并行替换攻击。我们在6个广泛采用的基准上进行了广泛的实验，从自动度量和人工度量两个方面对所提出的方法进行了评估。结果表明，我们的方法在可转移性上始终优于以前的方法，并提供了显著的速度改进，比以前的攻击策略快20倍。



## **35. DistillSeq: A Framework for Safety Alignment Testing in Large Language Models using Knowledge Distillation**

DistillSeq：使用知识蒸馏在大型语言模型中进行安全一致测试的框架 cs.SE

**SubmitDate**: 2024-09-08    [abs](http://arxiv.org/abs/2407.10106v4) [paper-pdf](http://arxiv.org/pdf/2407.10106v4)

**Authors**: Mingke Yang, Yuqi Chen, Yi Liu, Ling Shi

**Abstract**: Large Language Models (LLMs) have showcased their remarkable capabilities in diverse domains, encompassing natural language understanding, translation, and even code generation. The potential for LLMs to generate harmful content is a significant concern. This risk necessitates rigorous testing and comprehensive evaluation of LLMs to ensure safe and responsible use. However, extensive testing of LLMs requires substantial computational resources, making it an expensive endeavor. Therefore, exploring cost-saving strategies during the testing phase is crucial to balance the need for thorough evaluation with the constraints of resource availability. To address this, our approach begins by transferring the moderation knowledge from an LLM to a small model. Subsequently, we deploy two distinct strategies for generating malicious queries: one based on a syntax tree approach, and the other leveraging an LLM-based method. Finally, our approach incorporates a sequential filter-test process designed to identify test cases that are prone to eliciting toxic responses. Our research evaluated the efficacy of DistillSeq across four LLMs: GPT-3.5, GPT-4.0, Vicuna-13B, and Llama-13B. In the absence of DistillSeq, the observed attack success rates on these LLMs stood at 31.5% for GPT-3.5, 21.4% for GPT-4.0, 28.3% for Vicuna-13B, and 30.9% for Llama-13B. However, upon the application of DistillSeq, these success rates notably increased to 58.5%, 50.7%, 52.5%, and 54.4%, respectively. This translated to an average escalation in attack success rate by a factor of 93.0% when compared to scenarios without the use of DistillSeq. Such findings highlight the significant enhancement DistillSeq offers in terms of reducing the time and resource investment required for effectively testing LLMs.

摘要: 大型语言模型(LLM)已经在不同的领域展示了它们非凡的能力，包括自然语言理解、翻译，甚至代码生成。低密度脂蛋白产生有害内容的可能性是一个重大关切。这种风险需要对LLMS进行严格的测试和全面评估，以确保安全和负责任地使用。然而，大规模的LLMS测试需要大量的计算资源，这使得它成为一项昂贵的工作。因此，在测试阶段探索节约成本的策略对于平衡彻底评估的需要和资源可用性的限制至关重要。为了解决这个问题，我们的方法首先将适度知识从LLM转移到一个小模型。随后，我们部署了两种不同的策略来生成恶意查询：一种基于语法树方法，另一种利用基于LLM的方法。最后，我们的方法结合了一个顺序的过滤测试过程，旨在识别容易引发有毒反应的测试用例。我们的研究评估了DistillSeq在四种低密度脂蛋白上的疗效：GPT-3.5、GPT-4.0、Vicuna-13B和Llama-13B。在没有DistillSeq的情况下，观察到的对这些LLMS的攻击成功率GPT-3.5为31.5%，GPT-4.0为21.4%，Vicuna-13B为28.3%，Llama-13B为30.9%。然而，在应用DistillSeq后，这些成功率分别显著增加到58.5%、50.7%、52.5%和54.4%。与未使用DistillSeq的情况相比，这意味着攻击成功率平均提升了93.0%。这些发现突出了DistillSeq在减少有效测试LLM所需的时间和资源投资方面所提供的显著增强。



## **36. Exploring Straightforward Conversational Red-Teaming**

探索直接的对话式红色团队 cs.CL

**SubmitDate**: 2024-09-07    [abs](http://arxiv.org/abs/2409.04822v1) [paper-pdf](http://arxiv.org/pdf/2409.04822v1)

**Authors**: George Kour, Naama Zwerdling, Marcel Zalmanovici, Ateret Anaby-Tavor, Ora Nova Fandina, Eitan Farchi

**Abstract**: Large language models (LLMs) are increasingly used in business dialogue systems but they pose security and ethical risks. Multi-turn conversations, where context influences the model's behavior, can be exploited to produce undesired responses. In this paper, we examine the effectiveness of utilizing off-the-shelf LLMs in straightforward red-teaming approaches, where an attacker LLM aims to elicit undesired output from a target LLM, comparing both single-turn and conversational red-teaming tactics. Our experiments offer insights into various usage strategies that significantly affect their performance as red teamers. They suggest that off-the-shelf models can act as effective red teamers and even adjust their attack strategy based on past attempts, although their effectiveness decreases with greater alignment.

摘要: 大型语言模型（LLM）越来越多地用于商业对话系统，但它们带来了安全和道德风险。上下文影响模型行为的多轮对话可以被利用来产生不希望的响应。在本文中，我们研究了在简单的红色团队方法中利用现成的LLM的有效性，其中攻击者LLM旨在从目标LLM中获取不希望的输出，并比较了单回合和对话式红色团队策略。我们的实验深入了解了各种使用策略，这些策略显着影响他们作为红色团队成员的表现。他们认为，现成的模型可以充当有效的红色团队成员，甚至根据过去的尝试调整他们的攻击策略，尽管它们的有效性随着一致性的提高而降低。



## **37. Goal-guided Generative Prompt Injection Attack on Large Language Models**

对大型语言模型的目标引导生成提示注入攻击 cs.CR

11 pages, 6 figures

**SubmitDate**: 2024-09-07    [abs](http://arxiv.org/abs/2404.07234v2) [paper-pdf](http://arxiv.org/pdf/2404.07234v2)

**Authors**: Chong Zhang, Mingyu Jin, Qinkai Yu, Chengzhi Liu, Haochen Xue, Xiaobo Jin

**Abstract**: Current large language models (LLMs) provide a strong foundation for large-scale user-oriented natural language tasks. A large number of users can easily inject adversarial text or instructions through the user interface, thus causing LLMs model security challenges. Although there is currently a large amount of research on prompt injection attacks, most of these black-box attacks use heuristic strategies. It is unclear how these heuristic strategies relate to the success rate of attacks and thus effectively improve model robustness. To solve this problem, we redefine the goal of the attack: to maximize the KL divergence between the conditional probabilities of the clean text and the adversarial text. Furthermore, we prove that maximizing the KL divergence is equivalent to maximizing the Mahalanobis distance between the embedded representation $x$ and $x'$ of the clean text and the adversarial text when the conditional probability is a Gaussian distribution and gives a quantitative relationship on $x$ and $x'$. Then we designed a simple and effective goal-guided generative prompt injection strategy (G2PIA) to find an injection text that satisfies specific constraints to achieve the optimal attack effect approximately. It is particularly noteworthy that our attack method is a query-free black-box attack method with low computational cost. Experimental results on seven LLM models and four datasets show the effectiveness of our attack method.

摘要: 现有的大型语言模型为大规模面向用户的自然语言任务提供了坚实的基础。大量用户可以很容易地通过用户界面注入敌意文本或指令，从而造成LLMS模型的安全挑战。虽然目前有大量关于即时注入攻击的研究，但这些黑盒攻击大多采用启发式策略。目前尚不清楚这些启发式策略如何与攻击成功率相关，从而有效地提高模型的稳健性。为了解决这个问题，我们重新定义了攻击的目标：最大化纯文本和敌意文本的条件概率之间的KL偏差。此外，我们证明了当条件概率为高斯分布时，最大化KL发散度等价于最大化明文和敌意文本的嵌入表示$x$和$x‘$之间的马氏距离，并给出了关于$x$和$x’$的定量关系。然后，设计了一种简单有效的目标引导生成性提示注入策略(G2PIA)，找到满足特定约束的注入文本，以近似达到最优的攻击效果。特别值得注意的是，我们的攻击方法是一种计算代价低的无查询黑盒攻击方法。在7个LLM模型和4个数据集上的实验结果表明了该攻击方法的有效性。



## **38. Recent Advances in Attack and Defense Approaches of Large Language Models**

大型语言模型攻击和防御方法的最新进展 cs.CR

**SubmitDate**: 2024-09-06    [abs](http://arxiv.org/abs/2409.03274v2) [paper-pdf](http://arxiv.org/pdf/2409.03274v2)

**Authors**: Jing Cui, Yishi Xu, Zhewei Huang, Shuchang Zhou, Jianbin Jiao, Junge Zhang

**Abstract**: Large Language Models (LLMs) have revolutionized artificial intelligence and machine learning through their advanced text processing and generating capabilities. However, their widespread deployment has raised significant safety and reliability concerns. Established vulnerabilities in deep neural networks, coupled with emerging threat models, may compromise security evaluations and create a false sense of security. Given the extensive research in the field of LLM security, we believe that summarizing the current state of affairs will help the research community better understand the present landscape and inform future developments. This paper reviews current research on LLM vulnerabilities and threats, and evaluates the effectiveness of contemporary defense mechanisms. We analyze recent studies on attack vectors and model weaknesses, providing insights into attack mechanisms and the evolving threat landscape. We also examine current defense strategies, highlighting their strengths and limitations. By contrasting advancements in attack and defense methodologies, we identify research gaps and propose future directions to enhance LLM security. Our goal is to advance the understanding of LLM safety challenges and guide the development of more robust security measures.

摘要: 大型语言模型(LLM)通过其先进的文本处理和生成能力，使人工智能和机器学习发生了革命性的变化。然而，它们的广泛部署引发了严重的安全和可靠性问题。深层神经网络中已建立的漏洞，再加上新出现的威胁模型，可能会危及安全评估，并造成一种错误的安全感。鉴于LLM安全领域的广泛研究，我们相信总结当前的事态将有助于研究界更好地了解目前的情况并为未来的发展提供信息。本文回顾了LLM漏洞和威胁的研究现状，并对现代防御机制的有效性进行了评估。我们分析了最近关于攻击载体和模型弱点的研究，提供了对攻击机制和不断演变的威胁环境的洞察。我们还研究了当前的防御战略，强调了它们的优势和局限性。通过对比攻击和防御方法的进展，我们发现了研究的差距，并提出了增强LLM安全的未来方向。我们的目标是促进对LLM安全挑战的理解，并指导开发更强大的安全措施。



## **39. LLM-PBE: Assessing Data Privacy in Large Language Models**

LLM-PBE：评估大型语言模型中的数据隐私 cs.CR

**SubmitDate**: 2024-09-06    [abs](http://arxiv.org/abs/2408.12787v2) [paper-pdf](http://arxiv.org/pdf/2408.12787v2)

**Authors**: Qinbin Li, Junyuan Hong, Chulin Xie, Jeffrey Tan, Rachel Xin, Junyi Hou, Xavier Yin, Zhun Wang, Dan Hendrycks, Zhangyang Wang, Bo Li, Bingsheng He, Dawn Song

**Abstract**: Large Language Models (LLMs) have become integral to numerous domains, significantly advancing applications in data management, mining, and analysis. Their profound capabilities in processing and interpreting complex language data, however, bring to light pressing concerns regarding data privacy, especially the risk of unintentional training data leakage. Despite the critical nature of this issue, there has been no existing literature to offer a comprehensive assessment of data privacy risks in LLMs. Addressing this gap, our paper introduces LLM-PBE, a toolkit crafted specifically for the systematic evaluation of data privacy risks in LLMs. LLM-PBE is designed to analyze privacy across the entire lifecycle of LLMs, incorporating diverse attack and defense strategies, and handling various data types and metrics. Through detailed experimentation with multiple LLMs, LLM-PBE facilitates an in-depth exploration of data privacy concerns, shedding light on influential factors such as model size, data characteristics, and evolving temporal dimensions. This study not only enriches the understanding of privacy issues in LLMs but also serves as a vital resource for future research in the field. Aimed at enhancing the breadth of knowledge in this area, the findings, resources, and our full technical report are made available at https://llm-pbe.github.io/, providing an open platform for academic and practical advancements in LLM privacy assessment.

摘要: 大型语言模型(LLM)已经成为许多领域不可或缺的一部分，极大地推动了数据管理、挖掘和分析方面的应用。然而，它们在处理和解释复杂语言数据方面的深厚能力暴露了人们对数据隐私的迫切关切，特别是无意中泄露培训数据的风险。尽管这一问题具有严重的性质，但目前还没有文献对低成本管理中的数据隐私风险进行全面评估。针对这一差距，我们引入了LLM-PBE，这是一个专门为系统评估LLMS中的数据隐私风险而设计的工具包。LLm-PBE旨在分析LLMS整个生命周期中的隐私，整合不同的攻击和防御策略，并处理各种数据类型和指标。通过对多个LLM的详细实验，LLM-PBE有助于深入探索数据隐私问题，揭示模型大小、数据特征和不断演变的时间维度等影响因素。这一研究不仅丰富了对LLMS中隐私问题的理解，也为该领域未来的研究提供了重要的资源。为了提高这一领域的知识广度，我们的调查结果、资源和完整的技术报告可在https://llm-pbe.github.io/，上获得，为LLM隐私评估的学术和实践进步提供了一个开放的平台。



## **40. SelfDefend: LLMs Can Defend Themselves against Jailbreaking in a Practical Manner**

SelfDefend：LLM可以以实用的方式保护自己免受越狱的侵害 cs.CR

This paper completes its earlier vision paper, available at  arXiv:2402.15727. Updated to the latest analysis and results

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2406.05498v2) [paper-pdf](http://arxiv.org/pdf/2406.05498v2)

**Authors**: Xunguang Wang, Daoyuan Wu, Zhenlan Ji, Zongjie Li, Pingchuan Ma, Shuai Wang, Yingjiu Li, Yang Liu, Ning Liu, Juergen Rahmel

**Abstract**: Jailbreaking is an emerging adversarial attack that bypasses the safety alignment deployed in off-the-shelf large language models (LLMs) and has evolved into multiple categories: human-based, optimization-based, generation-based, and the recent indirect and multilingual jailbreaks. However, delivering a practical jailbreak defense is challenging because it needs to not only handle all the above jailbreak attacks but also incur negligible delays to user prompts, as well as be compatible with both open-source and closed-source LLMs. Inspired by how the traditional security concept of shadow stacks defends against memory overflow attacks, this paper introduces a generic LLM jailbreak defense framework called SelfDefend, which establishes a shadow LLM as a defense instance to concurrently protect the target LLM instance in the normal stack and collaborate with it for checkpoint-based access control. The effectiveness of SelfDefend builds upon our observation that existing LLMs (both target and defense LLMs) have the capability to identify harmful prompts or intentions in user queries, which we empirically validate using the commonly used GPT-3.5/4 models across all major jailbreak attacks. To further improve the defense's robustness and minimize costs, we employ a data distillation approach to tune dedicated open-source defense models. These models outperform six state-of-the-art defenses and match the performance of GPT-4-based SelfDefend, with significantly lower extra delays. We also empirically show that the tuned models are robust to adaptive jailbreaks and prompt injections.

摘要: 越狱是一种新兴的对抗性攻击，它绕过了现成的大型语言模型(LLM)中部署的安全对齐，并已演变为多种类别：基于人的、基于优化的、基于代的以及最近的间接和多语言越狱。然而，提供实际的越狱防御是具有挑战性的，因为它不仅需要处理所有上述越狱攻击，还需要对用户提示造成可以忽略不计的延迟，以及与开源和闭源LLM兼容。受传统影子堆栈安全概念防御内存溢出攻击的启发，提出了一种通用的LLM越狱防御框架--SelfDefend。该框架建立一个影子LLM作为防御实例，同时保护正常堆栈中的目标LLM实例，并与其协作进行基于检查点的访问控制。SelfDefend的有效性建立在我们的观察基础上，即现有的LLM(目标和防御LLM)能够识别用户查询中的有害提示或意图，我们使用所有主要越狱攻击中常用的GPT-3.5/4模型进行了经验验证。为了进一步提高防御的健壮性并将成本降至最低，我们使用数据蒸馏方法来优化专用的开源防御模型。这些型号的性能超过了六种最先进的防御系统，并与基于GPT-4的SelfDefend的性能相当，额外延迟显著降低。我们的经验还表明，调整后的模型对自适应越狱和快速注入具有较强的鲁棒性。



## **41. Towards Neural Network based Cognitive Models of Dynamic Decision-Making by Humans**

基于神经网络的人类动态决策认知模型 cs.LG

Our code is available at https://github.com/shshnkreddy/NCM-HDM

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2407.17622v2) [paper-pdf](http://arxiv.org/pdf/2407.17622v2)

**Authors**: Changyu Chen, Shashank Reddy Chirra, Maria José Ferreira, Cleotilde Gonzalez, Arunesh Sinha, Pradeep Varakantham

**Abstract**: Modeling human cognitive processes in dynamic decision-making tasks has been an endeavor in AI for a long time because such models can help make AI systems more intuitive, personalized, mitigate any human biases, and enhance training in simulation. Some initial work has attempted to utilize neural networks (and large language models) but often assumes one common model for all humans and aims to emulate human behavior in aggregate. However, the behavior of each human is distinct, heterogeneous, and relies on specific past experiences in certain tasks. For instance, consider two individuals responding to a phishing email: one who has previously encountered and identified similar threats may recognize it quickly, while another without such experience might fall for the scam. In this work, we build on Instance Based Learning (IBL) that posits that human decisions are based on similar situations encountered in the past. However, IBL relies on simple fixed form functions to capture the mapping from past situations to current decisions. To that end, we propose two new attention-based neural network models to have open form non-linear functions to model distinct and heterogeneous human decision-making in dynamic settings. We experiment with two distinct datasets gathered from human subject experiment data, one focusing on detection of phishing email by humans and another where humans act as attackers in a cybersecurity setting and decide on an attack option. We conducted extensive experiments with our two neural network models, IBL, and GPT3.5, and demonstrate that the neural network models outperform IBL significantly in representing human decision-making, while providing similar interpretability of human decisions as IBL. Overall, our work yields promising results for further use of neural networks in cognitive modeling of human decision making.

摘要: 长期以来，在动态决策任务中对人类认知过程进行建模一直是人工智能领域的一项努力，因为这样的模型可以帮助人工智能系统变得更加直观、个性化，缓解任何人类偏见，并加强模拟训练。一些最初的工作试图利用神经网络(和大型语言模型)，但通常假设一个适用于所有人类的通用模型，并旨在总体上模拟人类的行为。然而，每个人的行为都是不同的、不同的，并依赖于某些任务中特定的过去经验。例如，假设两个人回复了一封钓鱼电子邮件：其中一个人之前遇到并识别了类似的威胁，可能很快就能识别出它，而另一个没有这种经验的人可能会上当。在这项工作中，我们建立在基于实例的学习(IBL)的基础上，该学习假设人类的决策是基于过去遇到的类似情况。然而，IBL依赖于简单的固定表单函数来捕获从过去情况到当前决策的映射。为此，我们提出了两个新的基于注意力的神经网络模型，它们具有开放形式的非线性函数，以在动态环境中模拟不同和不同种类的人类决策。我们使用从人类受试者实验数据中收集的两个不同的数据集进行实验，一个专注于检测人类发送的钓鱼电子邮件，另一个则是人类在网络安全环境中充当攻击者，并决定攻击选项。我们用我们的两个神经网络模型IBL和GPT3.5进行了广泛的实验，并证明了神经网络模型在表示人类决策方面显著优于IBL，同时提供了与IBL相似的人类决策的可解释性。总体而言，我们的工作为进一步使用神经网络对人类决策进行认知建模带来了令人振奋的结果。



## **42. Unleashing the potential of prompt engineering in Large Language Models: a comprehensive review**

释放大型语言模型中提示工程的潜力：全面回顾 cs.CL

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2310.14735v5) [paper-pdf](http://arxiv.org/pdf/2310.14735v5)

**Authors**: Banghao Chen, Zhaofeng Zhang, Nicolas Langrené, Shengxin Zhu

**Abstract**: This comprehensive review delves into the pivotal role of prompt engineering in unleashing the capabilities of Large Language Models (LLMs). The development of Artificial Intelligence (AI), from its inception in the 1950s to the emergence of advanced neural networks and deep learning architectures, has made a breakthrough in LLMs, with models such as GPT-4o and Claude-3, and in Vision-Language Models (VLMs), with models such as CLIP and ALIGN. Prompt engineering is the process of structuring inputs, which has emerged as a crucial technique to maximize the utility and accuracy of these models. This paper explores both foundational and advanced methodologies of prompt engineering, including techniques such as self-consistency, chain-of-thought, and generated knowledge, which significantly enhance model performance. Additionally, it examines the prompt method of VLMs through innovative approaches such as Context Optimization (CoOp), Conditional Context Optimization (CoCoOp), and Multimodal Prompt Learning (MaPLe). Critical to this discussion is the aspect of AI security, particularly adversarial attacks that exploit vulnerabilities in prompt engineering. Strategies to mitigate these risks and enhance model robustness are thoroughly reviewed. The evaluation of prompt methods is also addressed, through both subjective and objective metrics, ensuring a robust analysis of their efficacy. This review also reflects the essential role of prompt engineering in advancing AI capabilities, providing a structured framework for future research and application.

摘要: 这篇全面的综述深入探讨了快速工程在释放大型语言模型(LLM)能力方面的关键作用。人工智能(AI)的发展，从20世纪50年代开始，到先进的神经网络和深度学习体系的出现，在LLMS方面取得了突破，有GPT-40和Claude-3等模型，以及视觉语言模型(VLMS)，有CLIP和ALIGN等模型。即时工程是对输入进行结构化的过程，它已成为最大化这些模型的实用性和准确性的关键技术。本文探讨了即时工程的基本方法和高级方法，包括自我一致性、思想链和生成知识等技术，这些技术显著提高了模型的性能。此外，它还通过诸如语境优化(COOP)、条件语境优化(CoCoOp)和多通道提示学习(Maple)等创新方法来研究虚拟学习模型的提示方法。对这一讨论至关重要的是人工智能安全方面，特别是利用即时工程中的漏洞进行的对抗性攻击。对缓解这些风险和增强模型稳健性的策略进行了彻底的回顾。还通过主观和客观指标对快速方法进行了评估，以确保对其有效性进行稳健的分析。这篇综述还反映了快速工程在推进人工智能能力方面的重要作用，为未来的研究和应用提供了一个结构化的框架。



## **43. LLM Detectors Still Fall Short of Real World: Case of LLM-Generated Short News-Like Posts**

LLM探测器仍然达不到现实世界：LLM生成的短新闻类帖子的案例 cs.CL

20 pages, 7 tables, 13 figures, under consideration for EMNLP

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2409.03291v1) [paper-pdf](http://arxiv.org/pdf/2409.03291v1)

**Authors**: Henrique Da Silva Gameiro, Andrei Kucharavy, Ljiljana Dolamic

**Abstract**: With the emergence of widely available powerful LLMs, disinformation generated by large Language Models (LLMs) has become a major concern. Historically, LLM detectors have been touted as a solution, but their effectiveness in the real world is still to be proven. In this paper, we focus on an important setting in information operations -- short news-like posts generated by moderately sophisticated attackers.   We demonstrate that existing LLM detectors, whether zero-shot or purpose-trained, are not ready for real-world use in that setting. All tested zero-shot detectors perform inconsistently with prior benchmarks and are highly vulnerable to sampling temperature increase, a trivial attack absent from recent benchmarks. A purpose-trained detector generalizing across LLMs and unseen attacks can be developed, but it fails to generalize to new human-written texts.   We argue that the former indicates domain-specific benchmarking is needed, while the latter suggests a trade-off between the adversarial evasion resilience and overfitting to the reference human text, with both needing evaluation in benchmarks and currently absent. We believe this suggests a re-consideration of current LLM detector benchmarking approaches and provides a dynamically extensible benchmark to allow it (https://github.com/Reliable-Information-Lab-HEVS/dynamic_llm_detector_benchmark).

摘要: 随着广泛使用的强大的LLM的出现，大型语言模型(LLM)产生的虚假信息已经成为一个主要关注的问题。从历史上看，LLM探测器一直被吹捧为一种解决方案，但它们在现实世界中的有效性仍有待证明。在这篇文章中，我们关注信息操作中的一个重要环境--由中等经验丰富的攻击者生成的类似新闻的短帖子。我们证明，现有的LLM探测器，无论是零炮还是专门训练的，都没有准备好在那种情况下用于现实世界。所有经过测试的零射击探测器的性能都与以前的基准不一致，并且非常容易受到采样温度升高的影响，这是最近的基准中所没有的一种轻微攻击。可以开发出一种专门训练的检测器，可以在LLMS和不可见攻击中推广，但它无法推广到新的人类书写的文本。我们认为，前者表明需要特定领域的基准测试，而后者则建议在对抗性回避韧性和对参考人类文本的过度匹配之间进行权衡，两者都需要在基准中进行评估，目前还没有。我们认为，这表明了对当前LLm探测器基准方法的重新考虑，并提供了一个动态可扩展的基准，以允许其(https://github.com/Reliable-Information-Lab-HEVS/dynamic_llm_detector_benchmark).



## **44. Revisiting Character-level Adversarial Attacks for Language Models**

重新审视语言模型的初级对抗攻击 cs.LG

Accepted in ICML 2024

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2405.04346v2) [paper-pdf](http://arxiv.org/pdf/2405.04346v2)

**Authors**: Elias Abad Rocamora, Yongtao Wu, Fanghui Liu, Grigorios G. Chrysos, Volkan Cevher

**Abstract**: Adversarial attacks in Natural Language Processing apply perturbations in the character or token levels. Token-level attacks, gaining prominence for their use of gradient-based methods, are susceptible to altering sentence semantics, leading to invalid adversarial examples. While character-level attacks easily maintain semantics, they have received less attention as they cannot easily adopt popular gradient-based methods, and are thought to be easy to defend. Challenging these beliefs, we introduce Charmer, an efficient query-based adversarial attack capable of achieving high attack success rate (ASR) while generating highly similar adversarial examples. Our method successfully targets both small (BERT) and large (Llama 2) models. Specifically, on BERT with SST-2, Charmer improves the ASR in 4.84% points and the USE similarity in 8% points with respect to the previous art. Our implementation is available in https://github.com/LIONS-EPFL/Charmer.

摘要: 自然语言处理中的对抗攻击对字符或令牌级别施加扰动。令牌级攻击因使用基于梯度的方法而变得越来越重要，很容易改变句子语义，从而导致无效的对抗性示例。虽然字符级攻击很容易维护语义，但它们受到的关注较少，因为它们不能轻易采用流行的基于梯度的方法，并且被认为很容易防御。基于这些信念，我们引入了Charmer，这是一种高效的基于查询的对抗性攻击，能够实现高攻击成功率（ASB），同时生成高度相似的对抗性示例。我们的方法成功地针对小型（BERT）和大型（Llama 2）模型。具体来说，在采用CST-2的BERT上，Charmer将ASB提高了4.84%，与之前的作品相比，USE相似性提高了8%。我们的实现可在https://github.com/LIONS-EPFL/Charmer上获取。



## **45. Alignment-Aware Model Extraction Attacks on Large Language Models**

对大型语言模型的对齐感知模型提取攻击 cs.CR

Source code: https://github.com/liangzid/alignmentExtraction

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2409.02718v1) [paper-pdf](http://arxiv.org/pdf/2409.02718v1)

**Authors**: Zi Liang, Qingqing Ye, Yanyun Wang, Sen Zhang, Yaxin Xiao, Ronghua Li, Jianliang Xu, Haibo Hu

**Abstract**: Model extraction attacks (MEAs) on large language models (LLMs) have received increasing research attention lately. Existing attack methods on LLMs inherit the extraction strategies from those designed for deep neural networks (DNNs) yet neglect the inconsistency of training tasks between MEA and LLMs' alignments. As such, they result in poor attack performances. To tackle this issue, we present Locality Reinforced Distillation (LoRD), a novel model extraction attack algorithm specifically for LLMs. In particular, we design a policy-gradient-style training task, which utilizes victim models' responses as a signal to guide the crafting of preference for the local model. Theoretical analysis has shown that i) LoRD's convergence procedure in MEAs is consistent with the alignments of LLMs, and ii) LoRD can reduce query complexity while mitigating watermark protection through exploration-based stealing. Extensive experiments on domain-specific extractions demonstrate the superiority of our method by examining the extraction of various state-of-the-art commercial LLMs.

摘要: 近年来，针对大型语言模型的模型提取攻击受到了越来越多的关注。现有的针对LLMS的攻击方法继承了针对深度神经网络(DNN)的提取策略，但忽略了MEA和LLMS对齐之间训练任务的不一致性。因此，他们的进攻表现很差。针对这一问题，我们提出了一种新的针对LLMS的模型提取攻击算法LOAD。特别是，我们设计了一个策略梯度式的训练任务，它利用受害者模型的反应作为一个信号来指导对局部模型的偏好的形成。理论分析表明：1)Lord在MEAS中的收敛过程与LLMS的比对是一致的；2)Lord在降低查询复杂度的同时，通过基于探测的窃取来减轻水印保护。对特定领域提取的大量实验通过检验各种最先进的商业LLM的提取来证明我们的方法的优越性。



## **46. Unveiling the Vulnerability of Private Fine-Tuning in Split-Based Frameworks for Large Language Models: A Bidirectionally Enhanced Attack**

揭露大型语言模型基于拆分的框架中的私人微调的漏洞：双向增强攻击 cs.CR

ACM Conference on Computer and Communications Security 2024 (CCS 24)

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2409.00960v2) [paper-pdf](http://arxiv.org/pdf/2409.00960v2)

**Authors**: Guanzhong Chen, Zhenghan Qin, Mingxin Yang, Yajie Zhou, Tao Fan, Tianyu Du, Zenglin Xu

**Abstract**: Recent advancements in pre-trained large language models (LLMs) have significantly influenced various domains. Adapting these models for specific tasks often involves fine-tuning (FT) with private, domain-specific data. However, privacy concerns keep this data undisclosed, and the computational demands for deploying LLMs pose challenges for resource-limited data holders. This has sparked interest in split learning (SL), a Model-as-a-Service (MaaS) paradigm that divides LLMs into smaller segments for distributed training and deployment, transmitting only intermediate activations instead of raw data. SL has garnered substantial interest in both industry and academia as it aims to balance user data privacy, model ownership, and resource challenges in the private fine-tuning of LLMs. Despite its privacy claims, this paper reveals significant vulnerabilities arising from the combination of SL and LLM-FT: the Not-too-far property of fine-tuning and the auto-regressive nature of LLMs. Exploiting these vulnerabilities, we propose Bidirectional Semi-white-box Reconstruction (BiSR), the first data reconstruction attack (DRA) designed to target both the forward and backward propagation processes of SL. BiSR utilizes pre-trained weights as prior knowledge, combining a learning-based attack with a bidirectional optimization-based approach for highly effective data reconstruction. Additionally, it incorporates a Noise-adaptive Mixture of Experts (NaMoE) model to enhance reconstruction performance under perturbation. We conducted systematic experiments on various mainstream LLMs and different setups, empirically demonstrating BiSR's state-of-the-art performance. Furthermore, we thoroughly examined three representative defense mechanisms, showcasing our method's capability to reconstruct private data even in the presence of these defenses.

摘要: 最近在预先训练的大型语言模型(LLM)方面的进展已经显著地影响了各个领域。针对特定任务调整这些模型通常涉及到使用特定领域的私有数据进行微调(FT)。然而，出于隐私考虑，这些数据不会被披露，而部署LLM的计算需求给资源有限的数据持有者带来了挑战。这引发了人们对拆分学习(SL)的兴趣，这是一种模型即服务(MAAS)范式，将LLM划分为较小的部分，用于分布式培训和部署，仅传输中间激活而不是原始数据。SL在工业界和学术界都引起了极大的兴趣，因为它旨在平衡用户数据隐私、模型所有权和LLM私下微调中的资源挑战。尽管声称其隐私，但本文揭示了SL和LLM-FT组合带来的重大漏洞：微调的不太远的特性和LLMS的自回归特性。利用这些漏洞，我们提出了双向半白盒重建(BiSR)，这是第一个针对SL的前向和后向传播过程的数据重建攻击(DRA)。BiSR利用预先训练的权值作为先验知识，将基于学习的攻击与基于双向优化的方法相结合，以实现高效的数据重建。此外，它还结合了噪声自适应专家混合(NaMoE)模型，以增强扰动下的重建性能。我们在各种主流的LLM和不同的设置上进行了系统的实验，实证地展示了BiSR的最先进的性能。此外，我们彻底检查了三种具有代表性的防御机制，展示了我们的方法即使在这些防御存在的情况下也能够重建私人数据的能力。



## **47. $\textit{MMJ-Bench}$: A Comprehensive Study on Jailbreak Attacks and Defenses for Vision Language Models**

$\texttit {MMJ-Bench}$：视觉语言模型越狱攻击和防御的综合研究 cs.CR

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2408.08464v2) [paper-pdf](http://arxiv.org/pdf/2408.08464v2)

**Authors**: Fenghua Weng, Yue Xu, Chengyan Fu, Wenjie Wang

**Abstract**: As deep learning advances, Large Language Models (LLMs) and their multimodal counterparts, Vision-Language Models (VLMs), have shown exceptional performance in many real-world tasks. However, VLMs face significant security challenges, such as jailbreak attacks, where attackers attempt to bypass the model's safety alignment to elicit harmful responses. The threat of jailbreak attacks on VLMs arises from both the inherent vulnerabilities of LLMs and the multiple information channels that VLMs process. While various attacks and defenses have been proposed, there is a notable gap in unified and comprehensive evaluations, as each method is evaluated on different dataset and metrics, making it impossible to compare the effectiveness of each method. To address this gap, we introduce \textit{MMJ-Bench}, a unified pipeline for evaluating jailbreak attacks and defense techniques for VLMs. Through extensive experiments, we assess the effectiveness of various attack methods against SoTA VLMs and evaluate the impact of defense mechanisms on both defense effectiveness and model utility for normal tasks. Our comprehensive evaluation contribute to the field by offering a unified and systematic evaluation framework and the first public-available benchmark for VLM jailbreak research. We also demonstrate several insightful findings that highlights directions for future studies.

摘要: 随着深度学习的深入，大型语言模型(LLM)及其多通道模型(Vision-Language Model，VLM)在许多实际任务中表现出了优异的性能。然而，VLM面临着重大的安全挑战，例如越狱攻击，攻击者试图绕过模型的安全对齐，以引发有害的响应。越狱攻击对VLMS的威胁既来自LLMS固有的脆弱性，也源于VLMS处理的多种信息渠道。虽然已经提出了各种攻击和防御方法，但在统一和综合评估方面存在显著差距，因为每种方法都是在不同的数据集和指标上进行评估，因此无法比较每种方法的有效性。为了弥补这一差距，我们引入了一个统一的管道，用于评估越狱攻击和针对VLM的防御技术。通过大量的实验，我们评估了各种攻击方法对SOTA VLMS的攻击效果，并评估了防御机制对正常任务的防御效果和模型效用的影响。我们的全面评估为越狱研究提供了统一和系统的评估框架和第一个公开可用的基准，从而为该领域做出了贡献。我们还展示了几个有洞察力的发现，这些发现突出了未来研究的方向。



## **48. LLM Defenses Are Not Robust to Multi-Turn Human Jailbreaks Yet**

LLM辩护对多次越狱还不强 cs.LG

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2408.15221v2) [paper-pdf](http://arxiv.org/pdf/2408.15221v2)

**Authors**: Nathaniel Li, Ziwen Han, Ian Steneker, Willow Primack, Riley Goodside, Hugh Zhang, Zifan Wang, Cristina Menghini, Summer Yue

**Abstract**: Recent large language model (LLM) defenses have greatly improved models' ability to refuse harmful queries, even when adversarially attacked. However, LLM defenses are primarily evaluated against automated adversarial attacks in a single turn of conversation, an insufficient threat model for real-world malicious use. We demonstrate that multi-turn human jailbreaks uncover significant vulnerabilities, exceeding 70% attack success rate (ASR) on HarmBench against defenses that report single-digit ASRs with automated single-turn attacks. Human jailbreaks also reveal vulnerabilities in machine unlearning defenses, successfully recovering dual-use biosecurity knowledge from unlearned models. We compile these results into Multi-Turn Human Jailbreaks (MHJ), a dataset of 2,912 prompts across 537 multi-turn jailbreaks. We publicly release MHJ alongside a compendium of jailbreak tactics developed across dozens of commercial red teaming engagements, supporting research towards stronger LLM defenses.

摘要: 最近的大型语言模型(LLM)防御极大地提高了模型拒绝有害查询的能力，即使在遭到恶意攻击时也是如此。然而，LLM防御主要是在单轮对话中针对自动对手攻击进行评估，这不足以构成现实世界恶意使用的威胁模型。我们证明，多轮人类越狱揭示了重大漏洞，针对使用自动化单轮攻击报告个位数ASR的防御系统，HarmB边上的攻击成功率(ASR)超过70%。人类越狱还揭示了机器遗忘防御的漏洞，成功地从未学习的模型中恢复了双重用途的生物安全知识。我们将这些结果汇编成多轮人类越狱(MHJ)，这是一个包含537次多轮越狱的2912个提示的数据集。我们公开发布了MHJ，以及在数十个商业红色团队交战中开发的越狱战术概要，支持对更强大的LLM防御的研究。



## **49. RACONTEUR: A Knowledgeable, Insightful, and Portable LLM-Powered Shell Command Explainer**

RACONTEur：一位知识渊博、富有洞察力且便携式的LLM驱动Shell命令解释者 cs.CR

Accepted by NDSS Symposium 2025. Please cite this paper as "Jiangyi  Deng, Xinfeng Li, Yanjiao Chen, Yijie Bai, Haiqin Weng, Yan Liu, Tao Wei,  Wenyuan Xu. RACONTEUR: A Knowledgeable, Insightful, and Portable LLM-Powered  Shell Command Explainer. In the 32nd Annual Network and Distributed System  Security Symposium (NDSS 2025)."

**SubmitDate**: 2024-09-03    [abs](http://arxiv.org/abs/2409.02074v1) [paper-pdf](http://arxiv.org/pdf/2409.02074v1)

**Authors**: Jiangyi Deng, Xinfeng Li, Yanjiao Chen, Yijie Bai, Haiqin Weng, Yan Liu, Tao Wei, Wenyuan Xu

**Abstract**: Malicious shell commands are linchpins to many cyber-attacks, but may not be easy to understand by security analysts due to complicated and often disguised code structures. Advances in large language models (LLMs) have unlocked the possibility of generating understandable explanations for shell commands. However, existing general-purpose LLMs suffer from a lack of expert knowledge and a tendency to hallucinate in the task of shell command explanation. In this paper, we present Raconteur, a knowledgeable, expressive and portable shell command explainer powered by LLM. Raconteur is infused with professional knowledge to provide comprehensive explanations on shell commands, including not only what the command does (i.e., behavior) but also why the command does it (i.e., purpose). To shed light on the high-level intent of the command, we also translate the natural-language-based explanation into standard technique & tactic defined by MITRE ATT&CK, the worldwide knowledge base of cybersecurity. To enable Raconteur to explain unseen private commands, we further develop a documentation retriever to obtain relevant information from complementary documentations to assist the explanation process. We have created a large-scale dataset for training and conducted extensive experiments to evaluate the capability of Raconteur in shell command explanation. The experiments verify that Raconteur is able to provide high-quality explanations and in-depth insight of the intent of the command.

摘要: 恶意的外壳命令是许多网络攻击的关键，但由于复杂且往往被伪装的代码结构，安全分析师可能不容易理解。大型语言模型(LLM)的进步使为外壳命令生成易于理解的解释成为可能。然而，现有的通用LLM存在缺乏专业知识和在解释外壳命令的任务中产生幻觉的倾向。在本文中，我们介绍了一个知识丰富、表达能力强、可移植的基于LLM的外壳命令解释器--raconteur。Raconteur具有丰富的专业知识，能够提供关于外壳命令的全面解释，不仅包括命令的作用(即行为)，还包括命令为什么这样做(即目的)。为了阐明该命令的高级意图，我们还将基于自然语言的解释转换为MITRE ATT&CK(全球网络安全知识库)定义的标准技术和策略。为了使raconteur能够解释看不见的私人命令，我们进一步开发了一个文档检索器，以从补充文档中获取相关信息，以帮助解释过程。我们创建了一个用于训练的大规模数据集，并进行了大量的实验来评估raconteur在解释外壳命令方面的能力。实验证明，raconteur能够提供高质量的解释和对命令意图的深入洞察。



## **50. FuzzCoder: Byte-level Fuzzing Test via Large Language Model**

FuzzCoder：通过大型语言模型进行字节级模糊测试 cs.CL

11 pages

**SubmitDate**: 2024-09-03    [abs](http://arxiv.org/abs/2409.01944v1) [paper-pdf](http://arxiv.org/pdf/2409.01944v1)

**Authors**: Liqun Yang, Jian Yang, Chaoren Wei, Guanglin Niu, Ge Zhang, Yunli Wang, Linzheng ChaI, Wanxu Xia, Hongcheng Guo, Shun Zhang, Jiaheng Liu, Yuwei Yin, Junran Peng, Jiaxin Ma, Liang Sun, Zhoujun Li

**Abstract**: Fuzzing is an important dynamic program analysis technique designed for finding vulnerabilities in complex software. Fuzzing involves presenting a target program with crafted malicious input to cause crashes, buffer overflows, memory errors, and exceptions. Crafting malicious inputs in an efficient manner is a difficult open problem and the best approaches often apply uniform random mutations to pre-existing valid inputs. In this work, we propose to adopt fine-tuned large language models (FuzzCoder) to learn patterns in the input files from successful attacks to guide future fuzzing explorations. Specifically, we develop a framework to leverage the code LLMs to guide the mutation process of inputs in fuzzing. The mutation process is formulated as the sequence-to-sequence modeling, where LLM receives a sequence of bytes and then outputs the mutated byte sequence. FuzzCoder is fine-tuned on the created instruction dataset (Fuzz-Instruct), where the successful fuzzing history is collected from the heuristic fuzzing tool. FuzzCoder can predict mutation locations and strategies locations in input files to trigger abnormal behaviors of the program. Experimental results show that FuzzCoder based on AFL (American Fuzzy Lop) gain significant improvements in terms of effective proportion of mutation (EPM) and number of crashes (NC) for various input formats including ELF, JPG, MP3, and XML.

摘要: 模糊是一种重要的动态程序分析技术，旨在发现复杂软件中的漏洞。Fuzing涉及向目标程序呈现精心编制的恶意输入，以导致崩溃、缓冲区溢出、内存错误和异常。以有效的方式创建恶意输入是一个困难的开放问题，最好的方法通常会对预先存在的有效输入应用统一的随机突变。在这项工作中，我们建议采用微调的大型语言模型(FuzzCoder)来从成功的攻击中学习输入文件中的模式，以指导未来的模糊探索。具体地说，我们开发了一个框架来利用代码LLMS来指导模糊化中输入的突变过程。突变过程被描述为序列到序列的建模，其中LLM接收字节序列，然后输出突变的字节序列。FuzzCoder在创建的指令数据集(Fuzz-Indict)上进行了微调，其中成功的模糊历史是从启发式模糊工具收集的。FuzzCoder可以预测输入文件中的突变位置和策略位置，从而触发程序的异常行为。实验结果表明，对于ELF、JPG、MP3和XML等多种输入格式，基于AFL(American FuzzLop)的FuzzCoder在有效变异比例(EPM)和崩溃次数(NC)方面都有明显的改善。



