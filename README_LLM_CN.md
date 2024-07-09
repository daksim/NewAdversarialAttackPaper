# Latest Large Language Model Attack Papers
**update at 2024-07-09 10:22:05**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Adaptive and robust watermark against model extraction attack**

抗模型提取攻击的自适应鲁棒水印 cs.CR

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2405.02365v2) [paper-pdf](http://arxiv.org/pdf/2405.02365v2)

**Authors**: Kaiyi Pang

**Abstract**: Large language models (LLMs) demonstrate general intelligence across a variety of machine learning tasks, thereby enhancing the commercial value of their intellectual property (IP). To protect this IP, model owners typically allow user access only in a black-box manner, however, adversaries can still utilize model extraction attacks to steal the model intelligence encoded in model generation. Watermarking technology offers a promising solution for defending against such attacks by embedding unique identifiers into the model-generated content. However, existing watermarking methods often compromise the quality of generated content due to heuristic alterations and lack robust mechanisms to counteract adversarial strategies, thus limiting their practicality in real-world scenarios. In this paper, we introduce an adaptive and robust watermarking method (named ModelShield) to protect the IP of LLMs. Our method incorporates a self-watermarking mechanism that allows LLMs to autonomously insert watermarks into their generated content to avoid the degradation of model content. We also propose a robust watermark detection mechanism capable of effectively identifying watermark signals under the interference of varying adversarial strategies. Besides, ModelShield is a plug-and-play method that does not require additional model training, enhancing its applicability in LLM deployments. Extensive evaluations on two real-world datasets and three LLMs demonstrate that our method surpasses existing methods in terms of defense effectiveness and robustness while significantly reducing the degradation of watermarking on the model-generated content.

摘要: 大型语言模型(LLM)在各种机器学习任务中展示了一般智能，从而提高了其知识产权(IP)的商业价值。为了保护这个IP，模型所有者通常只允许用户以黑盒方式访问，但是，攻击者仍然可以利用模型提取攻击来窃取模型生成中编码的模型情报。水印技术通过在模型生成的内容中嵌入唯一标识符，为防御此类攻击提供了一种很有前途的解决方案。然而，现有的水印方法往往会由于启发式修改而影响生成内容的质量，并且缺乏强大的机制来对抗对抗性策略，从而限制了它们在现实世界场景中的实用性。本文提出了一种自适应的稳健水印算法(ModelShield)来保护LLMS的IP地址。我们的方法结合了一种自水印机制，允许LLM自主地在其生成的内容中插入水印，以避免模型内容的降级。我们还提出了一种稳健的水印检测机制，能够在不同的对抗策略的干扰下有效地识别水印信号。此外，ModelShield是一种即插即用的方法，不需要额外的模型培训，增强了其在LLM部署中的适用性。在两个真实数据集和三个LLM上的广泛评估表明，我们的方法在防御有效性和稳健性方面优于现有方法，同时显着降低了水印对模型生成内容的退化。



## **2. Exploring the Adversarial Capabilities of Large Language Models**

探索大型语言模型的对抗能力 cs.AI

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2402.09132v4) [paper-pdf](http://arxiv.org/pdf/2402.09132v4)

**Authors**: Lukas Struppek, Minh Hieu Le, Dominik Hintersdorf, Kristian Kersting

**Abstract**: The proliferation of large language models (LLMs) has sparked widespread and general interest due to their strong language generation capabilities, offering great potential for both industry and research. While previous research delved into the security and privacy issues of LLMs, the extent to which these models can exhibit adversarial behavior remains largely unexplored. Addressing this gap, we investigate whether common publicly available LLMs have inherent capabilities to perturb text samples to fool safety measures, so-called adversarial examples resp.~attacks. More specifically, we investigate whether LLMs are inherently able to craft adversarial examples out of benign samples to fool existing safe rails. Our experiments, which focus on hate speech detection, reveal that LLMs succeed in finding adversarial perturbations, effectively undermining hate speech detection systems. Our findings carry significant implications for (semi-)autonomous systems relying on LLMs, highlighting potential challenges in their interaction with existing systems and safety measures.

摘要: 大型语言模型因其强大的语言生成能力而引起了广泛的关注，为工业和研究提供了巨大的潜力。虽然之前的研究已经深入研究了LLMS的安全和隐私问题，但这些模型在多大程度上可以表现出敌对行为，仍然很大程度上还没有被探索。针对这一差距，我们调查了常见的公开可用的LLM是否具有固有的能力来扰乱文本样本以愚弄安全措施，即所谓的对抗性示例攻击。更具体地说，我们调查LLM是否天生就能够从良性样本中制作敌意示例，以愚弄现有的安全Rail。我们的实验集中在仇恨语音检测上，实验表明，LLMS成功地发现了敌意扰动，有效地破坏了仇恨语音检测系统。我们的发现对依赖LLMS的(半)自治系统具有重大影响，突显了它们与现有系统和安全措施相互作用的潜在挑战。



## **3. TrojanRAG: Retrieval-Augmented Generation Can Be Backdoor Driver in Large Language Models**

TrojanRAG：检索增强生成可以成为大型语言模型中的后门驱动程序 cs.CR

19 pages, 14 figures, 4 tables

**SubmitDate**: 2024-07-07    [abs](http://arxiv.org/abs/2405.13401v4) [paper-pdf](http://arxiv.org/pdf/2405.13401v4)

**Authors**: Pengzhou Cheng, Yidong Ding, Tianjie Ju, Zongru Wu, Wei Du, Ping Yi, Zhuosheng Zhang, Gongshen Liu

**Abstract**: Large language models (LLMs) have raised concerns about potential security threats despite performing significantly in Natural Language Processing (NLP). Backdoor attacks initially verified that LLM is doing substantial harm at all stages, but the cost and robustness have been criticized. Attacking LLMs is inherently risky in security review, while prohibitively expensive. Besides, the continuous iteration of LLMs will degrade the robustness of backdoors. In this paper, we propose TrojanRAG, which employs a joint backdoor attack in the Retrieval-Augmented Generation, thereby manipulating LLMs in universal attack scenarios. Specifically, the adversary constructs elaborate target contexts and trigger sets. Multiple pairs of backdoor shortcuts are orthogonally optimized by contrastive learning, thus constraining the triggering conditions to a parameter subspace to improve the matching. To improve the recall of the RAG for the target contexts, we introduce a knowledge graph to construct structured data to achieve hard matching at a fine-grained level. Moreover, we normalize the backdoor scenarios in LLMs to analyze the real harm caused by backdoors from both attackers' and users' perspectives and further verify whether the context is a favorable tool for jailbreaking models. Extensive experimental results on truthfulness, language understanding, and harmfulness show that TrojanRAG exhibits versatility threats while maintaining retrieval capabilities on normal queries.

摘要: 尽管大型语言模型(LLM)在自然语言处理(NLP)中表现出色，但仍引发了人们对潜在安全威胁的担忧。后门攻击最初证实了LLM在所有阶段都在造成实质性的危害，但其成本和健壮性受到了批评。在安全审查中，攻击LLMS固有的风险，同时代价高得令人望而却步。此外，LLMS的连续迭代会降低后门的健壮性。在本文中，我们提出了TrojanRAG，它在检索-增强生成中使用联合后门攻击，从而在通用攻击场景下操纵LLMS。具体地说，对手构建了精心设计的目标上下文和触发集。通过对比学习对多对后门捷径进行正交化优化，从而将触发条件约束到一个参数子空间以提高匹配性。为了提高RAG对目标上下文的查全率，我们引入了知识图来构建结构化数据，以实现细粒度的硬匹配。此外，我们对LLMS中的后门场景进行了规范化，从攻击者和用户的角度分析了后门造成的真实危害，并进一步验证了上下文是否为越狱模型的有利工具。在真实性、语言理解和危害性方面的大量实验结果表明，TrojanRAG在保持对正常查询的检索能力的同时，表现出通用性威胁。



## **4. BadCLM: Backdoor Attack in Clinical Language Models for Electronic Health Records**

BadCLM：电子健康记录临床语言模型中的后门攻击 cs.CL

AMIA 2024

**SubmitDate**: 2024-07-06    [abs](http://arxiv.org/abs/2407.05213v1) [paper-pdf](http://arxiv.org/pdf/2407.05213v1)

**Authors**: Weimin Lyu, Zexin Bi, Fusheng Wang, Chao Chen

**Abstract**: The advent of clinical language models integrated into electronic health records (EHR) for clinical decision support has marked a significant advancement, leveraging the depth of clinical notes for improved decision-making. Despite their success, the potential vulnerabilities of these models remain largely unexplored. This paper delves into the realm of backdoor attacks on clinical language models, introducing an innovative attention-based backdoor attack method, BadCLM (Bad Clinical Language Models). This technique clandestinely embeds a backdoor within the models, causing them to produce incorrect predictions when a pre-defined trigger is present in inputs, while functioning accurately otherwise. We demonstrate the efficacy of BadCLM through an in-hospital mortality prediction task with MIMIC III dataset, showcasing its potential to compromise model integrity. Our findings illuminate a significant security risk in clinical decision support systems and pave the way for future endeavors in fortifying clinical language models against such vulnerabilities.

摘要: 集成到电子健康记录(EHR)中用于临床决策支持的临床语言模型的出现标志着一项重大进步，它利用临床笔记的深度来改进决策。尽管取得了成功，但这些模型的潜在漏洞在很大程度上仍未被发掘。针对临床语言模型的后门攻击问题，提出了一种新的基于注意力的后门攻击方法BadCLM(Bad Clinic Language Model)。这种技术在模型中秘密地嵌入了一个后门，当输入中存在预定义的触发器时，导致模型产生错误的预测，而在其他情况下则准确地发挥作用。我们通过使用MIMIC III数据集的医院内死亡率预测任务，展示了BadCLM的有效性，展示了其损害模型完整性的潜力。我们的发现揭示了临床决策支持系统中的一个重大安全风险，并为未来加强临床语言模型以抵御此类漏洞的努力铺平了道路。



## **5. LLMCloudHunter: Harnessing LLMs for Automated Extraction of Detection Rules from Cloud-Based CTI**

LLMCloudHunter：利用LLM从基于云的RTI自动提取检测规则 cs.CR

**SubmitDate**: 2024-07-06    [abs](http://arxiv.org/abs/2407.05194v1) [paper-pdf](http://arxiv.org/pdf/2407.05194v1)

**Authors**: Yuval Schwartz, Lavi Benshimol, Dudu Mimran, Yuval Elovici, Asaf Shabtai

**Abstract**: As the number and sophistication of cyber attacks have increased, threat hunting has become a critical aspect of active security, enabling proactive detection and mitigation of threats before they cause significant harm. Open-source cyber threat intelligence (OS-CTI) is a valuable resource for threat hunters, however, it often comes in unstructured formats that require further manual analysis. Previous studies aimed at automating OSCTI analysis are limited since (1) they failed to provide actionable outputs, (2) they did not take advantage of images present in OSCTI sources, and (3) they focused on on-premises environments, overlooking the growing importance of cloud environments. To address these gaps, we propose LLMCloudHunter, a novel framework that leverages large language models (LLMs) to automatically generate generic-signature detection rule candidates from textual and visual OSCTI data. We evaluated the quality of the rules generated by the proposed framework using 12 annotated real-world cloud threat reports. The results show that our framework achieved a precision of 92% and recall of 98% for the task of accurately extracting API calls made by the threat actor and a precision of 99% with a recall of 98% for IoCs. Additionally, 99.18% of the generated detection rule candidates were successfully compiled and converted into Splunk queries.

摘要: 随着网络攻击的数量和复杂性的增加，威胁追捕已经成为主动安全的一个关键方面，能够在威胁造成重大伤害之前主动检测和缓解威胁。开源网络威胁情报(OS-CTI)对于威胁猎人来说是一种宝贵的资源，然而，它通常是非结构化的格式，需要进一步的手动分析。以前旨在自动化OSCTI分析的研究是有限的，因为(1)它们未能提供可操作的输出，(2)它们没有利用OSCTI源中存在的图像，以及(3)它们专注于内部部署环境，忽视了云环境日益增长的重要性。为了弥补这些不足，我们提出了LLMCloudHunter，这是一个新的框架，它利用大型语言模型(LLM)来从文本和可视OSCTI数据自动生成通用签名检测规则候选。我们使用12个带注释的真实云威胁报告评估了所提出的框架生成的规则的质量。实验结果表明，该框架对于准确提取威胁行为人的API调用的准确率为92%，召回率为98%，对于IOC的准确率为99%，召回率为98%。此外，99.18%的生成检测规则候选被成功编译并转换为Splunk查询。



## **6. On Evaluating The Performance of Watermarked Machine-Generated Texts Under Adversarial Attacks**

关于评估带有水印的机器生成文本在对抗性攻击下的性能 cs.CR

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2407.04794v1) [paper-pdf](http://arxiv.org/pdf/2407.04794v1)

**Authors**: Zesen Liu, Tianshuo Cong, Xinlei He, Qi Li

**Abstract**: Large Language Models (LLMs) excel in various applications, including text generation and complex tasks. However, the misuse of LLMs raises concerns about the authenticity and ethical implications of the content they produce, such as deepfake news, academic fraud, and copyright infringement. Watermarking techniques, which embed identifiable markers in machine-generated text, offer a promising solution to these issues by allowing for content verification and origin tracing. Unfortunately, the robustness of current LLM watermarking schemes under potential watermark removal attacks has not been comprehensively explored.   In this paper, to fill this gap, we first systematically comb the mainstream watermarking schemes and removal attacks on machine-generated texts, and then we categorize them into pre-text (before text generation) and post-text (after text generation) classes so that we can conduct diversified analyses. In our experiments, we evaluate eight watermarks (five pre-text, three post-text) and twelve attacks (two pre-text, ten post-text) across 87 scenarios. Evaluation results indicate that (1) KGW and Exponential watermarks offer high text quality and watermark retention but remain vulnerable to most attacks; (2) Post-text attacks are found to be more efficient and practical than pre-text attacks; (3) Pre-text watermarks are generally more imperceptible, as they do not alter text fluency, unlike post-text watermarks; (4) Additionally, combined attack methods can significantly increase effectiveness, highlighting the need for more robust watermarking solutions. Our study underscores the vulnerabilities of current techniques and the necessity for developing more resilient schemes.

摘要: 大型语言模型(LLM)在各种应用中表现出色，包括文本生成和复杂任务。然而，LLMS的滥用引发了人们对它们产生的内容的真实性和伦理影响的担忧，例如深度假新闻、学术欺诈和侵犯版权。在机器生成的文本中嵌入可识别标记的水印技术，通过允许内容验证和来源追踪，为这些问题提供了一种有前途的解决方案。为了填补这一空白，本文首先对主流的机器生成文本水印算法和去除攻击进行了系统的梳理，然后将其分为前文本类(文本生成前)和后文本类(文本生成后)，以便进行多样化的分析。在我们的实验中，我们评估了87个场景中的8个水印(5个前置文本，3个后置文本)和12个攻击(2个前置文本，10个后置文本)。评估结果表明：(1)KGW和指数水印具有高的文本质量和水印保留率，但仍然容易受到大多数攻击；(2)后文本攻击被发现比前文本攻击更有效和实用；(3)前文本水印通常更不可察觉，因为它们不像后文本水印那样改变文本的流畅性；(4)此外，组合攻击方法可以显著提高攻击效果，突出了对更健壮的水印解决方案的需求。我们的研究强调了当前技术的脆弱性，以及开发更具弹性的方案的必要性。



## **7. Controlling Whisper: Universal Acoustic Adversarial Attacks to Control Speech Foundation Models**

控制耳语：控制语音基础模型的通用声学对抗攻击 cs.SD

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2407.04482v1) [paper-pdf](http://arxiv.org/pdf/2407.04482v1)

**Authors**: Vyas Raina, Mark Gales

**Abstract**: Speech enabled foundation models, either in the form of flexible speech recognition based systems or audio-prompted large language models (LLMs), are becoming increasingly popular. One of the interesting aspects of these models is their ability to perform tasks other than automatic speech recognition (ASR) using an appropriate prompt. For example, the OpenAI Whisper model can perform both speech transcription and speech translation. With the development of audio-prompted LLMs there is the potential for even greater control options. In this work we demonstrate that with this greater flexibility the systems can be susceptible to model-control adversarial attacks. Without any access to the model prompt it is possible to modify the behaviour of the system by appropriately changing the audio input. To illustrate this risk, we demonstrate that it is possible to prepend a short universal adversarial acoustic segment to any input speech signal to override the prompt setting of an ASR foundation model. Specifically, we successfully use a universal adversarial acoustic segment to control Whisper to always perform speech translation, despite being set to perform speech transcription. Overall, this work demonstrates a new form of adversarial attack on multi-tasking speech enabled foundation models that needs to be considered prior to the deployment of this form of model.

摘要: 以灵活的基于语音识别的系统或音频提示的大型语言模型(LLM)的形式启用语音的基础模型正变得越来越受欢迎。这些模型的一个有趣方面是，它们能够使用适当的提示执行自动语音识别(ASR)以外的任务。例如，OpenAI Whisper模型可以执行语音转录和语音翻译。随着音频提示LLMS的发展，有可能出现更大的控制选项。在这项工作中，我们证明了有了这种更大的灵活性，系统可以容易受到模型控制的对抗性攻击。在不访问模型提示的情况下，可以通过适当地改变音频输入来修改系统的行为。为了说明这一风险，我们证明了有可能在任何输入语音信号之前添加一个简短的通用对抗性声学片段，以覆盖ASR基础模型的提示设置。具体地说，我们成功地使用了一个通用的对抗性声学段来控制Whisper始终执行语音翻译，尽管被设置为执行语音转录。总体而言，这项工作展示了一种对多任务语音启用的基础模型的新形式的对抗性攻击，在部署这种形式的模型之前需要考虑这种形式。



## **8. Waterfall: Framework for Robust and Scalable Text Watermarking**

瀑布：稳健且可扩展的文本水印框架 cs.CR

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2407.04411v1) [paper-pdf](http://arxiv.org/pdf/2407.04411v1)

**Authors**: Gregory Kang Ruey Lau, Xinyuan Niu, Hieu Dao, Jiangwei Chen, Chuan-Sheng Foo, Bryan Kian Hsiang Low

**Abstract**: Protecting intellectual property (IP) of text such as articles and code is increasingly important, especially as sophisticated attacks become possible, such as paraphrasing by large language models (LLMs) or even unauthorized training of LLMs on copyrighted text to infringe such IP. However, existing text watermarking methods are not robust enough against such attacks nor scalable to millions of users for practical implementation. In this paper, we propose Waterfall, the first training-free framework for robust and scalable text watermarking applicable across multiple text types (e.g., articles, code) and languages supportable by LLMs, for general text and LLM data provenance. Waterfall comprises several key innovations, such as being the first to use LLM as paraphrasers for watermarking along with a novel combination of techniques that are surprisingly effective in achieving robust verifiability and scalability. We empirically demonstrate that Waterfall achieves significantly better scalability, robust verifiability, and computational efficiency compared to SOTA article-text watermarking methods, and also showed how it could be directly applied to the watermarking of code.

摘要: 保护文章和代码等文本的知识产权(IP)越来越重要，特别是在可能进行复杂攻击的情况下，例如利用大型语言模型(LLM)进行释义，甚至未经授权对受版权保护的文本进行LLM培训，以侵犯此类IP。然而，现有的文本水印方法对此类攻击不够健壮，也不能扩展到数百万用户进行实际实现。在本文中，我们提出了瀑布，这是第一个无训练的文本水印框架，适用于LLMS支持的多种文本类型(例如，文章、代码)和语言，用于一般文本和LLM数据来源。瀑布由几项关键创新组成，例如第一个使用LLM作为水印解释程序，以及在实现强大的可验证性和可扩展性方面出人意料地有效的技术组合。实验证明，与SOTA的文章-文本水印方法相比，瀑布算法具有更好的可扩展性、健壮的可验证性和计算效率，并且可以直接应用于代码的水印。



## **9. Jailbreak Attacks and Defenses Against Large Language Models: A Survey**

针对大型语言模型的越狱攻击和防御：调查 cs.CR

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2407.04295v1) [paper-pdf](http://arxiv.org/pdf/2407.04295v1)

**Authors**: Sibo Yi, Yule Liu, Zhen Sun, Tianshuo Cong, Xinlei He, Jiaxing Song, Ke Xu, Qi Li

**Abstract**: Large Language Models (LLMs) have performed exceptionally in various text-generative tasks, including question answering, translation, code completion, etc. However, the over-assistance of LLMs has raised the challenge of "jailbreaking", which induces the model to generate malicious responses against the usage policy and society by designing adversarial prompts. With the emergence of jailbreak attack methods exploiting different vulnerabilities in LLMs, the corresponding safety alignment measures are also evolving. In this paper, we propose a comprehensive and detailed taxonomy of jailbreak attack and defense methods. For instance, the attack methods are divided into black-box and white-box attacks based on the transparency of the target model. Meanwhile, we classify defense methods into prompt-level and model-level defenses. Additionally, we further subdivide these attack and defense methods into distinct sub-classes and present a coherent diagram illustrating their relationships. We also conduct an investigation into the current evaluation methods and compare them from different perspectives. Our findings aim to inspire future research and practical implementations in safeguarding LLMs against adversarial attacks. Above all, although jailbreak remains a significant concern within the community, we believe that our work enhances the understanding of this domain and provides a foundation for developing more secure LLMs.

摘要: 大型语言模型(LLMS)在问答、翻译、代码补全等文本生成任务中表现出色。然而，LLMS的过度协助带来了越狱的挑战，这导致该模型通过设计敌意提示来生成针对使用策略和社会的恶意响应。随着利用LLMS中不同漏洞的越狱攻击方法的出现，相应的安全对齐措施也在不断发展。在本文中，我们提出了一个全面和详细的分类越狱攻防方法。例如，根据目标模型的透明性，将攻击方法分为黑盒攻击和白盒攻击。同时，我们将防御方法分为提示级防御和模型级防御。此外，我们还将这些攻击和防御方法进一步细分为不同的子类，并提供了一个连贯的图来说明它们之间的关系。我们还对现有的评估方法进行了调查，并从不同的角度对它们进行了比较。我们的发现旨在启发未来在保护LLM免受对手攻击方面的研究和实际实现。最重要的是，尽管越狱在社区中仍然是一个重要的问题，但我们相信我们的工作增进了对这个领域的了解，并为开发更安全的LLM提供了基础。



## **10. Defending Jailbreak Prompts via In-Context Adversarial Game**

通过上下文对抗游戏为越狱辩护 cs.LG

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2402.13148v2) [paper-pdf](http://arxiv.org/pdf/2402.13148v2)

**Authors**: Yujun Zhou, Yufei Han, Haomin Zhuang, Kehan Guo, Zhenwen Liang, Hongyan Bao, Xiangliang Zhang

**Abstract**: Large Language Models (LLMs) demonstrate remarkable capabilities across diverse applications. However, concerns regarding their security, particularly the vulnerability to jailbreak attacks, persist. Drawing inspiration from adversarial training in deep learning and LLM agent learning processes, we introduce the In-Context Adversarial Game (ICAG) for defending against jailbreaks without the need for fine-tuning. ICAG leverages agent learning to conduct an adversarial game, aiming to dynamically extend knowledge to defend against jailbreaks. Unlike traditional methods that rely on static datasets, ICAG employs an iterative process to enhance both the defense and attack agents. This continuous improvement process strengthens defenses against newly generated jailbreak prompts. Our empirical studies affirm ICAG's efficacy, where LLMs safeguarded by ICAG exhibit significantly reduced jailbreak success rates across various attack scenarios. Moreover, ICAG demonstrates remarkable transferability to other LLMs, indicating its potential as a versatile defense mechanism.

摘要: 大型语言模型(LLM)在不同的应用程序中展示了卓越的功能。然而，对他们的安全，特别是对越狱攻击的脆弱性的担忧依然存在。从深度学习和LLM代理学习过程中的对抗性训练中获得灵感，我们引入了无需微调的上下文对抗性游戏(ICAG)来防御越狱。ICAG利用代理学习进行对抗性游戏，旨在动态扩展知识来防御越狱。与依赖静态数据集的传统方法不同，ICAG采用迭代过程来增强防御和攻击代理。这一不断改进的过程加强了对新生成的越狱提示的防御。我们的经验研究肯定了ICAG的有效性，在不同的攻击场景中，由ICAG保护的LLM显示出显著降低的越狱成功率。此外，ICAG表现出显著的可转移性，表明其作为一种多功能防御机制的潜力。



## **11. Defense Against Syntactic Textual Backdoor Attacks with Token Substitution**

利用令牌替换防御语法文本后门攻击 cs.CL

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2407.04179v1) [paper-pdf](http://arxiv.org/pdf/2407.04179v1)

**Authors**: Xinglin Li, Xianwen He, Yao Li, Minhao Cheng

**Abstract**: Textual backdoor attacks present a substantial security risk to Large Language Models (LLM). It embeds carefully chosen triggers into a victim model at the training stage, and makes the model erroneously predict inputs containing the same triggers as a certain class. Prior backdoor defense methods primarily target special token-based triggers, leaving syntax-based triggers insufficiently addressed. To fill this gap, this paper proposes a novel online defense algorithm that effectively counters syntax-based as well as special token-based backdoor attacks. The algorithm replaces semantically meaningful words in sentences with entirely different ones but preserves the syntactic templates or special tokens, and then compares the predicted labels before and after the substitution to determine whether a sentence contains triggers. Experimental results confirm the algorithm's performance against these two types of triggers, offering a comprehensive defense strategy for model integrity.

摘要: 文本后门攻击给大型语言模型（LLM）带来了巨大的安全风险。它在训练阶段将精心选择的触发器嵌入到受害者模型中，并使模型错误地预测包含与某个类别相同触发器的输入。先前的后门防御方法主要针对特殊的基于令牌的触发器，而基于语法的触发器则没有得到充分解决。为了填补这一空白，本文提出了一种新颖的在线防御算法，可以有效地对抗基于语法以及特殊的基于令牌的后门攻击。该算法用完全不同的单词替换句子中具有语义意义的单词，但保留语法模板或特殊标记，然后比较替换前后的预测标签，以确定句子是否包含触发器。实验结果证实了该算法针对这两种类型触发器的性能，为模型完整性提供了全面的防御策略。



## **12. Securing Multi-turn Conversational Language Models Against Distributed Backdoor Triggers**

保护多轮对话语言模型免受分布式后门触发器的影响 cs.CL

Submitted to EMNLP 2024

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2407.04151v1) [paper-pdf](http://arxiv.org/pdf/2407.04151v1)

**Authors**: Terry Tong, Jiashu Xu, Qin Liu, Muhao Chen

**Abstract**: The security of multi-turn conversational large language models (LLMs) is understudied despite it being one of the most popular LLM utilization. Specifically, LLMs are vulnerable to data poisoning backdoor attacks, where an adversary manipulates the training data to cause the model to output malicious responses to predefined triggers. Specific to the multi-turn dialogue setting, LLMs are at the risk of even more harmful and stealthy backdoor attacks where the backdoor triggers may span across multiple utterances, giving lee-way to context-driven attacks. In this paper, we explore a novel distributed backdoor trigger attack that serves to be an extra tool in an adversary's toolbox that can interface with other single-turn attack strategies in a plug and play manner. Results on two representative defense mechanisms indicate that distributed backdoor triggers are robust against existing defense strategies which are designed for single-turn user-model interactions, motivating us to propose a new defense strategy for the multi-turn dialogue setting that is more challenging. To this end, we also explore a novel contrastive decoding based defense that is able to mitigate the backdoor with a low computational tradeoff.

摘要: 多话轮会话大语言模型(LLMS)是目前应用最广泛的大语言模型之一，但其安全性仍未得到充分的研究。具体地说，LLM容易受到数据中毒后门攻击，在这种攻击中，敌手操纵训练数据，使模型输出对预定义触发器的恶意响应。具体到多轮对话设置，LLM面临着更具危害性和更隐蔽的后门攻击的风险，其中后门触发可能跨越多个话语，使Lee-way成为上下文驱动的攻击。在本文中，我们探索了一种新型的分布式后门触发攻击，它是对手工具箱中的一个额外工具，可以以即插即用的方式与其他单轮攻击策略对接。在两种典型防御机制上的结果表明，分布式后门触发器对现有的针对单回合用户-模型交互的防御策略具有较强的健壮性，这促使我们提出了一种新的针对更具挑战性的多回合对话环境的防御策略。为此，我们还探索了一种新的基于对比解码的防御方案，该方案能够以较低的计算代价缓解后门问题。



## **13. Is LLM-as-a-Judge Robust? Investigating Universal Adversarial Attacks on Zero-shot LLM Assessment**

法学硕士作为法官稳健吗？调查零射击LLM评估中的普遍对抗攻击 cs.CL

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2402.14016v2) [paper-pdf](http://arxiv.org/pdf/2402.14016v2)

**Authors**: Vyas Raina, Adian Liusie, Mark Gales

**Abstract**: Large Language Models (LLMs) are powerful zero-shot assessors used in real-world situations such as assessing written exams and benchmarking systems. Despite these critical applications, no existing work has analyzed the vulnerability of judge-LLMs to adversarial manipulation. This work presents the first study on the adversarial robustness of assessment LLMs, where we demonstrate that short universal adversarial phrases can be concatenated to deceive judge LLMs to predict inflated scores. Since adversaries may not know or have access to the judge-LLMs, we propose a simple surrogate attack where a surrogate model is first attacked, and the learned attack phrase then transferred to unknown judge-LLMs. We propose a practical algorithm to determine the short universal attack phrases and demonstrate that when transferred to unseen models, scores can be drastically inflated such that irrespective of the assessed text, maximum scores are predicted. It is found that judge-LLMs are significantly more susceptible to these adversarial attacks when used for absolute scoring, as opposed to comparative assessment. Our findings raise concerns on the reliability of LLM-as-a-judge methods, and emphasize the importance of addressing vulnerabilities in LLM assessment methods before deployment in high-stakes real-world scenarios.

摘要: 大型语言模型(LLM)是用于评估笔试和基准系统等真实世界情况的强大的零分评价器。尽管有这些关键的应用，但现有的工作还没有分析JUSTER-LLMS对对抗操纵的脆弱性。这项工作首次研究了评估LLMS的对抗稳健性，其中我们证明了短的通用对抗性短语可以被串联起来欺骗裁判LLMS来预测夸大的分数。由于敌手可能不知道或无法访问裁判LLMS，我们提出了一种简单的代理攻击，首先攻击代理模型，然后将学习到的攻击短语传输到未知的JUSTER-LLMS。我们提出了一个实用的算法来确定简短的通用攻击短语，并证明了当转移到看不见的模型时，分数可以被大幅夸大，以至于无论评估的文本是什么，都可以预测最高分数。研究发现，与比较评估相比，JUSICE-LLM在用于绝对评分时更容易受到这些对抗性攻击。我们的发现引起了人们对LLM作为判断方法的可靠性的担忧，并强调了在高风险的现实世界场景中部署之前解决LLM评估方法中的漏洞的重要性。



## **14. DART: Deep Adversarial Automated Red Teaming for LLM Safety**

DART：深度对抗自动化红色团队，确保LLM安全 cs.CR

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2407.03876v1) [paper-pdf](http://arxiv.org/pdf/2407.03876v1)

**Authors**: Bojian Jiang, Yi Jing, Tianhao Shen, Qing Yang, Deyi Xiong

**Abstract**: Manual Red teaming is a commonly-used method to identify vulnerabilities in large language models (LLMs), which, is costly and unscalable. In contrast, automated red teaming uses a Red LLM to automatically generate adversarial prompts to the Target LLM, offering a scalable way for safety vulnerability detection. However, the difficulty of building a powerful automated Red LLM lies in the fact that the safety vulnerabilities of the Target LLM are dynamically changing with the evolution of the Target LLM. To mitigate this issue, we propose a Deep Adversarial Automated Red Teaming (DART) framework in which the Red LLM and Target LLM are deeply and dynamically interacting with each other in an iterative manner. In each iteration, in order to generate successful attacks as many as possible, the Red LLM not only takes into account the responses from the Target LLM, but also adversarially adjust its attacking directions by monitoring the global diversity of generated attacks across multiple iterations. Simultaneously, to explore dynamically changing safety vulnerabilities of the Target LLM, we allow the Target LLM to enhance its safety via an active learning based data selection mechanism. Experimential results demonstrate that DART significantly reduces the safety risk of the target LLM. For human evaluation on Anthropic Harmless dataset, compared to the instruction-tuning target LLM, DART eliminates the violation risks by 53.4\%. We will release the datasets and codes of DART soon.

摘要: 手动Red Teaming是一种常用的识别大型语言模型(LLM)漏洞的方法，该方法代价高昂且不可扩展。相比之下，自动红色团队使用Red LLM自动生成针对Target LLM的敌意提示，为安全漏洞检测提供了一种可扩展的方法。然而，构建功能强大的自动Red LLM的难点在于，目标LLM的安全漏洞随着目标LLM的演化而动态变化。为了缓解这一问题，我们提出了一个深度对抗性自动红团队(DART)框架，在该框架中，Red LLM和Target LLM以迭代的方式进行深度和动态的交互。在每一次迭代中，为了生成尽可能多的成功攻击，Red LLM不仅考虑目标LLM的响应，而且通过监控生成的攻击在多个迭代中的全局多样性来相反地调整其攻击方向。同时，为了探索动态变化的安全漏洞，我们允许目标LLM通过一种基于主动学习的数据选择机制来增强其安全性。实验结果表明，DART显著降低了目标LLM的安全风险。对于人类无害数据集的人工评估，与指令调优目标LLM相比，DART消除了53.4%的违规风险。我们将很快公布DART的数据集和代码。



## **15. Jailbreaking Black Box Large Language Models in Twenty Queries**

二十分钟内越狱黑匣子大型语言模型 cs.LG

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2310.08419v3) [paper-pdf](http://arxiv.org/pdf/2310.08419v3)

**Authors**: Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, Eric Wong

**Abstract**: There is growing interest in ensuring that large language models (LLMs) align with human values. However, the alignment of such models is vulnerable to adversarial jailbreaks, which coax LLMs into overriding their safety guardrails. The identification of these vulnerabilities is therefore instrumental in understanding inherent weaknesses and preventing future misuse. To this end, we propose Prompt Automatic Iterative Refinement (PAIR), an algorithm that generates semantic jailbreaks with only black-box access to an LLM. PAIR -- which is inspired by social engineering attacks -- uses an attacker LLM to automatically generate jailbreaks for a separate targeted LLM without human intervention. In this way, the attacker LLM iteratively queries the target LLM to update and refine a candidate jailbreak. Empirically, PAIR often requires fewer than twenty queries to produce a jailbreak, which is orders of magnitude more efficient than existing algorithms. PAIR also achieves competitive jailbreaking success rates and transferability on open and closed-source LLMs, including GPT-3.5/4, Vicuna, and Gemini.

摘要: 人们对确保大型语言模型(LLM)与人类价值观保持一致的兴趣与日俱增。然而，这类模型的调整很容易受到对抗性越狱的影响，这会诱使低收入国家凌驾于他们的安全护栏之上。因此，确定这些漏洞有助于了解固有的弱点并防止今后的滥用。为此，我们提出了即时自动迭代求精(Pair)，这是一种仅通过黑盒访问LLM来生成语义越狱的算法。Pair受到社会工程攻击的启发，它使用攻击者LLM自动为单独的目标LLM生成越狱，而无需人工干预。通过这种方式，攻击者LLM迭代地查询目标LLM以更新和改进候选越狱。根据经验，Pair通常只需要不到20次查询就可以产生越狱，这比现有算法的效率高出几个数量级。Pair还在开放和封闭源代码的LLM上实现了具有竞争力的越狱成功率和可转移性，包括GPT-3.5/4、维库纳和双子座。



## **16. JailBreakV-28K: A Benchmark for Assessing the Robustness of MultiModal Large Language Models against Jailbreak Attacks**

JailBreakV-28 K：评估多模式大型语言模型对抗越狱攻击的稳健性的基准 cs.CR

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2404.03027v3) [paper-pdf](http://arxiv.org/pdf/2404.03027v3)

**Authors**: Weidi Luo, Siyuan Ma, Xiaogeng Liu, Xiaoyu Guo, Chaowei Xiao

**Abstract**: With the rapid advancements in Multimodal Large Language Models (MLLMs), securing these models against malicious inputs while aligning them with human values has emerged as a critical challenge. In this paper, we investigate an important and unexplored question of whether techniques that successfully jailbreak Large Language Models (LLMs) can be equally effective in jailbreaking MLLMs. To explore this issue, we introduce JailBreakV-28K, a pioneering benchmark designed to assess the transferability of LLM jailbreak techniques to MLLMs, thereby evaluating the robustness of MLLMs against diverse jailbreak attacks. Utilizing a dataset of 2, 000 malicious queries that is also proposed in this paper, we generate 20, 000 text-based jailbreak prompts using advanced jailbreak attacks on LLMs, alongside 8, 000 image-based jailbreak inputs from recent MLLMs jailbreak attacks, our comprehensive dataset includes 28, 000 test cases across a spectrum of adversarial scenarios. Our evaluation of 10 open-source MLLMs reveals a notably high Attack Success Rate (ASR) for attacks transferred from LLMs, highlighting a critical vulnerability in MLLMs that stems from their text-processing capabilities. Our findings underscore the urgent need for future research to address alignment vulnerabilities in MLLMs from both textual and visual inputs.

摘要: 随着多模式大型语言模型(MLLMS)的快速发展，保护这些模型不受恶意输入的影响，同时使它们与人类的价值观保持一致，已经成为一项关键的挑战。在本文中，我们研究了一个重要而未被探索的问题，即成功越狱大语言模型(LLMS)的技术是否可以在越狱MLLM中同样有效。为了探讨这一问题，我们引入了JailBreakV-28K，这是一个开创性的基准测试，旨在评估LLM越狱技术到MLLM的可转移性，从而评估MLLMS对各种越狱攻击的健壮性。利用本文提出的包含2,000个恶意查询的数据集，我们使用针对LLMS的高级越狱攻击生成了20,000个基于文本的越狱提示，以及来自最近MLLMS越狱攻击的8,000个基于图像的越狱输入，我们的综合数据集包括来自各种对抗场景的28,000个测试用例。我们对10个开源MLLMS的评估显示，对于从LLMS转移的攻击，攻击成功率(ASR)非常高，这突显了MLLMS中源于其文本处理能力的一个严重漏洞。我们的发现强调了未来研究的迫切需要，以解决MLLMS中从文本和视觉输入的对齐漏洞。



## **17. On Large Language Models in National Security Applications**

国家安全应用中的大型语言模型 cs.CR

20 pages

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2407.03453v1) [paper-pdf](http://arxiv.org/pdf/2407.03453v1)

**Authors**: William N. Caballero, Phillip R. Jenkins

**Abstract**: The overwhelming success of GPT-4 in early 2023 highlighted the transformative potential of large language models (LLMs) across various sectors, including national security. This article explores the implications of LLM integration within national security contexts, analyzing their potential to revolutionize information processing, decision-making, and operational efficiency. Whereas LLMs offer substantial benefits, such as automating tasks and enhancing data analysis, they also pose significant risks, including hallucinations, data privacy concerns, and vulnerability to adversarial attacks. Through their coupling with decision-theoretic principles and Bayesian reasoning, LLMs can significantly improve decision-making processes within national security organizations. Namely, LLMs can facilitate the transition from data to actionable decisions, enabling decision-makers to quickly receive and distill available information with less manpower. Current applications within the US Department of Defense and beyond are explored, e.g., the USAF's use of LLMs for wargaming and automatic summarization, that illustrate their potential to streamline operations and support decision-making. However, these applications necessitate rigorous safeguards to ensure accuracy and reliability. The broader implications of LLM integration extend to strategic planning, international relations, and the broader geopolitical landscape, with adversarial nations leveraging LLMs for disinformation and cyber operations, emphasizing the need for robust countermeasures. Despite exhibiting "sparks" of artificial general intelligence, LLMs are best suited for supporting roles rather than leading strategic decisions. Their use in training and wargaming can provide valuable insights and personalized learning experiences for military personnel, thereby improving operational readiness.

摘要: 2023年初GPT-4的压倒性成功突显了大型语言模型(LLM)在包括国家安全在内的各个部门的变革潜力。本文探讨了国家安全背景下LLM集成的含义，分析了它们为信息处理、决策和操作效率带来革命性变化的潜力。虽然LLMS提供了大量的好处，如自动化任务和增强数据分析，但它们也带来了重大风险，包括幻觉、数据隐私问题和易受对手攻击。通过与决策理论原则和贝叶斯推理相结合，LLMS可以显著改善国家安全组织内的决策过程。也就是说，低成本管理系统可以促进从数据到可操作决策的转变，使决策者能够以更少的人力快速接收和提取可用的信息。探讨了美国国防部内外目前的应用，例如，美国空军使用LLM进行战争游戏和自动摘要，说明了它们在简化操作和支持决策方面的潜力。然而，这些应用需要严格的安全措施来确保准确性和可靠性。LLM一体化的更广泛影响延伸到战略规划、国际关系和更广泛的地缘政治格局，敌对国家利用LLM进行虚假信息和网络行动，强调需要强有力的对策。尽管LLM展现出人工智能的“火花”，但它们最适合扮演辅助角色，而不是领导战略决策。它们在训练和战争游戏中的使用可以为军事人员提供有价值的见解和个性化的学习经验，从而提高作战准备。



## **18. Eraser: Jailbreaking Defense in Large Language Models via Unlearning Harmful Knowledge**

橡皮擦：通过忘记有害知识在大型语言模型中进行越狱防御 cs.CL

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2404.05880v2) [paper-pdf](http://arxiv.org/pdf/2404.05880v2)

**Authors**: Weikai Lu, Ziqian Zeng, Jianwei Wang, Zhengdong Lu, Zelin Chen, Huiping Zhuang, Cen Chen

**Abstract**: Jailbreaking attacks can enable Large Language Models (LLMs) to bypass the safeguard and generate harmful content. Existing jailbreaking defense methods have failed to address the fundamental issue that harmful knowledge resides within the model, leading to potential jailbreak risks for LLMs. In this paper, we propose a novel defense method called Eraser, which mainly includes three goals: unlearning harmful knowledge, retaining general knowledge, and maintaining safety alignment. The intuition is that if an LLM forgets the specific knowledge required to answer a harmful question, it will no longer have the ability to answer harmful questions. The training of Erase does not actually require the model's own harmful knowledge, and it can benefit from unlearning general answers related to harmful queries, which means it does not need assistance from the red team. The experimental results show that Eraser can significantly reduce the jailbreaking success rate for various attacks without compromising the general capabilities of the model. Our codes are available at https://github.com/ZeroNLP/Eraser.

摘要: 越狱攻击可使大型语言模型(LLM)绕过安全保护并生成有害内容。现有的越狱防御方法未能解决有害知识驻留在模型中的根本问题，导致低收入国家存在潜在的越狱风险。在本文中，我们提出了一种新的防御方法--橡皮擦，它主要包括三个目标：忘记有害知识，保留一般知识，保持安全对齐。直觉是，如果LLM忘记了回答有害问题所需的特定知识，它将不再有能力回答有害问题。ERASE的训练实际上并不需要模型本身的有害知识，而且它可以受益于忘记与有害查询相关的一般答案，这意味着它不需要红色团队的帮助。实验结果表明，在不影响模型整体性能的前提下，橡皮擦能显著降低各种攻击的越狱成功率。我们的代码可在https://github.com/ZeroNLP/Eraser.上获得



## **19. Soft Begging: Modular and Efficient Shielding of LLMs against Prompt Injection and Jailbreaking based on Prompt Tuning**

软乞讨：基于即时调优，模块化且高效地屏蔽LLM，防止即时注入和越狱 cs.CR

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2407.03391v1) [paper-pdf](http://arxiv.org/pdf/2407.03391v1)

**Authors**: Simon Ostermann, Kevin Baum, Christoph Endres, Julia Masloh, Patrick Schramowski

**Abstract**: Prompt injection (both direct and indirect) and jailbreaking are now recognized as significant issues for large language models (LLMs), particularly due to their potential for harm in application-integrated contexts. This extended abstract explores a novel approach to protecting LLMs from such attacks, termed "soft begging." This method involves training soft prompts to counteract the effects of corrupted prompts on the LLM's output. We provide an overview of prompt injections and jailbreaking, introduce the theoretical basis of the "soft begging" technique, and discuss an evaluation of its effectiveness.

摘要: 即时注入（直接和间接）和越狱现在被认为是大型语言模型（LLM）的重要问题，特别是因为它们在应用程序集成上下文中可能造成伤害。这篇扩展摘要探讨了一种保护LLM免受此类攻击的新型方法，称为“软乞讨”。“这种方法涉及训练软提示，以抵消损坏提示对LLM输出的影响。我们概述了及时注射和越狱，介绍了“软乞讨”技术的理论基础，并讨论了对其有效性的评估。



## **20. SOS! Soft Prompt Attack Against Open-Source Large Language Models**

求救！针对开源大型语言模型的软提示攻击 cs.CR

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2407.03160v1) [paper-pdf](http://arxiv.org/pdf/2407.03160v1)

**Authors**: Ziqing Yang, Michael Backes, Yang Zhang, Ahmed Salem

**Abstract**: Open-source large language models (LLMs) have become increasingly popular among both the general public and industry, as they can be customized, fine-tuned, and freely used. However, some open-source LLMs require approval before usage, which has led to third parties publishing their own easily accessible versions. Similarly, third parties have been publishing fine-tuned or quantized variants of these LLMs. These versions are particularly appealing to users because of their ease of access and reduced computational resource demands. This trend has increased the risk of training time attacks, compromising the integrity and security of LLMs. In this work, we present a new training time attack, SOS, which is designed to be low in computational demand and does not require clean data or modification of the model weights, thereby maintaining the model's utility intact. The attack addresses security issues in various scenarios, including the backdoor attack, jailbreak attack, and prompt stealing attack. Our experimental findings demonstrate that the proposed attack is effective across all evaluated targets. Furthermore, we present the other side of our SOS technique, namely the copyright token -- a novel technique that enables users to mark their copyrighted content and prevent models from using it.

摘要: 开源的大型语言模型(LLM)在普通公众和行业中变得越来越受欢迎，因为它们可以定制、微调和自由使用。然而，一些开源的LLM在使用之前需要获得批准，这导致第三方发布了他们自己的易于访问的版本。同样，第三方一直在发布这些LLM的微调或量化变体。这些版本对用户特别有吸引力，因为它们易于访问并减少了计算资源需求。这一趋势增加了训练时间攻击的风险，损害了LLMS的完整性和安全性。在这项工作中，我们提出了一种新的训练时间攻击，SOS，它被设计为计算量低，不需要干净的数据或修改模型权重，从而保持了模型的实用性。该攻击针对各种场景的安全问题，包括后门攻击、越狱攻击、提示窃取攻击。我们的实验结果表明，所提出的攻击对所有被评估目标都是有效的。此外，我们还介绍了SOS技术的另一面，即版权令牌--这是一种使用户能够标记其受版权保护的内容并防止模型使用它的新技术。



## **21. JailbreakHunter: A Visual Analytics Approach for Jailbreak Prompts Discovery from Large-Scale Human-LLM Conversational Datasets**

越狱猎人：越狱的视觉分析方法从大规模人类LLM对话数据集中进行发现 cs.HC

18 pages, 9 figures

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2407.03045v1) [paper-pdf](http://arxiv.org/pdf/2407.03045v1)

**Authors**: Zhihua Jin, Shiyi Liu, Haotian Li, Xun Zhao, Huamin Qu

**Abstract**: Large Language Models (LLMs) have gained significant attention but also raised concerns due to the risk of misuse. Jailbreak prompts, a popular type of adversarial attack towards LLMs, have appeared and constantly evolved to breach the safety protocols of LLMs. To address this issue, LLMs are regularly updated with safety patches based on reported jailbreak prompts. However, malicious users often keep their successful jailbreak prompts private to exploit LLMs. To uncover these private jailbreak prompts, extensive analysis of large-scale conversational datasets is necessary to identify prompts that still manage to bypass the system's defenses. This task is highly challenging due to the immense volume of conversation data, diverse characteristics of jailbreak prompts, and their presence in complex multi-turn conversations. To tackle these challenges, we introduce JailbreakHunter, a visual analytics approach for identifying jailbreak prompts in large-scale human-LLM conversational datasets. We have designed a workflow with three analysis levels: group-level, conversation-level, and turn-level. Group-level analysis enables users to grasp the distribution of conversations and identify suspicious conversations using multiple criteria, such as similarity with reported jailbreak prompts in previous research and attack success rates. Conversation-level analysis facilitates the understanding of the progress of conversations and helps discover jailbreak prompts within their conversation contexts. Turn-level analysis allows users to explore the semantic similarity and token overlap between a singleturn prompt and the reported jailbreak prompts, aiding in the identification of new jailbreak strategies. The effectiveness and usability of the system were verified through multiple case studies and expert interviews.

摘要: 大型语言模型(LLM)获得了极大的关注，但也因误用的风险而引起了关注。越狱提示是一种流行的针对LLMS的对抗性攻击类型，已经出现并不断演变为违反LLMS的安全协议。为了解决这个问题，LLM会根据报告的越狱提示定期更新安全补丁。然而，恶意用户通常会将他们成功的越狱提示保密，以利用LLMS。为了发现这些私密的越狱提示，有必要对大规模对话数据集进行广泛分析，以确定仍然设法绕过系统防御的提示。由于对话数据量巨大，越狱提示的特点多种多样，而且它们存在于复杂的多轮对话中，这项任务具有极大的挑战性。为了应对这些挑战，我们引入了JailBreakHunter，这是一种视觉分析方法，用于在大规模的人-LLM对话数据集中识别越狱提示。我们设计了一个具有三个分析级别的工作流：小组级别、会话级别和话轮级别。组级分析使用户能够掌握对话的分布，并使用多种标准识别可疑对话，例如与之前研究中报告的越狱提示相似，以及攻击成功率。会话级别的分析有助于了解会话的进度，并帮助发现会话上下文中的越狱提示。话轮水平分析允许用户探索单一URN提示和报告的越狱提示之间的语义相似性和标记重叠，有助于识别新的越狱策略。通过多个案例研究和专家访谈，验证了该系统的有效性和可用性。



## **22. Towards More Realistic Extraction Attacks: An Adversarial Perspective**

走向更真实的提取攻击：对抗的角度 cs.CR

To be presented at PrivateNLP@ACL2024

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2407.02596v1) [paper-pdf](http://arxiv.org/pdf/2407.02596v1)

**Authors**: Yash More, Prakhar Ganesh, Golnoosh Farnadi

**Abstract**: Language models are prone to memorizing large parts of their training data, making them vulnerable to extraction attacks. Existing research on these attacks remains limited in scope, often studying isolated trends rather than the real-world interactions with these models. In this paper, we revisit extraction attacks from an adversarial perspective, exploiting the brittleness of language models. We find significant churn in extraction attack trends, i.e., even minor, unintuitive changes to the prompt, or targeting smaller models and older checkpoints, can exacerbate the risks of extraction by up to $2-4 \times$. Moreover, relying solely on the widely accepted verbatim match underestimates the extent of extracted information, and we provide various alternatives to more accurately capture the true risks of extraction. We conclude our discussion with data deduplication, a commonly suggested mitigation strategy, and find that while it addresses some memorization concerns, it remains vulnerable to the same escalation of extraction risks against a real-world adversary. Our findings highlight the necessity of acknowledging an adversary's true capabilities to avoid underestimating extraction risks.

摘要: 语言模型容易记住它们的大部分训练数据，这使得它们很容易受到提取攻击。现有对这些攻击的研究范围仍然有限，往往研究孤立的趋势，而不是与这些模型的现实世界互动。在本文中，我们从敌意的角度重新审视提取攻击，利用语言模型的脆弱性。我们发现提取攻击趋势中的显著波动，即即使对提示进行微小的、不直观的更改，或者针对较小的模型和较旧的检查点，都可能使提取的风险增加高达2-4倍$。此外，仅依靠被广泛接受的逐字匹配低估了提取信息的程度，我们提供了各种替代方案来更准确地捕获提取的真实风险。我们以重复数据删除结束我们的讨论，这是一种通常建议的缓解策略，并发现虽然它解决了一些记忆问题，但它仍然容易受到针对现实世界对手的提取风险的相同升级的影响。我们的发现突显了承认对手真实能力的必要性，以避免低估开采风险。



## **23. A False Sense of Safety: Unsafe Information Leakage in 'Safe' AI Responses**

错误的安全感：“安全”人工智能响应中不安全的信息泄露 cs.CR

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2407.02551v1) [paper-pdf](http://arxiv.org/pdf/2407.02551v1)

**Authors**: David Glukhov, Ziwen Han, Ilia Shumailov, Vardan Papyan, Nicolas Papernot

**Abstract**: Large Language Models (LLMs) are vulnerable to jailbreaks$\unicode{x2013}$methods to elicit harmful or generally impermissible outputs. Safety measures are developed and assessed on their effectiveness at defending against jailbreak attacks, indicating a belief that safety is equivalent to robustness. We assert that current defense mechanisms, such as output filters and alignment fine-tuning, are, and will remain, fundamentally insufficient for ensuring model safety. These defenses fail to address risks arising from dual-intent queries and the ability to composite innocuous outputs to achieve harmful goals. To address this critical gap, we introduce an information-theoretic threat model called inferential adversaries who exploit impermissible information leakage from model outputs to achieve malicious goals. We distinguish these from commonly studied security adversaries who only seek to force victim models to generate specific impermissible outputs. We demonstrate the feasibility of automating inferential adversaries through question decomposition and response aggregation. To provide safety guarantees, we define an information censorship criterion for censorship mechanisms, bounding the leakage of impermissible information. We propose a defense mechanism which ensures this bound and reveal an intrinsic safety-utility trade-off. Our work provides the first theoretically grounded understanding of the requirements for releasing safe LLMs and the utility costs involved.

摘要: 大型语言模型(LLM)容易受到越狱$\Unicode{x2013}$方法的攻击，从而导致有害或通常不允许的输出。制定了安全措施，并对其在防御越狱攻击方面的有效性进行了评估，表明了一种信念，即安全等同于健壮性。我们断言，目前的防御机制，如输出过滤器和对齐微调，对于确保模型安全来说，无论是现在还是将来，都是根本不够的。这些防御措施未能解决双重意图询问产生的风险，以及将无害的产出综合起来以实现有害目标的能力。为了解决这一关键差距，我们引入了一个名为推理对手的信息论威胁模型，该模型利用模型输出中不允许的信息泄漏来实现恶意目标。我们将这些区别于通常研究的安全对手，后者只寻求迫使受害者模型生成特定的不允许的输出。我们论证了通过问题分解和响应聚合自动化推理对手的可行性。为了提供安全保障，我们为审查机制定义了信息审查标准，限制了不允许信息的泄露。我们提出了一种防御机制，确保了这一界限，并揭示了内在的安全和效用之间的权衡。我们的工作首次提供了对发布安全LLM的要求和涉及的公用事业成本的理论上的理解。



## **24. Uncovering Safety Risks of Large Language Models through Concept Activation Vector**

通过概念激活载体揭示大型语言模型的安全风险 cs.CL

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2404.12038v3) [paper-pdf](http://arxiv.org/pdf/2404.12038v3)

**Authors**: Zhihao Xu, Ruixuan Huang, Changyu Chen, Shuai Wang, Xiting Wang

**Abstract**: Despite careful safety alignment, current large language models (LLMs) remain vulnerable to various attacks. To further unveil the safety risks of LLMs, we introduce a Safety Concept Activation Vector (SCAV) framework, which effectively guides the attacks by accurately interpreting LLMs' safety mechanisms. We then develop an SCAV-guided attack method that can generate both attack prompts and embedding-level attacks with automatically selected perturbation hyperparameters. Both automatic and human evaluations demonstrate that our attack method significantly improves the attack success rate and response quality while requiring less training data. Additionally, we find that our generated attack prompts may be transferable to GPT-4, and the embedding-level attacks may also be transferred to other white-box LLMs whose parameters are known. Our experiments further uncover the safety risks present in current LLMs. For example, we find that six out of seven open-source LLMs that we attack consistently provide relevant answers to more than 85\% malicious instructions. Finally, we provide insights into the safety mechanism of LLMs.

摘要: 尽管进行了仔细的安全调整，但当前的大型语言模型(LLM)仍然容易受到各种攻击。为了进一步揭示LLMS的安全隐患，我们引入了安全概念激活向量(SCAV)框架，通过准确解释LLMS的安全机制来有效地指导攻击。然后，我们开发了一种SCAV引导的攻击方法，该方法可以生成攻击提示和带有自动选择的扰动超参数的嵌入级攻击。自动和人工评估都表明，我们的攻击方法在需要更少的训练数据的情况下，显著地提高了攻击成功率和响应质量。此外，我们发现我们生成的攻击提示可以转移到GPT-4上，嵌入级攻击也可以转移到参数已知的其他白盒LLM上。我们的实验进一步揭示了当前LLM中存在的安全风险。例如，我们发现，我们攻击的七个开源LLM中有六个始终为超过85%的恶意指令提供相关答案。最后，我们对LLMS的安全机制提供了见解。



## **25. Adversarial Search Engine Optimization for Large Language Models**

大型语言模型的对抗性搜索引擎优化 cs.CR

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2406.18382v2) [paper-pdf](http://arxiv.org/pdf/2406.18382v2)

**Authors**: Fredrik Nestaas, Edoardo Debenedetti, Florian Tramèr

**Abstract**: Large Language Models (LLMs) are increasingly used in applications where the model selects from competing third-party content, such as in LLM-powered search engines or chatbot plugins. In this paper, we introduce Preference Manipulation Attacks, a new class of attacks that manipulate an LLM's selections to favor the attacker. We demonstrate that carefully crafted website content or plugin documentations can trick an LLM to promote the attacker products and discredit competitors, thereby increasing user traffic and monetization. We show this leads to a prisoner's dilemma, where all parties are incentivized to launch attacks, but the collective effect degrades the LLM's outputs for everyone. We demonstrate our attacks on production LLM search engines (Bing and Perplexity) and plugin APIs (for GPT-4 and Claude). As LLMs are increasingly used to rank third-party content, we expect Preference Manipulation Attacks to emerge as a significant threat.

摘要: 大型语言模型（LLM）越来越多地用于模型从竞争的第三方内容中进行选择的应用程序中，例如LLM支持的搜索引擎或聊天机器人插件。在本文中，我们引入了偏好操纵攻击，这是一类新的攻击，可以操纵LLM的选择以有利于攻击者。我们证明，精心制作的网站内容或插件文档可以欺骗LLM来推广攻击者的产品并抹黑竞争对手，从而增加用户流量和货币化。我们表明，这导致了囚犯困境，所有各方都受到激励发起攻击，但集体效应降低了LLM对每个人的产出。我们展示了对生产LLM搜索引擎（Bing和Perplexity）和插件API（适用于GPT-4和Claude）的攻击。随着LLM越来越多地用于对第三方内容进行排名，我们预计偏好操纵攻击将成为一个重大威胁。



## **26. SoP: Unlock the Power of Social Facilitation for Automatic Jailbreak Attack**

SoP：充分利用自动越狱攻击的社会促进力量 cs.CR

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2407.01902v1) [paper-pdf](http://arxiv.org/pdf/2407.01902v1)

**Authors**: Yan Yang, Zeguan Xiao, Xin Lu, Hongru Wang, Hailiang Huang, Guanhua Chen, Yun Chen

**Abstract**: The widespread applications of large language models (LLMs) have brought about concerns regarding their potential misuse. Although aligned with human preference data before release, LLMs remain vulnerable to various malicious attacks. In this paper, we adopt a red-teaming strategy to enhance LLM safety and introduce SoP, a simple yet effective framework to design jailbreak prompts automatically. Inspired by the social facilitation concept, SoP generates and optimizes multiple jailbreak characters to bypass the guardrails of the target LLM. Different from previous work which relies on proprietary LLMs or seed jailbreak templates crafted by human expertise, SoP can generate and optimize the jailbreak prompt in a cold-start scenario using open-sourced LLMs without any seed jailbreak templates. Experimental results show that SoP achieves attack success rates of 88% and 60% in bypassing the safety alignment of GPT-3.5-1106 and GPT-4, respectively. Furthermore, we extensively evaluate the transferability of the generated templates across different LLMs and held-out malicious requests, while also exploring defense strategies against the jailbreak attack designed by SoP. Code is available at https://github.com/Yang-Yan-Yang-Yan/SoP.

摘要: 大型语言模型(LLM)的广泛应用引起了人们对其潜在滥用的担忧。尽管在发布之前与人类偏好数据保持一致，但LLM仍然容易受到各种恶意攻击。在本文中，我们采用了红队策略来增强LLM的安全性，并引入了一个简单而有效的框架SOP来自动设计越狱提示。受社交促进概念的启发，SOP生成并优化了多个越狱角色，以绕过目标LLM的护栏。与以往依赖专有LLM或人工制作的种子越狱模板的工作不同，SOP可以在冷启动场景下使用开源LLM生成和优化越狱提示，而不需要任何种子越狱模板。实验结果表明，该算法绕过GPT-3.5-1106和GPT-4的安全对齐，攻击成功率分别达到88%和60%。此外，我们还广泛评估了生成的模板在不同LLM和拒绝恶意请求之间的可移植性，同时也探索了针对SOP设计的越狱攻击的防御策略。代码可在https://github.com/Yang-Yan-Yang-Yan/SoP.上找到



## **27. Revisiting Backdoor Attacks against Large Vision-Language Models**

重新审视针对大型视觉语言模型的后门攻击 cs.CV

24 pages, 8 figures

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2406.18844v3) [paper-pdf](http://arxiv.org/pdf/2406.18844v3)

**Authors**: Siyuan Liang, Jiawei Liang, Tianyu Pang, Chao Du, Aishan Liu, Ee-Chien Chang, Xiaochun Cao

**Abstract**: Instruction tuning enhances large vision-language models (LVLMs) but raises security risks through potential backdoor attacks due to their openness. Previous backdoor studies focus on enclosed scenarios with consistent training and testing instructions, neglecting the practical domain gaps that could affect attack effectiveness. This paper empirically examines the generalizability of backdoor attacks during the instruction tuning of LVLMs for the first time, revealing certain limitations of most backdoor strategies in practical scenarios. We quantitatively evaluate the generalizability of six typical backdoor attacks on image caption benchmarks across multiple LVLMs, considering both visual and textual domain offsets. Our findings indicate that attack generalizability is positively correlated with the backdoor trigger's irrelevance to specific images/models and the preferential correlation of the trigger pattern. Additionally, we modify existing backdoor attacks based on the above key observations, demonstrating significant improvements in cross-domain scenario generalizability (+86% attack success rate). Notably, even without access to the instruction datasets, a multimodal instruction set can be successfully poisoned with a very low poisoning rate (0.2%), achieving an attack success rate of over 97%. This paper underscores that even simple traditional backdoor strategies pose a serious threat to LVLMs, necessitating more attention and in-depth research.

摘要: 指令调优增强了大型视觉语言模型(LVLM)，但由于其开放性，通过潜在的后门攻击增加了安全风险。以前的后门研究侧重于具有一致训练和测试指令的封闭场景，而忽略了可能影响攻击效果的实际领域差距。本文首次对LVLMS指令调优过程中后门攻击的泛化能力进行了实证检验，揭示了大多数后门策略在实际应用中的局限性。我们定量地评估了六种典型的后门攻击在多个LVLM上对图像字幕基准的泛化能力，同时考虑了视觉和文本域偏移。我们的研究结果表明，攻击的概括性与后门触发器与特定图像/模型的无关性以及触发模式的优先相关性呈正相关。此外，我们根据上述关键观察结果修改了现有的后门攻击，显示出跨域场景通用性的显著改进(+86%的攻击成功率)。值得注意的是，即使不访问指令数据集，多模式指令集也可以以非常低的投毒率(0.2%)成功中毒，实现超过97%的攻击成功率。这篇文章强调，即使是简单的传统后门策略也对LVLMS构成了严重威胁，需要更多的关注和深入的研究。



## **28. Image-to-Text Logic Jailbreak: Your Imagination can Help You Do Anything**

图像到文本逻辑越狱：你的想象力可以帮助你做任何事情 cs.CR

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.02534v1) [paper-pdf](http://arxiv.org/pdf/2407.02534v1)

**Authors**: Xiaotian Zou, Yongkang Chen

**Abstract**: Large Visual Language Models (VLMs) such as GPT-4 have achieved remarkable success in generating comprehensive and nuanced responses, surpassing the capabilities of large language models. However, with the integration of visual inputs, new security concerns emerge, as malicious attackers can exploit multiple modalities to achieve their objectives. This has led to increasing attention on the vulnerabilities of VLMs to jailbreak. Most existing research focuses on generating adversarial images or nonsensical image collections to compromise these models. However, the challenge of leveraging meaningful images to produce targeted textual content using the VLMs' logical comprehension of images remains unexplored. In this paper, we explore the problem of logical jailbreak from meaningful images to text. To investigate this issue, we introduce a novel dataset designed to evaluate flowchart image jailbreak. Furthermore, we develop a framework for text-to-text jailbreak using VLMs. Finally, we conduct an extensive evaluation of the framework on GPT-4o and GPT-4-vision-preview, with jailbreak rates of 92.8% and 70.0%, respectively. Our research reveals significant vulnerabilities in current VLMs concerning image-to-text jailbreak. These findings underscore the need for a deeper examination of the security flaws in VLMs before their practical deployment.

摘要: 像GPT-4这样的大型视觉语言模型(VLM)在生成全面和细微差别的响应方面取得了显著的成功，超过了大型语言模型的能力。然而，随着视觉输入的集成，新的安全问题出现了，因为恶意攻击者可以利用多种模式来实现他们的目标。这引起了人们对越狱漏洞的越来越多的关注。现有的大多数研究都集中在生成敌意图像或无意义的图像集合来危害这些模型。然而，利用VLMS对图像的逻辑理解来利用有意义的图像来产生有针对性的文本内容的挑战仍然没有被探索。在本文中，我们探讨了从有意义的图像到文本的逻辑越狱问题。为了研究这个问题，我们引入了一个新的数据集，用于评估流程图图像越狱。此外，我们使用VLMS开发了一个文本到文本越狱的框架。最后，我们在GPT-4O和GPT-4-VISION-PREVIEW上对该框架进行了广泛的评估，越狱率分别为92.8%和70.0%。我们的研究揭示了当前VLM在图像到文本越狱方面的重大漏洞。这些调查结果突出表明，在实际部署VLM之前，需要更深入地审查VLM的安全缺陷。



## **29. Enhancing the Capability and Robustness of Large Language Models through Reinforcement Learning-Driven Query Refinement**

通过强化学习驱动的查询细化增强大型语言模型的能力和鲁棒性 cs.CL

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01461v1) [paper-pdf](http://arxiv.org/pdf/2407.01461v1)

**Authors**: Zisu Huang, Xiaohua Wang, Feiran Zhang, Zhibo Xu, Cenyuan Zhang, Xiaoqing Zheng, Xuanjing Huang

**Abstract**: The capacity of large language models (LLMs) to generate honest, harmless, and helpful responses heavily relies on the quality of user prompts. However, these prompts often tend to be brief and vague, thereby significantly limiting the full potential of LLMs. Moreover, harmful prompts can be meticulously crafted and manipulated by adversaries to jailbreak LLMs, inducing them to produce potentially toxic content. To enhance the capabilities of LLMs while maintaining strong robustness against harmful jailbreak inputs, this study proposes a transferable and pluggable framework that refines user prompts before they are input into LLMs. This strategy improves the quality of the queries, empowering LLMs to generate more truthful, benign and useful responses. Specifically, a lightweight query refinement model is introduced and trained using a specially designed reinforcement learning approach that incorporates multiple objectives to enhance particular capabilities of LLMs. Extensive experiments demonstrate that the refinement model not only improves the quality of responses but also strengthens their robustness against jailbreak attacks. Code is available at: https://github.com/Huangzisu/query-refinement .

摘要: 大型语言模型(LLM)生成诚实、无害和有用的响应的能力在很大程度上取决于用户提示的质量。然而，这些提示往往简短而含糊，从而极大地限制了LLM的全部潜力。此外，有害的提示可以被对手精心制作和操纵，以越狱LLM，诱导它们产生潜在的有毒内容。为了增强LLMS的能力，同时保持对有害越狱输入的强大健壮性，本研究提出了一个可移植和可插拔的框架，在将用户提示输入到LLMS之前对其进行提炼。这一策略提高了查询的质量，使LLMS能够生成更真实、良性和有用的响应。具体地说，引入了一种轻量级查询精化模型，并使用专门设计的强化学习方法进行训练，该方法结合了多个目标来增强LLMS的特定能力。大量实验表明，改进模型不仅提高了响应的质量，而且增强了对越狱攻击的健壮性。代码可从以下网址获得：https://github.com/Huangzisu/query-refinement。



## **30. Jailbreak Vision Language Models via Bi-Modal Adversarial Prompt**

通过双模式对抗提示的越狱视觉语言模型 cs.CV

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2406.04031v2) [paper-pdf](http://arxiv.org/pdf/2406.04031v2)

**Authors**: Zonghao Ying, Aishan Liu, Tianyuan Zhang, Zhengmin Yu, Siyuan Liang, Xianglong Liu, Dacheng Tao

**Abstract**: In the realm of large vision language models (LVLMs), jailbreak attacks serve as a red-teaming approach to bypass guardrails and uncover safety implications. Existing jailbreaks predominantly focus on the visual modality, perturbing solely visual inputs in the prompt for attacks. However, they fall short when confronted with aligned models that fuse visual and textual features simultaneously for generation. To address this limitation, this paper introduces the Bi-Modal Adversarial Prompt Attack (BAP), which executes jailbreaks by optimizing textual and visual prompts cohesively. Initially, we adversarially embed universally harmful perturbations in an image, guided by a few-shot query-agnostic corpus (e.g., affirmative prefixes and negative inhibitions). This process ensures that image prompt LVLMs to respond positively to any harmful queries. Subsequently, leveraging the adversarial image, we optimize textual prompts with specific harmful intent. In particular, we utilize a large language model to analyze jailbreak failures and employ chain-of-thought reasoning to refine textual prompts through a feedback-iteration manner. To validate the efficacy of our approach, we conducted extensive evaluations on various datasets and LVLMs, demonstrating that our method significantly outperforms other methods by large margins (+29.03% in attack success rate on average). Additionally, we showcase the potential of our attacks on black-box commercial LVLMs, such as Gemini and ChatGLM.

摘要: 在大型视觉语言模型(LVLM)领域，越狱攻击是一种绕过护栏并发现安全隐患的红队方法。现有的越狱主要集中在视觉形式上，只干扰攻击提示中的视觉输入。然而，当面对同时融合视觉和文本特征以生成的对齐模型时，它们不能满足要求。为了解决这一局限性，本文引入了双模式对抗性提示攻击(BAP)，它通过结合优化文本和视觉提示来执行越狱。最初，我们不利地在图像中嵌入普遍有害的扰动，由几个与查询无关的语料库(例如，肯定前缀和否定抑制)引导。此过程确保图像提示LVLMS对任何有害查询做出积极响应。随后，利用敌意图像，我们优化了具有特定有害意图的文本提示。特别是，我们利用一个大的语言模型来分析越狱失败，并使用思想链推理来通过反馈迭代的方式来提炼文本提示。为了验证我们方法的有效性，我们在不同的数据集和LVLM上进行了广泛的评估，结果表明我们的方法在很大程度上优于其他方法(攻击成功率平均为+29.03%)。此外，我们还展示了我们对黑盒商业LVLM的攻击潜力，如Gemini和ChatGLM。



## **31. A Fingerprint for Large Language Models**

大型语言模型的指纹 cs.CR

https://scholar.google.com/citations?user=IdiF7M0AAAAJ&hl=en

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01235v1) [paper-pdf](http://arxiv.org/pdf/2407.01235v1)

**Authors**: Zhiguang Yang, Hanzhou Wu

**Abstract**: Recent advances show that scaling a pre-trained language model could achieve state-of-the-art performance on many downstream tasks, prompting large language models (LLMs) to become a hot research topic in the field of artificial intelligence. However, due to the resource-intensive nature of training LLMs from scratch, it is urgent and crucial to protect the intellectual property of LLMs against infringement. This has motivated the authors in this paper to propose a novel black-box fingerprinting technique for LLMs, which requires neither model training nor model fine-tuning. We first demonstrate that the outputs of LLMs span a unique vector space associated with each model. We model the problem of ownership authentication as the task of evaluating the similarity between the victim model's space and the output's space of the suspect model. To deal with this problem, we propose two solutions, where the first solution involves verifying whether the outputs of the suspected large model are in the same space as those of the victim model, enabling rapid identification of model infringement, and the second one reconstructs the union of the vector spaces for LLM outputs and the victim model to address situations where the victim model has undergone the Parameter-Efficient Fine-Tuning (PEFT) attacks. Experimental results indicate that the proposed technique achieves superior performance in ownership verification and robustness against PEFT attacks. This work reveals inherent characteristics of LLMs and provides a promising solution for ownership verification of LLMs in black-box scenarios, ensuring efficiency, generality and practicality.

摘要: 最近的进展表明，扩展一个预先训练的语言模型可以在许多下游任务上获得最先进的性能，这促使大型语言模型(LLM)成为人工智能领域的研究热点。然而，由于从无到有培训低成本管理人员的资源密集型性质，保护低收入管理人员的知识产权不受侵犯是迫切和关键的。为此，本文提出了一种新的黑盒指纹识别方法，该方法既不需要模型训练，也不需要模型微调。我们首先证明了LLMS的输出跨越了与每个模型相关的唯一向量空间。我们将所有权认证问题建模为评估受害者模型的空间与嫌疑人模型的输出空间之间的相似度的任务。为了解决这个问题，我们提出了两种解决方案，第一种方案涉及验证可疑大模型的输出是否与受害者模型的输出在同一空间中，从而能够快速识别模型违规；第二种方案重构LLM输出和受害者模型的向量空间的并集，以应对受害者模型经历了参数高效精调(PEFT)攻击的情况。实验结果表明，该方法具有较好的所有权验证性能和对PEFT攻击的稳健性。这项工作揭示了LLMS的内在特征，为黑盒场景下LLMS的所有权验证提供了一种很有前途的解决方案，确保了效率、通用性和实用性。



## **32. Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications**

通过修剪和低级修改评估安全对齐的脆弱性 cs.LG

22 pages, 9 figures. Project page is available at  https://boyiwei.com/alignment-attribution/

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2402.05162v3) [paper-pdf](http://arxiv.org/pdf/2402.05162v3)

**Authors**: Boyi Wei, Kaixuan Huang, Yangsibo Huang, Tinghao Xie, Xiangyu Qi, Mengzhou Xia, Prateek Mittal, Mengdi Wang, Peter Henderson

**Abstract**: Large language models (LLMs) show inherent brittleness in their safety mechanisms, as evidenced by their susceptibility to jailbreaking and even non-malicious fine-tuning. This study explores this brittleness of safety alignment by leveraging pruning and low-rank modifications. We develop methods to identify critical regions that are vital for safety guardrails, and that are disentangled from utility-relevant regions at both the neuron and rank levels. Surprisingly, the isolated regions we find are sparse, comprising about $3\%$ at the parameter level and $2.5\%$ at the rank level. Removing these regions compromises safety without significantly impacting utility, corroborating the inherent brittleness of the model's safety mechanisms. Moreover, we show that LLMs remain vulnerable to low-cost fine-tuning attacks even when modifications to the safety-critical regions are restricted. These findings underscore the urgent need for more robust safety strategies in LLMs.

摘要: 大型语言模型（LLM）在其安全机制中表现出固有的脆弱性，这一点从它们容易越狱甚至非恶意微调中得到了证明。这项研究通过利用修剪和低等级修改来探索安全对齐的脆弱性。我们开发了识别对安全护栏至关重要的关键区域的方法，并且在神经元和等级水平上与公用事业相关区域脱钩。令人惊讶的是，我们发现的孤立区域很稀疏，参数级别约为3美元，排名级别约为2.5美元。删除这些区域会损害安全性，而不会显着影响实用性，这证实了该模型安全机制固有的脆弱性。此外，我们表明，即使对安全关键区域的修改受到限制，LLM仍然容易受到低成本微调攻击。这些发现凸显了LLM迫切需要制定更稳健的安全策略。



## **33. Large Language Models Are Involuntary Truth-Tellers: Exploiting Fallacy Failure for Jailbreak Attacks**

大型语言模型是不自愿的真话者：利用谬误失败进行越狱攻击 cs.CL

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.00869v1) [paper-pdf](http://arxiv.org/pdf/2407.00869v1)

**Authors**: Yue Zhou, Henry Peng Zou, Barbara Di Eugenio, Yang Zhang

**Abstract**: We find that language models have difficulties generating fallacious and deceptive reasoning. When asked to generate deceptive outputs, language models tend to leak honest counterparts but believe them to be false. Exploiting this deficiency, we propose a jailbreak attack method that elicits an aligned language model for malicious output. Specifically, we query the model to generate a fallacious yet deceptively real procedure for the harmful behavior. Since a fallacious procedure is generally considered fake and thus harmless by LLMs, it helps bypass the safeguard mechanism. Yet the output is factually harmful since the LLM cannot fabricate fallacious solutions but proposes truthful ones. We evaluate our approach over five safety-aligned large language models, comparing four previous jailbreak methods, and show that our approach achieves competitive performance with more harmful outputs. We believe the findings could be extended beyond model safety, such as self-verification and hallucination.

摘要: 我们发现，语言模型很难生成错误和欺骗性的推理。当被要求生成欺骗性输出时，语言模型往往会泄露诚实的对应结果，但认为它们是错误的。利用这一缺陷，我们提出了一种越狱攻击方法，该方法可以得到恶意输出的对齐语言模型。具体地说，我们对该模型提出质疑，以便为有害行为生成一个虚假但虚假的真实过程。由于错误的程序通常被LLMS认为是虚假的，因此是无害的，它有助于绕过保障机制。然而，这些结果实际上是有害的，因为LLM不能捏造虚假的解决方案，而是提出真实的解决方案。我们在五个安全对齐的大型语言模型上对我们的方法进行了评估，并与之前的四种越狱方法进行了比较，结果表明我们的方法在具有更多有害输出的情况下获得了具有竞争力的性能。我们认为，这些发现可以扩展到模型安全之外，例如自我验证和幻觉。



## **34. Virtual Context: Enhancing Jailbreak Attacks with Special Token Injection**

虚拟上下文：通过特殊代币注入增强越狱攻击 cs.CR

14 pages, 4 figures

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.19845v1) [paper-pdf](http://arxiv.org/pdf/2406.19845v1)

**Authors**: Yuqi Zhou, Lin Lu, Hanchi Sun, Pan Zhou, Lichao Sun

**Abstract**: Jailbreak attacks on large language models (LLMs) involve inducing these models to generate harmful content that violates ethics or laws, posing a significant threat to LLM security. Current jailbreak attacks face two main challenges: low success rates due to defensive measures and high resource requirements for crafting specific prompts. This paper introduces Virtual Context, which leverages special tokens, previously overlooked in LLM security, to improve jailbreak attacks. Virtual Context addresses these challenges by significantly increasing the success rates of existing jailbreak methods and requiring minimal background knowledge about the target model, thus enhancing effectiveness in black-box settings without additional overhead. Comprehensive evaluations show that Virtual Context-assisted jailbreak attacks can improve the success rates of four widely used jailbreak methods by approximately 40% across various LLMs. Additionally, applying Virtual Context to original malicious behaviors still achieves a notable jailbreak effect. In summary, our research highlights the potential of special tokens in jailbreak attacks and recommends including this threat in red-teaming testing to comprehensively enhance LLM security.

摘要: 针对大型语言模型(LLM)的越狱攻击涉及诱导这些模型生成违反道德或法律的有害内容，对LLM安全构成重大威胁。目前的越狱攻击面临两个主要挑战：防御性措施导致的成功率较低，以及制作特定提示所需的资源较高。本文介绍了虚拟上下文技术，它利用了以前在LLM安全中被忽视的特殊令牌来改进越狱攻击。虚拟环境通过显著提高现有越狱方法的成功率和只需要最少的目标模型背景知识来解决这些挑战，从而在不增加额外开销的情况下提高黑箱设置的效率。综合评估表明，虚拟情境辅助越狱攻击可以将四种广泛使用的越狱方法的成功率提高约40%。此外，将虚拟情境应用于原始恶意行为仍然可以达到显著的越狱效果。综上所述，我们的研究强调了特殊令牌在越狱攻击中的潜力，并建议将此威胁包括在红团队测试中，以全面增强LLM安全。



## **35. SafeAligner: Safety Alignment against Jailbreak Attacks via Response Disparity Guidance**

SafeAligner：通过响应差异指导针对越狱攻击的安全调整 cs.CR

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.18118v2) [paper-pdf](http://arxiv.org/pdf/2406.18118v2)

**Authors**: Caishuang Huang, Wanxu Zhao, Rui Zheng, Huijie Lv, Shihan Dou, Sixian Li, Xiao Wang, Enyu Zhou, Junjie Ye, Yuming Yang, Tao Gui, Qi Zhang, Xuanjing Huang

**Abstract**: As the development of large language models (LLMs) rapidly advances, securing these models effectively without compromising their utility has become a pivotal area of research. However, current defense strategies against jailbreak attacks (i.e., efforts to bypass security protocols) often suffer from limited adaptability, restricted general capability, and high cost. To address these challenges, we introduce SafeAligner, a methodology implemented at the decoding stage to fortify defenses against jailbreak attacks. We begin by developing two specialized models: the Sentinel Model, which is trained to foster safety, and the Intruder Model, designed to generate riskier responses. SafeAligner leverages the disparity in security levels between the responses from these models to differentiate between harmful and beneficial tokens, effectively guiding the safety alignment by altering the output token distribution of the target model. Extensive experiments show that SafeAligner can increase the likelihood of beneficial tokens, while reducing the occurrence of harmful ones, thereby ensuring secure alignment with minimal loss to generality.

摘要: 随着大型语言模型(LLM)的发展，在不影响其实用性的情况下有效地保护这些模型已成为一个关键的研究领域。然而，当前针对越狱攻击的防御策略(即绕过安全协议的努力)往往存在适应性有限、通用能力有限和成本较高的问题。为了应对这些挑战，我们引入了SafeAligner，这是一种在解码阶段实施的方法，用于加强对越狱攻击的防御。我们首先开发两个专门的模型：哨兵模型和入侵者模型，前者旨在促进安全，后者旨在产生更高风险的反应。SafeAligner利用这些模型响应之间的安全级别差异来区分有害令牌和有益令牌，通过更改目标模型的输出令牌分布有效地指导安全对齐。广泛的实验表明，SafeAligner可以增加有益令牌的可能性，同时减少有害令牌的发生，从而确保安全对齐，并将对一般性的损失降至最低。



## **36. GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts**

GPTFUZER：Red将大型语言模型与自动生成的越狱脚本结合起来 cs.AI

**SubmitDate**: 2024-06-27    [abs](http://arxiv.org/abs/2309.10253v4) [paper-pdf](http://arxiv.org/pdf/2309.10253v4)

**Authors**: Jiahao Yu, Xingwei Lin, Zheng Yu, Xinyu Xing

**Abstract**: Large language models (LLMs) have recently experienced tremendous popularity and are widely used from casual conversations to AI-driven programming. However, despite their considerable success, LLMs are not entirely reliable and can give detailed guidance on how to conduct harmful or illegal activities. While safety measures can reduce the risk of such outputs, adversarial jailbreak attacks can still exploit LLMs to produce harmful content. These jailbreak templates are typically manually crafted, making large-scale testing challenging.   In this paper, we introduce GPTFuzz, a novel black-box jailbreak fuzzing framework inspired by the AFL fuzzing framework. Instead of manual engineering, GPTFuzz automates the generation of jailbreak templates for red-teaming LLMs. At its core, GPTFuzz starts with human-written templates as initial seeds, then mutates them to produce new templates. We detail three key components of GPTFuzz: a seed selection strategy for balancing efficiency and variability, mutate operators for creating semantically equivalent or similar sentences, and a judgment model to assess the success of a jailbreak attack.   We evaluate GPTFuzz against various commercial and open-source LLMs, including ChatGPT, LLaMa-2, and Vicuna, under diverse attack scenarios. Our results indicate that GPTFuzz consistently produces jailbreak templates with a high success rate, surpassing human-crafted templates. Remarkably, GPTFuzz achieves over 90% attack success rates against ChatGPT and Llama-2 models, even with suboptimal initial seed templates. We anticipate that GPTFuzz will be instrumental for researchers and practitioners in examining LLM robustness and will encourage further exploration into enhancing LLM safety.

摘要: 大型语言模型(LLM)最近经历了巨大的流行，并被广泛使用，从随意的对话到人工智能驱动的编程。然而，尽管LLM取得了相当大的成功，但它们并不完全可靠，可以就如何进行有害或非法活动提供详细指导。虽然安全措施可以降低此类输出的风险，但对抗性越狱攻击仍然可以利用LLMS产生有害内容。这些越狱模板通常是手动制作的，这使得大规模测试具有挑战性。在本文中，我们介绍了一种新的黑盒越狱模糊框架GPTFuzz，该框架受到AFL模糊框架的启发。GPTFuzz不是手动设计，而是自动生成用于红队LLM的越狱模板。在其核心，GPTFuzz以人类编写的模板作为初始种子，然后对它们进行突变以产生新的模板。我们详细介绍了GPTFuzz的三个关键组成部分：用于平衡效率和可变性的种子选择策略，用于创建语义等价或相似句子的变异算子，以及用于评估越狱攻击成功的判断模型。我们在不同的攻击场景下，针对各种商业和开源LLM，包括ChatGPT、骆驼2和维库纳，对GPTFuzz进行了评估。我们的结果表明，GPTFuzz一致地生成了成功率较高的越狱模板，超过了人工制作的模板。值得注意的是，GPTFuzz对ChatGPT和Llama-2模型的攻击成功率超过90%，即使在初始种子模板不是最优的情况下也是如此。我们预计，GPTFuzz将有助于研究人员和从业者检查LLM的稳健性，并将鼓励进一步探索增强LLM的安全性。



## **37. Seeing Is Believing: Black-Box Membership Inference Attacks Against Retrieval Augmented Generation**

亲眼所见：针对检索增强生成的黑匣子成员推断攻击 cs.CR

**SubmitDate**: 2024-06-27    [abs](http://arxiv.org/abs/2406.19234v1) [paper-pdf](http://arxiv.org/pdf/2406.19234v1)

**Authors**: Yuying Li, Gaoyang Liu, Yang Yang, Chen Wang

**Abstract**: Retrieval-Augmented Generation (RAG) is a state-of-the-art technique that enhances Large Language Models (LLMs) by retrieving relevant knowledge from an external, non-parametric database. This approach aims to mitigate common LLM issues such as hallucinations and outdated knowledge. Although existing research has demonstrated security and privacy vulnerabilities within RAG systems, making them susceptible to attacks like jailbreaks and prompt injections, the security of the RAG system's external databases remains largely underexplored. In this paper, we employ Membership Inference Attacks (MIA) to determine whether a sample is part of the knowledge database of a RAG system, using only black-box API access. Our core hypothesis posits that if a sample is a member, it will exhibit significant similarity to the text generated by the RAG system. To test this, we compute the cosine similarity and the model's perplexity to establish a membership score, thereby building robust features. We then introduce two novel attack strategies: a Threshold-based Attack and a Machine Learning-based Attack, designed to accurately identify membership. Experimental validation of our methods has achieved a ROC AUC of 82%.

摘要: 检索-增强生成(RAG)是一种最先进的技术，它通过从外部非参数数据库检索相关知识来增强大型语言模型(LLMS)。这种方法旨在缓解常见的LLM问题，如幻觉和过时的知识。尽管现有的研究已经证明RAG系统中存在安全和隐私漏洞，使它们容易受到越狱和快速注射等攻击，但RAG系统外部数据库的安全性在很大程度上仍未得到充分研究。在本文中，我们使用成员关系推理攻击(MIA)来确定样本是否属于RAG系统的知识库的一部分，只使用黑盒API访问。我们的核心假设是，如果样本是成员，它将显示出与RAG系统生成的文本的显著相似性。为了测试这一点，我们计算余弦相似度和模型的困惑度来建立隶属度分数，从而建立稳健的特征。然后，我们介绍了两种新的攻击策略：基于阈值的攻击和基于机器学习的攻击，旨在准确识别成员身份。通过实验验证，我们的方法获得了82%的ROC AUC。



## **38. Chat AI: A Seamless Slurm-Native Solution for HPC-Based Services**

Chat AI：针对基于HP的服务的无缝SlurmNative解决方案 cs.DC

27 pages, 5 figures, 2 tables

**SubmitDate**: 2024-06-27    [abs](http://arxiv.org/abs/2407.00110v1) [paper-pdf](http://arxiv.org/pdf/2407.00110v1)

**Authors**: Ali Doosthosseini, Jonathan Decker, Hendrik Nolte, Julian M. Kunkel

**Abstract**: The increasing adoption of large language models (LLMs) has created a pressing need for an efficient, secure and private serving infrastructure, which allows researchers to run open-source or custom fine-tuned LLMs and ensures users that their data remains private and is not stored without their consent. While high-performance computing (HPC) systems equipped with state-of-the-art GPUs are well-suited for training LLMs, their batch scheduling paradigm is not designed to support real-time serving of AI applications. Cloud systems, on the other hand, are well suited for web services but commonly lack access to the computational power of clusters, especially expensive and scarce high-end GPUs, which are required for optimal inference speed. We propose an architecture with an implementation consisting of a web service that runs on a cloud VM with secure access to a scalable backend running a multitude of AI models on HPC systems. By offering a web service using our HPC infrastructure to host LLMs, we leverage the trusted environment of local universities and research centers to offer a private and secure alternative to commercial LLM services. Our solution natively integrates with Slurm, enabling seamless deployment on HPC clusters and is able to run side by side with regular Slurm workloads, while utilizing gaps in the schedule created by Slurm. In order to ensure the security of the HPC system, we use the SSH ForceCommand directive to construct a robust circuit breaker, which prevents successful attacks on the web-facing server from affecting the cluster. We have successfully deployed our system as a production service, and made the source code available at https://github.com/gwdg/chat-ai

摘要: 大型语言模型(LLM)的日益采用产生了对高效、安全和私有的服务基础设施的迫切需求，该基础设施允许研究人员运行开源或定制的微调LLM，并确保用户的数据保持隐私，并且在未经用户同意的情况下不被存储。虽然配备最先进的GPU的高性能计算(HPC)系统非常适合训练LLM，但它们的批处理调度范例并不是为支持AI应用的实时服务而设计的。另一方面，云系统非常适合Web服务，但通常无法获得集群的计算能力，特别是昂贵而稀缺的高端GPU，这是实现最佳推理速度所必需的。我们提出了一种体系结构，其实现包括在云VM上运行的Web服务，该服务可以安全地访问在HPC系统上运行大量AI模型的可扩展后端。通过提供使用我们的HPC基础设施来托管LLM的Web服务，我们利用当地大学和研究中心的可信环境来提供商业LLM服务的私有且安全的替代方案。我们的解决方案与SLurm进行了本机集成，实现了在HPC群集上的无缝部署，并能够与常规SLurm工作负载并行运行，同时利用SLurm创建的时间表中的空白。为了确保HPC系统的安全，我们使用SSH ForceCommand指令来构建一个健壮的断路器，以防止对面向Web的服务器的成功攻击影响到集群。我们已经成功地将我们的系统部署为生产服务，并在https://github.com/gwdg/chat-ai上提供了源代码



## **39. Assessing the Effectiveness of LLMs in Android Application Vulnerability Analysis**

评估LLC在Android应用程序漏洞分析中的有效性 cs.CR

**SubmitDate**: 2024-06-27    [abs](http://arxiv.org/abs/2406.18894v1) [paper-pdf](http://arxiv.org/pdf/2406.18894v1)

**Authors**: Vasileios Kouliaridis, Georgios Karopoulos, Georgios Kambourakis

**Abstract**: The increasing frequency of attacks on Android applications coupled with the recent popularity of large language models (LLMs) necessitates a comprehensive understanding of the capabilities of the latter in identifying potential vulnerabilities, which is key to mitigate the overall risk. To this end, the work at hand compares the ability of nine state-of-the-art LLMs to detect Android code vulnerabilities listed in the latest Open Worldwide Application Security Project (OWASP) Mobile Top 10. Each LLM was evaluated against an open dataset of over 100 vulnerable code samples, including obfuscated ones, assessing each model's ability to identify key vulnerabilities. Our analysis reveals the strengths and weaknesses of each LLM, identifying important factors that contribute to their performance. Additionally, we offer insights into context augmentation with retrieval-augmented generation (RAG) for detecting Android code vulnerabilities, which in turn may propel secure application development. Finally, while the reported findings regarding code vulnerability analysis show promise, they also reveal significant discrepancies among the different LLMs.

摘要: 针对Android应用程序的攻击日益频繁，加上最近大型语言模型(LLM)的流行，需要全面了解后者在识别潜在漏洞方面的能力，这是降低总体风险的关键。为此，手头的工作比较了9个最先进的LLM检测Android代码漏洞的能力，这些漏洞列在最新的Open Worldwide Application Security Project(OWASP)Mobile Top 10中。每个LLM都是根据100多个易受攻击的代码样本(包括混淆的代码样本)的开放数据集进行评估的，以评估每个模型识别关键漏洞的能力。我们的分析揭示了每个LLM的优势和劣势，找出了影响它们表现的重要因素。此外，我们通过检索增强生成(RAG)提供了对上下文增强的见解，以检测Android代码漏洞，这反过来可能会推动安全的应用程序开发。最后，虽然有关代码漏洞分析的报告结果显示前景看好，但它们也揭示了不同LLM之间的显著差异。



## **40. Dysca: A Dynamic and Scalable Benchmark for Evaluating Perception Ability of LVLMs**

Dysca：评估LVLM感知能力的动态和可扩展基准 cs.CV

**SubmitDate**: 2024-06-27    [abs](http://arxiv.org/abs/2406.18849v1) [paper-pdf](http://arxiv.org/pdf/2406.18849v1)

**Authors**: Jie Zhang, Zhongqi Wang, Mengqi Lei, Zheng Yuan, Bei Yan, Shiguang Shan, Xilin Chen

**Abstract**: Currently many benchmarks have been proposed to evaluate the perception ability of the Large Vision-Language Models (LVLMs). However, most benchmarks conduct questions by selecting images from existing datasets, resulting in the potential data leakage. Besides, these benchmarks merely focus on evaluating LVLMs on the realistic style images and clean scenarios, leaving the multi-stylized images and noisy scenarios unexplored. In response to these challenges, we propose a dynamic and scalable benchmark named Dysca for evaluating LVLMs by leveraging synthesis images. Specifically, we leverage Stable Diffusion and design a rule-based method to dynamically generate novel images, questions and the corresponding answers. We consider 51 kinds of image styles and evaluate the perception capability in 20 subtasks. Moreover, we conduct evaluations under 4 scenarios (i.e., Clean, Corruption, Print Attacking and Adversarial Attacking) and 3 question types (i.e., Multi-choices, True-or-false and Free-form). Thanks to the generative paradigm, Dysca serves as a scalable benchmark for easily adding new subtasks and scenarios. A total of 8 advanced open-source LVLMs with 10 checkpoints are evaluated on Dysca, revealing the drawbacks of current LVLMs. The benchmark is released in \url{https://github.com/Benchmark-Dysca/Dysca}.

摘要: 目前，人们已经提出了许多基准来评估大型视觉语言模型的感知能力。然而，大多数基准测试通过从现有数据集中选择图像来进行问题，从而导致潜在的数据泄漏。此外，这些基准只关注真实感风格的图像和干净的场景来评估LVLMS，而对多风格化的图像和噪声场景没有进行探索。为了应对这些挑战，我们提出了一种动态的、可扩展的基准测试DYSCA，用于利用合成图像来评估LVLMS。具体地说，我们利用稳定扩散并设计了一种基于规则的方法来动态生成新的图像、问题和相应的答案。我们考虑了51种图像风格，并在20个子任务中评估了感知能力。此外，我们还在4个场景(即廉洁、腐败、打印攻击和对抗性攻击)和3个问题类型(即多项选择、对错和自由形式)下进行了评估。多亏了生成性范式，Dysca成为了一个可伸缩的基准，可以轻松添加新的子任务和场景。在Dysca上对8个具有10个检查点的高级开源LVLMS进行了评估，揭示了当前LVLMS的缺陷。该基准测试在\url{https://github.com/Benchmark-Dysca/Dysca}.中发布



## **41. Jailbreaking LLMs with Arabic Transliteration and Arabizi**

使用阿拉伯语拼音和Arabizi语越狱LLM cs.LG

14 pages, 4 figures

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18725v1) [paper-pdf](http://arxiv.org/pdf/2406.18725v1)

**Authors**: Mansour Al Ghanim, Saleh Almohaimeed, Mengxin Zheng, Yan Solihin, Qian Lou

**Abstract**: This study identifies the potential vulnerabilities of Large Language Models (LLMs) to 'jailbreak' attacks, specifically focusing on the Arabic language and its various forms. While most research has concentrated on English-based prompt manipulation, our investigation broadens the scope to investigate the Arabic language. We initially tested the AdvBench benchmark in Standardized Arabic, finding that even with prompt manipulation techniques like prefix injection, it was insufficient to provoke LLMs into generating unsafe content. However, when using Arabic transliteration and chatspeak (or arabizi), we found that unsafe content could be produced on platforms like OpenAI GPT-4 and Anthropic Claude 3 Sonnet. Our findings suggest that using Arabic and its various forms could expose information that might remain hidden, potentially increasing the risk of jailbreak attacks. We hypothesize that this exposure could be due to the model's learned connection to specific words, highlighting the need for more comprehensive safety training across all language forms.

摘要: 这项研究确定了大型语言模型(LLM)对‘越狱’攻击的潜在漏洞，特别是关注阿拉伯语及其各种形式。虽然大多数研究都集中在基于英语的即时操作上，但我们的调查扩大了对阿拉伯语的研究范围。我们最初用标准化的阿拉伯语测试了AdvBtch基准测试，发现即使使用前缀注入等快速操作技术，也不足以激发LLMS生成不安全的内容。然而，当使用阿拉伯语音译和聊天(或Arabizi)时，我们发现在OpenAI GPT-4和人类克劳德3十四行诗等平台上可能会产生不安全的内容。我们的发现表明，使用阿拉伯语及其各种形式可能会暴露可能仍然隐藏的信息，潜在地增加越狱攻击的风险。我们假设，这种接触可能是由于模型与特定单词的习得联系，强调了在所有语言形式中进行更全面的安全培训的必要性。



## **42. MultiAgent Collaboration Attack: Investigating Adversarial Attacks in Large Language Model Collaborations via Debate**

多Agent协作攻击：通过辩论调查大型语言模型协作中的对抗性攻击 cs.CL

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.14711v2) [paper-pdf](http://arxiv.org/pdf/2406.14711v2)

**Authors**: Alfonso Amayuelas, Xianjun Yang, Antonis Antoniades, Wenyue Hua, Liangming Pan, William Wang

**Abstract**: Large Language Models (LLMs) have shown exceptional results on current benchmarks when working individually. The advancement in their capabilities, along with a reduction in parameter size and inference times, has facilitated the use of these models as agents, enabling interactions among multiple models to execute complex tasks. Such collaborations offer several advantages, including the use of specialized models (e.g. coding), improved confidence through multiple computations, and enhanced divergent thinking, leading to more diverse outputs. Thus, the collaborative use of language models is expected to grow significantly in the coming years. In this work, we evaluate the behavior of a network of models collaborating through debate under the influence of an adversary. We introduce pertinent metrics to assess the adversary's effectiveness, focusing on system accuracy and model agreement. Our findings highlight the importance of a model's persuasive ability in influencing others. Additionally, we explore inference-time methods to generate more compelling arguments and evaluate the potential of prompt-based mitigation as a defensive strategy.

摘要: 大型语言模型(LLM)在单独工作时，在当前基准上显示了特殊的结果。它们能力的进步，加上参数大小和推理时间的减少，促进了这些模型作为代理的使用，使多个模型之间能够相互作用，以执行复杂的任务。这种协作提供了几个优势，包括使用专门的模型(例如编码)、通过多次计算提高信心以及增强发散思维，从而产生更多样化的产出。因此，语言模型的协作使用预计在未来几年将显著增长。在这项工作中，我们评估了一个模型网络在对手的影响下通过辩论进行合作的行为。我们引入了相关的度量来评估对手的有效性，重点是系统的准确性和模型的一致性。我们的发现突显了模特的说服力在影响他人方面的重要性。此外，我们探索推理时间方法来生成更令人信服的论点，并评估基于即时缓解作为一种防御策略的潜力。



## **43. Towards Understanding Jailbreak Attacks in LLMs: A Representation Space Analysis**

了解LLC中的越狱攻击：表示空间分析 cs.CL

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.10794v2) [paper-pdf](http://arxiv.org/pdf/2406.10794v2)

**Authors**: Yuping Lin, Pengfei He, Han Xu, Yue Xing, Makoto Yamada, Hui Liu, Jiliang Tang

**Abstract**: Large language models (LLMs) are susceptible to a type of attack known as jailbreaking, which misleads LLMs to output harmful contents. Although there are diverse jailbreak attack strategies, there is no unified understanding on why some methods succeed and others fail. This paper explores the behavior of harmful and harmless prompts in the LLM's representation space to investigate the intrinsic properties of successful jailbreak attacks. We hypothesize that successful attacks share some similar properties: They are effective in moving the representation of the harmful prompt towards the direction to the harmless prompts. We leverage hidden representations into the objective of existing jailbreak attacks to move the attacks along the acceptance direction, and conduct experiments to validate the above hypothesis using the proposed objective. We hope this study provides new insights into understanding how LLMs understand harmfulness information.

摘要: 大型语言模型（LLM）容易受到一种称为越狱的攻击，这种攻击会误导LLM输出有害内容。尽管越狱攻击策略多种多样，但对于为什么有些方法成功而另一些方法失败，人们并没有统一的理解。本文探讨了LLM表示空间中有害和无害提示的行为，以研究成功越狱攻击的内在属性。我们假设成功的攻击具有一些相似的属性：它们有效地将有害提示的表示移向无害提示的方向。我们将隐藏的表示利用到现有越狱攻击的目标中，以沿着接受方向移动攻击，并使用提出的目标进行实验来验证上述假设。我们希望这项研究为理解LLM如何理解有害信息提供新的见解。



## **44. Are AI-Generated Text Detectors Robust to Adversarial Perturbations?**

人工智能生成的文本检测器对对抗性扰动是否稳健？ cs.CL

Accepted to ACL 2024 main conference

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.01179v2) [paper-pdf](http://arxiv.org/pdf/2406.01179v2)

**Authors**: Guanhua Huang, Yuchen Zhang, Zhe Li, Yongjian You, Mingze Wang, Zhouwang Yang

**Abstract**: The widespread use of large language models (LLMs) has sparked concerns about the potential misuse of AI-generated text, as these models can produce content that closely resembles human-generated text. Current detectors for AI-generated text (AIGT) lack robustness against adversarial perturbations, with even minor changes in characters or words causing a reversal in distinguishing between human-created and AI-generated text. This paper investigates the robustness of existing AIGT detection methods and introduces a novel detector, the Siamese Calibrated Reconstruction Network (SCRN). The SCRN employs a reconstruction network to add and remove noise from text, extracting a semantic representation that is robust to local perturbations. We also propose a siamese calibration technique to train the model to make equally confidence predictions under different noise, which improves the model's robustness against adversarial perturbations. Experiments on four publicly available datasets show that the SCRN outperforms all baseline methods, achieving 6.5\%-18.25\% absolute accuracy improvement over the best baseline method under adversarial attacks. Moreover, it exhibits superior generalizability in cross-domain, cross-genre, and mixed-source scenarios. The code is available at \url{https://github.com/CarlanLark/Robust-AIGC-Detector}.

摘要: 大型语言模型(LLM)的广泛使用引发了人们对人工智能生成的文本可能被滥用的担忧，因为这些模型可以生成与人类生成的文本非常相似的内容。目前的人工智能生成文本检测器(AIGT)缺乏对对手扰动的稳健性，即使是字符或单词的微小变化也会导致在区分人工生成文本和人工智能生成文本方面出现逆转。本文研究了现有的AIGT检测方法的稳健性，并介绍了一种新的检测器--暹罗校准重建网络(SCRN)。SCRN使用重构网络来添加和去除文本中的噪声，提取对局部扰动具有鲁棒性的语义表示。我们还提出了一种暹罗校正技术来训练模型，使其在不同的噪声下做出相同的置信度预测，从而提高了模型对对抗性扰动的鲁棒性。在四个公开可用的数据集上的实验表明，SCRN的性能优于所有的基线方法，在对抗性攻击下，其绝对准确率比最佳基线方法提高了6.5-18.25。此外，它在跨域、跨流派和混合来源的场景中表现出出色的泛化能力。代码可在\url{https://github.com/CarlanLark/Robust-AIGC-Detector}.上获得



## **45. Enhancing Data Privacy in Large Language Models through Private Association Editing**

通过私人关联编辑增强大型语言模型中的数据隐私 cs.CL

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18221v1) [paper-pdf](http://arxiv.org/pdf/2406.18221v1)

**Authors**: Davide Venditti, Elena Sofia Ruzzetti, Giancarlo A. Xompero, Cristina Giannone, Andrea Favalli, Raniero Romagnoli, Fabio Massimo Zanzotto

**Abstract**: Large Language Models (LLMs) are powerful tools with extensive applications, but their tendency to memorize private information raises significant concerns as private data leakage can easily happen. In this paper, we introduce Private Association Editing (PAE), a novel defense approach for private data leakage. PAE is designed to effectively remove Personally Identifiable Information (PII) without retraining the model. Our approach consists of a four-step procedure: detecting memorized PII, applying PAE cards to mitigate memorization of private data, verifying resilience to targeted data extraction (TDE) attacks, and ensuring consistency in the post-edit LLMs. The versatility and efficiency of PAE, which allows for batch modifications, significantly enhance data privacy in LLMs. Experimental results demonstrate the effectiveness of PAE in mitigating private data leakage. We believe PAE will serve as a critical tool in the ongoing effort to protect data privacy in LLMs, encouraging the development of safer models for real-world applications.

摘要: 大型语言模型(LLM)是应用广泛的强大工具，但它们存储私人信息的倾向引起了人们的极大担忧，因为私人数据很容易泄露。本文介绍了一种新的隐私数据泄露防御方法--私有关联编辑(PAE)。PAE旨在有效地删除个人身份信息(PII)，而无需对模型进行重新培训。我们的方法包括四个步骤：检测记忆的PII，应用PAE卡来减少私人数据的记忆，验证对目标数据提取(TDE)攻击的恢复能力，以及确保编辑后LLM的一致性。PAE的多功能性和效率允许批量修改，显著增强了LLMS中的数据隐私。实验结果证明了PAE在缓解私有数据泄露方面的有效性。我们相信，PAE将作为正在进行的保护LLMS数据隐私的努力中的关键工具，鼓励为现实世界应用程序开发更安全的模型。



## **46. Poisoned LangChain: Jailbreak LLMs by LangChain**

中毒的LangChain：LangChain的越狱LLMS cs.CL

6 pages,2 figures,This paper is a submission to ACM TURC. It has been  accepted by the editor of the organizer

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18122v1) [paper-pdf](http://arxiv.org/pdf/2406.18122v1)

**Authors**: Ziqiu Wang, Jun Liu, Shengkai Zhang, Yang Yang

**Abstract**: With the development of natural language processing (NLP), large language models (LLMs) are becoming increasingly popular. LLMs are integrating more into everyday life, raising public concerns about their security vulnerabilities. Consequently, the security of large language models is becoming critically important. Currently, the techniques for attacking and defending against LLMs are continuously evolving. One significant method type of attack is the jailbreak attack, which designed to evade model safety mechanisms and induce the generation of inappropriate content. Existing jailbreak attacks primarily rely on crafting inducement prompts for direct jailbreaks, which are less effective against large models with robust filtering and high comprehension abilities. Given the increasing demand for real-time capabilities in large language models, real-time updates and iterations of new knowledge have become essential. Retrieval-Augmented Generation (RAG), an advanced technique to compensate for the model's lack of new knowledge, is gradually becoming mainstream. As RAG enables the model to utilize external knowledge bases, it provides a new avenue for jailbreak attacks.   In this paper, we conduct the first work to propose the concept of indirect jailbreak and achieve Retrieval-Augmented Generation via LangChain. Building on this, we further design a novel method of indirect jailbreak attack, termed Poisoned-LangChain (PLC), which leverages a poisoned external knowledge base to interact with large language models, thereby causing the large models to generate malicious non-compliant dialogues.We tested this method on six different large language models across three major categories of jailbreak issues. The experiments demonstrate that PLC successfully implemented indirect jailbreak attacks under three different scenarios, achieving success rates of 88.56%, 79.04%, and 82.69% respectively.

摘要: 随着自然语言处理(NLP)的发展，大语言模型(LLM)变得越来越流行。LLMS正在更多地融入日常生活，这引发了公众对其安全漏洞的担忧。因此，大型语言模型的安全性变得至关重要。目前，攻击和防御LLMS的技术正在不断发展。越狱攻击是一种重要的攻击方法类型，旨在逃避模型安全机制并诱导生成不适当的内容。现有的越狱攻击主要依赖于精心制作直接越狱的诱导提示，这对具有强大过滤和高理解能力的大型模型效果较差。鉴于大型语言模型对实时能力的需求日益增加，实时更新和迭代新知识变得至关重要。检索-增强生成(RAG)是一种弥补模型缺乏新知识的先进技术，正逐渐成为主流。由于RAG使模型能够利用外部知识库，它为越狱攻击提供了一种新的途径。在本文中，我们首次提出了间接越狱的概念，并通过LangChain实现了检索增强生成。在此基础上，我们进一步设计了一种新的间接越狱攻击方法，称为毒化朗链(PLC)，它利用有毒的外部知识库与大型语言模型交互，从而导致大型模型生成恶意的不符合规则的对话框，并在三大类越狱问题的六个不同的大型语言模型上测试了该方法。实验表明，PLC在三种不同的场景下成功实现了间接越狱攻击，成功率分别为88.56%、79.04%和82.69%。



## **47. Safely Learning with Private Data: A Federated Learning Framework for Large Language Model**

利用私人数据安全学习：大型语言模型的联邦学习框架 cs.CR

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.14898v2) [paper-pdf](http://arxiv.org/pdf/2406.14898v2)

**Authors**: JiaYing Zheng, HaiNan Zhang, LingXiang Wang, WangJie Qiu, HongWei Zheng, ZhiMing Zheng

**Abstract**: Private data, being larger and quality-higher than public data, can greatly improve large language models (LLM). However, due to privacy concerns, this data is often dispersed in multiple silos, making its secure utilization for LLM training a challenge. Federated learning (FL) is an ideal solution for training models with distributed private data, but traditional frameworks like FedAvg are unsuitable for LLM due to their high computational demands on clients. An alternative, split learning, offloads most training parameters to the server while training embedding and output layers locally, making it more suitable for LLM. Nonetheless, it faces significant challenges in security and efficiency. Firstly, the gradients of embeddings are prone to attacks, leading to potential reverse engineering of private data. Furthermore, the server's limitation of handle only one client's training request at a time hinders parallel training, severely impacting training efficiency. In this paper, we propose a Federated Learning framework for LLM, named FL-GLM, which prevents data leakage caused by both server-side and peer-client attacks while improving training efficiency. Specifically, we first place the input block and output block on local client to prevent embedding gradient attacks from server. Secondly, we employ key-encryption during client-server communication to prevent reverse engineering attacks from peer-clients. Lastly, we employ optimization methods like client-batching or server-hierarchical, adopting different acceleration methods based on the actual computational capabilities of the server. Experimental results on NLU and generation tasks demonstrate that FL-GLM achieves comparable metrics to centralized chatGLM model, validating the effectiveness of our federated learning framework.

摘要: 私有数据比公共数据更大、质量更高，可以极大地改进大型语言模型(LLM)。然而，出于隐私方面的考虑，这些数据通常分散在多个竖井中，这使得将其安全地用于LLM培训成为一项挑战。联邦学习(FL)是一种适用于具有分布式私有数据的模型训练的理想解决方案，但FedAvg等传统框架由于对客户端的计算要求较高而不适用于LLM。另一种选择是分离学习，将大部分训练参数卸载到服务器，同时在本地训练嵌入和输出层，使其更适合LLM。尽管如此，它在安全和效率方面仍面临重大挑战。首先，嵌入的梯度容易受到攻击，从而导致对私有数据的潜在逆向工程。此外，服务器一次只能处理一个客户端的训练请求的限制阻碍了并行训练，严重影响了训练效率。本文提出了一种用于LLM的联邦学习框架FL-GLM，该框架在提高训练效率的同时，防止了服务器端攻击和对等客户端攻击引起的数据泄漏。具体地说，我们首先将输入块和输出块放置在本地客户端，以防止来自服务器的嵌入梯度攻击。其次，我们在客户-服务器通信过程中使用密钥加密，以防止来自对等客户端的反向工程攻击。最后，我们采用了客户端批处理或服务器分层等优化方法，根据服务器的实际计算能力采用不同的加速方法。在NLU和生成任务上的实验结果表明，FL-GLM达到了与集中式ChatGLM模型相当的指标，验证了联邦学习框架的有效性。



## **48. InstructTA: Instruction-Tuned Targeted Attack for Large Vision-Language Models**

DirectTA：针对大型视觉语言模型的指令调整有针对性的攻击 cs.CV

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2312.01886v3) [paper-pdf](http://arxiv.org/pdf/2312.01886v3)

**Authors**: Xunguang Wang, Zhenlan Ji, Pingchuan Ma, Zongjie Li, Shuai Wang

**Abstract**: Large vision-language models (LVLMs) have demonstrated their incredible capability in image understanding and response generation. However, this rich visual interaction also makes LVLMs vulnerable to adversarial examples. In this paper, we formulate a novel and practical targeted attack scenario that the adversary can only know the vision encoder of the victim LVLM, without the knowledge of its prompts (which are often proprietary for service providers and not publicly available) and its underlying large language model (LLM). This practical setting poses challenges to the cross-prompt and cross-model transferability of targeted adversarial attack, which aims to confuse the LVLM to output a response that is semantically similar to the attacker's chosen target text. To this end, we propose an instruction-tuned targeted attack (dubbed \textsc{InstructTA}) to deliver the targeted adversarial attack on LVLMs with high transferability. Initially, we utilize a public text-to-image generative model to "reverse" the target response into a target image, and employ GPT-4 to infer a reasonable instruction $\boldsymbol{p}^\prime$ from the target response. We then form a local surrogate model (sharing the same vision encoder with the victim LVLM) to extract instruction-aware features of an adversarial image example and the target image, and minimize the distance between these two features to optimize the adversarial example. To further improve the transferability with instruction tuning, we augment the instruction $\boldsymbol{p}^\prime$ with instructions paraphrased from GPT-4. Extensive experiments demonstrate the superiority of our proposed method in targeted attack performance and transferability. The code is available at https://github.com/xunguangwang/InstructTA.

摘要: 大型视觉语言模型在图像理解和响应生成方面表现出了令人难以置信的能力。然而，这种丰富的视觉交互也使LVLM容易受到对抗性例子的攻击。在该文中，我们提出了一种新颖而实用的定向攻击场景，攻击者只能知道受害者LVLM的视觉编码器，而不知道其提示(通常是服务提供商的专有提示，而不是公开可用的)及其底层的大语言模型(LLM)。这一实际设置对目标对抗性攻击的跨提示和跨模型可转移性提出了挑战，其目的是混淆LVLM以输出与攻击者选择的目标文本在语义上相似的响应。为此，我们提出了一种指令调谐的定向攻击(称为\Textsc{InstructTA})，以提供对具有高可转移性的LVLM的定向对抗性攻击。首先，我们利用一个公开的文本到图像的生成模型将目标响应“反转”成目标图像，并使用GPT-4从目标响应中推断出合理的指令符号。然后，我们形成一个局部代理模型(与受害者LVLM共享相同的视觉编码器)来提取对抗性图像示例和目标图像的指令感知特征，并最小化这两个特征之间的距离以优化对抗性示例。为了进一步提高指令调优的可转移性，我们用GPT-4中转译的指令扩充了指令$\boldsign{p}^\Prime$。大量实验证明了该方法在目标攻击性能和可转移性方面的优越性。代码可在https://github.com/xunguangwang/InstructTA.上获得



## **49. Inherent Challenges of Post-Hoc Membership Inference for Large Language Models**

大型语言模型事后成员推理的内在挑战 cs.CL

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2406.17975v1) [paper-pdf](http://arxiv.org/pdf/2406.17975v1)

**Authors**: Matthieu Meeus, Shubham Jain, Marek Rei, Yves-Alexandre de Montjoye

**Abstract**: Large Language Models (LLMs) are often trained on vast amounts of undisclosed data, motivating the development of post-hoc Membership Inference Attacks (MIAs) to gain insight into their training data composition. However, in this paper, we identify inherent challenges in post-hoc MIA evaluation due to potential distribution shifts between collected member and non-member datasets. Using a simple bag-of-words classifier, we demonstrate that datasets used in recent post-hoc MIAs suffer from significant distribution shifts, in some cases achieving near-perfect distinction between members and non-members. This implies that previously reported high MIA performance may be largely attributable to these shifts rather than model memorization. We confirm that randomized, controlled setups eliminate such shifts and thus enable the development and fair evaluation of new MIAs. However, we note that such randomized setups are rarely available for the latest LLMs, making post-hoc data collection still required to infer membership for real-world LLMs. As a potential solution, we propose a Regression Discontinuity Design (RDD) approach for post-hoc data collection, which substantially mitigates distribution shifts. Evaluating various MIA methods on this RDD setup yields performance barely above random guessing, in stark contrast to previously reported results. Overall, our findings highlight the challenges in accurately measuring LLM memorization and the need for careful experimental design in (post-hoc) membership inference tasks.

摘要: 大型语言模型(LLM)通常基于大量未公开的数据进行训练，这促使了后自组织成员推理攻击(MIA)的发展，以深入了解它们的训练数据组成。然而，在这篇文章中，我们识别了由于收集的成员和非成员数据集之间潜在的分布变化而导致的后MIA评估的内在挑战。使用一个简单的词袋分类器，我们证明了在最近的后自组织MIA中使用的数据集遭受了显著的分布偏移，在某些情况下实现了成员和非成员之间的近乎完美的区分。这意味着之前报道的高MIA成绩可能在很大程度上归因于这些变化，而不是模型记忆。我们确认，随机的、受控的设置消除了这种转变，从而使新的MIA的开发和公平评估成为可能。然而，我们注意到，这种随机化设置很少适用于最新的LLM，这使得仍然需要事后数据收集来推断真实世界LLM的成员资格。作为一种潜在的解决方案，我们提出了一种回归不连续设计(RDD)方法，用于后自组织数据收集，大大减轻了分布漂移。在这种RDD设置上评估各种MIA方法的性能仅略高于随机猜测，这与之前报道的结果形成了鲜明对比。总体而言，我们的发现突出了在准确测量LLM记忆方面的挑战，以及在(后即席)成员推理任务中仔细实验设计的必要性。



## **50. CoSafe: Evaluating Large Language Model Safety in Multi-Turn Dialogue Coreference**

CoSafe：评估多轮对话共指涉中的大型语言模型安全性 cs.CL

Submitted to EMNLP 2024

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2406.17626v1) [paper-pdf](http://arxiv.org/pdf/2406.17626v1)

**Authors**: Erxin Yu, Jing Li, Ming Liao, Siqi Wang, Zuchen Gao, Fei Mi, Lanqing Hong

**Abstract**: As large language models (LLMs) constantly evolve, ensuring their safety remains a critical research problem. Previous red-teaming approaches for LLM safety have primarily focused on single prompt attacks or goal hijacking. To the best of our knowledge, we are the first to study LLM safety in multi-turn dialogue coreference. We created a dataset of 1,400 questions across 14 categories, each featuring multi-turn coreference safety attacks. We then conducted detailed evaluations on five widely used open-source LLMs. The results indicated that under multi-turn coreference safety attacks, the highest attack success rate was 56% with the LLaMA2-Chat-7b model, while the lowest was 13.9% with the Mistral-7B-Instruct model. These findings highlight the safety vulnerabilities in LLMs during dialogue coreference interactions.

摘要: 随着大型语言模型（LLM）的不断发展，确保其安全性仍然是一个关键的研究问题。之前的LLM安全红色团队方法主要集中在单次提示攻击或目标劫持上。据我们所知，我们是第一个在多回合对话共指涉中研究LLM安全性的公司。我们创建了一个包含14个类别的1，400个问题的数据集，每个问题都具有多轮共指安全攻击。然后，我们对五个广泛使用的开源LLM进行了详细的评估。结果表明，在多轮共指安全攻击下，LLaMA 2-Chat-7 b模型的攻击成功率最高为56%，而Mistral-7 B-Direct模型的攻击成功率最低为13.9%。这些发现凸显了LLM在对话共指互动期间的安全漏洞。



